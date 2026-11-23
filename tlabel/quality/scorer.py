"""
QualityScorer — 数据质量评分引擎

对标国标《具身智能数据质量规范》TC28/SC42

4个评分维度：
1. physical_consistency (30%): 物理一致性 — 联动规则是否满足
2. temporal_smoothness (25%): 时序平滑度 — 相邻帧是否突变
3. completeness (25%): 完整性 — 字段缺失/全零比例
4. coverage (20%): 标注覆盖率 — 有意义的标注占比

综合评分 = 加权平均，映射到 0-100
等级: A(≥90) / B(≥75) / C(≥60) / D(≥40) / F(<40)

v0.17 Breaking Change: 移除旧 22 维兼容逻辑，只使用 Schema V2 (14维)。
"""

import math
from typing import Dict, List, Optional

from tlabel.core.types import TLabelData, TLabelFrame, _sv2_scalar


# 14 维 Schema V2 特征完整列表
SCHEMA_V2_DIMS = [
    "contact", "contact_centroid", "contact_region", "force_magnitude",
    "force_vector", "torque_vector", "slip_event", "slip_velocity",
    "manipulation_phase", "texture_class", "object_deformation",
    "temperature", "confidence", "compliance_level",
]

# Schema V2 标量数值字段（用于时序平滑度检测）
SCALAR_DIMS = [
    "contact", "force_magnitude", "slip_event",
    "object_deformation", "temperature", "confidence",
]


class QualityScorer:
    """数据质量评分器 — 只使用 Schema V2"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def score(self, data: TLabelData) -> Dict:
        """计算数据质量评分"""
        if not data.frames:
            return {
                "overall": 0.0,
                "physical_consistency": 0.0,
                "temporal_smoothness": 0.0,
                "completeness": 0.0,
                "coverage": 0.0,
                "warnings": ["数据为空"],
                "grade": "F",
            }

        warnings = []

        phys_score, phys_warnings = self._physical_consistency(data)
        warnings.extend(phys_warnings)

        temporal_score, temporal_warnings = self._temporal_smoothness(data)
        warnings.extend(temporal_warnings)

        comp_score, comp_warnings = self._completeness(data)
        warnings.extend(comp_warnings)

        cov_score, cov_warnings = self._coverage(data)
        warnings.extend(cov_warnings)

        overall = (
            phys_score * 0.30 +
            temporal_score * 0.25 +
            comp_score * 0.25 +
            cov_score * 0.20
        )
        overall = round(overall, 1)
        grade = self._grade(overall)

        return {
            "overall": overall,
            "physical_consistency": round(phys_score, 1),
            "temporal_smoothness": round(temporal_score, 1),
            "completeness": round(comp_score, 1),
            "coverage": round(cov_score, 1),
            "warnings": warnings,
            "grade": grade,
        }

    def _physical_consistency(self, data: TLabelData) -> tuple:
        """
        物理一致性检查（Schema V2 字段名）：
        - contact=0 时，force_magnitude应为0
        - contact=0 时，slip_event应为0
        - slip_event>0 时，contact应>0
        - force_magnitude>0 时，contact应>0
        - object_deformation>0 时，contact应>0
        """
        violations = 0
        total_checks = 0
        warnings = []

        for frame in data.frames:
            sv2 = frame.schema_v2
            contact = 1.0 if sv2.contact else 0.0
            # P9：这里把缺失填成 0.0 是**安全的**，因为下面所有规则都是
            # "值 > 阈值才算违规"。缺失（无数据）不构成违规证据，填 0.0
            # 等价于"不参与判定"，不会误报也不会漏报。
            force = sv2.force_magnitude if sv2.force_magnitude is not None else 0.0
            slip = 1.0 if sv2.slip_event else 0.0
            deformation = sv2.object_deformation if sv2.object_deformation is not None else 0.0

            # Rule 1: contact=0 → force should be ~0
            if contact < 0.1:
                total_checks += 1
                if force > 0.15:
                    violations += 1
                    if self.verbose:
                        warnings.append(f"Frame {frame.frame_idx}: contact={contact:.2f} but force={force:.2f}")

            # Rule 2: contact=0 → slip should be 0
            if contact < 0.1:
                total_checks += 1
                if slip > 0.5:
                    violations += 1
                    if self.verbose:
                        warnings.append(f"Frame {frame.frame_idx}: contact={contact:.2f} but slip_event={slip:.2f}")

            # Rule 3: slip>0 → contact should be >0
            if slip > 0.5:
                total_checks += 1
                if contact < 0.1:
                    violations += 1
                    if self.verbose:
                        warnings.append(f"Frame {frame.frame_idx}: slip_event={slip:.2f} but contact={contact:.2f}")

            # Rule 4: force>0 → contact should be >0
            if force > 0.15:
                total_checks += 1
                if contact < 0.1:
                    violations += 1

            # Rule 5: object_deformation>0 → contact should be >0
            if deformation > 0.1:
                total_checks += 1
                if contact < 0.1:
                    violations += 1

        if total_checks == 0:
            return 100.0, []

        consistency = (1 - violations / total_checks) * 100

        if violations > 0:
            warnings.insert(0, f"物理一致性违规: {violations}/{total_checks} 次检查未通过")

        return consistency, warnings

    def _temporal_smoothness(self, data: TLabelData) -> tuple:
        """时序平滑度：检查相邻帧之间的突变（Schema V2 标量字段）"""
        warnings = []
        if len(data.frames) < 2:
            return 100.0, []

        total_jumps = 0
        total_pairs = 0

        for key in SCALAR_DIMS:
            # P9：缺失值不参与差分。此前缺失被填 0.0，会让"有值→无值"的边界
            # 产生一个虚假的巨幅跳变，被算成时序突变。
            values = [_sv2_scalar(f, key, missing=None) for f in data.frames]
            diffs = [
                abs(values[i + 1] - values[i])
                for i in range(len(values) - 1)
                if values[i] is not None and values[i + 1] is not None
            ]

            if not diffs:
                continue

            mean_diff = sum(diffs) / len(diffs)
            var_diff = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
            std_diff = math.sqrt(var_diff) if var_diff > 0 else 0

            threshold = mean_diff + 3 * std_diff if std_diff > 0 else mean_diff * 5

            for i, d in enumerate(diffs):
                total_pairs += 1
                if threshold > 0 and d > threshold:
                    total_jumps += 1

        if total_pairs == 0:
            return 100.0, []

        smoothness = (1 - total_jumps / total_pairs) * 100

        if total_jumps > total_pairs * 0.05:
            warnings.append(f"时序突变较多: {total_jumps}/{total_pairs} 帧对存在突变")

        return smoothness, warnings

    def _completeness(self, data: TLabelData) -> tuple:
        """完整性：检查字段缺失和全零比例（Schema V2 14维）"""
        warnings = []

        keys = data.dimension_keys
        expected_dims = len(SCHEMA_V2_DIMS)  # 14
        dim_ratio = min(len(keys) / expected_dims, 1.0)
        if len(keys) < expected_dims:
            warnings.append(f"维度不完整: {len(keys)}/{expected_dims} (Schema V2)")

        # Check all-zero frames
        all_zero_count = 0
        for frame in data.frames:
            sv2 = frame.schema_v2
            contact_related = [
                sv2.force_magnitude if sv2.force_magnitude is not None else 0.0,
                1.0 if sv2.slip_event else 0.0,
                sv2.object_deformation if sv2.object_deformation is not None else 0.0,
            ]
            if not sv2.contact:
                if all(v == 0 for v in contact_related):
                    all_zero_count += 1

        # Check if contact frames have meaningful data
        contact_frames = [f for f in data.frames if f.contact > 0.5]
        empty_contact = 0
        for f in contact_frames:
            sv2 = f.schema_v2
            has_force = sv2.force_magnitude is not None and sv2.force_magnitude > 0
            has_deform = sv2.object_deformation is not None and sv2.object_deformation > 0
            if not (has_force or has_deform):
                empty_contact += 1

        if contact_frames and empty_contact > 0:
            empty_ratio = empty_contact / len(contact_frames)
            warnings.append(f"接触帧中{empty_ratio:.0%}缺少力度/形变数据")
            contact_quality = (1 - empty_ratio) * 100
        else:
            contact_quality = 100.0

        completeness = dim_ratio * 0.3 + contact_quality * 0.7

        return completeness, warnings

    def _coverage(self, data: TLabelData) -> tuple:
        """覆盖率：有意义的标注占比"""
        warnings = []

        total = len(data.frames)
        if total == 0:
            return 0.0, ["无帧数据"]

        contact_count = sum(1 for f in data.frames if f.contact > 0.5)
        contact_ratio = contact_count / total

        high_conf = sum(1 for f in data.frames if f.confidence > 0.5)
        conf_ratio = high_conf / total

        phases = set(f.manipulation_phase for f in data.frames)
        phase_diversity = min(len(phases) / 5, 1.0)

        slip_count = sum(1 for f in data.frames if f.slip_event > 0.5)
        has_slip = 1.0 if slip_count > 0 else (0.5 if total < 50 else 0.0)

        if contact_ratio < 0.05:
            warnings.append("接触帧占比极低 ({:.1%})，数据可能无实际交互".format(contact_ratio))
        if conf_ratio < 0.8:
            warnings.append("高置信度帧占比低 ({:.1%})".format(conf_ratio))
        if not has_slip and total >= 50:
            warnings.append("无滑移事件标注，建议补充slip_event")

        coverage = (
            min(contact_ratio * 5, 1.0) * 0.3 +
            conf_ratio * 0.3 +
            phase_diversity * 0.2 +
            has_slip * 0.2
        ) * 100

        return coverage, warnings

    @staticmethod
    def _grade(score: float) -> str:
        """将分数映射为等级"""
        if score >= 90:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"
