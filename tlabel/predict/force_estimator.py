"""
Force Estimator — v0.14.0 新增, v0.17 Breaking Change Schema V2 Only

从视触觉传感器数据（形变、图像）推断力的大小。
让只有 GelSight / DIGIT / PaXini 等视触觉传感器（无F/T传感器）的用户也能进行 primitive 标注。

核心物理模型:
  - F = k × δ  (弹性体刚度 × 形变量)
  - 形变来源: object_deformation 字段 (Schema V2) / 图像帧差异

v0.17 Breaking Change:
  - 移除 _compat 兼容层，直接访问 frame.schema_v2
  - deformation_magnitude → object_deformation
  - has_real_force_data() 只检查 schema_v2
  - integrate() 写入 schema_v2 而非 tlabel_v2

不引入外部ML依赖（无PyTorch/TensorFlow），纯物理模型+简单图像处理。
"""

import math
from typing import List, Dict, Optional, Any

from tlabel.core.types import TLabelData, TLabelFrame, _sv2_scalar


# ============================================================
# 传感器默认弹性体刚度 (N/m)
# ============================================================

DEFAULT_ELASTOMER_STIFFNESS = {
    "gelsight_mini": 800.0,
    "gelsight": 600.0,
    "digit": 1200.0,
    "paxini": 500.0,
    "daimon": 700.0,
    "touchd": 450.0,
    "contactile": 350.0,
}


def get_default_stiffness(sensor_type: str) -> float:
    """获取传感器默认刚度值"""
    sensor_type_lower = sensor_type.lower().replace(" ", "_")
    for key, val in DEFAULT_ELASTOMER_STIFFNESS.items():
        if key in sensor_type_lower:
            return val
    return 500.0


# ============================================================
# 推断结果容器
# ============================================================

class ForceEstimate:
    """单帧力推断结果"""
    __slots__ = ['frame_idx', 'estimated_force_n', 'confidence',
                 'method', 'deformation_used', 'stiffness_used']

    def __init__(self, frame_idx: int, estimated_force_n: float,
                 confidence: float, method: str,
                 deformation_used: float, stiffness_used: float):
        self.frame_idx = frame_idx
        self.estimated_force_n = estimated_force_n
        self.confidence = confidence
        self.method = method
        self.deformation_used = deformation_used
        self.stiffness_used = stiffness_used

    def to_dict(self) -> Dict:
        return {
            "frame_idx": self.frame_idx,
            "estimated_force_n": round(self.estimated_force_n, 4),
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "source": "estimated_force",
            "deformation_used": round(self.deformation_used, 4),
            "stiffness_used": round(self.stiffness_used, 2),
        }


# ============================================================
# 基类
# ============================================================

class ForceEstimator:
    """力推断器基类"""

    def can_estimate(self, data: TLabelData) -> bool:
        raise NotImplementedError

    def estimate(self, data: TLabelData) -> List[ForceEstimate]:
        raise NotImplementedError

    def integrate(self, data: TLabelData) -> TLabelData:
        """将推断结果写入TLabelData的frames中（schema_v2）"""
        estimates = self.estimate(data)
        est_by_frame = {e.frame_idx: e for e in estimates}

        for frame in data.frames:
            est = est_by_frame.get(frame.frame_idx)
            if est is None:
                continue
            # 写入 schema_v2
            frame.schema_v2.force_magnitude = est.estimated_force_n
            # 在 sensor_specific 中记录推断信息
            frame.sensor_specific["force_source"] = "estimated_force"
            frame.sensor_specific["force_estimate_method"] = est.method
            frame.sensor_specific["force_estimate_confidence"] = est.confidence
            frame.sensor_specific["force_estimate_stiffness"] = est.stiffness_used

        # 在 TLabelData 级别标记
        data._force_estimates = estimates
        data._force_estimate_summary = {
            "method": self.__class__.__name__,
            "total_estimates": len(estimates),
            "avg_confidence": (
                sum(e.confidence for e in estimates) / len(estimates)
                if estimates else 0.0
            ),
        }
        return data


# ============================================================
# 基于形变 + 弹性体刚度的推断
# ============================================================

class DeformationForceEstimator(ForceEstimator):
    """
    利用 object_deformation 字段 + sensor_profile 中的弹性体刚度参数
    通过物理模型 F = k × δ 推断力
    """

    def can_estimate(self, data: TLabelData) -> bool:
        if not data.frames:
            return False
        for f in data.frames:
            deform = _sv2_scalar(f, "object_deformation", missing=None)
            if deform is not None and deform > 0.01:
                return True
        return False

    def _get_stiffness(self, data: TLabelData):
        """获取弹性体刚度，返回 (stiffness_n_m, is_default)"""
        sp = data.sensor_profile or {}
        elastomer = sp.get("elastomer", {})
        stiffness = elastomer.get("stiffness_n_m")
        if stiffness is not None and stiffness > 0:
            return float(stiffness), False
        sensor_type = data.sensor_info.get("type", "")
        default_k = get_default_stiffness(sensor_type)
        return default_k, True

    def estimate(self, data: TLabelData) -> List[ForceEstimate]:
        stiffness, is_default = self._get_stiffness(data)
        k_reference = 1000.0
        scale = stiffness / k_reference

        results = []
        for frame in data.frames:
            deform = _sv2_scalar(frame, "object_deformation", missing=None)
            # P9：无形变数据 ≠ 无形变。缺失帧不产出估算，不能填成"估算力 0.0 N"
            if deform is None:
                continue
            if deform < 0.001:
                results.append(ForceEstimate(
                    frame_idx=frame.frame_idx,
                    estimated_force_n=0.0,
                    confidence=0.9 if is_default else 0.95,
                    method="deformation_model",
                    deformation_used=deform,
                    stiffness_used=stiffness,
                ))
                continue

            force_n = deform * scale
            conf = 0.75 if not is_default else 0.55

            results.append(ForceEstimate(
                frame_idx=frame.frame_idx,
                estimated_force_n=force_n,
                confidence=conf,
                method="deformation_model",
                deformation_used=deform,
                stiffness_used=stiffness,
            ))
        return results


# ============================================================
# 基于图像帧差异的推断
# ============================================================

class ImageForceEstimator(ForceEstimator):
    """
    利用图像帧差异（pixel intensity change）推断变形量→映射为力估计
    """

    def can_estimate(self, data: TLabelData) -> bool:
        if not data.frames:
            return False
        images = data.get_images(max_frames=5)
        return any(img is not None for img in images)

    def estimate(self, data: TLabelData) -> List[ForceEstimate]:
        stiffness = get_default_stiffness(data.sensor_info.get("type", ""))
        images = data.get_images()

        intensity_diffs = []
        for i in range(len(images)):
            if i == 0 or images[i] is None or images[i - 1] is None:
                intensity_diffs.append(0.0)
            else:
                try:
                    img_curr = images[i].astype(float)
                    img_prev = images[i - 1].astype(float)
                    diff = abs(img_curr - img_prev).mean()
                    intensity_diffs.append(float(diff))
                except Exception:
                    intensity_diffs.append(0.0)

        max_diff = max(intensity_diffs) if intensity_diffs else 1.0
        if max_diff < 0.01:
            max_diff = 1.0

        results = []
        for i, frame in enumerate(data.frames):
            pseudo_deform = intensity_diffs[i] / max_diff if max_diff > 0 else 0.0
            k_reference = 1000.0
            force_n = pseudo_deform * (stiffness / k_reference) * 0.5
            conf = 0.45 if pseudo_deform > 0.1 else 0.35

            results.append(ForceEstimate(
                frame_idx=frame.frame_idx,
                estimated_force_n=force_n,
                confidence=conf,
                method="image_difference",
                deformation_used=pseudo_deform,
                stiffness_used=stiffness,
            ))
        return results


# ============================================================
# 组合推断器 + 自动推断入口
# ============================================================

class CompositeForceEstimator(ForceEstimator):
    """组合推断器 — 自动选择最佳推断策略"""

    def __init__(self):
        self._deformation_estimator = DeformationForceEstimator()
        self._image_estimator = ImageForceEstimator()
        self._chosen_method = None

    def can_estimate(self, data: TLabelData) -> bool:
        return (self._deformation_estimator.can_estimate(data) or
                self._image_estimator.can_estimate(data))

    def estimate(self, data: TLabelData) -> List[ForceEstimate]:
        if self._deformation_estimator.can_estimate(data):
            self._chosen_method = "deformation"
            return self._deformation_estimator.estimate(data)
        elif self._image_estimator.can_estimate(data):
            self._chosen_method = "image"
            return self._image_estimator.estimate(data)
        else:
            return []

    def integrate(self, data: TLabelData) -> TLabelData:
        if self._deformation_estimator.can_estimate(data):
            self._chosen_method = "deformation"
            return self._deformation_estimator.integrate(data)
        elif self._image_estimator.can_estimate(data):
            self._chosen_method = "image"
            return self._image_estimator.integrate(data)
        else:
            data._force_estimate_summary = {
                "method": "none",
                "reason": "No usable data for force estimation",
                "total_estimates": 0,
            }
            return data


def auto_force_estimate(data: TLabelData) -> TLabelData:
    """自动力推断入口函数"""
    estimator = CompositeForceEstimator()
    return estimator.integrate(data)


def has_real_force_data(data: TLabelData) -> bool:
    """
    判断TLabelData是否包含真实的力/力矩数据（而非形变别名）

    v0.17: 只检查 schema_v2，不检查旧 tlabel_v2。
    """
    if not data.frames:
        return False

    has_nonzero_force = False
    for f in data.frames:
        sv2 = f.schema_v2
        if sv2 is None:
            continue
        # force_vector 存在且非零 → L3数据，明确有力
        if sv2.force_vector is not None:
            fv = sv2.force_vector
            if any(abs(v) > 0.01 for v in fv):
                return True
        # force_magnitude 与 object_deformation 不同 → 独立力数据
        force = sv2.force_magnitude
        deform = sv2.object_deformation
        if force is not None and force > 0.01 and deform is not None:
            if abs(force - deform) > 0.001:
                return True
        if force is not None and force > 0.01:
            has_nonzero_force = True

    return False
