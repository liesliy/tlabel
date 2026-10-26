"""
TLabel Format v2 数据结构 — 统一触觉标注容器

所有适配器输出此格式，review阶段交互完全统一。

v0.17 Breaking Change: 彻底移除旧 tlabel_v2 字典格式兼容逻辑，
所有数据只使用 Schema V2 (14维结构化) 路径。
"""

import json
import math
from pathlib import Path
from typing import Optional, List, Dict, Any

from tlabel.core.schema import TLabelSchemaV2, SCHEMA_V2_FIELD_NAMES
from tlabel.core.capabilities import warn_if_invalid_capabilities


def _sv2_scalar(frame: "TLabelFrame", field: str, default: float = 0.0) -> float:
    """
    从 frame.schema_v2 获取标量字段值。

    如果 schema_v2 为 None 则抛出 ValueError。
    向量/枚举类型返回 default。
    """
    sv2 = frame.schema_v2
    if sv2 is None:
        raise ValueError(
            f"frame.schema_v2 is None — v0.17 requires Schema V2 data. "
            f"Field '{field}' cannot be accessed."
        )
    val = getattr(sv2, field, None)
    if val is None:
        return default
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, (int, float)):
        return float(val)
    # 向量/列表/枚举 — 返回 default
    return default


class TLabelFrame:
    """
    单帧标注数据 — v0.17 Breaking Change

    所有数据通过 schema_v2 (TLabelSchemaV2) 访问，不再使用 tlabel_v2 字典。
    """
    __slots__ = [
        "frame_idx", "timestamp_s", "manipulation_phase",
        "confidence", "sensor_specific", "patches",
        "image", "image_path",  # v0.12: 原始图像数据（numpy数组）或图像路径
        "primitive_label",  # v0.13: primitive名称，如 'wrap', 'lift', 'grasp'
        "primitive_confidence",  # v0.13: AI预标注的置信度 0.0-1.0
        "schema_v2",  # v0.17: Schema V2.1 结构化标注 (TLabelSchemaV2) — 必填
    ]

    def __init__(self, frame_idx: int, timestamp_s: float,
                 schema_v2: TLabelSchemaV2,
                 manipulation_phase: str = "idle",
                 confidence: float = 1.0,
                 sensor_specific: Optional[Dict] = None,
                 image: Optional[Any] = None,
                 image_path: Optional[str] = None,
                 primitive_label: Optional[str] = None,
                 primitive_confidence: float = 1.0):
        if schema_v2 is None:
            raise ValueError(
                "schema_v2 is required since v0.17 (Breaking Change). "
                "Construct with TLabelSchemaV2 instead of tlabel_v2 dict."
            )
        self.frame_idx = frame_idx
        self.timestamp_s = timestamp_s
        self.schema_v2 = schema_v2
        self.manipulation_phase = manipulation_phase
        self.confidence = confidence
        self.sensor_specific = sensor_specific or {}
        self.patches = []
        self.image = image  # numpy数组，用于可视化
        self.image_path = image_path  # 图像文件路径，用于懒加载
        self.primitive_label = primitive_label  # v0.13: primitive名称
        self.primitive_confidence = primitive_confidence  # v0.13: 置信度

    @property
    def contact(self) -> bool:
        return bool(self.schema_v2.contact)

    @property
    def slip_event(self) -> bool:
        return bool(self.schema_v2.slip_event)

    @property
    def force_magnitude(self) -> float:
        return self.schema_v2.force_magnitude if self.schema_v2.force_magnitude is not None else 0.0

    @property
    def is_modified(self) -> bool:
        return len(self.patches) > 0

    def to_schema_v2(self) -> TLabelSchemaV2:
        """返回 schema_v2 对象（v0.17 便捷方法）"""
        return self.schema_v2

    def patch(self, field: str, new_value: Any, cascade: bool = True):
        """修正单帧标注，支持联动 — 操作 schema_v2 字段"""
        sv2 = self.schema_v2
        old_value = getattr(sv2, field, None)
        patch_record = {
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
        }
        # 写入 schema_v2 字段
        if field == "contact":
            sv2.contact = bool(new_value)
        elif field == "slip_event":
            sv2.slip_event = bool(new_value)
        elif field in ("force_magnitude", "object_deformation", "temperature", "confidence"):
            setattr(sv2, field, new_value)
        else:
            setattr(sv2, field, new_value)

        cascades = []
        if cascade:
            cascades = self._apply_cascade(field, new_value)

        patch_record["cascade"] = cascades
        self.patches.append(patch_record)
        return patch_record

    def _apply_cascade(self, field: str, new_value: Any) -> List[Dict]:
        """联动规则 — 物理一致性约束（Schema V2 字段）"""
        sv2 = self.schema_v2
        cascades = []

        if field == "contact":
            if not new_value:
                # 接触归零 → 力度/滑移/形变 全部归零
                zero_fields = ["force_magnitude", "slip_event", "object_deformation"]
                for zf in zero_fields:
                    old_val = getattr(sv2, zf, None)
                    if old_val is not None and old_val != 0 and old_val is not False:
                        setattr(sv2, zf, 0.0 if zf != "slip_event" else False)
                        cascades.append({"field": zf, "old_value": old_val, "new_value": 0.0 if zf != "slip_event" else False})
                # force_vector 归零
                if sv2.force_vector is not None:
                    old_fv = sv2.force_vector
                    sv2.force_vector = None
                    cascades.append({"field": "force_vector", "old_value": old_fv, "new_value": None})
                if self.manipulation_phase in ("initial_contact", "stable_contact", "slip", "grasp", "hold"):
                    old_phase = self.manipulation_phase
                    self.manipulation_phase = "idle"
                    cascades.append({"field": "manipulation_phase", "old_value": old_phase, "new_value": "idle"})

        elif field == "slip_event":
            if new_value and not sv2.contact:
                # 滑移必须发生在接触状态 → 联动设置contact
                old_contact = sv2.contact
                sv2.contact = True
                cascades.append({"field": "contact", "old_value": old_contact, "new_value": True})
                if self.manipulation_phase == "idle":
                    old_phase = self.manipulation_phase
                    self.manipulation_phase = "slip"
                    cascades.append({"field": "manipulation_phase", "old_value": old_phase, "new_value": "slip"})

        elif field == "force_magnitude":
            if new_value is not None and new_value > 0 and not sv2.contact:
                # 有力必须有接触 → 联动设置contact
                old_contact = sv2.contact
                sv2.contact = True
                cascades.append({"field": "contact", "old_value": old_contact, "new_value": True})

        return cascades

    def to_dict(self, is_first: bool = False, is_last: bool = False) -> Dict:
        d = {
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 4),
            "is_first": is_first,
            "is_last": is_last,
            "schema_v2": self.schema_v2.to_dict(),
            "manipulation_phase": self.manipulation_phase,
            "confidence": round(self.confidence, 2),
        }
        if self.sensor_specific:
            d["sensor_specific"] = self.sensor_specific
        if self.patches:
            d["patches"] = self.patches
        if self.primitive_label is not None:
            d["primitive_label"] = self.primitive_label
            d["primitive_confidence"] = round(self.primitive_confidence, 4)
        return d


class TLabelData:
    """
    统一触觉标注容器 — 所有适配器的输出

    v0.17 Breaking Change: 所有数据通过 Schema V2 (14维) 访问。

    用法:
        data = tlabel.load("file.pkl")
        data.review()        # Jupyter彩色面板
        data.export("out")   # 导出
    """

    def __init__(self, frames: List[TLabelFrame],
                 sensor_info: Dict,
                 episode_info: Dict,
                 capabilities: Dict,
                 schema_version: str = "0.17.0",
                 sensor_id: Optional[str] = None,
                 calibration_params: Optional[Dict] = None,
                 sensor_profile: Optional[Dict] = None):
        self.frames = frames
        self.sensor_info = sensor_info
        self.episode_info = episode_info
        self.capabilities = capabilities
        # v0.23.3：capabilities 合规性检查（仅告警，不改写输出；
        # 0.24.0 起改为出口归一化）。详见 tlabel/core/capabilities.py。
        warn_if_invalid_capabilities(capabilities, source="TLabelData")
        self.schema_version = schema_version
        self.sensor_id = sensor_id
        self.calibration_params = calibration_params or {}
        self.sensor_profile = self._apply_sensor_profile_defaults(sensor_profile, sensor_info)
        self._predict_results = None
        self.primitive_annotations = []
        self.tactile_events = []

    @staticmethod
    def _apply_sensor_profile_defaults(sensor_profile: Optional[Dict],
                                        sensor_info: Optional[Dict]) -> Optional[Dict]:
        """为sensor_profile填充弹性体刚度默认值"""
        from tlabel.predict.force_estimator import get_default_stiffness

        if sensor_profile is None:
            sensor_profile = {}

        elastomer = sensor_profile.get("elastomer", {})
        if not elastomer:
            elastomer = {}

        if "stiffness_n_m" not in elastomer or elastomer["stiffness_n_m"] is None:
            sensor_type = (sensor_info or {}).get("type", "")
            elastomer["stiffness_n_m"] = get_default_stiffness(sensor_type)

        sensor_profile["elastomer"] = elastomer
        return sensor_profile

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames[-1].timestamp_s - self.frames[0].timestamp_s

    @property
    def sensor_type(self) -> str:
        return self.sensor_info.get("type", "unknown")

    @property
    def dimension_keys(self) -> List[str]:
        """返回14维 Schema V2.1 字段名列表"""
        return list(SCHEMA_V2_FIELD_NAMES)

    @property
    def modified_count(self) -> int:
        return sum(1 for f in self.frames if f.is_modified)

    def get_frame(self, frame_idx: int, logical: bool = False) -> Optional[TLabelFrame]:
        """按帧索引获取帧数据"""
        if logical:
            if 0 <= frame_idx < len(self.frames):
                return self.frames[frame_idx]
            return None
        for f in self.frames:
            if f.frame_idx == frame_idx:
                return f
        return None

    def batch_patch(self, start_frame: int, end_frame: int,
                    field: str, new_value: Any, cascade: bool = True) -> int:
        """批量修正帧区间"""
        count = 0
        for f in self.frames:
            if start_frame <= f.frame_idx <= end_frame:
                old = getattr(f.schema_v2, field, None)
                if old != new_value:
                    f.patch(field, new_value, cascade=cascade)
                    count += 1
        return count

    # ============================================================
    # v0.13.0: Motor Primitive 标注
    # ============================================================

    def add_primitive(self, name: str, start_frame: int, end_frame: int,
                    confidence: float = 1.0, source: str = 'manual') -> None:
        """添加 Motor Primitive 标注"""
        from tlabel.core.primitive import PrimitiveAnnotation
        self.primitive_annotations.append(
            PrimitiveAnnotation(name, start_frame, end_frame, confidence, source)
        )

    def predict_primitives(self, taxonomy=None, min_confidence: float = 0.4) -> int:
        """自动预标注 Motor Primitive"""
        from tlabel.predict.engine import PredictEngine
        engine = PredictEngine()
        annotations = engine.predict_primitives(self, taxonomy=taxonomy)
        count = 0
        for ann in annotations:
            if ann['confidence'] >= min_confidence:
                try:
                    self.add_primitive(
                        name=ann['primitive_name'],
                        start_frame=ann['start_frame'],
                        end_frame=ann['end_frame'],
                        confidence=ann['confidence'],
                        source=ann.get('source', 'ai_predicted'),
                    )
                    count += 1
                except ValueError:
                    pass
        return count

    def get_primitive_timeline(self) -> List:
        """返回 primitive 时间线"""
        return [(p.primitive_name, p.start_frame, p.end_frame)
                for p in self.primitive_annotations]

    def get_primitive_at_frame(self, frame_idx: int) -> Optional[str]:
        """获取某帧对应的 primitive 名称"""
        for p in self.primitive_annotations:
            if p.contains_frame(frame_idx):
                return p.primitive_name
        return None

    # ============================================================
    # v0.14.0: Tactile Event 标注
    # ============================================================

    def add_event(self, event_type: str, frame_idx: int,
                  confidence: float = 1.0, source: str = "manual",
                  start_frame: Optional[int] = None, end_frame: Optional[int] = None,
                  metadata: Optional[Dict] = None) -> None:
        """添加触觉事件标注"""
        from tlabel.core.events import TactileEvent
        te = TactileEvent(
            event_type=event_type,
            frame_idx=frame_idx,
            confidence=confidence,
            source=source,
            start_frame=start_frame,
            end_frame=end_frame,
            metadata=metadata,
        )
        self.tactile_events.append(te)

    def get_events_at_frame(self, frame_idx: int) -> List:
        """获取某帧对应的所有触觉事件"""
        return [e for e in self.tactile_events if e.contains_frame(frame_idx)]

    def get_events_by_type(self, event_type: str) -> List:
        """按类型获取事件列表"""
        return [e for e in self.tactile_events if e.event_type == event_type]

    def detect_events(self, min_confidence: float = 0.5) -> int:
        """自动检测触觉事件并应用"""
        from tlabel.predict.engine import PredictEngine
        engine = PredictEngine()
        return engine.apply_events(self, min_confidence=min_confidence)

    # ============================================================
    # v0.18.0: Primitive + Event 标注工具集成
    # ============================================================

    def validate_annotations(self) -> Dict:
        """校验 primitive/event 标注一致性

        检查项:
        - Primitive 名称有效性
        - 帧范围合法性
        - Event 类型有效性
        - confidence/source 合法性

        Returns:
            dict: {valid: bool, errors: list, warnings: list, stats: dict}
        """
        from tlabel.core.annotation import validate_annotations
        return validate_annotations(self)

    def annotate_from_taxonomy(self, taxonomy=None, min_confidence: float = 0.4,
                                clear_existing: bool = False) -> int:
        """批量应用 taxonomy 规则进行 primitive 预标注

        Args:
            taxonomy: TaxonomyConfig，None 使用默认
            min_confidence: 最低置信度阈值
            clear_existing: 是否清除已有标注

        Returns:
            新增标注数量
        """
        from tlabel.core.annotation import annotate_from_taxonomy
        return annotate_from_taxonomy(self, taxonomy, min_confidence, clear_existing)

    def annotate_events_auto(self, clear_existing: bool = False) -> int:
        """从数据模式自动检测触觉事件

        检测: contact_onset/loss, slip区间, force_spike, stable_grip

        Returns:
            新增事件数量
        """
        from tlabel.core.annotation import annotate_events_from_data
        return annotate_events_from_data(self, clear_existing)

    def clear_annotations(self, primitives: bool = True, events: bool = True) -> Dict:
        """清除标注

        Returns:
            dict: {cleared_primitives: int, cleared_events: int}
        """
        from tlabel.core.annotation import clear_annotations
        return clear_annotations(self, primitives, events)

    def get_annotation_summary(self) -> Dict:
        """获取标注摘要 — 统一查询接口

        Returns:
            dict: {primitives, events, timeline, validation}
        """
        from tlabel.core.annotation import get_annotation_summary
        return get_annotation_summary(self)

    def review(self, lang: str = "auto", **kwargs):
        """弹出Jupyter彩色标注面板"""
        from tlabel.viewer.panel import TLabelPanel
        panel = TLabelPanel(self, lang=lang, **kwargs)
        return panel

    def auto_label(self, min_confidence: float = 0.6,
                   target_fields: Optional[List[str]] = None,
                   fit_first: bool = True,
                   engine: str = "auto",
                   enabled_fields: Optional[List[str]] = None,
                   enable_postprocess: bool = True,
                   enable_hmm_phase: bool = True) -> Dict:
        """AI辅助预标注"""
        use_ml = False
        predict_results = None

        if engine in ("auto", "ml"):
            try:
                from tlabel.predict.ml_engine import MLEngine, MLEngineConfig
                config = MLEngineConfig(
                    enabled_fields=enabled_fields,
                    enable_postprocess=enable_postprocess,
                    enable_hmm_phase=enable_hmm_phase,
                )
                ml_engine = MLEngine(config)
                if fit_first:
                    ml_engine.fit(self)
                if ml_engine._is_fitted:
                    results = ml_engine.predict(self, target_fields=target_fields)
                    predict_results = results
                    applied = ml_engine.apply(self, results, min_confidence=min_confidence)
                    summary = ml_engine.summary(results)
                    summary["applied_count"] = applied
                    summary["engine"] = "ml"
                    use_ml = True
                elif engine == "ml":
                    return {"error": "ML engine failed to fit", "fit_report": ml_engine.fit_report(), "engine": "ml"}
            except ImportError:
                if engine == "ml":
                    return {"error": "ML engine requires: pip install tlabel[ml]", "engine": "ml"}

        if not use_ml:
            from tlabel.predict.engine import PredictEngine, PredictConfig
            config = PredictConfig(
                enable_postprocess=enable_postprocess,
                enable_hmm_phase=enable_hmm_phase,
            )
            rule_engine = PredictEngine(config)
            if fit_first:
                rule_engine.fit(self)
            results = rule_engine.predict(self, target_fields=target_fields)
            predict_results = results
            applied = rule_engine.apply(self, results, min_confidence=min_confidence)
            summary = rule_engine.summary(results)
            summary["applied_count"] = applied
            summary["engine"] = "rule"

        self._predict_results = predict_results
        summary["low_confidence_frames"] = [
            r.frame_idx for r in (predict_results or [])
            if any(c < min_confidence for c in r.confidence.values())
        ]

        return summary

    def export(self, output_path: str, format: str = "auto"):
        """导出标注数据"""
        from tlabel.export.writer import export_data
        return export_data(self, output_path, format=format)

    def export_ftp1(self, output_path: str, sensor_name: str = "GelSightMini",
                    functional_areas=None, side: str = "right",
                    group: str = "gripper", **kwargs) -> Dict:
        """导出为FTP-1/MTTS Zarr格式"""
        from tlabel.converters.ftp1 import tlabel_to_ftp1
        return tlabel_to_ftp1(
            self, output_path,
            sensor_name=sensor_name,
            functional_areas=functional_areas,
            side=side,
            group=group,
            **kwargs,
        )

    # ============================================================
    # Episode级标注
    # ============================================================

    def label_episode(self,
                      outcome: str = "inconclusive",
                      manipulation_type: str = "other",
                      difficulty: str = "medium",
                      notes: str = "",
                      annotator: str = "",
                      verified: bool = False) -> "EpisodeLabel":
        """Episode级标注"""
        valid_outcomes = {"success", "partial", "failure", "inconclusive"}
        valid_types = {"grasp", "pinch", "poke", "slide", "push", "pull", "tap", "lift", "place", "other"}
        valid_diffs = {"easy", "medium", "hard"}

        if outcome not in valid_outcomes:
            raise ValueError(f"Invalid outcome '{outcome}', must be one of {valid_outcomes}")
        if manipulation_type not in valid_types:
            raise ValueError(f"Invalid manipulation_type '{manipulation_type}', must be one of {valid_types}")
        if difficulty not in valid_diffs:
            raise ValueError(f"Invalid difficulty '{difficulty}', must be one of {valid_diffs}")

        label = EpisodeLabel(
            outcome=outcome,
            manipulation_type=manipulation_type,
            difficulty=difficulty,
            notes=notes,
            annotator=annotator,
            verified=verified,
        )
        self.episode_info["episode_label"] = label.to_dict()
        return label

    @property
    def episode_label(self) -> Optional["EpisodeLabel"]:
        """获取当前Episode级标注"""
        raw = self.episode_info.get("episode_label")
        if raw is None:
            return None
        if isinstance(raw, EpisodeLabel):
            return raw
        return EpisodeLabel.from_dict(raw)

    # ============================================================
    # 数据质量评分
    # ============================================================

    def quality_score(self, verbose: bool = False) -> Dict:
        """数据质量评分"""
        from tlabel.quality.scorer import QualityScorer
        scorer = QualityScorer(verbose=verbose)
        return scorer.score(self)

    # ============================================================
    # describe统计摘要
    # ============================================================

    def describe(self, fields: Optional[List[str]] = None) -> Dict:
        """
        统计摘要 — 类pandas describe（Schema V2 数值字段）
        """
        if not self.frames:
            return {}

        # Schema V2 的标量数值字段
        SCALAR_FIELDS = [
            "contact", "force_magnitude", "slip_event",
            "object_deformation", "temperature", "confidence",
        ]
        keys = fields if fields else SCALAR_FIELDS
        result = {}

        for key in keys:
            values = [_sv2_scalar(f, key) for f in self.frames]
            if not values:
                continue

            n = len(values)
            mean = sum(values) / n
            var = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(var)
            sorted_vals = sorted(values)
            q25 = sorted_vals[n // 4] if n >= 4 else sorted_vals[0]
            q50 = sorted_vals[n // 2]
            q75 = sorted_vals[3 * n // 4] if n >= 4 else sorted_vals[-1]

            result[key] = {
                "count": n,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(min(values), 4),
                "25%": round(q25, 4),
                "50%": round(q50, 4),
                "75%": round(q75, 4),
                "max": round(max(values), 4),
            }

        return result

    def to_dict(self) -> Dict:
        """转换为字典（Schema V2 格式）"""
        contact_count = sum(1 for f in self.frames if f.contact > 0.5)
        slip_count = sum(1 for f in self.frames if f.slip_event > 0.5)

        # v0.7: 加载特征元数据
        try:
            from tlabel.features_meta import get_feature_metadata_summary
            feature_metadata = get_feature_metadata_summary()
        except ImportError:
            feature_metadata = None

        result = {
            "schema_version": self.schema_version,
            "format": "tlabel_schema_v2",
            "schema_version_v2": "2.1",
            "tlabel_dimensions_v2": 14,
            "feature_names_v2": list(SCHEMA_V2_FIELD_NAMES),
            "feature_metadata": feature_metadata,
            "sensor": self.sensor_info,
            "sensor_id": self.sensor_id,
            "sensor_profile": self.sensor_profile,
            "calibration": self.calibration_params if self.calibration_params else None,
            "episode": {
                **self.episode_info,
                "episode_label": self.episode_label.to_dict() if self.episode_label else None,
                "num_frames": self.num_frames,
                "duration_s": round(self.duration_s, 2),
                "stats": {
                    "contact_frames": contact_count,
                    "contact_ratio": round(contact_count / max(self.num_frames, 1), 4),
                    "slip_frames": slip_count,
                    "slip_ratio": round(slip_count / max(self.num_frames, 1), 4),
                    "modified_frames": self.modified_count,
                }
            },
            "capabilities": self.capabilities,
            "frames": [f.to_dict(is_first=(i == 0), is_last=(i == len(self.frames) - 1))
                      for i, f in enumerate(self.frames)],
            "primitive_annotations": [p.to_dict() for p in self.primitive_annotations]
                if self.primitive_annotations else [],
            "tactile_events": [e.to_dict() for e in self.tactile_events]
                if self.tactile_events else [],
        }
        return result

    def get_images(self, max_frames: Optional[int] = None) -> List[Any]:
        """提取所有帧的图像数据（用于可视化）"""
        images = []
        limit = max_frames if max_frames else len(self.frames)
        for i, frame in enumerate(self.frames):
            if i >= limit:
                break
            if hasattr(frame, 'image') and frame.image is not None:
                images.append(frame.image)
            elif hasattr(frame, 'image_path') and frame.image_path:
                try:
                    import cv2
                    img = cv2.imread(frame.image_path)
                    images.append(img)
                except Exception:
                    images.append(None)
            else:
                images.append(None)
        return images

    def _repr_html_(self):
        """Jupyter自动渲染面板"""
        panel = self.review()
        return panel._repr_html_()

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        """支持 data[0] 按索引访问帧"""
        return self.frames[index]

    def __repr__(self):
        return (f"TLabelData(sensor={self.sensor_type}, "
                f"frames={self.num_frames}, "
                f"duration={self.duration_s:.1f}s, "
                f"modified={self.modified_count})")


# ============================================================
# Episode级标注
# ============================================================

from dataclasses import dataclass, field as dc_field
from enum import Enum


class ManipulationType(str, Enum):
    """操作类型枚举"""
    GRASP = "grasp"
    PINCH = "pinch"
    POKE = "poke"
    SLIDE = "slide"
    PUSH = "push"
    PULL = "pull"
    TAP = "tap"
    LIFT = "lift"
    PLACE = "place"
    OTHER = "other"


class EpisodeOutcome(str, Enum):
    """Episode结果枚举"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    INCONCLUSIVE = "inconclusive"


class Difficulty(str, Enum):
    """难度等级"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class EpisodeLabel:
    """Episode级标注 — 从帧级提升到任务级"""
    outcome: str = "inconclusive"
    manipulation_type: str = "other"
    difficulty: str = "medium"
    notes: str = ""
    annotator: str = ""
    verified: bool = False

    def to_dict(self) -> Dict:
        return {
            "outcome": self.outcome,
            "manipulation_type": self.manipulation_type,
            "difficulty": self.difficulty,
            "notes": self.notes,
            "annotator": self.annotator,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "EpisodeLabel":
        return cls(
            outcome=d.get("outcome", "inconclusive"),
            manipulation_type=d.get("manipulation_type", "other"),
            difficulty=d.get("difficulty", "medium"),
            notes=d.get("notes", ""),
            annotator=d.get("annotator", ""),
            verified=d.get("verified", False),
        )
