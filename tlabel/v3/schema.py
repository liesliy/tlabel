"""
TLabel V3.0 参考实现 — 字段定义层

本模块是规范 `tlabel-spec-v3.md` 的最小 Python 映射。每个常量都标注了它编码的条款编号。

设计原则：
1. 这里只放「规范说了什么」，不放「怎么方便怎么来」。
2. 任何偏离规范的便利性设计都是缺陷，不是优化。
3. 不依赖 V2 的任何结构——本模块可以被第三方单独阅读和移植。

版本：3.0.0
权威来源：tlabel-spec-v3.md
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple

SPEC_VERSION = "3.0.0"

# ─────────────────────────────────────────────────────────────
# 字段集合（TL-4.4-1）
# ─────────────────────────────────────────────────────────────

#: 核心字段。每一条序列化记录中这 8 个键必须存在，即使值为 null。
#: 顺序即规范第 5 章顺序。
CORE_FIELDS: Tuple[str, ...] = (
    "contact",
    "contact_centroid",
    "normal_force",
    "force_vector",
    "torque_vector",
    "slip_event",
    "slip_velocity",
    "temperature",
)

#: 上下文扩展（TL-6.1、TL-6.2）
CONTEXT_EXTENSION_FIELDS: Tuple[str, ...] = (
    "manipulation_phase",
    "contact_region",
)

#: 语义扩展（TL-6.3、TL-6.4）
SEMANTIC_EXTENSION_FIELDS: Tuple[str, ...] = (
    "texture_class",
    "object_deformation",
)

#: 弃用字段（TL-6.5）。原生 V3.0 数据不得输出非 null 值。
DEPRECATED_FIELDS: Tuple[str, ...] = (
    "force_magnitude",
    "confidence",
    "compliance_level",
)

ALL_FIELDS: Tuple[str, ...] = (
    CORE_FIELDS + CONTEXT_EXTENSION_FIELDS + SEMANTIC_EXTENSION_FIELDS + DEPRECATED_FIELDS
)

#: 数据集级允许出现的全部键（TL-10.1-3 的大小写变体检测用）。
#:
#: 和帧级字段一样，`Time_Base` 在 Python 里能读、在 JSON 里能存，但下游
#: `ds["time_base"]` 会拿到 None——而 None 在这个规范里是**合法值**（表示
#: 原点未声明，会由 9.7-2 报 KEY_MISSING）。所以大小写错误的表现不是崩溃，
#: 是「数据看起来少了点什么」，最难查的那一类。
DATASET_KEYS: Tuple[str, ...] = (
    "session_id",
    "time_base",
    "frames_of_reference",
    "capabilities",
    "frames",
    "provenance",
    "migration",
    "transforms",
    "instances",
    "spec_version",
)

# ─────────────────────────────────────────────────────────────
# 数据类型（TL-4.3-2）
# ─────────────────────────────────────────────────────────────

DATA_TYPES: Dict[str, str] = {
    "contact": "tristate_boolean",
    "contact_centroid": "array2",
    "normal_force": "scalar",
    "force_vector": "array3",
    "torque_vector": "array3",
    "slip_event": "tristate_boolean",
    "slip_velocity": "array2",
    "temperature": "scalar",
    "manipulation_phase": "string",
    "contact_region": "string",
    "texture_class": "string",
    "object_deformation": "scalar",
    "force_magnitude": "scalar",
    "confidence": "scalar",
    "compliance_level": "string",
}

#: 每个字段必须引用的坐标系维度；None 表示该字段不涉及坐标
#: TL-5.2-3 / TL-5.4-1 / TL-5.5-1 / TL-5.7-2
FIELD_DIMENSION: Dict[str, Optional[int]] = {
    "contact_centroid": 2,
    "force_vector": 3,
    "torque_vector": 3,
    "slip_velocity": 2,
}

#: 引用坐标系的字段 → 存放 frame_id 的键名
FRAME_KEY: Dict[str, str] = {
    "contact_centroid": "contact_centroid_frame",
    "force_vector": "force_vector_frame",
    "torque_vector": "torque_vector_frame",
    "slip_velocity": "slip_velocity_frame",
}

# ─────────────────────────────────────────────────────────────
# 单位（TL-4.2-1、附录 A）
# ─────────────────────────────────────────────────────────────

UNITS: Dict[str, str] = {
    "contact_centroid": "mm",
    "normal_force": "N",
    "force_vector": "N",
    "torque_vector": "N.m",
    "slip_velocity": "mm/s",
    "temperature": "Cel",
    "object_deformation": "mm",
    "force_magnitude": "N",
    "confidence": "1",
}

# ─────────────────────────────────────────────────────────────
# 枚举（附录 B）
# ─────────────────────────────────────────────────────────────

CONTACT_REGION_VALUES: FrozenSet[str] = frozenset(
    {"palmar", "digital", "lateral", "proximal", "distal", "dorsal", "other"}
)

TEXTURE_CLASS_VALUES: FrozenSet[str] = frozenset(
    {"smooth", "rough", "granular", "fibrous", "sticky", "slippery"}
)

NORMAL_REFERENCE_VALUES: FrozenSet[str] = frozenset(
    {"sensor_surface_nominal", "contact_interface_local", "measurement_axis"}
)

MEASUREMENT_TARGET_VALUES: FrozenSet[str] = frozenset(
    {"sensor_body", "contact_interface", "object"}
)

#: 附录 A 的 UCUM 单位码（TL-4.2-1）。
#:
#: 核心字段的单位由字段定义固定（centroid=mm、force=N、torque=N.m、
#: slip_velocity=mm/s、temperature=Cel），数据里不写单位——写出来就有两个
#: 真相源（TL-4.2-2）。这个集合只用于 `sensor_specific` 里生产者自报的单位，
#: 以及拒绝「把像素当长度」这类冒名单位。
UNIT_CODES: FrozenSet[str] = frozenset(
    {"N", "N.m", "mm", "mm/s", "s", "Cel", "1"}
)

CENTROID_METHOD_VALUES: FrozenSet[str] = frozenset({"geometric", "pressure_weighted"})

#: TL-8.4-4：一个记录的变换必须声明的键。
#:
#: 少了任何一项，接收方就得猜——而变换猜错的后果是「看起来对齐了，
#: 数值差一个旋转」，比没有变换更难发现。
TRANSFORM_REQUIRED_KEYS: Tuple[str, ...] = (
    "transform_id",
    "source_frame",
    "target_frame",
    "rotation_representation",
    "rotation_element_order",
    "translation",
    "translation_frame",
    "translation_unit",
    "calibration",
)

#: 旋转表示方式（TL-8.4-4）
ROTATION_REPRESENTATIONS: FrozenSet[str] = frozenset(
    {"quaternion", "matrix_3x3", "euler_zyx"}
)

#: 变换是标定得来还是估计得来（TL-8.4-4）
CALIBRATION_KINDS: FrozenSet[str] = frozenset({"calibrated", "estimated"})


FRAME_TYPE_VALUES: FrozenSet[str] = frozenset(
    {"sensor_local", "robot_link", "robot_base", "world"}
)

HANDEDNESS_VALUES: FrozenSet[str] = frozenset({"right"})

TIME_BASE_VALUES: FrozenSet[str] = frozenset(
    {"dataset_start", "session_start", "unix_epoch", "device_clock"}
)

# ─────────────────────────────────────────────────────────────
# 可用性状态（TL-7.2）
# ─────────────────────────────────────────────────────────────

#: 核心状态：回答「这个物理量为什么没有值」
CORE_STATES: FrozenSet[str] = frozenset(
    {"not_acquired", "invalid", "unsupported", "unknown_for_migration"}
)

#: 扩展状态：回答「工作流走到哪一步」或「这一帧是否适用」
#: TL-7.2-4：Core 校验器必须识别，不得报为未知枚举
EXTENSION_STATES: FrozenSet[str] = frozenset({"not_annotated", "not_applicable"})

ALL_STATES: FrozenSet[str] = CORE_STATES | EXTENSION_STATES

#: 表示「值必须为 null」的状态（TL-7.2-5）
STATES_REQUIRING_NULL: FrozenSet[str] = frozenset(
    {"not_acquired", "invalid", "not_annotated", "not_applicable"}
)

#: 声称「有能力、只是这次没拿到」的状态（TL-4.5-3）。
#:
#: 只有这三个与 `capability: false` 矛盾。`unsupported`（不支持）、
#: `not_applicable`（这帧无意义）、`unknown_for_migration`（不知道当时状态）
#: 都不声称能力存在，可以和 `capability: false` 并存——它们从不同角度描述
#: 同一件事「没有可用值」。
STATES_CLAIMING_CAPABILITY: FrozenSet[str] = frozenset(
    {"not_acquired", "invalid", "not_annotated"}
)

# ─────────────────────────────────────────────────────────────
# 来源类型（TL-9.3）
# ─────────────────────────────────────────────────────────────

PROVENANCE_TYPES: FrozenSet[str] = frozenset(
    {
        "direct_measurement",
        "derived_measurement",
        "model_inference",
        "human_annotation",
        "externally_supplied",
    }
)

#: 各类来源必须携带的记录项（TL-9.3-2）
PROVENANCE_REQUIRED_KEYS: Dict[str, Tuple[str, ...]] = {
    "direct_measurement": ("sensor_or_channel_id",),
    "derived_measurement": ("input_refs", "method_id"),
    "model_inference": ("method_id",),
    "human_annotation": ("annotator_id",),
    "externally_supplied": ("source_id",),
}

# 注：规范没有规定「哪个字段只能用哪类来源」的矩阵。
# 早期草稿里曾有一张 ALLOWED_PROVENANCE 表，已删除——它会把
# 「由力信号阈值判断 contact」这类合法的 derived_measurement 判成违规。
# 校验器只查两件事：类型在五类内（TL-9.3-1）、该类型必需信息齐全（TL-9.3-2）。

# ─────────────────────────────────────────────────────────────
# 数值容差
# ─────────────────────────────────────────────────────────────

#: 法向单位向量的模长容差（TL-5.3-4）
NORMAL_VECTOR_TOLERANCE = 1e-3

#: normal_force 与 |F·n| 的一致性容差（TL-5.3-3）
NORMAL_FORCE_TOLERANCE = 1e-2

# ─────────────────────────────────────────────────────────────
# 错误码（TL-10.5）
# ─────────────────────────────────────────────────────────────

ERROR_CODES: Dict[str, str] = {
    "TYPE_MISMATCH": "A",
    "LENGTH_MISMATCH": "A",
    "ENUM_INVALID": "A",
    "RANGE_INVALID": "A",
    "NON_FINITE_VALUE": "A",
    "KEY_MISSING": "A",
    "UNIT_UNKNOWN": "A",
    "FIELD_NAME_CASE": "A",
    "NORMAL_FORCE_MISMATCH": "B",
    "VALUE_WITHOUT_CAPABILITY": "B",
    "NULL_WITHOUT_STATUS": "B",
    "STATUS_WITHOUT_NULL": "B",
    "STATUS_CAPABILITY_CONFLICT": "B",
    "UNSUPPORTED_WITH_CAPABILITY": "B",
    "DATASET_UNIT_OVERRIDE": "B",
    "SLIP_VELOCITY_WITHOUT_SLIP": "B",
    "CENTROID_CONTACT_INCONSISTENT": "B",
    "VALUE_WITHOUT_PROVENANCE": "B",
    "PROVENANCE_INCOMPLETE": "B",
    "PROVENANCE_TYPE_FORBIDDEN": "B",
    "REQUIRED_CONTEXT_MISSING": "B",
    "DEPRECATED_FIELD_POPULATED": "B",
    "MIGRATION_INFERRED": "B",
    "MIGRATION_STATE_IN_NATIVE": "B",
    "MIGRATION_AUDIT_MISSING": "B",
    "INSTANCE_NOT_DECLARED": "B",
    "TIME_BASE_INCONSISTENT": "B",
    "FRAME_NOT_DECLARED": "C",
    "FRAME_DIMENSION_MISMATCH": "C",
    "FRAME_AXES_MISMATCH": "C",
    "FRAME_OUT_OF_PLANE": "C",
    "NORMAL_VECTOR_MISSING": "C",
    "NORMAL_VECTOR_NOT_UNIT": "C",
    "NORMAL_VECTOR_FRAME_MISMATCH": "C",
    "TORQUE_REFERENCE_MISSING": "C",
    "TORQUE_UNIT_SCALE": "C",
    "SENSOR_SPECIFIC_UNSTRUCTURED": "D",
    "METHOD_ID_MISSING": "D",
}

#: 校验结论三态（TL-10.4-2）
OUTCOME_ERROR = "error"
OUTCOME_WARNING = "warning"
OUTCOME_NOT_VERIFIED = "not_verified"


def expected_keys() -> Tuple[str, ...]:
    """一条序列化记录里必须出现的全部字段键（TL-4.4-1）。"""
    return ALL_FIELDS


def is_core(field: str) -> bool:
    return field in CORE_FIELDS


def is_deprecated(field: str) -> bool:
    return field in DEPRECATED_FIELDS
