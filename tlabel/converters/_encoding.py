"""
Schema V2 → 14 维 float32 向量编码（LeRobot 两条路径的共享事实源）

TLabel 有两条写入 LeRobot 的路径，它们必须使用同一套编码规则：

  1. ``tlabel/converters/lerobot_export.py``  —— 从零创建 LeRobot 数据集
  2. ``tlabel/converters/lerobot.py``         —— 把标注写回已有 LeRobot 数据集

v0.23.3 之前两条路径各自实现，写回路径对非标量字段一律填 ``0.0``，
导致向量字段（contact_centroid / force_vector / torque_vector / slip_velocity）
和枚举字段（contact_region / manipulation_phase / texture_class / compliance_level）
被静默抹平 —— 不报错、不告警，数据直接丢失（PIT-6）。

本模块是唯一事实源，编码规则：

  - bool                → 1.0 / 0.0
  - 标量 float          → 直接取值，None → 0.0（confidence 缺省 1.0）
  - 向量 list/tuple     → L2 范数，None → 0.0
  - str 枚举            → 类别索引，None / 未知值 → -1.0
                          （compliance_level 缺省 1.0，沿用历史行为）
"""

import math
from typing import Any, Dict, List, Mapping

from tlabel.core.schema import SCHEMA_V2_FIELD_NAMES

# ================================================================
# 枚举 → 类别索引映射
# ================================================================

CONTACT_REGION_MAP: Dict[str, float] = {
    v: float(i) for i, v in enumerate(
        ["palmar", "digital", "lateral", "proximal", "distal", "dorsal", "other"]
    )
}

MANIPULATION_PHASE_MAP: Dict[str, float] = {
    v: float(i) for i, v in enumerate(
        ["pre_contact", "approach", "grasp", "lift", "hold", "place"]
    )
}

TEXTURE_CLASS_MAP: Dict[str, float] = {
    v: float(i) for i, v in enumerate(
        ["smooth", "rough", "granular", "fibrous", "sticky", "slippery"]
    )
}

COMPLIANCE_LEVEL_MAP: Dict[str, float] = {"L1": 1.0, "L2": 2.0, "L3": 3.0, "L4": 4.0}

# 字段分类
BOOL_FIELDS = frozenset({"contact", "slip_event"})

VECTOR_FIELDS = frozenset({
    "contact_centroid",   # [x, y]           → L2 范数
    "force_vector",       # [Fx, Fy, Fz]     → L2 范数
    "torque_vector",      # [Mx, My, Mz]     → L2 范数
    "slip_velocity",      # [vx, vy]         → L2 范数
})

ENUM_FIELDS: Dict[str, Dict[str, float]] = {
    "contact_region": CONTACT_REGION_MAP,
    "manipulation_phase": MANIPULATION_PHASE_MAP,
    "texture_class": TEXTURE_CLASS_MAP,
    "compliance_level": COMPLIANCE_LEVEL_MAP,
}

# 枚举字段的缺省值（None / 未知值）
ENUM_DEFAULTS: Dict[str, float] = {
    "contact_region": -1.0,
    "manipulation_phase": -1.0,
    "texture_class": -1.0,
    "compliance_level": 1.0,  # 历史行为：缺省按 L1 编码
}

# 标量字段的缺省值
SCALAR_DEFAULTS: Dict[str, float] = {
    "confidence": 1.0,
}


# ================================================================
# 单字段编码
# ================================================================

def _vec_norm(value: Any) -> float:
    """向量 → L2 范数；None / 不可迭代 → 0.0"""
    if value is None:
        return 0.0
    try:
        return float(math.sqrt(sum(float(x) ** 2 for x in value)))
    except (TypeError, ValueError):
        return 0.0


def _enum_idx(value: Any, mapping: Mapping[str, float], default: float) -> float:
    """枚举字符串 → 类别索引；None / 未知值 → default"""
    if value is None:
        return default
    try:
        return float(mapping.get(value, default))
    except TypeError:  # 不可哈希
        return default


def _scalar(value: Any, default: float) -> float:
    """标量 → float；None / 不可转换 → default"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def encode_field(name: str, value: Any) -> float:
    """按 Schema V2 字段名编码单个字段值。

    未知字段名走保守回退：bool → 1.0/0.0，数值 → float，其余 → 0.0。
    """
    # bool 必须在数值之前判断：Python 中 bool 是 int 的子类
    if name in BOOL_FIELDS:
        return 1.0 if value else 0.0

    if name in VECTOR_FIELDS:
        return _vec_norm(value)

    if name in ENUM_FIELDS:
        return _enum_idx(value, ENUM_FIELDS[name], ENUM_DEFAULTS[name])

    # 标量字段及未知字段
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # 未知字段名却给了字符串：无法编码，保守置 0.0
        return 0.0
    if value is None:
        return SCALAR_DEFAULTS.get(name, 0.0)
    # list/tuple 出现在非向量字段名上：按范数处理而不是丢弃
    if isinstance(value, (list, tuple)):
        return _vec_norm(value)
    return 0.0


# ================================================================
# 整帧编码
# ================================================================

def _read_field(sv2: Any, name: str) -> Any:
    """从 dataclass 或 dict 读取字段。"""
    if isinstance(sv2, Mapping):
        return sv2.get(name)
    return getattr(sv2, name, None)


def encode_schema_v2(sv2: Any, field_names: List[str] = None) -> List[float]:
    """把一帧 Schema V2 编码为 float 列表。

    参数:
        sv2: ``TLabelSchemaV2`` 实例，或等价的 dict
        field_names: 字段顺序；默认 ``SCHEMA_V2_FIELD_NAMES``（14 维）

    返回:
        与 ``field_names`` 等长的 float 列表
    """
    names = field_names if field_names is not None else SCHEMA_V2_FIELD_NAMES
    return [encode_field(name, _read_field(sv2, name)) for name in names]
