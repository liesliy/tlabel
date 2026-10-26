"""
TLabel capabilities 能力声明的单一事实源与合规校验

``schema/tlabel-schema.json`` 的 ``properties.capabilities`` 把能力键集合钉死为
13 个键，且 ``additionalProperties: false``、``required: ["contact"]``、
所有值必须是 boolean。规范依据见 ``docs/spec/tlabel-spec-v1.md``：

  - S-6.3-1  返回的键必须是 13 键集合的子集，值为 bool
  - S-6.3-2  声明 false 的维度每帧输出 null
  - S-6.3-3  不得声明做不到的能力，也不得填充声明为 false 的维度
  - S-7.4-1  13 键 + contact 必填
  - S-7.4-2  compliance_level 禁止出现在 capabilities 中（它是帧级元字段）
  - C-15     conformance 要求键集合 ⊆ 13 键且无 compliance_level

v0.23.3 起本模块是这些常量的唯一 Python 事实源，测试与校验器都应复用，
不要在别处重复手写键名列表。
"""

import warnings
from typing import Dict, Iterable, List, Mapping

# ================================================================
# 常量
# ================================================================

#: 合法能力键（恰好 13 个，注意不是 14 维；compliance_level 不属于能力键）
CAPABILITY_KEYS = frozenset({
    "confidence",
    "contact",
    "contact_centroid",
    "contact_region",
    "force_magnitude",
    "force_vector",
    "manipulation_phase",
    "object_deformation",
    "slip_event",
    "slip_velocity",
    "temperature",
    "texture_class",
    "torque_vector",
})

#: 唯一必填的能力键
REQUIRED_CAPABILITY_KEYS = frozenset({"contact"})

#: 帧级元字段，不允许出现在 capabilities 中
FORBIDDEN_CAPABILITY_KEYS = frozenset({"compliance_level"})


# ================================================================
# 校验
# ================================================================

def validate_capabilities(capabilities: Mapping) -> List[str]:
    """检查 capabilities 是否满足 schema 约束。

    返回问题描述列表，空列表表示合规。不抛异常、不修改入参。
    """
    problems: List[str] = []

    if not isinstance(capabilities, Mapping):
        return [f"capabilities 必须是 mapping，实际为 {type(capabilities).__name__}"]

    keys = set(capabilities)

    unknown = keys - CAPABILITY_KEYS
    if unknown:
        problems.append(
            "存在非法能力键 " + ", ".join(sorted(unknown))
            + f"（合法键共 {len(CAPABILITY_KEYS)} 个，见 schema/tlabel-schema.json）"
        )

    missing = REQUIRED_CAPABILITY_KEYS - keys
    if missing:
        problems.append("缺少必填能力键 " + ", ".join(sorted(missing)))

    non_bool = sorted(k for k, v in capabilities.items() if not isinstance(v, bool))
    if non_bool:
        problems.append("能力值必须是 bool，以下键不满足：" + ", ".join(non_bool))

    return problems


def filter_capabilities(capabilities: Mapping) -> Dict[str, bool]:
    """白名单过滤：只保留 13 个合法键，值强制转为 bool。

    用于"出口兜底"——把外部传入的任意字典（例如 LeRobot 的 feature 名）
    收敛为 schema 合法集合。不做补全，缺失的 contact 仍由调用方负责。
    """
    if not isinstance(capabilities, Mapping):
        return {}
    return {k: bool(v) for k, v in capabilities.items() if k in CAPABILITY_KEYS}


def warn_if_invalid_capabilities(capabilities: Mapping, source: str = "") -> List[str]:
    """校验并在不合规时发 ``UserWarning``，返回问题列表。"""
    problems = validate_capabilities(capabilities)
    if problems:
        prefix = f"[tlabel] capabilities 不合规（{source}）：" if source \
            else "[tlabel] capabilities 不合规："
        warnings.warn(prefix + "；".join(problems), UserWarning, stacklevel=3)
    return problems


def capabilities_subset(keys: Iterable[str]) -> Dict[str, bool]:
    """把一组字段名收敛为合法能力键（全部声明为 True）。"""
    return {k: True for k in keys if k in CAPABILITY_KEYS}


__all__ = [
    "CAPABILITY_KEYS",
    "REQUIRED_CAPABILITY_KEYS",
    "FORBIDDEN_CAPABILITY_KEYS",
    "validate_capabilities",
    "filter_capabilities",
    "warn_if_invalid_capabilities",
    "capabilities_subset",
]
