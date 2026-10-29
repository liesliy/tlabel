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

import os
import warnings
from typing import Dict, Iterable, List, Mapping, Optional

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


# ================================================================
# 出口归一化
# ================================================================

#: 开启严格模式的环境变量名。设为 1/true/yes/on 后，出口遇到不合规
#: capabilities 直接抛 ValueError 而不是告警。
STRICT_ENV_VAR = "TLABEL_STRICT_CAPABILITIES"


def strict_enabled(strict: Optional[bool] = None) -> bool:
    """判断是否启用严格模式；``strict`` 显式传入时优先于环境变量。"""
    if strict is not None:
        return bool(strict)
    return os.environ.get(STRICT_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_capabilities(
    capabilities: Mapping,
    source: str = "",
    *,
    strict: Optional[bool] = None,
    warn: bool = True,
) -> Dict[str, bool]:
    """出口归一化：把任意 capabilities 收敛成 schema 合法字典。

    做三件事：白名单剔除非法键、值强制转 bool、缺失的必填 ``contact`` 补 False。

    - 严格模式（``strict=True`` 或环境变量 ``TLABEL_STRICT_CAPABILITIES``）：
      发现非法键或缺失必填键时抛 ``ValueError``，不做静默修复。
    - 非严格模式（默认）：修复并告警，保证旧文件仍能读写。

    返回值总是 13 键集合的合法子集。
    """
    if not isinstance(capabilities, Mapping):
        if strict_enabled(strict):
            raise ValueError(
                f"capabilities 必须是 mapping，实际为 {type(capabilities).__name__}"
                + (f"（来源：{source}）" if source else "")
            )
        if warn:
            warnings.warn(
                f"[tlabel] capabilities 不是 mapping（{source or '未知来源'}），"
                f"已按空声明处理",
                UserWarning,
                stacklevel=3,
            )
        return {"contact": False}

    problems = validate_capabilities(capabilities)
    if problems and strict_enabled(strict):
        prefix = f"[tlabel] capabilities 不合规（{source}）：" if source \
            else "[tlabel] capabilities 不合规："
        raise ValueError(prefix + "；".join(problems))

    cleaned = filter_capabilities(capabilities)
    for key in REQUIRED_CAPABILITY_KEYS:
        cleaned.setdefault(key, False)

    # 只在真的剔除了键时才告警，避免与构造期告警重复刷屏
    dropped = sorted(set(capabilities) - CAPABILITY_KEYS)
    if dropped and warn:
        prefix = f"[tlabel] capabilities 含 {len(dropped)} 个非法键（{source}）" \
            if source else f"[tlabel] capabilities 含 {len(dropped)} 个非法键"
        warnings.warn(
            prefix + "，已在输出中剔除：" + ", ".join(dropped),
            UserWarning,
            stacklevel=3,
        )

    return cleaned


__all__ = [
    "CAPABILITY_KEYS",
    "REQUIRED_CAPABILITY_KEYS",
    "FORBIDDEN_CAPABILITY_KEYS",
    "STRICT_ENV_VAR",
    "validate_capabilities",
    "filter_capabilities",
    "warn_if_invalid_capabilities",
    "capabilities_subset",
    "normalize_capabilities",
    "strict_enabled",
]
