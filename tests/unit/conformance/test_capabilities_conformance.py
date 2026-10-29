"""
Capabilities 合规守护测试

钉住三件事，防止后续改动把非法能力键又带回来：

1. 每个已注册适配器声明的 capabilities 必须是 13 键集合的子集、值为 bool、
   含必填的 contact、不含 compliance_level；
2. 适配器的 get_capabilities() 必须能无参调用（基类契约）；
3. TLabelData 导出的 dict 里 capabilities 过 schema 校验。

规范依据：S-6.3-1 / S-6.3-3 / S-7.4-1 / S-7.4-2 / C-15。
"""

import json
from pathlib import Path

import pytest

from tlabel.core.capabilities import (
    CAPABILITY_KEYS,
    FORBIDDEN_CAPABILITY_KEYS,
    REQUIRED_CAPABILITY_KEYS,
    validate_capabilities,
)
from tlabel.core.registry import get_adapter, list_adapters, _ensure_adapters
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2


SCHEMA_PATH = Path(__file__).parent.parent.parent.parent / "schema" / "tlabel-schema.json"


def _all_adapter_names():
    """所有已注册适配器名称（排除预留占位）。"""
    _ensure_adapters()
    return sorted(list_adapters())


@pytest.fixture(scope="module")
def adapter_names():
    return _all_adapter_names()


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


# ============================================================
# 1. 全适配器 capabilities 合规
# ============================================================

def test_adapter_list_not_empty(adapter_names):
    """守护测试自身的前提：适配器注册表不能是空的。"""
    assert len(adapter_names) >= 10, f"只发现 {len(adapter_names)} 个适配器"


@pytest.mark.parametrize("name", _all_adapter_names())
def test_capabilities_keys_are_legal(name):
    """S-6.3-1 / S-7.4-1：键集合 ⊆ 13 键，值必须 bool。"""
    adapter = get_adapter(name)()
    caps = adapter.get_capabilities()

    assert isinstance(caps, dict), f"{name}: get_capabilities() 必须返回 dict"

    illegal = sorted(set(caps) - CAPABILITY_KEYS)
    assert not illegal, (
        f"{name}: 存在非法能力键 {illegal}。"
        f"合法键共 {len(CAPABILITY_KEYS)} 个，见 tlabel/core/capabilities.py"
    )

    non_bool = sorted(k for k, v in caps.items() if not isinstance(v, bool))
    assert not non_bool, f"{name}: 能力值必须是 bool，以下键不满足 {non_bool}"


@pytest.mark.parametrize("name", _all_adapter_names())
def test_capabilities_has_required_contact(name):
    """S-7.4-1：contact 是唯一必填能力键。"""
    caps = get_adapter(name)().get_capabilities()
    assert REQUIRED_CAPABILITY_KEYS <= set(caps), (
        f"{name}: 缺少必填能力键 {sorted(REQUIRED_CAPABILITY_KEYS - set(caps))}"
    )


@pytest.mark.parametrize("name", _all_adapter_names())
def test_capabilities_excludes_compliance_level(name):
    """S-7.4-2：compliance_level 是帧级元字段，不允许出现在 capabilities。"""
    caps = get_adapter(name)().get_capabilities()
    assert not (FORBIDDEN_CAPABILITY_KEYS & set(caps)), (
        f"{name}: capabilities 不允许包含 {sorted(FORBIDDEN_CAPABILITY_KEYS)}"
    )


@pytest.mark.parametrize("name", _all_adapter_names())
def test_capabilities_pass_validator(name):
    """validate_capabilities() 必须零问题——这是 validate 命令的同一口径。"""
    caps = get_adapter(name)().get_capabilities()
    problems = validate_capabilities(caps)
    assert not problems, f"{name}: " + "；".join(problems)


@pytest.mark.parametrize("name", _all_adapter_names())
def test_get_capabilities_callable_without_args(name):
    """基类契约：get_capabilities() 必须无参可调。

    TLabelAdapter 早期版本要求 file_path 参数，导致所有遍历型调用方
    只能靠 TypeError 跳过。此处钉死无参可调。
    """
    adapter = get_adapter(name)()
    caps = adapter.get_capabilities()  # 不带任何参数
    assert isinstance(caps, dict)


# ============================================================
# 2. 出口校验：TLabelData → dict → schema
# ============================================================

def test_exported_capabilities_match_schema(schema):
    """导出的 capabilities 必须过 schema 的 additionalProperties/required 约束。"""
    caps_schema = schema["properties"]["capabilities"]
    allowed = set(caps_schema["properties"])

    caps = {"contact": True, "force_magnitude": True, "confidence": True}
    data = TLabelData(
        frames=[TLabelFrame(frame_idx=0, timestamp_s=0.0,
                            schema_v2=TLabelSchemaV2(contact=True))],
        sensor_info={"type": "test"},
        episode_info={},
        capabilities=caps,
    )
    out = data.to_dict()

    assert set(out["capabilities"]) <= allowed
    assert not (set(out["capabilities"]) - CAPABILITY_KEYS)


def test_schema_capability_keys_match_python_constant(schema):
    """schema JSON 与 Python 常量必须同源，避免两边漂移。"""
    schema_keys = set(schema["properties"]["capabilities"]["properties"])
    assert schema_keys == set(CAPABILITY_KEYS), (
        f"schema 与 CAPABILITY_KEYS 不一致："
        f"仅 JSON 有 {sorted(schema_keys - set(CAPABILITY_KEYS))}，"
        f"仅 Python 有 {sorted(set(CAPABILITY_KEYS) - schema_keys)}"
    )
