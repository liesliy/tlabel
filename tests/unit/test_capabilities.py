"""
capabilities 合规性测试（PIT-7）

``schema/tlabel-schema.json`` 把能力键集合钉死为 13 键、``additionalProperties: false``、
``required: ["contact"]``、值必须 bool；``compliance_level`` 是帧级元字段，
不得出现在 capabilities 中（规范 S-7.4-2 / C-15）。

本测试锁定 v0.23.3 起的校验行为与已清理的脏源。
"""

import json
import warnings
from pathlib import Path

import pytest

from tlabel.core.capabilities import (
    CAPABILITY_KEYS,
    FORBIDDEN_CAPABILITY_KEYS,
    REQUIRED_CAPABILITY_KEYS,
    STRICT_ENV_VAR,
    capabilities_subset,
    filter_capabilities,
    normalize_capabilities,
    strict_enabled,
    validate_capabilities,
    warn_if_invalid_capabilities,
)
from tlabel.core.types import TLabelData

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "tlabel-schema.json"


def _list_demos() -> list:
    """延迟导入：``tlabel`` 顶层导入会拉起适配器注册，放在模块级太重。"""
    import tlabel

    return tlabel.list_demos()


def _demo(name: str):
    import tlabel

    return tlabel.demo(name)


# ================================================================
# 常量与 schema 一致性
# ================================================================

class TestCapabilityConstants:
    def test_schema_file_exists(self):
        assert SCHEMA_PATH.exists(), f"未找到 schema 文件: {SCHEMA_PATH}"

    def test_keys_match_json_schema(self):
        """Python 常量必须与 JSON Schema 的 capabilities.properties 完全一致。"""
        schema = json.loads(SCHEMA_PATH.read_text())
        schema_keys = set(schema["properties"]["capabilities"]["properties"])
        assert CAPABILITY_KEYS == schema_keys, (
            f"常量与 schema 不一致：多 {CAPABILITY_KEYS - schema_keys}，"
            f"缺 {schema_keys - CAPABILITY_KEYS}"
        )

    def test_key_count_is_13(self):
        assert len(CAPABILITY_KEYS) == 13

    def test_compliance_level_is_forbidden(self):
        assert "compliance_level" in FORBIDDEN_CAPABILITY_KEYS
        assert "compliance_level" not in CAPABILITY_KEYS

    def test_contact_is_required(self):
        assert REQUIRED_CAPABILITY_KEYS == {"contact"}

    def test_schema_declares_additional_properties_false(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        assert schema["properties"]["capabilities"]["additionalProperties"] is False


# ================================================================
# validate_capabilities
# ================================================================

class TestValidateCapabilities:
    def test_clean_capabilities_pass(self):
        assert validate_capabilities({"contact": True, "confidence": True}) == []

    def test_all_13_keys_pass(self):
        caps = {k: True for k in CAPABILITY_KEYS}
        assert validate_capabilities(caps) == []

    def test_legacy_key_rejected(self):
        problems = validate_capabilities({"contact": True, "deformation_magnitude": True})
        assert any("deformation_magnitude" in p for p in problems)

    def test_compliance_level_rejected(self):
        problems = validate_capabilities({"contact": True, "compliance_level": True})
        assert any("compliance_level" in p for p in problems)

    def test_missing_contact_rejected(self):
        problems = validate_capabilities({"confidence": True})
        assert any("contact" in p for p in problems)

    def test_non_bool_value_rejected(self):
        problems = validate_capabilities({"contact": 1})
        assert any("bool" in p for p in problems)

    def test_empty_dict_rejected(self):
        """空 capabilities 缺必填 contact。"""
        assert validate_capabilities({}) != []


# ================================================================
# filter_capabilities
# ================================================================

class TestFilterCapabilities:
    def test_drops_unknown_keys(self):
        out = filter_capabilities({"contact": True, "bogus_feature": True})
        assert out == {"contact": True}

    def test_drops_compliance_level(self):
        out = filter_capabilities({"contact": True, "compliance_level": True})
        assert "compliance_level" not in out

    def test_coerces_to_bool(self):
        out = filter_capabilities({"contact": 1, "confidence": 0})
        assert out == {"contact": True, "confidence": False}

    def test_empty_input(self):
        assert filter_capabilities({}) == {}

    def test_capabilities_subset(self):
        out = capabilities_subset(["contact", "force_vector", "not_a_key"])
        assert out == {"contact": True, "force_vector": True}


# ================================================================
# TLabelData 出口告警
# ================================================================

class TestTLabelDataWarning:
    def _make_data(self, caps):
        return TLabelData(
            frames=[],
            sensor_info={"type": "test"},
            episode_info={},
            capabilities=caps,
        )

    def test_clean_capabilities_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._make_data({"contact": True, "confidence": True})

    def test_dirty_capabilities_warns(self):
        with pytest.warns(UserWarning, match="capabilities"):
            self._make_data({"contact": True, "deformation_magnitude": True})

    def test_warning_does_not_mutate_capabilities(self):
        """v0.23.3 只告警，不改写输出（归一化留到 0.24.0）。"""
        dirty = {"contact": True, "deformation_magnitude": True}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self._make_data(dict(dirty))
        assert data.capabilities == dirty


# ================================================================
# 已清理的脏源回归
# ================================================================

class TestDirtySourcesCleaned:
    def test_xela_compliant(self):
        from tlabel.adapters.xela import XelaUskinAdapter

        caps = XelaUskinAdapter().get_capabilities()
        assert set(caps) == CAPABILITY_KEYS

    @pytest.mark.parametrize("module,cls_name", [
        ("tlabel.adapters.syntouch", "SynTouchBioTacAdapter"),
        ("tlabel.adapters.tashan_ts_f_a", "TashanTsFAAdapter"),
        ("tlabel.adapters.paxini_px6d", "PaxiniPX6DAdapter"),
    ])
    def test_compliance_level_removed(self, module, cls_name):
        import importlib

        cls = getattr(importlib.import_module(module), cls_name)
        caps = cls().get_capabilities()
        assert "compliance_level" not in caps, f"{cls_name} 仍输出 compliance_level"
        assert validate_capabilities(caps) == [], f"{cls_name} 仍不合规"

    def test_demo_capabilities_clean(self):
        """demo.py 的合成 caps 不得含 compliance_level。"""
        from tlabel.demo import demo

        # daimon_dm_tac 走 _build_realtime_sensor_demo 的合成 caps 分支
        data = demo("daimon_dm_tac")
        assert "compliance_level" not in data.capabilities
        assert validate_capabilities(data.capabilities) == []

    def test_lerobot_capabilities_filtered(self, tmp_path):
        """LeRobot feature 名不在 13 键内时，不得泄漏为能力键。"""
        pyarrow = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq

        from tlabel.converters.lerobot import lerobot_to_tlabel

        (tmp_path / "data").mkdir()
        (tmp_path / "meta").mkdir()
        table = pyarrow.table({
            "frame_index": [0, 1],
            "observation.tactile": [[0.0] * 14] * 2,
            "some_random_feature": [1.0, 2.0],
        })
        pq.write_table(table, tmp_path / "data" / "chunk-0000.parquet")
        (tmp_path / "meta" / "info.json").write_text(
            json.dumps({"robot_type": "test", "features": {}})
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = lerobot_to_tlabel(str(tmp_path))

        assert "some_random_feature" not in data.capabilities
        assert set(data.capabilities) <= CAPABILITY_KEYS


# ================================================================
# 出口归一化与严格模式（S3）
# ================================================================

class TestNormalizeCapabilities:
    """出口归一化：剔除非法键、补必填键、严格模式抛错。"""

    def test_drops_illegal_keys_and_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = normalize_capabilities(
                {"contact": True, "has_image": True, "dimensions": 22},
                source="test",
            )
        assert out == {"contact": True}
        assert any("非法键" in str(w.message) for w in caught)

    def test_adds_missing_contact(self):
        assert normalize_capabilities({"force_magnitude": True}) == {
            "force_magnitude": True,
            "contact": False,
        }

    def test_returns_legal_subset_always(self):
        out = normalize_capabilities(
            {"compliance_level": "L2", "bogus": 1, "contact": True, "slip_event": True}
        )
        assert set(out) <= CAPABILITY_KEYS
        assert "compliance_level" not in out

    def test_non_mapping_degrades_to_empty_declaration(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert normalize_capabilities(None) == {"contact": False}

    def test_legal_input_is_unchanged_in_strict_mode(self):
        caps = {"contact": True, "force_magnitude": False}
        assert normalize_capabilities(caps, strict=True) == caps

    def test_strict_raises_on_illegal_key(self):
        with pytest.raises(ValueError, match="非法能力键"):
            normalize_capabilities({"has_image": True}, source="test", strict=True)

    def test_strict_raises_on_missing_contact(self):
        with pytest.raises(ValueError, match="缺少必填能力键"):
            normalize_capabilities({"force_magnitude": True}, strict=True)

    def test_strict_raises_on_non_bool_value(self):
        with pytest.raises(ValueError, match="必须是 bool"):
            normalize_capabilities({"contact": "yes"}, strict=True)

    def test_strict_via_environment_variable(self, monkeypatch):
        monkeypatch.setenv(STRICT_ENV_VAR, "1")
        assert strict_enabled() is True
        with pytest.raises(ValueError):
            normalize_capabilities({"has_image": True})

    def test_strict_off_by_default(self, monkeypatch):
        monkeypatch.delenv(STRICT_ENV_VAR, raising=False)
        assert strict_enabled() is False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert normalize_capabilities({"has_image": True}) == {"contact": False}

    def test_explicit_strict_overrides_environment(self, monkeypatch):
        monkeypatch.setenv(STRICT_ENV_VAR, "0")
        with pytest.raises(ValueError):
            normalize_capabilities({"has_image": True}, strict=True)


class TestExitsNormalize:
    """所有出口都必须产出合法 capabilities，旧数据不得泄漏非法键。"""

    def _dirty_data(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TLabelData(
                frames=[],
                sensor_info={"type": "test"},
                episode_info={},
                capabilities={"has_image": True, "dimensions": 22, "contact": True},
            )

    def test_to_dict_outputs_legal_capabilities(self):
        payload = self._dirty_data().to_dict()
        assert set(payload["capabilities"]) <= CAPABILITY_KEYS
        assert payload["capabilities"]["contact"] is True
        assert "has_image" not in payload["capabilities"]

    def test_to_dict_keeps_original_capabilities_attribute(self):
        """出口归一化不得篡改对象本身，避免影响后续逻辑。"""
        data = self._dirty_data()
        assert "has_image" in data.capabilities


class TestDemoCapabilitiesMatchOutput:
    """内置 demo 的 capabilities 必须覆盖帧里实际有值的维度。

    v0.24.2 之前，``demo_data/*.json`` 携带的是 22 维时代的旧 caps
    （``has_image`` / ``dimensions``），与 13 个能力键无交集，归一化后为空集，
    于是退化成"只声明 contact"。但帧里 force_magnitude / slip_event 等有真实
    数据，导出文件的 capabilities 与实际输出自相矛盾（S-6.3-3），下游按 caps
    判断字段是否存在会整列漏读。这里用守护测试钉死"声明即输出"。
    """

    @staticmethod
    def _actual_output_keys(data) -> set:
        from tlabel.demo import _value_has_signal

        actual = set()
        for frame in data.frames:
            schema = frame.schema_v2
            for key in CAPABILITY_KEYS:
                value = getattr(schema, key, None)
                if value is None:
                    continue
                if isinstance(value, bool) or _value_has_signal(value):
                    actual.add(key)
        return actual

    @pytest.mark.parametrize("name", _list_demos())
    def test_declared_caps_cover_actual_output(self, name):
        data = _demo(name)
        declared = {k for k, v in (data.capabilities or {}).items() if v}
        actual = self._actual_output_keys(data)
        assert actual <= declared, (
            f"{name} 声明缺失: {sorted(actual - declared)}"
            f"（声明 {sorted(declared)}）"
        )

    @pytest.mark.parametrize("name", _list_demos())
    def test_caps_are_legal_keys(self, name):
        caps = _demo(name).capabilities or {}
        assert set(caps) <= CAPABILITY_KEYS
        assert caps.get("contact") is not None

    def test_legacy_caps_preserved(self):
        """旧 22 键 caps 仍要原样留档，不能因为反推声明而丢信息。"""
        data = _demo("gelsight")
        legacy = (data.episode_info or {}).get("legacy_capabilities")
        assert legacy, "legacy_capabilities 应保留旧 caps"
        assert "has_force" in legacy or "dimensions" in legacy

    def test_gelsight_declares_force_and_slip(self):
        """回归锚点：gelsight demo 有 106 帧力数据、17 帧滑移，必须声明。"""
        caps = _demo("gelsight").capabilities
        assert caps.get("force_magnitude") is True
        assert caps.get("slip_event") is True
