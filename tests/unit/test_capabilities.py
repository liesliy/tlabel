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
    capabilities_subset,
    filter_capabilities,
    validate_capabilities,
    warn_if_invalid_capabilities,
)
from tlabel.core.types import TLabelData

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "tlabel-schema.json"


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
