"""
Tests for TLabel scenario templates (v0.24.0)

覆盖：
  - 4 个场景模板都能加载
  - validate --all 全部通过
  - 非法模板（引用不存在的字段名 / primitive）能被检出
  - CLI 两个子命令（template list / template validate）可用
"""

import yaml
import pytest

from tlabel.scenarios import (
    ScenarioTemplate,
    list_scenarios,
    get_scenario,
    load_all_scenarios,
    validate_scenario,
    validate_scenario_file,
    validate_all_scenarios,
    validate_template_data,
    VALID_SCHEMA_FIELDS,
    VALID_PRIMITIVES,
    VALID_EVENTS,
)
from tlabel.cli import cmd_template_list, cmd_template_validate


EXPECTED_IDS = [
    "gelsight_grasp",
    "paxini_force_control",
    "tashan_fingertip",
    "xela_uskin",
]


class TestScenarioLoading:
    def test_four_scenarios_present(self):
        assert list_scenarios() == EXPECTED_IDS

    def test_load_all(self):
        templates = load_all_scenarios()
        assert set(templates.keys()) == set(EXPECTED_IDS)
        for sid, tpl in templates.items():
            assert isinstance(tpl, ScenarioTemplate)
            assert tpl.id == sid
            assert tpl.name
            assert tpl.description
            assert tpl.adapters
            assert tpl.schema_fields
            assert tpl.primitives
            assert tpl.events
            assert tpl.min_compliance_level in ("L1", "L2", "L3", "L4")

    def test_get_scenario_fields_in_schema(self):
        for sid in EXPECTED_IDS:
            tpl = get_scenario(sid)
            for fld in tpl.schema_fields:
                assert fld in VALID_SCHEMA_FIELDS, f"{sid}: {fld}"
            for p in tpl.primitives:
                assert p in VALID_PRIMITIVES, f"{sid}: {p}"
            for ev in tpl.events:
                assert ev in VALID_EVENTS, f"{sid}: {ev}"

    def test_get_missing_scenario_raises(self):
        with pytest.raises(FileNotFoundError):
            get_scenario("does_not_exist")

    def test_to_dict_roundtrip_keys(self):
        tpl = get_scenario("gelsight_grasp")
        d = tpl.to_dict()
        assert d["id"] == "gelsight_grasp"
        assert d["min_compliance_level"] == "L2"
        assert set(d.keys()) == {
            "id", "name", "description", "adapters", "sensors",
            "schema_fields", "primitives", "events",
            "min_compliance_level", "constraints",
        }

    def test_expected_sensor_mapping(self):
        """每个场景的适配器与最低合规等级与适配器代码一致"""
        expected = {
            "gelsight_grasp": (["gelsight"], "L2"),
            "paxini_force_control": (["paxini_gen3"], "L2"),
            "tashan_fingertip": (["tashan_ts_f_a"], "L3"),
            "xela_uskin": (["xela"], "L1"),
        }
        for sid, (adapters, level) in expected.items():
            tpl = get_scenario(sid)
            assert tpl.adapters == adapters
            assert tpl.min_compliance_level == level


class TestValidateAll:
    def test_validate_all_pass(self):
        results = validate_all_scenarios()
        assert set(results.keys()) == set(EXPECTED_IDS)
        for sid, (ok, errors) in results.items():
            assert ok, f"{sid} 未通过: {errors}"

    def test_validate_single_pass(self):
        ok, errors = validate_scenario("gelsight_grasp")
        assert ok
        assert errors == []


class TestInvalidTemplates:
    def _write(self, tmp_path, name, data):
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return p

    def _base(self):
        return {
            "id": "demo",
            "name": "Demo",
            "description": "demo scenario",
            "adapters": ["gelsight"],
            "schema_fields": ["contact", "force_magnitude"],
            "primitives": ["grasp", "lift"],
            "events": ["contact_onset"],
            "min_compliance_level": "L2",
        }

    def test_unknown_schema_field_detected(self, tmp_path):
        data = self._base()
        data["schema_fields"] = ["contact", "not_a_real_field"]
        p = self._write(tmp_path, "demo", data)
        ok, errors = validate_scenario_file(str(p))
        assert not ok
        assert any("not_a_real_field" in e for e in errors)

    def test_unknown_primitive_detected(self, tmp_path):
        data = self._base()
        data["primitives"] = ["grasp", "teleport"]
        p = self._write(tmp_path, "demo", data)
        ok, errors = validate_scenario_file(str(p))
        assert not ok
        assert any("teleport" in e for e in errors)

    def test_unknown_event_detected(self, tmp_path):
        data = self._base()
        data["events"] = ["contact_onset", "explode"]
        p = self._write(tmp_path, "demo", data)
        ok, errors = validate_scenario_file(str(p))
        assert not ok
        assert any("explode" in e for e in errors)

    def test_unknown_compliance_level_detected(self, tmp_path):
        data = self._base()
        data["min_compliance_level"] = "L9"
        p = self._write(tmp_path, "demo", data)
        ok, errors = validate_scenario_file(str(p))
        assert not ok
        assert any("L9" in e for e in errors)

    def test_unknown_adapter_detected(self, tmp_path):
        data = self._base()
        data["adapters"] = ["no_such_adapter_xyz"]
        p = self._write(tmp_path, "demo", data)
        ok, errors = validate_scenario_file(str(p))
        assert not ok
        assert any("no_such_adapter_xyz" in e for e in errors)

    def test_id_mismatch_detected(self, tmp_path):
        data = self._base()
        data["id"] = "other_name"
        p = self._write(tmp_path, "demo", data)
        ok, errors = validate_scenario_file(str(p))
        assert not ok
        assert any("不一致" in e for e in errors)

    def test_missing_required_key_detected(self):
        data = self._base()
        del data["primitives"]
        errors = validate_template_data(data, expected_id="demo")
        assert errors
        assert any("primitives" in e for e in errors)

    def test_unknown_constraint_field_detected(self, tmp_path):
        data = self._base()
        data["constraints"] = {"required_when_contact": ["bogus_field"]}
        p = self._write(tmp_path, "demo", data)
        ok, errors = validate_scenario_file(str(p))
        assert not ok
        assert any("bogus_field" in e for e in errors)

    def test_unknown_constraint_key_detected(self, tmp_path):
        data = self._base()
        data["constraints"] = {"when_contact": ["contact_centroid"]}
        p = self._write(tmp_path, "demo", data)
        ok, errors = validate_scenario_file(str(p))
        assert not ok
        assert any("when_contact" in e for e in errors)

    def test_valid_temp_template_passes(self, tmp_path):
        data = self._base()
        p = self._write(tmp_path, "demo", data)
        ok, errors = validate_scenario_file(str(p))
        assert ok, errors


class TestTemplateCLI:
    def test_template_list(self, capsys):
        class Args:
            pass

        result = cmd_template_list(Args())
        captured = capsys.readouterr()
        assert result == 0
        for sid in EXPECTED_IDS:
            assert sid in captured.out
        assert "场景模板" in captured.out

    def test_template_validate_all(self, capsys):
        class Args:
            id = None
            all = True

        result = cmd_template_validate(Args())
        captured = capsys.readouterr()
        assert result == 0
        assert "4 通过" in captured.out

    def test_template_validate_single(self, capsys):
        class Args:
            id = "gelsight_grasp"
            all = False

        result = cmd_template_validate(Args())
        captured = capsys.readouterr()
        assert result == 0
        assert "gelsight_grasp" in captured.out

    def test_template_validate_unknown(self, capsys):
        class Args:
            id = "nope_xyz"
            all = False

        result = cmd_template_validate(Args())
        captured = capsys.readouterr()
        assert result == 1
        assert "nope_xyz" in captured.out

    def test_template_validate_no_target(self, capsys):
        class Args:
            id = None
            all = False

        result = cmd_template_validate(Args())
        assert result == 1
