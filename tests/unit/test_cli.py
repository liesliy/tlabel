"""
Tests for tlabel CLI tool

v0.16.0 — validate, list, info, version commands
"""

import pytest
import json
import tempfile
from pathlib import Path

from tlabel.cli import (
    validate_tlabel_file, TLABEL_V2_DIMENSIONS,
    ValidationResult, cmd_list, cmd_version, cmd_info,
)


class TestDimensions:
    def test_22_dimensions(self):
        assert len(TLABEL_V2_DIMENSIONS) == 22

    def test_core_dims_present(self):
        assert "contact" in TLABEL_V2_DIMENSIONS
        assert "deformation_magnitude" in TLABEL_V2_DIMENSIONS
        assert "slip_event" in TLABEL_V2_DIMENSIONS


class TestValidationResult:
    def test_error_repr(self):
        r = ValidationResult("error", "missing field", "frames[0].tlabel_v2")
        s = repr(r)
        assert "❌" in s
        assert "frames[0].tlabel_v2" in s

    def test_warning_repr(self):
        r = ValidationResult("warning", "low coverage")
        s = repr(r)
        assert "⚠️" in s


class TestValidateJSON:
    def test_valid_file(self, tmp_path):
        data = {
            "schema_version": "0.16.0",
            "format": "tlabel_v2",
            "sensor": {"type": "test"},
            "capabilities": {"contact": True, "deformation_magnitude": True},
            "frames": [
                {
                    "frame_idx": 0,
                    "timestamp_s": 0.0,
                    "tlabel_v2": {
                        "contact": 1.0,
                        "deformation_magnitude": 0.5,
                        "slip_event": 0.0,
                    }
                }
            ]
        }
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))

        results = validate_tlabel_file(str(f))
        errors = [r for r in results if r.level == "error"]
        assert len(errors) == 0

    def test_missing_frames(self, tmp_path):
        data = {"schema_version": "0.16.0", "format": "tlabel_v2"}
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(data))

        results = validate_tlabel_file(str(f))
        errors = [r for r in results if r.level == "error"]
        assert any("frames" in r.message for r in errors)

    def test_missing_tlabel_v2_in_frame(self, tmp_path):
        data = {
            "schema_version": "0.16.0",
            "format": "tlabel_v2",
            "frames": [{"frame_idx": 0, "timestamp_s": 0.0}]
        }
        f = tmp_path / "bad2.json"
        f.write_text(json.dumps(data))

        results = validate_tlabel_file(str(f))
        errors = [r for r in results if r.level == "error"]
        assert any("tlabel_v2" in r.message for r in errors)

    def test_nonexistent_file(self):
        results = validate_tlabel_file("/nonexistent/path.json")
        errors = [r for r in results if r.level == "error"]
        assert len(errors) == 1
        assert "不存在" in errors[0].message

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "broken.json"
        f.write_text("{not valid json}")

        results = validate_tlabel_file(str(f))
        errors = [r for r in results if r.level == "error"]
        assert any("JSON" in r.message for r in errors)


class TestValidateDirectory:
    def test_directory_validation(self, tmp_path):
        # Create a simple directory structure
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{}")
        
        results = validate_tlabel_file(str(tmp_path))
        infos = [r for r in results if r.level == "info"]
        assert any("目录" in r.message for r in infos)


class TestCLICommands:
    def test_version(self):
        """Version command should print without error"""
        class Args:
            pass
        result = cmd_version(Args())
        assert result == 0

    def test_list(self, capsys):
        """List command should output adapter info"""
        class Args:
            pass
        result = cmd_list(Args())
        captured = capsys.readouterr()
        assert "适配器" in captured.out
        assert result == 0

    def test_info_existing(self, capsys):
        """Info for existing adapter should show details"""
        class Args:
            name = "gelsight"
        result = cmd_info(Args())
        captured = capsys.readouterr()
        assert "gelsight" in captured.out
        assert result == 0

    def test_info_nonexistent(self, capsys):
        """Info for non-existent adapter should return error code"""
        class Args:
            name = "nonexistent_sensor_xyz"
        result = cmd_info(Args())
        assert result == 1


class TestExportFormatAliases:
    """CLI 格式名与 registry exporter id 的映射契约。

    背景：registry 里的 ``lerobot`` 指"写回"，但 CLI 自发布以来 ``-f lerobot``
    一直是"新建"。两者必须显式区分，且错误提示只能展示 CLI 名字——否则用户
    照着提示输入 ``-f lerobot`` 会得到和提示暗示相反的行为。
    """

    def test_lerobot_maps_to_create(self):
        from tlabel.cli import _resolve_export_format
        assert _resolve_export_format("lerobot") == "lerobot_create"

    def test_lerobot_writeback_maps_to_registry_lerobot(self):
        from tlabel.cli import _resolve_export_format
        assert _resolve_export_format("lerobot_writeback") == "lerobot"

    def test_other_formats_pass_through(self):
        from tlabel.cli import _resolve_export_format
        for fmt in ("json", "csv", "hdf5", "ftp1", "rlds", "ros2"):
            assert _resolve_export_format(fmt) == fmt

    def test_error_hint_uses_cli_names_not_internal_ids(self):
        from tlabel.cli import _cli_format_names
        from tlabel.export.registry import get_registry

        names = _cli_format_names(get_registry())
        # 内部 id 不得泄漏到提示里
        assert "lerobot_create" not in names
        # 写回必须以 CLI 名字出现，否则用户无从发现
        assert "lerobot_writeback" in names
        assert "lerobot" in names
        # 别名不得造成重复项
        assert len(names) == len(set(names))

    def test_error_hint_is_sorted_and_complete(self):
        from tlabel.cli import _cli_format_names
        from tlabel.export.registry import get_registry

        names = _cli_format_names(get_registry())
        assert names == sorted(names)
        assert {"json", "csv", "hdf5", "lerobot", "lerobot_writeback"} <= set(names)


class TestValidateFormatTags:
    """validate 必须接受本工具自己导出的 format 标记（v0.24.2）。

    导出器 ``core/types.py`` 的 ``to_dict`` 写 ``format: tlabel_schema_v2``，
    而 validate 原先只认旧 22 维的 ``tlabel_v2``，于是用户导出完顺手校验，
    第一眼就是一个黄色警告——文件本身没问题，是校验器认错了名字。
    """

    def test_exported_json_has_no_format_warning(self, tmp_path):
        import tlabel

        out = tmp_path / "g.json"
        tlabel.demo("gelsight").export(str(out), format="json")

        results = validate_tlabel_file(str(out))
        fmt_problems = [r for r in results if r.field == "format"]
        assert fmt_problems == [], [repr(r) for r in fmt_problems]

    def test_legacy_tag_still_accepted(self, tmp_path):
        data = {
            "schema_version": "0.16.0",
            "format": "tlabel_v2",
            "sensor": {"type": "test"},
            "capabilities": {"contact": True},
            "frames": [
                {"frame_idx": 0, "timestamp_s": 0.0,
                 "tlabel_v2": {"contact": 1.0}}
            ],
        }
        f = tmp_path / "legacy.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        results = validate_tlabel_file(str(f))
        assert [r for r in results if r.field == "format"] == []

    def test_unknown_tag_still_warns(self, tmp_path):
        data = {
            "schema_version": "0.16.0",
            "format": "something_else",
            "sensor": {"type": "test"},
            "capabilities": {"contact": True},
            "frames": [
                {"frame_idx": 0, "timestamp_s": 0.0,
                 "tlabel_v2": {"contact": 1.0}}
            ],
        }
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        warns = [r for r in validate_tlabel_file(str(f))
                 if r.field == "format" and r.level == "warning"]
        assert warns, "未知 format 标记仍应告警"


class TestCliVersionFlag:
    """`--version` 是 CLI 通用约定，不能只有 `version` 子命令。"""

    def test_version_flag_exits_zero(self, capsys, monkeypatch):
        from tlabel.cli import main
        from tlabel._version import __version__

        monkeypatch.setattr("sys.argv", ["tlabel", "--version"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_version_subcommand_still_works(self, capsys):
        from tlabel.cli import cmd_version
        from tlabel._version import __version__

        assert cmd_version(None) == 0
        assert __version__ in capsys.readouterr().out
