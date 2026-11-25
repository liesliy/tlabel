"""
V2 → V3.0 迁移测试

守两条铁律（TL-11.3）：
    TL-11.3-1  不能推断——V2 没有的信息，迁移后不能凭空出现
    TL-11.3-2  不能丢失——旧值必须保留在可追溯位置

以及一条实现原则：宁可拒绝，不产出「看起来成功、实际全 null」的数据集。
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from tlabel.v3 import validate
from tlabel.v3.migrate import MigrationSourceUnsupported, migrate_dataset

SURFACE_2D = {
    "frame_id": "surface_0",
    "dimension": 2,
    "frame_type": "sensor_local",
    "handedness": "right",
    "axes": {"x": "tangential_1", "y": "tangential_2"},
    "reference_object": "sensor_sensitive_surface",
}

SENSOR_3D = {
    "frame_id": "sensor_0",
    "dimension": 3,
    "frame_type": "sensor_local",
    "handedness": "right",
    "axes": {"x": "tangential_1", "y": "tangential_2", "z": "nominal_surface_normal"},
    "reference_object": "sensor_sensitive_surface",
}


def _v2() -> Dict[str, Any]:
    return {
        "format": "tlabel_schema_v2",
        "schema_version_v2": "2.1",
        "sensor_id": "gs01",
        "capabilities": {
            "contact": True,
            "contact_centroid": True,
            "force_magnitude": True,
            "slip_event": True,
            "manipulation_phase": True,
            "object_deformation": True,
        },
        "episode": {"episode_id": "ep_001"},
        "frames": [
            {
                "frame_idx": 0,
                "timestamp_s": 0.0,
                "schema_v2": {
                    "contact": True,
                    "slip_event": False,
                    "confidence": 0.95,
                    "compliance_level": "L2",
                    "contact_centroid": [12.4, 8.7],
                    "force_magnitude": 3.2,
                    "manipulation_phase": "grasp",
                    "object_deformation": 0.37,
                },
                "sensor_specific": {"deformation_rms": 0.37},
            },
            {
                "frame_idx": 1,
                "timestamp_s": 0.033,
                "schema_v2": {
                    "contact": True,
                    "slip_event": True,
                    "confidence": 0.88,
                    "compliance_level": "L2",
                    "contact_centroid": [13.1, 9.0],
                    "force_magnitude": 3.4,
                    "manipulation_phase": "slip",
                    "object_deformation": 0.41,
                },
            },
        ],
    }


# ─────────────────────────────────────────────────────────────
# TL-11.3-1 不能推断
# ─────────────────────────────────────────────────────────────

def test_force_magnitude_never_becomes_normal_force():
    """V2 的 force_magnitude 语义不明，任何情况下都不得变成 normal_force。"""
    v3, _ = migrate_dataset(_v2())
    for frame in v3["frames"]:
        assert frame["normal_force"] is None
        assert frame["field_status"]["normal_force"] == "unknown_for_migration"


def test_force_magnitude_never_becomes_normal_force_even_with_all_context():
    """即使把所有上下文都给了，force_magnitude 仍然不能推断成 normal_force。"""
    v3, _ = migrate_dataset(
        _v2(),
        frames_of_reference=[SURFACE_2D, SENSOR_3D],
        frame_map={"contact_centroid": "surface_0", "force_vector": "sensor_0"},
        centroid_method="geometric",
        torque_reference_point="sensor_origin",
        measurement_target="sensor_body",
    )
    for frame in v3["frames"]:
        assert frame["normal_force"] is None


def test_vector_fields_downgraded_without_frame():
    """没有坐标系就不搬向量字段，而不是假定一个坐标系。"""
    v3, _ = migrate_dataset(_v2())
    for frame in v3["frames"]:
        assert frame["contact_centroid"] is None
        assert frame["field_status"]["contact_centroid"] == "unknown_for_migration"


def test_centroid_downgraded_when_pressure_weighted_without_quantity():
    """声明 pressure_weighted 却不给权重来源，属于无法复现的派生，必须降级。"""
    v3, _ = migrate_dataset(
        _v2(),
        frames_of_reference=[SURFACE_2D],
        frame_map={"contact_centroid": "surface_0"},
        centroid_method="pressure_weighted",
    )
    assert v3["frames"][0]["contact_centroid"] is None
    assert "contact_centroid" in v3["frames"][0]["field_status"]


# ─────────────────────────────────────────────────────────────
# TL-11.3-2 不能丢失
# ─────────────────────────────────────────────────────────────

def test_original_values_preserved_in_sensor_specific():
    v3, _ = migrate_dataset(_v2())
    legacy = v3["frames"][0]["sensor_specific"]["legacy_force_magnitude"]
    assert legacy["value"] == 3.2
    assert legacy["unit"] == "N"
    assert legacy["namespace"] == "tlabel.v2"


def test_v2_sensor_specific_carried_over():
    v3, _ = migrate_dataset(_v2())
    assert v3["frames"][0]["sensor_specific"]["deformation_rms"] == 0.37


def test_deprecated_fields_preserved_as_legacy():
    v3, _ = migrate_dataset(_v2())
    ss = v3["frames"][0]["sensor_specific"]
    assert ss["legacy_confidence"]["value"] == 0.95
    assert ss["legacy_compliance_level"]["value"] == "L2"
    assert v3["frames"][0].get("confidence") is None


def test_audit_records_every_downgrade():
    v3, _ = migrate_dataset(_v2())
    fields = {a["field"] for a in v3["migration"]["audit"]}
    assert "force_magnitude" in fields
    assert "contact_centroid" in fields


def test_audit_keeps_distinct_values():
    """两帧原值不同时，审计不能只留第一帧的值。"""
    v3, _ = migrate_dataset(_v2())
    entry = next(a for a in v3["migration"]["audit"] if a["field"] == "force_magnitude")
    assert entry["original_value"] == "<varies>"
    assert set(entry["original_values"]) == {3.2, 3.4}
    assert entry["frame_count"] == 2


# ─────────────────────────────────────────────────────────────
# 迁移产物必须是合法 V3 数据
# ─────────────────────────────────────────────────────────────

def test_migrated_output_passes_core_validation():
    v3, _ = migrate_dataset(_v2())
    report = validate(v3, level="core")
    assert report.ok, [(f.code, f.path, f.message) for f in report.errors]


def test_migrated_output_with_context_passes_validation():
    v3, _ = migrate_dataset(
        _v2(),
        frames_of_reference=[SURFACE_2D],
        frame_map={"contact_centroid": "surface_0"},
        centroid_method="pressure_weighted",
        weighting_quantity="pixel_intensity",
    )
    assert v3["frames"][0]["contact_centroid"] == [12.4, 8.7]
    report = validate(v3, level="core")
    assert report.ok, [(f.code, f.path) for f in report.errors]


def test_migrated_output_declares_time_base():
    v3, _ = migrate_dataset(_v2())
    assert v3["time_base"] in {"dataset_start", "session_start", "unix_epoch", "device_clock"}
    assert v3["migration"]["time_base_inferred"] is True

    v3b, _ = migrate_dataset(_v2(), time_base="unix_epoch")
    assert v3b["time_base"] == "unix_epoch"
    assert v3b["migration"]["time_base_inferred"] is False


def test_capabilities_derived_from_data_not_only_declaration():
    """V2 的 capabilities 常填不全；数据里出现过的字段也算有能力，否则会报无能力却有值。"""
    v2 = _v2()
    v2["capabilities"].pop("slip_event", None)
    v3, _ = migrate_dataset(v2)
    assert v3["capabilities"]["slip_event"] is True
    assert validate(v3, level="core").ok


# ─────────────────────────────────────────────────────────────
# 拒绝不可迁移的源
# ─────────────────────────────────────────────────────────────

def test_rejects_v1_format():
    """V1 的 22 维格式没有 schema_v2，必须拒绝而不是静默产出全 null 数据集。"""
    v1 = {
        "format": "tlabel_v1",
        "tlabel_dimensions": 22,
        "frames": [{"frame_idx": 0, "timestamp_s": 0.0, "tlabel_v2": {"contact": 0.0}}],
    }
    with pytest.raises(MigrationSourceUnsupported, match="schema_v2"):
        migrate_dataset(v1)


def test_rejects_empty_dataset():
    with pytest.raises(MigrationSourceUnsupported, match="frames"):
        migrate_dataset({"format": "tlabel_schema_v2", "frames": []})


def test_rejects_partially_v1_mixed_frames():
    """只要所有帧都没有 schema_v2，就整体拒绝。"""
    v2 = _v2()
    for f in v2["frames"]:
        f.pop("schema_v2", None)
        f["tlabel_v2"] = {"contact": 0.0}
    with pytest.raises(MigrationSourceUnsupported):
        migrate_dataset(v2)


# ─────────────────────────────────────────────────────────────
# object_deformation：V2 里可能是图像 RMS，不是毫米
# ─────────────────────────────────────────────────────────────

def test_object_deformation_not_migrated_as_is():
    """V2 的 object_deformation 可能是图像强度 RMS（无量纲），不能直接当毫米搬过去。

    5 个视觉触觉 adapter（gelsight / vtouch / daimon_dataset /
    daimon_dm_tac / ycb_slide）在 V2 时代把它填成了图像 RMS。迁移工具
    只看到一个数字，无法判断单位，按 TL-11.3-1 一律不推断。
    """
    v3, _ = migrate_dataset(_v2())
    for frame in v3["frames"]:
        # TL-6.4-1：未识别形变对象时 MUST NOT 报告该字段，所以键直接不出现。
        assert frame.get("object_deformation") is None
        assert frame["field_status"]["object_deformation"] == "unknown_for_migration"


def test_object_deformation_preserved_in_legacy():
    """TL-11.3-2：不推断 ≠ 丢弃。原值必须留在可追溯位置。"""
    v3, _ = migrate_dataset(_v2())
    for frame, expected in zip(v3["frames"], (0.37, 0.41)):
        assert frame["sensor_specific"]["legacy_object_deformation"]["value"] == expected


def test_object_deformation_unknown_does_not_dirty_normal_force():
    """状态必须挂在目标字段上：object_deformation 的未知不能把 normal_force 也标脏。"""
    v2 = _v2()
    for f in v2["frames"]:
        f["schema_v2"]["force_magnitude"] = None
    v3, _ = migrate_dataset(v2)
    for frame in v3["frames"]:
        assert "normal_force" not in (frame.get("field_status") or {})


def test_object_deformation_audit_unit_is_honest():
    """审计里不能替源数据背书单位。单位不确定就写 unknown。"""
    v3, _ = migrate_dataset(_v2())
    entry = next(a for a in v3["migration"]["audit"] if a["field"] == "object_deformation")
    assert entry["original_unit"] == "unknown"
