"""第 11 章 —— 版本与迁移的测试。

迁移是整份规范里最容易「看起来成功」的一环：工具跑完不报错，产出一份
结构合法的数据集，但它可能把 `force_magnitude` 当成 `normal_force` 搬过去，
也可能把看不懂的值悄悄丢掉。两者都不报错，下游照常训练，结果是错的。

所以 TL-11.3 的四条不是「工具实现建议」，是**可被测试的行为契约**。
"""

from __future__ import annotations

import pytest

from tlabel.v3 import schema as S
from tlabel.v3 import validate
from tlabel.v3.migrate import MigrationSourceUnsupported, migrate_dataset

from conftest import base_dataset, base_frame, run


# ─────────────────────────────────────────────────────────────
# 测试夹具：一份最小的 V2 数据集
# ─────────────────────────────────────────────────────────────


def _v2_dataset(**frame_overrides):
    """构造一份 V2.1 数据集。

    V2 的特征是：帧里带 `schema_v2` 键、字段名是 `force_magnitude` 而不是
    `normal_force`、没有坐标系声明。
    """
    inner = {
        "contact": True,
        "contact_centroid": [1.0, 2.0, 3.0],
        "force_magnitude": 2.5,
        "force_vector": [0.0, 0.0, 2.5],
        "slip_event": False,
        "manipulation_phase": "grasp",
        "contact_region": "distal",
        "texture_class": "smooth",
    }
    inner.update(frame_overrides)
    return {
        "format": "tlabel_schema_v2",
        "schema_version_v2": "2.1",
        "sensor_id": "test_sensor",
        "capabilities": {
            "contact": True,
            "contact_centroid": True,
            "force_vector": True,
            "force_magnitude": True,
        },
        "frames": [
            {
                "frame_idx": 0,
                "timestamp_s": 0.0,
                "schema_v2": inner,
            }
        ],
    }


def _migrate(**kwargs):
    return migrate_dataset(_v2_dataset(), **kwargs)


# ─────────────────────────────────────────────────────────────
# 11.1 版本号
# ─────────────────────────────────────────────────────────────


def test_semantically_different_field_is_not_migrated_directly():
    """TL-11.1-1：Breaking 的判据是「下游会不会静默算错」。

    `force_magnitude` → `normal_force` 就是这个判据的样板：数值合法、单位
    都是 N、不报任何错，但 V2 的 `force_magnitude` 可能是法向力也可能是
    合力模长。直接搬运，下游算出的一切都错，而且不会有人发现。

    所以迁移后 `normal_force` 必须是 null，不能是 2.5。
    """
    v3, _ = _migrate()
    frame = v3["frames"][0]
    assert frame.get("normal_force") is None, "不得把 force_magnitude 直接搬成 normal_force"
    assert frame["field_status"]["normal_force"] == "unknown_for_migration"


def test_breaking_change_requires_null_not_guess():
    """TL-11.1-1 的推论：不能确定的值只能是 null。

    「猜一个最可能的」比 null 危险得多——null 会被下游当作缺失处理，
    猜错的值会被当作真值。
    """
    v3, _ = _migrate()
    frame = v3["frames"][0]
    for name in S.CORE_FIELDS:
        value = frame.get(name)
        if value is None:
            continue
        # 有值就必须有来源，否则它就是被猜出来的
        prov = (frame.get("field_provenance") or {}).get(name)
        if name == "timestamp_s":
            continue  # 时间戳从 V2 原样搬运，不需要 provenance
        assert prov is not None, f"{name} 有值但没有 provenance，来源不明"


# ─────────────────────────────────────────────────────────────
# 11.2 弃用
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(S.DEPRECATED_FIELDS))
def test_deprecated_field_remains_readable(name):
    """TL-11.2-1：弃用字段至少一个 MAJOR 内必须仍可读。

    「可读」的准确含义是：出现它不构成 error。校验器可以提示「这是弃用
    字段」，但不能拒绝整份数据——否则老数据全部失效，这条就形同虚设。

    注意这里用的是**带迁移审计**的数据，所以按 TL-6.5-2 只报 warning。
    """
    v3, _ = _migrate()
    frame = v3["frames"][0]
    frame[name] = 0.9
    report = validate(v3, level="core")
    codes = {f.code for f in report.errors}
    assert "DEPRECATED_FIELD_POPULATED" not in codes, "迁移数据的弃用字段不应报 error"


def test_deprecated_fields_are_not_removed_from_schema():
    """TL-11.2-1：弃用不等于删除。

    如果 `DEPRECATED_FIELDS` 里的字段从 schema 里消失，校验器会把它们当
    未知键，老数据就无法读取了。
    """
    for name in S.DEPRECATED_FIELDS:
        assert name in S.ALL_FIELDS, f"{name} 已弃用但不在 ALL_FIELDS 内"


# ─────────────────────────────────────────────────────────────
# 11.3 迁移
# ─────────────────────────────────────────────────────────────


def test_migration_does_not_infer_missing_context():
    """TL-11.3-1：源数据里没有的信息，迁移后也不能凭空出现。

    这份 V2 数据没有坐标系。迁移工具不能替它编一个 `sensor_0`——编出来的
    坐标系看起来是确定的，比缺失更危险。所以 `force_vector` 必须是 null。
    """
    v3, _ = _migrate()
    frame = v3["frames"][0]
    assert frame.get("force_vector") is None
    assert frame["field_status"]["force_vector"] == "unknown_for_migration"


def test_migration_does_not_discard_relocatable_value():
    """TL-11.3-2：能搬进扩展容器的值不得丢弃。

    `force_magnitude` 无法映射到 `normal_force`，但它的数值本身有价值——
    研究者可能想回查原始记录。所以它必须出现在 `sensor_specific` 里，
    而不是被扔掉。
    """
    v3, _ = _migrate()
    frame = v3["frames"][0]
    legacy = frame["sensor_specific"]["legacy_force_magnitude"]
    assert legacy["value"] == 2.5
    assert legacy["unit"] == "N"
    assert legacy["namespace"] == "tlabel.v2"


def test_migration_audit_records_six_items():
    """TL-11.3-3：每一次非自动变更都必须记录六项。

    少任何一项，审计就无法复现：知道「字段被降级了」但不知道「原值多少」，
    或者知道「值搬到了 sensor_specific」但不知道「为什么」，都无法人工复核。

    六项：原字段名、原值、原单位、原方法、目标位置、原因。
    """
    v3, _ = _migrate()
    audit = v3["migration"]["audit"]
    assert audit, "应有审计记录"
    for entry in audit:
        for key in (
            "field",
            "original_value",
            "original_unit",
            "original_method",
            "target_location",
            "reason",
        ):
            assert key in entry, f"审计条目缺少 {key}：{entry}"
        assert entry["reason"], f"{entry['field']} 的降级没有说明原因"


def test_migration_audit_covers_every_degraded_field():
    """TL-11.3-3：审计必须覆盖**每一处**降级，不能只记一部分。

    抽查方式是反向的：帧里每个 `unknown_for_migration` 状态，都必须在
    审计里找到对应记录。少记一条，就意味着有一处变更无人知晓。
    """
    v3, _ = _migrate()
    frame = v3["frames"][0]
    audited_fields = {a["field"] for a in v3["migration"]["audit"]}
    for field_name, status in (frame.get("field_status") or {}).items():
        if status != "unknown_for_migration":
            continue
        # force_magnitude 在 V3 里改叫 normal_force，审计记的是 V2 原名
        source_name = "force_magnitude" if field_name == "normal_force" else field_name
        assert source_name in audited_fields, f"{field_name} 被降级但审计里没有记录"


def test_unmigratable_value_becomes_null_with_status():
    """TL-11.3-4：无法迁移的值必须置 null 并标 `unknown_for_migration`。

    不能改成 0、不能改成 NaN、不能保留原值——这三种做法都会让下游以为
    「这里有个数，可以用」。
    """
    v3, _ = _migrate()
    frame = v3["frames"][0]
    for name in ("normal_force", "force_vector", "contact_centroid"):
        assert frame.get(name) is None, f"{name} 应为 null"
        assert frame["field_status"][name] == "unknown_for_migration"


def test_migrated_dataset_passes_core_validation():
    """TL-11.3 的总体验收：迁移产物必须是合法的 V3.0 数据。

    这条是整章的兜底——上面每一条都对，但组合起来产出一份校验不过的数据，
    说明某条条款的实现方式与另一条冲突。
    """
    v3, _ = _migrate()
    report = validate(v3, level="core")
    assert report.ok, [(f.code, f.clause, f.path) for f in report.errors]


def test_migration_refuses_non_v2_source():
    """TL-11.3 的前提：只迁移它认识的格式。

    对 V1 的 22 维数据静默跑一遍，会产出一份「结构合法、语义全空」的
    数据集——比直接失败更难被发现。
    """
    with pytest.raises(MigrationSourceUnsupported):
        migrate_dataset({"frames": [{"timestamp_s": 0.0}]})


def test_migration_provenance_is_external_not_measured():
    """TL-11.3-1 的延伸：迁移来的值不得伪装成测量值。

    值确实来自外部数据集，`type` 就应该是 `externally_supplied`。标成
    `measured` 会让下游以为这是本设备的原始读数，从而误判精度。
    """
    v3, _ = _migrate(frame_map={"force_vector": "sensor_0"},
                     frames_of_reference=[{
                         "frame_id": "sensor_0",
                         "dimension": 3,
                         "frame_type": "sensor_local",
                         "handedness": "right",
                         "axes": {"x": "normal", "y": "tangential_1", "z": "tangential_2"},
                         "reference_object": "sensor_surface",
                     }])
    prov = v3["frames"][0]["field_provenance"]["force_vector"]
    assert prov["type"] == "externally_supplied"
    assert prov["source_id"] == "test_sensor"
