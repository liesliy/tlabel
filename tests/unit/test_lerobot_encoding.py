"""
LeRobot 编码路径回归测试（PIT-6）

v0.23.1 之前，写回路径 ``tlabel_to_lerobot()`` 对非标量字段一律填 0.0，
导致向量字段（contact_centroid / force_vector / torque_vector / slip_velocity）
和枚举字段（contact_region / manipulation_phase / texture_class）被静默抹平。

本测试锁定 v0.23.3 起的正确编码语义，并保证两条 LeRobot 路径
（lerobot.py 写回 / lerobot_export.py 新建）输出一致。
"""

import json
import math

import pytest

from tlabel.converters._encoding import (
    COMPLIANCE_LEVEL_MAP,
    CONTACT_REGION_MAP,
    ENUM_DEFAULTS,
    MANIPULATION_PHASE_MAP,
    TEXTURE_CLASS_MAP,
    encode_field,
    encode_schema_v2,
)
from tlabel.core.schema import SCHEMA_V2_FIELD_NAMES, TLabelSchemaV2


# ================================================================
# 单字段编码
# ================================================================

class TestEncodeField:
    def test_bool_fields(self):
        assert encode_field("contact", True) == 1.0
        assert encode_field("contact", False) == 0.0
        assert encode_field("slip_event", True) == 1.0
        assert encode_field("slip_event", None) == 0.0

    def test_bool_not_confused_with_int(self):
        """bool 是 int 子类，必须先判 bool 再判数值。"""
        assert encode_field("contact", 1) == 1.0
        assert encode_field("contact", 0) == 0.0

    @pytest.mark.parametrize("field,value,expected", [
        ("contact_centroid", [3.0, 4.0], 5.0),
        ("force_vector", [1.0, 2.0, 2.0], 3.0),
        ("slip_velocity", [3.0, 4.0], 5.0),
        ("torque_vector", [0.0, 0.0, 2.0], 2.0),
    ])
    def test_vector_fields_use_l2_norm(self, field, value, expected):
        assert encode_field(field, value) == pytest.approx(expected)

    @pytest.mark.parametrize("field", [
        "contact_centroid", "force_vector", "slip_velocity", "torque_vector",
    ])
    def test_vector_none_is_zero(self, field):
        assert encode_field(field, None) == 0.0

    @pytest.mark.parametrize("field,mapping", [
        ("contact_region", CONTACT_REGION_MAP),
        ("manipulation_phase", MANIPULATION_PHASE_MAP),
        ("texture_class", TEXTURE_CLASS_MAP),
    ])
    def test_enum_fields_map_to_category_index(self, field, mapping):
        for name, idx in mapping.items():
            assert encode_field(field, name) == pytest.approx(idx)

    def test_enum_unknown_and_none_use_sentinel(self):
        assert encode_field("contact_region", None) == -1.0
        assert encode_field("contact_region", "not_a_region") == -1.0
        assert encode_field("manipulation_phase", None) == -1.0
        assert encode_field("texture_class", None) == -1.0

    def test_compliance_level_default_is_l1(self):
        assert encode_field("compliance_level", "L3") == 3.0
        # 历史行为：缺省按 L1 编码
        assert encode_field("compliance_level", None) == 1.0

    def test_scalar_fields(self):
        assert encode_field("force_magnitude", 12.5) == 12.5
        assert encode_field("temperature", 25) == 25.0
        assert encode_field("object_deformation", None) == 0.0
        assert encode_field("confidence", 0.9) == pytest.approx(0.9)

    def test_confidence_default_is_one(self):
        assert encode_field("confidence", None) == 1.0


# ================================================================
# 整帧编码
# ================================================================

class TestEncodeSchemaV2:
    def test_field_count_matches_schema(self):
        sv2 = TLabelSchemaV2(contact=True)
        vec = encode_schema_v2(sv2)
        assert len(vec) == len(SCHEMA_V2_FIELD_NAMES) == 14

    def test_full_frame_from_object(self):
        sv2 = TLabelSchemaV2(
            contact=True,
            contact_centroid=[3.0, 4.0],
            contact_region="digital",
            force_magnitude=2.5,
            force_vector=[1.0, 2.0, 2.0],
            slip_event=False,
            slip_velocity=[3.0, 4.0],
            manipulation_phase="grasp",
            texture_class="rough",
            temperature=25.0,
            confidence=0.9,
            compliance_level="L2",
        )
        vec = encode_schema_v2(sv2)
        got = dict(zip(SCHEMA_V2_FIELD_NAMES, vec))

        assert got["contact"] == 1.0
        assert got["contact_centroid"] == pytest.approx(5.0)
        assert got["contact_region"] == pytest.approx(CONTACT_REGION_MAP["digital"])
        assert got["force_magnitude"] == pytest.approx(2.5)
        assert got["force_vector"] == pytest.approx(3.0)
        assert got["slip_velocity"] == pytest.approx(5.0)
        assert got["manipulation_phase"] == pytest.approx(MANIPULATION_PHASE_MAP["grasp"])
        assert got["texture_class"] == pytest.approx(TEXTURE_CLASS_MAP["rough"])
        assert got["temperature"] == pytest.approx(25.0)
        assert got["confidence"] == pytest.approx(0.9)
        assert got["compliance_level"] == 2.0

    def test_dict_and_object_produce_same_vector(self):
        """lerobot.py 传 dict、lerobot_export.py 传对象，结果必须一致。"""
        sv2 = TLabelSchemaV2(
            contact=True,
            contact_centroid=[3.0, 4.0],
            contact_region="digital",
            force_vector=[1.0, 2.0, 2.0],
            manipulation_phase="grasp",
        )
        as_dict = {name: getattr(sv2, name) for name in SCHEMA_V2_FIELD_NAMES}
        assert encode_schema_v2(sv2) == encode_schema_v2(as_dict)

    def test_custom_field_order(self):
        sv2 = {"contact": True, "force_vector": [3.0, 4.0]}
        vec = encode_schema_v2(sv2, ["contact", "force_vector"])
        assert vec == [1.0, 5.0]

    def test_no_silent_zero_for_vector_or_enum(self):
        """回归防线：向量与枚举字段不得被静默置 0.0。"""
        sv2 = {
            "contact_centroid": [3.0, 4.0],
            "force_vector": [1.0, 2.0, 2.0],
            "slip_velocity": [3.0, 4.0],
            "contact_region": "digital",
            "manipulation_phase": "grasp",
        }
        vec = dict(zip(SCHEMA_V2_FIELD_NAMES, encode_schema_v2(sv2)))
        for field in ("contact_centroid", "force_vector", "slip_velocity"):
            assert vec[field] != 0.0, f"{field} 被静默置零"
        assert vec["contact_region"] == 1.0
        assert vec["manipulation_phase"] == 2.0


# ================================================================
# 两条 LeRobot 路径一致性
# ================================================================

class TestPathConsistency:
    def test_export_and_writeback_share_encoding(self):
        from tlabel.converters.lerobot_export import _encode_schema_v2_to_vector

        sv2 = TLabelSchemaV2(
            contact=True,
            contact_centroid=[3.0, 4.0],
            contact_region="palmar",
            force_magnitude=1.5,
            force_vector=[0.0, 3.0, 4.0],
            slip_velocity=[1.0, 0.0],
            manipulation_phase="hold",
            texture_class="smooth",
            compliance_level="L3",
        )
        assert _encode_schema_v2_to_vector(sv2) == encode_schema_v2(sv2)


# ================================================================
# 端到端：写回已有 LeRobot 数据集
# ================================================================

pyarrow = pytest.importorskip("pyarrow", reason="LeRobot 转换器需要 pyarrow")


def _build_minimal_lerobot_dataset(root):
    """构造一个最小可写回的 LeRobot 数据集（data + meta 目录）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "data").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)

    table = pa.table({"frame_index": [0, 1], "action": [[0.0, 0.0]] * 2})
    pq.write_table(table, root / "data" / "chunk-0000.parquet")
    (root / "meta" / "info.json").write_text(
        json.dumps({"robot_type": "test", "features": {}})
    )


def test_writeback_preserves_vector_and_enum_fields(tmp_path):
    """端到端：tlabel_to_lerobot 写回的触觉矩阵不得丢失向量/枚举语义。"""
    import pyarrow.parquet as pq

    from tlabel.converters.lerobot import tlabel_to_lerobot

    lerobot_dir = tmp_path / "lerobot_ds"
    _build_minimal_lerobot_dataset(lerobot_dir)

    tlabel_json = {
        "feature_names_v2": list(SCHEMA_V2_FIELD_NAMES),
        "frames": [
            {"schema_v2": {
                "contact": True,
                "contact_centroid": [3.0, 4.0],
                "contact_region": "digital",
                "force_vector": [1.0, 2.0, 2.0],
                "manipulation_phase": "grasp",
                "confidence": 0.9,
            }},
            {"schema_v2": {
                "contact": False,
                "slip_velocity": [3.0, 4.0],
                "texture_class": "rough",
                "temperature": 25.0,
            }},
        ],
    }
    tlabel_path = tmp_path / "ann.json"
    tlabel_path.write_text(json.dumps(tlabel_json))

    tlabel_to_lerobot(str(tlabel_path), str(lerobot_dir), overwrite=True)

    df = pq.read_table(lerobot_dir / "data" / "chunk-0000.parquet").to_pandas()
    rows = [list(map(float, r)) for r in df["observation.tactile"]]

    f0 = dict(zip(SCHEMA_V2_FIELD_NAMES, rows[0]))
    assert f0["contact_centroid"] == pytest.approx(5.0), "contact_centroid 丢失"
    assert f0["force_vector"] == pytest.approx(3.0), "force_vector 丢失"
    assert f0["contact_region"] == pytest.approx(1.0), "contact_region 丢失"
    assert f0["manipulation_phase"] == pytest.approx(2.0), "manipulation_phase 丢失"
    assert f0["confidence"] == pytest.approx(0.9)

    f1 = dict(zip(SCHEMA_V2_FIELD_NAMES, rows[1]))
    assert f1["slip_velocity"] == pytest.approx(5.0), "slip_velocity 丢失"
    assert f1["texture_class"] == pytest.approx(1.0), "texture_class 丢失"
    assert f1["temperature"] == pytest.approx(25.0)
    # 缺失的枚举字段用 -1.0 哨兵，而不是 0.0（0.0 是 palmar 的索引，会造假）
    assert f1["contact_region"] == -1.0
    assert f1["manipulation_phase"] == -1.0
