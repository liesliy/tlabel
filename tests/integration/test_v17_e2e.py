#!/usr/bin/env python3
"""
TLabel v0.17 Phase 1-3 全流程端到端测试

覆盖范围:
  Test 1:  Schema V2 基础 (14维字段、序列化、验证、compliance_level)
  Test 2:  TLabelFrame + TLabelData 集成 (新旧格式双轨兼容)
  Test 3:  适配器 extract_schema() (paxini_gen3, touchd, gelsight, ycb_slide, tacquad)
  Test 4:  CLI 校验 (新旧格式校验、非法数据)
  Test 5:  导出 (Schema V2 only: CSV/HDF5/JSON、_flatten_schema_v2)
  Test 6:  质量评分 (新旧数据 QualityScorer.score())
  Test 7:  预测引擎 (新旧数据 PredictEngine)
  Test 8:  数据增强 (16列/22列矩阵 AugmentEngine)
  Test 9:  转换器 (tlabel_to_ftp1 新旧数据)
  Test 10: 端到端流水线 (适配器 → validate → predict → quality → export)
"""

import sys
import os
import json
import math
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# 确保 tlabel 在 sys.path 上
# v0.23: derive the repo root from this file's location
# (tests/integration/<file>.py -> repo root is two levels up)
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ============================================================================
# 测试结果收集
# ============================================================================

test_results = {}
issues_found = []


def record_test(test_name: str, passed: bool, details: str):
    """记录测试结果"""
    test_results[test_name] = {"passed": passed, "details": details}
    status = "✅" if passed else "❌"
    print(f"  {status} {test_name}: {details}")
    if not passed:
        issues_found.append(f"{test_name}: {details}")


# ============================================================================
# 辅助工具：构造模拟数据
# ============================================================================

def make_legacy_v2_dict(contact=1.0, force_magnitude=0.5, slip_event=0.0,
                        deformation_magnitude=0.3, **overrides) -> Dict[str, float]:
    """构造旧版22维 tlabel_v2 flat dict"""
    d = {
        "contact": contact,
        "deformation_magnitude": deformation_magnitude,
        "force_magnitude": force_magnitude,
        "force_peak": 0.6,
        "force_direction": 45.0,
        "slip_entropy": 0.1,
        "slip_event": slip_event,
        "texture_energy": 0.2,
        "edge_density": 0.15,
        "contact_area": 0.25,
        "centroid_x": 0.5,
        "normal_field_magnitude": 0.4,
        "normal_field_variance": 0.1,
        "shear_field_magnitude": 0.05,
        "shear_field_direction": 30.0,
        "delta_force_normal": 0.02,
        "delta_force_shear": 0.01,
        "friction_cone_ratio": 0.3,
        "optical_flow_magnitude": 0.1,
        "optical_flow_direction": 90.0,
        "temporal_deformation_rate": 0.05,
        "contact_transition": 0.0,
    }
    d.update(overrides)
    return d


def make_schema_v2_instance(contact=True, force_magnitude=0.5, force_vector=None,
                            slip_event=False, compliance_level="L2",
                            contact_centroid=None, **kwargs):
    """构造 TLabelSchemaV2 实例"""
    from tlabel.core.schema import TLabelSchemaV2
    return TLabelSchemaV2(
        contact=contact,
        contact_centroid=contact_centroid or ([0.5, 0.3] if contact else None),
        contact_region=kwargs.get("contact_region"),
        force_magnitude=force_magnitude,
        force_vector=force_vector,
        torque_vector=kwargs.get("torque_vector"),
        slip_event=slip_event,
        slip_velocity=kwargs.get("slip_velocity"),
        manipulation_phase=kwargs.get("manipulation_phase"),
        texture_class=kwargs.get("texture_class"),
        object_deformation=kwargs.get("object_deformation", 0.3),
        temperature=kwargs.get("temperature"),
        confidence=kwargs.get("confidence", 0.9),
        compliance_level=compliance_level,
    )


def make_legacy_frame(frame_idx=0, contact=1.0, **kwargs) -> "TLabelFrame":
    """构造旧格式 TLabelFrame (只有 tlabel_v2 dict，无 schema_v2)"""
    from tlabel.core.types import TLabelFrame
    v2_dict = make_legacy_v2_dict(contact=contact, **kwargs)
    return TLabelFrame(
        frame_idx=frame_idx,
        timestamp_s=frame_idx * 0.033,
        tlabel_v2=v2_dict,
        manipulation_phase="stable_contact" if contact > 0.5 else "idle",
        confidence=0.9,
    )


def make_new_frame(frame_idx=0, contact=True, compliance_level="L2",
                    force_vector=None, **kwargs) -> "TLabelFrame":
    """构造新格式 TLabelFrame (带 schema_v2)"""
    from tlabel.core.types import TLabelFrame
    v2_dict = make_legacy_v2_dict(contact=1.0 if contact else 0.0, **kwargs)
    schema_v2 = make_schema_v2_instance(
        contact=contact, compliance_level=compliance_level,
        force_vector=force_vector, **kwargs
    )
    return TLabelFrame(
        frame_idx=frame_idx,
        timestamp_s=frame_idx * 0.033,
        tlabel_v2=v2_dict,
        manipulation_phase="grasp" if contact else "pre_contact",
        confidence=0.9,
        schema_v2=schema_v2,
    )


def make_legacy_tlabel_data(n_frames=20) -> "TLabelData":
    """构造旧格式 TLabelData (frames 无 schema_v2)"""
    from tlabel.core.types import TLabelData
    frames = []
    for i in range(n_frames):
        contact = 1.0 if 5 <= i < 15 else 0.0
        slip = 1.0 if 10 <= i < 12 else 0.0
        frames.append(make_legacy_frame(
            frame_idx=i, contact=contact, slip_event=slip,
            force_magnitude=0.3 + 0.05 * i,
        ))
    return TLabelData(
        frames=frames,
        sensor_info={"type": "vision-based_tactile", "manufacturer": "test"},
        episode_info={"source": "test"},
        capabilities={"contact": True, "force_magnitude": True},
    )


def make_new_tlabel_data(n_frames=20, compliance_level="L2",
                          force_vector=None) -> "TLabelData":
    """构造新格式 TLabelData (frames 带 schema_v2)"""
    from tlabel.core.types import TLabelData
    frames = []
    for i in range(n_frames):
        contact = True if 5 <= i < 15 else False
        slip = True if 10 <= i < 12 else False
        frames.append(make_new_frame(
            frame_idx=i, contact=contact, compliance_level=compliance_level,
            force_vector=force_vector, slip_event=slip,
            force_magnitude=0.3 + 0.05 * i,
        ))
    return TLabelData(
        frames=frames,
        sensor_info={"type": "vision-based_tactile", "manufacturer": "test"},
        episode_info={"source": "test"},
        capabilities={"contact": True, "force_magnitude": True},
        schema_version="0.17.0",
    )


# ============================================================================
# Test 1: Schema V2 基础
# ============================================================================

def test_1_schema_v2_basic():
    print("\n" + "=" * 60)
    print("Test 1: Schema V2 基础")
    print("=" * 60)

    from tlabel.core.schema import (
        TLabelSchemaV2, SCHEMA_V2_FIELD_NAMES,
        VALID_COMPLIANCE_LEVELS, VALID_CONTACT_REGIONS,
        VALID_MANIPULATION_PHASES, VALID_TEXTURE_CLASSES,
    )

    # 1.1 14维字段完整性
    try:
        assert len(SCHEMA_V2_FIELD_NAMES) == 14, f"Expected 14 fields, got {len(SCHEMA_V2_FIELD_NAMES)}"
        record_test("1.1 14维字段数量", True, f"{len(SCHEMA_V2_FIELD_NAMES)} 维字段")
    except Exception as e:
        record_test("1.1 14维字段数量", False, str(e))

    # 1.2 创建实例并验证字段
    try:
        schema = TLabelSchemaV2(
            contact=True,
            contact_centroid=[0.5, 0.3],
            force_magnitude=0.5,
            force_vector=[0.1, 0.0, 0.49],
            torque_vector=[0.01, 0.02, 0.03],
            slip_event=False,
            slip_velocity=None,
            manipulation_phase="grasp",
            texture_class="smooth",
            object_deformation=0.3,
            temperature=25.0,
            confidence=0.95,
            compliance_level="L4",
            contact_region="digital",
        )
        # 验证所有14个字段有值
        assert schema.contact is True
        assert schema.contact_centroid == [0.5, 0.3]
        assert schema.force_magnitude == 0.5
        assert schema.force_vector == [0.1, 0.0, 0.49]
        assert schema.torque_vector == [0.01, 0.02, 0.03]
        assert schema.slip_event is False
        assert schema.manipulation_phase == "grasp"
        assert schema.texture_class == "smooth"
        assert schema.object_deformation == 0.3
        assert schema.temperature == 25.0
        assert schema.confidence == 0.95
        assert schema.compliance_level == "L4"
        assert schema.contact_region == "digital"
        record_test("1.2 创建实例14维完整", True, "所有14个字段正确赋值")
    except Exception as e:
        record_test("1.2 创建实例14维完整", False, str(e))

    # 1.3 from_dict / to_dict 往返一致性
    try:
        original = make_schema_v2_instance(
            contact=True, force_magnitude=0.5,
            force_vector=[0.1, 0.0, 0.49],
            compliance_level="L3",
            contact_centroid=[0.5, 0.3],
            manipulation_phase="grasp",
        )
        d = original.to_dict()
        restored = TLabelSchemaV2.from_dict(d)
        assert restored.contact == original.contact
        assert restored.force_magnitude == original.force_magnitude
        assert restored.force_vector == original.force_vector
        assert restored.compliance_level == original.compliance_level
        assert restored.contact_centroid == original.contact_centroid
        assert restored.manipulation_phase == original.manipulation_phase
        assert restored.confidence == original.confidence
        record_test("1.3 from_dict/to_dict 往返", True, "序列化往返一致")
    except Exception as e:
        record_test("1.3 from_dict/to_dict 往返", False, str(e))

    # 1.4 validate() 对合法数据
    try:
        schema_valid = make_schema_v2_instance(
            contact=True, compliance_level="L2",
            contact_centroid=[0.5, 0.3], force_magnitude=0.5,
        )
        is_valid, errors = schema_valid.validate()
        assert is_valid, f"合法数据应通过验证, errors: {errors}"
        record_test("1.4a validate 合法数据", True, "通过验证")
    except Exception as e:
        record_test("1.4a validate 合法数据", False, str(e))

    # 1.5 validate() 对非法数据
    try:
        schema_invalid = TLabelSchemaV2(
            contact=True,
            contact_centroid=None,  # contact=True 但无 centroid → 应报错
            confidence=1.5,         # 超出范围
            compliance_level="L5",  # 非法枚举
            force_vector=[1.0, 2.0],  # 维度错误(需3)
        )
        is_valid, errors = schema_invalid.validate()
        assert not is_valid, "非法数据应不通过验证"
        # 检查错误数量
        assert len(errors) >= 3, f"至少3个错误, got {len(errors)}: {errors}"
        record_test("1.4b validate 非法数据", True, f"检测到 {len(errors)} 个错误: {errors[:2]}...")
    except Exception as e:
        record_test("1.4b validate 非法数据", False, str(e))

    # 1.6 compliance_level L1-L4 枚举合法性
    try:
        # 构造满足各 level 条件的 schema，仅验证 compliance_level 枚举值本身合法
        level_schemas = {
            "L1": TLabelSchemaV2(compliance_level="L1", contact=False),
            "L2": TLabelSchemaV2(compliance_level="L2", contact=True,
                                  contact_centroid=[0.5, 0.3], force_magnitude=0.5),
            "L3": TLabelSchemaV2(compliance_level="L3", contact=True,
                                  contact_centroid=[0.5, 0.3], force_magnitude=0.5,
                                  force_vector=[0.1, 0.0, 0.49]),
            "L4": TLabelSchemaV2(compliance_level="L4", contact=True,
                                  contact_centroid=[0.5, 0.3], force_magnitude=0.5,
                                  force_vector=[0.1, 0.0, 0.49]),
        }
        all_ok = True
        for level, s in level_schemas.items():
            is_valid, errors = s.validate()
            compliance_errors = [e for e in errors if "compliance_level" in e]
            if compliance_errors:
                all_ok = False
                break
        assert all_ok, "L1-L4 应为合法 compliance_level 枚举值"
        record_test("1.5 compliance_level L1-L4", True, "L1-L4 均为合法枚举值")
    except Exception as e:
        record_test("1.5 compliance_level L1-L4", False, str(e))


# ============================================================================
# Test 2: TLabelFrame + TLabelData 集成
# ============================================================================

def test_2_frame_data_integration():
    print("\n" + "=" * 60)
    print("Test 2: TLabelFrame + TLabelData 集成")
    print("=" * 60)

    from tlabel.core.types import TLabelFrame, TLabelData
    from tlabel.core.schema import TLabelSchemaV2

    # 2.1 带schema_v2的TLabelFrame
    try:
        frame = make_new_frame(frame_idx=0, contact=True, compliance_level="L2")
        assert hasattr(frame, "schema_v2"), "frame 应有 schema_v2 属性"
        assert isinstance(frame.schema_v2, TLabelSchemaV2), "schema_v2 应为 TLabelSchemaV2 实例"
        assert frame.schema_v2.compliance_level == "L2"
        record_test("2.1 带schema_v2的Frame", True, "schema_v2 属性正确")
    except Exception as e:
        record_test("2.1 带schema_v2的Frame", False, str(e))

    # 2.2 带schema_v2的TLabelData
    try:
        data = make_new_tlabel_data(n_frames=5, compliance_level="L2")
        assert data.schema_version == "0.17.0"
        assert len(data.frames) == 5
        assert data.frames[0].schema_v2 is not None
        record_test("2.2 带schema_v2的TLabelData", True, "schema_version 和 frames 正确")
    except Exception as e:
        record_test("2.2 带schema_v2的TLabelData", False, str(e))

    # 2.3 旧格式TLabelFrame不报错
    try:
        frame_legacy = make_legacy_frame(frame_idx=0, contact=1.0)
        assert frame_legacy.schema_v2 is None, "旧格式 frame.schema_v2 应为 None"
        record_test("2.3 旧格式Frame不报错", True, "schema_v2=None，无异常")
    except Exception as e:
        record_test("2.3 旧格式Frame不报错", False, str(e))

    # 2.4 to_dict 新格式序列化
    try:
        data = make_new_tlabel_data(n_frames=3)
        d = data.to_dict()
        assert "schema_version_v2" in d, "to_dict 应包含 schema_version_v2"
        assert d["schema_version_v2"] == "2.1"
        assert "feature_names_v2" in d, "to_dict 应包含 feature_names_v2"
        assert len(d["feature_names_v2"]) == 14
        record_test("2.4a 新格式to_dict", True, "包含 schema_version_v2 和 feature_names_v2")
    except Exception as e:
        record_test("2.4a 新格式to_dict", False, str(e))

    # 2.5 旧格式to_dict序列化
    try:
        data_legacy = make_legacy_tlabel_data(n_frames=3)
        d = data_legacy.to_dict()
        assert "schema_version" in d
        assert "frames" in d
        assert len(d["frames"]) == 3
        record_test("2.4b 旧格式to_dict", True, "旧格式序列化正常")
    except Exception as e:
        record_test("2.4b 旧格式to_dict", False, str(e))

    # 2.6 to_schema_v2() 方法 — 旧帧自动转换
    try:
        frame_legacy = make_legacy_frame(frame_idx=0, contact=1.0)
        sv2 = frame_legacy.to_schema_v2()
        assert isinstance(sv2, TLabelSchemaV2), "to_schema_v2 应返回 TLabelSchemaV2"
        assert sv2.contact is True, f"contact 应为 True, got {sv2.contact}"
        # 转换后应缓存
        assert frame_legacy.schema_v2 is not None, "转换后应缓存到 schema_v2"
        record_test("2.5 to_schema_v2() 旧帧自动转换", True, "旧帧通过 to_schema_v2() 正确转换")
    except Exception as e:
        record_test("2.5 to_schema_v2() 旧帧自动转换", False, str(e))

    # 2.7 dimension_keys 返回14维
    try:
        data_new = make_new_tlabel_data(n_frames=3)
        dims = data_new.dimension_keys
        assert len(dims) == 14, f"dimension_keys 应为14维, got {len(dims)}"
        record_test("2.6 dimension_keys 返回14维", True, f"返回 {len(dims)} 维")
    except Exception as e:
        record_test("2.6 dimension_keys 返回14维", False, str(e))


# ============================================================================
# Test 3: 适配器 extract_schema()
# ============================================================================

def test_3_adapters_extract_schema():
    print("\n" + "=" * 60)
    print("Test 3: 适配器 extract_schema()")
    print("=" * 60)

    from tlabel.core.schema import TLabelSchemaV2

    # 3.1 paxini_gen3
    try:
        from tlabel.adapters.paxini_gen3 import PaxiniGen3Adapter
        adapter = PaxiniGen3Adapter()
        raw = {
            "pressure_map_norm": __import__("numpy").zeros((8, 8), dtype=__import__("numpy").float32),
            "contact_mask": __import__("numpy").ones((8, 8), dtype=bool),
            "total_force_n": 5.0,
            "centroid": (4, 4),
            "in_contact": True,
            "contact_area_mm2": 50.0,
            "timestamp_s": 0.1,
            "dt": 0.01,
            "prev_state": None,
        }
        schema = adapter.extract_schema(raw)
        assert isinstance(schema, TLabelSchemaV2), "应返回 TLabelSchemaV2"
        assert schema.compliance_level == "L2", f"paxini_gen3 应为 L2, got {schema.compliance_level}"
        assert schema.force_vector is None, "L2 适配器 force_vector 应为 None"
        assert schema.force_magnitude is not None, "应有 force_magnitude"
        record_test("3.1 paxini_gen3 extract_schema", True,
                     f"L2, force_mag={schema.force_magnitude}, force_vec=None")
    except Exception as e:
        record_test("3.1 paxini_gen3 extract_schema", False, str(e))

    # 3.2 touchd
    try:
        from tlabel.adapters.touchd import ToucHDAdapter
        adapter = ToucHDAdapter()
        # 构造带3D力的模拟帧
        frame = make_new_frame(frame_idx=0, contact=True)
        frame.sensor_specific = {
            "force_xyz_normalized": [0.3, 0.1, 0.8],
            "force_xyz_raw_N": [1.5, 0.5, 4.0],
        }
        schema = adapter.extract_schema(frame)
        assert isinstance(schema, TLabelSchemaV2)
        assert schema.compliance_level == "L3", f"touchd 有3D力应为 L3, got {schema.compliance_level}"
        assert schema.force_vector is not None, "touchd 应有 force_vector"
        assert len(schema.force_vector) == 3, "force_vector 应为3维"
        record_test("3.2 touchd extract_schema", True,
                     f"L3, force_vector={schema.force_vector}")
    except Exception as e:
        record_test("3.2 touchd extract_schema", False, str(e))

    # 3.3 gelsight (L2, 无标定力时)
    try:
        from tlabel.adapters.gelsight import GelSightAdapter
        adapter = GelSightAdapter()
        # 模拟无标定力的 raw_frame_data
        raw = {
            "diff_img": __import__("numpy").random.randn(120, 160, 3).astype(__import__("numpy").float32) * 10,
            "is_contact": True,
        }
        schema = adapter.extract_schema(raw)
        assert isinstance(schema, TLabelSchemaV2)
        assert schema.compliance_level == "L2", f"gelsight 无标定应为 L2, got {schema.compliance_level}"
        assert schema.force_vector is None, "无标定力时 force_vector 应为 None"
        record_test("3.3a gelsight extract_schema (无标定)", True,
                     f"L2, force_vec=None")
    except Exception as e:
        record_test("3.3a gelsight extract_schema (无标定)", False, str(e))

    # 3.4 gelsight (有标定力时)
    try:
        raw_with_force = {
            "diff_img": __import__("numpy").random.randn(120, 160, 3).astype(__import__("numpy").float32) * 10,
            "is_contact": True,
            "force_vector_N": [1.0, 0.5, 3.0],
        }
        schema = adapter.extract_schema(raw_with_force)
        assert schema.compliance_level == "L3", f"有标定力应为 L3, got {schema.compliance_level}"
        assert schema.force_vector == [1.0, 0.5, 3.0], f"force_vector 不正确: {schema.force_vector}"
        record_test("3.3b gelsight extract_schema (有标定)", True,
                     f"L3, force_vector={schema.force_vector}")
    except Exception as e:
        record_test("3.3b gelsight extract_schema (有标定)", False, str(e))

    # 3.5 ycb_slide
    try:
        from tlabel.adapters.ycb_slide import YCBSlideAdapter
        adapter = YCBSlideAdapter()
        raw = {
            "diff_img": __import__("numpy").random.randn(240, 320, 3).astype(__import__("numpy").float32) * 5,
            "is_contact": True,
        }
        schema = adapter.extract_schema(raw)
        assert isinstance(schema, TLabelSchemaV2)
        assert schema.compliance_level == "L2", f"ycb_slide 应为 L2, got {schema.compliance_level}"
        assert schema.force_vector is None, "ycb_slide 无3D力, force_vector 应为 None"
        record_test("3.4 ycb_slide extract_schema", True, "L2, force_vec=None")
    except Exception as e:
        record_test("3.4 ycb_slide extract_schema", False, str(e))

    # 3.6 tacquad
    try:
        from tlabel.adapters.tacquad import TacQuadAdapter
        adapter = TacQuadAdapter()
        raw = {
            "diff_img": __import__("numpy").random.randn(120, 160, 3).astype(__import__("numpy").float32) * 5,
            "is_contact": True,
        }
        schema = adapter.extract_schema(raw)
        assert isinstance(schema, TLabelSchemaV2)
        assert schema.compliance_level == "L1", f"tacquad 应为 L1, got {schema.compliance_level}"
        assert schema.force_magnitude is None, "L1 force_magnitude 应为 None"
        assert schema.force_vector is None, "L1 force_vector 应为 None"
        record_test("3.5 tacquad extract_schema", True, "L1, force_mag=None, force_vec=None")
    except Exception as e:
        record_test("3.5 tacquad extract_schema", False, str(e))

    # 3.7 touchd 关键字段验证: force_vector 应有值
    try:
        frame = make_new_frame(frame_idx=0, contact=True)
        frame.sensor_specific = {
            "force_xyz_normalized": [0.2, -0.1, 0.7],
        }
        schema = adapter_t = ToucHDAdapter().extract_schema(frame)
        assert schema.force_vector is not None, "touchd force_vector 应有值"
        assert len(schema.force_vector) == 3
        record_test("3.6 touchd force_vector 验证", True,
                     f"force_vector={schema.force_vector}")
    except Exception as e:
        record_test("3.6 touchd force_vector 验证", False, str(e))

    # 3.8 tacquad 关键字段验证: force_vector 应为 None
    try:
        raw_tacquad = {
            "diff_img": __import__("numpy").random.randn(120, 160, 3).astype(__import__("numpy").float32) * 3,
            "is_contact": True,
        }
        schema = TacQuadAdapter().extract_schema(raw_tacquad)
        assert schema.force_vector is None, "tacquad force_vector 应为 None"
        record_test("3.7 tacquad force_vector=None 验证", True, "force_vector=None 正确")
    except Exception as e:
        record_test("3.7 tacquad force_vector=None 验证", False, str(e))


# ============================================================================
# Test 4: CLI 校验
# ============================================================================

def test_4_cli_validation():
    print("\n" + "=" * 60)
    print("Test 4: CLI 校验")
    print("=" * 60)

    from tlabel.cli import (
        _validate_schema_v2_dict, _validate_tlabel_v2_dict,
        SCHEMA_V2_DIMENSIONS, LEGACY_V2_DIMENSIONS,
    )

    # 4.1 新格式14维校验通过
    try:
        new_dict = {
            "contact": True,
            "contact_centroid": [0.5, 0.3],
            "contact_region": "digital",
            "force_magnitude": 0.5,
            "force_vector": [0.1, 0.0, 0.49],
            "torque_vector": None,
            "slip_event": False,
            "slip_velocity": None,
            "manipulation_phase": "grasp",
            "texture_class": "smooth",
            "object_deformation": 0.3,
            "temperature": None,
            "confidence": 0.9,
            "compliance_level": "L3",
        }
        results = _validate_schema_v2_dict(new_dict, "test")
        errors = [r for r in results if r.level == "error"]
        assert len(errors) == 0, f"新格式14维应校验通过, errors: {errors}"
        record_test("4.1 新格式14维校验通过", True, "0 个错误")
    except Exception as e:
        record_test("4.1 新格式14维校验通过", False, str(e))

    # 4.2 旧格式22维校验通过
    try:
        legacy_dict = make_legacy_v2_dict()
        results = _validate_tlabel_v2_dict(legacy_dict, "test")
        errors = [r for r in results if r.level == "error"]
        assert len(errors) == 0, f"旧格式22维应校验通过, errors: {errors}"
        record_test("4.2 旧格式22维校验通过", True, "0 个错误")
    except Exception as e:
        record_test("4.2 旧格式22维校验通过", False, str(e))

    # 4.3 非法数据校验失败
    try:
        # 新格式: compliance_level 非法 + confidence 超范围
        bad_dict = {
            "contact": True,
            "confidence": 1.5,
            "compliance_level": "L5",
            "force_vector": [1.0, 2.0],  # 维度错误
        }
        results = _validate_schema_v2_dict(bad_dict, "test")
        errors = [r for r in results if r.level == "error"]
        assert len(errors) >= 1, "非法数据应有错误"
        record_test("4.3 非法数据校验失败", True, f"检测到 {len(errors)} 个错误")
    except Exception as e:
        record_test("4.3 非法数据校验失败", False, str(e))

    # 4.4 _validate_tlabel_v2_dict 自动检测 Schema V2
    try:
        # 包含 Schema V2 字段 → 自动走 Schema V2 校验
        auto_dict = dict(make_legacy_v2_dict())
        auto_dict["compliance_level"] = "L2"
        auto_dict["force_vector"] = [0.1, 0.0, 0.49]
        results = _validate_tlabel_v2_dict(auto_dict, "test")
        # 应自动识别为 Schema V2
        has_v2_info = any("Schema V2" in r.message for r in results if r.level == "info")
        record_test("4.4 自动检测 Schema V2", True,
                     f"自动识别为 Schema V2: {has_v2_info}")
    except Exception as e:
        record_test("4.4 自动检测 Schema V2", False, str(e))


# ============================================================================
# Test 5: 导出 (Schema V2 only, v0.17+)
# ============================================================================

def test_5_export_v2():
    print("\n" + "=" * 60)
    print("Test 5: 导出 (Schema V2)")
    print("=" * 60)

    from tlabel.export.writer import export_data, _flatten_schema_v2

    # 5.1 CSV导出 (27列: 7基础 + 20展开)
    try:
        data = make_new_tlabel_data(n_frames=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_v2.csv")
            export_data(data, csv_path, format="csv")
            import csv as csv_mod
            with open(csv_path, "r") as f:
                reader = csv_mod.reader(f)
                headers = next(reader)
                v2_flat_cols = 20
                expected = 7 + v2_flat_cols
                assert len(headers) == expected, \
                    f"CSV应有 {expected} 列, got {len(headers)}: {headers}"
            record_test("5.1 CSV导出", True, f"{len(headers)} 列 (预期 {expected})")
    except Exception as e:
        record_test("5.1 CSV导出", False, str(e))

    # 5.2 HDF5导出 (16数值维度)
    try:
        import h5py
        data = make_new_tlabel_data(n_frames=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "test_v2.h5")
            export_data(data, h5_path, format="hdf5")
            with h5py.File(h5_path, "r") as f:
                features = f["tactile_features"]
                shape = features.shape
                assert shape[0] == 5, f"应有5帧, got {shape[0]}"
                assert shape[1] == 16, f"HDF5应有16列, got {shape[1]}"
                sv = features.attrs.get("schema_version", "")
                assert sv == "v2", f"schema_version 应为 v2, got {sv}"
            record_test("5.2 HDF5导出", True,
                         f"shape={shape}, schema={sv}")
    except ImportError:
        record_test("5.2 HDF5导出", False, "h5py 未安装，跳过")
    except Exception as e:
        record_test("5.2 HDF5导出", False, str(e))

    # 5.3 _flatten_schema_v2 验证
    try:
        from tlabel.core.schema import TLabelSchemaV2
        schema = TLabelSchemaV2(
            contact=True,
            contact_centroid=[0.5, 0.3],
            force_magnitude=0.5,
            force_vector=[0.1, 0.0, 0.49],
            slip_event=False,
            compliance_level="L3",
            object_deformation=0.3,
            confidence=0.9,
        )
        flat = _flatten_schema_v2(schema)
        assert "centroid_x" in flat and "centroid_y" in flat, "contact_centroid 应展开"
        assert "force_x" in flat and "force_y" in flat and "force_z" in flat, "force_vector 应展开"
        assert "compliance_level" in flat, "compliance_level 应保留"
        record_test("5.3 _flatten_schema_v2", True, f"展开后 {len(flat)} 个字段")
    except Exception as e:
        record_test("5.3 _flatten_schema_v2", False, str(e))

    # 5.4 JSON导出验证
    try:
        data = make_new_tlabel_data(n_frames=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test_v2.json")
            export_data(data, json_path, format="json")
            with open(json_path, "r") as f:
                loaded = json.load(f)
            assert "metadata" in loaded, "JSON应含metadata"
            assert "frames" in loaded, "JSON应含frames"
            sv = loaded.get("metadata", {}).get("schema_version", "")
            assert sv == "v2", f"schema_version 应为 v2, got {sv}"
            record_test("5.4 JSON导出", True, f"schema_version={sv}")
    except Exception as e:
        record_test("5.4 JSON导出", False, str(e))


# ============================================================================
# Test 6: 质量评分
# ============================================================================

def test_6_quality_score():
    print("\n" + "=" * 60)
    print("Test 6: 质量评分")
    print("=" * 60)

    from tlabel.quality.scorer import QualityScorer

    # 6.1 旧数据评分不报错
    try:
        data_legacy = make_legacy_tlabel_data(n_frames=20)
        scorer = QualityScorer()
        result = scorer.score(data_legacy)
        assert "overall" in result, "评分结果应包含 overall"
        assert "grade" in result, "评分结果应包含 grade"
        assert 0 <= result["overall"] <= 100, f"overall 应在 0-100, got {result['overall']}"
        record_test("6.1 旧数据评分不报错", True,
                     f"overall={result['overall']}, grade={result['grade']}")
    except Exception as e:
        record_test("6.1 旧数据评分不报错", False, str(e))

    # 6.2 新数据评分不报错
    try:
        data_new = make_new_tlabel_data(n_frames=20, compliance_level="L2")
        scorer = QualityScorer()
        result = scorer.score(data_new)
        assert "overall" in result
        assert "grade" in result
        assert 0 <= result["overall"] <= 100
        record_test("6.2 新数据评分不报错", True,
                     f"overall={result['overall']}, grade={result['grade']}")
    except Exception as e:
        record_test("6.2 新数据评分不报错", False, str(e))

    # 6.3 V2 数据评分维度正确
    try:
        data_new = make_new_tlabel_data(n_frames=20, compliance_level="L2")
        scorer = QualityScorer()
        result = scorer.score(data_new)
        assert "physical_consistency" in result
        assert "temporal_smoothness" in result
        assert "completeness" in result
        assert "coverage" in result
        record_test("6.3 V2评分4维度", True,
                     f"phys={result['physical_consistency']:.1f}, "
                     f"temp={result['temporal_smoothness']:.1f}, "
                     f"comp={result['completeness']:.1f}, "
                     f"cov={result['coverage']:.1f}")
    except Exception as e:
        record_test("6.3 V2评分4维度", False, str(e))

    # 6.4 L3数据评分（有force_vector）
    try:
        data_l3 = make_new_tlabel_data(
            n_frames=20, compliance_level="L3",
            force_vector=[0.1, 0.05, 0.5],
        )
        scorer = QualityScorer()
        result = scorer.score(data_l3)
        assert "overall" in result
        record_test("6.4 L3数据评分", True,
                     f"overall={result['overall']}, grade={result['grade']}")
    except Exception as e:
        record_test("6.4 L3数据评分", False, str(e))


# ============================================================================
# Test 7: 预测引擎
# ============================================================================

def test_7_predict_engine():
    print("\n" + "=" * 60)
    print("Test 7: 预测引擎")
    print("=" * 60)

    from tlabel.predict.engine import PredictEngine, PredictConfig

    # 7.1 旧数据 predict 不报错
    try:
        data_legacy = make_legacy_tlabel_data(n_frames=20)
        engine = PredictEngine(PredictConfig(enable_postprocess=False, enable_hmm_phase=False))
        results = engine.predict(data_legacy)
        assert len(results) == 20, f"应有20个结果, got {len(results)}"
        record_test("7.1 旧数据predict不报错", True, f"{len(results)} 帧结果")
    except Exception as e:
        record_test("7.1 旧数据predict不报错", False, str(e))

    # 7.2 新数据 predict 不报错
    try:
        data_new = make_new_tlabel_data(n_frames=20, compliance_level="L2")
        engine = PredictEngine(PredictConfig(enable_postprocess=False, enable_hmm_phase=False))
        results = engine.predict(data_new)
        assert len(results) == 20
        record_test("7.2 新数据predict不报错", True, f"{len(results)} 帧结果")
    except Exception as e:
        record_test("7.2 新数据predict不报错", False, str(e))

    # 7.3 taxonomy 规则对新旧数据都能匹配
    try:
        from tlabel.core.taxonomy import get_default_taxonomy
        taxonomy = get_default_taxonomy()

        # 旧数据
        data_legacy = make_legacy_tlabel_data(n_frames=20)
        engine = PredictEngine()
        prims_legacy = engine.predict_primitives(data_legacy, taxonomy=taxonomy)

        # 新数据
        data_new = make_new_tlabel_data(n_frames=20)
        prims_new = engine.predict_primitives(data_new, taxonomy=taxonomy)

        # 两者都应有结果（可能为空列表，但不报错）
        assert isinstance(prims_legacy, list), "旧数据primitive结果应为列表"
        assert isinstance(prims_new, list), "新数据primitive结果应为列表"
        record_test("7.3 taxonomy规则新旧数据", True,
                     f"legacy: {len(prims_legacy)} primitives, "
                     f"new: {len(prims_new)} primitives")
    except Exception as e:
        record_test("7.3 taxonomy规则新旧数据", False, str(e))


# ============================================================================
# Test 8: 数据增强
# ============================================================================

def test_8_augmentation():
    print("\n" + "=" * 60)
    print("Test 8: 数据增强")
    print("=" * 60)

    import numpy as np
    from tlabel.augment.engine import AugmentEngine

    # 8.1 16列矩阵 (Schema V2 展开后)
    try:
        features_v2 = np.random.randn(20, 16).astype(np.float32)
        augmented = AugmentEngine.augment(features_v2, ['time_warp', 'noise_inject'], seed=42)
        assert augmented.shape == features_v2.shape, \
            f"增强后形状应一致: {augmented.shape} != {features_v2.shape}"
        record_test("8.1 16列矩阵增强", True, f"shape={augmented.shape}")
    except Exception as e:
        record_test("8.1 16列矩阵增强", False, str(e))

    # 8.2 22列矩阵 (Legacy)
    try:
        features_legacy = np.random.randn(20, 22).astype(np.float32)
        augmented = AugmentEngine.augment(features_legacy, ['time_warp', 'noise_inject'], seed=42)
        assert augmented.shape == features_legacy.shape
        record_test("8.2 22列矩阵增强", True, f"shape={augmented.shape}")
    except Exception as e:
        record_test("8.2 22列矩阵增强", False, str(e))

    # 8.3 force_scale 自动检测格式
    try:
        from tlabel.augment.transforms import force_scale, SCHEMA_V2_FORCE_INDICES, LEGACY_FORCE_INDICES

        # 16列 → Schema V2 索引
        features_16 = np.ones((10, 16), dtype=np.float32)
        scaled_16 = force_scale(features_16, factor_range=(2.0, 2.0), seed=42)
        # force_magnitude(3), force_x(4), force_y(5), force_z(6) 应被缩放
        for idx in SCHEMA_V2_FORCE_INDICES:
            if idx < 16:
                assert scaled_16[0, idx] == 2.0, \
                    f"16列: force_idx={idx} 应被缩放为 2.0, got {scaled_16[0, idx]}"

        # 22列 → Legacy 索引
        features_22 = np.ones((10, 22), dtype=np.float32)
        scaled_22 = force_scale(features_22, factor_range=(2.0, 2.0), seed=42)
        for idx in LEGACY_FORCE_INDICES:
            if idx < 22:
                assert scaled_22[0, idx] == 2.0, \
                    f"22列: force_idx={idx} 应被缩放为 2.0, got {scaled_22[0, idx]}"

        record_test("8.3 force_scale格式自动检测", True,
                     f"16列用V2索引, 22列用Legacy索引")
    except Exception as e:
        record_test("8.3 force_scale格式自动检测", False, str(e))

    # 8.4 所有5种增强方法对两种格式
    try:
        methods = ['time_warp', 'noise_inject', 'random_crop', 'force_scale', 'frame_dropout']
        for n_cols in [16, 22]:
            features = np.random.randn(20, n_cols).astype(np.float32)
            for method in methods:
                try:
                    result = AugmentEngine.augment(features, [method], seed=42)
                    assert result.shape == features.shape, \
                        f"{method} on {n_cols}col: shape mismatch {result.shape} != {features.shape}"
                except Exception as me:
                    raise AssertionError(f"{method} on {n_cols}col failed: {me}")
        record_test("8.4 5种增强方法双格式", True, "所有方法对16/22列均正常")
    except Exception as e:
        record_test("8.4 5种增强方法双格式", False, str(e))


# ============================================================================
# Test 9: 转换器
# ============================================================================

def test_9_converter():
    print("\n" + "=" * 60)
    print("Test 9: 转换器")
    print("=" * 60)

    # 9.1 旧数据 tlabel_to_ftp1
    try:
        import zarr
        data_legacy = make_legacy_tlabel_data(n_frames=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test_legacy.zarr")
            from tlabel.converters.ftp1 import tlabel_to_ftp1
            result = tlabel_to_ftp1(
                data_legacy, zarr_path,
                sensor_name="GelSightMini",
                functional_areas=[0, 1],
                side="right",
                group="gripper",
                append=False,
            )
            assert "output_path" in result, "结果应包含 output_path"
            assert result["time_steps"] == 5, f"应有5个时间步, got {result['time_steps']}"
            record_test("9.1 旧数据tlabel_to_ftp1", True,
                         f"timesteps={result['time_steps']}, type={result['tactile_type']}")
    except ImportError:
        record_test("9.1 旧数据tlabel_to_ftp1", False, "zarr 未安装，跳过")
    except Exception as e:
        record_test("9.1 旧数据tlabel_to_ftp1", False, str(e))

    # 9.2 新数据 tlabel_to_ftp1
    try:
        import zarr
        data_new = make_new_tlabel_data(n_frames=5, compliance_level="L2")
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test_new.zarr")
            from tlabel.converters.ftp1 import tlabel_to_ftp1
            result = tlabel_to_ftp1(
                data_new, zarr_path,
                sensor_name="GelSightMini",
                functional_areas=[0, 1],
                side="right",
                group="gripper",
                append=False,
            )
            assert result["time_steps"] == 5
            record_test("9.2 新数据tlabel_to_ftp1", True,
                         f"timesteps={result['time_steps']}, type={result['tactile_type']}")
    except ImportError:
        record_test("9.2 新数据tlabel_to_ftp1", False, "zarr 未安装，跳过")
    except Exception as e:
        record_test("9.2 新数据tlabel_to_ftp1", False, str(e))

    # 9.3 FTP-1 降级方案中新格式优先使用 schema_v2
    try:
        import zarr
        from tlabel.converters.ftp1 import tlabel_to_ftp1

        # 新格式数据带 schema_v2，应优先使用
        data_new = make_new_tlabel_data(n_frames=3, compliance_level="L3",
                                         force_vector=[0.1, 0.05, 0.5])
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "test_priority.zarr")
            result = tlabel_to_ftp1(
                data_new, zarr_path,
                sensor_name="GelSightMini",
                functional_areas=[0, 1],
                append=False,
            )
            # 验证数据已写入
            root = zarr.open(zarr_path, mode='r')
            data_key = "right_tactile_data_gripper"
            assert data_key in root, f"应有数据键 {data_key}"
            stored = root[data_key]
            assert stored.shape[0] == 3, f"应有3帧, got {stored.shape[0]}"
            record_test("9.3 FTP-1 新格式优先schema_v2", True,
                         f"data_shape={stored.shape}")
    except ImportError:
        record_test("9.3 FTP-1 新格式优先schema_v2", False, "zarr 未安装，跳过")
    except Exception as e:
        record_test("9.3 FTP-1 新格式优先schema_v2", False, str(e))


# ============================================================================
# Test 10: 端到端流水线
# ============================================================================

def test_10_e2e_pipeline():
    print("\n" + "=" * 60)
    print("Test 10: 端到端流水线")
    print("=" * 60)

    # 10.1 旧格式完整流水线
    try:
        from tlabel.quality.scorer import QualityScorer
        from tlabel.predict.engine import PredictEngine, PredictConfig
        from tlabel.export.writer import export_data
        from tlabel.core.schema import TLabelSchemaV2

        # 1) 用旧格式创建数据
        data = make_legacy_tlabel_data(n_frames=20)

        # 2) validate — 用 to_schema_v2 逐帧验证
        all_valid = True
        for f in data.frames:
            sv2 = f.to_schema_v2()
            is_valid, errors = sv2.validate()
            if not is_valid:
                all_valid = False
                break

        # 3) predict
        engine = PredictEngine(PredictConfig(enable_postprocess=False, enable_hmm_phase=False))
        results = engine.predict(data)
        assert len(results) == 20

        # 4) quality score
        scorer = QualityScorer()
        score = scorer.score(data)
        assert "overall" in score

        # 5) export CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "e2e_legacy.csv")
            export_data(data, csv_path, format="csv")
            # 验证 CSV 可读
            import csv as csv_mod
            with open(csv_path, "r") as f:
                reader = csv_mod.reader(f)
                headers = next(reader)
                rows = list(reader)
                assert len(rows) == 20, f"CSV应有20行, got {len(rows)}"

        record_test("10.1 旧格式端到端", True,
                     f"validate={all_valid}, predict={len(results)}帧, "
                     f"quality={score['overall']:.1f}, CSV={len(rows)}行")
    except Exception as e:
        record_test("10.1 旧格式端到端", False, str(e))

    # 10.2 新格式完整流水线
    try:
        from tlabel.quality.scorer import QualityScorer
        from tlabel.predict.engine import PredictEngine, PredictConfig
        from tlabel.export.writer import export_data

        # 1) 用适配器创建数据 (模拟)
        data = make_new_tlabel_data(n_frames=20, compliance_level="L3",
                                     force_vector=[0.1, 0.05, 0.5])

        # 2) validate — 直接用 schema_v2 验证
        all_valid = True
        for f in data.frames:
            if f.schema_v2 is not None:
                is_valid, errors = f.schema_v2.validate()
                if not is_valid:
                    all_valid = False
                    break

        # 3) predict
        engine = PredictEngine(PredictConfig(enable_postprocess=False, enable_hmm_phase=False))
        results = engine.predict(data)
        assert len(results) == 20

        # 4) quality score
        scorer = QualityScorer()
        score = scorer.score(data)
        assert "overall" in score

        # 5) export CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "e2e_new.csv")
            export_data(data, csv_path, format="csv")
            import csv as csv_mod
            with open(csv_path, "r") as f:
                reader = csv_mod.reader(f)
                headers = next(reader)
                rows = list(reader)
                assert len(rows) == 20

        record_test("10.2 新格式端到端", True,
                     f"validate={all_valid}, predict={len(results)}帧, "
                     f"quality={score['overall']:.1f}, CSV={len(rows)}行")
    except Exception as e:
        record_test("10.2 新格式端到端", False, str(e))


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 60)
    print("TLabel v0.17 Phase 1-3 全流程测试")
    print("=" * 60)

    # 先检查 import 是否正常
    try:
        import tlabel
        from tlabel.core.schema import TLabelSchemaV2
        from tlabel.core.types import TLabelData, TLabelFrame
        print(f"\n✅ tlabel v{tlabel.__version__} 导入成功")
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        print("=== TLabel v0.17 Phase 1-3 全流程测试报告 ===")
        print("Import Error — 测试无法继续")
        print(f"错误: {e}")
        print("\n总结: 0/10 通过")
        print(f"发现的问题: 导入失败 - {e}")
        return

    # 执行测试
    tests = [
        test_1_schema_v2_basic,
        test_2_frame_data_integration,
        test_3_adapters_extract_schema,
        test_4_cli_validation,
        test_5_export_dual_mode,
        test_6_quality_score,
        test_7_predict_engine,
        test_8_augmentation,
        test_9_converter,
        test_10_e2e_pipeline,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            test_name = test_fn.__name__
            print(f"\n  ❌ {test_name} 整体异常: {e}")
            traceback.print_exc()
            issues_found.append(f"{test_name} 整体异常: {e}")

    # 统计
    total_tests = len(test_results)
    passed_tests = sum(1 for v in test_results.values() if v["passed"])
    failed_tests = total_tests - passed_tests

    # 生成报告
    print("\n" + "=" * 60)
    print("=== TLabel v0.17 Phase 1-3 全流程测试报告 ===")
    print("=" * 60)

    # 按 Test 组汇总
    test_groups = {}
    for name, result in test_results.items():
        group = name.split(".")[0] if "." in name else name
        # 映射到 Test X
        group_num = group
        if group_num not in test_groups:
            test_groups[group_num] = {"passed": 0, "failed": 0, "details": []}
        if result["passed"]:
            test_groups[group_num]["passed"] += 1
        else:
            test_groups[group_num]["failed"] += 1
        test_groups[group_num]["details"].append(result["details"])

    # 映射到10个Test
    test_names = {
        "1": "Schema V2 基础",
        "2": "Frame + Data 集成",
        "3": "适配器 extract_schema()",
        "4": "CLI 校验",
        "5": "导出双模式",
        "6": "质量评分",
        "7": "预测引擎",
        "8": "数据增强",
        "9": "转换器",
        "10": "端到端",
    }

    overall_pass_count = 0
    for i in range(1, 11):
        key = str(i)
        if key in test_groups:
            g = test_groups[key]
            group_passed = g["failed"] == 0
            status = "✅" if group_passed else "❌"
            detail_str = f"{g['passed']}/{g['passed'] + g['failed']} 子项通过"
            if not group_passed:
                detail_str += f" ({g['failed']} 失败)"
            print(f"Test {i}: {test_names[key]} — {status} ({detail_str})")
            if group_passed:
                overall_pass_count += 1
        else:
            print(f"Test {i}: {test_names[key]} — ⚠️ 未执行")

    print(f"\n总结: {overall_pass_count}/10 通过")

    if issues_found:
        print(f"\n发现的问题 ({len(issues_found)}):")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("\n发现的问题: 无")


if __name__ == "__main__":
    main()
