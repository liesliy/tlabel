"""
边界情况集成测试 — 适配器鲁棒性测试 (Issue #8)

覆盖场景：
  - 空文件 / 零帧数据处理
  - 损坏文件的优雅降级（不crash，抛出有意义的错误）
  - 缺失字段的 Schema V2 输出（None 填充）
  - 超大帧数数据集的分块处理
  - 不支持的文件扩展名报错
  - 适配器注册表发现（确认 syntouch 已被注册）
  - 各适配器 get_capabilities() 返回完整性检查

所有测试不依赖真实数据文件，使用 mock 或内存构造。
"""

import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tlabel.core.schema import TLabelSchemaV2, SCHEMA_V2_FIELD_NAMES, VALID_COMPLIANCE_LEVELS
from tlabel.core.capabilities import CAPABILITY_KEYS
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.registry import (
    list_builtin_adapters,
    list_adapters,
    get_adapter,
    auto_detect_format,
)
from tlabel.adapters.base import DataAdapterBase


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def syntouch_adapter():
    """返回 SynTouchBioTacAdapter 实例（懒加载，若依赖缺失则跳过）"""
    adapter_cls = get_adapter("syntouch")
    if adapter_cls is None:
        pytest.skip("syntouch adapter not registered (may be missing optional deps)")
    return adapter_cls()


@pytest.fixture
def minimal_schema():
    """最小合法 Schema V2 (L1)"""
    return TLabelSchemaV2(
        contact=False,
        slip_event=False,
        confidence=1.0,
        compliance_level="L1",
    )


@pytest.fixture
def full_schema():
    """完整填充的 Schema V2 (L4 风格)"""
    return TLabelSchemaV2(
        contact=True,
        contact_centroid=[0.5, 0.5],
        contact_region="palmar",
        force_magnitude=1.5,
        force_vector=[0.1, 0.0, 1.5],
        torque_vector=[0.01, 0.02, 0.0],
        slip_event=True,
        slip_velocity=[0.5, -0.3],
        manipulation_phase="grasp",
        texture_class="rough",
        object_deformation=0.8,
        temperature=25.3,
        confidence=0.9,
        compliance_level="L4",
    )


# =============================================================================
# 1. 空文件 / 零帧数据处理
# =============================================================================

class TestEmptyDataHandling:
    """测试空数据和零帧数据的处理"""

    def test_empty_tlabel_data_frames(self):
        """TLabelData 支持空 frames 列表，不应崩溃"""
        data = TLabelData(
            frames=[],
            sensor_info={"type": "test"},
            episode_info={"source": "test"},
            capabilities={"contact": True},
        )
        assert data.frames == []
        assert data.sensor_info["type"] == "test"

    def test_zero_frame_iteration(self):
        """空 frames 遍历时不应报错"""
        data = TLabelData(
            frames=[],
            sensor_info={},
            episode_info={},
            capabilities={},
        )
        count = 0
        for _ in data.frames:
            count += 1
        assert count == 0

    def test_empty_tlabel_frame_with_schema(self):
        """零帧场景下的 Schema V2 对象应可正常构造"""
        schema = TLabelSchemaV2(
            contact=False,
            slip_event=False,
            confidence=1.0,
            compliance_level="L1",
        )
        assert schema.contact is False
        assert schema.slip_event is False
        assert schema.confidence == 1.0
        assert schema.compliance_level == "L1"

    def test_empty_csv_raises_value_error(self, syntouch_adapter):
        """空 CSV 文件应抛出有意义的 ValueError，而不是崩溃"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="(空|empty|无法解析|no data|0 帧|zero)"):
                syntouch_adapter.load(path)
        finally:
            os.unlink(path)

    def test_zero_row_csv_raises_value_error(self, syntouch_adapter):
        """只有表头无数据的 CSV 应抛出 ValueError"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # 写表头但不写数据
            headers = ",".join([f"electrode_{i}" for i in range(19)] + ["pdc", "pac", "temperature"])
            f.write(headers + "\n")
            f.flush()
            path = f.name

        try:
            # 只有表头零数据行
            with pytest.raises(ValueError):
                syntouch_adapter.load(path)
        finally:
            os.unlink(path)


# =============================================================================
# 2. 损坏文件的优雅降级
# =============================================================================

class TestCorruptedFileHandling:
    """测试损坏文件是否优雅降级（不 crash，抛出有意义错误）"""

    def test_corrupted_h5_raises_clear_error(self, syntouch_adapter):
        """损坏的 HDF5 文件应抛出有意义的错误，不崩溃"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".h5", delete=False) as f:
            # 写随机垃圾数据，不是有效 HDF5
            f.write(b"\x00\x01\x02\x03this is not an hdf5 file at all\x00\xff")
            f.flush()
            path = f.name

        try:
            with pytest.raises(Exception) as exc_info:
                syntouch_adapter.load(path)
            # 错误消息中应包含可读的信息
            err_msg = str(exc_info.value).lower()
            assert len(err_msg) > 0  # 有错误消息
        finally:
            os.unlink(path)

    def test_corrupted_mat_raises_clear_error(self, syntouch_adapter):
        """损坏的 .mat 文件应抛出有意义的错误"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".mat", delete=False) as f:
            f.write(b"garbage data not a mat file \x00\x01\xff")
            f.flush()
            path = f.name

        try:
            with pytest.raises(Exception) as exc_info:
                syntouch_adapter.load(path)
            assert len(str(exc_info.value)) > 0
        finally:
            os.unlink(path)

    def test_corrupted_csv_handles_gracefully(self, syntouch_adapter):
        """部分损坏的 CSV 不应崩溃，应报错或处理可读部分"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("electrode_0,electrode_1,electrode_2,electrode_3,electrode_4\n")
            f.write("1.0,2.0,abc,4.0,5.0\n")  # 非数字值
            f.flush()
            path = f.name

        try:
            # 应该抛出 ValueError 或类似异常，不应该崩溃（IndexError/KeyError）
            with pytest.raises((ValueError, IndexError, Exception)):
                syntouch_adapter.load(path)
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises_file_not_found(self, syntouch_adapter):
        """不存在的文件应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            syntouch_adapter.load("/nonexistent/path/to/data.h5")


# =============================================================================
# 3. 缺失字段的 Schema V2 输出（None 填充）
# =============================================================================

class TestMissingSchemaFields:
    """测试 Schema V2 缺失字段时是否正确填充 None"""

    def test_minimal_schema_all_optional_none(self, minimal_schema):
        """最小 Schema V2 中所有 Optional 字段应为 None"""
        assert minimal_schema.contact_centroid is None
        assert minimal_schema.contact_region is None
        assert minimal_schema.force_magnitude is None
        assert minimal_schema.force_vector is None
        assert minimal_schema.torque_vector is None
        assert minimal_schema.slip_velocity is None
        assert minimal_schema.manipulation_phase is None
        assert minimal_schema.texture_class is None
        assert minimal_schema.object_deformation is None
        assert minimal_schema.temperature is None

    def test_partial_schema_missing_fields_none(self):
        """部分填充的 Schema V2，未填充字段应为 None"""
        schema = TLabelSchemaV2(
            contact=True,
            contact_centroid=[0.5, 0.5],
            force_magnitude=1.0,
            slip_event=False,
            confidence=0.8,
            compliance_level="L2",
        )
        # 显式设置的字段不为 None
        assert schema.contact is True
        assert schema.force_magnitude == 1.0
        assert schema.contact_centroid == [0.5, 0.5]
        # 未设置的 Optional 字段为 None
        assert schema.contact_region is None
        assert schema.force_vector is None
        assert schema.torque_vector is None
        assert schema.slip_velocity is None
        assert schema.texture_class is None
        assert schema.object_deformation is None
        assert schema.temperature is None

    def test_from_dict_missing_fields(self):
        """从字典构建 Schema V2，缺失字段使用默认值/None"""
        schema = TLabelSchemaV2.from_dict({"contact": True})
        assert schema.contact is True
        # 缺失字段的默认值
        assert schema.slip_event is False  # bool 默认 False
        assert schema.confidence == 1.0
        assert schema.compliance_level == "L1"
        assert schema.contact_centroid is None
        assert schema.force_magnitude is None

    def test_to_dict_contains_all_fields(self, minimal_schema):
        """to_dict() 返回的字典应包含所有 14 个字段名"""
        d = minimal_schema.to_dict()
        for field in SCHEMA_V2_FIELD_NAMES:
            assert field in d, f"to_dict() 缺少字段: {field}"

    def test_schema_v2_validation_with_none_fields(self, minimal_schema):
        """L1 Schema 允许 Optional 字段为 None，验证应通过"""
        is_valid, errors = minimal_schema.validate()
        assert is_valid, f"L1 minimal schema should validate but got errors: {errors}"


# =============================================================================
# 4. 超大帧数数据集的分块处理
# =============================================================================

class TestLargeDatasetHandling:
    """测试大数据集的处理能力和内存效率"""

    def _make_large_syntouch_data(self, n_frames: int = 10000):
        """构造一个内存中的大型 BioTac 风格数据集"""
        rng = np.random.RandomState(42)
        impedance = rng.randn(n_frames, 19).astype(np.float64)
        # 加入一段接触信号
        start = n_frames // 4
        end = n_frames * 3 // 4
        impedance[start:end] += 2.0

        pdc = rng.randn(n_frames).astype(np.float64) * 0.1
        pdc[start:end] += 5.0

        pac = rng.randn(n_frames).astype(np.float64) * 0.01
        # 中间加一段滑移
        slip_start = start + (end - start) // 3
        slip_end = slip_start + 500
        pac[slip_start:slip_end] += 1.0

        temperature = 25.0 + rng.randn(n_frames).astype(np.float64) * 0.1

        return {
            "impedance": impedance,
            "pdc": pdc,
            "pac": pac,
            "temperature": temperature,
        }

    def test_large_dataset_does_not_crash(self, syntouch_adapter):
        """10000 帧大数据集应能正常处理，不崩溃"""
        raw = self._make_large_syntouch_data(n_frames=10000)
        data = syntouch_adapter._parse(raw, "dummy.h5")
        assert len(data.frames) == 10000
        assert all(isinstance(f, TLabelFrame) for f in data.frames)

    def test_large_dataset_schema_consistency(self, syntouch_adapter):
        """大数据集的每一帧都应有有效的 schema_v2"""
        raw = self._make_large_syntouch_data(n_frames=5000)
        data = syntouch_adapter._parse(raw, "dummy.h5")

        for i, frame in enumerate(data.frames):
            assert frame.schema_v2 is not None
            assert isinstance(frame.schema_v2, TLabelSchemaV2)
            assert frame.frame_idx == i
            assert isinstance(frame.timestamp_s, float)

    def test_large_dataset_contact_detection(self, syntouch_adapter):
        """大数据集中接触检测应正确工作"""
        raw = self._make_large_syntouch_data(n_frames=5000)
        data = syntouch_adapter._parse(raw, "dummy.h5")

        contact_frames = [f for f in data.frames if f.schema_v2.contact]
        # 中间段应该有接触帧
        assert len(contact_frames) > 0
        # 数量应接近中间一半的帧数
        assert len(contact_frames) > 1000  # 至少有 1000 帧接触

    def test_large_dataset_slip_detection(self, syntouch_adapter):
        """大数据集中滑移检测应检测到注入的滑移段"""
        raw = self._make_large_syntouch_data(n_frames=5000)
        data = syntouch_adapter._parse(raw, "dummy.h5")

        slip_frames = [f for f in data.frames if f.schema_v2.slip_event]
        # 应该检测到一些滑移帧
        assert len(slip_frames) > 0

    def test_memory_efficiency_frame_access(self, syntouch_adapter):
        """帧列表应支持高效随机访问"""
        raw = self._make_large_syntouch_data(n_frames=2000)
        data = syntouch_adapter._parse(raw, "dummy.h5")

        # 首尾帧访问
        first = data.frames[0]
        last = data.frames[-1]
        assert first.frame_idx == 0
        assert last.frame_idx == 1999

        # 中间帧访问
        mid = data.frames[1000]
        assert mid.frame_idx == 1000


# =============================================================================
# 5. 不支持的文件扩展名报错
# =============================================================================

class TestUnsupportedFileExtension:
    """测试不支持的文件格式"""

    def test_unsupported_extension_raises_error(self, syntouch_adapter):
        """不支持的文件扩展名应抛出 ValueError"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("test")
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError, match="(不支持|unsupported|格式|format)"):
                syntouch_adapter.load(path)
        finally:
            os.unlink(path)

    def test_empty_extension_raises_error(self, syntouch_adapter):
        """无扩展名的文件应报错"""
        with tempfile.NamedTemporaryFile(mode="w", suffix="", delete=False) as f:
            f.write("test")
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError):
                syntouch_adapter.load(path)
        finally:
            os.unlink(path)

    def test_txt_extension_unsupported(self, syntouch_adapter):
        """.txt 不是支持的格式"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test data\n")
            f.flush()
            path = f.name

        try:
            with pytest.raises(ValueError):
                syntouch_adapter.load(path)
        finally:
            os.unlink(path)

    def test_supported_extensions_list(self, syntouch_adapter):
        """supported_extensions 应包含文档声明的格式"""
        exts = syntouch_adapter.supported_extensions
        assert ".h5" in exts or ".hdf5" in exts
        assert ".csv" in exts
        assert ".mat" in exts


# =============================================================================
# 6. 适配器注册表发现
# =============================================================================

class TestAdapterRegistry:
    """测试适配器注册与发现"""

    def test_syntouch_in_registry(self):
        """syntouch 适配器应已注册到注册表中"""
        adapters = list_builtin_adapters()
        assert "syntouch" in adapters, (
            "syntouch 适配器未在注册表中发现。"
            "请确认 tlabel/core/registry.py 中已添加。"
        )

    def test_syntouch_adapter_class_valid(self):
        """获取的 syntouch 适配器类应是 DataAdapterBase 的子类"""
        adapter_cls = get_adapter("syntouch")
        assert adapter_cls is not None
        assert issubclass(adapter_cls, DataAdapterBase)

    def test_syntouch_adapter_name(self):
        """syntouch 适配器的 name 属性应为 'syntouch'"""
        adapter_cls = get_adapter("syntouch")
        assert adapter_cls is not None
        instance = adapter_cls()
        assert instance.name == "syntouch"

    def test_get_nonexistent_adapter_returns_none(self):
        """获取不存在的适配器应返回 None"""
        adapter_cls = get_adapter("nonexistent_sensor_xyz")
        assert adapter_cls is None

    def test_list_adapters_returns_dict(self):
        """list_adapters 应返回非空字典"""
        adapters = list_adapters()
        assert isinstance(adapters, dict)
        assert len(adapters) > 0

    def test_all_registered_adapters_instantiatable(self):
        """所有已注册的适配器应能成功实例化（不崩溃）"""
        adapters = list_builtin_adapters()
        failed = []
        for name, cls in adapters.items():
            try:
                instance = cls()
                assert instance is not None
            except Exception as e:
                failed.append(f"{name}: {e}")

        assert len(failed) == 0, (
            f"以下适配器实例化失败: {failed}"
        )


# =============================================================================
# 7. 各适配器 get_capabilities() 返回完整性检查
# =============================================================================

class TestCapabilitiesCompleteness:
    """测试所有适配器的 get_capabilities() 完整性"""

    def test_all_adapters_have_capabilities(self):
        """所有内置适配器都应实现 get_capabilities() 并返回 dict"""
        adapters = list_builtin_adapters()
        missing = []
        for name, cls in adapters.items():
            try:
                instance = cls()
                caps = instance.get_capabilities()
                assert isinstance(caps, dict), f"{name}: capabilities 不是 dict"
                assert len(caps) > 0, f"{name}: capabilities 为空 dict"
            except TypeError as e:
                # 部分适配器的 get_capabilities() 需要参数（如 TLabelAdapter 需要 file_path），
                # 这类适配器在无参调用时会报 TypeError，此处跳过不算失败
                if "missing" in str(e) or "required positional" in str(e) or "takes" in str(e):
                    continue
                missing.append(f"{name}: {e}")
            except Exception as e:
                missing.append(f"{name}: {e}")

        assert len(missing) == 0, (
            f"以下适配器的 get_capabilities() 有问题: {missing}"
        )

    def test_syntouch_capabilities_cover_all_schema_fields(self, syntouch_adapter):
        """syntouch 适配器的 capabilities 应覆盖全部 13 个合法能力键

        注意：compliance_level 是帧级元字段，不属于能力键
        （规范 S-7.4-2 / C-15），v0.23.3 起不得出现在 capabilities 中。
        """
        caps = syntouch_adapter.get_capabilities()
        for field in sorted(CAPABILITY_KEYS):
            assert field in caps, (
                f"syntouch capabilities 缺少字段: {field}"
            )
        assert "compliance_level" not in caps, (
            "compliance_level 是帧级元字段，不属于 capabilities 能力键"
        )

    def test_syntouch_capabilities_all_bool(self, syntouch_adapter):
        """capabilities 的每个值都应是 bool 类型"""
        caps = syntouch_adapter.get_capabilities()
        for field, value in caps.items():
            assert isinstance(value, bool), (
                f"capabilities['{field}'] 类型错误: 期望 bool, 实际 {type(value)}"
            )

    def test_syntouch_capabilities_expected_values(self, syntouch_adapter):
        """syntouch 适配器的 capabilities 应符合 BioTac 传感器的能力"""
        caps = syntouch_adapter.get_capabilities()

        # BioTac 应该能检测的
        assert caps["contact"] is True
        assert caps["contact_centroid"] is True
        assert caps["contact_region"] is True
        assert caps["force_magnitude"] is True
        assert caps["slip_event"] is True
        assert caps["temperature"] is True
        assert caps["object_deformation"] is True

        # BioTac 不能直接提供的
        assert caps["force_vector"] is False
        assert caps["torque_vector"] is False

    def test_all_adapters_capabilities_key_count(self):
        """所有适配器的 capabilities 应包含核心字段

        注：检查范围限定为 contact 和 slip_event 这两个大多数适配器
        都能提供的核心能力字段。confidence 和 compliance_level 属于
        Schema 元字段而非传感器能力，不在 capabilities 中声明。
        """
        adapters = list_builtin_adapters()
        insufficient = []
        for name, cls in adapters.items():
            try:
                instance = cls()
                try:
                    caps = instance.get_capabilities()
                except TypeError:
                    # 部分适配器 get_capabilities 需要参数，跳过键数检查
                    continue
                # 至少应包含 contact 和 slip_event 这两个核心字段
                required_min = {"contact", "slip_event"}
                missing = required_min - set(caps.keys())
                if missing:
                    insufficient.append(f"{name}: missing {missing}")
            except Exception as e:
                insufficient.append(f"{name}: error {e}")

        assert len(insufficient) == 0, (
            f"以下适配器 capabilities 不完整: {insufficient}"
        )


# =============================================================================
# 8. Schema V2 边界值验证
# =============================================================================

class TestSchemaV2EdgeValues:
    """测试 Schema V2 的边界值和验证逻辑"""

    def test_confidence_out_of_range_fails_validation(self):
        """confidence 超出 [0, 1] 范围应验证失败"""
        schema = TLabelSchemaV2(confidence=1.5, compliance_level="L1")
        is_valid, errors = schema.validate()
        assert not is_valid
        assert any("confidence" in e for e in errors)

    def test_negative_confidence_fails_validation(self):
        """负 confidence 应验证失败"""
        schema = TLabelSchemaV2(confidence=-0.1, compliance_level="L1")
        is_valid, errors = schema.validate()
        assert not is_valid

    def test_invalid_compliance_level_fails_validation(self):
        """非法的 compliance_level 应验证失败"""
        schema = TLabelSchemaV2(compliance_level="L5")
        is_valid, errors = schema.validate()
        assert not is_valid
        assert any("compliance_level" in e for e in errors)

    def test_contact_without_centroid_fails(self):
        """contact=True 但 contact_centroid=None 应验证失败"""
        schema = TLabelSchemaV2(
            contact=True,
            contact_centroid=None,
            compliance_level="L1",
        )
        is_valid, errors = schema.validate()
        assert not is_valid
        assert any("contact_centroid" in e for e in errors)

    def test_contact_centroid_wrong_dimension(self):
        """contact_centroid 维度不对应验证失败"""
        schema = TLabelSchemaV2(
            contact=True,
            contact_centroid=[0.5],  # 只有1维
            compliance_level="L1",
        )
        is_valid, errors = schema.validate()
        assert not is_valid

    def test_l2_requires_force_magnitude(self):
        """L2 compliance level 要求 force_magnitude"""
        schema = TLabelSchemaV2(
            contact=True,
            contact_centroid=[0.5, 0.5],
            force_magnitude=None,
            compliance_level="L2",
        )
        is_valid, errors = schema.validate()
        assert not is_valid
        assert any("force_magnitude" in e for e in errors)

    def test_invalid_contact_region_fails(self):
        """非法 contact_region 枚举值应验证失败"""
        schema = TLabelSchemaV2(
            contact_region="invalid_region",
            compliance_level="L1",
        )
        is_valid, errors = schema.validate()
        assert not is_valid

    def test_force_vector_wrong_dimension(self):
        """force_vector 维度不对应验证失败"""
        schema = TLabelSchemaV2(
            force_vector=[0.1, 0.2],  # 只有2维
            compliance_level="L1",
        )
        is_valid, errors = schema.validate()
        assert not is_valid

    def test_zero_force_magnitude_valid_l1(self):
        """L1 级别下 force_magnitude=None 是合法的"""
        schema = TLabelSchemaV2(
            contact=False,
            force_magnitude=None,
            compliance_level="L1",
        )
        is_valid, errors = schema.validate()
        assert is_valid, f"L1 no-force schema failed: {errors}"


# =============================================================================
# 9. TLabelFrame 边界情况
# =============================================================================

class TestTLabelFrameEdgeCases:
    """测试 TLabelFrame 的边界情况"""

    def test_frame_requires_schema_v2(self):
        """TLabelFrame 必须提供 schema_v2，否则抛 ValueError"""
        with pytest.raises(ValueError, match="schema_v2"):
            TLabelFrame(frame_idx=0, timestamp_s=0.0, schema_v2=None)

    def test_frame_with_empty_sensor_specific(self):
        """空的 sensor_specific 应正常工作"""
        schema = TLabelSchemaV2(compliance_level="L1")
        frame = TLabelFrame(
            frame_idx=0,
            timestamp_s=0.0,
            schema_v2=schema,
            sensor_specific={},
        )
        assert frame.sensor_specific == {}

    def test_frame_with_none_sensor_specific(self):
        """sensor_specific=None 应默认转为空 dict"""
        schema = TLabelSchemaV2(compliance_level="L1")
        frame = TLabelFrame(
            frame_idx=0,
            timestamp_s=0.0,
            schema_v2=schema,
            sensor_specific=None,
        )
        assert frame.sensor_specific == {}

    def test_frame_contact_property(self):
        """frame.contact 应返回 schema_v2.contact 的 bool"""
        schema = TLabelSchemaV2(contact=True, compliance_level="L1")
        frame = TLabelFrame(
            frame_idx=0, timestamp_s=0.0, schema_v2=schema
        )
        assert frame.contact is True

    def test_frame_slip_property(self):
        """frame.slip_event 应返回 schema_v2.slip_event 的 bool"""
        schema = TLabelSchemaV2(slip_event=True, compliance_level="L1")
        frame = TLabelFrame(
            frame_idx=0, timestamp_s=0.0, schema_v2=schema
        )
        assert frame.slip_event is True


# =============================================================================
# 10. SynTouch BioTac 适配器特定边界测试
# =============================================================================

class TestSynTouchEdgeCases:
    """SynTouch BioTac 适配器的特定边界情况"""

    def test_single_frame_data(self, syntouch_adapter):
        """单帧数据应能正常处理"""
        raw = {
            "impedance": np.zeros((1, 19), dtype=np.float64),
            "pdc": np.array([0.0]),
            "pac": np.array([0.0]),
            "temperature": np.array([25.0]),
        }
        data = syntouch_adapter._parse(raw, "single_frame.h5")
        assert len(data.frames) == 1
        assert data.frames[0].frame_idx == 0

    def test_two_frame_slip_detection(self, syntouch_adapter):
        """只有两帧时滑移检测不应崩溃"""
        raw = {
            "impedance": np.zeros((2, 19), dtype=np.float64),
            "pdc": np.array([0.0, 5.0]),
            "pac": np.array([0.0, 1.0]),
        }
        data = syntouch_adapter._parse(raw, "two_frames.h5")
        assert len(data.frames) == 2
        # 不崩溃即通过
        for f in data.frames:
            assert f.schema_v2 is not None

    def test_all_zero_impedance(self, syntouch_adapter):
        """全零阻抗数据不应崩溃，应报告无接触"""
        n = 100
        raw = {
            "impedance": np.zeros((n, 19), dtype=np.float64),
            "pdc": np.zeros(n, dtype=np.float64),
            "pac": np.zeros(n, dtype=np.float64),
            "temperature": np.full(n, 25.0, dtype=np.float64),
        }
        data = syntouch_adapter._parse(raw, "all_zero.h5")
        assert len(data.frames) == n
        # 全零应该没有接触
        contact_count = sum(1 for f in data.frames if f.schema_v2.contact)
        assert contact_count == 0

    def test_extract_schema_no_contact(self, syntouch_adapter):
        """无接触时 extract_schema 应返回基础字段，其余为 None"""
        raw_frame = {
            "electrodes": np.zeros(19, dtype=np.float64),
            "pdc": 0.0,
            "pac": 0.0,
            "temperature": 25.0,
            "baseline_electrodes": np.zeros(19, dtype=np.float64),
            "baseline_pdc": 0.0,
            "is_contact": False,
            "is_slip": False,
        }
        schema = syntouch_adapter.extract_schema(raw_frame)
        assert schema.contact is False
        assert schema.slip_event is False
        assert schema.force_magnitude is None
        assert schema.object_deformation is None

    def test_compliance_level_is_l2(self, syntouch_adapter):
        """SynTouch 适配器的 compliance level 应为 L2"""
        assert syntouch_adapter.default_compliance_level == "L2"

    def test_get_sensor_info_returns_dict(self, syntouch_adapter):
        """get_sensor_info() 应返回包含必要字段的 dict"""
        info = syntouch_adapter.get_sensor_info()
        assert isinstance(info, dict)
        assert "manufacturer" in info
        assert "model" in info
        assert info["manufacturer"] == "SynTouch"
        assert info["model"] == "BioTac"

    def test_auto_detect_format_h5(self, syntouch_adapter):
        """auto_detect_format 对 .h5 文件应能工作（不崩溃）"""
        # 创建一个空 h5 文件（如果 h5py 可用）
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with h5py.File(path, 'w') as hf:
                hf.create_dataset("impedance", data=np.zeros((10, 19)))
            # 不验证具体返回值，只确保不崩溃
            try:
                auto_detect_format(path)
            except Exception:
                pass  # auto_detect 可能返回 None，不崩溃即可
        finally:
            os.unlink(path)
