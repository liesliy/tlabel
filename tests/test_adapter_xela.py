"""
XELA uSkin 适配器测试 — UniTac-NV CSV 格式 (Issue #5)

覆盖场景：
  - UniTac-NV 风格 CSV 的正常加载（表头别名 / 无表头按位置 / 扁平 taxel 格式）
  - 多帧、多 taxel 力/位移数据的解析与 sensor_specific 原始保留
  - 接触检测与归一化接触质心
  - 单位未文档化字段的 None 填充（不填 0 或虚构单位）
  - force_scale 标定路径（force_vector / force_magnitude 填充，L3 升级）
  - Schema V2 逐帧校验与导出 JSON 的 JSON Schema 合规
  - 错误数组形状 / malformed CSV / 空文件 / 缺失列 / 空单元格
  - 注册表发现（内部注册，非 external entry-point）

所有数据为测试内构造的 synthetic fixture，不依赖真实数据文件与硬件。
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tlabel.adapters.base import DataAdapterBase
from tlabel.core.registry import (
    get_adapter,
    list_builtin_adapters,
    list_external_adapters,
)
from tlabel.core.schema import TLabelSchemaV2
from tlabel.core.types import TLabelData, TLabelFrame

REPO_ROOT = Path(__file__).resolve().parent.parent
JSON_SCHEMA_PATH = REPO_ROOT / "schema" / "tlabel-schema.json"

# UniTac-NV 数据集的表头命名（用户提供的字段名）
XELA_HEADER = ["time", "seq", "sensor_matrices_force",
               "sensor_matrices_displacement", "ft", "end_effector_pose"]


# =============================================================================
# Fixtures 与 synthetic 数据构造
# =============================================================================

@pytest.fixture
def xela_adapter():
    """返回 XelaUskinAdapter 实例（先触发注册表懒加载；未注册则跳过）"""
    # get_adapter() 是纯字典查询，需先通过 list_* 触发懒加载注册
    list_builtin_adapters()
    adapter_cls = get_adapter("xela")
    if adapter_cls is None:
        pytest.skip("xela adapter not registered")
    return adapter_cls()


def _matrix_cell(arr) -> str:
    """将矩阵编码为 CSV 单元格（Python 字面量嵌套列表，与 UniTac-NV 一致）"""
    return str(np.asarray(arr).tolist())


def _recording(n_frames=15, press_frames=(), amplitude=1000.0,
               taxel=(2, 3), rows=4, cols=6):
    """构造合成录制：基线零帧 + 指定帧在指定 taxel 施加 Z 法向力"""
    forces = []
    for i in range(n_frames):
        f = np.zeros((rows, cols, 3))
        if i in press_frames:
            f[taxel[0], taxel[1], 2] = amplitude
        forces.append(f)
    return forces


def _write_recording(path, forces, header=XELA_HEADER,
                     displacements=None, fts=None, poses=None):
    """写入 UniTac-NV 风格 CSV（未提供的辅助列写空单元格）"""
    n = len(forces)
    base = datetime(2026, 1, 1, 12, 0, 0)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header is not None:
            w.writerow(header)
        for i in range(n):
            t = (base + timedelta(seconds=0.01 * i)).strftime(
                "%Y-%m-%d %H:%M:%S.%f")
            row = [t, str(i), _matrix_cell(forces[i])]
            row.append(_matrix_cell(displacements[i])
                       if displacements is not None else "")
            row.append(str(list(fts[i])) if fts is not None else "")
            row.append(str(list(poses[i])) if poses is not None else "")
            w.writerow(row)


def _write_rows(path, rows, header=None):
    """写入原始 CSV 行（用于 malformed / 自定义列测试）"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header is not None:
            w.writerow(header)
        for row in rows:
            w.writerow(row)


# =============================================================================
# 1. 正常 CSV 加载
# =============================================================================

class TestXelaCsvLoading:
    """UniTac-NV 风格 CSV 的正常加载"""

    def test_load_with_header(self, xela_adapter, tmp_path):
        """带表头的标准 CSV 应正确加载为 TLabelData"""
        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(press_frames=(10, 11, 12)))

        data = xela_adapter.load(str(path))

        assert isinstance(data, TLabelData)
        assert data.num_frames == 15
        assert all(isinstance(f, TLabelFrame) for f in data.frames)
        assert [f.frame_idx for f in data.frames] == list(range(15))
        assert data.sensor_id == "xela_uskin_0"
        assert data.episode_info["source"] == "xela_uskin_univtac_nv"
        assert data.episode_info["taxel_layout"] == [4, 6]
        assert data.episode_info["num_taxels"] == 24
        assert data.sensor_info["sensor_type"] == "distributed_array"
        assert data.capabilities == xela_adapter.get_capabilities()

    def test_load_headerless_positional(self, xela_adapter, tmp_path):
        """无表头 CSV 应按列位置解析（与 UniTac-NV 上游加载器一致）"""
        path = tmp_path / "xela_noheader.csv"
        _write_recording(str(path), _recording(press_frames=(10, 11, 12)),
                         header=None)

        data = xela_adapter.load(str(path))

        assert data.num_frames == 15
        contact_frames = [f.frame_idx for f in data.frames if f.contact]
        assert contact_frames == [10, 11, 12]

    def test_timestamps_relative_to_first_frame(self, xela_adapter, tmp_path):
        """时间戳应转换为相对首帧的秒数（100 Hz → 0.01s 步进）"""
        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(n_frames=5))

        data = xela_adapter.load(str(path))

        assert data.frames[0].timestamp_s == 0.0
        assert data.frames[1].timestamp_s == pytest.approx(0.01, abs=1e-6)
        assert data.frames[4].timestamp_s == pytest.approx(0.04, abs=1e-6)
        assert data.episode_info["sample_rate_hz"] == pytest.approx(100.0)

    def test_multi_frame_multi_taxel_values_preserved(self, xela_adapter, tmp_path):
        """多帧多 taxel 力/位移数据应逐值保留在 sensor_specific"""
        forces = _recording(n_frames=12, press_frames=(5,))
        # 施加两个不同 taxel、不同轴的力: (1,2) X 剪切 500, (3,0) Z 法向 -800
        forces[5][1, 2, 0] = 500.0
        forces[5][3, 0, 2] = -800.0
        displacements = [np.zeros((4, 6, 3)) for _ in range(12)]
        displacements[5][1, 2, 1] = 0.5
        fts = [[0.1 * i, -0.2, 1.5, 0.0, 0.0, 0.01] for i in range(12)]
        poses = [[0.1, 0.2, 0.3, 0.0, 3.14, 0.0]] * 12

        path = tmp_path / "xela.csv"
        _write_recording(str(path), forces, displacements=displacements,
                         fts=fts, poses=poses)

        data = xela_adapter.load(str(path))

        ss = data.frames[5].sensor_specific
        # taxel_force: 24 taxel × 3 轴，扁平化顺序 = 行主序 (r*6+c)
        assert len(ss["taxel_force"]) == 24
        assert ss["taxel_force"][1 * 6 + 2] == [500.0, 0.0, 0.0]
        assert ss["taxel_force"][3 * 6 + 0] == [0.0, 0.0, -800.0]
        assert ss["taxel_force"][0] == [0.0, 0.0, 0.0]
        # 位移同样保留
        assert ss["taxel_displacement"][1 * 6 + 2] == [0.0, 0.5, 0.0]
        # 辅助字段保留
        assert ss["seq"] == 5
        assert ss["ft_ground_truth"] == [0.5, -0.2, 1.5, 0.0, 0.0, 0.01]
        assert ss["end_effector_pose"] == [0.1, 0.2, 0.3, 0.0, 3.14, 0.0]
        # 非施压帧不受影响
        assert data.frames[0].sensor_specific["taxel_force"][8] == [0.0, 0.0, 0.0]

    def test_single_frame_loads(self, xela_adapter, tmp_path):
        """单帧 CSV 应能正常处理"""
        path = tmp_path / "xela_single.csv"
        _write_recording(str(path), [_recording(n_frames=1, press_frames=())[0]])

        data = xela_adapter.load(str(path))

        assert data.num_frames == 1
        assert data.frames[0].frame_idx == 0
        assert data.frames[0].schema_v2 is not None

    def test_flat_taxel_list_format(self, xela_adapter, tmp_path):
        """扁平 (24,3) taxel 列表（UniTac-NV 预处理输出格式）应可加载"""
        n = 12
        forces = []
        for i in range(n):
            f = np.zeros((24, 3))
            if i in (10, 11):
                f[15, 2] = 1000.0  # flat idx 15 = (2,3) in 4x6 网格
            forces.append(f)

        path = tmp_path / "xela_flat.csv"
        _write_recording(str(path), forces)

        data = xela_adapter.load(str(path))

        assert data.num_frames == n
        assert data.episode_info["taxel_layout"] == [24]
        assert data.episode_info["num_taxels"] == 24
        contact_frames = [f.frame_idx for f in data.frames if f.contact]
        assert contact_frames == [10, 11]
        # 扁平布局按单行排列：x=15/23, y=0.5
        centroid = data.frames[10].schema_v2.contact_centroid
        assert centroid[0] == pytest.approx(15.0 / 23.0, abs=1e-3)
        assert centroid[1] == pytest.approx(0.5, abs=1e-3)


# =============================================================================
# 2. 接触检测与质心
# =============================================================================

class TestContactDetection:
    """接触判定与归一化接触质心"""

    def test_press_frames_detected(self, xela_adapter, tmp_path):
        """施压帧应被判定为接触，基线帧不接触"""
        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(press_frames=(10, 11, 12)))

        data = xela_adapter.load(str(path))

        contact_frames = [f.frame_idx for f in data.frames if f.contact]
        assert contact_frames == [10, 11, 12]
        assert data.episode_info["contact_frames"] == 3

    def test_baseline_only_no_contact(self, xela_adapter, tmp_path):
        """全零基线数据应无任何接触帧"""
        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(press_frames=()))

        data = xela_adapter.load(str(path))

        assert sum(1 for f in data.frames if f.contact) == 0

    def test_noisy_baseline_auto_threshold(self, xela_adapter, tmp_path):
        """小幅基线噪声不应误报接触（自动阈值下限保护）"""
        forces = _recording(press_frames=())
        for i, f in enumerate(forces):
            f[0, 0, 0] = 0.02 if i % 2 == 0 else -0.02  # 偏差 0.02 < 0.1

        path = tmp_path / "xela.csv"
        _write_recording(str(path), forces)

        data = xela_adapter.load(str(path))

        assert sum(1 for f in data.frames if f.contact) == 0

    def test_centroid_at_pressed_taxel(self, xela_adapter, tmp_path):
        """接触质心应落在受压 taxel 的归一化网格坐标"""
        # taxel (0,0) → [0, 0]
        path = tmp_path / "corner.csv"
        _write_recording(str(path),
                         _recording(press_frames=(10,), taxel=(0, 0)))
        data = xela_adapter.load(str(path))
        assert data.frames[10].schema_v2.contact_centroid == [0.0, 0.0]

        # taxel (3,5) → [1, 1]
        path = tmp_path / "corner2.csv"
        _write_recording(str(path),
                         _recording(press_frames=(10,), taxel=(3, 5)))
        data = xela_adapter.load(str(path))
        assert data.frames[10].schema_v2.contact_centroid == [1.0, 1.0]

        # taxel (2,3) → [3/5, 2/3]
        path = tmp_path / "center.csv"
        _write_recording(str(path),
                         _recording(press_frames=(10,), taxel=(2, 3)))
        data = xela_adapter.load(str(path))
        centroid = data.frames[10].schema_v2.contact_centroid
        assert centroid[0] == pytest.approx(0.6, abs=1e-3)
        assert centroid[1] == pytest.approx(2.0 / 3.0, abs=1e-3)

    def test_centroid_weighted_average(self, xela_adapter, tmp_path):
        """两个等幅受压 taxel 的质心应在两者中点"""
        forces = _recording(press_frames=())
        forces[10][0, 0, 2] = 800.0
        forces[10][0, 5, 2] = 800.0

        path = tmp_path / "xela.csv"
        _write_recording(str(path), forces)

        data = xela_adapter.load(str(path))

        centroid = data.frames[10].schema_v2.contact_centroid
        assert centroid[0] == pytest.approx(0.5, abs=1e-3)
        assert centroid[1] == pytest.approx(0.0, abs=1e-3)

    def test_centroid_within_unit_range(self, xela_adapter, tmp_path):
        """质心坐标必须归一化到 [0, 1]"""
        forces = _recording(press_frames=(10,))
        forces[10][0, 2, 0] = 300.0
        forces[10][3, 4, 1] = 900.0
        forces[10][2, 2, 2] = 600.0

        path = tmp_path / "xela.csv"
        _write_recording(str(path), forces)

        data = xela_adapter.load(str(path))

        centroid = data.frames[10].schema_v2.contact_centroid
        assert 0.0 <= centroid[0] <= 1.0
        assert 0.0 <= centroid[1] <= 1.0

    def test_custom_contact_threshold(self, xela_adapter, tmp_path):
        """自定义接触阈值应生效"""
        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(press_frames=(10,)))

        # 阈值高于施压幅值 → 无接触
        data_strict = xela_adapter.load(str(path), contact_threshold=2000.0)
        assert sum(1 for f in data_strict.frames if f.contact) == 0

        # 阈值低于施压幅值 → 有接触
        data_loose = xela_adapter.load(str(path), contact_threshold=500.0)
        assert sum(1 for f in data_loose.frames if f.contact) == 1


# =============================================================================
# 3. Schema 输出与合规
# =============================================================================

class TestSchemaOutput:
    """Schema V2 输出的字段填充、单位诚实性与合规校验"""

    def test_unsupported_fields_are_none_not_zero(self, xela_adapter, tmp_path):
        """单位无法可靠映射的字段应为 None，而不是 0 或虚构值"""
        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(press_frames=(10,)))

        data = xela_adapter.load(str(path))
        sv2 = data.frames[10].schema_v2  # 接触帧

        # 可靠字段已填充
        assert sv2.contact is True
        assert sv2.contact_centroid is not None
        # 无法可靠映射的字段为 None（不是 0）
        assert sv2.force_magnitude is None
        assert sv2.force_vector is None
        assert sv2.torque_vector is None
        assert sv2.slip_velocity is None
        assert sv2.contact_region is None
        assert sv2.texture_class is None
        assert sv2.object_deformation is None
        assert sv2.temperature is None
        assert sv2.manipulation_phase is None
        # slip_event 是 Required bool 字段，保持 False 而非 None
        assert sv2.slip_event is False

    def test_default_compliance_level_is_l1(self, xela_adapter, tmp_path):
        """数据集未文档化力单位，默认 compliance level 应为 L1"""
        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(press_frames=(10,)))

        data = xela_adapter.load(str(path))

        assert xela_adapter.default_compliance_level == "L1"
        assert all(f.schema_v2.compliance_level == "L1" for f in data.frames)

    def test_force_scale_fills_force_fields(self, xela_adapter, tmp_path):
        """提供 force_scale 标定系数时应填充力字段并升级为 L3"""
        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(press_frames=(10,),
                                               amplitude=1000.0))

        data = xela_adapter.load(str(path), force_scale=0.001)

        # 接触帧: 单 taxel Z 法向 1000 raw × 0.001 = 1.0 N
        sv2 = data.frames[10].schema_v2
        assert sv2.force_vector == [0.0, 0.0, 1.0]
        assert sv2.force_magnitude == pytest.approx(1.0)
        assert sv2.compliance_level == "L3"
        assert sv2.contact_centroid is not None
        # 标定信息记录
        assert data.calibration_params == {"force_scale": 0.001}

    def test_force_scale_no_contact_frames(self, xela_adapter, tmp_path):
        """force_scale 下无接触帧的力应为零向量（合法 L3，可过校验）"""
        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(press_frames=(10,)))

        data = xela_adapter.load(str(path), force_scale=0.001)

        sv2 = data.frames[0].schema_v2
        assert sv2.contact is False
        assert sv2.force_vector == [0.0, 0.0, 0.0]
        assert sv2.force_magnitude == 0.0
        assert sv2.compliance_level == "L3"

    def test_every_frame_schema_validate(self, xela_adapter, tmp_path):
        """所有帧的 schema_v2 应通过 TLabelSchemaV2.validate()"""
        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(press_frames=(10, 11)))

        for kwargs in ({}, {"force_scale": 0.001}):
            data = xela_adapter.load(str(path), **kwargs)
            for frame in data.frames:
                is_valid, errors = frame.schema_v2.validate()
                assert is_valid, (
                    f"kwargs={kwargs}, frame={frame.frame_idx}: {errors}"
                )

    def test_exported_json_passes_json_schema(self, xela_adapter, tmp_path):
        """导出的 to_dict() 应通过 schema/tlabel-schema.json 校验"""
        jsonschema = pytest.importorskip("jsonschema")

        path = tmp_path / "xela.csv"
        _write_recording(str(path), _recording(press_frames=(10,)))

        with open(JSON_SCHEMA_PATH, encoding="utf-8") as f:
            schema = json.load(f)

        for kwargs in ({}, {"force_scale": 0.001}):
            data = xela_adapter.load(str(path), **kwargs)
            exported = data.to_dict()
            # 抛出异常即失败；无异常说明合规
            jsonschema.validate(exported, schema)

    def test_capabilities_match_json_schema_keys(self, xela_adapter):
        """capabilities 键集应与 JSON Schema 的 capabilities 定义完全一致"""
        caps = xela_adapter.get_capabilities()

        with open(JSON_SCHEMA_PATH, encoding="utf-8") as f:
            schema = json.load(f)
        schema_caps = set(schema["properties"]["capabilities"]["properties"])

        assert set(caps) == schema_caps
        assert len(caps) == 13  # 不含 compliance_level 元字段


# =============================================================================
# 4. 异常输入处理
# =============================================================================

class TestMalformedInput:
    """错误形状 / malformed CSV / 缺失字段 / 空值处理"""

    def test_empty_file_raises_value_error(self, xela_adapter, tmp_path):
        """空 CSV 应抛出有意义的 ValueError"""
        path = tmp_path / "empty.csv"
        path.write_text("")

        with pytest.raises(ValueError, match="(空|empty|无法解析|0 帧|no data)"):
            xela_adapter.load(str(path))

    def test_header_only_raises_value_error(self, xela_adapter, tmp_path):
        """只有表头无数据行的 CSV 应抛出 ValueError"""
        path = tmp_path / "header_only.csv"
        _write_rows(str(path), [], header=XELA_HEADER)

        with pytest.raises(ValueError, match="(表头|无数据|0 帧|empty)"):
            xela_adapter.load(str(path))

    def test_wrong_shape_force_cell(self, xela_adapter, tmp_path):
        """力矩阵单元格形状错误（最后一维不为 3）应抛 ValueError"""
        path = tmp_path / "bad_shape.csv"
        rows = [["2026-01-01 12:00:00.000000", "0", "[[1, 2], [3, 4]]"]]
        _write_rows(str(path), rows, header=XELA_HEADER)

        with pytest.raises(ValueError, match="(形状|shape)"):
            xela_adapter.load(str(path))

    def test_wrong_last_dim_cell(self, xela_adapter, tmp_path):
        """taxel 轴数不为 3（如 (1,3,2)）应抛 ValueError"""
        path = tmp_path / "bad_dim.csv"
        rows = [["2026-01-01 12:00:00.000000", "0",
                 "[[[1, 2], [3, 4], [5, 6]]]"]]
        _write_rows(str(path), rows, header=XELA_HEADER)

        with pytest.raises(ValueError, match="(形状|shape)"):
            xela_adapter.load(str(path))

    def test_malformed_literal_cell(self, xela_adapter, tmp_path):
        """非字面量列表的力矩阵单元格应抛 ValueError"""
        path = tmp_path / "bad_literal.csv"
        rows = [["2026-01-01 12:00:00.000000", "0", "{{{not a list"]]
        _write_rows(str(path), rows, header=XELA_HEADER)

        with pytest.raises(ValueError):
            xela_adapter.load(str(path))

    def test_non_numeric_matrix_cell(self, xela_adapter, tmp_path):
        """包含非数值元素的力矩阵应抛 ValueError"""
        path = tmp_path / "bad_values.csv"
        rows = [["2026-01-01 12:00:00.000000", "0",
                 "[[['a', 'b', 'c']]]"]]
        _write_rows(str(path), rows, header=XELA_HEADER)

        with pytest.raises(ValueError, match="(非数值|无法解析|形状|shape)"):
            xela_adapter.load(str(path))

    def test_inconsistent_layout_between_frames(self, xela_adapter, tmp_path):
        """帧间 taxel 布局不一致应抛 ValueError"""
        path = tmp_path / "inconsistent.csv"
        rows = [
            ["2026-01-01 12:00:00.000000", "0",
             _matrix_cell(np.zeros((4, 6, 3)))],
            ["2026-01-01 12:00:00.010000", "1",
             _matrix_cell(np.zeros((3, 5, 3)))],
        ]
        _write_rows(str(path), rows, header=XELA_HEADER)

        with pytest.raises(ValueError, match="(不一致|形状|shape)"):
            xela_adapter.load(str(path))

    def test_displacement_shape_mismatch(self, xela_adapter, tmp_path):
        """位移矩阵与力矩阵形状不一致应抛 ValueError"""
        path = tmp_path / "disp_mismatch.csv"
        rows = [[
            "2026-01-01 12:00:00.000000", "0",
            _matrix_cell(np.zeros((4, 6, 3))),
            _matrix_cell(np.zeros((3, 4, 3))),
        ]]
        _write_rows(str(path), rows, header=XELA_HEADER)

        with pytest.raises(ValueError, match="(不一致|形状|shape)"):
            xela_adapter.load(str(path))

    def test_header_without_force_column(self, xela_adapter, tmp_path):
        """表头缺少力矩阵列应抛出明确的 ValueError"""
        path = tmp_path / "no_force.csv"
        rows = [["2026-01-01 12:00:00.000000", "0", "[1.0, 2.0, 3.0]"]]
        _write_rows(str(path), rows, header=["time", "seq", "ft"])

        with pytest.raises(ValueError, match="(力矩阵|force)"):
            xela_adapter.load(str(path))

    def test_empty_force_cell_raises(self, xela_adapter, tmp_path):
        """力矩阵单元格为空应抛 ValueError"""
        path = tmp_path / "empty_force.csv"
        rows = [["2026-01-01 12:00:00.000000", "0", ""]]
        _write_rows(str(path), rows, header=XELA_HEADER)

        with pytest.raises(ValueError, match="(力矩阵|为空|empty)"):
            xela_adapter.load(str(path))

    def test_blank_optional_cells_treated_as_missing(self, xela_adapter, tmp_path):
        """辅助列空单元格应按缺失处理（不中断加载）"""
        path = tmp_path / "blank_optional.csv"
        _write_recording(str(path), _recording(press_frames=(10,)))
        # _write_recording 默认将 displacement/ft/pose 写为空单元格

        data = xela_adapter.load(str(path))

        assert data.num_frames == 15
        ss = data.frames[10].sensor_specific
        assert "taxel_force" in ss
        assert "taxel_displacement" not in ss
        assert "ft_ground_truth" not in ss
        assert "end_effector_pose" not in ss

    def test_fewer_columns_ok(self, xela_adapter, tmp_path):
        """仅含 time/seq/force 三列的 CSV 应可正常加载"""
        path = tmp_path / "three_cols.csv"
        base = datetime(2026, 1, 1, 12, 0, 0)
        rows = []
        for i in range(5):
            t = (base + timedelta(seconds=0.01 * i)).strftime(
                "%Y-%m-%d %H:%M:%S.%f")
            rows.append([t, str(i), _matrix_cell(np.zeros((4, 6, 3)))])
        _write_rows(str(path), rows,
                    header=["time", "seq", "sensor_matrices_force"])

        data = xela_adapter.load(str(path))

        assert data.num_frames == 5
        assert "taxel_force" in data.frames[0].sensor_specific

    def test_nonexistent_file_raises_file_not_found(self, xela_adapter):
        """不存在的文件应抛 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            xela_adapter.load("/nonexistent/path/to/xela_data.csv")

    def test_unsupported_extension_raises_value_error(self, xela_adapter, tmp_path):
        """不支持的扩展名应抛 ValueError"""
        path = tmp_path / "xela.h5"
        path.write_text("not hdf5")

        with pytest.raises(ValueError, match="(不支持|unsupported|格式|format)"):
            xela_adapter.load(str(path))


# =============================================================================
# 5. 注册表集成
# =============================================================================

class TestRegistryIntegration:
    """适配器注册与发现（内部 registry，非 external entry-point）"""

    def test_xela_in_builtin_registry(self):
        """xela 适配器应注册在内置注册表中"""
        adapters = list_builtin_adapters()
        assert "xela" in adapters, (
            "xela 适配器未在注册表中发现。"
            "请确认 tlabel/core/registry.py 中已添加。"
        )

    def test_xela_not_in_external_registry(self):
        """xela 应为内置适配器，不在 external/community 注册表中"""
        external = list_external_adapters()
        assert "xela" not in external

    def test_adapter_class_interface(self):
        """适配器类应符合 DataAdapterBase 接口约定"""
        list_builtin_adapters()
        adapter_cls = get_adapter("xela")

        assert adapter_cls is not None
        assert issubclass(adapter_cls, DataAdapterBase)

        instance = adapter_cls()
        assert instance.name == "xela"
        assert instance.supported_extensions == [".csv"]
        assert instance.default_compliance_level == "L1"

    def test_instantiable_without_args(self):
        """适配器应可无参实例化（CI 遍历所有注册适配器时依赖此性质）"""
        list_builtin_adapters()
        adapter_cls = get_adapter("xela")
        assert adapter_cls() is not None


# =============================================================================
# 6. 能力声明与传感器信息
# =============================================================================

class TestCapabilitiesAndSensorInfo:
    """get_capabilities() / get_sensor_info() 的完整性"""

    def test_capabilities_all_bool(self, xela_adapter):
        """capabilities 的每个值都应是 bool 类型"""
        caps = xela_adapter.get_capabilities()
        assert isinstance(caps, dict) and len(caps) > 0
        for field, value in caps.items():
            assert isinstance(value, bool), (
                f"capabilities['{field}'] 类型错误: 期望 bool, 实际 {type(value)}"
            )

    def test_capabilities_expected_values(self, xela_adapter):
        """能力声明应符合 uSkin 数据源的实际情况"""
        caps = xela_adapter.get_capabilities()

        # 可靠提供的
        assert caps["contact"] is True
        assert caps["contact_centroid"] is True
        assert caps["confidence"] is True
        # 需标定系数才能输出 SI 单位的（默认输出 None）
        assert caps["force_magnitude"] is True
        assert caps["force_vector"] is True
        # 数据源无法提供的
        assert caps["temperature"] is False      # uSkin 无温度测量
        assert caps["torque_vector"] is False    # 无 taxel SI 力臂坐标
        assert caps["texture_class"] is False    # 非视觉传感器
        assert caps["slip_event"] is False       # v1 未实现
        assert caps["object_deformation"] is False
        assert caps["manipulation_phase"] is False

    def test_sensor_info_json_schema_fields(self, xela_adapter):
        """sensor_info 应包含 JSON Schema sensor 块的必需字段"""
        info = xela_adapter.get_sensor_info()

        assert info["sensor_name"] == "XELA uSkin"
        assert info["sensor_type"] == "distributed_array"  # JSON Schema 枚举值
        assert isinstance(info["adapter_name"], str) and info["adapter_name"]
        assert isinstance(info["adapter_version"], str) and info["adapter_version"]

    def test_sensor_info_basic_fields(self, xela_adapter):
        """sensor_info 应包含制造商/型号等基本字段"""
        info = xela_adapter.get_sensor_info()

        assert info["manufacturer"] == "XELA Robotics"
        assert info["type"] == "distributed_taxel_array"
        assert info["axes_per_taxel"] == 3
        assert info["taxel_layout_reference"] == [4, 6]
        assert info["channels"]["taxel_force"]["count"] == 24


# =============================================================================
# 8. 随仓库提交的合成样例（tests/data/xela/）
# =============================================================================

class TestCommittedSample:
    """随仓库提交的合成样例数据应可加载且合规（PR 模板要求的 tests/data/ 样例）"""

    SAMPLE_CSV = REPO_ROOT / "tests" / "data" / "xela" / "sample_univtac_nv_style.csv"
    SAMPLE_JSON = REPO_ROOT / "tests" / "data" / "xela" / "sample_export.json"

    def test_committed_sample_loads(self, xela_adapter):
        """提交的样例 CSV 应可正常加载并检出接触段"""
        if not self.SAMPLE_CSV.exists():
            pytest.skip("committed sample not found")
        data = xela_adapter.load(str(self.SAMPLE_CSV))

        assert data.num_frames == 30
        assert data.episode_info["taxel_layout"] == [4, 6]
        assert data.episode_info["contact_frames"] == 11
        # 位移通道恒为 0（该数据集的实测特征），应原样保留
        assert data.frames[0].sensor_specific["taxel_displacement"][0] == [0.0, 0.0, 0.0]

    def test_committed_sample_export_validates(self, xela_adapter):
        """提交的导出 JSON 应通过 JSON Schema 校验"""
        jsonschema = pytest.importorskip("jsonschema")
        if not self.SAMPLE_JSON.exists():
            pytest.skip("committed sample export not found")

        with open(self.SAMPLE_JSON, encoding="utf-8") as f:
            exported = json.load(f)
        with open(JSON_SCHEMA_PATH, encoding="utf-8") as f:
            schema = json.load(f)

        jsonschema.validate(exported, schema)


# =============================================================================
# 7. extract_schema 直接调用（无文件）
# =============================================================================

class TestExtractSchemaDirect:
    """extract_schema() 的直接单元测试（内存数据，无文件）"""

    def test_extract_schema_no_contact(self, xela_adapter):
        """无接触时除基础字段外其余应为 None"""
        raw_frame = {
            "force": np.zeros((4, 6, 3)),
            "baseline_force": np.zeros((4, 6, 3)),
            "contact_threshold": 5.0,
        }
        schema = xela_adapter.extract_schema(raw_frame)

        assert isinstance(schema, TLabelSchemaV2)
        assert schema.contact is False
        assert schema.contact_centroid is None
        assert schema.force_magnitude is None
        assert schema.force_vector is None
        assert schema.compliance_level == "L1"

    def test_extract_schema_with_force_scale(self, xela_adapter):
        """提供 force_scale 时应填充力字段并升级为 L3"""
        force = np.zeros((4, 6, 3))
        force[2, 3, 2] = 2000.0
        raw_frame = {
            "force": force,
            "baseline_force": np.zeros((4, 6, 3)),
            "contact_threshold": 10.0,
            "force_scale": 0.01,
        }
        schema = xela_adapter.extract_schema(raw_frame)

        assert schema.contact is True
        assert schema.force_vector == [0.0, 0.0, 20.0]
        assert schema.force_magnitude == pytest.approx(20.0)
        assert schema.compliance_level == "L3"
        assert schema.contact_centroid is not None

    def test_extract_schema_invalid_shape(self, xela_adapter):
        """力矩阵最后一维不为 3 时应抛 ValueError"""
        with pytest.raises(ValueError, match="(形状|shape)"):
            xela_adapter.extract_schema({"force": np.zeros((4, 6, 2))})

    def test_extract_schema_baseline_shape_mismatch(self, xela_adapter):
        """baseline 形状与 force 不一致时应抛 ValueError"""
        with pytest.raises(ValueError, match="(不一致|shape)"):
            xela_adapter.extract_schema({
                "force": np.zeros((4, 6, 3)),
                "baseline_force": np.zeros((2, 3, 3)),
            })
