"""Grabette 双阵列触觉适配器测试

覆盖场景：
  - LeRobot v3.0 目录结构的加载（meta/info.json + meta/tasks.parquet + data parquet）
  - meta/tasks.parquet 的 pandas index 陷阱（任务文本在 index 而非列）
  - 双阵列不对称尺寸（6×6 + 4×8）的独立统计与加权合并
  - 全零帧（未接触）不产生虚假质心
  - 九宫格接触区域判定
  - compliance_level 恒为 L2，force_vector / torque_vector 恒为 None
  - force_magnitude 为无量纲原始读数（不填牛顿值）
  - slip 检测（质心位移 / 压力相对变化）
  - iter_frames 流式路径：episode 过滤、max_frames、与 load 一致
  - 注册表发现

所有数据为测试内构造的 synthetic fixture，不依赖真实数据文件。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tlabel.adapters.base import DataAdapterBase
from tlabel.core.registry import get_adapter, list_builtin_adapters
from tlabel.core.schema import TLabelSchemaV2

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from tlabel.adapters.grabette import (  # noqa: E402
    GrabetteAdapter,
    _array_stats,
    _region_of,
)

S1_KEY = "observation.tactile.right_sensor_1"
S2_KEY = "observation.tactile.right_sensor_2"


# =============================================================================
# synthetic 数据集构造
# =============================================================================

def _grid(shape, fill=0, peak_at=None, peak_val=0):
    """构造一个压力网格；peak_at=(r,c) 时在该 taxel 打一个峰值"""
    g = np.full(shape, fill, dtype=np.int16)
    if peak_at is not None:
        g[peak_at] = peak_val
    return g


def _make_dataset(root: Path, episodes, fps=50):
    """构造一个最小 LeRobot v3.0 数据集

    参数:
        episodes: list[list[(g1, g2, is_lost)]]，每个子列表是一个 episode 的帧
    """
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "grabette",
        "total_episodes": len(episodes),
        "total_frames": sum(len(e) for e in episodes),
        "fps": fps,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(info), encoding="utf-8")

    # tasks.parquet：故意把任务文本放在 pandas index（复现真实数据集的坑）
    import pandas as pd
    tasks_df = pd.DataFrame(
        {"task_index": np.array([0], dtype=np.int64)},
        index=pd.Index(["Pick up the white cube"], name="task"),
    )
    tasks_df.to_parquet(root / "meta" / "tasks.parquet")

    g1_list, g2_list, ts, fi, eps, tix, lost = [], [], [], [], [], [], []
    idx = 0
    for ep_i, frames in enumerate(episodes):
        for local_i, (g1, g2, is_lost) in enumerate(frames):
            g1_list.append(np.asarray(g1, dtype=np.int16).tolist())
            g2_list.append(np.asarray(g2, dtype=np.int16).tolist())
            ts.append(local_i / fps)
            fi.append(idx)
            eps.append(ep_i)
            tix.append(0)
            lost.append(float(is_lost))
            idx += 1

    table = pa.table({
        S1_KEY: pa.array(g1_list),
        S2_KEY: pa.array(g2_list),
        "timestamp": pa.array(ts, type=pa.float32()),
        "frame_index": pa.array(fi, type=pa.int64()),
        "episode_index": pa.array(eps, type=pa.int64()),
        "task_index": pa.array(tix, type=pa.int64()),
        "is_lost": pa.array(lost, type=pa.float32()),
    })
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return root


@pytest.fixture
def grabette_adapter():
    """返回 GrabetteAdapter 实例（先触发注册表懒加载）"""
    list_builtin_adapters()
    cls = get_adapter("grabette")
    if cls is None:
        pytest.skip("grabette adapter not registered")
    return cls()


@pytest.fixture
def dataset(tmp_path):
    """两个 episode：第一个含接触序列，第二个全零（纯未接触）"""
    ep0 = [
        (_grid((6, 6)), _grid((4, 8)), 0),                                  # 全零
        (_grid((6, 6), peak_at=(2, 1), peak_val=500),
         _grid((4, 8), peak_at=(1, 2), peak_val=800), 0),                    # 接触
        (_grid((6, 6), peak_at=(2, 1), peak_val=520),
         _grid((4, 8), peak_at=(1, 2), peak_val=810), 0),                    # 轻微变化
        (_grid((6, 6), peak_at=(2, 5), peak_val=900),
         _grid((4, 8), peak_at=(1, 7), peak_val=1500), 0),                   # 大位移 → slip
    ]
    ep1 = [(_grid((6, 6)), _grid((4, 8)), 0) for _ in range(3)]
    return _make_dataset(tmp_path / "ds", [ep0, ep1])


# =============================================================================
# 单元：_array_stats
# =============================================================================

class TestArrayStats:
    def test_all_zero_has_no_centroid(self):
        """全零帧不能把"没有接触"算成"接触在原点" """
        s = _array_stats(_grid((6, 6)))
        assert s["total"] == 0.0
        assert s["centroid"] is None
        assert s["active"] == 0

    def test_single_taxel_centroid(self):
        """单 taxel 受力时质心就是该 taxel 的归一化坐标"""
        g = _grid((6, 6), peak_at=(0, 0), peak_val=100)
        s = _array_stats(g)
        assert s["total"] == 100.0
        assert s["centroid"] == pytest.approx((0.0, 0.0))
        assert s["active"] == 1
        assert s["peak"] == 100.0

    def test_bottom_right_taxel(self):
        g = _grid((6, 6), peak_at=(5, 5), peak_val=42)
        s = _array_stats(g)
        assert s["centroid"] == pytest.approx((1.0, 1.0))

    def test_uniform_grid_centroid_is_center(self):
        g = np.full((6, 6), 10, dtype=np.int16)
        s = _array_stats(g)
        assert s["centroid"] == pytest.approx((0.5, 0.5))
        assert s["active"] == 36

    def test_two_equal_peaks_average(self):
        g = _grid((6, 6))
        g[0, 0] = 100
        g[5, 5] = 100
        s = _array_stats(g)
        assert s["centroid"] == pytest.approx((0.5, 0.5))

    def test_weighted_peak_dominates(self):
        g = _grid((6, 6))
        g[0, 0] = 900
        g[5, 5] = 100
        s = _array_stats(g)
        # 0.9*0 + 0.1*1 = 0.1
        assert s["centroid"] == pytest.approx((0.1, 0.1))

    def test_second_array_shape(self):
        """4×8 阵列同样走网格分支，不是扁平 taxel 列表"""
        g = _grid((4, 8), peak_at=(0, 7), peak_val=10)
        s = _array_stats(g)
        assert s["centroid"] == pytest.approx((1.0, 0.0))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            _array_stats(np.zeros((6, 6, 1)))


# =============================================================================
# 单元：_region_of
# =============================================================================

class TestRegionOf:
    @pytest.mark.parametrize("cx,cy,expected", [
        (0.1, 0.1, "top_left"),
        (0.5, 0.1, "top_center"),
        (0.9, 0.1, "top_right"),
        (0.1, 0.5, "middle_left"),
        (0.5, 0.5, "middle_center"),
        (0.9, 0.5, "middle_right"),
        (0.1, 0.9, "bottom_left"),
        (0.5, 0.9, "bottom_center"),
        (0.9, 0.9, "bottom_right"),
    ])
    def test_nine_cells(self, cx, cy, expected):
        assert _region_of(cx, cy) == expected

    def test_boundary_goes_to_center(self):
        """1/3 和 2/3 的分界点归 middle/center"""
        assert _region_of(1 / 3, 1 / 3) == "middle_center"
        assert _region_of(2 / 3, 2 / 3) == "middle_center"


# =============================================================================
# 单元：extract_schema
# =============================================================================

class TestExtractSchema:
    def test_no_contact_returns_early(self, grabette_adapter):
        s = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6)),
            "grid_2": _grid((4, 8)),
        })
        assert s.contact is False
        assert s.force_magnitude is None
        assert s.contact_centroid is None
        assert s.contact_region is None
        assert s.compliance_level == "L2"

    def test_contact_sets_force_magnitude(self, grabette_adapter):
        s = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6), peak_at=(2, 1), peak_val=500),
            "grid_2": _grid((4, 8), peak_at=(1, 2), peak_val=800),
        })
        assert s.contact is True
        assert s.force_magnitude == pytest.approx(1300.0)

    def test_force_vector_and_torque_always_none(self, grabette_adapter):
        """单轴法向压力，拿不到矢量力和力矩"""
        s = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6), peak_at=(2, 1), peak_val=500),
            "grid_2": _grid((4, 8), peak_at=(1, 2), peak_val=800),
        })
        assert s.force_vector is None
        assert s.torque_vector is None
        assert s.temperature is None
        assert s.texture_class is None

    def test_compliance_level_is_l2(self, grabette_adapter):
        s = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6), peak_at=(2, 1), peak_val=500),
            "grid_2": _grid((4, 8), peak_at=(1, 2), peak_val=800),
        })
        assert s.compliance_level == "L2"

    def test_is_lost_lowers_confidence(self, grabette_adapter):
        base = {
            "grid_1": _grid((6, 6), peak_at=(2, 1), peak_val=500),
            "grid_2": _grid((4, 8), peak_at=(1, 2), peak_val=800),
        }
        normal = grabette_adapter.extract_schema(dict(base))
        lost = grabette_adapter.extract_schema({**base, "is_lost": 1.0})
        assert lost.confidence < normal.confidence

    def test_single_array_contact_lowers_confidence(self, grabette_adapter):
        """只有一个阵列有读数时，置信度应低于双阵列一致"""
        both = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6), peak_at=(2, 1), peak_val=500),
            "grid_2": _grid((4, 8), peak_at=(1, 2), peak_val=800),
        })
        one = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6), peak_at=(2, 1), peak_val=500),
            "grid_2": _grid((4, 8)),
        })
        assert one.confidence < both.confidence

    def test_slip_detected_on_large_shift(self, grabette_adapter):
        s = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6), peak_at=(2, 5), peak_val=900),
            "grid_2": _grid((4, 8), peak_at=(1, 7), peak_val=1500),
            "prev_centroid": [0.0, 0.0],
            "prev_total": 2400.0,
            "dt": 0.02,
        })
        assert s.slip_event is True
        assert s.slip_velocity is not None
        assert len(s.slip_velocity) == 2

    def test_no_slip_on_stable_pressure(self, grabette_adapter):
        # 该帧合并质心约 (0.2528, 0.3590)：g1 峰在 (2,1)→(0.2,0.4)，
        # g2 峰在 (1,2)→(2/7, 1/3)，按 500:800 加权
        s = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6), peak_at=(2, 1), peak_val=500),
            "grid_2": _grid((4, 8), peak_at=(1, 2), peak_val=800),
            "prev_centroid": [0.2528, 0.3590],
            "prev_total": 1300.0,
            "dt": 0.02,
        })
        assert s.slip_event is False
        assert s.slip_velocity is None

    def test_object_deformation_is_dimensionless_ratio(self, grabette_adapter):
        s = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6), peak_at=(2, 1), peak_val=500),
            "grid_2": _grid((4, 8), peak_at=(1, 2), peak_val=800),
            "prev_centroid": [0.2528, 0.3590],
            "prev_total": 2600.0,
            "dt": 0.02,
        })
        # |1300 - 2600| / max(1300, 2600, 1) = 0.5
        assert s.object_deformation == pytest.approx(0.5, abs=1e-5)

    def test_custom_slip_threshold(self, grabette_adapter):
        """把阈值调到极高时不应触发 slip"""
        s = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6), peak_at=(2, 5), peak_val=900),
            "grid_2": _grid((4, 8), peak_at=(1, 7), peak_val=1500),
            "prev_centroid": [0.0, 0.0],
            "prev_total": 2400.0,
            "dt": 0.02,
            "slip_shift_threshold": 99.0,
            "slip_pressure_threshold": 99.0,
        })
        assert s.slip_event is False

    def test_schema_is_valid_v2(self, grabette_adapter):
        s = grabette_adapter.extract_schema({
            "grid_1": _grid((6, 6), peak_at=(2, 1), peak_val=500),
            "grid_2": _grid((4, 8), peak_at=(1, 2), peak_val=800),
        })
        assert isinstance(s, TLabelSchemaV2)
        assert s.to_dict()["contact"] is True


# =============================================================================
# 集成：load / iter_frames
# =============================================================================

class TestLoad:
    def test_adapter_is_data_adapter(self, grabette_adapter):
        assert isinstance(grabette_adapter, DataAdapterBase)
        assert grabette_adapter.name == "grabette"
        assert ".parquet" in grabette_adapter.supported_extensions

    def test_load_first_episode(self, grabette_adapter, dataset):
        data = grabette_adapter.load(str(dataset))
        assert len(data.frames) == 4
        assert data.frames[0].schema_v2.contact is False
        assert data.frames[1].schema_v2.contact is True

    def test_load_specific_episode(self, grabette_adapter, dataset):
        data = grabette_adapter.load(str(dataset), episode_index=1)
        assert len(data.frames) == 3
        assert all(not f.schema_v2.contact for f in data.frames)

    def test_load_accepts_parquet_file(self, grabette_adapter, dataset):
        pf = dataset / "data" / "chunk-000" / "file-000.parquet"
        data = grabette_adapter.load(str(pf))
        assert len(data.frames) == 4

    def test_task_from_pandas_index(self, grabette_adapter, dataset):
        """tasks.parquet 的任务文本在 pandas index，不是普通列"""
        data = grabette_adapter.load(str(dataset))
        assert data.episode_info["task"] == "Pick up the white cube"

    def test_episode_info_fields(self, grabette_adapter, dataset):
        data = grabette_adapter.load(str(dataset))
        ei = data.episode_info
        assert ei["source"] == "grabette-tactile-full-synced"
        assert ei["robot_type"] == "grabette"
        assert ei["fps"] == 50
        assert ei["num_frames"] == 4
        assert ei["total_episodes"] == 2

    def test_all_frames_are_l2(self, grabette_adapter, dataset):
        data = grabette_adapter.load(str(dataset))
        assert all(f.schema_v2.compliance_level == "L2" for f in data.frames)

    def test_raw_grids_preserved(self, grabette_adapter, dataset):
        """原始读数必须保留，便于下游自行换标定系数"""
        data = grabette_adapter.load(str(dataset))
        ss = data.frames[1].sensor_specific
        assert np.asarray(ss["raw_grid_1"]).shape == (6, 6)
        assert np.asarray(ss["raw_grid_2"]).shape == (4, 8)
        assert ss["raw_grid_1"][2][1] == 500

    def test_sensor_specific_has_both_totals(self, grabette_adapter, dataset):
        data = grabette_adapter.load(str(dataset))
        ss = data.frames[1].sensor_specific
        assert ss["sensor_1_total"] == pytest.approx(500.0)
        assert ss["sensor_2_total"] == pytest.approx(800.0)
        assert ss["sensor_1_active_taxels"] == 1
        assert ss["sensor_2_active_taxels"] == 1

    def test_manipulation_phase_progression(self, grabette_adapter, dataset):
        data = grabette_adapter.load(str(dataset))
        phases = [f.manipulation_phase for f in data.frames]
        assert phases[0] == "idle"
        assert phases[1] == "initial_contact"
        assert phases[2] == "stable_contact"

    def test_stream_true_returns_no_frames(self, grabette_adapter, dataset):
        data = grabette_adapter.load(str(dataset), stream=True)
        assert data.frames == []
        assert getattr(data, "streamed", False) is True

    def test_missing_directory_raises(self, grabette_adapter, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError)):
            grabette_adapter.load(str(tmp_path / "nope"))


class TestIterFrames:
    def test_yields_frames(self, grabette_adapter, dataset):
        frames = list(grabette_adapter.iter_frames(str(dataset)))
        assert len(frames) == 4

    def test_matches_load(self, grabette_adapter, dataset):
        loaded = grabette_adapter.load(str(dataset))
        streamed = list(grabette_adapter.iter_frames(str(dataset)))
        assert len(streamed) == len(loaded.frames)
        for a, b in zip(loaded.frames, streamed):
            assert a.frame_idx == b.frame_idx
            assert a.schema_v2.contact == b.schema_v2.contact
            assert a.schema_v2.force_magnitude == b.schema_v2.force_magnitude
            assert a.manipulation_phase == b.manipulation_phase

    def test_max_frames(self, grabette_adapter, dataset):
        frames = list(grabette_adapter.iter_frames(str(dataset), max_frames=2))
        assert len(frames) == 2

    def test_episode_filter(self, grabette_adapter, dataset):
        frames = list(grabette_adapter.iter_frames(str(dataset), episode_index=1))
        assert len(frames) == 3
        assert all(f.sensor_specific["episode_index"] == 1 for f in frames)

    def test_meta_out_populated(self, grabette_adapter, dataset):
        meta = {}
        list(grabette_adapter.iter_frames(str(dataset), meta_out=meta))
        assert "sensor_info" in meta
        assert "episode_info" in meta
        assert meta["sensor_info"]["layout"]["sensor_1_shape"] == [6, 6]
        assert meta["sensor_info"]["layout"]["sensor_2_shape"] == [4, 8]


# =============================================================================
# 能力声明与注册表
# =============================================================================

class TestCapabilities:
    def test_true_capabilities(self, grabette_adapter):
        caps = grabette_adapter.get_capabilities()
        for k in ("contact", "contact_centroid", "contact_region",
                  "force_magnitude", "slip_event", "slip_velocity",
                  "manipulation_phase", "object_deformation", "confidence"):
            assert caps[k] is True, k

    def test_false_capabilities(self, grabette_adapter):
        caps = grabette_adapter.get_capabilities()
        for k in ("force_vector", "torque_vector", "texture_class",
                  "temperature"):
            assert caps[k] is False, k

    def test_thirteen_keys(self, grabette_adapter):
        assert len(grabette_adapter.get_capabilities()) == 13

    def test_sensor_info_has_both_channels(self, grabette_adapter):
        """channels 是按名字索引的 dict，两个阵列形状各自声明"""
        si = grabette_adapter.get_sensor_info()
        ch = si["channels"]
        assert set(ch) == {"tactile_sensor_1", "tactile_sensor_2"}
        assert ch["tactile_sensor_1"]["shape"] == [6, 6]
        assert ch["tactile_sensor_2"]["shape"] == [4, 8]
        assert ch["tactile_sensor_1"]["count"] == 36
        assert ch["tactile_sensor_2"]["count"] == 32

    def test_registered_in_registry(self):
        list_builtin_adapters()
        assert get_adapter("grabette") is not None


# =============================================================================
# 导出 JSON 合规
# =============================================================================

class TestExport:
    def test_export_json_roundtrip(self, grabette_adapter, dataset, tmp_path):
        data = grabette_adapter.load(str(dataset))
        out = tmp_path / "out.json"
        data.export(str(out))
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["format"] == "tlabel_schema_v2"
        assert len(payload["frames"]) == 4
        for f in payload["frames"]:
            assert f["schema_v2"]["compliance_level"] == "L2"
