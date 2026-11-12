"""SharpaDex 指尖六轴力/力矩适配器测试

覆盖场景：
  - 60 维触觉维度名解析（2 手 × 5 指 × 6 轴）
  - LeRobot v2.1 路径模板替换（chunk / episode 取整）
  - LeRobot v3.0 的 file_index 模板：拿不到真实映射必须报错，不能取模猜
  - force_magnitude = 各指力矢量模长之和（N，可跨指相加的标量）
  - force_vector / torque_vector = 受力最大那指的局部系矢量
  - contact_centroid 落在手指序位空间（不是物理坐标）
  - contact_region 恒为 "digital"（指尖，不是手掌）
  - 无接触帧降 L1（L2+ 要求 force_magnitude 非 None）
  - 分段标注 → manipulation_phase，段外按接触状态补
  - 视频特征 → object_deformation（无量纲像素变化比例）
  - texture_class / temperature 恒为 None，等级不因此上抬到 L4
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
from tlabel.core.schema import VALID_CONTACT_REGIONS, TLabelSchemaV2

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from tlabel.adapters.sharpadex import (  # noqa: E402
    SharpaDexAdapter,
    _FINGER_X,
    _HAND_Y,
    _format_template,
    _parse_tactile_names,
    _phase_for,
    _resolve_dataset_root,
)

HANDS = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "little")
AXES = ("fx", "fy", "fz", "tx", "ty", "tz")


# =============================================================================
# synthetic 数据集构造
# =============================================================================

def _tactile_names():
    return [f"{h}_{f}_{a}" for h in HANDS for f in FINGERS for a in AXES]


def _zero_frame():
    return np.zeros(60, dtype=np.float32)


def _frame(**kw):
    """构造一帧 60 维读数；kw 形如 right_thumb_fx=10.0"""
    v = np.zeros(60, dtype=np.float32)
    names = _tactile_names()
    for key, val in kw.items():
        v[names.index(key)] = val
    return v


def _make_dataset(root: Path, episodes, fps=30, segments=None,
                  codebase="v2.1", with_video=False):
    """构造一个最小 LeRobot 数据集

    参数:
        episodes: list[list[np.ndarray(60,)]]
        segments: {episode_index: [(start, end, skill), ...]}
        codebase: "v2.1" 走取整模板；"v3.0" 走 file_index 模板
    """
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    if codebase == "v2.1":
        data_path = ("data/chunk-{episode_chunk:03d}/"
                     "episode_{episode_index:06d}.parquet")
    else:
        data_path = ("data/chunk-{chunk_index:03d}/"
                     "file-{file_index:03d}.parquet")

    info = {
        "codebase_version": codebase,
        "robot_type": "sharpa_wave",
        "total_episodes": len(episodes),
        "total_frames": sum(len(e) for e in episodes),
        "fps": fps,
        "chunks_size": 1000,
        "data_path": data_path,
        "video_path": ("videos/chunk-{episode_chunk:03d}/{video_key}/"
                       "episode_{episode_index:06d}.mp4"),
        "features": {
            "observation.tactile": {
                "dtype": "float32",
                "shape": [60],
                "names": _tactile_names(),
            },
        },
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    # tasks
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Pick the ball"},
                   ensure_ascii=False) + "\n", encoding="utf-8")

    # annotations
    ann = []
    for ep_i, frames in enumerate(episodes):
        segs = []
        for (s, e, skill) in (segments or {}).get(ep_i, []):
            segs.append({
                "start_step": s, "end_step": e,
                "task_skill": skill, "text": f"{skill} | Skill: {skill}",
                "id": f"subtask-{len(segs)+1:03d}", "label": "subtasks",
            })
        ann.append({
            "episode_index": ep_i,
            "num_frames": len(frames),
            "global_task": "Pick the ball",
            "segments": segs,
            "language_segments": segs,
        })
    (root / "meta" / "annotations.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ann) + "\n",
        encoding="utf-8")

    # parquet
    for ep_i, frames in enumerate(episodes):
        n = len(frames)
        tbl = pa.table({
            "observation.tactile": pa.array(
                [list(map(float, f)) for f in frames],
                type=pa.list_(pa.float32())),
            "timestamp": pa.array([i / fps for i in range(n)],
                                  type=pa.float32()),
            "frame_index": pa.array(list(range(n)), type=pa.int64()),
            "episode_index": pa.array([ep_i] * n, type=pa.int64()),
            "task_index": pa.array([0] * n, type=pa.int64()),
            "subtask_index": pa.array([0] * n, type=pa.int64()),
        })
        if codebase == "v2.1":
            out = (root / "data" / "chunk-000" /
                   f"episode_{ep_i:06d}.parquet")
        else:
            out = (root / "data" / "chunk-000" / "file-000.parquet")
        pq.write_table(tbl, str(out))

    if with_video:
        (root / "videos" / "chunk-000" /
         "observation.images.tactile_raw").mkdir(parents=True, exist_ok=True)

    return root


@pytest.fixture
def adapter():
    return SharpaDexAdapter()


@pytest.fixture
def dataset(tmp_path):
    """3 帧：无接触 / 拇指单指接触 / 拇指+食指接触"""
    eps = [[
        _zero_frame(),
        _frame(right_thumb_fx=10.0, right_thumb_fz=10.0),
        _frame(right_thumb_fx=10.0, right_thumb_fz=10.0,
               right_index_fx=3.0),
    ]]
    segs = {0: [(0, 0, "pick"), (1, 1, "move"), (2, 2, "place")]}
    return _make_dataset(tmp_path, eps, segments=segs)


@pytest.fixture
def dataset_multi(tmp_path):
    """2 个 episode，各 4 帧"""
    eps = [
        [_zero_frame(), _zero_frame(),
         _frame(right_thumb_fz=5.0), _zero_frame()],
        [_zero_frame(),
         _frame(right_index_fx=8.0), _frame(right_index_fx=8.0),
         _zero_frame()],
    ]
    segs = {
        0: [(0, 1, "pick"), (2, 2, "move")],
        1: [(0, 0, "pick"), (1, 2, "move"), (3, 3, "place")],
    }
    return _make_dataset(tmp_path, eps, segments=segs)


# =============================================================================
# 单元：_parse_tactile_names
# =============================================================================

class TestParseTactileNames:
    def test_groups_60_dims(self):
        g = _parse_tactile_names(_tactile_names())
        assert len(g) == 10
        assert all(len(v) == 6 for v in g.values())
        assert ("right", "thumb") in g
        assert ("left", "little") in g
        # 名字按 hand → finger → axis 排列，所以 left_thumb_fx 是 0
        assert g[("left", "thumb")]["fx"] == 0
        assert g[("right", "thumb")]["fx"] == 30
        assert g[("left", "little")]["tz"] == 29
        assert g[("right", "little")]["tz"] == 59

    def test_index_maps_to_correct_position(self):
        names = _tactile_names()
        g = _parse_tactile_names(names)
        for key, ax in g.items():
            for axis, idx in ax.items():
                assert names[idx] == f"{key[0]}_{key[1]}_{axis}"

    def test_rejects_malformed_name(self):
        names = _tactile_names()
        names[5] = "garbage"
        with pytest.raises(ValueError, match="无法解析"):
            _parse_tactile_names(names)

    def test_rejects_unknown_axis(self):
        names = _tactile_names()
        names[0] = "left_thumb_fw"
        with pytest.raises(ValueError, match="无法解析"):
            _parse_tactile_names(names)

    def test_rejects_missing_axis(self):
        names = _tactile_names()[:-1]      # 少一个 tz
        with pytest.raises(ValueError, match="缺少部分轴"):
            _parse_tactile_names(names)


# =============================================================================
# 单元：_format_template
# =============================================================================

class TestFormatTemplate:
    def test_v21_episode_chunk(self):
        tpl = ("data/chunk-{episode_chunk:03d}/"
               "episode_{episode_index:06d}.parquet")
        assert _format_template(tpl, 0, 1000) == \
            "data/chunk-000/episode_000000.parquet"
        assert _format_template(tpl, 1234, 1000) == \
            "data/chunk-001/episode_001234.parquet"

    def test_v30_file_index_requires_extra(self):
        tpl = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        with pytest.raises(ValueError, match="file_index"):
            _format_template(tpl, 5, 1000)

    def test_v30_with_extra(self):
        tpl = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        got = _format_template(tpl, 5, 1000,
                               {"chunk_index": 0, "file_index": 3})
        assert got == "data/chunk-000/file-003.parquet"

    def test_no_placeholder(self):
        assert _format_template("meta/info.json", 7, 1000) == "meta/info.json"


# =============================================================================
# 单元：_phase_for
# =============================================================================

class TestPhaseFor:
    SEGS = [
        {"start": 0, "end": 9, "skill": "pick", "text": None},
        {"start": 10, "end": 19, "skill": "move", "text": None},
        {"start": 20, "end": 29, "skill": "place", "text": None},
    ]

    def test_pick_without_force_is_pre_contact(self):
        assert _phase_for(3, False, self.SEGS) == ("pre_contact",
                                                   "annotation_segment")

    def test_pick_with_force_is_grasp(self):
        assert _phase_for(3, True, self.SEGS) == ("grasp",
                                                  "annotation_segment")

    def test_move_is_lift(self):
        assert _phase_for(15, True, self.SEGS) == ("lift",
                                                   "annotation_segment")

    def test_place_is_place(self):
        assert _phase_for(25, True, self.SEGS) == ("place",
                                                   "annotation_segment")

    def test_outside_segment_uses_heuristic(self):
        assert _phase_for(35, True, self.SEGS) == ("hold",
                                                   "contact_heuristic")
        assert _phase_for(35, False, self.SEGS) == ("pre_contact",
                                                    "contact_heuristic")

    def test_no_segments_uses_heuristic(self):
        assert _phase_for(0, True, []) == ("hold", "contact_heuristic")

    def test_unknown_skill_falls_back_to_hold(self):
        segs = [{"start": 0, "end": 5, "skill": "wipe", "text": None}]
        assert _phase_for(2, True, segs) == ("hold", "annotation_segment")

    def test_only_returns_schema_enum_values(self):
        from tlabel.core.schema import VALID_MANIPULATION_PHASES
        for i in range(0, 40):
            for c in (True, False):
                ph, _ = _phase_for(i, c, self.SEGS)
                assert ph in VALID_MANIPULATION_PHASES


# =============================================================================
# 单元：_resolve_dataset_root
# =============================================================================

class TestResolveDatasetRoot:
    def test_direct_lerobot_dir(self, tmp_path):
        _make_dataset(tmp_path, [[_zero_frame()]])
        assert _resolve_dataset_root(tmp_path) == tmp_path

    def test_season_dir_prefers_v21(self, tmp_path):
        season = tmp_path / "season_x"
        v21 = season / "lerobot_v2.1"
        v21.mkdir(parents=True)
        _make_dataset(v21, [[_zero_frame()]])
        assert _resolve_dataset_root(season) == v21

    def test_missing_meta_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="info.json"):
            _resolve_dataset_root(tmp_path)


# =============================================================================
# extract_schema
# =============================================================================

class TestExtractSchema:
    @pytest.fixture
    def groups(self):
        return _parse_tactile_names(_tactile_names())

    def test_no_contact_returns_l1(self, adapter, groups):
        s = adapter.extract_schema({"tactile": _zero_frame(),
                                    "groups": groups})
        assert s.contact is False
        assert s.compliance_level == "L1"
        assert s.force_magnitude is None
        assert s.force_vector is None
        assert s.torque_vector is None
        assert s.contact_centroid is None
        assert s.contact_region is None

    def test_contact_is_l3(self, adapter, groups):
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fz=10.0), "groups": groups})
        assert s.contact is True
        assert s.compliance_level == "L3"

    def test_force_magnitude_is_sum_of_norms(self, adapter, groups):
        # thumb |F| = sqrt(3^2+4^2) = 5；index |F| = sqrt(12^2+5^2) = 13
        # 和 = 18，而不是合矢量的模（那需要坐标变换）
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fx=3.0, right_thumb_fy=4.0,
                              right_index_fx=12.0, right_index_fy=5.0),
            "groups": groups,
        })
        assert s.force_magnitude == pytest.approx(18.0, abs=1e-3)

    def test_force_vector_is_strongest_finger(self, adapter, groups):
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fx=3.0, right_thumb_fy=4.0,
                              right_index_fx=12.0, right_index_fy=5.0,
                              right_index_tz=0.7),
            "groups": groups,
        })
        assert s.force_vector == pytest.approx([12.0, 5.0, 0.0], abs=1e-4)
        assert s.torque_vector == pytest.approx([0.0, 0.0, 0.7], abs=1e-4)

    def test_force_vector_reports_which_finger(self, adapter, groups, dataset):
        data = adapter.load(str(dataset))
        f = data.frames[2]
        assert f.sensor_specific["force_vector_finger"] == "right_thumb"
        assert f.sensor_specific["force_vector_frame"] == \
            "fingertip_elastomer_local"

    def test_contact_region_is_digital(self, adapter, groups):
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fz=10.0), "groups": groups})
        assert s.contact_region == "digital"
        assert s.contact_region in VALID_CONTACT_REGIONS

    def test_centroid_in_finger_ordinal_space(self, adapter, groups):
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fz=10.0), "groups": groups})
        # 只有右拇指受力 → 落在拇指列、右手行
        assert s.contact_centroid[0] == pytest.approx(_FINGER_X["thumb"],
                                                      abs=1e-4)
        assert s.contact_centroid[1] == pytest.approx(_HAND_Y["right"],
                                                      abs=1e-4)

    def test_centroid_weights_by_force(self, adapter, groups):
        # thumb |F|=5，index |F|=13 → 加权偏向 index
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fx=3.0, right_thumb_fy=4.0,
                              right_index_fx=12.0, right_index_fy=5.0),
            "groups": groups,
        })
        expect_x = (5 * _FINGER_X["thumb"] + 13 * _FINGER_X["index"]) / 18
        assert s.contact_centroid[0] == pytest.approx(expect_x, abs=1e-4)

    def test_two_hands_weighted(self, adapter, groups):
        s = adapter.extract_schema({
            "tactile": _frame(left_thumb_fz=10.0, right_thumb_fz=10.0),
            "groups": groups,
        })
        assert s.contact_centroid[1] == pytest.approx(0.5, abs=1e-4)

    def test_validate_passes(self, adapter, groups):
        for tac in (_zero_frame(),
                    _frame(right_thumb_fz=10.0),
                    _frame(right_thumb_fz=10.0, right_index_fz=3.0)):
            s = adapter.extract_schema({"tactile": tac, "groups": groups})
            ok, errors = s.validate()
            assert ok, errors

    def test_threshold_respected(self, adapter, groups):
        # 0.3 N 低于默认 0.5 N 阈值
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fz=0.3), "groups": groups})
        assert s.contact is False
        # 显式把阈值降到 0.2 就应该判接触
        s2 = adapter.extract_schema({
            "tactile": _frame(right_thumb_fz=0.3), "groups": groups,
            "contact_threshold": 0.2})
        assert s2.contact is True

    def test_wrong_vector_length_raises(self, adapter, groups):
        with pytest.raises(ValueError, match="长度"):
            adapter.extract_schema({"tactile": np.zeros(59),
                                    "groups": groups})

    def test_missing_groups_raises(self, adapter):
        with pytest.raises(ValueError, match="分组"):
            adapter.extract_schema({"tactile": _zero_frame()})


# =============================================================================
# slip / deformation
# =============================================================================

class TestSlipAndDeformation:
    @pytest.fixture
    def groups(self):
        return _parse_tactile_names(_tactile_names())

    def test_slip_on_centroid_shift(self, adapter, groups):
        s = adapter.extract_schema({
            "tactile": _frame(right_index_fz=10.0),
            "groups": groups,
            "prev_centroid": [_FINGER_X["thumb"], _HAND_Y["right"]],
            "prev_total": 10.0,
            "dt": 1 / 30,
        })
        assert s.slip_event is True
        assert s.slip_velocity is not None
        assert s.slip_velocity[0] > 0

    def test_no_slip_when_stable(self, adapter, groups):
        c = [_FINGER_X["thumb"], _HAND_Y["right"]]
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fz=10.0),
            "groups": groups,
            "prev_centroid": c,
            "prev_total": 10.0,
            "dt": 1 / 30,
        })
        assert s.slip_event is False
        assert s.slip_velocity is None

    def test_deformation_needs_video(self, adapter, groups):
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fz=10.0), "groups": groups})
        assert s.object_deformation is None

    def test_deformation_from_video_area(self, adapter, groups):
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fz=10.0),
            "groups": groups,
            "video_features": {
                ("right", "thumb"): {"area": 0.42, "rms": 30.0},
                ("right", "index"): {"area": 0.10, "rms": 5.0},
            },
        })
        assert s.object_deformation == pytest.approx(0.42, abs=1e-6)

    def test_video_does_not_lift_level_to_l4(self, adapter, groups):
        """有视频也只到 L3：texture_class 判不出来、temperature 没有传感器"""
        s = adapter.extract_schema({
            "tactile": _frame(right_thumb_fz=10.0),
            "groups": groups,
            "video_features": {
                ("right", "thumb"): {"area": 0.42, "rms": 30.0},
            },
        })
        assert s.compliance_level == "L3"
        assert s.texture_class is None
        assert s.temperature is None


# =============================================================================
# load / iter_frames
# =============================================================================

class TestLoad:
    def test_load_frames(self, adapter, dataset):
        data = adapter.load(str(dataset))
        assert len(data.frames) == 3
        assert data.sensor_id == "sharpadex"
        assert data.streamed is False

    def test_frame_metadata(self, adapter, dataset):
        data = adapter.load(str(dataset))
        f = data.frames[1]
        assert f.frame_idx == 1
        assert f.timestamp_s == pytest.approx(1 / 30, abs=1e-4)
        assert f.sensor_specific["task"] == "Pick the ball"
        assert f.sensor_specific["contact_centroid_space"] == "finger_ordinal"
        assert f.sensor_specific["force_unit"] == "N"
        assert f.sensor_specific["torque_unit"] == "N·m"

    def test_phase_follows_segments(self, adapter, dataset):
        data = adapter.load(str(dataset))
        assert [f.manipulation_phase for f in data.frames] == \
            ["pre_contact", "lift", "place"]
        assert data.frames[0].sensor_specific["phase_source"] == \
            "annotation_segment"

    def test_schema_manipulation_phase_filled(self, adapter, dataset):
        data = adapter.load(str(dataset))
        for f in data.frames:
            assert f.schema_v2.manipulation_phase == f.manipulation_phase

    def test_all_frames_validate(self, adapter, dataset_multi):
        data = adapter.load(str(dataset_multi), episode_index=0)
        for f in data.frames:
            ok, errors = f.schema_v2.validate()
            assert ok, errors

    def test_episode_index_selects(self, adapter, dataset_multi):
        d0 = adapter.load(str(dataset_multi), episode_index=0)
        d1 = adapter.load(str(dataset_multi), episode_index=1)
        assert d0.episode_info["episode_index"] == 0
        assert d1.episode_info["episode_index"] == 1
        assert d0.frames[2].schema_v2.contact is True      # right_thumb
        assert d1.frames[1].schema_v2.contact is True      # right_index

    def test_max_frames(self, adapter, dataset_multi):
        data = adapter.load(str(dataset_multi), episode_index=0, max_frames=2)
        assert len(data.frames) == 2

    def test_stream_mode_drops_frames(self, adapter, dataset_multi):
        data = adapter.load(str(dataset_multi), episode_index=0, stream=True)
        assert data.frames == []
        assert data.streamed is True
        assert data.aggregate_stats["num_frames"] == 4

    def test_sensor_info(self, adapter, dataset):
        data = adapter.load(str(dataset))
        si = data.sensor_info
        assert si["type"] == "fingertip_six_axis"
        assert si["layout"]["dims"] == 60
        assert si["layout"]["fps"] == 30

    def test_single_parquet_path(self, adapter, dataset):
        p = dataset / "data" / "chunk-000" / "episode_000000.parquet"
        data = adapter.load(str(p))
        assert len(data.frames) == 3

    def test_v30_template_without_episodes_dir_raises(self, adapter, tmp_path):
        root = _make_dataset(tmp_path, [[_zero_frame()]], codebase="v3.0")
        with pytest.raises(ValueError, match="file_index|v3.0"):
            adapter.load(str(root))


# =============================================================================
# 能力声明与注册
# =============================================================================

class TestCapabilities:
    def test_thirteen_keys(self, adapter):
        caps = adapter.get_capabilities()
        assert len(caps) == 13

    def test_unsupported_declared(self, adapter):
        caps = adapter.get_capabilities()
        assert caps["temperature"] is False
        assert caps["texture_class"] is False

    def test_supported_declared(self, adapter):
        caps = adapter.get_capabilities()
        for k in ("contact", "contact_centroid", "contact_region",
                  "force_magnitude", "force_vector", "torque_vector",
                  "slip_event", "slip_velocity", "manipulation_phase",
                  "object_deformation", "confidence"):
            assert caps[k] is True

    def test_default_level_is_l3(self, adapter):
        assert adapter.default_compliance_level == "L3"

    def test_is_data_adapter(self, adapter):
        assert isinstance(adapter, DataAdapterBase)

    def test_registered(self):
        assert "sharpadex" in list_builtin_adapters()

    def test_adapter_metadata(self, adapter):
        assert adapter.name == "sharpadex"
        assert adapter.supported_extensions == [".parquet"]


# =============================================================================
# 端到端：13 维覆盖
# =============================================================================

class TestCoverage:
    def test_dimension_fill(self, adapter, dataset_multi):
        data = adapter.load(str(dataset_multi), episode_index=0)
        filled = {k: 0 for k in (
            "contact", "contact_centroid", "contact_region",
            "force_magnitude", "force_vector", "torque_vector",
            "slip_event", "slip_velocity", "manipulation_phase",
            "texture_class", "object_deformation", "temperature",
            "confidence")}
        for f in data.frames:
            for k in filled:
                if getattr(f.schema_v2, k) is not None:
                    filled[k] += 1
        n = len(data.frames)
        # 无条件字段
        for k in ("contact", "slip_event", "manipulation_phase",
                  "confidence"):
            assert filled[k] == n, k
        # 有接触的帧才有
        for k in ("contact_centroid", "contact_region", "force_magnitude",
                  "force_vector", "torque_vector"):
            assert filled[k] == 1, k
        # 数据集没有的，必须老实空着
        assert filled["texture_class"] == 0
        assert filled["temperature"] == 0
        assert filled["object_deformation"] == 0   # 没有视频
