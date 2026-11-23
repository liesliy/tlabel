"""P9 回归测试：缺失值（None）不得被伪装成零值（0.0）。

背景
----
v0.25 及之前，``TLabelFrame.force_magnitude`` 把 ``schema_v2.force_magnitude``
的 ``None`` 转成 ``0.0``，``_sv2_scalar`` 也有隐式 ``default=0.0``。结果是：

- 一个不支持力测量的传感器，在统计摘要里表现为"测得恒定零力"；
- 事件检测把 0.0 当作有效测量，凭空生成 ``stable_grip`` / ``force_spike``；
- 质量评分把"有值→无值"的边界读成一次巨幅突变。

修复要点（阶段 0 / T0-1~T0-3）：

1. ``frame.force_magnitude`` 直接透传，返回 ``Optional[float]``；
2. ``_sv2_scalar(frame, field, missing)`` 的 ``missing`` 改为必填参数，
   调用方必须显式声明缺失策略；
3. 统计 / 差分 / 事件判定传 ``None`` 并过滤；只有模型特征提取才允许
   传 ``0.0``（并在源码中标注为"特征填充"）。

本文件锁定这些语义，防止回归。
"""

import pytest

from tlabel.core.annotation import annotate_events_from_data
from tlabel.core.capabilities import CAPABILITY_KEYS
from tlabel.core.schema import TLabelSchemaV2
from tlabel.core.types import TLabelData, TLabelFrame, _sv2_scalar
from tlabel.predict.engine import PredictEngine
from tlabel.quality.scorer import QualityScorer

SENSOR_INFO = {"type": "test_sensor", "model": "test_model"}
EPISODE_INFO = {"task": "test_task"}


def _caps(contact=True, **overrides):
    caps = {k: False for k in CAPABILITY_KEYS}
    caps["contact"] = contact
    caps.update(overrides)
    return caps


def _make_data(n=10, force_fn=None, deform_fn=None, contact=True, caps=None):
    """构造 n 帧数据；force_fn(i) / deform_fn(i) 返回 None 表示该帧缺失。"""
    frames = []
    for i in range(n):
        sv2 = TLabelSchemaV2(
            contact=contact,
            force_magnitude=force_fn(i) if force_fn else None,
            object_deformation=deform_fn(i) if deform_fn else None,
        )
        frames.append(TLabelFrame(
            frame_idx=i, timestamp_s=float(i) * 0.1, schema_v2=sv2,
        ))
    return TLabelData(
        frames=frames,
        sensor_info=SENSOR_INFO,
        episode_info=EPISODE_INFO,
        capabilities=caps if caps is not None else _caps(),
    )


# ============================================================
# 1. 便利属性透传 None
# ============================================================

class TestForceMagnitudeProperty:
    def test_returns_none_when_missing(self):
        data = _make_data(3, force_fn=lambda i: None)
        for f in data.frames:
            assert f.force_magnitude is None

    def test_returns_value_when_present(self):
        data = _make_data(3, force_fn=lambda i: 1.5 + i)
        assert [f.force_magnitude for f in data.frames] == [1.5, 2.5, 3.5]

    def test_zero_is_not_confused_with_missing(self):
        """0.0 是有效测量，None 是缺失 —— 两者必须可区分。"""
        data = _make_data(2, force_fn=lambda i: 0.0 if i == 0 else None)
        assert data.frames[0].force_magnitude == 0.0
        assert data.frames[1].force_magnitude is None


# ============================================================
# 2. _sv2_scalar 的缺失策略必须显式声明
# ============================================================

class TestSv2ScalarMissingPolicy:
    def test_missing_argument_is_required(self):
        data = _make_data(1)
        with pytest.raises(TypeError):
            _sv2_scalar(data.frames[0], "force_magnitude")  # noqa: 故意缺参

    def test_none_policy_passes_through(self):
        data = _make_data(1, force_fn=lambda i: None)
        assert _sv2_scalar(data.frames[0], "force_magnitude", missing=None) is None

    def test_zero_policy_fills(self):
        data = _make_data(1, force_fn=lambda i: None)
        assert _sv2_scalar(data.frames[0], "force_magnitude", missing=0.0) == 0.0

    def test_bool_converted_to_float(self):
        data = _make_data(1, contact=True)
        assert _sv2_scalar(data.frames[0], "contact", missing=None) == 1.0

    def test_vector_field_uses_missing_policy(self):
        """向量字段不可标量化，走 missing 策略而不是报错。"""
        data = _make_data(1)
        data.frames[0].schema_v2.force_vector = [1.0, 2.0, 3.0]
        assert _sv2_scalar(data.frames[0], "force_vector", missing=None) is None


# ============================================================
# 3. 统计摘要不得被缺失值污染
# ============================================================

class TestDescribeMissing:
    def test_fully_missing_field_absent(self):
        """整段无该字段数据 → 不产出统计条目。"""
        data = _make_data(10, force_fn=lambda i: None)
        assert "force_magnitude" not in data.describe()

    def test_partial_missing_reports_count_and_missing(self):
        data = _make_data(10, force_fn=lambda i: 1.0 + i * 0.01 if i < 4 else None)
        s = data.describe()["force_magnitude"]
        assert s["count"] == 4
        assert s["missing"] == 6
        assert s["min"] == pytest.approx(1.0)
        assert s["max"] == pytest.approx(1.03)
        assert s["mean"] == pytest.approx(1.015)

    def test_missing_does_not_drag_mean_down(self):
        """关键回归：6 帧缺失不得把均值从 1.015 拉到 0.406。"""
        data = _make_data(10, force_fn=lambda i: 1.0 + i * 0.01 if i < 4 else None)
        s = data.describe()["force_magnitude"]
        assert s["mean"] > 1.0

    def test_complete_field_has_zero_missing(self):
        data = _make_data(5, force_fn=lambda i: float(i))
        s = data.describe()["force_magnitude"]
        assert s["count"] == 5
        assert s["missing"] == 0


# ============================================================
# 4. 事件生成不得凭空造标签
# ============================================================

class TestEventsWithoutForceData:
    def test_annotation_no_stable_grip_without_force(self):
        """持续接触但无力测量 → 不得判定为"抓握稳定"。"""
        data = _make_data(20, force_fn=lambda i: None)
        annotate_events_from_data(data)
        types = [e.event_type for e in data.tactile_events]
        assert "stable_grip" not in types

    def test_annotation_no_force_spike_without_force(self):
        data = _make_data(20, force_fn=lambda i: None)
        annotate_events_from_data(data)
        types = [e.event_type for e in data.tactile_events]
        assert "force_spike" not in types

    def test_prediction_no_force_events_without_force(self):
        data = _make_data(20, force_fn=lambda i: None)
        events = PredictEngine().predict_events(data)
        types = [e["event_type"] for e in events]
        assert "stable_grip" not in types
        assert "force_spike" not in types

    def test_prediction_no_deformation_anomaly_without_deform(self):
        data = _make_data(20, deform_fn=lambda i: None)
        events = PredictEngine().predict_events(data)
        types = [e["event_type"] for e in events]
        assert "deformation_anomaly" not in types

    def test_stable_grip_still_detected_with_real_force(self):
        """能力保留：真有稳定力数据时仍应检测到 stable_grip。"""
        data = _make_data(20, force_fn=lambda i: 1.0)
        annotate_events_from_data(data)
        types = [e.event_type for e in data.tactile_events]
        assert "stable_grip" in types


# ============================================================
# 5. 质量评分不得把缺失边界读成突变
# ============================================================

class TestScorerMissing:
    def test_partial_missing_does_not_create_fake_jumps(self):
        """前 5 帧有力、后 5 帧缺失：不应产生虚假跳变。"""
        data = _make_data(10, force_fn=lambda i: 1.0 if i < 5 else None)
        score = QualityScorer().score(data)
        # 缺失边界被跳过 → 平滑度保持满分
        assert score["temporal_smoothness"] == pytest.approx(100.0, abs=1e-6)

    def test_all_missing_does_not_crash(self):
        data = _make_data(10, force_fn=lambda i: None, deform_fn=lambda i: None)
        score = QualityScorer().score(data)
        assert score["temporal_smoothness"] >= 0.0
        assert score["physical_consistency"] >= 0.0
