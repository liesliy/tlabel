"""无效数据与占位实现的缺失表达。

TL-5.1-3：*A producer MUST NOT emit `false` to mean "not measured".*

这里锁住两个曾经违反该条款的地方：
- `tashan_ts_f_a` 遇到 uint16 无效标记（65535）时，曾返回 `contact=False` + `force_magnitude=0.0`
- `paxini_px6d` 是占位符 adapter，曾每帧返回 `contact=False` + `slip_event=False`

两者都会让下游把「测不到」当成「确定没有」，进而错误丢弃数据或用 0.0 污染统计。
"""

from __future__ import annotations

from tlabel.adapters.paxini_px6d import PaxiniPX6DAdapter
from tlabel.adapters.tashan_ts_f_a import INVALID_MARKER, TashanTsFAAdapter


class TestTashanInvalidFrame:
    """无效标记帧必须是 None，不能是 False / 0.0。"""

    def _invalid_frame(self):
        return {
            "normal_force": INVALID_MARKER,
            "tangential_force": 1.0,
            "tangential_direction": 0.0,
            "tangential_fx": 0.0,
            "tangential_fy": 0.0,
            "contact_indicator": 1.0,
        }

    def test_contact_is_none_not_false(self):
        s = TashanTsFAAdapter().extract_schema(self._invalid_frame())
        assert s.contact is None, "无效帧不能宣称『确定无接触』"

    def test_force_is_none_not_zero(self):
        s = TashanTsFAAdapter().extract_schema(self._invalid_frame())
        assert s.force_magnitude is None, "无效数据不是零力，0.0 会污染下游均值方差"

    def test_invalid_flag_preserved(self):
        s = TashanTsFAAdapter().extract_schema(self._invalid_frame())
        assert s.raw_sensor_values["invalid"] is True
        assert s.raw_sensor_values["invalid_marker_value"] == INVALID_MARKER

    def test_raw_values_are_original(self):
        """原始读数要保真，不能被抹成 0.0。"""
        s = TashanTsFAAdapter().extract_schema(self._invalid_frame())
        assert s.raw_sensor_values["raw_normal_force"] == INVALID_MARKER

    def test_valid_frame_still_produces_values(self):
        """不能因为改无效分支把正常路径改坏。"""
        s = TashanTsFAAdapter().extract_schema({
            "normal_force": 3.5,
            "tangential_force": 1.0,
            "tangential_direction": 0.0,
            "tangential_fx": 1.0,
            "tangential_fy": 0.0,
            "contact_indicator": 1.0,
        })
        assert s.contact is True
        assert s.force_magnitude == 3.5


class TestPX6DPlaceholder:
    """占位实现不产出任何测量值。"""

    def test_contact_is_none(self):
        s = PaxiniPX6DAdapter().extract_schema({})
        assert s.contact is None, "未实现 ≠ 确定无接触"

    def test_slip_is_none(self):
        s = PaxiniPX6DAdapter().extract_schema({})
        assert s.slip_event is None

    def test_confidence_is_none(self):
        s = PaxiniPX6DAdapter().extract_schema({})
        assert s.confidence is None, "0.0 会被当成『测到零信心』"

    def test_compliance_still_l1(self):
        """基类兜底不可破坏。"""
        s = PaxiniPX6DAdapter().extract_schema({})
        assert s.compliance_level == "L1"
