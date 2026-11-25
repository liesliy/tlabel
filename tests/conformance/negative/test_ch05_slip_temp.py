"""第 5.6–5.8 章：`slip_event`、`slip_velocity`、`temperature` 的负向测试。"""

from __future__ import annotations

import copy

from conftest import (
    CORE_CAPS,
    SURFACE_2D,
    assert_accepted,
    assert_rejected,
    base_dataset,
    base_frame,
    run,
)


def _caps(**overrides):
    c = dict(CORE_CAPS)
    c.update(overrides)
    return c


#: 一个「轴数对、但含出平面轴」的二维坐标系。
#: 轴的数量能通过 TL-8.2-1，出平面这件事只有 TL-5.7-2 能抓。
OUT_OF_PLANE_2D = {
    **copy.deepcopy(SURFACE_2D),
    "frame_id": "bad2d",
    "axes": {"x": "nominal_surface_normal", "y": "tangential_1"},
}


def _dataset_with_bad_frame(**overrides):
    ds = base_dataset(**overrides)
    ds["frames_of_reference"] = list(ds["frames_of_reference"]) + [copy.deepcopy(OUT_OF_PLANE_2D)]
    return ds


def _slip_velocity_frame(**overrides):
    fr = base_frame(slip_velocity=[1.5, -0.5], slip_velocity_frame="surface_0")
    fr["field_provenance"]["slip_velocity"] = {
        "type": "derived_measurement",
        "input_refs": ["raw_sensor_values"],
        "method_id": "conf:slip_velocity_v1",
    }
    fr.update(overrides)
    return fr


SLIP_VEL_DS = base_dataset(capabilities=_caps(slip_velocity=True))


def _temperature_frame(value, **overrides):
    fr = base_frame(temperature=value)
    fr["field_provenance"]["temperature"] = {
        "type": "direct_measurement",
        "sensor_or_channel_id": "sensor_0.ch0",
        "measurement_target": "contact_interface",
    }
    fr.update(overrides)
    return fr


TEMP_DS = base_dataset(capabilities=_caps(temperature=True))


# ── 5.6 slip_event ─────────────────────────────────────────

def test_slip_event_as_direct_measurement_is_rejected():
    """TL-5.6-1：滑移不是可直接观测的物理量。

    任何滑移判定都基于某种信号（振动、位移、力突变）的推断。记成直接测量
    会隐藏判定方法，使结果不可复现——这正是「代理量冒充物理量」的入口。
    """
    fr = base_frame()
    fr["field_provenance"]["slip_event"] = {"type": "direct_measurement"}
    assert_rejected(fr, "TL-5.6-1")


def test_slip_event_numeric_value_is_rejected():
    """TL-5.6-2：slip_event 只接受 true / false / null。"""
    assert_rejected(base_frame(slip_event=1), "TL-5.6-2")


def test_slip_event_value_under_false_capability_is_rejected():
    """TL-5.6-3：没有滑移判定能力时不得给出 `false`。

    与 TL-5.1-3 同构：`false` 是断言，不是「没数据」的默认值。
    实际判定由 TL-7.3-2 完成。
    """
    assert_rejected(
        base_frame(slip_event=False),
        "TL-7.3-1",
        base_dataset(capabilities=_caps(slip_event=False)),
    )


# ── 5.7 slip_velocity ──────────────────────────────────────

def test_slip_velocity_with_three_components_is_rejected():
    """TL-5.7-1：滑移速度是切平面内的二维量，恰好 2 个分量。"""
    assert_rejected(
        _slip_velocity_frame(slip_velocity=[1.5, -0.5, 0.2]), "TL-5.7-1", SLIP_VEL_DS
    )


def test_slip_velocity_with_out_of_plane_axis_is_rejected():
    """TL-5.7-2：坐标轴必须都在接触切平面内。

    轴的数量对（x、y 两个）不代表语义对——把法向轴命名为 x 就能骗过维度检查。
    """
    assert_rejected(
        _slip_velocity_frame(slip_velocity_frame="bad2d"),
        "TL-5.7-2",
        _dataset_with_bad_frame(capabilities=_caps(slip_velocity=True)),
    )


def test_slip_velocity_referencing_3d_frame_is_rejected():
    """TL-5.7-2：三维坐标系不是切平面。"""
    assert_rejected(
        _slip_velocity_frame(slip_velocity_frame="sensor_0"), "TL-5.7-2", SLIP_VEL_DS
    )


def test_slip_velocity_with_tangent_plane_frame_is_accepted():
    """TL-5.7-2 的对照：两个切向轴张成的二维坐标系是合法形态。"""
    assert_accepted(_slip_velocity_frame(), SLIP_VEL_DS)


# ── 5.8 temperature ────────────────────────────────────────

def test_temperature_nan_is_rejected():
    """TL-5.8-1：温度必须是有限数。

    NaN 走的是全局的 TL-4.3-1（所有数值必须有限），因为「有限」这条约束
    不专属于温度——报在这里是因为它是温度字段的入口。
    """
    assert_rejected(_temperature_frame(float("nan")), "TL-4.3-1", TEMP_DS)


def test_temperature_infinity_is_rejected():
    """TL-5.8-1：Inf 同样不是有限数。"""
    assert_rejected(_temperature_frame(float("inf")), "TL-4.3-1", TEMP_DS)


def test_temperature_without_measurement_target_is_rejected():
    """TL-5.8-2：温度非 null 时必须声明测量对象。

    传感器本体温度、接触界面温度、物体温度是三个不同的量，同一个数字
    不加标注无法解释。
    """
    fr = base_frame(temperature=31.5)
    fr["field_provenance"]["temperature"] = {
        "type": "direct_measurement",
        "sensor_or_channel_id": "sensor_0.ch0",
    }
    assert_rejected(fr, "TL-5.8-2", TEMP_DS)


def test_temperature_measurement_target_outside_closed_set_is_rejected():
    """TL-5.8-2：measurement_target 只能是 sensor_body / contact_interface / object。"""
    fr = _temperature_frame(31.5)
    fr["field_provenance"]["temperature"]["measurement_target"] = "skin"
    assert_rejected(fr, "TL-5.8-2", TEMP_DS)


def test_temperature_with_valid_target_is_accepted():
    """TL-5.8-2 的对照：声明了合法测量对象后通过。"""
    assert_accepted(_temperature_frame(31.5), TEMP_DS)


# ── TL-5.7-4 slip_velocity 与 slip_event 的一致性（SHOULD 级）─────


def test_slip_velocity_without_slip_is_warning_not_error():
    """TL-5.7-4：`slip_event=false` 却有 `slip_velocity` 只报 warning。

    这条是 SHOULD 而不是 MUST，规范也给了理由：传感器可能报出低于滑移阈值
    的残余运动——真的动了，但没到判定为滑移的程度。把它判成 error，会逼
    生产者要么丢掉真实的运动信息，要么把阈值调低到失去意义。

    但也不能不管：warning 是唯一能提示「这两个字段的说法不一致」的通道。
    """
    ds = base_dataset()
    ds["capabilities"]["slip_velocity"] = True

    fr = base_frame()
    fr["slip_event"] = False
    fr["slip_velocity"] = [1.0, 0.0]
    fr["slip_velocity_frame"] = "surface_0"
    fr["field_provenance"]["slip_velocity"] = {
        "type": "model_inference",
        "method_id": "conf:slip_velocity_v1",
    }

    rep = run(fr, ds)
    hits = [f for f in rep.findings if f.clause == "TL-5.7-4"]
    assert hits, f"期望 TL-5.7-4 的提示，实际：{[(f.code, f.clause) for f in rep.findings]}"
    assert all(f.outcome == "warning" for f in hits), (
        f"TL-5.7-4 是 SHOULD 级，不得报成 error：{[(f.code, f.outcome) for f in hits]}"
    )
    # warning 不得让整份数据变成不通过——否则 SHOULD 就成了事实上的 MUST
    assert rep.ok, f"warning 不应导致拒绝，实际错误：{[(f.code, f.clause) for f in rep.errors]}"


def test_slip_velocity_with_slip_is_accepted():
    """TL-5.7-4 的对照：`slip_event=true` 时 `slip_velocity` 有值完全正常。

    没有这条，一个「永远报 warning」的实现也能让上面那条变绿。
    """
    ds = base_dataset()
    ds["capabilities"]["slip_velocity"] = True

    fr = base_frame()
    fr["slip_event"] = True
    fr["slip_velocity"] = [1.0, 0.0]
    fr["slip_velocity_frame"] = "surface_0"
    fr["field_provenance"]["slip_velocity"] = {
        "type": "model_inference",
        "method_id": "conf:slip_velocity_v1",
    }

    rep = run(fr, ds)
    assert not [f for f in rep.findings if f.clause == "TL-5.7-4"], (
        f"有滑移时不该提示：{[(f.code, f.clause) for f in rep.findings]}"
    )
