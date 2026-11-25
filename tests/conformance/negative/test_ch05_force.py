"""第 5.3–5.5 章：`normal_force`、`force_vector`、`torque_vector` 的负向测试。

第 5.3 节是整份规范最容易出错的地方：法向力、合力、合力模长是三个不同的量，
而它们在数值上经常接近。基底的 F = [0.42, -0.18, 3.10] 刻意让
`|F·n| = 3.10` 与 `‖F‖ = 3.1309…` 不相等，否则这些测试全部失效。
"""

from __future__ import annotations

import math

from conftest import (
    CORE_CAPS,
    assert_accepted,
    assert_rejected,
    base_dataset,
    base_frame,
    run,
)

#: 基底的合力模长。用它冒充 normal_force 是 RFC-004 §7 T1 的核心反例。
NORM_F = math.sqrt(0.42 ** 2 + 0.18 ** 2 + 3.10 ** 2)


def _caps(**overrides):
    c = dict(CORE_CAPS)
    c.update(overrides)
    return c


def _local_normal(k: float):
    """把基底的法向参考改成局部法向，并给出模长为 k 的 n。"""
    fr = base_frame()
    fr["field_provenance"]["normal_force"]["normal_reference"] = "contact_interface_local"
    fr["field_provenance"]["normal_force"]["normal_vector"] = [0.0, 0.0, k]
    return fr


def _torque_frame(**overrides):
    """带完整力矩来源的帧（能力声明与 provenance 都补齐）。"""
    fr = base_frame(torque_vector=[0.1, 0.2, 0.3], torque_vector_frame="sensor_0")
    fr["field_provenance"]["torque_vector"] = {
        "type": "direct_measurement",
        "sensor_or_channel_id": "sensor_0.ch0",
        "reference_point": [0.0, 0.0, 0.0],
    }
    fr.update(overrides)
    return fr


TORQUE_DS = base_dataset(capabilities=_caps(torque_vector=True))


# ── 5.3 normal_force ────────────────────────────────────────

def test_normal_force_negative_is_rejected():
    """TL-5.3-1：法向力是大小，不得为负。"""
    assert_rejected(base_frame(normal_force=-1.0), "TL-5.3-1")


def test_normal_force_equal_to_norm_is_rejected():
    """TL-5.3-2：把合力模长当法向力，即使结果看起来合理也必须拒。

    这是 RFC-004 §7 的 T1 反例。用 ‖F‖ 算出来的数永远「像那么回事」，
    只有和切向分量对照才能识别：只要 F 有切向分量，|F·n| 就严格小于 ‖F‖。
    """
    assert_rejected(base_frame(normal_force=NORM_F), "TL-5.3-2")


def test_normal_force_equal_to_norm_with_local_normal_is_rejected():
    """TL-5.3-2：局部法向路径同样不得用模长。

    法向参考换成 `contact_interface_local` 后，判据不变——这条约束针对的是
    「用了什么公式」，不是「声明了哪种法向」。
    """
    assert_rejected(_local_normal(1.0) | {"normal_force": NORM_F}, "TL-5.3-2")


def test_normal_force_equal_to_norm_along_normal_is_accepted():
    """TL-5.3-2 的边界：F 完全沿法向时 ‖F‖ = |F·n|，这是合法的。

    没有这条对照，一个「只要 normal_force 等于 ‖F‖ 就报错」的实现也能全绿，
    而它会误杀所有纯法向受力的数据。
    """
    assert_accepted(base_frame(normal_force=3.10, force_vector=[0.0, 0.0, 3.10]))


def test_derived_normal_force_without_normal_reference_is_rejected():
    """TL-5.3-3：由 force_vector 派生却无法确定法向方向。

    此时按 TL-5.3-6 应把值置 null 并给出可用性状态，而不是猜一个方向。
    """
    fr = base_frame()
    del fr["field_provenance"]["normal_force"]["normal_reference"]
    assert_rejected(fr, "TL-5.3-3")


def test_normal_force_not_matching_projection_is_rejected():
    """TL-5.3-3：值不等于 |F·n| 时报错。

    与 T1 不同：这里值不等于模长，也不等于投影，纯粹是算错了或填错了。
    """
    assert_rejected(base_frame(normal_force=2.0), "TL-5.3-3")


def test_direct_normal_force_without_measurement_axis_is_rejected():
    """TL-5.3-5：直接测量的法向力必须声明测量轴。

    单轴传感器的敏感轴与接触界面法向有夹角时，测到的是轴向分量而不是法向力；
    不声明轴，这个值无法归入任何一类。
    """
    fr = base_frame()
    fr["field_provenance"]["normal_force"] = {
        "type": "direct_measurement",
        "sensor_or_channel_id": "sensor_0.ch0",
    }
    assert_rejected(fr, "TL-5.3-5")


def test_measurement_axis_without_angular_tolerance_is_rejected():
    """TL-5.3-10：拿测量轴当法向参考，必须声明角度容差。

    单轴传感器测的是它自己敏感轴上的分量。轴向和接触法向差 2° 时，读数
    还是法向力；差 40° 时，那是轴向力，不是法向力。容差就是这个分界。

    校验器验不了容差**值**对不对——那是生产者的物理判断。但它能验有没有
    声明。不声明时，两类值在数据里长得完全一样，下游无从区分。
    """
    fr = base_frame()
    fr["field_provenance"]["normal_force"] = {
        "type": "derived_measurement",
        "normal_reference": "measurement_axis",
        "measurement_axis": "sensor_0.z",
        "method_id": "conf:axial_projection_v1",
    }
    assert_rejected(fr, "TL-5.3-10")


def test_measurement_axis_with_angular_tolerance_is_accepted():
    """TL-5.3-10 的对照：声明了容差就通过。

    没有这条，一个「所有 measurement_axis 都拒掉」的实现也能让上面那条全绿，
    而它会误杀单轴力传感器——那类传感器只有一根敏感轴，本来就没法用别的
    方式声明法向参考。
    """
    fr = base_frame()
    fr["field_provenance"]["normal_force"] = {
        "type": "derived_measurement",
        "normal_reference": "measurement_axis",
        "measurement_axis": "sensor_0.z",
        "angular_tolerance_deg": 2.0,
        "input_refs": ["force_vector"],
        "method_id": "conf:axial_projection_v1",
    }
    assert_accepted(fr)


def test_direct_normal_force_with_measurement_axis_is_accepted():
    """TL-5.3-5 的对照：声明了测量轴就通过。

    直接测量不需要 normal_reference，也不需要 force_vector 投影——
    这正是一个「所有 normal_force 都必须有 normal_reference」的实现会误杀的。
    """
    fr = base_frame()
    fr["field_provenance"]["normal_force"] = {
        "type": "direct_measurement",
        "sensor_or_channel_id": "sensor_0.ch0",
        "measurement_axis": "z",
    }
    assert_accepted(fr)


def test_normal_reference_outside_closed_set_is_rejected():
    """TL-5.3-8：normal_reference 若声明，必须是三值之一。"""
    fr = base_frame()
    fr["field_provenance"]["normal_force"]["normal_reference"] = "sensor_z"
    assert_rejected(fr, "TL-5.3-8")


def test_local_normal_without_vector_is_rejected():
    """TL-5.3-9：声明 contact_interface_local 却不给 n，是自相矛盾的。

    曲面指尖在不同位置接触时局部法向不同；声明局部法向却不给向量，
    下游无法判断这个值沿哪个方向投影。
    """
    fr = base_frame()
    fr["field_provenance"]["normal_force"]["normal_reference"] = "contact_interface_local"
    assert_rejected(fr, "TL-5.3-9")


def test_normal_vector_not_unit_is_rejected():
    """TL-5.3-4：n 必须是单位向量。

    未归一化的 n 会把投影结果按 ‖n‖ 放大，而且数值看起来仍然合理。
    """
    assert_rejected(_local_normal(1.1), "TL-5.3-4")


def test_normal_vector_within_tolerance_is_accepted():
    """TL-5.3-4 的边界：|‖n‖ − 1| ≤ 1e-3 必须通过。

    容差存在的意义就是让浮点误差不算违规；写成 `<` 而不是 `≤` 的实现
    会把合法数据判错。
    """
    assert_accepted(_local_normal(1.0 + 9e-4))


def test_normal_vector_at_tolerance_boundary_is_rejected():
    """TL-5.3-4 的边界：超出 1e-3 必须拒。"""
    assert_rejected(_local_normal(1.0 + 1.1e-3), "TL-5.3-4")


def test_normal_force_alone_without_force_vector_is_accepted():
    """TL-5.3-7：只测法向力的传感器不需要 force_vector。

    这条是正向的：要求 force_vector 存在会把大量单轴力传感器挡在门外。
    """
    fr = base_frame()
    fr["force_vector"] = None
    fr["field_provenance"].pop("force_vector")
    assert_accepted(
        fr, base_dataset(capabilities=_caps(force_vector=False))
    )


# ── 5.4 force_vector ───────────────────────────────────────

def test_force_vector_with_two_components_is_rejected():
    """TL-5.4-1：恰好 3 个分量。"""
    assert_rejected(base_frame(force_vector=[0.42, -0.18]), "TL-5.4-1")


def test_force_vector_with_string_component_is_rejected():
    """TL-5.4-1：分量必须是数值。"""
    assert_rejected(base_frame(force_vector=["0.42", "-0.18", "3.10"]), "TL-5.4-1")


def test_force_vector_referencing_2d_frame_is_rejected():
    """TL-5.4-2：力的坐标系必须是三维。"""
    assert_rejected(base_frame(force_vector_frame="surface_0"), "TL-5.4-2")


def test_force_vector_without_frame_is_rejected():
    """TL-5.4-3 的可验证部分：底层坐标系未知时不得输出 force_vector。

    规范原文要求「不得在坐标系未知时输出」，校验器能判断的形态是
    「引用了未声明或未指定的坐标系」，实际判定由 TL-8.1-1 完成。
    """
    fr = base_frame()
    del fr["force_vector_frame"]
    assert_rejected(fr, "TL-8.1-1")


# ── 5.5 torque_vector ──────────────────────────────────────

def test_torque_vector_with_two_components_is_rejected():
    """TL-5.5-1：恰好 3 个分量。"""
    assert_rejected(_torque_frame(torque_vector=[0.1, 0.2]), "TL-5.5-1", TORQUE_DS)


def test_torque_vector_referencing_2d_frame_is_rejected():
    """TL-5.5-2：力矩的坐标系必须是三维。"""
    assert_rejected(_torque_frame(torque_vector_frame="surface_0"), "TL-5.5-2", TORQUE_DS)


def test_torque_without_reference_point_is_rejected():
    """TL-5.5-3：力矩随参考点变化，没有参考点的值无法解释。

    两个传感器都输出 [0, 0, 1.0]，一个以传感器原点为参考、一个以接触点为参考，
    物理意义完全不同却看起来一样。
    """
    fr = _torque_frame()
    del fr["field_provenance"]["torque_vector"]["reference_point"]
    assert_rejected(fr, "TL-5.5-3", TORQUE_DS)


def test_torque_with_reference_point_is_accepted():
    """TL-5.5-3 的对照：补齐参考点后通过。"""
    assert_accepted(_torque_frame(), TORQUE_DS)


def test_reference_point_with_two_components_is_rejected():
    """TL-5.5-5：参考点是三维坐标系中的位置，必须 3 个分量。"""
    fr = _torque_frame()
    fr["field_provenance"]["torque_vector"]["reference_point"] = [0.0, 0.0]
    assert_rejected(fr, "TL-5.5-5", TORQUE_DS)
