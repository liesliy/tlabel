"""第 8 章 —— 坐标系的反向测试。

覆盖 TL-8.1-1 / 8.1-2 / 8.1-3 / 8.2-1 / 8.2-2 / 8.3-1 / 8.3-2 / 8.3-3 / 8.4-4 / 8.4-5。

本章的存在理由是 V3.0 早期草案的一个真实漏洞：附录示例里二维质心引用了
三维坐标系，而规范自身无法发现。维度声明、轴集合、方向约定这三件事必须
显式写下来，机器才有得查。
"""

from __future__ import annotations

import copy

import pytest

from conftest import (
    SENSOR_3D,
    SURFACE_2D,
    base_dataset,
    base_frame,
    assert_accepted,
    assert_rejected,
    run,
)


def _ds_with(frames=None, transforms=None):
    """构造一个替换了坐标系声明（和可选变换）的数据集。"""
    ds = base_dataset()
    if frames is not None:
        ds["frames_of_reference"] = frames
    if transforms is not None:
        ds["transforms"] = transforms
    return ds


def _frame_variant(base=SENSOR_3D, **overrides):
    """在某个坐标系声明上做单点修改。"""
    f = copy.deepcopy(base)
    f.update(overrides)
    return f


#: 一个六项齐全、可直接使用的变换。用于所有「缺一项」的反例做基底。
GOOD_TRANSFORM = {
    "transform_id": "sensor_to_surface",
    "source_frame": "sensor_0",
    "target_frame": "surface_0",
    "rotation_representation": "quaternion",
    "rotation_element_order": "wxyz",
    "translation": [0.0, 0.0, 12.5],
    "translation_frame": "surface_0",
    "translation_unit": "mm",
    "calibration": "calibrated",
}


# ─────────────────────────────────────────────────────────────
# 8.1 声明要求
# ─────────────────────────────────────────────────────────────


def test_vector_referencing_undeclared_frame_is_rejected():
    """TL-8.1-1：向量字段必须引用数据集中已声明的坐标系。

    引用一个不存在的 frame_id 时，那个向量就只是三个数——没有原点、没有
    朝向、没有维度。下游只能猜，而猜错的向量比缺失的向量更危险：
    它看起来是完整的。
    """
    assert_rejected(base_frame(force_vector_frame="nope"), "TL-8.1-1")


@pytest.mark.parametrize(
    "missing_key", ["dimension", "frame_type", "handedness", "axes", "reference_object"]
)
def test_frame_missing_required_key_is_rejected(missing_key):
    """TL-8.1-2：坐标系声明必须回答六项（`frame_id` 由键名承担）。

    这五项缺任何一项，下游都得猜。特别是 `reference_object`——「这个坐标系
    相对什么定义」决定了原点在哪，缺了它，坐标值的物理含义就不完整。
    """
    broken = {k: v for k, v in SENSOR_3D.items() if k != missing_key}
    assert_rejected(base_frame(), "TL-8.1-2", dataset=_ds_with(frames=[broken, SURFACE_2D]))


def test_complete_frame_declaration_is_accepted():
    """TL-8.1-2 的对照：六项齐全时必须通过。"""
    assert_accepted(base_frame(), dataset=_ds_with(frames=[SENSOR_3D, SURFACE_2D]))


def test_frame_with_invalid_dimension_is_rejected():
    """TL-8.1-3：`dimension` 只能是 2 或 3。

    这条挡住的是「dimension 写成字符串 `"3"`」和「写成 4」两种错法——
    它们都会让「二维量引用三维系」这类检查失效。
    """
    assert_rejected(
        base_frame(),
        "TL-8.1-3",
        dataset=_ds_with(frames=[_frame_variant(dimension=4), SURFACE_2D]),
    )


@pytest.mark.parametrize(
    "dim, axes",
    [
        (2, {"x": "tangential_1"}),
        (2, {"x": "tangential_1", "y": "tangential_2", "z": "nominal_surface_normal"}),
        (3, {"x": "tangential_1", "y": "tangential_2"}),
        (3, {"x": "tangential_1", "y": "tangential_2", "w": "nominal_surface_normal"}),
    ],
)
def test_frame_axes_do_not_match_dimension(dim, axes):
    """TL-8.1-3：2 维必须恰好有 x/y，3 维必须恰好有 x/y/z。

    「恰好」是关键——多一个轴或少一个轴都错。多轴的情况（2 维却声明了 z）
    最隐蔽：它看起来信息更全，实际上让「这是平面量」的断言失效了。
    """
    assert_rejected(
        base_frame(),
        "TL-8.1-3",
        dataset=_ds_with(frames=[_frame_variant(dimension=dim, axes=axes), SURFACE_2D]),
    )


# ─────────────────────────────────────────────────────────────
# 8.2 坐标系类型
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["camera", "base_link", "world_frame", "", "SENSOR_LOCAL"])
def test_frame_type_outside_closed_set_is_rejected(bad):
    """TL-8.2-1：`frame_type` 必须是四个值之一。

    `base_link` 是最容易混进来的——它是 ROS 里的常见名字，但规范用的是
    `robot_base`。这类「差一个词」的值不报错，跨工具链对接时才炸。
    """
    assert_rejected(
        base_frame(),
        "TL-8.2-1",
        dataset=_ds_with(frames=[_frame_variant(frame_type=bad), SURFACE_2D]),
    )


@pytest.mark.parametrize("good", ["sensor_local", "robot_link", "robot_base", "world"])
def test_all_four_frame_types_are_accepted(good):
    """TL-8.2-1 的对照：四个值逐个验证。"""
    assert_accepted(
        base_frame(),
        dataset=_ds_with(frames=[_frame_variant(frame_type=good), SURFACE_2D]),
    )


@pytest.mark.parametrize("bad", ["left", "Left", "either", None])
def test_non_right_handedness_is_rejected(bad):
    """TL-8.2-2：`handedness` 必须是 `right`。

    规范只允许右手系，是为了让「绕 z 轴逆时针」这句话无歧义。允许左手系
    就等于允许同一份数据有两种符号约定，而下游无法从数值看出用的是哪种。
    """
    assert_rejected(
        base_frame(),
        "TL-8.2-2",
        dataset=_ds_with(frames=[_frame_variant(handedness=bad), SURFACE_2D]),
    )


# ─────────────────────────────────────────────────────────────
# 8.3 轴语义
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "axes",
    [
        {"x": "", "y": "tangential_1", "z": "nominal_surface_normal"},
        {"x": "tangential_1", "y": None, "z": "nominal_surface_normal"},
        {"x": "tangential_1", "y": "tangential_2", "z": ""},
    ],
)
def test_axis_without_declared_meaning_is_rejected(axes):
    """TL-8.3-1：每个轴都必须说明含义。

    空字符串是「声明了但没说」——与缺键不同。缺键由 8.1-2 抓，空值由这条抓。
    两者分开报，第三方才知道是「忘了写 axes」还是「写了 axes 但留了空」。
    """
    assert_rejected(
        base_frame(),
        "TL-8.3-1",
        dataset=_ds_with(frames=[_frame_variant(axes=axes), SURFACE_2D]),
    )


def test_nominal_normal_axis_is_accepted():
    """TL-8.3-2 的对照：把某个轴声明为 `nominal_surface_normal` 是合法的。

    这条与下面的反例配对。规范允许这个声明，只是不允许把它理解成
    「每个接触点的法向」——那是语义约束，不是结构约束。
    """
    assert_accepted(base_frame(), dataset=_ds_with(frames=[SENSOR_3D, SURFACE_2D]))


def test_nominal_normal_used_as_local_contact_normal_is_rejected():
    """TL-8.3-2 / 8.3-3：把坐标系的名义法向当作接触点局部法向。

    `normal_reference = contact_interface_local` 的意思是「法向随接触位置变」，
    此时必须给出该点的法向向量。若生产者拿坐标系那个固定轴来充当它，
    就等于声称「所有接触点的法向都一样」——这对平面成立，对弧形指尖不成立。
    """
    frame = base_frame()
    frame["field_provenance"]["normal_force"] = {
        "type": "derived_measurement",
        "normal_reference": "contact_interface_local",
        "input_refs": ["force_vector"],
        "method_id": "conf:local_normal_v1",
        # 没有 normal_vector
    }
    report = run(frame)
    assert report.findings, "声称用局部法向却没有给出法向向量，必须被拒"
    assert any(f.outcome == "error" for f in report.findings)


# ─────────────────────────────────────────────────────────────
# 8.4 变换
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", list(GOOD_TRANSFORM.keys()))
def test_transform_missing_required_key_is_rejected(key):
    """TL-8.4-4：变换必须声明九项（六类信息）。

    少任何一项，接收方都得猜——而变换猜错的后果是「看起来对齐了，数值
    差一个旋转」，比没有变换更难发现。`rotation_element_order` 尤其如此：
    同一个 3×3 矩阵按行读和按列读互为转置。
    """
    t = {k: v for k, v in GOOD_TRANSFORM.items() if k != key}
    if key in ("source_frame", "target_frame"):
        # 方向缺失有专门条款（8.4-5），报的必须是那条，不是通用缺项。
        # 若这里也报 8.4-4，第三方按条款号回溯会找不到「为什么方向单独重要」。
        report = assert_rejected(base_frame(), "TL-8.4-5", dataset=_ds_with(transforms=[t]))
        assert not any(f.clause == "TL-8.4-4" for f in report.findings), [
            (f.code, f.clause) for f in report.findings
        ]
        return
    assert_rejected(base_frame(), "TL-8.4-4", dataset=_ds_with(transforms=[t]))


def test_complete_transform_is_accepted():
    """TL-8.4-4 的对照：九项齐全的变换必须通过。

    没有这条，一个「见 transforms 就报错」的实现也能通过上面所有反例。
    """
    assert_accepted(base_frame(), dataset=_ds_with(transforms=[GOOD_TRANSFORM]))


@pytest.mark.parametrize("drop", ["source_frame", "target_frame"])
def test_transform_without_direction_is_rejected(drop):
    """TL-8.4-5：没有方向的变换视为不可用，接收方不得猜。

    `p_B = R_BA · p_A + t_BA` 与 `p_A = R_AB · p_B + t_AB` 互为逆变换。
    只写「变换矩阵」，不同实现者会取不同方向，结果差一个旋转。
    """
    t = {k: v for k, v in GOOD_TRANSFORM.items() if k != drop}
    assert_rejected(base_frame(), "TL-8.4-5", dataset=_ds_with(transforms=[t]))


def test_transform_with_unknown_rotation_representation_is_rejected():
    """TL-8.4-4：旋转表示必须是三种之一。

    `matrix`（没写维度）和 `rotvec` 是最常见的两种混入——前者不知道是 3×3
    还是 4×4，后者不在本规范支持的表示里。
    """
    t = {**GOOD_TRANSFORM, "rotation_representation": "rotvec"}
    assert_rejected(base_frame(), "TL-8.4-4", dataset=_ds_with(transforms=[t]))


def test_transform_with_bad_translation_unit_is_rejected():
    """TL-8.4-4：平移单位必须是附录 A 的 UCUM 码。

    变换的平移量单位不声明，跨数据集拼接时会出现「差 1000 倍」的错误，
    而且它只在两个数据集混用时才暴露。
    """
    t = {**GOOD_TRANSFORM, "translation_unit": "meter"}
    assert_rejected(base_frame(), "TL-8.4-4", dataset=_ds_with(transforms=[t]))


def test_transform_referencing_undeclared_frame_is_rejected():
    """TL-8.4-4 / 8.1-1：变换引用的坐标系必须存在。

    引用了不存在的 frame_id 时，这个变换无处落地——它描述的是一条
    不存在的边。
    """
    t = {**GOOD_TRANSFORM, "target_frame": "ghost_frame"}
    assert_rejected(base_frame(), "TL-8.1-1", dataset=_ds_with(transforms=[t]))


def test_absent_transforms_is_accepted():
    """TL-8.4-1 的边界：`transforms` 整个可以缺席。

    「记录变换」和「执行变换」是两个责任，前者是数据格式的一部分，
    后者是工具链的工作。没有变换的数据集完全合法——只是不能做跨传感器
    的原始分量比较（TL-8.4-2）。
    """
    assert_accepted(base_frame(), dataset=_ds_with())
