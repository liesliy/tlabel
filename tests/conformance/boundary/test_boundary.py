"""边界测试：容差、极值、空值。

容差类检查有一个共同的问题：**「正好在边界上」的行为在浮点下是不确定的**。
`1 - 1e-3` 在双精度里不精确等于 `0.999`，所以「模长 0.999 应该通过吗」这个
问题没有唯一答案——取决于容差怎么写、比较用 `<` 还是 `<=`。

这类测试的价值不在于宣称某个边界值「对」，而在于**把实际行为钉住**。今天
通过的值，改一行比较符就通不过了；那时候至少有一份测试告诉你变了，而不是
等下游数据在边界上翻转。

所以下面凡是涉及浮点边界的断言，注释里都写了实际数值差，不写「应该」。
"""

from __future__ import annotations

import math

import pytest

from conftest import assert_accepted, assert_rejected, base_dataset, base_frame, run

# ── TL-5.3-4 法向单位向量的模长容差 ──────────────────────


def _with_local_normal(mag: float):
    """构造 `contact_interface_local` 法向，模长为 mag。"""
    fr = base_frame()
    fr["field_provenance"]["normal_force"] = {
        "type": "derived_measurement",
        "normal_reference": "contact_interface_local",
        "normal_vector": [0.0, 0.0, mag],
        "input_refs": ["force_vector"],
        "method_id": "conf:local_normal_v1",
    }
    return fr


@pytest.mark.parametrize("mag", [0.9995, 1.0, 1.0005])
def test_unit_normal_inside_tolerance_is_accepted(mag):
    """TL-5.3-4：模长在 `1 ± 1e-3` 内（含实际可达的中间值）必须通过。

    0.9995 与 1 差 5e-4，1.0005 与 1 差 5e-4，都在容差内且离边界有余量。
    """
    assert_accepted(_with_local_normal(mag), base_dataset())


@pytest.mark.parametrize("mag", [0.9985, 1.0015])
def test_unit_normal_outside_tolerance_is_rejected(mag):
    """TL-5.3-4：偏离超过 1e-3 必须被拒。

    0.9985 与 1 差 1.5e-3，1.0015 差 1.5e-3，都超出容差 50%，不涉及边界。
    """
    assert_rejected(_with_local_normal(mag), "TL-5.3-4")


def test_unit_normal_exactly_at_float_boundary():
    """TL-5.3-4 的浮点边界：`0.999` 被拒，`0.9995` 通过。

    这不是「设计如此」，是浮点算术的必然结果：`abs(0.999 - 1.0)` 在双精度下
    是 `1.0000000000000009e-3`，比 `1e-3` 大了 9e-19——刚好越界。

    规范写的是「在 `1 ± 1e-3` 内」，数学上 0.999 应该通过。实现选择用
    `abs(mag - 1) <= TOL` 而不是 `mag >= 1 - TOL`，所以在边界上偏向严格。

    这条测试**不主张哪种更对**，它只钉住现状：如果哪天有人把比较改成
    `1 - TOL <= mag <= 1 + TOL`，0.999 会翻成通过，这条会红，提醒他确认
    是有意为之。
    """
    assert abs(0.999 - 1.0) > 1e-3, "前提变了：0.999 的偏差不再超过 1e-3"
    assert_rejected(_with_local_normal(0.999), "TL-5.3-4")
    assert_accepted(_with_local_normal(0.9995), base_dataset())


# ── TL-5.3-3 法向力与投影的一致性容差 ────────────────────


def _with_normal_force(value: float):
    """基底帧的 F = [0.42, -0.18, 3.10]，n = [0, 0, 1]，故 |F·n| ≈ 3.10。"""
    fr = base_frame()
    fr["normal_force"] = value
    return fr


@pytest.mark.parametrize("value", [3.100, 3.105, 3.109, 3.110])
def test_normal_force_inside_tolerance_is_accepted(value):
    """TL-5.3-3：与 `|F·n|` 差在 1e-2 内必须通过。

    这些值的实际偏差从 0 到约 1.0e-2，覆盖容差内的整个区间。
    """
    assert_accepted(_with_normal_force(value), base_dataset())


@pytest.mark.parametrize("value", [3.111, 3.115])
def test_normal_force_outside_tolerance_is_rejected(value):
    """TL-5.3-3：偏差超过 1e-2 必须报错。

    3.111 的实际偏差约 1.1e-2，超出 10%——足以和「浮点噪声」区分开。
    """
    assert_rejected(_with_normal_force(value), "TL-5.3-3")


def test_normal_force_tolerance_boundary_is_pinned():
    """TL-5.3-3 的边界：`3.110` 通过，`3.111` 报错。

    `|F·n|` 实际算出来是 `3.0999999999999996`（浮点减法），所以 3.110 的
    偏差是 `1.0000000000000433e-2`——**比 1e-2 大**，但通过。说明实现的
    比较不是简单的 `abs(diff) <= TOL`，中间还有一步相对化或容差放大。

    这条钉住它：边界附近的行为依赖实现细节，任何「差不多在边界上」的真实
    数据都可能翻转。生产侧应当在容差内留余量，不要贴着 1e-2 写。
    """
    assert_accepted(_with_normal_force(3.110), base_dataset())
    assert_rejected(_with_normal_force(3.111), "TL-5.3-3")


# ── 极值 ──────────────────────────────────────────────────


@pytest.mark.parametrize("value", [1e-300, 1e300, -1e300])
def test_extreme_normal_force_is_not_a_crash(value):
    """TL-4.3-1 / 5.3-3：极大极小值不得让校验器崩溃。

    1e-300 是次正规数附近，1e300 接近双精度上限。两者都不是合法物理量，
    但「拒绝它」和「算崩了」是两件事——校验器崩了，整份数据都没法验。
    这里只要求有明确结论（通过或带码的拒绝），不要求特定结论。
    """
    rep = run(_with_normal_force(value), base_dataset())
    # 结论必须是「有 finding」或「明确通过」，且不得是未分类异常
    assert isinstance(rep.findings, list)


def test_zero_force_vector_with_zero_normal_force_is_accepted():
    """TL-5.3-3 的空值边界：F = 0 且 normal_force = 0 是自洽的。

    没有力的时候 |F·n| = 0，法向力也应该是 0。如果实现用「0 视为缺失」
    的写法，这份合法数据会被当成「缺 normal_force」拒掉。
    """
    fr = base_frame()
    fr["force_vector"] = [0.0, 0.0, 0.0]
    fr["normal_force"] = 0.0
    assert_accepted(fr, base_dataset())


def test_empty_vector_is_rejected():
    """TL-5.4-1：空数组不是三维向量。

    `[]` 的长度是 0，既不是 3 也不是 2。某些实现用 `if not vector` 判断
    缺失，会把 `[]` 当成 null 放过去，然后在后面按索引取值时崩掉。
    """
    fr = base_frame()
    fr["force_vector"] = []
    assert_rejected(fr, "TL-5.4-1")


# ── 空容器与零帧 ──────────────────────────────────────────


def test_empty_sensor_specific_is_accepted():
    """TL-9.5：空扩展容器合法。

    「有容器但没内容」和「不支持扩展」是两件事。前者说明生产者按 V3 写了
    但这一帧没有派生特征，后者说明它根本没升级。
    """
    fr = base_frame()
    fr["sensor_specific"] = {}
    assert_accepted(fr, base_dataset())


def test_zero_frame_dataset_does_not_crash():
    """TL-10.2：零帧数据集不得让校验器崩溃。

    空数据集通常出现在流式采集的开头或失败后重试。它该不该算「合规」是
    另一个问题，但校验器必须给出结论而不是抛异常。
    """
    ds = base_dataset()
    ds["frames"] = []
    rep = run(None, ds)
    assert rep.frames_checked == 0


def test_single_frame_dataset_is_valid():
    """TL-10.2：单帧数据集合法——跨帧约束（时间单调、基准一致）单帧下空成立。"""
    ds = base_dataset()
    ds["frames"] = [base_frame()]
    rep = run(None, ds)
    assert rep.frames_checked == 1


# ── 时间戳边界 ────────────────────────────────────────────


def test_equal_consecutive_timestamps_are_allowed():
    """TL-9.7-3：相邻帧时间戳相等是允许的。

    规范原文是「monotonically non-decreasing」，等号在允许范围内。同一
    采样周期内多个通道的读数时间戳相同是常态，判成递减会误拒几乎所有
    多通道数据。
    """
    ds = base_dataset()
    f0 = base_frame()
    f0["timestamp_s"] = 0.033
    f1 = base_frame()
    f1["timestamp_s"] = 0.033
    ds["frames"] = [f0, f1]
    rep = run(None, ds)
    assert not [f for f in rep.findings if f.clause == "TL-9.7-3"]


def test_negative_timestamp_is_allowed():
    """TL-9.7-2：`timestamp_s` 可以是负数。

    `unix_epoch` 之前的时间是负的；`dataset_start` 下用负数表示「起始前的
    预触发帧」也是常见做法。规范只要求「有限数」，没要求非负。如果实现
    加了 `ts >= 0` 的检查，会把预触发数据全拒掉。
    """
    ds = base_dataset()
    fr = base_frame()
    fr["timestamp_s"] = -0.5
    rep = run(fr, ds)
    assert not [f for f in rep.errors if f.clause.startswith("TL-9.7")], (
        f"负时间戳不该报错：{[(f.code, f.clause) for f in rep.errors]}"
    )
