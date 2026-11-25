"""第 4 章 —— 表示层约束的反向测试。

覆盖 TL-4.2-1 / 4.2-2 / 4.3-1 / 4.4-1 / 4.4-2 / 4.5-3。

这一章管的是「数据长什么样」，不是「数据是什么意思」。所以这里的反例都
不涉及物理语义——它们是同一个物理事实的两种写法，其中一种是错的。
"""

from __future__ import annotations

import copy

import pytest

from conftest import (
    CORE_CAPS,
    base_dataset,
    base_frame,
    assert_accepted,
    assert_rejected,
    run,
)


def _caps(**overrides):
    caps = dict(CORE_CAPS)
    caps.update(overrides)
    return caps


# ─────────────────────────────────────────────────────────────
# 4.2 单位
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_unit",
    ["newton", "N ", "N·m", "Nm", "millimeter", "mm/sec", "°C", "celsius", "", "dimensionless"],
)
def test_non_ucum_unit_in_sensor_specific_is_rejected(bad_unit):
    """TL-4.2-1：自报的单位也必须是附录 A 的 UCUM 码。

    这条挡住的是「把像素写成长度」最常见的入口。附录 A 的「常见错误写法」
    列里每一个都在这里跑一遍——单位是最容易「看起来对」的地方，
    `N·m`（中点）和 `N.m`（点号）在人眼里没差别，在机器眼里是两个字符串。
    """
    frame = base_frame(
        sensor_specific={
            "elastomer_displacement": {
                "value": 0.3,
                "unit": bad_unit,
                "definition": "弹性体法向位移",
                "method_id": "conf:gel_displacement_v1",
                "namespace": "org.touchlabel.gelsight",
            }
        }
    )
    # sensor_specific 是扩展容器，其条目的结构约束只在 Extended 等级生效
    # （TL-9.5-3）。Core 等级不要求实现认识扩展字段——这是分层的目的。
    assert_rejected(frame, "TL-4.2-1", level="extended")


def test_ucum_unit_in_sensor_specific_is_accepted():
    """TL-4.2-1 的对照：`N.m` 是合法码。

    只测非法值时，一个「见 unit 就报错」的实现也能全绿，而它会拒掉所有
    合法的自报单位——那等于禁止扩展字段带单位。
    """
    frame = base_frame(
        sensor_specific={
            "elastomer_torque": {
                "value": 0.3,
                "unit": "N.m",
                "definition": "弹性体扭矩",
                "method_id": "conf:gel_torque_v1",
                "namespace": "org.touchlabel.gelsight",
            }
        }
    )
    assert_accepted(frame, level="extended")


@pytest.mark.parametrize("key", ["units", "unit_overrides", "field_units"])
def test_dataset_level_unit_table_is_rejected(key):
    """TL-4.2-2：单位由字段定义固定，不得用数据集级元数据改写。

    V2 的 `contact_centroid` 允许「像素或 mm」，下游每次都要问「这个 310 是
    什么」。数据集级单位表会把同样的歧义重新引进来，而且藏得更深——
    它不在帧里，读取单条记录时看不到。
    """
    assert_rejected(base_frame(), "TL-4.2-2", dataset=base_dataset(**{key: {"normal_force": "mN"}}))


def test_plain_dataset_without_unit_table_is_accepted():
    """TL-4.2-2 的对照：没有单位表的数据集必须通过。

    防止把「数据集里出现任何 dict 键就报错」写成实现。
    """
    assert_accepted(base_frame(), dataset=base_dataset())


# ─────────────────────────────────────────────────────────────
# 4.3 数值表示
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scalar_is_rejected(bad):
    """TL-4.3-1：NaN / Inf 不是合法值，它们的含义等同缺失。

    规范要求用 `null` + 状态表达缺失，而不是用 NaN。原因是 NaN 会静默
    污染下游统计——`np.mean` 遇到 NaN 返回 NaN，整批数据无声报废。
    """
    assert_rejected(base_frame(normal_force=bad), "TL-4.3-1")


def test_non_finite_vector_component_is_rejected():
    """TL-4.3-1：向量分量同样不得为非有限值。

    单测标量会漏掉向量路径——它们走的是不同的分支（`vector` vs `scalar`）。
    """
    assert_rejected(base_frame(force_vector=[0.42, float("nan"), 3.10]), "TL-4.3-1")


def test_non_finite_does_not_produce_duplicate_findings():
    """TL-4.3-1 与 5.3 的关系：非有限值只报一次，不再叠加语义错误。

    NaN 参与任何比较都返回 False，把它送进 `|F·n|` 复算会得到「不一致」，
    而真实原因是「无法比较」。一个根因报两条 finding，第三方按条款回溯时
    会以为有两处独立问题。
    """
    report = run(base_frame(normal_force=float("nan")))
    codes = [f.code for f in report.findings]
    assert codes == ["NON_FINITE_VALUE"], codes


# ─────────────────────────────────────────────────────────────
# 4.4 键存在性
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "field",
    [
        "contact",
        "contact_centroid",
        "normal_force",
        "force_vector",
        "torque_vector",
        "slip_event",
        "slip_velocity",
        "temperature",
    ],
)
def test_missing_core_key_is_rejected(field):
    """TL-4.4-1：8 个核心字段的键必须全部出现，即使值是 null。

    若允许省略键，「没有温度传感器」和「有温度传感器但这一帧没测到」
    就无法区分——前者 capability=false，后者 capability=true + 状态。
    两种情况的处理方式完全不同：一个不用管，一个要补采。
    """
    frame = base_frame()
    del frame[field]
    assert_rejected(frame, "TL-4.4-1")


def test_null_value_with_key_present_is_accepted():
    """TL-4.4-2 的对照：键在、值是 null，这是合法的缺失表达。

    没有这条对照，一个「见 null 就报 4.4-1」的实现也能全绿，
    而它会禁止所有「有能力但这一帧没测到」的数据。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": "not_acquired"}
    assert_accepted(frame)


def test_missing_key_differs_from_null_value():
    """TL-4.4-2：缺键与 null 必须产生不同结果。

    这是对「等价性」的直接检验：同一个字段，键在值是 null 时通过，
    键不在时报错。如果两者结果相同，说明实现把「没有这个键」当成了 null。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": "not_acquired"}
    assert run(frame).ok

    broken = copy.deepcopy(frame)
    del broken["contact"]
    assert not run(broken).ok


# ─────────────────────────────────────────────────────────────
# 4.5 缺失的表达
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("state", ["not_acquired", "invalid", "not_annotated"])
def test_capability_false_with_capability_claiming_state_is_rejected(state):
    """TL-4.5-3：能力为 false，却声明「有能力但没拿到」的状态。

    这三个状态都在说「有本事，这次没拿到」，而能力声明说的是「没这个本事」。
    同一次缺失被表达了两次，而且两次互相矛盾——下游无法判断该不该去补采。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": state}
    assert_rejected(frame, "TL-4.5-3", dataset=base_dataset(capabilities=_caps(contact=False)))


@pytest.mark.parametrize("state", ["unsupported", "not_applicable", "unknown_for_migration"])
def test_capability_false_with_non_claiming_state_is_accepted(state):
    """TL-4.5-3 的边界：不声称能力存在的状态与 capability=false 可以并存。

    规范原文说的是「一致」，不是「相同」。`unsupported` 是「不支持」，
    `not_applicable` 是「这帧无意义」，`unknown_for_migration` 是「不知道
    当时是什么状态」（TL-7.2）——三者都不声称有能力，只是从不同角度描述
    「没有可用值」。

    没有这条对照，一个「capability=false 时禁止任何状态」的实现也能全绿，
    而它会误杀所有迁移数据——迁移工具保守填 false 是常态。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": state}
    dataset = base_dataset(capabilities=_caps(contact=False))
    if state == "unknown_for_migration":
        dataset["migration"] = {"source_spec_version": "2.1", "tool": "tlabel-migrate"}
    report = run(frame, dataset)
    assert not any(f.clause == "TL-4.5-3" for f in report.findings), [
        (f.code, f.clause) for f in report.findings
    ]


def test_capability_true_with_unsupported_is_rejected():
    """TL-7.3-4：有能力却声明 unsupported。

    反过来也矛盾：能力说「有」，状态说「不支持」。这条常出现在生产者
    想临时屏蔽某个字段时——正确的做法是改能力声明，不是改状态。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": "unsupported"}
    assert_rejected(frame, "TL-7.3-4")
