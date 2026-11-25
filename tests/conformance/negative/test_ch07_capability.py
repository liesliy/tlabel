"""第 7 章 —— 能力与可用性状态的反向测试。

覆盖 TL-7.1-1 / 7.2-1 / 7.2-2 / 7.2-3 / 7.2-4 / 7.2-5 / 7.3-1 到 7.3-4。

本章是整份规范里最容易被实现成「差不多就行」的部分：能力、值、状态三者
互相约束，任意两个之间都可能矛盾。这组测试的核心方法是**成对**——
每一条违规都配一条只差一个条件的合法情形，否则「见某字段就报错」的实现
也能全绿。
"""

from __future__ import annotations

import copy

import pytest

from conftest import CORE_CAPS, base_dataset, base_frame, assert_accepted, assert_rejected, run


def _caps(**overrides):
    caps = dict(CORE_CAPS)
    caps.update(overrides)
    return caps


def _migrated(dataset=None):
    """给数据集补一份最小迁移审计（TL-7.2-2）。"""
    ds = copy.deepcopy(dataset) if dataset is not None else base_dataset()
    ds["migration"] = {"source_spec_version": "2.1", "tool": "tlabel-migrate"}
    return ds


# ─────────────────────────────────────────────────────────────
# 7.1 能力声明
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
def test_missing_capability_declaration_is_rejected(field):
    """TL-7.1-1：每个核心字段都必须声明能力。

    缺键不能当 `false` 处理。「不知道有没有这个能力」和「确定没有」是两件
    不同的事：前者要去问生产者，后者可以直接跳过。把缺键当 false 会让
    「忘了声明」永久藏在数据里，下游永远不知道自己在看一份不完整的能力表。
    """
    dataset = base_dataset()
    del dataset["capabilities"][field]
    assert_rejected(base_frame(), "TL-7.1-1", dataset=dataset)


def test_complete_capability_table_is_accepted():
    """TL-7.1-1 的对照：八个键齐全时必须通过。

    防止把「capabilities 里有任何键就报错」写成实现。
    """
    assert_accepted(base_frame(), dataset=base_dataset())


# ─────────────────────────────────────────────────────────────
# 7.2 可用性状态
# ─────────────────────────────────────────────────────────────


def test_migration_state_in_native_data_is_rejected():
    """TL-7.2-3：校验器必须拒绝没有迁移审计的迁移态。

    这个状态说的是「旧数据无法判断当时状态」。原生 V3.0 生产者说这句话
    是矛盾的——它就在生产现场，没有「当时」可言。允许它出现，等于给
    「懒得判断」开了一个合法出口。

    引 7.2-3 而不是 7.2-1：7.2-1 是「生产者不得产出」的义务，7.2-3 才是
    「校验器必须拒绝」——这里执行的是后者。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": "unknown_for_migration"}
    assert_rejected(frame, "TL-7.2-3")


def test_migration_state_without_audit_is_rejected():
    """TL-7.2-2：有迁移态就必须有源规范版本。

    没有源版本的迁移态无法追溯「这个值是从哪个旧字段来的」，等于把
    不确定性留在了数据里。审计不是为了记账，是为了让下游能判断
    「这个 null 是 V2 本来就没有，还是迁移时丢了」。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": "unknown_for_migration"}
    dataset = base_dataset()
    dataset["migration"] = {"tool": "tlabel-migrate"}  # 缺 source_spec_version
    assert_rejected(frame, "TL-7.2-2", dataset=dataset)


def test_migration_state_with_audit_is_not_reported_as_7_2_2():
    """TL-7.2-2 的对照：补齐源版本后不再报这条。

    7.2-3 是「无审计的迁移态」，7.2-2 是「有审计但缺源版本」。补上源版本
    只解决后者，前者仍然成立（这条数据本来就带迁移态）。所以这里只检查
    7.2-2 消失，不要求 7.2-3 也消失。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": "unknown_for_migration"}
    report = run(frame, _migrated())
    assert not any(f.clause == "TL-7.2-2" for f in report.findings), [
        (f.code, f.clause) for f in report.findings
    ]


@pytest.mark.parametrize("state", ["not_annotated", "not_applicable"])
def test_extension_states_are_recognized(state):
    """TL-7.2-4：Core 校验器必须认识这两个扩展状态。

    它们回答的是「工作流走到哪一步」和「这一帧是否适用」，不是「为什么
    测不到」。若校验器把它判为未知枚举，允许人工标注的 `contact` 就无法
    表达「图像已有、标注未完成」——而这是真实存在的数据形态。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": state}
    report = run(frame)
    assert report.ok, [(f.code, f.clause, f.message) for f in report.findings]


@pytest.mark.parametrize("state", ["not_acquired", "invalid", "not_annotated", "not_applicable"])
def test_state_with_non_null_value_is_rejected(state):
    """TL-7.2-5：声明了缺失状态却带非 null 值。

    这等于说「我没拿到」和「这就是我拿到的值」同时成立。下游按状态
    跳过这一帧时，那个值就被静默丢弃了——而它可能是真实的。
    """
    frame = base_frame(contact=True)
    frame["field_status"] = {"contact": state}
    assert_rejected(frame, "TL-7.2-5")


def test_state_with_null_value_is_accepted():
    """TL-7.2-5 的对照：状态 + null 是正常组合。

    这是「有能力但这一帧没拿到」的标准写法，也是规范鼓励的写法。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": "not_acquired"}
    assert_accepted(frame)


# ─────────────────────────────────────────────────────────────
# 7.3 能力、值、状态的一致性
# ─────────────────────────────────────────────────────────────


def test_value_without_capability_is_rejected():
    """TL-7.3-1：能力为 false 的字段必须为 null。

    给了值说明两件事之一：能力声明错了，或者值是从别处推出来的。
    两种都要拦——前者会让下游以为这个量测不到，后者是代理量混入核心字段。
    """
    assert_rejected(
        base_frame(contact=True),
        "TL-7.3-1",
        dataset=base_dataset(capabilities=_caps(contact=False)),
    )


def test_capability_with_null_and_no_status_is_rejected():
    """TL-7.3-2：有能力、值为 null，却没有状态。

    null 本身不说明为什么缺。未采集（去补采）、无效（去查传感器）、
    未标注（去继续标注）是三种完全不同的后续动作，缺了状态就无法分派。
    """
    frame = base_frame(contact=None)
    assert_rejected(frame, "TL-7.3-2")


def test_capability_false_with_null_and_no_status_is_accepted():
    """TL-7.3-3：能力为 false、值为 null、不写状态 —— 这是合法的。

    能力声明本身已经完整表达了「不支持」，再要求一个状态只是冗余。
    这条与 7.3-2 只差一个 `capability` 的取值，所以必须成对测——
    一个「见 null 就要求状态」的实现会误杀所有无该传感器的数据集。
    """
    assert_accepted(
        base_frame(contact=None),
        dataset=base_dataset(capabilities=_caps(contact=False)),
    )


def test_capability_true_with_unsupported_is_rejected():
    """TL-7.3-4：有能力却声明 unsupported。

    能力说「有」，状态说「不支持」。这条常出现在生产者想临时屏蔽某个
    字段时——正确做法是改能力声明，不是改状态。改状态会让「能力表」
    与实际不符，而能力表是数据集级的，影响所有帧。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": "unsupported"}
    assert_rejected(frame, "TL-7.3-4")


def test_capability_false_with_unsupported_is_accepted():
    """TL-7.3-4 的对照：能力 false + unsupported 是一致的，必须通过。

    与上一条只差 capability 的取值。这两条配对，才能区分「按一致性判断」
    和「见 unsupported 就报错」。
    """
    frame = base_frame(contact=None)
    frame["field_status"] = {"contact": "unsupported"}
    assert_accepted(frame, dataset=base_dataset(capabilities=_caps(contact=False)))
