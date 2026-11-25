"""第 5.1–5.2 章：`contact` 与 `contact_centroid` 的负向测试。

每个测试只改基底一处，改的那处就是它要证明的那件事。断言同时要求命中条款号，
因为「有错」不等于「这条条款被验证了」——一处无关错误也能让裸断言变绿。
"""

from __future__ import annotations

from conftest import (
    CORE_CAPS,
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


# ── 5.1 contact ──────────────────────────────────────────────

def test_contact_validator_rejects_non_tristate():
    """TL-5.1-1 / TL-5.1-5：三值之外的 `contact` 必须被拒。

    TL-5.1-1 约束生产者（必须填三个值之一），TL-5.1-5 约束校验器（必须拒绝
    其他值）。两条是同一条规则的产出侧与检查侧，一条测试同时验证两者。

    finding 引的是 5.1-5——校验器正在履行的就是这条义务，引 5.1-1 会让按
    条款号回溯的人看到一条生产者义务而找不到验证要求。

    断言错误码而不只是「被拒」，是为了排除「碰巧报了别的错」。
    """
    rep = assert_rejected(base_frame(contact=0.5), "TL-5.1-5")
    assert "TYPE_MISMATCH" in [f.code for f in rep.findings], (
        f"TL-5.1-5 要求校验器显式拒绝非三态，实际码：{[(f.code, f.clause) for f in rep.findings]}"
    )


def test_contact_string_value_is_rejected():
    """TL-5.1-5：`"true"` 是字符串，不是布尔。

    和上面那条分开，是因为把 `if not isinstance(value, bool)` 写成
    `if value not in (True, False, None)` 时，`0.5` 会被拦下而 `"true"` 会
    漏过去——两者走的是不同的分支，必须分别覆盖。
    """
    assert_rejected(base_frame(contact="true"), "TL-5.1-5")


def test_contact_value_under_false_capability_is_rejected():
    """TL-5.1-3：没有接触判定能力时不得给出 `false`。

    `false` 是一个断言（"确定没有接触"），不是"没数据"的默认值。能力声明为
    false 时，帧值只能是 null。这一条的实际判定由 TL-7.3-2 完成，所以断言的
    条款号是 7.3-2——5.1-3 只能部分验证，理由记在 requirements.yaml。
    """
    assert_rejected(
        base_frame(contact=False),
        "TL-7.3-1",
        base_dataset(capabilities=_caps(contact=False)),
    )


def test_human_annotated_contact_without_annotator_is_rejected():
    """TL-5.1-4：人工标注的 contact 必须带 annotator_id。

    报的是 5.1-4 而不是 TL-9.3-2 的通则——规范专门为这个字段写了这条，
    按条款号回溯的人应该在 5.1-4 下看到验证。
    """
    fr = base_frame()
    fr["field_provenance"]["contact"] = {"type": "human_annotation"}
    assert_rejected(fr, "TL-5.1-4")


def test_human_annotated_contact_with_annotator_is_accepted():
    """TL-5.1-4 的对照：补上 annotator_id 后必须通过。

    只有负向断言时，一个「见 human_annotation 就报错」的实现也能全绿。
    """
    fr = base_frame()
    fr["field_provenance"]["contact"] = {
        "type": "human_annotation",
        "annotator_id": "annotator-07",
    }
    assert_accepted(fr)


# ── 5.2 contact_centroid ─────────────────────────────────────

def test_centroid_with_one_component_is_rejected():
    """TL-5.2-1：恰好 2 个元素，少一个也不行。"""
    assert_rejected(base_frame(contact_centroid=[10.0]), "TL-5.2-1")


def test_centroid_with_three_components_is_rejected():
    """TL-5.2-1：3 个也不行——三维是 force_vector 的形态，不是平面位置。"""
    assert_rejected(base_frame(contact_centroid=[10.0, 20.0, 30.0]), "TL-5.2-1")


def test_centroid_referencing_3d_frame_is_rejected():
    """TL-5.2-3：centroid 必须引用 dimension=2 的坐标系。

    报 5.2-3 而不是通用的 TL-8.1-3：规范专门为 centroid 写了这条。
    """
    assert_rejected(base_frame(contact_centroid_frame="sensor_0"), "TL-5.2-3")


def test_centroid_without_method_is_rejected():
    """TL-5.2-5：非 null centroid 必须声明 centroid_method。

    几何中心与压力加权中心对同一接触区会给出不同的数，都叫
    `contact_centroid`，不声明方法下游无法判断这个坐标代表什么。
    """
    fr = base_frame()
    del fr["field_provenance"]["contact_centroid"]["centroid_method"]
    assert_rejected(fr, "TL-5.2-5")


def test_centroid_method_outside_closed_set_is_rejected():
    """TL-5.2-5：centroid_method 只能是 geometric 或 pressure_weighted。"""
    fr = base_frame()
    fr["field_provenance"]["contact_centroid"]["centroid_method"] = "median"
    assert_rejected(fr, "TL-5.2-5")


def test_pressure_weighted_centroid_without_weighting_quantity_is_rejected():
    """TL-5.2-6：压力加权中心必须说明加权量。

    没有加权量，"加权"两个字无法复现——用压强加权和用温度加权是两回事。
    """
    fr = base_frame()
    del fr["field_provenance"]["contact_centroid"]["weighting_quantity"]
    assert_rejected(fr, "TL-5.2-6")


def test_geometric_centroid_without_weighting_quantity_is_accepted():
    """TL-5.2-6 的边界：只有 pressure_weighted 才要求加权量。

    几何中心不需要权重，要求它反而会让合法数据被拒。
    """
    fr = base_frame()
    fr["field_provenance"]["contact_centroid"]["centroid_method"] = "geometric"
    del fr["field_provenance"]["contact_centroid"]["weighting_quantity"]
    assert_accepted(fr)


def test_contact_true_with_null_centroid_under_true_capability_is_rejected():
    """TL-5.2-4 的可验证部分：有能力却给 null，必须带可用性状态。

    规范里 5.2-4 是 SHOULD，但「能力为 true、值为 null、无 field_status」
    被 TL-7.3-3 的 MUST 覆盖。5.2-4 没有独立可验证的场景——能力为 false 时
    缺值是正常形态，能力为 true 时缺值已被 MUST 判错。
    """
    fr = base_frame(contact_centroid=None)
    fr["field_provenance"].pop("contact_centroid")
    assert_rejected(
        fr, "TL-7.3-2", base_dataset(capabilities=_caps(contact_centroid=True))
    )


# ─────────────────────────────────────────────────────────────
# TL-5.2-4（SHOULD 级）
# ─────────────────────────────────────────────────────────────


def test_contact_true_without_centroid_warns():
    """TL-5.2-4：有接触却没有接触位置，应提示。

    SHOULD 级所以只报 warning。这里必须给 `contact_centroid` 一个状态，
    否则同一个根因会先被 TL-7.3-2 报成 error——测试要验的是 5.2-4 这条，
    不是 7.3-2 那条。
    """
    frame = base_frame(contact=True, contact_centroid=None)
    frame["field_status"] = {"contact_centroid": "not_acquired"}
    report = run(frame)
    hits = [f for f in report.findings if f.clause == "TL-5.2-4"]
    assert hits, [(f.code, f.clause) for f in report.findings]
    assert all(f.outcome == "warning" for f in hits)


def test_no_contact_with_centroid_warns():
    """TL-5.2-4：无接触却有接触位置，同样提示。

    没有接触时「接触中心」没有物理含义——这个值要么是残留，要么是
    contact 判错了。两种都值得看一眼。
    """
    report = run(base_frame(contact=False))
    hits = [f for f in report.findings if f.clause == "TL-5.2-4"]
    assert hits
    assert all(f.outcome == "warning" for f in hits)


def test_contact_and_centroid_consistent_does_not_warn():
    """TL-5.2-4 的对照：两者一致时不得报。

    防止把「只要有 contact 就检查」写成无条件 warning。
    """
    report = run(base_frame(contact=True))
    assert not any(f.clause == "TL-5.2-4" for f in report.findings)


def test_null_contact_without_centroid_does_not_warn():
    """TL-5.2-4 的边界：contact 为 null 时两种情况都合规。

    规范原文只说了 `true` 和 `false` 两种情形，`null` 不在约束范围内——
    接触未知时，centroid 有没有都不矛盾。
    """
    frame = base_frame(contact=None, contact_centroid=None)
    frame["field_status"] = {"contact": "not_acquired", "contact_centroid": "not_acquired"}
    report = run(frame)
    assert not any(f.clause == "TL-5.2-4" for f in report.findings)
