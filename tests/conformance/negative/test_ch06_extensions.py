"""第 6 章 —— 扩展字段与弃用字段的反向测试。

覆盖 TL-6.2-1 / 6.3-1 / 6.5-1 / 6.5-2。

本章的四个字段都是「降级」的：`contact_region` 和 `texture_class` 从核心降到
扩展，`object_deformation` 因为语义不清被降级，`force_magnitude` 等三个字段
直接弃用。降级不等于不管——它们仍在数据里，仍必须可被机器判定。
"""

from __future__ import annotations

import pytest

from conftest import base_dataset, base_frame, assert_accepted, assert_rejected, run


# ─────────────────────────────────────────────────────────────
# 6.2 contact_region
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["palm", "PALMAR", "finger", "palmar ", "", "other "])
def test_contact_region_outside_closed_set_is_rejected(bad):
    """TL-6.2-1：`contact_region` 必须是七个值之一。

    这里特意放了 `PALMAR`（大小写）、`palmar `（尾空格）、`other `
    ——这三个在人工录入里极常见，但都不是同一个字符串。闭集的意思是
    逐字符相等，不是「看起来一样」。
    """
    assert_rejected(base_frame(contact_region=bad), "TL-6.2-1")


@pytest.mark.parametrize(
    "good", ["palmar", "digital", "lateral", "proximal", "distal", "dorsal", "other"]
)
def test_all_seven_contact_regions_are_accepted(good):
    """TL-6.2-1 的对照：七个值逐个验证。

    只测一个合法值会漏掉「实现里少写了一个值」的情况——而漏写 `other`
    是最容易发生的，因为它看起来像兜底值。
    """
    assert_accepted(base_frame(contact_region=good))


def test_absent_contact_region_is_accepted():
    """TL-6.2-1 的边界：字段可以整个缺席（值为 null）。

    原文是「MUST be one of the seven listed values **when present**」——
    缺席是合法的，因为这是扩展字段，Core 实现不要求产出它。
    """
    assert_accepted(base_frame(contact_region=None))


# ─────────────────────────────────────────────────────────────
# 6.3 texture_class
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["coarse", "smoothness", "rough ", "SMOOTH", "textured"])
def test_texture_class_outside_closed_set_is_rejected(bad):
    """TL-6.3-1：`texture_class` 必须是六个值之一。

    `coarse` 是最容易混进来的——它和 `rough` 在英文里几乎同义，但不在闭集内。
    """
    assert_rejected(base_frame(texture_class=bad), "TL-6.3-1")


@pytest.mark.parametrize(
    "good", ["smooth", "rough", "granular", "fibrous", "sticky", "slippery"]
)
def test_all_six_texture_classes_are_accepted(good):
    """TL-6.3-1 的对照：六个值逐个验证。"""
    assert_accepted(base_frame(texture_class=good))


# ─────────────────────────────────────────────────────────────
# 6.5 弃用字段
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("field", ["force_magnitude", "confidence", "compliance_level"])
def test_deprecated_field_in_native_data_is_rejected(field):
    """TL-6.5-1：原生 V3.0 数据不得让弃用字段带非 null 值。

    保留键名是为了让旧数据可读，不是为了让人继续写。这三个字段各有各的
    问题：`force_magnitude` 的名称与力学惯例冲突（magnitude 是 ‖F‖，
    不是法向分量）；`confidence` 的「对什么置信」没有定义；`compliance_level`
    可以由 capabilities + field_status 算出来，存下来只会制造两个真相源。
    """
    assert_rejected(base_frame(**{field: 0.9}), "TL-6.5-1")


def test_deprecated_field_with_null_is_accepted():
    """TL-6.5-1 的边界：键在、值为 null，是合法的。

    序列化器通常会输出完整键集（TL-4.4-1 只要求核心字段，但很多实现
    一并输出弃用字段）。值为 null 说明「没在用」，这符合要求。
    """
    assert_accepted(base_frame(force_magnitude=None))


def test_deprecated_field_in_migrated_data_is_only_a_warning():
    """TL-6.5-2：迁移数据里的弃用字段只报 warning，不判失败。

    分界线是数据来源，不是字段本身。V2 里 `force_magnitude` 是合法字段，
    迁移工具保留原值是为了可追溯（TL-11.3 要求不推断、不丢失）。若这里
    也报 error，每一份迁移数据都会被判为不合规——而它们并没有做错什么。

    没有这条对照，一个「见弃用字段就报 error」的实现也能通过上面所有测试，
    而它会让整个迁移路径失效。
    """
    dataset = base_dataset()
    dataset["migration"] = {"source_spec_version": "2.1", "tool": "tlabel-migrate"}
    report = run(base_frame(force_magnitude=3.1), dataset)

    hits = [f for f in report.findings if f.code == "DEPRECATED_FIELD_POPULATED"]
    assert hits, "迁移数据里的弃用字段必须被报告"
    assert all(f.outcome == "warning" for f in hits), [
        (f.code, f.outcome) for f in hits
    ]
    assert all(f.clause == "TL-6.5-2" for f in hits), [f.clause for f in hits]


def test_deprecated_field_in_native_data_is_not_a_warning():
    """TL-6.5-2 的反向对照：原生数据里必须是 error，不能降级。

    与上一条配对，确保分级是按数据来源分，不是按字段名分。
    """
    report = run(base_frame(confidence=0.8))
    hits = [f for f in report.findings if f.code == "DEPRECATED_FIELD_POPULATED"]
    assert hits
    assert all(f.outcome == "error" for f in hits), [(f.code, f.outcome) for f in hits]
    assert all(f.clause == "TL-6.5-1" for f in hits), [f.clause for f in hits]
