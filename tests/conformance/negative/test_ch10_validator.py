"""第 10 章 —— 校验器行为的测试。

这一章与前面各章不同：它约束的不是**数据**，而是**校验器**。所以这里的
断言对象是 `Report` 和 `Finding` 的结构，不是「某帧被拒」。

为什么标准要管校验器的输出形状：两个实现都能报出「这条数据错了」，但一个
给 `{"code": "TYPE_MISMATCH", "clause": "TL-5.4-1"}`，另一个给一段中文说明。
下游既写不了断言，也统计不了跨版本差异。TL-10.4 / 10.5 把这些变成可枚举的
封闭集合。
"""

from __future__ import annotations

import pytest

from tlabel.v3 import schema as S
from tlabel.v3 import validate

from conftest import base_dataset, base_frame, run


# ─────────────────────────────────────────────────────────────
# 10.1 序列化
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_key, canonical",
    [
        ("Force_Vector", "force_vector"),
        ("Normal_Force", "normal_force"),
        ("contact_Centroid", "contact_centroid"),
        ("SLIP_EVENT", "slip_event"),
        ("Timestamp_S", "timestamp_s"),
        ("Field_Provenance", "field_provenance"),
    ],
)
def test_field_name_case_variant_is_rejected(bad_key, canonical):
    """TL-10.1-3：字段名必须逐字符等于规范定义的形式。

    大小写变体是序列化层最常见的错误来源：`Force_Vector` 在 Python 里能读、
    在 JSON 里能存，但下游 `frame["force_vector"]` 会静默变成 KeyError 或
    默认值。规范说「Case variants are validation errors」，所以是 error 不是
    warning。

    报错时同时给出原始键名，下游才能知道要改哪个。
    """
    frame = base_frame()
    frame[bad_key] = frame.pop(canonical)
    report = run(frame)
    hits = [f for f in report.findings if f.clause == "TL-10.1-3"]
    assert hits, [(f.code, f.clause, f.path) for f in report.findings]
    assert hits[0].value == bad_key


@pytest.mark.parametrize(
    "bad_key, canonical",
    [
        ("Session_Id", "session_id"),
        ("Time_Base", "time_base"),
        ("Frames_Of_Reference", "frames_of_reference"),
        ("Capabilities", "capabilities"),
    ],
)
def test_dataset_level_key_case_variant_is_rejected(bad_key, canonical):
    """TL-10.1-3：数据集级键名同样要逐字符一致。

    这一组单独测，因为帧级检查覆盖不到数据集级——两者的键空间不同。漏掉
    数据集级的表现很隐蔽：`Time_Base` 写错时 `dataset.get("time_base")` 返回
    None，而 None 是**合法值**（表示原点未声明），校验器报的是「缺少
    time_base」，生产者会去补一个正确的键，同时把错键留在数据里。
    """
    dataset = base_dataset()
    dataset[bad_key] = dataset.pop(canonical)
    report = run(base_frame(), dataset)
    hits = [f for f in report.findings if f.clause == "TL-10.1-3"]
    assert hits, [(f.code, f.clause, f.path) for f in report.findings]
    assert hits[0].value == bad_key


def test_instances_key_case_variant_is_rejected():
    """TL-10.1-3：`instances` 的大小写变体——只声明一个实例也要查。

    `Instances` 拼错时，多实例数据集退化成单传感器数据集：`instance_id` 在
    帧里还在，但 `instances` 查不到，于是每帧都被报「未声明的实例」。
    症状指向实例声明，根因在键名。
    """
    dataset = base_dataset()
    dataset["Instances"] = [{"instance_id": "finger_0"}]
    frame = base_frame()
    frame["instance_id"] = "finger_0"
    report = run(frame, dataset)
    hits = [f for f in report.findings if f.clause == "TL-10.1-3"]
    assert hits, [(f.code, f.clause, f.path) for f in report.findings]
    assert hits[0].value == "Instances"


def test_unknown_key_in_sensor_specific_is_not_rejected():
    """TL-10.1-3 的边界：扩展容器里的键由生产者决定。

    `sensor_specific` 就是用来放规范没定义的量的。若校验器对不认识的键
    一律报错，扩展容器就废了——而它的存在意义正是容纳标准之外的东西。
    """
    frame = base_frame(sensor_specific={"MyCustom_Feature": {"value": 1}})
    report = run(frame)
    assert not any(f.clause == "TL-10.1-3" for f in report.findings), [
        (f.code, f.clause) for f in report.findings
    ]


def test_correct_field_names_are_accepted():
    """TL-10.1-3 的对照：规范形式必须通过。"""
    report = run(base_frame())
    assert report.ok, [(f.code, f.clause) for f in report.findings]


# ─────────────────────────────────────────────────────────────
# 10.2 符合性等级
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("level", ["structural", "core", "extended"])
def test_report_carries_conformance_level(level):
    """TL-10.2-1：报告必须写明达到的符合性等级。

    只给「通过 / 不通过」是不够的——同一个「通过」，在 Structural 下意味着
    「键和类型对了」，在 Extended 下意味着「连来源可追溯都查过了」。两者
    不能混为一谈，否则「通过」这个词就没有信息量了。
    """
    report = run(base_frame(), level=level)
    assert report.level == level


def test_structural_report_lists_what_it_did_not_check():
    """TL-10.2-3：Structural 报告必须说明哪些语义检查没做。

    「通过了」和「没查」在报告里长得一模一样——都是零 finding。一份不说明
    范围的 Structural 报告会被读成「数据符合标准」，而实际上能力/状态一致性、
    来源、上下文、坐标语义一条都没查。这些恰好是 V2 出问题的地方。
    """
    report = run(base_frame(), base_dataset(), level="structural")
    d = report.as_dict()

    assert d["not_performed"], (
        "Structural 报告必须列出未执行的检查；空清单等于宣称「全查了」"
    )
    joined = " ".join(d["not_performed"]).lower()
    for topic in ("capability", "provenance", "coordinate"):
        assert topic in joined, (
            f"未执行清单应覆盖 {topic} 相关检查，实际：{d['not_performed']}"
        )


def test_core_report_only_skips_extended_checks():
    """TL-10.2-3：Core 级只跳过 Extended 的检查，不跳过语义检查。

    这条防的是反向错误：把 Core 的 `not_performed` 也填成 Structural 那一堆，
    读报告的人会以为 Core 没查语义一致性——那 Core 就白做了。
    """
    report = run(base_frame(), base_dataset(), level="core")
    joined = " ".join(report.not_performed).lower()

    assert "extended" in joined or "sensor_specific" in joined, (
        f"Core 应说明 Extended 级检查未执行，实际：{report.not_performed}"
    )
    assert "capability" not in joined, (
        f"Core 已执行能力一致性检查，不应列在未执行清单里：{report.not_performed}"
    )


def test_structural_level_is_distinguishable_from_core():
    """TL-10.2-2 / 10.2-3：Structural 与 Core 的结论必须可区分。

    构造一份在 Core 下不通过、在 Structural 下通过的数据（能力声明与值
    矛盾）。如果两个等级给出相同结论，说明等级只是个标签，没实际作用。
    """
    frame = base_frame(contact=True)
    dataset = base_dataset()
    dataset["capabilities"] = {**dataset["capabilities"], "contact": False}

    structural = run(frame, dataset, level="structural")
    core = run(frame, dataset, level="core")

    assert structural.ok, "Structural 不查语义一致性，应通过"
    assert not core.ok, "Core 必须查到能力与值矛盾"
    assert structural.level != core.level


# ─────────────────────────────────────────────────────────────
# 10.3 校验层次
# ─────────────────────────────────────────────────────────────


def test_coordinate_semantics_are_checked_not_just_types():
    """TL-10.3-1：必须实现坐标语义检查，只查类型和单位不够。

    这个反例是整条条款的由来：把世界系的 `force_vector` 标成 `sensor_local`，
    数值合法、单位正确、类型没错，但物理意义完全不同。JSON Schema 查不出
    这类错误，只有把「轴含义与字段语义」对照才能查。

    这里用「二维质心引用三维坐标系」——V3.0 早期草案的真实漏洞。
    """
    frame = base_frame()
    frame["contact_centroid_frame"] = "sensor_0"  # 三维系
    report = run(frame)
    assert not report.ok
    assert any(f.clause.startswith("TL-5.2") for f in report.errors), [
        (f.code, f.clause) for f in report.findings
    ]


# ─────────────────────────────────────────────────────────────
# 10.4 错误报告
# ─────────────────────────────────────────────────────────────


def test_every_finding_cites_a_clause():
    """TL-10.4-1：每条 finding 都必须引用它违反的条款。

    没有条款号的报错，第三方无法回到规范确认「这条到底要求了什么」，
    也无法判断这个实现是不是理解错了条款。条款号是规范与实现之间的
    唯一锚点。
    """
    frames = [
        base_frame(normal_force=float("nan")),
        base_frame(contact="true"),
        base_frame(force_vector=[1.0, 2.0]),
        base_frame(slip_event="yes"),
    ]
    for frame in frames:
        for f in run(frame).findings:
            assert f.clause, f"finding {f.code} 没有条款号"
            assert f.clause.startswith("TL-"), f"条款号格式不对：{f.clause}"


def test_finding_carries_all_required_fields():
    """TL-10.5-2：每条 finding 必须带 code / clause / path / offending value。

    `value` 单独存在，是因为 message 是给人读的解释，value 是给机器比的
    原始值。下游要按值回原始数据里定位这一条，靠 message 做正则匹配不可靠。
    """
    report = run(base_frame(contact="true"))
    assert report.findings
    for f in report.findings:
        d = f.as_dict()
        for key in ("code", "clause", "path", "value", "outcome"):
            assert key in d, f"finding 缺少 {key}"
        assert d["path"], "path 不能为空"


def test_three_outcomes_are_distinguished():
    """TL-10.4-2：error / warning / not_verified 三态必须可区分。

    规范专门列了一张表说明这三态，因为把三态压成两态会让验证结果不可信：
    「证据不足」既不能说数据有错，也不能说数据通过。
    """
    # error
    err = run(base_frame(contact="true"))
    assert err.errors and not err.ok

    # warning：contact=false 却有 centroid
    warn = run(base_frame(contact=False))
    assert warn.warnings, [(f.code, f.outcome) for f in warn.findings]
    assert warn.ok, "warning 不应导致失败"

    # not_verified：引用不存在的坐标系时，法向力投影无法复算
    nv = run(base_frame(force_vector_frame="ghost"))
    assert nv.not_verified, [(f.code, f.outcome) for f in nv.findings]


def test_not_verified_is_neither_error_nor_pass():
    """TL-10.4-3：`not_verified` 不得被报成 error，也不得被当成通过。

    这条最容易被实现成「跳过」——反正验证不了，就当没问题。但「无法验证」
    和「验证通过」是两件事：前者需要补证据，后者可以直接用。混在一起，
    报告就失去了可信度。
    """
    report = run(base_frame(force_vector_frame="ghost"))
    nv = report.not_verified
    assert nv, "应产生 not_verified"
    for f in nv:
        assert f.outcome == "not_verified"
        assert f.outcome != "error"
    # not_verified 不计入 errors
    assert not any(f.outcome == "not_verified" for f in report.errors)


# ─────────────────────────────────────────────────────────────
# 10.5 错误码
# ─────────────────────────────────────────────────────────────


def test_all_reported_codes_are_defined_in_spec():
    """TL-10.5-1：必须使用本节定义的码，不得自创。

    `X_` 前缀留给「规范之外的要求」——一个实现可以加 `X_VENDOR_CHECK`，
    但不能把「法向力不一致」改名叫 `NF_BAD`。码表是跨实现比对的基础。
    """
    cases = [
        base_frame(contact="true"),
        base_frame(normal_force=float("nan")),
        base_frame(force_vector=[1.0, 2.0]),
        base_frame(force_vector_frame="ghost"),
        base_frame(contact_region="palm"),
        base_frame(texture_class="coarse"),
        base_frame(force_magnitude=3.1),
        base_frame(contact=None),
        base_frame(contact=None, slip_velocity=[1.0, 0.0], slip_velocity_frame="surface_0"),
    ]
    for frame in cases:
        for f in run(frame).findings:
            assert f.code in S.ERROR_CODES, f"未定义的码：{f.code}"


def test_every_defined_code_has_a_layer():
    """TL-10.5-1：码表里的每个码都必须有分层（A / B / C / D）。

    分层不是装饰——它对应 10.3 的四个校验层，下游可以据此决定「只跑语法层
    够不够」。一个没有层的码说明它没被认真归类。
    """
    for code, layer in S.ERROR_CODES.items():
        assert layer in ("A", "B", "C", "D"), f"{code} 的层 {layer} 非法"


def test_code_table_has_no_duplicates():
    """TL-10.5-1：码必须唯一。

    `ERROR_CODES` 是 dict，重复键会被静默覆盖——一个码定义两次，后一次
    赢，而前一次的层信息就丢了。这里直接查 schema 里的源码文本。
    """
    import re
    import pathlib

    src = pathlib.Path(S.__file__).read_text(encoding="utf-8")
    codes = re.findall(r'^\s{4}"([A-Z][A-Z0-9_]*)": "[ABCD]",$', src, re.MULTILINE)
    dupes = {c for c in codes if codes.count(c) > 1}
    assert not dupes, f"重复定义的码：{dupes}"
