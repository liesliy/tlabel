"""往返测试：数据写出再读回，语义不得改变。

TLabel 的规范性编码是 JSON（TL-10.1-1），而 JSON 往返是**有损**的——只是
损失发生在少数几个地方，且都不报错：

- `false` / `0` / `""` / `[]` / `{}` 在 Python 里都是假值，一个 `if not v`
  就足以把「明确是 false」读成「没有值」。三态布尔里这是致命的：`contact=false`
  和 `contact=null` 是两件完全不同的事（有接触判定但没接触 vs 没有判定能力）。
- 浮点数的十进制表示不是双射。`0.1 + 0.2` 写出去是 `0.30000000000000004`，
  读回来还是它，但如果某个实现先 `round(x, 3)` 再写，语义就变了。
- `NaN` / `Infinity` 不是合法 JSON，`json.dumps` 默认会写成 `NaN` 这种非法
  字面量，严格的解析器读不回来。

所以这里测的不是「json 库对不对」，而是**校验结论在往返前后是否一致**。
一份数据在往返后从合法变成非法（或反过来），说明某个字段的语义在序列化
边界上丢了。
"""

from __future__ import annotations

import copy
import json

import pytest

from conftest import base_dataset, base_frame, run


def _roundtrip(obj):
    """JSON 往返。`ensure_ascii=False` 是为了让非 ASCII 的 method_id 保持原样——
    用 \\uXXXX 转义也能读回，但那掩盖了「实现是否按字节比较标识符」这件事。
    """
    return json.loads(json.dumps(obj, ensure_ascii=False))


def _signature(report):
    """把报告压成可比较的签名：码 + 条款 + 路径，不含 message。

    message 是给人读的，措辞可以改；code/clause/path 是机器接口，必须稳定。
    """
    return sorted((f.code, f.clause, f.path, f.outcome) for f in report.findings)


# ── 合法数据往返后仍合法 ──────────────────────────────────


def test_legal_frame_survives_roundtrip():
    """TL-10.1-1：合法帧往返后必须仍通过，且不产生新 finding。"""
    ds = base_dataset()
    fr = base_frame()
    before = run(fr, ds)
    after = run(_roundtrip(fr), _roundtrip(ds))
    assert before.ok and after.ok
    assert _signature(before) == _signature(after)


# ── 假值不得被读成缺失 ────────────────────────────────────


@pytest.mark.parametrize(
    "field, value",
    [
        ("contact", False),          # 三态布尔：false ≠ null
        ("normal_force", 0.0),       # 零力是合法读数，不是「没有」
        ("force_vector", [0.0, 0.0, 0.0]),
        ("temperature", 0.0),
    ],
)
def test_falsy_values_survive_roundtrip(field, value):
    """TL-10.1-1：`false` / `0` / `0.0` 往返后不得变成 `null`。

    这是最经典的一类序列化 bug：用 `if not value` 判断缺失。`0.0` 的法向力
    （接触但没用力）和 `null` 的法向力（没测到）在物理上是两件事，混淆之后
    下游会把「轻触」当成「无数据」丢掉。
    """
    ds = base_dataset()
    fr = base_frame()
    fr[field] = value
    if field == "temperature":
        ds["capabilities"]["temperature"] = True
        fr["field_provenance"]["temperature"] = {
            "type": "direct_measurement",
            "sensor_or_channel_id": "sensor_0.temp",
            "measurement_target": "sensor_body",
        }
    if field == "contact":
        # contact=false 时接触位置没有物理含义，一并清掉（TL-5.2-4）
        fr["contact_centroid"] = None
        fr["contact_centroid_frame"] = None
        fr["field_provenance"].pop("contact_centroid", None)

    rt = _roundtrip(fr)
    assert rt[field] == value, f"{field} 往返后变成 {rt[field]!r}，期望 {value!r}"
    assert rt[field] is not None
    # 校验结论也必须一致
    assert _signature(run(fr, ds)) == _signature(run(rt, _roundtrip(ds)))


def test_empty_containers_are_distinguishable_from_missing():
    """TL-10.1-1：`{}` 和「键不存在」是两件事，往返后不得互换。

    空 `sensor_specific` 的意思是「有扩展容器但没有内容」，缺键的意思是
    「这个生产者不支持扩展」。前者可以后续追加，后者不能。混淆会让下游
    误判数据来源的能力。
    """
    ds = base_dataset()

    with_empty = base_frame()
    with_empty["sensor_specific"] = {}
    without = base_frame()
    assert "sensor_specific" not in without

    rt_empty = _roundtrip(with_empty)
    rt_without = _roundtrip(without)
    assert rt_empty["sensor_specific"] == {}
    assert "sensor_specific" not in rt_without


# ── 数值精度 ──────────────────────────────────────────────


def test_float_precision_is_preserved():
    """TL-10.1-1：浮点往返必须保精度，否则容差判定会在边界上翻转。

    `normal_force` 与 `|F·n|` 的容差是 1e-2。如果序列化把 3.0999999 写成
    3.1，两份数据一份过一份不过，而这个差异在数值上看不出来。
    """
    ds = base_dataset()
    fr = base_frame()
    fr["normal_force"] = 3.0999999
    rt = _roundtrip(fr)
    assert rt["normal_force"] == 3.0999999
    assert _signature(run(fr, ds)) == _signature(run(rt, _roundtrip(ds)))


def test_scientific_notation_roundtrips():
    """TL-10.1-1：科学计数法写出的值读回来仍是同一个数。

    小力（1e-6 N）和大力（1e6 N）都真实存在。序列化如果按定点写，小的会被
    写成 0.000001 没问题，但某些实现会截断成 0。
    """
    ds = base_dataset()
    for v in (1e-6, 1e6, 1.5e-9):
        fr = base_frame()
        fr["normal_force"] = v
        rt = _roundtrip(fr)
        assert rt["normal_force"] == v, f"{v} 往返后变成 {rt['normal_force']}"


# ── 非 ASCII 标识符 ───────────────────────────────────────


def test_non_ascii_method_id_survives_roundtrip():
    """TL-10.1-1：`method_id` 可以是任意字符串，往返后必须逐字符相同。

    中文标识符在小团队里很常见（`method_id: "压力加权质心_v1"`）。如果某个
    实现按字节比较标识符，而序列化改用了转义形式，可追溯性就断了——同一个
    方法会变成两个不同的 id。
    """
    ds = base_dataset()
    fr = base_frame()
    fr["field_provenance"]["normal_force"]["method_id"] = "压力加权质心_v1"
    rt = _roundtrip(fr)
    assert rt["field_provenance"]["normal_force"]["method_id"] == "压力加权质心_v1"


# ── 非法数据往返后仍非法，且理由相同 ──────────────────────


def test_invalid_frame_keeps_same_findings_after_roundtrip():
    """TL-10.1-1：非法数据往返后必须报**同一批** finding。

    只测「还报错」不够——往返可能让一条错误消失、另一条冒出来，总数一样。
    所以比的是完整签名。
    """
    ds = base_dataset()
    fr = base_frame()
    fr["normal_force"] = 3.1309  # = ‖F‖，把模长当法向
    before = run(fr, ds)
    after = run(_roundtrip(fr), _roundtrip(ds))
    assert not before.ok and not after.ok
    assert _signature(before) == _signature(after)


def test_missing_key_and_null_stay_distinct_after_roundtrip():
    """TL-4.4-2 / TL-10.1-1：缺键与 null 往返后仍是两条不同的路径。

    缺键是格式错误（4.4-1），null 是合法的「无值」（7.3-2）。如果序列化把
    缺键补成 null（某些 schema 驱动的实现会这么做），格式错误就被静默修好了，
    生产者永远不知道自己的输出少了字段。
    """
    ds = base_dataset()
    missing = base_frame()
    del missing["force_vector"]
    nulled = base_frame()
    nulled["force_vector"] = None

    rt_missing = _roundtrip(missing)
    rt_nulled = _roundtrip(nulled)
    assert "force_vector" not in rt_missing
    assert "force_vector" in rt_nulled and rt_nulled["force_vector"] is None
    assert _signature(run(missing, ds)) == _signature(run(rt_missing, _roundtrip(ds)))
    assert _signature(run(nulled, ds)) == _signature(run(rt_nulled, _roundtrip(ds)))


# ── 整体数据集往返 ────────────────────────────────────────


def test_whole_dataset_roundtrip_keeps_cross_frame_state():
    """TL-10.1-1：往返后跨帧状态（时间单调、time_base 一致性）不得丢失。

    跨帧检查依赖帧的顺序和逐帧累积的状态。序列化如果重排了 `frames`（例如
    把 dict 键排序时顺手排了数组），时间轴的语义就变了。
    """
    ds = base_dataset()
    ds["instances"] = [{"instance_id": "a"}]
    frames = []
    for i in range(5):
        fr = base_frame()
        fr["instance_id"] = "a"
        fr["timestamp_s"] = i * 0.033
        frames.append(fr)
    ds["frames"] = frames

    rt = _roundtrip(ds)
    assert [f["timestamp_s"] for f in rt["frames"]] == [f["timestamp_s"] for f in frames]
    assert _signature(run(None, ds)) == _signature(run(None, rt))
