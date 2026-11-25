"""跨实现测试：只依赖规范正文 + 测试向量，不读参考实现的源码。

这是六类里唯一能回答「第三方能不能独立实现」的一类。它和 negative/positive
的区别不在数据，在**依赖方向**：

- `negative/` 的测试 import 了 `conftest.base_frame()`，基底是 Python 构造的。
  一个外部实现者拿不到这个基底，他的实现和我们的测试是两套东西。
- 这里只用 `tlabel-v3-spec/conformance/cases.yaml`（纯数据，语言无关）和
  `validate()` 这一个公开入口。向量里的每一条都是**完整自足的数据集**，
  复制出去就能用，不需要任何 Python 上下文。

向量文件里写死了「期望的错误码 + 期望的条款号」。这个写死是刻意的：它让
**实现改了条款归属时向量立刻失败**。真实发生过一次——校验器把迁移态报成
7.2-1（生产者义务）而不是 7.2-3（校验器义务），向量里那条立刻红了。

如果第三方实现能跑通同一批向量，就说明规范里这部分足够明确，不依赖某个
实现的具体写法。
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest
import yaml

from tlabel.v3 import validate

#: 向量文件放在规范仓库里，不在本仓库。它是**规范的一部分**——换实现不改向量。
#:
#: 用向上查找而不是写死 `parents[N]`：层级随仓库布局会变（测试文件从
#: `tests/` 挪进 `tests/conformance/cross_impl/` 就变过一次），写死的结果
#: 是「文件明明存在却报 FileNotFoundError」，比路径本身更费时间。
def _find_vectors() -> pathlib.Path:
    rel = pathlib.Path("tlabel-v3-spec") / "conformance" / "cases.yaml"
    for base in pathlib.Path(__file__).resolve().parents:
        candidate = base / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"找不到规范向量文件 {rel}；已从 {pathlib.Path(__file__).parent} 向上逐级查找"
    )


VECTORS = _find_vectors()


def _load_cases():
    """把分组结构摊平成 (组名, 核心条款, 用例) 三元组。"""
    doc = yaml.safe_load(VECTORS.read_text(encoding="utf-8"))
    out = []
    for key, group in doc.items():
        if not key.startswith("group_"):
            continue
        for case in group["cases"]:
            out.append(
                pytest.param(
                    case,
                    id=case["id"],
                    marks=pytest.mark.skipif(
                        False, reason=""
                    ),
                )
            )
    return out


CASES = _load_cases()


def test_vector_file_is_present_and_parseable():
    """向量文件必须存在且能被解析。

    单独一条，是为了让「文件丢了」和「某条用例失败」在报告里长得不一样。
    合并进参数化的话，文件缺失会表现为 20 条用例全红，看不出根因。
    """
    assert VECTORS.is_file(), f"向量文件不存在：{VECTORS}"
    doc = yaml.safe_load(VECTORS.read_text(encoding="utf-8"))
    assert doc.get("spec_version"), "向量文件必须声明它对应的规范版本"


def test_vector_file_declares_assembly_rule():
    """向量文件必须写明 `dataset` 与 `frame` 怎么合并。

    两个块分开存是为了避免同一条条款的用例重复大半个 dataset。但分开存
    就产生了一个必须回答的问题：「frame 放哪儿」。文件里不写，第三方实现
    者只能猜——猜错的话所有反例都会「通过」，因为校验器根本看不到帧。
    """
    doc = yaml.safe_load(VECTORS.read_text(encoding="utf-8"))
    assert "assembly" in doc, "向量文件缺 assembly 块：合并规则没写"
    assert doc["assembly"].get("rule"), "assembly 块缺 rule 说明"


def test_vectors_are_self_contained():
    """每条向量的输入必须自足，不依赖任何外部基底。

    这是跨实现的前提。如果向量只写了「把 base_frame 的 normal_force 改成
    ‖F‖」，第三方实现者还得先知道 base_frame 长什么样——那向量就没用了。
    """
    doc = yaml.safe_load(VECTORS.read_text(encoding="utf-8"))
    for key, group in doc.items():
        if not key.startswith("group_"):
            continue
        for case in group["cases"]:
            inp = case["input"]
            ds = inp.get("dataset")
            frame = inp.get("frame")
            frames = inp.get("frames")
            assert ds, f"{case['id']} 缺 input.dataset"
            assert frame or frames, f"{case['id']} 缺 input.frame / input.frames"
            for required in ("session_id", "time_base", "frames_of_reference", "capabilities"):
                assert required in ds, (
                    f"{case['id']} 的数据集缺 {required}——向量不自足，"
                    "第三方实现者无法只凭这份文件复现"
                )
            # 帧本身也要能独立成立：时间戳和 provenance 是每帧必有的
            for f in (frames or [frame]):
                for required in ("timestamp_s", "field_provenance"):
                    assert required in f, f"{case['id']} 的帧缺 {required}"


@pytest.mark.parametrize("case", CASES)
def test_vector_matches_reference_implementation(case):
    """TL-10.2 / 10.4：参考实现对每条向量的结论必须与向量声明一致。

    比对三项，缺一不可：
    - `outcome`：通过 / 报错
    - `code`：错误码（机器接口）
    - `clause`：条款号（追责依据）

    只比 outcome 会让「报了错但报错了原因」蒙混过关——那种情况下数据确实
    被拒了，但拒绝的理由不对，下游按条款号去查会查到不相干的要求。
    """
    inp = case["input"]
    ds = copy.deepcopy(inp["dataset"])
    # 按 cases.yaml 的 assembly 规则组装：input.frames 优先（多帧场景），
    # 否则把单个 input.frame 作为唯一元素。两种写法都在 assembly 里写明。
    if "frames" in inp:
        ds["frames"] = copy.deepcopy(inp["frames"])
    else:
        ds["frames"] = [copy.deepcopy(inp["frame"])]
    level = inp.get("level", "core")
    report = validate(ds, level=level)
    expect = case["expect"]

    if expect["outcome"] == "pass":
        assert report.ok, (
            f"{case['id']}（{case['description']}）期望通过，"
            f"实际错误：{[(f.code, f.clause) for f in report.errors]}"
        )
        return

    hits = [f for f in report.findings if f.code == expect["code"]]
    assert hits, (
        f"{case['id']}（{case['description']}）期望 {expect['code']}，"
        f"实际：{[(f.code, f.clause) for f in report.findings]}"
    )
    assert any(f.clause == expect["clause"] for f in hits), (
        f"{case['id']} 的 {expect['code']} 应归到 {expect['clause']}，"
        f"实际归到 {sorted({f.clause for f in hits})}。"
        "条款归属错了，按条款号回溯的人会查到不相干的要求。"
    )


def test_vectors_cover_both_directions():
    """每一条核心条款的向量组都必须有正例和反例。

    只有反例的组，无法区分「校验器查得对」和「校验器见谁都拒」。
    只有正例的组，无法区分「校验器放行」和「校验器根本没查」。
    """
    doc = yaml.safe_load(VECTORS.read_text(encoding="utf-8"))
    for key, group in doc.items():
        if not key.startswith("group_"):
            continue
        kinds = {c["kind"] for c in group["cases"]}
        assert "positive" in kinds and "negative" in kinds, (
            f"{key}（{group['name']}）缺{'正例' if 'positive' not in kinds else '反例'}"
        )
