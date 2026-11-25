#!/usr/bin/env python3
"""符合性覆盖率报告 —— 阶段 5 的验收工具。

路线文档 §468 的验收要求是「每条可测试要求有覆盖记录」，**不能只报告单元
测试总通过数量**。一个数字（「867 passed」）不告诉你哪条规范条款没人验过，
而漏掉的那条恰恰可能是最危险的——比如 RFC-004 §7 的 T1 反例：用合力模长
冒充法向力，代码能跑、数值合理、测试全绿，但它就是错的。

本脚本做四件事：

1. **双向核对**：规范正文里的每条 `[TL-x.y-n]` 必须出现在注册表里；注册表里
   的每条也必须能在规范里找到。单向核对会漏掉两类错误——规范新增了条款但
   没人登记，或注册表登记了一条已经不存在的条款。
2. **测试节点存在性**：注册表里 `tests` 引用的节点必须真的能被 pytest 收集到。
   写错路径的引用和没写引用一样，都是假的覆盖。
3. **按章统计**：逐章给出 full / partial / none 与覆盖率，让「哪一章还没做」
   一眼可见，而不是藏在一个总分里。
4. **退出码**：有未覆盖的可测试条款就返回 1，可以直接挂进 CI。

用法::

    python tests/conformance/coverage.py            # 完整报告
    python tests/conformance/coverage.py --quiet    # 只输出未覆盖清单
    python tests/conformance/coverage.py --chapter 9  # 只看第 9 章
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SPEC = REPO.parent / "tlabel-v3-spec" / "tlabel-spec-v3.md"
REGISTRY = HERE / "requirements.yaml"

#: 规范正文里的条款标注，形如 `[TL-5.3-2]`。表格里的引用不带方括号，不算定义。
CLAUSE_DEF = re.compile(r"\[(TL-\d+\.\d+-\d+)\]")


def load_spec_clauses() -> Dict[str, int]:
    """从规范正文提取条款号 → 所在行号。"""
    text = SPEC.read_text(encoding="utf-8")
    found: Dict[str, int] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        for clause in CLAUSE_DEF.findall(line):
            found.setdefault(clause, lineno)
    return found


def load_registry() -> List[dict]:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def collect_pytest_nodes() -> Set[str]:
    """收集 pytest 能看到的全部测试节点。

    用 `--collect-only -q` 而不是自己扫文件名：只有 pytest 真正收集到的节点
    才算数。一个 import 就报错的测试文件在磁盘上存在，但它不验证任何东西。
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/conformance", "--collect-only", "-q",
         "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    )
    nodes: Set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        # 形如 `tests/conformance/negative/test_x.py::test_y[param]`
        if "::" not in line or line.startswith("<"):
            continue
        base = line.split("[", 1)[0]
        nodes.add(base)
        nodes.add(line)
    return nodes


def main() -> int:
    ap = argparse.ArgumentParser(description="符合性覆盖率报告")
    ap.add_argument("--quiet", action="store_true", help="只输出未覆盖清单")
    ap.add_argument("--chapter", type=int, default=None, help="只看某一章")
    args = ap.parse_args()

    spec = load_spec_clauses()
    reg = load_registry()
    nodes = collect_pytest_nodes()

    reg_ids = [r["id"] for r in reg]
    reg_set = set(reg_ids)

    problems: List[str] = []

    # ── 双向核对 ──────────────────────────────────────────────
    for clause in sorted(spec):
        if clause not in reg_set:
            problems.append(f"规范有、注册表无：{clause}（规范第 {spec[clause]} 行）")
    for clause in reg_ids:
        if clause not in spec:
            problems.append(f"注册表有、规范无：{clause}")

    dupes = {c for c in reg_ids if reg_ids.count(c) > 1}
    for c in sorted(dupes):
        problems.append(f"注册表重复：{c}")

    # ── 测试节点存在性 ────────────────────────────────────────
    for item in reg:
        for node in item.get("tests") or []:
            if node not in nodes:
                problems.append(f"{item['id']} 引用了收集不到的测试节点：{node}")

    # ── 按章统计 ──────────────────────────────────────────────
    by_chapter: Dict[int, List[dict]] = defaultdict(list)
    for item in reg:
        by_chapter[item["chapter"]].append(item)

    total_need = total_cov = 0
    uncovered: List[dict] = []

    if not args.quiet:
        print(f"规范条款 {len(spec)} 条 / 注册表 {len(reg)} 条 / 已收集测试节点 {len(nodes)} 个")
        print()
        print(f"{'章':<4}{'full':>6}{'partial':>9}{'none':>6}{'需覆盖':>8}{'已覆盖':>8}{'覆盖率':>9}")
        print("-" * 50)

    for ch in sorted(by_chapter):
        if args.chapter is not None and ch != args.chapter:
            continue
        items = by_chapter[ch]
        full = sum(1 for x in items if x["testable"] == "full")
        part = sum(1 for x in items if x["testable"] == "partial")
        none = sum(1 for x in items if x["testable"] == "none")
        need = [x for x in items if x["testable"] in ("full", "partial")]
        cov = [x for x in need if x.get("tests")]
        total_need += len(need)
        total_cov += len(cov)
        uncovered.extend(x for x in need if not x.get("tests"))
        if not args.quiet:
            pct = f"{len(cov) / len(need) * 100:.0f}%" if need else "—"
            name = items[0].get("chapter_name", "")
            print(f"{ch:<4}{full:>6}{part:>9}{none:>6}{len(need):>8}{len(cov):>8}{pct:>9}  {name}")

    if not args.quiet:
        print("-" * 50)
        pct = f"{total_cov / total_need * 100:.1f}%" if total_need else "—"
        print(f"{'合计':<4}{'':>6}{'':>9}{'':>6}{total_need:>8}{total_cov:>8}{pct:>9}")
        print()

    if uncovered:
        print(f"未覆盖的可测试条款 {len(uncovered)} 条：")
        for x in uncovered:
            print(f"  {x['id']:<12} {x['testable']:<8} {x.get('note', '')[:70]}")
        print()

    if problems:
        print(f"一致性问题 {len(problems)} 处：")
        for p in problems:
            print(f"  {p}")
        print()

    return 1 if (uncovered or problems) else 0


if __name__ == "__main__":
    sys.exit(main())
