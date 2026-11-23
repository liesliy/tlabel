#!/usr/bin/env python3
"""发版前包内容审查：确保内部管理文件不进 sdist / wheel。

背景
----
TLabel 仓库里有若干「只给自己看」的文档（internal-docs/、docs/internal/、
发版规范、RFC 草稿、设计稿等）。这些文件一旦进了 sdist 或 wheel，就会随
`pip download` / PyPI 页面被外部看到。靠人记着一定会忘，所以用脚本拦。

用法
----
    python scripts/check_package_contents.py                # 构建后检查
    python scripts/check_package_contents.py --skip-build   # 复用 ./dist
    python scripts/check_package_contents.py --keep         # 保留构建目录

退出码
------
    0  干净
    1  命中黑名单（禁止发版）
    2  构建失败

接入
----
发版流程第 ⑤ 步固定调用（见 `牛宿科技/TLabel发版规范与平台规则_归档_20261123.md` §4）。
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 黑名单：路径或文件名命中即失败
# ---------------------------------------------------------------------------
BLOCKED_PATH_PARTS = (
    "internal-docs/",
    "docs/internal/",
    "tlabel-v050-src/",
    "__pycache__/",
    ".git/",
)

BLOCKED_NAME_PATTERNS = (
    "*发版*",
    "*.发版.*",
    "RFC-*.md",
    "RFC*.md",
    "deviation-*.md",
    "*_design.md",
    "*-design.md",
    "DESIGN-INTERNAL.md",
    "PRODUCT_DEFINITION.md",
    "*.bak",
    "*.tmp",
    "*.swp",
    "*.orig",
    "*.rej",
)

# 根目录允许出现的 md / 文本文件（其余仅告警，不失败）
ALLOWED_ROOT_FILES = {
    "README.md",
    "README_CN.md",
    "CHANGELOG.md",
    "MIGRATION.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "LICENSE",
    "LICENSE.txt",
    "MANIFEST.in",
    "VERSION",
    "CITATION.cff",
    "pyproject.toml",
    "PKG-INFO",
    "setup.cfg",
    "setup.py",
}

# 根目录允许出现的 md 后缀之外，也不该出现这些（直接失败）
ROOT_LEAK_SUFFIXES = (".md",)


def _norm(name: str) -> str:
    return name.replace("\\", "/")


def _strip_top_dir(names: list[str]) -> list[str]:
    """sdist 内所有路径都带一个顶层目录 `tlabel-x.y.z/`，去掉它便于比对。"""
    if not names:
        return names
    first = _norm(names[0])
    if "/" in first:
        top = first.split("/", 1)[0]
        if all(_norm(n).startswith(top + "/") or _norm(n) == top for n in names):
            return [_norm(n)[len(top) + 1:] for n in names if _norm(n) != top]
    return [_norm(n) for n in names]


def _is_blocked(rel_path: str) -> str | None:
    """命中黑名单则返回原因，否则 None。"""
    p = _norm(rel_path)
    for part in BLOCKED_PATH_PARTS:
        if part in p:
            return f"path contains `{part}`"
    base = p.rsplit("/", 1)[-1]
    for pat in BLOCKED_NAME_PATTERNS:
        if fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(p, pat):
            return f"name matches `{pat}`"
    return None


def _root_warnings(files: list[str]) -> list[str]:
    """根目录出现非白名单的 .md 之类，仅告警。"""
    warns = []
    for f in files:
        p = _norm(f)
        if "/" in p:
            continue
        if p.lower().endswith(ROOT_LEAK_SUFFIXES) and p not in ALLOWED_ROOT_FILES:
            warns.append(p)
    return warns


def build_dist(outdir: Path, isolated: bool = False, with_wheel: bool = False) -> None:
    kinds = "sdist + wheel" if with_wheel else "sdist"
    print(f"[build] 构建 {kinds} → {outdir}")
    cmd = [sys.executable, "-m", "build", "--outdir", str(outdir), "--sdist"]
    if with_wheel:
        cmd.append("--wheel")
    if not isolated:
        # 默认不走隔离环境：隔离模式会临时建 venv 装 setuptools/wheel，网络慢时能卡几分钟
        cmd.append("--no-isolation")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        print("[build] ❌ 构建失败", file=sys.stderr)
        sys.exit(2)


def collect_sdist(tarball: Path) -> list[str]:
    with tarfile.open(tarball, "r:*") as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
    return _strip_top_dir(names)


def collect_wheel(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return [_norm(n) for n in zf.namelist() if not n.endswith("/")]


def main() -> int:
    ap = argparse.ArgumentParser(description="发版前包内容审查")
    ap.add_argument("--skip-build", action="store_true", help="不重新构建，直接检查 ./dist")
    ap.add_argument("--keep", action="store_true", help="保留临时构建目录")
    ap.add_argument("--isolated", action="store_true", help="用隔离环境构建（慢，但更接近 PyPI 构建）")
    ap.add_argument("--with-wheel", action="store_true", help="同时构建 wheel（本机很慢，约 4 分钟以上；默认只查 sdist）")
    args = ap.parse_args()

    tmpdir: Path | None = None
    if args.skip_build:
        dist = REPO_ROOT / "dist"
        if not dist.is_dir():
            print(f"[err] {dist} 不存在，先跑 python -m build", file=sys.stderr)
            return 2
        workdir = dist
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="tlabel-pkgcheck-"))
        build_dist(tmpdir, isolated=args.isolated, with_wheel=args.with_wheel)
        workdir = tmpdir

    try:
        sdists = sorted(workdir.glob("*.tar.gz"))
        wheels = sorted(workdir.glob("*.whl"))
        if not sdists:
            print(f"[err] {workdir} 里没找到 sdist", file=sys.stderr)
            return 2
        if not wheels:
            print("\n[i] 未检查 wheel（wheel 只含 tlabel 包，由 pyproject 的 packages.find 控制，"
                  "历史上一向干净；需要完整检查时加 --with-wheel）")

        failures: list[tuple[str, str, str]] = []  # (package, file, reason)
        warnings: list[tuple[str, str]] = []

        targets = [(sdists[-1].name, collect_sdist(sdists[-1]))]
        if wheels:
            targets.append((wheels[-1].name, collect_wheel(wheels[-1])))
        for label, files in targets:
            print(f"\n=== {label} · {len(files)} files ===")
            for f in sorted(files):
                reason = _is_blocked(f)
                if reason:
                    failures.append((label, f, reason))
                    print(f"  ❌ {f}   ({reason})")
            for w in _root_warnings(files):
                warnings.append((label, w))
                print(f"  ⚠️  {w}   (根目录非白名单文件)")

        print("\n" + "=" * 60)
        if warnings:
            print(f"⚠️  {len(warnings)} 条告警（不阻断发版，但建议清理）：")
            for label, f in warnings:
                print(f"   - {label}: {f}")

        if failures:
            print(f"\n❌ 命中黑名单 {len(failures)} 项，禁止发版：")
            for label, f, reason in failures:
                print(f"   - {label}: {f}  → {reason}")
            print("\n修复方向：查 .gitignore / MANIFEST.in，必要时 git rm --cached。")
            return 1

        print("✅ 包内容干净，可以发版。")
        return 0
    finally:
        if tmpdir is not None and not args.keep:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
