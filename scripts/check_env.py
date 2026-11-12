#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地环境快照 — 把本地仓库状态写到云盘，让 Agent 能看到。

背景：Agent 运行在云端，看不到本地 E:\\ 盘。但云盘（Coze Drive）是双向同步的。
本脚本在本地收集环境信息，写到云盘的 docs/_env_snapshot.md，
云盘一同步，Agent 就能读到本地到底长什么样。

用法（在仓库根目录跑）：

    python scripts/check_env.py

输出：
    1. 控制台打印摘要
    2. 写入云盘路径（自动探测）：.../牛宿科技/tlabel-repo/docs/_env_snapshot.md
       如果探测不到云盘，写到当前仓库的 docs/_env_snapshot.md

不修改仓库任何代码，只读 + 写一个快照文件。
"""

import hashlib
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

CLOUD_SUBPATH = Path("扣龙虾") / "所有对话" / "主对话" / "牛宿科技" / "tlabel-repo" / "docs"

# 需要记录指纹的关键文件（相对仓库根）
KEY_FILES = [
    "VERSION",
    "pyproject.toml",
    "tlabel/__init__.py",
    "tlabel/adapters/__init__.py",
    "tlabel/adapters/base.py",
    "tlabel/adapters/daimon_dataset.py",
    "tlabel/adapters/xela.py",
    "tlabel/adapters/tashan_ts_f_a.py",
    "tlabel/core/schema.py",
]

EXPECTED_REMOTE_KEYWORD = "liesliy/tlabel"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def run(cmd, cwd=None):
    """跑一条命令，返回 (ok, stdout)。失败不抛异常。"""
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        out = p.stdout.decode("utf-8", errors="replace").strip()
        err = p.stderr.decode("utf-8", errors="replace").strip()
        return p.returncode == 0, (out or err)
    except Exception as e:  # noqa: BLE001
        return False, "ERR: %s" % e


def sha256_16(path):
    """文件 SHA256 前 16 位。文件不存在返回 None。"""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def find_repo_root():
    """从当前目录往上找 .git。"""
    cur = Path.cwd().resolve()
    for p in [cur] + list(cur.parents):
        if (p / ".git").exists():
            return p
    return None


def find_cloud_docs():
    """探测云盘 docs 目录。找不到返回 None。"""
    candidates = []

    # Windows
    home = Path.home()
    candidates.append(home / "Coze" / "Drive" / CLOUD_SUBPATH)
    # 其他盘符的 Coze Drive（少见但兜一下）
    for drive in "CDEFG":
        candidates.append(Path("%s:/Coze/Drive" % drive) / CLOUD_SUBPATH)
    # Linux / 云端
    candidates.append(Path("/Coze/Drive") / CLOUD_SUBPATH)

    for c in candidates:
        try:
            if c.is_dir():
                return c
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------

def collect(repo):
    info = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "os": "%s %s" % (platform.system(), platform.release()),
        "python": sys.version.split()[0],
        "repo_path": str(repo),
    }

    # --- git ---
    ok, remote = run(["git", "remote", "get-url", "origin"], cwd=repo)
    info["remote"] = remote if ok else "(未配置 origin)"

    ok, branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    info["branch"] = branch if ok else "(未知)"

    ok, commit = run(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
    info["commit"] = commit if ok else "(未知)"

    ok, msg = run(["git", "log", "-1", "--pretty=%s"], cwd=repo)
    info["commit_msg"] = msg if ok else ""

    ok, when = run(["git", "log", "-1", "--pretty=%cd", "--date=format:%Y-%m-%d %H:%M"], cwd=repo)
    info["commit_time"] = when if ok else ""

    # 领先/落后 origin
    ok, ab = run(["git", "rev-list", "--left-right", "--count", "HEAD...@{u}"], cwd=repo)
    if ok and ab:
        parts = ab.split()
        if len(parts) == 2:
            info["ahead"] = int(parts[0])
            info["behind"] = int(parts[1])
        else:
            info["ahead"] = info["behind"] = None
    else:
        info["ahead"] = info["behind"] = None

    # 未提交改动
    ok, status = run(["git", "status", "--porcelain"], cwd=repo)
    info["dirty"] = [ln for ln in status.splitlines() if ln.strip()] if ok else []

    # --- VERSION ---
    vf = repo / "VERSION"
    info["version"] = vf.read_text(encoding="utf-8").strip() if vf.exists() else "(无 VERSION 文件)"

    # --- 适配器 ---
    adapters = None
    source = ""
    try:
        sys.path.insert(0, str(repo))
        from tlabel.adapters import AVAILABLE_ADAPTERS  # noqa: PLC0415

        adapters = list(AVAILABLE_ADAPTERS.keys())
        source = "import tlabel"
    except Exception as e:  # noqa: BLE001
        # fallback：解析 __init__.py
        init = repo / "tlabel" / "adapters" / "__init__.py"
        if init.exists():
            txt = init.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"AVAILABLE_ADAPTERS\s*=\s*\{(.*?)\n\}", txt, re.S)
            if m:
                adapters = re.findall(r'"([a-z0-9_]+)"\s*:', m.group(1))
                source = "解析 __init__.py（import 失败：%s）" % type(e).__name__
    info["adapters"] = adapters or []
    info["adapter_source"] = source

    # --- 关键文件指纹 ---
    info["files"] = []
    for rel in KEY_FILES:
        p = repo / rel
        info["files"].append({
            "path": rel,
            "exists": p.exists(),
            "size": p.stat().st_size if p.exists() else None,
            "sha16": sha256_16(p),
        })

    # --- 依赖版本 ---
    deps = {}
    for mod in ("numpy", "pyarrow", "cv2", "scipy", "pandas"):
        try:
            m = __import__(mod)
            deps[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            deps[mod] = "(未安装)"
    info["deps"] = deps

    return info


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render(info, cloud_path):
    L = []
    a = L.append

    a("# 本地环境快照")
    a("")
    a("> 由 `scripts/check_env.py` 自动生成，**请勿手工编辑**。")
    a("> 用途：让云端 Agent 看到本地仓库的真实状态。")
    a("")
    a("| 项 | 值 |")
    a("|---|---|")
    a("| 生成时间 | %s |" % info["generated_at"])
    a("| 主机 | %s |" % info["host"])
    a("| 系统 | %s |" % info["os"])
    a("| Python | %s |" % info["python"])
    a("| 仓库路径 | `%s` |" % info["repo_path"])
    a("| 快照落盘位置 | `%s` |" % (cloud_path or "(未找到云盘，写到了仓库内)"))
    a("")

    # 仓库状态
    a("## 仓库状态")
    a("")
    a("| 项 | 值 |")
    a("|---|---|")
    a("| remote | `%s` |" % info["remote"])
    a("| 分支 | `%s` |" % info["branch"])
    a("| commit | `%s` |" % info["commit"])
    a("| commit 时间 | %s |" % info["commit_time"])
    a("| commit 说明 | %s |" % (info["commit_msg"] or "—"))
    a("| VERSION | **%s** |" % info["version"])

    if info["ahead"] is None:
        a("| 与 origin 比较 | 无法比较（无 upstream 或未 fetch） |")
    else:
        a("| 与 origin 比较 | 领先 **%d** / 落后 **%d** |" % (info["ahead"], info["behind"]))
    a("")

    # remote 校验
    if EXPECTED_REMOTE_KEYWORD not in (info["remote"] or ""):
        a("> ⚠️ **remote 不匹配**：期望包含 `%s`，实际是 `%s`。" % (EXPECTED_REMOTE_KEYWORD, info["remote"]))
        a("> 如果本地仓库的 origin 不是这个，push/pull 通道不通，需要先修正。")
        a("")

    # 未提交改动
    a("## 未提交改动")
    a("")
    if info["dirty"]:
        a("**共 %d 项**（有未提交改动，说明本地可能领先云端看不到的修改）：" % len(info["dirty"]))
        a("")
        a("```")
        for ln in info["dirty"][:50]:
            a(ln)
        if len(info["dirty"]) > 50:
            a("... 还有 %d 项" % (len(info["dirty"]) - 50))
        a("```")
    else:
        a("工作区干净，无未提交改动。")
    a("")

    # 适配器
    a("## 适配器")
    a("")
    a("来源：%s" % (info["adapter_source"] or "—"))
    a("")
    if info["adapters"]:
        a("**共 %d 个**：" % len(info["adapters"]))
        a("")
        a("`%s`" % "`, `".join(info["adapters"]))
    else:
        a("⚠️ 未取到适配器列表。")
    a("")
    a("> 基线核对：v0.24.2 应有 **12 个数据集适配器 + 2 个传感器适配器 + 1 个占位符 = 15 个键**。")
    a("> 如果数量对不上，说明本地代码与云端仓库不一致。")
    a("")

    # 文件指纹
    a("## 关键文件指纹（SHA256 前 16 位）")
    a("")
    a("| 文件 | 存在 | 大小 (B) | SHA256-16 |")
    a("|---|---|---|---|")
    for f in info["files"]:
        a("| `%s` | %s | %s | `%s` |" % (
            f["path"],
            "✓" if f["exists"] else "✗",
            "{:,}".format(f["size"]) if f["size"] is not None else "—",
            f["sha16"] or "—",
        ))
    a("")
    a("> 用途：云端比对同一文件的指纹，若不同则说明本地代码被手改过，或版本不一致。")
    a("")

    # 依赖
    a("## 依赖版本")
    a("")
    a("| 包 | 版本 |")
    a("|---|---|")
    for k, v in info["deps"].items():
        a("| %s | %s |" % (k, v))
    a("")
    a("> SharpaDex 适配需要 `pyarrow` + `opencv-python` + `scipy`，以及系统级 `ffmpeg`。")
    a("")

    # 自检结论
    a("## 自检结论")
    a("")
    checks = []
    checks.append(("remote 指向 liesliy/tlabel", EXPECTED_REMOTE_KEYWORD in (info["remote"] or "")))
    checks.append(("VERSION 为 0.24.x 或更高", info["version"].startswith("0.24") or _ver_ge(info["version"], "0.24")))
    checks.append(("适配器数 = 15 键", len(info["adapters"]) == 15))
    checks.append(("工作区干净", not info["dirty"]))
    for name, ok in checks:
        a("- %s %s" % ("✅" if ok else "❌", name))
    a("")
    a("如果有 ❌，先解决再开工。具体见 `docs/ROADMAP_industrial_datasets.md` 第 7 节。")

    return "\n".join(L) + "\n"


def _ver_ge(ver, base):
    """粗糙版本比较：ver >= base 返回 True。"""
    try:
        def parse(v):
            return tuple(int(x) for x in re.findall(r"\d+", v)[:3])
        return parse(ver) >= parse(base)
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    repo = find_repo_root()
    if not repo:
        print("❌ 当前目录不在 git 仓库内。请在 tlabel 仓库根目录运行。")
        return 1

    print("仓库：%s" % repo)
    info = collect(repo)
    print("采集完成：VERSION=%s commit=%s 适配器=%d" % (
        info["version"], info["commit"], len(info["adapters"])))

    cloud_docs = find_cloud_docs()
    if cloud_docs:
        out = cloud_docs / "_env_snapshot.md"
        print("云盘：%s" % cloud_docs)
    else:
        out = repo / "docs" / "_env_snapshot.md"
        print("⚠️ 未找到云盘目录，快照写到仓库内：%s" % out)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(info, cloud_docs), encoding="utf-8")
    print("✅ 已写入：%s" % out)

    # 摘要
    print("")
    print("摘要：")
    print("  remote      %s" % info["remote"])
    print("  commit      %s (%s)" % (info["commit"], info["branch"]))
    if info["ahead"] is not None:
        print("  领先/落后   %d / %d" % (info["ahead"], info["behind"]))
    print("  未提交      %d 项" % len(info["dirty"]))
    print("  适配器      %d 个" % len(info["adapters"]))
    print("")
    print("把这份快照留给云端 Agent 就行，它会自己读。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
