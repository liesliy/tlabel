#!/usr/bin/env python3
"""流式改造等价性验证脚本（ROADMAP 第 1 步验收用）

用途
----
对比两个代码基线上同一个 daimon episode 的 load() 输出，逐字段比对，
用于证明 v0.25.0 的流式改造没有改变任何输出。

用法
----
    python scripts/verify_stream_equiv.py <repo_path> <data_path> <out_json>

    repo_path  要验证的 tlabel 仓库根目录（决定 import 的是哪份代码）
    data_path  daimon episode 目录或 parquet 文件
    out_json   输出 JSON 路径

典型用法（先在改造前 worktree 跑一遍，再在当前仓库跑一遍，然后 diff）：

    python scripts/verify_stream_equiv.py C:/Users/sisiw/tlabel-baseline <data> base.json
    python scripts/verify_stream_equiv.py <当前仓库>                  <data> curr.json
    python scripts/verify_stream_equiv.py --diff base.json curr.json

说明
----
- 浮点值按 bit 级完全相等比较，不做容差。改造只改数据通路，不改算法，
  任何浮点差异都说明改造引入了副作用。
- 同时校验 iter_frames() 与 load().frames 逐帧一致（仅新版本支持）。
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path


# v0.25.0 有意新增、baseline 不存在的字段路径。对比时跳过，不算差异。
EXPECTED_NEW_PATHS = ("$.streamed", "$.aggregate_stats")


def _jsonable(obj):
    """把 numpy 标量/数组与不可序列化对象转成 JSON 可写形式"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__NaN__"
        if math.isinf(obj):
            return "__Inf__" if obj > 0 else "__-Inf__"
        return obj
    # numpy 标量 / 数组
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return _jsonable(obj.item())
        if isinstance(obj, np.ndarray):
            return _jsonable(obj.tolist())
    except Exception:
        pass
    return repr(obj)


def _frame_dict(frame):
    sv = getattr(frame, "schema_v2", None)
    if sv is None:
        sv_dict = None
    elif hasattr(sv, "to_dict"):
        sv_dict = sv.to_dict()
    elif hasattr(sv, "__dict__"):
        sv_dict = dict(sv.__dict__)
    else:
        sv_dict = repr(sv)
    return {
        "frame_idx": frame.frame_idx,
        "timestamp_s": frame.timestamp_s,
        "manipulation_phase": frame.manipulation_phase,
        "confidence": frame.confidence,
        "sensor_specific": frame.sensor_specific,
        "schema_v2": sv_dict,
    }


def dump_repo(repo: str, data: str, out_json: str) -> int:
    repo = os.path.abspath(repo)
    data = os.path.abspath(data)
    os.chdir(repo)
    sys.path.insert(0, repo)

    import tlabel  # noqa: F401
    from tlabel.adapters.daimon_dataset import DaimonAdapter

    print("[info] repo    =", repo)
    print("[info] tlabel  =", getattr(tlabel, "__file__", "?"))
    print("[info] version =", getattr(tlabel, "__version__", "?"))

    adapter = DaimonAdapter()
    payload = {
        "repo": repo,
        "data": data,
        "tlabel_version": getattr(tlabel, "__version__", None),
        "has_iter_frames": hasattr(adapter, "iter_frames"),
    }

    # --- load() ---
    d = adapter.load(data)
    payload["load"] = {
        "num_frames": len(d.frames),
        "sensor_info": d.sensor_info,
        "episode_info": d.episode_info,
        "capabilities": d.capabilities,
        "sensor_id": d.sensor_id,
        "schema_version": d.schema_version,
        "streamed": getattr(d, "streamed", None),
        "frames": [_frame_dict(f) for f in d.frames],
    }
    print("[info] load frames =", len(d.frames))

    # --- iter_frames()（仅新版本有） ---
    if hasattr(adapter, "iter_frames"):
        meta = {}
        it_frames = [_frame_dict(f) for f in adapter.iter_frames(data, meta_out=meta)]
        payload["iter"] = {
            "num_frames": len(it_frames),
            "meta_keys": sorted(meta.keys()),
            "frames": it_frames,
        }
        print("[info] iter frames =", len(it_frames))
        print("[info] iter meta   =", sorted(meta.keys()))

    # --- stream=True ---
    try:
        ds = adapter.load(data, stream=True)
        payload["load_stream"] = {
            "num_frames": len(ds.frames),
            "aggregate_stats": getattr(ds, "aggregate_stats", None),
            "streamed": getattr(ds, "streamed", None),
        }
        print("[info] stream=True frames =", len(ds.frames))
    except Exception as e:
        payload["load_stream"] = {"error": repr(e)}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=1)
    print("[info] written:", out_json)
    return 0


def _walk_diff(a, b, path="$", out=None, limit=40):
    if out is None:
        out = []
    # v0.25.0 有意新增的字段：baseline 没有，current 有，属于预期差异
    if path in EXPECTED_NEW_PATHS:
        return out
    if len(out) >= limit:
        return out
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        out.append("%s: type %s != %s" % (path, type(a).__name__, type(b).__name__))
        return out
    if isinstance(a, dict):
        ka, kb = set(a.keys()), set(b.keys())
        for k in sorted(ka - kb):
            out.append("%s.%s: 仅 baseline 有" % (path, k))
        for k in sorted(kb - ka):
            out.append("%s.%s: 仅 current 有" % (path, k))
        for k in sorted(ka & kb):
            _walk_diff(a[k], b[k], "%s.%s" % (path, k), out, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append("%s: 长度 %d != %d" % (path, len(a), len(b)))
        for i in range(min(len(a), len(b))):
            _walk_diff(a[i], b[i], "%s[%d]" % (path, i), out, limit)
    else:
        if a != b:
            out.append("%s: %r != %r" % (path, a, b))
    return out


def diff_files(base_json: str, curr_json: str) -> int:
    with open(base_json, encoding="utf-8") as f:
        base = json.load(f)
    with open(curr_json, encoding="utf-8") as f:
        curr = json.load(f)

    print("=== 对比 load() 输出 ===")
    diffs = _walk_diff(base.get("load"), curr.get("load"))
    if diffs:
        print("❌ load() 存在 %d 处差异（最多显示 40 处）：" % len(diffs))
        for d in diffs:
            print("   ", d)
        rc = 1
    else:
        n = base.get("load", {}).get("num_frames")
        print("✅ load() 逐字段完全一致（%s 帧）" % n)
        rc = 0

    # iter_frames 与 load 是否一致（同一份 current）
    if "iter" in curr:
        print("\n=== 对比 iter_frames() vs load()（同一代码基线）===")
        diffs2 = _walk_diff(curr.get("load", {}).get("frames"),
                            curr.get("iter", {}).get("frames"))
        if diffs2:
            print("❌ iter_frames 与 load 存在 %d 处差异（最多显示 40 处）：" % len(diffs2))
            for d in diffs2:
                print("   ", d)
            rc = 1
        else:
            print("✅ iter_frames() 与 load().frames 逐帧一致（%d 帧）"
                  % curr.get("iter", {}).get("num_frames"))
    else:
        print("\n[skip] baseline 无 iter_frames()，跳过该项")

    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", nargs=2, metavar=("BASE_JSON", "CURR_JSON"))
    ap.add_argument("repo", nargs="?")
    ap.add_argument("data", nargs="?")
    ap.add_argument("out_json", nargs="?")
    args = ap.parse_args()

    if args.diff:
        sys.exit(diff_files(args.diff[0], args.diff[1]))

    if not (args.repo and args.data and args.out_json):
        ap.print_help()
        sys.exit(2)
    sys.exit(dump_repo(args.repo, args.data, args.out_json))


if __name__ == "__main__":
    main()
