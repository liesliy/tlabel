#!/usr/bin/env python3
"""测量 daimon 适配器的峰值内存（ROADMAP 第 1 步验收用）

用途
----
在同一份合成 episode 上分别跑改造前/改造后的代码，记录 tracemalloc 峰值，
证明流式改造后峰值内存不再随帧数增长。

用法
----
    python scripts/bench_stream_memory.py <repo> <data> --mode load --out r.json

    repo   要测试的 tlabel 仓库（决定 import 哪份代码）
    data   episode 目录
    mode   load   完整加载（改造前后都有）
           iter   逐帧迭代（仅 v0.25.0+ 有 iter_frames）
           stream load(stream=True)，只留统计量（仅 v0.25.0+）

注意
----
tracemalloc 只统计 Python 层分配，ffmpeg 子进程自身内存不计入。
这正好对应我们要证明的事：改造前整段视频被读成 Python list（可测），
改造后只有当前帧在 Python 侧存在。
"""

import argparse
import json
import os
import sys
import time
import tracemalloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("data")
    ap.add_argument("--mode", choices=["load", "iter", "stream"], default="load")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    data = os.path.abspath(args.data)
    os.chdir(repo)
    sys.path.insert(0, repo)

    import tlabel  # noqa: F401
    from tlabel.adapters.daimon_dataset import DaimonAdapter

    adapter = DaimonAdapter()
    result = {
        "repo": repo,
        "data": data,
        "mode": args.mode,
        "tlabel_version": getattr(tlabel, "__version__", None),
    }

    keep = None
    frames = 0
    tracemalloc.start()
    t0 = time.time()
    try:
        if args.mode == "load":
            data_obj = adapter.load(data)
            keep = data_obj
            frames = len(data_obj.frames)
        elif args.mode == "iter":
            for _f in adapter.iter_frames(data):
                frames += 1
        else:
            data_obj = adapter.load(data, stream=True)
            keep = data_obj
            frames = int(data_obj.aggregate_stats.get("num_frames", 0))
        err = None
    except Exception as e:
        err = repr(e)
    elapsed = time.time() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result.update({
        "frames": frames,
        "peak_mb": round(peak / 1e6, 1),
        "current_mb": round(cur / 1e6, 1),
        "elapsed_s": round(elapsed, 1),
        "error": err,
        "kept_alive": keep is not None,
    })
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
