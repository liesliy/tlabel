#!/usr/bin/env python3
"""生成内存压力测试用的合成 daimon 数据集（ROADMAP 第 1 步验收用）

做法
----
从真实 daimon 数据集里截取前 N 行，改写成单 episode，再配一段长触觉视频，
得到一个"N 帧 parquet + 长视频"的 episode，用于测量流式改造前后的峰值内存。

为什么需要它
-----------
真实数据集里 parquet 有 36 万帧，但单 episode 只有几百帧、触觉视频只有 3 帧，
测不出全量加载的内存问题。工业级场景是"单 episode 几万帧 + 视频同样长"，
必须合成才能复现。

用法
----
    python scripts/make_memtest_dataset.py <src_dataset> <dst_dataset> \
        --frames 30000 --video <long.mov> [--fps 30]
"""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="真实 daimon 数据集目录（含 data/ 与 meta/）")
    ap.add_argument("dst", help="输出目录")
    ap.add_argument("--frames", type=int, default=30000)
    ap.add_argument("--video", required=True, help="长触觉视频（FFV1 gbrp16le）")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    src = Path(args.src)
    dst = Path(args.dst)
    n = args.frames

    src_pq = src / "data" / "chunk-000" / "file-000.parquet"
    if not src_pq.exists():
        print("找不到源 parquet:", src_pq)
        return 2

    print("[1/3] 读取源 parquet ...")
    t = pq.read_table(str(src_pq))
    if t.num_rows < n:
        print("源数据只有 %d 行，少于要求的 %d 行" % (t.num_rows, n))
        return 2
    t = t.slice(0, n)

    print("[2/3] 改写为单 episode（%d 帧）..." % n)
    cols = {name: t.column(name).to_pylist() for name in t.column_names}
    cols["episode_index"] = [0] * n
    cols["frame_index"] = list(range(n))
    cols["index"] = list(range(n))
    cols["timestamp"] = [i / args.fps for i in range(n)]
    new_t = pa.table(cols)

    out_pq = dst / "data" / "chunk-000" / "file-000.parquet"
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_t, str(out_pq))
    print("      parquet:", out_pq, "%.1f MB" % (out_pq.stat().st_size / 1e6))

    print("[3/3] 复制 meta 与视频 ...")
    meta_dst = dst / "meta"
    if meta_dst.exists():
        shutil.rmtree(meta_dst)
    shutil.copytree(src / "meta", meta_dst)

    video_src = Path(args.video)
    # 与真实数据集一致的 v3.0 布局：videos/<video_key>/chunk-000/file-000.mov
    for key in ("observation.deformation.gripperrighttactile",
                "observation.deformation.gripper_left_tactile"):
        vdst = dst / "videos" / key / "chunk-000" / "file-000.mov"
        vdst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_src, vdst)
        print("      video:", vdst, "%.1f MB" % (vdst.stat().st_size / 1e6))

    print("完成:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
