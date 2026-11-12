"""
SharpaDex 数据适配器 — 将 SharpaDex v1.0 数据集转换为 TLabelData (Schema V2)

数据来源: Sharpa-Robotics/SharpaDex-v1.0 (HuggingFace, CC-BY-4.0)
  - LeRobot v2.1 与 v3.0 两套格式并存，路径模板由 ``meta/info.json`` 的
    ``data_path`` / ``video_path`` 决定，不硬编码目录结构
  - 顶层 59 个任务目录，每个任务下多个 ``season_XXX``，每个 season 是一份
    独立数据集。实测 ``place_ball_in_box`` 一个 season：74 episodes /
    27,361 帧 / 30 fps
  - 触觉: ``observation.tactile``，60 维 = 2 手 × 5 指 × 6 轴。
    维度名由 ``info.json`` 的 ``features["observation.tactile"]["names"]``
    给出，形如 ``right_thumb_fx`` / ``left_little_tz``。``meta/modality.json``
    **不含** tactile 切片定义，不能只读 modality.json。
  - 其他模态: 65 维关节状态、65 维关节力矩、24 维 TCP（前 12 位姿 +
    12 位恒零的力/力矩）、6 路视频

单位说明（重要，来自数据集 README 第 530 行原文）:

  > Each fingertip's six-axis force/torque values are expressed in its
  > corresponding ``*_elastomer`` local coordinate frame in the Sharpa Wave
  > URDF. Force components Fx, Fy, Fz are in newtons (N); torque components
  > Tx, Ty, Tz are in newton-metres (N·m).

  即 **10 根手指各有自己的局部坐标系**。README 第 428 行同时明确：把指尖
  位置投影到世界系还需要手部几何与关节状态，数据集本身不提供该变换。

  因此本适配器：
    - ``force_magnitude`` = 各指力矢量模长之和（N）。标量，不受坐标系
      旋转影响，可以跨指相加
    - ``force_vector`` / ``torque_vector`` = **受力最大那根手指**的
      [Fx,Fy,Fz] / [Tx,Ty,Tz]，在该指的局部系下。``sensor_specific`` 里用
      ``force_vector_finger`` / ``force_vector_frame`` 标明是哪根手指、
      在哪个坐标系，不冒充整手合力
    - **不把 10 指的矢量直接相加**：那需要先做坐标变换，直接相加得到的
      合力方向是错的

坐标说明:

  ``contact_centroid`` 用的是**手指序位空间**（finger-ordinal space），
  不是手部或世界坐标。10 根手指排在 2×5 的虚拟网格上：
  x = thumb 0.1 / index 0.3 / middle 0.5 / ring 0.7 / little 0.9，
  y = left 0.25 / right 0.75，按各指力模长加权。

  这是适配器的建模约定，不是数据集的物理事实。``sensor_specific`` 里的
  ``contact_centroid_space = "finger_ordinal"`` 显式标注了这一点，
  下游若需要真实空间坐标，应结合 URDF 与关节状态自行投影。

Compliance Level:

  有接触帧 L3（contact / centroid / region / force_magnitude /
  force_vector / torque_vector / slip / phase / deformation / confidence），
  无接触帧 L1（L2 及以上的定义要求 force_magnitude 非 None，无接触帧没有
  力可报，不能挂高等级）。``texture_class`` 需要 ``tactile_raw`` 视频，
  本版本尚未实现，恒为 None；``temperature`` 数据集没有温度传感器，
  恒为 None。
"""

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from tlabel.adapters.base import DataAdapterBase
from tlabel.core.schema import TLabelSchemaV2
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel._version import __version__

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:  # pragma: no cover - 取决于安装环境
    HAS_PYARROW = False


# =============================================================================
#  常量
# =============================================================================

_SHARPA_HANDS: Tuple[str, ...] = ("left", "right")
_SHARPA_FINGERS: Tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")
_SHARPA_AXES: Tuple[str, ...] = ("fx", "fy", "fz", "tx", "ty", "tz")

_SHARPA_DEFAULT_FPS = 30.0
_SHARPA_BATCH_ROWS = 2048

# 手指序位空间的虚拟坐标（见模块 docstring「坐标说明」）
_FINGER_X: Dict[str, float] = {
    "thumb": 0.1, "index": 0.3, "middle": 0.5, "ring": 0.7, "little": 0.9,
}
_HAND_Y: Dict[str, float] = {"left": 0.25, "right": 0.75}

# 接触阈值。实测该 season 的 episode 0：空手阶段各指力模长之和
# 0.03~0.09 N，抓取阶段 13~25 N，中间没有歧义区间，取 0.5 N。
# 0.5 N 也是常见触觉"可感知接触"量级，有物理含义而不是拟合出来的数。
_SHARPA_CONTACT_THRESHOLD_N = 0.5

# 滑移阈值：手指序位空间的质心位移 / 力模长和的相对变化
_SHARPA_SLIP_SHIFT_DEFAULT = 0.05
_SHARPA_SLIP_FORCE_DEFAULT = 0.25

# 标注里的 task_skill → Schema 的 manipulation_phase
# 未命中的 skill 按接触状态落到 hold / pre_contact
_SKILL_TO_PHASE: Dict[str, str] = {
    "pick": "grasp",
    "move": "lift",
    "place": "place",
}


# =============================================================================
#  内部工具函数
# =============================================================================

def _parse_tactile_names(names: List[str]
                         ) -> Dict[Tuple[str, str], Dict[str, int]]:
    """把 60 个触觉维度名解析成 ``{(hand, finger): {axis: index}}``

    ``info.json`` 的 names 形如 ``right_thumb_fx``。解析失败直接报错而不是
    静默跳过——静默跳过会让某根手指的力凭空消失，下游看不出来。
    """
    groups: Dict[Tuple[str, str], Dict[str, int]] = {}
    bad: List[str] = []
    for i, name in enumerate(names):
        parts = str(name).split("_")
        if len(parts) != 3:
            bad.append(name)
            continue
        hand, finger, axis = parts
        if hand not in _SHARPA_HANDS or axis not in _SHARPA_AXES:
            bad.append(name)
            continue
        groups.setdefault((hand, finger), {})[axis] = i

    if bad:
        raise ValueError(
            f"无法解析的触觉维度名: {bad[:6]}"
            f"{' ...' if len(bad) > 6 else ''}"
            f"（期望 <hand>_<finger>_<axis>，axis ∈ {list(_SHARPA_AXES)}）"
        )

    incomplete = {k: sorted(set(_SHARPA_AXES) - set(v))
                  for k, v in groups.items() if len(v) != len(_SHARPA_AXES)}
    if incomplete:
        raise ValueError(
            f"以下手指缺少部分轴: {incomplete}"
        )
    return groups


def _format_template(template: str, episode_index: int, chunks_size: int,
                     extra: Optional[Dict[str, int]] = None) -> str:
    """按 LeRobot 的路径模板生成具体文件路径

    模板里出现的占位符都必须是已知量。``{file_index}`` / ``{file_chunk}``
    属于 v3.0 的"一个文件装多个 episode"布局，它们的值只能从
    ``meta/episodes/`` 分片里查，**不能靠取模猜**——猜错会静默读到别的
    episode 的数据。调用方必须先把真实映射放进 ``extra``。
    """
    chunk = episode_index // max(int(chunks_size), 1)
    fields: Dict[str, int] = {
        "episode_chunk": chunk,
        "chunk_index": chunk,
        "episode_index": episode_index,
    }
    if extra:
        fields.update(extra)

    missing: List[str] = []

    def repl(m: "re.Match[str]") -> str:
        name, _, fmt = m.group(1).partition(":")
        if name not in fields:
            missing.append(name)
            return m.group(0)
        val = fields[name]
        return format(val, fmt) if fmt else str(val)

    out = re.sub(r"\{([^}]+)\}", repl, template)
    if missing:
        raise ValueError(
            f"路径模板 {template!r} 含无法确定的占位符 {sorted(set(missing))}。"
            f"v3.0 的 file_index / file_chunk 必须从 meta/episodes/ 分片读取，"
            f"不能按取模推断。"
        )
    return out


def _load_info(meta_dir: Path) -> Dict[str, Any]:
    p = meta_dir / "info.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"缺少 meta/info.json: {p}\n"
            f"SharpaDex 的目录结构是 <task>/season_XXX/lerobot_v2.1/，"
            f"传路径时请指到 lerobot_v2.1 或 lerobot_v3.0 那一层"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _load_tasks(meta_dir: Path) -> Dict[int, str]:
    """任务描述。v2.1 在 tasks.jsonl，v3.0 在 tasks.parquet

    两者都缺失时（实测该 season 的 tasks.jsonl 在镜像上反复下载失败），
    退回 ``annotations.jsonl`` 的 ``global_task`` / ``tags.task_name``。
    只取真实存在的文本，不拼默认值。
    """
    out: Dict[int, str] = {}

    jl = meta_dir / "tasks.jsonl"
    if jl.is_file():
        for line in jl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "task_index" in r and "task" in r:
                out[int(r["task_index"])] = str(r["task"])
        if out:
            return out

    pqp = meta_dir / "tasks.parquet"
    if pqp.is_file():
        try:
            df = pq.read_table(str(pqp)).to_pandas()
            if "task_index" in df.columns:
                if "task" in df.columns:
                    return {int(k): str(v)
                            for k, v in zip(df["task_index"], df["task"])}
                # v3.0 把任务文本放在 pandas index 里（Grabette 也踩过）
                if df.index.name in ("task", "vlm_hybrid_task"):
                    return {int(k): str(v)
                            for k, v in zip(df["task_index"], df.index)}
        except Exception:
            pass

    # 兜底：annotations.jsonl 里每个 episode 都带全局任务描述
    ann = meta_dir / "annotations.jsonl"
    if ann.is_file():
        for line in ann.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (r.get("global_task")
                    or (r.get("tags") or {}).get("task_name"))
            ti = r.get("task_index", 0)
            if text:
                out[int(ti)] = str(text)
        if out:
            return out
    return out


def _load_segments(meta_dir: Path) -> Dict[int, List[Dict[str, Any]]]:
    """从 ``annotations.jsonl`` 读每个 episode 的分段标注

    每行含 ``episode_index`` 与 ``segments``，segment 有 ``start_step`` /
    ``end_step`` / ``task_skill`` / ``text``。实测该 season 的 74 个 episode
    全部带标注，段覆盖到 362 帧，后面约 96 帧是任务收尾（松手回位）没有
    段覆盖，由调用方按接触状态补。
    """
    out: Dict[int, List[Dict[str, Any]]] = {}
    p = meta_dir / "annotations.jsonl"
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = r.get("episode_index")
        if ep is None:
            continue
        segs: List[Dict[str, Any]] = []
        for s in (r.get("segments") or []):
            if "start_step" not in s or "end_step" not in s:
                continue
            segs.append({
                "start": int(s["start_step"]),
                "end": int(s["end_step"]),
                "skill": (s.get("task_skill") or "").strip().lower(),
                "text": s.get("text"),
            })
        segs.sort(key=lambda s: s["start"])
        out[int(ep)] = segs
    return out


def _load_episode_file_map(meta_dir: Path) -> Dict[int, Dict[str, int]]:
    """v3.0 的 ``meta/episodes/`` 分片 → ``{episode_index: {chunk_index, file_index}}``

    v3.0 把一个 chunk 内的多个 episode 打包进 ``file-XXX.parquet``，episode
    到文件的映射只存在于这些分片里。分片列名随版本略有差异，这里按常见
    列名探测；探测不到就返回空 dict，由调用方报清晰错误。
    """
    out: Dict[int, Dict[str, int]] = {}
    ep_dir = meta_dir / "episodes"
    if not ep_dir.is_dir():
        return out

    cand = {
        "chunk_index": ("chunk_index", "data/chunk_index", "episode_chunk"),
        "file_index": ("file_index", "data/file_index"),
    }
    for p in sorted(ep_dir.rglob("*.parquet")):
        try:
            tbl = pq.read_table(str(p))
        except Exception:
            continue
        names = set(tbl.schema.names)
        if "episode_index" not in names:
            continue
        cols = tbl.to_pydict()
        n = len(cols["episode_index"])
        for i in range(n):
            ep = int(cols["episode_index"][i])
            rec: Dict[str, int] = {}
            for key, aliases in cand.items():
                for a in aliases:
                    if a in names:
                        rec[key] = int(cols[a][i])
                        break
            if rec:
                out[ep] = rec
    return out


def _resolve_dataset_root(file_path: Path) -> Path:
    """把传入路径规整成"含 meta/info.json 的数据集根目录"

    用户可能传到 season 目录（里面有 lerobot_v2.1 / lerobot_v3.0 两个子目录），
    也可能直接传到 lerobot_v2.1。两种都接受，优先用已下载到本地的那个：
    v2.1 优先于 v3.0，因为实测 SharpaDex 的 v3.0 在镜像上大量 503。
    """
    p = Path(file_path)
    if (p / "meta" / "info.json").is_file():
        return p
    for sub in ("lerobot_v2.1", "lerobot_v3.0"):
        cand = p / sub
        if (cand / "meta" / "info.json").is_file():
            return cand
    raise FileNotFoundError(
        f"{p} 下找不到 meta/info.json。\n"
        f"应指向 <task>/season_XXX/lerobot_v2.1（或 lerobot_v3.0）目录。"
    )


def _phase_for(frame_idx: int, contact: bool,
               segments: List[Dict[str, Any]]) -> Tuple[str, str]:
    """按分段标注 + 接触状态定 manipulation_phase

    段内：
      pick  → 有力是 grasp，无力是 pre_contact（还没抓到）
      move  → 有力是 lift，无力是 pre_contact
      place → place
    段外（任务收尾，实测该 season 每 episode 约 20-96 帧）：
      有力是 hold，无力是 pre_contact。

    只输出 Schema 枚举内的值（pre_contact / approach / grasp / lift /
    hold / place），不引入自定义阶段名。

    返回 ``(phase, source)``，source 为 ``annotation_segment``（有标注段覆盖）
    或 ``contact_heuristic``（段外，只能按力判）。
    """
    for s in segments:
        if s["start"] <= frame_idx <= s["end"]:
            skill = s["skill"]
            if skill == "pick":
                return ("grasp" if contact else "pre_contact",
                        "annotation_segment")
            if skill == "move":
                return ("lift" if contact else "pre_contact",
                        "annotation_segment")
            if skill == "place":
                return ("place", "annotation_segment")
            return ("hold" if contact else "pre_contact",
                    "annotation_segment")
    return ("hold" if contact else "pre_contact", "contact_heuristic")


# =============================================================================
#  tactile_raw 视频：10 指指尖图像拼图
# =============================================================================

# 实测布局：2 行 × 5 列，每 tile 320×240，整图 1600×480。
# 行 = 手（row 0 left / row 1 right），列 = 指（thumb/index/middle/ring/little）。
# 验证方法：对每 tile 算帧间 std 下降量，只有 [row1][col0] 与 right_thumb
# 的力曲线同步（frame 195/229/248/300 的 std 下降 -1.43/-1.67/-1.90/-1.95，
# 对应力 13.32/18.84/20.81/19.55）。用"亮斑强度"或 Laplacian 峰值定位会失败，
# 因为传感器表面本身有固定棋盘格纹路，静态结构盖过接触信号。
_TILE_ROWS = 2
_TILE_COLS = 5
_TILE_H = 240
_TILE_W = 320

_TILE_SLICES: Dict[Tuple[str, str], Tuple[slice, slice]] = {}
for _r, _hand in enumerate(_SHARPA_HANDS):
    for _c, _finger in enumerate(_SHARPA_FINGERS):
        _TILE_SLICES[(_hand, _finger)] = (
            slice(_r * _TILE_H, (_r + 1) * _TILE_H),
            slice(_c * _TILE_W, (_c + 1) * _TILE_W),
        )

# ── 视频特征阈值 ──
# 逐像素与基线的绝对差超过该灰度值才算"该像素发生了变化"。
# 实测空手帧的 |diff| 均值 1.2-1.5（纯压缩噪声），接触帧 30+，
# 阈值 15 落在两者之间，远离噪声底。
_DIFF_PIXEL_THRESHOLD = 15.0


class _TactileRawReader:
    """``observation.images.tactile_raw`` 的逐帧 tile 特征提取

    视频是 10 个指尖相机画面的拼图，不是形变场，不能套用 daimon 的
    deformation/shear 特征函数。这里按 tile 做**逐像素与基线的差异**：

    - ``area``：|frame - baseline| > 15 的像素比例（0-1）
      接触时指尖弹性体被压出形变，大片像素灰度改变；空手时只剩
      压缩噪声。实测 episode 0 的 right_thumb：空手 area=0.0000，
      抓稳 area=0.56，释放后 0.03-0.08。
    - ``rms`` ：|frame - baseline| 的均方根（灰度级 0-255）
      同时反映形变面积和形变深度。

    为什么不用全局 std 或 Laplacian：实测接触时 right_thumb 的
    std 只从 32.69 升到 34.53（+5.6%），Laplacian 方差变化不到 10%，
    和压缩噪声同量级；而逐像素差异是 20 倍以上的变化。全局统计量
    被传感器固定的棋盘格纹路稀释了。

    基线取**前 30 帧**逐 tile 的像素均值。实测每个 episode 前约
    190 帧是空手接近阶段，前 30 帧一定是无接触状态。
    """

    def __init__(self, video_path: Path, baseline_frames: int = 30,
                 diff_threshold: float = _DIFF_PIXEL_THRESHOLD):
        try:
            import cv2
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "读取 tactile_raw 需要 opencv-python: pip install opencv-python"
            ) from e
        self._cv2 = cv2
        self.path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"无法打开视频: {self.path}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.baseline_frames = baseline_frames
        self.diff_threshold = float(diff_threshold)
        self._baseline: Optional[Dict[Tuple[str, str], np.ndarray]] = None
        self._head: List[np.ndarray] = []
        self._cursor = 0
        self._prev_active: List[Tuple[str, str]] = []

    # ── 内部 ──

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY
                                  ).astype(np.float32)

    def _ensure_baseline(self) -> None:
        if self._baseline is not None:
            return
        acc: Dict[Tuple[str, str], List[np.ndarray]] = {
            k: [] for k in _TILE_SLICES
        }
        while len(self._head) < self.baseline_frames:
            ok, frame = self.cap.read()
            if not ok:
                break
            self._head.append(frame)
            g = self._gray(frame)
            for key, (ys, xs) in _TILE_SLICES.items():
                acc[key].append(g[ys, xs])
            self._cursor += 1
        if not self._head:
            raise IOError(f"视频读不到任何帧: {self.path}")
        self._baseline = {
            k: np.mean(v, axis=0).astype(np.float32)
            for k, v in acc.items() if v
        }

    def _tile_diff(self, g: np.ndarray, key: Tuple[str, str]
                   ) -> Optional[Dict[str, float]]:
        b = self._baseline.get(key) if self._baseline else None
        if b is None:
            return None
        ys, xs = _TILE_SLICES[key]
        d = np.abs(g[ys, xs] - b)
        area = float((d > self.diff_threshold).mean())
        rms = float(np.sqrt(float((d * d).mean())))
        return {"area": area, "rms": rms}

    # ── 对外 ──

    def read_next(self, keys: Optional[List[Tuple[str, str]]] = None
                  ) -> Optional[Dict[Tuple[str, str], Dict[str, float]]]:
        """按顺序读下一帧，返回指定 tile 的差异特征；读完返回 None

        ``keys`` 为 None 时算全部 10 个 tile。传入手上有力的手指列表可以
        只算那几个 tile，同时并上上一帧的集合，保证释放过程不断档。
        """
        self._ensure_baseline()
        if self._cursor < len(self._head):
            frame = self._head[self._cursor]
        else:
            ok, frame = self.cap.read()
            if not ok:
                return None
        self._cursor += 1

        prev = list(self._prev_active)
        targets = set(keys or []) | set(prev)
        if not targets:
            targets = set(_TILE_SLICES)
        self._prev_active = list(keys or [])

        g = self._gray(frame)
        feats: Dict[Tuple[str, str], Dict[str, float]] = {}
        for key in targets:
            f = self._tile_diff(g, key)
            if f is not None:
                feats[key] = f
        return feats

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

    def __enter__(self) -> "_TactileRawReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()



# =============================================================================
#  SharpaDexAdapter
# =============================================================================

class SharpaDexAdapter(DataAdapterBase):
    """SharpaDex 五指六轴触觉数据集适配器（LeRobot v2.1 / v3.0）

    60 维触觉（2 手 × 5 指 × 6 轴）映射到 TLabel Schema V2 (14 维)。

    信号到 Schema 的映射：
      - 各指力模长之和 → force_magnitude（N，可跨指相加的标量）
      - 受力最大那指 → force_vector / torque_vector（N / N·m，该指局部系）
      - 各指力加权 → contact_centroid（手指序位空间，非物理坐标）
      - 指尖接触 → contact_region 恒为 "digital"
      - 帧间质心位移 / 力突变 → slip_event、slip_velocity
      - 帧间力变化率 → object_deformation（无量纲）
      - 分段标注 task_skill + 接触状态 → manipulation_phase
      - 参与接触的指数量 → confidence

    参考:
      - 数据集: https://huggingface.co/datasets/Sharpa-Robotics/SharpaDex-v1.0
    """

    default_compliance_level: str = "L3"

    @property
    def name(self) -> str:
        return "sharpadex"

    @property
    def supported_extensions(self) -> List[str]:
        return [".parquet"]

    # ─── 能力声明 ────────────────────────────────────────────────────────

    def get_capabilities(self) -> Dict[str, bool]:
        """13 个合法能力键（Schema V2，规范 S-7.4-1）

        ``texture_class`` 与 ``temperature`` 为 False：数据集没有温度传感器，
        纹理分类需要 tactile_raw 视频，本版本未实现。
        """
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": True,
            "force_magnitude": True,   # N，各指力模长之和
            "force_vector": True,      # N，最大受力指的局部系矢量
            "torque_vector": True,     # N·m，同一指的局部系力矩
            "slip_event": True,
            "slip_velocity": True,     # 手指序位空间/秒
            "manipulation_phase": True,
            "texture_class": False,
            "object_deformation": True,  # 无量纲力变化率
            "temperature": False,
            "confidence": True,
        }

    # ─── 传感器信息 ──────────────────────────────────────────────────────

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "sensor_name": "Sharpa Wave fingertip six-axis tactile",
            "sensor_type": "fingertip_wrench_array",
            "type": "fingertip_six_axis",
            "manufacturer": "Sharpa Robotics",
            "model": "Sharpa Wave (5 fingers x 6 axes per hand)",
            "adapter_name": "tlabel-sharpadex",
            "adapter_version": __version__,
            "modality": "six-axis force/torque per fingertip",
            "description": (
                "Sharpa Wave 五指指尖六轴力/力矩传感器：每指 3 轴力 + 3 轴力矩，"
                "双手共 60 维。30 fps 录制。"
            ),
            "axes_per_taxel": 6,
            "channels": {
                "tactile": {
                    "type": "fingertip_wrench",
                    "shape": [2, 5, 6],
                    "count": 60,
                    "unit": "N (force) / N·m (torque)",
                    "frame": "per-fingertip *_elastomer local frame (Sharpa Wave URDF)",
                },
            },
            "typical_sample_rate_hz": _SHARPA_DEFAULT_FPS,
            "compliance_level": self.default_compliance_level,
            "dataset": (
                "Sharpa-Robotics/SharpaDex-v1.0 "
                "(https://huggingface.co/datasets/Sharpa-Robotics/SharpaDex-v1.0)"
            ),
        }

    # ─── Schema 提取 ─────────────────────────────────────────────────────

    def extract_schema(self, raw_frame_data: Dict[str, Any],
                       _stats_out: Optional[List[Dict[str, Any]]] = None
                       ) -> TLabelSchemaV2:
        """将单帧 60 维触觉读数转换为 TLabel Schema V2

        参数:
            raw_frame_data: 字典，包含：
                - tactile: np.ndarray (60,) 触觉读数，顺序同 info.json 的 names
                - groups: {(hand, finger): {axis: index}}，由
                  ``_parse_tactile_names`` 得到；缺省用实例上缓存的
                - prev_centroid: 上一帧 contact_centroid（可选）
                - prev_total: 上一帧 force_magnitude（可选）
                - dt: 帧间隔秒数（可选，默认 1/30）
                - contact_threshold: 接触阈值 N（可选，默认 0.5）
                - slip_shift_threshold / slip_force_threshold: 滑移阈值
            _stats_out: 内部使用。传入列表时写入逐指统计，供 iter_frames 复用。

        返回:
            TLabelSchemaV2 — 有接触 L3，无接触 L1
        """
        vec = np.asarray(raw_frame_data["tactile"], dtype=np.float64).reshape(-1)
        groups: Dict[Tuple[str, str], Dict[str, int]] = (
            raw_frame_data.get("groups") or getattr(self, "_groups", None) or {}
        )
        if not groups:
            raise ValueError("缺少触觉维度分组（groups），无法解析 60 维读数")

        expected = len(groups) * len(_SHARPA_AXES)
        if vec.size != expected:
            raise ValueError(
                f"触觉向量长度 {vec.size} 与分组预期 {expected} 不符"
            )

        threshold = float(raw_frame_data.get(
            "contact_threshold", _SHARPA_CONTACT_THRESHOLD_N))
        dt = float(raw_frame_data.get("dt", 1.0 / _SHARPA_DEFAULT_FPS))

        # ── 逐指力/力矩 ──
        per_finger: Dict[Tuple[str, str], Dict[str, float]] = {}
        for key, ax in groups.items():
            f = np.array([vec[ax[a]] for a in ("fx", "fy", "fz")], dtype=np.float64)
            t = np.array([vec[ax[a]] for a in ("tx", "ty", "tz")], dtype=np.float64)
            per_finger[key] = {
                "force_norm": float(np.linalg.norm(f)),
                "torque_norm": float(np.linalg.norm(t)),
                "force": f,
                "torque": t,
            }
        if _stats_out is not None:
            _stats_out.append({
                k: {"force_norm": round(v["force_norm"], 6),
                    "torque_norm": round(v["torque_norm"], 6)}
                for k, v in per_finger.items()
            })

        total = float(sum(v["force_norm"] for v in per_finger.values()))
        is_contact = bool(total > threshold)

        schema = TLabelSchemaV2(
            contact=is_contact,
            slip_event=False,
            confidence=0.9,
            compliance_level=self.default_compliance_level,
        )

        if not is_contact:
            # 无接触帧没有力可报。Schema 规定 L2 及以上必须 force_magnitude
            # 非 None，所以只能挂 L1。
            schema.compliance_level = "L1"
            schema.confidence = 0.95
            return schema

        # ── force_magnitude：各指模长之和（N） ──
        schema.force_magnitude = round(total, 4)

        # ── 受力最大的指：矢量与力矩取它的局部系 ──
        top_key = max(per_finger, key=lambda k: per_finger[k]["force_norm"])
        top = per_finger[top_key]
        schema.force_vector = [round(float(x), 4) for x in top["force"]]
        schema.torque_vector = [round(float(x), 4) for x in top["torque"]]

        # ── contact_centroid：手指序位空间加权 ──
        wx = sum(_FINGER_X[f] * v["force_norm"]
                 for (h, f), v in per_finger.items())
        wy = sum(_HAND_Y[h] * v["force_norm"]
                 for (h, f), v in per_finger.items())
        cx = wx / total
        cy = wy / total
        schema.contact_centroid = [round(float(cx), 4), round(float(cy), 4)]

        # 指尖接触，属"digital"分区；不是手掌（palmar）
        schema.contact_region = "digital"

        # ── slip / 形变：需要上一帧 ──
        prev_centroid = raw_frame_data.get("prev_centroid")
        prev_total = raw_frame_data.get("prev_total")
        if prev_centroid is not None and prev_total is not None and prev_total > 0.0:
            dx = cx - float(prev_centroid[0])
            dy = cy - float(prev_centroid[1])
            shift = float(np.hypot(dx, dy))
            rel_force = abs(total - float(prev_total)) / max(
                total, float(prev_total), 1.0
            )

            # object_deformation：无量纲力变化率（不是 mm）
            schema.object_deformation = round(rel_force, 6)

            shift_th = float(raw_frame_data.get(
                "slip_shift_threshold", _SHARPA_SLIP_SHIFT_DEFAULT))
            force_th = float(raw_frame_data.get(
                "slip_force_threshold", _SHARPA_SLIP_FORCE_DEFAULT))

            if shift > shift_th or rel_force > force_th:
                schema.slip_event = True
                if dt > 0:
                    # 手指序位空间/秒，不是 mm/s
                    schema.slip_velocity = [round(dx / dt, 4), round(dy / dt, 4)]

        # ── confidence：参与接触的指数越多越可信 ──
        active_keys = [k for k, v in per_finger.items()
                       if v["force_norm"] > threshold]
        if len(active_keys) >= 3:
            schema.confidence = 0.92
        elif len(active_keys) == 2:
            schema.confidence = 0.85
        else:
            schema.confidence = 0.75

        # ── 视频派生：object_deformation ──
        # ``tactile_deform`` 实测 459 帧全黑（mean=0, std=0），是空占位视频；
        # 只能用 ``tactile_raw`` 的逐像素差异。
        #
        # 有视频时接触帧能给出形变比例，但等级仍是 L3：
        #   - texture_class 判不出来（见 _TactileRawReader 的说明，图像证据
        #     只反映形变区域，不反映表面材质）
        #   - temperature 数据集没有温度传感器
        # 所以 Optional 维度并没有填满，不能声明 L4。
        video_feats = raw_frame_data.get("video_features")
        if video_feats:
            areas = [video_feats[k]["area"] for k in active_keys
                     if k in video_feats]
            if areas:
                # 指尖表面发生灰度变化的像素比例，无量纲 ratio，不是 mm
                schema.object_deformation = round(float(max(areas)), 4)

        return schema

    # ─── 流式逐帧 ────────────────────────────────────────────────────────

    def iter_frames(self, file_path: str,
                    episode_index: Optional[int] = None,
                    max_frames: Optional[int] = None,
                    meta_out: Optional[Dict[str, Any]] = None,
                    **kwargs) -> Iterator[TLabelFrame]:
        """流式逐帧产出 TLabelFrame

        参数:
            file_path: 数据集根目录（含 meta/info.json），或 season 目录，
                或单个 episode 的 parquet 文件
            episode_index: 指定 episode；None 表示第一个
            max_frames: 最多产出多少帧
            meta_out: 传入可变 dict 时写入 sensor_info / episode_info /
                capabilities，供 load() 组装 TLabelData
            **kwargs: contact_threshold / slip_shift_threshold /
                slip_force_threshold / data_path_override
        """
        if not HAS_PYARROW:
            raise ImportError("sharpadex 适配器需要 pyarrow: pip install pyarrow")

        src = Path(file_path)
        if src.is_file() and src.suffix.lower() == ".parquet":
            parquet_path = src
            # data/chunk-XXX/episode_YYYYYY.parquet → 上三级才是数据集根
            cand = [src.parent.parent.parent / "meta",
                    src.parent.parent / "meta"]
            meta_dir = next((c for c in cand if (c / "info.json").is_file()),
                            cand[0])
            info = _load_info(meta_dir) if meta_dir.is_dir() else {}
            root = meta_dir.parent
        else:
            root = _resolve_dataset_root(src)
            meta_dir = root / "meta"
            info = _load_info(meta_dir)
            parquet_path = self._episode_parquet(
                root, meta_dir, info,
                int(episode_index if episode_index is not None else 0),
                kwargs.get("data_path_override"),
            )

        if not parquet_path.is_file():
            raise FileNotFoundError(f"episode parquet 不存在: {parquet_path}")

        # ── 触觉维度分组 ──
        tac_feat = (info.get("features") or {}).get("observation.tactile") or {}
        names = tac_feat.get("names")
        if not names:
            raise ValueError(
                "meta/info.json 缺少 features['observation.tactile']['names']，"
                "无法把 60 维读数对应到手指和轴"
            )
        groups = _parse_tactile_names(list(names))
        self._groups = groups

        fps = float(info.get("fps", _SHARPA_DEFAULT_FPS))
        dt = 1.0 / fps if fps > 0 else 1.0 / _SHARPA_DEFAULT_FPS

        tasks = _load_tasks(meta_dir) if meta_dir.is_dir() else {}
        segments = _load_segments(meta_dir) if meta_dir.is_dir() else {}

        contact_threshold = float(kwargs.get(
            "contact_threshold", _SHARPA_CONTACT_THRESHOLD_N))
        slip_shift_th = float(kwargs.get(
            "slip_shift_threshold", _SHARPA_SLIP_SHIFT_DEFAULT))
        slip_force_th = float(kwargs.get(
            "slip_force_threshold", _SHARPA_SLIP_FORCE_DEFAULT))

        target_ep = int(episode_index if episode_index is not None else 0)
        ep_segments = segments.get(target_ep, [])

        # ── tactile_raw 视频（可选） ──
        # tactile_deform 实测 459 帧全黑（mean=0, std=0），是空占位视频，
        # 不能用来算形变/纹理。可用的只有 tactile_raw。
        video_key = kwargs.get("video_key", "observation.images.tactile_raw")
        use_video = kwargs.get("use_video", True)
        reader: Optional[_TactileRawReader] = None
        video_note = "disabled"
        if use_video:
            try:
                vpath = self._video_path(root, info, target_ep, video_key)
                if vpath.is_file():
                    reader = _TactileRawReader(vpath)
                    video_note = f"{video_key} ({reader.n_frames} frames)"
                else:
                    video_note = f"missing: {vpath.name}"
            except Exception as e:      # 视频坏了不该让整条链路失败
                video_note = f"unavailable: {e}"
                reader = None

        pf = pq.ParquetFile(str(parquet_path))
        available = set(pf.schema_arrow.names)
        if "observation.tactile" not in available:
            raise ValueError(
                f"parquet 缺少 observation.tactile 列\n实际列: {sorted(available)}"
            )

        cols = [c for c in ("observation.tactile", "timestamp", "frame_index",
                            "episode_index", "task_index", "subtask_index")
                if c in available]

        prev_centroid: Optional[List[float]] = None
        prev_total: Optional[float] = None
        emitted = 0
        num_frames_total = 0
        task_desc = "unknown task"
        first_task_index: Optional[int] = None
        # 滑移比例的滑动窗口，供 texture_class 的 slippery 判据使用
        slip_window: deque = deque(maxlen=10)
        video_frames_used = 0
        prev_active_keys: List[Tuple[str, str]] = []

        try:
            for batch in pf.iter_batches(batch_size=_SHARPA_BATCH_ROWS,
                                         columns=cols):
                d = batch.to_pydict()
                n_rows = len(d["observation.tactile"])

                tac = np.asarray(d["observation.tactile"], dtype=np.float64)
                if tac.ndim != 2:
                    tac = tac.reshape(n_rows, -1)

                ts = np.asarray(d.get("timestamp", list(range(n_rows))),
                                dtype=np.float64).reshape(-1)
                fi = np.asarray(d.get("frame_index", list(range(n_rows))),
                                dtype=np.int64).reshape(-1)
                ep_col = (np.asarray(d["episode_index"], dtype=np.int64)
                          .reshape(-1) if "episode_index" in d else None)
                st_col = (np.asarray(d["subtask_index"], dtype=np.int64)
                          .reshape(-1) if "subtask_index" in d else None)
                tix = (np.asarray(d["task_index"], dtype=np.int64).reshape(-1)
                       if "task_index" in d else np.zeros(n_rows, dtype=np.int64))

                if first_task_index is None and len(tix):
                    first_task_index = int(tix[0])
                    task_desc = tasks.get(first_task_index, "unknown task")

                for k in range(n_rows):
                    # 视频与 parquet 行按顺序一一推进。视频帧数比行数少时
                    # 后续帧拿不到特征，宁可返回 None 也不错位对齐。
                    # 只算上一帧有受力的 tile（内部再并上一帧集合），
                    # 避免每帧都算全部 10 个 tile。
                    vfeats = None
                    if reader is not None:
                        try:
                            vfeats = reader.read_next(prev_active_keys)
                        except Exception:
                            vfeats = None
                    if vfeats is not None:
                        video_frames_used += 1

                    stats_buf: List[Dict[str, Any]] = []
                    schema = self.extract_schema({
                        "tactile": tac[k],
                        "groups": groups,
                        "prev_centroid": prev_centroid,
                        "prev_total": prev_total,
                        "dt": dt,
                        "contact_threshold": contact_threshold,
                        "slip_shift_threshold": slip_shift_th,
                        "slip_force_threshold": slip_force_th,
                        "video_features": vfeats,
                    }, _stats_out=stats_buf)

                    slip_window.append(1 if schema.slip_event else 0)

                    prev_centroid = schema.contact_centroid
                    prev_total = schema.force_magnitude

                    per_finger = stats_buf[0]
                    prev_active_keys = [kk for kk, vv in per_finger.items()
                                        if vv["force_norm"] > contact_threshold]
                    top_finger = max(per_finger,
                                     key=lambda kk: per_finger[kk]["force_norm"])
                    frame_no = int(fi[k])

                    phase, phase_src = _phase_for(
                        frame_no, schema.contact, ep_segments)
                    # manipulation_phase 本身是 Schema 字段；frame 层再挂一份，
                    # 方便下游直接读 frame.manipulation_phase
                    schema.manipulation_phase = phase

                    frame = TLabelFrame(
                        frame_idx=frame_no,
                        timestamp_s=round(float(ts[k]), 4),
                        schema_v2=schema,
                        manipulation_phase=phase,
                        confidence=schema.confidence,
                        sensor_specific={
                            "task": task_desc,
                            "episode_index": int(
                                ep_col[k] if ep_col is not None else target_ep),
                            "subtask_index": int(
                                st_col[k] if st_col is not None else -1),
                            # 手指序位空间，不是物理坐标
                            "contact_centroid_space": "finger_ordinal",
                            # force_vector / torque_vector 属于哪根手指、哪个坐标系
                            "force_vector_finger": (
                                f"{top_finger[0]}_{top_finger[1]}"),
                            "force_vector_frame": "fingertip_elastomer_local",
                            "force_unit": "N",
                            "torque_unit": "N·m",
                            # 逐指力/力矩模长，单位 N / N·m
                            # key 用 "right_thumb" 字符串，tuple key 无法进 JSON
                            "per_finger": {
                                f"{h}_{f}": v
                                for (h, f), v in per_finger.items()
                            },
                            "contact_threshold_n": contact_threshold,
                            "phase_source": phase_src,
                            # texture_class / object_deformation 是从
                            # tactile_raw 图像算的启发式推断，不是数据集标注
                            "video_source": (
                                video_key if vfeats is not None else None),
                            "texture_class_source": (
                                "heuristic_tactile_raw"
                                if schema.texture_class else None),
                            "object_deformation_unit": "ratio",
                        },
                    )
                    num_frames_total += 1
                    yield frame
                    emitted += 1
                    if max_frames is not None and emitted >= max_frames:
                        break

                if max_frames is not None and emitted >= max_frames:
                    break
        finally:
            if reader is not None:
                reader.close()

        if meta_out is not None:
            meta_out["sensor_info"] = {
                "type": "fingertip_six_axis",
                "model": "Sharpa Wave (5 fingers x 6 axes per hand)",
                "manufacturer": "Sharpa Robotics",
                "modality": "six-axis force/torque per fingertip",
                "layout": {
                    "hands": list(_SHARPA_HANDS),
                    "fingers": list(_SHARPA_FINGERS),
                    "axes": list(_SHARPA_AXES),
                    "dims": len(groups) * len(_SHARPA_AXES),
                    "fps": fps,
                    "episode_index": target_ep,
                    "num_frames": num_frames_total,
                    "note": (
                        "力 N / 力矩 N·m，各自在 *_elastomer 局部坐标系；"
                        "force_magnitude 为各指模长之和；"
                        "contact_centroid 为手指序位空间"
                    ),
                },
            }
            meta_out["episode_info"] = {
                "source": "SharpaDex-v1.0",
                "file": parquet_path.name,
                "robot_type": info.get("robot_type", "robot"),
                "episode_index": target_ep,
                "task": task_desc,
                "fps": fps,
                "num_frames": num_frames_total,
                "total_episodes": info.get("total_episodes"),
                "num_segments": len(ep_segments),
                "video": video_note,
                "video_frames_used": video_frames_used,
            }
            meta_out["capabilities"] = self.get_capabilities()

    # ─── 路径定位 ────────────────────────────────────────────────────────

    def _episode_parquet(self, root: Path, meta_dir: Path,
                         info: Dict[str, Any], episode_index: int,
                         template_override: Optional[str] = None) -> Path:
        """按 info.json 的 data_path 模板定位某个 episode 的 parquet

        v2.1 模板是 ``data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet``，
        只靠取整就能算出来。v3.0 模板含 ``{file_index}``，必须查
        ``meta/episodes/`` 分片，否则会读错 episode。
        """
        template = template_override or info.get("data_path")
        if not template:
            raise ValueError(
                "meta/info.json 缺少 data_path，无法定位 episode 文件"
            )
        chunks_size = int(info.get("chunks_size", 1000))

        extra: Dict[str, int] = {}
        if "{file_index" in template or "{file_chunk" in template:
            mapping = _load_episode_file_map(meta_dir)
            if episode_index not in mapping:
                raise ValueError(
                    f"这是 LeRobot v3.0 的布局（data_path={template!r}），"
                    f"episode {episode_index} 的 chunk/file 索引只能从 "
                    f"meta/episodes/ 分片读取，但没读到对应记录。"
                    f"请确认 meta/episodes/ 已完整下载；"
                    f"或改用同一 season 的 lerobot_v2.1 目录。"
                )
            extra = mapping[episode_index]

        rel = _format_template(template, episode_index, chunks_size, extra)
        return root / rel

    def _video_path(self, root: Path, info: Dict[str, Any],
                    episode_index: int, video_key: str) -> Path:
        """按 info.json 的 video_path 模板定位某个视频

        模板形如 ``videos/chunk-{episode_chunk:03d}/{video_key}/
        episode_{episode_index:06d}.mp4``。``{video_key}`` 是字符串字段，
        先单独替换，剩下的数字占位符交给 ``_format_template``。
        """
        template = info.get("video_path")
        if not template:
            return root / "__no_video_template__"
        chunks_size = int(info.get("chunks_size", 1000))
        tpl = template.replace("{video_key}", video_key)
        return root / _format_template(tpl, episode_index, chunks_size)

    # ─── 批量加载 ────────────────────────────────────────────────────────

    def load(self, file_path: str,
             episode_index: Optional[int] = None,
             max_frames: Optional[int] = None,
             stream: bool = False,
             **kwargs) -> TLabelData:
        """加载 SharpaDex 数据集并转换为 TLabelData

        参数:
            file_path: 数据集根目录 / season 目录 / 单个 episode parquet
            episode_index: 指定 episode（默认第一个）
            max_frames: 最大帧数
            stream: True 时不保留完整 frames 列表，只返回 aggregate_stats
            **kwargs: contact_threshold / slip_shift_threshold /
                slip_force_threshold / data_path_override

        返回:
            TLabelData — 统一标注容器

        异常:
            FileNotFoundError: 路径或数据文件不存在
            ValueError: 缺 info.json / 缺触觉维度名 / 触觉向量长度不符
            ImportError: 未安装 pyarrow
        """
        meta: Dict[str, Any] = {}
        frames: List[TLabelFrame] = []

        for frame in self.iter_frames(file_path,
                                      episode_index=episode_index,
                                      max_frames=max_frames,
                                      meta_out=meta,
                                      **kwargs):
            if not stream:
                frames.append(frame)

        return TLabelData(
            frames=frames,
            sensor_info=meta.get("sensor_info", {}),
            episode_info=meta.get("episode_info", {}),
            capabilities=meta.get("capabilities", self.get_capabilities()),
            sensor_id="sharpadex",
            streamed=stream,
            aggregate_stats={
                "num_frames": meta.get("episode_info", {}).get("num_frames", 0),
                "episode_index": meta.get("episode_info", {}).get("episode_index"),
            },
        )
