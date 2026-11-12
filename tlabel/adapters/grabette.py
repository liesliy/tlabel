"""
Grabette 数据适配器 — 将 Grabette Tactile 数据集转换为 TLabelData (Schema V2)

数据来源: CarolinePascal/grabette-tactile-full-synced (HuggingFace, Apache-2.0)
  - LeRobot v3.0 格式，48 episodes / 19,353 帧 / 50 fps
  - 任务: "Pick up the white cube and put it in the correct cup"
  - 触觉数据直接存放在 parquet 里（不是视频），两个 int16 网格：
      observation.tactile.right_sensor_1   6×6 = 36 taxel
      observation.tactile.right_sensor_2   4×8 = 32 taxel
  - 其他模态: 2 路 RGB 960×720 (h264)、8 维 action、is_lost 标志

单位说明（重要）:
  数据集未文档化触觉读数的物理单位，也未提供标定系数。因此本适配器：
    - contact / contact_centroid / contact_region 基于原始读数计算，可靠
    - force_magnitude 是**网格压力积分（无量纲原始读数）**，不是牛顿值
    - slip_velocity 是**归一化坐标/秒**，不是 mm/s
    - object_deformation 是**无量纲压力变化率**，不是 mm
  默认 Compliance Level 为 L2（L1 + force_magnitude 的无量纲代理）。
  不填 0、不虚构单位——使用者若有标定系数，可自行换算。

坐标说明:
  contact_centroid 为归一化坐标 [x, y] ∈ [0,1]²（x=列方向, y=行方向）。
  两个阵列尺寸不同（6×6 与 4×8），各自归一化后按受力大小加权合并。
  数据集未文档化两个阵列的物理相对位置，故合并只做加权平均，不做空间拼接
  ——这是适配器的建模约定，不是数据集的物理事实。
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from tlabel.adapters.base import DataAdapterBase
from tlabel.adapters.xela import _taxel_positions
from tlabel.core.schema import TLabelSchemaV2
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel._version import __version__

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:  # pragma: no cover - 取决于安装环境
    HAS_PYARROW = False


# 数据集里的触觉列名与网格形状（顺序即合并时的默认次序）
_GRABETTE_SENSORS: Tuple[Tuple[str, Tuple[int, int]], ...] = (
    ("observation.tactile.right_sensor_1", (6, 6)),
    ("observation.tactile.right_sensor_2", (4, 8)),
)

_GRABETTE_DEFAULT_FPS = 50.0
_GRABETTE_BATCH_ROWS = 2048

# 阈值默认值来自该数据集 19,353 帧的实测分布（接触-接触帧对）：
#   质心位移 p95 = 0.053、压力相对变化 p95 = 0.22
# 取 p95 作为默认，约 9% 的接触帧被判为滑移。
_GRABETTE_SLIP_SHIFT_DEFAULT = 0.05
_GRABETTE_SLIP_PRESSURE_DEFAULT = 0.25


# =============================================================================
#  内部工具函数
# =============================================================================

# 网格形状 → 归一化 taxel 坐标的缓存。坐标只取决于形状，逐帧重建没有意义。
_POSITION_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _positions_for(shape: Tuple[int, int]) -> np.ndarray:
    """按网格形状取归一化 taxel 坐标（带缓存）"""
    pos = _POSITION_CACHE.get(shape)
    if pos is None:
        # _taxel_positions 的三维分支按 (行, 列, 轴) 归一化；传二维会被当成扁平
        # taxel 列表，所以补一个轴维度，复用 xela 的坐标生成逻辑。
        pos = _taxel_positions(shape + (1,))
        _POSITION_CACHE[shape] = pos
    return pos


def _array_stats(grid: Any) -> Dict[str, Any]:
    """单阵列统计：总压力、归一化质心、峰值、有效 taxel 数

    质心为压力加权重心，x=列方向、y=行方向，均在 [0, 1]。
    全零帧（未接触）返回 centroid=None，避免把"没有接触"算成"接触在原点"。
    """
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"触觉网格必须是二维数组，收到 shape={list(g.shape)}")

    total = float(g.sum())
    if total <= 0.0:
        return {"total": 0.0, "centroid": None, "peak": 0.0, "active": 0}

    positions = _positions_for((int(g.shape[0]), int(g.shape[1])))
    weights = g.reshape(-1)
    cx = float(np.dot(weights, positions[:, 0]) / total)
    cy = float(np.dot(weights, positions[:, 1]) / total)
    return {
        "total": total,
        "centroid": (cx, cy),
        "peak": float(g.max()),
        "active": int((g > 0).sum()),
    }


def _grid_region(cx: float, cy: float) -> str:
    """把归一化质心映射到阵列内部的九宫格标签，如 middle_center / top_left

    注意这是**阵列内部的位置**，不是 Schema 的解剖区域。Schema 的
    `contact_region` 枚举是 palmar/digital/lateral/proximal/distal/dorsal/other，
    描述的是手部解剖分区。Grabette 是单指尖传感器，数据集没有文档化阵列
    哪一侧朝向指尖、哪一侧朝近端，按九宫格硬映射成 distal/proximal 就是
    凭空编造。所以这个标签只写进 `sensor_specific["grid_region"]` 作为
    附加信息，`contact_region` 统一取 "palmar"（整个感应面都是指腹）。
    """
    if cx < 1.0 / 3.0:
        xb = "left"
    elif cx > 2.0 / 3.0:
        xb = "right"
    else:
        xb = "center"
    if cy < 1.0 / 3.0:
        yb = "top"
    elif cy > 2.0 / 3.0:
        yb = "bottom"
    else:
        yb = "middle"
    return f"{yb}_{xb}"


class _PhaseTracker:
    """操作阶段在线状态机

    状态转移与 daimon_dataset._PhaseTracker 完全一致，让两个适配器产出的
    manipulation_phase 语义可以直接比较。这里单独实现而不是跨模块 import，
    是为了让 grabette 适配器不依赖 daimon 模块（后者会连带加载 cv2/ffmpeg）。
    """

    __slots__ = ("current",)

    def __init__(self) -> None:
        self.current = "idle"

    def step(self, is_contact: bool, is_slip: bool) -> str:
        c = self.current
        if c == "idle":
            if is_contact:
                c = "initial_contact"
        elif c == "initial_contact":
            if is_slip:
                c = "slip"
            elif is_contact:
                c = "stable_contact"
        elif c == "stable_contact":
            if is_slip:
                c = "slip"
            elif not is_contact:
                c = "release"
        elif c == "slip":
            if not is_slip and is_contact:
                c = "stable_contact"
            elif not is_contact:
                c = "release"
        elif c == "release":
            if is_contact:
                c = "re_contact"
            else:
                c = "idle"
        elif c == "re_contact":
            if is_slip:
                c = "slip"
            elif is_contact:
                c = "stable_contact"
        self.current = c
        return c


def _load_info_json(parquet_path: Path) -> Dict[str, Any]:
    """向上查找并加载 meta/info.json（LeRobot v3.0 的元数据入口）"""
    current = parquet_path.parent
    for _ in range(5):
        info_file = current / "meta" / "info.json"
        if info_file.exists():
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        current = current.parent
    return {}


def _load_tasks(parquet_path: Path) -> Dict[int, str]:
    """加载 meta/tasks.parquet 里的任务描述映射

    注意 LeRobot v3.0 把 task 文本写成 pandas index 而不是普通列
    （parquet 的 pandas metadata 里 index_columns=["task"]），所以不能只查
    df.columns——那样会退化成把 task_index 映射到自己，任务描述变成数字。
    """
    if not HAS_PYARROW:
        return {}
    current = parquet_path.parent
    for _ in range(5):
        tasks_file = current / "meta" / "tasks.parquet"
        if tasks_file.exists():
            try:
                df = pq.read_table(str(tasks_file)).to_pandas()
                if "task_index" not in df.columns:
                    return {}
                for col in ("task", "vlm_hybrid_task"):
                    if col in df.columns:
                        return dict(zip(df["task_index"], df[col]))
                if df.index.name in ("task", "vlm_hybrid_task"):
                    return dict(zip(df["task_index"], df.index))
                return {}
            except Exception:
                pass
        current = current.parent
    return {}


def _resolve_parquet(file_path: Path) -> Path:
    """把目录或 parquet 路径统一成 parquet 文件路径"""
    if file_path.is_file() and file_path.suffix.lower() == ".parquet":
        return file_path
    if file_path.is_dir():
        # LeRobot v3.0: data/chunk-XXX/file-XXX.parquet
        candidates = sorted((file_path / "data").rglob("*.parquet"))
        if not candidates:
            candidates = sorted(file_path.rglob("*.parquet"))
        # 排除 meta 目录下的 parquet（那是 episodes 索引，不是数据）
        candidates = [p for p in candidates if "meta" not in p.parts]
        if not candidates:
            raise FileNotFoundError(f"目录下没有找到数据 parquet: {file_path}")
        return candidates[0]
    raise ValueError(
        f"不支持的路径类型: {file_path}（需要 .parquet 文件或 LeRobot 数据集目录）"
    )


# =============================================================================
#  GrabetteAdapter
# =============================================================================

class GrabetteAdapter(DataAdapterBase):
    """Grabette 双阵列触觉数据集适配器（LeRobot v3.0 parquet）

    将两个尺寸不同的压力网格（6×6 与 4×8，各 taxel 单轴法向读数）映射到
    TLabel Schema V2 (14维)。

    信号到 Schema 的映射：
      - 网格压力积分 → contact（总和超阈值）、force_magnitude（无量纲）
      - 压力加权重心 → contact_centroid（双阵列分别归一化后按受力加权合并）
      - 单指尖指腹 → contact_region 恒为 "palmar"（阵列内部九宫格位置另存
        sensor_specific["grid_region"]，不冒充解剖分区）
      - 帧间质心位移 / 压力突变 → slip_event、slip_velocity
      - 帧间压力变化率 → object_deformation
      - 双阵列接触一致性 + is_lost → confidence

    Compliance Level: L2。force_magnitude 是无量纲原始读数代理，不是牛顿值；
    数据集未提供标定系数，也不提供切向分量，因此 force_vector / torque_vector
    保持 None。

    逐帧 compliance_level：有接触时 L2；无接触帧没有 force_magnitude，而 Schema
    硬约束禁止此时声明 L2，故降为 L1（与 touchd / demo 的处理一致）。adapter 的
    capabilities 仍声明 default_compliance_level = L2。

    参考:
      - 数据集: https://huggingface.co/datasets/CarolinePascal/grabette-tactile-full-synced
    """

    default_compliance_level: str = "L2"

    @property
    def name(self) -> str:
        return "grabette"

    @property
    def supported_extensions(self) -> List[str]:
        return [".parquet"]

    # ─── 能力声明 ────────────────────────────────────────────────────────

    def get_capabilities(self) -> Dict[str, bool]:
        """13 个合法能力键（Schema V2，规范 S-7.4-1）

        以 load() 实际产出的帧内容为准。force_magnitude 会填值，但单位是
        无量纲原始读数——能力声明的是"有这个字段"，不是"有牛顿值"。
        """
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": True,
            "force_magnitude": True,   # 无量纲：网格压力积分
            "force_vector": False,     # 只有法向单轴，无切向分量
            "torque_vector": False,    # 无多轴力、无 taxel 物理间距
            "slip_event": True,
            "slip_velocity": True,     # 归一化坐标/秒，非 mm/s
            "manipulation_phase": True,
            "texture_class": False,    # 非视觉传感器
            "object_deformation": True,  # 无量纲压力变化率
            "temperature": False,
            "confidence": True,
        }

    # ─── 传感器信息 ──────────────────────────────────────────────────────

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "sensor_name": "Grabette tactile array",
            "sensor_type": "distributed_array",
            "type": "distributed_taxel_array",
            "manufacturer": "Grabette",
            "model": "dual pressure grid (6x6 + 4x8)",
            "adapter_name": "tlabel-grabette",
            "adapter_version": __version__,
            "modality": "single-axis normal pressure taxel array",
            "description": (
                "Grabette 机器人手部两片压力阵列：右手 6×6 与 4×8 两个 int16 "
                "网格，每 taxel 只有单轴法向读数。50 Hz 录制。"
            ),
            "axes_per_taxel": 1,
            "taxel_layout_reference": [6, 6],
            "channels": {
                "tactile_sensor_1": {
                    "type": "pressure_grid", "shape": [6, 6], "count": 36,
                    "unit": "raw int16 (数据集未文档化单位)",
                },
                "tactile_sensor_2": {
                    "type": "pressure_grid", "shape": [4, 8], "count": 32,
                    "unit": "raw int16 (数据集未文档化单位)",
                },
            },
            "typical_sample_rate_hz": _GRABETTE_DEFAULT_FPS,
            "compliance_level": self.default_compliance_level,
            "dataset": (
                "CarolinePascal/grabette-tactile-full-synced "
                "(https://huggingface.co/datasets/CarolinePascal/"
                "grabette-tactile-full-synced)"
            ),
        }

    # ─── Schema 提取 ─────────────────────────────────────────────────────

    def extract_schema(self, raw_frame_data: Dict[str, Any],
                       _stats_out: Optional[List[Dict[str, Any]]] = None
                       ) -> TLabelSchemaV2:
        """将单帧双阵列压力数据转换为 TLabel Schema V2

        参数:
            raw_frame_data: 字典，包含：
                - grid_1: np.ndarray (6,6) 右手阵列 1 原始读数
                - grid_2: np.ndarray (4,8) 右手阵列 2 原始读数
                - prev_centroid: 上一帧合并质心 [x,y]（可选，首帧为 None）
                - prev_total: 上一帧总压力（可选）
                - dt: 帧间隔秒数（可选，默认 1/50）
                - is_lost: 该帧是否被标为数据丢失（可选，默认 0）
                - contact_threshold: 总压力接触阈值（可选，默认 0.0）
                - slip_shift_threshold: 质心位移滑移阈值（可选，默认 0.05）
                - slip_pressure_threshold: 压力相对变化滑移阈值（默认 0.25）
            _stats_out: 内部使用。传入列表时，把两个阵列的统计写进去，供调用方
                复用，避免 iter_frames 里重复计算同一帧的质心。

        返回:
            TLabelSchemaV2 — 恒为 L2；force_vector / torque_vector 恒为 None
        """
        stats = [_array_stats(raw_frame_data["grid_1"]),
                 _array_stats(raw_frame_data["grid_2"])]
        if _stats_out is not None:
            _stats_out.extend(stats)

        threshold = float(raw_frame_data.get("contact_threshold", 0.0))
        dt = float(raw_frame_data.get("dt", 1.0 / _GRABETTE_DEFAULT_FPS))
        is_lost = float(raw_frame_data.get("is_lost", 0.0)) > 0.5

        total = stats[0]["total"] + stats[1]["total"]
        is_contact = bool(total > threshold)

        schema = TLabelSchemaV2(
            contact=is_contact,
            slip_event=False,
            confidence=0.9,
            compliance_level=self.default_compliance_level,
        )

        if not is_contact:
            # 无接触帧没有 force_magnitude。Schema 校验规则是
            # "level_num >= 2 必须 force_magnitude 非 None"，所以这里不能
            # 继续挂着 L2，降为 L1。有接触帧仍是 L2。
            schema.compliance_level = "L1"
            schema.confidence = 0.5 if is_lost else 0.95
            return schema

        # force_magnitude: 网格压力积分（无量纲原始读数，不是牛顿值）
        schema.force_magnitude = round(total, 4)

        # contact_centroid: 双阵列分别归一化后按受力加权合并
        weighted = [(s["centroid"], s["total"]) for s in stats
                    if s["centroid"] is not None and s["total"] > 0.0]
        cx = sum(c[0] * w for c, w in weighted) / total
        cy = sum(c[1] * w for c, w in weighted) / total
        schema.contact_centroid = [round(float(cx), 4), round(float(cy), 4)]
        # 单指尖传感器，整个感应面都是指腹；阵列内部的九宫格位置写进
        # sensor_specific["grid_region"]，不冒充解剖分区
        schema.contact_region = "palmar"

        # slip / 形变：需要上一帧
        prev_centroid = raw_frame_data.get("prev_centroid")
        prev_total = raw_frame_data.get("prev_total")
        if prev_centroid is not None and prev_total is not None and prev_total > 0.0:
            dx = cx - float(prev_centroid[0])
            dy = cy - float(prev_centroid[1])
            shift = float(np.hypot(dx, dy))
            rel_pressure = abs(total - float(prev_total)) / max(
                total, float(prev_total), 1.0
            )

            # object_deformation: 无量纲压力变化率（不是 mm）
            schema.object_deformation = round(rel_pressure, 6)

            shift_th = float(raw_frame_data.get(
                "slip_shift_threshold", _GRABETTE_SLIP_SHIFT_DEFAULT))
            press_th = float(raw_frame_data.get(
                "slip_pressure_threshold", _GRABETTE_SLIP_PRESSURE_DEFAULT))

            if shift > shift_th or rel_pressure > press_th:
                schema.slip_event = True
                if dt > 0:
                    # 归一化坐标/秒，不是 mm/s
                    schema.slip_velocity = [round(dx / dt, 4), round(dy / dt, 4)]

        # confidence: 双阵列接触一致性 + 数据丢失标志
        if is_lost:
            schema.confidence = 0.5
        else:
            c1 = stats[0]["total"] > threshold
            c2 = stats[1]["total"] > threshold
            schema.confidence = 0.9 if c1 == c2 else 0.75

        return schema

    # ─── 流式逐帧 ────────────────────────────────────────────────────────

    def iter_frames(self, file_path: str,
                    episode_index: Optional[int] = None,
                    max_frames: Optional[int] = None,
                    meta_out: Optional[Dict[str, Any]] = None,
                    **kwargs) -> Iterator[TLabelFrame]:
        """流式逐帧产出 TLabelFrame

        用 pyarrow 的 iter_batches 按批读 parquet，内存与 batch 大小同阶，
        不随 episode 长度增长。Grabette 的触觉数据在 parquet 里（无视频解码），
        这个数据集只有 1.55 MB，流式在这里的价值是验证路径正确，而不是省内存。

        参数:
            file_path: parquet 文件或 LeRobot 数据集目录
            episode_index: 指定 episode；None 表示第一个
            max_frames: 最多产出多少帧
            meta_out: 传入可变 dict 时写入 sensor_info / episode_info /
                capabilities，供 load() 组装 TLabelData
        """
        if not HAS_PYARROW:
            raise ImportError("grabette 适配器需要 pyarrow: pip install pyarrow")

        parquet_path = _resolve_parquet(Path(file_path))
        info = _load_info_json(parquet_path)
        tasks = _load_tasks(parquet_path)
        fps = float(info.get("fps", _GRABETTE_DEFAULT_FPS))
        dt = 1.0 / fps if fps > 0 else 1.0 / _GRABETTE_DEFAULT_FPS

        contact_threshold = float(kwargs.get("contact_threshold", 0.0))
        slip_shift_th = float(kwargs.get(
            "slip_shift_threshold", _GRABETTE_SLIP_SHIFT_DEFAULT))
        slip_press_th = float(kwargs.get(
            "slip_pressure_threshold", _GRABETTE_SLIP_PRESSURE_DEFAULT))

        sensor_keys = [k for k, _ in _GRABETTE_SENSORS]
        expected_shapes = [s for _, s in _GRABETTE_SENSORS]

        pf = pq.ParquetFile(str(parquet_path))
        available = set(pf.schema_arrow.names)
        missing = [k for k in sensor_keys if k not in available]
        if missing:
            raise ValueError(
                f"parquet 缺少触觉列: {missing}\n实际列: {sorted(available)}"
            )

        target_ep = episode_index
        tracker = _PhaseTracker()
        prev_centroid: Optional[List[float]] = None
        prev_total: Optional[float] = None
        emitted = 0
        num_frames_total = 0
        task_desc = "unknown task"

        columns = sensor_keys + ["timestamp", "frame_index",
                                 "episode_index", "task_index", "is_lost"]
        columns = [c for c in columns if c in available]

        for batch in pf.iter_batches(batch_size=_GRABETTE_BATCH_ROWS,
                                     columns=columns):
            d = batch.to_pydict()
            n_rows = len(d[columns[0]])

            grids = [np.asarray(d[k], dtype=np.float64) for k in sensor_keys]
            for g, shape in zip(grids, expected_shapes):
                if g.shape[1:] != shape:
                    raise ValueError(
                        f"触觉网格形状 {list(g.shape[1:])} 与预期 {list(shape)} 不符"
                    )

            ts = np.asarray(d.get("timestamp", list(range(n_rows))),
                            dtype=np.float64).reshape(-1)
            fi = np.asarray(d.get("frame_index", list(range(n_rows))),
                            dtype=np.int64).reshape(-1)
            eps = (np.asarray(d["episode_index"], dtype=np.int64).reshape(-1)
                   if "episode_index" in d else np.zeros(n_rows, dtype=np.int64))
            lost = (np.asarray(d["is_lost"], dtype=np.float64).reshape(-1)
                    if "is_lost" in d else np.zeros(n_rows, dtype=np.float64))
            tix = (np.asarray(d["task_index"], dtype=np.int64).reshape(-1)
                   if "task_index" in d else np.zeros(n_rows, dtype=np.int64))

            if target_ep is None:
                target_ep = int(eps[0])
                task_desc = tasks.get(int(tix[0]), "unknown task")
            else:
                # LeRobot 的 parquet 按 episode_index 单调排列（已实测确认）。
                # 整批都在目标之前就跳过，整批都在目标之后就可以收工，
                # 否则指定第 N 个 episode 要重扫全表，48 个 episode 会慢一个数量级。
                if n_rows and int(eps.max()) < target_ep:
                    continue
                if n_rows and int(eps.min()) > target_ep:
                    break
                if task_desc == "unknown task" and len(tix):
                    task_desc = tasks.get(int(tix[0]), "unknown task")

            for k in range(n_rows):
                if int(eps[k]) != target_ep:
                    continue

                grid_1, grid_2 = grids[0][k], grids[1][k]
                stats_buf: List[Dict[str, Any]] = []
                schema = self.extract_schema({
                    "grid_1": grid_1,
                    "grid_2": grid_2,
                    "prev_centroid": prev_centroid,
                    "prev_total": prev_total,
                    "dt": dt,
                    "is_lost": float(lost[k]),
                    "contact_threshold": contact_threshold,
                    "slip_shift_threshold": slip_shift_th,
                    "slip_pressure_threshold": slip_press_th,
                }, _stats_out=stats_buf)

                # 供下一帧用的状态
                prev_centroid = schema.contact_centroid
                prev_total = schema.force_magnitude

                s1, s2 = stats_buf

                frame = TLabelFrame(
                    frame_idx=int(fi[k]),
                    timestamp_s=round(float(ts[k]), 4),
                    schema_v2=schema,
                    manipulation_phase="idle",
                    confidence=schema.confidence,
                    sensor_specific={
                        "task": task_desc,
                        "episode_index": int(eps[k]),
                        "is_lost": bool(lost[k]),
                        # 阵列内部九宫格位置（不是 Schema 的解剖区域）
                        "grid_region": (
                            _grid_region(*schema.contact_centroid)
                            if schema.contact_centroid else None
                        ),
                        "sensor_1_total": round(s1["total"], 4),
                        "sensor_2_total": round(s2["total"], 4),
                        "sensor_1_active_taxels": s1["active"],
                        "sensor_2_active_taxels": s2["active"],
                        "sensor_1_peak": round(s1["peak"], 4),
                        "sensor_2_peak": round(s2["peak"], 4),
                        # 原始读数保留，便于下游自行换标定系数重算
                        "raw_grid_1": grid_1.astype(int).tolist(),
                        "raw_grid_2": grid_2.astype(int).tolist(),
                    },
                )
                frame.manipulation_phase = tracker.step(
                    schema.contact, schema.slip_event
                )

                num_frames_total += 1
                yield frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break

            if max_frames is not None and emitted >= max_frames:
                break

        if meta_out is not None:
            meta_out["sensor_info"] = {
                "type": "distributed_taxel_array",
                "model": "dual pressure grid (6x6 + 4x8)",
                "manufacturer": "Grabette",
                "modality": "single-axis normal pressure",
                "layout": {
                    "sensor_1_shape": [6, 6],
                    "sensor_2_shape": [4, 8],
                    "fps": fps,
                    "episode_index": target_ep,
                    "num_frames": num_frames_total,
                    "note": (
                        "force_magnitude 为无量纲原始读数积分；"
                        "slip_velocity 单位为归一化坐标/秒"
                    ),
                },
            }
            meta_out["episode_info"] = {
                "source": "grabette-tactile-full-synced",
                "file": parquet_path.name,
                "robot_type": info.get("robot_type", "grabette"),
                "episode_index": target_ep,
                "task": task_desc,
                "fps": fps,
                "num_frames": num_frames_total,
                "total_episodes": info.get("total_episodes"),
            }
            meta_out["capabilities"] = self.get_capabilities()

    # ─── 批量加载 ────────────────────────────────────────────────────────

    def load(self, file_path: str,
             episode_index: Optional[int] = None,
             max_frames: Optional[int] = None,
             stream: bool = False,
             **kwargs) -> TLabelData:
        """加载 Grabette 数据集并转换为 TLabelData

        参数:
            file_path: parquet 文件路径或 LeRobot 数据集目录
            episode_index: 指定 episode（默认第一个）
            max_frames: 最大帧数
            stream: True 时不保留完整 frames 列表，只返回 aggregate_stats
            **kwargs: contact_threshold / slip_shift_threshold /
                slip_pressure_threshold

        返回:
            TLabelData — 统一标注容器

        异常:
            FileNotFoundError: 路径或数据文件不存在
            ValueError: 缺少触觉列 / 网格形状不符
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
            sensor_id="grabette",
            streamed=stream,
            aggregate_stats={
                "num_frames": meta.get("episode_info", {}).get("num_frames", 0),
                "episode_index": meta.get("episode_info", {}).get("episode_index"),
            },
        )
