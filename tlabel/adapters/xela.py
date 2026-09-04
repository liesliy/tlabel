"""
XELA uSkin 数据适配器 — 将 UniTac-NV 公开数据集的 XELA CSV 转换为 TLabelData (Schema V2)

XELA Robotics uSkin (uSPa 46 贴片) 是基于霍尔效应的分布式触觉传感器：
  - 4×6 = 24 个 taxel，每个 taxel 测量 3 轴力（X/Y 剪切 + Z 法向，见 XELA 官方文档）
  - 每帧 4×6×3 = 72 个原始读数

第一版支持的数据来源: UniTac-NV 数据集
  https://github.com/JiannnH/UniTac-NV (IROS 2025, arXiv:2506.19699)
  - 每个录制一个 CSV，每行 6 列（按列位置，与上游加载器一致）:
      0: time                          时间戳 "YYYY-MM-DD HH:MM:SS.ffffff"
      1: seq                           序列号
      2: sensor_matrices_force         力矩阵，Python字面量嵌套列表 (4×6×3)
      3: sensor_matrices_displacement  位移矩阵，同上编码（该数据集实测恒为 0）
      4: FT 真值                       六维力/力矩列表（论文标注力单位为 N）
      5: 末端位姿                      (x, y, z, rx, ry, rz) 列表
  - 采样率约 100 Hz（论文标称）
  - 表头命名以实际文件为准；适配器优先按表头别名识别，无法识别时按列位置解析

单位说明（重要）:
  UniTac-NV 数据集未文档化 uSkin 力/位移读数的物理单位（论文仅注明 FT 真值
  力单位为 N）。因此本适配器默认 Compliance Level 为 L1：
    - contact / contact_centroid 基于原始读数偏差计算（无量纲/归一化，可靠）
    - force_magnitude / force_vector 等物理量字段默认为 None，不填 0 或虚构单位
    - 使用者已知标定系数时，可通过 load(force_scale=...) 将原始读数换算为 N，
      此时填充力字段并将 compliance_level 升级为 L3（力向量在传感器坐标系，
      Z 为法向轴；taxel 物理间距数据集未提供，不做力矩换算）

坐标说明:
  contact_centroid 为 taxel 网格的归一化坐标 [x, y] ∈ [0,1]²（x=列方向,
  y=行方向）。这是适配器的建模约定——数据集未文档化贴片的物理朝向。
"""

import ast
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tlabel.adapters.base import DataAdapterBase
from tlabel.core.schema import TLabelSchemaV2
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel._version import __version__

# UniTac-NV 论文标称采样率 100 Hz（实际录制约 99.6–100.0 Hz）
_XELA_SAMPLE_RATE_DEFAULT = 100.0

# CSV 表头别名（大小写不敏感）。未匹配到力矩阵列时按列位置 0–5 解析。
_HEADER_ALIASES = {
    "time": ("time", "timestamp", "t"),
    "seq": ("seq", "sequence", "seq_num", "frame"),
    "force": ("sensor_matrices_force", "sensor_matrices_forces",
              "force_matrix", "forces", "force"),
    "displacement": ("sensor_matrices_displacement", "sensor_matrices_displacements",
                     "displacement_matrix", "displacements", "displacement"),
    "ft": ("ft", "ft_values", "ft_ground_truth", "ft_sensor", "wrench"),
    "pose": ("end_effector_pose", "end_effector_poses", "pose", "ee_pose"),
}

# 无表头时的列位置（与 UniTac-NV 上游加载器一致）
_POSITIONAL_COLUMNS = {"time": 0, "seq": 1, "force": 2,
                       "displacement": 3, "ft": 4, "pose": 5}


# =============================================================================
#  内部工具函数
# =============================================================================

def _parse_taxel_matrix(cell: str, field_name: str, row_idx: int) -> np.ndarray:
    """解析单个 taxel 矩阵单元格（Python 字面量嵌套列表）并校验形状

    接受的形状（XELA uSkin 每 taxel 3 轴）:
      - (行, 列, 3)：UniTac-NV 原始 CSV 的 4×6×3 力/位移矩阵
      - (taxel数, 3)：UniTac-NV 预处理脚本的扁平输出
      - (3,)：单 taxel

    异常:
        ValueError: 无法解析、包含非数值元素或形状不符
    """
    try:
        obj = ast.literal_eval(cell)
    except (SyntaxError, ValueError) as e:
        raise ValueError(
            f"CSV 第 {row_idx} 行 {field_name} 无法解析为嵌套列表: {e}"
        )
    try:
        arr = np.asarray(obj, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"CSV 第 {row_idx} 行 {field_name} 包含非数值元素: {e}"
        )
    if arr.size == 0:
        raise ValueError(f"CSV 第 {row_idx} 行 {field_name} 为空数组")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    if arr.ndim == 1 and arr.shape[0] == 3:
        return arr.reshape(1, 3)
    raise ValueError(
        f"CSV 第 {row_idx} 行 {field_name} 数组形状 {list(arr.shape)} 无效: "
        f"期望 (行, 列, 3) 或 (taxel数, 3)，最后一维必须是 3"
    )


def _parse_timestamp(cell: str) -> Optional[float]:
    """解析时间戳单元格为秒级浮点数，无法解析时返回 None

    支持 "YYYY-MM-DD HH:MM:SS.ffffff"（UniTac-NV 格式）、无小数秒变体、纯数字。
    """
    text = cell.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def _parse_float_list(cell: str) -> Optional[List[float]]:
    """解析辅助字段单元格（FT 真值/末端位姿）为浮点列表

    这些字段不参与 Schema 映射，解析失败按缺失（None）处理，不中断加载。
    """
    try:
        obj = ast.literal_eval(cell)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(obj, (list, tuple)):
        return None
    try:
        return [float(v) for v in obj]
    except (TypeError, ValueError):
        return None


def _taxel_positions(shape: Tuple[int, ...]) -> np.ndarray:
    """生成 taxel 归一化 2D 坐标 (N, 2)，x=列方向、y=行方向，均在 [0, 1]

    - (行, 列, 3): 按网格归一化（行/列数为 1 时该轴取 0.5）
    - (taxel数, 3) / (3,): 布局未知，按单行排列 x=i/(N-1)、y=0.5

    返回顺序与 reshape(-1, 3) 的行主序一致。
    """
    if len(shape) == 3:
        rows, cols = int(shape[0]), int(shape[1])
        ys = np.linspace(0.0, 1.0, rows) if rows > 1 else np.array([0.5])
        xs = np.linspace(0.0, 1.0, cols) if cols > 1 else np.array([0.5])
        positions = [[float(xs[c]), float(ys[r])]
                     for r in range(rows) for c in range(cols)]
        return np.asarray(positions, dtype=np.float64)
    # 扁平 taxel 列表（含单 taxel）
    n = int(shape[0])
    if n > 1:
        xs = np.linspace(0.0, 1.0, n)
    else:
        xs = np.array([0.5])
    return np.stack([xs, np.full(n, 0.5)], axis=1)


# =============================================================================
#  XelaUskinAdapter
# =============================================================================

class XelaUskinAdapter(DataAdapterBase):
    """XELA Robotics uSkin 分布式触觉传感器数据适配器（UniTac-NV CSV 格式）

    将 uSkin 的 taxel 力阵列（24 taxel × 3 轴）映射到 TLabel Schema V2 (14维)。

    信号到 Schema 的映射：
      - taxel 力矩阵 → contact（偏差阈值判定）、contact_centroid（力加权质心）
      - 力矩阵按 taxel 求和 × force_scale → force_vector / force_magnitude
        （仅当提供标定系数；Z 为法向轴，见 XELA 官方文档）
      - 位移矩阵 / FT 真值 / 末端位姿 → frame.sensor_specific 原始保留
        （单位/坐标系未文档化，不映射进 Schema 字段）

    Compliance Level: L1（默认）。数据集未文档化力单位，物理量字段保持 None；
    提供 force_scale（raw→N）时升级为 L3。

    参考:
      - XELA Robotics uSkin: https://xelarobotics.com/technology/
      - UniTac-NV 数据集: https://github.com/JiannnH/UniTac-NV
    """

    name = "xela"
    supported_extensions = [".csv"]
    default_compliance_level = "L1"

    # ─── 能力声明 ────────────────────────────────────────────────────────

    def get_capabilities(self) -> Dict[str, bool]:
        """返回 uSkin 数据源在 Schema V2 下的能力声明

        键集与 schema/tlabel-schema.json 的 capabilities 定义一致（13 维，
        不含 compliance_level 元字段）。force_magnitude / force_vector 需要
        使用者提供 force_scale 标定系数才能输出 SI 单位，默认输出为 None。
        """
        return {
            "contact": True,             # taxel 力偏差阈值判定
            "contact_centroid": True,    # taxel 力加权质心（归一化）
            "contact_region": False,     # 平面贴片无 palmar/digital 等区域概念
            "force_magnitude": True,     # 需 force_scale 标定（默认 None）
            "force_vector": True,        # 需 force_scale 标定（默认 None）
            "torque_vector": False,      # 需 taxel SI 力臂坐标，数据集未提供
            "slip_event": False,         # v1 未实现（无经过验证的滑移判据）
            "slip_velocity": False,
            "manipulation_phase": False, # 数据集无阶段标注，不做推断
            "texture_class": False,      # 非视觉传感器
            "object_deformation": False, # 位移单位未文档化且该数据集实测恒为 0
            "temperature": False,        # uSkin 无温度测量
            "confidence": True,
        }

    # ─── 传感器信息 ──────────────────────────────────────────────────────

    def get_sensor_info(self) -> Dict[str, Any]:
        """返回 XELA uSkin 传感器元信息

        sensor_name / sensor_type / adapter_name / adapter_version 为
        schema/tlabel-schema.json 中 sensor 块的必需字段。
        """
        return {
            "sensor_name": "XELA uSkin",
            "sensor_type": "distributed_array",    # JSON Schema 枚举值
            "type": "distributed_taxel_array",     # TLabelData.sensor_type 约定
            "manufacturer": "XELA Robotics",
            "model": "uSkin uSPa 46 (4x6 taxel patch)",
            "adapter_name": "tlabel-xela",
            "adapter_version": __version__,
            "modality": "Hall-effect 3-axis taxel array",
            "description": (
                "uSkin 是 XELA Robotics 的分布式触觉传感器，每个 taxel 同时"
                "测量 X/Y 剪切与 Z 法向 3 轴力。UniTac-NV 数据集使用 uSPa 46 "
                "贴片（4x6=24 taxel，约 100 Hz 录制）。"
            ),
            "axes_per_taxel": 3,
            "taxel_layout_reference": [4, 6],  # UniTac-NV 录制布局；实际布局从数据解析
            "channels": {
                "taxel_force": {
                    "type": "hall_effect_3axis", "count": 24,
                    "unit": "raw (数据集未文档化单位)",
                },
                "taxel_displacement": {
                    "type": "hall_effect_3axis", "count": 24,
                    "unit": "raw (数据集未文档化单位)",
                },
                "ft_ground_truth": {
                    "type": "force_torque", "count": 6,
                    "unit": "N (力分量, UniTac-NV 论文标注)",
                },
                "end_effector_pose": {
                    "type": "pose", "count": 6, "unit": "raw",
                },
            },
            "typical_sample_rate_hz": _XELA_SAMPLE_RATE_DEFAULT,
            "compliance_level": self.default_compliance_level,
            "dataset": "UniTac-NV (https://github.com/JiannnH/UniTac-NV)",
        }

    # ─── Schema 提取 ─────────────────────────────────────────────────────

    def extract_schema(self, raw_frame_data: Dict[str, Any]) -> TLabelSchemaV2:
        """将单帧 uSkin 原始数据转换为 TLabel Schema V2

        参数:
            raw_frame_data: 字典，包含以下键：
                - force: np.ndarray (行,列,3) / (taxel数,3) / (3,) 当前帧原始力矩阵
                - baseline_force: 同形状基线矩阵（可选，缺省为 0）
                - contact_threshold: float 接触判定阈值（taxel 偏差模长最大值，
                  非负；缺省 0.0）
                - force_scale: float 或 None raw→N 标定系数
                - confidence: float 置信度（可选，默认 0.85）

        返回:
            TLabelSchemaV2 — 默认 L1（力字段 None）；提供 force_scale 时为 L3
        """
        force = np.asarray(raw_frame_data["force"], dtype=np.float64)
        if (force.ndim not in (1, 2, 3) or force.shape[-1] != 3
                or force.size == 0):
            raise ValueError(
                f"力矩阵形状 {list(force.shape)} 无效: "
                f"期望 (行, 列, 3) 或 (taxel数, 3)"
            )

        baseline = raw_frame_data.get("baseline_force")
        if baseline is None:
            baseline = np.zeros_like(force)
        else:
            baseline = np.asarray(baseline, dtype=np.float64)
            if baseline.shape != force.shape:
                raise ValueError(
                    f"baseline_force 形状 {list(baseline.shape)} 与 force "
                    f"{list(force.shape)} 不一致"
                )

        threshold = max(float(raw_frame_data.get("contact_threshold", 0.0)), 0.0)
        force_scale = raw_frame_data.get("force_scale")
        confidence = float(raw_frame_data.get("confidence", 0.85))

        flat = (force - baseline).reshape(-1, 3)
        mags = np.linalg.norm(flat, axis=1)
        is_contact = bool(mags.size > 0 and float(mags.max()) > threshold)

        schema = TLabelSchemaV2(
            contact=is_contact,
            slip_event=False,
            confidence=confidence,
            compliance_level=self.default_compliance_level,
        )

        if force_scale is not None:
            # 已标定：填充 SI 力字段并升级 compliance level
            total = flat.sum(axis=0) * float(force_scale)
            schema.force_vector = [round(float(v), 4) for v in total]
            schema.force_magnitude = round(float(np.linalg.norm(total)), 4)
            schema.compliance_level = "L3"

        if not is_contact:
            return schema

        # contact_centroid: taxel 力加权质心（归一化 [0,1]²，建模约定见模块 docstring）
        positions = _taxel_positions(force.shape)
        weights = mags
        w_sum = float(weights.sum())
        if w_sum > 0.0:
            cx = float(np.dot(weights, positions[:, 0]) / w_sum)
            cy = float(np.dot(weights, positions[:, 1]) / w_sum)
            schema.contact_centroid = [round(cx, 4), round(cy, 4)]

        return schema

    # ─── 加载数据文件 ────────────────────────────────────────────────────

    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             contact_threshold: Optional[float] = None,
             baseline_frames: int = 10,
             force_scale: Optional[float] = None,
             **kwargs) -> TLabelData:
        """加载 UniTac-NV 风格的 XELA uSkin CSV，转换为 TLabelData

        参数:
            file_path: CSV 文件路径
            trajectory_id: 保留参数（每文件一个 episode）
            contact_threshold: 接触判定阈值（taxel 原始力偏差模长，非负），
                None 则由基线段统计自动计算（基线标准差×5，下限 0.1）
            baseline_frames: 用于计算零接触基线的前 N 帧（至少 1）
            force_scale: raw→N 标定系数；提供时填充 force_vector /
                force_magnitude 并将 compliance_level 升级为 L3
            **kwargs: 额外参数（忽略）

        返回:
            TLabelData — 统一标注容器

        异常:
            FileNotFoundError: 文件不存在
            ValueError: 格式不支持 / 数据为空 / 矩阵形状错误 / 帧间布局不一致
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"XELA 数据文件不存在: {file_path}")
        ext = path.suffix.lower()
        if ext not in self.supported_extensions:
            raise ValueError(
                f"不支持的文件格式: {ext}\n"
                f"XELA 适配器支持: {self.supported_extensions}"
            )

        raw = self._read_csv(str(path))
        return self._parse(raw, file_path,
                           contact_threshold=contact_threshold,
                           baseline_frames=baseline_frames,
                           force_scale=force_scale, **kwargs)

    # ─── 内部：CSV 读取 ──────────────────────────────────────────────────

    def _read_csv(self, file_path: str) -> Dict[str, Any]:
        """读取并解析 UniTac-NV 风格 XELA CSV

        列识别策略：优先按表头别名匹配；表头未识别时按列位置
        0=time, 1=seq, 2=force, 3=displacement, 4=ft, 5=pose 解析
        （与 UniTac-NV 上游加载器一致）。辅助列（time/seq/ft/pose）缺失或
        解析失败按 None 处理；力矩阵列为必需，缺失或形状错误抛 ValueError。

        返回字典:
            times / seqs / forces / displacements / fts / poses
        """
        rows: List[Tuple[int, List[str]]] = []
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            for line_no, row in enumerate(csv.reader(f), start=1):
                if row and any(c.strip() for c in row):
                    rows.append((line_no, row))

        if not rows:
            raise ValueError(f"XELA CSV 文件为空或无法解析: {file_path}")

        # 首行表头识别
        col_map = self._match_header([c.strip().lower() for c in rows[0][1]])
        if col_map is None:
            # 未识别出表头 → 首行按数据行处理，按列位置解析
            col_map = dict(_POSITIONAL_COLUMNS)
            data_rows = rows
        else:
            data_rows = rows[1:]

        if not data_rows:
            raise ValueError(f"XELA CSV 只有表头，无数据行: {file_path}")

        times: List[Optional[float]] = []
        seqs: List[Optional[int]] = []
        forces: List[np.ndarray] = []
        displacements: List[Optional[np.ndarray]] = []
        fts: List[Optional[List[float]]] = []
        poses: List[Optional[List[float]]] = []

        ref_shape: Optional[Tuple[int, ...]] = None

        for line_no, row in data_rows:
            def _cell(field: str) -> str:
                idx = col_map.get(field)
                if idx is None or idx >= len(row):
                    return ""
                return row[idx].strip()

            # 时间戳（辅助字段，解析失败按缺失处理）
            time_cell = _cell("time")
            times.append(_parse_timestamp(time_cell) if time_cell else None)

            # 序列号
            seq_cell = _cell("seq")
            seq_val: Optional[int] = None
            if seq_cell:
                try:
                    seq_val = int(float(seq_cell))
                except ValueError:
                    seq_val = None
            seqs.append(seq_val)

            # 力矩阵（必需）
            force_cell = _cell("force")
            if not force_cell:
                raise ValueError(
                    f"CSV 第 {line_no} 行力矩阵列为空"
                )
            force_arr = _parse_taxel_matrix(
                force_cell, "力矩阵(sensor_matrices_force)", line_no)
            if ref_shape is None:
                ref_shape = force_arr.shape
            elif force_arr.shape != ref_shape:
                raise ValueError(
                    f"CSV 第 {line_no} 行力矩阵形状 {list(force_arr.shape)} "
                    f"与首帧 {list(ref_shape)} 不一致"
                )
            forces.append(force_arr)

            # 位移矩阵（可选；提供时形状须与力矩阵一致）
            disp_cell = _cell("displacement")
            if disp_cell:
                disp_arr = _parse_taxel_matrix(
                    disp_cell, "位移矩阵(sensor_matrices_displacement)", line_no)
                if disp_arr.shape != ref_shape:
                    raise ValueError(
                        f"CSV 第 {line_no} 行位移矩阵形状 "
                        f"{list(disp_arr.shape)} 与力矩阵 "
                        f"{list(ref_shape)} 不一致"
                    )
                displacements.append(disp_arr)
            else:
                displacements.append(None)

            # FT 真值 / 末端位姿（辅助字段）
            ft_cell = _cell("ft")
            fts.append(_parse_float_list(ft_cell) if ft_cell else None)
            pose_cell = _cell("pose")
            poses.append(_parse_float_list(pose_cell) if pose_cell else None)

        return {
            "times": times,
            "seqs": seqs,
            "forces": forces,
            "displacements": displacements,
            "fts": fts,
            "poses": poses,
        }

    @staticmethod
    def _match_header(header: List[str]) -> Optional[Dict[str, int]]:
        """按别名识别表头，返回字段→列号映射

        识别到任意别名但缺少力矩阵列时，抛 ValueError（明确的表头却缺必需列）；
        完全无法识别时返回 None（调用方按列位置解析）。
        """
        col_map: Dict[str, int] = {}
        for idx, name in enumerate(header):
            for field, aliases in _HEADER_ALIASES.items():
                if name in aliases and field not in col_map:
                    col_map[field] = idx
                    break
        if "force" not in col_map:
            if col_map:
                raise ValueError(
                    f"CSV 表头未包含力矩阵列 (sensor_matrices_force / "
                    f"force_matrix / forces): {header}"
                )
            return None
        return col_map

    # ─── 内部：构建 TLabelData ───────────────────────────────────────────

    def _parse(self, raw: Dict[str, Any],
               file_path: str,
               contact_threshold: Optional[float] = None,
               baseline_frames: int = 10,
               force_scale: Optional[float] = None,
               **kwargs) -> TLabelData:
        """解析已加载的 XELA 原始数据，构建 TLabelData"""
        forces = raw["forces"]
        n_frames = len(forces)
        if n_frames == 0:
            raise ValueError("XELA 数据文件包含 0 帧数据")

        layout = forces[0].shape            # (行, 列, 3) 或 (taxel数, 3)
        if len(layout) == 3:
            taxel_layout = [int(layout[0]), int(layout[1])]
        else:
            taxel_layout = [int(layout[0])]
        num_taxels = int(np.prod(layout[:-1]))

        # 基线（前 N 帧均值；与 UniTac-NV 的标准化做法一致，用于接触判定）
        n_bl = min(max(int(baseline_frames), 1), n_frames)
        baseline_force = np.mean(np.stack(forces[:n_bl]), axis=0)

        # 每帧 taxel 偏差的最大模长（原始单位）
        stacked = np.stack(forces)
        devs = np.linalg.norm(
            stacked.reshape(n_frames, -1, 3) - baseline_force.reshape(-1, 3),
            axis=2)
        frame_max_dev = devs.max(axis=1)

        # 接触阈值
        if contact_threshold is not None:
            threshold = max(float(contact_threshold), 0.0)
        else:
            bl_std = float(np.std(frame_max_dev[:n_bl])) if n_bl >= 3 else 0.0
            threshold = max(bl_std * 5.0, 0.1)

        contacts = [bool(m > threshold) for m in frame_max_dev]

        # 时间戳：相对首帧的秒数；不可解析的帧按索引/采样率补齐
        times = raw.get("times") or [None] * n_frames
        valid_times = [t for t in times if t is not None]
        if len(valid_times) >= 2:
            med_diff = float(np.median(np.diff(np.sort(valid_times))))
            sr = float(1.0 / med_diff) if med_diff > 0 else _XELA_SAMPLE_RATE_DEFAULT
        else:
            sr = _XELA_SAMPLE_RATE_DEFAULT
        dt = 1.0 / sr if sr > 0 else 0.01
        t0 = min(valid_times) if valid_times else 0.0
        timestamps = [
            round(t - t0, 4) if t is not None else round(i * dt, 4)
            for i, t in enumerate(times)
        ]

        # 逐帧构建 TLabelFrame
        tlabel_frames: List[TLabelFrame] = []
        for i in range(n_frames):
            raw_frame = {
                "force": forces[i],
                "baseline_force": baseline_force,
                "contact_threshold": threshold,
                "force_scale": force_scale,
                "confidence": 0.85,
            }
            schema = self.extract_schema(raw_frame)

            # 传感器特有数据（保留原始读数，单位见 sensor_info.units_note）
            sensor_specific: Dict[str, Any] = {
                "taxel_force": [
                    [round(float(v), 4) for v in taxel]
                    for taxel in forces[i].reshape(-1, 3)
                ],
            }
            if raw["seqs"][i] is not None:
                sensor_specific["seq"] = raw["seqs"][i]
            if raw["displacements"][i] is not None:
                sensor_specific["taxel_displacement"] = [
                    [round(float(v), 4) for v in taxel]
                    for taxel in raw["displacements"][i].reshape(-1, 3)
                ]
            if raw["fts"][i] is not None:
                sensor_specific["ft_ground_truth"] = [
                    round(float(v), 4) for v in raw["fts"][i]
                ]
            if raw["poses"][i] is not None:
                sensor_specific["end_effector_pose"] = [
                    round(float(v), 4) for v in raw["poses"][i]
                ]

            tlabel_frames.append(TLabelFrame(
                frame_idx=i,
                timestamp_s=timestamps[i],
                schema_v2=schema,
                confidence=schema.confidence,
                sensor_specific=sensor_specific,
            ))

        contact_count = sum(1 for c in contacts if c)

        sensor_info = dict(self.get_sensor_info())
        sensor_info.update({
            "sample_rate_hz": round(sr, 2),
            "taxel_layout": taxel_layout,
            "num_taxels": num_taxels,
            "baseline_frames_used": n_bl,
            "contact_threshold": round(threshold, 4),
            "units_note": (
                "UniTac-NV 数据集未文档化 uSkin 力/位移单位；"
                "taxel_force / taxel_displacement 为原始读数。"
                "Schema 力字段默认 None，提供 force_scale 后按 N 输出。"
            ),
        })
        if force_scale is not None:
            sensor_info["force_scale"] = float(force_scale)

        episode_info = {
            "source": "xela_uskin_univtac_nv",
            "file": Path(file_path).name,
            "total_frames": n_frames,
            "contact_frames": contact_count,
            "sample_rate_hz": round(sr, 2),
            "duration_s": round(float(timestamps[-1]), 4),
            "taxel_layout": taxel_layout,
            "num_taxels": num_taxels,
        }

        calibration_params = (
            {"force_scale": float(force_scale)}
            if force_scale is not None else None
        )

        return TLabelData(
            frames=tlabel_frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
            sensor_id="xela_uskin_0",
            calibration_params=calibration_params,
        )
