"""
SynTouch BioTac 数据适配器 — 将BioTac多通道数据转换为TLabelData (Schema V2)

BioTac 是 SynTouch 公司的仿生触觉传感器，通过仿生指尖结构同时测量
四种物理信号：
  - impedance (阻抗): 19电极阻抗阵列，反映接触区域与形变
  - static pressure (静态压力): 流体静压，反映法向力大小
  - dynamic pressure (AC动态压力): 高频振动，反映微滑移与纹理
  - temperature (温度): 热敏电阻，反映热传导

参考: https://www.syntouchinc.com/biotac
"""

import csv
import math
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from tlabel.adapters.base import DataAdapterBase
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2

# ─── Lazy imports for optional dependencies ──────────────────────────────

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
#  BioTac 信号物理常量
# =============================================================================

# BioTac 典型参数（参考官方数据手册）
_BIOTAC_NUM_ELECTRODES = 19       # 阻抗电极数
_BIOTAC_SAMPLE_RATE_DEFAULT = 100.0  # 默认采样率 Hz
_BIOTAC_PRESSURE_SCALE = 1.0      # 压力到力的缩放（单位：任意比例，L2精度）


# =============================================================================
#  内部工具函数
# =============================================================================

def _infer_phases(contacts: List[bool], slips: List[bool]) -> List[str]:
    """
    从接触/滑移序列推断操作阶段（简化版）

    阶段枚举: pre_contact / approach / grasp / lift / hold / place
    """
    n = len(contacts)
    phases = ["pre_contact"] * n

    # 寻找第一个接触帧和最后一个接触帧
    first_contact = None
    last_contact = None
    for i in range(n):
        if contacts[i]:
            if first_contact is None:
                first_contact = i
            last_contact = i

    if first_contact is None:
        return phases  # 全程无接触

    # approach: 接触前1帧到首次接触后几帧（接近阶段）
    approach_end = min(first_contact + max(3, n // 20), n - 1)

    # 统计滑移帧
    has_slip = any(slips)

    for i in range(n):
        if i < first_contact:
            phases[i] = "pre_contact"
        elif i <= approach_end:
            phases[i] = "approach"
        elif i > last_contact:
            phases[i] = "place"
        else:
            # 稳定接触阶段
            if has_slip and slips[i]:
                phases[i] = "grasp"
            else:
                phases[i] = "hold"

    # lift: 从接触稳定到脱离之间的过渡（如果接触时间够长）
    contact_duration = last_contact - first_contact
    if contact_duration > 10:
        lift_start = last_contact - min(5, contact_duration // 5)
        for i in range(lift_start, last_contact + 1):
            if phases[i] == "hold":
                phases[i] = "lift"

    return phases


def _detect_slip_from_pac(pac_signal: np.ndarray,
                          threshold_std_ratio: float = 2.0) -> List[bool]:
    """
    从 PAC (dynamic pressure) 信号检测滑移事件

    原理：微滑移引起高频振动 → PAC 信号能量突变

    参数:
        pac_signal: 动态压力信号 (1D array)
        threshold_std_ratio: 阈值 = 均值 + ratio * 标准差

    返回:
        每帧是否滑移的 bool 列表
    """
    n = len(pac_signal)
    slips = [False] * n

    if n < 3:
        return slips

    pac_abs = np.abs(pac_signal)
    mean_val = float(np.mean(pac_abs))
    std_val = float(np.std(pac_abs))
    threshold = mean_val + threshold_std_ratio * std_val

    # 使用滑动窗口能量
    window = max(3, n // 50)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        energy = float(np.mean(pac_abs[start:end]))
        slips[i] = energy > threshold and energy > 1e-6

    return slips


def _pressure_to_force(pdc: float, baseline: float) -> float:
    """
    将静态压力 (PDC) 转换为力的估计值

    参数:
        pdc: 当前静态压力值
        baseline: 零接触基线压力

    返回:
        估计的法向力大小（无量纲，L2精度）
    """
    diff = abs(pdc - baseline)
    return round(float(diff) * _BIOTAC_PRESSURE_SCALE, 4)


def _electrodes_to_contact_region(electrodes: np.ndarray,
                                  baseline: np.ndarray) -> str:
    """
    从19电极阻抗模式推断接触区域

    BioTac电极布局大致分为指尖顶部(distal)、指腹(palmar)、
    侧面(lateral)等区域。根据阻抗变化的空间分布粗略分类。

    简化处理：根据电极索引分组，取变化最大的组名。
    """
    if len(electrodes) != _BIOTAC_NUM_ELECTRODES:
        return "other"

    change = np.abs(electrodes - baseline)
    total_change = float(np.sum(change))

    if total_change < 1e-6:
        return "other"

    # 简化的电极区域划分（近似 BioTac 电极布局）
    # 电极 0-5: distal (指尖顶部)
    # 电极 6-12: palmar (指腹中央)
    # 电极 13-15: lateral (侧面)
    # 电极 16-18: proximal (近端)
    regions = {
        "distal": float(np.sum(change[0:6])),
        "palmar": float(np.sum(change[6:13])),
        "lateral": float(np.sum(change[13:16])),
        "proximal": float(np.sum(change[16:19])),
    }

    max_region = max(regions, key=regions.get)
    # 如果最显著区域占比超过总变化的 30%，认为有效
    if regions[max_region] / total_change > 0.3:
        return max_region

    return "other"


def _electrodes_to_centroid(electrodes: np.ndarray,
                            baseline: np.ndarray) -> Optional[List[float]]:
    """
    从19电极阻抗变化计算接触质心 (归一化坐标 [0,1] × [0,1])

    简化模型：将电极按近似位置分布在 2D 平面上，加权计算质心。
    """
    if len(electrodes) != _BIOTAC_NUM_ELECTRODES:
        return None

    change = np.abs(electrodes - baseline)
    total = float(np.sum(change))

    if total < 1e-6:
        return None

    # 19个电极的近似 2D 坐标（归一化到 [0,1]）
    # 这是 BioTac 指尖展开的简化模型
    electrode_positions = np.array([
        # distal row (0-5)
        [0.50, 0.95], [0.30, 0.90], [0.70, 0.90],
        [0.20, 0.80], [0.50, 0.82], [0.80, 0.80],
        # palmar row (6-12)
        [0.25, 0.65], [0.40, 0.62], [0.55, 0.62],
        [0.70, 0.65], [0.30, 0.48], [0.50, 0.45],
        [0.70, 0.48],
        # lateral/proximal (13-18)
        [0.15, 0.35], [0.50, 0.30], [0.85, 0.35],
        [0.25, 0.15], [0.50, 0.12], [0.75, 0.15],
    ])

    weights = change / total
    centroid_x = float(np.sum(weights * electrode_positions[:, 0]))
    centroid_y = float(np.sum(weights * electrode_positions[:, 1]))

    return [round(centroid_x, 4), round(centroid_y, 4)]


# =============================================================================
#  数据加载：多格式支持
# =============================================================================

def _load_hdf5(file_path: str) -> Dict[str, np.ndarray]:
    """
    从 HDF5 文件加载 BioTac 数据

    预期结构（兼容常见 BioTac 记录格式）:
      /impedance  or  /electrodes  → N×19 阻抗数据
      /pdc        or  /static_pressure  → N×1 静态压力
      /pac        or  /dynamic_pressure → N×1 动态压力 (AC)
      /temperature or /tac          → N×1 温度
      /timestamp  (可选)            → N×1 时间戳

    若数据在子组中（如 /biotac/...），自动递归发现。
    """
    if not HAS_H5PY:
        raise ImportError("加载 HDF5 格式的 BioTac 数据需要 h5py: pip install h5py")

    data = {}
    with h5py.File(file_path, 'r') as f:
        # 递归查找已知数据集
        def _find_datasets(group, prefix=""):
            for key in group.keys():
                item = group[key]
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(item, h5py.Dataset):
                    name_lower = key.lower()
                    # 阻抗 / 电极
                    if name_lower in ("impedance", "electrodes", "imp"):
                        data["impedance"] = np.array(item[()], dtype=np.float64)
                    # 静态压力
                    elif name_lower in ("pdc", "static_pressure", "pressure", "static_p"):
                        arr = np.array(item[()], dtype=np.float64)
                        data["pdc"] = arr.flatten()
                    # 动态压力
                    elif name_lower in ("pac", "dynamic_pressure", "ac_pressure", "dynamic_p"):
                        arr = np.array(item[()], dtype=np.float64)
                        data["pac"] = arr.flatten()
                    # 温度
                    elif name_lower in ("temperature", "tac", "temp", "thermistor"):
                        arr = np.array(item[()], dtype=np.float64)
                        data["temperature"] = arr.flatten()
                    # 时间戳
                    elif name_lower in ("timestamp", "timestamps", "time", "t"):
                        arr = np.array(item[()], dtype=np.float64)
                        data["timestamp"] = arr.flatten()
                elif isinstance(item, h5py.Group):
                    _find_datasets(item, path)

        _find_datasets(f)

    # 校验必选通道
    if "impedance" not in data:
        raise ValueError(
            f"HDF5 文件中未找到 BioTac 阻抗数据 (impedance/electrodes)。\n"
            f"可用数据集: {list(data.keys())}"
        )

    return data


def _load_csv(file_path: str) -> Dict[str, np.ndarray]:
    """
    从 CSV 文件加载 BioTac 数据

    预期列（兼容多种命名）:
      impedance 列: electrode_0 ... electrode_18 或 imp_0 ... imp_18 或 19列连续阻抗
      pressure 列: pdc / static_pressure / pressure
      dynamic 列: pac / dynamic_pressure / ac
      temperature 列: temperature / temp / tac
      timestamp 列: timestamp / time / t (可选)
    """
    rows = []
    headers = None

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if headers is None:
                # 判断是否是表头（首行含非数字）
                try:
                    [float(x) for x in row]
                    # 全是数字，不是表头，作为第一行数据
                    headers = None
                    rows.append([float(x) for x in row])
                except ValueError:
                    headers = [h.strip().lower() for h in row]
            else:
                rows.append([float(x) for x in row])

    if not rows:
        raise ValueError(f"CSV 文件为空或无法解析: {file_path}")

    data_arr = np.array(rows, dtype=np.float64)
    n_rows, n_cols = data_arr.shape

    data = {}

    if headers is not None:
        # 有表头，按列名匹配
        electrode_cols = []
        for i, h in enumerate(headers):
            if h.startswith("electrode_") or h.startswith("imp_") or h.startswith("e"):
                try:
                    idx = int(h.split("_")[-1])
                    electrode_cols.append((idx, i))
                except ValueError:
                    pass
            elif h in ("pdc", "static_pressure", "pressure", "static_p"):
                data["pdc"] = data_arr[:, i].copy()
            elif h in ("pac", "dynamic_pressure", "ac_pressure", "ac", "dynamic_p"):
                data["pac"] = data_arr[:, i].copy()
            elif h in ("temperature", "temp", "tac", "thermistor"):
                data["temperature"] = data_arr[:, i].copy()
            elif h in ("timestamp", "time", "t"):
                data["timestamp"] = data_arr[:, i].copy()

        if electrode_cols:
            electrode_cols.sort()
            indices = [col for _, col in electrode_cols]
            data["impedance"] = data_arr[:, indices].astype(np.float64)
    else:
        # 无表头，按列数推断
        # 常见模式: 19电极 + pdc + pac + temp (+ timestamp) = 22 或 23 列
        if n_cols >= 22:
            # BioTac 标准通道顺序：19电极 + pac + pdc + tac (+ tdc)
            # 前19列为电极阻抗，第20列pac(动态压力)，第21列pdc(静态压力)，
            # 第22列tac(温度)，第23列tdc(可选，热传导直流分量)
            data["impedance"] = data_arr[:, :19].astype(np.float64)
            data["pac"] = data_arr[:, 19].copy()
            data["pdc"] = data_arr[:, 20].copy()
            data["temperature"] = data_arr[:, 21].copy()
            if n_cols >= 23:
                data["tdc"] = data_arr[:, 22].copy()
        elif n_cols >= 19:
            # 假设前19列是电极
            data["impedance"] = data_arr[:, :19].astype(np.float64)
            if n_cols >= 20:
                data["pdc"] = data_arr[:, 19].copy()
            if n_cols >= 21:
                data["pac"] = data_arr[:, 20].copy()
            if n_cols >= 22:
                data["temperature"] = data_arr[:, 21].copy()
        else:
            raise ValueError(
                f"CSV 列数不足 ({n_cols})，BioTac 数据至少需要19个阻抗电极列。"
            )

    if "impedance" not in data:
        raise ValueError(
            f"CSV 文件中无法识别 BioTac 阻抗数据列。列数: {n_cols}"
        )

    return data


def _load_mat(file_path: str) -> Dict[str, np.ndarray]:
    """
    从 MATLAB .mat 文件加载 BioTac 数据

    支持 scipy.io.loadmat 可读的 MATLAB v4/v6/v7/v7.0 格式。
    字段命名规则与 HDF5 相同。
    """
    if not HAS_SCIPY:
        raise ImportError("加载 MATLAB 格式的 BioTac 数据需要 scipy: pip install scipy")

    mat = loadmat(file_path)
    data = {}

    # 遍历 mat 文件中的变量
    for key, value in mat.items():
        # 跳过 scipy 内部变量
        if key.startswith("__"):
            continue

        name_lower = key.lower()
        arr = np.array(value, dtype=np.float64)

        if name_lower in ("impedance", "electrodes", "impedances", "imp"):
            data["impedance"] = arr.reshape(arr.shape[0], -1) if arr.ndim >= 1 else arr
        elif name_lower in ("pdc", "static_pressure", "pressure", "static_p"):
            data["pdc"] = arr.flatten()
        elif name_lower in ("pac", "dynamic_pressure", "ac_pressure", "dynamic_p"):
            data["pac"] = arr.flatten()
        elif name_lower in ("temperature", "tac", "temp", "thermistor"):
            data["temperature"] = arr.flatten()
        elif name_lower in ("timestamp", "timestamps", "time", "t"):
            data["timestamp"] = arr.flatten()

    if "impedance" not in data:
        raise ValueError(
            f"MAT 文件中未找到 BioTac 阻抗数据。可用变量: "
            f"{[k for k in mat.keys() if not k.startswith('__')]}"
        )

    return data


# =============================================================================
#  SynTouchBioTacAdapter
# =============================================================================

class SynTouchBioTacAdapter(DataAdapterBase):
    """
    SynTouch BioTac 仿生触觉传感器数据适配器

    将 BioTac 的 4 通道数据（阻抗阵列、静态压力、动态压力、温度）
    映射到 TLabel Schema V2 (14维)。

    信号到 Schema 的映射：
      - impedance (19电极) → contact_region, contact_centroid, object_deformation
      - static pressure (PDC) → force_magnitude, contact (阈值判定)
      - dynamic pressure (PAC) → slip_event, texture 相关特征
      - temperature → temperature (Schema V2 原生支持)

    Compliance Level: L2 (提供 force_magnitude 等 L2 必选字段)
    """

    name = "syntouch"
    supported_extensions = [".h5", ".csv", ".mat"]
    default_compliance_level = "L2"

    # ─── 能力声明 ────────────────────────────────────────────────────────

    def get_capabilities(self) -> Dict[str, bool]:
        """返回 BioTac 传感器在 Schema V2 下的能力声明"""
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": True,
            "force_magnitude": True,
            "force_vector": False,
            "torque_vector": False,
            "slip_event": True,
            "slip_velocity": False,
            "manipulation_phase": True,
            "texture_class": False,
            "object_deformation": True,
            "temperature": True,
            "confidence": True,
            # 注意：compliance_level 是帧级元字段，不属于 capabilities 能力键
            # （规范 S-7.4-2 / C-15），v0.23.3 起不再出现在此处。
        }

    # ─── 传感器信息 ──────────────────────────────────────────────────────

    def get_sensor_info(self) -> Dict[str, Any]:
        """返回 BioTac 传感器元信息"""
        return {
            "type": "multi_modal_tactile",
            "manufacturer": "SynTouch",
            "model": "BioTac",
            "modality": "impedance + piezoresistive + thermistor",
            "description": (
                "BioTac是一种仿生触觉传感器，集成19电极阻抗阵列、"
                "流体静压传感器、高频振动传感器和热敏电阻，"
                "可同时测量接触形状、力、振动和温度。"
            ),
            "channels": {
                "impedance": {
                    "type": "electrode_array",
                    "count": _BIOTAC_NUM_ELECTRODES,
                    "unit": "kΩ (归一化)",
                },
                "static_pressure": {
                    "type": "piezoresistive",
                    "count": 1,
                    "unit": "digitized counts",
                },
                "dynamic_pressure": {
                    "type": "ac_pressure",
                    "count": 1,
                    "unit": "digitized counts (AC)",
                },
                "temperature": {
                    "type": "thermistor",
                    "count": 1,
                    "unit": "°C",
                },
            },
            "typical_sample_rate_hz": _BIOTAC_SAMPLE_RATE_DEFAULT,
            "compliance_level": self.default_compliance_level,
        }

    # ─── Schema 提取 ─────────────────────────────────────────────────────

    def extract_schema(self, raw_frame_data: Dict[str, Any]) -> TLabelSchemaV2:
        """
        将单帧 BioTac 原始数据转换为 TLabel Schema V2

        参数:
            raw_frame_data: 字典，包含以下键：
                - electrodes: np.ndarray (19,) 阻抗值
                - pdc: float 静态压力
                - pac: float 动态压力
                - temperature: float (可选) 温度
                - baseline_electrodes: np.ndarray (19,) 基线阻抗
                - baseline_pdc: float 基线静态压力
                - is_contact: bool 是否接触
                - is_slip: bool 是否滑移
                - phase: str 操作阶段（可选）
                - confidence: float 置信度（可选）

        返回:
            TLabelSchemaV2 对象
        """
        electrodes = raw_frame_data.get("electrodes")
        pdc = raw_frame_data.get("pdc", 0.0)
        pac = raw_frame_data.get("pac", 0.0)
        temperature = raw_frame_data.get("temperature")
        baseline_electrodes = raw_frame_data.get("baseline_electrodes")
        baseline_pdc = raw_frame_data.get("baseline_pdc", 0.0)
        is_contact = raw_frame_data.get("is_contact", False)
        is_slip = raw_frame_data.get("is_slip", False)
        phase = raw_frame_data.get("phase")
        confidence = raw_frame_data.get("confidence", 0.85)

        schema = TLabelSchemaV2(
            contact=bool(is_contact),
            slip_event=bool(is_slip),
            confidence=float(confidence),
            compliance_level=self.default_compliance_level,
        )

        if not is_contact:
            # 无接触时，除基础字段外其余保持 None
            if temperature is not None:
                schema.temperature = round(float(temperature), 2)
            return schema

        # 1. contact_centroid（从阻抗电极分布计算）
        if electrodes is not None and baseline_electrodes is not None:
            centroid = _electrodes_to_centroid(electrodes, baseline_electrodes)
            if centroid is not None:
                schema.contact_centroid = centroid

        # 2. contact_region
        if electrodes is not None and baseline_electrodes is not None:
            schema.contact_region = _electrodes_to_contact_region(
                electrodes, baseline_electrodes
            )

        # 3. force_magnitude（从静态压力偏移估计）
        force_mag = _pressure_to_force(pdc, baseline_pdc)
        schema.force_magnitude = force_mag if force_mag > 0 else 0.0

        # 4. object_deformation（从阻抗总变化估计）
        if electrodes is not None and baseline_electrodes is not None:
            deformation = float(np.sum(np.abs(electrodes - baseline_electrodes)))
            schema.object_deformation = round(deformation, 4)

        # 5. manipulation_phase
        if phase is not None:
            schema.manipulation_phase = phase

        # 6. temperature
        if temperature is not None:
            schema.temperature = round(float(temperature), 2)

        # 7. slip_velocity: BioTac 单 PAC 信号不足以估计速度向量，保持 None
        #    （需要阵列信号或更高采样率才能估计方向）

        return schema

    # ─── 加载数据文件 ────────────────────────────────────────────────────

    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             sample_rate: Optional[float] = None,
             contact_threshold: Optional[float] = None,
             baseline_frames: int = 10,
             **kwargs) -> TLabelData:
        """
        加载 BioTac 数据文件，转换为 TLabelData

        支持的格式：.h5 / .hdf5, .csv, .mat

        参数:
            file_path: 数据文件路径
            trajectory_id: 保留参数（BioTac每文件一个episode）
            sample_rate: 采样率 Hz，None 则自动检测或使用默认值
            contact_threshold: 接触判定阈值（压力偏移比例），None 则自动计算
            baseline_frames: 用于计算基线的前 N 帧
            **kwargs: 额外参数

        返回:
            TLabelData — 统一标注容器

        异常:
            ValueError: 文件格式不支持或数据格式错误
            ImportError: 缺少必要依赖 (h5py / scipy)
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if not path.exists():
            raise FileNotFoundError(f"BioTac 数据文件不存在: {file_path}")

        # 按格式分发
        if ext in (".h5", ".hdf5"):
            raw = _load_hdf5(str(path))
        elif ext == ".csv":
            raw = _load_csv(str(path))
        elif ext == ".mat":
            raw = _load_mat(str(path))
        else:
            raise ValueError(
                f"不支持的文件格式: {ext}\n"
                f"BioTac 适配器支持: {self.supported_extensions}"
            )

        return self._parse(raw, file_path, sample_rate=sample_rate,
                           contact_threshold=contact_threshold,
                           baseline_frames=baseline_frames, **kwargs)

    # ─── 内部解析 ────────────────────────────────────────────────────────

    def _parse(self, raw: Dict[str, np.ndarray],
               file_path: str,
               sample_rate: Optional[float] = None,
               contact_threshold: Optional[float] = None,
               baseline_frames: int = 10,
               **kwargs) -> TLabelData:
        """
        解析已加载的 BioTac 原始数据，构建 TLabelData
        """
        impedance = raw["impedance"]
        n_frames = impedance.shape[0]

        if n_frames == 0:
            raise ValueError("BioTac 数据文件包含 0 帧数据")

        pdc = raw.get("pdc", np.zeros(n_frames, dtype=np.float64))
        pac = raw.get("pac", np.zeros(n_frames, dtype=np.float64))
        temperature = raw.get("temperature")
        timestamp = raw.get("timestamp")

        # 采样率
        if sample_rate is not None:
            sr = float(sample_rate)
        elif timestamp is not None and len(timestamp) > 1:
            # 从时间戳计算
            dt = np.mean(np.diff(timestamp[:min(100, len(timestamp))]))
            sr = 1.0 / dt if dt > 0 else _BIOTAC_SAMPLE_RATE_DEFAULT
        else:
            sr = _BIOTAC_SAMPLE_RATE_DEFAULT

        dt = 1.0 / sr if sr > 0 else 0.01

        # 基线计算（前 N 帧的均值）
        n_bl = min(baseline_frames, n_frames)
        baseline_electrodes = np.mean(impedance[:n_bl], axis=0)
        baseline_pdc = float(np.mean(pdc[:n_bl]))

        # 接触判定（基于静态压力偏离基线的程度）
        pdc_deviation = np.abs(pdc - baseline_pdc)
        if contact_threshold is not None:
            threshold = float(contact_threshold)
        else:
            # 自动阈值：基线标准差 × 5 + 极小值保护
            pdc_std = float(np.std(pdc[:n_bl])) if n_bl >= 3 else 0.0
            threshold = max(pdc_std * 5.0, 0.1)

        contacts = [bool(d > threshold) for d in pdc_deviation]

        # 滑移检测（基于动态压力能量）
        slips = _detect_slip_from_pac(pac)

        # 操作阶段推断
        phases = _infer_phases(contacts, slips)

        # 逐帧构建 TLabelFrame
        tlabel_frames = []
        for i in range(n_frames):
            electrodes_i = impedance[i]
            pdc_i = float(pdc[i])
            pac_i = float(pac[i])
            temp_i = float(temperature[i]) if temperature is not None else None
            ts_i = float(timestamp[i]) if timestamp is not None else round(i * dt, 4)

            raw_frame = {
                "electrodes": electrodes_i,
                "pdc": pdc_i,
                "pac": pac_i,
                "temperature": temp_i,
                "baseline_electrodes": baseline_electrodes,
                "baseline_pdc": baseline_pdc,
                "is_contact": contacts[i],
                "is_slip": slips[i],
                "phase": phases[i],
                "confidence": 0.85,
            }

            schema = self.extract_schema(raw_frame)

            # 传感器特有数据（保留原始信号）
            sensor_specific = {
                "impedance": [round(float(x), 4) for x in electrodes_i],
                "pdc": round(pdc_i, 4),
                "pac": round(pac_i, 4),
            }
            if temp_i is not None:
                sensor_specific["temperature"] = round(temp_i, 2)

            frame = TLabelFrame(
                frame_idx=i,
                timestamp_s=ts_i,
                schema_v2=schema,
                manipulation_phase=phases[i],
                confidence=schema.confidence,
                sensor_specific=sensor_specific,
            )
            tlabel_frames.append(frame)

        # 统计
        contact_count = sum(1 for c in contacts if c)
        slip_count = sum(1 for s in slips if s)

        sensor_info = {
            "type": "multi_modal_tactile",
            "manufacturer": "SynTouch",
            "model": "BioTac",
            "modality": "impedance + piezoresistive + thermistor",
            "sample_rate_hz": round(sr, 1),
            "num_electrodes": _BIOTAC_NUM_ELECTRODES,
            "channels": [
                "impedance (19 electrodes)",
                "static_pressure (PDC)",
                "dynamic_pressure (PAC)",
                "temperature",
            ],
            "baseline_frames_used": n_bl,
            "contact_threshold": threshold,
        }

        episode_info = {
            "source": "syntouch_biotac",
            "file": Path(file_path).name,
            "total_frames": n_frames,
            "contact_frames": contact_count,
            "slip_frames": slip_count,
            "sample_rate_hz": round(sr, 1),
            "duration_s": round(n_frames * dt, 4),
            "baseline_pressure": round(baseline_pdc, 4),
            "temperature_channel": temperature is not None,
        }

        return TLabelData(
            frames=tlabel_frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
            sensor_id="syntouch_biotac_0",
        )
