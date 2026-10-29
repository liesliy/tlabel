"""
PaXini GEN3 实时传感器适配器 — 通过PaXini SDK读取GEN3系列传感器数据

支持型号:
  GEN3-1 (单区指尖) / GEN3-2 (双区指尖+掌心) / GEN3-5 (五指) / GEN3-5L (左手版)
  每个taxel: 8×8 (指尖, 14×14mm, 1.75mm间距) 或 16×12 (掌心, 48×36mm, 3mm间距)
  压力范围: 0-600 kPa
  采样率: USB-C 50-500Hz, BLE 50-200Hz

数据映射:
  SDK TactileFrame 字段          →  TLabel v2 字段
  ──────────────────────────────    ────────────────────────────
  pressure_map (H,W float32 kPa) →  deformation_magnitude (归一化均值)
                                   normal_field_magnitude (同deformation)
                                   normal_field_variance (空间方差)
  contact_mask (H,W bool)        →  contact (any True → 1.0)
                                   contact_area (激活比例)
  total_force_n (float)          →  force_magnitude, force_peak (归一化)
  contact_centroid (row,col)     →  centroid_x (归一化到0-1)
  in_contact (bool)              →  contact 辅助确认
  timestamp_ns (int)             →  timestamp_s
  seq (int)                      →  frame_idx 校验

  时序计算:
  帧间centroid偏移              →  slip_event (超阈值 → 1.0)
  帧间force变化率               →  delta_force_normal
  帧间deformation差分/dt        →  temporal_deformation_rate
  帧间contact状态变化            →  contact_transition

依赖: pip install paxini-sdk  (或 pip install "paxini-sdk[viz]")
数据集(离线)适配器见 paxini_dataset.py

v0.16.0 架构升级：
- 继承SensorAdapterBase，实现connect/disconnect/stream_frames实时接口
- 保留load()向后兼容（用于处理录制好的.paxini文件）
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator

from tlabel.adapters.base import SensorAdapterBase
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2

try:
    import paxini
    HAS_PAXINI_SDK = True
except ImportError:
    HAS_PAXINI_SDK = False


# ─── 传感器配置 ───────────────────────────────────────────────

# GEN3系列已知传感器布局配置
GEN3_LAYOUTS = {
    "gen3_1": {
        "description": "GEN3-1 单区指尖",
        "regions": {"fingertip": {"shape": (8, 8), "area_mm2": 14 * 14}},
    },
    "gen3_2": {
        "description": "GEN3-2 指尖+掌心",
        "regions": {
            "fingertip": {"shape": (8, 8), "area_mm2": 14 * 14},
            "palm": {"shape": (16, 12), "area_mm2": 48 * 36},
        },
    },
    "gen3_5": {
        "description": "GEN3-5 五指全覆盖",
        "regions": {
            "thumb": {"shape": (8, 8), "area_mm2": 14 * 14},
            "index": {"shape": (8, 8), "area_mm2": 14 * 14},
            "middle": {"shape": (8, 8), "area_mm2": 14 * 14},
            "ring": {"shape": (8, 8), "area_mm2": 14 * 14},
            "pinky": {"shape": (8, 8), "area_mm2": 14 * 14},
        },
    },
}

# 压力归一化: SDK输出0-600kPa → TLabel v2的0-1
MAX_PRESSURE_KPA = 600.0

# 滑移检测阈值
SLIP_CENTROID_SHIFT_THRESHOLD = 0.15   # 质心归一化偏移
SLIP_FORCE_RATE_THRESHOLD = 0.25       # 力变化率


class PaxiniGen3Adapter(SensorAdapterBase):
    """PaXini GEN3 实时传感器 → TLabelData

    支持两种工作模式:
      1. 实时流采集: 使用connect() + stream_frames() + disconnect()
      2. 文件加载: 使用load()加载.paxini录制文件（向后兼容）

    Compliance Level: L2（只有 total_force_n 法向力标量，无剪切力）

    依赖: pip install paxini-sdk

    用法示例:
        # 实时采集模式
        adapter = PaxiniGen3Adapter()
        adapter.connect("auto")
        for frame in adapter.stream_frames(num_frames=100):
            process(frame)
        adapter.disconnect()

        # 录制文件处理模式（向后兼容）
        data = adapter.load("recording.paxini")
    """

    default_compliance_level: str = "L2"

    def __init__(self):
        self._sensor = None
        self._connected = False
        self._sensor_index = 0
        self._sample_rate = 100

    @property
    def name(self) -> str:
        return "paxini_gen3"

    @property
    def supported_extensions(self) -> List[str]:
        return [".paxini"]

    # ─── Schema V2 提取 ───────────────────────────────────────

    def extract_schema(self, raw_frame_data) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2（14维）

        参数:
            raw_frame_data: dict，包含以下键：
                - pressure_map_norm: ndarray (H,W) 归一化压力图
                - contact_mask: ndarray (H,W) bool 接触掩码
                - total_force_n: float 法向力（N）
                - centroid: tuple (row, col) 接触质心
                - in_contact: bool 是否接触
                - contact_area_mm2: float 接触面积 mm²
                - timestamp_s: float 时间戳
                - prev_state: dict 或 None 上一帧状态
                - dt: float 帧间隔
                - slip_threshold: float 滑移质心偏移阈值
                - force_slip_threshold: float 滑移力变化率阈值

        返回:
            TLabelSchemaV2 — L2级别，force_magnitude 从 total_force_n 映射，
            force_vector 为 None（无法测3D力）
        """
        # 复用现有 _compute_tlabel_v2 计算逻辑获取22维dict
        tlabel_v2, _ = self._compute_tlabel_v2(
            pressure_map_norm=raw_frame_data["pressure_map_norm"],
            contact_mask=raw_frame_data["contact_mask"],
            total_force_n=raw_frame_data["total_force_n"],
            centroid=raw_frame_data["centroid"],
            in_contact=raw_frame_data["in_contact"],
            contact_area_mm2=raw_frame_data["contact_area_mm2"],
            timestamp_s=raw_frame_data["timestamp_s"],
            dt=raw_frame_data.get("dt", 1.0 / 100),
            prev_state=raw_frame_data.get("prev_state"),
            slip_threshold=raw_frame_data.get("slip_threshold", SLIP_CENTROID_SHIFT_THRESHOLD),
            force_slip_threshold=raw_frame_data.get("force_slip_threshold", SLIP_FORCE_RATE_THRESHOLD),
        )

        # 从22维dict映射到14维Schema V2
        return self._v2_to_schema_v2(tlabel_v2)

    def _v2_to_schema_v2(self, tlabel_v2: Dict[str, Any]) -> TLabelSchemaV2:
        """把 22 维 flat 特征映射到 14 维 Schema V2。

        ``load()`` 与 ``extract_schema()`` 共用这一段：load 侧已经算过
        ``_compute_tlabel_v2``，再调一次 extract_schema 只会重复计算，
        且 ``prev_state`` 的滑移检测语义会错位。
        """
        contact = float(tlabel_v2["contact"]) > 0.5
        centroid_x = tlabel_v2.get("centroid_x", 0.5)

        return TLabelSchemaV2(
            contact=contact,
            contact_centroid=[float(centroid_x), 0.0] if contact else None,
            contact_region=None,
            force_magnitude=float(tlabel_v2["force_magnitude"]),
            force_vector=None,  # L2: 无法测3D力
            torque_vector=None,
            slip_event=float(tlabel_v2["slip_event"]) > 0.5,
            slip_velocity=None,
            manipulation_phase=None,
            texture_class=None,
            object_deformation=float(tlabel_v2["deformation_magnitude"]),
            temperature=None,
            confidence=self._compute_confidence(tlabel_v2),
            compliance_level=self.default_compliance_level,
        )

    # ─── 能力声明 ─────────────────────────────────────────────

    def get_capabilities(self) -> Dict[str, bool]:
        """13 个合法能力键（Schema V2，规范 S-7.4-1）。

        判定口径：以 load() 实际产出的帧内容为准，不按 extract_schema 的
        理论能力。compliance_level 是帧级元字段，不属于能力键（S-7.4-2）。
        帕西尼 Gen3 压力阵列（实时 + 兼容 load）。压力阵列无法测 3D 力方向；
        object_deformation 实为压力均值代理量。
        """
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": False,
            "force_magnitude": True,  # 由 total_force_n 映射（SDK 标定值）
            "force_vector": False,
            "torque_vector": False,
            "slip_event": True,  # 真实滑移检测（质心偏移 + 力变化率）
            "slip_velocity": False,
            "manipulation_phase": True,
            "texture_class": False,
            "object_deformation": True,  # pressure proxy, not physical deformation
            "temperature": False,
            "confidence": True,
        }

    def get_sensor_info(self, file_path=None) -> Dict[str, Any]:
        info = {
            "type": "distributed_taxel_array",
            "manufacturer": "paxini",
            "model": "GEN3",
            "modality": "pressure_array",
            "pressure_range_kpa": "0-600",
            "sampling_rate_range_hz": "50-500 (USB) / 50-200 (BLE)",
        }
        if self._connected and self._sensor is not None:
            info["connection"] = {
                "type": "usb" if hasattr(self._sensor, 'connection_type') and 'usb' in str(self._sensor.connection_type).lower() else "ble",
                "firmware_version": str(getattr(self._sensor, 'firmware_version', 'unknown')),
                "status": "connected",
            }
        else:
            info["connection"] = {"status": "disconnected"}
        return info

    # ─── SensorAdapterBase 实时接口实现 ─────────────────────────────────

    def connect(self, device_id: str = "auto", **kwargs) -> bool:
        """连接到PaXini GEN3传感器

        参数:
            device_id: 设备标识符
                - "auto": 自动检测第一个可用传感器（index=0）
                - 数字字符串: 传感器索引（如"0"）
                - 序列号: 传感器序列号（暂不支持）
            **kwargs:
                - sample_rate: 期望采样率Hz，默认100

        返回:
            bool — 连接是否成功

        异常:
            ImportError: 缺少paxini-sdk
            RuntimeError: 传感器连接失败
        """
        if not HAS_PAXINI_SDK:
            raise ImportError(
                "PaXini GEN3 适配器需要 paxini-sdk:\n"
                "  pip install paxini-sdk\n"
                "  pip install \"paxini-sdk[viz]\"  # 含可视化依赖"
            )

        if self._connected:
            self.disconnect()

        # 解析device_id
        if device_id == "auto":
            sensor_index = kwargs.get("sensor_index", 0)
        else:
            try:
                sensor_index = int(device_id)
            except ValueError:
                sensor_index = 0

        self._sample_rate = kwargs.get("sample_rate", 100)

        try:
            self._sensor = paxini.Sensor(index=sensor_index)
            self._sensor.start()
            self._sensor_index = sensor_index
            self._connected = True
            return True
        except Exception as e:
            raise RuntimeError(
                f"无法连接PaXini GEN3传感器 (index={sensor_index}): {e}"
            )

    def disconnect(self) -> None:
        """断开传感器连接，释放资源"""
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except Exception:
                pass  # 静默处理断开时的异常
            self._sensor = None
        self._connected = False

    def is_connected(self) -> bool:
        """检查当前连接状态"""
        return self._connected and self._sensor is not None

    def stream_frames(self, num_frames: int = -1,
                      **kwargs) -> Iterator[TLabelFrame]:
        """实时数据流生成器

        参数:
            num_frames: 采集帧数，-1表示无限采集直到手动停止
            **kwargs:
                - slip_threshold: 滑移质心偏移阈值，默认0.15
                - force_slip_threshold: 滑移力变化率阈值，默认0.25

        返回:
            Iterator[TLabelFrame] — 逐帧产出TLabelFrame

        异常:
            RuntimeError: 未连接时调用
        """
        if not self.is_connected():
            raise RuntimeError(
                f"传感器 {self.name} 未连接，请先调用 connect()"
            )

        slip_threshold = kwargs.get("slip_threshold", SLIP_CENTROID_SHIFT_THRESHOLD)
        force_slip_threshold = kwargs.get("force_slip_threshold", SLIP_FORCE_RATE_THRESHOLD)

        dt = 1.0 / self._sample_rate
        prev_state = None
        frame_idx = 0
        max_frames = num_frames if num_frames > 0 else float('inf')

        # 使用 SDK 的 stream() 迭代器
        for seq, tframe in enumerate(self._sensor.stream()):
            if frame_idx >= max_frames:
                break

            # ── 从 TactileFrame 提取数据 ──
            timestamp_s = tframe.timestamp_ns / 1e9  # ns → s
            pressure_map = tframe.pressure_map        # ndarray (H, W), float32, kPa
            contact_mask = tframe.contact_mask        # ndarray (H, W), bool
            total_force_n = tframe.total_force_n      # float, Newtons
            centroid = tframe.contact_centroid         # tuple (row, col)
            in_contact = tframe.in_contact             # bool
            contact_area_mm2 = tframe.contact_area_mm2  # float

            # ── 归一化计算 ──
            pmap = np.asarray(pressure_map, dtype=np.float32)
            cmask = np.asarray(contact_mask, dtype=bool)

            # 压力归一化到 0-1 (相对于 600kPa 满量程)
            pmap_norm = np.clip(pmap / MAX_PRESSURE_KPA, 0.0, 1.0)

            # 22维特征计算
            tlabel_v2, prev_state = self._compute_tlabel_v2(
                pressure_map_norm=pmap_norm,
                contact_mask=cmask,
                total_force_n=total_force_n,
                centroid=centroid,
                in_contact=in_contact,
                contact_area_mm2=contact_area_mm2,
                timestamp_s=timestamp_s,
                dt=dt,
                prev_state=prev_state,
                slip_threshold=slip_threshold,
                force_slip_threshold=force_slip_threshold,
            )

            # 操作阶段推断
            phase = self._infer_phase(tlabel_v2, frame_idx, max_frames if max_frames != float('inf') else 1000)

            # 置信度
            confidence = self._compute_confidence(tlabel_v2)

            # 生成伪触觉图像 (用于可视化): 压力阵列 → 灰度图
            pseudo_image = self._make_pseudo_image(pmap_norm)

            # 传感器特有数据
            sensor_specific = {
                "seq": tframe.seq if hasattr(tframe, 'seq') else seq,
                "pressure_map_shape": list(pmap.shape),
                "max_pressure_kpa": float(pmap.max()),
                "mean_pressure_kpa": float(pmap.mean()),
                "total_force_n": float(total_force_n),
                "contact_area_mm2": float(contact_area_mm2),
                "centroid_raw": list(centroid) if centroid is not None else None,
                "in_contact_sdk": bool(in_contact),
                "source": "paxini_gen3_realtime",
            }

            frame = TLabelFrame(
                frame_idx=frame_idx,
                timestamp_s=round(timestamp_s, 6),
                schema_v2=self._v2_to_schema_v2(tlabel_v2),
                manipulation_phase=phase,
                confidence=confidence,
                sensor_specific=sensor_specific,
                image=pseudo_image,  # 用于 viewer 可视化
            )
            yield frame

            frame_idx += 1

    # ─── 向后兼容：load() 方法 ─────────────────────────────────────────

    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             **kwargs) -> TLabelData:
        """加载 PaXini GEN3 传感器数据

        参数:
            file_path: .paxini 录制文件路径, 或空字符串/None 进入实时流模式
            trajectory_id: 未使用
            **kwargs:
                sensor_index (int): 传感器索引, 默认 0
                num_frames (int): 实时模式下采集帧数, 默认 200
                sample_rate (int): 期望采样率 Hz, 默认 100
                layout (str): 传感器布局, 如 "gen3_1"/"gen3_2"/"gen3_5", 默认自动检测
                slip_threshold (float): 滑移质心偏移阈值, 默认 0.15
                force_slip_threshold (float): 滑移力变化率阈值, 默认 0.25

        返回:
            TLabelData
        """
        if not HAS_PAXINI_SDK:
            raise ImportError(
                "PaXini GEN3 适配器需要 paxini-sdk:\n"
                "  pip install paxini-sdk\n"
                "  pip install \"paxini-sdk[viz]\"  # 含可视化依赖"
            )

        # 参数提取
        sensor_index = kwargs.get("sensor_index", 0)
        num_frames = kwargs.get("num_frames", 200)
        sample_rate = kwargs.get("sample_rate", 100)
        layout_name = kwargs.get("layout", None)
        slip_threshold = kwargs.get("slip_threshold", SLIP_CENTROID_SHIFT_THRESHOLD)
        force_slip_threshold = kwargs.get("force_slip_threshold", SLIP_FORCE_RATE_THRESHOLD)

        # 连接传感器
        sensor = paxini.Sensor(index=sensor_index)
        sensor.start()

        try:
            frames = self._capture_stream(
                sensor, num_frames, sample_rate,
                slip_threshold, force_slip_threshold
            )
        finally:
            sensor.stop()

        # 构建 sensor_info
        detected_layout = self._detect_layout(frames)
        sensor_info = {
            "type": "distributed_taxel_array",
            "manufacturer": "paxini",
            "model": "GEN3",
            "modality": "pressure_array",
            "layout": {
                "type": layout_name or detected_layout or "auto",
                "sampling_rate_hz": sample_rate,
                "total_frames": len(frames),
                "pressure_map_shape": frames[0].sensor_specific.get("pressure_map_shape") if frames else None,
            },
            "connection": {
                "type": "usb" if hasattr(sensor, 'connection_type') and 'usb' in str(sensor.connection_type).lower() else "ble",
                "firmware_version": str(getattr(sensor, 'firmware_version', 'unknown')),
            },
        }

        episode_info = {
            "source": "paxini_gen3_realtime",
            "sensor_index": sensor_index,
            "sample_rate_hz": sample_rate,
        }

        return TLabelData(
            frames=frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
            sensor_id=f"paxini_gen3_idx{sensor_index}",
        )

    # ─── 流式采集（供load()内部使用）─────────────────────────────

    def _capture_stream(self, sensor, num_frames: int, sample_rate: int,
                        slip_threshold: float, force_slip_threshold: float
                        ) -> List[TLabelFrame]:
        """从传感器实时采集帧并转换为 TLabelFrame 列表（供load()使用）"""

        dt = 1.0 / sample_rate
        frames: List[TLabelFrame] = []
        prev_state = None  # 用于时序计算

        # 使用 SDK 的 stream() 迭代器
        for seq, tframe in enumerate(sensor.stream()):
            if seq >= num_frames:
                break

            # ── 从 TactileFrame 提取数据 ──
            timestamp_s = tframe.timestamp_ns / 1e9  # ns → s
            pressure_map = tframe.pressure_map        # ndarray (H, W), float32, kPa
            contact_mask = tframe.contact_mask        # ndarray (H, W), bool
            total_force_n = tframe.total_force_n      # float, Newtons
            centroid = tframe.contact_centroid         # tuple (row, col)
            in_contact = tframe.in_contact             # bool
            contact_area_mm2 = tframe.contact_area_mm2  # float

            # ── 归一化计算 ──
            pmap = np.asarray(pressure_map, dtype=np.float32)
            cmask = np.asarray(contact_mask, dtype=bool)

            # 压力归一化到 0-1 (相对于 600kPa 满量程)
            pmap_norm = np.clip(pmap / MAX_PRESSURE_KPA, 0.0, 1.0)

            # 22维特征计算
            tlabel_v2, prev_state = self._compute_tlabel_v2(
                pressure_map_norm=pmap_norm,
                contact_mask=cmask,
                total_force_n=total_force_n,
                centroid=centroid,
                in_contact=in_contact,
                contact_area_mm2=contact_area_mm2,
                timestamp_s=timestamp_s,
                dt=dt,
                prev_state=prev_state,
                slip_threshold=slip_threshold,
                force_slip_threshold=force_slip_threshold,
            )

            # 操作阶段推断
            phase = self._infer_phase(tlabel_v2, seq, num_frames)

            # 置信度
            confidence = self._compute_confidence(tlabel_v2)

            # 生成伪触觉图像 (用于可视化): 压力阵列 → 灰度图
            pseudo_image = self._make_pseudo_image(pmap_norm)

            # 传感器特有数据
            sensor_specific = {
                "seq": tframe.seq if hasattr(tframe, 'seq') else seq,
                "pressure_map_shape": list(pmap.shape),
                "max_pressure_kpa": float(pmap.max()),
                "mean_pressure_kpa": float(pmap.mean()),
                "total_force_n": float(total_force_n),
                "contact_area_mm2": float(contact_area_mm2),
                "centroid_raw": list(centroid) if centroid is not None else None,
                "in_contact_sdk": bool(in_contact),
                "source": "paxini_gen3_load",
            }

            frame = TLabelFrame(
                frame_idx=seq,
                timestamp_s=round(timestamp_s, 6),
                schema_v2=self._v2_to_schema_v2(tlabel_v2),
                manipulation_phase=phase,
                confidence=confidence,
                sensor_specific=sensor_specific,
                image=pseudo_image,  # 用于 viewer 可视化
            )
            frames.append(frame)

        return frames

    # ─── 22维特征计算 ──────────────────────────────────────────

    def _compute_tlabel_v2(
        self,
        pressure_map_norm: np.ndarray,
        contact_mask: np.ndarray,
        total_force_n: float,
        centroid: tuple,
        in_contact: bool,
        contact_area_mm2: float,
        timestamp_s: float,
        dt: float,
        prev_state: Optional[Dict],
        slip_threshold: float,
        force_slip_threshold: float,
    ) -> tuple:
        """计算 22 维 TLabel v2 特征向量

        返回: (tlabel_v2_dict, state_dict_for_next_frame)
        """
        H, W = pressure_map_norm.shape

        # ── 基础接触特征 ──
        contact = 1.0 if (in_contact or contact_mask.any()) else 0.0
        deform_mag = float(pressure_map_norm.mean())
        force_mag = min(float(total_force_n) / 50.0, 1.0)  # 归一化: ~50N 为满量程
        force_peak = min(float(pressure_map_norm.max()), 1.0)

        # 接触面积: SDK 提供 mm², 归一化为激活比例
        total_taxel_area_mm2 = H * W * (1.75 ** 2)  # 假设指尖 1.75mm 间距
        contact_area_ratio = min(float(contact_mask.mean()), 1.0)

        # 质心 X: SDK 提供 (row, col), 归一化到 0-1
        if centroid is not None and contact > 0:
            centroid_x = float(centroid[1]) / max(W - 1, 1)
        else:
            centroid_x = 0.5

        # 法向力场
        nf_mag = deform_mag  # 压力阵列主方向为法向
        nf_var = float(pressure_map_norm.var())

        # ── 滑移检测 ──
        slip_event = 0.0
        slip_entropy = 0.0
        delta_fn = 0.0

        if prev_state is not None:
            # 质心偏移法检测滑移
            prev_centroid_x = prev_state.get("centroid_x", 0.5)
            centroid_shift = abs(centroid_x - prev_centroid_x)
            if centroid_shift > slip_threshold and contact > 0:
                slip_event = 1.0

            # 力变化率法辅助确认
            prev_force = prev_state.get("force_mag", 0.0)
            if prev_force > 0.01:
                force_rate = abs(force_mag - prev_force) / prev_force
                if force_rate > force_slip_threshold:
                    slip_event = 1.0  # 任一条件触发即标记
                delta_fn = abs(force_mag - prev_force)

            # 滑移熵: 压力分布的信息熵变化
            active_vals = pressure_map_norm[pressure_map_norm > 0.01]
            if len(active_vals) >= 2:
                total = active_vals.sum()
                if total > 1e-10:
                    probs = active_vals / total
                    probs = probs[probs > 0]
                    slip_entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        # ── 时序特征 ──
        temporal_deform_rate = 0.0
        contact_trans = 0.0

        if prev_state is not None and dt > 0:
            prev_deform = prev_state.get("deform_mag", 0.0)
            temporal_deform_rate = abs(deform_mag - prev_deform) / dt

            prev_contact = prev_state.get("contact", 0.0)
            prev_area = prev_state.get("contact_area_ratio", 0.0)
            contact_trans = min(1.0, abs(contact - prev_contact) +
                               abs(contact_area_ratio - prev_area) * 5.0)

        # ── 组装 22 维 ──
        tlabel_v2 = {
            "contact": contact,
            "deformation_magnitude": round(deform_mag, 4),
            "force_magnitude": round(force_mag, 4),
            "force_peak": round(force_peak, 4),
            "force_direction": 0.0,
            "slip_entropy": round(slip_entropy, 4),
            "slip_event": slip_event,
            "texture_energy": 0.0,
            "edge_density": 0.0,
            "contact_area": round(contact_area_ratio, 4),
            "centroid_x": round(centroid_x, 4),
            "normal_field_magnitude": round(nf_mag, 4),
            "normal_field_variance": round(nf_var, 4),
            "shear_field_magnitude": 0.0,
            "shear_field_direction": 0.0,
            "delta_force_normal": round(delta_fn, 4),
            "delta_force_shear": 0.0,
            "friction_cone_ratio": 0.0,
            "optical_flow_magnitude": 0.0,
            "optical_flow_direction": 0.0,
            "temporal_deformation_rate": round(temporal_deform_rate, 4),
            "contact_transition": round(contact_trans, 4),
        }

        # 保存状态供下一帧使用
        next_state = {
            "centroid_x": centroid_x,
            "force_mag": force_mag,
            "deform_mag": deform_mag,
            "contact": contact,
            "contact_area_ratio": contact_area_ratio,
        }

        return tlabel_v2, next_state

    # ─── 辅助方法 ──────────────────────────────────────────────

    @staticmethod
    def _infer_phase(tlabel_v2: Dict, seq: int, total: int) -> str:
        """推断操作阶段"""
        contact = tlabel_v2["contact"]
        slip = tlabel_v2["slip_event"]

        if slip > 0.5:
            return "slip"
        if contact < 0.5:
            # 无接触: 前半段为接近, 后半段为撤回
            return "approach" if seq < total * 0.4 else "retract"
        # 有接触且稳定
        if tlabel_v2["contact_transition"] > 0.5:
            return "manipulation"
        return "stable_contact"

    @staticmethod
    def _compute_confidence(tlabel_v2: Dict) -> float:
        """计算标注置信度"""
        contact = tlabel_v2["contact"]
        slip = tlabel_v2["slip_event"]

        if contact < 0.5 and slip < 0.5:
            return 0.95  # 无接触, 数据简单, 置信度高
        if contact > 0.5 and slip < 0.5:
            return 0.85  # 稳定接触
        return 0.6  # 滑移中, 数据较复杂

    def detect_image_shape(self, file_path: Optional[str] = None):
        """检测PaXini GEN3触觉图像（压力阵列）形状

        根据传感器布局配置返回已知形状：
          - GEN3-1 指尖: (8, 8, 1) — 单通道灰度
          - GEN3-2 掌心: (16, 12, 1)

        如果传感器已连接，从实际设备读取。
        如果提供 file_path，尝试从 .paxini 文件读取元数据。

        Returns:
            (height, width, channels) 或 None
        """
        # 如果已连接，从实际传感器获取
        if self._connected and self._sensor is not None:
            try:
                for seq, tframe in enumerate(self._sensor.stream()):
                    if seq >= 1:
                        break
                    shape = tuple(tframe.pressure_map.shape)
                    if len(shape) == 2:
                        return (shape[0], shape[1], 1)
                    elif len(shape) == 3:
                        return tuple(shape)
            except Exception:
                pass

        # 尝试从文件读取
        if file_path is not None:
            try:
                import h5py
                with h5py.File(file_path, 'r') as hf:
                    if 'dataset' in hf and 'observation' in hf['dataset']:
                        obs = hf['dataset']['observation']
                        if 'pressure_map' in obs:
                            shape = obs['pressure_map'].shape
                            if len(shape) >= 2:
                                # 取最后几帧中的一帧
                                if len(shape) == 3:
                                    return (shape[1], shape[2], 1)
                                elif len(shape) == 4:
                                    return (shape[2], shape[3], 1)
            except Exception:
                pass

        # 根据已知布局返回（默认指尖）
        if self._connected:
            # 尝试从传感器型号推断
            info = self.get_sensor_info()
            layout = info.get("layout", {})
            if isinstance(layout, dict):
                layout_name = layout.get("type", "")
                if layout_name in GEN3_LAYOUTS:
                    regions = GEN3_LAYOUTS[layout_name]["regions"]
                    first_region = next(iter(regions.values()))
                    shape = first_region["shape"]
                    return (shape[0], shape[1], 1)

        # 默认：GEN3-1 指尖 (8, 8, 1)
        return (8, 8, 1)

    @staticmethod
    def _make_pseudo_image(pressure_map_norm: np.ndarray) -> np.ndarray:
        """将归一化压力图转为伪触觉图像 (uint8 灰度)

        用于 viewer 可视化。标注为 "Pressure Heatmap" 避免误导。
        """
        # 归一化到 0-255
        img = (pressure_map_norm * 255).astype(np.uint8)
        # 应用 colormap (jet-like): 低值蓝, 高值红
        # 简单实现: 灰度即可, viewer 端可做伪彩
        return img

    @staticmethod
    def _detect_layout(frames: List[TLabelFrame]) -> Optional[str]:
        """从采集到的帧数据推断传感器布局"""
        if not frames:
            return None

        shape = tuple(frames[0].sensor_specific.get("pressure_map_shape", []))
        if shape == (8, 8):
            return "gen3_1"  # 单区指尖
        elif shape == (16, 12):
            return "gen3_2"  # 可能掌心区
        return None
