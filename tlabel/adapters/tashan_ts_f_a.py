"""
他山科技 TS-F-A 触觉传感器数据集适配器

命名规范: 品牌名称(tashan) + 传感器型号(TS-F-A) -> tashan_ts_f_a
数据来源: RoboMIND V2.0 (AgileX Cobot Magic)
HDF5结构: tactile_observations/tactile_{left/right}_align/data
数据形状: (T, 2, 6) float32
  - dim0: normal_force (法向力, N)
  - dim1: tangential_force (切向力, N)
  - dim2: tangential_direction (切向方向, rad)
  - dim3: tangential_fx (切向力x分量, N)
  - dim4: tangential_fy (切向力y分量, N)
  - dim5: contact_indicator (接触指示)
  - 65535.0 = 无效值/溢出标记 (uint16 overflow)

Compliance Level: L3 (有完整3D力向量: tangential_fx, tangential_fy, normal_force)

传感器信息层级:
  - 适配器类级别: get_sensor_info() / get_capabilities() → 厂商、型号、维度、单位
  - 帧级别: TLabelFrame.sensor_id → 标识来源传感器 (如 "left_sensor0")
  - Schema级别: raw_sensor_values → 保留原始6维数据值（核心字段只取 force_magnitude/force_vector，原始值会丢失）
  - 三层传感器信息架构：get_sensor_info() / TLabelFrame.sensor_id / raw_sensor_values
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from tlabel.adapters.base import DataAdapterBase
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

INVALID_MARKER = 65535.0
CONTACT_THRESHOLD = 0.01


class TashanTsFAAdapter(DataAdapterBase):
    """他山科技 TS-F-A 指尖力传感器 -> TLabelData

    解析 RoboMIND V2.0 中 AgileX 移动双臂机器人采集的 Tashan 触觉数据。
    Compliance Level: L3（有法向力 + 切向力x/y分量 = 完整3D力向量）
    """

    default_compliance_level: str = "L3"

    @property
    def name(self) -> str:
        return "tashan_ts_f_a"

    @property
    def manufacturer(self) -> str:
        return "Tashan"

    @property
    def model(self) -> str:
        return "TS-F-A"

    @property
    def supported_extensions(self) -> List[str]:
        return [".hdf5", ".h5"]

    def get_capabilities(self) -> Dict[str, bool]:
        """Schema V2 14维能力声明

        TS-F-A 是力传感器（非视觉触觉），只能提供接触检测 + 力信息，
        无法提供形变、纹理、滑移等视觉类传感器能力。
        """
        return {
            "contact": True,               # 有 contact_indicator 通道
            "contact_centroid": False,
            "contact_region": False,
            "force_magnitude": True,       # 可计算法向力幅值
            "force_vector": True,          # L3: 完整3D力向量 [fx, fy, nf]
            "torque_vector": False,
            "slip_event": False,           # 无直接滑检测
            "slip_velocity": False,
            "manipulation_phase": False,
            "texture_class": False,
            "object_deformation": False,
            "temperature": False,
            "confidence": True,            # extract_schema 输出 confidence
            # 注意：compliance_level 是帧级元字段，不属于 capabilities 能力键
            # （规范 S-7.4-2 / C-15），v0.23.3 起不再出现在此处。
        }

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "type": "force",
            "dimensions": 6,
            "channels": ["normal_force", "tangential_force", "tangential_direction",
                         "tangential_fx", "tangential_fy", "contact_indicator"],
            "units": {
                "normal_force": "N",
                "tangential_force": "N",
                "tangential_direction": "rad",
                "tangential_fx": "N",
                "tangential_fy": "N",
                "contact_indicator": "binary",
            },
            "num_sensors_per_hand": 2,
            "invalid_marker": INVALID_MARKER,
            "notes": "RoboMIND V2.0 data; 65535.0 = invalid/no-contact marker",
        }

    def extract_schema(self, raw_frame_data: Dict[str, Any]) -> TLabelSchemaV2:
        """将 Tashan TS-F-A 原始数据转换为 TLabel Schema V2

        传感器身份信息承载方式:
          - get_sensor_info() → 厂商/型号/维度/单位（类级别）
          - TLabelFrame.sensor_id → 来源传感器标识（帧级别）
          - raw_sensor_values → 原始6维数据值保留（防止核心字段映射后丢失）

        参数:
            raw_frame_data: dict with keys:
                - normal_force: float (dim0)
                - tangential_force: float (dim1)
                - tangential_direction: float (dim2)
                - tangential_fx: float (dim3)
                - tangential_fy: float (dim4)
                - contact_indicator: float (dim5)

        返回:
            TLabelSchemaV2 - L3级别（有完整3D力向量）
        """
        nf = raw_frame_data.get("normal_force", 0.0)
        tf = raw_frame_data.get("tangential_force", 0.0)
        fx = raw_frame_data.get("tangential_fx", 0.0)
        fy = raw_frame_data.get("tangential_fy", 0.0)

        # 无效值处理
        if nf == INVALID_MARKER or tf == INVALID_MARKER:
            return TLabelSchemaV2(
                contact=False,
                force_magnitude=0.0,
                force_vector=None,
                compliance_level="L1",
                raw_sensor_values={
                    "raw_normal_force": 0.0,
                    "raw_tangential_force": 0.0,
                    "invalid": True,
                    "invalid_marker_value": INVALID_MARKER,
                },
            )

        contact = bool(nf > CONTACT_THRESHOLD or tf > CONTACT_THRESHOLD)

        # raw_sensor_values: 保留原始6维数据值，核心字段只取 force_magnitude 和 force_vector，原始值会丢失
        dq = {
            "raw_normal_force": float(nf),
            "raw_tangential_force": float(tf),
            "tangential_direction": float(raw_frame_data.get("tangential_direction", 0.0)),
            "tangential_fx": float(fx),
            "tangential_fy": float(fy),
            "contact_indicator": float(raw_frame_data.get("contact_indicator", 0.0)),
            "invalid_marker_value": INVALID_MARKER,
        }

        return TLabelSchemaV2(
            contact=contact,
            contact_centroid=None,
            force_magnitude=float(nf) if contact else 0.0,
            force_vector=[float(fx), float(fy), float(nf)] if contact else None,
            slip_event=None,
            manipulation_phase=None,
            object_deformation=None,
            temperature=None,
            texture_class=None,
            confidence=0.9 if contact else 0.5,
            compliance_level="L3" if contact else "L1",
            raw_sensor_values=dq,
        )

    def load(self, file_path: str, **kwargs) -> Optional[TLabelData]:
        """加载 RoboMIND V2.0 HDF5 触觉数据"""
        if not HAS_H5PY:
            raise ImportError("h5py is required")
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        frames = []
        with h5py.File(str(path), "r") as f:
            for side in ["left", "right"]:
                tactile_key = f"tactile_observations/tactile_{side}_align/data"
                if tactile_key not in f:
                    raise ValueError(f"Missing tactile data: {tactile_key}")
                data = f[tactile_key]
                T = data.shape[0]
                for t in range(T):
                    for sensor_idx in range(data.shape[1]):
                        raw = {
                            "normal_force": float(data[t, sensor_idx, 0]),
                            "tangential_force": float(data[t, sensor_idx, 1]),
                            "tangential_direction": float(data[t, sensor_idx, 2]),
                            "tangential_fx": float(data[t, sensor_idx, 3]),
                            "tangential_fy": float(data[t, sensor_idx, 4]),
                            "contact_indicator": float(data[t, sensor_idx, 5]),
                        }
                        schema = self.extract_schema(raw)
                        # 传感器身份由 TLabelFrame.sensor_id 承载
                        frame = TLabelFrame(
                            frame_index=t,
                            sensor_id=f"{side}_sensor{sensor_idx}",
                            schema_v2=schema,
                        )
                        frames.append(frame)
        return TLabelData(
            frames=frames,
            sensor_type="force",
            source_adapter=self.name,
            source_file=str(path),
            total_frames=len(frames),
        )
