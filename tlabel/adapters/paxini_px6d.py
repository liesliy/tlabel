"""
PaXini PX6D 六维力/触觉传感器适配器 — placeholder（开发中）

PX6D 是 PaXini 新一代六维力触觉传感器，集成高精度力觉与触觉感知。
该适配器为占位符，目前尚未实现，后续版本将提供完整支持。

实时数据读取相关方法（connect / stream_frames 等）仍抛出 NotImplementedError，
其余元信息方法返回占位数据，以保证适配器可正常实例化并通过 CI 基础检查。
"""

from typing import Optional, Dict, Any, Iterator, List

from tlabel.adapters.base import SensorAdapterBase
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2


class PaxiniPX6DAdapter(SensorAdapterBase):
    """PaXini PX6D 六维力触觉传感器 → TLabelData (placeholder)

    **占位符适配器 — 正在开发中**

    PX6D 是 PaXini 新一代六维力/触觉一体化传感器，支持：
      - 6 轴力/力矩测量（Fx/Fy/Fz/Mx/My/Mz）
      - 高分辨率触觉阵列（待公布规格）
      - 集成温度补偿与实时标定

    本类为注册占位符，实时数据流相关方法（connect / stream_frames 等）
    仍抛出 NotImplementedError。完整实现将在后续版本中提供。

    依赖: paxini-px6d-sdk（厂商 SDK，待发布）
    """

    def __init__(self):
        # 占位符：不连接硬件，仅完成实例化
        pass

    @property
    def name(self) -> str:
        return "paxini_px6d"

    @property
    def supported_extensions(self) -> List[str]:
        return []

    def get_capabilities(self) -> Dict[str, bool]:
        """返回占位符能力声明（全部为 False，表示暂未实现）

        包含 Schema V2 的 13 个能力键，值统一为 False。
        注意：compliance_level 是帧级元字段，不属于能力键（规范 S-7.4-2）。
        """
        return {
            "contact": False,
            "contact_centroid": False,
            "contact_region": False,
            "force_magnitude": False,
            "force_vector": False,
            "torque_vector": False,
            "slip_event": False,
            "slip_velocity": False,
            "manipulation_phase": False,
            "texture_class": False,
            "object_deformation": False,
            "temperature": False,
            "confidence": False,
        }

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "name": "PaXini PX6D",
            "manufacturer": "PaXini",
            "status": "placeholder",
            "note": "Adapter under development",
        }

    def extract_schema(self, raw_frame_data) -> TLabelSchemaV2:
        """占位实现：返回空 Schema（compliance_level=L1）"""
        return TLabelSchemaV2(
            contact=False,
            slip_event=False,
            confidence=0.0,
            compliance_level="L1",
        )

    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             **kwargs) -> TLabelData:
        raise NotImplementedError("PaXini PX6D adapter is under development")

    def connect(self, device_id: str = "auto", **kwargs) -> bool:
        raise NotImplementedError("PaXini PX6D adapter is under development")

    def disconnect(self) -> None:
        raise NotImplementedError("PaXini PX6D adapter is under development")

    def is_connected(self) -> bool:
        raise NotImplementedError("PaXini PX6D adapter is under development")

    def stream_frames(self, num_frames: int = -1,
                      **kwargs) -> Iterator[TLabelFrame]:
        raise NotImplementedError("PaXini PX6D adapter is under development")
