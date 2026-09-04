"""Adapter package

v0.16.0 架构分离：
- DataAdapterBase: 数据集适配器基类（解析离线数据文件）
- SensorAdapterBase: 传感器适配器基类（实时对接传感器SDK）
- BaseAdapter: 向后兼容别名，等同于DataAdapterBase
"""

from tlabel.adapters.base import (
    BaseAdapter,        # 向后兼容
    DataAdapterBase,    # 数据集适配器基类
    SensorAdapterBase,  # 传感器适配器基类
)

# Lazy imports to avoid dependency failures
# Adapters are registered via registry._ensure_adapters()

__all__ = [
    "BaseAdapter",        # v0.15及之前版本使用
    "DataAdapterBase",    # v0.16+ 数据集适配器
    "SensorAdapterBase",  # v0.16+ 传感器适配器
]

# Available adapters (for documentation and autocomplete)
AVAILABLE_ADAPTERS = {
    # 数据集适配器 (DataAdapterBase)
    "gelsight":      "GelSight Mini / DIGIT (.pkl)",
    "paxini":        "PaXini PXCap dataset (.h5)",
    "daimon":        "Daimon DM-TacClaw dataset (.parquet / LeRobot)",
    "tlabel":        "TLabel Format JSON (.json)",
    "touchd":        "ToucHD-Force / AnyTouch 2 (.npy / directory)",
    "univtac":       "UniVTAC Cross-Dataset (.hdf5 / .h5)",
    "vtouch":        "VTouch vision-based tactile (.h5 / .hdf5)",
    "ycb_slide":     "YCB-Slide CMU DIGIT sliding (.npy / directory)",
    "tacquad":       "TacQuad AnyTouch multi-sensor (directory)",
    "syntouch":      "SynTouch BioTac (.h5 / .csv / .mat)",
    "xela":          "XELA uSkin / UniTac-NV dataset (.csv)",
    "tashan_ts_f_a": "RoboMIND Tashan TS-F-A tactile (.hdf5 / .h5)",
    # 传感器适配器 (SensorAdapterBase)
    "paxini_gen3":   "PaXini GEN3 realtime (SDK / .paxini)",
    "daimon_dm_tac": "Daimon DM-Tac realtime (USB / UVC)",
    # 占位符
    "paxini_px6d":   "PaXini PX6D 6-axis force (Modbus) [placeholder]",
}

# 按类型分组（方便文档生成和贡献者理解）
DATA_ADAPTERS = {
    k: v for k, v in AVAILABLE_ADAPTERS.items()
    if k not in ("paxini_gen3", "daimon_dm_tac", "paxini_px6d")
}

SENSOR_ADAPTERS = {
    k: v for k, v in AVAILABLE_ADAPTERS.items()
    if k in ("paxini_gen3", "daimon_dm_tac")
}
