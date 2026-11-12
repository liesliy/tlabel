"""
适配器抽象基类 — v0.17.0 Schema V2 迁移

两个基类：
- DataAdapterBase: 数据集适配器，解析离线数据文件
- SensorAdapterBase: 传感器适配器，通过SDK实时读取传感器数据

v0.15及之前版本使用统一的BaseAdapter，v0.16起拆分为两个专用基类，
为第三方贡献机制提供更清晰的接口契约。
v0.17起新增 extract_schema() 方法，支持14维Schema V2输出。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Iterator, List, Tuple

from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2, SCHEMA_V2_FIELD_NAMES


# =============================================================================
# DataAdapterBase — 数据集适配器基类
# =============================================================================

class DataAdapterBase(ABC):
    """数据集适配器基类 — 用于解析离线数据文件

    适用于：公开数据集、本地存储的传感器数据文件
    特点：不需要传感器硬件，只需要数据文件即可工作

    实现步骤：
    1. 继承DataAdapterBase
    2. 实现name、supported_extensions、load()、get_capabilities()、get_sensor_info()
    3. 实现 extract_schema() 将原始数据转换为14维Schema V2格式
    4. 在load()中将原始数据转换为TLabelData（14维Schema V2格式）
    5. 使用@register_adapter注册到适配器表
    """

    # 子类覆盖此属性声明其 Compliance Level（默认 L1）
    default_compliance_level: str = "L1"

    @property
    @abstractmethod
    def name(self) -> str:
        """适配器名称（唯一标识符，小写+下划线）"""
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名列表，如['.h5', '.pkl']"""
        pass

    @abstractmethod
    def extract_schema(self, raw_frame_data) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2（14维结构化触觉语义标注）

        子类必须实现此方法，将传感器/数据集的原始数据映射到14维Schema V2。
        不能填的字段填 None，compliance_level 使用 self.default_compliance_level。

        参数:
            raw_frame_data: 原始数据帧（格式由子类定义，通常为dict或SDK对象）

        返回:
            TLabelSchemaV2 — 14维结构化触觉语义标注
        """
        pass

    @abstractmethod
    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             **kwargs) -> TLabelData:
        """
        加载数据文件，转换为TLabelData

        .. deprecated::
            load() 内部产出仍使用22维tlabel_v2 dict，将在后续版本迁移为
            Schema V2。新增 extract_schema() 为推荐接口。

        参数:
            file_path: 数据文件路径
            trajectory_id: 轨迹ID（可选，用于多轨迹数据集）
            **kwargs: 适配器特有参数

        返回:
            TLabelData — 统一标注容器，包含14维Schema V2特征
        """
        pass

    def iter_frames(self, file_path: str, **kwargs) -> Iterator[TLabelFrame]:
        """流式逐帧产出 TLabelFrame（v0.25.0 新增）。

        这是面向大数据集的低内存入口。默认实现基于 load()：先完整加载
        再逐帧产出，语义正确但不节省内存。真正的流式能力由具体适配器
        覆盖实现（例如 daimon 逐帧解码 FFV1 视频，不把整段视频读进内存）。

        参数:
            file_path: 数据文件路径
            **kwargs: 与 load() 一致；适配器可额外解释 stream 等关键字

        产出:
            TLabelFrame — 与 load().frames 中的元素逐字段一致
        """
        data = self.load(file_path, **kwargs)
        for frame in data.frames:
            yield frame

    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """返回该数据源的14维能力声明

        返回格式示例:
            {
                "contact": True,
                "force_magnitude": True,
                "force_vector": False,
                ...
            }
        """
        pass

    @abstractmethod
    def get_sensor_info(self) -> Dict[str, Any]:
        """返回数据源元信息

        返回格式示例:
            {
                "type": "vision-based_tactile",
                "manufacturer": "gelsight",
                "model": "GelSight Mini",
                "dataset_name": "example_dataset",
                ...
            }
        """
        pass

    def build_provenance(self, raw_frame_data: Any = None) -> Optional[Dict[str, Any]]:
        """根据传感器信息自动填充 provenance 最小元数据（v0.21.0）

        默认实现从 get_sensor_info() 提取 sensor_model；子类可 override 以
        填充 firmware/calibration_date/sampling_rate_hz 等字段。

        参数:
            raw_frame_data: 原始帧数据（可选），某些传感器需要从当前帧中
                           提取 sampling_rate 或 firmware 信息。

        返回:
            符合 TLabelSchemaV2.provenance 结构的字典，或 None（无可填信息时）
        """
        info = self.get_sensor_info()
        if not info:
            return None

        provenance: Dict[str, Any] = {}

        # sensor_model: 优先取 info['model']，其次拼 manufacturer+model
        model = info.get("model")
        if model:
            provenance["sensor_model"] = str(model)
        elif info.get("manufacturer"):
            provenance["sensor_model"] = str(info["manufacturer"])

        # sampling_rate_hz: 部分适配器在 sensor_info 里声明
        sr = info.get("sampling_rate_hz") or info.get("sample_rate")
        if sr is not None:
            provenance["sampling_rate_hz"] = sr

        # firmware: 部分适配器在 sensor_info 里声明
        fw = info.get("firmware") or info.get("sensor_firmware")
        if fw is not None:
            provenance["sensor_firmware"] = str(fw)

        return provenance if provenance else None

    def detect_image_shape(self, file_path: Optional[str] = None) -> Optional[Tuple[int, int, int]]:
        """检测该数据源输出的触觉图像形状

        用于 LeRobot 等下游框架集成时确定 observation space。
        返回 (height, width, channels)，如无法确定则返回 None。

        参数:
            file_path: 可选，数据文件路径。某些适配器需要从实际数据中
                       推断图像形状；如果适配器可以从配置直接确定，则不需要此参数。

        返回:
            (height, width, channels) 或 None
        """
        return None


# =============================================================================
# SensorAdapterBase — 传感器适配器基类
# =============================================================================

class SensorAdapterBase(ABC):
    """传感器适配器基类 — 用于实时对接传感器SDK

    适用于：通过SDK/硬件接口实时读取传感器数据
    特点：需要传感器硬件 + 厂商SDK，支持实时数据流

    实现步骤：
    1. 继承SensorAdapterBase
    2. 实现name、supported_extensions、load()、get_capabilities()、get_sensor_info()
    3. 实现 extract_schema() 将原始数据转换为14维Schema V2格式
    4. 实现connect()、stream_frames()、disconnect()实时数据流接口
    5. 使用@register_adapter注册到适配器表

    注意：传感器适配器需要额外的依赖（厂商SDK），建议在pyproject.toml中
    使用optional dependencies声明，如：tlabel[sensor-paxini]
    """

    # 子类覆盖此属性声明其 Compliance Level（默认 L1）
    default_compliance_level: str = "L1"

    @property
    @abstractmethod
    def name(self) -> str:
        """适配器名称（唯一标识符，小写+下划线）"""
        pass

    @property
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名列表（传感器适配器通常为空或特定录制格式）

        对于实时传感器，这个方法返回录制文件的格式（如.avi, .bag），
        用于支持"先录制后处理"的工作流。
        """
        return []

    @abstractmethod
    def extract_schema(self, raw_frame_data) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2（14维结构化触觉语义标注）

        子类必须实现此方法，将传感器的原始数据映射到14维Schema V2。
        不能填的字段填 None，compliance_level 使用 self.default_compliance_level。

        参数:
            raw_frame_data: 原始数据帧（格式由子类定义，通常为SDK TactileFrame对象或dict）

        返回:
            TLabelSchemaV2 — 14维结构化触觉语义标注
        """
        pass

    @abstractmethod
    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             **kwargs) -> TLabelData:
        """
        加载数据文件，转换为TLabelData

        对于传感器适配器，这个方法用于处理录制好的数据文件。
        实时采集请使用stream_frames()接口。

        .. deprecated::
            load() 内部产出仍使用22维tlabel_v2 dict，将在后续版本迁移为
            Schema V2。新增 extract_schema() 为推荐接口。

        参数:
            file_path: 数据文件路径
            trajectory_id: 轨迹ID（可选）
            **kwargs: 适配器特有参数

        返回:
            TLabelData — 统一标注容器
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """返回该传感器的14维能力声明"""
        pass

    @abstractmethod
    def get_sensor_info(self) -> Dict[str, Any]:
        """返回传感器元信息"""
        pass

    def build_provenance(self, raw_frame_data: Any = None) -> Optional[Dict[str, Any]]:
        """根据传感器信息自动填充 provenance 最小元数据（v0.21.0）

        与 DataAdapterBase.build_provenance 行为一致。子类可 override 以
        在连接后动态填充 firmware / calibration_date / sampling_rate_hz 等字段。
        """
        info = self.get_sensor_info()
        if not info:
            return None
        provenance: Dict[str, Any] = {}
        model = info.get("model")
        if model:
            provenance["sensor_model"] = str(model)
        elif info.get("manufacturer"):
            provenance["sensor_model"] = str(info["manufacturer"])
        sr = info.get("sampling_rate_hz") or info.get("sample_rate")
        if sr is not None:
            provenance["sampling_rate_hz"] = sr
        fw = info.get("firmware") or info.get("sensor_firmware")
        if fw is not None:
            provenance["sensor_firmware"] = str(fw)
        return provenance if provenance else None

    # ─── 实时数据流接口（SensorAdapterBase特有）────────────────────────────

    @abstractmethod
    def connect(self, device_id: str = "auto", **kwargs) -> bool:
        """
        连接到传感器

        参数:
            device_id: 设备标识符
                - "auto": 自动检测第一个可用设备
                - 数字字符串: 设备索引（如"0"表示/dev/video0）
                - 序列号: 设备序列号
            **kwargs: 传感器特有的连接参数
                - 采样率、分辨率、通信协议等

        返回:
            bool — 连接是否成功

        异常:
            ConnectionError: 连接失败时抛出
            ImportError: 缺少SDK依赖时抛出
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        断开传感器连接，释放资源

        必须确保：
        - 设备句柄正确关闭
        - 内存/文件句柄释放
        - 可安全再次调用connect()
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查当前连接状态

        返回:
            bool — 是否已连接到传感器
        """
        pass

    @abstractmethod
    def stream_frames(self, num_frames: int = -1,
                      **kwargs) -> Iterator[TLabelFrame]:
        """
        实时数据流生成器

        参数:
            num_frames: 采集帧数，-1表示无限采集直到手动停止
            **kwargs: 采集参数
                - timeout_ms: 单帧超时时间
                - skip_frames: 跳帧数（降采样）

        返回:
            Iterator[TLabelFrame] — 逐帧产出TLabelFrame

        用法示例:
            adapter.connect("auto")
            for frame in adapter.stream_frames(num_frames=100):
                process(frame)
            adapter.disconnect()
        """
        pass

    def collect(self, num_frames: int, **kwargs) -> TLabelData:
        """
        便捷方法：采集指定帧数并返回TLabelData

        参数:
            num_frames: 采集帧数
            **kwargs: 传递给stream_frames()的参数

        返回:
            TLabelData — 包含所有采集帧的统一标注容器

        异常:
            RuntimeError: 未连接时调用
        """
        if not self.is_connected():
            raise RuntimeError(
                f"传感器 {self.name} 未连接，请先调用 connect()"
            )

        frames: List[TLabelFrame] = []
        for frame in self.stream_frames(num_frames=num_frames, **kwargs):
            frames.append(frame)

        # 组装为TLabelData
        from tlabel.core.types import TLabelData
        return TLabelData(
            frames=frames,
            sensor_info=self.get_sensor_info(),
            episode_info={},
            capabilities=self.get_capabilities(),
        )

    def detect_image_shape(self, file_path: Optional[str] = None) -> Optional[Tuple[int, int, int]]:
        """检测该传感器输出的触觉图像形状

        用于 LeRobot 等下游框架集成时确定 observation space。
        返回 (height, width, channels)，如无法确定则返回 None。

        对于实时传感器，通常可以从已知配置直接返回；
        对于需要数据的场景，可传入 file_path 从录制文件中采样。

        参数:
            file_path: 可选，录制文件路径。某些传感器需要从实际数据中
                       推断图像形状。

        返回:
            (height, width, channels) 或 None
        """
        return None


# =============================================================================
# 向后兼容：BaseAdapter 作为 DataAdapterBase 的别名
# =============================================================================

# v0.15及之前版本使用BaseAdapter，v0.16起拆分为DataAdapterBase和SensorAdapterBase
# 为保持向后兼容，BaseAdapter作为DataAdapterBase的别名
BaseAdapter = DataAdapterBase
