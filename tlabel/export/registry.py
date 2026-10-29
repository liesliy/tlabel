"""
Exporter Plugin Registry — 可扩展的导出格式注册中心

架构设计：
    每种导出格式实现 ExporterBase 并注册到全局 ExporterRegistry。
    UI（Panel / CLI / API）通过 registry.list_formats() 动态发现可用格式，
    通过 ExporterSpec.fields 动态生成配置表单，无需硬编码。

新增导出格式只需：
    1. 创建 ExporterBase 子类
    2. 调用 registry.register(YourExporter())

无需修改 UI 代码或注册表逻辑。

使用示例：
    from tlabel.export.registry import get_registry

    registry = get_registry()
    # 列出所有可用格式
    for fmt in registry.list_formats():
        print(fmt.name, fmt.fields)

    # 获取指定格式并导出
    exporter = registry.get("ftp1")
    result = exporter.export(data, output_path, sensor_name="GelSightMini")
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ============================================================
# Field Types — UI 控件类型
# ============================================================

class FieldType(Enum):
    """导出配置字段的 UI 控件类型"""
    SELECT = "select"           # 单选下拉
    MULTISELECT = "multiselect" # 多选 + 可选预设
    TEXT = "text"               # 自由文本
    BOOL = "bool"               # 布尔开关
    INT = "int"                 # 整数输入
    FLOAT = "float"             # 浮点数输入


@dataclass
class ExportField:
    """
    单个导出配置字段的描述。

    UI 根据此描述动态生成表单控件。

    Attributes:
        key:         参数键名（传给 export() 的 kwargs key）
        label:       用户可见的标签（如 "传感器类型"）
        field_type:  UI 控件类型
        required:    是否必填
        default:     默认值
        options:     SELECT/MULTISELECT 的可选项
                     格式: ["A", "B"] 或 [{"value": "a", "label": "选项A"}]
        presets:     MULTIPLESELECT 的预设组合
                     格式: {"name": {"label": "显示名", "values": [...]}}
        description: 字段说明，用于 tooltip 或帮助文本
        placeholder: 输入框占位文本
        min_value:   INT/FLOAT 的最小值
        max_value:   INT/FLOAT 的最大值
    """
    key: str
    label: str
    field_type: FieldType = FieldType.TEXT
    required: bool = False
    default: Any = None
    options: Optional[List] = None
    presets: Optional[Dict[str, Dict]] = None
    description: Optional[str] = None
    placeholder: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def to_dict(self) -> dict:
        """序列化为 UI 可消费的 dict"""
        d: Dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "field_type": self.field_type.value,
            "required": self.required,
        }
        if self.default is not None:
            d["default"] = self.default
        if self.options is not None:
            d["options"] = self.options
        if self.presets is not None:
            d["presets"] = self.presets
        if self.description is not None:
            d["description"] = self.description
        if self.placeholder is not None:
            d["placeholder"] = self.placeholder
        if self.min_value is not None:
            d["min_value"] = self.min_value
        if self.max_value is not None:
            d["max_value"] = self.max_value
        return d


# ============================================================
# Export Spec — 导出格式元信息
# ============================================================

@dataclass
class ExporterSpec:
    """
    导出格式的完整元信息。

    Attributes:
        id:          唯一标识（如 "ftp1", "lerobot", "json"）
        name:        用户可见名称（如 "FTP-1 / MTTS Zarr"）
        description: 格式说明
        category:    分类
                     "basic"    = 零配置基础格式（JSON/CSV/HDF5）
                     "ecosystem"= 需要配置的生态集成（FTP-1/LeRobot）
        fields:      配置字段列表（category="basic" 时为空）
        icon:        图标标识或 emoji
        file_ext:    默认文件扩展名（如 ".zarr", ".parquet"）
    """
    id: str
    name: str
    description: str
    category: str = "basic"           # "basic" | "ecosystem"
    fields: List[ExportField] = field(default_factory=list)
    icon: str = "📦"
    file_ext: str = ""

    def to_dict(self) -> dict:
        """序列化为 UI 可消费的 dict"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "icon": self.icon,
            "file_ext": self.file_ext,
            "fields": [f.to_dict() for f in self.fields],
        }


# ============================================================
# Export Result — 导出结果
# ============================================================

@dataclass
class ExporterResult:
    """
    单次导出的结果。

    Attributes:
        output_path:     输出文件/目录路径
        format_id:       格式 ID
        format_name:     格式名称
        file_size_bytes: 产物大小（字节），0 表示未计算
        stats:           格式特定的统计信息
        generated_code:  可复现的 Python 代码片段（可选，用于 UI 展示）
        success:         是否成功
        error:           错误信息（失败时）
    """
    output_path: str = ""
    format_id: str = ""
    format_name: str = ""
    file_size_bytes: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    generated_code: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "format_id": self.format_id,
            "format_name": self.format_name,
            "file_size_bytes": self.file_size_bytes,
            "stats": self.stats,
            "generated_code": self.generated_code,
            "success": self.success,
            "error": self.error,
        }


# ============================================================
# Exporter Base — 导出器基类
# ============================================================

class ExporterBase:
    """
    所有导出器的基类。

    子类必须实现:
        - spec: 返回 ExporterSpec 描述自身能力和配置需求
        - export: 执行导出操作

    子类可选实现:
        - list_targets: 列出可用的导出目标（如 FTP-1 的已知传感器列表）
    """

    def spec(self) -> ExporterSpec:
        """返回此导出器的元信息和配置字段描述"""
        raise NotImplementedError

    def export(
        self,
        data,
        output_path: Union[str, Path],
        **kwargs,
    ) -> ExporterResult:
        """
        执行导出。

        Args:
            data:        TLabelData 实例
            output_path: 输出路径
            **kwargs:    由 spec().fields 定义的配置参数

        Returns:
            ExporterResult
        """
        raise NotImplementedError

    def list_targets(self) -> Dict[str, Any]:
        """
        列出可用的导出目标（可选）。

        例如 FTP-1 导出器可返回已知传感器列表和功能区定义，
        供 UI 动态填充下拉框。

        Returns:
            目标信息 dict，结构由具体导出器定义
        """
        return {}


# ============================================================
# Registry — 导出器注册中心
# ============================================================

class ExporterRegistry:
    """
    导出器注册中心。

    管理所有已注册的导出格式，提供统一查询和导出接口。
    可序列化为 JSON 供 UI 层消费。
    """

    def __init__(self):
        self._exporters: Dict[str, ExporterBase] = {}

    # ---- 注册 ----

    def register(self, exporter: ExporterBase) -> None:
        """注册一个导出器"""
        spec = exporter.spec()
        self._exporters[spec.id] = exporter

    def unregister(self, exporter_id: str) -> None:
        """注销一个导出器"""
        self._exporters.pop(exporter_id, None)

    # ---- 查询 ----

    def get(self, exporter_id: str) -> Optional[ExporterBase]:
        """按 ID 获取导出器"""
        return self._exporters.get(exporter_id)

    def list_formats(self) -> List[ExporterSpec]:
        """列出所有已注册的导出格式（按 category 排序：basic 在前）"""
        exporters = list(self._exporters.values())
        exporters.sort(key=lambda e: (0 if e.spec().category == "basic" else 1, e.spec().name))
        return [e.spec() for e in exporters]

    def list_by_category(self, category: str) -> List[ExporterSpec]:
        """按分类列出导出格式"""
        return [
            e.spec() for e in self._exporters.values()
            if e.spec().category == category
        ]

    # ---- 导出 ----

    def export(
        self,
        exporter_id: str,
        data,
        output_path: Union[str, Path],
        **kwargs,
    ) -> ExporterResult:
        """
        使用指定导出器执行导出。

        Args:
            exporter_id: 导出器 ID
            data:        TLabelData 实例
            output_path: 输出路径
            **kwargs:    导出器配置参数

        Returns:
            ExporterResult

        Raises:
            KeyError: 导出器不存在
        """
        exporter = self._exporters.get(exporter_id)
        if exporter is None:
            available = ", ".join(self._exporters.keys())
            raise KeyError(
                f"Unknown export format: '{exporter_id}'. "
                f"Available: {available}"
            )
        return exporter.export(data, output_path, **kwargs)

    # ---- 序列化（供 UI / API） ----

    def to_dict(self) -> dict:
        """
        序列化为 UI 可消费的完整结构。

        返回格式：
        {
            "formats": [
                {
                    "id": "ftp1",
                    "name": "FTP-1 / MTTS Zarr",
                    "category": "ecosystem",
                    "fields": [...],
                    ...
                },
                ...
            ],
            "targets": {
                "ftp1": { "sensors": {...}, "functional_areas": {...} },
                ...
            }
        }
        """
        formats = []
        targets = {}
        for exporter in self._exporters.values():
            formats.append(exporter.spec().to_dict())
            tgt = exporter.list_targets()
            if tgt:
                targets[exporter.spec().id] = tgt
        return {"formats": formats, "targets": targets}


# ============================================================
# Basic Exporters — 基础格式（零配置，直接导出）
# ============================================================

class _JSONExporter(ExporterBase):
    """TLabel Format v2 JSON — 零配置"""

    def spec(self) -> ExporterSpec:
        return ExporterSpec(
            id="json",
            name="TLabel JSON",
            description="TLabel Format v2 原生 JSON 格式，保留全部标注信息",
            category="basic",
            icon="📄",
            file_ext=".json",
        )

    def export(self, data, output_path, **kwargs) -> ExporterResult:
        from tlabel.export.writer import _export_json
        path = _export_json(data, str(output_path))
        size = Path(path).stat().st_size if Path(path).exists() else 0
        return ExporterResult(
            output_path=path,
            format_id="json",
            format_name="TLabel JSON",
            file_size_bytes=size,
            stats={"num_frames": data.num_frames},
        )


class _CSVExporter(ExporterBase):
    """CSV 平面表 — 零配置"""

    def spec(self) -> ExporterSpec:
        return ExporterSpec(
            id="csv",
            name="CSV",
            description="CSV 平面表，每帧一行，22维特征展开，方便 Excel/ pandas 分析",
            category="basic",
            icon="📊",
            file_ext=".csv",
        )

    def export(self, data, output_path, **kwargs) -> ExporterResult:
        from tlabel.export.writer import _export_csv
        path = _export_csv(data, str(output_path))
        size = Path(path).stat().st_size if Path(path).exists() else 0
        return ExporterResult(
            output_path=path,
            format_id="csv",
            format_name="CSV",
            file_size_bytes=size,
            stats={"num_frames": data.num_frames},
        )


class _HDF5Exporter(ExporterBase):
    """HDF5 科学计算格式 — 零配置"""

    def spec(self) -> ExporterSpec:
        return ExporterSpec(
            id="hdf5",
            name="HDF5",
            description="HDF5 格式，适合科学计算和大规模数据集",
            category="basic",
            icon="🔬",
            file_ext=".h5",
        )

    def export(self, data, output_path, **kwargs) -> ExporterResult:
        from tlabel.export.writer import _export_hdf5
        path = _export_hdf5(data, str(output_path))
        size = Path(path).stat().st_size if Path(path).exists() else 0
        return ExporterResult(
            output_path=path,
            format_id="hdf5",
            format_name="HDF5",
            file_size_bytes=size,
            stats={"num_frames": data.num_frames},
        )


# ============================================================
# Ecosystem Exporters — 生态集成（需配置）
# ============================================================

class _FTP1Exporter(ExporterBase):
    """FTP-1 / MTTS Zarr 导出器"""

    def spec(self) -> ExporterSpec:
        return ExporterSpec(
            id="ftp1",
            name="FTP-1 / MTTS Zarr",
            description="导出为 FTP-1 兼容的 MTTS Zarr 格式，可直接用于基础模型微调或推理",
            category="ecosystem",
            icon="🚀",
            file_ext=".zarr",
            fields=[
                ExportField(
                    key="sensor_name",
                    label="传感器类型",
                    field_type=FieldType.SELECT,
                    required=True,
                    default="GelSightMini",
                    description="选择触觉传感器型号，决定数据解释方式",
                ),
                ExportField(
                    key="functional_areas",
                    label="功能区",
                    field_type=FieldType.MULTISELECT,
                    required=True,
                    default=[0, 1],
                    description="选择要导出的手部功能区",
                    presets={
                        "parallel_gripper": {
                            "label": "平行夹爪",
                            "values": [0, 1],
                        },
                        "three_finger": {
                            "label": "三指手",
                            "values": [0, 1, 2],
                        },
                        "five_finger": {
                            "label": "五指手",
                            "values": [0, 1, 2, 3, 4],
                        },
                        "dexterous_hand": {
                            "label": "灵巧手",
                            "values": list(range(15)),
                        },
                    },
                ),
                ExportField(
                    key="side",
                    label="手侧",
                    field_type=FieldType.SELECT,
                    required=False,
                    default="right",
                    options=[
                        {"value": "left", "label": "左手"},
                        {"value": "right", "label": "右手"},
                    ],
                ),
                ExportField(
                    key="group",
                    label="组名",
                    field_type=FieldType.TEXT,
                    required=False,
                    default="gripper",
                    placeholder="gripper",
                    description="Zarr key 中的 group 标识",
                ),
                ExportField(
                    key="target_image_size",
                    label="图像目标尺寸",
                    field_type=FieldType.INT,
                    required=False,
                    default=224,
                    description="图像缩放到 NxN（仅 image 类传感器）",
                ),
                ExportField(
                    key="store_raw_uint8",
                    label="存储原始 uint8",
                    field_type=FieldType.BOOL,
                    required=False,
                    default=True,
                    description="True=存 uint8（FTP-1 训练格式），False=存 float32 归一化后",
                ),
            ],
        )

    def list_targets(self) -> Dict[str, Any]:
        """返回 FTP-1 已知传感器和功能区定义"""
        from tlabel.converters.ftp1 import (
            FTP1_KNOWN_SENSORS,
            ALL_FUNCTIONAL_AREAS,
        )
        return {
            "sensors": {
                name: info for name, info in FTP1_KNOWN_SENSORS.items()
            },
            "functional_areas": {
                str(k): v for k, v in ALL_FUNCTIONAL_AREAS.items()
            },
        }

    def export(self, data, output_path, **kwargs) -> ExporterResult:
        from tlabel.converters.ftp1 import tlabel_to_ftp1
        stats = tlabel_to_ftp1(data, output_path, **kwargs)
        p = Path(stats["output_path"])
        size = sum(
            f.stat().st_size
            for f in p.rglob("*")
            if f.is_file()
        ) if p.exists() else 0
        code = (
            f"from tlabel.converters.ftp1 import tlabel_to_ftp1\n"
            f"from tlabel.core.loader import load\n\n"
            f'data = load("your_data.json")\n'
            f'stats = tlabel_to_ftp1(\n'
            f'    data, "{stats["output_path"]}",\n'
            f'    sensor_name="{stats["sensor_name"]}",\n'
            f'    functional_areas={stats["functional_areas"]},\n'
            f'    side="{stats["side"]}",\n'
            f'    group="{stats["group"]}",\n'
            f")"
        )
        return ExporterResult(
            output_path=stats["output_path"],
            format_id="ftp1",
            format_name="FTP-1 / MTTS Zarr",
            file_size_bytes=size,
            # converter 用 time_steps 表示帧数，这里补一个统一键，
            # 让所有导出器的 stats 都至少能读到 num_frames。
            stats={**stats, "num_frames": data.num_frames},
            generated_code=code,
        )


class _LeRobotExporter(ExporterBase):
    """LeRobot Parquet 双向转换器"""

    def spec(self) -> ExporterSpec:
        return ExporterSpec(
            id="lerobot",
            name="LeRobot",
            description="将标注写回 LeRobot 数据集的 Parquet 文件，更新 meta/info.json",
            category="ecosystem",
            icon="🤖",
            file_ext=".parquet",
            fields=[
                ExportField(
                    key="tactile_field",
                    label="触觉字段名",
                    field_type=FieldType.TEXT,
                    required=False,
                    default="observation.tactile",
                    description="Parquet 中触觉数据的列名",
                ),
                ExportField(
                    key="action_field",
                    label="动作字段名",
                    field_type=FieldType.TEXT,
                    required=False,
                    default="action",
                    description="Parquet 中动作数据的列名",
                ),
                ExportField(
                    key="overwrite",
                    label="覆盖已有字段",
                    field_type=FieldType.BOOL,
                    required=False,
                    default=False,
                    description="是否覆盖 Parquet 中已存在的同名字段",
                ),
            ],
        )

    def export(self, data, output_path, **kwargs) -> ExporterResult:
        """
        导出到 LeRobot 格式。

        注意：LeRobot 导出是"写回"操作，output_path 应指向
        LeRobot episode 目录（包含 meta/ 和 data/）。
        """
        from tlabel.converters.lerobot import tlabel_to_lerobot

        # 需要先保存 TLabel JSON 作为中间文件
        import tempfile, json
        tmp_json = Path(output_path).parent / ".tlabel_export_tmp.json"
        from tlabel.export.writer import _export_json
        _export_json(data, str(tmp_json))

        try:
            tlabel_to_lerobot(
                str(tmp_json),
                str(output_path),
                tactile_field=kwargs.get("tactile_field", "observation.tactile"),
                action_field=kwargs.get("action_field", "action"),
                overwrite=kwargs.get("overwrite", False),
            )
        finally:
            tmp_json.unlink(missing_ok=True)

        code = (
            f"from tlabel.converters.lerobot import tlabel_to_lerobot\n\n"
            f'tlabel_to_lerobot(\n'
            f'    "annotations.json",\n'
            f'    "{output_path}",\n'
            f'    tactile_field="{kwargs.get("tactile_field", "observation.tactile")}",\n'
            f'    overwrite={kwargs.get("overwrite", False)},\n'
            f")"
        )
        return ExporterResult(
            output_path=str(output_path),
            format_id="lerobot",
            format_name="LeRobot",
            stats={"num_frames": data.num_frames},
            generated_code=code,
        )



class _LeRobotCreateExporter(ExporterBase):
    """从零创建 LeRobot v2.1 数据集导出器"""

    def spec(self) -> ExporterSpec:
        return ExporterSpec(
            id="lerobot_create",
            name="LeRobot (新建)",
            description="从标注数据创建全新的 LeRobot v2.1 数据集",
            category="ecosystem",
            icon="🤖",
            file_ext="",
            fields=[
                ExportField(
                    key="adapter_name",
                    label="适配器名称",
                    field_type=FieldType.SELECT,
                    required=False,
                    default="",
                    description="数据来源适配器名称（用于元数据记录）",
                    options=[],  # 运行时动态填充
                ),
                ExportField(
                    key="task",
                    label="任务名称",
                    field_type=FieldType.TEXT,
                    required=False,
                    default="task",
                    description="LeRobot 数据集的任务名称",
                    placeholder="task",
                ),
                ExportField(
                    key="task_description",
                    label="任务描述",
                    field_type=FieldType.TEXT,
                    required=False,
                    default="",
                    description="任务的详细描述（可选）",
                    placeholder="Task description...",
                ),
                ExportField(
                    key="fps",
                    label="帧率",
                    field_type=FieldType.INT,
                    required=False,
                    default=30,
                    min_value=1,
                    max_value=1000,
                    description="数据采集帧率（FPS）",
                ),
            ],
        )

    def export(self, data, output_path, **kwargs) -> ExporterResult:
        """
        从零创建 LeRobot v2.1 数据集。

        output_path 为输出的数据集目录路径。
        """
        from tlabel.converters.lerobot_export import tlabeldata_to_lerobot

        task_name = kwargs.get("task", "task")
        task_description = kwargs.get("task_description", "")
        adapter_name = kwargs.get("adapter_name", "")

        # episode_index / chunk_id / observation_state_fields 与
        # create_lerobot_dataset 的同名参数语义一致，必须透传，
        # 否则多 episode 场景会全部落到 episode_000000.parquet。
        result = tlabeldata_to_lerobot(
            data,
            str(output_path),
            task_name=task_name,
            task_description=task_description,
            episode_index=kwargs.get("episode_index", 0),
            chunk_id=kwargs.get("chunk_id", 0),
            observation_state_fields=kwargs.get("observation_state_fields"),
        )

        code = (
            f"from tlabel.converters.lerobot_export import tlabeldata_to_lerobot\n\n"
            f'tlabeldata_to_lerobot(\n'
            f'    data,\n'
            f'    "lerobot_dataset",\n'
            f'    task_name="{task_name}",\n'
            f'    task_description="{task_description}",\n'
            f")"
        )
        return ExporterResult(
            output_path=str(output_path),
            format_id="lerobot_create",
            format_name="LeRobot (新建)",
            stats={
                "num_frames": result.get("num_frames", data.num_frames),
                "duration_s": result.get("duration_s", 0.0),
                "tactile_dim": result.get("tactile_dim", 14),
                "num_episodes": result.get("num_episodes", 1),
                "adapter": adapter_name,
            },
            generated_code=code,
        )


# ============================================================
# Future Stubs — 预留扩展位
# ============================================================

class _RLDSExporter(ExporterBase):
    """RLDS (Reinforcement Learning Datasets) — 预留"""

    def spec(self) -> ExporterSpec:
        return ExporterSpec(
            id="rlds",
            name="RLDS (Coming Soon)",
            description="TensorFlow RLDS 格式，用于 Google DeepMind 生态",
            category="ecosystem",
            icon="🧪",
            file_ext=".tfrecord",
        )

    def export(self, data, output_path, **kwargs) -> ExporterResult:
        return ExporterResult(
            success=False,
            error="RLDS export is not yet implemented. Coming in a future release.",
            format_id="rlds",
            format_name="RLDS",
        )


class _ROS2Exporter(ExporterBase):
    """ROS2 Bag — 预留"""

    def spec(self) -> ExporterSpec:
        return ExporterSpec(
            id="ros2",
            name="ROS2 Bag (Coming Soon)",
            description="ROS2 bag 格式，用于实时机器人系统集成",
            category="ecosystem",
            icon="🦾",
            file_ext=".db3",
        )

    def export(self, data, output_path, **kwargs) -> ExporterResult:
        return ExporterResult(
            success=False,
            error="ROS2 export is not yet implemented. Coming in a future release.",
            format_id="ros2",
            format_name="ROS2 Bag",
        )


# ============================================================
# Global Registry Singleton
# ============================================================

_registry: Optional[ExporterRegistry] = None


def get_registry() -> ExporterRegistry:
    """获取全局导出器注册表单例（首次调用时自动注册所有内置格式）"""
    global _registry
    if _registry is None:
        _registry = ExporterRegistry()
        # 基础格式
        _registry.register(_JSONExporter())
        _registry.register(_CSVExporter())
        _registry.register(_HDF5Exporter())
        # 生态集成
        _registry.register(_FTP1Exporter())
        _registry.register(_LeRobotExporter())
        _registry.register(_LeRobotCreateExporter())
        # 预留
        _registry.register(_RLDSExporter())
        _registry.register(_ROS2Exporter())
    return _registry

