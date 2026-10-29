"""
TLabel Format 适配器

用于直接加载 TLabel Format v2 JSON 文件，绕过原始传感器格式转换。

v0.17: 新增 Schema V2 支持 — 可加载/验证/导出14维结构化标注格式。
"""

import json
import warnings
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple

from tlabel.adapters.base import BaseAdapter
from tlabel.core.capabilities import (
    CAPABILITY_KEYS,
    filter_capabilities,
    validate_capabilities,
)
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2, SCHEMA_V2_FIELD_NAMES, VALID_COMPLIANCE_LEVELS


class TLabelAdapter(BaseAdapter):
    """TLabel Format JSON 适配器"""

    def __init__(self):
        # 无参 get_capabilities() 的返回值：load() 成功后缓存本次归一化的
        # caps，未加载过时给保守空声明，避免遍历型调用方拿不到数据源。
        self._last_capabilities: Dict[str, bool] = {"contact": False}

    @property
    def name(self) -> str:
        return "tlabel"
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".json"]
    
    def get_sensor_info(self, file_path: Union[str, Path]) -> Dict:
        """获取传感器信息"""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("sensor", {})

    def get_capabilities(self) -> Dict[str, bool]:
        """返回最近一次 load() 归一化后的能力声明（无数据源时为保守空声明）。

        TLabel 适配器没有固定数据源，能力取决于读的是哪个文件。基类要求
        无参可调用，因此这里返回缓存值；需要按文件读取请用
        ``get_capabilities_for_file()``。
        """
        return dict(self._last_capabilities)

    def get_capabilities_for_file(self, file_path: Union[str, Path]) -> Dict[str, bool]:
        """按文件读取能力声明，并按 13 键白名单归一化。"""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return self._normalize_capabilities(raw.get("capabilities", {}), str(path))

    def _normalize_capabilities(self, raw_caps: Dict, source: str) -> Dict[str, bool]:
        """把文件里的 capabilities 收敛到 13 键，并保留原始值以免信息丢失。

        读宽容、写规范：旧 22 键文件仍可加载，但导出的新文件必须合规。
        被剔除的键通过 warning 提示，并原样保存在
        ``episode_info["legacy_capabilities"]`` 里。
        """
        if not isinstance(raw_caps, dict):
            return {"contact": False}

        cleaned = filter_capabilities(raw_caps)
        dropped = {k: v for k, v in raw_caps.items() if k not in CAPABILITY_KEYS}

        if dropped:
            warnings.warn(
                f"[tlabel] {source} 的 capabilities 含 {len(dropped)} 个非法键，"
                f"已从 capabilities 剔除并保留在 episode_info.legacy_capabilities："
                + ", ".join(sorted(dropped)),
                UserWarning,
                stacklevel=3,
            )

        # contact 是必填键，文件没写时按"未知即不声明"处理。
        cleaned.setdefault("contact", False)

        problems = validate_capabilities(cleaned)
        if problems:
            warnings.warn(
                f"[tlabel] {source} 归一化后仍不合规：" + "；".join(problems),
                UserWarning,
                stacklevel=3,
            )

        return cleaned

    def extract_schema(self, raw_frame_data: Union[TLabelFrame, Dict]) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2 (14维结构化)

        TLabel Format适配器策略:
          - 如果frame已有schema_v2，直接返回
          - 否则从tlabel_v2自动转换（使用from_tlabel_v1通用映射）
          - compliance_level从schema_v2中读取，否则默认L1

        Args:
            raw_frame_data: TLabelFrame实例或tlabel_v2字典

        Returns:
            TLabelSchemaV2 — 14维结构化标注
        """
        if isinstance(raw_frame_data, TLabelFrame):
            return raw_frame_data.to_schema_v2()
        elif isinstance(raw_frame_data, dict):
            v1_dict = dict(raw_frame_data)
            v1_dict["confidence"] = v1_dict.get("confidence", 1.0)
            return TLabelSchemaV2.from_tlabel_v1(v1_dict)
        else:
            raise TypeError(f"raw_frame_data 类型不支持: {type(raw_frame_data)}")

    def load(self, file_path: Union[str, Path], **kwargs) -> TLabelData:
        """
        从 TLabel Format JSON 文件加载数据
        
        参数:
            file_path: JSON 文件路径
            **kwargs: 额外参数（未使用）
        
        返回:
            TLabelData 对象
        """
        path = Path(file_path)
        
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        # 解析 frames
        frames = []
        for fd in raw.get("frames", []):
            # v0.17: 尝试加载 schema_v2 字段
            schema_v2 = None
            if "schema_v2" in fd:
                schema_v2 = TLabelSchemaV2.from_dict(fd["schema_v2"])

            frame = TLabelFrame(
                frame_idx=fd["frame_idx"],
                timestamp_s=fd["timestamp_s"],
                schema_v2=schema_v2 if schema_v2 is not None else TLabelSchemaV2.from_tlabel_v1(fd.get("tlabel_v2", {})),
                manipulation_phase=fd.get("manipulation_phase", "idle"),
                confidence=fd.get("confidence", 1.0),
                sensor_specific=fd.get("sensor_specific"),
            )
            frames.append(frame)
        
        # 构建 TLabelData
        # 读宽容、写规范：caps 归一化到 13 键，被剔除的旧键存进 episode_info
        # 以免信息丢失；归一化结果同时缓存给无参 get_capabilities()。
        episode_info = dict(raw.get("episode", {}) or {})
        raw_caps = raw.get("capabilities", {}) or {}
        clean_caps = self._normalize_capabilities(raw_caps, str(file_path))
        dropped_caps = {k: v for k, v in raw_caps.items() if k not in CAPABILITY_KEYS}
        if dropped_caps:
            episode_info["legacy_capabilities"] = dropped_caps
        self._last_capabilities = dict(clean_caps)

        data = TLabelData(
            frames=frames,
            sensor_info=raw.get("sensor", {}),
            episode_info=episode_info,
            capabilities=clean_caps,
            schema_version=raw.get("schema_version", "0.4.0"),
        )
        
        # v0.13: 加载 primitive_annotations（向后兼容：旧JSON没有此字段也能正常加载）
        if "primitive_annotations" in raw:
            from tlabel.core.primitive import PrimitiveAnnotation
            for pa_dict in raw["primitive_annotations"]:
                try:
                    pa = PrimitiveAnnotation(
                        name=pa_dict["primitive_name"],
                        start=pa_dict["start_frame"],
                        end=pa_dict["end_frame"],
                        confidence=pa_dict.get("confidence", 1.0),
                        source=pa_dict.get("source", "manual"),
                    )
                    data.primitive_annotations.append(pa)
                except (ValueError, KeyError):
                    pass
        
        # v0.17: 如果JSON中包含schema_v2_frames，加载到各帧
        if "schema_v2_frames" in raw:
            sv2_list = raw["schema_v2_frames"]
            for i, sv2_dict in enumerate(sv2_list):
                if i < len(data.frames):
                    data.frames[i].schema_v2 = TLabelSchemaV2.from_dict(sv2_dict)
        
        return data


# ============================================================
# Schema V2 辅助函数
# ============================================================

def validate_schema_v2_dict(schema_dict: Dict) -> Tuple[bool, List[str]]:
    """验证 Schema V2 字典的完整性和合规性

    Args:
        schema_dict: 包含14维 Schema V2 字段的字典

    Returns:
        (is_valid, errors): is_valid=True 表示通过；errors 是错误消息列表
    """
    schema = TLabelSchemaV2.from_dict(schema_dict)
    return schema.validate()


def validate_schema_v2_json(json_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """验证 JSON 文件中的 Schema V2 数据

    支持两种格式:
    1. 独立 Schema V2 JSON: 顶层包含 schema_v2 字段或直接是14维字段
    2. TLabel Format JSON: frames[].schema_v2 中包含14维字段

    Args:
        json_path: JSON 文件路径

    Returns:
        (is_valid, errors): is_valid=True 表示通过；errors 是错误消息列表
    """
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    all_errors = []

    # 格式1: 独立 Schema V2 JSON（顶层直接包含14维字段）
    if "compliance_level" in raw:
        is_valid, errors = validate_schema_v2_dict(raw)
        if not is_valid:
            all_errors.extend(errors)
        return (len(all_errors) == 0, all_errors)

    # 格式2: 包含 schema_v2 字段
    if "schema_v2" in raw:
        is_valid, errors = validate_schema_v2_dict(raw["schema_v2"])
        if not is_valid:
            all_errors.extend(errors)
        return (len(all_errors) == 0, all_errors)

    # 格式3: TLabel Format JSON（从 frames 中提取）
    frames = raw.get("frames", [])
    if not frames:
        all_errors.append("JSON文件不包含任何帧数据")
        return (False, all_errors)

    for i, fd in enumerate(frames):
        if "schema_v2" not in fd:
            # 从 tlabel_v2 自动转换后验证
            tlabel_v2 = fd.get("tlabel_v2", {})
            confidence = fd.get("confidence", 1.0)
            tlabel_v2["confidence"] = confidence
            schema = TLabelSchemaV2.from_tlabel_v1(tlabel_v2)
        else:
            schema = TLabelSchemaV2.from_dict(fd["schema_v2"])

        is_valid, errors = schema.validate()
        if not is_valid:
            all_errors.extend([f"frame[{i}]: {e}" for e in errors])

    return (len(all_errors) == 0, all_errors)


def convert_tlabel_v1_to_v2(tlabel_v2_dict: Dict,
                             confidence: float = 1.0,
                             compliance_level: str = "L1") -> TLabelSchemaV2:
    """将22维 tlabel_v2 字典转换为14维 Schema V2 结构化标注

    便捷函数，用于将旧格式数据一次性转换为新Schema。

    Args:
        tlabel_v2_dict: 22维 tlabel_v2 flat dict
        confidence: 帧级置信度
        compliance_level: 指定 Compliance Level (L1-L4)

    Returns:
        TLabelSchemaV2 — 14维结构化标注
    """
    v1_dict = dict(tlabel_v2_dict)
    v1_dict["confidence"] = confidence
    schema = TLabelSchemaV2.from_tlabel_v1(v1_dict)
    schema.compliance_level = compliance_level
    return schema


def export_schema_v2(data: TLabelData, output_path: Union[str, Path],
                     include_tlabel_v2: bool = True) -> Dict:
    """将 TLabelData 导出为包含 Schema V2 的 JSON 文件

    在原有 TLabel Format 输出基础上，每帧追加 schema_v2 字段。
    同时在顶层输出 schema_v2_frames 列表，方便批量读取。

    Args:
        data: TLabelData 实例
        output_path: 输出 JSON 文件路径
        include_tlabel_v2: 是否同时保留旧的22维 tlabel_v2（向后兼容）

    Returns:
        导出统计信息 dict
    """
    # 获取基础 dict（复用 TLabelData.to_dict()）
    base = data.to_dict()

    # 为每帧添加 schema_v2
    schema_v2_frames = []
    frames_with_schema = []

    for i, frame in enumerate(data.frames):
        schema = frame.to_schema_v2()
        sv2_dict = schema.to_dict()
        schema_v2_frames.append(sv2_dict)

        frame_dict = base["frames"][i] if i < len(base.get("frames", [])) else {}
        if include_tlabel_v2:
            frame_dict["schema_v2"] = sv2_dict
        else:
            # 仅输出 schema_v2，移除旧的 tlabel_v2
            frame_dict = {
                "frame_idx": frame_dict.get("frame_idx", i),
                "timestamp_s": frame_dict.get("timestamp_s", 0.0),
                "schema_v2": sv2_dict,
                "manipulation_phase": frame_dict.get("manipulation_phase", "idle"),
                "confidence": frame_dict.get("confidence", 1.0),
            }
        frames_with_schema.append(frame_dict)

    base["frames"] = frames_with_schema
    base["schema_v2_frames"] = schema_v2_frames

    # 写入文件
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

    # 统计
    level_counts = {}
    for sv2 in schema_v2_frames:
        cl = sv2.get("compliance_level", "L1")
        level_counts[cl] = level_counts.get(cl, 0) + 1

    return {
        "output_path": str(path),
        "total_frames": len(schema_v2_frames),
        "compliance_level_distribution": level_counts,
        "include_tlabel_v2": include_tlabel_v2,
    }


def get_compliance_level_summary(data: TLabelData) -> Dict[str, int]:
    """获取 TLabelData 中各帧的 Compliance Level 分布

    Args:
        data: TLabelData 实例

    Returns:
        {"L1": count, "L2": count, "L3": count, "L4": count}
    """
    level_counts = {level: 0 for level in VALID_COMPLIANCE_LEVELS}
    for frame in data.frames:
        schema = frame.to_schema_v2()
        cl = schema.compliance_level
        if cl in level_counts:
            level_counts[cl] += 1
    return level_counts
