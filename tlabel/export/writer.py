"""
TLabel数据导出器

支持格式：JSON (Schema V2) / CSV / HDF5

v0.17 Breaking Change: 只支持 Schema V2 (14维) 导出，移除旧 22 维格式分支。
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Optional

from tlabel.core.types import TLabelData
from tlabel.core.schema import SCHEMA_V2_FIELD_NAMES, TLabelSchemaV2
from tlabel.core.capabilities import normalize_capabilities


# Schema V2 展开列（向量展开为多列）
V2_FLAT_DIMS = [
    "contact", "centroid_x", "centroid_y", "contact_region",
    "force_magnitude", "force_x", "force_y", "force_z",
    "torque_x", "torque_y", "torque_z",
    "slip_event", "slip_vx", "slip_vy",
    "manipulation_phase", "texture_class",
    "object_deformation", "temperature",
    "confidence", "compliance_level",
]


def _flatten_schema_v2(schema: TLabelSchemaV2) -> dict:
    """
    将 TLabelSchemaV2 的14维结构化字段展开为扁平 dict（用于 CSV/HDF5 导出）。
    
    向量字段展开为多列：contact_centroid → centroid_x, centroid_y 等。
    """
    flat = {
        "contact": float(schema.contact),
        "force_magnitude": schema.force_magnitude if schema.force_magnitude is not None else 0.0,
        "slip_event": float(schema.slip_event),
        "confidence": schema.confidence,
        "compliance_level": schema.compliance_level,
    }
    # 展开向量
    if schema.contact_centroid is not None:
        flat["centroid_x"] = schema.contact_centroid[0]
        flat["centroid_y"] = schema.contact_centroid[1]
    else:
        flat["centroid_x"] = 0.0
        flat["centroid_y"] = 0.0
    
    if schema.force_vector is not None:
        flat["force_x"] = schema.force_vector[0]
        flat["force_y"] = schema.force_vector[1]
        flat["force_z"] = schema.force_vector[2]
    else:
        flat["force_x"] = 0.0
        flat["force_y"] = 0.0
        flat["force_z"] = 0.0
    
    if schema.torque_vector is not None:
        flat["torque_x"] = schema.torque_vector[0]
        flat["torque_y"] = schema.torque_vector[1]
        flat["torque_z"] = schema.torque_vector[2]
    else:
        flat["torque_x"] = 0.0
        flat["torque_y"] = 0.0
        flat["torque_z"] = 0.0
    
    if schema.slip_velocity is not None:
        flat["slip_vx"] = schema.slip_velocity[0]
        flat["slip_vy"] = schema.slip_velocity[1]
    else:
        flat["slip_vx"] = 0.0
        flat["slip_vy"] = 0.0
    
    # 枚举字段：存为字符串
    flat["contact_region"] = schema.contact_region or ""
    flat["manipulation_phase"] = schema.manipulation_phase or ""
    flat["texture_class"] = schema.texture_class or ""
    
    # 单值 Optional 字段
    flat["object_deformation"] = schema.object_deformation if schema.object_deformation is not None else 0.0
    flat["temperature"] = schema.temperature if schema.temperature is not None else 0.0
    
    return flat


class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON序列化"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def export_data(data: TLabelData, output_path: str, format: str = "auto"):
    """
    导出TLabelData为文件（Schema V2 格式）
    
    参数:
        data: TLabelData实例
        output_path: 输出路径
        format: "json" | "csv" | "hdf5" | "auto"（根据文件后缀自动判断）
    """
    if format == "auto":
        suffix = Path(output_path).suffix.lower()
        if suffix == ".csv":
            format = "csv"
        elif suffix in (".h5", ".hdf5"):
            format = "hdf5"
        else:
            format = "json"

    if format == "json":
        return _export_json(data, output_path)
    elif format == "csv":
        return _export_csv(data, output_path)
    elif format == "hdf5":
        return _export_hdf5(data, output_path)
    else:
        raise ValueError(f"不支持的导出格式: {format}，可选: json, csv, hdf5, auto")


def _export_json(data: TLabelData, output_path: str):
    """导出为Schema V2 JSON"""
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(".json")

    path.parent.mkdir(parents=True, exist_ok=True)

    result = data.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    return str(path)


def _export_csv(data: TLabelData, output_path: str):
    """导出为CSV平面表（Schema V2 14维展开）"""
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(".csv")

    path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["frame_idx", "timestamp_s", "is_first", "is_last",
               "primitive_label", "primitive_source", "primitive_confidence"] + V2_FLAT_DIMS

    # v0.14: 如果有tactile_events，追加event_type列
    has_events = hasattr(data, 'tactile_events') and data.tactile_events
    if has_events:
        headers.append("event_type")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i, frame in enumerate(data.frames):
            is_first = (i == 0)
            is_last = (i == len(data.frames) - 1)
            # 获取该帧的primitive + source + confidence
            primitive = ""
            primitive_source = ""
            primitive_confidence = ""
            if hasattr(data, 'primitive_annotations') and data.primitive_annotations:
                for p in data.primitive_annotations:
                    if p.start_frame <= frame.frame_idx <= p.end_frame:
                        primitive = p.primitive_name
                        primitive_source = p.source
                        primitive_confidence = round(p.confidence, 4)
                        break

            row_base = [
                frame.frame_idx,
                frame.timestamp_s,
                is_first,
                is_last,
                primitive,
                primitive_source,
                primitive_confidence,
            ]

            # 使用 schema_v2 展开
            sv2 = frame.schema_v2
            flat = _flatten_schema_v2(sv2)
            row_base.extend([
                flat.get(dim, 0.0) if dim not in ("contact_region", "manipulation_phase", "texture_class", "compliance_level")
                else flat.get(dim, "")
                for dim in V2_FLAT_DIMS
            ])

            if has_events:
                frame_events = []
                for ev in data.tactile_events:
                    if ev.contains_frame(frame.frame_idx):
                        frame_events.append(ev.event_type)
                row_base.append("|".join(frame_events) if frame_events else "")

            writer.writerow(row_base)

    return str(path)


def _export_hdf5(data: TLabelData, output_path: str):
    """导出为HDF5格式（Schema V2 14维展开）"""
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "HDF5 export requires h5py. Install with: pip install h5py"
        )
    
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(".h5")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Schema V2 展开列（向量展开为多列，仅数值字段）
    FEATURE_DIMS = [
        "contact", "centroid_x", "centroid_y", "force_magnitude",
        "force_x", "force_y", "force_z",
        "torque_x", "torque_y", "torque_z",
        "slip_event", "slip_vx", "slip_vy",
        "object_deformation", "temperature", "confidence",
    ]
    description = "TLabel Schema V2 tactile features (14 dimensions, vectors expanded)"
    
    with h5py.File(path, "w") as f:
        n_frames = data.num_frames
        n_dims = len(FEATURE_DIMS)
        
        feature_matrix = np.zeros((n_frames, n_dims), dtype=np.float32)
        timestamps = np.zeros(n_frames, dtype=np.float64)
        frame_indices = np.zeros(n_frames, dtype=np.int32)
        is_first_arr = np.zeros(n_frames, dtype=bool)
        is_last_arr = np.zeros(n_frames, dtype=bool)
        
        for i, frame in enumerate(data.frames):
            timestamps[i] = frame.timestamp_s
            frame_indices[i] = frame.frame_idx
            is_first_arr[i] = (i == 0)
            is_last_arr[i] = (i == n_frames - 1)
            
            sv2 = frame.schema_v2
            flat = _flatten_schema_v2(sv2)
            for j, dim in enumerate(FEATURE_DIMS):
                val = flat.get(dim, 0.0)
                if isinstance(val, (int, float)):
                    feature_matrix[i, j] = float(val)
                else:
                    feature_matrix[i, j] = 0.0
        
        f.create_dataset("timestamps", data=timestamps)
        f.create_dataset("frame_indices", data=frame_indices)
        f.create_dataset("is_first", data=is_first_arr)
        f.create_dataset("is_last", data=is_last_arr)
        f.create_dataset("tactile_features", data=feature_matrix)
        
        f["tactile_features"].attrs["feature_names"] = json.dumps(FEATURE_DIMS)
        f["tactile_features"].attrs["description"] = description
        f["tactile_features"].attrs["schema_version"] = "v2"
        
        # 枚举字段存储到 /enums 组
        enum_group = f.create_group("enums")
        enum_fields = ["contact_region", "manipulation_phase", "texture_class", "compliance_level"]
        for ef in enum_fields:
            vals = []
            for frame in data.frames:
                sv2 = frame.schema_v2
                flat = _flatten_schema_v2(sv2)
                vals.append(str(flat.get(ef, "")))
            enum_group.create_dataset(ef, data=vals)
        
        # 2. 元数据组 /metadata
        meta_group = f.create_group("metadata")
        meta_group.attrs["schema_version"] = data.schema_version
        meta_group.attrs["format"] = "tlabel_schema_v2"
        meta_group.attrs["sensor_type"] = data.sensor_type
        meta_group.attrs["sensor_id"] = data.sensor_id or ""
        meta_group.attrs["num_frames"] = n_frames
        meta_group.attrs["duration_s"] = data.duration_s
        
        sensor_info_json = json.dumps(data.sensor_info)
        meta_group.attrs["sensor_info"] = sensor_info_json
        
        episode_info_json = json.dumps(data.episode_info)
        meta_group.attrs["episode_info"] = episode_info_json
        
        capabilities_json = json.dumps(
            normalize_capabilities(data.capabilities, source="hdf5_writer", warn=False)
        )
        meta_group.attrs["capabilities"] = capabilities_json
        
        if data.calibration_params:
            calib_json = json.dumps(data.calibration_params)
            meta_group.attrs["calibration_params"] = calib_json
    
    return str(path)
