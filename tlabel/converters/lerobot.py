"""
LeRobot ↔ TLabel 双向转换器

v0.17 Breaking Change: 只使用 Schema V2 (14维) 格式。

用法:
    # LeRobot → TLabel
    from tlabel.converters import lerobot_to_tlabel
    data = lerobot_to_tlabel("path/to/lerobot_episode/")

    # TLabel → LeRobot
    from tlabel.converters import tlabel_to_lerobot
    tlabel_to_lerobot("tlabel_annotations.json", "path/to/lerobot_episode/")
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    raise ImportError(
        "LeRobot converter requires pyarrow. Install with: pip install pyarrow"
    )

from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2


def lerobot_to_tlabel(
    lerobot_path: Union[str, Path],
    tactile_field: str = "observation.tactile",
    output_format: str = "tlabel"
) -> TLabelData:
    """从 LeRobot Parquet 文件加载触觉数据，转换为 TLabelData（Schema V2）"""
    path = Path(lerobot_path)
    
    # 1. 读取 meta/info.json
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot meta file not found: {info_path}")
    
    with open(info_path, "r") as f:
        meta = json.load(f)
    
    # 2. 读取 Parquet 文件
    data_dir = path / "data"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    tables = [pq.read_table(pf) for pf in parquet_files]
    table = pa.concat_tables(tables)
    df = table.to_pandas()
    
    # 3. 提取触觉字段
    if tactile_field not in df.columns:
        available = [c for c in df.columns if "tactile" in c.lower() or "force" in c.lower()]
        raise ValueError(
            f"Tactile field '{tactile_field}' not found in parquet.\n"
            f"Available columns: {available[:10]}..."
        )
    
    tactile_data = df[tactile_field].values
    
    # 4. 构建 TLabelFrame 列表
    frames = []
    for i, tact in enumerate(tactile_data):
        if isinstance(tact, (list, np.ndarray)):
            tact_array = np.array(tact)
        else:
            tact_array = np.array([float(tact)])
        
        # 构建 Schema V2
        feature_names = meta.get("features", {}).get(tactile_field, {}).get("feature_names", [])
        schema_v2 = TLabelSchemaV2()  # 默认值
        
        if len(feature_names) == len(tact_array):
            for name, val in zip(feature_names, tact_array):
                fval = float(val)
                if name == "contact":
                    schema_v2.contact = fval > 0.5
                elif name == "force_magnitude":
                    schema_v2.force_magnitude = fval
                elif name == "slip_event":
                    schema_v2.slip_event = fval > 0.5
                elif name == "object_deformation":
                    schema_v2.object_deformation = fval
                elif name == "confidence":
                    schema_v2.confidence = fval
                elif name == "temperature":
                    schema_v2.temperature = fval
        
        timestamp = float(df["timestamp"].iloc[i]) if "timestamp" in df.columns else i * 0.033
        
        frame = TLabelFrame(
            frame_idx=i,
            timestamp_s=round(timestamp, 4),
            schema_v2=schema_v2,
            manipulation_phase="idle",
            confidence=1.0,
        )
        frames.append(frame)
    
    # 5. 构建 sensor_info
    sensor_info = {
        "type": "lerobot_tactile",
        "model": meta.get("robot_type", "unknown"),
        "source": "LeRobot Parquet",
    }
    
    episode_info = {
        "episode_id": meta.get("episode_id", path.stem),
        "task": meta.get("task", ""),
    }
    
    capabilities = {name: True for name in feature_names} if feature_names else {}
    
    return TLabelData(
        frames=frames,
        sensor_info=sensor_info,
        episode_info=episode_info,
        capabilities=capabilities,
        sensor_id=f"lerobot_{path.stem}",
    )


def tlabel_to_lerobot(
    tlabel_path: Union[str, Path],
    lerobot_path: Union[str, Path],
    tactile_field: str = "observation.tactile",
    action_field: str = "action",
    overwrite: bool = False,
    image_shape: Optional[tuple] = None,
    adapter: Optional[object] = None,
):
    """将 TLabel 标注写回 LeRobot Parquet 文件（Schema V2 格式）

    参数:
        tlabel_path: TLabel JSON 文件路径
        lerobot_path: LeRobot 数据集目录路径
        tactile_field: 触觉字段名
        action_field: 动作字段名（保留）
        overwrite: 是否覆盖已有字段
        image_shape: 可选，手动指定触觉图像形状 (H, W, C)。
                     如果未提供且提供了 adapter，将调用 adapter.detect_image_shape()。
        adapter: 可选，传感器/数据集适配器实例，用于自动检测图像形状。
    """
    tlabel_path = Path(tlabel_path)
    lerobot_path = Path(lerobot_path)

    # 1. 读取 TLabel JSON
    with open(tlabel_path, "r") as f:
        tlabel_data = json.load(f)

    frames = tlabel_data.get("frames", [])
    if not frames:
        raise ValueError("TLabel JSON has no frames")

    # 2. 读取现有 Parquet
    data_dir = lerobot_path / "data"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    tables = [pq.read_table(pf) for pf in parquet_files]
    table = pa.concat_tables(tables)
    df = table.to_pandas()

    # 3. 验证帧数匹配
    if len(df) != len(frames):
        raise ValueError(
            f"Frame count mismatch: LeRobot has {len(df)} frames, "
            f"TLabel has {len(frames)} frames"
        )

    # 4. 从 Schema V2 构建触觉特征矩阵
    feature_names = tlabel_data.get("feature_names_v2", [])
    if not feature_names:
        # 从第一帧的 schema_v2 推断
        sv2 = frames[0].get("schema_v2", {})
        feature_names = list(sv2.keys()) if sv2 else []

    # 构建触觉特征矩阵
    tactile_matrix = []
    for frame in frames:
        sv2 = frame.get("schema_v2", {})
        row = []
        for name in feature_names:
            val = sv2.get(name, 0.0)
            # 只转换数值类型，非数值字段用 0.0 填充
            if val is None:
                row.append(0.0)
            elif isinstance(val, (int, float)):
                row.append(float(val))
            elif isinstance(val, bool):
                row.append(1.0 if val else 0.0)
            else:
                # 字符串或其他类型用 0.0
                row.append(0.0)
        tactile_matrix.append(row)

    tactile_array = np.array(tactile_matrix, dtype=np.float32)

    # 5. 添加到 DataFrame
    if tactile_field in df.columns and not overwrite:
        raise ValueError(
            f"Field '{tactile_field}' already exists. Use overwrite=True to replace."
        )

    df[tactile_field] = list(tactile_array)

    # 6. 写回 Parquet
    output_parquet = data_dir / "chunk-0000.parquet"
    new_table = pa.Table.from_pandas(df)
    pq.write_table(new_table, output_parquet)

    for pf in parquet_files:
        if pf != output_parquet:
            pf.unlink()

    # 7. 更新 meta/info.json
    info_path = lerobot_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {"features": {}}

    # 7a. 触觉特征 shape
    meta["features"][tactile_field] = {
        "dtype": "float32",
        "shape": [len(feature_names)],
        "type": "vector",
        "feature_names": feature_names,
        "description": "TLabel annotated tactile features (14 dimensions, Schema V2)",
    }

    # 7b. 触觉图像 shape（如果提供了 image_shape 或 adapter）
    if image_shape is None and adapter is not None:
        if hasattr(adapter, 'detect_image_shape'):
            image_shape = adapter.detect_image_shape()

    if image_shape is not None:
        image_field = tactile_field.replace(".tactile", ".tactile_image")
        if image_field == tactile_field:
            image_field = "observation.tactile_image"
        meta["features"][image_field] = {
            "dtype": "uint8",
            "shape": list(image_shape),
        }

    with open(info_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Tactile annotations written to {output_parquet}")
    if image_shape is not None:
        print(f"[OK] Tactile image shape detected: {image_shape}")
    print(f"[OK] Updated meta/info.json with '{tactile_field}' schema")


def detect_image_shape_for_lerobot(
    lerobot_path: Union[str, Path],
    adapter: Optional[object] = None,
    tactile_field: str = "observation.tactile",
) -> Optional[tuple]:
    """从 LeRobot 数据集或适配器检测触觉图像形状

    优先级:
    1. adapter.detect_image_shape() 如果提供了 adapter
    2. meta/info.json 中的 image shape 字段
    3. 从 parquet 数据中采样推断

    参数:
        lerobot_path: LeRobot 数据集目录路径
        adapter: 可选，传感器/数据集适配器实例
        tactile_field: 触觉字段名

    返回:
        (height, width, channels) 或 None
    """
    path = Path(lerobot_path)

    # 1. 从 adapter 获取
    if adapter is not None and hasattr(adapter, 'detect_image_shape'):
        shape = adapter.detect_image_shape()
        if shape is not None:
            return shape

    # 2. 从 meta/info.json 获取
    info_path = path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            meta = json.load(f)

        features = meta.get("features", {})
        # 查找可能的图像字段
        for key, feat in features.items():
            if "image" in key.lower() or "tactile_image" in key.lower():
                shape = feat.get("shape")
                if shape and len(shape) >= 2:
                    return tuple(shape)

    # 3. 从 parquet 数据采样推断
    data_dir = path / "data"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if parquet_files:
        try:
            table = pq.read_table(parquet_files[0])
            df = table.to_pandas()
            if tactile_field in df.columns:
                sample = df[tactile_field].iloc[0]
                if isinstance(sample, (list, np.ndarray)):
                    arr = np.array(sample)
                    if arr.ndim >= 1:
                        return arr.shape
        except Exception:
            pass

    return None


__all__ = ["lerobot_to_tlabel", "tlabel_to_lerobot", "detect_image_shape_for_lerobot"]
