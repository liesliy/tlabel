"""
LeRobot Exporter — 从零创建 LeRobot v2.1 数据集

与 tlabel/converters/lerobot.py 不同（后者是"写回"操作，
要求已存在 LeRobot 数据集），本模块从原始传感器数据
直接生成完整的 LeRobot v2.1 目录结构。

LeRobot v2.1 格式:
    output_dir/
    ├── data/
    │   └── chunk-000/
    │       └── episode_000000.parquet
    └── meta/
        ├── info.json          # features 定义
        ├── episodes.jsonl     # episode 元数据
        └── tasks.jsonl        # 任务定义

用法:
    from tlabel.converters.lerobot_export import create_lerobot_dataset
    create_lerobot_dataset(
        input_path="raw_data.hdf5",
        output_path="./lerobot_dataset",
        adapter_name="tashan_ts_f_a",
    )
"""

import json
import math
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError(
        "LeRobot exporter requires pyarrow. Install with: pip install pyarrow"
    )

from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2, SCHEMA_V2_FIELD_NAMES
from tlabel.converters._encoding import encode_schema_v2
from tlabel.core.capabilities import normalize_capabilities


# ================================================================
# Schema V2 → 14 维 float32 向量编码
# ================================================================

def _encode_schema_v2_to_vector(sv2: TLabelSchemaV2) -> List[float]:
    """
    将 TLabelSchemaV2 编码为 14 维 float32 向量，
    与 SCHEMA_V2_FIELD_NAMES 顺序一致。

    编码规则由 ``tlabel/converters/_encoding.py`` 统一提供，与写回路径
    ``tlabel/converters/lerobot.py`` 共用同一实现（v0.23.3 起）。

    编码规则：
      - bool: 1.0 / 0.0
      - 标量 float: 直接取值，None → 0.0（confidence 缺省 1.0）
      - 向量 (list/tuple): 取 L2 范数，None → 0.0
      - str 枚举: 映射为类别索引，None / 未知 → -1.0
    """
    return encode_schema_v2(sv2)


# ================================================================
# Parquet 构建
# ================================================================

def _build_parquet_data(
    data: TLabelData,
    episode_index: int = 0,
    observation_state: Optional[List[str]] = None,
) -> pa.Table:
    """
    从 TLabelData 构建 PyArrow Table。

    Parquet Schema 列:
      - observation.tactile: float32[14] — Schema V2 14维
      - timestamp: float32
      - episode_index: int64
      - frame_index: int64

    参数:
        data: TLabelData 实例
        episode_index: 当前 episode 编号（从 0 开始）
        observation_state: 可选，从 sensor_specific 中提取的关节状态字段名列表

    返回:
        pa.Table
    """
    frames = data.frames
    n = len(frames)
    if n == 0:
        raise ValueError("TLabelData has no frames — cannot create empty dataset")

    # 1. 构建 tactile 特征矩阵 (n, 14) float32
    tactile_rows: List[List[float]] = []
    for frame in frames:
        sv2 = frame.schema_v2
        if sv2 is None:
            # 兜底：空 schema
            sv2 = TLabelSchemaV2()
        tactile_rows.append(_encode_schema_v2_to_vector(sv2))

    tactile_array = pa.array(
        tactile_rows,
        type=pa.list_(pa.float32(), list_size=14),
    )

    # 2. timestamp
    timestamps = [float(f.timestamp_s) for f in frames]
    timestamp_array = pa.array(timestamps, type=pa.float32())

    # 3. episode_index
    episode_array = pa.array([episode_index] * n, type=pa.int64())

    # 4. frame_index
    frame_indices = [int(f.frame_idx) for f in frames]
    frame_index_array = pa.array(frame_indices, type=pa.int64())

    # 5. 组装 table
    columns = [
        ("observation.tactile", tactile_array),
        ("timestamp", timestamp_array),
        ("episode_index", episode_array),
        ("frame_index", frame_index_array),
    ]

    # 6. 可选 observation.state（从 sensor_specific 提取）
    if observation_state:
        state_rows: List[List[float]] = []
        for frame in frames:
            ss = frame.sensor_specific or {}
            row = [float(ss.get(k, 0.0)) for k in observation_state]
            state_rows.append(row)
        state_dim = len(observation_state)
        state_array = pa.array(
            state_rows,
            type=pa.list_(pa.float32(), list_size=state_dim),
        )
        columns.append(("observation.state", state_array))

    # 7. 可选 action（从 sensor_specific 提取）
    action_fields = []
    # 从第一帧 sensor_specific 中检测 action.* 字段
    if frames and frames[0].sensor_specific:
        ss0 = frames[0].sensor_specific
        action_fields = sorted(
            k for k in ss0.keys()
            if k.startswith("action.") or k == "action"
        )
    if action_fields:
        action_rows: List[List[float]] = []
        for frame in frames:
            ss = frame.sensor_specific or {}
            row = [float(ss.get(k, 0.0)) for k in action_fields]
            action_rows.append(row)
        action_dim = len(action_fields)
        action_array = pa.array(
            action_rows,
            type=pa.list_(pa.float32(), list_size=action_dim),
        )
        columns.append(("action", action_array))

    # 构建 table
    names = [c[0] for c in columns]
    arrays = [c[1] for c in columns]
    table = pa.table(arrays, names=names)
    return table


# ================================================================
# Meta 文件生成
# ================================================================

def _generate_info_json(
    data: TLabelData,
    tactile_dim: int = 14,
    state_dim: int = 0,
    action_dim: int = 0,
    task_description: str = "",
) -> Dict[str, Any]:
    """
    生成 LeRobot v2.1 meta/info.json 内容。

    包含 features 定义、统计信息和传感器元数据。
    """
    features: Dict[str, Any] = {}

    # observation.tactile
    features["observation.tactile"] = {
        "dtype": "float32",
        "shape": [tactile_dim],
        "type": "vector",
        "feature_names": list(SCHEMA_V2_FIELD_NAMES),
        "description": "TLabel Schema V2 tactile features (14 dimensions)",
    }

    # observation.state
    if state_dim > 0:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [state_dim],
            "type": "vector",
            "description": "Robot joint state (from sensor_specific)",
        }

    # action
    if action_dim > 0:
        features["action"] = {
            "dtype": "float32",
            "shape": [action_dim],
            "type": "vector",
            "description": "Action commands (from sensor_specific)",
        }

    # timestamp
    features["timestamp"] = {
        "dtype": "float32",
        "shape": [],
        "type": "scalar",
        "description": "Frame timestamp in seconds",
    }

    # episode_index
    features["episode_index"] = {
        "dtype": "int64",
        "shape": [],
        "type": "scalar",
        "description": "Episode index",
    }

    # frame_index
    features["frame_index"] = {
        "dtype": "int64",
        "shape": [],
        "type": "scalar",
        "description": "Frame index within episode",
    }

    # 顶层信息
    info = {
        "codebase_version": "lerobot-v2.1",
        "task": task_description or data.episode_info.get("task", "tactile_episode"),
        "robot_type": data.sensor_info.get("type", "unknown"),
        "total_episodes": 1,
        "total_frames": data.num_frames,
        "total_duration_s": round(data.duration_s, 4),
        "features": features,
        "data_path": "data/chunk-000/episode_000000.parquet",
        "source_adapter": data.sensor_info.get("adapter", data.episode_info.get("source_adapter", "")),
        "sensor_info": data.sensor_info,
        "capabilities": normalize_capabilities(
            data.capabilities, source="lerobot_export", warn=False
        ),
    }

    return info


def _generate_episodes_jsonl(
    data: TLabelData,
    episode_index: int = 0,
    task_name: str = "",
) -> List[Dict[str, Any]]:
    """
    生成 meta/episodes.jsonl 内容（LeRobot v2.1 格式）。
    """
    episode = {
        "episode_index": episode_index,
        "frame_index": list(range(data.num_frames)) if data.num_frames <= 100 else None,
        "timestamp": (
            [round(f.timestamp_s, 4) for f in data.frames]
            if data.num_frames <= 100 else None
        ),
        "duration": round(data.duration_s, 4),
        "num_frames": data.num_frames,
        "task": task_name or data.episode_info.get("task", "tactile_episode"),
        "episode_id": data.episode_info.get("episode_id", f"episode_{episode_index:06d}"),
        "sensor_id": data.sensor_id or "",
    }
    # 帧数过多时省略 frame_index 列表，减小文件体积
    if episode["frame_index"] is None:
        del episode["frame_index"]
    if episode["timestamp"] is None:
        del episode["timestamp"]
    return [episode]


def _generate_tasks_jsonl(task_name: str = "", task_description: str = "") -> List[Dict[str, Any]]:
    """
    生成 meta/tasks.jsonl 内容（LeRobot v2.1 格式）。
    """
    task = {
        "task": task_name or "tactile_episode",
        "description": task_description or "TLabel exported tactile episode",
        "language": "en",
        "instruction": task_name or "tactile_episode",
    }
    return [task]


# ================================================================
# 主函数
# ================================================================

def create_lerobot_dataset(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    adapter_name: str,
    task_name: str = "",
    task_description: str = "",
    episode_id: str = "",
    episode_index: int = 0,
    chunk_id: int = 0,
    observation_state_fields: Optional[List[str]] = None,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    从原始传感器数据创建完整的 LeRobot v2.1 数据集。

    参数:
        input_path: 原始传感器数据文件路径
        output_path: 输出的 LeRobot 数据集目录路径
        adapter_name: 适配器名称（如 "tashan_ts_f_a"、"gelsight"）
        task_name: 任务名称（用于 meta/tasks.jsonl 和 episodes.jsonl）
        task_description: 任务描述
        episode_id: 自定义 episode ID（默认自动生成）
        episode_index: episode 编号（默认 0）
        chunk_id: chunk 编号（默认 0）
        observation_state_fields: 可选，从 sensor_specific 提取的关节状态字段列表
        adapter_kwargs: 传递给适配器 load() 的额外参数

    返回:
        dict: 导出统计信息
            - output_path: 输出目录路径
            - num_frames: 帧数
            - num_episodes: episode 数
            - duration_s: 时长（秒）
            - tactile_dim: 触觉特征维度
            - parquet_file: parquet 文件路径
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    adapter_kwargs = adapter_kwargs or {}

    # 1. 使用适配器加载原始数据
    from tlabel.core.registry import _ensure_adapters, get_adapter

    _ensure_adapters()
    adapter_cls = get_adapter(adapter_name)
    if adapter_cls is None:
        raise ValueError(
            f"Adapter '{adapter_name}' not found. "
            f"Use 'tlabel list' to see available adapters."
        )

    adapter = adapter_cls()
    data = adapter.load(str(input_path), **adapter_kwargs)
    if data is None:
        raise ValueError(f"Adapter '{adapter_name}' returned None for {input_path}")

    # 2. 构建目录结构
    data_dir = output_path / "data" / f"chunk-{chunk_id:03d}"
    meta_dir = output_path / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 3. 构建并写入 Parquet
    parquet_file = data_dir / f"episode_{episode_index:06d}.parquet"
    table = _build_parquet_data(
        data,
        episode_index=episode_index,
        observation_state=observation_state_fields,
    )
    pq.write_table(table, parquet_file)

    # 4. 计算 state_dim 和 action_dim
    state_dim = len(observation_state_fields) if observation_state_fields else 0
    action_dim = 0
    if data.num_frames > 0 and data.frames[0].sensor_specific:
        ss0 = data.frames[0].sensor_specific
        action_dim = len(
            [k for k in ss0.keys() if k.startswith("action.") or k == "action"]
        )

    # 5. 生成 meta/info.json
    info = _generate_info_json(
        data,
        tactile_dim=14,
        state_dim=state_dim,
        action_dim=action_dim,
        task_description=task_name or task_description,
    )
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # 6. 生成 meta/episodes.jsonl
    task_name_final = task_name or data.episode_info.get("task", "tactile_episode")
    episodes = _generate_episodes_jsonl(
        data,
        episode_index=episode_index,
        task_name=task_name_final,
    )
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # 7. 生成 meta/tasks.jsonl
    tasks = _generate_tasks_jsonl(
        task_name=task_name_final,
        task_description=task_description,
    )
    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    stats = {
        "output_path": str(output_path),
        "num_frames": data.num_frames,
        "num_episodes": 1,
        "duration_s": round(data.duration_s, 4),
        "tactile_dim": 14,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "parquet_file": str(parquet_file),
        "adapter_name": adapter_name,
    }

    print(f"[OK] LeRobot dataset created at {output_path}")
    print(f"     Frames: {stats['num_frames']}")
    print(f"     Duration: {stats['duration_s']}s")
    print(f"     Tactile dim: {stats['tactile_dim']}")
    if state_dim > 0:
        print(f"     State dim: {state_dim}")
    if action_dim > 0:
        print(f"     Action dim: {action_dim}")
    print(f"     Parquet: {parquet_file}")

    return stats


# ================================================================
# 便捷接口：从 TLabelData 直接导出
# ================================================================

def tlabeldata_to_lerobot(
    data: TLabelData,
    output_path: Union[str, Path],
    task_name: str = "",
    task_description: str = "",
    episode_index: int = 0,
    chunk_id: int = 0,
    observation_state_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    从 TLabelData 对象直接导出为 LeRobot v2.1 数据集。

    与 create_lerobot_dataset 的区别：
    - create_lerobot_dataset: 从原始数据文件 + 适配器 → LeRobot
    - tlabeldata_to_lerobot: 从 TLabelData → LeRobot（跳过适配器步骤）

    参数:
        data: TLabelData 实例
        output_path: 输出的 LeRobot 数据集目录路径
        task_name: 任务名称
        task_description: 任务描述
        episode_index: episode 编号
        chunk_id: chunk 编号
        observation_state_fields: 可选，从 sensor_specific 提取的关节状态字段

    返回:
        dict: 导出统计信息
    """
    output_path = Path(output_path)

    # 构建目录
    data_dir = output_path / "data" / f"chunk-{chunk_id:03d}"
    meta_dir = output_path / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 构建 Parquet
    parquet_file = data_dir / f"episode_{episode_index:06d}.parquet"
    table = _build_parquet_data(
        data,
        episode_index=episode_index,
        observation_state=observation_state_fields,
    )
    pq.write_table(table, parquet_file)

    # 计算维度
    state_dim = len(observation_state_fields) if observation_state_fields else 0
    action_dim = 0
    if data.num_frames > 0 and data.frames[0].sensor_specific:
        ss0 = data.frames[0].sensor_specific
        action_dim = len(
            [k for k in ss0.keys() if k.startswith("action.") or k == "action"]
        )

    # 生成 meta/info.json
    info = _generate_info_json(
        data,
        tactile_dim=14,
        state_dim=state_dim,
        action_dim=action_dim,
        task_description=task_name or task_description,
    )
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # 生成 meta/episodes.jsonl
    task_name_final = task_name or data.episode_info.get("task", "tactile_episode")
    episodes = _generate_episodes_jsonl(
        data,
        episode_index=episode_index,
        task_name=task_name_final,
    )
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # 生成 meta/tasks.jsonl
    tasks = _generate_tasks_jsonl(
        task_name=task_name_final,
        task_description=task_description,
    )
    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    stats = {
        "output_path": str(output_path),
        "num_frames": data.num_frames,
        "num_episodes": 1,
        "duration_s": round(data.duration_s, 4),
        "tactile_dim": 14,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "parquet_file": str(parquet_file),
    }

    print(f"[OK] LeRobot dataset created at {output_path}")
    print(f"     Frames: {stats['num_frames']}")
    print(f"     Duration: {stats['duration_s']}s")
    print(f"     Tactile dim: {stats['tactile_dim']}")
    print(f"     Parquet: {parquet_file}")

    return stats


__all__ = [
    "create_lerobot_dataset",
    "tlabeldata_to_lerobot",
]
