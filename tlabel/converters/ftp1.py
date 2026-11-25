"""
TLabel → FTP-1/MTTS 格式转换器

将TLabel标注数据导出为FTP-1兼容的Zarr格式（MTTS — Morphology-Aware Tactile Token Space）。
导出后可直接用于FTP-1模型微调或推理。

MTTS Zarr格式（4个key per side per group）：
    <side>_tactile_data_<group>:   (T, N, *tac_shape)   # 触觉数据
    <side>_tactile_area_<group>:   (T, N)                # 功能区ID
    <side>_tactile_sensor_<group>: (T,)                   # 传感器名称
    <side>_tactile_type_<group>:   (T,)                   # 类型: image/matrix/binary

用法:
    from tlabel.converters.ftp1 import tlabel_to_ftp1

    # 单传感器导出
    tlabel_to_ftp1(data, "output.zarr",
                    sensor_name="GelSightMini",
                    functional_areas=[0, 1],   # 拇指尖+食指尖
                    side="right")

    # 多Episode批量导出
    from tlabel.converters.ftp1 import batch_to_ftp1
    batch_to_ftp1(["ep1.json", "ep2.json"], "dataset.zarr",
                   sensor_name="GelSightMini",
                   functional_areas=[0, 1])
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from tlabel.core.types import TLabelData


# ============================================================
# MTTS 功能区定义
# ============================================================

# 手部功能区槽位（0-14）
HAND_FUNCTIONAL_AREAS = {
    0: "thumb_tip",           # 拇指尖
    1: "index_fingertip",     # 食指尖
    2: "middle_fingertip",    # 中指尖
    3: "ring_fingertip",      # 无名指尖
    4: "pinky_fingertip",     # 小指尖
    5: "thumb_pad",           # 拇指指腹
    6: "index_pad",           # 食指指腹
    7: "middle_pad",          # 中指指腹
    8: "ring_pad",            # 无名指指腹
    9: "pinky_pad",           # 小指指腹
    10: "thenar",             # 鱼际（拇指根部掌面）
    11: "hypothenar",         # 小鱼际（小指侧掌面）
    12: "palm_center",        # 掌心
    13: "proximal_phalanx",   # 近节指骨
    14: "dorsum",             # 手背
}

# 力/力矩槽位（15-20）
TORQUE_AREAS = {
    15: "wrist_fx",           # 腕部力X
    16: "wrist_fy",           # 腕部力Y
    17: "wrist_fz",           # 腕部力Z
    18: "wrist_tx",           # 腕部力矩X
    19: "wrist_ty",           # 腕部力矩Y
    20: "wrist_tz",           # 腕部力矩Z
}

ALL_FUNCTIONAL_AREAS = {**HAND_FUNCTIONAL_AREAS, **TORQUE_AREAS}

# ============================================================
# FTP-1 已知传感器注册表
# ============================================================

FTP1_KNOWN_SENSORS = {
    # image 类传感器
    "GelSight":        {"type": "image", "modality": "image", "default_shape": (224, 224, 3)},
    "GelSightMini":    {"type": "image", "modality": "image", "default_shape": (224, 224, 3)},
    "FreeTacMan":      {"type": "image", "modality": "image", "default_shape": (224, 224, 3)},
    "ViTaMIn":         {"type": "image", "modality": "image", "default_shape": (224, 224, 3)},
    # matrix 类传感器
    "3DViTac":         {"type": "matrix", "modality": "matrix", "default_shape": (12, 32)},
    "Contactile":      {"type": "matrix", "modality": "matrix", "default_shape": (12, 32)},
    # binary 类传感器
    "BinaryContact":   {"type": "binary", "modality": "binary", "default_shape": (1,)},
}

# 默认功能区映射：常见传感器配置
DEFAULT_AREA_MAPPINGS = {
    "parallel_gripper": [0, 1],      # 夹爪 → 拇指尖+食指尖
    "three_finger":     [0, 1, 2],    # 三指 → 拇指+食指+中指
    "five_finger":      [0, 1, 2, 3, 4],  # 五指 → 全指尖
    "dexterous_hand":   list(range(15)),  # 灵巧手 → 全部手部区域
}


def _check_zarr():
    if not HAS_ZARR:
        raise ImportError(
            "FTP-1 export requires zarr. Install with: pip install zarr"
        )


def _zarr_create_dataset(group, key, data, chunks=None):
    """兼容 zarr v2 (create_dataset) 和 v3 (create_array) 的数据集创建。
    
    zarr v3 要求 chunks 维度数必须与 data 维度数一致，
    此函数会自动对齐维度，避免 chunk_grid/shape 维度不匹配错误。
    """
    data_shape = np.asarray(data).shape
    # 对齐 chunks 维度到 data 维度
    if chunks is not None:
        if len(chunks) != len(data_shape):
            if len(chunks) > len(data_shape):
                # chunks 维度过多，截断到与 data 相同
                chunks = chunks[:len(data_shape)]
            else:
                # chunks 维度过少，用完整维度填充
                chunks = tuple(chunks) + data_shape[len(chunks):]
        # 确保每个 chunk 维度不超过 data 对应维度
        chunks = tuple(min(c, s) for c, s in zip(chunks, data_shape))

    if hasattr(group, 'create_array'):
        # zarr v3: object dtype 不支持，需转为 numpy 原生 string dtype
        arr = np.asarray(data)
        if arr.dtype == object:
            arr = arr.astype(str)  # → <U{N} unicode string
        kwargs = dict(data=arr)
        if chunks is not None:
            kwargs['chunks'] = chunks
        return group.create_array(key, **kwargs)
    else:
        # zarr v2: also convert object dtype to avoid "missing object_codec" error
        arr = np.asarray(data)
        if arr.dtype == object:
            arr = arr.astype(str)
        kwargs = dict(data=arr)
        if chunks is not None:
            kwargs['chunks'] = chunks
        return group.create_dataset(key, **kwargs)


def _zarr_resize(array, new_shape):
    """兼容 zarr v2 (resize(*shape)) 和 v3 (resize(tuple))。"""
    if isinstance(new_shape, tuple):
        try:
            array.resize(new_shape)
        except TypeError:
            array.resize(*new_shape)
    else:
        array.resize(new_shape)


def _resize_image(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    将图像缩放到FTP-1要求的目标尺寸。

    Args:
        img: 输入图像 (H, W, C) uint8
        target_size: 目标 (H, W)

    Returns:
        缩放后的图像 (target_H, target_W, C) uint8
    """
    if not HAS_CV2:
        raise ImportError(
            "Image resizing requires opencv-python. Install with: pip install opencv-python"
        )
    h, w = target_size
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized


def _normalize_image_to_float(img: np.ndarray) -> np.ndarray:
    """
    将uint8 [0, 255] 图像归一化为 float32 [-1, 1]（FTP-1要求）。

    Args:
        img: uint8 图像

    Returns:
        float32 图像，值域 [-1, 1]
    """
    return img.astype(np.float32) / 127.5 - 1.0


def _denormalize_float_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    将float32 [-1, 1] 图像反归一化为 uint8 [0, 255]。

    Args:
        img: float32 图像，值域 [-1, 1]

    Returns:
        uint8 图像
    """
    return np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)


def _extract_raw_images(data: TLabelData) -> Optional[np.ndarray]:
    """
    从TLabelData中提取原始图像数据。

    搜索顺序：
    1. sensor_specific中的raw_images / images键
    2. sensor_specific中的frame级image数据
    3. 如果都没有，返回None

    Returns:
        (T, H, W, 3) uint8 数组，或 None
    """
    if not data.frames:
        return None

    # 尝试从sensor_specific提取
    images_list = []

    for frame in data.frames:
        ss = frame.sensor_specific
        if not ss:
            continue

        # 直接有raw image
        if "raw_image" in ss:
            img = ss["raw_image"]
            if isinstance(img, np.ndarray):
                images_list.append(img)
                continue

        # JPEG编码的图像
        if "raw_image_jpeg" in ss:
            raw_bytes = ss["raw_image_jpeg"]
            if HAS_CV2:
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    images_list.append(img)
                    continue

        # 尝试背景减除后的差分图像（降级方案）
        if "diff_image" in ss:
            diff = ss["diff_image"]
            if isinstance(diff, np.ndarray) and diff.ndim == 3:
                # 差分图像转为伪RGB
                gray = np.mean(np.abs(diff), axis=2, keepdims=True)
                pseudo_rgb = np.repeat(gray, 3, axis=2).astype(np.uint8)
                images_list.append(pseudo_rgb)
                continue

    if not images_list:
        return None

    return np.stack(images_list, axis=0)  # (T, H, W, 3)


def _extract_matrix_data(data: TLabelData) -> Optional[np.ndarray]:
    """
    从TLabelData中提取矩阵类传感器数据（如Contactile的rows×cols阵列）。

    Returns:
        (T, rows, cols) float32 数组，或 None
    """
    if not data.frames:
        return None

    matrices = []
    for frame in data.frames:
        ss = frame.sensor_specific
        if not ss:
            continue
        if "matrix_data" in ss:
            matrices.append(np.array(ss["matrix_data"], dtype=np.float32))
        elif "taxel_values" in ss:
            matrices.append(np.array(ss["taxel_values"], dtype=np.float32))

    if not matrices:
        return None

    return np.stack(matrices, axis=0)


def _extract_binary_data(data: TLabelData) -> Optional[np.ndarray]:
    """
    从TLabelData中提取二值传感器数据。

    Returns:
        (T, num_sensors) float32 数组，或 None
    """
    if not data.frames:
        return None

    binary_list = []
    for frame in data.frames:
        ss = frame.sensor_specific
        if not ss:
            continue
        if "binary_contact" in ss:
            binary_list.append(np.array(ss["binary_contact"], dtype=np.float32))
        elif "contact_binary" in ss:
            binary_list.append(np.array(ss["contact_binary"], dtype=np.float32))

    if not binary_list:
        return None

    return np.stack(binary_list, axis=0)


def tlabel_to_ftp1(
    data: TLabelData,
    output_path: Union[str, Path],
    sensor_name: str = "GelSightMini",
    functional_areas: Optional[List[int]] = None,
    side: str = "right",
    group: str = "gripper",
    target_image_size: Tuple[int, int] = (224, 224),
    store_raw_uint8: bool = True,
    append: bool = True,
) -> Dict:
    """
    将TLabelData导出为FTP-1 MTTS Zarr格式。

    Args:
        data: TLabelData实例
        output_path: 输出.zarr路径
        sensor_name: FTP-1注册的传感器名（如"GelSightMini"）
        functional_areas: 功能区ID列表（如[0, 1]表示拇指尖+食指尖）
        side: "left" 或 "right"
        group: 组名（如"gripper", "dexterous"）
        target_image_size: 图像目标尺寸 (H, W)
        store_raw_uint8: 是否存储uint8原始图像（True=FTP-1训练格式，False=存float32归一化后）
        append: 是否追加到已有Zarr（True=追加时间步，False=新建）

    Returns:
        导出统计信息dict
    """
    _check_zarr()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 确定传感器类型
    sensor_info = FTP1_KNOWN_SENSORS.get(sensor_name, {})
    tactile_type = sensor_info.get("type", "image")

    # 如果用户没指定功能区，用默认值
    if functional_areas is None:
        if side == "right" or side == "left":
            functional_areas = DEFAULT_AREA_MAPPINGS.get("parallel_gripper", [0, 1])
        else:
            functional_areas = [0]

    N = len(functional_areas)  # 功能区数量
    T = data.num_frames        # 时间步数

    # 提取数据
    tactile_data = None
    if tactile_type == "image":
        tactile_data = _extract_raw_images(data)
    elif tactile_type == "matrix":
        tactile_data = _extract_matrix_data(data)
    elif tactile_type == "binary":
        tactile_data = _extract_binary_data(data)

    # 如果没找到raw数据，尝试从 Schema V2 特征构造降级数据
    if tactile_data is None:
        # 降级方案：用 Schema V2 (14维) 特征作为matrix数据
        feature_matrix = []
        for frame in data.frames:
            sv2 = frame.schema_v2
            # v0.26（D8）：contact / slip_event 是三态。FTP1 的矩阵是纯浮点，
            # 装不下 null，因此未知用 NaN 表示 —— 下游能识别为缺失。
            # 填 0.0 等于把「不知道」导出成一次「确定没有接触」的观测。
            _nan = float("nan")
            row = [
                _nan if sv2.contact is None else (1.0 if sv2.contact else 0.0),
                sv2.contact_centroid[0] if sv2.contact_centroid else 0.0,
                sv2.contact_centroid[1] if sv2.contact_centroid else 0.0,
                sv2.force_magnitude if sv2.force_magnitude is not None else 0.0,
                _nan if sv2.slip_event is None else (1.0 if sv2.slip_event else 0.0),
                sv2.object_deformation if sv2.object_deformation is not None else 0.0,
                sv2.confidence,
            ]
            feature_matrix.append(row)
        tactile_data = np.array(feature_matrix, dtype=np.float32)  # (T, 14 or 22)
        tactile_type = "matrix"  # 降级为matrix类型

    # 处理图像数据
    if tactile_type == "image" and tactile_data is not None:
        # 调整尺寸
        processed_images = []
        for i in range(tactile_data.shape[0]):
            img = tactile_data[i]
            if img.shape[:2] != target_image_size:
                img = _resize_image(img, target_image_size)
            processed_images.append(img)
        tactile_data = np.stack(processed_images, axis=0)

        if store_raw_uint8:
            tactile_data = tactile_data.astype(np.uint8)
        else:
            tactile_data = _normalize_image_to_float(tactile_data)

    # 构建形状: (T, N, *tac_shape)
    tac_shape = tactile_data.shape[1:]  # 单帧单功能区的shape
    full_shape = (T, N) + tac_shape

    # 功能区数据: (T, N) — 每个时间步每个功能区一个ID
    area_data = np.zeros((T, N), dtype=np.int32)
    for t in range(T):
        for n in range(N):
            area_data[t, n] = functional_areas[n]

    # 传感器名: (T,) 字符串
    sensor_data = np.array([sensor_name] * T, dtype=object)

    # 类型: (T,) 字符串
    type_data = np.array([tactile_type] * T, dtype=object)

    # 打开/创建Zarr
    if append and output_path.exists():
        root = zarr.open(str(output_path), mode='a')
    else:
        root = zarr.open(str(output_path), mode='w')

    # 构建key名称
    prefix = f"{side}_tactile"

    # 写入数据
    data_key = f"{prefix}_data_{group}"
    area_key = f"{prefix}_area_{group}"
    sensor_key = f"{prefix}_sensor_{group}"
    type_key = f"{prefix}_type_{group}"

    # 触觉数据
    if data_key in root:
        # 追加模式：沿时间轴扩展
        existing = root[data_key]
        old_T = existing.shape[0]
        _zarr_resize(existing, (old_T + T, *existing.shape[1:]))
        if tactile_type == "image" and store_raw_uint8:
            existing[old_T:old_T + T] = tactile_data.astype(np.uint8)
        else:
            existing[old_T:old_T + T] = tactile_data
    else:
        if tactile_type == "image" and store_raw_uint8:
            _zarr_create_dataset(root, data_key, data=tactile_data.astype(np.uint8),
                              chunks=(10, N, *tac_shape))
        else:
            _zarr_create_dataset(root, data_key, data=tactile_data,
                              chunks=(10, N, *tac_shape))

    # 功能区
    if area_key in root:
        existing = root[area_key]
        old_T = existing.shape[0]
        _zarr_resize(existing, old_T + T)
        existing[old_T:old_T + T] = area_data
    else:
        _zarr_create_dataset(root, area_key, data=area_data)

    # 传感器名
    if sensor_key in root:
        existing = root[sensor_key]
        old_T = existing.shape[0]
        _zarr_resize(existing, old_T + T)
        for i in range(T):
            existing[old_T + i] = sensor_name
    else:
        _zarr_create_dataset(root, sensor_key, data=sensor_data)

    # 类型
    if type_key in root:
        existing = root[type_key]
        old_T = existing.shape[0]
        _zarr_resize(existing, old_T + T)
        for i in range(T):
            existing[old_T + i] = tactile_type
    else:
        _zarr_create_dataset(root, type_key, data=type_data)

    # 写入元数据
    root.attrs.setdefault("ftp1_version", "1.0")
    root.attrs.setdefault("tlabel_version", data.schema_version)
    root.attrs["last_export_time"] = str(np.datetime64('now'))

    # 导出统计
    stats = {
        "output_path": str(output_path),
        "sensor_name": sensor_name,
        "tactile_type": tactile_type,
        "side": side,
        "group": group,
        "functional_areas": functional_areas,
        "functional_area_names": [ALL_FUNCTIONAL_AREAS.get(a, f"unknown_{a}") for a in functional_areas],
        "time_steps": T,
        "num_slots": N,
        "data_shape": full_shape,
        "data_dtype": str(tactile_data.dtype),
        "zarr_keys": [data_key, area_key, sensor_key, type_key],
        "append_mode": append,
    }

    return stats


def batch_to_ftp1(
    data_list: List[Union[TLabelData, str, Path]],
    output_path: Union[str, Path],
    sensor_name: str = "GelSightMini",
    functional_areas: Optional[List[int]] = None,
    side: str = "right",
    group: str = "gripper",
    target_image_size: Tuple[int, int] = (224, 224),
) -> Dict:
    """
    批量导出多个Episode到同一个Zarr文件。

    Args:
        data_list: TLabelData实例列表，或TLabel JSON文件路径列表
        output_path: 输出.zarr路径
        sensor_name: 传感器名
        functional_areas: 功能区ID列表
        side: "left" 或 "right"
        group: 组名
        target_image_size: 图像目标尺寸

    Returns:
        总导出统计
    """
    _check_zarr()

    total_stats = {
        "output_path": str(output_path),
        "episodes": 0,
        "total_time_steps": 0,
        "sensor_name": sensor_name,
        "functional_areas": functional_areas or DEFAULT_AREA_MAPPINGS.get("parallel_gripper", [0, 1]),
    }

    for i, item in enumerate(data_list):
        # 加载数据
        if isinstance(item, (str, Path)):
            from tlabel.core.loader import load
            data = load(str(item))
        else:
            data = item

        # 第一个episode创建，后续追加
        append = (i > 0)
        stats = tlabel_to_ftp1(
            data, output_path,
            sensor_name=sensor_name,
            functional_areas=functional_areas,
            side=side,
            group=group,
            target_image_size=target_image_size,
            append=append,
        )

        total_stats["episodes"] += 1
        total_stats["total_time_steps"] += stats["time_steps"]

    total_stats["data_shape"] = stats["data_shape"]  # 最后一次export的shape信息

    return total_stats


def list_functional_areas() -> Dict[int, str]:
    """列出所有MTTS功能区定义"""
    return ALL_FUNCTIONAL_AREAS.copy()


def list_known_sensors() -> Dict[str, Dict]:
    """列出FTP-1已知的所有传感器"""
    return FTP1_KNOWN_SENSORS.copy()


# 公共API
__all__ = [
    "tlabel_to_ftp1",
    "batch_to_ftp1",
    "list_functional_areas",
    "list_known_sensors",
    "HAND_FUNCTIONAL_AREAS",
    "TORQUE_AREAS",
    "ALL_FUNCTIONAL_AREAS",
    "FTP1_KNOWN_SENSORS",
    "DEFAULT_AREA_MAPPINGS",
]
