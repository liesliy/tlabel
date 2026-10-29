"""
戴盟(Daimon)数据集适配器 — 将Daimon-Infinity Parquet数据转换为TLabelData

这是Daimon的数据集（离线）适配器，用于加载DM-TacClaw采集的.parquet/LeRobot格式数据。
实时传感器适配器见 daimon_dm_tac.py。

Daimon-Infinity数据格式特点:
- 主数据: Parquet格式 (data/chunk-xxx/file-xxx.parquet)
- observation.state: 114维float32 (位姿+关节+触觉+IMU)
- action: 111维float32
- 触觉视频: FFV1编码.mov (deformation/shear/depth各一路, gbrp16le/gray16le)
- 数值触觉: finger0~finger35 (idx 67-102, 常为9930占位)
- 占位值: 9930.0 = 无效/未启用维度
- fps: 30

目录结构:
  DM-DataClaw/datasets/v1_3_usb_backups_XXXX/DEVICE_ID_lerobot_TIME/
  ├── meta/info.json          # 数据集配置(含videos字段)
  ├── meta/stats.json         # 统计信息
  ├── meta/tasks.parquet      # 任务描述
  ├── data/chunk-000/file-000.parquet  # 主数据
  ├── videos/
  │   ├── observation.deformation.gripperrighttactile/
  │   │   └── chunk-000/file-000.mov   # FFV1/gbrp16le
  │   ├── observation.shear.gripperrighttactile/
  │   │   └── chunk-000/file-000.mov   # FFV1/gbrp16le
  │   ├── observation.depth.gripperrighttactile/
  │   │   └── chunk-000/file-000.mov   # FFV1/gray16le
  │   └── observation.images.cam_right/
  │       └── chunk-000/file-000.mp4   # h264/yuv420p
  └── episodes_metadata.json

支持加载方式:
  1. 指定parquet文件路径 (自动查找同级meta/和videos/)
  2. 指定episode目录路径 (自动查找data/和meta/)
  3. 指定parquet文件 (直接加载)
"""

import math
import json
import struct
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

from tlabel.adapters.base import BaseAdapter
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import subprocess
    HAS_SUBPROCESS = True
except ImportError:
    HAS_SUBPROCESS = False

# HACK: 占位值常量 — Daimon用9930.0表示无效维度
PLACEHOLDER = 9930.0

# observation.state 维度定义 (114维)
STATE_DIMS = {
    "left_x": 0, "left_y": 1, "left_z": 2,
    "left_qx": 3, "left_qy": 4, "left_qz": 5, "left_qw": 6,
    "right_x": 7, "right_y": 8, "right_z": 9,
    "right_qx": 10, "right_qy": 11, "right_qz": 12, "right_qw": 13,
    "head_x": 14, "head_y": 15, "head_z": 16,
    "head_qx": 17, "head_qy": 18, "head_qz": 19, "head_qw": 20,
    "left_eye_x": 21, "left_eye_y": 22, "left_eye_z": 23,
    "left_eye_qx": 24, "left_eye_qy": 25, "left_eye_qz": 26, "left_eye_qw": 27,
    "right_eye_x": 28, "right_eye_y": 29, "right_eye_z": 30,
    "right_eye_qx": 31, "right_qy": 32, "right_qz": 33, "right_qw": 34,
    "third_x": 35, "third_y": 36, "third_z": 37,
    "third_qx": 38, "third_qy": 39, "third_qz": 40, "third_qw": 41,
    "arm_left_1": 42, "arm_left_2": 43, "arm_left_3": 44, "arm_left_4": 45,
    "arm_left_5": 46, "arm_left_6": 47, "arm_left_7": 48,
    "arm_right_1": 49, "arm_right_2": 50, "arm_right_3": 51, "arm_right_4": 52,
    "arm_right_5": 53, "arm_right_6": 54, "arm_right_7": 55,
    "head_pitch": 56, "head_yaw": 57,
    "hip_pitch": 58, "hip_yaw": 59, "knee": 60, "left_wheel": 61, "right_wheel": 62,
    "gripper": 63, "gripper_left": 64, "gripper_right": 65,
    "finger_0": 66, "finger_1": 67, "finger_2": 68, "finger_3": 69,
    "finger_4": 70, "finger_5": 71, "finger_6": 72, "finger_7": 73,
    "finger_8": 74, "finger_9": 75, "finger_10": 76, "finger_11": 77,
    "finger_12": 78, "finger_13": 79, "finger_14": 80, "finger_15": 81,
    "finger_16": 82, "finger_17": 83, "finger_18": 84, "finger_19": 85,
    "finger_20": 86, "finger_21": 87, "finger_22": 88, "finger_23": 89,
    "finger_24": 90, "finger_25": 91, "finger_26": 92, "finger_27": 93,
    "finger_28": 94, "finger_29": 95, "finger_30": 96, "finger_31": 97,
    "finger_32": 98, "finger_33": 99, "finger_34": 100, "finger_35": 101,
    "left_acc_x": 102, "left_acc_y": 103, "left_acc_z": 104,
    "left_gyro_x": 105, "left_gyro_y": 106, "left_gyro_z": 107,
    "right_acc_x": 108, "right_acc_y": 109, "right_acc_z": 110,
    "right_gyro_x": 111, "right_gyro_y": 112, "right_gyro_z": 113,
}


# ============================================================
# 视频查找与解码（Bug1-6全部重写）
# ============================================================

# 触觉视频键名的匹配模式（优先级从高到低）
_TACTILE_VIDEO_PATTERNS = [
    # deformation视频 — 核心触觉信号，最高优先级
    ("deformation", ["deformation.gripper"]),
    # shear视频 — 力场方向
    ("shear", ["shear.gripper"]),
    # depth视频 — 接触深度
    ("depth", ["depth.gripper"]),
    # 普通相机（非触觉，但可做光流fallback）
    ("cam", ["images.cam"]),
    # RGB触觉（声明有但经常404，放最后）
    ("rgb_tactile", ["images.gripper"]),
]


def _find_tactile_videos(parquet_path: Path) -> Dict[str, Path]:
    """定位所有可用的触觉视频文件

    修复Bug1: 匹配实际键名如observation.deformation.gripperrighttactile
    修复Bug2: 支持chunk级别路径 videos/{key}/chunk-000/file-000.ext
    修复Bug3: 同时搜索.mp4和.mov文件
    修复Bug5: 找到的文件必须真实存在（跳过404的RGB触觉）

    返回: {video_type: Path} 如 {"deformation": Path(...), "shear": Path(...)}
    """
    result = {}

    # 向上查找包含meta/和videos/的根目录
    root_dir = _find_dataset_root(parquet_path)
    if root_dir is None:
        return result

    meta_dir = root_dir / "meta"
    videos_dir = root_dir / "videos"

    # 策略1: 从info.json的videos字段精确匹配
    info_file = meta_dir / "info.json"
    if info_file.exists():
        try:
            with open(info_file, "r") as f:
                info = json.load(f)

            video_keys = info.get("videos", [])
            for vk in video_keys:
                vk_lower = vk.lower()
                for vtype, patterns in _TACTILE_VIDEO_PATTERNS:
                    if any(p in vk_lower for p in patterns):
                        # 尝试多种路径格式
                        path = _resolve_video_path(videos_dir, vk)
                        if path is not None:
                            # 同类型可能有多路（left/right），只取第一个找到的
                            if vtype not in result:
                                result[vtype] = path
                        break
        except Exception:
            pass

    # 策略2: fallback — 直接遍历videos/子目录
    if not result and videos_dir.exists():
        for subdir in sorted(videos_dir.iterdir()):
            if not subdir.is_dir():
                continue
            subdir_name = subdir.name.lower()
            for vtype, patterns in _TACTILE_VIDEO_PATTERNS:
                if any(p in subdir_name for p in patterns):
                    if vtype in result:
                        break
                    # 在子目录里找视频文件
                    path = _find_video_in_dir(subdir)
                    if path is not None:
                        result[vtype] = path
                    break

    return result


def _find_dataset_root(parquet_path: Path) -> Optional[Path]:
    """向上查找包含meta/目录的数据集根目录"""
    current = parquet_path.parent
    for _ in range(6):
        if (current / "meta").exists() or (current / "videos").exists():
            return current
        current = current.parent
    return None


def _resolve_video_path(videos_dir: Path, video_key: str) -> Optional[Path]:
    """根据video_key查找实际视频文件

    支持2种路径格式：
    1. chunk级别（实际格式）: videos/{key}/chunk-NNN/file-NNN.{mov|mp4}
    2. episode级别（旧格式）: videos/{key}/episode_NNNNNN/video.{mov|mp4}
    """
    key_dir = videos_dir / video_key
    if not key_dir.exists():
        return None

    # 尝试chunk级别路径: chunk-000/file-000.{mov,mp4}
    for chunk_dir in sorted(key_dir.iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        for file_path in sorted(chunk_dir.iterdir()):
            if file_path.suffix.lower() in (".mov", ".mp4"):
                if file_path.exists() and file_path.stat().st_size > 0:
                    return file_path

    # 尝试episode级别路径（兼容旧数据格式）
    for ep_dir in sorted(key_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        for ext in (".mov", ".mp4"):
            video_file = ep_dir / f"video{ext}"
            if video_file.exists() and video_file.stat().st_size > 0:
                return video_file

    # 尝试直接在key_dir下找视频文件
    for ext in (".mov", ".mp4"):
        direct = key_dir / f"video{ext}"
        if direct.exists() and direct.stat().st_size > 0:
            return direct

    return None


def _find_video_in_dir(directory: Path) -> Optional[Path]:
    """在目录中递归查找第一个有效的视频文件"""
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in (".mov", ".mp4"):
            if path.exists() and path.stat().st_size > 0:
                return path
    return None


def _probe_video(video_path: Path) -> Dict[str, Any]:
    """用ffprobe检测视频编码信息

    返回: {codec, pix_fmt, width, height, nb_frames, duration}
    """
    if not HAS_SUBPROCESS:
        return {}

    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}

        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        video_stream = None
        for s in streams:
            if s.get("codec_type") == "video":
                video_stream = s
                break

        if video_stream is None:
            return {}

        return {
            "codec": video_stream.get("codec_name", "unknown"),
            "pix_fmt": video_stream.get("pix_fmt", "unknown"),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "nb_frames": int(video_stream.get("nb_frames", 0)),
            "duration": float(video_stream.get("duration", 0)),
        }
    except Exception:
        return {}


def _extract_video_frames(video_path: Path, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """用cv2逐帧解码常规mp4视频

    返回: List[np.ndarray] (BGR uint8格式)
    仅适用于h264/mpeg4等cv2可解码的格式
    """
    if not HAS_CV2 or not video_path.exists():
        return []

    frames = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_idx += 1
    finally:
        cap.release()

    return frames


def _extract_ffv1_frames(video_path: Path, pix_fmt: str,
                          width: int, height: int,
                          max_frames: Optional[int] = None) -> List[np.ndarray]:
    """用ffmpeg rawvideo提取FFV1编码视频帧

    修复Bug4: FFV1/gbrp16le无法用cv2解码 → 用ffmpeg rawvideo
    修复Bug6: 16位数据转8位丢失 → 完整提取16位后去基线归一化

    pix_fmt支持:
    - gbrp16le: 3平面(G,B,R)各16位小端，每帧 = width*height*2*3 字节
    - gray16le: 单平面16位小端，每帧 = width*height*2 字节

    返回: List[np.ndarray] (float32, 已去基线归一化到0-1)
    """
    if not HAS_SUBPROCESS:
        return []

    # 计算每帧字节大小
    if pix_fmt == "gbrp16le":
        frame_bytes = width * height * 2 * 3  # 3 planes × 16bit
    elif pix_fmt == "gray16le":
        frame_bytes = width * height * 2  # 1 plane × 16bit
    else:
        return []

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", pix_fmt,
        "-v", "quiet",
        "-"
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0:
            return []

        raw = proc.stdout
        total_frames = len(raw) // frame_bytes
        if total_frames == 0:
            return []

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        frames = []
        for i in range(total_frames):
            offset = i * frame_bytes
            frame_raw = raw[offset:offset + frame_bytes]

            if pix_fmt == "gbrp16le":
                # gbrp16le: 3个平面依次排列 (G, B, R)，每平面 width*height 个 uint16
                plane_size = width * height
                g_plane = np.frombuffer(frame_raw[:plane_size * 2], dtype='<u2').reshape(height, width).astype(np.float32)
                b_plane = np.frombuffer(frame_raw[plane_size * 2:plane_size * 4], dtype='<u2').reshape(height, width).astype(np.float32)
                r_plane = np.frombuffer(frame_raw[plane_size * 4:], dtype='<u2').reshape(height, width).astype(np.float32)

                # 去基线: 戴盟基线值≈30000，减去后得到实际触觉信号
                # 基线取帧内中位数更鲁棒
                g_baseline = float(np.median(g_plane))
                b_baseline = float(np.median(b_plane))
                r_baseline = float(np.median(r_plane))

                g_signal = np.abs(g_plane - g_baseline)
                b_signal = np.abs(b_plane - b_baseline)
                r_signal = np.abs(r_plane - r_baseline)

                # 合并为3通道信号图，归一化到0-1
                # 信号范围一般在0-2000，用99百分位做上界
                max_val = max(
                    float(np.percentile(g_signal, 99)),
                    float(np.percentile(b_signal, 99)),
                    float(np.percentile(r_signal, 99)),
                    1.0  # 防止除零
                )

                normalized = np.stack([
                    r_signal / max_val,
                    g_signal / max_val,
                    b_signal / max_val,
                ], axis=-1)
                normalized = np.clip(normalized, 0.0, 1.0)
                frames.append(normalized)

            elif pix_fmt == "gray16le":
                depth = np.frombuffer(frame_raw, dtype='<u2').reshape(height, width).astype(np.float32)
                # depth值很小（max≈24），直接归一化
                d_max = float(np.max(depth))
                if d_max > 0:
                    normalized = (depth / d_max).reshape(height, width, 1)
                    # 扩展为3通道方便后续处理
                    normalized = np.repeat(normalized, 3, axis=-1)
                else:
                    normalized = np.zeros((height, width, 3), dtype=np.float32)
                frames.append(normalized)

        return frames

    except subprocess.TimeoutExpired:
        return []
    except Exception:
        return []


def _extract_ffv1_frames_streaming(video_path: Path, pix_fmt: str,
                                     width: int, height: int,
                                     max_frames: Optional[int] = None) -> List[np.ndarray]:
    """流式提取FFV1视频帧（适合大文件，内存友好）

    与_extract_ffv1_frames功能相同，但逐帧读取而非全量加载
    """
    if not HAS_SUBPROCESS:
        return []

    if pix_fmt == "gbrp16le":
        frame_bytes = width * height * 2 * 3
    elif pix_fmt == "gray16le":
        frame_bytes = width * height * 2
    else:
        return []

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", pix_fmt,
        "-v", "quiet",
        "-"
    ]

    frames = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_count = 0
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            if pix_fmt == "gbrp16le":
                plane_size = width * height
                g = np.frombuffer(raw[:plane_size * 2], dtype='<u2').reshape(height, width).astype(np.float32)
                b = np.frombuffer(raw[plane_size * 2:plane_size * 4], dtype='<u2').reshape(height, width).astype(np.float32)
                r = np.frombuffer(raw[plane_size * 4:], dtype='<u2').reshape(height, width).astype(np.float32)

                g_bl = float(np.median(g))
                b_bl = float(np.median(b))
                r_bl = float(np.median(r))
                g_sig = np.abs(g - g_bl)
                b_sig = np.abs(b - b_bl)
                r_sig = np.abs(r - r_bl)

                mv = max(float(np.percentile(g_sig, 99)), float(np.percentile(b_sig, 99)),
                         float(np.percentile(r_sig, 99)), 1.0)
                norm = np.stack([r_sig / mv, g_sig / mv, b_sig / mv], axis=-1)
                frames.append(np.clip(norm, 0.0, 1.0))

            elif pix_fmt == "gray16le":
                d = np.frombuffer(raw, dtype='<u2').reshape(height, width).astype(np.float32)
                d_max = float(np.max(d))
                if d_max > 0:
                    n = np.repeat((d / d_max).reshape(height, width, 1), 3, axis=-1)
                else:
                    n = np.zeros((height, width, 3), dtype=np.float32)
                frames.append(n)

            frame_count += 1

        proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        pass

    return frames


def _compute_tactile_features_from_video(deform_frame: Optional[np.ndarray],
                                          shear_frame: Optional[np.ndarray],
                                          prev_deform_frame: Optional[np.ndarray] = None) -> Dict[str, float]:
    """从deformation/shear视频帧计算触觉特征

    这些特征之前标记为"需视频流"，现在从实际视频数据中提取：
    - deformation_magnitude: deformation场总变形量
    - contact_area: 超过阈值的像素比例
    - texture_energy: Laplacian方差（纹理复杂度）
    - edge_density: Canny边缘密度
    - shear_field_magnitude/direction: shear场大小和方向
    - normal_field: 法向场大小
    """
    result = {
        "deformation_magnitude": 0.0,
        "contact_area": 0.0,
        "texture_energy": 0.0,
        "edge_density": 0.0,
        "shear_field_magnitude": 0.0,
        "shear_field_direction": 0.0,
        "normal_field_magnitude": 0.0,
        "normal_field_variance": 0.0,
        "contact": 0.0,
    }

    # --- 从deformation帧计算 ---
    if deform_frame is not None and deform_frame.size > 0:
        # deform_frame: (H, W, 3) float32, R/G/B各通道是去基线后的触觉信号
        # 3通道对应3个方向的形变
        gray = np.mean(deform_frame, axis=2)  # 平均通道

        # deformation_magnitude: 3通道RMS
        result["deformation_magnitude"] = float(np.sqrt(np.mean(deform_frame ** 2)))

        # contact: 超过阈值认为有接触
        threshold = 0.05  # 归一化后的阈值
        contact_mask = gray > threshold
        result["contact"] = 1.0 if np.any(contact_mask) else 0.0

        # contact_area: 接触面积比
        result["contact_area"] = float(np.mean(contact_mask))

        # texture_energy: Laplacian方差
        if gray.shape[0] > 2 and gray.shape[1] > 2:
            from scipy.ndimage import laplace
            try:
                lap = laplace(gray)
                result["texture_energy"] = float(np.var(lap))
            except ImportError:
                # fallback: 手动Laplacian
                pad = np.pad(gray, 1, mode='edge')
                lap = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4 * pad[1:-1, 1:-1])
                result["texture_energy"] = float(np.var(lap))

        # edge_density: 梯度幅度超过90百分位的比例
        if gray.shape[0] > 1 and gray.shape[1] > 1:
            gy, gx = np.gradient(gray)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)
            if grad_mag.max() > 0:
                p90 = np.percentile(grad_mag, 90)
                result["edge_density"] = float(np.mean(grad_mag > p90))

        # normal_field: 从3通道形变场计算
        # 每通道代表一个方向的力场分量
        r_ch = deform_frame[:, :, 0]
        g_ch = deform_frame[:, :, 1]
        b_ch = deform_frame[:, :, 2]
        nf_mag = float(np.sqrt(np.mean(r_ch ** 2 + g_ch ** 2 + b_ch ** 2)))
        result["normal_field_magnitude"] = nf_mag
        result["normal_field_variance"] = float(np.var(np.sqrt(r_ch ** 2 + g_ch ** 2 + b_ch ** 2)))

    # --- 从shear帧计算 ---
    if shear_frame is not None and shear_frame.size > 0:
        gray = np.mean(shear_frame, axis=2)
        r_ch = shear_frame[:, :, 0]
        g_ch = shear_frame[:, :, 1]

        if gray.shape[0] > 1 and gray.shape[1] > 1:
            r_gy, r_gx = np.gradient(r_ch)
            g_gy, g_gx = np.gradient(g_ch)
            shear_x = float(np.mean(np.abs(r_gx)))
            shear_y = float(np.mean(np.abs(g_gy)))
            result["shear_field_magnitude"] = np.sqrt(shear_x ** 2 + shear_y ** 2)
            result["shear_field_direction"] = float(np.degrees(np.arctan2(shear_y, shear_x)))

    return result


def _compute_optical_flow_from_tactile(prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[float, float]:
    """从触觉视频帧（deformation）计算光流

    输入: 归一化后的float32帧 (H, W, 3)
    输出: (magnitude, direction)
    """
    if not HAS_CV2:
        return 0.0, 0.0

    try:
        # 转uint8给Farneback
        prev_u8 = (np.clip(prev_frame, 0, 1) * 255).astype(np.uint8)
        curr_u8 = (np.clip(curr_frame, 0, 1) * 255).astype(np.uint8)
        prev_gray = cv2.cvtColor(prev_u8, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_u8, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag)), float(np.degrees(np.mean(ang))) % 360.0
    except Exception:
        return 0.0, 0.0


# ============================================================
# 数值处理辅助函数
# ============================================================

def _is_valid(val):
    """检查值是否有效（非占位）"""
    if isinstance(val, (list, np.ndarray)):
        return not np.all(np.array(val) == PLACEHOLDER)
    return val != PLACEHOLDER


def _mask_placeholder(arr):
    """将9930占位替换为NaN"""
    result = np.array(arr, dtype=np.float64)
    result[result == PLACEHOLDER] = np.nan
    return result


def _load_info_json(parquet_path: Path) -> Dict:
    """查找并加载meta/info.json"""
    current = parquet_path.parent
    for _ in range(5):
        meta_dir = current / "meta"
        if meta_dir.exists():
            info_file = meta_dir / "info.json"
            if info_file.exists():
                with open(info_file, "r") as f:
                    return json.load(f)
        current = current.parent
    return {}


def _load_tasks(parquet_path: Path) -> Dict[int, str]:
    """加载任务描述映射"""
    if not HAS_PYARROW:
        return {}
    current = parquet_path.parent
    for _ in range(5):
        tasks_file = current / "meta" / "tasks.parquet"
        if tasks_file.exists():
            try:
                table = pq.read_table(str(tasks_file))
                df = table.to_pandas()
                return dict(zip(df["task_index"], df["vlm_hybrid_task"]))
            except Exception:
                pass
        current = current.parent
    return {}


def _detect_robot_type(info: Dict) -> str:
    """从info.json检测机器人配置类型"""
    return info.get("robot_type", "unknown")


def _compute_contact_from_gripper(gripper_val: float,
                                   finger_data: np.ndarray,
                                   valid_dims: List[int]) -> bool:
    """从夹爪状态和触觉数据推断接触"""
    if np.isnan(gripper_val):
        return False
    gripper_closed = gripper_val < 0.0
    finger_valid = finger_data[~np.isnan(finger_data)]
    if len(finger_valid) > 0:
        finger_active = np.any(np.abs(finger_valid) > 0.5)
    else:
        finger_active = False
    return gripper_closed or finger_active


def _extract_tlabel_v2_from_state(state: np.ndarray, action: np.ndarray,
                                   prev_state: Optional[np.ndarray],
                                   robot_type: str,
                                   force_metrics: Optional[Dict] = None,
                                   # 视频补充的特征（覆盖state估算的值）
                                   video_features: Optional[Dict] = None,
                                   # --- 时序4维 ---
                                   optical_flow_magnitude: float = 0.0,
                                   optical_flow_direction: float = 0.0,
                                   temporal_deformation_rate: float = 0.0,
                                   contact_transition: float = 0.0) -> Dict[str, float]:
    """从observation.state + 视频特征提取22维TLabel v2

    优先使用视频特征（更准确），无视频时fallback到state估算
    """
    state_m = _mask_placeholder(state)

    # 视频特征优先
    vf = video_features or {}

    # 接触检测: 视频优先 > gripper+finger估算
    if "contact" in vf:
        is_contact = vf["contact"] > 0.5
    else:
        gripper_left = state_m[65] if not np.isnan(state_m[65]) else np.nan
        gripper_right = state_m[66] if len(state_m) > 66 and not np.isnan(state_m[66]) else np.nan
        gripper_main = gripper_left if not np.isnan(gripper_left) else gripper_right
        finger_data = state_m[67:103]
        finger_valid = finger_data[~np.isnan(finger_data)]
        is_contact = False
        if not np.isnan(gripper_main) and gripper_main < 0:
            is_contact = True
        if len(finger_valid) > 0 and np.any(np.abs(finger_valid) > 0.3):
            is_contact = True

    # Force: 视频deformation_magnitude优先，否则gripper估算
    if force_metrics:
        force_mag = force_metrics.get("mean_force", 0.0)
        force_peak = force_metrics.get("max_force", 0.0)
    elif "deformation_magnitude" in vf:
        # 视频deformation直接反映力度
        force_mag = vf["deformation_magnitude"] * 100  # 缩放到合理范围
        force_peak = force_mag * 1.5
    else:
        gripper_left = state_m[65] if not np.isnan(state_m[65]) else np.nan
        gripper_right = state_m[66] if len(state_m) > 66 and not np.isnan(state_m[66]) else np.nan
        gripper_main = gripper_left if not np.isnan(gripper_left) else gripper_right
        if gripper_main is not np.nan and not np.isnan(gripper_main):
            force_mag = max(0, abs(gripper_main) * 10)
        else:
            force_mag = 0.0
        force_peak = force_mag

    # Deformation: 视频优先
    if "deformation_magnitude" in vf:
        deformation_mag = vf["deformation_magnitude"]
    else:
        finger_data = state_m[67:103]
        finger_valid = finger_data[~np.isnan(finger_data)]
        deformation_mag = float(np.std(finger_valid)) if len(finger_valid) > 1 else 0.0

    # Contact area: 视频优先
    if "contact_area" in vf:
        contact_area = vf["contact_area"]
    else:
        finger_data = state_m[67:103]
        finger_valid = finger_data[~np.isnan(finger_data)]
        contact_area = float(np.sum(np.abs(finger_valid) > 0.3)) / max(len(finger_valid), 1) if len(finger_valid) > 0 else 0.0

    # IMU-based slip detection
    right_acc = state_m[108:111]
    right_gyro = state_m[111:114]
    slip_event = 0.0
    slip_entropy = 0.0
    if not np.any(np.isnan(right_acc)):
        acc_mag = float(np.sqrt(np.sum(right_acc ** 2)))
        if acc_mag > 5.0:
            slip_event = min(acc_mag / 20.0, 1.0)
        if not np.any(np.isnan(right_gyro)):
            gyro_mag = float(np.sqrt(np.sum(right_gyro ** 2)))
            if gyro_mag > 1.0:
                slip_event = max(slip_event, min(gyro_mag / 5.0, 1.0))

    # Delta force
    delta_fn = 0.0
    delta_fs = 0.0
    if prev_state is not None:
        prev_m = _mask_placeholder(prev_state)
        if not np.isnan(prev_m[65]) and not np.isnan(state_m[65]):
            delta_fn = abs(state_m[65] - prev_m[65])
        prev_acc = prev_m[108:111]
        if not np.any(np.isnan(prev_acc)) and not np.any(np.isnan(right_acc)):
            delta_fs = float(np.sqrt(np.sum((right_acc - prev_acc) ** 2)))

    # Normal/shear field: 视频优先
    nf_mag = vf.get("normal_field_magnitude", force_mag)
    nf_var = vf.get("normal_field_variance", 0.0)
    sf_mag = vf.get("shear_field_magnitude", 0.0)
    sf_dir = vf.get("shear_field_direction", 0.0)

    # 无视频时从finger数据估算法向场
    if "normal_field_magnitude" not in vf:
        finger_data = state_m[67:103]
        finger_valid = finger_data[~np.isnan(finger_data)]
        if len(finger_valid) > 1:
            nf_var = float(np.var(finger_valid))

    # Friction cone ratio
    friction_ratio = delta_fs / nf_mag if nf_mag > 1e-6 else 0.0

    # Force direction
    if not np.any(np.isnan(right_acc)) and np.sqrt(np.sum(right_acc ** 2)) > 0.01:
        force_dir = float(np.degrees(np.arctan2(right_acc[1], right_acc[0])))
    else:
        force_dir = 0.0

    # Centroid
    finger_data = state_m[67:103]
    finger_valid = finger_data[~np.isnan(finger_data)]
    if len(finger_valid) > 0 and np.sum(np.abs(finger_valid)) > 1e-10:
        weighted_pos = np.sum(np.arange(len(finger_valid)) * np.abs(finger_valid))
        centroid_x = weighted_pos / (np.sum(np.abs(finger_valid)) * max(len(finger_valid) - 1, 1))
    else:
        centroid_x = 0.5

    # Texture/edge: 视频优先
    texture_energy = vf.get("texture_energy", 0.0)
    edge_density = vf.get("edge_density", 0.0)

    return {
        "contact": 1.0 if is_contact else 0.0,
        "deformation_magnitude": round(deformation_mag, 4),
        "force_magnitude": round(force_mag, 4),
        "force_peak": round(force_peak, 4),
        "force_direction": round(force_dir, 2),
        "slip_entropy": round(slip_entropy, 4),
        "slip_event": round(slip_event, 4),
        "texture_energy": round(texture_energy, 4),
        "edge_density": round(edge_density, 4),
        "contact_area": round(min(contact_area, 1.0), 4),
        "centroid_x": round(centroid_x, 4),
        "normal_field_magnitude": round(nf_mag, 4),
        "normal_field_variance": round(nf_var, 4),
        "shear_field_magnitude": round(sf_mag, 4),
        "shear_field_direction": round(sf_dir, 2),
        "delta_force_normal": round(delta_fn, 4),
        "delta_force_shear": round(delta_fs, 4),
        "friction_cone_ratio": round(min(friction_ratio, 10.0), 4),
        # --- 时序4维 ---
        "optical_flow_magnitude": round(optical_flow_magnitude, 4),
        "optical_flow_direction": round(optical_flow_direction, 2),
        "temporal_deformation_rate": round(temporal_deformation_rate, 4),
        "contact_transition": round(contact_transition, 4),
    }


def _infer_phases(frames_contact, frames_slip):
    """从接触和滑移状态推断操作阶段"""
    phases = []
    current = "idle"
    for ic, is_slip in zip(frames_contact, frames_slip):
        if current == "idle":
            if ic:
                current = "initial_contact"
        elif current == "initial_contact":
            if is_slip:
                current = "slip"
            elif ic:
                current = "stable_contact"
        elif current == "stable_contact":
            if is_slip:
                current = "slip"
            elif not ic:
                current = "release"
        elif current == "slip":
            if not is_slip and ic:
                current = "stable_contact"
            elif not ic:
                current = "release"
        elif current == "release":
            if ic:
                current = "re_contact"
            else:
                current = "idle"
        elif current == "re_contact":
            if is_slip:
                current = "slip"
            elif ic:
                current = "stable_contact"
        phases.append(current)
    return phases


class DaimonAdapter(BaseAdapter):
    """戴盟 Daimon-Infinity Parquet → TLabelData"""

    # v0.17: Compliance Level — Daimon有deformation视频估算force_magnitude(L2)，
    # 有shear视频时可合成force_vector(L3)，默认L2
    default_compliance_level: str = "L2"

    @property
    def name(self) -> str:
        return "daimon"

    @property
    def supported_extensions(self):
        return [".parquet"]

    def get_capabilities(self) -> Dict[str, bool]:
        """13 个合法能力键（Schema V2，规范 S-7.4-1）。

        判定口径：以 load() 实际产出的帧内容为准，不按 extract_schema 的
        理论能力。compliance_level 是帧级元字段，不属于能力键（S-7.4-2）。
        戴盟视觉触觉数据集。force_vector / slip_velocity 均为条件能力：
        仅在数据带 shear 视频 / optical flow 时产出。
        """
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": False,
            "force_magnitude": True,  # 由形变视频估算，非标定牛顿值
            "force_vector": True,  # 条件：有 shear 视频时合成 [Fx,Fy,Fz]
            "torque_vector": False,
            "slip_event": True,
            "slip_velocity": True,  # 条件：有 optical flow 时换算滑移速度
            "manipulation_phase": True,
            "texture_class": False,
            "object_deformation": True,  # 由 deformation_magnitude 映射
            "temperature": False,
            "confidence": True,
        }

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "type": "vision-based_tactile",
            "manufacturer": "daimon",
            "model": "DM-TacClaw",
        }

    def extract_schema(self, raw_frame_data: Union[TLabelFrame, Dict]) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2 (14维结构化)

        Daimon适配器策略:
          - contact: 从tlabel_v2.contact或视频特征推断
          - contact_centroid: 从centroid_x估算 [cx, cy]
          - force_magnitude: 从deformation视频估算（视频优先 > gripper估算）
          - force_vector: 如果有shear视频数据可合成 [Fx, Fy, Fz]，否则None
          - object_deformation: 从deformation_magnitude提取
          - compliance_level: 有shear视频时L3，否则L2

        Args:
            raw_frame_data: TLabelFrame实例或tlabel_v2字典

        Returns:
            TLabelSchemaV2 — 14维结构化标注
        """
        # 统一获取 tlabel_v2 字典和 sensor_specific
        if isinstance(raw_frame_data, TLabelFrame):
            v2_dict = raw_frame_data.schema_v2.to_dict() if hasattr(raw_frame_data, 'schema_v2') and raw_frame_data.schema_v2 is not None else raw_frame_data
            sensor_specific = raw_frame_data.sensor_specific or {}
        elif isinstance(raw_frame_data, dict):
            v2_dict = raw_frame_data
            sensor_specific = raw_frame_data.get("sensor_specific", {})
        else:
            raise TypeError(f"raw_frame_data 类型不支持: {type(raw_frame_data)}")

        # 基础字段：复用 from_tlabel_v1 通用映射
        v1_dict = dict(v2_dict)
        v1_dict["confidence"] = v2_dict.get("confidence", 1.0)
        schema = TLabelSchemaV2.from_tlabel_v1(v1_dict)

        # --- Daimon特有增强 ---

        # 1. contact_centroid: v1只有centroid_x，补充 [cx, 0]
        centroid_x = v2_dict.get("centroid_x")
        if centroid_x is not None and schema.contact:
            schema.contact_centroid = [float(centroid_x), 0.0]

        # 2. force_magnitude: 优先使用视频deformation估算
        #    _extract_tlabel_v2_from_state中已计算，直接复用
        force_mag = v2_dict.get("force_magnitude")
        if force_mag is not None and force_mag > 0:
            schema.force_magnitude = float(force_mag)
        else:
            # fallback: 从deformation_magnitude估算
            deform_mag = v2_dict.get("deformation_magnitude", 0.0)
            schema.force_magnitude = float(deform_mag) * 100.0 if deform_mag > 0 else None

        # 3. force_vector: 从force_magnitude + shear方向合成
        #    Daimon有3路视频(deformation/shear/depth)，shear视频提供剪切力方向
        shear_mag = v2_dict.get("shear_field_magnitude", 0.0)
        shear_dir_deg = v2_dict.get("shear_field_direction", 0.0)
        has_shear = shear_mag > 1e-6 and sensor_specific.get("video_type", "").startswith("deformation")

        if schema.force_magnitude is not None and has_shear:
            # 从法向力 + 剪切力方向合成3D力矢量
            # Fz ≈ force_magnitude (法向), Fx/Fy从shear方向分解
            shear_dir_rad = math.radians(shear_dir_deg)
            # 剪切力大小从shear_field_magnitude估算（缩放到N）
            shear_force = float(shear_mag) * 10.0  # 缩放因子
            fx = shear_force * math.cos(shear_dir_rad)
            fy = shear_force * math.sin(shear_dir_rad)
            fz = float(schema.force_magnitude)
            schema.force_vector = [round(fx, 4), round(fy, 4), round(fz, 4)]
            schema.compliance_level = "L3"
        else:
            schema.force_vector = None
            schema.compliance_level = self.default_compliance_level

        # 4. object_deformation: 从deformation_magnitude直接提取
        deform = v2_dict.get("deformation_magnitude")
        if deform is not None and deform > 0:
            schema.object_deformation = float(deform)

        # 5. slip_velocity: 如果有slip_event，从optical_flow提取
        if schema.slip_event:
            of_mag = v2_dict.get("optical_flow_magnitude", 0.0)
            of_dir = v2_dict.get("optical_flow_direction", 0.0)
            if of_mag > 1e-6:
                of_rad = math.radians(of_dir)
                schema.slip_velocity = [round(of_mag * math.cos(of_rad), 4),
                                        round(of_mag * math.sin(of_rad), 4)]

        return schema

    def load(self, file_path: str,
             episode_index: Optional[int] = None,
             max_frames: Optional[int] = None,
             **kwargs) -> TLabelData:
        """
        加载Daimon-Infinity parquet数据

        参数:
            file_path: parquet文件路径或episode目录路径
            episode_index: 指定episode (默认加载第一个)
            max_frames: 最大帧数
        """
        if not HAS_PYARROW:
            raise ImportError("戴盟适配器需要pyarrow: pip install pyarrow")

        path = Path(file_path)

        # 查找parquet文件
        if path.is_dir():
            parquet_path = self._find_parquet(path)
        elif path.suffix == ".parquet":
            parquet_path = path
        else:
            raise ValueError(f"不支持的路径类型: {path}")

        # 加载meta信息
        info = _load_info_json(parquet_path)
        tasks = _load_tasks(parquet_path)
        robot_type = _detect_robot_type(info)
        fps = info.get("fps", 30)
        total_episodes = info.get("total_episodes", 1)

        # 读取parquet
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()

        # 筛选episode
        if episode_index is not None and "episode_index" in df.columns:
            df = df[df["episode_index"] == episode_index]
        elif "episode_index" in df.columns:
            first_ep = df["episode_index"].iloc[0]
            df = df[df["episode_index"] == first_ep]
            episode_index = int(first_ep)

        if max_frames is not None and len(df) > max_frames:
            df = df.head(max_frames)

        num_frames = len(df)

        # 任务描述
        task_idx = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
        task_desc = tasks.get(task_idx, "unknown task")

        # === 加载触觉视频（全链路修复） ===
        video_paths = _find_tactile_videos(parquet_path)
        deform_frames = []
        shear_frames = []
        cam_frames = []  # 普通相机fallback
        video_available = False
        video_type = "none"

        # 优先加载deformation视频
        if "deformation" in video_paths:
            probe = _probe_video(video_paths["deformation"])
            if probe.get("pix_fmt") in ("gbrp16le", "gray16le"):
                # FFV1编码 → 用ffmpeg rawvideo提取
                deform_frames = _extract_ffv1_frames_streaming(
                    video_paths["deformation"],
                    probe["pix_fmt"],
                    probe["width"], probe["height"],
                    max_frames=num_frames,
                )
                if len(deform_frames) > 0:
                    video_available = True
                    video_type = "deformation_ffv1"
            elif probe.get("codec") in ("h264", "mpeg4", "vp9"):
                # 常规编码 → cv2解码
                if HAS_CV2:
                    deform_frames = _extract_video_frames(
                        video_paths["deformation"], max_frames=num_frames
                    )
                    if len(deform_frames) > 0:
                        video_available = True
                        video_type = "deformation_h264"

        # 加载shear视频
        if "shear" in video_paths:
            probe = _probe_video(video_paths["shear"])
            if probe.get("pix_fmt") in ("gbrp16le", "gray16le"):
                shear_frames = _extract_ffv1_frames_streaming(
                    video_paths["shear"],
                    probe["pix_fmt"],
                    probe["width"], probe["height"],
                    max_frames=num_frames,
                )
            elif probe.get("codec") in ("h264", "mpeg4", "vp9") and HAS_CV2:
                shear_frames = _extract_video_frames(
                    video_paths["shear"], max_frames=num_frames
                )

        # fallback: 普通相机做光流
        if not video_available and "cam" in video_paths and HAS_CV2:
            cam_frames = _extract_video_frames(video_paths["cam"], max_frames=num_frames)
            if len(cam_frames) > 0:
                video_available = True
                video_type = "cam_h264"

        # 逐帧处理
        tlabel_frames = []
        frames_contact = []
        frames_slip = []

        states = df["observation.state"].values
        actions = df["action"].values if "action" in df.columns else None
        timestamps = df["timestamp"].values if "timestamp" in df.columns else None
        frame_indices = df["frame_index"].values if "frame_index" in df.columns else np.arange(num_frames)

        prev_tlabel_v2 = None
        prev_deform_frame = None
        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

        for i in range(num_frames):
            state = np.array(states[i])
            action = np.array(actions[i]) if actions is not None else np.zeros(111)
            prev_state = np.array(states[i - 1]) if i > 0 else None

            # --- 视频特征计算 ---
            video_features = {}
            optical_flow_mag = 0.0
            optical_flow_dir = 0.0
            temp_deform_rate = 0.0
            contact_trans = 0.0

            # 帧对齐: 视频帧和parquet帧可能不对齐，用最近邻
            if len(deform_frames) > 0:
                fidx = min(i, len(deform_frames) - 1)
                curr_deform = deform_frames[fidx]
                curr_shear = shear_frames[fidx] if fidx < len(shear_frames) else None

                # 从视频帧计算触觉特征
                video_features = _compute_tactile_features_from_video(
                    curr_deform, curr_shear, prev_deform_frame
                )

                # 光流: 从deformation帧计算触觉光流
                if prev_deform_frame is not None:
                    optical_flow_mag, optical_flow_dir = _compute_optical_flow_from_tactile(
                        prev_deform_frame, curr_deform
                    )

                prev_deform_frame = curr_deform

            elif len(cam_frames) > 0:
                # fallback: 用普通相机做光流
                fidx = min(i, len(cam_frames) - 1)
                curr_cam = cam_frames[fidx]
                if i > 0 and fidx > 0:
                    prev_cam = cam_frames[fidx - 1]
                    if HAS_CV2:
                        try:
                            prev_gray = cv2.cvtColor(prev_cam, cv2.COLOR_BGR2GRAY)
                            curr_gray = cv2.cvtColor(curr_cam, cv2.COLOR_BGR2GRAY)
                            flow = cv2.calcOpticalFlowFarneback(
                                prev_gray, curr_gray, None,
                                0.5, 3, 15, 3, 5, 1.2, 0
                            )
                            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            optical_flow_mag = float(np.mean(mag))
                            optical_flow_dir = float(np.degrees(np.mean(ang)))
                        except Exception:
                            pass

            # temporal_deformation_rate
            if prev_tlabel_v2 is not None and dt > 0:
                prev_deform = prev_tlabel_v2.get("deformation_magnitude", 0.0)
                curr_deform_val = video_features.get("deformation_magnitude", 0.0)
                if curr_deform_val == 0.0:
                    # fallback到state估算
                    state_m = _mask_placeholder(state)
                    finger_data = state_m[67:103]
                    finger_valid = finger_data[~np.isnan(finger_data)]
                    curr_deform_val = float(np.std(finger_valid)) if len(finger_valid) > 1 else 0.0
                temp_deform_rate = abs(curr_deform_val - prev_deform) / dt

            # 构建22维特征
            tlabel_v2 = _extract_tlabel_v2_from_state(
                state, action, prev_state, robot_type,
                video_features=video_features,
                optical_flow_magnitude=optical_flow_mag,
                optical_flow_direction=optical_flow_dir,
                temporal_deformation_rate=temp_deform_rate,
                contact_transition=0.0,  # 后面补算
            )

            # contact_transition
            if prev_tlabel_v2 is not None:
                curr_contact = tlabel_v2["contact"]
                prev_contact = prev_tlabel_v2.get("contact", 0.0)
                curr_area = tlabel_v2.get("contact_area", 0.0)
                prev_area = prev_tlabel_v2.get("contact_area", 0.0)
                tlabel_v2["contact_transition"] = round(
                    min(1.0, abs(curr_contact - prev_contact) +
                        abs(curr_area - prev_area) * 5.0), 4
                )

            frames_contact.append(tlabel_v2["contact"] > 0.5)
            frames_slip.append(tlabel_v2["slip_event"] > 0.5)

            # 传感器特有数据
            state_m = _mask_placeholder(state)
            sensor_specific = {
                "task": task_desc,
                "score": float(df["score"].iloc[i]) if "score" in df.columns else 1.0,
                "gripper_left": float(state_m[65]) if not np.isnan(state_m[65]) else None,
                "gripper_right": float(state_m[66]) if len(state_m) > 66 and not np.isnan(state_m[66]) else None,
                "right_acc": [float(v) for v in state_m[108:111]] if not np.any(np.isnan(state_m[108:111])) else None,
                "right_gyro": [float(v) for v in state_m[111:114]] if not np.any(np.isnan(state_m[111:114])) else None,
                "robot_type": robot_type,
                "video_available": video_available,
                "video_type": video_type,
            }

            confidence = self._compute_confidence(tlabel_v2, video_available)

            frame = TLabelFrame(
                frame_idx=int(frame_indices[i]),
                timestamp_s=round(float(timestamps[i]) if timestamps is not None else i / fps, 4),
                # 走 extract_schema 而不是裸 from_tlabel_v1：后者会丢掉
                # shear 合成 force_vector、slip_velocity 和 compliance_level 增强。
                schema_v2=self.extract_schema(tlabel_v2),
                manipulation_phase="idle",
                confidence=confidence,
                sensor_specific=sensor_specific,
            )
            tlabel_frames.append(frame)
            prev_tlabel_v2 = tlabel_v2

        # 批量推断操作阶段
        phases = _infer_phases(frames_contact, frames_slip)
        for frame, phase in zip(tlabel_frames, phases):
            frame.manipulation_phase = phase

        sensor_info = {
            "type": "vision-based_tactile",
            "model": "DM-TacClaw",
            "manufacturer": "daimon",
            "modality": "vtla_multimodal",
            "layout": {
                "type": robot_type,
                "observation_state_dim": 114,
                "action_dim": 111,
                "fps": fps,
                "total_episodes": total_episodes,
                "placeholder_value": PLACEHOLDER,
                "video_available": video_available,
                "video_type": video_type,
                "videos_found": list(video_paths.keys()),
                "note": "v0.2.0a3: 视频全链路修复（FFV1解码+chunk路径+16位保留）",
            }
        }

        episode_info = {
            "source": "Daimon-Infinity",
            "file": parquet_path.name,
            "robot_type": robot_type,
            "episode_index": episode_index,
            "task": task_desc,
            "fps": fps,
        }

        caps = self.get_capabilities()
        # 无视频时部分维度降级：force_vector 依赖 shear 视频合成，
        # slip_velocity 依赖 optical flow，object_deformation 依赖形变视频。
        if not video_available:
            caps["force_vector"] = False
            caps["slip_velocity"] = False
            caps["object_deformation"] = False

        return TLabelData(
            frames=tlabel_frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=caps,
            sensor_id="daimon_taclaw",
        )

    @staticmethod
    def _find_parquet(directory: Path) -> Path:
        """在目录中查找parquet数据文件"""
        for pattern in ["data/chunk-*/file-*.parquet", "**/*.parquet"]:
            matches = list(directory.glob(pattern))
            if matches:
                return sorted(matches)[0]
        raise FileNotFoundError(f"目录中没有找到parquet文件: {directory}")

    @staticmethod
    def _compute_confidence(tlabel_v2: Dict, video_available: bool = False) -> float:
        """计算标注置信度"""
        if not video_available:
            # 无视频时，数值触觉精度较低
            if tlabel_v2["contact"] < 0.5 and tlabel_v2["slip_event"] < 0.5:
                return 0.95
            if tlabel_v2["contact"] > 0.5:
                return 0.6  # gripper推断接触，不太确定
            return 0.5
        else:
            # 有视频时，特征更可靠
            if tlabel_v2["contact"] < 0.5 and tlabel_v2["slip_event"] < 0.5:
                return 0.98
            if tlabel_v2["contact"] > 0.5 and tlabel_v2["slip_event"] < 0.5:
                return 0.85
            if tlabel_v2["slip_event"] > 0.5:
                return 0.6
            return 0.75
