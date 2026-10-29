"""
白虎-VTouch 适配器 — 将VTouch HDF5数据转换为TLabelData

白虎-VTouch数据集是全球首个跨本体视触觉多模态数据集，
包含视触觉传感器数据（GelSight风格RGB图）、RGB-D数据、关节位姿数据等。

数据来源: https://ai.atomgit.com/openloong/visuo-tactile
论文: arXiv 2604.20444

支持的机器人构型: Qingloong(双足), Wheelloong M1(轮臂), Pika(手持)
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from tlabel.adapters.base import BaseAdapter
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _detect_robot_type(hf) -> str:
    """从HDF5结构推断机器人构型"""
    # Qingloong有leg关节数据
    if 'joints' in hf:
        state = hf['joints'].get('state', {})
        if 'leg' in state:
            return 'qingloong'
        if 'robot' in state:
            return 'wheelloong'
    # Pika有fisheye相机
    if 'cameras' in hf:
        cams = hf['cameras']
        if 'fisheye_left' in cams or 'fisheye_right' in cams:
            return 'pika'
    return 'unknown'


def _get_tactile_sensors(hf) -> List[Dict]:
    """
    发现所有触觉传感器路径
    返回: [{"path": "tactile/hand_left/left", "hand": "left", "side": "left"}, ...]
    """
    sensors = []
    if 'tactile' not in hf:
        return sensors
    tactile = hf['tactile']
    for hand in ['hand_left', 'hand_right']:
        if hand not in tactile:
            continue
        hand_grp = tactile[hand]
        for side in ['left', 'right']:
            if side in hand_grp and 'data' in hand_grp[side]:
                sensors.append({
                    "path": f"tactile/{hand}/{side}",
                    "hand": hand.replace('hand_', ''),
                    "side": side,
                })
    return sensors


def _decode_jpeg_from_h5(raw_bytes) -> Optional[np.ndarray]:
    """从HDF5中解码JPEG图像"""
    if not HAS_CV2:
        return None
    try:
        if isinstance(raw_bytes, bytes):
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        elif isinstance(raw_bytes, np.ndarray):
            if raw_bytes.ndim == 3 and raw_bytes.shape[2] == 3:
                # 已经是RGB numpy数组
                return raw_bytes
            arr = raw_bytes.astype(np.uint8).ravel()
        else:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _decode_tactile_frame(tactile_data, frame_idx: int) -> Optional[np.ndarray]:
    """
    从HDF5 tactile数据集中解码单帧触觉图像

    VTouch的触觉数据存储格式可能是:
    1. N×H×W×3 numpy数组 (直接RGB)
    2. N个JPEG压缩帧的变长数组
    3. 单帧就是H×W×3
    """
    try:
        item = tactile_data[frame_idx]

        if isinstance(item, np.ndarray):
            if item.ndim == 3 and item.shape[2] in (3, 4):
                # 直接RGB/RGBA数组
                return item[:, :, :3] if item.shape[2] == 4 else item
            elif item.ndim == 2:
                # 灰度图 → RGB
                return np.stack([item, item, item], axis=-1)
            elif item.dtype == np.uint8 and item.size > 1000:
                # 可能是JPEG编码的字节流
                img = _decode_jpeg_from_h5(item)
                if img is not None:
                    return img

        if isinstance(item, (bytes, np.bytes_)):
            return _decode_jpeg_from_h5(bytes(item))

        # h5py特殊类型: 变长字节
        if hasattr(item, 'decode'):
            return _decode_jpeg_from_h5(item)

    except Exception:
        pass

    return None


def _get_background(tactile_data, n_frames: int, sample_size: int = 50) -> Optional[np.ndarray]:
    """从触觉数据采样计算背景"""
    if not HAS_CV2 or n_frames == 0:
        return None

    np.random.seed(42)
    sample_idx = np.random.choice(n_frames, min(sample_size, n_frames), replace=False)
    sample_imgs = []

    for idx in sample_idx:
        img = _decode_tactile_frame(tactile_data, idx)
        if img is not None:
            sample_imgs.append(img.astype(np.float32))

    if not sample_imgs:
        return None

    return np.median(np.stack(sample_imgs, axis=0), axis=0)


def _bg_subtract(img: np.ndarray, bg: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """背景减除"""
    if bg is None:
        return img.astype(np.float32) if img is not None else None
    return img.astype(np.float32) - bg


def _extract_tlabel_v2(diff_img, is_contact: bool,
                        optical_flow_magnitude: float = 0.0,
                        optical_flow_direction: float = 0.0,
                        temporal_deformation_rate: float = 0.0,
                        contact_transition: float = 0.0) -> Dict[str, float]:
    """
    从背景减除后的图像提取TLabel v2 22维特征

    复用GelSight适配器的核心特征提取逻辑，
    因为VTouch触觉传感器同样是vision-based tactile (GelSight系)。
    """
    empty = {k: 0.0 for k in [
        "contact", "deformation_magnitude", "force_magnitude", "force_peak",
        "force_direction", "slip_entropy", "slip_event", "texture_energy",
        "edge_density", "contact_area", "centroid_x",
        "normal_field_magnitude", "normal_field_variance",
        "shear_field_magnitude", "shear_field_direction",
        "delta_force_normal", "delta_force_shear", "friction_cone_ratio",
        "optical_flow_magnitude", "optical_flow_direction",
        "temporal_deformation_rate", "contact_transition",
    ]}

    if diff_img is None or diff_img.size == 0:
        return empty

    gray = np.mean(diff_img, axis=2) if diff_img.ndim == 3 else diff_img

    contact = float(is_contact)
    deformation_mag = float(np.sqrt(np.mean(diff_img**2)))
    force_mag = deformation_mag
    force_peak = float(np.max(np.abs(gray)))

    if gray.max() > gray.min():
        gy, gx = np.gradient(gray)
        weight = np.abs(gray)
        total_w = weight.sum()
        if total_w > 0:
            mean_gx = np.sum(gx * weight) / total_w
            mean_gy = np.sum(gy * weight) / total_w
            force_dir = float(np.degrees(np.arctan2(mean_gy, mean_gx)))
        else:
            force_dir = 0.0

        hist, _ = np.histogram(gray.ravel(), bins=32, density=True)
        hist += 1e-10
        slip_ent = float(-np.sum(hist * np.log(hist)))

        grad_angle = np.arctan2(gy, gx)
        angle_var = np.var(grad_angle)
        slip_ev = min(float(angle_var) / 100.0, 1.0)
    else:
        force_dir = 0.0
        slip_ent = 0.0
        slip_ev = 0.0

    texture_e = float(np.mean(gray**2))
    edges = np.abs(np.gradient(gray))
    edge_d = float(np.mean(edges > np.percentile(edges, 90))) if edges.max() > 0 else 0.0

    std_val = np.std(gray)
    threshold = std_val * 2 if std_val > 0 else 1
    contact_a = float(np.mean(np.abs(gray) > threshold))

    col_sums = np.sum(np.abs(gray), axis=0)
    centroid_x = float(np.average(np.arange(gray.shape[1]), weights=col_sums)) / gray.shape[1] if col_sums.sum() > 0 else 0.5

    # normal field
    if diff_img.ndim == 3 and diff_img.shape[0] > 2 and diff_img.shape[1] > 2:
        nf_mag = float(np.sqrt(np.mean(
            diff_img[:, :, 0]**2 + diff_img[:, :, 1]**2 + diff_img[:, :, 2]**2
        )))
        nf_var = float(np.var(np.sqrt(gx**2 + gy**2)))
    else:
        nf_mag = deformation_mag
        nf_var = 0.0

    # shear field
    if diff_img.ndim == 3 and diff_img.shape[0] > 2 and diff_img.shape[1] > 2:
        r_ch = diff_img[:, :, 0]
        g_ch = diff_img[:, :, 1]
        r_gy, r_gx = np.gradient(r_ch)
        g_gy, g_gx = np.gradient(g_ch)
        sf_mag = float(np.sqrt(float(np.mean(np.abs(r_gx)))**2 + float(np.mean(np.abs(g_gy)))**2))
        sf_dir = float(np.degrees(np.arctan2(float(np.mean(np.abs(g_gy))), float(np.mean(np.abs(r_gx))))))
    else:
        sf_mag = 0.0
        sf_dir = 0.0

    friction_ratio = sf_mag / nf_mag if nf_mag > 1e-6 else 0.0

    return {
        "contact": contact,
        "deformation_magnitude": round(deformation_mag, 4),
        "force_magnitude": round(force_mag, 4),
        "force_peak": round(force_peak, 4),
        "force_direction": round(force_dir, 2),
        "slip_entropy": round(slip_ent, 4),
        "slip_event": round(slip_ev, 4),
        "texture_energy": round(texture_e, 4),
        "edge_density": round(edge_d, 4),
        "contact_area": round(contact_a, 4),
        "centroid_x": round(centroid_x, 4),
        "normal_field_magnitude": round(nf_mag, 4),
        "normal_field_variance": round(nf_var, 4),
        "shear_field_magnitude": round(sf_mag, 4),
        "shear_field_direction": round(sf_dir, 2),
        "delta_force_normal": 0.0,  # 需要prev帧，在load中计算
        "delta_force_shear": 0.0,
        "friction_cone_ratio": round(min(friction_ratio, 10.0), 4),
        "optical_flow_magnitude": round(optical_flow_magnitude, 4),
        "optical_flow_direction": round(optical_flow_direction, 2),
        "temporal_deformation_rate": round(temporal_deformation_rate, 4),
        "contact_transition": round(contact_transition, 4),
    }


def _infer_phases(contacts: List[bool], slips: List[bool]) -> List[str]:
    """从接触状态推断操作阶段"""
    phases = []
    current = "idle"
    for ic, is_slip in zip(contacts, slips):
        if current == "idle":
            if ic: current = "initial_contact"
        elif current == "initial_contact":
            if is_slip: current = "slip"
            elif ic: current = "stable_contact"
        elif current == "stable_contact":
            if is_slip: current = "slip"
            elif not ic: current = "release"
        elif current == "slip":
            if not is_slip and ic: current = "stable_contact"
            elif not ic: current = "release"
        elif current == "release":
            if ic: current = "re_contact"
            else: current = "idle"
        elif current == "re_contact":
            if is_slip: current = "slip"
            elif ic: current = "stable_contact"
        phases.append(current)
    return phases


def _is_vtouch_hdf5(file_path: str) -> bool:
    """判断HDF5文件是否为白虎-VTouch格式（与PaXini区分）"""
    if not HAS_H5PY:
        return False
    try:
        with h5py.File(file_path, 'r') as hf:
            # VTouch有tactile/hand_left结构, PaXini有dataset/observation结构
            return 'tactile' in hf or 'cameras' in hf
    except Exception:
        return False


class VTouchAdapter(BaseAdapter):
    """白虎-VTouch HDF5 → TLabelData

    支持三种机器人构型: Qingloong(双足), Wheelloong M1(轮臂), Pika(手持)
    每个构型有4个GelSight风格视触觉传感器
    """

    def detect_image_shape(self, file_path: Optional[str] = None):
        """检测VTouch触觉图像形状

        从HDF5文件中读取第一帧触觉图像确定形状。
        VTouch使用GelSight风格传感器，典型分辨率约(120, 160, 3)。

        Returns:
            (height, width, channels) 或 None
        """
        if file_path is None or not HAS_H5PY:
            return None

        try:
            with h5py.File(file_path, 'r') as hf:
                sensors = _get_tactile_sensors(hf)
                if not sensors:
                    return None

                # 读取第一个传感器的第一帧
                sensor_path = sensors[0]['path']
                tactile_data = hf[f"{sensor_path}/data"]

                img = _decode_tactile_frame(tactile_data, 0)
                if img is not None:
                    return img.shape
        except Exception:
            pass

        return None

    # v0.17: Compliance Level L2 — VTouch有GelSight风格触觉，可从图像估算force_magnitude
    default_compliance_level: str = "L2"

    @property
    def name(self) -> str:
        return "vtouch"

    @property
    def supported_extensions(self):
        return [".h5", ".hdf5"]

    def get_capabilities(self) -> Dict[str, bool]:
        """13 个合法能力键（Schema V2，规范 S-7.4-1）。

        判定口径：以 load() 实际产出的帧内容为准，不按 extract_schema 的
        理论能力。compliance_level 是帧级元字段，不属于能力键（S-7.4-2）。
        VTouch（L2）。无力矢量数据；slip_velocity 由 optical flow 换算。
        """
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": False,
            "force_magnitude": True,  # GelSight 风格形变估算，非标定牛顿值
            "force_vector": False,
            "torque_vector": False,
            "slip_event": True,
            "slip_velocity": True,  # 条件：slip_event 为真时由 optical flow 换算
            "manipulation_phase": True,
            "texture_class": False,
            "object_deformation": True,  # 由 deformation_magnitude 映射
            "temperature": False,
            "confidence": True,
        }

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "type": "vision-based_tactile",
            "manufacturer": "纬钛科技 (Weitai/VTouch)",
            "model": "GelSight-style VTouch Sensor",
        }

    def extract_schema(self, raw_frame_data: Union[TLabelFrame, Dict]) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2 (14维结构化)

        VTouch适配器策略（L2）:
          - contact: 从tlabel_v2.contact推断
          - contact_centroid: 从centroid_x估算 [cx, cy]
          - force_magnitude: 从deformation估算（GelSight风格背景减除）
          - force_vector: None（VTouch无力矢量数据，仅L2）
          - object_deformation: 从deformation_magnitude提取
          - compliance_level: L2

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

        # --- VTouch特有增强 ---

        # 1. contact_centroid: 从centroid_x估算
        centroid_x = v2_dict.get("centroid_x")
        if centroid_x is not None and schema.contact:
            schema.contact_centroid = [float(centroid_x), 0.0]

        # 2. force_magnitude: 从deformation估算（GelSight风格）
        fm = v2_dict.get("force_magnitude")
        if fm is not None and fm > 0:
            schema.force_magnitude = float(fm)

        # 3. force_vector: VTouch无力矢量数据，保持None
        schema.force_vector = None

        # 4. object_deformation: 从deformation_magnitude提取
        deform = v2_dict.get("deformation_magnitude")
        if deform is not None and deform > 0:
            schema.object_deformation = float(deform)

        # 5. slip_velocity: 从optical_flow提取
        if schema.slip_event:
            of_mag = v2_dict.get("optical_flow_magnitude", 0.0)
            of_dir = v2_dict.get("optical_flow_direction", 0.0)
            if of_mag > 1e-6:
                of_rad = math.radians(of_dir)
                schema.slip_velocity = [round(of_mag * math.cos(of_rad), 4),
                                        round(of_mag * math.sin(of_rad), 4)]

        # 6. compliance_level: VTouch始终L2
        schema.compliance_level = self.default_compliance_level

        return schema

    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             hand: str = "left",
             sensor_side: str = "left",
             **kwargs) -> TLabelData:
        """
        加载VTouch HDF5文件，转换为TLabelData

        参数:
            file_path: HDF5文件路径
            trajectory_id: 保留参数（VTouch每文件一个episode）
            hand: 选择哪只手的数据 "left"/"right"
            sensor_side: 选择哪侧触觉传感器 "left"/"right"
            **kwargs: 额外参数

        返回:
            TLabelData — 统一标注容器
        """
        if not HAS_H5PY:
            raise ImportError("VTouch适配器需要h5py: pip install h5py")

        hf = h5py.File(file_path, 'r')
        try:
            return self._parse(hf, file_path, hand=hand, sensor_side=sensor_side, **kwargs)
        finally:
            hf.close()

    def _parse(self, hf, file_path: str, hand: str = "left",
               sensor_side: str = "left", **kwargs) -> TLabelData:
        """解析VTouch HDF5文件"""

        # 检测机器人构型
        robot_type = _detect_robot_type(hf)

        # 读取metadata
        metadata = {}
        try:
            meta_raw = hf['metadata.json']
            if isinstance(meta_raw, h5py.Dataset):
                meta_bytes = meta_raw[()]
                if isinstance(meta_bytes, bytes):
                    metadata = json.loads(meta_bytes.decode('utf-8'))
                else:
                    metadata = json.loads(str(meta_bytes))
        except Exception:
            pass

        # 发现触觉传感器
        sensors = _get_tactile_sensors(hf)
        if not sensors:
            raise ValueError(
                f"未找到VTouch触觉传感器数据\n"
                f"请确认HDF5文件包含 tactile/hand_left 或 tactile/hand_right 路径\n"
                f"当前文件顶层keys: {list(hf.keys())}"
            )

        # 选择目标传感器
        target_hand = f"hand_{hand}"
        target_sensor = None
        for s in sensors:
            if s['hand'] == hand and s['side'] == sensor_side:
                target_sensor = s
                break

        if target_sensor is None:
            # 降级到第一个可用传感器
            target_sensor = sensors[0]
            hand = target_sensor['hand']
            sensor_side = target_sensor['side']
            target_hand = f"hand_{hand}"

        # 加载触觉图像序列
        tactile_data = hf[f"{target_sensor['path']}/data"]
        n_frames = tactile_data.shape[0] if hasattr(tactile_data, 'shape') else len(tactile_data)

        # 采样率（从metadata或默认30Hz）
        sample_rate = metadata.get('camera_fps', 30.0)
        if isinstance(sample_rate, (list, np.ndarray)):
            sample_rate = float(sample_rate[0])
        dt = 1.0 / sample_rate

        # 计算背景
        background = _get_background(tactile_data, n_frames, sample_size=100)

        # 加载关节数据（可选）
        joint_positions = None
        joint_efforts = None
        gripper_positions = None
        try:
            if 'joints' in hf:
                state = hf['joints'].get('state', {})
                action = hf['joints'].get('action', {})

                # 关节位置
                arm_state = state.get('arm', {})
                if 'position' in arm_state:
                    joint_positions = arm_state['position'][:]

                # 关节力矩
                if 'effort' in arm_state:
                    joint_efforts = arm_state['effort'][:]

                # 夹爪
                effector = state.get('effector', action.get('effector', {}))
                if 'position' in effector:
                    gripper_positions = effector['position'][:]
        except Exception:
            pass

        # 逐帧提取特征
        frames_info = []
        all_diff_imgs = []

        for i in range(n_frames):
            img = _decode_tactile_frame(tactile_data, i)
            diff_img = _bg_subtract(img, background) if img is not None else None

            # 判断接触: deformation > 阈值
            is_contact = False
            if diff_img is not None:
                deformation = float(np.sqrt(np.mean(diff_img**2)))
                is_contact = deformation > 5.0  # 与GelSight适配器一致的阈值

            frames_info.append({
                "is_contact": is_contact,
                "diff_img": diff_img,
            })
            all_diff_imgs.append(diff_img)

        # 滑移检测
        slips = []
        for i, fi in enumerate(frames_info):
            is_slip = False
            if i > 0 and fi["is_contact"] and frames_info[i-1]["is_contact"]:
                curr = frames_info[i]["diff_img"]
                prev = frames_info[i-1]["diff_img"]
                if curr is not None and prev is not None:
                    frame_diff = float(np.sqrt(np.mean((curr - prev)**2)))
                    is_slip = frame_diff > 3.0  # 帧间差异阈值
            slips.append(is_slip)

        # 推断操作阶段
        contacts = [fi["is_contact"] for fi in frames_info]
        phases = _infer_phases(contacts, slips)

        # 构建TLabelFrame列表
        tlabel_frames = []
        prev_img = None
        prev_deformation = 0.0
        prev_contact = 0.0
        prev_contact_area = 0.0

        for i, fi in enumerate(frames_info):
            diff_img = fi["diff_img"]

            # 时序4维计算
            optical_flow_mag = 0.0
            optical_flow_dir = 0.0
            temp_deform_rate = 0.0
            contact_trans = 0.0

            # 光流计算
            if HAS_CV2 and i > 0:
                prev_img_decoded = _decode_tactile_frame(tactile_data, i - 1)
                curr_img_decoded = _decode_tactile_frame(tactile_data, i)
                if prev_img_decoded is not None and curr_img_decoded is not None:
                    try:
                        prev_gray = cv2.cvtColor(prev_img_decoded, cv2.COLOR_BGR2GRAY)
                        curr_gray = cv2.cvtColor(curr_img_decoded, cv2.COLOR_BGR2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        optical_flow_mag = float(np.mean(mag))
                        optical_flow_dir = float(np.degrees(np.mean(ang)))
                    except Exception:
                        pass

            # temporal_deformation_rate
            if diff_img is not None:
                curr_deformation = float(np.sqrt(np.mean(diff_img**2)))
                if prev_deformation > 0 and dt > 0:
                    temp_deform_rate = abs(curr_deformation - prev_deformation) / dt
                prev_deformation = curr_deformation

            # contact_transition
            curr_contact_val = 1.0 if fi["is_contact"] else 0.0
            curr_contact_area = 0.0
            if diff_img is not None:
                gray = np.mean(diff_img, axis=2) if diff_img.ndim == 3 else diff_img
                std_val = np.std(gray)
                threshold = std_val * 2 if std_val > 0 else 1
                curr_contact_area = float(np.mean(np.abs(gray) > threshold))
            contact_trans = min(1.0, abs(curr_contact_val - prev_contact) +
                               abs(curr_contact_area - prev_contact_area) * 5.0)

            # delta_force计算
            delta_fn = 0.0
            delta_fs = 0.0
            if i > 0 and all_diff_imgs[i-1] is not None and diff_img is not None:
                prev_gray = np.mean(all_diff_imgs[i-1], axis=2) if all_diff_imgs[i-1].ndim == 3 else all_diff_imgs[i-1]
                curr_gray = np.mean(diff_img, axis=2) if diff_img.ndim == 3 else diff_img
                delta_fn = float(np.sqrt(np.mean((curr_gray - prev_gray)**2)))

                # shear delta
                if diff_img.ndim == 3 and all_diff_imgs[i-1].ndim == 3:
                    from tlabel.adapters.gelsight import _compute_shear_field
                    prev_sf = _compute_shear_field(all_diff_imgs[i-1])[0]
                    curr_sf = _compute_shear_field(diff_img)[0]
                    delta_fs = abs(curr_sf - prev_sf)

            # 提取22维特征
            tlabel_v2 = _extract_tlabel_v2(
                diff_img, fi["is_contact"],
                optical_flow_magnitude=optical_flow_mag,
                optical_flow_direction=optical_flow_dir,
                temporal_deformation_rate=temp_deform_rate,
                contact_transition=contact_trans,
            )

            # 填充delta值
            tlabel_v2["delta_force_normal"] = round(delta_fn, 4)
            tlabel_v2["delta_force_shear"] = round(delta_fs, 4)

            # 置信度
            confidence = 0.95
            if fi["is_contact"] and slips[i]:
                confidence = 0.5
            elif fi["is_contact"]:
                confidence = 0.8

            # 传感器特有数据
            sensor_specific = {}
            if joint_positions is not None and i < len(joint_positions):
                sensor_specific["joint_positions"] = [round(float(x), 4) for x in joint_positions[i]]
            if joint_efforts is not None and i < len(joint_efforts):
                sensor_specific["joint_efforts"] = [round(float(x), 4) for x in joint_efforts[i]]
            if gripper_positions is not None and i < len(gripper_positions):
                sensor_specific["gripper_positions"] = [round(float(x), 4) for x in gripper_positions[i]]

            frame = TLabelFrame(
                frame_idx=i,
                timestamp_s=round(i * dt, 4),
                # 走 extract_schema 而不是裸 from_tlabel_v1：后者会丢掉
                # slip_velocity 提取和 compliance_level（恒写 L1）等增强。
                schema_v2=self.extract_schema(tlabel_v2),
                manipulation_phase=phases[i],
                confidence=confidence,
                sensor_specific=sensor_specific,
            )
            tlabel_frames.append(frame)
            prev_contact = curr_contact_val
            prev_contact_area = curr_contact_area

        # 统计
        contact_count = sum(1 for c in contacts if c)
        slip_count = sum(1 for s in slips if s)

        # 任务名称（从文件路径提取）
        file_name = Path(file_path).stem
        parent_name = Path(file_path).parent.name

        sensor_info = {
            "type": "vision-based_tactile",
            "model": "GelSight-style VTouch",
            "manufacturer": "纬钛科技 (Weitai)",
            "modality": "vision-based_tactile",
            "layout": {
                "type": "dual_hand_dual_sensor",
                "hand": hand,
                "sensor_side": sensor_side,
                "robot_type": robot_type,
                "total_sensors": len(sensors),
                "available_sensors": [f"{s['hand']}_{s['side']}" for s in sensors],
                "sampling_rate_hz": round(sample_rate, 1),
            }
        }

        episode_info = {
            "source": "baihu-vtouch",
            "file": file_name,
            "task": parent_name,
            "robot_type": robot_type,
            "metadata": metadata,
            "total_frames": n_frames,
            "contact_frames": contact_count,
            "slip_frames": slip_count,
        }

        return TLabelData(
            frames=tlabel_frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
            sensor_id=f"vtouch_{hand}_{sensor_side}",
        )
