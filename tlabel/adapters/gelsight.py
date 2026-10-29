"""
GelSight/DIGIT 适配器 — 将force pkl数据转换为TLabelData

复用gelsight_adapter.py的核心逻辑，封装为BaseAdapter接口。
DIGIT与GelSight共用此适配器（force数据集schema一致）。
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from tlabel.adapters.base import BaseAdapter
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _decode_jpeg(raw_bytes):
    if not HAS_CV2:
        return None
    try:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None


def _get_background(images_list, sample_size=50):
    n = len(images_list)
    if n == 0:
        return None
    np.random.seed(42)
    sample_idx = np.random.choice(n, min(sample_size, n), replace=False)
    sample_imgs = []
    for idx in sample_idx:
        img = _decode_jpeg(images_list[idx])
        if img is not None:
            sample_imgs.append(img.astype(np.float32))
    if not sample_imgs:
        return None
    return np.median(np.stack(sample_imgs, axis=0), axis=0)


def _bg_subtract(img, bg):
    if bg is None:
        return img.astype(np.float32) if img is not None else None
    return img.astype(np.float32) - bg


def _compute_normal_field(diff_img):
    if diff_img is None or diff_img.size == 0:
        return 0.0, 0.0
    normal_mag = float(np.sqrt(np.mean(
        diff_img[:, :, 0]**2 + diff_img[:, :, 1]**2 + diff_img[:, :, 2]**2
    )))
    gray = np.mean(diff_img, axis=2)
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)
        normal_var = float(np.var(grad_mag))
    else:
        normal_var = 0.0
    return normal_mag, normal_var


def _compute_shear_field(diff_img):
    if diff_img is None or diff_img.size == 0:
        return 0.0, 0.0
    if diff_img.shape[0] < 3 or diff_img.shape[1] < 3:
        return 0.0, 0.0
    r_ch = diff_img[:, :, 0]
    g_ch = diff_img[:, :, 1]
    r_gy, r_gx = np.gradient(r_ch)
    g_gy, g_gx = np.gradient(g_ch)
    shear_x = float(np.mean(np.abs(r_gx)))
    shear_y = float(np.mean(np.abs(g_gy)))
    shear_mag = np.sqrt(shear_x**2 + shear_y**2)
    shear_dir = float(np.degrees(np.arctan2(shear_y, shear_x)))
    return shear_mag, shear_dir


def _extract_tlabel_v2(diff_img, is_contact, delta_mag_shear=None,
                        delta_mag_normal=None, coef_friction=None,
                        prev_diff_img=None,
                        # --- 时序4维新参数 ---
                        optical_flow_magnitude=0.0,
                        optical_flow_direction=0.0,
                        temporal_deformation_rate=0.0,
                        contact_transition=0.0):
    """从单帧背景减除后的图像提取TLabel v2 22维特征"""
    empty = {k: 0.0 for k in [
        "contact", "deformation_magnitude", "force_magnitude", "force_peak",
        "force_direction", "slip_entropy", "slip_event", "texture_energy",
        "edge_density", "contact_area", "centroid_x",
        "normal_field_magnitude", "normal_field_variance",
        "shear_field_magnitude", "shear_field_direction",
        "delta_force_normal", "delta_force_shear", "friction_cone_ratio",
        # --- 时序4维 ---
        "optical_flow_magnitude", "optical_flow_direction",
        "temporal_deformation_rate", "contact_transition",
        # --- v0.7: force_magnitude 诚实替代 ---
        "deformation_magnitude_peak",
    ]}

    if diff_img is None or diff_img.size == 0:
        # even no_contact frames need global stats for classifier
        return empty

    gray = np.mean(diff_img, axis=2)

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

    nf_mag, nf_var = _compute_normal_field(diff_img)
    sf_mag, sf_dir = _compute_shear_field(diff_img)

    if delta_mag_normal is not None:
        delta_fn = float(delta_mag_normal)
    elif prev_diff_img is not None:
        prev_gray = np.mean(prev_diff_img, axis=2)
        delta_fn = float(np.sqrt(np.mean((gray - prev_gray)**2)))
    else:
        delta_fn = 0.0

    if delta_mag_shear is not None:
        delta_fs = float(delta_mag_shear)
    elif prev_diff_img is not None:
        prev_sf = _compute_shear_field(prev_diff_img)[0]
        delta_fs = abs(sf_mag - prev_sf)
    else:
        delta_fs = 0.0

    friction_ratio = sf_mag / nf_mag if nf_mag > 1e-6 else 0.0

    return {
        "contact": contact,
        "deformation_magnitude": round(deformation_mag, 4),
        "force_magnitude": round(force_mag, 4),  # deprecated in v0.7, kept for backward compat
        "deformation_magnitude_peak": round(force_mag, 4),  # v0.7: honest replacement for force_magnitude
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
        "delta_force_normal": round(delta_fn, 4),
        "delta_force_shear": round(delta_fs, 4),
        "friction_cone_ratio": round(min(friction_ratio, 10.0), 4),
        # --- 时序4维 ---
        "optical_flow_magnitude": round(optical_flow_magnitude, 4),
        "optical_flow_direction": round(optical_flow_direction, 2),
        "temporal_deformation_rate": round(temporal_deformation_rate, 4),
        "contact_transition": round(contact_transition, 4),
    }


def _infer_phases(frames_info):
    """从接触状态推断操作阶段"""
    phases = []
    current = "idle"
    for fi in frames_info:
        ic, is_slip = fi["is_contact"], fi["is_slip"]
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


class GelSightAdapter(BaseAdapter):
    """GelSight/DIGIT force pkl → TLabelData

    Compliance Level: L2（默认，无标定数据时）/ L3（有标定数据可提取 force_vector）
    """

    default_compliance_level: str = "L2"

    @property
    def name(self) -> str:
        return "gelsight"

    @property
    def supported_extensions(self):
        return [".pkl", ".pickle"]

    def extract_schema(self, raw_frame_data) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2（14维）

        参数:
            raw_frame_data: dict，包含以下键：
                - diff_img: ndarray 或 None 背景减除后的图像
                - is_contact: bool 是否接触
                - force_vector_N: list[float] 或 None [Fx,Fy,Fz] 标定力向量
                - delta_mag_shear: float 或 None
                - delta_mag_normal: float 或 None
                - coef_friction: float 或 None
                - prev_diff_img: ndarray 或 None 上一帧背景减除图像
                - optical_flow_magnitude: float（默认0.0）
                - optical_flow_direction: float（默认0.0）
                - temporal_deformation_rate: float（默认0.0）
                - contact_transition: float（默认0.0）

        返回:
            TLabelSchemaV2 — L2（默认）或 L3（有标定力向量时），
            force_magnitude 从变形区域估算，force_vector 有标定数据则填，
            object_deformation 从图像提取
        """
        # 复用现有模块级 _extract_tlabel_v2 函数获取22维dict
        tlabel_v2 = _extract_tlabel_v2(
            diff_img=raw_frame_data["diff_img"],
            is_contact=raw_frame_data["is_contact"],
            delta_mag_shear=raw_frame_data.get("delta_mag_shear"),
            delta_mag_normal=raw_frame_data.get("delta_mag_normal"),
            coef_friction=raw_frame_data.get("coef_friction"),
            prev_diff_img=raw_frame_data.get("prev_diff_img"),
            optical_flow_magnitude=raw_frame_data.get("optical_flow_magnitude", 0.0),
            optical_flow_direction=raw_frame_data.get("optical_flow_direction", 0.0),
            temporal_deformation_rate=raw_frame_data.get("temporal_deformation_rate", 0.0),
            contact_transition=raw_frame_data.get("contact_transition", 0.0),
        )

        contact = float(tlabel_v2["contact"]) > 0.5
        centroid_x = tlabel_v2.get("centroid_x", 0.5)

        # 判断是否有标定力向量 → 动态升级 compliance_level
        force_vector_N = raw_frame_data.get("force_vector_N")
        if force_vector_N is not None and len(force_vector_N) == 3:
            compliance_level = "L3"
            force_vector = [float(v) for v in force_vector_N]
        else:
            compliance_level = self.default_compliance_level  # L2
            force_vector = None

        return TLabelSchemaV2(
            contact=contact,
            contact_centroid=[float(centroid_x), 0.5] if contact else None,
            contact_region=None,
            force_magnitude=float(tlabel_v2["deformation_magnitude"]),
            force_vector=force_vector,
            torque_vector=None,
            slip_event=float(tlabel_v2["slip_event"]) > 0.5,
            slip_velocity=None,
            manipulation_phase=None,
            texture_class=None,
            object_deformation=float(tlabel_v2["deformation_magnitude"]),
            temperature=None,
            confidence=0.8 if contact else 0.95,
            compliance_level=compliance_level,
        )

    def get_capabilities(self) -> Dict[str, bool]:
        """13 个合法能力键（Schema V2，规范 S-7.4-1）。

        判定口径：以 load() 实际产出的帧内容为准，不按 extract_schema 的
        理论能力。compliance_level 是帧级元字段，不属于能力键（S-7.4-2）。
        视觉触觉（GelSight / DIGIT）。force_magnitude 是形变代理量、非 SI 标定值；
        force_vector 仅在 pkl 携带 force_vector_N 时产出，属条件能力。
        """
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": False,
            "force_magnitude": True,  # 视觉形变代理，非标定牛顿值
            "force_vector": True,  # 条件：pkl 含 force_vector_N 时产出 3D 力
            "torque_vector": False,
            "slip_event": True,  # 图像梯度角方差代理，非物理滑移检测
            "slip_velocity": False,
            "manipulation_phase": True,
            "texture_class": False,
            "object_deformation": True,  # 由 deformation_magnitude 映射
            "temperature": False,
            "confidence": True,
        }

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "type": "vision-based_tactile",
            "manufacturer": "GelSight Inc.",
        }

    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             **kwargs) -> TLabelData:
        pkl_path = Path(file_path)

        # 检测传感器类型
        is_digit = "digit" in str(pkl_path).lower()
        sensor_model = "DIGIT" if is_digit else "GelSight Mini"
        sensor_res = "160x120" if is_digit else "240x320"
        sample_rate = 60 if is_digit else 25

        # 加载标签数据
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if "trajectories" not in data or "in_contact" not in data:
            raise ValueError("不是有效的GelSight/DIGIT force数据格式")

        in_contact = data["in_contact"]
        trajectories = data["trajectories"]
        traj_keys = sorted(trajectories.keys(), key=lambda x: int(x) if isinstance(x, str) else x)

        if not traj_keys:
            raise ValueError("数据中没有轨迹")

        selected_key = trajectory_id if trajectory_id is not None else traj_keys[0]
        if selected_key not in trajectories:
            raise ValueError(f"轨迹 {selected_key} 不存在")

        traj = trajectories[selected_key]
        indexes = traj["indexes"]
        forces = traj["forces"]
        slip_label = traj["slip_label"]
        poses = traj.get("poses", None)
        delta_mag_shear = traj.get("delta_mag_shear", None)
        delta_mag_normal = traj.get("delta_mag_normal", None)
        coef_friction = traj.get("coef_friction", None)

        n_frames = min(len(indexes), len(forces), len(slip_label))

        # 加载图像数据（同目录下的dataset_gelsight_XX.pkl）
        image_dir = pkl_path.parent
        all_images = []
        for i in ["00", "01", "02", "03"]:
            img_file = image_dir / f"dataset_gelsight_{i}.pkl"
            if not img_file.exists():
                # 也试试digit
                img_file = image_dir / f"dataset_digit_{i}.pkl"
            if not img_file.exists():
                continue
            with open(img_file, "rb") as f:
                batch = pickle.load(f)
            if isinstance(batch, np.ndarray):
                batch = list(batch)
            all_images.extend(batch)

        # 计算背景
        background = _get_background(all_images, sample_size=100) if all_images else None

        # 构建帧信息
        frames_info = []
        for i in range(n_frames):
            gidx = indexes[i]
            ic = bool(in_contact[gidx]) if gidx < len(in_contact) else False
            frames_info.append({
                "global_idx": gidx,
                "is_contact": ic,
                "is_slip": bool(slip_label[i]),
                "delta_mag_shear": float(delta_mag_shear[i]) if delta_mag_shear is not None and i < len(delta_mag_shear) else None,
                "delta_mag_normal": float(delta_mag_normal[i]) if delta_mag_normal is not None and i < len(delta_mag_normal) else None,
                "coef_friction": float(coef_friction) if coef_friction is not None else None,
            })

        phases = _infer_phases(frames_info)

        # 提取TLabel v2特征
        tlabel_frames = []
        prev_diff = None
        prev_img = None
        prev_contact = 0.0
        prev_contact_area = 0.0
        prev_deformation = 0.0
        dt = 1.0 / sample_rate  # 帧间隔

        for i, fi in enumerate(frames_info):
            gidx = fi["global_idx"]

            if gidx < len(all_images):
                img = _decode_jpeg(all_images[gidx])
                diff_img = _bg_subtract(img, background) if img is not None else None
            else:
                img = None
                diff_img = None

            # --- 时序4维计算 ---
            optical_flow_mag = 0.0
            optical_flow_dir = 0.0
            temp_deform_rate = 0.0
            contact_trans = 0.0

            if HAS_CV2 and prev_img is not None and img is not None:
                # Farneback光流计算
                prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    optical_flow_mag = float(np.mean(mag))
                    optical_flow_dir = float(np.degrees(np.mean(ang)))
                except Exception:
                    pass  # 光流计算失败则保持0

            # temporal_deformation_rate: 帧间deformation差分
            if prev_deformation > 0 and diff_img is not None:
                curr_deform = float(np.sqrt(np.mean(diff_img**2)))
                temp_deform_rate = abs(curr_deform - prev_deformation) / dt

            # contact_transition: 帧间contact + contact_area变化
            curr_contact = 1.0 if fi["is_contact"] else 0.0
            curr_contact_area = temp_deform_rate  # HACK: 用diff_img范数近似
            contact_trans = min(1.0, abs(curr_contact - prev_contact) +
                               abs(curr_contact_area - prev_contact_area) * 5.0)

            tlabel_v2 = _extract_tlabel_v2(
                diff_img, fi["is_contact"],
                delta_mag_shear=fi["delta_mag_shear"],
                delta_mag_normal=fi["delta_mag_normal"],
                coef_friction=fi["coef_friction"],
                prev_diff_img=prev_diff,
                optical_flow_magnitude=optical_flow_mag,
                optical_flow_direction=optical_flow_dir,
                temporal_deformation_rate=temp_deform_rate,
                contact_transition=contact_trans,
            )

            # 计算置信度
            confidence = self._compute_confidence(fi, tlabel_v2)

            sensor_specific = {
                "force_vector_N": [round(float(forces[i][j]), 4) for j in range(3)],
                "slip_label": fi["is_slip"],
            }
            if poses is not None and i < len(poses):
                sensor_specific["pose"] = [round(float(poses[i][j]), 4) for j in range(len(poses[i]))]
            if fi["coef_friction"] is not None:
                sensor_specific["coef_friction"] = fi["coef_friction"]

            # 构造 raw_frame_data 用于 extract_schema，让真实力向量进入标准字段
            raw_frame_for_schema = {
                "diff_img": diff_img,
                "is_contact": fi["is_contact"],
                "force_vector_N": [round(float(forces[i][j]), 4) for j in range(3)],
                "delta_mag_shear": fi["delta_mag_shear"],
                "delta_mag_normal": fi["delta_mag_normal"],
                "coef_friction": fi["coef_friction"],
                "prev_diff_img": prev_diff,
                "optical_flow_magnitude": optical_flow_mag,
                "optical_flow_direction": optical_flow_dir,
                "temporal_deformation_rate": temp_deform_rate,
                "contact_transition": contact_trans,
            }

            frame = TLabelFrame(
                frame_idx=gidx,
                timestamp_s=round(gidx / sample_rate, 4),  # 修复: 用实际采样率替代硬编码 30.0
                schema_v2=self.extract_schema(raw_frame_for_schema),  # 修复P0: 用 extract_schema 替代 from_tlabel_v1
                manipulation_phase=phases[i],
                confidence=confidence,
                sensor_specific=sensor_specific,
                image=img,  # v0.12: 保留原始图像用于可视化
            )
            tlabel_frames.append(frame)
            # 更新prev状态
            prev_diff = diff_img
            prev_img = img
            prev_contact = 1.0 if fi["is_contact"] else 0.0
            prev_contact_area = curr_contact_area
            prev_deformation = float(np.sqrt(np.mean(diff_img**2))) if diff_img is not None else 0.0

        contact_count = sum(1 for fi in frames_info if fi["is_contact"])
        slip_count = sum(1 for fi in frames_info if fi["is_slip"])

        sensor_info = {
            "type": "DIGIT" if is_digit else "GelSight",
            "model": sensor_model,
            "manufacturer": "Meta FAIR" if is_digit else "GelSight Inc.",
            "modality": "vision-based_tactile",
            "layout": {
                "type": "single_sensor",
                "resolution": sensor_res,
                "sampling_rate_hz": sample_rate,
            }
        }

        episode_info = {
            "source": f"facebook/{'digit' if is_digit else 'gelsight'}-force-estimation",
            "total_trajectories": len(traj_keys),
            "selected_trajectory": selected_key,
            "trajectory_ids": [int(k) if isinstance(k, str) else k for k in traj_keys],
        }

        return TLabelData(
            frames=tlabel_frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
            sensor_id="gelsight_main",  # 默认传感器ID，用户可覆盖
        )

    def detect_image_shape(self, file_path: Optional[str] = None):
        """检测GelSight/DIGIT触觉图像形状

        根据传感器型号返回已知分辨率：
          - DIGIT: (120, 160, 3)
          - GelSight Mini: (240, 320, 3)

        如果提供 file_path，尝试从数据文件中读取第一帧确认。

        Returns:
            (height, width, channels) 或 None
        """
        # 尝试从文件路径推断传感器类型
        if file_path is not None:
            path_str = str(file_path).lower()
            if "digit" in path_str:
                return (120, 160, 3)
            elif "gelsight" in path_str:
                return (240, 320, 3)

            # 尝试从pkl文件读取第一帧
            try:
                pkl_path = Path(file_path)
                if pkl_path.exists():
                    # 查找同目录下的图像pkl
                    image_dir = pkl_path.parent
                    for prefix in ["dataset_gelsight_", "dataset_digit_"]:
                        for suffix in ["00.pkl", "01.pkl"]:
                            img_file = image_dir / f"{prefix}{suffix}"
                            if img_file.exists():
                                with open(img_file, "rb") as f:
                                    batch = pickle.load(f)
                                if isinstance(batch, np.ndarray) and len(batch) > 0:
                                    img = _decode_jpeg(batch[0])
                                    if img is not None:
                                        return img.shape
                                elif isinstance(batch, list) and len(batch) > 0:
                                    img = _decode_jpeg(batch[0])
                                    if img is not None:
                                        return img.shape
            except Exception:
                pass

        # 默认返回GelSight Mini分辨率
        return (240, 320, 3)

    def _compute_confidence(self, frame_info, tlabel_v2):
        """计算标注置信度"""
        # 简单启发式：接触=0且slip=0 → 高置信度
        # 接触过渡帧 → 低置信度
        if not frame_info["is_contact"] and not frame_info["is_slip"]:
            return 0.95
        if frame_info["is_contact"] and not frame_info["is_slip"]:
            return 0.8
        # 滑移帧最不确定
        if frame_info["is_slip"]:
            return 0.5
        return 0.7
