"""
YCB-Slide 适配器 — 将 CMU RPL YCB-Slide 数据转换为 TLabelData

数据来源: MidasTouch (CoRL 2022), Carnegie Mellon University RI
作者: Sudharshan Suresh et al.
数据格式: synced_data.npy (real) / tactile_data.pkl (sim) + JPEG 触觉帧
传感器: DIGIT (vision-based tactile)

数据目录结构:
    Real data:
    real/
    ├── 004_sugar_box/
    │   ├── dataset_0/
    │   │   ├── synced_data.npy   ← timestamps, digit_frames, poses
    │   │   ├── digit/            ← DIGIT JPEG images
    │   │   └── webcam/           ← webcam JPEG images
    │   ├── dataset_1/ ... dataset_4/
    ├── 005_tomato_soup_can/ ...

    Sim data:
    sim/
    ├── 004_sugar_box/
    │   ├── 00/
    │   │   ├── tactile_data.pkl  ← gelposes_meas, gelposes, camposes
    │   │   └── tactile_images/   ← rendered tactile JPEG images
    │   ├── 01/ ...
    ├── 005_tomato_soup_can/ ...

synced_data.npy 字段:
    - timestamps: list[float] — 同步时间戳
    - digit_frames: list[str] — DIGIT 图像相对路径
    - webcam_frames: list[str] — webcam 图像相对路径
    - poses: dict[str, ndarray(N,7)] — {rigid_body_name: [x,y,z,qx,qy,qz,qw]}

引用:
    @inproceedings{suresh2022midastouch,
        title={MidasTouch: {M}onte-{C}arlo Inference over Distributions
               across Sliding Touch for Robust Physical Property Estimation},
        author={Suresh, Sudharshan and Si, Zihan and Wu, Yanan and
                Srinivasa, Siddhartha S.},
        booktitle={CoRL},
        year={2022}
    }
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from tlabel.adapters.base import BaseAdapter
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import dill as pickle
    HAS_DILL = True
except ImportError:
    try:
        import pickle
        HAS_DILL = True
    except ImportError:
        HAS_DILL = False


# YCB-Slide 物体列表 (10 个标准 YCB 物体)
YCB_OBJECTS = [
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "021_bleach_cleanser",
    "025_mug",
    "035_power_drill",
    "037_scissors",
    "042_adjustable_wrench",
    "048_hammer",
    "055_baseball",
]

# DIGIT 传感器参数
DIGIT_RESOLUTION = (240, 320)  # H x W
DIGIT_FPS = 30.0


def _load_image(image_path: str) -> Optional[np.ndarray]:
    """读取 JPEG 图像"""
    if not HAS_CV2:
        return None
    try:
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        return None


def _compute_background(images: List[str], sample_size: int = 30
                        ) -> Optional[np.ndarray]:
    """从采样图像计算背景帧"""
    if not images or not HAS_CV2:
        return None
    np.random.seed(42)
    indices = np.random.choice(len(images), min(sample_size, len(images)),
                               replace=False)
    sample_imgs = []
    for idx in indices:
        img = _load_image(images[idx])
        if img is not None:
            sample_imgs.append(img.astype(np.float32))
    if not sample_imgs:
        return None
    return np.median(np.stack(sample_imgs, axis=0), axis=0)


def _bg_subtract(img: Optional[np.ndarray],
                 bg: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """背景减除"""
    if img is None or bg is None:
        return img.astype(np.float32) if img is not None else None
    return img.astype(np.float32) - bg


def _extract_tlabel_v2(diff_img: Optional[np.ndarray],
                       is_contact: bool) -> Dict[str, float]:
    """从背景减除后的图像提取 TLabel v2 22维特征 (简化版)"""
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
        empty["contact"] = 1.0 if is_contact else 0.0
        return empty

    gray = np.mean(diff_img, axis=2)

    contact = float(is_contact)
    deformation_mag = float(np.sqrt(np.mean(diff_img ** 2)))
    force_mag = deformation_mag
    force_peak = float(np.max(np.abs(gray)))

    # 力方向
    force_dir = 0.0
    slip_ent = 0.0
    slip_ev = 0.0
    if gray.max() > gray.min():
        gy, gx = np.gradient(gray)
        weight = np.abs(gray)
        total_w = weight.sum()
        if total_w > 0:
            mean_gx = np.sum(gx * weight) / total_w
            mean_gy = np.sum(gy * weight) / total_w
            force_dir = float(np.degrees(np.arctan2(mean_gy, mean_gx)))

        hist, _ = np.histogram(gray.ravel(), bins=32, density=True)
        hist += 1e-10
        slip_ent = float(-np.sum(hist * np.log(hist)))

        grad_angle = np.arctan2(gy, gx)
        angle_var = np.var(grad_angle)
        slip_ev = min(float(angle_var) / 100.0, 1.0)

    texture_e = float(np.mean(gray ** 2))

    edges = np.abs(np.gradient(gray))
    edge_d = float(np.mean(edges > np.percentile(edges, 90)
                           )) if edges.max() > 0 else 0.0

    std_val = np.std(gray)
    threshold = std_val * 2 if std_val > 0 else 1
    contact_a = float(np.mean(np.abs(gray) > threshold))

    col_sums = np.sum(np.abs(gray), axis=0)
    centroid_x = (float(np.average(np.arange(gray.shape[1]),
                                   weights=col_sums)) / gray.shape[1]
                  if col_sums.sum() > 0 else 0.5)

    # 法向场
    nf_mag = float(np.sqrt(np.mean(
        diff_img[:, :, 0]**2 + diff_img[:, :, 1]**2 + diff_img[:, :, 2]**2
    )))
    nf_var = 0.0
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)
        nf_var = float(np.var(grad_mag))

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
        "shear_field_magnitude": 0.0,
        "shear_field_direction": 0.0,
        "delta_force_normal": 0.0,
        "delta_force_shear": 0.0,
        "friction_cone_ratio": 0.0,
        "optical_flow_magnitude": 0.0,
        "optical_flow_direction": 0.0,
        "temporal_deformation_rate": 0.0,
        "contact_transition": 0.0,
    }


def _detect_contact_from_image(diff_img: Optional[np.ndarray],
                               threshold_factor: float = 2.0) -> bool:
    """从图像差异检测接触状态"""
    if diff_img is None or diff_img.size == 0:
        return False
    gray = np.mean(diff_img, axis=2)
    std_val = np.std(gray)
    return float(np.mean(np.abs(gray) > std_val * threshold_factor)) > 0.05


class YCBSlideAdapter(BaseAdapter):
    """YCB-Slide (MidasTouch, CoRL 2022) → TLabelData

    Compliance Level: L2（deformation_mag 近似 force_magnitude，force_direction 恒为0.0）
    """

    default_compliance_level: str = "L2"

    @property
    def name(self) -> str:
        return "ycb_slide"

    @property
    def supported_extensions(self) -> list:
        return [".npy"]

    def extract_schema(self, raw_frame_data) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2（14维）

        参数:
            raw_frame_data: dict，包含以下键：
                - diff_img: ndarray 或 None 背景减除后的图像
                - is_contact: bool 是否接触
                - prev_diff_img: ndarray 或 None 上一帧背景减除图像（可选）

        返回:
            TLabelSchemaV2 — L2级别，deformation_mag 近似 force_magnitude，
            force_vector 为 None（无3D力测量）
        """
        # 复用现有模块级 _extract_tlabel_v2 函数获取22维dict
        tlabel_v2 = _extract_tlabel_v2(
            diff_img=raw_frame_data["diff_img"],
            is_contact=raw_frame_data["is_contact"],
        )

        return self._v2_to_schema_v2(tlabel_v2)

    def _v2_to_schema_v2(self, tlabel_v2: Dict[str, Any]) -> TLabelSchemaV2:
        """把 22 维 flat 特征映射到 14 维 Schema V2（L2 自评）。

        ``load()`` 与 ``extract_schema()`` 共用。注意 force_magnitude 是
        ``deformation_magnitude`` 的视觉近似、confidence 是按接触状态写死的
        常数，不是真实力/置信度估计。
        """
        contact = float(tlabel_v2["contact"]) > 0.5
        centroid_x = tlabel_v2.get("centroid_x", 0.5)

        return TLabelSchemaV2(
            contact=contact,
            contact_centroid=[float(centroid_x), 0.5] if contact else None,
            contact_region=None,
            force_magnitude=float(tlabel_v2["deformation_magnitude"]),  # 近似
            force_vector=None,  # L2: 无3D力
            torque_vector=None,
            slip_event=float(tlabel_v2["slip_event"]) > 0.5,
            slip_velocity=None,
            manipulation_phase=None,
            texture_class=None,
            object_deformation=float(tlabel_v2["deformation_magnitude"]),
            temperature=None,
            confidence=0.95 if not contact else 0.8,
            compliance_level=self.default_compliance_level,
        )

    def get_capabilities(self) -> Dict[str, bool]:
        """13 个合法能力键（Schema V2，规范 S-7.4-1）。

        判定口径：以 load() 实际产出的帧内容为准，不按 extract_schema 的
        理论能力。compliance_level 是帧级元字段，不属于能力键（S-7.4-2）。
        YCB-Slide（DIGIT）。force_magnitude 是形变近似；confidence 是按接触
        状态写死的常数（0.95/0.8），不是真实置信度估计。
        """
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": False,
            "force_magnitude": True,  # 由 deformation_magnitude 近似，非标定牛顿值
            "force_vector": False,
            "torque_vector": False,
            "slip_event": True,
            "slip_velocity": False,
            "manipulation_phase": True,
            "texture_class": False,
            "object_deformation": True,  # 由 deformation_magnitude 映射
            "temperature": False,
            "confidence": True,  # heuristic：按接触状态写死 0.95/0.8
        }

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "type": "vision-based_tactile",
            "manufacturer": "GelSight Inc.",
            "model": "DIGIT",
            "resolution": list(DIGIT_RESOLUTION),
            "sampling_rate_hz": DIGIT_FPS,
        }

    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             **kwargs) -> TLabelData:
        """
        加载 YCB-Slide 数据

        参数:
            file_path: 数据目录路径。可以是:
                - 顶层目录 (含 real/ 和/或 sim/ 子目录)
                - 单个物体目录 (如 004_sugar_box/)
                - 单个数据集目录 (如 004_sugar_box/dataset_0/)
            trajectory_id: 数据集编号 (仅 real 模式, 如 0-4)
            split: 'real', 'sim', or 'auto' (默认 'auto')
        """
        root = Path(file_path)
        split = kwargs.get("split", "auto")

        # 检测数据类型
        data_type = self._detect_data_type(root)

        if data_type == "real":
            return self._load_real(root, trajectory_id, split)
        elif data_type == "sim":
            return self._load_sim(root, trajectory_id, split)
        elif data_type == "mixed":
            # 顶层目录，加载 real 数据
            real_dir = root / "real"
            if real_dir.exists():
                return self._load_real(real_dir, trajectory_id, split)
            raise ValueError(f"无法识别的数据目录结构: {root}")
        else:
            raise ValueError(
                f"无法识别 YCB-Slide 数据目录: {root}\n"
                f"期望结构:\n"
                f"  顶层: real/<object>/dataset_X/synced_data.npy\n"
                f"  或:   sim/<object>/XX/tactile_data.pkl"
            )

    def _detect_data_type(self, root: Path) -> str:
        """检测数据类型: real / sim / mixed"""
        if (root / "real").exists() and (root / "sim").exists():
            return "mixed"
        if (root / "real").exists():
            return "real_top"
        if (root / "sim").exists():
            return "sim_top"

        # 检查是否是单个物体目录
        if root.name.startswith(tuple(o[:3] for o in YCB_OBJECTS)):
            if any(root.glob("dataset_*/synced_data.npy")):
                return "real"
            if any(root.glob("*/tactile_data.pkl")):
                return "sim"

        # 检查是否是单个 dataset 目录
        if (root / "synced_data.npy").exists():
            return "real"
        if (root / "tactile_data.pkl").exists():
            return "sim"

        return "unknown"

    def _load_real(self, root: Path, trajectory_id: Optional[int],
                   split: str) -> TLabelData:
        """加载 real 数据"""
        all_frames = []
        sequences_meta = []
        frame_offset = 0

        # 找到所有物体目录
        if root.name.startswith("0"):
            # 直接是物体目录
            obj_dirs = [root]
        else:
            obj_dirs = sorted([
                d for d in root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])

        for obj_dir in obj_dirs:
            object_name = obj_dir.name

            # 找到 dataset_X 目录
            dataset_dirs = sorted([
                d for d in obj_dir.iterdir()
                if d.is_dir() and d.name.startswith("dataset_")
            ])

            if trajectory_id is not None:
                dataset_dirs = [
                    d for d in dataset_dirs
                    if int(d.name.split("_")[1]) == trajectory_id
                ]

            for ds_dir in dataset_dirs:
                synced_path = ds_dir / "synced_data.npy"
                if not synced_path.exists():
                    continue

                synced = np.load(str(synced_path), allow_pickle=True).item()
                timestamps = synced.get("timestamps", [])
                digit_frames = synced.get("digit_frames", [])
                poses = synced.get("poses", {})

                # 找到 DIGIT 传感器位姿
                digit_pose = None
                obj_pose = None
                for key, val in poses.items():
                    if "DIGIT" in key.upper() or "digit" in key.lower():
                        digit_pose = val
                    else:
                        obj_pose = val
                # fallback: 第一个是 sensor，第二个是 object
                if digit_pose is None and len(poses) >= 1:
                    keys = list(poses.keys())
                    digit_pose = poses[keys[0]]
                    if len(poses) >= 2:
                        obj_pose = poses[keys[1]]

                num_frames = len(timestamps)

                # 计算背景 (采样前30帧无接触帧或所有帧)
                digit_image_paths = []
                for rel_path in digit_frames:
                    abs_path = ds_dir / rel_path
                    if abs_path.exists():
                        digit_image_paths.append(str(abs_path))
                    else:
                        digit_image_paths.append("")

                background = _compute_background(
                    [p for p in digit_image_paths if p],
                    sample_size=30
                )

                # 逐帧处理
                prev_diff = None
                for i in range(num_frames):
                    ts = float(timestamps[i]) if i < len(timestamps) else i / DIGIT_FPS

                    # 加载 DIGIT 图像
                    img = None
                    diff_img = None
                    img_path = digit_image_paths[i] if i < len(digit_image_paths) else ""
                    if img_path and HAS_CV2:
                        img = _load_image(img_path)
                        diff_img = _bg_subtract(img, background)

                    # 接触检测
                    is_contact = _detect_contact_from_image(diff_img)

                    # 提取 22 维特征
                    tlabel_v2 = _extract_tlabel_v2(diff_img, is_contact)

                    # 帧间差分
                    if prev_diff is not None and diff_img is not None:
                        prev_gray = np.mean(prev_diff, axis=2)
                        curr_gray = np.mean(diff_img, axis=2)
                        delta = float(np.sqrt(np.mean(
                            (curr_gray - prev_gray) ** 2)))
                        tlabel_v2["delta_force_normal"] = round(delta, 4)
                        tlabel_v2["temporal_deformation_rate"] = round(
                            delta * DIGIT_FPS, 4)

                    # 接触状态变化
                    if prev_diff is not None:
                        prev_contact = _detect_contact_from_image(prev_diff)
                        if prev_contact != is_contact:
                            tlabel_v2["contact_transition"] = 1.0

                    # 位姿
                    sensor_pose = None
                    object_pose = None
                    if digit_pose is not None and i < len(digit_pose):
                        p = digit_pose[i]
                        sensor_pose = {
                            "position": [round(float(p[0]), 6),
                                         round(float(p[1]), 6),
                                         round(float(p[2]), 6)],
                            "orientation": [round(float(p[3]), 6),
                                            round(float(p[4]), 6),
                                            round(float(p[5]), 6),
                                            round(float(p[6]), 6)],
                        }
                    if obj_pose is not None and i < len(obj_pose):
                        p = obj_pose[i]
                        object_pose = {
                            "position": [round(float(p[0]), 6),
                                         round(float(p[1]), 6),
                                         round(float(p[2]), 6)],
                            "orientation": [round(float(p[3]), 6),
                                            round(float(p[4]), 6),
                                            round(float(p[5]), 6),
                                            round(float(p[6]), 6)],
                        }

                    # 操作阶段推断
                    phase = "approach"
                    if is_contact:
                        if tlabel_v2.get("slip_event", 0) > 0.5:
                            phase = "slip"
                        else:
                            phase = "stable_contact"

                    confidence = 0.95 if not is_contact else 0.8

                    sensor_specific = {}
                    if sensor_pose:
                        sensor_specific["sensor_pose"] = sensor_pose
                    if object_pose:
                        sensor_specific["object_pose"] = object_pose
                    if img_path:
                        sensor_specific["digit_image_path"] = img_path

                    frame = TLabelFrame(
                        frame_idx=frame_offset + i,
                        timestamp_s=round(ts, 4),
                        schema_v2=self._v2_to_schema_v2(tlabel_v2),
                        manipulation_phase=phase,
                        confidence=confidence,
                        sensor_specific=sensor_specific,
                    )
                    all_frames.append(frame)
                    prev_diff = diff_img

                ds_id = int(ds_dir.name.split("_")[1])
                sequences_meta.append({
                    "seq_id": f"{object_name}_dataset{ds_id}",
                    "object_name": object_name,
                    "dataset_id": ds_id,
                    "num_frames": num_frames,
                    "duration_s": round(
                        float(timestamps[-1] - timestamps[0]), 4
                    ) if num_frames > 0 else 0.0,
                })

                frame_offset += num_frames

        if not all_frames:
            raise ValueError(
                f"未找到有效的 YCB-Slide real 数据。\n"
                f"请检查目录结构: {root}\n"
                f"期望: <object>/dataset_X/synced_data.npy"
            )

        sensor_info = {
            "type": "vision-based_tactile",
            "model": "DIGIT",
            "manufacturer": "GelSight Inc.",
            "modality": "vision-based_tactile",
            "layout": {
                "type": "single_sensor",
                "resolution": list(DIGIT_RESOLUTION),
                "sampling_rate_hz": DIGIT_FPS,
            }
        }

        episode_info = {
            "source": "CMU-RPL/YCB-Slide",
            "paper": "MidasTouch (CoRL 2022)",
            "url": "https://github.com/rpl-cmu/YCB-Slide",
            "split": "real",
            "sequences": sequences_meta,
            "num_objects": len(set(s["object_name"] for s in sequences_meta)),
            "stats": {
                "total_frames": len(all_frames),
                "contact_frames": sum(
                    1 for f in all_frames
                    if f.contact > 0.5
                ),
            }
        }

        return TLabelData(
            frames=all_frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
            sensor_id="digit_ycb_slide",
        )

    def _load_sim(self, root: Path, trajectory_id: Optional[int],
                  split: str) -> TLabelData:
        """加载 sim 数据"""
        if not HAS_DILL:
            raise ImportError(
                "YCB-Slide sim 数据需要 dill 或 pickle 包。\n"
                "安装: pip install dill"
            )

        all_frames = []
        sequences_meta = []
        frame_offset = 0

        # 找到所有物体目录
        if root.name.startswith("0"):
            obj_dirs = [root]
        else:
            obj_dirs = sorted([
                d for d in root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])

        for obj_dir in obj_dirs:
            object_name = obj_dir.name

            # 检测并处理 Google Drive 下载时多一层同名子目录的情况
            # 例如: 004_sugar_box/004_sugar_box/00/ 而不是 004_sugar_box/00/
            actual_obj_dir = obj_dir
            same_name_subdirs = [
                d for d in obj_dir.iterdir()
                if d.is_dir() and d.name == obj_dir.name
            ]
            if len(same_name_subdirs) == 1:
                actual_obj_dir = same_name_subdirs[0]

            # 找到轨迹目录 (00, 01, ...)
            traj_dirs = sorted([
                d for d in actual_obj_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ])

            if trajectory_id is not None:
                traj_dirs = [
                    d for d in traj_dirs
                    if int(d.name) == trajectory_id
                ]

            for traj_dir in traj_dirs:
                pkl_path = traj_dir / "tactile_data.pkl"
                if not pkl_path.exists():
                    continue

                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)

                gelposes_meas = np.array(data.get("gelposes_meas", []))
                gelposes_gt = np.array(data.get("gelposes", []))
                num_frames = len(gelposes_meas)

                traj_id = int(traj_dir.name)

                # 加载触觉图像
                img_dir = traj_dir / "tactile_images"
                img_files = sorted(img_dir.glob("*.jpg")) if img_dir.exists() else []
                background = _compute_background(
                    [str(f) for f in img_files], sample_size=30
                ) if img_files else None

                for i in range(num_frames):
                    ts = i / DIGIT_FPS

                    # 加载图像
                    diff_img = None
                    if i < len(img_files):
                        img = _load_image(str(img_files[i]))
                        diff_img = _bg_subtract(img, background)

                    is_contact = _detect_contact_from_image(diff_img)
                    tlabel_v2 = _extract_tlabel_v2(diff_img, is_contact)

                    # 位姿
                    sensor_pose_meas = None
                    sensor_pose_gt = None
                    if i < len(gelposes_meas):
                        p = gelposes_meas[i]
                        sensor_pose_meas = {
                            "position": [round(float(p[0]), 6),
                                         round(float(p[1]), 6),
                                         round(float(p[2]), 6)],
                            "orientation": [round(float(p[3]), 6),
                                            round(float(p[4]), 6),
                                            round(float(p[5]), 6),
                                            round(float(p[6]), 6)],
                        }
                    if i < len(gelposes_gt):
                        p = gelposes_gt[i]
                        sensor_pose_gt = {
                            "position": [round(float(p[0]), 6),
                                         round(float(p[1]), 6),
                                         round(float(p[2]), 6)],
                            "orientation": [round(float(p[3]), 6),
                                            round(float(p[4]), 6),
                                            round(float(p[5]), 6),
                                            round(float(p[6]), 6)],
                        }

                    phase = "stable_contact" if is_contact else "approach"
                    confidence = 0.95 if not is_contact else 0.85

                    sensor_specific = {}
                    if sensor_pose_meas:
                        sensor_specific["sensor_pose_noisy"] = sensor_pose_meas
                    if sensor_pose_gt:
                        sensor_specific["sensor_pose_gt"] = sensor_pose_gt

                    frame = TLabelFrame(
                        frame_idx=frame_offset + i,
                        timestamp_s=round(ts, 4),
                        schema_v2=self._v2_to_schema_v2(tlabel_v2),
                        manipulation_phase=phase,
                        confidence=confidence,
                        sensor_specific=sensor_specific,
                    )
                    all_frames.append(frame)

                sequences_meta.append({
                    "seq_id": f"{object_name}_{traj_id:02d}",
                    "object_name": object_name,
                    "trajectory_id": traj_id,
                    "num_frames": num_frames,
                    "duration_s": round(num_frames / DIGIT_FPS, 4),
                })
                frame_offset += num_frames

        if not all_frames:
            raise ValueError(
                f"未找到有效的 YCB-Slide sim 数据。\n"
                f"请检查目录结构: {root}\n"
                f"期望: <object>/XX/tactile_data.pkl"
            )

        sensor_info = {
            "type": "vision-based_tactile",
            "model": "DIGIT",
            "manufacturer": "GelSight Inc.",
            "modality": "vision-based_tactile",
            "layout": {
                "type": "single_sensor",
                "resolution": list(DIGIT_RESOLUTION),
                "sampling_rate_hz": DIGIT_FPS,
            }
        }

        episode_info = {
            "source": "CMU-RPL/YCB-Slide",
            "paper": "MidasTouch (CoRL 2022)",
            "url": "https://github.com/rpl-cmu/YCB-Slide",
            "split": "sim",
            "sequences": sequences_meta,
            "num_objects": len(set(s["object_name"] for s in sequences_meta)),
            "stats": {
                "total_frames": len(all_frames),
                "contact_frames": sum(
                    1 for f in all_frames
                    if f.contact > 0.5
                ),
            }
        }

        return TLabelData(
            frames=all_frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
            sensor_id="digit_ycb_slide_sim",
        )
