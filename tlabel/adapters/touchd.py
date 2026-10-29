"""
ToucHD适配器 — 将AnyTouch2的ToucHD-Force数据转换为TLabelData

数据来源: AnyTouch 2 (ICLR 2026), GeWu-Lab + BAAI
数据格式: JSON元数据 + PNG触觉帧
层级: Tier 1 (Touch-Force Paired Data) — 含3D接触力标签

数据目录结构:
    ToucHD-Force/
    ├── all_data_direction.json    ← 元数据(含image_id, Fx, Fy, Fz, action)
    ├── obj{N}_speed{S}.zip       ← 压缩包(需先解压)
    ├── obj{N}_speed{S}/
    │   ├── digit/
    │   │   ├── image_{id}_l.png
    │   │   ├── image_{id}_r.png
    │   │   └── ...
    │   ├── biotip/
    │   ├── gelsight/
    │   ├── duragel/
    │   └── dm/                   ← 双模态(无JSON映射，暂不支持)
    │   ├── *data.csv             ← 辅助CSV数据
    │   └── *tactile.csv

JSON结构:
    {
        "obj{N}_speed{S}": {
            "sensor_name": [
                [image_id, Fx, Fy, Fz, action_label],
                ...
            ]
        }
    }

传感器: digit, biotip, gelsight, duragel
力标签: 3D接触力 (Fx, Fy, Fz)，按传感器归一化
动作标签: press, slide等
图片后缀: _l.png(左) / _r.png(右)，默认使用_r

引用:
    @inproceedings{fenganytouch2,
        title={AnyTouch 2: General Optical Tactile Representation Learning
               For Dynamic Tactile Perception},
        author={Feng, Ruoxuan and Zhou, Yuxuan and Mei, Siyu and
                Zhou, Dongzhan and Wang, Pengwei and Cui, Shaowei and
                Fang, Bin and Yao, Guocai and Hu, Di},
        booktitle={The Fourteenth International Conference on Learning Representations},
        year={2026}
    }
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
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── 传感器力归一化参数 (来自AnyTouch2源码) ──
MAX_FORCE_ABS_XYZ = {
    "digit": [5.25, 10.61, 14.14],
    "gelsight": [6.84, 9.87, 8.52],
    "duragel": [3.92, 3.64, 7.89],
    "biotip": [2.98, 3.91, 5.68],
}

SUPPORTED_SENSORS = ["digit", "biotip", "gelsight", "duragel"]

# 训练/测试物体划分 (来自AnyTouch2源码)
TRAIN_OBJ_IDS = [6, 41, 52, 53, 59, 69, 70]
TEST_OBJ_IDS = [18, 22, 61]


def _decode_image(image_path: str):
    """读取PNG触觉帧"""
    if not HAS_CV2:
        return None
    try:
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        return None


def _get_background_from_dir(sensor_dir: Path, sample_size: int = 50,
                               hand: str = "r"):
    """从传感器目录采样计算背景"""
    if not sensor_dir.exists():
        return None
    png_files = sorted(sensor_dir.glob(f"image_*_{hand}.png"))
    if not png_files:
        return None
    np.random.seed(42)
    sample_idx = np.random.choice(
        len(png_files), min(sample_size, len(png_files)), replace=False
    )
    sample_imgs = []
    for idx in sample_idx:
        img = _decode_image(str(png_files[idx]))
        if img is not None:
            sample_imgs.append(img.astype(np.float32))
    if not sample_imgs:
        return None
    return np.median(np.stack(sample_imgs, axis=0), axis=0)


def _bg_subtract(img, bg):
    """背景减除"""
    if bg is None:
        return img.astype(np.float32) if img is not None else None
    return img.astype(np.float32) - bg


def _compute_normal_field(diff_img):
    """计算法向场特征"""
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
    """计算剪切场特征"""
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


def _extract_tlabel_v2(diff_img, is_contact, prev_diff_img=None,
                        optical_flow_mag=0.0, optical_flow_dir=0.0,
                        temporal_deform_rate=0.0, contact_transition=0.0,
                        force_xyz=None, force_scale=None):
    """
    从背景减除后的图像提取TLabel v2 22维特征

    ToucHD特有: force_xyz和force_scale直接来自3D力传感器标注，
    可用于力相关维度的ground truth验证
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
        if force_xyz is not None:
            fz = abs(force_xyz[2])
            empty["contact"] = 1.0 if fz > 0.01 else 0.0
            empty["force_magnitude"] = min(fz, 1.0)
            if force_scale is not None:
                empty["force_magnitude"] = min(fz * force_scale[2], 1.0)
        return empty

    gray = np.mean(diff_img, axis=2)

    contact = float(is_contact)
    deformation_mag = float(np.sqrt(np.mean(diff_img**2)))

    # 优先使用力传感器的ground truth
    if force_xyz is not None:
        fz = abs(force_xyz[2])
        force_mag = min(fz, 1.0)
        if force_scale is not None:
            force_mag = min(fz * force_scale[2] / max(force_scale[2], 1e-6), 1.0)
    else:
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

    if prev_diff_img is not None:
        prev_gray = np.mean(prev_diff_img, axis=2)
        delta_fn = float(np.sqrt(np.mean((gray - prev_gray)**2)))
        prev_sf = _compute_shear_field(prev_diff_img)[0]
        delta_fs = abs(sf_mag - prev_sf)
    else:
        delta_fn = 0.0
        delta_fs = 0.0

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
        "delta_force_normal": round(delta_fn, 4),
        "delta_force_shear": round(delta_fs, 4),
        "friction_cone_ratio": round(min(friction_ratio, 10.0), 4),
        "optical_flow_magnitude": round(optical_flow_mag, 4),
        "optical_flow_direction": round(optical_flow_dir, 2),
        "temporal_deformation_rate": round(temporal_deform_rate, 4),
        "contact_transition": round(contact_transition, 4),
    }


def _infer_phases(frames_info):
    """从接触状态推断操作阶段"""
    phases = []
    current = "idle"
    for fi in frames_info:
        ic, is_slip = fi["is_contact"], fi["is_slip"]
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


def _detect_slip_from_force(force_xyz, prev_force_xyz=None, threshold=0.3):
    """
    从3D力数据检测滑移事件

    ToucHD特有优势: 有真实Fx/Fy剪切力，可从力信号检测滑移
    简单启发式: 剪切力占比 > threshold × 法向力 → 滑移
    """
    if force_xyz is None:
        return False
    fz = abs(force_xyz[2])
    fxy = np.sqrt(force_xyz[0]**2 + force_xyz[1]**2)
    if fz < 1e-6:
        return fxy > 0.01
    return (fxy / fz) > threshold


class ToucHDAdapter(BaseAdapter):
    """
    ToucHD-Force → TLabelData

    将AnyTouch2的ToucHD Tier 1力数据转换为TLabel统一标注格式。
    支持4种传感器(digit/biotip/gelsight/duragel)，71个压头。

    用法:
        from tlabel.core.loader import load
        data = load("ToucHD-Force/", format="touchd", sensor="gelsight")
        data.review()

    高级用法:
        # 加载特定物体和传感器
        data = load("ToucHD-Force/", format="touchd",
                     sensor="digit", obj_id=6)

        # 选择左手或右手图像
        data = load("ToucHD-Force/", format="touchd",
                     sensor="digit", hand="l")

        # 加载所有传感器的所有物体
        data = load("ToucHD-Force/", format="touchd", sensor="all")
    """

    def detect_image_shape(self, file_path: Optional[str] = None):
        """检测ToucHD触觉图像形状

        从数据目录中读取第一张PNG图像确定形状。
        不同传感器分辨率可能不同，需从实际数据确认。

        已知典型分辨率:
          - digit: (120, 160, 3)
          - gelsight: (240, 320, 3)
          - biotip/duragel: 各有差异

        Returns:
            (height, width, channels) 或 None
        """
        if file_path is None:
            return None

        root = Path(file_path)
        if not root.exists():
            return None

        # 尝试查找JSON元数据
        json_path = root / "all_data_direction.json"
        if not json_path.exists():
            return None

        try:
            with open(json_path, "r") as f:
                json_data = json.load(f)

            # 遍历找到第一个有图像的传感器目录
            for obj_speed_name, obj_data in json_data.items():
                for sensor_name in SUPPORTED_SENSORS:
                    if sensor_name not in obj_data or not obj_data[sensor_name]:
                        continue
                    sensor_dir = root / obj_speed_name / sensor_name
                    if not sensor_dir.exists():
                        continue
                    # 找第一张PNG
                    pngs = sorted(sensor_dir.glob("image_*_r.png"))
                    if not pngs:
                        pngs = sorted(sensor_dir.glob("image_*_l.png"))
                    if pngs:
                        img = _decode_image(str(pngs[0]))
                        if img is not None:
                            return img.shape
        except Exception:
            pass

        return None

    # v0.17: Compliance Level L3 — ToucHD是唯一拥有完整3D力标签(Fx,Fy,Fz)的数据集
    default_compliance_level: str = "L3"

    @property
    def name(self) -> str:
        return "touchd"

    @property
    def supported_extensions(self):
        return [".json"]

    def get_capabilities(self) -> Dict[str, bool]:
        """13 个合法能力键（Schema V2，规范 S-7.4-1）。

        判定口径：以 load() 实际产出的帧内容为准，不按 extract_schema 的
        理论能力。compliance_level 是帧级元字段，不属于能力键（S-7.4-2）。
        ToucHD（唯一带真实 3D 力 ground truth 的数据集）。force_vector 直接取
        传感器真值；slip_velocity 为条件能力。
        """
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": False,
            "force_magnitude": True,  # norm(force_vector)，来自 3D 力真值
            "force_vector": True,  # 传感器 3D 力 ground truth
            "torque_vector": False,
            "slip_event": True,
            "slip_velocity": True,  # 条件：有 optical flow 时换算
            "manipulation_phase": True,
            "texture_class": False,
            "object_deformation": True,  # 由 deformation_magnitude 映射
            "temperature": False,
            "confidence": True,
        }

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "type": "vision-based_tactile",
            "source": "ToucHD-Force (AnyTouch 2, ICLR 2026)",
            "manufacturers": [
                "Meta FAIR (DIGIT)",
                "GelSight Inc. (GelSight)",
                "Custom (DuraGel)",
                "Custom (BioTip)",
            ],
        }

    def extract_schema(self, raw_frame_data: Union[TLabelFrame, Dict]) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2 (14维结构化)

        ToucHD适配器策略（L3 — 唯一拥有完整3D力标签的数据集）:
          - contact: 从tlabel_v2.contact推断
          - contact_centroid: 从centroid_x估算 [cx, cy]
          - force_magnitude: norm(force_vector)，从3D力标签直接计算
          - force_vector: [Fx, Fy, Fz] 直接从传感器ground truth取
          - compliance_level: L3（始终有完整3D力矢量）

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

        # --- ToucHD特有增强: 直接使用3D力标签 ---

        # 1. force_vector: 直接从sensor_specific中取3D力标注
        #    ToucHD数据中有 force_xyz_normalized 和 force_xyz_raw_N
        force_xyz = None
        if "force_xyz_normalized" in sensor_specific:
            force_xyz = sensor_specific["force_xyz_normalized"]
        elif "force_xyz_raw_N" in sensor_specific:
            force_xyz = sensor_specific["force_xyz_raw_N"]

        if force_xyz is not None and len(force_xyz) == 3:
            # force_xyz: [Fx, Fy, Fz]（Fz已取绝对值，正值=下压）
            fx, fy, fz = float(force_xyz[0]), float(force_xyz[1]), float(force_xyz[2])
            schema.force_vector = [round(fx, 4), round(fy, 4), round(fz, 4)]

            # 2. force_magnitude: 从3D力矢量直接计算（比图像估算更准确）
            schema.force_magnitude = round(math.sqrt(fx**2 + fy**2 + fz**2), 4)
        else:
            # fallback: 使用tlabel_v2中的force_magnitude（从图像估算）
            fm = v2_dict.get("force_magnitude")
            if fm is not None and fm > 0:
                schema.force_magnitude = float(fm)
            schema.force_vector = None

        # 3. contact_centroid: 从centroid_x估算
        centroid_x = v2_dict.get("centroid_x")
        if centroid_x is not None and schema.contact:
            schema.contact_centroid = [float(centroid_x), 0.0]

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

        # 6. compliance_level: 有force_vector时L3，否则降级到L2
        #    正常load()路径中sensor_specific始终包含force_xyz，此时为L3
        #    但如果从dict输入且缺少sensor_specific，则降级
        if schema.force_vector is not None:
            schema.compliance_level = "L3"
        elif schema.force_magnitude is not None:
            schema.compliance_level = "L2"
        else:
            schema.compliance_level = "L1"

        return schema

    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             sensor: str = "gelsight",
             obj_id: Optional[int] = None,
             hand: str = "r",
             direction: Optional[str] = None,
             num_frames: int = 4,
             stride: int = 2,
             max_objects: Optional[int] = None,
             **kwargs) -> TLabelData:
        """
        加载ToucHD-Force数据

        Args:
            file_path: ToucHD-Force数据集根目录路径
            trajectory_id: 未使用(兼容接口)
            sensor: 传感器名 "digit"/"biotip"/"gelsight"/"duragel"/"all"
            obj_id: 指定物体ID(如6, 18等)，None=加载所有
            hand: 选择手侧图像 "l"(左)/"r"(右)，默认"r"
            direction: 指定滑动方向(None=全部)
            num_frames: 每个样本的帧数
            stride: 帧间步长
            max_objects: 最多加载的物体数(调试用)

        Returns:
            TLabelData — 统一标注容器
        """
        root = Path(file_path)
        json_path = root / "all_data_direction.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"ToucHD元数据文件不存在: {json_path}\n"
                f"请先从HuggingFace下载ToucHD-Force数据集:\n"
                f"  huggingface-cli download --repo-type dataset "
                f"BAAI/ToucHD-Force --local-dir {root}"
            )

        with open(json_path, "r") as f:
            json_data = json.load(f)

        # 验证传感器参数
        if sensor not in SUPPORTED_SENSORS and sensor != "all":
            raise ValueError(
                f"不支持的传感器: {sensor}，可选: "
                f"{'/'.join(SUPPORTED_SENSORS)}/all"
            )

        if hand not in ("l", "r"):
            raise ValueError(f"hand参数只接受 'l' 或 'r'，收到: {hand}")

        sensors_to_load = (
            SUPPORTED_SENSORS if sensor == "all" else [sensor]
        )

        all_tlabel_frames = []
        episode_objects = []
        total_force_samples = 0
        action_counts = {}

        for obj_speed_name in json_data.keys():
            obj_speed_data = json_data[obj_speed_name]

            # 解析物体ID: "obj006_speed1" -> 6
            obj_name = obj_speed_name.split("_speed")[0]
            obj_name_id = int(obj_name.replace("obj", ""))

            # 过滤物体
            if obj_id is not None and obj_name_id != obj_id:
                continue

            # 限制物体数
            if max_objects is not None and len(episode_objects) >= max_objects:
                break

            for sensor_name in sensors_to_load:
                if sensor_name not in obj_speed_data:
                    continue

                sensor_data = obj_speed_data[sensor_name]
                if not sensor_data:
                    continue

                # 传感器目录: obj006_speed1/digit/
                sensor_dir = root / obj_speed_name / sensor_name
                if not sensor_dir.exists():
                    # 数据可能未解压，跳过
                    continue

                # 预计算背景（使用对应hand后缀的图像）
                background = _get_background_from_dir(
                    sensor_dir, sample_size=30, hand=hand
                )

                # 获取力归一化参数
                force_scale = MAX_FORCE_ABS_XYZ.get(
                    sensor_name, [10.0, 10.0, 10.0]
                )

                # 传感器步长调整
                this_stride = stride
                if sensor_name == "gelsight":
                    this_stride = max(1, stride // 2)

                # 遍历帧序列
                for i in range(
                    num_frames * this_stride - 1, len(sensor_data)
                ):
                    frame_list = []
                    force_list = []

                    for j in range(num_frames):
                        now_index = (
                            i - (num_frames - 1) * this_stride
                            + j * this_stride
                        )
                        if now_index >= len(sensor_data):
                            break
                        entry = sensor_data[now_index]

                        # JSON格式: [image_id, Fx, Fy, Fz, action_label]
                        image_id = int(entry[0])
                        # Fz在原始数据中负值=下压，取绝对值
                        force_xyz = [
                            float(entry[1]),
                            float(entry[2]),
                            abs(float(entry[3])),
                        ]
                        action = str(entry[4]) if len(entry) > 4 else "unknown"
                        action_counts[action] = action_counts.get(action, 0) + 1

                        # 图片路径: image_{id}_{hand}.png
                        img_path = (
                            sensor_dir / f"image_{image_id}_{hand}.png"
                        )
                        frame_list.append({
                            "image_id": image_id,
                            "img_path": str(img_path),
                            "force_xyz": force_xyz,
                            "action": action,
                        })
                        force_list.append(force_xyz)

                    if len(frame_list) < num_frames:
                        continue

                    # 只标注最后一帧（与AnyTouch2训练一致）
                    last_frame = frame_list[-1]
                    last_force = force_list[-1]
                    prev_force = (
                        force_list[-2] if len(force_list) > 1 else None
                    )

                    # 归一化力
                    norm_force = [
                        last_force[0] / force_scale[0],
                        last_force[1] / force_scale[1],
                        min(max(last_force[2] / force_scale[2], 0.0), 1.0),
                    ]

                    # 判断接触和滑移
                    is_contact = norm_force[2] > 0.01
                    is_slip = _detect_slip_from_force(last_force, prev_force)

                    # 读取图像并计算特征
                    img = _decode_image(last_frame["img_path"])
                    diff_img = (
                        _bg_subtract(img, background)
                        if img is not None else None
                    )

                    # 光流计算(简化版)
                    optical_flow_mag = 0.0
                    optical_flow_dir = 0.0
                    if len(frame_list) >= 2 and HAS_CV2:
                        prev_img_path = frame_list[-2]["img_path"]
                        prev_img = _decode_image(prev_img_path)
                        if prev_img is not None and img is not None:
                            try:
                                prev_gray = cv2.cvtColor(
                                    prev_img, cv2.COLOR_RGB2GRAY
                                )
                                curr_gray = cv2.cvtColor(
                                    img, cv2.COLOR_RGB2GRAY
                                )
                                flow = cv2.calcOpticalFlowFarneback(
                                    prev_gray, curr_gray, None,
                                    0.5, 3, 15, 3, 5, 1.2, 0
                                )
                                mag, ang = cv2.cartToPolar(
                                    flow[..., 0], flow[..., 1]
                                )
                                optical_flow_mag = float(np.mean(mag))
                                optical_flow_dir = float(
                                    np.degrees(np.mean(ang))
                                )
                            except Exception:
                                pass

                    # 力的时间变化
                    temp_deform_rate = 0.0
                    if prev_force is not None:
                        df = np.array(last_force) - np.array(prev_force)
                        temp_deform_rate = float(np.linalg.norm(df))

                    contact_trans = min(
                        1.0, temp_deform_rate / max(force_scale[2], 1e-6)
                    )

                    # 计算TLabel v2特征
                    tlabel_v2 = _extract_tlabel_v2(
                        diff_img, is_contact,
                        force_xyz=norm_force,
                        force_scale=force_scale,
                        optical_flow_mag=optical_flow_mag,
                        optical_flow_dir=optical_flow_dir,
                        temporal_deform_rate=temp_deform_rate,
                        contact_transition=contact_trans,
                    )

                    # 用力数据修正contact和force_magnitude
                    tlabel_v2["contact"] = 1.0 if is_contact else 0.0
                    tlabel_v2["force_magnitude"] = norm_force[2]

                    # 传感器特定数据 — 保留原始3D力标注
                    sensor_specific = {
                        "force_xyz_normalized": [
                            round(f, 4) for f in norm_force
                        ],
                        "force_xyz_raw_N": [
                            round(f, 4) for f in last_force
                        ],
                        "force_scale_N": force_scale,
                        "object_id": obj_name_id,
                        "obj_speed_name": obj_speed_name,
                        "sensor_name": sensor_name,
                        "image_id": last_frame["image_id"],
                        "hand": hand,
                        "action_label": last_frame["action"],
                    }

                    # 置信度
                    confidence = 0.95 if is_contact else 0.9

                    # 阶段推断
                    phase = "idle"
                    if is_contact and is_slip:
                        phase = "slip"
                    elif is_contact:
                        phase = "stable_contact"

                    frame = TLabelFrame(
                        frame_idx=total_force_samples,
                        timestamp_s=round(total_force_samples / 30.0, 4),
                        # 走 extract_schema 而不是裸 from_tlabel_v1：ToucHD 是唯一
                        # 带真实 3D 力 ground truth 的数据集，裸 from_tlabel_v1 会把
                        # force_vector 合成成 Fy=0 的假值并丢掉 L3 增强。
                        schema_v2=self.extract_schema(tlabel_v2),
                        manipulation_phase=phase,
                        confidence=confidence,
                        sensor_specific=sensor_specific,
                    )
                    all_tlabel_frames.append(frame)
                    total_force_samples += 1

            episode_objects.append(obj_name_id)

        if not all_tlabel_frames:
            raise ValueError(
                f"未找到任何有效数据。请检查:\n"
                f"  1. 数据目录: {root}\n"
                f"  2. 传感器: {sensor} "
                f"(可用: {'/'.join(SUPPORTED_SENSORS)})\n"
                f"  3. zip文件是否已解压 (需unzip *.zip)"
            )

        # 统一推断阶段
        frames_info = []
        for f in all_tlabel_frames:
            frames_info.append({
                "is_contact": f.contact > 0.5,
                "is_slip": f.slip_event > 0.5,
            })
        phases = _infer_phases(frames_info)
        for f, phase in zip(all_tlabel_frames, phases):
            f.manipulation_phase = phase

        # 统计
        contact_count = sum(
            1 for f in all_tlabel_frames
            if f.contact > 0.5
        )
        slip_count = sum(
            1 for f in all_tlabel_frames
            if f.slip_event > 0.5
        )

        sensor_info = {
            "type": "vision-based_tactile",
            "source": "ToucHD-Force",
            "paper": "AnyTouch 2 (ICLR 2026)",
            "sensors_loaded": sensors_to_load,
            "tier": "Tier 1 — Physical Dynamics (Touch-Force Paired)",
            "layout": {
                "type": "multi_sensor",
                "sensors": sensors_to_load,
                "num_indenters": 71,
                "hand": hand,
                "image_suffix": f"image_{{id}}_{hand}.png",
            }
        }

        episode_info = {
            "source": "BAAI/ToucHD-Force",
            "url": "https://huggingface.co/datasets/BAAI/ToucHD-Force",
            "objects_loaded": sorted(set(episode_objects)),
            "num_objects": len(set(episode_objects)),
            "train_obj_ids": TRAIN_OBJ_IDS,
            "test_obj_ids": TEST_OBJ_IDS,
            "action_distribution": action_counts,
            "stats": {
                "total_samples": total_force_samples,
                "contact_frames": contact_count,
                "contact_ratio": round(
                    contact_count / max(total_force_samples, 1), 4
                ),
                "slip_frames": slip_count,
                "slip_ratio": round(
                    slip_count / max(total_force_samples, 1), 4
                ),
            },
            "pyramid_reference": {
                "framework": "Tactile Dynamic Pyramid",
                "tier": 1,
                "description": "3D contact force (Fx, Fy, Fz) paired with tactile frames",
            }
        }

        calibration_params = {
            "force_normalization": MAX_FORCE_ABS_XYZ,
            "note": "Fz: abs(raw_Fz) / scale clipped [0,1]; "
                    "Fx/Fy: raw / scale; "
                    "raw Fz < 0 means pressing down",
        }

        return TLabelData(
            frames=all_tlabel_frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
            sensor_id=sensor,
            calibration_params=calibration_params,
        )
