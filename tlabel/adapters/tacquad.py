"""
TacQuad adapter - AnyTouch (ICLR 2025) multi-sensor paired tactile dataset to TLabelData

Source: GeWu-Lab AnyTouch (ICLR 2025)
Download: HuggingFace xxuan01/TacQuad or hyper.ai
Dataset: 72,606 contact frames, 4 sensors (GelSight Mini, DIGIT, DuraGel, Tac3D)

Directory structure:
    tactile_datasets/tacquad/
    +-- contact_indoor.csv        # fine-grained subset metadata
    +-- contact_outdoor.csv       # coarse-grained subset metadata
    +-- data_indoor/{item}/       # fine-grained data
    |   +-- gelsight/0.png,1.png,...
    |   +-- digit/0.png,1.png,...
    |   +-- duragel/0.png,1.png,...
    |   +-- img_gelsight/...      # paired vision images
    |   +-- img_digit/...
    |   +-- img_duragel/...
    |   +-- [tac3d/0.npy,...]     # [optional] Tac3D force field
    +-- data_outdoor/{item}/      # coarse-grained data (same structure)

CSV format: 7 fields - item_name, gelsight_start, gelsight_end,
            digit_start, digit_end, duragel_start, duragel_end (inclusive)

Sensor codes: 0=GelSight, 1=DIGIT, 2=GelSlim, 3=GelSight Mini, 4=DuraGel
TacQuad uses: 1(DIGIT), 3(GelSight Mini), 4(DuraGel)
"""

import csv
import logging
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

logger = logging.getLogger(__name__)

# ======================== Sensor Config ========================

SENSOR_CONFIG = {
    'gelsight': {
        'dir_name': 'gelsight',
        'vision_dir': 'img_gelsight',
        'code': 3,
        'model': 'GelSight Mini',
        'manufacturer': 'GelSight Inc.',
    },
    'digit': {
        'dir_name': 'digit',
        'vision_dir': 'img_digit',
        'code': 1,
        'model': 'DIGIT',
        'manufacturer': 'Meta FAIR',
    },
    'duragel': {
        'dir_name': 'duragel',
        'vision_dir': 'img_duragel',
        'code': 4,
        'model': 'DuraGel',
        'manufacturer': 'GeWu-Lab (Custom)',
    },
}

CSV_FIELD_MAP = {
    'gelsight': (1, 2),
    'digit': (3, 4),
    'duragel': (5, 6),
}


def _read_image(image_path):
    """Read PNG image and convert to RGB numpy array"""
    if HAS_CV2:
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
    else:
        try:
            from PIL import Image
            img = Image.open(str(image_path)).convert('RGB')
            return np.array(img)
        except Exception:
            pass
    return None


def _bg_subtract(img, bg):
    """Background subtraction"""
    if img is None:
        return None
    if bg is None:
        return img.astype(np.float32)
    return img.astype(np.float32) - bg.astype(np.float32)


def _compute_background(sensor_dir, start, end, sample_size=20):
    """Compute background from first frames (median)"""
    if not sensor_dir.exists():
        return None
    available = [i for i in range(start, min(start + 10, end + 1))
                 if (sensor_dir / f'{i}.png').exists()]
    if not available:
        return None
    np.random.seed(42)
    sample_idx = np.random.choice(
        len(available), min(sample_size, len(available)), replace=False
    )
    sample_imgs = []
    for idx in sample_idx:
        img = _read_image(str(sensor_dir / f'{available[idx]}.png'))
        if img is not None:
            sample_imgs.append(img.astype(np.float32))
    if not sample_imgs:
        return None
    return np.median(np.stack(sample_imgs, axis=0), axis=0)


def _compute_normal_field(diff_img):
    """Compute normal field features"""
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
    """Compute shear field features"""
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
                       optical_flow_mag=0.0, optical_flow_dir_val=0.0,
                       temporal_deform_rate=0.0, contact_transition=0.0,
                       tac3d_force=None):
    """Extract 22-dim TLabel v2 features from background-subtracted image"""
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

    # Use Tac3D force if available
    if tac3d_force is not None:
        fz = float(np.abs(tac3d_force[:, :, 2]).mean())
        empty["contact"] = 1.0 if fz > 0.01 else 0.0
        empty["force_magnitude"] = min(fz / 10.0, 1.0)
        if diff_img is None or diff_img.size == 0:
            return empty

    if diff_img is None or diff_img.size == 0:
        return empty

    gray = np.mean(diff_img, axis=2)
    contact = 1.0 if is_contact else 0.0
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
    if col_sums.sum() > 0:
        centroid_x = float(np.average(
            np.arange(gray.shape[1]), weights=col_sums
        )) / gray.shape[1]
    else:
        centroid_x = 0.5

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
        "optical_flow_direction": round(optical_flow_dir_val, 2),
        "temporal_deformation_rate": round(temporal_deform_rate, 4),
        "contact_transition": round(contact_transition, 4),
    }


def _infer_phases(frames_info):
    """Infer manipulation phases from contact state"""
    phases = []
    current = "idle"
    for fi in frames_info:
        ic = fi.get("is_contact", False)
        is_slip = fi.get("is_slip", False)
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


def _detect_slip_from_diff(prev_img, curr_img, threshold=5.0):
    """Detect slip event from image difference"""
    if prev_img is None or curr_img is None:
        return False
    diff = np.abs(curr_img.astype(float) - prev_img.astype(float))
    mean_diff = float(np.mean(diff))
    return mean_diff > threshold


class TacQuadAdapter(BaseAdapter):
    """TacQuad (AnyTouch ICLR 2025) -> TLabelData adapter

    Supports multi-sensor paired tactile dataset:
    - GelSight Mini (code 3)
    - DIGIT (code 1)
    - DuraGel (code 4)
    - Tac3D force field (optional, loaded when tac3d/ directory detected)

    Compliance Level: L1（主要只有图像，部分子集有力信息但不确定）

    Usage:
        data = tlabel.load("tactile_datasets/tacquad/", format='tacquad')
        data = tlabel.load("tactile_datasets/tacquad/", sensor='digit')
        data = tlabel.load("tactile_datasets/tacquad/", subset='indoor')
    """

    default_compliance_level: str = "L1"

    @property
    def name(self):
        return "tacquad"

    @property
    def supported_extensions(self):
        return [".csv"]

    def extract_schema(self, raw_frame_data) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2（14维）

        参数:
            raw_frame_data: dict，包含以下键：
                - diff_img: ndarray 或 None 背景减除后的图像
                - is_contact: bool 是否接触
                - prev_diff_img: ndarray 或 None 上一帧背景减除图像（可选）
                - optical_flow_mag: float（默认0.0）
                - optical_flow_dir_val: float（默认0.0）
                - temporal_deform_rate: float（默认0.0）
                - contact_transition: float（默认0.0）
                - tac3d_force: ndarray 或 None Tac3D力场数据（可选）

        返回:
            TLabelSchemaV2 — L1级别，主要填 contact, contact_centroid,
            slip_event, confidence；force_magnitude/force_vector 填 None
        """
        # 复用现有模块级 _extract_tlabel_v2 函数获取22维dict
        tlabel_v2 = _extract_tlabel_v2(
            diff_img=raw_frame_data["diff_img"],
            is_contact=raw_frame_data["is_contact"],
            prev_diff_img=raw_frame_data.get("prev_diff_img"),
            optical_flow_mag=raw_frame_data.get("optical_flow_mag", 0.0),
            optical_flow_dir_val=raw_frame_data.get("optical_flow_dir_val", 0.0),
            temporal_deform_rate=raw_frame_data.get("temporal_deform_rate", 0.0),
            contact_transition=raw_frame_data.get("contact_transition", 0.0),
            tac3d_force=raw_frame_data.get("tac3d_force"),
        )

        return self._v2_to_schema_v2(tlabel_v2)

    def _v2_to_schema_v2(self, tlabel_v2: Dict[str, Any]) -> TLabelSchemaV2:
        """把 22 维 flat 特征映射到 14 维 Schema V2（L1 自评）。

        ``load()`` 与 ``extract_schema()`` 共用：TacQuad 无可靠力测量，
        force_magnitude / object_deformation 一律 None，避免
        ``from_tlabel_v1`` 把视觉形变代理当成力写进帧里。
        """
        contact = float(tlabel_v2["contact"]) > 0.5
        centroid_x = tlabel_v2.get("centroid_x", 0.5)
        is_slip = float(tlabel_v2["slip_event"]) > 0.5

        return TLabelSchemaV2(
            contact=contact,
            contact_centroid=[float(centroid_x), 0.5] if contact else None,
            contact_region=None,
            force_magnitude=None,  # L1: 无可靠力测量
            force_vector=None,     # L1: 无3D力
            torque_vector=None,
            slip_event=is_slip,
            slip_velocity=None,
            manipulation_phase=None,
            texture_class=None,
            object_deformation=None,
            temperature=None,
            confidence=self._compute_confidence(contact, is_slip),
            compliance_level=self.default_compliance_level,
        )

    def get_capabilities(self) -> Dict[str, bool]:
        """13 个合法能力键（Schema V2，规范 S-7.4-1）。

        判定口径：以 load() 实际产出的帧内容为准，不按 extract_schema 的
        理论能力。compliance_level 是帧级元字段，不属于能力键（S-7.4-2）。
        TacQuad 视觉触觉（作者 L1 自评）。无可靠力测量，force_magnitude 与
        object_deformation 在帧里一律 null，不得声明。
        """
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": False,
            "force_magnitude": False,  # L1：无可靠力测量，帧内恒 null
            "force_vector": False,
            "torque_vector": False,
            "slip_event": True,  # 由差分图梯度检测
            "slip_velocity": False,
            "manipulation_phase": True,
            "texture_class": False,
            "object_deformation": False,  # L1：无可靠形变测量，帧内恒 null
            "temperature": False,
            "confidence": True,
        }

    def get_sensor_info(self):
        return {
            "type": "vision-based_tactile",
            "source": "TacQuad (AnyTouch ICLR 2025)",
            "manufacturer": "GeWu-Lab, Renmin University of China",
            "modality": "multi_sensor_paired",
            "sensors": {
                "gelsight_mini": {"code": 3, "resolution": "vision-based RGB"},
                "digit": {"code": 1, "resolution": "vision-based RGB"},
                "duragel": {"code": 4, "resolution": "vision-based RGB"},
                "tac3d": {"code": -1, "resolution": "20x20x3 force field"},
            },
            "layout": {
                "type": "multi_sensor",
                "num_sensors": 4,
                "paired_with_vision": True,
                "paired_with_text": True,
            }
        }

    def load(self, file_path, trajectory_id=None, sensor="gelsight",
             subset="both", max_items=None, skip_transition=0, **kwargs):
        """Load TacQuad dataset and convert to TLabelData

        Args:
            file_path: Dataset root directory path
            trajectory_id: Specific item index (optional, None=all)
            sensor: Sensor name ('gelsight'/'digit'/'duragel'/'all')
            subset: Subset ('indoor'/'outdoor'/'both')
            max_items: Max items to load (None=all)
            skip_transition: Skip first N frames per item (cross_dataset uses 3)
            **kwargs: Reserved for extensions

        Returns:
            TLabelData -- unified annotation container
        """
        root = self._resolve_root(file_path)
        sensors_to_load = self._resolve_sensors(sensor)

        csv_rows = self._load_csv_metadata(root, subset)
        if not csv_rows:
            raise ValueError(
                f"No CSV metadata found. Check directory:\n"
                f"  {root}/contact_indoor.csv or {root}/contact_outdoor.csv"
            )

        if max_items is not None:
            csv_rows = csv_rows[:max_items]

        if trajectory_id is not None:
            if trajectory_id < len(csv_rows):
                csv_rows = [csv_rows[trajectory_id]]
            else:
                raise ValueError(
                    f"trajectory_id={trajectory_id} out of range, "
                    f"total {len(csv_rows)} items"
                )

        all_frames = []
        stats = {"total_frames": 0, "items_loaded": 0, "tac3d_loaded": 0}

        for row_idx, row in enumerate(csv_rows):
            item_name, item_frames, tac3d_count = self._load_item(
                root, row, row_idx, sensors_to_load, subset, skip_transition
            )
            all_frames.extend(item_frames)
            stats["total_frames"] += len(item_frames)
            stats["items_loaded"] += 1
            stats["tac3d_loaded"] += tac3d_count

        if not all_frames:
            raise ValueError(
                f"No valid data frames found. Check:\n"
                f"  1. Data directory: {root}\n"
                f"  2. Sensor: {sensor}\n"
                f"  3. PNG files exist"
            )

        # Infer manipulation phases
        frames_info = [
            {"is_contact": f.contact > 0.5,
             "is_slip": f.slip_event > 0.5}
            for f in all_frames
        ]
        phases = _infer_phases(frames_info)
        for f, phase in zip(all_frames, phases):
            f.manipulation_phase = phase

        sensor_info = self._build_sensor_info(sensors_to_load, subset)
        episode_info = self._build_episode_info(
            root, subset, stats, sensors_to_load
        )

        return TLabelData(
            frames=all_frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
            sensor_id=sensor,
        )

    # ======================== Internal Methods ========================

    def _resolve_root(self, file_path):
        """Resolve dataset root directory"""
        p = Path(file_path)

        if (p / "contact_indoor.csv").exists() or (p / "contact_outdoor.csv").exists():
            return p
        if (p / "tacquad" / "contact_indoor.csv").exists():
            return p / "tacquad"
        if (p / "data_indoor").exists() or (p / "data_outdoor").exists():
            return p
        if (p / "gelsight").exists() and (p / "digit").exists():
            return p.parent.parent

        raise FileNotFoundError(
            f"Cannot locate TacQuad dataset directory: {file_path}\n"
            f"Point to directory containing contact_indoor.csv, "
            f"or data_indoor/ and data_outdoor/"
        )

    def _resolve_sensors(self, sensor):
        """Resolve sensor list"""
        if sensor == "all":
            return list(SENSOR_CONFIG.keys())
        if sensor not in SENSOR_CONFIG:
            raise ValueError(
                f"Unsupported sensor: {sensor}, "
                f"Options: {'/'.join(SENSOR_CONFIG.keys())}/all"
            )
        return [sensor]

    def _load_csv_metadata(self, root, subset):
        """Load CSV metadata, returns List of (subset_name, row) tuples"""
        csv_rows = []
        subsets_to_load = []
        if subset in ("indoor", "both"):
            subsets_to_load.append(("indoor", root / "contact_indoor.csv"))
        if subset in ("outdoor", "both"):
            subsets_to_load.append(("outdoor", root / "contact_outdoor.csv"))

        for subset_name, csv_path in subsets_to_load:
            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}")
                continue
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 7:
                        logger.warning(
                            f"CSV row has fewer than 7 fields: {row}"
                        )
                        continue
                    csv_rows.append((subset_name, row))
        return csv_rows

    def _load_item(self, root, row_tuple, row_idx, sensors,
                   subset, skip_transition):
        """Load all frames for a single item

        Returns:
            (item_name, frames_list, tac3d_count)
        """
        subset_name, row = row_tuple
        item_name = row[0]

        if subset_name == "indoor":
            data_dir = root / "data_indoor" / item_name
        else:
            data_dir = root / "data_outdoor" / item_name

        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return (item_name, [], 0)

        frames = []
        tac3d_count = 0
        tac3d_dir = data_dir / "tac3d"
        has_tac3d = tac3d_dir.exists()

        for sensor_name in sensors:
            cfg = SENSOR_CONFIG[sensor_name]
            start_idx, end_idx = CSV_FIELD_MAP[sensor_name]

            try:
                start_frame = int(row[start_idx])
                end_frame = int(row[end_idx])
            except (ValueError, IndexError):
                logger.warning(
                    f"CSV frame range parse failed: "
                    f"item={item_name}, sensor={sensor_name}"
                )
                continue

            sensor_dir = data_dir / cfg['dir_name']
            vision_dir = data_dir / cfg['vision_dir']

            if not sensor_dir.exists():
                logger.debug(f"Sensor directory not found: {sensor_dir}")
                continue

            background = _compute_background(
                sensor_dir, start_frame, end_frame, sample_size=10
            )

            vision_frame_count = 0
            if vision_dir.exists():
                vision_frame_count = len(list(vision_dir.glob("*.png")))

            prev_diff_img = None
            prev_img = None

            for t in range(start_frame + skip_transition, end_frame + 1):
                frame_path = sensor_dir / f'{t}.png'
                if not frame_path.exists():
                    continue

                img = _read_image(str(frame_path))
                diff_img = _bg_subtract(img, background)

                # Read Tac3D force (optional)
                tac3d_force = None
                if has_tac3d:
                    tac3d_path = tac3d_dir / f'{t}.npy'
                    if tac3d_path.exists():
                        try:
                            tac3d_force = np.load(str(tac3d_path))
                            tac3d_count += 1
                        except Exception as e:
                            logger.debug(
                                f"Tac3D load failed: {tac3d_path}, {e}"
                            )

                is_contact = self._is_contact_frame(diff_img, tac3d_force)

                optical_flow_mag = 0.0
                optical_flow_dir_v = 0.0
                temporal_deform_rate = 0.0
                contact_trans = 0.0

                if HAS_CV2 and prev_img is not None and img is not None:
                    try:
                        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
                        curr_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray, curr_gray, None,
                            0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        mag, ang = cv2.cartToPolar(
                            flow[..., 0], flow[..., 1]
                        )
                        optical_flow_mag = float(np.mean(mag))
                        optical_flow_dir_v = float(
                            np.degrees(np.mean(ang))
                        )
                    except Exception:
                        pass

                if prev_diff_img is not None and diff_img is not None:
                    prev_d = float(np.sqrt(np.mean(prev_diff_img**2)))
                    curr_d = float(np.sqrt(np.mean(diff_img**2)))
                    temporal_deform_rate = abs(curr_d - prev_d)

                prev_has = (prev_diff_img is not None and
                            float(np.sqrt(
                                np.mean(prev_diff_img**2))) > 0.01)
                curr_has = (diff_img is not None and
                            float(np.sqrt(
                                np.mean(diff_img**2))) > 0.01)
                if not prev_has and curr_has:
                    contact_trans = 1.0
                elif prev_has and not curr_has:
                    contact_trans = -1.0
                elif prev_has and curr_has:
                    contact_trans = 0.5

                is_slip = _detect_slip_from_diff(prev_img, img)

                tlabel_v2 = _extract_tlabel_v2(
                    diff_img, is_contact,
                    prev_diff_img=prev_diff_img,
                    optical_flow_mag=optical_flow_mag,
                    optical_flow_dir_val=optical_flow_dir_v,
                    temporal_deform_rate=temporal_deform_rate,
                    contact_transition=contact_trans,
                    tac3d_force=tac3d_force,
                )

                sensor_specific = {
                    "item_name": item_name,
                    "subset": subset_name,
                    "sensor_name": sensor_name,
                    "sensor_code": cfg['code'],
                    "frame_range": [start_frame, end_frame],
                    "has_vision_pair": vision_frame_count > 0,
                    "has_tac3d": tac3d_force is not None,
                }
                if tac3d_force is not None:
                    sensor_specific["tac3d_shape"] = list(
                        tac3d_force.shape
                    )

                confidence = self._compute_confidence(
                    is_contact, is_slip
                )

                frame = TLabelFrame(
                    frame_idx=len(frames),
                    timestamp_s=round((t - start_frame) / 30.0, 4),
                    schema_v2=self._v2_to_schema_v2(tlabel_v2),
                    manipulation_phase="idle",
                    confidence=confidence,
                    sensor_specific=sensor_specific,
                )
                frames.append(frame)

                prev_diff_img = diff_img
                prev_img = img

        return (item_name, frames, tac3d_count)

    def _is_contact_frame(self, diff_img, tac3d_force=None):
        """Determine if frame is in contact"""
        if tac3d_force is not None:
            fz = float(np.abs(tac3d_force[:, :, 2]).mean())
            return fz > 0.01
        if diff_img is None or diff_img.size == 0:
            return False
        magnitude = float(np.sqrt(np.mean(diff_img**2)))
        return magnitude > 0.5

    def _compute_confidence(self, is_contact, is_slip):
        """Compute annotation confidence"""
        if not is_contact and not is_slip:
            return 0.95
        if is_contact and not is_slip:
            return 0.85
        if is_contact and is_slip:
            return 0.6
        return 0.75

    def _build_sensor_info(self, sensors, subset):
        """Build sensor_info dict"""
        sensor_details = {}
        for s in sensors:
            cfg = SENSOR_CONFIG[s]
            sensor_details[s] = {
                "code": cfg['code'],
                "model": cfg['model'],
                "manufacturer": cfg['manufacturer'],
            }
        return {
            "type": "vision-based_tactile",
            "source": "TacQuad (AnyTouch ICLR 2025)",
            "manufacturer": "GeWu-Lab, Renmin University of China",
            "modality": "multi_sensor_paired",
            "subset": subset,
            "sensors_loaded": sensors,
            "sensor_details": sensor_details,
            "layout": {
                "type": "multi_sensor",
                "num_sensors": len(sensors),
                "paired_with_vision": True,
                "paired_with_text": True,
                "tac3d_optional": True,
            }
        }

    def _build_episode_info(self, root, subset, stats, sensors):
        """Build episode_info dict"""
        return {
            "source": "GeWu-Lab/TacQuad",
            "url": "https://huggingface.co/datasets/xxuan01/TacQuad",
            "paper": "AnyTouch (ICLR 2025)",
            "subset": subset,
            "sensors_loaded": sensors,
            "stats": {
                "total_frames": stats["total_frames"],
                "items_loaded": stats["items_loaded"],
                "tac3d_frames": stats["tac3d_loaded"],
            },
            "dataset_info": {
                "indoor_items": 25,
                "indoor_touches": 30,
                "indoor_frames": 17524,
                "outdoor_items": 99,
                "outdoor_touches": 151,
                "outdoor_frames": 55082,
                "total_frames": 72606,
                "total_size_gb": 63.44,
            },
            "pyramid_reference": {
                "framework": "Tactile Dynamic Pyramid",
                "tier": 2,
                "description": "Multi-sensor paired tactile data "
                               "with vision and text embeddings",
            }
        }
