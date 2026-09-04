"""
UniVTAC 适配器 — 将 UniVTAC HDF5 数据转换为 TLabelData

UniVTAC (https://github.com/univtac/UniVTAC) 跨数据集触觉基准：
  - 双 GelSight Mini (左右各一), 240x320 depth + 1200 marker
  - 9个操作任务, 800 episodes
  - HDF5 格式: tactile/{side}/{depth,marker,pose,rgb,rgb_marker}

依赖: pip install tlabel[univtac]  (需要 h5py)
"""

import json
import logging
import hashlib
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

import numpy as np

from tlabel.adapters.base import BaseAdapter
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

logger = logging.getLogger(__name__)

# ======================== 传感器配置 ========================

SENSOR_CONFIG = {
    'left_gsmini': {
        'name': 'GelSight Mini Left',
        'type': 'vision_based',
        'description': 'GelSight Mini left finger tactile sensor, resolution 240x320, 30Hz',
        'resolution': [240, 320],
        'frame_rate': 30.0,
        'marker_count': 1200,
    },
    'right_gsmini': {
        'name': 'GelSight Mini Right',
        'type': 'vision_based',
        'description': 'GelSight Mini right finger tactile sensor, resolution 240x320, 30Hz',
        'resolution': [240, 320],
        'frame_rate': 30.0,
        'marker_count': 1200,
    },
    # v0.23: new-format UniVTAC recordings use left_tactile/right_tactile keys
    'left_tactile': {
        'name': 'GelSight Mini Left',
        'type': 'vision_based',
        'description': 'GelSight Mini left finger tactile sensor (new-format key), resolution 240x320, 30Hz',
        'resolution': [240, 320],
        'frame_rate': 30.0,
        'marker_count': 63,
    },
    'right_tactile': {
        'name': 'GelSight Mini Right',
        'type': 'vision_based',
        'description': 'GelSight Mini right finger tactile sensor (new-format key), resolution 240x320, 30Hz',
        'resolution': [240, 320],
        'frame_rate': 30.0,
        'marker_count': 63,
    }
}

# ======================== 标签映射 ========================

TAG_TO_PHASE = {
    b'approach': 'approach',
    b'grasp':    'grasp',
    b'lift':     'lift',
    b'hold':     'hold',
    b'place':    'place',
    b'release':  'release',
    b'insert':   'insert',
    b'idle':     'idle',
    b'contact':  'initial_contact',
    b'delay':    'hold',
    b'move':     'grasp',
}


class UniVTACAdapter(BaseAdapter):
    """UniVTAC HDF5 → TLabelData 适配器

    支持 UniVTAC 跨数据集触觉基准的 HDF5 格式。
    默认加载左侧 GelSight Mini，可通过 sensor_id 参数选择右侧。
    """

    # v0.17: Compliance Level L2-L3 — UniVTAC有depth估算force_magnitude(L2)，
    # 有marker位移可推算shear方向，在shear信息足够时可升级L3
    default_compliance_level: str = "L2"

    @property
    def name(self) -> str:
        return "univtac"

    @property
    def supported_extensions(self) -> list:
        return [".hdf5", ".h5"]

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "contact": True, "deformation_magnitude": True,
            "force_magnitude": True, "force_peak": True,
            "force_direction": True, "slip_entropy": True,
            "slip_event": True, "texture_energy": True,
            "edge_density": True, "contact_area": True,
            "centroid_x": True,
            "normal_field_magnitude": True, "normal_field_variance": True,
            "shear_field_magnitude": True, "shear_field_direction": True,
            "delta_force_normal": True, "delta_force_shear": True,
            "friction_cone_ratio": True,
            "optical_flow_magnitude": True,
            "optical_flow_direction": True,
            "temporal_deformation_rate": True,
            "contact_transition": True,
        }

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "type": "vision-based_tactile",
            "manufacturer": "GelSight Inc.",
            "modality": "dual_gsmini",
            "layout": {
                "type": "bimanual",
                "sensors": ["left_gsmini", "right_gsmini"],
                "resolution": "240x320",
                "marker_count": 1200,
                "sampling_rate_hz": 30.0,
            }
        }

    def extract_schema(self, raw_frame_data: Union[TLabelFrame, Dict]) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2 (14维结构化)

        UniVTAC适配器策略:
          - contact: 从tlabel_v2.contact推断
          - contact_centroid: 从centroid_x和centroid_y组合 [cx, cy]
          - force_magnitude: 从depth delta估算（_compute_force_direction已计算）
          - force_vector: 如果有shear+force信息可合成 [Fx, Fy, Fz]，否则None
          - object_deformation: 从deformation_magnitude提取
          - compliance_level: 默认L2，shear信息足够时升级L3

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

        # --- UniVTAC特有增强 ---

        # 1. contact_centroid: UniVTAC有centroid_x和centroid_y（通过_compute_centroid计算）
        centroid_x = v2_dict.get("centroid_x")
        centroid_y = sensor_specific.get("centroid_y")
        if centroid_x is not None and schema.contact:
            cx = float(centroid_x)
            cy = float(centroid_y) if centroid_y is not None else 0.0
            schema.contact_centroid = [cx, cy]

        # 2. force_magnitude: 从depth delta估算（已在_compute_force_direction中计算）
        fm = v2_dict.get("force_magnitude")
        if fm is not None and fm > 0:
            schema.force_magnitude = float(fm)

        # 3. force_vector: 从force_magnitude + shear_field_direction合成
        #    UniVTAC有marker位移可以计算shear方向
        shear_mag = v2_dict.get("shear_field_magnitude", 0.0)
        shear_dir_deg = v2_dict.get("shear_field_direction", 0.0)

        if schema.force_magnitude is not None and shear_mag > 1e-6:
            # 从法向力 + 剪切力合成3D力矢量
            shear_dir_rad = math.radians(shear_dir_deg)
            # 剪切力从shear_field_magnitude估算
            shear_force = float(shear_mag)
            fx = shear_force * math.cos(shear_dir_rad)
            fy = shear_force * math.sin(shear_dir_rad)
            fz = float(schema.force_magnitude)
            schema.force_vector = [round(fx, 4), round(fy, 4), round(fz, 4)]
            schema.compliance_level = "L3"
        else:
            schema.force_vector = None
            schema.compliance_level = self.default_compliance_level

        # 4. object_deformation: 从deformation_magnitude提取
        deform = v2_dict.get("deformation_magnitude")
        if deform is not None and deform > 0:
            schema.object_deformation = float(deform)

        # 5. slip_velocity: 从marker optical_flow提取
        if schema.slip_event:
            of_mag = v2_dict.get("optical_flow_magnitude", 0.0)
            of_dir = v2_dict.get("optical_flow_direction", 0.0)
            if of_mag > 1e-6:
                of_rad = math.radians(of_dir)
                schema.slip_velocity = [round(of_mag * math.cos(of_rad), 4),
                                        round(of_mag * math.sin(of_rad), 4)]

        return schema

    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             sensor_id: str = "left_gsmini",
             **kwargs) -> TLabelData:
        """加载 UniVTAC HDF5 文件并转换为 TLabelData

        Args:
            file_path: HDF5 文件路径
            trajectory_id: 未使用（UniVTAC 每个文件即一个 episode）
            sensor_id: 选择传感器，"left_gsmini" 或 "right_gsmini"
            **kwargs: 预留扩展

        Returns:
            TLabelData — 统一标注容器
        """
        if not HAS_H5PY:
            raise ImportError(
                "UniVTAC 适配器需要 h5py: pip install tlabel[univtac]"
            )

        h5_path = Path(file_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 运行时状态
        depth_prev = {}
        marker_prev = {}
        contact_state_prev = {}

        frames = []

        with h5py.File(h5_path, 'r') as f:
            T = self._get_num_frames(f)

            # v0.23: enumerate sensors via the tactile subgroup directly.
            # (Membership checks against the root group with a full path like
            # 'tactile/<key>' are unreliable across h5py/HDF5 versions and the
            # check must run while the file is still open.)
            tactile_group = f['tactile'] if 'tactile' in f else None
            sensor_ids = [k for k in SENSOR_CONFIG
                          if tactile_group is not None and k in tactile_group]

            # 验证传感器存在
            if tactile_group is None or sensor_id not in tactile_group:
                raise ValueError(
                    f"传感器 '{sensor_id}' 不存在，可用: {sensor_ids}"
                )

            # v0.23: infer marker_count from the actual marker dataset
            # shape (T, 2, marker_size, 2); fall back to SENSOR_CONFIG.
            marker_count = self._detect_marker_count(f, sensor_id)
            if marker_count is None:
                marker_count = SENSOR_CONFIG[sensor_id]['marker_count']

            for t in range(T):
                frame = self._convert_frame(
                    f, t, T, sensor_id,
                    depth_prev, marker_prev, contact_state_prev
                )
                if frame is not None:
                    frames.append(frame)

        # 构建 TLabelData
        sensor_cfg = SENSOR_CONFIG[sensor_id]
        contact_count = sum(
            1 for fr in frames if fr.contact > 0.5
        )

        sensor_info = {
            "type": "vision-based_tactile",
            "model": sensor_cfg['name'],
            "manufacturer": "GelSight Inc.",
            "modality": "vision-based_tactile",
            "layout": {
                "type": "single_sensor",
                "sensor_id": sensor_id,
                "resolution": f"{sensor_cfg['resolution'][0]}x{sensor_cfg['resolution'][1]}",
                "marker_count": int(marker_count),
                "sampling_rate_hz": sensor_cfg['frame_rate'],
            }
        }

        episode_info = {
            "source": "univtac/UniVTAC",
            "source_file": h5_path.name,
            "total_frames": len(frames),
            "original_frames": T,
            "sensor_ids": sensor_ids,
        }

        return TLabelData(
            frames=frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
        )

    # ======================== 帧转换 ========================

    def _convert_frame(self, f, t, T, sensor_id,
                       depth_prev, marker_prev, contact_state_prev):
        """转换单帧 HDF5 → TLabelFrame"""
        sensor_path = f'tactile/{sensor_id}'
        timestamp = float(f['step'][t]) if 'step' in f else float(t)

        # 获取 pose
        pose = None
        if f'{sensor_path}/pose' in f:
            p = f[f'{sensor_path}/pose'][t]
            pose = [float(v) for v in p]

        # 计算 22 维特征
        tlabel_v2, extras = self._compute_all_features(
            f, sensor_path, sensor_id, t, T,
            depth_prev, marker_prev, contact_state_prev
        )

        # 操作阶段
        phase = self._get_phase(f, t)

        # 置信度：基于接触状态
        confidence = self._compute_confidence(tlabel_v2)

        # sensor_specific: 额外信息
        sensor_specific = {}
        if pose:
            sensor_specific['pose'] = pose
        if extras.get('deformation_magnitude_peak') is not None:
            sensor_specific['deformation_magnitude_peak'] = extras['deformation_magnitude_peak']
        if extras.get('centroid_y') is not None:
            sensor_specific['centroid_y'] = extras['centroid_y']
        if f'actor' in f:
            # 物体位姿
            for obj_name in f['actor'].keys():
                if f'actor/{obj_name}' in f:
                    obj_pose = f[f'actor/{obj_name}'][t]
                    sensor_specific[f'actor_{obj_name}'] = [float(v) for v in obj_pose]
                    break  # 只取第一个物体的位姿

        # 末端执行器
        if 'embodiment/ee' in f:
            ee = f['embodiment/ee'][t]
            sensor_specific['ee_pose'] = [float(v) for v in ee]
        if 'embodiment/joint' in f:
            joint = f['embodiment/joint'][t]
            sensor_specific['joint_states'] = [float(v) for v in joint]

        return TLabelFrame(
            frame_idx=t,
            timestamp_s=timestamp,
            schema_v2=TLabelSchemaV2.from_tlabel_v1(tlabel_v2),
            manipulation_phase=phase,
            confidence=confidence,
            sensor_specific=sensor_specific if sensor_specific else None,
        )

    # ======================== 22维特征计算 ========================

    def _compute_all_features(self, f, sensor_path, sensor_id, t, T,
                               depth_prev, marker_prev, contact_state_prev):
        """计算完整 22 维 TLabel v2 特征 + 额外字段"""
        tlabel_v2 = {
            "contact": 0.0, "deformation_magnitude": 0.0,
            "force_magnitude": 0.0, "force_peak": 0.0,
            "force_direction": 0.0, "slip_entropy": 0.0,
            "slip_event": 0.0, "texture_energy": 0.0,
            "edge_density": 0.0, "contact_area": 0.0,
            "centroid_x": 0.0,
            "normal_field_magnitude": 0.0, "normal_field_variance": 0.0,
            "shear_field_magnitude": 0.0, "shear_field_direction": 0.0,
            "delta_force_normal": 0.0, "delta_force_shear": 0.0,
            "friction_cone_ratio": 0.0,
            "optical_flow_magnitude": 0.0, "optical_flow_direction": 0.0,
            "temporal_deformation_rate": 0.0, "contact_transition": 0.0,
        }
        extras = {"deformation_magnitude_peak": None, "centroid_y": None}

        # ---- 从 depth 计算 ----
        depth = self._read_depth(f, sensor_path, t)
        if depth is not None:
            # contact
            contact = self._compute_contact(depth)
            tlabel_v2['contact'] = float(contact['contact_binary'])
            in_contact = bool(contact['in_contact'])

            # deformation
            defo = self._compute_deformation(depth)
            tlabel_v2['deformation_magnitude'] = defo['magnitude']
            extras['deformation_magnitude_peak'] = defo['peak']

            # contact_area
            area = self._compute_contact_area(depth)
            tlabel_v2['contact_area'] = area['pixel_count']

            # normal field
            nf = self._compute_normal_field(depth)
            tlabel_v2['normal_field_magnitude'] = nf['magnitude']
            tlabel_v2['normal_field_variance'] = nf['variance']

            # centroid
            cent = self._compute_centroid(depth)
            tlabel_v2['centroid_x'] = cent['x']
            extras['centroid_y'] = cent['y']

            # edge_density & texture_energy
            edge = self._compute_edge_density(depth)
            tlabel_v2['edge_density'] = edge['density']
            tlabel_v2['texture_energy'] = edge['energy']

            # 保存状态
            contact_state_prev[sensor_id] = in_contact

            # ---- 帧间特征 (需要上一帧 depth) ----
            depth_prev_frame = None
            if t > 0:
                depth_prev_frame = self._read_depth(f, sensor_path, t - 1)

            if depth_prev_frame is not None:
                # force direction from depth delta
                fd = self._compute_force_direction(depth, depth_prev_frame)
                tlabel_v2['force_magnitude'] = fd['force_magnitude']
                tlabel_v2['force_peak'] = fd['force_peak']
                tlabel_v2['force_direction'] = fd['force_direction']

                # delta forces
                dd = self._compute_delta_force(depth, depth_prev_frame)
                tlabel_v2['delta_force_normal'] = dd['normal']
                tlabel_v2['delta_force_shear'] = dd['shear']
                tlabel_v2['temporal_deformation_rate'] = dd['temporal_rate']

                # friction_cone_ratio
                fm = tlabel_v2['force_magnitude']
                sm = 0.0  # 下面 marker 计算后会更新
                if sm > 1e-6:
                    tlabel_v2['friction_cone_ratio'] = fm / sm

                # contact_transition
                prev_contact = contact_state_prev.get(sensor_id, False)
                ct = self._contact_transition(prev_contact, in_contact)
                tlabel_v2['contact_transition'] = ct

            depth_prev[sensor_id] = depth

        # ---- 从 marker 计算 ----
        marker = self._read_marker(f, sensor_path, t)
        if marker is not None:
            # slip
            slip = self._compute_slip(marker, marker_prev.get(sensor_id))
            tlabel_v2['slip_event'] = float(slip['event'])
            tlabel_v2['slip_entropy'] = slip['entropy']

            # optical flow
            of = self._compute_optical_flow(marker, marker_prev.get(sensor_id))
            tlabel_v2['optical_flow_magnitude'] = of['magnitude']
            tlabel_v2['optical_flow_direction'] = of['direction']

            # shear
            sh = self._compute_shear(marker, marker_prev.get(sensor_id))
            tlabel_v2['shear_field_magnitude'] = sh['magnitude']
            tlabel_v2['shear_field_direction'] = sh['direction']

            # 更新 friction_cone_ratio (现在有 shear 了)
            if t > 0 and depth is not None:
                fm = tlabel_v2['force_magnitude']
                sm = sh['magnitude']
                nm = tlabel_v2['normal_field_magnitude']
                if nm > 1e-6:
                    tlabel_v2['friction_cone_ratio'] = min(fm / nm, 10.0)
                elif sm > 1e-6:
                    tlabel_v2['friction_cone_ratio'] = float('inf')
                # 否则保持 0.0

            marker_prev[sensor_id] = marker

        return tlabel_v2, extras

    # ======================== Depth 特征方法 ========================

    def _compute_contact(self, depth):
        baseline = np.median(depth)
        threshold = baseline * 0.05
        contact_pixels = np.sum(depth > (baseline + threshold))
        ratio = contact_pixels / depth.size
        return {
            'contact_binary': 1 if ratio > 0.01 else 0,
            'in_contact': ratio > 0.01,
        }

    def _compute_deformation(self, depth):
        baseline = np.median(depth)
        deformation = np.maximum(depth - baseline, 0)
        return {
            'magnitude': round(float(np.mean(deformation)), 4),
            'peak': round(float(np.max(deformation)), 4),
        }

    def _compute_contact_area(self, depth):
        baseline = np.median(depth)
        threshold = baseline * 0.05
        mask = depth > (baseline + threshold)
        return {'pixel_count': float(np.sum(mask))}

    def _compute_normal_field(self, depth):
        gy, gx = np.gradient(depth)
        normal_mag = np.sqrt(gx**2 + gy**2)
        return {
            'magnitude': round(float(np.mean(normal_mag)), 4),
            'variance': round(float(np.var(normal_mag)), 4),
        }

    def _compute_centroid(self, depth):
        baseline = np.median(depth)
        threshold = baseline * 0.05
        mask = (depth > (baseline + threshold)).astype(float)
        total = np.sum(mask)
        if total < 1:
            return {'x': 0.0, 'y': 0.0}
        y_coords, x_coords = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
        cx = float(np.sum(x_coords * mask) / total)
        cy = float(np.sum(y_coords * mask) / total)
        return {'x': round(cx, 2), 'y': round(cy, 2)}

    def _compute_edge_density(self, depth):
        gy, gx = np.gradient(depth)
        edges = np.sqrt(gx**2 + gy**2)
        threshold = np.percentile(edges, 90)
        density = float(np.sum(edges > threshold)) / edges.size
        energy = float(np.sum(edges**2)) / edges.size
        return {'density': round(density, 4), 'energy': round(energy, 4)}

    def _compute_force_direction(self, depth_curr, depth_prev):
        delta = depth_curr - depth_prev
        force_mag = float(np.sqrt(np.sum(delta**2)))
        force_peak = float(np.max(np.abs(delta)))
        gy, gx = np.gradient(delta)
        direction = float(np.degrees(np.arctan2(np.sum(gy), np.sum(gx))))
        return {
            'force_magnitude': round(force_mag, 4),
            'force_peak': round(force_peak, 4),
            'force_direction': round(direction, 2),
        }

    def _compute_delta_force(self, depth_curr, depth_prev):
        delta = depth_curr - depth_prev
        normal = float(np.mean(np.abs(delta)))
        gy, gx = np.gradient(delta)
        shear = float(np.mean(np.sqrt(gx**2 + gy**2)))
        temporal_rate = float(np.mean(delta))
        return {
            'normal': round(normal, 4),
            'shear': round(shear, 4),
            'temporal_rate': round(temporal_rate, 4),
        }

    def _contact_transition(self, prev, curr):
        if not prev and curr:
            return 1.0
        elif prev and not curr:
            return -1.0
        elif prev and curr:
            return 0.5
        return 0.0

    # ======================== Marker 特征方法 ========================

    def _compute_slip(self, marker, prev_marker):
        if prev_marker is None:
            return {'event': 0, 'entropy': 0.0}
        curr_pos = marker[1]   # (1200, 2) 当前帧
        prev_pos = prev_marker[1] if prev_marker.ndim == 3 else marker[0]  # 上一帧
        displacement = curr_pos - prev_pos
        distances = np.sqrt(np.sum(displacement**2, axis=1))
        slip_threshold = 2.0
        slip_markers = distances > slip_threshold
        event = 1 if np.sum(slip_markers) > 10 else 0
        if np.sum(slip_markers) > 0:
            slip_dist = distances[slip_markers]
            hist, _ = np.histogram(slip_dist, bins=10, density=True)
            hist = hist[hist > 0]
            hist = hist / np.sum(hist)
            entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))
        else:
            entropy = 0.0
        return {'event': event, 'entropy': round(entropy, 4)}

    def _compute_optical_flow(self, marker, prev_marker):
        if prev_marker is None:
            return {'magnitude': 0.0, 'direction': 0.0}
        curr_pos = marker[1]
        prev_pos = prev_marker[1] if prev_marker.ndim == 3 else marker[0]
        disp = curr_pos - prev_pos
        mags = np.sqrt(np.sum(disp**2, axis=1))
        mean_mag = float(np.mean(mags))
        valid = mags > 1e-6
        if np.any(valid):
            angles = np.arctan2(disp[valid, 1], disp[valid, 0])
            direction = float(np.degrees(np.arctan2(
                np.mean(np.sin(angles)), np.mean(np.cos(angles))
            )))
        else:
            direction = 0.0
        return {'magnitude': round(mean_mag, 4), 'direction': round(direction, 2)}

    def _compute_shear(self, marker, prev_marker):
        if prev_marker is None:
            return {'magnitude': 0.0, 'direction': 0.0}
        curr_pos = marker[1]
        prev_pos = prev_marker[1] if prev_marker.ndim == 3 else marker[0]
        disp = curr_pos - prev_pos
        mags = np.sqrt(np.sum(disp**2, axis=1))
        shear_mag = float(np.mean(mags))
        valid = mags > 1e-6
        if np.any(valid):
            angles = np.arctan2(disp[valid, 1], disp[valid, 0])
            shear_dir = float(np.degrees(np.arctan2(
                np.mean(np.sin(angles)), np.mean(np.cos(angles))
            )))
        else:
            shear_dir = 0.0
        return {'magnitude': round(shear_mag, 4), 'direction': round(shear_dir, 2)}

    # ======================== 辅助方法 ========================

    def _get_num_frames(self, f):
        if 'step' in f:
            return len(f['step'])
        for key in f.keys():
            ds = f[key]
            if isinstance(ds, h5py.Dataset) and ds.ndim >= 1:
                return ds.shape[0]
        return 0

    def _get_phase(self, f, t):
        if 'atom/tag' not in f:
            return 'idle'
        tag = f['atom/tag'][t]
        if isinstance(tag, bytes):
            return TAG_TO_PHASE.get(tag, tag.decode('utf-8', errors='replace'))
        return TAG_TO_PHASE.get(tag, str(tag))

    def _read_depth(self, f, sensor_path, t):
        path = f'{sensor_path}/depth'
        if path in f:
            data = f[path][t]
            if data.ndim == 2:
                return data
        return None

    def _read_marker(self, f, sensor_path, t):
        path = f'{sensor_path}/marker'
        if path in f:
            data = f[path][t]
            if data.ndim == 3:
                return data
        return None

    def _detect_marker_count(self, f, sensor_id):
        """v0.23: infer marker count from the marker dataset shape.

        UniVTAC marker datasets have shape (T, 2, marker_size, 2); the number
        of markers is shape[2]. Returns None when the dataset is missing or
        the shape does not match, so callers can fall back to SENSOR_CONFIG.
        """
        path = f'tactile/{sensor_id}/marker'
        if path in f:
            shape = f[path].shape
            if len(shape) == 4:
                return int(shape[2])
        return None

    def _compute_confidence(self, tlabel_v2):
        contact = tlabel_v2.get('contact', 0)
        slip = tlabel_v2.get('slip_event', 0)
        if contact < 0.5 and slip < 0.5:
            return 0.95
        if contact > 0.5 and slip < 0.5:
            return 0.85
        if contact > 0.5 and slip > 0.5:
            return 0.6
        return 0.75

