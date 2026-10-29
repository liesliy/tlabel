"""
帕西尼数据集适配器 — 将OmniSharingDB HDF5数据转换为TLabelData

这是PaXini的数据集（离线）适配器，用于加载PXCap采集的.h5/.hdf5格式数据。
实时传感器适配器见 paxini_gen3.py。

复用paxini_adapter.py的PaxiniParser核心逻辑，封装为BaseAdapter接口。
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from tlabel.adapters.base import BaseAdapter
from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# 区域名称映射
REGION_NAME_MAP = {
    "palm_sensor1": "palm",
    "J32L": "index_prox", "J42L": "index_mid",
    "M6L": "middle",
    "S22L": "ring_prox", "S32L": "ring_dist", "S6L": "ring_base",
    "Z22L": "thumb_prox", "Z32L": "thumb_dist", "Z6L": "thumb_base",
    "W22L": "pinky_prox", "W32L": "pinky_dist", "W6L": "pinky_base",
}

# 接触阈值
CONTACT_THRESHOLD = 0.3


class PaxiniAdapter(BaseAdapter):
    """帕西尼 HDF5 → TLabelData

    Compliance Level: L2（只有法向力标量，无剪切力）
    """

    default_compliance_level: str = "L2"

    @property
    def name(self) -> str:
        return "paxini"

    @property
    def supported_extensions(self):
        return [".h5", ".hdf5"]

    def extract_schema(self, raw_frame_data) -> TLabelSchemaV2:
        """将原始数据帧转换为 TLabel Schema V2（14维）

        参数:
            raw_frame_data: dict，包含以下键：
                - contacts: list[dict] 各区域接触检测结果
                - slip_detected: bool 是否检测到滑移
                - prev_contacts: list[dict] 或 None 上一帧接触状态
                - prev_tlabel_v2: dict 或 None 上一帧22维特征
                - dt: float 帧间隔

        返回:
            TLabelSchemaV2 — L2级别，force_magnitude 从压力均值映射，
            force_vector 为 None（无法测3D力）
        """
        # 复用现有 _extract_tlabel_v2 静态方法获取22维dict
        tlabel_v2 = self._extract_tlabel_v2(
            contacts=raw_frame_data["contacts"],
            slip_detected=raw_frame_data["slip_detected"],
            prev_contacts=raw_frame_data.get("prev_contacts"),
            prev_tlabel_v2=raw_frame_data.get("prev_tlabel_v2"),
            dt=raw_frame_data.get("dt", 1.0),
        )

        contact = float(tlabel_v2["contact"]) > 0.5
        centroid_x = tlabel_v2.get("centroid_x", 0.5)

        return TLabelSchemaV2(
            contact=contact,
            contact_centroid=[float(centroid_x), 0.0] if contact else None,
            contact_region=None,
            force_magnitude=float(tlabel_v2["force_magnitude"]),
            force_vector=None,  # L2: 无法测3D力
            torque_vector=None,
            slip_event=float(tlabel_v2["slip_event"]) > 0.5,
            slip_velocity=None,
            manipulation_phase=None,
            texture_class=None,
            object_deformation=float(tlabel_v2["deformation_magnitude"]),
            temperature=None,
            confidence=self._compute_confidence(
                any(c["state"] != "no_contact" for c in raw_frame_data["contacts"]),
                raw_frame_data["slip_detected"],
            ),
            compliance_level=self.default_compliance_level,
        )

    def get_capabilities(self) -> Dict[str, bool]:
        """13 个合法能力键（Schema V2，规范 S-7.4-1）。

        判定口径：以 load() 实际产出的帧内容为准，不按 extract_schema 的
        理论能力。compliance_level 是帧级元字段，不属于能力键（S-7.4-2）。
        帕西尼压力阵列数据集。压力阵列无法测 3D 力方向；object_deformation
        实为压力均值代理量，不是物理形变。
        """
        return {
            "contact": True,
            "contact_centroid": True,
            "contact_region": False,
            "force_magnitude": True,  # 由压力阵列均值映射，非标定牛顿值
            "force_vector": False,
            "torque_vector": False,
            "slip_event": True,  # 真实滑移检测（质心偏移 + 力变化率）
            "slip_velocity": False,
            "manipulation_phase": True,
            "texture_class": False,
            "object_deformation": True,  # pressure proxy, not physical deformation
            "temperature": False,
            "confidence": True,
        }

    def get_sensor_info(self) -> Dict[str, Any]:
        return {
            "type": "distributed_taxel_array",
            "manufacturer": "paxini",
            "model": "PXCap",
        }

    def load(self, file_path: str,
             trajectory_id: Optional[int] = None,
             **kwargs) -> TLabelData:
        if not HAS_H5PY:
            raise ImportError("帕西尼适配器需要h5py: pip install h5py")

        hf = h5py.File(file_path, 'r')
        try:
            return self._parse(hf, file_path, **kwargs)
        finally:
            hf.close()

    def _parse(self, hf, file_path: str, **kwargs) -> TLabelData:
        """解析HDF5文件"""
        # 元数据
        meta = hf['dataset/meta'].attrs
        objects = self._decode_list(meta.get('物品种类', []))

        # 传感器布局
        obs = hf['dataset/observation']
        ref_hand = 'righthand' if 'righthand' in obs else 'lefthand'
        tactile_data = obs[f'{ref_hand}/tactile/data']
        sensor_lengths = [int(x) for x in tactile_data.attrs.get('sensor_lengths', [])]
        sensor_names = self._decode_list(tactile_data.attrs.get('sensor_names', []))
        if not sensor_names:
            sensor_names = [f"sensor_{i}" for i in range(len(sensor_lengths))]

        # 触觉数据
        try:
            right_tac = hf['dataset/observation/righthand/tactile/data'][:]
            left_tac = hf['dataset/observation/lefthand/tactile/data'][:]
        except:
            right_tac = np.zeros((1, sum(sensor_lengths)))
            left_tac = np.zeros((1, sum(sensor_lengths)))

        # 手部位姿
        try:
            right_pose = hf['dataset/observation/righthand/handpose/data'][:]
            left_pose = hf['dataset/observation/lefthand/handpose/data'][:]
        except:
            right_pose = np.zeros((1, 7))
            left_pose = np.zeros((1, 7))

        # 时间戳
        try:
            timestamps = hf['dataset/observation/aligned_timestamp'][:]
        except:
            num_frames = right_tac.shape[0]
            timestamps = np.arange(num_frames) * 33  # ~30Hz

        num_frames = len(timestamps)
        if num_frames > 1:
            dt_ms = np.mean(np.diff(timestamps))
            sample_rate = 1000.0 / dt_ms
        else:
            sample_rate = 30.0

        # 计算基线
        baseline_n = min(10, num_frames)
        baseline_right = np.mean(right_tac[:baseline_n], axis=0)
        baseline_left = np.mean(left_tac[:baseline_n], axis=0)

        # 逐帧处理
        tlabel_frames = []
        prev_contacts = None
        prev_tlabel_v2 = None
        dt = 1.0 / sample_rate if sample_rate > 0 else 1.0

        for fi in range(num_frames):
            r_tac = right_tac[fi]
            l_tac = left_tac[fi]

            # 切分为区域
            r_regions = self._split_regions(r_tac, sensor_lengths)
            l_regions = self._split_regions(l_tac, sensor_lengths)
            r_bl = self._split_regions(baseline_right, sensor_lengths)
            l_bl = self._split_regions(baseline_left, sensor_lengths)

            # 逐区域检测接触
            r_contacts = [self._detect_contact(reg, bl) for reg, bl in zip(r_regions, r_bl)]
            l_contacts = [self._detect_contact(reg, bl) for reg, bl in zip(l_regions, l_bl)]
            all_contacts = r_contacts + l_contacts

            # 滑移检测
            slip_detected = False
            if prev_contacts is not None:
                slip_detected = self._detect_slip(all_contacts, prev_contacts)

            # 提取22维TLabel v2
            tlabel_v2 = self._extract_tlabel_v2(
                all_contacts, slip_detected, prev_contacts,
                prev_tlabel_v2=prev_tlabel_v2, dt=dt
            )

            # 操作阶段
            any_contact = any(c["state"] != "no_contact" for c in all_contacts)
            phase = self._infer_phase(any_contact, slip_detected, fi, num_frames)

            # 置信度
            confidence = self._compute_confidence(any_contact, slip_detected)

            # 传感器特有数据
            sensor_specific = {
                "right_regions": {sensor_names[i]: r_contacts[i] for i in range(min(len(sensor_names), len(r_contacts)))},
                "left_regions": {sensor_names[i]: l_contacts[i] for i in range(min(len(sensor_names), len(l_contacts)))},
            }

            frame = TLabelFrame(
                frame_idx=fi,
                timestamp_s=round(float(timestamps[fi]) / 1000.0, 4),
                # 走 extract_schema 而不是裸 from_tlabel_v1：load 手上已有
                # extract_schema 需要的全部原始输入，绕过去只会丢掉增强。
                schema_v2=self.extract_schema({
                    "contacts": all_contacts,
                    "slip_detected": slip_detected,
                    "prev_contacts": prev_contacts,
                    "prev_tlabel_v2": prev_tlabel_v2,
                    "dt": dt,
                }),
                manipulation_phase=phase,
                confidence=confidence,
                sensor_specific=sensor_specific,
            )
            tlabel_frames.append(frame)
            prev_contacts = all_contacts
            prev_tlabel_v2 = tlabel_v2

        sensor_info = {
            "type": "distributed_taxel_array",
            "model": "PXCap",
            "manufacturer": "paxini",
            "modality": "pressure_array",
            "layout": {
                "type": "whole_hand",
                "num_sensors": len(sensor_lengths),
                "sensor_names": sensor_names,
                "sensor_lengths": sensor_lengths,
                "total_taxels": sum(sensor_lengths),
                "sampling_rate_hz": round(sample_rate, 1),
            }
        }

        episode_info = {
            "source": "OmniSharingDB",
            "file": Path(file_path).name,
            "objects": objects,
        }

        return TLabelData(
            frames=tlabel_frames,
            sensor_info=sensor_info,
            episode_info=episode_info,
            capabilities=self.get_capabilities(),
            sensor_id="paxini_pxcap",
        )

    @staticmethod
    def _decode_list(arr) -> List[str]:
        result = []
        for item in arr:
            if isinstance(item, bytes):
                result.append(item.decode('utf-8', errors='replace'))
            elif isinstance(item, np.bytes_):
                result.append(str(item, encoding='utf-8', errors='replace'))
            else:
                result.append(str(item))
        return result

    @staticmethod
    def _split_regions(data, sensor_lengths):
        regions = []
        offset = 0
        for length in sensor_lengths:
            regions.append(data[offset:offset + length])
            offset += length
        return regions

    @staticmethod
    def _detect_contact(region, baseline):
        adjusted = region - baseline
        active_mask = adjusted > CONTACT_THRESHOLD
        num_active = int(np.sum(active_mask))
        total = len(region)

        if num_active == 0:
            return {"state": "no_contact", "active_taxels": 0,
                    "contact_ratio": 0.0, "mean_force": 0.0, "max_force": 0.0}

        active_vals = adjusted[active_mask]
        ratio = num_active / total

        if ratio < 0.1:
            state = "initial_contact"
        elif ratio < 0.5:
            state = "partial_contact"
        else:
            state = "full_contact"

        return {
            "state": state,
            "active_taxels": num_active,
            "contact_ratio": round(float(ratio), 4),
            "mean_force": round(float(np.mean(active_vals)), 4),
            "max_force": round(float(np.max(active_vals)), 4),
        }

    @staticmethod
    def _detect_slip(curr_contacts, prev_contacts):
        """质心偏移+力变化率检测滑移"""
        for cc, pc in zip(curr_contacts, prev_contacts):
            if cc["state"] == "no_contact" or pc["state"] == "no_contact":
                continue
            centroid_shift = abs(cc["mean_force"] - pc["mean_force"])
            force_rate = abs(cc["mean_force"] - pc["mean_force"]) / max(pc["mean_force"], 0.01)
            if centroid_shift > 1.0 or force_rate > 0.3:
                return True
        return False

    @staticmethod
    def _extract_tlabel_v2(contacts, slip_detected, prev_contacts=None,
                           prev_tlabel_v2=None, dt=1.0):
        """帕西尼22维TLabel v2提取"""
        active_forces = [c["mean_force"] for c in contacts if c["state"] != "no_contact"]
        active_max = [c["max_force"] for c in contacts if c["state"] != "no_contact"]
        active_ratios = [c["contact_ratio"] for c in contacts if c["state"] != "no_contact"]

        any_contact = len(active_forces) > 0
        total_ratio = sum(active_ratios)

        nf_mag = float(np.mean(active_forces)) if active_forces else 0.0
        nf_var = float(np.var(active_forces)) if len(active_forces) > 1 else 0.0

        delta_fn = 0.0
        if prev_contacts is not None:
            prev_active = [c["mean_force"] for c in prev_contacts if c["state"] != "no_contact"]
            curr_mean = np.mean(active_forces) if active_forces else 0.0
            prev_mean = np.mean(prev_active) if prev_active else 0.0
            delta_fn = abs(curr_mean - prev_mean)

        # slip entropy
        forces = [c["mean_force"] for c in contacts if c["mean_force"] > 0]
        if len(forces) >= 2:
            total = sum(forces)
            if total > 1e-10:
                probs = np.array(forces) / total
                probs = probs[probs > 0]
                slip_ent = float(-np.sum(probs * np.log(probs)))
            else:
                slip_ent = 0.0
        else:
            slip_ent = 0.0

        # pressure centroid
        all_forces = [c["mean_force"] for c in contacts]
        if sum(all_forces) > 1e-10:
            weighted_pos = sum(i * f for i, f in enumerate(all_forces))
            centroid_x = weighted_pos / (sum(all_forces) * max(len(all_forces) - 1, 1))
        else:
            centroid_x = 0.5

        # --- 时序4维计算 ---
        # temporal_deformation_rate: 帧间deformation_magnitude差分
        temporal_deform_rate = 0.0
        if prev_tlabel_v2 is not None and dt > 0:
            prev_deform = prev_tlabel_v2.get("deformation_magnitude", 0.0)
            temporal_deform_rate = abs(nf_mag - prev_deform) / dt

        # contact_transition: 帧间contact + contact_area差分
        contact_trans = 0.0
        if prev_tlabel_v2 is not None:
            curr_contact = 1.0 if any_contact else 0.0
            prev_contact = prev_tlabel_v2.get("contact", 0.0)
            curr_area = min(total_ratio, 1.0)
            prev_area = prev_tlabel_v2.get("contact_area", 0.0)
            contact_trans = min(1.0, abs(curr_contact - prev_contact) +
                               abs(curr_area - prev_area) * 5.0)

        return {
            "contact": 1.0 if any_contact else 0.0,
            "deformation_magnitude": round(nf_mag, 4),
            "force_magnitude": round(nf_mag, 4),
            "force_peak": round(float(np.max(active_max)) if active_max else 0.0, 4),
            "force_direction": 0.0,
            "slip_entropy": round(slip_ent, 4),
            "slip_event": 1.0 if slip_detected else 0.0,
            "texture_energy": 0.0,
            "edge_density": 0.0,
            "contact_area": round(min(total_ratio, 1.0), 4),
            "centroid_x": round(centroid_x, 4),
            "normal_field_magnitude": round(nf_mag, 4),
            "normal_field_variance": round(nf_var, 4),
            "shear_field_magnitude": 0.0,
            "shear_field_direction": 0.0,
            "delta_force_normal": round(delta_fn, 4),
            "delta_force_shear": 0.0,
            "friction_cone_ratio": 0.0,
            # --- 时序4维 (光流2维为0) ---
            "optical_flow_magnitude": 0.0,
            "optical_flow_direction": 0.0,
            "temporal_deformation_rate": round(temporal_deform_rate, 4),
            "contact_transition": round(contact_trans, 4),
        }

    @staticmethod
    def _infer_phase(any_contact, slip, fi, total):
        if not any_contact:
            return "approach" if fi < total * 0.5 else "retract"
        if slip:
            return "slip"
        return "stable_contact"

    @staticmethod
    def _compute_confidence(any_contact, slip):
        if not any_contact and not slip:
            return 0.95
        if any_contact and not slip:
            return 0.8
        return 0.5
