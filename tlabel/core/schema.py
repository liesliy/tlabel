"""
TLabel Schema V2.1 — 14维结构化语义空间

基于设计文档 v2.1 §3.1 定义，支持 Compliance Level (L1-L4) 分层。
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Mapping


# 枚举值定义（字符串类型，方便JSON序列化）
VALID_CONTACT_REGIONS = ("palmar", "digital", "lateral", "proximal", "distal", "dorsal", "other")
VALID_MANIPULATION_PHASES = ("pre_contact", "approach", "grasp", "lift", "hold", "place")
VALID_TEXTURE_CLASSES = ("smooth", "rough", "granular", "fibrous", "sticky", "slippery")
VALID_COMPLIANCE_LEVELS = ("L1", "L2", "L3", "L4")

# data_quality.level 枚举（Q1-Q4 用户自声明）
VALID_DATA_QUALITY_LEVELS = ("Q1", "Q2", "Q3", "Q4")

def _tristate(value: Any) -> Optional[bool]:
    """把任意输入归一成 True / False / None。

    v0.26（D8）：contact 与 slip_event 是三态字段。None 不是「假」，
    而是「这个数据集或这一帧无法给出该判断」。`bool(None)` 会把它压成
    False，那是把「不知道」谎报成「确定没有」，下游据此训练会学错。

    字符串分支是为了 CSV 往返：导出时未知写成空字段，读回是空串，
    `bool("")` 会得到 False——同一个坑换了个入口。这里把空串还原成
    None，把 `"0"` / `"false"` 之类按字面理解，避免 `bool("0") == True`
    这种反向错误。
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "":
            return None
        if s in ("true", "1", "1.0", "yes", "y", "t"):
            return True
        if s in ("false", "0", "0.0", "no", "n", "f"):
            return False
        raise ValueError(
            f"无法把字符串 {value!r} 解释为三态布尔值；"
            "期望 true/false/1/0 或空（空表示未知）"
        )
    return bool(value)



# 14维字段名列表（有序）
SCHEMA_V2_FIELD_NAMES = [
    "contact",
    "contact_centroid",
    "contact_region",
    "force_magnitude",
    "force_vector",
    "torque_vector",
    "slip_event",
    "slip_velocity",
    "manipulation_phase",
    "texture_class",
    "object_deformation",
    "temperature",
    "confidence",
    "compliance_level",
]


@dataclass
class TLabelSchemaV2:
    """
    TLabel Schema V2.1 — 14维结构化触觉语义标注
    
    Compliance Level 分层（§3.3）：
      L1 Basic:        contact, contact_centroid, slip_event, confidence
      L2 Force-Aware:  L1 + force_magnitude (约定必填)
      L3 Full-Vector:  L2 + force_vector [Fx,Fy,Fz]
      L4 Rich-Semantic: L3 + 所有 Optional 字段
    """

    # --- Required (无条件) ---
    # v0.26（D8）：二态布尔装不下真实数据。None 表示「不知道」——
    # 数据集没有该判断能力，或该帧无法判断。False 表示「确定的否」。
    # 两者语义不同，不得互相替代。
    contact: Optional[bool] = None
    slip_event: Optional[bool] = None
    confidence: float = 1.0
    compliance_level: str = "L1"

    # --- 条件 Required ---
    contact_centroid: Optional[List[float]] = None  # Required when contact=true

    # --- Optional ---
    contact_region: Optional[str] = None
    force_magnitude: Optional[float] = None          # L2+ 约定必填, 法向力(N)
    force_vector: Optional[List[float]] = None        # L3+ Optional, [Fx,Fy,Fz] (N)
    torque_vector: Optional[List[float]] = None       # Optional, [Mx,My,Mz] (N·m)
    slip_velocity: Optional[List[float]] = None       # Optional when slip_event=true, [vx,vy] (mm/s)
    manipulation_phase: Optional[str] = None          # enum: pre_contact/approach/grasp/lift/hold/place
    texture_class: Optional[str] = None               # enum: smooth/rough/granular/fibrous/sticky/slippery
    object_deformation: Optional[float] = None        # mm or ratio
    temperature: Optional[float] = None               # °C

    # ================================================================
    # Optional 扩展元数据（不属于 14 维核心语义，全部可选，向后兼容）
    # ================================================================

    data_quality: Optional[Dict[str, Any]] = None
    """
    用户自声明的数据处理级别（Q1-Q4）。TLabel 不做数据清洗/判断，
    仅提供字段和定义规范，类似食品包装的"保质期"。

    结构示例：
    {
        "level": "Q2",           # 必填：Q1/Q2/Q3/Q4
        "raw_processed": False,  # 是否经过原始数据处理
        "denoised": False,       # 是否已去噪
        "calibrated": True,      # 是否已校准
        "verified": False,       # 是否经过第三方/自动抽检
        "verified_by": None,     # 验证方（字符串或 None）
        "notes": ""              # 备注
    }

    Q1: 用户声明"原始数据，未处理"
    Q2: 用户声明"做过去噪/校准/时序对齐"
    Q3: 有第三方或自动抽检的置信度评分
    Q4: 人工全量标注+多传感器交叉验证
    """

    provenance: Optional[Dict[str, Any]] = None
    """
    最小来源元数据（"出生证明"），全部可选。仅保留直接影响数据
    可比性和校准的 4 个字段。其他（采集人员/环境温湿度/处理历史）
    属于数据管理平台职责，不在此处。

    结构示例：
    {
        "sensor_model": "GelSight Mini v2",
        "sensor_firmware": "1.3.0",
        "calibration_date": "2026-08-01",  # ISO 8601 日期
        "sampling_rate_hz": 270
    }
    """

    raw_sensor_values: Optional[Dict[str, Any]] = None
    """
    原始传感器读数保留（v2.2 新增）。

    用于存储核心 Schema 字段映射后可能丢失的原始传感器数据。
    与 data_quality（Q1-Q4 质量级别）语义完全不同。

    典型用途：
    - Tashan TS-F-A：6维原始数据中 tangential_direction、contact_indicator
      不会被 force_magnitude/force_vector 覆盖，通过此字段保留
    - 其他传感器：保留原始 ADC 值、温度补偿前数据等

    结构无固定 Schema，由适配器自行定义键名。
    """

    # ================================================================
    # 序列化
    # ================================================================

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "contact": self.contact,
            "contact_centroid": self.contact_centroid,
            "contact_region": self.contact_region,
            "force_magnitude": self.force_magnitude,
            "force_vector": self.force_vector,
            "torque_vector": self.torque_vector,
            "slip_event": self.slip_event,
            "slip_velocity": self.slip_velocity,
            "manipulation_phase": self.manipulation_phase,
            "texture_class": self.texture_class,
            "object_deformation": self.object_deformation,
            "temperature": self.temperature,
            "confidence": self.confidence,
            "compliance_level": self.compliance_level,
        }
        # Optional 扩展元数据：仅当非空时输出，保持向后兼容
        if self.data_quality is not None:
            result["data_quality"] = self.data_quality
        if self.provenance is not None:
            result["provenance"] = self.provenance
        if self.raw_sensor_values is not None:
            result["raw_sensor_values"] = self.raw_sensor_values
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TLabelSchemaV2":
        """从字典构建 TLabelSchemaV2"""
        return cls(
            contact=_tristate(data.get("contact")),
            contact_centroid=data.get("contact_centroid"),
            contact_region=data.get("contact_region"),
            force_magnitude=data.get("force_magnitude"),
            force_vector=data.get("force_vector"),
            torque_vector=data.get("torque_vector"),
            slip_event=_tristate(data.get("slip_event")),
            slip_velocity=data.get("slip_velocity"),
            manipulation_phase=data.get("manipulation_phase"),
            texture_class=data.get("texture_class"),
            object_deformation=data.get("object_deformation"),
            temperature=data.get("temperature"),
            confidence=float(data.get("confidence", 1.0)),
            compliance_level=str(data.get("compliance_level", "L1")),
            data_quality=data.get("data_quality"),
            provenance=data.get("provenance"),
            raw_sensor_values=data.get("raw_sensor_values"),
        )

    # ================================================================
    # 从 v1 (22维 flat dict) 转换
    # ================================================================

    @classmethod
    def from_tlabel_v1(cls, v1_dict: Dict[str, float]) -> "TLabelSchemaV2":
        """
        从旧版22维 tlabel_v2 flat dict 转换为新14维 Schema V2。
        
        映射规则：
          contact → contact (bool: >0.5 为 True)
          centroid_x → contact_centroid[0]  (需与 centroid_y 配对)
          force_magnitude → force_magnitude
          force_magnitude + force_direction → force_vector [Fx, Fy, Fz]
          slip_event → slip_event (bool: >0.5 为 True)
          slip_direction → slip_velocity [vx, vy] (方向→单位速度向量)
          manipulation_phase → manipulation_phase (映射枚举值)
          deformation_magnitude → object_deformation
          confidence → confidence
          无法导出的字段填 None
          compliance_level 默认 "L1"
        """
        # contact: v1 是 float 0/1, v2 是三态 bool
        # v0.26（D8）：源字段缺失或显式为 null 时必须保持 None。以前默认 0.0
        # 会把「源数据没给这个判断」变成「确定没有接触」——那是凭空断言。
        # 同时 float(None) 会直接抛 TypeError，这层防护一并解决。
        _raw_contact = v1_dict.get("contact")
        contact_val = None if _raw_contact is None else float(_raw_contact) > 0.5

        # contact_centroid: v1 有 centroid_x (归一化), 无 centroid_y → 填 None 或 [centroid_x, 0]
        centroid_x = v1_dict.get("centroid_x")
        if centroid_x is not None:
            contact_centroid = [float(centroid_x), 0.0]
        else:
            contact_centroid = None

        # force_magnitude
        force_mag = v1_dict.get("force_magnitude")
        if force_mag is not None:
            force_mag = float(force_mag)
            if force_mag == 0.0:
                force_mag = None  # 0可能是"传感器不支持"的占位

        # force_vector: 从 force_magnitude + force_direction 合成
        force_vector = None
        if force_mag is not None:
            force_dir = v1_dict.get("force_direction")
            if force_dir is not None:
                force_dir_rad = math.radians(float(force_dir))
                # v1 的 force_direction 是变形场梯度方向(degree)
                # 近似映射: [Fx, Fy, Fz] = [mag*sin(dir), 0, mag*cos(dir)]
                # Fz 为法向分量(主), Fx 为剪切分量
                fx = force_mag * math.sin(force_dir_rad)
                fy = 0.0
                fz = force_mag * math.cos(force_dir_rad)
                force_vector = [round(fx, 4), round(fy, 4), round(fz, 4)]

        # slip_event
        # v0.26（D8）：同上，缺失保持 None 而不是断言「没滑」。
        _raw_slip = v1_dict.get("slip_event")
        slip_val = None if _raw_slip is None else float(_raw_slip) > 0.5

        # slip_velocity: v1 的 slip_direction 是角度 → 转单位速度向量
        slip_velocity = None
        if slip_val:
            slip_dir = v1_dict.get("slip_direction")
            if slip_dir is not None:
                slip_dir_rad = math.radians(float(slip_dir))
                vx = math.cos(slip_dir_rad)
                vy = math.sin(slip_dir_rad)
                slip_velocity = [round(vx, 4), round(vy, 4)]

        # manipulation_phase: v1 枚举 → v2 枚举映射
        v1_phase = v1_dict.get("manipulation_phase", "")
        phase_map = {
            "idle": "pre_contact",
            "initial_contact": "approach",
            "stable_contact": "grasp",
            "slip": "grasp",
            "release": "place",
            "re_contact": "approach",
            "approach": "approach",
            "retract": "place",
            "grasp": "grasp",
            "transport": "lift",
            "hold": "hold",
        }
        manipulation_phase = phase_map.get(str(v1_phase))

        # object_deformation
        deformation = v1_dict.get("deformation_magnitude")
        if deformation is not None:
            deformation = float(deformation)

        # confidence: v1 中 confidence 是帧级别字段，不在 tlabel_v2 dict 里
        # 但 v1_dict 可能已经包含（从 TLabelFrame.confidence 传入）
        confidence = float(v1_dict.get("confidence", 1.0))

        return cls(
            contact=contact_val,
            contact_centroid=contact_centroid,
            contact_region=None,
            force_magnitude=force_mag,
            force_vector=force_vector,
            torque_vector=None,
            slip_event=slip_val,
            slip_velocity=slip_velocity,
            manipulation_phase=manipulation_phase,
            texture_class=None,
            object_deformation=deformation,
            temperature=None,
            confidence=confidence,
            compliance_level="L1",
        )

    # ================================================================
    # 验证
    # ================================================================

    def validate(self, capabilities: Optional[Mapping] = None) -> Tuple[bool, List[str]]:
        """验证 Schema 完整性与 compliance_level 一致性。

        Args:
            capabilities: 可选。传入时额外做「能力声明 ↔ 帧值」交叉校验：
                声明为 ``False`` 的维度，对应字段必须是 ``None``。
                默认不启用，避免破坏既有 adapter；新 adapter 与符合性校验器
                应显式传入。

        Returns:
            (is_valid, errors): is_valid=True 表示通过；errors 是错误消息列表。
        """
        errors: List[str] = []

        # 1. Required 字段检查
        if self.contact is not None and not isinstance(self.contact, bool):
            errors.append("contact must be bool or None")
        if self.slip_event is not None and not isinstance(self.slip_event, bool):
            errors.append("slip_event must be bool or None")

        # 2. confidence 范围
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence must be in [0.0, 1.0], got {self.confidence}")

        # 3. compliance_level 枚举
        if self.compliance_level not in VALID_COMPLIANCE_LEVELS:
            errors.append(
                f"compliance_level must be one of {VALID_COMPLIANCE_LEVELS}, got '{self.compliance_level}'"
            )

        # 4. contact=true 时 contact_centroid 不能为 null
        if self.contact is True and self.contact_centroid is None:
            errors.append("contact_centroid is required when contact=true")

        # 5. contact_centroid 维度检查
        if self.contact_centroid is not None:
            if not isinstance(self.contact_centroid, (list, tuple)) or len(self.contact_centroid) != 2:
                errors.append("contact_centroid must be [x, y] (2 floats)")

        # 6. compliance_level 条件必填
        level_order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
        level_num = level_order.get(self.compliance_level, 0)

        if level_num >= 2 and self.force_magnitude is None:
            errors.append("force_magnitude is required for compliance_level >= L2")

        if level_num >= 3 and self.force_vector is None:
            errors.append("force_vector is required for compliance_level >= L3")

        # 7. 枚举值合法性检查
        if self.contact_region is not None and self.contact_region not in VALID_CONTACT_REGIONS:
            errors.append(
                f"contact_region must be one of {VALID_CONTACT_REGIONS}, got '{self.contact_region}'"
            )

        if self.manipulation_phase is not None and self.manipulation_phase not in VALID_MANIPULATION_PHASES:
            errors.append(
                f"manipulation_phase must be one of {VALID_MANIPULATION_PHASES}, got '{self.manipulation_phase}'"
            )

        if self.texture_class is not None and self.texture_class not in VALID_TEXTURE_CLASSES:
            errors.append(
                f"texture_class must be one of {VALID_TEXTURE_CLASSES}, got '{self.texture_class}'"
            )

        # 8. 向量维度检查
        if self.force_vector is not None:
            if not isinstance(self.force_vector, (list, tuple)) or len(self.force_vector) != 3:
                errors.append("force_vector must be [Fx, Fy, Fz] (3 floats)")

        if self.torque_vector is not None:
            if not isinstance(self.torque_vector, (list, tuple)) or len(self.torque_vector) != 3:
                errors.append("torque_vector must be [Mx, My, Mz] (3 floats)")

        if self.slip_velocity is not None:
            if not isinstance(self.slip_velocity, (list, tuple)) or len(self.slip_velocity) != 2:
                errors.append("slip_velocity must be [vx, vy] (2 floats)")

        # 9. data_quality 结构检查（仅当提供时）
        if self.data_quality is not None:
            if not isinstance(self.data_quality, dict):
                errors.append("data_quality must be a dict")
            else:
                dq_level = self.data_quality.get("level")
                if dq_level is not None and dq_level not in VALID_DATA_QUALITY_LEVELS:
                    errors.append(
                        f"data_quality.level must be one of {VALID_DATA_QUALITY_LEVELS}, got '{dq_level}'"
                    )
                # 已知的 bool/str 字段类型检查
                for bool_field in ("raw_processed", "denoised", "calibrated", "verified"):
                    val = self.data_quality.get(bool_field)
                    if val is not None and not isinstance(val, bool):
                        errors.append(f"data_quality.{bool_field} must be bool, got {type(val).__name__}")
                notes_val = self.data_quality.get("notes")
                if notes_val is not None and not isinstance(notes_val, str):
                    errors.append("data_quality.notes must be str")
                vby_val = self.data_quality.get("verified_by")
                if vby_val is not None and not isinstance(vby_val, (str, type(None))):
                    errors.append("data_quality.verified_by must be str or None")

        # 10. provenance 结构检查（仅当提供时）
        if self.provenance is not None:
            if not isinstance(self.provenance, dict):
                errors.append("provenance must be a dict")
            else:
                for str_field in ("sensor_model", "sensor_firmware", "calibration_date"):
                    val = self.provenance.get(str_field)
                    if val is not None and not isinstance(val, str):
                        errors.append(f"provenance.{str_field} must be str")
                sr_val = self.provenance.get("sampling_rate_hz")
                if sr_val is not None:
                    if not isinstance(sr_val, (int, float)) or isinstance(sr_val, bool):
                        errors.append("provenance.sampling_rate_hz must be number")
                    elif sr_val <= 0:
                        errors.append("provenance.sampling_rate_hz must be > 0")
                # calibration_date ISO 8601 粗校验（YYYY-MM-DD）
                cal_date = self.provenance.get("calibration_date")
                if cal_date is not None and isinstance(cal_date, str):
                    import re
                    if not re.match(r"^\d{4}-\d{2}-\d{2}$", cal_date):
                        errors.append("provenance.calibration_date must be ISO 8601 date (YYYY-MM-DD)")

        # 11. raw_sensor_values 结构检查（仅当提供时）
        if self.raw_sensor_values is not None:
            if not isinstance(self.raw_sensor_values, dict):
                errors.append("raw_sensor_values must be a dict")

        # 12. capabilities 交叉校验（P1 / P14，可选）
        #
        # 打通"能力声明 ↔ 帧值"的第一条校验通道。一个维度若在 capabilities 中
        # 声明为 False，说明该传感器不产出这个物理量，对应字段必须是 None。
        # 填 False / 0.0 会把"没有这个能力"伪装成"测得没有 / 测得零"，
        # 下游无法再区分（P14）。
        if capabilities is not None:
            for key, has_cap in capabilities.items():
                if has_cap:
                    continue
                val = getattr(self, key, None)
                if val is None:
                    continue
                errors.append(
                    f"capability '{key}' is declared False, so the field must be None; "
                    f"got {val!r} — do not fill 0.0/False to stand in for 'not measured'"
                )

        return (len(errors) == 0, errors)
