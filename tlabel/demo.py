"""
tlabel.demo() — 一键体验内置Demo数据

无需任何文件，直接加载内置数据集，在Jupyter中弹出交互面板。
"""

from pathlib import Path
from typing import Optional, List

from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2
from tlabel.core.capabilities import CAPABILITY_KEYS, filter_capabilities

# 内置Demo数据目录
_DEMO_DIR = Path(__file__).parent / "demo_data"


def _value_has_signal(value) -> bool:
    """判断一个维度取值是否携带真实输出（而非占位零值）。"""
    if isinstance(value, (list, tuple)):
        return any(
            v is not None and abs(float(v)) > 0.0 for v in value
        )
    try:
        return abs(float(value)) > 0.0
    except (TypeError, ValueError):
        # 枚举等非数值类型：只要能取到就不是缺失
        return True


def _infer_capabilities_from_frames(frames) -> dict:
    """扫描帧内容反推 capabilities 声明。

    内置 demo 文件携带的是 22 维时代的旧 caps（``has_image`` / ``dimensions``
    等），与 Schema V2 的 13 个能力键没有逐键对应关系，归一化后会是空集。
    此时若退化为"只声明 contact"，导出文件的 ``capabilities`` 就会与实际帧
    内容自相矛盾（S-6.3-3）——例如 gelsight demo 有 106 帧 force_magnitude
    真实数据，文件头却只写 contact，下游按 caps 解析会漏读整列。

    判定规则：

    - ``None`` 是 tlabel 表达"该帧缺失此维度"的唯一标记，跳过；
    - bool 键取到 ``False`` 也算有效输出（"这一帧没有接触/没有滑移"是判断，
      不是缺失）；
    - 数值与向量只在出现过非零值时声明，避免把全零占位列误报为能力；
    - ``contact`` 是 schema 必填键，始终出现在结果里。
    """
    inferred: dict = {}
    for frame in frames:
        schema = getattr(frame, "schema_v2", None)
        if schema is None:
            continue
        for key in CAPABILITY_KEYS:
            if inferred.get(key):
                continue
            value = getattr(schema, key, None)
            if value is None:
                continue
            if isinstance(value, bool) or _value_has_signal(value):
                inferred[key] = True
        if len(inferred) == len(CAPABILITY_KEYS):
            break
    inferred.setdefault("contact", False)
    return inferred

# 可用的Demo传感器列表
AVAILABLE_SENSORS = [
    "gelsight", "digit", "paxini", "daimon", "touchd",
    "gelsight_images", "primitives_demo",
    "gelsight_force_demo",   # v0.14: 视触觉→力推断→Primitive标注
    "tactile_events_demo",   # v0.14: 触觉事件标注系统
    "paxini_gen3",           # v0.14: PaXini GEN3 实时传感器
    "daimon_dm_tac",         # v0.14: DM-Tac 实时传感器
]



def _generate_synthetic_tactile_images(num_frames: int = 10) -> list:
    """生成合成触觉图像（用于demo演示）"""
    import numpy as np
    
    images = []
    for i in range(num_frames):
        # 创建320x240灰度图像
        img = np.zeros((240, 320), dtype=np.uint8)
        
        # 添加高斯噪声模拟传感器噪声
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 根据帧阶段添加不同的接触区域
        phase = i / num_frames
        
        if phase < 0.3:  # approach
            # 小圆形接触区域，逐渐增大
            radius = int(20 + phase * 50)
            cv2 = None
            try:
                import cv2
                cv2.circle(img, (160, 120), radius, 180, -1)
            except:
                pass
        elif phase < 0.7:  # contact
            # 大圆形接触区域
            try:
                import cv2
                cv2.circle(img, (160, 120), 60, 200, -1)
            except:
                pass
        else:  # slip
            # 接触区域偏移，模拟滑动
            try:
                import cv2
                offset_x = int((phase - 0.7) * 100)
                cv2.circle(img, (160 + offset_x, 120), 50, 180, -1)
            except:
                pass
        
        images.append(img)
    
    return images


def demo(sensor: Optional[str] = None, **kwargs) -> TLabelData:
    """
    加载内置Demo数据集，快速体验TLabel标注面板

    参数:
        sensor: 传感器类型，可选:
            "gelsight" / "digit" / "paxini" / "daimon" / "touchd" — 基础传感器demo
            "gelsight_images" — 带触觉图像的GelSight demo
            "primitives_demo" — Motor Primitive 标注 demo (v0.13)
            "gelsight_force_demo" — 视触觉→力推断→Primitive标注 (v0.14)
            "tactile_events_demo" — 触觉事件标注系统 (v0.14)
            "paxini_gen3" — PaXini GEN3 实时传感器 demo (v0.14)
            "daimon_dm_tac" — DM-Tac 实时传感器 demo (v0.14)
            不传则默认加载 gelsight demo


    返回:
        TLabelData — 可直接 review() / export()

    用法:
        import tlabel
        data = tlabel.demo()              # 默认GelSight demo
        data = tlabel.demo('digit')       # DIGIT demo
        data = tlabel.demo('touchd')      # ToucHD-Force demo
        data = tlabel.demo('gelsight_images')  # 带触觉图像的GelSight demo
        data = tlabel.demo('primitives_demo')  # Motor Primitive 标注 demo
        data = tlabel.demo('gelsight_force_demo')  # 力推断 demo (v0.14)
        data = tlabel.demo('tactile_events_demo')  # 事件检测 demo (v0.14)

        data.review()                     # Jupyter弹出面板
    """
    if sensor is None:
        sensor = "gelsight"

    sensor = sensor.lower().strip()
    if sensor not in AVAILABLE_SENSORS:
        raise ValueError(
            f"未知的传感器类型: '{sensor}'\n"
            f"可用选项: {AVAILABLE_SENSORS}"
        )

    demo_file = _DEMO_DIR / f"demo_{sensor}.json"
    # v0.14: 动态demos复用gelsight基础数据
    if not demo_file.exists() and sensor in ("gelsight_force_demo", "tactile_events_demo"):
        demo_file = _DEMO_DIR / "demo_gelsight.json"
    # v0.14: paxini_gen3 / daimon_dm_tac 没有预设JSON，生成随机帧
    if not demo_file.exists() and sensor in ("paxini_gen3", "daimon_dm_tac"):
        return _build_realtime_sensor_demo(sensor)
    if not demo_file.exists():
        raise FileNotFoundError(f"Demo数据文件缺失: {demo_file}")

    # 直接从TLabel Format JSON加载，不走适配器
    import json
    with open(demo_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 如果是gelsight_images，生成合成图像
    synthetic_images = None
    if sensor == "gelsight_images":
        try:
            num_frames = len(raw.get("frames", []))
            synthetic_images = _generate_synthetic_tactile_images(num_frames)
        except Exception:
            pass

    frames = []
    for idx, fd in enumerate(raw.get("frames", [])):
        # 如果有合成图像，使用合成图像
        image_data = None
        if synthetic_images and idx < len(synthetic_images):
            image_data = synthetic_images[idx]
        
        frame = TLabelFrame(
            frame_idx=fd["frame_idx"],
            timestamp_s=fd["timestamp_s"],
            schema_v2=TLabelSchemaV2.from_tlabel_v1(fd["tlabel_v2"]),
            manipulation_phase=fd.get("manipulation_phase", "idle"),
            confidence=fd.get("confidence", 1.0),
            sensor_specific=fd.get("sensor_specific"),
            image=image_data,  # v0.12: 支持图像数据
            image_path=fd.get("image_path"),  # v0.12: 支持图像路径
        )
        frames.append(frame)

    # v0.23.3: 内置 demo 数据文件携带的是 22 维时代的旧 caps 结构
    # （has_image / dimensions / 旧字段名），与 Schema V2 的 13 个能力键
    # 没有可靠的逐键对应关系。原始旧 caps 完整保留在
    # episode_info.legacy_capabilities，不丢信息。
    #
    # v0.24.2: 旧 caps 归一化后若为空集，原先退化为"只声明 contact"。但帧里
    # 明明有 force_magnitude / slip_event 等真实数据，导出文件因此声明与输出
    # 自相矛盾（违反 S-6.3-3），下游按 caps 判断字段是否存在会整列漏读。
    # 改为扫描帧内容反推，保证"声明即输出"。
    raw_caps = raw.get("capabilities", {}) or {}
    episode_info = dict(raw.get("episode", {}) or {})
    legacy_caps = filter_capabilities(raw_caps)
    if raw_caps and legacy_caps != raw_caps:
        # 旧 caps 未被完整表达（含无交集的旧结构），原始内容原样留档
        episode_info.setdefault("legacy_capabilities", raw_caps)

    normalized_caps = _infer_capabilities_from_frames(frames)
    for key, value in legacy_caps.items():
        if value:
            normalized_caps[key] = True
    normalized_caps = filter_capabilities(normalized_caps)

    data = TLabelData(
        frames=frames,
        sensor_info=raw.get("sensor", {}),
        episode_info=episode_info,
        capabilities=normalized_caps,
        schema_version=raw.get("schema_version", "0.4.0"),
    )

    # v0.13: 加载 primitive_annotations
    if "primitive_annotations" in raw:
        from tlabel.core.primitive import PrimitiveAnnotation
        for pa_dict in raw["primitive_annotations"]:
            try:
                pa = PrimitiveAnnotation(
                    name=pa_dict["primitive_name"],
                    start=pa_dict["start_frame"],
                    end=pa_dict["end_frame"],
                    confidence=pa_dict.get("confidence", 1.0),
                    source=pa_dict.get("source", "manual"),
                )
                data.primitive_annotations.append(pa)
            except (ValueError, KeyError):
                pass

    # v0.14: 加载 tactile_events
    if "tactile_events" in raw:
        from tlabel.core.events import TactileEvent
        for ev_dict in raw["tactile_events"]:
            try:
                te = TactileEvent.from_dict(ev_dict)
                data.tactile_events.append(te)
            except (ValueError, KeyError):
                pass

    # v0.14: 动态生成 demos（无预设JSON文件）
    if sensor == "gelsight_force_demo":
        data = _build_gelsight_force_demo(data)
    elif sensor == "tactile_events_demo":
        data = _build_tactile_events_demo(data)


    return data


def list_demos() -> List[str]:
    """列出所有可用的Demo传感器类型"""
    return AVAILABLE_SENSORS.copy()


# ============================================================
# v0.14.0: 动态 Demo 构建器
# ============================================================

def _build_gelsight_force_demo(data: TLabelData) -> TLabelData:
    """
    构建GelSight力推断Demo

    模拟一个只有GelSight图像数据（无力/力矩传感器）的场景：
    1. 将原始force_magnitude清零（模拟无F/T传感器）
    2. 保留deformation_magnitude数据
    3. 调用auto_force_estimate()推断力
    4. 调用predict_primitives()展示完整流程
    """
    from tlabel.predict.force_estimator import auto_force_estimate
    from tlabel.predict.engine import PredictEngine

    # 步骤1: 清零force_magnitude（模拟无力传感器）— Schema V2
    for frame in data.frames:
        frame.schema_v2.force_magnitude = None
        frame.schema_v2.force_vector = None
        frame.sensor_specific["normal_field_magnitude"] = 0.0
        frame.sensor_specific["normal_field_variance"] = 0.0

    # 步骤2: 设置传感器类型信息
    data.sensor_info = {
        "type": "gelsight_mini",
        "model": "GelSight Mini (No F/T Sensor)",
        "resolution": "240x320",
    }

    # 步骤3: 用推断的力进行primitive标注
    # predict_primitives内部会自动检测力缺失并调用force_estimator
    engine = PredictEngine()
    annotations = engine.predict_primitives(data)

    # 应用primitive标注
    from tlabel.core.primitive import PrimitiveAnnotation
    for ann in annotations:
        try:
            # v0.14: 映射 estimated source
            source = ann.get('source', 'ai_predicted')
            if source == 'ai_predicted_estimated':
                source = 'ai_predicted'
            pa = PrimitiveAnnotation(
                name=ann['primitive_name'],
                start=ann['start_frame'],
                end=ann['end_frame'],
                confidence=ann['confidence'],
                source=source,
            )
            data.primitive_annotations.append(pa)
        except ValueError:
            pass

    # 更新episode信息
    data.episode_info["description"] = (
        "GelSight力推断Demo — 模拟无F/T传感器场景，"
        "通过deformation_magnitude + 弹性体刚度推断力，"
        "再基于推断力进行Motor Primitive标注"
    )

    return data


def _build_tactile_events_demo(data: TLabelData) -> TLabelData:
    """
    构建触觉事件标注Demo

    展示自动检测的6种触觉事件:
    - contact_onset / contact_loss: 接触开始/丢失
    - slip: 滑移事件
    - force_spike: 力值突变
    - deformation_anomaly: 形变异常
    - stable_grip: 稳定抓握
    """
    from tlabel.predict.engine import PredictEngine

    engine = PredictEngine()

    # 自动检测事件
    count = engine.apply_events(data, min_confidence=0.5)

    # 更新episode信息
    event_summary = {}
    for ev in data.tactile_events:
        t = ev.event_type
        event_summary[t] = event_summary.get(t, 0) + 1

    data.episode_info["description"] = (
        "触觉事件标注Demo — 自动检测6种关键事件: "
        f"contact_onset/contact_loss/slip/force_spike/"
        f"deformation_anomaly/stable_grip。"
        f"共检测到 {count} 个事件"
    )
    data.episode_info["event_summary"] = event_summary

    return data


def _build_realtime_sensor_demo(sensor: str) -> TLabelData:
    """
    构建实时传感器Demo（paxini_gen3 / daimon_dm_tac）— Schema V2

    由于这些传感器没有预设JSON文件，生成随机帧数据用于演示。
    v0.17: 直接构建 TLabelSchemaV2，不再使用旧 22 维 tlabel_v2 dict。
    """
    import random

    num_frames = 50
    is_paxini = sensor == "paxini_gen3"

    frames = []
    prev_schema = None
    sample_rate = 100 if is_paxini else 120
    dt = 1.0 / sample_rate

    # 模拟一段接触序列: idle → contact → slip → stable → release
    for fi in range(num_frames):
        phase_frac = fi / num_frames
        if phase_frac < 0.2:
            contact = False
            force_mag = 0.0
            slip = False
            phase = "idle"
        elif phase_frac < 0.4:
            contact = True
            force_mag = random.uniform(0.3, 0.7)
            slip = False
            phase = "approach"
        elif phase_frac < 0.6:
            contact = True
            force_mag = random.uniform(0.5, 0.9)
            slip = random.random() > 0.5
            phase = "grasp" if not slip else "grasp"
        elif phase_frac < 0.8:
            contact = True
            force_mag = random.uniform(0.4, 0.7)
            slip = False
            phase = "hold"
        else:
            contact = False
            force_mag = 0.0
            slip = False
            phase = "place"

        deformation_mag = force_mag * 0.8
        contact_area = 1.0 * random.uniform(0.3, 0.7) if contact else 0.0

        # Schema V2 compliance level
        compliance_level = "L2" if force_mag > 0 else "L1"

        # Build TLabelSchemaV2 directly
        schema = TLabelSchemaV2(
            contact=contact,
            contact_centroid=[0.5, 0.5] if contact else None,
            force_magnitude=round(force_mag, 4) if force_mag > 0 else None,
            force_vector=None,
            slip_event=slip,
            slip_velocity=None,
            manipulation_phase=phase if phase in ("pre_contact", "approach", "grasp", "lift", "hold", "place") else None,
            object_deformation=round(deformation_mag, 4) if deformation_mag > 0 else None,
            confidence=0.8 if contact else 0.95,
            compliance_level=compliance_level,
        )

        frame = TLabelFrame(
            frame_idx=fi,
            timestamp_s=round(fi / sample_rate, 4),
            schema_v2=schema,
            manipulation_phase=phase,
            confidence=0.8 if contact else 0.95,
            sensor_specific={
                "demo": True, "source": sensor,
                "contact_area": round(min(contact_area, 1.0), 4),
            },
        )
        frames.append(frame)
        prev_schema = schema

    if is_paxini:
        sensor_info = {
            "type": "distributed_taxel_array",
            "model": "GEN3",
            "manufacturer": "paxini",
            "modality": "pressure_array",
        }
        sensor_id = "paxini_gen3"
    else:
        sensor_info = {
            "type": "vision-based_tactile",
            "model": "DM-Tac",
            "manufacturer": "daimon",
            "modality": "vtla_multimodal",
        }
        sensor_id = "daimon_dm_tac"

    # Schema V2 capabilities（13 个合法能力键，compliance_level 属帧级元字段）
    caps = {
        "contact": True, "contact_centroid": True,
        "contact_region": False, "force_magnitude": True,
        "force_vector": False, "torque_vector": False,
        "slip_event": True, "slip_velocity": False,
        "manipulation_phase": True, "texture_class": False,
        "object_deformation": True, "temperature": False,
        "confidence": True,
    }

    return TLabelData(
        frames=frames,
        sensor_info=sensor_info,
        episode_info={"source": f"{sensor}_demo", "description": f"{sensor} 合成Demo数据"},
        capabilities=caps,
        sensor_id=sensor_id,
    )