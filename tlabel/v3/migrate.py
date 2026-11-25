"""
TLabel V2 → V3.0 迁移

两条铁律（TL-11.3）：
    TL-11.3-1  不能推断。旧数据里没有的信息，迁移后也不能凭空出现。
    TL-11.3-2  不能丢失。旧值即使无法映射到新字段，也必须保留在可追溯的位置。

因此本工具的行为是：
    · 能一一对应的字段，原值搬运，来源标 `externally_supplied`
    · 语义不明确的字段（如 V2 的 force_magnitude），一律置 null + `unknown_for_migration`，
      原值完整保留在 `sensor_specific.legacy_<name>`
    · 缺少必要上下文（坐标系、参考点、测量对象）的字段，同样置 null，
      而不是由工具编一个默认值
    · 每一次「没搬」都写进 `migration.audit`，说明原因

版本：3.0.0
权威来源：tlabel-spec-v3.md 第 11 章
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import schema as S

MIGRATE_TOOL = "tlabel-migrate"
MIGRATE_VERSION = "3.0.0"
LEGACY_NAMESPACE = "tlabel.v2"


class MigrationSourceUnsupported(ValueError):
    """源数据不是可迁移的 V2 格式。

    宁可拒绝，也不产出一份「看起来迁移成功、实际全为 null」的数据集——
    后者比失败更难被发现。
    """

#: 同名同义、可直接搬运的字段
#:
#: 注意 object_deformation 不在这里。V2 时代有 5 个视觉触觉 adapter
#: （gelsight / vtouch / daimon_dataset / daimon_dm_tac / ycb_slide）把
#: **图像强度 RMS**（无量纲）填进了这个单位是 mm 的字段。迁移工具看到
#: 一个数字无法判断它是毫米还是灰度比值，按 TL-11.3-1 一律不推断。
_DIRECT_FIELDS: Tuple[str, ...] = (
    "contact",
    "slip_event",
    "manipulation_phase",
    "contact_region",
    "texture_class",
)

#: 需要坐标系才能表达的三维/二维向量字段
_NEEDS_FRAME: Tuple[str, ...] = (
    "contact_centroid",
    "force_vector",
    "torque_vector",
    "slip_velocity",
)

#: V2 中存在但 V3 无法确定语义的字段 —— 一律不推断
_AMBIGUOUS_FIELDS: Tuple[str, ...] = ("force_magnitude", "object_deformation")

#: 语义不明的 V2 字段 → 它在 V3 中对应哪个字段。
#: field_status 必须挂在**目标字段**上：force_magnitude 在 V3 里没有同名项，
#: 它的去向是 normal_force；挂错地方会让校验器认为 normal_force 有值无状态。
_AMBIGUOUS_TARGET: Dict[str, str] = {
    "force_magnitude": "normal_force",
    "object_deformation": "object_deformation",
}

#: V2 中已弃用、只保留为扩展记录的字段
_DEPRECATED_FIELDS: Tuple[str, ...] = ("confidence", "compliance_level")

#: V2 字段 → 原值保留时使用的单位（用于 legacy 记录）
_LEGACY_UNITS: Dict[str, str] = {
    "force_magnitude": "N",
    "contact_centroid": "mm",
    "force_vector": "N",
    "torque_vector": "N.m",
    "slip_velocity": "mm/s",
    "temperature": "Cel",
    "confidence": "1",
    "compliance_level": "1",
    # V2 里这个字段既可能是真毫米，也可能是图像强度 RMS，单位不确定。
    # 审计里如实写 unknown，不替源数据背书。
    "object_deformation": "unknown",
}

#: V2 字段 → V3 中对应的上下文需求说明
_CONTEXT_HINT: Dict[str, str] = {
    "contact_centroid": "V2 未声明坐标系与计算方法（TL-5.2-3、TL-5.2-5）",
    "force_vector": "V2 未声明坐标系（TL-8.1-1）",
    "torque_vector": "V2 未声明坐标系与参考点（TL-5.5-1、TL-5.5-2）",
    "slip_velocity": "V2 未声明坐标系（TL-5.7-2）",
    "temperature": "V2 未声明测量对象（TL-5.8-2）",
    "force_magnitude": "V2 未区分法向力与合力模长，无法确定其物理含义（TL-5.3-2）",
    "object_deformation": (
        "V2 部分 adapter 填入图像强度 RMS（无量纲），无法确认是否为毫米（TL-5.4-1）"
    ),
}


def _legacy_entry(name: str, value: Any) -> Dict[str, Any]:
    """把无法映射的原值打包成结构化扩展记录（TL-11.3-2）。"""
    return {
        "value": value,
        "unit": _LEGACY_UNITS.get(name, "1"),
        "definition": f"V2 {name} 原值；{_CONTEXT_HINT.get(name, '语义未在新规范中定义')}",
        "method_id": f"{MIGRATE_TOOL}:legacy_passthrough_v1",
        "namespace": LEGACY_NAMESPACE,
    }


def _external_provenance(name: str, source_id: str, source_version: str) -> Dict[str, Any]:
    """迁移值的来源记录：值确实来自外部（V2 数据集），不伪造成测量或派生。"""
    return {
        "type": "externally_supplied",
        "source_id": source_id,
        "source_version": source_version,
        "source_ref": f"{LEGACY_NAMESPACE}:{name}",
    }


def migrate_dataset(
    v2: Dict[str, Any],
    *,
    frames_of_reference: Optional[List[Dict[str, Any]]] = None,
    frame_map: Optional[Dict[str, str]] = None,
    centroid_method: Optional[str] = None,
    weighting_quantity: Optional[str] = None,
    torque_reference_point: Optional[str] = None,
    measurement_target: Optional[str] = None,
    time_base: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    把一份 V2 数据集字典迁移成 V3.0。

    参数：
        v2:                      V2 `to_dict()` 的输出。
        frames_of_reference:     V3 坐标系定义列表。缺失时所有向量字段降级为 null。
        frame_map:               字段名 → frame_id。缺失时对应字段降级为 null。
        centroid_method:         "geometric" | "pressure_weighted"；缺失时 contact_centroid 降级。
        weighting_quantity:      centroid_method 为 pressure_weighted 时必填（TL-5.2-6）；
                                 缺失则同样降级——不替调用者编一个权重来源。
        torque_reference_point:  力矩参考点；缺失时 torque_vector 降级。
        measurement_target:      "sensor_body" | "contact_interface" | "object"；
                                 缺失时 temperature 降级。
        time_base:               TL-9.7-1 四种之一。省略时取 dataset_start，
                                 并在 migration 块里标记为推断值。

    返回：
        (v3_dataset, warnings)
    """
    frames_in = v2.get("frames") or []
    if not frames_in:
        raise MigrationSourceUnsupported("源数据没有 frames")
    if not any("schema_v2" in f for f in frames_in):
        dims = v2.get("tlabel_dimensions")
        hint = f"（tlabel_dimensions={dims}）" if dims is not None else ""
        raise MigrationSourceUnsupported(
            f"源帧缺少 schema_v2 键{hint}：本工具只迁移 V2.1 格式。"
            "若这是 V1 的 22 维格式，请先用 V1→V2 转换器升级。"
        )

    frame_map = dict(frame_map or {})
    time_base_inferred = time_base is None
    if time_base_inferred:
        time_base = "dataset_start"
    warnings: List[str] = []
    audit: List[Dict[str, Any]] = []

    source_id = str(v2.get("sensor_id") or v2.get("sensor", {}).get("sensor_id") or "unknown")
    source_version = str(v2.get("schema_version_v2") or v2.get("schema_version") or "2.x")

    v2_caps: Dict[str, Any] = dict(v2.get("capabilities") or {})

    # ── 数据集级 capability（TL-7.1-2：能力是数据集级声明）
    def _can(field: str) -> bool:
        return bool(v2_caps.get(field, False))

    # 能力判定的依据有两个：V2 的 capabilities 声明，以及数据里是否真的产出过该字段。
    # 只看声明会漏——V2 的 capabilities 常年填不全，而 slip_event 这类布尔字段
    # 默认就有 false（一个确定的判断），若能力却声明为 false，迁移后就成了
    # 「无能力却有值」（TL-7.3-2）。数据里出现过值，本身就是能力的证据。
    observed: set = set()
    for f in v2.get("frames") or []:
        sv2 = f.get("schema_v2") or {}
        for name in S.CORE_FIELDS:
            if name == "normal_force":
                continue
            if sv2.get(name) is not None:
                observed.add(name)

    def _has(field: str) -> bool:
        return _can(field) or field in observed

    v3_caps: Dict[str, bool] = {}
    for f in S.CORE_FIELDS:
        if f == "normal_force":
            # 有力度量能力存在，但不能据此宣称能产出法向力
            v3_caps[f] = bool(v2_caps.get("force_magnitude", False) or v2_caps.get("force_vector", False))
        elif f == "contact_centroid":
            v3_caps[f] = (
                _has(f)
                and bool(frame_map.get(f))
                and centroid_method is not None
                and (centroid_method != "pressure_weighted" or weighting_quantity is not None)
            )
        elif f in ("force_vector", "slip_velocity"):
            v3_caps[f] = _has(f) and bool(frame_map.get(f))
        elif f == "torque_vector":
            v3_caps[f] = _has(f) and bool(frame_map.get(f)) and bool(torque_reference_point)
        elif f == "temperature":
            v3_caps[f] = _has(f) and measurement_target is not None
        else:
            v3_caps[f] = _has(f)

    out_frames: List[Dict[str, Any]] = []

    for idx, v2_frame in enumerate(v2.get("frames") or []):
        sv2: Dict[str, Any] = dict(v2_frame.get("schema_v2") or {})
        # V2 帧顶层也带 manipulation_phase / confidence，与 schema_v2 内重复；
        # 以 schema_v2 为准，缺失时回退到顶层。
        if sv2.get("manipulation_phase") is None and v2_frame.get("manipulation_phase"):
            sv2["manipulation_phase"] = v2_frame["manipulation_phase"]
        if sv2.get("confidence") is None and v2_frame.get("confidence") is not None:
            sv2["confidence"] = v2_frame["confidence"]

        out: Dict[str, Any] = {f: None for f in S.CORE_FIELDS}
        status: Dict[str, str] = {}
        prov: Dict[str, Any] = {}
        sensor_specific: Dict[str, Any] = dict(v2_frame.get("sensor_specific") or {})

        # ── 直接搬运
        for name in _DIRECT_FIELDS:
            value = sv2.get(name)
            if value is None:
                continue
            out[name] = value
            prov[name] = _external_provenance(name, source_id, source_version)

        # ── 需要坐标系的向量字段
        for name in _NEEDS_FRAME:
            value = sv2.get(name)
            if value is None:
                continue
            fid = frame_map.get(name)
            extra_missing = (
                (
                    name == "contact_centroid"
                    and (
                        centroid_method is None
                        or (centroid_method == "pressure_weighted" and weighting_quantity is None)
                    )
                )
                or (name == "torque_vector" and not torque_reference_point)
            )
            if not fid or extra_missing:
                sensor_specific[f"legacy_{name}"] = _legacy_entry(name, value)
                status[name] = "unknown_for_migration"
                audit.append(
                    {
                        "field": name,
                        "frame_index": idx,
                        "original_value": value,
                        "original_unit": _LEGACY_UNITS.get(name, "1"),
                        "original_method": "unknown",
                        "target_location": f"sensor_specific.legacy_{name}",
                        "reason": _CONTEXT_HINT.get(name, "缺少必要上下文"),
                    }
                )
                continue
            out[name] = value
            out[S.FRAME_KEY[name]] = fid
            entry = _external_provenance(name, source_id, source_version)
            if name == "contact_centroid":
                entry["centroid_method"] = centroid_method
                if weighting_quantity is not None:
                    entry["weighting_quantity"] = weighting_quantity
            if name == "torque_vector":
                entry["reference_point"] = torque_reference_point
            prov[name] = entry

        # ── 温度
        temp = sv2.get("temperature")
        if temp is not None:
            if measurement_target is None:
                sensor_specific["legacy_temperature"] = _legacy_entry("temperature", temp)
                status["temperature"] = "unknown_for_migration"
                audit.append(
                    {
                        "field": "temperature",
                        "frame_index": idx,
                        "original_value": temp,
                        "original_unit": "Cel",
                        "original_method": "unknown",
                        "target_location": "sensor_specific.legacy_temperature",
                        "reason": _CONTEXT_HINT["temperature"],
                    }
                )
            else:
                out["temperature"] = temp
                entry = _external_provenance("temperature", source_id, source_version)
                entry["measurement_target"] = measurement_target
                prov["temperature"] = entry

        # ── 语义不明确的字段：一律不推断（TL-11.3-1）
        for name in _AMBIGUOUS_FIELDS:
            value = sv2.get(name)
            if value is None:
                continue
            sensor_specific[f"legacy_{name}"] = _legacy_entry(name, value)
            # 状态挂在目标字段上（force_magnitude → normal_force），
            # 不是一律挂 normal_force，也不是挂 V2 的旧名。
            status[_AMBIGUOUS_TARGET[name]] = "unknown_for_migration"
            audit.append(
                {
                    "field": name,
                    "frame_index": idx,
                    "original_value": value,
                    "original_unit": _LEGACY_UNITS.get(name, "unknown"),
                    "original_method": "unknown",
                    "target_location": f"sensor_specific.legacy_{name}",
                    "reason": _CONTEXT_HINT.get(name, ""),
                }
            )

        # ── 弃用字段：不丢，进扩展记录
        for name in _DEPRECATED_FIELDS:
            value = sv2.get(name)
            if value is None:
                continue
            sensor_specific[f"legacy_{name}"] = _legacy_entry(name, value)

        v3_frame: Dict[str, Any] = {
            "timestamp_s": v2_frame.get("timestamp_s"),
            **out,
        }
        if status:
            v3_frame["field_status"] = status
        if prov:
            v3_frame["field_provenance"] = prov
        if sensor_specific:
            v3_frame["sensor_specific"] = sensor_specific

        out_frames.append(v3_frame)

    # 审计去重：同一字段多帧重复记录时合并成一条，附受影响帧数。
    # TL-11.3-2：各帧原值不同时不能只留第一帧的值，否则审计本身就成了丢失源。
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    distinct: Dict[Tuple[str, str], List[Any]] = {}
    for a in audit:
        key = (a["field"], a["target_location"])
        vals = distinct.setdefault(key, [])
        if a["original_value"] not in vals:
            vals.append(a["original_value"])
        if key in merged:
            merged[key]["frame_count"] += 1
            continue
        item = dict(a)
        item.pop("frame_index", None)
        item["frame_count"] = 1
        merged[key] = item

    for key, item in merged.items():
        vals = distinct[key]
        if len(vals) == 1:
            item["original_value"] = vals[0]
        else:
            item["original_value"] = "<varies>"
            item["original_values"] = vals
            item["distinct_value_count"] = len(vals)

    if not frames_of_reference and any(frame_map.get(f) for f in _NEEDS_FRAME):
        warnings.append("提供了 frame_map 但没有 frames_of_reference，校验会报 FRAME_NOT_DECLARED")
    if time_base not in S.TIME_BASE_VALUES:
        warnings.append(f"time_base={time_base!r} 不在允许集内")

    v3: Dict[str, Any] = {
        "spec_version": S.SPEC_VERSION,
        "session_id": v2.get("episode", {}).get("episode_id")
        or v2.get("session_id")
        or "migrated",
        "time_base": time_base,
        "capabilities": v3_caps,
        "frames_of_reference": frames_of_reference or [],
        "frames": out_frames,
        "migration": {
            "tool": MIGRATE_TOOL,
            "tool_version": MIGRATE_VERSION,
            "time_base_inferred": time_base_inferred,
            "source_spec_version": source_version,
            "source_format": v2.get("format"),
            "source_id": source_id,
            "audit": list(merged.values()),
        },
    }
    return v3, warnings
