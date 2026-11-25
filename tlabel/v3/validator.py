"""
TLabel V3.0 参考实现 — 校验器

实现规范第 10 章的校验层次 L-A / L-B / L-C / L-D，输出 10.5 定义的稳定错误码。

设计原则：
1. 只报规范明文要求的东西。规范没说的，不判 error。
2. 证据不足时输出 not_verified，不压成 error 或 pass（TL-10.4-3）。
3. 每条 finding 都带条款号和字段路径（TL-10.5-2）。

版本：3.0.0
权威来源：tlabel-spec-v3.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import schema as S


# ─────────────────────────────────────────────────────────────
# 结果结构
# ─────────────────────────────────────────────────────────────

@dataclass
class Finding:
    """一条校验发现。"""

    code: str
    clause: str
    path: str
    message: str
    outcome: str = S.OUTCOME_ERROR
    #: TL-10.5-2 要求报告带上「出问题的那个值」。它和 message 里的描述不同：
    #: message 是给人读的解释，value 是给机器比的原始值——下游要按值去原始
    #: 数据里定位这一条，靠 message 做正则匹配是不可靠的。
    value: Any = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "clause": self.clause,
            "path": self.path,
            "message": self.message,
            "outcome": self.outcome,
            "value": self.value,
        }


@dataclass
class Report:
    """一次校验的完整结果。"""

    findings: List[Finding] = field(default_factory=list)
    frames_checked: int = 0
    level: str = "structural"
    #: 本级别**没有执行**的检查（TL-10.2-3）。
    #:
    #: 必须有这个字段，因为「通过了」和「没查」在报告里长得一模一样——
    #: 都是零 finding。一份 Structural 级报告如果不说明它跳过了语义检查，
    #: 使用者会以为数据已经符合标准，而实际上能力/状态一致性、来源、
    #: 坐标语义一条都没查。
    not_performed: List[str] = field(default_factory=list)

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.outcome == S.OUTCOME_ERROR]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.outcome == S.OUTCOME_WARNING]

    @property
    def not_verified(self) -> List[Finding]:
        return [f for f in self.findings if f.outcome == S.OUTCOME_NOT_VERIFIED]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def codes(self) -> List[str]:
        return [f.code for f in self.errors]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "level": self.level,
            "frames_checked": self.frames_checked,
            "not_performed": list(self.not_performed),
            "errors": [f.as_dict() for f in self.errors],
            "warnings": [f.as_dict() for f in self.warnings],
            "not_verified": [f.as_dict() for f in self.not_verified],
        }


#: 每个符合性级别**未执行**的检查（TL-10.2-3）。
#:
#: Structural 只查键、类型、单位代码。语义一致性、来源、上下文、坐标语义、
#: 扩展容器结构、多实例与时间语义全部不查——这些恰好是 V2 出问题的地方。
_SKIPPED_AT: Dict[str, List[str]] = {
    "structural": [
        "capability / value / status consistency (TL-7.3)",
        "value-level provenance for every non-null core field (TL-9.2-1)",
        "required measurement context per field (TL-9.4-1)",
        "coordinate semantic checks C-1..C-12 (TL-10.3-1)",
        "extension container structure (TL-9.5)",
        "multi-instance declaration and time semantics (TL-9.6, TL-9.7)",
    ],
    "core": [
        "fully structured sensor_specific entries, Extended only (TL-9.5-3)",
    ],
    "extended": [],
}


class _Ctx:
    """校验过程中的共享状态。"""

    def __init__(self, dataset: Dict[str, Any], report: Report):
        self.dataset = dataset
        self.report = report
        self.frames: Dict[str, Dict[str, Any]] = {}
        for f in dataset.get("frames_of_reference") or []:
            if isinstance(f, dict) and "frame_id" in f:
                self.frames[f["frame_id"]] = f
        self.capabilities: Dict[str, Any] = dataset.get("capabilities") or {}
        self.migration: Optional[Dict[str, Any]] = dataset.get("migration")

    def err(self, code: str, clause: str, path: str, msg: str,
            value: Any = None) -> None:
        self.report.findings.append(Finding(code, clause, path, msg, value=value))

    def warn(self, code: str, clause: str, path: str, msg: str,
             value: Any = None) -> None:
        self.report.findings.append(
            Finding(code, clause, path, msg, S.OUTCOME_WARNING, value=value)
        )

    def nv(self, code: str, clause: str, path: str, msg: str,
           value: Any = None) -> None:
        self.report.findings.append(
            Finding(code, clause, path, msg, S.OUTCOME_NOT_VERIFIED, value=value)
        )


#: 字段 → 其类型 / 长度约束所在的条款。
#:
#: 规范 §10.5 的码表把 `TYPE_MISMATCH` / `LENGTH_MISMATCH` / `ENUM_INVALID` 标为
#: 「条款由字段决定」，因为一个码必然覆盖多个字段，而每个字段的约束写在各自的
#: 条款里。TL-10.4-1 要求引用**它违反的**条款：`LENGTH_MISMATCH` 报在
#: `force_vector` 上时违反的是 TL-5.4-1 的「exactly 3」，不是 TL-5.2-1 的
#: 「exactly 2」。照抄一个默认条款会让第三方按条款号回溯时对不上号。
_TYPE_CLAUSE: Dict[str, str] = {
    # 用 5.1-5 而不是 5.1-1：5.1-1 约束生产者「必须填三个值之一」，
    # 5.1-5 才约束校验器「必须拒绝非三值」。校验器报错时引的应该是后者，
    # 否则按条款号回溯的人会看到一条「生产者义务」而找不到验证要求。
    "contact": "TL-5.1-5",
    "contact_centroid": "TL-5.2-1",
    "normal_force": "TL-5.3-1",
    "force_vector": "TL-5.4-1",
    "torque_vector": "TL-5.5-1",
    "slip_event": "TL-5.6-2",
    "slip_velocity": "TL-5.7-1",
    "temperature": "TL-5.8-1",
}

#: 向量字段 → 定义其坐标系维度要求的条款。
_FIELD_DIM_CLAUSE: Dict[str, str] = {
    "contact_centroid": "TL-5.2-3",
    "force_vector": "TL-5.4-2",
    "torque_vector": "TL-5.5-2",
    "slip_velocity": "TL-5.7-2",
}

#: 词表字段 → 定义其闭集的条款（TL-6.2-1 / TL-6.3-1）。
_ENUM_CLAUSE: Dict[str, str] = {
    "contact_region": "TL-6.2-1",
    "texture_class": "TL-6.3-1",
}

#: 字段自身的来源约束条款，比 TL-9.3-2 的通则更具体，命中时优先引用。
#: 规范在 5.1 节专门写了「contact 由人工标注时必须带 annotator_id」——报 9.3-2
#: 虽然不算错（通则确实被违反），但按条款号回溯 5.1-4 的人会看不到它的验证。
_FIELD_PROV_CLAUSE: Dict[Tuple[str, str], str] = {
    ("contact", "human_annotation"): "TL-5.1-4",
}


# ─────────────────────────────────────────────────────────────
# L-A 语法
# ─────────────────────────────────────────────────────────────

#: 帧内允许出现的全部键。用于 TL-10.1-3 的大小写变体检测。
_FRAME_KEYS: FrozenSet[str] = frozenset(
    S.ALL_FIELDS
    + tuple(S.FRAME_KEY.values())
    + ("timestamp_s", "field_provenance", "field_status", "sensor_specific",
       "raw_sensor_values", "instance_id")
)

_FRAME_KEYS_LOWER: Dict[str, str] = {k.lower(): k for k in _FRAME_KEYS}

#: 数据集级键的大小写对照，供 TL-10.1-3 用。与帧级分开是因为两者的键空间
#: 不同——`frames` 只在数据集级出现，`timestamp_s` 只在帧级出现，混在一起
#: 会让 `frames` 在帧内被当成合法键。
_DATASET_KEYS_LOWER: Dict[str, str] = {k.lower(): k for k in S.DATASET_KEYS}


def _check_key_names(ctx: _Ctx, path: str, frame: Dict[str, Any]) -> None:
    """TL-10.1-3：字段名必须逐字符等于规范定义的形式。

    大小写变体是序列化层最常见的错误来源：`Force_Vector` 在 Python 里能读、
    在 JSON 里能存、在下游 `frame["force_vector"]` 里悄悄变成 KeyError 或
    默认值。规范原文说「Case variants are validation errors」，所以这不是
    警告——它必须拦住。

    只报「与某个已知字段仅差大小写」的键。完全不认识的键不报：扩展容器
    （`sensor_specific`）里放什么由生产者决定，不能因为不认识就报错。
    """
    for key in frame:
        if key in _FRAME_KEYS:
            continue
        canonical = _FRAME_KEYS_LOWER.get(key.lower())
        if canonical is not None and canonical != key:
            ctx.err(
                "FIELD_NAME_CASE", "TL-10.1-3", f"{path}.{key}",
                f"字段名 {key!r} 不是规范形式，应为 {canonical!r}",
                value=key,
            )


def _check_syntax(ctx: _Ctx, path: str, frame: Dict[str, Any]) -> None:
    # 键存在（TL-4.4-1）
    for name in S.CORE_FIELDS:
        if name not in frame:
            ctx.err(
                "KEY_MISSING", "TL-4.4-1", f"{path}.{name}",
                f"核心字段 {name} 的键必须存在，即使值为 null",
            )

    for name in S.CORE_FIELDS:
        if name not in frame:
            continue
        value = frame[name]
        if value is None:
            continue
        dtype = S.DATA_TYPES[name]
        fpath = f"{path}.{name}"

        if dtype == "scalar":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                ctx.err("TYPE_MISMATCH", _TYPE_CLAUSE[name], fpath, f"{name} 必须是标量")
                continue
            if not math.isfinite(value):
                ctx.err("NON_FINITE_VALUE", "TL-4.3-1", fpath, f"{name} 不得为 NaN 或 Inf")
                continue
            if name == "normal_force" and value < 0:
                ctx.err(
                    "RANGE_INVALID", "TL-5.3-1", fpath,
                    f"normal_force 不得为负（得到 {value}）",
                )

        elif dtype in ("array2", "array3"):
            want = 2 if dtype == "array2" else 3
            if not isinstance(value, (list, tuple)):
                ctx.err("TYPE_MISMATCH", _TYPE_CLAUSE[name], fpath, f"{name} 必须是数组")
                continue
            if len(value) != want:
                ctx.err(
                    "LENGTH_MISMATCH", _TYPE_CLAUSE[name], fpath,
                    f"{name} 必须是 {want} 维（得到 {len(value)} 维）",
                )
                continue
            bad = [x for x in value if isinstance(x, bool) or not isinstance(x, (int, float))]
            if bad:
                ctx.err("TYPE_MISMATCH", _TYPE_CLAUSE[name], fpath, f"{name} 的分量必须是数值")
                continue
            if not all(math.isfinite(x) for x in value):
                ctx.err("NON_FINITE_VALUE", "TL-4.3-1", fpath, f"{name} 的分量不得为 NaN 或 Inf")

        elif dtype == "tristate_boolean":
            if not isinstance(value, bool):
                ctx.err(
                    "TYPE_MISMATCH", _TYPE_CLAUSE[name], fpath,
                    f"{name} 必须是 true / false / null",
                )

        elif dtype == "string":
            if not isinstance(value, str):
                ctx.err("TYPE_MISMATCH", "TL-4.3-2", fpath, f"{name} 必须是字符串")

    # 闭集枚举（TL-6.2-2、TL-6.3-2）
    if frame.get("contact_region") is not None:
        if frame["contact_region"] not in S.CONTACT_REGION_VALUES:
            ctx.err(
                "ENUM_INVALID", _ENUM_CLAUSE["contact_region"], f"{path}.contact_region",
                f"contact_region 取值不在闭集内：{frame['contact_region']}",
            )
    if frame.get("texture_class") is not None:
        if frame["texture_class"] not in S.TEXTURE_CLASS_VALUES:
            ctx.err(
                "ENUM_INVALID", _ENUM_CLAUSE["texture_class"], f"{path}.texture_class",
                f"texture_class 取值不在闭集内：{frame['texture_class']}",
            )


# ─────────────────────────────────────────────────────────────
# L-B 语义
# ─────────────────────────────────────────────────────────────

def _check_capability_consistency(ctx: _Ctx, path: str, frame: Dict[str, Any]) -> None:
    status = frame.get("field_status") or {}
    caps = ctx.capabilities

    for name in S.CORE_FIELDS:
        if name not in frame:
            continue
        value = frame[name]
        cap = caps.get(name)
        st = status.get(name)
        fpath = f"{path}.{name}"

        # TL-7.3-1：能力为 false 的字段必须为 null。给了值说明生产者
        # 要么能力声明错了，要么值是从别处推出来的——两种都要拦。
        if cap is False and value is not None:
            ctx.err(
                "VALUE_WITHOUT_CAPABILITY", "TL-7.3-1", fpath,
                f"能力声明 {name}=false，但值非 null（{value}）",
            )
            continue

        # TL-7.3-2：有能力、值为 null，却没有状态。null 本身不说明为什么
        # 缺——未采集、无效、未标注是三种不同的缺失，下游要据此决定是否可用。
        if cap is True and value is None and st is None:
            ctx.err(
                "NULL_WITHOUT_STATUS", "TL-7.3-2", fpath,
                f"{name} 声明有能力但值为 null，必须给出 field_status",
            )
            continue

        # TL-7.2-5：声明「未采集/未标注/不适用」却带非 null 值
        if st in S.STATES_REQUIRING_NULL and value is not None:
            ctx.err(
                "STATUS_WITHOUT_NULL", "TL-7.2-5", fpath,
                f"{name} 声明 {st}，但值非 null（{value}）",
            )
            continue

        # TL-4.5-3：显式状态必须与能力声明一致。
        # 能力为 false 意味着「不支持这个量」，写 not_acquired / invalid /
        # not_annotated 意味着「有能力但这次没拿到」——两者不能同时成立。
        # 只有声称「有能力」的状态才与 capability=false 矛盾。unsupported /
        # not_applicable / unknown_for_migration 都不声称能力存在——迁移态尤其
        # 弱于 not_acquired，它说的是「不知道当时是什么状态」（TL-7.2），
        # 与「不支持」可以并存。
        if cap is False and st in S.STATES_CLAIMING_CAPABILITY:
            ctx.err(
                "STATUS_CAPABILITY_CONFLICT", "TL-4.5-3", fpath,
                f"能力声明 {name}=false（不支持），却声明状态 {st}（有能力但未取到）",
            )
            continue

        # TL-7.3-4：有能力却声明 unsupported，同样是矛盾。
        if cap is True and st == "unsupported":
            ctx.err(
                "UNSUPPORTED_WITH_CAPABILITY", "TL-7.3-4", fpath,
                f"能力声明 {name}=true，却把缺失归因为 unsupported",
            )
            continue

        # TL-7.2-4：状态必须是已知集合内的值
        if st is not None and st not in S.ALL_STATES:
            ctx.err(
                "ENUM_INVALID", "TL-7.2-4", fpath,
                f"{name} 的状态 {st} 不是已定义状态",
            )

    # TL-5.2-4：contact 与 contact_centroid 的有无应当一致。
    #
    # 这是 SHOULD 级，所以只报 warning。规范说偏离需有正当理由并记录
    # （TL-4.1），而数据里没有「理由」的存放位置——warning 是唯一的提示通道。
    # 只在有状态声明时才报：contact=true、centroid=null、无状态这个组合
    # 已经由 TL-7.3-2 报过 error，同一个根因不再叠加。
    _ct = frame.get("contact")
    _cen = frame.get("contact_centroid")
    if _ct is True and _cen is None and status.get("contact_centroid") is not None:
        ctx.warn(
            "CENTROID_CONTACT_INCONSISTENT", "TL-5.2-4",
            f"{path}.contact_centroid",
            f"contact 为 true，但 contact_centroid 为 null"
            f"（状态 {status.get('contact_centroid')}）；有接触时建议给出接触位置",
        )
    elif _ct is False and _cen is not None:
        ctx.warn(
            "CENTROID_CONTACT_INCONSISTENT", "TL-5.2-4",
            f"{path}.contact_centroid",
            "contact 为 false，但 contact_centroid 有值；无接触时该位置没有物理含义",
        )

    # TL-5.7-4：slip_event 不是 true 时 slip_velocity 应为 null。
    # 这是 warning 不是 error——传感器可能报告低于自身滑移阈值的残余运动，
    # 那个值是真的，只是不代表发生了滑移。
    if (
        frame.get("slip_velocity") is not None
        and frame.get("slip_event") is not True
    ):
        ctx.warn(
            "SLIP_VELOCITY_WITHOUT_SLIP", "TL-5.7-4", f"{path}.slip_velocity",
            f"slip_event 为 {frame.get('slip_event')!r} 却给出了 slip_velocity；"
            "非滑移帧的速度若低于阈值应置 null，若确为残余运动需在来源中说明",
        )

    # 弃用字段（TL-6.5-1 / TL-6.5-2）
    #
    # 分界线是数据来源而不是字段本身：原生数据出现即 error（TL-6.5-1 的
    # 强制执行），迁移数据出现即 warning（V2 里这些字段合法，迁移工具保留
    # 原值是为了可追溯，报 error 会把每份迁移数据都判为不合规）。
    has_migration = bool(ctx.migration)
    for name in S.DEPRECATED_FIELDS:
        if frame.get(name) is None:
            continue
        if has_migration:
            ctx.warn(
                "DEPRECATED_FIELD_POPULATED", "TL-6.5-2", f"{path}.{name}",
                f"迁移数据保留了弃用字段 {name} 的原值；可读，但下游不应依赖",
            )
        else:
            ctx.err(
                "DEPRECATED_FIELD_POPULATED", "TL-6.5-1", f"{path}.{name}",
                f"弃用字段 {name} 在原生 V3.0 数据中不得有非 null 值",
            )


def _check_provenance(ctx: _Ctx, path: str, frame: Dict[str, Any]) -> None:
    prov = frame.get("field_provenance") or {}

    for name in S.CORE_FIELDS:
        value = frame.get(name)
        if value is None:
            continue
        fpath = f"{path}.{name}"
        entry = prov.get(name)

        # TL-9.2-1：非 null 值必须有来源
        if not isinstance(entry, dict):
            ctx.err(
                "VALUE_WITHOUT_PROVENANCE", "TL-9.2-1", fpath,
                f"{name} 非 null，但没有 field_provenance 条目",
            )
            continue

        ptype = entry.get("type")
        if ptype is None:
            ctx.err(
                "PROVENANCE_INCOMPLETE", "TL-9.3-2", fpath,
                f"{name} 的来源条目缺少 type",
            )
            continue

        # TL-9.3-1：五类之外没有第六类。`proxy` / `estimated` 之类自封类型
        # 不能把代理量合法化——这是本规范最想拦住的一种绕过方式。
        if ptype not in S.PROVENANCE_TYPES:
            ctx.err(
                "PROVENANCE_TYPE_FORBIDDEN", "TL-9.3-1", fpath,
                f"{name} 的来源类型 {ptype!r} 未定义；只允许 {sorted(S.PROVENANCE_TYPES)}",
            )
            continue

        # TL-5.6-1：滑移不是可直接观测的物理量，任何滑移判定都是推断。
        # 记成直接测量会隐藏判定方法，使结果不可复现。
        if name == "slip_event" and ptype == "direct_measurement":
            ctx.err(
                "PROVENANCE_TYPE_FORBIDDEN", "TL-5.6-1", fpath,
                "slip_event 不得声明 direct_measurement；滑移判定必须是推断并给出 method_id",
            )
            continue

        # TL-9.3-2：该类来源的必填记录项。同一根因合并成一条，列出全部缺失项，
        # 避免一个 provenance 条目刷出多条同码 finding 掩盖真正的考点。
        missing = [k for k in S.PROVENANCE_REQUIRED_KEYS.get(ptype, ()) if not entry.get(k)]
        if missing:
            clause = _FIELD_PROV_CLAUSE.get((name, ptype), "TL-9.3-2")
            ctx.err(
                "PROVENANCE_INCOMPLETE", clause, fpath,
                f"{name} 的来源 {ptype} 缺少必填项：{', '.join(missing)}",
            )


def _check_required_context(ctx: _Ctx, path: str, frame: Dict[str, Any]) -> None:
    prov = frame.get("field_provenance") or {}

    # TL-5.2-5：contact_centroid 非 null 必须声明计算方法
    if frame.get("contact_centroid") is not None:
        entry = prov.get("contact_centroid") or {}
        method = entry.get("centroid_method")
        if method is None:
            ctx.err(
                "REQUIRED_CONTEXT_MISSING", "TL-5.2-5", f"{path}.contact_centroid",
                "contact_centroid 非 null 时必须声明 centroid_method",
            )
        elif method not in S.CENTROID_METHOD_VALUES:
            ctx.err(
                "ENUM_INVALID", "TL-5.2-5", f"{path}.contact_centroid",
                f"centroid_method 取值 {method} 不在允许集内",
            )
        elif method == "pressure_weighted" and not entry.get("weighting_quantity"):
            ctx.err(
                "REQUIRED_CONTEXT_MISSING", "TL-5.2-6", f"{path}.contact_centroid",
                "pressure_weighted 必须声明 weighting_quantity",
            )

    # TL-5.5-3：torque_vector 非 null 必须声明参考点。
    # 力矩随参考点变化，没有参考点的值无法被解释，也无法与其他数据比较。
    if frame.get("torque_vector") is not None:
        entry = prov.get("torque_vector") or {}
        rp = entry.get("reference_point")
        if not rp:
            ctx.err(
                "TORQUE_REFERENCE_MISSING", "TL-5.5-3", f"{path}.torque_vector",
                "torque_vector 非 null 时必须声明 reference_point",
            )
        elif not isinstance(rp, (list, tuple)) or len(rp) != 3:
            # TL-5.5-5：参考点是三维坐标系中的位置，必须 3 个分量。
            ctx.err(
                "LENGTH_MISMATCH", "TL-5.5-5", f"{path}.torque_vector.reference_point",
                f"reference_point 必须是三维位置（得到 {len(rp) if isinstance(rp, (list, tuple)) else type(rp).__name__}）",
            )
        else:
            bad = [
                x for x in rp
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            ]
            if bad:
                ctx.err(
                    "TYPE_MISMATCH", "TL-5.5-5", f"{path}.torque_vector.reference_point",
                    "reference_point 的分量必须是有限数值",
                )

    # TL-5.8-2：temperature 非 null 必须声明测量对象
    if frame.get("temperature") is not None:
        entry = prov.get("temperature") or {}
        target = entry.get("measurement_target")
        if target is None:
            ctx.err(
                "REQUIRED_CONTEXT_MISSING", "TL-5.8-2", f"{path}.temperature",
                "temperature 非 null 时必须声明 measurement_target",
            )
        elif target not in S.MEASUREMENT_TARGET_VALUES:
            ctx.err(
                "ENUM_INVALID", "TL-5.8-2", f"{path}.temperature",
                f"measurement_target 取值 {target} 不在允许集内",
            )


def _nominal_normal(ctx: _Ctx, frame: Dict[str, Any]) -> Optional[List[float]]:
    """从 `force_vector` 所在坐标系取名义法向单位向量。

    `normal_reference = sensor_surface_nominal` 的含义是「法向 = 坐标系声明的
    `nominal_surface_normal` 轴」。这个向量写在 `frames_of_reference` 里，是
    生产者声明过的，不是猜的——所以这种法向同样可以复算 `|F·n|`。
    """
    fid = frame.get("force_vector_frame")
    spec = ctx.frames.get(fid) if fid else None
    if not spec or spec.get("dimension") != 3:
        return None
    axes = spec.get("axes") or {}
    for i, k in enumerate(("x", "y", "z")):
        if axes.get(k) == "nominal_surface_normal":
            v = [0.0, 0.0, 0.0]
            v[i] = 1.0
            return v
    return None


def _normal_force_is_norm(
    ctx: _Ctx, fpath: str, frame: Dict[str, Any],
    entry: Dict[str, Any], ref: str, value: float,
) -> bool:
    """TL-5.3-2：`normal_force` 等于合力模长且 F 有切向分量 → 报错。返回是否已报。

    这条判据独立于 `n` 的具体数值，只需要知道法向方向。规范原文说「即使结果在
    物理上合理，也是校验错误」——用 `‖F‖` 算出来的数看起来永远合理，只有和
    切向分量对照才能识别：只要 F 有切向分量，`|F·n|` 就严格小于 `‖F‖`。
    """
    F = frame.get("force_vector")
    if not F or not isinstance(F, (list, tuple)) or len(F) != 3:
        return False

    if ref == "contact_interface_local":
        n_hat = entry.get("normal_vector")
    elif ref == "sensor_surface_nominal":
        n_hat = _nominal_normal(ctx, frame)
    else:
        # measurement_axis：轴由生产者声明，数据里没有第二个来源可对照。
        return False

    if not isinstance(n_hat, (list, tuple)) or len(n_hat) != 3:
        return False
    try:
        F = [float(x) for x in F]
        n_hat = [float(x) for x in n_hat]
    except (TypeError, ValueError):
        return False

    n_norm = math.sqrt(sum(x * x for x in n_hat))
    if n_norm == 0.0:
        return False
    # 先归一：未归一的 n 会让切向分量算错，而它自己的错误由 TL-5.3-4 单独报。
    n_hat = [x / n_norm for x in n_hat]

    norm_F = math.sqrt(sum(x * x for x in F))
    if abs(value - norm_F) > S.NORMAL_FORCE_TOLERANCE:
        return False

    dot = sum(a * b for a, b in zip(F, n_hat))
    t_vec = [a - dot * b for a, b in zip(F, n_hat)]
    t_norm = math.sqrt(sum(x * x for x in t_vec))
    if t_norm <= S.NORMAL_FORCE_TOLERANCE:
        # F 完全沿法向，‖F‖ = |F·n| 本来就成立，不是这个错误。
        return False

    ctx.err(
        "NORMAL_FORCE_MISMATCH", "TL-5.3-2", fpath,
        f"normal_force = {value} 等于合力模长 ‖F‖ = {norm_F:.4f}，"
        f"但 F 有切向分量（‖F_t‖ = {t_norm:.4f}）；"
        f"沿 {ref} 的投影应为 |F·n| = {abs(dot):.4f}",
    )
    return True


def _verify_projection(
    ctx: _Ctx, fpath: str, frame: Dict[str, Any],
    n: Optional[List[float]], value: float,
) -> None:
    """TL-5.3-3：`normal_force` 必须等于 `|F·n|`；缺 F 或 n 时降级 `not_verified`。"""
    if n is None:
        ctx.nv(
            "NORMAL_FORCE_MISMATCH", "TL-5.3-3", fpath,
            "无法确定法向单位向量，无法复算 |F·n|",
        )
        return
    F = frame.get("force_vector")
    if not F:
        ctx.nv(
            "NORMAL_FORCE_MISMATCH", "TL-5.3-3", fpath,
            "缺少 force_vector，无法复算 |F·n|",
        )
        return
    try:
        finite = all(math.isfinite(float(x)) for x in F)
    except (TypeError, ValueError):
        finite = False
    if not finite:
        ctx.nv(
            "NORMAL_FORCE_MISMATCH", "TL-5.3-3", fpath,
            "force_vector 分量不可用（非数值或非有限），无法复算 |F·n|",
        )
        return
    expected = abs(sum(float(a) * float(b) for a, b in zip(F, n)))
    if abs(value - expected) > S.NORMAL_FORCE_TOLERANCE:
        ctx.err(
            "NORMAL_FORCE_MISMATCH", "TL-5.3-3", fpath,
            f"normal_force = {value}，但 |F·n| = {expected:.4f}",
        )


def _check_normal_force_semantics(ctx: _Ctx, path: str, frame: Dict[str, Any]) -> None:
    value = frame.get("normal_force")
    if value is None:
        return
    # NaN / Inf 已经由 TL-4.3-1 报过。拿它算投影只会产生一条噪音 finding，
    # 而且 NaN 参与任何比较都返回 False，会把「无法比较」伪装成「不一致」。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if not math.isfinite(value):
        return
    prov = frame.get("field_provenance") or {}
    entry = prov.get("normal_force") or {}
    fpath = f"{path}.normal_force"

    # 来源条目整体缺失时，VALUE_WITHOUT_PROVENANCE 已报出根因。
    # 此处不再追加 REQUIRED_CONTEXT_MISSING —— 一个根因只报一次。
    if not entry:
        return

    ptype = entry.get("type")
    ref = entry.get("normal_reference")

    # TL-5.3-8：normal_reference 若声明，必须是三值之一。
    # 注意这是**值域**约束，不是必填——规范原文是「The declared
    # normal_reference MUST be one of ...」，约束的是声明的值，不是声明本身。
    if ref is not None and ref not in S.NORMAL_REFERENCE_VALUES:
        ctx.err(
            "ENUM_INVALID", "TL-5.3-8", fpath,
            f"normal_reference 取值 {ref} 不在允许集内",
        )
        return

    # TL-5.3-5：直接测量的法向力必须声明测量轴。
    # 单轴传感器的敏感轴与接触界面法向有夹角时，测到的是轴向分量而不是法向力；
    # 不声明轴，这个值无法归入任何一类。
    if ptype == "direct_measurement":
        if not entry.get("measurement_axis"):
            ctx.err(
                "REQUIRED_CONTEXT_MISSING", "TL-5.3-5", fpath,
                "normal_force 是直接测量，必须在来源中声明 measurement_axis",
            )
        return

    # TL-5.3-10：拿 measurement_axis 当法向参考时，必须声明角度容差。
    # 容差本身是生产者的判断（轴与法向差多少还算「对齐」），校验器验不了
    # 这个数对不对，但可以验它有没有声明。没声明时，「轴向力」和「法向力」
    # 在数据里长得一模一样，下游无从判断这个值属于哪一类。
    if ref == "measurement_axis" and entry.get("angular_tolerance_deg") is None:
        ctx.err(
            "REQUIRED_CONTEXT_MISSING", "TL-5.3-10", fpath,
            "normal_reference 为 measurement_axis，必须声明 angular_tolerance_deg",
        )

    # TL-5.3-3 只约束「由 force_vector 派生」这一种来源。模型推断、人工标注、
    # 外部提供都不是从 F 算出来的，数据里没有第二个来源可对照。
    if ptype != "derived_measurement":
        return

    # TL-5.3-3：派生却无法确定法向方向。此时按 TL-5.3-6 应当把值置 null 并
    # 给出可用性状态，而不是猜一个方向——所以这里报「缺 n」而不是「缺 ref」。
    if ref is None:
        ctx.err(
            "REQUIRED_CONTEXT_MISSING", "TL-5.3-3", fpath,
            "normal_force 由 force_vector 派生，但未声明 normal_reference，"
            "无法确定投影方向；按 TL-5.3-6 应置 null",
        )
        return

    # TL-5.3-2：不得把合力模长当法向力。放在投影复算之前，因为命中后
    # |F·n| 复算会报同一个根因的第二条 finding。
    if _normal_force_is_norm(ctx, fpath, frame, entry, ref, value):
        return

    if ref == "sensor_surface_nominal":
        _verify_projection(ctx, fpath, frame, _nominal_normal(ctx, frame), value)
        return

    if ref != "contact_interface_local":
        # measurement_axis：轴由生产者声明，数据里没有第二个来源可对照。
        return

    # TL-5.3-9：声明局部法向必须提供 n
    n = entry.get("normal_vector")
    if not n:
        ctx.err(
            "NORMAL_VECTOR_MISSING", "TL-5.3-9", fpath,
            "声明 contact_interface_local 但未提供 normal_vector",
        )
        return

    if not isinstance(n, (list, tuple)) or len(n) != 3:
        ctx.err(
            "LENGTH_MISMATCH", "TL-5.3-9", fpath,
            "normal_vector 必须是三维向量",
        )
        return

    # TL-5.3-4：n 必须是单位向量
    norm = math.sqrt(sum(float(x) * float(x) for x in n))
    if abs(norm - 1.0) > S.NORMAL_VECTOR_TOLERANCE:
        ctx.err(
            "NORMAL_VECTOR_NOT_UNIT", "TL-5.3-4", fpath,
            f"normal_vector 不是单位向量：‖n‖ = {norm:.6f}",
        )
        return

    # TL-5.3-3：n 与 F 必须同系
    n_frame = entry.get("normal_vector_frame")
    f_frame = frame.get("force_vector_frame")
    if n_frame and f_frame and n_frame != f_frame:
        ctx.err(
            "NORMAL_VECTOR_FRAME_MISMATCH", "TL-5.3-3", fpath,
            f"normal_vector 在 {n_frame}，force_vector 在 {f_frame}",
        )
        return

    _verify_projection(ctx, fpath, frame, list(n), value)


def _check_migration(ctx: _Ctx, path: str, frame: Dict[str, Any]) -> None:
    status = frame.get("field_status") or {}

    # TL-7.2-1 / TL-7.2-3：迁移态必须伴随迁移审计
    #
    # 报 7.2-3 而不是 7.2-1：7.2-1 说「生产者不得产出迁移态」，那是生产者义务；
    # 7.2-3 说「校验器必须拒绝没有审计的迁移态」，这才是校验器正在做的事。
    has_mig = any(v == "unknown_for_migration" for v in status.values())
    if has_mig and not ctx.migration:
        ctx.err(
            "MIGRATION_STATE_IN_NATIVE", "TL-7.2-3", path,
            "出现 unknown_for_migration，但数据集没有 migration 审计块",
        )
    elif has_mig:
        # TL-7.2-2：有迁移态就必须有源规范版本。没有源版本的迁移态无法追溯
        # 「这个值是从哪个旧字段来的」，等于把不确定性留在了数据里。
        if not ctx.migration.get("source_spec_version"):
            ctx.err(
                "MIGRATION_AUDIT_MISSING", "TL-7.2-2", path,
                "出现 unknown_for_migration，但迁移审计缺少 source_spec_version",
            )

    # TL-11.3-1：V2 源数据不得推断出 V3 新字段的值
    if ctx.migration:
        src = str(ctx.migration.get("source_spec_version") or "")
        if src.startswith("2.") and frame.get("normal_force") is not None:
            audit = ctx.migration.get("audit") or []
            documented = any(
                isinstance(a, dict) and a.get("field") == "normal_force" for a in audit
            )
            if not documented:
                ctx.err(
                    "MIGRATION_INFERRED", "TL-11.3-1", f"{path}.normal_force",
                    "V2 数据没有 normal_force，此值只能是推断的，且审计未记录",
                )


# ─────────────────────────────────────────────────────────────
# L-C 坐标
# ─────────────────────────────────────────────────────────────

def _check_transforms(ctx: _Ctx) -> None:
    """数据集级的 `transforms` 数组（TL-8.4-4 / 8.4-5）。

    只检查声明完整性，不执行变换——TL-8.4-1 明确说 TLabel 不实现变换引擎。
    但「记录了什么」是数据格式的一部分：一个没有方向的变换，接收方无法判断
    它映射 A→B 还是 B→A，而两种取法互为逆变换，结果差一个旋转。
    """
    transforms = ctx.dataset.get("transforms")
    if transforms is None:
        return
    if not isinstance(transforms, list):
        ctx.err(
            "TYPE_MISMATCH", "TL-8.4-4", "transforms",
            f"transforms 必须是数组（得到 {type(transforms).__name__}）",
        )
        return

    for i, t in enumerate(transforms):
        tpath = f"transforms[{i}]"
        if not isinstance(t, dict):
            ctx.err(
                "TYPE_MISMATCH", "TL-8.4-4", tpath,
                "变换条目必须是对象",
            )
            continue

        # TL-8.4-5：方向缺失时这个变换不可用，而且不能被猜。
        # 单独报这条，因为它的后果比「缺个单位」严重得多。
        lacks_direction = not t.get("source_frame") or not t.get("target_frame")
        if lacks_direction:
            ctx.err(
                "KEY_MISSING", "TL-8.4-5", tpath,
                "变换未声明方向（缺 source_frame 或 target_frame）；"
                "接收方不得猜测它映射 A→B 还是 B→A",
            )
            # 方向缺失已经说明这个变换不可用，再逐项列缺什么没有增量信息。
            continue

        missing = [k for k in S.TRANSFORM_REQUIRED_KEYS if t.get(k) in (None, "", [])]
        if missing:
            ctx.err(
                "KEY_MISSING", "TL-8.4-4", tpath,
                f"变换缺少必填项：{missing}",
            )
            continue

        if t["rotation_representation"] not in S.ROTATION_REPRESENTATIONS:
            ctx.err(
                "ENUM_INVALID", "TL-8.4-4", f"{tpath}.rotation_representation",
                f"旋转表示 {t['rotation_representation']!r} 不在允许集内",
            )
        if t["calibration"] not in S.CALIBRATION_KINDS:
            ctx.err(
                "ENUM_INVALID", "TL-8.4-4", f"{tpath}.calibration",
                f"calibration 必须是 calibrated 或 estimated（得到 {t['calibration']!r}）",
            )
        if t["translation_unit"] not in S.UNIT_CODES:
            ctx.err(
                "UNIT_UNKNOWN", "TL-8.4-4", f"{tpath}.translation_unit",
                f"平移单位 {t['translation_unit']!r} 不在附录 A 内",
            )
        # 变换引用的坐标系必须存在，否则这个变换无处落地。
        for side in ("source_frame", "target_frame"):
            fid = t[side]
            if fid not in ctx.frames:
                ctx.err(
                    "FRAME_NOT_DECLARED", "TL-8.1-1", f"{tpath}.{side}",
                    f"变换引用的坐标系 {fid} 未在 frames_of_reference 中声明",
                )


def _check_frames(ctx: _Ctx, path: str, frame: Dict[str, Any]) -> None:
    # 坐标系自身合法性（TL-8.1-2 / 8.1-3 / 8.2-1 / 8.2-2 / 8.3-1）
    for fid, spec in ctx.frames.items():
        fpath = f"frames_of_reference[{fid}]"

        # TL-8.1-2：声明必须回答六项。缺任何一项，下游就得猜——猜出来的
        # 坐标系比没有坐标系更危险，因为它看起来是确定的。
        missing = [
            k for k in ("dimension", "frame_type", "handedness", "axes", "reference_object")
            if spec.get(k) in (None, "", {})
        ]
        if missing:
            ctx.err(
                "KEY_MISSING", "TL-8.1-2", fpath,
                f"坐标系声明缺少：{missing}",
            )

        dim = spec.get("dimension")
        axes = spec.get("axes") or {}
        if dim not in (2, 3):
            # TL-8.1-3 明确写了 dimension MUST be 2 or 3。
            ctx.err(
                "FRAME_DIMENSION_MISMATCH", "TL-8.1-3", fpath,
                f"dimension 必须是 2 或 3（得到 {dim}）",
            )
            continue
        required = {"x", "y"} if dim == 2 else {"x", "y", "z"}
        if set(axes.keys()) != required:
            # 轴集合同样出自 TL-8.1-3：「MUST declare exactly the axes ...」。
            ctx.err(
                "FRAME_AXES_MISMATCH", "TL-8.1-3", fpath,
                f"dimension={dim} 要求轴 {sorted(required)}，实际 {sorted(axes.keys())}",
            )
        else:
            # TL-8.3-1：每个轴必须说明含义。空字符串是「声明了但没说」，
            # 与缺键不同——缺键由 8.1-2 抓，空值由这条抓。
            blank = [k for k, val in axes.items() if not val]
            if blank:
                ctx.err(
                    "FRAME_AXES_MISMATCH", "TL-8.3-1", fpath,
                    f"轴 {blank} 未说明含义",
                )
        if spec.get("frame_type") not in S.FRAME_TYPE_VALUES:
            ctx.err(
                "ENUM_INVALID", "TL-8.2-1", fpath,
                f"frame_type 取值 {spec.get('frame_type')} 不在允许集内",
            )
        if spec.get("handedness") not in S.HANDEDNESS_VALUES:
            ctx.err(
                "ENUM_INVALID", "TL-8.2-2", fpath,
                f"handedness 取值 {spec.get('handedness')} 不在允许集内",
            )

    # 字段引用的坐标系（TL-8.1-1、TL-8.1-3）
    for name, want_dim in S.FIELD_DIMENSION.items():
        if frame.get(name) is None:
            continue
        key = S.FRAME_KEY[name]
        fid = frame.get(key)
        fpath = f"{path}.{name}"
        if not fid:
            ctx.err(
                "FRAME_NOT_DECLARED", "TL-8.1-1", fpath,
                f"{name} 必须引用一个坐标系（{key} 缺失）",
            )
            continue
        spec = ctx.frames.get(fid)
        if spec is None:
            ctx.err(
                "FRAME_NOT_DECLARED", "TL-8.1-1", fpath,
                f"{name} 引用的坐标系 {fid} 未在 frames_of_reference 中声明",
            )
            continue
        if spec.get("dimension") != want_dim:
            # 每个向量字段都有自己的维度条款，比 TL-8.1-3 的通用条款更具体。
            clause = _FIELD_DIM_CLAUSE.get(name, "TL-8.1-3")
            ctx.err(
                "FRAME_DIMENSION_MISMATCH", clause, fpath,
                f"{name} 要求 {want_dim} 维坐标系，{fid} 是 {spec.get('dimension')} 维",
            )
            continue
        # TL-5.7-2：slip_velocity 的轴必须在接触切平面内
        if name == "slip_velocity":
            axes = spec.get("axes") or {}
            if any(str(v) == "nominal_surface_normal" for v in axes.values()):
                ctx.err(
                    "FRAME_OUT_OF_PLANE", "TL-5.7-2", fpath,
                    f"slip_velocity 的坐标系 {fid} 含有出平面轴",
                )


# ─────────────────────────────────────────────────────────────
# L-D 扩展
# ─────────────────────────────────────────────────────────────

def _check_dataset_key_names(ctx: _Ctx) -> None:
    """TL-10.1-3：数据集级键名同样必须逐字符等于规范形式。

    帧级已经查过，数据集级不能漏：`Time_Base` 写错时，`dataset.get("time_base")`
    返回 None，而 None 在本规范里是**合法值**——校验器会报「缺少 time_base」
    而不是「键名错了」，生产者会去补一个 time_base，同时把错键留在数据里。
    """
    for key in ctx.dataset:
        if key in S.DATASET_KEYS:
            continue
        canonical = _DATASET_KEYS_LOWER.get(key.lower())
        if canonical is not None and canonical != key:
            ctx.err(
                "FIELD_NAME_CASE", "TL-10.1-3", key,
                f"数据集级键名 {key!r} 不是规范形式，应为 {canonical!r}",
                value=key,
            )


def _check_extensions(ctx: _Ctx, path: str, frame: Dict[str, Any], level: str) -> None:
    if level != "extended":
        return
    ss = frame.get("sensor_specific")
    if not ss:
        return
    required = ("value", "unit", "definition", "method_id", "namespace")
    for key, entry in ss.items():
        if not isinstance(entry, dict):
            ctx.err(
                "SENSOR_SPECIFIC_UNSTRUCTURED", "TL-9.5-3", f"{path}.sensor_specific.{key}",
                f"Extended 等级要求 sensor_specific 条目为结构化对象（{key} 是裸值）",
            )
            continue
        missing = [k for k in required if k not in entry]
        if missing:
            # 缺键走 TL-9.5-2（结构化条目必须含这五项）；裸值才走 TL-9.5-3
            # （V2 兼容允许非结构化，Extended 不允许）。两者根因不同：
            # 一个是「写了但没写全」，一个是「根本没写结构化」。
            ctx.err(
                "SENSOR_SPECIFIC_UNSTRUCTURED", "TL-9.5-2", f"{path}.sensor_specific.{key}",
                f"sensor_specific.{key} 缺少必填键：{missing}",
            )
        # TL-4.2-1：自报的单位也必须是 UCUM 码。这条挡住了「把像素写成长度」
        # 最常见的入口——附录 A 的「常见错误写法」列就是为了这件事。
        unit = entry.get("unit")
        if unit is not None and unit not in S.UNIT_CODES:
            ctx.err(
                "UNIT_UNKNOWN", "TL-4.2-1", f"{path}.sensor_specific.{key}.unit",
                f"单位 {unit!r} 不在附录 A 的允许集内；只允许 {sorted(S.UNIT_CODES)}",
            )


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

def validate(dataset: Dict[str, Any], level: str = "core") -> Report:
    """
    校验一份 V3.0 数据集。

    参数：
        dataset: 含 `capabilities`、`frames_of_reference`、`frames` 的字典。
        level:   "structural" | "core" | "extended"（TL-10.2）

    返回：
        Report。`ok` 为 True 表示无 error（warning 与 not_verified 不算失败）。
    """
    report = Report(level=level)
    report.not_performed = list(_SKIPPED_AT.get(level, []))
    ctx = _Ctx(dataset, report)

    # TL-10.1-3：键名检查在所有级别都做——它属于 L-A 语法层，
    # 与 level 无关。放在最前面是因为键名错会让后面所有检查读到 None。
    _check_dataset_key_names(ctx)

    frames = dataset.get("frames") or []
    report.frames_checked = len(frames)

    # 数据集级：时间基准（TL-9.7-2 / TL-9.7-5）
    tb = dataset.get("time_base")
    if tb is None:
        # 没有 time_base，timestamp_s 的原点未知——同一个数可能是「数据集
        # 起点后 0.033 秒」，也可能是 Unix 时间戳的低位。下游无法对齐。
        ctx.err(
            "KEY_MISSING", "TL-9.7-2", "time_base",
            "缺少数据集级 time_base，timestamp_s 的原点无法确定",
        )
    elif tb not in S.TIME_BASE_VALUES:
        ctx.err(
            "ENUM_INVALID", "TL-9.7-2", "time_base",
            f"time_base 取值 {tb} 不在允许集内",
        )
    elif tb == "device_clock":
        # TL-9.7-5：用设备时钟就必须说明是哪台设备的时钟。两台设备的
        # device_clock 之间没有可比性，不声明身份就无法判断能否合并。
        dev = dataset.get("provenance") or {}
        if not dev.get("clock_identity"):
            ctx.err(
                "REQUIRED_CONTEXT_MISSING", "TL-9.7-5", "time_base",
                "time_base=device_clock，但设备级 provenance 未声明 clock_identity",
            )

    # 数据集级：每个核心字段都必须声明能力（TL-7.1-1）。
    # 缺键不能当 false 处理——「不知道有没有这个能力」和「确定没有」是两件事。
    for name in S.CORE_FIELDS:
        if name not in ctx.capabilities:
            ctx.err(
                "KEY_MISSING", "TL-7.1-1", f"capabilities.{name}",
                f"capabilities 未声明核心字段 {name} 的能力",
            )

    # 数据集级：不得用数据集级元数据改变字段单位（TL-4.2-2）。
    # V2 的 `contact_centroid` 允许「像素或 mm」，下游每次都要问「这个 310 是什么」；
    # 数据集级单位表会把同样的歧义重新引进来。
    for key in ("units", "unit_overrides", "field_units"):
        if key in dataset:
            ctx.err(
                "DATASET_UNIT_OVERRIDE", "TL-4.2-2", key,
                f"数据集级键 {key} 试图覆盖字段单位；单位由字段定义固定",
            )

    # TL-9.6-1：多实例数据集必须声明 instances，且每帧带 instance_id。
    # 判据用「声明了多个实例」和「帧声称自己是某实例」两个方向分别查：
    # 只查前者会漏掉「单传感器数据却给帧打了 instance_id」——那不是多实例，
    # 是把一个传感器拆成了两个身份，下游按实例聚合时会得到两个半条数据。
    instances = dataset.get("instances")
    if isinstance(instances, list) and instances:
        declared_ids = {
            inst.get("instance_id")
            for inst in instances
            if isinstance(inst, dict) and inst.get("instance_id")
        }
        multi = len(instances) > 1
        # TL-9.6-4：实例声明的 time_base 覆盖。只有声明过的值才可用——
        # 否则「实例独立时钟」这个场景没法表达，而不检查就等于允许任何值，
        # 那和没有 time_base 是一样的。
        inst_tb = {
            inst.get("instance_id"): inst.get("time_base")
            for inst in instances
            if isinstance(inst, dict) and inst.get("instance_id") and inst.get("time_base")
        }
    else:
        declared_ids = set()
        multi = False
        inst_tb = {}

    # TL-9.7-3 / TL-9.7-4 都是「同一实例内」的约束，所以按实例分组保存状态。
    # 用全局单变量会让两个实例的时间轴互相比较，第二个实例的 t=0 会被判成
    # 「比第一个实例的末尾递减」。
    prev_ts_by_inst: Dict[str, float] = {}
    tb_by_inst: Dict[str, str] = {}

    # 数据集级：transforms 只检查一次，不随帧重复
    _check_transforms(ctx)

    for i, frame in enumerate(frames):
        path = f"frames[{i}]"
        _check_key_names(ctx, path, frame)
        _check_syntax(ctx, path, frame)

        if level in ("core", "extended"):
            _check_capability_consistency(ctx, path, frame)
            _check_provenance(ctx, path, frame)
            _check_required_context(ctx, path, frame)
            _check_normal_force_semantics(ctx, path, frame)
            _check_migration(ctx, path, frame)

        if level in ("core", "extended"):
            _check_frames(ctx, path, frame)

        _check_extensions(ctx, path, frame, level)

        # TL-9.6-1：instance_id 的存在性要与 instances 声明对齐
        iid = frame.get("instance_id")
        if multi and not iid:
            ctx.err(
                "KEY_MISSING", "TL-9.6-1", f"{path}.instance_id",
                "数据集声明了多个实例，但这一帧没有 instance_id",
            )
        elif iid and not declared_ids:
            ctx.err(
                "INSTANCE_NOT_DECLARED", "TL-9.6-1", f"{path}.instance_id",
                f"帧声明 instance_id={iid!r}，但数据集没有 instances 数组",
                value=iid,
            )
        elif iid and iid not in declared_ids:
            ctx.err(
                "INSTANCE_NOT_DECLARED", "TL-9.6-1", f"{path}.instance_id",
                f"instance_id={iid!r} 未在 instances 中声明",
                value=iid,
            )
        inst_key = iid or "__single__"

        # TL-9.7-4：同一实例的帧之间不得改变 time_base。
        # 帧级值优先，缺则继承数据集级——比较的是「这一帧实际用的时间基准」，
        # 不是「这一帧有没有写这个键」，否则省略写法的合法数据会被误报。
        frame_tb = frame.get("time_base") or tb
        if frame_tb is not None:
            seen = tb_by_inst.get(inst_key)
            if seen is None:
                tb_by_inst[inst_key] = frame_tb
            elif seen != frame_tb:
                ctx.err(
                    "TIME_BASE_INCONSISTENT", "TL-9.7-4", f"{path}.time_base",
                    f"实例 {inst_key!r} 的 time_base 从 {seen!r} 变成 {frame_tb!r}；"
                    "同一实例的时间轴必须共用一个原点",
                    value=frame_tb,
                )

            # TL-9.6-4：实例可以用数据集级的值，也可以用它自己声明的覆盖值；
            # 用第三个值即为错误。不查这一条，`time_base` 就退化成一个随便填
            # 的字符串——下游拿两个实例的时间戳做对齐时，原点不同却看不出来。
            allowed = {tb, inst_tb.get(inst_key)} - {None}
            if allowed and frame_tb not in allowed:
                ctx.err(
                    "TIME_BASE_INCONSISTENT", "TL-9.6-4", f"{path}.time_base",
                    f"实例 {inst_key!r} 用了 time_base={frame_tb!r}，"
                    f"但数据集级是 {tb!r}，该实例也未声明覆盖",
                    value=frame_tb,
                )

        # TL-9.7-3：时间单调不减
        ts = frame.get("timestamp_s")
        # TL-9.7-1：timestamp_s 必须是有限数（秒）。
        # 字符串 "0.033" 和 NaN 都会让后面的单调性比较静默失效——
        # 字符串比较是字典序，"10" < "9"，时间轴会乱序而不报错。
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            ctx.err(
                "TYPE_MISMATCH", "TL-9.7-1", f"{path}.timestamp_s",
                f"timestamp_s 必须是数值（得到 {type(ts).__name__}）",
            )
            ts = None
        elif not math.isfinite(ts):
            ctx.err(
                "NON_FINITE_VALUE", "TL-9.7-1", f"{path}.timestamp_s",
                "timestamp_s 必须是有限数",
            )
            ts = None
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            prev_ts = prev_ts_by_inst.get(inst_key)
            if prev_ts is not None and ts < prev_ts:
                ctx.err(
                    "RANGE_INVALID", "TL-9.7-3", f"{path}.timestamp_s",
                    f"时间戳递减：{ts} < {prev_ts}",
                    value=ts,
                )
            prev_ts_by_inst[inst_key] = ts

    return report


def validate_frame(
    frame: Dict[str, Any],
    dataset: Dict[str, Any],
    level: str = "core",
) -> Report:
    """校验单帧。dataset 提供 capabilities / frames_of_reference / migration。"""
    ds = dict(dataset)
    ds["frames"] = [frame]
    return validate(ds, level=level)
