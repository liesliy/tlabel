"""
Primitive + Event 标注校验与批量工具

v0.18 新增：
- validate_annotations(): 校验 primitive/event 标注一致性
- annotate_from_taxonomy(): 批量应用 taxonomy 规则预标注
- annotate_events_from_data(): 从数据模式自动检测事件
- clear_annotations(): 清除标注
"""

from typing import List, Dict, Optional, Tuple


def validate_annotations(data) -> Dict:
    """校验 primitive/event 标注与数据的一致性

    检查项:
    1. Primitive 名称是否有效（在 taxonomy 或 PRIMITIVE_PRESETS 中）
    2. Primitive/Event 帧范围是否在数据帧数内
    3. Primitive 帧区间 start <= end
    4. Event 类型是否在 EVENT_PRESETS 中
    5. 点事件 start_frame == end_frame == frame_idx
    6. confidence 在 [0, 1] 范围内
    7. source 为 'manual' 或 'ai_predicted'

    Args:
        data: TLabelData 实例

    Returns:
        dict with keys:
            - valid (bool): 是否全部通过
            - errors (list[str]): 错误列表
            - warnings (list[str]): 警告列表
            - stats (dict): 统计摘要
    """
    from tlabel.core.primitive import PRIMITIVE_PRESETS, _custom_registry
    from tlabel.core.events import EVENT_PRESETS

    errors = []
    warnings = []
    num_frames = data.num_frames

    # --- Primitive 校验 ---
    valid_primitives = set(PRIMITIVE_PRESETS) | _custom_registry

    for i, ann in enumerate(data.primitive_annotations):
        prefix = f"primitive[{i}]"

        # 1. 名称有效性
        if ann.primitive_name not in valid_primitives:
            errors.append(f"{prefix}: unknown primitive '{ann.primitive_name}'")

        # 2. 帧范围
        if ann.start_frame < 0:
            errors.append(f"{prefix}: start_frame={ann.start_frame} < 0")
        if ann.end_frame < ann.start_frame:
            errors.append(f"{prefix}: end_frame={ann.end_frame} < start_frame={ann.start_frame}")
        if ann.end_frame >= num_frames:
            errors.append(f"{prefix}: end_frame={ann.end_frame} >= num_frames={num_frames}")

        # 3. confidence
        if not (0.0 <= ann.confidence <= 1.0):
            errors.append(f"{prefix}: confidence={ann.confidence} out of [0,1]")

        # 4. source
        if ann.source not in ("manual", "ai_predicted", "ai_predicted_estimated"):
            errors.append(f"{prefix}: invalid source '{ann.source}'")

    # --- Event 校验 ---
    for i, evt in enumerate(data.tactile_events):
        prefix = f"event[{i}]"

        # 1. 类型有效性
        if evt.event_type not in EVENT_PRESETS:
            errors.append(f"{prefix}: unknown event_type '{evt.event_type}'")
        else:
            # 2. 点事件 vs 区间事件一致性
            preset = EVENT_PRESETS[evt.event_type]
            if not preset["is_interval"]:
                # 点事件: start_frame == end_frame == frame_idx
                if evt.start_frame != evt.frame_idx:
                    warnings.append(
                        f"{prefix}: point event start_frame={evt.start_frame} "
                        f"!= frame_idx={evt.frame_idx}"
                    )
            else:
                # 区间事件: start_frame <= end_frame
                if evt.end_frame < evt.start_frame:
                    errors.append(
                        f"{prefix}: interval end_frame={evt.end_frame} "
                        f"< start_frame={evt.start_frame}"
                    )

        # 3. 帧范围
        if evt.frame_idx < 0 or evt.frame_idx >= num_frames:
            errors.append(f"{prefix}: frame_idx={evt.frame_idx} out of [0, {num_frames-1}]")
        if evt.start_frame < 0:
            errors.append(f"{prefix}: start_frame={evt.start_frame} < 0")
        if evt.end_frame >= num_frames:
            errors.append(f"{prefix}: end_frame={evt.end_frame} >= num_frames={num_frames}")

        # 4. confidence
        if not (0.0 <= evt.confidence <= 1.0):
            errors.append(f"{prefix}: confidence={evt.confidence} out of [0,1]")

        # 5. source
        if evt.source not in ("manual", "ai_predicted"):
            errors.append(f"{prefix}: invalid source '{evt.source}'")

    # --- 统计 ---
    primitive_types = {}
    for ann in data.primitive_annotations:
        primitive_types[ann.primitive_name] = primitive_types.get(ann.primitive_name, 0) + 1

    event_types = {}
    for evt in data.tactile_events:
        event_types[evt.event_type] = event_types.get(evt.event_type, 0) + 1

    stats = {
        "num_primitive_annotations": len(data.primitive_annotations),
        "num_tactile_events": len(data.tactile_events),
        "primitive_type_distribution": primitive_types,
        "event_type_distribution": event_types,
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def annotate_from_taxonomy(data, taxonomy=None, min_confidence: float = 0.4,
                            clear_existing: bool = False) -> int:
    """批量应用 taxonomy 规则进行 primitive 预标注

    遍历所有帧，用 taxonomy 规则评估每帧，合并连续相同 primitive 为区间标注。

    Args:
        data: TLabelData 实例
        taxonomy: TaxonomyConfig 实例，None 则使用默认 taxonomy
        min_confidence: 最低置信度阈值
        clear_existing: 是否清除已有 primitive 标注

    Returns:
        新增的标注数量
    """
    from tlabel.core.taxonomy import get_default_taxonomy, evaluate_rule

    if taxonomy is None:
        taxonomy = get_default_taxonomy()

    if clear_existing:
        data.primitive_annotations.clear()

    # 逐帧评估
    frame_primitives = []  # [(frame_idx, primitive_name, confidence), ...]
    for frame in data.frames:
        best_name = None
        best_conf = 0.0
        for name, rule in taxonomy.primitives.items():
            matched, conf = evaluate_rule(rule, frame)
            if matched and conf > best_conf:
                best_name = name
                best_conf = conf
        if best_name is not None and best_conf >= min_confidence:
            frame_primitives.append((frame.frame_idx, best_name, best_conf))

    # 合并连续相同 primitive 为区间
    count = 0
    if frame_primitives:
        start_idx = frame_primitives[0][0]
        current_name = frame_primitives[0][1]
        current_conf = frame_primitives[0][2]
        end_idx = start_idx

        for i in range(1, len(frame_primitives)):
            fidx, name, conf = frame_primitives[i]
            if name == current_name and fidx == end_idx + 1:
                # 连续同 primitive，扩展区间
                end_idx = fidx
                current_conf = min(current_conf, conf)
            else:
                # 切换，保存上一段
                if end_idx >= start_idx:
                    try:
                        data.add_primitive(
                            name=current_name,
                            start_frame=start_idx,
                            end_frame=end_idx,
                            confidence=round(current_conf, 4),
                            source="ai_predicted",
                        )
                        count += 1
                    except ValueError:
                        pass
                start_idx = fidx
                current_name = name
                current_conf = conf
                end_idx = fidx

        # 保存最后一段
        if end_idx >= start_idx:
            try:
                data.add_primitive(
                    name=current_name,
                    start_frame=start_idx,
                    end_frame=end_idx,
                    confidence=round(current_conf, 4),
                    source="ai_predicted",
                )
                count += 1
            except ValueError:
                pass

    return count


def annotate_events_from_data(data, clear_existing: bool = False) -> int:
    """从数据模式自动检测触觉事件

    检测逻辑:
    - contact_onset: contact 从 False→True 的跳变帧
    - contact_loss: contact 从 True→False 的跳变帧
    - slip: slip_event 为 True 的连续区间
    - force_spike: force_magnitude 突变超过 2σ
    - stable_grip: 连续 N 帧 contact=True 且 force 波动 < 阈值

    Args:
        data: TLabelData 实例
        clear_existing: 是否清除已有 event 标注

    Returns:
        新增的事件数量
    """
    import math

    if clear_existing:
        data.tactile_events.clear()

    count = 0
    frames = data.frames
    n = len(frames)
    if n == 0:
        return 0

    # 提取时间序列
    contacts = [f.contact for f in frames]
    slips = [f.slip_event for f in frames]
    # P9：None 表示"没有这个值"，不能当 0.0 参与统计
    forces = [f.force_magnitude for f in frames]

    # --- contact_onset / contact_loss ---
    for i in range(1, n):
        if contacts[i] and not contacts[i - 1]:
            data.add_event("contact_onset", frame_idx=i, source="ai_predicted")
            count += 1
        elif not contacts[i] and contacts[i - 1]:
            data.add_event("contact_loss", frame_idx=i, source="ai_predicted")
            count += 1

    # --- slip (区间事件) ---
    slip_start = None
    for i in range(n):
        if slips[i] and contacts[i]:
            if slip_start is None:
                slip_start = i
        else:
            if slip_start is not None:
                data.add_event(
                    "slip", frame_idx=slip_start,
                    start_frame=slip_start, end_frame=i - 1,
                    source="ai_predicted",
                )
                count += 1
                slip_start = None
    # 处理末尾还在 slip 的情况
    if slip_start is not None:
        data.add_event(
            "slip", frame_idx=slip_start,
            start_frame=slip_start, end_frame=n - 1,
            source="ai_predicted",
        )
        count += 1

    # --- force_spike ---
    # P9：只用真实存在的力值参与统计；None 不参与，也不得用 0.0 替代
    valid_forces = [x for x in forces if x is not None]
    if len(valid_forces) >= 3:
        nv = len(valid_forces)
        mean_f = sum(valid_forces) / nv
        var_f = sum((f - mean_f) ** 2 for f in valid_forces) / nv
        std_f = math.sqrt(var_f) if var_f > 0 else 0.0
        threshold = mean_f + 2 * std_f if std_f > 0 else 1.0

        for i in range(1, n):
            cur, prev = forces[i], forces[i - 1]
            if cur is None or prev is None:
                continue
            delta = abs(cur - prev)
            if cur > threshold and delta > std_f:
                data.add_event(
                    "force_spike", frame_idx=i,
                    source="ai_predicted",
                    metadata={"magnitude": round(cur, 4), "delta": round(delta, 4)},
                )
                count += 1

    # --- stable_grip ---
    # P9：稳定抓握依赖力数据。无力数据的帧不参与判定，也不生成 stable_grip
    # 事件 —— 否则不支持力测量的传感器会被判定为"抓握稳定"，是凭空造标签。
    MIN_STABLE_FRAMES = 5
    FORCE_STABLE_THRESHOLD = 0.02

    stable_start = None
    for i in range(n):
        if contacts[i] and not slips[i]:
            if stable_start is None:
                stable_start = i
            # 检查力波动（只用真实存在的力值）
            if i > stable_start:
                window = [x for x in forces[stable_start:i + 1] if x is not None]
                if len(window) < 2:
                    # 无力数据，无法判断稳定性 → 结束区间，不生成事件
                    stable_start = None
                    continue
                w_max = max(window)
                w_min = min(window)
                if (w_max - w_min) > FORCE_STABLE_THRESHOLD:
                    # 力波动过大，结束当前区间
                    if (i - stable_start) >= MIN_STABLE_FRAMES:
                        data.add_event(
                            "stable_grip", frame_idx=stable_start,
                            start_frame=stable_start, end_frame=i - 1,
                            source="ai_predicted",
                            metadata={"duration_frames": i - stable_start},
                        )
                        count += 1
                    stable_start = None
        else:
            if stable_start is not None:
                if (i - stable_start) >= MIN_STABLE_FRAMES:
                    data.add_event(
                        "stable_grip", frame_idx=stable_start,
                        start_frame=stable_start, end_frame=i - 1,
                        source="ai_predicted",
                        metadata={"duration_frames": i - stable_start},
                    )
                    count += 1
                stable_start = None

    # 处理末尾还在 stable 的情况
    if stable_start is not None and (n - stable_start) >= MIN_STABLE_FRAMES:
        data.add_event(
            "stable_grip", frame_idx=stable_start,
            start_frame=stable_start, end_frame=n - 1,
            source="ai_predicted",
            metadata={"duration_frames": n - stable_start},
        )
        count += 1

    return count


def clear_annotations(data, primitives: bool = True, events: bool = True) -> Dict:
    """清除标注

    Args:
        data: TLabelData 实例
        primitives: 是否清除 primitive 标注
        events: 是否清除 event 标注

    Returns:
        dict: {"cleared_primitives": int, "cleared_events": int}
    """
    cleared = {"cleared_primitives": 0, "cleared_events": 0}

    if primitives:
        cleared["cleared_primitives"] = len(data.primitive_annotations)
        data.primitive_annotations.clear()

    if events:
        cleared["cleared_events"] = len(data.tactile_events)
        data.tactile_events.clear()

    return cleared


def get_annotation_summary(data) -> Dict:
    """获取标注摘要 — 统一查询接口

    Returns:
        dict with:
            - primitives: list of primitive annotation dicts
            - events: list of event dicts
            - timeline: 帧级标注时间线
            - validation: validate_annotations() 结果
    """
    validation = validate_annotations(data)

    # 构建帧级时间线
    timeline = []
    for i, frame in enumerate(data.frames):
        entry = {"frame_idx": frame.frame_idx, "primitives": [], "events": []}

        for ann in data.primitive_annotations:
            if ann.contains_frame(frame.frame_idx):
                entry["primitives"].append({
                    "name": ann.primitive_name,
                    "confidence": ann.confidence,
                })

        for evt in data.tactile_events:
            if evt.contains_frame(frame.frame_idx):
                entry["events"].append({
                    "type": evt.event_type,
                    "confidence": evt.confidence,
                })

        # 只保留有标注的帧
        if entry["primitives"] or entry["events"]:
            timeline.append(entry)

    return {
        "primitives": [p.to_dict() for p in data.primitive_annotations],
        "events": [e.to_dict() for e in data.tactile_events],
        "timeline": timeline,
        "validation": validation,
    }
