"""第 9.6–9.7 章：多实例与时间语义的负向测试。

这两节约束的东西看起来平淡，但错了之后**没有症状**：多实例没声明，下游按
实例聚合会得到两个半条数据；时间基准中途变了，两段数据会被拼成一条错误的
时间轴，而且每一帧的时间戳单看都合法。所以这里每条都成对写——正例证明
合法形态没被挡住，反例证明真错了会被抓。
"""

from __future__ import annotations

import copy

from conftest import assert_rejected, base_dataset, base_frame, run

# ── 9.6 多实例 ────────────────────────────────────────────


def _two_instances():
    ds = base_dataset()
    ds["instances"] = [
        {"instance_id": "finger_0", "instance_role": "contact_array"},
        {"instance_id": "finger_1", "instance_role": "contact_array"},
    ]
    return ds


def test_frame_without_instance_id_in_multi_instance_dataset_is_rejected():
    """TL-9.6-1：声明了多个实例，每帧就必须带 `instance_id`。

    缺了它，这一帧属于哪个实例无从判断。下游要么丢掉它，要么默认归到第一个
    实例——后者会让 finger_0 的数据里混进 finger_1 的帧，而且看起来完全正常。
    """
    ds = _two_instances()
    fr = base_frame()
    assert "instance_id" not in fr
    assert_rejected(fr, "TL-9.6-1", ds)


def test_instance_id_not_declared_is_rejected():
    """TL-9.6-1：`instance_id` 必须能在 `instances` 里找到。

    帧说自己是 `finger_2`，声明里只有 finger_0 / finger_1。这通常是拼写
    错误或实例被删了没更新声明——两种情况都该在写入时被拦下，而不是让下游
    遇到一个「未知实例」时自己决定怎么办。
    """
    ds = _two_instances()
    fr = base_frame()
    fr["instance_id"] = "finger_2"
    assert_rejected(fr, "TL-9.6-1", ds)


def test_instance_id_in_single_sensor_dataset_is_rejected():
    """TL-9.6-1：单传感器数据集出现 `instance_id` 是矛盾。

    没有 `instances` 声明，却给帧打了实例标签。这不是「多实例」，是把一个
    传感器拆成了两个身份——下游按实例聚合时会得到两个半条数据。
    """
    ds = base_dataset()
    assert "instances" not in ds
    fr = base_frame()
    fr["instance_id"] = "finger_0"
    assert_rejected(fr, "TL-9.6-1", ds)


def test_instance_id_in_declared_single_instance_is_accepted():
    """TL-9.6-1 的边界：声明了一个实例、帧也带 id → 合法。

    `len(instances) == 1` 时，「多实例每帧必须带 id」这条不适用，但带了也
    不算错——声明和标注是一致的。如果实现把判据写成「len > 1 才允许
    instance_id」，这份合法数据会被误拒。
    """
    ds = base_dataset()
    ds["instances"] = [{"instance_id": "finger_0"}]
    fr = base_frame()
    fr["instance_id"] = "finger_0"
    rep = run(fr, ds)
    assert not [f for f in rep.errors if f.clause == "TL-9.6-1"], (
        f"单实例声明 + 对应 id 应通过：{[(f.code, f.clause) for f in rep.errors]}"
    )


# ── 9.7 时间 ──────────────────────────────────────────────


def test_time_base_change_between_frames_is_rejected():
    """TL-9.7-4：同一实例的帧之间不得改变 `time_base`。

    这是最隐蔽的一类错误：`dataset_start` 下 t=0.033 是「数据集起点后 33 ms」，
    `device_clock` 下同一个 0.033 可能是某个设备计数器的低 32 位。两个数
    长得一样，原点完全不同。中途换基准，前后两段就拼不到一条时间轴上。
    """
    ds = base_dataset()
    f0 = base_frame()
    f0["timestamp_s"] = 0.0
    f1 = base_frame()
    f1["timestamp_s"] = 0.033
    f1["time_base"] = "device_clock"
    ds["frames"] = [f0, f1]

    rep = run(None, ds)
    hits = [f for f in rep.findings if f.clause == "TL-9.7-4"]
    assert hits, f"期望 TL-9.7-4，实际：{[(f.code, f.clause) for f in rep.findings]}"
    assert all(f.outcome == "error" for f in hits), "TL-9.7-4 是 MUST，必须报 error"


def test_time_base_inherited_from_dataset_is_not_a_change():
    """TL-9.7-4 的边界：省略帧级 `time_base` 不算「改变」。

    比较的是「这一帧实际用的基准」，不是「这一帧有没有写这个键」。如果实现
    拿 `frame.get("time_base")` 直接比，第一帧有值第二帧 None 会被判成变化，
    而省略写法的合法数据全都会被误拒。
    """
    ds = base_dataset()  # time_base = dataset_start
    f0 = base_frame()
    f0["timestamp_s"] = 0.0
    f0["time_base"] = "dataset_start"  # 显式写出，与数据集级相同
    f1 = base_frame()
    f1["timestamp_s"] = 0.033
    # 省略，继承数据集级 —— 语义上仍是 dataset_start
    ds["frames"] = [f0, f1]

    rep = run(None, ds)
    assert not [f for f in rep.findings if f.clause == "TL-9.7-4"], (
        f"省略写法不算改变：{[(f.code, f.clause) for f in rep.findings]}"
    )


def test_time_base_change_across_instances_is_allowed():
    """TL-9.7-4 / 9.6-4：实例声明了覆盖，就可以各用自己的基准。

    两个传感器接在两台采集卡上，时钟本来就不共享。TL-9.6-4 允许实例覆盖
    `time_base`，TL-9.7-4 的约束只在「同一实例内」。如果实现用全局单变量
    比较，这份合法数据会被判成变化。

    注意 `b` 的覆盖是**显式声明在 instances 里**的。少了那句声明，下面那条
    测试说明同样的数据就成了非法——差别只在声明，不在值。
    """
    ds = base_dataset()
    ds["instances"] = [
        {"instance_id": "a"},
        {"instance_id": "b", "time_base": "device_clock"},
    ]

    fa = base_frame()
    fa["instance_id"] = "a"
    fa["timestamp_s"] = 0.0
    fa["time_base"] = "dataset_start"
    fb = base_frame()
    fb["instance_id"] = "b"
    fb["timestamp_s"] = 0.0
    fb["time_base"] = "device_clock"
    ds["frames"] = [fa, fb]

    rep = run(None, ds)
    assert not [f for f in rep.findings if f.clause in ("TL-9.7-4", "TL-9.6-4")], (
        f"实例已声明覆盖，跨实例换基准应允许：{[(f.code, f.clause) for f in rep.findings]}"
    )


def test_time_base_override_without_declaration_is_rejected():
    """TL-9.6-4：实例用了第三个基准，却没在 `instances` 里声明。

    和上一条的数据只差 `instances` 里那句 `time_base`。这个差别很小，后果
    很大：下游读 `time_base: device_clock` 的帧时，会以为时间戳来自设备
    时钟，但数据集的声明里没有任何地方说这台设备有独立时钟。
    """
    ds = base_dataset()
    ds["instances"] = [{"instance_id": "a"}, {"instance_id": "b"}]

    fb = base_frame()
    fb["instance_id"] = "b"
    fb["time_base"] = "device_clock"
    ds["frames"] = [fb]

    rep = run(None, ds)
    hits = [f for f in rep.findings if f.clause == "TL-9.6-4"]
    assert hits, (
        f"期望 TL-9.6-4，实际：{[(f.code, f.clause) for f in rep.findings]}"
    )
    assert hits[0].code == "TIME_BASE_INCONSISTENT"


def test_time_base_inherited_from_dataset_is_allowed():
    """TL-9.6-4 的边界：实例不写 `time_base` 时继承数据集级，不算覆盖。

    继承是常态。如果实现把「实例没声明覆盖」理解成「实例没有基准」，
    绝大多数多实例数据集都会被判错。
    """
    ds = base_dataset()
    ds["instances"] = [{"instance_id": "a"}, {"instance_id": "b"}]

    fb = base_frame()
    fb["instance_id"] = "b"
    fb["time_base"] = "dataset_start"
    ds["frames"] = [fb]

    rep = run(None, ds)
    assert not [f for f in rep.findings if f.clause == "TL-9.6-4"], (
        f"继承数据集级基准应允许：{[(f.code, f.clause) for f in rep.findings]}"
    )


def test_timestamp_decrease_within_one_instance_is_rejected():
    """TL-9.7-3：同一实例内时间戳递减必须报错。

    和 9.7-4 一起构成时间语义的两条腿：基准不能变（原点），顺序不能退
    （方向）。缺任何一条，时间轴都可以被拼成任意形状。
    """
    ds = base_dataset()
    f0 = base_frame()
    f0["timestamp_s"] = 0.100
    f1 = base_frame()
    f1["timestamp_s"] = 0.050
    ds["frames"] = [f0, f1]

    rep = run(None, ds)
    assert [f for f in rep.findings if f.clause == "TL-9.7-3"], (
        f"期望 TL-9.7-3，实际：{[(f.code, f.clause) for f in rep.findings]}"
    )
