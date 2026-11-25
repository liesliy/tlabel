"""第 9 章 —— 来源、上下文、多实例与时间的反向测试。

覆盖 TL-9.2-1 / 9.2-2 / 9.3-1 / 9.3-2 / 9.3-4 / 9.4-1 / 9.4-2 /
9.5-2 / 9.5-3 / 9.6-1 / 9.6-2 / 9.7-1 / 9.7-2 / 9.7-3 / 9.7-5。

这一章回答的是「这个数从哪来」。V2 里同一列数值可能由三种完全不同的方式
产生（实测、派生、代理量），而数据本身不说明，下游只能一律当实测用。
第 9 章的存在就是为了堵住这个洞。
"""

from __future__ import annotations

import copy

import pytest

from conftest import (
    base_dataset,
    base_frame,
    assert_accepted,
    assert_rejected,
    run,
)


def _without(d: dict, *keys):
    return {k: v for k, v in d.items() if k not in keys}


def _two_frames(f1=None, f2=None):
    """构造一个两帧数据集，用于时间单调性检查。"""
    ds = base_dataset()
    ds["frames"] = [
        copy.deepcopy(f1 if f1 is not None else base_frame()),
        copy.deepcopy(f2 if f2 is not None else base_frame(timestamp_s=1.0)),
    ]
    return ds


# ─────────────────────────────────────────────────────────────
# 9.2 值级来源
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "field", ["contact", "contact_centroid", "normal_force", "force_vector", "slip_event"]
)
def test_non_null_value_without_provenance_is_rejected(field):
    """TL-9.2-1：每个非 null 的核心值都必须声明来源类型。

    没有来源类型的数值，下游无法判断它是测出来的还是算出来的。V2 里
    「形变图像的 RMS」和「毫米级形变」都以数字形式存在，看起来一样——
    来源类型是唯一能区分它们的东西。
    """
    frame = base_frame()
    frame["field_provenance"] = _without(frame["field_provenance"], field)
    assert_rejected(frame, "TL-9.2-1")


def test_null_value_without_provenance_is_accepted():
    """TL-9.2-1 的边界：值为 null 时不需要来源。

    null 没有「怎么产生的」可言。要求为 null 也填来源，会逼生产者
    编一个出来——那比留空更糟。
    """
    frame = base_frame(contact=None, temperature=None, torque_vector=None)
    frame["field_status"] = {"contact": "not_acquired"}
    frame["field_provenance"] = _without(frame["field_provenance"], "contact")
    assert_accepted(frame)


def test_device_level_provenance_does_not_substitute_value_level():
    """TL-9.2-2：设备级来源不能代替值级来源。

    「这台设备有六轴力传感器」不等于「这个力值来自六轴通道」。设备级
    描述的是硬件，值级描述的是这一帧这个数的产生方式——后者可能经过了
    滤波、融合、降采样，设备级记录对此一无所知。
    """
    frame = base_frame()
    frame["field_provenance"] = _without(frame["field_provenance"], "force_vector")
    dataset = base_dataset()
    dataset["provenance"] = {
        "device_model": "GelSight Mini",
        "firmware": "2.1.0",
        "calibration_date": "2026-01-15",
    }
    report = run(frame, dataset)
    assert any(f.clause == "TL-9.2-1" for f in report.findings), [
        (f.code, f.clause) for f in report.findings
    ]


# ─────────────────────────────────────────────────────────────
# 9.3 来源类型
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["proxy", "estimated", "simulated", "unknown", ""])
def test_forbidden_provenance_type_is_rejected(bad):
    """TL-9.3-1：没有「代理量」这个来源类型。

    这是规范里最重要的一条否定式规定。`proxy` 一旦被允许，所有「能算出一个
    数但不是这个量」的情况都会流进核心字段——形变图像 RMS 冒充毫米级形变
    正是这么发生的。五类之外没有第六类，也没有逃生舱。
    """
    frame = base_frame()
    frame["field_provenance"]["contact"] = {"type": bad}
    assert_rejected(frame, "TL-9.3-1")


@pytest.mark.parametrize(
    "ptype, missing",
    [
        ("direct_measurement", "sensor_or_channel_id"),
        ("model_inference", "method_id"),
        ("externally_supplied", "external_source_id"),
    ],
)
def test_provenance_type_without_required_info_is_rejected(ptype, missing):
    """TL-9.3-2：每种来源类型都必须带齐自己的必需信息。

    `direct_measurement` 缺通道标识时，无法回溯是哪个传感器通道给的；
    `human_annotation` 缺标注者时，无法评估标注质量。这些不是可选项——
    缺了就等于没有来源。
    """
    frame = base_frame()
    frame["field_provenance"]["contact"] = {"type": ptype}
    assert_rejected(frame, "TL-9.3-2")


def test_human_annotation_without_annotator_id_is_rejected():
    """TL-5.1-4：`contact` 由人工标注时必须带 `annotator_id`。

    这条比 TL-9.3-2 的通则更具体，所以校验器报的是 5.1-4。规范在 5.1 节
    专门写了它，因为 `contact` 是唯一允许人工标注的核心字段——人工判定
    「有没有接触」需要知道是谁判的，否则无法评估一致性。
    """
    frame = base_frame()
    frame["field_provenance"]["contact"] = {"type": "human_annotation"}
    assert_rejected(frame, "TL-5.1-4")


def test_derived_measurement_without_method_id_is_rejected():
    """TL-9.3-4：派生值必须声明 `method_id`。

    派生值由计算得到，而计算方式决定了结果的含义。`geometric_median` 和
    `pressure_weighted` 都是「接触中心」，但对多点接触给出的位置不同。
    没有 method_id，这个值既不能复现也不能比较。
    """
    frame = base_frame()
    frame["field_provenance"]["contact_centroid"] = {
        "type": "derived_measurement",
        "centroid_method": "pressure_weighted",
        "weighting_quantity": "pressure_matrix",
        "input_refs": ["raw_sensor_values.pressure_matrix"],
        # 缺 method_id
    }
    assert_rejected(frame, "TL-9.3-2")


# ─────────────────────────────────────────────────────────────
# 9.4 测量上下文
# ─────────────────────────────────────────────────────────────


def test_centroid_without_method_is_rejected():
    """TL-9.4-1：`contact_centroid` 非 null 时必须声明计算方法。"""
    frame = base_frame()
    frame["field_provenance"]["contact_centroid"] = {
        "type": "derived_measurement",
        "input_refs": ["raw_sensor_values.pressure_matrix"],
        "method_id": "conf:centroid_v1",
    }
    report = run(frame)
    assert report.findings, "缺少 centroid_method 必须被拒"


def test_pressure_weighted_centroid_without_weighting_source_is_rejected():
    """TL-9.4-1：`pressure_weighted` 必须说明权重来自哪个量。

    「按压力加权」里的「压力」可能是 taxel 原始读数、标定后的帕斯卡值、
    或者别的通道。权重来源不同，算出的中心不同。
    """
    frame = base_frame()
    frame["field_provenance"]["contact_centroid"] = {
        "type": "derived_measurement",
        "centroid_method": "pressure_weighted",
        "input_refs": ["raw_sensor_values.pressure_matrix"],
        "method_id": "conf:centroid_v1",
    }
    report = run(frame)
    assert report.findings, "pressure_weighted 缺权重来源必须被拒"


def test_direct_normal_force_without_measurement_axis_is_rejected():
    """TL-9.4-1：直接测量的法向力必须声明测量轴。

    单轴力传感器的敏感轴与接触界面法向有夹角时，测到的是轴向分量而不是
    法向力。不声明轴，这个值无法归入任何一类——既不能说它是法向力，
    也不能说它是别的什么。
    """
    frame = base_frame()
    frame["field_provenance"]["normal_force"] = {"type": "direct_measurement"}
    assert_rejected(frame, "TL-5.3-5")


def test_temperature_without_measurement_target_is_rejected():
    """TL-9.4-1：`temperature` 非 null 时必须声明测量对象。

    传感器表面温度、环境温度、被操作物体温度是三个不同的量，都叫
    「温度」。不声明测的是什么，这个数没有物理含义。
    """
    frame = base_frame(temperature=25.4)
    frame["field_provenance"]["temperature"] = {
        "type": "direct_measurement",
        "sensor_or_channel_id": "sensor_0.temp",
    }
    dataset = base_dataset()
    dataset["capabilities"] = {**dataset["capabilities"], "temperature": True}
    report = run(frame, dataset)
    assert report.findings, "temperature 缺 measurement_target 必须被拒"


# ─────────────────────────────────────────────────────────────
# 9.5 派生特征
# ─────────────────────────────────────────────────────────────


def test_structured_sensor_specific_entry_missing_key_is_rejected():
    """TL-9.5-2：结构化条目必须含 `value` / `unit` / `definition` / `method_id` / `namespace`。

    这五项各自解决一个问题：值是什么、单位是什么、定义是什么、怎么算的、
    谁定义的。少 `definition`，一个叫 `rms` 的数就没人知道是图像的还是
    压力的 RMS；少 `namespace`，两个厂商的同名特征会被当成同一个。
    """
    frame = base_frame(sensor_specific={"deformation_rms": {"value": 0.37}})
    assert_rejected(frame, "TL-9.5-2", level="extended")


def test_unstructured_sensor_specific_entry_fails_extended():
    """TL-9.5-3：裸值在 Core 可接受（V2 兼容），在 Extended 必须被拒。

    这条与上一条的分界很重要：裸值不是「写错了」，是「V2 就这样」。
    Core 不强制结构化是为了让 V2 数据能直接读；Extended 要求结构化是
    为了让新数据可被机器理解。两级的要求不同，报的条款也不同。
    """
    frame = base_frame(sensor_specific={"deformation_rms": 0.37})
    assert_accepted(frame)  # Core 放过
    assert_rejected(frame, "TL-9.5-3", level="extended")


def test_fully_structured_sensor_specific_entry_is_accepted():
    """TL-9.5-2 的对照：五项齐全必须通过。"""
    frame = base_frame(
        sensor_specific={
            "deformation_rms": {
                "value": 0.37,
                "unit": "1",
                "definition": "形变图像灰度残差的均方根，无量纲",
                "method_id": "conf:deform_rms_v1",
                "namespace": "org.touchlabel.gelsight",
            }
        }
    )
    assert_accepted(frame, level="extended")


# ─────────────────────────────────────────────────────────────
# 9.6 多实例
# ─────────────────────────────────────────────────────────────


def test_single_sensor_dataset_may_omit_instances():
    """TL-9.6-2：单传感器数据集可以省略 `instances`。

    这条是正向断言，但它保护的是一个具体错误：把「必须有 instances」
    写成硬要求，会让所有单传感器数据集（绝大多数）都需要改格式。
    """
    ds = base_dataset()
    assert "instances" not in ds
    assert_accepted(base_frame(), dataset=ds)


# ─────────────────────────────────────────────────────────────
# 9.7 时间
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["0.033", None, True])
def test_non_numeric_timestamp_is_rejected(bad):
    """TL-9.7-1：`timestamp_s` 必须是数值。

    字符串时间戳最危险：它不报错，但让所有比较退化为字典序——"10" < "9"，
    时间轴会静默乱序。`True` 也要拦，因为 Python 里 `isinstance(True, int)`
    为真，一个布尔值能混进数值字段。
    """
    assert_rejected(base_frame(timestamp_s=bad), "TL-9.7-1")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_timestamp_is_rejected(bad):
    """TL-9.7-1：`timestamp_s` 必须是有限数。

    NaN 参与任何比较都返回 False，单调性检查会静默通过——一个坏帧混在
    中间，时间轴看起来仍然有序。
    """
    assert_rejected(base_frame(timestamp_s=bad), "TL-9.7-1")


def test_missing_time_base_is_rejected():
    """TL-9.7-2：数据集必须声明 `time_base`。

    没有它，`timestamp_s = 0.033` 可能是「数据集起点后 33 毫秒」，也可能是
    Unix 时间戳的低位。V2 里这三种填法都叫 `timestamp_s`，下游拿两个
    数据集对齐会得到静默错误的结果。
    """
    ds = _without(base_dataset(), "time_base")
    assert_rejected(base_frame(), "TL-9.7-2", dataset=ds)


@pytest.mark.parametrize(
    "good", ["dataset_start", "session_start", "unix_epoch", "device_clock"]
)
def test_all_four_time_bases_are_accepted(good):
    """TL-9.7-2 的对照：四个值逐个验证。

    `device_clock` 需要额外声明时钟身份（TL-9.7-5），所以这里一并补上。
    """
    ds = {**base_dataset(), "time_base": good}
    if good == "device_clock":
        ds["provenance"] = {"clock_identity": "mcu-0"}
    assert_accepted(base_frame(), dataset=ds)


def test_decreasing_timestamp_is_rejected():
    """TL-9.7-3：同一实例内时间戳必须单调不减。

    递减说明帧顺序错了，或者时钟回拨了。两种情况都会让下游按时间
    对齐时得到错位的配对——而它不会报错，只会给出错误的关联。
    """
    ds = _two_frames(base_frame(timestamp_s=1.0), base_frame(timestamp_s=0.5))
    report = run(dataset=ds)
    assert any(f.clause == "TL-9.7-3" for f in report.findings), [
        (f.code, f.clause) for f in report.findings
    ]


def test_equal_consecutive_timestamps_are_accepted():
    """TL-9.7-3 的边界：相等是允许的。

    规范原文是「monotonically non-decreasing」，不是 strictly increasing。
    同一时刻采到多个传感器读数时，时间戳相同是正常的。
    """
    ds = _two_frames(base_frame(timestamp_s=1.0), base_frame(timestamp_s=1.0))
    assert_accepted(dataset=ds)


def test_device_clock_without_clock_identity_is_rejected():
    """TL-9.7-5：用设备时钟必须说明是哪台设备的时钟。

    两台设备的 `device_clock` 之间没有可比性——它们的零点是各自的开机时刻。
    不声明身份，就无法判断两个数据集能否按时间合并。
    """
    ds = {**base_dataset(), "time_base": "device_clock"}
    assert_rejected(base_frame(), "TL-9.7-5", dataset=ds)


def test_device_clock_with_clock_identity_is_accepted():
    """TL-9.7-5 的对照：声明了时钟身份必须通过。"""
    ds = {
        **base_dataset(),
        "time_base": "device_clock",
        "provenance": {"clock_identity": "mcu-0"},
    }
    assert_accepted(base_frame(), dataset=ds)
