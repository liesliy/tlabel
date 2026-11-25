"""正向测试：规范明确**允许**的形态必须通过。

这一类测试的作用和 negative 相反——它防的是**校验器过严**。一个只会报错的
校验器看起来「很安全」，但它会把合法的生产者挡在门外，逼他们改数据去迎合
工具，那时标准就已经失效了。

所以这里每条都对应规范里一个 `MAY` 或「合法形态」的明确说法。删掉这些测试，
校验器就可以悄悄收紧而没人发现。
"""

from __future__ import annotations

import copy

from conftest import assert_accepted, base_dataset, base_frame

# ── TL-5.3-6 法向无法确定时 normal_force 必须是 null ──────────


def test_normal_force_null_when_normal_undetermined_is_accepted():
    """TL-5.3-6：法向确定不了时，`null` + `not_acquired` 是唯一正确做法。

    这是整份规范最核心的场景（附录 D.2）：传感器给了三维力，但接触法向未知。
    此时 `normal_force` 必须是 `null`，不许拿 `‖F‖` 顶上。

    这条测试防的是**校验器过严**：如果实现要求「有 `force_vector` 就必须有
    `normal_force`」，生产者唯一能过校验的做法就是拿模长冒充投影——那正是
    规范要禁的东西。校验器逼人造假，比校验器漏报更糟。
    """
    fr = base_frame()
    fr["normal_force"] = None
    fr["field_status"] = {"normal_force": "not_acquired"}
    # force_vector 非 null，且 provenance 里没有 normal_reference——
    # 这正是「有三维力但法向未知」的形态
    fr["field_provenance"]["normal_force"] = {
        "type": "derived_measurement",
        "method_id": "conf:normal_pending_v1",
        "note": "contact normal undetermined; sensor z-axis is nominal surface normal only",
    }
    assert_accepted(fr)


# ── TL-5.2-7 像素坐标无法转毫米 ────────────────────────────


def test_pixel_centroid_as_null_with_unsupported_is_accepted():
    """TL-5.2-7：转不了毫米时，null + unsupported 是正确做法，不是缺陷。

    这条约束的是生产者：不能假装像素是毫米。但规范同时给了出路——把像素
    留在 `sensor_specific` 里。这条测试证明校验器没有把「有接触但没坐标」
    当成错误，否则生产者只能二选一：要么撒谎填像素，要么丢掉位置信息。
    """
    ds = base_dataset()
    ds["capabilities"]["contact_centroid"] = False

    fr = base_frame()
    fr["contact_centroid"] = None
    fr["contact_centroid_frame"] = None
    fr["field_status"] = {"contact_centroid": "unsupported"}
    fr["sensor_specific"] = {
        # 像素坐标保留在这里是规范明说允许的
        "centroid_pixel": {
            "value": [312.0, 208.0],
            "unit": "1",
            "definition": "contact centroid in sensor pixel indices, uncalibrated",
            "method_id": "conf:pixel_centroid_v1",
            "namespace": "conf",
        }
    }

    assert_accepted(fr, ds)


# 注：TL-5.2-7 没有「反面」测试。像素索引 312 和物理长度 312 mm 在 JSON 里
# 是同一个数，校验器无法从数值判别。规范把这条定为生产者义务，是因为只有
# 生产者知道自己做没做标定。硬要测反面，只能靠匹配 definition 的措辞——那
# 是把自然语言当 schema 用，误报率会比漏报更麻烦。


# ── TL-5.3-7 只有法向力、没有三维力 ────────────────────────


def test_normal_force_without_force_vector_is_accepted():
    """TL-5.3-7：只有 `normal_force`、没有 `force_vector` 是合法的。

    很多单轴力传感器只能给出法向分量。要求它们必须有三维力，等于把这些
    传感器排除在标准之外。规范允许 `force_vector` 为 null，只要法向量的
    来源说清楚（这里是 `sensor_surface_nominal`，不需要 `n`）。
    """
    ds = base_dataset()
    ds["capabilities"]["force_vector"] = False

    fr = base_frame()
    fr["force_vector"] = None
    fr["force_vector_frame"] = None
    fr["field_status"] = {"force_vector": "unsupported"}
    fr["normal_force"] = 3.10
    fr["field_provenance"]["normal_force"] = {
        "type": "direct_measurement",
        "sensor_or_channel_id": "sensor_0.fz",
        # 单轴传感器：直接测量，所以必须说明测的是哪根轴（TL-5.3-5），
        # 并把轴与接触法向的对齐容差写出来（TL-5.3-10）——不写容差的话，
        # 「轴对齐法向」只是个假设，读到侧向力时会被当成法向力。
        "normal_reference": "measurement_axis",
        "measurement_axis": "sensor_0.z",
        "angular_tolerance_deg": 2.0,
    }

    assert_accepted(fr, ds)


# ── TL-6.4-2 弹性体形变记在 sensor_specific ───────────────


def test_elastomer_deformation_in_sensor_specific_is_accepted():
    """TL-6.4-2：弹性体形变属于派生特征，放 `sensor_specific` 是对的。

    它不是「物体形变」——传感器自己的胶层被压了多少，和被抓物体变形多少
    是两件事。前者是传感器状态，后者是被测对象状态。混在一起会让下游把
    胶层压缩量当成物体变形，结论完全反了。
    """
    fr = base_frame()
    fr["sensor_specific"] = {
        "elastomer_deformation_rms": {
            "value": 0.37,
            "unit": "1",
            "definition": "RMS of sensor elastomer layer deformation, normalized by layer thickness",
            "method_id": "conf:elastomer_rms_v1",
            "namespace": "conf",
        }
    }
    assert_accepted(fr, base_dataset())


# ── TL-7.3-3 无能力时 null 可省略状态 ─────────────────────


def test_null_without_status_under_false_capability_is_accepted():
    """TL-7.3-3：`capability: false` 时，null 可以不带状态。

    状态是用来解释「为什么这个有能力的字段没有值」的。既然声明了没这个
    能力，null 就是唯一可能的值，再要求填状态只是重复信息。
    """
    ds = base_dataset()
    ds["capabilities"]["temperature"] = False

    fr = base_frame()
    fr["temperature"] = None
    # 刻意不设 field_status —— 这正是本条允许的省略
    assert "temperature" not in (fr.get("field_status") or {})

    assert_accepted(fr, ds)


# ── TL-9.5-3 V2 兼容的裸值 ────────────────────────────────


def test_bare_value_in_sensor_specific_is_accepted_at_core_level():
    """TL-9.5-3：core 级允许 `sensor_specific` 用裸值（V2 兼容）。

    V2 里这个容器就是 `{name: number}`。core 级接受裸值，是为了让存量数据
    不必一次性重写成结构化条目——这是标准落地的现实条件，不是妥协。
    Extended 级才要求结构化。
    """
    fr = base_frame()
    fr["sensor_specific"] = {"deformation_image_rms": 0.37}
    assert_accepted(fr, base_dataset(), level="core")


def test_bare_value_in_sensor_specific_is_rejected_at_extended_level():
    """TL-9.5-3 的边界：同一份数据在 Extended 级必须被拒。

    和上一条配对，证明「core 放宽」是分级的结果，不是根本没查。
    """
    from conftest import assert_rejected

    fr = base_frame()
    fr["sensor_specific"] = {"deformation_image_rms": 0.37}
    assert_rejected(fr, "TL-9.5-3", level="extended")


# ── TL-9.6-2 单传感器省略 instances ───────────────────────


def test_single_sensor_may_omit_instances():
    """TL-9.6-2：单传感器数据集可以完全不写 `instances`。

    绝大多数现有数据集就是一个传感器。强制声明一个长度为 1 的 `instances`
    数组，只会给所有生产者加一步无信息的机械操作。
    """
    ds = base_dataset()
    assert "instances" not in ds
    fr = base_frame()
    assert "instance_id" not in fr
    assert_accepted(fr, ds)


def test_multi_instance_declared_and_tagged_is_accepted():
    """TL-9.6-1 的合法面：声明了两个实例、每帧都带 id → 通过。

    多实例数据的时序约束是**按实例**算的（TL-9.7-3）。这条用两个实例各自
    从 t=0 开始的数据证明校验器没有把它们放进同一条时间轴比较——那是最容易
    写错的地方：全局单变量会让第二个实例的 t=0 被判成「比第一个实例末尾递减」。
    """
    ds = base_dataset()
    ds["instances"] = [
        {"instance_id": "finger_0", "instance_role": "contact_array"},
        {"instance_id": "finger_1", "instance_role": "contact_array"},
    ]

    f0 = base_frame()
    f0["instance_id"] = "finger_0"
    f0["timestamp_s"] = 0.0
    f1 = base_frame()
    f1["instance_id"] = "finger_1"
    f1["timestamp_s"] = 0.0  # 与 finger_0 同时刻，不是递减

    ds["frames"] = [f0, f1]
    assert_accepted(None, ds)


def test_instance_time_axis_is_independent():
    """TL-9.7-3 / 9.6-1：两个实例的时间轴各自单调即可，不要求交错有序。

    `finger_1` 的 t=0.0 排在 `finger_0` 的 t=0.033 之后，全局看是递减的。
    但同一实例内部没有递减，所以合法。这是多实例数据最常见的形状——两路
    传感器各自按自己的节奏采样，落盘顺序不代表时间顺序。
    """
    ds = base_dataset()
    ds["instances"] = [
        {"instance_id": "finger_0"},
        {"instance_id": "finger_1"},
    ]
    a0 = base_frame()
    a0["instance_id"] = "finger_0"
    a0["timestamp_s"] = 0.033
    b0 = base_frame()
    b0["instance_id"] = "finger_1"
    b0["timestamp_s"] = 0.0
    a1 = base_frame()
    a1["instance_id"] = "finger_0"
    a1["timestamp_s"] = 0.066

    ds["frames"] = [a0, b0, a1]
    assert_accepted(None, ds)
