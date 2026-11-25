"""符合性测试的公共夹具。

设计原则：**基底必须合法**。每个测试只改一处，改的那处就是它要证明的那件事。
如果基底本身有错，所有测试都会因为同一个无关原因变红，覆盖面看着很全，实际
什么都没测到。

所以 `test_base_is_clean` 是这个目录里最重要的一条测试——它证明基底是干净的。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# 让 positive/ negative/ 等子目录里的测试能 `from conftest import base_frame`。
# pytest 的 prepend 模式只会把「测试文件所在目录」放进 sys.path，子目录测试
# 拿到的是 negative/ 而不是 conformance/，找不到本模块。
sys.path.insert(0, str(Path(__file__).parent))

from tlabel.v3 import validate

# ─────────────────────────────────────────────────────────────
# 坐标系
# ─────────────────────────────────────────────────────────────

SENSOR_3D: Dict[str, Any] = {
    "frame_id": "sensor_0",
    "dimension": 3,
    "frame_type": "sensor_local",
    "handedness": "right",
    "axes": {
        "x": "tangential_1",
        "y": "tangential_2",
        "z": "nominal_surface_normal",
    },
    "reference_object": "sensor_sensitive_surface",
}

SURFACE_2D: Dict[str, Any] = {
    "frame_id": "surface_0",
    "dimension": 2,
    "frame_type": "sensor_local",
    "handedness": "right",
    "axes": {"x": "tangential_1", "y": "tangential_2"},
    "reference_object": "sensor_sensitive_surface",
}

CORE_CAPS: Dict[str, bool] = {
    "contact": True,
    "contact_centroid": True,
    "normal_force": True,
    "force_vector": True,
    "torque_vector": False,
    "slip_event": True,
    "slip_velocity": False,
    "temperature": False,
}


# ─────────────────────────────────────────────────────────────
# 基底
# ─────────────────────────────────────────────────────────────

def base_dataset(**overrides: Any) -> Dict[str, Any]:
    """最小合法数据集（core 级）。"""
    ds: Dict[str, Any] = {
        "session_id": "conf-base",
        "time_base": "dataset_start",
        "frames_of_reference": [copy.deepcopy(SENSOR_3D), copy.deepcopy(SURFACE_2D)],
        "capabilities": dict(CORE_CAPS),
    }
    ds.update(overrides)
    return ds


def base_frame(**overrides: Any) -> Dict[str, Any]:
    """最小合法帧（core 级）。

    数值刻意选成「法向分量 ≠ 模长」：F = [0.42, -0.18, 3.10]，n = [0, 0, 1]，
    |F·n| = 3.10，‖F‖ = 3.1309…。这样任何把模长当法向的实现都会被现有测试抓到，
    基底不会替它掩盖问题。
    """
    fr: Dict[str, Any] = {
        "timestamp_s": 0.0,
        "contact": True,
        "contact_centroid": [10.0, 20.0],
        "contact_centroid_frame": "surface_0",
        "normal_force": 3.10,
        "force_vector": [0.42, -0.18, 3.10],
        "force_vector_frame": "sensor_0",
        "torque_vector": None,
        "slip_event": False,
        "slip_velocity": None,
        "temperature": None,
        "field_provenance": {
            "contact": {
                "type": "direct_measurement",
                "sensor_or_channel_id": "sensor_0.ch0",
            },
            "contact_centroid": {
                "type": "derived_measurement",
                "centroid_method": "pressure_weighted",
                "weighting_quantity": "pressure_matrix",
                "input_refs": ["raw_sensor_values.pressure_matrix"],
                "method_id": "conf:centroid_v1",
            },
            "normal_force": {
                "type": "derived_measurement",
                "normal_reference": "sensor_surface_nominal",
                "input_refs": ["force_vector"],
                "method_id": "conf:nominal_projection_v1",
            },
            "force_vector": {
                "type": "direct_measurement",
                "sensor_or_channel_id": "sensor_0.ch0",
            },
            "slip_event": {
                "type": "model_inference",
                "method_id": "conf:slip_detector_v1",
            },
        },
    }
    fr.update(overrides)
    return fr


# ─────────────────────────────────────────────────────────────
# 运行与断言辅助
# ─────────────────────────────────────────────────────────────

def run(
    frame: Optional[Dict[str, Any]] = None,
    dataset: Optional[Dict[str, Any]] = None,
    level: str = "core",
):
    """校验。

    `frame` 给了就校验这一帧；没给则用 `dataset` 里已有的 `frames`
    （多帧场景，如时间单调性）；两者都没有时退化为单个基底帧。
    """
    ds = dataset if dataset is not None else base_dataset()
    ds = copy.deepcopy(ds)
    if frame is not None:
        ds["frames"] = [copy.deepcopy(frame)]
    elif "frames" not in ds:
        # 只在**键不存在**时补基底帧。`frames: []` 是调用方明确表达的
        # 「零帧数据集」，补一帧会让「空数据集」这个场景永远测不到。
        ds["frames"] = [base_frame()]
    return validate(ds, level=level)


def codes(report) -> List[str]:
    return [f.code for f in report.findings]


def clauses(report) -> List[str]:
    return [f.clause for f in report.findings]


def errors_of(report) -> List[str]:
    return [f.code for f in report.findings if f.outcome == "error"]


def assert_rejected(
    frame: Dict[str, Any],
    clause: str,
    dataset: Optional[Dict[str, Any]] = None,
    level: str = "core",
):
    """断言某帧被拒，且**拒的理由是这条条款**。

    只断言「有错」是不够的：任何一处无关错误都会让测试变绿，条款其实没被验证。
    所以这里同时要求 clause 命中。
    """
    rep = run(frame, dataset, level=level)
    assert rep.findings, f"期望因 {clause} 被拒，实际没有任何 finding"
    hit = [f for f in rep.findings if f.clause == clause]
    assert hit, (
        f"期望命中 {clause}，实际 findings："
        f"{[(f.code, f.clause, f.path) for f in rep.findings]}"
    )
    assert any(f.outcome == "error" for f in hit), (
        f"{clause} 命中了但只是 {hit[0].outcome}，应判 error"
    )
    return rep


def assert_accepted(
    frame: Optional[Dict[str, Any]] = None,
    dataset: Optional[Dict[str, Any]] = None,
    level: str = "core",
):
    """断言某帧通过（无 error）。warning / not_verified 不算失败。"""
    rep = run(frame, dataset, level=level)
    assert rep.ok, (
        f"期望通过，实际错误：{[(f.code, f.clause, f.path) for f in rep.errors]}"
    )
    return rep


# ─────────────────────────────────────────────────────────────
# 基底自检
# ─────────────────────────────────────────────────────────────

def test_base_is_clean():
    """基底必须零错误。这条红了，整个目录的结论都不可信。"""
    rep = run()
    assert rep.ok, f"基底不干净：{[(f.code, f.clause, f.path) for f in rep.errors]}"


def test_base_has_all_core_keys():
    """TL-4.4-1：8 个核心键必须在场。"""
    fr = base_frame()
    for k in ("contact", "contact_centroid", "normal_force", "force_vector",
              "torque_vector", "slip_event", "slip_velocity", "temperature"):
        assert k in fr, f"基底缺核心键 {k}"


def test_base_normal_force_is_not_norm():
    """基底的法向力必须不等于模长，否则「模长冒充法向」类测试全部失效。"""
    import math
    fr = base_frame()
    norm = math.sqrt(sum(v * v for v in fr["force_vector"]))
    assert abs(norm - fr["normal_force"]) > 1e-3, (
        "基底里 normal_force 等于 ‖F‖，会让 T1 类测试失去判别力"
    )
