"""
V3.0 符合性用例测试

把 `tlabel-v3-spec/conformance/cases.yaml` 的 26 个用例变成 pytest 参数化测试，
并固化若干「改好应当消失」的对照断言——防止校验器退化成恒真。

权威来源：tlabel-spec-v3.md 第 5、7、8、9、10、11 章
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import pytest

yaml = pytest.importorskip("yaml")

from tlabel.v3 import validate  # noqa: E402


def _find_cases() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "tlabel-v3-spec" / "conformance" / "cases.yaml"
        if cand.is_file():
            return cand
    pytest.skip("找不到 conformance/cases.yaml")


CASES_PATH = _find_cases()


def _load() -> Dict[str, Dict[str, Any]]:
    data = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for key, group in data.items():
        if not key.startswith("group_"):
            continue
        for case in group.get("cases") or []:
            out[case["id"]] = case
    return out


CASES = _load()


def _run(case: Dict[str, Any], level: str = "core"):
    """按 cases.yaml 的 assembly 规则组装。

    `input.frames` 优先（多帧场景），否则把单个 `input.frame` 作为唯一元素。
    这条规则和 `conformance/run_cases.py` 必须一致——两处各写一套，向量在
    一边过、另一边不过，排查时先怀疑向量本身，而问题在组装。
    """
    inp = case["input"]
    ds = dict(inp["dataset"])
    ds["frames"] = list(inp["frames"]) if "frames" in inp else [inp["frame"]]
    return validate(ds, level=level)


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_conformance_case(case_id: str):
    case = CASES[case_id]
    report = _run(case)
    expect = case.get("expect") or {}

    if case.get("kind") == "positive":
        assert report.ok, (
            f"{case_id} 正例被判失败："
            f"{[(f.code, f.path) for f in report.errors]}"
        )
    else:
        want = expect.get("code")
        assert want, f"{case_id} 反例未声明 expect.code"
        assert want in report.codes, (
            f"{case_id} 期望 {want}，实际 {report.codes}"
        )


def test_all_cases_loaded():
    """用例数量必须是 7 组 26 例（正 8 / 反 18）。

    数字写死是故意的：向量文件被误删一组、或新增组时忘了更新 summary，
    这里会立刻红。读 summary 再比对 summary 是自证，比不出东西。
    """
    assert len(CASES) == 26, f"用例数 {len(CASES)}，应为 26"
    positives = [c for c in CASES.values() if c.get("kind") == "positive"]
    negatives = [c for c in CASES.values() if c.get("kind") == "negative"]
    assert len(positives) == 8
    assert len(negatives) == 18


def test_every_negative_has_unique_expected_code():
    """每个反例的期望码必须是已定义的稳定码（TL-10.5-1）。"""
    from tlabel.v3.schema import ERROR_CODES

    for cid, case in CASES.items():
        if case.get("kind") != "negative":
            continue
        code = (case.get("expect") or {}).get("code")
        assert code in ERROR_CODES, f"{cid} 的期望码 {code} 不在稳定码表中"


# ─────────────────────────────────────────────────────────────
# 对照断言：把错误改对之后，错误必须消失
# 这些断言防的是「校验器恒真」——如果哪条检查被写坏成永远报错，
# 正例测试会先红；如果被写坏成永远不报，反例测试会红。
# 这一组额外防「报错了但报的不是这件事」。
# ─────────────────────────────────────────────────────────────

SURFACE_2D = {
    "frame_id": "surface_0",
    "dimension": 2,
    "frame_type": "sensor_local",
    "handedness": "right",
    "axes": {"x": "tangential_1", "y": "tangential_2"},
    "reference_object": "sensor_sensitive_surface",
}


def _with_surface_2d(case: Dict[str, Any]) -> Dict[str, Any]:
    c = copy.deepcopy(case)
    c["input"]["dataset"]["frames_of_reference"] = list(
        c["input"]["dataset"]["frames_of_reference"]
    ) + [SURFACE_2D]
    return c


def test_centroid_dimension_error_disappears_when_frame_fixed():
    c = _with_surface_2d(CASES["G5-N1"])
    c["input"]["frame"]["contact_centroid_frame"] = "surface_0"
    assert "FRAME_DIMENSION_MISMATCH" not in _run(c).codes


def test_slip_plane_error_disappears_when_frame_fixed():
    c = _with_surface_2d(CASES["G5-N3"])
    c["input"]["frame"]["slip_velocity_frame"] = "surface_0"
    assert "FRAME_OUT_OF_PLANE" not in _run(c).codes


def test_normal_force_mismatch_disappears_when_value_corrected():
    c = copy.deepcopy(CASES["G1-N1"])
    c["input"]["frame"]["normal_force"] = 2.196
    assert "NORMAL_FORCE_MISMATCH" not in _run(c).codes


def test_normal_vector_missing_disappears_when_supplied():
    c = copy.deepcopy(CASES["G1-N2"])
    prov = c["input"]["frame"]["field_provenance"]["normal_force"]
    prov["normal_vector"] = [0.8, 0.0, 0.6]
    prov["normal_vector_frame"] = "sensor_0"
    c["input"]["frame"]["normal_force"] = 2.196
    assert "NORMAL_VECTOR_MISSING" not in _run(c).codes


def test_provenance_type_error_disappears_when_allowed():
    c = copy.deepcopy(CASES["G2-N3"])
    c["input"]["frame"]["field_provenance"]["slip_event"] = {
        "type": "model_inference",
        "method_id": "m:v1",
    }
    assert "PROVENANCE_TYPE_FORBIDDEN" not in _run(c).codes


def test_null_status_error_disappears_when_status_removed():
    """G4-P1 是正例；删掉 field_status 后必须变成 NULL_WITHOUT_STATUS。"""
    c = copy.deepcopy(CASES["G4-P1"])
    c["input"]["frame"].pop("field_status", None)
    assert "NULL_WITHOUT_STATUS" in _run(c).codes


def test_positive_breaks_when_force_becomes_norm():
    """G1-P1 正例；把 normal_force 换成 ‖F‖ 必须被抓。"""
    c = copy.deepcopy(CASES["G1-P1"])
    c["input"]["frame"]["normal_force"] = 3.13
    assert "NORMAL_FORCE_MISMATCH" in _run(c).codes


def test_migration_state_error_disappears_when_audit_present():
    c = copy.deepcopy(CASES["G6-N2"])
    c["input"]["dataset"]["migration"] = {
        "tool": "tlabel-migrate",
        "tool_version": "1.0.0",
        "source_spec_version": "2.0.0",
        "audit": [],
    }
    assert "MIGRATION_STATE_IN_NATIVE" not in _run(c).codes


# ─────────────────────────────────────────────────────────────
# 状态与能力的一致性规则（TL-7.2-4 / TL-7.2-5）
# ─────────────────────────────────────────────────────────────

def test_extension_states_recognized_on_core_fields():
    """TL-7.2-4：Core 校验器必须识别 not_annotated / not_applicable。"""
    c = copy.deepcopy(CASES["G4-P1"])
    c["input"]["frame"]["field_status"] = {"normal_force": "not_annotated"}
    report = _run(c)
    assert "ENUM_INVALID" not in report.codes, (
        f"not_annotated 被当成未知枚举：{report.codes}"
    )
    assert report.ok


def test_extension_state_with_value_rejected():
    """TL-7.2-5：声明 not_annotated 却带非 null 值必须拒绝。"""
    c = copy.deepcopy(CASES["G4-P1"])
    c["input"]["frame"]["field_status"] = {"normal_force": "not_annotated"}
    c["input"]["frame"]["normal_force"] = 3.2
    assert "STATUS_WITHOUT_NULL" in _run(c).codes
