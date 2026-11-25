"""
视觉触觉 adapter 的形变语义（v0.26 / V3.0 对齐）

背景：
    5 个 adapter 把**图像强度 RMS**（无量纲）填进了 `object_deformation`
    这个单位为 mm 的字段。相机看到的是传感器弹性体的亮度变化，
    不是物体被压下去多少毫米。

规范依据（V3.0）：
    TL-6.4-1  object_deformation MUST NOT be reported unless the deformed
              object is identified. Deformation of the sensor's elastomer
              is NOT object deformation.
    TL-6.4-2  A producer that measures sensor elastomer deformation MUST
              record it as a derived feature in `sensor_specific`,
              not as `object_deformation`.

因此本文件锁三件事：
    1. 这 5 个 adapter 不再声明 object_deformation 能力
    2. 原值仍然可取（搬进 sensor_specific.deformation_image_rms）
    3. 原值带 unit / method_id / namespace，不裸放
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ADAPTERS = [
    ("tlabel/adapters/gelsight.py", "GelSightAdapter", "gelsight"),
    ("tlabel/adapters/vtouch.py", "VTouchAdapter", "vtouch"),
    ("tlabel/adapters/daimon_dataset.py", "DaimonAdapter", "daimon"),
    ("tlabel/adapters/daimon_dm_tac.py", "DaimonDmTacAdapter", "daimon_dm_tac"),
    ("tlabel/adapters/ycb_slide.py", "YCBSlideAdapter", "ycb_slide"),
]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_class(module_path: str, class_name: str):
    import importlib

    mod_name = module_path.replace("/", ".").removesuffix(".py")
    mod = importlib.import_module(mod_name)
    return getattr(mod, class_name)


@pytest.mark.parametrize("module_path,class_name,label", ADAPTERS)
def test_deformation_capability_not_declared(module_path, class_name, label):
    """图像 RMS 不是毫米，不得声明 object_deformation 能力。"""
    cls = _load_class(module_path, class_name)
    caps = cls().get_capabilities()
    assert caps.get("object_deformation") is False, (
        f"{label}: 仍声明 object_deformation 能力。"
        "相机看到的是弹性体亮度变化，不是物体形变（TL-6.4-1）。"
    )


@pytest.mark.parametrize("module_path,class_name,label", ADAPTERS)
def test_raw_rms_moved_to_sensor_specific(module_path, class_name, label):
    """原值必须还在，且带单位与方法标识——不能因为修正语义就把数据丢了。"""
    src = (REPO_ROOT / module_path).read_text(encoding="utf-8")
    assert "deformation_image_rms" in src, (
        f"{label}: 找不到 deformation_image_rms，原值可能被直接丢弃了。"
    )
    # 结构化三件套
    for key in ('"unit"', '"method_id"', '"namespace"'):
        assert key in src, f"{label}: sensor_specific 条目缺少 {key}"
    assert '"1"' in src, f"{label}: 无量纲必须显式写 unit=\"1\"，不能留空让下游猜"


@pytest.mark.parametrize("module_path,class_name,label", ADAPTERS)
def test_schema_does_not_assign_object_deformation(module_path, class_name, label):
    """extract_schema 里不得再给 object_deformation 赋值（只能留 None 或不出现）。"""
    src = (REPO_ROOT / module_path).read_text(encoding="utf-8")
    # 允许 `object_deformation=None`，不允许 `object_deformation=<变量>`
    bad = re.findall(r"object_deformation\s*=\s*(?!None\b)[A-Za-z_(]", src)
    assert not bad, (
        f"{label}: 仍在给 object_deformation 赋非 None 值：{bad}。"
        "形变量应进 sensor_specific.deformation_image_rms。"
    )


# ─────────────────────────────────────────────────────────────
# CSV 往返：未知不能变成「确定没有」
# ─────────────────────────────────────────────────────────────

class TestTristateRoundTrip:
    """CSV 导出把未知写成空字段；读回时空串必须还原成 None。"""

    def test_empty_string_is_unknown_not_false(self):
        from tlabel.core.schema import _tristate

        assert _tristate("") is None
        assert _tristate("   ") is None

    def test_numeric_zero_string_is_false(self):
        from tlabel.core.schema import _tristate

        # bool("0") == True，这是反向错误：CSV 里的 0 必须是 false
        assert _tristate("0") is False
        assert _tristate("0.0") is False
        assert _tristate("1") is True

    def test_word_forms(self):
        from tlabel.core.schema import _tristate

        assert _tristate("true") is True
        assert _tristate("False") is False

    def test_garbage_rejected(self):
        import pytest as _pytest

        from tlabel.core.schema import _tristate

        with _pytest.raises(ValueError, match="三态布尔值"):
            _tristate("maybe")

    def test_csv_roundtrip_keeps_unknown(self, tmp_path):
        import csv as _csv

        from tlabel.core.schema import TLabelSchemaV2
        from tlabel.core.types import TLabelData, TLabelFrame
        from tlabel.export.writer import export_data

        frames = [
            TLabelFrame(frame_idx=i, timestamp_s=i * 0.1,
                        schema_v2=TLabelSchemaV2(contact=c, confidence=0.9))
            for i, c in enumerate((True, False, None))
        ]
        data = TLabelData(frames=frames, sensor_info={"type": "vision-based_tactile"},
                          episode_info={"episode_id": "e1"}, capabilities={"contact": True})
        p = tmp_path / "o.csv"
        export_data(data, str(p), format="csv")

        rows = list(_csv.DictReader(open(p)))
        got = [TLabelSchemaV2.from_dict({"contact": r["contact"]}).contact for r in rows]
        assert got == [True, False, None], f"CSV 往返丢了三态：{got}"
