"""
TLabel Schema Conformance Test Suite

Validates that TLabelData output conforms to the JSON schema (tlabel-schema.json).
Run with: pytest tests/conformance/test_schema_validation.py
"""

import json
import pytest
from pathlib import Path

from tlabel.core.types import TLabelData, TLabelFrame
from tlabel.core.schema import TLabelSchemaV2, SCHEMA_V2_FIELD_NAMES


SCHEMA_PATH = Path(__file__).parent.parent.parent.parent / "schema" / "tlabel-schema.json"


@pytest.fixture
def schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture
def sample_data():
    """Create a minimal valid TLabelData instance."""
    frames = [
        TLabelFrame(
            frame_idx=0,
            timestamp_s=0.0,
            schema_v2=TLabelSchemaV2(
                contact=True,
                force_magnitude=0.234,
                slip_event=False,
                contact_centroid=[0.52, 0.48],
                contact_region="palmar",
                object_deformation=0.234,
                confidence=0.95,
                compliance_level="L2",
            ),
            manipulation_phase="stable_contact",
            confidence=0.95,
        )
    ]
    return TLabelData(
        frames=frames,
        sensor_info={
            "sensor_name": "TestSensor",
            "sensor_type": "vision_based",
            "adapter_name": "test_adapter",
            "adapter_version": "0.8.0",
        },
        episode_info={"episode_id": "test_ep", "task": "test_task"},
        capabilities={"contact": True},
        schema_version="0.17.0",
        sensor_id="test_sensor_01",
    )


def test_schema_file_exists():
    """Schema file must exist."""
    assert SCHEMA_PATH.exists(), f"Schema not found at {SCHEMA_PATH}"


def test_schema_is_valid_json():
    """Schema must be valid JSON."""
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    assert "$schema" in schema
    assert schema["title"] == "TLabel Schema V2.1"


def test_tlabel_data_conforms_to_schema(sample_data, schema):
    """TLabelData.to_dict() output must validate against the JSON schema.

    Note: The schema file contains OpenAPI-style extensions (required, unit,
    required_when inside property definitions) which are not strictly valid
    JSON Schema. We use a lenient validator that ignores unknown keywords.
    """
    try:
        import jsonschema
    except ImportError:
        pytest.skip("jsonschema not installed")

    data_dict = sample_data.to_dict()

    # Strip non-standard keywords from schema for strict JSON Schema validation
    # The schema uses OpenAPI-style extensions; strip them for jsonschema compat
    def _strip_extensions(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if k in ("required", "required_when", "unit"):
                    # Skip OpenAPI-style extensions inside property definitions
                    # (but keep 'required' at the schema/object level where it's valid)
                    if k == "required" and isinstance(v, list):
                        cleaned[k] = _strip_extensions(v)
                    continue
                cleaned[k] = _strip_extensions(v)
            return cleaned
        elif isinstance(obj, list):
            return [_strip_extensions(item) for item in obj]
        return obj

    clean_schema = _strip_extensions(schema)
    jsonschema.validate(instance=data_dict, schema=clean_schema)


def test_all_14_dimensions_present(sample_data):
    """Every frame must have all 14 Schema V2 dimensions."""
    for frame in sample_data.frames:
        sv2_dict = frame.schema_v2.to_dict()
        for field_name in SCHEMA_V2_FIELD_NAMES:
            assert field_name in sv2_dict, (
                f"Missing Schema V2 field: {field_name}"
            )
        assert len(sv2_dict) == 14, (
            f"Expected 14 dimensions, got {len(sv2_dict)}"
        )


def test_required_top_level_fields(sample_data):
    """Top-level output must have all required fields."""
    d = sample_data.to_dict()
    required = ["schema_version", "format", "sensor", "capabilities", "frames"]
    for field in required:
        assert field in d, f"Missing required field: {field}"


def test_frame_structure(sample_data):
    """Each frame must have required fields."""
    d = sample_data.to_dict()
    valid_phases = {
        "idle", "initial_contact", "stable_contact", "slip",
        "release", "re_contact", "approach", "retract",
        "grasp", "transport", "hold"
    }
    for frame in d["frames"]:
        assert "frame_idx" in frame
        assert "timestamp_s" in frame
        assert "schema_v2" in frame
        assert "manipulation_phase" in frame
        assert frame["manipulation_phase"] in valid_phases


def test_cascade_contact_zero():
    """Setting contact=0 should clear other force fields to None (not 0.0).

    P9/T0-4：0.0 表示"测得零力"，无接触帧上力不可测，正确表达是 None。
    """
    frame = TLabelFrame(
        frame_idx=0, timestamp_s=0.0,
        schema_v2=TLabelSchemaV2(
            contact=True,
            force_magnitude=0.5,
            slip_event=True,
            object_deformation=0.5,
            contact_centroid=[0.5, 0.5],
            confidence=0.9,
            compliance_level="L2",
        ),
        manipulation_phase="stable_contact",
    )
    frame.patch("contact", 0.0, cascade=True)

    assert frame.schema_v2.force_magnitude is None
    # v0.26（D8）：接触结束后滑移不再有意义，但也不能断言「没滑」。
    # 规范 TL-5.6-3 要求无法确定滑移时必须为 null，不得用 false 表示「未测量」。
    assert frame.schema_v2.slip_event is None
    assert frame.schema_v2.object_deformation is None
    assert frame.manipulation_phase == "idle"
    assert len(frame.patches) == 1
    assert len(frame.patches[0]["cascade"]) > 0


def test_cascade_slip_without_contact():
    """Setting slip_event > 0.5 without contact should auto-set contact=1."""
    frame = TLabelFrame(
        frame_idx=0, timestamp_s=0.0,
        schema_v2=TLabelSchemaV2(
            contact=False, slip_event=False, confidence=0.9,
        ),
        manipulation_phase="idle",
    )
    frame.patch("slip_event", 0.8, cascade=True)

    assert frame.schema_v2.contact is True
    assert frame.manipulation_phase == "slip"


def test_cascade_force_without_contact():
    """Setting force > 0 without contact should auto-set contact=1."""
    frame = TLabelFrame(
        frame_idx=0, timestamp_s=0.0,
        schema_v2=TLabelSchemaV2(
            contact=False, force_magnitude=None, slip_event=False, confidence=0.9,
        ),
        manipulation_phase="idle",
    )
    frame.patch("force_magnitude", 0.5, cascade=True)

    assert frame.schema_v2.contact is True


def test_feature_metadata_completeness():
    """Feature metadata must cover all 22 dimensions (v1 compat)."""
    from tlabel.features_meta import FEATURE_REGISTRY

    expected_dims = [
        "contact", "deformation_magnitude", "force_magnitude", "force_peak",
        "force_direction", "slip_entropy", "slip_event", "texture_energy",
        "edge_density", "contact_area", "centroid_x",
        "normal_field_magnitude", "normal_field_variance",
        "shear_field_magnitude", "shear_field_direction",
        "delta_force_normal", "delta_force_shear", "friction_cone_ratio",
        "optical_flow_magnitude", "optical_flow_direction",
        "temporal_deformation_rate", "contact_transition",
    ]
    for dim in expected_dims:
        assert dim in FEATURE_REGISTRY, f"Missing metadata for: {dim}"
        meta = FEATURE_REGISTRY[dim]
        assert "feature_id" in meta
        assert "category" in meta
        assert meta["category"] in ["deformation", "gradient", "force_semantic", "temporal"]


def test_force_magnitude_deprecated():
    """force_magnitude must be marked deprecated in metadata."""
    from tlabel.features_meta import FEATURE_REGISTRY

    meta = FEATURE_REGISTRY["force_magnitude"]
    assert meta["deprecated"] is True
    assert meta.get("deprecated_since") == "0.7.0"
    assert meta.get("replacement") == "deformation_magnitude_peak"
