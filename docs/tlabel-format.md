# TLabel Format Specification

> **📌 Informative document — not the normative specification.** This is a
> tutorial-style walkthrough. The normative source of truth for the TLabel data
> format is **[`docs/spec/tlabel-spec-v1.md`](spec/tlabel-spec-v1.md)** (Schema
> V2, 14 dimensions). Where this document disagrees with that specification or
> with the code, the specification governs (see its §11 for the conflict list).

**Version:** 2.1.0  
**Status:** Draft  
**Last Updated:** 2026-07-24

---

## Overview

TLabel Format is a cross-sensor tactile annotation schema that uses capability declarations and Compliance Level stratification to manage sensor heterogeneity. Instead of forcing all sensors into a single representation, TLabel allows each sensor adapter to declare which semantic dimensions it can annotate, producing compatible but not identical output.

## Core Design Principles

1. **Capability Declaration**: Each adapter explicitly declares which dimensions it supports. Unsupported dimensions are omitted from output — never fabricated.
2. **Compliance Level Stratification**: Four levels (L1–L4) let sensors of different physical capabilities all participate at their appropriate information density.
3. **Semantic Level Unification**: TLabel unifies at the annotation level (contact, force, slip), not at the raw signal level. Different sensors capture different aspects of the same phenomena.
4. **Schema-Versioned**: All files carry a `schema_version` field, enabling backward-compatible evolution.

## Schema Structure

### Top-Level

```json
{
  "schema_version": "2.1.0",
  "sensor_info": { ... },
  "capabilities": { ... },
  "episodes": [ ... ]
}
```

### sensor_info

Describes the sensor hardware and adapter software:

```json
{
  "sensor_info": {
    "sensor_name": "Daimon-Infinity",
    "sensor_type": "vision_based",
    "manufacturer": "Daimon",
    "model": "Infinity",
    "resolution": "640x480",
    "frame_rate": 30,
    "adapter_name": "tlabel-daimon",
    "adapter_version": "2.1.0"
  }
}
```

### capabilities

Boolean declarations for each semantic dimension:

| Dimension | Type | Description |
|-----------|------|-------------|
| `contact` | bool | Binary contact detection |
| `contact_centroid` | bool | Contact center coordinate [x, y] |
| `contact_region` | bool | Coarse-grained contact region label |
| `force_magnitude` | bool | Normal contact force scalar (N) |
| `force_vector` | bool | 3D contact force vector [Fx, Fy, Fz] (N) — L3+ |
| `torque_vector` | bool | 3D torque vector [Mx, My, Mz] (N·m) |
| `slip_event` | bool | Binary slip detection |
| `slip_velocity` | bool | Slip velocity vector [vx, vy] (mm/s) |
| `manipulation_phase` | bool | Phase classification |
| `texture_class` | bool | Surface texture classification |
| `object_deformation` | bool | Object deformation measurement |
| `temperature` | bool | Contact surface temperature (°C) |
| `confidence` | bool | Annotation confidence score |

```json
{
  "capabilities": {
    "contact": true,
    "contact_centroid": true,
    "contact_region": true,
    "force_magnitude": true,
    "force_vector": false,
    "torque_vector": false,
    "slip_event": true,
    "slip_velocity": true,
    "manipulation_phase": true,
    "texture_class": true,
    "object_deformation": true,
    "temperature": false,
    "confidence": true
  }
}
```

### episodes

Array of annotated episodes. Each episode contains metadata and per-frame annotations:

```json
{
  "episodes": [
    {
      "episode_id": "ep_001",
      "task": "pick_place",
      "object": "plastic_bottle",
      "metadata": {
        "duration_s": 12.5,
        "num_frames": 375
      },
      "frames": [ ... ]
    }
  ]
}
```

### Per-Frame Annotation (14 Dimensions)

Only dimensions declared in `capabilities` appear in frame output:

```json
{
  "frame_idx": 42,
  "timestamp_s": 1.4,
  "contact": true,
  "contact_centroid": [310, 245],
  "contact_region": "fingertip_left",
  "force_magnitude": 1.2,
  "force_vector": null,
  "torque_vector": null,
  "slip_event": false,
  "slip_velocity": null,
  "manipulation_phase": "hold",
  "texture_class": "smooth_plastic",
  "object_deformation": 0.3,
  "temperature": null,
  "confidence": 0.92,
  "compliance_level": "L2"
}
```

## The 14 Semantic Dimensions

### 1. contact (Required)
- **Type**: boolean
- **True when**: Any detectable contact between sensor and object
- **False when**: No contact or signal below noise floor

### 2. contact_centroid (Required if contact=true)
- **Type**: [float, float] or null
- **Format**: Contact center coordinate in sensor-local frame (pixels or mm)
- **null when**: contact=false or centroid cannot be determined

### 3. contact_region (Optional)
- **Type**: string (enum) or null
- **Values**: Sensor-specific regions (e.g., "fingertip_left", "palm_center", "taxel_group_thumb")
- **Note**: Region names are sensor-dependent; TLabel does not enforce a universal spatial vocabulary

### 4. force_magnitude (Required at L2+)
- **Type**: float ≥ 0 or null
- **Unit**: Newtons (N)
- **Description**: Normal contact force scalar
- **null when**: Sensor cannot measure force

### 5. force_vector (Optional, L3+)
- **Type**: [float, float, float] or null
- **Unit**: Newtons (N)
- **Format**: 3D contact force [Fx, Fy, Fz] in sensor-local coordinate frame
- **null when**: Sensor cannot measure 3D contact force

### 6. torque_vector (Optional)
- **Type**: [float, float, float] or null
- **Unit**: Newton-meters (N·m)
- **Format**: 3D torque [Mx, My, Mz] in sensor-local frame
- **null when**: Not available

### 7. slip_event (Required)
- **Type**: boolean
- **True when**: Detectable relative motion between sensor surface and contact object
- **Detection method**: Sensor-specific (optical flow for vision-based, threshold for array)

### 8. slip_velocity (Optional if slip_event=true)
- **Type**: [float, float] or null
- **Unit**: mm/s
- **Format**: Slip velocity vector [vx, vy] in sensor-local tangential frame
- **null when**: No slip detected or sensor cannot measure slip velocity

### 9. manipulation_phase (Optional)
- **Type**: string (enum) or null
- **Values**: "idle" | "approach" | "grasp" | "contact" | "lift" | "hold" | "translate" | "place" | "release" | "retract"
- **Transition rules**: Sequential; "hold" may repeat; "release" may transition to "approach"

### 10. texture_class (Optional)
- **Type**: string (enum) or null
- **Values**: "smooth" | "rough" | "granular" | "fibrous" | "ridged" | "slimy" | "sticky" | "hard" | "soft" | "smooth_plastic" | "rough_cloth" | "metal_grid" | "rubber" | "wood" | "glass" | "ceramic"
- **Note**: Standardized enum for cross-sensor consistency

### 11. object_deformation (Optional)
- **Type**: float or null
- **Unit**: mm or ratio
- **Description**: Object deformation magnitude
- **null when**: Sensor cannot detect deformation

### 12. temperature (Optional)
- **Type**: float or null
- **Unit**: °C
- **Description**: Contact surface temperature
- **null when**: Sensor cannot measure temperature

### 13. confidence (Required)
- **Type**: float [0.0, 1.0]
- **Description**: Annotation confidence score. 1.0 = fully confident, 0.0 = no confidence.
- **Purpose**: Data provenance and quality tracking

### 14. compliance_level (Required)
- **Type**: string enum: "L1" | "L2" | "L3" | "L4"
- **Description**: Compliance Level indicating the data information density and capability of the sensor for this frame

## Compliance Level

TLabel uses Compliance Level stratification so sensors with different physical capabilities can all participate at their appropriate information density.

### Level Definitions

| Level | Name | Required Fields | Typical Sensors |
|-------|------|----------------|-----------------|
| **L1 Basic** | Basic Tactile | contact, contact_centroid, slip_event, confidence | All sensors (single-point resistive, proximity, etc.) |
| **L2 Force-Aware** | Force Sensing | L1 + **force_magnitude** (normal force scalar) | Paxini, YCB-Slide, DM-TAC, GelSight (with calibration) |
| **L3 Full-Vector** | Complete Force Vector | L2 + **force_vector** [Fx, Fy, Fz] | ToucHD, calibrated DM-TAC/GelSight, BioTac |
| **L4 Rich-Semantic** | Full Semantics | L3 + all Optional fields populated (torque, texture, temperature, etc.) | BioTac, next-gen multimodal sensors |

### Design Principles

1. **Cumulative**: L3 satisfies all L2 and L1 requirements; L4 satisfies all L3, L2, and L1.
2. **Physical reality first**: A sensor that physically cannot measure shear force is correctly L2, not "non-compliant."
3. **Downstream-transparent**: The `compliance_level` field lets models and algorithms know the data boundary.
4. **Adapter-driven**: Each adapter auto-sets `compliance_level` based on sensor capability; no user input required.

### Adapter → Compliance Level Mapping

| Adapter | Level | Rationale |
|---------|-------|-----------|
| Paxini Gen3 (real-time) | L2 | Only total_force_n (normal scalar) |
| Paxini Dataset | L2 | Same as above |
| YCB-Slide | L2 | deformation_mag approximates normal force |
| DM-TAC (real-time) | L2–L3 | Can synthesize 3D force from deformation+shear (requires calibration) |
| GelSight (dataset) | L2–L3 | Visuo-tactile; force info extractable from images |
| Daimon Dataset | L2–L3 | deformation/shear/depth video streams |
| ToucHD | L3 | Ground-truth 3D force labels (Fx, Fy, Fz) |
| UnivTac | L2–L3 | Has force_magnitude + shear |
| TacQuad | L1–L2 | Tac3D has approximate force; others only images |
| VTouch | L2 | Has force information |

> **Note**: L2–L3 means the adapter may choose L2 or L3 depending on calibration status. Default is L2 when uncalibrated.

## Sensor Type Categories

TLabel recognizes three broad sensor categories:

| Category | Typical Capabilities | Examples |
|----------|---------------------|----------|
| **vision_based** | contact, contact_centroid, contact_region, slip_event, slip_velocity, texture_class, manipulation_phase, object_deformation | GelSight, DIGIT, Daimon-Infinity |
| **distributed_array** | contact, contact_centroid, contact_region, force_magnitude, slip_event, manipulation_phase | PaXini PXCap, tactile gloves |
| **hybrid** | All dimensions (potentially) | BioTac, next-generation multi-modal sensors |

## Validation Rules

A TLabel annotation file is valid if and only if:

1. `schema_version` is present and matches `^2\.1\.0$`
2. `capabilities` declares at least 1 dimension as `true`
3. Every Required field in `frames` is present and non-null:
   - `contact`, `slip_event`, `confidence`, `compliance_level` — always required
   - `contact_centroid` — required when `contact=true`
4. No field appears in `frames` that is declared `false` in `capabilities` (must be `null`)
5. `force_magnitude` values are ≥ 0
6. `force_vector` (when present and non-null) is a 3-element number array
7. `slip_velocity` (when present and non-null) is a 2-element number array
8. `manipulation_phase` values are from the defined enum
9. `compliance_level` is one of: L1, L2, L3, L4
10. Compliance Level consistency: if `compliance_level` = L2, then `force_magnitude` must be non-null; if L3, `force_vector` must also be non-null

## Versioning

TLabel follows semantic versioning:
- **Major**: Breaking changes to schema structure
- **Minor**: New optional dimensions or fields
- **Patch**: Documentation or clarification updates

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.2.0 | 2026-05-27 | Initial draft with 10 dimensions |
| 2.0.0 | 2026-07-23 | Three-layer architecture; 10→12 dimensions; contact_centroid, torque_vector, temperature, confidence added; force_direction→force_vector; slip_direction→slip_velocity; texture→texture_class; whole_hand_coordination removed from per-frame |
| 2.1.0 | 2026-07-24 | Compliance Level (L1–L4); force_vector downgraded to Optional L3+; force_magnitude added (L2+ Required); compliance_level added (Required); 12→14 dimensions |

## Relationship to Other Standards

| Standard | Level | TLabel's Relationship |
|----------|-------|----------------------|
| LeRobot | Raw data format | TLabel annotations can augment LeRobot episodes |
| Open X-Embodiment | Task-level metadata | TLabel provides per-frame tactile detail |
| RoboMimic | Demonstration format | TLabel annotations are compatible with RoboMimic HDF5 |

---

*This specification is released under MIT License. Feedback and contributions welcome.*
