# TLabel Annotation Specification

> **📌 Informative document — not the normative specification.** This is an
> annotation-guidelines narrative. The normative source of truth for the TLabel
> data format is **[`docs/spec/tlabel-spec-v1.md`](spec/tlabel-spec-v1.md)**
> (Schema V2, 14 dimensions). Where this document disagrees with that
> specification or with the code, the specification governs (see its §11 for
> the conflict list).

**Version:** 2.1.0  
**Status:** Draft  
**Last Updated:** 2026-07-24

---

## Overview

This document describes the annotation methodology used by TLabel adapters. It covers the general approach for each of the 14 semantic dimensions, but does **not** include sensor-specific thresholds or parameters — those are internal to each adapter implementation.

## General Annotation Flow

```
Raw Sensor Data → Preprocessing → Dimension-wise Annotation → Compliance Level Assignment → TLabel JSON Output
```

1. **Preprocessing**: Each adapter applies sensor-specific normalization (e.g., baseline subtraction for vision-based, calibration for force arrays)
2. **Dimension-wise Annotation**: Each supported dimension is annotated independently using the methods described below
3. **Compliance Level Assignment**: Adapter determines L1–L4 based on sensor physical capability
4. **Capability Filtering**: Only dimensions declared in `capabilities` are included in the output

The `capabilities` object is a **closed set of exactly 13 boolean keys** (S-7.4-1).
`compliance_level` is frame-level metadata and is **not** a capability key — it must
not appear in `capabilities` (S-7.4-2). `contact` is required. Annotators must not
declare a dimension `true` unless the emitted frames actually carry a non-null value
for it (S-6.3-3); the declaration is a statement about file content, not hardware
potential.

## Annotation Methods by Dimension

### 1. Contact Detection (Required)

**Vision-based sensors:**
- Method: Pixel deviation from baseline (no-contact reference frame)
- Output: Boolean per frame

**Distributed array sensors:**
- Method: Force threshold on individual taxels
- Output: Boolean per frame

### 2. Contact Centroid (Required if contact=true)

**Vision-based sensors:**
- Method: Center of mass of the contact region pixel mask
- Output: [x, y] in sensor pixel coordinates

**Distributed array sensors:**
- Method: Weighted centroid of activated taxels
- Output: [x, y] in taxel grid coordinates

### 3. Contact Region (Optional)

**Vision-based sensors:**
- Method: Connected component analysis on contact mask
- Output: Region label (sensor-specific vocabulary)

**Distributed array sensors:**
- Method: Taxel group identification based on spatial mapping
- Output: Region label based on taxel layout

### 4. Force Magnitude (Required at L2+)

**Vision-based sensors:**
- Method: Photometric stereo reconstruction → depth map → contact force estimation
- Output: Float in Newtons (N)
- **Note**: Indirect estimation; precision limited by optical properties and calibration

**Distributed array sensors:**
- Method: Direct force measurement from taxel readings
- Output: Float in Newtons (N)
- **Note**: More reliable than vision-based estimation

**v2.1 Change**: Force magnitude is now in absolute Newtons (not normalized [0,1]). This enables cross-sensor force comparison without external calibration metadata.

### 5. Force Vector (Optional, L3+)

**Vision-based sensors:**
- Method: Displacement field analysis on contact region → 3D force synthesis
- Output: [Fx, Fy, Fz] in sensor-local frame (N)
- **Note**: Requires calibration; some adapters produce only normal force

**Distributed array sensors:**
- Method: 6-axis force/torque computation from distributed taxel readings
- Output: [Fx, Fy, Fz] in sensor-local frame (N)
- **Note**: Only available for sensors with shear force channels

**v2.1 Change**: Downgraded from Required to Optional L3+. Many sensors (e.g., Paxini Gen3) physically cannot measure shear force; forcing force_vector was causing invalid annotations.

### 6. Torque Vector (Optional)

**Vision-based sensors:**
- Typically **not supported** — requires multi-point contact model
- **Note**: May be available in multi-finger setups

**Distributed array sensors:**
- Method: Cross-taxel moment computation
- Output: [Mx, My, Mz] in sensor-local frame (N·m)
- **Note**: Only available for high-density arrays with known spatial layout

### 7. Slip Detection (Required)

**Vision-based sensors:**
- Method: Optical flow analysis on contact region between consecutive frames
- Output: Boolean (slip detected)

**Distributed array sensors:**
- Method: Temporal force pattern analysis (shear force change rate)
- Output: Boolean (slip detected)

### 8. Slip Velocity (Optional if slip_event=true)

**Vision-based sensors:**
- Method: Optical flow magnitude and direction in tangential plane
- Output: [vx, vy] in mm/s

**Distributed array sensors:**
- Method: Force rate-of-change projected onto tangential velocity
- Output: [vx, vy] in mm/s
- **Note**: Less precise than vision-based; many array sensors cannot measure this

**v2.1 Change**: Replaces `slip_direction` (unit vector). `slip_velocity` includes both direction and magnitude, providing more complete physical information.

### 9. Manipulation Phase Classification (Optional)

**All sensors:**
- Method: Rule-based state machine using contact and force signals
- Phases: idle → approach → grasp → contact → lift → hold → translate → place → release → retract
- Transitions determined by:
  - Contact onset/offset
  - Force magnitude thresholds
  - Temporal stability of contact state

**v2.1 Change**: Added `idle`, `grasp`, `translate`, `retract` phases for richer manipulation modeling.

### 10. Texture Classification (Optional)

**Vision-based sensors:**
- Method: Surface feature analysis on contact region (spatial frequency, edge density)
- Output: Enum label from standardized taxonomy

**Distributed array sensors:**
- Method: Force distribution pattern matching
- Output: Enum label from standardized taxonomy
- **Note**: Limited texture resolution compared to vision-based sensors

**v2.1 Change**: `texture` (free-form string) replaced by `texture_class` (standardized enum) for cross-sensor consistency.

### 11. Object Deformation (Optional)

**Vision-based sensors:**
- Method: Contact region shape change analysis across frames
- Output: Deformation magnitude (mm or ratio)

**Distributed array sensors:**
- Typically **not supported** — force arrays measure external force, not object response
- Exception: High-density arrays with known object stiffness priors

**v2.1 Change**: Changed from object type to float for simpler cross-sensor representation.

### 12. Temperature (Optional)

**Vision-based sensors:**
- Typically **not supported** — cameras cannot measure contact temperature

**Distributed array sensors with thermal channels:**
- Method: On-board temperature sensor at contact point
- Output: Temperature in °C

**v2.1 Change**: New optional dimension for multi-modal sensors.

### 13. Confidence (Required)

**All sensors:**
- Method: Adapter-specific quality scoring based on signal-to-noise ratio, model prediction certainty, or heuristic rules
- Output: Float [0.0, 1.0]
- **Guidelines**:
  - 0.9–1.0: High confidence, signal clearly above noise
  - 0.7–0.9: Moderate confidence, some uncertainty
  - 0.5–0.7: Low confidence, near detection threshold
  - < 0.5: Very low confidence, consider flagging for review

**v2.1 Change**: New required dimension for data provenance tracking.

### 14. Compliance Level (Required)

**All adapters:**
- Method: Automatically determined by adapter based on sensor physical capability
- Output: "L1" | "L2" | "L3" | "L4"
- **Determination rules**:
  - Sensor measures only contact → L1
  - Sensor additionally measures normal force → L2
  - Sensor additionally measures 3D force vector → L3
  - Sensor additionally populates all optional fields → L4
- **Note**: Users do not manually set compliance_level; the adapter assigns it

**v2.1 Change**: New required dimension for data capability transparency.

## Determining Force Magnitude

Force magnitude annotation requires careful handling:

1. **Sensors with direct force measurement** (Paxini, ToucHD):
   - Read force directly from sensor hardware
   - Apply calibration curve if needed
   - Report in Newtons

2. **Vision-based sensors** (GelSight, Daimon, DIGIT):
   - Estimate force from elastomer deformation depth
   - Requires calibration with known weights
   - Precision depends on optical resolution and elastomer properties
   - Report estimated force in Newtons with appropriate confidence

3. **Proxy measurements** (YCB-Slide deformation_mag):
   - Use deformation magnitude as force proxy
   - Conversion factor depends on elastomer stiffness
   - Report as approximate force with lower confidence

## Determining Compliance Level

To determine the correct compliance level for a sensor/adapter:

1. **Inventory sensor channels**: What physical quantities can this sensor measure?
   - Contact only → L1 candidate
   - Contact + normal force → L2 candidate
   - Contact + normal force + 3D force → L3 candidate
   - Contact + normal force + 3D force + all optional fields → L4 candidate

2. **Verify with test data**: Run a known manipulation episode and check:
   - Are L1 fields always populated? (contact, contact_centroid, slip_event, confidence)
   - Is force_magnitude non-null? → Confirm L2
   - Is force_vector non-null? → Confirm L3
   - Are texture_class, temperature, torque_vector all non-null? → Confirm L4

3. **Set in adapter**: The adapter hardcodes the maximum level based on sensor specs. If calibration enables higher levels, adapter may select dynamically.

## Validation and Quality Control

### Hard Error Check
A hard error occurs when an annotation contradicts observable sensor data:
- Contact marked `true` when sensor signal is at baseline
- Slip marked `true` when no relative motion is detectable
- Force magnitude is negative
- Compliance level claims L2 but force_magnitude is null

### Soft Anomaly Check
A soft anomaly occurs when an annotation is unusual but not impossible:
- Contact region unexpectedly large or small
- Manipulation phase transition out of normal sequence
- Force magnitude at extreme values
- Confidence < 0.5

### Quality Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Hard error rate | 0% | Must be zero for release |
| Soft anomaly rate | <1% | Flagged for manual review |
| Coverage | >95% | Fraction of frames with valid annotations |
| Compliance consistency | 100% | compliance_level must match populated fields |

## Adapter Implementation Guide

To create a new TLabel adapter:

1. **Determine compliance level** — based on sensor physical capability (L1–L4)
2. **Implement `capabilities` declaration** — define which dimensions your sensor supports
3. **Implement per-dimension annotators** — follow the methods above or develop sensor-specific alternatives
4. **Set confidence** — implement quality scoring for each frame
5. **Validate output** — run the validation rules from [tlabel-format.md](tlabel-format.md)
6. **Test on ground-truth episodes** — manually verify annotations on a small dataset
7. **Report metrics** — hard error rate, soft anomaly rate, coverage

## Limitations

- **Force magnitude precision varies**: Vision-based force estimation is less precise than direct measurement. Cross-sensor comparison requires understanding each sensor's precision limits.
- **No standardized texture taxonomy yet**: texture_class uses an initial enum, but the taxonomy will expand based on community feedback.
- **Manipulation phase model may need extension**: The current 10-phase model covers common pick-place scenarios. More complex operations (e.g., tool use, in-hand reorientation) may require additional phases.
- **Compliance Level is sensor-level, not frame-level**: Currently, all frames from a sensor share the same compliance_level. Per-frame level variation (e.g., when calibration degrades) is not yet supported.

---

*This specification is released under MIT License. Feedback and contributions welcome.*
