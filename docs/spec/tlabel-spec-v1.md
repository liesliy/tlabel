# TLabel Data Format Specification v1

| Field | Value |
|---|---|
| **Specification ID** | TLABEL-SPEC-V1 |
| **Title** | TLabel Data Format Specification v1 (Schema V2) |
| **Status** | Normative — this document is the single normative source of truth for the TLabel data format |
| **Schema version** | Schema V2, revision 2.1 (14 dimensions) |
| **Applies to package** | `tlabel` >= 0.17.0 (current reference release: 0.23.1) |
| **Supersedes** | The non-normative descriptions in `docs/tlabel-format.md`, `docs/annotation-spec.md`, `docs/DESIGN.md`, `docs/TLabel_Design_Document.md`, `docs/whitepaper.md` |
| **License** | MIT |

---

## Abstract

This document specifies the TLabel Data Format: a sensor-agnostic annotation
schema for robotic tactile and force sensing. The format defines a
**14-dimension semantic schema (Schema V2)** together with a four-level
**Compliance Level** mechanism (L1–L4) that lets sensors with different physical
capabilities participate in one interoperable standard at their appropriate
information density. This specification states normative requirements using
RFC 2119 keywords so that implementations can be tested for conformance.

The reference implementation lives in the `tlabel` package. Where this document
and the reference implementation differ, the difference is recorded in
[§11](#11-differences-from-existing-documentation), and the reference
implementation is authoritative.

---

## 1. Scope and Applicability

### 1.1 Scope

This specification defines:

1. the **document structure** of a TLabel annotation file;
2. the **14 Schema V2 dimensions**, each with its name, type, unit,
   requiredness, compliance-level gate, and example value;
3. the **Compliance Level rules** (L1–L4) that determine which dimensions MUST
   be populated;
4. the **conformance requirements** for adapters (Layer 2);
5. the **serialization and interoperability** rules;
6. the **versioning and backward-compatibility** policy; and
7. the **extension mechanism**.

### 1.2 Applicability

This specification applies to:

- any **producer** (adapter) that maps raw sensor or dataset data into TLabel
  Schema V2 annotations;
- any **consumer** (downstream model, dataset loader, validator, converter)
  that reads TLabel annotations;
- any **validator** used to check conformance of a TLabel file or adapter.

### 1.3 Out of scope

This specification does **not** define, and MUST NOT be interpreted to define:

- any feature-extraction method, model, or task-specific feature vector. Such
  artifacts belong to **Layer 3 (downstream derivation)** and are explicitly
  outside the standard;
- sensor-specific thresholds, calibration procedures, or signal-processing
  algorithms internal to an adapter;
- storage engines, transport protocols, or file-system layouts beyond the
  serialization rules in [§7](#7-serialization-and-interoperability).

> **Normative note.** TLabel standardizes *what the data means*, not *how it is
> computed*. Any statement that attributes a task-specific feature vector
> (e.g. a 18-element downstream vector) to the TLabel schema itself is
> non-conformant with this specification.

### 1.4 Conventions

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this
document are to be interpreted as described in RFC 2119.

- Normative clauses are numbered (e.g. `[S-4.3-2]`) so that conformance tests
  can cite them directly.
- `null` denotes the JSON null value and the Python `None` value.
- "Populated" means "not null and not absent".

### 1.5 Normative references (source of truth)

The following files in the reference repository are the machine-checkable
sources for this specification. In case of conflict, they take precedence over
prose, including this document:

| Reference | Path | Defines |
|---|---|---|
| Schema dataclass | `tlabel/core/schema.py` (`TLabelSchemaV2`, `SCHEMA_V2_FIELD_NAMES`) | field set, types, enums, `validate()` |
| Container types | `tlabel/core/types.py` (`TLabelData`, `TLabelFrame`) | frame/document structure, serialization |
| Adapter contract | `tlabel/adapters/base.py` (`DataAdapterBase`, `SensorAdapterBase`) | conformance interface |
| JSON Schema | `schema/tlabel-schema.json` | JSON-level document schema |

---

## 2. Terminology and Definitions

- **Schema V2** — the 14-dimension structured tactile semantic space defined in
  [§4](#4-schema-v2--the-14-dimensions). It is the only data model in the format.
- **Dimension** — one of the 14 named semantic fields of Schema V2.
- **Compliance Level (L1–L4)** — a stratification label describing the maximum
  information density an annotation satisfies (see [§5](#5-compliance-levels-l1l4)).
- **Adapter (Layer 2)** — a software component that maps raw sensor or dataset
  data into Schema V2. Adapters are the *reference implementation* layer.
- **Capabilities** — a dictionary of boolean declarations stating which
  dimensions an adapter can populate (see [§7.4](#74-capabilities-object)).
- **Frame** — the per-timestep unit of annotation; carries one Schema V2 record.
- **Episode** — an ordered sequence of frames plus episode-level metadata.
- **Document** — a serialized TLabel file containing metadata, capabilities,
  and one or more frames.
- **Layer 1 / Layer 2 / Layer 3** — Schema (standard core) / Adapter (reference
  implementation) / downstream derivation, respectively.
- **Null-vs-zero policy** — an unsupported dimension MUST be `null`, never a
  numeric placeholder such as `0` (see [S-6.4](#64-null-vs-zero-policy)).

---

## 3. Data Model Overview

### 3.1 Three-layer architecture

| Layer | Name | Contents | Stability |
|---|---|---|---|
| **Layer 1** | Schema | the 14 Schema V2 dimensions + Compliance Level | Very stable; changes require a version bump and an RFC |
| **Layer 2** | Adapter | `tlabel.adapters.*` mapping raw data → Layer 1 | May iterate frequently |
| **Layer 3** | Derivation | task-specific feature vectors | Not standardized; not part of TLabel |

> **[S-3.1-1]** A conformant producer MUST emit only Layer 1 Schema V2 records.
> A producer MUST NOT emit Layer 3 feature vectors as part of a TLabel
> annotation document.

### 3.2 Document structure

A TLabel document contains, at minimum:

```
document
├── schema_version        : string   (package version, e.g. "0.23.1")
├── schema_version_v2     : string   (Schema revision, MUST be "2.1")
├── format                : string   (MUST be "tlabel_schema_v2")
├── tlabel_dimensions_v2  : integer  (MUST be 14)
├── sensor                : object   (hardware + adapter description)
├── capabilities          : object   (boolean capability declarations)
├── episode               : object   (episode metadata and statistics)
└── frames                : array    (per-frame annotations)
```

> **[S-3.2-1]** A conformant document MUST contain the members
> `schema_version`, `sensor`, `capabilities`, and `frames`.
> **[S-3.2-2]** A conformant document MUST declare `schema_version_v2` equal to
> `"2.1"` and `tlabel_dimensions_v2` equal to `14`.
> **[S-3.2-3]** Each element of `frames` MUST carry exactly one `schema_v2`
> object conforming to [§4](#4-schema-v2--the-14-dimensions).

### 3.3 The frame

Each frame object contains:

| Field | Type | Requiredness | Description |
|---|---|---|---|
| `frame_idx` | integer | REQUIRED | Frame index within the episode |
| `timestamp_s` | number | REQUIRED | Timestamp in seconds |
| `schema_v2` | object | REQUIRED | The 14-dimension Schema V2 record ([§4](#4-schema-v2--the-14-dimensions)) |
| `confidence` | number `[0,1]` | REQUIRED | Frame-level annotation confidence |
| `manipulation_phase` | string \| null | OPTIONAL | Frame-level phase mirror (see note) |
| `is_first` | boolean | OPTIONAL | True for the first frame |
| `is_last` | boolean | OPTIONAL | True for the last frame |
| `sensor_specific` | object | OPTIONAL | Adapter-specific auxiliary data (NOT a Schema V2 dimension) |
| `patches` | array | OPTIONAL | Human-edit audit records |
| `primitive_label` / `primitive_confidence` | string / number | OPTIONAL | Motor-primitive pre-annotation (Layer 3-adjacent) |

> **[S-3.3-1]** `sensor_specific`, `patches`, `primitive_label`, and
> `primitive_confidence` MUST NOT be interpreted as Schema V2 dimensions and
> MUST NOT be used to satisfy any Compliance Level gate.
> **[S-3.3-2]** The frame-level `confidence` and the Schema V2 `confidence`
> field SHOULD carry the same value; the Schema V2 `confidence` is normative.

---

## 4. Schema V2 — The 14 Dimensions

### 4.0 Summary table

The 14 dimensions, in normative order, are:

| # | Field name | Type | Unit | Requiredness | Level gate | Example |
|---|---|---|---|---|---|---|
| 1 | `contact` | boolean | — | REQUIRED (unconditional) | L1+ | `true` |
| 2 | `contact_centroid` | `[number, number]` | sensor pixels or mm (sensor-local) | REQUIRED when `contact == true` | L1+ | `[310.0, 245.0]` |
| 3 | `contact_region` | string enum | — | OPTIONAL | — | `"palmar"` |
| 4 | `force_magnitude` | number (≥ 0) | N (newton) | REQUIRED when level ≥ L2 | L2+ | `1.2` |
| 5 | `force_vector` | `[number, number, number]` | N | REQUIRED when level ≥ L3 | L3+ | `[0.12, -0.30, 2.50]` |
| 6 | `torque_vector` | `[number, number, number]` | N·m | OPTIONAL | — | `[0.01, 0.0, -0.02]` |
| 7 | `slip_event` | boolean | — | REQUIRED (unconditional) | L1+ | `false` |
| 8 | `slip_velocity` | `[number, number]` | mm/s | OPTIONAL (semantically implied when `slip_event == true`) | — | `[12.5, -3.0]` |
| 9 | `manipulation_phase` | string enum | — | OPTIONAL | — | `"hold"` |
| 10 | `texture_class` | string enum | — | OPTIONAL | — | `"smooth"` |
| 11 | `object_deformation` | number | mm or ratio | OPTIONAL | — | `0.3` |
| 12 | `temperature` | number | °C | OPTIONAL | — | `32.5` |
| 13 | `confidence` | number `[0.0, 1.0]` | — | REQUIRED (unconditional) | L1+ | `0.92` |
| 14 | `compliance_level` | string enum | — | REQUIRED (unconditional) | L1+ | `"L2"` |

> **[S-4.0-1]** The field set, order, and names above MUST match
> `SCHEMA_V2_FIELD_NAMES` in `tlabel/core/schema.py`. A conformant record MUST
> contain all 14 keys when serialized via the reference `to_dict()`.
> **[S-4.0-2]** The count of dimensions MUST be reported as exactly **14**.

### 4.1 `contact`

- **Type:** boolean
- **Unit:** dimensionless
- **Requiredness:** REQUIRED, unconditionally
- **Level gate:** part of L1 and above
- **Example:** `true`

> **[S-4.1-1]** `contact` MUST be a JSON boolean (not `0`/`1`, not a string).
> **[S-4.1-2]** `contact` MUST be `true` if and only if any detectable contact
> between the sensor surface and an object is present in that frame.
> **[S-4.1-3]** A conformant validator MUST reject a record whose `contact` is
> not a boolean.

### 4.2 `contact_centroid`

- **Type:** array of exactly 2 numbers `[x, y]`
- **Unit:** sensor-local coordinates — pixels (vision-based) or mm (array)
- **Requiredness:** REQUIRED **when** `contact == true`; otherwise `null`
- **Level gate:** part of L1 and above
- **Example:** `[310.0, 245.0]`

> **[S-4.2-1]** When `contact == true`, `contact_centroid` MUST be populated
> with exactly two numeric components.
> **[S-4.2-2]** When `contact == false`, `contact_centroid` SHOULD be `null`.
> **[S-4.2-3]** The coordinate frame MUST be sensor-local and its unit MUST be
> documented by the adapter's `get_sensor_info()`. TLabel does not mandate a
> global coordinate frame.

### 4.3 `contact_region`

- **Type:** string enum
- **Allowed values:** `palmar`, `digital`, `lateral`, `proximal`, `distal`, `dorsal`, `other`
- **Unit:** dimensionless
- **Requiredness:** OPTIONAL
- **Level gate:** none
- **Example:** `"palmar"`

> **[S-4.3-1]** When populated, `contact_region` MUST be one of the seven values
> listed above. Any other value MUST be rejected by a conformant validator.

### 4.4 `force_magnitude`

- **Type:** number (SHOULD be ≥ 0)
- **Unit:** newton (N) — absolute, **not** a normalized `[0,1]` quantity
- **Requiredness:** REQUIRED when `compliance_level` ∈ {L2, L3, L4}; otherwise OPTIONAL
- **Level gate:** L2 (Force-Aware) and above
- **Example:** `1.2`

> **[S-4.4-1]** When `compliance_level` ≥ L2, `force_magnitude` MUST be
> populated (non-null).
> **[S-4.4-2]** `force_magnitude` MUST be expressed in newtons. Adapters that
> cannot produce SI units MUST emit `null` rather than a pixel-space proxy
> (see [S-6.4](#64-null-vs-zero-policy)).
> **[S-4.4-3]** A negative `force_magnitude` is physically invalid and MUST NOT
> be emitted.

### 4.5 `force_vector`

- **Type:** array of exactly 3 numbers `[Fx, Fy, Fz]`
- **Unit:** newton (N)
- **Requiredness:** REQUIRED when `compliance_level` ∈ {L3, L4}; otherwise OPTIONAL
- **Level gate:** L3 (Full-Vector) and above
- **Example:** `[0.12, -0.30, 2.50]`

> **[S-4.5-1]** When `compliance_level` ≥ L3, `force_vector` MUST be populated
> with exactly three numeric components.
> **[S-4.5-2]** The components MUST be expressed in the sensor-local coordinate
> frame in newtons.
> **[S-4.5-3]** By convention, component order is `[Fx, Fy, Fz]`; the reference
> adapter `tashan_ts_f_a` orders the tangential components first and the normal
> component last (`[fx, fy, normal_force]`). Consumers MUST NOT assume a fixed
> semantic axis without consulting `sensor_info`.

### 4.6 `torque_vector`

- **Type:** array of exactly 3 numbers `[Mx, My, Mz]`
- **Unit:** newton-metre (N·m)
- **Requiredness:** OPTIONAL
- **Level gate:** none (but REQUIRED to reach L4 per [S-5.2-4](#52-normative-determination-rules))
- **Example:** `[0.01, 0.0, -0.02]`

> **[S-4.6-1]** When populated, `torque_vector` MUST have exactly three numeric
> components in N·m.

### 4.7 `slip_event`

- **Type:** boolean
- **Unit:** dimensionless
- **Requiredness:** REQUIRED, unconditionally
- **Level gate:** part of L1 and above
- **Example:** `false`

> **[S-4.7-1]** `slip_event` MUST be a JSON boolean.
> **[S-4.7-2]** `slip_event` MUST be `true` if and only if detectable relative
> motion between the sensor surface and the contact object is present.
> **[S-4.7-3]** A conformant validator MUST reject a record whose `slip_event`
> is not a boolean.

### 4.8 `slip_velocity`

- **Type:** array of exactly 2 numbers `[vx, vy]`
- **Unit:** mm/s
- **Requiredness:** OPTIONAL; semantically implied when `slip_event == true`
- **Level gate:** none
- **Example:** `[12.5, -3.0]`

> **[S-4.8-1]** When populated, `slip_velocity` MUST have exactly two numeric
> components in mm/s, expressed in the sensor-local tangential frame.
> **[S-4.8-2]** `slip_velocity` carries both direction and magnitude. The
> legacy unit-vector field `slip_direction` is NOT part of Schema V2 and MUST NOT
> be emitted.

### 4.9 `manipulation_phase`

- **Type:** string enum
- **Allowed values:** `pre_contact`, `approach`, `grasp`, `lift`, `hold`, `place`
- **Unit:** dimensionless
- **Requiredness:** OPTIONAL
- **Level gate:** none
- **Example:** `"hold"`

> **[S-4.9-1]** When populated, `manipulation_phase` MUST be one of the six
> values listed above. Any other value MUST be rejected by a conformant
> validator.
> **[S-4.9-2]** The normative phase vocabulary is the six-value set above. The
> extended phase names found in earlier documentation (`idle`, `contact`,
> `translate`, `release`, `retract`, `stable_contact`, `slip`, etc.) are
> **not** valid Schema V2 values.

### 4.10 `texture_class`

- **Type:** string enum
- **Allowed values:** `smooth`, `rough`, `granular`, `fibrous`, `sticky`, `slippery`
- **Unit:** dimensionless
- **Requiredness:** OPTIONAL
- **Level gate:** none (but REQUIRED to reach L4)
- **Example:** `"smooth"`

> **[S-4.10-1]** When populated, `texture_class` MUST be one of the six values
> listed above. The extended texture vocabulary found in earlier documentation
> (e.g. `smooth_plastic`, `metal_grid`, `ridged`, `slimy`) is **not** a valid
> Schema V2 value.
> **[S-4.10-2]** The legacy free-form field `texture` (string) MUST NOT be
> emitted.

### 4.11 `object_deformation`

- **Type:** number
- **Unit:** millimetres (mm) or a dimensionless ratio
- **Requiredness:** OPTIONAL
- **Level gate:** none (but REQUIRED to reach L4)
- **Example:** `0.3`

> **[S-4.11-1]** When populated, `object_deformation` MUST be numeric. The unit
> (mm or ratio) MUST be documented by the adapter and MUST be consistent within
> an episode.

### 4.12 `temperature`

- **Type:** number
- **Unit:** degrees Celsius (°C)
- **Requiredness:** OPTIONAL
- **Level gate:** none (but REQUIRED to reach L4)
- **Example:** `32.5`

> **[S-4.12-1]** When populated, `temperature` MUST be numeric and expressed in
> °C.

### 4.13 `confidence`

- **Type:** number in the closed interval `[0.0, 1.0]`
- **Unit:** dimensionless
- **Requiredness:** REQUIRED, unconditionally
- **Level gate:** part of L1 and above
- **Example:** `0.92`

> **[S-4.13-1]** `confidence` MUST be a number in `[0.0, 1.0]` inclusive.
> **[S-4.13-2]** A conformant validator MUST reject a record whose `confidence`
> lies outside `[0.0, 1.0]`.
> **[S-4.13-3]** When the adapter has no meaningful quality signal, the default
> value is `1.0` (as produced by `from_tlabel_v1`).

### 4.14 `compliance_level`

- **Type:** string enum
- **Allowed values:** `L1`, `L2`, `L3`, `L4`
- **Unit:** dimensionless
- **Requiredness:** REQUIRED, unconditionally
- **Level gate:** self-describing
- **Example:** `"L2"`

> **[S-4.14-1]** `compliance_level` MUST be one of `L1`, `L2`, `L3`, `L4`.
> **[S-4.14-2]** `compliance_level` MUST be assigned automatically by the
> adapter; it MUST NOT be set manually by a user.
> **[S-4.14-3]** `compliance_level` MUST be consistent with the populated
> dimensions according to [§5.2](#52-normative-determination-rules).

---

## 5. Compliance Levels L1–L4

### 5.1 Level definitions

| Level | Name | Meaning | Required dimensions (in addition to lower levels) |
|---|---|---|---|
| **L1** | Basic Tactile | Contact-level annotation only | `contact`, `contact_centroid` (when `contact`), `slip_event`, `confidence`, `compliance_level` |
| **L2** | Force-Aware | Normal force scalar available | L1 + `force_magnitude` |
| **L3** | Full-Vector | Complete 3-D force vector available | L2 + `force_vector` |
| **L4** | Rich-Semantic | Full semantics; all optional dimensions populated | L3 + `contact_region`, `torque_vector`, `slip_velocity`, `manipulation_phase`, `texture_class`, `object_deformation`, `temperature` |

### 5.2 Normative determination rules

The following clauses are written to be directly translatable into conformance
tests. Let `r` be a Schema V2 record and `L(r)` its `compliance_level`.

> **[S-5.2-1] (L1 base)** Every conformant record MUST satisfy:
> `r.contact` is boolean, `r.slip_event` is boolean,
> `r.confidence ∈ [0,1]`, and `r.compliance_level ∈ {L1,L2,L3,L4}`.
> If `r.contact == true` then `r.contact_centroid` is a 2-element numeric array.
>
> **[S-5.2-2] (L2 gate)** If `L(r) ∈ {L2, L3, L4}` then `r.force_magnitude` is
> a non-null number.
>
> **[S-5.2-3] (L3 gate)** If `L(r) ∈ {L3, L4}` then `r.force_vector` is a
> 3-element numeric array.
>
> **[S-5.2-4] (L4 gate)** If `L(r) == L4` then all of `contact_region`,
> `torque_vector`, `slip_velocity`, `manipulation_phase`, `texture_class`,
> `object_deformation`, and `temperature` are non-null.
>
> **[S-5.2-5] (Cumulative)** L2 MUST satisfy every L1 clause; L3 MUST satisfy
> every L2 and L1 clause; L4 MUST satisfy every L3, L2, and L1 clause.
>
> **[S-5.2-6] (Enum conformance)** When `contact_region`, `manipulation_phase`,
> or `texture_class` is non-null, its value MUST be a member of the respective
> enum in [§4](#4-schema-v2--the-14-dimensions).
>
> **[S-5.2-7] (Vector arity)** When non-null, `contact_centroid`, `force_vector`,
> `torque_vector`, and `slip_velocity` MUST have arity 2, 3, 3, and 2
> respectively.

### 5.3 Design principles (informative)

1. **Cumulative** — higher levels strictly include lower-level requirements.
2. **Physical reality first** — a sensor that physically cannot measure shear
   force is correctly L2, not "non-compliant".
3. **Downstream-transparent** — `compliance_level` tells consumers the data
   boundary without inspecting every field.
4. **Adapter-driven** — the adapter assigns the level; users do not.

### 5.4 Reference-implementation note (informative)

The reference validator `TLabelSchemaV2.validate()` enforces [S-5.2-1] through
[S-5.2-3] and [S-5.2-6]/[S-5.2-7]. It does **not** currently enforce the L4
clause [S-5.2-4], nor the non-negativity of `force_magnitude` [S-4.4-3]. See
[§11](#11-differences-from-existing-documentation). Implementations that claim
full L4 conformance MUST enforce [S-5.2-4] themselves.

### 5.5 Current adapter levels (informative)

| Adapter | `default_compliance_level` | Basis |
|---|---|---|
| `tacquad` | L1 | Images only; approximate force for Tac3D |
| `xela` | L1 | Upgrades to L3 only when `force_scale` calibration is supplied |
| `paxini` / `paxini_gen3` | L2 | Normal force scalar only |
| `ycb_slide` | L2 | Deformation magnitude approximates normal force |
| `vtouch` | L2 | Force information available |
| `syntouch` | L2 | BioTac force channels |
| `univtac` | L2 | Force magnitude + shear |
| `gelsight` | L2 | Visuo-tactile; 3-D force requires calibration |
| `daimon` / `daimon_dm_tac` | L2 | Deformation/shear/depth streams; calibration-dependent |
| `touchd` | L3 | Ground-truth 3-D force labels |
| `tashan_ts_f_a` | L3 | Complete 3-D force vector `[fx, fy, normal_force]` |
| `tlabel` | L1 | Built-in format example |

> **[S-5.5-1]** `default_compliance_level` declares the **maximum** level the
> adapter can satisfy. An adapter MAY emit a lower level for individual frames
> (e.g. `tashan_ts_f_a` emits L1 for no-contact frames), but MUST NOT emit a
> level higher than its declared capability.

---

## 6. Adapter Conformance

### 6.1 Base classes

> **[S-6.1-1]** A conformant adapter MUST subclass exactly one of
> `DataAdapterBase` (offline dataset files) or `SensorAdapterBase` (real-time
> SDK). The legacy alias `BaseAdapter` is equivalent to `DataAdapterBase` and
> SHOULD NOT be used in new code.

### 6.2 Required interface

> **[S-6.2-1]** An adapter MUST implement:
> - `name` (property, unique lowercase identifier);
> - `supported_extensions` (list of file extensions);
> - `get_capabilities()` → capability dictionary ([§7.4](#74-capabilities-object));
> - `get_sensor_info()` → sensor metadata dictionary;
> - `extract_schema(raw_frame_data)` → `TLabelSchemaV2`;
> - `load(file_path, ...)` → `TLabelData`.
>
> **[S-6.2-2]** A `SensorAdapterBase` subclass MUST additionally implement
> `connect()`, `disconnect()`, `is_connected()`, and `stream_frames()`.
>
> **[S-6.2-3]** `extract_schema()` MUST return a `TLabelSchemaV2` whose
> `compliance_level` is drawn from `self.default_compliance_level` (or a
> justified lower level for individual frames).

### 6.3 Capability declaration

> **[S-6.3-1]** `get_capabilities()` MUST return a dictionary whose keys are a
> subset of the Schema V2 capability key set ([§7.4](#74-capabilities-object))
> and whose values are booleans.
> **[S-6.3-2]** A dimension declared `false` in `capabilities` MUST be emitted
> as `null` in every frame; a dimension declared `true` SHOULD be populated
> whenever the physical signal is available.
> **[S-6.3-3]** A conformant adapter MUST NOT declare a capability it cannot
> honour, and MUST NOT populate a dimension it declared `false`.

### 6.4 Null-vs-zero policy

> **[S-6.4-1]** An unsupported dimension MUST be `null`. It MUST NOT be filled
> with `0` or any other numeric placeholder.
> **[S-6.4-2]** The rationale is that `0` is a meaningful measurement (zero
> force), whereas `null` means "this sensor does not provide this dimension".

### 6.5 Compliance level assignment

> **[S-6.5-1]** An adapter MUST set `default_compliance_level` to the highest
> level for which it can satisfy [§5.2](#52-normative-determination-rules) on
> every frame that is physically in contact.

### 6.6 Registration

> **[S-6.6-1]** An adapter SHOULD register itself via `register_adapter()` or
> via the entry-point group `tlabel.adapters` for community/third-party adapters.
> **[S-6.6-2]** Adapter names MUST be unique within a registry.

---

## 7. Serialization and Interoperability

### 7.1 Canonical JSON form

> **[S-7.1-1]** JSON is the canonical interchange serialization of TLabel.
> **[S-7.1-2]** A Schema V2 record serialized via `TLabelSchemaV2.to_dict()`
> MUST contain the 14 keys of [§4](#4-schema-v2--the-14-dimensions) in that order.
> **[S-7.1-3]** The optional extension members `data_quality`, `provenance`, and
> `raw_sensor_values` MUST be omitted when null (they are not dimensions).

### 7.2 Per-frame serialization

> **[S-7.2-1]** A frame MUST serialize its Schema V2 record under the key
> `schema_v2`.
> **[S-7.2-2]** A document MAY additionally expose a top-level
> `schema_v2_frames` array (one record per frame) for batch consumers; when
> present it MUST be consistent with `frames[].schema_v2`.

### 7.3 Top-level serialization

> **[S-7.3-1]** The document MUST set `format` to `"tlabel_schema_v2"`.
> **[S-7.3-2]** The document MUST set `tlabel_dimensions_v2` to `14` and
> `feature_names_v2` to the ordered 14 field names.
> **[S-7.3-3]** The `sensor` member MUST describe the hardware and adapter; it
> SHOULD include `type` and, when known, `manufacturer` and `model`.
> **[S-7.3-4]** The `episode` member MUST include `num_frames`, `duration_s`,
> and a `stats` object containing `contact_frames`, `contact_ratio`,
> `slip_frames`, `slip_ratio`, and `modified_frames`.

### 7.4 Capabilities object

> **[S-7.4-1]** The capabilities object MUST declare booleans for the Schema V2
> capability key set. The normative key set (matching
> `schema/tlabel-schema.json`) is the following **13** keys:
>
> `contact`, `contact_centroid`, `contact_region`, `force_magnitude`,
> `force_vector`, `torque_vector`, `slip_event`, `slip_velocity`,
> `manipulation_phase`, `texture_class`, `object_deformation`, `temperature`,
> `confidence`.
>
> **[S-7.4-2]** `contact` MUST be present. `compliance_level` is a
> self-describing dimension and MUST NOT appear in the capabilities object.

### 7.5 Other serializations (informative)

- **CSV** — the reference exporter emits the Schema V2 dimensions as flattened
  columns (e.g. `contact_centroid_x`, `contact_centroid_y`, `force_vector_x`).
- **HDF5** — Schema V2 is the only written path since v0.17.0.
- **FTP-1 / MTTS Zarr** — available via `TLabelData.export_ftp1()`.
- **Downstream** — TLabel annotations augment LeRobot episodes, Open
  X-Embodiment metadata, and RoboMimic HDF5 demonstrations.

---

## 8. Versioning and Backward Compatibility

> **[S-8.1-1]** The format is versioned by two independent strings:
> - `schema_version` — the `tlabel` package version (semver, e.g. `"0.23.1"`);
> - `schema_version_v2` — the Schema revision (currently `"2.1"`).
>
> **[S-8.2-1]** Semantic-versioning rules for the schema:
> - **Major** — a breaking change to the document or record structure;
> - **Minor** — a new OPTIONAL dimension or extension member;
> - **Patch** — documentation or clarification only.
>
> **[S-8.3-1]** Since package v0.17.0 the legacy `tlabel_v2` flat feature
> dictionary (pre-v2.0) is **removed**. Producers MUST NOT emit it and consumers
> MUST NOT rely on it.
>
> **[S-8.4-1]** `TLabelSchemaV2.from_tlabel_v1()` provides a best-effort
> migration from the legacy flat dictionary; migrated records default to
> `compliance_level = "L1"` and `confidence = 1.0`.
>
> **[S-8.5-1]** A consumer MUST ignore unknown extension members
> (`data_quality`, `provenance`, `raw_sensor_values`, `sensor_specific`) rather
> than failing, to preserve forward compatibility.

### Version history (informative)

| Schema rev. | Package line | Change |
|---|---|---|
| 2.0 | v0.16 | Three-layer architecture; 10 → 12 dimensions |
| 2.1 | v0.17+ | Compliance Levels L1–L4; `force_vector` downgraded to L3+; `force_magnitude` (L2+) and `compliance_level` added; 12 → 14 dimensions |

---

## 9. Extension Mechanism

> **[S-9.1-1]** New **optional** dimensions or extension members MUST be added
> via a minor version bump and MUST be backward compatible.
> **[S-9.1-2]** Extension members are carried **outside** the 14-dimension core:
> - `data_quality` — user-declared processing level (Q1–Q4);
> - `provenance` — minimal source metadata (`sensor_model`, `sensor_firmware`,
>   `calibration_date` as ISO-8601 `YYYY-MM-DD`, `sampling_rate_hz` > 0);
> - `raw_sensor_values` — adapter-defined raw readings preserved after mapping;
> - `sensor_specific` — per-frame adapter-specific auxiliary data.
> **[S-9.1-3]** Any change that adds, removes, or renames a **core** dimension,
> changes the JSON structure, or alters cascade rules MUST go through the
> TLabel RFC process (`internal-docs/RFC-README.md`) before implementation.
> **[S-9.1-4]** Adding a new dimension MUST NOT change the meaning of existing
> dimensions; reusing a name with a new meaning is prohibited.

### Q1–Q4 data-quality levels (informative)

- **Q1** — user declares "raw, unprocessed data";
- **Q2** — user declares "denoised / calibrated / time-aligned";
- **Q3** — third-party or automated spot-check confidence score exists;
- **Q4** — full manual annotation plus multi-sensor cross-validation.

---

## 10. Conformance Test Checklist

A test suite can implement the following checks directly against
[§4](#4-schema-v2--the-14-dimensions) and [§5.2](#52-normative-determination-rules).

| ID | Check | Clause |
|---|---|---|
| C-01 | Record contains all 14 keys in normative order | S-4.0-1 |
| C-02 | `contact` is boolean | S-4.1-1, S-5.2-1 |
| C-03 | `slip_event` is boolean | S-4.7-1, S-5.2-1 |
| C-04 | `confidence ∈ [0.0, 1.0]` | S-4.13-1 |
| C-05 | `compliance_level ∈ {L1,L2,L3,L4}` | S-4.14-1 |
| C-06 | `contact == true` ⇒ `contact_centroid` is 2 numbers | S-4.2-1 |
| C-07 | level ≥ L2 ⇒ `force_magnitude` non-null | S-5.2-2 |
| C-08 | level ≥ L3 ⇒ `force_vector` is 3 numbers | S-5.2-3 |
| C-09 | level == L4 ⇒ all optional dimensions non-null | S-5.2-4 |
| C-10 | `contact_region` ∈ enum (when non-null) | S-4.3-1 |
| C-11 | `manipulation_phase` ∈ enum (when non-null) | S-4.9-1 |
| C-12 | `texture_class` ∈ enum (when non-null) | S-4.10-1 |
| C-13 | `torque_vector` arity 3, `slip_velocity` arity 2 | S-5.2-7 |
| C-14 | `force_magnitude ≥ 0` (when non-null) | S-4.4-3 |
| C-15 | `capabilities` keys ⊆ 13-key set; `compliance_level` absent | S-7.4-1/2 |
| C-16 | every dimension declared `false` is `null` in frames | S-6.3-2 |

> **Reference implementation coverage.** `TLabelSchemaV2.validate()` satisfies
> C-01…C-13. C-09, C-14, C-15, and C-16 are **not** enforced by the reference
> validator and require an additional conformance layer.

---

## 11. Differences from Existing Documentation

The following conflicts exist between the existing documentation and the
reference implementation. In every case **the code is authoritative**, and the
resolution is stated.

| # | Existing document | Conflicting statement | Code / normative resolution |
|---|---|---|---|
| D-01 | `docs/whitepaper.md` (v0.8.0) | Describes the removed pre-v2.0 flat feature space with an obsolete dimensionality | Obsolete pre-v2.0 framing. Normative model is **Schema V2 = 14 dimensions** (`schema.py`). The whitepaper is superseded by this spec. |
| D-02 | `docs/tutorial-gelsight.md`, `docs/tutorial-daimon.md` | Reference the removed pre-v2.0 flat feature space and its obsolete dimensional count | Obsolete. Adapters emit `TLabelSchemaV2` (14 dimensions). |
| D-03 | `docs/demo.html` | Radar chart and legend visualize the removed legacy flat feature dictionary | Obsolete visualization of a removed structure. Not part of the format. |
| D-04 | `docs/tlabel-format.md` §"Top-Level" | Uses top-level `sensor_info` and an `episodes` array | Code emits top-level **`sensor`** and a **`frames`** array (`TLabelData.to_dict()`). |
| D-05 | `docs/tlabel-format.md` §"manipulation_phase" | Enum includes `idle, contact, translate, release, retract, …` | Normative enum is `pre_contact, approach, grasp, lift, hold, place` (`VALID_MANIPULATION_PHASES`). |
| D-06 | `docs/tlabel-format.md` §"texture_class" | Enum includes `smooth_plastic, metal_grid, ridged, slimy, hard, soft, …` | Normative enum is `smooth, rough, granular, fibrous, sticky, slippery` (`VALID_TEXTURE_CLASSES`). |
| D-07 | `docs/tlabel-format.md` §"contact_region" | "sensor-dependent regions … TLabel does not enforce a universal vocabulary" | Code enforces a fixed enum: `palmar, digital, lateral, proximal, distal, dorsal, other` (`VALID_CONTACT_REGIONS`). |
| D-08 | `docs/tlabel-format.md` §"Validation Rules" #1 | `schema_version` matches `^2\.1\.0$` | `schema_version` is the **package** version (e.g. `0.23.1`); the schema revision is `schema_version_v2 = "2.1"`. |
| D-09 | `docs/tlabel-format.md` §"Validation Rules" #5 | "`force_magnitude` values are ≥ 0" enforced | `validate()` does **not** check non-negativity (verified: a negative value passes). Treat S-4.4-3 as a producer obligation. |
| D-10 | `docs/tlabel-format.md` / `docs/annotation-spec.md` | Compliance consistency checked only up to L3; L4 requires "all Optional fields" | `validate()` enforces L2/L3 gates only; the L4 gate (S-5.2-4) is normative here but not implemented in the reference validator. |
| D-11 | `docs/annotation-spec.md` | Lists "Force magnitude is negative" as a hard error | Not enforced by `validate()`; still a normative producer requirement (S-4.4-3). |
| D-12 | `docs/tlabel-format.md` §"Validation Rules" #4 | "No field appears in frames that is declared false in capabilities" | Not enforced by `validate()`; enforced only by producer conformance (S-6.3-2). |
| D-13 | `docs/tlabel-format.md` §"capabilities" table | Lists 13 dimensions (correct) but omits the explicit normative key set | Normative key set fixed in [§7.4](#74-capabilities-object). **Resolved in 0.24.0**: the table now states the closed 13-key set, the `contact` requirement and the `compliance_level` exclusion. |
| D-14 | `docs/DESIGN.md`, `docs/TLabel_Design_Document.md` §3.1 caption | "为什么是12个维度？" heading under a 14-dimension table | Heading is stale; the normative count is 14. |
| D-15 | `docs/DESIGN.md` / `TLabel_Design_Document.md` §2 diagram | "`tlabel-schema.json` — 12个语义维度定义" | Stale; `schema/tlabel-schema.json` defines the 14-dimension Schema V2 (13 capability keys + `compliance_level`). |
| D-16 | `schema/tlabel-schema.json` `sensor.required` | Requires `sensor_name, sensor_type, adapter_name, adapter_version` | Reference adapters emit `type/manufacturer/model`; `to_dict()` does not guarantee the JSON-Schema-required subfields. Treat the JSON Schema `sensor` requirements as aspirational. |
| D-17 | Adapter `get_capabilities()` (legacy adapters) | `gelsight`, `touchd`, `tacquad`, `daimon*`, `paxini*`, `univtac`, `vtouch`, `ycb_slide` still return the legacy 22-key capability dictionary | Normative capability key set is the 13-key Schema V2 set. **Resolved in 0.24.0**: all built-in adapters now return the 13-key set; conformance is pinned by `tests/unit/conformance/test_capabilities_conformance.py`. |
| D-18 | Adapter `get_capabilities()` (`tashan_ts_f_a`, `syntouch`) | Include a `compliance_level` key in capabilities | Violates `additionalProperties: false` in `schema/tlabel-schema.json` and [S-7.4-2]. **Resolved in 0.23.3** (adapters) and enforced at every exit path in **0.24.0** (`normalize_capabilities()`). |
| D-19 | `docs/TLabel_Design_Document.md` §3.1 note | "9 个 Optional（其中 force_magnitude 为 L2+ 合规的约定必填项）" | Correct count of optional dimensions is 9, but `force_magnitude` is conditionally REQUIRED at L2+ (S-4.4-1), not merely "conventional". |
| D-20 | `docs/annotation-spec.md` §"Limitations" | "Compliance Level is sensor-level, not frame-level" | Partially superseded: the code assigns `compliance_level` per frame (e.g. `tashan_ts_f_a` emits L1 for no-contact frames). Per-frame levels are valid. |

> **Resolution rule.** Where any statement in `docs/` conflicts with
> `tlabel/core/schema.py`, `tlabel/core/types.py`, or
> `tlabel/adapters/base.py`, the code governs. This specification is the
> normative reference; the documents in the table above are informative.

---

## Appendix A — Enum reference

| Field | Allowed values |
|---|---|
| `contact_region` | `palmar`, `digital`, `lateral`, `proximal`, `distal`, `dorsal`, `other` |
| `manipulation_phase` | `pre_contact`, `approach`, `grasp`, `lift`, `hold`, `place` |
| `texture_class` | `smooth`, `rough`, `granular`, `fibrous`, `sticky`, `slippery` |
| `compliance_level` | `L1`, `L2`, `L3`, `L4` |
| `data_quality.level` | `Q1`, `Q2`, `Q3`, `Q4` |

## Appendix B — Minimal Schema V2 record example

```json
{
  "contact": true,
  "contact_centroid": [310.0, 245.0],
  "contact_region": "palmar",
  "force_magnitude": 1.2,
  "force_vector": [0.12, -0.30, 2.50],
  "torque_vector": null,
  "slip_event": false,
  "slip_velocity": null,
  "manipulation_phase": "hold",
  "texture_class": "smooth",
  "object_deformation": 0.3,
  "temperature": null,
  "confidence": 0.92,
  "compliance_level": "L2"
}
```

This record is L2-conformant: L1 base fields are present, `force_magnitude` is
populated ([S-5.2-2]), and no L3/L4 claim is made.

---

*End of specification.*
