# Changelog

All notable changes to the TLabel project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.23.0] - 2026-09-04

### Added
- **XELA uSkin sensor adapter** (`xela`): DataAdapter for XELA Robotics uSkin uSPa 46 tactile sensors (4×6 = 24 taxels × 3-axis Hall-effect sensing) and the UniTac-NV public dataset in CSV format (IROS 2025). Follows the DataAdapterBase standard pattern with dual parsing (header aliases + positional columns) and per-frame layout validation. Honest physical-unit policy: force fields default to `null` when the upstream dataset does not document force units, and the record upgrades to compliance level L3 once `force_scale` is provided. Contributed by gaolebaigao (PR #24).
- **UniVTAC adapter**: support for the new recording format's tactile keys `left_tactile` / `right_tactile` (legacy `*_gsmini` keys remain fully supported).

### Fixed
- **UniVTAC adapter `episode_info.sensor_ids` was always empty**: membership checks like `f'tactile/{key}' in f` were evaluated against the h5py root group (whose direct members are only top-level groups) *after* the HDF5 file had been closed, so every sensor key resolved as absent. The adapter now probes the `tactile` subgroup while the file is open and records the actually present sensor ids.
- **UniVTAC adapter `marker_count` was hardcoded** (1200 for legacy gsmini, 63 fallback for new keys): the sensor layout `marker_count` is now inferred dynamically from the marker dataset shape `(T, 2, marker_size, 2)` (using `shape[2]`), falling back to the `SENSOR_CONFIG` value only when the dataset is unavailable or has an unexpected shape.
- **Tests: hardcoded local absolute paths** in `tests/test_v18_real_hdf5.py`, `tests/integration/test_v17_e2e.py` and `tests/unit/test_tacquad.py` polluted `sys.path` (causing the suite to import a stale developer copy of `tlabel`) and crashed pytest collection. Paths are now derived from `__file__`, and the real-HDF5 integration test skips gracefully when the source data files are not present.

## [0.22.4] - 2026-09-03
### Added
- LeRobot dataset exporter: create new LeRobot v2.1 datasets from TLabel annotations
- UI: LeRobot export panel in the Export tab
- Registry: new `lerobot_create` exporter plugin
## [0.22.0] - 2026-09-03

### Added
- **Tashan TS-F-A adapter** (`tashan_ts_f_a`): DataAdapter for RoboMIND V2.0 AgileX tactile data
  - Sensor: Tashan (他山科技) TS-F-A 3D Force fingertip sensor
  - Parses HDF5 files with shape `(T, 2, 6)` float32 tactile observations
  - 6D per-sensor: normal_force, tangential_force, tangential_direction, tangential_fx, tangential_fy, contact_indicator
  - 65535.0 (uint16 overflow) treated as invalid/no-contact marker
  - Compliance Level: L3 (full 3D force vector: fx, fy, fz)
  - Naming follows brand+model convention: `tashan_ts_f_a`
  - Validated against 2 trajectories: `data/` (1294 frames) and `data1/` (2066 frames)

## [0.21.1] - 2026-08-31

### Fixed
- **GelSightAdapter.load() P0 bug**: `load()` was using `TLabelSchemaV2.from_tlabel_v1()` instead of `self.extract_schema()`, causing:
  - `compliance_level` always set to L1 (default from `from_tlabel_v1`), ignoring real calibrated force data
  - Real force vectors (e.g., ATI nano17) only stored in `sensor_specific.force_vector_N` but not in standard `schema_v2.force_vector`
  - Fix: `load()` now calls `self.extract_schema(raw_frame_data)` with `force_vector_N`, enabling automatic L3 upgrade when calibrated force data is present
- **GelSightAdapter timestamp**: `timestamp_s` was hardcoded to `gidx / 30.0` instead of using actual `sample_rate` (GelSight=25Hz, DIGIT=60Hz)

### Validation
- Tested with Sparsh T1 Force dataset (Meta FAIR): 50 frames all correctly achieve L3 compliance level
- `schema_v2.force_vector` now contains real ATI nano17 force data (unit: Newton), matching `sensor_specific.force_vector_N`
- Backward compatible: when no `force_vector_N` is available, defaults to L2 as before

## [0.21.0] - 2026-08-28

### Added
- **Optional metadata fields** (non-invasive, fully backward compatible):
  - `data_quality`: User self-declared data processing level (Q1-Q4). Q1=raw, Q2=denoised/calibrated, Q3=third-party verified, Q4=full manual annotation + cross-sensor validation. TLabel provides the field and definition only — it does not perform data cleaning or quality judgment.
  - `provenance`: Minimal provenance metadata ("birth certificate") with 4 optional fields: `sensor_model`, `sensor_firmware`, `calibration_date` (ISO 8601), `sampling_rate_hz`. Only fields that directly affect data comparability and calibration are included; other lifecycle metadata belongs to data management platforms.
- Both fields serialize to JSON only when non-None, preserving backward compatibility with older annotation files.

### Validation
- New structure validators in `TLabelSchemaV2.validate()`:
  - `data_quality`: dict with enum `level` ∈ {Q1,Q2,Q3,Q4}, typed bool/str sub-fields
  - `provenance`: dict with typed string fields + positive numeric `sampling_rate_hz` + ISO date format for `calibration_date`
- Validation errors are independent from compliance level (L1-L4) rules.

### Tests
- 22 new tests in `tests/test_v21_metadata_fields.py`: backward compat, data_quality structure, provenance structure, round-trip serialization, validation independence.
- All 80 core tests (conformance + test_tlabel) still pass with no regression.

## [0.19.0] - 2026-08-06

### Added
- **CLI convert commands**: `tlabel convert` for single-file format conversion, `tlabel batch-convert` for batch directory conversion. Supports 9 data adapters (gelsight, paxini, daimon, tlabel, touchd, univtac, vtouch, ycb_slide, tacquad) and 2 output formats (lerobot, ftp1)
- **CLI adapter discovery**: `tlabel list-adapters` shows all available DataAdapters and SensorAdapters with supported formats; `tlabel adapter-info <name>` displays detailed adapter information including field mapping table and compliance level
- **Converter base layer** (`converters/base.py`): Unified converter interface (`BaseConverter` with `export()` method), wrapping `LeRobotConverter` and `FTP1Converter` for consistent API

### Fixed
- **lerobot.py `_safe_float()`**: Fixed crash when converting vector/list fields (e.g., `contact_centroid`) — now safely handles vectors (takes magnitude), booleans (0/1), and None (0.0)

### Tests
- 28 new tests for CLI convert commands (tests/test_cli_convert.py)
- 14 regression tests passing (tests/test_cli.py)
- End-to-end validation: tlabel→ftp1 (150 frames .zarr), tlabel→lerobot (150 frames parquet+meta), batch-convert (3 files)

## [0.18.2] - 2026-08-03

### Fixed

- `force_vector_field()` and `contact_region_overlay()` now handle single-channel (grayscale) input images — auto-convert to 3-channel RGB

## [0.18.1] - 2026-08-03

### Fixed
- **REGRESSION**: Moved `import math` to module level in `tlabel/core/taxonomy.py` — fixes `NameError` when `evaluate_rule()` computes `force_vector_magnitude` (the fix from v0.17.2 was lost during v0.18 refactoring)

## [0.18.0] - 2026-08-03

### Added
- **Image shape detection** (`detect_image_shape()`): Each adapter now reports its native tactile image dimensions `(H, W, C)`. Supports GelSight (240×320×3), PaXini (8×8×1), ToucHD, VTouch, and LeRobot converter integration with `image_shape`/`adapter` parameters
- **Annotation module** (`core/annotation.py`): Schema-aware annotation toolkit — `validate_annotations()`, `annotate_from_taxonomy()` (primitive auto-labeling from taxonomy rules), `annotate_events_from_data()` (event detection from signal patterns: contact_onset/loss, slip, force_spike, stable_grip), `clear_annotations()`, `get_annotation_summary()` with timeline view. `TLabelData` convenience methods: `.annotate_from_taxonomy()`, `.annotate_events_auto()`, `.validate_annotations()`, `.clear_annotations()`, `.get_annotation_summary()`
- **Tactile visualization** (`viewer/tactile_vis.py`): Rich visualization suite — `contact_heatmap()` (pseudo-color deformation overlay), `force_vector_field()` (quiver plot), `contact_region_overlay()` (centroid + region highlight), `composite_view()` (all-in-one from TLabelFrame), `frame_animation()` (GIF/HTML), `text_summary()` (text fallback). Three-tier degradation: Level 1 (numpy+image) → Level 2 (numpy only) → Level 3 (pure text)

### Fixed
- `contact_heatmap()` now accepts scalar intensity values (auto-broadcasts to full image)
- `force_vector_field()` now accepts list/tuple input (auto-converts to numpy array)
- `text_summary()` handles 2D force vectors (force_vector with only x,y components)

### Tests
- 52 unit tests passing (22 detect_image_shape + 30 annotation/visualization)
- 18 integration tests on real UniVTAC HDF5 data (schema_v2, 57+55 frames, GelSight Mini sensors)

## [0.17.2] - 2026-07-25

### Fixed
- **DEV-004**: Added missing `import math` in `tlabel/core/taxonomy.py` — `_resolve_field_value()` used `math.sqrt()` but math was only imported inside `evaluate_rule()`, causing `NameError`
- **DEV-005**: Lazy-load predict/quality/batch/augment modules via `__getattr__` — prevents eager sklearn/joblib import, reducing `import tlabel` time from ~1.06s to ~0.1s in full-extras environments
- **DEV-001**: `TLabelFrame.contact` and `TLabelFrame.slip_event` now return `bool` (matching TLabelSchemaV2 design) instead of `float`
- **DEV-002**: Added `_check_ml_deps()` helper in `tlabel/predict/__init__.py` with helpful `pip install tlabel[ml]` hint when ML dependencies are missing

## [0.17.1] - 2026-07-24

### Changed
- Documentation overhaul: aligned all docs to 14-dimensional Schema V2
- Updated README, annotation-spec, tlabel-format to reflect Compliance Level (L1-L4)

### Removed
- `examples/tacquad_benchmark/` directory (moved to [tlabel-bench](https://github.com/liesliy/tlabel-bench))

## [0.17.0] - 2026-07-24

### ⚠️ Breaking Changes
- **Schema V2 Only**: Removed all legacy `tlabel_v2` format support
- Schema expanded from 12 to **14 dimensions**: added `force_magnitude` (Required at L2+) and `compliance_level` (Required, L1-L4)
- `force_vector` downgraded from Required to **Optional (L3+)**
- Introduced **Compliance Level** system (L1 Basic → L4 Rich-Semantic)
- Introduced **dual base class architecture**: `DataAdapterBase` + `SensorAdapterBase`

### Added
- 7 public dataset adapters + 2 real-time sensor adapters
- CLI tools: `tlabel validate`, `tlabel info`, `tlabel export`
- JSON, CSV, HDF5 export support
- `compliance_level` auto-declaration per adapter

### Migration
- See [MIGRATION.md](MIGRATION.md) for v0.16 → v0.17 migration guide
- All code must use Schema V2 path; legacy format detection removed

## [0.16.0] - 2026-07-23

### Added
- Open architecture with dual base classes (`DataAdapterBase` + `SensorAdapterBase`)
- 7 dataset adapters (Daimon, PaXini, YCB-Slide, DM-TAC, etc.)
- CLI interface
- CSDN tutorial published

## [0.15.0] and earlier

Earlier versions used a feature-vector-centric design (18/22-dimensional). These have been superseded by the Schema V2 architecture introduced in v0.17.0.

