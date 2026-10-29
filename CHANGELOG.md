# Changelog

All notable changes to the TLabel project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.24.1] - 2026-10-29

补发版，修正 v0.24.0 的打包缺陷。

### Fixed

- **`lerobot` extra 缺失，LeRobot 读写装完包即报错**：`tlabel/converters/lerobot.py` 有三处 `Table.to_pandas()`，但 pyproject 从未声明过 LeRobot 相关的 extra，主依赖也只有 numpy + pyyaml。用户执行 `pip install tlabel` 后调用 LeRobot 导出/写回会直接 `ModuleNotFoundError: No module named 'pandas'`。新增 `lerobot = ["pyarrow>=10.0", "pandas>=1.5"]`，并同步进 `all` 与 `dev`。
- 无代码改动，无 API 变化。

### CI

- 固定 runners 到 `ubuntu-22.04`：`ubuntu-latest` 已升级到 Ubuntu 26.04，其 setup-python 不再提供 Python 3.9，导致 unit-tests (3.9) 长期失败（自 v0.23.3 起）。

---

## [0.24.0] - 2026-10-29

> 迁移说明见 [MIGRATION.md](MIGRATION.md) 的 v0.23 → v0.24 一节。

### ⚠️ Breaking Changes

- **`capabilities` 收敛为闭合的 13 键布尔集合**：合法键由 `tlabel/core/capabilities.py` 的 `CAPABILITY_KEYS` 唯一定义（与 `schema/tlabel-schema.json` 同源）。`contact` 必填（S-7.4-1），`compliance_level` 禁止出现（S-7.4-2）。旧 22 键能力字典（`has_image` / `has_force` / `dimensions` / `texture_energy` / `edge_density` / `shear_field_*` 等）不再被任何内置适配器产出。**读路径保持宽容**：旧文件仍可加载，非法键被剔除并原样保留到 `episode_info["legacy_capabilities"]`。
- **9 个适配器的 `load()` 改走 `extract_schema()`**：`daimon_dataset` / `daimon_dm_tac` / `paxini_dataset` / `paxini_gen3` / `tacquad` / `touchd` / `univtac` / `vtouch` / `ycb_slide` 此前绕过 `extract_schema()` 直接调用 `TLabelSchemaV2.from_tlabel_v1()`，导致 L3 增强全部丢失。**导出文件的帧内容因此变化**：`compliance_level` 不再恒为 `L1`、`slip_velocity` 不再恒为 `null`、`force_vector` 不再 Fy 恒 0。基于旧（错误）输出的下游基线需要重新生成。
- **`get_capabilities()` 统一为无参**：`TLabelAdapter` 此前要求 `file_path`，破坏了基类契约。文件驱动的能力查询改用新增的 `get_capabilities_for_file(path)`。
- **`ExporterResult.stats` 键统一为 `num_frames`**：此前 `json`/`csv`/`hdf5`/`lerobot` 用 `frames`、`ftp1` 用 `time_steps`。读 `stats["frames"]` 的代码需要改。
- **`-f lerobot` 语义显式化**：CLI 的 `lerobot` 一直指"新建数据集"（registry 里叫 `lerobot_create`），现已通过别名映射固定下来；写回已有数据集需显式指定 `-f lerobot_writeback`。旧命令行为不变。

### Added

- **CLI 信息命令增强**：`tlabel list --verbose` 输出每个适配器的类名、模块、默认合规等级、能力明细与非法键警告；`--json` 供脚本消费。`tlabel info <adapter>` 补齐同样信息并支持 `--image-shape`。
- **`tlabel export` 接入导出器注册表**：`-f` 现支持 8 种格式（`json` / `csv` / `hdf5` / `lerobot` / `lerobot_writeback` / `ftp1` / `rlds` / `ros2`），非法格式会列出可用项。其中 `rlds` 与 `ros2` 为预留导出器，当前为占位实现。新增 `--sensor` / `--areas`（FTP-1）、`--overwrite`（LeRobot 写回）。
- **批量导出**：`tlabel export <目录> -f <格式> --batch`，递归按适配器支持的扩展名处理，保持子目录结构，单文件失败不中断其余，退出码反映失败数。
- **`capabilities` 出口归一化与严格模式**：新增 `normalize_capabilities()` / `strict_enabled()`。所有出口（`TLabelData.to_dict()`、LeRobot 导出、HDF5 writer）统一过白名单，保证产物 schema 合法。设 `TLABEL_STRICT_CAPABILITIES=1` 后，不合规数据在出口直接抛 `ValueError` 而非静默修复。
- **适配器 conformance 守护测试** `tests/unit/conformance/test_capabilities_conformance.py`（78 个用例）：钉死 13 键合规、`contact` 必填、`compliance_level` 禁入、`get_capabilities()` 无参契约、导出产物与 JSON Schema 一致、Python 常量与 schema JSON 同源。

### Fixed

- **LeRobot 写回读不回自己产出的数据集**：`lerobot_create` 产出 LeRobot v2.1 的嵌套布局 `data/chunk-000/episode_000000.parquet`，但 `lerobot_to_tlabel()` / `tlabel_to_lerobot()` 用 `glob("*.parquet")` 只匹配平铺布局，导致 `No parquet files found`。改用 `rglob`，两种布局都能覆盖。
- **LeRobot 写回破坏 v2.1 目录结构**：写回时把全部 parquet 合并成单个 `data/chunk-0000.parquet` 再 `unlink` 原文件，会压平多 episode、使 `meta/episodes.jsonl` 的分片信息失真。改为记录每个文件的行区间后原地逐文件更新。
- **批量导出把 0 帧文件当成功**：适配器对无效输入返回 0 帧 `TLabelData` 而不报错，此前会打印成功并产出空文件。现在单文件与批量路径都校验 `num_frames > 0`。
- **`daimon_dataset` 的无视频降级逻辑作用于已删除的旧键**（`texture_energy` / `edge_density` / `shear_field_*`），是死代码。改为降级 `force_vector` / `slip_velocity` / `object_deformation`。
- **`paxini_gen3` / `tacquad` 的 `get_capabilities()` 签名违规**：前者带 `file_path` 参数、后者缺返回类型标注，均已修正。

### Documentation

- 新增 `MIGRATION.md` 的 v0.23 → v0.24 章节：13 键规则、旧 22 键的三类去向、旧文件读取行为、帧内容变化清单、CLI 变化、严格模式、适配器自查清单。
- `docs/tlabel-format.md` §capabilities 补规范性说明（闭合键集、13 键 vs 14 维的区别、声明与输出必须一致、旧文件兼容）。
- `docs/annotation-spec.md` 补 `capabilities` 契约。
- `docs/spec/tlabel-spec-v1.md` §11 的 D-13 / D-17 / D-18 标记为已解决。

### Tests

- 479 passed, 1 skipped（此前 383 passed，skip 数不变）。新增 78 个 conformance 测试、13 个 `capabilities` 归一化/严格模式测试、5 个 CLI 导出格式别名测试。
- 移除 `tests/unit/test_edge_cases.py` 中对 `TypeError` 跳过的豁免分支——适配器签名已统一，该豁免不再需要。

## [0.23.3] - 2026-10-26

### Fixed

- **LeRobot 写回路径静默丢失向量与枚举字段**（数据损坏，优先级最高）：`tlabel_to_lerobot()` 在把标注写回已有 LeRobot 数据集时，对所有非标量字段一律填 `0.0`。这导致向量字段（`contact_centroid` / `force_vector` / `torque_vector` / `slip_velocity`）和枚举字段（`contact_region` / `manipulation_phase` / `texture_class` / `compliance_level`）被静默抹平——不报错、不告警。0.23.1 的崩溃修复只是把异常改成了静默丢数据。现在两条 LeRobot 路径（`lerobot.py` 写回 / `lerobot_export.py` 新建）共用同一套编码实现，语义一致：向量取 L2 范数、枚举映射类别索引、缺失枚举用 `-1.0` 哨兵。
- **`capabilities` 中的 `compliance_level` 键**：`syntouch`、`tashan_ts_f_a`、`paxini_px6d` 三个适配器与 `demo.py` 的合成能力声明曾把 `compliance_level` 当作能力键输出。它是帧级元字段，不属于能力声明（规范 S-7.4-2 / C-15），已移除。这些文件此前无法通过仓库自带的 JSON Schema 校验（`additionalProperties: false`）。
- **LeRobot 导入的 `capabilities` 键未做白名单**：`lerobot_to_tlabel()` 曾把 parquet 的任意 feature 名直接当作能力键，可能产出 schema 非法文件。现在只保留 13 个合法能力键的交集。
- **内置 demo 数据的旧 `capabilities`**：`tlabel/demo_data/*.json` 携带 22 维时代的旧能力结构（`has_image` / `dimensions` / 旧字段名），与 Schema V2 的 13 键无可靠对应。加载时归一化为最小合法声明，原始结构完整保留在 `episode_info["legacy_capabilities"]`。

### Added

- **`tlabel/core/capabilities.py`**：13 个合法能力键的单一事实源，以及 `validate_capabilities()` / `filter_capabilities()` / `warn_if_invalid_capabilities()`。`TLabelData` 构造时会对不合规的 `capabilities` 发出 `UserWarning`（本版本只告警、不改写输出；出口归一化与 strict 模式留待 0.24.0）。
- **`tlabel/converters/_encoding.py`**：Schema V2 → 14 维向量的共享编码模块，`lerobot.py` 与 `lerobot_export.py` 共用。
- 新增 51 个测试：`tests/unit/test_lerobot_encoding.py`（编码正确性、两条路径一致性、端到端写回回归）与 `tests/unit/test_capabilities.py`（键集与 JSON Schema 一致性、校验/过滤、出口告警、脏源回归）。

### Notes

- 本次修复会改变 `tlabel_to_lerobot()` 的**输出数值**：此前被写成 `0.0` 的向量与枚举字段现在有真实编码。这属于 bug fix，但下游若有基于旧（错误）输出的基线，需要重新生成。
- 仍有 10 个适配器返回旧 22/23 键能力字典（`gelsight` / `paxini` / `daimon` / `daimon_dm_tac` / `tacquad` / `touchd` / `univtac` / `vtouch` / `ycb_slide` / `paxini_gen3`），会在 `TLabelData` 构造时产生告警。迁移计划见 `docs/internal/PIT7_capabilities_migration_plan.md`，安排在 0.24.0 完成。

## [0.23.2] - 2026-10-26

### Added
- **Normative format specification** (`docs/spec/tlabel-spec-v1.md`): a ROS REP-style normative specification for the TLabel data format — numbered normative clauses (`S-x.y-z`), RFC 2119 keywords, per-dimension field definitions (type, unit, requiredness, compliance-level threshold, example), compliance-level (L1-L4) decision rules, adapter conformance requirements, and a conformance-test checklist (C-01…C-16). `docs/tlabel-format.md` and `docs/annotation-spec.md` are now explicitly marked as *informative* and point at the specification.
- **Scenario templates** (`tlabel/scenarios/`): four YAML scenario definitions — `gelsight_grasp`, `paxini_force_control`, `tashan_fingertip`, `xela_uskin` — describing which schema fields, primitives and events matter for each data-collection scenario. New CLI commands: `tlabel template list` and `tlabel template validate [<id>|--all]`.

### Changed
- `pyyaml>=5.4` is now a core dependency (required by the scenario template loader).
- Packaging: `tlabel/scenarios/*.yaml` is now included in the wheel and sdist via `package-data` and a new `MANIFEST.in`.

### Documentation
- 20 documentation/code inconsistencies found while writing the specification are catalogued in `docs/spec/tlabel-spec-v1.md` §11. Notable ones: several `docs/` pages still describe the deprecated pre-v2.0 flat feature space, and `tlabel-format.md` documents top-level keys (`sensor_info`/`episodes`) and enum value sets that do not match the implementation.

## [0.23.1] - 2026-09-07

### Fixed
- Fix LeRobot converter crash when processing non-numeric fields (e.g., texture_class)

### Added
- Add LeRobot integration examples (examples/lerobot_integration.py)
- Add LeRobot Integration section to README (English and Chinese) 

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

## [0.22.3] - 2026-09-03
### Added
- **Schema v2.2 `raw_sensor_values`**: new optional field for raw sensor data retention, split out from `data_quality`
### Changed
- Tashan TS-F-A adapter now retains raw data in `raw_sensor_values` instead of `data_quality`
- `get_capabilities()` standardized to the 14-dimensional Schema V2 format
- Design document updated to v2.2 (three-layer sensor info architecture)

## [0.22.2] - 2026-09-03
### Changed
- Removed redundant sensor information from `provenance`; sensor info now follows the three-layer architecture without duplication

## [0.22.1] - 2026-09-03
### Fixed
- Removed invalid fields (`slip_direction`, `sensor_id`, `sensor_specific`) from the `TLabelSchemaV2` constructor call in `TashanTsFAAdapter`

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
- **Converter base layer** (`converters/base.py`): Unified converter interface (`BaseConverter` with `export()` method) plus a `get_converter()` registry, wrapping `LeRobotConverter` and `FTP1Converter` behind a consistent API

### Fixed
- **lerobot.py `_safe_float()`**: Fixed crash when converting vector/list fields (e.g., `contact_centroid`) — now safely handles vectors (takes magnitude), booleans (0/1), and None (0.0)

### Notes
- **This entry was corrected on 2026-10-26.** An earlier revision listed CLI commands (`tlabel convert`, `tlabel batch-convert`, `tlabel list-adapters`, `tlabel adapter-info`) and a 28-test suite for them. Those were developed on the `feat/v0.19.0-convert-cli` branch but were never merged into `main`, and therefore were never part of any published release. Only the two items above actually shipped in 0.19.0.
- **Known gap:** `converters/base.py` and the `_safe_float()` fix were not carried forward when development resumed on `main` for 0.20.0, and are absent from 0.23.1. Tracked for restoration.

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

