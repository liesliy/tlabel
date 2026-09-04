# TLabel

**A Unified Annotation Framework for Cross-Sensor Tactile Manipulation Data**

[![PyPI](https://img.shields.io/pypi/v/tlabel)](https://pypi.org/project/tlabel/)
[![Tests](https://github.com/liesliy/tlabel/actions/workflows/tests.yml/badge.svg)](https://github.com/liesliy/tlabel/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://img.shields.io/pepy/dt/tlabel)](https://pepy.tech/projects/tlabel)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.22227847.svg)](https://doi.org/10.5281/zenodo.22227847)
[![中文文档](https://img.shields.io/badge/README-中文-red)](README_CN.md)

TLabel is the first cross-sensor tactile annotation schema with capability declarations and Compliance Level stratification. It enables heterogeneous tactile sensors — regardless of operating principle — to produce compatible 14-dimensional semantic annotations while preserving their unique strengths.

> **TL;DR** — The Unicode for tactile data: one standard schema, every sensor.

## Why TLabel?

Tactile datasets today ship as raw sensor signals without semantic annotations. Each sensor type demands its own ad-hoc processing, and results from different sensors cannot be compared or fused. TLabel addresses this by:

- **Standardizing annotations** — 14 dimensions covering spatial, mechanical, surface, dynamic, and meta perceptions
- **Declaring capabilities** — each adapter explicitly states which dimensions it can and cannot annotate
- **Stratifying compliance** — Compliance Level (L1–L4) ensures every sensor participates at its appropriate information density
- **Enabling cross-sensor comparison** through a shared output format

## Quick Start

### Install

```bash
pip install tlabel
```

### Load and explore data

```python
import tlabel

# Load tactile data (auto-detects sensor format)
data = tlabel.load("path/to/sensor_data.pkl")

# Or try the built-in demo — no files needed
data = tlabel.demo("gelsight")

# Inspect annotation metadata
print(data.describe())
```

### Interactive annotation (Jupyter)

```python
data.review()  # Bilingual annotation panel (Chinese / English)
```

### Export to training formats

```python
data.export("output.json")              # JSON / CSV
data.export_ftp1("output.zarr")         # FTP-1 Zarr for foundation models

from tlabel.converters import tlabel_to_lerobot
tlabel_to_lerobot("annotations.json", "lerobot_episode/")  # LeRobot
```

### CLI

```bash
tlabel list                       # List all registered adapters
tlabel info gelsight              # Adapter details & compliance level
tlabel validate data.json         # Schema compliance check
```

### Optional dependencies

```bash
pip install tlabel[gelsight]      # GelSight / DIGIT (.pkl)
pip install tlabel[paxini]        # PaXini PXCap (.h5)
pip install tlabel[daimon]        # Daimon DM-TacClaw (.parquet)
pip install tlabel[ftp1]          # FTP-1 export (zarr)
pip install tlabel[all]           # Everything
```

## Schema — 14 Dimensions, 4 Compliance Levels

TLabel defines **14 semantic dimensions** with **Compliance Levels (L1–L4)** indicating annotation completeness:

| Level | Name | Required Fields | Example Sensors |
|-------|------|----------------|-----------------|
| **L1** | Basic Tactile | contact, centroid, slip, confidence | Single-point resistive, proximity |
| **L2** | Force-Aware | L1 + force_magnitude | Paxini, YCB-Slide, GelSight |
| **L3** | Full-Vector | L2 + force_vector [3D] | ToucHD, calibrated DM-TAC |
| **L4** | Rich-Semantic | L3 + all optional fields | BioTac, next-gen multimodal |

The 14 dimensions span: `contact`, `contact_centroid`, `force_magnitude`, `slip_event`, `confidence`, `compliance_level`, `contact_region`, `force_vector`, `torque_vector`, `slip_velocity`, `manipulation_phase`, `texture_class`, `object_deformation`, `temperature`.

Full dimension spec → [docs/tlabel-format.md](docs/tlabel-format.md)

## Supported Sensors

**Dataset Adapters** (offline): GelSight/DIGIT (L3) · Daimon DM-TacClaw (L3) · PaXini PXCap (L2) · UniVTAC (L3) · TacQuad/AnyTouch (L3) · VTouch (L3) · YCB-Slide (L3) · XELA uSkin/UniTac-NV (L1)

**Real-time Adapters** (hardware): PaXini GEN3 (L2) · Daimon DM-Tac (L3)

Adding a new sensor takes ~30 min — fork [contrib/adapter-template/](contrib/adapter-template/)

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Schema                                │
│  14 semantic dimensions + Compliance Level L1-L4│
├─────────────────────────────────────────────────┤
│  Layer 2: Adapters                              │
│  DataAdapterBase │ SensorAdapterBase             │
│  (7 built-in + community-extensible)            │
├─────────────────────────────────────────────────┤
│  Layer 3: Downstream                            │
│  Feature derivation · Augmentation · Export      │
│  PredictEngine · FTP-1 · LeRobot · RLDS · ROS2 │
└─────────────────────────────────────────────────┘
```

## Paper

**TLabel: A Unified Annotation Framework for Cross-Sensor Tactile Manipulation Data**

*Xi Luo, Sheng Wu* (Niuxiu Tech)

Submitted to *SoftwareX*, 2026. Manuscript: SOFTX-S-26-01665

[[PDF]](paper/tlabel-softwarex.pdf) · LaTeX source: [`paper/`](paper/)

## Citation

```bibtex
@software{tlabel2026,
  title  = {TLabel: A Sensor-Agnostic Tactile Data Annotation Toolkit and Format Standard},
  author = {Wu, Sheng and Luo, Xi},
  year   = {2026},
  url    = {https://github.com/liesliy/tlabel}
}
```

## Documentation

| Document | Description |
|----------|-------------|
| [TLabel Format Spec](docs/tlabel-format.md) | Complete annotation schema specification |
| [Annotation Spec](docs/annotation-spec.md) | Annotation methodology and guidelines |
| [Design Document](docs/TLabel_Design_Document.md) | Core design decisions and architecture |
| [中文文档](README_CN.md) | Chinese README |

## Contributing

TLabel is designed to be extensible. Add your sensor in ~30 minutes:

1. Fork [contrib/adapter-template/](contrib/adapter-template/)
2. Subclass `DataAdapterBase` or `SensorAdapterBase`
3. Submit a PR or publish as a standalone package

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT © 2026 Niuxiu Tech

---

<p align="center">
  <strong>TouchLabel AI</strong> — Tactile Data Annotation Infrastructure<br>
  <a href="https://github.com/liesliy/tlabel">GitHub</a> ·
  <a href="https://pypi.org/project/tlabel/">PyPI</a> ·
  <a href="https://discord.gg/2ab8EWaBM">Discord</a><br>
  <a href="https://www.niuxutech.com">Niuxiu Tech</a> · Hangzhou, China
</p>
