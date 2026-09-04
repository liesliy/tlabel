<div align="center">

# 🦞 TLabel

**统一的触觉数据标注标准与工具集**

加载任意传感器 · 统一格式标注 · 多框架导出

[![PyPI](https://img.shields.io/pypi/v/tlabel?color=e85d75&label=PyPI)](https://pypi.org/project/tlabel/)
[![Python](https://img.shields.io/pypi/pyversions/tlabel)](https://pypi.org/project/tlabel/)
[![License](https://img.shields.io/pypi/l/tlabel)](LICENSE)
[![Downloads](https://img.shields.io/pepy/dt/tlabel?color=blue)](https://pepy.tech/projects/tlabel)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.22227847.svg)](https://doi.org/10.5281/zenodo.22227847)
[![English](https://img.shields.io/badge/README-English-blue)](README.md)

</div>

---

## TLabel 是什么？

每个触觉传感器都有自己的数据格式，不同传感器之间无法直接对比或融合。TLabel 定义了一套**统一的标注 Schema**（14 个语义维度，4 级合规等级），让异构传感器自动输出兼容的语义标注——类比**触觉数据的 Unicode**。

```
 GelSight .pkl ──┐                        ┌── JSON / CSV
 PaXini .h5 ─────┤   TLabel 适配器         ├── FTP-1 Zarr
 Daimon .parquet─┤   ─────────────────►   ├── LeRobot / RLDS
 VTouch .h5 ─────┤                        └── ROS2
 任意格式 ────────┘
```

**核心特点：**
- **14 维语义标注** — 空间、力学、表面、动态、元信息全覆盖
- **能力声明** — 每个适配器明确声明能标什么、不能标什么
- **合规分级 (L1–L4)** — 不同传感器按自身能力参与，不强制对齐
- **跨传感器兼容** — 统一输出格式，支持直接对比和融合

---

## 快速开始

```bash
pip install tlabel
```

```python
import tlabel

# 加载数据（自动识别传感器格式）
data = tlabel.load("path/to/data")

# 内置 demo（无需任何文件）
data = tlabel.demo("gelsight")

# 查看标注元数据
print(data.describe())

# 交互式标注面板（Jupyter 中英双语）
data.review()

# 导出
data.export("output.json")
data.export_ftp1("out.zarr")
```

```bash
# CLI
tlabel list                    # 查看所有已注册适配器
tlabel info gelsight           # 适配器详情与合规等级
tlabel validate data.json      # Schema 合规性检查
```

### 可选依赖

```bash
pip install tlabel[gelsight]   # GelSight / DIGIT
pip install tlabel[paxini]     # PaXini PXCap
pip install tlabel[daimon]     # Daimon DM-TacClaw
pip install tlabel[ftp1]       # FTP-1 导出（zarr）
pip install tlabel[all]        # 全部安装
```

---

## Schema — 14 维度，4 级合规

| 合规等级 | 说明 | 必填字段 | 典型传感器 |
|:--------:|------|---------|-----------|
| **L1** | 基础触觉 | contact, centroid, slip, confidence | 单点电阻式、接近式 |
| **L2** | 力感知 | L1 + force_magnitude | PaXini, YCB-Slide, GelSight |
| **L3** | 完整向量 | L2 + force_vector [3D] | ToucHD, DM-TAC |
| **L4** | 丰富语义 | L3 + 所有可选字段 | BioTac, 新一代多模态 |

14 个语义维度：`contact`, `contact_centroid`, `force_magnitude`, `slip_event`, `confidence`, `compliance_level`, `contact_region`, `force_vector`, `torque_vector`, `slip_velocity`, `manipulation_phase`, `texture_class`, `object_deformation`, `temperature`

📖 完整维度规范 → [docs/tlabel-format.md](docs/tlabel-format.md)

---

## 支持的传感器

**数据集适配器**（离线加载）：GelSight/DIGIT (L3) · Daimon DM-TacClaw (L3) · PaXini PXCap (L2) · UniVTAC (L3) · TacQuad/AnyTouch (L3) · VTouch (L3) · YCB-Slide (L3) · XELA uSkin/UniTac-NV (L1)

**实时适配器**（硬件直连）：PaXini GEN3 (L2) · Daimon DM-Tac (L3)

📖 添加新传感器仅需 ~30 分钟 — Fork [contrib/adapter-template/](contrib/adapter-template/)

---

## 架构

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Schema                                │
│  14 维语义标准 + 合规等级 (L1-L4)               │
├─────────────────────────────────────────────────┤
│  Layer 2: Adapters                              │
│  DataAdapterBase │ SensorAdapterBase             │
│  （7 个内置 + 社区可扩展）                       │
├─────────────────────────────────────────────────┤
│  Layer 3: Downstream                            │
│  特征派生 · 数据增强 · 导出                       │
│  PredictEngine · FTP-1 · LeRobot · RLDS · ROS2 │
└─────────────────────────────────────────────────┘
```

---

## 论文

**TLabel: A Unified Annotation Framework for Cross-Sensor Tactile Manipulation Data**

*Xi Luo, Sheng Wu*（牛宿科技）

已投稿至 *SoftwareX*, 2026. 稿件编号: SOFTX-S-26-01665

[[PDF]](paper/tlabel-softwarex.pdf) · LaTeX 源码：[`paper/`](paper/)

---

## 引用

```bibtex
@software{tlabel2026,
  title  = {TLabel: A Sensor-Agnostic Tactile Data Annotation Toolkit and Format Standard},
  author = {Wu, Sheng and Luo, Xi},
  year   = {2026},
  url    = {https://github.com/liesliy/tlabel}
}
```

---

## 文档

| 文档 | 说明 |
|------|------|
| [TLabel 格式规范](docs/tlabel-format.md) | 完整标注 Schema 规范 |
| [标注方法论](docs/annotation-spec.md) | 标注方法与指南 |
| [设计文档](docs/TLabel_Design_Document.md) | 核心设计决策与架构 |

---

## 参与贡献

TLabel 天生可扩展，30 分钟即可添加你的传感器适配器。详见 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 许可证

[MIT](LICENSE) © 2026 牛宿科技

---

<div align="center">

**如果 TLabel 对你有帮助，欢迎给个 ⭐**

[⭐ Star](https://github.com/liesliy/tlabel/stargazers) · [📦 PyPI](https://pypi.org/project/tlabel/) · [💬 Discord](https://discord.gg/2ab8EWaBM)

**技术服务：** 定制适配器开发 · 数据管线咨询 · 具身智能工具链
**联系：** 微信 `wxid_olqx5z6trmtn21` · 邮箱 `luoxi@touchlabelai.cn`

</div>
