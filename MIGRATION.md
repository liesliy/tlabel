# Migration Guide

本文件按版本倒序记录 breaking change。**当前最新：v0.23 → v0.24**。

| 版本跃迁 | 主题 |
|---|---|
| [v0.23 → v0.24](#v023--v024) | `capabilities` 键空间收敛到 13 键、适配器 `load()` 走 `extract_schema`、CLI 导出接口调整 |
| [v0.16 → v0.17](#v016--v017) | 移除 22 维 `tlabel_v2`，统一 Schema V2（14 维） |

---

## v0.23 → v0.24

### 一句话总结

`capabilities` 从"适配器随手写的任意字典"变成**闭合的 13 键布尔集合**；同时 9 个适配器的 `load()` 改走 `extract_schema()`，因此**导出文件里的帧内容会变**。

这两件事必须一起迁移：键空间按"文件里真实有什么"来声明，而文件内容刚刚变了。

### 1. `capabilities` 只允许 13 个键

合法键集合（`tlabel/core/capabilities.py` 的 `CAPABILITY_KEYS`，与 `schema/tlabel-schema.json` 同源）：

```
contact  contact_centroid  contact_region  force_magnitude  force_vector
torque_vector  slip_event  slip_velocity  manipulation_phase
texture_class  object_deformation  temperature  confidence
```

规则：
- **`contact` 必填**（S-7.4-1）。做不到就写 `"contact": false`，不要省略。
- **`compliance_level` 禁止出现**（S-7.4-2）。它是帧级元字段，不是能力。
- 值为 `bool`，不接受 `0`/`1`/`"true"`。
- 声明 `true` 就必须在帧里真的给出非 null 值（S-6.3-2、S-6.3-3）。

### 2. 旧 22 键的去向

旧 `capabilities` 字典来自 22 维 `tlabel_v2` flat 特征空间。按语义分三类：

**A. 迁移到新键（语义有降级，但可以继承）**

| 旧键 | 新键 | 说明 |
|---|---|---|
| `contact` | `contact` | 直接对应（float 0/1 → bool） |
| `centroid_x` | `contact_centroid` | 近似：旧数据只有 x，y 为占位值 |
| `force_magnitude` | `force_magnitude` | 同名但**单位不诚实**：视觉触觉是形变代理量，不是牛顿 |
| `force_direction` | `force_vector` | 仅在能合成真实 3D 力时继承（部分适配器 Fy 恒 0） |
| `slip_event` | `slip_event` | 直接对应（注意视觉系是图像梯度代理，非物理滑移检测） |
| `deformation_magnitude` | `object_deformation` | 语义近似映射 |
| `optical_flow_*` | `slip_velocity` | 间接：经 `extract_schema` 换算成滑移速度向量 |

**B. 直接丢弃（13 维无语义对应，仍保留在帧级特征层）**

`contact_area`、`force_peak`、`deformation_magnitude_peak`、`texture_energy`、`edge_density`、
`normal_field_magnitude`、`normal_field_variance`、`shear_field_magnitude`、`shear_field_direction`、
`delta_force_normal`、`delta_force_shear`、`friction_cone_ratio`、`slip_entropy`、
`temporal_deformation_rate`、`contact_transition`

这些值**没有从库里消失**，只是不再是"能力声明"。帧级 22 维特征层短期保留。

**C. 旧字典里没有、需要重新判断的新键**

`contact_region`、`torque_vector`、`temperature`、`confidence`、`manipulation_phase`——
按各适配器 `extract_schema` 的实际产出决定。

### 3. 旧文件仍然能读

读路径是宽容的：

```python
from tlabel import load

data = load("legacy_file.json")   # 不报错
data.capabilities                 # 已是 13 键合法子集
data.episode_info["legacy_capabilities"]
# {'has_image': True, 'has_force': True, 'has_optical_flow': True, ...}  ← 原始值完整保留
```

- 非法键在读取时被剔除，同时**原样保留**在 `episode_info["legacy_capabilities"]`，不丢信息。
- 同时发一条 `UserWarning` 说明剔除了哪些键。
- 写路径是严格的：所有出口（`TLabelData.to_dict()`、LeRobot 导出、HDF5 writer）都过
  `normalize_capabilities()`，产物一定合法。

### 4. 帧内容会变（`load()` 接回 `extract_schema`）

此前只有 `gelsight` 的 `load()` 真正调用 `self.extract_schema()`，其余 9 个适配器走
`TLabelSchemaV2.from_tlabel_v1(tlabel_v2)`。后者只做字段改名，会**丢掉 `extract_schema` 里写的
L3 增强**：

| 字段 | 之前 | 现在 |
|---|---|---|
| `compliance_level` | 恒为 `"L1"` | 按实际数据判定（`vtouch` 实测 L1 → L2） |
| `slip_velocity` | 恒 `null` | 由 optical flow 换算（`vtouch` 实测 `[1.7678, 1.7678]`） |
| `force_vector` | `[Fx, 0, Fz]`，Fy 恒 0 | 由 shear 场合成的真实 3D 向量 |

**受影响适配器**：`daimon_dataset`、`daimon_dm_tac`、`paxini_dataset`、`paxini_gen3`、`tacquad`、
`touchd`、`univtac`、`vtouch`、`ycb_slide`。

如果你的下游有基于旧（错误）输出的基线，需要重新生成。

### 5. `get_capabilities()` 签名统一为无参

```python
# 旧
caps = adapter.get_capabilities(file_path)

# 新
caps = adapter.get_capabilities()
```

`TLabelAdapter`（读 `.json` 文件）的能力由文件决定，额外提供：

```python
caps = adapter.get_capabilities_for_file("episode.json")
```

无参调用返回最近一次 `load()` 的归一化结果；没 load 过时返回保守声明 `{"contact": False}`。

### 6. CLI 变化

| 变化 | 说明 |
|---|---|
| `-f lerobot` 语义不变 | 仍是"新建 LeRobot 数据集"（内部映射到 `lerobot_create`） |
| 新增 `-f lerobot_writeback` | 写回已有 LeRobot 数据集（显式触发，避免误覆盖） |
| 新增 `--overwrite` | 仅 `lerobot_writeback` 生效 |
| 新增 `--sensor` / `--areas` | FTP-1 导出参数（传感器型号 / 手部功能区，逗号分隔整数） |
| 新增 `--batch` | 批量导出：输入目录，递归按适配器支持的扩展名处理，保持子目录结构 |
| 新增 `-f` 别名解析 | 非法格式会列出全部 8 个可用格式 |

`--batch` 的失败语义：单个文件失败不中断，其余继续；退出码非 0 表示有失败项。
`lerobot_writeback` 拒绝批量（写回必须指定单个数据集目录）。
读出 0 帧的文件按失败处理（不再当作成功）。

### 7. `ExporterResult.stats` 键统一

旧代码里同一个含义有三个键名（`frames` / `num_frames` / `time_steps`）。现在统一为 `num_frames`：

```python
# 旧
n = result.stats.get("frames") or result.stats.get("time_steps")

# 新
n = result.stats["num_frames"]
```

### 8. 严格模式（可选）

默认是"修复 + 告警"。CI 或产线想直接卡住不合规数据，打开环境变量：

```bash
export TLABEL_STRICT_CAPABILITIES=1
```

此后任何不合规的 `capabilities`（非法键、缺 `contact`、非 bool 值）都会在出口抛 `ValueError`，
而不是静默归一化。也可以按调用传参：`normalize_capabilities(caps, strict=True)`。

### 9. 适配器作者自查清单

- [ ] `get_capabilities()` 无参，返回 13 键的子集，含 `contact`
- [ ] 没有 `compliance_level` 键
- [ ] 每个 `true` 的键，在 `load()` 产出的帧里真的有非 null 值（不是只在 `extract_schema` 里有）
- [ ] `load()` 调用了 `self.extract_schema()`，没有绕过它直接 `from_tlabel_v1()`
- [ ] `python -m pytest tests/unit/conformance/test_capabilities_conformance.py`

---

## v0.16 → v0.17

### ⚠️ Breaking Change

**v0.17.0 彻底移除了旧版 `tlabel_v2` (22维) 数据格式兼容。** 所有数据路径现在只走 **Schema V2 (14维结构化)**。

这意味着：
- 旧的 `TLabelFrame.tlabel_v2` 字典属性已删除
- 旧版生成的数据文件无法直接加载
- 所有适配器输出统一为 `TLabelSchemaV2` 对象

### 为什么这么做？

v0.16 及之前的 22 维架构存在根本问题：

1. **数据质量不可控**：不是所有传感器都能提供全部 22 维，大量字段填 0/None，下游无法区分"真实为 0"和"传感器不支持"
2. **同名字段语义不一致**：不同适配器的同名维度可能含义不同
3. **下游模型引入噪声**：模型无法判断数据质量，训练时混入大量无效字段

v0.17 通过 **14 维语义 Schema + Compliance Level (L1-L4)** 解决这些问题：
- 每个字段有明确的物理/语义定义
- Required/Optional 清晰标注
- 适配器如实声明能力等级，不再"假装完整"

### 具体变化

#### ❌ 移除的 API

| 移除项 | 替代方案 |
|--------|----------|
| `TLabelFrame.tlabel_v2` | `TLabelFrame.schema_v2` |
| `TLabelFrame._original_tlabel` | 不再需要 |
| `predict/_compat.py` 兼容层 | 不需要 |
| `_detect_schema_version()` | 不需要 |
| `_detect_use_schema_v2()` | 不需要 |
| `quality/scorer.py` 中的 `LEGACY_V2_DIMS` | 使用 14 维 Schema |
| `augment/transforms.py` 中的 `LEGACY_V2_FEATURE_NAMES` | 使用 V2 `FEATURE_NAMES` |
| `export/writer.py` 中的 legacy 22 列 CSV 导出 | 只有 V2 展开列 (20列) |

#### ✅ 新增的 API

| 新增项 | 说明 |
|--------|------|
| `TLabelFrame.schema_v2` | 必填，类型 `TLabelSchemaV2` |
| `TLabelSchemaV2` | 14 维 dataclass，含 Compliance Level |
| `TLabelSchemaV2.compliance_level` | 枚举 L1/L2/L3/L4 |
| `TLabelSchemaV2.force_magnitude` | 标量力 (N)，L2+ 约定必填 |
| `DataAdapterBase.extract_schema()` | 将原始数据转换为 `TLabelSchemaV2` |
| `DataAdapterBase.default_compliance_level` | 适配器声明能力等级 |
| `SensorAdapterBase.extract_schema()` | 实时传感器的 Schema V2 提取 |

#### 🔄 行为变化

| 模块 | 变化 |
|------|------|
| `TLabelFrame.__init__()` | `schema_v2` 参数改为必填，传 None 抛 ValueError |
| `TLabelFrame.from_dict()` | 只支持 Schema V2 格式 |
| `TLabelFrame.to_dict()` | 只输出 Schema V2 格式 |
| `export/writer.py` | CSV 导出 20 列 (V2 展开)，HDF5 只写 V2 路径 |
| `predict/engine.py` | 预标注引擎完全重写，只读 Schema V2 字段 |
| `predict/force_estimator.py` | 力推断结果写入 `schema_v2.force_magnitude` |
| `predict/ml_engine.py` | 特征字段缩减为 5 个 V2 核心字段 |
| `quality/scorer.py` | 评分规则基于 14 维 Schema + Compliance Level |
| 所有 10 个适配器 | 统一改为 `TLabelFrame(schema_v2=TLabelSchemaV2.from_tlabel_v1(...))` |

### 迁移代码示例

#### 1. 构造 TLabelFrame

**旧代码 (v0.16):**
```python
frame = TLabelFrame(
    frame_idx=0,
    timestamp_s=0.0,
    tlabel_v2={"contact": 1.0, "force_x": 0.5, ...}  # 22维字典
)
```

**新代码 (v0.17):**
```python
from tlabel.core.schema import TLabelSchemaV2

sv2 = TLabelSchemaV2(
    contact=True,
    contact_centroid=[32.5, 48.2],
    force_magnitude=1.5,
    compliance_level="L2",
)
frame = TLabelFrame(
    frame_idx=0,
    timestamp_s=0.0,
    schema_v2=sv2,  # 必填
)
```

#### 2. 读取标注字段

**旧代码:**
```python
force_x = frame.tlabel_v2.get("force_x", 0.0)
contact = frame.tlabel_v2.get("contact", 0.0)
```

**新代码:**
```python
force_mag = frame.schema_v2.force_magnitude or 0.0
contact = 1.0 if frame.schema_v2.contact else 0.0
# 或使用便捷属性
contact = frame.contact
force_mag = frame.force_magnitude
```

#### 3. 适配器开发

**旧代码:**
```python
class MyAdapter(BaseAdapter):
    def load(self, path):
        frames = []
        for raw in data:
            frame = TLabelFrame(tlabel_v2={"contact": 1.0, ...})
            frames.append(frame)
        return TLabelData(frames=frames)
```

**新代码:**
```python
from tlabel.adapters.base import DataAdapterBase
from tlabel.core.schema import TLabelSchemaV2

class MyAdapter(DataAdapterBase):
    default_compliance_level = "L2"

    def extract_schema(self, raw_frame_data) -> TLabelSchemaV2:
        return TLabelSchemaV2(
            contact=True,
            contact_centroid=[raw_frame_data["cx"], raw_frame_data["cy"]],
            force_magnitude=raw_frame_data["force"],
            compliance_level=self.default_compliance_level,
        )

    def load(self, path, **kwargs):
        frames = []
        for raw in data:
            sv2 = self.extract_schema(raw)
            frame = TLabelFrame(schema_v2=sv2, ...)
            frames.append(frame)
        return TLabelData(frames=frames, sensor_info=self.get_sensor_info())
```

#### 4. 导出 CSV

**旧代码 (22列):**
```python
# 输出包含 force_x, force_y, force_z, deformation_1..5 等 22 列
tlabel.export(data, "output.csv")
```

**新代码 (20列 — V2 展开):**
```python
# 输出包含 contact, contact_centroid_x, contact_centroid_y, force_magnitude,
# force_vector_x, force_vector_y, force_vector_z, ... 等 20 列
tlabel.export(data, "output.csv")
```

### Compliance Level 速查

| Level | 名称 | 必填字段 | 典型传感器 |
|-------|------|----------|-----------|
| L1 | Basic | contact, contact_centroid, slip_event, confidence | TacQuad (仅图像) |
| L2 | Force-Aware | L1 + force_magnitude | PaXini, YCB-Slide, VTouch |
| L3 | Full-Vector | L2 + force_vector [Fx,Fy,Fz] | ToucHD, UniVTAC, GelSight(标定后) |
| L4 | Rich-Semantic | L3 + 所有 Optional 字段 | 理想化全传感器 |

---

## 需要帮助？

- GitHub Issues: https://github.com/liesliy/tlabel/issues
- 设计文档: https://github.com/liesliy/tlabel/blob/main/docs/TLabel_Design_Document.md
- 规范原文: [docs/spec/tlabel-spec-v1.md](docs/spec/tlabel-spec-v1.md) §7.4 capabilities
