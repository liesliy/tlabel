# Grabette 双阵列触觉标注教程

本教程演示如何为 Grabette 双阵列压力传感器数据添加标注。

Grabette 是**纯压力阵列**数据集：触觉数据直接存在 Parquet 里，没有视频解码，也不需要 ffmpeg。如果你的机器上装不了 OpenCV 或 ffmpeg，这是目前最轻的一个适配器。

## 前置准备

```bash
pip install tlabel[grabette]
```

只装 `pyarrow`。不需要 OpenCV，不需要 ffmpeg。

## 步骤 1: 准备数据

Grabette 使用 LeRobot v3.0 目录结构：

```
grabette-tactile-full-synced/
├── meta/
│   ├── info.json                      # 数据集元信息
│   ├── stats.json                     # 各列统计量
│   ├── tasks.parquet                  # 任务描述映射
│   └── episodes/chunk-000/file-000.parquet
├── data/
│   └── chunk-000/file-000.parquet     # 触觉 + 本体数据（就是全部标注来源）
└── videos/
    ├── observation.images.right_cam0/chunk-000/file-000.mp4
    └── observation.images.right_cam1/chunk-000/file-000.mp4
```

**关键说明：**

- 触觉数据在 `data/**/*.parquet` 里，**不在视频里**。`videos/` 下的两路摄像头（各 720×960 h264）占整个数据集的 99% 体积，但适配器完全不读它们——可以只下载 `meta/` 和 `data/`，省掉 240 MB 里的 239 MB。
- `meta/info.json` 里的 `data_path` 和 `video_path` 是模板字符串（`chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`），不要当真实路径用。
- **`meta/tasks.parquet` 有个坑**：任务文本是 pandas index 而不是普通列（parquet 的 pandas metadata 里写 `index_columns: ["task"]`）。自己写脚本读的时候，`df.columns` 里只有 `task_index`，直接取列名会拿到数字而不是任务描述。

### 触觉列

| 列名 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `observation.tactile.right_sensor_1` | 6×6 | int16 | 右手阵列 1，纯法向压力 |
| `observation.tactile.right_sensor_2` | 4×8 | int16 | 右手阵列 2，纯法向压力 |

两个阵列**尺寸不同**，这是 Grabette 的特点，不是数据错误。

实测统计（19,353 帧全量）：

| 阵列 | 最小值 | 最大值 | 均值 | 非零帧占比 |
|---|---|---|---|---|
| sensor_1 | 0 | 1774 | 31.38 | 38.2% |
| sensor_2 | 0 | 2133 | 39.60 | 39.3% |

**基线恒为 0**——未接触时全零，不是"中位数基线"。所以接触判定用 `总压力 > 0` 就够了，不需要减基线。

## 步骤 2: 加载数据

```python
import tlabel

# 传目录：自动在 data/**/*.parquet 里找数据文件
data = tlabel.load("path/to/grabette-tactile-full-synced/", adapter="grabette")

print(f"加载了 {data.num_frames} 帧")
print(f"任务: {data.episode_info['task']}")
```

也可以直接传 parquet 文件，或指定 episode：

```python
data = tlabel.load("path/to/data/chunk-000/file-000.parquet",
                   adapter="grabette", episode_index=17)
```

**发生了什么：**

1. 向上找 `meta/info.json`，读 `fps`（50）、`robot_type`、`total_episodes`（48）
2. 读 `meta/tasks.parquet`，把 `task_index` 映射到任务描述
3. 用 `pyarrow` 的 `iter_batches` 按批（2048 行）读 Parquet，不整表载入
4. 逐帧对两个阵列分别算：总压力、压力加权质心、峰值、有效 taxel 数
5. 两个阵列各自归一化到 [0,1]² 后，**按受力大小加权合并**成一个质心
6. 与上一帧比较质心位移和压力变化，判定滑移
7. 跑 6 状态机推断操作阶段

## 步骤 3: 查看和修正标注

```python
data.review()
```

**Grabette 特有的注意事项：**

### force_magnitude 是无量纲的

Grabette 没有公开触觉读数的物理单位，也没有提供标定系数。所以：

- `force_magnitude` = **网格压力积分**，是无量纲原始读数，**不是牛顿值**
- `slip_velocity` = **归一化坐标/秒**，**不是 mm/s**
- `object_deformation` = **无量纲压力变化率**，**不是毫米**

适配器不会把 0 填进去假装有值，也不会编一个单位。如果你手上有标定系数，`sensor_specific` 里保留了 `raw_grid_1` / `raw_grid_2`（原始 int16 二维数组），可以自己换算。

```python
frame = data[100]
print(frame.schema_v2.force_magnitude)   # 例如 2847.0，无量纲
print(frame.sensor_specific["raw_grid_1"])  # 6×6 原始读数
```

### contact_centroid 是加权合并的，不是空间拼接

两个阵列尺寸不同（6×6 和 4×8），适配器的做法是：

1. 各自归一化到 [0,1]² 算质心
2. 按各自总压力加权平均

这意味着"传感器 1 的左侧"和"传感器 2 的左侧"在合并后被视为同一位置。数据集没有文档化两个阵列的物理相对位置，所以**这是适配器的建模约定，不是数据集的物理事实**。如果你知道两个阵列的实际空间关系，建议直接看 `sensor_specific` 里的原始网格自己算。

### slip 阈值是这个数据集标定的

`slip_event` 的判定条件：

```
质心位移 > 0.05  或  压力相对变化 > 0.25
```

这两个数是从该数据集 7,596 个"接触→接触"帧对的实测分布取的 p95。换成别的传感器应该重新标定：

```python
data = tlabel.load(path, adapter="grabette",
                   slip_shift_threshold=0.08,
                   slip_pressure_threshold=0.30)
```

### manipulation_phase 的取值

用的是和 Daimon 适配器一致的 6 状态机：

`idle` → `initial_contact` → `stable_contact` → `slip` / `release` → `re_contact`

Schema 里 `manipulation_phase` 字段没有 enum 强约束，下游别假设只有某一套值。

**常见问题及修正方法：**

| 问题 | 表现 | 修正方法 |
|------|------|----------|
| 任务描述是数字 | `task` 显示 `0` | 读 `meta/tasks.parquet` 时没处理 pandas index，升级到 v0.25.0+ |
| 接触区域总是同一侧 | `contact_region` 只有 `*_left` | 单个 episode 内可能是真实物理；先跑全量再判断 |
| 接触率异常高 | 几乎每帧都 contact | 检查是否误用了减基线逻辑；Grabette 基线就是 0 |
| 找不到 parquet | `FileNotFoundError` | 传目录时适配器只找 `data/**/*.parquet` 并排除 `meta/` |

## 步骤 4: 导出标注

### 命令行

```bash
# 导出 TLabel JSON
tlabel export path/to/grabette/ ep0.json -f json -a grabette --episode-index 0

# 自校验
tlabel validate ep0.json
```

输出示例：

```
🔍 校验中: ep0.json
============================================================
ℹ️ 共 330 帧
ℹ️ capabilities 声明 9/13 个合法能力键
============================================================
结果: 0 错误, 0 警告, 2 信息

✅ 校验通过
```

### Python API

```python
data.export("grabette_annotated.json")
data.export("grabette_annotated.csv")
```

## 进阶技巧

### 批量处理 48 个 episode

```python
from tlabel.adapters.grabette import GrabetteAdapter

adapter = GrabetteAdapter()
ds = "path/to/grabette-tactile-full-synced/"

for ep in range(48):
    data = adapter.load(ds, episode_index=ep)
    contacts = sum(1 for f in data.frames if f.schema_v2.contact)
    print(f"episode {ep:2d}: {len(data.frames):4d} 帧, 接触 {contacts:4d} "
          f"({contacts / max(len(data.frames), 1):.0%})")
```

实测全量 19,353 帧、48 个 episode，峰值内存 80.7 MB，耗时约 49 秒。

### 流式处理（内存恒定）

```python
adapter = GrabetteAdapter()

for frame in adapter.iter_frames(ds, episode_index=0):
    ss = frame.sensor_specific
    print(frame.frame_idx, ss["sensor_1_total"], ss["sensor_2_total"])
```

`iter_frames` 和 `load` 走同一条流式路径，内存与 episode 长度无关。Grabette 数据只有 1.55 MB，这里省不了多少内存，但换了大数据集就是必需的。

### 分别看两个阵列

合并质心会丢信息。需要区分时直接看 `sensor_specific`：

```python
for f in data.frames[:10]:
    ss = f.sensor_specific
    print(f"s1={ss['sensor_1_total']:6.0f} "
          f"s2={ss['sensor_2_total']:6.0f} "
          f"s1_taxels={ss['sensor_1_active_taxels']:2d} "
          f"s2_taxels={ss['sensor_2_active_taxels']:2d}")
```

## 常见问题

### Q: 必须下载视频吗？

A: 不用。适配器只读 `meta/` 和 `data/`。从 HuggingFace 拉数据时用 `--include "meta/*" "data/*"` 跳过 `videos/`，239.7 MB 会变成 1.7 MB。

### Q: 为什么 compliance_level 是 L2 不是 L1？

A: L1 要求只做接触判定，不输出任何力相关量。Grabette 有完整的压力阵列，可以算出网格压力积分（`force_magnitude`）和质心位移速度（`slip_velocity`），所以是 L2。但因为没有标定系数、拿不到牛顿值和 mm/s，够不上 L3。

### Q: 能算 force_vector 吗？

A: 不能。Grabette 每个 taxel 只有单轴法向压力，没有切向分量，所以 `force_vector` 和 `torque_vector` 恒为 `None`，`get_capabilities()` 里也标了 `False`。这是传感器物理限制，不是适配器没实现。

### Q: 下载地址？

A: `CarolinePascal/grabette-tactile-full-synced`（Apache-2.0）。国内访问 `huggingface.co` 可能超时，可以用 `hf-mirror.com`，注意必须带 User-Agent，否则返回 403。

---

[返回 README](../README.md) | [Daimon 教程](tutorial-daimon.md) | [GelSight 教程](tutorial-gelsight.md) | [PaXini 教程](tutorial-paxini.md)
