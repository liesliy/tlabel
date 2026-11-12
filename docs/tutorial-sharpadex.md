# SharpaDex 六轴指尖力触觉标注教程

SharpaDex 是 Sharpa Robotics 公开的双手机器人操作数据集，触觉部分是 **10 个指尖各 6 轴力/力矩**（每只手 5 指 × 3 力 + 3 力矩，共 60 维，30 fps）。

这份教程带你从零把 SharpaDex 的一个 season 转成 TLabel JSON，并说明三个必须提前知道的边界：**`tactile_deform` 视频是空的**、**`texture_class` 判不出来**、**等级最高到 L3**。

---

## 前置准备

```bash
pip install "tlabel[sharpadex]"
```

`sharpadex` extra 会带上 `pyarrow`（读 parquet）和 `opencv-python`（读 `tactile_raw` 视频）。不装 opencv 也能跑，只是拿不到 `object_deformation`。

检查一下：

```bash
tlabel list
```

应该能在适配器列表里看到 `sharpadex`。

---

## 步骤 1: 准备数据

### 数据集结构

SharpaDex 按 season 分目录，每个 season 下同时存在多个 LeRobot 版本：

```
place_ball_in_box/
└── season_POC22018_2026_04_30_15_47_22_train/
    └── lerobot_v2.1/          ← 用这个
        ├── meta/
        │   ├── info.json
        │   ├── modality.json
        │   ├── subtasks.jsonl
        │   └── annotations.jsonl
        ├── data/
        │   └── chunk-000/
        │       └── episode_000000.parquet
        └── videos/
            └── chunk-000/
                ├── observation.images.tactile_raw/
                │   └── episode_000000.mp4
                └── observation.images.tactile_deform/
                    └── episode_000000.mp4
```

适配器会自动往下找 `lerobot_v2.1`，你也可以直接传 `lerobot_v2.1` 这一级。

**只下一个 season。** 全量 3.4 TB，一个 season 约 19.6 GB，但适配器实际只读 `meta/` 和 `data/`——74 个 parquet 加起来 30.7 MB。视频只在需要 `object_deformation` 时才读。

### v3.0 下不了

同一 season 的 `lerobot_v3.0/` 目录下所有文件都返回 503/502（三次验证，规律稳定），只能用 `v2.1`。适配器按 `info.json` 里的 `data_path` 模板解析路径，模板里出现 `{file_index}` 时会去 `meta/episodes/` 查真实映射，查不到就报错——不会猜一个数字读错 episode。

---

## 步骤 2: 加载数据

```python
from tlabel.adapters.sharpadex import SharpaDexAdapter

adapter = SharpaDexAdapter()
data = adapter.load(
    r"D:\sharpadex\place_ball_in_box\season_xxx\lerobot_v2.1",
    episode_index=0,
)

print(len(data.frames))        # 459
print(data.episode_info["task"])  # Task: Pick the Ball and Place it in the Box | ...
```

流式读取，内存不随帧数涨：

```python
for frame in adapter.iter_frames(root, episode_index=0):
    s = frame.schema_v2
    print(frame.frame_idx, s.contact, s.force_magnitude)
```

---

## 步骤 3: 查看标注结果

### force_magnitude 是十指模长之和

数据集里 60 维是**每根手指各自一个局部坐标系**（README 第 530 行：力/力矩表达在对应的 `*_elastomer` 局部系，力单位 N，力矩单位 N·m）。

所以不能把 10 根手指的力矢量直接相加——它们不在同一坐标系里，加起来没有物理意义。

TLabel 的做法：

```python
force_magnitude = sum(||f_finger||₂ for finger in 10)   # 标量，与坐标系无关
```

`force_vector` / `torque_vector` 只取**受力最大的那一根手指**的矢量，坐标系跟着它走，并在 `sensor_specific` 里写清楚：

```json
{
  "force_vector_finger": "right_thumb",
  "force_vector_frame": "fingertip_elastomer_local",
  "force_unit": "N",
  "torque_unit": "N·m"
}
```

看到 `force_vector_frame` 是 `fingertip_elastomer_local` 就说明：这个矢量不能直接跟世界坐标系的力做加减。

### 逐指力都在 sensor_specific 里

```python
print(data.frames[250].sensor_specific["per_finger"])
```

```python
{
  "left_thumb":  {"force_norm": 0.012, "torque_norm": 0.0004},
  "right_thumb": {"force_norm": 20.81, "torque_norm": 0.342},
  "right_index": {"force_norm": 0.42,  "torque_norm": 0.009},
  ...
}
```

### contact_centroid 在手指序位空间

触觉是 10 个离散点，没有连续表面，所以质心只能落在"手指序位"空间：

```python
_FINGER_X = {"thumb": 0.1, "index": 0.3, "middle": 0.5, "ring": 0.7, "little": 0.9}
_HAND_Y   = {"left": 0.25, "right": 0.75}
```

按各指力模长加权平均。`x` 越接近 0.1 说明力越集中在拇指，`y=0.75` 说明是右手。

这个坐标**不是**物理坐标，`sensor_specific["contact_centroid_space"]` 里标了 `finger_ordinal`。要换到物理坐标得配合手部关节状态和 URDF 做正运动学，数据集不提供这个换算。

### contact_region 恒为 digital

10 个点全在指尖，所以 `contact_region = "digital"`。这比 Grabette 的 `palmar` 更准确——那个是单指尖阵列，只能说是指腹。

### manipulation_phase 来自数据集的 segments

`meta/annotations.jsonl` 每行带完整的 `segments`：

```json
{"start_step": 0,   "end_step": 222, "task_skill": "pick",  "text": "Pick up the ball. | Skill: pick"}
{"start_step": 223, "end_step": 287, "task_skill": "move",  "text": "Move the ball above the box. | Skill: move"}
{"start_step": 288, "end_step": 362, "task_skill": "place", "text": "Place the ball in the box. | Skill: place"}
```

映射规则：

| 段 | 有接触 | 无接触 |
|---|---|---|
| `pick` | `grasp` | `pre_contact` |
| `move` | `lift` | `lift` |
| `place` | `place` | `place` |
| 段外 | `hold` | `pre_contact` |

注意 `subtask_index` 有一列值 `3` 是**占位符**，不是第四个子任务——`subtasks.jsonl` 第 4 行是全局任务描述（`Task: ... | Instruction: ... | Scene: ...`）。真正的动作只有 3 段，最后 96 帧（363-458）没有段覆盖，是松手后的收尾。

`sensor_specific["phase_source"]` 会标明每帧的来源：

- `annotation_segment`：落在数据集标注段内（全量 22,335 帧）
- `contact_heuristic`：段外，按接触状态推断（全量 5,026 帧）

### object_deformation 来自 tactile_raw 的逐像素差异

`tactile_raw` 是 10 个指尖相机的拼图：**2 行 × 5 列，row 0 = left，row 1 = right，col 0..4 = thumb / index / middle / ring / little**，每 tile 320×240。

做法是逐像素与基线（前 30 帧的均值）比：

```python
diff = |frame - baseline|
area = mean(diff > 15)     # 变化像素比例，0-1
```

实测 episode 0 的 `right_thumb`：

| 阶段 | area |
|---|---|
| 空手（0-200 帧） | 0.0000 |
| 抓稳（210-350 帧） | 0.56 |
| 释放后残余（380-458 帧） | 0.03-0.08 |

`object_deformation` 取该帧所有受力手指里最大的 `area`。

**为什么不用全局 std**：接触时 `right_thumb` 的 std 只从 32.69 升到 34.53（+5.6%），Laplacian 方差变化不到 10%，和 H.264 压缩噪声同量级。全局统计量被传感器表面固定的棋盘格纹路稀释了，逐像素差异才有 30 倍的变化。

---

## 步骤 4: 导出标注

### 命令行

```bash
tlabel export \
  --format json \
  --adapter sharpadex \
  --episode-index 0 \
  "D:\sharpadex\place_ball_in_box\season_xxx\lerobot_v2.1" \
  "out\ep0.json"

tlabel validate "out\ep0.json"
```

实测 episode 0：459 帧，1251.9 KB，0 错误 0 警告。

### Python API

```python
data = adapter.load(root, episode_index=0)
data.to_json("out/ep0.json")
```

---

## 实测数据

全量 74 个 episode：

| 项 | 值 |
|---|---|
| 帧数 | 27,361（与 `meta/info.json` 的 `total_frames` 一致） |
| 处理耗时 | `iter_frames()` 23.4 s / `load()` 单 episode 8.4 s（含视频解码） |
| 峰值内存 | 167.0 MB |
| `compliance_level` | **L1 × 13,112（无接触）+ L3 × 14,249（有接触）** |
| `contact` | 14,249 / 27,361 |
| `slip_event` | 773 |
| `contact_region` | `digital` × 14,249，无接触帧为 None |
| `manipulation_phase` | pre_contact 13,112 / lift 5,813 / hold 5,023 / grasp 1,895 / place 1,518 |
| `phase_source` | annotation_segment 22,335 / contact_heuristic 5,026 |
| 受力最大手指 | right_thumb 8,744 / right_index 4,862 / right_middle 643 |
| `schema.validate()` | 0 失败 |
| 单元测试 | 57 个 |

左手 10 个通道全程 ≤ 0.085 N，这个 season 是纯右手操作。

---

## 已知局限

这一节是重点。SharpaDex 有三个**数据本身**的限制，不是适配器没写好：

### 1. tactile_deform 视频是空的

74 个文件加起来 9.9 MB（每个 0.08-0.19 MB），459 帧全部 `mean=0, std=0`，相邻帧差 0.000。

它是占位文件。ROADMAP 里原本计划从这里取 `object_deformation` 和 `texture_class`，实际上一个都取不到，只能用 `tactile_raw`。

### 2. texture_class 判不出来

`texture_class` 的 6 个枚举（smooth / rough / granular / fibrous / sticky / slippery）需要表面纹理信息。

`tactile_raw` 拍到的是指尖弹性体的**形变**，不是被接触物体表面的纹理。从形变图能看出"压了多大面积"，看不出"压在什么材质上"。

所以这个字段恒为 `None`，`capabilities["texture_class"] = False`。

硬要填的话只能靠"任务描述说是球，球是光滑的"这种外部知识，那不是触觉标注，是把任务名抄进标签。

### 3. temperature 没有传感器

数据集不提供温度通道，恒为 `None`，`capabilities["temperature"] = False`。

### 结论：最高 L3

Schema 的等级定义是 L1 基础 / L2 +力标量 / L3 +力矢量 / L4 所有 Optional 维度。

SharpaDex 满足 L3（`force_magnitude` 和 `force_vector` 都有），但 `texture_class` 和 `temperature` 两个 Optional 维度填不上，**所以不声明 L4**。

全量 13 维里实际有值的是 11 维：

```
contact ✓  contact_centroid ✓  contact_region ✓  force_magnitude ✓
force_vector ✓  torque_vector ✓  slip_event ✓  slip_velocity ✓（仅滑移帧）
manipulation_phase ✓  object_deformation ✓（仅接触帧 + 有视频时）
confidence ✓
texture_class ✗  temperature ✗
```

这是数据集的上限，不是适配器的缺陷。ROADMAP 里"L4 / 13 维非 None"的验收口径按此修正。

---

## 常见问题

### Q: 为什么 force_magnitude 比单指力大好几倍？

因为它把 10 根手指的力模长**加起来**了。episode 0 抓稳阶段 `right_thumb` 单独 20.81 N，`right_index` 0.42 N，加起来 21.2 N 左右——这是"整只手施加的总法向力"，不是"最大接触力"。

要看单指力，读 `sensor_specific["per_finger"]`。

### Q: 为什么 force_vector 有时是拇指的、有时是食指的？

取的是**当帧受力最大**的那根手指。抓球时拇指一直压着（mean 6.1 N），食指只在接触瞬间冲高（峰值 11.1 N，均值 0.6 N），所以多数帧是拇指，少数帧是食指。

跨帧比较 `force_vector` 时一定要先看 `force_vector_finger`——不同手指的坐标系不同。

### Q: 为什么有些帧 compliance_level 是 L1？

无接触帧没有 `force_magnitude`，而 Schema 硬约束禁止此时声明 L2 及以上（`level_num >= 2 必须 force_magnitude 非 None`）。全量 13,112 帧属于这种情况，占 48%。

这和 Grabette、touchd、demo 的处理一致。

### Q: object_deformation 是毫米吗？

不是，是无量纲比例（0-1），表示该帧指尖表面发生灰度变化的像素占比。Schema 允许 `ratio`。

要换算成毫米需要弹性体厚度和相机标定参数，数据集没给。

### Q: 视频要全下吗？

不用。不装 opencv 或不下视频时，`object_deformation` 为 None，其他 10 维照常输出，等级仍是 L3。

要 `object_deformation` 就下 `observation.images.tactile_raw/`，74 个文件约 1.6 GB。`tactile_deform/` 不用下——它是空的。

### Q: 数据集地址？

<https://huggingface.co/datasets/Sharpa-Robotics/SharpaDex-v1.0>（CC-BY-4.0）
