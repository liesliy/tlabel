# TLabel 工业级数据集适配路线图 v1

> **这是一份交接单，写给未来的自己。**
> 如果你（未来的 Agent）在半年后接手这件事，请从第 0 节开始读，不要跳。
> 如果你觉得顺序不对，先读第 5 节，再决定要不要改。

| 项 | 值 |
|---|---|
| 制定日期 | 2026-11-12 |
| 代码基线 | tlabel **v0.24.2**，commit `50bb021` |
| 制定人 | 扣龙虾（Agent），受吴圣（圣姐）委托 |
| 决策依据 | 见第 6 节「数据源事实」，全部经 2026-11-12 联网核实 |
| 状态 | 未开工 |

---

## 0. 三十秒版

**目的**：让 TLabel 能处理工业级触觉数据集（千万帧、TB 级），第一个目标是 SharpaDex v1.0（3.4 TB / 3,239 万帧）。

**不是目的**：不是「支持更多数据集刷数量」。适配器数量从 12 涨到 14 不是成绩，能跑通 3,239 万帧才是。

**顺序（不可跳）**：

```
第 1 步  流式改造          1.5 天
第 2 步  Grabette 适配器    1 天
第 3 步  SharpaDex 阶段 1   5-7 天
第 4 步  决策点（看结果再定）
```

**为什么必须这个顺序**：

- 不做第 1 步，第 3 步**一定 OOM**。单个 season（302,020 帧）全量解码需要 **1.2 TB 内存**，128 GB 机器也救不了。
- 第 2 步是第 1 步的**最小验收**。Grabette 只有 19,353 帧、格式简单、无视频解码，一天能跑完，用来验证流式改造有没有改错。改错了立刻发现，不用等 SharpaDex 跑三天才发现。
- 第 3 步才是真正的目标。
- 第 4 步不要提前做，也不要跳过后直接上全量。

---

## 1. 现状基线（2026-11-12 实测）

### 1.1 适配器清单

`tlabel/adapters/__init__.py` 的 `AVAILABLE_ADAPTERS` 当前有 **12 个数据集适配器 + 2 个传感器适配器 + 1 个占位符**：

| 类型 | 名称 | 说明 |
|---|---|---|
| 数据集 | gelsight | GelSight Mini / DIGIT (.pkl) |
| 数据集 | paxini | PaXini PXCap (.h5) |
| 数据集 | daimon | Daimon DM-TacClaw (.parquet / LeRobot) |
| 数据集 | tlabel | TLabel Format JSON (.json) |
| 数据集 | touchd | ToucHD-Force / AnyTouch 2 (.npy / directory) |
| 数据集 | univtac | UniVTAC (.hdf5 / .h5) |
| 数据集 | vtouch | VTouch (.h5 / .hdf5) |
| 数据集 | ycb_slide | YCB-Slide CMU DIGIT (.npy / directory) |
| 数据集 | tacquad | TacQuad AnyTouch (directory) |
| 数据集 | syntouch | SynTouch BioTac (.h5 / .csv / .mat) |
| 数据集 | xela | XELA uSkin / UniTac-NV (.csv) |
| 数据集 | tashan_ts_f_a | RoboMIND Tashan TS-F-A (.hdf5 / .h5) |
| 传感器 | paxini_gen3 | PaXini GEN3 实时 (SDK) |
| 传感器 | daimon_dm_tac | Daimon DM-Tac 实时 (USB/UVC) |
| 占位 | paxini_px6d | PX6D 六轴力 (Modbus)，未实现 |

**注意**：如果你看到的是 9 个适配器，说明你读的是 `tlabel-dev/`（v0.18.2 旧副本），**不是主线**。主线是 `牛宿科技/tlabel-repo/`。

### 1.2 已确认的技术缺陷

| 缺陷 | 位置 | 后果 |
|---|---|---|
| 伪流式：`_extract_ffv1_frames_streaming` 返回 `List[np.ndarray]` | `tlabel/adapters/daimon_dataset.py:418` | 名字叫 streaming，实际全量进内存 |
| `load()` 全量累积帧 | `daimon_dataset.py:1109`（`tlabel_frames = []`）、`:1235`（append）、`:1280`（组装） | 大 episode 直接 OOM |
| 视频路径按 v2.1 glob 匹配 | `daimon_dataset.py:128` `_find_tactile_videos` | LeRobot v3.0 上**静默返回空**，不报错 |
| `base.py` 无流式接口 | `tlabel/adapters/base.py:69` `load()` | 只能一次返回完整 `TLabelData` |

### 1.3 已有的可复用资产

| 资产 | 位置 | 用途 |
|---|---|---|
| 触觉特征计算 | `daimon_dataset.py:491` `_compute_tactile_features_from_video` | 输入 (H,W,3) float32，输出 deformation_magnitude / contact_area / texture_energy / edge_density / shear_field / normal_field |
| 光流滑移检测 | `daimon_dataset.py:579` `_compute_optical_flow_from_tactile` | Farneback 光流 |
| taxel 网格 → 归一化坐标 | `xela.py:145` `_taxel_positions` | 任意 (rows, cols) 或扁平 taxel 列表 → (N,2) ∈ [0,1]² |
| taxel 矩阵解析与形状校验 | `xela.py:71` `_parse_taxel_matrix` | 支持 (行,列,3) / (taxel数,3) / (3,) |
| 阶段推断 | `daimon_dataset.py:832` `_infer_phases` | 从 contact/slip 序列推断 manipulation_phase |
| 溯源元数据 | `base.py:118` `build_provenance` | 0.24.x 新增，用于记录数据来源 |

---

## 2. 为什么做这件事（别理解错）

### 2.1 真实动机

圣姐在 2026-11-12 的原始需求（原话）：

> 「搜搜公开数据集有没有新的适合用 tlabel 处理的，然后分析下……既然是工业级数据集，用 tlabel 处理时，需要什么硬件资源」

背景链条：

1. TLabel 已有一张浙江省数据知识产权证（「机器人滑动触觉位姿证明」），登记的是 YCB-Slide 相关的小规模数据集。
2. 2026-11-12 讨论的结论是：**证书要多在种类不在数量**，下一步该补的是数据产权证，不是再登记一个数据集。
3. 但要让 RDA / TLabel 在具身数据这条线上站得住，需要**能扛工业级数据量的实证**，不是「我支持 12 个数据集」的自述。

所以这件事的产出物是**三个可验证的东西**：

- 一份能跑通 3,239 万帧的处理记录（不是承诺，是日志）
- 一个能写进论文/BD 材料的硬指标（「单机 16 核 22.5 小时处理 300 小时真机触觉数据」）
- 一条证明 TLabel 能到 **L4 Rich-Semantic** 的样例（SharpaDex 是第一个 14 维能填满 13 维的数据集）

### 2.2 容易被误解成什么（读到这里请自我检查）

| 误解 | 为什么错 |
|---|---|
| 「多支持几个数据集，显得生态好」 | 适配器数量不是目标。12 个里有 6 个是单人数据集，再多两个也不改变什么。 |
| 「SharpaDex 数据量大，所以要优化性能」 | 不是「优化」，是**架构不支持**。现在是全量加载，不是慢，是跑不了。 |
| 「先把 SharpaDex 适配器写完，流式以后再说」 | 写完也跑不了，因为 1.2 TB 内存。而且会在调试时被 OOM 反复打断，效率更低。 |
| 「顺手把 TacVerse / OpenNeoData 也做了」 | 不做。一次只推进一个，见第 4 节红线。 |
| 「顺便重构一下 daimon_dataset.py，它 1314 行太乱了」 | 不做。这次只改流式路径，不动其余结构。重构会把风险从「可能 OOM」扩大到「可能全错」。 |

---

## 3. 执行顺序与验收标准

### 第 1 步：流式改造（1.5 天）

**为什么先做这个**

见第 0 节。补充一个量化依据：单个 season 302,020 帧 × 4.03 MB/帧（480×1200×3 + 480×1600×3）= **1.2 TB**。全量 32,396,172 帧 = **130 TB**。这不是内存调优能解决的问题。

**做什么**

1. 新增 `DataAdapterBase.iter_frames(file_path, **kwargs) -> Iterator[TLabelFrame]`，作为流式入口。`load()` 保持原签名不变（向后兼容 12 个适配器），内部改为消费 `iter_frames()`。
2. 把 `_extract_ffv1_frames_streaming` 和 `_extract_video_frames` 的返回从 `List` 改成 generator（用 `yield`），逐帧产出。
3. `load()` 里的 `tlabel_frames` 累积改为可选：`stream=True` 时不累积，直接聚合统计量（帧数、contact 占比、平均力值等），只保留 14 维标量序列。
4. `TLabelData` 增加 `streamed: bool` 标记和 `aggregate_stats: dict` 字段，让下游知道这批数据是不是完整帧序列。

**做完的标志（缺一不可）**

- [x] `pytest tests/` 全绿，现有 508 个测试一个不挂 — **505 passed, 4 skipped**（Windows 需 `PYTHONUTF8=1`；`test_json_export` 是既有的 GBK 编码问题）
- [x] 用 daimon 现有数据集跑 `load()`，输出与改造前**逐字段一致**（写一个 diff 脚本对比）— 663 帧全一致，唯一差异是新增的 `streamed` 字段
- [x] 用 `iter_frames()` 处理一个 30,000 帧的 episode，`tracemalloc` 峰值内存 **< 1 GB** — **44.9 MB**
- [x] 改造前同一 episode 峰值内存记录在案（作为对比基线）— **2,625 MB**
- [x] `CHANGELOG.md` 写明这是 breaking change 还是 minor（预期 minor，因为 `load()` 签名不变）— **minor**

**实测（2026-11-12）**

30,000 帧合成 episode（96×72 gbrp16le 触觉视频，569.6 MB × 2 路）：

| 路径 | 峰值内存 | 耗时 |
| --- | --- | --- |
| v0.24.2 `load()` | 2,625 MB | 142.5 s |
| v0.25.0 `load()` | 124 MB | 126.6 s |
| v0.25.0 `load(stream=True)` | 85.5 MB | 125.3 s |
| v0.25.0 `iter_frames()` | 44.9 MB | 117.9 s |

比预期更好的一点：连 `load()` 本身都从 2.6 GB 降到 124 MB。视频帧不再累积，只剩 30,000 个 `TLabelFrame` 对象（约 4 KB/帧）。v0.24.2 那 2.6 GB 全是视频数组（30,000 × 96×72×3×4 B ≈ 2.5 GB），随分辨率线性增长，真实 288×384 数据下会到约 40 GB。

复现脚本：`scripts/verify_stream_equiv.py`（逐字段对比）、`scripts/make_memtest_dataset.py`（合成数据集）、`scripts/bench_stream_memory.py`（内存基准）。

**坑**

- `cv2.VideoCapture` 在多进程下不安全，每个 worker 要独立开 capture
- ffmpeg 子进程要显式 `terminate()` + `wait()`，否则僵尸进程堆积
- generator 被提前 break 时要能正确释放资源（用 `try/finally`）

---

### 第 2 步：Grabette 适配器（1 天）

**为什么做这个**

它是第 1 步的最小验收。数据小（19,353 帧）、格式简单（无视频解码，纯 parquet 里的 int16 数组）、一天能跑完。如果流式改造有 bug，这里会立刻暴露。

**数据集事实**（2026-11-12 核实）

| 项 | 值 |
|---|---|
| HuggingFace 仓库 | `CarolinePascal/grabette-tactile-full-synced` |
| 规模 | 48 episodes / 19,353 帧 / 50 fps |
| 触觉 | parquet 里两个 int16 网格：`observation.tactile.right_sensor_1` **6×6**、`observation.tactile.right_sensor_2` **4×8** |
| 其他模态 | 2 路 RGB `right_cam0`/`right_cam1` 960×720、8 维 action、`is_lost` 标志 |
| 许可证 | Apache-2.0 |
| 体积 | **实测 239.7 MB**（9 个文件）；其中视频 238.0 MB，适配器只需要 `meta/` + `data/` 共 **1.7 MB** |

**做什么**

1. 新建 `tlabel/adapters/grabette.py`，继承 `DataAdapterBase`，`name = "grabette"`。
2. 复用 `xela.py` 的 `_taxel_positions` 和 `_parse_taxel_matrix`（注意 Grabette 是 1 轴压力，xela 是 3 轴，解析逻辑要简化）。
3. 双传感器不对称处理：6×6 和 4×8 两个阵列尺寸不同，`contact_centroid` 要分别归一化到 [0,1]²，再按受力大小加权合并。
4. 在 `__init__.py` 的 `AVAILABLE_ADAPTERS` 注册，`pyproject.toml` 加 `grabette = ["pyarrow>=10.0"]` 可选依赖。
5. 写 `docs/tutorial-grabette.md`，风格照 `tutorial-daimon.md`。

**能填到哪一级：L2**

| 14 维字段 | 能否填 | 依据 |
|---|---|---|
| contact | ✓ | 网格压力 > 阈值 |
| contact_centroid | ✓ | 压力加权重心，双阵列分别归一化后合并 |
| contact_region | ✓ | 恒为 `palmar`（单指尖指腹）；阵列内部九宫格位置另存 `sensor_specific["grid_region"]` |
| force_magnitude | ✓ | 网格压力积分（无量纲，因无标定系数） |
| force_vector | ✗ | 只有法向单轴，无切向分量 |
| torque_vector | ✗ | 无多轴力、无 taxel 间距 |
| slip_event | ✓ | 帧间网格位移 + 压力突变 |
| slip_velocity | ✓ | 帧间质心位移 / dt |
| manipulation_phase | ✓ | 压力曲线分段 |
| texture_class | ✗ | 无纹理图像 |
| object_deformation | ✓ | 网格总压力变化率 |
| temperature | ✗ | 无温度传感器 |
| confidence | ✓ | 双传感器一致性 |
| compliance_level | **L2** | 按定义：L1 + force_magnitude |

**做完的标志**

- [x] `tlabel convert --adapter grabette <path>` 能跑通 48 个 episode — **19,353/19,353 帧，0 失败**
- [x] 输出 JSON 通过 `tlabel validate` 自校验 — **0 错误 0 警告**
- [x] `compliance_level` 判定为 L2，不是 L1 也不是 L3 — **有接触 7,652 帧全 L2；无接触 11,701 帧降 L1（L2 要求 `force_magnitude` 非 None）**。19,353 帧过 `TLabelSchemaV2.validate()` 0 失败
- [x] `force_magnitude` 明确标注为「无量纲原始读数」，**不能填 0 或虚构牛顿值**（照 `xela.py` 第 24-27 行的做法）
- [x] 处理全程峰值内存 < 500 MB（验证第 1 步在无视频场景下也生效） — **`iter_frames()` 33.9 MB / `load()` 80.7 MB**

> CLI 实际没有 `convert` 子命令，用 `tlabel export <path> <out> -f json -a grabette --episode-index N`。

**实测（2026-11-12）**

| 项 | 值 |
|---|---|
| 全量 48 个 episode | 19,353 帧，0 失败，`iter_frames()` 42.6 秒 / `load()` 49.2 秒 |
| 峰值内存（tracemalloc） | `iter_frames()` 33.9 MB / `load()` 80.7 MB |
| `compliance_level` | L2 × 7,652（有接触）+ L1 × 11,701（无接触） |
| `manipulation_phase` | idle 11,492 / stable_contact 7,126 / slip 470 / re_contact 156 / release 55 / initial_contact 54 |
| `contact_region` | 恒为 `palmar` × 7,652（全量 0 个非法枚举值） |
| `sensor_specific["grid_region"]` | 九宫格 9 种全部出现（middle_center 3,455 / middle_left 1,634 / bottom_center 871 / bottom_left 837 / top_center 411 / middle_right 352 / top_left 39 / bottom_right 28 / top_right 25） |
| Schema 自校验 | 19,353/19,353 帧 `validate()` 通过，0 失败 |
| 单 episode 导出 | 330 帧 → 814.8 KB JSON，`tlabel validate` 通过（0 错误 0 警告） |
| 测试 | 565 passed / 4 skipped（新增 60 个） |

复现：

```bash
pip install tlabel[grabette]

python -c "
from tlabel.adapters.grabette import GrabetteAdapter
ad = GrabetteAdapter()
for ep in range(48):
    d = ad.load('path/to/grabette-tactile-full-synced/', episode_index=ep)
    print(ep, len(d.frames))
"
```

**坑（实测补充）**

- `meta/tasks.parquet` 的任务文本是 **pandas index** 而不是普通列（parquet 的 pandas metadata 里 `index_columns: ["task"]`）。只查 `df.columns` 会退化成 `{task_index: task_index}`，任务描述变成数字 `0`。必须查 `df.index.name`。
- `episode_index` 列**单调不减**，指定第 N 个 episode 时可以提前终止扫描。不做这个优化时 48 个 episode 要重扫全表 48 遍（78 秒），做了之后 49 秒。
- 基线**恒为 0**（未接触全零），不是中位数基线，不需要减基线。
- 两个阵列归一化后的质心分布不同（episode 0 里 sensor_1 的 y 在 0~1 跳、sensor_2 的 y 恒在 0.67~0.82），合并后九宫格位置在单个 episode 内可能只落在一两格。跑全量后九宫格都会出现，不要看到单个 episode 只有 left 就以为有 bug。
- **`contact_region` 不能用九宫格名字**。`VALID_CONTACT_REGIONS` 只有 `palmar / digital / lateral / proximal / distal / dorsal / other`（手部解剖分区）。Grabette 是单指尖传感器，数据集没文档化阵列哪一侧朝指尖，硬映射成 `distal` / `proximal` 就是编造。`contact_region` 恒取 `palmar`，九宫格进 `sensor_specific["grid_region"]`。
- **`tlabel validate` 不调用 `TLabelSchemaV2.validate()`**，所以非法枚举值、`force_magnitude` 缺失这类语义违规它查不出来。适配器自测里必须直接调 `schema.validate()`（返回 `(ok, errors)`）。
- **无接触帧不能挂 L2**。L2 定义要求 `force_magnitude` 非 `None`，无接触帧没有力可报，必须降 L1（`touchd` / `demo` 都是这么处理的）。全量 19,353 帧里有 11,701 帧属于这种情况，占 60%。

---

### 第 3 步：SharpaDex 阶段 1（5-7 天）

**数据集事实**（2026-11-12 核实）

| 项 | 值 |
|---|---|
| HuggingFace 仓库 | `Sharpa-Robotics/SharpaDex-v1.0` |
| 规模 | **28,993 episodes / 59 tasks / 495 seasons / 32,396,172 帧 / 300 小时** |
| 总体积 | **3.4 TB** |
| 格式 | LeRobot **v3.0**（另有 v2.1 导出） |
| 触觉视频 | `tactile_deform` **480×1200**、`tactile_raw` **480×1600**（MP4 流） |
| 触觉向量 | `observation.tactile` **60 维** = 10 指 × 6 轴（fx/fy/fz/tx/ty/tz），指尖为 thumb/index/middle/ring/little |
| 本体 | `observation.state` 65 维、`observation.state.tcp` 24 维（每臂位姿 + 力/矩）、`observation.state.joint_torque`（左右臂/手/躯干分列） |
| 视觉 | 4 路 RGB 480×480：`head_left` / `head_right` / `wrist_left` / `wrist_right` |
| 语言 | `subtask_index` 逐帧子任务索引，三层标注（59 任务 / 224,087 子任务段 / 45 原子动作） |
| 硬件来源 | Sharpa D01：双 7-DoF 臂 + Wave 灵巧手 + 全身电子皮肤（100 Hz 采样 / 0.2 N 力分辨率） |
| 许可证 | **CC-BY-4.0** |
| 单 season 参考 | 410 episodes / 302,020 帧 / **19.6 GB** |

**为什么值这个工作量：L4 Rich-Semantic**

60 维触觉向量直接给出每指三轴力 + 三轴力矩，**不需要从图像反推**：

| 14 维字段 | SharpaDex 来源 | 等级 |
|---|---|---|
| contact | 60 维力值非零判定 | ✓ |
| contact_centroid | 10 指受力加权位置 | ✓ |
| contact_region | 指节分区 | ✓ |
| force_magnitude | √(fx²+fy²+fz²) 逐指 | ✓ |
| **force_vector** | **[fx,fy,fz] 直接读** | ✓ |
| **torque_vector** | **[tx,ty,tz] 直接读** | ✓ |
| slip_event | 触觉视频光流 + 力突变 | ✓ |
| slip_velocity | 帧间光流幅值 | ✓ |
| manipulation_phase | `subtask_index` + 力曲线分段 | ✓ |
| texture_class | `tactile_deform` Laplacian 方差 | ✓ |
| object_deformation | deform 帧 RMS | ✓ |
| temperature | **无温度传感器** | ✗ |
| confidence | 多源一致性 | ✓ |
| compliance_level | **L4** | ✓ |

**13/14 填满，缺的只有温度。** 现有适配器里：GelSight → L2、YCB-Slide → L2、Daimon → L3（有 shear 无 torque）。**SharpaDex 是第一个能到 L4 的**。

这是写进论文和 BD 材料的核心卖点：「TLabel 在 SharpaDex 上达到 L4 完整语义覆盖，是目前唯一做到的触觉标注框架」。

**做什么（拆解）**

| 模块 | 工时 | 要点 |
|---|---|---|
| v3.0 路径解析 | 1 天 | **最大的坑，见下** |
| episode 偏移切片 | 0.5 天 | 用 `dataset_from_index` / `dataset_to_index` |
| 真流式解码接入 | 0.5 天 | 复用第 1 步的 `iter_frames` |
| 60 维触觉映射 | 1 天 | 10 指 × 6 轴 → force_vector / torque_vector |
| 65 维 state + 24 维 tcp 解析 | 1 天 | 对着 `meta/info.json` 逐字段核对，不要猜 |
| 视频特征复用 | 0.5 天 | 直接用 `_compute_tactile_features_from_video` |
| 测试 + 文档 | 1.5 天 | 单测 + `docs/tutorial-sharpadex.md` |

**最大的坑：LeRobot v3.0 的路径模板是反的**

```
v2.1:  videos/chunk-000/<camera_key>/file-000.mp4
v3.0:  videos/<video_key>/chunk-000/file-000.mp4
```

`video_key` 跑到了 `chunk` 前面。**所有按 v2.1 写的 glob 在 v3.0 上会静默返回空列表**——不报错，只是找不到视频，然后降级到「无触觉」，最后产出一批看起来正常但触觉全空的数据。这是最危险的失败模式。

`daimon_dataset.py:128` 的 `_find_tactile_videos` 就是 v2.1 风格。

**正确做法**：读 `meta/info.json` 里的 `data_path` 和 `video_path` 模板字符串，按模板解析，不要硬编码路径。v3.0 的设计意图就是「路径是数据，不是约定」。

```
"data_path":  "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
```

**第二个坑：`meta/episodes/` 是分片 parquet**

v2.1 的 `episodes.jsonl` 是单文件，v3.0 改成 `meta/episodes/chunk-XXX/file-XXX.parquet`，**可能有多个文件**（DROID 有 7 个共 592 MB）。只读第一个会静默拿到 1/N 的 episode，已经有第三方实现踩过这个坑，产出了只有 14% episodes 的数据集。

必须遍历 `meta/episodes/` 下所有 parquet 文件再合并。

**做完的标志**

- [ ] 只下载 **1 个 season**（19.6 GB），不下载全量
- [ ] 该 season 的 410 个 episode 全部处理成功，无一失败
- [ ] `compliance_level` 判定为 **L4**，14 维里 13 个非 None
- [ ] `force_vector` / `torque_vector` 的数值与 parquet 原始 `observation.tactile` 逐帧比对一致（误差 < 1e-6）
- [ ] 峰值内存 < 2 GB（16 核并行 8 进程时 < 16 GB）
- [ ] 输出 JSON 通过 `tlabel validate`
- [ ] 处理耗时记录在案，作为第 4 步决策依据

**坑（除上面两个）**

- 60 维的排列顺序必须对着 `meta/info.json` 的 feature 定义逐项核对，**不能按「拇指到小指、力到力矩」的顺序猜**
- `tactile_raw`（480×1600）和 `tactile_deform`（480×1200）语义不同：raw 是原始传感器读数，deform 是去基线后的形变场。`texture_class` 用 deform，`contact` 用 raw
- SharpaDex 是 CC-BY-4.0，做衍生数据要**保留署名**。如果要拿去登记数据知识产权，材料里必须写明「原始数据 CC-BY-4.0，本成果为算法加工的衍生数据集合，原始权利人不主张衍生部分权利」

---

### 第 4 步：决策点

**不要提前做这一步。** 做完第 3 步、拿到实测数据后，再回答这三个问题：

1. 单个 season 的处理耗时和内存占用是多少？
2. L4 的判定结果在人工抽查下准不准（抽 20 个 episode 人工看）？
3. 有没有真实的商业需求（客户要看、要买、要入表、要报项目）？

**三种走向**：

| 情况 | 下一步 |
|---|---|
| 有客户明确要看全量 | 上阶段 2（抽 12-15 个任务，200-300 GB），或直接云上跑全量 |
| 只有技术验证需求 | 停在第 3 步。1 个 season 的 L4 样例已经够写论文和 BD |
| 发现 L4 判定不准 | 回退修特征算法，不要急着扩数据量 |

**明确不建议现在做的**（2026-11-12 决策）：

- **TacVerse**（106,800 张 / 7 种传感器）：gated 需申请，等 SharpaDex 做完再说。它的价值是「7 种传感器一次归一化」，是很好的论文素材，但不紧急。
- **OpenNeoData**（N0-Foundation，5,000 小时开源子集）：TB 级，处理成本高，优先级最低。
- **TacClaw-Infinity / Daimon-Infinity**：CC BY-NC-SA，**禁止商用**，不能用于登记。
- **Intern 三套**（Pour 等）：**无许可证声明**，不能假设有复用权。
- **ROMI-2K**：许可证未知。
- **EgoTouch**：论文写「will publicly release」，截至 2026-11-12 未放出。

---

## 4. 红线（不要做什么）

这一节是给未来的我踩刹车的。如果你正在做下面任何一件事，**停下来**。

1. **不要一次改多个适配器。** 12 个适配器里，只有 daimon 需要改流式路径，其他不要碰。
2. **不要顺手重构 `daimon_dataset.py`。** 1314 行确实乱，但这次只改流式，其余结构保持原样。重构会把风险从「可能 OOM」扩大到「可能全错」。
3. **不要下载 SharpaDex 全量 3.4 TB。** 除非第 4 步的决策明确要求。
4. **不要跳过第 2 步直接做第 3 步。** Grabette 是流式改造的最小验收，跳过等于裸奔。
5. **不要为了让数字好看而虚构字段值。** 填不了就填 `None`，照 `xela.py` 第 24-27 行的做法：`force_magnitude` 无量纲就标无量纲，不要填 0 或虚构牛顿值。
6. **不要把 `compliance_level` 往上抬。** 按定义判：L1 只有 4 个基础字段，L2 加 force_magnitude，L3 加 force_vector，L4 全填。填不满就老实降级。
7. **不要动 `.git/config` 的 `http.postBuffer`**（524288000 会导致 push 超时，改 1MB）。
8. **不要把凭证、token、许可证文件提交进仓库。**

---

## 5. 如果你觉得顺序不对

先问自己三个问题：

1. **你是不是读到了旧代码？** 检查 `VERSION` 是不是 0.24.2 或更高。如果是 0.18.x，你读的是 `tlabel-dev/`，那是废弃副本。主线在 `牛宿科技/tlabel-repo/`。
2. **你是不是把「性能优化」和「架构不支持」搞混了？** 现在是全量加载，单 season 要 1.2 TB 内存。这不是调优能解决的。
3. **你是不是想一次做完所有数据集？** 回到第 2.2 节的表格，逐行对一遍。

如果这三个都排除了，再回来改这份文档。**改的时候把改动原因写在下面**，别默默改。

### 变更记录

| 日期 | 改动 | 原因 | 改动人 |
|---|---|---|---|
| 2026-11-12 | 创建 | — | 扣龙虾 |

---

## 6. 数据源事实（2026-11-12 联网核实）

这一节是所有判断的依据。如果你要推翻某个结论，先来这里核对原始事实。

### 6.1 SharpaDex v1.0

- 仓库：`Sharpa-Robotics/SharpaDex-v1.0`（HuggingFace）
- 开源时间：2026-10-22
- 规模：28,993 episodes / 59 tasks / 495 seasons / 32,396,172 帧 / 约 300 小时 / 3.4 TB
- 格式：LeRobot v3.0 和 v2.1 双导出
- 硬件：Sharpa D01（双 7-DoF 臂 + Wave 灵巧手 + 全身电子皮肤 100 Hz / 0.2 N）
- 指尖传感器：W02 为 Dynamic Tactile Array，力范围 5 mN – 30 N，空间分辨率 1 mm；电子皮肤 0.1 – 20 N / 5 mm
- 许可证：CC-BY-4.0

### 6.2 Grabette Tactile

- 仓库：`CarolinePascal/grabette-tactile-full-synced`（HuggingFace）
- 规模：48 episodes / 19,353 帧 / 50 fps
- 触觉：6×6 与 4×8 两个 int16 taxel 网格（纯法向压力）
- 其他：2 路 RGB 960×720、8 维 action、`is_lost` 标志
- 许可证：Apache-2.0

### 6.3 LeRobot v3.0 格式变更

- 路径模板从硬编码改为 `meta/info.json` 里的 `data_path` / `video_path` 字符串
- `video_path` 中 `{video_key}` 在 `{chunk_index}` **之前**（v2.1 相反）
- episode 边界靠 `meta/episodes/` 里的 `dataset_from_index` / `dataset_to_index` 偏移解析，不靠文件名
- `meta/episodes/` 是分片 parquet，可能多个文件
- 单 parquet 文件上限 100 MB，单视频上限 200 MB，超限自动切分

### 6.4 硬件资源测算依据

| 项 | 数值 | 来源 |
|---|---|---|
| 单帧解码后大小 | 480×1200×3 + 480×1600×3 = **4.03 MB** | 计算 |
| 单 season 内存（全量） | 302,020 × 4.03 MB = **1.2 TB** | 计算 |
| 全量内存（全量） | 32,396,172 × 4.03 MB = **130 TB** | 计算 |
| 单帧处理耗时 | cv2 解码 7 ms + 特征计算 8-12 ms + 光流 20-30 ms ≈ **40 ms**（单核） | 经验估算，**需实测校准** |
| 全量算力 | 32,396,172 × 0.04 s = **360 核·小时** | 计算 |
| 8 核耗时 | 45 小时 | 计算 |
| 16 核耗时 | 22.5 小时 | 计算 |
| 32 核耗时 | 11 小时 | 计算 |
| 降采样 3 倍（30→10 fps） | 120 核·小时，8 核 15 小时 | 计算 |
| 输出体积 | 3,239 万帧 × 14 字段 × 8 字节 = 3.6 GB 未压缩，parquet+zstd 约 **0.8-1.5 GB** | 计算 |

**「40 ms/帧」是估算，不是实测。** 第 3 步做完后要用真实数据校准，并更新此表。

### 6.5 分阶段硬件配置

| 阶段 | 数据量 | CPU | 内存 | 磁盘 | 耗时 |
|---|---|---|---|---|---|
| Grabette | <5 GB | 4 核 | 4 GB | 10 GB | 4 分钟（4 核） |
| SharpaDex 阶段 1 | 19.6 GB（1 season） | 8 核 | 16 GB | 100 GB SSD | 下载 20-40 分钟 + 处理 4 小时 |
| SharpaDex 阶段 2 | 200-300 GB（12-15 任务） | 16 核 | 32 GB | 1 TB NVMe | 下载 1-1.5 小时 + 处理 15 小时 |
| SharpaDex 阶段 3 | 3.4 TB（全量） | 32 核 | 64 GB | 4 TB NVMe | 下载 9-19 小时 + 处理 11-15 小时 |

阶段 3 云成本粗估：32 核 64 GB 按 1.5 元/核·时 × 32 × 15 ≈ **720 元**（不含存储和流量），跑完即释放。

---

## 7. 本地与云端的通道

### 7.1 两条通道

```
【云盘通道】文件级，实时双向
  Windows: C:\Users\sisiw\Coze\Drive\扣龙虾\所有对话\主对话\
  云端:     /Coze/Drive/扣龙虾/所有对话/主对话/
  → 同一个挂载点。Agent 写的文件，Windows 上立刻可见，反之亦然。
  → 但 E:\ 不在其中。

【git 通道】代码级，手动同步
  云端: /Coze/Drive/.../牛宿科技/tlabel-repo/   (0.24.2 @ 50bb021)
  本地: E:\tlabel\                              (待确认)
  中转: github.com/liesliy/tlabel.git
  → 代码必须走这条路，不能走云盘（云盘是整文件覆盖，双端同时改会互相冲掉）
```

### 7.2 这份文档怎么到本地

本文件位于 `牛宿科技/tlabel-repo/docs/ROADMAP_industrial_datasets.md`，**在仓库内**，所以：

- 云端：直接读 `/Coze/Drive/.../牛宿科技/tlabel-repo/docs/ROADMAP_industrial_datasets.md`
- 本地：`cd E:\tlabel && git pull` 后读 `E:\tlabel\docs\ROADMAP_industrial_datasets.md`

**一份文件，两条路都能到。** 不要在两处各放一份，会不一致。

### 7.3 Agent 怎么知道本地状态

Agent 看不到 `E:\`。解决方案是**让本地把状态写到云盘**：

运行 `scripts/check_env.py`，它会在本地收集环境信息，并写入云盘路径
`牛宿科技/tlabel-repo/docs/_env_snapshot.md`。云盘同步后，Agent 立即可读。

脚本检查项：

1. `git remote get-url origin` 是不是 `github.com/liesliy/tlabel`
2. 当前 commit hash 和 branch
3. 与 origin 的领先/落后数
4. 未提交改动清单
5. `VERSION` 文件内容
6. 实际注册的适配器数量（import tlabel 后读 `AVAILABLE_ADAPTERS`）
7. 关键文件 SHA256：`tlabel/adapters/base.py`、`daimon_dataset.py`、`xela.py`、`__init__.py`
8. Python 版本、关键依赖版本（numpy / pyarrow / opencv / scipy）

### 7.4 约定

- **文档只在云端改**，本地只 `git pull` 不 `git push` 这个文件，避免冲突
- **代码只在本地改**，改完 `git push`，云端 `git pull` 后 Agent 才能看到
- 每次交接，先看 `_env_snapshot.md` 的生成时间，超过 3 天就重跑一次

---

## 8. 一页速查

```
目标      让 TLabel 跑通 SharpaDex 3,239 万帧
不是      刷适配器数量

顺序      1 流式 → 2 Grabette → 3 SharpaDex 阶段1 → 4 决策
不可跳    跳过 1 必 OOM（单 season 1.2 TB）
          跳过 2 是裸奔（没有最小验收）

基线      v0.24.2 @ 50bb021，12 数据集适配器
旧副本    tlabel-dev/ 是 0.18.2，别读它

最大坑    LeRobot v3.0 的 video_path 里 {video_key} 在 chunk 前面
          v2.1 风格的 glob 会静默返回空，不报错
          必须读 meta/info.json 的模板字符串

第二坑    meta/episodes/ 是分片 parquet，只读第一个会丢 6/7 的数据

数据量    只下 1 个 season（19.6 GB），不下全量 3.4 TB

红线      不重构、不多适配器并行、不虚构字段值、不抬 compliance_level
```
