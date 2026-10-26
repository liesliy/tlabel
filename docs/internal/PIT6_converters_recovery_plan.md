# PIT-6：v0.19.0 丢失能力（converters/base.py / _safe_float / 4 个 CLI 命令）落地方案评估

- 评估日期：2026-10-27
- 评估基线：`main` @ `a41fb0b`（v0.23.2）vs 分支 `feat/v0.19.0-convert-cli`（tip `e3867ab`，version 0.19.0）
- 结论性质：**只做调研与方案，不写业务代码、不提交 git**
- 背景溯源：CHANGELOG.md:106-119（0.19.0 条目已于 2026-10-26 勘误，明确记录 base.py 与 `_safe_float()` 未带入 0.20.0，"Tracked for restoration"）

---

## 0. TL;DR（先给结论）

| 丢失组件 | 建议 | 一句话理由 |
|---|---|---|
| `_safe_float()`（lerobot.py 写回路径） | **恢复，但按 main 语义重写**（向量取模 + 枚举映射） | 经验证当前 main 存在**真实静默数据丢失**：向量字段和枚举字段全部被写成 0.0（见 §2 实测）。半天工作量，零 breaking |
| `converters/base.py`（BaseConverter + 注册表） | **放弃原样恢复**；如需统一入口，在 `export/registry.py` 上薄封装 | main 已有严格超集架构 `ExporterBase + ExporterRegistry`（含 UI 字段描述、结果对象、8 个格式），恢复旧基类会造成两套并行注册体系 |
| `convert` 命令 | **以新形式恢复：扩展现有 `export` 子命令走 registry**，不新增 `convert` | 端到端能力（adapter.load → 目标格式）今天在 Python API 已通（§3 实测 ftp1 导出成功），缺的只是 CLI 接线；语义与 `export` 重叠 |
| `batch-convert` 命令 | **恢复（唯一真正缺失的能力）** | main 没有任何批量入口；`batch_to_ftp1()` 只吃 JSON 路径列表，不能从任意适配器批量转换。建议挂在 `export --batch` 或 `export-batch` |
| `list-adapters` 命令 | **合并进现有 `list`**（加 `--verbose/--json`），不新增命令 | 与 `list` 同数据源，仅展示列更多 |
| `adapter-info` 命令 | **合并进现有 `info`**（补 capabilities/compliance/扩展名详情），不新增命令 | 现有 `info` 是同一函数的精简版（cli.py:467-515） |
| `tests/test_cli_convert.py`（28 测试） | **不原样移植**，拆解重写 | 约 13 个测试与 base.py 强绑定会直接失效；端到端测试引用的 demo 数据是旧 22 维格式，断言也需按新输出改。约 60-70% 内容可回收 |

**总体判断：不是"回归事故"，而是"架构换代后留下两个真缺口 + 一堆已被取代的旧壳"。** 真缺口只有两个：① lerobot 写回路径的向量/枚举编码；② 任意适配器 → FTP-1 的批量 CLI。其余均已被 `export/registry.py`（2026-09 后建立）和 `lerobot_export.py`（v2.1 新建数据集路径）取代。

---

## 1. 现状对比表

### 1.1 仓库结构对比

| 维度 | 分支 0.19.0（`e3867ab`） | main v0.23.2（`a41fb0b`） |
|---|---|---|
| converters 目录 | `ftp1.py` / `lerobot.py` / **`base.py`(303行)** / `__init__.py` | `ftp1.py`(598行) / `lerobot.py`(326行) / **`lerobot_export.py`(618行，新增)** / `__init__.py`；**无 base.py**（`ls tlabel/converters/` 实测） |
| 统一导出抽象 | `converters/base.py`：`BaseConverter`（4 个静态方法）+ `CONVERTERS` dict，2 个格式 | `export/registry.py`(822行)：`ExporterBase` + `ExporterSpec`(含 UI 字段描述) + `ExporterResult` + 单例 registry，注册 **8 个格式**：json/csv/hdf5/ftp1/lerobot/lerobot_create/rlds/ros2（registry.py:805-821，实测 `list_formats()` 输出） |
| LeRobot 路径 | 仅"写回已有 parquet"一条路（`tlabel_to_lerobot`）+ base.py 内部自建最小骨架 | 两条路：①写回 `tlabel_to_lerobot`（lerobot.py:131）；②**从零创建 v2.1 数据集** `create_lerobot_dataset` / `tlabeldata_to_lerobot`（lerobot_export.py:370/510，含 tasks.jsonl/episodes.jsonl/state/action） |
| CLI 命令 | validate / list / info / version + **convert / batch-convert / list-adapters / adapter-info** | validate / list / info / version / **export** / template list/validate（cli.py:644-715） |
| 适配器数量 | 9 个（硬编码 `_CONVERTIBLE_ADAPTERS`，分支 cli.py:532-537） | 15 个已注册（实测 `registry.list_adapters()`，含 syntouch/xela/tashan_ts_f_a 等），registry 动态发现 |
| 测试 | 28 个（test_cli_convert.py） | 共 333 collected；`cmd_export` **零测试覆盖**（grep tests/ 无命中） |

### 1.2 四个分支命令与 main 现有命令逐一对照

| 分支命令 | 数据源/核心逻辑 | main 对应物 | 关系判定 |
|---|---|---|---|
| `list-adapters`（分支 cli.py:753-813） | `core.registry.list_adapters()` + `AVAILABLE_ADAPTERS`，按 Data/Sensor 分组，展示扩展名+描述 | `list`（cli.py:433-465）：**同一数据源**，同样按 SensorAdapterBase 分组，只展示 docstring 首行 | **重复**，分支版仅多两列（extensions、描述）和 convert 提示 |
| `adapter-info`（分支 cli.py:816-926） | `get_adapter()` + 实例化，展示类名/模块/类型/扩展名/compliance_level/完整 capabilities 表/图像形状 | `info`（cli.py:467-515）：**同一函数的精简版**，已有扩展名、capabilities 计数（14维/22维两口径）、sensor_info；缺：compliance_level、capabilities 明细表、docstring、图像形状 | **互补 70%**，差异是增强项而非新能力 |
| `convert`（分支 cli.py:541-621） | `adapter.load(input)` → `converters.base.get_converter(to).export(...)`，支持 lerobot/ftp1 | `export`（cli.py:527-575）：`create_lerobot_dataset(input, output, adapter_name, ...)`，**只支持 lerobot（新建），不支持 ftp1，不支持写回已有 LeRobot 数据集** | **部分重叠**：lerobot 新建场景已被覆盖且更强；**ftp1 端到端转换在 CLI 层缺失**（Python API 已通） |
| `batch-convert`（分支 cli.py:624-751） | rglob `supported_extensions` → 逐文件 load/convert → 汇总成败 | 无。`ftp1.batch_to_ftp1()`（ftp1.py:515）只接受 **JSON 路径列表**，不走适配器 | **真正缺失，无替代** |

---

## 2. `_safe_float()` 专项核实：当前 main 确有未修复的数据丢失路径

### 2.1 分支版做了什么

分支 `tlabel/converters/lerobot.py:190-206`（`git diff main..feat/...` 确认这是两版 lerobot.py 的**唯一差异**）：

```python
def _safe_float(val) -> float:
    if val is None: return 0.0
    if isinstance(val, bool): return 1.0 if val else 0.0
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.array(val, dtype=float)
        return float(np.linalg.norm(arr)) if arr.size > 0 else 0.0   # 向量→L2 模长
    if isinstance(val, str): return 1.0 if val else 0.0              # 枚举→0/1
    return 0.0
```

### 2.2 0.23.1 那次修复（`be7121d`）与它的关系

`be7121d` "fix(lerobot): handle non-numeric fields in converter"（2026-09-07）把原来的
`row = [float(sv2.get(name, 0.0)) for name in feature_names]`（遇 list/str 直接抛 TypeError 崩溃）
改成 main 现在的 lerobot.py:191-203：

```python
if val is None: row.append(0.0)
elif isinstance(val, (int, float)): row.append(float(val))
elif isinstance(val, bool): ...          # 注意：此分支不可达，bool 是 int 子类，已被上一行捕获（结果恰好一致）
else: row.append(0.0)                    # list/tuple/ndarray/str 全部落这里
```

即：0.23.1 只把"崩溃"修成了"不崩溃"，**语义上把向量和枚举一律抹成 0.0**，属于"用静默数据丢失换不崩溃"。

### 2.3 实测（在 main 工作区直接跑，非推测）

构造合法 Schema V2 标注（向量字段 `contact_centroid=[3,4]`、`force_vector=[1,2,2]`、`slip_velocity=[3,4]`；枚举 `contact_region="digital"`、`manipulation_phase="grasp"`、`compliance_level="L2"`），调用 main 的 `tlabel_to_lerobot(overwrite=True)`，读回 parquet 得到：

```
帧0: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2, 0, 0.9, 0]
帧1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 25, 1, 0]
```

对照 14 维字段顺序（lerobot_export.py:96-120）：
- 第 2 维 `contact_centroid`：期望 L2 模长 **5.0**，实际 **0.0** ❌
- 第 5 维 `force_vector`：期望 **3.0**，实际 **0.0** ❌
- 第 8 维 `slip_velocity`：期望 **5.0**，实际 **0.0** ❌
- 第 3/9/10/14 维枚举（contact_region/manipulation_phase/texture_class/compliance_level）：全部 **0.0** ❌

这不是边角路径：`tlabel_to_lerobot` 是公开 API（`converters/__init__.py:25` 导出）、被 `export/registry.py:616-631` 的 `_LeRobotExporter` 调用、被 `examples/lerobot_integration.py` 和 `tests/test_v18_detect_image_shape.py` 使用。任何走"标注写回已有 LeRobot 数据集"流程的用户都会中招；而"新建数据集"路径 `lerobot_export._encode_schema_v2_to_vector`（lerobot_export.py:48-120）反而有**正确且更严格的编码**（向量 L2、枚举映射成类别索引、None→-1.0）。

**结论：两条 LeRobot 路径编码语义不一致（lerobot_export.py:55 注释自称"与 lerobot.py 保持一致"，实际并不一致），这是比 0.19.0 丢函数更值得修的真 bug。**

### 2.4 恢复建议（不直接抄 0.19.0 版）

不要原样恢复 `_safe_float` 的"str→0/1"，应抽取 `lerobot_export.py:77-95` 的 `_vec_norm / _enum_idx / _scalar` 为共享编码函数（如 `converters/_encoding.py`），让写回路径与新建路径共用同一套 14 维编码，枚举按类别索引（含 -1 哨兵）而非 0/1。改动点仅 lerobot.py:187-204 一段 + 新增单测。

---

## 3. `converters/base.py` 去留判断

### 3.1 它提供什么

- `BaseConverter`：类属性 `name/description/file_extension` + 静态方法 `export()/is_available()/required_dependencies()`（base.py:33-70）
- 两个实现：`FTP1Converter`（薄包装 `tlabel_to_ftp1`，base.py:77-131）、`LeRobotConverter`（临时 JSON + 自建最小 parquet/info.json 骨架再调 `tlabel_to_lerobot`，base.py:138-271）
- 注册表：`CONVERTERS` / `get_converter()` / `list_converters()` / `list_available_converters()`（base.py:278-303）

### 3.2 与 main `export/registry.py` 的关系

`ExporterBase`（registry.py:193-238）是**严格超集**：

| 能力 | 0.19 BaseConverter | main ExporterBase/Registry |
|---|---|---|
| 统一 export 签名 | ✅ 静态方法 | ✅ 实例方法（registry.py:209） |
| 依赖可用性检查 | ✅ `is_available()` | ❌ 无显式抽象（依赖在 export 内惰性 import），但用 spec 分类表达 |
| 格式元信息 | name/description/file_extension | `ExporterSpec`：id/name/category/icon/file_ext + **UI 动态表单 `fields`**（registry.py:111-145） |
| 返回值 | 裸 dict | `ExporterResult`（大小/统计/可复现代码片段/错误态，registry.py:152-186） |
| 动态发现 | 硬编码 2 个 | 单例注册 8 个，含预留 RLDS/ROS2（registry.py:805-821） |
| UI/API 序列化 | 无 | `to_dict()`（registry.py:321-350） |
| FTP-1 配置透传 | 4 个固定参数 | 6 个字段（多 target_image_size、store_raw_uint8，registry.py:450-521） |

实测 main 的 registry 端到端可用：`tlabel` 适配器加载 `demo_gelsight.json`（150 帧）→ `registry.export("ftp1", ...)` 成功产出 zarr（150 time_steps）。

### 3.3 判断

**放弃恢复 base.py 原文件。** 理由：
1. 恢复它会立刻产生"converters.CONVERTERS"和"export.get_registry()"两套注册中心，违反单一事实源；JOSS 审稿人看到两套并存会扣分。
2. 它唯一相对 registry 多的东西是 `is_available()/required_dependencies()` 的**声明式依赖探测**——这个值得做，但正确做法是给 `ExporterSpec` 加一个可选 `required_packages: List[str]` 字段并在 registry 加 `is_available(id)`，而不是复活旧基类。
3. LeRobotConverter 里"自建最小骨架"的职责已被 `lerobot_export.py` 完整取代（而且产物是合规 v2.1 结构：tasks/episodes jsonl 都有）。

---

## 4. 四个 CLI 命令的处置建议（含具体落点）

> 原则：main 的命令名 `export` 已发布多个版本，不引入 `convert` 同义词制造分裂；能并入现有命令就并入。

### 4.1 `list-adapters` → 增强 `list`，放弃新命令名
- 改动：cli.py:433-465 `cmd_list` 增加 `--verbose/-v`（展示 `supported_extensions`、`AVAILABLE_ADAPTERS` 描述）和 `--json`（脚本友好）；参数在 cli.py:657-660 的 parser 上加。
- 信息源全部现成：`tlabel.adapters.AVAILABLE_ADAPTERS`（adapters/__init__.py:25-44）、实例 `supported_extensions`。
- 工作量：半天。风险：无。Breaking：无。

### 4.2 `adapter-info` → 增强 `info`，放弃新命令名
- 在 cli.py:467-515 现有输出中补：`default_compliance_level`（adapters/base.py:39/194 类属性，gelsight.py:238 有覆写实测可读）、capabilities 明细表（现有只打印计数）、类名/模块/docstring 首行、`detect_image_shape()`（可选，包 try/except，分支 cli.py:884-888 已有范式）。
- 工作量：半天。风险：无。Breaking：无。

### 4.3 `convert` → 不新增命令；把 `export` 接到 registry 并补 ftp1/lerobot 写回
现状 `cmd_export`（cli.py:527-575）绕过 registry 直接调 `create_lerobot_dataset`，等于手里有 8 格式的 registry 却只接了 1 个。改造：
- `tlabel export <input> <output> --adapter/-a <name> --format/-f <id>` 的既有签名保留（不 breaking）；
- `-f` 的 choices 从 registry.list_formats() 动态取，支持 `json/csv/hdf5/ftp1/lerobot/lerobot_create`；
- 实现统一为：`adapter.load(input)` → `registry.export(fmt, data, output, **spec_kwargs)`；
  - ftp1 所需的 sensor_name/functional_areas 等用 `--opt key=value` 透传（或第一版只做 `--sensor`/`--areas` 两个显式参数，避免过度设计）；
  - `lerobot`（写回已有数据集）与 `lerobot_create`（新建）作为两个 format id 天然区分，不用新造概念；
  - `--task/--task-description/--episode-index` 仅 lerobot_create 消费，现状如此，保持。
- 这一步顺带**让 registry 第一次有了真实调用方**（目前除 __init__ 导出外全包内无任何调用，grep 实测），架构债变资产。
- 工作量：1 天（含联调 ftp1 两种传感器类型）。风险：中（export 是已发布命令，需保证 lerobot 默认行为不变——把当前默认 `lerobot` 映射到 `lerobot_create` 即可保持）。Breaking：默认值不变即无；help 文案更新。

### 4.4 `batch-convert` → 恢复，作为唯一新命令（建议 `export --batch` 或 `export-batch`）
- 分支实现逻辑（cli.py:624-751）今天仍成立：`supported_extensions` rglob → 去重 → 保持相对目录结构 → 逐文件 load/export → 成败汇总，失败不中断，返回码反映失败数。这套逻辑与 registry 无关，可直接借鉴。
- 落点：复用 4.3 改造后的单文件导出函数，循环调用即可；输出命名按 `file_ext`（spec 已提供，registry.py:133）决定后缀。
- 工作量：半天（在 4.3 完成之后）。风险：低。Breaking：无（纯新增）。
- 注意：分支批量逻辑里"lerobot 目标输出为目录、ftp1 输出 .zarr"的后缀分支（分支 cli.py:717-721）应改成查 spec.file_ext，别硬编码。

---

## 5. 测试可移植性评估（28 个测试逐类盘点）

分支 `tests/test_cli_convert.py` 实际 28 个 `def test_`（grep 实测），分 5 组：

| 测试组 | 数量 | 可移植性 | 原因 |
|---|---|---|---|
| `TestConverterBase` | 8 | **全部废弃重写** | 全部 import `tlabel.converters.base`（BaseConverter/FTP1Converter/.../CONVERTERS），base.py 不恢复即全灭；对应能力改用 registry API 重写（`get_registry().get("ftp1").spec()` 等），约 8 个新断言，半天 |
| `TestListAdapters` | 3 | 改命令名后可复用 | 断言 cmd 存在、输出含 gelsight/paxini/扩展名/convert 提示；合并到 §4.1 后改为调 `cmd_list(["--verbose"])`，提示文案改断言 `export`/`ftp1` |
| `TestAdapterInfo` | 5 | 基本可复用 | 依赖 gelsight/paxini 适配器仍在（实测已注册）；`GelSightAdapter`/`PaxiniAdapter` 类名需在当前代码核对；nonexistent 返回 1 的断言与现 cmd_info 行为一致（cli.py:471-474） |
| `TestConvert` | 4 | 需改造 | 2 个端到端用例引用 `tlabel/demo_data/demo_gelsight.json`——该文件**当前是旧 22 维 tlabel_v2 格式**（实测 frames[0] 无 schema_v2，顶层也无 feature_names_v2）。经 tlabel 适配器加载会走 from_tlabel_v1 转换，能跑通但断言的输出结构是旧骨架；建议改用/新增 14 维 schema_v2 demo fixture，且走 `lerobot_create`（产物为 episode_000000.parquet，不是分支断言的 chunk-0000.parquet，路径断言必须改） |
| `TestBatchConvert` | 3 | 可回收 2 个 | 空目录/不存在目录的错误路径断言直接可用；ftp1 批量端到端在新命令上重写，zarr 依赖 CI 有则跑（zarr 3.2.1 本地实测可用） |
| `TestCLIIntegration` | 5 | 改断言 | help 中命令清单要改成新集合；subprocess 调 `convert`/`list-adapters` 改成 `export`/`list --verbose`/`info`；端到端 ftp1 用例可保留思路 |

**净评估：原样移植跑不通（import 即失败 8 个、命令不存在失败 ≥8 个、parquet 路径断言失败 2 个，约 18/28 红）；按本方案重写可回收约 60-70% 的测试意图，需要 1 天，且应同时补齐 §2.4 的编码一致性测试（向量取模、枚举索引、None 哨兵、两路径结果一致）。** 现有 main 对 `cmd_export` 零覆盖，这批测试重写后顺带补上该盲区。

---

## 6. 分阶段实施方案（每步可独立验收/独立发 minor/patch）

### Phase 1：修 lerobot 写回路径编码（P0，建议 v0.23.3 patch）
- 改动文件：
  - `tlabel/converters/lerobot_export.py`：把 `_vec_norm/_enum_idx/_scalar` 与 4 个枚举映射（lerobot_export.py:59-95）抽为模块级/新 `tlabel/converters/_encoding.py`；
  - `tlabel/converters/lerobot.py`：lerobot.py:187-204 改为调用共享编码，按 feature_names 逐字段编码（注意保持"特征名驱动"而非硬编码 14 个，以兼容 feature_names_v2 缺省从帧推断的分支，lerobot.py:181-185）；
  - 新增 `tests/unit/test_lerobot_encoding.py`：向量模长、枚举索引、None→-1.0、bool、两路径对同一 TLabelData 产出一致 tactile 向量；
  - CHANGELOG.md：Fixed 条目。
- 工作量：半天。风险：低。Breaking：**行为变更但属 bugfix 性质**（之前输出 0.0 的列将变为真实值；meta feature_names 描述不变）。发版说明需明示"写回路径非数值字段此前被错误置零"。
- 验收：新增测试 + 全量 pytest（当前基线 333 collected）通过；§2.3 实测脚本期望向量位为 5/3/5。

### Phase 2：CLI 信息命令增强（P1，可随 v0.24.0）
- 改动文件：`tlabel/cli.py`（cmd_list/cmd_info 及两个 parser，cli.py:433-515、657-665）；对应 tests/unit/test_cli.py 增补（该文件现有 14 个测试，风格可直接沿用）。
- 工作量：半天-1 天。风险：低。Breaking：无（纯加 flag 和输出行）。
- 验收：`tlabel list -v`、`tlabel list --json`、`tlabel info gelsight` 输出含扩展名/compliance/能力明细。

### Phase 3：export 接 registry，补 ftp1/写回（P1，建议 v0.24.0）
- 改动文件：`tlabel/cli.py`（cmd_export 重写为 registry 驱动，cli.py:527-575、669-692）；`tlabel/export/registry.py` 可选加 `is_available`/`required_packages`（把 base.py 唯一有价值的设计在新架构里复活）；新增测试文件（取代 test_cli_convert.py 中 convert 组）。
- 工作量：1 天。风险：中（已发布命令的默认行为，需保 lerobot 默认 == 现状 lerobot_create）。Breaking：默认行为不变；新增 `-f` 取值算增强。
- 验收：`tlabel export demo.json out.zarr -a tlabel -f ftp1 --sensor GelSightMini --areas 0,1` 端到端；现有 lerobot 调用方式结果不变。

### Phase 4：批量导出（P2，可 v0.24.0 或 v0.25.0）
- 改动文件：`tlabel/cli.py`（新增批量入口，逻辑借鉴分支 cli.py:624-751 但后缀查 spec.file_ext）；测试 3-4 个（空目录/不存在/部分失败返回码/ftp1 批量）。
- 工作量：半天。风险：低。Breaking：无。
- 验收：混合多文件目录批量转换，失败隔离、汇总正确。

### Phase 5（可选，不建议本期做）
- `converters/base.py`：**不做**。若未来要彻底统一，方向是把 ftp1.py/lerobot*.py 里的函数式入口全部收编成 ExporterBase 实现并废弃函数式 API——那是 v1.0 级别的重构，不在本回归修复范围。

总工作量：**3-3.5 人天**（Phase1 半天 + Phase2 半天 + Phase3 一天 + Phase4 半天 + 测试与文档半天）。

---

## 7. 风险与注意事项

1. **版本合规**：本仓库 MEMORY 锚定的三层架构（Layer1 Schema 14 维 / Layer2 Adapter / Layer3 派生）不受影响——本方案全部改动在 Layer2/3 边界内，不动 schema 与适配器基类。
2. **capabilities 迁移半成品是另一个坑**（10 个适配器仍输出旧 22 键，见 v0.24 P0 记录），与本问题正交；批量/转换 CLI 会同时加载新旧两类适配器，Phase 3 联调时应以 22 维 demo（demo_gelsight.json）和 14 维 fixture 各跑一遍。
3. **zarr v3 兼容**：本地环境 zarr 3.2.1，ftp1 导出实测通过；CI 需确认锁版本策略，`is_available` 恢复时依赖声明建议写 `zarr>=2.16`（沿用分支 base.py:131）但要实测 v3 API。
4. **lerobot.py 中 bool 判断顺序**（lerobot.py:197-200）是死分支（bool 先被 int/float 捕获），Phase 1 重写时顺手消除，不必单独发版。
5. **CHANGELOG 勘误已存在**（CHANGELOG.md:115-119），方案落地后应在该 Notes 区更新状态为"resolved in vX.Y.Z"，保持考古链可追溯。
6. 禁止直接 merge 分支：`git diff --stat main..feat/v0.19.0-convert-cli` = 133 文件、+6573/-19104（基于 0.18.2，落后四个大版本），本方案全部为在 main 现状上的小步重写。

---

## 8. 最终明确建议

1. **恢复 `_safe_float` 的修复意图，但不是恢复那个函数**：以共享编码模块统一 lerobot 两条路径的 14 维编码（向量 L2、枚举类别索引），这是经实测确认的静默数据丢失 bug，应作为最高优先级 patch 单独发出。
2. **`base.py` 和 `convert` / `list-adapters` / `adapter-info` 三个命令名不恢复**：它们已被 `export/registry.py` + `lerobot_export.py` + 现有 list/info 在架构上取代，原样复活只会制造双注册体系和命令别名垃圾。
3. **值得恢复的真实能力只有两个**：① export CLI 走 registry 后对 ftp1（及 lerobot 写回）的端到端支持；② 批量转换。用"增强现有 export/list/info + 一个批量入口"实现，对外仍然是干净的一套命令。
4. **测试重写而非移植**，顺手补掉 cmd_export 当前零覆盖的盲区。
5. 全部工作 3-3.5 人天，可拆 4 个独立验收步，仅 Phase 1 建议尽快单独发 patch。
