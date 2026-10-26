# PIT-7：适配器 capabilities 能力声明不一致 —— 落地方案评估

- 状态：调研/方案（不含业务代码改动、不含 git 提交）
- 基线版本：v0.23.2（main，HEAD `a41fb0b`）
- 调研日期：2026-10-27
- 规范依据：`schema/tlabel-schema.json`（`properties.capabilities`，13 键，`additionalProperties: false`，`required: ["contact"]`）、`docs/spec/tlabel-spec-v1.md` §6.3（S-6.3-1/2/3）、§7.4（S-7.4-1/2）、conformance 项 C-15、deviation 登记 D-17/D-18
- 合规范本：`tlabel/adapters/xela.py:199-220`

> 结论先行：规范（S-7.4）与 JSON Schema 已经把目标键集钉死，而 14 个适配器中只有 `xela` 1 个真正合规；3 个多出 `compliance_level` 键，10 个仍返回旧 22/23 键扁平字典。这些脏 capabilities 经 `TLabelData.capabilities` → `to_dict()`（`tlabel/core/types.py:656`）原样写入导出的 TLabel JSON，**实测无法通过仓库自带 schema 校验**（本文 §4 有复现）。建议在 **0.24.0 minor** 中分 4 步修完，patch 版本只加 warning 不做破坏。

---

## 1. 规范要求（目标态）

`schema/tlabel-schema.json` 的 `properties.capabilities`：

- 合法键 **恰好 13 个**（注意：不是 14 维；`compliance_level` 是帧级 schema_v2 元字段，不是能力键）：
  `confidence, contact, contact_centroid, contact_region, force_magnitude, force_vector, manipulation_phase, object_deformation, slip_event, slip_velocity, temperature, texture_class, torque_vector`
- `additionalProperties: false`；`required: ["contact"]`；所有值必须是 boolean。

规范条款：
- S-6.3-1：`get_capabilities()` 返回的键必须是 13 键集合的子集，值为 bool；
- S-6.3-2：声明 `false` 的维度每帧必须输出 `null`；声明 `true` 的维度应在物理信号可得时尽量填充；
- S-6.3-3：不得声明做不到的能力，也不得填充声明为 false 的维度；
- S-7.4-1/2：13 键 + `contact` 必填 + `compliance_level` 禁止出现；
- C-15：conformance 测试要求键集合 ⊆ 13 键且无 `compliance_level`。

当前合规实现只有 `xela`（`tlabel/adapters/xela.py:199-220`，且有守护测试 `tests/test_adapter_xela.py:432-441` 断言 `set(caps) == schema_caps` 且 `len(caps) == 13`）。

---

## 2. 现状复核（14 个适配器，实测）

通过注册表逐个实例化（`tlabel/core/registry.py:89` `list_builtin_adapters()`）实测：

| 注册名 | 类 / 文件:行 | 键数 | 问题分类 |
|---|---|---:|---|
| `xela` | `XelaUskinAdapter` / `xela.py:199` | 13 | ✅ 完全合规 |
| `syntouch` | `SynTouchBioTacAdapter` / `syntouch.py:463` | 14 | ⚠️ 多 `compliance_level`（`:479`） |
| `tashan_ts_f_a` | `TashanTsFAAdapter` / `tashan_ts_f_a.py:68` | 14 | ⚠️ 多 `compliance_level`（`:89`） |
| `paxini_px6d` | `TLabelPX6DAdapter`（placeholder）/ `paxini_px6d.py:46` | 14 | ⚠️ 多 `compliance_level`（`:65`）；注释自称"14 个字段"（`:49`）是错误口径 |
| `gelsight` | `GelSightAdapter` / `gelsight.py:313` | **23** | ❌ 旧键，且比标准 22 键还多一个 `deformation_magnitude_peak`（`:318`） |
| `paxini` | `PaXiniAdapter` / `paxini_dataset.py:102` | 22 | ❌ 旧键（部分为 false） |
| `daimon` | `DaimonAdapter` / `daimon_dataset.py:884` | 22 | ❌ 旧键；且 `load()` 运行时还会改 caps（`:1271-1280`） |
| `daimon_dm_tac` | `DaimonDmTacAdapter`（SensorAdapterBase）/ `daimon_dm_tac.py:62` | 22 | ❌ 旧键，与 `daimon_dataset` 完全相同的字典 |
| `tacquad` | `TacQuadAdapter` / `tacquad.py:402` | 22 | ❌ 旧键（无返回类型注解） |
| `touchd` | `ToucHDAdapter` / `touchd.py:420` | 22 | ❌ 旧键 |
| `univtac` | `UniVTACAdapter` / `univtac.py:108` | 22 | ❌ 旧键 |
| `vtouch` | `VTouchAdapter` / `vtouch.py:362` | 22 | ❌ 旧键 |
| `ycb_slide` | `YCBSlideAdapter` / `ycb_slide.py:292` | 22 | ❌ 旧键（部分为 false） |
| `paxini_gen3` | `PaxiniGen3Adapter`（SensorAdapterBase）/ `paxini_gen3.py:188` | 22 | ❌ 旧键；**签名违规**：`get_capabilities(self, file_path=None)`，与基类 abstractmethod 无参签名不一致（参数还从未被使用） |
| `tlabel`（第 15 个） | `TLabelAdapter` / `tlabel_format.py:36` | 透传 | 🔶 特殊：`get_capabilities(self, file_path)` 从文件读取并原样透传，无法无参调用（被所有遍历型测试以 TypeError 跳过） |

旧 22 键字典的模板见 `gelsight.py:313-333`；模块级空模板 `_extract_tlabel_v2()` 列出了 23 个旧字段（`gelsight.py:97-110`，含 `deformation_magnitude_peak`）。旧字段常量还残留在 `tlabel/cli.py:29-53`（`LEGACY_V2_DIMENSIONS`）。

CHANGELOG `0.22.3` 声称"`get_capabilities()` standardized to the 14-dimensional Schema V2 format"——**实际只迁移了当时新增的 xela/tashan/syntouch 三个适配器，10 个老适配器漏迁**，与 D-17 的登记一致。

---

## 3. 14 个适配器 → 13 合法键的目标映射表

判定口径（重要，避免"只删键不判语义"）：
- 以 **`load()`/`stream_frames()` 实际产出的帧内容**为准（capabilities 描述的是导出文件的能力），而不是 `extract_schema()` 的理论能力——因为多数适配器的 `load()` 路径**根本不调用自己的 `extract_schema()`**（见 §3.2），两者不一致；本次迁移以 load 实际产出为真值，extract_schema 的偏差登记为后续债。
- 帧内 `manipulation_phase`：所有视觉/阵列适配器在 `load()` 中都做了规则推断（`_infer_phases` 等，见 §3.3），虽然写在 `TLabelFrame.manipulation_phase` 而不是 `schema_v2.manipulation_phase`（那是另一个已存在的不一致，见 §3.4），按 xela 的严格口径（"不做推断 → false"）本应全部 false，但与"文件实际携带阶段信息"冲突。**推荐按实际产出标 true，并在注释里写明"heuristic inference"**；xela/tashan/paxini_px6d 保持 false（它们确实不推断）。这个取舍需要圣姐/罗西拍板，表中给推荐值并注明争议。

### 3.1 总表（推荐值）

| 适配器 | contact | contact_centroid | contact_region | force_magnitude | force_vector | torque_vector | slip_event | slip_velocity | manipulation_phase | texture_class | object_deformation | temperature | confidence |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `xela`（已合规，不改） | T | T | F | T¹ | T¹ | F | F | F | F | F | F | F | T |
| `syntouch`（删 compliance_level） | T | T | T | T | F | F | T | F | T² | F | T | T | T |
| `tashan_ts_f_a`（删 compliance_level） | T | F | F | T | T | F | F³ | F | F | F | F | F | T |
| `paxini_px6d`（删 compliance_level） | F | F | F | F | F | F | F | F | F | F | F | F | F |
| `gelsight` | T | T | F | T | T⁴ | F | T | F | T† | F | T | F | T |
| `paxini`(dataset) | T | T | F | T | F | F | T | F | T† | F | T⁵ | F | T |
| `paxini_gen3` | T | T | F | T | F | F | T | F | T† | F | T⁵ | F | T |
| `daimon`(dataset) | T | T | F | T | T⁶ | F | T | F | T† | F | T | F | T |
| `daimon_dm_tac` | T | T | F | T | F | F | T | F | F†? | F | T | F | T |
| `tacquad` | T | T | F | F⁷ | F | F | T | F | T† | F | F⁷ | F | T |
| `touchd` | T | T | F | T | T⁸ | F | T | F | T† | F | T | F | T |
| `univtac` | T | T | F | T | T⁹ | F | T | F | T†² | F | T | F | T |
| `vtouch` | T | T | F | T | F | F | T | F | T† | F | T | F | T |
| `ycb_slide` | T | T | F | T | F | F | T | F | T† | F | T | F | T（见注） |

标注说明：

- ¹ `xela`：力字段需 `force_scale` 标定才有 SI 值，默认 null（`xela.py:210-211, 319-323`）。xela 自己的既定先例是"可经标定获得 → 声明 true"，沿用。
- ² `syntouch`：`load()` 用数据集 phase 标签经 `_infer_phases` 推断（`syntouch.py:707-709, 746`），`extract_schema` 也会透传 `phase`（`:591-593`），标 T。
- ³ `tashan_ts_f_a`：`extract_schema()` 中 `slip_event=None`（`tashan_ts_f_a.py:169`）——注意这本身违反 schema_v2 required bool，是另一个已存在的脏点，本次不动；cap 保持 false。
- ⁴ `gelsight`：`extract_schema()` 在 pkl 提供 `force_vector_N` 时输出 3D 力并升级 L3（`gelsight.py:286-292`），`load()` 对 GelSight force 数据集始终传 `forces[i]`（`:497-498`），所以 force_vector=T 有依据；DIGIT 子集无力向量时为 null，符合 S-6.3-2 的"signal available 时填"。
- ⁵ `paxini`/`paxini_gen3` 的 `object_deformation` 实为**压力均值代理量**（`paxini_dataset.py:347` `deformation_magnitude = nf_mag`；`paxini_gen3.py:586` `deform_mag = pressure_map_norm.mean()`），语义上不是物体形变。但 `extract_schema()` 已把它写进 `object_deformation`（`paxini_dataset.py:93`、`paxini_gen3.py:180`）。**务实选择：本次保留 true + 注释标注"pressure proxy, not physical deformation"**；严格选择（推荐后续债）：改标 false 并把 extract_schema 的映射改成 null。需拍板。
- ⁶ `daimon`：`extract_schema()` 在有 shear 视频时合成 force_vector 并升 L3（`daimon_dataset.py:968-985`）；但 `load()` 帧走的是 `from_tlabel_v1`，**没有调用这段增强**（`:1230`），所以导出文件实际拿不到 force_vector。严格按 load 口径应是 **false**；按 extract 口径是 true。**推荐 true（条件能力，仿 xela 先例）+ 在 `daimon_dataset.py:1271-1280` 的运行时降级逻辑里同步关掉它**，同时登记"load 未接 extract_schema 增强"的债（§8 R2）。
- ⁷ `tacquad`：`extract_schema()` 明确把 force_magnitude / object_deformation 设为 None，自评 L1（`tacquad.py:385-401`）；load 也走 from_tlabel_v1，其中 force_magnitude 来自视觉经验映射（`gelsight.py:178` 同款）。**矛盾点**：from_tlabel_v1 产出的 force_magnitude 恒等于 deformation_magnitude（视觉代理），而 tacquad 的 extract 自评不提供力。推荐 **force_magnitude=false、object_deformation=false**（尊重适配器作者的 L1 自评），但这意味着 load 路径会出现"cap=false 但帧里有值"——解决方式是在 R2 把 load 接回 extract_schema（或在帧构建时把这些代理量置 null）。若暂不动 load，则二选一妥协：都标 true（与实际产出一致，但违背作者自评）。**默认推荐：尊重自评标 false，R2 修 load。**
- ⁸ `touchd`：是唯一带真实 3D 力 ground-truth 的数据集（`touchd.py:453` 注释 "L3 — 唯一拥有完整3D力标签"），`sensor_specific.force_xyz_*` 恒有（`:761-768`）；但 `load()` 帧同样走 `from_tlabel_v1` 而**没接 extract_schema 的 L3 增强**（`:790` vs `:484-540`）。能力声明按物理真值标 force_vector=T，R2 必须修 load，否则文件层面不满足 S-6.3-2 的 SHOULD。
- ⁹ `univtac`：`extract_schema()` 用 shear 场合成 force_vector（`univtac.py:189-204`），条件 L3；load 同样未接（`:372`）。force_vector=T + R2。
- † `manipulation_phase` 的推断依据：
  - gelsight `_infer_phases`（`gelsight.py:205-229`，load 在 `:412/:507`）
  - paxini dataset `_infer_phase`（`paxini_dataset.py:427`，load `:227/:242`）
  - paxini_gen3 `_infer_phase`（`paxini_gen3.py:686`，`:365/:390/:528/:553`）
  - daimon `_infer_phases`（`daimon_dataset.py:832-866`，`:1239-1241`）
  - tacquad `_infer_phases`（`tacquad.py:279 同款`，`:498-506`）
  - touchd `_infer_phases`（`touchd.py:279-...`，`:810-818`）
  - univtac：来自数据集 `atom/tag` 真值标签映射 `TAG_TO_PHASE`（`univtac.py:74-86, 654-660`）——这是 10 个旧适配器里唯一的**真标注**，标 true 最硬
  - vtouch `_infer_phases`（`:697`）
  - ycb_slide（`:517-521, 540` 与 `:710, 722`）
  - daimon_dm_tac：实时/兼容 load 都写死 `"idle"`（`daimon_dm_tac.py:382, 515`），**不算阶段能力，标 false**（若拍板口径是"文件带 phase 字段即 true"则可改 true，但 idle 是哨兵默认值，推荐 false）
- †² univtac 的 phase 标签可能出现 V2 枚举外的值（`TAG_TO_PHASE` 含 `insert`，`univtac.py:81`），schema_v2 枚举为 `pre_contact/approach/grasp/lift/hold/place`（`tlabel/core/schema.py:12`）——另一个已存在的脏点，登记为后续债，不阻塞本次。
- `ycb_slide.confidence=T`：其 confidence 是按 contact 分支写死的 0.95/0.8/0.85（`ycb_slide.py:288, 527, 710`），不是真实置信度估计，但 `TLabelFrame.confidence` 恒被序列化（`types.py:184`，schema 中 confidence 是 required），文件确实携带该字段，按 xela 对 force_scale 的同款口径（"可提供即 true"）标 true；注释中标注"heuristic/contact-based constant"。

### 3.2 关键陷阱：`extract_schema()` 与 `load()` 产出不一致

10 个旧适配器里，只有 `gelsight` 的 `load()` 真正调用了 `self.extract_schema()`（`gelsight.py:505`）。其余全部在 load 内直接 `TLabelSchemaV2.from_tlabel_v1(tlabel_v2)`：

- `daimon_dataset.py:1230`、`daimon_dm_tac.py:381,514`、`paxini_dataset.py:241`、`paxini_gen3.py:389,552`、`tacquad.py:730`、`touchd.py:790`、`univtac.py:372`、`vtouch.py:696`、`ycb_slide.py:539,721`

后果：`extract_schema()` 里精心写的条件 L3 增强（touchd 的真实 force_vector、daimon/univtac 的 shear 合成 force_vector、slip_velocity 从 optical_flow 提取等，见 `touchd.py:484-540`、`daimon_dataset.py:968-1000`、`univtac.py:119-218`、`vtouch.py:421-447`）**在导出文件里全部不存在**。而 `from_tlabel_v1`（`tlabel/core/schema.py:187-292`）的产出是：
- contact / slip_event：0/1 float → bool；
- contact_centroid：有 centroid_x 就给 `[cx, 0.0]`（y 是假的，`schema.py:208-212`）；
- force_magnitude：v1 的同名值（视觉代理，单位不是 N）；
- force_vector：仅当 v1 dict 含 `force_direction` 时合成（22 键 dict 恒有此键，`gelsight.py:180`），所以**视觉适配器的帧里实际上会出现 force_vector**——与多个适配器 extract 自评的 L2/false 矛盾；
- slip_velocity：依赖 v1 的 `slip_direction` 键，但 22 键 dict 里**根本没有这个键**（全仓 grep 仅 `schema.py:241` 引用），所以恒为 None；
- manipulation_phase：phase_map 对旧状态机值可映射，但 22 键 tlabel_v2 dict 里也没有该键（它在 TLabelFrame 顶层），所以 schema_v2 内恒为 None；
- texture_class / contact_region / temperature / torque_vector：恒 None；
- compliance_level：恒写死 "L1"（`schema.py:290`）——**即使适配器类声明 default L2/L3，load 出来的帧 compliance_level 仍然全是 L1**。

这意味着 capabilities 迁移若只改字典、不顺 load 路径，新的 13 键声明会和帧内容产生大面积 S-6.3-2/S-6.3-3 冲突。因此本方案把"load 接回 extract_schema 或在帧构建时做诚实化处理"列为 **R2（迁移同期必须做的对齐项）**，但允许分 PR 推进。

### 3.3 旧键 → 13 键的语义映射判断

旧 capabilities 键全部来自 22 维 tlabel_v2 flat 特征空间（`gelsight.py:97-110`）。映射分三类：

**A. 有直接/近似对应，迁移为对应新键（语义降级但可继承 true）：**

| 旧键 | 新键 | 判断 |
|---|---|---|
| `contact` | `contact` | 直接（float 0/1 → bool） |
| `contact_area` | （无直接对应） | 不映射为 contact_region（面积 ≠ 解剖区域）；信息保留在 sensor_specific/特征层，cap 丢弃 |
| `centroid_x` | `contact_centroid` | **近似**：v1 只有 x，from_tlabel_v1 用 `[cx, 0.0]` 造假 y（`schema.py:211`）；daimon_dm_tac/gelsight/tacquad/ycb_slide 的 extract 用 `[cx, 0.5]`。标 true 但必须注释"y 轴为占位/估算" |
| `force_magnitude` | `force_magnitude` | 同名但单位不诚实：v1 是图像范数/压力均值代理（`gelsight.py:177-178` `force_mag=deformation_mag`；`daimon_dm_tac.py:167` `*100` 经验映射），不是 N。标 true 时注释"vision proxy, uncalibrated"；xela 式标定后才是 SI |
| `force_direction` | `force_vector` | **仅合成依据**：from_tlabel_v1 把 magnitude+direction 合成为 `[Fx,0,Fz]`（`schema.py:222-231`），Fy 恒 0。只有物理上有 3D 力/剪切的适配器才应据此或据 extract 增强标 true（daimon、univtac、touchd、gelsight 条件性）；paxini/ycb/vtouch 等 force_direction 恒 0 或无剪切，标 false |
| `slip_event` | `slip_event` | 直接 bool；但注意 gelsight 系的 v1 `slip_event` 是**图像梯度角方差代理** `slip_ev`（`gelsight.py:141-143,182`），paxini 才是真实滑移检测。标 true 但注释来源 |
| `deformation_magnitude` | `object_deformation` | **语义近似映射**：from_tlabel_v1 明确此映射（`schema.py:199,266-268`）。视觉触觉适配器可标 true；paxini 压力阵列下它是压力代理（见 ⁵） |
| `force_peak` / `deformation_magnitude_peak` | — | 丢弃（派生统计量，13 维无对应） |
| `optical_flow_magnitude`/`optical_flow_direction` | `slip_velocity` | **间接**：只在 extract_schema 增强里被换算成滑移速度向量（`daimon_dataset.py:992-1000` 等 4 处），且 load 未接。load 接回 extract 后可支撑 slip_velocity=true；**当前阶段一律标 false**（文件中恒 null） |
| `texture_energy` / `edge_density` | `texture_class` | **不映射**：能量/边缘密度是连续视觉统计量，不是纹理类别枚举（smooth/rough/...）。无适配器做分类，全部 false |

**B. 纯中间特征/派生量，直接丢弃（13 维无语义对应）：**

`normal_field_magnitude`、`normal_field_variance`、`shear_field_magnitude`、`shear_field_direction`、`delta_force_normal`、`delta_force_shear`、`friction_cone_ratio`、`slip_entropy`、`temporal_deformation_rate`、`contact_transition`。

这些仍是帧级 22 维 dict 的字段（tlabel_v2 特征层短期保留），但**不属于能力声明**，capabilities 里直接删除。注意 shear 场在 extract_schema 中作为 force_vector 的合成**输入**使用，能力声明只认产出的 force_vector，不认输入。

**C. 旧字典完全没有、但需要按实现新增判断的键：**

`contact_region`（仅 syntouch=true，其 `_electrodes_to_contact_region` 返回枚举值，`syntouch.py:158-190,576`；daimon_dm_tac 的 extract 写死 "palmar" 但 load 走 from_tlabel_v1 恒 None，标 false 并登记）、`torque_vector`（无适配器支持，全 false）、`temperature`（仅 syntouch=true，热敏电阻 `:511-514`）、`confidence`（按帧是否真实携带，见 §3.1 ycb 注）、`manipulation_phase`（见 §3.1 †）。

**禁止项**：`compliance_level` 绝不是能力键（帧级元字段，S-7.4-2），3 个适配器直接删。

---

## 4. 下游消费面排查与 breaking 影响

全仓搜索 capabilities 的读取点（`grep -rn "\.capabilities\|capabilities\["` 与 `get_capabilities(`）：

### 4.1 生产/透传链（脏数据出口）

| 位置 | 用法 | 改 13 键后的影响 |
|---|---|---|
| `tlabel/adapters/base.py:381` | `SensorAdapterBase.collect()` 用 `capabilities=self.get_capabilities()` 组装 TLabelData | 无代码依赖旧键，自动变干净 |
| `tlabel/core/types.py:209-217` | `TLabelData.__init__` 存 `self.capabilities`，无校验 | 无 |
| `tlabel/core/types.py:656` | `to_dict()` 原样输出到 JSON | **修复点**：输出从此符合 schema |
| `tlabel/export/writer.py:296` | HDF5 导出 `meta.attrs["capabilities"]=json.dumps(...)` 透传 | 无代码依赖，自动修复 |
| `tlabel/converters/lerobot_export.py:318` | 透传进 LeRobot `info.json` 的 `capabilities` | 无键级依赖 |
| `tlabel/demo.py:162` | 读旧 demo JSON 的 caps 透传 | 无 |
| `tlabel/demo.py:404-417` | 合成 demo caps：14 键含 `compliance_level`（`:417`） | **需同步删除该键**（第 4 个脏源，适配器清单外） |
| `tlabel/converters/lerobot.py:120,126` | `capabilities = {name: True for name in feature_names}`，键名来自 LeRobot parquet 元数据，**可能是任意字符串** | **第 5 个脏源**：不保证 ⊆ 13 键。迁移时应改为白名单过滤（只保留 13 键交集，未知 feature 名放进 sensor_specific 或丢弃） |

### 4.2 读取/判断链（真正"读键"的代码）

| 位置 | 读法 | 是否依赖旧键 | 影响 |
|---|---|---|---|
| `tlabel/cli.py:166-180` | `validate` 子命令对 caps 做统计：分别数 13/14 键与 `LEGACY_V2_DIMENSIONS` 的命中，只产出 info 级提示 | 是（为兼容而读旧键） | 迁移后 `found_legacy` 恒 0，逻辑可保留（读旧文件仍需要），不报错。**无需改动**，后续版本可考虑对未知键发 warning |
| `tlabel/cli.py:487-496` | `tlabel info <adapter>` 同时统计 v2/legacy 活跃维度 | 同上 | 无需改动，legacy 分支自然沉默 |
| `tlabel/adapters/daimon_dataset.py:1271-1280` | `load()` 后按 `video_available` 把 6 个**旧键**改 false（`texture_energy/edge_density/shear_field_*/optical_flow_*`） | **是，会崩**：迁成 13 键后 `caps["texture_energy"]=False` 抛 KeyError | **必须同步改**：替换为对新键的条件控制（无视频时 `object_deformation/force_vector/slip_velocity` 降级），或直接让静态声明取保守值、删掉运行时改字典逻辑 |

除此之外：`tlabel/viewer/`、`tlabel/batch/`、`tlabel/scenarios/`、`tlabel/predict/`、`tlabel/augment/`、`tlabel/features_meta.py`、`tlabel/quality/` **均无任何代码读取 capabilities 键**（已 grep 确认）。场景模板 YAML 走自己的字段，不消费 adapter caps。

**结论：没有任何业务逻辑依赖 deformation_magnitude 等旧能力键做分支判断**；唯一会因删键而崩的是 daimon_dataset 自己的运行时降级代码。旧 22 键在帧级 tlabel_v2 dict 中继续存在（这是特征层，不在本次范围），不受影响。

### 4.3 validator 会不会因新键报错 / 旧文件怎么办

- 正向：schema 校验器 `tlabel/validate.py` 直接用 `jsonschema.Draft7Validator(schema)`（`:50`），**当前旧适配器导出文件实测即不通过**（复现：对 caps `{"contact":true,"deformation_magnitude":true}` 报 `[capabilities] Additional properties are not allowed`；含 compliance_level 同样报错）。迁移后由"全挂"变"通过"，不存在"新键报错"。
- `cli.py` 的内置校验是宽松自研逻辑（只 info 提示，不 error），不受影响。
- 旧文件读取：`tlabel_format.py:36-40` 原样透传旧 caps（§7）；`from_tlabel_v1` 继续吃旧帧格式。读旧文件不需要改。

---

## 5. 测试影响面清单

### 5.1 会失败、必须更新的测试

| 测试 | 现状断言 | 迁移后 | 处理 |
|---|---|---|---|
| `tests/integration/release_regression.py:92-102` `test_adapter_capabilities` | 断言 touchd caps **`len(caps)==22`** 且含 `contact_transition` | 必挂（13 键） | 改为断言 13 键集合合规 + contact/slip_event/force_vector（touchd 真值）为 true。注意该文件是脚本式（直接 `sys.exit`，pytest 收不到 test，实测 INTERNALERROR），当前不在 pytest 集内，但仍应改 |
| `tests/unit/test_edge_cases.py:506-512` `test_syntouch_capabilities_cover_all_schema_fields` | 遍历 **SCHEMA_V2_FIELD_NAMES（14 个，含 compliance_level）** 断言 syntouch caps 含全部 14 键 | 删 compliance_level 后必挂 | 改为遍历 13 个合法能力键（建议从 schema json 的 capabilities.properties 读取，与 xela 测试同款），并断言不含 compliance_level |
| `tests/unit/test_edge_cases.py:483-504` `test_all_adapters_have_capabilities` | 只断言 dict 非空，且**靠 TypeError 跳过 tlabel_format** | 不会挂，但掩盖问题 | 升级：tlabel_format 特殊处理（文件输入型）；paxini_gen3 的 `file_path=None` 怪签名修掉后纳入无参调用检查 |
| `tests/unit/test_edge_cases.py:539-566` `test_all_adapters_capabilities_key_count` | 只要求 contact/slip_event 两键存在（注释已明确 compliance_level 是元字段，口径其实正确） | 不会挂 | 升级为守护测试（见 §8 S4）：键集合 ⊆ schema 13 键 + contact 必在 + 值全 bool |

### 5.2 不会受影响但相关的测试

- `tests/test_adapter_xela.py:432-441`：已断言 xela caps 键集 == schema 键集、len==13——**直接抽成全适配器参数化版本**即可作为守护测试；`:643-670` 的逐值断言保持。
- `tests/unit/test_cli.py:47`：validate 用例里喂的 caps 是 `{"contact": true, "deformation_magnitude": true}` 旧样例——cli 宽松校验不报错，测试可继续过；建议同步换成 13 键样例（或保留一条 legacy 兼容样例）。
- `tests/unit/conformance/test_schema_validation.py:55,126`：样例 caps `{"contact": True}` 本就合规，无需动。
- `tests/unit/test_tlabel.py:195,265,288,577,...`、`tests/unit/test_v040.py`、`tests/unit/test_v050.py:234 等`：手工构造的 caps 任意（`{"contact":True}`、`{"dimensions":14}` 等），若新增**强制校验**（strict 模式）这些会挂——所以基类校验必须默认 warning/opt-in strict（§6），且这些历史样例分批清洗。
- `tests/test_v21_metadata_fields.py:237,254,268`、`tests/test_v18_detect_image_shape.py:73,90`：mock 适配器 `get_capabilities()` 返回 `{}`——强制校验会挂，同样依赖 warning 默认策略。
- `tests/unit/test_tacquad.py:144`：只 print 长度不断言，无影响。

### 5.3 建议新增的测试（S4 守护）

1. 参数化：对注册表每个内置适配器（tlabel 除外）断言：
   - `set(caps) ⊆ SCHEMA_CAP_KEYS`；`"contact" in caps`；`"compliance_level" not in caps`；全值 bool；
2. 端到端 schema 合规：每个能离线跑的适配器导出 `to_dict()` 后过 `jsonschema`（xela 已有此测试 `test_adapter_xela.py:425-430`，推广到全部，无数据/重依赖的 skip）；
3. daimon 无视频降级路径的 caps 测试（随 §4.2 必改项）。

---

## 6. 基类是否应强制约束

现状：`DataAdapterBase.get_capabilities`（`base.py:90-97`）与 `SensorAdapterBase.get_capabilities`（`:251-253`）都只是 `@abstractmethod`，无键集、无类型、无 contact 校验。这是 13/14 适配器能静默产出 schema 非法文件的根本原因。

### 推荐做法：模板方法 + 双层校验（warning 默认，strict 可开）

在 `tlabel/adapters/base.py` 增加：

```python
# 单一事实源：从 schema/tlabel-schema.json 的 capabilities.properties 读，
# 或在 core 里定义后让两边共用（测试也复用）
CAPABILITY_KEYS = frozenset({...13 keys...})

def validate_capabilities(caps: dict, *, strict: bool = False, source: str = ""):
    """返回问题列表；strict=True 时由调用方决定抛错。"""
```

并把抽象方法改为**模板方法模式**：
- 抽象钩子改名建议 `_declare_capabilities()`（子类返回原始声明）；
- 基类提供**具体** `get_capabilities()`：调用钩子 → 校验 → 不合规则 `warnings.warn(...)`（列出非法键/缺失 contact/非 bool 值）→ 返回**归一化后的副本**（剔除未知键、补 contact）。
- 兼容性折中：为避免一次改动所有子类签名，**v1 可不动抽象方法名**，改为在两个"出口"做校验+归一化：
  1. `SensorAdapterBase.collect()`（`base.py:373-382`）；
  2. 更推荐的统一出口——在 `TLabelData.__init__`（`types.py:209`）或 `to_dict()`（`types.py:656`）做一次 `validate_capabilities`，warning + 剔除非法键。这样所有适配器（含社区 entry_point 适配器、lerobot 转换器、手工构造）都被兜住，且不碰 ABC 签名。
- strict 开关：环境变量/模块配置 `TLABEL_STRICT_CAPS=1` 或 `validate(strict=True)`，开启后非法即 `ValueError`。CLI `validate` 子命令未来把未知键升级为 warning 级 ValidationResult（不破坏退出码语义）。

利：
- 脏文件无法静默产出（warning 至少可见，strict 直接拦截）；
- 不依赖贡献者自觉；C-15 有了执行机制；
- 出口归一化保证 `to_dict()` 永远 schema 合法（这是对外承诺的核心）；
- 对第三方适配器温和（默认不抛异常，符合"标准推广不强推"的既定原则）。

弊/代价：
- 若在 `to_dict()` 静默剔键，会出现"调用方传入的 caps 与输出不一致"的隐性行为——必须在 warning 里明确写出被剔除的键和来源，且文档化；
- strict 默认开启会打爆现有测试与外部脚本（§5.2 的手工 caps），所以只能 opt-in；
- tlabel_format 透传场景需要豁免/单独策略（§7），否则读旧文件再导出会被自动清洗——这其实是想要的行为，但要让用户可见（warning）。

**最终推荐：出口校验落在 `TLabelData`（构造时 warning，to_dict 时归一化）+ 基类提供静态校验工具 + 未来 1.0 把 strict 转默认。** 仅在基类加抽象方法但不碰出口，挡不住转换器/手工构造，不推荐单独使用。

---

## 7. `tlabel_format.py` 透传兼容策略

现状：`TLabelAdapter.get_capabilities(file_path)`（`tlabel_format.py:36-40`）直接返回文件里的 caps，原样进入 `TLabelData(capabilities=raw.get("capabilities", {}))`（`:104`）。读旧 22 键文件再导出，等于把脏 caps 洗白传播。

推荐策略（读宽容、写规范）：

1. **读取保持宽容**：不拒绝旧文件（旧文件是历史事实，且帧层 tlabel_v2 本来就要支持读）。
2. **加载时清洗 + 报警**：在 `load()` 组装 TLabelData 时调用 §6 的校验/归一化：
   - 13 键内的：原样保留；
   - `compliance_level` 键：剔除（它属于帧 schema_v2，顶层 caps 不允许）；
   - 旧 22 键：从 caps 剔除，但**不丢信息**——把原始 caps 存入 `episode_info["legacy_capabilities"]`（或 sensor_specific）并 `warnings.warn("legacy capability keys ... dropped from capabilities; preserved in episode_info.legacy_capabilities")`；
   - 完全无法识别的键：同样剔除+warning。
3. **签名问题一并修**：`get_capabilities(file_path)` 违反基类无参契约（paxini_gen3 的 `file_path=None` 同理）。方案：TLabelAdapter 是"无数据源就无能力"的特殊适配器，建议
   - 保留按文件读取的能力为 `get_capabilities_for_file(path)` 静态/显式方法；
   - 无参 `get_capabilities()` 返回保守空声明 `{"contact": False}` 或返回文件最近一次 load 的归一化 caps（load 后缓存到实例）；
   - 这样遍历型测试不必再靠 TypeError 跳过（`test_edge_cases.py:494-498` 的跳过分支可删）。
4. **round-trip 原则**：旧文件 → load → export 得到的新 JSON 必须 schema 合规；若用户需要原样保留旧 caps，只能拿原始文件，不走本库导出（在 warning/MIGRATION.md 说明）。

---

## 8. 分阶段实施步骤（文件级、工作量、breaking）

### S0：基类/出口校验（只 warning，不改变输出）—— 0.23.3 patch，约半天

- 新增 `tlabel/core/capabilities.py`（或放 `core/schema.py`）：`CAPABILITY_KEYS`（13 键，单一事实源，测试/schema 共用）、`validate_capabilities()`；
- `tlabel/core/types.py:209` `TLabelData.__init__`：调用校验，违规则 `warnings.warn`（**不剔键、不抛错**）；
- `tlabel/cli.py:166-180`：保留 legacy 统计，新增未知键 warning 级结果（可选，做就放这步）；
- 测试：新增 warning 触发用例（用现有脏适配器断言 warning 出现），不动任何适配器。
- Breaking：无；CHANGELOG：Added 段记一条"capabilities conformance warnings"。

### S1：删除 3 个 `compliance_level` 键 + 修 demo/lerobot 两个脏源 —— 0.23.3 patch 或 0.24.0 开头，约半天

- `tlabel/adapters/syntouch.py:479` 删键（顺带修正 `:464` 注释口径）；
- `tlabel/adapters/tashan_ts_f_a.py:89` 删键；
- `tlabel/adapters/paxini_px6d.py:65` 删键，修 `:49` 注释（14→13）；
- `tlabel/demo.py:417` 删 compliance_level；
- `tlabel/converters/lerobot.py:120` 改为 13 键白名单交集；
- 测试：`tests/unit/test_edge_cases.py:506-512` 改 13 键口径（这一步必须同 PR，否则红）；其余测试不用动。
- Breaking：**文件格式层面是修复**（这些文件原本就过不了 schema），但若有人外部代码在读 caps["compliance_level"]，属破坏性——概率极低，CHANGELOG 记 Fixed；不需要 Breaking Changes 段（输出从非法变合法）。

### S2：迁移 10 个旧 22 键适配器（按传感器类分 2-3 批，每批 1 个 PR）—— 0.24.0，每批约一天

按 §3.1 总表替换 get_capabilities 字典（仿 `xela.py:199-220` 注释风格，每个键注释数据来源与诚实限制）：

- **批次 A（视觉触觉 GelSight 系，约一天）**：`gelsight.py:313`、`ycb_slide.py:292`、`vtouch.py:362`、`tacquad.py:402`；
- **批次 B（力/阵列与数据集，约一天）**：`paxini_dataset.py:102`、`paxini_gen3.py:188`（同时修签名去掉未使用的 file_path 参数）、`daimon_dataset.py:884` + **必改** `:1271-1280` 运行时降级逻辑（替换旧键名为新键名）、`daimon_dm_tac.py:62`；
- **批次 C（富语义/多传感器，约半天）**：`touchd.py:420`、`univtac.py:108`；
- 同批更新 `tests/integration/release_regression.py:92-102`（22→13 断言）；
- §3.1 中标注的语义争议项（paxini object_deformation 代理、tacquad L1 自评、force_vector 条件能力、manipulation_phase 口径、ycb confidence）在开工前用半小时与罗西对齐一次，结论写进本文件或 ADR。
- 同期登记但不阻塞的债：
  - **R2（推荐与 S2 同期或紧随）**：统一 load 路径——让 §3.2 列出的 9 处 `from_tlabel_v1` 改走各适配器 extract_schema（或抽公共的"诚实帧构建"helper），解决 force_vector 假值、compliance_level 恒 L1、slip_velocity 丢失、schema_v2.manipulation_phase 恒 null。工作量约 1.5-2 天，风险中等（帧数值会变，回归测试需重跑/重录 golden）。
  - R3：tashan slip_event=None 违反 bool required；univtac phase 枚举外值；daimon_dm_tac extract 的 contact_region="palmar" 死值——各开 issue。
- Breaking：**是**。caps 的键空间变了，且若 S0 只 warning，S2 落地后 warning 消失、输出键集变化。外部代码若消费旧 caps 键会失效（全仓内部零消费，外部概率低）。CHANGELOG **需要 Changed/Breaking Changes 段**，并在 `MIGRATION.md` 加迁移说明（"capabilities 旧 22 键已移除；旧特征仍在帧级/raw 层"）。

### S3：tlabel_format 清洗 + 守护测试 + strict 开关 —— 0.24.0 收尾，约半天~一天

- `tlabel_format.py` 按 §7 实现：归一化 + warning + legacy 保留到 episode_info + 修 get_capabilities 签名；
- 基类/`TLabelData` 出口归一化：把 S0 的 warning-only 升级为"to_dict 时剔除非法键 + warning"（strict 环境变量直接抛错）；
- 落地 §5.3 三个守护测试（参数化全适配器 caps 合规、全适配器 to_dict 过 jsonschema）；
- 删除 `test_edge_cases.py:494-498` 对 TypeError 跳过的豁免分支；
- `cli.py` validate 对 caps 未知键给 warning（非 error，不改变旧文件校验的退出码）；
- 文档：更新 `docs/tlabel-format.md` capabilities 段（D-13 已登记其缺规范键集）、`docs/annotation-spec.md`、`docs/spec/tlabel-spec-v1.md` §11 的 D-17/D-18 标记为 resolved（C-15 已有执行机制）。
- Breaking：to_dict 剔键对依赖脏键透传的调用方是行为变化（但本来就非法），记 Breaking Changes；strict 默认仍关闭，不破坏。

工作量汇总：S0 半天 + S1 半天 + S2 两天半（含 R2 则 4-5 天）+ S3 一天 ≈ **不含 R2 约 4 人日；含 load 路径对齐约 6 人日**。

---

## 9. 版本节奏建议

**建议：S0/S1 进 0.23.x patch（0.23.3），S2/S3 进 0.24.0 minor。**

理由：
1. S0 纯增量 warning、S1 是把"schema 非法输出"修成合法输出——不改变任何已承诺的合法行为，patch 语义安全；还能尽快止住新增脏文件，并为 11-27 的 JOSS 窗口消除一个显眼的 conformance 硬伤（spec §11 D-17/D-18 白纸黑字登记着 13 个适配器不合规，reviewer 一对照代码就能看到）。
2. S2 改 10 个适配器的公开字典键空间、S3 改 to_dict 行为，属于键集层面的 breaking，按 SemVer 必须放 minor（0.x 阶段 minor 即允许 breaking），配 MIGRATION.md 与 CHANGELOG Breaking Changes。
3. 与既定 backlog v6 中 0.24.0 "JOSS 就绪版 / 技术债修复 / issue 收口"的定位完全一致，不需要单独开版本。
4. 不建议把 S2 也塞进 patch：patch 改公开键空间违反发版规范，且外部 PR 生态（5 个待合外部 PR）若基于旧 caps 写代码会产生隐性冲突；集中在 0.24.0 一次性迁移 + 守护测试，对外沟通成本最低。
5. R2（load 接 extract_schema）会改变帧内数值/合规级别，风险最高，建议**不阻塞 capabilities 迁移**，在 0.24.0 周期内单独 PR 推进；若人力不足，0.24.0 至少保证"caps 声明按 load 实际产出的保守口径"（§3.1 中 tacquad force=false、daimon/touchd force_vector 取保守值的二选一），避免声明与文件冲突。

---

## 10. 附：关键证据索引

- Schema 键集与禁令：`schema/tlabel-schema.json` capabilities.properties（13 键 / additionalProperties:false / required contact）
- 规范条款：`docs/spec/tlabel-spec-v1.md:524-531`（S-6.3）、`:585-596`（S-7.4）、`:686`（C-15）、`:719-720`（D-17/D-18）
- 合规范本：`tlabel/adapters/xela.py:199-220`；守护测试 `tests/test_adapter_xela.py:432-441`
- 脏数据出口：`tlabel/core/types.py:209-217,656`；`tlabel/export/writer.py:296`；`tlabel/adapters/base.py:373-382`
- v1→v2 映射（含假 y、假 Fy、恒 L1）：`tlabel/core/schema.py:187-292`
- load 未接 extract_schema 的 10 处：见 §3.2 行号清单
- 唯一消费旧 caps 键的业务代码：`tlabel/adapters/daimon_dataset.py:1271-1280`
- 宽松/严格两种 validator：`tlabel/cli.py:166-180`（info）vs `tlabel/validate.py:50`（jsonschema，旧文件实测不过）
- 适配器外的另外 3 个脏源：`tlabel/demo.py:417`、`tlabel/converters/lerobot.py:120`、`examples/generate_demo_data.py:92-140`（示例脚本，旧 0.4 格式，可不随库迁移）
- 注册口径：注册表 15 项（含 tlabel 透传型），`tlabel/core/registry.py:24-38,89`
