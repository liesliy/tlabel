# 本地环境快照

> 由 `scripts/check_env.py` 自动生成，**请勿手工编辑**。
> 用途：让云端 Agent 看到本地仓库的真实状态。

| 项 | 值 |
|---|---|
| 生成时间 | 2026-11-12 17:32:45 |
| 主机 | VM-1-203-ubuntu |
| 系统 | Linux 5.15.0-100-generic |
| Python | 3.13.12 |
| 仓库路径 | `/Coze/Drive/扣龙虾/所有对话/主对话/牛宿科技/tlabel-repo` |
| 快照落盘位置 | `/Coze/Drive/扣龙虾/所有对话/主对话/牛宿科技/tlabel-repo/docs` |

## 仓库状态

| 项 | 值 |
|---|---|
| remote | `https://github.com/liesliy/tlabel.git` |
| 分支 | `main` |
| commit | `50bb021` |
| commit 时间 | 2026-10-30 14:21 |
| commit 说明 | fix: demo capabilities must match actual output; validate accepts own format tag |
| VERSION | **0.24.2** |
| 与 origin 比较 | 领先 **0** / 落后 **0** |

## 未提交改动

工作区干净，无未提交改动。

## 适配器

来源：import tlabel

**共 15 个**：

`gelsight`, `paxini`, `daimon`, `tlabel`, `touchd`, `univtac`, `vtouch`, `ycb_slide`, `tacquad`, `syntouch`, `xela`, `tashan_ts_f_a`, `paxini_gen3`, `daimon_dm_tac`, `paxini_px6d`

> 基线核对：v0.24.2 应有 **12 个数据集适配器 + 2 个传感器适配器 + 1 个占位符 = 15 个键**。
> 如果数量对不上，说明本地代码与云端仓库不一致。

## 关键文件指纹（SHA256 前 16 位）

| 文件 | 存在 | 大小 (B) | SHA256-16 |
|---|---|---|---|
| `VERSION` | ✓ | 7 | `35cb6510464522cc` |
| `pyproject.toml` | ✓ | 1,918 | `3780d51b7f875f74` |
| `tlabel/__init__.py` | ✓ | 2,651 | `2d803c19934aa5fa` |
| `tlabel/adapters/__init__.py` | ✓ | 2,157 | `651cf5ef7cfe42c6` |
| `tlabel/adapters/base.py` | ✓ | 14,408 | `5488cd77a90d1039` |
| `tlabel/adapters/daimon_dataset.py` | ✓ | 52,590 | `b0890756ba415330` |
| `tlabel/adapters/xela.py` | ✓ | 28,547 | `a7d9226a52b87be6` |
| `tlabel/adapters/tashan_ts_f_a.py` | ✓ | 8,873 | `47a9095bdc89ca16` |
| `tlabel/core/schema.py` | ✓ | 18,157 | `74f98dc203347256` |

> 用途：云端比对同一文件的指纹，若不同则说明本地代码被手改过，或版本不一致。

## 依赖版本

| 包 | 版本 |
|---|---|
| numpy | 2.5.2 |
| pyarrow | 23.0.1 |
| cv2 | 4.13.0 |
| scipy | 1.17.1 |
| pandas | 2.3.3 |

> SharpaDex 适配需要 `pyarrow` + `opencv-python` + `scipy`，以及系统级 `ffmpeg`。

## 自检结论

- ✅ remote 指向 liesliy/tlabel
- ✅ VERSION 为 0.24.x 或更高
- ✅ 适配器数 = 15 键
- ✅ 工作区干净

如果有 ❌，先解决再开工。具体见 `docs/ROADMAP_industrial_datasets.md` 第 7 节。
