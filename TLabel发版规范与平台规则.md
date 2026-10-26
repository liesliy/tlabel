# TLabel 发版规范与平台规则

> 适用范围：TLabel（PyPI 包名 `tlabel`，GitHub 仓库 `liesliy/tlabel`）的每一次版本发布。
> 参考基线：`rda/governance/DEV_PROTOCOL.md` §3.3 发布流程。
> 最后更新：2026-10-26（v0.23.2 发版时建立）

---

## 一、铁律

**发版流程禁止跳步。** 任何一步没做完就进入下一步，都算发版失败。

发版是不可逆操作：PyPI 上的版本号一旦上传就永久占用，删不掉、改不了，只能 yank（下架但版本号仍被占用）。所以宁可多花十分钟检查，也不要发了再补救。

---

## 二、发版流程（七步）

版本号三处对齐 → CHANGELOG → 全量测试 → 打包验证 → GitHub 推送 → Tag + Release → PyPI 构建上传 → pip install 验证

### 第 0 步：发版前先查 PyPI 现状

**禁止凭本地文件猜版本号。** 每次发版前先拉 PyPI 的已发布版本列表：

```bash
curl -s https://pypi.org/pypi/tlabel/json | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('latest:', d['info']['version'])
print('已发布版本数:', len([v for v,f in d['releases'].items() if f]))
"
```

确认两件事：
1. 你准备发的版本号**不在**这个列表里（在就直接失败，见第六节）
2. 本地代码库的版本号与 PyPI 最新版的关系符合预期（是连续递增，不是回退）

### 第 1 步：版本号三处对齐

TLabel 的版本号写在**三个**地方，必须完全一致：

| 文件 | 字段 |
|---|---|
| `VERSION` | 纯文本，一行版本号 |
| `tlabel/_version.py` | `__version__` 和 `version` 两个变量 |
| `pyproject.toml` | `[project] version` |

```bash
cat VERSION
grep __version__ tlabel/_version.py
grep -m1 '^version' pyproject.toml
```

三处必须输出同一个号。历史上出过 `VERSION` 停在 `0.23.0`、另外两处是 `0.23.1` 的漂移，这种不一致会让 sdist 元数据和运行时 `tlabel.__version__` 对不上。

### 第 2 步：CHANGELOG

在 `CHANGELOG.md` 里把 `## [Unreleased]` 改成 `## [X.Y.Z] - YYYY-MM-DD`，并确保内容完整。

要求：
- 分类用 Keep a Changelog 标准段落：`Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`
- 破坏性变更必须单独开 `### Breaking Changes` 段落，写清"谁受影响、怎么迁移"
- 只写**真正发布出去**的内容。如果某个功能开发在未合并分支上、没进这个包，就不能写进来（历史上 v0.19.0 就踩过这个坑，详见第五节）

### 第 3 步：全量测试

```bash
cd 牛宿科技/tlabel-repo
python3 -m pytest tests/ -q
```

通过标准：**0 failed**，skipped 数量与上一版一致（允许因环境缺可选依赖而 skip，但不允许新增 skip）。

发版记录里要留下实际的 `X passed, Y skipped` 数字。

### 第 4 步：打包验证

**这一步最容易被跳过，也最容易出事。**

```bash
rm -rf /tmp/tlabel-dist
python3 -m build --outdir /tmp/tlabel-dist
twine check /tmp/tlabel-dist/*
```

然后必须**逐个检查非 .py 资源有没有真的进包**：

```bash
# wheel 里该有的资源
python3 -c "
import zipfile, glob
f = glob.glob('/tmp/tlabel-dist/*.whl')[0]
names = zipfile.ZipFile(f).namelist()
print('YAML:', len([n for n in names if n.endswith('.yaml')]))
print('demo_data:', len([n for n in names if 'demo_data' in n]))
"

# sdist 里该有的文档
python3 -c "
import tarfile, glob
f = glob.glob('/tmp/tlabel-dist/*.tar.gz')[0]
names = tarfile.open(f).getnames()
for kw in ['VERSION', 'docs/spec/tlabel-spec-v1.md', 'MANIFEST.in']:
    print('OK ' if any(n.endswith(kw) for n in names) else 'MISS', kw)
"
```

当前必须进 wheel 的资源：`tlabel/scenarios/*.yaml`（4 个）、`tlabel/demo_data/*.json`
当前必须进 sdist 的资源：上面全部 + `VERSION` + `docs/*.md`（含规范正本）

任何一项 MISS 都要先修 `MANIFEST.in` / `[tool.setuptools.package-data]`，不能带着缺口发。

### 第 5 步：GitHub 推送

```bash
git add -A
git commit -m "chore(release): vX.Y.Z - <一句话摘要>"
git push origin main
```

推送前确认：
- `git status --short` 干净（除了故意的未跟踪文件）
- 没有凭证泄露：`git diff --cached | grep -iE "ghp_|pypi-|api[_-]?key|password"` 应为空

### 第 6 步：Tag + GitHub Release

Tag 命名规范：`vX.Y.Z`，用 annotated tag（不是 lightweight）。

```bash
git tag -a vX.Y.Z -m "vX.Y.Z - <摘要>"
git push origin vX.Y.Z
```

Tag 必须与 PyPI 版本**一一对应**。发一个 PyPI 版本就必须有一个 tag，反过来 tag 也不该指向没发布的版本。

GitHub Release 用 API 创建（CLI 环境没有 gh）：

```bash
python3 - <<'PY'
import json, urllib.request
TOKEN = open('/Coze/Drive/扣龙虾/所有对话/主对话/codeact/output/.github_token').read().strip()
body = {
    "tag_name": "vX.Y.Z",
    "name": "vX.Y.Z - <标题>",
    "body": "<从 CHANGELOG 摘对应段落>",
    "draft": False, "prerelease": False,
}
req = urllib.request.Request(
    "https://api.github.com/repos/liesliy/tlabel/releases",
    data=json.dumps(body).encode(),
    headers={"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json",
             "User-Agent": "tlabel-release", "Content-Type": "application/json"},
    method="POST")
print(urllib.request.urlopen(req).status)
PY
```

### 第 7 步：PyPI 上传

**先确认凭证还没过期。** 凭证在 `/root/.pypirc`（`__token__` 格式）。PyPI token 会过期，发版前先确认它还在、还能用。

```bash
twine upload /tmp/tlabel-dist/*
```

上传成功后立刻验证：

```bash
# PyPI JSON API 应该能看到新版本
curl -s https://pypi.org/pypi/tlabel/json | python3 -c "
import sys, json; print(json.load(sys.stdin)['info']['version'])
"
```

### 第 8 步：干净环境安装验证

**不要在自己已经装过 tlabel 的环境里验证。** 装到独立目录：

```bash
rm -rf /tmp/tlabel-verify
pip install --no-deps --target /tmp/tlabel-verify tlabel==X.Y.Z
PYTHONPATH=/tmp/tlabel-verify python3 -c "import tlabel; print(tlabel.__version__)"
PYTHONPATH=/tmp/tlabel-verify python3 -m tlabel.cli template list
PYTHONPATH=/tmp/tlabel-verify python3 -m tlabel.cli version
```

要求：版本号打印正确、CLI 子命令都能跑、非 .py 资源能读到（`template list` 能列出 4 个场景就说明 YAML 打包成功）。

---

## 三、版本号规则

### 递增规则

| 变更类型 | 版本号变化 | 例子 |
|---|---|---|
| 新增用户可见功能 | MINOR +1 | 0.22.4 → 0.23.0（XELA 适配器） |
| 修复 bug / 文档 / 打包 | PATCH +1 | 0.23.1 → 0.23.2 |
| 破坏性变更（公开字段、API 签名、格式） | MINOR +1，且 CHANGELOG 必须有 `Breaking Changes` 段 | capabilities 键集变更 |

项目处于 0.x 阶段，SemVer 允许一定弹性，但**必须保证单调递增、不重号、不回退**。

### 禁止事项

- **禁止复用已发布过的版本号。** PyPI 不允许覆盖，重发会直接 400 `File already exists`
- **禁止跳过 CHANGELOG 直接发。** 跳号、缺条目会让发布记录不可追溯（历史上 0.22.1/0.22.2/0.22.3 三连发就漏了条目）
- **禁止从未合并进 `main` 的分支发 PyPI 包。** 这是最严重的一类事故，见第五节

---

## 四、凭证与平台规则

| 用途 | 位置 | 注意 |
|---|---|---|
| PyPI 上传 | `/root/.pypirc`，`__token__` 格式 | 会过期，发版前确认；不要在仓库里存 |
| GitHub API | `codeact/output/.github_token` | 约 1 个月有效期；权限为 `liesliy` 账号 |
| git push 认证 | 仓库 `.git/config` 的 `credential.helper` | 从上面那个 token 文件读，**禁止把 token 明文写进 remote URL** |

### git push 超时排查

如果 `git push` 报 `Failed to connect to github.com port 443 after 130xxx ms: Connection timed out` 或 `GnuTLS recv error (-110)`：

```bash
git config --get http.postBuffer
```

若输出 `524288000`（524MB），**这就是元凶** —— 过大的 postBuffer 会导致连接阶段就超时。处理：

```bash
git config --unset http.postBuffer
# 或临时：git -c http.postBuffer=1048576 push origin main
```

同时清理 repo-local 的 `http.lowSpeedLimit` / `http.lowSpeedTime` / `http.version`，这些会让连接卡在低速等待。

注意：`git ls-remote` 能通但 `git push` 超时，基本可以确定是 postBuffer 问题，不是网络问题。

---

## 五、历史异常记录（务必了解）

发版时如果发现"文档说发过、代码里却没有"，先查这张表。

### 5.1 v0.19.0 —— 从未合并分支发布，能力已丢失

- 0.19.0 的代码提交在 `feat/v0.19.0-convert-cli` 分支（commit `e3867ab`），**从未合并进 `main`**
- `main` 在 2026-08-03（v0.18.2）到 2026-08-25（v0.20.0）之间零提交，直接跳版
- 结果：0.19.0 wheel 里的 `converters/base.py` 和 `lerobot._safe_float()` 修复，**在 0.20.0 及之后的版本里都不存在**
- 教训：**发布前必须确认代码在 `main` 上**（`git merge-base --is-ancestor HEAD main`），不能从工作分支直接 build

### 5.2 CHANGELOG 曾虚报 v0.19.0 内容

- 原 CHANGELOG 声称 0.19.0 新增了 `tlabel convert` / `batch-convert` / `list-adapters` / `adapter-info` 四个 CLI 命令和 28 个测试
- 实测 0.19.0 wheel 的 `cli.py` 只有 `info` / `list` / `validate` / `version`
- 已于 2026-10-26 更正并加 `### Notes` 说明

### 5.3 v0.22.1–0.22.3 同日三连发，CHANGELOG 缺条目

- 2026-09-03 当天连发 0.22.1 / 0.22.2 / 0.22.3，但 CHANGELOG 从未写这三个版本
- 已于 2026-10-26 补记
- 教训：CHANGELOG 必须在**上传之前**写，不能事后补

### 5.4 历史 tag 缺口

PyPI 上有 62 个版本，GitHub tag 原本只有 46 个。2026-10-26 已补近期 6 个：

| 版本 | commit | 说明 |
|---|---|---|
| v0.19.0 | `e3867ab` | 指向未合并分支，main 无对应代码 |
| v0.22.0 | `a92e40f` | |
| v0.22.1 | `6dff78e` | |
| v0.22.2 | `72371a6` | |
| v0.22.3 | `a54d8b8` | |
| v0.22.4 | `13cfa32` | |

早期 11 个版本（`0.2.0a8`、`0.3.0`、`0.6.0`、`0.10.x`、`0.12.0`、`0.12.1`、`0.13.1`、`0.14.0`）**不补 tag**：年代久远、commit 归属无法可靠考证，硬打 tag 反而制造错误信息。这些版本以 PyPI 记录为准。

---

## 六、发版后核对清单

发完必须逐项确认，任何一项不通过都要立刻处理：

- [ ] PyPI `info.version` 等于新版本号
- [ ] PyPI 上有 sdist + wheel 两个文件
- [ ] GitHub `main` 最新 commit 与本地 HEAD 一致
- [ ] GitHub 有 `vX.Y.Z` tag，且指向正确的 commit
- [ ] GitHub Release 已创建且正文与 CHANGELOG 一致
- [ ] 干净环境 `pip install` 后 `tlabel.__version__` 正确
- [ ] 干净环境 `tlabel template list` 能列出场景（验证非 .py 资源）
- [ ] CHANGELOG 顶部已无遗留的 `## [Unreleased]` 空段（除非有新的未发布内容）
- [ ] 本地 `git status` 干净

---

## 七、发版节奏建议

- **JOSS 审稿看"持续开发"证据**，发布节奏是重要信号。但不要为了刷版本号而发空包
- 一个版本只承载一组相关变更，不要把无关的修复攒在一起
- 破坏性变更尽量集中到一个 MINOR 版本里，避免连续多个版本都在破坏兼容性
- 每个版本发完，把「版本号 / 日期 / 主要变更 / 测试结果」记进 `recent_memory/project/`，方便后续追溯
