# XELA uSkin 样例数据（合成）

本目录的样例数据为**程序生成的合成数据**，不包含 UniTac-NV 数据集的任何原始录制，
无许可限制。

| 文件 | 说明 |
|---|---|
| `sample_univtac_nv_style.csv` | 合成的 UniTac-NV 风格 XELA uSkin CSV（30 帧，4×6×3 taxel 力矩阵） |
| `sample_export.json` | 上表经 `XelaUskinAdapter` 加载后导出的 tlabel JSON |

## 格式说明

CSV 格式模仿 [UniTac-NV 数据集](https://github.com/JiannnH/UniTac-NV)（IROS 2025）
中 XELA uSkin 传感器的录制格式：

- 每行 6 列：`time`（`YYYY-MM-DD HH:MM:SS.ffffff`）、`seq`、
  `sensor_matrices_force`、`sensor_matrices_displacement`、`ft`、`end_effector_pose`
- 力/位移矩阵单元格为 Python 字面量嵌套列表，形状 4×6×3（24 taxel × 3 轴）
- 采样率 100 Hz（论文标称）
- 位移矩阵恒为 0——如实保留该数据集的实测特征（`dataset_info.json` 统计显示
  XELA 的 displacement 通道 mean=min=max=0）

## 复现命令

```bash
# 加载样例并查看
python -c "
from tlabel.core.registry import get_adapter, list_builtin_adapters
list_builtin_adapters()
data = get_adapter('xela')().load('tests/data/xela/sample_univtac_nv_style.csv')
print(data.num_frames, 'frames,', data.episode_info['contact_frames'], 'contact frames')
"

# schema 校验（PR 模板的 validate 步骤）
tlabel validate tests/data/xela/sample_export.json
```
