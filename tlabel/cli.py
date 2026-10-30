"""
TLabel CLI — 命令行工具

提供以下命令:
  tlabel validate <path>    校验数据文件是否符合 tlabel_v2 schema
  tlabel list               列出所有已注册的适配器
  tlabel info <adapter>     查看适配器详细信息
  tlabel version            显示版本号
  tlabel template list      列出所有场景模板
  tlabel template validate  校验场景模板（id 或 --all）

v0.17.0 更新: validate 命令支持 14 维 Schema V2 校验，向后兼容 22 维旧格式
v0.24.0 更新: 新增 template 命令（场景模板 list / validate）
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

from tlabel.core.schema import SCHEMA_V2_FIELD_NAMES


# 14 维 Schema V2 特征空间字段列表
SCHEMA_V2_DIMENSIONS = list(SCHEMA_V2_FIELD_NAMES)

#: validate 接受的 format 标记。``tlabel_v2`` 是旧 22 维格式的标记，
#: ``tlabel_schema_v2`` 是 Schema V2 导出器（core/types.py 的 to_dict）写入的标记。
#: 两者都是本工具产出的合法格式，校验时不能只认其中一个。
_ACCEPTED_FORMAT_TAGS = ("tlabel_v2", "tlabel_schema_v2")

# 旧 22 维 tlabel_v2 特征空间（向后兼容）
LEGACY_V2_DIMENSIONS = [
    "contact",
    "deformation_magnitude",
    "force_magnitude",  # deprecated, alias of deformation_magnitude
    "force_peak",
    "force_direction",
    "slip_entropy",
    "slip_event",
    "texture_energy",
    "edge_density",
    "contact_area",
    "centroid_x",
    "normal_field_magnitude",
    "normal_field_variance",
    "shear_field_magnitude",
    "shear_field_direction",
    "delta_force_normal",
    "delta_force_shear",
    "friction_cone_ratio",
    "optical_flow_magnitude",
    "optical_flow_direction",
    "temporal_deformation_rate",
    "contact_transition",
]

# 保持旧名兼容
TLABEL_V2_DIMENSIONS = LEGACY_V2_DIMENSIONS


class ValidationResult:
    """单条校验结果"""
    def __init__(self, level: str, message: str, field: str = ""):
        self.level = level  # "error", "warning", "info"
        self.message = message
        self.field = field

    def __repr__(self):
        prefix = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(self.level, "?")
        field_str = f" [{self.field}]" if self.field else ""
        return f"{prefix}{field_str} {self.message}"


def validate_tlabel_file(file_path: str) -> List[ValidationResult]:
    """校验单个文件是否符合 tlabel_v2 schema"""
    results = []
    path = Path(file_path)

    if not path.exists():
        results.append(ValidationResult("error", f"文件不存在: {file_path}"))
        return results

    # 1. 尝试作为 JSON 格式校验
    if path.suffix == ".json":
        results.extend(_validate_json_file(path))
    elif path.suffix in (".h5", ".hdf5"):
        results.extend(_validate_hdf5_file(path))
    elif path.suffix == ".parquet":
        results.extend(_validate_parquet_file(path))
    elif path.is_dir():
        results.extend(_validate_directory(path))
    else:
        results.append(ValidationResult("warning",
            f"不支持的文件格式: {path.suffix}，尝试自动检测..."))
        results.extend(_validate_auto_detect(path))

    return results


def _validate_json_file(path: Path) -> List[ValidationResult]:
    """校验 JSON 格式的 tlabel 文件"""
    results = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        results.append(ValidationResult("error", f"JSON 解析失败: {e}"))
        return results

    # 检查顶层字段
    if "schema_version" not in data:
        results.append(ValidationResult("error", "缺少 schema_version 字段", "schema_version"))
    else:
        sv = data["schema_version"]
        if not isinstance(sv, str) or not sv.replace(".", "").replace("-", "").isalnum():
            results.append(ValidationResult("error", f"schema_version 格式异常: {sv}", "schema_version"))

    if "format" not in data:
        results.append(ValidationResult("error", "缺少 format 字段", "format"))
    elif data["format"] not in _ACCEPTED_FORMAT_TAGS:
        # v0.24.2: 原先只认 'tlabel_v2'，而导出器写的是 'tlabel_schema_v2'，
        # 导致 validate 对本工具自己导出的文件报警告。
        expected = " 或 ".join(f"'{t}'" for t in _ACCEPTED_FORMAT_TAGS)
        results.append(ValidationResult("warning",
            f"format 为 '{data['format']}'，预期 {expected}", "format"))

    # 检查 frames
    if "frames" not in data:
        results.append(ValidationResult("error", "缺少 frames 字段", "frames"))
    elif not isinstance(data["frames"], list):
        results.append(ValidationResult("error", "frames 应为数组", "frames"))
    else:
        frames = data["frames"]
        results.append(ValidationResult("info", f"共 {len(frames)} 帧"))

        if len(frames) == 0:
            results.append(ValidationResult("warning", "frames 为空数组", "frames"))
        else:
            # 校验第一帧的维度数据
            frame0 = frames[0]
            if "schema_v2" in frame0:
                # v0.17+: Schema V2 格式（14维）
                results.extend(_validate_schema_v2_dict(frame0["schema_v2"], "frames[0]"))
            elif "tlabel_v2" in frame0:
                # 旧格式（22维），向后兼容
                results.extend(_validate_tlabel_v2_dict(frame0["tlabel_v2"], "frames[0]"))
            else:
                results.append(ValidationResult("error", "帧缺少 schema_v2 或 tlabel_v2 字段", "frames[0]"))

            # 抽样校验（最多检查 10 帧）
            sample_indices = [0]
            if len(frames) > 10:
                step = max(1, len(frames) // 10)
                sample_indices = list(range(0, len(frames), step))
            
            error_count = 0
            for idx in sample_indices[1:]:
                frame = frames[idx]
                if "schema_v2" in frame:
                    errs = _validate_schema_v2_dict(frame["schema_v2"], f"frames[{idx}]")
                elif "tlabel_v2" in frame:
                    errs = _validate_tlabel_v2_dict(frame["tlabel_v2"], f"frames[{idx}]")
                else:
                    error_count += 1
                    continue
                error_count += sum(1 for e in errs if e.level == "error")
            
            if error_count > 0:
                results.append(ValidationResult("warning",
                    f"抽样 {len(sample_indices)} 帧中有 {error_count} 帧存在维度错误"))

    # 检查 capabilities（可选但推荐）
    if "capabilities" in data:
        caps = data["capabilities"]
        if isinstance(caps, dict):
            from tlabel.core.capabilities import CAPABILITY_KEYS, validate_capabilities

            # caps 只认 13 个合法键（compliance_level 是帧级元字段，不算能力）。
            declared = [k for k in caps if k in CAPABILITY_KEYS and caps[k]]
            results.append(ValidationResult(
                "info",
                f"capabilities 声明 {len(declared)}/{len(CAPABILITY_KEYS)} 个合法能力键",
            ))
            # 非法键只报 warning，不改变退出码——旧文件仍要能校验通过。
            for problem in validate_capabilities(caps):
                results.append(ValidationResult("warning", problem, "capabilities"))

    # 检查 sensor 信息
    if "sensor" not in data:
        results.append(ValidationResult("warning", "缺少 sensor 字段（推荐提供）", "sensor"))

    return results


def _validate_tlabel_v2_dict(tlabel_v2: Dict, prefix: str) -> List[ValidationResult]:
    """
    校验单个 tlabel_v2 字典的维度完整性（旧 22 维格式）。
    
    自动检测：如果 dict 包含 Schema V2 特征字段（如 contact_centroid, compliance_level），
    则使用 14 维 Schema V2 校验；否则使用旧 22 维校验。
    """
    results = []
    
    if not isinstance(tlabel_v2, dict):
        results.append(ValidationResult("error", f"{prefix} 不是字典", f"{prefix}"))
        return results

    # 自动检测 Schema 版本
    v2_indicators = ["contact_centroid", "contact_region", "compliance_level",
                     "force_vector", "torque_vector", "slip_velocity",
                     "manipulation_phase", "texture_class", "object_deformation", "temperature"]
    is_schema_v2 = any(k in tlabel_v2 for k in v2_indicators)

    if is_schema_v2:
        return _validate_schema_v2_dict(tlabel_v2, prefix)

    # --- 旧 22 维校验 ---
    # 检查核心维度是否存在
    core_dims = ["contact", "deformation_magnitude"]
    for dim in core_dims:
        if dim not in tlabel_v2:
            results.append(ValidationResult("error",
                f"缺少核心维度 '{dim}'", f"{prefix}.{dim}"))

    # 检查维度值类型
    for key, value in tlabel_v2.items():
        if key in LEGACY_V2_DIMENSIONS or key.startswith(tuple(LEGACY_V2_DIMENSIONS)):
            if value is not None and not isinstance(value, (int, float)):
                results.append(ValidationResult("warning",
                    f"维度 '{key}' 值类型异常: {type(value).__name__}（应为 float）",
                    f"{prefix}.{key}"))

    # 统计覆盖维度数
    covered = sum(1 for d in LEGACY_V2_DIMENSIONS if d in tlabel_v2)
    total = len(LEGACY_V2_DIMENSIONS)
    if covered < total:
        missing = [d for d in LEGACY_V2_DIMENSIONS if d not in tlabel_v2]
        results.append(ValidationResult("info",
            f"覆盖 {covered}/{total} 维 (legacy)，缺少: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}",
            f"{prefix}"))

    return results


def _validate_schema_v2_dict(schema_v2: Dict, prefix: str) -> List[ValidationResult]:
    """校验 Schema V2 (14维) 字典的完整性和类型合法性"""
    results = []

    if not isinstance(schema_v2, dict):
        results.append(ValidationResult("error", f"{prefix} 不是字典", f"{prefix}"))
        return results

    # 必填字段检查
    required_fields = ["contact", "confidence", "compliance_level"]
    for field in required_fields:
        if field not in schema_v2:
            results.append(ValidationResult("error",
                f"缺少必填字段 '{field}'", f"{prefix}.{field}"))

    # contact 类型检查（Schema V2 中应为 bool 或 0/1）
    if "contact" in schema_v2:
        val = schema_v2["contact"]
        if val is not None and not isinstance(val, (bool, int, float)):
            results.append(ValidationResult("warning",
                f"contact 值类型异常: {type(val).__name__}（应为 bool/float）",
                f"{prefix}.contact"))

    # compliance_level 枚举检查
    if "compliance_level" in schema_v2:
        valid_levels = ("L1", "L2", "L3", "L4")
        if schema_v2["compliance_level"] not in valid_levels:
            results.append(ValidationResult("error",
                f"compliance_level 值 '{schema_v2['compliance_level']}' 不合法，应为 {valid_levels}",
                f"{prefix}.compliance_level"))

    # confidence 范围检查
    if "confidence" in schema_v2:
        conf = schema_v2["confidence"]
        if isinstance(conf, (int, float)) and not (0.0 <= conf <= 1.0):
            results.append(ValidationResult("warning",
                f"confidence 超出范围: {conf}（应为 0.0-1.0）",
                f"{prefix}.confidence"))

    # 向量维度检查
    vector_specs = {
        "contact_centroid": 2,
        "force_vector": 3,
        "torque_vector": 3,
        "slip_velocity": 2,
    }
    for vec_field, expected_len in vector_specs.items():
        if vec_field in schema_v2 and schema_v2[vec_field] is not None:
            vec = schema_v2[vec_field]
            if isinstance(vec, list) and len(vec) != expected_len:
                results.append(ValidationResult("warning",
                    f"{vec_field} 维度异常: {len(vec)}（应为 {expected_len}）",
                    f"{prefix}.{vec_field}"))

    # 枚举字段检查
    from tlabel.core.schema import (
        VALID_CONTACT_REGIONS, VALID_MANIPULATION_PHASES,
        VALID_TEXTURE_CLASSES, VALID_COMPLIANCE_LEVELS,
    )
    enum_specs = {
        "contact_region": VALID_CONTACT_REGIONS,
        "manipulation_phase": VALID_MANIPULATION_PHASES,
        "texture_class": VALID_TEXTURE_CLASSES,
    }
    for enum_field, valid_vals in enum_specs.items():
        if enum_field in schema_v2 and schema_v2[enum_field] is not None:
            if schema_v2[enum_field] not in valid_vals:
                results.append(ValidationResult("warning",
                    f"{enum_field} 值 '{schema_v2[enum_field]}' 不合法",
                    f"{prefix}.{enum_field}"))

    # 统计覆盖维度数
    covered = sum(1 for d in SCHEMA_V2_DIMENSIONS if d in schema_v2)
    total = len(SCHEMA_V2_DIMENSIONS)
    if covered < total:
        missing = [d for d in SCHEMA_V2_DIMENSIONS if d not in schema_v2]
        results.append(ValidationResult("info",
            f"覆盖 {covered}/{total} 维 (Schema V2)，缺少: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}",
            f"{prefix}"))

    return results


def _validate_hdf5_file(path: Path) -> List[ValidationResult]:
    """校验 HDF5 格式文件"""
    results = []
    try:
        import h5py
        with h5py.File(path, "r") as f:
            results.append(ValidationResult("info", f"HDF5 文件，顶层 keys: {list(f.keys())}"))
            
            # 检查是否有 tactile 数据组
            if "tactile" in f:
                tactile_keys = list(f["tactile"].keys())
                results.append(ValidationResult("info", f"tactile 组包含 {len(tactile_keys)} 个数据通道"))
            else:
                results.append(ValidationResult("warning", "未找到 'tactile' 数据组"))
    except ImportError:
        results.append(ValidationResult("error", "需要安装 h5py: pip install h5py"))
    except Exception as e:
        results.append(ValidationResult("error", f"HDF5 读取失败: {e}"))
    return results


def _validate_parquet_file(path: Path) -> List[ValidationResult]:
    """校验 Parquet 格式文件"""
    results = []
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        results.append(ValidationResult("info", f"Parquet 文件，{table.num_rows} 行 × {table.num_columns} 列"))
        results.append(ValidationResult("info", f"列名: {table.column_names[:10]}{'...' if len(table.column_names) > 10 else ''}"))
    except ImportError:
        results.append(ValidationResult("error", "需要安装 pyarrow: pip install pyarrow"))
    except Exception as e:
        results.append(ValidationResult("error", f"Parquet 读取失败: {e}"))
    return results


def _validate_directory(path: Path) -> List[ValidationResult]:
    """校验目录格式"""
    results = []
    files = list(path.iterdir())
    results.append(ValidationResult("info", f"目录包含 {len(files)} 个文件/子目录"))
    
    # 尝试自动检测格式
    from tlabel.core.registry import auto_detect_format
    fmt = auto_detect_format(str(path))
    if fmt:
        results.append(ValidationResult("info", f"自动检测到格式: {fmt}"))
    else:
        results.append(ValidationResult("warning", "无法自动检测数据格式"))

    # 检查常见的元数据文件
    meta_files = ["meta/info.json", "README.md", "info.json"]
    for mf in meta_files:
        if (path / mf).exists():
            results.append(ValidationResult("info", f"找到元数据: {mf}"))

    return results


def _validate_auto_detect(path: Path) -> List[ValidationResult]:
    """尝试用 tlabel loader 加载并校验"""
    results = []
    try:
        from tlabel.core.loader import load
        data = load(str(path))
        if data and hasattr(data, "frames"):
            results.append(ValidationResult("info", f"成功加载，共 {len(data.frames)} 帧"))
            if data.frames:
                frame0 = data.frames[0]
                # v0.17: Schema V2 only
                if hasattr(frame0, "schema_v2") and frame0.schema_v2 is not None:
                    results.extend(_validate_schema_v2_dict(frame0.schema_v2.to_dict(), "frame[0]"))
                else:
                    results.append(ValidationResult("error", "frame[0] 缺少 schema_v2 字段", "frame[0]"))
        else:
            results.append(ValidationResult("warning", "加载成功但数据为空"))
    except Exception as e:
        results.append(ValidationResult("error", f"加载失败: {e}"))
    return results


def cmd_validate(args):
    """执行 validate 子命令"""
    path = args.path
    print(f"🔍 校验中: {path}")
    print("=" * 60)

    results = validate_tlabel_file(path)

    # 统计
    errors = [r for r in results if r.level == "error"]
    warnings = [r for r in results if r.level == "warning"]
    infos = [r for r in results if r.level == "info"]

    # 输出结果
    for r in results:
        print(r)

    print("=" * 60)
    print(f"结果: {len(errors)} 错误, {len(warnings)} 警告, {len(infos)} 信息")

    if errors:
        print("\n❌ 校验未通过")
        return 1
    elif warnings:
        print("\n⚠️ 校验通过（有警告）")
        return 0
    else:
        print("\n✅ 校验通过")
        return 0


def _adapter_detail(adapter_cls, with_image_shape: bool = False) -> dict:
    """收集单个适配器的详细信息，供 ``list --verbose`` 与 ``info`` 共用。

    实例化可能因为缺少厂商 SDK / 未连接硬件而失败，这在纯软件环境里是正常情况。
    因此这里把每一步都降级成 None 或错误字符串，绝不因为探测失败中断整个输出。
    """
    from tlabel.adapters.base import SensorAdapterBase

    detail = {
        "class": adapter_cls.__name__,
        "module": adapter_cls.__module__,
        "kind": "sensor" if issubclass(adapter_cls, SensorAdapterBase) else "dataset",
        "description": (adapter_cls.__doc__ or "").strip().split("\n")[0],
        "default_compliance_level": getattr(adapter_cls, "default_compliance_level", None),
        "supported_extensions": [],
        "capabilities": {},
        "sensor_info": {},
    }

    try:
        adapter = adapter_cls()
    except Exception as exc:
        detail["instantiable"] = False
        detail["instantiate_error"] = str(exc)
        return detail

    detail["instantiable"] = True

    try:
        detail["supported_extensions"] = list(adapter.supported_extensions)
    except Exception:
        pass

    try:
        detail["capabilities"] = dict(adapter.get_capabilities())
    except Exception:
        pass

    try:
        detail["sensor_info"] = dict(adapter.get_sensor_info())
    except Exception:
        pass

    if with_image_shape:
        try:
            shape = adapter.detect_image_shape()
            detail["image_shape"] = list(shape) if shape else None
        except Exception:
            detail["image_shape"] = None

    return detail


def _print_capabilities_table(caps: dict) -> None:
    """按 Schema V2 的 14 维顺序打印能力明细。

    顺序固定用 ``SCHEMA_V2_FIELD_NAMES`` 而不是 ``CAPABILITY_KEYS``（后者是 frozenset，
    迭代顺序不稳定）。``compliance_level`` 是帧级元字段，不属于能力键，会被过滤掉。
    """
    from tlabel.core.capabilities import CAPABILITY_KEYS

    ordered = [k for k in SCHEMA_V2_FIELD_NAMES if k in CAPABILITY_KEYS]
    active = [k for k in ordered if caps.get(k)]

    print(f"  能力明细（{len(active)}/{len(ordered)} 活跃）:")
    for key in ordered:
        mark = "✅" if caps.get(key) else "— "
        print(f"    {mark} {key}")

    unknown = sorted(set(caps) - set(CAPABILITY_KEYS))
    if unknown:
        print(f"  ⚠️ 非法能力键（schema 不接受）: {', '.join(unknown)}")


def cmd_list(args):
    """执行 list 子命令"""
    import json as _json

    from tlabel.core.registry import (
        list_builtin_adapters,
        list_external_adapters,
        _ensure_adapters,
    )

    # 触发懒加载
    _ensure_adapters()

    builtin = list_builtin_adapters()
    external = list_external_adapters()
    verbose = bool(getattr(args, "verbose", False))
    as_json = bool(getattr(args, "json", False))

    if as_json:
        payload = []
        for group, items in (("builtin", builtin), ("external", external)):
            for name, cls in sorted(items.items()):
                entry = {"name": name, "group": group}
                entry.update(_adapter_detail(cls, with_image_shape=verbose))
                if not verbose:
                    # 精简模式只给定位信息，能力明细留给 -v
                    entry.pop("capabilities", None)
                    entry.pop("sensor_info", None)
                    entry.pop("description", None)
                payload.append(entry)
        print(_json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("📦 TLabel 已注册适配器")
    print("=" * 60)

    if builtin:
        print(f"\n内置适配器 ({len(builtin)}):")
        for name, cls in sorted(builtin.items()):
            # 判断类型
            from tlabel.adapters.base import SensorAdapterBase
            adapter_type = "🔌 传感器" if issubclass(cls, SensorAdapterBase) else "📁 数据集"
            doc = (cls.__doc__ or "").strip().split("\n")[0][:50]
            print(f"  {adapter_type} {name:20s} {doc}")
            if verbose:
                detail = _adapter_detail(cls)
                exts = detail.get("supported_extensions") or []
                compliance = detail.get("default_compliance_level") or "L1"
                caps = detail.get("capabilities") or {}
                print(f"       扩展名: {', '.join(exts) if exts else '(无)'}")
                print(f"       默认合规: {compliance}   活跃能力: {sum(1 for v in caps.values() if v)} 项")
                if not detail.get("instantiable", True):
                    print(f"       ⚠️ 无法实例化: {detail.get('instantiate_error')}")

    if external:
        print(f"\n社区适配器 ({len(external)}):")
        for name, cls in sorted(external.items()):
            doc = (cls.__doc__ or "").strip().split("\n")[0][:50]
            print(f"  🌐 {name:20s} {doc}")
    else:
        print("\n社区适配器: 暂无（欢迎贡献！详见 CONTRIBUTING.md）")

    print(f"\n总计: {len(builtin) + len(external)} 个适配器")
    if not verbose:
        print("提示: `tlabel list -v` 查看扩展名与能力，`tlabel list --json` 供脚本消费")
    return 0


def cmd_info(args):
    """执行 info 子命令"""
    from tlabel.core.registry import get_adapter, _ensure_adapters

    _ensure_adapters()
    adapter_cls = get_adapter(args.name)
    if adapter_cls is None:
        print(f"❌ 未找到适配器: {args.name}")
        print("使用 'tlabel list' 查看所有可用适配器")
        return 1

    detail = _adapter_detail(
        adapter_cls,
        with_image_shape=bool(getattr(args, "image_shape", False)),
    )

    print(f"📋 适配器详情: {args.name}")
    print("=" * 40)
    print(f"  名称: {args.name}")
    print(f"  类名: {detail['class']}")
    print(f"  模块: {detail['module']}")
    print(f"  类型: {'🔌 传感器' if detail['kind'] == 'sensor' else '📁 数据集'}")
    if detail.get("description"):
        print(f"  描述: {detail['description']}")
    print(f"  默认合规等级: {detail.get('default_compliance_level') or 'L1'}")

    exts = detail.get("supported_extensions") or []
    print(f"  支持格式: {', '.join(exts) if exts else '(无)'}")

    if not detail.get("instantiable", True):
        print(f"  ⚠️ 无法实例化: {detail.get('instantiate_error')}")
        print("     （通常表示缺少厂商 SDK 依赖，见 pyproject.toml 的 optional-dependencies）")
        return 0

    caps = detail.get("capabilities") or {}
    if caps:
        active_v2 = [k for k, v in caps.items() if v and k in SCHEMA_V2_DIMENSIONS]
        active_legacy = [k for k, v in caps.items() if v and k in LEGACY_V2_DIMENSIONS]
        if active_v2:
            print(f"  活跃维度 (Schema V2): {len(active_v2)}/14")
        if active_legacy:
            print(f"  活跃维度 (legacy): {len(active_legacy)}/22")
        print()
        _print_capabilities_table(caps)

    sensor_info = detail.get("sensor_info") or {}
    if sensor_info:
        print()
        print(f"  传感器类型: {sensor_info.get('type', 'unknown')}")
        print(f"  制造商: {sensor_info.get('manufacturer', 'unknown')}")

    if "image_shape" in detail:
        shape = detail["image_shape"]
        print(f"  图像形状: {tuple(shape) if shape else '未检测到'}")

    return 0


def cmd_version(args):
    """执行 version 子命令"""
    from tlabel._version import __version__
    print(f"tlabel {__version__}")
    return 0




def _resolve_export_format(requested: str) -> str:
    """把 CLI 的格式名映射到 registry 的 exporter id。

    ``lerobot`` 在 registry 里指"写回已有数据集"，但 CLI 自发布以来
    ``-f lerobot`` 的语义一直是"新建数据集"，对应 ``lerobot_create``。
    为保持旧行为不变，这里做别名映射；写回路径改用 ``lerobot_writeback`` 显式触发。
    """
    aliases = {
        "lerobot": "lerobot_create",
        "lerobot_writeback": "lerobot",
    }
    return aliases.get(requested, requested)


#: registry id → CLI 展示名。错误提示必须用 CLI 名字，否则用户照着提示
#: 输入 ``-f lerobot`` 会得到"新建"而不是提示里那个"写回"导出器。
_CLI_FORMAT_DISPLAY = {
    "lerobot_create": "lerobot",
    "lerobot": "lerobot_writeback",
}


def _cli_format_names(registry) -> list:
    """按 CLI 语义列出可用格式名（不含内部 id，不含别名重复项）。"""
    names = {_CLI_FORMAT_DISPLAY.get(s.id, s.id) for s in registry.list_formats()}
    return sorted(names)


def _export_kwargs(args, exporter_id: str, task_override=None) -> dict:
    """按目标导出器组装参数。

    只透传显式暴露的 CLI 选项——导出器对未知 kwargs 是静默忽略的，
    整包塞 argparse 命名空间会把拼写错误藏起来。

    ``task_override`` 供批量模式使用：单文件导出时 task 来自 ``--task``，
    批量时若用户没指定，用文件名作为任务名比全部叫 "task" 更有意义。
    """
    if exporter_id == "lerobot_create":
        return {
            "adapter_name": args.adapter,
            "task": args.task or task_override or "task",
            "task_description": args.task_description or "",
            "episode_index": args.episode_index,
        }

    if exporter_id == "lerobot":
        return {"overwrite": bool(getattr(args, "overwrite", False))}

    if exporter_id == "ftp1":
        kwargs = {}
        sensor = getattr(args, "sensor", None)
        if sensor:
            kwargs["sensor_name"] = sensor
        areas = getattr(args, "areas", None)
        if areas:
            try:
                kwargs["functional_areas"] = [
                    int(part) for part in areas.split(",") if part.strip()
                ]
            except ValueError as exc:
                raise ValueError(
                    f"--areas 需要逗号分隔的整数（如 0,1），收到 {areas!r}"
                ) from exc
        return kwargs

    return {}


def _cmd_export_batch(args, exporter_id, spec, registry, adapter_cls):
    """批量导出：递归目录、逐文件导出、失败隔离、汇总成败。

    输出路径保持输入目录的相对结构，后缀由 ``spec.file_ext`` 决定
    （不再硬编码 lerobot/ftp1 的分支）。``lerobot`` 是写回模式，
    语义上只能作用于单个已有数据集，批量场景直接拒绝。
    """
    from pathlib import Path

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return 1
    if not input_dir.is_dir():
        print(f"❌ --batch 需要输入目录，收到的是文件: {input_dir}")
        return 1
    if exporter_id == "lerobot":
        print("❌ lerobot（写回）不支持批量：写回必须指定单个已有数据集目录")
        print("   批量新建数据集请用 -f lerobot")
        return 1

    try:
        exts = {e.lower() for e in adapter_cls().supported_extensions}
    except Exception:
        exts = set()
    if not exts:
        exts = {".json"}

    files = []
    for ext in sorted(exts):
        files.extend(input_dir.rglob(f"*{ext}"))
    files = sorted(set(files))

    if not files:
        print(f"❌ 目录下没有匹配 {'/'.join(sorted(exts))} 的文件: {input_dir}")
        return 1

    print(f"📦 批量导出 {spec.name}")
    print("=" * 60)
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  适配器:   {args.adapter}")
    print(f"  格式:     {spec.id}")
    print(f"  文件数:   {len(files)}")
    print("=" * 60)

    succeeded, failed = [], []
    for idx, path in enumerate(files, 1):
        rel = path.relative_to(input_dir)
        rel_stem = rel.with_suffix("")
        out_path = output_dir / f"{rel_stem}{spec.file_ext}"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            adapter = adapter_cls()
            data = adapter.load(str(path))
            if data is None:
                raise ValueError("适配器未能读出数据")
            if getattr(data, "num_frames", 0) <= 0:
                raise ValueError("读出的数据为 0 帧，源文件可能不含有效帧")
            kwargs = _export_kwargs(args, exporter_id, task_override=rel_stem.name)
            result = registry.export(exporter_id, data, str(out_path), **kwargs)
            if not result.success:
                raise RuntimeError(result.error or "导出器返回失败")
            succeeded.append((rel, out_path))
            print(f"  [{idx}/{len(files)}] ✅ {rel} → {out_path.name}")
        except Exception as exc:
            failed.append((rel, str(exc)))
            print(f"  [{idx}/{len(files)}] ❌ {rel}: {exc}")

    print("=" * 60)
    print(f"批量导出完成：成功 {len(succeeded)}，失败 {len(failed)}")
    if failed:
        print("失败清单：")
        for rel, err in failed:
            print(f"  - {rel}: {err}")
        return 1
    return 0


def cmd_export(args):
    """执行 export 子命令"""
    from tlabel.core.registry import get_adapter, _ensure_adapters
    from tlabel.export.registry import get_registry

    requested = (args.format or "lerobot").lower()
    exporter_id = _resolve_export_format(requested)
    registry = get_registry()

    spec = next((s for s in registry.list_formats() if s.id == exporter_id), None)
    if spec is None:
        available = ", ".join(_cli_format_names(registry))
        print(f"❌ 不支持的导出格式: {requested}")
        print(f"   支持的格式: {available}")
        return 1

    _ensure_adapters()
    adapter_cls = get_adapter(args.adapter)
    if adapter_cls is None:
        print(f"❌ 未找到适配器: {args.adapter}")
        print("使用 'tlabel list' 查看所有可用适配器")
        return 1

    if getattr(args, "batch", False):
        return _cmd_export_batch(args, exporter_id, spec, registry, adapter_cls)

    # 写回需要指向一个已存在的 LeRobot 数据集；提前拦下能给出可操作的提示，
    # 否则会一路走到转换器里报 "No parquet files found"。
    if exporter_id == "lerobot":
        target = Path(args.output)
        if not target.is_dir():
            print(f"❌ lerobot_writeback 需要指向已存在的 LeRobot 数据集目录")
            print(f"   目录不存在: {target}")
            print("   新建数据集请用 -f lerobot")
            return 1
        if not any((target / "data").rglob("*.parquet")):
            print(f"❌ {target} 下没有 data/**/*.parquet，不是有效的 LeRobot 数据集")
            print("   新建数据集请用 -f lerobot")
            return 1

    print(f"📦 导出 {spec.name}")
    print("=" * 60)
    print(f"  输入:   {args.input}")
    print(f"  输出:   {args.output}")
    print(f"  适配器: {args.adapter}")
    print(f"  格式:   {spec.id}")
    if args.task:
        print(f"  任务:   {args.task}")
    print("=" * 60)

    try:
        adapter = adapter_cls()
        data = adapter.load(args.input)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return 1

    if data is None:
        print(f"❌ 适配器 {args.adapter} 未能从 {args.input} 读出数据")
        return 1
    if getattr(data, "num_frames", 0) <= 0:
        print(f"❌ 读出的数据为 0 帧，{args.input} 可能不含有效帧")
        return 1

    try:
        kwargs = _export_kwargs(args, exporter_id)
    except ValueError as e:
        print(f"❌ 参数错误: {e}")
        return 1

    try:
        result = registry.export(exporter_id, data, args.output, **kwargs)
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return 1

    if not result.success:
        print(f"❌ 导出失败: {result.error}")
        return 1

    print("=" * 60)
    print(f"✅ 导出成功！")
    print(f"   输出:   {result.output_path}")

    stats = result.stats or {}
    num_frames = stats.get("num_frames") or stats.get("frames")
    if num_frames:
        print(f"   帧数:   {num_frames}")
    if stats.get("duration_s"):
        print(f"   时长:   {stats['duration_s']}s")
    if stats.get("tactile_dim"):
        print(f"   触觉维度: {stats['tactile_dim']}")
    if stats.get("state_dim"):
        print(f"   状态维度: {stats['state_dim']}")
    if stats.get("action_dim"):
        print(f"   动作维度: {stats['action_dim']}")
    if result.file_size_bytes:
        print(f"   产物大小: {result.file_size_bytes / 1024:.1f} KB")

    return 0


def cmd_template_list(args):
    """执行 template list 子命令 — 列出所有场景模板"""
    from tlabel.scenarios import list_scenarios, get_scenario

    ids = list_scenarios()

    print("📋 TLabel 场景模板")
    print("=" * 60)

    if not ids:
        print("  暂无场景模板")
        return 0

    for sid in ids:
        try:
            tpl = get_scenario(sid)
            adapters = ",".join(tpl.adapters) if tpl.adapters else "-"
            print(f"  {sid:22s} {tpl.name:20s} 适配器: {adapters:16s} 最低合规: {tpl.min_compliance_level}")
        except Exception as e:
            print(f"  {sid:22s} ⚠️ 加载失败: {e}")

    print(f"\n共 {len(ids)} 个场景模板")
    return 0


def cmd_template_validate(args):
    """执行 template validate 子命令 — 校验指定或全部场景模板"""
    from tlabel.scenarios import validate_scenario, validate_all_scenarios, list_scenarios

    scenario_id = getattr(args, "id", None)
    validate_all = getattr(args, "all", False)

    if not scenario_id and not validate_all:
        print("❌ 请指定场景模板 id，或使用 --all 校验全部模板")
        print(f"   可用模板: {list_scenarios()}")
        return 1

    print("🔍 校验场景模板")
    print("=" * 60)

    if scenario_id:
        targets = {scenario_id: validate_scenario(scenario_id)}
    else:
        targets = validate_all_scenarios()

    passed = 0
    failed = 0
    for sid, (ok, errors) in targets.items():
        if ok:
            passed += 1
            print(f"✅ {sid} — 通过")
        else:
            failed += 1
            print(f"❌ {sid} — 失败")
            for err in errors:
                print(f"     • {err}")

    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")

    if failed:
        print("\n❌ 校验未通过")
        return 1
    print("\n✅ 校验通过")
    return 0


def main():
    from tlabel._version import __version__

    parser = argparse.ArgumentParser(
        prog="tlabel",
        description="TLabel — 触觉数据标注工具包 CLI",
    )
    # v0.24.2: 支持 `tlabel --version`。此前只有 `tlabel version` 子命令，
    # 而 --version 是绝大多数 CLI 的通用约定，新用户第一反应都是敲这个。
    parser.add_argument(
        "--version", action="version", version=f"tlabel {__version__}",
        help="显示版本号（等价于 `tlabel version`）",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # validate
    p_validate = subparsers.add_parser("validate", help="校验数据文件是否符合 tlabel_v2 schema")
    p_validate.add_argument("path", help="数据文件或目录路径")
    p_validate.set_defaults(func=cmd_validate)

    # list
    p_list = subparsers.add_parser("list", help="列出所有已注册的适配器")
    p_list.add_argument(
        "--verbose", "-v", action="store_true",
        help="展示扩展名、默认合规等级与活跃能力数",
    )
    p_list.add_argument(
        "--json", action="store_true",
        help="以 JSON 输出（供脚本消费）",
    )
    p_list.set_defaults(func=cmd_list)

    # info
    p_info = subparsers.add_parser("info", help="查看适配器详细信息")
    p_info.add_argument("name", help="适配器名称")
    p_info.add_argument(
        "--image-shape", action="store_true",
        help="探测传感器图像形状（需要实例化，可能较慢）",
    )
    p_info.set_defaults(func=cmd_info)

    # version
    p_version = subparsers.add_parser("version", help="显示版本号")
    p_version.set_defaults(func=cmd_version)
    # export
    p_export = subparsers.add_parser(
        "export",
        help="导出数据为其他格式（lerobot / lerobot_create / ftp1 / json / csv / hdf5）",
    )
    p_export.add_argument("input", help="输入数据文件路径")
    p_export.add_argument("output", help="输出目录或文件路径")
    p_export.add_argument(
        "--format", "-f", default="lerobot",
        help=(
            "导出格式（默认 lerobot，等价于 lerobot_create 新建数据集；"
            "写回已有数据集用 lerobot_writeback；FTP-1 用 ftp1）"
        ),
    )
    p_export.add_argument(
        "--adapter", "-a", required=True,
        help="适配器名称（如 tashan_ts_f_a, gelsight）",
    )
    p_export.add_argument(
        "--task", default="",
        help="任务名称（用于 meta 数据）",
    )
    p_export.add_argument(
        "--task-description", default="",
        help="任务描述",
    )
    p_export.add_argument(
        "--episode-index", type=int, default=0,
        help="Episode 编号（默认 0）",
    )
    p_export.add_argument(
        "--sensor", default=None,
        help="FTP-1 专用：传感器型号（如 GelSightMini），默认取导出器预设",
    )
    p_export.add_argument(
        "--areas", default=None,
        help="FTP-1 专用：手部功能区，逗号分隔（如 0,1）",
    )
    p_export.add_argument(
        "--overwrite", action="store_true",
        help="lerobot_writeback 专用：允许覆盖已有数据集",
    )
    p_export.add_argument(
        "--batch", action="store_true",
        help=(
            "批量模式：INPUT 视为目录，按适配器支持的扩展名递归收集文件，"
            "OUTPUT 视为输出目录并保持相对结构；单个文件失败不中断"
        ),
    )
    p_export.set_defaults(func=cmd_export)

    # template
    p_template = subparsers.add_parser("template", help="场景模板（scenario template）管理")
    template_sub = p_template.add_subparsers(dest="template_command", help="可用子命令")

    # template list
    p_tpl_list = template_sub.add_parser("list", help="列出所有场景模板")
    p_tpl_list.set_defaults(func=cmd_template_list)

    # template validate
    p_tpl_validate = template_sub.add_parser("validate", help="校验场景模板（字段名/primitive 名合法性）")
    p_tpl_validate.add_argument(
        "id", nargs="?", default=None,
        help="场景模板 id（省略时配合 --all 校验全部）",
    )
    p_tpl_validate.add_argument(
        "--all", action="store_true",
        help="校验全部场景模板",
    )
    p_tpl_validate.set_defaults(func=cmd_template_validate)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    func = getattr(args, "func", None)
    if func is None:
        # 例如 `tlabel template` 未指定子命令
        parser.print_help()
        return 0

    return func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
