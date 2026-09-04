"""
Sensor adapter registry

Adapters digest format differences at load time, outputting unified TLabel Format v2.

v0.14: 重构为表驱动注册，新增适配器只需在 _ADAPTER_MODULES 加一行。
v0.16: 新增外部/社区适配器注册支持（entry_points + 手动注册）。

"""

import importlib
import logging
from typing import Dict, Type, Optional, List

logger = logging.getLogger(__name__)

_ADAPTERS: Dict[str, Type] = {}
_EXTERNAL_ADAPTERS: Dict[str, Type] = {}  # 社区/外部适配器

# 内置适配器注册表
_ADAPTER_MODULES = {
    "gelsight":      ("tlabel.adapters.gelsight",      "GelSightAdapter"),
    "paxini":        ("tlabel.adapters.paxini_dataset", "PaxiniAdapter"),
    "daimon":        ("tlabel.adapters.daimon_dataset", "DaimonAdapter"),
    "tlabel":        ("tlabel.adapters.tlabel_format",  "TLabelAdapter"),
    "touchd":        ("tlabel.adapters.touchd",         "ToucHDAdapter"),
    "univtac":       ("tlabel.adapters.univtac",        "UniVTACAdapter"),
    "vtouch":        ("tlabel.adapters.vtouch",         "VTouchAdapter"),
    "ycb_slide":     ("tlabel.adapters.ycb_slide",      "YCBSlideAdapter"),
    "tacquad":       ("tlabel.adapters.tacquad",        "TacQuadAdapter"),
    "paxini_gen3":   ("tlabel.adapters.paxini_gen3",    "PaxiniGen3Adapter"),
    "paxini_px6d":   ("tlabel.adapters.paxini_px6d",    "PaxiniPX6DAdapter"),  # placeholder
    "daimon_dm_tac": ("tlabel.adapters.daimon_dm_tac",  "DaimonDmTacAdapter"),
    "syntouch":      ("tlabel.adapters.syntouch",      "SynTouchBioTacAdapter"),
    "xela":          ("tlabel.adapters.xela",          "XelaUskinAdapter"),
    "tashan_ts_f_a": ("tlabel.adapters.tashan_ts_f_a", "TashanTsFAAdapter"),
}

# 社区适配器 entry_point group name
_COMMUNITY_ADAPTER_GROUP = "tlabel.adapters"


def register_adapter(name: str, adapter_cls: Type):
    """注册适配器（内置或手动注册）

    参数:
        name: 适配器唯一标识符（小写+下划线）
        adapter_cls: 适配器类（需继承 DataAdapterBase 或 SensorAdapterBase）
    """
    _ADAPTERS[name] = adapter_cls
    logger.debug(f"Registered adapter: {name}")


def register_external_adapter(name: str, adapter_cls: Type):
    """注册外部/社区适配器

    社区贡献的适配器通过此方法注册，与内置适配器分开管理但统一可用。
    也可通过 entry_points 自动发现（在 pyproject.toml 中声明）。

    参数:
        name: 适配器唯一标识符
        adapter_cls: 适配器类

    示例:
        # 在第三方包中手动注册
        from tlabel.core.registry import register_external_adapter
        register_external_adapter("my_sensor", MySensorAdapter)

        # 或通过 entry_points 自动注册（pyproject.toml）
        # [project.entry-points."tlabel.adapters"]
        # my_sensor = "my_package.adapter:MySensorAdapter"
    """
    _EXTERNAL_ADAPTERS[name] = adapter_cls
    _ADAPTERS[name] = adapter_cls
    logger.info(f"Registered external adapter: {name}")


def get_adapter(name: str) -> Optional[Type]:
    """Get an adapter class"""
    return _ADAPTERS.get(name)


def list_adapters() -> Dict[str, Type]:
    """List all registered adapters (内置 + 外部)"""
    _ensure_adapters()
    return dict(_ADAPTERS)


def list_builtin_adapters() -> Dict[str, Type]:
    """仅列出内置适配器"""
    _ensure_adapters()
    return {k: v for k, v in _ADAPTERS.items() if k not in _EXTERNAL_ADAPTERS}


def list_external_adapters() -> Dict[str, Type]:
    """仅列出外部/社区适配器"""
    _ensure_adapters()
    return dict(_EXTERNAL_ADAPTERS)


def auto_detect_format(file_path: str) -> Optional[str]:
    """Auto-detect format from file extension and content"""
    path = str(file_path).lower()

    if path.endswith(".pkl") or path.endswith(".pickle"):
        return "gelsight"
    if path.endswith(".npy"):
        return "ycb_slide"
    if path.endswith(".h5") or path.endswith(".hdf5"):
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                if 'tactile' in f:
                    tactile_keys = list(f['tactile'].keys())
                    if any('gsmini' in k for k in tactile_keys):
                        return "univtac"
                    if any(k.startswith('hand_') for k in tactile_keys):
                        return "vtouch"
            return "paxini"
        except (ImportError, Exception):
            return "paxini"
    if path.endswith(".parquet"):
        return "daimon"
    if path.endswith(".json"):
        import json
        from pathlib import Path
        try:
            p = Path(file_path)
            if not p.exists():
                return None
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "schema_version" in data and "frames" in data:
                return "tlabel"
            if "episodes" in data:
                return "tlabel"
            if "robot_type" in data and "codebase_version" in data:
                return "daimon"
            if "frames" in data and "channels" in data:
                return "daimon"
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            pass
        return None

    from pathlib import Path
    p = Path(file_path)
    if p.is_dir():
        if (p / "meta" / "info.json").exists() or list(p.glob("data/chunk-*/file-*.parquet")):
            return "daimon"
        if (p / "all_data_direction.json").exists():
            return "touchd"
        if any(p.glob("*/synced_data.npy")) or any(p.glob("*/tactile_data.pkl")):
            return "ycb_slide"
        if (p / "synced_data.npy").exists() or (p / "tactile_data.pkl").exists():
            return "ycb_slide"

        # TacQuad detection: directory with contact_indoor.csv or contact_outdoor.csv
        # Also check for data_indoor/ + data_outdoor/ structure
        if ((p / "contact_indoor.csv").exists() and (p / "data_indoor").exists()) or \
           ((p / "contact_outdoor.csv").exists() and (p / "data_outdoor").exists()):
            return "tacquad"
        # TacQuad parent directory: tactile_datasets/ with tacquad/ subdirectory
        if (p / "tacquad" / "contact_indoor.csv").exists() and \
           (p / "tacquad" / "data_indoor").exists():
            return "tacquad"

    return None


def _discover_community_adapters():
    """通过 entry_points 自动发现社区适配器

    第三方包可在 pyproject.toml 中声明 entry_points:
        [project.entry-points."tlabel.adapters"]
        my_sensor = "my_package.adapter:MySensorAdapter"

    安装后 tlabel 会自动发现并注册。
    """
    try:
        from importlib.metadata import entry_points
        # Python 3.12+ 和 3.9-3.11 的 entry_points API 不同
        try:
            # Python 3.12+
            eps = entry_points(group=_COMMUNITY_ADAPTER_GROUP)
        except TypeError:
            # Python 3.9-3.11
            eps = entry_points().get(_COMMUNITY_ADAPTER_GROUP, [])

        for ep in eps:
            if ep.name not in _ADAPTERS:
                try:
                    cls = ep.load()
                    register_external_adapter(ep.name, cls)
                    logger.info(f"Discovered community adapter via entry_points: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load community adapter '{ep.name}': {e}")
    except ImportError:
        pass  # importlib.metadata not available (very old Python)


def _ensure_adapters():
    """Lazy registration — 内置适配器表驱动 + 社区适配器 entry_points 自动发现"""
    # 1. 注册内置适配器
    for name, (module_path, class_name) in _ADAPTER_MODULES.items():
        if name not in _ADAPTERS:
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                register_adapter(name, cls)
            except (ImportError, AttributeError):
                pass

    # 2. 发现社区适配器
    if not _EXTERNAL_ADAPTERS:
        _discover_community_adapters()

