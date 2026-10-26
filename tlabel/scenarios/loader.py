"""
TLabel 场景模板（Scenario Template）— v0.24.0 新增

场景模板描述「某类典型触觉采集场景下，TLabel 标注应该关注哪些字段 /
primitive / 事件」，让使用者在标注前有一份可校验的场景基线。

设计原则：
- 模板以 YAML 声明式定义，存放于 tlabel/scenarios/<id>.yaml
- 所有字段名 / primitive 名 / 事件类型必须与代码中的 Schema V2 与
  taxonomy 严格一致，加载时自动校验，避免模板与代码漂移
- 只做声明与校验，不做 primitive 推断（TLabel 定位为统一标注格式标准）

用法：
    from tlabel.scenarios import list_scenarios, get_scenario, validate_all_scenarios

    ids = list_scenarios()               # ['gelsight_grasp', ...]
    tpl = get_scenario('gelsight_grasp') # ScenarioTemplate
    print(tpl.schema_fields, tpl.primitives)

    results = validate_all_scenarios()   # {id: (is_valid, errors)}
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tlabel.core.schema import (
    SCHEMA_V2_FIELD_NAMES,
    VALID_COMPLIANCE_LEVELS,
)
from tlabel.core.events import EVENT_PRESETS
from tlabel.core.primitive import PRIMITIVE_PRESETS


# 场景模板目录（与本模块同目录）
SCENARIOS_DIR = Path(__file__).parent

# 模板必需键
REQUIRED_KEYS = (
    "id",
    "name",
    "description",
    "adapters",
    "schema_fields",
    "primitives",
    "events",
    "min_compliance_level",
)

# 约束块允许的键
CONSTRAINT_KEYS = (
    "required_when_contact",
    "required_when_slip",
    "recommended_optional",
)

# 合法的 schema 字段名集合（14 维 Schema V2）
VALID_SCHEMA_FIELDS = set(SCHEMA_V2_FIELD_NAMES)

# 合法的 primitive 名集合（内置 22 种）
VALID_PRIMITIVES = set(PRIMITIVE_PRESETS)

# 合法的触觉事件类型集合（6 种）
VALID_EVENTS = set(EVENT_PRESETS.keys())

# 合法的合规等级集合（L1-L4）
VALID_COMPLIANCE = set(VALID_COMPLIANCE_LEVELS)


@dataclass
class ScenarioTemplate:
    """单个场景模板

    Attributes:
        id: 场景 id（须与文件名一致）
        name: 用户可见名称
        description: 场景说明
        adapters: 适用适配器名称列表（须为已注册适配器）
        schema_fields: 关注的 Schema V2 (14维) 字段名列表
        primitives: 期望的 primitive 集合（须在内置 22 种中）
        events: 期望的触觉事件类型集合（须为 6 种预设之一）
        min_compliance_level: 最低合规等级要求（L1-L4）
        sensors: 适用传感器型号列表（自由文本，可选）
        constraints: 可选参数约束（如 required_when_contact）
        source_path: 模板 YAML 来源路径
    """

    id: str
    name: str
    description: str
    adapters: List[str]
    schema_fields: List[str]
    primitives: List[str]
    events: List[str]
    min_compliance_level: str
    sensors: List[str] = field(default_factory=list)
    constraints: Dict[str, List[str]] = field(default_factory=dict)
    source_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any],
                  source_path: Optional[str] = None) -> "ScenarioTemplate":
        """从字典构建 ScenarioTemplate（不做合法性校验）"""
        return cls(
            id=data["id"],
            name=data["name"],
            description=str(data["description"]).strip(),
            adapters=list(data.get("adapters", [])),
            schema_fields=list(data.get("schema_fields", [])),
            primitives=list(data.get("primitives", [])),
            events=list(data.get("events", [])),
            min_compliance_level=data["min_compliance_level"],
            sensors=list(data.get("sensors", [])),
            constraints=dict(data.get("constraints", {})),
            source_path=source_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（UI / API 可消费）"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "adapters": list(self.adapters),
            "sensors": list(self.sensors),
            "schema_fields": list(self.schema_fields),
            "primitives": list(self.primitives),
            "events": list(self.events),
            "min_compliance_level": self.min_compliance_level,
            "constraints": {k: list(v) for k, v in self.constraints.items()},
        }

    def __repr__(self) -> str:
        return (f"ScenarioTemplate(id='{self.id}', name='{self.name}', "
                f"adapters={self.adapters}, min_compliance_level='{self.min_compliance_level}')")


# ============================================================
# 加载
# ============================================================

def _scenario_path(scenario_id: str) -> Path:
    """返回场景模板 YAML 路径"""
    return SCENARIOS_DIR / f"{scenario_id}.yaml"


def list_scenarios() -> List[str]:
    """列出所有场景模板 id（按文件名排序）"""
    ids = sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml"))
    return ids


def get_scenario(scenario_id: str) -> ScenarioTemplate:
    """按 id 读取单个场景模板

    Args:
        scenario_id: 场景 id（不含 .yaml 后缀）

    Returns:
        ScenarioTemplate 实例

    Raises:
        FileNotFoundError: 模板文件不存在
        ValueError: YAML 解析失败或缺少必需键
    """
    path = _scenario_path(scenario_id)
    if not path.exists():
        raise FileNotFoundError(
            f"场景模板不存在: '{scenario_id}'（{path}）。"
            f"可用模板: {list_scenarios()}"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"场景模板 YAML 解析失败: {scenario_id}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"场景模板内容必须为映射（dict）: {scenario_id}")

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"场景模板缺少必需键 {missing}: {scenario_id}")

    return ScenarioTemplate.from_dict(data, source_path=str(path))


def load_all_scenarios() -> Dict[str, ScenarioTemplate]:
    """加载全部场景模板

    Returns:
        {scenario_id: ScenarioTemplate}
    """
    return {sid: get_scenario(sid) for sid in list_scenarios()}


# ============================================================
# 校验
# ============================================================

def validate_template_data(data: Dict[str, Any],
                           expected_id: Optional[str] = None,
                           source_path: Optional[str] = None) -> List[str]:
    """校验场景模板字典的合法性（不读文件）

    检查项：
      - 必需键是否齐全，类型是否正确
      - id 是否与文件名（expected_id）一致
      - schema_fields 是否为 Schema V2 (14维) 字段名
      - primitives 是否为内置 22 种 primitive
      - events 是否为 6 种预设事件类型
      - min_compliance_level 是否为 L1-L4
      - adapters 是否为已注册适配器
      - constraints 键是否合法、引用的字段名是否存在

    Args:
        data: 模板字典
        expected_id: 期望的 id（通常是文件名 stem），用于一致性检查
        source_path: 来源路径（仅用于错误信息）

    Returns:
        错误消息列表，空列表表示通过
    """
    errors: List[str] = []
    loc = source_path or expected_id or "<dict>"

    if not isinstance(data, dict):
        return [f"模板内容必须为映射（dict）: {loc}"]

    # 1. 必需键
    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"缺少必需键 '{key}'")
    if errors:
        # 缺少必需键时无法继续，直接返回
        return [f"{e} ({loc})" for e in errors]

    # 2. 基础类型
    for str_key in ("id", "name", "description", "min_compliance_level"):
        if not isinstance(data.get(str_key), str) or not data.get(str_key).strip():
            errors.append(f"'{str_key}' 必须为非空字符串")

    for list_key in ("adapters", "schema_fields", "primitives", "events"):
        val = data.get(list_key)
        if not isinstance(val, list) or not val:
            errors.append(f"'{list_key}' 必须为非空列表")

    # 3. id 与文件名一致
    if isinstance(data.get("id"), str) and expected_id is not None:
        if data["id"] != expected_id:
            errors.append(
                f"id '{data['id']}' 与文件名 '{expected_id}' 不一致"
            )

    # 4. schema_fields 合法性
    schema_fields = data.get("schema_fields")
    if isinstance(schema_fields, list):
        for fld in schema_fields:
            if not isinstance(fld, str) or fld not in VALID_SCHEMA_FIELDS:
                errors.append(
                    f"未知 schema 字段 '{fld}'（不在 Schema V2 14维中）"
                )

    # 5. primitives 合法性
    primitives = data.get("primitives")
    if isinstance(primitives, list):
        for p in primitives:
            if not isinstance(p, str) or p not in VALID_PRIMITIVES:
                errors.append(
                    f"未知 primitive '{p}'（不在内置 22 种 primitive 中）"
                )

    # 6. events 合法性
    events = data.get("events")
    if isinstance(events, list):
        for ev in events:
            if not isinstance(ev, str) or ev not in VALID_EVENTS:
                errors.append(
                    f"未知事件类型 '{ev}'（不在 6 种预设事件中）"
                )

    # 7. compliance level
    mcl = data.get("min_compliance_level")
    if isinstance(mcl, str) and mcl not in VALID_COMPLIANCE:
        errors.append(
            f"未知合规等级 '{mcl}'（须为 {sorted(VALID_COMPLIANCE)} 之一）"
        )

    # 8. adapters 是否已注册
    adapters = data.get("adapters")
    if isinstance(adapters, list):
        known = _known_adapter_names()
        for a in adapters:
            if a not in known:
                errors.append(f"未注册的适配器 '{a}'")

    # 9. constraints 结构
    constraints = data.get("constraints", {})
    if constraints is not None:
        if not isinstance(constraints, dict):
            errors.append("'constraints' 必须为映射（dict）")
        else:
            for ckey, cval in constraints.items():
                if ckey not in CONSTRAINT_KEYS:
                    errors.append(
                        f"未知 constraints 键 '{ckey}'（须为 {list(CONSTRAINT_KEYS)} 之一）"
                    )
                    continue
                if not isinstance(cval, list):
                    errors.append(f"constraints.{ckey} 必须为列表")
                    continue
                for fld in cval:
                    if fld not in VALID_SCHEMA_FIELDS:
                        errors.append(
                            f"constraints.{ckey} 引用了未知 schema 字段 '{fld}'"
                        )

    return [f"{e} ({loc})" for e in errors]


def _known_adapter_names() -> set:
    """返回已注册的内置适配器名称集合（含外部适配器）"""
    try:
        from tlabel.core.registry import _ADAPTER_MODULES, list_external_adapters
        names = set(_ADAPTER_MODULES.keys())
        names |= set(list_external_adapters().keys())
        return names
    except Exception:
        # 极端情况下退化为不做适配器名校验
        return set()


def validate_scenario(scenario_id: str) -> Tuple[bool, List[str]]:
    """校验单个场景模板

    Args:
        scenario_id: 场景 id

    Returns:
        (is_valid, errors): is_valid=True 表示通过
    """
    path = _scenario_path(scenario_id)
    if not path.exists():
        return False, [f"场景模板不存在: '{scenario_id}'"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"YAML 解析失败: {e}"]

    errors = validate_template_data(data, expected_id=scenario_id,
                                    source_path=str(path))
    return (len(errors) == 0, errors)


def validate_scenario_file(path: str) -> Tuple[bool, List[str]]:
    """校验任意路径下的场景模板文件（用于校验非法模板）

    Args:
        path: YAML 文件路径

    Returns:
        (is_valid, errors)
    """
    p = Path(path)
    if not p.exists():
        return False, [f"文件不存在: {path}"]
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"YAML 解析失败: {e}"]

    expected_id = p.stem
    errors = validate_template_data(data, expected_id=expected_id,
                                    source_path=str(p))
    return (len(errors) == 0, errors)


def validate_all_scenarios() -> Dict[str, Tuple[bool, List[str]]]:
    """校验全部场景模板

    Returns:
        {scenario_id: (is_valid, errors)}
    """
    return {sid: validate_scenario(sid) for sid in list_scenarios()}
