"""
TLabel 场景模板（Scenario Template）— v0.24.0 新增

场景模板描述「某类典型触觉采集场景下，TLabel 标注应该关注哪些字段 /
primitive / 事件」，让使用者在标注前有一份可校验的场景基线。

模板以 YAML 声明式定义，存放于本目录下的 <id>.yaml；所有字段名 /
primitive 名 / 事件类型在加载时都会与代码中的 Schema V2 与 taxonomy 校验，
避免模板与代码漂移。

用法：
    from tlabel.scenarios import list_scenarios, get_scenario, validate_all_scenarios

    ids = list_scenarios()                  # ['gelsight_grasp', ...]
    tpl = get_scenario('gelsight_grasp')    # ScenarioTemplate
    ok, errors = validate_scenario('gelsight_grasp')
"""

from tlabel.scenarios.loader import (
    ScenarioTemplate,
    SCENARIOS_DIR,
    REQUIRED_KEYS,
    CONSTRAINT_KEYS,
    VALID_SCHEMA_FIELDS,
    VALID_PRIMITIVES,
    VALID_EVENTS,
    VALID_COMPLIANCE,
    list_scenarios,
    get_scenario,
    load_all_scenarios,
    validate_template_data,
    validate_scenario,
    validate_scenario_file,
    validate_all_scenarios,
)

__all__ = [
    "ScenarioTemplate",
    "SCENARIOS_DIR",
    "REQUIRED_KEYS",
    "CONSTRAINT_KEYS",
    "VALID_SCHEMA_FIELDS",
    "VALID_PRIMITIVES",
    "VALID_EVENTS",
    "VALID_COMPLIANCE",
    "list_scenarios",
    "get_scenario",
    "load_all_scenarios",
    "validate_template_data",
    "validate_scenario",
    "validate_scenario_file",
    "validate_all_scenarios",
]
