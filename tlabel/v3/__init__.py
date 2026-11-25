"""
TLabel V3.0 参考实现

本子包是规范 `tlabel-spec-v3.md` 的可执行参考，与 V2 实现（`tlabel.core`）并存、互不依赖。

模块：
    schema     字段定义、枚举、单位、状态、错误码
    validator  四层校验（L-A/L-B/L-C/L-D）
    migrate    V2 → V3 迁移

用法：
    from tlabel.v3 import validate, validate_frame

    report = validate(dataset, level="core")
    if not report.ok:
        for f in report.errors:
            print(f.code, f.clause, f.path, f.message)
"""

from .schema import (  # noqa: F401
    ALL_FIELDS,
    ALL_STATES,
    CORE_FIELDS,
    CORE_STATES,
    DEPRECATED_FIELDS,
    ERROR_CODES,
    EXTENSION_STATES,
    SPEC_VERSION,
)
from .validator import Finding, Report, validate, validate_frame  # noqa: F401

__all__ = [
    "ALL_FIELDS",
    "ALL_STATES",
    "CORE_FIELDS",
    "CORE_STATES",
    "DEPRECATED_FIELDS",
    "ERROR_CODES",
    "EXTENSION_STATES",
    "SPEC_VERSION",
    "Finding",
    "Report",
    "validate",
    "validate_frame",
]
