"""Dependency-free validation and serialization primitives."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


ID_PATTERNS = {
    "paper": re.compile(r"^pap_[a-z0-9][a-z0-9_]*$"),
    "claim": re.compile(r"^clm_[a-z0-9][a-z0-9_]*$"),
    "evidence": re.compile(r"^evd_[a-z0-9][a-z0-9_]*$"),
    "belief": re.compile(r"^blf_[a-z0-9][a-z0-9_]*$"),
    "observable": re.compile(r"^obs_[a-z0-9][a-z0-9_]*$"),
    "intervention": re.compile(r"^int_[a-z0-9][a-z0-9_]*$"),
    "context": re.compile(r"^ctx_[a-z0-9][a-z0-9_]*$"),
    "outcome": re.compile(r"^out_[a-z0-9][a-z0-9_]*$"),
    "tool_proposal": re.compile(r"^tlp_[a-z0-9][a-z0-9_]*$"),
    "capability_gap": re.compile(r"^gap_[a-z0-9][a-z0-9_]*$"),
    "mechanism": re.compile(r"^mech_[a-z0-9][a-z0-9_]*$"),
    "hypothesis": re.compile(r"^hyp_[a-z0-9][a-z0-9_]*$"),
    "idea": re.compile(r"^idea_[a-z0-9][a-z0-9_]*$"),
    "hourly_report": re.compile(r"^hrp_[a-z0-9][a-z0-9_]*$"),
    "experiment": re.compile(r"^exp_[a-z0-9][a-z0-9_]*$"),
    "run": re.compile(r"^run_[a-z0-9][a-z0-9_]*$"),
    "evidence_update": re.compile(r"^upd_[a-z0-9][a-z0-9_]*$"),
    "audit": re.compile(r"^aud_[a-z0-9][a-z0-9_]*$"),
    "decision": re.compile(r"^dec_[a-z0-9][a-z0-9_]*$"),
    "deprecation": re.compile(r"^dep_[a-z0-9][a-z0-9_]*$"),
    "campaign_batch": re.compile(r"^cmp_[a-z0-9][a-z0-9_]*$"),
}


class SchemaError(ValueError):
    """Raised when a research record violates a scientific data contract."""


def require_text(value: Any, field: str) -> str:
    result = str(value).strip() if value is not None else ""
    if not result:
        raise SchemaError(f"{field} must be non-empty")
    return result


def require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{field} must be boolean")
    return value


def require_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaError(f"{field} must be >= {minimum}")
    return value


def require_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{field} must be numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise SchemaError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise SchemaError(f"{field} must be <= {maximum}")
    return result


def require_enum(value: Any, field: str, choices: set[str]) -> str:
    result = require_text(value, field)
    if result not in choices:
        raise SchemaError(f"{field} must be one of {sorted(choices)}, got {result!r}")
    return result


def require_id(value: Any, kind: str, field: str | None = None) -> str:
    result = require_text(value, field or f"{kind}_id")
    try:
        pattern = ID_PATTERNS[kind]
    except KeyError as exc:
        raise SchemaError(f"unknown ID kind: {kind}") from exc
    if not pattern.fullmatch(result):
        raise SchemaError(f"{field or kind + '_id'} must match {pattern.pattern}")
    return result


def local_id(value: Any, field: str) -> str:
    result = require_text(value, field)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", result):
        raise SchemaError(f"{field} must use lowercase snake_case")
    return result


def json_mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        result: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise SchemaError(f"{field} must be an object")
    try:
        json.dumps(result, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field} must contain JSON-safe values") from exc
    return result


def json_mappings(value: Any, field: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SchemaError(f"{field} must be an array of objects")
    return tuple(json_mapping(item, f"{field}[{index}]") for index, item in enumerate(value))


def strings(value: Any, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if value is None:
        result: tuple[str, ...] = ()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(require_text(item, field) for item in value)
    else:
        raise SchemaError(f"{field} must be an array of strings")
    if not allow_empty and not result:
        raise SchemaError(f"{field} must be non-empty")
    if len(set(result)) != len(result):
        raise SchemaError(f"{field} values must be unique")
    return result


def tags(value: Any, field: str = "tags") -> tuple[str, ...]:
    result = strings(value, field)
    for item in result:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", item):
            raise SchemaError(f"{field} must use lowercase snake_case: {item!r}")
    return result


def require_keys(value: Mapping[str, Any], field: str, keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise SchemaError(f"{field} is missing required keys: {missing}")


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


def check_fingerprint(payload: Mapping[str, Any], actual: str) -> None:
    declared = payload.get("fingerprint")
    if declared is None:
        raise SchemaError("versioned records must declare fingerprint")
    if str(declared) != actual:
        raise SchemaError(
            f"declared fingerprint {declared!r} does not match computed {actual!r}"
        )


def atomic_write_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, destination)


__all__ = [
    "ID_PATTERNS",
    "SchemaError",
    "atomic_write_text",
    "canonical_json",
    "check_fingerprint",
    "fingerprint",
    "json_mapping",
    "json_mappings",
    "local_id",
    "require_bool",
    "require_enum",
    "require_id",
    "require_int",
    "require_keys",
    "require_number",
    "require_text",
    "strings",
    "tags",
]
