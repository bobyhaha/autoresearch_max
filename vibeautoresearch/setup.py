"""Reference-reconciliation records for the frozen experimental setup."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .core import (
    SchemaError,
    check_fingerprint,
    fingerprint,
    json_mapping,
    require_enum,
    require_id,
    require_int,
    require_keys,
    require_number,
    require_text,
    strings,
)


RECONCILIATION_STATUSES = {"pending", "passed", "failed"}
SCOPE_KEY_FIELDS = ("data_split_sha256", "max_steps", "stop_mode", "outcome_id")

# A scope is either step-budgeted (the adopt frame) or wall-clock budgeted.
# Wall-clock scopes additionally require an explicit positive time_budget, because
# tools/run_gated.py derives TIME_BUDGET from the scope key and must never invent one.
STOP_MODES = {"steps", "time", "walltime"}
TIME_STOP_MODES = {"time", "walltime"}

# Every registered scope is a DECISION FRAME in its own right. A change adopted in
# one frame is adopted IN THAT FRAME only: each carries its own baseline, its own
# sigma, and its own seed requirement. A `report` scope is the weaker variant that
# may be run and recorded but never promotes.
#
# The invariant that survives in BOTH cases: a comparison never crosses frames. A
# 5-minute result is never judged against a 2000-step baseline, and vice versa.
SCOPE_ROLES = {"adopt", "report"}
SCOPE_RUN_ENV_RESERVED = {
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "MAX_STEPS",
    "SEED",
    "STOP_MODE",
    "TIME_BUDGET",
}


def validate_scope_key(value: Any, field: str = "scope_key") -> dict[str, Any]:
    """Validate the portable identity of an experimental world."""
    scope = json_mapping(value, field)
    require_keys(scope, field, SCOPE_KEY_FIELDS)
    digest = require_text(scope["data_split_sha256"], f"{field}.data_split_sha256")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise SchemaError(f"{field}.data_split_sha256 must be a lowercase SHA-256 digest")
    require_int(scope["max_steps"], f"{field}.max_steps", minimum=11)
    stop_mode = require_enum(scope["stop_mode"], f"{field}.stop_mode", STOP_MODES)
    if stop_mode in TIME_STOP_MODES:
        if "time_budget" not in scope:
            raise SchemaError(
                f"{field}.time_budget is required when stop_mode is {stop_mode!r}"
            )
        require_int(scope["time_budget"], f"{field}.time_budget", minimum=1)
    elif "time_budget" in scope:
        raise SchemaError(
            f"{field}.time_budget is only meaningful for a wall-clock stop_mode"
        )
    require_id(scope["outcome_id"], "outcome", f"{field}.outcome_id")
    return scope


def validate_secondary_scope(value: Any, index: int) -> dict[str, Any]:
    """Validate one additional frame with its own baseline and run environment."""
    field = f"setup_reconciliation.scopes[{index}]"
    entry = json_mapping(value, field)
    require_keys(entry, field, ("scope_id", "role", "status", "scope_key", "baseline"))
    scope_id = require_text(entry["scope_id"], f"{field}.scope_id")
    require_enum(entry["role"], f"{field}.role", SCOPE_ROLES)
    status = require_enum(entry["status"], f"{field}.status", RECONCILIATION_STATUSES)
    validate_scope_key(entry["scope_key"], f"{field}.scope_key")
    run_env = json_mapping(entry.get("run_env"), f"{field}.run_env")
    for key, value in run_env.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(key)):
            raise SchemaError(f"{field}.run_env has an unsafe environment key {key!r}")
        if key in SCOPE_RUN_ENV_RESERVED:
            raise SchemaError(
                f"{field}.run_env may not override frame control {key!r}; "
                "put budget controls in scope_key"
            )
        require_text(value, f"{field}.run_env[{key!r}]")
    # A noisier frame needs MORE seeds, not the same number. A scope may raise its own
    # seed requirement above the global 3-seed minimum; it may never lower it.
    min_seeds = entry.get("min_seeds", 3)
    require_int(min_seeds, f"{field}.min_seeds", minimum=3)
    baseline = json_mapping(entry["baseline"], f"{field}.baseline")
    if baseline:
        seeds = baseline.get("seeds")
        if isinstance(seeds, list) and len(seeds) < int(min_seeds):
            raise SchemaError(
                f"{field}.baseline has {len(seeds)} seeds but this scope requires "
                f"min_seeds={min_seeds}"
            )
    if status == "passed":
        # A reporting scope may only claim 'passed' once it has its own measured
        # baseline; an unmeasured scope must stay 'pending' and block execution.
        _validate_baseline(baseline, f"{field}.baseline", status)
    elif baseline:
        _validate_baseline(baseline, f"{field}.baseline", status)
    return {**entry, "scope_id": scope_id, "run_env": run_env}


def _validate_baseline(baseline: Mapping[str, Any], field: str, status: str) -> None:
    """Shared baseline validation for the adopt scope and any reporting scope."""
    require_keys(
        baseline,
        field,
        (
            "metric",
            "expected",
            "observed",
            "effective_sigma",
            "tolerance_sigma",
            "seeds",
            "artifact_paths",
        ),
    )
    require_text(baseline["metric"], f"{field}.metric")
    expected = require_number(baseline["expected"], f"{field}.expected")
    observed = require_number(baseline["observed"], f"{field}.observed")
    sigma = require_number(baseline["effective_sigma"], f"{field}.effective_sigma", minimum=0)
    tolerance = require_number(baseline["tolerance_sigma"], f"{field}.tolerance_sigma", minimum=0)
    if sigma == 0:
        raise SchemaError(f"{field}.effective_sigma must be positive")
    seeds = baseline["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise SchemaError(f"{field} requires at least three seeds")
    for seed in seeds:
        require_int(seed, f"{field}.seed", minimum=0)
    if len(set(seeds)) != len(seeds):
        raise SchemaError(f"{field} seeds must be unique")
    strings(baseline["artifact_paths"], f"{field}.artifact_paths", allow_empty=False)
    if status == "passed" and abs(observed - expected) > tolerance * sigma:
        raise SchemaError(
            f"{field} cannot pass: baseline differs from the reference "
            f"by {abs(observed - expected):.6g}, above {tolerance:g}σ "
            f"({tolerance * sigma:.6g})"
        )


@dataclass(frozen=True)
class SetupReconciliationRecord:
    """A checked baseline/data/code reconciliation that precedes experiments."""

    version: int
    status: str
    scope_key: Mapping[str, Any]
    frozen_files: Mapping[str, Any]
    reference_code: Mapping[str, Any]
    baseline: Mapping[str, Any]
    reconciled_at: str
    reconciled_by: str
    limitations: tuple[str, ...] = ()
    scopes: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        require_int(self.version, "setup_reconciliation.version", minimum=1)
        require_enum(self.status, "setup_reconciliation.status", RECONCILIATION_STATUSES)
        scope = validate_scope_key(self.scope_key, "setup_reconciliation.scope_key")

        frozen_files = json_mapping(
            self.frozen_files, "setup_reconciliation.frozen_files"
        )
        if not frozen_files:
            raise SchemaError("setup_reconciliation.frozen_files must be non-empty")
        for relative_path, digest in frozen_files.items():
            require_text(relative_path, "setup_reconciliation.frozen_files path")
            digest_text = require_text(
                digest, f"setup_reconciliation.frozen_files[{relative_path!r}]"
            )
            if len(digest_text) != 64 or any(
                char not in "0123456789abcdef" for char in digest_text
            ):
                raise SchemaError(
                    f"setup_reconciliation.frozen_files[{relative_path!r}] "
                    "must be a lowercase SHA-256 digest"
                )
        if frozen_files.get("data_split.json") != scope["data_split_sha256"]:
            raise SchemaError(
                "setup scope data_split_sha256 must match frozen_files[data_split.json]"
            )

        reference = json_mapping(
            self.reference_code, "setup_reconciliation.reference_code"
        )
        require_keys(
            reference,
            "setup_reconciliation.reference_code",
            (
                "local_path",
                "local_sha256",
                "upstream_url",
                "upstream_sha256",
                "comparison",
                "diff",
            ),
        )
        for key in ("local_path", "local_sha256", "upstream_url", "upstream_sha256"):
            require_text(reference[key], f"setup_reconciliation.reference_code.{key}")
        for key in ("local_sha256", "upstream_sha256"):
            digest_text = str(reference[key])
            if len(digest_text) != 64 or any(
                char not in "0123456789abcdef" for char in digest_text
            ):
                raise SchemaError(
                    f"setup_reconciliation.reference_code.{key} must be a lowercase "
                    "SHA-256 digest"
                )
        require_enum(
            reference["comparison"],
            "setup_reconciliation.reference_code.comparison",
            {"identical", "fork_with_differences"},
        )
        diff = json_mapping(reference["diff"], "setup_reconciliation.reference_code.diff")
        require_keys(diff, "setup_reconciliation.reference_code.diff", ("added", "removed", "hunks"))
        for key in ("added", "removed", "hunks"):
            require_int(diff[key], f"setup_reconciliation.reference_code.diff.{key}", minimum=0)
        if reference["comparison"] == "identical" and any(diff.values()):
            raise SchemaError("identical reference code must have a zero diff")
        if reference["comparison"] == "fork_with_differences" and not any(diff.values()):
            raise SchemaError("fork_with_differences must report a non-zero diff")

        baseline = json_mapping(self.baseline, "setup_reconciliation.baseline")
        _validate_baseline(baseline, "setup_reconciliation.baseline", self.status)

        # Additional decision frames (e.g. the 5-minute wall-clock frame). Each
        # carries its own baseline and may adopt only within its own scope.
        seen_scope_ids: set[str] = set()
        adopt_stop_mode = str(scope["stop_mode"])
        for index, entry in enumerate(self.scopes):
            validated = validate_secondary_scope(entry, index)
            scope_id = validated["scope_id"]
            if scope_id in seen_scope_ids:
                raise SchemaError(
                    f"setup_reconciliation.scopes duplicate scope_id {scope_id!r}"
                )
            seen_scope_ids.add(scope_id)
            secondary_key = validated["scope_key"]
            if secondary_key.get("data_split_sha256") != scope.get("data_split_sha256"):
                raise SchemaError(
                    f"setup_reconciliation.scopes[{index}] must share the adopt scope's "
                    "data split: comparing across data content is not a fair lever"
                )
            if str(secondary_key["stop_mode"]) == adopt_stop_mode:
                raise SchemaError(
                    f"setup_reconciliation.scopes[{index}] duplicates the adopt scope's "
                    f"stop_mode {adopt_stop_mode!r}; a reporting scope must define a "
                    "different budget frame"
                )

        require_text(self.reconciled_at, "setup_reconciliation.reconciled_at")
        require_text(self.reconciled_by, "setup_reconciliation.reconciled_by")
        strings(self.limitations, "setup_reconciliation.limitations")

    def definition(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "status": self.status,
            "scope_key": dict(self.scope_key),
            "frozen_files": dict(self.frozen_files),
            "reference_code": dict(self.reference_code),
            "baseline": dict(self.baseline),
            "reconciled_at": self.reconciled_at,
            "reconciled_by": self.reconciled_by,
            "limitations": list(self.limitations),
        }
        # Only fingerprint `scopes` when present, so a single-scope record keeps the
        # fingerprint it was frozen with and no historical reconciliation is invalidated.
        if self.scopes:
            payload["scopes"] = [dict(entry) for entry in self.scopes]
        return payload

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {**self.definition(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SetupReconciliationRecord":
        record = cls(
            version=int(payload.get("version", 1)),
            status=str(payload.get("status", "pending")),
            scope_key=json_mapping(
                payload.get("scope_key"), "setup_reconciliation.scope_key"
            ),
            frozen_files=json_mapping(
                payload.get("frozen_files"), "setup_reconciliation.frozen_files"
            ),
            reference_code=json_mapping(
                payload.get("reference_code"), "setup_reconciliation.reference_code"
            ),
            baseline=json_mapping(
                payload.get("baseline"), "setup_reconciliation.baseline"
            ),
            reconciled_at=require_text(
                payload.get("reconciled_at"), "setup_reconciliation.reconciled_at"
            ),
            reconciled_by=require_text(
                payload.get("reconciled_by"), "setup_reconciliation.reconciled_by"
            ),
            limitations=strings(
                payload.get("limitations"), "setup_reconciliation.limitations"
            ),
            scopes=tuple(
                json_mapping(entry, f"setup_reconciliation.scopes[{index}]")
                for index, entry in enumerate(payload.get("scopes") or ())
            ),
        )
        check_fingerprint(payload, record.fingerprint)
        return record

    def scope_for(self, scope_id: str) -> dict[str, Any]:
        """Return a registered reporting scope by id, or raise."""
        for entry in self.scopes:
            if str(entry.get("scope_id")) == scope_id:
                return dict(entry)
        known = ", ".join(sorted(str(e.get("scope_id")) for e in self.scopes)) or "none"
        raise SchemaError(
            f"unknown reporting scope {scope_id!r}; registered scopes: {known}"
        )

    @property
    def report_scopes(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(entry) for entry in self.scopes)

    def verify_frozen_files(self, project_root: Path) -> tuple[str, ...]:
        """Return deterministic setup-drift errors without touching the filesystem."""
        errors: list[str] = []
        for relative_path, expected_digest in self.frozen_files.items():
            path = project_root / relative_path
            if not path.is_file():
                errors.append(f"missing frozen setup file: {relative_path}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected_digest:
                errors.append(
                    f"frozen setup file drift: {relative_path} expected "
                    f"{expected_digest}, found {actual}"
                )
        for artifact_path in self.baseline["artifact_paths"]:
            if "://" not in artifact_path and not (project_root / artifact_path).exists():
                errors.append(f"missing baseline reconciliation artifact: {artifact_path}")
        # Freeze the training code: the declared reference_code file must exist AND
        # match its hash, otherwise train.py could change (or be absent) while the
        # gate keeps passing. Fixtures now ship a matching train.py, so this is
        # strict — a missing or drifted reference file is an error.
        ref = self.reference_code
        local_path = ref.get("local_path")
        expected = ref.get("local_sha256")
        if local_path and expected and local_path not in self.frozen_files:
            code_path = project_root / local_path
            if not code_path.is_file():
                errors.append(f"missing reference-code file: {local_path}")
            else:
                actual = hashlib.sha256(code_path.read_bytes()).hexdigest()
                if actual != expected:
                    errors.append(
                        f"reference-code drift: {local_path} expected {expected}, "
                        f"found {actual} — re-reconcile the setup before running"
                    )
        return tuple(errors)


__all__ = [
    "RECONCILIATION_STATUSES",
    "SCOPE_RUN_ENV_RESERVED",
    "SCOPE_KEY_FIELDS",
    "SetupReconciliationRecord",
    "validate_scope_key",
]
