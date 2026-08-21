#!/usr/bin/env python3
"""Fail-closed runtime orchestrator for Paper-020 Round-1 GPAS.

This file is deliberately separate from registry construction.  It consumes a
committed, detached runtime-authority JSON document and never creates or edits
paper, toolkit, idea, experiment, run, evidence, setup, chart, or SOTA records.

The default invocation is a read-only dry run.  Execution additionally requires
``--execute`` and the authority's one-time nonce on the command line.  Even then,
the runner refuses unless:

* the exact Paper-020 source/setup bindings and current R0 recipe match;
* the authority, runner, and separately supplied no-validation worker are
  committed at the authority's exact Git commit;
* one replay H200 and two fixed timing H200s have exact index/UUID bindings;
* every PACKER_* diagnostic flag is zero; and
* output is a new path below ``research/experiments/artifacts``.

The worker is a deliberately separate capability boundary.  ``train.py`` is
not an admissible worker because its normal terminal path evaluates validation
BPB.  A worker conforming to the JSON protocol below must stop before all
validation access and expose the helper-produced parity/variance attestations.
This runner supervises that worker, including exact PID-set checks at every
GPU inventory sample, and evaluates only training-time mechanism facts.

The frozen order is:

1. same-H200 sequential step-0 control then alpha-zero treatment replay;
2. one concurrent 256-step eager treatment/control mediator pair, whose first
   32 control rows are the applicability gate; and
3. two clean compiled 256-step timing placements with roles swapped on the
   same two H200 UUIDs.

No phase reads or reports validation BPB.  A passing diagnostic authorizes only
the separately governed endpoint proposal described by Paper-020.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import select
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "vibeautoresearch").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("could not locate vibeautoresearch repository root")
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vibeautoresearch.registry import ResearchRegistry  # noqa: E402

AUTHORITY_KIND = "paper020_gpas_round1_runtime_authority"
AUTHORITY_SCHEMA_VERSION = 1
WORKER_PROTOCOL_VERSION = 1
EXPERIMENT_ID = "exp_paper020_gpas_attribution_diagnostic_s66"
DIAGNOSTIC_SEED = 66
SETUP_VERSION = 40
SETUP_FINGERPRINT = "7e2b06b51256f737"
CHALLENGE_SELECTION_FINGERPRINT = "74f0a10788bf42f0"

REMOTE_EXECUTABLE_FILES: dict[str, str] = {
    "train.py": "6dc22cc5351dd37d9473d74bbbb2402b61c2db6477e302be5f87017a62b37906",
    "lib.py": "05691d87f253b0f7692d2da40a909d6fed6ec83ea37f3e890a687f874ebf49c4",
    "observable.py": (
        "0d24adfcb0ac529b6f520b675d9321f115f3990a0f67582a439e816b856cfb87"
    ),
    "data_split.json": (
        "ed8ea0554010df9fbfa47746adc6643d6446797753eb377c0b35413bdcf2fca3"
    ),
    "tools/gpas_mechanism_diagnostic.py": (
        "c1f31f046eac55eefb2c59c849e75db336ee75be278e7d7859d721e3a67eb437"
    ),
}
LOCAL_GOVERNANCE_FILES: dict[str, str] = {
    "tools/register_paper020_gpas_chain.py": (
        "af3f78319079c7529e8aeb88d46af6ad22fef3f8ebe0584aba38c6bb259582b1"
    ),
    "tests/test_register_paper020_gpas_chain.py": (
        "e85ff4c1e4bbe08661042eadea931cd1a0de1be8fcfd5108839025b2ff476334"
    ),
    "tests/test_gpas_impl.py": (
        "98c11af51af600d5f97ab874f219659987186755debb2ecb99046c2246856cb1"
    ),
    "research/setup/reconciliation.json": (
        "4acc742d43cc2736b31fbc47a355177b97926e709523098290fc138c8d5101c5"
    ),
    "vibeautoresearch/registry.py": (
        "b0f9b5dc4049076e9e7cafde0ca31dd34c9a33c611cb770221995f047263fb65"
    ),
    "tools/run_gated.py": (
        "12acd281bc1b79622c858e4c67a18afc710a8f604daa68ac7fd7ee7611ce421c"
    ),
    "tests/test_run_gated.py": (
        "bdfe39c2721ceaf1c554882f382fa67a44dd331e97dad3d5abe84fe2ad109aa9"
    ),
    "tests/test_diagnostic_pilot_authority.py": (
        "d15da7a0f94d4db59d3b2ce0cfea5c5727aa894cd53cedb141ce329a164f8039"
    ),
    (
        "AI_papers/"
        "paper_020_from_failed_attribution_to_local_mechanism_assays_20260729.tex"
    ): "c9053c401cb2aa53027aa457bad4bf6ebe439af65b4cdae04eac190bed36e592",
    (
        "AI_papers/"
        "paper_020_from_failed_attribution_to_local_mechanism_assays_20260729.pdf"
    ): "1ebb95deee3f489b8016a0d28115e9aa4ec7725e7534deee93011d66725db9bb",
}
BOUND_FILES: dict[str, str] = {
    **REMOTE_EXECUTABLE_FILES,
    **LOCAL_GOVERNANCE_FILES,
}

CURRENT_R0_ENV: dict[str, str] = {
    "ATTN_BACKEND": "fa3",
    "COMPILE_MODE": "max-autotune-no-cudagraphs",
    "DEVICE_BATCH_SIZE": "128",
    "DOC_MASK": "1",
    "DOC_MASK_IMPL": "varlen",
    "DOC_MASK_MODE": "both",
    "MATRIX_LR": "0.03",
    "MUON_MOMENTUM_CONTINUOUS": "0",
    "NGRAM_TABLE_MULT": "64",
    "SCALAR_LR": "0.8",
    "TOTAL_BATCH_SIZE": "262144",
    "WARMDOWN_RATIO": "0.95",
    "WINDOW_PATTERN": "SSSL",
}
DATA_AUTHORITY: dict[str, Any] = {
    "data_split_path": "data_split.json",
    "data_split_sha256": (
        "ed8ea0554010df9fbfa47746adc6643d6446797753eb377c0b35413bdcf2fca3"
    ),
    "training_shard_ids": list(range(1, 11)),
    "held_out_shard_ids": [6542],
    "training_only": True,
    "held_out_access_authorized": False,
}

PACKER_ZERO_ENV: dict[str, str] = {
    "PACKER_DOC_BOUNDARIES": "0",
    "PACKER_BOUNDARY_VERIFY_BATCHES": "0",
    "PACKER_DIAGNOSTIC_HASHES": "0",
    "PACKER_SIDECAR_ACTIVATE_STEP": "0",
}

# This is the historical experiment-501 configuration family, included only to
# produce a specific refusal rather than a generic recipe mismatch.  Exact R0
# equality below rejects every partial or complete drift as well.
FORBIDDEN_0927_RECIPE: dict[str, str] = {
    "WINDOW_PATTERN": "TTTL",
    "DEVICE_BATCH_SIZE": "72",
    "TOTAL_BATCH_SIZE": "147456",
    "MATRIX_LR": "0.04",
    "MUON_MOMENTUM_CONTINUOUS": "1",
    "WARMDOWN_RATIO": "0.85",
}

RESULT_ROOT = (REPO_ROOT / "research/experiments/artifacts").resolve()
RUNNER_RELATIVE_PATH = "tools/run_paper020_gpas_runtime.py"
EXPECTED_WORKER_RELATIVE_PATH = "tools/run_paper020_gpas_worker.py"
EXPECTED_REMOTE_WORK_DIR = (
    "/home/user/ai4ai/ophis_autoresearch_12h_20260729_0810z"
)
EXPECTED_REMOTE_PYTHON = "/home/user/ph/autoresearch/.venv/bin/python"
EXPECTED_REMOTE_TARGET = "user@223.167.85.180"
EXPECTED_KNOWN_HOSTS_PATH = "/Users/baiyu/.ssh/known_hosts"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{16}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,127}$")
GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")
WORKER_COMMAND_TEMPLATE = (
    "{python}",
    "{worker}",
    "--phase-manifest",
    "{manifest}",
    "--result",
    "{result}",
)

FORBIDDEN_RESULT_KEYS = {
    "bpb",
    "best_val_bpb",
    "endpoint_bpb",
    "endpoint_val_bpb",
    "local_sota",
    "sota",
    "val_bpb",
    "val_loss",
    "validation_bpb",
    "validation_loss",
}
FORBIDDEN_OUTPUT_RE = re.compile(
    r"(?i)(?:\bbpb\b|\bval_bpb\b|\bval_loss\b|\bvalidation_(?:bpb|loss)\b|"
    r"\bbest_val_bpb\b|\blocal_sota\b|\bsota\b)"
)


class AuthorityError(RuntimeError):
    """The detached runtime authority is absent, stale, or unsafe."""


class RuntimeIntegrityError(RuntimeError):
    """A runtime identity, parity, mediator, or safety check failed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise AuthorityError(f"required file is missing: {path}")
    return _sha256_bytes(path.read_bytes())


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthorityError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_unique_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AuthorityError(f"cannot read authority {path}: {exc}") from exc
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AuthorityError(f"non-finite JSON constant {value!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"authority is not canonical UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuthorityError("authority root must be an object")
    canonical = _canonical_json(payload)
    if raw != canonical:
        raise AuthorityError(
            "authority bytes must equal canonical sorted compact JSON plus one LF"
        )
    return payload, raw


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    where: str,
) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise AuthorityError(
            f"{where} keys differ: missing={missing} extra={extra}"
        )


def _require_bool(value: Any, expected: bool, where: str) -> None:
    if value is not expected:
        raise AuthorityError(f"{where} must be exactly {expected}")


def _require_string_map(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise AuthorityError(f"{where} must be an object")
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise AuthorityError(f"{where} must contain only string keys and values")
    return dict(value)


def _require_safe_argv(value: Any, where: str, executable: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise AuthorityError(f"{where} must be a non-empty string array")
    if Path(value[0]).name != executable:
        raise AuthorityError(f"{where}[0] must resolve to {executable!r}")
    if any("\x00" in item or "\n" in item or "\r" in item for item in value):
        raise AuthorityError(f"{where} contains an unsafe control character")
    return tuple(value)


def _resolve_repo_file(relative: str, where: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise AuthorityError(f"{where} must be a non-empty repository-relative path")
    candidate = (REPO_ROOT / relative).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise AuthorityError(f"{where} resolves outside the repository") from exc
    if candidate.is_symlink():
        raise AuthorityError(f"{where} may not be a symlink")
    return candidate


def _validate_device(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuthorityError(f"{where} must be an object")
    _require_exact_keys(value, {"index", "uuid", "product"}, where)
    index = value["index"]
    uuid = value["uuid"]
    product = value["product"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise AuthorityError(f"{where}.index must be a non-negative integer")
    if not isinstance(uuid, str) or not GPU_UUID_RE.fullmatch(uuid):
        raise AuthorityError(f"{where}.uuid must be an exact NVIDIA GPU UUID")
    if product != "NVIDIA H200":
        raise AuthorityError(f"{where}.product must be exactly 'NVIDIA H200'")
    return {"index": index, "uuid": uuid, "product": product}


def _validate_authority_payload(
    payload: Mapping[str, Any],
    *,
    authority_path: Path,
    authority_sha256: str,
) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "kind",
        "authority_id",
        "experiment_id",
        "experiment_fingerprint",
        "git_commit",
        "seed",
        "diagnostic_only",
        "validation_data_access",
        "endpoint_scoring_authorized",
        "sota_update_authorized",
        "single_use_nonce",
        "bindings",
        "governance",
        "recipe_env",
        "packer_env",
        "gpus",
        "remote",
        "worker",
        "result_dir",
    }
    _require_exact_keys(payload, expected_top, "authority")
    if payload["schema_version"] != AUTHORITY_SCHEMA_VERSION:
        raise AuthorityError("unsupported authority schema_version")
    if payload["kind"] != AUTHORITY_KIND:
        raise AuthorityError(f"authority.kind must equal {AUTHORITY_KIND!r}")
    authority_id = payload["authority_id"]
    if not isinstance(authority_id, str) or not ID_RE.fullmatch(authority_id):
        raise AuthorityError("authority_id is not a safe stable identifier")
    if payload["experiment_id"] != EXPERIMENT_ID:
        raise AuthorityError(f"experiment_id must equal {EXPERIMENT_ID!r}")
    if (
        not isinstance(payload["experiment_fingerprint"], str)
        or not FINGERPRINT_RE.fullmatch(payload["experiment_fingerprint"])
    ):
        raise AuthorityError("experiment_fingerprint must be 16 lowercase hex digits")
    if not isinstance(payload["git_commit"], str) or not COMMIT_RE.fullmatch(
        payload["git_commit"]
    ):
        raise AuthorityError("git_commit must be a full lowercase SHA-1")
    if payload["seed"] != DIAGNOSTIC_SEED:
        raise AuthorityError(f"seed must equal {DIAGNOSTIC_SEED}")
    _require_bool(payload["diagnostic_only"], True, "diagnostic_only")
    _require_bool(payload["validation_data_access"], False, "validation_data_access")
    _require_bool(
        payload["endpoint_scoring_authorized"],
        False,
        "endpoint_scoring_authorized",
    )
    _require_bool(payload["sota_update_authorized"], False, "sota_update_authorized")
    nonce = payload["single_use_nonce"]
    if not isinstance(nonce, str) or not SHA256_RE.fullmatch(nonce):
        raise AuthorityError("single_use_nonce must be 64 lowercase hex digits")

    bindings = payload["bindings"]
    if not isinstance(bindings, dict):
        raise AuthorityError("bindings must be an object")
    _require_exact_keys(
        bindings,
        {
            "files",
            "runner_sha256",
            "setup_version",
            "setup_fingerprint",
            "challenge_selection_fingerprint",
        },
        "bindings",
    )
    bound_files = _require_string_map(bindings["files"], "bindings.files")
    if bound_files != BOUND_FILES:
        raise AuthorityError(
            "bindings.files must exactly equal the Paper-020 audited source map"
        )
    if (
        not isinstance(bindings["runner_sha256"], str)
        or not SHA256_RE.fullmatch(bindings["runner_sha256"])
    ):
        raise AuthorityError("bindings.runner_sha256 must be lowercase SHA-256")
    if bindings["setup_version"] != SETUP_VERSION:
        raise AuthorityError(f"bindings.setup_version must equal {SETUP_VERSION}")
    if bindings["setup_fingerprint"] != SETUP_FINGERPRINT:
        raise AuthorityError("bindings.setup_fingerprint is stale")
    if (
        bindings["challenge_selection_fingerprint"]
        != CHALLENGE_SELECTION_FINGERPRINT
    ):
        raise AuthorityError("bindings.challenge_selection_fingerprint is stale")

    governance = payload["governance"]
    if not isinstance(governance, dict):
        raise AuthorityError("governance must be an object")
    _require_exact_keys(
        governance,
        {
            "scope_id",
            "frame_max_steps",
            "gated_ledger_path",
            "gated_ledger_sha256",
            "control_arm_id",
            "treatment_arm_id",
            "control_authorization_sha256",
            "treatment_authorization_sha256",
            "diagnostic_policy_sha256",
        },
        "governance",
    )
    expected_governance_scalars = {
        "scope_id": "walltime_5min_h200",
        "frame_max_steps": 100000,
        "gated_ledger_path": "research/experiments/gated/experiments.jsonl",
        "control_arm_id": "gpas_disabled_control",
        "treatment_arm_id": "gpas_shared_gate_treatment",
    }
    for key, expected in expected_governance_scalars.items():
        if governance[key] != expected:
            raise AuthorityError(
                f"governance.{key} must equal {expected!r}"
            )
    for key in (
        "gated_ledger_sha256",
        "control_authorization_sha256",
        "treatment_authorization_sha256",
        "diagnostic_policy_sha256",
    ):
        if not isinstance(governance[key], str) or not SHA256_RE.fullmatch(
            governance[key]
        ):
            raise AuthorityError(f"governance.{key} must be lowercase SHA-256")

    recipe_env = _require_string_map(payload["recipe_env"], "recipe_env")
    historical_matches = {
        key: value
        for key, value in FORBIDDEN_0927_RECIPE.items()
        if recipe_env.get(key) == value
    }
    if historical_matches:
        raise AuthorityError(
            "historical experiment-501/0.927183 recipe values are forbidden: "
            f"{historical_matches}"
        )
    if recipe_env != CURRENT_R0_ENV:
        raise AuthorityError(
            "recipe_env must exactly equal current R0 "
            "(SSSL/DBS128/TBS262144/MLR0.03/momentum-continuous0/warmdown0.95)"
        )
    packer_env = _require_string_map(payload["packer_env"], "packer_env")
    if packer_env != PACKER_ZERO_ENV:
        raise AuthorityError("every PACKER_* diagnostic flag must resolve to zero")

    gpus = payload["gpus"]
    if not isinstance(gpus, dict):
        raise AuthorityError("gpus must be an object")
    _require_exact_keys(gpus, {"replay", "timing_a", "timing_b"}, "gpus")
    replay = _validate_device(gpus["replay"], "gpus.replay")
    timing_a = _validate_device(gpus["timing_a"], "gpus.timing_a")
    timing_b = _validate_device(gpus["timing_b"], "gpus.timing_b")
    if timing_a["index"] == timing_b["index"] or timing_a["uuid"] == timing_b["uuid"]:
        raise AuthorityError("timing_a and timing_b must be distinct physical H200s")
    if replay["uuid"] not in {timing_a["uuid"], timing_b["uuid"]}:
        raise AuthorityError(
            "replay GPU must be one of the same two fixed timing H200 UUIDs"
        )
    if replay["uuid"] == timing_a["uuid"] and replay["index"] != timing_a["index"]:
        raise AuthorityError("replay/timing_a UUID has inconsistent NVML index")
    if replay["uuid"] == timing_b["uuid"] and replay["index"] != timing_b["index"]:
        raise AuthorityError("replay/timing_b UUID has inconsistent NVML index")

    remote = payload["remote"]
    if not isinstance(remote, dict):
        raise AuthorityError("remote must be an object")
    _require_exact_keys(
        remote,
        {
            "ssh_argv",
            "scp_argv",
            "scp_target",
            "work_dir",
            "python",
            "python_sha256",
            "known_hosts_path",
            "known_hosts_sha256",
        },
        "remote",
    )
    ssh_argv = _require_safe_argv(remote["ssh_argv"], "remote.ssh_argv", "ssh")
    scp_argv = _require_safe_argv(remote["scp_argv"], "remote.scp_argv", "scp")
    hardened_options = (
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={EXPECTED_KNOWN_HOSTS_PATH}",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    )
    if ssh_argv != (
        "ssh",
        "-p",
        "50002",
        *hardened_options,
        EXPECTED_REMOTE_TARGET,
    ):
        raise AuthorityError(
            "remote.ssh_argv must equal the audited noninteractive, "
            "strict-host-key, timeout-bounded endpoint"
        )
    if scp_argv != ("scp", "-P", "50002", *hardened_options):
        raise AuthorityError(
            "remote.scp_argv must equal the audited noninteractive, "
            "strict-host-key, timeout-bounded transport"
        )
    for key in ("scp_target", "work_dir", "python"):
        if (
            not isinstance(remote[key], str)
            or not remote[key]
            or "\x00" in remote[key]
            or "\n" in remote[key]
        ):
            raise AuthorityError(f"remote.{key} must be a safe non-empty string")
    if not remote["work_dir"].startswith("/") or not remote["python"].startswith("/"):
        raise AuthorityError("remote work_dir and python must be absolute paths")
    if remote["scp_target"] != EXPECTED_REMOTE_TARGET:
        raise AuthorityError(
            f"remote.scp_target must equal {EXPECTED_REMOTE_TARGET}"
        )
    if remote["work_dir"] != EXPECTED_REMOTE_WORK_DIR:
        raise AuthorityError(
            f"remote.work_dir must equal the audited checkout {EXPECTED_REMOTE_WORK_DIR!r}"
        )
    if remote["python"] != EXPECTED_REMOTE_PYTHON:
        raise AuthorityError(
            f"remote.python must equal the qualified interpreter {EXPECTED_REMOTE_PYTHON!r}"
        )
    if remote["known_hosts_path"] != EXPECTED_KNOWN_HOSTS_PATH:
        raise AuthorityError(
            "remote.known_hosts_path must equal the audited absolute path"
        )
    if (
        not isinstance(remote["known_hosts_sha256"], str)
        or not SHA256_RE.fullmatch(remote["known_hosts_sha256"])
    ):
        raise AuthorityError("remote.known_hosts_sha256 must be lowercase SHA-256")
    if (
        not isinstance(remote["python_sha256"], str)
        or not SHA256_RE.fullmatch(remote["python_sha256"])
    ):
        raise AuthorityError("remote.python_sha256 must be lowercase SHA-256")

    worker = payload["worker"]
    if not isinstance(worker, dict):
        raise AuthorityError("worker must be an object")
    _require_exact_keys(
        worker,
        {
            "local_path",
            "remote_path",
            "sha256",
            "protocol_version",
            "command_template",
            "command_template_sha256",
        },
        "worker",
    )
    if worker["protocol_version"] != WORKER_PROTOCOL_VERSION:
        raise AuthorityError("worker.protocol_version is unsupported")
    worker_path = _resolve_repo_file(worker["local_path"], "worker.local_path")
    if worker_path == REPO_ROOT / "train.py":
        raise AuthorityError(
            "train.py is forbidden as the worker because it performs terminal validation"
        )
    if worker_path == REPO_ROOT / "tools/gpas_mechanism_diagnostic.py":
        raise AuthorityError(
            "gpas_mechanism_diagnostic.py is a helper, not a no-validation worker"
        )
    if worker["local_path"] != EXPECTED_WORKER_RELATIVE_PATH:
        raise AuthorityError(
            "worker.local_path must equal the separately reviewed "
            f"{EXPECTED_WORKER_RELATIVE_PATH!r}"
        )
    if (
        not isinstance(worker["sha256"], str)
        or not SHA256_RE.fullmatch(worker["sha256"])
    ):
        raise AuthorityError("worker.sha256 must be lowercase SHA-256")
    if not isinstance(worker["remote_path"], str) or not worker[
        "remote_path"
    ].startswith("/"):
        raise AuthorityError("worker.remote_path must be absolute")
    expected_remote_worker = (
        f"{remote['work_dir'].rstrip('/')}/{EXPECTED_WORKER_RELATIVE_PATH}"
    )
    if worker["remote_path"] != expected_remote_worker:
        raise AuthorityError(
            f"worker.remote_path must equal {expected_remote_worker!r}"
        )
    if worker["command_template"] != list(WORKER_COMMAND_TEMPLATE):
        raise AuthorityError(
            "worker.command_template must equal the frozen six-token protocol"
        )
    command_template_sha256 = _sha256_bytes(
        _canonical_json(list(WORKER_COMMAND_TEMPLATE))
    )
    if worker["command_template_sha256"] != command_template_sha256:
        raise AuthorityError("worker.command_template_sha256 differs from template bytes")

    result_dir_raw = payload["result_dir"]
    if not isinstance(result_dir_raw, str) or not result_dir_raw:
        raise AuthorityError("result_dir must be a non-empty repository-relative path")
    result_dir = _resolve_repo_file(result_dir_raw, "result_dir")
    try:
        result_dir.relative_to(RESULT_ROOT)
    except ValueError as exc:
        raise AuthorityError(
            "result_dir must be below research/experiments/artifacts"
        ) from exc
    required_suffix = f"{authority_id}-{nonce[:16]}"
    if result_dir.name != required_suffix:
        raise AuthorityError(
            f"result_dir basename must equal {required_suffix!r}"
        )

    return {
        "authority_path": authority_path,
        "authority_sha256": authority_sha256,
        "authority_id": authority_id,
        "experiment_fingerprint": payload["experiment_fingerprint"],
        "git_commit": payload["git_commit"],
        "nonce": nonce,
        "bindings": dict(bindings),
        "governance": dict(governance),
        "recipe_env": recipe_env,
        "packer_env": packer_env,
        "gpus": {
            "replay": replay,
            "timing_a": timing_a,
            "timing_b": timing_b,
        },
        "remote": {
            **dict(remote),
            "ssh_argv": ssh_argv,
            "scp_argv": scp_argv,
        },
        "worker": {**dict(worker), "local_resolved": worker_path},
        "result_dir": result_dir,
    }


def load_authority(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(expected_sha256):
        raise AuthorityError("--authority-sha256 must be 64 lowercase hex digits")
    payload, raw = _read_unique_json(path)
    actual = _sha256_bytes(raw)
    if actual != expected_sha256:
        raise AuthorityError(
            f"authority SHA-256 mismatch: expected={expected_sha256} actual={actual}"
        )
    return _validate_authority_payload(
        payload,
        authority_path=path.resolve(),
        authority_sha256=actual,
    )


def _git_output(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise AuthorityError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _git_blob_sha256(commit: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise AuthorityError(
            f"{relative} is not committed at authority commit {commit}"
        )
    return _sha256_bytes(result.stdout)


def _governance_blockers(authority: Mapping[str, Any]) -> list[str]:
    """Re-run diagnostic-only arm authorization against the exact gated record."""

    blockers: list[str] = []
    governance = authority["governance"]
    ledger_path = REPO_ROOT / governance["gated_ledger_path"]
    try:
        ledger_sha = _sha256(ledger_path)
    except AuthorityError as exc:
        return [str(exc)]
    if ledger_sha != governance["gated_ledger_sha256"]:
        blockers.append(
            "gated experiment ledger differs from detached authority: "
            f"expected={governance['gated_ledger_sha256']} actual={ledger_sha}"
        )

    registry = ResearchRegistry(REPO_ROOT / "research")
    try:
        gated = registry.gated_experiments.by_id()
    except Exception as exc:
        blockers.append(f"cannot load gated experiment registry: {exc}")
        return blockers
    experiment = gated.get(EXPERIMENT_ID)
    if experiment is None:
        blockers.append(f"{EXPERIMENT_ID} is absent from the gated registry")
        return blockers
    if experiment.fingerprint != authority["experiment_fingerprint"]:
        blockers.append(
            "gated experiment fingerprint differs from detached authority: "
            f"expected={authority['experiment_fingerprint']} "
            f"actual={experiment.fingerprint}"
        )
    if experiment.status not in {"approved", "running"}:
        blockers.append(
            f"gated experiment status {experiment.status!r} is not runnable"
        )
    if experiment.stage != "pilot" or experiment.search_policy:
        blockers.append(
            "gated experiment is not an implementation-only diagnostic pilot"
        )
    if tuple(experiment.seeds) != (DIAGNOSTIC_SEED,):
        blockers.append(
            f"gated experiment seeds differ from exactly ({DIAGNOSTIC_SEED},)"
        )
    analysis = experiment.analysis_plan or {}
    if (
        analysis.get("diagnostic_only") is not True
        or analysis.get("validation_data_access") is not False
        or analysis.get("endpoint_val_bpb_measured") is not False
    ):
        blockers.append(
            "gated experiment analysis plan does not forbid endpoint access"
        )

    authorization_reports: dict[str, Mapping[str, Any]] = {}
    for role, arm_key in (
        ("control", "control_arm_id"),
        ("treatment", "treatment_arm_id"),
    ):
        arm_id = governance[arm_key]
        try:
            report = registry.authorize_run(
                EXPERIMENT_ID,
                arm_id,
                DIAGNOSTIC_SEED,
                governance["frame_max_steps"],
                governance["scope_id"],
                diagnostic_only=True,
            )
        except Exception as exc:
            blockers.append(
                f"diagnostic-only authorize_run refused {role} arm: {exc}"
            )
            continue
        authorization_reports[role] = report
        expected_hash = governance[f"{role}_authorization_sha256"]
        actual_hash = _sha256_bytes(_canonical_json(report))
        if actual_hash != expected_hash:
            blockers.append(
                f"{role} diagnostic authorization report drift: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        if (
            report.get("authorized") is not True
            or report.get("diagnostic_only_authority") is not True
            or report.get("endpoint_scoring_authorized") is not False
            or report.get("experiment_fingerprint")
            != authority["experiment_fingerprint"]
            or report.get("setup_fingerprint") != SETUP_FINGERPRINT
            or report.get("challenge_selection_fingerprint")
            != CHALLENGE_SELECTION_FINGERPRINT
        ):
            blockers.append(
                f"{role} diagnostic authorization report is not fail-closed"
            )

    if len(authorization_reports) == 2:
        control_policy = authorization_reports["control"].get("search_policy")
        treatment_policy = authorization_reports["treatment"].get("search_policy")
        if control_policy != treatment_policy:
            blockers.append("control/treatment diagnostic policy reports differ")
        else:
            policy_hash = _sha256_bytes(_canonical_json(control_policy))
            if policy_hash != governance["diagnostic_policy_sha256"]:
                blockers.append(
                    "diagnostic policy report drift: "
                    f"expected={governance['diagnostic_policy_sha256']} "
                    f"actual={policy_hash}"
                )
            if (
                not isinstance(control_policy, Mapping)
                or control_policy.get("diagnostic_only") is not True
                or control_policy.get("effect_funnel_stage_consumed") is not False
                or control_policy.get("endpoint_scoring_authorized") is not False
                or control_policy.get("adoption_authorized") is not False
                or control_policy.get("registered_pairs") != 1
                or control_policy.get("seed") != DIAGNOSTIC_SEED
            ):
                blockers.append(
                    "diagnostic search-policy allocation is not exactly one "
                    "non-endpoint pair at seed 66"
                )
    return blockers


def _worker_protocol_blockers(authority: Mapping[str, Any]) -> list[str]:
    """Reject generic launchers and workers that can reach validation code."""

    path = authority["worker"]["local_resolved"]
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"NO_CONFORMING_GPAS_WORKER: cannot inspect {path}: {exc}"]
    required_literals = (
        "PAPER020_GPAS_WORKER_PROTOCOL_VERSION = 1",
        "PAPER020_GPAS_NO_VALIDATION = True",
        "--phase-manifest",
        "--result",
        "endpoint_scoring_performed",
        "validation_data_access",
        "worker_execution",
        "training_shard_ids",
        "held_out_access_authorized",
    )
    missing = [literal for literal in required_literals if literal not in source]
    forbidden_patterns = {
        "evaluate_bpb": r"\bevaluate_bpb(?:_fast)?\b",
        "validation_loader": r"\bval_loader\b",
        "validation_split": r"""["']val["']""",
        "validation_metric": r"\bval_(?:bpb|loss)\b",
    }
    forbidden = [
        name
        for name, pattern in forbidden_patterns.items()
        if re.search(pattern, source)
    ]
    blockers = []
    if missing or forbidden:
        blockers.append(
            "NO_CONFORMING_GPAS_WORKER: separately hash-bound training-only "
            f"worker contract failed; missing={missing} forbidden={forbidden}"
        )
    return blockers


def readiness_blockers(authority: Mapping[str, Any]) -> list[str]:
    """Return every local, committed-authority, and source blocker."""

    blockers: list[str] = []
    blockers.extend(_governance_blockers(authority))
    blockers.extend(_worker_protocol_blockers(authority))
    for relative, expected in BOUND_FILES.items():
        path = REPO_ROOT / relative
        try:
            actual = _sha256(path)
        except AuthorityError as exc:
            blockers.append(str(exc))
            continue
        if actual != expected:
            blockers.append(
                f"local source drift {relative}: expected={expected} actual={actual}"
            )

    runner_path = REPO_ROOT / RUNNER_RELATIVE_PATH
    try:
        runner_sha = _sha256(runner_path)
    except AuthorityError as exc:
        blockers.append(str(exc))
        runner_sha = ""
    if runner_sha != authority["bindings"]["runner_sha256"]:
        blockers.append(
            "runner SHA-256 differs from detached authority: "
            f"expected={authority['bindings']['runner_sha256']} actual={runner_sha}"
        )

    worker = authority["worker"]
    try:
        worker_sha = _sha256(worker["local_resolved"])
    except AuthorityError as exc:
        blockers.append(str(exc))
        worker_sha = ""
    if worker_sha != worker["sha256"]:
        blockers.append(
            "worker SHA-256 differs from detached authority: "
            f"expected={worker['sha256']} actual={worker_sha}"
        )
    known_hosts_path = Path(authority["remote"]["known_hosts_path"])
    try:
        known_hosts_sha = _sha256(known_hosts_path)
    except AuthorityError as exc:
        blockers.append(str(exc))
    else:
        if known_hosts_sha != authority["remote"]["known_hosts_sha256"]:
            blockers.append(
                "known_hosts continuity file drift: "
                f"expected={authority['remote']['known_hosts_sha256']} "
                f"actual={known_hosts_sha}"
            )

    setup_path = REPO_ROOT / "research/setup/reconciliation.json"
    try:
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"cannot parse setup reconciliation: {exc}")
    else:
        if setup.get("version") != SETUP_VERSION:
            blockers.append(
                f"setup version drift: expected={SETUP_VERSION} "
                f"actual={setup.get('version')!r}"
            )
        if setup.get("fingerprint") != SETUP_FINGERPRINT:
            blockers.append(
                "setup fingerprint drift: "
                f"expected={SETUP_FINGERPRINT} actual={setup.get('fingerprint')!r}"
            )

    try:
        head = _git_output(["rev-parse", "HEAD"])
    except AuthorityError as exc:
        blockers.append(str(exc))
        head = ""
    if head != authority["git_commit"]:
        blockers.append(
            f"authority Git commit is not current HEAD: "
            f"authority={authority['git_commit']} HEAD={head}"
        )
    try:
        engine_dirty = _git_output(
            [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                "vibeautoresearch",
            ]
        )
    except AuthorityError as exc:
        blockers.append(str(exc))
    else:
        if engine_dirty:
            blockers.append(
                "live governance engine has uncommitted/untracked drift: "
                f"{engine_dirty.splitlines()}"
            )

    committed_paths = {
        **BOUND_FILES,
        authority["governance"]["gated_ledger_path"]: authority["governance"][
            "gated_ledger_sha256"
        ],
        RUNNER_RELATIVE_PATH: authority["bindings"]["runner_sha256"],
        worker["local_path"]: worker["sha256"],
    }
    for relative, expected in committed_paths.items():
        try:
            committed_sha = _git_blob_sha256(authority["git_commit"], relative)
        except AuthorityError as exc:
            blockers.append(str(exc))
            continue
        if committed_sha != expected:
            blockers.append(
                f"committed blob drift {relative}: "
                f"expected={expected} committed={committed_sha}"
            )

    authority_path = authority["authority_path"]
    try:
        authority_relative = str(authority_path.relative_to(REPO_ROOT))
    except ValueError:
        blockers.append("execution authority must itself be stored inside the repository")
    else:
        try:
            committed_authority = _git_blob_sha256(
                authority["git_commit"], authority_relative
            )
        except AuthorityError as exc:
            blockers.append(str(exc))
        else:
            if committed_authority != authority["authority_sha256"]:
                blockers.append(
                    "committed authority bytes differ from --authority-sha256"
                )

    if authority["result_dir"].exists():
        blockers.append(
            f"single-use immutable result path already exists: {authority['result_dir']}"
        )
    return blockers


@dataclass(frozen=True)
class Arm:
    role: str
    gpu_slot: str
    gpas_enable: str


@dataclass(frozen=True)
class Phase:
    phase_id: str
    mode: str
    steps: int
    execution: str
    arms: tuple[Arm, ...]
    timing_start_step: int | None = None
    timing_block_size: int | None = None


def phase_plan() -> tuple[Phase, ...]:
    return (
        Phase(
            "replay_control_step0",
            "step0_parity",
            1,
            "eager_after_backward_before_optimizer_step",
            (Arm("control", "replay", "0"),),
        ),
        Phase(
            "replay_treatment_alpha_zero_step0",
            "step0_parity",
            1,
            "eager_after_backward_before_optimizer_step",
            (Arm("treatment", "replay", "1"),),
        ),
        Phase(
            "eager_mediator_pair_256",
            "eager_mediator",
            256,
            "eager_training_only",
            (
                Arm("control", "timing_a", "0"),
                Arm("treatment", "timing_b", "1"),
            ),
        ),
        Phase(
            "timing_placement_1",
            "clean_timing",
            256,
            "compiled_no_probes_no_validation",
            (
                Arm("control", "timing_a", "0"),
                Arm("treatment", "timing_b", "1"),
            ),
            timing_start_step=33,
            timing_block_size=8,
        ),
        Phase(
            "timing_placement_2_role_swapped",
            "clean_timing",
            256,
            "compiled_no_probes_no_validation",
            (
                Arm("treatment", "timing_a", "1"),
                Arm("control", "timing_b", "0"),
            ),
            timing_start_step=33,
            timing_block_size=8,
        ),
    )


def _require_next_phase(completed: Sequence[str], phase: Phase) -> None:
    expected_order = [item.phase_id for item in phase_plan()]
    if list(completed) != expected_order[: len(completed)]:
        raise RuntimeIntegrityError(
            f"runtime state history is not a valid prefix: {list(completed)}"
        )
    if len(completed) >= len(expected_order):
        raise RuntimeIntegrityError("runtime state machine forbids an extra phase")
    expected = expected_order[len(completed)]
    if phase.phase_id != expected:
        raise RuntimeIntegrityError(
            f"runtime state transition expected {expected!r}, "
            f"received {phase.phase_id!r}"
        )


@dataclass
class RuntimeStateMachine:
    """Explicit fail-fast barriers between the five non-collapsible phases."""

    completed: list[str] = field(default_factory=list)
    replay_barrier_passed: bool = False
    mediator_barrier_passed: bool = False
    placement_1_integrity_barrier_passed: bool = False

    def before_phase(self, phase: Phase) -> None:
        _require_next_phase(self.completed, phase)
        if (
            phase.phase_id == "eager_mediator_pair_256"
            and not self.replay_barrier_passed
        ):
            raise RuntimeIntegrityError(
                "eager mediator pair is blocked until exact replay passes"
            )
        if (
            phase.phase_id == "timing_placement_1"
            and not self.mediator_barrier_passed
        ):
            raise RuntimeIntegrityError(
                "timing is blocked until applicability and mediation pass"
            )
        if (
            phase.phase_id == "timing_placement_2_role_swapped"
            and not self.placement_1_integrity_barrier_passed
        ):
            raise RuntimeIntegrityError(
                "role-swapped timing is blocked until placement-1 compile "
                "and row integrity pass"
            )

    def finish_phase(self, phase: Phase) -> None:
        self.completed.append(phase.phase_id)

    def pass_replay(self) -> None:
        if self.completed != [
            "replay_control_step0",
            "replay_treatment_alpha_zero_step0",
        ]:
            raise RuntimeIntegrityError(
                "replay barrier cannot pass before both sequential replays"
            )
        self.replay_barrier_passed = True

    def pass_mediator(self) -> None:
        if (
            self.completed
            != [
                "replay_control_step0",
                "replay_treatment_alpha_zero_step0",
                "eager_mediator_pair_256",
            ]
            or not self.replay_barrier_passed
        ):
            raise RuntimeIntegrityError(
                "mediator barrier cannot pass before the eager pair"
            )
        self.mediator_barrier_passed = True

    def pass_placement_1_integrity(self) -> None:
        if (
            not self.mediator_barrier_passed
            or not self.completed
            or self.completed[-1] != "timing_placement_1"
        ):
            raise RuntimeIntegrityError(
                "placement-1 integrity barrier cannot pass out of order"
            )
        self.placement_1_integrity_barrier_passed = True


def _remote_phase_paths(
    authority: Mapping[str, Any],
    phase: Phase,
    arm: Arm,
) -> dict[str, str]:
    stem = f"{phase.phase_id}.{arm.role}"
    remote_base = (
        f"{authority['remote']['work_dir'].rstrip('/')}/"
        f".paper020-gpas-{authority['nonce'][:16]}"
    )
    return {
        "base": remote_base,
        "manifest": f"{remote_base}/{stem}.manifest.json",
        "result": f"{remote_base}/{stem}.result.json",
        "pidfile": f"{remote_base}/{stem}.pid",
    }


def _render_worker_argv(
    authority: Mapping[str, Any],
    paths: Mapping[str, str],
) -> list[str]:
    replacements = {
        "{python}": authority["remote"]["python"],
        "{worker}": authority["worker"]["remote_path"],
        "{manifest}": paths["manifest"],
        "{result}": paths["result"],
    }
    return [replacements.get(token, token) for token in WORKER_COMMAND_TEMPLATE]


def _phase_manifest(
    authority: Mapping[str, Any],
    phase: Phase,
    arm: Arm,
) -> dict[str, Any]:
    gpu = authority["gpus"][arm.gpu_slot]
    paths = _remote_phase_paths(authority, phase, arm)
    worker_argv = _render_worker_argv(authority, paths)
    return {
        "schema_version": 1,
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "authority_sha256": authority["authority_sha256"],
        "experiment_id": EXPERIMENT_ID,
        "experiment_fingerprint": authority["experiment_fingerprint"],
        "phase_id": phase.phase_id,
        "mode": phase.mode,
        "execution": phase.execution,
        "role": arm.role,
        "seed": DIAGNOSTIC_SEED,
        "steps": phase.steps,
        "gpu": dict(gpu),
        "source_hashes": dict(BOUND_FILES),
        "data_authority": dict(DATA_AUTHORITY),
        "recipe_env": dict(authority["recipe_env"]),
        "packer_env": dict(authority["packer_env"]),
        "run_env": {
            **authority["recipe_env"],
            **authority["packer_env"],
            "GPAS_ENABLE": arm.gpas_enable,
            "MAX_STEPS": str(phase.steps),
            "SEED": str(DIAGNOSTIC_SEED),
            "STOP_MODE": "steps",
        },
        "training_data_only": True,
        "validation_data_access": False,
        "endpoint_scoring_authorized": False,
        "sota_update_authorized": False,
        "timing_start_step": phase.timing_start_step,
        "timing_block_size": phase.timing_block_size,
        "worker_execution": {
            "worker_sha256": authority["worker"]["sha256"],
            "python_sha256": authority["remote"]["python_sha256"],
            "command_template": list(WORKER_COMMAND_TEMPLATE),
            "command_template_sha256": authority["worker"][
                "command_template_sha256"
            ],
            "argv": worker_argv,
            "argv_sha256": _sha256_bytes(_canonical_json(worker_argv)),
        },
    }


def build_dry_run(authority: Mapping[str, Any]) -> dict[str, Any]:
    phases = []
    for phase in phase_plan():
        manifests = []
        for arm in phase.arms:
            manifest = _phase_manifest(authority, phase, arm)
            manifests.append(
                {
                    "role": arm.role,
                    "gpu": manifest["gpu"],
                    "gpas_enable": arm.gpas_enable,
                    "manifest_sha256": _sha256_bytes(_canonical_json(manifest)),
                    "worker_sha256": manifest["worker_execution"][
                        "worker_sha256"
                    ],
                    "worker_argv_sha256": manifest["worker_execution"][
                        "argv_sha256"
                    ],
                }
            )
        phases.append(
            {
                "phase_id": phase.phase_id,
                "mode": phase.mode,
                "steps": phase.steps,
                "execution": phase.execution,
                "arms": manifests,
            }
        )
    blockers = readiness_blockers(authority)
    return {
        "schema_version": 1,
        "mode": "dry_run",
        "launches_performed": 0,
        "ready": not blockers,
        "blockers": blockers,
        "authority_sha256": authority["authority_sha256"],
        "runner_sha256": _sha256(REPO_ROOT / RUNNER_RELATIVE_PATH),
        "source_hashes": dict(BOUND_FILES),
        "current_R0": dict(CURRENT_R0_ENV),
        "packer_flags": dict(PACKER_ZERO_ENV),
        "historical_0_927183_role": "chart_origin_only_forbidden_as_arm",
        "result_dir": str(authority["result_dir"].relative_to(REPO_ROOT)),
        "phases": phases,
        "non_claims": {
            "validation_data_access": False,
            "endpoint_bpb": False,
            "comparator_rebind": False,
            "chart_update": False,
            "sota": False,
        },
    }


def _write_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return _sha256_bytes(payload)


def _write_exclusive_json(path: Path, payload: Any) -> str:
    return _write_exclusive(path, _canonical_json(payload))


def _enumerate_immutable_artifacts(artifact_dir: Path) -> dict[str, str]:
    """Hash every finalized output file present before FINAL.json is written."""

    observed: dict[str, str] = {}
    for path in sorted(artifact_dir.iterdir()):
        if path.is_dir():
            # The only directory is the private TemporaryDirectory used for
            # transfer staging; it is not part of the immutable evidence set.
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeIntegrityError(
                f"unexpected non-regular artifact before FINAL: {path}"
            )
        if path.name in {"FINAL.json", "FINAL.sha256.json"}:
            raise RuntimeIntegrityError(
                f"terminal artifact unexpectedly pre-exists: {path}"
            )
        observed[path.name] = _sha256_bytes(path.read_bytes())
    return observed


def _assert_no_forbidden_result_keys(value: Any, where: str = "result") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_RESULT_KEYS or normalized.startswith("val_"):
                raise RuntimeIntegrityError(
                    f"{where} contains forbidden endpoint key {key!r}"
                )
            _assert_no_forbidden_result_keys(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_forbidden_result_keys(item, f"{where}[{index}]")


def _require_finite_number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeIntegrityError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise RuntimeIntegrityError(f"{where} must be finite" + (" and positive" if positive else ""))
    return result


def _require_variances(row: Mapping[str, Any], where: str) -> list[float]:
    raw = row.get("post_mlp_residual_variance")
    if not isinstance(raw, list) or len(raw) != 8:
        raise RuntimeIntegrityError(f"{where} must contain exactly eight variances")
    return [
        _require_finite_number(value, f"{where}[{index}]", positive=True)
        for index, value in enumerate(raw)
    ]


def _average_tie_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for original, _ in indexed[start:end]:
            ranks[original] = rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise RuntimeIntegrityError("correlation vectors have invalid lengths")
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator <= 0 or not math.isfinite(denominator):
        raise RuntimeIntegrityError("correlation denominator is non-positive")
    return sum(
        lvalue * rvalue
        for lvalue, rvalue in zip(centered_left, centered_right, strict=True)
    ) / denominator


def spearman_layer_log_variance(variances: Sequence[float]) -> float:
    if len(variances) != 8:
        raise RuntimeIntegrityError("Spearman input must contain eight layers")
    logs = [math.log(_require_finite_number(value, "variance", positive=True)) for value in variances]
    return _pearson(list(range(8)), _average_tie_ranks(logs))


def evaluate_step0_replay(
    control_result: Mapping[str, Any],
    treatment_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Strengthen the helper comparator to the complete Paper-020 projection."""

    from tools.gpas_mechanism_diagnostic import compare_step0_parity

    control = control_result["parity"]
    treatment = treatment_result["parity"]
    if not isinstance(control, dict) or not isinstance(treatment, dict):
        raise RuntimeIntegrityError("step0 parity payloads must be objects")
    parity_keys = {
        "schema_version",
        "diagnostic_only",
        "packer_flags",
        "device_uuid",
        "data_sha256",
        "targets_sha256",
        "token_bytes_sha256",
        "boundaries_sha256",
        "data_cursor_sha256",
        "epoch",
        "cpu_rng_before_forward_sha256",
        "cuda_rng_before_forward_sha256",
        "cpu_rng_sha256",
        "cuda_rng_sha256",
        "logits_sha256",
        "loss_sha256",
        "preexisting_parameters_sha256",
        "canonical_preexisting_model_state_sha256",
        "preexisting_gradients_sha256",
        "preexisting_gradient_coverage",
        "gpas_gates",
        "compile_identity",
    }
    _require_exact_runtime_keys(control, parity_keys, "control parity")
    _require_exact_runtime_keys(treatment, parity_keys, "treatment parity")
    helper_result = compare_step0_parity(control, treatment)
    for key in (
        "data_cursor_sha256",
        "epoch",
        "cpu_rng_before_forward_sha256",
        "cuda_rng_before_forward_sha256",
        "canonical_preexisting_model_state_sha256",
        "preexisting_gradient_coverage",
    ):
        if control[key] != treatment[key]:
            raise RuntimeIntegrityError(f"step0 canonical replay differs in {key}")
    for side, parity in (("control", control), ("treatment", treatment)):
        for key in (
            "data_cursor_sha256",
            "cpu_rng_before_forward_sha256",
            "cuda_rng_before_forward_sha256",
            "canonical_preexisting_model_state_sha256",
        ):
            if (
                not isinstance(parity[key], str)
                or not SHA256_RE.fullmatch(parity[key])
            ):
                raise RuntimeIntegrityError(
                    f"{side} parity {key} must be lowercase SHA-256"
                )
        coverage = parity["preexisting_gradient_coverage"]
        if not isinstance(coverage, dict):
            raise RuntimeIntegrityError(
                f"{side} preexisting_gradient_coverage must be an object"
            )
        _require_exact_runtime_keys(
            coverage,
            {
                "parameter_count",
                "gradient_tensor_count",
                "missing_gradient_names",
                "manifest_sha256",
            },
            f"{side} preexisting_gradient_coverage",
        )
        if (
            isinstance(coverage["parameter_count"], bool)
            or not isinstance(coverage["parameter_count"], int)
            or coverage["parameter_count"] <= 0
            or coverage["gradient_tensor_count"] != coverage["parameter_count"]
            or coverage["missing_gradient_names"] != []
            or not isinstance(coverage["manifest_sha256"], str)
            or not SHA256_RE.fullmatch(coverage["manifest_sha256"])
        ):
            raise RuntimeIntegrityError(
                f"{side} parity does not cover every pre-existing gradient"
            )

    if control["gpas_gates"] != []:
        raise RuntimeIntegrityError("control replay unexpectedly contains GPAS gates")
    gates = treatment["gpas_gates"]
    if not isinstance(gates, list) or len(gates) != 8:
        raise RuntimeIntegrityError("treatment replay must contain exactly eight gates")
    expected_names = [
        f"transformer.h.{layer}.gpas_alpha" for layer in range(8)
    ]
    observed_names = []
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            raise RuntimeIntegrityError(f"treatment gate {index} must be an object")
        _require_exact_runtime_keys(
            gate,
            {"name", "shape", "numel", "alpha", "gradient", "forward_scale"},
            f"treatment gate {index}",
        )
        observed_names.append(gate["name"])
        gradient = _require_finite_number(
            gate["gradient"], f"treatment gate {index} gradient"
        )
        if (
            gate["shape"] != []
            or gate["numel"] != 1
            or gate["alpha"] != 0.0
            or gate["forward_scale"] != 1.0
            or gradient == 0.0
        ):
            raise RuntimeIntegrityError(
                f"treatment gate {index} violates exact alpha-zero contract"
            )
    if observed_names != expected_names:
        raise RuntimeIntegrityError(
            f"treatment gate names differ: {observed_names}"
        )

    control_optimizer = control_result["optimizer_attestation"]
    treatment_optimizer = treatment_result["optimizer_attestation"]
    optimizer_keys = {
        "canonical_preexisting_optimizer_groups_sha256",
        "canonical_preexisting_optimizer_state_sha256",
        "preexisting_optimizer_manifest_sha256",
        "gpas_group",
    }
    if not isinstance(control_optimizer, dict) or not isinstance(
        treatment_optimizer, dict
    ):
        raise RuntimeIntegrityError("optimizer attestations must be objects")
    _require_exact_runtime_keys(
        control_optimizer, optimizer_keys, "control optimizer attestation"
    )
    _require_exact_runtime_keys(
        treatment_optimizer, optimizer_keys, "treatment optimizer attestation"
    )
    for key in optimizer_keys - {"gpas_group"}:
        for side, value in (
            ("control", control_optimizer[key]),
            ("treatment", treatment_optimizer[key]),
        ):
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise RuntimeIntegrityError(
                    f"{side} optimizer {key} must be lowercase SHA-256"
                )
        if control_optimizer[key] != treatment_optimizer[key]:
            raise RuntimeIntegrityError(
                f"step0 canonical optimizer projection differs in {key}"
            )
    if control_optimizer["gpas_group"] is not None:
        raise RuntimeIntegrityError("control optimizer unexpectedly has a GPAS group")
    group = treatment_optimizer["gpas_group"]
    if not isinstance(group, dict):
        raise RuntimeIntegrityError("treatment optimizer lacks the GPAS-only group")
    _require_exact_runtime_keys(
        group,
        {
            "parameter_names",
            "parameter_shapes",
            "total_numel",
            "learning_rate",
            "betas",
            "eps",
            "weight_decay",
            "demon_beta1",
            "schedule_exempt",
            "member_of_any_other_group",
        },
        "treatment GPAS optimizer group",
    )
    if (
        group["parameter_names"] != expected_names
        or group["parameter_shapes"] != [[] for _ in range(8)]
        or group["total_numel"] != 8
        or group["learning_rate"] != 0.005
        or group["betas"] != [0.8, 0.95]
        or group["eps"] != 1e-10
        or group["weight_decay"] != 0.0
        or group["demon_beta1"] is not False
        or group["schedule_exempt"] is not True
        or group["member_of_any_other_group"] is not False
    ):
        raise RuntimeIntegrityError(
            "treatment GPAS optimizer group differs from Paper-020"
        )
    return {
        **helper_result,
        "complete_preexisting_gradient_coverage": True,
        "canonical_model_state_equal": True,
        "canonical_optimizer_groups_and_state_equal": True,
        "exact_eight_named_gates": True,
        "data_cursor_epoch_and_pre_forward_rng_equal": True,
    }


def evaluate_applicability(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list) or len(rows) != 32:
        raise RuntimeIntegrityError("applicability must contain exactly 32 rows")
    valid = 0
    evaluations = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or row.get("step") != index:
            raise RuntimeIntegrityError("applicability steps must be exactly 1..32")
        variances = _require_variances(
            row, f"applicability.rows[{index - 1}].post_mlp_residual_variance"
        )
        try:
            correlation: float | None = spearman_layer_log_variance(variances)
        except RuntimeIntegrityError as exc:
            if "correlation denominator is non-positive" not in str(exc):
                raise
            # A fully tied depth profile has undefined Spearman correlation.
            # It is a failed applicability step, not evidence for the premise.
            correlation = None
        ratio = variances[7] / variances[0]
        passes = correlation is not None and correlation >= 0.60 and ratio >= 1.25
        valid += int(passes)
        evaluations.append(
            {
                "step": index,
                "spearman_layer_vs_log_variance": correlation,
                "v7_over_v0": ratio,
                "passes_joint_gate": passes,
            }
        )
    return {
        "valid_steps": valid,
        "minimum_valid_steps": 24,
        "passed": valid >= 24,
        "steps": evaluations,
    }


def _paired_row_hashes_equal(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    where: str,
) -> None:
    digest_keys = (
        "data_sha256",
        "targets_sha256",
        "token_bytes_sha256",
        "boundaries_sha256",
    )
    integer_keys = (
        "data_cursor",
        "epoch",
    )
    for side, row in (("control", control), ("treatment", treatment)):
        for key in digest_keys:
            value = row.get(key)
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise RuntimeIntegrityError(
                    f"{where} {side}.{key} must be present lowercase SHA-256"
                )
        for key in integer_keys:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeIntegrityError(
                    f"{where} {side}.{key} must be a non-negative integer"
                )
    for key in (*digest_keys, *integer_keys):
        if control.get(key) != treatment.get(key):
            raise RuntimeIntegrityError(f"{where} differs in {key}")


def _validate_assay_row(
    row: Mapping[str, Any],
    *,
    treatment: bool,
    where: str,
) -> None:
    keys = {
        "step",
        "post_mlp_residual_variance",
        "data_sha256",
        "targets_sha256",
        "token_bytes_sha256",
        "boundaries_sha256",
        "data_cursor",
        "epoch",
        "preexisting_gradients_finite",
    }
    if treatment:
        keys |= {"gpas_gradients_finite", "gpas_forward_scales"}
    _require_exact_runtime_keys(row, keys, where)


def _validate_eager_pair_structure(
    control_rows: Any,
    treatment_rows: Any,
    counterfactual: Any,
) -> dict[str, Any]:
    """Validate the complete eager artifact before interpreting its premise.

    This is deliberately structure-only: it establishes exact schemas, finite
    measurements, and batch identity for all 256 pairs.  It does not judge the
    applicability or mediator thresholds.
    """

    if (
        not isinstance(control_rows, list)
        or not isinstance(treatment_rows, list)
        or len(control_rows) != 256
        or len(treatment_rows) != 256
    ):
        raise RuntimeIntegrityError(
            "eager mediator arms must each contain exactly 256 rows"
        )
    for index in range(1, 257):
        control = control_rows[index - 1]
        treatment = treatment_rows[index - 1]
        if not isinstance(control, dict) or not isinstance(treatment, dict):
            raise RuntimeIntegrityError(
                f"eager mediator step {index} rows must be objects"
            )
        if control.get("step") != index or treatment.get("step") != index:
            raise RuntimeIntegrityError(
                "eager mediator steps must be exactly 1..256"
            )
        _validate_assay_row(
            control,
            treatment=False,
            where=f"eager mediator control step {index}",
        )
        _validate_assay_row(
            treatment,
            treatment=True,
            where=f"eager mediator treatment step {index}",
        )
        _paired_row_hashes_equal(
            control, treatment, f"eager mediator step {index}"
        )
        _require_variances(
            control,
            f"eager mediator control step {index}.post_mlp_residual_variance",
        )
        _require_variances(
            treatment,
            f"eager mediator treatment step {index}.post_mlp_residual_variance",
        )
        if not isinstance(control["preexisting_gradients_finite"], bool):
            raise RuntimeIntegrityError(
                f"eager mediator control step {index} finite-gradient flag "
                "must be boolean"
            )
        for key in ("preexisting_gradients_finite", "gpas_gradients_finite"):
            if not isinstance(treatment[key], bool):
                raise RuntimeIntegrityError(
                    f"eager mediator treatment step {index} {key} must be boolean"
                )
        scales = treatment["gpas_forward_scales"]
        if not isinstance(scales, list) or len(scales) != 8:
            raise RuntimeIntegrityError(
                f"eager mediator treatment step {index} must contain "
                "exactly eight GPAS scales"
            )
        for layer, scale in enumerate(scales):
            _require_finite_number(
                scale,
                f"eager mediator treatment step {index} scale {layer}",
            )

    if not isinstance(counterfactual, dict):
        raise RuntimeIntegrityError("treatment counterfactual must be an object")
    counterfactual_keys = {
        "step",
        "data_sha256",
        "targets_sha256",
        "token_bytes_sha256",
        "boundaries_sha256",
        "data_cursor",
        "epoch",
        "post_mlp_residual_variance",
        "gpas_gate_values_sha256_before_zero",
        "gpas_gate_values_sha256_while_zero",
        "gpas_gate_values_sha256_after_restore",
        "preexisting_model_state_sha256_before",
        "preexisting_model_state_sha256_after",
        "all_gpas_gates_exact_zero_during_measurement",
        "gpas_alpha_restored_after_measurement",
    }
    _require_exact_runtime_keys(
        counterfactual, counterfactual_keys, "counterfactual"
    )
    if counterfactual["step"] != 256:
        raise RuntimeIntegrityError(
            "counterfactual must use treatment step-256 weights"
        )
    _paired_row_hashes_equal(
        treatment_rows[-1], counterfactual, "counterfactual endpoint batch"
    )
    _require_variances(
        counterfactual, "counterfactual.post_mlp_residual_variance"
    )
    for key in (
        "gpas_gate_values_sha256_before_zero",
        "gpas_gate_values_sha256_while_zero",
        "gpas_gate_values_sha256_after_restore",
        "preexisting_model_state_sha256_before",
        "preexisting_model_state_sha256_after",
    ):
        value = counterfactual[key]
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise RuntimeIntegrityError(
                f"counterfactual {key} must be lowercase SHA-256"
            )
    for key in (
        "all_gpas_gates_exact_zero_during_measurement",
        "gpas_alpha_restored_after_measurement",
    ):
        if not isinstance(counterfactual[key], bool):
            raise RuntimeIntegrityError(
                f"counterfactual {key} must be boolean"
            )
    return {
        "rows_per_arm": 256,
        "paired_batch_identities_verified": 256,
        "counterfactual_endpoint_batch_verified": True,
        "thresholds_evaluated": False,
    }


def evaluate_attribution(control_rows: Any, treatment_rows: Any, counterfactual: Any) -> dict[str, Any]:
    _validate_eager_pair_structure(
        control_rows, treatment_rows, counterfactual
    )
    if (
        not isinstance(control_rows, list)
        or not isinstance(treatment_rows, list)
        or len(control_rows) != 256
        or len(treatment_rows) != 256
    ):
        raise RuntimeIntegrityError("attribution arms must each contain exactly 256 rows")
    control_variances: list[list[float]] = []
    treatment_variances: list[list[float]] = []
    all_scales_safe = True
    all_gradients_finite = True
    for index, (control, treatment) in enumerate(
        zip(control_rows, treatment_rows, strict=True), start=1
    ):
        if (
            not isinstance(control, dict)
            or not isinstance(treatment, dict)
            or control.get("step") != index
            or treatment.get("step") != index
        ):
            raise RuntimeIntegrityError("attribution steps must be exactly 1..256")
        _validate_assay_row(
            control,
            treatment=False,
            where=f"attribution control step {index}",
        )
        _validate_assay_row(
            treatment,
            treatment=True,
            where=f"attribution treatment step {index}",
        )
        _paired_row_hashes_equal(control, treatment, f"attribution step {index}")
        c_values = _require_variances(
            control, f"attribution.control[{index - 1}].post_mlp_residual_variance"
        )
        t_values = _require_variances(
            treatment,
            f"attribution.treatment[{index - 1}].post_mlp_residual_variance",
        )
        control_variances.append(c_values)
        treatment_variances.append(t_values)
        if control.get("preexisting_gradients_finite") is not True:
            all_gradients_finite = False
        if (
            treatment.get("preexisting_gradients_finite") is not True
            or treatment.get("gpas_gradients_finite") is not True
        ):
            all_gradients_finite = False
        scales = treatment.get("gpas_forward_scales")
        if not isinstance(scales, list) or len(scales) != 8:
            raise RuntimeIntegrityError(
                f"attribution treatment step {index} lacks eight GPAS scales"
            )
        for layer, scale in enumerate(scales):
            value = _require_finite_number(
                scale, f"attribution treatment step {index} scale {layer}"
            )
            if value < 0.5 or value > 1.5:
                all_scales_safe = False

    c_ratios = [values[7] / values[0] for values in control_variances]
    t_ratios = [values[7] / values[0] for values in treatment_variances]
    mean_c_ratio = sum(c_ratios) / 256
    mean_t_ratio = sum(t_ratios) / 256
    mean_c_v7 = sum(values[7] for values in control_variances) / 256
    mean_t_v7 = sum(values[7] for values in treatment_variances) / 256
    ratio_gate = mean_t_ratio / mean_c_ratio
    deepest_gate = mean_t_v7 / mean_c_v7

    layer_control = [
        sum(values[layer] / values[0] for values in control_variances) / 256
        for layer in range(8)
    ]
    layer_treatment = [
        sum(values[layer] / values[0] for values in treatment_variances) / 256
        for layer in range(8)
    ]
    reductions = [
        max(0.0, layer_control[layer] - layer_treatment[layer])
        for layer in range(8)
    ]
    denominator = sum(reductions[1:])
    if denominator <= 0:
        raise RuntimeIntegrityError(
            "deep-layer contribution denominator is non-positive"
        )
    deep_fraction = sum(reductions[4:8]) / denominator

    if not isinstance(counterfactual, dict):
        raise RuntimeIntegrityError("treatment counterfactual must be an object")
    _require_exact_runtime_keys(
        counterfactual,
        {
            "step",
            "data_sha256",
            "targets_sha256",
            "token_bytes_sha256",
            "boundaries_sha256",
            "data_cursor",
            "epoch",
            "post_mlp_residual_variance",
            "gpas_gate_values_sha256_before_zero",
            "gpas_gate_values_sha256_while_zero",
            "gpas_gate_values_sha256_after_restore",
            "preexisting_model_state_sha256_before",
            "preexisting_model_state_sha256_after",
            "all_gpas_gates_exact_zero_during_measurement",
            "gpas_alpha_restored_after_measurement",
        },
        "counterfactual",
    )
    if counterfactual["step"] != 256:
        raise RuntimeIntegrityError("counterfactual must use treatment step-256 weights")
    if counterfactual["gpas_alpha_restored_after_measurement"] is not True:
        raise RuntimeIntegrityError("counterfactual did not restore learned GPAS gates")
    treatment_last = treatment_rows[-1]
    for key in (
        "data_sha256",
        "targets_sha256",
        "token_bytes_sha256",
        "boundaries_sha256",
        "data_cursor",
        "epoch",
    ):
        if counterfactual[key] != treatment_last.get(key):
            raise RuntimeIntegrityError(
                f"counterfactual fixed endpoint batch differs in {key}"
            )
    for key in (
        "gpas_gate_values_sha256_before_zero",
        "gpas_gate_values_sha256_while_zero",
        "gpas_gate_values_sha256_after_restore",
        "preexisting_model_state_sha256_before",
        "preexisting_model_state_sha256_after",
    ):
        if (
            not isinstance(counterfactual[key], str)
            or not SHA256_RE.fullmatch(counterfactual[key])
        ):
            raise RuntimeIntegrityError(
                f"counterfactual {key} must be lowercase SHA-256"
            )
    if counterfactual["all_gpas_gates_exact_zero_during_measurement"] is not True:
        raise RuntimeIntegrityError(
            "counterfactual gates were not attested exact zero"
        )
    if (
        counterfactual["gpas_gate_values_sha256_before_zero"]
        != counterfactual["gpas_gate_values_sha256_after_restore"]
    ):
        raise RuntimeIntegrityError("counterfactual GPAS gate-state restore hash differs")
    if (
        counterfactual["gpas_gate_values_sha256_before_zero"]
        == counterfactual["gpas_gate_values_sha256_while_zero"]
    ):
        raise RuntimeIntegrityError(
            "counterfactual zero-gate hash equals learned gate-state hash"
        )
    if (
        counterfactual["preexisting_model_state_sha256_before"]
        != counterfactual["preexisting_model_state_sha256_after"]
    ):
        raise RuntimeIntegrityError(
            "counterfactual changed pre-existing model state"
        )
    cf_values = _require_variances(
        counterfactual, "counterfactual.post_mlp_residual_variance"
    )
    endpoint_control_ratio = control_variances[-1][7] / control_variances[-1][0]
    endpoint_treatment_ratio = treatment_variances[-1][7] / treatment_variances[-1][0]
    endpoint_counterfactual_ratio = cf_values[7] / cf_values[0]
    restore_denominator = endpoint_control_ratio - endpoint_treatment_ratio
    if restore_denominator <= 0:
        raise RuntimeIntegrityError(
            "counterfactual restoration denominator is non-positive"
        )
    restore_fraction = (
        endpoint_counterfactual_ratio - endpoint_treatment_ratio
    ) / restore_denominator

    gates = {
        "depth_ratio_treatment_over_control": ratio_gate,
        "depth_ratio_max": 0.85,
        "deepest_variance_treatment_over_control": deepest_gate,
        "deepest_variance_max": 0.90,
        "counterfactual_restore_fraction": restore_fraction,
        "counterfactual_restore_min": 0.50,
        "deep_layer_contribution_fraction": deep_fraction,
        "deep_layer_contribution_min": 0.50,
        "all_scales_finite_in_range": all_scales_safe,
        "all_gradients_finite": all_gradients_finite,
    }
    passed = (
        ratio_gate <= 0.85
        and deepest_gate <= 0.90
        and restore_fraction >= 0.50
        and deep_fraction >= 0.50
        and all_scales_safe
        and all_gradients_finite
    )
    return {"passed": passed, **gates}


def _require_exact_runtime_keys(
    value: Mapping[str, Any],
    expected: set[str],
    where: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise RuntimeIntegrityError(
            f"{where} keys differ: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _clean_timing_blocks(rows: Any, where: str) -> list[tuple[int, float]]:
    if not isinstance(rows, list) or len(rows) != 256:
        raise RuntimeIntegrityError(f"{where} must contain exactly 256 timing rows")
    clean: list[tuple[int, float]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise RuntimeIntegrityError(f"{where}[{index - 1}] must be an object")
        _require_exact_runtime_keys(
            row,
            {
                "step",
                "step_time_ms",
                "tokens",
                "data_sha256",
                "targets_sha256",
                "token_bytes_sha256",
                "boundaries_sha256",
                "data_cursor",
                "epoch",
            },
            f"{where}[{index - 1}]",
        )
        if row["step"] != index:
            raise RuntimeIntegrityError(f"{where} steps must be exactly 1..256")
        elapsed = _require_finite_number(
            row["step_time_ms"], f"{where}[{index - 1}].step_time_ms", positive=True
        )
        tokens = row["tokens"]
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens != 262144
        ):
            raise RuntimeIntegrityError(
                f"{where}[{index - 1}].tokens must equal frozen TBS 262144"
            )
        if index >= 33:
            clean.append((tokens, elapsed))
    if len(clean) != 224:
        raise RuntimeIntegrityError(f"{where} must contain 224 clean timing steps")
    return [
        (
            sum(tokens for tokens, _ in clean[start : start + 8]),
            sum(elapsed for _, elapsed in clean[start : start + 8]),
        )
        for start in range(0, 224, 8)
    ]


def _validate_timing_placement_structure(
    placement: Mapping[str, Any],
    where: str,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Validate one complete placement before another placement may launch."""

    if not isinstance(placement, Mapping):
        raise RuntimeIntegrityError(f"{where} must be an object")
    _require_exact_runtime_keys(placement, {"control", "treatment"}, where)
    control_rows = placement["control"]
    treatment_rows = placement["treatment"]
    control_blocks = _clean_timing_blocks(control_rows, f"{where}.control")
    treatment_blocks = _clean_timing_blocks(
        treatment_rows, f"{where}.treatment"
    )
    # _clean_timing_blocks establishes that both sides are exact 256-row
    # object arrays before pairwise identity is inspected.
    for index in range(1, 257):
        _paired_row_hashes_equal(
            control_rows[index - 1],
            treatment_rows[index - 1],
            f"{where} step {index}",
        )
    return control_blocks, treatment_blocks


def _block_rate_ratio(
    control: Sequence[tuple[int, float]],
    treatment: Sequence[tuple[int, float]],
    indices: Iterable[int] | None = None,
) -> float:
    selected = list(range(len(control))) if indices is None else list(indices)
    control_tokens = sum(control[index][0] for index in selected)
    treatment_tokens = sum(treatment[index][0] for index in selected)
    control_ms = sum(control[index][1] for index in selected)
    treatment_ms = sum(treatment[index][1] for index in selected)
    aggregates = (
        control_tokens,
        treatment_tokens,
        control_ms,
        treatment_ms,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in aggregates
    ):
        raise RuntimeIntegrityError(
            "timing aggregate must be finite and positive"
        )
    ratio = (treatment_tokens / treatment_ms) / (
        control_tokens / control_ms
    )
    if not math.isfinite(ratio) or ratio <= 0:
        raise RuntimeIntegrityError(
            "timing rate ratio must be finite and positive"
        )
    return ratio


def evaluate_role_swapped_timing(
    placement_1: Mapping[str, Any],
    placement_2: Mapping[str, Any],
    *,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples <= 0
    ):
        raise RuntimeIntegrityError(
            "bootstrap_resamples must be a positive integer"
        )
    block_pairs = []
    for label, placement in (
        ("placement_1", placement_1),
        ("placement_2", placement_2),
    ):
        control_blocks, treatment_blocks = (
            _validate_timing_placement_structure(placement, label)
        )
        block_pairs.append((control_blocks, treatment_blocks))

    p1 = _block_rate_ratio(*block_pairs[0])
    p2 = _block_rate_ratio(*block_pairs[1])
    point_product = p1 * p2
    if not math.isfinite(point_product) or point_product <= 0:
        raise RuntimeIntegrityError(
            "role-normalized timing ratio product must be finite and positive"
        )
    point = math.sqrt(point_product)
    if not math.isfinite(point) or point <= 0:
        raise RuntimeIntegrityError(
            "role-normalized timing point must be finite and positive"
        )
    rng = random.Random(19_019_063)
    bootstraps = []
    for _ in range(bootstrap_resamples):
        indices_1 = [rng.randrange(28) for _ in range(28)]
        indices_2 = [rng.randrange(28) for _ in range(28)]
        ratio_1 = _block_rate_ratio(*block_pairs[0], indices_1)
        ratio_2 = _block_rate_ratio(*block_pairs[1], indices_2)
        replicate_product = ratio_1 * ratio_2
        if not math.isfinite(replicate_product) or replicate_product <= 0:
            raise RuntimeIntegrityError(
                "bootstrap timing ratio product must be finite and positive"
            )
        replicate = math.sqrt(replicate_product)
        if not math.isfinite(replicate) or replicate <= 0:
            raise RuntimeIntegrityError(
                "bootstrap timing replicate must be finite and positive"
            )
        bootstraps.append(replicate)
    bootstraps.sort()
    lower_index = max(0, math.ceil(0.05 * bootstrap_resamples) - 1)
    lcb = bootstraps[lower_index]
    if not math.isfinite(lcb) or lcb <= 0:
        raise RuntimeIntegrityError(
            "bootstrap timing lower bound must be finite and positive"
        )
    return {
        "placement_1_ratio": p1,
        "placement_2_ratio": p2,
        "role_normalized_point_ratio": point,
        "one_sided_95pct_lcb": lcb,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": 19_019_063,
        "clean_blocks_per_arm_per_placement": 28,
        "passed": point >= 0.99 and lcb >= 0.985,
    }


def _validate_compile_identity(value: Any, where: str) -> None:
    if not isinstance(value, dict):
        raise RuntimeIntegrityError(f"{where} compile_identity must be an object")
    _require_exact_runtime_keys(
        value,
        {
            "dynamo_graph_count",
            "dynamo_recompile_count",
            "cudagraph_recording_count",
            "compile_ids_sha256",
        },
        f"{where}.compile_identity",
    )
    if (
        value["dynamo_graph_count"] != 1
        or value["dynamo_recompile_count"] != 0
        or value["cudagraph_recording_count"] != 0
        or not isinstance(value["compile_ids_sha256"], str)
        or not SHA256_RE.fullmatch(value["compile_ids_sha256"])
    ):
        raise RuntimeIntegrityError(f"{where} compile identity violates Paper-020")


def _validate_worker_common(
    result: Any,
    *,
    authority: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    expected_pid: int,
) -> None:
    if not isinstance(result, dict):
        raise RuntimeIntegrityError("worker result root must be an object")
    _assert_no_forbidden_result_keys(result)
    common = {
        "schema_version",
        "worker_protocol_version",
        "authority_sha256",
        "phase_manifest_sha256",
        "phase_id",
        "mode",
        "role",
        "seed",
        "steps_completed",
        "gpu",
        "source_hashes",
        "data_authority",
        "recipe_env",
        "packer_env",
        "run_env",
        "worker_execution",
        "training_data_only",
        "diagnostic_only",
        "validation_data_access",
        "endpoint_scoring_performed",
        "sota_update_performed",
    }
    phase_specific = {
        "step0_parity": {"parity", "optimizer_attestation"},
        "eager_mediator": {"rows", "counterfactual"},
        "clean_timing": {"rows", "compile_identity"},
    }[manifest["mode"]]
    _require_exact_runtime_keys(result, common | phase_specific, "worker result")
    expected_scalars = {
        "schema_version": 1,
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "authority_sha256": authority["authority_sha256"],
        "phase_manifest_sha256": manifest_sha256,
        "phase_id": manifest["phase_id"],
        "mode": manifest["mode"],
        "role": manifest["role"],
        "seed": DIAGNOSTIC_SEED,
        "steps_completed": manifest["steps"],
        "source_hashes": BOUND_FILES,
        "data_authority": DATA_AUTHORITY,
        "recipe_env": CURRENT_R0_ENV,
        "packer_env": PACKER_ZERO_ENV,
        "run_env": manifest["run_env"],
        "worker_execution": manifest["worker_execution"],
        "training_data_only": True,
        "diagnostic_only": True,
        "validation_data_access": False,
        "endpoint_scoring_performed": False,
        "sota_update_performed": False,
    }
    for key, expected in expected_scalars.items():
        if result.get(key) != expected:
            raise RuntimeIntegrityError(
                f"worker result {key} differs: expected={expected!r} "
                f"actual={result.get(key)!r}"
            )
    gpu = result["gpu"]
    if not isinstance(gpu, dict):
        raise RuntimeIntegrityError("worker result gpu must be an object")
    _require_exact_runtime_keys(
        gpu, {"index", "uuid", "product", "pid"}, "worker result gpu"
    )
    if gpu != {**manifest["gpu"], "pid": expected_pid}:
        raise RuntimeIntegrityError("worker result GPU index/UUID/PID binding differs")


def _remote_run(
    authority: Mapping[str, Any],
    command: str,
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [*authority["remote"]["ssh_argv"], command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeIntegrityError(
            f"remote command timed out after {timeout}s"
        ) from exc


@contextmanager
def _transaction_gpu_locks(authority: Mapping[str, Any]):
    """Hold run_stage's two per-index cooperative locks for the whole assay."""

    indices = sorted(
        {
            authority["gpus"]["timing_a"]["index"],
            authority["gpus"]["timing_b"]["index"],
        }
    )
    if len(indices) != 2:
        raise RuntimeIntegrityError("transaction requires exactly two GPU locks")
    lock_dir = "/tmp/vibeautoresearch-gpu-locks"
    nested = " ".join(
        f"flock -n {shlex.quote(f'{lock_dir}/stage-scheduler-gpu{index}.lock')}"
        for index in indices
    )
    marker = "PAPER020_GPAS_TRANSACTION_LOCK_ACQUIRED"
    command = (
        f"mkdir -p {shlex.quote(lock_dir)} && {nested} "
        f"sh -c 'printf \"%s\\n\" {marker}; cat >/dev/null'"
    )
    process = subprocess.Popen(
        [*authority["remote"]["ssh_argv"], command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], 30)
    if not ready:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise RuntimeIntegrityError(
            "two-GPU cooperative transaction lock timed out"
        )
    observed = process.stdout.readline().strip()
    if observed != marker:
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read(2000).strip()
        process.terminate()
        raise RuntimeIntegrityError(
            "one of the fixed GPUs is held by another cooperative scheduler"
            + (f": {stderr}" if stderr else "")
        )
    try:
        yield {
            "indices": indices,
            "lock_paths": [
                f"{lock_dir}/stage-scheduler-gpu{index}.lock"
                for index in indices
            ],
            "same_lock_namespace_as_run_stage": True,
        }
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if process.returncode not in {0, None}:
            raise RuntimeIntegrityError(
                f"transaction GPU lock process exited {process.returncode}"
            )


def _parse_gpu_inventory(text: str) -> dict[int, tuple[str, str]]:
    result: dict[int, tuple[str, str]] = {}
    for line in text.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 3:
            raise RuntimeIntegrityError(f"malformed GPU inventory line: {line!r}")
        try:
            index = int(parts[0])
        except ValueError as exc:
            raise RuntimeIntegrityError(
                f"malformed GPU index in inventory: {line!r}"
            ) from exc
        if (
            not GPU_UUID_RE.fullmatch(parts[1])
            or "H200" not in parts[2].upper()
            or index in result
        ):
            raise RuntimeIntegrityError(f"invalid/duplicate GPU inventory line: {line!r}")
        result[index] = (parts[1], parts[2])
    return result


def _query_gpu_inventory(authority: Mapping[str, Any]) -> dict[int, tuple[str, str]]:
    result = _remote_run(
        authority,
        "nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader,nounits",
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeIntegrityError(
            f"GPU inventory failed: {result.stderr.strip()}"
        )
    return _parse_gpu_inventory(result.stdout)


def _query_compute_pids(authority: Mapping[str, Any], uuid: str) -> set[int]:
    command = (
        f"nvidia-smi -i {shlex.quote(uuid)} --query-compute-apps=pid "
        "--format=csv,noheader,nounits"
    )
    result = _remote_run(authority, command, timeout=30)
    if result.returncode != 0:
        raise RuntimeIntegrityError(
            f"compute-PID inventory failed for {uuid}: {result.stderr.strip()}"
        )
    pids: set[int] = set()
    for raw in result.stdout.splitlines():
        item = raw.strip()
        if not item:
            continue
        try:
            pid = int(item)
        except ValueError as exc:
            raise RuntimeIntegrityError(
                f"malformed compute PID for {uuid}: {item!r}"
            ) from exc
        if pid <= 0 or pid in pids:
            raise RuntimeIntegrityError(
                f"invalid/duplicate compute PID for {uuid}: {pid}"
            )
        pids.add(pid)
    return pids


def _require_exact_gpu_bindings(authority: Mapping[str, Any]) -> None:
    inventory = _query_gpu_inventory(authority)
    for slot in ("replay", "timing_a", "timing_b"):
        device = authority["gpus"][slot]
        observed = inventory.get(device["index"])
        if (
            observed is None
            or observed[0] != device["uuid"]
            or observed[1] != device["product"]
        ):
            raise RuntimeIntegrityError(
                f"{slot} index/UUID/product binding drift: expected "
                f"{device['index']}=({device['uuid']}, {device['product']}) "
                f"observed={observed}"
            )


def _require_clean_gpus(authority: Mapping[str, Any], devices: Iterable[Mapping[str, Any]]) -> None:
    for device in devices:
        pids = _query_compute_pids(authority, device["uuid"])
        if pids:
            raise RuntimeIntegrityError(
                f"GPU {device['uuid']} is not tenant-free: compute_pids={sorted(pids)}"
            )


@dataclass
class ActiveArm:
    phase: Phase
    arm: Arm
    manifest: dict[str, Any]
    manifest_sha256: str
    local_manifest: Path
    local_result_tmp: Path
    remote_manifest: str
    remote_result: str
    remote_pidfile: str
    process: subprocess.Popen[str]
    pid: int
    observed_pid: bool = False
    inventory_samples: int = 0


def _scp_to_remote(
    authority: Mapping[str, Any],
    local: Path,
    remote: str,
) -> None:
    destination = f"{authority['remote']['scp_target']}:{remote}"
    result = subprocess.run(
        [*authority["remote"]["scp_argv"], str(local), destination],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeIntegrityError(f"scp upload failed: {result.stderr.strip()}")


def _scp_from_remote(
    authority: Mapping[str, Any],
    remote: str,
    local: Path,
) -> None:
    source = f"{authority['remote']['scp_target']}:{remote}"
    result = subprocess.run(
        [*authority["remote"]["scp_argv"], source, str(local)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeIntegrityError(f"scp download failed: {result.stderr.strip()}")


def _launch_arm(
    authority: Mapping[str, Any],
    phase: Phase,
    arm: Arm,
    staging_dir: Path,
) -> ActiveArm:
    manifest = _phase_manifest(authority, phase, arm)
    manifest_payload = _canonical_json(manifest)
    manifest_sha = _sha256_bytes(manifest_payload)
    stem = f"{phase.phase_id}.{arm.role}"
    local_manifest = staging_dir / f"{stem}.manifest.json"
    _write_exclusive(local_manifest, manifest_payload)
    remote_paths = _remote_phase_paths(authority, phase, arm)
    remote_base = remote_paths["base"]
    remote_manifest = remote_paths["manifest"]
    remote_result = remote_paths["result"]
    remote_pidfile = remote_paths["pidfile"]
    prepare = (
        f"set -eu; umask 077; mkdir -p {shlex.quote(remote_base)}; "
        f"test ! -e {shlex.quote(remote_manifest)}; "
        f"test ! -e {shlex.quote(remote_result)}; "
        f"test ! -e {shlex.quote(remote_pidfile)}"
    )
    prepared = _remote_run(authority, prepare, timeout=30)
    if prepared.returncode != 0:
        raise RuntimeIntegrityError(
            f"remote immutable phase path preparation failed: {prepared.stderr.strip()}"
        )
    _scp_to_remote(authority, local_manifest, remote_manifest)
    device = manifest["gpu"]
    run_env = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(manifest["run_env"].items())
    )
    worker_argv = manifest["worker_execution"]["argv"]
    rendered_worker = " ".join(shlex.quote(token) for token in worker_argv)
    command = (
        "set -eu; "
        f"cd {shlex.quote(authority['remote']['work_dir'])}; "
        f"test \"$(sha256sum {shlex.quote(authority['worker']['remote_path'])} "
        f"| awk '{{print $1}}')\" = {shlex.quote(authority['worker']['sha256'])}; "
        f"test \"$(sha256sum {shlex.quote(authority['remote']['python'])} "
        f"| awk '{{print $1}}')\" = "
        f"{shlex.quote(authority['remote']['python_sha256'])}; "
        f"printf '%s\\n' \"$$\" > {shlex.quote(remote_pidfile)}; "
        f"exec env CUDA_DEVICE_ORDER=PCI_BUS_ID "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(device['uuid'])} {run_env} "
        f"{rendered_worker}"
    )
    process = subprocess.Popen(
        [*authority["remote"]["ssh_argv"], command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pid = 0
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        pid_result = _remote_run(
            authority,
            f"test -s {shlex.quote(remote_pidfile)} && cat {shlex.quote(remote_pidfile)}",
            timeout=10,
        )
        if pid_result.returncode == 0 and pid_result.stdout.strip():
            try:
                pid = int(pid_result.stdout.strip())
            except ValueError as exc:
                process.terminate()
                raise RuntimeIntegrityError(
                    f"worker PID file is malformed: {pid_result.stdout!r}"
                ) from exc
            break
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeIntegrityError(
                f"worker exited before PID attestation: {output[-2000:]}"
            )
        time.sleep(0.25)
    if pid <= 0:
        process.terminate()
        raise RuntimeIntegrityError("worker PID attestation timed out")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f"{stem}.", suffix=".result.tmp", dir=staging_dir
    )
    os.close(descriptor)
    local_result_tmp = Path(temporary)
    return ActiveArm(
        phase=phase,
        arm=arm,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        local_manifest=local_manifest,
        local_result_tmp=local_result_tmp,
        remote_manifest=remote_manifest,
        remote_result=remote_result,
        remote_pidfile=remote_pidfile,
        process=process,
        pid=pid,
    )


def _terminate_active(
    authority: Mapping[str, Any],
    arms: Iterable[ActiveArm],
) -> None:
    active_arms = list(arms)
    failures: list[str] = []
    for active in active_arms:
        pid = active.pid
        expected_worker = authority["worker"]["remote_path"]
        expected_cwd = authority["remote"]["work_dir"]
        terminate = (
            "set -eu; "
            f"pid={pid}; "
            "if kill -0 \"$pid\" 2>/dev/null; then "
            f"test \"$(readlink /proc/$pid/cwd)\" = {shlex.quote(expected_cwd)}; "
            f"tr '\\0' '\\n' < /proc/$pid/cmdline | "
            f"grep -Fqx -- {shlex.quote(expected_worker)}; "
            "kill -TERM -- \"$pid\"; "
            "i=0; while kill -0 \"$pid\" 2>/dev/null && [ \"$i\" -lt 40 ]; "
            "do sleep 0.25; i=$((i+1)); done; "
            "if kill -0 \"$pid\" 2>/dev/null; then "
            f"test \"$(readlink /proc/$pid/cwd)\" = {shlex.quote(expected_cwd)}; "
            f"tr '\\0' '\\n' < /proc/$pid/cmdline | "
            f"grep -Fqx -- {shlex.quote(expected_worker)}; "
            "kill -KILL -- \"$pid\"; "
            "i=0; while kill -0 \"$pid\" 2>/dev/null && [ \"$i\" -lt 20 ]; "
            "do sleep 0.25; i=$((i+1)); done; "
            "fi; "
            "test ! -e /proc/$pid; "
            "fi"
        )
        try:
            result = _remote_run(authority, terminate, timeout=30)
        except RuntimeIntegrityError as exc:
            failures.append(
                f"remote termination failed for attested PID {pid}: {exc}"
            )
        else:
            if result.returncode != 0:
                failures.append(
                    f"remote termination failed for attested PID {pid}: "
                    f"{result.stderr.strip()}"
                )
    for active in active_arms:
        if active.process.poll() is None:
            active.process.terminate()
    for active in active_arms:
        try:
            active.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            active.process.kill()
            try:
                active.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                failures.append(
                    f"local SSH supervisor for PID {active.pid} did not exit"
                )
    devices = [active.manifest["gpu"] for active in active_arms]
    if devices:
        try:
            _require_clean_gpus(authority, devices)
        except RuntimeIntegrityError as exc:
            failures.append(f"post-termination GPU cleanliness failed: {exc}")
    if failures:
        raise RuntimeIntegrityError("; ".join(failures))


def _monitor_active(
    authority: Mapping[str, Any],
    active_arms: Sequence[ActiveArm],
) -> list[dict[str, Any]]:
    by_uuid = {
        active.manifest["gpu"]["uuid"]: active
        for active in active_arms
    }
    if len(by_uuid) != len(active_arms):
        _terminate_active(authority, active_arms)
        raise RuntimeIntegrityError("two concurrent arms were mapped to one UUID")
    deadline = time.monotonic() + 3600
    while True:
        all_done = all(active.process.poll() is not None for active in active_arms)
        for uuid, active in by_uuid.items():
            pids = _query_compute_pids(authority, uuid)
            active.inventory_samples += 1
            foreign = pids - {active.pid}
            if foreign:
                _terminate_active(authority, active_arms)
                raise RuntimeIntegrityError(
                    f"co-tenancy detected on {uuid}: expected_pid={active.pid} "
                    f"observed={sorted(pids)}"
                )
            if active.pid in pids:
                active.observed_pid = True
        if all_done:
            break
        if time.monotonic() >= deadline:
            _terminate_active(authority, active_arms)
            raise RuntimeIntegrityError("phase exceeded one-hour hard timeout")
        time.sleep(0.25)

    monitor_rows = []
    for active in active_arms:
        output = active.process.stdout.read() if active.process.stdout else ""
        forbidden_output = FORBIDDEN_OUTPUT_RE.search(output)
        if forbidden_output is not None:
            raise RuntimeIntegrityError(
                f"worker stdout exposed forbidden endpoint token "
                f"{forbidden_output.group(0)!r}"
            )
        if active.process.returncode != 0:
            raise RuntimeIntegrityError(
                f"worker {active.phase.phase_id}/{active.arm.role} exited "
                f"{active.process.returncode}: {output[-4000:]}"
            )
        if not active.observed_pid:
            raise RuntimeIntegrityError(
                f"worker PID {active.pid} was never observed on "
                f"{active.manifest['gpu']['uuid']}"
            )
        monitor_rows.append(
            {
                "phase_id": active.phase.phase_id,
                "role": active.arm.role,
                "gpu": active.manifest["gpu"],
                "pid": active.pid,
                "pid_observed": active.observed_pid,
                "inventory_sample_interval_seconds": 0.25,
                "inventory_samples": active.inventory_samples,
                "foreign_pids_observed": [],
                "stdout": output,
            }
        )
    _require_clean_gpus(
        authority, [active.manifest["gpu"] for active in active_arms]
    )
    return monitor_rows


def _load_worker_result(
    authority: Mapping[str, Any],
    active: ActiveArm,
    artifact_dir: Path,
    monitor: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    _scp_from_remote(authority, active.remote_result, active.local_result_tmp)
    raw = active.local_result_tmp.read_bytes()
    try:
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeIntegrityError(f"non-finite JSON constant {value!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeIntegrityError(f"worker result is not UTF-8 JSON: {exc}") from exc
    _validate_worker_common(
        result,
        authority=authority,
        manifest=active.manifest,
        manifest_sha256=active.manifest_sha256,
        expected_pid=active.pid,
    )
    stem = f"{active.phase.phase_id}.{active.arm.role}"
    manifest_path = artifact_dir / f"{stem}.manifest.json"
    raw_path = artifact_dir / f"{stem}.worker-result.json"
    log_path = artifact_dir / f"{stem}.supervisor.json"
    hashes = {
        str(manifest_path.relative_to(artifact_dir)): _write_exclusive(
            manifest_path, active.local_manifest.read_bytes()
        ),
        str(raw_path.relative_to(artifact_dir)): _write_exclusive(raw_path, raw),
        str(log_path.relative_to(artifact_dir)): _write_exclusive_json(
            log_path, monitor
        ),
    }
    active.local_result_tmp.unlink(missing_ok=True)
    return result, hashes


def _parse_sha256sum_output(
    stdout: str,
    *,
    expected: Mapping[str, str],
    where: str,
    relative_to: str | None = None,
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for line in stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0]):
            raise RuntimeIntegrityError(
                f"malformed {where} sha256sum line: {line!r}"
            )
        raw_path = parts[1].lstrip("*")
        if not raw_path:
            raise RuntimeIntegrityError(
                f"malformed {where} sha256sum path: {line!r}"
            )
        if relative_to is None:
            key = raw_path
        else:
            try:
                key = str(Path(raw_path).relative_to(relative_to))
            except ValueError as exc:
                raise RuntimeIntegrityError(
                    f"{where} sha256sum path escapes audited root: {raw_path!r}"
                ) from exc
        if key in observed:
            raise RuntimeIntegrityError(
                f"duplicate {where} sha256sum path: {key!r}"
            )
        observed[key] = parts[0]
    if observed != expected:
        raise RuntimeIntegrityError(
            f"{where} hashes differ: expected={dict(expected)} "
            f"observed={observed}"
        )
    return observed


def _remote_source_preflight(authority: Mapping[str, Any]) -> None:
    quoted_paths = " ".join(
        shlex.quote(f"{authority['remote']['work_dir'].rstrip('/')}/{relative}")
        for relative in REMOTE_EXECUTABLE_FILES
    )
    result = _remote_run(authority, f"sha256sum {quoted_paths}", timeout=60)
    if result.returncode != 0:
        raise RuntimeIntegrityError(
            f"remote source hash preflight failed: {result.stderr.strip()}"
        )
    _parse_sha256sum_output(
        result.stdout,
        expected=REMOTE_EXECUTABLE_FILES,
        where="remote executable source",
        relative_to=authority["remote"]["work_dir"],
    )
    executable_paths = {
        authority["remote"]["python"]: authority["remote"]["python_sha256"],
        authority["worker"]["remote_path"]: authority["worker"]["sha256"],
    }
    executable_result = _remote_run(
        authority,
        "sha256sum "
        + " ".join(shlex.quote(path) for path in executable_paths),
        timeout=60,
    )
    if executable_result.returncode != 0:
        raise RuntimeIntegrityError(
            "remote Python/worker hash preflight failed: "
            f"{executable_result.stderr.strip()}"
        )
    _parse_sha256sum_output(
        executable_result.stdout,
        expected=executable_paths,
        where="remote Python/worker",
    )


def _run_phase(
    authority: Mapping[str, Any],
    phase: Phase,
    artifact_dir: Path,
    staging_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    devices = [authority["gpus"][arm.gpu_slot] for arm in phase.arms]
    _require_exact_gpu_bindings(authority)
    _require_clean_gpus(authority, devices)
    active: list[ActiveArm] = []
    try:
        for arm in phase.arms:
            active.append(_launch_arm(authority, phase, arm, staging_dir))
        monitors = _monitor_active(authority, active)
    except Exception:
        _terminate_active(authority, active)
        raise
    results: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for active_arm, monitor in zip(active, monitors, strict=True):
        result, result_hashes = _load_worker_result(
            authority, active_arm, artifact_dir, monitor
        )
        results[active_arm.arm.role] = result
        hashes.update(result_hashes)
    return results, hashes


def execute(authority: Mapping[str, Any], confirmation_nonce: str) -> dict[str, Any]:
    if confirmation_nonce != authority["nonce"]:
        raise AuthorityError("--execute-nonce does not match detached authority")
    blockers = readiness_blockers(authority)
    if blockers:
        raise AuthorityError(
            "execution blocked before remote contact:\n- " + "\n- ".join(blockers)
        )
    # Freeze the exact ready plan before claiming the single-use output path.
    # Recomputing it after mkdir would manufacture a self-referential
    # "result path already exists" blocker in an otherwise authorized plan.
    plan = build_dry_run(authority)
    if not plan["ready"] or plan["blockers"]:
        raise AuthorityError(
            f"ready-plan recheck changed before mkdir: {plan['blockers']}"
        )
    artifact_dir = authority["result_dir"]
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(mode=0o700)
    hashes: dict[str, str] = {}
    started_at = time.time()
    final_payload: dict[str, Any]
    with tempfile.TemporaryDirectory(
        prefix=".paper020-gpas-staging-", dir=artifact_dir
    ) as staging:
        staging_dir = Path(staging)
        hashes["authority.json"] = _write_exclusive(
            artifact_dir / "authority.json",
            authority["authority_path"].read_bytes(),
        )
        hashes["execution-plan.json"] = _write_exclusive_json(
            artifact_dir / "execution-plan.json", plan
        )
        results_by_phase: dict[str, dict[str, dict[str, Any]]] = {}
        state_machine = RuntimeStateMachine()
        lock_manager = None
        try:
            lock_manager = _transaction_gpu_locks(authority)
            lock_attestation = lock_manager.__enter__()
            hashes["transaction-gpu-locks.json"] = _write_exclusive_json(
                artifact_dir / "transaction-gpu-locks.json",
                lock_attestation,
            )
            _remote_source_preflight(authority)

            def run_frozen_phase(phase: Phase) -> dict[str, dict[str, Any]]:
                state_machine.before_phase(phase)
                results, phase_hashes = _run_phase(
                    authority, phase, artifact_dir, staging_dir
                )
                results_by_phase[phase.phase_id] = results
                hashes.update(phase_hashes)
                state_machine.finish_phase(phase)
                return results

            phases = phase_plan()
            run_frozen_phase(phases[0])
            run_frozen_phase(phases[1])

            replay = evaluate_step0_replay(
                results_by_phase["replay_control_step0"]["control"],
                # Full worker result: helper parity plus canonical optimizer
                # projection and complete gradient-coverage attestation.
                results_by_phase["replay_treatment_alpha_zero_step0"]["treatment"],
            )
            state_machine.pass_replay()
            run_frozen_phase(phases[2])
            assay = results_by_phase["eager_mediator_pair_256"]
            eager_structure = _validate_eager_pair_structure(
                assay["control"]["rows"],
                assay["treatment"]["rows"],
                assay["treatment"]["counterfactual"],
            )
            applicability = evaluate_applicability(
                assay["control"]["rows"][:32]
            )
            if not applicability["passed"]:
                verdict = "KILL_ROUND1_LOCAL_SOURCE_MECHANISM_ABSENT"
                attribution = {"evaluated": False, "passed": False}
                timing = {"evaluated": False, "passed": False}
            else:
                attribution = evaluate_attribution(
                    assay["control"]["rows"],
                    assay["treatment"]["rows"],
                    assay["treatment"]["counterfactual"],
                )
                if not attribution["passed"]:
                    timing = {"evaluated": False, "passed": False}
                    verdict = "KILL_ROUND1_MECHANISM_GATE_FAILED"
                else:
                    state_machine.pass_mediator()
                    run_frozen_phase(phases[3])
                    for role, result in results_by_phase[
                        "timing_placement_1"
                    ].items():
                        _validate_compile_identity(
                            result["compile_identity"],
                            f"timing_placement_1/{role}",
                        )
                    placement_1_rows = {
                        role: result["rows"]
                        for role, result in results_by_phase[
                            "timing_placement_1"
                        ].items()
                    }
                    _validate_timing_placement_structure(
                        placement_1_rows, "timing_placement_1"
                    )
                    state_machine.pass_placement_1_integrity()
                    # Placement 2 is not allowed to consume GPU time unless
                    # every placement-1 arm has passed compiler, row-schema,
                    # token-count, and paired-batch-identity barriers.
                    run_frozen_phase(phases[4])
                    for role, result in results_by_phase[
                        "timing_placement_2_role_swapped"
                    ].items():
                        _validate_compile_identity(
                            result["compile_identity"],
                            f"timing_placement_2_role_swapped/{role}",
                        )
                    placement_2_rows = {
                        role: result["rows"]
                        for role, result in results_by_phase[
                            "timing_placement_2_role_swapped"
                        ].items()
                    }
                    _validate_timing_placement_structure(
                        placement_2_rows,
                        "timing_placement_2_role_swapped",
                    )
                    timing = evaluate_role_swapped_timing(
                        placement_1_rows,
                        placement_2_rows,
                    )
                    verdict = (
                        "PASS_DIAGNOSTIC_ONLY_ENDPOINT_PROPOSAL_REQUIRED"
                        if timing["passed"]
                        else "KILL_ROUND1_RUNTIME_GATE_FAILED"
                    )
            lock_manager.__exit__(None, None, None)
            lock_manager = None
            final_payload = {
                "schema_version": 1,
                "authority_sha256": authority["authority_sha256"],
                "experiment_id": EXPERIMENT_ID,
                "seed": DIAGNOSTIC_SEED,
                "diagnostic_only": True,
                "validation_data_access": False,
                "endpoint_scoring_performed": False,
                "sota_update_performed": False,
                "verdict": verdict,
                "zero_effect_replay": replay,
                "eager_structure": eager_structure,
                "applicability": applicability,
                "attribution": attribution,
                "timing": timing,
                "artifact_hashes": dict(sorted(hashes.items())),
                "started_unix_seconds": started_at,
                "finished_unix_seconds": time.time(),
                "immutable_output_contract": (
                    "files created with O_CREAT|O_EXCL, SHA-256 indexed, "
                    "chmod read-only after terminal write; not filesystem WORM"
                ),
                "historical_0_927183_role": (
                    "chart_origin_only_forbidden_as_control_or_arm"
                ),
                "non_claims": {
                    "endpoint_result": False,
                    "adoption": False,
                    "comparator_rebind": False,
                    "chart_point": False,
                    "sota": False,
                },
            }
        except Exception as exc:
            release_failure = ""
            if lock_manager is not None:
                try:
                    lock_manager.__exit__(*sys.exc_info())
                except Exception as release_exc:
                    release_failure = (
                        f"; transaction lock release also failed: {release_exc}"
                    )
            final_payload = {
                "schema_version": 1,
                "authority_sha256": authority["authority_sha256"],
                "experiment_id": EXPERIMENT_ID,
                "seed": DIAGNOSTIC_SEED,
                "diagnostic_only": True,
                "validation_data_access": False,
                "endpoint_scoring_performed": False,
                "sota_update_performed": False,
                "verdict": "INVALID_DIAGNOSTIC",
                "failure_type": type(exc).__name__,
                "failure": str(exc) + release_failure,
                "artifact_hashes": dict(sorted(hashes.items())),
                "started_unix_seconds": started_at,
                "finished_unix_seconds": time.time(),
                "historical_0_927183_role": (
                    "chart_origin_only_forbidden_as_control_or_arm"
                ),
            }
        hashes = _enumerate_immutable_artifacts(artifact_dir)
        final_payload["artifact_hashes"] = dict(sorted(hashes.items()))
        final_sha = _write_exclusive_json(
            artifact_dir / "FINAL.json", final_payload
        )
        _write_exclusive_json(
            artifact_dir / "FINAL.sha256.json",
            {"FINAL.json": final_sha},
        )
    for path in artifact_dir.iterdir():
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    artifact_dir.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    return final_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authority",
        type=Path,
        required=True,
        help="committed detached runtime-authority JSON",
    )
    parser.add_argument(
        "--authority-sha256",
        required=True,
        help="exact SHA-256 of the authority bytes",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the frozen runtime plan; omitted means read-only dry run",
    )
    parser.add_argument(
        "--execute-nonce",
        default="",
        help="must equal authority.single_use_nonce when --execute is present",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    authority = load_authority(args.authority, args.authority_sha256)
    if args.execute:
        result = execute(authority, args.execute_nonce)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["verdict"].startswith("PASS_") else 2
    if args.execute_nonce:
        raise AuthorityError("--execute-nonce is only valid with --execute")
    result = build_dry_run(authority)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuthorityError, RuntimeIntegrityError) as exc:
        print(f"GPAS_RUNTIME_REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
