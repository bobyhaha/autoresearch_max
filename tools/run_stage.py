#!/usr/bin/env python3
"""Run one bounded tranche of a policy-aware paired experiment.

This scheduler launches at most three new ordinary funnel pairs per invocation,
or exactly the explicit four-pair standalone qualification supported by the
policy schema. It can also launch exactly one two-GPU, fixed-step diagnostic pair
when ``--diagnostic-steps`` is explicit. Diagnostics traverse the ordinary
authorization gate but are durably marked non-scored and never advance the
funnel. Qualification consumes one durable whole-node wave only after two clean
inventories and the frozen local/remote execution-authority hashes pass.
The scheduler holds per-GPU flocks and delegates each arm to tools/run_gated.py,
which holds cooperative advisory locks and samples the exact UUID-bound physical
GPU for the full training process. Re-run an ordinary tranche only after
`evaluate-stage` reports that its previous checkpoint cleared.
"""

from __future__ import annotations

import os

import argparse
import fcntl
import hashlib
import json
import random
import select
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tools/ dir for _sshutil

from _sshutil import ssh_argv  # noqa: E402

from vibeautoresearch.core import SchemaError  # noqa: E402
from vibeautoresearch.challenges import challenge_fingerprint  # noqa: E402
from vibeautoresearch.registry import ResearchRegistry  # noqa: E402
from vibeautoresearch.search_policy import (  # noqa: E402
    QUALIFICATION_STAGE_PAIRS,
    authorizable_seed_count,
    evaluate_stage,
    validate_search_policy,
)


SCHEDULER_LOCK = "/tmp/vibeautoresearch-gpu-locks/stage-scheduler.lock"
INVENTORY_GPU_BEGIN = "QUALIFICATION_GPU_INVENTORY_BEGIN"
INVENTORY_COMPUTE_BEGIN = "QUALIFICATION_COMPUTE_INVENTORY_BEGIN"
INVENTORY_END = "QUALIFICATION_INVENTORY_END"
REMOTE_HASH_BEGIN = "QUALIFICATION_REMOTE_HASHES_BEGIN"
REMOTE_HASH_END = "QUALIFICATION_REMOTE_HASHES_END"
PACKER_BOUNDARY_BOOTSTRAP_SEED = 19_019_063
QUALIFICATION_WAVE_ATTEMPTS_PATH = (
    "experiments/runs/qualification_wave_attempts.jsonl"
)


def _gpu_ids(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from exc
    if len(result) < 2 or len(result) % 2:
        raise argparse.ArgumentTypeError("provide an even number of at least two GPU IDs")
    if any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("GPU IDs must be distinct non-negative integers")
    return result


def _diagnostic_steps(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "diagnostic steps must be an integer"
        ) from exc
    if result <= 10:
        raise argparse.ArgumentTypeError(
            "diagnostic steps must exceed the 10 compilation/warmup steps"
        )
    return result


def _diagnostic_source_snapshot_sha256(repo_root: Path) -> str:
    """Freeze the exact run-source snapshot shared by both diagnostic arms."""
    # Import lazily so ordinary stage scheduling retains its existing module
    # loading and behavior. CODE_FILES is the authoritative list used by the
    # gated runner's remote pre-training hash check.
    from run_gated import CODE_FILES, code_hashes_fingerprint

    hashes: dict[str, str] = {}
    for filename in CODE_FILES:
        path = repo_root / filename
        if not path.is_file():
            raise SchemaError(
                f"diagnostic source snapshot is missing required file {path}"
            )
        hashes[str(filename)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return code_hashes_fingerprint(hashes)


def _require_fresh_diagnostic_pair(
    experiment: Any,
    runs: list[Any],
    seed: int,
) -> None:
    """Refuse retries or repairs after either diagnostic arm was recorded."""
    prior = [
        run
        for run in runs
        if run.experiment_id == experiment.experiment_id
        and int(run.seed) == int(seed)
        and any(
            str(tag).startswith("diagnostic_offbudget")
            for tag in run.tags
        )
    ]
    if prior:
        labels = sorted(
            f"{run.arm_id}:{run.status}" for run in prior
        )
        raise SchemaError(
            f"{experiment.experiment_id}/seed{seed} already has diagnostic "
            f"RunRecord(s) {labels}; a paired diagnostic is immutable and "
            "cannot be retried or repaired one arm at a time"
        )


def _acquire_scheduler_lock(ssh: list[str], gpu_ids: list[int] | None = None) -> subprocess.Popen:
    # Per-GPU scheduler locks: a scheduler reserves ONLY the GPUs it will use, so
    # experiments on DISJOINT GPU sets run concurrently while two schedulers can
    # never target the same GPU (preserving the single-scheduler-per-GPU guarantee
    # that makes wall-clock results trusted). Nested flock -n over each requested
    # GPU fails fast if any is already reserved by another scheduler.
    lock_dir = str(Path(SCHEDULER_LOCK).parent)
    if gpu_ids:
        nested = ""
        for g in sorted(gpu_ids):
            per = shlex.quote(f"{lock_dir}/stage-scheduler-gpu{g}.lock")
            nested += f"flock -n {per} "
        remote = (
            f"mkdir -p {shlex.quote(lock_dir)} && "
            f"{nested}bash -c 'echo SCHEDULER_ADVISORY_LOCK_ACQUIRED; cat >/dev/null'"
        )
    else:
        remote = (
            f"mkdir -p {shlex.quote(lock_dir)} && "
            f"flock -n {shlex.quote(SCHEDULER_LOCK)} "
            "bash -c 'echo SCHEDULER_ADVISORY_LOCK_ACQUIRED; cat >/dev/null'"
        )
    # stderr -> tempfile (NOT a pipe): a blocked readline on stdout would otherwise
    # deadlock if the remote writes >64KB to a full stderr pipe.
    err_file = tempfile.TemporaryFile(mode="w+")
    process = subprocess.Popen(
        ssh + [remote],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err_file,
        text=True,
    )
    assert process.stdout is not None

    def _fail(reason: str) -> "RuntimeError":
        try:
            err_file.seek(0)
            stderr = err_file.read().strip()
        except Exception:
            stderr = ""
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        return RuntimeError(reason + (f": {stderr}" if stderr else ""))

    # Bounded wait for the ACQUIRED marker — if sshd itself is jammed, fail loud
    # instead of hanging the whole scheduler forever.
    ready, _, _ = select.select([process.stdout], [], [], 30)
    if not ready:
        raise _fail("remote lock host did not respond within 30s (sshd unresponsive?)")
    marker = process.stdout.readline().strip()
    if marker != "SCHEDULER_ADVISORY_LOCK_ACQUIRED":
        raise _fail(
            "another cooperating stage scheduler holds one of the requested GPUs"
        )
    return process


def _release_scheduler_lock(process: subprocess.Popen) -> None:
    if process.stdin is not None:
        process.stdin.close()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=10)


def _preflight_gpus(ssh: list[str], gpu_ids: list[int]) -> None:
    for gpu in gpu_ids:
        check = (
            f"pids=$(nvidia-smi -i {gpu} --query-compute-apps=pid "
            "--format=csv,noheader,nounits 2>/dev/null) || exit 2; "
            "test -z \"$(printf '%s' \"$pids\" | tr -d '[:space:]')\""
        )
        try:
            result = subprocess.run(
                ssh + [check], capture_output=True, text=True, timeout=60
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"GPU {gpu} preflight timed out (remote unresponsive); no pair was launched"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"GPU {gpu} is occupied or could not be inventoried; no pair was launched"
            )


def _qualification_schedule(
    experiment: Any,
    policy: dict[str, Any],
) -> tuple[tuple[int, int, str, int, str], ...]:
    """Return the immutable seed/control/treatment physical-GPU schedule.

    The search-policy validator is the primary schema boundary. These checks are
    intentionally duplicated at the scheduler boundary so a future relaxed or
    bypassed validator cannot silently remap a qualification arm.
    """
    qualification = policy.get("qualification")
    if not isinstance(qualification, dict) or not qualification:
        return ()
    raw_schedule = qualification.get("gpu_schedule")
    if not isinstance(raw_schedule, (list, tuple)):
        raise SchemaError(
            "qualification.gpu_schedule must freeze one entry per seed"
        )
    if len(raw_schedule) != QUALIFICATION_STAGE_PAIRS:
        raise SchemaError(
            "qualification.gpu_schedule must contain exactly "
            f"{QUALIFICATION_STAGE_PAIRS} seed entries"
        )
    if len(experiment.seeds) != QUALIFICATION_STAGE_PAIRS:
        raise SchemaError(
            "qualification experiment must freeze exactly "
            f"{QUALIFICATION_STAGE_PAIRS} seeds"
        )

    result: list[tuple[int, int, str, int, str]] = []
    used_indices: set[int] = set()
    used_uuids: set[str] = set()
    for offset, (seed, raw_entry) in enumerate(
        zip(experiment.seeds, raw_schedule, strict=True)
    ):
        if not isinstance(raw_entry, dict):
            raise SchemaError(
                f"qualification.gpu_schedule[{offset}] must be an object"
            )
        if set(raw_entry) != {"seed", "control", "treatment"}:
            raise SchemaError(
                f"qualification.gpu_schedule[{offset}] must contain exactly "
                "seed, control, and treatment"
            )
        scheduled_seed = raw_entry["seed"]
        if (
            isinstance(scheduled_seed, bool)
            or not isinstance(scheduled_seed, int)
            or scheduled_seed != seed
        ):
            raise SchemaError(
                "qualification.gpu_schedule must follow the frozen experiment "
                f"seed order; expected seed {seed} at index {offset}"
            )
        roles: list[tuple[int, str]] = []
        for role in ("control", "treatment"):
            raw_gpu = raw_entry[role]
            if not isinstance(raw_gpu, dict) or set(raw_gpu) != {"index", "uuid"}:
                raise SchemaError(
                    f"qualification.gpu_schedule[{offset}].{role} must contain "
                    "exactly index and uuid"
                )
            index = raw_gpu["index"]
            uuid = raw_gpu["uuid"]
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
            ):
                raise SchemaError(
                    f"qualification.gpu_schedule[{offset}].{role}.index must be "
                    "a non-negative integer"
                )
            if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
                raise SchemaError(
                    f"qualification.gpu_schedule[{offset}].{role}.uuid must be "
                    "an explicit NVIDIA GPU UUID"
                )
            if index in used_indices or uuid in used_uuids:
                raise SchemaError(
                    "qualification.gpu_schedule must assign eight distinct GPU "
                    "indices and UUIDs"
                )
            used_indices.add(index)
            used_uuids.add(uuid)
            roles.append((index, uuid))
        result.append((seed, roles[0][0], roles[0][1], roles[1][0], roles[1][1]))
    return tuple(result)


def _qualification_inventory_gate(policy: dict[str, Any]) -> tuple[int, float]:
    qualification = policy.get("qualification")
    if not isinstance(qualification, dict) or not qualification:
        raise SchemaError("qualification inventory gate requested for ordinary policy")
    raw_gate = qualification.get("inventory_gate")
    if not isinstance(raw_gate, dict) or set(raw_gate) != {
        "required_clean_snapshots",
        "minimum_interval_seconds",
        "require_full_node_clean",
    }:
        raise SchemaError(
            "qualification.inventory_gate must contain exactly "
            "required_clean_snapshots, minimum_interval_seconds, and "
            "require_full_node_clean"
        )
    snapshots = raw_gate["required_clean_snapshots"]
    interval = raw_gate["minimum_interval_seconds"]
    if (
        isinstance(snapshots, bool)
        or not isinstance(snapshots, int)
        or snapshots != 2
    ):
        raise SchemaError(
            "qualification.inventory_gate.required_clean_snapshots must be exactly 2"
        )
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not float(interval) >= 5.0
    ):
        raise SchemaError(
            "qualification.inventory_gate.minimum_interval_seconds must be at least 5"
        )
    if raw_gate["require_full_node_clean"] is not True:
        raise SchemaError(
            "qualification.inventory_gate.require_full_node_clean must be true"
        )
    return snapshots, float(interval)


def _sha256_text(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise SchemaError(f"{field} must be 64 lowercase hexadecimal characters")
    return text


def _load_qualification_execution_authority(
    root: Path,
    policy: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    """Verify and load the frozen local authority for remote runtime bytes."""
    qualification = policy.get("qualification")
    if not isinstance(qualification, dict) or not qualification:
        raise SchemaError("execution authority requested for ordinary policy")
    raw_authority = qualification.get("execution_authority")
    if not isinstance(raw_authority, dict) or set(raw_authority) != {
        "path",
        "sha256",
    }:
        raise SchemaError(
            "qualification.execution_authority must contain exactly path and sha256"
        )
    raw_path = raw_authority["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SchemaError("qualification.execution_authority.path must be non-empty")
    expected_authority_hash = _sha256_text(
        raw_authority["sha256"],
        "qualification.execution_authority.sha256",
    )

    root_resolved = root.resolve()
    declared = Path(raw_path)
    if declared.is_absolute():
        candidate = declared
    elif declared.parts and declared.parts[0] == root.name:
        candidate = root.parent / declared
    else:
        candidate = root / declared
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SchemaError(
            "qualification execution authority is missing or unreadable: "
            f"{raw_path}"
        ) from exc
    if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
        raise SchemaError(
            "qualification execution authority must resolve to a regular file "
            "inside the research root"
        )
    payload = resolved.read_bytes()
    actual_authority_hash = hashlib.sha256(payload).hexdigest()
    if actual_authority_hash != expected_authority_hash:
        raise SchemaError(
            "qualification execution authority hash mismatch: "
            f"expected {expected_authority_hash}, got {actual_authority_hash}"
        )
    try:
        authority = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError(
            "qualification execution authority must be valid UTF-8 JSON"
        ) from exc
    if not isinstance(authority, dict):
        raise SchemaError("qualification execution authority JSON must be an object")
    raw_remote_hashes = authority.get("remote_file_hashes")
    if not isinstance(raw_remote_hashes, dict) or not raw_remote_hashes:
        raise SchemaError(
            "qualification execution authority must bind non-empty "
            "remote_file_hashes"
        )
    remote_hashes: dict[str, str] = {}
    for raw_remote_path, raw_hash in raw_remote_hashes.items():
        if (
            not isinstance(raw_remote_path, str)
            or not raw_remote_path.startswith("/")
            or any(char.isspace() for char in raw_remote_path)
        ):
            raise SchemaError(
                "execution authority remote_file_hashes keys must be absolute "
                "remote paths without whitespace"
            )
        remote_hashes[raw_remote_path] = _sha256_text(
            raw_hash,
            f"execution authority remote_file_hashes[{raw_remote_path!r}]",
        )
    return expected_authority_hash, remote_hashes


def _verify_remote_execution_authority(
    ssh: list[str],
    remote_file_hashes: dict[str, str],
) -> None:
    """Batch-check every frozen remote runtime file in one SSH invocation."""
    if not remote_file_hashes:
        raise SchemaError("qualification remote_file_hashes must not be empty")
    ordered = sorted(remote_file_hashes.items())
    quoted_paths = " ".join(shlex.quote(path) for path, _hash in ordered)
    remote = (
        "set -eu; "
        f"printf '%s\\n' {shlex.quote(REMOTE_HASH_BEGIN)}; "
        f"sha256sum -- {quoted_paths}; "
        f"printf '%s\\n' {shlex.quote(REMOTE_HASH_END)}"
    )
    try:
        result = subprocess.run(
            ssh + [remote],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "qualification remote runtime hash batch timed out; "
            "no arm was authorized"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip()
        raise RuntimeError(
            "qualification remote runtime hash batch failed; no arm was authorized"
            + (f": {detail}" if detail else "")
        )

    lines = [line.strip() for line in result.stdout.splitlines()]
    try:
        start = lines.index(REMOTE_HASH_BEGIN) + 1
        end = lines.index(REMOTE_HASH_END, start)
    except ValueError as exc:
        raise RuntimeError(
            "qualification remote runtime hash batch returned malformed markers; "
            "no arm was authorized"
        ) from exc
    if lines[end + 1 :] or end - start != len(ordered):
        raise RuntimeError(
            "qualification remote runtime hash batch returned an unexpected "
            "number of rows; no arm was authorized"
        )
    for line, (expected_path, expected_hash) in zip(
        lines[start:end], ordered, strict=True
    ):
        fields = line.split()
        if len(fields) != 2:
            raise RuntimeError(
                "qualification remote runtime hash batch returned a malformed row; "
                "no arm was authorized"
            )
        actual_hash, actual_path = fields
        if actual_path != expected_path or actual_hash != expected_hash:
            raise RuntimeError(
                "qualification remote runtime hash mismatch for "
                f"{expected_path}: expected {expected_hash}, got {actual_hash}; "
                "no arm was authorized"
            )


def _query_full_gpu_inventory(
    ssh: list[str],
) -> tuple[dict[int, str], tuple[tuple[str, int], ...]]:
    """Return one UUID map plus every live compute application on the node."""
    remote = (
        "set -eu; "
        f"printf '%s\\n' {shlex.quote(INVENTORY_GPU_BEGIN)}; "
        "nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits; "
        f"printf '%s\\n' {shlex.quote(INVENTORY_COMPUTE_BEGIN)}; "
        "nvidia-smi --query-compute-apps=gpu_uuid,pid "
        "--format=csv,noheader,nounits; "
        f"printf '%s\\n' {shlex.quote(INVENTORY_END)}"
    )
    try:
        result = subprocess.run(
            ssh + [remote],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "full-node qualification GPU inventory timed out; no arm was authorized"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip()
        raise RuntimeError(
            "full-node qualification GPU inventory failed; no arm was authorized"
            + (f": {detail}" if detail else "")
        )

    lines = [line.strip() for line in result.stdout.splitlines()]
    try:
        gpu_start = lines.index(INVENTORY_GPU_BEGIN) + 1
        compute_start = lines.index(INVENTORY_COMPUTE_BEGIN, gpu_start)
        end = lines.index(INVENTORY_END, compute_start + 1)
    except ValueError as exc:
        raise RuntimeError(
            "full-node qualification GPU inventory returned malformed markers; "
            "no arm was authorized"
        ) from exc
    if lines[end + 1 :]:
        raise RuntimeError(
            "full-node qualification GPU inventory returned trailing output; "
            "no arm was authorized"
        )

    gpu_map: dict[int, str] = {}
    for line in lines[gpu_start:compute_start]:
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(
                "full-node qualification GPU inventory returned a malformed GPU row"
            )
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise RuntimeError(
                "full-node qualification GPU inventory returned a non-integer index"
            ) from exc
        uuid = fields[1]
        if index in gpu_map or not uuid.startswith("GPU-"):
            raise RuntimeError(
                "full-node qualification GPU inventory returned an invalid GPU map"
            )
        gpu_map[index] = uuid
    if not gpu_map:
        raise RuntimeError(
            "full-node qualification GPU inventory found no GPUs; no arm was authorized"
        )

    applications: list[tuple[str, int]] = []
    for line in lines[compute_start + 1 : end]:
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(
                "full-node qualification GPU inventory returned a malformed compute row"
            )
        try:
            pid = int(fields[1])
        except ValueError as exc:
            raise RuntimeError(
                "full-node qualification GPU inventory returned a non-integer PID"
            ) from exc
        applications.append((fields[0], pid))
    return gpu_map, tuple(applications)


def _preflight_qualification_gpus(
    ssh: list[str],
    schedule: tuple[tuple[int, int, str, int, str], ...],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Require two stable, clean, exact full-node inventory snapshots."""
    snapshots, interval = _qualification_inventory_gate(policy)
    expected = {
        index: uuid
        for _seed, control_index, control_uuid, treatment_index, treatment_uuid in schedule
        for index, uuid in (
            (control_index, control_uuid),
            (treatment_index, treatment_uuid),
        )
    }
    verified_snapshots: list[dict[str, Any]] = []
    for snapshot_index in range(snapshots):
        observed, applications = _query_full_gpu_inventory(ssh)
        if observed != expected:
            raise RuntimeError(
                "qualification GPU index-to-UUID map differs from the frozen "
                f"full-node schedule at inventory {snapshot_index + 1}; "
                "no arm was authorized"
            )
        if applications:
            occupied = ", ".join(
                f"{uuid}/pid{pid}" for uuid, pid in applications
            )
            raise RuntimeError(
                "qualification requires the full node to be tenant-free in both "
                f"inventories; inventory {snapshot_index + 1} found {occupied}; "
                "no arm was authorized"
            )
        verified_snapshots.append(
            {
                "snapshot": snapshot_index + 1,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "gpu_index_to_uuid": {
                    str(index): uuid for index, uuid in sorted(observed.items())
                },
                "compute_applications": [
                    {"gpu_uuid": uuid, "pid": pid}
                    for uuid, pid in applications
                ],
            }
        )
        if snapshot_index + 1 < snapshots:
            time.sleep(interval)
    return tuple(verified_snapshots)


def _qualification_wave_attempts(
    root: Path,
    experiment_id: str,
) -> list[dict[str, Any]]:
    path = root / QUALIFICATION_WAVE_ATTEMPTS_PATH
    if not path.exists():
        return []
    attempts: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record is not an object")
                    if record.get("experiment_id") == experiment_id:
                        attempts.append(record)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SchemaError(
            "qualification wave-attempt ledger is malformed or unreadable; "
            "refusing a possible retry"
        ) from exc
    return attempts


def _require_no_qualification_wave_attempt(
    root: Path,
    experiment_id: str,
) -> None:
    attempts = _qualification_wave_attempts(root, experiment_id)
    if attempts:
        wave_ids = [str(attempt.get("wave_id", "unknown")) for attempt in attempts]
        raise SchemaError(
            f"{experiment_id} qualification already has durable wave attempt(s) "
            f"{wave_ids}; one-wave execution cannot be retried"
        )


def _record_qualification_wave_attempt(
    root: Path,
    experiment: Any,
    schedule: tuple[tuple[int, int, str, int, str], ...],
    inventory_snapshots: tuple[dict[str, Any], ...],
    execution_authority_sha256: str,
    remote_file_hashes: dict[str, str],
    challenge_selection_fingerprint: str,
    scope_id: str,
) -> str:
    """Atomically consume a qualification's only launch attempt."""
    if len(inventory_snapshots) != 2:
        raise SchemaError(
            "qualification wave attempt requires exactly two verified inventories"
        )
    schedule_payload = [
        {
            "seed": seed,
            "control": {"index": control_index, "uuid": control_uuid},
            "treatment": {"index": treatment_index, "uuid": treatment_uuid},
        }
        for seed, control_index, control_uuid, treatment_index, treatment_uuid in schedule
    ]
    stable = {
        "version": 1,
        "experiment_id": experiment.experiment_id,
        "experiment_fingerprint": str(getattr(experiment, "fingerprint", "")),
        "scope_id": scope_id,
        "challenge_selection_fingerprint": challenge_selection_fingerprint,
        "execution_authority_sha256": execution_authority_sha256,
        "remote_file_hashes": dict(sorted(remote_file_hashes.items())),
        "gpu_schedule": schedule_payload,
        "inventory_snapshots": list(inventory_snapshots),
    }
    wave_id = "qwave_" + hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    record = {
        **stable,
        "wave_id": wave_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": "launch_consumed",
    }

    path = root / QUALIFICATION_WAVE_ATTEMPTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                for line in handle:
                    if not line.strip():
                        continue
                    previous = json.loads(line)
                    if not isinstance(previous, dict):
                        raise ValueError("record is not an object")
                    if previous.get("experiment_id") == experiment.experiment_id:
                        raise SchemaError(
                            f"{experiment.experiment_id} qualification wave was "
                            "already consumed; retry refused"
                        )
                handle.seek(0, os.SEEK_END)
                handle.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except SchemaError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SchemaError(
            "could not atomically consume the qualification wave attempt; "
            "no arm was launched"
        ) from exc
    return wave_id


# Hard ceiling for a whole tranche's wait. Must exceed run_gated's poll cap so a
# genuinely-slow-but-alive run isn't torn down; a stuck one fails loud instead of
# hanging the scheduler forever.
STAGE_TIMEOUT_SEC = 3 * 3600


def _packer_pair_attestation(
    control_facts: dict[str, Any],
    treatment_facts: dict[str, Any],
    *,
    bootstrap_resamples: int = 10_000,
    block_size: int = 10,
) -> tuple[dict[str, Any], list[str]]:
    """Cross-check semantic parity and estimate the sidecar token-rate gain."""
    errors: list[str] = []
    if not control_facts.get("verified"):
        errors.append("control per-arm packer integrity is not verified")
    if not treatment_facts.get("verified"):
        errors.append("treatment per-arm packer integrity is not verified")

    exact_fields = (
        "grad_accum_steps",
        "preclock_batches",
        "preclock_data_sha256",
        "preclock_boundary_sha256",
        "aa_steps",
        "aa_micro_batches",
        "aa_model_sha256",
        "aa_optimizer_sha256",
        "aa_loss_sha256",
        "aa_data_sha256",
        "aa_boundary_sha256",
        "aa_cpu_rng_sha256",
        "aa_cuda_rng_sha256",
        "aa_dynamo_graph_count",
        "activation_step",
        "ab_start_step",
        "ab_end_step",
        "ab_model_sha256",
        "ab_optimizer_sha256",
        "ab_loss_sha256",
        "ab_data_sha256",
        "ab_boundary_sha256",
        "ab_cpu_rng_sha256",
        "ab_cuda_rng_sha256",
        "dynamo_graph_count",
        "dynamo_recompile_count",
        "cudagraph_recording_count",
        "compile_ids_sha256",
        "compile_sites",
        "clean_profile_start_step",
        "terminal_steps",
        "terminal_model_sha256",
        "terminal_optimizer_sha256",
        "terminal_aa_loss_sha256",
        "terminal_aa_boundary_sha256",
        "terminal_ab_loss_sha256",
        "terminal_ab_data_sha256",
        "terminal_ab_boundary_sha256",
        "terminal_tail_loss_sha256",
        "terminal_aa_data_sha256",
    )
    mismatched_fields = [
        field
        for field in exact_fields
        if control_facts.get(field) != treatment_facts.get(field)
    ]
    if mismatched_fields:
        errors.append(
            "cross-arm exact parity failed for "
            + ",".join(mismatched_fields)
        )
    if control_facts.get("sidecar_active") is not False:
        errors.append("control activation marker must keep the scanner active")
    if treatment_facts.get("sidecar_active") is not True:
        errors.append("treatment activation marker must enable the sidecar")

    raw_control = control_facts.get("profile_step_time_ms", [])
    raw_treatment = treatment_facts.get("profile_step_time_ms", [])
    control_ms = (
        [float(value) for value in raw_control]
        if isinstance(raw_control, list)
        else []
    )
    treatment_ms = (
        [float(value) for value in raw_treatment]
        if isinstance(raw_treatment, list)
        else []
    )
    expected_profile_samples = 200
    if (
        len(control_ms) != expected_profile_samples
        or len(treatment_ms) != expected_profile_samples
    ):
        errors.append(
            "paired clean profiles must each contain exactly "
            f"{expected_profile_samples} samples"
        )

    point_ratio = 0.0
    lower_confidence_bound = 0.0
    block_count = 0
    if not errors:
        block_count = len(control_ms) // block_size
        complete_sample_count = len(control_ms)
        point_ratio = (
            sum(control_ms) / complete_sample_count
        ) / (
            sum(treatment_ms) / complete_sample_count
        )
        control_blocks = [
            sum(control_ms[index * block_size:(index + 1) * block_size])
            for index in range(block_count)
        ]
        treatment_blocks = [
            sum(treatment_ms[index * block_size:(index + 1) * block_size])
            for index in range(block_count)
        ]
        rng = random.Random(PACKER_BOUNDARY_BOOTSTRAP_SEED)
        bootstrap_ratios: list[float] = []
        for _ in range(bootstrap_resamples):
            indices = [
                rng.randrange(block_count) for _ in range(block_count)
            ]
            control_total = sum(control_blocks[index] for index in indices)
            treatment_total = sum(
                treatment_blocks[index] for index in indices
            )
            bootstrap_ratios.append(control_total / treatment_total)
        bootstrap_ratios.sort()
        lower_index = max(
            0, int(0.05 * bootstrap_resamples) - 1
        )
        lower_confidence_bound = bootstrap_ratios[lower_index]

    point_threshold = 1.054
    lcb_threshold = 1.041
    single_placement_threshold_passed = (
        not errors
        and point_ratio >= point_threshold
        and lower_confidence_bound >= lcb_threshold
    )
    result = {
        "schema_version": 1,
        "kind": "packer_boundary_parity_pair_attestation",
        "integrity_verified": not errors,
        "exact_fields_compared": list(exact_fields),
        "mismatched_exact_fields": mismatched_fields,
        "profile_samples_per_arm": len(control_ms),
        "block_size_steps": block_size,
        "bootstrap_blocks": block_count,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": PACKER_BOUNDARY_BOOTSTRAP_SEED,
        "token_rate_ratio_treatment_over_control": point_ratio,
        "token_rate_ratio_95pct_lcb": lower_confidence_bound,
        "point_threshold": point_threshold,
        "lcb_threshold": lcb_threshold,
        "single_placement_threshold_passed": (
            single_placement_threshold_passed
        ),
        "counterbalanced_profile_required": True,
        "performance_gate_passed": False,
        "verdict": (
            "REQUIRE_COUNTERBALANCED_PROFILE"
            if single_placement_threshold_passed
            else (
                "KILL_BEFORE_GPU_SCORING"
                if not errors
                else "INVALID_DIAGNOSTIC"
            )
        ),
        "errors": errors,
        "control_profile_sha256": control_facts.get(
            "profile_sha256", ""
        ),
        "treatment_profile_sha256": treatment_facts.get(
            "profile_sha256", ""
        ),
    }
    return result, errors


def _packer_run_bindings(
    records: list[Any],
    control: str,
    treatment: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Bind the pair conclusion to exactly two immutable execution records."""
    errors: list[str] = []
    if len(records) != 2:
        errors.append(
            "packer diagnostic requires exactly two new RunRecords; "
            f"observed={len(records)}"
        )
    arm_counts = {
        arm_id: sum(record.arm_id == arm_id for record in records)
        for arm_id in (control, treatment)
    }
    if arm_counts != {control: 1, treatment: 1}:
        errors.append(
            "packer diagnostic RunRecord roles must be exactly one control "
            f"and one treatment; observed={arm_counts}"
        )

    bindings = [
        {
            "run_id": record.run_id,
            "run_fingerprint": record.fingerprint,
            "arm_id": record.arm_id,
            "status": record.status,
            "config_hash": record.config_hash,
            "git_commit": record.git_commit,
            "gpu_id": record.tracker.get("gpu_id"),
            "gpu_uuid": record.tracker.get("gpu_uuid", ""),
            "code_hashes_sha256": record.tracker.get(
                "code_hashes_sha256", ""
            ),
            "challenge_id": record.tracker.get("challenge_id", ""),
            "challenge_definition_fingerprint": record.tracker.get(
                "challenge_definition_fingerprint", ""
            ),
            "challenge_selection_fingerprint": record.tracker.get(
                "challenge_selection_fingerprint", ""
            ),
        }
        for record in sorted(records, key=lambda item: item.arm_id)
    ]
    if any(binding["status"] != "complete" for binding in bindings):
        errors.append("both packer diagnostic RunRecords must be complete")
    uuids = [str(binding["gpu_uuid"]) for binding in bindings]
    if (
        len(uuids) != 2
        or any(not value.startswith("GPU-") for value in uuids)
        or len(set(uuids)) != 2
    ):
        errors.append(
            "packer diagnostic requires two distinct, bound physical GPU UUIDs"
        )
    for field in (
        "code_hashes_sha256",
        "challenge_id",
        "challenge_definition_fingerprint",
        "challenge_selection_fingerprint",
    ):
        values = {str(binding[field]) for binding in bindings}
        if len(values) != 1 or "" in values:
            errors.append(
                f"packer diagnostic RunRecords disagree on bound {field}"
            )

    if len(records) == 2 and arm_counts == {control: 1, treatment: 1}:
        by_arm = {record.arm_id: record for record in records}
        control_config = dict(
            by_arm[control].tracker.get("resolved_training_config", {})
        )
        treatment_config = dict(
            by_arm[treatment].tracker.get(
                "resolved_training_config", {}
            )
        )
        if set(control_config) != set(treatment_config):
            errors.append(
                "packer diagnostic resolved config key sets differ across arms"
            )
        else:
            differences = {
                key
                for key in control_config
                if control_config[key] != treatment_config[key]
            }
            if differences != {"PACKER_DOC_BOUNDARIES"}:
                errors.append(
                    "registered PACKER_DOC_BOUNDARIES must be the sole "
                    f"resolved arm difference; observed={sorted(differences)}"
                )
    return bindings, errors


def _finalize_packer_pair(
    root: Path,
    experiment_id: str,
    seed: int,
    control: str,
    treatment: str,
    records: list[Any],
    *,
    launch_errors: list[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Write one pair result from existing records without ever relaunching."""
    bindings, binding_errors = _packer_run_bindings(
        records, control, treatment
    )
    by_arm = {
        record.arm_id: record
        for record in records
        if record.arm_id in {control, treatment}
    }
    if (
        len(records) == 2
        and set(by_arm) == {control, treatment}
        and not launch_errors
    ):
        control_facts = dict(
            by_arm[control].tracker.get(
                "packer_boundary_integrity", {}
            )
        )
        treatment_facts = dict(
            by_arm[treatment].tracker.get(
                "packer_boundary_integrity", {}
            )
        )
        result, attestation_errors = _packer_pair_attestation(
            control_facts,
            treatment_facts,
        )
    else:
        attestation_errors = []
        result = {
            "schema_version": 1,
            "kind": "packer_boundary_parity_pair_attestation",
            "integrity_verified": False,
            "performance_gate_passed": False,
            "single_placement_threshold_passed": False,
            "counterbalanced_profile_required": True,
            "verdict": "INVALID_DIAGNOSTIC",
        }
    all_errors = [
        *(launch_errors or []),
        *binding_errors,
        *attestation_errors,
    ]
    if all_errors:
        result.update(
            {
                "integrity_verified": False,
                "performance_gate_passed": False,
                "verdict": "INVALID_DIAGNOSTIC",
                "errors": all_errors,
            }
        )
    result["run_bindings"] = bindings
    binding_payload = json.dumps(
        bindings, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    result["run_bindings_sha256"] = hashlib.sha256(
        binding_payload
    ).hexdigest()
    artifact_path = _write_packer_pair_attestation(
        root,
        experiment_id,
        seed,
        result,
    )
    return result, artifact_path


def _packer_pair_artifact_path(
    root: Path,
    experiment_id: str,
    seed: int,
) -> Path:
    """Return the experiment-specific immutable pair-artifact path."""
    artifact_dir = (
        root
        / "experiments"
        / "artifacts"
        / "paper019_round1_fa3_boundary_sidecar"
    )
    return (
        artifact_dir
        / f"diagnostic_pair_{experiment_id}_seed{int(seed)}.json"
    )


def _validate_existing_packer_pair_artifact(
    payload: Any,
    *,
    experiment_id: str,
    seed: int,
    ledger_records: list[Any],
) -> None:
    """Fail closed unless a recovered artifact binds the current run ledger."""
    if not isinstance(payload, dict):
        raise SchemaError("existing packer pair artifact must be a JSON object")
    if payload.get("experiment_id") != experiment_id:
        raise SchemaError(
            "existing packer pair artifact experiment_id does not match "
            f"{experiment_id!r}"
        )
    stored_seed = payload.get("seed")
    if (
        isinstance(stored_seed, bool)
        or not isinstance(stored_seed, int)
        or stored_seed != int(seed)
    ):
        raise SchemaError(
            "existing packer pair artifact seed does not match "
            f"{int(seed)}"
        )

    bindings = payload.get("run_bindings")
    if not isinstance(bindings, list) or any(
        not isinstance(binding, dict) for binding in bindings
    ):
        raise SchemaError(
            "existing packer pair artifact run_bindings must be a list of objects"
        )
    binding_payload = json.dumps(
        bindings, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    recomputed_sha256 = hashlib.sha256(binding_payload).hexdigest()
    if payload.get("run_bindings_sha256") != recomputed_sha256:
        raise SchemaError(
            "existing packer pair artifact run_bindings_sha256 mismatch"
        )

    relevant_records = [
        record
        for record in ledger_records
        if record.experiment_id == experiment_id
        and int(record.seed) == int(seed)
        and "diagnostic_packer_boundary_parity" in record.tags
    ]
    ledger_by_run_id = {record.run_id: record for record in relevant_records}
    if len(ledger_by_run_id) != len(relevant_records):
        raise SchemaError(
            "current packer diagnostic ledger contains duplicate run IDs"
        )
    binding_run_ids: list[str] = []
    for binding in bindings:
        run_id = binding.get("run_id")
        run_fingerprint = binding.get("run_fingerprint")
        if not isinstance(run_id, str) or not isinstance(
            run_fingerprint, str
        ):
            raise SchemaError(
                "existing packer pair artifact bindings require run_id and "
                "run_fingerprint strings"
            )
        if run_id in binding_run_ids:
            raise SchemaError(
                "existing packer pair artifact repeats a bound run ID"
            )
        binding_run_ids.append(run_id)
        record = ledger_by_run_id.get(run_id)
        if record is None or record.fingerprint != run_fingerprint:
            raise SchemaError(
                "existing packer pair artifact RunRecord fingerprint does not "
                f"match the current ledger for {run_id!r}"
            )
    if set(binding_run_ids) != set(ledger_by_run_id):
        raise SchemaError(
            "existing packer pair artifact bindings do not exactly match the "
            "current packer diagnostic RunRecords"
        )
    if payload.get("integrity_verified") is True:
        if len(bindings) != 2:
            raise SchemaError(
                "integrity-verified packer pair artifact must bind exactly two "
                "RunRecords"
            )
        if any(
            ledger_by_run_id[run_id].status != "complete"
            for run_id in binding_run_ids
        ):
            raise SchemaError(
                "integrity-verified packer pair artifact must bind two complete "
                "RunRecords"
            )


def _write_packer_pair_attestation(
    root: Path,
    experiment_id: str,
    seed: int,
    result: dict[str, Any],
) -> Path:
    """Atomically persist the immutable pair-level diagnostic conclusion."""
    artifact_dir = (
        root
        / "experiments"
        / "artifacts"
        / "paper019_round1_fa3_boundary_sidecar"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = _packer_pair_artifact_path(
        root,
        experiment_id,
        seed,
    )
    if destination.exists():
        raise SchemaError(
            f"packer diagnostic pair artifact already exists: {destination}"
        )
    payload = {
        **result,
        "experiment_id": experiment_id,
        "seed": int(seed),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=artifact_dir,
        prefix=".packer-pair-",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
        directory_fd = os.open(artifact_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _teardown(runners: "list[tuple[str, int, subprocess.Popen, socket.socket]]") -> None:
    """Terminate every spawned runner and close its death-signal socket.

    Closing the socket makes each runner's watchdog fire (freeing its GPU) even if
    the runner is wedged in an ssh call; terminate/kill is the local backstop.
    """
    for _arm, _seed, process, sock in runners:
        try:
            sock.close()  # signals the runner's watchdog to self-destruct
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
    for _arm, _seed, process, _sock in runners:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def _receive_line(connection: socket.socket, limit: int = 16_384) -> bytes:
    payload = bytearray()
    while len(payload) < limit:
        chunk = connection.recv(min(4096, limit - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
        if b"\n" in chunk:
            break
    if b"\n" not in payload:
        raise RuntimeError("runner closed its scheduler capability channel")
    return bytes(payload).split(b"\n", 1)[0]


def _spawn_authorized_runner(
    command: list[str],
    expected_request: dict[str, object],
) -> tuple[subprocess.Popen, socket.socket]:
    """Spawn one arm; approve only the exact staged tuple.

    Returns (process, scheduler_socket). The scheduler socket is kept OPEN for the
    whole run: its closure — which happens automatically when this scheduler dies,
    SIGKILL included — is the death signal the runner's watchdog blocks on, so an
    orphaned runner self-destructs instead of hanging. The caller MUST hold the
    socket and close it during teardown.
    """
    scheduler_side, runner_side = socket.socketpair()
    # 60s (not 10s): the child loads the entire registry before it requests, and a
    # large runs ledger can exceed 10s — a spurious timeout would kill a valid arm.
    scheduler_side.settimeout(60)
    command = [*command, "--scheduler-fd", str(runner_side.fileno())]
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            command,
            text=True,
            pass_fds=(runner_side.fileno(),),
        )
        runner_side.close()
        request = json.loads(_receive_line(scheduler_side).decode("utf-8"))
        if request != expected_request:
            scheduler_side.sendall(b"REFUSED\n")
            raise RuntimeError(
                "run_gated requested a tuple different from the staged authorization"
            )
        scheduler_side.sendall(b"AUTHORIZED\n")
        scheduler_side.settimeout(None)  # keep open for the run as the death signal
        return process, scheduler_side
    except (OSError, ValueError, RuntimeError):
        try:
            runner_side.close()
        except OSError:
            pass
        scheduler_side.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise


def _completed_pairs(experiment, runs) -> set[int]:
    control = next(
        str(arm["arm_id"])
        for arm in experiment.arms
        if str(arm.get("role")) == "control"
    )
    treatment = next(
        str(arm["arm_id"])
        for arm in experiment.arms
        if str(arm.get("role")) == "treatment"
    )
    recorded = {
        (run.arm_id, run.seed): run.status
        for run in runs
        if run.experiment_id == experiment.experiment_id
        and not any(
            str(tag).startswith("diagnostic_offbudget") for tag in run.tags
        )
    }
    partial = [
        seed
        for seed in experiment.seeds
        if ((control, seed) in recorded) != ((treatment, seed) in recorded)
    ]
    if partial:
        raise SchemaError(
            f"experiment has one-sided recorded arms for seeds {partial}; "
            "do not repair a wall-time pair in a different round"
        )
    invalid_pairs = [
        seed
        for seed in experiment.seeds
        if (control, seed) in recorded
        and {
            recorded[(control, seed)],
            recorded[(treatment, seed)],
        }
        != {"complete"}
    ]
    if invalid_pairs:
        raise SchemaError(
            f"experiment has failed/invalid recorded pairs for seeds {invalid_pairs}; "
            "immutable attempts cannot be silently retried"
        )
    return {
        seed
        for seed in experiment.seeds
        if recorded.get((control, seed)) == "complete"
        and recorded.get((treatment, seed)) == "complete"
    }


def _tranche_pair_limit(policy) -> int:
    """Keep ordinary funnels bounded while permitting the frozen all-GPU assay."""
    return 4 if policy.get("qualification") else 3


def _select_tranche(
    experiment: Any,
    policy: dict[str, Any],
    runs: list[Any],
    evaluation: dict[str, Any],
    allowed: int,
    completed: set[int],
    gpu_ids: list[int],
) -> tuple[list[int], tuple[tuple[int, int, str, int, str], ...]]:
    pending = [
        seed
        for index, seed in enumerate(experiment.seeds)
        if index < allowed and seed not in completed
    ]
    if not policy.get("qualification"):
        capacity = len(gpu_ids) // 2
        tranche = pending[: min(_tranche_pair_limit(policy), capacity)]
        return tranche, ()

    prior = [
        run
        for run in runs
        if run.experiment_id == experiment.experiment_id
    ]
    if prior:
        raise SchemaError(
            f"{experiment.experiment_id} qualification already has {len(prior)} "
            "run record(s); an immutable one-wave qualification cannot be "
            "continued, remapped, or retried"
        )
    frozen_seeds = list(experiment.seeds)
    if (
        len(frozen_seeds) != QUALIFICATION_STAGE_PAIRS
        or allowed != QUALIFICATION_STAGE_PAIRS
        or completed
        or int(evaluation.get("complete_pairs", 0)) != 0
        or pending != frozen_seeds
    ):
        raise SchemaError(
            f"{experiment.experiment_id} qualification must launch exactly all "
            f"{QUALIFICATION_STAGE_PAIRS} frozen pairs as one untouched wave"
        )
    schedule = _qualification_schedule(experiment, policy)
    scheduled_gpu_ids = [
        gpu
        for _seed, control_gpu, _control_uuid, treatment_gpu, _treatment_uuid in schedule
        for gpu in (control_gpu, treatment_gpu)
    ]
    if len(gpu_ids) != 2 * QUALIFICATION_STAGE_PAIRS:
        raise SchemaError(
            f"{experiment.experiment_id} qualification requires exactly "
            f"{2 * QUALIFICATION_STAGE_PAIRS} requested GPUs"
        )
    if gpu_ids != scheduled_gpu_ids:
        raise SchemaError(
            f"{experiment.experiment_id} qualification --gpus must exactly match "
            "the frozen seed-order control,treatment schedule "
            f"{','.join(str(gpu) for gpu in scheduled_gpu_ids)}; remapping is forbidden"
        )
    # Parse and fail on an invalid inventory contract before taking remote locks.
    _qualification_inventory_gate(policy)
    return frozen_seeds, schedule


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch one <=3-pair ordinary tranche, or one explicit four-pair "
            "qualification, or one explicit two-GPU non-scored diagnostic "
            "pair through the gated GPU runner."
        )
    )
    parser.add_argument("experiment_id")
    parser.add_argument("--root", default="research")
    parser.add_argument(
        "--scope",
        default=None,
        help="compatibility selector; if supplied it must match the sticky active challenge",
    )
    parser.add_argument("--gpus", type=_gpu_ids, required=True, help="e.g. 0,1,2,3,4,5")
    parser.add_argument("--ssh", required=True)
    parser.add_argument("--remote-wd", required=True)
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("OPHIS_REMOTE_PYTHON", "python3"),
    )
    parser.add_argument(
        "--diagnostic-steps",
        type=_diagnostic_steps,
        default=0,
        help=(
            "launch exactly one two-GPU non-scored parity pair at this fixed "
            "step count; ordinary funnel behavior is unchanged when omitted"
        ),
    )
    parser.add_argument(
        "--packer-boundary-parity",
        action="store_true",
        help=(
            "inject the fixed 1000-batch packer-boundary/hash parity contract "
            "into both diagnostic arms; requires --diagnostic-steps"
        ),
    )
    args = parser.parse_args(argv)
    if args.packer_boundary_parity and not args.diagnostic_steps:
        raise SchemaError(
            "--packer-boundary-parity requires --diagnostic-steps"
        )
    if args.packer_boundary_parity:
        from run_gated import PACKER_BOUNDARY_PARITY_STEPS

        if args.diagnostic_steps != PACKER_BOUNDARY_PARITY_STEPS:
            raise SchemaError(
                "--packer-boundary-parity requires --diagnostic-steps exactly "
                f"{PACKER_BOUNDARY_PARITY_STEPS}"
            )
    if args.diagnostic_steps and len(args.gpus) != 2:
        raise SchemaError(
            "governed diagnostics require exactly two GPUs: one concurrent "
            "control and one concurrent treatment"
        )

    root = Path(args.root)
    registry = ResearchRegistry(root)
    args.scope = registry.resolve_scope_id(args.scope)
    selected_challenge = registry.challenge_for_scope(args.scope)
    selection_event = registry.challenge_events()[-1]
    challenge_definition_fingerprint = challenge_fingerprint(selected_challenge)
    experiments = registry.gated_experiments.by_id()
    if args.experiment_id not in experiments:
        raise SchemaError(f"unknown gated experiment {args.experiment_id!r}")
    experiment = experiments[args.experiment_id]
    policy = validate_search_policy(experiment.search_policy, experiment)
    if args.diagnostic_steps and policy.get("qualification"):
        raise SchemaError(
            "qualification experiments cannot be repurposed as diagnostics"
        )

    runs = registry.runs.load()
    evaluation = evaluate_stage(experiment, runs)
    if evaluation["verdict"] == "invalid":
        raise SchemaError(
            f"{experiment.experiment_id} has an invalid policy checkpoint: "
            f"{evaluation['reason']}"
        )
    allowed = authorizable_seed_count(experiment, evaluation)
    completed = _completed_pairs(experiment, runs)
    tranche, qualification_schedule = _select_tranche(
        experiment,
        policy,
        runs,
        evaluation,
        allowed,
        completed,
        args.gpus,
    )
    execution_authority_sha256 = ""
    remote_file_hashes: dict[str, str] = {}
    if qualification_schedule:
        (
            execution_authority_sha256,
            remote_file_hashes,
        ) = _load_qualification_execution_authority(root, policy)
        _require_no_qualification_wave_attempt(
            root,
            experiment.experiment_id,
        )
    if not tranche:
        print(
            f"no seeds launchable: verdict={evaluation['verdict']} "
            f"complete_pairs={evaluation['complete_pairs']}"
        )
        return 0
    control = next(
        str(arm["arm_id"])
        for arm in experiment.arms
        if str(arm.get("role")) == "control"
    )
    treatment = next(
        str(arm["arm_id"])
        for arm in experiment.arms
        if str(arm.get("role")) == "treatment"
    )
    if args.diagnostic_steps:
        if len(tranche) != 1:
            raise SchemaError(
                "governed diagnostics must resolve to exactly one paired seed"
            )
        prior_diagnostic_records = [
            run
            for run in runs
            if run.experiment_id == args.experiment_id
            and int(run.seed) == int(tranche[0])
            and any(
                str(tag).startswith("diagnostic_offbudget")
                for tag in run.tags
            )
        ]
        if args.packer_boundary_parity and prior_diagnostic_records:
            existing_artifact = _packer_pair_artifact_path(
                root,
                args.experiment_id,
                tranche[0],
            )
            if existing_artifact.exists():
                existing = json.loads(
                    existing_artifact.read_text(encoding="utf-8")
                )
                _validate_existing_packer_pair_artifact(
                    existing,
                    experiment_id=args.experiment_id,
                    seed=tranche[0],
                    ledger_records=runs,
                )
                print(
                    "packer diagnostic already finalized; no relaunch: "
                    f"verdict={existing.get('verdict')} "
                    f"artifact={existing_artifact}"
                )
                return (
                    0
                    if existing.get("integrity_verified") is True
                    else 2
                )
            recovery_errors = (
                []
                if len(prior_diagnostic_records) == 2
                else [
                    "comparison-only recovery found an incomplete immutable "
                    "diagnostic pair; GPU relaunch is forbidden"
                ]
            )
            recovered, recovered_path = _finalize_packer_pair(
                root,
                args.experiment_id,
                tranche[0],
                control,
                treatment,
                prior_diagnostic_records,
                launch_errors=recovery_errors,
            )
            print(
                "packer diagnostic comparison-only recovery; no relaunch: "
                f"verdict={recovered['verdict']} "
                f"artifact={recovered_path}"
            )
            return 0 if recovered.get("integrity_verified") else 2
        _require_fresh_diagnostic_pair(
            experiment,
            runs,
            tranche[0],
        )

    setup = registry._require_current_setup(args.scope)
    frame_key = (
        setup.scope_for(args.scope)["scope_key"]
        if args.scope
        else setup.scope_key
    )
    if args.diagnostic_steps > int(frame_key["max_steps"]):
        raise SchemaError(
            f"diagnostic step budget {args.diagnostic_steps} exceeds the "
            f"selected challenge safety cap MAX_STEPS={frame_key['max_steps']}"
        )
    diagnostic_source_hashes_sha256 = (
        _diagnostic_source_snapshot_sha256(root.resolve().parent)
        if args.diagnostic_steps
        else ""
    )
    is_qualification = bool(qualification_schedule)
    if not is_qualification:
        for seed in tranche:
            for arm_id in (control, treatment):
                registry.authorize_run(
                    experiment.experiment_id,
                    arm_id,
                    seed,
                    int(frame_key["max_steps"]),
                    scope_id=args.scope,
                )
    # Hardened argv (ControlMaster mux + timeouts) for our OWN ssh subprocesses.
    # The raw args.ssh string still flows verbatim to runners and into the
    # capability-handshake tuple, so we never mutate it — only build argv here.
    ssh = ssh_argv(args.ssh)
    scheduler_lock = _acquire_scheduler_lock(ssh, args.gpus)
    # Signals -> SystemExit so they unwind through the finally that tears down
    # runners, instead of killing us and orphaning them.
    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(_sig, lambda *_: sys.exit(1))
    selection_path = root / ResearchRegistry.CHALLENGE_EVENTS_PATH
    selection_guard = None
    processes: list[tuple[str, int, subprocess.Popen, socket.socket]] = []
    qualification_inventory_snapshots: tuple[dict[str, Any], ...] = ()
    try:
        selection_guard = selection_path.open("r", encoding="utf-8")
        fcntl.flock(selection_guard.fileno(), fcntl.LOCK_SH)
        current_selection = registry.challenge_events()[-1]
        if current_selection.fingerprint != selection_event.fingerprint:
            raise RuntimeError(
                "active challenge changed after stage selection; no arm was launched"
            )
        selected_gpus = args.gpus[: 2 * len(tranche)]
        if is_qualification:
            # The paper freezes a one-shot whole-node assay. Both clean snapshots
            # therefore precede even read-only authorization, and all eight arms
            # are authorized only after the UUID map and tenant-free state hold.
            qualification_inventory_snapshots = _preflight_qualification_gpus(
                ssh,
                qualification_schedule,
                policy,
            )
            _verify_remote_execution_authority(ssh, remote_file_hashes)
            print(
                "qualification execution authority verified "
                f"sha256={execution_authority_sha256}"
            )
            for seed in tranche:
                for arm_id in (control, treatment):
                    registry.authorize_run(
                        experiment.experiment_id,
                        arm_id,
                        seed,
                        int(frame_key["max_steps"]),
                        scope_id=args.scope,
                    )
            wave_id = _record_qualification_wave_attempt(
                root,
                experiment,
                qualification_schedule,
                qualification_inventory_snapshots,
                execution_authority_sha256,
                remote_file_hashes,
                selection_event.fingerprint,
                args.scope,
            )
            print(f"qualification one-wave launch consumed wave_id={wave_id}")
        else:
            _preflight_gpus(ssh, selected_gpus)
        runner = Path(__file__).with_name("run_gated.py")
        for index, seed in enumerate(tranche):
            for arm_id, gpu in (
                (control, selected_gpus[index * 2]),
                (treatment, selected_gpus[index * 2 + 1]),
            ):
                command = [
                    sys.executable,
                    str(runner),
                    args.experiment_id,
                    arm_id,
                    "--seed",
                    str(seed),
                    "--root",
                    str(root),
                    "--gpu",
                    str(gpu),
                    "--ssh",
                    args.ssh,
                    "--remote-wd",
                    args.remote_wd,
                    "--python-bin",
                    args.python_bin,
                    "--challenge-id",
                    str(selected_challenge["challenge_id"]),
                    "--challenge-selection-fingerprint",
                    selection_event.fingerprint,
                    "--challenge-definition-fingerprint",
                    challenge_definition_fingerprint,
                    "--scope",
                    args.scope,
                ]
                if is_qualification:
                    placement = qualification_schedule[index]
                    expected_gpu_uuid = (
                        placement[2] if arm_id == control else placement[4]
                    )
                    command.extend(
                        [
                            "--qualification-runtime-attestation-sha256",
                            execution_authority_sha256,
                            "--qualification-expected-gpu-uuid",
                            expected_gpu_uuid,
                            "--qualification-wave-id",
                            wave_id,
                        ]
                    )
                if args.diagnostic_steps:
                    command.extend(
                        [
                            "--diagnostic-steps",
                            str(args.diagnostic_steps),
                            "--diagnostic-source-hashes-sha256",
                            diagnostic_source_hashes_sha256,
                        ]
                    )
                if args.packer_boundary_parity:
                    command.append("--packer-boundary-parity")
                expected_request = {
                    "experiment_id": args.experiment_id,
                    "arm_id": arm_id,
                    "seed": seed,
                    "root": str(root),
                    "gpu": gpu,
                    "ssh": args.ssh,
                    "remote_wd": args.remote_wd,
                    "python_bin": args.python_bin,
                    "scope": args.scope,
                    "challenge_id": str(selected_challenge["challenge_id"]),
                    "challenge_selection_fingerprint": selection_event.fingerprint,
                    "challenge_definition_fingerprint": challenge_definition_fingerprint,
                    "diagnostic_steps": args.diagnostic_steps,
                }
                if args.diagnostic_steps:
                    expected_request["diagnostic_source_hashes_sha256"] = (
                        diagnostic_source_hashes_sha256
                    )
                if args.packer_boundary_parity:
                    expected_request["packer_boundary_parity"] = True
                if is_qualification:
                    expected_request[
                        "qualification_runtime_attestation_sha256"
                    ] = execution_authority_sha256
                    expected_request[
                        "qualification_expected_gpu_uuid"
                    ] = expected_gpu_uuid
                    expected_request["qualification_wave_id"] = wave_id
                proc, sock = _spawn_authorized_runner(command, expected_request)
                processes.append((arm_id, seed, proc, sock))

        failures: list[str] = []
        deadline = time.monotonic() + STAGE_TIMEOUT_SEC
        for arm_id, seed, process, _sock in processes:
            remaining = max(1.0, deadline - time.monotonic())
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                failures.append(
                    f"{arm_id}/seed{seed}: stage timeout after {STAGE_TIMEOUT_SEC}s"
                )
                continue
            if code:
                failures.append(f"{arm_id}/seed{seed}: exit {code}")
        if failures:
            if args.packer_boundary_parity:
                failed_records = [
                    run
                    for run in ResearchRegistry(root).runs.load()
                    if run.experiment_id == args.experiment_id
                    and int(run.seed) == int(tranche[0])
                    and "diagnostic_packer_boundary_parity" in run.tags
                ]
                invalid_result, invalid_path = _finalize_packer_pair(
                    root,
                    args.experiment_id,
                    tranche[0],
                    control,
                    treatment,
                    failed_records,
                    launch_errors=failures,
                )
                print(
                    "packer diagnostic invalid pair persisted "
                    f"verdict={invalid_result['verdict']} "
                    f"artifact={invalid_path}"
                )
            raise RuntimeError("stage tranche failed: " + "; ".join(failures))
    finally:
        _teardown(processes)
        if selection_guard is not None:
            fcntl.flock(selection_guard.fileno(), fcntl.LOCK_UN)
            selection_guard.close()
        _release_scheduler_lock(scheduler_lock)

    if args.packer_boundary_parity:
        completed_runs = [
            run
            for run in ResearchRegistry(root).runs.load()
            if run.experiment_id == args.experiment_id
            and int(run.seed) == int(tranche[0])
            and "diagnostic_packer_boundary_parity" in run.tags
        ]
        pair_result, artifact_path = _finalize_packer_pair(
            root,
            args.experiment_id,
            tranche[0],
            control,
            treatment,
            completed_runs,
        )
        print(
            "packer diagnostic pair attestation "
            f"verdict={pair_result['verdict']} "
            "token_rate_ratio="
            f"{pair_result.get('token_rate_ratio_treatment_over_control', 0.0):.6f} "
            "lcb95="
            f"{pair_result.get('token_rate_ratio_95pct_lcb', 0.0):.6f} "
            f"artifact={artifact_path}"
        )
        if not pair_result.get("integrity_verified"):
            raise RuntimeError(
                "packer diagnostic pair integrity failed: "
                + "; ".join(pair_result.get("errors", []))
            )

    if args.diagnostic_steps:
        print(
            f"completed non-scored diagnostic pair seed={tranche[0]} "
            f"steps={args.diagnostic_steps}"
            f"{' packer_boundary_parity=1' if args.packer_boundary_parity else ''}; "
            "it is excluded from "
            "evaluate-stage and cannot advance or close the funnel"
        )
    else:
        print(
            f"completed staged tranche seeds={tranche}; run "
            f"`python -m vibeautoresearch --root {root} evaluate-stage "
            f"{args.experiment_id}` before launching more"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
