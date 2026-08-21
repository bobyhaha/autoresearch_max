#!/usr/bin/env python3
"""Bind a gated experiment arm to an exact executed config, then auto-record it.

This closes the experiment-to-ledger gap: authorize_run() only validates labels
(experiment/arm/seed/steps), and the old launchers accepted free-form TREAT_ENVS.
Here the arm's environment is resolved ONLY from its intervention record, a
canonical manifest (env + code/data hashes + config_hash) is emitted, the run is
launched with EXACTLY that env, the logged RESOLVED_CONFIG is verified against the
manifest, and an immutable RunRecord is appended automatically.

Flow:  gate --(manifest)--> launcher --(RESOLVED_CONFIG)--> verify --> RunRecord

Usage:
  # 1. emit the manifest (no execution) and see the exact launch command
  python tools/run_gated.py exp_id arm_id --seed 42 --dry-run
  # 2. execute paired work through tools/run_stage.py. Direct remote execution
  #    is rejected so an agent cannot silently launch only one arm.

The arm's exact env lives in its intervention record under parameters.env
(a mapping of ENV_VAR -> value). A control arm may resolve to an empty env
(the frozen baseline defaults).
"""
from __future__ import annotations

import os

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import random
import re
import math
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tools/ dir for _sshutil

from _sshutil import ssh_argv  # noqa: E402
from vibeautoresearch.core import canonical_json  # noqa: E402

# Set to the remote tmux session name once training is launched; the scheduler
# death-watchdog reads it to kill the session (freeing the GPU) if the parent dies.
_SESSION: "str | None" = None
from vibeautoresearch.challenges import challenge_fingerprint  # noqa: E402
from vibeautoresearch.experiments import RunRecord  # noqa: E402
from vibeautoresearch.registry import ResearchRegistry  # noqa: E402
from vibeautoresearch.search_policy import validate_search_policy  # noqa: E402

# Frozen files whose content defines "the executed code/data" for a run.
CODE_FILES = [
    "train.py",
    "lib.py",
    "observable.py",
    "prepare.py",
    "data_split.json",
]
# Env keys that are part of the frozen challenge, never an arm's free choice.
# Resolve them from setup reconciliation instead of duplicating the current
# regime here.  In particular, do not describe a same-step scope as "compute
# matched": it fixes optimizer steps and tokens, while architecture changes may
# alter FLOPs, memory traffic, and wall time per step.
def _frozen_env_from_scope_key(sk: Mapping, source: str = "scope_key") -> dict[str, str]:
    try:
        stop_mode = str(sk["stop_mode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{source} must define stop_mode") from exc
    if stop_mode not in {"steps", "time", "walltime"}:
        raise RuntimeError(
            f"unsupported stop_mode {stop_mode!r} in {source}"
        )

    env = {
        "STOP_MODE": stop_mode,
        "ATTN_BACKEND": "sdpa",
        # Align CUDA ordinal selection with NVML/PCI ordering. The remote runner
        # additionally resolves the selected index to a UUID and uses that UUID
        # for both CUDA_VISIBLE_DEVICES and every nvidia-smi query.
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    if stop_mode in {"time", "walltime"}:
        time_budget = sk.get("time_budget")
        if (
            isinstance(time_budget, bool)
            or not isinstance(time_budget, (int, float))
            or time_budget <= 0
        ):
            raise RuntimeError(
                f"{stop_mode} scope requires a positive numeric time_budget in {source}"
            )
        env["TIME_BUDGET"] = str(time_budget)
    return env


def _frozen_env_from_frame(frame: Mapping, source: str = "decision frame") -> dict[str, str]:
    """Bind budget controls plus the frame's measured non-budget environment."""
    env = _frozen_env_from_scope_key(frame["scope_key"], f"{source}.scope_key")
    run_env = frame.get("run_env") or {}
    if not isinstance(run_env, Mapping):
        raise RuntimeError(f"{source}.run_env must be an object")
    reserved = {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "MAX_STEPS",
        "SEED",
        "STOP_MODE",
        "TIME_BUDGET",
    }
    illegal = sorted(set(str(key) for key in run_env) & reserved)
    if illegal:
        raise RuntimeError(f"{source}.run_env overrides frozen budget controls: {illegal}")
    return {**env, **{str(key): str(value) for key, value in run_env.items()}}


def _diagnostic_env_from_frame(
    frame: Mapping, source: str = "decision frame"
) -> dict[str, str]:
    """Keep the current frame config but replace its stop rule with fixed steps.

    A governed parity diagnostic is still tied to the selected challenge's
    measured runtime environment (backend, compile mode, batch recipe, and time
    budget banner). Only the stop rule changes: ``MAX_STEPS`` becomes the
    scheduler-bound diagnostic budget and ``STOP_MODE=steps`` makes that budget
    effective. The retained ``TIME_BUDGET`` is inert in step mode and remains
    config-verified, so the diagnostic cannot silently drift from the live H200
    recipe.
    """
    env = _frozen_env_from_frame(frame, source)
    env["STOP_MODE"] = "steps"
    return env


def _frozen_env_from_scope():
    import json as _json
    from pathlib import Path as _Path

    setup_path = (
        _Path(__file__).resolve().parents[1]
        / "research/setup/reconciliation.json"
    )
    try:
        setup = _json.loads(setup_path.read_text(encoding="utf-8"))
        sk = setup["scope_key"]
    except (OSError, KeyError, TypeError, ValueError, _json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot resolve frozen run controls from {setup_path}: {exc}"
        ) from exc
    return _frozen_env_from_scope_key(sk, str(setup_path))


FROZEN_ENV = _frozen_env_from_scope()
RUN_CONTROL_ENV = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "MAX_STEPS",
        "TIME_BUDGET",
        "STOP_MODE",
        "SEED",
    }
)
PACKER_BOUNDARY_PARITY_ENV = {
    "PACKER_BOUNDARY_VERIFY_BATCHES": "1000",
    "PACKER_DIAGNOSTIC_HASHES": "1",
    "PACKER_SIDECAR_ACTIVATE_STEP": "20",
}
PACKER_BOUNDARY_PARITY_STEPS = 250
DIAGNOSTIC_ONLY_ENV = frozenset(PACKER_BOUNDARY_PARITY_ENV)
# FLOOR-GATE: an experiment may only reach a GPU if its pre-registered minimum effect
# of interest is at least this multiple of the seed-to-seed decision floor. This
# mechanically bars sub-floor knob screens that cannot be adopted even when they "work".
FLOOR_MULT = 2.0


class FloorGateError(SystemExit):
    pass


def _floor_gate(experiment, frame: Mapping | None = None):
    """Reject an experiment whose minimum effect of interest is below the floor.

    When running in a decision frame, the floor is THAT frame's measured
    effective_sigma -- a noisier frame demands a correspondingly larger effect.
    """
    noise = (experiment.analysis_plan or {}).get("noise_model", {})
    sigma = noise.get("effective_sigma")
    if frame:
        frame_sigma = (frame.get("baseline") or {}).get("effective_sigma")
        if not isinstance(frame_sigma, (int, float)) or isinstance(frame_sigma, bool) or frame_sigma <= 0:
            raise FloorGateError(
                f"FLOOR-GATE: frame {frame.get('scope_id')!r} has no positive measured "
                "effective_sigma; its floor is unknown so nothing can be judged in it."
            )
        sigma = frame_sigma
    min_eff = noise.get("minimum_effect")
    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or sigma <= 0
        or isinstance(min_eff, bool)
        or not isinstance(min_eff, (int, float))
    ):
        raise FloorGateError(
            f"FLOOR-GATE: {experiment.experiment_id} analysis_plan.noise_model must declare "
            "a positive numeric effective_sigma and a numeric minimum_effect before a run "
            "can reach a GPU."
        )
    if min_eff < FLOOR_MULT * sigma:
        raise FloorGateError(
            f"FLOOR-GATE REJECTED: {experiment.experiment_id} minimum_effect={min_eff} < "
            f"{FLOOR_MULT}x effective_sigma ({sigma}) = {FLOOR_MULT * sigma:.6g}. "
            "Sub-floor experiments cannot be adopted even if confirmed, so they cannot reach a "
            "GPU. Register a bet whose minimum effect of interest is >= 2x the decision floor."
        )


def _sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required run input is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes_fingerprint(code_hashes: Mapping[str, str]) -> str:
    """Fingerprint the complete, filename-keyed source snapshot."""
    return hashlib.sha256(canonical_json(dict(code_hashes)).encode()).hexdigest()


def _arm_env(registry, experiment, arm_id: str) -> tuple[dict[str, str], str, str]:
    arm = next((a for a in experiment.arms if str(a["arm_id"]) == arm_id), None)
    if arm is None:
        sys.exit(f"ERROR: arm {arm_id!r} not in {experiment.experiment_id}")
    role = arm.get("role", "treatment")
    intervention_id = arm.get("intervention_id", "")
    env: dict[str, str] = {}
    if intervention_id:
        interventions = registry.interventions.by_id()
        if intervention_id not in interventions:
            sys.exit(f"ERROR: intervention {intervention_id!r} for arm {arm_id!r} not registered")
        params = interventions[intervention_id].parameters or {}
        raw = params.get("env")
        if raw is None:
            # Early intervention records placed environment variables directly
            # in `parameters`. Preserve those executable definitions while
            # ignoring descriptive lowercase metadata such as dependency URLs.
            raw = {
                key: value
                for key, value in params.items()
                if re.fullmatch(r"[A-Z][A-Z0-9_]*", str(key))
            }
        if not isinstance(raw, Mapping):
            raise ValueError(f"intervention {intervention_id!r} parameters.env must be an object")
        env = {str(k): str(v) for k, v in raw.items()}
        reserved = sorted(set(env) & RUN_CONTROL_ENV)
        if reserved:
            raise ValueError(
                f"intervention {intervention_id!r} may not override run controls: {reserved}"
            )
        diagnostic_only = sorted(set(env) & DIAGNOSTIC_ONLY_ENV)
        if diagnostic_only:
            raise ValueError(
                f"intervention {intervention_id!r} may not set diagnostic-only "
                f"controls: {diagnostic_only}; tools/run_stage.py injects them "
                "only for its fixed packer-parity capability"
            )
        # ===== NO-OP TREATMENT GUARD =====
        # A treatment arm that exports NOTHING is byte-identical to its control, so the
        # "experiment" measures only seed noise while still reporting config_verified=True
        # (verify_config iterates the bound env -- an empty env trivially verifies).
        # This silently invalidated an entire block of results on 2026-07-28: three
        # interventions carried only descriptive lowercase parameters, so _arm_env
        # resolved env={} and every treatment ran the control configuration. Fail closed
        # instead: an intervention that cannot change the run is a registry defect.
        if not env:
            raise ValueError(
                f"intervention {intervention_id!r} resolves to an EMPTY environment: a "
                "treatment arm that exports nothing is identical to its control and can "
                "only measure noise. Add an executable parameters.env mapping (uppercase "
                "env keys) to the intervention record before gating any run against it."
            )
    return env, role, intervention_id


def _bound_env(manifest: Mapping, root: Path | None = None) -> dict[str, str]:
    """Return every setting that must appear in the run's resolved config."""
    arm_env = {str(key): str(value) for key, value in manifest["env"].items()}
    reserved = sorted(set(arm_env) & RUN_CONTROL_ENV)
    if reserved:
        raise ValueError(f"manifest env may not override run controls: {reserved}")
    diagnostic_only = sorted(set(arm_env) & DIAGNOSTIC_ONLY_ENV)
    if diagnostic_only:
        raise ValueError(
            "manifest arm env may not set diagnostic-only controls: "
            f"{diagnostic_only}"
        )
    packer_parity = manifest.get("packer_boundary_parity")
    if (
        "packer_boundary_parity" in manifest
        and packer_parity is not True
    ):
        raise ValueError(
            "packer_boundary_parity manifest marker, when present, must be true"
        )
    if packer_parity:
        if not manifest.get("diagnostic"):
            raise ValueError(
                "packer boundary parity requires a governed diagnostic manifest"
            )
        raw_diagnostic_env = manifest.get("diagnostic_env")
        if raw_diagnostic_env != PACKER_BOUNDARY_PARITY_ENV:
            raise ValueError(
                "packer boundary parity diagnostic_env must equal the fixed "
                f"contract {PACKER_BOUNDARY_PARITY_ENV}"
            )
        diagnostic_env = dict(PACKER_BOUNDARY_PARITY_ENV)
    else:
        if "diagnostic_env" in manifest:
            raise ValueError(
                "diagnostic_env is allowed only for the fixed packer-boundary "
                "parity capability"
            )
        diagnostic_env = {}
    frozen_env = {str(key): str(value) for key, value in manifest["frozen_env"].items()}
    scope_id = str(manifest.get("scope_id") or "")
    if scope_id:
        # A registered REPORTING scope supplies its own budget frame. Re-derive it from
        # the setup record rather than trusting the manifest, so a hand-edited manifest
        # still cannot invent an environment.
        from vibeautoresearch.registry import ResearchRegistry as _RR

        # M1: honour the caller's --root; re-resolving a hardcoded path made
        # authorization and env-verification read two different records.
        setup = _RR(root or (Path(__file__).resolve().parents[1] / "research")).setup_reconciliation()
        frame = _resolve_report_scope(setup, scope_id)
        expected = (
            _diagnostic_env_from_frame(
                frame,
                f"diagnostic scope {scope_id!r}",
            )
            if manifest.get("diagnostic")
            else _frozen_env_from_frame(
                frame,
                f"reporting scope {scope_id!r}",
            )
        )
        if frozen_env != expected:
            raise ValueError(
                f"manifest frozen_env for reporting scope {scope_id!r} must equal {expected}"
            )
    elif frozen_env != FROZEN_ENV:
        raise ValueError(f"manifest frozen_env must equal {FROZEN_ENV}")
    return {
        **frozen_env,
        # Non-budget frame settings are the baseline defaults, not forbidden
        # levers. An explicitly registered arm may change one (for example,
        # SDPA control versus the frame's FA3 default); the manifest binds the
        # resulting exact environment.
        **arm_env,
        # This map is never supplied by an intervention or free-form CLI input.
        # It is an exact scheduler capability injected symmetrically into both
        # arms and verified in RESOLVED_CONFIG like every other bound setting.
        **diagnostic_env,
        "MAX_STEPS": str(manifest["max_steps"]),
        "SEED": str(manifest["seed"]),
    }


def _shell_env(env: Mapping[str, str]) -> str:
    assignments = []
    for key, value in env.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"unsafe env key in manifest: {key!r}")
        assignments.append(f"{key}={shlex.quote(value)}")
    return " ".join(assignments)


def _resolve_decision_frame(setup, scope_id: str) -> dict:
    """Resolve a registered secondary DECISION FRAME, refusing anything unmeasured.

    Both frames are real metrics. A change is adopted IN A FRAME, judged against that
    frame's own baseline and floor -- never against the other frame's.
    """
    entry = setup.scope_for(scope_id)
    status = str(entry.get("status"))
    if status != "passed":
        raise RuntimeError(
            f"decision frame {scope_id!r} is status={status!r} and cannot authorize a run.\n"
            f"Measure its baseline first: run >={entry.get('min_seeds', 3)} seeds in this frame "
            "on the target GPU, "
            "then record mean/effective_sigma into its `baseline` block and set status='passed'.\n"
            "Until then the frame is registered but unmeasured, so no result from it can be "
            "judged against a noise floor."
        )
    return entry


# Backwards-compatible alias for the earlier name.
_resolve_report_scope = _resolve_decision_frame


def build_manifest(
    root: Path,
    registry,
    experiment_id: str,
    arm_id: str,
    seed: int,
    diagnostic_steps: int = 0,
    scope_id: str = "",
    *,
    packer_boundary_parity: bool = False,
):
    """Fail closed via authorize_run, then emit the canonical bound manifest.

    ``diagnostic_steps > 0`` selects governed DIAGNOSTIC mode. It still traverses
    ``authorize_run`` at the selected frame's registered budget, including the
    live challenge, hourly-paper, literature, seed, and source gates. An ordinary
    effect experiment must also pass the effect-floor/funnel gates. A one-seed
    implementation-only pilot instead passes the registry's diagnostic-only
    authority and consumes no effect-funnel stage. Only then does the manifest
    bind a fixed diagnostic step budget. The resulting RunRecord is mechanically
    excluded from effect-stage evaluation and can never drive adoption.
    """
    if (
        isinstance(diagnostic_steps, bool)
        or not isinstance(diagnostic_steps, int)
        or diagnostic_steps < 0
    ):
        raise RuntimeError("diagnostic_steps must be a non-negative integer")
    if 0 < diagnostic_steps <= 10:
        raise RuntimeError(
            "diagnostic_steps must exceed the 10 compilation/warmup steps"
        )
    if not isinstance(packer_boundary_parity, bool):
        raise RuntimeError("packer_boundary_parity must be a boolean capability")
    setup = registry.setup_reconciliation()
    challenge = registry.challenge_for_scope(scope_id)
    selection_event = registry.challenge_events()[-1]
    experiment = registry.gated_experiments.by_id()[experiment_id]
    diagnostic = diagnostic_steps > 0
    diagnostic_pilot = (
        diagnostic
        and getattr(experiment, "stage", "") == "pilot"
        and not experiment.search_policy
    )
    policy = (
        {}
        if diagnostic_pilot
        else validate_search_policy(experiment.search_policy, experiment)
    )
    if diagnostic and not scope_id:
        raise RuntimeError(
            "governed diagnostics require the sticky active challenge scope"
        )
    if diagnostic and policy.get("qualification"):
        raise RuntimeError(
            "qualification experiments cannot be repurposed as diagnostics"
        )
    if packer_boundary_parity and not diagnostic:
        raise RuntimeError(
            "packer boundary parity requires nonzero diagnostic_steps"
        )
    if (
        packer_boundary_parity
        and diagnostic_steps != PACKER_BOUNDARY_PARITY_STEPS
    ):
        raise RuntimeError(
            "packer boundary parity requires diagnostic_steps exactly "
            f"{PACKER_BOUNDARY_PARITY_STEPS}"
        )
    report_scope = None
    if scope_id:
        report_scope = _resolve_decision_frame(setup, scope_id)
        authorized_max_steps = int(report_scope["scope_key"]["max_steps"])
        # A diagnostic gets the SAME authorization as its selected decision
        # frame. Its shorter fixed-step execution budget is recorded separately.
        if diagnostic_pilot:
            auth = registry.authorize_run(
                experiment_id,
                arm_id,
                seed,
                authorized_max_steps,
                scope_id=scope_id,
                diagnostic_only=True,
            )
        else:
            auth = registry.authorize_run(
                experiment_id,
                arm_id,
                seed,
                authorized_max_steps,
                scope_id=scope_id,
            )
            # H3: judge against THIS frame's floor, not the
            # experiment's frame-agnostic one.
            _floor_gate(experiment, frame=report_scope)
        setup_fp = auth["setup_fingerprint"]
        experiment_fp = auth["experiment_fingerprint"]
    else:
        authorized_max_steps = int(setup.scope_key["max_steps"])
        # authorize_run enforces scope/arm/seed/claims and returns the fingerprints.
        if diagnostic_pilot:
            auth = registry.authorize_run(
                experiment_id,
                arm_id,
                seed,
                authorized_max_steps,
                diagnostic_only=True,
            )
        else:
            auth = registry.authorize_run(
                experiment_id,
                arm_id,
                seed,
                authorized_max_steps,
            )
            # Hard exit: no sub-floor effect experiment reaches a GPU.
            _floor_gate(experiment)
        setup_fp = auth["setup_fingerprint"]
        experiment_fp = auth["experiment_fingerprint"]
    max_steps = int(diagnostic_steps) if diagnostic else authorized_max_steps
    env, role, intervention_id = _arm_env(registry, experiment, arm_id)
    frozen_env = (
        (
            _diagnostic_env_from_frame(
                report_scope,
                f"diagnostic scope {scope_id!r}",
            )
            if diagnostic
            else _frozen_env_from_frame(
                report_scope,
                f"reporting scope {scope_id!r}",
            )
        )
        if report_scope
        else FROZEN_ENV
    )
    code_hashes = {f: _sha(root / f) for f in CODE_FILES}
    hyp = registry.hypotheses.by_id()[experiment.hypothesis_id]
    outcome_id = hyp.outcome["outcome_id"] if isinstance(hyp.outcome, dict) else str(hyp.outcome)
    outcome_fp = registry.outcomes.by_id()[outcome_id].fingerprint
    manifest = {
        "experiment_id": experiment_id,
        "hypothesis_id": experiment.hypothesis_id,
        "arm_id": arm_id,
        "role": role,
        "intervention_id": intervention_id,
        "seed": seed,
        "max_steps": max_steps,
        "diagnostic": diagnostic,
        "scope_id": scope_id,
        "challenge_id": challenge["challenge_id"],
        "challenge_definition_fingerprint": challenge_fingerprint(challenge),
        "challenge_selection_fingerprint": selection_event.fingerprint,
        "decision_frame": challenge["decision_frame"],
        "env": env,
        # A reporting scope defines its own budget frame (e.g. STOP_MODE=time +
        # TIME_BUDGET); the adopt scope's frozen env must not leak into it.
        "frozen_env": frozen_env,
        "code_hashes": code_hashes,
        "setup_fingerprint": setup_fp,
        "spec_fingerprints": {
            "experiment": experiment_fp,
            "hypothesis": experiment.hypothesis_fingerprint,
            "outcome": outcome_fp,
        },
    }
    if diagnostic:
        manifest.update(
            {
                "authorized_max_steps": authorized_max_steps,
                "diagnostic_kind": (
                    "packer_boundary_parity"
                    if packer_boundary_parity
                    else "paired_fixed_step_parity"
                ),
                "non_scored": True,
                "code_hashes_sha256": code_hashes_fingerprint(code_hashes),
            }
        )
    if packer_boundary_parity:
        manifest.update(
            {
                "packer_boundary_parity": True,
                "diagnostic_env": dict(PACKER_BOUNDARY_PARITY_ENV),
            }
        )
    qualification = policy.get("qualification") or {}
    if int(qualification.get("integrity_version", 0)) == 2:
        authority_ref = qualification["execution_authority"]
        authority_path = (registry.root / str(authority_ref["path"])).resolve()
        research_root = registry.root.resolve()
        try:
            authority_path.relative_to(research_root)
        except ValueError as exc:
            raise RuntimeError(
                "qualification execution authority resolves outside research/"
            ) from exc
        authority_bytes = authority_path.read_bytes()
        authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
        if authority_sha256 != str(authority_ref["sha256"]):
            raise RuntimeError(
                "qualification execution-authority hash changed after registration"
            )
        authority_payload = json.loads(authority_bytes)
        remote_hashes = authority_payload.get("remote_file_hashes")
        if not isinstance(remote_hashes, Mapping) or not remote_hashes:
            raise RuntimeError(
                "qualification execution authority lacks remote_file_hashes"
            )
        manifest["qualification"] = qualification
        manifest["execution_authority"] = {
            "path": str(authority_ref["path"]),
            "sha256": authority_sha256,
        }
        manifest["remote_file_hashes"] = {
            str(path): str(digest) for path, digest in remote_hashes.items()
        }
    manifest["config_hash"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()[:16]
    return manifest, experiment


def launch_command(manifest, remote_wd: str, gpu: int, python_bin: str, root: Path | None = None) -> tuple[str, str]:
    seed = manifest["seed"]
    tag = f"{manifest['experiment_id']}__{manifest['arm_id']}__s{seed}"
    if manifest.get("diagnostic"):
        tag += f"__diag{manifest['max_steps']}"  # keep off-budget logs distinct (no clobber)
    if manifest.get("scope_id"):
        # M3: without this, a frame run and a default-frame run of the same
        # exp/arm/seed write the SAME log, so one frame's evidence would point at
        # the other frame's data -- cross-frame contamination of the ledger.
        tag += f"__{manifest['scope_id']}"
    log = f"{tag}.log"
    log = re.sub(r"[^A-Za-z0-9_.-]", "_", log)  # keep the log name shell-safe
    env_map = _bound_env(manifest, root=root)
    env_str = _shell_env(env_map)
    banner = shlex.quote(f"MANIFEST_CONFIG_HASH: {manifest['config_hash']}")
    qlog, qwd, qpy = shlex.quote(log), shlex.quote(remote_wd), shlex.quote(python_bin)
    remote_log_path = f"{remote_wd.rstrip('/')}/{log}"
    qlog_absolute = shlex.quote(remote_log_path)
    hash_checks = []
    for filename, expected in manifest["code_hashes"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid code hash for {filename!r}: {expected!r}")
        hash_checks.append(
            f'test "$(sha256sum {shlex.quote(filename)} | cut -d " " -f 1)" = {expected}'
        )
    if not hash_checks:
        raise ValueError("manifest must bind at least one code hash")
    verify_code = " && ".join(hash_checks)
    lock_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(manifest["config_hash"]))
    lock_dir = "/tmp/vibeautoresearch-gpu-locks"
    body = (
        f"cd {qwd} || exit 70; "
        f"echo {banner} > {qlog}; "
        f"mkdir -p {shlex.quote(lock_dir)} || exit 71; "
        f"exec 8>{shlex.quote(f'{lock_dir}/run-{lock_tag}.lock')}; "
        f"if ! flock -n 8; then echo RUN_ADVISORY_LOCK_REJECTED=1 >> {qlog}; exit 72; fi; "
        f"exec 9>{shlex.quote(f'{lock_dir}/gpu-{int(gpu)}.lock')}; "
        f"if ! flock -n 9; then echo GPU_ADVISORY_LOCK_REJECTED={int(gpu)} >> {qlog}; exit 73; fi; "
        f"echo GPU_ADVISORY_LOCK_ACQUIRED={int(gpu)} >> {qlog}; "
        f"if ! gpu_uuid=$(nvidia-smi -i {int(gpu)} --query-gpu=uuid "
        f"--format=csv,noheader 2>/dev/null | tr -d '[:space:]'); then "
        f"echo GPU_UUID_RESOLUTION_FAILED={int(gpu)} >> {qlog}; exit 74; fi; "
        f"case \"$gpu_uuid\" in GPU-*) ;; *) "
        f"echo GPU_UUID_INVALID=$gpu_uuid >> {qlog}; exit 74;; esac; "
        f"echo GPU_UUID_BOUND=$gpu_uuid >> {qlog}; "
        f"export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
        f"export CUDA_VISIBLE_DEVICES=\"$gpu_uuid\"; "
        f"echo RUNNER_CONFIG: CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER "
        f"CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES GPU_UUID=$gpu_uuid >> {qlog}; "
        f"if ! initial_pids=$(nvidia-smi -i \"$gpu_uuid\" --query-compute-apps=pid "
        f"--format=csv,noheader,nounits 2>/dev/null); then "
        f"echo GPU_INVENTORY_FAILED=preflight >> {qlog}; exit 75; fi; "
        f"if [ -n \"$(printf '%s' \"$initial_pids\" | tr -d '[:space:]')\" ]; then "
        f"echo GPU_COTENANCY_REJECTED_PIDS=$(printf '%s' \"$initial_pids\" | tr '\\n' ',') "
        f">> {qlog}; exit 76; fi; "
        f"if ! ({verify_code}); then echo CODE_HASHES_VERIFIED=0 >> {qlog}; exit 77; fi; "
        f"echo CODE_HASHES_VERIFIED=1 >> {qlog}; "
        f"{env_str} {qpy} -u train.py >> {qlog} 2>&1 & train_pid=$!; "
        f"echo TRAIN_PROCESS_STARTED=$train_pid >> {qlog}; "
        f"contaminated=0; inventory_failed=0; training_pid_seen=0; "
        f"while kill -0 \"$train_pid\" 2>/dev/null; do "
        f"sleep 2; "
        f"if ! observed_pids=$(nvidia-smi -i \"$gpu_uuid\" --query-compute-apps=pid "
        f"--format=csv,noheader,nounits 2>/dev/null); then inventory_failed=1; break; fi; "
        f"for observed_pid in $observed_pids; do "
        f"if [ \"$observed_pid\" = \"$train_pid\" ]; then "
        f"if [ \"$training_pid_seen\" -eq 0 ]; then "
        f"echo GPU_TRAIN_PID_OBSERVED=$train_pid >> {qlog}; fi; "
        f"training_pid_seen=1; "
        f"else "
        f"echo GPU_COTENANCY_DETECTED_PID=$observed_pid >> {qlog}; "
        f"contaminated=1; break 2; fi; done; done; "
        f"if [ \"$inventory_failed\" -ne 0 ] || [ \"$contaminated\" -ne 0 ]; then "
        f"kill \"$train_pid\" 2>/dev/null || true; fi; "
        f"wait \"$train_pid\"; train_status=$?; "
        f"if [ \"$inventory_failed\" -ne 0 ]; then "
        f"echo GPU_INVENTORY_FAILED=runtime >> {qlog}; exit 78; fi; "
        f"if [ \"$contaminated\" -ne 0 ]; then exit 79; fi; "
        f"if [ \"$training_pid_seen\" -ne 1 ]; then "
        f"echo GPU_TRAIN_PID_NEVER_OBSERVED=$train_pid >> {qlog}; exit 80; fi; "
        f"if ! final_pids=$(nvidia-smi -i \"$gpu_uuid\" --query-compute-apps=pid "
        f"--format=csv,noheader,nounits 2>/dev/null); then "
        f"echo GPU_INVENTORY_FAILED=postflight >> {qlog}; exit 81; fi; "
        f"if [ -n \"$(printf '%s' \"$final_pids\" | tr -d '[:space:]')\" ]; then "
        f"echo GPU_COTENANCY_DETECTED_POST=$(printf '%s' \"$final_pids\" | tr '\\n' ',') "
        f">> {qlog}; exit 82; fi; "
        f"echo GPU_COTENANCY_SAMPLING_VERIFIED=1 >> {qlog}; "
        f"exit \"$train_status\""
    )
    cmd = (
        f"bash -lc {shlex.quote(body)}; run_status=$?; "
        f"echo EXIT_CODE=$run_status >> {qlog_absolute}; exit $run_status"
    )
    return cmd, log


def parse_resolved(text: str) -> dict[str, str]:
    out = {}
    for label in ("RESOLVED_CONFIG", "RUNNER_CONFIG"):
        out.update(parse_labeled_config(text, label))
    return out


def parse_labeled_config(
    text: str, label: str, *, require_singleton: bool = False
) -> dict[str, str]:
    """Parse only the final occurrence of one exact config banner."""
    out: dict[str, str] = {}
    matches = re.findall(rf"^{re.escape(label)}:\s*(.*)$", text, re.MULTILINE)
    if require_singleton and len(matches) != 1:
        raise ValueError(f"expected exactly one {label} line, got {len(matches)}")
    if not matches:
        return out
    for token in matches[-1].split():
        if "=" in token:
            key, value = token.split("=", 1)
            if require_singleton and key in out:
                raise ValueError(f"duplicate {label} key {key!r}")
            out[key] = value
    return out


def parse_outcome_metrics(
    text: str, *, require_singleton: bool = False
) -> dict[str, int | float]:
    """Parse exact scored and exposure endpoints from a completed run log."""
    patterns: tuple[tuple[str, str, type[int] | type[float]], ...] = (
        ("val_bpb", r"^val_bpb:\s*([0-9]+(?:\.[0-9]+)?)\s*$", float),
        ("num_steps", r"^num_steps:\s*([0-9]+)\s*$", int),
        ("total_tokens", r"^total_tokens:\s*([0-9]+)\s*$", int),
        (
            "charged_training_seconds",
            r"^charged_training_seconds:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
            float,
        ),
    )
    metrics: dict[str, int | float] = {}
    for name, pattern, cast in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if require_singleton and len(matches) != 1:
            raise ValueError(
                f"expected exactly one terminal {name} metric, got {len(matches)}"
            )
        if matches:
            metrics[name] = cast(matches[-1])
    return metrics


def verify_qualification_integrity(
    manifest: Mapping,
    text: str,
    resolved_training: Mapping[str, str],
    metrics: Mapping[str, int | float],
    runtime_attestation_sha256: str,
) -> tuple[bool, dict[str, object], list[str]]:
    """Verify post-run facts that make a four-pair qualification interpretable."""
    qualification = manifest.get("qualification")
    if not isinstance(qualification, Mapping):
        return True, {}, []
    errors: list[str] = []
    fresh = qualification["fresh_integrity"]
    expected_role = str(manifest["role"])
    expected_resolved = qualification["expected_arm_resolved"][expected_role]
    expected_keys = set(expected_resolved) | {"SEED"}
    if set(resolved_training) != expected_keys:
        errors.append(
            "resolved training key set does not equal frozen full config: "
            f"missing={sorted(expected_keys - set(resolved_training))}, "
            f"extra={sorted(set(resolved_training) - expected_keys)}"
        )
    if str(resolved_training.get("SEED")) != str(manifest["seed"]):
        errors.append(
            f"resolved SEED={resolved_training.get('SEED')!r}, "
            f"expected {manifest['seed']}"
        )

    bool_alias = {"true": "1", "false": "0"}

    def norm(value: object) -> str:
        text_value = str(value).strip()
        lowered = text_value.lower()
        if lowered in bool_alias:
            return bool_alias[lowered]
        try:
            return format(float(text_value), ".15g")
        except ValueError:
            return text_value

    for key, expected in expected_resolved.items():
        got = resolved_training.get(key)
        if got is None or norm(got) != norm(expected):
            errors.append(
                f"qualification resolved {key}={got!r}, expected {expected!r}"
            )

    compute_matches = re.findall(
        r"^compute_complete:\s*([01])\s*$", text, re.MULTILINE
    )
    compute_complete = compute_matches == ["1"]
    if not compute_complete:
        errors.append(
            "qualification requires exactly one terminal compute_complete: 1"
        )

    charged = metrics.get("charged_training_seconds")
    charged_min = float(fresh["charged_seconds_min"])
    charged_max = float(fresh["charged_seconds_max_exclusive"])
    charged_valid = (
        isinstance(charged, (int, float))
        and not isinstance(charged, bool)
        and math.isfinite(float(charged))
        and charged_min <= float(charged) < charged_max
    )
    if not charged_valid:
        errors.append(
            f"charged_training_seconds={charged!r} outside "
            f"[{charged_min},{charged_max})"
        )

    grad_matches = re.findall(
        r"^Gradient accumulation steps:\s*([0-9]+)\s*$",
        text,
        re.MULTILINE,
    )
    expected_grad = int(
        fresh["expected_grad_accum_steps"][expected_role]
    )
    grad_accum_steps = int(grad_matches[0]) if len(grad_matches) == 1 else 0
    if grad_accum_steps != expected_grad:
        errors.append(
            f"grad_accum_steps={grad_accum_steps}, expected {expected_grad}"
        )

    num_steps = metrics.get("num_steps")
    total_tokens = metrics.get("total_tokens")
    expected_batch = int(expected_resolved["TOTAL_BATCH_SIZE"])
    token_arithmetic_verified = (
        isinstance(num_steps, int)
        and isinstance(total_tokens, int)
        and total_tokens == num_steps * expected_batch
    )
    if not token_arithmetic_verified:
        errors.append(
            "total_tokens must equal num_steps * resolved TOTAL_BATCH_SIZE"
        )

    boundary_attested = len(
        re.findall(
            r"^DOC_MASK_BOUNDARY_ATTESTED=1(?:\s|$)",
            text,
            re.MULTILINE,
        )
    ) == 1
    attestation_position = text.find("DOC_MASK_BOUNDARY_ATTESTED=1")
    time_budget_position = text.find("Time budget:")
    if (
        attestation_position < 0
        or time_budget_position < 0
        or attestation_position >= time_budget_position
    ):
        boundary_attested = False
    if not boundary_attested:
        errors.append(
            "qualification requires exactly one pre-clock boundary attestation"
        )
    boundary_lines = re.findall(
        (
            r"^DOC_MASK_BOUNDARY_COUNTS_VERIFIED=1 "
            r"batches=([0-9]+) min=([0-9]+) max=([0-9]+) "
            r"sha256=([0-9a-f]{64})\s*$"
        ),
        text,
        re.MULTILINE,
    )
    counts_lines = re.findall(
        r"^DOC_MASK_BOUNDARY_COUNTS=([0-9]+(?:,[0-9]+)*)\s*$",
        text,
        re.MULTILINE,
    )
    boundary_counts_verified = len(boundary_lines) == 1 and len(counts_lines) == 1
    boundary_digest_verified = False
    boundary_batches = 0
    boundary_digest = ""
    if boundary_counts_verified:
        batch_text, declared_min, declared_max, boundary_digest = boundary_lines[0]
        counts_csv = counts_lines[0]
        counts = [int(item) for item in counts_csv.split(",")]
        boundary_batches = int(batch_text)
        recomputed = hashlib.sha256(counts_csv.encode("ascii")).hexdigest()
        expected_batches = (
            int(num_steps) * expected_grad if isinstance(num_steps, int) else -1
        )
        boundary_digest_verified = (
            len(counts) == boundary_batches == expected_batches
            and min(counts) == int(declared_min)
            and max(counts) == int(declared_max)
            and recomputed == boundary_digest
        )
    if not boundary_counts_verified:
        errors.append(
            "qualification requires unique terminal boundary-count markers"
        )
    elif not boundary_digest_verified:
        errors.append(
            "boundary-count list, batch arithmetic, or SHA-256 verification failed"
        )

    authority = manifest["execution_authority"]
    runtime_hashes_verified = (
        bool(runtime_attestation_sha256)
        and runtime_attestation_sha256 == str(authority["sha256"])
    )
    if not runtime_hashes_verified:
        errors.append(
            "scheduler runtime/data attestation hash does not match authority"
        )
    facts: dict[str, object] = {
        "compute_complete": compute_complete,
        "charged_time_verified": charged_valid,
        "token_arithmetic_verified": token_arithmetic_verified,
        "grad_accum_steps": grad_accum_steps,
        "boundary_attested": boundary_attested,
        "boundary_counts_verified": boundary_counts_verified,
        "boundary_digest_verified": boundary_digest_verified,
        "boundary_batches": boundary_batches,
        "boundary_counts_sha256": boundary_digest,
        "runtime_hashes_verified": runtime_hashes_verified,
        "execution_authority_sha256": str(authority["sha256"]),
    }
    return not errors, facts, errors


def verify_packer_boundary_diagnostic(
    manifest: Mapping,
    text: str,
) -> tuple[bool, dict[str, object], list[str]]:
    """Verify one arm's semantic attestations for the governed sidecar diagnostic."""
    if not manifest.get("packer_boundary_parity"):
        return True, {}, []

    errors: list[str] = []

    def singleton(pattern: str, label: str) -> tuple[str, ...] | None:
        matches = re.findall(pattern, text, re.MULTILINE)
        if len(matches) != 1:
            errors.append(
                f"packer diagnostic requires exactly one {label} marker; "
                f"observed={len(matches)}"
            )
            return None
        value = matches[0]
        return value if isinstance(value, tuple) else (value,)

    preclock = singleton(
        (
            r"^PACKER_BOUNDARY_VERIFY_PASSED=1 batches=([0-9]+) "
            r"data_sha256=([0-9a-f]{64}) boundary_sha256=([0-9a-f]{64}) "
            r"x_exact=([01]) y_exact=([01]) epoch_exact=([01]) "
            r"boundaries_exact=([01]) first_zero=([01]) terminal=([01]) "
            r"strictly_increasing=([01]) row_starts=([01]) "
            r"max_gap_lte_t=([01])\s*$"
        ),
        "preclock-equivalence",
    )
    aa = singleton(
        (
            r"^PACKER_DIAGNOSTIC_AA_ATTESTED=1 steps=([0-9]+) "
            r"micro_batches=([0-9]+) model_sha256=([0-9a-f]{64}) "
            r"optimizer_sha256=([0-9a-f]{64}) "
            r"loss_sha256=([0-9a-f]{64}) data_sha256=([0-9a-f]{64}) "
            r"boundary_sha256=([0-9a-f]{64}) "
            r"cpu_rng_sha256=([0-9a-f]{64}) "
            r"cuda_rng_sha256=([0-9a-f]{64}) "
            r"dynamo_graph_count=([0-9]+)\s*$"
        ),
        "A/A checkpoint",
    )
    activation = singleton(
        (
            r"^PACKER_SIDECAR_PHASE_ACTIVATED=1 step=([0-9]+) "
            r"sidecar_active=([01]) registered_sidecar_arm=([01])\s*$"
        ),
        "sidecar-activation",
    )
    ab = singleton(
        (
            r"^PACKER_DIAGNOSTIC_AB_ATTESTED=1 "
            r"start_step=([0-9]+) end_step=([0-9]+) mode=(scanner|sidecar) "
            r"model_sha256=([0-9a-f]{64}) "
            r"optimizer_sha256=([0-9a-f]{64}) "
            r"loss_sha256=([0-9a-f]{64}) data_sha256=([0-9a-f]{64}) "
            r"boundary_sha256=([0-9a-f]{64}) "
            r"cpu_rng_sha256=([0-9a-f]{64}) "
            r"cuda_rng_sha256=([0-9a-f]{64}) "
            r"scanner_micro_batches=([0-9]+) "
            r"sidecar_micro_batches=([0-9]+)\s*$"
        ),
        "A/B checkpoint",
    )
    graph = singleton(
        (
            r"^PACKER_DIAGNOSTIC_GRAPH_ATTESTED=1 "
            r"dynamo_graph_count=([0-9]+) "
            r"dynamo_recompile_count=([0-9]+) "
            r"lazy_backward_count=([0-9]+) "
            r"cudagraph_recording_count=([0-9]+) "
            r"compile_ids_sha256=([0-9a-f]{64}) "
            r"compile_sites=([^ ]+) "
            r"activation_step=([0-9]+) clean_profile_start_step=([0-9]+)\s*$"
        ),
        "runtime-graph",
    )
    profile = singleton(
        (
            r"^PACKER_DIAGNOSTIC_PROFILE_ATTESTED=1 start_step=([0-9]+) "
            r"samples=([0-9]+) expected_samples=([0-9]+) "
            r"sha256=([0-9a-f]{64})\s*$"
        ),
        "clean-profile",
    )
    profile_values = singleton(
        r"^PACKER_DIAGNOSTIC_PROFILE_MS=([0-9]+(?:\.[0-9]+)?(?:,[0-9]+(?:\.[0-9]+)?)*)\s*$",
        "clean-profile-values",
    )
    terminal = singleton(
        (
            r"^PACKER_DIAGNOSTIC_HASHES_VERIFIED=1 steps=([0-9]+) "
            r"model_sha256=([0-9a-f]{64}) "
            r"optimizer_sha256=([0-9a-f]{64}) "
            r"aa_loss_sha256=([0-9a-f]{64}) "
            r"aa_boundary_sha256=([0-9a-f]{64}) "
            r"ab_loss_sha256=([0-9a-f]{64}) "
            r"ab_data_sha256=([0-9a-f]{64}) "
            r"ab_boundary_sha256=([0-9a-f]{64}) "
            r"tail_loss_sha256=([0-9a-f]{64}) "
            r"aa_data_sha256=([0-9a-f]{64}) "
            r"scanner_micro_batches=([0-9]+) "
            r"sidecar_micro_batches=([0-9]+)\s*$"
        ),
        "terminal-hashes",
    )

    marker_positions = [
        text.find("PACKER_BOUNDARY_VERIFY_PASSED=1"),
        text.find("Time budget:"),
        text.find("PACKER_DIAGNOSTIC_AA_ATTESTED=1"),
        text.find("PACKER_SIDECAR_PHASE_ACTIVATED=1"),
        text.find("PACKER_DIAGNOSTIC_AB_ATTESTED=1"),
        text.find("PACKER_DIAGNOSTIC_GRAPH_ATTESTED=1"),
        text.find("PACKER_DIAGNOSTIC_PROFILE_ATTESTED=1"),
        text.find("PACKER_DIAGNOSTIC_PROFILE_MS="),
        text.find("PACKER_DIAGNOSTIC_HASHES_VERIFIED=1"),
    ]
    if (
        any(position < 0 for position in marker_positions)
        or marker_positions != sorted(marker_positions)
    ):
        errors.append(
            "packer diagnostic markers are missing or not in "
            "preclock/A-A/activation/graph/terminal order"
        )
    if re.search(
        r"(?:^|\n)PACKER_[A-Z0-9_]*FAILED(?:=|\s)|CUDA(?:_|\s)ERROR",
        text,
        re.IGNORECASE,
    ):
        errors.append(
            "packer diagnostic log contains a packer failure or CUDA error"
        )

    grad_matches = re.findall(
        r"^Gradient accumulation steps:\s*([0-9]+)\s*$",
        text,
        re.MULTILINE,
    )
    grad_accum_steps = (
        int(grad_matches[0]) if len(grad_matches) == 1 else 0
    )
    if grad_accum_steps <= 0:
        errors.append(
            "packer diagnostic requires exactly one positive "
            "gradient-accumulation marker"
        )

    facts: dict[str, object] = {
        "verified": False,
        "grad_accum_steps": grad_accum_steps,
    }
    expected_steps = int(manifest["max_steps"])
    if expected_steps != PACKER_BOUNDARY_PARITY_STEPS:
        errors.append(
            "packer diagnostic manifest must bind exactly "
            f"{PACKER_BOUNDARY_PARITY_STEPS} steps; "
            f"observed={expected_steps}"
        )
    expected_sidecar = int(
        str(manifest["env"].get("PACKER_DOC_BOUNDARIES", "0")) == "1"
    )
    if preclock is not None:
        (
            raw_batches,
            data_sha,
            boundary_sha,
            *raw_exact,
        ) = preclock
        preclock_exact = (
            int(raw_batches) == 1000
            and all(value == "1" for value in raw_exact)
        )
        if not preclock_exact:
            errors.append(
                "preclock marker did not attest 1000 exact batches and all "
                "boundary invariants"
            )
        facts.update(
            {
                "preclock_batches": int(raw_batches),
                "preclock_data_sha256": data_sha,
                "preclock_boundary_sha256": boundary_sha,
                "preclock_exact": preclock_exact,
            }
        )

    if aa is not None:
        (
            aa_steps,
            aa_micro_batches,
            aa_model_sha,
            aa_optimizer_sha,
            aa_loss_sha,
            aa_data_sha,
            aa_boundary_sha,
            aa_cpu_rng_sha,
            aa_cuda_rng_sha,
            aa_graphs,
        ) = aa
        if (
            int(aa_steps) != 20
            or int(aa_micro_batches) != 20 * grad_accum_steps
            or int(aa_graphs) != 1
        ):
            errors.append(
                "A/A checkpoint must cover exactly 20 optimizer steps, "
                "20*grad_accum scanner batches, and one forward graph"
            )
        facts.update(
            {
                "aa_steps": int(aa_steps),
                "aa_micro_batches": int(aa_micro_batches),
                "aa_model_sha256": aa_model_sha,
                "aa_optimizer_sha256": aa_optimizer_sha,
                "aa_loss_sha256": aa_loss_sha,
                "aa_data_sha256": aa_data_sha,
                "aa_boundary_sha256": aa_boundary_sha,
                "aa_cpu_rng_sha256": aa_cpu_rng_sha,
                "aa_cuda_rng_sha256": aa_cuda_rng_sha,
                "aa_dynamo_graph_count": int(aa_graphs),
            }
        )

    if activation is not None:
        activation_step, sidecar_active, registered_sidecar = activation
        if (
            int(activation_step) != 20
            or int(sidecar_active) != expected_sidecar
            or int(registered_sidecar) != expected_sidecar
        ):
            errors.append(
                "activation marker disagrees with step 20 or the registered arm"
            )
        facts.update(
            {
                "activation_step": int(activation_step),
                "sidecar_active": bool(int(sidecar_active)),
                "registered_sidecar_arm": bool(
                    int(registered_sidecar)
                ),
            }
        )

    if ab is not None:
        (
            ab_start,
            ab_end,
            ab_mode,
            ab_model_sha,
            ab_optimizer_sha,
            ab_loss_sha,
            ab_data_sha,
            ab_boundary_sha,
            ab_cpu_rng_sha,
            ab_cuda_rng_sha,
            ab_scanner_batches,
            ab_sidecar_batches,
        ) = ab
        expected_mode = "sidecar" if expected_sidecar else "scanner"
        expected_ab_scanner = (
            20 * grad_accum_steps
            if expected_sidecar
            else 40 * grad_accum_steps
        )
        expected_ab_sidecar = (
            20 * grad_accum_steps if expected_sidecar else 0
        )
        if (
            int(ab_start) != 20
            or int(ab_end) != 40
            or ab_mode != expected_mode
            or int(ab_scanner_batches) != expected_ab_scanner
            or int(ab_sidecar_batches) != expected_ab_sidecar
        ):
            errors.append(
                "A/B checkpoint must cover steps 20--39 and its cumulative "
                "scanner/sidecar batches must match the registered arm"
            )
        facts.update(
            {
                "ab_start_step": int(ab_start),
                "ab_end_step": int(ab_end),
                "ab_mode": ab_mode,
                "ab_model_sha256": ab_model_sha,
                "ab_optimizer_sha256": ab_optimizer_sha,
                "ab_loss_sha256": ab_loss_sha,
                "ab_data_sha256": ab_data_sha,
                "ab_boundary_sha256": ab_boundary_sha,
                "ab_cpu_rng_sha256": ab_cpu_rng_sha,
                "ab_cuda_rng_sha256": ab_cuda_rng_sha,
                "ab_scanner_micro_batches": int(ab_scanner_batches),
                "ab_sidecar_micro_batches": int(ab_sidecar_batches),
            }
        )

    if graph is not None:
        (
            graph_count,
            recompile_count,
            lazy_backward_count,
            cudagraph_count,
            compile_ids_sha,
            compile_sites,
            activation_step,
            profile_start,
        ) = graph
        if (
            int(graph_count) != 1
            or int(recompile_count) != 0
            or int(cudagraph_count) != 0
            or compile_sites != "0:0"
            or int(activation_step) != 20
            or int(profile_start) != 50
        ):
            errors.append(
                "runtime graph attestation must show exactly one training "
                "Dynamo graph compiled at 0:0, no recompile/cudagraph, and "
                "a clean window from step 50"
            )
        facts.update(
            {
                "dynamo_graph_count": int(graph_count),
                "dynamo_recompile_count": int(recompile_count),
                "lazy_backward_count": int(lazy_backward_count),
                "cudagraph_recording_count": int(cudagraph_count),
                "compile_ids_sha256": compile_ids_sha,
                "compile_sites": compile_sites,
                "clean_profile_start_step": int(profile_start),
            }
        )

    parsed_profile: list[float] = []
    if profile is not None and profile_values is not None:
        start_step, samples, expected_samples, declared_sha = profile
        profile_csv = profile_values[0]
        try:
            parsed_profile = [
                float(value) for value in profile_csv.split(",")
            ]
        except ValueError:
            parsed_profile = []
        recomputed_sha = hashlib.sha256(
            profile_csv.encode("ascii")
        ).hexdigest()
        expected_sample_count = max(0, expected_steps - 50)
        profile_ok = (
            int(start_step) == 50
            and int(samples) == int(expected_samples)
            == expected_sample_count
            == len(parsed_profile)
            and len(parsed_profile) == 200
            and all(
                math.isfinite(value) and value > 0
                for value in parsed_profile
            )
            and declared_sha == recomputed_sha
        )
        if not profile_ok:
            errors.append(
                "clean profile count, range, positivity, or SHA-256 failed"
            )
        facts.update(
            {
                "profile_samples": len(parsed_profile),
                "profile_sha256": declared_sha,
                "profile_step_time_ms": parsed_profile,
            }
        )

    if terminal is not None:
        (
            terminal_steps,
            model_sha,
            optimizer_sha,
            aa_loss_sha,
            aa_boundary_sha,
            ab_loss_sha,
            ab_data_sha,
            ab_boundary_sha,
            tail_loss_sha,
            aa_data_sha,
            scanner_batches,
            sidecar_batches,
        ) = terminal
        expected_scanner = (
            20 * grad_accum_steps
            if expected_sidecar
            else expected_steps * grad_accum_steps
        )
        expected_sidecar_batches = (
            (expected_steps - 20) * grad_accum_steps
            if expected_sidecar
            else 0
        )
        if (
            int(terminal_steps) != expected_steps
            or int(scanner_batches) != expected_scanner
            or int(sidecar_batches) != expected_sidecar_batches
        ):
            errors.append(
                "terminal step or scanner/sidecar batch arithmetic failed"
            )
        if aa is not None and (
            aa_loss_sha != facts.get("aa_loss_sha256")
            or aa_data_sha != facts.get("aa_data_sha256")
            or aa_boundary_sha != facts.get("aa_boundary_sha256")
        ):
            errors.append(
                "terminal marker does not preserve the A/A prefix digests"
            )
        facts.update(
            {
                "terminal_steps": int(terminal_steps),
                "terminal_model_sha256": model_sha,
                "terminal_optimizer_sha256": optimizer_sha,
                "terminal_aa_loss_sha256": aa_loss_sha,
                "terminal_aa_boundary_sha256": aa_boundary_sha,
                "terminal_ab_loss_sha256": ab_loss_sha,
                "terminal_ab_data_sha256": ab_data_sha,
                "terminal_ab_boundary_sha256": ab_boundary_sha,
                "terminal_tail_loss_sha256": tail_loss_sha,
                "terminal_aa_data_sha256": aa_data_sha,
                "scanner_micro_batches": int(scanner_batches),
                "sidecar_micro_batches": int(sidecar_batches),
            }
        )
        if ab is not None and (
            ab_loss_sha != facts.get("ab_loss_sha256")
            or ab_data_sha != facts.get("ab_data_sha256")
            or ab_boundary_sha != facts.get("ab_boundary_sha256")
        ):
            errors.append(
                "terminal marker does not preserve the A/B phase digests"
            )

    facts["verified"] = not errors
    return not errors, facts, errors


def verify_config(manifest, resolved: Mapping[str, str], root: Path | None = None) -> tuple[bool, list[str]]:
    """Verify arm settings and frozen run controls against RESOLVED_CONFIG."""
    # Normalize Python bool reprs so an env flag "1"/"0" matches a RESOLVED_CONFIG that
    # prints the resolved bool "True"/"False" (train.py prints e.g. SHARED_TRIGRAM_VE=True
    # for env=1). Without this, boolean levers spuriously fail config verification.
    _BOOL = {"true": "1", "false": "0"}

    def _norm(x):
        return _BOOL.get(str(x).strip().lower(), x)

    mismatches = []
    for key, value in _bound_env(manifest, root=root).items():
        got = resolved.get(key)
        # UNECHOED-FLAG GUARD: train.py must print every bound key in RESOLVED_CONFIG.
        # A key that is absent (rather than merely different) means the training script
        # never resolved it -- i.e. the flag is dead code or a typo, and the arm silently
        # ran the control configuration. Report it distinctly from a value mismatch so
        # the failure mode is unambiguous in the run log.
        if got is None:
            mismatches.append(
                f"{key}: manifest={value} but ABSENT from RESOLVED_CONFIG -- the training "
                "script never read this flag, so the arm did not apply its intervention"
            )
            continue
        got_n, val_n = _norm(got), _norm(value)
        # numeric-tolerant compare (0.0 == 0, True == 1)
        try:
            matches = abs(float(got_n) - float(val_n)) < 1e-9
        except (TypeError, ValueError):
            matches = got_n == val_n
        if not matches:
            mismatches.append(f"{key}: manifest={value} resolved={got}")
    return (not mismatches), mismatches


_ROLE_MAP = {"control": "baseline", "treatment": "treatment"}


def make_run_record(
    manifest,
    log_path,
    val_bpb,
    config_verified,
    status,
    git_commit,
    git_dirty,
    started_at,
    ended_at,
    gpu_hours=0.0,
    gpu_id=0,
    gpu_uuid="",
    num_steps=None,
    total_tokens=None,
    charged_training_seconds=None,
    execution_isolated=False,
    resolved_training_config=None,
    qualification_integrity=None,
    diagnostic_integrity=None,
):
    seed = manifest["seed"]
    rid = f"run_{manifest['experiment_id'][:20]}_{manifest['arm_id'][:16]}_s{seed}_{manifest['config_hash']}"
    rid = re.sub(r"[^a-z0-9_]", "_", rid.lower())
    return RunRecord(
        run_id=rid,
        experiment_id=manifest["experiment_id"],
        hypothesis_id=manifest["hypothesis_id"],
        arm_id=manifest["arm_id"],
        seed=seed,
        role=_ROLE_MAP.get(manifest["role"], manifest["role"]),
        status=status,
        git_commit=git_commit or "uncommitted",
        git_dirty=git_dirty,
        config_hash=manifest["config_hash"],
        spec_fingerprints=manifest["spec_fingerprints"],
        tracker={
            "kind": "remote_tmux",
            "log": log_path,
            "gpu_id": int(gpu_id),
            "gpu_uuid": str(gpu_uuid),
            "gpu_sampling_verified": bool(execution_isolated),
            "challenge_id": str(manifest["challenge_id"]),
            "challenge_definition_fingerprint": str(
                manifest["challenge_definition_fingerprint"]
            ),
            "challenge_selection_fingerprint": str(
                manifest["challenge_selection_fingerprint"]
            ),
            **(
                {
                    "diagnostic": True,
                    "non_scored": True,
                    "diagnostic_kind": str(manifest["diagnostic_kind"]),
                    "authorized_max_steps": int(
                        manifest["authorized_max_steps"]
                    ),
                    "executed_max_steps": int(manifest["max_steps"]),
                    "code_hashes_sha256": str(
                        manifest["code_hashes_sha256"]
                    ),
                    **(
                        {
                            "packer_boundary_parity": True,
                            "diagnostic_env": dict(
                                manifest["diagnostic_env"]
                            ),
                        }
                        if manifest.get("packer_boundary_parity")
                        else {}
                    ),
                }
                if manifest.get("diagnostic")
                else {}
            ),
            **(
                {"resolved_training_config": dict(resolved_training_config)}
                if resolved_training_config is not None
                else {}
            ),
            **(
                dict(qualification_integrity)
                if qualification_integrity is not None
                else {}
            ),
            **(
                {"packer_boundary_integrity": dict(diagnostic_integrity)}
                if diagnostic_integrity is not None
                else {}
            ),
        },
        artifact_paths={"log": log_path},
        intervention_events_path="none",
        outcome_values={
            **({"val_bpb": val_bpb} if val_bpb is not None else {}),
            **({"num_steps": int(num_steps)} if num_steps is not None else {}),
            **(
                {"total_tokens": int(total_tokens)}
                if total_tokens is not None
                else {}
            ),
            **(
                {"charged_training_seconds": float(charged_training_seconds)}
                if charged_training_seconds is not None
                else {}
            ),
            **(
                {
                    "grad_accum_steps": int(
                        qualification_integrity["grad_accum_steps"]
                    )
                }
                if qualification_integrity
                and qualification_integrity.get("grad_accum_steps")
                else {}
            ),
        },
        actual_cost={"gpu_hours": gpu_hours, "currency_cost": 0.0, "observable_overhead_pct": 0.0},
        started_at=started_at,
        ended_at=ended_at,
        failure_reason="" if status == "complete" else "config_mismatch_or_nonzero_exit",
        tags=(
            "bound_run",
            "config_verified" if config_verified else "config_unverified",
            (
                "gpu_sampling_verified"
                if execution_isolated
                else "gpu_sampling_unverified"
            ),
            *(
                (
                    f"diagnostic_offbudget_{manifest['max_steps']}steps",
                    "diagnostic_non_scored",
                    "diagnostic_pair_symmetric",
                    *(
                        ("diagnostic_packer_boundary_parity",)
                        if manifest.get("packer_boundary_parity")
                        else ()
                    ),
                )
                if manifest.get("diagnostic")
                else ()
            ),
            *((f"frame_{manifest['scope_id']}",) if manifest.get("scope_id") else ()),
            f"challenge_{manifest['challenge_id']}",
        ),
    )


def append_run(root: Path, record: RunRecord):
    ResearchRegistry(root).runs.add(record)


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
        raise RuntimeError("scheduler capability channel closed without a response")
    return bytes(payload).split(b"\n", 1)[0]


def _scheduler_request(args: argparse.Namespace) -> dict[str, object]:
    request: dict[str, object] = {
        "experiment_id": args.experiment_id,
        "arm_id": args.arm_id,
        "seed": args.seed,
        "root": args.root,
        "gpu": args.gpu,
        "ssh": args.ssh,
        "remote_wd": args.remote_wd,
        "python_bin": args.python_bin,
        "scope": args.scope,
        "challenge_id": args.challenge_id,
        "challenge_selection_fingerprint": args.challenge_selection_fingerprint,
        "challenge_definition_fingerprint": args.challenge_definition_fingerprint,
        "diagnostic_steps": args.diagnostic_steps,
    }
    if args.diagnostic_steps:
        request["diagnostic_source_hashes_sha256"] = (
            args.diagnostic_source_hashes_sha256
        )
    if getattr(args, "packer_boundary_parity", False):
        request["packer_boundary_parity"] = True
    runtime_attestation = getattr(
        args, "qualification_runtime_attestation_sha256", ""
    )
    if runtime_attestation:
        request["qualification_runtime_attestation_sha256"] = (
            runtime_attestation
        )
        request["qualification_expected_gpu_uuid"] = (
            getattr(args, "qualification_expected_gpu_uuid", "")
        )
        request["qualification_wave_id"] = getattr(
            args, "qualification_wave_id", ""
        )
    return request


def _require_scheduler_capability(args: argparse.Namespace) -> None:
    """Require approval over a live inherited channel from run_stage.py.

    SECURITY BOUNDARY, stated honestly: this is NOT cryptographic. The channel is
    an inherited fd and the reply is a bare ``AUTHORIZED`` with no secret, nonce,
    or signature, so a local user who controls their own process tree can forge a
    responder. What it *does* guarantee is that a naive direct ``--ssh`` launch
    fails closed (no fd), and that the launched arm matches the exact tuple the
    scheduler staged. The real authorization gate is ``registry.authorize_run``
    (funnel stage, challenge, seed pre-registration, GPU isolation) which every
    path still traverses; a forged handshake bypasses only run_stage's atomic
    pairing and host flock, never the funnel or the active challenge.
    """
    if not args.ssh:
        return
    if args.scheduler_fd < 0:
        raise SystemExit(
            "ERROR: direct remote execution is disabled; use tools/run_stage.py "
            "so both paired arms are pre-authorized and scheduled together"
        )
    connection = None
    try:
        connection = socket.socket(fileno=args.scheduler_fd)
        connection.settimeout(60)  # symmetric with the scheduler's 60s handshake window
        connection.sendall(
            canonical_json(_scheduler_request(args)).encode("utf-8") + b"\n"
        )
        response = _receive_line(connection)
    except (OSError, RuntimeError) as exc:
        if connection is not None:
            connection.close()
        raise SystemExit(f"ERROR: scheduler capability verification failed: {exc}") from None
    if response != b"AUTHORIZED":
        connection.close()
        raise SystemExit("ERROR: scheduler refused this exact arm launch")

    # AUTHORIZED: keep the socket OPEN as the parent-death signal. A daemon watchdog
    # blocks on recv; the instant the scheduler dies (SIGKILL included) recv returns
    # EOF, and we kill the remote tmux session (freeing the GPU) and exit — so a
    # killed scheduler leaves NO orphaned runner hanging on ssh. An unrecorded run
    # can never become a RunRecord, so killing it is strictly correct here.
    connection.settimeout(None)
    ssh_prefix = args.ssh

    def _watch(conn: socket.socket) -> None:
        try:
            while conn.recv(1):
                pass
        except OSError:
            pass
        if _SESSION:
            try:
                subprocess.run(
                    ssh_argv(ssh_prefix) + [f"tmux kill-session -t {_SESSION}"],
                    timeout=30, capture_output=True,
                )
            except Exception:
                pass
        os._exit(3)

    threading.Thread(target=_watch, args=(connection,), daemon=True).start()


def _main(selection_stack: ExitStack) -> int:
    ap = argparse.ArgumentParser(description="Bind a gated arm to an exact config and auto-record the run.")
    ap.add_argument("experiment_id")
    ap.add_argument("arm_id")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--root", default="research")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ssh", default="", help="ssh prefix, e.g. 'ssh -p 50002 -o BatchMode=yes user@host'")
    ap.add_argument("--remote-wd", default="")
    ap.add_argument("--python-bin", default=os.environ.get("OPHIS_REMOTE_PYTHON", "python3"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scheduler-fd", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--challenge-id", default="", help=argparse.SUPPRESS)
    ap.add_argument(
        "--challenge-selection-fingerprint", default="", help=argparse.SUPPRESS
    )
    ap.add_argument(
        "--challenge-definition-fingerprint", default="", help=argparse.SUPPRESS
    )
    ap.add_argument(
        "--qualification-runtime-attestation-sha256",
        default="",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--qualification-expected-gpu-uuid",
        default="",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--qualification-wave-id",
        default="",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--scope",
        default=None,
        help="compatibility selector; if supplied it must match the sticky active challenge",
    )
    ap.add_argument(
        "--diagnostic-steps",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--diagnostic-source-hashes-sha256",
        default="",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--packer-boundary-parity",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--print-env",
        action="store_true",
        help="print only the shell-quoted bound environment for another launcher",
    )
    args = ap.parse_args()
    if args.packer_boundary_parity and not args.diagnostic_steps:
        raise SystemExit(
            "ERROR: --packer-boundary-parity requires --diagnostic-steps"
        )
    if (
        args.packer_boundary_parity
        and args.diagnostic_steps != PACKER_BOUNDARY_PARITY_STEPS
    ):
        raise SystemExit(
            "ERROR: --packer-boundary-parity requires --diagnostic-steps exactly "
            f"{PACKER_BOUNDARY_PARITY_STEPS}"
        )
    if args.diagnostic_steps:
        if args.diagnostic_steps <= 10:
            raise SystemExit(
                "ERROR: diagnostic steps must exceed the 10 compilation/warmup steps"
            )
        if args.scheduler_fd < 0:
            raise SystemExit(
                "ERROR: diagnostic execution is internal to tools/run_stage.py; "
                "a live scheduler capability is required"
            )
    elif args.diagnostic_source_hashes_sha256:
        raise SystemExit(
            "ERROR: diagnostic source-hash capability supplied for an ordinary run"
        )
    root = Path(args.root)
    if args.ssh:
        # Keep the append-only selection ledger read-locked through the actual
        # remote tmux spawn. A stop/change event therefore cannot land after
        # capability approval but before this arm is launched. ExitStack also
        # releases the lock on every earlier validation error or return path.
        selection_guard = selection_stack.enter_context(
            (root / ResearchRegistry.CHALLENGE_EVENTS_PATH).open(
                "r", encoding="utf-8"
            )
        )
        fcntl.flock(selection_guard.fileno(), fcntl.LOCK_SH)
        selection_stack.callback(
            fcntl.flock, selection_guard.fileno(), fcntl.LOCK_UN
        )
    registry = ResearchRegistry(root)
    args.scope = registry.resolve_scope_id(args.scope)
    challenge = registry.challenge_for_scope(args.scope)
    selection_event = registry.challenge_events()[-1]
    if args.ssh and args.scheduler_fd < 0:
        _require_scheduler_capability(args)
    if args.ssh:
        expected_challenge = {
            "challenge_id": str(challenge["challenge_id"]),
            "challenge_selection_fingerprint": selection_event.fingerprint,
            "challenge_definition_fingerprint": challenge_fingerprint(challenge),
        }
        received_challenge = {
            "challenge_id": args.challenge_id,
            "challenge_selection_fingerprint": args.challenge_selection_fingerprint,
            "challenge_definition_fingerprint": args.challenge_definition_fingerprint,
        }
        if received_challenge != expected_challenge:
            raise SystemExit(
                "ERROR: scheduler challenge capability is stale or does not match "
                "the sticky active challenge"
            )
    _require_scheduler_capability(args)

    manifest, _ = build_manifest(
        root.resolve().parent,
        registry,
        args.experiment_id,
        args.arm_id,
        args.seed,
        diagnostic_steps=args.diagnostic_steps,
        scope_id=args.scope,
        packer_boundary_parity=args.packer_boundary_parity,
    )
    if manifest.get("diagnostic"):
        expected_source_hash = args.diagnostic_source_hashes_sha256
        if not re.fullmatch(r"[0-9a-f]{64}", expected_source_hash):
            raise SystemExit(
                "ERROR: governed diagnostic lacks a valid scheduler-bound "
                "source-hash fingerprint"
            )
        actual_source_hash = code_hashes_fingerprint(manifest["code_hashes"])
        if actual_source_hash != expected_source_hash:
            raise SystemExit(
                "ERROR: diagnostic source files changed after the scheduler "
                "froze the paired source snapshot"
            )
        if bool(manifest.get("packer_boundary_parity")) != bool(
            args.packer_boundary_parity
        ):
            raise SystemExit(
                "ERROR: packer-boundary parity manifest differs from the "
                "scheduler capability"
            )
    qualification = manifest.get("qualification")
    if isinstance(qualification, Mapping):
        authority_sha256 = str(manifest["execution_authority"]["sha256"])
        if (
            args.ssh
            and args.qualification_runtime_attestation_sha256
            != authority_sha256
        ):
            raise SystemExit(
                "ERROR: scheduler runtime attestation does not match the "
                "qualification execution authority"
            )
        placement = next(
            row[str(manifest["role"])]
            for row in qualification["gpu_schedule"]
            if int(row["seed"]) == int(manifest["seed"])
        )
        if args.ssh and (
            int(placement["index"]) != int(args.gpu)
            or str(placement["uuid"])
            != args.qualification_expected_gpu_uuid
        ):
            raise SystemExit(
                "ERROR: scheduler GPU placement differs from frozen qualification "
                "index/UUID"
            )
        if args.ssh and not re.fullmatch(
            r"qwave_[0-9a-f]{16}", args.qualification_wave_id
        ):
            raise SystemExit(
                "ERROR: qualification runner lacks the fsynced wave-attempt ID"
            )
    elif (
        args.qualification_runtime_attestation_sha256
        or args.qualification_expected_gpu_uuid
        or args.qualification_wave_id
    ):
        raise SystemExit(
            "ERROR: qualification-only scheduler arguments supplied to an "
            "ordinary experiment"
        )
    if args.print_env:
        print(_shell_env(_bound_env(manifest, root=root)))
        return 0
    print("=== BOUND MANIFEST ===")
    print(json.dumps(manifest, indent=1))

    if args.dry_run or not args.ssh:
        cmd, log = launch_command(manifest, args.remote_wd or "<remote-wd>", args.gpu, args.python_bin, root=root)
        print("\n=== EXACT LAUNCH (no free-form env) ===\n" + cmd)
        print("\n(dry-run: not executed; no RunRecord written)")
        return 0

    cmd, log = launch_command(manifest, args.remote_wd, args.gpu, args.python_bin, root=root)
    # Hardened argv: ControlMaster mux (immune to sshd MaxStartups) + connect timeouts.
    ssh = ssh_argv(args.ssh)
    print(f"\nlaunching arm {args.arm_id} seed {args.seed} on gpu {args.gpu} ...")
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    session = f"bound_{manifest['config_hash']}"
    remote_launch = f"tmux new-session -d -s {session} {shlex.quote(cmd)}"
    # Launch: bounded + retried. A MaxStartups drop is transient, not fatal; an
    # UNBOUNDED run here was the production hang (train.py never started, GPU 0 MiB).
    last_err = ""
    for backoff in (0, 5, 15, 45):
        if backoff:
            time.sleep(backoff)
        try:
            subprocess.run(
                ssh + [remote_launch], check=True, timeout=90,
                capture_output=True, text=True,
            )
            last_err = ""
            break
        except subprocess.TimeoutExpired:
            last_err = "ssh launch timed out (remote unresponsive)"
        except subprocess.CalledProcessError as exc:
            last_err = (exc.stderr or "").strip() or f"tmux launch exit {exc.returncode}"
    if last_err:
        raise SystemExit(f"failed to launch remote training after retries: {last_err}")
    global _SESSION
    _SESSION = session  # the death-watchdog kills this tmux session if the scheduler dies
    selection_stack.close()
    # Poll for completion — bounded per call, abort on repeated transport failures
    # (returncode 1 = "not finished yet"; >=255 or timeout = ssh transport error).
    remote_log = shlex.quote(f"{args.remote_wd}/{log}")
    consecutive_fail = 0
    done = False
    for _ in range(240):
        try:
            completed = subprocess.run(
                ssh + [f"grep -q '^EXIT_CODE=' {remote_log}"],
                capture_output=True, timeout=60,
            )
            if completed.returncode == 0:
                done = True
                break
            consecutive_fail = 0 if completed.returncode == 1 else consecutive_fail + 1
        except subprocess.TimeoutExpired:
            consecutive_fail += 1
        if consecutive_fail >= 20:
            raise TimeoutError(
                f"remote unreachable for ~20 consecutive polls; aborting ({remote_log})"
            )
        time.sleep(30 + random.uniform(0, 5))  # jitter so co-runners don't align
    if not done:
        raise TimeoutError(
            f"remote run did not finish within two hours; no RunRecord was written ({remote_log})"
        )
    # Fetch the completed log — bounded + retried. An empty result must NOT be
    # misread as "refused before training" (a transient ssh blip at collection time).
    text = ""
    for backoff in (0, 5, 15):
        if backoff:
            time.sleep(backoff)
        try:
            res = subprocess.run(
                ssh + [f"cat {remote_log}"], capture_output=True, text=True, timeout=120,
            )
            if res.returncode == 0 and res.stdout:
                text = res.stdout
                break
        except subprocess.TimeoutExpired:
            pass
    if not text:
        raise TimeoutError(
            f"could not fetch the completed remote log after retries ({remote_log})"
        )
    if "TRAIN_PROCESS_STARTED=" not in text:
        print(
            "run refused before training started; no RunRecord was appended "
            f"({args.remote_wd}/{log})",
            file=sys.stderr,
        )
        return 2
    strict_qualification = isinstance(manifest.get("qualification"), Mapping)
    strict_log = strict_qualification or bool(
        manifest.get("packer_boundary_parity")
    )
    try:
        resolved_training = parse_labeled_config(
            text,
            "RESOLVED_CONFIG",
            require_singleton=strict_log,
        )
        resolved_runner = parse_labeled_config(
            text,
            "RUNNER_CONFIG",
            require_singleton=strict_log,
        )
        overlapping = set(resolved_training) & set(resolved_runner)
        if strict_log and any(
            resolved_training[key] != resolved_runner[key] for key in overlapping
        ):
            raise ValueError("RESOLVED_CONFIG and RUNNER_CONFIG conflict")
        resolved = {**resolved_training, **resolved_runner}
        metrics = parse_outcome_metrics(
            text, require_singleton=strict_log
        )
    except ValueError as exc:
        print(f"STRICT LOG PARSE FAILED: {exc}", file=sys.stderr)
        resolved_training = {}
        resolved = {}
        metrics = {}
    verified, mism = verify_config(manifest, resolved, root=root)
    exit_matches = re.findall(r"^EXIT_CODE=([0-9]+)$", text, re.MULTILINE)
    exit_ok = (
        exit_matches == ["0"]
        if strict_log
        else "0" in exit_matches
    )
    code_matches = re.findall(
        r"^CODE_HASHES_VERIFIED=([01])$", text, re.MULTILINE
    )
    code_verified = (
        code_matches == ["1"]
        if strict_log
        else "1" in code_matches
    )
    val_bpb = metrics.get("val_bpb")
    num_steps = metrics.get("num_steps")
    total_tokens = metrics.get("total_tokens")
    charged_training_seconds = metrics.get("charged_training_seconds")
    gpu_uuid_match = re.search(r"^GPU_UUID_BOUND=(GPU-[A-Za-z0-9-]+)$", text, re.MULTILINE)
    gpu_uuid = gpu_uuid_match.group(1) if gpu_uuid_match else ""
    expected_gpu_uuid = (
        args.qualification_expected_gpu_uuid if strict_qualification else ""
    )
    execution_isolated = (
        f"GPU_ADVISORY_LOCK_ACQUIRED={args.gpu}" in text
        and bool(gpu_uuid)
        and "GPU_TRAIN_PID_OBSERVED=" in text
        and "GPU_TRAIN_PID_NEVER_OBSERVED" not in text
        and "GPU_COTENANCY_SAMPLING_VERIFIED=1" in text
        and "GPU_COTENANCY_DETECTED" not in text
        and "GPU_COTENANCY_REJECTED" not in text
        and "GPU_INVENTORY_FAILED" not in text
        and "GPU_UUID_RESOLUTION_FAILED" not in text
        and "GPU_UUID_INVALID" not in text
        and (not expected_gpu_uuid or gpu_uuid == expected_gpu_uuid)
    )
    verified = verified and code_verified
    qualification_verified, qualification_facts, qualification_errors = (
        verify_qualification_integrity(
            manifest,
            text,
            resolved_training,
            metrics,
            args.qualification_runtime_attestation_sha256,
        )
    )
    (
        diagnostic_verified,
        diagnostic_facts,
        diagnostic_errors,
    ) = verify_packer_boundary_diagnostic(manifest, text)
    if strict_qualification:
        qualification_facts["qualification_wave_id"] = args.qualification_wave_id
    status = (
        "complete"
        if (
            verified
            and execution_isolated
            and exit_ok
            and val_bpb is not None
            and num_steps is not None
            and total_tokens is not None
            and charged_training_seconds is not None
            and qualification_verified
            and diagnostic_verified
        )
        else "invalid"
    )
    if mism:
        print("CONFIG MISMATCH:", mism)
    if qualification_errors:
        print("QUALIFICATION INTEGRITY MISMATCH:", qualification_errors)
    if diagnostic_errors:
        print("PACKER DIAGNOSTIC INTEGRITY MISMATCH:", diagnostic_errors)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    ended_at = datetime.now(timezone.utc)
    rec = make_run_record(
        manifest,
        f"{args.remote_wd}/{log}",
        val_bpb,
        verified,
        status,
        git_commit=git_commit,
        git_dirty=git_dirty,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        gpu_hours=(time.monotonic() - started_monotonic) / 3600,
        gpu_id=args.gpu,
        gpu_uuid=gpu_uuid,
        num_steps=num_steps,
        total_tokens=total_tokens,
        charged_training_seconds=charged_training_seconds,
        execution_isolated=execution_isolated,
        resolved_training_config=resolved_training,
        qualification_integrity=qualification_facts,
        diagnostic_integrity=(
            diagnostic_facts
            if manifest.get("packer_boundary_parity")
            else None
        ),
    )
    append_run(root, rec)
    print(f"RunRecord {rec.run_id} appended: status={status} config_verified={verified} val_bpb={val_bpb}")
    return 0 if status == "complete" else 2


def main() -> int:
    with ExitStack() as selection_stack:
        return _main(selection_stack)


if __name__ == "__main__":
    sys.exit(main())
