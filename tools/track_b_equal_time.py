#!/usr/bin/env python3
"""Prepare a fair equal-*steady-training-time* Track B diagnostic.

This tool deliberately does not call the result "compute matched" or
"equal-wall-clock":

* ``MAX_STEPS`` matches optimizer steps and training tokens, not FLOPs.
* Track B step timing covers the synchronized forward/backward/optimizer core.
  It excludes setup, compilation warmup, logging, probes, and final evaluation.
* A true end-to-end wall-clock benchmark must use ``STOP_MODE=walltime`` and
  report the exact included lifecycle. That is a different estimand.

The old helper converted median step times into a different ``MAX_STEPS`` for
each configuration. That approximation was avoidable and could drift when step
latency changed during a run. The fair steady-training-time diagnostic runs all
configs directly with the same ``STOP_MODE=time`` and ``TIME_BUDGET``. The
training loop then schedules warmdown against elapsed training time and stops on
the same measured budget for every arm.

Historical schema-v1 Track B files omit critical configuration and GPU identity
fields. They can be inspected with ``--allow-legacy`` but cannot establish a
provenance-complete timing verdict. New schema-v2 files are checked against the
configuration encoded in their filename.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections.abc import Mapping, Sequence

# Exact env sets used by the named configurations. Common diagnostic controls
# (TRACK_B_MODE/probe cadence/stop mode) are added by build_time_command().
CONFIG_ENVS: dict[str, str] = {
    "baseline": "",
    "sota1": (
        "NGRAM_TABLE_MULT=256 DOC_MASK=1 DOC_MASK_MODE=seg WARMDOWN_RATIO=0.75 "
        "NGRAM_SPARSE_GRAD=1 NGRAM_STATE_ROWWISE=1 "
        "COMPILE_MODE=max-autotune-no-cudagraphs"
    ),
    "sota2": (
        "NGRAM_TABLE_MULT=256 DOC_MASK=1 DOC_MASK_MODE=seg WARMDOWN_RATIO=0.75 "
        "NGRAM_SPARSE_GRAD=1 NGRAM_STATE_ROWWISE=1 "
        "COMPILE_MODE=max-autotune-no-cudagraphs NGRAM_FOURGRAM_MULT=256"
    ),
}

# Defaults matter for the baseline because CONFIG_ENVS intentionally stays empty.
EXPECTED_CONFIG: dict[str, dict[str, object]] = {
    "baseline": {
        "NGRAM_TABLE_MULT": 64,
        "NGRAM_FOURGRAM_MULT": 0,
        "DOC_MASK": False,
        "WARMDOWN_RATIO": 0.95,
        "NGRAM_SPARSE_GRAD": False,
        "NGRAM_STATE_ROWWISE": False,
        "COMPILE_MODE": "max-autotune",
    },
    "sota1": {
        "NGRAM_TABLE_MULT": 256,
        "NGRAM_FOURGRAM_MULT": 0,
        "DOC_MASK": True,
        "DOC_MASK_MODE": "seg",
        "WARMDOWN_RATIO": 0.75,
        "NGRAM_SPARSE_GRAD": True,
        "NGRAM_STATE_ROWWISE": True,
        "COMPILE_MODE": "max-autotune-no-cudagraphs",
    },
    "sota2": {
        "NGRAM_TABLE_MULT": 256,
        "NGRAM_FOURGRAM_MULT": 256,
        "DOC_MASK": True,
        "DOC_MASK_MODE": "seg",
        "WARMDOWN_RATIO": 0.75,
        "NGRAM_SPARSE_GRAD": True,
        "NGRAM_STATE_ROWWISE": True,
        "COMPILE_MODE": "max-autotune-no-cudagraphs",
    },
}

RESULT_FILE_RE = re.compile(r"^([A-Za-z0-9_]+)_s(\d+)\.json$")
MIN_VERDICT_SEEDS = 3


def _equivalent(actual: object, expected: object) -> bool:
    """Compare JSON config values without confusing bool with int."""
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual is expected
        return str(actual).lower() in ({"1", "true"} if expected else {"0", "false"})
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_result(
    config_name: str,
    data: Mapping[str, object],
    *,
    allow_legacy: bool = False,
) -> list[str]:
    """Validate timing scope and embedded config; return non-fatal warnings."""
    warnings: list[str] = []
    schema_version = int(data.get("schema_version", 1))
    if schema_version < 2:
        if not allow_legacy:
            raise ValueError(
                f"{config_name}: schema-v1 result lacks complete config/GPU/timing "
                "provenance; pass --allow-legacy for exploratory inspection only"
            )
        warnings.append(
            f"{config_name}: LEGACY result; missing fields are not evidence of equality"
        )

    timing = data.get("timing_protocol", {})
    if schema_version >= 2:
        if not isinstance(timing, Mapping):
            raise ValueError(f"{config_name}: timing_protocol must be an object")
        if timing.get("scope") != "synchronized_steady_state_training_step":
            raise ValueError(
                f"{config_name}: unsupported timing scope {timing.get('scope')!r}"
            )
        if timing.get("is_end_to_end_wall_clock") is not False:
            raise ValueError(
                f"{config_name}: Track B result must explicitly say it is not "
                "end-to-end wall clock"
            )

    config = data.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{config_name}: missing structured config")
    for key, expected in EXPECTED_CONFIG[config_name].items():
        if key not in config:
            if allow_legacy and schema_version < 2:
                warnings.append(f"{config_name}: legacy config omits {key}")
                continue
            raise ValueError(f"{config_name}: config omits fairness-critical key {key}")
        if not _equivalent(config[key], expected):
            raise ValueError(
                f"{config_name}: mislabeled result, expected {key}={expected!r}, "
                f"found {config[key]!r}"
            )

    if schema_version >= 2:
        runtime = data.get("runtime")
        if not isinstance(runtime, Mapping) or not runtime.get("device_name"):
            raise ValueError(f"{config_name}: schema-v2 result lacks GPU identity")
        if not runtime.get("gpu_uuid") or not runtime.get("paired_run_id"):
            raise ValueError(
                f"{config_name}: schema-v2 result lacks physical GPU/pair identity"
            )
        code_hashes = data.get("code_hashes")
        if not isinstance(code_hashes, Mapping):
            raise ValueError(f"{config_name}: schema-v2 result lacks code hashes")
        for name in ("train.py", "lib.py", "prepare.py", "data_split.json"):
            digest = code_hashes.get(name)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(
                    f"{config_name}: missing/invalid SHA-256 for {name}"
                )
    return warnings


def validate_paired_provenance(
    by_config: Mapping[str, Mapping[int, Mapping[str, object]]],
    config_names: Sequence[str],
    seeds: Sequence[int],
) -> None:
    """Require exact code and physical-GPU identity within each seed block."""
    for seed in seeds:
        results = [by_config[name][seed] for name in config_names]
        code_hashes = [result.get("code_hashes") for result in results]
        if any(hashes != code_hashes[0] for hashes in code_hashes[1:]):
            raise ValueError(f"seed {seed}: code/data hashes differ across arms")

        runtimes = [result.get("runtime") for result in results]
        if not all(isinstance(runtime, Mapping) for runtime in runtimes):
            raise ValueError(f"seed {seed}: missing runtime provenance")
        gpu_uuids = [
            runtime.get("gpu_uuid")  # type: ignore[union-attr]
            for runtime in runtimes
        ]
        if any(gpu_uuid != gpu_uuids[0] for gpu_uuid in gpu_uuids[1:]):
            raise ValueError(f"seed {seed}: physical GPU differs across paired arms")
        pair_ids = [
            runtime.get("paired_run_id")  # type: ignore[union-attr]
            for runtime in runtimes
        ]
        if any(pair_id != pair_ids[0] for pair_id in pair_ids[1:]):
            raise ValueError(f"seed {seed}: paired_run_id differs across arms")


def load_results(
    results_dir: str,
    *,
    allow_legacy: bool = False,
) -> tuple[dict[str, dict[int, Mapping[str, object]]], list[str]]:
    by_config: dict[str, dict[int, Mapping[str, object]]] = {}
    warnings: list[str] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_s*.json"))):
        match = RESULT_FILE_RE.match(os.path.basename(path))
        if not match:
            continue
        config_name, seed_text = match.groups()
        if config_name not in CONFIG_ENVS:
            warnings.append(f"ignoring unknown config file {path}")
            continue
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        seed = int(seed_text)
        if int(data.get("seed", -1)) != seed:
            raise ValueError(
                f"{path}: filename seed {seed} != embedded seed {data.get('seed')!r}"
            )
        warnings.extend(
            f"{path}: {warning}"
            for warning in validate_result(
                config_name, data, allow_legacy=allow_legacy
            )
        )
        if seed in by_config.setdefault(config_name, {}):
            raise ValueError(f"duplicate result for {config_name} seed {seed}")
        by_config[config_name][seed] = data
    return by_config, warnings


def run_median_step_ms(data: Mapping[str, object]) -> float:
    samples = [float(value) for value in data.get("step_times_ms", [])]
    if not samples:
        raise ValueError("result has no step_times_ms")
    return statistics.median(samples)


def median_of_run_medians(results: Sequence[Mapping[str, object]]) -> float:
    """Treat the run/seed—not thousands of autocorrelated steps—as the unit."""
    if not results:
        raise ValueError("no results")
    return statistics.median(run_median_step_ms(result) for result in results)


def reference_steady_seconds(results: Sequence[Mapping[str, object]]) -> float:
    """Median measured steady-training duration across baseline runs."""
    durations: list[float] = []
    for result in results:
        value = result.get("cumulative_steady_time_s")
        if value is None:
            value = sum(float(item) for item in result.get("step_times_ms", [])) / 1000.0
        durations.append(float(value))
    if not durations or any(value <= 0 for value in durations):
        raise ValueError("baseline results lack positive steady-training durations")
    return statistics.median(durations)


def build_time_command(config_name: str, seed: int, budget_s: int) -> str:
    """Return an explicit off-scope diagnostic command with probes disabled."""
    if config_name not in CONFIG_ENVS:
        raise KeyError(config_name)
    if budget_s <= 0:
        raise ValueError("budget_s must be positive")
    config_env = CONFIG_ENVS[config_name]
    if config_env:
        config_env += " "
    return (
        f"STOP_MODE=time TIME_BUDGET={budget_s} TRACK_B_MODE=1 "
        "VAL_LOSS_EVERY=0 VAL_BPB_PROBE_EVERY=0 OBSERVE_LAYER_PROBES=0 "
        f"TRACK_B_PAIR_ID=seed{seed} SEED={seed} {config_env}python train.py"
    )


def counterbalanced_order(config_names: Sequence[str], block_index: int) -> list[str]:
    """Rotate arm order across seed blocks to reduce thermal/order confounding."""
    if not config_names:
        return []
    offset = block_index % len(config_names)
    return [*config_names[offset:], *config_names[:offset]]


def _parse_seed_list(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique CSV list")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="track_b_results")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument(
        "--time-budget",
        type=int,
        default=0,
        help="shared steady-training budget in seconds; default is the median "
        "measured baseline steady duration",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seed_list,
        default=None,
        help="diagnostic seeds as CSV; default is the intersection in result files",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="inspect schema-v1 files despite incomplete config/GPU provenance",
    )
    args = parser.parse_args()

    try:
        by_config, warnings = load_results(
            args.results_dir, allow_legacy=args.allow_legacy
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if args.baseline not in by_config:
        print(
            f"error: no validated {args.baseline}_s*.json files in {args.results_dir}",
            file=sys.stderr,
        )
        return 2

    available = [name for name in CONFIG_ENVS if name in by_config]
    common_seeds = sorted(
        set.intersection(*(set(by_config[name]) for name in available))
    )
    seeds = args.seeds or common_seeds
    missing = {
        name: sorted(set(seeds) - set(by_config[name]))
        for name in available
        if set(seeds) - set(by_config[name])
    }
    if missing:
        print(f"error: requested seeds are not balanced across configs: {missing}", file=sys.stderr)
        return 2
    if len(seeds) < MIN_VERDICT_SEEDS:
        print(
            f"error: only {len(seeds)} balanced seeds; Track B effect verdicts "
            f"require at least {MIN_VERDICT_SEEDS}",
            file=sys.stderr,
        )
        return 2
    if not args.allow_legacy:
        try:
            validate_paired_provenance(by_config, available, seeds)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    # The budget is only a common duration, so use every same-code baseline run
    # for a robust estimate of a full baseline training duration. Hardware is
    # still paired within each comparison block below; the identical budget is
    # then applied to every arm.
    selected_baseline = by_config[args.baseline][seeds[0]]
    selected_hashes = selected_baseline.get("code_hashes")
    baseline_runs = [
        result
        for result in by_config[args.baseline].values()
        if result.get("code_hashes") == selected_hashes
    ]
    budget_s = args.time_budget or round(reference_steady_seconds(baseline_runs))
    print(
        f"Shared steady-training budget: {budget_s}s "
        f"(baseline median across {len(baseline_runs)} runs)"
    )
    print("Timing unit: synchronized train-step core; NOT end-to-end wall clock.")
    print()
    print(f"{'config':<12} {'runs':>5} {'median_of_run_medians_ms':>26}")
    for name in available:
        balanced_runs = [by_config[name][seed] for seed in seeds]
        print(
            f"{name:<12} {len(balanced_runs):>5} "
            f"{median_of_run_medians(balanced_runs):>26.1f}"
        )

    print()
    print(
        "OFF-SCOPE DIAGNOSTIC commands. Register/pin code and pair hardware "
        "before treating their outputs as evidence:"
    )
    for block_index, seed in enumerate(seeds):
        for name in counterbalanced_order(available, block_index):
            print(f"  {build_time_command(name, seed, budget_s)}  # {name}")
    print()
    print(
        "Fairness requirements: run each seed's arms on the same physical GPU "
        "in the counterbalanced order printed above (or use a randomized crossover), "
        "use identical code/data/eval hashes, disable all probes, and analyze paired "
        "deltas. A single-seed result is exploratory."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
