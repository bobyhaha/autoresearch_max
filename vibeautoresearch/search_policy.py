"""Hardware-aware experiment portfolio and staged stopping policy.

The research ledger answers "what ran?"  This module answers the separate
question "should this idea receive another GPU?"  New experiments must move
through the 1 -> 3 -> 6 -> 10 paired-seed funnel, declare their expected
throughput and matched-step quality effects, and clear a preregistered
checkpoint before more seeds can be authorized.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from .core import (
    SchemaError,
    json_mapping,
    local_id,
    require_bool,
    require_id,
    require_int,
    require_keys,
    require_number,
    require_text,
)
from .idea_archive import RESEARCH_DIRECTIONS


SEARCH_POLICY_VERSION = 2
FUNNEL_STAGES = (1, 3, 6, 10)
QUALIFICATION_STAGE_PAIRS = 4
# A two-sided 90% t interval and a one-sided 95% t upper bound use the
# same critical value. Qualification is deliberately restricted to four
# paired seeds, hence df=3 is fixed and does not require a scipy dependency.
T_95_DF3 = 2.3533634348018264
DIRECTION_ROUND_LIMIT = 5
TERMINAL_VERDICTS = {
    "adoption_candidate",
    "invalid",
    "qualification_fail",
    "qualification_pass",
    "stop_futility",
    "stop_no_signal",
}
FAILURE_UPDATE_RESULTS = {"mixed", "null", "oppose"}
SUCCESS_UPDATE_RESULTS = {"support", "weak_support"}

QUALIFICATION_ALLOWED_RESOLVED_DIFFERENCES = frozenset(
    {
        "WINDOW_PATTERN",
        "MATRIX_LR",
        "DEVICE_BATCH_SIZE",
        "TOTAL_BATCH_SIZE",
        "WARMDOWN_RATIO",
        "MUON_MOMENTUM_CONTINUOUS",
    }
)
QUALIFICATION_RESOLVED_KEYS = frozenset(
    {
        "STOP_MODE",
        "TIME_BUDGET",
        "MAX_STEPS",
        "ATTN_BACKEND",
        "WINDOW_PATTERN",
        "DEVICE_BATCH_SIZE",
        "TOTAL_BATCH_SIZE",
        "MATRIX_LR",
        "NGRAM_VE_LR_SCALE",
        "NGRAM_VE_BETA",
        "FINAL_LR_FRAC",
        "WEIGHT_DECAY",
        "EMBEDDING_LR",
        "UNEMBEDDING_LR",
        "SCALAR_LR",
        "WARMDOWN_RATIO",
        "ADAM_WARMDOWN_RATIO",
        "DEMON_FINAL_BETA1",
        "MLP_TYPE",
        "WARMUP_RATIO",
        "CAUTIOUS_UPDATE",
        "CAUTIOUS_MUON",
        "TIE_EMBED",
        "RHO1_GAMMA",
        "LR_SCALE",
        "RESID_SCALE",
        "DOC_MASK",
        "DOC_MASK_MODE",
        "DOC_MASK_IMPL",
        "RESET_ROPE",
        "NGRAM_TABLE_MULT",
        "NGRAM_BIGRAM_MULT",
        "NGRAM_TRIGRAM_MULT",
        "NGRAM_FOURGRAM_MULT",
        "NGRAM_FIVEGRAM_MULT",
        "NGRAM_FOURGRAM_SPAN",
        "SHARED_TRIGRAM_VE",
        "SHUFFLE_DATA",
        "VAL_LOSS_EVERY",
        "OBSERVE_LAYER_PROBES",
        "LM_HEAD_INIT_STD",
        "SOFTCAP_CAP",
        "SOFTCAP_TAU",
        "NGRAM_STATE_ROWWISE",
        "COMPILE_MODE",
        "MUON_MOMENTUM_CONTINUOUS",
        "MUON_NS_STEPS",
        "ADAM_BETA2",
        "NGRAM_VE_EPS",
        "BF16_CE",
        "CANON",
        "CANON_K",
        "CANON_LR",
        "NGRAM_SPARSE_GRAD",
        "NGRAM_PK_MEMORY",
        "PK_MEM_SUBKEYS",
        "PK_MEM_TOPK",
        "PK_MEM_QDIM",
        "PK_MEM_LR_SCALE",
        "TRACK_B_MODE",
        "VAL_BPB_PROBE_EVERY",
        "EVAL_TOKENS_PROBE",
        "ATTNRES_ENABLE",
        "NGRAM_WD_LAMBDA",
        "NGRAM_BACKOFF_KAPPA",
        "TOKEN_SHIFT",
        "NGPT_SPHERE",
        "NGPT_ALPHA_INIT",
        "DOC_MASK_BLOCK",
    }
)


def _sha256(value: Any, field: str) -> str:
    digest = require_text(value, field)
    if len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise SchemaError(f"{field} must be 64 lowercase hex characters")
    return digest


def _string_mapping(value: Any, field: str) -> dict[str, str]:
    raw = json_mapping(value, field)
    normalized: dict[str, str] = {}
    for key, item in raw.items():
        name = require_text(key, f"{field}.key")
        normalized[name] = str(item)
    return normalized


def _prediction(value: Any, field: str) -> dict[str, Any]:
    prediction = json_mapping(value, field)
    require_keys(prediction, field, ("expected_delta", "unit", "measurement", "rationale"))
    require_number(prediction["expected_delta"], f"{field}.expected_delta")
    require_text(prediction["unit"], f"{field}.unit")
    require_text(prediction["measurement"], f"{field}.measurement")
    require_text(prediction["rationale"], f"{field}.rationale")
    return prediction


def _qualification(value: Any, field: str) -> dict[str, Any]:
    """Validate a four-pair historical-equivalence qualification contract."""
    qualification = json_mapping(value, field)
    if not qualification:
        return {}
    require_keys(
        qualification,
        field,
        (
            "kind",
            "equivalence_margin",
            "superiority_minimum_effect",
            "required_helping_pairs",
            "token_ratio_lower",
            "token_ratio_upper",
            "historical_anchors",
        ),
    )
    if str(qualification["kind"]) != "historical_equivalence_bridge":
        raise SchemaError(
            f"{field}.kind must be 'historical_equivalence_bridge'"
        )
    # Version 0 denotes a preserved pre-audit proposal only. The registry
    # mutation gate rejects it; allowing it to deserialize keeps the append-only
    # proposal ledger honest without making that draft executable.
    integrity_version = require_int(
        qualification.get("integrity_version", 0),
        f"{field}.integrity_version",
        minimum=0,
    )
    if integrity_version not in {0, 2}:
        raise SchemaError(f"{field}.integrity_version must be 2 when present")
    equivalence_margin = require_number(
        qualification["equivalence_margin"],
        f"{field}.equivalence_margin",
        minimum=0,
    )
    if equivalence_margin <= 0:
        raise SchemaError(f"{field}.equivalence_margin must be positive")
    superiority_minimum_effect = require_number(
        qualification["superiority_minimum_effect"],
        f"{field}.superiority_minimum_effect",
        minimum=0,
    )
    if superiority_minimum_effect <= 0:
        raise SchemaError(
            f"{field}.superiority_minimum_effect must be positive"
        )
    helping = require_int(
        qualification["required_helping_pairs"],
        f"{field}.required_helping_pairs",
        minimum=1,
    )
    if helping != QUALIFICATION_STAGE_PAIRS:
        raise SchemaError(
            f"{field}.required_helping_pairs must be "
            f"{QUALIFICATION_STAGE_PAIRS}"
        )
    token_lower = require_number(
        qualification["token_ratio_lower"],
        f"{field}.token_ratio_lower",
        minimum=0,
    )
    token_upper = require_number(
        qualification["token_ratio_upper"],
        f"{field}.token_ratio_upper",
        minimum=0,
    )
    if not 0 < token_lower < 1 < token_upper:
        raise SchemaError(
            f"{field} token ratio bounds must satisfy 0 < lower < 1 < upper"
        )

    raw_anchors = qualification["historical_anchors"]
    if not isinstance(raw_anchors, (list, tuple)):
        raise SchemaError(f"{field}.historical_anchors must be an array")
    if len(raw_anchors) != QUALIFICATION_STAGE_PAIRS:
        raise SchemaError(
            f"{field}.historical_anchors must contain exactly "
            f"{QUALIFICATION_STAGE_PAIRS} anchors"
        )
    anchors: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for index, raw_anchor in enumerate(raw_anchors):
        anchor_field = f"{field}.historical_anchors[{index}]"
        anchor = json_mapping(raw_anchor, anchor_field)
        required_anchor_keys = [
            "seed",
            "val_bpb",
            "total_tokens",
            "artifact_path",
            "artifact_sha256",
        ]
        if integrity_version == 2:
            required_anchor_keys.extend(
                (
                    "num_steps",
                    "total_batch_size",
                    "launch_provenance_path",
                    "launch_provenance_sha256",
                )
            )
        require_keys(anchor, anchor_field, tuple(required_anchor_keys))
        seed = require_int(anchor["seed"], f"{anchor_field}.seed", minimum=0)
        if seed in seen_seeds:
            raise SchemaError(f"{field}.historical_anchors has duplicate seed {seed}")
        seen_seeds.add(seed)
        val_bpb = require_number(
            anchor["val_bpb"], f"{anchor_field}.val_bpb", minimum=0
        )
        num_steps = (
            require_int(
                anchor["num_steps"],
                f"{anchor_field}.num_steps",
                minimum=1,
            )
            if integrity_version == 2
            else 0
        )
        total_batch_size = (
            require_int(
                anchor["total_batch_size"],
                f"{anchor_field}.total_batch_size",
                minimum=1,
            )
            if integrity_version == 2
            else 0
        )
        total_tokens = require_int(
            anchor["total_tokens"],
            f"{anchor_field}.total_tokens",
            minimum=1,
        )
        if (
            integrity_version == 2
            and num_steps * total_batch_size != total_tokens
        ):
            raise SchemaError(
                f"{anchor_field}.total_tokens must equal "
                "num_steps * total_batch_size"
            )
        artifact_path = require_text(
            anchor["artifact_path"], f"{anchor_field}.artifact_path"
        )
        artifact_sha256 = _sha256(
            anchor["artifact_sha256"], f"{anchor_field}.artifact_sha256"
        )
        launch_provenance = (
            {
                "launch_provenance_path": require_text(
                    anchor["launch_provenance_path"],
                    f"{anchor_field}.launch_provenance_path",
                ),
                "launch_provenance_sha256": _sha256(
                    anchor["launch_provenance_sha256"],
                    f"{anchor_field}.launch_provenance_sha256",
                ),
            }
            if integrity_version == 2
            else {}
        )
        anchors.append(
            {
                "seed": seed,
                "val_bpb": float(val_bpb),
                **(
                    {
                        "num_steps": num_steps,
                        "total_batch_size": total_batch_size,
                    }
                    if integrity_version == 2
                    else {}
                ),
                **launch_provenance,
                "total_tokens": total_tokens,
                "artifact_path": artifact_path,
                "artifact_sha256": artifact_sha256,
            }
        )

    # Legacy unlaunched proposal definitions can remain append-only and
    # inspectable, but the registry mutation gate refuses them. Only v2 binds
    # the execution-integrity contract required for a fresh qualification.
    if integrity_version == 0:
        return {
            **qualification,
            "integrity_version": 0,
            "equivalence_margin": float(equivalence_margin),
            "superiority_minimum_effect": float(superiority_minimum_effect),
            "required_helping_pairs": helping,
            "token_ratio_lower": float(token_lower),
            "token_ratio_upper": float(token_upper),
            "historical_anchors": anchors,
        }

    require_keys(
        qualification,
        field,
        (
            "allowed_resolved_differences",
            "expected_arm_resolved",
            "fresh_integrity",
            "execution_authority",
            "gpu_schedule",
            "inventory_gate",
        ),
    )
    raw_allowed = qualification["allowed_resolved_differences"]
    if not isinstance(raw_allowed, (list, tuple)) or not raw_allowed:
        raise SchemaError(
            f"{field}.allowed_resolved_differences must be a non-empty array"
        )
    allowed = tuple(
        require_text(item, f"{field}.allowed_resolved_differences")
        for item in raw_allowed
    )
    if len(set(allowed)) != len(allowed):
        raise SchemaError(
            f"{field}.allowed_resolved_differences must not contain duplicates"
        )
    if set(allowed) != QUALIFICATION_ALLOWED_RESOLVED_DIFFERENCES:
        raise SchemaError(
            f"{field}.allowed_resolved_differences must be exactly the canonical "
            f"six coordinates {sorted(QUALIFICATION_ALLOWED_RESOLVED_DIFFERENCES)}"
        )

    expected = json_mapping(
        qualification["expected_arm_resolved"],
        f"{field}.expected_arm_resolved",
    )
    require_keys(
        expected,
        f"{field}.expected_arm_resolved",
        ("control", "treatment"),
    )
    control_resolved = _string_mapping(
        expected["control"], f"{field}.expected_arm_resolved.control"
    )
    treatment_resolved = _string_mapping(
        expected["treatment"], f"{field}.expected_arm_resolved.treatment"
    )
    if not control_resolved or set(control_resolved) != set(treatment_resolved):
        raise SchemaError(
            f"{field}.expected_arm_resolved arms must bind the same non-empty key set"
        )
    if set(control_resolved) != QUALIFICATION_RESOLVED_KEYS:
        raise SchemaError(
            f"{field}.expected_arm_resolved must bind the complete current "
            f"RESOLVED_CONFIG key set: missing="
            f"{sorted(QUALIFICATION_RESOLVED_KEYS - set(control_resolved))}, "
            f"extra={sorted(set(control_resolved) - QUALIFICATION_RESOLVED_KEYS)}"
        )
    actual_differences = {
        key
        for key in control_resolved
        if control_resolved[key] != treatment_resolved[key]
    }
    if actual_differences != set(allowed):
        raise SchemaError(
            f"{field}.allowed_resolved_differences must exactly equal the "
            f"declared arm differences: {sorted(actual_differences)}"
        )

    fresh = json_mapping(
        qualification["fresh_integrity"], f"{field}.fresh_integrity"
    )
    require_keys(
        fresh,
        f"{field}.fresh_integrity",
        (
            "charged_seconds_min",
            "charged_seconds_max_exclusive",
            "require_compute_complete",
            "require_token_arithmetic",
            "require_boundary_attested",
            "require_boundary_counts_verified",
            "require_boundary_digest",
            "expected_grad_accum_steps",
        ),
    )
    charged_min = require_number(
        fresh["charged_seconds_min"],
        f"{field}.fresh_integrity.charged_seconds_min",
        minimum=0,
    )
    charged_max = require_number(
        fresh["charged_seconds_max_exclusive"],
        f"{field}.fresh_integrity.charged_seconds_max_exclusive",
        minimum=0,
    )
    if charged_max <= charged_min:
        raise SchemaError(
            f"{field}.fresh_integrity charged-time interval must be non-empty"
        )
    if float(charged_min) != 300.0 or float(charged_max) != 301.0:
        raise SchemaError(
            f"{field}.fresh_integrity charged-time interval must be [300.0,301.0)"
        )
    for key in (
        "require_compute_complete",
        "require_token_arithmetic",
        "require_boundary_attested",
        "require_boundary_counts_verified",
        "require_boundary_digest",
    ):
        if require_bool(
            fresh[key], f"{field}.fresh_integrity.{key}"
        ) is not True:
            raise SchemaError(
                f"{field}.fresh_integrity.{key} must be true"
            )
    expected_grad = json_mapping(
        fresh["expected_grad_accum_steps"],
        f"{field}.fresh_integrity.expected_grad_accum_steps",
    )
    require_keys(
        expected_grad,
        f"{field}.fresh_integrity.expected_grad_accum_steps",
        ("control", "treatment"),
    )
    normalized_grad = {
        role: require_int(
            expected_grad[role],
            f"{field}.fresh_integrity.expected_grad_accum_steps.{role}",
            minimum=1,
        )
        for role in ("control", "treatment")
    }
    if normalized_grad != {"control": 1, "treatment": 1}:
        raise SchemaError(
            f"{field}.fresh_integrity.expected_grad_accum_steps must be "
            "{'control': 1, 'treatment': 1}"
        )

    authority = json_mapping(
        qualification["execution_authority"],
        f"{field}.execution_authority",
    )
    require_keys(
        authority,
        f"{field}.execution_authority",
        ("path", "sha256"),
    )
    normalized_authority = {
        "path": require_text(
            authority["path"],
            f"{field}.execution_authority.path",
        ),
        "sha256": _sha256(
            authority["sha256"],
            f"{field}.execution_authority.sha256",
        ),
    }

    raw_schedule = qualification["gpu_schedule"]
    if not isinstance(raw_schedule, (list, tuple)):
        raise SchemaError(f"{field}.gpu_schedule must be an array")
    if len(raw_schedule) != QUALIFICATION_STAGE_PAIRS:
        raise SchemaError(
            f"{field}.gpu_schedule must contain exactly "
            f"{QUALIFICATION_STAGE_PAIRS} seed rows"
        )
    normalized_schedule: list[dict[str, Any]] = []
    schedule_seeds: set[int] = set()
    schedule_indices: set[int] = set()
    schedule_uuids: set[str] = set()
    for index, raw_row in enumerate(raw_schedule):
        row_field = f"{field}.gpu_schedule[{index}]"
        row = json_mapping(raw_row, row_field)
        require_keys(row, row_field, ("seed", "control", "treatment"))
        seed = require_int(row["seed"], f"{row_field}.seed", minimum=0)
        if seed in schedule_seeds:
            raise SchemaError(f"{field}.gpu_schedule has duplicate seed {seed}")
        schedule_seeds.add(seed)
        normalized_row: dict[str, Any] = {"seed": seed}
        for role in ("control", "treatment"):
            placement = json_mapping(row[role], f"{row_field}.{role}")
            require_keys(
                placement, f"{row_field}.{role}", ("index", "uuid")
            )
            gpu_index = require_int(
                placement["index"], f"{row_field}.{role}.index", minimum=0
            )
            gpu_uuid = require_text(
                placement["uuid"], f"{row_field}.{role}.uuid"
            )
            if not gpu_uuid.startswith("GPU-"):
                raise SchemaError(
                    f"{row_field}.{role}.uuid must start with 'GPU-'"
                )
            if gpu_index in schedule_indices or gpu_uuid in schedule_uuids:
                raise SchemaError(
                    f"{field}.gpu_schedule must use every index and UUID once"
                )
            schedule_indices.add(gpu_index)
            schedule_uuids.add(gpu_uuid)
            normalized_row[role] = {"index": gpu_index, "uuid": gpu_uuid}
        normalized_schedule.append(normalized_row)
    if schedule_indices != set(range(2 * QUALIFICATION_STAGE_PAIRS)):
        raise SchemaError(
            f"{field}.gpu_schedule must freeze GPU indices 0 through 7 exactly"
        )

    inventory = json_mapping(
        qualification["inventory_gate"], f"{field}.inventory_gate"
    )
    require_keys(
        inventory,
        f"{field}.inventory_gate",
        (
            "required_clean_snapshots",
            "minimum_interval_seconds",
            "require_full_node_clean",
        ),
    )
    clean_snapshots = require_int(
        inventory["required_clean_snapshots"],
        f"{field}.inventory_gate.required_clean_snapshots",
        minimum=2,
    )
    interval = require_number(
        inventory["minimum_interval_seconds"],
        f"{field}.inventory_gate.minimum_interval_seconds",
        minimum=0,
    )
    if interval < 5:
        raise SchemaError(
            f"{field}.inventory_gate.minimum_interval_seconds must be >= 5"
        )
    if require_bool(
        inventory["require_full_node_clean"],
        f"{field}.inventory_gate.require_full_node_clean",
    ) is not True:
        raise SchemaError(
            f"{field}.inventory_gate.require_full_node_clean must be true"
        )

    return {
        **qualification,
        "integrity_version": integrity_version,
        "equivalence_margin": float(equivalence_margin),
        "superiority_minimum_effect": float(superiority_minimum_effect),
        "required_helping_pairs": helping,
        "token_ratio_lower": float(token_lower),
        "token_ratio_upper": float(token_upper),
        "historical_anchors": anchors,
        "allowed_resolved_differences": list(allowed),
        "expected_arm_resolved": {
            "control": control_resolved,
            "treatment": treatment_resolved,
        },
        "fresh_integrity": {
            **fresh,
            "charged_seconds_min": float(charged_min),
            "charged_seconds_max_exclusive": float(charged_max),
            "expected_grad_accum_steps": normalized_grad,
        },
        "execution_authority": normalized_authority,
        "gpu_schedule": normalized_schedule,
        "inventory_gate": {
            **inventory,
            "required_clean_snapshots": clean_snapshots,
            "minimum_interval_seconds": float(interval),
        },
    }


def validate_search_policy(value: Any, experiment: Any | None = None) -> dict[str, Any]:
    """Validate the immutable search-policy block on a new experiment."""
    field = "experiment.search_policy"
    policy = json_mapping(value, field)
    require_keys(
        policy,
        field,
        (
            "version",
            "challenge_id",
            "decision_frame",
            "direction",
            "subsystem",
            "stage_pairs",
            "parent_experiment_id",
            "predictions",
            "stopping",
            "portfolio",
        ),
    )
    version = require_int(policy["version"], f"{field}.version", minimum=1)
    if version != SEARCH_POLICY_VERSION:
        raise SchemaError(
            f"{field}.version must be {SEARCH_POLICY_VERSION}, got {version}"
        )
    local_id(policy["challenge_id"], f"{field}.challenge_id")
    local_id(policy["decision_frame"], f"{field}.decision_frame")
    if str(policy["direction"]) not in RESEARCH_DIRECTIONS:
        raise SchemaError(
            f"{field}.direction must be one of {sorted(RESEARCH_DIRECTIONS)}"
        )
    local_id(policy["subsystem"], f"{field}.subsystem")
    qualification = _qualification(
        policy.get("qualification"), f"{field}.qualification"
    )
    stage_pairs = require_int(policy["stage_pairs"], f"{field}.stage_pairs", minimum=1)
    if qualification:
        if stage_pairs != QUALIFICATION_STAGE_PAIRS:
            raise SchemaError(
                f"{field}.qualification requires stage_pairs="
                f"{QUALIFICATION_STAGE_PAIRS}"
            )
    elif stage_pairs not in FUNNEL_STAGES:
        raise SchemaError(
            f"{field}.stage_pairs must be one of {list(FUNNEL_STAGES)} "
            f"unless an explicit four-pair qualification is declared"
        )

    parent_id = str(policy["parent_experiment_id"]).strip()
    if qualification:
        if parent_id:
            raise SchemaError(
                f"{field}.parent_experiment_id must be empty for a standalone "
                "qualification; qualification cannot release an ordinary child"
            )
    elif stage_pairs == 1:
        if parent_id:
            raise SchemaError(f"{field}.parent_experiment_id must be empty at stage 1")
    else:
        require_id(parent_id, "experiment", f"{field}.parent_experiment_id")

    predictions = json_mapping(policy["predictions"], f"{field}.predictions")
    require_keys(
        predictions,
        f"{field}.predictions",
        ("delta_steps", "delta_quality_per_step", "delta_endpoint"),
    )
    normalized_predictions = {
        name: _prediction(predictions[name], f"{field}.predictions.{name}")
        for name in ("delta_steps", "delta_quality_per_step", "delta_endpoint")
    }
    if not any(
        float(item["expected_delta"]) != 0.0
        for item in normalized_predictions.values()
    ):
        raise SchemaError(
            f"{field}.predictions must make at least one non-zero, falsifiable prediction"
        )

    stopping = json_mapping(policy["stopping"], f"{field}.stopping")
    require_keys(
        stopping,
        f"{field}.stopping",
        (
            "min_futility_pairs",
            "promote_if_mean_endpoint_delta_lte",
            "stop_if_mean_endpoint_delta_gte",
            "stop_if_mean_step_delta_lte",
        ),
    )
    minimum_pairs = require_int(
        stopping["min_futility_pairs"],
        f"{field}.stopping.min_futility_pairs",
        minimum=1,
    )
    expected_minimum = (
        QUALIFICATION_STAGE_PAIRS
        if qualification
        else (1 if stage_pairs == 1 else 3)
    )
    if minimum_pairs != expected_minimum:
        raise SchemaError(
            f"{field}.stopping.min_futility_pairs must be {expected_minimum} "
            f"at the {stage_pairs}-pair stage"
        )
    promote_at = require_number(
        stopping["promote_if_mean_endpoint_delta_lte"],
        f"{field}.stopping.promote_if_mean_endpoint_delta_lte",
    )
    stop_at = require_number(
        stopping["stop_if_mean_endpoint_delta_gte"],
        f"{field}.stopping.stop_if_mean_endpoint_delta_gte",
    )
    require_number(
        stopping["stop_if_mean_step_delta_lte"],
        f"{field}.stopping.stop_if_mean_step_delta_lte",
    )
    if promote_at >= stop_at:
        raise SchemaError(
            f"{field}.stopping promotion threshold must be below its futility threshold "
            "for a lower-is-better endpoint"
        )
    if promote_at >= 0 or stop_at < 0:
        raise SchemaError(
            f"{field}.stopping must require an endpoint improvement for promotion "
            "and use a non-negative regression boundary for futility"
        )
    expected_endpoint = float(
        normalized_predictions["delta_endpoint"]["expected_delta"]
    )
    if expected_endpoint > promote_at:
        raise SchemaError(
            f"{field}.predictions.delta_endpoint.expected_delta must clear the "
            "preregistered promotion threshold"
        )
    step_stop = float(stopping["stop_if_mean_step_delta_lte"])
    expected_steps = float(normalized_predictions["delta_steps"]["expected_delta"])
    if step_stop < min(0.0, expected_steps):
        raise SchemaError(
            f"{field}.stopping.stop_if_mean_step_delta_lte is too permissive "
            "for the predicted step effect"
        )

    portfolio = json_mapping(policy["portfolio"], f"{field}.portfolio")
    require_keys(
        portfolio,
        f"{field}.portfolio",
        ("max_consecutive_failures", "pivot_override"),
    )
    max_failures = require_int(
        portfolio["max_consecutive_failures"],
        f"{field}.portfolio.max_consecutive_failures",
        minimum=1,
    )
    if max_failures > 3:
        raise SchemaError(
            f"{field}.portfolio.max_consecutive_failures must be <= 3"
        )
    override = str(portfolio["pivot_override"]).strip()
    if override:
        raise SchemaError(
            f"{field}.portfolio.pivot_override must be empty in search policy v2; "
            "an automated campaign cannot waive its own subsystem-diversity gate"
        )

    if experiment is not None:
        if len(experiment.seeds) != stage_pairs:
            raise SchemaError(
                f"{experiment.experiment_id} freezes {len(experiment.seeds)} seeds, "
                f"but search_policy.stage_pairs={stage_pairs}"
            )
        controls = [arm for arm in experiment.arms if str(arm.get("role")) == "control"]
        treatments = [
            arm for arm in experiment.arms if str(arm.get("role")) == "treatment"
        ]
        if len(controls) != 1 or len(treatments) != 1:
            raise SchemaError(
                f"{experiment.experiment_id} search policy requires exactly one "
                "control arm and one treatment arm"
            )
        if qualification:
            anchor_seeds = tuple(
                int(anchor["seed"])
                for anchor in qualification["historical_anchors"]
            )
            if anchor_seeds != tuple(experiment.seeds):
                raise SchemaError(
                    f"{experiment.experiment_id} qualification anchors must match "
                    "the frozen experiment seeds in order"
                )
            if int(qualification.get("integrity_version", 0)) == 2:
                schedule_seeds = tuple(
                    int(row["seed"]) for row in qualification["gpu_schedule"]
                )
                if schedule_seeds != tuple(experiment.seeds):
                    raise SchemaError(
                        f"{experiment.experiment_id} qualification GPU schedule "
                        "must match the frozen experiment seeds in order"
                    )
        if stage_pairs == 1 and experiment.stage != "discovery":
            raise SchemaError(
                "the 1-pair search stage must use experiment.stage='discovery'; "
                "a pilot is implementation-only and cannot earn effect promotion"
            )
        if stage_pairs > 1 and experiment.stage == "pilot":
            raise SchemaError("search-funnel stages cannot use experiment.stage='pilot'")

    return {
        **policy,
        "predictions": normalized_predictions,
        "stopping": stopping,
        "portfolio": portfolio,
        "qualification": qualification,
    }


def _numeric_outcome(run: Any, key: str) -> float | None:
    value = run.outcome_values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _normalized_config_value(value: Any) -> str:
    text = str(value).strip()
    bool_alias = {"true": "1", "false": "0"}
    lowered = text.lower()
    if lowered in bool_alias:
        return bool_alias[lowered]
    try:
        return format(float(text), ".15g")
    except ValueError:
        return text


def _qualification_run_integrity_errors(
    run: Any,
    *,
    role: str,
    seed: int,
    qualification: Mapping[str, Any],
) -> list[str]:
    """Recheck every per-arm qualification integrity fact at evaluation time."""
    errors: list[str] = []
    if int(qualification.get("integrity_version", 0)) != 2:
        return ["qualification integrity_version is not executable v2"]
    fresh = qualification["fresh_integrity"]
    expected = qualification["expected_arm_resolved"][role]
    tracker = run.tracker
    resolved = tracker.get("resolved_training_config")
    if not isinstance(resolved, Mapping) or not resolved:
        errors.append("missing resolved_training_config")
        resolved = {}
    expected_keys = set(expected) | {"SEED"}
    if set(resolved) != expected_keys:
        errors.append(
            "resolved_training_config key set does not equal the frozen full "
            f"configuration: missing={sorted(expected_keys - set(resolved))}, "
            f"extra={sorted(set(resolved) - expected_keys)}"
        )
    if _normalized_config_value(resolved.get("SEED")) != str(seed):
        errors.append(
            f"resolved SEED={resolved.get('SEED')!r}, expected {seed}"
        )
    for key, value in expected.items():
        got = resolved.get(key)
        if got is None or _normalized_config_value(got) != _normalized_config_value(
            value
        ):
            errors.append(
                f"resolved {key}={got!r}, expected {value!r} for {role}"
            )

    charged = _numeric_outcome(run, "charged_training_seconds")
    charged_min = float(fresh["charged_seconds_min"])
    charged_max = float(fresh["charged_seconds_max_exclusive"])
    if (
        charged is None
        or not math.isfinite(charged)
        or not charged_min <= charged < charged_max
    ):
        errors.append(
            f"charged_training_seconds={charged!r} outside "
            f"[{charged_min},{charged_max})"
        )
    if tracker.get("compute_complete") is not True:
        errors.append("compute_complete is not true")
    if tracker.get("boundary_attested") is not True:
        errors.append("boundary attestation is not uniquely verified")
    if tracker.get("boundary_counts_verified") is not True:
        errors.append("terminal boundary counts are not verified")
    if tracker.get("boundary_digest_verified") is not True:
        errors.append("boundary-count digest is not recomputed and verified")
    if tracker.get("runtime_hashes_verified") is not True:
        errors.append("runtime/data hash attestation is not verified")
    authority = qualification["execution_authority"]
    if tracker.get("execution_authority_sha256") != authority["sha256"]:
        errors.append("execution-authority hash does not match the frozen policy")

    num_steps = _numeric_outcome(run, "num_steps")
    total_tokens = _numeric_outcome(run, "total_tokens")
    expected_batch = int(expected["TOTAL_BATCH_SIZE"])
    if (
        num_steps is None
        or total_tokens is None
        or int(total_tokens) != int(num_steps) * expected_batch
    ):
        errors.append(
            f"token arithmetic failed: total_tokens={total_tokens!r}, "
            f"num_steps={num_steps!r}, batch={expected_batch}"
        )
    grad_accum = _numeric_outcome(run, "grad_accum_steps")
    expected_grad = int(fresh["expected_grad_accum_steps"][role])
    if grad_accum is None or int(grad_accum) != expected_grad:
        errors.append(
            f"grad_accum_steps={grad_accum!r}, expected {expected_grad}"
        )
    boundary_batches = tracker.get("boundary_batches")
    if (
        num_steps is None
        or not isinstance(boundary_batches, int)
        or boundary_batches != int(num_steps) * expected_grad
    ):
        errors.append(
            f"boundary_batches={boundary_batches!r} does not equal "
            f"num_steps*grad_accum"
        )

    placement = next(
        (
            row[role]
            for row in qualification["gpu_schedule"]
            if int(row["seed"]) == seed
        ),
        None,
    )
    if placement is None:
        errors.append("missing frozen GPU placement")
    else:
        if tracker.get("gpu_id") != int(placement["index"]):
            errors.append(
                f"gpu_id={tracker.get('gpu_id')!r}, expected {placement['index']}"
            )
        if tracker.get("gpu_uuid") != str(placement["uuid"]):
            errors.append(
                f"gpu_uuid={tracker.get('gpu_uuid')!r}, expected {placement['uuid']}"
            )
    return errors


def _is_fingerprint(value: Any) -> bool:
    text = str(value)
    return len(text) == 16 and all(char in "0123456789abcdef" for char in text)


def _mean_t_interval(values: list[float]) -> tuple[float, float, float, float]:
    """Return mean, sample SD, and the fixed-df 90% interval endpoints."""
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    half_width = T_95_DF3 * sample_sd / math.sqrt(QUALIFICATION_STAGE_PAIRS)
    return mean, sample_sd, mean - half_width, mean + half_width


def _evaluate_qualification(
    experiment: Any,
    policy: Mapping[str, Any],
    control_id: str,
    treatment_id: str,
    by_key: Mapping[tuple[str, int], Any],
) -> dict[str, Any]:
    """Evaluate the preregistered replay/control qualification intersection."""
    qualification = policy["qualification"]
    anchors = {
        int(anchor["seed"]): anchor
        for anchor in qualification["historical_anchors"]
    }
    pairs: list[dict[str, float | int | str]] = []
    for seed in experiment.seeds:
        control = by_key.get((control_id, seed))
        treatment = by_key.get((treatment_id, seed))
        if control is None or treatment is None:
            continue
        if int(qualification.get("integrity_version", 0)) == 2:
            for role, run in (("control", control), ("treatment", treatment)):
                integrity_errors = _qualification_run_integrity_errors(
                    run,
                    role=role,
                    seed=seed,
                    qualification=qualification,
                )
                if integrity_errors:
                    return {
                        "verdict": "invalid",
                        "reason": (
                            f"seed {seed} {role} qualification integrity failed: "
                            + "; ".join(integrity_errors)
                        ),
                        "complete_pairs": len(pairs),
                        "stage_pairs": QUALIFICATION_STAGE_PAIRS,
                    }
            control_config = {
                str(key): _normalized_config_value(value)
                for key, value in control.tracker[
                    "resolved_training_config"
                ].items()
            }
            treatment_config = {
                str(key): _normalized_config_value(value)
                for key, value in treatment.tracker[
                    "resolved_training_config"
                ].items()
            }
            if set(control_config) != set(treatment_config):
                return {
                    "verdict": "invalid",
                    "reason": (
                        f"seed {seed} arms emitted different resolved-config key sets"
                    ),
                    "complete_pairs": len(pairs),
                    "stage_pairs": QUALIFICATION_STAGE_PAIRS,
                }
            observed_differences = {
                key
                for key in control_config
                if control_config[key] != treatment_config[key]
            }
            allowed_differences = set(
                qualification["allowed_resolved_differences"]
            )
            if observed_differences != allowed_differences:
                return {
                    "verdict": "invalid",
                    "reason": (
                        f"seed {seed} resolved-config difference set "
                        f"{sorted(observed_differences)} does not equal frozen "
                        f"{sorted(allowed_differences)}"
                    ),
                    "complete_pairs": len(pairs),
                    "stage_pairs": QUALIFICATION_STAGE_PAIRS,
                }
        control_endpoint = _numeric_outcome(control, "val_bpb")
        treatment_endpoint = _numeric_outcome(treatment, "val_bpb")
        treatment_tokens = _numeric_outcome(treatment, "total_tokens")
        if (
            control_endpoint is None
            or treatment_endpoint is None
            or treatment_tokens is None
            or treatment_tokens <= 0
        ):
            return {
                "verdict": "invalid",
                "reason": (
                    f"seed {seed} lacks numeric val_bpb or positive total_tokens; "
                    "qualification cannot be evaluated"
                ),
                "complete_pairs": len(pairs),
                "stage_pairs": QUALIFICATION_STAGE_PAIRS,
            }
        anchor = anchors[seed]
        endpoint_delta = treatment_endpoint - control_endpoint
        replay_delta = treatment_endpoint - float(anchor["val_bpb"])
        token_log_ratio = math.log(
            treatment_tokens / float(anchor["total_tokens"])
        )
        pairs.append(
            {
                "seed": seed,
                "endpoint_delta": endpoint_delta,
                "replay_delta": replay_delta,
                "token_log_ratio": token_log_ratio,
                "historical_val_bpb": float(anchor["val_bpb"]),
                "historical_total_tokens": int(anchor["total_tokens"]),
                "anchor_artifact_path": str(anchor["artifact_path"]),
                "anchor_artifact_sha256": str(anchor["artifact_sha256"]),
            }
        )

    count = len(pairs)
    if count < QUALIFICATION_STAGE_PAIRS:
        return {
            "verdict": "pending",
            "reason": (
                f"need {QUALIFICATION_STAGE_PAIRS} complete pairs for the "
                "qualification intersection"
            ),
            "complete_pairs": count,
            "stage_pairs": QUALIFICATION_STAGE_PAIRS,
            "pairs": pairs,
        }

    replay_values = [float(pair["replay_delta"]) for pair in pairs]
    endpoint_values = [float(pair["endpoint_delta"]) for pair in pairs]
    token_values = [float(pair["token_log_ratio"]) for pair in pairs]
    replay_mean, replay_sd, replay_lower, replay_upper = _mean_t_interval(
        replay_values
    )
    endpoint_mean, endpoint_sd, _, endpoint_upper = _mean_t_interval(
        endpoint_values
    )
    token_mean, token_sd, token_lower, token_upper = _mean_t_interval(
        token_values
    )

    margin = float(qualification["equivalence_margin"])
    minimum_effect = float(qualification["superiority_minimum_effect"])
    required_helping = int(qualification["required_helping_pairs"])
    helping = sum(value < 0 for value in endpoint_values)
    tolerance = 1e-12
    replay_pass = (
        replay_lower >= -margin - tolerance
        and replay_upper <= margin + tolerance
    )
    superiority_pass = (
        endpoint_upper <= -minimum_effect + tolerance
        and helping >= required_helping
    )
    log_ratio_lower = math.log(float(qualification["token_ratio_lower"]))
    log_ratio_upper = math.log(float(qualification["token_ratio_upper"]))
    token_pass = (
        token_lower >= log_ratio_lower - tolerance
        and token_upper <= log_ratio_upper + tolerance
    )
    passed = replay_pass and superiority_pass and token_pass
    return {
        "verdict": "qualification_pass" if passed else "qualification_fail",
        "reason": (
            "all replay-equivalence, contemporaneous minimum-effect, sign, and "
            "token-fidelity gates passed; this is non-adoption qualification only"
            if passed
            else "one or more preregistered qualification intersection gates failed"
        ),
        "complete_pairs": count,
        "stage_pairs": QUALIFICATION_STAGE_PAIRS,
        "pairs": pairs,
        "replay_equivalence": {
            "mean_delta": replay_mean,
            "sample_sd": replay_sd,
            "ci90_lower": replay_lower,
            "ci90_upper": replay_upper,
            "margin": margin,
            "passed": replay_pass,
        },
        "concurrent_superiority": {
            "mean_delta": endpoint_mean,
            "sample_sd": endpoint_sd,
            "one_sided_95_upper": endpoint_upper,
            "minimum_effect": minimum_effect,
            "helping_pairs": helping,
            "required_helping_pairs": required_helping,
            "passed": superiority_pass,
        },
        "token_fidelity": {
            "mean_log_ratio": token_mean,
            "sample_sd": token_sd,
            "ci90_lower": token_lower,
            "ci90_upper": token_upper,
            "required_log_ratio_lower": log_ratio_lower,
            "required_log_ratio_upper": log_ratio_upper,
            "passed": token_pass,
        },
        "releases_ordinary_child": False,
        "adoption_authorized": False,
    }


def evaluate_stage(experiment: Any, runs: Iterable[Any]) -> dict[str, Any]:
    """Evaluate completed A/B pairs against preregistered stopping thresholds."""
    policy = validate_search_policy(experiment.search_policy, experiment)
    control_id = next(
        str(arm["arm_id"])
        for arm in experiment.arms
        if str(arm.get("role")) == "control"
    )
    treatment_id = next(
        str(arm["arm_id"])
        for arm in experiment.arms
        if str(arm.get("role")) == "treatment"
    )

    challenge_id = str(policy["challenge_id"])
    expected_challenge_tag = f"challenge_{challenge_id}"
    by_key: dict[tuple[str, int], Any] = {}
    duplicates: list[str] = []
    for run in runs:
        if run.experiment_id != experiment.experiment_id:
            continue
        run_id = str(getattr(run, "run_id", f"{run.arm_id}/seed{run.seed}"))
        run_tags = {str(tag) for tag in run.tags}
        if any(tag.startswith("diagnostic_offbudget") for tag in run_tags):
            continue
        if policy["qualification"] and run.status in {"failed", "invalid"}:
            return {
                "verdict": "invalid",
                "reason": (
                    f"{run_id} is an immutable {run.status} qualification arm; "
                    "the attempt cannot be silently ignored or repaired"
                ),
                "complete_pairs": 0,
                "stage_pairs": QUALIFICATION_STAGE_PAIRS,
            }
        if run.status != "complete":
            continue
        challenge_tags = {tag for tag in run_tags if tag.startswith("challenge_")}
        if expected_challenge_tag not in challenge_tags:
            return {
                "verdict": "invalid",
                "reason": (
                    f"{run_id} is complete in the wrong challenge; expected "
                    f"{challenge_id!r}"
                ),
                "complete_pairs": 0,
                "stage_pairs": policy["stage_pairs"],
            }
        if (
            "bound_run" not in run_tags
            or "gpu_sampling_verified" not in run_tags
            or run.tracker.get("gpu_sampling_verified") is not True
            or run.tracker.get("challenge_id") != challenge_id
            or not _is_fingerprint(
                run.tracker.get("challenge_definition_fingerprint", "")
            )
            or not _is_fingerprint(
                run.tracker.get("challenge_selection_fingerprint", "")
            )
        ):
            return {
                "verdict": "invalid",
                "reason": (
                    f"{run_id} lacks bound-run, challenge, or GPU-sampling provenance; "
                    "policy checkpoints accept only gated, physically bound runs "
                    "whose training PID was observed without a sampled co-tenant"
                ),
                "complete_pairs": 0,
                "stage_pairs": policy["stage_pairs"],
            }
        key = (run.arm_id, run.seed)
        if key in by_key:
            duplicates.append(f"{run.arm_id}/seed{run.seed}")
        by_key[key] = run
    if duplicates:
        return {
            "verdict": "invalid",
            "reason": f"duplicate completed runs: {sorted(set(duplicates))}",
            "complete_pairs": 0,
            "stage_pairs": policy["stage_pairs"],
        }

    if policy["qualification"]:
        return _evaluate_qualification(
            experiment,
            policy,
            control_id,
            treatment_id,
            by_key,
        )

    pairs: list[dict[str, float | int]] = []
    for seed in experiment.seeds:
        control = by_key.get((control_id, seed))
        treatment = by_key.get((treatment_id, seed))
        if control is None or treatment is None:
            continue
        control_endpoint = _numeric_outcome(control, "val_bpb")
        treatment_endpoint = _numeric_outcome(treatment, "val_bpb")
        control_steps = _numeric_outcome(control, "num_steps")
        treatment_steps = _numeric_outcome(treatment, "num_steps")
        if None in (control_endpoint, treatment_endpoint, control_steps, treatment_steps):
            return {
                "verdict": "invalid",
                "reason": (
                    f"seed {seed} lacks numeric val_bpb or num_steps outcomes; "
                    "staged stopping cannot be evaluated"
                ),
                "complete_pairs": len(pairs),
                "stage_pairs": policy["stage_pairs"],
            }
        pairs.append(
            {
                "seed": seed,
                "endpoint_delta": treatment_endpoint - control_endpoint,
                "step_delta": treatment_steps - control_steps,
            }
        )

    count = len(pairs)
    minimum_pairs = int(policy["stopping"]["min_futility_pairs"])
    report: dict[str, Any] = {
        "complete_pairs": count,
        "stage_pairs": int(policy["stage_pairs"]),
        "pairs": pairs,
        "verdict": "pending",
        "reason": f"need {minimum_pairs} complete pairs for the first checkpoint",
    }
    if count < minimum_pairs:
        return report

    mean_endpoint = statistics.mean(float(pair["endpoint_delta"]) for pair in pairs)
    mean_steps = statistics.mean(float(pair["step_delta"]) for pair in pairs)
    report.update(
        {
            "mean_endpoint_delta": mean_endpoint,
            "mean_step_delta": mean_steps,
        }
    )
    stopping = policy["stopping"]
    if mean_endpoint >= float(stopping["stop_if_mean_endpoint_delta_gte"]):
        report.update(
            verdict="stop_futility",
            reason="endpoint regression crossed the preregistered futility boundary",
        )
        return report
    if (
        mean_steps <= float(stopping["stop_if_mean_step_delta_lte"])
        and mean_endpoint > float(stopping["promote_if_mean_endpoint_delta_lte"])
    ):
        report.update(
            verdict="stop_futility",
            reason="throughput loss crossed its boundary without compensating endpoint quality",
        )
        return report
    if mean_endpoint > float(stopping["promote_if_mean_endpoint_delta_lte"]):
        report.update(
            verdict="stop_no_signal",
            reason="checkpoint did not clear the preregistered promotion threshold",
        )
        return report
    if count < int(policy["stage_pairs"]):
        report.update(
            verdict="continue",
            reason="checkpoint cleared; authorize the next funnel tranche",
        )
        return report
    report.update(
        verdict=(
            "adoption_candidate"
            if int(policy["stage_pairs"]) == FUNNEL_STAGES[-1]
            else "promote"
        ),
        reason=(
            "10-pair confirmation cleared; adoption still requires the frame decision gate"
            if int(policy["stage_pairs"]) == FUNNEL_STAGES[-1]
            else "stage cleared; a proposal for the next funnel stage may be considered"
        ),
    )
    return report


def authorizable_seed_count(experiment: Any, evaluation: Mapping[str, Any]) -> int:
    """Return how many leading preregistered seeds may currently reach a GPU."""
    verdict = str(evaluation["verdict"])
    if verdict in TERMINAL_VERDICTS or verdict == "promote":
        return 0
    policy = validate_search_policy(experiment.search_policy, experiment)
    stage_pairs = int(policy["stage_pairs"])
    if policy["qualification"]:
        # Qualification has no efficacy interim and is designed as one concurrent
        # four-pair wave. Its terminal verdict never releases an ordinary child.
        return QUALIFICATION_STAGE_PAIRS
    complete = int(evaluation.get("complete_pairs", 0))
    if stage_pairs == 1:
        return 1
    if complete < 3:
        return min(3, stage_pairs)
    if complete < 6:
        return min(6, stage_pairs)
    return stage_pairs


__all__ = [
    "DIRECTION_ROUND_LIMIT",
    "FAILURE_UPDATE_RESULTS",
    "FUNNEL_STAGES",
    "QUALIFICATION_STAGE_PAIRS",
    "SEARCH_POLICY_VERSION",
    "T_95_DF3",
    "SUCCESS_UPDATE_RESULTS",
    "TERMINAL_VERDICTS",
    "authorizable_seed_count",
    "evaluate_stage",
    "validate_search_policy",
]
