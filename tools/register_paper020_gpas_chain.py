#!/usr/bin/env python3
"""Preflight or stage Paper-020 Round-1 GPAS attribution authority.

The default mode is read-only.  It constructs the complete typed chain for the
source-faithful GPAS applicability assay, verifies the paper and external-source
capsules, checks the active five-minute H200 frame, and reports every blocker.
It never launches work.

The local source-faithful model path and diagnostic primitives are hash-bound,
and the registry now recognizes a diagnostic-only GPU allocation without
consuming an endpoint-funnel stage.  The governed same-device replay,
role-swapped diagnostic evaluator, and complete immutable pair-artifact route
remain unimplemented and unbound, so this helper stays fail-closed.  It never
misclassifies the implementation-only pilot as the one-pair ``discovery``
stage of the ordinary endpoint funnel.  The endpoint prior and 1->3->6->10
funnel are preserved only as non-executable metadata for a separate future
proposal.

Once every blocker is implemented, a new audit must bind exact SHA-256 values
and rerun this helper.  Even then, ``--apply`` appends only exact missing
toolkit/idea records and the *planned* proposal.  It never writes directly to
the append-only gated ledger and therefore never creates launch authority.

Exact replays are idempotent.  A colliding ID with different typed content
fails closed.  ``--apply`` is protected by a repository-global registration
lock and refuses before the first write while any implementation, scheduler,
paper, setup, or executable-tool blocker remains.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "vibeautoresearch").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("could not locate the vibeautoresearch repository root")
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vibeautoresearch.core import SchemaError, canonical_json  # noqa: E402
from vibeautoresearch.evidence import resolve_terminal_evidence  # noqa: E402
from vibeautoresearch.experiments import ExperimentRecord  # noqa: E402
from vibeautoresearch.idea_archive import IdeaRecord, duplicate_ideas  # noqa: E402
from vibeautoresearch.ideas import HypothesisRecord, MechanismRecord  # noqa: E402
from vibeautoresearch.registry import ResearchRegistry  # noqa: E402
from vibeautoresearch.toolkit import (  # noqa: E402
    CapabilityGapRecord,
    InterventionRecord,
    ObservableRecord,
    ToolProposalRecord,
)


RESEARCH_ROOT = REPO_ROOT / "research"
LOCK_PATH = (
    REPO_ROOT
    / "tmp"
    / "research_registration_transactions"
    / "global.lock"
)

# This typed chain was constructed only after setup reconciliation v31
# (17:48:40Z) and challenge selection generation 16 (17:49:43Z).  Do not
# backdate it to the earlier paper-drafting window.
CREATED_AT = "2026-07-29T18:10:00Z"
SCHEMA_PREVIEW_FROZEN_AT = "2026-07-29T18:10:00Z"
SCOPE_ID = "walltime_5min_h200"
SCOPE_KEY: dict[str, Any] = {
    "data_split_sha256": (
        "ed8ea0554010df9fbfa47746adc6643d6446797753eb377c0b35413bdcf2fca3"
    ),
    "max_steps": 100000,
    "outcome_id": "out_val_bpb",
    "stop_mode": "time",
    "time_budget": 300,
}

PAPER020_TEX = (
    "AI_papers/"
    "paper_020_from_failed_attribution_to_local_mechanism_assays_20260729.tex"
)
PAPER020_PDF = (
    "AI_papers/"
    "paper_020_from_failed_attribution_to_local_mechanism_assays_20260729.pdf"
)
PAPER020_TEX_SHA256 = (
    "c9053c401cb2aa53027aa457bad4bf6ebe439af65b4cdae04eac190bed36e592"
)
PAPER020_PDF_SHA256 = (
    "1ebb95deee3f489b8016a0d28115e9aa4ec7725e7534deee93011d66725db9bb"
)
# The amendment was completed prospectively before any Round-1 execution and
# visually inspected on all six rendered pages.
PAPER020_OPTIMIZER_AUTHORITY_FINAL = True
# The paper's eager/compiled wording is frozen, but the later independent
# worker-boundary audit retired this execution authority before launch.  The
# monolithic train.py has no validation-inaccessible worker boundary, and the
# proposed registration-helper/runner digest closure is cyclic.  A prospectively
# amended import-safe training core plus one-way detached authority is required.
PAPER020_RUNTIME_AUTHORITY_FINAL = False

PREINTERVENTION_GIT_COMMIT = "16d8fae959b67f45ed1f6338aaa554a6fbb91f07"
PREINTERVENTION_GIT_BLOB_HASHES: dict[str, str] = {
    "train.py": "e2d4ac9e91c6ef716a8fdab35ab72da3e0e4bb23cbbecee04cc528b66d172b8b",
    "lib.py": "13de28bcb398866be65cfca489413818d75dc2fcd2461e367be06198f879123d",
    "research/setup/reconciliation.json": (
        "526dcdf38f90c83ca097776c69ec2d9610c6f39091c307f1d60f8403ce698acd"
    ),
}
AUDITED_PRECAPABILITY_SCHEDULER_HASHES: dict[str, str] = {
    "tools/run_stage.py": (
        "c7d3ab45af2983598936859ea9f14c34ebb6da3ce9b1d0cc9d00ca6c764b16c0"
    ),
    "tools/run_gated.py": (
        "47d08ec38f6dff53ed50cdc5e4ddaf88342da0f2e9c5993219baf4121711868a"
    ),
}

# The source-faithful model path and diagnostic primitives were independently
# handed off with these exact hashes.  They do not include a launch harness:
# the governed same-device replay, role swap, complete mediator evaluator, and
# immutable pair-artifact route remain separate scheduler blockers below.
REQUIRED_LOCAL_IMPLEMENTATION_FILES = frozenset(
    {
        "train.py",
        "tools/gpas_mechanism_diagnostic.py",
        "tests/test_gpas_impl.py",
        "lib.py",
        "research/setup/reconciliation.json",
    }
)
LOCAL_IMPLEMENTATION_HASHES: dict[str, str] = {
    "train.py": "6dc22cc5351dd37d9473d74bbbb2402b61c2db6477e302be5f87017a62b37906",
    "tools/gpas_mechanism_diagnostic.py": (
        "c1f31f046eac55eefb2c59c849e75db336ee75be278e7d7859d721e3a67eb437"
    ),
    "tests/test_gpas_impl.py": (
        "98c11af51af600d5f97ab874f219659987186755debb2ecb99046c2246856cb1"
    ),
    "lib.py": "05691d87f253b0f7692d2da40a909d6fed6ec83ea37f3e890a687f874ebf49c4",
    "research/setup/reconciliation.json": (
        "4acc742d43cc2736b31fbc47a355177b97926e709523098290fc138c8d5101c5"
    ),
}
# Scheduler readiness is an all-or-nothing capability boundary.  The runtime
# runner and its tests are now designed, but the separately reviewed
# no-validation worker, its tests, and the committed single-use authority do
# not exist.  Keep the implementation map empty until every exact path below
# exists, has been reviewed together, and can be bound to its final SHA-256.
RUNTIME_AUTHORITY_PATH = (
    "research/experiments/artifacts/paper020_gpas_runtime/"
    "execution_authority.json"
)
REQUIRED_SCHEDULER_IMPLEMENTATION_FILES = frozenset(
    {
        "vibeautoresearch/registry.py",
        "tools/run_gated.py",
        "tests/test_run_gated.py",
        "tests/test_diagnostic_pilot_authority.py",
        "tools/run_paper020_gpas_runtime.py",
        "tests/test_run_paper020_gpas_runtime.py",
        "tools/run_paper020_gpas_worker.py",
        "tests/test_run_paper020_gpas_worker.py",
        RUNTIME_AUTHORITY_PATH,
    }
)
SCHEDULER_IMPLEMENTATION_HASHES: dict[str, str] = {}

GPAS_CAPSULE_HASHES: dict[str, str] = {
    (
        "research/experiments/artifacts/external_code/"
        "pap_gpas_neurips2025/manifest.json"
    ): "871e6689facf3453a19c2d8cf565103d95b29f0ca3a16c0a493c2c0e0764093e",
    (
        "research/experiments/artifacts/external_code/"
        "pap_gpas_neurips2025/GPAS_NeurIPS_2025.pdf"
    ): "d644ed31e31a693fa889e01655e94cfea1cdb0c312901ee11b001af68b14a35e",
    (
        "research/experiments/artifacts/external_code/"
        "pap_gpas_neurips2025/gpas-31980688.tar.zst"
    ): "1177e6766f53cf0bc59b880a96ca09d7732984db62f8d54b9bf5083b1b45acf0",
    (
        "research/experiments/artifacts/external_code/"
        "pap_gpas_neurips2025/gpas-31980688.bundle"
    ): "8e2abb5ba02b16639d92cb532b3134c5fa4bcce0f43156585fa2e591294edaf1",
}
GPAS_UPSTREAM_COMMIT = "31980688f4cbb1b0cff59bca9077e6fc52dab3f0"
GPAS_UPSTREAM_TREE = "076a1d81a48f4d03dbd455d4197345480bd818bb"

PAPER_ID = "pap_gpas_neurips2025"
CLAIM_IDS = (
    "clm_gpas_gradient_preserving_scaling_contract",
    "clm_gpas_preln_multiscale_perplexity",
    "clm_gpas_stop_gradient_and_placement_ablation",
    "clm_gpas_activation_gradient_layer_importance",
)
LITERATURE_EVIDENCE_IDS = (
    "evd_lit_gpas_method_contract",
    "evd_lit_gpas_multiscale_pretraining",
    "evd_lit_gpas_mechanism_ablations",
    "evd_lit_gpas_dynamics",
)
OUTCOME_ID = "out_val_bpb"

TOOL_PROPOSAL_ID = "tlp_paper020_gpas_attribution_observable"
CAPABILITY_GAP_ID = "gap_paper020_gpas_attribution_authority"
OBSERVABLE_ID = "obs_paper020_gpas_post_mlp_residual_variance"
CONTROL_INTERVENTION_ID = "int_paper020_gpas_control"
TREATMENT_INTERVENTION_ID = "int_paper020_gpas_treatment"
MECHANISM_ID = "mech_paper020_gpas_residual_transport"
HYPOTHESIS_ID = "hyp_paper020_gpas_local_applicability"
IDEA_ID = "idea_paper020_gpas_shared_residual_gate"
EXPERIMENT_ID = "exp_paper020_gpas_attribution_diagnostic_s66"

EFFECTIVE_SIGMA = 0.001198
MINIMUM_ENDPOINT_EFFECT = 0.002396
EXPECTED_ENDPOINT_DELTA = -0.0030
DIAGNOSTIC_SEED = 66
DIAGNOSTIC_STEPS = 256

# Paper-020 freezes this prior for a *future ordinary endpoint proposal*.  It
# is intentionally not an ExperimentRecord.search_policy here: search policy
# v2 interprets stage_pairs=1 as a discovery-stage effect screen, whereas this
# Round-1 pilot measures no endpoint and can earn only implementation support.
FUTURE_ENDPOINT_PRIOR: dict[str, Any] = {
    "status": "non_executable_requires_separate_discovery_registration",
    "funnel_pairs": [1, 3, 6, 10],
    "predicted_raw_paired_val_bpb_delta_lte": EXPECTED_ENDPOINT_DELTA,
    "ordinary_promotion_magnitude_min": MINIMUM_ENDPOINT_EFFECT,
    "comparator": "fresh_same_seed_concurrent_current_R0",
    "diagnostic_pass_authority": (
        "may_propose_and_register_the_ordinary_one_pair_endpoint_stage_only"
    ),
    "forbidden_inference": (
        "this_diagnostic_is_not_a_discovery_pair_and_cannot_earn_effect_promotion"
    ),
}

RATING_AXIS_ORDER = (
    "novelty",
    "primary_source_provenance",
    "causal_validity",
    "expected_local_impact",
    "transfer_reliability",
    "feasibility_cost",
    "numerical_falsifiability",
    "connected_program_coherence",
)
PROPOSER_RATING: dict[str, int] = dict(
    zip(RATING_AXIS_ORDER, (4, 5, 5, 4, 4, 5, 5, 5), strict=True)
)
AUTHORITATIVE_CRITIC_RATING: dict[str, int] = dict(
    zip(RATING_AXIS_ORDER, (4, 5, 4, 3, 2, 5, 5, 5), strict=True)
)

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
    "PACKER_DOC_BOUNDARIES": "0",
}
CURRENT_R0_AUTHORITY: dict[str, Any] = {
    "name": "current_reconciled_R0",
    "setup_version": 40,
    "setup_fingerprint": "7e2b06b51256f737",
    "challenge_selection_fingerprint": "74f0a10788bf42f0",
    "operational_reference_val_bpb": 0.931857,
    "reference_status": "provisional_operational_not_scientific_adoption",
    "configuration_source": (
        "RE-AUDITED 2026-07-31 for reconciliation v35 (was v31/f5b656c0, reference "
        "0.934930). The guard above fired exactly as designed and this is the "
        "deliberate re-audit it demands, not a silent transport. Three things "
        "changed and each is evidence-backed: (1) v32 marked the 0.934930 baseline "
        "CONTAMINATED and set the frame pending, naming 'a fresh, honestly-measured "
        "baseline' as the promotion condition; (2) that condition is now met -- "
        "0.931857 from 10 concurrently-run control seeds with zero contention "
        "quarantines, corroborated by 0.932051 pooled over n=33 clean control arms "
        "from blocks 24-27; (3) the frame is therefore v35 status=passed. The R0 "
        "arm configuration itself is UNCHANGED: default GPAS-off train.py on the "
        "walltime_5min_h200 frame. Only the reference number it is compared against "
        "moved, and it moved because the old one was measured under contention."
    ),
    "historical_0_927183_role": "chart_origin_only_forbidden_as_arm_configuration",
}
PACKER_ZERO_DEFAULTS: dict[str, str] = {
    "PACKER_DOC_BOUNDARIES": "0",
    "PACKER_BOUNDARY_VERIFY_BATCHES": "0",
    "PACKER_DIAGNOSTIC_HASHES": "0",
    "PACKER_SIDECAR_ACTIVATE_STEP": "0",
}

GPAS_PARAMETER_CONTRACT: dict[str, Any] = {
    "formula": "z_out = z - silu(alpha[layer]) * stop_gradient(z)",
    "layers": 8,
    "parameter_tensor_count": 8,
    "per_tensor_shape": [],
    "total_numel": 8,
    "parameter_name_pattern": "transformer.h.{layer_0_to_7}.gpas_alpha",
    "conceptual_scalars_per_layer": 1,
    "applications_per_layer": 2,
    "application_sites": [
        "immediately_after_attention_residual_sum",
        "immediately_after_mlp_residual_sum",
    ],
    "shared_same_scalar_between_sites": True,
    "initialization": 0.0,
    "dtype_and_device": "runtime_same_as_resid_lambdas",
    "forward_scale": "1-silu(alpha[layer])",
    "hidden_state_jacobian": "identity",
    "source_commit": GPAS_UPSTREAM_COMMIT,
    "source_tree": GPAS_UPSTREAM_TREE,
}
GPAS_OPTIMIZER_CONTRACT: dict[str, Any] = {
    "kind": "adamw",
    "parameter_membership": (
        "exactly_eight_zero_dimensional_gpas_alpha_parameters_total_numel_8"
    ),
    "learning_rate": 0.005,
    "learning_rate_authority": "paper_frozen_absolute_constant",
    "learning_rate_derivation": "not_derived_from_current_R0_scalar_lr",
    "current_R0_scalar_lr": 0.8,
    "learning_rate_over_current_R0_scalar_lr": 0.00625,
    "prospective_paper_amendment_required": False,
    "paper_amendment_status": (
        "completed_prospectively_before_any_round1_execution"
    ),
    "betas": [0.8, 0.95],
    "eps": 1e-10,
    "weight_decay": 0.0,
    "demon_beta1": False,
    "is_x0_muon_warmdown": False,
    "forbidden_other_groups": ["muon", "rmsprop", "x0_adamw", "embedding_adamw"],
    "source_mapping_disclosure": (
        "OPHIS-specific extrapolation; the GPAS source uses Adam at 5e-4 "
        "for all parameters"
    ),
}

CANONICAL_REPLAY_CONTRACT: dict[str, Any] = {
    "hardware": "one_same_physical_H200_UUID",
    "construction": (
        "separately_constructed_control_and_treatment_from_the_same_seed_and_"
        "source_snapshot"
    ),
    "treatment_alpha": "all_exact_zero",
    "step": 0,
    "exact_equal": [
        "input_token_ids",
        "target_token_ids",
        "token_byte_counts",
        "boundaries_sha256",
        "data_cursor_and_epoch",
        "cpu_rng_state",
        "cuda_rng_state_before_forward",
        "compiled_graph_signature",
        "logits",
        "loss",
        "all_preexisting_parameter_gradients",
        "canonical_preexisting_model_state",
        "canonical_preexisting_optimizer_groups_and_state",
    ],
    "canonical_projection": {
        "treatment_only_model_keys_removed": [
            "transformer.h.0.gpas_alpha",
            "transformer.h.1.gpas_alpha",
            "transformer.h.2.gpas_alpha",
            "transformer.h.3.gpas_alpha",
            "transformer.h.4.gpas_alpha",
            "transformer.h.5.gpas_alpha",
            "transformer.h.6.gpas_alpha",
            "transformer.h.7.gpas_alpha",
        ],
        "treatment_only_optimizer_group_removed": "exact_gpas_alpha_group",
        "all_other_keys_order_shapes_dtypes_values_equal": True,
        "serialization": "sorted_key_dtype_shape_raw_bytes_sha256_v1",
    },
    "treatment_only_requirements": [
        "there_are_exactly_8_gpas_alpha_tensors_each_shape_empty_total_numel_8",
        "gpas_alpha_values_are_exactly_zero",
        "gpas_alpha_gradients_are_finite_and_nonzero",
        "gpas_optimizer_group_exactly_matches_frozen_contract",
    ],
    "packer_diagnostic_flags": dict(PACKER_ZERO_DEFAULTS),
    "on_any_mismatch": "INVALID_DIAGNOSTIC_NO_RELAXATION_OR_REPAIR",
}

APPLICABILITY_CONTRACT: dict[str, Any] = {
    "control_steps": 32,
    "row_source": (
        "control_steps_1_to_32_of_the_single_eager_256_step_mediator_pair"
    ),
    "separate_32_step_worker_execution": False,
    "training_only": True,
    "uses_validation": False,
    "tensor": (
        "post_MLP_residual_stream_after_the_second_GPAS_application_or_the_"
        "corresponding_control_site"
    ),
    "per_layer_statistic": (
        "FP64 mean_over_batch_and_sequence_of_population_variance_over_hidden_"
        "dimension_correction_0"
    ),
    "spearman": (
        "Pearson_correlation_of_layer_indices_0_to_7_and_average_tie_ranks_of_"
        "log_positive_layer_variances"
    ),
    "valid_step_conjunction": {
        "spearman_layer_index_vs_log_variance_min": 0.60,
        "v7_over_v0_min": 1.25,
    },
    "minimum_valid_steps": 24,
    "zero_or_nonfinite_variance": "fail_closed",
    "on_fail": "KILL_ROUND1_LOCAL_SOURCE_MECHANISM_ABSENT",
}

MEDIATION_CONTRACT: dict[str, Any] = {
    "paired_steps": DIAGNOSTIC_STEPS,
    "execution": "single_eager_training_only_control_treatment_pair",
    "compiled": False,
    "contains_control_applicability_steps": [1, 32],
    "aggregation": "arithmetic_mean_of_each_per_step_statistic_over_256_steps",
    "treatment_mean_v7_over_v0_over_control_max": 0.85,
    "treatment_mean_v7_over_control_mean_v7_max": 0.90,
    "endpoint_zero_gate_counterfactual": {
        "batch": "same_fixed_training_batch_no_update",
        "exact_batch_identity_fields": [
            "data_sha256",
            "targets_sha256",
            "token_bytes_sha256",
            "boundaries_sha256",
            "data_cursor",
            "epoch",
        ],
        "weights": "treatment_step256_weights",
        "gate_restore_attestation": (
            "exact_pre_counterfactual_gpas_values_sha256_equals_post_restore_sha256"
        ),
        "restore_fraction_formula": (
            "(counterfactual_v7_over_v0-treatment_v7_over_v0)/"
            "(control_v7_over_v0-treatment_v7_over_v0)"
        ),
        "minimum": 0.50,
        "nonpositive_denominator": "fail_closed",
    },
    "scale_safety": {
        "quantity": "1-silu(alpha[layer])",
        "inclusive_range": [0.5, 1.5],
        "finite": True,
        "all_steps_and_layers": True,
    },
    "gradient_safety": {
        "gpas_and_all_preexisting_gradients_finite": True,
        "initial_gpas_gradients_nonzero": True,
    },
    "deep_layer_contribution": {
        "layer_reduction": (
            "max(0,(control_v_l/control_v0)-(treatment_v_l/treatment_v0))"
        ),
        "numerator_layers": [4, 5, 6, 7],
        "denominator_layers": [1, 2, 3, 4, 5, 6, 7],
        "minimum_fraction": 0.50,
        "nonpositive_denominator": "fail_closed",
    },
}

RUNTIME_CONTRACT: dict[str, Any] = {
    "registered_diagnostic_seed_pairs": 1,
    "governed_pair_executions": 3,
    "execution_order": [
        "eager_256_step_mediator_pair_control_rows_1_to_32_supply_applicability",
        "compiled_256_step_clean_timing_control_A_treatment_B",
        "compiled_256_step_clean_timing_treatment_A_control_B",
    ],
    "fail_fast_barriers": [
        "evaluate_same_device_step0_replay_before_any_pair_execution",
        "evaluate_applicability_and_mediation_after_the_eager_pair_before_timing",
        "evaluate_each_clean_timing_compile_identity_before_the_next_execution",
    ],
    "mediator_execution": "eager_with_probes_not_compile_scored",
    "compiled_graph_scope": "each_clean_timing_arm_in_each_placement",
    "compiled_graph_count_per_clean_timing_arm": 1,
    "recompile_count_per_clean_timing_arm": 0,
    "cudagraph_recording_count_per_clean_timing_arm": 0,
    "resolved_compile_mode": "max-autotune-no-cudagraphs",
    "timing": {
        "placements": 2,
        "same_two_physical_UUIDs": True,
        "placement_1": "control_on_uuid_A_treatment_on_uuid_B",
        "placement_2": "treatment_on_uuid_A_control_on_uuid_B",
        "clean_step_range_inclusive": [33, 256],
        "clean_step_count_per_arm_per_placement": 224,
        "contiguous_block_size": 8,
        "blocks_per_arm_per_placement": 28,
        "role_normalization": (
            "geometric_mean_of_treatment_over_control_token_rate_across_the_"
            "two_role_swapped_placements"
        ),
        "bootstrap": {
            "unit": "paired_contiguous_blocks_within_placement",
            "resamples": 10000,
            "seed": 19019063,
            "one_sided_lcb_quantile": 0.05,
        },
        "point_ratio_min": 0.99,
        "one_sided_95pct_lcb_min": 0.985,
        "any_missing_or_nonfinite_block": "fail_closed",
    },
}

GPU_AUTHORITY_CONTRACT: dict[str, Any] = {
    "round1_registered_pairs": 1,
    "round1_seed": DIAGNOSTIC_SEED,
    "round1_governed_pair_executions": 3,
    "pair_execution_roles": [
        "eager_mediator_control_A_treatment_B",
        "compiled_timing_control_A_treatment_B",
        "compiled_timing_treatment_A_control_B",
    ],
    "same_device_replay_gpus": 1,
    "attribution_assay_gpus": 2,
    "required_gpu_product": "NVIDIA H200",
    "fixed_uuid_role_swap": True,
    "registry_authorization": (
        "exact_gated_experiment_fingerprint_and_diagnostic_only_policy_report"
    ),
    "cooperative_lock_contract": (
        "/tmp/vibeautoresearch-gpu-locks/"
        "stage-scheduler-gpu{physical_index}.lock_for_both_UUIDs_held_"
        "across_the_complete_transaction"
    ),
    "second_round1_pair_allowed": False,
    "opportunistic_filler_allowed": False,
    "other_free_gpus": "idle_unless_running_independent_CPU_or_non_GPU_analysis",
    "future_endpoint_funnel": {
        "requires_new_separately_gated_experiments": True,
        "stages": [1, 3, 6, 10],
        "independent_registered_seed_pairs_may_run_concurrently": True,
        "single_global_scheduler": True,
        "maximum_concurrent_pairs_on_8_H200": 3,
        "maximum_occupied_H200": 6,
        "attribution_reserve_H200": 2,
        "cross_stage_mixing": False,
        "post_freeze_remapping": False,
        "unregistered_filler": False,
    },
}


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise SchemaError(f"required authority file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_implementation_ready() -> bool:
    return set(LOCAL_IMPLEMENTATION_HASHES) == REQUIRED_LOCAL_IMPLEMENTATION_FILES


def _scheduler_ready() -> bool:
    return (
        set(SCHEDULER_IMPLEMENTATION_HASHES)
        == REQUIRED_SCHEDULER_IMPLEMENTATION_FILES
    )


def _implementation_ready() -> bool:
    return _local_implementation_ready() and _scheduler_ready()


def _ordinary_env(*, enabled: bool) -> dict[str, str]:
    return {
        **CURRENT_R0_ENV,
        "GPAS_ENABLE": "1" if enabled else "0",
    }


def _intervention(*, enabled: bool) -> InterventionRecord:
    role = "treatment" if enabled else "control"
    local_executable = _local_implementation_ready()
    return InterventionRecord(
        intervention_id=(
            TREATMENT_INTERVENTION_ID if enabled else CONTROL_INTERVENTION_ID
        ),
        name=(
            "Current R0 with source-faithful shared GPAS gates"
            if enabled
            else "Current R0 without GPAS"
        ),
        version=1,
        action=(
            "apply_shared_per_layer_gradient_preserving_activation_scaling"
            if enabled
            else "retain_current_residual_transport_without_gpas"
        ),
        target={
            "type": "residual_transport",
            "selector": "all_eight_blocks_after_attention_and_MLP_residual_sums",
        },
        parameters={
            "env": _ordinary_env(enabled=enabled),
            "comparator_authority": dict(CURRENT_R0_AUTHORITY),
            "gpas_parameter_contract": {
                **GPAS_PARAMETER_CONTRACT,
                "enabled": enabled,
                "allocated_parameters": 8 if enabled else 0,
            },
            "optimizer_group_contract": (
                dict(GPAS_OPTIMIZER_CONTRACT)
                if enabled
                else {
                    "kind": "absent",
                    "parameter_membership": "none",
                    "reason": (
                        "GPAS_ENABLE=0 allocates neither gpas_alpha parameters "
                        "nor their dedicated optimizer group"
                    ),
                    "canonical_replay_projection": (
                        "treatment_removes_exact_gpas_alpha_group_before_comparison"
                    ),
                }
            ),
            "packer_zero_defaults": dict(PACKER_ZERO_DEFAULTS),
        },
        timing={"stage": "run_start_and_every_transformer_block"},
        duration={"type": "full_diagnostic_run"},
        reversible=True,
        cost_tier="moderate",
        safety={
            "fast_fail_nonfinite_loss": True,
            "max_train_loss": 100,
            "same_data_order_required": True,
            "same_preexisting_initialization_required": True,
            "canonical_replay": dict(CANONICAL_REPLAY_CONTRACT),
            "applicability": dict(APPLICABILITY_CONTRACT),
            "mediation": dict(MEDIATION_CONTRACT),
            "runtime": dict(RUNTIME_CONTRACT),
            "validation_policy_control": False,
            "endpoint_scoring_authorized": False,
            "sota_update_authorized": False,
        },
        implementation={
            "entrypoint": (
                "train.py:Block.forward"
                if local_executable
                else "not_implemented:train.py:Paper020GPASDiagnostic"
            ),
            "test": (
                "tests/test_gpas_impl.py:GPASModelContractTest"
                if local_executable
                else "not_implemented:tests/test_gpas_impl.py"
            ),
        },
        status="unit_tested" if local_executable else "proposed",
        description=(
            f"Paper-020 Round-1 {role}. The only executable arm-environment "
            "difference is GPAS_ENABLE. The three packer diagnostic controls "
            "remain absent from env and therefore at their frozen zero defaults; "
            "PACKER_DOC_BOUNDARIES is explicitly zero."
        ),
        tags=(
            "paper020",
            "round1",
            "gpas",
            "residual_transport",
            "track_explore",
            "diagnostic_only",
            role,
        ),
    )


@dataclass(frozen=True)
class ChainRecords:
    tool_proposal: ToolProposalRecord
    capability_gap: CapabilityGapRecord
    observable: ObservableRecord
    control_intervention: InterventionRecord
    treatment_intervention: InterventionRecord
    mechanism: MechanismRecord
    hypothesis: HypothesisRecord
    idea: IdeaRecord
    proposal: ExperimentRecord
    gated_schema_preview: ExperimentRecord

    def append_sequence(self) -> tuple[tuple[str, Any], ...]:
        """Return the exact dependency-safe *planned* registration sequence."""
        return (
            ("observables", self.observable),
            ("interventions", self.control_intervention),
            ("interventions", self.treatment_intervention),
            ("mechanisms", self.mechanism),
            ("hypotheses", self.hypothesis),
            ("tool_proposals", self.tool_proposal),
            ("capability_gaps", self.capability_gap),
            ("idea_archive", self.idea),
            ("experiment_proposals", self.proposal),
        )


def records() -> ChainRecords:
    local_executable = _local_implementation_ready()
    workflow_executable = _implementation_ready()
    observable = ObservableRecord(
        observable_id=OBSERVABLE_ID,
        name="Post-MLP residual-stream variance by layer",
        version=1,
        sources=(
            {
                "name": "post_mlp_residual_stream",
                "target": (
                    "train.py:Block.forward immediately after the MLP residual "
                    "sum and second GPAS application"
                ),
                "tensor_type": "activation",
                "axes": ["hidden", "batch_sequence"],
                "context": "training_only_no_validation",
            },
        ),
        operation="axis_reduction_pipeline",
        reductions=(
            {
                "axis": "hidden",
                "op": "fp64_population_variance",
                "params": {"correction": 0},
            },
            {
                "axis": "batch_sequence",
                "op": "fp64_arithmetic_mean",
                "params": {"mask": "all_training_tokens_in_fixed_batch"},
            },
        ),
        output={"type": "per_layer_scalar", "units": "activation_variance"},
        collection={
            "mode": "online",
            "cadence_steps": 1,
            "cost_tier": "moderate",
            "estimated_overhead_pct": 1.0,
            "profile_status": "unprofiled",
        },
        causal_availability={
            "available_before_action": False,
            "uses_validation": False,
            "stage": "after_each_post_mlp_residual_update",
        },
        implementation={
            "entrypoint": (
                "tools/gpas_mechanism_diagnostic.py:ResidualVarianceRecorder"
                if local_executable
                else "not_implemented:train.py:Paper020GPASDiagnostic"
            ),
            "test": (
                "tests/test_gpas_impl.py:GPASDiagnosticContractTest"
                if local_executable
                else "not_implemented:tests/test_gpas_impl.py"
            ),
        },
        # The source sites and recorder exist, but the current focused tests do
        # not yet exercise the complete 32/256-step numerical evaluator.
        # "implemented" is intentionally non-executable at check-gate.
        status=(
            "unit_tested"
            if workflow_executable
            else ("implemented" if local_executable else "proposed")
        ),
        description=(
            "FP64 training-only mediator. It operationalizes the source's "
            "depthwise activation-variance mechanism without reading validation "
            "data and cannot itself support endpoint adoption or SOTA."
        ),
        tags=(
            "paper020",
            "gpas",
            "activation_variance",
            "mechanism_mediator",
            "training_only",
            "diagnostic",
        ),
    )
    control = _intervention(enabled=False)
    treatment = _intervention(enabled=True)
    mechanism = MechanismRecord(
        mechanism_id=MECHANISM_ID,
        name="Gradient-preserving scaling keeps deep residual features usable",
        version=1,
        description=(
            "A shared zero-initialized scalar after both residual sums in each "
            "layer downscales forward activations through a stopped-gradient "
            "branch while leaving the hidden-state Jacobian equal to identity. "
            "If OPHIS locally exhibits depthwise residual-variance growth, this "
            "can reduce that growth without attenuating backpropagation."
        ),
        origin_type="mixed",
        claim_ids=CLAIM_IDS,
        observation_ids=(),
        causal_chain=(
            "The current eight-layer residual stream may accumulate increasing post-MLP activation variance with depth.",
            "One zero-initialized alpha scalar per layer is reused after its attention and MLP residual sums.",
            "Subtracting SiLU(alpha) times the stopped-gradient state reduces forward magnitude while preserving an identity hidden-state Jacobian.",
            "The treatment should reduce the depthwise variance ratio, with deep layers supplying at least half of the reduction and a zero-gate counterfactual restoring the gap.",
            "Only after local mediation, numerical safety, compiler identity, and throughput pass may an ordinary paired BPB funnel be proposed.",
        ),
        assumptions=(
            "At least 24 of 32 control steps satisfy both rho>=0.60 and v7/v0>=1.25.",
            "The exact source placement transfers to OPHIS despite its learned residual/x0 paths, Peri-style normalizations, zero c_proj initialization, and hashed memory.",
            "The local AdamW scalar-group mapping is safe despite differing from the source's all-parameter Adam optimizer.",
            "A same-device alpha-zero canonical replay isolates the new parameter/group without relaxing equality after seeing results.",
            "The 256-step diagnostic is mechanism-only and cannot establish validation-BPB efficacy.",
        ),
        competing_mechanism_ids=(),
        scope={
            "domain": "transformer_lm_pretraining",
            "decision_frame": SCOPE_ID,
            "paper_round": "paper020_round1",
            "hardware": "H200",
            "model_depth": 8,
            "accepted_only_comparator": "current_reconciled_recipe_R0",
            "comparator_authority": dict(CURRENT_R0_AUTHORITY),
            "connected_program_position": (
                "gradient_preserving_residual_depth_before_hash_independence_"
                "and_attention_transport"
            ),
            "source_commit": GPAS_UPSTREAM_COMMIT,
            "source_tree": GPAS_UPSTREAM_TREE,
            "local_applicability_mediator": dict(APPLICABILITY_CONTRACT),
            "proposer_rating": dict(PROPOSER_RATING),
            "authoritative_critic_rating": dict(AUTHORITATIVE_CRITIC_RATING),
        },
        observable_predictions=(
            {
                "observable_id": OBSERVABLE_ID,
                "expected_pattern": (
                    "Control first shows the frozen local depth-growth premise; "
                    "then GPAS reduces mean v7/v0 to <=0.85 of control and mean "
                    "v7 to <=0.90 of control with deep-layer and zero-gate "
                    "mediation."
                ),
            },
        ),
        intervention_predictions=(
            {
                "intervention_id": TREATMENT_INTERVENTION_ID,
                "expected_change": (
                    "Source-faithful shared GPAS gates satisfy the full 256-step "
                    "mediation and safety intersection without token-rate point "
                    "ratio below 0.99 or one-sided 95% LCB below 0.985."
                ),
            },
        ),
        status="proposed",
        tags=(
            "paper020",
            "round1",
            "gpas",
            "residual_transport",
            "literature_grounded",
            "track_explore",
        ),
        notes=(
            "The source method contract and paper-scoped ablations are external "
            "evidence. Local applicability, optimizer mapping, compiled behavior, "
            "throughput, endpoint BPB, adoption, and SOTA all remain untested. "
            "R0 uses SSSL/DBS128/TBS262144/MATRIX_LR=0.03/momentum-continuous=0/"
            "warmdown=0.95. The historical TTTL/DBS72 recipe that produced "
            "0.927183 is forbidden as an arm."
        ),
    )
    hypothesis = HypothesisRecord(
        hypothesis_id=HYPOTHESIS_ID,
        title=(
            "Source-faithful GPAS mediates local depthwise residual-variance "
            "growth before endpoint scoring"
        ),
        version=1,
        mechanism_ids=(MECHANISM_ID,),
        context_ids=(),
        observable_predictions=(
            {
                "observable_id": OBSERVABLE_ID,
                "version": 1,
                "expected_pattern": {
                    "control_applicability": dict(APPLICABILITY_CONTRACT),
                    "treatment_mediation": dict(MEDIATION_CONTRACT),
                },
                "window": {
                    "applicability_steps": [1, 32],
                    "attribution_steps": [1, DIAGNOSTIC_STEPS],
                    "endpoint_zero_gate_counterfactual": DIAGNOSTIC_STEPS,
                },
            },
        ),
        intervention={
            "intervention_id": TREATMENT_INTERVENTION_ID,
            "version": 1,
            "parameters": {
                "env": _ordinary_env(enabled=True),
                "gpas_parameter_contract": dict(GPAS_PARAMETER_CONTRACT),
                "optimizer_group_contract": dict(GPAS_OPTIMIZER_CONTRACT),
            },
        },
        trigger={
            "type": "run_start",
            "observable_id": "",
            "condition": {
                "operator": (
                    "execute_treatment_only_after_source_hash_unit_tests_and_"
                    "same_device_zero_effect_replay_pass"
                )
            },
        },
        outcome={"outcome_id": OUTCOME_ID, "version": 1},
        prediction=(
            "If the control satisfies the local depth-growth premise, GPAS will "
            "pass every frozen mediation, gate-safety, graph, and counterbalanced "
            "timing threshold. A pass supports only implementation/applicability "
            "and authorizes drafting a separate ordinary 1->3->6->10 endpoint "
            "proposal; it is not a BPB result."
        ),
        expected_effect={
            "direction": "decrease",
            "estimand": (
                "paired_treatment_over_control_depthwise_post_mlp_residual_"
                "variance_ratios_with_zero_gate_mediation"
            ),
            "latency_steps": 1,
        },
        controls=(
            {
                "control_id": "current_recipe_gpas_disabled",
                "kind": "no_intervention",
                "description": (
                    "Fresh current-recipe control with GPAS_ENABLE=0, the same "
                    "source/seed/data/checkpoint contract, all PACKER_* defaults "
                    "zero, and symmetric diagnostic collection."
                ),
                "matching": {
                    "same_checkpoint": True,
                    "same_data_order": True,
                },
            },
            {
                "control_id": "treatment_alpha_zero_same_device_replay",
                "kind": "negative_control",
                "description": (
                    "Separately constructed alpha-zero treatment on the same "
                    "physical H200; canonical pre-existing state and all forward/"
                    "backward facts must exactly match control."
                ),
                "matching": {
                    "same_checkpoint": True,
                    "same_data_order": True,
                },
            },
        ),
        falsification={
            "minimum_seeds": 1,
            "decision_rule": (
                "This is an implementation-only intersection gate. The control "
                "must first satisfy applicability; then every replay, mediation, "
                "safety, compiler, and counterbalanced timing criterion must pass."
            ),
            "failure_condition": (
                "Any source/hash/config mismatch, alpha-zero canonical mismatch, "
                "absent local depth-growth premise, failed mediator, nonfinite or "
                "unsafe gate/gradient, extra graph/recompile/cudagraph, timing "
                "point ratio <0.99, timing LCB <0.985, co-tenancy, or scheduler "
                "integrity fault kills Round 1 without endpoint scoring."
            ),
        },
        estimated_cost={
            "gpu_hours": 0.5,
            "currency_cost": 0.0,
            "cost_tier": "moderate",
            "basis": (
                "One same-device zero-effect replay, one eager two-H200 "
                "256-step mediator pair, and two clean compiled two-H200 "
                "256-step timing executions with roles swapped on the same "
                "UUIDs. These are three executions of one registered diagnostic "
                "seed/pair, not three endpoint-funnel pairs."
            ),
        },
        status=(
            "approved_for_pilot" if workflow_executable else "blocked"
        ),
        created_at=CREATED_AT,
        tags=(
            "paper020",
            "round1",
            "gpas",
            "residual_transport",
            "track_explore",
            "diagnostic_only",
        ),
        notes=(
            "Historical experiment 501 at 0.927183 is chart provenance only and "
            "is neither this diagnostic's control nor an endpoint target. The "
            "current accepted-only R0 recipe is the comparator. No local SOTA "
            "claim is possible from this assay."
        ),
    )
    tool_proposal = ToolProposalRecord(
        proposal_id=TOOL_PROPOSAL_ID,
        tool_type="observable",
        name="GPAS same-device residual-variance attribution diagnostic",
        motivation=(
            "Paper 020 forbids endpoint work until OPHIS first exhibits the "
            "source mechanism and the shared GPAS gate causally mediates it."
        ),
        required_by_hypotheses=(HYPOTHESIS_ID,),
        estimated_implementation_cost="moderate",
        estimated_runtime_cost="moderate",
        suggested_collection_frequency_steps=1,
        acceptance_tests=(
            "eight zero-dimensional zero-init parameters, one shared exactly "
            "twice within each layer",
            "dedicated AdamW group exactly matches the frozen optimizer contract",
            "same-device alpha-zero canonical replay is bitwise exact after removing only GPAS-only state",
            "FP64 population variance implementation matches a tiny eager oracle",
            "control rows 1-32 of the eager 256-step pair supply applicability and the full pair supplies mediation",
            "zero-gate endpoint counterfactual uses the same treatment weights and fixed training batch",
            "each clean compiled timing arm has one graph, zero recompiles, and zero cudagraph recordings",
            "two fixed UUIDs complete both compiled role-swapped timing executions with no co-tenancy",
        ),
        # The local primitives exist, but several acceptance tests require the
        # missing governed harness/evaluator.  "approved" avoids overstating
        # completion while preserving that the proposal passed design review.
        status=(
            "implemented"
            if workflow_executable
            else ("approved" if local_executable else "proposed")
        ),
        created_at=CREATED_AT,
    )
    capability_gap = CapabilityGapRecord(
        gap_id=CAPABILITY_GAP_ID,
        question=(
            "Can OPHIS execute the source-faithful GPAS local-mechanism assay "
            "with same-device canonical replay and counterbalanced timing?"
        ),
        missing_capabilities=(
            "complete tested 32/256-step applicability, mediation, safety, counterfactual, and timing evaluator",
            "governed same-device sequential alpha-zero replay before the paired assay",
            "three-execution same-two-UUID scheduler and immutable diagnostic evaluator",
            "remote/local source-hash equality for the frozen local implementation",
        ),
        blocked_hypothesis_ids=(HYPOTHESIS_ID,),
        priority="high",
        recommended_solution=(
            "Retain the hash-bound default-off GPAS path and diagnostic "
            "primitives; complete their numerical evaluator and extend the "
            "single locked run_stage/run_gated route with a Paper-020 diagnostic "
            "manifest that performs one same-device replay, one eager mediator "
            "pair, then two compiled role-swapped timing executions on the same "
            "two clean UUIDs; emit one immutable evaluator "
            "artifact through the implementation-only diagnostic allocation gate "
            "that is distinct from search-policy-v2 discovery; bind every "
            "local/remote SHA-256 before changing this gap to resolved."
        ),
        status="resolved" if workflow_executable else "open",
        created_at=CREATED_AT,
    )
    proposer_vector = "/".join(str(PROPOSER_RATING[key]) for key in RATING_AXIS_ORDER)
    critic_vector = "/".join(
        str(AUTHORITATIVE_CRITIC_RATING[key]) for key in RATING_AXIS_ORDER
    )
    idea = IdeaRecord(
        idea_id=IDEA_ID,
        title="Source-faithful shared GPAS gates with local mechanism first",
        version=1,
        summary=(
            "Reuse one zero-initialized scalar twice per layer to downscale "
            "forward residual states through stop-gradient while preserving the "
            "hidden-state Jacobian, but require local variance mediation before "
            "any ordinary BPB experiment."
        ),
        experimental_plan=(
            "Bind source and local implementation hashes; pass one same-device "
            "alpha-zero canonical replay; use control rows 1-32 of one eager "
            "256-step mediator pair for applicability; then run two clean "
            "compiled 256-step timing executions with roles swapped on the same "
            "two UUIDs. Apply every mediation, safety, graph, and timing gate. "
            "A pass only permits a new endpoint paper/proposal and does not "
            "enter the 1->3->6->10 funnel automatically."
        ),
        direction="architecture",
        subsystem="gpas_residual_transport",
        parent_idea_ids=(),
        scores={
            "interestingness": {
                "score": 9,
                "rationale": (
                    "The source isolates stop-gradient and predicts a measurable "
                    "local mediator before a costly endpoint campaign."
                ),
            },
            "novelty": {
                "score": 6,
                "rationale": (
                    "GPAS itself is prior art; novelty lies only in the explicit "
                    "OPHIS transfer assay, canonical replay, and causal funnel."
                ),
            },
            "feasibility": {
                "score": 8,
                "rationale": (
                    "The model change is eight scalars, but exact replay, probes, "
                    "scheduler support, and H200 timing must be implemented first."
                ),
            },
        },
        novelty_check={
            "provider": "literature_registry",
            "status": "passed",
            "query_rounds": [
                {
                    "query": (
                        "GPAS gradient-preserving activation scaling stop-gradient "
                        "shared layer gate residual variance pretraining"
                    ),
                    "result_paper_ids": [PAPER_ID],
                    "assessment": (
                        "The NeurIPS 2025 paper and exact official commit define "
                        "the method and paper-scoped evidence; they do not test "
                        "the eight-layer OPHIS transfer or this attribution design."
                    ),
                },
                {
                    "query": (
                        "same-device zero-effect replay GPAS causal mediation "
                        "counterbalanced H200 throughput"
                    ),
                    "result_paper_ids": [PAPER_ID],
                    "assessment": (
                        "No registered source supplies this complete local "
                        "replay/mediator/timing protocol. The algorithm remains "
                        "source-faithful rather than claimed as new."
                    ),
                },
            ],
            "closest_paper_ids": [PAPER_ID],
            "evidence_ids": list(LITERATURE_EVIDENCE_IDS),
            "max_semantic_similarity": 0.75,
            "discard_threshold": 0.90,
            "assessment": (
                "Novelty is restricted to local transfer and attribution. "
                "Method provenance remains Chen et al.; source results are not "
                "local efficacy evidence."
            ),
        },
        status="selected",
        hypothesis_id=HYPOTHESIS_ID,
        created_at=CREATED_AT,
        notes=(
            "Paper-020 proposer rating (N/P/V/I/R/F/X/C) = "
            f"{proposer_vector} = {sum(PROPOSER_RATING.values())}/40. "
            "Authoritative independent critic rating = "
            f"{critic_vector} = {sum(AUTHORITATIVE_CRITIC_RATING.values())}/40. "
            "The earlier 34/40 portfolio screen is superseded for execution by "
            "the deeper 33/40 transfer audit; its reliability=2 reflects the "
            "eight-layer/local-optimizer mismatch. GPAS ranks first because its "
            "stop-gradient ablation and variance mediator make the mechanism "
            "unusually falsifiable, not because venue prestige proves transfer."
        ),
    )
    proposal = ExperimentRecord(
        experiment_id=EXPERIMENT_ID,
        title=(
            "Paper-020 Round 1: source-faithful GPAS 256-step attribution "
            "diagnostic"
        ),
        version=1,
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis_fingerprint=hypothesis.fingerprint,
        stage="pilot",
        status="planned",
        arms=(
            {
                "arm_id": "gpas_disabled_control",
                "role": "control",
                "description": (
                    "Current accepted-only R0 recipe, symmetric diagnostic "
                    "collection, GPAS_ENABLE=0."
                ),
                "intervention_id": CONTROL_INTERVENTION_ID,
                "trigger": {"type": "run_start"},
                "control_id": "",
            },
            {
                "arm_id": "gpas_shared_gate_treatment",
                "role": "treatment",
                "description": (
                    "Exact same recipe and diagnostic environment except "
                    "GPAS_ENABLE=1; one shared zero-init scalar twice per layer."
                ),
                "intervention_id": TREATMENT_INTERVENTION_ID,
                "trigger": {"type": "run_start"},
                "control_id": "gpas_disabled_control",
            },
        ),
        seeds=(DIAGNOSTIC_SEED,),
        checkpoint={
            "matching": "same_checkpoint_per_seed",
            "source": "canonical_random_init_seed66_with_separate_construction",
        },
        randomization={
            "unit": "seed_physical_gpu_uuid_and_role_order",
            "method": (
                "First sequential canonical replay on one fixed H200. Then one "
                "eager 256-step mediator pair on two frozen clean UUIDs, one "
                "clean compiled 256-step timing execution in the same placement, "
                "and one clean compiled timing execution with roles swapped on "
                "those UUIDs. No remap or second seed."
            ),
        },
        analysis_plan={
            "primary_estimand": (
                "paired_treatment_over_control_depthwise_post_mlp_residual_"
                "variance_ratios_with_zero_gate_mediation"
            ),
            "outcome_id": OUTCOME_ID,
            "baseline_covariates": [
                "training_step",
                "train_loss",
                "learning_rate",
                "data_digest",
                "boundary_digest",
                "gpu_uuid",
                "physical_role",
                "token_rate",
            ],
            "uncertainty_method": (
                "deterministic_intersection_gates_plus_preregistered_whole_block_"
                "bootstrap_for_counterbalanced_timing_only"
            ),
            "multiplicity": {
                "family_id": "fam_paper020_round1_gpas_attribution",
                "method": "intersection_union_all_preregistered_gates_required",
            },
            "diagnostic_only": True,
            "validation_data_access": False,
            "endpoint_val_bpb_measured": False,
            "canonical_replay": dict(CANONICAL_REPLAY_CONTRACT),
            "applicability": dict(APPLICABILITY_CONTRACT),
            "mediation": dict(MEDIATION_CONTRACT),
            "optimizer_group_contract": dict(GPAS_OPTIMIZER_CONTRACT),
            "runtime": dict(RUNTIME_CONTRACT),
            "gpu_authority": dict(GPU_AUTHORITY_CONTRACT),
            "future_endpoint_prior": dict(FUTURE_ENDPOINT_PRIOR),
            "historical_0_927183_role": "chart_origin_only_not_control_or_target",
            "comparator_authority": dict(CURRENT_R0_AUTHORITY),
        },
        budget={
            "estimated_gpu_hours": 0.5,
            "estimated_currency_cost": 0.0,
            "hard_cap_currency_cost": 0.0,
            "basis": (
                "One same-device replay and exactly three governed executions "
                "of one registered control/treatment seed: one eager mediator "
                "pair plus two clean compiled timing placements with roles "
                "swapped. At most two GPUs may be occupied; six remain idle "
                "because no other diagnostic pair is registered."
            ),
        },
        promotion_gate={
            "verdict": "implementation_only",
            "criteria": [
                "paper, source capsule, local implementation, scheduler, setup, and remote/local hashes match exactly",
                "all PACKER_* diagnostics resolve to zero and the two arm envs differ only GPAS_ENABLE",
                "same-device alpha-zero canonical replay exactly matches every frozen control fact after removing only GPAS-only state",
                "there are exactly eight zero-dimensional gpas_alpha tensors, "
                "one per layer, each zero initialized and shared at both residual "
                "sites, with finite nonzero initial gradients and the exact AdamW group",
                "at least 24/32 control steps jointly satisfy Spearman>=0.60 and v7/v0>=1.25",
                "over 256 steps treatment mean(v7/v0)/control mean(v7/v0)<=0.85 and treatment mean(v7)/control mean(v7)<=0.90",
                "step-256 zero-gate counterfactual restores at least 50% of the variance-ratio gap",
                "all forward scales stay finite in [0.5,1.5], all gradients are finite, and layers 4-7 contribute >=50% of aggregate reduction",
                "each clean timing arm has one graph, zero recompiles, zero cudagraph recordings; role-normalized timing point ratio>=0.99 and one-sided 95% LCB>=0.985",
                "one registered seed/pair, exactly three governed pair executions, fixed UUID timing role swap, no co-tenancy/remapping/second diagnostic pair/unregistered filler",
            ],
            "on_pass": (
                "Record implementation/applicability support only. Draft and "
                "register a separate ordinary one-pair endpoint proposal under "
                "the Paper-020 1->3->6->10 funnel; do not adopt, rebind the "
                "comparator, update the chart, or claim SOTA."
            ),
            "on_fail": (
                "Record the first failed causal link and kill Paper-020 Round 1. "
                "Leave R0 unchanged; do not relax equality, repair, relaunch, "
                "reinterpret, score validation BPB, or claim a mechanism result "
                "when the diagnostic is invalid."
            ),
        },
        data_policy={
            "split": "train_shards_1_10_diagnostic_only",
            "proposal_loop_access": True,
            "scope_key": dict(SCOPE_KEY),
        },
        created_at=CREATED_AT,
        search_policy=None,
        idea_id=IDEA_ID,
        frozen_at="",
        tags=(
            "paper020",
            "round1",
            "gpas",
            "architecture",
            "track_explore",
            "diagnostic_only",
            "implementation_only",
        ),
        notes=(
            "This proposal contains no endpoint efficacy, adoption, comparator "
            "succession, chart point, or SOTA authority. Generic two-GPU "
            "--diagnostic-steps support is insufficient until same-device replay "
            "and same-UUID role swapping are implemented and hash-bound."
        ),
    )
    gated_schema_preview = replace(
        proposal,
        status="approved",
        frozen_at=SCHEMA_PREVIEW_FROZEN_AT,
    )
    return ChainRecords(
        tool_proposal=tool_proposal,
        capability_gap=capability_gap,
        observable=observable,
        control_intervention=control,
        treatment_intervention=treatment,
        mechanism=mechanism,
        hypothesis=hypothesis,
        idea=idea,
        proposal=proposal,
        gated_schema_preview=gated_schema_preview,
    )


class _OverlayStore:
    """Read-only in-memory registry overlay used for prospective checks."""

    def __init__(self, base: Any, additions: tuple[Any, ...]):
        self._base = base
        self._additions = additions

    def by_id(self) -> dict[str, Any]:
        records_ = self._base.by_id()
        for record in self._additions:
            identifier = str(record.registry_id)
            existing = records_.get(identifier)
            if existing is not None and canonical_json(
                existing.to_dict()
            ) != canonical_json(record.to_dict()):
                raise SchemaError(
                    f"ID {identifier!r} exists with different typed content"
                )
            records_[identifier] = record
        return records_

    def load(self) -> list[Any]:
        return list(self.by_id().values())


@contextmanager
def _prospective_overlay(
    registry: ResearchRegistry, chain: ChainRecords
) -> Iterator[None]:
    originals = {
        "observables": registry.observables,
        "interventions": registry.interventions,
        "mechanisms": registry.mechanisms,
        "hypotheses": registry.hypotheses,
        "tool_proposals": registry.tool_proposals,
        "capability_gaps": registry.capability_gaps,
        "idea_archive": registry.idea_archive,
        "experiment_proposals": registry.experiment_proposals,
    }
    additions = {
        "observables": (chain.observable,),
        "interventions": (
            chain.control_intervention,
            chain.treatment_intervention,
        ),
        "mechanisms": (chain.mechanism,),
        "hypotheses": (chain.hypothesis,),
        "tool_proposals": (chain.tool_proposal,),
        "capability_gaps": (chain.capability_gap,),
        "idea_archive": (chain.idea,),
        "experiment_proposals": (chain.proposal,),
    }
    for name, records_ in additions.items():
        setattr(registry, name, _OverlayStore(originals[name], records_))
    try:
        yield
    finally:
        for name, store in originals.items():
            setattr(registry, name, store)


def _exact_action(store: Any, record: Any, *, apply: bool) -> str:
    existing = store.by_id().get(str(record.registry_id))
    if existing is not None:
        if canonical_json(existing.to_dict()) != canonical_json(record.to_dict()):
            raise SchemaError(
                f"ID {record.registry_id!r} exists with different typed content"
            )
        return "skip_exact"
    if apply:
        store.add(record)
        return "appended"
    return "would_append"


def _validate_ratings() -> dict[str, Any]:
    expected_proposer = dict(
        zip(RATING_AXIS_ORDER, (4, 5, 5, 4, 4, 5, 5, 5), strict=True)
    )
    expected_critic = dict(
        zip(RATING_AXIS_ORDER, (4, 5, 4, 3, 2, 5, 5, 5), strict=True)
    )
    if PROPOSER_RATING != expected_proposer or sum(PROPOSER_RATING.values()) != 37:
        raise SchemaError("Paper-020 GPAS proposer vector changed from 37/40")
    if (
        AUTHORITATIVE_CRITIC_RATING != expected_critic
        or sum(AUTHORITATIVE_CRITIC_RATING.values()) != 33
    ):
        raise SchemaError(
            "Paper-020 GPAS authoritative critic vector changed from 33/40"
        )
    return {
        "axis_order": list(RATING_AXIS_ORDER),
        "proposer": {
            "vector": list(PROPOSER_RATING.values()),
            "total": 37,
        },
        "authoritative_critic": {
            "vector": list(AUTHORITATIVE_CRITIC_RATING.values()),
            "total": 33,
        },
    }


def _validate_hashes(
    expected: Mapping[str, str], *, label: str
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for relative, digest in expected.items():
        path = REPO_ROOT / relative
        actual = _sha256(path)
        if actual != digest:
            raise SchemaError(
                f"{label} SHA-256 mismatch for {relative}: "
                f"expected {digest}, got {actual}"
            )
        report[relative] = {
            "sha256": actual,
            "bytes": path.stat().st_size,
            "present": True,
        }
    return report


def _validate_git_blob_hashes(
    commit: str,
    expected: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Verify preserved pre-intervention bytes without comparing them to live files."""

    report: dict[str, dict[str, Any]] = {}
    for relative, digest in expected.items():
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise SchemaError(
                f"cannot read pre-intervention Git blob {commit}:{relative}: "
                f"{detail}"
            )
        actual = hashlib.sha256(completed.stdout).hexdigest()
        if actual != digest:
            raise SchemaError(
                f"pre-intervention Git blob SHA-256 mismatch for {relative}: "
                f"expected {digest}, got {actual}"
            )
        report[relative] = {
            "git_commit": commit,
            "sha256": actual,
            "bytes": len(completed.stdout),
            "present": True,
        }
    return report


def _source_contract() -> dict[str, Any]:
    paper = _validate_hashes(
        {
            PAPER020_TEX: PAPER020_TEX_SHA256,
            PAPER020_PDF: PAPER020_PDF_SHA256,
        },
        label="Paper-020",
    )
    capsule = _validate_hashes(GPAS_CAPSULE_HASHES, label="GPAS capsule")
    preintervention = _validate_git_blob_hashes(
        PREINTERVENTION_GIT_COMMIT,
        PREINTERVENTION_GIT_BLOB_HASHES,
    )
    local_ready = _local_implementation_ready()
    scheduler_ready = _scheduler_ready()
    if local_ready:
        local = _validate_hashes(
            LOCAL_IMPLEMENTATION_HASHES,
            label="local GPAS implementation",
        )
    else:
        local = {}
    if scheduler_ready:
        scheduler = _validate_hashes(
            SCHEDULER_IMPLEMENTATION_HASHES,
            label="GPAS diagnostic scheduler",
        )
        scheduler_baseline = {}
        scheduler_live_unbound = {}
    else:
        # Preserve the exact audited pre-capability scheduler as a Git-blob
        # baseline.  Live scheduler edits are allowed during implementation,
        # but cannot become authority until their completed hashes are reviewed
        # and moved into SCHEDULER_IMPLEMENTATION_HASHES.
        scheduler = {}
        scheduler_baseline = _validate_git_blob_hashes(
            PREINTERVENTION_GIT_COMMIT,
            AUDITED_PRECAPABILITY_SCHEDULER_HASHES,
        )
        scheduler_live_unbound = {
            relative: {
                "sha256": _sha256(REPO_ROOT / relative),
                "bound_for_execution": False,
            }
            for relative in AUDITED_PRECAPABILITY_SCHEDULER_HASHES
        }
    if local_ready and scheduler_ready:
        mode = "implementation_and_scheduler_hashes_bound"
    elif local_ready:
        mode = "local_implementation_bound_scheduler_blocked"
    else:
        mode = "preimplementation_source_blocked"
    return {
        "mode": mode,
        "paper020": paper,
        "gpas_capsule": capsule,
        "preintervention_git_blobs": preintervention,
        "local": local,
        "scheduler": scheduler,
        "scheduler_pre_capability_git_blobs": scheduler_baseline,
        "scheduler_live_unbound": scheduler_live_unbound,
        "upstream_commit": GPAS_UPSTREAM_COMMIT,
        "upstream_tree": GPAS_UPSTREAM_TREE,
        "local_implementation_ready": local_ready,
        "scheduler_ready": scheduler_ready,
        "required_scheduler_implementation_files": sorted(
            REQUIRED_SCHEDULER_IMPLEMENTATION_FILES
        ),
        "missing_scheduler_implementation_files": sorted(
            REQUIRED_SCHEDULER_IMPLEMENTATION_FILES
            - set(SCHEDULER_IMPLEMENTATION_HASHES)
        ),
        "unexpected_scheduler_implementation_files": sorted(
            set(SCHEDULER_IMPLEMENTATION_HASHES)
            - REQUIRED_SCHEDULER_IMPLEMENTATION_FILES
        ),
        "implementation_ready": local_ready and scheduler_ready,
    }


def _validate_external_refs(registry: ResearchRegistry) -> dict[str, Any]:
    papers = registry.papers.by_id()
    claims = registry.claims.by_id()
    evidence = resolve_terminal_evidence(
        registry.literature_evidence.by_id()
    ).terminals
    outcomes = registry.outcomes.by_id()
    missing = {
        "paper": [] if PAPER_ID in papers else [PAPER_ID],
        "claims": [claim_id for claim_id in CLAIM_IDS if claim_id not in claims],
        "literature_evidence": [
            evidence_id
            for evidence_id in LITERATURE_EVIDENCE_IDS
            if evidence_id not in evidence
        ],
        "outcome": [] if OUTCOME_ID in outcomes else [OUTCOME_ID],
    }
    unresolved = {key: value for key, value in missing.items() if value}
    if unresolved:
        raise SchemaError(f"missing Paper-020 GPAS prerequisites: {unresolved}")
    for claim_id in CLAIM_IDS:
        if claims[claim_id].paper_id != PAPER_ID:
            raise SchemaError(f"{claim_id} does not bind {PAPER_ID}")
    for claim_id, evidence_id in zip(
        CLAIM_IDS, LITERATURE_EVIDENCE_IDS, strict=True
    ):
        item = evidence[evidence_id]
        if claim_id not in item.claim_ids:
            raise SchemaError(f"{evidence_id} does not assess {claim_id}")
        if item.assessment["relation"] != "supports":
            raise SchemaError(f"{evidence_id} is not terminal support for {claim_id}")
    return {
        "paper_id": PAPER_ID,
        "claim_ids": list(CLAIM_IDS),
        "terminal_literature_evidence_ids": list(LITERATURE_EVIDENCE_IDS),
        "outcome_id": OUTCOME_ID,
    }


def _validate_arm_separation(chain: ChainRecords) -> dict[str, Any]:
    control_params = chain.control_intervention.parameters
    treatment_params = chain.treatment_intervention.parameters
    control_env = dict(control_params["env"])
    treatment_env = dict(treatment_params["env"])
    differing = {
        key
        for key in set(control_env) | set(treatment_env)
        if control_env.get(key) != treatment_env.get(key)
    }
    if differing != {"GPAS_ENABLE"}:
        raise SchemaError(
            "Paper-020 diagnostic arm envs must differ only GPAS_ENABLE, "
            f"found {sorted(differing)}"
        )
    if control_env["GPAS_ENABLE"] != "0" or treatment_env["GPAS_ENABLE"] != "1":
        raise SchemaError("GPAS_ENABLE must be control=0 and treatment=1")
    forbidden_in_env = {
        "PACKER_BOUNDARY_VERIFY_BATCHES",
        "PACKER_DIAGNOSTIC_HASHES",
        "PACKER_SIDECAR_ACTIVATE_STEP",
    }
    leaked = forbidden_in_env & (set(control_env) | set(treatment_env))
    if leaked:
        raise SchemaError(
            "diagnostic-only PACKER controls may not enter arm env even at zero: "
            f"{sorted(leaked)}"
        )
    for params in (control_params, treatment_params):
        if dict(params["packer_zero_defaults"]) != PACKER_ZERO_DEFAULTS:
            raise SchemaError("PACKER_* zero-default contract changed")
    treatment_contract = dict(treatment_params["gpas_parameter_contract"])
    if (
        treatment_contract["parameter_tensor_count"] != 8
        or treatment_contract["per_tensor_shape"] != []
        or treatment_contract["total_numel"] != 8
        or treatment_contract["applications_per_layer"] != 2
        or treatment_contract["shared_same_scalar_between_sites"] is not True
        or treatment_contract["initialization"] != 0.0
    ):
        raise SchemaError("source-faithful shared GPAS parameter contract changed")
    if (
        dict(treatment_params["optimizer_group_contract"])
        != GPAS_OPTIMIZER_CONTRACT
    ):
        raise SchemaError("GPAS optimizer group contract changed")
    control_optimizer = dict(control_params["optimizer_group_contract"])
    if (
        control_optimizer.get("kind") != "absent"
        or control_optimizer.get("parameter_membership") != "none"
    ):
        raise SchemaError("GPAS-disabled control must have no GPAS optimizer group")
    return {
        "only_env_difference": "GPAS_ENABLE",
        "control": "0",
        "treatment": "1",
        "packer_zero_defaults": dict(PACKER_ZERO_DEFAULTS),
        "source_faithful_parameter_contract": True,
        "optimizer_group_contract": dict(GPAS_OPTIMIZER_CONTRACT),
    }


def _validate_current_r0(
    registry: ResearchRegistry,
    setup: Any,
    frame: Mapping[str, Any],
    selection_event: Any,
) -> dict[str, Any]:
    """Bind Paper-020 to current R0, never the historical 0.927 recipe."""

    expected = {
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
        "PACKER_DOC_BOUNDARIES": "0",
    }
    if CURRENT_R0_ENV != expected:
        raise SchemaError("current R0 environment changed from Paper-020 authority")
    if (
        int(setup.version) != int(CURRENT_R0_AUTHORITY["setup_version"])
        or setup.fingerprint != CURRENT_R0_AUTHORITY["setup_fingerprint"]
        or selection_event.fingerprint
        != CURRENT_R0_AUTHORITY["challenge_selection_fingerprint"]
    ):
        raise SchemaError(
            "current R0 setup/challenge fingerprints changed; re-audit rather "
            "than silently transporting an old arm"
        )
    baseline = frame["baseline"]
    if (
        float(baseline["observed"])
        != CURRENT_R0_AUTHORITY["operational_reference_val_bpb"]
        or str(baseline["code"]["train.py"])
        != LOCAL_IMPLEMENTATION_HASHES["train.py"]
    ):
        raise SchemaError("current R0 operational reference/code binding changed")

    # The existing explicit-current intervention supplies an independent typed
    # cross-check for every non-frame/default coordinate.  ATTN/compile are
    # injected by the selected frame; SCALAR_LR and PACKER_DOC_BOUNDARIES are
    # explicit current source defaults.
    explicit = registry.interventions.by_id().get(
        "int_block22_current_recipe_explicit"
    )
    if explicit is None:
        raise SchemaError("missing explicit current-R0 intervention cross-check")
    typed_env = dict(explicit.parameters["env"])
    expected_typed = {
        key: expected[key]
        for key in (
            "DEVICE_BATCH_SIZE",
            "DOC_MASK",
            "DOC_MASK_IMPL",
            "DOC_MASK_MODE",
            "MATRIX_LR",
            "MUON_MOMENTUM_CONTINUOUS",
            "NGRAM_TABLE_MULT",
            "TOTAL_BATCH_SIZE",
            "WARMDOWN_RATIO",
            "WINDOW_PATTERN",
        )
    }
    if typed_env != expected_typed:
        raise SchemaError(
            "typed current-R0 cross-check differs from Paper-020 comparator"
        )
    forbidden_historical = {
        "DEVICE_BATCH_SIZE": "72",
        "MATRIX_LR": "0.04",
        "MUON_MOMENTUM_CONTINUOUS": "1",
        "TOTAL_BATCH_SIZE": "147456",
        "WARMDOWN_RATIO": "0.85",
        "WINDOW_PATTERN": "TTTL",
    }
    accidental_matches = {
        key: value
        for key, value in forbidden_historical.items()
        if CURRENT_R0_ENV.get(key) == value
    }
    if accidental_matches:
        raise SchemaError(
            "current R0 accidentally imports historical 0.927 coordinates: "
            f"{accidental_matches}"
        )
    return {
        **CURRENT_R0_AUTHORITY,
        "env": dict(CURRENT_R0_ENV),
        "typed_cross_check_intervention_id": explicit.intervention_id,
        "forbidden_historical_coordinates_absent": True,
    }


def _prospective_check(
    registry: ResearchRegistry, chain: ChainRecords
) -> dict[str, Any]:
    validation = registry.validate(check_generated_state=False).to_dict()
    if validation.get("warnings"):
        raise SchemaError(
            "pre-existing authoritative validation warnings: "
            f"{validation['warnings']}"
        )
    source = _source_contract()
    refs = _validate_external_refs(registry)
    ratings = _validate_ratings()
    arms = _validate_arm_separation(chain)

    setup = registry._require_current_setup(SCOPE_ID)
    frame = setup.scope_for(SCOPE_ID)
    if dict(frame["scope_key"]) != SCOPE_KEY or frame["status"] != "passed":
        raise SchemaError(f"{SCOPE_ID} no longer matches the frozen scope")
    selected, event = registry.selected_challenge()
    if (
        selected["challenge_id"] != SCOPE_ID
        or selected["scope_id"] != SCOPE_ID
        or event.action != "activated"
    ):
        raise SchemaError(f"sticky active challenge is not activated {SCOPE_ID}")
    current_r0 = _validate_current_r0(registry, setup, frame, event)

    registry._validate_analysis_gate(chain.proposal)
    if (
        canonical_json(chain.proposal.definition())
        != canonical_json(chain.gated_schema_preview.definition())
        or chain.proposal.fingerprint != chain.gated_schema_preview.fingerprint
    ):
        raise SchemaError(
            "planned proposal and approved schema preview are not definition-identical"
        )
    current_ideas = registry.idea_archive.load()
    duplicates = [
        item
        for item in duplicate_ideas(current_ideas + [chain.idea])
        if chain.idea.idea_id in item[:2]
    ]
    if duplicates:
        raise SchemaError(f"prospective GPAS idea fails novelty dedup: {duplicates}")

    with _prospective_overlay(registry, chain):
        registry._require_literature_assessments(
            chain.proposal.experiment_id, chain.hypothesis
        )
        diagnostic_policy_report = registry._validate_search_gate(
            chain.proposal,
            scope_id=SCOPE_ID,
            frame=frame,
        )
        if (
            diagnostic_policy_report.get("diagnostic_only") is not True
            or diagnostic_policy_report.get("effect_funnel_stage_consumed")
            is not False
            or diagnostic_policy_report.get("endpoint_scoring_authorized")
            is not False
            or diagnostic_policy_report.get("adoption_authorized") is not False
            or int(diagnostic_policy_report.get("registered_pairs", 0)) != 1
            or int(diagnostic_policy_report.get("seed", -1)) != DIAGNOSTIC_SEED
        ):
            raise SchemaError(
                "registry diagnostic-only allocation report is not fail-closed: "
                f"{diagnostic_policy_report}"
            )
        executable_blocker = ""
        try:
            registry._validate_executable_gate(
                chain.proposal,
                chain.hypothesis,
                registry.observables.by_id(),
                registry.interventions.by_id(),
                registry.contexts.by_id(),
                registry.outcomes.by_id(),
            )
        except SchemaError as exc:
            executable_blocker = str(exc)

    blockers: list[str] = []
    if not PAPER020_OPTIMIZER_AUTHORITY_FINAL:
        blockers.append(
            "Paper-020 optimizer authority is not final: its current TeX/PDF "
            "incorrectly describes the paper-frozen GPAS lr=0.005 as "
            "current scalar_lr*0.01 even though current R0 SCALAR_LR=0.8; "
            "prospective amendment and new reviewed hashes are required"
        )
    if not PAPER020_RUNTIME_AUTHORITY_FINAL:
        blockers.append(
            "Paper-020 runtime authority was retired by "
            "aud_paper020_worker_boundary_20260729: the frozen train.py "
            "monolith has no validation-inaccessible training worker and the "
            "registration-helper/runner digest graph is cyclic; an import-safe "
            "training core and one-way detached authority must be prospectively "
            "paper-, source-, setup-, runner-, worker-, and test-bound"
        )
    if not _local_implementation_ready():
        missing_local = sorted(
            REQUIRED_LOCAL_IMPLEMENTATION_FILES
            - set(LOCAL_IMPLEMENTATION_HASHES)
        )
        unexpected_local = sorted(
            set(LOCAL_IMPLEMENTATION_HASHES)
            - REQUIRED_LOCAL_IMPLEMENTATION_FILES
        )
        blockers.append(
            "LOCAL_IMPLEMENTATION_HASHES does not exactly cover the frozen "
            "source-faithful model, evaluator/helper, and focused tests; "
            f"missing={missing_local}, unexpected={unexpected_local}"
        )
    if not _scheduler_ready():
        missing_scheduler = sorted(
            REQUIRED_SCHEDULER_IMPLEMENTATION_FILES
            - set(SCHEDULER_IMPLEMENTATION_HASHES)
        )
        unexpected_scheduler = sorted(
            set(SCHEDULER_IMPLEMENTATION_HASHES)
            - REQUIRED_SCHEDULER_IMPLEMENTATION_FILES
        )
        blockers.append(
            "SCHEDULER_IMPLEMENTATION_HASHES does not exactly cover the frozen "
            "Paper-020 workflow: the runtime design exists, but the dedicated "
            "no-validation worker, worker tests, and committed detached "
            "authority remain required; "
            f"missing={missing_scheduler}, unexpected={unexpected_scheduler}"
        )
    if executable_blocker:
        blockers.append(executable_blocker)
    if chain.capability_gap.status != "resolved":
        blockers.append(
            f"{CAPABILITY_GAP_ID} remains {chain.capability_gap.status}"
        )
    if chain.hypothesis.status != "approved_for_pilot":
        blockers.append(
            f"{HYPOTHESIS_ID} remains {chain.hypothesis.status}"
        )

    return {
        "authoritative_validation": validation,
        "source_contract": source,
        "external_refs": refs,
        "ratings": ratings,
        "arm_separation": arms,
        "current_r0": current_r0,
        "scope_key": dict(frame["scope_key"]),
        "setup_fingerprint": setup.fingerprint,
        "challenge_selection_fingerprint": event.fingerprint,
        "search_policy": {
            "attached": False,
            "reason": "implementation_only_diagnostic_not_endpoint_discovery",
            "future_endpoint_prior": dict(FUTURE_ENDPOINT_PRIOR),
            "diagnostic_policy_authority": dict(diagnostic_policy_report),
        },
        "proposal_fingerprint": chain.proposal.fingerprint,
        "gated_schema_preview_fingerprint": (
            chain.gated_schema_preview.fingerprint
        ),
        "definition_identical": True,
        "gpu_authority": dict(GPU_AUTHORITY_CONTRACT),
        "blockers": blockers,
        "proposal_registration_ready": not blockers,
        "gated_registration_supported": False,
        "gated_registration_blocker": (
            "The registry exposes no atomic proposal-to-gated promotion. After "
            "all blockers clear and the planned proposal is persisted, run the "
            "normal check-gate/freeze path; never append gated authority here."
        ),
    }


def preflight_or_apply(*, apply: bool) -> dict[str, Any]:
    registry = ResearchRegistry(RESEARCH_ROOT)
    chain = records()
    gate = _prospective_check(registry, chain)

    # Resolve every exact-ID action before any potential write so a conflict in
    # a later registry cannot leave an earlier partial append.
    actions: list[dict[str, str]] = []
    for store_name, record in chain.append_sequence():
        action = _exact_action(
            getattr(registry, store_name), record, apply=False
        )
        actions.append(
            {
                "registry": store_name,
                "id": str(record.registry_id),
                "action": action,
            }
        )

    if apply and gate["blockers"]:
        raise SchemaError(
            "Paper-020 GPAS apply is blocked before any write: "
            + "; ".join(gate["blockers"])
        )

    if apply:
        applied_actions: list[dict[str, str]] = []
        for store_name, record in chain.append_sequence():
            action = _exact_action(
                getattr(registry, store_name), record, apply=True
            )
            applied_actions.append(
                {
                    "registry": store_name,
                    "id": str(record.registry_id),
                    "action": action,
                }
            )
        actions = applied_actions
        post = registry.validate(check_generated_state=False).to_dict()
        if post.get("warnings"):
            raise SchemaError(
                "post-append authoritative validation warnings: "
                f"{post['warnings']}"
            )
    else:
        post = gate["authoritative_validation"]

    return {
        "mode": "apply" if apply else "preflight",
        "actions": actions,
        "prospective_gate": gate,
        "post_validation": post,
        "gated_schema_preview": {
            "id": chain.gated_schema_preview.experiment_id,
            "status": chain.gated_schema_preview.status,
            "fingerprint": chain.gated_schema_preview.fingerprint,
            "to_dict": chain.gated_schema_preview.to_dict(),
            "schema_only": True,
            "persisted_by_this_helper": False,
        },
        "launch_authority": False,
        "endpoint_scoring_authority": False,
        "comparator_rebind": False,
        "chart_update": False,
        "sota_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "append exact missing records and the planned proposal only after "
            "all implementation/scheduler blockers are hash-bound; default is "
            "read-only preflight"
        ),
    )
    args = parser.parse_args()

    if not args.apply:
        result = preflight_or_apply(apply=False)
    else:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOCK_PATH.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SchemaError(
                    f"global research-registration lock is held: {LOCK_PATH}"
                ) from exc
            try:
                result = preflight_or_apply(apply=True)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SchemaError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
