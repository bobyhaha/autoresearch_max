#!/usr/bin/env python3
"""Preflight or stage the Paper-019 round-1 FA3 boundary-sidecar chain.

The default mode is read-only. It constructs every prospective object through
the repository dataclasses, checks the live graph and current five-minute H200
frame, and evaluates the same private gate components used by ``check_gate``
against an in-memory overlay. It never launches work.

``--apply`` appends exact missing toolkit/idea records and the *planned*
experiment proposal in dependency order. It deliberately does not append the
approved experiment: the current registry has no atomic proposal-to-gated
promotion API, and writing directly to the append-only gated ledger would
bypass ``check_gate``. The helper emits the byte-identical approved-form
preview so the normal freeze operation can copy the unchanged definition after
the persisted proposal clears ``check_gate``.

Replays are idempotent only for exact typed serialization. Any colliding ID
with different content fails closed. Non-scored packer diagnostics are
prerequisites, not ordinary-arm environment variables.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import json
from pathlib import Path
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
from vibeautoresearch.search_policy import validate_search_policy  # noqa: E402
from vibeautoresearch.toolkit import InterventionRecord  # noqa: E402


RESEARCH_ROOT = REPO_ROOT / "research"
ARTIFACT_ROOT = (
    RESEARCH_ROOT
    / "experiments"
    / "artifacts"
    / "paper019_round1_fa3_boundary_sidecar"
)
LOCK_PATH = (
    REPO_ROOT
    / "tmp"
    / "research_registration_transactions"
    / "paper019_round1_fa3_boundary_sidecar.lock"
)

CREATED_AT = "2026-07-29T15:40:00Z"
FROZEN_AT = "2026-07-29T16:49:30Z"
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

PAPER_ID = "pap_flash_attention_3_neurips2024"
CLAIM_ID = "clm_fa3_varlen_external_cuseqlens_contract"
LITERATURE_EVIDENCE_ID = "evd_lit_fa3_varlen_api_contract"
OUTCOME_ID = "out_val_bpb"

CONTROL_INTERVENTION_ID = "int_paper019_fa3_boundary_sidecar_control"
TREATMENT_INTERVENTION_ID = "int_paper019_fa3_boundary_sidecar_treatment"
MECHANISM_ID = "mech_paper019_fa3_boundary_sidecar"
HYPOTHESIS_ID = "hyp_paper019_fa3_boundary_sidecar"
IDEA_ID = "idea_paper019_fa3_boundary_sidecar"
EXPERIMENT_ID = "exp_paper019_fa3_boundary_sidecar_s1"

EFFECTIVE_SIGMA = 0.001198
MINIMUM_EFFECT = 0.002396
EXPECTED_ENDPOINT_DELTA = -0.002396
THROUGHPUT_POINT_RATIO = 1.054
THROUGHPUT_LCB_RATIO = 1.041

# Paper authority: AI_papers/paper_019_observe_attribute_then_improve_20260729.tex
# uses N/P/V/I/R/F/X/C and lists this exact critic vector.
PAPER_CRITIC_RATING: dict[str, int] = {
    "novelty": 4,
    "provenance": 4,
    "validity": 5,
    "impact": 3,
    "reliability": 3,
    "feasibility": 4,
    "falsifiability": 5,
    "coherence": 5,
}
IMPLEMENTATION_AWARE_CRITIC_RATING: dict[str, int] = {
    "novelty": 4,
    "provenance": 4,
    "validity": 5,
    "impact": 2,
    "reliability": 2,
    "feasibility": 3,
    "falsifiability": 5,
    "coherence": 5,
}

RECOVERED_RECIPE_ENV: dict[str, str] = {
    "ATTN_BACKEND": "fa3",
    "COMPILE_MODE": "max-autotune-no-cudagraphs",
    "DEVICE_BATCH_SIZE": "72",
    "DOC_MASK": "1",
    "DOC_MASK_IMPL": "varlen",
    "DOC_MASK_MODE": "both",
    "MATRIX_LR": "0.04",
    "MUON_MOMENTUM_CONTINUOUS": "1",
    "NGRAM_TABLE_MULT": "64",
    "TOTAL_BATCH_SIZE": "147456",
    "WARMDOWN_RATIO": "0.85",
    "WINDOW_PATTERN": "TTTL",
}

NON_SCORED_DIAGNOSTIC_PREREQUISITES: dict[str, Any] = {
    "authority": (
        "governed diagnostic scheduler capability; never a direct SSH or "
        "ordinary scored-arm override"
    ),
    "scored_arm_env_exclusion": [
        "PACKER_BOUNDARY_VERIFY_BATCHES",
        "PACKER_DIAGNOSTIC_HASHES",
        "PACKER_SIDECAR_ACTIVATE_STEP",
    ],
    "fixed_steps": 250,
    "exact_equivalence": {
        "batches": 1000,
        "required_env": {
            "PACKER_BOUNDARY_VERIFY_BATCHES": "1000",
            "PACKER_DIAGNOSTIC_HASHES": "1",
            "PACKER_SIDECAR_ACTIVATE_STEP": "20",
        },
        "must_match": [
            "inputs",
            "targets",
            "epoch",
            "cu_seqlens",
            "max_seqlen",
            "boundary_count",
            "data_digest",
            "boundary_digest",
        ],
        "interior_bos_mismatch": "hard_kill",
    },
    "state_parity": {
        "phases": {
            "semantic_state_aa": {
                "steps": [0, 19],
                "consumed_boundaries": "scanner_vs_scanner",
                "disclosure": (
                    "the treatment still advances and safely discards the "
                    "sidecar lease, so this is not workload A/A"
                ),
            },
            "mechanism_ab": {
                "steps": [20, 39],
                "consumed_boundaries": "scanner_vs_sidecar",
            },
            "washout": {"steps": [40, 49]},
            "clean_profile": {
                "steps": [50, 249],
                "samples": 200,
                "blocks": 20,
                "block_size": 10,
                "bootstrap_seed": 19019063,
                "bootstrap_resamples": 10000,
            },
        },
        "must_match_bitwise": [
            "data_digest",
            "boundary_digest",
            "loss_digest",
            "model_state",
            "optimizer_state",
            "cpu_rng_state",
            "cuda_rng_state",
        ],
    },
    "runtime": {
        "hardware": "H200",
        "validate_pinned_int32_h2d_and_event_lifetime": True,
        "compiled_graph_count_after_warmup": 1,
        "compiled_site": "0:0",
        "recompile_count": 0,
        "cudagraph_recording_count": 0,
        "h200_systems_preflight_result_sha256": (
            "e5a96b00475fbcf1425eb72315ef0ef02e6138fc82cb84983529ef452f750127"
        ),
        "systems_preflight_is_not_throughput_or_bpb_evidence": True,
    },
    "single_placement_on_threshold_pass": "REQUIRE_COUNTERBALANCED_PROFILE",
    "single_placement_authorizes_endpoint": False,
    "scoring": "non_scored",
}


@dataclass(frozen=True)
class ChainRecords:
    control_intervention: InterventionRecord
    treatment_intervention: InterventionRecord
    mechanism: MechanismRecord
    hypothesis: HypothesisRecord
    idea: IdeaRecord
    proposal: ExperimentRecord
    gated_preview: ExperimentRecord

    def append_sequence(self) -> tuple[tuple[str, Any], ...]:
        """Return the dependency-safe persistent sequence.

        The approved preview is intentionally excluded. Persisting it requires
        the normal proposal freeze operation after a real ``check_gate`` call.
        """

        return (
            ("interventions", self.control_intervention),
            ("interventions", self.treatment_intervention),
            ("mechanisms", self.mechanism),
            ("hypotheses", self.hypothesis),
            ("idea_archive", self.idea),
            ("experiment_proposals", self.proposal),
        )


class _OverlayStore:
    """Read-only in-memory overlay for prospective gate validation."""

    def __init__(self, base: Any, additions: tuple[Any, ...]):
        self._base = base
        self._additions = additions

    def by_id(self) -> dict[str, Any]:
        records = self._base.by_id()
        for record in self._additions:
            identifier = str(record.registry_id)
            existing = records.get(identifier)
            if existing is not None:
                if canonical_json(existing.to_dict()) != canonical_json(
                    record.to_dict()
                ):
                    raise SchemaError(
                        f"overlay ID {identifier!r} conflicts with live content"
                    )
                continue
            records[identifier] = record
        return records

    def load(self) -> list[Any]:
        records = list(self._base.load())
        existing_ids = {str(record.registry_id) for record in records}
        for record in self._additions:
            if str(record.registry_id) not in existing_ids:
                records.append(record)
        return records


@contextmanager
def _prospective_registry_overlay(
    registry: ResearchRegistry, records: ChainRecords
) -> Iterator[None]:
    originals = {
        "interventions": registry.interventions,
        "mechanisms": registry.mechanisms,
        "hypotheses": registry.hypotheses,
        "idea_archive": registry.idea_archive,
        "experiment_proposals": registry.experiment_proposals,
    }
    registry.interventions = _OverlayStore(
        registry.interventions,
        (records.control_intervention, records.treatment_intervention),
    )
    registry.mechanisms = _OverlayStore(
        registry.mechanisms, (records.mechanism,)
    )
    registry.hypotheses = _OverlayStore(
        registry.hypotheses, (records.hypothesis,)
    )
    registry.idea_archive = _OverlayStore(
        registry.idea_archive, (records.idea,)
    )
    registry.experiment_proposals = _OverlayStore(
        registry.experiment_proposals, (records.proposal,)
    )
    try:
        yield
    finally:
        for name, store in originals.items():
            setattr(registry, name, store)


def _ordinary_env(*, sidecar: bool) -> dict[str, str]:
    return {
        **RECOVERED_RECIPE_ENV,
        "PACKER_DOC_BOUNDARIES": "1" if sidecar else "0",
    }


def _intervention(*, sidecar: bool) -> InterventionRecord:
    role = "treatment" if sidecar else "control"
    action = (
        "transport_packer_sourced_fa3_boundaries_under_recovered_recipe"
        if sidecar
        else "retain_scanner_sourced_fa3_boundaries_under_recovered_recipe"
    )
    return InterventionRecord(
        intervention_id=(
            TREATMENT_INTERVENTION_ID if sidecar else CONTROL_INTERVENTION_ID
        ),
        name=(
            "Recovered recipe with packer-sourced FA3 boundary sidecar"
            if sidecar
            else "Recovered recipe with scanner-sourced FA3 boundaries"
        ),
        version=1,
        action=action,
        target={
            "type": "systems_transport",
            "selector": (
                "best-fit packer to CausalSelfAttention FA3 varlen "
                "cumulative-offset path"
            ),
        },
        parameters={"env": _ordinary_env(sidecar=sidecar)},
        timing={"stage": "run_start_and_each_training_batch"},
        duration={"type": "full_run"},
        reversible=True,
        cost_tier="moderate",
        safety={
            "fast_fail_nonfinite_loss": True,
            "max_train_loss": 100,
            "same_token_order_required": True,
            "same_initialization_and_optimizer_required": True,
            "one_compiled_graph_required": True,
            "prelaunch_non_scored_diagnostics": (
                NON_SCORED_DIAGNOSTIC_PREREQUISITES
            ),
            "validation_policy_control": False,
        },
        implementation={
            "entrypoint": (
                "lib.py:make_dataloader/BoundaryLease; "
                "train.py:build_doc_masks_from_boundaries"
            ),
            "test": (
                "tests/test_packer_doc_boundaries.py and governed H200 "
                "packer-parity diagnostic"
            ),
        },
        status="unit_tested",
        description=(
            f"Paper-019 round-1 {role} under the complete recovered "
            "experiment-501 recipe. The ordinary scored environment excludes "
            "PACKER_BOUNDARY_VERIFY_BATCHES and PACKER_DIAGNOSTIC_HASHES; "
            "those flags are injected only by the non-scored governed "
            "diagnostic route."
        ),
        tags=(
            "paper019",
            "round1",
            "fa3",
            "boundary_sidecar",
            "systems_kernel",
            "track_explore",
            role,
        ),
    )


def records() -> ChainRecords:
    control = _intervention(sidecar=False)
    treatment = _intervention(sidecar=True)
    mechanism = MechanismRecord(
        mechanism_id=MECHANISM_ID,
        name="Packer-known boundaries remove redundant FA3 offset reconstruction",
        version=1,
        description=(
            "The best-fit packer already knows every document start while it "
            "constructs each packed row. Transporting those immutable int32 "
            "offsets through a pinned-memory/event-governed sidecar can replace "
            "the charged CUDA BOS scan and cumulative-offset reconstruction "
            "while preserving the exact FA3 varlen segmentation."
        ),
        origin_type="mixed",
        claim_ids=(CLAIM_ID,),
        observation_ids=(),
        causal_chain=(
            "The local best-fit packer identifies literal document starts while placing tokens.",
            "The scanner path reconstructs the same starts from each packed CUDA batch inside the charged step.",
            "The sidecar stages the packer-known int32 cumulative offsets and supplies them through the same FA3 varlen API.",
            "Exact inputs, targets, order, segments, compiled graph, parameters, and optimizer state are preserved while charged boundary work decreases.",
            "If isolated-H200 token throughput clears robust break-even, the fixed 300-second recipe completes more useful updates and lowers endpoint validation BPB.",
        ),
        assumptions=(
            "Packer starts and scanner-detected interior BOS positions are exactly equivalent over 1000 independent packed batches.",
            "Pinned source lifetime and CUDA producer/consumer event ordering remain valid through FA3 backward on the target H200.",
            "The sidecar-off generator allocates no sidecar buffers, streams, events, or leases and preserves the legacy triple.",
            "Variable boundary counts retain exactly one compiled graph after warmup.",
            "A 20-step semantic/state scanner/scanner prefix precedes a 20-step scanner/sidecar phase, a 10-step washout, and exactly 200 clean samples.",
            "A single physical placement cannot authorize an endpoint; a threshold pass requires a separately preregistered role-swapped profile.",
        ),
        competing_mechanism_ids=(),
        scope={
            "domain": "transformer_lm_pretraining",
            "decision_frame": SCOPE_ID,
            "hardware": "H200",
            "attention_backend": "fa3_varlen",
            "training_recipe": "complete_recovered_experiment_501_recipe",
            "local_extrapolation": (
                "packer-side generation, asynchronous transport, and local "
                "throughput/BPB effects are not established by the external claim"
            ),
        },
        observable_predictions=(),
        intervention_predictions=(
            {
                "intervention_id": TREATMENT_INTERVENTION_ID,
                "expected_change": (
                    "token-rate ratio point estimate >=1.054 and one-sided 95% "
                    "lower confidence bound >=1.041, conditional on exact parity"
                ),
            },
        ),
        status="proposed",
        tags=(
            "paper019",
            "round1",
            "fa3",
            "boundary_sidecar",
            "systems_kernel",
            "mixed_provenance",
        ),
        notes=(
            "External provenance is limited to "
            "clm_fa3_varlen_external_cuseqlens_contract: the official pinned "
            "FA3 code accepts and reuses CUDA int32 cumulative offsets. The "
            "packer-side generator, ring/event lifetime, exact local parity, "
            "throughput, and BPB effect are local extrapolations. The mixed "
            "MechanismRecord schema permits no observation_ids; static code "
            "inspection is not misrepresented as a run-derived ObservationRecord."
        ),
    )
    hypothesis = HypothesisRecord(
        hypothesis_id=HYPOTHESIS_ID,
        title=(
            "Exact packer-sourced FA3 boundaries improve the recovered recipe's "
            "five-minute endpoint"
        ),
        version=1,
        mechanism_ids=(MECHANISM_ID,),
        # Static run-start comparisons omit observables and contexts under
        # docs/AGENT_PROTOCOL.md; the exact decision frame lives in the
        # experiment's frozen data_policy/search_policy instead.
        context_ids=(),
        observable_predictions=(),
        intervention={
            "intervention_id": TREATMENT_INTERVENTION_ID,
            "version": 1,
            "parameters": {"env": _ordinary_env(sidecar=True)},
        },
        trigger={
            "type": "run_start",
            "observable_id": "",
            "condition": {
                "operator": "all_non_scored_diagnostic_prerequisites_pass"
            },
        },
        outcome={"outcome_id": OUTCOME_ID, "version": 1},
        prediction=(
            "Conditional on exactness, phased state parity, compiler identity, "
            "and H200 event-lifetime prerequisites, one placement either kills "
            "the sidecar or reaches point ratio >=1.054 and one-sided 95% LCB "
            ">=1.041, which requires a separately frozen role-swapped profile. "
            "Only a counterbalanced pass may make the ordinary endpoint "
            "proposal eligible for the normal typed gate."
        ),
        expected_effect={
            "direction": "decrease",
            "estimand": (
                "paired_endpoint_val_bpb_sidecar_on_minus_off_under_identical_"
                "recovered_recipe"
            ),
            "latency_steps": 0,
        },
        controls=(
            {
                "control_id": "recovered_recipe_scanner_boundaries",
                "kind": "no_intervention",
                "description": (
                    "The same recovered experiment-501 recipe, source, seed, "
                    "data order, FA3 varlen segmentation, and ordinary scored "
                    "environment with PACKER_DOC_BOUNDARIES=0."
                ),
                "matching": {
                    "same_checkpoint": True,
                    "same_data_order": True,
                },
            },
        ),
        falsification={
            "minimum_seeds": 10,
            "decision_rule": (
                "Prelaunch diagnostics are intersection gates. A single "
                "placement below either timing threshold kills the round; a "
                "pass only requests a separately preregistered role-swapped "
                "profile. An ordinary endpoint remains ineligible until the "
                "counterbalanced profile passes. Full support then requires "
                "the preregistered 1->3->6->10 BPB funnel."
            ),
            "failure_condition": (
                "Kill this mechanism before endpoint scoring on any 1000-batch "
                "data/boundary/digest or interior-BOS mismatch, extra compiled "
                "graph, phased state mismatch, invalid H200 event lifetime, "
                "throughput point ratio <1.054, throughput LCB <1.041, or "
                "integrity failure. A valid stage-1 endpoint above -0.002396 "
                "stops the funnel."
            ),
        },
        estimated_cost={
            "gpu_hours": 1.6667,
            "currency_cost": 0.0,
            "cost_tier": "moderate",
            "basis": (
                "Full ten-pair funnel: twenty one-H200 arms with a frozen "
                "300-second charged training budget; non-scored diagnostics "
                "are separately governed and excluded."
            ),
        },
        status="proposed",
        created_at=CREATED_AT,
        tags=(
            "paper019",
            "round1",
            "fa3",
            "boundary_sidecar",
            "systems_kernel",
            "track_explore",
        ),
        notes=(
            "The 0.927183 historical row is provenance-only and is neither a "
            "global nor concurrent control. The within-experiment comparator "
            "is a fresh sidecar-off execution of the recovered recipe. An "
            "accepted mechanism becomes the next cumulative recipe comparator; "
            "a rejection leaves that comparator unchanged."
        ),
    )
    critic_vector = "/".join(str(value) for value in PAPER_CRITIC_RATING.values())
    critic_sum = sum(PAPER_CRITIC_RATING.values())
    live_critic_vector = "/".join(
        str(value) for value in IMPLEMENTATION_AWARE_CRITIC_RATING.values()
    )
    live_critic_sum = sum(IMPLEMENTATION_AWARE_CRITIC_RATING.values())
    idea = IdeaRecord(
        idea_id=IDEA_ID,
        title="Transport packer-known boundaries directly to FA3 varlen attention",
        version=1,
        summary=(
            "Remove redundant charged BOS scanning and cumulative-offset "
            "construction by transporting exact packer-known boundaries through "
            "a fail-closed pinned int32 sidecar, without changing model inputs, "
            "targets, document segments, data order, or optimizer updates."
        ),
        experimental_plan=(
            "First run the fixed 250-step non-scored diagnostic: 1000-batch "
            "pre-clock exactness, 20 semantic/state A/A steps, 20 A/B steps, "
            "10 washout steps, exactly 200 clean timing samples, one graph, and "
            "H200 event-lifetime checks. A single-placement threshold pass "
            "requires a separately preregistered role-swapped profile; only a "
            "counterbalanced pass can make seed 63 eligible for the ordinary "
            "1->3->6->10 paired walltime funnel."
        ),
        direction="systems_kernel",
        subsystem="fa3_boundary_sidecar_transport",
        parent_idea_ids=(),
        scores={
            "interestingness": {
                "score": 8,
                "rationale": (
                    "It tests a semantically exact systems bottleneck whose "
                    "measured speed can cross the five-minute BPB decision floor."
                ),
            },
            "novelty": {
                "score": 8,
                "rationale": (
                    "The external source defines the varlen API, but does not "
                    "generate or asynchronously transport boundaries from this "
                    "best-fit packer."
                ),
            },
            "feasibility": {
                "score": 8,
                "rationale": (
                    "The local implementation is bounded, but H200 event "
                    "lifetime and robust throughput remain hard prerequisites."
                ),
            },
        },
        novelty_check={
            "provider": "literature_registry",
            "status": "passed",
            "query_rounds": [
                {
                    "query": (
                        "FlashAttention-3 packed varlen cu_seqlens packer "
                        "document boundary sidecar asynchronous training"
                    ),
                    "result_paper_ids": [PAPER_ID],
                    "assessment": (
                        "The NeurIPS paper and pinned official code establish "
                        "Hopper FA3 and the external cumulative-offset API, not "
                        "the OPHIS packer-side sidecar or its efficacy."
                    ),
                },
                {
                    "query": (
                        "best-fit language-model document packing reuse known "
                        "boundaries CUDA event ring exact optimizer parity"
                    ),
                    "result_paper_ids": [PAPER_ID],
                    "assessment": (
                        "No registered source tests the exact local combination "
                        "of packer-known offsets, a three-slot event-governed "
                        "transport ring, and state-identical five-minute scoring."
                    ),
                },
            ],
            "closest_paper_ids": [PAPER_ID],
            "evidence_ids": [LITERATURE_EVIDENCE_ID],
            "max_semantic_similarity": 0.45,
            "discard_threshold": 0.85,
            "assessment": (
                "Novelty applies only to the local systems composition and "
                "prospective evidence design. The FA3 varlen API itself is "
                "external prior art, and its evidence is not transferred to "
                "local speed or BPB."
            ),
        },
        status="selected",
        hypothesis_id=HYPOTHESIS_ID,
        created_at=CREATED_AT,
        notes=(
            "Paper-019 critic rating (N/P/V/I/R/F/X/C) = "
            f"{critic_vector} = {critic_sum}/40, i.e. novelty=4, "
            "provenance=4, validity=5, impact=3, reliability=3, "
            "feasibility=4, falsifiability=5, coherence=5. The later "
            "implementation-aware independent critic rating is "
            f"{live_critic_vector} = {live_critic_sum}/40, with impact=2, "
            "reliability=2, and feasibility=3 because the effect is unmeasured "
            "and physical placement is a confound. This corrects the "
            "non-canonical 3/4/5/4/4/5/5/3 vector in the registration prompt; "
            "paper 019 remains historical authority and paper 019a governs the "
            "live execution risk. Rank 1 is operational "
            "because it validates a dependency needed before later mechanisms, "
            "not because its revised 30/40 is the largest portfolio score."
        ),
    )
    proposal = ExperimentRecord(
        experiment_id=EXPERIMENT_ID,
        title=(
            "Paper-019 round 1: recovered-recipe FA3 boundary sidecar, "
            "one-pair screen"
        ),
        version=1,
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis_fingerprint=hypothesis.fingerprint,
        stage="discovery",
        status="planned",
        arms=(
            {
                "arm_id": "scanner_control",
                "role": "control",
                "description": (
                    "Complete recovered experiment-501 recipe with "
                    "PACKER_DOC_BOUNDARIES=0."
                ),
                "intervention_id": CONTROL_INTERVENTION_ID,
                "trigger": {"type": "run_start"},
                "control_id": "",
            },
            {
                "arm_id": "sidecar_treatment",
                "role": "treatment",
                "description": (
                    "Byte-identical ordinary recipe environment except "
                    "PACKER_DOC_BOUNDARIES=1."
                ),
                "intervention_id": TREATMENT_INTERVENTION_ID,
                "trigger": {"type": "run_start"},
                "control_id": "scanner_control",
            },
        ),
        seeds=(63,),
        checkpoint={
            "matching": "same_checkpoint_per_seed",
            "source": "random_init",
        },
        randomization={
            "unit": "seed_and_physical_gpu_role",
            "method": (
                "same-seed concurrent pair; freeze role assignment only after "
                "two clean UUID/PID inventories; retain one-to-two free GPUs; "
                "wait rather than remap after freeze"
            ),
        },
        analysis_plan={
            "primary_estimand": (
                "paired_endpoint_val_bpb_sidecar_on_minus_off_under_identical_"
                "recovered_recipe"
            ),
            "outcome_id": OUTCOME_ID,
            "baseline_covariates": [
                "training_step",
                "train_loss",
                "learning_rate",
                "total_tokens",
                "charged_training_seconds",
                "gpu_uuid",
            ],
            "uncertainty_method": (
                "single_pair_endpoint_screen_after_non_scored_block_bootstrap_"
                "throughput_gate"
            ),
            "multiplicity": {
                "family_id": "fam_paper019_round1_fa3_boundary_sidecar",
                "method": "none_single_preregistered_mechanism",
            },
            "noise_model": {
                "effective_sigma": EFFECTIVE_SIGMA,
                "minimum_effect": MINIMUM_EFFECT,
                "source": (
                    "research/setup/reconciliation.json v30 "
                    "walltime_5min_h200 operational baseline, n=10"
                ),
            },
            "prelaunch_non_scored_diagnostics": (
                NON_SCORED_DIAGNOSTIC_PREREQUISITES
            ),
            "throughput_gate": {
                "metric": "treatment_tokens_per_second_over_control",
                "point_ratio_minimum": THROUGHPUT_POINT_RATIO,
                "one_sided_95pct_lcb_minimum": THROUGHPUT_LCB_RATIO,
                "fixed_steps": 250,
                "clean_profile_steps": [50, 249],
                "clean_samples_exact": 200,
                "block_size": 10,
                "block_bootstrap_blocks_exact": 20,
                "bootstrap_seed": 19019063,
                "bootstrap_resamples": 10000,
                "integrity_failures_allowed": 0,
                "on_single_placement_pass": (
                    "REQUIRE_COUNTERBALANCED_PROFILE"
                ),
                "single_placement_authorizes_endpoint": False,
            },
        },
        budget={
            "estimated_gpu_hours": 0.1667,
            "estimated_currency_cost": 0.0,
            "hard_cap_currency_cost": 0.0,
            "basis": (
                "One same-seed pair; two one-H200 ordinary arms with a frozen "
                "300-second charged training budget. Diagnostics are "
                "non-scored and separately governed."
            ),
        },
        promotion_gate={
            "criteria": [
                "non-scored 1000-batch x/y/epoch/boundary/digest equality passes with no interior-BOS mismatch",
                "non-scored 20-step semantic/state A/A then 20-step A/B model, optimizer, loss, data, boundary, CPU-RNG, and CUDA-RNG hashes are bitwise equal",
                "H200 pinned-int32 producer/consumer event lifetime is valid; exactly one compile at 0:0, zero recompiles, and zero cudagraph recordings",
                "exactly 250 steps yield exactly 200 clean samples in 20 complete blocks of 10",
                "single-placement point ratio >=1.054 and one-sided 95% lower bound >=1.041 yields REQUIRE_COUNTERBALANCED_PROFILE, never endpoint authority",
                "a separately preregistered role-swapped profile passes the same combined point and lower-bound thresholds",
                "raw paired endpoint val_bpb sidecar-minus-scanner <=-0.002396",
                "treatment optimizer-step delta >0",
                "all exact config/source/data, physical-PID, completion, boundary, and co-tenancy checks pass",
            ],
            "on_pass": (
                "Only after the counterbalanced diagnostic passes, permit the "
                "ordinary seed-63 endpoint proposal to enter its normal typed "
                "gate. A valid seed-63 endpoint may freeze the three-pair child "
                "with seed prefix [63,64,65]; no adoption at n=1."
            ),
            "on_fail": (
                "Kill round 1 at the first failed prerequisite or valid "
                "stage-1 gate, attribute the failed link, leave the recovered "
                "recipe comparator unchanged, and advance the connected "
                "Paper-019 program under a new typed decision."
            ),
        },
        data_policy={
            "split": "train_shards_1_10",
            "proposal_loop_access": True,
            "scope_key": SCOPE_KEY,
        },
        created_at=CREATED_AT,
        search_policy={
            "version": 2,
            "challenge_id": SCOPE_ID,
            "decision_frame": SCOPE_ID,
            "direction": "systems_kernel",
            "subsystem": "fa3_boundary_sidecar_transport",
            "stage_pairs": 1,
            "parent_experiment_id": "",
            "predictions": {
                "delta_steps": {
                    "expected_delta": 108,
                    "unit": "optimizer_steps_per_300s",
                    "measurement": (
                        "final treatment num_steps minus paired scanner-control "
                        "num_steps"
                    ),
                    "rationale": (
                        "A 1.054 token-rate ratio at approximately 2000 "
                        "recovered-recipe steps implies about 108 additional "
                        "updates when batch shape is identical."
                    ),
                },
                "delta_quality_per_step": {
                    "expected_delta": 0.0,
                    "unit": "val_bpb_at_matched_optimizer_step",
                    "measurement": (
                        "secondary matched-step treatment minus control BPB; "
                        "never the stage-1 verdict"
                    ),
                    "rationale": (
                        "Exact data, segmentation, initialization, and update "
                        "parity predict no intrinsic quality-per-step change."
                    ),
                },
                "delta_endpoint": {
                    "expected_delta": EXPECTED_ENDPOINT_DELTA,
                    "unit": "val_bpb",
                    "measurement": (
                        "raw final sidecar-treatment minus scanner-control "
                        "endpoint under the 300-second clock"
                    ),
                    "rationale": (
                        "Conservative robust break-even prediction at the "
                        "preregistered two-sigma decision floor; the speed "
                        "profile must clear its stricter point and LCB gates."
                    ),
                },
            },
            "stopping": {
                "min_futility_pairs": 1,
                "promote_if_mean_endpoint_delta_lte": -MINIMUM_EFFECT,
                "stop_if_mean_endpoint_delta_gte": 0.0,
                "stop_if_mean_step_delta_lte": 0,
            },
            "portfolio": {
                "max_consecutive_failures": 3,
                "pivot_override": "",
            },
        },
        idea_id=IDEA_ID,
        frozen_at="",
        tags=(
            "paper019",
            "round1",
            "fa3",
            "boundary_sidecar",
            "systems_kernel",
            "track_explore",
            "stage1",
        ),
        notes=(
            "The within-experiment control is a fresh recovered-recipe arm, "
            "not the historical scalar 0.927183 and not a qualified global "
            "control. PACKER_BOUNDARY_VERIFY_BATCHES=1000 and "
            "PACKER_DIAGNOSTIC_HASHES=1 are non-scored diagnostic-only flags "
            "and are absent from both ordinary interventions."
        ),
    )
    gated_preview = replace(
        proposal,
        status="approved",
        frozen_at=FROZEN_AT,
    )
    return ChainRecords(
        control_intervention=control,
        treatment_intervention=treatment,
        mechanism=mechanism,
        hypothesis=hypothesis,
        idea=idea,
        proposal=proposal,
        gated_preview=gated_preview,
    )


def _source_contract() -> dict[str, Any]:
    required_markers = {
        "lib.py": (
            "class BoundaryLease:",
            "class _BoundaryRing:",
            "return_doc_boundaries=False",
            "PACKER_BOUNDARY_RING_REUSE_WITH_OUTSTANDING_LEASE",
        ),
        "train.py": (
            'os.environ.get("PACKER_DOC_BOUNDARIES", "0")',
            "def build_doc_masks_from_boundaries(",
            "def verify_packer_boundary_equivalence(",
            "PACKER_BOUNDARY_VERIFY_PASSED=1",
            "PACKER_DIAGNOSTIC_HASHES_VERIFIED=1",
            "MAX_STEPS=250",
        ),
        "tests/test_packer_doc_boundaries.py": (
            "PACKER_DOC_BOUNDARIES",
            "PACKER_BOUNDARY_VERIFY_BATCHES",
            "PACKER_DIAGNOSTIC_HASHES",
            "PACKER_SIDECAR_ACTIVATE_STEP",
            "BoundaryLease",
        ),
    }
    observed: dict[str, Any] = {}
    for relative, markers in required_markers.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            raise SchemaError(f"missing sidecar source contract file: {relative}")
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise SchemaError(
                f"{relative} is missing required sidecar markers: {missing}"
            )
        observed[relative] = {
            "required_markers": len(markers),
            "present": True,
        }
    return observed


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


def _validate_required_external_refs(registry: ResearchRegistry) -> None:
    papers = registry.papers.by_id()
    claims = registry.claims.by_id()
    evidence = registry.literature_evidence.by_id()
    outcomes = registry.outcomes.by_id()
    missing = {
        "paper": [] if PAPER_ID in papers else [PAPER_ID],
        "claim": [] if CLAIM_ID in claims else [CLAIM_ID],
        "literature_evidence": (
            [] if LITERATURE_EVIDENCE_ID in evidence else [LITERATURE_EVIDENCE_ID]
        ),
        "outcome": [] if OUTCOME_ID in outcomes else [OUTCOME_ID],
    }
    unresolved = {key: value for key, value in missing.items() if value}
    if unresolved:
        raise SchemaError(f"missing Paper-019 sidecar prerequisites: {unresolved}")
    if claims[CLAIM_ID].paper_id != PAPER_ID:
        raise SchemaError(f"{CLAIM_ID} does not bind {PAPER_ID}")
    terminal = resolve_terminal_evidence(evidence).terminals
    assessment = terminal.get(LITERATURE_EVIDENCE_ID)
    if assessment is None or CLAIM_ID not in assessment.claim_ids:
        raise SchemaError(
            f"{LITERATURE_EVIDENCE_ID} is not terminal support for {CLAIM_ID}"
        )
    if assessment.assessment["relation"] != "supports":
        raise SchemaError(
            f"{LITERATURE_EVIDENCE_ID} does not support its scoped API claim"
        )


def _validate_arm_separation(records_: ChainRecords) -> dict[str, Any]:
    control_env = dict(records_.control_intervention.parameters["env"])
    treatment_env = dict(records_.treatment_intervention.parameters["env"])
    differing = {
        key
        for key in set(control_env) | set(treatment_env)
        if control_env.get(key) != treatment_env.get(key)
    }
    if differing != {"PACKER_DOC_BOUNDARIES"}:
        raise SchemaError(
            "ordinary arm environments must differ only "
            f"PACKER_DOC_BOUNDARIES, found {sorted(differing)}"
        )
    forbidden = {
        "PACKER_BOUNDARY_VERIFY_BATCHES",
        "PACKER_DIAGNOSTIC_HASHES",
        "PACKER_SIDECAR_ACTIVATE_STEP",
    }
    leaked = forbidden & (set(control_env) | set(treatment_env))
    if leaked:
        raise SchemaError(
            f"diagnostic-only flags leaked into scored interventions: {sorted(leaked)}"
        )
    if control_env["PACKER_DOC_BOUNDARIES"] != "0":
        raise SchemaError("control must bind PACKER_DOC_BOUNDARIES=0")
    if treatment_env["PACKER_DOC_BOUNDARIES"] != "1":
        raise SchemaError("treatment must bind PACKER_DOC_BOUNDARIES=1")
    return {
        "ordinary_env_key_count": len(control_env),
        "only_difference": "PACKER_DOC_BOUNDARIES",
        "diagnostic_flags_absent": sorted(forbidden),
    }


def _prospective_gate_check(
    registry: ResearchRegistry, records_: ChainRecords
) -> dict[str, Any]:
    validation = registry.validate(check_generated_state=False).to_dict()
    if validation.get("warnings"):
        raise SchemaError(
            "pre-existing authoritative validation warnings: "
            f"{validation['warnings']}"
        )
    _validate_required_external_refs(registry)
    source = _source_contract()
    arm_separation = _validate_arm_separation(records_)

    if PAPER_CRITIC_RATING != {
        "novelty": 4,
        "provenance": 4,
        "validity": 5,
        "impact": 3,
        "reliability": 3,
        "feasibility": 4,
        "falsifiability": 5,
        "coherence": 5,
    } or sum(PAPER_CRITIC_RATING.values()) != 33:
        raise SchemaError("Paper-019 critic vector changed from 33/40 authority")
    if IMPLEMENTATION_AWARE_CRITIC_RATING != {
        "novelty": 4,
        "provenance": 4,
        "validity": 5,
        "impact": 2,
        "reliability": 2,
        "feasibility": 3,
        "falsifiability": 5,
        "coherence": 5,
    } or sum(IMPLEMENTATION_AWARE_CRITIC_RATING.values()) != 30:
        raise SchemaError(
            "Paper-019a implementation-aware critic vector changed from "
            "30/40 authority"
        )

    setup = registry._require_current_setup(SCOPE_ID)
    frame = setup.scope_for(SCOPE_ID)
    if dict(frame["scope_key"]) != SCOPE_KEY or frame["status"] != "passed":
        raise SchemaError(
            f"{SCOPE_ID} no longer matches the frozen prospective scope"
        )
    selected, event = registry.selected_challenge()
    if (
        selected["challenge_id"] != SCOPE_ID
        or selected["scope_id"] != SCOPE_ID
        or event.action != "activated"
    ):
        raise SchemaError(
            f"sticky active challenge is not activated {SCOPE_ID}"
        )
    hourly_report = registry.require_current_hourly_report()

    validate_search_policy(records_.proposal.search_policy, records_.proposal)
    registry._validate_analysis_gate(records_.proposal)
    if (
        canonical_json(records_.proposal.definition())
        != canonical_json(records_.gated_preview.definition())
        or records_.proposal.fingerprint != records_.gated_preview.fingerprint
    ):
        raise SchemaError(
            "planned proposal and approved preview do not have a byte-identical "
            "experiment definition"
        )

    current_ideas = registry.idea_archive.load()
    duplicates = [
        item
        for item in duplicate_ideas(current_ideas + [records_.idea])
        if records_.idea.idea_id in item[:2]
    ]
    if duplicates:
        raise SchemaError(
            f"prospective idea fails computed novelty dedup: {duplicates}"
        )

    with _prospective_registry_overlay(registry, records_):
        registry._require_literature_assessments(
            records_.proposal.experiment_id, records_.hypothesis
        )
        registry._validate_executable_gate(
            records_.proposal,
            records_.hypothesis,
            registry.observables.by_id(),
            registry.interventions.by_id(),
            registry.contexts.by_id(),
            registry.outcomes.by_id(),
        )
        search_report = registry._validate_search_gate(
            records_.proposal,
            scope_id=SCOPE_ID,
            frame=frame,
        )

    return {
        "authoritative_validation": validation,
        "source_contract": source,
        "arm_separation": arm_separation,
        "scope_key": dict(frame["scope_key"]),
        "setup_fingerprint": setup.fingerprint,
        "challenge_selection_fingerprint": event.fingerprint,
        "hourly_report": hourly_report,
        "search_policy": dict(search_report),
        "proposal_fingerprint": records_.proposal.fingerprint,
        "gated_preview_fingerprint": records_.gated_preview.fingerprint,
        "definition_byte_identical": True,
        "gate_equivalent_ready": True,
        "gated_registration_supported": False,
        "gated_registration_blocker": (
            "ResearchRegistry exposes no atomic proposal-to-gated promotion; "
            "persist the proposal, run check_gate, then use the normal freeze "
            "operation rather than bypassing the append-only gated ledger."
        ),
    }


def preflight_or_apply(*, apply: bool) -> dict[str, Any]:
    registry = ResearchRegistry(RESEARCH_ROOT)
    records_ = records()
    gate = _prospective_gate_check(registry, records_)

    gated_existing = registry.gated_experiments.by_id().get(EXPERIMENT_ID)
    if gated_existing is not None:
        if canonical_json(gated_existing.to_dict()) != canonical_json(
            records_.gated_preview.to_dict()
        ):
            raise SchemaError(
                f"gated ID {EXPERIMENT_ID!r} exists with different content"
            )
        proposal_action = "skip_promoted_exact"
    else:
        proposal_action = ""

    actions: list[dict[str, str]] = []
    for store_name, record in records_.append_sequence():
        if (
            store_name == "experiment_proposals"
            and proposal_action == "skip_promoted_exact"
        ):
            action = proposal_action
        else:
            action = _exact_action(
                getattr(registry, store_name), record, apply=apply
            )
        actions.append(
            {
                "registry": store_name,
                "id": str(record.registry_id),
                "action": action,
            }
        )

    if apply:
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
        "approved_preview": {
            "id": records_.gated_preview.experiment_id,
            "status": records_.gated_preview.status,
            "frozen_at": records_.gated_preview.frozen_at,
            "fingerprint": records_.gated_preview.fingerprint,
            "to_dict": records_.gated_preview.to_dict(),
            "persisted_by_this_helper": False,
        },
        "launch_authority": False,
        "sota_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "append exact missing records and the planned proposal only; "
            "default is read-only preflight"
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
                    f"sidecar-chain registration lock is held: {LOCK_PATH}"
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
