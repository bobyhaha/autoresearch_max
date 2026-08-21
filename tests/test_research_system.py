import json
import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vibeautoresearch.campaign import CampaignBatchRecord, campaign_slice_sha256
from vibeautoresearch.core import SchemaError
from vibeautoresearch.challenges import (
    ChallengeCatalog,
    ChallengeSelectionEvent,
    challenge_fingerprint,
)
from vibeautoresearch.experiments import ExperimentRecord, RunRecord
from vibeautoresearch.idea_archive import IdeaRecord
from vibeautoresearch.ideas import HypothesisRecord, MechanismRecord
from vibeautoresearch.knowledge import (
    BeliefRecord,
    ClaimRecord,
    EvidenceRecord,
    ObservationRecord,
    PaperRecord,
)
from vibeautoresearch.refinement import EvidenceUpdateRecord
from vibeautoresearch.registry import JsonlRegistry, ResearchRegistry
from vibeautoresearch.setup import SetupReconciliationRecord
from vibeautoresearch.toolkit import ContextRecord, InterventionRecord, ObservableRecord, OutcomeRecord


def create_empty_research(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    project_root = root.parent
    (project_root / "data_split.json").write_text(
        '{"train":[1,2,3],"test":[6542]}\n', encoding="utf-8"
    )
    (project_root / "lib.py").write_text("# frozen test harness\n", encoding="utf-8")
    (project_root / "baseline.md").write_text("test baseline\n", encoding="utf-8")
    (project_root / "train.py").write_text("# frozen test train\n", encoding="utf-8")
    data_hash = hashlib.sha256((project_root / "data_split.json").read_bytes()).hexdigest()
    lib_hash = hashlib.sha256((project_root / "lib.py").read_bytes()).hexdigest()
    train_hash = hashlib.sha256((project_root / "train.py").read_bytes()).hexdigest()
    manifest = {
        "schema_version": ResearchRegistry.SCHEMA_VERSION,
        "registries": ResearchRegistry.PATHS,
        "append_only": [
            "literature_evidence",
            "run_evidence",
            "beliefs",
            "idea_archive",
            "gated_experiments",
            "runs",
        ],
        "research_state": "knowledge/RESEARCH_STATE.md",
        "literature_synthesis": ResearchRegistry.LITERATURE_SYNTHESIS_PATH,
        "setup_reconciliation": ResearchRegistry.SETUP_PATH,
        "challenge_catalog": ResearchRegistry.CHALLENGE_CATALOG_PATH,
        "challenge_events": ResearchRegistry.CHALLENGE_EVENTS_PATH,
        "campaign_ledger": {
            "source_path": "campaign_log.jsonl",
            "coverage_start_exp_num": 1,
            "history_anchor_exp_num": 0,
            "history_anchor_sha256": campaign_slice_sha256([]),
        },
        "raw_artifacts": "experiments/artifacts",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (project_root / "campaign_log.jsonl").write_text("", encoding="utf-8")
    for relative in ResearchRegistry.PATHS.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    setup = SetupReconciliationRecord(
        version=1,
        status="pending",
        scope_key={
            "data_split_sha256": data_hash,
            "max_steps": 100,
            "stop_mode": "steps",
            "outcome_id": "out_test_validation_bpb",
        },
        frozen_files={"data_split.json": data_hash, "lib.py": lib_hash},
        reference_code={
            "local_path": "train.py",
            "local_sha256": train_hash,
            "upstream_url": "https://example.com/upstream.py",
            "upstream_sha256": "1" * 64,
            "comparison": "fork_with_differences",
            "diff": {"added": 1, "removed": 0, "hunks": 1},
        },
        baseline={
            "metric": "validation_bpb",
            "expected": 1.0,
            "observed": 1.0,
            "effective_sigma": 0.01,
            "tolerance_sigma": 2.0,
            "seeds": [1, 2, 3],
            "artifact_paths": ["baseline.md"],
        },
        reconciled_at="2026-07-16",
        reconciled_by="test",
    )
    setup_path = root / ResearchRegistry.SETUP_PATH
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_text(json.dumps(setup.to_dict()), encoding="utf-8")
    challenge_catalog = ChallengeCatalog(
        version=1,
        campaign_policy={
            "direction_round_limit": 5,
            "direction_cooldown_rounds": 2,
            "hourly_reports_required": False,
            "hourly_report_interval_minutes": 60,
        },
        challenges=(
            {
                "challenge_id": "fixed_steps_test",
                "order": 1,
                "name": "Fixture fixed-step challenge",
                "scope_id": "",
                "decision_frame": "default_steps",
                "aliases": ["steps"],
                "description": "Test fixture challenge.",
            },
        ),
    )
    catalog_path = root / ResearchRegistry.CHALLENGE_CATALOG_PATH
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(challenge_catalog.to_dict()), encoding="utf-8"
    )
    challenge_event = ChallengeSelectionEvent(
        event_id="challenge_selection_1",
        generation=1,
        version=1,
        action="activated",
        challenge_id="fixed_steps_test",
        catalog_fingerprint=challenge_catalog.fingerprint,
        challenge_fingerprint=challenge_fingerprint(
            challenge_catalog.resolve("fixed_steps_test")
        ),
        setup_fingerprint=setup.fingerprint,
        selected_at="2026-07-16",
        selected_by="test",
        reason="Select the fixture challenge.",
    )
    events_path = root / ResearchRegistry.CHALLENGE_EVENTS_PATH
    events_path.write_text(
        json.dumps(challenge_event.to_dict()) + "\n", encoding="utf-8"
    )
    synthesis_path = root / ResearchRegistry.LITERATURE_SYNTHESIS_PATH
    synthesis_path.parent.mkdir(parents=True, exist_ok=True)
    synthesis_path.write_text(
        "# Literature Synthesis\n\n"
        f"{ResearchRegistry.LITERATURE_SNAPSHOT_START}\n"
        f"{ResearchRegistry.LITERATURE_SNAPSHOT_END}\n",
        encoding="utf-8",
    )
    ResearchRegistry(root).write_literature_synthesis()


def current_scope(registry: ResearchRegistry) -> dict[str, object]:
    return dict(registry.setup_reconciliation().scope_key)


def pass_setup(registry: ResearchRegistry) -> None:
    setup = replace(registry.setup_reconciliation(), status="passed")
    (registry.root / ResearchRegistry.SETUP_PATH).write_text(
        json.dumps(setup.to_dict()), encoding="utf-8"
    )
    registry.append_challenge_event(
        action="activated",
        challenge_selector="fixed_steps_test",
        selected_at="2026-07-16T00:00:01Z",
        selected_by="test",
        reason="Rebind the fixture challenge to the passed setup.",
    )


def paper() -> PaperRecord:
    return PaperRecord(
        paper_id="pap_test_2026",
        title="Test paper",
        authors=("A. Researcher",),
        year=2026,
        venue={"name": "Test Venue", "peer_reviewed": True},
        urls={"primary": "https://example.com/paper"},
        retrieved_at="2026-07-16",
    )


def claim() -> ClaimRecord:
    return ClaimRecord(
        claim_id="clm_test_transition",
        paper_id="pap_test_2026",
        statement="A signal changes before a transition.",
        claim_type="temporal_association",
        scope={"model": "toy"},
        limitations=("Toy model only.",),
        locator="Section 3",
        extracted_at="2026-07-16",
    )


def literature_evidence() -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="evd_test_literature",
        source_type="literature",
        paper_ids=("pap_test_2026",),
        claim_ids=("clm_test_transition",),
        run_ids=(),
        experiment_id="",
        hypothesis_ids=(),
        facts={"reported_seeds": 3},
        analysis={"method": "paper_review", "code_ref": "manual"},
        trust={
            "design": "adequate",
            "replication": "weak",
            "scope_match": "adequate",
            "directness": "strong",
            "limitations": ["No independent replication."],
        },
        assessment={
            "relation": "supports",
            "strength": "moderate",
            "limitations": ["Single task."],
        },
        artifact_paths=(),
        created_at="2026-07-16",
        created_by="test",
    )


def belief() -> BeliefRecord:
    return BeliefRecord(
        belief_id="blf_test_transition",
        version=1,
        statement="The reported transition may generalize to this context.",
        status="plausible",
        subject_ids=("clm_test_transition",),
        evidence_ids=("evd_test_literature",),
        scope={"model": "toy"},
        rationale="The literature evidence is direct but not independently replicated.",
        limitations=("No internal run evidence yet.",),
        next_test="Run a preregistered pilot.",
        created_at="2026-07-16",
        created_by="test",
    )


def observable(status: str = "available") -> ObservableRecord:
    return ObservableRecord(
        observable_id="obs_test_signal",
        name="Test signal",
        version=1,
        sources=(
            {
                "name": "activation",
                "target": "model.layer_0.activation",
                "tensor_type": "activation",
                "axes": ["batch", "channel"],
                "context": "train",
            },
        ),
        operation="axis_reduction_pipeline",
        reductions=(
            {"axis": "channel", "op": "mean", "params": {}},
            {"axis": "batch", "op": "mean", "params": {}},
        ),
        output={"type": "scalar", "units": "unitless"},
        collection={
            "mode": "online",
            "cadence_steps": 25,
            "cost_tier": "moderate",
            "estimated_overhead_pct": 3.0,
            "profile_status": "measured",
        },
        causal_availability={
            "available_before_action": True,
            "uses_validation": False,
            "stage": "forward",
        },
        implementation={"entrypoint": "tests:signal", "test": "tests:test_signal"},
        status=status,
    )


def intervention(status: str = "available") -> InterventionRecord:
    return InterventionRecord(
        intervention_id="int_test_noop",
        name="No-op",
        version=1,
        action="no_op",
        target={"type": "training_process", "selector": "all"},
        parameters={},
        timing={"stage": "before_optimizer_step"},
        duration={"type": "one_shot"},
        reversible=True,
        cost_tier="light",
        safety={"must_preserve_update": True},
        implementation={"entrypoint": "tests:no_op", "test": "tests:test_no_op"},
        status=status,
    )


def context() -> ContextRecord:
    return ContextRecord(
        context_id="ctx_test_step",
        name="Training step",
        version=1,
        source={"type": "runtime_counter"},
        value={"type": "integer"},
        availability={"available_before_action": True},
        implementation={"entrypoint": "train:step", "test": "tests:test_step"},
        status="available",
    )


def outcome(policy_visible: bool = False) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_id="out_test_validation_bpb",
        name="Validation BPB",
        version=1,
        metric={"name": "validation_bpb", "units": "bits_per_byte"},
        direction="minimize",
        data_split="validation",
        aggregation={"type": "endpoint"},
        detector={},
        policy_visible=policy_visible,
        implementation={"entrypoint": "train:evaluate_bpb", "test": "tests:test_bpb"},
        status="available",
    )


def mechanism() -> MechanismRecord:
    return MechanismRecord(
        mechanism_id="mech_test_transition",
        name="Test transition",
        version=1,
        description="The state changes the signal and outcome.",
        origin_type="literature",
        claim_ids=("clm_test_transition",),
        observation_ids=(),
        causal_chain=("State changes.", "Signal changes.", "Outcome changes."),
        assumptions=("Signal is measured accurately.",),
        competing_mechanism_ids=(),
        scope={"model": "toy"},
        observable_predictions=(
            {"observable_id": "obs_test_signal", "expected_pattern": "increase"},
        ),
        intervention_predictions=(
            {"intervention_id": "int_test_noop", "expected_change": "none"},
        ),
        status="active",
    )


def hypothesis() -> HypothesisRecord:
    return HypothesisRecord(
        hypothesis_id="hyp_test_trigger",
        title="Test triggered action",
        version=1,
        mechanism_ids=("mech_test_transition",),
        context_ids=("ctx_test_step",),
        observable_predictions=(
            {
                "observable_id": "obs_test_signal",
                "version": 1,
                "expected_pattern": {"type": "greater_than", "threshold": 1.0},
                "window": {"type": "pre_intervention", "steps": 50},
            },
        ),
        intervention={"intervention_id": "int_test_noop", "version": 1, "parameters": {}},
        trigger={
            "type": "observable_threshold",
            "observable_id": "obs_test_signal",
            "condition": {"operator": "greater_than", "threshold": 1.0},
        },
        outcome={"outcome_id": "out_test_validation_bpb", "version": 1},
        prediction="The observable-conditioned action changes future BPB.",
        expected_effect={
            "direction": "decrease",
            "estimand": "trigger_by_action_interaction",
            "latency_steps": 100,
        },
        controls=(
            {
                "control_id": "no_action",
                "kind": "no_intervention",
                "description": "No action.",
                "matching": {"same_checkpoint": True, "same_data_order": True},
            },
        ),
        falsification={
            "minimum_seeds": 3,
            "decision_rule": "Compare paired effects.",
            "failure_condition": "No improvement over matched controls.",
        },
        estimated_cost={
            "gpu_hours": 1.0,
            "currency_cost": 5.0,
            "cost_tier": "moderate",
            "basis": "One-seed pilot estimate.",
        },
        status="proposed",
        created_at="2026-07-16",
    )


def idea(
    *,
    idea_id: str = "idea_test_signal",
    direction: str = "systems_kernel",
    subsystem: str = "test_signal",
) -> IdeaRecord:
    # Distinct ideas must carry distinct text; the cross-idea dedup floor
    # (duplicate_ideas) rejects a funded idea that is lexically at/above its own
    # discard threshold from an unrelated one. Real ideas differ in content, so
    # key the description off idea_id/direction rather than shared boilerplate.
    marker = idea_id.replace("idea_", "").replace("_", " ")
    return IdeaRecord(
        idea_id=idea_id,
        title=f"{marker} experiment",
        version=1,
        summary=f"{marker} intervention targeting {direction}",
        experimental_plan=f"{marker} {direction} {subsystem} matched pair staged funnel",
        direction=direction,
        subsystem=subsystem,
        parent_idea_ids=(),
        scores={
            "interestingness": {
                "score": 7,
                "rationale": "It tests a concrete systems bottleneck.",
            },
            "novelty": {
                "score": 6,
                "rationale": "The exact intervention is absent from the closest work.",
            },
            "feasibility": {
                "score": 9,
                "rationale": "The implementation and matched control are small.",
            },
        },
        novelty_check={
            "provider": "combined",
            "status": "passed",
            "query_rounds": [
                {
                    "query": "redundant per-step signal work language model training",
                    "result_paper_ids": ["pap_test_2026"],
                    "assessment": "The paper is related but does not test this intervention.",
                }
            ],
            "closest_paper_ids": ["pap_test_2026"],
            "evidence_ids": ["evd_test_literature"],
            "max_semantic_similarity": 0.2,
            "discard_threshold": 0.8,
            "assessment": "Novel enough to enter the experimental funnel.",
        },
        status="selected",
        hypothesis_id="hyp_test_trigger",
        created_at="2026-07-16",
    )


def search_policy(
    *,
    stage_pairs: int = 1,
    parent_experiment_id: str = "",
    subsystem: str = "test_signal",
    direction: str = "systems_kernel",
) -> dict[str, object]:
    return {
        "version": 2,
        "challenge_id": "fixed_steps_test",
        "decision_frame": "default_steps",
        "direction": direction,
        "subsystem": subsystem,
        "stage_pairs": stage_pairs,
        "parent_experiment_id": parent_experiment_id,
        "predictions": {
            "delta_steps": {
                "expected_delta": 10,
                "unit": "steps_per_budget",
                "measurement": "final num_steps treatment minus control",
                "rationale": "The intervention removes measured per-step work.",
            },
            "delta_quality_per_step": {
                "expected_delta": 0,
                "unit": "val_bpb",
                "measurement": "validation BPB at a matched optimizer step",
                "rationale": "The intervention should preserve the update rule.",
            },
            "delta_endpoint": {
                "expected_delta": -0.03,
                "unit": "val_bpb",
                "measurement": "final treatment BPB minus paired control BPB",
                "rationale": "More steps should improve the wall-clock endpoint.",
            },
        },
        "stopping": {
            "min_futility_pairs": 1 if stage_pairs == 1 else 3,
            "promote_if_mean_endpoint_delta_lte": -0.02,
            "stop_if_mean_endpoint_delta_gte": 0.002,
            "stop_if_mean_step_delta_lte": 0,
        },
        "portfolio": {
            "max_consecutive_failures": 2,
            "pivot_override": "",
        },
    }


def pilot_experiment(hyp: HypothesisRecord | None = None) -> ExperimentRecord:
    hyp = hyp or hypothesis()
    return ExperimentRecord(
        experiment_id="exp_test_pilot",
        title="Test pilot",
        version=1,
        hypothesis_id=hyp.hypothesis_id,
        hypothesis_fingerprint=hyp.fingerprint,
        idea_id="idea_test_signal",
        stage="discovery",
        status="approved",
        arms=(
            {
                "arm_id": "no_action",
                "role": "control",
                "description": "No intervention.",
                "intervention_id": "",
                "trigger": {},
                "control_id": "no_action",
            },
            {
                "arm_id": "treatment",
                "role": "treatment",
                "description": "Triggered action.",
                "intervention_id": "int_test_noop",
                "trigger": {"type": "observable_threshold"},
                "control_id": "",
            },
        ),
        seeds=(1,),
        checkpoint={"matching": "same_checkpoint_per_seed", "source": "checkpoint:test"},
        randomization={"unit": "branch", "method": "seeded"},
        analysis_plan={
            "primary_estimand": "trigger_by_action_interaction",
            "outcome_id": "out_test_validation_bpb",
            "baseline_covariates": ["training_step", "train_loss", "learning_rate"],
            "uncertainty_method": "paired_bootstrap",
            "multiplicity": {"family_id": "test_family", "method": "none_single_test"},
            "noise_model": {
                "effective_sigma": 0.01,
                "minimum_effect": 0.02,
                "source": "Three-seed setup baseline.",
            },
        },
        budget={
            "estimated_gpu_hours": 1.0,
            "estimated_currency_cost": 5.0,
            "hard_cap_currency_cost": 10.0,
            "basis": "Measured short-run throughput.",
        },
        promotion_gate={
            "criteria": ["No numerical failure.", "Cost remains below cap."],
            "on_pass": "Design discovery experiment.",
            "on_fail": "Stop and record evidence.",
        },
        data_policy={"split": "discovery", "proposal_loop_access": True},
        created_at="2026-07-16",
        search_policy=search_policy(),
        frozen_at="2026-07-16",
    )


def qualification_experiment(
    root: Path,
    *,
    experiment_id: str = "exp_test_qualification",
    status: str = "approved",
) -> ExperimentRecord:
    anchors = []
    anchor_dir = root / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(47, 51):
        path = anchor_dir / f"seed{seed}.log"
        path.write_text(f"seed={seed} val_bpb=0.93 tokens=100000\n", encoding="utf-8")
        anchors.append(
            {
                "seed": seed,
                "val_bpb": 0.93,
                "total_tokens": 100_000,
                "artifact_path": f"anchors/seed{seed}.log",
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    policy = search_policy()
    policy["stage_pairs"] = 4
    policy["parent_experiment_id"] = ""
    policy["stopping"]["min_futility_pairs"] = 4
    policy["qualification"] = {
        "kind": "historical_equivalence_bridge",
        "equivalence_margin": 0.02,
        "superiority_minimum_effect": 0.02,
        "required_helping_pairs": 4,
        "token_ratio_lower": 0.98,
        "token_ratio_upper": 1.02,
        "historical_anchors": anchors,
    }
    return replace(
        pilot_experiment(),
        experiment_id=experiment_id,
        title="Standalone historical-control qualification",
        stage="validation",
        status=status,
        seeds=(47, 48, 49, 50),
        search_policy=policy,
        created_at="2026-07-16T00:10:00Z",
        frozen_at="2026-07-16T00:10:00Z" if status != "planned" else "",
    )


def run(exp: ExperimentRecord | None = None) -> RunRecord:
    exp = exp or pilot_experiment()
    return RunRecord(
        run_id="run_test_seed1_treatment",
        experiment_id=exp.experiment_id,
        hypothesis_id=exp.hypothesis_id,
        arm_id="treatment",
        seed=1,
        role="treatment",
        status="complete",
        git_commit="abc123",
        git_dirty=False,
        config_hash="cfg123",
        spec_fingerprints={
            "hypothesis": hypothesis().fingerprint,
            "experiment": exp.fingerprint,
            "outcome": outcome().fingerprint,
        },
        tracker={
            "provider": "test",
            "run_id": "remote-1",
            "gpu_sampling_verified": True,
            "challenge_id": "fixed_steps_test",
            "challenge_definition_fingerprint": "a" * 16,
            "challenge_selection_fingerprint": "b" * 16,
        },
        artifact_paths={},
        intervention_events_path="tracker://remote-1/events",
        outcome_values={
            "out_test_validation_bpb": 1.25,
            "val_bpb": 1.25,
            "num_steps": 100,
        },
        actual_cost={"gpu_hours": 1.0, "currency_cost": 5.0, "observable_overhead_pct": 3.0},
        started_at="2026-07-16T00:00:00Z",
        ended_at="2026-07-16T01:00:00Z",
        tags=(
            "test",
            "bound_run",
            "gpu_sampling_verified",
            "challenge_fixed_steps_test",
        ),
    )


def add_core_graph(registry: ResearchRegistry, *, executable: bool = True, gated: bool = False) -> None:
    registry.papers.add(paper())
    registry.claims.add(claim())
    registry.literature_evidence.add(literature_evidence())
    registry.observables.add(observable("available" if executable else "proposed"))
    registry.interventions.add(intervention("available" if executable else "proposed"))
    registry.contexts.add(context())
    registry.outcomes.add(outcome())
    registry.mechanisms.add(mechanism())
    hyp = hypothesis()
    registry.hypotheses.add(hyp)
    registry.idea_archive.add(idea())
    pass_setup(registry)
    if gated:
        registry.gated_experiments.add(pilot_experiment(hyp))


class SchemaUnitTest(unittest.TestCase):
    def test_internal_mechanism_requires_typed_campaign_or_observation_provenance(self):
        with self.assertRaisesRegex(
            SchemaError, "requires observation_ids or campaign_batch_ids"
        ):
            replace(
                mechanism(),
                origin_type="internal_experiment",
                claim_ids=(),
                observation_ids=(),
                campaign_batch_ids=(),
            )

    def test_quarantined_campaign_evidence_is_weak_and_never_conclusive(self):
        evidence = EvidenceRecord(
            evidence_id="evd_test_quarantined_campaign",
            source_type="quarantined_campaign",
            paper_ids=(),
            claim_ids=(),
            run_ids=(),
            experiment_id="",
            hypothesis_ids=("hyp_test_trigger",),
            facts={"reported_delta": -0.01},
            analysis={"method": "retrospective", "code_ref": "campaign_log.jsonl"},
            trust={
                "design": "weak",
                "replication": "weak",
                "scope_match": "adequate",
                "directness": "weak",
                "limitations": ["No RunRecords."],
            },
            assessment={
                "relation": "supports",
                "strength": "weak",
                "limitations": ["Quarantined source."],
            },
            artifact_paths=("campaign_log.jsonl",),
            created_at="2026-07-29T00:00:00Z",
            created_by="test",
            campaign_batch_ids=("cmp_test_direct",),
        )
        self.assertEqual(
            ResearchRegistry._conclusively_tested_hypothesis_ids(
                {evidence.evidence_id: evidence}
            ),
            set(),
        )
        with self.assertRaisesRegex(SchemaError, "cannot exceed weak"):
            replace(
                evidence,
                assessment={
                    "relation": "supports",
                    "strength": "strong",
                    "limitations": ["Invalid promotion."],
                },
            )

    def test_axis_reduction_must_reduce_every_axis(self):
        with self.assertRaises(SchemaError):
            ObservableRecord(
                observable_id="obs_incomplete",
                name="Incomplete",
                version=1,
                sources=(
                    {
                        "name": "x",
                        "target": "model.x",
                        "tensor_type": "activation",
                        "axes": ["batch", "channel"],
                        "context": "train",
                    },
                ),
                operation="axis_reduction_pipeline",
                reductions=({"axis": "channel", "op": "mean", "params": {}},),
                output={"type": "scalar", "units": "unitless"},
                collection={
                    "mode": "online",
                    "cadence_steps": 1,
                    "cost_tier": "light",
                    "estimated_overhead_pct": 1,
                    "profile_status": "estimated",
                },
                causal_availability={
                    "available_before_action": True,
                    "uses_validation": False,
                    "stage": "forward",
                },
                implementation={"entrypoint": "tests:x", "test": "tests:x"},
                status="available",
            )

    def test_validation_outcome_cannot_control_training(self):
        with self.assertRaises(SchemaError):
            outcome(policy_visible=True)

    def test_experiment_budget_is_hard_cap(self):
        base = pilot_experiment().to_dict()
        base["budget"]["estimated_currency_cost"] = 11.0
        base["budget"]["hard_cap_currency_cost"] = 10.0
        base["fingerprint"] = ExperimentRecord.from_dict(pilot_experiment().to_dict()).fingerprint
        with self.assertRaises(SchemaError):
            ExperimentRecord.from_dict(base)

    def test_versioned_record_detects_definition_edit(self):
        payload = hypothesis().to_dict()
        payload["prediction"] = "Changed after seeing data."
        with self.assertRaises(SchemaError):
            HypothesisRecord.from_dict(payload)

    def test_static_hypothesis_does_not_require_dummy_observable_or_context(self):
        record = replace(
            hypothesis(),
            context_ids=(),
            observable_predictions=(),
            trigger={"type": "run_start", "observable_id": "", "condition": {}},
        )

        self.assertEqual(record.observable_ids, ())
        self.assertEqual(record.context_ids, ())

    def test_evidence_assessment_is_immutable(self):
        payload = literature_evidence().to_dict()
        payload["assessment"]["strength"] = "strong"
        with self.assertRaises(SchemaError):
            EvidenceRecord.from_dict(payload)

    def test_observation_and_run_facts_are_immutable(self):
        observation = ObservationRecord(
            observation_id="evd_test_observation",
            run_ids=("run_test_seed1_treatment",),
            statement="The signal increased before the endpoint.",
            status="exploratory",
            possible_confounds=("Training step.",),
            followup_required=True,
            created_at="2026-07-16",
        )
        observation_payload = observation.to_dict()
        observation_payload["statement"] = "Retrofitted interpretation."
        with self.assertRaises(SchemaError):
            ObservationRecord.from_dict(observation_payload)

        run_payload = run().to_dict()
        run_payload["seed"] = 2
        with self.assertRaises(SchemaError):
            RunRecord.from_dict(run_payload)


class RegistryIntegrationTest(unittest.TestCase):
    def test_precommit_validates_the_staged_tree_not_the_worktree(self):
        source = (
            Path(__file__).resolve().parents[1] / ".githooks" / "pre-commit"
        ).read_text(encoding="utf-8")
        self.assertIn("checkout-index --all --force", source)
        self.assertIn("-m vibeautoresearch validate", source)
        self.assertNotIn("git diff --cached -U0 -- campaign_log.jsonl", source)

    def test_campaign_log_rows_require_exact_typed_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            row = {
                "exp_num": 1,
                "exp": "direct_test",
                "phase": "block_test",
                "status": "keep",
            }
            (root.parent / "campaign_log.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            registry = ResearchRegistry(root)
            with self.assertRaisesRegex(SchemaError, "lack a registered or quarantined"):
                registry.validate(check_generated_state=False)

            batch = CampaignBatchRecord(
                campaign_batch_id="cmp_test_direct",
                source_path="campaign_log.jsonl",
                phase="block_test",
                first_exp_num=1,
                last_exp_num=1,
                entry_count=1,
                source_sha256=campaign_slice_sha256([row]),
                disposition="quarantined",
                experiment_ids=(),
                run_ids=(),
                evidence_ids=(),
                provenance={
                    "launch_method": "direct_ssh",
                    "gate_binding": False,
                    "run_records_complete": False,
                    "artifact_completeness": "none",
                    "limitations": ["No gated launch or immutable run artifact."],
                },
                summary="A direct test row with explicitly weak provenance.",
                recorded_at="2026-07-29T00:00:00Z",
                recorded_by="test",
            )
            registry.campaign_batches.add(batch)
            self.assertEqual(
                registry.validate(check_generated_state=False).counts[
                    "campaign_batches"
                ],
                1,
            )

            row["status"] = "discard"
            (root.parent / "campaign_log.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(SchemaError, "source digest mismatch"):
                registry.validate(check_generated_state=False)

    def test_quarantined_campaign_batch_cannot_claim_formal_bindings(self):
        with self.assertRaisesRegex(
            SchemaError, "cannot imply formal registry bindings"
        ):
            CampaignBatchRecord(
                campaign_batch_id="cmp_test_invalid",
                source_path="campaign_log.jsonl",
                phase="block_test",
                first_exp_num=1,
                last_exp_num=1,
                entry_count=1,
                source_sha256="a" * 64,
                disposition="quarantined",
                experiment_ids=("exp_test_pilot",),
                run_ids=(),
                evidence_ids=(),
                provenance={
                    "launch_method": "direct_ssh",
                    "gate_binding": False,
                    "run_records_complete": False,
                    "artifact_completeness": "partial",
                    "limitations": ["No formal launch provenance."],
                },
                summary="Invalid implied binding.",
                recorded_at="2026-07-29T00:00:00Z",
                recorded_by="test",
            )

    def test_concurrent_registry_appends_are_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = JsonlRegistry(Path(directory) / "papers.jsonl", PaperRecord)
            records = [
                replace(
                    paper(),
                    paper_id=f"pap_concurrent_{index}",
                    title=f"Concurrent paper {index}",
                )
                for index in range(12)
            ]

            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(registry.add, records))

            self.assertEqual(
                {record.paper_id for record in registry.load()},
                {record.paper_id for record in records},
            )

    def test_empty_repository_is_valid_and_state_is_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            report = registry.validate()
            self.assertEqual(report.counts["runs"], 0)
            self.assertTrue(report.warnings)
            registry.write_state()
            self.assertEqual(
                registry.validate().warnings,
                (
                    "default setup reconciliation is pending; "
                    "default-frame execution is blocked",
                ),
            )
            snapshot = registry.write_snapshot("2026_07_16")
            self.assertTrue(snapshot.exists())
            with self.assertRaises(SchemaError):
                registry.write_snapshot("2026_07_16")

    def test_hourly_paper_is_data_driven_and_blocks_only_when_overdue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            gated = replace(
                pilot_experiment(),
                data_policy={
                    "split": "discovery",
                    "proposal_loop_access": True,
                    "scope_key": current_scope(registry),
                },
            )
            registry.gated_experiments.add(gated)
            catalog = replace(
                registry.challenge_catalog(),
                campaign_policy={
                    "direction_round_limit": 5,
                    "direction_cooldown_rounds": 2,
                    "hourly_reports_required": True,
                    "hourly_report_interval_minutes": 60,
                },
            )
            (root / ResearchRegistry.CHALLENGE_CATALOG_PATH).write_text(
                json.dumps(catalog.to_dict()), encoding="utf-8"
            )
            now = datetime.now(timezone.utc)
            registry.append_challenge_event(
                action="activated",
                challenge_selector="fixed_steps_test",
                selected_at=(now - timedelta(hours=2)).isoformat(),
                selected_by="test",
                reason="Exercise hourly reporting enforcement.",
            )

            self.assertTrue(registry.hourly_report_status(now=now)["due"])
            with self.assertRaisesRegex(SchemaError, "hourly research paper is overdue"):
                registry.authorize_run(
                    gated.experiment_id,
                    "treatment",
                    1,
                    100,
                )

            report, paper_path = registry.publish_hourly_report(
                {
                    "created_by": "test",
                    "title": "Hourly test campaign paper",
                    "abstract": "No GPU evidence was created; this paper records the boundary.",
                    "methods": "The registry and active challenge were inspected.",
                    "directions_explored": [],
                    "idea_ids": [],
                    "experiment_ids": [],
                    "run_ids": [],
                    "findings": [],
                    "negative_results": ["No experimental result was claimed."],
                    "limitations": ["This was an operational test only."],
                    "decisions": ["Do not infer an effect from this report."],
                    "next_hour_plan": [
                        {
                            "direction": "architecture",
                            "question": "Which architecture idea merits a novelty search?",
                            "why_now": "The archive is empty.",
                            "stop_condition": "Stop if literature search finds a close prior.",
                        }
                    ],
                },
                now=now,
            )

            self.assertTrue(paper_path.exists())
            self.assertEqual(report.challenge_id, "fixed_steps_test")
            self.assertFalse(
                registry.hourly_report_status(now=now + timedelta(minutes=30))["due"]
            )
            self.assertTrue(
                registry.authorize_run(
                    gated.experiment_id,
                    "treatment",
                    1,
                    100,
                )["authorized"]
            )
            self.assertEqual(
                registry.validate(check_generated_state=False).counts["hourly_reports"],
                1,
            )

    def test_complete_cross_reference_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry, gated=True)
            registry.beliefs.add(belief())
            registry.runs.add(run())
            report = registry.validate(check_generated_state=False)
            self.assertEqual(report.counts["hypotheses"], 1)
            self.assertEqual(report.counts["beliefs"], 1)
            self.assertEqual(report.counts["runs"], 1)

    def test_literature_synthesis_snapshot_is_generated_and_drift_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            source = registry.literature_synthesis_path.read_text(encoding="utf-8")
            self.assertIn("Papers: **0**", source)
            self.assertIn("Atomic claims: **0**", source)
            self.assertIn("Scope key:", source)

            registry.literature_synthesis_path.write_text(
                source.replace("Papers: **0**", "Papers: **999**"),
                encoding="utf-8",
            )
            self.assertTrue(
                any(
                    "LITERATURE_SYNTHESIS.md snapshot is stale" in warning
                    for warning in registry.validate().warnings
                )
            )
            self.assertIn(
                "stale_literature_synthesis",
                {issue["code"] for issue in registry.audit().issues},
            )
            registry.write_literature_synthesis()
            self.assertFalse(
                any(
                    "LITERATURE_SYNTHESIS.md" in warning
                    for warning in registry.validate().warnings
                )
            )

    def test_gated_experiment_rejects_unimplemented_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry, executable=False, gated=True)
            with self.assertRaises(SchemaError):
                registry.validate(check_generated_state=False)

    def test_completed_run_appears_in_audit_until_evidence_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry, gated=True)
            registry.runs.add(run())
            codes = {item["code"] for item in registry.audit().issues}
            self.assertIn("completed_run_without_evidence", codes)

    def test_check_gate_is_read_only_and_reports_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            proposal = replace(
                pilot_experiment(),
                status="planned",
                frozen_at="",
                data_policy={
                    "split": "discovery",
                    "proposal_loop_access": True,
                    "scope_key": current_scope(registry),
                },
                promotion_gate={
                    "criteria": ["Implementation is numerically safe."],
                    "on_pass": "Design a powered discovery experiment.",
                    "on_fail": "Stop and record the implementation failure.",
                    "verdict": "implementation_only",
                },
            )
            registry.experiment_proposals.add(proposal)
            result = registry.check_gate(proposal.experiment_id)
            self.assertTrue(result["ready_to_freeze"])
            self.assertEqual(result["hard_cap_currency_cost"], 10.0)
            self.assertEqual(len(registry.experiment_proposals.load()), 1)
            self.assertEqual(len(registry.gated_experiments.load()), 0)

    def test_check_gate_rejects_a_new_proposal_without_search_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            proposal = replace(
                pilot_experiment(),
                status="planned",
                stage="pilot",
                frozen_at="",
                search_policy=None,
                data_policy={
                    "split": "discovery",
                    "proposal_loop_access": True,
                    "scope_key": current_scope(registry),
                },
                promotion_gate={
                    "criteria": ["Implementation is numerically safe."],
                    "on_pass": "Design discovery.",
                    "on_fail": "Stop.",
                    "verdict": "implementation_only",
                },
            )
            registry.experiment_proposals.add(proposal)

            with self.assertRaisesRegex(SchemaError, "has no search_policy"):
                registry.check_gate(proposal.experiment_id)

    def test_passed_secondary_frame_runs_while_default_frame_is_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            setup = registry.setup_reconciliation()
            frame = {
                "scope_id": "walltime_test",
                "role": "adopt",
                "status": "passed",
                "scope_key": {
                    **dict(setup.scope_key),
                    "max_steps": 100_000,
                    "stop_mode": "time",
                    "time_budget": 300,
                },
                "run_env": {"ATTN_BACKEND": "fa3"},
                "min_seeds": 3,
                "baseline": {
                    **dict(setup.baseline),
                    "artifact_paths": ["baseline.md"],
                },
            }
            pending = replace(setup, status="pending", scopes=(frame,))
            (root / ResearchRegistry.SETUP_PATH).write_text(
                json.dumps(pending.to_dict()), encoding="utf-8"
            )

            selected = registry._require_current_setup("walltime_test")
            self.assertEqual(selected.scope_for("walltime_test")["status"], "passed")
            with self.assertRaisesRegex(SchemaError, "experiments are stopped"):
                registry._require_current_setup()

    def test_run_seed_must_be_preregistered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry, gated=True)
            registry.runs.add(replace(run(), seed=2))
            with self.assertRaises(SchemaError):
                registry.validate(check_generated_state=False)

    def test_render_state_only_lists_terminal_beliefs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            predecessor = replace(
                belief(),
                scope={"model": "toy", "scope_key": current_scope(registry)},
            )
            successor = replace(
                predecessor,
                belief_id="blf_test_transition_v2",
                version=2,
                statement="New evidence challenges the reported transition.",
                status="challenged",
                supersedes_belief_id=predecessor.belief_id,
            )
            registry.beliefs.add(predecessor)
            registry.beliefs.add(successor)

            state = registry.render_state()

            self.assertNotIn("`blf_test_transition`", state)
            self.assertIn("`blf_test_transition_v2`", state)
            self.assertIn("1 scope-matched, evidence-backed belief records", state)

    def test_belief_supersession_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            registry.beliefs.add(
                replace(
                    belief(),
                    belief_id="blf_test_cycle_a",
                    supersedes_belief_id="blf_test_cycle_b",
                )
            )
            registry.beliefs.add(
                replace(
                    belief(),
                    belief_id="blf_test_cycle_b",
                    supersedes_belief_id="blf_test_cycle_a",
                )
            )

            with self.assertRaisesRegex(SchemaError, "supersession cycle"):
                registry.validate(check_generated_state=False)

    def test_current_belief_without_evidence_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            unsupported = replace(
                belief(),
                belief_id="blf_test_unsupported",
                evidence_ids=(),
                status="supported",
                scope={"model": "toy", "scope_key": current_scope(registry)},
            )
            registry.beliefs.add(unsupported)

            validation = registry.validate(check_generated_state=False)
            audit = registry.audit()
            state = registry.render_state()

            self.assertTrue(any("no evidence" in warning for warning in validation.warnings))
            self.assertIn(
                "terminal_belief_without_evidence",
                {item["code"] for item in audit.issues},
            )
            self.assertIn("`blf_test_unsupported`", state.split("## Provenance gaps", 1)[1])

    def test_scope_mismatch_demotes_terminal_belief_and_approved_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry, gated=True)
            registry.beliefs.add(belief())

            state = registry.render_state()

            self.assertIn("`blf_test_transition` [unscoped]", state)
            active_section = state.split("## Approved or running experiments", 1)[1]
            self.assertNotIn("`exp_test_pilot`", active_section)
            with self.assertRaisesRegex(SchemaError, "unscoped or stale"):
                registry.authorize_run("exp_test_pilot", "treatment", 1, 100)

    def test_authorize_run_requires_exact_current_gate_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            gated = replace(
                pilot_experiment(),
                data_policy={
                    "split": "discovery",
                    "proposal_loop_access": True,
                    "scope_key": current_scope(registry),
                },
                promotion_gate={
                    "criteria": ["Implementation is numerically safe."],
                    "on_pass": "Design discovery.",
                    "on_fail": "Stop.",
                    "verdict": "implementation_only",
                },
            )
            registry.gated_experiments.add(gated)

            result = registry.authorize_run(gated.experiment_id, "treatment", 1, 100)

            self.assertTrue(result["authorized"])
            self.assertEqual(result["challenge_id"], "fixed_steps_test")
            self.assertEqual(
                result["challenge_selection_fingerprint"],
                registry.challenge_events()[-1].fingerprint,
            )
            with self.assertRaisesRegex(SchemaError, "MAX_STEPS"):
                registry.authorize_run(gated.experiment_id, "treatment", 1, 99)

    def test_sticky_challenge_selection_persists_and_stop_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)

            self.assertEqual(registry.resolve_scope_id(None), "")
            with self.assertRaisesRegex(
                SchemaError, "cannot reset the hourly reporting clock"
            ):
                registry.append_challenge_event(
                    action="activated",
                    challenge_selector="fixed_steps_test",
                    selected_at="2026-07-16T00:00:30Z",
                    selected_by="test",
                    reason="Attempt a redundant activation.",
                )
            stopped = registry.append_challenge_event(
                action="stopped",
                challenge_selector="fixed_steps_test",
                selected_at="2026-07-16T00:01:00Z",
                selected_by="test",
                reason="Stop fixture challenge work.",
            )
            restarted_registry = ResearchRegistry(root)
            with self.assertRaisesRegex(SchemaError, "stopped"):
                restarted_registry.resolve_scope_id(None)
            with self.assertRaisesRegex(SchemaError, "cannot bypass"):
                restarted_registry.resolve_scope_id("steps")

            restarted_registry.append_challenge_event(
                action="activated",
                challenge_selector="fixed_steps_test",
                selected_at="2026-07-16T00:02:00Z",
                selected_by="test",
                reason="Resume fixture challenge work.",
            )
            self.assertEqual(ResearchRegistry(root).resolve_scope_id(None), "")
            self.assertEqual(stopped.generation, 2)
            self.assertEqual(
                [event.generation for event in registry.challenge_events()],
                [1, 2, 3],
            )

    def test_explicit_scope_cannot_override_the_sticky_challenge(self):
        root = Path(__file__).resolve().parents[1] / "research"
        registry = ResearchRegistry(root)

        self.assertEqual(
            registry.resolve_scope_id(None),
            "walltime_5min_h200",
        )
        with self.assertRaisesRegex(SchemaError, "differs from the sticky"):
            registry.resolve_scope_id("fixed_steps_2000")

    def test_setup_change_makes_selection_stale_until_explicit_reselection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            changed = replace(
                registry.setup_reconciliation(),
                reconciled_by="changed-test-setup",
            )
            (root / ResearchRegistry.SETUP_PATH).write_text(
                json.dumps(changed.to_dict()), encoding="utf-8"
            )

            with self.assertRaisesRegex(SchemaError, "stale for the current setup"):
                registry.resolve_scope_id(None)

            registry.append_challenge_event(
                action="activated",
                challenge_selector="fixed_steps_test",
                selected_at="2026-07-16T00:03:00Z",
                selected_by="test",
                reason="Explicitly bind the changed setup.",
            )
            self.assertEqual(registry.resolve_scope_id(None), "")

    def test_challenge_catalog_orders_five_minutes_before_two_thousand_steps(self):
        root = Path(__file__).resolve().parents[1] / "research"
        registry = ResearchRegistry(root)
        catalog = registry.challenge_catalog()
        setup = registry.setup_reconciliation()
        ordered = catalog.ordered()

        self.assertEqual(
            [entry["challenge_id"] for entry in ordered],
            ["walltime_5min_h200", "fixed_steps_2000"],
        )
        five_minute = setup.scope_for(str(ordered[0]["scope_id"]))
        self.assertEqual(five_minute["scope_key"]["stop_mode"], "time")
        self.assertEqual(five_minute["scope_key"]["time_budget"], 300)
        self.assertEqual(setup.scope_key["stop_mode"], "steps")
        self.assertEqual(setup.scope_key["max_steps"], 2000)

    def test_authorize_run_and_audit_expose_policy_inert_legacy_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            legacy = replace(
                pilot_experiment(),
                idea_id="",
                search_policy=None,
                stage="pilot",
                promotion_gate={
                    "criteria": ["Implementation is numerically safe."],
                    "on_pass": "Design discovery.",
                    "on_fail": "Stop.",
                    "verdict": "implementation_only",
                },
                data_policy={
                    "split": "discovery",
                    "proposal_loop_access": True,
                    "scope_key": current_scope(registry),
                },
            )
            registry.gated_experiments.add(legacy)

            with self.assertRaisesRegex(SchemaError, "has no search_policy"):
                registry.authorize_run(legacy.experiment_id, "treatment", 1, 100)
            self.assertIn(
                "policy_inert_legacy_gates",
                {item["code"] for item in registry.audit().issues},
            )

    def test_validate_rejects_fork_by_near_duplicate_idea(self):
        """A near-duplicate idea minted under a fresh id is caught at validate()."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)  # seeds idea() = idea_test_signal, status selected
            seed = idea()
            # A distinct idea_id whose text is byte-identical to the seeded idea.
            twin = replace(
                idea(idea_id="idea_fork_twin"),
                title=seed.title,
                summary=seed.summary,
                experimental_plan=seed.experimental_plan,
            )
            registry.idea_archive.add(twin)
            with self.assertRaisesRegex(SchemaError, "fork-by-duplicate"):
                registry.validate()

    def test_one_archived_idea_cannot_start_sibling_funnel_chains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            scoped = {
                "split": "discovery",
                "proposal_loop_access": True,
                "scope_key": current_scope(registry),
            }
            first = replace(pilot_experiment(), data_policy=scoped)
            sibling = replace(
                pilot_experiment(),
                experiment_id="exp_test_sibling_retry",
                title="Sibling retry",
                status="planned",
                frozen_at="",
                data_policy=scoped,
            )
            registry.gated_experiments.add(first)
            registry.experiment_proposals.add(sibling)

            with self.assertRaisesRegex(SchemaError, "already entered"):
                registry.check_gate(sibling.experiment_id)

    def test_pre_audit_four_pair_qualification_is_preserved_but_non_authorizing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            scoped = {
                "split": "discovery",
                "proposal_loop_access": True,
                "scope_key": current_scope(registry),
            }
            # An ordinary stage-one experiment for the same idea neither parents
            # nor conflicts with the standalone provenance qualification.
            ordinary = replace(pilot_experiment(), data_policy=scoped)
            registry.gated_experiments.add(ordinary)
            qualification = replace(
                qualification_experiment(root),
                data_policy=scoped,
            )
            registry.gated_experiments.add(qualification)

            with self.assertRaisesRegex(
                SchemaError, "preserved pre-audit qualification draft"
            ):
                registry.authorize_run(
                    qualification.experiment_id,
                    "treatment",
                    47,
                    100,
                )

    def test_qualification_does_not_block_an_ordinary_stage_one_funnel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            scoped = {
                "split": "discovery",
                "proposal_loop_access": True,
                "scope_key": current_scope(registry),
            }
            qualification = replace(
                qualification_experiment(root),
                data_policy=scoped,
            )
            registry.gated_experiments.add(qualification)
            ordinary = replace(
                pilot_experiment(),
                experiment_id="exp_ordinary_after_qualification",
                title="Ordinary stage one after qualification",
                status="planned",
                frozen_at="",
                data_policy=scoped,
            )
            registry.experiment_proposals.add(ordinary)

            result = registry.check_gate(ordinary.experiment_id)

            self.assertEqual(result["search_policy"]["stage_pairs"], 1)
            self.assertEqual(result["search_policy"]["idea_funnel_stage_count"], 1)
            self.assertNotIn("qualification", result["search_policy"])

    def test_duplicate_standalone_qualification_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            scoped = {
                "split": "discovery",
                "proposal_loop_access": True,
                "scope_key": current_scope(registry),
            }
            first = replace(
                qualification_experiment(root),
                data_policy=scoped,
            )
            second = replace(
                qualification_experiment(
                    root,
                    experiment_id="exp_test_qualification_retry",
                    status="planned",
                ),
                data_policy=scoped,
            )
            registry.gated_experiments.add(first)
            registry.experiment_proposals.add(second)

            with self.assertRaisesRegex(
                SchemaError, "already has a standalone qualification"
            ):
                registry.check_gate(second.experiment_id)

    def test_qualification_anchor_paths_and_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            scoped = {
                "split": "discovery",
                "proposal_loop_access": True,
                "scope_key": current_scope(registry),
            }

            outside = root.parent / "outside-anchor.log"
            outside.write_text("outside\n", encoding="utf-8")
            outside_exp = qualification_experiment(
                root,
                experiment_id="exp_qualification_outside",
                status="planned",
            )
            outside_policy = dict(outside_exp.search_policy)
            outside_policy["qualification"] = dict(
                outside_policy["qualification"]
            )
            outside_policy["qualification"]["historical_anchors"] = [
                dict(anchor)
                for anchor in outside_policy["qualification"]["historical_anchors"]
            ]
            outside_anchor = outside_policy["qualification"]["historical_anchors"][0]
            outside_anchor["artifact_path"] = "../outside-anchor.log"
            outside_anchor["artifact_sha256"] = hashlib.sha256(
                outside.read_bytes()
            ).hexdigest()
            outside_exp = replace(
                outside_exp,
                search_policy=outside_policy,
                data_policy=scoped,
            )
            registry.experiment_proposals.add(outside_exp)
            with self.assertRaisesRegex(SchemaError, "resolves outside"):
                registry.check_gate(outside_exp.experiment_id)

            mismatch = qualification_experiment(
                root,
                experiment_id="exp_qualification_hash_mismatch",
                status="planned",
            )
            mismatch_policy = dict(mismatch.search_policy)
            mismatch_policy["qualification"] = dict(
                mismatch_policy["qualification"]
            )
            mismatch_policy["qualification"]["historical_anchors"] = [
                dict(anchor)
                for anchor in mismatch_policy["qualification"]["historical_anchors"]
            ]
            mismatch_policy["qualification"]["historical_anchors"][0][
                "artifact_sha256"
            ] = "0" * 64
            mismatch = replace(
                mismatch,
                search_policy=mismatch_policy,
                data_policy=scoped,
            )
            registry.experiment_proposals.add(mismatch)
            with self.assertRaisesRegex(SchemaError, "hash mismatch"):
                registry.check_gate(mismatch.experiment_id)

            missing = qualification_experiment(
                root,
                experiment_id="exp_qualification_missing_anchor",
                status="planned",
            )
            missing_policy = dict(missing.search_policy)
            missing_policy["qualification"] = dict(missing_policy["qualification"])
            missing_policy["qualification"]["historical_anchors"] = [
                dict(anchor)
                for anchor in missing_policy["qualification"]["historical_anchors"]
            ]
            missing_policy["qualification"]["historical_anchors"][0][
                "artifact_path"
            ] = "anchors/does-not-exist.log"
            missing = replace(
                missing,
                search_policy=missing_policy,
                data_policy=scoped,
            )
            registry.experiment_proposals.add(missing)
            with self.assertRaisesRegex(SchemaError, "missing or unresolvable"):
                registry.check_gate(missing.experiment_id)

    def test_qualification_rechecks_anchor_hash_at_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            scoped = {
                "split": "discovery",
                "proposal_loop_access": True,
                "scope_key": current_scope(registry),
            }
            qualification = replace(
                qualification_experiment(root),
                data_policy=scoped,
            )
            registry.gated_experiments.add(qualification)
            (root / "anchors" / "seed47.log").write_text(
                "mutated after freeze\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(SchemaError, "hash mismatch"):
                registry.evaluate_search_stage(qualification.experiment_id)

    def test_invalid_update_releases_subsystem_for_a_new_child_idea(self):
        """An immutable invalid attempt is closed work, not an eternal GPU lock."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            scoped = {
                "split": "discovery",
                "proposal_loop_access": True,
                "scope_key": current_scope(registry),
            }
            invalid_attempt = replace(pilot_experiment(), data_policy=scoped)
            registry.gated_experiments.add(invalid_attempt)
            registry.evidence_updates.add(
                EvidenceUpdateRecord(
                    update_id="upd_test_invalid_attempt",
                    experiment_id=invalid_attempt.experiment_id,
                    evidence_ids=("evd_test_literature",),
                    affected_belief_ids=(),
                    affected_hypothesis_ids=(invalid_attempt.hypothesis_id,),
                    result="invalid",
                    reason="Isolation failed before a causal comparison existed.",
                    recommended_action="Create a new child idea and re-gate.",
                    created_at="2026-07-16T01:00:00Z",
                )
            )
            child_idea = replace(
                idea(idea_id="idea_integrity_repaired_signal"),
                title="Isolation-watchdog integrity repair",
                summary="Repair physical-GPU isolation after an invalid scheduler attempt.",
                experimental_plan="Freeze a new preflight-attested placement on eligible GPUs.",
                parent_idea_ids=("idea_test_signal",),
            )
            registry.idea_archive.add(child_idea)
            replacement = replace(
                pilot_experiment(),
                experiment_id="exp_integrity_repaired_signal",
                title="Integrity-repaired stage-one screen",
                status="planned",
                frozen_at="",
                idea_id=child_idea.idea_id,
                data_policy=scoped,
            )
            registry.experiment_proposals.add(replacement)

            report = registry.check_gate(replacement.experiment_id)
            self.assertEqual(report["search_policy"]["stage_pairs"], 1)

    def test_sixth_stage_one_round_must_change_research_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            scoped = {
                "split": "discovery",
                "proposal_loop_access": True,
                "scope_key": current_scope(registry),
            }
            for index in range(5):
                previous = replace(
                    pilot_experiment(),
                    experiment_id=f"exp_direction_round_{index}",
                    title=f"Direction round {index}",
                    created_at=f"2026-07-{10 + index:02d}",
                    data_policy=scoped,
                )
                registry.gated_experiments.add(previous)
                registry.runs.add(
                    replace(
                        run(previous),
                        run_id=f"run_direction_round_{index}",
                        config_hash=f"cfg_direction_{index}",
                        started_at=f"2026-07-{10 + index:02d}T00:00:00Z",
                        ended_at=f"2026-07-{10 + index:02d}T00:01:00Z",
                    )
                )

            same_direction_idea = idea(
                idea_id="idea_same_direction_next",
                subsystem="fresh_signal",
            )
            alternate_direction_idea = idea(
                idea_id="idea_optimizer_next",
                direction="optimizer",
                subsystem="fresh_signal",
            )
            registry.idea_archive.add(same_direction_idea)
            registry.idea_archive.add(alternate_direction_idea)
            same_direction = replace(
                pilot_experiment(),
                experiment_id="exp_same_direction_next",
                title="Same direction next",
                status="planned",
                frozen_at="",
                idea_id=same_direction_idea.idea_id,
                search_policy=search_policy(subsystem="fresh_signal"),
                data_policy=scoped,
            )
            alternate_direction = replace(
                same_direction,
                experiment_id="exp_optimizer_direction_next",
                title="Optimizer direction next",
                idea_id=alternate_direction_idea.idea_id,
                search_policy=search_policy(
                    subsystem="fresh_signal", direction="optimizer"
                ),
            )
            registry.experiment_proposals.add(same_direction)
            registry.experiment_proposals.add(alternate_direction)

            with self.assertRaisesRegex(SchemaError, "next round must use a different"):
                registry.check_gate(same_direction.experiment_id)
            result = registry.check_gate(alternate_direction.experiment_id)
            self.assertEqual(
                result["search_policy"]["consecutive_direction_rounds"], 0
            )

            optimizer_cooldown_idea = replace(
                idea(
                    idea_id="idea_optimizer_cooldown_round",
                    direction="optimizer",
                    subsystem="optimizer_core",
                ),
                title="Muon state geometry",
                summary="Alter matrix-update geometry in the optimizer state.",
                experimental_plan="Compare an orthogonalized update with matched seeds.",
            )
            registry.idea_archive.add(optimizer_cooldown_idea)
            optimizer_round = replace(
                pilot_experiment(),
                experiment_id="exp_optimizer_cooldown_round",
                title="Optimizer cooldown round",
                idea_id=optimizer_cooldown_idea.idea_id,
                search_policy=search_policy(
                    subsystem="optimizer_core", direction="optimizer"
                ),
                data_policy=scoped,
            )
            registry.gated_experiments.add(optimizer_round)
            registry.runs.add(
                replace(
                    run(optimizer_round),
                    run_id="run_optimizer_cooldown_round",
                    config_hash="cfg_optimizer_cooldown_round",
                    started_at="2026-07-15T00:00:00Z",
                    ended_at="2026-07-15T00:01:00Z",
                )
            )

            with self.assertRaisesRegex(SchemaError, "remains on cooldown"):
                registry.check_gate(same_direction.experiment_id)

            architecture_idea = replace(
                idea(
                    idea_id="idea_architecture_cooldown_round",
                    direction="architecture",
                    subsystem="architecture_core",
                ),
                title="Residual pathway topology",
                summary="Change the topology of residual information routing.",
                experimental_plan="Screen a gated residual branch against the frozen model.",
            )
            registry.idea_archive.add(architecture_idea)
            architecture_round = replace(
                pilot_experiment(),
                experiment_id="exp_architecture_cooldown_round",
                title="Architecture cooldown round",
                idea_id=architecture_idea.idea_id,
                search_policy=search_policy(
                    subsystem="architecture_core", direction="architecture"
                ),
                data_policy=scoped,
            )
            registry.gated_experiments.add(architecture_round)
            registry.runs.add(
                replace(
                    run(architecture_round),
                    run_id="run_architecture_cooldown_round",
                    config_hash="cfg_architecture_cooldown_round",
                    started_at="2026-07-16T00:00:00Z",
                    ended_at="2026-07-16T00:01:00Z",
                )
            )

            refreshed = registry.check_gate(same_direction.experiment_id)
            self.assertEqual(
                refreshed["search_policy"]["direction_cooldown_remaining"], 0
            )

    def test_setup_drift_stops_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            add_core_graph(ResearchRegistry(root))
            (root.parent / "data_split.json").write_text(
                '{"train":[9],"test":[6542]}\n', encoding="utf-8"
            )

            with self.assertRaisesRegex(SchemaError, "frozen setup file drift"):
                ResearchRegistry(root).validate(check_generated_state=False)

    def test_conclusive_evidence_update_closes_rendered_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry, gated=True)
            registry.runs.add(run())
            evidence = EvidenceRecord(
                evidence_id="evd_test_run_result",
                source_type="internal_run",
                paper_ids=(),
                claim_ids=(),
                run_ids=("run_test_seed1_treatment",),
                experiment_id="exp_test_pilot",
                hypothesis_ids=("hyp_test_trigger",),
                facts={"out_test_validation_bpb": 1.25},
                analysis={"method": "endpoint", "code_ref": "tests:test"},
                trust={
                    "design": "adequate",
                    "replication": "weak",
                    "scope_match": "strong",
                    "directness": "strong",
                    "limitations": ["One seed."],
                },
                assessment={
                    "relation": "opposes",
                    "strength": "moderate",
                    "limitations": ["One seed."],
                },
                artifact_paths=(),
                created_at="2026-07-16",
                created_by="test",
            )
            registry.run_evidence.add(evidence)
            registry.evidence_updates.add(
                EvidenceUpdateRecord(
                    update_id="upd_test_result",
                    experiment_id="exp_test_pilot",
                    evidence_ids=(evidence.evidence_id,),
                    affected_belief_ids=(),
                    affected_hypothesis_ids=("hyp_test_trigger",),
                    result="oppose",
                    reason="The endpoint missed the frozen gate.",
                    recommended_action="Stop this experiment.",
                    created_at="2026-07-16",
                )
            )

            state = registry.render_state()

            active_section = state.split("## Approved or running experiments", 1)[1]
            self.assertNotIn("`exp_test_pilot`", active_section)

    def test_current_views_and_audit_resolve_superseded_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            predecessor = replace(
                literature_evidence(),
                evidence_id="evd_test_literature_hypothesis_old",
                hypothesis_ids=("hyp_test_trigger",),
            )
            correction = replace(
                predecessor,
                evidence_id="evd_test_literature_hypothesis_correction",
                assessment={
                    "relation": "not_tested",
                    "strength": "weak",
                    "limitations": ["The original transfer inference was too broad."],
                },
                supersedes_evidence_id=predecessor.evidence_id,
            )
            registry.literature_evidence.add(predecessor)
            registry.literature_evidence.add(correction)
            registry.beliefs.add(
                replace(
                    belief(),
                    belief_id="blf_test_terminal_evidence_resolution",
                    evidence_ids=(predecessor.evidence_id,),
                    scope={
                        "model": "toy",
                        "scope_key": current_scope(registry),
                    },
                )
            )

            validation = registry.validate(check_generated_state=False)
            audit = registry.audit()
            state = registry.render_state()
            snapshot = registry.render_literature_snapshot()

            self.assertGreater(validation.counts["literature_evidence"], 0)
            self.assertIn("current evidence: 2 terminal / 3 append-only", state)
            self.assertIn(
                "`blf_test_terminal_evidence_resolution`",
                state,
            )
            self.assertIn(
                "Current literature-evidence records: **2** terminal / "
                "**3** append-only",
                snapshot,
            )
            issue_codes = {
                (item["code"], item["record_id"]) for item in audit.issues
            }
            self.assertIn(
                (
                    "hypothesis_without_conclusive_evidence",
                    "hyp_test_trigger",
                ),
                issue_codes,
            )

    def test_literature_evidence_gate_fails_closed_on_ambiguous_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            predecessor = literature_evidence()
            registry.literature_evidence.add(
                replace(
                    predecessor,
                    evidence_id="evd_test_literature_branch_left",
                    supersedes_evidence_id=predecessor.evidence_id,
                )
            )
            registry.literature_evidence.add(
                replace(
                    predecessor,
                    evidence_id="evd_test_literature_branch_right",
                    supersedes_evidence_id=predecessor.evidence_id,
                )
            )

            with self.assertRaisesRegex(SchemaError, "multiple successors"):
                registry._require_literature_assessments(
                    "exp_test_gate", hypothesis()
                )

    def test_audit_summarizes_missing_literature_assessments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            registry.papers.add(paper())
            registry.claims.add(claim())

            issue = next(
                item
                for item in registry.audit().issues
                if item["code"] == "literature_claims_without_evidence"
            )

            self.assertEqual(issue["severity"], "warning")
            self.assertIn("1 literature claims", issue["message"])


if __name__ == "__main__":
    unittest.main()


class AppendOnlyGuaranteeTest(unittest.TestCase):
    """Lock the REAL immutability mechanism the append_only flag documents."""

    def test_editing_a_fingerprinted_record_in_place_is_detected_on_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            registry.papers.add(paper())
            path = registry.papers.path
            # Tamper: change a field but keep the (now stale) fingerprint line.
            corrupted = path.read_text(encoding="utf-8").replace(
                "Test paper", "Tampered title"
            )
            path.write_text(corrupted, encoding="utf-8")
            with self.assertRaises(SchemaError):
                registry.papers.load()

    def test_registry_exposes_no_in_place_mutation_path(self):
        # The immutability guarantee rests on add() being the only mutator.
        for name in ("update", "replace", "delete", "remove", "rewrite", "overwrite"):
            self.assertFalse(
                hasattr(JsonlRegistry, name),
                f"JsonlRegistry unexpectedly exposes a {name}() mutation path",
            )
