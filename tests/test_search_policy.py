import copy
import math
import unittest
from types import SimpleNamespace

from vibeautoresearch.core import SchemaError
from vibeautoresearch.search_policy import (
    authorizable_seed_count,
    evaluate_stage,
    validate_search_policy,
)


def policy(stage_pairs=3, parent="exp_parent"):
    return {
        "version": 2,
        "challenge_id": "walltime_5min_h200",
        "decision_frame": "walltime_5min_h200",
        "direction": "systems_kernel",
        "subsystem": "attention_kernel",
        "stage_pairs": stage_pairs,
        "parent_experiment_id": "" if stage_pairs == 1 else parent,
        "predictions": {
            "delta_steps": {
                "expected_delta": 100,
                "unit": "steps_per_budget",
                "measurement": "final num_steps treatment minus control",
                "rationale": "The candidate removes a measured kernel bottleneck.",
            },
            "delta_quality_per_step": {
                "expected_delta": 0,
                "unit": "val_bpb",
                "measurement": "validation BPB at a matched optimizer step",
                "rationale": "The attention computation is mathematically equivalent.",
            },
            "delta_endpoint": {
                "expected_delta": -0.003,
                "unit": "val_bpb",
                "measurement": "final treatment minus paired control BPB",
                "rationale": "Additional optimizer steps should improve the endpoint.",
            },
        },
        "stopping": {
            "min_futility_pairs": 1 if stage_pairs == 1 else 3,
            "promote_if_mean_endpoint_delta_lte": -0.001,
            "stop_if_mean_endpoint_delta_gte": 0.002,
            "stop_if_mean_step_delta_lte": 0,
        },
        "portfolio": {
            "max_consecutive_failures": 2,
            "pivot_override": "",
        },
    }


def experiment(stage_pairs=3):
    return SimpleNamespace(
        experiment_id=f"exp_stage_{stage_pairs}",
        stage="discovery",
        seeds=tuple(range(42, 42 + stage_pairs)),
        arms=(
            {"arm_id": "control", "role": "control"},
            {"arm_id": "treatment", "role": "treatment"},
        ),
        search_policy=policy(stage_pairs),
    )


def paired_runs(exp, endpoint_delta, step_delta, count):
    result = []
    for seed in exp.seeds[:count]:
        result.extend(
            (
                SimpleNamespace(
                    experiment_id=exp.experiment_id,
                    arm_id="control",
                    seed=seed,
                    status="complete",
                    outcome_values={"val_bpb": 0.93, "num_steps": 1900},
                    tags=(
                        "bound_run",
                        "gpu_sampling_verified",
                        "challenge_walltime_5min_h200",
                        "frame_walltime_5min_h200",
                    ),
                    tracker={
                        "gpu_sampling_verified": True,
                        "challenge_id": "walltime_5min_h200",
                        "challenge_definition_fingerprint": "a" * 16,
                        "challenge_selection_fingerprint": "b" * 16,
                    },
                ),
                SimpleNamespace(
                    experiment_id=exp.experiment_id,
                    arm_id="treatment",
                    seed=seed,
                    status="complete",
                    outcome_values={
                        "val_bpb": 0.93 + endpoint_delta,
                        "num_steps": 1900 + step_delta,
                    },
                    tags=(
                        "bound_run",
                        "gpu_sampling_verified",
                        "challenge_walltime_5min_h200",
                        "frame_walltime_5min_h200",
                    ),
                    tracker={
                        "gpu_sampling_verified": True,
                        "challenge_id": "walltime_5min_h200",
                        "challenge_definition_fingerprint": "a" * 16,
                        "challenge_selection_fingerprint": "b" * 16,
                    },
                ),
            )
        )
    return result


def qualification_policy():
    result = policy(stage_pairs=3)
    result["stage_pairs"] = 4
    result["parent_experiment_id"] = ""
    result["stopping"]["min_futility_pairs"] = 4
    result["stopping"]["promote_if_mean_endpoint_delta_lte"] = -0.002396
    result["qualification"] = {
        "kind": "historical_equivalence_bridge",
        "equivalence_margin": 0.002396,
        "superiority_minimum_effect": 0.002396,
        "required_helping_pairs": 4,
        "token_ratio_lower": 0.98,
        "token_ratio_upper": 1.02,
        "historical_anchors": [
            {
                "seed": seed,
                "val_bpb": 0.93,
                "total_tokens": 100_000,
                "artifact_path": f"anchors/seed{seed}.log",
                "artifact_sha256": f"{seed:064x}",
            }
            for seed in range(47, 51)
        ],
    }
    return result


def qualification_experiment():
    return SimpleNamespace(
        experiment_id="exp_qualification_four",
        stage="validation",
        seeds=(47, 48, 49, 50),
        arms=(
            {"arm_id": "control", "role": "control"},
            {"arm_id": "treatment", "role": "treatment"},
        ),
        search_policy=qualification_policy(),
    )


def qualification_runs(
    exp,
    *,
    replay_delta=0.0,
    endpoint_delta=-0.003,
    token_ratio=1.0,
    count=4,
):
    result = []
    anchors = {
        anchor["seed"]: anchor
        for anchor in exp.search_policy["qualification"]["historical_anchors"]
    }
    for seed in exp.seeds[:count]:
        anchor = anchors[seed]
        treatment_bpb = anchor["val_bpb"] + replay_delta
        common = {
            "experiment_id": exp.experiment_id,
            "seed": seed,
            "status": "complete",
            "tags": (
                "bound_run",
                "gpu_sampling_verified",
                "challenge_walltime_5min_h200",
            ),
            "tracker": {
                "gpu_sampling_verified": True,
                "challenge_id": "walltime_5min_h200",
                "challenge_definition_fingerprint": "a" * 16,
                "challenge_selection_fingerprint": "b" * 16,
            },
        }
        result.extend(
            (
                SimpleNamespace(
                    **common,
                    arm_id="control",
                    outcome_values={
                        "val_bpb": treatment_bpb - endpoint_delta,
                        "num_steps": 1100,
                    },
                ),
                SimpleNamespace(
                    **common,
                    arm_id="treatment",
                    outcome_values={
                        "val_bpb": treatment_bpb,
                        "num_steps": 2000,
                        "total_tokens": int(
                            anchor["total_tokens"] * token_ratio
                        ),
                    },
                ),
            )
        )
    return result


class SearchPolicySchemaTest(unittest.TestCase):
    def test_effect_funnel_cannot_use_implementation_only_pilot_stage(self):
        exp = experiment(1)
        exp.stage = "pilot"
        with self.assertRaisesRegex(SchemaError, "implementation-only"):
            validate_search_policy(exp.search_policy, exp)

    def test_requires_falsifiable_step_and_quality_predictions(self):
        exp = experiment()
        broken = policy()
        del broken["predictions"]["delta_quality_per_step"]["measurement"]
        with self.assertRaisesRegex(SchemaError, "measurement"):
            validate_search_policy(broken, exp)

    def test_seed_count_must_equal_funnel_stage(self):
        exp = experiment()
        exp.seeds = (42, 43)
        with self.assertRaisesRegex(SchemaError, "freezes 2 seeds"):
            validate_search_policy(exp.search_policy, exp)

    def test_campaign_cannot_self_waive_or_delay_the_pivot(self):
        exp = experiment()
        too_long = policy()
        too_long["portfolio"]["max_consecutive_failures"] = 4
        with self.assertRaisesRegex(SchemaError, "must be <= 3"):
            validate_search_policy(too_long, exp)

        waived = policy()
        waived["portfolio"]["pivot_override"] = (
            "The agent wants to keep searching the same subsystem."
        )
        with self.assertRaisesRegex(SchemaError, "cannot waive"):
            validate_search_policy(waived, exp)


class StagedStoppingTest(unittest.TestCase):
    def test_first_three_large_regressions_stop_a_six_pair_stage(self):
        exp = experiment(6)
        report = evaluate_stage(exp, paired_runs(exp, 0.0105, -150, 3))
        self.assertEqual(report["verdict"], "stop_futility")
        self.assertEqual(authorizable_seed_count(exp, report), 0)

    def test_first_three_promising_pairs_unlock_only_the_next_tranche(self):
        exp = experiment(10)
        report = evaluate_stage(exp, paired_runs(exp, -0.003, 120, 3))
        self.assertEqual(report["verdict"], "continue")
        self.assertEqual(authorizable_seed_count(exp, report), 6)

    def test_ten_confirming_pairs_become_an_adoption_candidate(self):
        exp = experiment(10)
        report = evaluate_stage(exp, paired_runs(exp, -0.003, 120, 10))
        self.assertEqual(report["verdict"], "adoption_candidate")
        self.assertEqual(report["complete_pairs"], 10)

    def test_missing_step_measurement_invalidates_the_checkpoint(self):
        exp = experiment(3)
        runs = paired_runs(exp, -0.003, 120, 3)
        runs[-1].outcome_values.pop("num_steps")
        report = evaluate_stage(exp, runs)
        self.assertEqual(report["verdict"], "invalid")

    def test_unsampled_complete_run_invalidates_the_checkpoint(self):
        exp = experiment(3)
        runs = paired_runs(exp, -0.003, 120, 3)
        runs[-1].tracker["gpu_sampling_verified"] = False
        report = evaluate_stage(exp, runs)
        self.assertEqual(report["verdict"], "invalid")
        self.assertIn("GPU-sampling provenance", report["reason"])

    def test_run_from_another_challenge_invalidates_the_checkpoint(self):
        exp = experiment(3)
        runs = paired_runs(exp, -0.003, 120, 3)
        runs[-1].tags = tuple(
            "challenge_fixed_steps_2000"
            if tag == "challenge_walltime_5min_h200"
            else tag
            for tag in runs[-1].tags
        )

        report = evaluate_stage(exp, runs)

        self.assertEqual(report["verdict"], "invalid")
        self.assertIn("wrong challenge", report["reason"])

    def test_off_budget_diagnostic_is_ignored(self):
        exp = experiment(3)
        runs = paired_runs(exp, -0.003, 120, 3)
        diagnostic = paired_runs(exp, 0.5, -500, 1)
        for item in diagnostic:
            item.tags = (*item.tags, "diagnostic_offbudget_4000steps")
        report = evaluate_stage(exp, [*runs, *diagnostic])
        self.assertEqual(report["verdict"], "promote")
        self.assertEqual(report["complete_pairs"], 3)


class QualificationPolicyTest(unittest.TestCase):
    def test_exact_equality_boundaries_and_zero_variance_pass(self):
        exp = qualification_experiment()
        upper = evaluate_stage(
            exp,
            qualification_runs(
                exp,
                replay_delta=0.002396,
                endpoint_delta=-0.002396,
                token_ratio=1.02,
            ),
        )
        self.assertEqual(upper["verdict"], "qualification_pass")
        self.assertTrue(upper["replay_equivalence"]["passed"])
        self.assertTrue(upper["concurrent_superiority"]["passed"])
        self.assertTrue(upper["token_fidelity"]["passed"])
        self.assertEqual(upper["replay_equivalence"]["sample_sd"], 0.0)
        self.assertFalse(upper["adoption_authorized"])
        self.assertFalse(upper["releases_ordinary_child"])
        self.assertEqual(authorizable_seed_count(exp, upper), 0)

        lower = evaluate_stage(
            exp,
            qualification_runs(
                exp,
                replay_delta=-0.002396,
                endpoint_delta=-0.002396,
                token_ratio=0.98,
            ),
        )
        self.assertEqual(lower["verdict"], "qualification_pass")

    def test_missing_or_misaligned_anchor_fails_schema(self):
        exp = qualification_experiment()
        missing = copy.deepcopy(exp.search_policy)
        missing["qualification"]["historical_anchors"].pop()
        exp.search_policy = missing
        with self.assertRaisesRegex(SchemaError, "exactly 4 anchors"):
            validate_search_policy(exp.search_policy, exp)

        exp = qualification_experiment()
        exp.search_policy["qualification"]["historical_anchors"][3]["seed"] = 51
        with self.assertRaisesRegex(SchemaError, "anchors must match"):
            validate_search_policy(exp.search_policy, exp)

    def test_incomplete_pairs_remain_pending_and_all_four_are_authorizable(self):
        exp = qualification_experiment()
        report = evaluate_stage(exp, qualification_runs(exp, count=3))
        self.assertEqual(report["verdict"], "pending")
        self.assertEqual(report["complete_pairs"], 3)
        self.assertEqual(authorizable_seed_count(exp, report), 4)

    def test_duplicate_completed_arm_invalidates_qualification(self):
        exp = qualification_experiment()
        runs = qualification_runs(exp)
        duplicate = copy.deepcopy(runs[0])
        runs.append(duplicate)
        report = evaluate_stage(exp, runs)
        self.assertEqual(report["verdict"], "invalid")
        self.assertIn("duplicate completed runs", report["reason"])

    def test_invalid_arm_invalidates_qualification(self):
        exp = qualification_experiment()
        runs = qualification_runs(exp)
        runs[-1].status = "invalid"
        report = evaluate_stage(exp, runs)
        self.assertEqual(report["verdict"], "invalid")
        self.assertIn("immutable invalid qualification arm", report["reason"])

    def test_token_fidelity_uses_log_ratio_interval(self):
        exp = qualification_experiment()
        runs = qualification_runs(exp)
        runs[-1].outcome_values["total_tokens"] = 105_000
        report = evaluate_stage(exp, runs)
        self.assertEqual(report["verdict"], "qualification_fail")
        self.assertFalse(report["token_fidelity"]["passed"])
        self.assertTrue(math.isfinite(report["token_fidelity"]["ci90_upper"]))


if __name__ == "__main__":
    unittest.main()
