"""Fail-closed authority tests for non-scored one-seed GPU diagnostics."""

from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from tests.test_research_system import (
    add_core_graph,
    create_empty_research,
    current_scope,
    pilot_experiment,
)
from vibeautoresearch.core import SchemaError
from vibeautoresearch.registry import ResearchRegistry


def diagnostic_pilot(registry: ResearchRegistry):
    base = pilot_experiment()
    analysis = {
        key: value
        for key, value in base.analysis_plan.items()
        if key != "noise_model"
    }
    analysis.update(
        {
            "diagnostic_only": True,
            "endpoint_val_bpb_measured": False,
            "validation_data_access": False,
            "gpu_authority": {
                "round1_registered_pairs": 1,
                "round1_seed": 1,
                "same_device_replay_gpus": 1,
                "attribution_assay_gpus": 2,
                "second_round1_pair_allowed": False,
                "opportunistic_filler_allowed": False,
            },
        }
    )
    return replace(
        base,
        stage="pilot",
        search_policy={},
        analysis_plan=analysis,
        promotion_gate={
            "criteria": ["Implementation and mechanism diagnostics pass."],
            "on_pass": "Draft a separate effect-funnel proposal.",
            "on_fail": "Close the diagnostic without endpoint scoring.",
            "verdict": "implementation_only",
        },
        data_policy={
            "split": "train_diagnostic",
            "proposal_loop_access": True,
            "scope_key": current_scope(registry),
        },
    )


class DiagnosticPilotAuthorityTest(unittest.TestCase):
    def test_pilot_cannot_receive_ordinary_endpoint_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            pilot = diagnostic_pilot(registry)
            registry.gated_experiments.add(pilot)

            with self.assertRaisesRegex(
                SchemaError, "diagnostic-only pilot"
            ):
                registry.authorize_run(
                    pilot.experiment_id,
                    "treatment",
                    1,
                    100,
                )

            result = registry.authorize_run(
                pilot.experiment_id,
                "treatment",
                1,
                100,
                diagnostic_only=True,
            )
            self.assertTrue(result["authorized"])
            self.assertTrue(result["diagnostic_only_authority"])
            self.assertFalse(result["endpoint_scoring_authorized"])
            self.assertEqual(
                result["stage_evaluation"]["verdict"],
                "diagnostic_open",
            )
            self.assertFalse(
                result["search_policy"]["effect_funnel_stage_consumed"]
            )

    def test_diagnostic_gpu_authority_must_forbid_filler_and_second_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            pilot = diagnostic_pilot(registry)
            bad_analysis = dict(pilot.analysis_plan)
            bad_gpu = dict(bad_analysis["gpu_authority"])
            bad_gpu["opportunistic_filler_allowed"] = True
            bad_analysis["gpu_authority"] = bad_gpu
            pilot = replace(pilot, analysis_plan=bad_analysis)
            registry.gated_experiments.add(pilot)

            with self.assertRaisesRegex(
                SchemaError, "diagnostic GPU authority is not fail-closed"
            ):
                registry.authorize_run(
                    pilot.experiment_id,
                    "treatment",
                    1,
                    100,
                    diagnostic_only=True,
                )

    def test_effect_experiment_without_search_policy_remains_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            effect = replace(
                pilot_experiment(),
                stage="pilot",
                search_policy={},
                promotion_gate={
                    "criteria": ["Implementation is numerically safe."],
                    "on_pass": "Design discovery.",
                    "on_fail": "Stop.",
                    "verdict": "implementation_only",
                },
                data_policy={
                    "split": "train",
                    "proposal_loop_access": True,
                    "scope_key": current_scope(registry),
                },
            )
            registry.gated_experiments.add(effect)

            with self.assertRaisesRegex(SchemaError, "has no search_policy"):
                registry.authorize_run(
                    effect.experiment_id,
                    "treatment",
                    1,
                    100,
                    diagnostic_only=True,
                )

    def test_effect_experiment_cannot_request_diagnostic_only_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "research"
            create_empty_research(root)
            registry = ResearchRegistry(root)
            add_core_graph(registry)
            effect = replace(
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
            registry.gated_experiments.add(effect)

            with self.assertRaisesRegex(
                SchemaError,
                "diagnostic-only authority is reserved",
            ):
                registry.authorize_run(
                    effect.experiment_id,
                    "treatment",
                    1,
                    100,
                    diagnostic_only=True,
                )


if __name__ == "__main__":
    unittest.main()
