"""Focused contracts for the Paper-019 FA3-sidecar typed-chain helper."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from tools import register_paper019_fa3_boundary_sidecar_chain as registration
from vibeautoresearch.core import SchemaError, canonical_json
from vibeautoresearch.jsonl import JsonlRegistry
from vibeautoresearch.toolkit import InterventionRecord


class Paper019Fa3BoundarySidecarRegistrationTest(unittest.TestCase):
    def test_constructs_full_typed_chain_and_paper_canonical_rating(self):
        chain = registration.records()

        self.assertEqual(
            chain.mechanism.claim_ids,
            ("clm_fa3_varlen_external_cuseqlens_contract",),
        )
        self.assertEqual(chain.mechanism.origin_type, "mixed")
        self.assertEqual(chain.mechanism.observation_ids, ())
        self.assertEqual(chain.hypothesis.mechanism_ids, (chain.mechanism.mechanism_id,))
        self.assertEqual(chain.hypothesis.observable_predictions, ())
        self.assertEqual(chain.hypothesis.context_ids, ())
        self.assertEqual(chain.idea.hypothesis_id, chain.hypothesis.hypothesis_id)
        self.assertEqual(chain.proposal.seeds, (63,))
        self.assertEqual(chain.proposal.idea_id, chain.idea.idea_id)
        self.assertEqual(chain.proposal.search_policy["stage_pairs"], 1)
        self.assertEqual(
            chain.proposal.search_policy["subsystem"],
            "fa3_boundary_sidecar_transport",
        )
        self.assertEqual(
            registration.PAPER_CRITIC_RATING,
            {
                "novelty": 4,
                "provenance": 4,
                "validity": 5,
                "impact": 3,
                "reliability": 3,
                "feasibility": 4,
                "falsifiability": 5,
                "coherence": 5,
            },
        )
        self.assertEqual(sum(registration.PAPER_CRITIC_RATING.values()), 33)
        self.assertIn("4/4/5/3/3/4/5/5 = 33/40", chain.idea.notes)
        self.assertEqual(
            sum(registration.IMPLEMENTATION_AWARE_CRITIC_RATING.values()),
            30,
        )
        self.assertIn("4/4/5/2/2/3/5/5 = 30/40", chain.idea.notes)
        self.assertEqual(
            chain.proposal.analysis_plan[
                "prelaunch_non_scored_diagnostics"
            ]["fixed_steps"],
            250,
        )
        self.assertFalse(
            chain.proposal.analysis_plan["throughput_gate"][
                "single_placement_authorizes_endpoint"
            ]
        )
        self.assertIn("static code inspection", chain.mechanism.notes)

    def test_ordinary_arms_differ_only_sidecar_flag(self):
        chain = registration.records()
        summary = registration._validate_arm_separation(chain)
        control = dict(chain.control_intervention.parameters["env"])
        treatment = dict(chain.treatment_intervention.parameters["env"])

        self.assertEqual(summary["only_difference"], "PACKER_DOC_BOUNDARIES")
        self.assertEqual(control["PACKER_DOC_BOUNDARIES"], "0")
        self.assertEqual(treatment["PACKER_DOC_BOUNDARIES"], "1")
        for forbidden in (
            "PACKER_BOUNDARY_VERIFY_BATCHES",
            "PACKER_DIAGNOSTIC_HASHES",
            "PACKER_SIDECAR_ACTIVATE_STEP",
        ):
            self.assertNotIn(forbidden, control)
            self.assertNotIn(forbidden, treatment)
        control.pop("PACKER_DOC_BOUNDARIES")
        treatment.pop("PACKER_DOC_BOUNDARIES")
        self.assertEqual(control, treatment)
        self.assertEqual(control, registration.RECOVERED_RECIPE_ENV)

    def test_non_scored_diagnostics_and_throughput_gate_are_explicit(self):
        chain = registration.records()
        analysis = chain.proposal.analysis_plan
        diagnostic = analysis["prelaunch_non_scored_diagnostics"]
        profile = analysis["throughput_gate"]

        self.assertEqual(diagnostic["exact_equivalence"]["batches"], 1000)
        self.assertEqual(
            diagnostic["state_parity"]["phases"]["semantic_state_aa"]["steps"],
            [0, 19],
        )
        self.assertEqual(
            diagnostic["state_parity"]["phases"]["mechanism_ab"]["steps"],
            [20, 39],
        )
        self.assertEqual(
            diagnostic["runtime"]["compiled_graph_count_after_warmup"], 1
        )
        self.assertEqual(diagnostic["runtime"]["recompile_count"], 0)
        self.assertEqual(profile["point_ratio_minimum"], 1.054)
        self.assertEqual(profile["one_sided_95pct_lcb_minimum"], 1.041)
        self.assertEqual(profile["clean_samples_exact"], 200)
        self.assertEqual(profile["block_bootstrap_blocks_exact"], 20)
        self.assertEqual(profile["integrity_failures_allowed"], 0)
        self.assertIn("interior-BOS mismatch", chain.proposal.promotion_gate["criteria"][0])

    def test_proposal_and_approved_preview_bind_identical_definition(self):
        chain = registration.records()

        self.assertEqual(chain.proposal.status, "planned")
        self.assertEqual(chain.proposal.frozen_at, "")
        self.assertEqual(chain.gated_preview.status, "approved")
        self.assertTrue(chain.gated_preview.frozen_at)
        self.assertEqual(chain.proposal.fingerprint, chain.gated_preview.fingerprint)
        self.assertEqual(
            canonical_json(chain.proposal.definition()),
            canonical_json(chain.gated_preview.definition()),
        )
        persisted_records = [record for _, record in chain.append_sequence()]
        self.assertNotIn(chain.gated_preview, persisted_records)
        self.assertIn(chain.proposal, persisted_records)
        self.assertEqual(
            [name for name, _ in chain.append_sequence()][-1],
            "experiment_proposals",
        )

    def test_stage_one_prediction_and_control_scope_are_conservative(self):
        chain = registration.records()
        policy = chain.proposal.search_policy

        self.assertEqual(
            policy["predictions"]["delta_endpoint"]["expected_delta"], -0.002396
        )
        self.assertEqual(
            policy["stopping"]["promote_if_mean_endpoint_delta_lte"], -0.002396
        )
        self.assertEqual(
            chain.proposal.analysis_plan["noise_model"]["effective_sigma"],
            0.001198,
        )
        self.assertEqual(
            chain.proposal.analysis_plan["noise_model"]["minimum_effect"],
            0.002396,
        )
        self.assertIn("0.927183", chain.hypothesis.notes)
        self.assertIn("neither a global nor concurrent control", chain.hypothesis.notes)
        self.assertEqual(
            chain.proposal.data_policy["scope_key"], registration.SCOPE_KEY
        )

    def test_exact_append_is_idempotent_and_conflicts_fail_closed(self):
        chain = registration.records()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interventions.jsonl"
            path.write_text("# test registry\n", encoding="utf-8")
            store = JsonlRegistry(path, InterventionRecord)

            first = registration._exact_action(
                store, chain.control_intervention, apply=True
            )
            replay = registration._exact_action(
                store, chain.control_intervention, apply=True
            )
            self.assertEqual(first, "appended")
            self.assertEqual(replay, "skip_exact")
            self.assertEqual(len(store.load()), 1)

            conflict = replace(
                chain.control_intervention,
                description="conflicting content",
            )
            with self.assertRaisesRegex(SchemaError, "different typed content"):
                registration._exact_action(store, conflict, apply=False)

    def test_source_contract_is_present_without_running_or_mutating(self):
        result = registration._source_contract()

        self.assertEqual(
            set(result),
            {
                "lib.py",
                "train.py",
                "tests/test_packer_doc_boundaries.py",
            },
        )
        self.assertTrue(all(item["present"] for item in result.values()))


if __name__ == "__main__":
    unittest.main()
