"""Focused contracts for the Paper-020 GPAS prospective-chain helper."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import register_paper020_gpas_chain as registration
from vibeautoresearch.core import SchemaError, canonical_json
from vibeautoresearch.jsonl import JsonlRegistry
from vibeautoresearch.toolkit import InterventionRecord


class Paper020GPASRegistrationTest(unittest.TestCase):
    def test_scheduler_readiness_requires_the_exact_complete_workflow(self):
        required = registration.REQUIRED_SCHEDULER_IMPLEMENTATION_FILES

        self.assertEqual(
            set(registration.LOCAL_IMPLEMENTATION_HASHES),
            registration.REQUIRED_LOCAL_IMPLEMENTATION_FILES,
        )
        self.assertTrue(registration._local_implementation_ready())
        self.assertFalse(registration.SCHEDULER_IMPLEMENTATION_HASHES)
        self.assertFalse(registration._scheduler_ready())
        self.assertEqual(
            required,
            {
                "vibeautoresearch/registry.py",
                "tools/run_gated.py",
                "tests/test_run_gated.py",
                "tests/test_diagnostic_pilot_authority.py",
                "tools/run_paper020_gpas_runtime.py",
                "tests/test_run_paper020_gpas_runtime.py",
                "tools/run_paper020_gpas_worker.py",
                "tests/test_run_paper020_gpas_worker.py",
                registration.RUNTIME_AUTHORITY_PATH,
            },
        )
        with mock.patch.object(
            registration,
            "SCHEDULER_IMPLEMENTATION_HASHES",
            {"tools/run_paper020_gpas_runtime.py": "0" * 64},
        ):
            self.assertFalse(registration._scheduler_ready())
        with mock.patch.object(
            registration,
            "SCHEDULER_IMPLEMENTATION_HASHES",
            {path: "0" * 64 for path in required},
        ):
            self.assertTrue(registration._scheduler_ready())

    def test_full_typed_chain_and_authoritative_ratings(self):
        chain = registration.records()

        self.assertEqual(chain.mechanism.claim_ids, registration.CLAIM_IDS)
        self.assertEqual(chain.mechanism.origin_type, "mixed")
        self.assertEqual(
            chain.hypothesis.mechanism_ids,
            (registration.MECHANISM_ID,),
        )
        self.assertEqual(
            chain.hypothesis.observable_predictions[0]["observable_id"],
            registration.OBSERVABLE_ID,
        )
        self.assertEqual(chain.idea.hypothesis_id, registration.HYPOTHESIS_ID)
        self.assertEqual(sum(registration.PROPOSER_RATING.values()), 37)
        self.assertEqual(
            sum(registration.AUTHORITATIVE_CRITIC_RATING.values()),
            33,
        )
        self.assertIn("4/5/5/4/4/5/5/5 = 37/40", chain.idea.notes)
        self.assertIn("4/5/4/3/2/5/5/5 = 33/40", chain.idea.notes)
        self.assertEqual(chain.tool_proposal.status, "approved")
        self.assertEqual(chain.hypothesis.status, "blocked")
        self.assertEqual(chain.capability_gap.status, "open")

    def test_source_faithful_shared_parameter_and_optimizer_contract(self):
        chain = registration.records()
        params = chain.treatment_intervention.parameters
        contract = params["gpas_parameter_contract"]
        optimizer = params["optimizer_group_contract"]
        control_optimizer = chain.control_intervention.parameters[
            "optimizer_group_contract"
        ]

        self.assertEqual(contract["parameter_tensor_count"], 8)
        self.assertEqual(contract["per_tensor_shape"], [])
        self.assertEqual(contract["total_numel"], 8)
        self.assertEqual(contract["applications_per_layer"], 2)
        self.assertTrue(contract["shared_same_scalar_between_sites"])
        self.assertEqual(contract["initialization"], 0.0)
        self.assertEqual(
            contract["formula"],
            "z_out = z - silu(alpha[layer]) * stop_gradient(z)",
        )
        self.assertEqual(optimizer["kind"], "adamw")
        self.assertEqual(optimizer["learning_rate"], 0.005)
        self.assertEqual(
            optimizer["learning_rate_authority"],
            "paper_frozen_absolute_constant",
        )
        self.assertEqual(optimizer["current_R0_scalar_lr"], 0.8)
        self.assertEqual(
            optimizer["learning_rate_over_current_R0_scalar_lr"],
            0.00625,
        )
        self.assertFalse(optimizer["prospective_paper_amendment_required"])
        self.assertTrue(registration.PAPER020_OPTIMIZER_AUTHORITY_FINAL)
        self.assertNotIn("scalar_lr_0.5_times_0.01", optimizer.values())
        self.assertEqual(optimizer["betas"], [0.8, 0.95])
        self.assertEqual(optimizer["eps"], 1e-10)
        self.assertEqual(optimizer["weight_decay"], 0.0)
        self.assertFalse(optimizer["demon_beta1"])
        self.assertFalse(optimizer["is_x0_muon_warmdown"])
        self.assertEqual(control_optimizer["kind"], "absent")
        self.assertEqual(control_optimizer["parameter_membership"], "none")

    def test_arm_difference_and_packer_zero_contract(self):
        chain = registration.records()
        report = registration._validate_arm_separation(chain)
        control = dict(chain.control_intervention.parameters["env"])
        treatment = dict(chain.treatment_intervention.parameters["env"])

        self.assertEqual(report["only_env_difference"], "GPAS_ENABLE")
        self.assertEqual(control["GPAS_ENABLE"], "0")
        self.assertEqual(treatment["GPAS_ENABLE"], "1")
        control.pop("GPAS_ENABLE")
        treatment.pop("GPAS_ENABLE")
        self.assertEqual(control, treatment)
        self.assertEqual(
            chain.treatment_intervention.parameters["packer_zero_defaults"],
            registration.PACKER_ZERO_DEFAULTS,
        )
        for forbidden in (
            "PACKER_BOUNDARY_VERIFY_BATCHES",
            "PACKER_DIAGNOSTIC_HASHES",
            "PACKER_SIDECAR_ACTIVATE_STEP",
        ):
            self.assertNotIn(forbidden, control)
        self.assertEqual(control["PACKER_DOC_BOUNDARIES"], "0")

    def test_same_device_canonical_replay_is_fail_closed(self):
        replay = registration.CANONICAL_REPLAY_CONTRACT

        self.assertEqual(replay["hardware"], "one_same_physical_H200_UUID")
        self.assertEqual(replay["step"], 0)
        self.assertIn("logits", replay["exact_equal"])
        self.assertIn(
            "all_preexisting_parameter_gradients",
            replay["exact_equal"],
        )
        self.assertEqual(
            replay["canonical_projection"]["treatment_only_model_keys_removed"],
            [
                f"transformer.h.{layer}.gpas_alpha"
                for layer in range(8)
            ],
        )
        self.assertEqual(
            replay["on_any_mismatch"],
            "INVALID_DIAGNOSTIC_NO_RELAXATION_OR_REPAIR",
        )
        self.assertEqual(
            replay["packer_diagnostic_flags"],
            registration.PACKER_ZERO_DEFAULTS,
        )

    def test_applicability_mediation_runtime_and_gpu_authority(self):
        chain = registration.records()
        analysis = chain.proposal.analysis_plan
        applicability = analysis["applicability"]
        mediation = analysis["mediation"]
        runtime = analysis["runtime"]
        gpu = analysis["gpu_authority"]

        self.assertEqual(applicability["control_steps"], 32)
        self.assertFalse(applicability["separate_32_step_worker_execution"])
        self.assertEqual(
            applicability["row_source"],
            "control_steps_1_to_32_of_the_single_eager_256_step_mediator_pair",
        )
        self.assertEqual(applicability["minimum_valid_steps"], 24)
        self.assertEqual(
            applicability["valid_step_conjunction"][
                "spearman_layer_index_vs_log_variance_min"
            ],
            0.60,
        )
        self.assertEqual(
            applicability["valid_step_conjunction"]["v7_over_v0_min"],
            1.25,
        )
        self.assertEqual(mediation["paired_steps"], 256)
        self.assertFalse(mediation["compiled"])
        self.assertEqual(
            mediation["treatment_mean_v7_over_v0_over_control_max"],
            0.85,
        )
        self.assertEqual(
            mediation["treatment_mean_v7_over_control_mean_v7_max"],
            0.90,
        )
        self.assertEqual(
            mediation["endpoint_zero_gate_counterfactual"]["minimum"],
            0.50,
        )
        self.assertEqual(runtime["registered_diagnostic_seed_pairs"], 1)
        self.assertEqual(runtime["governed_pair_executions"], 3)
        self.assertEqual(
            runtime["compiled_graph_count_per_clean_timing_arm"],
            1,
        )
        self.assertEqual(
            runtime["recompile_count_per_clean_timing_arm"],
            0,
        )
        self.assertEqual(
            runtime["cudagraph_recording_count_per_clean_timing_arm"],
            0,
        )
        self.assertEqual(
            runtime["timing"]["clean_step_range_inclusive"],
            [33, 256],
        )
        self.assertEqual(
            runtime["timing"]["clean_step_count_per_arm_per_placement"],
            224,
        )
        self.assertEqual(runtime["timing"]["point_ratio_min"], 0.99)
        self.assertEqual(
            runtime["timing"]["one_sided_95pct_lcb_min"],
            0.985,
        )
        self.assertEqual(gpu["round1_registered_pairs"], 1)
        self.assertEqual(gpu["round1_governed_pair_executions"], 3)
        self.assertFalse(gpu["second_round1_pair_allowed"])
        self.assertEqual(
            gpu["future_endpoint_funnel"]["maximum_concurrent_pairs_on_8_H200"],
            3,
        )
        self.assertEqual(
            gpu["future_endpoint_funnel"]["attribution_reserve_H200"],
            2,
        )
        self.assertFalse(
            gpu["future_endpoint_funnel"]["unregistered_filler"]
        )

    def test_pilot_is_implementation_only_and_preview_is_definition_identical(self):
        chain = registration.records()

        self.assertEqual(chain.proposal.stage, "pilot")
        self.assertEqual(chain.proposal.seeds, (66,))
        self.assertEqual(
            chain.proposal.promotion_gate["verdict"],
            "implementation_only",
        )
        self.assertFalse(chain.proposal.analysis_plan["endpoint_val_bpb_measured"])
        self.assertIsNone(chain.proposal.search_policy)
        endpoint_prior = chain.proposal.analysis_plan["future_endpoint_prior"]
        self.assertEqual(endpoint_prior["funnel_pairs"], [1, 3, 6, 10])
        self.assertEqual(
            endpoint_prior["predicted_raw_paired_val_bpb_delta_lte"],
            -0.003,
        )
        self.assertEqual(
            endpoint_prior["status"],
            "non_executable_requires_separate_discovery_registration",
        )
        self.assertEqual(chain.proposal.status, "planned")
        self.assertEqual(chain.gated_schema_preview.status, "approved")
        self.assertEqual(
            canonical_json(chain.proposal.definition()),
            canonical_json(chain.gated_schema_preview.definition()),
        )
        self.assertEqual(
            chain.proposal.fingerprint,
            chain.gated_schema_preview.fingerprint,
        )
        persisted = [record for _, record in chain.append_sequence()]
        self.assertNotIn(chain.gated_schema_preview, persisted)
        self.assertIn(chain.proposal, persisted)

    @unittest.skipUnless(
        (registration.REPO_ROOT / ".git").exists(),
        "pre-intervention blob audit requires repository Git history",
    )
    def test_source_hashes_audit_and_preflight_stays_blocked(self):
        source = registration._source_contract()
        result = registration.preflight_or_apply(apply=False)

        self.assertEqual(
            source["mode"],
            "local_implementation_bound_scheduler_blocked",
        )
        self.assertTrue(source["local_implementation_ready"])
        self.assertFalse(source["implementation_ready"])
        self.assertEqual(
            source["paper020"][registration.PAPER020_TEX]["sha256"],
            registration.PAPER020_TEX_SHA256,
        )
        self.assertEqual(
            source["paper020"][registration.PAPER020_PDF]["sha256"],
            registration.PAPER020_PDF_SHA256,
        )
        self.assertFalse(registration.PAPER020_RUNTIME_AUTHORITY_FINAL)
        self.assertTrue(
            any(
                "aud_paper020_worker_boundary_20260729" in blocker
                for blocker in result["prospective_gate"]["blockers"]
            )
        )
        self.assertEqual(
            registration.PAPER020_TEX_SHA256,
            "c9053c401cb2aa53027aa457bad4bf6ebe439af65b4cdae04eac190bed36e592",
        )
        self.assertEqual(
            registration.PAPER020_PDF_SHA256,
            "1ebb95deee3f489b8016a0d28115e9aa4ec7725e7534deee93011d66725db9bb",
        )
        gate = result["prospective_gate"]
        self.assertTrue(gate["blockers"])
        self.assertFalse(gate["search_policy"]["attached"])
        diagnostic_authority = gate["search_policy"][
            "diagnostic_policy_authority"
        ]
        self.assertTrue(diagnostic_authority["diagnostic_only"])
        self.assertFalse(
            diagnostic_authority["effect_funnel_stage_consumed"]
        )
        self.assertFalse(diagnostic_authority["endpoint_scoring_authorized"])
        self.assertFalse(gate["proposal_registration_ready"])
        self.assertFalse(gate["gated_registration_supported"])
        self.assertFalse(result["launch_authority"])
        self.assertFalse(result["endpoint_scoring_authority"])
        self.assertFalse(result["chart_update"])
        self.assertFalse(result["sota_claim"])
        with self.assertRaisesRegex(
            SchemaError,
            "blocked before any write",
        ):
            registration.preflight_or_apply(apply=True)

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


if __name__ == "__main__":
    unittest.main()
