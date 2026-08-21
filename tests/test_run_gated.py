import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools import run_gated


def manifest(**overrides):
    value = {
        "experiment_id": "exp_test",
        "arm_id": "treatment",
        "seed": 42,
        "max_steps": 2_000,
        "env": {"WARMDOWN_RATIO": "0.75"},
        "frozen_env": {
            "STOP_MODE": "steps",
            "ATTN_BACKEND": "sdpa",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        },
        "code_hashes": {
            "train.py": "a" * 64,
            "observable.py": "b" * 64,
        },
        "config_hash": "abc123",
    }
    value.update(overrides)
    return value


class FloorGateTest(unittest.TestCase):
    def test_rejects_sub_floor_or_invalid_noise_models(self):
        for noise_model in (
            {"effective_sigma": 0.000847, "minimum_effect": 0.000847},
            {},
            {"effective_sigma": "unknown", "minimum_effect": 0.01},
        ):
            experiment = SimpleNamespace(
                experiment_id="exp_sub",
                analysis_plan={"noise_model": noise_model},
            )
            with self.subTest(noise_model=noise_model):
                with self.assertRaises(run_gated.FloorGateError):
                    run_gated._floor_gate(experiment)

    def test_accepts_effect_at_twice_the_floor(self):
        experiment = SimpleNamespace(
            experiment_id="exp_ok",
            analysis_plan={
                "noise_model": {
                    "effective_sigma": 0.000847,
                    "minimum_effect": 0.001700,
                }
            },
        )

        run_gated._floor_gate(experiment)


class PackerDiagnosticIntegrityTest(unittest.TestCase):
    def _log(self, *, sidecar=False):
        digest = "a" * 64
        profile_csv = ",".join(["100.000000"] * 200)
        profile_sha = hashlib.sha256(profile_csv.encode("ascii")).hexdigest()
        scanner_batches = 40 if sidecar else 500
        sidecar_batches = 460 if sidecar else 0
        ab_scanner_batches = 40 if sidecar else 80
        ab_sidecar_batches = 40 if sidecar else 0
        return "\n".join(
            [
                "PACKER_BOUNDARY_VERIFY_PASSED=1 batches=1000 "
                f"data_sha256={digest} boundary_sha256={digest} "
                "x_exact=1 y_exact=1 epoch_exact=1 boundaries_exact=1 "
                "first_zero=1 terminal=1 strictly_increasing=1 "
                "row_starts=1 max_gap_lte_t=1",
                "Time budget: 300s",
                "Gradient accumulation steps: 2",
                "PACKER_DIAGNOSTIC_AA_ATTESTED=1 steps=20 "
                f"micro_batches=40 model_sha256={digest} "
                f"optimizer_sha256={digest} loss_sha256={digest} "
                f"data_sha256={digest} boundary_sha256={digest} "
                f"cpu_rng_sha256={digest} cuda_rng_sha256={digest} "
                "dynamo_graph_count=1",
                "PACKER_SIDECAR_PHASE_ACTIVATED=1 step=20 "
                f"sidecar_active={int(sidecar)} "
                f"registered_sidecar_arm={int(sidecar)}",
                "PACKER_DIAGNOSTIC_AB_ATTESTED=1 "
                f"start_step=20 end_step=40 "
                f"mode={'sidecar' if sidecar else 'scanner'} "
                f"model_sha256={digest} optimizer_sha256={digest} "
                f"loss_sha256={digest} data_sha256={digest} "
                f"boundary_sha256={digest} cpu_rng_sha256={digest} "
                f"cuda_rng_sha256={digest} "
                f"scanner_micro_batches={ab_scanner_batches} "
                f"sidecar_micro_batches={ab_sidecar_batches}",
                "PACKER_DIAGNOSTIC_GRAPH_ATTESTED=1 "
                "dynamo_graph_count=1 dynamo_recompile_count=0 "
                "lazy_backward_count=1 cudagraph_recording_count=0 "
                f"compile_ids_sha256={digest} compile_sites=0:0 "
                "activation_step=20 clean_profile_start_step=50",
                "PACKER_DIAGNOSTIC_PROFILE_ATTESTED=1 start_step=50 "
                f"samples=200 expected_samples=200 sha256={profile_sha}",
                f"PACKER_DIAGNOSTIC_PROFILE_MS={profile_csv}",
                "PACKER_DIAGNOSTIC_HASHES_VERIFIED=1 steps=250 "
                f"model_sha256={digest} optimizer_sha256={digest} "
                f"aa_loss_sha256={digest} aa_boundary_sha256={digest} "
                f"ab_loss_sha256={digest} ab_data_sha256={digest} "
                f"ab_boundary_sha256={digest} tail_loss_sha256={digest} "
                f"aa_data_sha256={digest} "
                f"scanner_micro_batches={scanner_batches} "
                f"sidecar_micro_batches={sidecar_batches}",
            ]
        )

    def test_packer_diagnostic_semantic_markers_pass_and_are_extracted(self):
        selected = manifest(
            max_steps=250,
            env={"PACKER_DOC_BOUNDARIES": "1"},
            packer_boundary_parity=True,
        )

        verified, facts, errors = (
            run_gated.verify_packer_boundary_diagnostic(
                selected,
                self._log(sidecar=True),
            )
        )

        self.assertTrue(verified)
        self.assertEqual(errors, [])
        self.assertEqual(facts["profile_samples"], 200)
        self.assertEqual(facts["scanner_micro_batches"], 40)
        self.assertEqual(facts["sidecar_micro_batches"], 460)

    def test_packer_diagnostic_rejects_activation_or_digest_drift(self):
        selected = manifest(
            max_steps=250,
            env={"PACKER_DOC_BOUNDARIES": "0"},
            packer_boundary_parity=True,
        )
        bad = self._log(sidecar=False).replace(
            "sidecar_active=0", "sidecar_active=1"
        )

        verified, _facts, errors = (
            run_gated.verify_packer_boundary_diagnostic(selected, bad)
        )

        self.assertFalse(verified)
        self.assertTrue(
            any("activation marker" in error for error in errors)
        )

    def test_packer_diagnostic_rejects_251_step_manifest(self):
        selected = manifest(
            max_steps=251,
            env={"PACKER_DOC_BOUNDARIES": "0"},
            packer_boundary_parity=True,
        )
        bad = self._log(sidecar=False).replace(
            "samples=200 expected_samples=200",
            "samples=201 expected_samples=201",
        ).replace(
            "PACKER_DIAGNOSTIC_HASHES_VERIFIED=1 steps=250",
            "PACKER_DIAGNOSTIC_HASHES_VERIFIED=1 steps=251",
        )
        profile_csv = ",".join(["100.000000"] * 201)
        profile_sha = hashlib.sha256(profile_csv.encode("ascii")).hexdigest()
        bad = bad.replace(
            "PACKER_DIAGNOSTIC_PROFILE_MS="
            + ",".join(["100.000000"] * 200),
            "PACKER_DIAGNOSTIC_PROFILE_MS=" + profile_csv,
        )
        bad = bad.replace(
            "sha256="
            + hashlib.sha256(
                ",".join(["100.000000"] * 200).encode("ascii")
            ).hexdigest(),
            "sha256=" + profile_sha,
        )

        verified, facts, errors = (
            run_gated.verify_packer_boundary_diagnostic(selected, bad)
        )

        self.assertFalse(verified)
        self.assertEqual(facts["profile_samples"], 201)
        self.assertTrue(
            any("exactly 250 steps" in error for error in errors)
        )
        self.assertTrue(
            any("clean profile" in error for error in errors)
        )


class ConfigBindingTest(unittest.TestCase):
    def test_legacy_direct_environment_parameters_remain_executable(self):
        experiment = SimpleNamespace(
            experiment_id="exp_test",
            arms=(
                {
                    "arm_id": "treatment",
                    "role": "treatment",
                    "intervention_id": "int_fa3",
                },
            ),
        )
        intervention = SimpleNamespace(
            parameters={
                "ATTN_BACKEND": "fa3",
                "dependency": "descriptive metadata, not an environment variable",
            }
        )
        registry = SimpleNamespace(
            interventions=SimpleNamespace(
                by_id=lambda: {"int_fa3": intervention}
            )
        )

        env, role, intervention_id = run_gated._arm_env(
            registry, experiment, "treatment"
        )

        self.assertEqual(env, {"ATTN_BACKEND": "fa3"})
        self.assertEqual(role, "treatment")
        self.assertEqual(intervention_id, "int_fa3")

    def test_interventions_cannot_supply_packer_diagnostic_controls(self):
        experiment = SimpleNamespace(
            experiment_id="exp_test",
            arms=(
                {
                    "arm_id": "treatment",
                    "role": "treatment",
                    "intervention_id": "int_bad_diagnostic",
                },
            ),
        )
        intervention = SimpleNamespace(
            parameters={
                "env": {
                    "PACKER_BOUNDARY_VERIFY_BATCHES": "7",
                    "PACKER_DIAGNOSTIC_HASHES": "0",
                }
            }
        )
        registry = SimpleNamespace(
            interventions=SimpleNamespace(
                by_id=lambda: {"int_bad_diagnostic": intervention}
            )
        )

        with self.assertRaisesRegex(ValueError, "diagnostic-only controls"):
            run_gated._arm_env(
                registry,
                experiment,
                "treatment",
            )

    def test_direct_remote_execution_is_rejected(self):
        with patch(
            "sys.argv",
            [
                "run_gated.py",
                "exp_test",
                "control",
                "--seed",
                "42",
                "--ssh",
                "ssh host",
            ],
        ):
            with self.assertRaisesRegex(SystemExit, "tools/run_stage.py"):
                run_gated.main()

    def test_direct_diagnostic_execution_requires_live_stage_capability(self):
        with patch(
            "sys.argv",
            [
                "run_gated.py",
                "exp_test",
                "control",
                "--seed",
                "42",
                "--diagnostic-steps",
                "20",
                "--diagnostic-source-hashes-sha256",
                "a" * 64,
            ],
        ):
            with self.assertRaisesRegex(
                SystemExit, "internal to tools/run_stage.py"
            ):
                run_gated.main()

    def test_packer_boundary_parity_cannot_be_enabled_directly_or_without_steps(self):
        with patch(
            "sys.argv",
            [
                "run_gated.py",
                "exp_test",
                "control",
                "--seed",
                "42",
                "--packer-boundary-parity",
            ],
        ):
            with self.assertRaisesRegex(
                SystemExit, "requires --diagnostic-steps"
            ):
                run_gated.main()

        with patch(
            "sys.argv",
            [
                "run_gated.py",
                "exp_test",
                "control",
                "--seed",
                "42",
                "--diagnostic-steps",
                "20",
                "--diagnostic-source-hashes-sha256",
                "a" * 64,
                "--packer-boundary-parity",
            ],
        ):
            with self.assertRaisesRegex(
                SystemExit, r"diagnostic-steps exactly 250"
            ):
                run_gated.main()

        with patch(
            "sys.argv",
            [
                "run_gated.py",
                "exp_test",
                "control",
                "--seed",
                "42",
                "--diagnostic-steps",
                "251",
                "--diagnostic-source-hashes-sha256",
                "a" * 64,
                "--packer-boundary-parity",
            ],
        ):
            with self.assertRaisesRegex(
                SystemExit, r"diagnostic-steps exactly 250"
            ):
                run_gated.main()

        with patch(
            "sys.argv",
            [
                "run_gated.py",
                "exp_test",
                "control",
                "--seed",
                "42",
                "--diagnostic-steps",
                "250",
                "--diagnostic-source-hashes-sha256",
                "a" * 64,
                "--packer-boundary-parity",
            ],
        ):
            with self.assertRaisesRegex(
                SystemExit, "internal to tools/run_stage.py"
            ):
                run_gated.main()

    def test_frozen_env_is_derived_from_scope_without_inventing_a_regime(self):
        self.assertEqual(
            run_gated._frozen_env_from_scope_key({"stop_mode": "steps"}),
            {
                "STOP_MODE": "steps",
                "ATTN_BACKEND": "sdpa",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            },
        )
        self.assertEqual(
            run_gated._frozen_env_from_scope_key(
                {"stop_mode": "time", "time_budget": 375}
            ),
            {
                "STOP_MODE": "time",
                "ATTN_BACKEND": "sdpa",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "TIME_BUDGET": "375",
            },
        )

    def test_time_scope_requires_explicit_positive_budget(self):
        for scope in (
            {"stop_mode": "time"},
            {"stop_mode": "walltime", "time_budget": 0},
            {"stop_mode": "unknown", "time_budget": 300},
        ):
            with self.subTest(scope=scope):
                with self.assertRaises(RuntimeError):
                    run_gated._frozen_env_from_scope_key(scope)

    def test_frame_environment_is_bound_on_top_of_budget_controls(self):
        env = run_gated._frozen_env_from_frame(
            {
                "scope_key": {"stop_mode": "time", "time_budget": 300},
                "run_env": {
                    "ATTN_BACKEND": "fa3",
                    "COMPILE_MODE": "max-autotune-no-cudagraphs",
                },
            }
        )
        self.assertEqual(env["STOP_MODE"], "time")
        self.assertEqual(env["TIME_BUDGET"], "300")
        self.assertEqual(env["ATTN_BACKEND"], "fa3")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(env["COMPILE_MODE"], "max-autotune-no-cudagraphs")

    def test_diagnostic_environment_changes_only_the_stop_rule(self):
        frame = {
            "scope_key": {
                "stop_mode": "time",
                "time_budget": 300,
            },
            "run_env": {
                "ATTN_BACKEND": "fa3",
                "COMPILE_MODE": "max-autotune-no-cudagraphs",
            },
        }

        env = run_gated._diagnostic_env_from_frame(frame)

        self.assertEqual(env["STOP_MODE"], "steps")
        self.assertEqual(env["TIME_BUDGET"], "300")
        self.assertEqual(env["ATTN_BACKEND"], "fa3")
        self.assertEqual(env["COMPILE_MODE"], "max-autotune-no-cudagraphs")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_diagnostic_manifest_authorizes_live_frame_before_budget_override(self):
        frame = {
            "scope_id": "walltime_5min_h200",
            "status": "passed",
            "scope_key": {
                "stop_mode": "time",
                "time_budget": 300,
                "max_steps": 100_000,
            },
            "run_env": {
                "ATTN_BACKEND": "fa3",
                "COMPILE_MODE": "max-autotune-no-cudagraphs",
            },
            "baseline": {"effective_sigma": 0.01},
        }
        setup = SimpleNamespace(
            fingerprint="setup-fingerprint",
            scope_for=lambda scope_id: frame,
        )
        experiment = SimpleNamespace(
            experiment_id="exp_diag",
            hypothesis_id="hyp_diag",
            hypothesis_fingerprint="hypothesis-fingerprint",
            fingerprint="experiment-fingerprint",
            search_policy={},
            analysis_plan={
                "noise_model": {
                    "effective_sigma": 0.01,
                    "minimum_effect": 0.02,
                }
            },
        )
        authorize_run = MagicMock(
            return_value={
                "setup_fingerprint": "authorized-setup",
                "experiment_fingerprint": "authorized-experiment",
            }
        )
        registry = SimpleNamespace(
            setup_reconciliation=lambda: setup,
            challenge_for_scope=lambda scope_id: {
                "challenge_id": "challenge-live",
                "decision_frame": "walltime_5min_h200",
            },
            challenge_events=lambda: [
                SimpleNamespace(fingerprint="selection-fingerprint")
            ],
            gated_experiments=SimpleNamespace(
                by_id=lambda: {"exp_diag": experiment}
            ),
            authorize_run=authorize_run,
            hypotheses=SimpleNamespace(
                by_id=lambda: {
                    "hyp_diag": SimpleNamespace(
                        outcome={"outcome_id": "out_val_bpb"}
                    )
                }
            ),
            outcomes=SimpleNamespace(
                by_id=lambda: {
                    "out_val_bpb": SimpleNamespace(
                        fingerprint="outcome-fingerprint"
                    )
                }
            ),
        )

        def arm_env(_registry, _experiment, arm_id):
            boundary_mode = "0" if arm_id == "control" else "1"
            role = "control" if arm_id == "control" else "treatment"
            return (
                {"PACKER_DOC_BOUNDARIES": boundary_mode},
                role,
                f"int_{arm_id}",
            )

        with (
            patch.object(
                run_gated,
                "validate_search_policy",
                return_value={"qualification": {}},
            ),
            patch.object(
                run_gated,
                "challenge_fingerprint",
                return_value="challenge-fingerprint",
            ),
            patch.object(
                run_gated,
                "_arm_env",
                side_effect=arm_env,
            ),
            patch.object(run_gated, "_sha", return_value="a" * 64),
        ):
            bound, _ = run_gated.build_manifest(
                Path("/repo"),
                registry,
                "exp_diag",
                "treatment",
                63,
                diagnostic_steps=20,
                scope_id="walltime_5min_h200",
            )
            packer_control, _ = run_gated.build_manifest(
                Path("/repo"),
                registry,
                "exp_diag",
                "control",
                63,
                diagnostic_steps=250,
                scope_id="walltime_5min_h200",
                packer_boundary_parity=True,
            )
            packer_treatment, _ = run_gated.build_manifest(
                Path("/repo"),
                registry,
                "exp_diag",
                "treatment",
                63,
                diagnostic_steps=250,
                scope_id="walltime_5min_h200",
                packer_boundary_parity=True,
            )

        authorize_run.assert_any_call(
            "exp_diag",
            "treatment",
            63,
            100_000,
            scope_id="walltime_5min_h200",
        )
        self.assertEqual(bound["authorized_max_steps"], 100_000)
        self.assertEqual(bound["max_steps"], 20)
        self.assertTrue(bound["diagnostic"])
        self.assertTrue(bound["non_scored"])
        self.assertEqual(bound["diagnostic_kind"], "paired_fixed_step_parity")
        self.assertEqual(bound["frozen_env"]["STOP_MODE"], "steps")
        self.assertEqual(bound["frozen_env"]["TIME_BUDGET"], "300")
        self.assertEqual(
            bound["code_hashes_sha256"],
            run_gated.code_hashes_fingerprint(bound["code_hashes"]),
        )
        self.assertNotIn("packer_boundary_parity", bound)
        self.assertNotIn("diagnostic_env", bound)
        self.assertEqual(
            packer_control["diagnostic_env"],
            run_gated.PACKER_BOUNDARY_PARITY_ENV,
        )
        self.assertEqual(
            packer_control["diagnostic_env"],
            packer_treatment["diagnostic_env"],
        )
        self.assertEqual(
            packer_control["env"]["PACKER_DOC_BOUNDARIES"],
            "0",
        )
        self.assertEqual(
            packer_treatment["env"]["PACKER_DOC_BOUNDARIES"],
            "1",
        )
        self.assertEqual(
            packer_control["diagnostic_kind"],
            "packer_boundary_parity",
        )
        self.assertTrue(packer_control["packer_boundary_parity"])
        self.assertEqual(
            packer_control["diagnostic_env"][
                "PACKER_SIDECAR_ACTIVATE_STEP"
            ],
            "20",
        )

    def test_implementation_pilot_uses_diagnostic_authority_without_effect_floor(self):
        frame = {
            "scope_id": "walltime_5min_h200",
            "status": "passed",
            "scope_key": {
                "stop_mode": "time",
                "time_budget": 300,
                "max_steps": 100_000,
            },
            "run_env": {
                "ATTN_BACKEND": "fa3",
                "COMPILE_MODE": "max-autotune-no-cudagraphs",
            },
            "baseline": {"effective_sigma": 0.01},
        }
        setup = SimpleNamespace(
            fingerprint="setup-fingerprint",
            scope_for=lambda scope_id: frame,
        )
        experiment = SimpleNamespace(
            experiment_id="exp_diag_pilot",
            hypothesis_id="hyp_diag",
            hypothesis_fingerprint="hypothesis-fingerprint",
            fingerprint="experiment-fingerprint",
            stage="pilot",
            search_policy={},
            analysis_plan={
                "diagnostic_only": True,
                "endpoint_val_bpb_measured": False,
            },
        )
        authorize_run = MagicMock(
            return_value={
                "setup_fingerprint": "authorized-setup",
                "experiment_fingerprint": "authorized-experiment",
            }
        )
        registry = SimpleNamespace(
            setup_reconciliation=lambda: setup,
            challenge_for_scope=lambda scope_id: {
                "challenge_id": "challenge-live",
                "decision_frame": "walltime_5min_h200",
            },
            challenge_events=lambda: [
                SimpleNamespace(fingerprint="selection-fingerprint")
            ],
            gated_experiments=SimpleNamespace(
                by_id=lambda: {"exp_diag_pilot": experiment}
            ),
            authorize_run=authorize_run,
            hypotheses=SimpleNamespace(
                by_id=lambda: {
                    "hyp_diag": SimpleNamespace(
                        outcome={"outcome_id": "out_val_bpb"}
                    )
                }
            ),
            outcomes=SimpleNamespace(
                by_id=lambda: {
                    "out_val_bpb": SimpleNamespace(
                        fingerprint="outcome-fingerprint"
                    )
                }
            ),
        )
        validate_policy = MagicMock(
            side_effect=AssertionError(
                "diagnostic pilot must not enter the effect search policy"
            )
        )
        floor_gate = MagicMock(
            side_effect=AssertionError(
                "diagnostic pilot must not enter the effect floor"
            )
        )

        with (
            patch.object(run_gated, "validate_search_policy", validate_policy),
            patch.object(run_gated, "_floor_gate", floor_gate),
            patch.object(
                run_gated,
                "challenge_fingerprint",
                return_value="challenge-fingerprint",
            ),
            patch.object(
                run_gated,
                "_arm_env",
                return_value=(
                    {"GPAS_ENABLE": "1"},
                    "treatment",
                    "int_gpas",
                ),
            ),
            patch.object(run_gated, "_sha", return_value="a" * 64),
        ):
            bound, _ = run_gated.build_manifest(
                Path("/repo"),
                registry,
                "exp_diag_pilot",
                "treatment",
                66,
                diagnostic_steps=256,
                scope_id="walltime_5min_h200",
            )

        authorize_run.assert_called_once_with(
            "exp_diag_pilot",
            "treatment",
            66,
            100_000,
            scope_id="walltime_5min_h200",
            diagnostic_only=True,
        )
        validate_policy.assert_not_called()
        floor_gate.assert_not_called()
        self.assertTrue(bound["diagnostic"])
        self.assertTrue(bound["non_scored"])
        self.assertEqual(bound["max_steps"], 256)

    def test_scheduler_request_binds_diagnostic_budget_and_source_snapshot(self):
        args = SimpleNamespace(
            experiment_id="exp_diag",
            arm_id="control",
            seed=63,
            root="research",
            gpu=0,
            ssh="ssh host",
            remote_wd="/remote/work",
            python_bin="python",
            scope="walltime_5min_h200",
            challenge_id="challenge_live",
            challenge_selection_fingerprint="selection-fingerprint",
            challenge_definition_fingerprint="definition-fingerprint",
            diagnostic_steps=20,
            diagnostic_source_hashes_sha256="a" * 64,
            qualification_runtime_attestation_sha256="",
        )

        request = run_gated._scheduler_request(args)

        self.assertEqual(request["diagnostic_steps"], 20)
        self.assertEqual(
            request["diagnostic_source_hashes_sha256"],
            "a" * 64,
        )
        self.assertNotIn("packer_boundary_parity", request)

        args.diagnostic_steps = 250
        args.packer_boundary_parity = True
        packer_request = run_gated._scheduler_request(args)
        self.assertTrue(packer_request["packer_boundary_parity"])

    def test_ordinary_scheduler_request_has_no_diagnostic_source_capability(self):
        args = SimpleNamespace(
            experiment_id="exp_ordinary",
            arm_id="control",
            seed=63,
            root="research",
            gpu=0,
            ssh="ssh host",
            remote_wd="/remote/work",
            python_bin="python",
            scope="walltime_5min_h200",
            challenge_id="challenge_live",
            challenge_selection_fingerprint="selection-fingerprint",
            challenge_definition_fingerprint="definition-fingerprint",
            diagnostic_steps=0,
            diagnostic_source_hashes_sha256="",
            qualification_runtime_attestation_sha256="",
        )

        request = run_gated._scheduler_request(args)

        self.assertEqual(request["diagnostic_steps"], 0)
        self.assertNotIn("diagnostic_source_hashes_sha256", request)

    def test_diagnostic_record_is_durably_non_scored_and_auditable(self):
        bound = manifest(
            hypothesis_id="hyp_diag",
            role="treatment",
            spec_fingerprints={
                "experiment": "experiment-fingerprint",
                "hypothesis": "hypothesis-fingerprint",
                "outcome": "outcome-fingerprint",
            },
            challenge_id="challenge_live",
            challenge_definition_fingerprint="a" * 16,
            challenge_selection_fingerprint="b" * 16,
            scope_id="walltime_5min_h200",
            diagnostic=True,
            non_scored=True,
            diagnostic_kind="paired_fixed_step_parity",
            authorized_max_steps=100_000,
            max_steps=20,
            code_hashes_sha256="c" * 64,
        )

        record = run_gated.make_run_record(
            bound,
            "/remote/diag.log",
            1.2,
            True,
            "complete",
            "deadbeef",
            False,
            "2026-07-29T14:00:00+00:00",
            "2026-07-29T14:01:00+00:00",
            gpu_id=0,
            gpu_uuid="GPU-test",
            num_steps=20,
            total_tokens=2_949_120,
            charged_training_seconds=1.0,
            execution_isolated=True,
            resolved_training_config={"MAX_STEPS": "20"},
        )

        self.assertIn("diagnostic_offbudget_20steps", record.tags)
        self.assertIn("diagnostic_non_scored", record.tags)
        self.assertIn("diagnostic_pair_symmetric", record.tags)
        self.assertTrue(record.tracker["non_scored"])
        self.assertEqual(record.tracker["authorized_max_steps"], 100_000)
        self.assertEqual(record.tracker["executed_max_steps"], 20)
        self.assertEqual(record.tracker["code_hashes_sha256"], "c" * 64)

    def test_verifies_frozen_controls_and_arm_environment(self):
        resolved = {
            "STOP_MODE": "steps",
            "ATTN_BACKEND": "sdpa",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "MAX_STEPS": "2000",
            "SEED": "42",
            "WARMDOWN_RATIO": "0.75",
        }

        verified, mismatches = run_gated.verify_config(manifest(), resolved)

        self.assertTrue(verified)
        self.assertEqual(mismatches, [])

    def test_registered_arm_can_override_a_non_budget_frame_default(self):
        selected = manifest(env={"ATTN_BACKEND": "fa3"})
        resolved = {
            "STOP_MODE": "steps",
            "ATTN_BACKEND": "fa3",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "MAX_STEPS": "2000",
            "SEED": "42",
        }

        verified, mismatches = run_gated.verify_config(selected, resolved)

        self.assertTrue(verified)
        self.assertEqual(mismatches, [])

    def test_rejects_mismatched_frozen_control(self):
        resolved = {
            "STOP_MODE": "time",
            "ATTN_BACKEND": "sdpa",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "MAX_STEPS": "2000",
            "SEED": "42",
            "WARMDOWN_RATIO": "0.75",
        }

        verified, mismatches = run_gated.verify_config(manifest(), resolved)

        self.assertFalse(verified)
        self.assertTrue(any(item.startswith("STOP_MODE:") for item in mismatches))

    def test_manifest_cannot_override_run_controls(self):
        for key in run_gated.RUN_CONTROL_ENV:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "may not override"):
                    run_gated.launch_command(
                        manifest(env={key: "unsafe"}),
                        "/remote/work",
                        0,
                        "python",
                    )

    def test_packer_diagnostic_env_is_fixed_verified_and_isolated(self):
        selected = manifest(
            env={"PACKER_DOC_BOUNDARIES": "1"},
            max_steps=250,
            diagnostic=True,
            packer_boundary_parity=True,
            diagnostic_env=dict(run_gated.PACKER_BOUNDARY_PARITY_ENV),
        )
        resolved = run_gated._bound_env(selected)

        self.assertEqual(resolved["PACKER_DOC_BOUNDARIES"], "1")
        self.assertEqual(
            resolved["PACKER_BOUNDARY_VERIFY_BATCHES"],
            "1000",
        )
        self.assertEqual(resolved["PACKER_DIAGNOSTIC_HASHES"], "1")
        self.assertEqual(resolved["PACKER_SIDECAR_ACTIVATE_STEP"], "20")
        verified, mismatches = run_gated.verify_config(selected, resolved)
        self.assertTrue(verified)
        self.assertEqual(mismatches, [])

        incomplete = dict(resolved)
        incomplete.pop("PACKER_DIAGNOSTIC_HASHES")
        verified, mismatches = run_gated.verify_config(
            selected,
            incomplete,
        )
        self.assertFalse(verified)
        self.assertTrue(
            any(
                item.startswith("PACKER_DIAGNOSTIC_HASHES:")
                for item in mismatches
            )
        )

        general_diagnostic = manifest(
            max_steps=20,
            diagnostic=True,
        )
        general_env = run_gated._bound_env(general_diagnostic)
        self.assertNotIn("PACKER_BOUNDARY_VERIFY_BATCHES", general_env)
        self.assertNotIn("PACKER_DIAGNOSTIC_HASHES", general_env)
        self.assertNotIn("PACKER_SIDECAR_ACTIVATE_STEP", general_env)

        with self.assertRaisesRegex(
            ValueError, "allowed only.*packer-boundary"
        ):
            run_gated._bound_env(
                manifest(
                    diagnostic_env=dict(
                        run_gated.PACKER_BOUNDARY_PARITY_ENV
                    )
                )
            )
        with self.assertRaisesRegex(
            ValueError, "diagnostic-only controls"
        ):
            run_gated._bound_env(
                manifest(env={"PACKER_DIAGNOSTIC_HASHES": "1"})
            )

    def test_manifest_cannot_change_frozen_environment(self):
        with self.assertRaisesRegex(ValueError, "frozen_env must equal"):
            run_gated.verify_config(
                manifest(
                    frozen_env={
                        "STOP_MODE": "time",
                        "ATTN_BACKEND": "sdpa",
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    }
                ),
                {},
            )

    def test_launch_verifies_bound_code_before_training(self):
        command, _ = run_gated.launch_command(
            manifest(),
            "/remote/work",
            0,
            "python",
        )

        self.assertIn("sha256sum train.py", command)
        self.assertIn("sha256sum observable.py", command)
        self.assertLess(command.index("sha256sum train.py"), command.index("python -u train.py"))

    def test_launch_binds_physical_gpu_and_samples_process_membership(self):
        command, _ = run_gated.launch_command(
            manifest(),
            "/remote/work",
            4,
            "python",
        )

        self.assertIn("run-abc123.lock", command)
        self.assertIn("gpu-4.lock", command)
        self.assertIn("flock -n", command)
        self.assertIn("GPU_ADVISORY_LOCK_ACQUIRED=4", command)
        self.assertIn("--query-gpu=uuid", command)
        self.assertIn("CUDA_DEVICE_ORDER=PCI_BUS_ID", command)
        self.assertIn('CUDA_VISIBLE_DEVICES=\"$gpu_uuid\"', command)
        self.assertIn("GPU_UUID_BOUND=", command)
        self.assertIn("GPU_TRAIN_PID_OBSERVED=", command)
        self.assertIn("GPU_TRAIN_PID_NEVER_OBSERVED=", command)
        self.assertIn("GPU_COTENANCY_SAMPLING_VERIFIED=1", command)
        self.assertIn("TRAIN_PROCESS_STARTED=", command)
        self.assertIn("--query-compute-apps=pid", command)
        self.assertIn('kill "$train_pid"', command)

    def test_runner_config_is_merged_with_training_config(self):
        parsed = run_gated.parse_resolved(
            "RESOLVED_CONFIG: STOP_MODE=steps MAX_STEPS=2000\n"
            "RUNNER_CONFIG: CUDA_DEVICE_ORDER=PCI_BUS_ID "
            "CUDA_VISIBLE_DEVICES=GPU-deadbeef GPU_UUID=GPU-deadbeef\n"
        )

        self.assertEqual(parsed["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(parsed["CUDA_VISIBLE_DEVICES"], "GPU-deadbeef")

    def test_exact_outcome_metrics_are_parsed_without_rounded_token_millions(self):
        parsed = run_gated.parse_outcome_metrics(
            "val_bpb:          0.927183\n"
            "total_tokens_M:   298.2\n"
            "total_tokens:     298156032\n"
            "charged_training_seconds: 300.012345678\n"
            "num_steps:        2022\n"
        )

        self.assertEqual(
            parsed,
            {
                "val_bpb": 0.927183,
                "num_steps": 2022,
                "total_tokens": 298156032,
                "charged_training_seconds": 300.012345678,
            },
        )

    def test_incomplete_exact_outcome_metrics_remain_absent(self):
        parsed = run_gated.parse_outcome_metrics(
            "val_bpb: 0.93\nnum_steps: 100\n"
        )

        self.assertNotIn("total_tokens", parsed)
        self.assertNotIn("charged_training_seconds", parsed)


if __name__ == "__main__":
    unittest.main()


class DualScopeTest(unittest.TestCase):
    """The 5-minute wall-clock frame is registered, runnable, and never adopt-grade."""

    def test_scope_key_accepts_wall_clock_and_requires_a_budget(self):
        from vibeautoresearch.setup import validate_scope_key
        from vibeautoresearch.core import SchemaError

        ok = validate_scope_key({
            "data_split_sha256": "a" * 64, "max_steps": 100000,
            "stop_mode": "time", "time_budget": 300, "outcome_id": "out_val_bpb",
        })
        self.assertEqual(ok["time_budget"], 300)

        for bad in (
            {"data_split_sha256": "a" * 64, "max_steps": 100000,
             "stop_mode": "time", "outcome_id": "out_val_bpb"},
            {"data_split_sha256": "a" * 64, "max_steps": 2000, "stop_mode": "steps",
             "time_budget": 300, "outcome_id": "out_val_bpb"},
        ):
            with self.assertRaises(SchemaError):
                validate_scope_key(bad)

    def test_registered_five_minute_scope_is_a_peer_frame(self):
        from pathlib import Path as _P
        from vibeautoresearch.registry import ResearchRegistry

        setup = ResearchRegistry(
            _P(__file__).resolve().parents[1] / "research"
        ).setup_reconciliation()
        entry = setup.scope_for("walltime_5min_h200")
        self.assertEqual(entry["role"], "adopt")
        self.assertGreaterEqual(int(entry["min_seeds"]), 10)
        # A frame may only claim 'passed' once it carries its own measured baseline
        # with at least min_seeds distinct seeds.
        if entry["status"] == "passed":
            seeds = entry["baseline"]["seeds"]
            self.assertGreaterEqual(len(seeds), int(entry["min_seeds"]))
            self.assertEqual(len(set(seeds)), len(seeds))
            self.assertGreater(entry["baseline"]["effective_sigma"], 0)
        self.assertEqual(entry["scope_key"]["stop_mode"], "time")
        self.assertEqual(entry["scope_key"]["time_budget"], 300)
        # The adopt frame must stay step-budgeted.
        self.assertEqual(setup.scope_key["stop_mode"], "steps")
        self.assertEqual(setup.scope_key["max_steps"], 2000)

    def test_unmeasured_frame_cannot_authorize_a_run(self):
        """A frame without a measured baseline must refuse, whatever the live record says."""

        class _Setup:
            def scope_for(self, scope_id):
                return {"scope_id": scope_id, "status": "pending", "min_seeds": 10}

        with self.assertRaises(RuntimeError) as ctx:
            run_gated._resolve_decision_frame(_Setup(), "unmeasured_frame")
        message = str(ctx.exception)
        self.assertIn("cannot authorize a run", message)
        self.assertIn(">=10 seeds", message)

    def test_unknown_frame_is_rejected(self):
        from pathlib import Path as _P
        from vibeautoresearch.registry import ResearchRegistry
        from vibeautoresearch.core import SchemaError

        setup = ResearchRegistry(
            _P(__file__).resolve().parents[1] / "research"
        ).setup_reconciliation()
        with self.assertRaises(SchemaError):
            run_gated._resolve_report_scope(setup, "no_such_scope")

    def test_frame_derives_a_wall_clock_env(self):
        env = run_gated._frozen_env_from_scope_key(
            {"stop_mode": "time", "time_budget": 300}, "test"
        )
        self.assertEqual(env["STOP_MODE"], "time")
        self.assertEqual(env["TIME_BUDGET"], "300")
