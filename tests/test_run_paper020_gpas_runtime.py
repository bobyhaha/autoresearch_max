import copy
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from tools import run_paper020_gpas_runtime as runner


class Paper020GPASRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = runner.REPO_ROOT
        cls.worker_relative = runner.EXPECTED_WORKER_RELATIVE_PATH
        cls.worker_sha = "e" * 64
        cls.runner_sha = runner._sha256(
            cls.root / runner.RUNNER_RELATIVE_PATH
        )
        # Authority parsing needs a syntactically valid full commit ID, not the
        # live repository identity.  Keep this unit-test fixture runnable in
        # the pre-commit index snapshot, which intentionally has no .git dir.
        cls.git_commit = "f" * 40

    def authority_payload(self):
        nonce = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        authority_id = f"auth_paper020_gpas_{nonce[:12]}"
        return {
            "schema_version": 1,
            "kind": runner.AUTHORITY_KIND,
            "authority_id": authority_id,
            "experiment_id": runner.EXPERIMENT_ID,
            "experiment_fingerprint": "1" * 16,
            "git_commit": self.git_commit,
            "seed": runner.DIAGNOSTIC_SEED,
            "diagnostic_only": True,
            "validation_data_access": False,
            "endpoint_scoring_authorized": False,
            "sota_update_authorized": False,
            "single_use_nonce": nonce,
            "bindings": {
                "files": dict(runner.BOUND_FILES),
                "runner_sha256": self.runner_sha,
                "setup_version": runner.SETUP_VERSION,
                "setup_fingerprint": runner.SETUP_FINGERPRINT,
                "challenge_selection_fingerprint": (
                    runner.CHALLENGE_SELECTION_FINGERPRINT
                ),
            },
            "governance": {
                "scope_id": "walltime_5min_h200",
                "frame_max_steps": 100000,
                "gated_ledger_path": (
                    "research/experiments/gated/experiments.jsonl"
                ),
                "gated_ledger_sha256": runner._sha256(
                    self.root
                    / "research/experiments/gated/experiments.jsonl"
                ),
                "control_arm_id": "gpas_disabled_control",
                "treatment_arm_id": "gpas_shared_gate_treatment",
                "control_authorization_sha256": "b" * 64,
                "treatment_authorization_sha256": "c" * 64,
                "diagnostic_policy_sha256": "d" * 64,
            },
            "recipe_env": dict(runner.CURRENT_R0_ENV),
            "packer_env": dict(runner.PACKER_ZERO_ENV),
            "gpus": {
                "replay": {
                    "index": 0,
                    "uuid": "GPU-aaaaaaaa",
                    "product": "NVIDIA H200",
                },
                "timing_a": {
                    "index": 0,
                    "uuid": "GPU-aaaaaaaa",
                    "product": "NVIDIA H200",
                },
                "timing_b": {
                    "index": 1,
                    "uuid": "GPU-bbbbbbbb",
                    "product": "NVIDIA H200",
                },
            },
            "remote": {
                "ssh_argv": [
                    "ssh",
                    "-p",
                    "50002",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    f"UserKnownHostsFile={runner.EXPECTED_KNOWN_HOSTS_PATH}",
                    "-o",
                    "ConnectTimeout=15",
                    "-o",
                    "ConnectionAttempts=1",
                    "-o",
                    "ServerAliveInterval=15",
                    "-o",
                    "ServerAliveCountMax=3",
                    runner.EXPECTED_REMOTE_TARGET,
                ],
                "scp_argv": [
                    "scp",
                    "-P",
                    "50002",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    f"UserKnownHostsFile={runner.EXPECTED_KNOWN_HOSTS_PATH}",
                    "-o",
                    "ConnectTimeout=15",
                    "-o",
                    "ConnectionAttempts=1",
                    "-o",
                    "ServerAliveInterval=15",
                    "-o",
                    "ServerAliveCountMax=3",
                ],
                "scp_target": runner.EXPECTED_REMOTE_TARGET,
                "work_dir": runner.EXPECTED_REMOTE_WORK_DIR,
                "python": runner.EXPECTED_REMOTE_PYTHON,
                "python_sha256": "a" * 64,
                "known_hosts_path": runner.EXPECTED_KNOWN_HOSTS_PATH,
                "known_hosts_sha256": runner._sha256(
                    Path(runner.EXPECTED_KNOWN_HOSTS_PATH)
                ),
            },
            "worker": {
                "local_path": self.worker_relative,
                "remote_path": (
                    f"{runner.EXPECTED_REMOTE_WORK_DIR}/"
                    "tools/run_paper020_gpas_worker.py"
                ),
                "sha256": self.worker_sha,
                "protocol_version": 1,
                "command_template": list(runner.WORKER_COMMAND_TEMPLATE),
                "command_template_sha256": hashlib.sha256(
                    runner._canonical_json(list(runner.WORKER_COMMAND_TEMPLATE))
                ).hexdigest(),
            },
            "result_dir": (
                "research/experiments/artifacts/paper020_gpas_runtime/"
                f"{authority_id}-{nonce[:16]}"
            ),
        }

    def load_payload(self, payload):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authority.json"
            raw = runner._canonical_json(payload)
            path.write_bytes(raw)
            return runner.load_authority(path, hashlib.sha256(raw).hexdigest())

    def test_exact_current_source_and_recipe_bindings(self):
        observed = {
            relative: runner._sha256(self.root / relative)
            for relative in runner.BOUND_FILES
        }
        self.assertEqual(observed, runner.BOUND_FILES)
        self.assertEqual(
            runner.CURRENT_R0_ENV,
            {
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
            },
        )
        self.assertNotEqual(
            {
                key: runner.CURRENT_R0_ENV.get(key)
                for key in runner.FORBIDDEN_0927_RECIPE
            },
            runner.FORBIDDEN_0927_RECIPE,
        )

    def test_authority_loads_and_dry_run_has_no_launch(self):
        authority = self.load_payload(self.authority_payload())
        result = runner.build_dry_run(authority)
        self.assertEqual(result["mode"], "dry_run")
        self.assertEqual(result["launches_performed"], 0)
        self.assertFalse(result["non_claims"]["endpoint_bpb"])
        self.assertFalse(result["non_claims"]["sota"])
        self.assertEqual(len(result["phases"]), 5)
        self.assertFalse(result["ready"])
        self.assertTrue(
            any(
                "NO_CONFORMING_GPAS_WORKER" in blocker
                for blocker in result["blockers"]
            )
        )
        self.assertFalse(authority["result_dir"].exists())

    def test_historical_recipe_is_rejected_by_name(self):
        payload = self.authority_payload()
        payload["recipe_env"].update(runner.FORBIDDEN_0927_RECIPE)
        with self.assertRaisesRegex(runner.AuthorityError, "0.927183"):
            self.load_payload(payload)

    def test_partial_r0_drift_and_nonzero_packer_are_rejected(self):
        payload = self.authority_payload()
        payload["recipe_env"]["SCALAR_LR"] = "0.5"
        with self.assertRaisesRegex(runner.AuthorityError, "exactly equal current R0"):
            self.load_payload(payload)

        payload = self.authority_payload()
        payload["packer_env"]["PACKER_DOC_BOUNDARIES"] = "1"
        with self.assertRaisesRegex(runner.AuthorityError, "must resolve to zero"):
            self.load_payload(payload)

    def test_authority_requires_canonical_bytes(self):
        payload = self.authority_payload()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authority.json"
            raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            path.write_bytes(raw)
            with self.assertRaisesRegex(
                runner.AuthorityError, "canonical sorted compact JSON"
            ):
                runner.load_authority(
                    path, hashlib.sha256(raw).hexdigest()
                )

    def test_transport_identity_is_noninteractive_and_exact(self):
        mutations = {
            "password prompts": lambda payload: payload["remote"][
                "ssh_argv"
            ].__setitem__(
                payload["remote"]["ssh_argv"].index("BatchMode=yes"),
                "BatchMode=no",
            ),
            "weakened host key": lambda payload: payload["remote"][
                "ssh_argv"
            ].__setitem__(
                payload["remote"]["ssh_argv"].index(
                    "StrictHostKeyChecking=yes"
                ),
                "StrictHostKeyChecking=accept-new",
            ),
            "scp password prompts": lambda payload: payload["remote"][
                "scp_argv"
            ].__setitem__(
                payload["remote"]["scp_argv"].index("BatchMode=yes"),
                "BatchMode=no",
            ),
            "other workdir": lambda payload: payload["remote"].update(
                {"work_dir": "/home/user/other-checkout"}
            ),
            "other python": lambda payload: payload["remote"].update(
                {"python": "/usr/bin/python3"}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                payload = self.authority_payload()
                mutate(payload)
                with self.assertRaises(runner.AuthorityError):
                    self.load_payload(payload)

        payload = self.authority_payload()
        payload["worker"]["local_path"] = "tools/some_other_worker.py"
        with self.assertRaisesRegex(
            runner.AuthorityError, "run_paper020_gpas_worker.py"
        ):
            self.load_payload(payload)

    def test_normal_train_and_helper_are_forbidden_as_worker(self):
        for relative, message in (
            ("train.py", "terminal validation"),
            ("tools/gpas_mechanism_diagnostic.py", "helper"),
        ):
            payload = self.authority_payload()
            payload["worker"]["local_path"] = relative
            payload["worker"]["sha256"] = runner._sha256(self.root / relative)
            payload["worker"]["remote_path"] = (
                f"{runner.EXPECTED_REMOTE_WORK_DIR}/{relative}"
            )
            with self.subTest(relative=relative):
                with self.assertRaisesRegex(runner.AuthorityError, message):
                    self.load_payload(payload)

    def test_gpu_bindings_must_be_fixed_and_physical(self):
        payload = self.authority_payload()
        payload["gpus"]["timing_b"] = {
            "index": 1,
            "uuid": "GPU-aaaaaaaa",
            "product": "NVIDIA H200",
        }
        with self.assertRaisesRegex(runner.AuthorityError, "distinct physical"):
            self.load_payload(payload)

        payload = self.authority_payload()
        payload["gpus"]["replay"] = {
            "index": 7,
            "uuid": "GPU-cccccccc",
            "product": "NVIDIA H200",
        }
        with self.assertRaisesRegex(runner.AuthorityError, "same two fixed"):
            self.load_payload(payload)

        payload = self.authority_payload()
        payload["gpus"]["timing_b"]["product"] = "NVIDIA B200"
        with self.assertRaisesRegex(runner.AuthorityError, "exactly 'NVIDIA H200'"):
            self.load_payload(payload)

        self.assertEqual(
            runner._parse_gpu_inventory(
                "0, GPU-aaaaaaaa, NVIDIA H200\n"
                "1, GPU-bbbbbbbb, NVIDIA H200\n"
            ),
            {
                0: ("GPU-aaaaaaaa", "NVIDIA H200"),
                1: ("GPU-bbbbbbbb", "NVIDIA H200"),
            },
        )
        with self.assertRaisesRegex(
            runner.RuntimeIntegrityError, "invalid/duplicate"
        ):
            runner._parse_gpu_inventory("0, GPU-aaaaaaaa, NVIDIA B200\n")

    def test_phase_plan_freezes_sequential_replay_and_role_swap(self):
        phases = runner.phase_plan()
        self.assertEqual(
            [phase.phase_id for phase in phases],
            [
                "replay_control_step0",
                "replay_treatment_alpha_zero_step0",
                "eager_mediator_pair_256",
                "timing_placement_1",
                "timing_placement_2_role_swapped",
            ],
        )
        self.assertEqual(phases[0].arms[0].gpu_slot, "replay")
        self.assertEqual(phases[1].arms[0].gpu_slot, "replay")
        self.assertEqual(phases[2].steps, 256)
        self.assertEqual(phases[2].mode, "eager_mediator")
        self.assertEqual(
            [(arm.role, arm.gpu_slot) for arm in phases[3].arms],
            [("control", "timing_a"), ("treatment", "timing_b")],
        )
        self.assertEqual(
            [(arm.role, arm.gpu_slot) for arm in phases[4].arms],
            [("treatment", "timing_a"), ("control", "timing_b")],
        )
        for phase in phases[3:]:
            self.assertEqual(phase.timing_start_step, 33)
            self.assertEqual(phase.timing_block_size, 8)

    def test_state_machine_cannot_advance_across_failed_barriers(self):
        phases = runner.phase_plan()
        state = runner.RuntimeStateMachine()
        for phase in phases[:2]:
            state.before_phase(phase)
            state.finish_phase(phase)
        with self.assertRaisesRegex(
            runner.RuntimeIntegrityError, "until exact replay passes"
        ):
            state.before_phase(phases[2])
        state.pass_replay()
        state.before_phase(phases[2])
        state.finish_phase(phases[2])
        with self.assertRaisesRegex(
            runner.RuntimeIntegrityError, "until applicability and mediation pass"
        ):
            state.before_phase(phases[3])
        state.pass_mediator()
        state.before_phase(phases[3])
        state.finish_phase(phases[3])
        with self.assertRaisesRegex(
            runner.RuntimeIntegrityError,
            "until placement-1 compile and row integrity pass",
        ):
            state.before_phase(phases[4])
        state.pass_placement_1_integrity()
        state.before_phase(phases[4])

    def execute_with_mocked_phases(self, phase_result):
        class FakeLock:
            def __enter__(self):
                return {
                    "indices": [0, 1],
                    "same_lock_namespace_as_run_stage": True,
                }

            def __exit__(self, exc_type, exc, traceback):
                return False

        calls = []

        def run_phase(authority, phase, artifact_dir, staging_dir):
            del authority, artifact_dir, staging_dir
            calls.append(phase.phase_id)
            return phase_result(phase), {}

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            payload = self.authority_payload()
            authority_path = temporary_root / "authority.json"
            raw = runner._canonical_json(payload)
            authority_path.write_bytes(raw)
            authority = runner.load_authority(
                authority_path, hashlib.sha256(raw).hexdigest()
            )
            authority["result_dir"] = temporary_root / "results"
            with (
                mock.patch.object(
                    runner, "readiness_blockers", return_value=[]
                ),
                mock.patch.object(
                    runner,
                    "build_dry_run",
                    return_value={"ready": True, "blockers": []},
                ),
                mock.patch.object(
                    runner,
                    "_transaction_gpu_locks",
                    return_value=FakeLock(),
                ),
                mock.patch.object(runner, "_remote_source_preflight"),
                mock.patch.object(
                    runner, "_run_phase", side_effect=run_phase
                ),
                mock.patch.object(
                    runner,
                    "evaluate_applicability",
                    wraps=runner.evaluate_applicability,
                ) as applicability,
            ):
                result = runner.execute(authority, authority["nonce"])
            authority["result_dir"].chmod(0o700)
            return result, calls, applicability.call_count

    def test_malformed_replay_cannot_launch_any_later_phase(self):
        control, treatment = self.step0_results()
        control.pop("parity")

        def phase_result(phase):
            if phase.phase_id == "replay_control_step0":
                return {"control": control}
            if phase.phase_id == "replay_treatment_alpha_zero_step0":
                return {"treatment": treatment}
            self.fail(f"unexpected phase launch: {phase.phase_id}")

        result, calls, applicability_calls = (
            self.execute_with_mocked_phases(phase_result)
        )
        self.assertEqual(result["verdict"], "INVALID_DIAGNOSTIC")
        self.assertEqual(
            calls,
            [
                "replay_control_step0",
                "replay_treatment_alpha_zero_step0",
            ],
        )
        self.assertEqual(applicability_calls, 0)

    def test_malformed_eager_rows_fail_before_applicability(self):
        control_replay, treatment_replay = self.step0_results()
        flat = [1.0] * 8
        control_rows = [
            self.assay_row(step, flat, treatment=False)
            for step in range(1, 257)
        ]
        treatment_rows = [
            self.assay_row(step, flat, treatment=True)
            for step in range(1, 257)
        ]
        counterfactual = self.counterfactual_row(treatment_rows, flat)
        control_rows[200].pop("data_sha256")

        def phase_result(phase):
            if phase.phase_id == "replay_control_step0":
                return {"control": control_replay}
            if phase.phase_id == "replay_treatment_alpha_zero_step0":
                return {"treatment": treatment_replay}
            if phase.phase_id == "eager_mediator_pair_256":
                return {
                    "control": {"rows": control_rows},
                    "treatment": {
                        "rows": treatment_rows,
                        "counterfactual": counterfactual,
                    },
                }
            self.fail(f"unexpected phase launch: {phase.phase_id}")

        result, calls, applicability_calls = (
            self.execute_with_mocked_phases(phase_result)
        )
        self.assertEqual(result["verdict"], "INVALID_DIAGNOSTIC")
        self.assertEqual(
            calls,
            [
                "replay_control_step0",
                "replay_treatment_alpha_zero_step0",
                "eager_mediator_pair_256",
            ],
        )
        self.assertEqual(applicability_calls, 0)

    def test_malformed_placement_one_blocks_role_swap(self):
        control_replay, treatment_replay = self.step0_results()
        control_values = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
        treatment_values = [
            1.0,
            1.05,
            1.1,
            1.15,
            1.2,
            1.25,
            1.3,
            1.35,
        ]
        control_assay = [
            self.assay_row(step, control_values, treatment=False)
            for step in range(1, 257)
        ]
        treatment_assay = [
            self.assay_row(step, treatment_values, treatment=True)
            for step in range(1, 257)
        ]
        counterfactual = self.counterfactual_row(
            treatment_assay,
            [1.0, 1.08, 1.17, 1.26, 1.4, 1.55, 1.7, 1.8],
        )
        timing_control = self.timing_rows(100.0)
        timing_treatment = copy.deepcopy(timing_control)
        timing_treatment[-1]["data_sha256"] = "f" * 64
        compile_identity = {
            "dynamo_graph_count": 1,
            "dynamo_recompile_count": 0,
            "cudagraph_recording_count": 0,
            "compile_ids_sha256": "a" * 64,
        }

        def phase_result(phase):
            if phase.phase_id == "replay_control_step0":
                return {"control": control_replay}
            if phase.phase_id == "replay_treatment_alpha_zero_step0":
                return {"treatment": treatment_replay}
            if phase.phase_id == "eager_mediator_pair_256":
                return {
                    "control": {"rows": control_assay},
                    "treatment": {
                        "rows": treatment_assay,
                        "counterfactual": counterfactual,
                    },
                }
            if phase.phase_id == "timing_placement_1":
                return {
                    "control": {
                        "rows": timing_control,
                        "compile_identity": copy.deepcopy(compile_identity),
                    },
                    "treatment": {
                        "rows": timing_treatment,
                        "compile_identity": copy.deepcopy(compile_identity),
                    },
                }
            self.fail(f"unexpected phase launch: {phase.phase_id}")

        result, calls, applicability_calls = (
            self.execute_with_mocked_phases(phase_result)
        )
        self.assertEqual(result["verdict"], "INVALID_DIAGNOSTIC")
        self.assertEqual(
            calls,
            [
                "replay_control_step0",
                "replay_treatment_alpha_zero_step0",
                "eager_mediator_pair_256",
                "timing_placement_1",
            ],
        )
        self.assertEqual(applicability_calls, 1)

    @staticmethod
    def step0_results():
        digest = "a" * 64
        parity = {
            "schema_version": 1,
            "diagnostic_only": True,
            "packer_flags": dict(runner.PACKER_ZERO_ENV),
            "device_uuid": "GPU-aaaaaaaa",
            "data_sha256": digest,
            "targets_sha256": digest,
            "token_bytes_sha256": digest,
            "boundaries_sha256": digest,
            "data_cursor_sha256": digest,
            "epoch": 0,
            "cpu_rng_before_forward_sha256": digest,
            "cuda_rng_before_forward_sha256": digest,
            "cpu_rng_sha256": digest,
            "cuda_rng_sha256": digest,
            "logits_sha256": digest,
            "loss_sha256": digest,
            "preexisting_parameters_sha256": digest,
            "canonical_preexisting_model_state_sha256": digest,
            "preexisting_gradients_sha256": digest,
            "preexisting_gradient_coverage": {
                "parameter_count": 100,
                "gradient_tensor_count": 100,
                "missing_gradient_names": [],
                "manifest_sha256": digest,
            },
            "gpas_gates": [],
            "compile_identity": {
                "execution": "eager",
                "identity_sha256": digest,
            },
        }
        optimizer = {
            "canonical_preexisting_optimizer_groups_sha256": digest,
            "canonical_preexisting_optimizer_state_sha256": digest,
            "preexisting_optimizer_manifest_sha256": digest,
            "gpas_group": None,
        }
        control = {
            "parity": copy.deepcopy(parity),
            "optimizer_attestation": copy.deepcopy(optimizer),
        }
        treatment = copy.deepcopy(control)
        names = [
            f"transformer.h.{layer}.gpas_alpha" for layer in range(8)
        ]
        treatment["parity"]["gpas_gates"] = [
            {
                "name": name,
                "shape": [],
                "numel": 1,
                "alpha": 0.0,
                "gradient": 0.25 + index / 100,
                "forward_scale": 1.0,
            }
            for index, name in enumerate(names)
        ]
        treatment["optimizer_attestation"]["gpas_group"] = {
            "parameter_names": names,
            "parameter_shapes": [[] for _ in range(8)],
            "total_numel": 8,
            "learning_rate": 0.005,
            "betas": [0.8, 0.95],
            "eps": 1e-10,
            "weight_decay": 0.0,
            "demon_beta1": False,
            "schedule_exempt": True,
            "member_of_any_other_group": False,
        }
        return control, treatment

    def test_complete_step0_replay_contract_and_adversarial_failures(self):
        control, treatment = self.step0_results()
        result = runner.evaluate_step0_replay(control, treatment)
        self.assertTrue(result["valid"])
        self.assertTrue(result["exact_eight_named_gates"])
        self.assertTrue(result["canonical_optimizer_groups_and_state_equal"])

        mutations = {
            "seven gates": lambda c, t: t["parity"]["gpas_gates"].pop(),
            "wrong gate name": lambda c, t: t["parity"]["gpas_gates"][3].update(
                {"name": "transformer.h.99.gpas_alpha"}
            ),
            "wrong gate shape": lambda c, t: t["parity"]["gpas_gates"][0].update(
                {"shape": [1]}
            ),
            "missing preexisting gradient": lambda c, t: t["parity"][
                "preexisting_gradient_coverage"
            ].update(
                {
                    "gradient_tensor_count": 99,
                    "missing_gradient_names": ["transformer.wte.weight"],
                }
            ),
            "optimizer hash mismatch": lambda c, t: t[
                "optimizer_attestation"
            ].update(
                {"canonical_preexisting_optimizer_state_sha256": "b" * 64}
            ),
            "wrong GPAS LR": lambda c, t: t["optimizer_attestation"][
                "gpas_group"
            ].update({"learning_rate": 0.05}),
            "duplicate optimizer membership": lambda c, t: t[
                "optimizer_attestation"
            ]["gpas_group"].update({"member_of_any_other_group": True}),
            "data cursor mismatch": lambda c, t: t["parity"].update(
                {"data_cursor_sha256": "c" * 64}
            ),
            "epoch mismatch": lambda c, t: t["parity"].update({"epoch": 1}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                bad_control, bad_treatment = self.step0_results()
                mutate(bad_control, bad_treatment)
                with self.assertRaises(
                    (runner.RuntimeIntegrityError, RuntimeError)
                ):
                    runner.evaluate_step0_replay(
                        bad_control, bad_treatment
                    )

    @staticmethod
    def applicability_rows(variances):
        return [
            {
                "step": step,
                "post_mlp_residual_variance": list(variances),
            }
            for step in range(1, 33)
        ]

    def test_applicability_uses_spearman_and_joint_24_of_32_gate(self):
        increasing = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]
        result = runner.evaluate_applicability(
            self.applicability_rows(increasing)
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["valid_steps"], 32)
        self.assertAlmostEqual(
            result["steps"][0]["spearman_layer_vs_log_variance"], 1.0
        )
        self.assertEqual(result["steps"][0]["v7_over_v0"], 2.0)

        rows = self.applicability_rows(increasing)
        for row in rows[23:]:
            row["post_mlp_residual_variance"] = [1.0] * 8
        result = runner.evaluate_applicability(rows)
        self.assertFalse(result["passed"])
        self.assertEqual(result["valid_steps"], 23)

    @staticmethod
    def assay_row(step, variances, *, treatment):
        digest = hashlib.sha256(f"batch-{step}".encode()).hexdigest()
        row = {
            "step": step,
            "post_mlp_residual_variance": list(variances),
            "data_sha256": digest,
            "targets_sha256": digest,
            "token_bytes_sha256": digest,
            "boundaries_sha256": digest,
            "data_cursor": step * 2,
            "epoch": 0,
            "preexisting_gradients_finite": True,
        }
        if treatment:
            row.update(
                {
                    "gpas_gradients_finite": True,
                    "gpas_forward_scales": [1.0] * 8,
                }
            )
        return row

    @staticmethod
    def counterfactual_row(treatment_rows, variances):
        endpoint = treatment_rows[-1]
        return {
            "step": 256,
            "data_sha256": endpoint["data_sha256"],
            "targets_sha256": endpoint["targets_sha256"],
            "token_bytes_sha256": endpoint["token_bytes_sha256"],
            "boundaries_sha256": endpoint["boundaries_sha256"],
            "data_cursor": endpoint["data_cursor"],
            "epoch": endpoint["epoch"],
            "post_mlp_residual_variance": list(variances),
            "gpas_gate_values_sha256_before_zero": "1" * 64,
            "gpas_gate_values_sha256_while_zero": "0" * 64,
            "gpas_gate_values_sha256_after_restore": "1" * 64,
            "preexisting_model_state_sha256_before": "2" * 64,
            "preexisting_model_state_sha256_after": "2" * 64,
            "all_gpas_gates_exact_zero_during_measurement": True,
            "gpas_alpha_restored_after_measurement": True,
        }

    def test_attribution_evaluator_applies_every_mediator(self):
        control_values = [1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
        treatment_values = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35]
        control = [
            self.assay_row(step, control_values, treatment=False)
            for step in range(1, 257)
        ]
        treatment = [
            self.assay_row(step, treatment_values, treatment=True)
            for step in range(1, 257)
        ]
        counterfactual = self.counterfactual_row(
            treatment,
            [
                1.0,
                1.08,
                1.17,
                1.26,
                1.4,
                1.55,
                1.7,
                1.8,
            ],
        )
        result = runner.evaluate_attribution(
            control, treatment, counterfactual
        )
        self.assertTrue(result["passed"])
        self.assertLessEqual(
            result["depth_ratio_treatment_over_control"], 0.85
        )
        self.assertGreaterEqual(
            result["counterfactual_restore_fraction"], 0.50
        )
        self.assertGreaterEqual(
            result["deep_layer_contribution_fraction"], 0.50
        )

        treatment[2]["data_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            runner.RuntimeIntegrityError, "differs in data_sha256"
        ):
            runner.evaluate_attribution(control, treatment, counterfactual)

    def test_eager_structure_is_validated_before_mechanism_judgments(self):
        control_values = [1.0] * 8
        treatment_values = [1.0] * 8

        def valid():
            control = [
                self.assay_row(step, control_values, treatment=False)
                for step in range(1, 257)
            ]
            treatment = [
                self.assay_row(step, treatment_values, treatment=True)
                for step in range(1, 257)
            ]
            counterfactual = self.counterfactual_row(
                treatment, treatment_values
            )
            return control, treatment, counterfactual

        control, treatment, counterfactual = valid()
        structure = runner._validate_eager_pair_structure(
            control, treatment, counterfactual
        )
        self.assertEqual(structure["paired_batch_identities_verified"], 256)
        self.assertFalse(structure["thresholds_evaluated"])

        mutations = {
            "extra control key": lambda c, t, cf: c[0].update(
                {"unexpected": True}
            ),
            "missing treatment digest": lambda c, t, cf: t[31].pop(
                "targets_sha256"
            ),
            "late pair mismatch": lambda c, t, cf: t[255].update(
                {"data_sha256": "f" * 64}
            ),
            "nonfinite treatment scale": lambda c, t, cf: t[100][
                "gpas_forward_scales"
            ].__setitem__(3, float("nan")),
            "counterfactual extra key": lambda c, t, cf: cf.update(
                {"unexpected": True}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                bad_control, bad_treatment, bad_counterfactual = valid()
                mutate(bad_control, bad_treatment, bad_counterfactual)
                with self.assertRaises(runner.RuntimeIntegrityError):
                    runner._validate_eager_pair_structure(
                        bad_control,
                        bad_treatment,
                        bad_counterfactual,
                    )

    def test_paired_identity_rejects_missing_or_malformed_fields(self):
        digest = "a" * 64
        base = {
            "data_sha256": digest,
            "targets_sha256": digest,
            "token_bytes_sha256": digest,
            "boundaries_sha256": digest,
            "data_cursor": 0,
            "epoch": 0,
        }
        runner._paired_row_hashes_equal(base, dict(base), "pair")
        mutations = {
            "missing digest": lambda row: row.pop("data_sha256"),
            "malformed digest": lambda row: row.update({"targets_sha256": "x"}),
            "negative cursor": lambda row: row.update({"data_cursor": -1}),
            "boolean epoch": lambda row: row.update({"epoch": False}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                treatment = dict(base)
                mutate(treatment)
                with self.assertRaises(runner.RuntimeIntegrityError):
                    runner._paired_row_hashes_equal(
                        base, treatment, "pair"
                    )

    @staticmethod
    def timing_rows(step_ms):
        rows = []
        for step in range(1, 257):
            digest = hashlib.sha256(f"timing-{step}".encode()).hexdigest()
            rows.append(
                {
                    "step": step,
                    "step_time_ms": step_ms,
                    "tokens": 262144,
                    "data_sha256": digest,
                    "targets_sha256": digest,
                    "token_bytes_sha256": digest,
                    "boundaries_sha256": digest,
                    "data_cursor": step * 2,
                    "epoch": 0,
                }
            )
        return rows

    def test_role_swapped_timing_uses_28_contiguous_blocks(self):
        control = self.timing_rows(100.0)
        treatment = self.timing_rows(99.0)
        result = runner.evaluate_role_swapped_timing(
            {"control": control, "treatment": treatment},
            {"control": copy.deepcopy(control), "treatment": copy.deepcopy(treatment)},
            bootstrap_resamples=200,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["clean_blocks_per_arm_per_placement"], 28)
        self.assertAlmostEqual(
            result["role_normalized_point_ratio"], 100.0 / 99.0
        )
        self.assertAlmostEqual(
            result["one_sided_95pct_lcb"], 100.0 / 99.0
        )

    def test_timing_structure_checks_every_row_and_frozen_tokens(self):
        control = self.timing_rows(100.0)
        treatment = copy.deepcopy(control)
        blocks = runner._validate_timing_placement_structure(
            {"control": control, "treatment": treatment},
            "placement",
        )
        self.assertEqual(len(blocks[0]), 28)
        self.assertEqual(len(blocks[1]), 28)

        mutations = {
            "wrong token count": lambda rows: rows[32].update(
                {"tokens": 147456}
            ),
            "last pair mismatch": lambda rows: rows[255].update(
                {"boundaries_sha256": "f" * 64}
            ),
            "extra row key": lambda rows: rows[200].update(
                {"unexpected": True}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                bad_treatment = copy.deepcopy(control)
                mutate(bad_treatment)
                with self.assertRaises(runner.RuntimeIntegrityError):
                    runner._validate_timing_placement_structure(
                        {
                            "control": copy.deepcopy(control),
                            "treatment": bad_treatment,
                        },
                        "placement",
                    )

    def test_timing_overflow_fails_closed(self):
        control = self.timing_rows(1e308)
        treatment = self.timing_rows(1e308)
        with self.assertRaisesRegex(
            runner.RuntimeIntegrityError, "finite and positive"
        ):
            runner.evaluate_role_swapped_timing(
                {"control": control, "treatment": treatment},
                {
                    "control": copy.deepcopy(control),
                    "treatment": copy.deepcopy(treatment),
                },
                bootstrap_resamples=10,
            )

    def test_forbidden_endpoint_keys_fail_recursively(self):
        for key in ("val_bpb", "best_val_bpb", "sota", "val_loss"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    runner.RuntimeIntegrityError, "forbidden endpoint key"
                ):
                    runner._assert_no_forbidden_result_keys(
                        {"nested": [{key: 0.1}]}
                    )
        self.assertIsNotNone(runner.FORBIDDEN_OUTPUT_RE.search("bpb=0.9"))

    def test_remote_preflight_rejects_missing_and_duplicate_paths(self):
        root = runner.EXPECTED_REMOTE_WORK_DIR
        complete = "\n".join(
            f"{digest}  {root}/{relative}"
            for relative, digest in runner.REMOTE_EXECUTABLE_FILES.items()
        )
        observed = runner._parse_sha256sum_output(
            complete,
            expected=runner.REMOTE_EXECUTABLE_FILES,
            where="remote executable source",
            relative_to=root,
        )
        self.assertEqual(observed, runner.REMOTE_EXECUTABLE_FILES)

        lines = complete.splitlines()
        with self.assertRaisesRegex(
            runner.RuntimeIntegrityError, "hashes differ"
        ):
            runner._parse_sha256sum_output(
                "\n".join(lines[:-1]),
                expected=runner.REMOTE_EXECUTABLE_FILES,
                where="remote executable source",
                relative_to=root,
            )
        with self.assertRaisesRegex(
            runner.RuntimeIntegrityError, "duplicate"
        ):
            runner._parse_sha256sum_output(
                "\n".join([*lines, lines[0]]),
                expected=runner.REMOTE_EXECUTABLE_FILES,
                where="remote executable source",
                relative_to=root,
            )

    def test_remote_termination_targets_attested_pid_and_checks_gpu(self):
        class FakeProcess:
            returncode = None

            def __init__(self):
                self.terminated = False
                self.waits = 0

            def poll(self):
                return None if not self.terminated else 0

            def terminate(self):
                self.terminated = True

            def wait(self, timeout):
                self.waits += 1
                self.returncode = 0
                return 0

            def kill(self):
                self.terminated = True

        phase = runner.phase_plan()[0]
        arm = phase.arms[0]
        process = FakeProcess()
        active = runner.ActiveArm(
            phase=phase,
            arm=arm,
            manifest={
                "gpu": {
                    "index": 0,
                    "uuid": "GPU-aaaaaaaa",
                    "product": "NVIDIA H200",
                }
            },
            manifest_sha256="a" * 64,
            local_manifest=Path("/tmp/manifest"),
            local_result_tmp=Path("/tmp/result"),
            remote_manifest="/remote/manifest",
            remote_result="/remote/result",
            remote_pidfile="/remote/pid",
            process=process,
            pid=12345,
        )
        authority = {
            "worker": {
                "remote_path": (
                    f"{runner.EXPECTED_REMOTE_WORK_DIR}/"
                    "tools/run_paper020_gpas_worker.py"
                )
            },
            "remote": {"work_dir": runner.EXPECTED_REMOTE_WORK_DIR},
        }
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with (
            mock.patch.object(
                runner, "_remote_run", return_value=completed
            ) as remote_run,
            mock.patch.object(runner, "_require_clean_gpus") as clean,
        ):
            runner._terminate_active(authority, [active])
        command = remote_run.call_args.args[1]
        self.assertIn("pid=12345", command)
        self.assertIn("kill -TERM", command)
        self.assertIn("kill -KILL", command)
        self.assertIn(runner.EXPECTED_REMOTE_WORK_DIR, command)
        self.assertIn("run_paper020_gpas_worker.py", command)
        clean.assert_called_once()

    def test_artifact_reenumeration_captures_partial_arm_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "placement.control.worker-result.json"
            second = root / "placement.control.manifest.json"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            observed = runner._enumerate_immutable_artifacts(root)
            self.assertEqual(
                observed,
                {
                    first.name: hashlib.sha256(b"first").hexdigest(),
                    second.name: hashlib.sha256(b"second").hexdigest(),
                },
            )

    def test_exclusive_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "immutable.json"
            first = runner._write_exclusive_json(path, {"value": 1})
            self.assertEqual(first, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                runner._write_exclusive_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_authority_is_required_by_cli(self):
        with self.assertRaises(SystemExit):
            runner.parse_args([])


if __name__ == "__main__":
    unittest.main()
