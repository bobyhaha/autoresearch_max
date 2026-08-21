import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import run_gated, run_stage
from vibeautoresearch.core import SchemaError


def experiment():
    return SimpleNamespace(
        experiment_id="exp_stage_test",
        seeds=(42, 43, 44),
        arms=(
            {"arm_id": "control", "role": "control"},
            {"arm_id": "treatment", "role": "treatment"},
        ),
    )


def qualification_experiment():
    return SimpleNamespace(
        experiment_id="exp_stage_qualification",
        seeds=(47, 48, 49, 50),
        arms=(
            {"arm_id": "control", "role": "control"},
            {"arm_id": "treatment", "role": "treatment"},
        ),
    )


def qualification_policy():
    schedule = []
    gpu_pairs = ((0, 1), (3, 2), (4, 5), (7, 6))
    for seed, (control, treatment) in zip(range(47, 51), gpu_pairs, strict=True):
        schedule.append(
            {
                "seed": seed,
                "control": {
                    "index": control,
                    "uuid": f"GPU-uuid-{control}",
                },
                "treatment": {
                    "index": treatment,
                    "uuid": f"GPU-uuid-{treatment}",
                },
            }
        )
    return {
        "qualification": {
            "kind": "historical_equivalence_bridge",
            "gpu_schedule": schedule,
            "inventory_gate": {
                "required_clean_snapshots": 2,
                "minimum_interval_seconds": 5,
                "require_full_node_clean": True,
            },
        }
    }


def recorded(arm_id, seed, status="complete", *tags):
    return SimpleNamespace(
        experiment_id="exp_stage_test",
        arm_id=arm_id,
        seed=seed,
        status=status,
        tags=tags,
    )


class StageSchedulerTest(unittest.TestCase):
    @staticmethod
    def _packer_facts(*, sidecar, step_ms):
        digest = "a" * 64
        profile = [float(step_ms)] * 200
        return {
            "verified": True,
            "grad_accum_steps": 2,
            "preclock_batches": 1000,
            "preclock_data_sha256": digest,
            "preclock_boundary_sha256": digest,
            "aa_steps": 20,
            "aa_micro_batches": 40,
            "aa_model_sha256": digest,
            "aa_optimizer_sha256": digest,
            "aa_loss_sha256": digest,
            "aa_data_sha256": digest,
            "aa_boundary_sha256": digest,
            "aa_cpu_rng_sha256": digest,
            "aa_cuda_rng_sha256": digest,
            "aa_dynamo_graph_count": 1,
            "activation_step": 20,
            "sidecar_active": sidecar,
            "ab_start_step": 20,
            "ab_end_step": 40,
            "ab_model_sha256": digest,
            "ab_optimizer_sha256": digest,
            "ab_loss_sha256": digest,
            "ab_data_sha256": digest,
            "ab_boundary_sha256": digest,
            "ab_cpu_rng_sha256": digest,
            "ab_cuda_rng_sha256": digest,
            "dynamo_graph_count": 1,
            "dynamo_recompile_count": 0,
            "cudagraph_recording_count": 0,
            "compile_ids_sha256": digest,
            "compile_sites": "0:0",
            "clean_profile_start_step": 50,
            "terminal_steps": 250,
            "terminal_model_sha256": digest,
            "terminal_optimizer_sha256": digest,
            "terminal_aa_loss_sha256": digest,
            "terminal_aa_boundary_sha256": digest,
            "terminal_ab_loss_sha256": digest,
            "terminal_ab_data_sha256": digest,
            "terminal_ab_boundary_sha256": digest,
            "terminal_tail_loss_sha256": digest,
            "terminal_aa_data_sha256": digest,
            "profile_sha256": "b" * 64 if sidecar else "c" * 64,
            "profile_step_time_ms": profile,
        }

    @classmethod
    def _packer_record(
        cls,
        arm_id,
        *,
        experiment_id="exp_packer_recovery",
        seed=63,
        sidecar=False,
        step_ms=100,
        status="complete",
    ):
        return SimpleNamespace(
            experiment_id=experiment_id,
            arm_id=arm_id,
            seed=seed,
            status=status,
            tags=("diagnostic_packer_boundary_parity",),
            run_id=f"run_{experiment_id}_{arm_id}_s{seed}",
            fingerprint=hashlib.sha256(
                f"{experiment_id}:{arm_id}:{seed}:{status}".encode()
            ).hexdigest(),
            config_hash=hashlib.sha256(arm_id.encode()).hexdigest()[:16],
            git_commit="1" * 40,
            tracker={
                "gpu_id": 0 if arm_id == "control" else 1,
                "gpu_uuid": (
                    "GPU-control-uuid"
                    if arm_id == "control"
                    else "GPU-treatment-uuid"
                ),
                "code_hashes_sha256": "a" * 64,
                "challenge_id": "challenge_test",
                "challenge_definition_fingerprint": "b" * 64,
                "challenge_selection_fingerprint": "c" * 64,
                "resolved_training_config": {
                    "STOP_MODE": "steps",
                    "MAX_STEPS": "250",
                    "PACKER_DOC_BOUNDARIES": "1" if sidecar else "0",
                },
                "packer_boundary_integrity": cls._packer_facts(
                    sidecar=sidecar,
                    step_ms=step_ms,
                ),
            },
        )

    def test_packer_pair_attestation_bootstraps_and_enforces_exact_state(self):
        control = self._packer_facts(sidecar=False, step_ms=100)
        treatment = self._packer_facts(sidecar=True, step_ms=90)

        result, errors = run_stage._packer_pair_attestation(
            control,
            treatment,
            bootstrap_resamples=1000,
        )

        self.assertEqual(errors, [])
        self.assertTrue(result["integrity_verified"])
        self.assertTrue(result["single_placement_threshold_passed"])
        self.assertFalse(result["performance_gate_passed"])
        self.assertEqual(
            result["verdict"], "REQUIRE_COUNTERBALANCED_PROFILE"
        )
        self.assertEqual(result["bootstrap_blocks"], 20)
        self.assertAlmostEqual(
            result["token_rate_ratio_treatment_over_control"],
            100 / 90,
        )

        treatment["terminal_model_sha256"] = "d" * 64
        result, errors = run_stage._packer_pair_attestation(
            control,
            treatment,
            bootstrap_resamples=100,
        )
        self.assertFalse(result["integrity_verified"])
        self.assertTrue(any("terminal_model_sha256" in item for item in errors))

    def test_packer_pair_rejects_251_step_profile_instead_of_truncating(self):
        control = self._packer_facts(sidecar=False, step_ms=100)
        treatment = self._packer_facts(sidecar=True, step_ms=90)
        control["profile_step_time_ms"].append(100.0)
        treatment["profile_step_time_ms"].append(90.0)

        result, errors = run_stage._packer_pair_attestation(
            control,
            treatment,
            bootstrap_resamples=100,
        )

        self.assertFalse(result["integrity_verified"])
        self.assertEqual(result["profile_samples_per_arm"], 201)
        self.assertTrue(
            any("exactly 200 samples" in error for error in errors)
        )

    def test_gpu_list_must_form_distinct_pairs(self):
        self.assertEqual(run_stage._gpu_ids("0,1,4,5"), [0, 1, 4, 5])
        for value in ("0", "0,1,2", "0,0", "gpu0,1"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    run_stage._gpu_ids(value)

    def test_diagnostic_step_budget_is_explicit_and_above_warmup(self):
        self.assertEqual(run_stage._diagnostic_steps("20"), 20)
        for value in ("0", "10", "-1", "twenty"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    run_stage._diagnostic_steps(value)

    def test_diagnostic_cli_requires_exactly_one_gpu_pair(self):
        with self.assertRaisesRegex(SchemaError, "exactly two GPUs"):
            run_stage.main(
                [
                    "exp_stage_test",
                    "--gpus",
                    "0,1,2,3",
                    "--ssh",
                    "ssh host",
                    "--remote-wd",
                    "/remote/work",
                    "--diagnostic-steps",
                    "20",
                ]
            )

    def test_packer_boundary_parity_requires_diagnostic_steps(self):
        with self.assertRaisesRegex(
            SchemaError, "requires --diagnostic-steps"
        ):
            run_stage.main(
                [
                    "exp_stage_test",
                    "--gpus",
                    "0,1",
                    "--ssh",
                    "ssh host",
                    "--remote-wd",
                    "/remote/work",
                    "--packer-boundary-parity",
                ]
            )

        with self.assertRaisesRegex(
            SchemaError, r"diagnostic-steps exactly 250"
        ):
            run_stage.main(
                [
                    "exp_stage_test",
                    "--gpus",
                    "0,1",
                    "--ssh",
                    "ssh host",
                    "--remote-wd",
                    "/remote/work",
                    "--diagnostic-steps",
                    "20",
                    "--packer-boundary-parity",
                ]
            )
        with self.assertRaisesRegex(
            SchemaError, r"diagnostic-steps exactly 250"
        ):
            run_stage.main(
                [
                    "exp_stage_test",
                    "--gpus",
                    "0,1",
                    "--ssh",
                    "ssh host",
                    "--remote-wd",
                    "/remote/work",
                    "--diagnostic-steps",
                    "251",
                    "--packer-boundary-parity",
                ]
            )

    def test_comparison_recovery_artifact_binds_current_run_fingerprints(self):
        records = [
            self._packer_record("control"),
            self._packer_record(
                "treatment",
                sidecar=True,
                step_ms=90,
            ),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, artifact = run_stage._finalize_packer_pair(
                root,
                "exp_packer_recovery",
                63,
                "control",
                "treatment",
                records,
            )
            payload = json.loads(artifact.read_text(encoding="utf-8"))

            run_stage._validate_existing_packer_pair_artifact(
                payload,
                experiment_id="exp_packer_recovery",
                seed=63,
                ledger_records=records,
            )

        self.assertTrue(result["integrity_verified"])
        self.assertIn("exp_packer_recovery", artifact.name)
        self.assertEqual(
            {item["run_fingerprint"] for item in payload["run_bindings"]},
            {record.fingerprint for record in records},
        )

    def test_recovery_artifact_fails_closed_on_identity_hash_or_ledger_drift(self):
        records = [
            self._packer_record("control"),
            self._packer_record("treatment", sidecar=True, step_ms=90),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _result, artifact = run_stage._finalize_packer_pair(
                root,
                "exp_packer_recovery",
                63,
                "control",
                "treatment",
                records,
            )
            payload = json.loads(artifact.read_text(encoding="utf-8"))

        mutations = (
            ("wrong experiment", {**payload, "experiment_id": "exp_other"}),
            ("wrong seed", {**payload, "seed": 64}),
            (
                "binding hash drift",
                {**payload, "run_bindings_sha256": "0" * 64},
            ),
        )
        for label, mutated in mutations:
            with self.subTest(label=label), self.assertRaises(SchemaError):
                run_stage._validate_existing_packer_pair_artifact(
                    mutated,
                    experiment_id="exp_packer_recovery",
                    seed=63,
                    ledger_records=records,
                )

        drifted_records = [
            records[0],
            SimpleNamespace(
                **{
                    **vars(records[1]),
                    "fingerprint": "f" * 64,
                }
            ),
        ]
        with self.assertRaisesRegex(SchemaError, "fingerprint"):
            run_stage._validate_existing_packer_pair_artifact(
                payload,
                experiment_id="exp_packer_recovery",
                seed=63,
                ledger_records=drifted_records,
            )

    def test_pair_artifact_paths_do_not_collide_across_experiments(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = run_stage._write_packer_pair_attestation(
                root,
                "exp_packer_alpha",
                63,
                {
                    "integrity_verified": False,
                    "run_bindings": [],
                    "run_bindings_sha256": hashlib.sha256(b"[]").hexdigest(),
                },
            )
            second = run_stage._write_packer_pair_attestation(
                root,
                "exp_packer_beta",
                63,
                {
                    "integrity_verified": False,
                    "run_bindings": [],
                    "run_bindings_sha256": hashlib.sha256(b"[]").hexdigest(),
                },
            )

        self.assertNotEqual(first, second)
        self.assertIn("exp_packer_alpha", first.name)
        self.assertIn("exp_packer_beta", second.name)

    def test_one_sided_crash_artifact_is_immutable_and_ledger_bound(self):
        control = self._packer_record("control")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, artifact = run_stage._finalize_packer_pair(
                root,
                "exp_packer_recovery",
                63,
                "control",
                "treatment",
                [control],
                launch_errors=["treatment runner crashed"],
            )
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            run_stage._validate_existing_packer_pair_artifact(
                payload,
                experiment_id="exp_packer_recovery",
                seed=63,
                ledger_records=[control],
            )
            with self.assertRaisesRegex(SchemaError, "already exists"):
                run_stage._finalize_packer_pair(
                    root,
                    "exp_packer_recovery",
                    63,
                    "control",
                    "treatment",
                    [control],
                    launch_errors=["retry forbidden"],
                )

        self.assertEqual(result["verdict"], "INVALID_DIAGNOSTIC")
        self.assertFalse(result["integrity_verified"])
        self.assertEqual(len(payload["run_bindings"]), 1)

    def test_diagnostic_source_snapshot_uses_runner_code_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes = {}
            for index, filename in enumerate(run_gated.CODE_FILES):
                payload = f"source-{index}".encode()
                (root / filename).write_bytes(payload)
                hashes[filename] = hashlib.sha256(payload).hexdigest()

            observed = run_stage._diagnostic_source_snapshot_sha256(root)

        self.assertEqual(
            observed,
            run_gated.code_hashes_fingerprint(hashes),
        )

    def test_diagnostic_pair_is_immutable_after_either_arm_records(self):
        exp = experiment()
        prior = recorded(
            "control",
            42,
            "invalid",
            "diagnostic_offbudget_20steps",
        )
        with self.assertRaisesRegex(SchemaError, "cannot be retried or repaired"):
            run_stage._require_fresh_diagnostic_pair(exp, [prior], 42)

        # Other seeds and ordinary records do not consume this diagnostic pair.
        run_stage._require_fresh_diagnostic_pair(
            exp,
            [recorded("control", 42), recorded("control", 43, "complete",
                                              "diagnostic_offbudget_20steps")],
            44,
        )

    def test_only_explicit_qualification_can_fill_four_gpu_pairs(self):
        self.assertEqual(
            run_stage._tranche_pair_limit(
                {"qualification": {"kind": "historical_equivalence_bridge"}}
            ),
            4,
        )
        self.assertEqual(
            run_stage._tranche_pair_limit({"qualification": {}}),
            3,
        )

    def test_completed_pairs_ignore_diagnostics(self):
        runs = [
            recorded("control", 42),
            recorded("treatment", 42),
            recorded("control", 43, "complete", "diagnostic_offbudget_4000steps"),
        ]
        self.assertEqual(run_stage._completed_pairs(experiment(), runs), {42})

    def test_one_sided_or_invalid_pair_stops_the_stage(self):
        with self.assertRaisesRegex(SchemaError, "one-sided"):
            run_stage._completed_pairs(
                experiment(),
                [recorded("control", 42)],
            )
        with self.assertRaisesRegex(SchemaError, "failed/invalid"):
            run_stage._completed_pairs(
                experiment(),
                [
                    recorded("control", 42),
                    recorded("treatment", 42, "invalid"),
                ],
            )

    def test_qualification_selects_exactly_one_frozen_eight_gpu_wave(self):
        exp = qualification_experiment()
        policy = qualification_policy()
        tranche, schedule = run_stage._select_tranche(
            exp,
            policy,
            [],
            {"verdict": "pending", "complete_pairs": 0},
            4,
            set(),
            [0, 1, 3, 2, 4, 5, 7, 6],
        )

        self.assertEqual(tranche, [47, 48, 49, 50])
        self.assertEqual(
            schedule,
            (
                (47, 0, "GPU-uuid-0", 1, "GPU-uuid-1"),
                (48, 3, "GPU-uuid-3", 2, "GPU-uuid-2"),
                (49, 4, "GPU-uuid-4", 5, "GPU-uuid-5"),
                (50, 7, "GPU-uuid-7", 6, "GPU-uuid-6"),
            ),
        )

    def test_qualification_rejects_partial_capacity_or_gpu_remapping(self):
        exp = qualification_experiment()
        policy = qualification_policy()
        common = (
            exp,
            policy,
            [],
            {"verdict": "pending", "complete_pairs": 0},
            4,
            set(),
        )
        with self.assertRaisesRegex(SchemaError, "exactly 8 requested GPUs"):
            run_stage._select_tranche(*common, [0, 1, 3, 2, 4, 5])
        with self.assertRaisesRegex(SchemaError, "remapping is forbidden"):
            run_stage._select_tranche(
                *common,
                [1, 0, 3, 2, 4, 5, 7, 6],
            )

    def test_qualification_rejects_any_prior_run_record(self):
        exp = qualification_experiment()
        prior = SimpleNamespace(
            experiment_id=exp.experiment_id,
            arm_id="control",
            seed=47,
            status="invalid",
            tags=("diagnostic_offbudget",),
        )
        with self.assertRaisesRegex(SchemaError, "cannot be continued, remapped, or retried"):
            run_stage._select_tranche(
                exp,
                qualification_policy(),
                [prior],
                {"verdict": "pending", "complete_pairs": 0},
                4,
                set(),
                [0, 1, 3, 2, 4, 5, 7, 6],
            )

    def test_qualification_rejects_non_pristine_pending_set(self):
        exp = qualification_experiment()
        with self.assertRaisesRegex(SchemaError, "one untouched wave"):
            run_stage._select_tranche(
                exp,
                qualification_policy(),
                [],
                {"verdict": "pending", "complete_pairs": 0},
                3,
                set(),
                [0, 1, 3, 2, 4, 5, 7, 6],
            )

    def test_qualification_requires_two_clean_stable_full_node_inventories(self):
        exp = qualification_experiment()
        policy = qualification_policy()
        schedule = run_stage._qualification_schedule(exp, policy)
        gpu_rows = "\n".join(
            f"{index}, GPU-uuid-{index}" for index in range(8)
        )
        output = (
            f"{run_stage.INVENTORY_GPU_BEGIN}\n"
            f"{gpu_rows}\n"
            f"{run_stage.INVENTORY_COMPUTE_BEGIN}\n"
            f"{run_stage.INVENTORY_END}\n"
        )
        response = SimpleNamespace(
            returncode=0,
            stdout=output,
            stderr="",
        )

        with (
            patch.object(run_stage.subprocess, "run", return_value=response) as run,
            patch.object(run_stage.time, "sleep") as sleep,
        ):
            snapshots = run_stage._preflight_qualification_gpus(
                ["ssh", "host"],
                schedule,
                policy,
            )

        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(5.0)
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(
            snapshots[0]["gpu_index_to_uuid"]["7"],
            "GPU-uuid-7",
        )
        self.assertEqual(snapshots[1]["compute_applications"], [])
        self.assertIn("--query-gpu=index,uuid", run.call_args.args[0][-1])
        self.assertIn("--query-compute-apps=gpu_uuid,pid", run.call_args.args[0][-1])

    def test_qualification_inventory_rejects_uuid_drift_and_tenant(self):
        exp = qualification_experiment()
        policy = qualification_policy()
        schedule = run_stage._qualification_schedule(exp, policy)
        drift_rows = "\n".join(
            f"{index}, GPU-uuid-{'changed' if index == 6 else index}"
            for index in range(8)
        )
        drift = SimpleNamespace(
            returncode=0,
            stdout=(
                f"{run_stage.INVENTORY_GPU_BEGIN}\n"
                f"{drift_rows}\n"
                f"{run_stage.INVENTORY_COMPUTE_BEGIN}\n"
                f"{run_stage.INVENTORY_END}\n"
            ),
            stderr="",
        )
        with (
            patch.object(run_stage.subprocess, "run", return_value=drift),
            self.assertRaisesRegex(RuntimeError, "index-to-UUID map differs"),
        ):
            run_stage._preflight_qualification_gpus(
                ["ssh", "host"],
                schedule,
                policy,
            )

        gpu_rows = "\n".join(
            f"{index}, GPU-uuid-{index}" for index in range(8)
        )
        occupied = SimpleNamespace(
            returncode=0,
            stdout=(
                f"{run_stage.INVENTORY_GPU_BEGIN}\n"
                f"{gpu_rows}\n"
                f"{run_stage.INVENTORY_COMPUTE_BEGIN}\n"
                "GPU-uuid-4, 12345\n"
                f"{run_stage.INVENTORY_END}\n"
            ),
            stderr="",
        )
        with (
            patch.object(run_stage.subprocess, "run", return_value=occupied),
            self.assertRaisesRegex(RuntimeError, "full node to be tenant-free"),
        ):
            run_stage._preflight_qualification_gpus(
                ["ssh", "host"],
                schedule,
                policy,
            )

    def test_qualification_execution_authority_binds_local_and_remote_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "research"
            authority_path = root / "authority.json"
            root.mkdir()
            remote_hashes = {
                "/remote/runtime/train.py": "a" * 64,
                "/remote/runtime/data.py": "b" * 64,
            }
            payload = json.dumps(
                {"remote_file_hashes": remote_hashes},
                sort_keys=True,
            ).encode()
            authority_path.write_bytes(payload)
            policy = qualification_policy()
            policy["qualification"]["execution_authority"] = {
                "path": "authority.json",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

            authority_hash, loaded_hashes = (
                run_stage._load_qualification_execution_authority(root, policy)
            )

            self.assertEqual(
                authority_hash,
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(loaded_hashes, remote_hashes)
            response = SimpleNamespace(
                returncode=0,
                stdout=(
                    f"{run_stage.REMOTE_HASH_BEGIN}\n"
                    f"{'b' * 64}  /remote/runtime/data.py\n"
                    f"{'a' * 64}  /remote/runtime/train.py\n"
                    f"{run_stage.REMOTE_HASH_END}\n"
                ),
                stderr="",
            )
            with patch.object(
                run_stage.subprocess,
                "run",
                return_value=response,
            ) as remote:
                run_stage._verify_remote_execution_authority(
                    ["ssh", "host"],
                    loaded_hashes,
                )

            remote.assert_called_once()
            command = remote.call_args.args[0][-1]
            self.assertIn("sha256sum --", command)
            self.assertIn("/remote/runtime/data.py", command)
            self.assertIn("/remote/runtime/train.py", command)

    def test_qualification_execution_authority_fails_closed_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "research"
            authority_path = root / "authority.json"
            root.mkdir()
            payload = json.dumps(
                {"remote_file_hashes": {"/remote/train.py": "a" * 64}},
            ).encode()
            authority_path.write_bytes(payload)
            policy = qualification_policy()
            policy["qualification"]["execution_authority"] = {
                "path": "authority.json",
                "sha256": "0" * 64,
            }
            with self.assertRaisesRegex(SchemaError, "authority hash mismatch"):
                run_stage._load_qualification_execution_authority(root, policy)

            policy["qualification"]["execution_authority"]["sha256"] = (
                hashlib.sha256(payload).hexdigest()
            )
            _authority_hash, remote_hashes = (
                run_stage._load_qualification_execution_authority(root, policy)
            )
            response = SimpleNamespace(
                returncode=0,
                stdout=(
                    f"{run_stage.REMOTE_HASH_BEGIN}\n"
                    f"{'c' * 64}  /remote/train.py\n"
                    f"{run_stage.REMOTE_HASH_END}\n"
                ),
                stderr="",
            )
            with (
                patch.object(run_stage.subprocess, "run", return_value=response),
                self.assertRaisesRegex(RuntimeError, "remote runtime hash mismatch"),
            ):
                run_stage._verify_remote_execution_authority(
                    ["ssh", "host"],
                    remote_hashes,
                )

    def test_qualification_wave_attempt_is_durable_and_single_use(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "research"
            root.mkdir()
            exp = qualification_experiment()
            exp.fingerprint = "experiment-fingerprint"
            schedule = run_stage._qualification_schedule(
                exp,
                qualification_policy(),
            )
            snapshots = tuple(
                {
                    "snapshot": number,
                    "captured_at": f"2026-07-29T12:00:0{number}+00:00",
                    "gpu_index_to_uuid": {
                        str(index): f"GPU-uuid-{index}" for index in range(8)
                    },
                    "compute_applications": [],
                }
                for number in (1, 2)
            )
            wave_id = run_stage._record_qualification_wave_attempt(
                root,
                exp,
                schedule,
                snapshots,
                "a" * 64,
                {"/remote/train.py": "b" * 64},
                "selection-fingerprint",
                "walltime_5min_h200",
            )

            self.assertRegex(wave_id, r"^qwave_[0-9a-f]{16}$")
            attempts = run_stage._qualification_wave_attempts(
                root,
                exp.experiment_id,
            )
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["wave_id"], wave_id)
            self.assertEqual(attempts[0]["inventory_snapshots"], list(snapshots))
            with self.assertRaisesRegex(SchemaError, "already has durable wave"):
                run_stage._require_no_qualification_wave_attempt(
                    root,
                    exp.experiment_id,
                )
            with self.assertRaisesRegex(SchemaError, "already consumed"):
                run_stage._record_qualification_wave_attempt(
                    root,
                    exp,
                    schedule,
                    snapshots,
                    "a" * 64,
                    {"/remote/train.py": "b" * 64},
                    "selection-fingerprint",
                    "walltime_5min_h200",
                )

    def test_ordinary_tranche_behavior_is_unchanged(self):
        exp = experiment()
        tranche, schedule = run_stage._select_tranche(
            exp,
            {},
            [],
            {"verdict": "pending", "complete_pairs": 0},
            3,
            set(),
            [0, 1],
        )
        self.assertEqual(tranche, [42])
        self.assertEqual(schedule, ())

    def test_scheduler_capability_authorizes_only_the_exact_run_tuple(self):
        expected = {
            "experiment_id": "exp_stage_test",
            "arm_id": "control",
            "seed": 42,
        }
        child = (
            "import json,socket,sys;"
            "s=socket.socket(fileno=int(sys.argv[-1]));"
            f"s.sendall(json.dumps({json.dumps(expected)}).encode()+b'\\n');"
            "reply=s.recv(128);"
            "sys.exit(0 if reply==b'AUTHORIZED\\n' else 2)"
        )

        process, sched_sock = run_stage._spawn_authorized_runner(
            [sys.executable, "-c", child],
            expected,
        )

        self.assertEqual(process.wait(timeout=10), 0)
        # The scheduler socket is now kept OPEN (the runner's death-signal); the
        # caller owns it and closes it during teardown.
        sched_sock.close()

    def test_scheduler_capability_rejects_a_changed_run_tuple(self):
        requested = {"experiment_id": "exp_stage_test", "arm_id": "control", "seed": 43}
        expected = {**requested, "seed": 42}
        child = (
            "import json,socket,sys;"
            "s=socket.socket(fileno=int(sys.argv[-1]));"
            f"s.sendall(json.dumps({json.dumps(requested)}).encode()+b'\\n');"
            "s.recv(128)"
        )

        with self.assertRaisesRegex(RuntimeError, "different"):
            run_stage._spawn_authorized_runner(
                [sys.executable, "-c", child],
                expected,
            )


if __name__ == "__main__":
    unittest.main()
