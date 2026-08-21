"""Tests for Track B segmented timing and time-matched endpoint logic.

These tests do NOT require a GPU -- they validate the pure-Python
data structures and analysis functions used by Track B.
Uses unittest (not pytest) for CI compatibility.
"""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observable import ObservableRegister
from tools import track_b_equal_time


class TestObservableRegisterRecordStep(unittest.TestCase):
    """Verify record_step / write_curves_json against the dashboard contract.

    training_dashboard/app.js reads observable_curves.json as
    {"metadata": {...}, "num_steps": N, "series": {key: [{"step", "value"}]}}
    (the same format written by research/sota_snapshots/*/observable.py),
    so these tests pin the *series* schema, not an ad-hoc row-list one.
    """

    def test_record_step_appends(self):
        obs = ObservableRegister()
        obs.record_step(0, 1, {"loss": 1.5})
        obs.record_step(1, 1, {"loss": 1.3})
        self.assertEqual(len(obs._steps), 2)
        self.assertEqual(obs._steps[0]["step"], 0)
        self.assertAlmostEqual(obs._steps[1]["loss"], 1.3)

    def test_write_curves_json(self):
        obs = ObservableRegister()
        obs.record_step(0, 1, {"loss": 1.5, "dt": 0.1})
        obs.record_step(1, 1, {"loss": 1.3, "dt": 0.09})
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        try:
            obs.write_curves_json(path, metadata={"run_id": "test"})
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["metadata"]["run_id"], "test")
            self.assertEqual(data["num_steps"], 2)
            self.assertEqual(
                data["series"]["loss"],
                [{"step": 0, "value": 1.5}, {"step": 1, "value": 1.3}],
            )
            self.assertEqual(len(data["series"]["dt"]), 2)
        finally:
            os.unlink(path)

    def test_record_step_negative_epoch(self):
        obs = ObservableRegister()
        obs.record_step(5, None, {"loss": 1.0})
        self.assertEqual(obs._steps[0]["epoch"], -1)


class TestTrackBTimeMatchedEndpoint(unittest.TestCase):
    """Validate the historical truncation lookup.

    NOTE: a probe value read off a longer run at cum_time <= T_ref is only a
    LOWER BOUND on equal-training-time quality: the LR schedule is defined over
    MAX_STEPS, so the truncated point has an incomplete warmdown.  The valid
    diagnostic runs each config directly under the same STOP_MODE=time budget
    (see tools/track_b_equal_time.py).
    """

    @staticmethod
    def compute_effective_steps(cum_times, t_ref):
        """Given cumulative times per step, find the last step whose
        cumulative time <= t_ref. Returns None if no valid point."""
        for i in range(len(cum_times) - 1, -1, -1):
            if cum_times[i] <= t_ref:
                return i
        return None

    def test_exact_match(self):
        cum = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.assertEqual(self.compute_effective_steps(cum, 0.3), 2)

    def test_between_steps(self):
        cum = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.assertEqual(self.compute_effective_steps(cum, 0.35), 2)

    def test_before_first_step(self):
        cum = [0.1, 0.2, 0.3]
        self.assertIsNone(self.compute_effective_steps(cum, 0.05))

    def test_t_ref_exceeds_all(self):
        cum = [0.1, 0.2, 0.3]
        self.assertEqual(self.compute_effective_steps(cum, 1.0), 2)

    def test_t_ref_zero(self):
        cum = [0.1, 0.2, 0.3]
        self.assertIsNone(self.compute_effective_steps(cum, 0.0))

    def test_empty(self):
        self.assertIsNone(self.compute_effective_steps([], 1.0))


class TestTrackBStepTimeStats(unittest.TestCase):
    """Validate step time statistics computation."""

    @staticmethod
    def compute_stats(step_times_ms):
        if not step_times_ms:
            return {"median": 0, "mean": 0, "std": 0, "p90": 0, "p95": 0}
        n = len(step_times_ms)
        s = sorted(step_times_ms)
        mean = sum(s) / n
        var = sum((x - mean) ** 2 for x in s) / n
        return {
            "median": s[n // 2],
            "mean": mean,
            "std": math.sqrt(var),
            "p90": s[int(n * 0.90)],
            "p95": s[int(n * 0.95)],
        }

    def test_constant_times(self):
        times = [100.0] * 100
        stats = self.compute_stats(times)
        self.assertAlmostEqual(stats["median"], 100.0)
        self.assertAlmostEqual(stats["mean"], 100.0)
        self.assertAlmostEqual(stats["std"], 0.0)

    def test_variable_times(self):
        times = [90.0, 100.0, 110.0, 95.0, 105.0]
        stats = self.compute_stats(times)
        self.assertAlmostEqual(stats["mean"], 100.0)
        self.assertGreater(stats["std"], 0)

    def test_empty(self):
        stats = self.compute_stats([])
        self.assertEqual(stats["median"], 0)

    def test_single_element(self):
        stats = self.compute_stats([123.0])
        self.assertAlmostEqual(stats["median"], 123.0)
        self.assertAlmostEqual(stats["std"], 0.0)


class TestTrackBTRefComputation(unittest.TestCase):
    """Pin the arithmetic used by the superseded step-budget approximation."""

    def test_t_ref_from_median(self):
        baseline_median_step_ms = 150.0
        t_ref = baseline_median_step_ms * 2000 / 1000.0
        self.assertAlmostEqual(t_ref, 300.0)

    def test_effective_steps_from_baseline(self):
        t_ref = 300.0
        treatment_step_time = 0.2
        effective = int(t_ref / treatment_step_time)
        self.assertEqual(effective, 1500)

    def test_no_extrapolation(self):
        t_ref = 300.0
        treatment_step_time = 0.2
        max_steps_run = 1000
        effective_from_time = int(t_ref / treatment_step_time)
        self.assertGreater(effective_from_time, max_steps_run)


class TestTrackBConfigValidation(unittest.TestCase):
    """Verify Track B config overrides."""

    def test_track_b_disables_val_loss_every(self):
        TRACK_B_MODE = True
        VAL_LOSS_EVERY = 1
        if TRACK_B_MODE:
            VAL_LOSS_EVERY = 0
        self.assertEqual(VAL_LOSS_EVERY, 0)

    def test_track_b_disables_layer_probes(self):
        TRACK_B_MODE = True
        OBSERVE_LAYER_PROBES = True
        if TRACK_B_MODE:
            OBSERVE_LAYER_PROBES = False
        self.assertFalse(OBSERVE_LAYER_PROBES)

    def test_track_b_off_preserves_defaults(self):
        TRACK_B_MODE = False
        OBSERVE_LAYER_PROBES = os.environ.get("OBSERVE_LAYER_PROBES", "0") == "1"
        VAL_LOSS_EVERY = int(os.environ.get("VAL_LOSS_EVERY", "1"))
        if TRACK_B_MODE:
            OBSERVE_LAYER_PROBES = False
            VAL_LOSS_EVERY = 0
        self.assertFalse(OBSERVE_LAYER_PROBES)
        self.assertEqual(VAL_LOSS_EVERY, 1)


class TestEqualTimeConfigEnvs(unittest.TestCase):
    """The env sets track_b_equal_time.py emits must be ones train.py accepts.

    Regression: an earlier revision emitted DOC_MASK=seg, which train.py
    rejects at import ("DOC_MASK must be 0 or 1" -- the mode lives in
    DOC_MASK_MODE). These checks mirror train.py's env validation contracts
    for every key the tool emits, so a bad launch command fails here instead
    of on the GPU box.
    """

    # Mirrors train.py's accepted values (train.py raises ValueError otherwise).
    BINARY_KEYS = {"DOC_MASK", "NGRAM_SPARSE_GRAD", "NGRAM_STATE_ROWWISE", "TIE_EMBED"}
    ENUM_KEYS = {
        "DOC_MASK_MODE": {"both", "seg", "window"},
        "COMPILE_MODE": {
            "max-autotune",
            "max-autotune-no-cudagraphs",
            "default",
            "reduce-overhead",
        },
    }
    NONNEG_INT_KEYS = {"NGRAM_TABLE_MULT", "NGRAM_FOURGRAM_MULT", "NGRAM_FIVEGRAM_MULT"}
    FLOAT_KEYS = {"WARMDOWN_RATIO", "WARMUP_RATIO", "MATRIX_LR"}

    def parsed_envs(self):
        for config, env_str in track_b_equal_time.CONFIG_ENVS.items():
            tokens = env_str.split()
            for token in tokens:
                self.assertIn("=", token, f"{config}: malformed env token {token!r}")
                key, value = token.split("=", 1)
                yield config, key, value

    def test_every_emitted_env_value_is_accepted_by_train_py(self):
        for config, key, value in self.parsed_envs():
            label = f"{config}: {key}={value}"
            if key in self.BINARY_KEYS:
                self.assertIn(value, {"0", "1"}, f"{label} (train.py: must be 0 or 1)")
            elif key in self.ENUM_KEYS:
                self.assertIn(value, self.ENUM_KEYS[key], label)
            elif key in self.NONNEG_INT_KEYS:
                self.assertGreaterEqual(int(value), 0, label)
            elif key in self.FLOAT_KEYS:
                float(value)
            else:
                self.fail(f"{label}: key not covered by this contract test; "
                          "add it to the appropriate bucket")

    def test_seg_mask_is_expressed_as_mode_not_mask_value(self):
        for name in ("sota1", "sota2"):
            env = track_b_equal_time.CONFIG_ENVS[name]
            self.assertNotIn("DOC_MASK=seg", env)
            self.assertIn("DOC_MASK=1", env)
            self.assertIn("DOC_MASK_MODE=seg", env)

    def test_sota2_is_sota1_plus_fourgram(self):
        sota1 = set(track_b_equal_time.CONFIG_ENVS["sota1"].split())
        sota2 = set(track_b_equal_time.CONFIG_ENVS["sota2"].split())
        self.assertEqual(sota2 - sota1, {"NGRAM_FOURGRAM_MULT=256"})


class TestEqualTimeResultFilePattern(unittest.TestCase):
    """Result-file matching must accept underscore config names and reject
    the summary/legacy files that share the *_s*.json glob."""

    def test_matches_simple_and_underscore_config_names(self):
        for name, expected in (
            ("baseline_s42.json", ("baseline", "42")),
            ("sota1_s44.json", ("sota1", "44")),
            ("capacity_only_s42.json", ("capacity_only", "42")),
        ):
            match = track_b_equal_time.RESULT_FILE_RE.match(name)
            self.assertIsNotNone(match, name)
            self.assertEqual((match.group(1), match.group(2)), expected)

    def test_rejects_summary_and_legacy_contaminated_files(self):
        for name in (
            "multi_seed_summary.json",
            "track_b_baseline.json",
            "track_b_sota1.json",
            "track_b_sota2.json",
        ):
            self.assertIsNone(track_b_equal_time.RESULT_FILE_RE.match(name), name)


class TestTrackBFairnessProtocol(unittest.TestCase):
    def schema_v2_result(self, config_name, seed=42, step_times=None):
        config = dict(track_b_equal_time.EXPECTED_CONFIG[config_name])
        return {
            "schema_version": 2,
            "mode": "track_b_bare",
            "seed": seed,
            "step_times_ms": step_times or [100.0, 110.0, 120.0],
            "cumulative_steady_time_s": 3.0,
            "timing_protocol": {
                "scope": "synchronized_steady_state_training_step",
                "is_end_to_end_wall_clock": False,
            },
            "runtime": {
                "device_name": "NVIDIA H200",
                "gpu_uuid": "GPU-test",
                "paired_run_id": f"seed{seed}",
            },
            "code_hashes": {
                name: "a" * 64
                for name in ("train.py", "lib.py", "prepare.py", "data_split.json")
            },
            "config": config,
        }

    def test_uses_run_as_timing_unit_not_pooled_steps(self):
        # Pooled median would be 100 because the first run contributes many
        # samples. Median-of-run-medians gives each independent run equal weight.
        runs = [
            self.schema_v2_result("baseline", 42, [100.0] * 101),
            self.schema_v2_result("baseline", 43, [200.0]),
            self.schema_v2_result("baseline", 44, [300.0]),
        ]
        self.assertEqual(track_b_equal_time.median_of_run_medians(runs), 200.0)

    def test_rejects_mislabeled_config(self):
        result = self.schema_v2_result("sota2")
        result["config"]["NGRAM_FOURGRAM_MULT"] = 0
        with self.assertRaisesRegex(ValueError, "mislabeled"):
            track_b_equal_time.validate_result("sota2", result)

    def test_rejects_legacy_result_by_default(self):
        legacy = self.schema_v2_result("baseline")
        legacy.pop("schema_version")
        legacy.pop("timing_protocol")
        legacy.pop("runtime")
        with self.assertRaisesRegex(ValueError, "schema-v1"):
            track_b_equal_time.validate_result("baseline", legacy)

    def test_rejects_cross_arm_code_or_gpu_mismatch(self):
        baseline = self.schema_v2_result("baseline")
        sota1 = self.schema_v2_result("sota1")
        by_config = {"baseline": {42: baseline}, "sota1": {42: sota1}}
        track_b_equal_time.validate_paired_provenance(
            by_config, ["baseline", "sota1"], [42]
        )

        sota1["runtime"]["gpu_uuid"] = "GPU-other"
        with self.assertRaisesRegex(ValueError, "physical GPU"):
            track_b_equal_time.validate_paired_provenance(
                by_config, ["baseline", "sota1"], [42]
            )
        sota1["runtime"]["gpu_uuid"] = "GPU-test"
        sota1["code_hashes"]["train.py"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            track_b_equal_time.validate_paired_provenance(
                by_config, ["baseline", "sota1"], [42]
            )

    def test_equal_time_command_uses_direct_time_stop_and_disables_probes(self):
        command = track_b_equal_time.build_time_command("sota2", 42, 375)
        self.assertIn("STOP_MODE=time", command)
        self.assertIn("TIME_BUDGET=375", command)
        self.assertIn("TRACK_B_MODE=1", command)
        self.assertIn("VAL_LOSS_EVERY=0", command)
        self.assertIn("VAL_BPB_PROBE_EVERY=0", command)
        self.assertIn("TRACK_B_PAIR_ID=seed42", command)
        self.assertNotIn("STOP_MODE=steps", command)

    def test_arm_order_is_counterbalanced_across_seed_blocks(self):
        names = ["baseline", "sota1", "sota2"]
        self.assertEqual(
            track_b_equal_time.counterbalanced_order(names, 0),
            ["baseline", "sota1", "sota2"],
        )
        self.assertEqual(
            track_b_equal_time.counterbalanced_order(names, 1),
            ["sota1", "sota2", "baseline"],
        )
        self.assertEqual(
            track_b_equal_time.counterbalanced_order(names, 2),
            ["sota2", "baseline", "sota1"],
        )

    def test_train_result_records_fairness_critical_provenance(self):
        source = (
            Path(__file__).resolve().parents[1] / "train.py"
        ).read_text(encoding="utf-8")
        for token in (
            '"schema_version": 2',
            '"timing_protocol": {',
            '"time_stop_clock": (',
            '"is_end_to_end_wall_clock": False',
            '"NGRAM_FOURGRAM_MULT": NGRAM_FOURGRAM_MULT',
            '"NGRAM_SPARSE_GRAD": NGRAM_SPARSE_GRAD',
            '"NGRAM_STATE_ROWWISE": NGRAM_STATE_ROWWISE',
            '"device_name": torch.cuda.get_device_name(device)',
            '"gpu_uuid": _track_b_gpu_uuid',
            '"code_hashes": _track_b_code_hashes',
            "compute_complete = training_loop_wall_time_s >= TIME_BUDGET",
            "_track_b_cumulative_time if TRACK_B_MODE else total_training_time",
        ):
            self.assertIn(token, source)

    def test_flop_telemetry_does_not_count_stored_embedding_rows_as_dense(self):
        source = (
            Path(__file__).resolve().parents[1] / "train.py"
        ).read_text(encoding="utf-8")
        method = source.split("    def estimate_flops(self):", 1)[1].split(
            "    def num_scaling_params(self):", 1
        )[0]
        self.assertIn("if isinstance(module, nn.Linear)", method)
        self.assertIn('ATTN_BACKEND != "sdpa"', method)
        self.assertIn('DOC_MASK_MODE in ("both", "window")', method)
        self.assertNotIn("sum(p.numel() for p in self.parameters())", method)


if __name__ == "__main__":
    unittest.main()
