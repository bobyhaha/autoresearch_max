import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BaselineTrainingSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_source = (ROOT / "train.py").read_text(encoding="utf-8")
        cls.train_tree = ast.parse(cls.train_source)
        cls.lib_source = (ROOT / "lib.py").read_text(encoding="utf-8")
        cls.observable_source = (ROOT / "observable.py").read_text(encoding="utf-8")
        cls.remote_launcher_source = (ROOT / "run_h200_training.sh").read_text(
            encoding="utf-8"
        )
        cls.ab_launcher_source = (ROOT / "tools/launch_ab.sh").read_text(
            encoding="utf-8"
        )
        cls.ab_monitor_source = (ROOT / "tools/monitor_ab.sh").read_text(
            encoding="utf-8"
        )

    def test_training_loss_observable_is_emitted(self):
        self.assertIn('"raw_train_loss"', self.observable_source)
        self.assertIn("build_step_observables", self.train_source)

    def test_current_schedule_entrypoints_exist(self):
        # The LR / momentum / beta2 / weight-decay schedules are computed INLINE in the
        # training loop (single source of truth). The old module-level get_* copies were
        # removed 2026-07-21 as a drift trap; this guards the LIVE inline computation so
        # a passing test can no longer mean "the dead copy still exists".
        src = self.train_source
        self.assertIn("lrm_muon = ", src)
        self.assertIn("muon_momentum = ", src)
        self.assertIn("muon_weight_decay = base_wd", src)
        self.assertIn("adam_beta1 = ADAM_BETAS[0]", src)
        self.assertIn("muon_beta2 = MUON_BETA2_PEAK", src)

    def test_validation_bpb_is_endpoint_metric(self):
        self.assertIn("val_bpb = evaluate_bpb", self.train_source)
        self.assertIn('print(f"val_bpb:', self.train_source)

    def test_control_transport_settings_and_exact_exposure_are_attested(self):
        resolved = self.train_source.index('"RESOLVED_CONFIG: "')
        for declaration in (
            'WINDOW_PATTERN = os.environ.get("WINDOW_PATTERN", "SSSL").upper()',
            'TOTAL_BATCH_SIZE = int(os.environ.get("TOTAL_BATCH_SIZE", str(2**18)))',
            'DEVICE_BATCH_SIZE = int(os.environ.get("DEVICE_BATCH_SIZE", "128"))',
        ):
            self.assertLess(self.train_source.index(declaration), resolved)
        for resolved_field in (
            "WINDOW_PATTERN={WINDOW_PATTERN}",
            "DEVICE_BATCH_SIZE={DEVICE_BATCH_SIZE}",
            "TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE}",
        ):
            self.assertIn(resolved_field, self.train_source)
        self.assertIn('print(f"total_tokens:     {total_tokens}")', self.train_source)
        self.assertIn(
            'print(f"charged_training_seconds: {total_training_time:.9f}")',
            self.train_source,
        )

    def test_environment_choices_fail_closed(self):
        self.assertIn('ATTN_BACKEND not in {"sdpa", "fa3", "fa4"}', self.train_source)
        self.assertIn('MLP_TYPE not in {"sqrelu", "swiglu"}', self.train_source)
        self.assertIn(
            'STOP_MODE not in {"steps", "time", "walltime"}',
            self.train_source,
        )
        self.assertIn('_cautious_update_raw not in {"0", "1"}', self.train_source)

    def test_attention_backend_is_selected_explicitly(self):
        self.assertIn('elif ATTN_BACKEND == "fa4":', self.train_source)
        self.assertIn('elif ATTN_BACKEND == "fa3":', self.train_source)
        self.assertNotIn("elif cap[0] >= 10:", self.train_source)

    def test_disabled_cautious_update_preserves_baseline_work(self):
        self.assertIn("if cautious:\n        raw_grad = stacked_grads.clone()", self.train_source)
        self.assertNotIn("self._cautious_t", self.train_source)

    def test_compute_completion_matches_stop_mode(self):
        self.assertIn("step >= MAX_STEPS", self.train_source)
        self.assertIn(
            "_track_b_cumulative_time if TRACK_B_MODE else total_training_time",
            self.train_source,
        )
        self.assertIn(
            "compute_complete = training_loop_wall_time_s >= TIME_BUDGET",
            self.train_source,
        )
        self.assertIn('print(f"compute_complete: {int(compute_complete)}")', self.train_source)

    def test_hopper_flash_attention_candidate_exists(self):
        self.assertIn('ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "sdpa")', self.train_source)
        self.assertIn('from kernels import get_kernel', self.train_source)
        self.assertIn('kernels-community/flash-attn3', self.train_source)

    def test_time_budget_is_five_minutes(self):
        self.assertIn("TIME_BUDGET = 300", self.lib_source)

    def test_launchers_require_exact_gate_authorization(self):
        self.assertIn("run_stage.py", self.remote_launcher_source)
        self.assertIn("retired", self.remote_launcher_source.lower())
        self.assertIn("run_stage.py", self.ab_launcher_source)
        self.assertIn("retired", self.ab_launcher_source.lower())
        for relative in (
            "tools/launch_treat.sh",
            "tools/launch_treat_2gpu.sh",
            "tools/launch_track_b_baseline.sh",
            "tools/walltime_campaign.sh",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("run_stage.py", source, relative)
            self.assertIn("retired", source.lower(), relative)
            self.assertNotIn("train.py", source, relative)
        self.assertIn('NEED=${2:-3}', self.ab_monitor_source)

    def test_ngram_ve_lr_candidate_is_frozen(self):
        registry = (ROOT / "research/toolkit/available/interventions.jsonl").read_text(encoding="utf-8")
        self.assertIn('"NGRAM_VE_LR_SCALE":1.375', registry)

    def test_ngram_ve_lr_1350_candidate_is_frozen(self):
        registry = (ROOT / "research/toolkit/available/interventions.jsonl").read_text(encoding="utf-8")
        self.assertIn('"NGRAM_VE_LR_SCALE":1.35', registry)

    def test_matrix_lr_0045_candidate_is_frozen(self):
        registry = (ROOT / "research/toolkit/available/interventions.jsonl").read_text(encoding="utf-8")
        self.assertIn('"MATRIX_LR":0.045', registry)

    def test_ngram_beta2_09995_candidate_is_frozen(self):
        registry = (ROOT / "research/toolkit/available/interventions.jsonl").read_text(encoding="utf-8")
        self.assertIn('"NGRAM_VE_BETAS":[0.5,0.9995]', registry)

    def test_ngram_beta2_warmdown_099995_candidate_is_frozen(self):
        registry = (ROOT / "research/toolkit/available/interventions.jsonl").read_text(encoding="utf-8")
        self.assertIn('"NGRAM_VE_BETA2_WARMDOWN":0.99995', registry)

    def test_adam_demon_final_beta1_060_candidate_is_frozen(self):
        registry = (ROOT / "research/toolkit/available/interventions.jsonl").read_text(encoding="utf-8")
        self.assertIn('"DEMON_FINAL_BETA1":0.6', registry)

    def test_beta09995_demon060_stack_is_frozen(self):
        registry = (ROOT / "research/toolkit/available/interventions.jsonl").read_text(encoding="utf-8")
        self.assertIn('"NGRAM_VE_BETAS":[0.5,0.9995]', registry)
        self.assertIn('"DEMON_FINAL_BETA1":0.6', registry)

    def test_beta09996_stack_refinement_is_frozen(self):
        registry = (ROOT / "research/toolkit/available/interventions.jsonl").read_text(encoding="utf-8")
        self.assertIn('"NGRAM_VE_BETAS":[0.5,0.9996]', registry)

    def test_beta099955_stack_midpoint_is_frozen(self):
        registry = (ROOT / "research/toolkit/available/interventions.jsonl").read_text(encoding="utf-8")
        self.assertIn('"NGRAM_VE_BETAS":[0.5,0.99955]', registry)

    def test_stack_matrix_lr_00375_candidate_is_frozen(self):
        registry = (ROOT / "research/toolkit/available/interventions.jsonl").read_text(encoding="utf-8")
        self.assertIn('"MATRIX_LR":0.0375', registry)

    def test_rsi_daniel_baseline_config_is_frozen(self):
        # Fresh start 2026-07-21: the baseline was reset to the rsi/daniel
        # hyperparameters (the prior 0.0405/1.375/0.99955/0.60/0.02 "validated
        # stack" was a contaminated-scope artifact and was reverted). This guards
        # the current baseline defaults against accidental drift.
        # ADOPTED 2026-07-28: MATRIX_LR became env-gated with default 0.03 when the
        # config was re-targeted to the sota_swin4 lineage. The old frozen constant
        # 0.04 is still reachable via the env flag and is what the pre-retarget
        # configuration uses; this guards the DEFAULT, not the reachable range.
        self.assertIn('MATRIX_LR = float(os.environ.get("MATRIX_LR", "0.03"))', self.train_source)
        self.assertIn("NGRAM_VE_LR_SCALE = 1.0", self.train_source)
        self.assertIn("NGRAM_VE_BETAS = (0.5, 0.999)", self.train_source)
        self.assertIn("NGRAM_VE_BETA2_WARMDOWN = 0.9999", self.train_source)
        self.assertIn("DEMON_FINAL_BETA1 = 0.55", self.train_source)
        self.assertIn("FINAL_LR_FRAC = 0.05", self.train_source)
        self.assertIn("ADAM_WARMDOWN_RATIO = 0.65", self.train_source)
        self.assertIn("UNEMBEDDING_LR = 0.004", self.train_source)
        self.assertIn("EMBEDDING_LR = 0.6", self.train_source)
        self.assertIn("WEIGHT_DECAY = 0.1", self.train_source)
        self.assertIn("SCALAR_LR = 0.8", self.train_source)
        self.assertIn("WARMDOWN_RATIO = 0.95", self.train_source)
        self.assertIn("MUON_BETA2_WARMDOWN = 0.97", self.train_source)


if __name__ == "__main__":
    unittest.main()
