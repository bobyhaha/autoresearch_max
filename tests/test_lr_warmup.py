"""Guards that WARMUP_RATIO is CONSUMED by the schedule, not merely logged.

Between 2026-07-22 and this module's addition, WARMUP_RATIO was read from the
environment and printed into RESOLVED_CONFIG but never referenced by any LR
computation. Every run that set it trained the identical no-warmup schedule, and
the resulting three-seed "null" (evd_run_ngram256_seg_warmup002, now retracted)
measured nothing. Config-presence tests could not catch that, so these tests
assert consumption and exercise the ramp arithmetic directly.

train.py touches CUDA at import, so the pure warmup helper is extracted from the
module AST and executed standalone -- this runs the real shipped function body,
not a reimplementation of it.
"""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_warmup_helper():
    """Exec the real `warmup_lr_mult` definition out of train.py, sans imports."""
    tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "warmup_lr_mult":
            namespace = {}
            exec(compile(ast.Module([node], []), "train.py", "exec"), namespace)
            return namespace["warmup_lr_mult"]
    raise AssertionError("train.py no longer defines a module-level warmup_lr_mult")


class WarmupRatioIsConsumedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_source = (ROOT / "train.py").read_text(encoding="utf-8")

    def test_warmup_ratio_is_read_inside_the_training_loop(self):
        # The exact defect this module exists for: being defined, env-resolved and
        # printed is not consumption. Only a read from inside the training loop
        # means the knob can change what the run does. Assignments, the _envf
        # resolution and the RESOLVED_CONFIG banner all sit outside the loop and
        # therefore cannot satisfy this.
        loop = self._training_loop_node()
        reads = [
            node
            for node in ast.walk(loop)
            if isinstance(node, ast.Name)
            and node.id == "WARMUP_RATIO"
            and isinstance(node.ctx, ast.Load)
        ]
        self.assertTrue(
            reads,
            "WARMUP_RATIO is never read inside the training loop -- it is a "
            "logging-only variable and any experiment sweeping it is vacuous",
        )

    def _training_loop_node(self):
        tree = ast.parse(self.train_source)
        loops = [
            node
            for node in tree.body
            if isinstance(node, ast.While)
            and isinstance(node.test, ast.Constant)
            and node.test.value is True
        ]
        self.assertEqual(
            len(loops), 1, "expected exactly one module-level `while True:` training loop"
        )
        return loops[0]

    def test_schedule_applies_the_warmup_multiplier_to_learning_rates(self):
        src = self.train_source
        self.assertIn("lrm_warmup = ", src)
        self.assertIn("warmup_lr_mult(progress, WARMUP_RATIO, FINAL_LR_FRAC)", src)
        # The multiplier must reach both optimizer families' LR multipliers.
        self.assertIn("lrm_muon = lrm_warmup", src)
        self.assertIn("lrm_adam = lrm_warmup", src)
        # ...and must actually be written onto param groups before warmdown,
        # where the baseline schedule performs no LR writes at all.
        self.assertIn('group["lr"] = initial_lr * lrm_warmup', src)

    def test_warmup_window_is_validated_against_the_warmdown_phase(self):
        src = self.train_source
        self.assertIn("WARMUP_RATIO must be in [0.0, 1.0)", src)
        self.assertIn("overlaps the warmdown phase", src)


class WarmupRampArithmeticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.warmup_lr_mult = staticmethod(_load_warmup_helper())

    def test_disabled_warmup_is_an_exact_no_op(self):
        # The frozen baseline is WARMUP_RATIO=0.0; it must return exactly 1.0 so
        # the schedule is bit-identical to the pre-warmup code.
        for progress in (0.0, 0.01, 0.5, 0.999, 1.0):
            self.assertEqual(self.warmup_lr_mult(progress, 0.0, 0.05), 1.0)

    def test_ramp_starts_at_the_floor_and_reaches_full_rate(self):
        self.assertAlmostEqual(self.warmup_lr_mult(0.0, 0.02, 0.05), 0.05)
        self.assertAlmostEqual(self.warmup_lr_mult(0.01, 0.02, 0.05), 0.525)
        # At and beyond the window boundary the multiplier is exactly 1.0, so the
        # flat phase resumes the unmodified baseline LR.
        self.assertEqual(self.warmup_lr_mult(0.02, 0.02, 0.05), 1.0)
        self.assertEqual(self.warmup_lr_mult(0.5, 0.02, 0.05), 1.0)

    def test_ramp_is_monotone_nondecreasing(self):
        values = [
            self.warmup_lr_mult(i / 1000.0, 0.02, 0.05) for i in range(0, 60)
        ]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later)
        self.assertLess(values[0], values[-1])


if __name__ == "__main__":
    unittest.main()
