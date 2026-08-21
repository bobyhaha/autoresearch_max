import json
import math
import tempfile
import unittest
from pathlib import Path

from observable import ObservableRegister, build_step_observables, format_observable_line


class ObservableRegisterTest(unittest.TestCase):
    def test_context_and_flush_are_applied_once(self):
        register = ObservableRegister()
        register.set_context("train")
        register.add("loss", 1.25)

        self.assertEqual(register.flush(), {"train.loss": 1.25})
        self.assertEqual(register.flush(), {})

    def test_nonfinite_and_unconvertible_values_are_ignored(self):
        register = ObservableRegister()
        register.add("nan", math.nan)
        register.add("inf", math.inf)
        register.add("object", object())

        self.assertEqual(register.flush(), {})

    def test_step_observables_compute_gap_and_units(self):
        values = build_step_observables(
            progress=0.25,
            train_loss=1.5,
            val_loss=1.75,
            dt=0.2,
            tok_per_sec=125_000,
            lrm_muon=1.0,
            lrm_adam=0.5,
            muon_momentum=0.9,
            muon_weight_decay=0.1,
            mfu=42.0,
        )

        self.assertEqual(values["train_val_gap"], -0.25)
        self.assertEqual(values["step_time_ms"], 200.0)
        self.assertEqual(values["tok_per_sec_k"], 125.0)

    def test_format_is_stable_and_sorted(self):
        line = format_observable_line(3, None, {"z": 2.0, "a": 1.0})

        self.assertEqual(line, "observable: step=3 epoch=-1 a=1 z=2")

    def test_structured_curves_match_dashboard_series_contract(self):
        register = ObservableRegister()
        register.record_step(3, 1, {"train.loss": 1.25})
        register.record_step(4, 1, {"train.loss": 1.0, "val.loss": 1.5})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observable_curves.json"
            register.write_curves_json(path, {"run_id": "test"})
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["metadata"], {"run_id": "test"})
        self.assertEqual(payload["num_steps"], 2)
        self.assertEqual(
            payload["series"]["val.loss"],
            [{"step": 4, "value": 1.5}],
        )


if __name__ == "__main__":
    unittest.main()
