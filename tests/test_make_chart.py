import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CampaignChartTest(unittest.TestCase):
    def test_generator_applies_typed_chart_policy_without_mutating_inputs(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shutil.copy2(project_root / "make_chart.py", root / "make_chart.py")
            (root / "research" / "experiments" / "runs").mkdir(parents=True)
            (root / "research" / "knowledge" / "internal").mkdir(parents=True)
            (root / "research" / "refinement").mkdir(parents=True)

            legacy_rows = [
                {
                    "exp_num": 501,
                    "exp": "legacy_exp501_anchor",
                    "val_bpb": 0.927183,
                    "steps": 100,
                    "status": "baseline",
                    "ts_end": "2026-07-27T00:00:00Z",
                },
                {
                    "exp_num": 502,
                    "exp": "legacy_exp",
                    "val_bpb": 0.9,
                    "steps": 110,
                    "status": "keep",
                    "ts_end": "2026-07-27T00:05:00Z",
                },
            ]
            run_rows = [
                {
                    "run_id": "run_chart_admissible_s42",
                    "experiment_id": "exp_chart_admissible",
                    "arm_id": "treatment",
                    "role": "treatment",
                    "seed": 42,
                    "status": "complete",
                    "started_at": "2026-07-27T00:05:00Z",
                    "ended_at": "2026-07-27T00:10:00Z",
                    # Governed points remain in the chart even above the legacy
                    # readability cutoff.
                    "outcome_values": {"val_bpb": 1.1, "num_steps": 120},
                },
                {
                    "run_id": "run_evidence_denied_s43",
                    "experiment_id": "exp_evidence_denied",
                    "arm_id": "treatment",
                    "role": "treatment",
                    "seed": 43,
                    "status": "complete",
                    "started_at": "2026-07-27T00:10:00Z",
                    "ended_at": "2026-07-27T00:15:00Z",
                    "outcome_values": {"val_bpb": 0.8, "num_steps": 125},
                },
                {
                    "run_id": "run_terminal_invalid_complete_s44",
                    "experiment_id": "exp_terminal_invalid",
                    "arm_id": "control",
                    "role": "baseline",
                    "seed": 44,
                    "status": "complete",
                    "started_at": "2026-07-27T00:15:00Z",
                    "ended_at": "2026-07-27T00:20:00Z",
                    "outcome_values": {"val_bpb": 0.7, "num_steps": 130},
                },
            ]
            evidence_rows = [
                {
                    "evidence_id": "evd_generic_chart_denial",
                    "experiment_id": "exp_evidence_denied",
                    "run_ids": ["run_evidence_denied_s43"],
                    "facts": {"chart_point_permitted": False},
                }
            ]
            update_rows = [
                {
                    "update_id": "upd_generic_terminal_invalid",
                    "experiment_id": "exp_terminal_invalid",
                    "result": "invalid",
                }
            ]
            legacy_path = root / "campaign_log.jsonl"
            runs_path = root / "research" / "experiments" / "runs" / "runs.jsonl"
            evidence_path = (
                root / "research" / "knowledge" / "internal" / "run_evidence.jsonl"
            )
            updates_path = (
                root / "research" / "refinement" / "evidence_updates.jsonl"
            )
            legacy_path.write_text(
                "".join(json.dumps(row) + "\n" for row in legacy_rows),
                encoding="utf-8",
            )
            runs_path.write_text(
                "".join(json.dumps(row) + "\n" for row in run_rows),
                encoding="utf-8",
            )
            evidence_path.write_text(
                "".join(json.dumps(row) + "\n" for row in evidence_rows),
                encoding="utf-8",
            )
            updates_path.write_text(
                "".join(json.dumps(row) + "\n" for row in update_rows),
                encoding="utf-8",
            )
            legacy_before = legacy_path.read_bytes()
            runs_before = runs_path.read_bytes()
            evidence_before = evidence_path.read_bytes()
            updates_before = updates_path.read_bytes()

            subprocess.run(
                [sys.executable, "make_chart.py"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(legacy_path.read_bytes(), legacy_before)
            self.assertEqual(runs_path.read_bytes(), runs_before)
            self.assertEqual(evidence_path.read_bytes(), evidence_before)
            self.assertEqual(updates_path.read_bytes(), updates_before)
            chart = (root / "charts" / "campaign.html").read_text(encoding="utf-8")
            payload_match = re.search(r"const DATA=(.*?);\nconst BASE=", chart, re.DOTALL)
            self.assertIsNotNone(payload_match)
            data = json.loads(payload_match.group(1))

            self.assertEqual(
                [row["source"] for row in data],
                [
                    "legacy_campaign",
                    "legacy_campaign",
                    "run_record",
                ],
            )
            self.assertEqual(data[0]["seq"], 501)
            self.assertEqual(data[0]["val_bpb"], 0.927183)
            self.assertTrue(data[0]["anchor"])
            self.assertFalse(data[0]["sota"])
            self.assertEqual(
                data[0]["milestone"], "0.927183 historical chart origin"
            )
            governed = next(
                row for row in data if row["exp"] == "run_chart_admissible_s42"
            )
            self.assertEqual(governed["experiment_id"], "exp_chart_admissible")
            self.assertEqual(governed["arm_id"], "treatment")
            self.assertEqual(governed["seed"], 42)
            self.assertEqual(governed["val_bpb"], 1.1)
            self.assertEqual(governed["t_min"], 10.0)
            self.assertFalse(governed["sota"])
            self.assertNotIn(
                "run_evidence_denied_s43", {row["exp"] for row in data}
            )
            self.assertNotIn(
                "run_terminal_invalid_complete_s44", {row["exp"] for row in data}
            )
            self.assertIn("1 with val_bpb", chart)
            self.assertIn(
                '<div class=v>0</div><div class=k>post-origin governed SOTA milestones',
                chart,
            )
            self.assertIn("0.927183 historical chart origin", chart)
            self.assertNotIn("SOTA anchor", chart)
            self.assertIn("experiment / run number (starts at 501)", chart)


if __name__ == "__main__":
    unittest.main()
