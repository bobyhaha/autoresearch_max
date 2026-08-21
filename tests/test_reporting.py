import unittest
from datetime import datetime, timezone

from vibeautoresearch.core import SchemaError
from vibeautoresearch.reporting import HourlyResearchReport


def hourly_report(**changes) -> HourlyResearchReport:
    values = {
        "report_id": "hrp_g2_0001",
        "version": 1,
        "challenge_id": "walltime_5min_h200",
        "challenge_selection_fingerprint": "a" * 16,
        "period_started_at": "2026-07-25T05:00:00Z",
        "period_ended_at": "2026-07-25T06:00:00Z",
        "created_at": "2026-07-25T06:00:00Z",
        "created_by": "test",
        "title": "One hour of bounded exploration",
        "abstract": "The campaign tested one systems hypothesis and retained uncertainty.",
        "methods": "Paired runs were interpreted only through registered evidence.",
        "directions_explored": ("systems_kernel",),
        "idea_ids": ("idea_kernel",),
        "experiment_ids": ("exp_kernel",),
        "run_ids": ("run_kernel",),
        "findings": (
            {
                "statement": "The launch path remained operational.",
                "assessment": "operational",
                "evidence_ids": [],
                "run_ids": [],
                "limitations": ["This is not an effect estimate."],
            },
        ),
        "negative_results": ("No adoptable effect was established.",),
        "limitations": ("The hour contained only one completed pair.",),
        "decisions": ("Continue only if the preregistered checkpoint clears.",),
        "next_hour_plan": (
            {
                "direction": "optimizer",
                "question": "Can an optimizer change improve quality per step?",
                "why_now": "It diversifies away from systems kernels.",
                "stop_condition": "Stop at the first failed funnel checkpoint.",
            },
        ),
    }
    values.update(changes)
    return HourlyResearchReport(**values)


class HourlyResearchReportTest(unittest.TestCase):
    def test_record_renders_a_reproducible_summary_paper(self):
        record = hourly_report()
        payload = record.to_dict()

        self.assertEqual(HourlyResearchReport.from_dict(payload), record)
        self.assertIn("# One hour of bounded exploration", record.render_markdown())
        self.assertIn("## Next-hour plan", record.render_markdown())

    def test_supported_finding_requires_structured_evidence(self):
        findings = (
            {
                "statement": "The treatment improves BPB.",
                "assessment": "supported",
                "evidence_ids": [],
                "run_ids": ["run_kernel"],
                "limitations": ["One frame only."],
            },
        )

        with self.assertRaisesRegex(SchemaError, "requires structured evidence"):
            hourly_report(findings=findings)

    def test_period_and_clock_require_honest_timezones(self):
        with self.assertRaisesRegex(SchemaError, "include a timezone"):
            hourly_report(created_at="2026-07-25T06:00:00")
        with self.assertRaisesRegex(SchemaError, "cannot precede"):
            hourly_report(period_ended_at="2026-07-25T04:00:00Z")

    def test_timestamp_comparison_is_timezone_aware(self):
        record = hourly_report()
        self.assertLess(
            datetime.fromisoformat(record.period_started_at.replace("Z", "+00:00")),
            datetime.now(timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
