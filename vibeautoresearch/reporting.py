"""Structured hourly research papers for an active challenge campaign."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .core import (
    SchemaError,
    check_fingerprint,
    fingerprint,
    json_mappings,
    require_enum,
    require_id,
    require_int,
    require_keys,
    require_text,
    strings,
)
from .idea_archive import RESEARCH_DIRECTIONS


FINDING_ASSESSMENTS = {
    "operational",
    "preliminary_observation",
    "supported",
    "null",
    "contradicted",
    "invalid",
}


def parse_timestamp(value: Any, field: str) -> datetime:
    """Parse an ISO-8601 timestamp and require an explicit timezone."""
    text = require_text(value, field)
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaError(f"{field} must be an ISO-8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise SchemaError(f"{field} must include a timezone")
    return result


def _validate_findings(value: Any) -> tuple[dict[str, Any], ...]:
    findings = json_mappings(value, "hourly_report.findings")
    for index, finding in enumerate(findings):
        field = f"hourly_report.findings[{index}]"
        require_keys(
            finding,
            field,
            ("statement", "assessment", "evidence_ids", "run_ids", "limitations"),
        )
        require_text(finding["statement"], f"{field}.statement")
        assessment = require_enum(
            finding["assessment"], f"{field}.assessment", FINDING_ASSESSMENTS
        )
        evidence_ids = strings(finding["evidence_ids"], f"{field}.evidence_ids")
        run_ids = strings(finding["run_ids"], f"{field}.run_ids")
        for evidence_id in evidence_ids:
            require_id(evidence_id, "evidence", f"{field}.evidence_ids")
        for run_id in run_ids:
            require_id(run_id, "run", f"{field}.run_ids")
        strings(finding["limitations"], f"{field}.limitations")
        if assessment in {"supported", "null", "contradicted"} and not evidence_ids:
            raise SchemaError(
                f"{field} assessment {assessment!r} requires structured evidence; "
                "an hourly paper cannot promote an unsupported impression"
            )
    return findings


def _validate_next_hour_plan(value: Any) -> tuple[dict[str, Any], ...]:
    plans = json_mappings(value, "hourly_report.next_hour_plan")
    if not plans:
        raise SchemaError("hourly_report.next_hour_plan must contain at least one item")
    for index, plan in enumerate(plans):
        field = f"hourly_report.next_hour_plan[{index}]"
        require_keys(
            plan,
            field,
            ("direction", "question", "why_now", "stop_condition"),
        )
        require_enum(plan["direction"], f"{field}.direction", RESEARCH_DIRECTIONS)
        require_text(plan["question"], f"{field}.question")
        require_text(plan["why_now"], f"{field}.why_now")
        require_text(plan["stop_condition"], f"{field}.stop_condition")
    return plans


@dataclass(frozen=True)
class HourlyResearchReport:
    """One immutable, evidence-aware summary paper for a campaign hour."""

    report_id: str
    version: int
    challenge_id: str
    challenge_selection_fingerprint: str
    period_started_at: str
    period_ended_at: str
    created_at: str
    created_by: str
    title: str
    abstract: str
    methods: str
    directions_explored: tuple[str, ...]
    idea_ids: tuple[str, ...]
    experiment_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    findings: tuple[Mapping[str, Any], ...]
    negative_results: tuple[str, ...]
    limitations: tuple[str, ...]
    decisions: tuple[str, ...]
    next_hour_plan: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        require_id(self.report_id, "hourly_report")
        require_int(self.version, "hourly_report.version", minimum=1)
        require_text(self.challenge_id, "hourly_report.challenge_id")
        selection = require_text(
            self.challenge_selection_fingerprint,
            "hourly_report.challenge_selection_fingerprint",
        )
        if len(selection) != 16 or any(
            char not in "0123456789abcdef" for char in selection
        ):
            raise SchemaError(
                "hourly_report.challenge_selection_fingerprint must be a "
                "16-character fingerprint"
            )
        started = parse_timestamp(
            self.period_started_at, "hourly_report.period_started_at"
        )
        ended = parse_timestamp(self.period_ended_at, "hourly_report.period_ended_at")
        created = parse_timestamp(self.created_at, "hourly_report.created_at")
        if ended < started:
            raise SchemaError(
                "hourly_report.period_ended_at cannot precede period_started_at"
            )
        if created < ended:
            raise SchemaError(
                "hourly_report.created_at cannot precede period_ended_at"
            )
        require_text(self.created_by, "hourly_report.created_by")
        require_text(self.title, "hourly_report.title")
        require_text(self.abstract, "hourly_report.abstract")
        require_text(self.methods, "hourly_report.methods")
        for direction in strings(
            self.directions_explored, "hourly_report.directions_explored"
        ):
            require_enum(
                direction,
                "hourly_report.directions_explored",
                RESEARCH_DIRECTIONS,
            )
        for idea_id in strings(self.idea_ids, "hourly_report.idea_ids"):
            require_id(idea_id, "idea", "hourly_report.idea_ids")
        for experiment_id in strings(
            self.experiment_ids, "hourly_report.experiment_ids"
        ):
            require_id(
                experiment_id, "experiment", "hourly_report.experiment_ids"
            )
        for run_id in strings(self.run_ids, "hourly_report.run_ids"):
            require_id(run_id, "run", "hourly_report.run_ids")
        _validate_findings(self.findings)
        strings(self.negative_results, "hourly_report.negative_results")
        strings(self.limitations, "hourly_report.limitations", allow_empty=False)
        strings(self.decisions, "hourly_report.decisions", allow_empty=False)
        _validate_next_hour_plan(self.next_hour_plan)

    @property
    def registry_id(self) -> str:
        return self.report_id

    def definition(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "challenge_id": self.challenge_id,
            "challenge_selection_fingerprint": self.challenge_selection_fingerprint,
            "period_started_at": self.period_started_at,
            "period_ended_at": self.period_ended_at,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "title": self.title,
            "abstract": self.abstract,
            "methods": self.methods,
            "directions_explored": list(self.directions_explored),
            "idea_ids": list(self.idea_ids),
            "experiment_ids": list(self.experiment_ids),
            "run_ids": list(self.run_ids),
            "findings": [dict(item) for item in self.findings],
            "negative_results": list(self.negative_results),
            "limitations": list(self.limitations),
            "decisions": list(self.decisions),
            "next_hour_plan": [dict(item) for item in self.next_hour_plan],
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {**self.definition(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HourlyResearchReport":
        record = cls(
            report_id=require_id(payload.get("report_id"), "hourly_report"),
            version=int(payload.get("version", 1)),
            challenge_id=require_text(
                payload.get("challenge_id"), "hourly_report.challenge_id"
            ),
            challenge_selection_fingerprint=require_text(
                payload.get("challenge_selection_fingerprint"),
                "hourly_report.challenge_selection_fingerprint",
            ),
            period_started_at=require_text(
                payload.get("period_started_at"),
                "hourly_report.period_started_at",
            ),
            period_ended_at=require_text(
                payload.get("period_ended_at"), "hourly_report.period_ended_at"
            ),
            created_at=require_text(
                payload.get("created_at"), "hourly_report.created_at"
            ),
            created_by=require_text(
                payload.get("created_by"), "hourly_report.created_by"
            ),
            title=require_text(payload.get("title"), "hourly_report.title"),
            abstract=require_text(payload.get("abstract"), "hourly_report.abstract"),
            methods=require_text(payload.get("methods"), "hourly_report.methods"),
            directions_explored=strings(
                payload.get("directions_explored"),
                "hourly_report.directions_explored",
            ),
            idea_ids=strings(payload.get("idea_ids"), "hourly_report.idea_ids"),
            experiment_ids=strings(
                payload.get("experiment_ids"), "hourly_report.experiment_ids"
            ),
            run_ids=strings(payload.get("run_ids"), "hourly_report.run_ids"),
            findings=_validate_findings(payload.get("findings")),
            negative_results=strings(
                payload.get("negative_results"),
                "hourly_report.negative_results",
            ),
            limitations=strings(
                payload.get("limitations"),
                "hourly_report.limitations",
                allow_empty=False,
            ),
            decisions=strings(
                payload.get("decisions"),
                "hourly_report.decisions",
                allow_empty=False,
            ),
            next_hour_plan=_validate_next_hour_plan(
                payload.get("next_hour_plan")
            ),
        )
        check_fingerprint(payload, record.fingerprint)
        return record

    @property
    def paper_filename(self) -> str:
        return f"{self.report_id}.md"

    def render_markdown(self) -> str:
        """Render the structured record as a compact reproducible paper."""

        def bullets(values: tuple[str, ...], *, empty: str = "None recorded.") -> str:
            if not values:
                return f"- {empty}"
            return "\n".join(f"- {value}" for value in values)

        finding_lines: list[str] = []
        for finding in self.findings:
            evidence = ", ".join(f"`{item}`" for item in finding["evidence_ids"])
            runs = ", ".join(f"`{item}`" for item in finding["run_ids"])
            provenance = "; ".join(
                item
                for item in (
                    f"evidence: {evidence}" if evidence else "",
                    f"runs: {runs}" if runs else "",
                )
                if item
            )
            finding_lines.append(
                f"- **{finding['assessment']}** — {finding['statement']}"
                + (f" ({provenance})" if provenance else "")
            )
            for limitation in finding["limitations"]:
                finding_lines.append(f"  - Limitation: {limitation}")
        if not finding_lines:
            finding_lines.append("- No evidence-bearing finding was recorded this hour.")

        plan_lines = [
            (
                f"- **{item['direction']}** — {item['question']} "
                f"Why now: {item['why_now']} Stop when: {item['stop_condition']}"
            )
            for item in self.next_hour_plan
        ]
        return (
            f"# {self.title}\n\n"
            f"- Report: `{self.report_id}`\n"
            f"- Challenge: `{self.challenge_id}`\n"
            f"- Selection: `{self.challenge_selection_fingerprint}`\n"
            f"- Period: {self.period_started_at} to {self.period_ended_at}\n"
            f"- Author: {self.created_by}\n"
            f"- Record fingerprint: `{self.fingerprint}`\n\n"
            "## Abstract\n\n"
            f"{self.abstract}\n\n"
            "## Methods and campaign accounting\n\n"
            f"{self.methods}\n\n"
            f"Directions explored: "
            f"{', '.join(f'`{item}`' for item in self.directions_explored) or 'none'}.\n\n"
            f"Ideas: {', '.join(f'`{item}`' for item in self.idea_ids) or 'none'}.\n\n"
            f"Experiments: "
            f"{', '.join(f'`{item}`' for item in self.experiment_ids) or 'none'}.\n\n"
            f"Runs: {', '.join(f'`{item}`' for item in self.run_ids) or 'none'}.\n\n"
            "## Findings\n\n"
            + "\n".join(finding_lines)
            + "\n\n## Negative and null results\n\n"
            + bullets(self.negative_results)
            + "\n\n## Limitations\n\n"
            + bullets(self.limitations)
            + "\n\n## Portfolio decisions\n\n"
            + bullets(self.decisions)
            + "\n\n## Next-hour plan\n\n"
            + "\n".join(plan_lines)
            + "\n"
        )


__all__ = [
    "FINDING_ASSESSMENTS",
    "HourlyResearchReport",
    "parse_timestamp",
]
