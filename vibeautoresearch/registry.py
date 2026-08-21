"""JSONL persistence, cross-reference validation, gates, audits, and state rendering."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, TypeVar

from .auditing import RegistryAuditor
from .campaign import CampaignBatchRecord
from .challenges import (
    ChallengeCatalog,
    ChallengeSelectionEvent,
    challenge_fingerprint,
)
from .core import (
    SchemaError,
    atomic_write_text,
    canonical_json,
    fingerprint,
    local_id,
    require_keys,
)
from .evidence import TerminalEvidenceIndex, resolve_terminal_evidence
from .experiments import ExperimentRecord, RunRecord
from .idea_archive import IdeaRecord
from .jsonl import AuditReport, JsonlRegistry, ValidationReport
from .ideas import HypothesisRecord, MechanismRecord
from .knowledge import BeliefRecord, ClaimRecord, EvidenceRecord, ObservationRecord, PaperRecord
from .refinement import AuditRecord, DecisionRecord, DeprecationRecord, EvidenceUpdateRecord
from .reporting import HourlyResearchReport, parse_timestamp
from .search_policy import (
    DIRECTION_ROUND_LIMIT,
    FAILURE_UPDATE_RESULTS,
    FUNNEL_STAGES,
    QUALIFICATION_STAGE_PAIRS,
    SUCCESS_UPDATE_RESULTS,
    authorizable_seed_count,
    evaluate_stage,
    validate_search_policy,
)
from .setup import SetupReconciliationRecord, validate_scope_key
from .toolkit import (
    CapabilityGapRecord,
    ContextRecord,
    InterventionRecord,
    ObservableRecord,
    OutcomeRecord,
    ToolProposalRecord,
)
from .validation import RegistryValidator


RecordT = TypeVar("RecordT")


class ResearchRegistry:
    """The authoritative research graph described by ``research/manifest.json``."""

    SCHEMA_VERSION = "1.8"
    SETUP_PATH = "setup/reconciliation.json"
    CHALLENGE_CATALOG_PATH = "setup/challenges.json"
    CHALLENGE_EVENTS_PATH = "setup/challenge_events.jsonl"
    LITERATURE_SYNTHESIS_PATH = "knowledge/LITERATURE_SYNTHESIS.md"
    LITERATURE_SNAPSHOT_START = "<!-- BEGIN GENERATED LITERATURE SNAPSHOT -->"
    LITERATURE_SNAPSHOT_END = "<!-- END GENERATED LITERATURE SNAPSHOT -->"
    PATHS = {
        "papers": "knowledge/external/papers.jsonl",
        "claims": "knowledge/external/claims.jsonl",
        "literature_evidence": "knowledge/external/literature_evidence.jsonl",
        "run_evidence": "knowledge/internal/run_evidence.jsonl",
        "observations": "knowledge/internal/observations.jsonl",
        "beliefs": "knowledge/beliefs.jsonl",
        "observables": "toolkit/available/observables.jsonl",
        "interventions": "toolkit/available/interventions.jsonl",
        "contexts": "toolkit/available/contexts.jsonl",
        "outcomes": "toolkit/available/outcomes.jsonl",
        "tool_proposals": "toolkit/proposed/tool_proposals.jsonl",
        "capability_gaps": "toolkit/capability_gaps.jsonl",
        "mechanisms": "ideas/mechanisms.jsonl",
        "hypotheses": "ideas/hypotheses.jsonl",
        "idea_archive": "ideas/archive.jsonl",
        "hourly_reports": "reports/hourly/reports.jsonl",
        "experiment_proposals": "experiments/proposals/experiments.jsonl",
        "gated_experiments": "experiments/gated/experiments.jsonl",
        "runs": "experiments/runs/runs.jsonl",
        "evidence_updates": "refinement/evidence_updates.jsonl",
        "audits": "refinement/audits.jsonl",
        "decisions": "refinement/decisions.jsonl",
        "deprecations": "refinement/deprecations.jsonl",
        "campaign_batches": "refinement/campaign_batches.jsonl",
    }

    def __init__(self, root: str | Path = "research"):
        self.root = Path(root)
        self.papers = JsonlRegistry(self.root / self.PATHS["papers"], PaperRecord)
        self.claims = JsonlRegistry(self.root / self.PATHS["claims"], ClaimRecord)
        self.literature_evidence = JsonlRegistry(
            self.root / self.PATHS["literature_evidence"], EvidenceRecord, append_only=True
        )
        self.run_evidence = JsonlRegistry(
            self.root / self.PATHS["run_evidence"], EvidenceRecord, append_only=True
        )
        self.observations = JsonlRegistry(self.root / self.PATHS["observations"], ObservationRecord)
        self.beliefs = JsonlRegistry(self.root / self.PATHS["beliefs"], BeliefRecord, append_only=True)
        self.observables = JsonlRegistry(self.root / self.PATHS["observables"], ObservableRecord)
        self.interventions = JsonlRegistry(self.root / self.PATHS["interventions"], InterventionRecord)
        self.contexts = JsonlRegistry(self.root / self.PATHS["contexts"], ContextRecord)
        self.outcomes = JsonlRegistry(self.root / self.PATHS["outcomes"], OutcomeRecord)
        self.tool_proposals = JsonlRegistry(self.root / self.PATHS["tool_proposals"], ToolProposalRecord)
        self.capability_gaps = JsonlRegistry(self.root / self.PATHS["capability_gaps"], CapabilityGapRecord)
        self.mechanisms = JsonlRegistry(self.root / self.PATHS["mechanisms"], MechanismRecord)
        self.hypotheses = JsonlRegistry(self.root / self.PATHS["hypotheses"], HypothesisRecord)
        self.idea_archive = JsonlRegistry(
            self.root / self.PATHS["idea_archive"], IdeaRecord, append_only=True
        )
        self.hourly_reports = JsonlRegistry(
            self.root / self.PATHS["hourly_reports"],
            HourlyResearchReport,
            append_only=True,
        )
        self.experiment_proposals = JsonlRegistry(
            self.root / self.PATHS["experiment_proposals"], ExperimentRecord
        )
        self.gated_experiments = JsonlRegistry(
            self.root / self.PATHS["gated_experiments"], ExperimentRecord, append_only=True
        )
        self.runs = JsonlRegistry(self.root / self.PATHS["runs"], RunRecord, append_only=True)
        self.evidence_updates = JsonlRegistry(
            self.root / self.PATHS["evidence_updates"], EvidenceUpdateRecord, append_only=True
        )
        self.audits = JsonlRegistry(self.root / self.PATHS["audits"], AuditRecord, append_only=True)
        self.decisions = JsonlRegistry(self.root / self.PATHS["decisions"], DecisionRecord, append_only=True)
        self.deprecations = JsonlRegistry(
            self.root / self.PATHS["deprecations"], DeprecationRecord, append_only=True
        )
        self.campaign_batches = JsonlRegistry(
            self.root / self.PATHS["campaign_batches"],
            CampaignBatchRecord,
            append_only=True,
        )

    @property
    def state_path(self) -> Path:
        return self.root / "knowledge" / "RESEARCH_STATE.md"

    @property
    def literature_synthesis_path(self) -> Path:
        return self.root / self.LITERATURE_SYNTHESIS_PATH

    def _manifest(self) -> Mapping[str, Any]:
        path = self.root / "manifest.json"
        if not path.exists():
            raise SchemaError(f"missing manifest: {path}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SchemaError(f"invalid manifest: {exc}") from exc
        if not isinstance(manifest, Mapping):
            raise SchemaError("manifest must be a JSON object")
        if str(manifest.get("schema_version")) != self.SCHEMA_VERSION:
            raise SchemaError(f"manifest schema_version must be {self.SCHEMA_VERSION}")
        if dict(manifest.get("registries", {})) != self.PATHS:
            raise SchemaError("manifest registry paths do not match the implemented schema")
        if manifest.get("research_state") != "knowledge/RESEARCH_STATE.md":
            raise SchemaError("manifest must identify the generated research-state path")
        if manifest.get("literature_synthesis") != self.LITERATURE_SYNTHESIS_PATH:
            raise SchemaError(
                "manifest must identify the literature-synthesis path"
            )
        if manifest.get("setup_reconciliation") != self.SETUP_PATH:
            raise SchemaError(
                f"manifest setup_reconciliation must be {self.SETUP_PATH!r}"
            )
        if manifest.get("challenge_catalog") != self.CHALLENGE_CATALOG_PATH:
            raise SchemaError(
                f"manifest challenge_catalog must be {self.CHALLENGE_CATALOG_PATH!r}"
            )
        if manifest.get("challenge_events") != self.CHALLENGE_EVENTS_PATH:
            raise SchemaError(
                f"manifest challenge_events must be {self.CHALLENGE_EVENTS_PATH!r}"
            )
        campaign_ledger = manifest.get("campaign_ledger")
        if not isinstance(campaign_ledger, Mapping):
            raise SchemaError("manifest must define campaign_ledger")
        if campaign_ledger.get("source_path") != "campaign_log.jsonl":
            raise SchemaError(
                "manifest campaign_ledger.source_path must be 'campaign_log.jsonl'"
            )
        coverage_start = campaign_ledger.get("coverage_start_exp_num")
        history_anchor = campaign_ledger.get("history_anchor_exp_num")
        history_digest = campaign_ledger.get("history_anchor_sha256")
        if (
            isinstance(coverage_start, bool)
            or not isinstance(coverage_start, int)
            or coverage_start < 1
        ):
            raise SchemaError(
                "manifest campaign_ledger.coverage_start_exp_num must be a "
                "positive integer"
            )
        if (
            isinstance(history_anchor, bool)
            or not isinstance(history_anchor, int)
            or history_anchor < coverage_start - 1
        ):
            raise SchemaError(
                "manifest campaign_ledger.history_anchor_exp_num must be at "
                "least coverage_start_exp_num - 1"
            )
        if (
            not isinstance(history_digest, str)
            or len(history_digest) != 64
            or any(char not in "0123456789abcdef" for char in history_digest)
        ):
            raise SchemaError(
                "manifest campaign_ledger.history_anchor_sha256 must be a "
                "lowercase SHA-256 digest"
            )
        return manifest

    def challenge_catalog(self) -> ChallengeCatalog:
        path = self.root / self.CHALLENGE_CATALOG_PATH
        if not path.exists():
            raise SchemaError(f"missing challenge catalog: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SchemaError(f"invalid challenge catalog: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SchemaError("challenge catalog must be a JSON object")
        return ChallengeCatalog.from_dict(payload)

    def challenge_events(self) -> list[ChallengeSelectionEvent]:
        path = self.root / self.CHALLENGE_EVENTS_PATH
        if not path.exists():
            raise SchemaError(f"missing challenge-selection event ledger: {path}")
        events = JsonlRegistry(
            path, ChallengeSelectionEvent, append_only=True
        ).load()
        if not events:
            raise SchemaError("challenge-selection event ledger is empty")
        generations = [event.generation for event in events]
        if generations != list(range(1, len(events) + 1)):
            raise SchemaError(
                "challenge-selection event generations must be ordered and contiguous"
            )
        return events

    def _validate_challenge_configuration(
        self, setup: SetupReconciliationRecord | None = None
    ) -> tuple[ChallengeCatalog, ChallengeSelectionEvent]:
        setup = setup or self.setup_reconciliation()
        catalog = self.challenge_catalog()
        active = self.challenge_events()[-1]
        known_secondary = {
            str(entry.get("scope_id")) for entry in setup.scopes
        }
        catalog_secondary = {
            str(entry["scope_id"])
            for entry in catalog.challenges
            if str(entry["scope_id"])
        }
        if catalog_secondary != known_secondary:
            raise SchemaError(
                "challenge catalog secondary scope IDs must exactly match setup "
                f"reconciliation scopes: catalog={sorted(catalog_secondary)}, "
                f"setup={sorted(known_secondary)}"
            )
        challenge = catalog.resolve(active.challenge_id)
        if active.catalog_fingerprint != catalog.fingerprint:
            raise SchemaError(
                "active challenge selection is stale for the current challenge catalog; "
                "select the challenge again"
            )
        if active.challenge_fingerprint != challenge_fingerprint(challenge):
            raise SchemaError(
                "active challenge selection is stale for its challenge definition; "
                "select the challenge again"
            )
        if active.setup_fingerprint != setup.fingerprint:
            raise SchemaError(
                "active challenge selection is stale for the current setup "
                "reconciliation; select the challenge again"
            )
        return catalog, active

    def challenge_for_scope(self, scope_id: str) -> Mapping[str, Any]:
        catalog, _ = self._validate_challenge_configuration()
        return catalog.resolve(scope_id)

    def resolve_scope_id(self, requested: str | None) -> str:
        """Resolve an explicit selector, or the persisted active challenge."""
        catalog, active = self._validate_challenge_configuration()
        if requested is None:
            if active.action != "activated":
                raise SchemaError(
                    f"challenge work is {active.action!r}; explicitly select a "
                    "challenge before checking or launching work"
                )
            selected = catalog.resolve(active.challenge_id)
        else:
            selected = catalog.resolve(str(requested))
            active_selected = catalog.resolve(active.challenge_id)
            if active.action != "activated":
                raise SchemaError(
                    "challenge work is stopped; an explicit scope cannot bypass "
                    "the stop event"
                )
            if selected["challenge_id"] != active_selected["challenge_id"]:
                raise SchemaError(
                    f"requested challenge {selected['challenge_id']!r} differs from "
                    f"the sticky active challenge {active_selected['challenge_id']!r}; "
                    "use select-challenge to change it"
                )
        return str(selected["scope_id"])

    def selected_challenge(
        self,
    ) -> tuple[Mapping[str, Any], ChallengeSelectionEvent]:
        """Return the last selected challenge, including when work is stopped."""
        catalog, active = self._validate_challenge_configuration()
        return catalog.resolve(active.challenge_id), active

    def append_challenge_event(
        self,
        *,
        action: str,
        challenge_selector: str,
        selected_at: str,
        selected_by: str,
        reason: str,
    ) -> ChallengeSelectionEvent:
        """Append one selection/stop event under a single flocked generation."""
        self._manifest()
        setup = self.setup_reconciliation()
        catalog = self.challenge_catalog()
        challenge = catalog.resolve(challenge_selector)
        path = self.root / self.CHALLENGE_EVENTS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                events: list[ChallengeSelectionEvent] = []
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    try:
                        events.append(
                            ChallengeSelectionEvent.from_dict(json.loads(stripped))
                        )
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        raise SchemaError(
                            f"invalid {path} line {line_number}: {exc}"
                        ) from exc
                expected = list(range(1, len(events) + 1))
                if [event.generation for event in events] != expected:
                    raise SchemaError(
                        "challenge-selection event generations are not contiguous"
                    )
                if events:
                    latest = events[-1]
                    already_current = (
                        latest.action == "activated"
                        and latest.challenge_id == str(challenge["challenge_id"])
                        and latest.catalog_fingerprint == catalog.fingerprint
                        and latest.challenge_fingerprint
                        == challenge_fingerprint(challenge)
                        and latest.setup_fingerprint == setup.fingerprint
                    )
                    if action == "activated" and already_current:
                        raise SchemaError(
                            f"challenge {latest.challenge_id!r} is already the "
                            "current active selection; redundant re-selection cannot "
                            "reset the hourly reporting clock"
                        )
                generation = len(events) + 1
                event = ChallengeSelectionEvent(
                    event_id=f"challenge_selection_{generation}",
                    generation=generation,
                    version=1,
                    action=action,
                    challenge_id=str(challenge["challenge_id"]),
                    catalog_fingerprint=catalog.fingerprint,
                    challenge_fingerprint=challenge_fingerprint(challenge),
                    setup_fingerprint=setup.fingerprint,
                    selected_at=selected_at,
                    selected_by=selected_by,
                    reason=reason,
                )
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json(event.to_dict()) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                return event
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def campaign_policy(self) -> Mapping[str, Any]:
        """Return the data-driven portfolio/reporting policy for this campaign."""
        return dict(self.challenge_catalog().campaign_policy)

    def hourly_report_status(
        self, *, now: datetime | None = None
    ) -> Mapping[str, Any]:
        """Return whether the active campaign owes its next hourly paper."""
        catalog, event = self._validate_challenge_configuration()
        challenge = catalog.resolve(event.challenge_id)
        policy = dict(catalog.campaign_policy)
        required = bool(policy.get("hourly_reports_required", False))
        interval_minutes = int(policy.get("hourly_report_interval_minutes", 0))
        if not required:
            return {
                "enabled": False,
                "due": False,
                "challenge_id": challenge["challenge_id"],
                "selection_fingerprint": event.fingerprint,
                "status": "disabled",
            }
        if interval_minutes < 1:
            raise SchemaError(
                "hourly reporting is enabled but its configured interval is invalid"
            )
        if event.action != "activated":
            return {
                "enabled": True,
                "due": False,
                "challenge_id": challenge["challenge_id"],
                "selection_fingerprint": event.fingerprint,
                "status": "challenge_stopped",
            }

        current_reports = [
            report
            for report in self.hourly_reports.load()
            if report.challenge_id == event.challenge_id
            and report.challenge_selection_fingerprint == event.fingerprint
        ]
        current_reports.sort(
            key=lambda report: (
                parse_timestamp(
                    report.period_ended_at, "hourly_report.period_ended_at"
                ),
                report.report_id,
            )
        )
        anchor_text = (
            current_reports[-1].period_ended_at
            if current_reports
            else event.selected_at
        )
        anchor = parse_timestamp(anchor_text, "hourly_report.anchor")
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise SchemaError("hourly report clock must include a timezone")
        if anchor > current_time + timedelta(seconds=5):
            raise SchemaError(
                "the active challenge/report timestamp is in the future; "
                "reselect the challenge with an honest timestamp"
            )
        due_at = anchor + timedelta(minutes=interval_minutes)
        due = current_time >= due_at
        return {
            "enabled": True,
            "due": due,
            "challenge_id": challenge["challenge_id"],
            "selection_fingerprint": event.fingerprint,
            "status": "due" if due else "current",
            "interval_minutes": interval_minutes,
            "period_started_at": anchor.isoformat(),
            "due_at": due_at.isoformat(),
            "latest_report_id": (
                current_reports[-1].report_id if current_reports else ""
            ),
        }

    def require_current_hourly_report(self) -> Mapping[str, Any]:
        """Fail closed before new GPU work when the campaign paper is overdue."""
        status = self.hourly_report_status()
        if status.get("due") is True:
            raise SchemaError(
                "the active challenge's hourly research paper is overdue "
                f"(due_at={status['due_at']}); publish a structured hourly report "
                "before authorizing more GPU work"
            )
        return status

    def _validate_hourly_report_references(
        self,
        report: HourlyResearchReport,
        *,
        data: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        data = data or self._load_all()
        all_evidence = {
            **data["literature_evidence"],
            **data["run_evidence"],
        }
        all_experiments = {
            **data["experiment_proposals"],
            **data["gated_experiments"],
        }
        self._require_refs(report.report_id, "ideas", report.idea_ids, data["idea_archive"])
        self._require_refs(
            report.report_id,
            "experiments",
            report.experiment_ids,
            all_experiments,
        )
        self._require_refs(report.report_id, "runs", report.run_ids, data["runs"])
        finding_evidence = tuple(
            str(evidence_id)
            for finding in report.findings
            for evidence_id in finding["evidence_ids"]
        )
        finding_runs = tuple(
            str(run_id)
            for finding in report.findings
            for run_id in finding["run_ids"]
        )
        self._require_refs(
            report.report_id, "finding evidence", finding_evidence, all_evidence
        )
        self._require_refs(
            report.report_id, "finding runs", finding_runs, data["runs"]
        )
        if not set(finding_runs).issubset(set(report.run_ids)):
            raise SchemaError(
                f"{report.report_id} finding run_ids must also appear in the "
                "paper's top-level run_ids"
            )
        if not set(finding_evidence).issubset(
            {
                evidence_id
                for evidence_id in all_evidence
            }
        ):
            raise SchemaError(
                f"{report.report_id} contains unregistered finding evidence"
            )
        expected_tag = f"challenge_{report.challenge_id}"
        for run_id in report.run_ids:
            run = data["runs"][run_id]
            if expected_tag not in {str(tag) for tag in run.tags}:
                raise SchemaError(
                    f"{report.report_id} mixes run {run_id!r} from another "
                    "challenge into its hourly paper"
                )
        for experiment_id in report.experiment_ids:
            experiment = all_experiments[experiment_id]
            if (
                experiment.search_policy
                and str(experiment.search_policy.get("challenge_id"))
                != report.challenge_id
            ):
                raise SchemaError(
                    f"{report.report_id} mixes experiment {experiment_id!r} from "
                    "another challenge into its hourly paper"
                )

    def publish_hourly_report(
        self,
        content: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> tuple[HourlyResearchReport, Path]:
        """Bind supplied paper content to the current challenge and write it."""
        required_fields = (
            "created_by",
            "title",
            "abstract",
            "methods",
            "directions_explored",
            "idea_ids",
            "experiment_ids",
            "run_ids",
            "findings",
            "negative_results",
            "limitations",
            "decisions",
            "next_hour_plan",
        )
        require_keys(content, "hourly_report_input", required_fields)
        challenge, event = self.selected_challenge()
        if event.action != "activated":
            raise SchemaError(
                "cannot publish an active-hour paper while challenge work is stopped"
            )
        status = self.hourly_report_status(now=now)
        if status.get("enabled") is not True:
            raise SchemaError("hourly reporting is disabled by the campaign policy")
        current_time = now or datetime.now(timezone.utc)
        current_time = current_time.astimezone(timezone.utc)
        timestamp = current_time.isoformat()
        current_reports = [
            report
            for report in self.hourly_reports.load()
            if report.challenge_selection_fingerprint == event.fingerprint
        ]
        report = HourlyResearchReport(
            report_id=f"hrp_g{event.generation}_{len(current_reports) + 1:04d}",
            version=1,
            challenge_id=str(challenge["challenge_id"]),
            challenge_selection_fingerprint=event.fingerprint,
            period_started_at=str(status["period_started_at"]),
            period_ended_at=timestamp,
            created_at=timestamp,
            created_by=str(content["created_by"]),
            title=str(content["title"]),
            abstract=str(content["abstract"]),
            methods=str(content["methods"]),
            directions_explored=tuple(content["directions_explored"]),
            idea_ids=tuple(content["idea_ids"]),
            experiment_ids=tuple(content["experiment_ids"]),
            run_ids=tuple(content["run_ids"]),
            findings=tuple(content["findings"]),
            negative_results=tuple(content["negative_results"]),
            limitations=tuple(content["limitations"]),
            decisions=tuple(content["decisions"]),
            next_hour_plan=tuple(content["next_hour_plan"]),
        )
        self._validate_hourly_report_references(report)
        paper_path = self.root / "reports" / "hourly" / report.paper_filename
        if paper_path.exists():
            raise SchemaError(f"hourly paper already exists: {paper_path}")
        self.hourly_reports.add(report)
        atomic_write_text(paper_path, report.render_markdown())
        return report, paper_path

    def setup_reconciliation(self) -> SetupReconciliationRecord:
        path = self.root / self.SETUP_PATH
        if not path.exists():
            raise SchemaError(f"missing setup reconciliation: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SchemaError(f"invalid setup reconciliation: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SchemaError("setup reconciliation must be a JSON object")
        return SetupReconciliationRecord.from_dict(payload)

    def _require_current_setup(self, scope_id: str = "") -> SetupReconciliationRecord:
        setup = self.setup_reconciliation()
        drift = setup.verify_frozen_files(self.root.parent)
        if drift:
            raise SchemaError("setup reconciliation failed: " + "; ".join(drift))
        if scope_id:
            frame = setup.scope_for(scope_id)
            if str(frame.get("status")) != "passed":
                raise SchemaError(
                    f"decision frame {scope_id!r} is {frame.get('status')!r}; "
                    "experiments in that frame are stopped"
                )
        elif setup.status != "passed":
            raise SchemaError(
                f"setup reconciliation is {setup.status!r}; experiments are stopped"
            )
        return setup

    @staticmethod
    def _missing(references: tuple[str, ...], available: Mapping[str, Any]) -> list[str]:
        return sorted(set(references) - set(available))

    @staticmethod
    def _require_refs(owner: str, label: str, references: tuple[str, ...], available: Mapping[str, Any]) -> None:
        missing = ResearchRegistry._missing(references, available)
        if missing:
            raise SchemaError(f"{owner} references missing {label}: {missing}")

    @staticmethod
    def _current_beliefs(beliefs: Mapping[str, BeliefRecord]) -> list[BeliefRecord]:
        """Return the terminal records in each append-only belief chain."""
        superseded_ids = {
            belief.supersedes_belief_id
            for belief in beliefs.values()
            if belief.supersedes_belief_id
        }
        return [
            belief
            for belief_id, belief in beliefs.items()
            if belief_id not in superseded_ids
        ]

    @staticmethod
    def _terminal_evidence(
        evidence: Mapping[str, EvidenceRecord],
    ) -> TerminalEvidenceIndex:
        """Return the validated current view over append-only evidence."""
        return resolve_terminal_evidence(evidence)

    @staticmethod
    def _scope_key(record: BeliefRecord | ExperimentRecord) -> Mapping[str, Any] | None:
        container = record.scope if isinstance(record, BeliefRecord) else record.data_policy
        candidate = container.get("scope_key")
        if candidate is None:
            return None
        return validate_scope_key(candidate, f"{record.registry_id}.scope_key")

    @classmethod
    def _matches_scope(
        cls,
        record: BeliefRecord | ExperimentRecord,
        current_scope: Mapping[str, Any],
    ) -> bool:
        scope_key = cls._scope_key(record)
        return scope_key is not None and dict(scope_key) == dict(current_scope)

    @staticmethod
    def _validate_belief_supersession(beliefs: Mapping[str, BeliefRecord]) -> None:
        """Require belief revision history to be a set of unambiguous chains."""
        successor_by_predecessor: dict[str, str] = {}
        for belief in beliefs.values():
            predecessor = belief.supersedes_belief_id
            if not predecessor:
                continue
            if predecessor == belief.belief_id:
                raise SchemaError(f"{belief.belief_id} cannot supersede itself")
            existing = successor_by_predecessor.get(predecessor)
            if existing is not None:
                raise SchemaError(
                    f"belief {predecessor} has multiple successors: "
                    f"{sorted((existing, belief.belief_id))}"
                )
            successor_by_predecessor[predecessor] = belief.belief_id

        for belief_id in beliefs:
            seen: set[str] = set()
            current = belief_id
            while current:
                if current in seen:
                    raise SchemaError(f"belief supersession cycle includes {current}")
                seen.add(current)
                current = beliefs[current].supersedes_belief_id

    @staticmethod
    def _validate_idea_parent_graph(ideas: Mapping[str, IdeaRecord]) -> None:
        """Require mutation ancestry to be an acyclic archive graph."""
        for idea in ideas.values():
            if idea.idea_id in idea.parent_idea_ids:
                raise SchemaError(f"{idea.idea_id} cannot name itself as a parent idea")
        for idea_id in ideas:
            visiting: set[str] = set()
            visited: set[str] = set()

            def visit(current: str) -> None:
                if current in visiting:
                    raise SchemaError(f"idea mutation ancestry cycle includes {current}")
                if current in visited:
                    return
                visiting.add(current)
                for parent_id in ideas[current].parent_idea_ids:
                    visit(parent_id)
                visiting.remove(current)
                visited.add(current)

            visit(idea_id)

    @staticmethod
    def _conclusively_tested_hypothesis_ids(
        evidence: Mapping[str, EvidenceRecord],
    ) -> set[str]:
        terminal_evidence = resolve_terminal_evidence(evidence).terminals
        conclusive_relations = {"supports", "opposes", "mixed"}
        return {
            hypothesis_id
            for item in terminal_evidence.values()
            if item.assessment["relation"] in conclusive_relations
            and item.assessment["strength"] in {"moderate", "strong"}
            and item.source_type != "quarantined_campaign"
            for hypothesis_id in item.hypothesis_ids
        }

    def _require_literature_assessments(
        self,
        experiment_id: str,
        hypothesis: HypothesisRecord,
    ) -> None:
        mechanisms = self.mechanisms.by_id()
        required_claim_ids = {
            claim_id
            for mechanism_id in hypothesis.mechanism_ids
            for claim_id in mechanisms[mechanism_id].claim_ids
        }
        terminal_evidence = resolve_terminal_evidence(
            self.literature_evidence.by_id()
        ).terminals
        assessed_claim_ids = {
            claim_id
            for evidence in terminal_evidence.values()
            for claim_id in evidence.claim_ids
        }
        missing = sorted(required_claim_ids - assessed_claim_ids)
        if missing:
            raise SchemaError(
                f"{experiment_id} relies on literature claims without current "
                f"terminal evidence assessments: {missing}"
            )

    @staticmethod
    def _validate_analysis_gate(experiment: ExperimentRecord) -> None:
        """Validate the evidence threshold needed by an experiment stage."""
        if experiment.stage == "offline_screen":
            return
        if experiment.stage == "pilot":
            if experiment.promotion_gate.get("verdict") != "implementation_only":
                raise SchemaError(
                    "pilot promotion_gate.verdict must be 'implementation_only'; "
                    "a one-seed pilot cannot establish an effect"
                )
            return

        noise = experiment.analysis_plan.get("noise_model")
        if not isinstance(noise, Mapping):
            raise SchemaError(
                f"{experiment.experiment_id} analysis_plan must include a noise_model"
            )
        for key in ("effective_sigma", "minimum_effect", "source"):
            if key not in noise:
                raise SchemaError(
                    f"{experiment.experiment_id} noise_model is missing {key!r}"
                )
        sigma = noise["effective_sigma"]
        minimum_effect = noise["minimum_effect"]
        if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or sigma <= 0:
            raise SchemaError("noise_model.effective_sigma must be positive")
        if (
            isinstance(minimum_effect, bool)
            or not isinstance(minimum_effect, (int, float))
            or minimum_effect < sigma
        ):
            raise SchemaError(
                "noise_model.minimum_effect must be at least effective_sigma"
            )
        if not isinstance(noise["source"], str) or not noise["source"].strip():
            raise SchemaError("noise_model.source must identify measured evidence")

    def _load_all(self) -> dict[str, dict[str, Any]]:
        return {
            name: getattr(self, name).by_id()
            for name in self.PATHS
        }

    def validate(self, *, check_generated_state: bool = True) -> ValidationReport:
        return RegistryValidator(self).validate(
            check_generated_state=check_generated_state
        )

    @staticmethod
    def _validate_executable_gate(
        experiment: ExperimentRecord,
        hypothesis: HypothesisRecord,
        observables: Mapping[str, ObservableRecord],
        interventions: Mapping[str, InterventionRecord],
        contexts: Mapping[str, ContextRecord],
        outcomes: Mapping[str, OutcomeRecord],
    ) -> None:
        executable = {"unit_tested", "cost_profiled", "available"}
        for observable_id in hypothesis.observable_ids:
            observable = observables[observable_id]
            if observable.status not in executable:
                raise SchemaError(
                    f"{experiment.experiment_id} cannot use non-executable observable {observable_id}"
                )
            availability = observable.causal_availability
            if experiment.stage != "offline_screen" and (
                availability["available_before_action"] is not True
                or availability["uses_validation"] is True
            ):
                raise SchemaError(
                    f"{observable_id} is not causally available to an online intervention"
                )
        if experiment.stage != "offline_screen":
            intervention = interventions[hypothesis.intervention_id]
            if intervention.status not in executable:
                raise SchemaError(
                    f"{experiment.experiment_id} cannot use non-executable intervention "
                    f"{hypothesis.intervention_id}"
                )
        for context_id in hypothesis.context_ids:
            if contexts[context_id].status not in executable:
                raise SchemaError(
                    f"{experiment.experiment_id} cannot use non-executable context {context_id}"
                )
        if outcomes[hypothesis.outcome_id].status not in executable:
            raise SchemaError(
                f"{experiment.experiment_id} cannot use non-executable outcome {hypothesis.outcome_id}"
            )

    @staticmethod
    def _frame_key(
        setup: SetupReconciliationRecord, scope_id: str
    ) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
        if not scope_id:
            return setup.scope_key, None
        frame = setup.scope_for(scope_id)
        return frame["scope_key"], frame

    def evaluate_search_stage(self, experiment_id: str) -> Mapping[str, Any]:
        """Return the deterministic funnel verdict from immutable RunRecords."""
        experiments = self.gated_experiments.by_id()
        if experiment_id not in experiments:
            raise SchemaError(f"gated experiment {experiment_id!r} is not registered")
        experiment = experiments[experiment_id]
        if not experiment.search_policy:
            raise SchemaError(
                f"{experiment_id} predates search policy v2 and has no staged verdict"
            )
        policy = validate_search_policy(experiment.search_policy, experiment)
        if policy["qualification"]:
            self._verify_qualification_anchors(policy["qualification"])
        return evaluate_stage(experiment, self.runs.load())

    def _consecutive_subsystem_failures(
        self, subsystem: str, *, exclude_experiment_id: str = ""
    ) -> int:
        """Count the latest policy-aware failures, resetting on a positive result."""
        updates_by_experiment: dict[str, str] = {}
        for update in self.evidence_updates.load():
            if update.result != "invalid":
                updates_by_experiment[update.experiment_id] = update.result

        candidates = [
            experiment
            for experiment in self.gated_experiments.load()
            if experiment.experiment_id != exclude_experiment_id
            and experiment.search_policy
            and str(experiment.search_policy.get("subsystem")) == subsystem
        ]
        candidates.sort(key=lambda item: (item.created_at, item.experiment_id))
        failures = 0
        runs = self.runs.load()
        for experiment in reversed(candidates):
            evaluation = evaluate_stage(experiment, runs)
            verdict = str(evaluation["verdict"])
            update_result = updates_by_experiment.get(experiment.experiment_id, "")
            if verdict in {"stop_futility", "stop_no_signal"} or (
                update_result in FAILURE_UPDATE_RESULTS
            ):
                failures += 1
                continue
            if verdict in {"promote", "adoption_candidate"} or (
                update_result in SUCCESS_UPDATE_RESULTS
            ):
                break
            # An incomplete/invalid experiment is not evidence about the direction
            # and therefore neither increments nor resets the failure streak.
        return failures

    def _direction_round_history(
        self, *, exclude_experiment_id: str = ""
    ) -> list[tuple[str, str, str]]:
        """Return launched stage-one rounds ordered by their first RunRecord."""
        first_launch: dict[str, str] = {}
        for run in self.runs.load():
            if any(str(tag).startswith("diagnostic_offbudget") for tag in run.tags):
                continue
            current = first_launch.get(run.experiment_id)
            if current is None or run.started_at < current:
                first_launch[run.experiment_id] = run.started_at

        history: list[tuple[str, str, str]] = []
        for experiment in self.gated_experiments.load():
            if (
                experiment.experiment_id == exclude_experiment_id
                or not experiment.search_policy
                or experiment.experiment_id not in first_launch
            ):
                continue
            policy = validate_search_policy(experiment.search_policy, experiment)
            if int(policy["stage_pairs"]) != FUNNEL_STAGES[0]:
                continue
            history.append(
                (
                    first_launch[experiment.experiment_id],
                    experiment.experiment_id,
                    str(policy["direction"]),
                )
            )
        history.sort()
        return history

    def _consecutive_direction_rounds(
        self, direction: str, *, exclude_experiment_id: str = ""
    ) -> int:
        """Count launched stage-1 rounds in the latest uninterrupted direction.

        A "round" is one policy-aware stage-1 experiment with at least one
        non-diagnostic RunRecord. Funnel continuations do not masquerade as new
        exploration, and off-budget diagnostics do not consume the diversity
        allowance.
        """
        consecutive = 0
        for _, _, candidate_direction in reversed(
            self._direction_round_history(
                exclude_experiment_id=exclude_experiment_id
            )
        ):
            if candidate_direction != direction:
                break
            consecutive += 1
        return consecutive

    def _direction_cooldown_remaining(
        self,
        direction: str,
        *,
        direction_round_limit: int,
        cooldown_rounds: int,
        exclude_experiment_id: str = "",
    ) -> int:
        """Prevent a one-token pivot from immediately returning to a saturated axis."""
        directions = [
            item[2]
            for item in self._direction_round_history(
                exclude_experiment_id=exclude_experiment_id
            )
        ]
        matching = [index for index, item in enumerate(directions) if item == direction]
        if not matching:
            return 0
        last_index = matching[-1]
        block_start = last_index
        while block_start > 0 and directions[block_start - 1] == direction:
            block_start -= 1
        block_length = last_index - block_start + 1
        rounds_since = len(directions) - last_index - 1
        if (
            block_length >= direction_round_limit
            and 0 < rounds_since < cooldown_rounds
        ):
            return cooldown_rounds - rounds_since
        return 0

    def _verify_qualification_anchors(
        self, qualification: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], ...]:
        """Bind every historical anchor to a current file inside research/.

        Search-policy validation binds the declared values and hash strings into
        the experiment fingerprint. This gate additionally proves that each
        frozen path still resolves to the bytes named by that hash at every
        check/authorization boundary.
        """
        root = self.root.resolve()
        verified: list[Mapping[str, Any]] = []
        for anchor in qualification["historical_anchors"]:
            declared_path = Path(str(anchor["artifact_path"]))
            if declared_path.is_absolute():
                candidate = declared_path
            elif declared_path.parts and declared_path.parts[0] == self.root.name:
                candidate = self.root.parent / declared_path
            else:
                candidate = self.root / declared_path
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise SchemaError(
                    f"qualification anchor for seed {anchor['seed']} is missing or "
                    f"unresolvable: {declared_path}"
                ) from exc
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise SchemaError(
                    f"qualification anchor for seed {anchor['seed']} resolves outside "
                    f"the research root: {declared_path} -> {resolved}"
                ) from exc
            if not resolved.is_file():
                raise SchemaError(
                    f"qualification anchor for seed {anchor['seed']} is not a regular "
                    f"file: {declared_path}"
                )
            actual_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
            expected_sha256 = str(anchor["artifact_sha256"])
            if actual_sha256 != expected_sha256:
                raise SchemaError(
                    f"qualification anchor hash mismatch for seed {anchor['seed']}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
            if int(qualification.get("integrity_version", 0)) == 2:
                text = resolved.read_text(encoding="utf-8")

                def unique(pattern: str, label: str) -> str:
                    matches = re.findall(pattern, text, re.MULTILINE)
                    if len(matches) != 1:
                        raise SchemaError(
                            f"qualification anchor seed {anchor['seed']} must "
                            f"contain exactly one terminal {label}, got "
                            f"{len(matches)}"
                        )
                    return str(matches[0])

                config_line = unique(
                    r"^RESOLVED_CONFIG:\s*(.*)$", "RESOLVED_CONFIG"
                )
                resolved_config: dict[str, str] = {}
                for token in config_line.split():
                    if "=" not in token:
                        continue
                    key, value = token.split("=", 1)
                    if key in resolved_config:
                        raise SchemaError(
                            f"qualification anchor seed {anchor['seed']} has "
                            f"duplicate resolved key {key!r}"
                        )
                    resolved_config[key] = value
                expected_seed = int(anchor["seed"])
                if (
                    resolved_config.get("SEED") != str(expected_seed)
                    or resolved_config.get("STOP_MODE") != "time"
                    or resolved_config.get("TIME_BUDGET") != "300"
                ):
                    raise SchemaError(
                        f"qualification anchor seed {expected_seed} has the "
                        "wrong seed or wall-time frame"
                    )
                parsed_bpb = float(
                    unique(
                        r"^val_bpb:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
                        "val_bpb",
                    )
                )
                parsed_steps = int(
                    unique(r"^num_steps:\s*([0-9]+)\s*$", "num_steps")
                )
                parsed_stop_mode = unique(
                    r"^stop_mode:\s*(\S+)\s*$", "stop_mode"
                )
                parsed_complete = unique(
                    r"^compute_complete:\s*([01])\s*$", "compute_complete"
                )
                displayed_tokens_m = unique(
                    r"^total_tokens_M:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
                    "total_tokens_M",
                )
                expected_tokens = int(anchor["total_tokens"])
                if (
                    parsed_bpb != float(anchor["val_bpb"])
                    or parsed_steps != int(anchor["num_steps"])
                    or parsed_stop_mode != "time"
                    or parsed_complete != "1"
                    or parsed_steps * int(anchor["total_batch_size"])
                    != expected_tokens
                    or displayed_tokens_m != f"{expected_tokens / 1e6:.1f}"
                ):
                    raise SchemaError(
                        f"qualification anchor seed {expected_seed} semantic "
                        "fields disagree with its frozen declaration"
                    )
            verified.append(
                {
                    "seed": int(anchor["seed"]),
                    "artifact_path": str(declared_path),
                    "resolved_path": str(resolved),
                    "artifact_sha256": actual_sha256,
                    **(
                        {
                            "val_bpb": float(anchor["val_bpb"]),
                            "num_steps": int(anchor["num_steps"]),
                            "total_batch_size": int(
                                anchor["total_batch_size"]
                            ),
                            "total_tokens": int(anchor["total_tokens"]),
                        }
                        if int(qualification.get("integrity_version", 0)) == 2
                        else {}
                    ),
                }
            )
        return tuple(verified)

    def _verify_qualification_authority(
        self, qualification: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Verify the immutable paper/source/runtime authority for a qualification."""
        reference = qualification["execution_authority"]
        declared_path = Path(str(reference["path"]))
        candidate = (
            declared_path
            if declared_path.is_absolute()
            else self.root / declared_path
        )
        root = self.root.resolve()
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise SchemaError(
                "qualification execution authority is missing or outside research/"
            ) from exc
        actual_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual_sha256 != str(reference["sha256"]):
            raise SchemaError(
                "qualification execution-authority SHA-256 mismatch"
            )
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SchemaError(
                "qualification execution authority is not valid JSON"
            ) from exc
        require_keys(
            payload,
            "qualification.execution_authority.payload",
            (
                "schema_version",
                "canonical_design",
                "canonical_source_sha256",
                "setup_fingerprint",
                "challenge_selection_fingerprint",
                "paper_authority",
                "local_file_hashes",
                "remote_file_hashes",
                "gpu_schedule",
            ),
        )
        if payload["schema_version"] != 1:
            raise SchemaError(
                "qualification execution authority schema_version must be 1"
            )
        if payload["canonical_design"] != "current_source_dual_anchor_all8":
            raise SchemaError(
                "qualification authority selects a non-canonical design"
            )
        if payload["gpu_schedule"] != qualification["gpu_schedule"]:
            raise SchemaError(
                "qualification authority GPU schedule differs from search policy"
            )
        repo_root = self.root.resolve().parent
        local_hashes = payload["local_file_hashes"]
        if not isinstance(local_hashes, Mapping) or not local_hashes:
            raise SchemaError(
                "qualification authority local_file_hashes must be non-empty"
            )
        for path_text, expected in local_hashes.items():
            path = (repo_root / str(path_text)).resolve()
            try:
                path.relative_to(repo_root)
            except ValueError as exc:
                raise SchemaError(
                    f"qualification local artifact escapes repository: {path_text}"
                ) from exc
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != str(expected)
            ):
                raise SchemaError(
                    f"qualification local artifact hash mismatch: {path_text}"
                )
        remote_hashes = payload["remote_file_hashes"]
        if not isinstance(remote_hashes, Mapping) or not remote_hashes:
            raise SchemaError(
                "qualification authority remote_file_hashes must be non-empty"
            )
        for path_text, expected in remote_hashes.items():
            if not str(path_text).startswith("/") or not re.fullmatch(
                r"[0-9a-f]{64}", str(expected)
            ):
                raise SchemaError(
                    "qualification remote hash bindings require absolute paths "
                    "and lowercase SHA-256 values"
                )
        paper_authority = payload["paper_authority"]
        if not isinstance(paper_authority, Mapping):
            raise SchemaError(
                "qualification paper_authority must be an object"
            )
        require_keys(
            paper_authority,
            "qualification.execution_authority.payload.paper_authority",
            (
                "status",
                "decision_id",
                "canonical_tex_path",
                "canonical_tex_sha256",
                "canonical_pdf_path",
                "canonical_pdf_sha256",
                "withdrawn_tex_path",
                "withdrawn_tex_sha256",
                "withdrawn_pdf_path",
                "withdrawn_pdf_sha256",
            ),
        )
        if (
            paper_authority["status"] != "canonical_prospective_authority"
            or paper_authority["canonical_tex_path"]
            == paper_authority["withdrawn_tex_path"]
            or paper_authority["canonical_pdf_path"]
            == paper_authority["withdrawn_pdf_path"]
        ):
            raise SchemaError(
                "qualification paper authority does not identify one canonical "
                "and one distinct withdrawn paper"
            )
        for prefix in ("canonical_tex", "canonical_pdf", "withdrawn_tex", "withdrawn_pdf"):
            path_text = str(paper_authority[f"{prefix}_path"])
            digest = str(paper_authority[f"{prefix}_sha256"])
            if local_hashes.get(path_text) != digest:
                raise SchemaError(
                    f"qualification paper authority hash is not locally bound: {prefix}"
                )
        decision_id = str(paper_authority.get("decision_id", ""))
        decisions = self.decisions.by_id()
        if decision_id not in decisions:
            raise SchemaError(
                "qualification canonical-paper DecisionRecord is missing"
            )
        decision = decisions[decision_id]
        if decision.decision != "canonicalize_paper_022_dual_anchor":
            raise SchemaError(
                "qualification paper authority decision is not canonical"
            )
        required_paper_paths = {
            str(paper_authority[f"{prefix}_path"])
            for prefix in (
                "canonical_tex",
                "canonical_pdf",
                "withdrawn_tex",
                "withdrawn_pdf",
            )
        }
        if not required_paper_paths.issubset(set(decision.affected_ids)):
            raise SchemaError(
                "qualification paper authority decision does not name every paper artifact"
            )
        if (
            "non-authorizing" not in decision.replacement_strategy
            or Path(str(paper_authority["canonical_tex_path"])).stem
            not in decision.replacement_strategy
            or Path(str(paper_authority["withdrawn_tex_path"])).stem
            not in decision.replacement_strategy
        ):
            raise SchemaError(
                "qualification paper authority decision lacks explicit supersession"
            )
        return {
            "path": str(declared_path),
            "sha256": actual_sha256,
            "canonical_source_sha256": str(
                payload["canonical_source_sha256"]
            ),
            "setup_fingerprint": str(payload["setup_fingerprint"]),
            "challenge_selection_fingerprint": str(
                payload["challenge_selection_fingerprint"]
            ),
            "paper_decision_id": decision_id,
            "remote_file_count": len(remote_hashes),
        }

    def _validate_search_gate(
        self,
        experiment: ExperimentRecord,
        *,
        scope_id: str,
        frame: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if not experiment.search_policy:
            # A one-seed pilot is allowed to spend GPU time only as a
            # non-scored implementation/mechanism diagnostic.  It must not be
            # mislabeled as the one-pair discovery stage: doing so would consume
            # an effect-funnel round and could let diagnostic telemetry drive an
            # endpoint promotion.
            analysis = experiment.analysis_plan
            gpu_authority = analysis.get("gpu_authority")
            if (
                experiment.stage != "pilot"
                or experiment.promotion_gate.get("verdict")
                != "implementation_only"
                or analysis.get("diagnostic_only") is not True
                or analysis.get("endpoint_val_bpb_measured") is not False
                or analysis.get("validation_data_access") is not False
                or not isinstance(gpu_authority, Mapping)
            ):
                raise SchemaError(
                    f"{experiment.experiment_id} has no search_policy; new effect "
                    "work must declare the 1->3->6->10 funnel, while a pilot must "
                    "freeze a diagnostic-only, non-validation GPU authority"
                )
            required_gpu_authority = {
                "round1_registered_pairs": 1,
                "second_round1_pair_allowed": False,
                "opportunistic_filler_allowed": False,
            }
            mismatched = {
                key: {
                    "expected": expected,
                    "observed": gpu_authority.get(key),
                }
                for key, expected in required_gpu_authority.items()
                if gpu_authority.get(key) != expected
            }
            if mismatched:
                raise SchemaError(
                    f"{experiment.experiment_id} diagnostic GPU authority is not "
                    f"fail-closed: {mismatched}"
                )
            if gpu_authority.get("round1_seed") != experiment.seeds[0]:
                raise SchemaError(
                    f"{experiment.experiment_id} diagnostic GPU authority seed "
                    "does not match its sole frozen seed"
                )
            ideas = self.idea_archive.by_id()
            if experiment.idea_id not in ideas:
                raise SchemaError(
                    f"{experiment.experiment_id} references missing archived idea "
                    f"{experiment.idea_id!r}"
                )
            idea = ideas[experiment.idea_id]
            if idea.status not in {"selected", "tested"}:
                raise SchemaError(
                    f"{experiment.experiment_id} idea {idea.idea_id!r} has "
                    f"non-runnable status {idea.status!r}"
                )
            if idea.novelty_check["status"] != "passed":
                raise SchemaError(
                    f"{experiment.experiment_id} idea {idea.idea_id!r} has not "
                    "passed the literature novelty gate"
                )
            if idea.hypothesis_id != experiment.hypothesis_id:
                raise SchemaError(
                    f"{experiment.experiment_id} changes the archived idea's "
                    "hypothesis"
                )
            selected_challenge = self.challenge_for_scope(scope_id)
            if str(selected_challenge["scope_id"]) != scope_id:
                raise SchemaError(
                    f"{experiment.experiment_id} diagnostic is not bound to the "
                    "sticky active challenge scope"
                )
            sibling_pilots = sorted(
                candidate.experiment_id
                for candidate in self.gated_experiments.load()
                if candidate.experiment_id != experiment.experiment_id
                and candidate.idea_id == experiment.idea_id
                and candidate.stage == "pilot"
                and not candidate.search_policy
            )
            if sibling_pilots:
                raise SchemaError(
                    f"idea {idea.idea_id!r} already has diagnostic pilot "
                    f"{sibling_pilots}; immutable diagnostic retries are not allowed"
                )
            return {
                "diagnostic_only": True,
                "effect_funnel_stage_consumed": False,
                "endpoint_scoring_authorized": False,
                "adoption_authorized": False,
                "registered_pairs": 1,
                "seed": experiment.seeds[0],
                "gpu_authority": dict(gpu_authority),
            }
        policy = validate_search_policy(experiment.search_policy, experiment)
        qualification = policy["qualification"]
        is_qualification = bool(qualification)
        qualification_integrity_version = int(
            qualification.get("integrity_version", 0)
        )
        if is_qualification and int(policy["stage_pairs"]) != QUALIFICATION_STAGE_PAIRS:
            # Defensive duplication of the schema invariant at the mutation gate.
            raise SchemaError(
                f"{experiment.experiment_id} qualification must freeze exactly "
                f"{QUALIFICATION_STAGE_PAIRS} pairs"
            )
        verified_anchors = (
            self._verify_qualification_anchors(qualification)
            if is_qualification
            else ()
        )
        verified_authority = (
            self._verify_qualification_authority(qualification)
            if is_qualification and qualification_integrity_version == 2
            else {}
        )
        if is_qualification and qualification_integrity_version == 2:
            setup = self.setup_reconciliation()
            active = self.challenge_events()[-1]
            expected_train_sha = str(
                setup.reference_code.get("local_sha256", "")
            )
            if (
                verified_authority["canonical_source_sha256"]
                != expected_train_sha
                or verified_authority["setup_fingerprint"]
                != setup.fingerprint
                or verified_authority["challenge_selection_fingerprint"]
                != active.fingerprint
            ):
                raise SchemaError(
                    "qualification execution authority is stale for the current "
                    "source/setup/challenge selection"
                )
        ideas = self.idea_archive.by_id()
        if experiment.idea_id not in ideas:
            raise SchemaError(
                f"{experiment.experiment_id} references missing archived idea "
                f"{experiment.idea_id!r}"
            )
        idea = ideas[experiment.idea_id]
        if idea.status not in {"selected", "tested"}:
            raise SchemaError(
                f"{experiment.experiment_id} idea {idea.idea_id!r} has "
                f"non-runnable status {idea.status!r}"
            )
        if idea.novelty_check["status"] != "passed":
            raise SchemaError(
                f"{experiment.experiment_id} idea {idea.idea_id!r} has not passed "
                "the literature novelty gate"
            )
        if idea.hypothesis_id != experiment.hypothesis_id:
            raise SchemaError(
                f"{experiment.experiment_id} changes the archived idea's hypothesis"
            )
        if (
            idea.direction != str(policy["direction"])
            or idea.subsystem != str(policy["subsystem"])
        ):
            raise SchemaError(
                f"{experiment.experiment_id} changes its archived idea's direction "
                "or subsystem"
            )
        selected_challenge = self.challenge_for_scope(scope_id)
        campaign_policy = self.campaign_policy()
        direction_round_limit = int(
            campaign_policy.get("direction_round_limit", DIRECTION_ROUND_LIMIT)
        )
        direction_cooldown_rounds = int(
            campaign_policy.get("direction_cooldown_rounds", 1)
        )
        expected_challenge_id = str(selected_challenge["challenge_id"])
        if str(policy["challenge_id"]) != expected_challenge_id:
            raise SchemaError(
                f"{experiment.experiment_id} search_policy.challenge_id="
                f"{policy['challenge_id']!r}, but the sticky active challenge is "
                f"{expected_challenge_id!r}"
            )
        expected_frame = str(selected_challenge["decision_frame"])
        if str(policy["decision_frame"]) != expected_frame:
            raise SchemaError(
                f"{experiment.experiment_id} search_policy.decision_frame="
                f"{policy['decision_frame']!r}, but the requested frame is "
                f"{expected_frame!r}"
            )

        stage_pairs = int(policy["stage_pairs"])
        same_idea_funnel_experiments = []
        same_idea_qualification_experiments = []
        for candidate in self.gated_experiments.load():
            if (
                candidate.experiment_id == experiment.experiment_id
                or not candidate.search_policy
                or candidate.idea_id != experiment.idea_id
            ):
                continue
            candidate_policy = validate_search_policy(
                candidate.search_policy, candidate
            )
            target = (
                same_idea_qualification_experiments
                if candidate_policy["qualification"]
                else same_idea_funnel_experiments
            )
            target.append(
                (candidate.experiment_id, int(candidate_policy["stage_pairs"]))
            )
        if (
            not is_qualification
            and stage_pairs == FUNNEL_STAGES[0]
            and same_idea_funnel_experiments
        ):
            raise SchemaError(
                f"idea {idea.idea_id!r} has already entered an experiment funnel; "
                "continue its earned stage chain or archive a genuinely new child idea"
            )
        if is_qualification and same_idea_qualification_experiments:
            raise SchemaError(
                f"idea {idea.idea_id!r} already has a standalone qualification "
                f"experiment {[item[0] for item in same_idea_qualification_experiments]}; "
                "immutable qualification retries are not allowed"
            )
        if is_qualification and qualification_integrity_version != 2:
            raise SchemaError(
                f"{experiment.experiment_id} is a preserved pre-audit "
                "qualification draft and cannot be frozen or authorized; "
                "register a new integrity_version=2 experiment"
            )
        duplicate_stage_ids = sorted(
            candidate_id
            for candidate_id, candidate_stage in same_idea_funnel_experiments
            if candidate_stage == stage_pairs
        )
        if not is_qualification and duplicate_stage_ids:
            raise SchemaError(
                f"idea {idea.idea_id!r} already has a {stage_pairs}-pair funnel "
                f"experiment {duplicate_stage_ids}; sibling retries are not allowed"
            )
        if frame is not None and stage_pairs == FUNNEL_STAGES[-1]:
            required = int(frame.get("min_seeds", FUNNEL_STAGES[-1]))
            if len(experiment.seeds) < required:
                raise SchemaError(
                    f"final frame {scope_id!r} confirmation requires >={required} seeds; "
                    f"{experiment.experiment_id!r} freezes only {len(experiment.seeds)}"
                )

        parent_id = str(policy["parent_experiment_id"])
        parent_report: Mapping[str, Any] | None = None
        if not is_qualification and stage_pairs > FUNNEL_STAGES[0]:
            gated = self.gated_experiments.by_id()
            if parent_id not in gated:
                raise SchemaError(
                    f"{experiment.experiment_id} references missing gated funnel parent "
                    f"{parent_id!r}"
                )
            parent = gated[parent_id]
            if not parent.search_policy:
                raise SchemaError(f"funnel parent {parent_id!r} has no search_policy")
            parent_policy = validate_search_policy(parent.search_policy, parent)
            expected_parent_stage = FUNNEL_STAGES[FUNNEL_STAGES.index(stage_pairs) - 1]
            if int(parent_policy["stage_pairs"]) != expected_parent_stage:
                raise SchemaError(
                    f"{experiment.experiment_id} stage {stage_pairs} requires a "
                    f"{expected_parent_stage}-pair parent, got "
                    f"{parent_policy['stage_pairs']}"
                )
            for key in ("challenge_id", "decision_frame", "direction", "subsystem"):
                if str(parent_policy[key]) != str(policy[key]):
                    raise SchemaError(
                        f"{experiment.experiment_id} changes search_policy.{key} "
                        f"across funnel stages"
                    )
            if parent.hypothesis_id != experiment.hypothesis_id:
                raise SchemaError(
                    f"{experiment.experiment_id} changes hypothesis across funnel stages"
                )
            if parent.idea_id != experiment.idea_id:
                raise SchemaError(
                    f"{experiment.experiment_id} changes idea across funnel stages"
                )
            parent_arms = [
                (
                    str(arm.get("arm_id")),
                    str(arm.get("role")),
                    str(arm.get("intervention_id")),
                    arm.get("trigger"),
                    str(arm.get("control_id")),
                )
                for arm in parent.arms
            ]
            child_arms = [
                (
                    str(arm.get("arm_id")),
                    str(arm.get("role")),
                    str(arm.get("intervention_id")),
                    arm.get("trigger"),
                    str(arm.get("control_id")),
                )
                for arm in experiment.arms
            ]
            if child_arms != parent_arms:
                raise SchemaError(
                    f"{experiment.experiment_id} changes arms across funnel stages"
                )
            if tuple(experiment.seeds[: len(parent.seeds)]) != tuple(parent.seeds):
                raise SchemaError(
                    f"{experiment.experiment_id} must preserve its parent's seeds "
                    "as the leading funnel prefix"
                )
            for key in ("predictions",):
                if policy[key] != parent_policy[key]:
                    raise SchemaError(
                        f"{experiment.experiment_id} changes search_policy.{key} "
                        "after observing an earlier stage"
                    )
            stopping = {
                key: value
                for key, value in policy["stopping"].items()
                if key != "min_futility_pairs"
            }
            parent_stopping = {
                key: value
                for key, value in parent_policy["stopping"].items()
                if key != "min_futility_pairs"
            }
            if stopping != parent_stopping:
                raise SchemaError(
                    f"{experiment.experiment_id} changes stopping thresholds "
                    "after observing an earlier stage"
                )
            parent_report = evaluate_stage(parent, self.runs.load())
            if parent_report["verdict"] != "promote":
                raise SchemaError(
                    f"funnel parent {parent_id!r} did not earn promotion: "
                    f"{parent_report['verdict']} ({parent_report['reason']})"
                )

        failures = self._consecutive_subsystem_failures(
            str(policy["subsystem"]),
            exclude_experiment_id=experiment.experiment_id,
        )
        direction_rounds = self._consecutive_direction_rounds(
            str(policy["direction"]),
            exclude_experiment_id=experiment.experiment_id,
        )
        direction_cooldown_remaining = self._direction_cooldown_remaining(
            str(policy["direction"]),
            direction_round_limit=direction_round_limit,
            cooldown_rounds=direction_cooldown_rounds,
            exclude_experiment_id=experiment.experiment_id,
        )
        if (
            stage_pairs == FUNNEL_STAGES[0]
            and direction_rounds >= direction_round_limit
        ):
            raise SchemaError(
                f"direction {policy['direction']!r} has occupied the latest "
                f"{direction_rounds} stage-1 experiment rounds; the next round must "
                "use a different research direction"
            )
        if (
            stage_pairs == FUNNEL_STAGES[0]
            and direction_cooldown_remaining > 0
        ):
            raise SchemaError(
                f"direction {policy['direction']!r} exhausted its "
                f"{direction_round_limit}-round block and remains on cooldown for "
                f"{direction_cooldown_remaining} more stage-1 round(s) in other "
                "directions"
            )
        max_failures = int(policy["portfolio"]["max_consecutive_failures"])
        override = str(policy["portfolio"]["pivot_override"]).strip()
        if stage_pairs == FUNNEL_STAGES[0] and failures >= max_failures:
            raise SchemaError(
                f"subsystem {policy['subsystem']!r} has {failures} consecutive failures "
                f"(limit {max_failures}); pivot to another subsystem"
            )
        if stage_pairs == FUNNEL_STAGES[0]:
            unresolved = []
            invalidated_experiments = {
                update.experiment_id
                for update in self.evidence_updates.load()
                if update.result == "invalid"
            }
            for candidate in self.gated_experiments.load():
                if (
                    candidate.experiment_id == experiment.experiment_id
                    or candidate.status not in {"approved", "running"}
                    or not candidate.search_policy
                    or candidate.experiment_id in invalidated_experiments
                ):
                    continue
                candidate_policy = validate_search_policy(
                    candidate.search_policy, candidate
                )
                if candidate_policy["qualification"]:
                    # Standalone provenance qualification is neither an ordinary
                    # stage-one round nor unresolved work in that efficacy funnel.
                    continue
                if (
                    str(candidate_policy["decision_frame"]) != expected_frame
                    or str(candidate_policy["subsystem"]) != str(policy["subsystem"])
                ):
                    continue
                verdict = str(evaluate_stage(candidate, self.runs.load())["verdict"])
                if verdict in {"pending", "continue"}:
                    unresolved.append(candidate.experiment_id)
            if unresolved:
                raise SchemaError(
                    f"subsystem {policy['subsystem']!r} already has unresolved GPU work "
                    f"in this frame: {sorted(unresolved)}; finish it or diversify"
                )

        noise = experiment.analysis_plan.get("noise_model")
        if isinstance(noise, Mapping):
            sigma = (
                (frame.get("baseline") or {}).get("effective_sigma")
                if frame is not None
                else noise.get("effective_sigma")
            )
            minimum_effect = noise.get("minimum_effect")
            if (
                not isinstance(sigma, bool)
                and isinstance(sigma, (int, float))
                and not isinstance(minimum_effect, bool)
                and isinstance(minimum_effect, (int, float))
                and minimum_effect < 2.0 * sigma
            ):
                raise SchemaError(
                    f"{experiment.experiment_id} minimum_effect={minimum_effect} is below "
                    f"the 2x decision floor {2.0 * sigma:.6g}"
                )
            promote_at = float(
                policy["stopping"]["promote_if_mean_endpoint_delta_lte"]
            )
            stop_at = float(
                policy["stopping"]["stop_if_mean_endpoint_delta_gte"]
            )
            if (
                not isinstance(minimum_effect, bool)
                and isinstance(minimum_effect, (int, float))
                and (
                    promote_at > -float(minimum_effect)
                    or stop_at > float(minimum_effect)
                )
            ):
                raise SchemaError(
                    f"{experiment.experiment_id} stopping thresholds are looser "
                    f"than its minimum_effect={minimum_effect}"
                )

        return {
            "idea_id": idea.idea_id,
            "decision_frame": expected_frame,
            "direction": policy["direction"],
            "subsystem": policy["subsystem"],
            "stage_pairs": stage_pairs,
            "parent_experiment_id": parent_id,
            "parent_verdict": parent_report["verdict"] if parent_report else "",
            "consecutive_subsystem_failures": failures,
            "consecutive_direction_rounds": direction_rounds,
            "direction_round_limit": direction_round_limit,
            "direction_cooldown_rounds": direction_cooldown_rounds,
            "direction_cooldown_remaining": direction_cooldown_remaining,
            "idea_funnel_stage_count": (
                0
                if is_qualification
                else len(same_idea_funnel_experiments) + 1
            ),
            "idea_funnel_stage_limit": len(FUNNEL_STAGES),
            "pivot_override": override,
            "predictions": policy["predictions"],
            **(
                {
                    "qualification": True,
                    "qualification_anchor_count": len(verified_anchors),
                    "qualification_anchors": list(verified_anchors),
                    "qualification_execution_authority": dict(
                        verified_authority
                    ),
                    "adoption_authorized": False,
                    "releases_ordinary_child": False,
                }
                if is_qualification
                else {}
            ),
        }

    def check_gate(
        self, experiment_id: str, scope_id: str | None = None
    ) -> Mapping[str, Any]:
        """Apply setup, knowledge, power, and executable gates without mutation."""
        scope_id = self.resolve_scope_id(scope_id)
        self.validate(check_generated_state=False)
        hourly_report = self.require_current_hourly_report()
        setup = self._require_current_setup(scope_id)
        proposals = self.experiment_proposals.by_id()
        if experiment_id not in proposals:
            raise SchemaError(f"proposal {experiment_id!r} is not registered")
        experiment = proposals[experiment_id]
        hypotheses = self.hypotheses.by_id()
        hypothesis = hypotheses[experiment.hypothesis_id]
        frame_key, frame = self._frame_key(setup, scope_id)
        if not self._matches_scope(experiment, frame_key):
            raise SchemaError(
                f"{experiment_id} must freeze the requested decision frame's "
                "scope_key in data_policy"
            )

        self._require_literature_assessments(experiment_id, hypothesis)
        self._validate_analysis_gate(experiment)
        search_report = self._validate_search_gate(
            experiment,
            scope_id=scope_id,
            frame=frame,
        )
        self._validate_executable_gate(
            experiment,
            hypothesis,
            self.observables.by_id(),
            self.interventions.by_id(),
            self.contexts.by_id(),
            self.outcomes.by_id(),
        )
        return {
            "experiment_id": experiment_id,
            "ready_to_freeze": True,
            "stage": experiment.stage,
            "challenge_id": self.challenge_for_scope(scope_id)["challenge_id"],
            "challenge_selection_fingerprint": self.challenge_events()[-1].fingerprint,
            "setup_fingerprint": setup.fingerprint,
            "scope_key": dict(frame_key),
            "hourly_report": hourly_report,
            "search_policy": search_report,
            "estimated_currency_cost": experiment.budget["estimated_currency_cost"],
            "hard_cap_currency_cost": experiment.budget["hard_cap_currency_cost"],
            "next_step": (
                "Copy the unchanged definition to gated/experiments.jsonl with "
                "status=approved and frozen_at set, then remove the proposal record."
            ),
        }

    def authorize_run(
        self,
        experiment_id: str,
        arm_id: str,
        seed: int,
        max_steps: int,
        scope_id: str | None = None,
        *,
        diagnostic_only: bool = False,
    ) -> Mapping[str, Any]:
        """Fail closed unless one exact run belongs to a current, unresolved gate.

        `scope_id` selects a registered secondary decision frame (e.g. the 5-minute
        wall-clock scope). Each frame is judged against its OWN baseline and floor;
        a comparison never crosses frames.
        """
        scope_id = self.resolve_scope_id(scope_id)
        self.validate(check_generated_state=False)
        hourly_report = self.require_current_hourly_report()
        setup = self._require_current_setup(scope_id)
        gated = self.gated_experiments.by_id()
        if experiment_id not in gated:
            raise SchemaError(f"experiment {experiment_id!r} is not gated")
        experiment = gated[experiment_id]
        if experiment.status not in {"approved", "running"}:
            raise SchemaError(
                f"experiment {experiment_id!r} has non-runnable status {experiment.status!r}"
            )
        frame_key, frame = self._frame_key(setup, scope_id)
        if scope_id:
            # H1: a frame run must still bind the experiment to a world. Without this
            # any stale/unscoped gate becomes runnable simply by passing --scope.
            if not self._matches_scope(experiment, frame_key):
                raise SchemaError(
                    f"experiment {experiment_id!r} is not frozen against frame "
                    f"{scope_id!r}; freeze it with that frame's scope_key before running it"
                )
        else:
            if not self._matches_scope(experiment, setup.scope_key):
                raise SchemaError(
                    f"experiment {experiment_id!r} is unscoped or stale for the current setup"
                )
        hypothesis = self.hypotheses.by_id()[experiment.hypothesis_id]
        self._require_literature_assessments(experiment_id, hypothesis)
        self._validate_analysis_gate(experiment)
        search_report = self._validate_search_gate(
            experiment,
            scope_id=scope_id,
            frame=frame,
        )
        diagnostic_pilot = experiment.stage == "pilot" and not experiment.search_policy
        if diagnostic_only and not diagnostic_pilot:
            raise SchemaError(
                f"{experiment_id} is an effect experiment; diagnostic-only "
                "authority is reserved for implementation-only pilots"
            )
        if diagnostic_pilot and not diagnostic_only:
            raise SchemaError(
                f"{experiment_id} is a diagnostic-only pilot; ordinary endpoint "
                "authorization is forbidden"
            )
        if arm_id not in {str(arm["arm_id"]) for arm in experiment.arms}:
            raise SchemaError(f"arm {arm_id!r} is not frozen by {experiment_id}")
        if seed not in experiment.seeds:
            raise SchemaError(f"seed {seed} is not frozen by {experiment_id}")
        if max_steps != frame_key["max_steps"]:
            raise SchemaError(
                f"MAX_STEPS={max_steps} differs from the frame's "
                f"MAX_STEPS={frame_key['max_steps']}"
            )
        concluded = {
            update.experiment_id
            for update in self.evidence_updates.load()
            if update.result != "invalid"
        }
        if experiment_id in concluded:
            raise SchemaError(f"experiment {experiment_id!r} already has a conclusive update")

        existing = [
            run
            for run in self.runs.load()
            if run.experiment_id == experiment_id
            and run.arm_id == arm_id
            and run.seed == seed
            and (
                diagnostic_pilot
                or not any(
                    str(tag).startswith("diagnostic_offbudget")
                    for tag in run.tags
                )
            )
        ]
        if existing:
            raise SchemaError(
                f"{experiment_id}/{arm_id}/seed{seed} already has a RunRecord; "
                "immutable attempts may not be silently repeated"
            )

        if diagnostic_pilot:
            stage_evaluation = {
                "verdict": "diagnostic_open",
                "reason": (
                    "implementation-only pilot; no effect-funnel stage or "
                    "endpoint scoring authority"
                ),
                "completed_pairs": 0,
            }
            allowed = 1
        else:
            stage_evaluation = evaluate_stage(experiment, self.runs.load())
            if stage_evaluation["verdict"] in {
                "adoption_candidate",
                "invalid",
                "promote",
                "qualification_fail",
                "qualification_pass",
                "stop_futility",
                "stop_no_signal",
            }:
                raise SchemaError(
                    f"{experiment_id} is closed by search policy: "
                    f"{stage_evaluation['verdict']} ({stage_evaluation['reason']})"
                )
            allowed = authorizable_seed_count(experiment, stage_evaluation)
        seed_index = list(experiment.seeds).index(seed)
        if seed_index >= allowed:
            raise SchemaError(
                f"seed {seed} is beyond the current staged tranche; only the first "
                f"{allowed} seeds are authorized until the preceding checkpoint clears"
            )
        return {
            "authorized": True,
            "experiment_id": experiment_id,
            "arm_id": arm_id,
            "seed": seed,
            "max_steps": max_steps,
            "scope_id": scope_id,
            "challenge_id": self.challenge_for_scope(scope_id)["challenge_id"],
            "challenge_selection_fingerprint": self.challenge_events()[-1].fingerprint,
            "hourly_report": hourly_report,
            "search_policy": search_report,
            "stage_evaluation": stage_evaluation,
            "diagnostic_only_authority": diagnostic_pilot,
            "endpoint_scoring_authorized": not diagnostic_pilot,
            "experiment_fingerprint": experiment.fingerprint,
            "setup_fingerprint": setup.fingerprint,
        }

    def render_state(self, *, data: Mapping[str, Mapping[str, Any]] | None = None) -> str:
        if data is None:
            self._manifest()
            data = self._load_all()
        setup = self.setup_reconciliation()
        selected_challenge, challenge_event = self.selected_challenge()
        selected_scope_id = str(selected_challenge["scope_id"])
        selected_frame = (
            setup.scope_for(selected_scope_id) if selected_scope_id else None
        )
        current_scope = (
            selected_frame["scope_key"] if selected_frame else setup.scope_key
        )
        selected_status = (
            str(selected_frame["status"]) if selected_frame else setup.status
        )
        all_beliefs = data["beliefs"]
        terminal_beliefs = self._current_beliefs(all_beliefs)
        scoped_beliefs = [
            belief
            for belief in terminal_beliefs
            if self._matches_scope(belief, current_scope)
        ]
        stale_beliefs = [
            belief
            for belief in terminal_beliefs
            if not self._matches_scope(belief, current_scope)
        ]
        all_evidence = {
            **data["literature_evidence"],
            **data["run_evidence"],
        }
        terminal_evidence = self._terminal_evidence(all_evidence)
        verified_beliefs: list[BeliefRecord] = []
        provisional_beliefs: list[BeliefRecord] = []
        for belief in scoped_beliefs:
            cited = terminal_evidence.terminal_records(belief.evidence_ids)
            cited_run_ids = {
                run_id
                for evidence in cited
                for run_id in evidence.run_ids
                if run_id in data["runs"] and data["runs"][run_id].status == "complete"
            }
            seeds = {data["runs"][run_id].seed for run_id in cited_run_ids}
            literature_only = bool(cited) and all(
                evidence.source_type == "literature" for evidence in cited
            )
            if belief.evidence_ids and (literature_only or len(seeds) >= 3):
                verified_beliefs.append(belief)
            else:
                provisional_beliefs.append(belief)
        conclusively_tested_hypothesis_ids = self._conclusively_tested_hypothesis_ids(
            all_evidence
        )
        current_experiment_hypothesis_ids = {
            experiment.hypothesis_id
            for collection in (
                data["experiment_proposals"].values(),
                data["gated_experiments"].values(),
            )
            for experiment in collection
            if self._matches_scope(experiment, current_scope)
        }
        hypotheses = [
            hypothesis
            for hypothesis in data["hypotheses"].values()
            if hypothesis.hypothesis_id not in conclusively_tested_hypothesis_ids
            and hypothesis.hypothesis_id in current_experiment_hypothesis_ids
        ]
        gaps = list(data["capability_gaps"].values())
        gated = list(data["gated_experiments"].values())
        evidence_updates = list(data["evidence_updates"].values())
        tools = {
            "observables": list(data["observables"].values()),
            "interventions": list(data["interventions"].values()),
            "contexts": list(data["contexts"].values()),
            "outcomes": list(data["outcomes"].values()),
        }
        lines = [
            "<!-- GENERATED FILE: edit registries, then run python -m vibeautoresearch render-state -->",
            "# Current Research State",
            "",
            "This view is generated from the structured registries. It contains no raw results.",
            "",
            "## Inventory",
            "",
        ]
        for name in self.PATHS:
            lines.append(f"- {name.replace('_', ' ')}: {len(data[name])}")
        lines.append(
            f"- current evidence: {len(terminal_evidence.terminals)} terminal / "
            f"{len(all_evidence)} append-only records"
        )
        lines.append("- setup reconciliation: 1")
        lines.extend(
            [
                "",
                "## Setup reconciliation",
                "",
                f"- Active challenge: **#{selected_challenge['order']} "
                f"{selected_challenge['challenge_id']}** "
                f"(selection generation {challenge_event.generation}, "
                f"action `{challenge_event.action}`)",
                f"- Active challenge status: **{selected_status}**",
                f"- Fingerprint: `{setup.fingerprint}`",
                f"- Scope key: `{canonical_json(dict(current_scope))}`",
                f"- Reference-code comparison: "
                f"**{setup.reference_code['comparison']}** "
                f"(+{setup.reference_code['diff']['added']} "
                f"/-{setup.reference_code['diff']['removed']} lines across "
                f"{setup.reference_code['diff']['hunks']} hunks)",
            ]
        )
        if setup.scopes:
            lines.extend(["", "### Secondary decision frames", ""])
            for frame in sorted(
                setup.scopes, key=lambda item: str(item.get("scope_id"))
            ):
                lines.append(
                    f"- `{frame['scope_id']}` [{frame['role']}/{frame['status']}]: "
                    f"scope `{canonical_json(dict(frame['scope_key']))}`; "
                    f"run env `{canonical_json(dict(frame.get('run_env', {})))}`"
                )
        lines.extend(["", "## Current verified beliefs", ""])
        if verified_beliefs:
            superseded_count = len(all_beliefs) - len(terminal_beliefs)
            lines.append(
                f"{len(verified_beliefs)} scope-matched, evidence-backed belief records; "
                f"{superseded_count} superseded "
                "records remain in append-only history."
            )
            lines.append("")
            for belief in sorted(verified_beliefs, key=lambda item: item.belief_id):
                lines.append(f"- `{belief.belief_id}` [{belief.status}]: {belief.statement}")
        else:
            lines.append("- None. No terminal belief is both current-scope and sufficiently evidenced.")
        lines.extend(["", "## Demoted beliefs", ""])
        lines.append(
            f"- {len(stale_beliefs)} terminal beliefs are stale/unscoped for the current "
            "active-challenge scope; they remain in append-only history."
        )
        lines.append(
            f"- {len(provisional_beliefs)} current-scope beliefs are provisional because "
            "they lack structured evidence or at least three completed seeds."
        )
        if stale_beliefs:
            lines.append("")
            for belief in sorted(stale_beliefs, key=lambda item: item.belief_id):
                label = "unscoped" if self._scope_key(belief) is None else "scope_mismatch"
                lines.append(f"- `{belief.belief_id}` [{label}]")
        if provisional_beliefs:
            lines.append("")
            for belief in sorted(provisional_beliefs, key=lambda item: item.belief_id):
                lines.append(f"- `{belief.belief_id}` [unverified]")
        lines.extend(["", "## Provenance gaps", ""])
        beliefs_without_evidence = sorted(
            belief.belief_id
            for belief in terminal_beliefs
            if belief.status != "speculative" and not belief.evidence_ids
        )
        literature_claim_ids = {
            claim_id
            for evidence in terminal_evidence.terminals.values()
            if evidence.source_type == "literature"
            for claim_id in evidence.claim_ids
        }
        claims_without_literature_evidence = set(data["claims"]) - literature_claim_ids
        if beliefs_without_evidence:
            lines.append("Terminal non-speculative beliefs without evidence:")
            lines.append("")
            for belief_id in beliefs_without_evidence:
                lines.append(f"- `{belief_id}`")
        else:
            lines.append("- Every current non-speculative belief cites structured evidence.")
        if claims_without_literature_evidence:
            lines.append(
                f"- Literature claims without a current terminal "
                f"literature-evidence assessment: "
                f"{len(claims_without_literature_evidence)}"
            )
        else:
            lines.append("- Every literature claim has a literature-evidence assessment.")
        lines.extend(["", "## Hypotheses and gates", ""])
        if hypotheses:
            lines.append(
                "Hypotheses without conclusive supporting, opposing, or mixed evidence:"
            )
            lines.append("")
            for hypothesis in sorted(hypotheses, key=lambda item: item.hypothesis_id):
                lines.append(
                    f"- `{hypothesis.hypothesis_id}` [{hypothesis.status}]: {hypothesis.prediction}"
                )
        else:
            lines.append("- No unresolved hypothesis is attached to a current-scope experiment.")
        lines.extend(["", "## Toolkit", ""])
        for kind, records in tools.items():
            available = sum(item.status in {"unit_tested", "cost_profiled", "available"} for item in records)
            lines.append(f"- {kind}: {available} executable / {len(records)} total")
        lines.extend(["", "## Capability gaps", ""])
        open_gaps = [gap for gap in gaps if gap.status in {"open", "in_progress"}]
        if open_gaps:
            for gap in sorted(open_gaps, key=lambda item: item.gap_id):
                lines.append(f"- `{gap.gap_id}` [{gap.priority}]: {gap.question}")
        else:
            lines.append("- None registered.")
        lines.extend(["", "## Approved or running experiments", ""])
        concluded_experiment_ids = {
            update.experiment_id
            for update in evidence_updates
            if update.result != "invalid"
        }
        active = [
            item
            for item in gated
            if self._matches_scope(item, current_scope)
            and (
                item.status == "running"
                or (
                    item.status == "approved"
                    and item.experiment_id not in concluded_experiment_ids
                )
            )
        ]
        if active:
            lines.append(
                "Approved gates with no conclusive evidence update yet "
                "(invalid attempts remain unresolved):"
            )
            lines.append("")
            for experiment in sorted(active, key=lambda item: item.experiment_id):
                lines.append(
                    f"- `{experiment.experiment_id}` [{experiment.stage}/{experiment.status}] "
                    f"budget cap={experiment.budget['hard_cap_currency_cost']}"
                )
        else:
            lines.append("- None. No training is authorized by the current state.")
        lines.extend(
            [
                "",
                "## Next action",
                "",
                "- Assess the literature claims needed by the next headroom question; unassessed claims cannot pass `check-gate`.",
                "- Create an active-challenge, scope-keyed proposal with a concurrent "
                "control, the required staged seeds, and that challenge's effective "
                "noise floor.",
                "- Run `python -m vibeautoresearch audit` and resolve provenance gaps before treating any demoted belief as current.",
                "",
            ]
        )
        return "\n".join(lines)

    def render_literature_snapshot(
        self,
        *,
        data: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> str:
        """Render the machine-owned scope/count block inside the editorial synthesis."""
        if data is None:
            self._manifest()
            data = self._load_all()
        setup = self.setup_reconciliation()
        selected_challenge, challenge_event = self.selected_challenge()
        selected_scope_id = str(selected_challenge["scope_id"])
        selected_frame = (
            setup.scope_for(selected_scope_id) if selected_scope_id else None
        )
        selected_scope = (
            selected_frame["scope_key"] if selected_frame else setup.scope_key
        )
        selected_baseline = (
            selected_frame["baseline"] if selected_frame else setup.baseline
        )
        selected_status = (
            str(selected_frame["status"]) if selected_frame else setup.status
        )
        all_evidence = {
            **data["literature_evidence"],
            **data["run_evidence"],
        }
        terminal_evidence = self._terminal_evidence(all_evidence)
        terminal_literature_evidence = {
            evidence_id: record
            for evidence_id, record in terminal_evidence.terminals.items()
            if record.source_type == "literature"
        }
        assessed_claim_ids = {
            claim_id
            for evidence in terminal_literature_evidence.values()
            for claim_id in evidence.claim_ids
        }
        terminal_beliefs = self._current_beliefs(data["beliefs"])
        planning_inputs_fingerprint = fingerprint(
            {
                "setup": setup.fingerprint,
                "challenge_selection": challenge_event.fingerprint,
                "papers": sorted(
                    (record_id, record.to_dict())
                    for record_id, record in data["papers"].items()
                ),
                "claims": sorted(
                    (record_id, record.to_dict())
                    for record_id, record in data["claims"].items()
                ),
                "literature_evidence": sorted(
                    (record_id, record.to_dict())
                    for record_id, record in terminal_literature_evidence.items()
                ),
                "terminal_beliefs": sorted(
                    (record.belief_id, record.to_dict())
                    for record in terminal_beliefs
                ),
                "mechanisms": sorted(
                    (record_id, record.to_dict())
                    for record_id, record in data["mechanisms"].items()
                ),
                "capability_gaps": sorted(
                    (record_id, record.to_dict())
                    for record_id, record in data["capability_gaps"].items()
                ),
            }
        )
        setup_drift = setup.verify_frozen_files(self.root.parent)
        setup_integrity = (
            "**verified**"
            if not setup_drift
            else "**DRIFTED — experiment execution is blocked; run `check-setup`**"
        )
        return "\n".join(
            [
                self.LITERATURE_SNAPSHOT_START,
                "<!-- GENERATED: run python -m vibeautoresearch render-literature; do not edit this block -->",
                "## Current registry and scope snapshot",
                "",
                f"- Planning-input fingerprint: `{planning_inputs_fingerprint}`",
                f"- Papers: **{len(data['papers'])}**",
                f"- Atomic claims: **{len(data['claims'])}**",
                f"- Current literature-evidence records: "
                f"**{len(terminal_literature_evidence)}** terminal / "
                f"**{len(data['literature_evidence'])}** append-only "
                f"(covering **{len(assessed_claim_ids)} / {len(data['claims'])}** claims)",
                f"- Current project mechanisms: **{len(data['mechanisms'])}** "
                "(these are not the historical literature-theme IDs)",
                f"- Active challenge: **#{selected_challenge['order']} "
                f"{selected_challenge['challenge_id']}**, status "
                f"**{selected_status}**, selection generation "
                f"**{challenge_event.generation}**",
                f"- Setup record: v{setup.version}, "
                f"fingerprint `{setup.fingerprint}`",
                f"- Setup integrity now: {setup_integrity}",
                f"- Scope key: `{canonical_json(dict(selected_scope))}`",
                f"- Frozen baseline: "
                f"`{selected_baseline['metric']}="
                f"{float(selected_baseline['observed']):.6f}`; effective σ "
                f"`{float(selected_baseline['effective_sigma']):.6f}`",
                "",
                "The registries, reconciliation record, and generated "
                "`RESEARCH_STATE.md` are authoritative. This synthesis is an "
                "editorial planning view and cannot authorize an experiment.",
                self.LITERATURE_SNAPSHOT_END,
            ]
        )

    def render_literature_synthesis(
        self,
        *,
        source: str | None = None,
        data: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> str:
        """Replace only the generated snapshot, preserving editorial strategy."""
        if source is None:
            if not self.literature_synthesis_path.exists():
                raise SchemaError(
                    f"missing literature synthesis: {self.literature_synthesis_path}"
                )
            source = self.literature_synthesis_path.read_text(encoding="utf-8")
        if (
            source.count(self.LITERATURE_SNAPSHOT_START) != 1
            or source.count(self.LITERATURE_SNAPSHOT_END) != 1
        ):
            raise SchemaError(
                "literature synthesis must contain one ordered generated snapshot block"
            )
        start = source.find(self.LITERATURE_SNAPSHOT_START)
        end = source.find(self.LITERATURE_SNAPSHOT_END)
        if end < start:
            raise SchemaError(
                "literature synthesis must contain one ordered generated snapshot block"
            )
        end += len(self.LITERATURE_SNAPSHOT_END)
        return (
            source[:start]
            + self.render_literature_snapshot(data=data)
            + source[end:]
        )

    def write_state(self) -> Path:
        self.validate(check_generated_state=False)
        atomic_write_text(self.state_path, self.render_state())
        return self.state_path

    def write_literature_synthesis(self) -> Path:
        """Refresh synthesis metadata even when setup drift blocks experiments."""
        self._manifest()
        data = self._load_all()
        rendered = self.render_literature_synthesis(data=data)
        atomic_write_text(self.literature_synthesis_path, rendered)
        return self.literature_synthesis_path

    def write_snapshot(self, label: str) -> Path:
        """Save the generated state without changing any source registry."""
        safe_label = local_id(label, "snapshot.label")
        self.validate(check_generated_state=False)
        path = self.root / "refinement" / "snapshots" / f"{safe_label}_RESEARCH_STATE.md"
        if path.exists():
            raise SchemaError(f"snapshot already exists: {path}")
        atomic_write_text(path, self.render_state())
        return path

    def audit(self) -> AuditReport:
        return RegistryAuditor(self).audit()


__all__ = ["AuditReport", "JsonlRegistry", "ResearchRegistry", "ValidationReport"]
