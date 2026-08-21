"""Deterministic research-ledger audits.

Validation rejects structurally unsafe state. Auditing reports refinement work
that can remain present without making the registry invalid. The issue families
live here so ``ResearchRegistry`` can orchestrate storage and gates without also
owning a monolithic report generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from .core import SchemaError
from .evidence import TerminalEvidenceIndex
from .jsonl import AuditReport

if TYPE_CHECKING:
    from .registry import ResearchRegistry


RegistryData = dict[str, dict[str, Any]]
AuditIssue = dict[str, Any]


@dataclass
class AuditContext:
    """Loaded state shared by one deterministic audit pass."""

    data: RegistryData
    active_scope: Mapping[str, Any]
    evidence: dict[str, Any]
    terminal_evidence: TerminalEvidenceIndex
    issues: list[AuditIssue]


class RegistryAuditor:
    """Collect non-fatal ledger issues by refinement domain."""

    SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}

    def __init__(self, registry: ResearchRegistry):
        self.registry = registry

    def audit(self) -> AuditReport:
        self.registry.validate(check_generated_state=False)
        context = self._load_context()
        self._audit_campaign(context)
        self._audit_beliefs(context)
        self._audit_literature_and_hypotheses(context)
        self._audit_runs(context)
        self._audit_experiments(context)
        self._audit_mechanisms_and_observations(context)
        self._audit_generated_views(context)
        return AuditReport(
            issues=tuple(
                sorted(
                    context.issues,
                    key=lambda item: (
                        self.SEVERITY_RANK[str(item["severity"])],
                        item["code"],
                        item["record_id"],
                    ),
                )
            )
        )

    def _load_context(self) -> AuditContext:
        registry = self.registry
        setup = registry.setup_reconciliation()
        selected_challenge, _ = registry.selected_challenge()
        selected_scope_id = str(selected_challenge["scope_id"])
        active_scope = (
            setup.scope_for(selected_scope_id)["scope_key"]
            if selected_scope_id
            else setup.scope_key
        )
        data = registry._load_all()
        evidence = {
            **data["literature_evidence"],
            **data["run_evidence"],
        }
        return AuditContext(
            data=data,
            active_scope=active_scope,
            evidence=evidence,
            terminal_evidence=registry._terminal_evidence(evidence),
            issues=[],
        )

    def _audit_campaign(self, context: AuditContext) -> None:
        selected_ideas = [
            idea
            for idea in context.data["idea_archive"].values()
            if idea.status in {"selected", "tested"}
        ]
        if not selected_ideas:
            context.issues.append(
                {
                    "severity": "warning",
                    "code": "no_selected_ideas_for_active_campaign",
                    "record_id": self.registry.PATHS["idea_archive"],
                    "message": (
                        "The active campaign has no selected, literature-cleared "
                        "IdeaRecord; generate and critique a diverse batch before "
                        "attempting the first GPU experiment."
                    ),
                }
            )
        if self.registry.hourly_report_status().get("due") is True:
            context.issues.append(
                {
                    "severity": "warning",
                    "code": "hourly_research_paper_overdue",
                    "record_id": self.registry.PATHS["hourly_reports"],
                    "message": (
                        "Publish the active challenge's hourly summary paper before "
                        "authorizing more GPU work."
                    ),
                }
            )

    def _audit_beliefs(self, context: AuditContext) -> None:
        registry = self.registry
        current_beliefs = registry._current_beliefs(
            context.data["beliefs"]
        )
        stale_belief_ids = sorted(
            belief.belief_id
            for belief in current_beliefs
            if not registry._matches_scope(
                belief, context.active_scope
            )
        )
        if stale_belief_ids:
            context.issues.append(
                {
                    "severity": "warning",
                    "code": "terminal_beliefs_outside_current_scope",
                    "record_id": registry.PATHS["beliefs"],
                    "message": (
                        f"{len(stale_belief_ids)} terminal beliefs are missing "
                        "the current active-challenge scope key or target a "
                        "different scope; render-state demotes them automatically."
                    ),
                }
            )
        for belief in current_beliefs:
            if belief.status != "speculative" and not belief.evidence_ids:
                context.issues.append(
                    {
                        "severity": "warning",
                        "code": "terminal_belief_without_evidence",
                        "record_id": belief.belief_id,
                        "message": (
                            "Terminal non-speculative belief has no structured "
                            "evidence; register evidence or revise it to an "
                            "explicitly speculative prior."
                        ),
                    }
                )

    def _audit_literature_and_hypotheses(
        self, context: AuditContext
    ) -> None:
        data = context.data
        literature_claim_ids = {
            claim_id
            for item in context.terminal_evidence.terminals.values()
            if item.source_type == "literature"
            for claim_id in item.claim_ids
        }
        missing_literature_assessments = (
            set(data["claims"]) - literature_claim_ids
        )
        if missing_literature_assessments:
            context.issues.append(
                {
                    "severity": "warning",
                    "code": "literature_claims_without_evidence",
                    "record_id": self.registry.PATHS[
                        "literature_evidence"
                    ],
                    "message": (
                        f"{len(missing_literature_assessments)} literature claims "
                        "have no current terminal literature-evidence assessment."
                    ),
                }
            )

        evidence_hypotheses = {
            hypothesis_id
            for item in context.terminal_evidence.terminals.values()
            for hypothesis_id in item.hypothesis_ids
        }
        conclusively_tested = (
            self.registry._conclusively_tested_hypothesis_ids(
                context.evidence
            )
        )
        for hypothesis in data["hypotheses"].values():
            if hypothesis.hypothesis_id not in evidence_hypotheses:
                context.issues.append(
                    {
                        "severity": "info",
                        "code": "hypothesis_without_evidence",
                        "record_id": hypothesis.hypothesis_id,
                        "message": "No evidence currently tests this hypothesis.",
                    }
                )
            elif hypothesis.hypothesis_id not in conclusively_tested:
                context.issues.append(
                    {
                        "severity": "info",
                        "code": "hypothesis_without_conclusive_evidence",
                        "record_id": hypothesis.hypothesis_id,
                        "message": (
                            "Existing evidence is inconclusive or does not test "
                            "the hypothesis; a conclusive result is still required."
                        ),
                    }
                )

    def _audit_runs(self, context: AuditContext) -> None:
        data = context.data
        run_evidence_ids = {
            run_id
            for item in context.terminal_evidence.terminals.values()
            if item.source_type in {"internal_run", "offline_analysis"}
            for run_id in item.run_ids
        }
        for run in data["runs"].values():
            if (
                run.status == "complete"
                and run.run_id not in run_evidence_ids
            ):
                context.issues.append(
                    {
                        "severity": "warning",
                        "code": "completed_run_without_evidence",
                        "record_id": run.run_id,
                        "message": (
                            "Completed run has not been converted into structured "
                            "evidence."
                        ),
                    }
                )
        legacy_challenge_run_ids = sorted(
            run.run_id
            for run in data["runs"].values()
            if run.status == "complete"
            and not any(
                str(tag).startswith("challenge_") for tag in run.tags
            )
            and not any(
                str(tag).startswith("diagnostic_offbudget")
                for tag in run.tags
            )
        )
        if legacy_challenge_run_ids:
            context.issues.append(
                {
                    # This is immutable, explicitly non-authorizable history, not
                    # actionable current debt. Keep it visible without making a
                    # zero-warning strict audit impossible forever.
                    "severity": "info",
                    "code": "legacy_runs_without_challenge_binding",
                    "record_id": self.registry.PATHS["runs"],
                    "message": (
                        f"{len(legacy_challenge_run_ids)} completed legacy runs "
                        "predate challenge IDs/selection fingerprints. They remain "
                        "historical facts but cannot satisfy the new search-policy "
                        "checkpoints."
                    ),
                }
            )

    def _audit_experiments(self, context: AuditContext) -> None:
        registry = self.registry
        data = context.data
        concluded_experiment_ids = {
            update.experiment_id
            for update in data["evidence_updates"].values()
            if update.result != "invalid"
        }
        unresolved_stale_gate_ids = sorted(
            experiment.experiment_id
            for experiment in data["gated_experiments"].values()
            if experiment.status in {"approved", "running"}
            and experiment.experiment_id not in concluded_experiment_ids
            and not registry._matches_scope(
                experiment, context.active_scope
            )
        )
        if unresolved_stale_gate_ids:
            context.issues.append(
                {
                    "severity": "warning",
                    "code": "runnable_gates_outside_current_scope",
                    "record_id": registry.PATHS["gated_experiments"],
                    "message": (
                        f"{len(unresolved_stale_gate_ids)} unresolved legacy "
                        "gates are not executable because they do not freeze the "
                        "current setup scope key: "
                        f"{unresolved_stale_gate_ids}"
                    ),
                }
            )
        policy_inert_gate_ids = sorted(
            experiment.experiment_id
            for experiment in data["gated_experiments"].values()
            if experiment.status in {"approved", "running"}
            and not experiment.search_policy
        )
        if policy_inert_gate_ids:
            context.issues.append(
                {
                    "severity": "warning",
                    "code": "policy_inert_legacy_gates",
                    "record_id": registry.PATHS["gated_experiments"],
                    "message": (
                        f"{len(policy_inert_gate_ids)} approved/running legacy "
                        "gates pass schema validation but cannot be authorized "
                        "because they predate search_policy v2: "
                        f"{policy_inert_gate_ids}"
                    ),
                }
            )
        refined_experiment_ids = {
            item.experiment_id
            for item in data["evidence_updates"].values()
        }
        for experiment in data["gated_experiments"].values():
            if (
                experiment.status == "complete"
                and experiment.experiment_id not in refined_experiment_ids
            ):
                context.issues.append(
                    {
                        "severity": "warning",
                        "code": "experiment_without_refinement",
                        "record_id": experiment.experiment_id,
                        "message": (
                            "Completed experiment has no evidence update."
                        ),
                    }
                )

    def _audit_mechanisms_and_observations(
        self, context: AuditContext
    ) -> None:
        data = context.data
        cited_observation_ids = {
            observation_id
            for mechanism in data["mechanisms"].values()
            for observation_id in mechanism.observation_ids
        }
        for mechanism in data["mechanisms"].values():
            if (
                mechanism.origin_type == "self_proposed"
                and not mechanism.observation_ids
                and not mechanism.campaign_batch_ids
            ):
                context.issues.append(
                    {
                        "severity": "warning",
                        "code": (
                            "self_proposed_mechanism_without_provenance"
                        ),
                        "record_id": mechanism.mechanism_id,
                        "message": (
                            "Self-proposed mechanism cites no observation "
                            "provenance; it cannot cite claims, so name the "
                            "observations that motivated it."
                        ),
                    }
                )
            quarantined_batches = [
                batch_id
                for batch_id in mechanism.campaign_batch_ids
                if data["campaign_batches"][batch_id].disposition == "quarantined"
            ]
            if quarantined_batches:
                context.issues.append(
                    {
                        "severity": "info",
                        "code": "mechanism_uses_quarantined_campaign_provenance",
                        "record_id": mechanism.mechanism_id,
                        "message": (
                            "Mechanism is grounded in campaign rows without gated "
                            "RunRecords and remains provisional: "
                            f"{quarantined_batches}"
                        ),
                    }
                )
        for observation in data["observations"].values():
            if observation.followup_required and observation.status in {
                "exploratory",
                "replicated",
            }:
                context.issues.append(
                    {
                        "severity": "info",
                        "code": "observation_followup_unresolved",
                        "record_id": observation.observation_id,
                        "message": (
                            "Observation is flagged followup_required but is still "
                            f"{observation.status!r}; resolve it to 'promoted' or "
                            "'dismissed'."
                        ),
                    }
                )
            elif (
                observation.status == "exploratory"
                and observation.observation_id not in cited_observation_ids
            ):
                context.issues.append(
                    {
                        "severity": "info",
                        "code": "observation_never_triaged",
                        "record_id": observation.observation_id,
                        "message": (
                            "Exploratory observation has never been replicated, "
                            "dismissed, or promoted into a mechanism."
                        ),
                    }
                )

    def _audit_generated_views(self, context: AuditContext) -> None:
        registry = self.registry
        if (
            registry.state_path.exists()
            and registry.state_path.read_text(encoding="utf-8")
            != registry.render_state(data=context.data)
        ):
            context.issues.append(
                {
                    "severity": "warning",
                    "code": "stale_research_state",
                    "record_id": str(registry.state_path),
                    "message": "Regenerate RESEARCH_STATE.md from registries.",
                }
            )
        if not registry.literature_synthesis_path.exists():
            return
        source = registry.literature_synthesis_path.read_text(
            encoding="utf-8"
        )
        try:
            literature_stale = (
                source
                != registry.render_literature_synthesis(
                    source=source, data=context.data
                )
            )
        except SchemaError:
            literature_stale = True
        if literature_stale:
            context.issues.append(
                {
                    "severity": "warning",
                    "code": "stale_literature_synthesis",
                    "record_id": str(
                        registry.literature_synthesis_path
                    ),
                    "message": (
                        "Refresh the generated literature snapshot and review "
                        "the editorial ranked queue."
                    ),
                }
            )


__all__ = ["AuditContext", "RegistryAuditor"]
