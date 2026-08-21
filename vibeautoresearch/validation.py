"""Cross-registry validation orchestration.

The persistence layer validates individual records as they are loaded. This
module validates the relationships between those records: setup bindings,
cross-references, experiment gates, run provenance, and generated views.
Keeping those invariant families separate makes the enforcement path reviewable
without weakening the fail-closed behavior exposed by ``ResearchRegistry``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from .campaign import campaign_slice_sha256
from .core import SchemaError
from .evidence import TerminalEvidenceIndex, resolve_terminal_evidence
from .idea_archive import duplicate_ideas
from .jsonl import ValidationReport
from .reporting import HourlyResearchReport, parse_timestamp
from .setup import SetupReconciliationRecord

if TYPE_CHECKING:
    from .registry import ResearchRegistry


RegistryData = dict[str, dict[str, Any]]


@dataclass
class ValidationContext:
    """Loaded state shared by one deterministic validation pass."""

    setup: SetupReconciliationRecord
    data: RegistryData
    evidence: dict[str, Any]
    terminal_evidence: TerminalEvidenceIndex | None
    warnings: list[str]


class RegistryValidator:
    """Validate one ``ResearchRegistry`` by invariant family."""

    def __init__(self, registry: ResearchRegistry):
        self.registry = registry

    def validate(self, *, check_generated_state: bool = True) -> ValidationReport:
        context = self._bootstrap()
        self._validate_setup_outcomes(context)
        self._validate_namespaces(context)
        self._validate_evidence_supersession(context)
        self._validate_campaign_ledger(context)
        self._validate_knowledge_graph(context)
        self._validate_mechanisms_and_hypotheses(context)
        self._validate_ideas(context)
        self._validate_hourly_reports(context)
        self._validate_tooling(context)
        self._validate_experiments(context)
        self._validate_runs(context)
        self._validate_refinement(context)
        if check_generated_state:
            self._check_generated_views(context)
        return ValidationReport(
            counts={
                **{
                    name: len(records)
                    for name, records in context.data.items()
                },
                "setup_reconciliation": 1,
            },
            warnings=tuple(context.warnings),
        )

    def _bootstrap(self) -> ValidationContext:
        registry = self.registry
        registry._manifest()
        setup = registry.setup_reconciliation()
        challenge_catalog, challenge_event = (
            registry._validate_challenge_configuration(setup)
        )
        selected_challenge = challenge_catalog.resolve(
            challenge_event.challenge_id
        )
        selected_scope_id = str(selected_challenge["scope_id"])
        selected_frame = (
            setup.scope_for(selected_scope_id) if selected_scope_id else None
        )
        selected_status = (
            str(selected_frame["status"]) if selected_frame else setup.status
        )
        setup_drift = setup.verify_frozen_files(registry.root.parent)
        has_executable_frame = setup.status == "passed" or any(
            str(frame.get("status")) == "passed" for frame in setup.scopes
        )
        if has_executable_frame and setup_drift:
            raise SchemaError(
                "setup reconciliation failed: " + "; ".join(setup_drift)
            )

        warnings: list[str] = []
        if challenge_event.action != "activated":
            warnings.append(
                f"challenge work is stopped at generation "
                f"{challenge_event.generation}; implicit gate checks and launches "
                "are blocked"
            )
        elif selected_status != "passed":
            passed_frames = sorted(
                str(frame.get("scope_id"))
                for frame in setup.scopes
                if str(frame.get("status")) == "passed"
            )
            if selected_scope_id:
                warnings.append(
                    f"active challenge {selected_challenge['challenge_id']} is "
                    f"{selected_status}; its execution is blocked"
                )
            else:
                warnings.append(
                    f"default setup reconciliation is {setup.status}; "
                    "default-frame execution is blocked"
                    + (
                        f"; passed secondary frames remain available: "
                        f"{passed_frames}"
                        if passed_frames
                        else ""
                    )
                )

        data = registry._load_all()
        return ValidationContext(
            setup=setup,
            data=data,
            evidence={
                **data["literature_evidence"],
                **data["run_evidence"],
            },
            terminal_evidence=None,
            warnings=warnings,
        )

    def _validate_setup_outcomes(self, context: ValidationContext) -> None:
        setup = context.setup
        outcomes = context.data["outcomes"]
        if setup.status == "passed":
            setup_outcome_id = str(setup.scope_key["outcome_id"])
            if setup_outcome_id not in outcomes:
                raise SchemaError(
                    "setup reconciliation references missing outcome "
                    f"{setup_outcome_id}"
                )
            if str(outcomes[setup_outcome_id].metric["name"]) != str(
                setup.baseline["metric"]
            ):
                raise SchemaError(
                    "setup baseline metric differs from the scoped outcome "
                    "definition"
                )
        for frame in setup.scopes:
            if str(frame.get("status")) != "passed":
                continue
            frame_outcome_id = str(frame["scope_key"]["outcome_id"])
            if frame_outcome_id not in outcomes:
                raise SchemaError(
                    f"decision frame {frame.get('scope_id')!r} references "
                    f"missing outcome {frame_outcome_id}"
                )
            if str(outcomes[frame_outcome_id].metric["name"]) != str(
                frame["baseline"]["metric"]
            ):
                raise SchemaError(
                    f"decision frame {frame.get('scope_id')!r} baseline metric "
                    "differs from its scoped outcome definition"
                )

    def _validate_namespaces(self, context: ValidationContext) -> None:
        data = context.data
        literature_evidence = data["literature_evidence"]
        run_evidence = data["run_evidence"]
        duplicate_evidence = set(literature_evidence) & set(run_evidence)
        if duplicate_evidence:
            raise SchemaError(
                "evidence IDs collide across external/internal registries: "
                f"{sorted(duplicate_evidence)}"
            )
        observation_evidence_collision = (
            set(data["observations"]) & set(context.evidence)
        )
        if observation_evidence_collision:
            raise SchemaError(
                "observation/evidence IDs share one namespace and must not "
                f"collide: {sorted(observation_evidence_collision)}"
            )
        duplicate_experiments = (
            set(data["experiment_proposals"])
            & set(data["gated_experiments"])
        )
        if duplicate_experiments:
            raise SchemaError(
                "an experiment ID cannot remain in proposals after it is copied "
                f"into gated: {sorted(duplicate_experiments)}"
            )

    def _validate_evidence_supersession(
        self, context: ValidationContext
    ) -> None:
        """Validate correction chains before any current-state consumer runs."""
        wrong_external = sorted(
            item.evidence_id
            for item in context.data["literature_evidence"].values()
            if item.source_type != "literature"
        )
        if wrong_external:
            raise SchemaError(
                "literature_evidence registry contains non-literature source "
                f"types: {wrong_external}"
            )
        wrong_internal = sorted(
            item.evidence_id
            for item in context.data["run_evidence"].values()
            if item.source_type == "literature"
        )
        if wrong_internal:
            raise SchemaError(
                "run_evidence registry contains literature source types: "
                f"{wrong_internal}"
            )
        context.terminal_evidence = resolve_terminal_evidence(context.evidence)

    def _validate_campaign_ledger(self, context: ValidationContext) -> None:
        """Require exact typed coverage for every governed chart-feed entry."""
        registry = self.registry
        policy = registry._manifest()["campaign_ledger"]
        source_path = str(policy["source_path"])
        coverage_start = int(policy["coverage_start_exp_num"])
        history_anchor = int(policy["history_anchor_exp_num"])
        history_digest = str(policy["history_anchor_sha256"])
        path = registry.root.parent / source_path
        if not path.is_file():
            raise SchemaError(f"missing campaign ledger source: {path}")

        rows_by_exp: dict[int, Mapping[str, Any]] = {}
        previous_exp_num = -1
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SchemaError(
                    f"invalid {source_path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise SchemaError(
                    f"invalid {source_path} line {line_number}: row must be an object"
                )
            exp_num = row.get("exp_num")
            if isinstance(exp_num, bool) or not isinstance(exp_num, int) or exp_num < 0:
                raise SchemaError(
                    f"invalid {source_path} line {line_number}: exp_num must be "
                    "a non-negative integer"
                )
            # The pre-control feed restarted numbering several times. Only the
            # governed suffix has a monotone identity contract.
            if exp_num < coverage_start:
                continue
            if exp_num <= previous_exp_num:
                raise SchemaError(
                    f"{source_path} exp_num values must be strictly increasing; "
                    f"line {line_number} has {exp_num} after {previous_exp_num}"
                )
            if exp_num in rows_by_exp:
                raise SchemaError(f"{source_path} has duplicate exp_num {exp_num}")
            previous_exp_num = exp_num
            rows_by_exp[exp_num] = dict(row)

        anchored_rows = [
            row
            for exp_num, row in rows_by_exp.items()
            if coverage_start <= exp_num <= history_anchor
        ]
        if history_anchor >= coverage_start and (
            not anchored_rows
            or int(anchored_rows[-1]["exp_num"]) != history_anchor
        ):
            raise SchemaError(
                "campaign ledger history anchor row is missing: "
                f"exp_num {history_anchor}"
            )
        actual_history_digest = campaign_slice_sha256(anchored_rows)
        if actual_history_digest != history_digest:
            raise SchemaError(
                "campaign ledger history anchor detects deleted or rewritten rows: "
                f"expected {history_digest}, found {actual_history_digest}"
            )

        covered_by: dict[int, str] = {}
        all_experiments = {
            **context.data["experiment_proposals"],
            **context.data["gated_experiments"],
        }
        for batch in context.data["campaign_batches"].values():
            if batch.source_path != source_path:
                raise SchemaError(
                    f"{batch.campaign_batch_id} source_path must match manifest "
                    f"campaign ledger {source_path!r}"
                )
            exp_nums = list(range(batch.first_exp_num, batch.last_exp_num + 1))
            missing_rows = [exp_num for exp_num in exp_nums if exp_num not in rows_by_exp]
            if missing_rows:
                raise SchemaError(
                    f"{batch.campaign_batch_id} covers missing campaign rows "
                    f"{missing_rows[:10]}"
                )
            overlap = [exp_num for exp_num in exp_nums if exp_num in covered_by]
            if overlap:
                raise SchemaError(
                    f"{batch.campaign_batch_id} overlaps campaign coverage from "
                    f"{covered_by[overlap[0]]} at exp_num {overlap[0]}"
                )
            rows = [rows_by_exp[exp_num] for exp_num in exp_nums]
            wrong_phases = sorted(
                {
                    str(row.get("phase", ""))
                    for row in rows
                    if str(row.get("phase", "")) != batch.phase
                }
            )
            if wrong_phases:
                raise SchemaError(
                    f"{batch.campaign_batch_id} phase {batch.phase!r} does not "
                    f"match source rows {wrong_phases}"
                )
            actual_digest = campaign_slice_sha256(rows)
            if batch.source_sha256 != actual_digest:
                raise SchemaError(
                    f"{batch.campaign_batch_id} source digest mismatch: expected "
                    f"{batch.source_sha256}, found {actual_digest}"
                )
            if batch.disposition == "registered":
                registry._require_refs(
                    batch.campaign_batch_id,
                    "experiments",
                    batch.experiment_ids,
                    all_experiments,
                )
                registry._require_refs(
                    batch.campaign_batch_id,
                    "runs",
                    batch.run_ids,
                    context.data["runs"],
                )
                registry._require_refs(
                    batch.campaign_batch_id,
                    "evidence",
                    batch.evidence_ids,
                    context.evidence,
                )
            for exp_num in exp_nums:
                covered_by[exp_num] = batch.campaign_batch_id

        governed = {
            exp_num for exp_num in rows_by_exp if exp_num >= coverage_start
        }
        missing_coverage = sorted(governed - set(covered_by))
        if missing_coverage:
            raise SchemaError(
                "campaign ledger rows lack a registered or quarantined batch: "
                f"{missing_coverage[:10]}"
            )

    def _validate_knowledge_graph(self, context: ValidationContext) -> None:
        registry = self.registry
        data = context.data
        papers = data["papers"]
        claims = data["claims"]
        runs = data["runs"]
        hypotheses = data["hypotheses"]
        mechanisms = data["mechanisms"]
        observations = data["observations"]
        beliefs = data["beliefs"]
        experiments = {
            **data["experiment_proposals"],
            **data["gated_experiments"],
        }

        for claim in claims.values():
            registry._require_refs(
                claim.claim_id, "papers", (claim.paper_id,), papers
            )
        for item in context.evidence.values():
            registry._require_refs(
                item.evidence_id, "papers", item.paper_ids, papers
            )
            registry._require_refs(
                item.evidence_id, "claims", item.claim_ids, claims
            )
            registry._require_refs(
                item.evidence_id, "runs", item.run_ids, runs
            )
            registry._require_refs(
                item.evidence_id,
                "campaign batches",
                item.campaign_batch_ids,
                data["campaign_batches"],
            )
            registry._require_refs(
                item.evidence_id,
                "hypotheses",
                item.hypothesis_ids,
                hypotheses,
            )
            if item.experiment_id and item.experiment_id not in experiments:
                raise SchemaError(
                    f"{item.evidence_id} references missing experiment "
                    f"{item.experiment_id}"
                )
        for observation in observations.values():
            registry._require_refs(
                observation.observation_id,
                "runs",
                observation.run_ids,
                runs,
            )

        mechanism_observation_ids = {
            observation_id
            for mechanism in mechanisms.values()
            for observation_id in mechanism.observation_ids
        }
        unbacked_promotions = sorted(
            observation.observation_id
            for observation in observations.values()
            if observation.status == "promoted"
            and observation.observation_id not in mechanism_observation_ids
        )
        if unbacked_promotions:
            raise SchemaError(
                "promoted observations must be cited by a mechanism's "
                f"observation_ids: {unbacked_promotions}"
            )

        valid_subjects = {**claims, **mechanisms, **hypotheses}
        for belief in beliefs.values():
            registry._require_refs(
                belief.belief_id,
                "evidence",
                belief.evidence_ids,
                context.evidence,
            )
            registry._require_refs(
                belief.belief_id,
                "belief subjects",
                belief.subject_ids,
                valid_subjects,
            )
            if (
                belief.supersedes_belief_id
                and belief.supersedes_belief_id not in beliefs
            ):
                raise SchemaError(
                    f"{belief.belief_id} supersedes missing belief "
                    f"{belief.supersedes_belief_id}"
                )
        registry._validate_belief_supersession(beliefs)

        current_without_evidence = sorted(
            belief.belief_id
            for belief in registry._current_beliefs(beliefs)
            if belief.status != "speculative" and not belief.evidence_ids
        )
        if current_without_evidence:
            context.warnings.append(
                f"{len(current_without_evidence)} terminal non-speculative "
                "beliefs have no evidence; run "
                "`python -m vibeautoresearch audit` for IDs"
            )

    def _validate_mechanisms_and_hypotheses(
        self, context: ValidationContext
    ) -> None:
        registry = self.registry
        data = context.data
        claims = data["claims"]
        observations = data["observations"]
        observables = data["observables"]
        interventions = data["interventions"]
        contexts = data["contexts"]
        outcomes = data["outcomes"]
        mechanisms = data["mechanisms"]

        for mechanism in mechanisms.values():
            registry._require_refs(
                mechanism.mechanism_id,
                "claims",
                mechanism.claim_ids,
                claims,
            )
            registry._require_refs(
                mechanism.mechanism_id,
                "observations",
                mechanism.observation_ids,
                observations,
            )
            registry._require_refs(
                mechanism.mechanism_id,
                "campaign batches",
                mechanism.campaign_batch_ids,
                data["campaign_batches"],
            )
            registry._require_refs(
                mechanism.mechanism_id,
                "competing mechanisms",
                mechanism.competing_mechanism_ids,
                mechanisms,
            )
            for prediction in mechanism.observable_predictions:
                observable_id = str(prediction["observable_id"])
                if observable_id not in observables:
                    raise SchemaError(
                        f"{mechanism.mechanism_id} references missing observable "
                        f"{observable_id}"
                    )
            for prediction in mechanism.intervention_predictions:
                intervention_id = str(prediction["intervention_id"])
                if intervention_id not in interventions:
                    raise SchemaError(
                        f"{mechanism.mechanism_id} references missing "
                        f"intervention {intervention_id}"
                    )

        for hypothesis in data["hypotheses"].values():
            registry._require_refs(
                hypothesis.hypothesis_id,
                "mechanisms",
                hypothesis.mechanism_ids,
                mechanisms,
            )
            registry._require_refs(
                hypothesis.hypothesis_id,
                "contexts",
                hypothesis.context_ids,
                contexts,
            )
            registry._require_refs(
                hypothesis.hypothesis_id,
                "observables",
                hypothesis.observable_ids,
                observables,
            )
            if hypothesis.intervention_id not in interventions:
                raise SchemaError(
                    f"{hypothesis.hypothesis_id} references missing intervention "
                    f"{hypothesis.intervention_id}"
                )
            if hypothesis.outcome_id not in outcomes:
                raise SchemaError(
                    f"{hypothesis.hypothesis_id} references missing outcome "
                    f"{hypothesis.outcome_id}"
                )
            for item in hypothesis.observable_predictions:
                registered = observables[str(item["observable_id"])]
                if int(item["version"]) != registered.version:
                    raise SchemaError(
                        f"{hypothesis.hypothesis_id} observable version does not "
                        "match registry"
                    )
            if (
                int(hypothesis.intervention["version"])
                != interventions[hypothesis.intervention_id].version
            ):
                raise SchemaError(
                    f"{hypothesis.hypothesis_id} intervention version does not "
                    "match registry"
                )
            if (
                int(hypothesis.outcome["version"])
                != outcomes[hypothesis.outcome_id].version
            ):
                raise SchemaError(
                    f"{hypothesis.hypothesis_id} outcome version does not match "
                    "registry"
                )

    def _validate_ideas(self, context: ValidationContext) -> None:
        registry = self.registry
        data = context.data
        ideas = data["idea_archive"]
        papers = data["papers"]
        literature_evidence = data["literature_evidence"]
        run_evidence = data["run_evidence"]
        hypotheses = data["hypotheses"]
        terminal_evidence = context.terminal_evidence
        if terminal_evidence is None:  # Defensive: validation order is invariant.
            raise SchemaError("terminal evidence was not resolved")

        for idea in ideas.values():
            registry._require_refs(
                idea.idea_id, "parent ideas", idea.parent_idea_ids, ideas
            )
            query_paper_ids = tuple(
                str(paper_id)
                for query_round in idea.novelty_check["query_rounds"]
                for paper_id in query_round["result_paper_ids"]
            )
            registry._require_refs(
                idea.idea_id,
                "novelty-query papers",
                query_paper_ids,
                papers,
            )
            registry._require_refs(
                idea.idea_id,
                "closest novelty papers",
                tuple(
                    str(item)
                    for item in idea.novelty_check["closest_paper_ids"]
                ),
                papers,
            )
            novelty_provider = str(idea.novelty_check["provider"])
            novelty_evidence = (
                {**literature_evidence, **run_evidence}
                if novelty_provider == "combined"
                else literature_evidence
            )
            registry._require_refs(
                idea.idea_id,
                (
                    "combined literature/internal evidence"
                    if novelty_provider == "combined"
                    else "literature evidence"
                ),
                tuple(
                    str(item)
                    for item in idea.novelty_check["evidence_ids"]
                ),
                novelty_evidence,
            )
            if novelty_provider == "combined":
                inherited_literature_ids = {
                    str(evidence_id)
                    for parent_id in idea.parent_idea_ids
                    if parent_id in ideas
                    for evidence_id in ideas[parent_id].novelty_check["evidence_ids"]
                    if terminal_evidence.terminal_record(
                        str(evidence_id)
                    ).source_type
                    == "literature"
                }
                direct_literature_ids = {
                    str(evidence_id)
                    for evidence_id in idea.novelty_check["evidence_ids"]
                    if terminal_evidence.terminal_record(
                        str(evidence_id)
                    ).source_type
                    == "literature"
                }
                if not direct_literature_ids and not inherited_literature_ids:
                    raise SchemaError(
                        f"{idea.idea_id} uses combined novelty evidence without "
                        "direct or parent-inherited structured literature evidence"
                    )
            if idea.hypothesis_id:
                registry._require_refs(
                    idea.idea_id,
                    "hypotheses",
                    (idea.hypothesis_id,),
                    hypotheses,
                )
        registry._validate_idea_parent_graph(ideas)
        duplicates = duplicate_ideas(ideas)
        if duplicates:
            rendered = ", ".join(
                f"{idea_id}~{other} ({similarity})"
                for idea_id, other, similarity in duplicates
            )
            raise SchemaError(
                "funded ideas exceed their own novelty discard threshold against "
                "an unrelated archived idea (fork-by-duplicate): "
                f"{rendered}"
            )

    def _validate_hourly_reports(self, context: ValidationContext) -> None:
        registry = self.registry
        challenge_events = {
            event.fingerprint: event for event in registry.challenge_events()
        }
        reports_by_selection: dict[str, list[HourlyResearchReport]] = {}
        for report in context.data["hourly_reports"].values():
            event = challenge_events.get(
                report.challenge_selection_fingerprint
            )
            if event is None:
                raise SchemaError(
                    f"{report.report_id} references an unknown "
                    "challenge-selection fingerprint"
                )
            if event.challenge_id != report.challenge_id:
                raise SchemaError(
                    f"{report.report_id} challenge differs from its selection event"
                )
            registry._validate_hourly_report_references(
                report, data=context.data
            )
            paper_path = (
                registry.root / "reports" / "hourly" / report.paper_filename
            )
            if not paper_path.exists():
                raise SchemaError(
                    f"{report.report_id} is missing its rendered hourly paper "
                    f"{paper_path}"
                )
            if (
                paper_path.read_text(encoding="utf-8")
                != report.render_markdown()
            ):
                raise SchemaError(
                    f"{report.report_id} rendered hourly paper has drifted from "
                    "its append-only record"
                )
            reports_by_selection.setdefault(
                report.challenge_selection_fingerprint, []
            ).append(report)

        for selection_fingerprint, reports in reports_by_selection.items():
            reports.sort(
                key=lambda report: (
                    parse_timestamp(
                        report.period_started_at,
                        "hourly_report.period_started_at",
                    ),
                    report.report_id,
                )
            )
            event = challenge_events[selection_fingerprint]
            previous_end = parse_timestamp(
                event.selected_at, "challenge_event.selected_at"
            )
            for report in reports:
                started = parse_timestamp(
                    report.period_started_at,
                    "hourly_report.period_started_at",
                )
                ended = parse_timestamp(
                    report.period_ended_at,
                    "hourly_report.period_ended_at",
                )
                if started < previous_end:
                    raise SchemaError(
                        f"{report.report_id} overlaps or predates its preceding "
                        "hourly reporting period"
                    )
                previous_end = ended

        if registry.hourly_report_status().get("due") is True:
            context.warnings.append(
                "active challenge hourly research paper is overdue; new gate "
                "checks and GPU authorizations are blocked until it is published"
            )

    def _validate_tooling(self, context: ValidationContext) -> None:
        registry = self.registry
        hypotheses = context.data["hypotheses"]
        for gap in context.data["capability_gaps"].values():
            registry._require_refs(
                gap.gap_id,
                "hypotheses",
                gap.blocked_hypothesis_ids,
                hypotheses,
            )
        for proposal in context.data["tool_proposals"].values():
            registry._require_refs(
                proposal.proposal_id,
                "hypotheses",
                proposal.required_by_hypotheses,
                hypotheses,
            )

    def _validate_experiments(self, context: ValidationContext) -> None:
        registry = self.registry
        data = context.data
        hypotheses = data["hypotheses"]
        ideas = data["idea_archive"]
        proposals = data["experiment_proposals"]
        gated = data["gated_experiments"]
        for collection_name, experiments in (
            ("proposal", proposals),
            ("gated", gated),
        ):
            for experiment in experiments.values():
                if experiment.hypothesis_id not in hypotheses:
                    raise SchemaError(
                        f"{experiment.experiment_id} references missing "
                        f"hypothesis {experiment.hypothesis_id}"
                    )
                hypothesis = hypotheses[experiment.hypothesis_id]
                if experiment.hypothesis_fingerprint != hypothesis.fingerprint:
                    raise SchemaError(
                        f"{experiment.experiment_id} was built against a "
                        "different hypothesis definition"
                    )
                if (
                    str(experiment.analysis_plan["outcome_id"])
                    != hypothesis.outcome_id
                ):
                    raise SchemaError(
                        f"{experiment.experiment_id} analysis outcome differs "
                        "from hypothesis"
                    )
                self._validate_experiment_idea(
                    experiment, hypothesis, ideas
                )
                if collection_name == "gated":
                    if experiment.status not in {
                        "approved",
                        "running",
                        "complete",
                        "failed",
                        "cancelled",
                    }:
                        raise SchemaError(
                            f"gated experiment {experiment.experiment_id} is "
                            "not frozen/approved"
                        )
                    registry._validate_executable_gate(
                        experiment,
                        hypothesis,
                        data["observables"],
                        data["interventions"],
                        data["contexts"],
                        data["outcomes"],
                    )

    @staticmethod
    def _validate_experiment_idea(
        experiment: Any,
        hypothesis: Any,
        ideas: dict[str, Any],
    ) -> None:
        if not experiment.idea_id:
            return
        if experiment.idea_id not in ideas:
            raise SchemaError(
                f"{experiment.experiment_id} references missing idea "
                f"{experiment.idea_id}"
            )
        idea = ideas[experiment.idea_id]
        if not experiment.search_policy:
            return
        if idea.status not in {"selected", "tested"}:
            raise SchemaError(
                f"{experiment.experiment_id} uses idea {idea.idea_id} with "
                f"non-runnable status {idea.status!r}"
            )
        if idea.novelty_check["status"] != "passed":
            raise SchemaError(
                f"{experiment.experiment_id} uses an idea without a passed "
                "novelty check"
            )
        if idea.hypothesis_id != hypothesis.hypothesis_id:
            raise SchemaError(
                f"{experiment.experiment_id} idea and experiment bind different "
                "hypotheses"
            )
        if (
            str(experiment.search_policy["direction"]) != idea.direction
            or str(experiment.search_policy["subsystem"]) != idea.subsystem
        ):
            raise SchemaError(
                f"{experiment.experiment_id} search policy changes its archived "
                "idea's direction or subsystem"
            )

    def _validate_runs(self, context: ValidationContext) -> None:
        data = context.data
        gated = data["gated_experiments"]
        hypotheses = data["hypotheses"]
        outcomes = data["outcomes"]
        for run in data["runs"].values():
            if run.experiment_id not in gated:
                raise SchemaError(
                    f"{run.run_id} must reference a gated experiment"
                )
            experiment = gated[run.experiment_id]
            if run.hypothesis_id != experiment.hypothesis_id:
                raise SchemaError(
                    f"{run.run_id} hypothesis differs from its experiment"
                )
            if run.arm_id not in {
                str(arm["arm_id"]) for arm in experiment.arms
            }:
                raise SchemaError(
                    f"{run.run_id} references an unknown experiment arm"
                )
            is_diagnostic = any(
                str(tag).startswith("diagnostic_offbudget")
                for tag in run.tags
            )
            if run.seed not in experiment.seeds and not is_diagnostic:
                raise SchemaError(
                    f"{run.run_id} seed is not preregistered by its experiment"
                )
            arm = next(
                item
                for item in experiment.arms
                if str(item["arm_id"]) == run.arm_id
            )
            expected_role = {
                "control": {"baseline", "matched_time", "shuffled_trigger"},
                "treatment": {"treatment"},
                "sham": {"sham"},
            }[str(arm["role"])]
            if (
                run.role not in expected_role
                and run.role not in {"validation", "locked_test"}
            ):
                raise SchemaError(
                    f"{run.run_id} role does not match its preregistered arm"
                )
            if (
                run.spec_fingerprints["hypothesis"]
                != hypotheses[run.hypothesis_id].fingerprint
            ):
                raise SchemaError(
                    f"{run.run_id} hypothesis fingerprint drift"
                )
            if (
                run.spec_fingerprints["experiment"]
                != experiment.fingerprint
            ):
                raise SchemaError(
                    f"{run.run_id} experiment fingerprint drift"
                )
            outcome = outcomes[
                hypotheses[run.hypothesis_id].outcome_id
            ]
            if run.spec_fingerprints["outcome"] != outcome.fingerprint:
                raise SchemaError(
                    f"{run.run_id} outcome fingerprint drift"
                )

    def _validate_refinement(self, context: ValidationContext) -> None:
        registry = self.registry
        data = context.data
        all_experiments = {
            **data["experiment_proposals"],
            **data["gated_experiments"],
        }
        for update in data["evidence_updates"].values():
            if update.experiment_id not in all_experiments:
                raise SchemaError(
                    f"{update.update_id} references missing experiment"
                )
            registry._require_refs(
                update.update_id,
                "evidence",
                update.evidence_ids,
                context.evidence,
            )
            registry._require_refs(
                update.update_id,
                "beliefs",
                update.affected_belief_ids,
                data["beliefs"],
            )
            registry._require_refs(
                update.update_id,
                "hypotheses",
                update.affected_hypothesis_ids,
                data["hypotheses"],
            )
        for decision in data["decisions"].values():
            registry._require_refs(
                decision.decision_id,
                "evidence",
                decision.evidence_ids,
                context.evidence,
            )
        valid_object_ids = set().union(
            data["claims"],
            data["mechanisms"],
            data["observables"],
            data["interventions"],
            data["contexts"],
            data["outcomes"],
            data["hypotheses"],
        )
        for deprecation in data["deprecations"].values():
            if deprecation.object_id not in valid_object_ids:
                raise SchemaError(
                    f"{deprecation.deprecation_id} references missing object"
                )
            registry._require_refs(
                deprecation.deprecation_id,
                "evidence",
                deprecation.evidence_ids,
                context.evidence,
            )

    def _check_generated_views(self, context: ValidationContext) -> None:
        registry = self.registry
        expected = registry.render_state(data=context.data)
        if (
            not registry.state_path.exists()
            or registry.state_path.read_text(encoding="utf-8") != expected
        ):
            context.warnings.append(
                "knowledge/RESEARCH_STATE.md is stale; run "
                "`python -m vibeautoresearch render-state`"
            )
        if not registry.literature_synthesis_path.exists():
            context.warnings.append(
                "knowledge/LITERATURE_SYNTHESIS.md is missing"
            )
            return
        source = registry.literature_synthesis_path.read_text(
            encoding="utf-8"
        )
        try:
            rendered = registry.render_literature_synthesis(
                source=source, data=context.data
            )
        except SchemaError as exc:
            context.warnings.append(
                "knowledge/LITERATURE_SYNTHESIS.md snapshot is invalid: "
                f"{exc}"
            )
        else:
            if source != rendered:
                context.warnings.append(
                    "knowledge/LITERATURE_SYNTHESIS.md snapshot is stale; run "
                    "`python -m vibeautoresearch render-literature` and review "
                    "the ranked queue"
                )


__all__ = ["RegistryValidator", "ValidationContext"]
