"""External knowledge, internal observations, evidence, and beliefs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .core import (
    SchemaError,
    check_fingerprint,
    fingerprint,
    json_mapping,
    require_enum,
    require_id,
    require_int,
    require_keys,
    require_text,
    strings,
    tags,
)
from .setup import validate_scope_key


CLAIM_TYPES = {
    "descriptive",
    "correlational",
    "predictive",
    "temporal_association",
    "causal",
    "mechanistic",
    "negative_result",
    "method_definition",
}
TRUST_LEVELS = {"weak", "adequate", "strong", "not_applicable"}
BELIEF_STATUSES = {
    "unsupported",
    "speculative",
    "plausible",
    "supported",
    "strongly_supported",
    "challenged",
    "contradicted",
    "context_dependent",
    "inconclusive",
}


def _version(value: Any, field: str) -> int:
    return require_int(value, field, minimum=1)


def _trust(value: Any, field: str) -> dict[str, Any]:
    result = json_mapping(value, field)
    require_keys(
        result,
        field,
        ("design", "replication", "scope_match", "directness", "limitations"),
    )
    for name in ("design", "replication", "scope_match", "directness"):
        require_enum(result[name], f"{field}.{name}", TRUST_LEVELS)
    strings(result["limitations"], f"{field}.limitations")
    return result


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    authors: tuple[str, ...]
    year: int
    venue: Mapping[str, Any]
    urls: Mapping[str, Any]
    retrieved_at: str
    version: str = ""
    status: str = "active"
    tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        require_id(self.paper_id, "paper")
        require_text(self.title, "paper.title")
        strings(self.authors, "paper.authors", allow_empty=False)
        require_int(self.year, "paper.year", minimum=1900)
        venue = json_mapping(self.venue, "paper.venue")
        require_keys(venue, "paper.venue", ("name", "peer_reviewed"))
        if not isinstance(venue["peer_reviewed"], bool):
            raise SchemaError("paper.venue.peer_reviewed must be boolean")
        urls = json_mapping(self.urls, "paper.urls")
        require_keys(urls, "paper.urls", ("primary",))
        require_text(urls["primary"], "paper.urls.primary")
        require_text(self.retrieved_at, "paper.retrieved_at")
        require_enum(self.status, "paper.status", {"active", "superseded", "retracted"})
        tags(self.tags, "paper.tags")

    @property
    def registry_id(self) -> str:
        return self.paper_id

    def definition(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": dict(self.venue),
            "urls": dict(self.urls),
            "version": self.version,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "retrieved_at": self.retrieved_at,
            "status": self.status,
            "tags": list(self.tags),
            "notes": self.notes,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PaperRecord":
        record = cls(
            paper_id=require_id(payload.get("paper_id"), "paper"),
            title=require_text(payload.get("title"), "paper.title"),
            authors=strings(payload.get("authors"), "paper.authors", allow_empty=False),
            year=int(payload.get("year", 0)),
            venue=json_mapping(payload.get("venue"), "paper.venue"),
            urls=json_mapping(payload.get("urls"), "paper.urls"),
            retrieved_at=require_text(payload.get("retrieved_at"), "paper.retrieved_at"),
            version=str(payload.get("version", "")),
            status=str(payload.get("status", "active")),
            tags=tags(payload.get("tags"), "paper.tags"),
            notes=str(payload.get("notes", "")),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    paper_id: str
    statement: str
    claim_type: str
    scope: Mapping[str, Any]
    limitations: tuple[str, ...]
    locator: str
    extracted_at: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.claim_id, "claim")
        require_id(self.paper_id, "paper")
        require_text(self.statement, "claim.statement")
        require_enum(self.claim_type, "claim.claim_type", CLAIM_TYPES)
        json_mapping(self.scope, "claim.scope")
        strings(self.limitations, "claim.limitations")
        require_text(self.locator, "claim.locator")
        require_text(self.extracted_at, "claim.extracted_at")
        tags(self.tags, "claim.tags")

    @property
    def registry_id(self) -> str:
        return self.claim_id

    def definition(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "paper_id": self.paper_id,
            "statement": self.statement,
            "claim_type": self.claim_type,
            "scope": dict(self.scope),
            "limitations": list(self.limitations),
            "locator": self.locator,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "extracted_at": self.extracted_at,
            "tags": list(self.tags),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClaimRecord":
        record = cls(
            claim_id=require_id(payload.get("claim_id"), "claim"),
            paper_id=require_id(payload.get("paper_id"), "paper"),
            statement=require_text(payload.get("statement"), "claim.statement"),
            claim_type=str(payload.get("claim_type", "")),
            scope=json_mapping(payload.get("scope"), "claim.scope"),
            limitations=strings(payload.get("limitations"), "claim.limitations"),
            locator=require_text(payload.get("locator"), "claim.locator"),
            extracted_at=require_text(payload.get("extracted_at"), "claim.extracted_at"),
            tags=tags(payload.get("tags"), "claim.tags"),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


@dataclass(frozen=True)
class EvidenceRecord:
    """Immutable measurements plus a clearly labelled assessment."""

    evidence_id: str
    source_type: str
    paper_ids: tuple[str, ...]
    claim_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    experiment_id: str
    hypothesis_ids: tuple[str, ...]
    facts: Mapping[str, Any]
    analysis: Mapping[str, Any]
    trust: Mapping[str, Any]
    assessment: Mapping[str, Any]
    artifact_paths: tuple[str, ...]
    created_at: str
    created_by: str
    supersedes_evidence_id: str = ""
    tags: tuple[str, ...] = ()
    campaign_batch_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.evidence_id, "evidence")
        require_enum(
            self.source_type,
            "evidence.source_type",
            {
                "literature",
                "internal_run",
                "offline_analysis",
                "quarantined_campaign",
            },
        )
        for paper_id in self.paper_ids:
            require_id(paper_id, "paper")
        for claim_id in self.claim_ids:
            require_id(claim_id, "claim")
        for run_id in self.run_ids:
            require_id(run_id, "run")
        for campaign_batch_id in self.campaign_batch_ids:
            require_id(campaign_batch_id, "campaign_batch")
        if self.experiment_id:
            require_id(self.experiment_id, "experiment")
        for hypothesis_id in self.hypothesis_ids:
            require_id(hypothesis_id, "hypothesis")
        if self.source_type == "literature" and (not self.paper_ids or not self.claim_ids):
            raise SchemaError("literature evidence requires paper_ids and claim_ids")
        if self.source_type in {"internal_run", "offline_analysis"} and not self.run_ids:
            raise SchemaError(f"{self.source_type} evidence requires run_ids")
        if self.source_type == "quarantined_campaign":
            if not self.campaign_batch_ids:
                raise SchemaError(
                    "quarantined_campaign evidence requires campaign_batch_ids"
                )
            if (
                self.paper_ids
                or self.claim_ids
                or self.run_ids
                or self.experiment_id
            ):
                raise SchemaError(
                    "quarantined_campaign evidence cannot imply paper, claim, run, "
                    "or gated-experiment provenance"
                )
        if not json_mapping(self.facts, "evidence.facts"):
            raise SchemaError("evidence.facts must be non-empty")
        analysis = json_mapping(self.analysis, "evidence.analysis")
        require_keys(analysis, "evidence.analysis", ("method", "code_ref"))
        _trust(self.trust, "evidence.trust")
        assessment = json_mapping(self.assessment, "evidence.assessment")
        require_keys(assessment, "evidence.assessment", ("relation", "strength", "limitations"))
        require_enum(assessment["relation"], "evidence.assessment.relation", {"supports", "opposes", "mixed", "inconclusive", "not_tested"})
        require_enum(assessment["strength"], "evidence.assessment.strength", {"none", "weak", "moderate", "strong"})
        strings(assessment["limitations"], "evidence.assessment.limitations")
        if (
            self.source_type == "quarantined_campaign"
            and assessment["strength"] not in {"none", "weak"}
        ):
            raise SchemaError(
                "quarantined_campaign evidence strength cannot exceed weak"
            )
        strings(self.artifact_paths, "evidence.artifact_paths")
        require_text(self.created_at, "evidence.created_at")
        require_text(self.created_by, "evidence.created_by")
        if self.supersedes_evidence_id:
            require_id(self.supersedes_evidence_id, "evidence")
        tags(self.tags, "evidence.tags")

    @property
    def registry_id(self) -> str:
        return self.evidence_id

    def definition(self) -> dict[str, Any]:
        payload = {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type,
            "paper_ids": list(self.paper_ids),
            "claim_ids": list(self.claim_ids),
            "run_ids": list(self.run_ids),
            "experiment_id": self.experiment_id,
            "hypothesis_ids": list(self.hypothesis_ids),
            "facts": dict(self.facts),
            "analysis": dict(self.analysis),
            "trust": dict(self.trust),
            "artifact_paths": list(self.artifact_paths),
            "created_at": self.created_at,
            "created_by": self.created_by,
            "supersedes_evidence_id": self.supersedes_evidence_id,
            "assessment": dict(self.assessment),
        }
        # Preserve fingerprints for records created before campaign quarantine
        # provenance existed; the optional binding is fingerprinted when present.
        if self.campaign_batch_ids:
            payload["campaign_batch_ids"] = list(self.campaign_batch_ids)
        return payload

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "tags": list(self.tags),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRecord":
        record = cls(
            evidence_id=require_id(payload.get("evidence_id"), "evidence"),
            source_type=str(payload.get("source_type", "")),
            paper_ids=strings(payload.get("paper_ids"), "evidence.paper_ids"),
            claim_ids=strings(payload.get("claim_ids"), "evidence.claim_ids"),
            run_ids=strings(payload.get("run_ids"), "evidence.run_ids"),
            experiment_id=str(payload.get("experiment_id", "")),
            hypothesis_ids=strings(payload.get("hypothesis_ids"), "evidence.hypothesis_ids"),
            facts=json_mapping(payload.get("facts"), "evidence.facts"),
            analysis=json_mapping(payload.get("analysis"), "evidence.analysis"),
            trust=json_mapping(payload.get("trust"), "evidence.trust"),
            assessment=json_mapping(payload.get("assessment"), "evidence.assessment"),
            artifact_paths=strings(payload.get("artifact_paths"), "evidence.artifact_paths"),
            created_at=require_text(payload.get("created_at"), "evidence.created_at"),
            created_by=require_text(payload.get("created_by"), "evidence.created_by"),
            supersedes_evidence_id=str(payload.get("supersedes_evidence_id", "")),
            tags=tags(payload.get("tags"), "evidence.tags"),
            campaign_batch_ids=strings(
                payload.get("campaign_batch_ids"),
                "evidence.campaign_batch_ids",
            ),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    run_ids: tuple[str, ...]
    statement: str
    status: str
    possible_confounds: tuple[str, ...]
    followup_required: bool
    created_at: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Observations use the evidence namespace because they may later be promoted.
        require_id(self.observation_id, "evidence", "observation_id")
        for run_id in self.run_ids:
            require_id(run_id, "run")
        if not self.run_ids:
            raise SchemaError("observation.run_ids must be non-empty")
        require_text(self.statement, "observation.statement")
        require_enum(self.status, "observation.status", {"exploratory", "replicated", "dismissed", "promoted"})
        strings(self.possible_confounds, "observation.possible_confounds")
        if not isinstance(self.followup_required, bool):
            raise SchemaError("observation.followup_required must be boolean")
        require_text(self.created_at, "observation.created_at")
        tags(self.tags, "observation.tags")

    @property
    def registry_id(self) -> str:
        return self.observation_id

    def definition(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "run_ids": list(self.run_ids),
            "statement": self.statement,
            "status": self.status,
            "possible_confounds": list(self.possible_confounds),
            "followup_required": self.followup_required,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "created_at": self.created_at,
            "tags": list(self.tags),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservationRecord":
        record = cls(
            observation_id=require_id(payload.get("observation_id"), "evidence", "observation_id"),
            run_ids=strings(payload.get("run_ids"), "observation.run_ids"),
            statement=require_text(payload.get("statement"), "observation.statement"),
            status=str(payload.get("status", "exploratory")),
            possible_confounds=strings(payload.get("possible_confounds"), "observation.possible_confounds"),
            followup_required=payload.get("followup_required"),
            created_at=require_text(payload.get("created_at"), "observation.created_at"),
            tags=tags(payload.get("tags"), "observation.tags"),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


@dataclass(frozen=True)
class BeliefRecord:
    belief_id: str
    version: int
    statement: str
    status: str
    subject_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    scope: Mapping[str, Any]
    rationale: str
    limitations: tuple[str, ...]
    next_test: str
    created_at: str
    created_by: str
    supersedes_belief_id: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.belief_id, "belief")
        _version(self.version, "belief.version")
        require_text(self.statement, "belief.statement")
        require_enum(self.status, "belief.status", BELIEF_STATUSES)
        if not self.subject_ids:
            raise SchemaError("belief.subject_ids must be non-empty")
        for subject_id in self.subject_ids:
            if not subject_id.startswith(("clm_", "mech_", "hyp_")):
                raise SchemaError(
                    "belief subjects must be claim, mechanism, or hypothesis IDs"
                )
        for evidence_id in self.evidence_ids:
            require_id(evidence_id, "evidence")
        scope = json_mapping(self.scope, "belief.scope")
        if "scope_key" in scope:
            validate_scope_key(scope["scope_key"], "belief.scope.scope_key")
        require_text(self.rationale, "belief.rationale")
        strings(self.limitations, "belief.limitations")
        require_text(self.next_test, "belief.next_test")
        require_text(self.created_at, "belief.created_at")
        require_text(self.created_by, "belief.created_by")
        if self.supersedes_belief_id:
            require_id(self.supersedes_belief_id, "belief")
        tags(self.tags, "belief.tags")

    @property
    def registry_id(self) -> str:
        return self.belief_id

    def definition(self) -> dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "version": self.version,
            "statement": self.statement,
            "scope": dict(self.scope),
            "status": self.status,
            "subject_ids": list(self.subject_ids),
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
            "limitations": list(self.limitations),
            "next_test": self.next_test,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "created_at": self.created_at,
            "created_by": self.created_by,
            "supersedes_belief_id": self.supersedes_belief_id,
            "tags": list(self.tags),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BeliefRecord":
        record = cls(
            belief_id=require_id(payload.get("belief_id"), "belief"),
            version=int(payload.get("version", 1)),
            statement=require_text(payload.get("statement"), "belief.statement"),
            status=str(payload.get("status", "speculative")),
            subject_ids=strings(payload.get("subject_ids"), "belief.subject_ids", allow_empty=False),
            evidence_ids=strings(payload.get("evidence_ids"), "belief.evidence_ids"),
            scope=json_mapping(payload.get("scope"), "belief.scope"),
            rationale=require_text(payload.get("rationale"), "belief.rationale"),
            limitations=strings(payload.get("limitations"), "belief.limitations"),
            next_test=require_text(payload.get("next_test"), "belief.next_test"),
            created_at=require_text(payload.get("created_at"), "belief.created_at"),
            created_by=require_text(payload.get("created_by"), "belief.created_by"),
            supersedes_belief_id=str(payload.get("supersedes_belief_id", "")),
            tags=tags(payload.get("tags"), "belief.tags"),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


__all__ = [
    "BELIEF_STATUSES",
    "CLAIM_TYPES",
    "BeliefRecord",
    "ClaimRecord",
    "EvidenceRecord",
    "ObservationRecord",
    "PaperRecord",
]
