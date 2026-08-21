"""Append-only refinement records; raw evidence and runs are never rewritten."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .core import (
    SchemaError,
    json_mappings,
    require_enum,
    require_id,
    require_keys,
    require_text,
    strings,
)


@dataclass(frozen=True)
class EvidenceUpdateRecord:
    update_id: str
    experiment_id: str
    evidence_ids: tuple[str, ...]
    affected_belief_ids: tuple[str, ...]
    affected_hypothesis_ids: tuple[str, ...]
    result: str
    reason: str
    recommended_action: str
    created_at: str

    def __post_init__(self) -> None:
        require_id(self.update_id, "evidence_update")
        require_id(self.experiment_id, "experiment")
        for evidence_id in self.evidence_ids:
            require_id(evidence_id, "evidence")
        if not self.evidence_ids:
            raise SchemaError("evidence update requires evidence_ids")
        for belief_id in self.affected_belief_ids:
            require_id(belief_id, "belief")
        for hypothesis_id in self.affected_hypothesis_ids:
            require_id(hypothesis_id, "hypothesis")
        require_enum(self.result, "evidence_update.result", {"support", "weak_support", "mixed", "null", "oppose", "invalid"})
        require_text(self.reason, "evidence_update.reason")
        require_text(self.recommended_action, "evidence_update.recommended_action")
        require_text(self.created_at, "evidence_update.created_at")

    @property
    def registry_id(self) -> str:
        return self.update_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "experiment_id": self.experiment_id,
            "evidence_ids": list(self.evidence_ids),
            "affected_belief_ids": list(self.affected_belief_ids),
            "affected_hypothesis_ids": list(self.affected_hypothesis_ids),
            "result": self.result,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceUpdateRecord":
        return cls(
            update_id=require_id(payload.get("update_id"), "evidence_update"),
            experiment_id=require_id(payload.get("experiment_id"), "experiment"),
            evidence_ids=strings(payload.get("evidence_ids"), "evidence_update.evidence_ids", allow_empty=False),
            affected_belief_ids=strings(payload.get("affected_belief_ids"), "evidence_update.affected_belief_ids"),
            affected_hypothesis_ids=strings(payload.get("affected_hypothesis_ids"), "evidence_update.affected_hypothesis_ids"),
            result=str(payload.get("result", "")),
            reason=require_text(payload.get("reason"), "evidence_update.reason"),
            recommended_action=require_text(payload.get("recommended_action"), "evidence_update.recommended_action"),
            created_at=require_text(payload.get("created_at"), "evidence_update.created_at"),
        )


@dataclass(frozen=True)
class AuditRecord:
    audit_id: str
    scope: str
    findings: tuple[Mapping[str, Any], ...]
    summary: str
    created_at: str
    created_by: str

    def __post_init__(self) -> None:
        require_id(self.audit_id, "audit")
        require_text(self.scope, "audit.scope")
        for index, finding in enumerate(json_mappings(self.findings, "audit.findings")):
            require_keys(finding, f"audit.findings[{index}]", ("severity", "code", "record_id", "message"))
            require_enum(finding["severity"], "audit.finding.severity", {"info", "warning", "error"})
        require_text(self.summary, "audit.summary")
        require_text(self.created_at, "audit.created_at")
        require_text(self.created_by, "audit.created_by")

    @property
    def registry_id(self) -> str:
        return self.audit_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "scope": self.scope,
            "findings": [dict(item) for item in self.findings],
            "summary": self.summary,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditRecord":
        return cls(
            audit_id=require_id(payload.get("audit_id"), "audit"),
            scope=require_text(payload.get("scope"), "audit.scope"),
            findings=json_mappings(payload.get("findings"), "audit.findings"),
            summary=require_text(payload.get("summary"), "audit.summary"),
            created_at=require_text(payload.get("created_at"), "audit.created_at"),
            created_by=require_text(payload.get("created_by"), "audit.created_by"),
        )


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    decision: str
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    affected_ids: tuple[str, ...]
    replacement_strategy: str
    created_at: str
    created_by: str

    def __post_init__(self) -> None:
        require_id(self.decision_id, "decision")
        require_text(self.decision, "decision.decision")
        strings(self.reasons, "decision.reasons", allow_empty=False)
        for evidence_id in self.evidence_ids:
            require_id(evidence_id, "evidence")
        strings(self.affected_ids, "decision.affected_ids")
        require_text(self.replacement_strategy, "decision.replacement_strategy")
        require_text(self.created_at, "decision.created_at")
        require_text(self.created_by, "decision.created_by")

    @property
    def registry_id(self) -> str:
        return self.decision_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "evidence_ids": list(self.evidence_ids),
            "affected_ids": list(self.affected_ids),
            "replacement_strategy": self.replacement_strategy,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionRecord":
        return cls(
            decision_id=require_id(payload.get("decision_id"), "decision"),
            decision=require_text(payload.get("decision"), "decision.decision"),
            reasons=strings(payload.get("reasons"), "decision.reasons", allow_empty=False),
            evidence_ids=strings(payload.get("evidence_ids"), "decision.evidence_ids"),
            affected_ids=strings(payload.get("affected_ids"), "decision.affected_ids"),
            replacement_strategy=require_text(payload.get("replacement_strategy"), "decision.replacement_strategy"),
            created_at=require_text(payload.get("created_at"), "decision.created_at"),
            created_by=require_text(payload.get("created_by"), "decision.created_by"),
        )


@dataclass(frozen=True)
class DeprecationRecord:
    deprecation_id: str
    object_id: str
    object_type: str
    reason: str
    replacement_id: str
    evidence_ids: tuple[str, ...]
    deprecated_at: str

    def __post_init__(self) -> None:
        require_id(self.deprecation_id, "deprecation")
        require_text(self.object_id, "deprecation.object_id")
        require_enum(
            self.object_type,
            "deprecation.object_type",
            {"claim", "mechanism", "observable", "intervention", "context", "outcome", "hypothesis"},
        )
        require_text(self.reason, "deprecation.reason")
        if self.replacement_id:
            require_text(self.replacement_id, "deprecation.replacement_id")
        for evidence_id in self.evidence_ids:
            require_id(evidence_id, "evidence")
        require_text(self.deprecated_at, "deprecation.deprecated_at")

    @property
    def registry_id(self) -> str:
        return self.deprecation_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "deprecation_id": self.deprecation_id,
            "object_id": self.object_id,
            "object_type": self.object_type,
            "reason": self.reason,
            "replacement_id": self.replacement_id,
            "evidence_ids": list(self.evidence_ids),
            "deprecated_at": self.deprecated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeprecationRecord":
        return cls(
            deprecation_id=require_id(payload.get("deprecation_id"), "deprecation"),
            object_id=require_text(payload.get("object_id"), "deprecation.object_id"),
            object_type=str(payload.get("object_type", "")),
            reason=require_text(payload.get("reason"), "deprecation.reason"),
            replacement_id=str(payload.get("replacement_id", "")),
            evidence_ids=strings(payload.get("evidence_ids"), "deprecation.evidence_ids"),
            deprecated_at=require_text(payload.get("deprecated_at"), "deprecation.deprecated_at"),
        )


__all__ = [
    "AuditRecord",
    "DecisionRecord",
    "DeprecationRecord",
    "EvidenceUpdateRecord",
]
