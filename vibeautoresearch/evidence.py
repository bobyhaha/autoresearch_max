"""Current-view resolution for append-only evidence correction chains.

``EvidenceRecord`` objects are immutable.  A correction therefore appends a
new record whose ``supersedes_evidence_id`` points at the record it corrects.
This module validates those links and provides one terminal record for every
historical ID.  Callers that answer a *current state* question must use this
index; callers that validate historical provenance may continue to use the raw
append-only mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .core import SchemaError
from .knowledge import EvidenceRecord


# This edge was written before evidence subject-continuity was enforced.  It
# replaces secondary Kimi K3 paper/claim identities with primary-source
# identities, so it cannot satisfy the claim-anchor rule.  Keep the exception
# exact and local: it must not become a tag- or source-based escape hatch.
LEGACY_EVIDENCE_SUBJECT_EXCEPTIONS = frozenset(
    {
        ("evd_lit_kimi_k3", "evd_lit_kimi_k3_primary"),
    }
)


@dataclass(frozen=True)
class TerminalEvidenceIndex:
    """Validated terminal resolution over one complete evidence namespace."""

    history: Mapping[str, EvidenceRecord]
    terminals: Mapping[str, EvidenceRecord]
    successor_by_predecessor: Mapping[str, str]
    terminal_id_by_evidence_id: Mapping[str, str]

    def terminal_id(self, evidence_id: str) -> str:
        """Return the unique terminal ID for a historical or terminal ID."""
        try:
            return self.terminal_id_by_evidence_id[evidence_id]
        except KeyError as exc:
            raise SchemaError(f"missing evidence {evidence_id!r}") from exc

    def terminal_record(self, evidence_id: str) -> EvidenceRecord:
        """Return the current record represented by ``evidence_id``."""
        return self.terminals[self.terminal_id(evidence_id)]

    def terminal_records(
        self, evidence_ids: Iterable[str]
    ) -> tuple[EvidenceRecord, ...]:
        """Resolve IDs to current records, de-duplicating converged citations."""
        result: list[EvidenceRecord] = []
        seen: set[str] = set()
        for evidence_id in evidence_ids:
            terminal_id = self.terminal_id(evidence_id)
            if terminal_id in seen:
                continue
            seen.add(terminal_id)
            result.append(self.terminals[terminal_id])
        return tuple(result)


def _operational_anchor_errors(
    predecessor: EvidenceRecord,
    successor: EvidenceRecord,
) -> list[str]:
    """Return continuity failures for operational evidence subjects."""
    errors: list[str] = []
    if (
        predecessor.experiment_id
        and successor.experiment_id != predecessor.experiment_id
    ):
        errors.append(
            f"experiment_id {predecessor.experiment_id!r} was not preserved"
        )
    missing_runs = sorted(set(predecessor.run_ids) - set(successor.run_ids))
    if missing_runs:
        errors.append(f"run_ids were dropped: {missing_runs}")
    missing_hypotheses = sorted(
        set(predecessor.hypothesis_ids) - set(successor.hypothesis_ids)
    )
    if missing_hypotheses:
        errors.append(f"hypothesis_ids were dropped: {missing_hypotheses}")
    missing_batches = sorted(
        set(predecessor.campaign_batch_ids)
        - set(successor.campaign_batch_ids)
    )
    if missing_batches:
        errors.append(f"campaign_batch_ids were dropped: {missing_batches}")
    return errors


def _validate_subject_continuity(
    predecessor: EvidenceRecord,
    successor: EvidenceRecord,
) -> None:
    """Require a correction edge to retain the evidence subject it revises."""
    edge = (predecessor.evidence_id, successor.evidence_id)
    if predecessor.source_type != successor.source_type:
        raise SchemaError(
            f"evidence supersession {predecessor.evidence_id} -> "
            f"{successor.evidence_id} changes source_type from "
            f"{predecessor.source_type!r} to {successor.source_type!r}"
        )
    if edge in LEGACY_EVIDENCE_SUBJECT_EXCEPTIONS:
        return

    has_operational_anchor = bool(
        predecessor.experiment_id
        or predecessor.run_ids
        or predecessor.hypothesis_ids
        or predecessor.campaign_batch_ids
    )
    if has_operational_anchor:
        errors = _operational_anchor_errors(predecessor, successor)
        if errors:
            raise SchemaError(
                f"evidence supersession {predecessor.evidence_id} -> "
                f"{successor.evidence_id} changes operational subject: "
                + "; ".join(errors)
            )

    # Every literature correction must retain each claim being reassessed,
    # including literature evidence that also binds an experiment, run, or
    # hypothesis.  It may replace deficient secondary paper metadata with a
    # primary paper identity.  Evidence for a genuinely new atomic claim is a
    # separate non-superseding record.
    if predecessor.source_type == "literature":
        missing_claims = sorted(
            set(predecessor.claim_ids) - set(successor.claim_ids)
        )
        if missing_claims:
            raise SchemaError(
                f"evidence supersession {predecessor.evidence_id} -> "
                f"{successor.evidence_id} changes literature subject; "
                f"claim_ids were dropped: {missing_claims}"
            )
        return

    if has_operational_anchor:
        return

    # EvidenceRecord's schema currently guarantees that every non-literature
    # source has at least one operational anchor.  Keep this fail-closed guard
    # in case a future source type relaxes that record-level invariant.
    raise SchemaError(
        f"evidence supersession {predecessor.evidence_id} -> "
        f"{successor.evidence_id} has no stable subject anchor"
    )


def resolve_terminal_evidence(
    evidence: Mapping[str, EvidenceRecord],
) -> TerminalEvidenceIndex:
    """Validate evidence supersession and resolve every ID to one terminal.

    The graph must be a set of non-branching, acyclic chains.  All raw records
    remain in ``history``; only ``terminals`` represents current evidence.
    """
    history = dict(evidence)
    successor_by_predecessor: dict[str, str] = {}

    for evidence_id, item in sorted(history.items()):
        if item.evidence_id != evidence_id:
            raise SchemaError(
                f"evidence mapping key {evidence_id!r} does not match "
                f"record ID {item.evidence_id!r}"
            )
        predecessor_id = item.supersedes_evidence_id
        if not predecessor_id:
            continue
        if predecessor_id == evidence_id:
            raise SchemaError(f"{evidence_id} cannot supersede itself")
        if predecessor_id not in history:
            raise SchemaError(
                f"{evidence_id} supersedes missing evidence {predecessor_id}"
            )
        existing = successor_by_predecessor.get(predecessor_id)
        if existing is not None:
            raise SchemaError(
                f"evidence {predecessor_id} has multiple successors: "
                f"{sorted((existing, evidence_id))}"
            )
        predecessor = history[predecessor_id]
        _validate_subject_continuity(predecessor, item)
        successor_by_predecessor[predecessor_id] = evidence_id

    terminal_id_by_evidence_id: dict[str, str] = {}
    for evidence_id in sorted(history):
        path: list[str] = []
        positions: dict[str, int] = {}
        current = evidence_id
        while current in successor_by_predecessor:
            if current in positions:
                cycle = path[positions[current] :] + [current]
                raise SchemaError(
                    "evidence supersession cycle includes "
                    + " -> ".join(cycle)
                )
            positions[current] = len(path)
            path.append(current)
            current = successor_by_predecessor[current]
        if current in positions:
            cycle = path[positions[current] :] + [current]
            raise SchemaError(
                "evidence supersession cycle includes " + " -> ".join(cycle)
            )
        terminal_id_by_evidence_id[evidence_id] = current

    terminal_ids = set(terminal_id_by_evidence_id.values())
    terminals = {
        evidence_id: history[evidence_id]
        for evidence_id in sorted(terminal_ids)
    }
    return TerminalEvidenceIndex(
        history=history,
        terminals=terminals,
        successor_by_predecessor=successor_by_predecessor,
        terminal_id_by_evidence_id=terminal_id_by_evidence_id,
    )


__all__ = [
    "LEGACY_EVIDENCE_SUBJECT_EXCEPTIONS",
    "TerminalEvidenceIndex",
    "resolve_terminal_evidence",
]
