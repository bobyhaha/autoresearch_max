"""Structured, literature-checked archive of generated research ideas."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .core import (
    SchemaError,
    check_fingerprint,
    fingerprint,
    json_mapping,
    json_mappings,
    local_id,
    require_enum,
    require_id,
    require_int,
    require_keys,
    require_number,
    require_text,
    strings,
)


RESEARCH_DIRECTIONS = {
    "activation",
    "architecture",
    "attention",
    "data_order",
    "evaluation_method",
    "initialization",
    "memory_retrieval",
    "normalization",
    "optimizer",
    "systems_kernel",
    "tokenization",
    "training_schedule",
}
IDEA_STATUSES = {
    "proposed",
    "novelty_rejected",
    "selected",
    "tested",
    "retired",
}
NOVELTY_STATUSES = {"pending", "passed", "rejected"}
NOVELTY_PROVIDERS = {"semantic_scholar", "literature_registry", "combined"}
SCORE_NAMES = ("interestingness", "novelty", "feasibility")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def idea_tokens(*parts: str) -> frozenset[str]:
    """Normalize an idea's text to a token set for lexical overlap."""
    text = " ".join(part for part in parts if part).lower()
    return frozenset(_TOKEN_RE.findall(text))


def lexical_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard token overlap in [0, 1].

    This is a COMPUTED backstop to the proposer's self-reported semantic score,
    not a substitute for a real embedding search. It cannot be gamed by
    re-wording alone: byte-identical or heavily-overlapping ideas score near 1.0
    regardless of what `max_semantic_similarity` the author declares.
    """
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _validate_scores(value: Any) -> dict[str, Any]:
    scores = json_mapping(value, "idea.scores")
    require_keys(scores, "idea.scores", SCORE_NAMES)
    for name in SCORE_NAMES:
        entry = json_mapping(scores[name], f"idea.scores.{name}")
        require_keys(entry, f"idea.scores.{name}", ("score", "rationale"))
        score = require_int(entry["score"], f"idea.scores.{name}.score", minimum=1)
        if score > 10:
            raise SchemaError(f"idea.scores.{name}.score must be <= 10")
        require_text(entry["rationale"], f"idea.scores.{name}.rationale")
    return scores


def _validate_novelty_check(value: Any) -> dict[str, Any]:
    check = json_mapping(value, "idea.novelty_check")
    require_keys(
        check,
        "idea.novelty_check",
        (
            "provider",
            "status",
            "query_rounds",
            "closest_paper_ids",
            "evidence_ids",
            "max_semantic_similarity",
            "discard_threshold",
            "assessment",
        ),
    )
    require_enum(check["provider"], "idea.novelty_check.provider", NOVELTY_PROVIDERS)
    status = require_enum(
        check["status"], "idea.novelty_check.status", NOVELTY_STATUSES
    )
    rounds = json_mappings(check["query_rounds"], "idea.novelty_check.query_rounds")
    if len(rounds) > 10:
        raise SchemaError("idea.novelty_check permits at most ten query rounds")
    for index, round_record in enumerate(rounds):
        field = f"idea.novelty_check.query_rounds[{index}]"
        require_keys(
            round_record,
            field,
            ("query", "result_paper_ids", "assessment"),
        )
        require_text(round_record["query"], f"{field}.query")
        for paper_id in strings(
            round_record["result_paper_ids"], f"{field}.result_paper_ids"
        ):
            require_id(paper_id, "paper")
        require_text(round_record["assessment"], f"{field}.assessment")
    for paper_id in strings(
        check["closest_paper_ids"], "idea.novelty_check.closest_paper_ids"
    ):
        require_id(paper_id, "paper")
    for evidence_id in strings(
        check["evidence_ids"], "idea.novelty_check.evidence_ids"
    ):
        require_id(evidence_id, "evidence")
    maximum = require_number(
        check["max_semantic_similarity"],
        "idea.novelty_check.max_semantic_similarity",
        minimum=0,
        maximum=1,
    )
    threshold = require_number(
        check["discard_threshold"],
        "idea.novelty_check.discard_threshold",
        minimum=0,
        maximum=1,
    )
    require_text(check["assessment"], "idea.novelty_check.assessment")
    if status in {"passed", "rejected"} and not rounds:
        raise SchemaError("a completed novelty check requires at least one query round")
    if status in {"passed", "rejected"} and not check["evidence_ids"]:
        raise SchemaError(
            "a completed novelty check requires structured literature evidence"
        )
    if status == "passed" and maximum >= threshold:
        raise SchemaError(
            "idea novelty cannot pass at or above its semantic-similarity discard threshold"
        )
    if status == "rejected" and maximum < threshold:
        raise SchemaError(
            "idea novelty cannot be rejected below its semantic-similarity threshold"
        )
    return check


@dataclass(frozen=True)
class IdeaRecord:
    idea_id: str
    title: str
    version: int
    summary: str
    experimental_plan: str
    direction: str
    subsystem: str
    parent_idea_ids: tuple[str, ...]
    scores: Mapping[str, Any]
    novelty_check: Mapping[str, Any]
    status: str
    hypothesis_id: str
    created_at: str
    notes: str = ""

    def __post_init__(self) -> None:
        require_id(self.idea_id, "idea")
        require_text(self.title, "idea.title")
        require_int(self.version, "idea.version", minimum=1)
        require_text(self.summary, "idea.summary")
        require_text(self.experimental_plan, "idea.experimental_plan")
        require_enum(self.direction, "idea.direction", RESEARCH_DIRECTIONS)
        local_id(self.subsystem, "idea.subsystem")
        for parent_id in self.parent_idea_ids:
            require_id(parent_id, "idea")
        _validate_scores(self.scores)
        novelty = _validate_novelty_check(self.novelty_check)
        status = require_enum(self.status, "idea.status", IDEA_STATUSES)
        if status in {"selected", "tested"} and novelty["status"] != "passed":
            raise SchemaError(
                f"idea status {status!r} requires a passed literature novelty check"
            )
        if status in {"selected", "tested"} and not self.hypothesis_id:
            raise SchemaError(
                f"idea status {status!r} requires a bound hypothesis_id"
            )
        if status == "novelty_rejected" and novelty["status"] != "rejected":
            raise SchemaError(
                "idea status 'novelty_rejected' requires a rejected novelty check"
            )
        if self.hypothesis_id:
            require_id(self.hypothesis_id, "hypothesis")
        require_text(self.created_at, "idea.created_at")

    @property
    def registry_id(self) -> str:
        return self.idea_id

    def definition(self) -> dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "version": self.version,
            "summary": self.summary,
            "experimental_plan": self.experimental_plan,
            "direction": self.direction,
            "subsystem": self.subsystem,
            "parent_idea_ids": list(self.parent_idea_ids),
            "scores": dict(self.scores),
            "novelty_check": dict(self.novelty_check),
            "hypothesis_id": self.hypothesis_id,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "notes": self.notes,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IdeaRecord":
        record = cls(
            idea_id=require_id(payload.get("idea_id"), "idea"),
            title=require_text(payload.get("title"), "idea.title"),
            version=int(payload.get("version", 1)),
            summary=require_text(payload.get("summary"), "idea.summary"),
            experimental_plan=require_text(
                payload.get("experimental_plan"), "idea.experimental_plan"
            ),
            direction=str(payload.get("direction", "")),
            subsystem=str(payload.get("subsystem", "")),
            parent_idea_ids=strings(
                payload.get("parent_idea_ids"), "idea.parent_idea_ids"
            ),
            scores=json_mapping(payload.get("scores"), "idea.scores"),
            novelty_check=json_mapping(
                payload.get("novelty_check"), "idea.novelty_check"
            ),
            status=str(payload.get("status", "proposed")),
            hypothesis_id=str(payload.get("hypothesis_id", "")),
            created_at=require_text(payload.get("created_at"), "idea.created_at"),
            notes=str(payload.get("notes", "")),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


def duplicate_ideas(ideas: "Mapping[str, IdeaRecord] | list[IdeaRecord]") -> list[tuple[str, str, float]]:
    """Return (idea_id, near_duplicate_idea_id, similarity) triples that FAIL dedup.

    A funded idea (`selected`/`tested`, novelty passed) may not be lexically at or
    above its own declared discard_threshold from any OTHER unrelated archived
    idea. Parent/child lineage is exempt (a mutation legitimately shares text with
    its parent). This is the computed backstop that makes "one funnel per idea"
    resistant to fork-by-near-duplicate: minting a fresh idea_id for identical text
    no longer earns a fresh funnel, because the duplicate is caught here.
    """
    records = list(ideas.values()) if isinstance(ideas, Mapping) else list(ideas)
    tokens = {rec.idea_id: idea_tokens(rec.title, rec.summary, rec.experimental_plan) for rec in records}
    funded = [rec for rec in records if rec.status in {"selected", "tested"}]
    findings: list[tuple[str, str, float]] = []
    for rec in funded:
        threshold = float(rec.novelty_check.get("discard_threshold", 1.0))
        related = set(rec.parent_idea_ids)
        for other in records:
            if other.idea_id == rec.idea_id:
                continue
            if other.idea_id in related or rec.idea_id in set(other.parent_idea_ids):
                continue
            sim = lexical_similarity(tokens[rec.idea_id], tokens[other.idea_id])
            if sim >= threshold:
                findings.append((rec.idea_id, other.idea_id, round(sim, 4)))
    return findings


__all__ = [
    "IDEA_STATUSES",
    "IdeaRecord",
    "NOVELTY_PROVIDERS",
    "NOVELTY_STATUSES",
    "RESEARCH_DIRECTIONS",
    "SCORE_NAMES",
    "duplicate_ideas",
    "idea_tokens",
    "lexical_similarity",
]
