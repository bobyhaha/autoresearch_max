"""Typed quarantine/registration records for the legacy campaign chart feed.

``campaign_log.jsonl`` predates the authoritative experiment/run/evidence chain.
These records do not upgrade chart rows into RunRecords.  They make every row
accounted for, bind the accounting to an exact content digest, and state whether
the rows have formal provenance or must remain quarantined.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .core import (
    SchemaError,
    canonical_json,
    check_fingerprint,
    fingerprint,
    json_mapping,
    require_bool,
    require_enum,
    require_id,
    require_int,
    require_keys,
    require_text,
    strings,
    tags,
)


CAMPAIGN_DISPOSITIONS = {"quarantined", "registered"}


def campaign_slice_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash semantic row content, independent of JSON whitespace."""
    payload = "".join(f"{canonical_json(dict(row))}\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CampaignBatchRecord:
    """Accounting for one contiguous phase of a non-authoritative campaign log."""

    campaign_batch_id: str
    source_path: str
    phase: str
    first_exp_num: int
    last_exp_num: int
    entry_count: int
    source_sha256: str
    disposition: str
    experiment_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    provenance: Mapping[str, Any]
    summary: str
    recorded_at: str
    recorded_by: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.campaign_batch_id, "campaign_batch")
        require_text(self.source_path, "campaign_batch.source_path")
        require_text(self.phase, "campaign_batch.phase")
        first = require_int(
            self.first_exp_num, "campaign_batch.first_exp_num", minimum=1
        )
        last = require_int(
            self.last_exp_num, "campaign_batch.last_exp_num", minimum=first
        )
        count = require_int(
            self.entry_count, "campaign_batch.entry_count", minimum=1
        )
        if count != last - first + 1:
            raise SchemaError(
                "campaign_batch.entry_count must equal the inclusive exp_num range"
            )
        digest = require_text(
            self.source_sha256, "campaign_batch.source_sha256"
        )
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SchemaError(
                "campaign_batch.source_sha256 must be a lowercase SHA-256 digest"
            )
        disposition = require_enum(
            self.disposition,
            "campaign_batch.disposition",
            CAMPAIGN_DISPOSITIONS,
        )
        for experiment_id in self.experiment_ids:
            require_id(experiment_id, "experiment")
        for run_id in self.run_ids:
            require_id(run_id, "run")
        for evidence_id in self.evidence_ids:
            require_id(evidence_id, "evidence")

        provenance = json_mapping(
            self.provenance, "campaign_batch.provenance"
        )
        require_keys(
            provenance,
            "campaign_batch.provenance",
            (
                "launch_method",
                "gate_binding",
                "run_records_complete",
                "artifact_completeness",
                "limitations",
            ),
        )
        require_enum(
            provenance["launch_method"],
            "campaign_batch.provenance.launch_method",
            {"gated_runner", "direct_ssh", "unknown"},
        )
        gate_binding = require_bool(
            provenance["gate_binding"],
            "campaign_batch.provenance.gate_binding",
        )
        runs_complete = require_bool(
            provenance["run_records_complete"],
            "campaign_batch.provenance.run_records_complete",
        )
        require_enum(
            provenance["artifact_completeness"],
            "campaign_batch.provenance.artifact_completeness",
            {"none", "partial", "complete"},
        )
        limitations = strings(
            provenance["limitations"],
            "campaign_batch.provenance.limitations",
        )

        formal_refs = self.experiment_ids or self.run_ids or self.evidence_ids
        if disposition == "quarantined":
            if formal_refs:
                raise SchemaError(
                    "quarantined campaign batches cannot imply formal registry bindings"
                )
            if gate_binding or runs_complete:
                raise SchemaError(
                    "quarantined campaign batches cannot claim complete provenance"
                )
            if not limitations:
                raise SchemaError(
                    "quarantined campaign batches require explicit provenance limitations"
                )
        elif (
            not self.experiment_ids
            or not self.run_ids
            or not self.evidence_ids
            or not gate_binding
            or not runs_complete
        ):
            raise SchemaError(
                "registered campaign batches require experiment, run, and evidence "
                "bindings plus complete gate/run provenance"
            )

        require_text(self.summary, "campaign_batch.summary")
        require_text(self.recorded_at, "campaign_batch.recorded_at")
        require_text(self.recorded_by, "campaign_batch.recorded_by")
        tags(self.tags, "campaign_batch.tags")

    @property
    def registry_id(self) -> str:
        return self.campaign_batch_id

    def definition(self) -> dict[str, Any]:
        return {
            "campaign_batch_id": self.campaign_batch_id,
            "source_path": self.source_path,
            "phase": self.phase,
            "first_exp_num": self.first_exp_num,
            "last_exp_num": self.last_exp_num,
            "entry_count": self.entry_count,
            "source_sha256": self.source_sha256,
            "disposition": self.disposition,
            "experiment_ids": list(self.experiment_ids),
            "run_ids": list(self.run_ids),
            "evidence_ids": list(self.evidence_ids),
            "provenance": dict(self.provenance),
            "summary": self.summary,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "recorded_at": self.recorded_at,
            "recorded_by": self.recorded_by,
            "tags": list(self.tags),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignBatchRecord":
        record = cls(
            campaign_batch_id=require_id(
                payload.get("campaign_batch_id"), "campaign_batch"
            ),
            source_path=require_text(
                payload.get("source_path"), "campaign_batch.source_path"
            ),
            phase=require_text(payload.get("phase"), "campaign_batch.phase"),
            first_exp_num=payload.get("first_exp_num"),
            last_exp_num=payload.get("last_exp_num"),
            entry_count=payload.get("entry_count"),
            source_sha256=require_text(
                payload.get("source_sha256"), "campaign_batch.source_sha256"
            ),
            disposition=str(payload.get("disposition", "")),
            experiment_ids=strings(
                payload.get("experiment_ids"), "campaign_batch.experiment_ids"
            ),
            run_ids=strings(payload.get("run_ids"), "campaign_batch.run_ids"),
            evidence_ids=strings(
                payload.get("evidence_ids"), "campaign_batch.evidence_ids"
            ),
            provenance=json_mapping(
                payload.get("provenance"), "campaign_batch.provenance"
            ),
            summary=require_text(
                payload.get("summary"), "campaign_batch.summary"
            ),
            recorded_at=require_text(
                payload.get("recorded_at"), "campaign_batch.recorded_at"
            ),
            recorded_by=require_text(
                payload.get("recorded_by"), "campaign_batch.recorded_by"
            ),
            tags=tags(payload.get("tags"), "campaign_batch.tags"),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


__all__ = [
    "CAMPAIGN_DISPOSITIONS",
    "CampaignBatchRecord",
    "campaign_slice_sha256",
]
