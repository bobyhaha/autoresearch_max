"""JSONL persistence primitives and report value objects.

Extracted from registry.py to separate STORAGE from ORCHESTRATION: this module
holds the append-oriented, file-locked, duplicate-protected line store and the
two immutable report containers. It depends only on core primitives, never on
ResearchRegistry, so the persistence layer can be read and tested in isolation.
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Mapping, TypeVar

from .core import SchemaError, canonical_json

RecordT = TypeVar("RecordT")


class JsonlRegistry(Generic[RecordT]):
    """One object per line, duplicate-protected, append-oriented.

    Immutability — what is and is NOT guaranteed, stated honestly:

    * ENFORCED. ``add`` is the only write path; it opens in append mode under an
      exclusive ``flock``, rejects a duplicate ``registry_id``, then flushes and
      ``fsync``s. There is no update/replace/delete method. In-place edits to a
      fingerprinted record are caught on ``load`` because ``from_dict`` recomputes
      and checks its fingerprint. So *editing history is detected*.
    * NOT ENFORCED. Deleting a line, truncating the file, or reordering records is
      NOT detectable here — the fingerprints are unkeyed self-hashes with no chain
      or external anchor. Tamper-EVIDENCE against deletion needs git or a keyed
      hash chain, which is a codebase-wide property (every registry), not this
      class's job.

    ``append_only`` is a manifest-level DECLARATION of intent, not an extra runtime
    mechanism: the guarantees above already apply to every registry because ``add``
    is the sole mutator. The flag marks which registries the audit treats as
    permanent history; it deliberately does not claim protection this class cannot
    deliver.
    """

    def __init__(self, path: str | Path, record_type: type[RecordT], *, append_only: bool = False):
        self.path = Path(path)
        self.record_type = record_type
        self.append_only = append_only

    def load(self) -> list[RecordT]:
        if not self.path.exists():
            raise SchemaError(f"missing registry: {self.path}")
        records: list[RecordT] = []
        seen: dict[str, int] = {}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise SchemaError("record must be a JSON object")
                record = self.record_type.from_dict(payload)  # type: ignore[attr-defined]
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise SchemaError(f"invalid {self.path} line {line_number}: {exc}") from exc
            identifier = str(record.registry_id)  # type: ignore[attr-defined]
            if identifier in seen:
                raise SchemaError(
                    f"duplicate ID {identifier!r} in {self.path} lines "
                    f"{seen[identifier]} and {line_number}"
                )
            seen[identifier] = line_number
            records.append(record)
        return records

    def by_id(self) -> dict[str, RecordT]:
        return {str(record.registry_id): record for record in self.load()}  # type: ignore[attr-defined]

    def add(self, record: RecordT) -> None:
        if not isinstance(record, self.record_type):
            raise TypeError(f"expected {self.record_type.__name__}")
        identifier = str(record.registry_id)  # type: ignore[attr-defined]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    try:
                        existing = self.record_type.from_dict(json.loads(stripped))  # type: ignore[attr-defined]
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        raise SchemaError(
                            f"invalid {self.path} line {line_number}: {exc}"
                        ) from exc
                    if str(existing.registry_id) == identifier:  # type: ignore[attr-defined]
                        raise SchemaError(
                            f"{identifier!r} already exists in {self.path}"
                        )
                handle.seek(0, os.SEEK_END)
                payload = canonical_json(record.to_dict()) + "\n"  # type: ignore[attr-defined]
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class ValidationReport:
    counts: Mapping[str, int]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "warnings": list(self.warnings)}


@dataclass(frozen=True)
class AuditReport:
    issues: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            severity = str(issue["severity"])
            counts[severity] = counts.get(severity, 0) + 1
        return {"counts": counts, "issues": [dict(item) for item in self.issues]}
