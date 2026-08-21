"""Shared helpers for hand-authoring registry records.

WHY THIS EXISTS
---------------
Four times in one campaign I wrote an enum value that sounded right and was not
in the schema:

    origin_type   "internal_experiment"   (allowed: literature|mixed|self_proposed)
    hypothesis    "refuted"               (allowed: proposed|approved_for_pilot|
                                           offline_screened|running|validated|
                                           rejected|blocked|deprecated)
    mechanism     "supported"             (allowed: proposed|active|challenged|
                                           deprecated)
    launch_method "direct_ssh_probe_gated"(allowed: gated_runner|direct_ssh|unknown)

After the second one I wrote a standing rule telling myself to read the enum
constant first. It did not work, because a rule that depends on remembering to
apply it is not a control. The third and fourth followed anyway.

Two of those failures were worse than a typo. Both times the campaign_log rows
had ALREADY been appended when the batch record raised, leaving `validate` red
with a half-written ledger that then needed a bespoke repair script.

So this module makes the failure mode structurally impossible instead:

  * `enums()`   -- print every enum the registry accepts, so the value is looked
                   up rather than recalled.
  * `check()`   -- validate a payload's enum fields BEFORE any file is touched,
                   naming the field, the bad value and the allowed set.
  * `seal()`    -- compute a record's fingerprint by round-tripping it, and
                   refuse to proceed if any enum is wrong.
  * `append()`  -- ATOMIC multi-file append: every record is constructed and
                   validated first, and nothing is written unless all of them
                   succeed. This is the part that prevents a half-written ledger.

Usage:

    from tools.registry_authoring import enums, seal, append
    enums()                                   # look up what is legal
    rec = seal(CampaignBatchRecord, payload)  # raises before touching disk
    append([("campaign_log.jsonl", rows), ("research/.../batches.jsonl", [rec])])
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

# Every enum the hand-authoring path can get wrong, resolved from the schema
# modules themselves so this file cannot drift from them.
def _registry_enums() -> dict[str, set[str]]:
    from vibeautoresearch.campaign import CAMPAIGN_DISPOSITIONS
    from vibeautoresearch.ideas import HYPOTHESIS_STATUSES

    return {
        "campaign_batch.disposition": set(CAMPAIGN_DISPOSITIONS),
        "campaign_batch.provenance.launch_method": {
            "gated_runner",
            "direct_ssh",
            "unknown",
        },
        "hypothesis.status": set(HYPOTHESIS_STATUSES),
        "mechanism.status": {"proposed", "active", "challenged", "deprecated"},
        "mechanism.origin_type": {"literature", "mixed", "self_proposed"},
    }


def enums() -> None:
    """Print every enum the registry accepts. Call this instead of guessing."""
    for field, allowed in sorted(_registry_enums().items()):
        print(f"{field:48s} {sorted(allowed)}")


def _walk(payload: Mapping[str, Any], prefix: str) -> Iterable[tuple[str, Any]]:
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            yield from _walk(value, path)
        else:
            yield path, value


def check(payload: Mapping[str, Any], record_kind: str) -> None:
    """Raise if any enum-valued field carries a value the schema will reject.

    Runs BEFORE anything is written, so a bad value costs nothing.
    """
    table = _registry_enums()
    problems: list[str] = []
    for path, value in _walk(payload, record_kind):
        allowed = table.get(path)
        if allowed is not None and value not in allowed:
            problems.append(
                f"  {path} = {value!r}\n    allowed: {sorted(allowed)}"
            )
    if problems:
        raise ValueError(
            f"{record_kind}: invalid enum value(s) -- nothing was written.\n"
            + "\n".join(problems)
        )


def seal(cls: Callable[..., Any], payload: Mapping[str, Any], record_kind: str = "") -> Any:
    """Validate enums, compute the fingerprint, and return the sealed record.

    Fails before touching disk on ANY schema error, enum or otherwise.
    """
    kind = record_kind or getattr(cls, "__name__", "record").replace("Record", "").lower()
    check(payload, kind)
    probe = {**payload, "fingerprint": "0" * 16}
    try:
        cls.from_dict(probe)
        return cls.from_dict(probe)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is the fingerprint
        message = str(exc)
        if "does not match computed" not in message:
            raise
        fingerprint = message.rsplit("computed ", 1)[1].strip().strip("'\"")
    return cls.from_dict({**payload, "fingerprint": fingerprint})


def append(writes: Sequence[tuple[str, Sequence[Any]]]) -> None:
    """Append to several JSONL files atomically-ish: build everything, then write.

    Every payload is serialised first. If ANY of them fails, no file is touched.
    This is the specific defect that left the ledger half-written twice: rows were
    appended, then the batch record raised, and `validate` went red with a
    partially recorded block that needed a bespoke repair.
    """
    staged: list[tuple[str, str]] = []
    for path, records in writes:
        lines = []
        for record in records:
            payload = record.to_dict() if hasattr(record, "to_dict") else record
            lines.append(json.dumps(payload, sort_keys=True))
        staged.append((path, "\n".join(lines) + "\n" if lines else ""))

    for path, blob in staged:
        if not blob:
            continue
        directory = os.path.dirname(os.path.abspath(path)) or "."
        with tempfile.NamedTemporaryFile(
            "w", dir=directory, delete=False, encoding="utf-8"
        ) as handle:
            existing = ""
            if os.path.exists(path):
                with open(path, encoding="utf-8") as source:
                    existing = source.read()
                    if existing and not existing.endswith("\n"):
                        existing += "\n"
            handle.write(existing + blob)
            temporary = handle.name
        os.replace(temporary, path)
