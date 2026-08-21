#!/usr/bin/env python3
"""Proper archival / cleanup for the research ledger.

Archiving is deprecation-with-provenance, never deletion:

- Ideas and tools (mechanism, hypothesis, observable, intervention, context,
  outcome) are archived by setting their record `status` to "deprecated" IN PLACE
  and appending an append-only `DeprecationRecord` that records why and (optionally)
  what replaces them. `render-state` then drops them from the active/executable
  views; the record and its history stay on disk.
- Beliefs are append-only and are NOT deprecated this way. Archive a belief by
  appending a superseding belief version (`supersedes_belief_id`, status in
  challenged/contradicted/context_dependent/inconclusive); stale-scope beliefs are
  demoted automatically by `render-state`. Use `--belief` here only to print that
  guidance.

Usage:
  python tools/archive.py <object_type> <object_id> "<reason>" \
      [--replacement <id>] [--evidence id1,id2] [--root research]

Example:
  python tools/archive.py hypothesis hyp_stale_lr_probe "bracketed and null on the \
2000-step baseline; superseded" --replacement hyp_ngram_shortmem_v1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone

# allow `import vibeautoresearch` when run as a script from tools/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TYPE_TO_PATH = {
    "mechanism": "ideas/mechanisms.jsonl",
    "hypothesis": "ideas/hypotheses.jsonl",
    "observable": "toolkit/available/observables.jsonl",
    "intervention": "toolkit/available/interventions.jsonl",
    "context": "toolkit/available/contexts.jsonl",
    "outcome": "toolkit/available/outcomes.jsonl",
}
ID_FIELD = {
    "mechanism": "mechanism_id", "hypothesis": "hypothesis_id",
    "observable": "observable_id", "intervention": "intervention_id",
    "context": "context_id", "outcome": "outcome_id",
}


def _load(path):
    lines = open(path).read().splitlines() if os.path.exists(path) else []
    header = [ln for ln in lines if ln.strip().startswith("#")]
    recs = [json.loads(ln) for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return header, recs


def _write(path, header, recs):
    # Atomic write: build the full content, write to a temp file, then os.replace
    # so a failure can never leave a partially-written registry.
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for h in header:
            f.write(h + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _refingerprint(obj_type, rec):
    """Recompute the fingerprint via the package record class (status is excluded
    from the fingerprinted definition, but we round-trip to stay safe)."""
    from vibeautoresearch import toolkit, ideas
    cls = {
        "observable": toolkit.ObservableRecord, "intervention": toolkit.InterventionRecord,
        "context": toolkit.ContextRecord, "outcome": toolkit.OutcomeRecord,
        "mechanism": ideas.MechanismRecord, "hypothesis": ideas.HypothesisRecord,
    }[obj_type]
    record = cls.from_dict(rec)  # validates the *current* fingerprint
    record = dataclasses.replace(record, status="deprecated")
    return record.to_dict()      # recomputes fingerprint for the new content


def main():
    ap = argparse.ArgumentParser(description="Archive (deprecate) a ledger record with provenance.")
    ap.add_argument("object_type", choices=list(TYPE_TO_PATH) + ["belief"])
    ap.add_argument("object_id")
    ap.add_argument("reason")
    ap.add_argument("--replacement", default="")
    ap.add_argument("--evidence", default="")
    ap.add_argument("--root", default="research")
    ap.add_argument("--at", default=None, help="ISO timestamp (default: now, UTC)")
    args = ap.parse_args()

    if args.object_type == "belief":
        print("Beliefs are append-only. Archive by appending a superseding belief "
              "(supersedes_belief_id + status challenged/contradicted/context_dependent) "
              "and let render-state demote stale-scope beliefs. Not handled here.",
              file=sys.stderr)
        return 2

    root = args.root
    path = os.path.join(root, TYPE_TO_PATH[args.object_type])
    header, recs = _load(path)
    idf = ID_FIELD[args.object_type]
    idx = next((i for i, r in enumerate(recs) if r.get(idf) == args.object_id), None)
    if idx is None:
        print(f"ERROR: {args.object_type} {args.object_id!r} not found in {path}", file=sys.stderr)
        return 1

    recs[idx] = _refingerprint(args.object_type, recs[idx])
    _write(path, header, recs)

    # append the append-only DeprecationRecord
    dep_path = os.path.join(root, "refinement/deprecations.jsonl")
    dhdr, dreps = _load(dep_path)
    at = args.at or datetime.now(timezone.utc).isoformat()
    dep_id = "dep_" + args.object_id.split("_", 1)[-1] + "_" + at.replace(":", "").replace("-", "")[:14].lower()
    dep_id = "".join(c if (c.isalnum() or c == "_") else "_" for c in dep_id)
    dep = {
        "deprecation_id": dep_id, "object_id": args.object_id,
        "object_type": args.object_type, "reason": args.reason,
        "replacement_id": args.replacement,
        "evidence_ids": [e for e in args.evidence.split(",") if e],
        "deprecated_at": at,
    }
    dreps.append(dep)
    _write(dep_path, dhdr, dreps)
    print(f"archived {args.object_type} {args.object_id} -> status=deprecated; "
          f"deprecation {dep_id} appended. Run: python -m vibeautoresearch validate && render-state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
