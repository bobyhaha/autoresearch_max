#!/usr/bin/env python3
"""Preflight or idempotently register the canonical-key offline kill decision.

Default execution is read-only.  ``--apply`` appends the exact DecisionRecord
only when it is absent; a byte-equivalent existing record is a successful
no-op, while a conflicting definition is a hard failure.

The registry schema requires an offline EvidenceRecord to bind at least one
RunRecord.  This assay intentionally launched no run, so the helper does not
fabricate one and does not misclassify the artifact as literature evidence.
It registers only the governance decision backed by immutable artifact paths
and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ARTIFACT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vibeautoresearch.refinement import DecisionRecord  # noqa: E402
from vibeautoresearch.registry import ResearchRegistry  # noqa: E402


EXPECTED_HASHES = {
    "assay_spec.json": "7721667dac142626ffc0e5d18f414c13c982231a8cc8fb00e9a86c8fa335d17e",
    "engram_canonical_key_assay.py": "c1be30554ed8075afb1a50fc8f71290108579be5285a13a9ec81003371e5900c",
    "result.json": "9d0e3a122a7e880471bc793368576a89afa5cfbe442f1149c1bbc651eed4d761",
    "REPORT.md": "ce1f7d1387c950d7efb06f260d27c32aa8b8bc7653f82c9fd0c5dc1791bca13f",
    "projection.jsonl": "24a8c5845ee1ff1f3b6b8a4d6d1ab2944fd2a1cb7b8c74423f6dbb2b13861d05",
    "alias_sets.jsonl": "cddb6deeb2d6fcee12959cb59ee68d7bf71302b737ae69d311de9e76034f86d1",
    "unexplained_alias_sets.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "packing_events.jsonl": "535c71e8e1336555ff282c300bd77b3362c8229e7b9147c148255a12767a033f",
    "output_manifest.json": "d15dc6b3be202b463befd270abf1af04b8b24c3ab3a6acce831a7a5a58189f63",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts() -> dict:
    for name, expected in EXPECTED_HASHES.items():
        path = ARTIFACT_DIR / name
        if not path.is_file():
            raise RuntimeError(f"missing assay artifact: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"assay artifact hash mismatch for {name}: "
                f"expected {expected}, observed {observed}"
            )
    result = json.loads((ARTIFACT_DIR / "result.json").read_text(encoding="utf-8"))
    expected_facts = {
        "status": "KILL_BEFORE_GPU_SCORING",
        "vocab_collapse": 0.1689453125,
        "minimum_strict_share": 0.0371551513671875,
        "bigram_strict_share": 0.177886962890625,
        "trigram_strict_share": 0.0371551513671875,
        "alias_explained_fraction": 1.0,
    }
    observed_facts = {
        "status": result["verdict"]["status"],
        "vocab_collapse": result["projection"][
            "vocabulary_key_cardinality_collapse"
        ],
        "minimum_strict_share": result["verdict"][
            "minimum_strict_affected_share"
        ],
        "bigram_strict_share": result["ngram_impact"]["2"][
            "strict_affected_occurrence_share"
        ],
        "trigram_strict_share": result["ngram_impact"]["3"][
            "strict_affected_occurrence_share"
        ],
        "alias_explained_fraction": result["projection"][
            "alias_set_explained_fraction"
        ],
    }
    if observed_facts != expected_facts:
        raise RuntimeError(
            f"assay result facts changed: expected {expected_facts}, "
            f"observed {observed_facts}"
        )
    if result["verdict"]["run_authority"] or result["verdict"]["bpb_claim"]:
        raise RuntimeError("assay artifact unexpectedly claims run authority or BPB")
    return result


def decision_record() -> DecisionRecord:
    return DecisionRecord(
        decision_id="dec_paper019_round2_canonical_keys_mediator_kill",
        decision=(
            "kill paper-019 round-2 canonicalized Engram memory keys before "
            "GPU scoring because the frozen strict occurrence mediator failed"
        ),
        reasons=(
            "The exact 8,192-token runtime vocabulary projected to 6,808 "
            "canonical keys, a 0.1689453125 cardinality collapse that passes "
            "the preregistered 0.10 gate.",
            "All 900 alias sets were explained by the declared normalization "
            "pipeline, and all 271 special/BOS/single-byte/replacement/empty "
            "protected IDs remained one-to-one.",
            "On the first 64 exact best-fit packed validation rows at T=2048 "
            "(131,072 positions; packed-input SHA-256 "
            "2f12f64e407c1e3308138483ff1146ad335a5c408097012598d3a43a661421d3), "
            "strict affected shares were 0.177886962890625 for bigrams and "
            "0.0371551513671875 for trigrams.",
            "The frozen occurrence aggregation is the minimum strict share "
            "across orders two and three. Its value, 0.0371551513671875, is "
            "below the preregistered 0.10 gate.",
            "The much larger alias-exposed shares are diagnostic only and "
            "cannot replace the prospective strict metric after observing "
            "the result.",
            "The assay is offline addressing evidence only: it launched no "
            "run, makes no BPB or SOTA claim, and grants no run authority.",
            "The result artifact SHA-256 is "
            "9d0e3a122a7e880471bc793368576a89afa5cfbe442f1149c1bbc651eed4d761; "
            "the output-manifest SHA-256 is "
            "d15dc6b3be202b463befd270abf1af04b8b24c3ab3a6acce831a7a5a58189f63.",
        ),
        evidence_ids=(),
        affected_ids=(
            "pap_deepseek_engram_2026",
            "AI_papers/paper_019_observe_attribute_then_improve_20260729.tex",
            "research/experiments/artifacts/paper019_round2_canonical_key_assay/assay_spec.json",
            "research/experiments/artifacts/paper019_round2_canonical_key_assay/result.json",
            "research/experiments/artifacts/paper019_round2_canonical_key_assay/output_manifest.json",
        ),
        replacement_strategy=(
            "Do not implement or score the canonical-key arm as currently "
            "specified. Leave the cumulative recipe comparator unchanged. "
            "Advance the connected paper-019 body only under the coordinator's "
            "normal typed-chain and run-authority protocol; any materially "
            "different key mechanism requires a new prospective mediator "
            "definition and must not reinterpret this frozen result."
        ),
        created_at="2026-07-29T15:28:28Z",
        created_by="codex_engram_local_census_agent",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="append the exact decision if absent; default is read-only preflight",
    )
    args = parser.parse_args()
    result = verify_artifacts()
    record = decision_record()
    registry = ResearchRegistry(REPO_ROOT / "research")
    existing = registry.decisions.by_id().get(record.decision_id)
    if existing is not None:
        if existing.to_dict() != record.to_dict():
            raise RuntimeError(
                f"conflicting decision already exists: {record.decision_id}"
            )
        print(f"NOOP_ALREADY_PRESENT {record.decision_id}")
        return 0
    if not args.apply:
        print(
            f"PREFLIGHT_OK would append {record.decision_id}; "
            f"verdict={result['verdict']['status']}"
        )
        return 0
    registry.decisions.add(record)
    print(f"APPENDED {record.decision_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
