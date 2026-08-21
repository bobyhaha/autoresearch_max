#!/usr/bin/env python3
"""Draft a conservative 2025--2026 literature-registry ingestion.

This file is deliberately inert by default.  It constructs the official typed
records and performs duplicate, cross-reference, rating, artifact, validation,
and audit checks without writing ``research/``.  ``--stage-check`` copies the
coherent project dependencies and research tree into ``tmp/``, appends there,
re-renders both generated views, strictly validates, audits, and discards the
stage.  ``--apply`` is available only for this payload-bound reconciled review.

``--apply`` uses a repository-global nonblocking lock,
revalidates the live tree, builds the same isolated stage, binds the full
canonical reviewed payload into a durable manifest, and promotes only an
explicit target allowlist.  Before the first live replacement it saves verified
backups and a recovery journal outside ``research/``; any copy, postvalidation,
postaudit, or whole-tree mismatch restores those targets.  The durable commit
receipt is promoted last.

No mode modifies experiments, creates beliefs, or promotes a literature claim
into a mechanism or hypothesis.  Those remain separately governed actions.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

# A directly invoked script receives its own directory, not the repository
# root, on sys.path.  Resolve and install the repository root before importing
# the typed registry package so the shebang and ``python path/to/script.py``
# behave the same way as ``uv run``.
REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / "vibeautoresearch").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("could not locate the vibeautoresearch repository root")
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vibeautoresearch.core import SchemaError, atomic_write_text, canonical_json  # noqa: E402
from vibeautoresearch.knowledge import ClaimRecord, EvidenceRecord, PaperRecord  # noqa: E402
from vibeautoresearch.registry import ResearchRegistry  # noqa: E402


RESEARCH_ROOT = REPO_ROOT / "research"
TRANSACTION_ROOT = REPO_ROOT / "tmp" / "literature_ingest_transactions"
GLOBAL_LOCK_PATH = TRANSACTION_ROOT / "global.lock"
RETRIEVED_AT = "2026-07-29"
CREATED_AT = "2026-07-29T10:00:32Z"
CREATED_BY = "codex_literature_ingest_draft"

# ---------------------------------------------------------------------------
# INDEPENDENT-CRITIC REVIEW BLOCK -- MUST BE RECONCILED BEFORE --apply
# ---------------------------------------------------------------------------
#
# Scores below are the independent critic's post-review N/P/V/I/R/F/X scores,
# not the proposers' memo scores.  N is local novelty; P is primary provenance;
# V is causal isolation in the stated scope; I is expected effect in the live
# OPHIS gate; R is replication strength; F is feasibility/cost; X is numeric
# falsifiability.  Prestige may raise P only, never V/I/R.
#
CRITIC_REVIEW: dict[str, Any] = {
    "status": "reconciled",
    "reviewer": "codex_literature_critic",
    "reviewed_at": "2026-07-29T10:21:51Z",
    "reviewed_payload_sha256": "6f5526cd50be8d17ca48f8793d316792d0737807ad222ad3e3b547f5d5c1bb5d",
    "required_record_checks": [
        "metadata",
        "atomicity",
        "quantitative_facts",
        "locator",
        "scope_and_assessment",
        "seven_axis_rating",
    ],
    "apply_blockers": [],
    "architecture_source": "architecture_literature_memo.md plus independent primary-source critique",
    "systems_source": "systems-literature multi-agent handoff",
    "systems_scores": {
        "pap_kernelbench_verified_2026": {
            "total": 28,
            "dimensions": {
                "novelty": 2, "provenance": 4, "validity": 5, "impact": 2,
                "reliability": 5, "feasibility_cost": 5, "falsifiability": 5,
            },
        },
        "pap_fastkernels_2026": {
            "total": 25,
            "dimensions": {
                "novelty": 2, "provenance": 4, "validity": 4, "impact": 2,
                "reliability": 4, "feasibility_cost": 4, "falsifiability": 5,
            },
        },
        "pap_sol_execbench_2026": {
            "total": 25,
            "dimensions": {
                "novelty": 2, "provenance": 4, "validity": 4, "impact": 2,
                "reliability": 4, "feasibility_cost": 4, "falsifiability": 5,
            },
        },
        "pap_beyond_random_sampling_2026": {
            "total": 25,
            "dimensions": {
                "novelty": 3, "provenance": 5, "validity": 3, "impact": 3,
                "reliability": 3, "feasibility_cost": 3, "falsifiability": 5,
            },
        },
        "pap_kernelbench_2025": {
            "total": 20,
            "dimensions": {
                "novelty": 1, "provenance": 5, "validity": 3, "impact": 1,
                "reliability": 3, "feasibility_cost": 3, "falsifiability": 4,
            },
        },
        "pap_sparsetransx_2025": {
            "total": 22,
            "dimensions": {
                "novelty": 1, "provenance": 5, "validity": 3, "impact": 2,
                "reliability": 3, "feasibility_cost": 3, "falsifiability": 5,
            },
        },
        "pap_cut_cross_entropy_2025": {
            "total": 23,
            "dimensions": {
                "novelty": 1, "provenance": 5, "validity": 4, "impact": 1,
                "reliability": 4, "feasibility_cost": 3, "falsifiability": 5,
            },
        },
    },
}


# Immutable source captures prepared outside this draft.  The manifest's own
# paper_id is retained as an alias because several capture jobs predate the
# canonical literature IDs used here.
CODE_SNAPSHOT_BINDINGS: dict[str, dict[str, Any]] = {
    "pap_gated_attention_neurips2025": {
        "manifest_paper_id": "pap_r2_arch_gated_attention",
        "repository": "https://github.com/qiuzh20/gated_attention",
        "commit": "f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2",
        "tree": "354de26ef46fb57d418ff98e85fefae2de6d5323",
        "manifest_path": "research/experiments/artifacts/external_code/pap_r2_arch_gated_attention/manifest.json",
        "manifest_sha256": "8b41b61cfa62e862e5272a1e0ddce030e9e099e55123aa6d91da751645ab88c1",
        "archive_path": "research/experiments/artifacts/external_code/pap_r2_arch_gated_attention/gated-attention-f4c2a5f6.tar.zst",
        "archive_sha256": "ac9e5d5d4270ca88071500343b525be2116dbfd68c0c422411d35f2a078270fb",
        "license_family": "MIT",
        "license_sha256": "e7ad7da3894829cde8c6476ceb790bae394d4de11223f1eaebb0cef4da24a136",
        "license_caveat": "No model weights are included; transfer from Qwen3 still requires a local shape/initialization design.",
    },
    "pap_flashattention4_primary": {
        "manifest_paper_id": "pap_flashattention4",
        "repository": "https://github.com/Dao-AILab/flash-attention",
        "commit": "849f660f73b176e5ad5670e7f822c7fa9f3eaf8b",
        "tree": "dbc07053f34000ba50274ad7fbb51ff5411f9ff0",
        "manifest_path": "research/experiments/artifacts/external_code/pap_flashattention4/manifest.json",
        "manifest_sha256": "10c76bbdd9243d38d3773f166974f413c2a60cf6bdfbbe7f4f9d063ccc03e5ff",
        "archive_path": "research/experiments/artifacts/external_code/pap_flashattention4/flash-attention-849f660f.tar.zst",
        "archive_sha256": "deb295841d111a2140f8e469db3997e62aa99fce08d91aa7212d7d41acc8c5db",
        "license_family": "BSD-3-Clause",
        "license_sha256": "8c9ccb96c065e706135b6cbad279b721da6156e51f3a5f27c6b3329af9416d73",
        "license_caveat": "Three pinned gitlink submodules are recorded but not vendored; their source and licenses require separate capture.",
    },
    "pap_flashattention4": {
        "manifest_alias_of": "pap_flashattention4_primary",
    },
    "pap_kernelbench_verified_2026": {
        "manifest_paper_id": "pap_kernelbench_verified",
        "repository": "https://github.com/facebookresearch/kernel_bench_verified",
        "commit": "3fdf6fec7372a4d0cb682635f00e7bdcbc55d50e",
        "tree": "c93ebe7c5325d5fd13624b61c7a38190afe8a50b",
        "manifest_path": "research/experiments/artifacts/external_code/pap_kernelbench_verified/manifest.json",
        "manifest_sha256": "ecf449888ad285b43966a1e7987614faec6932f3fc3bb81a9d7b1559fe293182",
        "archive_path": "research/experiments/artifacts/external_code/pap_kernelbench_verified/kernel-bench-verified-3fdf6fec.tar.zst",
        "archive_sha256": "9d3439ba947e677acd0ea12dd37f35588504f7c7c971e139b1cb5c74c403a013",
        "license_family": "MIT",
        "license_sha256": "da6d3703ed11cbe42bd212c725957c98da23cbff1998c05fa4b3d976d1a58e93",
        "license_caveat": "Published hidden-test files cease to be hidden if proposal agents can inspect this archive.",
    },
    "pap_cut_cross_entropy_2025": {
        "manifest_paper_id": "pap_cut_cross_entropy",
        "repository": "https://github.com/apple/ml-cross-entropy",
        "commit": "b7a02791b234e187b524fb1dba6a812d521b203a",
        "tree": "b0afdd514aed1c47c7470af6bd4cc835bb666ff1",
        "manifest_path": "research/experiments/artifacts/external_code/pap_cut_cross_entropy/manifest.json",
        "manifest_sha256": "9b203d15d8054fb76332ed19cf6e28bfcb69f1a51b6b3e9e490ba72fd6215882",
        "archive_path": "research/experiments/artifacts/external_code/pap_cut_cross_entropy/ml-cross-entropy-b7a02791.tar.zst",
        "archive_sha256": "4222a83d4d25304beac914424bb43d108c9894f78c4d6fb44f9fbf776f28ec94",
        "license_family": "Apple source-code license; non-SPDX label retained",
        "license_sha256": "4862c77f9bf843ff8fca191ae1d124a94849105dba5891bc33aa934440732600",
        "license_caveat": "Redistribution and modification must follow the exact custom Apple license text; permissive SPDX terms are not assumed.",
    },
    "pap_kernelbench_2025": {
        "manifest_paper_id": "pap_kernelbench_original",
        "repository": "https://github.com/ScalingIntelligence/KernelBench",
        "commit": "423217d9fda91e0c2d67e4a43bf62f96f6d104f1",
        "tree": "c4de9ae283314a272a8bd08bec3b246dd8605086",
        "manifest_path": "research/experiments/artifacts/external_code/pap_kernelbench_original/manifest.json",
        "manifest_sha256": "45c5844ef6c9f27373eb817a639c5cab20a1ff4dda0a29806745ffa1182662f2",
        "archive_path": "research/experiments/artifacts/external_code/pap_kernelbench_original/kernelbench-423217d9.tar.zst",
        "archive_sha256": "a7fa556288a7cce83c04986bb0e65faecae3551c4ba39ec7da9cfdf19f8ad041",
        "license_family": "MIT",
        "license_sha256": "fb5917dd8e4476fa75e89ef6f03dccf07d4859636bc23c7db50e6c0413887b9e",
        "license_caveat": "Historical comparison only: standard-input evaluation is superseded-risk and cannot establish a new kernel claim.",
    },
    "pap_recursive_nanochat_autoresearch": {
        "manifest_paper_id": "pap_recursive_nanochat_autoresearch",
        "repository": "https://github.com/recursive-org/first-steps-toward-automated-ai-research",
        "commit": "a962ec43e2e3d7c018e59a2ece623fe6e232fdfb",
        "tree": "d911775753edfd201e7ee6c869b91f4f4bf4544f",
        "manifest_path": "research/experiments/artifacts/external_code/pap_recursive_nanochat_autoresearch/manifest.json",
        "manifest_sha256": "a3c75ba452c103d107225738d881f82985f0e785a075c80d967cd3e74a59d719",
        "archive_path": "research/experiments/artifacts/external_code/pap_recursive_nanochat_autoresearch/recursive-a962ec43.tar.zst",
        "archive_sha256": "4465847c4b9c12a2d51bbbe9349c4a17444255347b8c9a1de245920729281d48",
        "archive_root": "recursive-a962ec43/",
        "bundle_path": "research/experiments/artifacts/external_code/pap_recursive_nanochat_autoresearch/recursive-a962ec43.bundle",
        "bundle_sha256": "432438abbcbcc22fabacf60294aefb2257e1c356d61536c7ad9b81224834b0d8",
        "tree_source": "derived from the verified git bundle because manifest schema v1 omits tree",
        "bundle_ref_caveat": (
            "The bundle advertises refs/heads/archive-a962ec43 while manifest v1 "
            "records refs/remotes/origin/archive-a962ec43; both resolve to the "
            "same captured commit."
        ),
        "license_family": "Apache-2.0 top-level; nested MIT notices",
        "license_sha256": "35e17a66a1afb04921338deac35eda7854f9851379efa545eb7fd36c65801065",
        "notice_sha256": "6c45a5da48887dbcb556bc76cf1548c570423e3cdf2fdee70fc6b417b1be1eae",
        "nested_mit_license_sha256": {
            "nanoGPT": "8c8d17398ecce3dd3fcb1bb77b5d268a427f8250f364ac655c9b38df5b693f6b",
            "nanochat": "42bb09f339a8af8b90d3b4969b6f6e4ba4aff3b5c614103b031fccdf71c8f772",
        },
        "license_caveat": (
            "The v1 manifest omitted license metadata; archive inspection finds "
            "a top-level Apache-2.0 LICENSE and NOTICE plus nested MIT license "
            "notices whose exact hashes are recorded here."
        ),
    },
}


def code_snapshot_binding(paper_id: str) -> dict[str, Any] | None:
    binding = CODE_SNAPSHOT_BINDINGS.get(paper_id)
    if binding is None:
        return None
    alias = binding.get("manifest_alias_of")
    if alias:
        binding = CODE_SNAPSHOT_BINDINGS[str(alias)]
    return dict(binding)


LITERATURE_SNAPSHOT_ALIASES = {
    "pap_flashattention4": "pap_flashattention4_primary",
    "pap_tensorizing_engram": "pap_tensorizing_engram_primary",
}


def _load_literature_snapshot_bindings() -> dict[str, dict[str, Any]]:
    archive_root = (
        REPO_ROOT / "research/experiments/artifacts/external_literature"
    )
    if not archive_root.is_dir():
        raise SchemaError(
            f"external primary-literature archive is absent: {archive_root}"
        )
    bindings: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(archive_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paper_id = str(manifest.get("paper_id", ""))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("classification") != "external_primary_literature"
            or not paper_id
            or manifest_path.parent.name != paper_id
        ):
            raise SchemaError(
                f"invalid primary-literature manifest identity: {manifest_path}"
            )
        artifact = dict(manifest.get("artifact", {}))
        http = dict(manifest.get("http", {}))
        artifact_path = manifest_path.parent / str(artifact.get("path", ""))
        headers_path = manifest_path.parent / str(
            http.get("headers_path", "")
        )
        if not artifact_path.is_file() or not headers_path.is_file():
            raise SchemaError(
                f"primary-literature manifest has absent artifacts: {manifest_path}"
            )
        bindings[paper_id] = {
            "manifest_paper_id": paper_id,
            "title": str(manifest.get("title", "")),
            "source": dict(manifest.get("source", {})),
            "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "artifact_path": str(artifact_path.relative_to(REPO_ROOT)),
            "artifact_bytes": int(artifact["bytes"]),
            "artifact_sha256": str(artifact["sha256"]),
            "headers_path": str(headers_path.relative_to(REPO_ROOT)),
            "headers_bytes": int(http["headers_bytes"]),
            "headers_sha256": str(http["headers_sha256"]),
            "http_status_code": int(http["status_code"]),
            "http_content_type": str(http.get("content_type", "")),
            "validation": dict(manifest.get("validation", {})),
            "access_license": dict(manifest.get("access_license", {})),
            "provenance_caveat": (
                "The snapshot establishes source identity and inspectability, "
                "not claim truth, causal validity, benchmark comparability, or "
                "successful local reproduction."
            ),
        }
    return bindings


LITERATURE_SNAPSHOT_BINDINGS = _load_literature_snapshot_bindings()


def literature_snapshot_binding(paper_id: str) -> dict[str, Any] | None:
    canonical = LITERATURE_SNAPSHOT_ALIASES.get(paper_id, paper_id)
    binding = LITERATURE_SNAPSHOT_BINDINGS.get(canonical)
    return dict(binding) if binding is not None else None


def architecture_rating(
    novelty: int,
    provenance: int,
    validity: int,
    impact: int,
    reliability: int,
    feasibility: int,
    falsifiability: int,
) -> dict[str, Any]:
    dimensions = {
        "novelty": novelty,
        "provenance": provenance,
        "validity": validity,
        "impact": impact,
        "reliability": reliability,
        "feasibility_cost": feasibility,
        "falsifiability": falsifiability,
    }
    return {
        "scale": "1_to_5_each",
        "dimensions": dimensions,
        "total": sum(dimensions.values()),
        "maximum": 35,
        "status": "independent_critic_reconciled",
    }


def systems_rating(paper_id: str) -> dict[str, Any]:
    score = CRITIC_REVIEW["systems_scores"][paper_id]
    return {
        "scale": "1_to_5_each",
        "dimensions": score["dimensions"],
        "total": score["total"],
        "maximum": 35,
        "status": "independent_critic_reconciled",
    }


RATING_DIMENSIONS = {
    "novelty",
    "provenance",
    "validity",
    "impact",
    "reliability",
    "feasibility_cost",
    "falsifiability",
}


def validate_rating(rating: dict[str, Any], *, allow_pending: bool) -> None:
    required_keys = {"scale", "dimensions", "total", "maximum", "status"}
    if set(rating) != required_keys:
        raise SchemaError(
            f"rating keys must be exactly {sorted(required_keys)}, got "
            f"{sorted(rating)}"
        )
    if rating.get("scale") != "1_to_5_each":
        raise SchemaError("rating scale must be '1_to_5_each'")
    if rating.get("status") != "independent_critic_reconciled":
        raise SchemaError("every stored rating must be independently reviewed")
    dimensions = rating.get("dimensions")
    if dimensions is None:
        if allow_pending:
            return
        raise SchemaError("reviewed rating cannot retain dimensions=None")
    if not isinstance(dimensions, dict) or set(dimensions) != RATING_DIMENSIONS:
        raise SchemaError(
            "rating dimensions must be exactly "
            f"{sorted(RATING_DIMENSIONS)}, got {dimensions!r}"
        )
    for name, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise SchemaError(f"rating {name} must be an integer from 1 to 5")
    if sum(dimensions.values()) != rating.get("total"):
        raise SchemaError(
            f"rating dimension sum {sum(dimensions.values())} does not match "
            f"declared total {rating.get('total')!r}"
        )
    if rating.get("maximum") != 35:
        raise SchemaError("rating maximum must be 35")


def make_paper(
    *,
    paper_id: str,
    title: str,
    authors: tuple[str, ...],
    year: int,
    venue_name: str,
    peer_reviewed: bool,
    primary: str,
    organizations: tuple[str, ...],
    tags: tuple[str, ...],
    version: str,
    archive: str = "",
    code: str = "",
    notes: str,
) -> PaperRecord:
    urls = {"primary": primary}
    if archive:
        urls["archive"] = archive
    if code:
        urls["code"] = code
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        venue={"name": venue_name, "peer_reviewed": peer_reviewed},
        urls=urls,
        retrieved_at=RETRIEVED_AT,
        version=version,
        status="active",
        tags=tags,
        notes=(
            f"Organizations: {'; '.join(organizations)}. "
            f"Primary mechanism/evidence read on {RETRIEVED_AT}. {notes}"
        ),
    )


def make_claim(
    *,
    claim_id: str,
    paper_id: str,
    statement: str,
    claim_type: str,
    applies_to: str,
    domain: str,
    evidence_class: str,
    direction: str,
    limitations: tuple[str, ...],
    locator: str,
    tags: tuple[str, ...],
) -> ClaimRecord:
    controlled_directions = {
        "beneficial",
        "harmful",
        "mixed",
        "null",
        "mechanism_only",
        "method_only",
        "descriptive_only",
    }
    if direction not in controlled_directions:
        raise SchemaError(
            f"claim {claim_id!r} direction must be one of "
            f"{sorted(controlled_directions)}, got {direction!r}"
        )
    if claim_type in {"causal", "predictive"} and direction not in {
        "beneficial",
        "harmful",
        "mixed",
        "null",
    }:
        raise SchemaError(
            f"{claim_type} claim {claim_id!r} requires an effect direction"
        )
    if claim_type == "mechanistic" and direction != "mechanism_only":
        raise SchemaError(
            f"mechanistic claim {claim_id!r} requires direction='mechanism_only'"
        )
    return ClaimRecord(
        claim_id=claim_id,
        paper_id=paper_id,
        statement=statement,
        claim_type=claim_type,
        scope={
            "applies_to": applies_to,
            "domain": domain,
            "evidence_class": evidence_class,
            "direction": direction,
        },
        limitations=limitations,
        locator=locator,
        extracted_at=RETRIEVED_AT,
        tags=tags,
    )


def make_evidence(
    *,
    evidence_id: str,
    paper_id: str,
    claim_id: str,
    organizations: tuple[str, ...],
    source_version: str,
    rating: dict[str, Any],
    reported: dict[str, Any],
    mechanism: str,
    method: str,
    code_ref: str,
    design: str,
    replication: str,
    scope_match: str,
    directness: str,
    trust_limitations: tuple[str, ...],
    relation: str,
    strength: str,
    assessment_limitations: tuple[str, ...],
    tags: tuple[str, ...],
    supersedes_evidence_id: str = "",
    hypothesis_ids: tuple[str, ...] = (),
    artifact_paths: tuple[str, ...] = (),
) -> EvidenceRecord:
    snapshot = code_snapshot_binding(paper_id)
    literature_snapshot = literature_snapshot_binding(paper_id)
    facts: dict[str, Any] = {
        "reported": reported,
        "mechanism": mechanism,
        "organizations": list(organizations),
        "source_version": source_version,
        "critic_rating": rating,
        "critic_review": {
            "reviewer": CRITIC_REVIEW["reviewer"],
            "reviewed_at": CRITIC_REVIEW["reviewed_at"],
            "checks": list(CRITIC_REVIEW["required_record_checks"]),
        },
        "rating_caveat": (
            "Prestige contributes only to provenance; it is not treated as "
            "causal validity. Local execution still requires a separately "
            "scored, preregistered idea and active promotion gate."
        ),
    }
    resolved_artifact_paths = list(artifact_paths)
    if snapshot is not None:
        facts["code_snapshot"] = snapshot
        for key in ("manifest_path", "archive_path", "bundle_path"):
            path = str(snapshot.get(key, ""))
            if path and path not in resolved_artifact_paths:
                resolved_artifact_paths.append(path)
    if literature_snapshot is not None:
        facts["primary_source_snapshot"] = literature_snapshot
        for key in ("manifest_path", "headers_path", "artifact_path"):
            path = str(literature_snapshot.get(key, ""))
            if path and path not in resolved_artifact_paths:
                resolved_artifact_paths.append(path)
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_type="literature",
        paper_ids=(paper_id,),
        claim_ids=(claim_id,),
        run_ids=(),
        experiment_id="",
        hypothesis_ids=hypothesis_ids,
        facts=facts,
        analysis={"method": method, "code_ref": code_ref},
        trust={
            "design": design,
            "replication": replication,
            "scope_match": scope_match,
            "directness": directness,
            "limitations": list(trust_limitations),
        },
        assessment={
            "relation": relation,
            "strength": strength,
            "limitations": list(assessment_limitations),
        },
        artifact_paths=tuple(resolved_artifact_paths),
        created_at=CREATED_AT,
        created_by=CREATED_BY,
        supersedes_evidence_id=supersedes_evidence_id,
        tags=tags,
    )


def make_split_evidence(
    *,
    base: EvidenceRecord,
    evidence_id: str,
    claim: ClaimRecord,
    reported: dict[str, Any],
    mechanism: str,
    method: str,
    relation: str,
    strength: str,
    assessment_limitations: tuple[str, ...],
    tags: tuple[str, ...],
    design: str | None = None,
    replication: str | None = None,
    scope_match: str | None = None,
    directness: str | None = None,
) -> EvidenceRecord:
    facts = dict(base.facts)
    facts["reported"] = reported
    facts["mechanism"] = mechanism
    analysis = dict(base.analysis)
    analysis["method"] = method
    trust = dict(base.trust)
    trust["design"] = design or str(base.trust["design"])
    trust["replication"] = replication or str(base.trust["replication"])
    trust["scope_match"] = scope_match or str(base.trust["scope_match"])
    trust["directness"] = directness or str(base.trust["directness"])
    trust["limitations"] = list(claim.limitations)
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_type="literature",
        paper_ids=base.paper_ids,
        claim_ids=(claim.claim_id,),
        run_ids=(),
        experiment_id="",
        hypothesis_ids=(),
        facts=facts,
        analysis=analysis,
        trust=trust,
        assessment={
            "relation": relation,
            "strength": strength,
            "limitations": list(assessment_limitations),
        },
        artifact_paths=base.artifact_paths,
        created_at=CREATED_AT,
        created_by=CREATED_BY,
        supersedes_evidence_id="",
        tags=tags,
    )


RecordTriple = tuple[PaperRecord, ClaimRecord, EvidenceRecord]
TRIPLES: list[RecordTriple] = []
EXTRA_CLAIM_EVIDENCE: list[tuple[ClaimRecord, EvidenceRecord]] = []


# 1. Conditioned Initialization for Attention -- ICLR 2026.
orgs = (
    "Australian Institute for Machine Learning",
    "Adelaide University",
)
paper = PaperRecord(
    paper_id="pap_conditioned_attention_init_iclr2026",
    title="Conditioned Initialization for Attention",
    authors=("Hemanth Saratchandran", "Simon Lucey"),
    year=2026,
    venue={"name": "ICLR 2026", "peer_reviewed": True},
    urls={
        "primary": (
            "https://proceedings.iclr.cc/paper_files/paper/2026/hash/"
            "66d3bfaa4dbc3d2dc579edaee449c5fa-Abstract-Conference.html"
        )
    },
    retrieved_at=RETRIEVED_AT,
    version=(
        "ICLR 2026 proceedings, paper hash "
        "66d3bfaa4dbc3d2dc579edaee449c5fa"
    ),
    status="active",
    tags=(
        "attention",
        "initialization",
        "conditioning",
        "language_modeling",
    ),
    notes=(
        "Bound to the locally captured official proceedings PDF with SHA-256 "
        "cd5d76bb3e849d0e142a647843d0f050611b6acde567093cecea00127c64e94e."
    ),
)
claim = make_claim(
    claim_id="clm_conditioned_init_tinystories_472m",
    paper_id=paper.paper_id,
    statement=(
        "Under the Appendix A.3.4 472M-parameter causal-language-model setting "
        "on TinyStories, independently semi-orthogonal per-head Q/K plus "
        "rectangular-identity V initialization reduced final perplexity from "
        "2.39 to 2.20."
    ),
    claim_type="causal",
    applies_to="qkv_initialization",
    domain="causal_transformer_lm_pretraining",
    evidence_class="paper_faithful",
    direction="beneficial",
    limitations=(
        "This claim is the TinyStories endpoint only; WikiText-103 and the analytical proxy are separate atomic claims.",
        "The paper does not establish seed variance near the local approximately 94.4M-total/56.6M-matrix-parameter model.",
        "OPHIS zero-initializes c_proj: Q/K/V parameter gradients begin at zero, although their attention features determine the first c_proj gradient.",
    ),
    locator="Section 3.3; Appendix A.3.4, TinyStories row in Tables 7-9",
    tags=("attention", "initialization", "qkv", "candidate", "paper_faithful"),
)
evidence = make_evidence(
    evidence_id="evd_lit_conditioned_init_tinystories_472m",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(4, 5, 4, 3, 3, 5, 5),
    reported={
        "472m_tinystories_perplexity": {"default": 2.39, "conditioned": 2.20},
        "runtime_forward_overhead": "none; initialization-only intervention",
    },
    mechanism=(
        "Per-head semi-orthogonal Q/K preserve scale while using independent "
        "orientations; rectangular-identity V has condition number one.  In "
        "the local [128,768] head blocks, the feasible identity is W W^T=I_128."
    ),
    method="Primary-paper mechanism, language tables, and local initialization-path scope comparison.",
    code_ref="train.py attention qkv construction and GPT.init_weights; c_proj zero initialization",
    design="adequate",
    replication="adequate",
    scope_match="adequate",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Strong no-runtime-cost candidate, but local benefit is untested and c_proj zero initialization weakens first-step transfer.",
        "A faithful arm changes Q, K, and V together; Q/K-only is a new local hypothesis.",
    ),
    tags=("attention", "initialization", "candidate", "high_priority"),
)
TRIPLES.append((paper, claim, evidence))

claim = make_claim(
    claim_id="clm_conditioned_init_wikitext103_472m",
    paper_id=paper.paper_id,
    statement=(
        "Under the Appendix A.3.4 472M-parameter causal-language-model setting "
        "on WikiText-103, the same conditioned Q/K/V initialization reduced "
        "final perplexity from 44.2 to 42.7."
    ),
    claim_type="causal",
    applies_to="qkv_initialization",
    domain="causal_transformer_lm_pretraining",
    evidence_class="paper_faithful",
    direction="beneficial",
    limitations=(
        "This claim is the WikiText-103 endpoint only; TinyStories and the analytical proxy are separate atomic claims.",
        "The paper does not establish seed variance near the local approximately 94.4M-total/56.6M-matrix-parameter model.",
        "OPHIS zero-initializes c_proj, so transfer must be diagnosed from the first c_proj gradient and subsequent Q/K/V activation.",
    ),
    locator="Appendix A.3.4, WikiText-103 row in Tables 7-9",
    tags=("attention", "initialization", "qkv", "candidate", "paper_faithful"),
)
evidence = make_evidence(
    evidence_id="evd_lit_conditioned_init_wikitext103_472m",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(4, 5, 4, 3, 3, 5, 5),
    reported={
        "472m_wikitext103_perplexity": {"default": 44.2, "conditioned": 42.7},
        "runtime_forward_overhead": "none; initialization-only intervention",
    },
    mechanism="The same joint semi-orthogonal Q/K and rectangular-identity V initialization is applied before training.",
    method="Primary-paper WikiText-103 table extraction; no local-effect inference.",
    code_ref="train.py attention qkv construction and GPT.init_weights; c_proj zero initialization",
    design="adequate",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Supports the paper's WikiText-103 endpoint, not a guaranteed local BPB effect.",
    ),
    tags=("attention", "initialization", "candidate", "paper_faithful"),
)
EXTRA_CLAIM_EVIDENCE.append((claim, evidence))

claim = make_claim(
    claim_id="clm_conditioned_init_proxy_bound_only",
    paper_id=paper.paper_id,
    statement=(
        "The paper's analysis bounds a surrogate involving the initialized "
        "attention Jacobian and shows the conditioned construction improves that "
        "bound; it does not prove the exact attention-Jacobian condition number."
    ),
    claim_type="mechanistic",
    applies_to="attention_initialization_analysis",
    domain="transformer_attention_theory",
    evidence_class="analytical_proxy",
    direction="mechanism_only",
    limitations=(
        "The result is a surrogate upper-bound comparison rather than an exact condition-number theorem.",
        "The bound alone does not establish a language-model quality effect.",
    ),
    locator="Theory section and proof of the initialization bound",
    tags=("attention", "initialization", "spectral_conditioning", "analytical_proxy"),
)
evidence = make_evidence(
    evidence_id="evd_lit_conditioned_init_proxy_bound_only",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(4, 5, 4, 3, 3, 5, 5),
    reported={
        "proved_object": "surrogate Jacobian upper bound",
        "not_proved": "exact attention-Jacobian condition number",
    },
    mechanism="Independent semi-orthogonal Q/K and rectangular-identity V reduce the paper's initialization proxy bound.",
    method="Primary theorem/proof read with explicit proxy-versus-exact scope separation.",
    code_ref="No runtime code; analytical provenance for a separately gated local initialization arm",
    design="strong",
    replication="not_applicable",
    scope_match="adequate",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="strong",
    assessment_limitations=(
        "Analytical support is confined to the named proxy and must not be rewritten as an exact conditioning guarantee.",
    ),
    tags=("attention", "initialization", "analytical_proxy"),
)
EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 2. Gated Attention -- NeurIPS 2025.
orgs = (
    "Qwen Team, Alibaba Group",
    "University of Edinburgh",
    "Stanford University",
    "Massachusetts Institute of Technology",
    "Tsinghua University",
)
paper = make_paper(
    paper_id="pap_gated_attention_neurips2025",
    title="Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free",
    authors=(
        "Zihan Qiu",
        "Zekun Wang",
        "Bo Zheng",
        "Zeyu Huang",
        "Kaiyue Wen",
        "Songlin Yang",
        "Rui Men",
        "Le Yu",
        "Fei Huang",
        "Suozhi Huang",
        "Dayiheng Liu",
        "Jingren Zhou",
        "Junyang Lin",
    ),
    year=2025,
    venue_name="NeurIPS 2025 Best Paper",
    peer_reviewed=True,
    primary="https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html",
    code="https://github.com/qiuzh20/gated_attention",
    organizations=orgs,
    tags=(
        "attention",
        "gating",
        "nonlinearity",
        "sparsity",
        "attention_sink",
        "training_stability",
        "existing_family",
        "candidate_elementwise",
        "primary_source",
    ),
    version="NeurIPS 2025 proceedings version retrieved 2026-07-29",
    notes="Official models are at https://huggingface.co/QwQZh/gated_attention.",
)
claim = make_claim(
    claim_id="clm_gated_attention_headwise_vs_baseline_15b",
    paper_id=paper.paper_id,
    statement=(
        "At 15B total/2.54B active parameters and 400B tokens, post-SDPA "
        "headwise sigmoid gating improved perplexity from 6.026 to 5.792, "
        "MMLU from 58.79 to 60.05, and GSM8K from 52.92 to 54.44."
    ),
    claim_type="causal",
    applies_to="post_sdpa_gate_granularity",
    domain="transformer_lm_pretraining",
    evidence_class="paper_faithful_headwise_baseline_comparison",
    direction="beneficial",
    limitations=(
        "No model near the local approximately 94.4M-total/56.6M-matrix-parameter scale is reported.",
        "This claim compares paper headwise gating with the paper baseline only; elementwise granularity and sink metrics are separate claims.",
        "OPHIS uses x[:32], 2*sigmoid identity scaling, and per-head output normalization before its gate; the paper uses full pre-normalized X, sigmoid in [0,1], directly after SDPA.",
    ),
    locator="Section 2.2, Equation 5; Table 1; Table 4",
    tags=("attention", "gating", "elementwise", "candidate", "paper_derived"),
)
evidence = make_evidence(
    evidence_id="evd_lit_gated_attention_headwise_vs_baseline_15b",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(2, 5, 3, 2, 3, 4, 5),
    reported={
        "baseline": {"perplexity": 6.026, "mmlu": 58.79, "gsm8k": 52.92},
        "headwise": {"perplexity": 5.792, "mmlu": 60.05, "gsm8k": 54.44},
        "reported_wall_clock_overhead": "less than 2 percent",
    },
    mechanism=(
        "A query-dependent sigmoid gate after attention introduces nonlinearity, "
        "suppresses sink-dominated outputs, and limits extreme activations."
    ),
    method="Primary-paper headwise-versus-baseline extraction plus explicit local implementation comparison.",
    code_ref="train.py query-derived headwise attention gate before c_proj",
    design="strong",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Supports the paper's headwise result; it does not validate the materially different existing OPHIS gate.",
    ),
    tags=("attention", "gating", "candidate", "existing_family"),
)
TRIPLES.append((paper, claim, evidence))

claim = make_claim(
    claim_id="clm_gated_attention_elementwise_vs_headwise_15b",
    paper_id=paper.paper_id,
    statement=(
        "In the same 15B total/2.54B active, 400B-token ablation, elementwise "
        "gating changed perplexity/MMLU/GSM8K from the headwise values "
        "5.792/60.05/54.44 to 5.761/60.82/55.27."
    ),
    claim_type="causal",
    applies_to="post_sdpa_gate_granularity",
    domain="transformer_lm_pretraining",
    evidence_class="paper_derived_local_granularity_test",
    direction="beneficial",
    limitations=(
        "The paper says gating granularity has relatively minor impact.",
        "Elementwise gating adds 201M parameters in the paper, versus 1.6M for headwise gating.",
        "The local width-only refinement adds exactly 195,072 weights but preserves x[:32], 2*sigmoid identity scaling, and a different placement, so it is paper-derived rather than paper-faithful.",
    ),
    locator="Section 2.2, Equation 5; Table 1",
    tags=("attention", "gating", "elementwise", "candidate", "paper_derived"),
)
evidence = make_evidence(
    evidence_id="evd_lit_gated_attention_elementwise_vs_headwise_15b",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(2, 5, 3, 2, 3, 4, 5),
    reported={
        "headwise": {
            "perplexity": 5.792, "mmlu": 60.05, "gsm8k": 54.44,
            "added_parameters": 1_600_000,
        },
        "elementwise": {
            "perplexity": 5.761, "mmlu": 60.82, "gsm8k": 55.27,
            "added_parameters": 201_000_000,
        },
        "paper_interpretation": "granularity has relatively minor impact",
        "local_width_only_added_weights": 195_072,
    },
    mechanism="Elementwise gates allow within-head channel modulation beyond one scalar per head.",
    method="Primary granularity ablation; local parameter delta recomputed from (32*768-32*6)*8.",
    code_ref="train.py existing query-derived headwise gate before c_proj",
    design="adequate",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "The direct paper advantage is small and confounded with a much larger parameter increase.",
        "Any local width-only arm is a new controlled hypothesis, not a faithful paper replication.",
    ),
    tags=("attention", "gating", "candidate", "paper_derived"),
)
EXTRA_CLAIM_EVIDENCE.append((claim, evidence))

claim = make_claim(
    claim_id="clm_gated_attention_sink_metrics",
    paper_id=paper.paper_id,
    statement=(
        "For the paper's reported gated configuration, first-token attention "
        "mass decreased from 0.467 in the baseline to 0.048."
    ),
    claim_type="causal",
    applies_to="attention_sink_mass",
    domain="transformer_lm_attention_diagnostics",
    evidence_class="paper_faithful_diagnostic",
    direction="beneficial",
    limitations=(
        "This is an attention diagnostic, not a standalone quality endpoint.",
        "The metric does not identify whether nonlinearity, sparsity, or gate placement is the sole cause.",
        "The local gate differs in input width, scale, and placement.",
    ),
    locator="Attention-sink analysis and reported first-token attention-mass table",
    tags=("attention", "gating", "attention_sink", "diagnostic"),
)
evidence = make_evidence(
    evidence_id="evd_lit_gated_attention_sink_metrics",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(2, 5, 3, 2, 3, 4, 5),
    reported={"first_token_attention_mass": {"baseline": 0.467, "gated": 0.048}},
    mechanism="Query-dependent sparse gates suppress sink-dominated SDPA outputs.",
    method="Primary-paper attention-sink diagnostic extraction.",
    code_ref="Any local test must log the same diagnostic; BPB alone cannot validate the sink mechanism",
    design="adequate",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Supports the paper's diagnostic only, not the local gate or a local BPB prediction.",
    ),
    tags=("attention", "gating", "attention_sink", "diagnostic"),
)
EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 3. Native Sparse Attention -- ACL 2025.
orgs = (
    "DeepSeek-AI",
    "Peking University Key Laboratory and PKU-Anker LLM Lab",
    "University of Washington",
)
paper = make_paper(
    paper_id="pap_native_sparse_attention_acl2025",
    title="Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention",
    authors=(
        "Jingyang Yuan",
        "Huazuo Gao",
        "Damai Dai",
        "Junyu Luo",
        "Liang Zhao",
        "Zhengyan Zhang",
        "Zhenda Xie",
        "Yuxing Wei",
        "Lean Wang",
        "Zhiping Xiao",
        "Yuqing Wang",
        "Chong Ruan",
        "Ming Zhang",
        "Wenfeng Liang",
        "Wangding Zeng",
    ),
    year=2025,
    venue_name="ACL 2025 Best Paper",
    peer_reviewed=True,
    primary="https://aclanthology.org/2025.acl-long.1126/",
    archive="https://arxiv.org/abs/2502.11089",
    organizations=orgs,
    tags=(
        "sparse_attention",
        "compression",
        "selection",
        "sliding_window",
        "hardware_codesign",
        "long_context",
        "exclude_local",
        "primary_source",
    ),
    version="ACL 2025 proceedings version retrieved 2026-07-29",
    notes="No official author implementation was linked by the paper.",
)
claim = make_claim(
    claim_id="clm_nsa_short_task_aggregate",
    paper_id=paper.paper_id,
    statement=(
        "At 27B total/3B active parameters after 270B-token pretraining and "
        "32K extension, a learned gated combination of compressed block summaries, "
        "top-selected blocks, and local attention improved the reported short-task "
        "aggregate from 0.443 to 0.456."
    ),
    claim_type="causal",
    applies_to="hierarchical_sparse_attention",
    domain="long_context_transformer_lm",
    evidence_class="paper_faithful_scope_mismatch",
    direction="beneficial",
    limitations=(
        "The cited system benefit depends on custom kernels and contexts up to 64K.",
        "At OPHIS length 2048, the reported local and selected-token coverage removes most sparsity advantage.",
        "No comparable approximately 94.4M-total/56.6M-matrix-parameter short-budget evidence is reported.",
    ),
    locator="Short-task aggregate row in Tables 1-2",
    tags=("sparse_attention", "long_context", "scope_mismatch", "exclude_local"),
)
evidence = make_evidence(
    evidence_id="evd_lit_nsa_short_task_aggregate",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(4, 5, 4, 1, 4, 1, 4),
    reported={
        "short_task_aggregate": {"baseline": 0.443, "nsa": 0.456},
    },
    mechanism="Learned compressed, selected, and local attention branches share selector information and use hardware-aligned blocks.",
    method="Primary-paper mechanism/evidence read followed by context-length and kernel-availability scope test.",
    code_ref="train.py FA3 varlen attention at sequence length 2048; no NSA custom kernel",
    design="strong",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Credible long-context mechanism but an explicit non-candidate in the present 2K/no-new-kernel frame.",
        "A PyTorch reconstruction would likely reduce optimizer steps under the fixed 300-second budget.",
    ),
    tags=("sparse_attention", "scope_mismatch", "negative_selection_evidence"),
)
TRIPLES.append((paper, claim, evidence))

nsa_base_evidence = evidence
for claim, evidence in (
    (
        make_claim(
            claim_id="clm_nsa_longbench_aggregate",
            paper_id=paper.paper_id,
            statement=(
                "In the same 27B-total/3B-active model comparison, NSA improved "
                "the reported LongBench aggregate from 0.437 to 0.469."
            ),
            claim_type="causal",
            applies_to="hierarchical_sparse_attention",
            domain="long_context_transformer_lm",
            evidence_class="paper_faithful_scope_mismatch",
            direction="beneficial",
            limitations=(
                "LongBench is a long-context aggregate and does not transfer directly to the local 2048-token validation BPB.",
                "The model scale and training budget are far larger than OPHIS.",
            ),
            locator="LongBench row in Tables 1-2",
            tags=("sparse_attention", "longbench", "scope_mismatch"),
        ),
        None,
    ),
    (
        make_claim(
            claim_id="clm_nsa_64k_forward_kernel_speedup",
            paper_id=paper.paper_id,
            statement="At 64K context on 8 A100 GPUs, the custom NSA forward kernel reported up to approximately 9x speedup.",
            claim_type="causal",
            applies_to="nsa_custom_forward_kernel",
            domain="long_context_attention_kernel",
            evidence_class="paper_faithful_hardware_scope",
            direction="beneficial",
            limitations=(
                "The measurement is 64K on 8 A100 GPUs, not 2K on one H200.",
                "No compatible local NSA kernel is available.",
            ),
            locator="Kernel evaluation, Figures 5-6",
            tags=("sparse_attention", "kernel", "forward", "scope_mismatch"),
        ),
        None,
    ),
    (
        make_claim(
            claim_id="clm_nsa_64k_backward_kernel_speedup",
            paper_id=paper.paper_id,
            statement="At 64K context on 8 A100 GPUs, the custom NSA backward kernel reported up to approximately 6x speedup.",
            claim_type="causal",
            applies_to="nsa_custom_backward_kernel",
            domain="long_context_attention_kernel",
            evidence_class="paper_faithful_hardware_scope",
            direction="beneficial",
            limitations=(
                "The measurement is 64K on 8 A100 GPUs, not 2K on one H200.",
                "No compatible local NSA kernel is available.",
            ),
            locator="Kernel evaluation, Figures 5-6",
            tags=("sparse_attention", "kernel", "backward", "scope_mismatch"),
        ),
        None,
    ),
    (
        make_claim(
            claim_id="clm_nsa_estimated_decoding_speedup",
            paper_id=paper.paper_id,
            statement="The paper estimates an 11.6x NSA decoding speedup in its long-context serving analysis.",
            claim_type="predictive",
            applies_to="nsa_long_context_decoding",
            domain="long_context_inference",
            evidence_class="paper_estimate_scope_mismatch",
            direction="beneficial",
            limitations=(
                "This is an estimate rather than a measured OPHIS training result.",
                "OPHIS has no decoding/KV-cache objective.",
            ),
            locator="Decoding-speed analysis",
            tags=("sparse_attention", "decoding", "estimate", "scope_mismatch"),
        ),
        None,
    ),
):
    if claim.claim_id == "clm_nsa_longbench_aggregate":
        reported = {"longbench": {"baseline": 0.437, "nsa": 0.469}}
        mechanism = "The three learned sparse branches retain long-context information with lower attention density."
    elif claim.claim_id == "clm_nsa_64k_forward_kernel_speedup":
        reported = {"64k_forward_speedup": "up to approximately 9x on 8xA100"}
        mechanism = "Hardware-aligned block sparsity reduces forward attention work at long context."
    elif claim.claim_id == "clm_nsa_64k_backward_kernel_speedup":
        reported = {"64k_backward_speedup": "up to approximately 6x on 8xA100"}
        mechanism = "Hardware-aligned block sparsity reduces backward attention work at long context."
    else:
        reported = {"estimated_decoding_speedup": "11.6x"}
        mechanism = "Sparse selected/local attention reduces long-context decoding work."
    evidence = make_split_evidence(
        base=nsa_base_evidence,
        evidence_id=f"evd_lit_{claim.claim_id.removeprefix('clm_')}",
        claim=claim,
        reported=reported,
        mechanism=mechanism,
        method="Primary-paper extraction of one endpoint with hardware/context scope held explicit.",
        relation="supports",
        strength="moderate" if claim.claim_id == "clm_nsa_longbench_aggregate" else "weak",
        assessment_limitations=(
            "This endpoint is not evidence of a local 2K H200 training benefit.",
        ),
        tags=tuple(claim.tags),
        scope_match="weak",
    )
    EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 4. Multi-Token Attention -- COLM 2025.
orgs = ("Meta Fundamental AI Research",)
paper = make_paper(
    paper_id="pap_multi_token_attention_colm2025",
    title="Multi-Token Attention",
    authors=("Olga Golovneva", "Tianlu Wang", "Jason Weston", "Sainbayar Sukhbaatar"),
    year=2025,
    venue_name="COLM 2025",
    peer_reviewed=True,
    primary="https://openreview.net/forum?id=UnQzC2cECV",
    code="https://github.com/facebookresearch/RAM/tree/main/projects/mta",
    organizations=orgs,
    tags=(
        "attention",
        "causal_convolution",
        "multi_token_context",
        "head_mixing",
        "attention_map",
        "fa3_incompatible",
        "exclude_local",
        "primary_source",
    ),
    version="COLM 2025 proceedings version retrieved 2026-07-29",
    notes="Official code is the RAM repository MTA project.",
)
claim = make_claim(
    claim_id="clm_mta_average_perplexity_880m",
    paper_id=paper.paper_id,
    statement=(
        "Causal convolutions over the query and key axes of attention maps and "
        "across heads improved 880M/105B-token average perplexity from 11.25 to "
        "10.91."
    ),
    claim_type="causal",
    applies_to="attention_map_multi_token_convolution",
    domain="transformer_lm_pretraining",
    evidence_class="paper_faithful_systems_negative",
    direction="beneficial",
    limitations=(
        "The mechanism requires attention-map access/materialization unavailable through the current FA3 call.",
        "Only two runs are reported for the central 880M setting.",
        "The paper itself reports a roughly 9.5x unoptimized H200 throughput loss.",
    ),
    locator="Quality ablation, Tables 2 and 5",
    tags=("attention", "multi_token", "throughput", "h200", "exclude_local"),
)
evidence = make_evidence(
    evidence_id="evd_lit_mta_average_perplexity_880m",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(4, 5, 3, 1, 3, 1, 3),
    reported={
        "average_perplexity": {"baseline": 11.25, "mta": 10.91},
    },
    mechanism="Causal convolution contextualizes attention scores along query/key token axes and mixes attention heads.",
    method="Primary-paper quality and candid H200 resource-table read; FA3 compatibility and wall-clock transfer analysis.",
    code_ref="train.py flash_attn_varlen_func; fixed 300-second scheduler",
    design="adequate",
    replication="weak",
    scope_match="adequate",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Mechanism is credible, but the paper's own H200 evidence falsifies feasibility for this campaign without a fused FA3-compatible kernel.",
    ),
    tags=("attention", "h200", "throughput", "negative_selection_evidence"),
)
TRIPLES.append((paper, claim, evidence))

mta_base_evidence = evidence
for spec in (
    {
        "claim_id": "clm_mta_lambada_perplexity_880m",
        "statement": "At 880M parameters and 105B training tokens, MTA reduced reported LAMBADA perplexity from 17.6 to 13.2.",
        "applies_to": "attention_map_multi_token_convolution",
        "direction": "beneficial",
        "reported": {"lambada_perplexity": {"baseline": 17.6, "mta": 13.2}},
        "locator": "LAMBADA row in the quality tables",
        "relation": "supports",
        "strength": "moderate",
    },
    {
        "claim_id": "clm_mta_h200_throughput_cost",
        "statement": "In the reported unoptimized 32-H200 implementation, MTA reduced throughput from 54.3K to 5.7K tokens per second.",
        "applies_to": "unoptimized_mta_h200_runtime",
        "direction": "harmful",
        "reported": {"unoptimized_h200_tokens_per_second": {"baseline": 54300, "mta": 5700}},
        "locator": "H200 resource Table 9",
        "relation": "supports",
        "strength": "strong",
    },
    {
        "claim_id": "clm_mta_h200_memory_cost",
        "statement": "In the reported unoptimized 32-H200 implementation, MTA increased memory from 17.5GB to 73.8GB.",
        "applies_to": "unoptimized_mta_h200_memory",
        "direction": "harmful",
        "reported": {"unoptimized_h200_memory_gb": {"baseline": 17.5, "mta": 73.8}},
        "locator": "H200 resource Table 10",
        "relation": "supports",
        "strength": "strong",
    },
):
    claim = make_claim(
        claim_id=spec["claim_id"],
        paper_id=paper.paper_id,
        statement=spec["statement"],
        claim_type="causal",
        applies_to=spec["applies_to"],
        domain="transformer_lm_pretraining",
        evidence_class="paper_faithful_atomic_endpoint",
        direction=spec["direction"],
        limitations=(
            "The mechanism requires attention-map access/materialization unavailable through the current FA3 call.",
            "Only two runs are reported for the central 880M setting.",
            "The systems measurements describe an unoptimized 32-H200 implementation.",
        ),
        locator=spec["locator"],
        tags=("attention", "multi_token", "h200", "atomic_endpoint"),
    )
    evidence = make_split_evidence(
        base=mta_base_evidence,
        evidence_id=f"evd_lit_{claim.claim_id.removeprefix('clm_')}",
        claim=claim,
        reported=spec["reported"],
        mechanism=(
            "Causal attention-map convolution changes the named endpoint; "
            "attention-map materialization drives the separately recorded costs."
        ),
        method="Primary-paper extraction of one quality or resource endpoint.",
        relation=spec["relation"],
        strength=spec["strength"],
        assessment_limitations=(
            "This endpoint is assessed independently and does not average quality with feasibility.",
        ),
        tags=tuple(claim.tags),
    )
    EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 5. Byte Latent Transformer -- ACL 2025.
orgs = (
    "Meta Fundamental AI Research",
    "Paul G. Allen School, University of Washington",
    "University of Chicago",
)
paper = make_paper(
    paper_id="pap_blt_acl2025",
    title="Byte Latent Transformer: Patches Scale Better Than Tokens",
    authors=(
        "Artidoro Pagnoni",
        "Ramakanth Pasunuru",
        "Pedro Rodriguez",
        "John Nguyen",
        "Benjamin Muller",
        "Margaret Li",
        "Chunting Zhou",
        "Lili Yu",
        "Jason Weston",
        "Luke Zettlemoyer",
        "Gargi Ghosh",
        "Mike Lewis",
        "Ari Holtzman",
        "Srinivasan Iyer",
    ),
    year=2025,
    venue_name="ACL 2025 Outstanding Paper",
    peer_reviewed=True,
    primary="https://aclanthology.org/2025.acl-long.453/",
    code="https://github.com/facebookresearch/blt",
    organizations=orgs,
    tags=(
        "byte_level",
        "dynamic_patching",
        "tokenization",
        "entropy",
        "ngram_hash",
        "robustness",
        "frozen_scope",
        "exclude_local",
        "primary_source",
    ),
    version="ACL 2025 proceedings version retrieved 2026-07-29",
    notes="Official code linked from the paper.",
)
claim = make_claim(
    claim_id="clm_blt_8b_flop_matched_aggregate",
    paper_id=paper.paper_id,
    statement=(
        "Entropy-patched byte modeling exceeded a FLOP-matched BPE Llama only at "
        "large scale: at 8B parameters, the reported aggregate was 61.1 for "
        "entropy BLT versus 60.0 for the FLOP-matched BPE Llama."
    ),
    claim_type="causal",
    applies_to="entropy_patched_byte_modeling",
    domain="large_scale_lm_pretraining",
    evidence_class="paper_faithful_negative_small_budget",
    direction="beneficial",
    limitations=(
        "BLT changes tokenizer, sequence construction, model topology, and effective compute rather than one architecture factor.",
        "The paper reports BPE superiority at smaller compute budgets.",
        "OPHIS tokenizer, data interface, and evaluation are frozen.",
    ),
    locator="8B row in Tables 1-3",
    tags=("byte_level", "tokenization", "negative_evidence", "exclude_local"),
)
evidence = make_evidence(
    evidence_id="evd_lit_blt_8b_flop_matched_aggregate",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(4, 5, 5, 1, 4, 1, 3),
    reported={
        "8b_flop_matched_aggregate": {"bpe_llama": 60.0, "entropy_blt": 61.1},
    },
    mechanism="Entropy-defined byte patches reduce global-transformer steps while local byte modules and hash embeddings recover fine detail.",
    method="Primary-paper compute-matching, scaling, and tokenizer-scope analysis.",
    code_ref="Frozen tokenizer/data/evaluation protocol; train.py BPE-token n-gram value memory",
    design="strong",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="strong",
    assessment_limitations=(
        "Valuable negative evidence: the paper itself predicts no advantage in this small frozen-tokenizer budget.",
    ),
    tags=("byte_level", "scope_mismatch", "negative_selection_evidence"),
)
TRIPLES.append((paper, claim, evidence))

blt_base_evidence = evidence
for spec in (
    {
        "claim_id": "clm_blt_small_family_crossover_150b_bytes",
        "statement": "The paper's smaller-family scaling analysis estimates that BLT does not overtake BPE before approximately 150B training bytes.",
        "claim_type": "predictive",
        "direction": "harmful",
        "reported": {"estimated_small_family_crossover_bytes": 150_000_000_000},
        "locator": "Scaling-law crossover discussion",
        "strength": "moderate",
    },
    {
        "claim_id": "clm_blt_noisy_hellaswag_8b",
        "statement": "In the reported noisy HellaSwag evaluation, BLT scored 64.3 versus 56.9 for the BPE comparator.",
        "claim_type": "causal",
        "direction": "beneficial",
        "reported": {"noisy_hellaswag": {"bpe": 56.9, "blt": 64.3}},
        "locator": "Noisy HellaSwag row",
        "strength": "strong",
    },
    {
        "claim_id": "clm_blt_hash_3to8gram_bpb_1b_100b",
        "statement": "For the reported 1B model trained on 100B bytes, adding 3-to-8-gram hash embeddings reduced BPB from 0.850 to 0.826.",
        "claim_type": "causal",
        "direction": "beneficial",
        "reported": {"one_billion_100b_train_bpb": {"without_hash": 0.850, "with_3_to_8gram_hash": 0.826}},
        "locator": "Hash-embedding ablation in Table 8",
        "strength": "strong",
    },
):
    claim = make_claim(
        claim_id=spec["claim_id"],
        paper_id=paper.paper_id,
        statement=spec["statement"],
        claim_type=spec["claim_type"],
        applies_to="entropy_patched_byte_modeling",
        domain="large_scale_lm_pretraining",
        evidence_class="paper_faithful_atomic_endpoint",
        direction=spec["direction"],
        limitations=(
            "BLT changes tokenizer, sequence construction, topology, and effective compute.",
            "OPHIS tokenizer, data interface, and evaluation are frozen.",
            "The named scale and endpoint cannot be generalized to the local frame.",
        ),
        locator=spec["locator"],
        tags=("byte_level", "tokenization", "atomic_endpoint", "scope_mismatch"),
    )
    evidence = make_split_evidence(
        base=blt_base_evidence,
        evidence_id=f"evd_lit_{claim.claim_id.removeprefix('clm_')}",
        claim=claim,
        reported=spec["reported"],
        mechanism="Entropy patches and optional hash features affect the named endpoint under the paper's full byte-level system.",
        method="Primary-paper extraction of one scale, comparator, and endpoint.",
        relation="supports",
        strength=spec["strength"],
        assessment_limitations=(
            "This atomic result neither authorizes tokenizer changes nor predicts local BPB.",
        ),
        tags=tuple(claim.tags),
        scope_match="weak",
    )
    EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 6. Peri-LN -- ICML 2025.
orgs = ("NAVER Cloud", "KAIST", "NAVER AI Lab")
paper = make_paper(
    paper_id="pap_peri_ln_icml2025",
    title="Peri-LN: Revisiting Normalization Layer in the Transformer Architecture",
    authors=(
        "Jeonghoon Kim",
        "Byeongchan Lee",
        "Cheonbok Park",
        "Yeontaek Oh",
        "Beomjun Kim",
        "Taehwan Yoo",
        "Seongjin Shin",
        "Dongyoon Han",
        "Jinwoo Shin",
        "Kang Min Yoo",
    ),
    year=2025,
    venue_name="ICML 2025",
    peer_reviewed=True,
    primary="https://proceedings.mlr.press/v267/kim25u.html",
    archive="https://arxiv.org/abs/2502.02732",
    organizations=orgs,
    tags=(
        "normalization",
        "residual",
        "rmsnorm",
        "variance",
        "gradient",
        "partial_existing",
        "candidate_attention_arm",
        "primary_source",
    ),
    version="ICML 2025 PMLR version retrieved 2026-07-29",
    notes="No author repository was linked from the primary proceedings record.",
)
claim = make_claim(
    claim_id="clm_peri_ln_loss_400m_30b",
    paper_id=paper.paper_id,
    statement=(
        "Normalizing both input and output of each residual sublayer improved "
        "30B-token loss at 400M parameters from 3.43 to 3.34."
    ),
    claim_type="causal",
    applies_to="residual_sublayer_input_output_normalization",
    domain="transformer_lm_pretraining",
    evidence_class="paper_derived_missing_attention_arm",
    direction="beneficial",
    limitations=(
        "The paper's models and token budgets are much larger than OPHIS.",
        "Attention and MLP effects are generally bundled.",
        "OPHIS already applies the MLP-side form and uses pre-head normalization plus zero-initialized attention c_proj.",
    ),
    locator="400M row in Table 1",
    tags=("normalization", "peri_ln", "candidate", "paper_derived"),
)
evidence = make_evidence(
    evidence_id="evd_lit_peri_ln_loss_400m_30b",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(2, 5, 3, 2, 3, 3, 5),
    reported={
        "loss_400m": {"pre_ln": 3.43, "peri_ln": 3.34},
        "seed_design": "five-seed studies included",
    },
    mechanism="Output normalization damps large residual-branch activations and gradients, complementing input normalization.",
    method="Primary-paper placement/scale evidence and exact local normalization-path comparison.",
    code_ref="train.py MLP Peri-LN path; attention head normalization and zero-initialized c_proj",
    design="strong",
    replication="strong",
    scope_match="adequate",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="strong",
    assessment_limitations=(
        "Supports only the missing attention post-c_proj arm locally.",
        "RMS-normalizing a newly activated near-zero c_proj can defeat the intended gradual warm start and requires fast-fail diagnostics.",
    ),
    tags=("normalization", "candidate", "partial_existing"),
)
TRIPLES.append((paper, claim, evidence))

peri_base_evidence = evidence
for spec in (
    ("clm_peri_ln_loss_1p5b_30b", "At 1.5B parameters and 30B training tokens, Peri-LN reduced reported loss from 3.29 to 3.18.", {"loss_1p5b": {"pre_ln": 3.29, "peri_ln": 3.18}}, "1.5B row in Table 1"),
    ("clm_peri_ln_loss_3p2b_30b", "At 3.2B parameters and 30B training tokens, Peri-LN reduced reported loss from 3.20 to 3.11.", {"loss_3p2b": {"pre_ln": 3.20, "peri_ln": 3.11}}, "3.2B row in Table 1"),
    ("clm_peri_ln_full_placement_loss_2p91", "In the placement ablation, the full two-sided Peri-LN configuration attained the best reported loss, 2.91.", {"placement_ablation_full_peri_ln": 2.91}, "Placement ablation in Tables 5-6"),
):
    claim = make_claim(
        claim_id=spec[0],
        paper_id=paper.paper_id,
        statement=spec[1],
        claim_type="causal",
        applies_to="residual_sublayer_input_output_normalization",
        domain="transformer_lm_pretraining",
        evidence_class="paper_faithful_atomic_endpoint",
        direction="beneficial",
        limitations=(
            "The paper's models and token budgets are much larger than OPHIS.",
            "Attention and MLP effects are bundled.",
            "Post-cproj RMS normalization can amplify a near-zero local attention branch.",
        ),
        locator=spec[3],
        tags=("normalization", "peri_ln", "atomic_endpoint", "scope_mismatch"),
    )
    evidence = make_split_evidence(
        base=peri_base_evidence,
        evidence_id=f"evd_lit_{claim.claim_id.removeprefix('clm_')}",
        claim=claim,
        reported={**spec[2], "seed_design": "five-seed studies included"},
        mechanism="Output normalization damps the named residual-branch scale endpoint.",
        method="Primary-paper extraction of one model scale or placement endpoint.",
        relation="supports",
        strength="strong",
        assessment_limitations=(
            "The paper endpoint does not establish safety of the missing local attention arm.",
        ),
        tags=tuple(claim.tags),
    )
    EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 7. Spectral Conditioning of Attention -- NeurIPS 2025.
orgs = (
    "Australian Institute for Machine Learning",
    "Adelaide University",
)
paper = make_paper(
    paper_id="pap_spectral_condition_attention_neurips2025",
    title="Spectral Conditioning of Attention Improves Transformer Performance",
    authors=("Hemanth Saratchandran", "Simon Lucey"),
    year=2025,
    venue_name="NeurIPS 2025",
    peer_reviewed=True,
    primary="https://proceedings.neurips.cc/paper_files/paper/2025/hash/5551e21f0ffb5955befb1209534354c6-Abstract-Conference.html",
    archive="https://arxiv.org/abs/2603.07162",
    organizations=orgs,
    tags=(
        "attention",
        "spectral_conditioning",
        "qkv",
        "jacobian",
        "fixed_correction",
        "backup",
        "primary_source",
    ),
    version="NeurIPS 2025 proceedings version retrieved 2026-07-29",
    notes="No author code repository was located from the primary record.",
)
claim = make_claim(
    claim_id="clm_spectral_conditioning_bert_glue_average",
    paper_id=paper.paper_id,
    statement=(
        "Adding fixed rectangular lambda-I corrections to Q/K/V with lambda=10 "
        "raised five-seed 110M BERT GLUE average from 78.6 to 79.4."
    ),
    claim_type="causal",
    applies_to="persistent_qkv_spectral_correction",
    domain="transformer_training",
    evidence_class="paper_faithful_backup",
    direction="beneficial",
    limitations=(
        "Language evidence is masked-LM fine-tuning rather than causal-LM BPB.",
        "The theory optimizes an upper bound rather than the exact attention-Jacobian condition number.",
        "A persistent lambda=10 correction interacts with OPHIS QK normalization and much smaller native Q/K/V scales.",
    ),
    locator="110M BERT GLUE row in Tables 3-5",
    tags=("attention", "spectral_conditioning", "backup", "paper_faithful"),
)
evidence = make_evidence(
    evidence_id="evd_lit_spectral_conditioning_bert_glue_average",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(3, 5, 3, 2, 2, 3, 5),
    reported={
        "bert_glue_average": {"baseline": 78.6, "conditioned": 79.4},
        "fixed_lambda": 10,
        "replication": "five trials/seeds in reported language and LRA tables",
    },
    mechanism="Fixed rectangular corrections improve a surrogate conditioning bound for Q/K/V transformations.",
    method="Primary-paper evidence plus QK-normalization and scale-interaction assessment.",
    code_ref="train.py Q/K RMS normalization and qkv projection; c_proj zero initialization",
    design="adequate",
    replication="strong",
    scope_match="weak",
    directness="adequate",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "A backup to the cleaner initialization-only paper, not a first-line local run.",
    ),
    tags=("attention", "spectral_conditioning", "backup"),
)
TRIPLES.append((paper, claim, evidence))

spectral_base_evidence = evidence
for spec in (
    {
        "claim_id": "clm_spectral_conditioning_all_reported_lra_tasks",
        "statement": "With lambda=10, fixed rectangular Q/K/V corrections improved every LRA task in the paper's five-trial comparison table.",
        "reported": {"all_reported_lra_tasks_improved": True, "fixed_lambda": 10, "replication": "five trials"},
        "direction": "beneficial",
        "locator": "LRA Table 7",
    },
    {
        "claim_id": "clm_spectral_rectangular_runtime_vs_baseline",
        "statement": "In the reported ViT timing, rectangular correction changed runtime from 29:29 to 29:33.",
        "reported": {"vit_runtime": {"baseline": "29:29", "rectangular_correction": "29:33"}},
        "direction": "harmful",
        "locator": "ViT runtime table",
    },
    {
        "claim_id": "clm_spectral_rectangular_runtime_vs_exact_svd",
        "statement": "In the reported ViT timing, rectangular correction ran in 29:33 versus 41:38 for exact SVD conditioning.",
        "reported": {"vit_runtime": {"rectangular_correction": "29:33", "exact_svd": "41:38"}},
        "direction": "beneficial",
        "locator": "ViT runtime table",
    },
):
    claim = make_claim(
        claim_id=spec["claim_id"],
        paper_id=paper.paper_id,
        statement=spec["statement"],
        claim_type="causal",
        applies_to="persistent_qkv_spectral_correction",
        domain="transformer_training",
        evidence_class="paper_faithful_atomic_endpoint",
        direction=spec["direction"],
        limitations=(
            "The task/model differs from causal-LM BPB.",
            "The theory controls a surrogate bound, not the exact Jacobian condition number.",
            "Local QK normalization and projection scales differ.",
        ),
        locator=spec["locator"],
        tags=("attention", "spectral_conditioning", "atomic_endpoint", "scope_mismatch"),
    )
    evidence = make_split_evidence(
        base=spectral_base_evidence,
        evidence_id=f"evd_lit_{claim.claim_id.removeprefix('clm_')}",
        claim=claim,
        reported=spec["reported"],
        mechanism="Fixed rectangular Q/K/V corrections change the named quality or runtime endpoint.",
        method="Primary-paper extraction of one task family or runtime comparison.",
        relation="supports",
        strength="moderate",
        assessment_limitations=(
            "This endpoint is a backup mechanism result, not local causal evidence.",
        ),
        tags=tuple(claim.tags),
        scope_match="weak",
    )
    EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 8. Memory Layers at Scale -- ICML 2025.
orgs = ("Meta Fundamental AI Research",)
paper = make_paper(
    paper_id="pap_memory_layers_scale_icml2025",
    title="Memory Layers at Scale",
    authors=(
        "Vincent-Pierre Berges",
        "Barlas Oğuz",
        "Daniel Haziza",
        "Wen-tau Yih",
        "Luke Zettlemoyer",
        "Gargi Ghosh",
    ),
    year=2025,
    venue_name="ICML 2025",
    peer_reviewed=True,
    primary="https://proceedings.mlr.press/v267/berges25a.html",
    code="https://github.com/facebookresearch/memory",
    organizations=orgs,
    tags=(
        "memory",
        "product_key",
        "sparse_lookup",
        "factuality",
        "shared_memory",
        "memory_plus",
        "bootstrap_gap",
        "exclude_local",
        "primary_source",
    ),
    version="ICML 2025 PMLR version retrieved 2026-07-29",
    notes="Official code linked from PMLR.",
)
claim = make_claim(
    claim_id="clm_memory_plus_mmlu_8b_200b",
    paper_id=paper.paper_id,
    statement=(
        "Shared product-key Memory+ layers with SwiLU gating materially improved "
        "MMLU for an 8B model at 200B training tokens from 41.35 to 50.14."
    ),
    claim_type="causal",
    applies_to="shared_product_key_memory_plus",
    domain="large_scale_transformer_lm_pretraining",
    evidence_class="paper_faithful_bootstrap_mismatch",
    direction="beneficial",
    limitations=(
        "Reported evidence begins at 200B tokens and up to 64B memory slots.",
        "Factual QA gains need not transfer to validation BPB.",
        "Efficient behavior depends on a custom embedding-bag path reported about 6x faster than plain PyTorch.",
        "A learned router cannot plausibly bootstrap in the 300-second local run.",
    ),
    locator="8B/200B MMLU row in Tables 1-4",
    tags=("memory", "product_key", "bootstrap_gap", "exclude_local"),
)
evidence = make_evidence(
    evidence_id="evd_lit_memory_plus_mmlu_8b_200b",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(2, 5, 3, 1, 4, 1, 3),
    reported={
        "8b_200b_mmlu": {"baseline": 41.35, "memory_plus": 50.14},
    },
    mechanism="Sparse product-key lookup offloads factual associations into shared, gated, full-dimensional external memory.",
    method="Primary-paper ablations plus data-bootstrap and kernel-dependency transfer assessment.",
    code_ref="Prior OPHIS product-key attempt was not faithful Memory+; no matching custom kernel",
    design="strong",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Do not mislabel the prior local attempt as a faithful replication.",
        "The faithful mechanism remains infeasible in the short-budget campaign.",
    ),
    tags=("memory", "bootstrap_gap", "negative_selection_evidence"),
)
TRIPLES.append((paper, claim, evidence))

memory_base_evidence = evidence
for spec in (
    ("clm_memory_plus_triviaqa_8b_200b", "For an 8B model at 200B training tokens, Memory+ improved TriviaQA from 51.74 to 57.64.", {"8b_200b_triviaqa": {"baseline": 51.74, "memory_plus": 57.64}}, "8B/200B TriviaQA row", "beneficial", "moderate"),
    ("clm_memory_plus_three_centered_placements", "The Memory+ placement ablation favored three centered shared memory placements.", {"preferred_shared_placements": 3}, "Placement ablation", "beneficial", "moderate"),
    ("clm_memory_plus_custom_embedding_bag_speedup", "The paper reports its custom embedding-bag path as approximately 6x faster than the plain PyTorch implementation.", {"custom_embedding_bag_speedup_vs_plain_pytorch": "approximately 6x"}, "Systems implementation comparison", "beneficial", "weak"),
):
    claim = make_claim(
        claim_id=spec[0],
        paper_id=paper.paper_id,
        statement=spec[1],
        claim_type="causal",
        applies_to="shared_product_key_memory_plus",
        domain="large_scale_transformer_lm_pretraining",
        evidence_class="paper_faithful_atomic_endpoint",
        direction=spec[4],
        limitations=(
            "Evidence begins at 200B tokens and uses much larger memory/model scales.",
            "The endpoint does not establish validation-BPB transfer.",
            "Efficient execution depends on a custom path absent locally.",
        ),
        locator=spec[3],
        tags=("memory", "product_key", "atomic_endpoint", "scope_mismatch"),
    )
    evidence = make_split_evidence(
        base=memory_base_evidence,
        evidence_id=f"evd_lit_{claim.claim_id.removeprefix('clm_')}",
        claim=claim,
        reported=spec[2],
        mechanism="Sparse product-key lookup and shared placement affect the named endpoint.",
        method="Primary-paper extraction of one benchmark, placement, or systems endpoint.",
        relation="supports",
        strength=spec[5],
        assessment_limitations=(
            "The atomic endpoint remains infeasible or scope-mismatched locally.",
        ),
        tags=tuple(claim.tags),
        scope_match="weak",
    )
    EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 9. Multi-Token Prediction Needs Registers -- NeurIPS 2025.
orgs = (
    "Archimedes, Athena Research Center",
    "valeo.ai",
    "University of Crete",
    "IACM-FORTH",
)
paper = make_paper(
    paper_id="pap_mutor_neurips2025",
    title="Multi-Token Prediction Needs Registers",
    authors=("Anastasios Gerontopoulos", "Spyridon Gidaris", "Nikos Komodakis"),
    year=2025,
    venue_name="NeurIPS 2025",
    peer_reviewed=True,
    primary="https://proceedings.neurips.cc/paper_files/paper/2025/hash/56cbb9d0efc05fb9d266adbf00f8eba7-Abstract-Conference.html",
    code="https://github.com/nasosger/MuToR",
    organizations=orgs,
    tags=(
        "multi_token_prediction",
        "register_tokens",
        "auxiliary_loss",
        "fine_tuning",
        "masking",
        "compute_overhead",
        "exclude_local",
        "primary_source",
    ),
    version="NeurIPS 2025 proceedings version retrieved 2026-07-29",
    notes="Official code linked from the paper.",
)
claim = make_claim(
    claim_id="clm_mutor_gemma2b_gsm8k",
    paper_id=paper.paper_id,
    statement=(
        "In math supervised fine-tuning, interleaved causal register tokens "
        "improved Gemma 2B GSM8K from 38.87 under NTP and 40.66 under conventional "
        "MTP to 42.10."
    ),
    claim_type="causal",
    applies_to="register_token_multi_token_prediction",
    domain="math_supervised_finetuning",
    evidence_class="paper_faithful_scope_mismatch",
    direction="beneficial",
    limitations=(
        "Language evidence is supervised fine-tuning accuracy, not causal-LM pretraining BPB.",
        "Register insertion changes FA3 varlen positions and document masks.",
        "The reported 1.4x wall-clock factor is costly under a fixed 300-second objective.",
    ),
    locator="Gemma 2B GSM8K row in Tables 1 and 3-4",
    tags=("multi_token_prediction", "register_tokens", "sft", "exclude_local"),
)
evidence = make_evidence(
    evidence_id="evd_lit_mutor_gemma2b_gsm8k",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(3, 5, 4, 1, 4, 1, 4),
    reported={
        "gemma2b_gsm8k": {"ntp": 38.87, "mtp": 40.66, "mutor": 42.10},
        "gemma_seed_count": 3,
    },
    mechanism="Random-offset causal register tokens provide dedicated representations for future-token auxiliary targets.",
    method="Primary-paper mask/ablation read and pretraining-objective plus FA3-packing scope analysis.",
    code_ref="train.py FA3 varlen document-packed batches; prior MTP removal under fixed wall time",
    design="adequate",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Does not support retrying generic MTP for local BPB.",
        "A differently masked stride-two auxiliary head would be a new unsupported hypothesis, not a MuToR replication.",
    ),
    tags=("multi_token_prediction", "scope_mismatch", "negative_selection_evidence"),
)
TRIPLES.append((paper, claim, evidence))

mutor_base_evidence = evidence
for spec in (
    ("clm_mutor_gemma2b_1m_gsm", "On Gemma 2B 1M-GSM, MuToR scored 68.33 versus 66.09 for NTP and 66.69 for conventional MTP.", {"gemma2b_1m_gsm": {"ntp": 66.09, "mtp": 66.69, "mutor": 68.33}, "gemma_seed_count": 3}, "beneficial", "Gemma 2B 1M-GSM row", "moderate"),
    ("clm_mutor_language_wall_clock_factor", "The paper reports approximately 1.4x language-training wall time for MuToR.", {"language_wall_clock_factor": 1.4}, "harmful", "Language-training resource report", "strong"),
):
    claim = make_claim(
        claim_id=spec[0],
        paper_id=paper.paper_id,
        statement=spec[1],
        claim_type="causal",
        applies_to="register_token_multi_token_prediction",
        domain="math_supervised_finetuning",
        evidence_class="paper_faithful_atomic_endpoint",
        direction=spec[3],
        limitations=(
            "The regime is supervised math fine-tuning, not causal-LM BPB pretraining.",
            "Register insertion changes FA3 positions and masks.",
            "The reported compute cost is material under a fixed wall-clock objective.",
        ),
        locator=spec[4],
        tags=("multi_token_prediction", "register_tokens", "atomic_endpoint", "scope_mismatch"),
    )
    evidence = make_split_evidence(
        base=mutor_base_evidence,
        evidence_id=f"evd_lit_{claim.claim_id.removeprefix('clm_')}",
        claim=claim,
        reported=spec[2],
        mechanism="Causal register tokens change the named quality or wall-clock endpoint.",
        method="Primary-paper extraction of one benchmark or runtime endpoint.",
        relation="supports",
        strength=spec[5],
        assessment_limitations=(
            "The endpoint does not support a generic local MTP retry.",
        ),
        tags=tuple(claim.tags),
        scope_match="weak",
    )
    EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 10. RoPE to NoPE and Back Again -- NeurIPS 2025.
orgs = ("Cohere",)
paper = make_paper(
    paper_id="pap_rnope_swa_neurips2025",
    title="RoPE to NoPE and Back Again: A New Hybrid Attention Strategy",
    authors=(
        "Bowen Yang",
        "Bharat Venkitesh",
        "Dwarak Talupuru",
        "Hangyu Lin",
        "David Cairuz",
        "Phil Blunsom",
        "Acyr Locatelli",
    ),
    year=2025,
    venue_name="NeurIPS 2025",
    peer_reviewed=True,
    primary="https://proceedings.neurips.cc/paper_files/paper/2025/hash/5c9ab393551b7a39b4c02d88fe5e7e69-Abstract-Conference.html",
    archive="https://arxiv.org/abs/2501.18795",
    organizations=orgs,
    tags=(
        "position_encoding",
        "rope",
        "nope",
        "sliding_window",
        "qknorm",
        "long_context",
        "hybrid",
        "candidate_bundle",
        "primary_source",
    ),
    version="NeurIPS 2025 proceedings version retrieved 2026-07-29",
    notes="No author code repository was located from the primary paper.",
)
claim = make_claim(
    claim_id="clm_rnope_128k_needle",
    paper_id=paper.paper_id,
    statement=(
        "In the paper's bundled hybrid, one full NoPE layer per three local RoPE "
        "layers, RoPE base 10K, and no QK normalization improved 128K needle score "
        "from 7.40 to 9.56."
    ),
    claim_type="causal",
    applies_to="hybrid_rope_nope_attention",
    domain="long_context_transformer_lm",
    evidence_class="paper_faithful_bundle",
    direction="beneficial",
    limitations=(
        "The mechanism is a bundle: full-layer NoPE, local-layer base 10K, removal of QK normalization, and 4096 local windows.",
        "Its strongest evidence is at 128K-256K, while OPHIS uses length 2048.",
        "The paper's own short-loss comparison makes pure NoPE worse.",
        "No seed intervals are reported.",
    ),
    locator="128K needle row in Tables 2-9",
    tags=("position_encoding", "rope", "nope", "bundle", "long_context"),
)
evidence = make_evidence(
    evidence_id="evd_lit_rnope_128k_needle",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(2, 5, 3, 1, 2, 4, 4),
    reported={
        "128k_needle": {"rope": 7.40, "rnope_swa": 9.56},
    },
    mechanism="Local RoPE supplies positional discrimination while periodic full NoPE layers preserve flexible global retrieval.",
    method="Primary-paper bundle and ablation read with exact local topology/base/QK-normalization comparison.",
    code_ref="train.py SSSL 1:3 topology, RoPE base 1e6, QK RMS normalization, sequence length 2048",
    design="adequate",
    replication="weak",
    scope_match="weak",
    directness="adequate",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "An atomic removal of RoPE on full layers is a local extrapolation, not a faithful replication.",
        "Reserve paper_faithful for the complete base/QKNorm/topology bundle.",
    ),
    tags=("position_encoding", "long_context", "scope_mismatch", "candidate_bundle"),
)
TRIPLES.append((paper, claim, evidence))

rnope_base_evidence = evidence
for spec in (
    ("clm_rnope_qknorm_validation_loss", "In the paper's reported short-loss comparison, adding QK normalization to RoPE changed validation loss from 1.52 to 1.53.", {"validation_loss": {"rope": 1.52, "qknorm_rope": 1.53}}, "harmful", "QK-normalization loss ablation"),
    ("clm_rnope_pure_nope_validation_loss", "In the paper's reported short-loss comparison, pure NoPE changed validation loss from 1.52 for RoPE to 1.58.", {"validation_loss": {"rope": 1.52, "nope": 1.58}}, "harmful", "Pure-NoPE loss ablation"),
    ("clm_rnope_256k_ruler_retrieval", "At 256K context, the hybrid improved reported RULER retrieval from 57.1 to 74.8.", {"256k_ruler_retrieval": {"baseline": 57.1, "hybrid": 74.8}}, "beneficial", "256K RULER row"),
    ("clm_rnope_training_speedup_64k", "At 64K context, the paper reports approximately 50 percent training speedup for the hybrid sparse topology.", {"training_speedup_64k": "approximately 50 percent"}, "beneficial", "64K training-speed comparison"),
    ("clm_rnope_training_speedup_128k", "At 128K context, the paper reports approximately 2x training speedup for the hybrid sparse topology.", {"training_speedup_128k": "approximately 2x"}, "beneficial", "128K training-speed comparison"),
):
    claim = make_claim(
        claim_id=spec[0],
        paper_id=paper.paper_id,
        statement=spec[1],
        claim_type="causal",
        applies_to="hybrid_rope_nope_attention",
        domain="long_context_transformer_lm",
        evidence_class="paper_faithful_atomic_endpoint",
        direction=spec[3],
        limitations=(
            "The paper's hybrid is a bundled topology/base/QKNorm/window intervention.",
            "The strongest measurements are at 64K-256K versus local length 2048.",
            "No seed intervals are reported.",
        ),
        locator=spec[4],
        tags=("position_encoding", "long_context", "atomic_endpoint", "scope_mismatch"),
    )
    evidence = make_split_evidence(
        base=rnope_base_evidence,
        evidence_id=f"evd_lit_{claim.claim_id.removeprefix('clm_')}",
        claim=claim,
        reported=spec[2],
        mechanism="The hybrid positional/topology bundle changes the named endpoint.",
        method="Primary-paper extraction of one loss, retrieval, or speed endpoint.",
        relation="supports",
        strength="moderate",
        assessment_limitations=(
            "The endpoint cannot isolate a single local RoPE/NoPE change.",
        ),
        tags=tuple(claim.tags),
        scope_match="weak",
    )
    EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 11. Differential Transformer -- ICLR 2025.
orgs = ("Microsoft Research", "Tsinghua University")
paper = make_paper(
    paper_id="pap_diff_transformer_iclr2025",
    title="Differential Transformer",
    authors=(
        "Tianzhu Ye",
        "Li Dong",
        "Yuqing Xia",
        "Yutao Sun",
        "Yi Zhu",
        "Gao Huang",
        "Furu Wei",
    ),
    year=2025,
    venue_name="ICLR 2025 Oral",
    peer_reviewed=True,
    primary="https://openreview.net/forum?id=OvoCm1gGhN",
    code="https://github.com/microsoft/unilm/tree/master/Diff-Transformer",
    organizations=orgs,
    tags=(
        "attention",
        "differential",
        "noise_cancellation",
        "negative_weights",
        "groupnorm",
        "custom_kernel",
        "defer",
        "primary_source",
    ),
    version="ICLR 2025 proceedings version retrieved 2026-07-29",
    notes="Official code is in Microsoft UniLM.",
)
claim = make_claim(
    claim_id="clm_diff_transformer_loss_1p4b",
    paper_id=paper.paper_id,
    statement=(
        "Subtracting a learned second softmax attention map improved 1.4B loss "
        "from 3.087 to 3.062."
    ),
    claim_type="causal",
    applies_to="differential_softmax_attention",
    domain="transformer_lm_pretraining",
    evidence_class="paper_faithful_kernel_dependent",
    direction="beneficial",
    limitations=(
        "The full mechanism needs two attention maps or a dedicated fused kernel.",
        "OPHIS already has per-head output normalization, overlapping the GroupNorm ingredient.",
        "Even the paper's optimized kernel has a 6-12 percent throughput penalty.",
    ),
    locator="1.4B loss row in Table 6",
    tags=("attention", "differential", "throughput", "defer"),
)
evidence = make_evidence(
    evidence_id="evd_lit_diff_transformer_loss_1p4b",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(2, 5, 3, 1, 3, 2, 5),
    reported={
        "1p4b_loss": {"baseline": 3.087, "differential": 3.062},
    },
    mechanism="A learned difference of two softmax maps cancels common-mode attention noise; GroupNorm stabilizes the signed output.",
    method="Primary-paper mechanism, normalization ablation, and custom-kernel throughput read.",
    code_ref="train.py single FA3 call and existing per-head attention-output normalization",
    design="adequate",
    replication="adequate",
    scope_match="adequate",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Do not run an unfused two-FA3-call reconstruction under a fixed wall-clock objective.",
    ),
    tags=("attention", "kernel_dependency", "negative_selection_evidence"),
)
TRIPLES.append((paper, claim, evidence))

diff_base_evidence = evidence
for spec in (
    ("clm_diff_groupnorm_ablation_loss_1p4b", "Removing Differential Transformer's GroupNorm worsened reported 1.4B loss from 3.062 to 3.122.", {"1p4b_loss": {"differential": 3.062, "without_groupnorm": 3.122}}, "harmful", "GroupNorm ablation in Table 6", "strong"),
    ("clm_diff_matched_head_control_near_baseline", "The paper's matched-head controls remained near the conventional Transformer baseline.", {"matched_head_control": "reported near baseline"}, "null", "Matched-head control in Table 6", "moderate"),
    ("clm_diff_h100_kernel_throughput_penalty", "The custom H100 Differential Transformer kernel retained a reported throughput penalty of approximately 6-12 percent.", {"custom_h100_throughput_penalty": "6 to 12 percent across reported settings"}, "harmful", "H100 throughput Table 7", "strong"),
    ("clm_diff_6p8b_matches_11b_scaling", "The paper reports a 6.8B Differential Transformer matching an 11B conventional Transformer on its scaling comparison.", {"scaling_result": "6.8B differential model reported to match an 11B conventional Transformer"}, "beneficial", "Scaling comparison", "moderate"),
):
    claim = make_claim(
        claim_id=spec[0],
        paper_id=paper.paper_id,
        statement=spec[1],
        claim_type="negative_result" if spec[3] == "null" else "causal",
        applies_to="differential_softmax_attention",
        domain="transformer_lm_pretraining",
        evidence_class="paper_faithful_atomic_endpoint",
        direction=spec[3],
        limitations=(
            "The full mechanism requires two attention maps or a fused kernel.",
            "Local per-head normalization overlaps the paper's GroupNorm.",
            "The model scale and H100 implementation differ from OPHIS.",
        ),
        locator=spec[4],
        tags=("attention", "differential", "atomic_endpoint", "kernel_dependency"),
    )
    evidence = make_split_evidence(
        base=diff_base_evidence,
        evidence_id=f"evd_lit_{claim.claim_id.removeprefix('clm_')}",
        claim=claim,
        reported=spec[2],
        mechanism="Signed dual-softmax attention and its stabilizing normalization affect the named endpoint.",
        method="Primary-paper extraction of one ablation, throughput, or scaling endpoint.",
        relation="supports",
        strength=spec[5],
        assessment_limitations=(
            "The endpoint does not justify an unfused local reconstruction.",
        ),
        tags=tuple(claim.tags),
        scope_match="weak",
    )
    EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 12. MTP curriculum -- ACL 2025 direct negative evidence.
orgs = ("Humboldt-Universität zu Berlin",)
paper = make_paper(
    paper_id="pap_mtp_curriculum_acl2025",
    title="Pre-Training Curriculum for Multi-Token Prediction in Language Models",
    authors=("Ansar Aynetdinov", "Alan Akbik"),
    year=2025,
    venue_name="ACL 2025 Long Paper",
    peer_reviewed=True,
    primary="https://aclanthology.org/2025.acl-long.1243/",
    code="https://github.com/aynetdia/mtp_curriculum",
    organizations=orgs,
    tags=(
        "multi_token_prediction",
        "curriculum",
        "small_lm",
        "negative_evidence",
        "do_not_retry",
        "primary_source",
    ),
    version="ACL 2025 proceedings version retrieved 2026-07-29",
    notes="FAU is acknowledged for compute support; official code linked from the paper.",
)
claim = make_claim(
    claim_id="clm_mtp_static_minipile_perplexity",
    paper_id=paper.paper_id,
    statement=(
        "For 1.3B subword models trained on 10B FineWeb tokens, static four-layer "
        "MTP worsened MiniPile perplexity from 47 for NTP to 57."
    ),
    claim_type="negative_result",
    applies_to="subword_multi_token_prediction_curriculum",
    domain="transformer_lm_pretraining",
    evidence_class="paper_faithful_negative",
    direction="harmful",
    limitations=(
        "The evaluated schedules and auxiliary heads are only a subset of possible MTP designs.",
        "Ten billion tokens still greatly exceeds the OPHIS short-budget regime.",
    ),
    locator="Static-MTP MiniPile cell in Table 5",
    tags=("multi_token_prediction", "negative_result", "do_not_retry"),
)
evidence = make_evidence(
    evidence_id="evd_lit_mtp_static_minipile_perplexity",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(1, 5, 5, 1, 5, 2, 5),
    reported={
        "minipile_perplexity": {"ntp": 47, "static_mtp": 57},
    },
    mechanism="Moving between next-token and multi-token objectives does not recover the main-token optimization lost to auxiliary prediction in this tested small-LM regime.",
    method="Primary-paper negative-result extraction and comparison to the locally removed wall-clock-costly MTP objective.",
    code_ref="Prior OPHIS MTP removal; fixed 300-second useful-optimization-step objective",
    design="strong",
    replication="adequate",
    scope_match="adequate",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="strong",
    assessment_limitations=(
        "Direct evidence against retrying generic MTP.",
        "A different future auxiliary head is a new hypothesis and cannot inherit positive support from this negative paper.",
    ),
    tags=("multi_token_prediction", "negative_result", "high_relevance"),
)
TRIPLES.append((paper, claim, evidence))

mtp_negative_base_evidence = evidence
for spec in (
    ("clm_mtp_static_lambada_perplexity", "Static four-layer MTP worsened LAMBADA perplexity from 74 for NTP to 100.", {"lambada_perplexity": {"ntp": 74, "static_mtp": 100}}, "Static-MTP LAMBADA cell"),
    ("clm_mtp_forward_minipile_perplexity", "The forward MTP curriculum worsened MiniPile perplexity from 47 for NTP to 57.", {"minipile_perplexity": {"ntp": 47, "forward_curriculum": 57}}, "Forward-curriculum MiniPile cell"),
    ("clm_mtp_forward_lambada_perplexity", "The forward MTP curriculum worsened LAMBADA perplexity from 74 for NTP to 101.", {"lambada_perplexity": {"ntp": 74, "forward_curriculum": 101}}, "Forward-curriculum LAMBADA cell"),
    ("clm_mtp_reverse_minipile_perplexity", "The reverse MTP curriculum worsened MiniPile perplexity from 47 for NTP to 50.", {"minipile_perplexity": {"ntp": 47, "reverse_curriculum": 50}}, "Reverse-curriculum MiniPile cell"),
    ("clm_mtp_reverse_lambada_perplexity", "The reverse MTP curriculum worsened LAMBADA perplexity from 74 for NTP to 75.", {"lambada_perplexity": {"ntp": 74, "reverse_curriculum": 75}}, "Reverse-curriculum LAMBADA cell"),
):
    claim = make_claim(
        claim_id=spec[0],
        paper_id=paper.paper_id,
        statement=(
            "For 1.3B subword models trained on 10B FineWeb tokens, " + spec[1]
        ),
        claim_type="negative_result",
        applies_to="subword_multi_token_prediction_curriculum",
        domain="transformer_lm_pretraining",
        evidence_class="paper_faithful_negative_atomic_endpoint",
        direction="harmful",
        limitations=(
            "Only the evaluated schedules and auxiliary heads are covered.",
            "Ten billion tokens exceeds the local short-budget regime.",
        ),
        locator=f"Table 5, {spec[3]}",
        tags=("multi_token_prediction", "negative_result", "atomic_endpoint"),
    )
    evidence = make_split_evidence(
        base=mtp_negative_base_evidence,
        evidence_id=f"evd_lit_{claim.claim_id.removeprefix('clm_')}",
        claim=claim,
        reported=spec[2],
        mechanism="The tested MTP objective/schedule loses next-token optimization on the named dataset endpoint.",
        method="Primary-paper extraction of one schedule-by-dataset perplexity cell.",
        relation="supports",
        strength="strong",
        assessment_limitations=(
            "Direct negative evidence for this exact schedule and endpoint only.",
        ),
        tags=tuple(claim.tags),
    )
    EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 13. KernelBench-Verified -- arXiv 2026, adversarial evaluation evidence.
orgs = ("Meta FAIR", "Stanford University")
paper = make_paper(
    paper_id="pap_kernelbench_verified_2026",
    title="KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?",
    authors=(
        "Yunxiang Zhang",
        "Ping Yu",
        "Jianyu Wang",
        "Max (Xiangjun) Fan",
        "Julian Reed",
        "Azalia Mirhoseini",
        "Will Su",
    ),
    year=2026,
    venue_name="arXiv 2607.16241",
    peer_reviewed=False,
    primary="https://arxiv.org/abs/2607.16241",
    code="https://github.com/facebookresearch/kernel_bench_verified",
    organizations=orgs,
    tags=(
        "gpu_kernel",
        "benchmark",
        "correctness",
        "reward_hacking",
        "tf32",
        "hidden_tests",
        "negative_evidence",
        "primary_source",
    ),
    version="arXiv 2607.16241 version retrieved 2026-07-29",
    notes="Official code from Meta Research.",
)
claim = make_claim(
    claim_id="clm_kernelbench_verified_geomean_speedup",
    paper_id=paper.paper_id,
    statement=(
        "Under the verified protocol, the best single-turn frontier model's "
        "geometric-mean speedup was 0.88x rather than the 1.43x reported under "
        "the standard KernelBench protocol."
    ),
    claim_type="negative_result",
    applies_to="llm_generated_gpu_kernel_evaluation",
    domain="gpu_kernel_benchmarking",
    evidence_class="primary_adversarial_benchmark",
    direction="harmful",
    limitations=(
        "The report is a 2026 arXiv preprint rather than peer-reviewed proceedings.",
        "Single-turn results do not characterize a governed iterative agent with profiling feedback.",
        "Benchmark kernels and hardware do not directly equal the OPHIS H200 training bottleneck.",
    ),
    locator="Abstract; verified single-turn geometric-mean speedup result",
    tags=("gpu_kernel", "reward_hacking", "negative_result", "evaluation_integrity"),
)
evidence = make_evidence(
    evidence_id="evd_lit_kernelbench_verified_geomean_speedup",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=systems_rating(paper.paper_id),
    reported={
        "standard_geometric_mean_speedup": 1.43,
        "verified_geometric_mean_speedup": 0.88,
    },
    mechanism="Realistic TF32 baselines and distribution-shifted hidden tests expose hardcoded bypasses and false speedups.",
    method="Primary preprint/code read; benchmark-integrity implications for any saved/generated SOTA kernel.",
    code_ref="Any OPHIS generated kernel must preserve exact train.py semantics and pass held-out randomized equivalence before timing",
    design="strong",
    replication="adequate",
    scope_match="adequate",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="strong",
    assessment_limitations=(
        "Treat unverified benchmark speedups as hypotheses, not SOTA evidence.",
        "Local kernel acceptance needs randomized correctness, realistic baselines, memory reporting, and end-to-end training throughput.",
    ),
    tags=("gpu_kernel", "evaluation_integrity", "negative_evidence"),
)
TRIPLES.append((paper, claim, evidence))
kernelbench_verified_base_evidence = evidence

for spec in (
    {
        "claim_id": "clm_kernelbench_verified_no_consistent_pytorch_win",
        "statement": (
            "No tested model consistently beat the PyTorch baseline under the "
            "KernelBench-Verified protocol."
        ),
        "claim_type": "negative_result",
        "direction": "null",
        "locator": "Verified single-turn evaluation results",
        "reported": {"models_consistently_beating_pytorch": 0},
        "mechanism": (
            "Hidden correctness distributions and a realistic baseline remove "
            "apparent wins that do not generalize."
        ),
        "method": "Primary preprint extraction of the consistency endpoint only.",
        "relation": "supports",
        "strength": "strong",
        "tags": ("gpu_kernel", "negative_result", "evaluation_integrity"),
    },
    {
        "claim_id": "clm_kernelbench_verified_peak_memory_increase",
        "statement": (
            "Twenty-eight percent of the best tested model's generated kernels "
            "increased peak GPU memory."
        ),
        "claim_type": "negative_result",
        "direction": "harmful",
        "locator": "Memory-efficiency results",
        "reported": {"best_model_kernels_increasing_peak_memory_percent": 28},
        "mechanism": (
            "Latency-only optimization can trade away peak-memory efficiency."
        ),
        "method": "Primary preprint extraction of the peak-memory endpoint only.",
        "relation": "supports",
        "strength": "strong",
        "tags": ("gpu_kernel", "memory", "negative_result"),
    },
    {
        "claim_id": "clm_kernelbench_verified_hidden_distributions",
        "statement": (
            "KernelBench-Verified evaluates correctness on four hidden input "
            "distributions."
        ),
        "claim_type": "method_definition",
        "direction": "method_only",
        "locator": "Verified evaluation protocol",
        "reported": {"hidden_correctness_distributions": 4},
        "mechanism": (
            "Multiple hidden distributions test whether generated kernels remain "
            "correct beyond the visible examples."
        ),
        "method": "Primary preprint extraction of one protocol-design fact.",
        "relation": "supports",
        "strength": "strong",
        "tags": ("gpu_kernel", "hidden_tests", "methodology"),
    },
):
    split_claim = make_claim(
        claim_id=spec["claim_id"],
        paper_id=paper.paper_id,
        statement=spec["statement"],
        claim_type=spec["claim_type"],
        applies_to="llm_generated_gpu_kernel_evaluation",
        domain="gpu_kernel_benchmarking",
        evidence_class="primary_adversarial_benchmark",
        direction=spec["direction"],
        limitations=claim.limitations,
        locator=spec["locator"],
        tags=spec["tags"],
    )
    split_evidence = make_split_evidence(
        base=kernelbench_verified_base_evidence,
        evidence_id=f"evd_lit_{split_claim.claim_id.removeprefix('clm_')}",
        claim=split_claim,
        reported=spec["reported"],
        mechanism=spec["mechanism"],
        method=spec["method"],
        relation=spec["relation"],
        strength=spec["strength"],
        assessment_limitations=(
            "This record assesses only the stated endpoint or protocol fact.",
            "It does not establish a local OPHIS kernel result.",
        ),
        tags=spec["tags"],
    )
    EXTRA_CLAIM_EVIDENCE.append((split_claim, split_evidence))


# 14. FastKernels -- arXiv 2026, production-alignment negative evidence.
orgs = (
    "Snowflake AI Research",
    "Carnegie Mellon University",
    "University of California San Diego",
)
paper = make_paper(
    paper_id="pap_fastkernels_2026",
    title="FastKernels: Benchmarking GPU Kernel Generation in Production",
    authors=(
        "Gabriele Oliaro",
        "Yichao Fu",
        "May Jiang",
        "Owen Lu",
        "Junli Wang",
        "Zhihao Jia",
        "Hao Zhang",
        "Samyam Rajbhandari",
    ),
    year=2026,
    venue_name="arXiv 2605.23215",
    peer_reviewed=False,
    primary="https://arxiv.org/abs/2605.23215",
    code="https://github.com/Snowflake-AI-Research/fastkernels",
    organizations=orgs,
    tags=(
        "gpu_kernel",
        "benchmark",
        "production_alignment",
        "inference",
        "correctness",
        "negative_evidence",
        "primary_source",
    ),
    version="arXiv 2605.23215 version retrieved 2026-07-29",
    notes="Official project page: https://fastkernels.github.io/.",
)
claim = make_claim(
    claim_id="clm_fastkernels_representative_architectures",
    paper_id=paper.paper_id,
    statement=(
        "FastKernels defines its production-aligned benchmark from 46 "
        "representative architectures."
    ),
    claim_type="method_definition",
    applies_to="production_aligned_generated_kernels",
    domain="gpu_inference_kernel_benchmarking",
    evidence_class="primary_production_benchmark",
    direction="method_only",
    limitations=(
        "The benchmark targets production inference rather than OPHIS training.",
        "Aggregate results depend on the selected architectures and production baselines.",
        "The arXiv report is not yet peer reviewed.",
    ),
    locator="Benchmark construction",
    tags=("gpu_kernel", "production_alignment", "methodology"),
)
evidence = make_evidence(
    evidence_id="evd_lit_fastkernels_representative_architectures",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=systems_rating(paper.paper_id),
    reported={
        "representative_architectures": 46,
    },
    mechanism="Production interfaces, captured tensors, compilation stacks, and hardened baselines remove sandbox-only optimization rewards.",
    method="Primary preprint/code read; transfer from microbenchmark speed to end-to-end system behavior.",
    code_ref="OPHIS requires end-to-end steps/tokens/val_bpb under the scheduler, not isolated-kernel latency alone",
    design="strong",
    replication="adequate",
    scope_match="adequate",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="strong",
    assessment_limitations=(
        "Supports profiling and production-shaped integration tests before accepting generated kernel code.",
        "Does not itself identify a kernel that accelerates the local training frame.",
    ),
    tags=("gpu_kernel", "production_alignment", "negative_evidence"),
)
TRIPLES.append((paper, claim, evidence))
fastkernels_base_evidence = evidence

for spec in (
    {
        "claim_id": "clm_fastkernels_architecture_categories",
        "statement": (
            "FastKernels groups its representative architectures into eight "
            "benchmark categories."
        ),
        "claim_type": "method_definition",
        "direction": "method_only",
        "locator": "Benchmark construction",
        "reported": {"architecture_categories": 8},
        "mechanism": "Category stratification broadens the benchmark's production workload coverage.",
        "method": "Primary preprint extraction of one benchmark-composition fact.",
        "tags": ("gpu_kernel", "production_alignment", "methodology"),
    },
    {
        "claim_id": "clm_fastkernels_huggingface_coverage",
        "statement": (
            "FastKernels covers 409 of 425 HuggingFace Transformer "
            "architectures, reported as 96.2 percent coverage."
        ),
        "claim_type": "descriptive",
        "direction": "descriptive_only",
        "locator": "Benchmark coverage results",
        "reported": {
            "huggingface_architectures_covered": {
                "covered": 409,
                "total": 425,
                "percent": 96.2,
            }
        },
        "mechanism": "Representative production interfaces map many model variants onto shared kernel shapes.",
        "method": "Primary preprint extraction of the benchmark coverage endpoint only.",
        "tags": ("gpu_kernel", "production_alignment", "coverage"),
    },
    {
        "claim_id": "clm_fastkernels_strongest_agent_speedup",
        "statement": (
            "The strongest evaluated FastKernels agent achieved 0.94x aggregate "
            "speedup over the production baseline."
        ),
        "claim_type": "negative_result",
        "direction": "harmful",
        "locator": "Aggregate agent evaluation",
        "reported": {"agent_rank": 1, "aggregate_speedup": 0.94},
        "mechanism": "Production-shaped interfaces and hardened baselines remove sandbox-only gains.",
        "method": "Primary preprint extraction of the rank-one aggregate endpoint only.",
        "tags": ("gpu_kernel", "production_alignment", "negative_result"),
    },
    {
        "claim_id": "clm_fastkernels_second_agent_speedup",
        "statement": (
            "The second-ranked evaluated FastKernels agent achieved 0.78x "
            "aggregate speedup over the production baseline."
        ),
        "claim_type": "negative_result",
        "direction": "harmful",
        "locator": "Aggregate agent evaluation",
        "reported": {"agent_rank": 2, "aggregate_speedup": 0.78},
        "mechanism": "Production-shaped interfaces and hardened baselines remove sandbox-only gains.",
        "method": "Primary preprint extraction of the rank-two aggregate endpoint only.",
        "tags": ("gpu_kernel", "production_alignment", "negative_result"),
    },
    {
        "claim_id": "clm_fastkernels_third_agent_speedup",
        "statement": (
            "The third-ranked evaluated FastKernels agent achieved 0.53x "
            "aggregate speedup over the production baseline."
        ),
        "claim_type": "negative_result",
        "direction": "harmful",
        "locator": "Aggregate agent evaluation",
        "reported": {"agent_rank": 3, "aggregate_speedup": 0.53},
        "mechanism": "Production-shaped interfaces and hardened baselines remove sandbox-only gains.",
        "method": "Primary preprint extraction of the rank-three aggregate endpoint only.",
        "tags": ("gpu_kernel", "production_alignment", "negative_result"),
    },
):
    split_claim = make_claim(
        claim_id=spec["claim_id"],
        paper_id=paper.paper_id,
        statement=spec["statement"],
        claim_type=spec["claim_type"],
        applies_to="production_aligned_generated_kernels",
        domain="gpu_inference_kernel_benchmarking",
        evidence_class="primary_production_benchmark",
        direction=spec["direction"],
        limitations=claim.limitations,
        locator=spec["locator"],
        tags=spec["tags"],
    )
    split_evidence = make_split_evidence(
        base=fastkernels_base_evidence,
        evidence_id=f"evd_lit_{split_claim.claim_id.removeprefix('clm_')}",
        claim=split_claim,
        reported=spec["reported"],
        mechanism=spec["mechanism"],
        method=spec["method"],
        relation="supports",
        strength="strong",
        assessment_limitations=(
            "This record assesses only the stated benchmark fact or endpoint.",
            "It does not establish a local training-kernel result.",
        ),
        tags=spec["tags"],
    )
    EXTRA_CLAIM_EVIDENCE.append((split_claim, split_evidence))


# 15. SOL-ExecBench -- arXiv 2026, Blackwell-specific benchmark method.
orgs = ("NVIDIA",)
paper = make_paper(
    paper_id="pap_sol_execbench_2026",
    title="SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits",
    authors=(
        "Edward Lin",
        "Sahil Modi",
        "Siva Kumar Sastry Hari",
        "Qijing Huang",
        "Zhifan Ye",
        "Nestor Qin",
        "Fengzhe Zhou",
        "Yuan Zhang",
        "Jingquan Wang",
        "Sana Damani",
        "Dheeraj Peri",
        "Ouye Xie",
        "Aditya Kane",
        "Moshe Maor",
        "Michael Behar",
        "Triston Cao",
        "Rishabh Mehta",
        "Vartika Singh",
        "Vikram Sharma Mailthody",
        "Terry Chen",
        "Zihao Ye",
        "Hanfeng Chen",
        "Tianqi Chen",
        "Vinod Grover",
        "Wei Chen",
        "Wei Liu",
        "Eric Chung",
        "Luis Ceze",
        "Roger Bringmann",
        "Cyril Zeller",
        "Michael Lightstone",
        "Christos Kozyrakis",
        "Humphrey Shi",
    ),
    year=2026,
    venue_name="arXiv 2603.19173",
    peer_reviewed=False,
    primary="https://arxiv.org/abs/2603.19173",
    code="https://github.com/NVIDIA/SOL-ExecBench",
    organizations=orgs,
    tags=(
        "gpu_kernel",
        "benchmark",
        "speed_of_light",
        "blackwell",
        "reward_hacking",
        "hardware_specific",
        "primary_source",
    ),
    version="arXiv 2603.19173 version retrieved 2026-07-29",
    notes="SOLAR code is separately released at https://github.com/NVlabs/SOLAR.",
)
claim = make_claim(
    claim_id="clm_sol_execbench_problem_count",
    paper_id=paper.paper_id,
    statement=(
        "SOL-ExecBench defines 235 CUDA optimization problems."
    ),
    claim_type="method_definition",
    applies_to="hardware_bound_gpu_kernel_evaluation",
    domain="nvidia_blackwell_gpu_kernels",
    evidence_class="primary_benchmark_method_scope_mismatch",
    direction="method_only",
    limitations=(
        "The benchmark targets NVIDIA Blackwell and includes Blackwell-specific BF16, FP8, and NVFP4 workloads.",
        "OPHIS runs on Hopper H200, so the numerical SOL bounds and candidate kernels do not transfer.",
        "The arXiv report is not yet peer reviewed.",
    ),
    locator="Abstract; benchmark composition",
    tags=("gpu_kernel", "speed_of_light", "blackwell", "scope_mismatch"),
)
evidence = make_evidence(
    evidence_id="evd_lit_sol_execbench_problem_count",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=systems_rating(paper.paper_id),
    reported={
        "cuda_optimization_problems": 235,
    },
    mechanism="An analytical hardware bound supplies a stable target when software baselines evolve; isolation controls reduce timing and reward-hacking artifacts.",
    method="Primary preprint and NVIDIA benchmark-description read; hardware-generation scope analysis.",
    code_ref="H200/Hopper campaign; local acceptance should adapt the methodology, not copy Blackwell bounds",
    design="strong",
    replication="not_applicable",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="strong",
    assessment_limitations=(
        "Adopt the methodological controls conceptually, but do not treat any Blackwell SOL score as H200 performance evidence.",
    ),
    tags=("gpu_kernel", "hardware_specific", "scope_mismatch", "methodology"),
)
TRIPLES.append((paper, claim, evidence))
sol_base_evidence = evidence

for spec in (
    {
        "claim_id": "clm_sol_execbench_source_model_count",
        "statement": (
            "SOL-ExecBench sources its CUDA optimization problems from 124 "
            "production and emerging AI models."
        ),
        "reported": {"source_models": 124},
        "locator": "Abstract; benchmark composition",
        "mechanism": "A broad source-model pool supplies production-shaped kernel problems.",
        "method": "Primary preprint extraction of one benchmark-composition fact.",
        "tags": ("gpu_kernel", "benchmark", "methodology"),
    },
    {
        "claim_id": "clm_sol_execbench_sol_score_definition",
        "statement": (
            "SOL-ExecBench scores a candidate by the fraction of the performance "
            "gap it closes toward an analytically derived hardware speed-of-light bound."
        ),
        "reported": {
            "sol_score_definition": (
                "fraction of the baseline-to-analytical-bound performance gap closed"
            )
        },
        "locator": "SOL Score definition",
        "mechanism": "An analytical hardware bound remains stable when software baselines change.",
        "method": "Primary preprint extraction of the score definition only.",
        "tags": ("gpu_kernel", "speed_of_light", "methodology"),
    },
    {
        "claim_id": "clm_sol_execbench_blackwell_scope",
        "statement": "SOL-ExecBench targets NVIDIA Blackwell GPUs.",
        "reported": {"target_hardware": "NVIDIA Blackwell"},
        "locator": "Benchmark hardware scope",
        "mechanism": "Hardware-specific bounds condition every reported SOL score.",
        "method": "Primary preprint extraction of the target-hardware scope only.",
        "tags": ("gpu_kernel", "blackwell", "scope_mismatch"),
    },
    {
        "claim_id": "clm_sol_execbench_precision_scope",
        "statement": "SOL-ExecBench includes BF16, FP8, and NVFP4 workloads.",
        "reported": {"precisions": ["BF16", "FP8", "NVFP4"]},
        "locator": "Benchmark workload precision scope",
        "mechanism": "Precision-specific kernels exercise Blackwell's heterogeneous arithmetic paths.",
        "method": "Primary preprint extraction of the benchmark precision domain only.",
        "tags": ("gpu_kernel", "precision", "blackwell", "scope_mismatch"),
    },
    {
        "claim_id": "clm_sol_execbench_clock_locking",
        "statement": "SOL-ExecBench locks GPU clocks during timing.",
        "reported": {"timing_control": "GPU clock locking"},
        "locator": "Evaluation-harness robustness controls",
        "mechanism": "Clock locking reduces frequency variation across measurements.",
        "method": "Primary preprint extraction of one timing-integrity control.",
        "tags": ("gpu_kernel", "timing_integrity", "methodology"),
    },
    {
        "claim_id": "clm_sol_execbench_l2_clearing",
        "statement": "SOL-ExecBench clears the L2 cache between controlled timings.",
        "reported": {"timing_control": "L2 cache clearing"},
        "locator": "Evaluation-harness robustness controls",
        "mechanism": "L2 clearing reduces cache-residency artifacts between candidates.",
        "method": "Primary preprint extraction of one timing-integrity control.",
        "tags": ("gpu_kernel", "timing_integrity", "methodology"),
    },
    {
        "claim_id": "clm_sol_execbench_subprocess_isolation",
        "statement": "SOL-ExecBench times candidates in isolated subprocesses.",
        "reported": {"timing_control": "isolated subprocesses"},
        "locator": "Evaluation-harness robustness controls",
        "mechanism": "Process isolation limits state leakage between candidate evaluations.",
        "method": "Primary preprint extraction of one evaluation-isolation control.",
        "tags": ("gpu_kernel", "process_isolation", "methodology"),
    },
    {
        "claim_id": "clm_sol_execbench_static_analysis",
        "statement": "SOL-ExecBench applies static anti-reward-hacking checks.",
        "reported": {"integrity_control": "static anti-reward-hacking analysis"},
        "locator": "Evaluation-harness robustness controls",
        "mechanism": "Static checks reject recognizable benchmark-specific bypasses before timing.",
        "method": "Primary preprint extraction of one evaluation-integrity control.",
        "tags": ("gpu_kernel", "reward_hacking", "methodology"),
    },
):
    split_claim = make_claim(
        claim_id=spec["claim_id"],
        paper_id=paper.paper_id,
        statement=spec["statement"],
        claim_type="method_definition",
        applies_to="hardware_bound_gpu_kernel_evaluation",
        domain="nvidia_blackwell_gpu_kernels",
        evidence_class="primary_benchmark_method_scope_mismatch",
        direction="method_only",
        limitations=claim.limitations,
        locator=spec["locator"],
        tags=spec["tags"],
    )
    split_evidence = make_split_evidence(
        base=sol_base_evidence,
        evidence_id=f"evd_lit_{split_claim.claim_id.removeprefix('clm_')}",
        claim=split_claim,
        reported=spec["reported"],
        mechanism=spec["mechanism"],
        method=spec["method"],
        relation="supports",
        strength="strong",
        assessment_limitations=(
            "This record assesses only the stated benchmark-method fact.",
            "Blackwell bounds and kernels are not H200 performance evidence.",
        ),
        tags=spec["tags"],
    )
    EXTRA_CLAIM_EVIDENCE.append((split_claim, split_evidence))


# 16. Beyond Random Sampling -- EACL 2026, data-order evidence.
orgs = ("École Polytechnique", "Mohamed bin Zayed University of Artificial Intelligence")
paper = make_paper(
    paper_id="pap_beyond_random_sampling_2026",
    title="Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning",
    authors=("Yang Zhang", "Amr Mohamed", "Hadi Abdine", "Guokan Shang", "Michalis Vazirgiannis"),
    year=2026,
    venue_name="EACL 2026 Long Paper",
    peer_reviewed=True,
    primary="https://aclanthology.org/2026.eacl-long.271/",
    archive="https://arxiv.org/abs/2506.11300",
    organizations=orgs,
    tags=(
        "curriculum",
        "data_order",
        "pretraining",
        "convergence",
        "warmup",
        "frozen_scope",
        "primary_source",
    ),
    version="EACL 2026 proceedings version retrieved 2026-07-29",
    notes="No official code repository was located in the primary record.",
)
claim = make_claim(
    claim_id="clm_curriculum_token_count_pacing_29p8",
    paper_id=paper.paper_id,
    statement=(
        "For the paper's 0.5B model trained on 10B CulturaX tokens, quadratic "
        "pacing by Number of Tokens reached the random baseline's performance "
        "in 29.8 percent fewer training steps."
    ),
    claim_type="causal",
    applies_to="pretraining_data_curriculum",
    domain="transformer_lm_pretraining",
    evidence_class="paper_faithful_frozen_data_order",
    direction="beneficial",
    limitations=(
        "The strongest settings use far larger models and token budgets than OPHIS.",
        "This claim is confined to Number-of-Tokens quadratic pacing; compression interleaving and the shuffled-prefix control are separate claims.",
        "Number of Tokens is not the compression-ratio signal proposed for the local arm.",
        "A local data-order intervention requires a separately preregistered experiment preserving an exact fixed packed-sequence multiset and packing controls.",
    ),
    locator="Section 4.2, Number of Tokens quadratic-pacing result",
    tags=("curriculum", "data_order", "convergence", "frozen_scope"),
)
evidence = make_evidence(
    evidence_id="evd_lit_curriculum_token_count_pacing_29p8",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=systems_rating(paper.paper_id),
    reported={
        "model_parameters": 500_000_000,
        "training_tokens": 10_000_000_000,
        "difficulty_signal": "Number of Tokens",
        "pacing_function": "quadratic",
        "steps_to_baseline_reduction_percent": 29.8,
    },
    mechanism="Quadratic pacing progressively expands access from shorter/easier to longer/harder examples.",
    method="Peer-reviewed primary-paper extraction of one schedule, signal, model, budget, and endpoint.",
    code_ref="Any OPHIS test must freeze identical packed content and isolate order from sample selection and FA3 work",
    design="strong",
    replication="weak",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "This source percentage does not imply a local BPB magnitude.",
        "The evidence motivates a controlled order experiment but does not authorize one by itself.",
    ),
    tags=("curriculum", "data_order", "frozen_scope"),
)
TRIPLES.append((paper, claim, evidence))

claim = make_claim(
    claim_id="clm_curriculum_compression_interleave_41p7",
    paper_id=paper.paper_id,
    statement=(
        "For the paper's 0.5B, 10B-token interleaved curriculum, Compression "
        "Ratio reached the best random-baseline performance in 41.7 percent "
        "fewer steps and finished 2.2 percent higher on the paper's averaged "
        "zero-shot accuracy metric."
    ),
    claim_type="causal",
    applies_to="compression_ratio_interleaved_curriculum",
    domain="transformer_lm_pretraining",
    evidence_class="paper_faithful",
    direction="beneficial",
    limitations=(
        "The endpoint is averaged zero-shot benchmark accuracy, not validation BPB.",
        "The 0.5B model and 10B-token budget are much larger than OPHIS.",
        "Interleaving is not the same intervention as a 15-percent ordered prefix over fixed packs.",
    ),
    locator="Section 4.3, Compression Ratio interleaving result; Figure 4",
    tags=("curriculum", "compression_ratio", "interleave", "convergence"),
)
evidence = make_evidence(
    evidence_id="evd_lit_curriculum_compression_interleave_41p7",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=systems_rating(paper.paper_id),
    reported={
        "model_parameters": 500_000_000,
        "training_tokens": 10_000_000_000,
        "difficulty_signal": "Compression Ratio",
        "strategy": "interleaved curriculum",
        "steps_to_best_baseline_reduction_percent": 41.7,
        "final_average_accuracy_improvement_percent": 2.2,
    },
    mechanism="Interleaving retains difficulty diversity while preserving a progressive structure within each interleave.",
    method="Primary Section 4.3 and Figure 4 extraction; endpoint and schedule kept explicit.",
    code_ref="Local proposal uses a fixed compression-ordered prefix, not paper-faithful interleaving",
    design="adequate",
    replication="strong",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Direct evidence for compression-ratio ordering, but only indirect support for the proposed local schedule and BPB endpoint.",
    ),
    tags=("curriculum", "compression_ratio", "interleave", "scope_mismatch"),
)
EXTRA_CLAIM_EVIDENCE.append((claim, evidence))

claim = make_claim(
    claim_id="clm_curriculum_shuffled_prefix_control",
    paper_id=paper.paper_id,
    statement=(
        "For three early-converging curriculum settings, retraining on the same "
        "selected prefix after random shuffling performed worse than retaining "
        "the curriculum order, separating an ordering contribution from the "
        "prefix's filtering contribution."
    ),
    claim_type="causal",
    applies_to="curriculum_order_vs_selected_subset",
    domain="transformer_lm_pretraining",
    evidence_class="paper_faithful_control",
    direction="beneficial",
    limitations=(
        "The control covers MTLD vanilla, Number-of-Tokens vanilla, and Compression-Ratio linear pacing only.",
        "The paper reports the controlled result graphically/qualitatively rather than as a single universal effect size.",
        "Subset selection and stopping at the best checkpoint remain design choices that limit transfer.",
    ),
    locator="Section 5.3; Appendix E, Figure 10",
    tags=("curriculum", "data_order", "shuffled_control", "causal_control"),
)
evidence = make_evidence(
    evidence_id="evd_lit_curriculum_shuffled_prefix_control",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=systems_rating(paper.paper_id),
    reported={
        "controlled_settings": [
            "MTLD vanilla curriculum",
            "Number of Tokens vanilla curriculum",
            "Compression Ratio linear pacing",
        ],
        "comparison": "ordered selected prefix versus the same prefix randomly shuffled",
        "result": "ordered curriculum outperformed shuffled same-prefix control",
        "single_numeric_effect_reported": False,
    },
    mechanism="Holding the selected subset fixed leaves presentation order as the manipulated factor.",
    method="Primary Section 5.3 and Appendix E controlled-comparison extraction.",
    code_ref="Local design should freeze pack identities, tokens, boundary counts, and FA3 worktile strata before changing order",
    design="adequate",
    replication="weak",
    scope_match="adequate",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Supports a shuffled same-prefix causal control; it does not supply a local effect-size prior.",
    ),
    tags=("curriculum", "data_order", "shuffled_control", "causal_control"),
)
EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 17. KernelBench -- ICML 2025, benchmark followed by a 2026 validity correction.
orgs = ("Stanford University", "Princeton University")
paper = make_paper(
    paper_id="pap_kernelbench_2025",
    title="KernelBench: Can LLMs Write Efficient GPU Kernels?",
    authors=(
        "Anne Ouyang",
        "Simon Guo",
        "Simran Arora",
        "Alex L. Zhang",
        "William Hu",
        "Christopher Ré",
        "Azalia Mirhoseini",
    ),
    year=2025,
    venue_name="ICML 2025",
    peer_reviewed=True,
    primary="https://proceedings.mlr.press/v267/ouyang25a.html",
    archive="https://arxiv.org/abs/2502.10517",
    code="https://github.com/ScalingIntelligence/KernelBench",
    organizations=orgs,
    tags=(
        "gpu_kernel",
        "benchmark",
        "fast_p",
        "iterative_refinement",
        "correctness",
        "superseded_evaluation_risk",
        "primary_source",
    ),
    version="ICML 2025 PMLR version retrieved 2026-07-29",
    notes="Interpret performance numbers with the later KernelBench-Verified adversarial correction.",
)
claim = make_claim(
    claim_id="clm_kernelbench_workload_count",
    paper_id=paper.paper_id,
    statement=(
        "The original KernelBench evaluation contains 250 selected PyTorch workloads."
    ),
    claim_type="method_definition",
    applies_to="llm_gpu_kernel_generation_benchmark",
    domain="gpu_kernel_engineering",
    evidence_class="primary_benchmark_later_challenged",
    direction="method_only",
    limitations=(
        "The later KernelBench-Verified paper identifies exploitable correctness tests and unrealistic non-TF32 baselines in the standard protocol.",
        "The 250 workloads are microbenchmarks rather than end-to-end OPHIS training.",
        "Matching PyTorch under the original protocol is not verified SOTA evidence.",
    ),
    locator="Benchmark composition",
    tags=("gpu_kernel", "benchmark", "methodology", "validity_correction"),
)
evidence = make_evidence(
    evidence_id="evd_lit_kernelbench_workload_count",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=systems_rating(paper.paper_id),
    reported={
        "workloads": 250,
    },
    mechanism="Execution/profiling feedback lets an agent repair correctness and target measured bottlenecks rather than optimize from text alone.",
    method="Primary ICML benchmark read, explicitly conditioned on the later verified-protocol correction.",
    code_ref="Use profilers and randomized semantic tests; never accept original fast_p alone as local evidence",
    design="adequate",
    replication="adequate",
    scope_match="adequate",
    directness="adequate",
    trust_limitations=claim.limitations,
    relation="mixed",
    strength="moderate",
    assessment_limitations=(
        "Supports iterative execution/profiling as a workflow.",
        "Does not support the validity of original benchmark speedups after the 2026 adversarial audit.",
    ),
    tags=("gpu_kernel", "benchmark", "later_challenged"),
)
TRIPLES.append((paper, claim, evidence))
kernelbench_base_evidence = evidence

for spec in (
    {
        "claim_id": "clm_kernelbench_fast_p_definition",
        "statement": (
            "KernelBench defines fast_p as the fraction of generated kernels "
            "that are correct and faster than threshold p."
        ),
        "claim_type": "method_definition",
        "direction": "method_only",
        "reported": {
            "metric": (
                "fast_p: fraction of generated kernels that are correct and "
                "faster than threshold p"
            )
        },
        "locator": "fast_p metric definition",
        "mechanism": "The metric joins correctness with a declared speed threshold.",
        "method": "Primary ICML paper extraction of the metric definition only.",
        "relation": "supports",
        "tags": ("gpu_kernel", "benchmark", "fast_p", "methodology"),
    },
    {
        "claim_id": "clm_kernelbench_frontier_out_of_box_match",
        "statement": (
            "Under the original KernelBench protocol, frontier reasoning models "
            "matched the PyTorch baseline in fewer than 20 percent of workloads "
            "without iterative feedback."
        ),
        "claim_type": "negative_result",
        "direction": "null",
        "reported": {"frontier_out_of_box_matching_pytorch_percent": "less than 20"},
        "locator": "Out-of-box frontier-model results",
        "mechanism": "Text-only kernel generation often misses correctness or baseline speed.",
        "method": (
            "Primary ICML paper extraction of the out-of-box endpoint, explicitly "
            "conditioned on the later verified-protocol correction."
        ),
        "relation": "mixed",
        "tags": ("gpu_kernel", "benchmark", "negative_result", "later_challenged"),
    },
    {
        "claim_id": "clm_kernelbench_execution_feedback_effect",
        "statement": (
            "Execution feedback improved iterative kernel refinement under the "
            "original KernelBench protocol."
        ),
        "claim_type": "causal",
        "direction": "beneficial",
        "reported": {"execution_feedback_improved_iterative_refinement": True},
        "locator": "Iterative-refinement execution-feedback experiment",
        "mechanism": "Execution failures expose correctness defects for repair.",
        "method": (
            "Primary ICML paper extraction of the execution-feedback intervention "
            "only, conditioned on the later benchmark-validity correction."
        ),
        "relation": "mixed",
        "tags": ("gpu_kernel", "execution_feedback", "iterative_refinement"),
    },
    {
        "claim_id": "clm_kernelbench_profiling_feedback_effect",
        "statement": (
            "Profiling feedback improved iterative kernel refinement under the "
            "original KernelBench protocol."
        ),
        "claim_type": "causal",
        "direction": "beneficial",
        "reported": {"profiling_feedback_improved_iterative_refinement": True},
        "locator": "Iterative-refinement profiling-feedback experiment",
        "mechanism": "Profiling identifies the measured bottleneck targeted in the next revision.",
        "method": (
            "Primary ICML paper extraction of the profiling-feedback intervention "
            "only, conditioned on the later benchmark-validity correction."
        ),
        "relation": "mixed",
        "tags": ("gpu_kernel", "profiling_feedback", "iterative_refinement"),
    },
):
    split_claim = make_claim(
        claim_id=spec["claim_id"],
        paper_id=paper.paper_id,
        statement=spec["statement"],
        claim_type=spec["claim_type"],
        applies_to="llm_gpu_kernel_generation_benchmark",
        domain="gpu_kernel_engineering",
        evidence_class="primary_benchmark_later_challenged",
        direction=spec["direction"],
        limitations=claim.limitations,
        locator=spec["locator"],
        tags=spec["tags"],
    )
    split_evidence = make_split_evidence(
        base=kernelbench_base_evidence,
        evidence_id=f"evd_lit_{split_claim.claim_id.removeprefix('clm_')}",
        claim=split_claim,
        reported=spec["reported"],
        mechanism=spec["mechanism"],
        method=spec["method"],
        relation=spec["relation"],
        strength="moderate",
        assessment_limitations=(
            "This record assesses only the stated method fact or endpoint.",
            "The later verified benchmark limits performance interpretation.",
        ),
        tags=spec["tags"],
    )
    EXTRA_CLAIM_EVIDENCE.append((split_claim, split_evidence))


# 18. SparseTransX -- MLSys 2025, deliberately retained scope-mismatch evidence.
orgs = ("Texas A&M University",)
paper = make_paper(
    paper_id="pap_sparsetransx_2025",
    title="SparseTransX: Efficient Training of Translation-Based Knowledge Graph Embeddings Using Sparse Matrix Operations",
    authors=("Md Saidul Hoque Anik", "Ariful Azad"),
    year=2025,
    venue_name="MLSys 2025",
    peer_reviewed=True,
    primary="https://proceedings.mlsys.org/paper_files/paper/2025/hash/36e2967f87c3362e37cf988781a887ad-Abstract-Conference.html",
    archive="https://arxiv.org/abs/2502.16949",
    code="https://github.com/HipGraph/SpTransX",
    organizations=orgs,
    tags=(
        "sparse_matrix",
        "spmm",
        "knowledge_graph",
        "embedding_gradient",
        "gpu_training",
        "scope_mismatch",
        "primary_source",
    ),
    version="MLSys 2025 proceedings version retrieved 2026-07-29",
    notes="Official SpTransX Python package linked from the paper.",
)
claim = make_claim(
    claim_id="clm_sparsetransx_spmm_mechanism",
    paper_id=paper.paper_id,
    statement=(
        "Replacing multiple gather/scatter embedding-gradient operations in "
        "translation-based knowledge-graph embedding training with a single "
        "sparse-dense matrix multiplication is the core SparseTransX mechanism."
    ),
    claim_type="mechanistic",
    applies_to="knowledge_graph_embedding_gradient_spmm",
    domain="translation_based_knowledge_graph_training",
    evidence_class="paper_faithful_scope_mismatch",
    direction="mechanism_only",
    limitations=(
        "This is knowledge-graph embedding training, not Transformer language-model training despite the name TransX.",
        "Its gather/scatter gradient structure does not match the dense OPHIS Transformer or hashed n-gram value path automatically.",
        "Only a measured local sparse gather/scatter bottleneck could justify a derived test.",
    ),
    locator="Sparse embedding-gradient framework definition",
    tags=("spmm", "knowledge_graph", "scope_mismatch", "negative_selection_evidence"),
)
evidence = make_evidence(
    evidence_id="evd_lit_sparsetransx_spmm_mechanism",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=systems_rating(paper.paper_id),
    reported={
        "operation": "unify embedding gather/scatter gradient work as SpMM",
    },
    mechanism="Sparse incidence matrices combine repeated gather/scatter embedding updates into optimized SpMM kernels.",
    method="Peer-reviewed primary-paper read with explicit workload-structure mismatch analysis.",
    code_ref="No matching knowledge-graph embedding loop in train.py; profile before any derived sparse-kernel proposal",
    design="strong",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Retained as a documented scope-mismatch so title similarity does not cause repeated irrelevant proposals.",
    ),
    tags=("spmm", "scope_mismatch", "negative_selection_evidence"),
)
TRIPLES.append((paper, claim, evidence))
sparsetransx_base_evidence = evidence

for spec in (
    {
        "claim_id": "clm_sparsetransx_cpu_speedup",
        "statement": (
            "SparseTransX reported up to 5.3x CPU speedup for the tested "
            "translation-based knowledge-graph embedding workloads."
        ),
        "claim_type": "causal",
        "direction": "beneficial",
        "reported": {
            "maximum_cpu_speedup": 5.3,
            "model_scope": ["TransE", "TransR", "TransH", "TorusE"],
        },
        "locator": "CPU performance evaluation",
        "mechanism": "Sparse SpMM consolidates repeated CPU gather/scatter gradient work.",
        "method": "Primary proceedings extraction of the maximum CPU-speed endpoint only.",
        "tags": ("spmm", "cpu", "scope_mismatch"),
    },
    {
        "claim_id": "clm_sparsetransx_gpu_speedup",
        "statement": (
            "SparseTransX reported up to 4.2x GPU speedup for the tested "
            "translation-based knowledge-graph embedding workloads."
        ),
        "claim_type": "causal",
        "direction": "beneficial",
        "reported": {
            "maximum_gpu_speedup": 4.2,
            "model_scope": ["TransE", "TransR", "TransH", "TorusE"],
        },
        "locator": "GPU performance evaluation",
        "mechanism": "Sparse SpMM consolidates repeated GPU gather/scatter gradient work.",
        "method": "Primary proceedings extraction of the maximum GPU-speed endpoint only.",
        "tags": ("spmm", "gpu", "scope_mismatch"),
    },
    {
        "claim_id": "clm_sparsetransx_gpu_memory",
        "statement": (
            "SparseTransX used less GPU memory across the tested TransE, TransR, "
            "TransH, and TorusE models."
        ),
        "claim_type": "causal",
        "direction": "beneficial",
        "reported": {
            "gpu_memory_vs_baseline": "lower",
            "model_scope": ["TransE", "TransR", "TransH", "TorusE"],
        },
        "locator": "GPU memory evaluation",
        "mechanism": "A consolidated sparse representation avoids duplicated gather/scatter intermediates.",
        "method": "Primary proceedings extraction of the GPU-memory endpoint only.",
        "tags": ("spmm", "gpu_memory", "scope_mismatch"),
    },
):
    split_claim = make_claim(
        claim_id=spec["claim_id"],
        paper_id=paper.paper_id,
        statement=spec["statement"],
        claim_type=spec["claim_type"],
        applies_to="knowledge_graph_embedding_gradient_spmm",
        domain="translation_based_knowledge_graph_training",
        evidence_class="paper_faithful_scope_mismatch",
        direction=spec["direction"],
        limitations=claim.limitations,
        locator=spec["locator"],
        tags=spec["tags"],
    )
    split_evidence = make_split_evidence(
        base=sparsetransx_base_evidence,
        evidence_id=f"evd_lit_{split_claim.claim_id.removeprefix('clm_')}",
        claim=split_claim,
        reported=spec["reported"],
        mechanism=spec["mechanism"],
        method=spec["method"],
        relation="supports",
        strength="moderate",
        assessment_limitations=(
            "This record assesses only the stated endpoint.",
            "The knowledge-graph workload structure does not transfer automatically to OPHIS.",
        ),
        tags=spec["tags"],
    )
    EXTRA_CLAIM_EVIDENCE.append((split_claim, split_evidence))


# 19. Cut Cross-Entropy -- ICLR 2025, memory mechanism with local vocab mismatch.
orgs = ("Apple",)
paper = make_paper(
    paper_id="pap_cut_cross_entropy_2025",
    title="Cut Your Losses in Large-Vocabulary Language Models",
    authors=("Erik Wijmans", "Brody Huval", "Alexander Hertzberg", "Vladlen Koltun", "Philipp Krähenbühl"),
    year=2025,
    venue_name="ICLR 2025 Oral",
    peer_reviewed=True,
    primary="https://proceedings.iclr.cc/paper_files/paper/2025/file/aa963ac256590bb7ad5fc26c68229a3a-Paper-Conference.pdf",
    archive="https://arxiv.org/abs/2411.09009",
    code="https://github.com/apple/ml-cross-entropy",
    organizations=orgs,
    tags=(
        "cross_entropy",
        "large_vocabulary",
        "memory",
        "kernel",
        "logsumexp",
        "gradient_sparsity",
        "scope_mismatch",
        "primary_source",
    ),
    version="ICLR 2025 proceedings version retrieved 2026-07-29",
    notes="Official Apple implementation linked from the publication.",
)
claim = make_claim(
    claim_id="clm_cut_cross_entropy_streaming_mechanism",
    paper_id=paper.paper_id,
    statement=(
        "Cut Cross-Entropy computes the correct-token logit and vocabulary "
        "log-sum-exp without materializing the full token-by-vocabulary matrix."
    ),
    claim_type="mechanistic",
    applies_to="large_vocabulary_cross_entropy_memory",
    domain="transformer_lm_training",
    evidence_class="paper_faithful_scope_mismatch",
    direction="mechanism_only",
    limitations=(
        "The benefit scales with a large vocabulary and token-by-vocabulary logit matrix.",
        "OPHIS uses vocabulary size 8192 and is not reported to be classifier-head-memory bound.",
        "Saving memory cannot improve BPB unless it unlocks a separately authorized batch/context/model change or increases end-to-end steps.",
    ),
    locator="Method definition; streamed correct-token and log-sum-exp computation",
    tags=("cross_entropy", "memory", "kernel", "scope_mismatch"),
)
evidence = make_evidence(
    evidence_id="evd_lit_cut_cross_entropy_streaming_mechanism",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=systems_rating(paper.paper_id),
    reported={
        "materializes_full_token_by_vocabulary_matrix": False,
    },
    mechanism="Fused on-chip matrix products and streaming log-sum-exp avoid global storage of the full logits matrix; negligible softmax-gradient terms can be skipped.",
    method="Peer-reviewed primary-paper/code read with local vocabulary and memory-bottleneck scope check.",
    code_ref="train.py output projection/cross entropy; local model about 94.4M total and 56.6M matrix parameters, vocab 8192",
    design="strong",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="strong",
    assessment_limitations=(
        "A strong systems result but not a direct quality lever in the current non-memory-bound, small-vocabulary frame.",
        "The live vocabulary is 8192 and the local microbenchmark attributes only about 1.6 percent of step time to cross-entropy; no governed full-run win exists.",
        "Profile first; do not infer BPB improvement from memory reduction alone.",
    ),
    tags=("cross_entropy", "memory", "scope_mismatch"),
)
TRIPLES.append((paper, claim, evidence))
cce_base_evidence = evidence

for spec in (
    {
        "claim_id": "clm_cut_cross_entropy_loss_memory",
        "statement": (
            "For Gemma 2 2B, Cut Cross-Entropy reduced loss-computation memory "
            "from 24GB to 1MB."
        ),
        "claim_type": "causal",
        "direction": "beneficial",
        "reported": {"gemma2_2b_loss_memory": {"baseline_gb": 24, "cce_mb": 1}},
        "locator": "Figure 1; Gemma 2 2B loss-memory evaluation",
        "mechanism": "Streaming log-sum-exp avoids storing the full logits matrix.",
        "method": "Primary ICLR paper extraction of the loss-memory endpoint only.",
        "relation": "supports",
        "tags": ("cross_entropy", "loss_memory", "scope_mismatch"),
    },
    {
        "claim_id": "clm_cut_cross_entropy_classifier_memory",
        "statement": (
            "For Gemma 2 2B, Cut Cross-Entropy reduced classifier-head training "
            "memory from 28GB to 1GB."
        ),
        "claim_type": "causal",
        "direction": "beneficial",
        "reported": {
            "gemma2_2b_classifier_head_training_memory": {
                "baseline_gb": 28,
                "cce_gb": 1,
            }
        },
        "locator": "Figure 1; Gemma 2 2B classifier-head memory evaluation",
        "mechanism": "Streaming vocabulary products avoids retaining a dense classifier-logit tensor.",
        "method": "Primary ICLR paper extraction of the classifier-memory endpoint only.",
        "relation": "supports",
        "tags": ("cross_entropy", "classifier_memory", "scope_mismatch"),
    },
    {
        "claim_id": "clm_cut_cross_entropy_training_speed",
        "statement": (
            "The paper reports no training-speed sacrifice from Cut Cross-Entropy "
            "in the evaluated setting."
        ),
        "claim_type": "negative_result",
        "direction": "null",
        "reported": {"reported_training_speed_sacrifice": "none"},
        "locator": "Training-efficiency evaluation",
        "mechanism": "Fused vocabulary products offset the work added by streaming.",
        "method": "Primary ICLR paper extraction of the training-speed null endpoint only.",
        "relation": "supports",
        "tags": ("cross_entropy", "training_speed", "null_result"),
    },
    {
        "claim_id": "clm_cut_cross_entropy_convergence",
        "statement": (
            "The paper reports no convergence sacrifice from Cut Cross-Entropy "
            "in the evaluated setting."
        ),
        "claim_type": "negative_result",
        "direction": "null",
        "reported": {"reported_convergence_sacrifice": "none"},
        "locator": "Training-convergence evaluation",
        "mechanism": "The streamed formulation preserves the cross-entropy objective.",
        "method": "Primary ICLR paper extraction of the convergence null endpoint only.",
        "relation": "supports",
        "tags": ("cross_entropy", "convergence", "null_result"),
    },
):
    split_claim = make_claim(
        claim_id=spec["claim_id"],
        paper_id=paper.paper_id,
        statement=spec["statement"],
        claim_type=spec["claim_type"],
        applies_to="large_vocabulary_cross_entropy_memory",
        domain="transformer_lm_training",
        evidence_class="paper_faithful_scope_mismatch",
        direction=spec["direction"],
        limitations=claim.limitations,
        locator=spec["locator"],
        tags=spec["tags"],
    )
    split_evidence = make_split_evidence(
        base=cce_base_evidence,
        evidence_id=f"evd_lit_{split_claim.claim_id.removeprefix('clm_')}",
        claim=split_claim,
        reported=spec["reported"],
        mechanism=spec["mechanism"],
        method=spec["method"],
        relation=spec["relation"],
        strength="strong",
        assessment_limitations=(
            "This record assesses only the stated endpoint.",
            "The local 8192-token vocabulary and measured bottleneck remain scope limitations.",
        ),
        tags=spec["tags"],
    )
    EXTRA_CLAIM_EVIDENCE.append((split_claim, split_evidence))


# ---------------------------------------------------------------------------
# Append-only provenance corrections.  These use new IDs because PaperRecord
# and ClaimRecord have no supersession field; the new evidence explicitly
# supersedes the old evidence.  Historical records are never edited.
# ---------------------------------------------------------------------------

# 20. Correct the FA4 secondary/blog metadata with the primary arXiv paper.
orgs = (
    "Princeton University",
    "Meta",
    "Colfax Research",
    "NVIDIA",
    "Georgia Institute of Technology",
    "Together AI",
)
paper = make_paper(
    paper_id="pap_flashattention4_primary",
    title="FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling",
    authors=("Ted Zadouri", "Markus Hoehnerbach", "Jay Shah", "Timmy Liu", "Vijay Thakkar", "Tri Dao"),
    year=2026,
    venue_name="arXiv 2603.05451",
    peer_reviewed=False,
    primary="https://arxiv.org/abs/2603.05451",
    code="https://github.com/Dao-AILab/flash-attention",
    organizations=orgs,
    tags=(
        "attention",
        "kernel",
        "blackwell",
        "hardware_specific",
        "provenance_correction",
        "primary_source",
    ),
    version="arXiv 2603.05451 version retrieved 2026-07-29",
    notes=(
        "Append-only primary-source correction for pap_flashattention4, whose "
        "primary URL is a Princeton blog and whose author list/numbers are abbreviated."
    ),
)
claim = make_claim(
    claim_id="clm_fa4_h200_lpt_scheduler_mha",
    paper_id=paper.paper_id,
    statement=(
        "On H200 with BF16 head dimension 128, FlashAttention-4's "
        "longest-processing-time-first tile scheduling improved reported "
        "performance by 4-8 percent for MHA."
    ),
    claim_type="causal",
    applies_to="attention_tile_scheduling",
    domain="nvidia_h200_attention",
    evidence_class="primary_provenance_correction",
    direction="beneficial",
    limitations=(
        "The H200 result isolates scheduling; the paper's main asynchronous pipeline, exponent, tensor-memory, and 2-CTA mechanisms remain Blackwell/B200-specific.",
        "The result is BF16 at head dimension 128 and does not establish end-to-end training speed.",
        "The local FA3 varlen interface does not expose the paper's tile scheduler as an isolated switch.",
        "Attention is a small fraction of the current local step, capping expected end-to-end impact.",
    ),
    locator="Section 3.3, H200 BF16 head-dimension-128 scheduling ablation",
    tags=("attention", "kernel", "h200", "scheduler", "provenance_correction"),
)
evidence = make_evidence(
    evidence_id="evd_lit_fa4_h200_lpt_scheduler_mha",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating={
        "scale": "1_to_5_each",
        "dimensions": {
            "novelty": 2,
            "provenance": 4,
            "validity": 3,
            "impact": 1,
            "reliability": 3,
            "feasibility_cost": 2,
            "falsifiability": 5,
        },
        "total": 20,
        "maximum": 35,
        "status": "independent_critic_reconciled",
    },
    reported={
        "hardware": "NVIDIA H200",
        "precision": "BF16",
        "head_dimension": 128,
        "attention_mode": "MHA",
        "lpt_scheduler_gain_percent": [4, 8],
    },
    mechanism="Scheduling the longest attention tiles first reduces tail imbalance across persistent workers.",
    method="Primary Section 3.3 extraction correcting the old blanket claim that the paper contains no H200 measurement.",
    code_ref="Campaign hardware is H200/Hopper with FA3; no isolated LPT scheduler switch exists in the local interface",
    design="adequate",
    replication="adequate",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Only the scheduler transfers at the hardware-generation level; the main FA4 core remains B200-specific.",
        "No local SOTA, adoption, or end-to-end speed claim follows from this microbenchmark ablation.",
    ),
    tags=("kernel", "hardware_specific", "provenance_correction"),
)
TRIPLES.append((paper, claim, evidence))
fa4_base_evidence = evidence

claim = make_claim(
    claim_id="clm_fa4_h200_lpt_scheduler_mqa8",
    paper_id=paper.paper_id,
    statement=(
        "On H200 with BF16 head dimension 128, FlashAttention-4's "
        "longest-processing-time-first tile scheduling improved reported "
        "performance by 7-14 percent for MQA-8."
    ),
    claim_type="causal",
    applies_to="attention_tile_scheduling",
    domain="nvidia_h200_attention",
    evidence_class="primary_provenance_correction",
    direction="beneficial",
    limitations=claim.limitations,
    locator="Section 3.3, H200 BF16 head-dimension-128 MQA-8 scheduling ablation",
    tags=("attention", "kernel", "h200", "scheduler", "provenance_correction"),
)
evidence = make_split_evidence(
    base=fa4_base_evidence,
    evidence_id="evd_lit_fa4_h200_lpt_scheduler_mqa8",
    claim=claim,
    reported={
        "hardware": "NVIDIA H200",
        "precision": "BF16",
        "head_dimension": 128,
        "attention_mode": "MQA-8",
        "lpt_scheduler_gain_percent": [7, 14],
    },
    mechanism="Scheduling the longest attention tiles first reduces tail imbalance across persistent workers.",
    method="Primary Section 3.3 extraction of the MQA-8 scheduler endpoint only.",
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "Only the scheduler transfers at the hardware-generation level.",
        "No local SOTA, adoption, or end-to-end speed claim follows.",
    ),
    tags=("kernel", "hardware_specific", "provenance_correction"),
)
EXTRA_CLAIM_EVIDENCE.append((claim, evidence))


# 21. Correct Tensorizing Engram authors, venue, mechanism, and direct evidence.
orgs = (
    "Imperial College London",
    "RIKEN Center for Advanced Intelligence Project",
)
paper = make_paper(
    paper_id="pap_tensorizing_engram_primary",
    title="Tensorizing Engram: Sharing Latents Across N-Gram Embeddings is Beneficial in LLMs",
    authors=("Wuyang Zhou", "Yuxuan Gu", "Giorgos Iacovides", "Yuning Qiu", "Qibin Zhao", "Danilo Mandic"),
    year=2026,
    venue_name="arXiv 2606.08347",
    peer_reviewed=False,
    primary="https://arxiv.org/abs/2606.08347",
    organizations=orgs,
    tags=(
        "ngram",
        "memory",
        "tensor_network",
        "cp_decomposition",
        "compression",
        "parameter_golf",
        "provenance_correction",
        "primary_source",
    ),
    version="arXiv 2606.08347v1 retrieved 2026-07-29",
    notes=(
        "Append-only correction for pap_tensorizing_engram, which lists Unknown "
        "authors and says the paper was not independently read. No official code "
        "repository was located in the primary record."
    ),
)
claim = make_claim(
    claim_id="clm_tngram_cp_shared_factors_mechanism",
    paper_id=paper.paper_id,
    statement=(
        "TN-gram replaces independent order-specific n-gram latent tables with "
        "Canonical-Polyadic token-position factors shared across orders plus "
        "order-absorption vectors that specialize each n-gram order."
    ),
    claim_type="mechanistic",
    applies_to="tensorized_shared_ngram_memory",
    domain="short_budget_transformer_lm_pretraining",
    evidence_class="primary_provenance_correction",
    direction="mechanism_only",
    limitations=(
        "At n=2 cross-order sharing is absent and the paper reports TN-gram can be worse.",
        "The mechanism is CP shared-factor memory, not two independent decorrelated hash codes; the prior local composite-code test is not a faithful replication.",
        "Only arXiv provenance was independently verified; no workshop acceptance or official code repository is asserted.",
    ),
    locator="Sections 4.2-4.4, Equations 6-13",
    tags=("ngram", "tensor_network", "cp_decomposition", "provenance_correction"),
)
evidence = make_evidence(
    evidence_id="evd_lit_tngram_cp_shared_factors_mechanism",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating={
        "scale": "1_to_5_each",
        "dimensions": {
            "novelty": 4,
            "provenance": 3,
            "validity": 4,
            "impact": 2,
            "reliability": 3,
            "feasibility_cost": 1,
            "falsifiability": 5,
        },
        "total": 22,
        "maximum": 35,
        "status": "independent_critic_reconciled",
    },
    reported={
        "factorization": "Canonical-Polyadic token-position factors shared across n-gram orders",
        "order_specialization": "order-absorption vectors",
        "maximum_order_main_comparison": 5,
    },
    mechanism="Shared CP token-position factors exploit nested suffix structure across n-gram orders; order-absorption vectors specialize each order.",
    method="Full primary HTML/paper read correcting secondary metadata and distinguishing CP factor sharing from composite hashing.",
    code_ref="train.py hashed bigram/trigram/fourgram tables; no CP shared-factor TN-gram implementation",
    design="adequate",
    replication="weak",
    scope_match="adequate",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "The local C2 composite-hash regression does not falsify TN-gram because the intervention omitted CP shared factors and order-absorption vectors.",
        "A faithful local test would be a new implementation and must be separately gated.",
    ),
    tags=("ngram", "tensor_network", "provenance_correction"),
)
TRIPLES.append((paper, claim, evidence))

claim = make_claim(
    claim_id="clm_tngram_9layer_parameters",
    paper_id=paper.paper_id,
    statement=(
        "In the 9-layer comparison, TN-gram used 19M added parameters versus "
        "Engram's 26M."
    ),
    claim_type="causal",
    applies_to="tensorized_shared_ngram_memory",
    domain="short_budget_transformer_lm_pretraining",
    evidence_class="primary_paper_endpoint",
    direction="beneficial",
    limitations=(
        "The BPB difference is 0.001 and the record does not claim it clears the local noise floor.",
        "The paper's model/configuration is not the live OPHIS frame.",
    ),
    locator="Table 1, 9-layer added-parameter comparison",
    tags=("ngram", "tensor_network", "parameters"),
)
evidence = make_evidence(
    evidence_id="evd_lit_tngram_9layer_parameters",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(4, 3, 4, 2, 3, 1, 5),
    reported={
        "engram_added_parameters_m": 26,
        "tngram_added_parameters_m": 19,
    },
    mechanism="Cross-order CP factor sharing reduces latent-memory parameters.",
    method="Primary Table 1 extraction of the 9-layer added-parameter endpoint only.",
    code_ref="No local CP shared-factor TN-gram implementation",
    design="adequate",
    replication="weak",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=("No local SOTA or meaningful-effect claim is made.",),
    tags=("ngram", "tensor_network", "parameters"),
)
EXTRA_CLAIM_EVIDENCE.append((claim, evidence))
tngram_9layer_base_evidence = evidence

claim = make_claim(
    claim_id="clm_tngram_9layer_val_bpb",
    paper_id=paper.paper_id,
    statement=(
        "In the 9-layer comparison, TN-gram changed validation BPB from "
        "Engram's 1.209 to 1.208."
    ),
    claim_type="causal",
    applies_to="tensorized_shared_ngram_memory",
    domain="short_budget_transformer_lm_pretraining",
    evidence_class="primary_paper_endpoint",
    direction="beneficial",
    limitations=(
        "The BPB difference is 0.001 and the record does not claim it clears the local noise floor.",
        "The paper's model/configuration is not the live OPHIS frame.",
    ),
    locator="Table 1, 9-layer validation-BPB comparison",
    tags=("ngram", "tensor_network", "val_bpb"),
)
evidence = make_split_evidence(
    base=tngram_9layer_base_evidence,
    evidence_id="evd_lit_tngram_9layer_val_bpb",
    claim=claim,
    reported={"engram_val_bpb": 1.209, "tngram_val_bpb": 1.208},
    mechanism="Cross-order CP factor sharing changes the learned n-gram memory representation.",
    method="Primary Table 1 extraction of the 9-layer validation-BPB endpoint only.",
    relation="supports",
    strength="weak",
    assessment_limitations=(
        "The 0.001 difference is recorded without a meaningful-effect or local-SOTA claim.",
    ),
    tags=("ngram", "tensor_network", "val_bpb"),
)
EXTRA_CLAIM_EVIDENCE.append((claim, evidence))

claim = make_claim(
    claim_id="clm_tngram_18layer_parameters",
    paper_id=paper.paper_id,
    statement=(
        "In the 18-layer comparison, TN-gram used 48M added parameters versus "
        "Engram's 60M."
    ),
    claim_type="causal",
    applies_to="tensorized_shared_ngram_memory",
    domain="short_budget_transformer_lm_pretraining",
    evidence_class="primary_paper_endpoint",
    direction="beneficial",
    limitations=(
        "BPB is 0.001 worse while CORE is 0.005 higher, so the evidence is mixed across endpoints.",
        "The paper's statistical-equivalence wording does not make the BPB numerically better.",
        "The paper's model/configuration is not the live OPHIS frame.",
    ),
    locator="Table 1, 18-layer added-parameter comparison",
    tags=("ngram", "tensor_network", "parameters"),
)
evidence = make_evidence(
    evidence_id="evd_lit_tngram_18layer_parameters",
    paper_id=paper.paper_id,
    claim_id=claim.claim_id,
    organizations=orgs,
    source_version=paper.version,
    rating=architecture_rating(4, 3, 4, 2, 3, 1, 5),
    reported={
        "engram_added_parameters_m": 60,
        "tngram_added_parameters_m": 48,
    },
    mechanism="Cross-order CP factor sharing reduces added n-gram-memory parameters.",
    method="Primary Table 1 extraction of the 18-layer added-parameter endpoint only.",
    code_ref="No local CP shared-factor TN-gram implementation",
    design="adequate",
    replication="weak",
    scope_match="weak",
    directness="strong",
    trust_limitations=claim.limitations,
    relation="supports",
    strength="moderate",
    assessment_limitations=(
        "This record assesses parameter count only and makes no quality-improvement claim.",
    ),
    tags=("ngram", "tensor_network", "parameters"),
)
EXTRA_CLAIM_EVIDENCE.append((claim, evidence))
tngram_18layer_base_evidence = evidence

for spec in (
    {
        "claim_id": "clm_tngram_18layer_val_bpb",
        "statement": (
            "In the 18-layer comparison, TN-gram changed validation BPB from "
            "Engram's 1.070 to 1.071."
        ),
        "direction": "harmful",
        "reported": {"engram_val_bpb": 1.070, "tngram_val_bpb": 1.071},
        "locator": "Table 1, 18-layer validation-BPB comparison",
        "mechanism": "The tensorized representation produces a slightly different validation-loss endpoint.",
        "method": "Primary Table 1 extraction of the 18-layer validation-BPB endpoint only.",
        "relation": "opposes",
        "tags": ("ngram", "tensor_network", "val_bpb"),
    },
    {
        "claim_id": "clm_tngram_18layer_core",
        "statement": (
            "In the 18-layer comparison, TN-gram changed CORE from Engram's "
            "0.115 to 0.120."
        ),
        "direction": "beneficial",
        "reported": {"engram_core": 0.115, "tngram_core": 0.120},
        "locator": "Table 1, 18-layer CORE comparison",
        "mechanism": "The tensorized representation produces a slightly different downstream CORE endpoint.",
        "method": "Primary Table 1 extraction of the 18-layer CORE endpoint only.",
        "relation": "supports",
        "tags": ("ngram", "tensor_network", "core"),
    },
):
    split_claim = make_claim(
        claim_id=spec["claim_id"],
        paper_id=paper.paper_id,
        statement=spec["statement"],
        claim_type="causal",
        applies_to="tensorized_shared_ngram_memory",
        domain="short_budget_transformer_lm_pretraining",
        evidence_class="primary_paper_endpoint",
        direction=spec["direction"],
        limitations=claim.limitations,
        locator=spec["locator"],
        tags=spec["tags"],
    )
    split_evidence = make_split_evidence(
        base=tngram_18layer_base_evidence,
        evidence_id=f"evd_lit_{split_claim.claim_id.removeprefix('clm_')}",
        claim=split_claim,
        reported=spec["reported"],
        mechanism=spec["mechanism"],
        method=spec["method"],
        relation=spec["relation"],
        strength="weak",
        assessment_limitations=(
            "This record preserves the endpoint sign without a universal quality claim.",
        ),
        tags=spec["tags"],
    )
    EXTRA_CLAIM_EVIDENCE.append((split_claim, split_evidence))


# Evidence-only records preserve the exact subjects of the historical evidence
# they correct.  New primary-paper claims above are separate evidence and do
# not masquerade as supersession edges.
EVIDENCE_ONLY: list[EvidenceRecord] = []

EVIDENCE_ONLY.append(
    make_evidence(
        evidence_id="evd_lit_ngram_regularizer_scope_correction",
        paper_id="pap_ngram_aware_memorization_regularizer",
        claim_id="clm_ngram_regularizer_reduces_memorization",
        organizations=(),
        source_version="TACL 2025, ACL Anthology 2025.tacl-1.66",
        rating=architecture_rating(2, 5, 4, 2, 2, 3, 4),
        reported={
            "model_families": ["Pythia", "Llama 3", "Mistral"],
            "parameter_range": "1.4B to 70B",
            "regime": "domain adaptation and instruction tuning",
            "maximum_memorization_reduction_percent": 40,
            "claim_not_reported": (
                "No from-scratch approximately 94.4M-total/56.6M-matrix-parameter "
                "pretraining BPB result and no test of decoupled n-gram-table "
                "weight decay or frequency backoff."
            ),
        },
        mechanism=(
            "An n-gram memorization score supplies early stopping and an "
            "n-gram-aware loss term penalizes memorization during fine-tuning."
        ),
        method=(
            "Subject-preserving correction separating the paper's measured "
            "fine-tuning memorization effect from the prior local-transfer inference."
        ),
        code_ref=(
            "train.py ngram_embeds weight-decay group and frequency-confidence "
            "backoff are different interventions from the paper's loss regularizer"
        ),
        design="strong",
        replication="adequate",
        scope_match="weak",
        directness="weak",
        trust_limitations=(
            "Fine-tuning rather than from-scratch pretraining.",
            "1.4B-70B rather than the local approximately 94.4M total/56.6M matrix-parameter model.",
            "The exact local table-WD/backoff mechanisms are not tested.",
            "No affiliation is asserted because it was not verified in the primary PDF.",
        ),
        relation="supports",
        strength="weak",
        assessment_limitations=(
            "Supports the broad in-paper memorization result only.",
            "Does not strongly corroborate C1/C2/C3, validate BPB, or establish intervention equivalence.",
            "Supersedes the prior strong local-transfer assessment.",
        ),
        tags=("ngram", "memorization", "scope_correction", "transfer_downgrade"),
        supersedes_evidence_id="evd_lit_ngram_regularizer_corroborates_family",
        hypothesis_ids=("hyp_c1_weak_lambda_5shard",),
    )
)

EVIDENCE_ONLY.append(
    make_evidence(
        evidence_id="evd_lit_fa4_hopper_scope_correction",
        paper_id="pap_flashattention4",
        claim_id="clm_fa4_blackwell_hardware_coupling",
        organizations=(
            "Princeton University",
            "Meta",
            "Colfax Research",
            "NVIDIA",
            "Georgia Institute of Technology",
            "Together AI",
        ),
        source_version="arXiv 2603.05451, Section 3.3, retrieved 2026-07-29",
        rating=architecture_rating(2, 4, 3, 1, 3, 2, 5),
        reported={
            "old_claim_fragment_corrected": "no Hopper (H200) measurement",
            "h200_precision": "BF16",
            "h200_head_dimension": 128,
            "h200_lpt_scheduler_gain_percent": {"mha": [4, 8], "mqa_8": [7, 14]},
            "core_pipeline_hardware_scope": "B200/Blackwell",
        },
        mechanism="LPT tile ordering transfers to H200; the paper's core asymmetric pipeline remains Blackwell-specific.",
        method="Subject-preserving primary-source correction of the earlier blog-derived FA4 evidence.",
        code_ref="H200/Hopper campaign uses FA3; no local LPT scheduler switch or end-to-end transfer result",
        design="adequate",
        replication="adequate",
        scope_match="weak",
        directness="strong",
        trust_limitations=(
            "Only the scheduler ablation is measured on H200.",
            "The main FA4 pipeline remains B200-specific.",
            "No local end-to-end training measurement is reported.",
        ),
        relation="mixed",
        strength="moderate",
        assessment_limitations=(
            "Corrects the false blanket statement that the paper has no H200 result.",
            "Retains the old conclusion only for the Blackwell-specific core, not the scheduler.",
            "No local SOTA or adoption claim is made.",
        ),
        tags=("kernel", "h200", "scope_correction", "provenance_correction"),
        supersedes_evidence_id="evd_lit_fa4_hopper_scope_mismatch",
    )
)

EVIDENCE_ONLY.append(
    make_evidence(
        evidence_id="evd_lit_tensorizing_engram_scope_correction",
        paper_id="pap_tensorizing_engram",
        claim_id="clm_engram_tensorizing",
        organizations=("Imperial College London", "RIKEN Center for Advanced Intelligence Project"),
        source_version="arXiv 2606.08347v1 retrieved 2026-07-29",
        rating=architecture_rating(4, 3, 4, 2, 3, 1, 5),
        reported={
            "faithful_mechanism": "shared CP token-position factors plus order-absorption vectors",
            "not_equivalent_to": "two independent decorrelated composite hash codes",
            "local_c2_result_relevance": "does not test the paper-faithful CP mechanism",
        },
        mechanism="The paper shares CP factors across n-gram orders rather than composing two hash codes.",
        method="Subject-preserving correction after full primary-paper mechanism and evidence read.",
        code_ref="train_c2.py composite_hash_ids omitted CP shared factors and order-absorption vectors",
        design="adequate",
        replication="weak",
        scope_match="weak",
        directness="strong",
        trust_limitations=(
            "The historical claim is broader than the primary mechanism.",
            "The local C2 intervention is not paper-faithful.",
            "Only arXiv provenance and no official implementation were verified.",
        ),
        relation="mixed",
        strength="moderate",
        assessment_limitations=(
            "The C2 regression remains valid for C2 but cannot falsify TN-gram.",
            "A faithful local TN-gram implementation would be a new, separately gated hypothesis.",
        ),
        tags=("ngram", "tensor_network", "scope_correction", "mechanism_correction"),
        supersedes_evidence_id="evd_lit_tensorizing_engram",
        hypothesis_ids=("hyp_c2_composite_codes",),
    )
)

# The legacy Kimi records were extracted from a secondary blog and the old
# combined evidence was later superseded by a primary-report read.  These
# independent, non-superseding records make the provenance debt explicit one
# claim at a time.  They are subordinate assessments, not successor branches.
for evidence_id, claim_id, reported_summary, assessment_note, tags in (
    (
        "evd_lit_kda_linear_full_ratio_provenance_debt",
        "clm_kda_linear_full_ratio",
        (
            "The legacy 3:1 KDA ratio, KV-cache, and decoding figures were "
            "extracted from an Agent One blog rather than the Kimi K3 report."
        ),
        (
            "Subordinate to evd_lit_kimi_k3_primary: use the primary report's "
            "KDA mechanism and scope assessment for research decisions."
        ),
        ("attention", "linear_attention", "provenance_debt"),
    ),
    (
        "evd_lit_attnres_selective_depth_provenance_debt",
        "clm_attnres_selective_depth",
        (
            "The legacy AttnRes efficiency and cost figures were extracted from "
            "an Agent One blog without equations or mechanism detail."
        ),
        (
            "Subordinate to evd_lit_kimi_k3_primary: use the primary report's "
            "AttnRes mechanism and scale assessment for research decisions."
        ),
        ("residual_stream", "depth", "provenance_debt"),
    ),
):
    EVIDENCE_ONLY.append(
        make_evidence(
            evidence_id=evidence_id,
            paper_id="pap_kimi_k3",
            claim_id=claim_id,
            organizations=("Moonshot AI",),
            source_version=(
                "Secondary Agent One blog record retrieved 2026-07-28; "
                "primary Kimi K3 report reviewed separately"
            ),
            rating=architecture_rating(1, 1, 1, 1, 1, 4, 2),
            reported={
                "legacy_source": "https://www.agent-one.dev/blog/kimi-k3-agentone",
                "provenance_debt": reported_summary,
                "subordinate_to_evidence_id": "evd_lit_kimi_k3_primary",
            },
            mechanism=(
                "This record documents provenance quality only; it does not "
                "independently establish the architecture mechanism."
            ),
            method=(
                "Claim-specific provenance audit against the later primary-report "
                "evidence, without creating a supersession edge."
            ),
            code_ref="No code provenance was established by the secondary source.",
            design="weak",
            replication="weak",
            scope_match="weak",
            directness="weak",
            trust_limitations=(
                "Secondary blog rather than the primary technical report.",
                "The legacy claim is broader than this provenance-only assessment.",
                "The 2.8T/1M-context setting is far outside the OPHIS frame.",
            ),
            relation="mixed",
            strength="weak",
            assessment_limitations=(
                assessment_note,
                "No new support, local-adoption claim, or supersession branch is created.",
            ),
            tags=tags,
        )
    )

for evidence_id, claim_id, subject in (
    (
        "evd_lit_recursive_ngram_value_gate_code_snapshot",
        "clm_recursive_ngram_value_gate",
        "the reported n-gram value-gate claim",
    ),
    (
        "evd_lit_recursive_no_kernel_lever_code_snapshot",
        "clm_recursive_no_kernel_lever_reported",
        "the reported absence of a kernel lever",
    ),
):
    EVIDENCE_ONLY.append(
        make_evidence(
            evidence_id=evidence_id,
            paper_id="pap_recursive_nanochat_autoresearch",
            claim_id=claim_id,
            organizations=("Recursive",),
            source_version="Git commit a962ec43e2e3d7c018e59a2ece623fe6e232fdfb",
            rating=architecture_rating(1, 4, 3, 1, 4, 5, 5),
            reported={
                "provenance_subject": subject,
                "repository": (
                    "https://github.com/recursive-org/"
                    "first-steps-toward-automated-ai-research"
                ),
                "commit": "a962ec43e2e3d7c018e59a2ece623fe6e232fdfb",
                "bundle_commit_present": True,
                "bundle_complete_history": True,
                "local_sota_adoption_claimed": False,
            },
            mechanism=(
                "An immutable source capture makes this one cited claim's "
                "implementation context inspectable at an exact commit."
            ),
            method=(
                "Local artifact SHA-256 verification, git-bundle verification, "
                "and exact commit-head check for this claim subject."
            ),
            code_ref=(
                "research/experiments/artifacts/external_code/"
                "pap_recursive_nanochat_autoresearch/manifest.json"
            ),
            design="not_applicable",
            replication="adequate",
            scope_match="adequate",
            directness="strong",
            trust_limitations=(
                "The snapshot establishes source identity, not an isolated causal effect.",
                "It is not a governed local replication.",
                "The exact commit is canonical even if generated archive bytes vary.",
            ),
            relation="supports",
            strength="weak",
            assessment_limitations=(
                "Supports provenance for this one existing claim only.",
                "Makes no SOTA, adoption, or local-performance claim.",
            ),
            tags=(
                "external_code",
                "immutable_commit",
                "sha256_verified",
                "provenance_only",
            ),
        )
    )


# The corrected Tensorizing Engram record intentionally shares a primary URL
# with its deficient historical metadata record.  No other paper-URL duplicate
# is allowed.
INTENTIONAL_PRIMARY_URL_DUPLICATES = {
    "pap_tensorizing_engram_primary": "pap_tensorizing_engram",
}


def all_new_papers() -> list[PaperRecord]:
    return [paper for paper, _, _ in TRIPLES]


def all_new_claims() -> list[ClaimRecord]:
    return [claim for _, claim, _ in TRIPLES] + [
        claim for claim, _ in EXTRA_CLAIM_EVIDENCE
    ]


def all_new_evidence() -> list[EvidenceRecord]:
    return (
        [evidence for _, _, evidence in TRIPLES]
        + [evidence for _, evidence in EXTRA_CLAIM_EVIDENCE]
        + EVIDENCE_ONLY
    )


def canonical_reviewed_payload() -> dict[str, list[dict[str, Any]]]:
    return {
        "papers": [record.to_dict() for record in all_new_papers()],
        "claims": [record.to_dict() for record in all_new_claims()],
        "literature_evidence": [
            record.to_dict() for record in all_new_evidence()
        ],
    }


def reviewed_payload_sha256() -> str:
    payload = canonical_reviewed_payload()
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_review_binding() -> None:
    reviewed_at = str(CRITIC_REVIEW.get("reviewed_at", ""))
    if not reviewed_at.endswith("Z"):
        raise SchemaError("critic reviewed_at must be an explicit UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaError("critic reviewed_at is not valid ISO-8601") from exc
    if parsed.tzinfo != timezone.utc:
        raise SchemaError("critic reviewed_at must parse as UTC")

    expected_digest = str(CRITIC_REVIEW.get("reviewed_payload_sha256", ""))
    actual_digest = reviewed_payload_sha256()
    if expected_digest != actual_digest:
        raise SchemaError(
            "critic review is not bound to this payload: expected "
            f"{expected_digest!r}, computed {actual_digest!r}"
        )

    required_checks = list(CRITIC_REVIEW["required_record_checks"])
    expected_stamp = {
        "reviewer": CRITIC_REVIEW["reviewer"],
        "reviewed_at": reviewed_at,
        "checks": required_checks,
    }
    for record in all_new_evidence():
        if dict(record.facts.get("critic_review", {})) != expected_stamp:
            raise SchemaError(
                f"evidence {record.evidence_id!r} lacks exact per-record critic coverage"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checked(command: list[str], *, purpose: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SchemaError(
            f"{purpose} requires unavailable executable {command[0]!r}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = "\n".join(
            part.strip() for part in (exc.stdout, exc.stderr) if part.strip()
        )
        raise SchemaError(f"{purpose} failed: {detail}") from exc
    return "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )


def _verify_archive_members(
    archive: Path,
    *,
    archive_root: str,
    expected: dict[str, str],
) -> None:
    """Stream a zstd tar once and verify every declared member digest."""
    try:
        decompressor = subprocess.Popen(
            ["zstd", "-dc", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise SchemaError("archive verification requires the zstd executable") from exc
    if decompressor.stdout is None or decompressor.stderr is None:
        raise SchemaError("failed to establish zstd verification pipes")
    observed: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=decompressor.stdout, mode="r|") as stream:
            for member in stream:
                if not member.isfile() or not member.name.startswith(archive_root):
                    continue
                relative = member.name[len(archive_root):]
                if relative not in expected:
                    continue
                source = stream.extractfile(member)
                if source is None:
                    raise SchemaError(
                        f"archive member cannot be read: {member.name!r}"
                    )
                digest = hashlib.sha256()
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                observed[relative] = digest.hexdigest()
    except tarfile.TarError as exc:
        decompressor.kill()
        decompressor.wait()
        raise SchemaError(f"invalid tar stream in {archive}") from exc
    stderr = decompressor.stderr.read().decode("utf-8", errors="replace").strip()
    status = decompressor.wait()
    if status:
        raise SchemaError(
            f"zstd decompression failed for {archive}: {stderr or status}"
        )
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise SchemaError(f"{archive} omits verified members: {missing}")
    mismatches = {
        name: {"expected": expected[name], "computed": observed[name]}
        for name in expected
        if observed[name] != expected[name]
    }
    if mismatches:
        raise SchemaError(f"{archive} member hash mismatches: {mismatches}")


def _validate_recursive_bundle(binding: dict[str, Any]) -> None:
    bundle = REPO_ROOT / str(binding["bundle_path"])
    output = _run_checked(
        ["git", "bundle", "verify", str(bundle)],
        purpose="Recursive git-bundle verification",
        cwd=REPO_ROOT,
    )
    if "complete history" not in output.lower():
        raise SchemaError("Recursive bundle does not report complete history")
    heads = _run_checked(
        ["git", "bundle", "list-heads", str(bundle)],
        purpose="Recursive git-bundle head listing",
        cwd=REPO_ROOT,
    )
    commit = str(binding["commit"])
    if commit not in heads:
        raise SchemaError("Recursive bundle does not advertise the captured commit")
    staging_parent = REPO_ROOT / "tmp"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="recursive_bundle_verify_",
        dir=staging_parent,
    ) as raw_directory:
        git_directory = Path(raw_directory) / "objects.git"
        _run_checked(
            ["git", "init", "--bare", "--quiet", str(git_directory)],
            purpose="temporary Recursive bundle repository initialization",
        )
        _run_checked(
            [
                "git",
                f"--git-dir={git_directory}",
                "fetch",
                "--quiet",
                "--no-tags",
                str(bundle),
                commit,
            ],
            purpose="Recursive bundle commit fetch",
        )
        tree = _run_checked(
            [
                "git",
                f"--git-dir={git_directory}",
                "rev-parse",
                f"{commit}^{{tree}}",
            ],
            purpose="Recursive bundle tree derivation",
        ).strip()
    if tree != binding["tree"]:
        raise SchemaError(
            "Recursive bundle tree mismatch: "
            f"expected {binding['tree']}, computed {tree}"
        )


def _validate_primary_literature_provenance() -> None:
    represented: set[str] = set()
    for record in all_new_evidence():
        if len(record.paper_ids) != 1:
            continue
        binding = literature_snapshot_binding(record.paper_ids[0])
        if binding is None:
            continue
        if dict(record.facts.get("primary_source_snapshot", {})) != binding:
            raise SchemaError(
                f"evidence {record.evidence_id!r} does not carry the exact "
                "reviewed primary-source binding"
            )
        required_paths = {
            str(binding[key])
            for key in ("manifest_path", "headers_path", "artifact_path")
        }
        if not required_paths.issubset(set(record.artifact_paths)):
            raise SchemaError(
                f"evidence {record.evidence_id!r} omits primary-source "
                f"artifacts {sorted(required_paths - set(record.artifact_paths))}"
            )
        represented.add(str(binding["manifest_paper_id"]))

    expected = set(LITERATURE_SNAPSHOT_BINDINGS)
    if len(expected) != 23:
        raise SchemaError(
            f"primary-literature archive must contain 23 records, got {len(expected)}"
        )
    if represented != expected:
        raise SchemaError(
            "reviewed evidence does not cover the complete primary-source "
            f"archive: missing={sorted(expected - represented)}, "
            f"unexpected={sorted(represented - expected)}"
        )

    pdf_count = 0
    html_count = 0
    observed_payload_hashes: set[str] = set()
    for paper_id, binding in sorted(LITERATURE_SNAPSHOT_BINDINGS.items()):
        manifest_path = REPO_ROOT / str(binding["manifest_path"])
        headers_path = REPO_ROOT / str(binding["headers_path"])
        artifact_path = REPO_ROOT / str(binding["artifact_path"])
        if _sha256_file(manifest_path) != binding["manifest_sha256"]:
            raise SchemaError(f"primary-source manifest hash mismatch: {paper_id}")
        if (
            headers_path.stat().st_size != binding["headers_bytes"]
            or _sha256_file(headers_path) != binding["headers_sha256"]
        ):
            raise SchemaError(f"primary-source header mismatch: {paper_id}")
        if (
            artifact_path.stat().st_size != binding["artifact_bytes"]
            or _sha256_file(artifact_path) != binding["artifact_sha256"]
        ):
            raise SchemaError(f"primary-source payload mismatch: {paper_id}")
        if binding["artifact_sha256"] in observed_payload_hashes:
            raise SchemaError(
                f"primary-source payload hash is duplicated: {paper_id}"
            )
        observed_payload_hashes.add(str(binding["artifact_sha256"]))

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("classification") != "external_primary_literature"
            or manifest.get("paper_id") != paper_id
            or manifest.get("title") != binding["title"]
            or manifest.get("source") != binding["source"]
            or manifest.get("artifact", {}).get("sha256")
            != binding["artifact_sha256"]
            or manifest.get("access_license") != binding["access_license"]
        ):
            raise SchemaError(
                f"primary-source manifest differs from reviewed binding: {paper_id}"
            )
        if binding["http_status_code"] != 200:
            raise SchemaError(f"primary-source HTTP status is not 200: {paper_id}")
        sensitive_headers = []
        for line in headers_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            name = line.partition(":")[0].strip().lower()
            if name in {
                "authorization",
                "proxy-authorization",
                "cookie",
                "set-cookie",
            }:
                sensitive_headers.append(name)
        if sensitive_headers:
            raise SchemaError(
                f"primary-source headers retain sensitive fields for {paper_id}: "
                f"{sensitive_headers}"
            )

        validation = dict(binding["validation"])
        if artifact_path.suffix == ".pdf":
            pdf_count += 1
            if artifact_path.read_bytes()[:5] != b"%PDF-":
                raise SchemaError(f"PDF magic mismatch: {paper_id}")
            for required in (
                "pdf_magic_valid",
                "pdfinfo_valid",
                "title_text_match",
                "first_page_visual_review_passed",
            ):
                if validation.get(required) is not True:
                    raise SchemaError(
                        f"PDF validation flag {required!r} failed: {paper_id}"
                    )
            page_count = int(validation.get("page_count", 0))
            if page_count <= 0:
                raise SchemaError(f"PDF page count is absent: {paper_id}")
            pdfinfo = _run_checked(
                ["pdfinfo", str(artifact_path)],
                purpose=f"PDF structure verification for {paper_id}",
            )
            pages_line = next(
                (
                    line
                    for line in pdfinfo.splitlines()
                    if line.lower().startswith("pages:")
                ),
                "",
            )
            if not pages_line or int(pages_line.split(":", 1)[1].strip()) != page_count:
                raise SchemaError(f"PDF page-count mismatch: {paper_id}")
        elif artifact_path.suffix == ".html":
            html_count += 1
            source = artifact_path.read_text(encoding="utf-8")
            if (
                validation.get("html_doctype_valid") is not True
                or validation.get("utf8_decode_valid") is not True
                or validation.get("title_text_match") is not True
                or not source.lstrip().lower().startswith("<!doctype html")
                or str(binding["title"]).lower() not in source.lower()
            ):
                raise SchemaError(f"HTML validation mismatch: {paper_id}")
        else:
            raise SchemaError(
                f"unsupported primary-source artifact type: {artifact_path}"
            )
    if (pdf_count, html_count) != (22, 1):
        raise SchemaError(
            "primary-literature archive must contain 22 PDFs and one HTML "
            f"artifact, got PDFs={pdf_count}, HTML={html_count}"
        )


def validate_artifact_provenance() -> None:
    represented_manifests: set[str] = set()
    for record in all_new_evidence():
        for relative in record.artifact_paths:
            path = (REPO_ROOT / relative).resolve()
            try:
                path.relative_to(REPO_ROOT.resolve())
            except ValueError as exc:
                raise SchemaError(
                    f"evidence {record.evidence_id!r} artifact escapes repository: "
                    f"{relative!r}"
                ) from exc
            if not path.is_file():
                raise SchemaError(
                    f"evidence {record.evidence_id!r} artifact is absent: {relative!r}"
                )
        if len(record.paper_ids) == 1:
            binding = code_snapshot_binding(record.paper_ids[0])
            if binding is not None:
                if dict(record.facts.get("code_snapshot", {})) != binding:
                    raise SchemaError(
                        f"evidence {record.evidence_id!r} does not carry the exact "
                        "reviewed code-snapshot binding"
                    )
                required_paths = {
                    str(binding[key])
                    for key in ("manifest_path", "archive_path", "bundle_path")
                    if binding.get(key)
                }
                if not required_paths.issubset(set(record.artifact_paths)):
                    raise SchemaError(
                        f"evidence {record.evidence_id!r} omits snapshot artifacts "
                        f"{sorted(required_paths - set(record.artifact_paths))}"
                    )
                represented_manifests.add(str(binding["manifest_paper_id"]))

    _validate_primary_literature_provenance()
    canonical_bindings = [
        dict(binding)
        for binding in CODE_SNAPSHOT_BINDINGS.values()
        if not binding.get("manifest_alias_of")
    ]
    expected_manifests = {
        str(binding["manifest_paper_id"]) for binding in canonical_bindings
    }
    if represented_manifests != expected_manifests:
        raise SchemaError(
            "reviewed evidence does not cover every prepared code snapshot: "
            f"missing={sorted(expected_manifests - represented_manifests)}, "
            f"unexpected={sorted(represented_manifests - expected_manifests)}"
        )

    for binding in canonical_bindings:
        manifest_path = REPO_ROOT / str(binding["manifest_path"])
        archive_path = REPO_ROOT / str(binding["archive_path"])
        if _sha256_file(manifest_path) != binding["manifest_sha256"]:
            raise SchemaError(
                f"manifest hash mismatch for {binding['manifest_paper_id']}"
            )
        if _sha256_file(archive_path) != binding["archive_sha256"]:
            raise SchemaError(
                f"archive hash mismatch for {binding['manifest_paper_id']}"
            )
        _run_checked(
            ["zstd", "-t", str(archive_path)],
            purpose=f"zstd integrity test for {binding['manifest_paper_id']}",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("paper_id") != binding["manifest_paper_id"]:
            raise SchemaError(
                f"manifest paper ID mismatch in {binding['manifest_path']}"
            )
        verification = dict(manifest.get("verification", {}))
        if verification.get("local_sota_adoption_claimed") is not False:
            raise SchemaError(
                f"{binding['manifest_paper_id']} must not claim local SOTA adoption"
            )
        artifacts = {
            str(item["path"]): dict(item)
            for item in manifest.get("artifacts", [])
        }
        for path_key, hash_key in (
            ("archive_path", "archive_sha256"),
            ("bundle_path", "bundle_sha256"),
        ):
            if not binding.get(path_key):
                continue
            artifact_path = REPO_ROOT / str(binding[path_key])
            expected_hash = str(binding[hash_key])
            if _sha256_file(artifact_path) != expected_hash:
                raise SchemaError(
                    f"artifact hash mismatch for {binding['manifest_paper_id']}: "
                    f"{artifact_path.name}"
                )
            manifest_artifact = artifacts.get(artifact_path.name)
            if manifest_artifact is None:
                raise SchemaError(
                    f"manifest omits artifact {artifact_path.name!r}"
                )
            if manifest_artifact.get("sha256") != expected_hash:
                raise SchemaError(
                    f"manifest artifact hash differs for {artifact_path.name!r}"
                )
            if manifest_artifact.get("bytes") != artifact_path.stat().st_size:
                raise SchemaError(
                    f"manifest artifact byte count differs for {artifact_path.name!r}"
                )

        schema_version = manifest.get("schema_version")
        verified_members = {
            str(item["path"]): str(item["sha256"])
            for item in manifest.get("verified_files", [])
        }
        if schema_version == 2:
            repository = dict(manifest.get("repository", {}))
            expected_repository = {
                "url": binding["repository"],
                "commit": binding["commit"],
                "tree": binding["tree"],
            }
            for key, expected in expected_repository.items():
                if repository.get(key) != expected:
                    raise SchemaError(
                        f"{binding['manifest_paper_id']} manifest repository "
                        f"{key} differs from the reviewed binding"
                    )
            license_record = dict(manifest.get("license", {}))
            if (
                license_record.get("declared_family") != binding["license_family"]
                or license_record.get("sha256") != binding["license_sha256"]
            ):
                raise SchemaError(
                    f"{binding['manifest_paper_id']} license metadata differs "
                    "from the reviewed binding"
                )
            if verification.get("archive_embedded_commit") != binding["commit"]:
                raise SchemaError(
                    f"{binding['manifest_paper_id']} archive commit differs"
                )
            archive_entry = artifacts[archive_path.name]
            archive_root = str(archive_entry.get("archive_root", ""))
            if not archive_root:
                raise SchemaError(
                    f"{binding['manifest_paper_id']} omits archive_root"
                )
        elif schema_version == 1:
            if (
                manifest.get("repository") != binding["repository"]
                or manifest.get("commit") != binding["commit"]
            ):
                raise SchemaError(
                    "Recursive v1 manifest repository or commit differs from binding"
                )
            if "omitted license metadata" not in binding["license_caveat"]:
                raise SchemaError(
                    "Recursive license caveat must disclose the v1 omission"
                )
            archive_root = str(binding["archive_root"])
            verified_members.update(
                {
                    "LICENSE": str(binding["license_sha256"]),
                    "NOTICE": str(binding["notice_sha256"]),
                    "nanoGPT_speedrun/LICENSE-modded-nanogpt": str(
                        binding["nested_mit_license_sha256"]["nanoGPT"]
                    ),
                    "nanochat_autoresearch/LICENSE-nanochat": str(
                        binding["nested_mit_license_sha256"]["nanochat"]
                    ),
                }
            )
            _validate_recursive_bundle(binding)
        else:
            raise SchemaError(
                f"unsupported external-code manifest schema {schema_version!r}"
            )
        _verify_archive_members(
            archive_path,
            archive_root=archive_root,
            expected=verified_members,
        )


def _record_json(record: Any) -> str:
    return canonical_json(record.to_dict())


def _check_internal_unique(records: Iterable[Any], kind: str) -> None:
    seen: dict[str, str] = {}
    for record in records:
        record_id = str(record.registry_id)
        payload = _record_json(record)
        if record_id in seen:
            raise SchemaError(f"draft contains duplicate {kind} ID: {record_id}")
        seen[record_id] = payload


def _existing_action(existing: dict[str, Any], record: Any, kind: str) -> str:
    record_id = str(record.registry_id)
    current = existing.get(record_id)
    if current is None:
        return "append"
    if _record_json(current) == _record_json(record):
        return "skip_exact"
    raise SchemaError(
        f"{kind} ID {record_id!r} already exists with different content; "
        "append-only ingestion refuses to overwrite or reinterpret it"
    )


def preflight(registry: ResearchRegistry) -> dict[str, list[tuple[str, str]]]:
    validate_review_binding()
    validate_artifact_provenance()
    papers = all_new_papers()
    claims = all_new_claims()
    evidence = all_new_evidence()
    _check_internal_unique(papers, "paper")
    _check_internal_unique(claims, "claim")
    _check_internal_unique(evidence, "evidence")
    evidence_per_new_claim = {record.claim_id: 0 for record in claims}
    for record in evidence:
        if len(record.paper_ids) != 1 or len(record.claim_ids) != 1:
            raise SchemaError(
                f"literature evidence {record.evidence_id!r} must assess exactly "
                "one paper/claim subject"
            )
        if record.claim_ids[0] in evidence_per_new_claim:
            evidence_per_new_claim[record.claim_ids[0]] += 1
    non_atomic_coverage = {
        claim_id: count
        for claim_id, count in evidence_per_new_claim.items()
        if count != 1
    }
    if non_atomic_coverage:
        raise SchemaError(
            "every new claim requires exactly one claim-specific evidence "
            f"assessment, got {non_atomic_coverage}"
        )
    for record in evidence:
        validate_rating(
            dict(record.facts["critic_rating"]),
            allow_pending=CRITIC_REVIEW["status"] != "reconciled",
        )

    existing_papers = registry.papers.by_id()
    existing_claims = registry.claims.by_id()
    existing_evidence = registry.literature_evidence.by_id()
    other_evidence_ids = (
        set(registry.run_evidence.by_id())
        | set(registry.observations.by_id())
    )
    collisions = {record.evidence_id for record in evidence} & other_evidence_ids
    if collisions:
        raise SchemaError(
            "draft literature evidence IDs collide with other evidence namespaces: "
            f"{sorted(collisions)}"
        )

    # Reject accidental duplicate primary URLs even when the proposed ID differs.
    existing_by_primary = {
        str(record.urls["primary"]): record.paper_id
        for record in existing_papers.values()
    }
    draft_by_primary: dict[str, str] = {}
    for record in papers:
        primary = str(record.urls["primary"])
        if primary in draft_by_primary:
            raise SchemaError(
                f"draft paper primary URL {primary!r} is shared by "
                f"{draft_by_primary[primary]!r} and {record.paper_id!r}"
            )
        draft_by_primary[primary] = record.paper_id
        prior = existing_by_primary.get(primary)
        if prior and prior != record.paper_id:
            allowed = INTENTIONAL_PRIMARY_URL_DUPLICATES.get(record.paper_id)
            if allowed != prior:
                raise SchemaError(
                    f"paper {record.paper_id!r} duplicates existing primary URL "
                    f"owned by {prior!r}: {primary}"
                )

    merged_paper_ids = set(existing_papers) | {record.paper_id for record in papers}
    merged_claim_ids = set(existing_claims) | {record.claim_id for record in claims}
    merged_evidence_ids = set(existing_evidence) | {
        record.evidence_id for record in evidence
    }

    for record in claims:
        if record.paper_id not in merged_paper_ids:
            raise SchemaError(
                f"claim {record.claim_id!r} refers to absent paper {record.paper_id!r}"
            )
    for record in evidence:
        absent_papers = set(record.paper_ids) - merged_paper_ids
        absent_claims = set(record.claim_ids) - merged_claim_ids
        if absent_papers or absent_claims:
            raise SchemaError(
                f"evidence {record.evidence_id!r} has absent references: "
                f"papers={sorted(absent_papers)}, claims={sorted(absent_claims)}"
            )
        claim_owners = {
            (
                existing_claims.get(claim_id)
                or next(
                    candidate
                    for candidate in claims
                    if candidate.claim_id == claim_id
                )
            ).paper_id
            for claim_id in record.claim_ids
        }
        if not claim_owners.issubset(set(record.paper_ids)):
            raise SchemaError(
                f"evidence {record.evidence_id!r} omits claim-owner papers "
                f"{sorted(claim_owners - set(record.paper_ids))}"
            )
        if (
            record.supersedes_evidence_id
            and record.supersedes_evidence_id not in merged_evidence_ids
        ):
            raise SchemaError(
                f"evidence {record.evidence_id!r} supersedes absent "
                f"{record.supersedes_evidence_id!r}"
            )
        if record.supersedes_evidence_id:
            prior = existing_evidence.get(record.supersedes_evidence_id)
            if prior is None:
                prior = next(
                    candidate
                    for candidate in evidence
                    if candidate.evidence_id == record.supersedes_evidence_id
                )
            if record.evidence_id == prior.evidence_id:
                raise SchemaError("evidence cannot supersede itself")
            if not (set(record.paper_ids) & set(prior.paper_ids)):
                raise SchemaError(
                    f"supersession {record.evidence_id!r} changes every paper subject"
                )
            if not (set(record.claim_ids) & set(prior.claim_ids)):
                raise SchemaError(
                    f"supersession {record.evidence_id!r} changes every claim subject"
                )
            if prior.hypothesis_ids and not (
                set(record.hypothesis_ids) & set(prior.hypothesis_ids)
            ):
                raise SchemaError(
                    f"supersession {record.evidence_id!r} drops all hypothesis subjects"
                )

    successors: dict[str, str] = {}
    for candidate in [*existing_evidence.values(), *evidence]:
        predecessor = candidate.supersedes_evidence_id
        if not predecessor:
            continue
        if predecessor in successors and successors[predecessor] != candidate.evidence_id:
            raise SchemaError(
                f"supersession branches at {predecessor!r}: "
                f"{successors[predecessor]!r}, {candidate.evidence_id!r}"
            )
        successors[predecessor] = candidate.evidence_id
    for start in successors:
        seen: set[str] = set()
        cursor = start
        while cursor in successors:
            if cursor in seen:
                raise SchemaError(f"supersession cycle includes {cursor!r}")
            seen.add(cursor)
            cursor = successors[cursor]

    return {
        "papers": [
            (record.paper_id, _existing_action(existing_papers, record, "paper"))
            for record in papers
        ],
        "claims": [
            (record.claim_id, _existing_action(existing_claims, record, "claim"))
            for record in claims
        ],
        "literature_evidence": [
            (
                record.evidence_id,
                _existing_action(existing_evidence, record, "literature evidence"),
            )
            for record in evidence
        ],
    }


def add_idempotent(store: Any, record: Any, kind: str) -> str:
    # This helper is used only inside an isolated staged research copy.  The
    # global transaction lock and source-tree precondition protect live commit.
    existing = store.by_id()
    action = _existing_action(existing, record, kind)
    if action == "skip_exact":
        return action
    store.add(record)
    return "appended"


def summarize(plan: dict[str, list[tuple[str, str]]]) -> dict[str, Any]:
    return {
        kind: {
            "total": len(actions),
            "append": sum(action in {"append", "appended"} for _, action in actions),
            "skip_exact": sum(action == "skip_exact" for _, action in actions),
            "records": [
                {"id": record_id, "action": action}
                for record_id, action in actions
            ],
        }
        for kind, actions in plan.items()
    }


def _script_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _record_hashes() -> dict[str, dict[str, str]]:
    groups = {
        "papers": all_new_papers(),
        "claims": all_new_claims(),
        "literature_evidence": all_new_evidence(),
    }
    return {
        kind: {
            str(record.registry_id): hashlib.sha256(
                _record_json(record).encode("utf-8")
            ).hexdigest()
            for record in records
        }
        for kind, records in groups.items()
    }


def _tree_sha256(root: Path) -> str:
    """Hash path names, file bytes, and symlink targets for a coherent snapshot."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(b"L\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(os.readlink(path).encode("utf-8"))
            digest.update(b"\0")
        elif path.is_file():
            digest.update(b"F\0")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_file(path).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _blocking_audit_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(issue)
        for issue in report.get("issues", [])
        if str(issue.get("severity", "")).lower()
        not in {"", "info", "informational"}
    ]


def _validate_registry_snapshot(
    registry: ResearchRegistry,
    *,
    check_generated_state: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = registry.validate(
        check_generated_state=check_generated_state
    ).to_dict()
    if validation.get("warnings"):
        raise SchemaError(
            f"strict registry validation emitted warnings: "
            f"{validation['warnings']}"
        )
    audit = registry.audit().to_dict()
    blocking = _blocking_audit_issues(audit)
    if blocking:
        raise SchemaError(f"registry audit has blocking issues: {blocking}")
    return validation, audit


def _live_prevalidation(
    registry: ResearchRegistry,
) -> tuple[
    dict[str, list[tuple[str, str]]],
    dict[str, Any],
    dict[str, Any],
]:
    # Generated views are deliberately excluded from the engine patch and are
    # regenerated only in the coherent stage.  Prevalidate authoritative live
    # sources here.  The two uncovered legacy Kimi claims are exactly the
    # claim-specific weak records in this payload.  No other warning is allowed;
    # the stage and postcommit checks require a warning-free generated snapshot.
    validation = registry.validate(check_generated_state=False).to_dict()
    if validation.get("warnings"):
        raise SchemaError(
            f"live authoritative validation emitted warnings: "
            f"{validation['warnings']}"
        )
    audit = registry.audit().to_dict()
    allowed_warning_codes = {
        "literature_claims_without_evidence",
        "stale_literature_synthesis",
        "stale_research_state",
    }
    blocking = [
        issue
        for issue in _blocking_audit_issues(audit)
        if str(issue.get("code", "")) not in allowed_warning_codes
    ]
    if blocking:
        raise SchemaError(f"live registry audit has blocking issues: {blocking}")
    observed_warning_codes = {
        str(issue.get("code", ""))
        for issue in audit.get("issues", [])
        if str(issue.get("severity", "")).lower() == "warning"
    }
    # The exact bridge warnings are expected before the first ingestion.  An
    # idempotent replay sees the already-integrated, warning-free registry.
    if observed_warning_codes not in (set(), allowed_warning_codes):
        raise SchemaError(
            "live audit warnings differ from the reviewed pre-stage bridge: "
            f"expected={sorted(allowed_warning_codes)}, "
            f"observed={sorted(observed_warning_codes)}"
        )
    plan = preflight(registry)
    return plan, validation, audit


@contextmanager
def _global_transaction_lock() -> Iterable[None]:
    TRANSACTION_ROOT.mkdir(parents=True, exist_ok=True)
    with GLOBAL_LOCK_PATH.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SchemaError(
                f"literature-ingest transaction lock is held: {GLOBAL_LOCK_PATH}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _transaction_relpaths() -> tuple[list[str], str, str]:
    payload_digest = reviewed_payload_sha256()
    durable_manifest = (
        f"refinement/literature_ingests/{payload_digest}.json"
    )
    durable_receipt = (
        f"refinement/literature_ingest_journals/{payload_digest}.json"
    )
    ordered = [
        ResearchRegistry.PATHS["papers"],
        ResearchRegistry.PATHS["claims"],
        ResearchRegistry.PATHS["literature_evidence"],
        ResearchRegistry.LITERATURE_SYNTHESIS_PATH,
        "knowledge/RESEARCH_STATE.md",
        durable_manifest,
        durable_receipt,
    ]
    return ordered, durable_manifest, durable_receipt


def _canonical_snapshot_bindings() -> list[dict[str, Any]]:
    return [
        dict(binding)
        for binding in CODE_SNAPSHOT_BINDINGS.values()
        if not binding.get("manifest_alias_of")
    ]


def _canonical_literature_snapshot_bindings() -> list[dict[str, Any]]:
    return [
        dict(binding)
        for _, binding in sorted(LITERATURE_SNAPSHOT_BINDINGS.items())
    ]


def _write_review_manifest(
    *,
    stage_root: Path,
    relative_path: str,
    plan: dict[str, list[tuple[str, str]]],
    source_tree_sha256: str,
    source_project_dependencies_sha256: str,
    validation: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    payload = canonical_reviewed_payload()
    payload_digest = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    if payload_digest != reviewed_payload_sha256():
        raise SchemaError("canonical reviewed payload changed during staging")
    target_relpaths, _, durable_receipt = _transaction_relpaths()
    manifest = {
        "schema_version": 1,
        "ingest_id": f"literature_ingest_{payload_digest[:20]}",
        "created_at": CREATED_AT,
        "created_by": CREATED_BY,
        "critic_review": CRITIC_REVIEW,
        "reviewed_payload_sha256": payload_digest,
        "reviewed_payload": payload,
        "record_sha256": _record_hashes(),
        "script": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": _script_sha256(),
        },
        "code_snapshot_bindings": _canonical_snapshot_bindings(),
        "primary_literature_snapshot_bindings": (
            _canonical_literature_snapshot_bindings()
        ),
        "primary_literature_snapshot_aliases": (
            LITERATURE_SNAPSHOT_ALIASES
        ),
        "code_snapshot_aliases": {
            paper_id: str(binding["manifest_alias_of"])
            for paper_id, binding in CODE_SNAPSHOT_BINDINGS.items()
            if binding.get("manifest_alias_of")
        },
        "live_source_tree_sha256": source_tree_sha256,
        "live_project_dependencies_sha256": (
            source_project_dependencies_sha256
        ),
        "project_dependency_paths": _project_dependency_paths(),
        "plan": summarize(plan),
        "staged_validation": validation,
        "staged_audit": audit,
        "transaction": {
            "global_lock": str(GLOBAL_LOCK_PATH.relative_to(REPO_ROOT)),
            "ordered_target_paths": target_relpaths,
            "durable_commit_receipt": durable_receipt,
            "policy": (
                "prevalidate live; copy and mutate an isolated research tree; "
                "render, strictly validate, and audit the coherent stage; verify "
                "the live-tree precondition; journal backups; promote explicit "
                "files atomically; validate/audit again; roll back on failure"
            ),
            "all_writers_must_honor_global_lock": True,
        },
    }
    atomic_write_text(
        stage_root / relative_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _append_records_to_stage(
    registry: ResearchRegistry,
) -> dict[str, list[tuple[str, str]]]:
    result: dict[str, list[tuple[str, str]]] = {
        "papers": [],
        "claims": [],
        "literature_evidence": [],
    }
    for record in all_new_papers():
        result["papers"].append(
            (record.paper_id, add_idempotent(registry.papers, record, "paper"))
        )
    for record in all_new_claims():
        result["claims"].append(
            (record.claim_id, add_idempotent(registry.claims, record, "claim"))
        )
    for record in all_new_evidence():
        result["literature_evidence"].append(
            (
                record.evidence_id,
                add_idempotent(
                    registry.literature_evidence,
                    record,
                    "literature evidence",
                ),
            )
        )
    return result


def _project_dependency_paths() -> list[str]:
    """Return every project-root path resolved by validation/rendering."""
    live_registry = ResearchRegistry(RESEARCH_ROOT)
    setup = live_registry.setup_reconciliation()
    manifest = live_registry._manifest()
    relative_paths = set(setup.frozen_files)
    relative_paths.update(str(path) for path in setup.baseline["artifact_paths"])
    reference_path = str(setup.reference_code.get("local_path", ""))
    if reference_path:
        relative_paths.add(reference_path)
    relative_paths.add(str(manifest["campaign_ledger"]["source_path"]))
    return sorted(relative_paths)


def _project_dependency_sha256(project_root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        path = project_root / relative
        if not path.is_file():
            raise SchemaError(f"missing project dependency: {relative!r}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_stage_project_dependencies(
    stage_project_root: Path,
    *,
    paths: list[str],
) -> None:
    """Copy the immutable/project ledgers that registry validation resolves."""
    for relative in paths:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SchemaError(f"unsafe staged project dependency: {relative!r}")
        source = REPO_ROOT / candidate
        if not source.is_file():
            raise SchemaError(f"missing staged project dependency: {relative!r}")
        destination = stage_project_root / candidate
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if _sha256_file(source) != _sha256_file(destination):
            raise SchemaError(
                f"staged project dependency hash mismatch: {relative!r}"
            )


def _build_staged_snapshot(
    *,
    workspace: Path,
    live_plan: dict[str, list[tuple[str, str]]],
    source_tree_sha256: str,
    source_project_dependencies_sha256: str,
) -> dict[str, Any]:
    stage_project_root = workspace / "project"
    stage_root = stage_project_root / "research"
    dependency_paths = _project_dependency_paths()
    shutil.copytree(RESEARCH_ROOT, stage_root)
    _copy_stage_project_dependencies(
        stage_project_root,
        paths=dependency_paths,
    )
    source_after_copy = _tree_sha256(RESEARCH_ROOT)
    stage_initial = _tree_sha256(stage_root)
    if (
        source_after_copy != source_tree_sha256
        or stage_initial != source_tree_sha256
    ):
        raise SchemaError(
            "live research changed during staging; refusing an incoherent snapshot"
        )
    if (
        _project_dependency_sha256(REPO_ROOT, dependency_paths)
        != source_project_dependencies_sha256
        or _project_dependency_sha256(stage_project_root, dependency_paths)
        != source_project_dependencies_sha256
    ):
        raise SchemaError(
            "project dependencies changed during staging; refusing an "
            "incoherent snapshot"
        )
    stage_registry = ResearchRegistry(stage_root)
    stage_plan = preflight(stage_registry)
    if stage_plan != live_plan:
        raise SchemaError("staged preflight plan differs from locked live preflight")

    staged_actions = _append_records_to_stage(stage_registry)
    stage_registry.write_literature_synthesis()
    stage_registry.write_state()
    validation, audit = _validate_registry_snapshot(stage_registry)

    target_relpaths, durable_manifest, durable_receipt = _transaction_relpaths()
    review_manifest = _write_review_manifest(
        stage_root=stage_root,
        relative_path=durable_manifest,
        plan=live_plan,
        source_tree_sha256=source_tree_sha256,
        source_project_dependencies_sha256=(
            source_project_dependencies_sha256
        ),
        validation=validation,
        audit=audit,
    )
    manifest_sha256 = _sha256_file(stage_root / durable_manifest)
    receipt = {
        "schema_version": 1,
        "state": "committed",
        "ingest_id": review_manifest["ingest_id"],
        "reviewed_payload_sha256": reviewed_payload_sha256(),
        "script_sha256": _script_sha256(),
        "durable_manifest_path": durable_manifest,
        "durable_manifest_sha256": manifest_sha256,
        "ordered_target_paths": target_relpaths,
        "staged_validation": validation,
        "staged_audit": audit,
        "recovery_policy": (
            "This receipt is promoted last. Until then the external transaction "
            "journal is authoritative and restores every explicit target from "
            "pre-commit backups after interruption or failed postvalidation."
        ),
    }
    atomic_write_text(
        stage_root / durable_receipt,
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    final_validation, final_audit = _validate_registry_snapshot(stage_registry)
    stage_hashes = {
        relative: _sha256_file(stage_root / relative)
        for relative in target_relpaths
    }
    return {
        "stage_root": stage_root,
        "source_tree_sha256": source_tree_sha256,
        "source_project_dependencies_sha256": (
            source_project_dependencies_sha256
        ),
        "project_dependency_paths": dependency_paths,
        "stage_tree_sha256": _tree_sha256(stage_root),
        "actions": staged_actions,
        "validation": final_validation,
        "audit": final_audit,
        "target_relpaths": target_relpaths,
        "stage_hashes": stage_hashes,
        "durable_manifest_path": durable_manifest,
        "durable_manifest_sha256": manifest_sha256,
        "durable_receipt_path": durable_receipt,
        "durable_receipt_sha256": stage_hashes[durable_receipt],
    }


def _safe_research_target(relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SchemaError(f"unsafe transaction target: {relative!r}")
    permitted_exact = {
        ResearchRegistry.PATHS["papers"],
        ResearchRegistry.PATHS["claims"],
        ResearchRegistry.PATHS["literature_evidence"],
        ResearchRegistry.LITERATURE_SYNTHESIS_PATH,
        "knowledge/RESEARCH_STATE.md",
    }
    permitted_prefixes = (
        "refinement/literature_ingests/",
        "refinement/literature_ingest_journals/",
    )
    if relative not in permitted_exact and not relative.startswith(permitted_prefixes):
        raise SchemaError(f"transaction target is outside the allowlist: {relative!r}")
    target = (RESEARCH_ROOT / candidate).resolve()
    try:
        target.relative_to(RESEARCH_ROOT.resolve())
    except ValueError as exc:
        raise SchemaError(f"transaction target escapes research: {relative!r}") from exc
    return target


def _path_state(relative: str) -> dict[str, Any]:
    target = _safe_research_target(relative)
    if not target.exists():
        return {"exists": False, "sha256": ""}
    if not target.is_file():
        raise SchemaError(f"transaction target is not a file: {relative!r}")
    return {"exists": True, "sha256": _sha256_file(target)}


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.ingest-{os.getpid()}.tmp"
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    try:
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some filesystems do not permit fsync on directory descriptors.  The
        # file itself was still fsynced before the atomic replacement.
        pass


def _write_transaction_journal(path: Path, journal: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(journal, indent=2, sort_keys=True) + "\n",
    )


def _rollback_from_journal(
    journal_path: Path,
    journal: dict[str, Any],
    *,
    recovered_after_restart: bool,
) -> None:
    for entry in reversed(list(journal["targets"])):
        relative = str(entry["relative_path"])
        target = _safe_research_target(relative)
        before = dict(entry["before"])
        if before["exists"]:
            backup = journal_path.parent / str(entry["backup_relative"])
            if (
                not backup.is_file()
                or _sha256_file(backup) != before["sha256"]
            ):
                raise SchemaError(
                    f"cannot recover {relative!r}: backup is absent or corrupt"
                )
            _atomic_copy(backup, target)
        elif target.exists():
            if not target.is_file():
                raise SchemaError(
                    f"cannot remove non-file recovery target {relative!r}"
                )
            target.unlink()
    journal["state"] = (
        "rolled_back_after_restart"
        if recovered_after_restart
        else "rolled_back_after_failure"
    )
    journal["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
    _write_transaction_journal(journal_path, journal)


def _recover_interrupted_transactions() -> list[str]:
    recovered: list[str] = []
    if not TRANSACTION_ROOT.exists():
        return recovered
    for journal_path in sorted(TRANSACTION_ROOT.glob("txn_*/journal.json")):
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        state = str(journal.get("state", ""))
        if state == "committing":
            _rollback_from_journal(
                journal_path,
                journal,
                recovered_after_restart=True,
            )
            recovered.append(str(journal_path.relative_to(REPO_ROOT)))
        elif state == "prepared":
            journal["state"] = "aborted_before_commit"
            journal["aborted_at"] = datetime.now(timezone.utc).isoformat()
            _write_transaction_journal(journal_path, journal)
    return recovered


def _assert_no_interrupted_transaction() -> None:
    if not TRANSACTION_ROOT.exists():
        return
    pending: list[str] = []
    for journal_path in sorted(TRANSACTION_ROOT.glob("txn_*/journal.json")):
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal.get("state") in {"prepared", "committing"}:
            pending.append(str(journal_path.relative_to(REPO_ROOT)))
    if pending:
        raise SchemaError(
            "stage-check refuses while recovery journals are incomplete; "
            f"a reconciled --apply must recover them first: {pending}"
        )


def _commit_staged_snapshot(
    staged: dict[str, Any],
    *,
    source_tree_sha256: str,
    source_project_dependencies_sha256: str,
) -> dict[str, Any]:
    if _tree_sha256(RESEARCH_ROOT) != source_tree_sha256:
        raise SchemaError(
            "live research changed after staging; no live mutation was performed"
        )
    dependency_paths = list(staged["project_dependency_paths"])
    if (
        _project_dependency_sha256(REPO_ROOT, dependency_paths)
        != source_project_dependencies_sha256
    ):
        raise SchemaError(
            "project dependencies changed after staging; no live mutation was "
            "performed"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    transaction_directory = TRANSACTION_ROOT / (
        f"txn_{reviewed_payload_sha256()[:16]}_{timestamp}"
    )
    backup_root = transaction_directory / "backups"
    backup_root.mkdir(parents=True, exist_ok=False)
    targets: list[dict[str, Any]] = []
    for relative in staged["target_relpaths"]:
        target = _safe_research_target(relative)
        before = _path_state(relative)
        backup_relative = ""
        if before["exists"]:
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            if _sha256_file(backup) != before["sha256"]:
                raise SchemaError(f"backup verification failed for {relative!r}")
            backup_relative = str(backup.relative_to(transaction_directory))
        targets.append(
            {
                "relative_path": relative,
                "before": before,
                "after_sha256": staged["stage_hashes"][relative],
                "backup_relative": backup_relative,
            }
        )
    journal_path = transaction_directory / "journal.json"
    journal: dict[str, Any] = {
        "schema_version": 1,
        "state": "prepared",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_payload_sha256": reviewed_payload_sha256(),
        "script_sha256": _script_sha256(),
        "source_tree_sha256": source_tree_sha256,
        "source_project_dependencies_sha256": (
            source_project_dependencies_sha256
        ),
        "stage_tree_sha256": staged["stage_tree_sha256"],
        "targets": targets,
        "promoted_paths": [],
    }
    _write_transaction_journal(journal_path, journal)
    if (
        _tree_sha256(RESEARCH_ROOT) != source_tree_sha256
        or _project_dependency_sha256(REPO_ROOT, dependency_paths)
        != source_project_dependencies_sha256
    ):
        journal["state"] = "aborted_before_commit"
        journal["aborted_at"] = datetime.now(timezone.utc).isoformat()
        _write_transaction_journal(journal_path, journal)
        raise SchemaError(
            "live research changed while backups were prepared; no live mutation "
            "was performed"
        )

    journal["state"] = "committing"
    journal["commit_started_at"] = datetime.now(timezone.utc).isoformat()
    _write_transaction_journal(journal_path, journal)
    try:
        for relative in staged["target_relpaths"]:
            source = Path(staged["stage_root"]) / relative
            target = _safe_research_target(relative)
            _atomic_copy(source, target)
            if _sha256_file(target) != staged["stage_hashes"][relative]:
                raise SchemaError(f"post-copy hash mismatch for {relative!r}")
            journal["promoted_paths"].append(relative)
            _write_transaction_journal(journal_path, journal)

        live_registry = ResearchRegistry(RESEARCH_ROOT)
        validation, audit = _validate_registry_snapshot(live_registry)
        final_plan = preflight(live_registry)
        if any(
            action != "skip_exact"
            for actions in final_plan.values()
            for _, action in actions
        ):
            raise SchemaError(
                "post-commit preflight does not resolve every reviewed record exactly"
            )
        if _tree_sha256(RESEARCH_ROOT) != staged["stage_tree_sha256"]:
            raise SchemaError(
                "post-commit research tree differs from the validated stage"
            )
        if (
            _project_dependency_sha256(REPO_ROOT, dependency_paths)
            != source_project_dependencies_sha256
        ):
            raise SchemaError(
                "project dependencies changed during commit"
            )
    except Exception:
        _rollback_from_journal(
            journal_path,
            journal,
            recovered_after_restart=False,
        )
        raise

    journal["state"] = "committed"
    journal["committed_at"] = datetime.now(timezone.utc).isoformat()
    journal["postvalidation"] = validation
    journal["postaudit"] = audit
    _write_transaction_journal(journal_path, journal)
    return {
        "validation": validation,
        "audit": audit,
        "recovery_journal": str(journal_path.relative_to(REPO_ROOT)),
        "recovery_journal_sha256": _sha256_file(journal_path),
    }


def apply_records() -> dict[str, Any]:
    if CRITIC_REVIEW["status"] != "reconciled":
        raise SchemaError(
            "--apply refused: CRITIC_REVIEW.status is pending. Required engine "
            f"and provenance capabilities remain explicit in apply_blockers: "
            f"{CRITIC_REVIEW['apply_blockers']}"
        )
    if not CRITIC_REVIEW["reviewer"] or not CRITIC_REVIEW["reviewed_at"]:
        raise SchemaError(
            "--apply refused: reconciled review requires reviewer and reviewed_at"
        )
    if CRITIC_REVIEW["apply_blockers"]:
        raise SchemaError(
            "--apply refused: a reconciled review cannot retain apply blockers"
        )
    for record in all_new_evidence():
        validate_rating(dict(record.facts["critic_rating"]), allow_pending=False)
    with _global_transaction_lock():
        recovered = _recover_interrupted_transactions()
        live_registry = ResearchRegistry(RESEARCH_ROOT)
        plan, live_validation, live_audit = _live_prevalidation(live_registry)
        source_tree_sha256 = _tree_sha256(RESEARCH_ROOT)
        dependency_paths = _project_dependency_paths()
        source_project_dependencies_sha256 = _project_dependency_sha256(
            REPO_ROOT,
            dependency_paths,
        )
        with tempfile.TemporaryDirectory(
            prefix="stage_",
            dir=TRANSACTION_ROOT,
        ) as raw_workspace:
            staged = _build_staged_snapshot(
                workspace=Path(raw_workspace),
                live_plan=plan,
                source_tree_sha256=source_tree_sha256,
                source_project_dependencies_sha256=(
                    source_project_dependencies_sha256
                ),
            )
            committed = _commit_staged_snapshot(
                staged,
                source_tree_sha256=source_tree_sha256,
                source_project_dependencies_sha256=(
                    source_project_dependencies_sha256
                ),
            )
        return {
            "recovered_transactions": recovered,
            "live_prevalidation": live_validation,
            "live_preaudit": live_audit,
            "plan": summarize(plan),
            "staged_actions": summarize(staged["actions"]),
            "staged_validation": staged["validation"],
            "staged_audit": staged["audit"],
            "durable_manifest_path": staged["durable_manifest_path"],
            "durable_manifest_sha256": staged["durable_manifest_sha256"],
            "durable_receipt_path": staged["durable_receipt_path"],
            "durable_receipt_sha256": staged["durable_receipt_sha256"],
            "commit": committed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stage-check",
        action="store_true",
        help=(
            "Build, append, render, validate, and audit an isolated research "
            "snapshot under the global lock, then discard it without changing "
            "research/. This remains available while critic status is pending."
        ),
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Commit the globally locked, journaled staged transaction. Refuses "
            "while CRITIC_REVIEW is pending or any apply blocker remains."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.apply:
        applied = apply_records()
        print(
            json.dumps(
                {
                    "mode": "apply",
                    "research_root": str(RESEARCH_ROOT),
                    "critic_review": CRITIC_REVIEW,
                    "result": applied,
                    "generated_views_updated": True,
                    "research_transaction_committed": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.stage_check:
        with _global_transaction_lock():
            _assert_no_interrupted_transaction()
            live_registry = ResearchRegistry(RESEARCH_ROOT)
            plan, live_validation, live_audit = _live_prevalidation(live_registry)
            source_tree_sha256 = _tree_sha256(RESEARCH_ROOT)
            dependency_paths = _project_dependency_paths()
            source_project_dependencies_sha256 = _project_dependency_sha256(
                REPO_ROOT,
                dependency_paths,
            )
            with tempfile.TemporaryDirectory(
                prefix="stage_check_",
                dir=TRANSACTION_ROOT,
            ) as raw_workspace:
                staged = _build_staged_snapshot(
                    workspace=Path(raw_workspace),
                    live_plan=plan,
                    source_tree_sha256=source_tree_sha256,
                    source_project_dependencies_sha256=(
                        source_project_dependencies_sha256
                    ),
                )
                if (
                    _tree_sha256(RESEARCH_ROOT) != source_tree_sha256
                    or _project_dependency_sha256(REPO_ROOT, dependency_paths)
                    != source_project_dependencies_sha256
                ):
                    raise SchemaError(
                        "live research changed during stage-check; no live mutation "
                        "was performed"
                    )
            output = {
                "mode": "stage_check",
                "research_root": str(RESEARCH_ROOT),
                "critic_review": CRITIC_REVIEW,
                "live_validation": live_validation,
                "live_audit": live_audit,
                "plan": summarize(plan),
                "staged_actions": summarize(staged["actions"]),
                "staged_validation": staged["validation"],
                "staged_audit": staged["audit"],
                "source_tree_sha256": source_tree_sha256,
                "source_project_dependencies_sha256": (
                    source_project_dependencies_sha256
                ),
                "stage_tree_sha256": staged["stage_tree_sha256"],
                "durable_manifest_preview": {
                    "path": staged["durable_manifest_path"],
                    "sha256": staged["durable_manifest_sha256"],
                },
                "durable_receipt_preview": {
                    "path": staged["durable_receipt_path"],
                    "sha256": staged["durable_receipt_sha256"],
                },
                "research_writes_performed": False,
                "temporary_stage_discarded": True,
            }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    registry = ResearchRegistry(RESEARCH_ROOT)
    plan, live_validation, live_audit = _live_prevalidation(registry)
    print(
        json.dumps(
            {
                "mode": "dry_run",
                "research_root": str(RESEARCH_ROOT),
                "critic_review": CRITIC_REVIEW,
                "live_validation": live_validation,
                "live_audit": live_audit,
                "plan": summarize(plan),
                "reviewed_payload_sha256": reviewed_payload_sha256(),
                "script_sha256": _script_sha256(),
                "research_writes_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
