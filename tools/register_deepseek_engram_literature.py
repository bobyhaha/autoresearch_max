#!/usr/bin/env python3
"""Register the reviewed DeepSeek DSE/Engram literature and code capture.

The ACL 2026 proceedings paper calls the module Deep Sparse Embedding (DSE).
The later arXiv/repository version calls it Engram.  This append helper keeps
that version distinction explicit while binding the already-captured paper and
code artifacts to typed PaperRecord, ClaimRecord, and EvidenceRecord objects.

The default mode is a read-only preflight.  ``--apply`` appends missing records
under the repository-wide literature-ingest lock.  Replays are idempotent only
when every existing record is byte-for-byte equivalent after canonical typed
serialization; an ID collision with different content fails closed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "vibeautoresearch").is_dir():
    if REPO_ROOT.parent == REPO_ROOT:
        raise RuntimeError("could not locate the vibeautoresearch repository root")
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vibeautoresearch.core import SchemaError, canonical_json  # noqa: E402
from vibeautoresearch.knowledge import (  # noqa: E402
    ClaimRecord,
    EvidenceRecord,
    PaperRecord,
)
from vibeautoresearch.registry import ResearchRegistry  # noqa: E402


RESEARCH_ROOT = REPO_ROOT / "research"
ARTIFACT_ROOT = (
    RESEARCH_ROOT
    / "experiments"
    / "artifacts"
    / "external_code"
    / "pap_deepseek_engram_2026"
)
LOCK_PATH = (
    REPO_ROOT / "tmp" / "literature_ingest_transactions" / "global.lock"
)
CREATED_AT = "2026-07-29T14:48:30Z"
CREATED_BY = "codex_engram_literature_registration"

PAPER_ID = "pap_deepseek_engram_2026"
CLAIM_PROJECTION = "clm_deepseek_engram_canonical_token_projection"
CLAIM_REDUCTION = "clm_deepseek_engram_128k_projection_reduction"
CLAIM_ABLATION = "clm_deepseek_engram_tokenizer_compression_ablation"
CLAIM_BRANCH_GATE = "clm_deepseek_engram_branch_specific_gating"

ARTIFACTS: dict[str, dict[str, Any]] = {
    "manifest.json": {
        "bytes": 3648,
        "sha256": "8bf752884195b74c511c63a53a5d2c00264e53354ec5dcb52b56f6c9ed7aad2d",
    },
    "engram-fb7f84a2.tar.zst": {
        "bytes": 1784714,
        "sha256": "789b81408ab98b4cb90a2b00f4f6a15ded1651f13972cd4498edcd5b52ac1061",
    },
    "engram-fb7f84a2.bundle": {
        "bytes": 2352059,
        "sha256": "03114f299985aba9267b79299a2331d96194c0d50eb89a5f6e6b5eec3d40c0f0",
    },
    "Engram_paper.pdf": {
        "bytes": 770776,
        "sha256": "1e8ae53dc527d4db264b40a7996780b0044914443c0ce5fd54f898c405cab94f",
    },
}

COMMIT = "fb7f84a21f91223715394a33a1dc24bbfb7f788e"
TREE = "9d4076466f3651aca4a8da8181c2f1ffa5b181db"

COMMON_ARTIFACT_PATHS = tuple(
    (
        "research/experiments/artifacts/external_code/"
        f"pap_deepseek_engram_2026/{name}"
    )
    for name in (
        "manifest.json",
        "Engram_paper.pdf",
        "engram-fb7f84a2.tar.zst",
        "engram-fb7f84a2.bundle",
    )
)

LOCAL_IDEA_RATING = {
    "scale": "1_to_5_each",
    "maximum": 35,
    "dimensions": {
        "novelty": 4,
        "provenance": 5,
        "validity": 4,
        "impact": 3,
        "reliability": 3,
        "feasibility_cost": 5,
        "falsifiability": 5,
    },
    "total": 29,
    "status": "independent_multiagent_critic_reconciled",
    "subject": (
        "local canonicalized n-gram memory keys, not the full DSE/Engram model"
    ),
}

VERSION_FACTS = {
    "acl_primary": {
        "name": "Deep Sparse Embedding (DSE)",
        "anthology_id": "2026.acl-long.226",
        "doi": "10.18653/v1/2026.acl-long.226",
        "pages": "4968-4990",
        "peer_reviewed": True,
    },
    "captured_update": {
        "name": "Engram",
        "paper_sha256": ARTIFACTS["Engram_paper.pdf"]["sha256"],
        "repository": "https://github.com/deepseek-ai/Engram",
        "commit": COMMIT,
        "tree": TREE,
    },
    "version_caveat": (
        "The ACL proceedings paper names the module DSE. The captured PDF and "
        "official repository are a later Engram-named version; claims below "
        "identify which source version supplies each precision level."
    ),
}

RATING_CAVEAT = (
    "The 29/35 score is the independent critic's rating of a local keys-only "
    "retrofit. Institutional and conference prestige affects provenance only; "
    "it does not establish local causal transfer, adoption, or SOTA."
)


def paper() -> PaperRecord:
    return PaperRecord(
        paper_id=PAPER_ID,
        title=(
            "Conditional Memory via Scalable Lookup: "
            "A New Axis of Sparsity for Large Language Models"
        ),
        authors=(
            "Xin Cheng",
            "Wangding Zeng",
            "Damai Dai",
            "Qinyu Chen",
            "Bingxuan Wang",
            "Zhenda Xie",
            "Kezhao Huang",
            "Xingkai Yu",
            "Zhewen Hao",
            "Han Zhang",
            "Yu-Kun Li",
            "Huishuai Zhang",
            "Dongyan Zhao",
            "Wenfeng Liang",
        ),
        year=2026,
        venue={
            "name": "ACL 2026, Volume 1: Long Papers",
            "peer_reviewed": True,
        },
        urls={
            "primary": "https://aclanthology.org/2026.acl-long.226/",
            "doi": "https://doi.org/10.18653/v1/2026.acl-long.226",
            "acl_pdf": "https://aclanthology.org/2026.acl-long.226.pdf",
            "arxiv": "https://arxiv.org/abs/2601.07372",
            "code": "https://github.com/deepseek-ai/Engram",
        },
        retrieved_at="2026-07-29",
        version=(
            "ACL Anthology 2026.acl-long.226; compared with the Engram-named "
            f"repository update captured at commit {COMMIT}"
        ),
        status="active",
        tags=(
            "ngram",
            "conditional_memory",
            "tokenizer_compression",
            "canonicalization",
            "multihead_hashing",
            "context_gating",
            "acl_2026",
            "primary_source",
            "external_code_captured",
        ),
        notes=(
            "Organizations: Peking University; DeepSeek-AI; the ACL author "
            "affiliation also names the National Engineering Research Center "
            "of New Electronic Publishing Technologies. ACL calls the module "
            "Deep Sparse Embedding (DSE), while updated arXiv/repository "
            "materials call it Engram. The captured 33-page PDF is the latter "
            "version, not the 23-page ACL camera-ready PDF. Official code is a "
            "demo and explicitly mocks or omits production-scale components."
        ),
    )


def claims() -> tuple[ClaimRecord, ...]:
    return (
        ClaimRecord(
            claim_id=CLAIM_PROJECTION,
            paper_id=PAPER_ID,
            statement=(
                "Before suffix n-gram hashing, DSE/Engram precomputes a "
                "surjective vocabulary projection P: V -> V' that maps raw "
                "token IDs to canonical IDs using normalized textual "
                "equivalence, including Unicode NFKC normalization and "
                "lowercasing."
            ),
            claim_type="method_definition",
            scope={
                "applies_to": "canonicalized_ngram_memory_addressing",
                "domain": "transformer_language_model_pretraining",
                "direction": "mechanism_only",
                "representation_scope": "memory_lookup_keys",
                "source_versions": ["acl_dse", "updated_engram"],
            },
            limitations=(
                "The paper's phrase 'NFKC, lowercasing, etc.' does not fully "
                "specify every normalization rule.",
                "The exact sequence in the repository is demo code and is not "
                "asserted to be the undisclosed production implementation.",
                "The mechanism canonicalizes lookup keys; it does not justify "
                "changing model input IDs, targets, byte accounting, packing, "
                "or token order.",
                "A projection learned for a 128k DeepSeek tokenizer need not "
                "produce material collision reduction for OPHIS's 8192-token "
                "vocabulary.",
            ),
            locator=(
                "ACL 2026 Section 2.2 (pp. 4969-4970); updated Engram PDF "
                "Section 2.2; engram_demo_v1.py lines 60-103 at commit "
                f"{COMMIT}"
            ),
            extracted_at="2026-07-29",
            tags=(
                "ngram",
                "canonicalization",
                "tokenizer_compression",
                "method",
            ),
        ),
        ClaimRecord(
            claim_id=CLAIM_REDUCTION,
            paper_id=PAPER_ID,
            statement=(
                "For the paper's 128k tokenizer, the canonical projection "
                "reduced effective vocabulary size by 23% in the ACL text; "
                "the updated Engram Appendix C reports the unrounded ratio "
                "as 23.43%."
            ),
            claim_type="descriptive",
            scope={
                "applies_to": "canonicalized_ngram_memory_addressing",
                "domain": "deepseek_128k_tokenizer",
                "direction": "effective_vocabulary_reduction",
                "reported_percent_acl": 23,
                "reported_percent_updated_engram": 23.43,
            },
            limitations=(
                "This is one tokenizer-specific descriptive result, not a "
                "universal tokenizer-compression rate.",
                "The ACL body rounds the number; 23.43% comes from the later "
                "Engram-named captured PDF.",
                "No OPHIS 8192-token collision census or local BPB result is "
                "reported.",
            ),
            locator=(
                "ACL 2026 Section 2.2 (23%); updated Engram PDF Appendix C, "
                "Table 6 (23.43%)"
            ),
            extracted_at="2026-07-29",
            tags=(
                "ngram",
                "canonicalization",
                "tokenizer_compression",
                "reported_fact",
            ),
        ),
        ClaimRecord(
            claim_id=CLAIM_ABLATION,
            paper_id=PAPER_ID,
            statement=(
                "In the 12-layer 3B MoE plus 1.6B memory ablation trained for "
                "100B tokens, removing tokenizer compression was reported as "
                "one of the largest validation-loss regressions from the "
                "1.768 reference configuration."
            ),
            claim_type="causal",
            scope={
                "applies_to": "full_dse_engram_component_ablation",
                "domain": "3b_moe_plus_1p6b_memory_100b_tokens",
                "direction": "beneficial",
                "outcome": "validation_loss",
                "reference_validation_loss": 1.768,
            },
            limitations=(
                "Figure 5 and the prose do not report an exact numeric loss "
                "or uncertainty for the tokenizer-compression marker.",
                "The paper does not report a multi-seed replication for this "
                "component ablation.",
                "The reference configuration bundles 2/3-gram memory, layers "
                "2 and 6, branch-specific fusion, contextual gating, short "
                "convolution, and a 1.6B table.",
                "The scale, optimizer, token budget, tokenizer, and MoE/mHC "
                "backbone differ sharply from the live 300-second OPHIS frame.",
                "A local keys-only retrofit is not a faithful replication of "
                "the full module ablation.",
            ),
            locator=(
                "ACL 2026 Section 6.2 and Figure 5; updated Engram PDF "
                "Section 6.2 and Figure 5"
            ),
            extracted_at="2026-07-29",
            tags=(
                "ngram",
                "tokenizer_compression",
                "component_ablation",
                "scope_limited",
            ),
        ),
        ClaimRecord(
            claim_id=CLAIM_BRANCH_GATE,
            paper_id=PAPER_ID,
            statement=(
                "For an M-branch residual backbone, DSE/Engram shares one "
                "sparse embedding table and one value projection across "
                "branches, but uses M distinct key projections so each branch "
                "has an independent sigmoid gate."
            ),
            claim_type="method_definition",
            scope={
                "applies_to": "multi_branch_conditional_memory_fusion",
                "domain": "mhc_transformer_backbone",
                "direction": "mechanism_only",
                "default_branches": 4,
            },
            limitations=(
                "The branch-specific design presupposes a multi-branch mHC/HC "
                "backbone and does not directly transfer to a single-stream "
                "OPHIS retrofit.",
                "The full component ablation does not isolate every algebraic "
                "choice within the branch gate.",
            ),
            locator=(
                "ACL 2026 Section 2.4, Equation 6; updated Engram PDF "
                "Section 2.4; engram_demo_v1.py lines 326-377 at commit "
                f"{COMMIT}"
            ),
            extracted_at="2026-07-29",
            tags=(
                "conditional_memory",
                "context_gating",
                "multi_branch",
                "method",
            ),
        ),
    )


def _common_facts(subject: str) -> dict[str, Any]:
    return {
        "reported": {"subject": subject, "local_sota_adoption_claimed": False},
        "source_versions": VERSION_FACTS,
        "organizations": [
            "Peking University",
            "DeepSeek-AI",
            (
                "National Engineering Research Center of New Electronic "
                "Publishing Technologies"
            ),
        ],
        "code_snapshot": {
            "repository": "https://github.com/deepseek-ai/Engram",
            "commit": COMMIT,
            "tree": TREE,
            "manifest_sha256": ARTIFACTS["manifest.json"]["sha256"],
            "archive_sha256": ARTIFACTS[
                "engram-fb7f84a2.tar.zst"
            ]["sha256"],
            "bundle_sha256": ARTIFACTS[
                "engram-fb7f84a2.bundle"
            ]["sha256"],
            "paper_sha256": ARTIFACTS["Engram_paper.pdf"]["sha256"],
            "demo_sha256": (
                "9d082070654df217e21bbca9926a4267bdf2cce7777aa6739747c24de30d2044"
            ),
            "license": "Apache-2.0",
            "license_sha256": (
                "9debfe045c2a2755a52d0ae1087605d15770fe6636798c09efd93c1dc732ef28"
            ),
        },
        "critic_rating": LOCAL_IDEA_RATING,
        "rating_caveat": RATING_CAVEAT,
    }


def evidence() -> tuple[EvidenceRecord, ...]:
    method_facts = _common_facts(
        "canonical vocabulary projection before n-gram hashing"
    )
    method_facts["reported"]["paper_definition"] = {
        "mapping": "surjective P: V -> V'",
        "paper_examples": ["NFKC", "lowercasing"],
        "demo_pipeline": [
            "NFKC",
            "NFD",
            "StripAccents",
            "Lowercase",
            "collapse whitespace",
            "strip with single-space sentinel handling",
        ],
        "demo_invalid_decode_fallback": (
            "use the tokenizer's token string when single-token decoding "
            "contains the Unicode replacement character"
        ),
    }

    reduction_facts = _common_facts(
        "effective-vocabulary reduction for the paper's 128k tokenizer"
    )
    reduction_facts["reported"].update(
        {
            "acl_body_percent": 23,
            "updated_engram_appendix_percent": 23.43,
            "updated_engram_locator": "Appendix C, Table 6",
            "version_precision_note": (
                "The 23.43% precision belongs to the captured Engram update; "
                "the ACL body reports 23%."
            ),
        }
    )

    ablation_facts = _common_facts(
        "tokenizer-compression component ablation"
    )
    ablation_facts["reported"].update(
        {
            "backbone": "12-layer 3B MoE",
            "activated_parameters_b": 0.56,
            "memory_parameters_b": 1.6,
            "training_tokens_b": 100,
            "reference_ngrams": [2, 3],
            "reference_layers": [2, 6],
            "reference_validation_loss": 1.768,
            "ablation_result": (
                "Removing tokenizer compression was described as one of the "
                "largest validation-loss regressions."
            ),
            "exact_ablation_loss_reported": False,
            "uncertainty_reported": False,
        }
    )

    gate_facts = _common_facts(
        "branch-specific context-aware gating"
    )
    gate_facts["reported"]["paper_definition"] = {
        "shared_across_branches": [
            "sparse embedding table",
            "value projection W_V",
        ],
        "branch_specific": ["key projections W_K^(m)", "sigmoid gates alpha_t^(m)"],
        "default_branch_count": 4,
    }

    return (
        EvidenceRecord(
            evidence_id="evd_lit_deepseek_engram_canonical_token_projection",
            source_type="literature",
            paper_ids=(PAPER_ID,),
            claim_ids=(CLAIM_PROJECTION,),
            run_ids=(),
            experiment_id="",
            hypothesis_ids=(),
            facts=method_facts,
            analysis={
                "method": (
                    "Read ACL Sections 2.2/2.4, the captured updated paper, "
                    "and the exact-commit demo; separated the published method "
                    "definition from any claim of local effect."
                ),
                "code_ref": (
                    "research/experiments/artifacts/external_code/"
                    "pap_deepseek_engram_2026/manifest.json; "
                    f"engram_demo_v1.py@{COMMIT}:60-103"
                ),
            },
            trust={
                "design": "not_applicable",
                "replication": "adequate",
                "scope_match": "adequate",
                "directness": "strong",
                "limitations": [
                    "Strong evidence for the published method definition only.",
                    "The public code is a demo, not the production training stack.",
                    "No local OPHIS performance effect is tested.",
                ],
            },
            assessment={
                "relation": "supports",
                "strength": "strong",
                "limitations": [
                    "Supports the existence and implementation shape of "
                    "canonicalized lookup keys.",
                    "Does not support changing model tokens or claim that the "
                    "keys improve local BPB.",
                ],
            },
            artifact_paths=COMMON_ARTIFACT_PATHS,
            created_at=CREATED_AT,
            created_by=CREATED_BY,
            tags=(
                "ngram",
                "canonicalization",
                "method_definition",
                "exact_commit",
            ),
        ),
        EvidenceRecord(
            evidence_id="evd_lit_deepseek_engram_128k_projection_reduction",
            source_type="literature",
            paper_ids=(PAPER_ID,),
            claim_ids=(CLAIM_REDUCTION,),
            run_ids=(),
            experiment_id="",
            hypothesis_ids=(),
            facts=reduction_facts,
            analysis={
                "method": (
                    "Cross-version extraction: retained the ACL body's rounded "
                    "23% separately from the updated Engram Appendix C value "
                    "of 23.43%."
                ),
                "code_ref": (
                    "research/experiments/artifacts/external_code/"
                    "pap_deepseek_engram_2026/Engram_paper.pdf Appendix C "
                    "Table 6"
                ),
            },
            trust={
                "design": "adequate",
                "replication": "weak",
                "scope_match": "weak",
                "directness": "strong",
                "limitations": [
                    "Single 128k tokenizer and one reported compression census.",
                    "No independent reproduction or statistical uncertainty.",
                    "OPHIS uses a much smaller vocabulary.",
                ],
            },
            assessment={
                "relation": "supports",
                "strength": "moderate",
                "limitations": [
                    "Supports the reported tokenizer-specific reduction.",
                    "Does not establish a comparable reduction or a BPB benefit "
                    "in OPHIS.",
                ],
            },
            artifact_paths=COMMON_ARTIFACT_PATHS,
            created_at=CREATED_AT,
            created_by=CREATED_BY,
            tags=(
                "ngram",
                "canonicalization",
                "reported_fact",
                "cross_version",
            ),
        ),
        EvidenceRecord(
            evidence_id=(
                "evd_lit_deepseek_engram_tokenizer_compression_ablation"
            ),
            source_type="literature",
            paper_ids=(PAPER_ID,),
            claim_ids=(CLAIM_ABLATION,),
            run_ids=(),
            experiment_id="",
            hypothesis_ids=(),
            facts=ablation_facts,
            analysis={
                "method": (
                    "Claim-specific reading of the fixed-memory component "
                    "ablation, preserving the qualitative ranking while "
                    "refusing to infer an unreported numeric marker or local "
                    "transfer effect."
                ),
                "code_ref": (
                    "research/experiments/artifacts/external_code/"
                    "pap_deepseek_engram_2026/Engram_paper.pdf Section 6.2, "
                    "Figure 5"
                ),
            },
            trust={
                "design": "adequate",
                "replication": "weak",
                "scope_match": "weak",
                "directness": "adequate",
                "limitations": [
                    "The component ablation is controlled at fixed memory "
                    "budget but remains embedded in the full DSE/Engram stack.",
                    "No exact tokenizer-compression marker, error bar, or "
                    "multi-seed result is reported.",
                    "The 3B/100B-token MoE+mHC frame is far from OPHIS.",
                ],
            },
            assessment={
                "relation": "supports",
                "strength": "moderate",
                "limitations": [
                    "Moderately supports the paper-scoped qualitative ablation "
                    "claim.",
                    "Provides only weak prior support for local causal transfer.",
                    "A local keys-only retrofit is not a full replication and "
                    "requires its own governed collision census and paired run.",
                ],
            },
            artifact_paths=COMMON_ARTIFACT_PATHS,
            created_at=CREATED_AT,
            created_by=CREATED_BY,
            tags=(
                "ngram",
                "tokenizer_compression",
                "component_ablation",
                "scope_limited",
            ),
        ),
        EvidenceRecord(
            evidence_id="evd_lit_deepseek_engram_branch_specific_gating",
            source_type="literature",
            paper_ids=(PAPER_ID,),
            claim_ids=(CLAIM_BRANCH_GATE,),
            run_ids=(),
            experiment_id="",
            hypothesis_ids=(),
            facts=gate_facts,
            analysis={
                "method": (
                    "Compared the ACL Equation 6 definition with the "
                    "exact-commit demo implementation; assessed mechanism "
                    "identity separately from ablation value."
                ),
                "code_ref": (
                    "research/experiments/artifacts/external_code/"
                    "pap_deepseek_engram_2026/manifest.json; "
                    f"engram_demo_v1.py@{COMMIT}:326-377"
                ),
            },
            trust={
                "design": "not_applicable",
                "replication": "adequate",
                "scope_match": "weak",
                "directness": "strong",
                "limitations": [
                    "Strong evidence for the method definition only.",
                    "The live OPHIS model is not an M-branch mHC backbone.",
                    "No local effect is tested.",
                ],
            },
            assessment={
                "relation": "supports",
                "strength": "strong",
                "limitations": [
                    "Supports the exact shared-value/distinct-key gate design.",
                    "Does not make branch gating an authorized local addition.",
                ],
            },
            artifact_paths=COMMON_ARTIFACT_PATHS,
            created_at=CREATED_AT,
            created_by=CREATED_BY,
            tags=(
                "conditional_memory",
                "context_gating",
                "multi_branch",
                "method_definition",
            ),
        ),
    )


def records() -> tuple[
    PaperRecord, tuple[ClaimRecord, ...], tuple[EvidenceRecord, ...]
]:
    return paper(), claims(), evidence()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts() -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, expected in ARTIFACTS.items():
        path = ARTIFACT_ROOT / name
        if not path.is_file():
            raise SchemaError(f"missing Engram artifact: {path}")
        size = path.stat().st_size
        digest = _sha256(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise SchemaError(
                f"artifact mismatch for {name}: expected={expected}, "
                f"observed={{'bytes': {size}, 'sha256': {digest!r}}}"
            )
        observed[name] = {"bytes": size, "sha256": digest}

    manifest = json.loads(
        (ARTIFACT_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    repository = dict(manifest.get("repository", {}))
    if (
        manifest.get("paper_id") != PAPER_ID
        or repository.get("commit") != COMMIT
        or repository.get("tree") != TREE
    ):
        raise SchemaError("Engram manifest paper/commit/tree identity mismatch")

    subprocess.run(
        ["zstd", "-q", "-t", str(ARTIFACT_ROOT / "engram-fb7f84a2.tar.zst")],
        check=True,
    )
    return observed


def _action(store: Any, record: Any, kind: str, *, apply: bool) -> str:
    existing = store.by_id().get(str(record.registry_id))
    if existing is not None:
        if canonical_json(existing.to_dict()) != canonical_json(record.to_dict()):
            raise SchemaError(
                f"{kind} ID {record.registry_id!r} exists with different content"
            )
        return "skip_exact"
    if apply:
        store.add(record)
        return "appended"
    return "would_append"


def preflight_or_apply(*, apply: bool) -> dict[str, Any]:
    verified = verify_artifacts()
    registry = ResearchRegistry(RESEARCH_ROOT)
    validation = registry.validate(check_generated_state=False).to_dict()
    if validation.get("warnings"):
        raise SchemaError(
            f"pre-existing authoritative validation warnings: "
            f"{validation['warnings']}"
        )

    paper_record, claim_records, evidence_records = records()
    actions: dict[str, list[dict[str, str]]] = {
        "papers": [],
        "claims": [],
        "literature_evidence": [],
    }
    actions["papers"].append(
        {
            "id": paper_record.paper_id,
            "action": _action(
                registry.papers, paper_record, "paper", apply=apply
            ),
        }
    )
    for record in claim_records:
        actions["claims"].append(
            {
                "id": record.claim_id,
                "action": _action(
                    registry.claims, record, "claim", apply=apply
                ),
            }
        )
    for record in evidence_records:
        actions["literature_evidence"].append(
            {
                "id": record.evidence_id,
                "action": _action(
                    registry.literature_evidence,
                    record,
                    "literature evidence",
                    apply=apply,
                ),
            }
        )

    if apply:
        post = registry.validate(check_generated_state=False).to_dict()
        if post.get("warnings"):
            raise SchemaError(
                f"post-append authoritative validation warnings: "
                f"{post['warnings']}"
            )
    else:
        post = validation

    return {
        "mode": "apply" if apply else "preflight",
        "artifact_verification": verified,
        "actions": actions,
        "authoritative_validation": post,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="append exact missing records; default is read-only preflight",
    )
    args = parser.parse_args()

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SchemaError(
                f"literature-ingest lock is already held: {LOCK_PATH}"
            ) from exc
        try:
            result = preflight_or_apply(apply=bool(args.apply))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
