#!/usr/bin/env python3
"""Verify and register the GPAS NeurIPS 2025 provenance capsule.

This helper binds the official proceedings paper and exact official-code
snapshot to one PaperRecord, four atomic ClaimRecords, and four corresponding
literature EvidenceRecords.  Published/source facts are kept separate from
the much weaker question of transfer to the live OPHIS model.

The default mode is a read-only preflight. ``--apply`` appends only missing
typed records under the repository-wide literature-ingest lock.  Replays are
idempotent only when an existing ID has exactly the same canonical typed
serialization; any divergent ID reuse fails closed before an append begins.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
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
    / "pap_gpas_neurips2025"
)
INVENTORY_PATH = (
    RESEARCH_ROOT / "experiments" / "artifacts" / "external_code" / "INVENTORY.md"
)
LOCK_PATH = REPO_ROOT / "tmp" / "literature_ingest_transactions" / "global.lock"

PAPER_ID = "pap_gpas_neurips2025"
CLAIM_CONTRACT = "clm_gpas_gradient_preserving_scaling_contract"
CLAIM_MULTISCALE = "clm_gpas_preln_multiscale_perplexity"
CLAIM_ABLATION = "clm_gpas_stop_gradient_and_placement_ablation"
CLAIM_DYNAMICS = "clm_gpas_activation_gradient_layer_importance"

EVIDENCE_CONTRACT = "evd_lit_gpas_method_contract"
EVIDENCE_MULTISCALE = "evd_lit_gpas_multiscale_pretraining"
EVIDENCE_ABLATION = "evd_lit_gpas_mechanism_ablations"
EVIDENCE_DYNAMICS = "evd_lit_gpas_dynamics"

CREATED_AT = "2026-07-29T17:14:25Z"
CREATED_BY = "codex_gpas_literature_registration"

COMMIT = "31980688f4cbb1b0cff59bca9077e6fc52dab3f0"
TREE = "076a1d81a48f4d03dbd455d4197345480bd818bb"
BUNDLE_REF = "refs/heads/archive-31980688"

ARTIFACTS: dict[str, dict[str, Any]] = {
    "manifest.json": {
        "bytes": 3724,
        "sha256": "871e6689facf3453a19c2d8cf565103d95b29f0ca3a16c0a493c2c0e0764093e",
    },
    "GPAS_NeurIPS_2025.pdf": {
        "bytes": 1407649,
        "sha256": "d644ed31e31a693fa889e01655e94cfea1cdb0c312901ee11b001af68b14a35e",
    },
    "gpas-31980688.tar.zst": {
        "bytes": 92641,
        "sha256": "1177e6766f53cf0bc59b880a96ca09d7732984db62f8d54b9bf5083b1b45acf0",
    },
    "gpas-31980688.bundle": {
        "bytes": 98272,
        "sha256": "8e2abb5ba02b16639d92cb532b3134c5fa4bcce0f43156585fa2e591294edaf1",
    },
}

VERIFIED_ARCHIVE_FILES = {
    "peft_pretraining/modeling_llama.py": (
        "5d108c750a8753c365845ee4b1807dda2517a1f0363d7f159ee9a433b5c4b702"
    ),
    "peft_pretraining/custom_activations.py": (
        "15a5c6bec1d92761686672f35b57821ad20b115f7eec3a9112b5a84091d46b24"
    ),
    "torchrun_main.py": (
        "0c439151847fc0c2ad2ed91f52bfbcb7a164eb78e06dd033b31b894b5d593962"
    ),
    "LICENSE": (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    ),
}

COMMON_ARTIFACT_PATHS = tuple(
    (
        "research/experiments/artifacts/external_code/"
        f"pap_gpas_neurips2025/{name}"
    )
    for name in (
        "manifest.json",
        "GPAS_NeurIPS_2025.pdf",
        "gpas-31980688.tar.zst",
        "gpas-31980688.bundle",
    )
)

INVENTORY_REQUIRED_FRAGMENTS = (
    "`pap_gpas_neurips2025/gpas-31980688.tar.zst`",
    "`pap_gpas_neurips2025/gpas-31980688.bundle`",
    "`pap_gpas_neurips2025/GPAS_NeurIPS_2025.pdf`",
)

ORGANIZATIONS = (
    "The Hong Kong University of Science and Technology",
    "International Digital Economy Academy",
    "Dalian University of Technology",
    "Emory University",
    "University of Texas at Austin",
    "NVIDIA",
    "University of Oxford",
    "University of Surrey",
)

LOCAL_IDEA_RATING = {
    "scale": "1_to_5_each",
    "maximum": 40,
    "axis_order": [
        "novelty",
        "primary_source_provenance",
        "causal_validity",
        "expected_local_impact",
        "transfer_reliability",
        "feasibility_cost",
        "numerical_falsifiability",
        "connected_program_coherence",
    ],
    "dimensions": {
        "novelty": 4,
        "primary_source_provenance": 5,
        "causal_validity": 4,
        "expected_local_impact": 3,
        "transfer_reliability": 2,
        "feasibility_cost": 5,
        "numerical_falsifiability": 5,
        "connected_program_coherence": 5,
    },
    "vector": "4/5/4/3/2/5/5/5",
    "total": 33,
    "status": "independent_multiagent_critic_reconciled",
    "subject": "source-faithful local GPAS attribution candidate",
}

RATING_CAVEAT = (
    "The 33/40 score is an independent critic's prospective rating of a "
    "source-faithful local attribution candidate, not a paper result. "
    "Institution and venue affect provenance only. The score does not "
    "authorize implementation, launch, adoption, or a SOTA update."
)

SOURCE_LIMITS = {
    "error_bars_reported_for_cited_results": False,
    "multi_seed_aggregation_reported_for_cited_results": False,
    "largest_main_pretraining_model_parameters": "1B",
    "paper_limitations": [
        "Main experiments are limited to 1B parameters.",
        "SiLU gates can undergo unstable updates that disrupt training.",
        "Learnable gates introduce pretraining uncertainty.",
        "The method is intended for pretraining from scratch.",
    ],
}

LOCAL_SCOPE = {
    "match": "weak",
    "paper_frame": (
        "12-layer-and-deeper Pre-LN LLaMA-family models trained from scratch"
    ),
    "ophis_frame": (
        "eight layers with learned residual/x0 mixing, hashed memory, "
        "zero-initialized output projections, and Peri-style normalization"
    ),
    "untested_local_endpoints": [
        "activation-variance mediation",
        "gate-gradient safety",
        "compiled/eager parity",
        "charged-clock throughput",
        "validation BPB",
    ],
}

NO_AUTHORITY = {
    "implementation_authorized": False,
    "launch_authorized": False,
    "adoption_authorized": False,
    "sota_update_authorized": False,
    "local_sota_adoption_claimed": False,
}


def paper() -> PaperRecord:
    return PaperRecord(
        paper_id=PAPER_ID,
        title=(
            "GPAS: Accelerating Convergence of LLM Pretraining via "
            "Gradient-Preserving Activation Scaling"
        ),
        authors=(
            "Tianhao Chen",
            "Xin Xu",
            "Zijing Liu",
            "Pengxiang Li",
            "Xinyuan Song",
            "Ajay Kumar Jaiswal",
            "Fan Zhang",
            "Jishan Hu",
            "Yang Wang",
            "Hao Chen",
            "Shizhe Diao",
            "Shiwei Liu",
            "Yu Li",
            "Lu Yin",
            "Can Yang",
        ),
        year=2025,
        venue={
            "name": (
                "Advances in Neural Information Processing Systems 38 "
                "(NeurIPS 2025), Main Conference Track"
            ),
            "peer_reviewed": True,
        },
        urls={
            "primary": (
                "https://papers.neurips.cc/paper_files/paper/2025/hash/"
                "140258c3c68409555e0fad247875af97-Abstract-Conference.html"
            ),
            "pdf": (
                "https://papers.nips.cc/paper_files/paper/2025/file/"
                "140258c3c68409555e0fad247875af97-Paper-Conference.pdf"
            ),
            "arxiv": "https://arxiv.org/abs/2506.22049",
            "code": "https://github.com/dandingsky/GPAS",
            "code_commit": "https://github.com/dandingsky/GPAS/tree/" + COMMIT,
        },
        retrieved_at="2026-07-29",
        version=(
            "NeurIPS 2025 proceedings paper hash "
            "140258c3c68409555e0fad247875af97; official repository captured "
            f"separately at commit {COMMIT}"
        ),
        status="active",
        tags=(
            "gpas",
            "pre_layernorm",
            "activation_scaling",
            "stop_gradient",
            "residual_stream",
            "training_dynamics",
            "neurips_2025",
            "primary_source",
            "external_code_captured",
        ),
        notes=(
            "Organizations: " + "; ".join(ORGANIZATIONS) + ". The paper and "
            "official exact-commit implementation were read for method, "
            "multiscale endpoint, ablation, and mechanism evidence. The cited "
            "tables and figures do not report error bars or multi-seed "
            "aggregation. Local OPHIS transfer remains untested and weakly "
            "scope-matched."
        ),
    )


def claims() -> tuple[ClaimRecord, ...]:
    return (
        ClaimRecord(
            claim_id=CLAIM_CONTRACT,
            paper_id=PAPER_ID,
            statement=(
                "GPAS transforms a post-sublayer hidden state h as "
                "h - SiLU(alpha_l) * sg(h), where sg is identity in the "
                "forward pass and stops gradients in backward, so the "
                "hidden-state Jacobian of the scaling operation is identity. "
                "The exact official implementation initializes one scalar "
                "gate to zero per decoder layer and reuses that same gate "
                "after both the attention and feed-forward residual sums."
            ),
            claim_type="method_definition",
            scope={
                "applies_to": "gpas_preln_residual_activation_scaling",
                "domain": "llm_pretraining_from_scratch",
                "direction": "mechanism_only",
                "forward_equation": "h_out = h - SiLU(alpha_l) * stopgrad(h)",
                "hidden_state_jacobian": "identity",
                "gate_initialization": 0.0,
                "gate_granularity": "one_scalar_per_decoder_layer",
                "applications_per_preln_layer": 2,
                "code_commit": COMMIT,
            },
            limitations=(
                "The paper/source contract does not establish a local OPHIS "
                "performance effect.",
                "A conventional multiplicative residual scale without the "
                "stop-gradient branch is not GPAS.",
                "The source shares one layer scalar across attention and FFN; "
                "a separate gate per sublayer would be a local variant.",
                "The paper targets pretraining from scratch and warns that "
                "retrofitting a model pretrained without GPAS is likely "
                "suboptimal.",
            ),
            locator=(
                "NeurIPS 2025 Section 3.1, Equations 1-3; official code "
                f"peft_pretraining/modeling_llama.py@{COMMIT}:261-268,"
                "336-340,491-510"
            ),
            extracted_at="2026-07-29",
            tags=(
                "gpas",
                "stop_gradient",
                "activation_scaling",
                "method_definition",
                "exact_commit",
            ),
        ),
        ClaimRecord(
            claim_id=CLAIM_MULTISCALE,
            paper_id=PAPER_ID,
            statement=(
                "In the paper's Pre-LN pretraining comparisons, adding GPAS "
                "reduced reported evaluation perplexity at every listed scale: "
                "33.98 to 33.38 at 71M, 26.61 to 26.25 at 130M, 21.54 to "
                "21.34 at 250M, 20.71 to 19.77 at 350M, and 16.53 to 16.11 "
                "at 1B parameters."
            ),
            claim_type="causal",
            scope={
                "applies_to": "paper_preln_plus_gpas_comparisons",
                "domain": "c4_llama_family_pretraining",
                "direction": "beneficial",
                "outcome": "evaluation_perplexity",
                "parameter_scales": ["71M", "130M", "250M", "350M", "1B"],
                "training_tokens_billions": [1, 2, 4, 6, 10],
            },
            limitations=(
                "Table 2 reports point estimates without error bars, "
                "confidence intervals, or statistical significance.",
                "The cited results are not identified as multi-seed "
                "aggregates and no seed-level variance is reported.",
                "The largest main experiment is 1B parameters.",
                "Perplexity on the paper's evaluation set is not OPHIS "
                "validation BPB under a 300-second charged clock.",
                "The live OPHIS architecture and optimization frame are only "
                "weakly scope-matched.",
            ),
            locator=(
                "NeurIPS 2025 Section 4.1 Table 1 and Section 4.2 Table 2"
            ),
            extracted_at="2026-07-29",
            tags=(
                "gpas",
                "pre_layernorm",
                "perplexity",
                "multiscale",
                "scope_limited",
            ),
        ),
        ClaimRecord(
            claim_id=CLAIM_ABLATION,
            paper_id=PAPER_ID,
            statement=(
                "In the paper's 350M Pre-LN ablations, default post-sublayer "
                "GPAS with stop-gradient reported perplexity 20.35 versus "
                "21.35 without GPAS and 21.34 when stop-gradient was removed; "
                "alternative insertion points reported 20.73 before the "
                "sublayer, 21.33 after LayerNorm, and 21.23 after only the "
                "attention/FFN module."
            ),
            claim_type="causal",
            scope={
                "applies_to": "paper_350m_preln_gpas_design_ablations",
                "domain": "c4_llama_family_pretraining",
                "direction": "default_stopgrad_post_sublayer_beneficial",
                "outcome": "evaluation_perplexity",
                "parameter_scale": "350M",
            },
            limitations=(
                "Tables 4 and 5 report point estimates without error bars or "
                "multi-seed aggregation.",
                "The ablation frame's no-GPAS and default values differ from "
                "the 350M entries in main Table 2 and must not be merged as "
                "one numerical experiment.",
                "Removing stop-gradient preserves a similar forward scaling "
                "shape but changes the backward mechanism.",
                "The source ablation does not test OPHIS's eight-layer, "
                "mixed-residual architecture or charged-clock throughput.",
            ),
            locator=(
                "NeurIPS 2025 Appendix A, Equation 12, Tables 4-5, "
                "and 'Where to apply GPAS'/'Necessity of stop gradient operator'"
            ),
            extracted_at="2026-07-29",
            tags=(
                "gpas",
                "stop_gradient",
                "placement_ablation",
                "perplexity",
                "scope_limited",
            ),
        ),
        ClaimRecord(
            claim_id=CLAIM_DYNAMICS,
            paper_id=PAPER_ID,
            statement=(
                "For the paper's 1B models, the reported diagnostics show "
                "near-50% lower highest activation variance and a more compact "
                "depth profile with GPAS; most layerwise gradient norms are "
                "described as about 0.05-0.5 versus 0.05-0.1 for Pre-LN; and "
                "single-layer removal after finetuning indicates increased "
                "importance for most layers, especially deeper layers."
            ),
            claim_type="descriptive",
            scope={
                "applies_to": "paper_1b_training_dynamics_and_layer_removal",
                "domain": "preln_llama_family_pretraining_and_sft",
                "direction": "mechanism_evidence",
                "parameter_scale": "1B",
                "diagnostics": [
                    "activation_variance_by_layer_and_step",
                    "gradient_norm_by_layer_and_step",
                    "post_sft_single_layer_removal",
                ],
            },
            limitations=(
                "The activation and gradient figures do not provide exact "
                "underlying values, uncertainty bands, or multi-seed results.",
                "The layer-importance diagnostic is measured after supervised "
                "finetuning as average benchmark-score drop on removal, not "
                "as pretraining loss mediation.",
                "The paper reports a gate/gradient spike around step 10K that "
                "temporarily disturbed pretraining.",
                "These diagnostics are compatible with the proposed mechanism "
                "but do not isolate a complete causal mediation path from "
                "variance control to perplexity.",
                "No comparable local OPHIS diagnostics have yet been run.",
            ),
            locator=(
                "NeurIPS 2025 Sections 5.2-5.5, Figures 4-7; "
                "Section 6 Limitations"
            ),
            extracted_at="2026-07-29",
            tags=(
                "gpas",
                "activation_variance",
                "gradient_norm",
                "layer_importance",
                "mechanism_evidence",
            ),
        ),
    )


def _common_facts(subject: str) -> dict[str, Any]:
    return {
        "reported": {
            "subject": subject,
            "source_is_local_result": False,
            **NO_AUTHORITY,
        },
        "source_version": {
            "venue": "NeurIPS 2025, proceedings volume 38, main track",
            "paper_hash": "140258c3c68409555e0fad247875af97",
            "repository": "https://github.com/dandingsky/GPAS",
            "commit": COMMIT,
            "tree": TREE,
        },
        "source_limitations": SOURCE_LIMITS,
        "organizations": list(ORGANIZATIONS),
        "code_snapshot": {
            "repository": "https://github.com/dandingsky/GPAS",
            "commit": COMMIT,
            "tree": TREE,
            "manifest_sha256": ARTIFACTS["manifest.json"]["sha256"],
            "archive_sha256": ARTIFACTS["gpas-31980688.tar.zst"]["sha256"],
            "bundle_sha256": ARTIFACTS["gpas-31980688.bundle"]["sha256"],
            "paper_sha256": ARTIFACTS["GPAS_NeurIPS_2025.pdf"]["sha256"],
            "modeling_llama_sha256": VERIFIED_ARCHIVE_FILES[
                "peft_pretraining/modeling_llama.py"
            ],
            "custom_activations_sha256": VERIFIED_ARCHIVE_FILES[
                "peft_pretraining/custom_activations.py"
            ],
            "training_entrypoint_sha256": VERIFIED_ARCHIVE_FILES[
                "torchrun_main.py"
            ],
            "license": "Apache-2.0",
            "license_sha256": VERIFIED_ARCHIVE_FILES["LICENSE"],
        },
        "local_scope_assessment": LOCAL_SCOPE,
        "critic_rating": LOCAL_IDEA_RATING,
        "rating_caveat": RATING_CAVEAT,
    }


def _record(
    *,
    evidence_id: str,
    claim_id: str,
    facts: dict[str, Any],
    method: str,
    code_ref: str,
    trust: dict[str, Any],
    assessment: dict[str, Any],
    tags: tuple[str, ...],
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_type="literature",
        paper_ids=(PAPER_ID,),
        claim_ids=(claim_id,),
        run_ids=(),
        experiment_id="",
        hypothesis_ids=(),
        facts=facts,
        analysis={"method": method, "code_ref": code_ref},
        trust=trust,
        assessment=assessment,
        artifact_paths=COMMON_ARTIFACT_PATHS,
        created_at=CREATED_AT,
        created_by=CREATED_BY,
        tags=tags,
    )


def evidence() -> tuple[EvidenceRecord, ...]:
    contract_facts = _common_facts(
        "gradient-preserving activation-scaling method contract"
    )
    contract_facts["reported"]["method_definition"] = {
        "equation": "h_out = h - SiLU(alpha_l) * stopgrad(h)",
        "forward_scale": "1 - SiLU(alpha_l)",
        "hidden_state_jacobian": "identity",
        "zero_initialized_scalar_per_layer": True,
        "same_scalar_after_attention_and_ffn": True,
        "official_code_matches_equation": True,
    }

    multiscale_facts = _common_facts(
        "Pre-LN plus GPAS multiscale pretraining perplexity"
    )
    multiscale_facts["reported"]["table_2_preln_perplexity"] = [
        {
            "parameters": "71M",
            "training_steps": 10000,
            "training_tokens_b": 1,
            "preln": 33.98,
            "preln_plus_gpas": 33.38,
            "reported_delta": -0.60,
        },
        {
            "parameters": "130M",
            "training_steps": 20000,
            "training_tokens_b": 2,
            "preln": 26.61,
            "preln_plus_gpas": 26.25,
            "reported_delta": -0.36,
        },
        {
            "parameters": "250M",
            "training_steps": 40000,
            "training_tokens_b": 4,
            "preln": 21.54,
            "preln_plus_gpas": 21.34,
            "reported_delta": -0.20,
        },
        {
            "parameters": "350M",
            "training_steps": 60000,
            "training_tokens_b": 6,
            "preln": 20.71,
            "preln_plus_gpas": 19.77,
            "reported_delta": -0.94,
        },
        {
            "parameters": "1B",
            "training_steps": 100000,
            "training_tokens_b": 10,
            "preln": 16.53,
            "preln_plus_gpas": 16.11,
            "reported_delta": -0.42,
        },
    ]

    ablation_facts = _common_facts(
        "stop-gradient necessity and insertion-position ablations"
    )
    ablation_facts["reported"].update(
        {
            "table_5_stop_gradient_perplexity": {
                "with_stop_gradient": 20.35,
                "without_stop_gradient": 21.34,
                "no_gpas": 21.35,
            },
            "table_4_position_perplexity": {
                "after_sublayer_default": 20.35,
                "before_sublayer": 20.73,
                "after_layernorm": 21.33,
                "after_attention_or_ffn_module": 21.23,
                "no_gpas": 21.35,
            },
            "parameter_scale": "350M",
            "main_table_numerical_frame_is_separate": True,
        }
    )

    dynamics_facts = _common_facts(
        "activation-variance, gradient, and layer-importance evidence"
    )
    dynamics_facts["reported"]["section_5_diagnostics"] = {
        "parameter_scale": "1B",
        "activation_variance": {
            "highest_variance_reduction": "near 50%",
            "distribution": "more uniform and compact across layers",
            "numeric_series_reported": False,
        },
        "gradient_norm": {
            "preln_most_layers": "approximately 0.05-0.1",
            "preln_plus_gpas_most_layers": "approximately 0.05-0.5",
            "gate_gradient_spike_around_step": 10000,
            "spike_disturbed_pretraining": True,
        },
        "layer_importance": {
            "definition": (
                "average downstream benchmark-score drop after removing one "
                "layer from the finetuned model"
            ),
            "reported_pattern": (
                "higher for most layers and especially deeper layers with GPAS"
            ),
        },
        "complete_causal_mediation_established": False,
    }

    return (
        _record(
            evidence_id=EVIDENCE_CONTRACT,
            claim_id=CLAIM_CONTRACT,
            facts=contract_facts,
            method=(
                "Read the NeurIPS method equations and inspected the exact "
                "official implementation's gate initialization and both "
                "per-layer call sites. Kept method identity separate from "
                "unperformed local transfer."
            ),
            code_ref=(
                "research/experiments/artifacts/external_code/"
                "pap_gpas_neurips2025/manifest.json; "
                f"peft_pretraining/modeling_llama.py@{COMMIT}:261-268,"
                "336-340,491-510"
            ),
            trust={
                "design": "not_applicable",
                "replication": "adequate",
                "scope_match": "weak",
                "directness": "strong",
                "limitations": [
                    "Strong direct evidence for the external method contract.",
                    "The captured implementation was inspected, not executed "
                    "inside the live OPHIS model by this registration task.",
                    "No local performance or safety effect is tested.",
                ],
            },
            assessment={
                "relation": "supports",
                "strength": "strong",
                "limitations": [
                    "Supports the exact stop-gradient algebra, initialization, "
                    "and shared per-layer gate shape.",
                    "Does not authorize a local implementation or launch.",
                    "Does not support a local SOTA or adoption claim.",
                ],
            },
            tags=(
                "gpas",
                "method_definition",
                "stop_gradient",
                "exact_commit",
                "local_transfer_not_tested",
            ),
        ),
        _record(
            evidence_id=EVIDENCE_MULTISCALE,
            claim_id=CLAIM_MULTISCALE,
            facts=multiscale_facts,
            method=(
                "Extracted each Pre-LN/Pre+GPAS point estimate from Table 2 "
                "and the corresponding scale, steps, and token budget from "
                "Table 1; recorded absent uncertainty and local scope mismatch."
            ),
            code_ref=(
                "research/experiments/artifacts/external_code/"
                "pap_gpas_neurips2025/GPAS_NeurIPS_2025.pdf, "
                "Sections 4.1-4.2, Tables 1-2"
            ),
            trust={
                "design": "adequate",
                "replication": "weak",
                "scope_match": "weak",
                "directness": "strong",
                "limitations": [
                    "Direct primary-table extraction across five scales.",
                    "No error bars, seed-level values, or multi-seed "
                    "aggregation are reported for the cited points.",
                    "The paper endpoint and local model/clock differ.",
                ],
            },
            assessment={
                "relation": "supports",
                "strength": "moderate",
                "limitations": [
                    "Supports the paper-scoped multiscale perplexity claim.",
                    "Provides only a prior, not local causal validation.",
                    "Does not authorize implementation, launch, adoption, or SOTA.",
                ],
            },
            tags=(
                "gpas",
                "pre_layernorm",
                "multiscale",
                "perplexity",
                "scope_limited",
            ),
        ),
        _record(
            evidence_id=EVIDENCE_ABLATION,
            claim_id=CLAIM_ABLATION,
            facts=ablation_facts,
            method=(
                "Extracted the 350M stop-gradient and insertion-position "
                "controls from Appendix Tables 4-5, preserving their separate "
                "numerical frame from main Table 2."
            ),
            code_ref=(
                "research/experiments/artifacts/external_code/"
                "pap_gpas_neurips2025/GPAS_NeurIPS_2025.pdf, "
                "Appendix A, Equation 12, Tables 4-5"
            ),
            trust={
                "design": "adequate",
                "replication": "weak",
                "scope_match": "weak",
                "directness": "strong",
                "limitations": [
                    "Direct controlled ablations at one reported 350M scale.",
                    "No error bars or multi-seed aggregation are reported.",
                    "The appendix frame is not numerically interchangeable "
                    "with main Table 2.",
                    "No local OPHIS arm is tested.",
                ],
            },
            assessment={
                "relation": "supports",
                "strength": "moderate",
                "limitations": [
                    "Supports stop-gradient and default placement within the "
                    "paper's 350M ablation frame.",
                    "Does not establish transfer to OPHIS.",
                    "Does not authorize implementation, launch, adoption, or SOTA.",
                ],
            },
            tags=(
                "gpas",
                "stop_gradient",
                "placement_ablation",
                "scope_limited",
            ),
        ),
        _record(
            evidence_id=EVIDENCE_DYNAMICS,
            claim_id=CLAIM_DYNAMICS,
            facts=dynamics_facts,
            method=(
                "Read the 1B activation-variance, gradient-norm, weight-norm, "
                "and post-SFT layer-removal sections; separated reported "
                "diagnostic patterns from an unproven complete causal mediation."
            ),
            code_ref=(
                "research/experiments/artifacts/external_code/"
                "pap_gpas_neurips2025/GPAS_NeurIPS_2025.pdf, "
                "Sections 5.2-5.5, Figures 4-7; Section 6"
            ),
            trust={
                "design": "adequate",
                "replication": "weak",
                "scope_match": "weak",
                "directness": "adequate",
                "limitations": [
                    "Primary-source diagnostic figures and accompanying prose.",
                    "Plots provide no exact series, error bars, or multi-seed "
                    "aggregation.",
                    "Layer removal occurs after SFT and is not a pretraining "
                    "mediation analysis.",
                    "No local OPHIS diagnostic is tested.",
                ],
            },
            assessment={
                "relation": "supports",
                "strength": "moderate",
                "limitations": [
                    "Supports the reported paper-scoped dynamics patterns.",
                    "Only weakly supports a complete causal explanation for "
                    "perplexity gains.",
                    "Does not authorize implementation, launch, adoption, or SOTA.",
                ],
            },
            tags=(
                "gpas",
                "activation_variance",
                "gradient_norm",
                "layer_importance",
                "mechanism_evidence",
                "scope_limited",
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


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        check=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _verify_archive_members(archive_path: Path) -> dict[str, Any]:
    observed: dict[str, str] = {}
    regular_files = 0
    symlinks = 0
    with tempfile.TemporaryDirectory(prefix="ophis-gpas-capsule-") as temp_dir:
        tar_path = Path(temp_dir) / "source.tar"
        with tar_path.open("wb") as handle:
            subprocess.run(
                ["zstd", "-q", "-d", "-c", str(archive_path)],
                check=True,
                stdout=handle,
            )
        with tarfile.open(tar_path, "r:") as archive:
            members = archive.getmembers()
            unsafe = [
                member.name
                for member in members
                if (
                    (
                        member.name != "gpas-31980688"
                        and not member.name.startswith("gpas-31980688/")
                    )
                    or Path(member.name).is_absolute()
                    or ".." in Path(member.name).parts
                )
            ]
            if unsafe:
                raise SchemaError(f"unsafe or misrooted GPAS archive members: {unsafe}")
            regular_files = sum(member.isfile() for member in members)
            symlinks = sum(member.issym() or member.islnk() for member in members)
            if regular_files != 30 or symlinks != 0:
                raise SchemaError(
                    "GPAS archive member accounting mismatch: "
                    f"regular_files={regular_files}, symlinks={symlinks}"
                )
            for relative, expected in VERIFIED_ARCHIVE_FILES.items():
                member_name = f"gpas-31980688/{relative}"
                try:
                    member = archive.getmember(member_name)
                except KeyError as exc:
                    raise SchemaError(
                        f"missing GPAS archive member: {member_name}"
                    ) from exc
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SchemaError(f"archive member is not a file: {member_name}")
                digest = hashlib.sha256(extracted.read()).hexdigest()
                if digest != expected:
                    raise SchemaError(
                        f"archive member mismatch for {relative}: "
                        f"expected={expected} observed={digest}"
                    )
                observed[relative] = digest
    return {
        "regular_files": regular_files,
        "symlinks": symlinks,
        "selected_sha256": observed,
    }


def _verify_bundle(bundle_path: Path) -> dict[str, str]:
    subprocess.run(
        ["git", "bundle", "verify", str(bundle_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    heads = _run_text(["git", "bundle", "list-heads", str(bundle_path)])
    expected_head = f"{COMMIT} {BUNDLE_REF}"
    if heads != expected_head:
        raise SchemaError(
            f"GPAS bundle head mismatch: expected={expected_head!r} observed={heads!r}"
        )

    with tempfile.TemporaryDirectory(prefix="ophis-gpas-bundle-") as temp_dir:
        bare = Path(temp_dir) / "verify.git"
        subprocess.run(
            ["git", "init", "--bare", "--quiet", str(bare)],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(bare),
                "fetch",
                "--quiet",
                str(bundle_path),
                BUNDLE_REF,
            ],
            check=True,
        )
        fetched_commit = _run_text(
            ["git", "-C", str(bare), "rev-parse", "FETCH_HEAD^{commit}"]
        )
        fetched_tree = _run_text(
            ["git", "-C", str(bare), "rev-parse", "FETCH_HEAD^{tree}"]
        )
        subprocess.run(
            ["git", "-C", str(bare), "fsck", "--full", "--strict"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if fetched_commit != COMMIT or fetched_tree != TREE:
        raise SchemaError(
            "GPAS bundle commit/tree mismatch: "
            f"expected={COMMIT}/{TREE} observed={fetched_commit}/{fetched_tree}"
        )
    return {"ref": BUNDLE_REF, "commit": fetched_commit, "tree": fetched_tree}


def _verify_pdf(pdf_path: Path) -> dict[str, Any]:
    if pdf_path.read_bytes()[:5] != b"%PDF-":
        raise SchemaError("GPAS proceedings artifact does not have PDF magic")
    info = _run_text(["pdfinfo", str(pdf_path)])
    pages: int | None = None
    for line in info.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
            break
    if pages != 26:
        raise SchemaError(f"GPAS PDF page-count mismatch: expected=26 observed={pages}")
    first_page = _run_text(
        ["pdftotext", "-f", "1", "-l", "1", str(pdf_path), "-"]
    )
    required = (
        "GPAS: Accelerating Convergence of LLM Pretraining",
        "Gradient-Preserving Activation Scaling",
        "Tianhao Chen",
        "Can Yang",
        "NeurIPS 2025",
    )
    missing = [fragment for fragment in required if fragment not in first_page]
    if missing:
        raise SchemaError(f"GPAS PDF identity fragments missing: {missing}")
    return {"pages": pages, "identity_fragments_verified": list(required)}


def _verify_manifest(manifest: dict[str, Any]) -> None:
    repository = dict(manifest.get("repository", {}))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("classification")
        != "external_mechanism_source_reference"
        or manifest.get("paper_id") != PAPER_ID
        or manifest.get("external_role")
        != "gradient_preserving_residual_activation_scaling_source"
        or repository.get("url") != "https://github.com/dandingsky/GPAS"
        or repository.get("commit") != COMMIT
        or repository.get("tree") != TREE
        or repository.get("tracked_paths") != 30
        or repository.get("gitlinks") != []
    ):
        raise SchemaError("GPAS manifest identity/repository fields mismatch")

    expected_manifest_artifacts = {
        "gpas-31980688.tar.zst": ARTIFACTS["gpas-31980688.tar.zst"],
        "gpas-31980688.bundle": ARTIFACTS["gpas-31980688.bundle"],
        "GPAS_NeurIPS_2025.pdf": ARTIFACTS["GPAS_NeurIPS_2025.pdf"],
    }
    observed_manifest_artifacts = {
        str(item.get("path")): {
            "bytes": item.get("bytes"),
            "sha256": item.get("sha256"),
        }
        for item in manifest.get("artifacts", [])
    }
    if observed_manifest_artifacts != expected_manifest_artifacts:
        raise SchemaError(
            "GPAS manifest artifact bindings mismatch: "
            f"expected={expected_manifest_artifacts} "
            f"observed={observed_manifest_artifacts}"
        )

    verified_files = {
        str(item.get("path")): item.get("sha256")
        for item in manifest.get("verified_files", [])
    }
    if verified_files != VERIFIED_ARCHIVE_FILES:
        raise SchemaError(
            "GPAS manifest verified-file bindings mismatch: "
            f"expected={VERIFIED_ARCHIVE_FILES} observed={verified_files}"
        )
    license_record = dict(manifest.get("license", {}))
    verification = dict(manifest.get("verification", {}))
    if (
        license_record.get("declared_family") != "Apache-2.0"
        or license_record.get("sha256") != VERIFIED_ARCHIVE_FILES["LICENSE"]
        or verification.get("zstd_integrity_test") is not True
        or verification.get("bundle_verification_passed") is not True
        or verification.get("bundle_records_complete_history") is not True
        or verification.get("archive_commit_matches_resolved_head") is not True
        or verification.get("archive_path_accounting_matches_git_tree") is not True
        or verification.get("local_sota_adoption_claimed") is not False
    ):
        raise SchemaError("GPAS manifest license/verification fields mismatch")


def verify_artifacts() -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, expected in ARTIFACTS.items():
        path = ARTIFACT_ROOT / name
        if not path.is_file():
            raise SchemaError(f"missing GPAS artifact: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if actual != expected:
            raise SchemaError(
                f"artifact mismatch for {name}: expected={expected}, observed={actual}"
            )
        observed[name] = actual

    manifest = json.loads(
        (ARTIFACT_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    _verify_manifest(manifest)

    archive_path = ARTIFACT_ROOT / "gpas-31980688.tar.zst"
    subprocess.run(
        ["zstd", "-q", "-t", str(archive_path)],
        check=True,
    )
    observed["archive_members"] = _verify_archive_members(archive_path)
    observed["bundle"] = _verify_bundle(
        ARTIFACT_ROOT / "gpas-31980688.bundle"
    )
    observed["pdf"] = _verify_pdf(ARTIFACT_ROOT / "GPAS_NeurIPS_2025.pdf")

    if not INVENTORY_PATH.is_file():
        raise SchemaError(f"missing external-code inventory: {INVENTORY_PATH}")
    inventory = INVENTORY_PATH.read_text(encoding="utf-8")
    missing = [
        fragment for fragment in INVENTORY_REQUIRED_FRAGMENTS if fragment not in inventory
    ]
    if missing:
        raise SchemaError(f"GPAS external-code inventory is incomplete: {missing}")
    observed["inventory_fragments_verified"] = list(INVENTORY_REQUIRED_FRAGMENTS)
    return observed


def _classify(store: Any, record: Any, kind: str) -> str:
    existing = store.by_id().get(str(record.registry_id))
    if existing is None:
        return "missing"
    if canonical_json(existing.to_dict()) != canonical_json(record.to_dict()):
        raise SchemaError(
            f"{kind} ID {record.registry_id!r} exists with different content"
        )
    return "skip_exact"


def preflight_or_apply(*, apply: bool) -> dict[str, Any]:
    verified = verify_artifacts()
    registry = ResearchRegistry(RESEARCH_ROOT)
    validation = registry.validate(check_generated_state=False).to_dict()
    if validation.get("warnings"):
        raise SchemaError(
            "pre-existing authoritative validation warnings: "
            f"{validation['warnings']}"
        )

    paper_record, claim_records, evidence_records = records()
    entries = {
        "papers": [(registry.papers, paper_record, "paper")],
        "claims": [
            (registry.claims, record, "claim") for record in claim_records
        ],
        "literature_evidence": [
            (registry.literature_evidence, record, "literature evidence")
            for record in evidence_records
        ],
    }

    classifications: dict[str, list[tuple[Any, Any, str, str]]] = {}
    for category, category_entries in entries.items():
        classifications[category] = [
            (store, record, kind, _classify(store, record, kind))
            for store, record, kind in category_entries
        ]

    actions: dict[str, list[dict[str, str]]] = {
        category: [] for category in entries
    }
    for category, category_entries in classifications.items():
        for store, record, _kind, classification in category_entries:
            action = classification
            if classification == "missing":
                if apply:
                    store.add(record)
                    action = "appended"
                else:
                    action = "would_append"
            actions[category].append(
                {"id": str(record.registry_id), "action": action}
            )

    if apply:
        post = registry.validate(check_generated_state=False).to_dict()
        if post.get("warnings"):
            raise SchemaError(
                f"post-append authoritative validation warnings: {post['warnings']}"
            )
    else:
        post = validation

    return {
        "mode": "apply" if apply else "preflight",
        "artifact_verification": verified,
        "actions": actions,
        "authoritative_validation": post,
        "authority": NO_AUTHORITY,
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
