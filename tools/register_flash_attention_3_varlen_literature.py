#!/usr/bin/env python3
"""Verify and register the FlashAttention-3 varlen API provenance capsule.

The NeurIPS 2024 proceedings paper establishes the Hopper-oriented FA3
algorithm.  The cumulative-offset contract is narrower and comes from the
official Dao-AILab repository at the exact commit frozen below.  This tool
keeps those source roles separate: it records strong evidence for the external
API semantics, while explicitly recording that OPHIS packer-side generation
and end-to-end speed remain untested.

The default mode is a read-only preflight. ``--apply`` appends missing typed
records through ``ResearchRegistry`` under the global literature-ingest lock.
Replays are idempotent only when an existing ID has exactly the same canonical
typed serialization; divergent ID reuse fails closed.
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
    / "pap_flash_attention_3_neurips2024"
)
INVENTORY_PATH = (
    RESEARCH_ROOT / "experiments" / "artifacts" / "external_code" / "INVENTORY.md"
)
LOCK_PATH = REPO_ROOT / "tmp" / "literature_ingest_transactions" / "global.lock"

PAPER_ID = "pap_flash_attention_3_neurips2024"
CLAIM_ID = "clm_fa3_varlen_external_cuseqlens_contract"
EVIDENCE_ID = "evd_lit_fa3_varlen_api_contract"
CREATED_AT = "2026-07-29T15:26:42Z"
CREATED_BY = "codex_fa3_varlen_provenance_registration"

COMMIT = "c75d019dea9d910312974417bc28f190dfdda6d9"
TREE = "be7da1c32fdca738d4d27f96d9e365e87d9f6658"
LICENSE_SHA256 = "8c9ccb96c065e706135b6cbad279b721da6156e51f3a5f27c6b3329af9416d73"

ARTIFACTS: dict[str, dict[str, Any]] = {
    "manifest.json": {
        "bytes": 7790,
        "sha256": "e52cbac891a067473a67104f1d86cf2b9f04c2ffc2fcdbbedf5d83ab9c531efd",
    },
    "FlashAttention-3_NeurIPS_2024.pdf": {
        "bytes": 697391,
        "sha256": "73e785a3dc6e378d8940dbc2f682f06e1fa9b0ed26295aeb0a6e7ce6bb2f8291",
    },
    "flash-attention-c75d019d.tar.zst": {
        "bytes": 13287191,
        "sha256": "89a16aa92cb9b84bcf6c8abab576e3304216092f15ee3e0340f734ff1970d9e8",
    },
    "flash-attention-c75d019d.bundle": {
        "bytes": 20308909,
        "sha256": "d5d096b964a66625b178a19e044c36a4fdbdcbe83720ac5f5976ee3221dc5e99",
    },
}

VERIFIED_ARCHIVE_FILES = {
    "LICENSE": LICENSE_SHA256,
    "README.md": "7bcb7ca466dcb819822003c5c15a8124e4866f4d446a5c60382615128c81c3a6",
    ".gitmodules": "0c57db58f50c4c7c7aa10c7a0b14d253cb6bfab5368caee0a7ca4c7d3f003b59",
    "hopper/flash_attn_interface.py": (
        "887eb85ea5380707a78cd15dc04169e654f4dee58f9bddab49a2ce71f1680e35"
    ),
    "hopper/test_flash_attn.py": (
        "61c61341e2a871cb65934d503e252a2ba023a2c11703e1558a61adff020c71cf"
    ),
    "hopper/setup.py": (
        "e3d9e67d1ec2bc61e4dec5591b46072f1b86ca7a185bd5dac7ec5e3cdce0933a"
    ),
    "hopper/generate_kernels.py": (
        "99b1d47053befd7cc4ca7e4c5c5a81c72e6f108979ccadd036f2b802ebb982eb"
    ),
    "hopper/flash_fwd_kernel_sm90.h": (
        "1d1e955e208b7830127573beece10d06416953b65143013a8ee7e57189967ea6"
    ),
    "hopper/flash_bwd_kernel_sm90.h": (
        "2e0052006b2733373709cdeed0af32edcaea69958936c748275dfb61ed115feb"
    ),
}

COMMON_ARTIFACT_PATHS = tuple(
    (
        "research/experiments/artifacts/external_code/"
        f"pap_flash_attention_3_neurips2024/{name}"
    )
    for name in (
        "manifest.json",
        "FlashAttention-3_NeurIPS_2024.pdf",
        "flash-attention-c75d019d.tar.zst",
        "flash-attention-c75d019d.bundle",
    )
)

INVENTORY_REQUIRED_FRAGMENTS = (
    "`pap_flash_attention_3_neurips2024/flash-attention-c75d019d.tar.zst`",
    "`pap_flash_attention_3_neurips2024/flash-attention-c75d019d.bundle`",
    "`pap_flash_attention_3_neurips2024/FlashAttention-3_NeurIPS_2024.pdf`",
)


def paper() -> PaperRecord:
    return PaperRecord(
        paper_id=PAPER_ID,
        title=(
            "FlashAttention-3: Fast and Accurate Attention with "
            "Asynchrony and Low-precision"
        ),
        authors=(
            "Jay Shah",
            "Ganesh Bikshandi",
            "Ying Zhang",
            "Vijay Thakkar",
            "Pradeep Ramani",
            "Tri Dao",
        ),
        year=2024,
        venue={
            "name": (
                "Advances in Neural Information Processing Systems 37 "
                "(NeurIPS 2024), Main Conference Track"
            ),
            "peer_reviewed": True,
        },
        urls={
            "primary": (
                "https://papers.nips.cc/paper_files/paper/2024/hash/"
                "7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html"
            ),
            "pdf": (
                "https://papers.nips.cc/paper_files/paper/2024/file/"
                "7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf"
            ),
            "doi": "https://doi.org/10.52202/079017-2193",
            "code": "https://github.com/Dao-AILab/flash-attention",
            "code_commit": (
                "https://github.com/Dao-AILab/flash-attention/tree/" + COMMIT
            ),
        },
        retrieved_at="2026-07-29",
        version=(
            "NeurIPS 2024 proceedings paper; official repository API inspected "
            f"separately at commit {COMMIT}"
        ),
        status="active",
        tags=(
            "attention",
            "flash_attention_3",
            "hopper",
            "varlen",
            "cu_seqlens",
            "neurips_2024",
            "primary_source",
            "external_code_captured",
        ),
        notes=(
            "Organizations: Colfax Research, Meta, NVIDIA, Georgia Institute "
            "of Technology, Princeton University, and Together AI. The paper "
            "supports FA3's Hopper algorithm and reported H100 kernel results. "
            "The cumulative-offset API contract below is sourced from the later "
            "exact official code commit and is not represented as paper text."
        ),
    )


def claim() -> ClaimRecord:
    return ClaimRecord(
        claim_id=CLAIM_ID,
        paper_id=PAPER_ID,
        statement=(
            "At official Dao-AILab commit c75d019dea9d910312974417bc28f190dfdda6d9, "
            "the Hopper flash_attn_varlen_func accepts packed Q/K/V together "
            "with cu_seqlens_q, cu_seqlens_k, max_seqlen_q, and max_seqlen_k; "
            "its autograd path passes the cumulative offsets into the CUDA "
            "forward operation, saves them, and reuses them in backward, while "
            "official Hopper tests exercise CUDA int32 cumulative-offset vectors "
            "and compare varlen forward and backward results with reference attention."
        ),
        claim_type="method_definition",
        scope={
            "applies_to": "official_flashattention3_hopper_varlen_api",
            "hardware_family": "nvidia_hopper",
            "domain": "packed_variable_length_attention",
            "direction": "api_semantics_only",
            "code_commit": COMMIT,
            "sequence_metadata": [
                "cu_seqlens_q",
                "cu_seqlens_k",
                "max_seqlen_q",
                "max_seqlen_k",
            ],
        },
        limitations=(
            "The NeurIPS paper establishes the Hopper FA3 context; this exact "
            "API contract is extracted from the later official code commit.",
            "The API does not prescribe how an OPHIS document packer should "
            "generate, pin, transfer, synchronize, or validate cumulative offsets.",
            "The upstream tests do not exercise the proposed OPHIS boundary "
            "sidecar, sample-order equivalence, stream lifetime, graph stability, "
            "or optimizer-state parity.",
            "No local end-to-end H200 throughput or validation-BPB effect is tested.",
            "Three pinned gitlink submodules are recorded but not vendored in "
            "the normalized source archive.",
        ),
        locator=(
            f"Dao-AILab/flash-attention@{COMMIT}: "
            "hopper/flash_attn_interface.py lines 59-121, 642-740, 890-934; "
            "hopper/test_flash_attn.py lines 404-660 and 1226-1251"
        ),
        extracted_at="2026-07-29",
        tags=(
            "flash_attention_3",
            "hopper",
            "varlen",
            "cu_seqlens",
            "method_definition",
            "exact_commit",
        ),
    )


def evidence() -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=EVIDENCE_ID,
        source_type="literature",
        paper_ids=(PAPER_ID,),
        claim_ids=(CLAIM_ID,),
        run_ids=(),
        experiment_id="",
        hypothesis_ids=(),
        facts={
            "reported": {
                "paper_context": {
                    "hardware": "H100 / NVIDIA Hopper",
                    "method": "FlashAttention-3",
                    "paper_api_contract_claimed": False,
                },
                "external_api_contract": {
                    "entrypoint": "flash_attn_varlen_func",
                    "required_sequence_arguments": [
                        "cu_seqlens_q",
                        "cu_seqlens_k",
                        "max_seqlen_q",
                        "max_seqlen_k",
                    ],
                    "offsets_forwarded_to_cuda_forward": True,
                    "offsets_saved_for_backward": True,
                    "offsets_reused_by_backward": True,
                    "official_test_offset_dtype": "torch.int32",
                    "official_forward_reference_comparison": True,
                    "official_backward_reference_comparison": True,
                },
            },
            "source_separation": {
                "hopper_algorithm_and_reported_kernel_results": (
                    "NeurIPS 2024 proceedings paper"
                ),
                "cumulative_offset_api_contract": (
                    f"official Dao-AILab code commit {COMMIT}"
                ),
            },
            "local_transfer_assessments": {
                "ophis_packer_sidecar_generation": {
                    "relation": "not_tested",
                    "strength": "weak",
                    "reason": (
                        "Neither source constructs or verifies OPHIS document "
                        "boundaries from the local best-fit packer."
                    ),
                },
                "ophis_end_to_end_h200_speed": {
                    "relation": "not_tested",
                    "strength": "weak",
                    "reason": (
                        "Kernel/API provenance does not measure charged OPHIS "
                        "training throughput or endpoint validation BPB."
                    ),
                },
            },
            "code_snapshot": {
                "repository": "https://github.com/Dao-AILab/flash-attention",
                "commit": COMMIT,
                "tree": TREE,
                "manifest_sha256": ARTIFACTS["manifest.json"]["sha256"],
                "archive_sha256": ARTIFACTS[
                    "flash-attention-c75d019d.tar.zst"
                ]["sha256"],
                "bundle_sha256": ARTIFACTS[
                    "flash-attention-c75d019d.bundle"
                ]["sha256"],
                "paper_sha256": ARTIFACTS[
                    "FlashAttention-3_NeurIPS_2024.pdf"
                ]["sha256"],
                "interface_sha256": VERIFIED_ARCHIVE_FILES[
                    "hopper/flash_attn_interface.py"
                ],
                "tests_sha256": VERIFIED_ARCHIVE_FILES[
                    "hopper/test_flash_attn.py"
                ],
                "license": "BSD-3-Clause",
                "license_sha256": LICENSE_SHA256,
                "excluded_gitlinks": 3,
            },
            "organizations": [
                "Colfax Research",
                "Meta",
                "NVIDIA",
                "Georgia Institute of Technology",
                "Princeton University",
                "Together AI",
            ],
            "local_sota_adoption_claimed": False,
        },
        analysis={
            "method": (
                "Read the official NeurIPS proceedings paper for FA3/Hopper "
                "scope, then inspected the exact official-code entrypoint, "
                "custom-op wrapper, autograd lifetime, and Hopper tests. "
                "Separated external API identity from any unperformed local "
                "sidecar-generation or end-to-end speed experiment."
            ),
            "code_ref": (
                "research/experiments/artifacts/external_code/"
                "pap_flash_attention_3_neurips2024/manifest.json; "
                f"hopper/flash_attn_interface.py@{COMMIT}:59-121,642-740,890-934; "
                f"hopper/test_flash_attn.py@{COMMIT}:404-660,1226-1251"
            ),
        },
        trust={
            "design": "not_applicable",
            "replication": "adequate",
            "scope_match": "weak",
            "directness": "strong",
            "limitations": [
                "Strong direct evidence for the external API semantics only.",
                "The official tests were captured and inspected, not executed "
                "in the OPHIS H200 runtime by this provenance task.",
                "The paper and inspected API commit are different temporal versions.",
                "No local sidecar-generation or end-to-end speed result exists.",
            ],
        },
        assessment={
            "relation": "supports",
            "strength": "strong",
            "limitations": [
                "Strongly supports only the typed external cumulative-offset API claim.",
                "Local packer-side boundary generation is explicitly not_tested/weak.",
                "Local end-to-end H200 speed is explicitly not_tested/weak.",
                "This record does not authorize implementation, launch, adoption, or SOTA.",
            ],
        },
        artifact_paths=COMMON_ARTIFACT_PATHS,
        created_at=CREATED_AT,
        created_by=CREATED_BY,
        tags=(
            "flash_attention_3",
            "hopper",
            "varlen",
            "cu_seqlens",
            "exact_commit",
            "api_contract",
            "local_transfer_not_tested",
        ),
    )


def records() -> tuple[PaperRecord, ClaimRecord, EvidenceRecord]:
    return paper(), claim(), evidence()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_archive_members(archive_path: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="ophis-fa3-capsule-") as temp_dir:
        tar_path = Path(temp_dir) / "source.tar"
        with tar_path.open("wb") as handle:
            subprocess.run(
                ["zstd", "-q", "-d", "-c", str(archive_path)],
                check=True,
                stdout=handle,
            )
        with tarfile.open(tar_path, "r:") as archive:
            for relative, expected in VERIFIED_ARCHIVE_FILES.items():
                member_name = f"flash-attention-c75d019d/{relative}"
                member = archive.getmember(member_name)
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
    return observed


def verify_artifacts() -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, expected in ARTIFACTS.items():
        path = ARTIFACT_ROOT / name
        if not path.is_file():
            raise SchemaError(f"missing FA3 artifact: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if actual != expected:
            raise SchemaError(
                f"artifact mismatch for {name}: expected={expected}, observed={actual}"
            )
        observed[name] = actual

    manifest = json.loads(
        (ARTIFACT_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    repository = dict(manifest.get("repository", {}))
    if (
        manifest.get("paper_id") != PAPER_ID
        or repository.get("commit") != COMMIT
        or repository.get("tree") != TREE
    ):
        raise SchemaError("FA3 manifest paper/commit/tree identity mismatch")

    subprocess.run(
        ["zstd", "-q", "-t", str(ARTIFACT_ROOT / "flash-attention-c75d019d.tar.zst")],
        check=True,
    )
    subprocess.run(
        ["git", "bundle", "verify", str(ARTIFACT_ROOT / "flash-attention-c75d019d.bundle")],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pdf_head = (
        ARTIFACT_ROOT / "FlashAttention-3_NeurIPS_2024.pdf"
    ).read_bytes()[:5]
    if pdf_head != b"%PDF-":
        raise SchemaError("FA3 proceedings artifact does not have PDF magic")

    observed["archive_members"] = _verify_archive_members(
        ARTIFACT_ROOT / "flash-attention-c75d019d.tar.zst"
    )

    inventory = INVENTORY_PATH.read_text(encoding="utf-8")
    missing = [
        fragment for fragment in INVENTORY_REQUIRED_FRAGMENTS if fragment not in inventory
    ]
    if missing:
        raise SchemaError(f"FA3 external-code inventory is incomplete: {missing}")
    observed["inventory_fragments_verified"] = list(INVENTORY_REQUIRED_FRAGMENTS)
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
            f"pre-existing authoritative validation warnings: {validation['warnings']}"
        )

    paper_record, claim_record, evidence_record = records()
    actions = {
        "papers": [
            {
                "id": paper_record.paper_id,
                "action": _action(
                    registry.papers, paper_record, "paper", apply=apply
                ),
            }
        ],
        "claims": [
            {
                "id": claim_record.claim_id,
                "action": _action(
                    registry.claims, claim_record, "claim", apply=apply
                ),
            }
        ],
        "literature_evidence": [
            {
                "id": evidence_record.evidence_id,
                "action": _action(
                    registry.literature_evidence,
                    evidence_record,
                    "literature evidence",
                    apply=apply,
                ),
            }
        ],
    }

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
