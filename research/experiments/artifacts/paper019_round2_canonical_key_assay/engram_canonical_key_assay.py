#!/usr/bin/env python3
"""Reproducible offline mediator assay for paper 019 round 2.

The program reads the pinned Engram demo, the current OPHIS tokenizer, and a
frozen local validation sample.  It writes only the files declared in
``assay_spec.json`` beside this script.  It never imports or edits train.py or
lib.py, never initializes CUDA, and never launches training.

The assay establishes only key-collapse and occurrence-exposure mediators.  It
does not establish source-exact equivalence to DeepSeek-V3, a BPB effect, or
authority to run an experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import tarfile
from typing import Any, Iterable

import pyarrow.parquet as pq
import tokenizers
from tokenizers import Regex, normalizers


ARTIFACT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ARTIFACT_DIR.parents[3]
SENTINEL = "\uE000"
PROTECTED_CLASSES = (
    "special",
    "bos",
    "single_raw_byte",
    "decoded_replacement_character",
    "empty_decoding",
)
OUTPUT_NAMES = {
    "result.json",
    "REPORT.md",
    "projection.jsonl",
    "alias_sets.jsonl",
    "unexplained_alias_sets.jsonl",
    "packing_events.jsonl",
    "output_manifest.json",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)
    else:
        text = json.dumps(
            value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
    return (text + "\n").encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def atomic_write(path: Path, data: bytes) -> None:
    if path.parent.resolve() != ARTIFACT_DIR:
        raise ValueError(f"refusing output outside artifact directory: {path}")
    if path.name not in OUTPUT_NAMES:
        raise ValueError(f"undeclared output: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resolve_project_path(value: str) -> Path:
    expanded = Path(value).expanduser()
    return expanded if expanded.is_absolute() else PROJECT_ROOT / expanded


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, observed {actual} ({path})"
        )
    return actual


def extract_demo(archive_path: Path, member: str) -> bytes:
    try:
        proc = subprocess.run(
            ["zstd", "-dc", str(archive_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("zstd executable is required to read the pinned archive") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"zstd failed for {archive_path}: {exc.stderr.decode('utf-8', 'replace')}"
        ) from exc
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as archive:
        selected = archive.extractfile(member)
        if selected is None:
            raise RuntimeError(f"missing pinned demo member {member}")
        return selected.read()


class SourceNormalizer:
    """Exact stage order present in the pinned Engram demonstration."""

    def __init__(self) -> None:
        self.stages = [
            ("NFKC", normalizers.NFKC()),
            ("NFD", normalizers.NFD()),
            ("StripAccents", normalizers.StripAccents()),
            ("Lowercase", normalizers.Lowercase()),
            (
                "WhitespaceCollapse",
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            ),
            (
                "SingleSpaceSentinel",
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
            ),
            ("Strip", normalizers.Strip()),
            ("SentinelRestore", normalizers.Replace(SENTINEL, " ")),
        ]
        self.sequence = normalizers.Sequence([stage for _, stage in self.stages])

    def trace(self, text: str) -> tuple[str, list[str], list[dict[str, str]]]:
        current = text
        changed: list[str] = []
        trace: list[dict[str, str]] = []
        for name, stage in self.stages:
            updated = stage.normalize_str(current)
            if updated != current:
                changed.append(name)
                trace.append({"stage": name, "before": current, "after": updated})
            current = updated
        sequence_value = self.sequence.normalize_str(text)
        if sequence_value != current:
            raise AssertionError("stage trace diverges from tokenizers Sequence")
        return current, changed, trace


def verify_source_demo(demo: bytes) -> list[str]:
    text = demo.decode("utf-8")
    required_fragments = [
        "normalizers.NFKC()",
        "normalizers.NFD()",
        "normalizers.StripAccents()",
        "normalizers.Lowercase()",
        'normalizers.Replace(Regex(r"[ \\t\\r\\n]+"), " ")',
        'normalizers.Replace(Regex(r"^ $"), SENTINEL)',
        "normalizers.Strip()",
        'normalizers.Replace(SENTINEL, " ")',
        'if "�" in text:',
        "key = norm if norm else text",
        "for tid in range(vocab_size):",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise RuntimeError(f"pinned demo no longer matches normalization spec: {missing}")
    return required_fragments


def verify_local_source(spec: dict[str, Any]) -> dict[str, Any]:
    source = spec["local_source"]
    paths = {
        "train": resolve_project_path(source["train_path"]),
        "lib": resolve_project_path(source["lib_path"]),
        "prepare": resolve_project_path(source["prepare_path"]),
        "data_split": resolve_project_path(source["data_split_path"]),
    }
    hashes = {
        name: require_hash(paths[name], source[f"{name}_sha256"], f"local {name}")
        for name in paths
    }
    train_text = paths["train"].read_text(encoding="utf-8")
    lib_text = paths["lib"].read_text(encoding="utf-8")
    train_fragments = [
        "prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)",
        "prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)",
        "((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size",
        "((prev2_idx * lp[0]) ^ (prev_idx * lp[1]) ^ (idx * lp[2]))",
    ]
    lib_fragments = [
        "row_capacity = T + 1",
        "while len(doc_buffer) < buffer_size:",
        "token_lists = tokenizer.encode(doc_batch, prepend=bos_token)",
        "if doc_len <= remaining and doc_len > best_len:",
        "cpu_inputs.copy_(row_buffer[:, :-1])",
        "cpu_targets.copy_(row_buffer[:, 1:])",
    ]
    missing_train = [item for item in train_fragments if item not in train_text]
    missing_lib = [item for item in lib_fragments if item not in lib_text]
    if missing_train or missing_lib:
        raise RuntimeError(
            "local semantics audit failed: "
            f"train missing={missing_train}, lib missing={missing_lib}"
        )
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "sha256": hashes,
        "verified_train_fragments": train_fragments,
        "verified_lib_fragments": lib_fragments,
    }


def load_tokenizer(
    spec: dict[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], dict[int, int]]:
    cfg = spec["tokenizer"]
    directory = resolve_project_path(cfg["directory"])
    tokenizer_path = directory / cfg["pickle_name"]
    token_bytes_path = directory / cfg["token_bytes_name"]
    tokenizer_hash = require_hash(
        tokenizer_path, cfg["pickle_sha256"], "tokenizer pickle"
    )
    token_bytes_hash = require_hash(
        token_bytes_path, cfg["token_bytes_sha256"], "token-byte table"
    )

    # The runtime itself loads this trusted local pickle.  The assay does the
    # same solely to inspect the exact encoding used by the current model.
    with tokenizer_path.open("rb") as handle:
        encoding = pickle.load(handle)

    if encoding.n_vocab != cfg["expected_vocab_size"]:
        raise RuntimeError(
            f"vocab mismatch: expected {cfg['expected_vocab_size']}, "
            f"observed {encoding.n_vocab}"
        )
    special_map = {
        token: int(encoding.encode_single_token(token))
        for token in cfg["special_tokens"]
    }
    bos_id = int(encoding.encode_single_token(cfg["bos_token"]))
    if bos_id != special_map[cfg["bos_token"]]:
        raise AssertionError("BOS ID is inconsistent with the special-token mapping")

    vocab_rows: list[dict[str, Any]] = []
    semantic_digest = hashlib.sha256()
    for token_id in range(encoding.n_vocab):
        raw = encoding.decode_single_token_bytes(token_id)
        decoded = encoding.decode([token_id])
        semantic_digest.update(struct.pack("<Q", token_id))
        semantic_digest.update(struct.pack("<Q", len(raw)))
        semantic_digest.update(raw)
        vocab_rows.append(
            {
                "token_id": token_id,
                "raw_bytes_hex": raw.hex(),
                "decoded_text": decoded,
            }
        )

    metadata = {
        "directory": str(directory),
        "pickle_path": str(tokenizer_path),
        "pickle_sha256": tokenizer_hash,
        "token_bytes_path": str(token_bytes_path),
        "token_bytes_sha256": token_bytes_hash,
        "tokenizer_train_split_metadata_present": (
            directory / "tokenizer_train_split.json"
        ).exists(),
        "name": getattr(encoding, "name", None),
        "vocab_size": encoding.n_vocab,
        "bos_token": cfg["bos_token"],
        "bos_id": bos_id,
        "special_token_ids": special_map,
        "semantic_id_to_raw_bytes_sha256": semantic_digest.hexdigest(),
        "pattern": getattr(encoding, "_pat_str", None),
    }
    return encoding, metadata, vocab_rows, special_map


def build_projection(
    vocab_rows: list[dict[str, Any]],
    special_map: dict[str, int],
    bos_id: int,
    normalizer: SourceNormalizer,
) -> tuple[
    list[int],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    special_ids = set(special_map.values())
    key_to_new: dict[tuple[str, Any], int] = {}
    key_members: dict[tuple[str, Any], list[int]] = defaultdict(list)
    projection: list[int] = []
    output_rows: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    protected_reason_counts: Counter[str] = Counter(
        {reason: 0 for reason in PROTECTED_CLASSES}
    )
    protected_ids_by_reason: dict[str, list[int]] = {
        reason: [] for reason in PROTECTED_CLASSES
    }

    for row in vocab_rows:
        token_id = int(row["token_id"])
        raw = bytes.fromhex(row["raw_bytes_hex"])
        decoded = str(row["decoded_text"])
        reasons: list[str] = []
        if token_id in special_ids:
            reasons.append("special")
        if token_id == bos_id:
            reasons.append("bos")
        if len(raw) == 1:
            reasons.append("single_raw_byte")
        if "\uFFFD" in decoded:
            reasons.append("decoded_replacement_character")
        if decoded == "":
            reasons.append("empty_decoding")
        protected_reason_counts.update(reasons)
        for reason in reasons:
            protected_ids_by_reason[reason].append(token_id)

        normalized_text: str | None
        changed_stages: list[str]
        trace: list[dict[str, str]]
        fallback_to_decoded = False
        if reasons:
            key: tuple[str, Any] = ("protected_identity", token_id)
            normalized_text = None
            changed_stages = []
            trace = []
            namespace = "protected_identity"
        else:
            normalized, changed_stages, trace = normalizer.trace(decoded)
            fallback_to_decoded = not bool(normalized)
            normalized_text = normalized if normalized else decoded
            key = ("normalized_text", normalized_text)
            namespace = "normalized_text"

        canonical_id = key_to_new.get(key)
        if canonical_id is None:
            canonical_id = len(key_to_new)
            key_to_new[key] = canonical_id
        projection.append(canonical_id)
        key_members[key].append(token_id)

        output = {
            "token_id": token_id,
            "canonical_id": canonical_id,
            "key_namespace": namespace,
            "canonical_text": normalized_text,
            "raw_bytes_hex": row["raw_bytes_hex"],
            "decoded_text": decoded,
            "protected_reasons": reasons,
            "normalization_changed_stages": changed_stages,
            "fallback_to_decoded_text": fallback_to_decoded,
        }
        output_rows.append(output)
        by_id[token_id] = {**output, "normalization_trace": trace}

    protected_ids = {
        row["token_id"] for row in output_rows if row["protected_reasons"]
    }
    protected_canonical_ids = [projection[token_id] for token_id in protected_ids]
    if len(protected_canonical_ids) != len(set(protected_canonical_ids)):
        raise AssertionError("protected IDs did not remain one-to-one")
    for token_id in protected_ids:
        if len(key_members[("protected_identity", token_id)]) != 1:
            raise AssertionError(f"protected token {token_id} has an alias")

    alias_rows: list[dict[str, Any]] = []
    unexplained_rows: list[dict[str, Any]] = []
    explanation_counts: Counter[str] = Counter()
    aliased_ids: set[int] = set()
    for key, member_ids in key_members.items():
        if len(member_ids) < 2:
            continue
        if key[0] != "normalized_text":
            raise AssertionError(f"protected namespace unexpectedly aliased: {key}")
        aliased_ids.update(member_ids)
        members = [by_id[token_id] for token_id in member_ids]
        changed = sorted(
            {
                stage
                for member in members
                for stage in member["normalization_changed_stages"]
            }
        )
        decoded_values = {member["decoded_text"] for member in members}
        explanations = list(changed)
        if len(decoded_values) == 1:
            explanations.append("identical_decoded_text")
        explained = bool(explanations)
        if explained:
            explanation_counts.update(explanations)
        alias_row = {
            "canonical_id": projection[member_ids[0]],
            "canonical_text": key[1],
            "member_count": len(member_ids),
            "member_ids": member_ids,
            "explanations": explanations,
            "explained_by_declared_pipeline": explained,
            "members": members,
        }
        alias_rows.append(alias_row)
        if not explained:
            unexplained_rows.append(alias_row)

    alias_rows.sort(key=lambda row: (-row["member_count"], row["canonical_id"]))
    unexplained_rows.sort(
        key=lambda row: (-row["member_count"], row["canonical_id"])
    )
    alias_count = len(alias_rows)
    explained_count = alias_count - len(unexplained_rows)
    new_vocab_size = len(key_to_new)
    summary = {
        "original_vocab_size": len(vocab_rows),
        "canonical_vocab_size": new_vocab_size,
        "collapsed_key_count": len(vocab_rows) - new_vocab_size,
        "vocabulary_key_cardinality_collapse": (
            (len(vocab_rows) - new_vocab_size) / len(vocab_rows)
        ),
        "protected_id_count": len(protected_ids),
        "protected_reason_counts": dict(sorted(protected_reason_counts.items())),
        "protected_ids_by_reason": protected_ids_by_reason,
        "protected_ids": sorted(protected_ids),
        "protected_ids_one_to_one": True,
        "alias_set_count": alias_count,
        "aliased_token_id_count": len(aliased_ids),
        "alias_set_explained_count": explained_count,
        "alias_set_unexplained_count": len(unexplained_rows),
        "alias_set_explained_fraction": (
            explained_count / alias_count if alias_count else 1.0
        ),
        "explanation_label_counts": dict(sorted(explanation_counts.items())),
        "aliased_ids": sorted(aliased_ids),
    }
    return projection, output_rows, alias_rows, unexplained_rows, summary


def load_data_split(path: Path) -> dict[str, list[int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    train = value.get("train")
    test = value.get("test")
    if not isinstance(train, list) or not isinstance(test, list):
        raise RuntimeError("data_split.json must contain train/test lists")
    if set(train) & set(test):
        raise RuntimeError("train/test shard overlap")
    return {"train": train, "test": test}


def iter_document_batches(
    shard_paths: list[tuple[int, Path]], batch_size: int
) -> Iterable[list[dict[str, Any]]]:
    ordinal = 0
    for shard_id, path in shard_paths:
        parquet = pq.ParquetFile(path)
        for row_group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group_index)
            texts = table.column("text").to_pylist()
            for start in range(0, len(texts), batch_size):
                records: list[dict[str, Any]] = []
                for offset, text in enumerate(texts[start : start + batch_size]):
                    if not isinstance(text, str):
                        raise RuntimeError(
                            f"non-string text at shard={shard_id}, "
                            f"row_group={row_group_index}, row={start + offset}"
                        )
                    records.append(
                        {
                            "source_doc_ordinal": ordinal,
                            "shard_id": shard_id,
                            "row_group": row_group_index,
                            "row_in_group": start + offset,
                            "text": text,
                        }
                    )
                    ordinal += 1
                yield records


def digest_documents(records: list[dict[str, Any]], digest: Any) -> None:
    for record in records:
        encoded = record["text"].encode("utf-8")
        digest.update(struct.pack("<Q", record["source_doc_ordinal"]))
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)


def digest_id_rows(rows: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for value in row:
            digest.update(struct.pack("<q", int(value)))
    return digest.hexdigest()


def pack_frozen_sample(
    encoding: Any,
    bos_id: int,
    shard_paths: list[tuple[int, Path]],
    cfg: dict[str, Any],
) -> tuple[list[list[int]], list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    tokenizer_batch_size = int(cfg["tokenizer_batch_size"])
    buffer_size = int(cfg["packing_buffer_size"])
    context_length = int(cfg["context_length"])
    packed_rows = int(cfg["packed_rows"])
    row_capacity = context_length + 1
    batches = iter(iter_document_batches(shard_paths, tokenizer_batch_size))
    doc_buffer: list[dict[str, Any]] = []
    loaded_digest = hashlib.sha256()
    documents_loaded = 0
    packing_events: list[dict[str, Any]] = []

    def refill() -> None:
        nonlocal documents_loaded
        records = next(batches)
        digest_documents(records, loaded_digest)
        token_lists = encoding.encode_ordinary_batch(
            [record["text"] for record in records],
            num_threads=8,
        )
        for record, token_ids in zip(records, token_lists, strict=True):
            doc_buffer.append(
                {
                    **{key: value for key, value in record.items() if key != "text"},
                    "tokens": [bos_id, *token_ids],
                }
            )
        documents_loaded += len(records)

    full_rows: list[list[int]] = []
    for row_index in range(packed_rows):
        row: list[int] = []
        while len(row) < row_capacity:
            while len(doc_buffer) < buffer_size:
                refill()
            remaining = row_capacity - len(row)
            best_index = -1
            best_length = 0
            for index, document in enumerate(doc_buffer):
                document_length = len(document["tokens"])
                if document_length <= remaining and document_length > best_length:
                    best_index = index
                    best_length = document_length
            if best_index >= 0:
                document = doc_buffer.pop(best_index)
                used = document["tokens"]
                cropped = False
            else:
                shortest_index = min(
                    range(len(doc_buffer)),
                    key=lambda index: len(doc_buffer[index]["tokens"]),
                )
                document = doc_buffer.pop(shortest_index)
                used = document["tokens"][:remaining]
                cropped = True
            start = len(row)
            row.extend(used)
            packing_events.append(
                {
                    "row_index": row_index,
                    "start_position": start,
                    "end_position_exclusive": len(row),
                    "source_doc_ordinal": document["source_doc_ordinal"],
                    "shard_id": document["shard_id"],
                    "row_group": document["row_group"],
                    "row_in_group": document["row_in_group"],
                    "original_token_length_with_bos": len(document["tokens"]),
                    "used_token_length": len(used),
                    "cropped": cropped,
                }
            )
        if len(row) != row_capacity:
            raise AssertionError("packer did not fill row exactly")
        if row[0] != bos_id:
            raise AssertionError(f"packed row {row_index} does not start with BOS")
        full_rows.append(row)

    inputs = [row[:-1] for row in full_rows]
    targets = [row[1:] for row in full_rows]
    for full, inputs_row, targets_row in zip(full_rows, inputs, targets, strict=True):
        if full[:-1] != inputs_row or full[1:] != targets_row:
            raise AssertionError("input/target shift mismatch")
    summary = {
        "packed_rows": packed_rows,
        "context_length": context_length,
        "input_token_positions": packed_rows * context_length,
        "documents_loaded_into_buffer": documents_loaded,
        "documents_consumed_or_cropped": len(packing_events),
        "documents_cropped": sum(event["cropped"] for event in packing_events),
        "documents_remaining_in_buffer": len(doc_buffer),
        "bos_occurrences_in_inputs": sum(
            token_id == bos_id for row in inputs for token_id in row
        ),
        "loaded_document_text_sha256": loaded_digest.hexdigest(),
        "packed_full_rows_int64le_sha256": digest_id_rows(full_rows),
        "packed_inputs_int64le_sha256": digest_id_rows(inputs),
        "packed_targets_int64le_sha256": digest_id_rows(targets),
        "packing_event_count": len(packing_events),
    }
    return inputs, targets, packing_events, summary


def local_ngram_tuple(row: list[int], position: int, order: int) -> tuple[int, ...]:
    current = row[position]
    if order == 2:
        previous = row[position - 1] if position >= 1 else row[0]
        return (previous, current)
    if order == 3:
        previous = row[position - 1] if position >= 1 else row[0]
        previous2 = row[position - 2] if position >= 2 else row[position]
        return (previous2, previous, current)
    raise ValueError(f"unsupported local n-gram order: {order}")


def measure_ngram_impact(
    inputs: list[list[int]],
    projection: list[int],
    aliased_ids: set[int],
    protected_ids: set[int],
    bos_id: int,
    orders: list[int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for order in orders:
        raw_counts: Counter[tuple[int, ...]] = Counter()
        raw_to_canonical: dict[tuple[int, ...], tuple[int, ...]] = {}
        canonical_to_raw: dict[tuple[int, ...], set[tuple[int, ...]]] = defaultdict(set)
        alias_exposed = 0
        protected_exposed = 0
        bos_exposed = 0
        total = 0
        for row in inputs:
            for position in range(len(row)):
                raw_tuple = local_ngram_tuple(row, position, order)
                canonical_tuple = tuple(projection[token_id] for token_id in raw_tuple)
                raw_counts[raw_tuple] += 1
                raw_to_canonical[raw_tuple] = canonical_tuple
                canonical_to_raw[canonical_tuple].add(raw_tuple)
                alias_exposed += any(token_id in aliased_ids for token_id in raw_tuple)
                protected_exposed += any(
                    token_id in protected_ids for token_id in raw_tuple
                )
                bos_exposed += bos_id in raw_tuple
                total += 1

        merged_canonical = {
            key for key, variants in canonical_to_raw.items() if len(variants) >= 2
        }
        strict_affected = sum(
            count
            for raw_tuple, count in raw_counts.items()
            if raw_to_canonical[raw_tuple] in merged_canonical
        )
        merged_examples = []
        for canonical_tuple in sorted(
            merged_canonical,
            key=lambda key: (
                -sum(raw_counts[item] for item in canonical_to_raw[key]),
                key,
            ),
        )[:25]:
            variants = canonical_to_raw[canonical_tuple]
            merged_examples.append(
                {
                    "canonical_tuple": list(canonical_tuple),
                    "raw_variant_count": len(variants),
                    "occurrence_count": sum(raw_counts[item] for item in variants),
                    "raw_variants": [
                        {"tuple": list(item), "count": raw_counts[item]}
                        for item in sorted(
                            variants, key=lambda item: (-raw_counts[item], item)
                        )[:12]
                    ],
                    "raw_variants_truncated": len(variants) > 12,
                }
            )

        raw_distinct = len(raw_counts)
        canonical_distinct = len(canonical_to_raw)
        results[str(order)] = {
            "order": order,
            "total_occurrences": total,
            "raw_distinct_ngram_tuples": raw_distinct,
            "canonical_distinct_ngram_tuples": canonical_distinct,
            "observed_distinct_key_collapse": (
                (raw_distinct - canonical_distinct) / raw_distinct
                if raw_distinct
                else 0.0
            ),
            "merged_canonical_tuple_count": len(merged_canonical),
            "strict_affected_occurrences": strict_affected,
            "strict_affected_occurrence_share": (
                strict_affected / total if total else 0.0
            ),
            "alias_exposed_occurrences": alias_exposed,
            "alias_exposed_occurrence_share": alias_exposed / total if total else 0.0,
            "protected_exposed_occurrences": protected_exposed,
            "protected_exposed_occurrence_share": (
                protected_exposed / total if total else 0.0
            ),
            "bos_exposed_occurrences": bos_exposed,
            "bos_exposed_occurrence_share": bos_exposed / total if total else 0.0,
            "top_merged_examples": merged_examples,
        }
    return results


def report_markdown(result: dict[str, Any]) -> str:
    projection = result["projection"]
    sample = result["sample"]
    impact = result["ngram_impact"]
    gates = result["gates"]
    verdict = result["verdict"]
    rows = []
    for order in sorted(impact, key=int):
        item = impact[order]
        rows.append(
            f"| {order} | {item['total_occurrences']:,} | "
            f"{item['raw_distinct_ngram_tuples']:,} | "
            f"{item['canonical_distinct_ngram_tuples']:,} | "
            f"{item['strict_affected_occurrence_share']:.6f} | "
            f"{item['alias_exposed_occurrence_share']:.6f} |"
        )
    gate_rows = []
    for gate_name, gate in gates.items():
        gate_rows.append(
            f"| `{gate_name}` | {gate['observed']:.6f} | "
            f"{gate['operator']} {gate['threshold']:.6f} | "
            f"{'PASS' if gate['pass'] else 'FAIL'} |"
        )
    caveats = "\n".join(f"- {item}" for item in result["scope_limits"])
    return f"""# Paper 019 round 2: canonical-key mediator assay

Verdict: **{verdict['status']}**.

This is an offline addressing-mediator result, not a BPB result and not run
authority.  The source-faithful portion is the pinned demo's normalization
stage order.  The 8,192-token RustBPE tokenizer, one-to-one protection policy,
packed sample, and local hash-site shifts are OPHIS-specific.

## Frozen inputs

- Engram demo: commit `{result['source']['repository_commit']}`, SHA-256
  `{result['source']['demo_sha256']}`.
- Tokenizer pickle: `{result['tokenizer']['pickle_sha256']}`;
  semantic ID-to-byte digest
  `{result['tokenizer']['semantic_id_to_raw_bytes_sha256']}`.
- Dataset shard(s): {", ".join(f"`{item['shard_id']}` `{item['sha256']}`" for item in result['dataset']['shards'])}.
- Packed input: {sample['input_token_positions']:,} positions,
  `{sample['packed_inputs_int64le_sha256']}`.
- Loaded source-document text digest:
  `{sample['loaded_document_text_sha256']}`.

## Vocabulary projection

- Original/canonical cardinality:
  {projection['original_vocab_size']:,}/{projection['canonical_vocab_size']:,}.
- Collapsed keys: {projection['collapsed_key_count']:,}
  ({projection['vocabulary_key_cardinality_collapse']:.6f}).
- Protected IDs: {projection['protected_id_count']:,}; all remain one-to-one.
- Alias sets: {projection['alias_set_count']:,}; explained
  {projection['alias_set_explained_count']:,};
  unexplained {projection['alias_set_unexplained_count']:,}
  (explained fraction {projection['alias_set_explained_fraction']:.6f}).

Full token mappings, alias members and stage traces, unexplained cases, and
packing events are in the sibling JSONL artifacts.

## Observed packed n-gram impact

The strict affected metric counts an occurrence only when its canonical tuple
is shared by at least two *distinct raw tuples observed in this sample*.  It is
therefore stricter than merely containing a vocabulary ID that belongs to an
alias set.

| order | occurrences | raw distinct | canonical distinct | strict affected share | alias-exposed share |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The occurrence gate uses the minimum strict share across orders 2 and 3:
`{verdict['minimum_strict_affected_share']:.6f}`.

## Preregistered gates

| gate | observed | requirement | result |
|---|---:|---:|:---:|
{chr(10).join(gate_rows)}

## Interpretation limits

{caveats}

The only supported conclusion is `{verdict['conclusion']}`.
"""


def run(spec_path: Path) -> dict[str, Any]:
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    if set(spec["declared_outputs"]) != OUTPUT_NAMES:
        raise RuntimeError("declared output set does not match script output allowlist")

    local_source = verify_local_source(spec)

    source_cfg = spec["source"]
    archive_path = resolve_project_path(source_cfg["archive_path"])
    source_manifest_path = resolve_project_path(source_cfg["manifest_path"])
    archive_hash = require_hash(
        archive_path, source_cfg["archive_sha256"], "Engram source archive"
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest["repository"]["commit"] != source_cfg["repository_commit"]:
        raise RuntimeError("source manifest commit does not match assay spec")
    demo = extract_demo(archive_path, source_cfg["demo_member"])
    demo_hash = sha256_bytes(demo)
    if demo_hash != source_cfg["demo_sha256"]:
        raise RuntimeError(
            f"Engram demo hash mismatch: {demo_hash} != {source_cfg['demo_sha256']}"
        )
    verified_demo_fragments = verify_source_demo(demo)

    normalizer = SourceNormalizer()
    # Exact behavioral tripwires for the source's whitespace sentinel and stage
    # sequence.  These are independent of the vocabulary results.
    normalizer_checks = {
        "single_space_preserved": normalizer.trace(" ")[0] == " ",
        "whitespace_collapsed": normalizer.trace("\t \n")[0] == " ",
        "leading_trailing_stripped": normalizer.trace("  Alpha  ")[0] == "alpha",
        "compatibility_case_accent": normalizer.trace("  ＣAFÉ\t")[0] == "cafe",
    }
    if not all(normalizer_checks.values()):
        raise RuntimeError(f"normalizer behavioral check failed: {normalizer_checks}")

    encoding, tokenizer_meta, vocab_rows, special_map = load_tokenizer(spec)
    projection, projection_rows, alias_rows, unexplained_rows, projection_summary = (
        build_projection(
            vocab_rows,
            special_map,
            tokenizer_meta["bos_id"],
            normalizer,
        )
    )

    sample_cfg = spec["sample"]
    split_path = resolve_project_path(spec["local_source"]["data_split_path"])
    split = load_data_split(split_path)
    canonical_split = "test" if sample_cfg["split"] in {"val", "test"} else "train"
    shard_ids = split[canonical_split]
    if shard_ids != sample_cfg["expected_shard_ids"]:
        raise RuntimeError(
            f"sample shard IDs changed: expected {sample_cfg['expected_shard_ids']}, "
            f"observed {shard_ids}"
        )
    if sample_cfg["shuffle_data"] or sample_cfg["epoch"] != 1:
        raise RuntimeError("v1 assay supports only sequential epoch-1 sampling")
    data_directory = resolve_project_path(sample_cfg["data_directory"])
    shard_paths: list[tuple[int, Path]] = []
    shard_records: list[dict[str, Any]] = []
    for shard_id in shard_ids:
        path = data_directory / f"shard_{shard_id:05d}.parquet"
        if not path.exists():
            raise RuntimeError(f"frozen sample shard is missing: {path}")
        observed_hash = require_hash(
            path,
            sample_cfg["expected_shard_sha256"][str(shard_id)],
            f"dataset shard {shard_id}",
        )
        parquet = pq.ParquetFile(path)
        shard_paths.append((shard_id, path))
        shard_records.append(
            {
                "shard_id": shard_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": observed_hash,
                "row_groups": parquet.num_row_groups,
                "rows": parquet.metadata.num_rows,
            }
        )

    inputs, targets, packing_events, sample_summary = pack_frozen_sample(
        encoding,
        tokenizer_meta["bos_id"],
        shard_paths,
        sample_cfg,
    )
    del targets

    aliased_ids = set(projection_summary.pop("aliased_ids"))
    protected_ids = set(projection_summary["protected_ids"])
    ngram_impact = measure_ngram_impact(
        inputs,
        projection,
        aliased_ids,
        protected_ids,
        tokenizer_meta["bos_id"],
        [int(order) for order in sample_cfg["ngram_orders"]],
    )

    minimum_strict_share = min(
        value["strict_affected_occurrence_share"] for value in ngram_impact.values()
    )
    gate_cfg = spec["kill_gates"]
    gates = {
        "vocabulary_key_cardinality_collapse": {
            "observed": projection_summary[
                "vocabulary_key_cardinality_collapse"
            ],
            "operator": ">=",
            "threshold": gate_cfg["vocabulary_key_cardinality_collapse_min"],
        },
        "observed_ngram_occurrence_affected": {
            "observed": minimum_strict_share,
            "operator": ">=",
            "threshold": gate_cfg["observed_ngram_occurrence_affected_min"],
        },
        "alias_set_explained_fraction": {
            "observed": projection_summary["alias_set_explained_fraction"],
            "operator": ">=",
            "threshold": gate_cfg["alias_set_explained_fraction_min"],
        },
    }
    for gate in gates.values():
        gate["pass"] = gate["observed"] >= gate["threshold"]
    all_gates_pass = all(gate["pass"] for gate in gates.values())
    status = (
        "MEDIATOR_GATES_PASS__NOT_RUN_AUTHORITY"
        if all_gates_pass
        else "KILL_BEFORE_GPU_SCORING"
    )
    conclusion = (
        "the frozen local key-collapse mediator clears the paper-019 offline "
        "thresholds, subject to all listed scope limits"
        if all_gates_pass
        else "the frozen local key-collapse mediator fails at least one "
        "paper-019 offline threshold and the arm should be killed before GPU scoring"
    )

    projection_bytes = jsonl_bytes(projection_rows)
    alias_bytes = jsonl_bytes(alias_rows)
    unexplained_bytes = jsonl_bytes(unexplained_rows)
    packing_bytes = jsonl_bytes(packing_events)
    intermediate_hashes = {
        "projection.jsonl": sha256_bytes(projection_bytes),
        "alias_sets.jsonl": sha256_bytes(alias_bytes),
        "unexplained_alias_sets.jsonl": sha256_bytes(unexplained_bytes),
        "packing_events.jsonl": sha256_bytes(packing_bytes),
    }

    result = {
        "schema_version": 1,
        "assay_id": spec["assay_id"],
        "assay_spec_sha256": sha256_bytes(spec_bytes),
        "assay_script_sha256": sha256_file(Path(__file__).resolve()),
        "mode": "offline_read_only_inputs_no_cuda_no_training",
        "source": {
            "paper_id": source_cfg["paper_id"],
            "repository_commit": source_cfg["repository_commit"],
            "archive_path": str(archive_path),
            "archive_sha256": archive_hash,
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "demo_member": source_cfg["demo_member"],
            "demo_sha256": demo_hash,
            "verified_demo_fragments": verified_demo_fragments,
            "source_faithful_component": (
                "normalization stage order and normalized-if-nonempty fallback"
            ),
            "local_safety_extension": spec["projection"]["local_safety_extension"],
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "pyarrow": importlib.metadata.version("pyarrow"),
            "tiktoken": importlib.metadata.version("tiktoken"),
            "tokenizers": tokenizers.__version__,
            "normalizer_behavioral_checks": normalizer_checks,
        },
        "local_source": local_source,
        "tokenizer": tokenizer_meta,
        "dataset": {
            "canonical_split": canonical_split,
            "data_split_sha256": sha256_file(split_path),
            "shards": shard_records,
        },
        "sample": sample_summary,
        "projection": projection_summary,
        "ngram_impact": ngram_impact,
        "gate_definitions": gate_cfg,
        "gates": gates,
        "verdict": {
            "status": status,
            "all_offline_mediator_gates_pass": all_gates_pass,
            "minimum_strict_affected_share": minimum_strict_share,
            "conclusion": conclusion,
            "run_authority": False,
            "bpb_claim": False,
            "sota_claim": False,
        },
        "intermediate_artifact_sha256": intermediate_hashes,
        "scope_limits": [
            "The upstream demo uses the DeepSeek-V3 tokenizer; this assay uses the current 8,192-token OPHIS RustBPE tokenizer.",
            "The upstream repository does not pin the tokenizers package version; this assay records the exact local version.",
            "The local tokenizer cache lacks tokenizer_train_split.json, so the current pickle is identified by bytes and semantic vocabulary digest, not by complete tokenizer-training provenance.",
            "One-to-one protection for special, BOS, every one-byte token, replacement-character decoding, and empty decoding is a conservative OPHIS safety extension.",
            "The strict occurrence metric is computed before the existing modulo hash tables; it measures canonical tuple mergers, not random hash collisions or learned value quality.",
            "The packed sample mirrors current best-fit packing and current row-prefix shifts; it is one frozen validation sample, not a population estimate.",
            "No model input ID, target, byte count, document boundary, token order, parameter, optimizer state, or RNG state is changed or tested here.",
            "Passing these mediator gates neither predicts nor establishes BPB improvement and does not authorize a GPU run.",
        ],
    }
    result_bytes = canonical_json_bytes(result, pretty=True)
    report_bytes = report_markdown(result).encode("utf-8")

    output_payloads = {
        "projection.jsonl": projection_bytes,
        "alias_sets.jsonl": alias_bytes,
        "unexplained_alias_sets.jsonl": unexplained_bytes,
        "packing_events.jsonl": packing_bytes,
        "result.json": result_bytes,
        "REPORT.md": report_bytes,
    }
    for name, payload in output_payloads.items():
        atomic_write(ARTIFACT_DIR / name, payload)

    manifest = {
        "schema_version": 1,
        "assay_id": spec["assay_id"],
        "inputs": {
            "assay_spec.json": sha256_bytes(spec_bytes),
            "engram_canonical_key_assay.py": sha256_file(Path(__file__).resolve()),
            "source_archive": archive_hash,
            "source_demo": demo_hash,
            "tokenizer_pickle": tokenizer_meta["pickle_sha256"],
            "tokenizer_semantic_vocabulary": tokenizer_meta[
                "semantic_id_to_raw_bytes_sha256"
            ],
            "data_split": sha256_file(split_path),
            "dataset_shards": {
                str(item["shard_id"]): item["sha256"] for item in shard_records
            },
        },
        "outputs": {
            name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in sorted(output_payloads.items())
        },
        "verdict": status,
        "run_authority": False,
    }
    manifest_bytes = canonical_json_bytes(manifest, pretty=True)
    atomic_write(ARTIFACT_DIR / "output_manifest.json", manifest_bytes)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ARTIFACT_DIR / "assay_spec.json",
        help="frozen assay specification",
    )
    args = parser.parse_args()
    try:
        result = run(args.spec.resolve())
    except Exception as exc:
        print(f"ASSAY_BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["verdict"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
