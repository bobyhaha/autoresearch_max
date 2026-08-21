# Copyright 2026 Recursive
# Copyright 2025 Andrej Karpathy
# SPDX-License-Identifier: Apache-2.0
"""Runtime utilities: tokenizer wrapper, dataloader, and BPB evaluation."""

import os
import json
import math
import pickle
import random

import pyarrow.parquet as pq
import torch

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 524288

# H200 adapter: Recursive's upstream baseline expects /data, while our prepared
# Karpathy shards/tokenizer live under ~/.cache/autoresearch when /data is absent.
# `os.path.isdir("/data")` alone is not sufficient: on a shared multi-tenant host
# /data can exist as a leftover mount/symlink farm from a DIFFERENT account (owned
# by another user, pointing at that user's now-unreachable home directory), which
# is readable-as-a-directory but contains no real shard/tokenizer files. Require
# an actual resolvable tokenizer file, not just the directory's existence.
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
_USE_UPSTREAM_DATA_DIR = os.path.isfile(os.path.join("/data", "tokenizer", "tokenizer.pkl"))
DATA_DIR = "/data" if _USE_UPSTREAM_DATA_DIR else os.path.join(_CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer") if _USE_UPSTREAM_DATA_DIR else os.path.join(_CACHE_DIR, "tokenizer")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_SPLIT_PATH = os.path.join(PROJECT_ROOT, "data_split.json")
BOS_TOKEN = "<|reserved_0|>"


def _shard_filename(index):
    return f"shard_{index:05d}.parquet"


def _normalize_shard_ids(values, split_name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"data_split.json must define non-empty list '{split_name}'")
    seen = set()
    ids = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid shard id in '{split_name}': {value!r}")
        if value in seen:
            raise ValueError(f"Duplicate shard id in '{split_name}': {value}")
        seen.add(value)
        ids.append(value)
    return ids


def load_data_split():
    with open(DATA_SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)
    train_ids = _normalize_shard_ids(split.get("train"), "train")
    test_ids = _normalize_shard_ids(split.get("test"), "test")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"Shard ids cannot appear in both train and test: {overlap}")
    return {"train": train_ids, "test": test_ids}


def _canonical_split(split):
    if split == "train":
        return "train"
    if split in ("val", "test"):
        return "test"
    raise ValueError(f"Unknown split: {split!r}")


def split_parquet_files(split):
    split_ids = load_data_split()[_canonical_split(split)]
    return [os.path.join(DATA_DIR, _shard_filename(index)) for index in split_ids]


def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    parquet_paths = split_parquet_files(split)
    missing = [p for p in parquet_paths if not os.path.exists(p)]
    assert not missing, f"Missing {split} shards. Run prepare.py first: {missing}"
    # Data-ORDER lever (fair: same documents, reordered — content-preserving).
    # SHUFFLE_DATA=1 decorrelates the otherwise strictly-sequential stream by
    # permuting row-group visitation order per epoch and shuffling docs within a
    # row-group (train split only, seeded, deterministic). SHUFFLE_DATA unset/0 is
    # byte-identical to the original sequential loader.
    shuffle = os.environ.get("SHUFFLE_DATA", "0") == "1" and split == "train"
    seed = int(os.environ.get("SEED", "42"))
    rg_units = []
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg_units.append((filepath, rg_idx))
    epoch = 1
    while True:
        order = list(rg_units)
        if shuffle:
            random.Random(seed * 100003 + epoch).shuffle(order)
        for filepath, rg_idx in order:
            pf = pq.ParquetFile(filepath)
            rg = pf.read_row_group(rg_idx)
            batch = rg.column('text').to_pylist()
            if shuffle:
                random.Random(seed * 7919 + epoch * 131 + rg_idx).shuffle(batch)
            for i in range(0, len(batch), tokenizer_batch_size):
                yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def _make_legacy_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """BOS-aligned dataloader with best-fit packing. Every row starts with BOS;
    documents are packed best-fit, cropping the shortest doc when nothing fits."""
    assert split in ["train", "val", "test"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


class _BoundarySlot:
    """One fixed-capacity packer-boundary staging slot.

    The CPU buffer may be overwritten once its previous H2D copy completes.
    The CUDA buffer is not overwritten until the stream that consumed the
    previous lease records ``consumer_done``. Those are distinct lifetimes, so
    the slot deliberately carries two events rather than relying on
    ``record_stream`` allocator heuristics.
    """

    def __init__(self, capacity, device):
        self.cpu = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self.cuda = torch.empty(capacity, dtype=torch.int32, device=device)
        self.h2d_done = torch.cuda.Event(enable_timing=False)
        self.consumer_done = torch.cuda.Event(enable_timing=False)
        self.h2d_recorded = False
        self.consumer_recorded = False
        self.leased = False
        self.generation = 0


class BoundaryLease:
    """A single-use view of packer-sourced FA3 ``cu_seqlens``.

    Call :meth:`wait_for_current_stream` before reading ``cu_seqlens`` and call
    :meth:`mark_consumed` on the consuming stream after its last use. Training
    does the latter immediately after ``loss.backward()``. Reusing a ring slot
    with an outstanding lease and releasing a lease twice both fail closed.
    """

    __slots__ = (
        "_slot",
        "_generation",
        "_length",
        "_max_seqlen",
        "_waited",
        "_consumer_stream_id",
        "_released",
    )

    def __init__(self, slot, generation, length, max_seqlen):
        self._slot = slot
        self._generation = generation
        self._length = length
        self._max_seqlen = max_seqlen
        self._waited = False
        self._consumer_stream_id = None
        self._released = False

    @property
    def length(self):
        return self._length

    @property
    def max_seqlen(self):
        return self._max_seqlen

    def _assert_live(self):
        if self._released:
            raise RuntimeError("PACKER_BOUNDARY_LEASE_DOUBLE_RELEASE")
        if not self._slot.leased or self._slot.generation != self._generation:
            raise RuntimeError("PACKER_BOUNDARY_LEASE_STALE")

    def wait_for_current_stream(self):
        """Order future work on the current stream after the sidecar H2D."""
        self._assert_live()
        stream = torch.cuda.current_stream(device=self._slot.cuda.device)
        stream_id = stream.cuda_stream
        if self._waited and stream_id != self._consumer_stream_id:
            raise RuntimeError("PACKER_BOUNDARY_LEASE_MULTIPLE_CONSUMER_STREAMS")
        stream.wait_event(self._slot.h2d_done)
        self._waited = True
        self._consumer_stream_id = stream_id
        return self._slot.cuda[:self._length]

    @property
    def cu_seqlens(self):
        """Return the logical boundary prefix after inserting its stream wait."""
        return self.wait_for_current_stream()

    def mark_consumed(self):
        """Release after recording all work on the stream that waited for H2D."""
        self._assert_live()
        if not self._waited:
            raise RuntimeError("PACKER_BOUNDARY_LEASE_RELEASE_BEFORE_WAIT")
        stream = torch.cuda.current_stream(device=self._slot.cuda.device)
        if stream.cuda_stream != self._consumer_stream_id:
            raise RuntimeError("PACKER_BOUNDARY_LEASE_RELEASE_STREAM_MISMATCH")
        self._slot.consumer_done.record(stream)
        self._slot.consumer_recorded = True
        self._slot.leased = False
        self._released = True

    def discard_unconsumed(self):
        """Release a deliberately prefetched lease that will never be consumed."""
        self._assert_live()
        if self._waited:
            raise RuntimeError("PACKER_BOUNDARY_LEASE_DISCARD_AFTER_WAIT")
        stream = torch.cuda.current_stream(device=self._slot.cuda.device)
        stream.wait_event(self._slot.h2d_done)
        self._slot.consumer_done.record(stream)
        self._slot.consumer_recorded = True
        self._slot.leased = False
        self._released = True


class _BoundaryRing:
    """Three-slot pinned-CPU/CUDA boundary transfer ring."""

    def __init__(self, capacity, device="cuda", slots=3):
        if slots != 3:
            raise ValueError("PACKER_BOUNDARY_RING_SLOTS must be exactly 3")
        self._slots = [_BoundarySlot(capacity, device) for _ in range(slots)]
        self._copy_stream = torch.cuda.Stream(device=device)
        self._next_slot = 0

    def stage(self, boundaries, max_seqlen):
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slots)
        if slot.leased:
            raise RuntimeError("PACKER_BOUNDARY_RING_REUSE_WITH_OUTSTANDING_LEASE")

        count = len(boundaries)
        if count > slot.cpu.numel():
            raise RuntimeError(
                "PACKER_BOUNDARY_CAPACITY_EXCEEDED "
                f"count={count} capacity={slot.cpu.numel()}"
            )

        # CPU storage is safe to overwrite once the previous H2D has completed;
        # CUDA storage has the longer consumer lifetime handled by wait_event.
        if slot.h2d_recorded:
            slot.h2d_done.synchronize()
        slot.cpu[:count].copy_(torch.tensor(boundaries, dtype=torch.int32))
        with torch.cuda.stream(self._copy_stream):
            if slot.consumer_recorded:
                self._copy_stream.wait_event(slot.consumer_done)
            slot.cuda[:count].copy_(slot.cpu[:count], non_blocking=True)
            slot.h2d_done.record(self._copy_stream)
        slot.h2d_recorded = True
        slot.leased = True
        slot.generation += 1
        return BoundaryLease(slot, slot.generation, count, max_seqlen)


def _validate_packer_boundaries(boundaries, B, T):
    """Validate cheap host-side invariants before staging a packed batch."""
    terminal = B * T
    if (
        len(boundaries) < B + 1
        or boundaries[0] != 0
        or boundaries[-1] != terminal
        or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
        or any(value < 0 or value > terminal for value in boundaries)
        or any(row * T not in boundaries[:-1] for row in range(B))
        or any(right - left > T for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise RuntimeError(
            "PACKER_DOC_BOUNDARY_INVARIANT_FAILED "
            f"count={len(boundaries) - 1} first={boundaries[0] if boundaries else 'none'} "
            f"last={boundaries[-1] if boundaries else 'none'} expected_last={terminal}"
        )


def _make_boundary_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """Best-fit loader with an asynchronous packer-sourced boundary sidecar."""
    assert split in ["train", "val", "test"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)
    boundary_ring = _BoundaryRing(B * T + 1, device="cuda", slots=3)

    while True:
        boundaries = []
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                # ``inputs`` has length T although the packer row has length T+1.
                # A document beginning at pos==T contributes only a target token,
                # so it is intentionally absent from the input-side FA3 vector.
                if pos < T:
                    boundaries.append(row_idx * T + pos)

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        boundaries.append(B * T)
        _validate_packer_boundaries(boundaries, B, T)
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        boundary_lease = boundary_ring.stage(boundaries, T)
        yield inputs, targets, epoch, boundary_lease


def make_dataloader(
    tokenizer,
    B,
    T,
    split,
    buffer_size=1000,
    *,
    return_doc_boundaries=False,
):
    """Construct the legacy loader or its explicitly opted-in sidecar variant.

    With ``return_doc_boundaries=False`` this returns the isolated legacy
    generator: no boundary list, pinned sidecar buffer, CUDA sidecar buffer,
    stream, event, or lease is allocated, and each iteration yields the original
    ``(inputs, targets, epoch)`` triple. The opt-in generator yields a fourth
    :class:`BoundaryLease` value.
    """
    if not isinstance(return_doc_boundaries, bool):
        raise ValueError("return_doc_boundaries must be a bool")
    if return_doc_boundaries:
        return _make_boundary_dataloader(tokenizer, B, T, split, buffer_size)
    return _make_legacy_dataloader(tokenizer, B, T, split, buffer_size)


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, eval_tokens=None, forward_kwargs_fn=None):
    """Bits per byte: vocab-size-independent metric. Sums per-token
    cross-entropy (nats) and target byte lengths, converts nats/byte to
    bits/byte; special tokens (byte length 0) are excluded.

    If *eval_tokens* is provided, uses that many tokens instead of the
    default EVAL_TOKENS (for lightweight mid-training probes).

    *forward_kwargs_fn*, if given, maps the input batch to extra keyword
    arguments for the model call. It exists for per-batch structures that must
    be built OUTSIDE the compiled region (the DOC_MASK_IMPL=flex block masks);
    when None the call is exactly as before.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    _eval_tokens = eval_tokens if eval_tokens is not None else EVAL_TOKENS
    steps = _eval_tokens // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        extra = forward_kwargs_fn(x) if forward_kwargs_fn is not None else {}
        loss_flat = model(x, y, reduction='none', **extra).view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


@torch.no_grad()
def evaluate_bpb_fast(model, tokenizer, batch_size, eval_tokens=2 * 524288, forward_kwargs_fn=None):
    """Lightweight BPB probe for mid-training time-matched comparison.

    Uses a small subset of validation tokens (~1M by default) to get a
    fast BPB estimate.  Intended for Track B (bare training) where the
    full 40M-token evaluation would be too expensive to run every 100 steps.
    """
    return evaluate_bpb(
        model,
        tokenizer,
        batch_size,
        eval_tokens=eval_tokens,
        forward_kwargs_fn=forward_kwargs_fn,
    )
