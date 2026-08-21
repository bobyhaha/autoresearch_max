#!/usr/bin/env python3
"""Adversarial H200 validation for the FA3 packer-boundary sidecar.

This is a systems test, not an experiment: it does not load campaign data,
train a model, evaluate BPB, or write a RunRecord. It exercises CUDA event
lifetimes, rotating storage, dynamic logical lengths, the exact FA3 varlen
forward/backward API, and PyTorch's fail-on-recompile stance before the
one-shot governed diagnostic is eligible to launch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib


DELAY_CYCLES = 5_000_000


def _expect_runtime_error(label, pattern, operation):
    try:
        operation()
    except RuntimeError as exc:
        if pattern not in str(exc):
            raise RuntimeError(
                f"{label}: wrong RuntimeError: {exc}"
            ) from exc
        return
    raise RuntimeError(f"{label}: expected RuntimeError containing {pattern!r}")


def _event_lifetime_tests():
    stream = torch.cuda.current_stream()

    delayed_h2d_ring = lib._BoundaryRing(16, slots=3)
    with torch.cuda.stream(delayed_h2d_ring._copy_stream):
        torch.cuda._sleep(DELAY_CYCLES)
    lease = delayed_h2d_ring.stage([0, 3, 8], max_seqlen=8)
    observed = lease.cu_seqlens.clone()
    lease.mark_consumed()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        observed.cpu(),
        torch.tensor([0, 3, 8], dtype=torch.int32),
        rtol=0,
        atol=0,
    )

    wrap_ring = lib._BoundaryRing(16, slots=3)
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index in range(8):
        expected = torch.tensor(
            [0, 1 + index % 3, 4 + index % 4, 8],
            dtype=torch.int32,
        )
        lease = wrap_ring.stage(expected.tolist(), max_seqlen=8)
        view = lease.cu_seqlens
        captured.append((view.clone(), expected))
        torch.cuda._sleep(DELAY_CYCLES)
        lease.mark_consumed()
    torch.cuda.synchronize()
    for observed, expected in captured:
        torch.testing.assert_close(
            observed.cpu(), expected, rtol=0, atol=0
        )

    outstanding_ring = lib._BoundaryRing(8, slots=3)
    outstanding = [
        outstanding_ring.stage([0, 4], max_seqlen=4)
        for _ in range(3)
    ]
    _expect_runtime_error(
        "fourth outstanding lease",
        "PACKER_BOUNDARY_RING_REUSE_WITH_OUTSTANDING_LEASE",
        lambda: outstanding_ring.stage([0, 4], max_seqlen=4),
    )
    for lease in outstanding:
        lease.discard_unconsumed()

    stream_ring = lib._BoundaryRing(8, slots=3)
    lease = stream_ring.stage([0, 2, 4], max_seqlen=4)
    consumer_a = torch.cuda.Stream()
    consumer_b = torch.cuda.Stream()
    with torch.cuda.stream(consumer_a):
        lease.wait_for_current_stream()
    with torch.cuda.stream(consumer_b):
        _expect_runtime_error(
            "wrong consumer stream",
            "PACKER_BOUNDARY_LEASE_RELEASE_STREAM_MISMATCH",
            lease.mark_consumed,
        )
    with torch.cuda.stream(consumer_a):
        lease.mark_consumed()

    discard_ring = lib._BoundaryRing(8, slots=3)
    unused = discard_ring.stage([0, 4], max_seqlen=4)
    unused.discard_unconsumed()
    waited = discard_ring.stage([0, 1, 4], max_seqlen=4)
    waited.wait_for_current_stream()
    _expect_runtime_error(
        "discard after wait",
        "PACKER_BOUNDARY_LEASE_DISCARD_AFTER_WAIT",
        waited.discard_unconsumed,
    )
    waited.mark_consumed()

    tail_ring = lib._BoundaryRing(12, slots=3)
    long_lease = tail_ring.stage([0, 1, 2, 3, 4, 8], max_seqlen=8)
    long_lease.wait_for_current_stream()
    long_lease.mark_consumed()
    for _ in range(2):
        filler = tail_ring.stage([0, 8], max_seqlen=8)
        filler.discard_unconsumed()
    short_lease = tail_ring.stage([0, 3, 8], max_seqlen=8)
    short_view = short_lease.cu_seqlens
    if short_view.numel() != 3:
        raise RuntimeError("logical boundary view exposed a poisoned tail")
    torch.testing.assert_close(
        short_view.cpu(),
        torch.tensor([0, 3, 8], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    short_lease.mark_consumed()

    capacity_ring = lib._BoundaryRing(4, slots=3)
    exact = capacity_ring.stage([0, 1, 2, 4], max_seqlen=4)
    exact.discard_unconsumed()
    _expect_runtime_error(
        "boundary capacity",
        "PACKER_BOUNDARY_CAPACITY_EXCEEDED",
        lambda: capacity_ring.stage([0, 1, 2, 3, 4], max_seqlen=4),
    )
    stream.synchronize()


def _compiler_guard_tests():
    from torch._dynamo.callback import callback_handler

    inside = False
    events: list[tuple[str, str]] = []

    def callback(args):
        if inside:
            events.append(
                (args.callback_trigger.name, str(args.compile_id))
            )

    registered = callback_handler.register_start_callback(callback)

    @torch.compile(dynamic=False, fullgraph=True)
    def dynamic_prefix(cu, payload):
        return payload + cu[-1].to(payload.dtype)

    payload = torch.ones(16, device="cuda")
    for index, values in enumerate(
        ([0, 4, 8], [0, 2, 5, 8], [0, 1, 3, 6, 8])
    ):
        cu = torch.tensor(values, dtype=torch.int32, device="cuda")
        torch._dynamo.mark_dynamic(cu, 0)
        inside = True
        stance = (
            nullcontext()
            if index == 0
            else torch.compiler.set_stance("fail_on_recompile")
        )
        with stance:
            result = dynamic_prefix(cu, payload)
        inside = False
        torch.testing.assert_close(
            result, payload + 8, rtol=0, atol=0
        )

    dynamo_events = [event for event in events if event[0] == "DYNAMO"]
    if len(dynamo_events) != 1:
        raise RuntimeError(
            "dynamic compile positive control expected exactly one Dynamo "
            f"compile, observed={dynamo_events}"
        )

    @torch.compile(dynamic=False, fullgraph=True)
    def forced_static_recompile(value):
        return value.sin()

    forced_static_recompile(torch.ones(8, device="cuda"))

    def trigger_forced_recompile():
        with torch.compiler.set_stance("fail_on_recompile"):
            forced_static_recompile(torch.ones(9, device="cuda"))

    _expect_runtime_error(
        "fail_on_recompile negative control",
        "Detected recompile",
        trigger_forced_recompile,
    )
    callback_handler.remove_start_callback(registered)
    torch.cuda.synchronize()
    compile_ids = ",".join(event[1] for event in dynamo_events)
    return {
        "dynamo_graph_count": len(dynamo_events),
        "compile_ids_sha256": hashlib.sha256(
            compile_ids.encode("utf-8")
        ).hexdigest(),
        "forced_recompile_rejected": True,
    }


def _fa3_forward_backward_tests():
    from kernels import get_kernel

    capability = torch.cuda.get_device_capability()
    repository = (
        "varunneal/flash-attention-3"
        if capability == (9, 0)
        else "kernels-community/flash-attn3"
    )
    module = get_kernel(repository).flash_attn_interface
    forward = module._flash_attn_forward
    backward = module._flash_attn_backward

    torch.manual_seed(19019063)
    torch.cuda.manual_seed_all(19019063)
    total, heads, head_dim, max_seqlen = 32, 2, 64, 16
    q = torch.randn(
        total, heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q)
    boundary_sets = (
        [0, 4, 12, 16, 25, 32],
        [0, 2, 8, 15, 16, 20, 28, 32],
        [0, 7, 16, 17, 19, 24, 32],
    )
    ring = lib._BoundaryRing(total + 1, slots=3)
    digest = hashlib.sha256()

    for index in range(9):
        boundaries = boundary_sets[index % len(boundary_sets)]
        scanner = torch.tensor(
            boundaries, dtype=torch.int32, device="cuda"
        )
        lease = ring.stage(boundaries, max_seqlen=max_seqlen)
        sidecar = lease.cu_seqlens

        def run(cu):
            out, lse, *_ = forward(
                q,
                k,
                v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size_left=max_seqlen,
                window_size_right=0,
            )
            dq, dk, dv, *_ = backward(
                dout,
                q,
                k,
                v,
                out,
                lse,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                is_causal=True,
                window_size_left=max_seqlen,
                window_size_right=0,
            )
            return out, lse, dq, dk, dv

        scanner_result = run(scanner)
        sidecar_result = run(sidecar)
        for left, right in zip(
            scanner_result, sidecar_result, strict=True
        ):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            digest.update(
                right.detach()
                .contiguous()
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
        lease.mark_consumed()
    torch.cuda.synchronize()
    return {
        "repository": repository,
        "ring_wrap_iterations": 9,
        "fa3_forward_backward_exact": True,
        "result_sha256": digest.hexdigest(),
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("H200 validation requires CUDA")
    name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    if "H200" not in name or capability != (9, 0):
        raise SystemExit(
            f"H200 validation requires NVIDIA H200 SM90, got {name} {capability}"
        )

    _event_lifetime_tests()
    compiler = _compiler_guard_tests()
    fa3 = _fa3_forward_backward_tests()
    result = {
        "schema_version": 1,
        "status": "pass",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": name,
        "device_capability": list(capability),
        "event_lifetime_tests": "pass",
        "compiler": compiler,
        "fa3": fa3,
    }
    print(
        "PACKER_H200_ADVERSARIAL_VALIDATION_V1="
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


if __name__ == "__main__":
    with nullcontext():
        main()
