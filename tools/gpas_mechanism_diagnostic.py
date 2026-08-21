#!/usr/bin/env python3
"""Diagnostic-only GPAS attribution helpers.

This module deliberately does not launch training or score validation BPB.  A
governed harness can import it around an *eager* model replay to:

* fail closed unless every PACKER_* intervention is zero;
* hash same-device step-0 logits/loss/data/boundaries/RNG and all pre-existing
  parameters/gradients for an exact control/treatment comparison;
* record FP64 post-MLP residual variance for 32/256-step mechanism assays;
* report the shared per-layer gate, gradient, and forward scale;
* run an endpoint counterfactual with all GPAS gates temporarily set to zero;
* attest Dynamo graph count, compile IDs, and recompiles in a separate clean
  compiled replay.

The helpers are inert unless called.  Torch is imported lazily so the
governance/test environment can inspect and compare JSON attestations without
a training installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping


PACKER_ZERO_CONTRACT = {
    "PACKER_DOC_BOUNDARIES": "0",
    "PACKER_BOUNDARY_VERIFY_BATCHES": "0",
    "PACKER_DIAGNOSTIC_HASHES": "0",
    "PACKER_SIDECAR_ACTIVATE_STEP": "0",
}
PARITY_EXACT_FIELDS = (
    "device_uuid",
    "data_sha256",
    "targets_sha256",
    "token_bytes_sha256",
    "boundaries_sha256",
    "cpu_rng_sha256",
    "cuda_rng_sha256",
    "logits_sha256",
    "loss_sha256",
    "preexisting_parameters_sha256",
    "preexisting_gradients_sha256",
)


def require_packer_flags_zero(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return resolved PACKER flags or fail closed on any nonzero value."""

    source = os.environ if environ is None else environ
    resolved = {key: source.get(key, "0") for key in PACKER_ZERO_CONTRACT}
    bad = {
        key: value
        for key, value in resolved.items()
        if value != PACKER_ZERO_CONTRACT[key]
    }
    if bad:
        raise RuntimeError(f"GPAS_DIAGNOSTIC_PACKER_FLAGS_NONZERO {bad}")
    return resolved


def _torch():
    import torch

    return torch


def tensor_sha256(tensor: Any) -> str:
    """Hash tensor metadata and every storage byte, never a sampled checksum."""

    torch = _torch()
    value = tensor.detach().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(struct.pack("<I", value.ndim))
    for dim in value.shape:
        hasher.update(struct.pack("<Q", dim))
    raw = value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    hasher.update(struct.pack("<Q", len(raw)))
    hasher.update(raw)
    return hasher.hexdigest()


def named_tensors_sha256(items: Iterable[tuple[str, Any]]) -> str:
    """Canonical hash of a name-sorted tensor collection."""

    hasher = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        encoded = name.encode("utf-8")
        hasher.update(struct.pack("<Q", len(encoded)))
        hasher.update(encoded)
        hasher.update(bytes.fromhex(tensor_sha256(tensor)))
    return hasher.hexdigest()


def capture_step0_parity(
    *,
    model: Any,
    logits: Any,
    loss: Any,
    data: Any,
    targets: Any,
    token_bytes: Any,
    boundaries: Any,
    device_uuid: str,
    compile_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture exact replay fields after backward and before optimizer.step()."""

    torch = _torch()
    require_packer_flags_zero()
    named = list(model.named_parameters())
    preexisting = [(name, value) for name, value in named if not name.endswith("gpas_alpha")]
    gradients = [
        (name, value.grad)
        for name, value in preexisting
        if value.grad is not None
    ]
    gates = [
        {
            "name": name,
            "alpha": float(value.detach().float().item()),
            "gradient": (
                float(value.grad.detach().float().item())
                if value.grad is not None
                else None
            ),
            "forward_scale": float(
                (1.0 - torch.nn.functional.silu(value.detach().float())).item()
            ),
        }
        for name, value in named
        if name.endswith("gpas_alpha")
    ]
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "packer_flags": dict(PACKER_ZERO_CONTRACT),
        "device_uuid": device_uuid,
        "data_sha256": tensor_sha256(data),
        "targets_sha256": tensor_sha256(targets),
        "token_bytes_sha256": tensor_sha256(token_bytes),
        "boundaries_sha256": tensor_sha256(boundaries),
        "cpu_rng_sha256": tensor_sha256(torch.get_rng_state()),
        "cuda_rng_sha256": tensor_sha256(torch.cuda.get_rng_state()),
        "logits_sha256": tensor_sha256(logits),
        "loss_sha256": tensor_sha256(loss),
        "preexisting_parameters_sha256": named_tensors_sha256(preexisting),
        "preexisting_gradients_sha256": named_tensors_sha256(gradients),
        "gpas_gates": gates,
        "compile_identity": dict(compile_identity),
    }


def compare_step0_parity(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless alpha=0 treatment is an exact same-device replay."""

    mismatches = [
        field
        for field in PARITY_EXACT_FIELDS
        if control.get(field) != treatment.get(field)
    ]
    if control.get("packer_flags") != PACKER_ZERO_CONTRACT:
        mismatches.append("control_packer_flags")
    if treatment.get("packer_flags") != PACKER_ZERO_CONTRACT:
        mismatches.append("treatment_packer_flags")
    if control.get("compile_identity") != treatment.get("compile_identity"):
        mismatches.append("compile_identity")
    gates = treatment.get("gpas_gates", [])
    gate_contract = bool(gates) and all(
        gate.get("alpha") == 0.0
        and gate.get("gradient") is not None
        and math.isfinite(gate["gradient"])
        and gate["gradient"] != 0.0
        and gate.get("forward_scale") == 1.0
        for gate in gates
    )
    if not gate_contract:
        mismatches.append("zero_alpha_gate_contract")
    result = {
        "valid": not mismatches,
        "verdict": "PASS" if not mismatches else "INVALID_DIAGNOSTIC",
        "mismatches": mismatches,
    }
    if mismatches:
        raise RuntimeError(f"GPAS_STEP0_PARITY_FAILED {json.dumps(result, sort_keys=True)}")
    return result


class ResidualVarianceRecorder:
    """Eager-only post-MLP hook recorder for Paper-020's FP64 mediator."""

    def __init__(self, blocks: Iterable[Any]):
        self.blocks = list(blocks)
        self.rows: list[dict[str, Any]] = []
        self._step: int | None = None
        self._values: dict[int, list[float]] = {}
        self._handles = [
            block.register_forward_hook(self._hook(layer))
            for layer, block in enumerate(self.blocks)
        ]

    def _hook(self, layer: int):
        def record(_module, _inputs, output):
            if self._step is None:
                return
            value = (
                output.detach()
                .double()
                .var(dim=-1, correction=0)
                .mean()
                .item()
            )
            self._values.setdefault(layer, []).append(float(value))

        return record

    def begin_step(self, step: int) -> None:
        if self._step is not None:
            raise RuntimeError("GPAS_DIAGNOSTIC_STEP_ALREADY_ACTIVE")
        self._step = int(step)
        self._values = {}

    def end_step(self) -> dict[str, Any]:
        if self._step is None:
            raise RuntimeError("GPAS_DIAGNOSTIC_NO_ACTIVE_STEP")
        torch = _torch()
        variances = [
            sum(self._values[layer]) / len(self._values[layer])
            for layer in range(len(self.blocks))
        ]
        gates = []
        for layer, block in enumerate(self.blocks):
            alpha = getattr(block, "gpas_alpha", None)
            if alpha is not None:
                gates.append(
                    {
                        "layer": layer,
                        "alpha": float(alpha.detach().float().item()),
                        "gradient": (
                            float(alpha.grad.detach().float().item())
                            if alpha.grad is not None
                            else None
                        ),
                        "forward_scale": float(
                            (1.0 - torch.nn.functional.silu(alpha.detach().float())).item()
                        ),
                    }
                )
        row = {
            "step": self._step,
            "post_mlp_residual_variance": variances,
            "gates": gates,
        }
        self.rows.append(row)
        self._step = None
        self._values = {}
        return row

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


@contextmanager
def zero_gpas_gates(model: Any):
    """Temporarily zero every learned gate for the endpoint counterfactual."""

    torch = _torch()
    gates = [
        value
        for name, value in model.named_parameters()
        if name.endswith("gpas_alpha")
    ]
    saved = [gate.detach().clone() for gate in gates]
    try:
        with torch.no_grad():
            for gate in gates:
                gate.zero_()
        yield
    finally:
        with torch.no_grad():
            for gate, value in zip(gates, saved):
                gate.copy_(value)


class CompileIdentityTracker:
    """Collect Dynamo identities around a clean replay without model probes."""

    def __init__(self):
        self.events: list[tuple[str, str]] = []
        self._callback = None

    def __enter__(self):
        from torch._dynamo.callback import callback_handler

        def record(args):
            self.events.append((args.callback_trigger.name, str(args.compile_id)))

        self._callback = callback_handler.register_start_callback(record)
        return self

    def __exit__(self, *_exc):
        from torch._dynamo.callback import callback_handler

        callback_handler.remove_start_callback(self._callback)

    def report(self) -> dict[str, Any]:
        dynamo = [compile_id for kind, compile_id in self.events if kind == "DYNAMO"]
        encoded = ",".join(dynamo).encode("utf-8")
        return {
            "dynamo_graph_count": len(dynamo),
            "dynamo_recompile_count": max(0, len(dynamo) - 1),
            "compile_ids_sha256": hashlib.sha256(encoded).hexdigest(),
            "lazy_backward_count": sum(kind == "LAZY_BACKWARD" for kind, _ in self.events),
            "cudagraph_recording_count": sum(
                kind == "CUDAGRAPH_RECORDING" for kind, _ in self.events
            ),
        }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-parity", type=Path, required=True)
    parser.add_argument("--treatment-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    control = json.loads(args.control_parity.read_text(encoding="utf-8"))
    treatment = json.loads(args.treatment_parity.read_text(encoding="utf-8"))
    result = compare_step0_parity(control, treatment)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
