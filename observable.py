"""Stable observable collection helpers for autoresearch training.

Keep this file out of the model-edit loop: train.py should import and use the
register, while architecture experiments can add probe calls without changing
the serialization and logging protocol.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


class ObservableRegister:
    """Collect scalar probes during one training step, then flush and clear."""

    def __init__(self) -> None:
        self.enabled = True
        self.context = ""
        self._values: dict[str, float] = {}
        self._steps: list[dict[str, Any]] = []

    def clear(self) -> None:
        self._values.clear()

    def record_step(self, step: int, epoch: int | None, observables: dict[str, float]) -> None:
        """Append one step's merged observable row to the in-memory curve buffer.

        Used with flush()/build_step_observables to accumulate per-step curves that
        write_curves_json() later serializes for the dashboard and offline analysis.
        Diagnostic-only: never influences training.
        """
        if not self.enabled:
            return
        row: dict[str, Any] = {
            "step": int(step),
            "epoch": int(epoch) if epoch is not None else -1,
        }
        for key, value in observables.items():
            scalar = self._to_float(value)
            if scalar is not None and math.isfinite(scalar):
                row[key] = scalar
        self._steps.append(row)

    def write_curves_json(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write the dashboard's structured observable-series format atomically."""
        keys: set[str] = set()
        for row in self._steps:
            keys.update(row.keys() - {"step", "epoch"})
        series = {
            key: [
                {"step": row["step"], "value": row[key]}
                for row in self._steps
                if key in row
            ]
            for key in sorted(keys)
        }
        payload = {
            "metadata": dict(metadata or {}),
            "num_steps": len(self._steps),
            "series": series,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, destination)

    def set_context(self, context: str) -> None:
        self.context = context.strip(".")

    def add(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        scalar = self._to_float(value)
        if scalar is None or not math.isfinite(scalar):
            return
        if self.context:
            name = f"{self.context}.{name}"
        self._values[name] = scalar

    def add_l1_l2_abs_max(self, prefix: str, tensor: Any) -> None:
        if not self.enabled:
            return
        try:
            t = tensor.detach().float()
            self.add(f"{prefix}.l1", t.abs().mean())
            self.add(f"{prefix}.l2", t.square().mean().sqrt())
            self.add(f"{prefix}.abs_max", t.abs().max())
        except Exception:
            return

    def add_norms(self, prefix: str, tensor: Any) -> None:
        self.add_l1_l2_abs_max(prefix, tensor)

    def flush(self) -> dict[str, float]:
        values = dict(self._values)
        self.clear()
        return values

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "float"):
                value = value.float()
            if hasattr(value, "mean") and getattr(value, "ndim", 0) != 0:
                value = value.mean()
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
        except Exception:
            return None


OBS = ObservableRegister()


def build_step_observables(
    *,
    progress: float,
    train_loss: float,
    val_loss: float | None,
    dt: float,
    tok_per_sec: int,
    lrm_muon: float,
    lrm_adam: float,
    muon_momentum: float,
    muon_weight_decay: float,
    mfu: float,
) -> dict[str, float]:
    """Default step-level probes; add stable global variables here."""
    return {
        "progress": progress,
        "raw_train_loss": train_loss,
        "train_val_gap": train_loss - val_loss if val_loss is not None else 0.0,
        "tok_per_sec_k": tok_per_sec / 1000.0,
        "step_time_ms": dt * 1000.0,
        "lrm_muon": lrm_muon,
        "lrm_adam": lrm_adam,
        "muon_momentum": muon_momentum,
        "muon_weight_decay": muon_weight_decay,
        "mfu_percent": mfu,
    }


def format_observable_line(step: int, epoch: int | None, values: dict[str, float]) -> str:
    payload = " ".join(f"{key}={value:.6g}" for key, value in sorted(values.items()))
    return f"observable: step={step} epoch={epoch if epoch is not None else -1} {payload}"
