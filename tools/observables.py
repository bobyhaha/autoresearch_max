#!/usr/bin/env python3
"""Versioned OBSERVABLE REGISTRY.

Every mechanism must declare what it emits to prove it engaged. Without this a result can
say code changed val_bpb, but not why -- which is exactly the state the campaign was in:
51 successful runs, all emitting the same endpoint/throughput telemetry, zero
mechanism-specific activation measurements.

OPTIONAL IN THIS VERSION (operator directive 2026-08-18): observables are emitted where
cheap and reported for information, but they are NOT required, do NOT gate a launch, and do
NOT determine whether an intervention counts as a mechanism. Classification is by causal
structure alone (see tools/direction.py).
"""
SCHEMA_VERSION = 1

# name -> spec. `emits` are the exact keys train.py must print in its final key:value block.
OBSERVABLES = {
    "mtp": {
        "mediator": "auxiliary t+2 supervision signal actually reaching the trunk",
        "emits": ["mtp_aux_loss_final", "mtp_aux_loss_drop", "mtp_grad_norm_frac"],
        "activation_rule": "mtp_aux_loss_drop > 0.5",
        "why": "proves the aux head learned rather than sitting at init contributing only noise",
        "cost": "one extra lm_head pass; ~+12% FLOPs at vocab 8192",
    },
    "unet": {
        "mediator": "learned cross-depth skip pathway carrying signal",
        "emits": ["skip_lambda_absmean", "skip_lambda_absmax"],
        "activation_rule": "skip_lambda_absmean > 0.02",
        "why": "lambdas init to EXACTLY zero, so the model starts bit-identical to baseline; "
               "any movement off zero is unambiguous proof the optimizer used the path. "
               "This was promised in a comment and never emitted -- the U-net verdict is "
               "currently unfalsifiable.",
        "cost": "4 scalars, ~0.3 ms/step",
    },
    "zloss": {
        "mediator": "logit partition-function magnitude",
        "emits": ["logz_sq_mean", "zloss_frac_of_total"],
        "activation_rule": "zloss_frac_of_total > 0.001",
        "why": "distinguishes 'the penalty bit' from 'the penalty was numerically irrelevant'",
        "cost": "one logsumexp already computed by cross_entropy",
    },
    "noqknorm": {
        "mediator": "q/k vector scale and attention-logit spread",
        "emits": ["q_rms_mean", "k_rms_mean"],
        # attn_logit_absmax was specified and then withdrawn: the FA3 kernel never
        # materialises the attention logits, so measuring it would mean computing q@k.T
        # separately -- ~O(T^2) extra work that would change the very throughput the
        # comparison depends on. q/k RMS is sufficient to prove the norm removal engaged.
        "withdrawn_emits": {"attn_logit_absmax": "not observable under the fused FA3 kernel"},
        "activation_rule": "q_rms_mean > 1.05 or q_rms_mean < 0.95",
        "why": "removing the norm should visibly move q/k scale; if it does not, the ablation "
               "did not engage and any bpb change is something else",
        "cost": "two norms every 25th step",
    },
}

def required_emits(cfg: dict) -> list[str]:
    out=[]
    for k, spec in OBSERVABLES.items():
        if cfg.get(k) is not None: out += spec["emits"]
    return out

def missing_observables(cfg: dict, metrics: dict) -> list[str]:
    """Which declared observables a completed run failed to emit."""
    return [e for e in required_emits(cfg) if e not in (metrics or {})]

def is_instrumented_mechanism(cfg: dict) -> bool:
    return any(cfg.get(k) is not None for k in OBSERVABLES)
