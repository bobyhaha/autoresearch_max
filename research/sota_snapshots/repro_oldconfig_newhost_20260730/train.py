# Copyright 2026 Recursive
# Copyright 2025 Andrej Karpathy
# SPDX-License-Identifier: Apache-2.0
"""
Nanochat pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Mean validation BPB: 0.9109 (10 seeds).
Usage: uv run train.py
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import hashlib
import math
import struct
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch

# Keep default inductor settings for this compile-capture ablation.
import torch.nn as nn
import torch.nn.functional as F

ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "sdpa").lower()
CANON = os.environ.get("CANON","0")=="1"
CANON_K = int(os.environ.get("CANON_K","4"))
# Convergence-speed levers (env-gated; defaults = the current hardcoded values, so
# unset == byte-identical). Vetted panel bets for the underfit 2000-step regime.
LM_HEAD_INIT_STD = float(os.environ.get("LM_HEAD_INIT_STD", "0.001"))  # unembedding init-std sweep
SOFTCAP_CAP = float(os.environ.get("SOFTCAP_CAP", "16.5"))  # logit-softcap amplitude
SOFTCAP_TAU = float(os.environ.get("SOFTCAP_TAU", "15.0"))  # logit-softcap temperature
MLP_TYPE = os.environ.get("MLP_TYPE", "sqrelu").lower()
# GPAS (Chen et al., NeurIPS 2025): one zero-initialized scalar per decoder
# layer, reused after both residual sums.  The flag is deliberately strict and
# defaults off.  When off, no parameter is allocated and neither residual path
# executes an additional tensor operation.
_gpas_enable_raw = os.environ.get("GPAS_ENABLE", "0")
if _gpas_enable_raw not in {"0", "1"}:
    raise ValueError("GPAS_ENABLE must be 0 or 1")
GPAS_ENABLE = _gpas_enable_raw == "1"
# ===================== RSI INTERVENTION: BLOCK-14 MECHANISM PORT BEGIN =====================
# These three levers were previously implemented ONLY as forked one-off scripts on the
# remote box (repo/train_attnres_c1_v2.py, repo/train_c3.py) and were NEVER present in the
# canonical train.py. The registry described them as env-gated interventions on train.py,
# so every "treatment" arm launched against this file silently ran the control config --
# see reconciliation.json v23 and the block-14 retraction records. Ported here verbatim
# from those forks so the flags do what the registry claims. All three default OFF and are
# exact no-ops when off (AttnRes allocates no params and adds no ops; NGRAM_WD_LAMBDA=0.0
# reproduces the previous weight_decay=0.0; NGRAM_BACKOFF_KAPPA=0 skips the gate entirely).
ATTNRES_ENABLE = bool(int(os.environ.get("ATTNRES_ENABLE", "0")))  # Kimi-K3 depth retrieval
NGRAM_WD_LAMBDA = float(os.environ.get("NGRAM_WD_LAMBDA", "0.0"))  # C1: n-gram-table-only decoupled WD
NGRAM_BACKOFF_KAPPA = float(os.environ.get("NGRAM_BACKOFF_KAPPA", "0"))  # C3: freq-confidence backoff
# Recursive's reported "token-shifting" lever, absent from this fork until 2026-07-28:
# blend the previous token's Q/K projections into the current token's with learned
# per-dimension coefficients. Zero-init => exact no-op at step 0.
TOKEN_SHIFT = bool(int(os.environ.get("TOKEN_SHIFT", "0")))
# nGPT (arXiv 2410.01131, clm_ngpt_step_reduction): keep the residual stream on a
# hypersphere by re-normalizing it after each block, with the block's contribution
# entering as a learnable-rate increment ("eigen learning rate") rather than a raw
# add. Reported 4-20x reduction in steps-to-quality, which is exactly the axis this
# wall-clock frame rewards. This is a STRUCTURAL change, not a no-op flag: at
# NGPT_ALPHA_INIT=1.0 the update reduces to x <- norm(block(x)), which is NOT the
# baseline. Expected to need refinement before it wins.
NGPT_SPHERE = bool(int(os.environ.get("NGPT_SPHERE", "0")))
NGPT_ALPHA_INIT = float(os.environ.get("NGPT_ALPHA_INIT", "0.25"))
if NGRAM_WD_LAMBDA < 0.0:
    raise ValueError("NGRAM_WD_LAMBDA must be >= 0")
if NGRAM_BACKOFF_KAPPA < 0.0:
    raise ValueError("NGRAM_BACKOFF_KAPPA must be >= 0")
# ====================== RSI INTERVENTION: BLOCK-14 MECHANISM PORT END ======================
if ATTN_BACKEND not in {"sdpa", "fa3", "fa4"}:
    raise ValueError("ATTN_BACKEND must be one of: sdpa, fa3, fa4")
if MLP_TYPE not in {"sqrelu", "swiglu"}:
    raise ValueError("MLP_TYPE must be one of: sqrelu, swiglu")
cap = torch.cuda.get_device_capability()

if ATTN_BACKEND == "sdpa":
    # H200 fast fix: avoid the flash-attention native extension path, which can
    # segfault before Python can report a traceback. This keeps the same data,
    # model scale, and causal attention shape while using PyTorch's CUDA SDPA.
    def attention_func(q, k, v, causal=True, window_size=(-1, -1)):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=causal,
        ).transpose(1, 2)

    # ===== EXPLORE:DOC_MASK BEGIN =====
    # Block-diagonal (intra-document) causal attention for the SDPA path only.
    # Reached from CausalSelfAttention.forward exclusively when DOC_MASK=1 and a
    # non-None mask is supplied, so the baseline attention_func path is never
    # perturbed. q/k/v arrive as (B, T, H, D). The masks themselves are built
    # ONCE PER FORWARD (not once per layer) by the builders below and shared by
    # every layer with the same left window -- in DOC_MASK_MODE=seg that is a
    # single mask for the whole stack.
    #
    # Two implementations, selected by DOC_MASK_IMPL:
    #   dense (default) -- materialize a (B, 1, T, T) boolean mask and take the
    #     explicit-mask SDPA path. Numerically the frozen behaviour, but the mask
    #     is re-read by the kernel on every layer in both fwd and bwd
    #     (B=72, T=2048 -> 302 MB per read), which is what dominates the measured
    #     "segmentation costs 1.78x" figure.
    #   flex -- compile a block-sparse BlockMask once per batch and run
    #     FlexAttention, so 128x128 blocks that are entirely cross-document are
    #     never computed and no dense mask is ever materialized. Same masking
    #     semantics, different kernel (so bf16 outputs match only to tolerance).
    def _doc_mask_causal_window(T, window_left, device):
        """(T, T) causal (+ optional left-window) envelope shared by both impls."""
        pos = torch.arange(T, device=device)
        allowed = pos[:, None] >= pos[None, :]
        if DOC_MASK_MODE in ("both", "window"):
            if window_left is not None and 0 < window_left < T:
                allowed = allowed & (pos[None, :] >= (pos[:, None] - window_left + 1))
        return allowed

    def _build_dense_doc_mask(seg, window_left):
        T = seg.shape[1]
        allowed = _doc_mask_causal_window(T, window_left, seg.device)
        # DOC_MASK_MODE disentangles the two effects: "both" (default) = window+seg,
        # "window" = sliding-window enforcement only, "seg" = intra-document mask only.
        if DOC_MASK_MODE in ("both", "seg"):
            # Same-document constraint, broadcast over heads -> (B, 1, T, T).
            same_seg = seg[:, :, None] == seg[:, None, :]
            return (same_seg & allowed[None, :, :]).unsqueeze(1)
        return allowed[None, None, :, :]

    def _build_flex_doc_mask(seg, window_left):
        """Block-sparse BlockMask with the same semantics as the dense mask.

        Built outside the compiled region (see build_doc_masks_for_batch); the
        returned BlockMask is passed into GPT.forward as a traced input.
        """
        from torch.nn.attention.flex_attention import create_block_mask

        B, T = seg.shape
        use_window = (
            DOC_MASK_MODE in ("both", "window")
            and window_left is not None
            and 0 < window_left < T
        )
        use_seg = DOC_MASK_MODE in ("both", "seg")

        def mask_mod(b, h, q_idx, kv_idx):
            allowed = q_idx >= kv_idx
            if use_window:
                allowed = allowed & (kv_idx > q_idx - window_left)
            if use_seg:
                allowed = allowed & (seg[b, q_idx] == seg[b, kv_idx])
            return allowed

        # B=None when the mask does not depend on the batch row (window-only
        # mode) so one block mask is broadcast instead of B identical copies.
        return create_block_mask(
            mask_mod,
            B if use_seg else None,
            None,
            T,
            T,
            device=seg.device,
            _compile=True,
            # ===== RSI INTERVENTION: DOC_MASK_BLOCK =====
            # Block-sparse granularity knob. Larger blocks track fewer blocks (less
            # mask/bookkeeping overhead) but force fully-computing any partially-masked
            # block (more wasted attention FLOPs); smaller blocks are the reverse. The
            # flex doc-mask is currently ~3% slower than fa3 in steps, so this is the
            # direct lever on that gap. 128 is FlexAttention's own default => unchanged.
            BLOCK_SIZE=DOC_MASK_BLOCK,
        )

    def _doc_masked_attention(q, k, v, mask):
        if DOC_MASK_IMPL == "flex":
            from torch.nn.attention.flex_attention import flex_attention

            return flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=mask,
            ).transpose(1, 2)
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            is_causal=False,
        ).transpose(1, 2)
    # ===== EXPLORE:DOC_MASK END =====

    print(f"Using PyTorch SDPA attention (GPU capability {cap})")
elif ATTN_BACKEND == "fa4":
    if cap[0] < 10:
        raise RuntimeError(
            f"ATTN_BACKEND=fa4 requires a Blackwell GPU (SM100+), got capability {cap}"
        )
    # Blackwell (B200, SM100): wrap flash-attn-4 as a custom op so torch.compile
    # treats it as opaque (no tracing into cutlass DSL, no recompile-cache thrash,
    # no per-call Python kernel build).
    from flash_attn.cute import flash_attn_func as _fa4_raw
    from flash_attn.cute.interface import _flash_attn_bwd as _fa4_bwd_raw

    @torch.library.custom_op("fa4::fa4_causal", mutates_args=())
    def _fa4_causal_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       window_left: int) -> tuple[torch.Tensor, torch.Tensor]:
        ws = (window_left, 0) if window_left > 0 else (None, None)
        out, lse = _fa4_raw(q, k, v, causal=True, window_size=ws, return_lse=True)
        return out, lse

    @_fa4_causal_op.register_fake
    def _fa4_causal_fake(q, k, v, window_left):
        B, T, H, D = q.shape
        return torch.empty_like(q), torch.empty(B, H, T, device=q.device, dtype=torch.float32)

    def _fa4_setup_context(ctx, inputs, output):
        q, k, v, window_left = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.window_left = window_left

    @torch.library.custom_op("fa4::fa4_bwd", mutates_args=())
    def _fa4_bwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor, grad_output: torch.Tensor, lse: torch.Tensor,
                    window_left: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wl = window_left if window_left > 0 else None
        dq, dk, dv = _fa4_bwd_raw(
            q, k, v, out, grad_output, lse,
            causal=True, window_size_left=wl, window_size_right=0,
        )
        return dq, dk, dv

    @_fa4_bwd_op.register_fake
    def _fa4_bwd_fake(q, k, v, out, grad_output, lse, window_left):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _fa4_backward(ctx, grad_output, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.ops.fa4.fa4_bwd(q, k, v, out, grad_output, lse, ctx.window_left)
        return dq, dk, dv, None

    _fa4_causal_op.register_autograd(_fa4_backward, setup_context=_fa4_setup_context)

    def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
        wl = window_size[0] if isinstance(window_size, tuple) else window_size
        if wl is None or wl <= 0 or wl >= q.shape[1]:
            wl = -1
        out, _lse = torch.ops.fa4.fa4_causal(q, k, v, wl)
        return out

    attention_func = flash_attn_func
    print(f"Using flash-attn-4 as custom op (GPU capability {cap})")
elif ATTN_BACKEND == "fa3":
    # Hopper (H100, H200): flash-attn-3 via kernels package, honoring per-layer
    # sliding windows (the model's TTTL design that SDPA silently ignores).
    # FIX: the raw kernel is a HF-kernels-loaded autograd.Function; letting
    # torch.compile(fullgraph=True) trace INTO it segfaults the process before
    # Python can report a traceback (the "H200 segfault" that forced SDPA).
    # allow_in_graph marks it as an opaque graph leaf so dynamo keeps it in the
    # graph without tracing its internals -- mirrors why fa4 is wrapped as a
    # custom_op. Autograd still flows through .apply at eager level.
    # FIX 2 (2026-07-24): allow_in_graph was NOT enough and still segfaulted.
    # Diagnosis: the kernel itself is fine -- probed directly, eager causal AND
    # eager windowed both pass. The fault is torch.compile tracing into
    # `flash_attn_func`, which dispatches through `FlashAttnFunc`, an
    # autograd.Function. allow_in_graph marks it opaque but supplies NO fake/meta
    # implementation, so shape propagation still walks in and dies.
    #
    # The kernels package already exposes `_flash_attn_forward` /
    # `_flash_attn_backward` as CustomOpDefs (with schemas and registered fakes).
    # Wrapping THOSE in an fa4-style torch.library.custom_op -- exactly how
    # upstream handles Blackwell -- gives dynamo an opaque op with a meta impl and
    # compiles cleanly under fullgraph=True. Verified: causal and windowed, finite
    # grads, no segfault.
    from kernels import get_kernel

    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    # `repo` above is one of exactly two hardcoded literals (never user/env input), both
    # long-adopted, campaign-vetted dependencies -- so trust is already scoped at the call
    # site. Newer `kernels` releases (>=~0.13) require trust_remote_code=True or fail with a
    # publisher-trust lookup that also breaks outright against a mirror HF_ENDPOINT (verified:
    # hf-mirror.com 404s on the org-overview API huggingface.co is unreachable without).
    # Older releases don't recognize the kwarg at all (TypeError: unexpected keyword
    # argument). Try the newer call first and fall back, so this works across the range of
    # `kernels` versions pip may resolve on a given host without pinning one exactly.
    try:
        _fa3_mod = get_kernel(repo, trust_remote_code=True).flash_attn_interface
    except TypeError:
        _fa3_mod = get_kernel(repo).flash_attn_interface
    _fa3_fwd_raw, _fa3_bwd_raw = _fa3_mod._flash_attn_forward, _fa3_mod._flash_attn_backward

    @torch.library.custom_op("fa3::fwd", mutates_args=())
    def _fa3_fwd_op(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_left: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse, *_ = _fa3_fwd_raw(
            q, k, v, causal=True, window_size_left=window_left, window_size_right=0
        )
        return out, lse

    @_fa3_fwd_op.register_fake
    def _fa3_fwd_fake(q, k, v, window_left):
        batch, seq, heads, _ = q.shape
        return torch.empty_like(q), torch.empty(
            batch, heads, seq, device=q.device, dtype=torch.float32
        )

    @torch.library.custom_op("fa3::bwd", mutates_args=())
    def _fa3_bwd_op(
        dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        out: torch.Tensor, lse: torch.Tensor, window_left: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # `_flash_attn_backward` in this kernel build returns ONLY softmax_d (a single
        # tensor, not a (dq, dk, dv, softmax_d) tuple as an older kernel-build API had):
        # dq/dk/dv are OUT-parameters -- pre-allocate and pass them in, the C++ call fills
        # them in place, and the Python wrapper's return value is unrelated (discarded
        # below). The prior code unpacked the single returned tensor as if it were a
        # 3+-tuple, silently slicing it along dim 0 into three wrong-shaped, mutually
        # aliasing "dq/dk/dv" -- undetected until torch's custom-op aliasing check (added
        # in a newer torch release) started raising on it.
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _fa3_bwd_raw(
            dout, q, k, v, out, lse, is_causal=True,
            window_size_left=window_left, window_size_right=0,
            dq=dq, dk=dk, dv=dv,
        )
        return dq, dk, dv

    @_fa3_bwd_op.register_fake
    def _fa3_bwd_fake(dout, q, k, v, out, lse, window_left):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _fa3_setup_context(ctx, inputs, output):
        q, k, v, window_left = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.window_left = window_left

    def _fa3_backward(ctx, grad_out, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.ops.fa3.bwd(
            grad_out.contiguous(), q, k, v, out, lse, ctx.window_left
        )
        return dq, dk, dv, None

    _fa3_fwd_op.register_autograd(_fa3_backward, setup_context=_fa3_setup_context)

    # ===== RSI INTERVENTION: FA3 VARLEN DOC-MASK BEGIN =====
    # Intra-document attention masking ON THE FA3 KERNEL, resolving
    # gap_fa3_varlen_docmask. Block-sparse doc-masking is a confirmed -0.0111 win
    # against a matched sdpa control, but the DOC_MASK path was gated on sdpa, and
    # the fa3->sdpa backend penalty (+0.012) cancelled the benefit. Running it on
    # fa3 removes that penalty entirely.
    #
    # The gap record assumed this needed wrapping FlashAttnVarlenFunc (an
    # autograd.Function -- the documented cause of this box's segfault under
    # fullgraph). It does not: `_flash_attn_forward`/`_flash_attn_backward` are
    # ALREADY registered custom ops whose schemas accept cu_seqlens_q/cu_seqlens_k/
    # max_seqlen_*, so varlen is reachable through the same safe path the non-varlen
    # wrapper already uses. Verified eagerly before implementing: with q,k,v of shape
    # (total,H,D) and int32 cu_seqlens, forward returns out (total,H,D) and lse
    # (H,total) fp32, and backward returns dq/dk/dv all (total,H,D).
    @torch.library.custom_op("fa3::fwd_varlen", mutates_args=())
    def _fa3_fwd_varlen_op(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        cu_seqlens: torch.Tensor, max_seqlen: int, window_left: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse, *_ = _fa3_fwd_raw(
            q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True, window_size_left=window_left, window_size_right=0,
        )
        return out, lse

    @_fa3_fwd_varlen_op.register_fake
    def _fa3_fwd_varlen_fake(q, k, v, cu_seqlens, max_seqlen, window_left):
        total, heads, _ = q.shape
        return torch.empty_like(q), torch.empty(
            heads, total, device=q.device, dtype=torch.float32
        )

    @torch.library.custom_op("fa3::bwd_varlen", mutates_args=())
    def _fa3_bwd_varlen_op(
        dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        out: torch.Tensor, lse: torch.Tensor,
        cu_seqlens: torch.Tensor, max_seqlen: int, window_left: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # See the identical fix + explanation in _fa3_bwd_op above: this kernel build's
        # _flash_attn_backward returns only softmax_d: dq/dk/dv are OUT-parameters that
        # must be pre-allocated and passed in, not unpacked from the return value.
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _fa3_bwd_raw(
            dout, q, k, v, out, lse, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, is_causal=True,
            window_size_left=window_left, window_size_right=0,
            dq=dq, dk=dk, dv=dv,
        )
        return dq, dk, dv

    @_fa3_bwd_varlen_op.register_fake
    def _fa3_bwd_varlen_fake(dout, q, k, v, out, lse, cu_seqlens, max_seqlen, window_left):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _fa3_varlen_setup_context(ctx, inputs, output):
        q, k, v, cu_seqlens, max_seqlen, window_left = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        ctx.window_left = window_left

    def _fa3_varlen_backward(ctx, grad_out, grad_lse):
        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = torch.ops.fa3.bwd_varlen(
            grad_out.contiguous(), q, k, v, out, lse,
            cu_seqlens, ctx.max_seqlen, ctx.window_left,
        )
        return dq, dk, dv, None, None, None

    _fa3_fwd_varlen_op.register_autograd(
        _fa3_varlen_backward, setup_context=_fa3_varlen_setup_context
    )

    def fa3_varlen_attn(q, k, v, cu_seqlens, max_seqlen, window_size=(-1, -1)):
        """(B,T,H,D) -> flatten to (B*T,H,D), varlen attend, reshape back.

        cu_seqlens marks document boundaries across the FLATTENED batch and must
        already include every batch-row start, otherwise a document ending row b
        would silently merge with the one beginning row b+1.
        """
        B, T, H, D = q.shape
        wl = window_size[0] if isinstance(window_size, tuple) else window_size
        if wl is None or wl <= 0 or wl >= T:
            wl = -1
        qf, kf, vf = (t.reshape(B * T, H, D) for t in (q, k, v))
        out, _lse = torch.ops.fa3.fwd_varlen(qf, kf, vf, cu_seqlens, max_seqlen, wl)
        return out.reshape(B, T, H, D)
    # ====== RSI INTERVENTION: FA3 VARLEN DOC-MASK END ======

    def flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)):
        wl = window_size[0] if isinstance(window_size, tuple) else window_size
        if wl is None or wl <= 0 or wl >= q.shape[1]:
            wl = -1
        out, _lse = torch.ops.fa3.fwd(q, k, v, wl)
        return out

    attention_func = flash_attn_func
    print(f"Using flash-attn-3 (custom_op) from {repo} (GPU capability {cap})")

from lib import (  # noqa: E402
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    evaluate_bpb,
    evaluate_bpb_fast,
    load_data_split,
    make_dataloader,
    split_parquet_files,
)
from observable import OBS, build_step_observables, format_observable_line  # noqa: E402


# ===== EXPLORE:DOC_MASK BEGIN =====
def _doc_mask_key(window_left):
    """Cache key for a per-layer doc mask.

    In DOC_MASK_MODE=seg the left window is not enforced, so every layer shares
    ONE mask (key -1) instead of one per distinct window.
    """
    if DOC_MASK_MODE == "seg":
        return -1
    return window_left


_DOC_MASK_BOUNDARY_COUNTS = []


def _compact_doc_boundaries(idx):
    """Construct the exact compact FA3 boundary vector for one packed batch."""
    assert DOC_MASK_BOS_ID is not None, "compact boundaries require a resolved BOS id"
    B, T = idx.shape
    starts = (idx == DOC_MASK_BOS_ID)
    # Force a boundary at every batch-row start: after flattening to (B*T,), a
    # document ending row b would otherwise merge with the one opening row b+1.
    starts = starts.clone()
    starts[:, 0] = True
    flat = starts.reshape(-1)
    pos = torch.nonzero(flat, as_tuple=False).flatten().to(torch.int32)
    cu = torch.cat([pos, torch.tensor([B * T], device=idx.device, dtype=torch.int32)])
    # cu_seqlens' length is data-dependent. One dynamic dimension lets the
    # fullgraph model serve every document count without padding the vector.
    torch._dynamo.mark_dynamic(cu, 0)
    return cu, T


def attest_compact_doc_boundaries(idx):
    """Fail closed on a real packed batch before either arm starts its clock.

    Both explicit arms execute this same pre-clock CUDA construction and host
    synchronization, so the attestation cannot create treatment-only warmup.
    Per-training-batch boundary counts are recorded separately without reading
    tensor contents or synchronizing inside the charged region.
    """
    cu, T = _compact_doc_boundaries(idx)
    B = idx.shape[0]
    cu_cpu = cu.detach().cpu()
    diffs = cu_cpu[1:] - cu_cpu[:-1]
    row_starts = torch.arange(B, dtype=torch.int32) * T
    row_start_mask = torch.isin(row_starts, cu_cpu[:-1])
    checks = {
        "first_zero": bool(cu_cpu.numel() >= 2 and cu_cpu[0].item() == 0),
        "last_equals_b_times_t": bool(cu_cpu[-1].item() == B * T),
        "strictly_increasing": bool(torch.all(diffs > 0).item()),
        "all_row_starts_present": bool(torch.all(row_start_mask).item()),
        "max_gap_lte_t": bool(torch.all(diffs <= T).item()),
    }
    if not all(checks.values()):
        detail = " ".join(f"{key}={int(value)}" for key, value in checks.items())
        raise RuntimeError(f"DOC_MASK_BOUNDARY_ATTESTATION_FAILED {detail}")
    print(
        "DOC_MASK_BOUNDARY_ATTESTED=1 "
        f"boundary_count={cu_cpu.numel() - 1} "
        f"cu_first={cu_cpu[0].item()} cu_last={cu_cpu[-1].item()} "
        f"b_times_t={B * T} min_gap={diffs.min().item()} "
        f"max_gap={diffs.max().item()} "
        + " ".join(f"{key}=1" for key in checks),
        flush=True,
    )


def build_doc_masks_for_batch(idx, window_lefts, record_boundary_count=False):
    """Per-batch doc masks, built OUTSIDE the compiled region.

    Returns ``None`` for every configuration that does not take the doc-mask
    attention path (so the baseline call sites are unchanged), and otherwise a
    ``{key: mask}`` dict covering the distinct per-layer left windows.

    Only the ``flex`` implementation *requires* this hoisting: ``create_block_mask``
    is itself a compiled helper and cannot be traced inside a
    ``fullgraph=True`` graph. The ``dense`` implementation is happy to build its
    masks inside ``GPT.forward`` and does so when ``doc_masks`` is None, which
    keeps every existing call site and test valid.
    """
    # ===== RSI INTERVENTION: FA3 VARLEN DOC-MASK BEGIN =====
    # In varlen mode there is no boolean mask at all: document structure is handed
    # to the fa3 kernel as cu_seqlens, so this returns (cu_seqlens, max_seqlen) for
    # every window key. Built here, outside the compiled region, because the
    # boundary scan is data-dependent (torch.nonzero) and cannot be traced under
    # fullgraph=True -- the same reason the flex BlockMask is hoisted.
    if DOC_MASK and DOC_MASK_IMPL == "varlen" and ATTN_BACKEND == "fa3":
        cu, T = _compact_doc_boundaries(idx)
        if record_boundary_count:
            # Shape metadata is already known after nonzero; this append performs
            # no tensor-content read and no extra device synchronization.
            _DOC_MASK_BOUNDARY_COUNTS.append(cu.shape[0] - 1)
        # RECOMPILE FIX: both of these are data-dependent and, left alone, blow the
        # fullgraph=True recompile limit (observed: FailOnRecompileLimitHit).
        #  - max_seqlen is a Python int that Dynamo specializes on, and it changes
        #    with every batch's longest document. It is only an UPPER BOUND used for
        #    kernel scheduling, not for correctness, and we force a boundary at each
        #    batch-row start, so no document can exceed T. Pin it to the constant T.
        payload = (cu, T)
        return {_doc_mask_key(w): payload for w in window_lefts}
    # ====== RSI INTERVENTION: FA3 VARLEN DOC-MASK END ======
    if not DOC_MASK or ATTN_BACKEND != "sdpa" or DOC_MASK_IMPL != "flex":
        return None
    assert DOC_MASK_BOS_ID is not None, "DOC_MASK=1 requires a resolved BOS id"
    seg = (idx == DOC_MASK_BOS_ID).cumsum(dim=1)
    masks = {}
    for window_left in window_lefts:
        key = _doc_mask_key(window_left)
        if key not in masks:
            masks[key] = _build_flex_doc_mask(seg, window_left)
    return masks


def build_doc_masks_from_boundaries(
    boundary_lease, window_lefts, record_boundary_count=False
):
    """Build the FA3 payload from the packer's asynchronous boundary sidecar.

    This is intentionally a separate entry point from
    :func:`build_doc_masks_for_batch`: the latter remains the scanner reference
    used by evaluation, control training, attestation, and equivalence checks.
    Accessing ``cu_seqlens`` inserts the required H2D-event wait on the current
    training stream before the compiled model can consume the tensor.
    """
    if not (DOC_MASK and DOC_MASK_IMPL == "varlen" and ATTN_BACKEND == "fa3"):
        raise RuntimeError(
            "PACKER_DOC_BOUNDARIES requires DOC_MASK=1 "
            "DOC_MASK_IMPL=varlen ATTN_BACKEND=fa3"
        )
    cu = boundary_lease.cu_seqlens
    if cu.dtype != torch.int32 or cu.device.type != "cuda":
        raise RuntimeError(
            "PACKER_DOC_BOUNDARY_DTYPE_DEVICE_MISMATCH "
            f"dtype={cu.dtype} device={cu.device}"
        )
    if boundary_lease.length < 2 or boundary_lease.max_seqlen != MAX_SEQ_LEN:
        raise RuntimeError(
            "PACKER_DOC_BOUNDARY_SHAPE_MISMATCH "
            f"length={boundary_lease.length} "
            f"max_seqlen={boundary_lease.max_seqlen} expected={MAX_SEQ_LEN}"
        )
    torch._dynamo.mark_dynamic(cu, 0)
    if record_boundary_count:
        _DOC_MASK_BOUNDARY_COUNTS.append(boundary_lease.length - 1)
    payload = (cu, boundary_lease.max_seqlen)
    return {_doc_mask_key(w): payload for w in window_lefts}
# ===== EXPLORE:DOC_MASK END =====


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # Source-faithful GPAS is opt-in.  Keeping this on the config (rather than
    # reading the environment from Block) also lets the same process construct
    # an explicit control and treatment for same-device diagnostic replay.
    gpas: bool = False
    # Rare-token / n-gram capacity: bigram & trigram VE hash tables are sized
    # vocab_size * ngram_table_mult. Default 64 reproduces the frozen baseline.
    ngram_table_mult: int = 64
    # Per-type capacity override (0 = inherit ngram_table_mult). Lets a run scale the
    # bigram and trigram VE tables independently, to decompose which one drives the win.
    ngram_bigram_mult: int = 0
    ngram_trigram_mult: int = 0
    # 4-gram value-embedding order (0 = OFF -> no 4-gram tables/gate/injection, byte-identical
    # baseline). >0 sizes the 4-gram VE hash tables at vocab_size * ngram_fourgram_mult.
    ngram_fourgram_mult: int = 0
    # ===== EXPLORE:NGRAM_FOURGRAM_SPAN BEGIN =====
    # Offset of the FURTHEST slot in the 4-gram context. 3 = contiguous
    # (w[-3],w[-2],w[-1],w[0]) = the frozen default. 4 = gappy (w[-4],w[-2],w[-1],w[0]):
    # same token count, longer reach. See the env-var block for the SPARSITY-vs-RANGE
    # rationale this lever exists to discriminate.
    ngram_fourgram_span: int = 3
    # ===== EXPLORE:NGRAM_FOURGRAM_SPAN END =====
    # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
    # 5-gram value-embedding order (0 = OFF -> no 5-gram tables/gate/injection,
    # byte-identical baseline). >0 sizes the 5-gram VE hash tables at
    # vocab_size * ngram_fivegram_mult.
    ngram_fivegram_mult: int = 0
    # ===== EXPLORE:NGRAM_FIVEGRAM END =====
    # Shared trigram VE (vora): all trigram layers share one table+prime family. Default False.
    shared_trigram_ve: bool = False
    # ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
    # Learned product-key memory (Lample et al. 2019) on the trigram layers.
    # pk_mem=False (default) -> no modules created, byte-identical baseline.
    # N = pk_mem_subkeys**2 slots; retrieval is O(sqrt(N)) via two sub-key
    # codebooks of pk_mem_subkeys keys each (dim pk_mem_qdim/2).
    pk_mem: bool = False
    pk_mem_subkeys: int = 2048
    pk_mem_topk: int = 16
    pk_mem_qdim: int = 128
    # ===== EXPLORE:NGRAM_PK_MEMORY END =====


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# ===================== RSI INTERVENTION: C3 BACKOFF GATE HELPER BEGIN =====================
def _c3_gate(model, kind, layer_i, k, ids):
    """Frequency-confidence backoff gate: count/(count+kappa), then record this step.

    A neural Good-Turing discount on the n-gram memory. Keys seen many times keep
    (nearly) their full learned correction; rarely-seen keys -- whose stored vector is
    dominated by shard-specific noise in this repeated-data regime -- are shrunk toward
    zero. Counts are updated AFTER computing the gate so a key's first occurrence sees
    gate=0 rather than a leaked count from its own current appearance.

    Returns a (B, T, 1) multiplier. Only reached when NGRAM_BACKOFF_KAPPA > 0.
    """
    buf = getattr(model, f"ngram_count_{kind}_{layer_i}_{k}")
    counts = buf[ids]                                            # (B, T)
    gate = counts / (counts + NGRAM_BACKOFF_KAPPA)               # (B, T), in [0, 1)
    with torch.no_grad():
        flat = ids.reshape(-1)
        buf.scatter_add_(0, flat, torch.ones_like(flat, dtype=buf.dtype))
    return gate.unsqueeze(-1)
# ====================== RSI INTERVENTION: C3 BACKOFF GATE HELPER END ======================


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


@torch.no_grad()
def observe_attention_distribution(prefix, q, k, window_size, sample_tokens=64):
    """Sampled causal attention entropy probe for layer diagnostics."""
    try:
        q0 = q[0].detach().float().transpose(0, 1)  # H, T, D
        k0 = k[0].detach().float().transpose(0, 1)  # H_kv, T, D
        if k0.size(0) != q0.size(0):
            repeat = q0.size(0) // k0.size(0)
            k0 = k0.repeat_interleave(repeat, dim=0)
        total_tokens = q0.size(1)
        q_count = min(sample_tokens, total_tokens)
        q_start = total_tokens - q_count
        q_sample = q0[:, q_start:, :]
        scores = torch.matmul(q_sample, k0.transpose(-2, -1)) / (q_sample.size(-1) ** 0.5)
        query_positions = torch.arange(q_start, total_tokens, device=scores.device)[:, None]
        key_positions = torch.arange(total_tokens, device=scores.device)[None, :]
        visible = key_positions <= query_positions
        left_window = window_size[0] if isinstance(window_size, tuple) else window_size
        if left_window is not None and left_window > 0 and left_window < total_tokens:
            visible = visible & (key_positions >= (query_positions - left_window + 1))
        scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        support = visible.sum(dim=-1).float().clamp_min(1)
        OBS.add(f"{prefix}.attn_entropy", entropy.mean())
        # Normalize by log(support) = max possible entropy. Clamp support (not its log)
        # to >=2: a single-key row has support=1 -> log(1)=0; clamping the log to 1e-6
        # would divide ~0 entropy by ~0 and blow the metric up ~1e6. log(2) keeps it bounded.
        OBS.add(f"{prefix}.attn_entropy_norm", (entropy / support.clamp_min(2).log()).mean())
        OBS.add(f"{prefix}.attn_max_prob", probs.max(dim=-1).values.mean())
    except Exception:
        return


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # ===== RSI INTERVENTION: TOKEN_SHIFT BEGIN =====
        # Per-dimension shift amount for Q and K, zero-init => exact no-op at step 0.
        # Allocated only when the lever is on so the OFF parameter set is unchanged.
        if TOKEN_SHIFT:
            self.tshift_q = nn.Parameter(torch.zeros(self.head_dim))
            self.tshift_k = nn.Parameter(torch.zeros(self.head_dim))
        # ====== RSI INTERVENTION: TOKEN_SHIFT END ======
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Separate gate for bigram VE on ALL VE layers reading decorrelated channels (32:64)
        self.bigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )
        # Trigram gate on layers 1, 5, and 7 for late/full-context coverage, reads channels 64:96
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        trigram_layers = (
            {ve_layers[0], ve_layers[-2], ve_layers[-1]}
            if len(ve_layers) >= 2
            else {ve_layers[-1]}
        )
        self.trigram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if layer_idx in trigram_layers
            else None
        )
        # 4-gram gate on the same layers as trigram, reads channels 96:128. Only created
        # when 4-gram VE is enabled (ngram_fourgram_mult>0); else None -> baseline-identical.
        fourgram_layers = trigram_layers if getattr(config, "ngram_fourgram_mult", 0) > 0 else set()
        self.fourgram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if layer_idx in fourgram_layers
            else None
        )
        # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
        # 5-gram gate on the same layers as trigram/fourgram. Slice bookkeeping:
        # slice 0 = ve (0:32), 1 = bigram (32:64), 2 = trigram (64:96),
        # 3 = fourgram (96:128), 4 = PK memory (128:160) -> the next FREE slice is
        # 5, so the 5-gram gate reads channels 160:192 and composes with
        # NGRAM_PK_MEMORY=1 without overlap (6*32 = 192 <= n_embd, asserted).
        # Only created when 5-gram VE is enabled (ngram_fivegram_mult>0); else
        # None -> baseline-identical param set.
        fivegram_layers = trigram_layers if getattr(config, "ngram_fivegram_mult", 0) > 0 else set()
        if fivegram_layers:
            assert config.n_embd >= 6 * self.ve_gate_channels, "fivegram gate reads channels 160:192"
        self.fivegram_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if layer_idx in fivegram_layers
            else None
        )
        # ===== EXPLORE:NGRAM_FIVEGRAM END =====
        # ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
        # Input-dependent per-KV-head gate for the product-key memory injection,
        # matching the ve/bigram/trigram/fourgram gate pattern; reads decorrelated
        # channels 128:160. Only created on PK layers (= trigram layers) when the
        # flag is on; None otherwise -> baseline param set unchanged.
        pk_layers = trigram_layers if getattr(config, "pk_mem", False) else set()
        self.pk_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if layer_idx in pk_layers
            else None
        )
        # ===== EXPLORE:NGRAM_PK_MEMORY END =====
        # Head-level MoE gate on ALL layers for attention output routing
        self.head_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size, bigram_ve=None, trigram_ve=None, fourgram_ve=None, doc_mask=None, pk_ve=None, fivegram_ve=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # ===== RSI INTERVENTION: TOKEN_SHIFT BEGIN =====
        # Recursive's own reported lever ("token-shifting": blend the previous token's
        # attention projections Q and K into the current token's with learned
        # per-dimension coefficients). Cheap RWKV-style shift: one roll + one lerp per
        # projection, no extra matmul, so throughput cost should be near zero. Gives each
        # position a learned amount of the immediately-preceding token's query/key
        # content, a local-composition prior the n-gram value memory does NOT supply
        # (that path feeds the VALUE stream, not the QK matching stream).
        # Coefficients are zero-init => sigmoid(0)=0.5 would be a 50/50 blend, so we
        # instead parameterize the shift amount directly and zero-init it: shift=0 means
        # pure current token, i.e. an EXACT no-op at step 0, byte-identical when off.
        if TOKEN_SHIFT:
            q_prev = torch.cat([q[:, :1], q[:, :-1]], dim=1)
            k_prev = torch.cat([k[:, :1], k[:, :-1]], dim=1)
            # (head_dim,) learned per-dimension mix, broadcast over B/T/heads
            q = q + self.tshift_q * (q_prev - q)
            k = k + self.tshift_k * (k_prev - k)
        # ====== RSI INTERVENTION: TOKEN_SHIFT END ======

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Bigram VE with its own independent gate reading from decorrelated channels (32:64)
        if bigram_ve is not None:
            bigram_ve = bigram_ve.view(B, T, self.n_kv_head, self.head_dim)
            bg_gate = 2 * torch.sigmoid(self.bigram_gate(x[..., self.ve_gate_channels:2*self.ve_gate_channels]))
            v = v + bg_gate.unsqueeze(-1) * bigram_ve

        # Trigram VE with its own gate reading from channels 64:96
        if trigram_ve is not None:
            trigram_ve = trigram_ve.view(B, T, self.n_kv_head, self.head_dim)
            tg_gate = 2 * torch.sigmoid(self.trigram_gate(x[..., 2*self.ve_gate_channels:3*self.ve_gate_channels]))
            v = v + tg_gate.unsqueeze(-1) * trigram_ve

        # 4-gram VE with its own gate reading from channels 96:128 (only when enabled)
        if fourgram_ve is not None:
            fourgram_ve = fourgram_ve.view(B, T, self.n_kv_head, self.head_dim)
            fg_gate = 2 * torch.sigmoid(self.fourgram_gate(x[..., 3*self.ve_gate_channels:4*self.ve_gate_channels]))
            v = v + fg_gate.unsqueeze(-1) * fourgram_ve

        # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
        # 5-gram VE with its own gate reading channels 160:192 (slice 5 -- slice 4
        # = 128:160 belongs to the PK-memory gate below, so no overlap when both
        # are enabled). fivegram_ve is None when disabled -> no ops traced
        # (byte-identical baseline).
        if fivegram_ve is not None:
            fivegram_ve = fivegram_ve.view(B, T, self.n_kv_head, self.head_dim)
            vg_gate = 2 * torch.sigmoid(self.fivegram_gate(x[..., 5*self.ve_gate_channels:6*self.ve_gate_channels]))
            v = v + vg_gate.unsqueeze(-1) * fivegram_ve
        # ===== EXPLORE:NGRAM_FIVEGRAM END =====

        # ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
        # Product-key memory contribution, injected into the attention VALUE path
        # alongside the n-gram VEs through its own input-dependent per-KV-head gate
        # (channels 128:160). pk_ve is None when the flag is off (no ops traced).
        # Zero-init note: the gate weight zero-init makes the gate 1.0 (neutral,
        # same convention as the other VE gates); the EXACT step-0 identity comes
        # from the zero-initialized PK value table, which makes pk_ve exactly 0.
        if pk_ve is not None:
            pk_ve = pk_ve.view(B, T, self.n_kv_head, self.head_dim)
            pk_gate = 2 * torch.sigmoid(self.pk_gate(x[..., 4*self.ve_gate_channels:5*self.ve_gate_channels]))
            v = v + pk_gate.unsqueeze(-1) * pk_ve
        # ===== EXPLORE:NGRAM_PK_MEMORY END =====

        cos, sin = cos_sin
        # QK-norm refinement: normalize BEFORE rotary instead of after
        q, k = norm(q), norm(k)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if OBSERVE_LAYER_PROBES:
            prefix = f"layer_{self.layer_idx}"
            OBS.add_norms(f"{prefix}.q", q)
            OBS.add_norms(f"{prefix}.k", k)
            OBS.add_norms(f"{prefix}.v", v)
            observe_attention_distribution(prefix, q, k, window_size)

        # Replaces the previous direct flash_attn_func call; default SDPA is less
        # optimized but avoids the H200 native-extension segfault path.
        # ===== EXPLORE:DOC_MASK BEGIN =====
        # When DOC_MASK=1 a prebuilt mask for this layer's window is threaded down
        # and (SDPA path only) attention is restricted to same-document keys. When
        # off, doc_mask is None and the original is_causal=True call below runs
        # unchanged (byte-identical).
        if DOC_MASK and doc_mask is not None and ATTN_BACKEND == "sdpa":
            y = _doc_masked_attention(q, k, v, doc_mask)
        elif DOC_MASK and DOC_MASK_IMPL == "varlen" and ATTN_BACKEND == "fa3" and doc_mask is not None:
            # ===== RSI INTERVENTION: FA3 VARLEN DOC-MASK =====
            # doc_mask carries (cu_seqlens, max_seqlen) in this mode, not a boolean
            # mask: the document structure is expressed to the kernel directly rather
            # than materialized. Keeps fa3 speed AND the SSSL window (passed through).
            _cu, _maxs = doc_mask
            y = fa3_varlen_attn(q, k, v, _cu, _maxs, window_size=window_size)
        else:
            y = attention_func(q, k, v, causal=True, window_size=window_size)
        # ===== EXPLORE:DOC_MASK END =====
        # Per-head RMSNorm on attention output (DiffTransformer-inspired sub-layer normalization)
        y = norm(y)

        # Head-level MoE: per-head routing gate on all layers
        head_gates = 2.0 * torch.sigmoid(self.head_gate(x[..., :self.ve_gate_channels]))
        if OBSERVE_LAYER_PROBES:
            OBS.add(f"layer_{self.layer_idx}.head_gate_mean", head_gates)
        y = y * head_gates.unsqueeze(-1)

        y = y.contiguous().view(B, T, -1)
        if OBSERVE_LAYER_PROBES:
            OBS.add_norms(f"layer_{self.layer_idx}.attn_out", y)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        # RSI MLP LEVER: env-gated SwiGLU vs squared-ReLU (default). SwiGLU hidden
        # h=8d/3 is exactly param-matched to the squared-ReLU 4d hidden (both 8*d^2).
        self.mlp_type = MLP_TYPE
        if self.mlp_type == "swiglu":
            d = config.n_embd
            h = round(8 * d / 3)
            self.c_fc = nn.Linear(d, h, bias=False)      # value branch
            self.c_gate = nn.Linear(d, h, bias=False)    # gate branch (SiLU)
            self.c_proj = nn.Linear(h, d, bias=False)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
            # Uniform tau=0.5: confirmed optimal threshold
            self.tau = 0.5

    def forward(self, x):
        if self.mlp_type == "swiglu":
            h_act = F.silu(self.c_gate(x)) * self.c_fc(x)
            h = self.c_proj(h_act)
            if OBSERVE_LAYER_PROBES:
                OBS.add_norms(f"layer_{self.layer_idx}.mlp_act", h_act)
                OBS.add_norms(f"layer_{self.layer_idx}.mlp_out", h)
            return h
        h_pre = self.c_fc(x)
        h_act = F.relu(h_pre - self.tau).square()
        h = self.c_proj(h_act)
        if OBSERVE_LAYER_PROBES:
            OBS.add_norms(f"layer_{self.layer_idx}.mlp_pre", h_pre)
            OBS.add_norms(f"layer_{self.layer_idx}.mlp_act", h_act)
            OBS.add_norms(f"layer_{self.layer_idx}.mlp_out", h)
        return h


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.config_gpas = bool(config.gpas)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)
        # Exact GPAS source contract: a single scalar is shared by the two
        # post-sublayer applications in this layer.  torch.zeros consumes no RNG.
        # No attribute is created on the disabled path.
        if config.gpas:
            self.gpas_alpha = nn.Parameter(torch.zeros(()))
        if CANON:
            self.canon_w = nn.Parameter(torch.zeros(config.n_embd, CANON_K))  # EXPLORE:CANON per-channel causal taps

    def forward(self, x, ve, cos_sin, window_size, bigram_ve=None, trigram_ve=None, fourgram_ve=None, doc_mask=None, pk_ve=None, fivegram_ve=None):
        # ===== EXPLORE:CANON (fused shifted-weighted-sum causal depthwise conv; zero-init -> byte-identical off) =====
        if CANON:
            _cx = norm(x)
            _acc = _cx * self.canon_w[:, 0]
            for _k in range(1, CANON_K):
                _acc = _acc + torch.nn.functional.pad(_cx, (0, 0, _k, 0))[:, :_cx.shape[1]] * self.canon_w[:, _k]
            x = x + _acc
        # Simplified attention residual: per-head norm + head gate inside CSA already sufficient
        x = x + self.attn(norm(x), ve, cos_sin, window_size, bigram_ve=bigram_ve, trigram_ve=trigram_ve, fourgram_ve=fourgram_ve, doc_mask=doc_mask, pk_ve=pk_ve, fivegram_ve=fivegram_ve)
        if self.config_gpas:
            # GPAS, not an ordinary learnable residual scale: forward activation
            # is scaled while the hidden-state Jacobian remains the identity.
            x = x - F.silu(self.gpas_alpha) * x.detach()
        if OBSERVE_LAYER_PROBES:
            OBS.add(
                f"layer_{self.layer_idx}.post_attn_residual_variance",
                x.detach().double().var(dim=-1, correction=0).mean(),
            )
        x = x + norm(self.mlp(norm(x)))
        if self.config_gpas:
            # Reuse the SAME per-layer scalar after the MLP residual sum.
            x = x - F.silu(self.gpas_alpha) * x.detach()
        if OBSERVE_LAYER_PROBES:
            # Paper-020's authoritative mediator: mean over tokens of the
            # feature-dimension population variance, evaluated in FP64.  Layer
            # probes force eager diagnostic execution and are never charged.
            OBS.add(
                f"layer_{self.layer_idx}.post_mlp_residual_variance",
                x.detach().double().var(dim=-1, correction=0).mean(),
            )
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # JEPA MTP removed: multi-token prediction hurts step count in 5-min budget
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Input-dependent x0 gating: per-layer scale for sigmoid gate on x0 skip (layers 4+)
        # gate = 2*sigmoid(scale * x.mean(-1)) modulates x0_lambdas contribution
        # Zero-init so gate starts at 1.0 (neutral = same as current scalar behavior)
        self.x0_gate_scales = nn.Parameter(torch.zeros(config.n_layer))
        # ===== RSI INTERVENTION: ATTNRES BEGIN =====
        # Kimi-K3 Attention Residuals: a learned per-layer pseudo-query attends (softmax
        # over RMSNorm'd keys) across the token embedding and all preceding layer outputs,
        # added into the residual stream through a zero-init gate. Params are allocated
        # ONLY when the flag is on, so the OFF path has an identical parameter set (and
        # thus an identical optimizer partition and RNG consumption) to the pre-port code.
        if ATTNRES_ENABLE:
            self.attnres_queries = nn.Parameter(torch.zeros(config.n_layer, config.n_embd))
            self.attnres_gate = nn.Parameter(torch.zeros(config.n_layer))  # zero => no-op at step 0
        # ====== RSI INTERVENTION: ATTNRES END ======
        # ===== RSI INTERVENTION: NGPT_SPHERE BEGIN =====
        # Per-layer "eigen learning rate": how far along the sphere this layer moves the
        # residual stream toward its own output. Allocated only when the lever is on.
        if NGPT_SPHERE:
            self.ngpt_alpha = nn.Parameter(torch.full((config.n_layer,), NGPT_ALPHA_INIT))
        # ====== RSI INTERVENTION: NGPT_SPHERE END ======
        # Multi-layer output pooling: aggregate last-K intermediate layers as additive correction
        self.n_pool_layers = min(4, config.n_layer)  # layers [n-4, n-3, n-2] contribute (3 weights)
        self.layer_pool_weights = nn.Parameter(torch.zeros(self.n_pool_layers - 1))
        # Value embeddings (unigram)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        # Factored multi-hash bigram VE: K=2 half-dim tables concatenated per layer
        # Crossover: K=2 simplification recovers throughput
        ve_layers = sorted(i for i in range(config.n_layer) if has_ve(i, config.n_layer))
        self.bigram_ve_layers = set(ve_layers)
        # CROSSOVER B: 64x bigram tables (baseline). EXPLORE:NGRAM_TABLE_MULT scales
        # the hash-bucket count; more buckets -> fewer collisions -> more rare-ngram capacity.
        self.bigram_table_size = config.vocab_size * (config.ngram_bigram_mult or config.ngram_table_mult)
        self.bigram_K = 2
        half_kv_dim = kv_dim // 2
        # PER-LAYER DECORRELATED: completely disjoint hash prime pairs per bigram VE layer
        # Each layer uses entirely distinct multipliers -- zero prime reuse within bigram type
        # Constants from Murmur/FNV/golden-ratio family for good avalanche behavior
        _decorr_bigram_primes = [
            [(2654435761, 2246822519), (1013904223, 6291469)],   # layer 1: golden-ratio family
            [(374761393, 668265263), (3266489917, 104729)],      # layer 3: prime family
            [(1640531527, 97531), (48271, 40503)],               # layer 5: LCG/Knuth family
            [(16777619, 2166136261), (3432918353, 461845907)],   # layer 7: MurmurHash3 family
        ]
        self.bigram_hash_primes_per_layer = {}
        self.bigram_ves = nn.ModuleDict()
        for j, layer_i in enumerate(ve_layers):
            self.bigram_ves[str(layer_i)] = nn.ModuleList([
                nn.Embedding(self.bigram_table_size, half_kv_dim),
                nn.Embedding(self.bigram_table_size, half_kv_dim),
            ])
            self.bigram_hash_primes_per_layer[layer_i] = _decorr_bigram_primes[j]
        # Multi-layer factored trigram VE: K=2 half-dim tables at layers 1+5 plus layer 7.
        self.trigram_ve_layers = (
            {ve_layers[0], ve_layers[-2], ve_layers[-1]}
            if len(ve_layers) >= 2
            else {ve_layers[-1]}
        )
        self.trigram_table_size = config.vocab_size * (config.ngram_trigram_mult or config.ngram_table_mult)  # CROSSOVER B baseline 64x; EXPLORE:NGRAM_TABLE_MULT
        # PER-LAYER DECORRELATED: completely disjoint 6-prime tuples per trigram VE layer
        # Using disjoint constant families: each layer uses different multiplier sources
        _decorr_trigram_primes = [
            (16777619, 2166136261, 3432918353, 461845907, 2654435769, 1540483477),  # layer 1: FNV+Murmur family
            (3405403843, 2654435761, 2246822519, 1013904223, 6291469, 374761393),   # layer 5: golden-ratio family
            (668265263, 3266489917, 104729, 1640531527, 97531, 48271),              # layer 7: prime family
        ]
        self.trigram_hash_primes_per_layer = {}
        self.trigram_ves = nn.ModuleDict()
        # EXPLORE:SHARED_TRIGRAM_VE (vora's structural simplification) -- when on, all trigram
        # layers share ONE table + ONE prime family (weight-sharing: 3x fewer sparse-table params).
        # In the repeated-data regime this acts as regularization (less memorization). Default off
        # = independent per-layer tables (byte-identical baseline). Forward pass is unchanged.
        if getattr(config, "shared_trigram_ve", False) and self.trigram_ve_layers:
            _shared_tri = nn.ModuleList([
                nn.Embedding(self.trigram_table_size, half_kv_dim),
                nn.Embedding(self.trigram_table_size, half_kv_dim),
            ])
            for layer_i in sorted(self.trigram_ve_layers):
                self.trigram_ves[str(layer_i)] = _shared_tri  # same module -> weight-shared
                self.trigram_hash_primes_per_layer[layer_i] = _decorr_trigram_primes[0]
        else:
            for j, layer_i in enumerate(sorted(self.trigram_ve_layers)):
                self.trigram_ves[str(layer_i)] = nn.ModuleList([
                    nn.Embedding(self.trigram_table_size, half_kv_dim),
                    nn.Embedding(self.trigram_table_size, half_kv_dim),
                ])
                self.trigram_hash_primes_per_layer[layer_i] = _decorr_trigram_primes[j]
        # EXPLORE:NGRAM_FOURGRAM -- 4-gram VE order (OFF by default -> empty, byte-identical).
        # ===== RSI INTERVENTION: C3 BACKOFF-GATE COUNT BUFFERS BEGIN =====
        # Online per-key observation counts backing the frequency-confidence gate
        # count/(count+kappa). Non-persistent buffers (not checkpointed, not
        # parameters) so the optimizer partition and parameter count are unchanged.
        # Allocated ONLY when the gate is on: kappa=0 leaves the model byte-identical
        # to the pre-port baseline (no buffers, no gather, no scatter_add).
        if NGRAM_BACKOFF_KAPPA > 0:
            for _c3_layer in sorted(self.bigram_ve_layers):
                for _c3_k in range(self.bigram_K):
                    self.register_buffer(
                        f"ngram_count_bigram_{_c3_layer}_{_c3_k}",
                        torch.zeros(self.bigram_table_size, dtype=torch.float32),
                        persistent=False,
                    )
            for _c3_layer in sorted(self.trigram_ve_layers):
                for _c3_k in range(2):  # trigram tables are K=2 factored half-dim lookups
                    self.register_buffer(
                        f"ngram_count_trigram_{_c3_layer}_{_c3_k}",
                        torch.zeros(self.trigram_table_size, dtype=torch.float32),
                        persistent=False,
                    )
        # ====== RSI INTERVENTION: C3 BACKOFF-GATE COUNT BUFFERS END ======
        # Same layers/structure as trigram (K=2 half-dim factored tables), but each hash mixes
        # 4 token indices (prev3,prev2,prev,cur) via disjoint 8-prime tuples per layer.
        self.fourgram_ve_layers = set(self.trigram_ve_layers) if config.ngram_fourgram_mult > 0 else set()
        self.fourgram_table_size = config.vocab_size * max(config.ngram_fourgram_mult, 1)
        _decorr_fourgram_primes = [
            (2654435761, 2246822519, 1013904223, 6291469, 374761393, 668265263, 3266489917, 104729),   # layer 1
            (16777619, 2166136261, 3432918353, 461845907, 1540483477, 3405403843, 1640531527, 97531),  # layer 5
            (48271, 40503, 2654435769, 2246822519, 374761393, 668265263, 3266489917, 1013904223),      # layer 7
        ]
        self.fourgram_hash_primes_per_layer = {}
        self.fourgram_ves = nn.ModuleDict()
        for j, layer_i in enumerate(sorted(self.fourgram_ve_layers)):
            self.fourgram_ves[str(layer_i)] = nn.ModuleList([
                nn.Embedding(self.fourgram_table_size, half_kv_dim),
                nn.Embedding(self.fourgram_table_size, half_kv_dim),
            ])
            self.fourgram_hash_primes_per_layer[layer_i] = _decorr_fourgram_primes[j]
        # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
        # 5-gram VE order (OFF by default -> empty ModuleDict, no params, byte-identical).
        # Same layers/structure as trigram/fourgram (K=2 half-dim factored tables), but each
        # hash mixes 5 token indices (prev4,prev3,prev2,prev,cur) via PER-LAYER DECORRELATED
        # disjoint 10-prime tuples. Constants drawn from families NOT already used by the
        # bigram/trigram/fourgram hashes (verified zero overlap with all 19 existing
        # multipliers) so 5-gram collisions decorrelate from every lower order.
        self.fivegram_ve_layers = set(self.trigram_ve_layers) if getattr(config, "ngram_fivegram_mult", 0) > 0 else set()
        self.fivegram_table_size = config.vocab_size * max(getattr(config, "ngram_fivegram_mult", 0), 1)
        _decorr_fivegram_primes = [
            # layer 1: Murmur3-fmix + xxHash64-prime-low32 family
            (2246822507, 3266489909, 2246822535, 668265295, 2654435833, 3266489955, 374761413, 2135587861, 3210233709, 2496678331),
            # layer 5: SplitMix64-low32 + PCG + classic-LCG multiplier family
            (484763065, 321982955, 747796405, 2891336453, 1103515245, 22695477, 134775813, 214013, 69069, 1664525),
            # layer 7: large-prime family (Mersenne/nth-prime constants)
            (2147483647, 1073741789, 4294967291, 2971215073, 433494437, 15485863, 32452843, 49979687, 67867967, 86028121),
        ]
        self.fivegram_hash_primes_per_layer = {}
        self.fivegram_ves = nn.ModuleDict()
        for j, layer_i in enumerate(sorted(self.fivegram_ve_layers)):
            self.fivegram_ves[str(layer_i)] = nn.ModuleList([
                nn.Embedding(self.fivegram_table_size, half_kv_dim),
                nn.Embedding(self.fivegram_table_size, half_kv_dim),
            ])
            self.fivegram_hash_primes_per_layer[layer_i] = _decorr_fivegram_primes[j]
        # ===== EXPLORE:NGRAM_FIVEGRAM END =====
        # ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
        # Adaptive sparse-memory residual with LEARNED retrieval (product-key
        # memory, Lample et al. 2019). Rationale: brute-force RANDOM hash capacity
        # is played out (256x->384x measured -0.000611, 55% of prediction, below
        # the advance threshold; per-SF slope halved; DIAGNOSE showed the capacity
        # gain is broad-spectrum, not rare-specific). This memory is therefore NOT
        # "more buckets": learned keys route RELATED contexts to a SHARED slot,
        # which random hashing cannot do -- generalization across similar contexts
        # instead of more isolated cells. Per-layer SEPARATE memories on the
        # trigram layers (weight-sharing across layers measured +0.0016 = hurts).
        # Query = hidden state projection + local n-gram context embeddings
        # (reusing the prev2/prev/cur shifted-index tensors the VE hashes use).
        # Retrieval: split query into 2 sub-queries, top-k over each sqrt(N)-sized
        # sub-key codebook, cartesian top-k -> O(sqrt(N)) search over N slots.
        # Values: (N, half_kv_dim) table, softmax-weighted top-k sum, projected to
        # kv_dim and injected into the attention value path via a zero-init gate.
        # pk_mem=False (default): only plain Python attrs are set here -- no
        # modules registered, no param-count change, byte-identical baseline.
        self.pk_mem_layers = set(self.trigram_ve_layers) if getattr(config, "pk_mem", False) else set()
        self.pk_mem_n_sub = config.pk_mem_subkeys
        self.pk_mem_n_slots = config.pk_mem_subkeys ** 2
        self.pk_mem_topk = config.pk_mem_topk
        self.pk_mem_qdim = config.pk_mem_qdim
        if self.pk_mem_layers:
            assert config.pk_mem_qdim % 2 == 0, "pk_mem_qdim must be even (split into 2 sub-queries)"
            assert config.n_embd >= 5 * 32, "pk gate reads channels 128:160"
            assert config.pk_mem_topk <= config.pk_mem_subkeys
            self.pk_mems = nn.ModuleDict()
            for layer_i in sorted(self.pk_mem_layers):
                self.pk_mems[str(layer_i)] = nn.ModuleDict({
                    # hidden-state query projection (Muon group)
                    "q_proj": nn.Linear(config.n_embd, config.pk_mem_qdim, bias=False),
                    # local n-gram context query embeddings (AdamW group);
                    # indexed by the SAME prev2/prev/cur tensors as the VE hashes
                    "ctx_prev2": nn.Embedding(config.vocab_size, config.pk_mem_qdim),
                    "ctx_prev": nn.Embedding(config.vocab_size, config.pk_mem_qdim),
                    "ctx_cur": nn.Embedding(config.vocab_size, config.pk_mem_qdim),
                    # product-key sub-codebooks, each sqrt(N) x (qdim/2)
                    "keys1": nn.Embedding(config.pk_mem_subkeys, config.pk_mem_qdim // 2),
                    "keys2": nn.Embedding(config.pk_mem_subkeys, config.pk_mem_qdim // 2),
                    # value memory: ZERO-INIT -> exact step-0 identity with SOTA
                    # NOTE: key must NOT be "values" -- nn.ModuleDict reserves dict method names.
                    "val_table": nn.Embedding(self.pk_mem_n_slots, half_kv_dim),
                    # half_kv_dim -> kv_dim up-projection (random init; exactly one
                    # of {values, out_proj} may be zero or both freeze forever)
                    "out_proj": nn.Linear(half_kv_dim, kv_dim, bias=False),
                })
        # ===== EXPLORE:NGRAM_PK_MEMORY END =====
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=LM_HEAD_INIT_STD)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            if hasattr(block.mlp, "c_gate"):  # SwiGLU gate branch, same init as value
                torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            # Model construction happens on meta and to_empty() does not preserve
            # initializer values.  Re-establish the exact source gate value without
            # consuming RNG after materialization.
            if hasattr(block, "gpas_alpha"):
                torch.nn.init.zeros_(block.gpas_alpha)
        # Per-layer scalars
        # ===== EXPLORE:RESID_SCALE BEGIN =====
        # Depth-muP-style residual init: scale resid_lambdas by 1/sqrt(2*n_layer)
        # instead of 1.0 when enabled. Only the INITIAL value changes; the parameter
        # remains learnable. When off, resid_lambdas.fill_(1.0) runs (byte-identical).
        if RESID_SCALE:
            self.resid_lambdas.fill_(1.0 / (2 * self.config.n_layer) ** 0.5)
        else:
            self.resid_lambdas.fill_(1.0)
        # ===== EXPLORE:RESID_SCALE END =====
        self.x0_lambdas.fill_(0.1)
        self.x0_gate_scales.fill_(0.0)  # Zero-init: sigmoid(0)=0.5, 2*0.5=1.0 = neutral gate
        self.layer_pool_weights.fill_(0.0)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.bigram_gate is not None:
                torch.nn.init.zeros_(block.attn.bigram_gate.weight)
            if block.attn.trigram_gate is not None:
                torch.nn.init.zeros_(block.attn.trigram_gate.weight)
            # BUGFIX: fourgram_gate was created (CausalSelfAttention.__init__) and used
            # in forward but never initialized here. Because the model is built on meta
            # and materialized with to_empty(), an uninitialized gate holds GARBAGE
            # memory whenever NGRAM_FOURGRAM_MULT>0 -- the 4-gram VE was being injected
            # through a random gate. Byte-identical for every default run (fourgram off
            # -> fourgram_gate is None), but it invalidates prior 4-gram results.
            if getattr(block.attn, "fourgram_gate", None) is not None:
                torch.nn.init.zeros_(block.attn.fourgram_gate.weight)
            # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
            # Zero-init the 5-gram gate, guarded by a None check, exactly like the
            # ve/bigram/trigram/fourgram gates above. DO NOT remove: the model is
            # built on meta and materialized with to_empty(), so a created-but-
            # uninitialized gate holds GARBAGE memory -- the exact bug that
            # falsified an entire 4-gram campaign (see BUGFIX note above).
            if getattr(block.attn, "fivegram_gate", None) is not None:
                torch.nn.init.zeros_(block.attn.fivegram_gate.weight)
            # ===== EXPLORE:NGRAM_FIVEGRAM END =====
            torch.nn.init.zeros_(block.attn.head_gate.weight)
        # Bigram VE: same init as regular VE (factored: two half-dim tables per layer)
        for layer_ves in self.bigram_ves.values():
            for bve in layer_ves:
                torch.nn.init.uniform_(bve.weight, -s, s)
                bve.to(dtype=torch.bfloat16)
        # Trigram VE init (factored: two half-dim tables per layer)
        for layer_tves in self.trigram_ves.values():
            for tve in layer_tves:
                torch.nn.init.uniform_(tve.weight, -s, s)
                tve.to(dtype=torch.bfloat16)
        # 4-gram VE init (same as trigram; empty when disabled)
        for layer_fves in self.fourgram_ves.values():
            for fve in layer_fves:
                torch.nn.init.uniform_(fve.weight, -s, s)
                fve.to(dtype=torch.bfloat16)
        # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
        # 5-gram VE init (same as trigram/fourgram). When disabled the ModuleDict is
        # empty -> the loop body never runs: ZERO RNG draws, global RNG stream and
        # every downstream init untouched (byte-identical baseline).
        for layer_gves in self.fivegram_ves.values():
            for gve in layer_gves:
                torch.nn.init.uniform_(gve.weight, -s, s)
                gve.to(dtype=torch.bfloat16)
        # ===== EXPLORE:NGRAM_FIVEGRAM END =====
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        # ===== EXPLORE:TIE_EMBED BEGIN =====
        # Weight-tie the unembedding to the input embedding. Done LAST so it fully
        # overrides the separate lm_head init above (std 0.001) by replacing the
        # lm_head weight with the (now bf16) wte weight tensor. When off, nothing
        # here runs and lm_head keeps its own init (byte-identical to baseline).
        # NOTE: tying interacts with the logit softcap (16.5*tanh(logits/15.0)).
        # lm_head and wte now share one tensor; setup_optimizer places that shared
        # tensor in the embedding group ONLY (the unembedding group is emptied and
        # dropped), so the tied weight trains at embedding_lr, not unembedding_lr.
        if TIE_EMBED:
            self.lm_head.weight = self.transformer.wte.weight
        # ===== EXPLORE:TIE_EMBED END =====
        # ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
        # PK-memory init. Placed LAST and drawn from a DEDICATED generator so the
        # GLOBAL RNG stream is untouched: with the flag on, every baseline
        # parameter (and any later global-RNG consumer, e.g. data shuffling)
        # receives exactly the same values as with the flag off.
        # Step-0 identity guarantee (flag ON == SOTA numerically at step 0):
        #   - values table is ZERO-init -> retrieval output = out_proj(0) = 0,
        #     so v = v + gate * 0 is bitwise unchanged;
        #   - pk_gate weight zero-init -> gate = 2*sigmoid(0) = 1.0 (neutral,
        #     the same convention as the other VE gates);
        #   - gradient flow at step 0: values receive nonzero grad (through the
        #     random out_proj and neutral gate) and start learning immediately;
        #     keys/query/ctx/out_proj/gate grads are all zero until values move
        #     (documented one-step bootstrap lag, standard for zero-init memory).
        if self.pk_mem_layers:
            g = torch.Generator(device=self.transformer.wte.weight.device)
            g.manual_seed(SEED)
            dh = self.pk_mem_qdim // 2
            s_key = 3**0.5 * dh**-0.5          # fan-in of a sub-key dot product
            s_out = 3**0.5 * (self.config.n_kv_head * self.config.n_embd // self.config.n_head // 2) ** -0.5
            for m in self.pk_mems.values():
                m["q_proj"].weight.uniform_(-s, s, generator=g)
                m["ctx_prev2"].weight.uniform_(-s, s, generator=g)
                m["ctx_prev"].weight.uniform_(-s, s, generator=g)
                m["ctx_cur"].weight.uniform_(-s, s, generator=g)
                m["keys1"].weight.uniform_(-s_key, s_key, generator=g)
                m["keys2"].weight.uniform_(-s_key, s_key, generator=g)
                torch.nn.init.zeros_(m["val_table"].weight)
                m["out_proj"].weight.uniform_(-s_out, s_out, generator=g)
                m["val_table"].to(dtype=torch.bfloat16)
            for block in self.transformer.h:
                if getattr(block.attn, "pk_gate", None) is not None:
                    torch.nn.init.zeros_(block.attn.pk_gate.weight)
        # ===== EXPLORE:NGRAM_PK_MEMORY END =====

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=1000000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SLT" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        tiny_window = long_window // 4
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0), "T": (tiny_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Lower-bound training FLOPs/token for dense linears + attention.

        Stored embedding rows are capacity/VRAM, not dense per-token compute.
        Counting every row of the 64x/256x n-gram tables as a matrix multiply
        made MFU grow with table capacity even though each token gathers only a
        handful of rows. Count each registered ``nn.Linear`` weight once
        instead. Explicit attention matmuls are added below. Sparse gather,
        hashing, top-k routing, masking, and elementwise kernels are omitted, so
        this remains telemetry—not a FLOP-equivalence proof.

        A tied input/output embedding is still counted once through ``lm_head``:
        the shared tensor participates in a dense vocabulary projection there.
        """
        linear_weights = {
            id(module.weight): module.weight
            for module in self.modules()
            if isinstance(module, nn.Linear)
        }
        dense_linear_params = sum(weight.numel() for weight in linear_weights.values())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        # The plain SDPA wrapper above ignores window_size and runs full causal
        # attention. Only the explicit doc-mask path applies the configured
        # window, and only in "both"/"window" mode. Reflect executed kernels
        # rather than the nominal TTTL configuration in telemetry.
        executes_window = ATTN_BACKEND != "sdpa" or (
            DOC_MASK and DOC_MASK_MODE in ("both", "window")
        )
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = (
                t
                if not executes_window or window < 0
                else min(window, t)
            )
            attn_flops += 12 * h * q * effective_seq
        return 6 * dense_linear_params + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.layer_pool_weights.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        ngram_ve_betas=None,  # if None, uses adam_betas
        ngram_ve_lr_scale=1.0,  # discriminative LR scale for n-gram VE (ULMFiT-inspired)
    ):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        _canon_params = []
        if CANON:
            _canon_params = [b.canon_w for b in self.transformer.h if hasattr(b, "canon_w")]
            _cids = {id(p) for p in _canon_params}
            matrix_params = [p for p in matrix_params if id(p) not in _cids]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        # ===== EXPLORE:TIE_EMBED BEGIN =====
        # When the unembedding is weight-tied (init_weights sets lm_head.weight to
        # wte.weight) the two are the SAME tensor. PyTorch forbids one parameter in
        # two optimizer groups, and self.parameters() dedups it -- so the shared
        # tensor must live in exactly ONE group. Keep it in the embedding group and
        # empty the unembedding group (filtered out below before optimizer build).
        # Off by default: lm_head_params is unchanged, so this is byte-identical to
        # the baseline. NOTE: this makes the tied weight train at embedding_lr only
        # (the separate unembedding_lr no longer applies once tied).
        if TIE_EMBED and self.lm_head.weight is self.transformer.wte.weight:
            lm_head_params = []
        # ===== EXPLORE:TIE_EMBED END =====
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas, self.x0_gate_scales]  # gate scales grouped with x0 lambdas
        # ===== RSI INTERVENTION: ATTNRES BEGIN =====
        # Empty when the flag is off, so the group is filtered out below and the
        # optimizer partition is byte-identical to the pre-port baseline.
        attnres_params = [self.attnres_queries, self.attnres_gate] if ATTNRES_ENABLE else []
        # ===== RSI INTERVENTION: TOKEN_SHIFT BEGIN =====
        # 1-D per-dimension scalars: they must NOT reach the Muon groups (which assume
        # 2-D matrices). Collected here and routed to the scalar AdamW group below;
        # empty list when off, so the group set and assert arithmetic are unchanged.
        # ===== RSI INTERVENTION: NGPT_SPHERE (scalar group; empty when off) =====
        ngpt_params = [self.ngpt_alpha] if NGPT_SPHERE else []
        # GPAS scalars live inside transformer.h and would otherwise enter a Muon
        # shape group.  Route them to a dedicated, schedule-exempt AdamW group.
        # Empty when disabled, leaving the pre-existing partition unchanged.
        gpas_params = [
            block.gpas_alpha
            for block in self.transformer.h
            if hasattr(block, "gpas_alpha")
        ]
        if gpas_params:
            _gpas_ids = {id(p) for p in gpas_params}
            matrix_params = [p for p in matrix_params if id(p) not in _gpas_ids]
        tshift_params = []
        if TOKEN_SHIFT:
            for _blk in self.transformer.h:
                tshift_params.extend([_blk.attn.tshift_q, _blk.attn.tshift_k])
            # These live inside transformer.h, so they are already in matrix_params and
            # would otherwise be handed to Muon, which assumes 2-D matrices. Remove them
            # exactly as the CANON lever does for its own non-2-D parameter.
            _tsids = {id(p) for p in tshift_params}
            matrix_params = [p for p in matrix_params if id(p) not in _tsids]
        # ====== RSI INTERVENTION: TOKEN_SHIFT END ======
        # ====== RSI INTERVENTION: ATTNRES END ======
        bigram_ve_params = list(self.bigram_ves.parameters())
        trigram_ve_params = list(self.trigram_ves.parameters())
        fourgram_ve_params = list(self.fourgram_ves.parameters())  # empty unless 4-gram enabled
        # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
        fivegram_ve_params = list(self.fivegram_ves.parameters())  # empty unless 5-gram enabled
        # ===== EXPLORE:NGRAM_FIVEGRAM END =====
        pool_params = [self.layer_pool_weights]
        # ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
        # Optimizer-group split for the PK memory (all lists empty when off ->
        # assert arithmetic and group set are byte-identical to baseline):
        #   - values      -> RMSProp is_ngram_ve group: sparsely-touched embedding
        #     rows, exactly like the hash VE tables; inherits NGRAM_STATE_ROWWISE
        #     (rows,1) fp32 state and the delayed beta2 warmdown. It does NOT take
        #     the NGRAM_SPARSE_GRAD capture path (only registered hash sites do);
        #     it falls through to the dense fused RMSProp kernel, so both flags
        #     compose safely. LR scaled by PK_MEM_LR_SCALE.
        #   - ctx embeds + sub-keys -> AdamW at embedding LR (dense-ish, row-
        #     indexed lookups; Muon's orthogonalized update is wrong for
        #     embedding-like tables). Also scaled by PK_MEM_LR_SCALE.
        #   - q_proj / out_proj -> appended to matrix_params: proper dense
        #     matmuls, Muon-optimized like every other projection.
        #   - pk_gate lives inside transformer.h -> already in matrix_params
        #     (Muon), identical treatment to the other VE gates.
        pk_value_params = []
        pk_dense_params = []
        if getattr(self, "pk_mem_layers", None):
            for m in self.pk_mems.values():
                pk_value_params.append(m["val_table"].weight)
                pk_dense_params += [
                    m["ctx_prev2"].weight,
                    m["ctx_prev"].weight,
                    m["ctx_cur"].weight,
                    m["keys1"].weight,
                    m["keys2"].weight,
                ]
                matrix_params += [m["q_proj"].weight, m["out_proj"].weight]
        # ===== EXPLORE:NGRAM_PK_MEMORY END =====
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embeds_params)
            + len(resid_params)
            + len(x0_params)
            + len(bigram_ve_params)
            + len(trigram_ve_params)
            + len(fourgram_ve_params)
            # ===== EXPLORE:NGRAM_FIVEGRAM (0 when off -> assert arithmetic unchanged) =====
            + len(fivegram_ve_params)
            + len(pk_value_params)
            + len(pk_dense_params)
            + len(pool_params)
            + len(_canon_params)
            # ===== RSI INTERVENTION: ATTNRES (0 when off -> assert arithmetic unchanged) =====
            + len(attnres_params)
            # ===== RSI INTERVENTION: TOKEN_SHIFT (0 when off) =====
            + len(tshift_params)
            # ===== RSI INTERVENTION: NGPT_SPHERE (0 when off) =====
            + len(ngpt_params)
            # GPAS: one scalar per layer when enabled, zero otherwise.
            + len(gpas_params)
        )
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if ngram_ve_betas is None:
            ngram_ve_betas = adam_betas
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            {
                "kind": "adamw",
                "params": lm_head_params,
                "lr": unembedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
                "demon_beta1": True,  # Apply Demon beta1 scheduling
            },
            {
                "kind": "adamw",
                "params": embedding_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
                "demon_beta1": True,
            },
            {
                "kind": "adamw",
                "params": value_embeds_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
                "demon_beta1": True,
            },
            {
                "kind": "adamw",
                "params": resid_params,
                "lr": scalar_lr * 0.01,
                "betas": adam_betas,
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
                # No demon_beta1: scalar params keep fixed beta1
            },
            # ===== RSI INTERVENTION: ATTNRES BEGIN =====
            # Empty params list when off; the empty-group filter below drops it, so the
            # optimizer sees exactly the pre-port group set. LR matches the reference
            # fork (scalar_lr, plain adam_betas, no weight decay).
            {
                "kind": "adamw",
                "params": attnres_params,
                "lr": scalar_lr,
                "betas": adam_betas,
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
            },
            # ====== RSI INTERVENTION: ATTNRES END ======
            # ===== RSI INTERVENTION: TOKEN_SHIFT BEGIN =====
            # Scalar-style per-dimension mix coefficients -> scalar_lr AdamW, no decay.
            # Empty when off; the empty-group filter drops it.
            {
                "kind": "adamw",
                "params": tshift_params,
                "lr": scalar_lr,
                "betas": adam_betas,
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
            },
            # ====== RSI INTERVENTION: TOKEN_SHIFT END ======
            # ===== RSI INTERVENTION: NGPT_SPHERE BEGIN =====
            {
                "kind": "adamw",
                "params": ngpt_params,
                "lr": scalar_lr,
                "betas": adam_betas,
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
            },
            # ====== RSI INTERVENTION: NGPT_SPHERE END ======
            {
                # Frozen Paper-020 mapping.  `is_gpas` excludes this group from
                # the global Adam warmdown and Demon beta1 bookkeeping below.
                "kind": "adamw",
                "params": gpas_params,
                "lr": 0.005,
                "betas": (0.8, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.0,
                "is_gpas": True,
            },
            {
                "kind": "adamw",
                "params": x0_params,
                "lr": scalar_lr,
                "betas": (0.96, 0.95),
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.002,  # x0WD=0.002 (proven optimal)
                "is_x0_muon_warmdown": True,  # x0 Muon warmdown
            },
            {
                "kind": "rmsprop",
                "params": bigram_ve_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale,
                "beta2": ngram_ve_betas[1],
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                # ===== RSI INTERVENTION: C1 n-gram-table-only decoupled weight decay =====
                # 0.0 by default == the pre-port value, so OFF is byte-identical.
                "weight_decay": NGRAM_WD_LAMBDA,
                "is_ngram_ve": True,
            },
            {
                "kind": "rmsprop",
                "params": trigram_ve_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale,
                "beta2": ngram_ve_betas[1],
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                # ===== RSI INTERVENTION: C1 n-gram-table-only decoupled weight decay =====
                "weight_decay": NGRAM_WD_LAMBDA,
                "is_ngram_ve": True,
            },
            {
                # 4-gram VE group (same RMSProp/ngram-VE treatment as bi/trigram).
                # Empty when 4-gram is disabled -> dropped by the empty-group filter below.
                "kind": "rmsprop",
                "params": fourgram_ve_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale,
                "beta2": ngram_ve_betas[1],
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
                "is_ngram_ve": True,
            },
            # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
            {
                # 5-gram VE group (same RMSProp/is_ngram_ve treatment as bi/tri/fourgram,
                # so it inherits NGRAM_STATE_ROWWISE and NGRAM_SPARSE_GRAD).
                # Empty when 5-gram is disabled -> dropped by the empty-group filter below.
                "kind": "rmsprop",
                "params": fivegram_ve_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale,
                "beta2": ngram_ve_betas[1],
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
                "is_ngram_ve": True,
            },
            # ===== EXPLORE:NGRAM_FIVEGRAM END =====
            # ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
            # Both groups are EMPTY when the flag is off and are dropped by the
            # empty-group filter below (baseline optimizer unchanged).
            {
                # PK value memory: RMSProp, tagged is_ngram_ve (see rationale above).
                "kind": "rmsprop",
                "params": pk_value_params,
                "lr": embedding_lr * dmodel_lr_scale * ngram_ve_lr_scale * PK_MEM_LR_SCALE,
                "beta2": ngram_ve_betas[1],
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
                "is_ngram_ve": True,
            },
            {
                # PK retrieval params (ctx embeds + sub-keys): AdamW at embedding
                # LR with Demon beta1, matching the value_embeds group treatment.
                "kind": "adamw",
                "params": pk_dense_params,
                "lr": embedding_lr * dmodel_lr_scale * PK_MEM_LR_SCALE,
                "betas": adam_betas,
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
                "demon_beta1": True,
            },
            # ===== EXPLORE:NGRAM_PK_MEMORY END =====
            {
                "kind": "adamw",
                "params": pool_params,
                "lr": scalar_lr * 0.15,  # revert to formula (0.75*0.15=0.1125)
                "betas": (0.96, 0.95),
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")),
                "weight_decay": 0.0,
            },
        ]
        if _canon_params:
            param_groups.append({"kind": "adamw", "params": _canon_params,
                "lr": float(os.environ.get("CANON_LR", "0.02")) * dmodel_lr_scale, "betas": adam_betas,
                "eps": float(os.environ.get("NGRAM_VE_EPS", "1e-10")), "weight_decay": 0.0})
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                {
                    "kind": "muon",
                    "params": group_params,
                    "lr": matrix_lr,
                    "momentum": 0.95,
                    "ns_steps": int(os.environ.get("MUON_NS_STEPS", "5")),  # EXPLORE: Newton-Schulz count (throughput)
                    "beta2": 0.95,
                    "weight_decay": weight_decay,
                }
            )
        # Drop any group that ended up with no parameters (e.g. the unembedding
        # group when TIE_EMBED shares its weight with the embedding group); an empty
        # param group is both meaningless and rejected by torch.optim. In the
        # default (untied) config no group is empty, so this is a no-op.
        param_groups = [g for g in param_groups if g["params"]]
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    # ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
    def _pk_mem_lookup(self, layer_i, x, prev2_idx, prev_idx, idx):
        """Product-key memory retrieval: O(sqrt(N)) search over N = n_sub^2 slots.

        Query = q_proj(norm(x)) + ctx embeddings of (prev2, prev, cur) token ids
        (the SAME shifted index tensors the n-gram VE hashes already computed --
        nothing is recomputed). The query splits into two RMS-normed sub-queries;
        each scores its sqrt(N)-sized sub-key codebook, takes top-k, and the
        cartesian top-k of the score sums selects k of the N = n_sub^2 slots
        (slot id = i1 * n_sub + i2; (i1, i2) pairs are distinct so no slot is
        selected twice). Values are softmax-weighted (fp32 softmax over k) and
        summed, then up-projected half_kv_dim -> kv_dim. All shapes are static
        (k, n_sub are Python constants), so torch.compile(fullgraph=True) traces
        this without graph breaks.
        """
        m = self.pk_mems[str(layer_i)]
        q = (
            m["q_proj"](norm(x))
            + m["ctx_prev2"](prev2_idx)
            + m["ctx_prev"](prev_idx)
            + m["ctx_cur"](idx)
        )
        dh = self.pk_mem_qdim // 2
        # RMS-norm each sub-query (stands in for Lample et al.'s query batchnorm:
        # keeps score scale bounded without batch statistics).
        q1, q2 = norm(q[..., :dh]), norm(q[..., dh:])
        s1 = q1 @ m["keys1"].weight.t()  # (B, T, n_sub)
        s2 = q2 @ m["keys2"].weight.t()  # (B, T, n_sub)
        k = self.pk_mem_topk
        v1, i1 = s1.topk(k, dim=-1)
        v2, i2 = s2.topk(k, dim=-1)
        cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)  # (B, T, k*k)
        scores, ci = cand.topk(k, dim=-1)
        slots = torch.gather(i1, -1, ci // k) * self.pk_mem_n_sub + torch.gather(i2, -1, ci % k)
        w = torch.softmax(scores.float(), dim=-1).to(x.dtype)  # (B, T, k)
        vals = m["val_table"](slots)  # (B, T, k, half_kv_dim)
        out = (w.unsqueeze(-1) * vals).sum(dim=-2)  # (B, T, half_kv_dim)
        return m["out_proj"](out)  # (B, T, kv_dim)
    # ===== EXPLORE:NGRAM_PK_MEMORY END =====

    def forward(self, idx, targets=None, reduction="mean", doc_masks=None):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        # ===== EXPLORE:RESET_ROPE BEGIN =====
        # Per-document RoPE position reset: rotary positions restart at 0 at each
        # BOS token (position = offset from last document start, computed via
        # cummax over BOS indices). Tokens before the first BOS in a row keep
        # their row-local arange, matching baseline for that head fragment.
        # self.cos/self.sin are (1, L, 1, hd/2); indexing dim 1 with per-row
        # positions yields (B, T, 1, hd/2), which broadcasts in apply_rotary_emb
        # against (B, T, H, hd/2) exactly like the shared slice does, so no
        # attention-layer plumbing changes. When off, the shared cos_sin slice
        # above is used unchanged (byte-identical baseline path).
        if RESET_ROPE:
            assert DOC_MASK_BOS_ID is not None, "RESET_ROPE=1 requires a resolved BOS id"
            ar = torch.arange(T, device=idx.device)
            last_bos = torch.cummax(ar * (idx == DOC_MASK_BOS_ID).long(), dim=1).values
            pos = ar - last_bos  # (B, T) offset from last document start
            cos_sin = self.cos[0][pos], self.sin[0][pos]
        # ===== EXPLORE:RESET_ROPE END =====

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        # ===== RSI INTERVENTION: ATTNRES BEGIN =====
        # Growing retrieval stack of x0 plus every preceding layer output.
        #
        # IMPLEMENTATION NOTE (this is a real bug fix, not a style choice): the
        # reference fork (repo/train_attnres_c1_v2.py) preallocated (L+1,B,T,d)
        # buffers and wrote each layer's output in-place. That fails under autograd
        # -- the slice taken at layer i is a view whose version counter is bumped by
        # the layer i+1 write, giving "one of the variables needed for gradient
        # computation has been modified by an inplace operation" at backward. This
        # matches the AttnRes crash already recorded in paper 013's negative results.
        # Appending to a list and stacking at use time keeps every tensor immutable,
        # so the autograd graph stays valid. torch.stack copies, but the stack is at
        # most (L+1) x (B,T,d) and only materializes on layers that actually read it.
        if ATTNRES_ENABLE:
            _ar_raw = [x0]
            _ar_key = [norm(x0)]
        # ====== RSI INTERVENTION: ATTNRES END ======
        # ===== EXPLORE:DOC_MASK BEGIN =====
        # Per-forward document masks for block-diagonal attention, built once and
        # shared by every layer with the same left window (in DOC_MASK_MODE=seg,
        # one mask for the whole stack) instead of re-materializing a (B, 1, T, T)
        # boolean per layer. Only computed when DOC_MASK=1; otherwise doc_masks
        # stays None and no overhead is incurred (baseline path is byte-identical).
        # A caller may pass prebuilt masks (required for DOC_MASK_IMPL=flex, whose
        # BlockMask cannot be constructed inside a fullgraph=True region).
        if DOC_MASK and doc_masks is None:
            assert DOC_MASK_IMPL != "flex", (
                "DOC_MASK_IMPL=flex requires masks built outside the compiled "
                "region; call build_doc_masks_for_batch(idx, window_lefts) and "
                "pass doc_masks=..."
            )
            seg = (idx == DOC_MASK_BOS_ID).cumsum(dim=1)
            doc_masks = {}
            for window_size in self.window_sizes:
                key = _doc_mask_key(window_size[0])
                if key not in doc_masks:
                    doc_masks[key] = _build_dense_doc_mask(seg, window_size[0])
        # ===== EXPLORE:DOC_MASK END =====
        # PER-LAYER DECORRELATED: precompute shifted indices (shared), compute per-layer hash indices inside loop
        prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        # Precompute per-layer bigram hash indices (different primes per layer for collision decorrelation)
        bigram_indices_per_layer = {}
        for layer_i in self.bigram_ve_layers:
            layer_bg_primes = self.bigram_hash_primes_per_layer[layer_i]
            bigram_indices_per_layer[layer_i] = [
                ((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size
                for p1, p2 in layer_bg_primes
            ]
        # Precompute per-layer trigram hash indices (different primes per layer for collision decorrelation)
        trigram_indices_per_layer = {}
        for layer_i in self.trigram_ve_layers:
            lp = self.trigram_hash_primes_per_layer[layer_i]
            trigram_indices_per_layer[layer_i] = (
                ((prev2_idx * lp[0]) ^ (prev_idx * lp[1]) ^ (idx * lp[2])) % self.trigram_table_size,
                ((prev2_idx * lp[3]) ^ (prev_idx * lp[4]) ^ (idx * lp[5])) % self.trigram_table_size,
            )
        # Precompute per-layer 4-gram hash indices (empty dict when 4-gram is disabled -> no overhead)
        fourgram_indices_per_layer = {}
        if self.fourgram_ve_layers:
            prev3_idx = torch.cat([idx[:, :3], idx[:, :-3]], dim=1)
            # ===== EXPLORE:NGRAM_FOURGRAM_SPAN BEGIN =====
            # far_idx is the FURTHEST slot of the 4-gram. With span=3 (default) it IS
            # prev3_idx, so the expression below is the original contiguous 4-gram and
            # no extra tensor is materialized -> byte-identical. With span=S>3 the slot
            # moves to w[-S], yielding the gappy context (w[-S],w[-2],w[-1],w[0]) at
            # UNCHANGED token count. prev3_idx is still built above because the 5-gram
            # path reuses it.
            _span = getattr(self.config, "ngram_fourgram_span", 3)
            if _span == 3:
                far_idx = prev3_idx
            else:
                far_idx = torch.cat([idx[:, :_span], idx[:, :-_span]], dim=1)
            # ===== EXPLORE:NGRAM_FOURGRAM_SPAN END =====
            for layer_i in self.fourgram_ve_layers:
                fp = self.fourgram_hash_primes_per_layer[layer_i]
                fourgram_indices_per_layer[layer_i] = (
                    ((far_idx * fp[0]) ^ (prev2_idx * fp[1]) ^ (prev_idx * fp[2]) ^ (idx * fp[3])) % self.fourgram_table_size,
                    ((far_idx * fp[4]) ^ (prev2_idx * fp[5]) ^ (prev_idx * fp[6]) ^ (idx * fp[7])) % self.fourgram_table_size,
                )
        # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
        # Per-layer 5-gram hash indices (empty dict when disabled -> no tensor ops).
        # Reuses the SAME prev/prev2 (and, when 4-gram is on, prev3) shifted-index
        # tensors computed above -- the token stream is never recomputed; only
        # prev4 (and prev3 iff 4-gram is off) is materialized here.
        fivegram_indices_per_layer = {}
        if self.fivegram_ve_layers:
            if not self.fourgram_ve_layers:
                prev3_idx = torch.cat([idx[:, :3], idx[:, :-3]], dim=1)
            prev4_idx = torch.cat([idx[:, :4], idx[:, :-4]], dim=1)
            for layer_i in self.fivegram_ve_layers:
                gp = self.fivegram_hash_primes_per_layer[layer_i]
                fivegram_indices_per_layer[layer_i] = (
                    ((prev4_idx * gp[0]) ^ (prev3_idx * gp[1]) ^ (prev2_idx * gp[2]) ^ (prev_idx * gp[3]) ^ (idx * gp[4])) % self.fivegram_table_size,
                    ((prev4_idx * gp[5]) ^ (prev3_idx * gp[6]) ^ (prev2_idx * gp[7]) ^ (prev_idx * gp[8]) ^ (idx * gp[9])) % self.fivegram_table_size,
                )
        # ===== EXPLORE:NGRAM_FIVEGRAM END =====
        n_layer = len(self.transformer.h)
        pool_start = n_layer - self.n_pool_layers
        pool_residual = None
        for i, block in enumerate(self.transformer.h):
            # Input-dependent x0 gate on ALL 8 layers: 2*sigmoid(scale*mean(x)) modulates x0 contribution
            # Starts at 1.0 (gate_scales=0 → sigmoid(0)=0.5 → 2*0.5=1.0)
            x0_gate = 2.0 * torch.sigmoid(self.x0_gate_scales[i] * x.float().mean(-1, keepdim=True)).to(x.dtype)
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0_gate * x0
            # ===== RSI INTERVENTION: ATTNRES BEGIN =====
            # Content-dependent softmax retrieval over x0 and every preceding layer
            # output, added through a zero-init per-layer gate (exact no-op at step 0).
            if ATTNRES_ENABLE:
                # KERNEL REFINEMENT (see ATTNRES block comment): never materialize the
                # stacked (S,B,T,d) states or keys. The score is a reduction over d, so
                # it is computed per-state into (B,T) and stacked at 1/d the size; the
                # weighted sum is then accumulated in place of a second (S,B,T,d) tensor.
                # Peak activation for this op drops from 2*S*B*T*d to S*B*T + B*T*d.
                # The block loop is a Python loop, so torch.compile unrolls it and S is a
                # compile-time constant at each layer -- these become fused elementwise
                # kernels rather than large allocations. Arithmetic is unchanged.
                _ar_q = self.attnres_queries[i]                          # (d,)
                _ar_scores = torch.stack(
                    [(_k * _ar_q).sum(-1) for _k in _ar_key], dim=0
                )                                                        # (S,B,T)
                _ar_w = torch.softmax(_ar_scores, dim=0)                 # (S,B,T)
                _ar_acc = None
                for _s, _st in enumerate(_ar_raw):
                    _term = _ar_w[_s].unsqueeze(-1) * _st
                    _ar_acc = _term if _ar_acc is None else _ar_acc + _term
                x = x + self.attnres_gate[i] * _ar_acc
            # ====== RSI INTERVENTION: ATTNRES END ======
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)
            else:
                ve = None
            # Factored multi-hash bigram VE: concat K=2 half-dim lookups from independent hashes (per-layer primes)
            if i in self.bigram_ve_layers:
                layer_ves = self.bigram_ves[str(i)]
                layer_indices = bigram_indices_per_layer[i]
                # ===== EXPLORE:NGRAM_SPARSE_GRAD BEGIN =====
                # Same gather values as the nn.Embedding call below, but backward
                # captures (indices, grad_rows) into preallocated buffers instead
                # of materializing a dense (table_size, dim) gradient. Only taken
                # when the flag is on AND grad is enabled (training fwd/bwd); the
                # eval/no_grad paths keep the plain embedding call. Flag off ->
                # the original line runs unchanged (byte-identical baseline).
                if NGRAM_SPARSE_GRAD and torch.is_grad_enabled():
                    bgve = torch.cat(
                        [
                            _ngram_sparse_gather("bigram", i, k, layer_ves[k].weight, layer_indices[k])
                            for k in range(self.bigram_K)
                        ],
                        dim=-1,
                    )
                else:
                    bgve = torch.cat([layer_ves[k](layer_indices[k]) for k in range(self.bigram_K)], dim=-1)
                # ===== EXPLORE:NGRAM_SPARSE_GRAD END =====
                # ===== RSI INTERVENTION: C3 BACKOFF GATE BEGIN =====
                # Scale each half-dim slice by count/(count+kappa) from its OWN hash's
                # count buffer, then accumulate this step's observations. Under-evidenced
                # (low-count) keys are suppressed; well-evidenced ones pass through.
                if NGRAM_BACKOFF_KAPPA > 0:
                    bgve = torch.cat(
                        [
                            bgve.chunk(self.bigram_K, dim=-1)[k]
                            * _c3_gate(self, "bigram", i, k, layer_indices[k]).to(bgve.dtype)
                            for k in range(self.bigram_K)
                        ],
                        dim=-1,
                    )
                # ====== RSI INTERVENTION: C3 BACKOFF GATE END ======
            else:
                bgve = None
            # Multi-layer factored trigram VE: concat two half-dim lookups per layer (per-layer primes)
            if i in self.trigram_ve_layers:
                tg_idx = trigram_indices_per_layer[i]
                layer_tves = self.trigram_ves[str(i)]
                # ===== EXPLORE:NGRAM_SPARSE_GRAD BEGIN =====
                # Sparse-gradient capture path (see bigram site above). With
                # SHARED_TRIGRAM_VE the same weight is gathered at several layers;
                # each layer writes its own capture slot and the optimizer
                # segment-sums all slots for the shared param.
                if NGRAM_SPARSE_GRAD and torch.is_grad_enabled():
                    tgve = torch.cat(
                        [
                            _ngram_sparse_gather("trigram", i, 0, layer_tves[0].weight, tg_idx[0]),
                            _ngram_sparse_gather("trigram", i, 1, layer_tves[1].weight, tg_idx[1]),
                        ],
                        dim=-1,
                    )
                else:
                    tgve = torch.cat([layer_tves[0](tg_idx[0]), layer_tves[1](tg_idx[1])], dim=-1)
                # ===== EXPLORE:NGRAM_SPARSE_GRAD END =====
                # ===== RSI INTERVENTION: C3 BACKOFF GATE BEGIN =====
                if NGRAM_BACKOFF_KAPPA > 0:
                    tgve = torch.cat(
                        [
                            tgve.chunk(2, dim=-1)[k]
                            * _c3_gate(self, "trigram", i, k, tg_idx[k]).to(tgve.dtype)
                            for k in range(2)
                        ],
                        dim=-1,
                    )
                # ====== RSI INTERVENTION: C3 BACKOFF GATE END ======
            else:
                tgve = None
            # Factored 4-gram VE (only on fourgram layers when enabled; else None)
            if i in self.fourgram_ve_layers:
                fg_idx = fourgram_indices_per_layer[i]
                layer_fves = self.fourgram_ves[str(i)]
                # ===== EXPLORE:NGRAM_SPARSE_GRAD BEGIN =====
                # Sparse-gradient capture path (see bigram site above).
                if NGRAM_SPARSE_GRAD and torch.is_grad_enabled():
                    fgve = torch.cat(
                        [
                            _ngram_sparse_gather("fourgram", i, 0, layer_fves[0].weight, fg_idx[0]),
                            _ngram_sparse_gather("fourgram", i, 1, layer_fves[1].weight, fg_idx[1]),
                        ],
                        dim=-1,
                    )
                else:
                    fgve = torch.cat([layer_fves[0](fg_idx[0]), layer_fves[1](fg_idx[1])], dim=-1)
                # ===== EXPLORE:NGRAM_SPARSE_GRAD END =====
            else:
                fgve = None
            # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
            # Factored 5-gram VE (only on fivegram layers when enabled; else None).
            if i in self.fivegram_ve_layers:
                vg_idx = fivegram_indices_per_layer[i]
                layer_gves = self.fivegram_ves[str(i)]
                # Sparse-gradient capture path (see bigram site above); registered
                # as "fivegram" sites in _ngram_sparse_setup so NGRAM_SPARSE_GRAD=1
                # covers these tables too.
                if NGRAM_SPARSE_GRAD and torch.is_grad_enabled():
                    vgve = torch.cat(
                        [
                            _ngram_sparse_gather("fivegram", i, 0, layer_gves[0].weight, vg_idx[0]),
                            _ngram_sparse_gather("fivegram", i, 1, layer_gves[1].weight, vg_idx[1]),
                        ],
                        dim=-1,
                    )
                else:
                    vgve = torch.cat([layer_gves[0](vg_idx[0]), layer_gves[1](vg_idx[1])], dim=-1)
            else:
                vgve = None
            # ===== EXPLORE:NGRAM_FIVEGRAM END =====
            # ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
            # Learned product-key memory residual. When the flag is off,
            # pk_mem_layers is empty -> pkve stays None and no ops are traced
            # (byte-identical baseline). Uses the block-input hidden state x
            # (pre-norm; _pk_mem_lookup norms it, matching what the attention
            # gates see) plus the already-computed prev2/prev/cur index tensors.
            if i in self.pk_mem_layers:
                pkve = self._pk_mem_lookup(i, x, prev2_idx, prev_idx, idx)
            else:
                pkve = None
            # ===== EXPLORE:NGRAM_PK_MEMORY END =====
            doc_mask = doc_masks[_doc_mask_key(self.window_sizes[i][0])] if doc_masks is not None else None
            # ===== RSI INTERVENTION: NGPT_SPHERE (capture pre-block stream) =====
            _pre_ngpt = x if NGPT_SPHERE else None
            x = block(x, ve, cos_sin, self.window_sizes[i], bigram_ve=bgve, trigram_ve=tgve, fourgram_ve=fgve, doc_mask=doc_mask, pk_ve=pkve, fivegram_ve=vgve)
            # ===== RSI INTERVENTION: NGPT_SPHERE BEGIN =====
            # Hypersphere update: instead of letting the residual stream's norm drift,
            # move it a learnable fraction alpha_i of the way toward this block's
            # (normalized) output and re-project onto the sphere. norm() is RMSNorm, so
            # this is elementwise and cheap -- unlike every quality lever tested this
            # block, it should cost ~no steps. _pre_ngpt is the stream BEFORE the block
            # ran, captured just above.
            if NGPT_SPHERE:
                x = norm(_pre_ngpt + self.ngpt_alpha[i] * (norm(x) - _pre_ngpt))
            # ====== RSI INTERVENTION: NGPT_SPHERE END ======
            # ===== RSI INTERVENTION: ATTNRES BEGIN =====
            if ATTNRES_ENABLE:
                _ar_raw.append(x)
                _ar_key.append(norm(x))
            # ====== RSI INTERVENTION: ATTNRES END ======
            if i == pool_start:
                pool_residual = self.layer_pool_weights[0] * x
            elif i == pool_start + 1:
                pool_residual = pool_residual + self.layer_pool_weights[1] * x
            elif i == pool_start + 2:
                pool_residual = pool_residual + self.layer_pool_weights[2] * x
        if pool_residual is not None:
            x = x + pool_residual
        x = norm(x)

        # Decoupled softcap in BF16: skip float() cast, halve logit tensor memory
        # Since model is natively BF16, softcap in BF16 should be numerically adequate
        logits = self.lm_head(x)
        logits = SOFTCAP_CAP * torch.tanh(logits / SOFTCAP_TAU)

        if targets is not None:
            # ===== EXPLORE:RHO1_GAMMA BEGIN =====
            # Rho-1-inspired online focal reweighting: upweight high-loss tokens by
            # w = ce**gamma (normalized to preserve loss scale). Only active for the
            # training reduction=="mean" path with gamma>0; the eval / reduction!=
            # "mean" paths and the gamma==0.0 default fall through to the ORIGINAL
            # F.cross_entropy call below, keeping the baseline byte-identical.
            # (self.training guard: the per-step val_loss probe also calls with
            # reduction="mean"; without the guard a RHO1 treatment would log
            # focal-REWEIGHTED val losses. Final val_bpb uses reduction="none"
            # and was never affected. No-op when RHO1_GAMMA=0.)
            if RHO1_GAMMA > 0.0 and reduction == "mean" and self.training:
                ce = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                    reduction="none",
                )
                w = ce.detach() ** RHO1_GAMMA
                w = w / w.mean()
                return (w * ce).mean()
            # ===== EXPLORE:RHO1_GAMMA END =====
            # Cast to float32 only for the CE loss computation (numerically sensitive)
            # EXPLORE:BF16_CE -- when on, keep logits in bf16 for CE (skip the float32
            # materialization of [B*T, vocab]); byte-identical to the cast path when off.
            _ce_logits = logits if BF16_CE else logits.float()
            loss = F.cross_entropy(
                _ce_logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        # Eval path: need float32 logits
        return logits.float()


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t, cautious
):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    u = exp_avg / denom
    if cautious:
        # Cautious optimizer (Liang et al. 2024): retain update components that
        # agree in sign with the current gradient, then preserve magnitude.
        mask = (u * grad > 0).to(u.dtype)
        scale = mask.mean().clamp_min(1e-3)
        u = u * mask / scale
    p.add_(u, alpha=-step_size)


@torch.compile(dynamic=False, fullgraph=True)
def rmsprop_step_fused(p, grad, exp_avg_sq, step_t, lr_t, beta2_t, eps_t, wd_t):
    """RMSProp with bias correction -- no first moment, saves 50% optimizer VRAM."""
    p.mul_(1 - lr_t * wd_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    p.add_(grad / denom, alpha=-lr_t)


# ===== EXPLORE:NGRAM_STATE_ROWWISE BEGIN =====
@torch.compile(dynamic=False, fullgraph=True)
def rmsprop_rowwise_step_fused(p, grad, row_ms, step_t, lr_t, beta2_t, eps_t, wd_t):
    """RMSProp with an Adafactor-style ROW-WISE second moment.

    Identical to rmsprop_step_fused except the second moment is a single fp32
    scalar per row (shape (rows, 1)) holding the row mean-square, instead of a
    full (rows, dim) tensor. The n-gram hash tables are gathered/updated
    row-atomically, so within a touched row the per-element second moments are
    highly correlated and collapsing them costs little. State bytes drop from
    1x table-bytes to dim/2 times less (bf16 (R,384) -> fp32 (R,1) = 1/192).
    """
    p.mul_(1 - lr_t * wd_t)
    g32 = grad.float()
    row_ms.lerp_(g32.square().mean(dim=1, keepdim=True), 1 - beta2_t)
    bias2 = 1 - beta2_t**step_t
    denom = (row_ms / bias2).sqrt() + eps_t
    p.add_((g32 / denom).to(p.dtype), alpha=-lr_t)
# ===== EXPLORE:NGRAM_STATE_ROWWISE END =====


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    wd_t,
    beta2_t,
    ns_steps,
    red_dim,
    cautious,
):
    # Avoid the full-gradient clone and mask operations when the experiment is
    # disabled, preserving baseline memory use and throughput.
    if cautious:
        raw_grad = stacked_grads.clone()
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    if cautious:
        # Mask the final Muon update where it disagrees with the raw gradient,
        # rescaled to preserve update magnitude.
        cmask = (g * raw_grad.to(g.dtype) > 0).to(g.dtype)
        cscale = cmask.mean().clamp_min(1e-3)
        g = g * cmask / cscale
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# Cautious-update optimizer (Liang et al. 2024): mask AdamW/Muon updates that
# fight the current gradient. Off by default; CAUTIOUS_UPDATE=1 enables it.
_cautious_update_raw = os.environ.get("CAUTIOUS_UPDATE", "0")
if _cautious_update_raw not in {"0", "1"}:
    raise ValueError("CAUTIOUS_UPDATE must be 0 or 1")
CAUTIOUS_UPDATE = _cautious_update_raw == "1"
# CAUTIOUS_MUON=0 restricts cautious masking to AdamW params only (Muon's
# orthogonalized update is disrupted by elementwise masking). Default 1 = both.
_cautious_muon_raw = os.environ.get("CAUTIOUS_MUON", "1")
if _cautious_muon_raw not in {"0", "1"}:
    raise ValueError("CAUTIOUS_MUON must be 0 or 1")
CAUTIOUS_MUON = _cautious_muon_raw == "1"

# ===== EXPLORE:NGRAM_STATE_ROWWISE BEGIN =====
# Adafactor-style row-wise second moment for the n-gram VE (is_ngram_ve) groups.
# The hash tables are touched all-or-nothing (a bucket hit gradients all 384 dims),
# so per-element exp_avg_sq is near-redundant within a row. Storing a (rows, 1)
# fp32 row mean-square instead of a (rows, 384) bf16 half-table cuts optimizer
# state by 192x for each half-table (768 bytes/row -> 4 bytes/row), matching
# the aggregate 22.5 GB -> ~117 MB estimate at 256x. State and parameter bytes
# must be reported separately. This is what previously capped
# NGRAM_TABLE_MULT (512x OOM). Off by default -> the
# original rmsprop_step_fused path runs unchanged (byte-identical baseline).
_ngram_state_rowwise_raw = os.environ.get("NGRAM_STATE_ROWWISE", "0")
if _ngram_state_rowwise_raw not in {"0", "1"}:
    raise ValueError("NGRAM_STATE_ROWWISE must be 0 or 1")
NGRAM_STATE_ROWWISE = _ngram_state_rowwise_raw == "1"
# ===== EXPLORE:NGRAM_STATE_ROWWISE END =====

# ===== EXPLORE:COMPILE_MODE BEGIN =====
# torch.compile mode. Default "max-autotune" reproduces the frozen baseline but
# reserves ~25 GB of CUDA-graph private pools. "max-autotune-no-cudagraphs" frees
# that VRAM for larger n-gram tables; the challenge is STEP-budgeted (STOP_MODE=
# steps), so the resulting step-time cost does not affect val_bpb.
COMPILE_MODE = os.environ.get("COMPILE_MODE", "max-autotune")
if COMPILE_MODE not in {"max-autotune", "max-autotune-no-cudagraphs", "default", "reduce-overhead"}:
    raise ValueError(
        "COMPILE_MODE must be one of: max-autotune, max-autotune-no-cudagraphs, default, reduce-overhead"
    )
# ===== EXPLORE:COMPILE_MODE END =====

# ===== EXPLORE:MUON_MOMENTUM_CONTINUOUS BEGIN =====
# Structural fix for a schedule bug: the Muon momentum warmup (in the training
# loop) is STEP-anchored (min(step/300,1)) while every other schedule is
# PROGRESS-anchored. With WARMDOWN_RATIO=0.95, warmdown starts at progress 0.05
# (~step 110), where the step-warmup has only reached ~0.887; the warmdown branch
# then snaps momentum to 0.95 (a ~+0.063 discontinuity in the highest-LR window)
# and decays to 0.79. The intended 300-step warmup is dead code past ~step 110 and
# peak momentum is never held. When 1, the warmup is re-anchored to progress so
# momentum reaches MUON_PEAK_MOMENTUM exactly at warmdown onset -> continuous, no
# jump. Off (0) is byte-identical to the current baseline.
MUON_MOMENTUM_CONTINUOUS = os.environ.get("MUON_MOMENTUM_CONTINUOUS", "0") == "1"
if os.environ.get("MUON_MOMENTUM_CONTINUOUS", "0") not in {"0", "1"}:
    raise ValueError("MUON_MOMENTUM_CONTINUOUS must be 0 or 1")
# ===== EXPLORE:MUON_MOMENTUM_CONTINUOUS END =====

# ===== EXPLORE:BF16_CE BEGIN =====
# Compute the training cross-entropy directly in bf16 instead of casting the full
# [B*T, vocab] logits to float32 first. The float32 cast materializes the single
# largest activation twice its bf16 size; skipping it cuts that memory + bandwidth,
# buying steps in the time-budgeted frame. Eval val_bpb still uses float32 logits
# (the reduction!="mean" path below is unchanged), so the *metric* is unaffected;
# only the training gradient uses a coarser bf16 softmax. Net effect (throughput vs
# per-step quality) is measured by the gated experiment. Off (0) is byte-identical.
BF16_CE = os.environ.get("BF16_CE", "0") == "1"
if os.environ.get("BF16_CE", "0") not in {"0", "1"}:
    raise ValueError("BF16_CE must be 0 or 1")
# ===== EXPLORE:BF16_CE END =====

# ===== EXPLORE:NGRAM_SPARSE_GRAD BEGIN =====
# Sparse gradients for the n-gram VE hash tables. The tables are pure gathers
# (nn.Embedding), yet autograd materializes a DENSE (table_size, half_kv_dim)
# bf16 gradient per half-table every step -- ~1x total table bytes (~22.5 GB at
# NGRAM_TABLE_MULT=256), scaling linearly with the table mult, which is what
# caps higher mults even after NGRAM_STATE_ROWWISE. When on, the table gather
# runs through a torch.library custom op (same pattern as the fa4 wrapper near
# the top of this file) whose backward WRITES the per-token (indices, grad_rows)
# pair -- exactly (B*T,) int64 + (B*T, half_kv_dim) bf16 per gather site,
# ~1.6 GB total, CONSTANT in table size -- into preallocated capture buffers
# and returns None for the weight grad, so the dense gradient tensor is never
# allocated. The optimizer branch then segment-sums duplicate indices
# (torch.unique + index_add_, eager) and applies the same bias-corrected
# RMSProp update via row-indexed ops:
#   - UNTOUCHED rows: identical decay semantics to the dense path -- the
#     second-moment state lerps toward zero by (1-beta2) every step exactly as
#     the fused kernel does with a zero grad row, and decoupled weight decay
#     would apply if it were nonzero (the n-gram groups use weight_decay=0.0).
#     This is the SAFE (non-lazy) version: no deferred/lazy decay.
#   - TOUCHED rows: the dense kernel fuses state.lerp_(g^2, 1-beta2) in one op;
#     here it is decay-then-index_add_, and duplicate-row grad contributions
#     are summed in fp32 (dense embedding backward sums them with bf16
#     atomics), so results match to rounding but are not bitwise identical.
# Composes with NGRAM_STATE_ROWWISE=1 ((rows,1) fp32 state) -- the intended
# pairing -- and also supports the full (rows,dim) state for ablation.
# Requires grad_accum_steps==1 (asserted at setup): the capture buffers hold
# exactly one micro-batch and accumulation would overwrite them. Off by
# default: no custom op is registered, GPT.forward keeps the plain nn.Embedding
# calls and the optimizer keeps the fused dense kernels (byte-identical).
_ngram_sparse_grad_raw = os.environ.get("NGRAM_SPARSE_GRAD", "0")
if _ngram_sparse_grad_raw not in {"0", "1"}:
    raise ValueError("NGRAM_SPARSE_GRAD must be 0 or 1")
NGRAM_SPARSE_GRAD = _ngram_sparse_grad_raw == "1"

# (kind, layer, k) -> (idx_buf, grad_buf); populated by _ngram_sparse_setup().
_NGRAM_SPARSE_SITE_BUFS = {}
# id(param) -> [(idx_buf, grad_buf), ...]; shared tables (SHARED_TRIGRAM_VE)
# get one slot per gather site and are segment-summed together at step time.
_NGRAM_SPARSE_PARAM_SLOTS = {}

if NGRAM_SPARSE_GRAD:

    @torch.library.custom_op("ngram_sparse::gather", mutates_args=())
    def _ngram_sparse_gather_op(
        weight: torch.Tensor,
        indices: torch.Tensor,
        idx_buf: torch.Tensor,
        grad_buf: torch.Tensor,
    ) -> torch.Tensor:
        # Forward is numerically identical to nn.Embedding: a pure row gather.
        return F.embedding(indices, weight)

    @_ngram_sparse_gather_op.register_fake
    def _ngram_sparse_gather_fake(weight, indices, idx_buf, grad_buf):
        return weight.new_empty(*indices.shape, weight.shape[1])

    @torch.library.custom_op(
        "ngram_sparse::capture_grad", mutates_args={"idx_buf", "grad_buf"}
    )
    def _ngram_sparse_capture_op(
        grad_out: torch.Tensor,
        indices: torch.Tensor,
        idx_buf: torch.Tensor,
        grad_buf: torch.Tensor,
    ) -> None:
        # Backward side effect: stash (indices, grad_rows) instead of scattering
        # into a dense (table_size, dim) tensor. mutates_args keeps this op from
        # being dead-code-eliminated under torch.compile even though the weight
        # grad returned to autograd is None.
        idx_buf.copy_(indices.reshape(-1))
        grad_buf.copy_(grad_out.reshape(-1, grad_out.shape[-1]))

    @_ngram_sparse_capture_op.register_fake
    def _ngram_sparse_capture_fake(grad_out, indices, idx_buf, grad_buf):
        return None

    def _ngram_sparse_setup_context(ctx, inputs, output):
        weight, indices, idx_buf, grad_buf = inputs
        ctx.save_for_backward(indices, idx_buf, grad_buf)

    def _ngram_sparse_backward(ctx, grad_out):
        indices, idx_buf, grad_buf = ctx.saved_tensors
        torch.ops.ngram_sparse.capture_grad(grad_out, indices, idx_buf, grad_buf)
        # None for the weight: the dense gradient is never allocated; the
        # optimizer reads the capture buffers instead. Remaining inputs are
        # non-differentiable (int indices / capture buffers).
        return None, None, None, None

    _ngram_sparse_gather_op.register_autograd(
        _ngram_sparse_backward, setup_context=_ngram_sparse_setup_context
    )

    def _ngram_sparse_gather(kind, layer_i, k, weight, indices):
        idx_buf, grad_buf = _NGRAM_SPARSE_SITE_BUFS[(kind, layer_i, k)]
        return torch.ops.ngram_sparse.gather(weight, indices, idx_buf, grad_buf)

    def _ngram_sparse_setup(model, batch_size, seq_len, device):
        """Preallocate capture buffers for every n-gram VE gather site.

        Must run after the model's bf16 cast (buffers match weight dtype) and
        before torch.compile (the buffer dict must be final when traced).
        """
        n = batch_size * seq_len
        sites = []
        for layer_i in sorted(model.bigram_ve_layers):
            for k in range(model.bigram_K):
                sites.append(("bigram", layer_i, k, model.bigram_ves[str(layer_i)][k].weight))
        for layer_i in sorted(model.trigram_ve_layers):
            for k in range(2):
                sites.append(("trigram", layer_i, k, model.trigram_ves[str(layer_i)][k].weight))
        for layer_i in sorted(model.fourgram_ve_layers):
            for k in range(2):
                sites.append(("fourgram", layer_i, k, model.fourgram_ves[str(layer_i)][k].weight))
        # ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
        # 5-gram capture sites (empty set when disabled -> site list unchanged).
        # With 5-gram on the trigram layers this adds 3 layers * 2 half-tables = 6
        # sites (20 -> 26 with 4-gram also on).
        for layer_i in sorted(model.fivegram_ve_layers):
            for k in range(2):
                sites.append(("fivegram", layer_i, k, model.fivegram_ves[str(layer_i)][k].weight))
        # ===== EXPLORE:NGRAM_FIVEGRAM END =====
        for kind, layer_i, k, weight in sites:
            idx_buf = torch.zeros(n, dtype=torch.long, device=device)
            grad_buf = torch.zeros(n, weight.shape[1], dtype=weight.dtype, device=device)
            # Static addresses so inductor cudagraphs can accept in-graph
            # mutation of these non-parameter tensors (best effort; if the API
            # is unavailable or cudagraphs still balk, run with
            # COMPILE_MODE=max-autotune-no-cudagraphs).
            try:
                import torch._dynamo as _dyn
                _dyn.mark_static_address(idx_buf)
                _dyn.mark_static_address(grad_buf)
            except Exception:
                pass
            _NGRAM_SPARSE_SITE_BUFS[(kind, layer_i, k)] = (idx_buf, grad_buf)
            _NGRAM_SPARSE_PARAM_SLOTS.setdefault(id(weight), []).append((idx_buf, grad_buf))
        total_mb = sum(
            ib.numel() * ib.element_size() + gb.numel() * gb.element_size()
            for ib, gb in _NGRAM_SPARSE_SITE_BUFS.values()
        ) / 1024 / 1024
        print(
            f"NGRAM_SPARSE_GRAD: {len(sites)} capture sites, "
            f"{len(_NGRAM_SPARSE_PARAM_SLOTS)} params, {total_mb:.1f} MB buffers",
            flush=True,
        )

    _ngram_sparse_zeros = {}

    def _ngram_sparse_zero(dtype, device):
        key = (dtype, device)
        z = _ngram_sparse_zeros.get(key)
        if z is None:
            z = torch.zeros((), dtype=dtype, device=device)
            _ngram_sparse_zeros[key] = z
        return z

    def _ngram_sparse_rmsprop_step(p, state, lr, beta2, eps, wd, step_num, rowwise):
        """Dense-equivalent RMSProp step from captured (indices, grad_rows).

        Matches rmsprop(_rowwise)_step_fused semantics: full-table state decay
        every step (safe, non-lazy) plus a gradient-driven update on touched
        rows only, with duplicate indices segment-summed FIRST (as the dense
        embedding backward does) so the second moment and denom see the SUMMED
        row gradient, not per-occurrence fragments. Runs eager on purpose:
        torch.unique has data-dependent output shape, incompatible with the
        dynamic=False fullgraph=True convention of the fused kernels. Step-time
        cost does not affect val_bpb under STOP_MODE=steps.
        """
        slots = _NGRAM_SPARSE_PARAM_SLOTS[id(p)]
        if len(slots) == 1:
            idx, g = slots[0]
        else:
            # Weight-shared table gathered at several sites: the dense grad is
            # the sum over all of them -> concat, then segment-sum below.
            idx = torch.cat([s[0] for s in slots])
            g = torch.cat([s[1] for s in slots])
        uniq, inverse = torch.unique(idx, return_inverse=True)
        g32 = torch.zeros(uniq.numel(), g.shape[1], dtype=torch.float32, device=g.device)
        g32.index_add_(0, inverse, g.float())
        if wd != 0.0:
            # Dense path applies decoupled WD to every row. The n-gram groups
            # use weight_decay=0.0, so this full-table op is normally skipped
            # (multiplying by exactly 1.0 is a numerical no-op).
            p.mul_(1 - lr * wd)
        bias2 = 1 - beta2 ** step_num
        if rowwise:
            row_ms = state["row_ms"]
            # Untouched rows: lerp toward 0 == the dense rowwise kernel's
            # lerp_(mean(g^2)=0, 1-beta2) exactly. Touched rows then get their
            # (1-beta2)*mean(g^2) contribution added back.
            row_ms.lerp_(_ngram_sparse_zero(row_ms.dtype, row_ms.device), 1 - beta2)
            row_ms.index_add_(
                0, uniq, g32.square().mean(dim=1, keepdim=True), alpha=(1 - beta2)
            )
            denom = (row_ms.index_select(0, uniq) / bias2).sqrt() + eps
        else:
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.lerp_(
                _ngram_sparse_zero(exp_avg_sq.dtype, exp_avg_sq.device), 1 - beta2
            )
            exp_avg_sq.index_add_(
                0, uniq, g32.square().to(exp_avg_sq.dtype), alpha=(1 - beta2)
            )
            denom = (exp_avg_sq.index_select(0, uniq).float() / bias2).sqrt() + eps
        p.index_add_(0, uniq, (g32 / denom).to(p.dtype), alpha=-lr)
# ===== EXPLORE:NGRAM_SPARSE_GRAD END =====

# ===== EXPLORE:TIE_EMBED BEGIN =====
# Weight-tie the output head to the input embedding. Off by default; when off the
# lm_head keeps its own separately-initialized weight (std 0.001) and nothing is
# tied, so the code path is byte-identical to baseline.
_tie_embed_raw = os.environ.get("TIE_EMBED", "0")
if _tie_embed_raw not in {"0", "1"}:
    raise ValueError("TIE_EMBED must be 0 or 1")
TIE_EMBED = _tie_embed_raw == "1"
# ===== EXPLORE:TIE_EMBED END =====

# ===== EXPLORE:RESID_SCALE BEGIN =====
# Depth-muP-style residual init scaling. When off, resid_lambdas init to 1.0
# exactly as baseline (byte-identical). When on, they init to 1/sqrt(2*n_layer).
_resid_scale_raw = os.environ.get("RESID_SCALE", "0")
if _resid_scale_raw not in {"0", "1"}:
    raise ValueError("RESID_SCALE must be 0 or 1")
RESID_SCALE = _resid_scale_raw == "1"
# ===== EXPLORE:RESID_SCALE END =====

# ===== EXPLORE:DOC_MASK BEGIN =====
# Intra-document (block-diagonal) causal attention on the SDPA path. Off by
# default: no segment ids are computed and the attention call stays is_causal=True
# (byte-identical). Only affects ATTN_BACKEND=="sdpa"; fa3/fa4 paths are untouched.
# ADOPTED 2026-07-29 (n=10 paired, mean -0.005222, t=-10.20, gate cleared): fa3
# varlen intra-document attention masking is now the default. Set DOC_MASK=0 to
# recover the pre-adoption baseline. See sota_snapshots/sota_fa3_varlen_docmask/.
_doc_mask_raw = os.environ.get("DOC_MASK", "1")
if _doc_mask_raw not in {"0", "1"}:
    raise ValueError("DOC_MASK must be 0 or 1")
DOC_MASK = _doc_mask_raw == "1"
DOC_MASK_MODE = os.environ.get("DOC_MASK_MODE", "both").lower()
if DOC_MASK_MODE not in {"both", "seg", "window"}:
    raise ValueError("DOC_MASK_MODE must be one of: both, seg, window")
# Masking SEMANTICS are identical across implementations; only the kernel differs.
#   dense (default) -- (B, 1, T, T) boolean + explicit-mask SDPA. Reproduces every
#     recorded DOC_MASK run bit-for-bit, and is the reference for equivalence tests.
#   flex -- block-sparse FlexAttention. Skips whole cross-document 128x128 blocks
#     instead of paying dense mask traffic on every layer, so segmented attention
#     is no longer priced at the dense-mask rate. bf16 outputs differ from dense
#     at kernel-reassociation tolerance, so it is a NEW arm, not a free swap.
DOC_MASK_IMPL = os.environ.get("DOC_MASK_IMPL", "varlen").lower()  # ADOPTED: varlen (was dense)
DOC_MASK_BLOCK = int(os.environ.get("DOC_MASK_BLOCK", "128"))  # flex block-sparse granularity
if DOC_MASK_IMPL not in {"dense", "flex", "varlen"}:
    raise ValueError("DOC_MASK_IMPL must be one of: dense, flex, varlen")
if DOC_MASK_IMPL == "flex" and ATTN_BACKEND != "sdpa":
    raise ValueError("DOC_MASK_IMPL=flex requires ATTN_BACKEND=sdpa")
# ===== RSI INTERVENTION: FA3 VARLEN DOC-MASK =====
if DOC_MASK_IMPL == "varlen" and ATTN_BACKEND != "fa3":
    raise ValueError("DOC_MASK_IMPL=varlen requires ATTN_BACKEND=fa3")
# Packer-sourced FA3 boundaries are an explicit execution intervention. OFF is
# the scanner path above; ON is valid only for the adopted FA3 varlen doc-mask
# semantics and changes no tokens, packing choices, or model graph.
_packer_doc_boundaries_raw = os.environ.get("PACKER_DOC_BOUNDARIES", "0")
if _packer_doc_boundaries_raw not in {"0", "1"}:
    raise ValueError("PACKER_DOC_BOUNDARIES must be 0 or 1")
PACKER_DOC_BOUNDARIES = _packer_doc_boundaries_raw == "1"
if PACKER_DOC_BOUNDARIES and not (
    DOC_MASK and DOC_MASK_IMPL == "varlen" and ATTN_BACKEND == "fa3"
):
    raise ValueError(
        "PACKER_DOC_BOUNDARIES=1 requires DOC_MASK=1 "
        "DOC_MASK_IMPL=varlen ATTN_BACKEND=fa3"
    )

try:
    PACKER_BOUNDARY_VERIFY_BATCHES = int(
        os.environ.get("PACKER_BOUNDARY_VERIFY_BATCHES", "0")
    )
except ValueError as exc:
    raise ValueError("PACKER_BOUNDARY_VERIFY_BATCHES must be 0 or 1000") from exc
if PACKER_BOUNDARY_VERIFY_BATCHES not in {0, 1000}:
    raise ValueError("PACKER_BOUNDARY_VERIFY_BATCHES must be 0 or 1000")
if PACKER_BOUNDARY_VERIFY_BATCHES and not (
    DOC_MASK and DOC_MASK_IMPL == "varlen" and ATTN_BACKEND == "fa3"
):
    raise ValueError(
        "PACKER_BOUNDARY_VERIFY_BATCHES=1000 requires DOC_MASK=1 "
        "DOC_MASK_IMPL=varlen ATTN_BACKEND=fa3"
    )

_packer_diagnostic_hashes_raw = os.environ.get("PACKER_DIAGNOSTIC_HASHES", "0")
if _packer_diagnostic_hashes_raw not in {"0", "1"}:
    raise ValueError("PACKER_DIAGNOSTIC_HASHES must be 0 or 1")
PACKER_DIAGNOSTIC_HASHES = _packer_diagnostic_hashes_raw == "1"
try:
    PACKER_SIDECAR_ACTIVATE_STEP = int(
        os.environ.get("PACKER_SIDECAR_ACTIVATE_STEP", "0")
    )
except ValueError as exc:
    raise ValueError(
        "PACKER_SIDECAR_ACTIVATE_STEP must be a non-negative integer"
    ) from exc
if PACKER_SIDECAR_ACTIVATE_STEP < 0:
    raise ValueError(
        "PACKER_SIDECAR_ACTIVATE_STEP must be a non-negative integer"
    )
if PACKER_SIDECAR_ACTIVATE_STEP and not PACKER_DIAGNOSTIC_HASHES:
    raise ValueError(
        "PACKER_SIDECAR_ACTIVATE_STEP is a diagnostic-only phase boundary"
    )
# DOC_MASK_BOS_ID is resolved after the tokenizer is constructed (see below).
DOC_MASK_BOS_ID = None
# ===== EXPLORE:DOC_MASK END =====

# ===== EXPLORE:RESET_ROPE BEGIN =====
# Per-document RoPE position reset. Off by default: GPT.forward keeps the shared
# monotonic cos/sin slice (byte-identical baseline). When on, rotary position
# indices restart at 0 at each BOS token, making positional encoding
# intra-document -- the structural partner of the DOC_MASK intra-doc attention
# mask. Reuses DOC_MASK_BOS_ID (resolved after tokenizer construction below).
_reset_rope_raw = os.environ.get("RESET_ROPE", "0")
if _reset_rope_raw not in {"0", "1"}:
    raise ValueError("RESET_ROPE must be 0 or 1")
RESET_ROPE = _reset_rope_raw == "1"
# ===== EXPLORE:RESET_ROPE END =====


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # RMSProp CPU tensors (no beta1 -- saves first moment VRAM)
        self._rmsprop_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._rmsprop_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        step_fns = {
            "adamw": self._step_adamw,
            "rmsprop": self._step_rmsprop,
            "muon": self._step_muon,
        }
        self._step_dispatch = tuple((step_fns[group["kind"]], group) for group in self.param_groups)

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
                CAUTIOUS_UPDATE,
            )

    def _step_rmsprop(self, group):
        """RMSProp: only second moment, no first moment -- 50% less optimizer VRAM for sparse tables."""
        # EXPLORE:NGRAM_STATE_ROWWISE -- row-wise second moment for the n-gram VE
        # tables only. Off by default => original full-size state path (byte-identical).
        rowwise = NGRAM_STATE_ROWWISE and group.get("is_ngram_ve", False)
        # ===== EXPLORE:NGRAM_SPARSE_GRAD BEGIN =====
        # Captured-sparse-gradient path for the n-gram VE tables. Off by default
        # => `sparse` is False and the loop below runs untouched (byte-identical).
        sparse = NGRAM_SPARSE_GRAD and group.get("is_ngram_ve", False)
        # ===== EXPLORE:NGRAM_SPARSE_GRAD END =====
        for p in group["params"]:
            # ===== EXPLORE:NGRAM_SPARSE_GRAD BEGIN =====
            if sparse and p.dim() == 2 and id(p) in _NGRAM_SPARSE_PARAM_SLOTS:
                # p.grad is None here BY DESIGN (the custom-op backward returns
                # None for the weight); the gradient lives in the capture
                # buffers, so this branch must run before the p.grad-None skip.
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    if rowwise:
                        state["row_ms"] = torch.zeros(
                            p.shape[0], 1, dtype=torch.float32, device=p.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                _ngram_sparse_rmsprop_step(
                    p,
                    state,
                    group["lr"],
                    group["beta2"],
                    group["eps"],
                    group["weight_decay"],
                    state["step"],
                    rowwise,
                )
                continue
            # ===== EXPLORE:NGRAM_SPARSE_GRAD END =====
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            # Row-wise state only applies to 2D tables; anything else falls back.
            p_rowwise = rowwise and p.dim() == 2
            if not state:
                state["step"] = 0
                if p_rowwise:
                    state["row_ms"] = torch.zeros(
                        p.shape[0], 1, dtype=torch.float32, device=p.device
                    )
                else:
                    state["exp_avg_sq"] = torch.zeros_like(p)
                # Note: NO exp_avg allocated -- this is the VRAM saving
            state["step"] += 1
            self._rmsprop_step_t.fill_(state["step"])
            self._rmsprop_lr_t.fill_(group["lr"])
            self._rmsprop_beta2_t.fill_(group["beta2"])
            self._rmsprop_eps_t.fill_(group["eps"])
            self._rmsprop_wd_t.fill_(group["weight_decay"])
            if p_rowwise:
                rmsprop_rowwise_step_fused(
                    p,
                    grad,
                    state["row_ms"],
                    self._rmsprop_step_t,
                    self._rmsprop_lr_t,
                    self._rmsprop_beta2_t,
                    self._rmsprop_eps_t,
                    self._rmsprop_wd_t,
                )
                continue
            rmsprop_step_fused(
                p,
                grad,
                state["exp_avg_sq"],
                self._rmsprop_step_t,
                self._rmsprop_lr_t,
                self._rmsprop_beta2_t,
                self._rmsprop_eps_t,
                self._rmsprop_wd_t,
            )

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
            CAUTIOUS_UPDATE and CAUTIOUS_MUON,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for step_fn, group in self._step_dispatch:
            step_fn(group)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 96  # model_dim = depth * ASPECT_RATIO (d8*96=768 -> dim=768, 6 heads)
HEAD_DIM = 128  # target head dimension for attention
# Env-gated for the post-adoption retune (H7, paper 007): the SSSL window was
# tuned WITHOUT doc-masking. Now that cross-document attention is blocked, a
# window may be redundant (documents are typically far shorter than T) or even
# harmful (it can cut in-document context the mask already protects). Default
# "SSSL" is unchanged, so unset == the adopted configuration.
WINDOW_PATTERN = os.environ.get("WINDOW_PATTERN", "SSSL").upper()

# Optimization
# Env-gated 2026-07-29 to test doc-masking on the PRE-RETARGET config. The v21
# re-target to the sota_swin4 lineage cost +0.0067 on the 10-seed baseline
# (0.933073 -> 0.939797); doc-masking then won back -0.0049. Whether doc-masking
# also pays on the OLD constants has never been tested. Defaults are the current
# adopted values, so unset == the adopted configuration.
TOTAL_BATCH_SIZE = int(os.environ.get("TOTAL_BATCH_SIZE", str(2**18)))
DEVICE_BATCH_SIZE = int(os.environ.get("DEVICE_BATCH_SIZE", "128"))
EMBEDDING_LR = 0.6  # locally bracketed token embedding learning rate
UNEMBEDDING_LR = 0.004  # locally bracketed lm_head learning rate
MATRIX_LR = float(os.environ.get("MATRIX_LR", "0.03"))
SCALAR_LR = 0.8  # rsi/daniel baseline
WEIGHT_DECAY = 0.1  # rsi/daniel baseline
ADAM_BETAS = (0.8, float(os.environ.get("ADAM_BETA2", "0.95")))  # Adam beta1, beta2 (EXPLORE: ADAM_BETA2)
DEMON_FINAL_BETA1 = 0.55  # rsi/daniel baseline
NGRAM_VE_BETAS = (0.5, 0.999)  # rsi/daniel baseline (RMSProp beta2=0.999)
# Fresh start from rsi/daniel: the prior 0.99955/0.99999 sparse-beta2 settings were
# broken-3-shard overfitting artifacts and are not carried over. Any change here is
# a fresh hypothesis to be gated on the 2000-step H200 baseline.
NGRAM_VE_LR_SCALE = 1.0  # rsi/daniel baseline (full LR, no reduction)
# Fraction of total progress spent ramping LR up before the flat phase. 0.0 (the
# baseline) disables warmup entirely and leaves every LR write path untouched, so
# the default run is bit-identical to the no-warmup schedule. When >0 the ramp
# runs from FINAL_LR_FRAC*lr to lr across [0, WARMUP_RATIO); FINAL_LR_FRAC is
# reused as the floor so warmup mirrors the warmdown tail rather than adding a
# second, separately-provenanced knob.
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.95  # rsi/daniel baseline Muon warmdown ratio
ADAM_WARMDOWN_RATIO = 0.65  # rsi/daniel baseline Adam/Demon warmdown ratio
NGRAM_WARMDOWN_RATIO = 0.0  # no warmdown for bigram/trigram VE (sparse tables benefit from full-rate training)
FINAL_LR_FRAC = 0.05  # rsi/daniel baseline terminal LR fraction

# ===================== RSI FIXED-COMPUTE STOP: BEGIN =====================
# Confound fix: stopping at a fixed wall-time (TIME_BUDGET) on a shared GPU makes
# the tokens processed -- and therefore val_bpb -- depend on GPU contention, not
# just the intervention. STOP_MODE="steps" (default) stops after a fixed MAX_STEPS
# so every run sees identical optimizer steps and training tokens. It does NOT
# equalize FLOPs, memory traffic, or seconds when the architecture changes.
# Contention changes elapsed time but not token exposure. Set STOP_MODE=time for
# an equal steady-training-time diagnostic. MAX_STEPS=2000 is the frozen scope.
STOP_MODE = os.environ.get("STOP_MODE", "steps").lower()  # "steps" | "time" | "walltime"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "2000"))       # fixed-compute target (frozen challenge)
SEED = int(os.environ.get("SEED", "42"))
# "walltime" stops on real elapsed time since t_start_training (including lazy
# compilation, probes, data stalls, and logging inside the loop). It still excludes
# process imports/model setup before the loop and final evaluation, so it is a
# training-loop wall clock—not an end-to-end job wall clock. "time" stops on summed
# steady train-step time; in TRACK_B_MODE that clock is exactly the synchronized
# forward/backward/optimizer/data-fetch samples recorded in the result.
if STOP_MODE not in {"steps", "time", "walltime"}:
    raise ValueError("STOP_MODE must be one of: steps, time, walltime")
# TIME_BUDGET is imported from lib (frozen default 300s = the 5-min speedrun frame).
# Allow an explicit env override so the wall-time budget is auditable in RESOLVED_CONFIG
# and short smoke runs are possible; default preserves the frozen 300s.
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", str(TIME_BUDGET)))
if MAX_STEPS <= 10:
    raise ValueError("MAX_STEPS must be greater than 10 compilation/warmup steps")
if PACKER_DIAGNOSTIC_HASHES and STOP_MODE != "steps":
    raise ValueError("PACKER_DIAGNOSTIC_HASHES=1 requires STOP_MODE=steps")
if PACKER_DIAGNOSTIC_HASHES and (
    PACKER_BOUNDARY_VERIFY_BATCHES != 1000
    or PACKER_SIDECAR_ACTIVATE_STEP != 20
    or MAX_STEPS != 250
):
    raise ValueError(
        "PACKER_DIAGNOSTIC_HASHES=1 requires the governed parity contract: "
        "PACKER_BOUNDARY_VERIFY_BATCHES=1000 "
        "PACKER_SIDECAR_ACTIVATE_STEP=20 MAX_STEPS=250"
    )
# ====================== RSI FIXED-COMPUTE STOP: END ======================

# ===================== RSI ENV KNOB OVERRIDES: BEGIN =====================
# Sweep knobs via environment without editing code. Every default is the value
# above, so behavior is unchanged unless a variable is explicitly set. Enables
# clean, reversible env-driven experiments.
def _envf(_name, _default):
    _v = os.environ.get(_name)
    return float(_v) if _v is not None and _v != "" else _default
MATRIX_LR = _envf("MATRIX_LR", MATRIX_LR)
NGRAM_VE_LR_SCALE = _envf("NGRAM_VE_LR_SCALE", NGRAM_VE_LR_SCALE)
FINAL_LR_FRAC = _envf("FINAL_LR_FRAC", FINAL_LR_FRAC)
WEIGHT_DECAY = _envf("WEIGHT_DECAY", WEIGHT_DECAY)
EMBEDDING_LR = _envf("EMBEDDING_LR", EMBEDDING_LR)
UNEMBEDDING_LR = _envf("UNEMBEDDING_LR", UNEMBEDDING_LR)
SCALAR_LR = _envf("SCALAR_LR", SCALAR_LR)
WARMDOWN_RATIO = _envf("WARMDOWN_RATIO", WARMDOWN_RATIO)
WARMUP_RATIO = _envf("WARMUP_RATIO", WARMUP_RATIO)
ADAM_WARMDOWN_RATIO = _envf("ADAM_WARMDOWN_RATIO", ADAM_WARMDOWN_RATIO)
DEMON_FINAL_BETA1 = _envf("DEMON_FINAL_BETA1", DEMON_FINAL_BETA1)
_ve_b2 = os.environ.get("NGRAM_VE_BETA2")
if _ve_b2 not in (None, ""):
    NGRAM_VE_BETAS = (NGRAM_VE_BETAS[0], float(_ve_b2))
# ===== EXPLORE:RHO1_GAMMA BEGIN =====
# Rho-1-inspired online focal reweighting of the training CE. Default 0.0 keeps
# the original F.cross_entropy(reduction="mean") path byte-identical. Values >0
# upweight high-loss (hard) tokens; see the guarded branch in GPT.forward.
RHO1_GAMMA = _envf("RHO1_GAMMA", 0.0)
# ===== EXPLORE:RHO1_GAMMA END =====
# ===== EXPLORE:LR_SCALE BEGIN =====
# Global learning-rate rescale for BUDGET-TRANSFER testing. All LRs (and the 43
# config beliefs) were tuned at the 1660-step budget; the decision budget is
# 2766 steps. Optimal peak LR shrinks with training length (~1/sqrt(T) heuristic
# -> ~0.775x for 1660->2766). LR_SCALE multiplies ALL base LRs coherently
# (matrix/embedding/unembedding/scalar; ngram-VE follows embedding) so the tuned
# RATIOS are preserved and only the global magnitude moves. Default 1.0 takes
# the guarded branch not at all -> byte-identical baseline.
LR_SCALE = _envf("LR_SCALE", 1.0)
if LR_SCALE != 1.0:
    MATRIX_LR *= LR_SCALE
    EMBEDDING_LR *= LR_SCALE
    UNEMBEDDING_LR *= LR_SCALE
    SCALAR_LR *= LR_SCALE
# ===== EXPLORE:LR_SCALE END =====
# ===== EXPLORE:NGRAM_TABLE_MULT BEGIN =====
# Rare-token / n-gram CAPACITY lever. The factored bigram & trigram value-embedding
# hash tables are sized vocab_size * NGRAM_TABLE_MULT buckets. More buckets -> fewer
# hash collisions -> more capacity to memorize rare bigrams/trigrams, which is where
# headroom plausibly lives in the repeated-data (~1.6-2.7 epoch) regime. Default 64
# is byte-identical to the frozen baseline (same integer -> identical table_size,
# init, and hashing). Embedding lookups are O(1) gather, so larger tables add
# PARAMETERS and VRAM/optimizer state but NOT per-step matmul FLOPs: compute-fair at
# the fixed 2000-step budget (report the parameter delta as a capacity, not a
# compute, change). Threaded through GPTConfig.ngram_table_mult.
NGRAM_TABLE_MULT = int(os.environ.get("NGRAM_TABLE_MULT", "64"))
if NGRAM_TABLE_MULT < 1:
    raise ValueError("NGRAM_TABLE_MULT must be a positive integer")
# Per-type capacity overrides (0 = inherit NGRAM_TABLE_MULT). Used to decompose the
# capacity win into its bigram vs trigram contributions. Default 0/0 -> byte-identical.
NGRAM_BIGRAM_MULT = int(os.environ.get("NGRAM_BIGRAM_MULT", "0"))
NGRAM_TRIGRAM_MULT = int(os.environ.get("NGRAM_TRIGRAM_MULT", "0"))
if NGRAM_BIGRAM_MULT < 0 or NGRAM_TRIGRAM_MULT < 0:
    raise ValueError("NGRAM_BIGRAM_MULT / NGRAM_TRIGRAM_MULT must be >= 0 (0 = inherit)")
# 4-gram VE order: 0 = OFF (byte-identical baseline); >0 sizes the 4-gram tables vocab*N.
NGRAM_FOURGRAM_MULT = int(os.environ.get("NGRAM_FOURGRAM_MULT", "0"))
if NGRAM_FOURGRAM_MULT < 0:
    raise ValueError("NGRAM_FOURGRAM_MULT must be >= 0 (0 = off)")
# ===== EXPLORE:NGRAM_FIVEGRAM BEGIN =====
# 5-gram VE order: 0 = OFF (byte-identical baseline); >0 sizes the 5-gram tables vocab*N.
NGRAM_FIVEGRAM_MULT = int(os.environ.get("NGRAM_FIVEGRAM_MULT", "0"))
if NGRAM_FIVEGRAM_MULT < 0:
    raise ValueError("NGRAM_FIVEGRAM_MULT must be >= 0 (0 = off)")
# ===== EXPLORE:NGRAM_FIVEGRAM END =====
# ===== EXPLORE:NGRAM_FOURGRAM_SPAN BEGIN =====
# GAPPY (skip-gram) 4-gram context shape. The 4-gram VE normally hashes the
# CONTIGUOUS window (w[-3], w[-2], w[-1], w[0]). This lever moves ONLY the
# furthest slot to offset -S, giving (w[-S], w[-2], w[-1], w[0]).
#
# WHY: the 5-gram rung (which extends the contiguous window to w[-4]) failed its
# funnel at -0.000470. Two rival explanations:
#   (a) SPARSITY -- a 5-token conjunction is too rare at 294.9M tokens for a
#       hashed row to accumulate enough gradient;
#   (b) RANGE DECAY -- w[-4] simply carries little marginal information.
# The gappy 4-gram is a DOSE-RESPONSE probe on the measured sparsity mediator.
#
# NOTE: it is NOT sparsity-matched to the contiguous 4-gram. That was the original
# design premise and the obs_ngram_context_repeat_rate observable REFUTED it --
# skipping a position decorrelates the tokens, so distinct-key count rises even at
# equal token count. Measured over the exact 294.9M-token training stream:
#     contiguous 4-gram  singleton 0.4576   median_occ 2
#     gappy (-4,-2,-1,0) singleton 0.6030   median_occ 1
#     contiguous 5-gram  singleton 0.6992   median_occ 1
# So gappy4 sits ~60% of the way from the 4-gram to the 5-gram in sparsity, which
# makes it an INTERMEDIATE DOSE rather than a matched control.
#
# That still discriminates (a) from (b), just as a graded prediction: if singleton
# rate is the mediator, the rung's value should interpolate between the measured
# 4-gram and 5-gram rung values, i.e. gappy4 should come in MODESTLY WORSE than the
# contiguous 4-gram (~+0.0004), not catastrophically so. If instead gappy4 loses
# most of the 4-gram's contribution (~+0.0012, i.e. as bad as no 4-gram at all),
# sparsity is not the whole story and the ADJACENT position w[-3] carries specific
# information that w[-4] cannot replace => RANGE DECAY (b).
# S=3 is the default and reproduces the contiguous 4-gram EXACTLY (prev3_idx),
# so this is byte-identical unless explicitly set.
NGRAM_FOURGRAM_SPAN = int(os.environ.get("NGRAM_FOURGRAM_SPAN", "3"))
if NGRAM_FOURGRAM_SPAN < 3:
    raise ValueError("NGRAM_FOURGRAM_SPAN must be >= 3 (3 = contiguous 4-gram, the default)")
# ===== EXPLORE:NGRAM_FOURGRAM_SPAN END =====
# vora-inspired structural simplification: share one trigram VE table across all trigram layers.
SHARED_TRIGRAM_VE = os.environ.get("SHARED_TRIGRAM_VE", "0") == "1"
# ===== EXPLORE:NGRAM_TABLE_MULT END =====
# ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
# PRIMARY ARCHITECTURE BET: learned-retrieval product-key memory on the trigram
# layers (see GPT.__init__ for the design rationale: random-hash capacity is
# played out, so this memory's justification is LEARNED routing of related
# contexts to shared slots, not more buckets). Default 0 -> byte-identical
# baseline: no modules registered, no tensor ops, no param-count change.
#   NGRAM_PK_MEMORY: 0|1 master switch.
#   PK_MEM_SUBKEYS:  sub-key codebook size n_sub; N = n_sub^2 slots
#                    (default 2048 -> N = 4,194,304; sized to fit alongside the
#                    256x trigram tables on a 141 GB H200, see report/VRAM math).
#   PK_MEM_TOPK:     retrieved slots per token per layer (k' per codebook = k,
#                    cartesian k^2 candidates -> top-k). Sensible range 8-32.
#   PK_MEM_QDIM:     query dim (split into two qdim/2 sub-queries). Must be even.
#   PK_MEM_LR_SCALE: LR multiplier for the PK value/retrieval groups relative to
#                    the (aggressive, 0.6) embedding-LR base they inherit.
_pk_mem_raw = os.environ.get("NGRAM_PK_MEMORY", "0")
if _pk_mem_raw not in {"0", "1"}:
    raise ValueError("NGRAM_PK_MEMORY must be 0 or 1")
NGRAM_PK_MEMORY = _pk_mem_raw == "1"
PK_MEM_SUBKEYS = int(os.environ.get("PK_MEM_SUBKEYS", "2048"))
PK_MEM_TOPK = int(os.environ.get("PK_MEM_TOPK", "16"))
PK_MEM_QDIM = int(os.environ.get("PK_MEM_QDIM", "128"))
PK_MEM_LR_SCALE = float(os.environ.get("PK_MEM_LR_SCALE", "1.0"))
if PK_MEM_SUBKEYS < 2:
    raise ValueError("PK_MEM_SUBKEYS must be >= 2")
if not (1 <= PK_MEM_TOPK <= PK_MEM_SUBKEYS):
    raise ValueError("PK_MEM_TOPK must be in [1, PK_MEM_SUBKEYS]")
if PK_MEM_QDIM < 2 or PK_MEM_QDIM % 2 != 0:
    raise ValueError("PK_MEM_QDIM must be a positive even integer")
if PK_MEM_LR_SCALE <= 0.0:
    raise ValueError("PK_MEM_LR_SCALE must be > 0")
# ===== EXPLORE:NGRAM_PK_MEMORY END =====

# ===== RSI LR WARMUP: BEGIN =====
# WARMUP_RATIO was resolved from the environment above but, until this block
# existed, no schedule code read it -- it reached RESOLVED_CONFIG and nothing
# else. Runs that set it trained the identical no-warmup schedule. See
# evd_run_ngram256_seg_warmup002_retracted in run_evidence.jsonl.
if not (0.0 <= WARMUP_RATIO < 1.0):
    raise ValueError("WARMUP_RATIO must be in [0.0, 1.0)")
_earliest_warmdown_start = 1.0 - max(WARMDOWN_RATIO, ADAM_WARMDOWN_RATIO)
if WARMUP_RATIO > _earliest_warmdown_start:
    raise ValueError(
        f"WARMUP_RATIO={WARMUP_RATIO} overlaps the warmdown phase starting at "
        f"{_earliest_warmdown_start}; warmup and warmdown must not overlap"
    )
WARMUP_ACTIVE = WARMUP_RATIO > 0.0


def warmup_lr_mult(progress, warmup_ratio, floor):
    """LR multiplier for the warmup phase: floor -> 1.0 linearly over the window.

    Returns exactly 1.0 once progress leaves the window, and for every progress
    when warmup_ratio == 0.0, so the baseline schedule is unchanged.
    """
    if warmup_ratio <= 0.0 or progress >= warmup_ratio:
        return 1.0
    return floor + (1.0 - floor) * (progress / warmup_ratio)
# ===== RSI LR WARMUP: END =====
# ====================== RSI ENV KNOB OVERRIDES: END ======================

# ===================== TRACK B: STEADY-STATE STEP TIMING =====================
# Track B measures the synchronized training-step core (forward, backward,
# optimizer, and in-loop data fetch). It intentionally excludes startup/model
# construction, compilation warmup, logging, periodic validation probes, and
# final evaluation. Therefore this is NOT end-to-end wall-clock timing.
#
# For a fair equal-training-time diagnostic, run every configuration directly
# with the same STOP_MODE=time and TIME_BUDGET, TRACK_B_MODE=1, and both
# validation cadences disabled. Do not infer an equal-time endpoint from a
# mid-run probe or describe same-MAX_STEPS runs as compute-matched: same steps
# fix tokens, while architectures may perform different work per token.
#
# When OFF (default), identity with the pre-Track-B loop is STRUCTURAL:
# no extra ops, host syncs, or capture sites are added (the two segment
# syncs below are gated on TRACK_B_MODE).  Do not read this as bitwise
# reproducibility -- this pipeline is not bitwise deterministic run-to-run
# even at fixed seed (see research/CAMPAIGN_SUMMARY.md, integrity finding 4).
TRACK_B_MODE = os.environ.get("TRACK_B_MODE", "0") == "1"
VAL_BPB_PROBE_EVERY = int(os.environ.get("VAL_BPB_PROBE_EVERY", "100"))  # 0 disables
EVAL_TOKENS_PROBE = int(os.environ.get("EVAL_TOKENS_PROBE", str(2 * 524288)))  # ~1M tokens
VAL_LOSS_EVERY = int(os.environ.get("VAL_LOSS_EVERY", "1"))  # 0 disables telemetry-only CE probes
OBSERVE_LAYER_PROBES = os.environ.get("OBSERVE_LAYER_PROBES", "0") == "1"
if VAL_BPB_PROBE_EVERY < 0:
    raise ValueError("VAL_BPB_PROBE_EVERY must be >= 0")
if EVAL_TOKENS_PROBE <= 0:
    raise ValueError("EVAL_TOKENS_PROBE must be > 0")
if VAL_LOSS_EVERY < 0:
    raise ValueError("VAL_LOSS_EVERY must be >= 0")
if TRACK_B_MODE:
    # These probes execute outside the timed core but perturb process wall time,
    # cache/thermal state, and the next measured step. Keep them off unless the
    # caller explicitly requests the lightweight BPB curve.
    VAL_LOSS_EVERY = 0
    OBSERVE_LAYER_PROBES = False
# ========================================================================

# Echo the EFFECTIVE resolved config so downstream verdict tooling can confirm a
# treatment actually differs from its baseline (guards against typo'd env vars
# silently running the baseline and being recorded as a rigorous NULL).
print(
    "RESOLVED_CONFIG: "
    f"STOP_MODE={STOP_MODE} TIME_BUDGET={TIME_BUDGET} MAX_STEPS={MAX_STEPS} SEED={SEED} ATTN_BACKEND={ATTN_BACKEND} "
    f"WINDOW_PATTERN={WINDOW_PATTERN} DEVICE_BATCH_SIZE={DEVICE_BATCH_SIZE} "
    f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} "
    f"MATRIX_LR={MATRIX_LR} NGRAM_VE_LR_SCALE={NGRAM_VE_LR_SCALE} "
    f"NGRAM_VE_BETA={NGRAM_VE_BETAS[1]} FINAL_LR_FRAC={FINAL_LR_FRAC} "
    f"WEIGHT_DECAY={WEIGHT_DECAY} EMBEDDING_LR={EMBEDDING_LR} "
    f"UNEMBEDDING_LR={UNEMBEDDING_LR} SCALAR_LR={SCALAR_LR} "
    f"WARMDOWN_RATIO={WARMDOWN_RATIO} ADAM_WARMDOWN_RATIO={ADAM_WARMDOWN_RATIO} "
    f"DEMON_FINAL_BETA1={DEMON_FINAL_BETA1} "
    f"MLP_TYPE={MLP_TYPE} WARMUP_RATIO={WARMUP_RATIO} "
    f"CAUTIOUS_UPDATE={CAUTIOUS_UPDATE} CAUTIOUS_MUON={CAUTIOUS_MUON} "
    # ===== EXPLORE: exploration levers resolved values =====
    f"TIE_EMBED={TIE_EMBED} RHO1_GAMMA={RHO1_GAMMA} LR_SCALE={LR_SCALE} "
    f"RESID_SCALE={RESID_SCALE} DOC_MASK={DOC_MASK} DOC_MASK_MODE={DOC_MASK_MODE} "
    f"DOC_MASK_IMPL={DOC_MASK_IMPL} PACKER_DOC_BOUNDARIES={PACKER_DOC_BOUNDARIES} "
    f"PACKER_BOUNDARY_VERIFY_BATCHES={PACKER_BOUNDARY_VERIFY_BATCHES} "
    f"PACKER_DIAGNOSTIC_HASHES={PACKER_DIAGNOSTIC_HASHES} "
    f"PACKER_SIDECAR_ACTIVATE_STEP={PACKER_SIDECAR_ACTIVATE_STEP} "
    f"RESET_ROPE={RESET_ROPE} NGRAM_TABLE_MULT={NGRAM_TABLE_MULT} "
    f"NGRAM_BIGRAM_MULT={NGRAM_BIGRAM_MULT} NGRAM_TRIGRAM_MULT={NGRAM_TRIGRAM_MULT} NGRAM_FOURGRAM_MULT={NGRAM_FOURGRAM_MULT} "
    # ===== EXPLORE:NGRAM_FIVEGRAM resolved value (harness verifies env against this line) =====
    f"NGRAM_FIVEGRAM_MULT={NGRAM_FIVEGRAM_MULT} "
    f"NGRAM_FOURGRAM_SPAN={NGRAM_FOURGRAM_SPAN} "
    f"SHARED_TRIGRAM_VE={SHARED_TRIGRAM_VE} "
    f"SHUFFLE_DATA={os.environ.get('SHUFFLE_DATA', '0')} "
    f"VAL_LOSS_EVERY={VAL_LOSS_EVERY} OBSERVE_LAYER_PROBES={OBSERVE_LAYER_PROBES} "
    f"LM_HEAD_INIT_STD={LM_HEAD_INIT_STD} SOFTCAP_CAP={SOFTCAP_CAP} SOFTCAP_TAU={SOFTCAP_TAU} "
    f"NGRAM_STATE_ROWWISE={NGRAM_STATE_ROWWISE} COMPILE_MODE={COMPILE_MODE} "
    f"MUON_MOMENTUM_CONTINUOUS={MUON_MOMENTUM_CONTINUOUS} "
    f"MUON_NS_STEPS={os.environ.get('MUON_NS_STEPS','5')} ADAM_BETA2={os.environ.get('ADAM_BETA2','0.95')} NGRAM_VE_EPS={os.environ.get('NGRAM_VE_EPS','1e-10')} "
    f"BF16_CE={BF16_CE} "
    f"CANON={CANON} CANON_K={CANON_K} CANON_LR={os.environ.get('CANON_LR','0.02')} "
    f"NGRAM_SPARSE_GRAD={NGRAM_SPARSE_GRAD} "
    # ===== EXPLORE:NGRAM_PK_MEMORY resolved values =====
    f"NGRAM_PK_MEMORY={NGRAM_PK_MEMORY} PK_MEM_SUBKEYS={PK_MEM_SUBKEYS} "
    f"PK_MEM_TOPK={PK_MEM_TOPK} PK_MEM_QDIM={PK_MEM_QDIM} PK_MEM_LR_SCALE={PK_MEM_LR_SCALE} "
    # ===== TRACK B =====
    f"TRACK_B_MODE={TRACK_B_MODE} VAL_BPB_PROBE_EVERY={VAL_BPB_PROBE_EVERY} "
    f"EVAL_TOKENS_PROBE={EVAL_TOKENS_PROBE} "
    # ===== RSI INTERVENTION: BLOCK-14 MECHANISM PORT =====
    # These MUST be echoed: the gated runner's config-verify step compares each
    # manifest env key against RESOLVED_CONFIG. A flag that train.py never echoes
    # cannot be verified as consumed -- which is exactly how the pre-port
    # ATTNRES_ENABLE / NGRAM_WD_LAMBDA / NGRAM_BACKOFF_KAPPA "treatments" passed
    # config_verified=True while silently running the control config.
    f"ATTNRES_ENABLE={ATTNRES_ENABLE} NGRAM_WD_LAMBDA={NGRAM_WD_LAMBDA} "
    f"NGRAM_BACKOFF_KAPPA={NGRAM_BACKOFF_KAPPA} TOKEN_SHIFT={TOKEN_SHIFT} "
    f"NGPT_SPHERE={NGPT_SPHERE} NGPT_ALPHA_INIT={NGPT_ALPHA_INIT} "
    f"GPAS_ENABLE={GPAS_ENABLE} DOC_MASK_BLOCK={DOC_MASK_BLOCK}",
    flush=True,
)

# Model size
DEPTH = 8  # number of transformer layers
if TRACK_B_MODE:
    print(
        "TRACK_B_MODE=1: synchronized steady-state training-step timing. "
        "Layer probes and per-step val-loss probes disabled; periodic BPB "
        f"probe cadence={VAL_BPB_PROBE_EVERY}.",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
# No autocast: model is natively BF16 -- eliminates FP32->BF16 cast overhead in compile graph
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False)
B200_BF16_PEAK_FLOPS = 2.25e15

# Data check log: print the exact split and parquet files used by the
# train/val dataloaders before training starts.
data_split = load_data_split()
print(f"Data split train ids: {data_split['train']}", flush=True)
print(f"Data split test ids: {data_split['test']}", flush=True)
print(f"Train parquet files: {split_parquet_files('train')}", flush=True)
print(f"Test/val parquet files: {split_parquet_files('test')}", flush=True)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# ===== EXPLORE:DOC_MASK BEGIN =====
# Resolve the BOS token id used to derive per-token document segment ids. Only
# consulted inside GPT.forward when DOC_MASK=1 or RESET_ROPE=1; harmless to set
# unconditionally.
DOC_MASK_BOS_ID = tokenizer.get_bos_token_id()
# ===== EXPLORE:DOC_MASK END =====


def _hash_tensor_bytes(hasher, label, tensor):
    """Update a canonical digest with tensor metadata and exact storage bytes."""
    source = tensor.detach()
    hasher.update(label.encode("utf-8"))
    hasher.update(str(source.dtype).encode("ascii"))
    hasher.update(str(source.layout).encode("ascii"))
    hasher.update(struct.pack("<I", source.ndim))
    for dim in source.shape:
        hasher.update(struct.pack("<Q", dim))
    for stride in source.stride():
        hasher.update(struct.pack("<q", stride))
    value = source.contiguous()
    raw = value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    hasher.update(struct.pack("<Q", len(raw)))
    hasher.update(raw)


def _canonical_state_sha256(value):
    """Hash nested model/optimizer state without pickle or object identities."""
    hasher = hashlib.sha256()

    def update(item):
        if torch.is_tensor(item):
            hasher.update(b"tensor")
            _hash_tensor_bytes(hasher, "", item)
        elif isinstance(item, dict):
            hasher.update(b"dict")
            ordered = sorted(item.items(), key=lambda pair: (type(pair[0]).__name__, repr(pair[0])))
            hasher.update(struct.pack("<Q", len(ordered)))
            for key, child in ordered:
                update(key)
                update(child)
        elif isinstance(item, (list, tuple)):
            hasher.update(b"list" if isinstance(item, list) else b"tuple")
            hasher.update(struct.pack("<Q", len(item)))
            for child in item:
                update(child)
        elif item is None:
            hasher.update(b"none")
        elif isinstance(item, bool):
            hasher.update(b"bool1" if item else b"bool0")
        elif isinstance(item, int):
            hasher.update(b"int")
            hasher.update(str(item).encode("ascii"))
            hasher.update(b"\0")
        elif isinstance(item, float):
            hasher.update(b"float")
            hasher.update(item.hex().encode("ascii"))
            hasher.update(b"\0")
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            hasher.update(b"str")
            hasher.update(struct.pack("<Q", len(encoded)))
            hasher.update(encoded)
        else:
            raise TypeError(
                "PACKER_DIAGNOSTIC_UNSUPPORTED_STATE_TYPE "
                f"{type(item).__module__}.{type(item).__qualname__}"
            )

    update(value)
    return hasher.hexdigest()


def _assert_boundary_vector_invariants(cu_cpu, B, T, batch_index):
    """Host-side exact invariants for one scanner/sidecar boundary vector."""
    if cu_cpu.dtype != torch.int32 or cu_cpu.ndim != 1 or cu_cpu.numel() < B + 1:
        raise RuntimeError(
            "PACKER_BOUNDARY_VERIFY_SHAPE_FAILED "
            f"batch={batch_index} dtype={cu_cpu.dtype} "
            f"shape={tuple(cu_cpu.shape)}"
        )
    diffs = cu_cpu[1:] - cu_cpu[:-1]
    row_starts = torch.arange(B, dtype=torch.int32) * T
    checks = {
        "first_zero": bool(cu_cpu[0].item() == 0),
        "terminal": bool(cu_cpu[-1].item() == B * T),
        "strictly_increasing": bool(torch.all(diffs > 0).item()),
        "row_starts": bool(torch.all(torch.isin(row_starts, cu_cpu[:-1])).item()),
        "max_gap": bool(torch.all(diffs <= T).item()),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "PACKER_BOUNDARY_VERIFY_INVARIANT_FAILED "
            f"batch={batch_index} "
            + " ".join(f"{key}={int(ok)}" for key, ok in checks.items())
        )


def verify_packer_boundary_equivalence(tokenizer, B, T, split, batches):
    """Pre-clock exact equivalence gate over two independent loader instances.

    Every arm that sets ``PACKER_BOUNDARY_VERIFY_BATCHES=1000`` pays this same
    scanner+sidecar construction, comparison, synchronization, and hashing
    burden, independently of whether training later consumes scanner or sidecar
    boundaries.
    """
    if batches != 1000:
        raise RuntimeError("PACKER_BOUNDARY_VERIFY_BATCHES must equal 1000")
    print(
        "PACKER_BOUNDARY_VERIFY_START=1 "
        f"batches={batches} split={split} B={B} T={T}",
        flush=True,
    )
    scanner_loader = make_dataloader(tokenizer, B, T, split)
    sidecar_loader = make_dataloader(
        tokenizer, B, T, split, return_doc_boundaries=True
    )
    data_hasher = hashlib.sha256()
    boundary_hasher = hashlib.sha256()
    for batch_index in range(batches):
        x_scan, y_scan, epoch_scan = next(scanner_loader)
        x_side, y_side, epoch_side, boundary_lease = next(sidecar_loader)
        if epoch_scan != epoch_side:
            raise RuntimeError(
                "PACKER_BOUNDARY_VERIFY_EPOCH_MISMATCH "
                f"batch={batch_index} scanner={epoch_scan} sidecar={epoch_side}"
            )
        if not torch.equal(x_scan, x_side) or not torch.equal(y_scan, y_side):
            raise RuntimeError(
                "PACKER_BOUNDARY_VERIFY_DATA_MISMATCH "
                f"batch={batch_index} epoch={epoch_scan}"
            )

        sidecar_cu = boundary_lease.cu_seqlens
        scanner_cu, scanner_max_seqlen = _compact_doc_boundaries(x_scan)
        sidecar_cpu = sidecar_cu.detach().cpu()
        scanner_cpu = scanner_cu.detach().cpu()
        _assert_boundary_vector_invariants(scanner_cpu, B, T, batch_index)
        _assert_boundary_vector_invariants(sidecar_cpu, B, T, batch_index)
        if (
            boundary_lease.max_seqlen != scanner_max_seqlen
            or not torch.equal(scanner_cpu, sidecar_cpu)
        ):
            raise RuntimeError(
                "PACKER_BOUNDARY_INTERIOR_BOS_MISMATCH "
                f"batch={batch_index} scanner_count={scanner_cpu.numel() - 1} "
                f"sidecar_count={sidecar_cpu.numel() - 1}"
            )

        _hash_tensor_bytes(data_hasher, "x", x_scan)
        _hash_tensor_bytes(data_hasher, "y", y_scan)
        data_hasher.update(struct.pack("<q", epoch_scan))
        _hash_tensor_bytes(boundary_hasher, "cu", scanner_cpu)
        boundary_lease.mark_consumed()

        if (batch_index + 1) % 100 == 0:
            print(
                "PACKER_BOUNDARY_VERIFY_PROGRESS=1 "
                f"batches={batch_index + 1} "
                f"data_sha256={data_hasher.copy().hexdigest()} "
                f"boundary_sha256={boundary_hasher.copy().hexdigest()}",
                flush=True,
            )

    torch.cuda.synchronize()
    print(
        "PACKER_BOUNDARY_VERIFY_PASSED=1 "
        f"batches={batches} data_sha256={data_hasher.hexdigest()} "
        f"boundary_sha256={boundary_hasher.hexdigest()} "
        "x_exact=1 y_exact=1 epoch_exact=1 boundaries_exact=1 "
        "first_zero=1 terminal=1 strictly_increasing=1 "
        "row_starts=1 max_gap_lte_t=1",
        flush=True,
    )


if PACKER_BOUNDARY_VERIFY_BATCHES:
    verify_packer_boundary_equivalence(
        tokenizer,
        DEVICE_BATCH_SIZE,
        MAX_SEQ_LEN,
        "train",
        PACKER_BOUNDARY_VERIFY_BATCHES,
    )


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        gpas=GPAS_ENABLE,
        ngram_table_mult=NGRAM_TABLE_MULT,
        ngram_bigram_mult=NGRAM_BIGRAM_MULT,
        ngram_trigram_mult=NGRAM_TRIGRAM_MULT,
        ngram_fourgram_mult=NGRAM_FOURGRAM_MULT,
        ngram_fourgram_span=NGRAM_FOURGRAM_SPAN,
        # ===== EXPLORE:NGRAM_FIVEGRAM (default 0 reproduces baseline) =====
        ngram_fivegram_mult=NGRAM_FIVEGRAM_MULT,
        shared_trigram_ve=SHARED_TRIGRAM_VE,
        # ===== EXPLORE:NGRAM_PK_MEMORY (defaults reproduce baseline when off) =====
        pk_mem=NGRAM_PK_MEMORY,
        pk_mem_subkeys=PK_MEM_SUBKEYS,
        pk_mem_topk=PK_MEM_TOPK,
        pk_mem_qdim=PK_MEM_QDIM,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
# Cast entire model to BF16: enables removing autocast, simplifies compile graph
model.to(dtype=torch.bfloat16)

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# ===== EXPLORE:NGRAM_SPARSE_GRAD BEGIN =====
# Preallocate (indices, grad_rows) capture buffers for every n-gram VE gather
# site. Runs after the bf16 cast (buffers match weight dtype) and before
# torch.compile below (the buffer dict must be final when the graph is traced).
if NGRAM_SPARSE_GRAD:
    assert grad_accum_steps == 1, (
        "NGRAM_SPARSE_GRAD=1 requires grad_accum_steps==1: the capture buffers "
        "hold exactly one micro-batch and accumulation would overwrite them"
    )
    _ngram_sparse_setup(model, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, device)
# ===== EXPLORE:NGRAM_SPARSE_GRAD END =====

# ===== EXPLORE:NGRAM_PK_MEMORY BEGIN =====
# Size echo for the PK memory (guarded: no output when off, keeping baseline
# stdout byte-identical). num_scaling_params deliberately excludes the PK
# tables, matching how the bigram/trigram hash tables are excluded from the
# scaling-param accounting.
if NGRAM_PK_MEMORY:
    _pk_total = sum(p.numel() for p in model.pk_mems.parameters())
    _pk_gates = sum(
        b.attn.pk_gate.weight.numel()
        for b in model.transformer.h
        if b.attn.pk_gate is not None
    )
    print(
        f"NGRAM_PK_MEMORY: layers={sorted(model.pk_mem_layers)} "
        f"N={model.pk_mem_n_slots:,} slots/layer (n_sub={PK_MEM_SUBKEYS}) "
        f"topk={PK_MEM_TOPK} qdim={PK_MEM_QDIM} "
        f"params={_pk_total + _pk_gates:,} (~{(_pk_total + _pk_gates) * 2 / 1e9:.2f} GB bf16)",
        flush=True,
    )
# ===== EXPLORE:NGRAM_PK_MEMORY END =====

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    ngram_ve_betas=NGRAM_VE_BETAS,
    ngram_ve_lr_scale=NGRAM_VE_LR_SCALE,
)

muon_groups = []
ngram_groups = []
x0_warmdown_groups = []
gpas_groups = []
adam_groups = []
adam_demon_groups = []
muon_group_lrs = []
x0_group_lrs = []
adam_group_lrs = []
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        muon_groups.append(group)
        muon_group_lrs.append((group, group["initial_lr"]))
    elif group.get("is_ngram_ve", False):
        ngram_groups.append(group)
    elif group.get("is_x0_muon_warmdown", False):
        x0_warmdown_groups.append(group)
        x0_group_lrs.append((group, group["initial_lr"]))
    elif group.get("is_gpas", False):
        # Paper-020 freezes GPAS at 0.005: no warmup, warmdown, or Demon.
        gpas_groups.append(group)
    else:
        adam_groups.append(group)
        adam_group_lrs.append((group, group["initial_lr"]))
        if group.get("demon_beta1", False):
            adam_demon_groups.append((group, group["betas"][1]))

# ===== EXPLORE:DOC_MASK BEGIN =====
# Captured BEFORE torch.compile rebinds `model` to the OptimizedModule wrapper,
# so the doc-mask builders never reach through the compile wrapper.
WINDOW_LEFTS = [w[0] for w in model.window_sizes]


def _doc_mask_forward_kwargs(idx):
    """Extra model kwargs for an eval batch (empty unless DOC_MASK_IMPL=flex)."""
    masks = build_doc_masks_for_batch(idx, WINDOW_LEFTS)
    return {} if masks is None else {"doc_masks": masks}
# ===== EXPLORE:DOC_MASK END =====

if OBSERVE_LAYER_PROBES:
    print("Observable layer probes enabled; torch.compile disabled for Python probe collection.")
else:
    model = torch.compile(model, dynamic=False, mode=COMPILE_MODE, fullgraph=True)

_diagnostic_inside_model_region = False
_diagnostic_callback_site = ""
_diagnostic_compile_events: list[tuple[str, str, str]] = []
_diagnostic_compile_callback = None
if PACKER_DIAGNOSTIC_HASHES:
    from torch._dynamo.callback import callback_handler

    def _record_diagnostic_compile_event(callback_args):
        if _diagnostic_inside_model_region:
            _diagnostic_compile_events.append(
                (
                    callback_args.callback_trigger.name,
                    str(callback_args.compile_id),
                    _diagnostic_callback_site,
                )
            )

    _diagnostic_compile_callback = (
        callback_handler.register_start_callback(
            _record_diagnostic_compile_event
        )
    )

train_loader = make_dataloader(
    tokenizer,
    DEVICE_BATCH_SIZE,
    MAX_SEQ_LEN,
    "train",
    return_doc_boundaries=PACKER_DOC_BOUNDARIES,
)
val_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")
if PACKER_DOC_BOUNDARIES:
    x, y, epoch, boundary_lease = next(train_loader)  # prefetch first batch
else:
    x, y, epoch = next(train_loader)  # exact legacy triple
    boundary_lease = None
if ATTN_BACKEND == "fa3" and DOC_MASK_IMPL == "varlen":
    # Symmetric, pre-clock, fail-closed attestation for both explicit arms.
    attest_compact_doc_boundaries(x)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)


# NOTE: the LR / momentum / beta2 / weight-decay schedules are computed INLINE in the
# training loop (single source of truth). Earlier module-level get_* schedule functions
# were removed 2026-07-21: they were never called and had drifted from the loop (a
# silent-misconfiguration trap). Edit the inlined loop schedule, not a separate copy.
MUON_PEAK_MOMENTUM = 0.95  # standard peak
MUON_WARMDOWN_MOMENTUM = 0.79  # testing VE beta2 ramp alone
# Reverse Demon for NorMuon beta2: INCREASE beta2 during warmdown for more stable variance normalization
MUON_BETA2_PEAK = 0.95  # standard beta2 during full-LR phase
MUON_BETA2_WARMDOWN = 0.97  # locally bracketed terminal Muon beta2
MUON_LR_BOOST = 1.0  # no LR boost
# VE RMSProp reverse-Demon: increase VE beta2 during last 30% of Muon warmdown
# Analogous to Muon's 0.95->0.97, but for ngram VE tables (0.999->0.9995)
NGRAM_VE_BETA2_WARMDOWN = 0.9999  # validated delayed sparse beta2 warmdown baseline
# WD pulse: RECTANGULAR shape -- with 95% warmdown (starts at 5%), pulses shifted earlier
# Main pulse at 3% center, 2% total duration (1% half-width): fires at 2-4%, before warmdown onset at 5%
# Early pulse at 1.5% center, 1% total duration: fires at 1-2% progress
# Both pulses fire in the full-LR phase (0-5%), maintaining the pre-warmdown regularization timing
WD_PULSE_CENTER = 0.03   # shift main pulse to 3% (fires before warmdown at 5%)
WD_PULSE_HALF_WIDTH = 0.01  # 1% half-width: 2% total duration (tighter for earlier firing)
WD_PULSE_MAGNITUDE = 5.0  # try 5x main pulse (vs 8x) -- 5x optimal WITH Muon Demon, 8x WITHOUT; current setup HAS Demon
WD_EARLY_PULSE_CENTER = 0.015  # shift early pulse to 1.5%
WD_EARLY_PULSE_HALF_WIDTH = 0.005  # 0.5% half-width: 1% total duration
WD_EARLY_PULSE_MAGNITUDE = 3.0  # 3x early pulse (gentler, to initialize regularization)
# Mid-warmdown triangular pulse: fires at 80% total progress (= ~79% through warmdown)
# This is WITHIN the VE beta2 ramp zone (which starts at 71.5% total = 70% through warmdown)
# Hypothesis: VE beta2 stabilization provides a safety net for a mid-warmdown WD perturbation
WD_MID_PULSE_CENTER = 0.80   # 80% total progress = ~79% through warmdown
WD_MID_PULSE_HALF_WIDTH = 0.025  # 2.5% half-width: 5% total triangular duration
WD_MID_PULSE_MAGNITUDE = 4.0  # 4x magnitude (triangular shape -- less harsh than rectangular)

# (weight-decay pulse schedule is computed inline in the training loop)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
inv_time_budget = 1.0 / TIME_BUDGET
inv_muon_warmdown = 1.0 / WARMDOWN_RATIO
inv_adam_warmdown = 1.0 / ADAM_WARMDOWN_RATIO
muon_warmdown_start = 1.0 - WARMDOWN_RATIO
adam_warmdown_start = 1.0 - ADAM_WARMDOWN_RATIO

# Track B: segmented timing accumulators
_track_b_step_times: list[float] = []       # per-step core (fwd+bwd+opt) time, seconds
_track_b_cumulative_time = 0.0              # cumulative steady-state core time
_track_b_val_bpb_curve: list[dict] = []    # [{"step": s, "val_bpb_probe": v, "cum_time": t}, ...]
_diagnostic_data_hasher = hashlib.sha256() if PACKER_DIAGNOSTIC_HASHES else None
_diagnostic_boundary_hasher = (
    hashlib.sha256() if PACKER_DIAGNOSTIC_HASHES else None
)
_diagnostic_loss_hasher = hashlib.sha256() if PACKER_DIAGNOSTIC_HASHES else None
_diagnostic_ab_data_hasher = (
    hashlib.sha256() if PACKER_DIAGNOSTIC_HASHES else None
)
_diagnostic_ab_boundary_hasher = (
    hashlib.sha256() if PACKER_DIAGNOSTIC_HASHES else None
)
_diagnostic_ab_loss_hasher = (
    hashlib.sha256() if PACKER_DIAGNOSTIC_HASHES else None
)
_diagnostic_tail_loss_hasher = (
    hashlib.sha256() if PACKER_DIAGNOSTIC_HASHES else None
)
_diagnostic_scanner_micro_batches = 0
_diagnostic_sidecar_micro_batches = 0
_diagnostic_ab_end_step = PACKER_SIDECAR_ACTIVATE_STEP + 20
_diagnostic_clean_profile_start_step = PACKER_SIDECAR_ACTIVATE_STEP + 30
_diagnostic_clean_profile_step_times: list[float] = []

while True:
    OBS.clear()
    OBS.set_context("train")
    torch.cuda.synchronize()
    t0 = time.time()
    _t_fwd_bwd_start = t0
    _use_sidecar_this_step = (
        PACKER_DOC_BOUNDARIES
        and step >= PACKER_SIDECAR_ACTIVATE_STEP
    )
    if PACKER_DIAGNOSTIC_HASHES and step == PACKER_SIDECAR_ACTIVATE_STEP:
        print(
            "PACKER_SIDECAR_PHASE_ACTIVATED=1 "
            f"step={step} sidecar_active={int(_use_sidecar_this_step)} "
            f"registered_sidecar_arm={int(PACKER_DOC_BOUNDARIES)}",
            flush=True,
        )
    for _micro_step in range(grad_accum_steps):
        if PACKER_DIAGNOSTIC_HASHES and step < _diagnostic_ab_end_step:
            _phase_data_hasher = (
                _diagnostic_data_hasher
                if step < PACKER_SIDECAR_ACTIVATE_STEP
                else _diagnostic_ab_data_hasher
            )
            _hash_tensor_bytes(_phase_data_hasher, "x", x)
            _hash_tensor_bytes(_phase_data_hasher, "y", y)
            _phase_data_hasher.update(struct.pack("<q", epoch))
        # Mask construction is inside the timed region: it is real per-step
        # training cost, not setup, and must be charged to the arm that needs it.
        if _use_sidecar_this_step:
            _doc_masks = build_doc_masks_from_boundaries(
                boundary_lease, WINDOW_LEFTS, record_boundary_count=True
            )
            if PACKER_DIAGNOSTIC_HASHES:
                _diagnostic_sidecar_micro_batches += 1
        else:
            _doc_masks = build_doc_masks_for_batch(
                x, WINDOW_LEFTS, record_boundary_count=True
            )
            if PACKER_DIAGNOSTIC_HASHES:
                _diagnostic_scanner_micro_batches += 1
        if PACKER_DIAGNOSTIC_HASHES:
            if step < _diagnostic_ab_end_step:
                _phase_boundary_hasher = (
                    _diagnostic_boundary_hasher
                    if step < PACKER_SIDECAR_ACTIVATE_STEP
                    else _diagnostic_ab_boundary_hasher
                )
                _phase_cu = next(iter(_doc_masks.values()))[0]
                _hash_tensor_bytes(
                    _phase_boundary_hasher, "cu", _phase_cu
                )
            _diagnostic_callback_site = f"{step}:{_micro_step}"
            _diagnostic_inside_model_region = True
        _compile_stance = (
            torch.compiler.set_stance("fail_on_recompile")
            if (
                PACKER_DIAGNOSTIC_HASHES
                and (step != 0 or _micro_step != 0)
            )
            else nullcontext()
        )
        with _compile_stance:
            with autocast_ctx:
                loss = model(x, y, doc_masks=_doc_masks)
            if (
                PACKER_DIAGNOSTIC_HASHES
                and step < _diagnostic_ab_end_step
            ):
                _phase_loss_hasher = (
                    _diagnostic_loss_hasher
                    if step < PACKER_SIDECAR_ACTIVATE_STEP
                    else _diagnostic_ab_loss_hasher
                )
                _hash_tensor_bytes(
                    _phase_loss_hasher, "loss", loss
                )
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
        if PACKER_DIAGNOSTIC_HASHES:
            _diagnostic_inside_model_region = False
            _diagnostic_callback_site = ""
        if PACKER_DOC_BOUNDARIES:
            # The consumer event must cover every forward/backward read of the
            # leased cu_seqlens. Recording it before backward would permit the
            # three-slot ring to overwrite CUDA storage still used by FA3.
            if _use_sidecar_this_step:
                boundary_lease.mark_consumed()
            else:
                # During the governed A/A prefix the treatment advances the
                # identical sidecar loader but deliberately consumes scanner
                # boundaries. The unused H2D lease is still retired safely.
                boundary_lease.discard_unconsumed()
            x, y, epoch, boundary_lease = next(train_loader)
        else:
            x, y, epoch = next(train_loader)
    if OBSERVE_LAYER_PROBES and GPAS_ENABLE:
        # Diagnostic-only gate telemetry, sampled after the accumulated backward
        # and before optimizer.step()/zero_grad.  This path is eager by contract.
        for _gpas_i, _gpas_block in enumerate(model.transformer.h):
            _gpas_alpha = _gpas_block.gpas_alpha
            _gpas_scale = 1.0 - F.silu(_gpas_alpha.detach().float())
            OBS.add(f"layer_{_gpas_i}.gpas_alpha", _gpas_alpha)
            OBS.add(f"layer_{_gpas_i}.gpas_forward_scale", _gpas_scale)
            if _gpas_alpha.grad is not None:
                OBS.add(f"layer_{_gpas_i}.gpas_alpha_grad", _gpas_alpha.grad)
    # Track B only: the extra host sync serializes fwd+bwd against the
    # optimizer launch, so it must not run in normal (TRACK_B_MODE=0) training.
    if TRACK_B_MODE:
        torch.cuda.synchronize()
        _t_fwd_bwd_end = time.time()

    # Progress and schedules (decoupled warmdown: Muon=0.9, Adam=0.7, Ngram VE=0.0)
    # RSI FIXED-COMPUTE STOP: step-based progress removes GPU-contention from the
    # schedule shape; time-based progress retained for STOP_MODE="time". Both
    # exclude the first 10 (compilation) steps.
    if STOP_MODE == "steps":
        progress = min(max(0, step - 10) / max(1, MAX_STEPS - 10), 1.0)
    elif STOP_MODE == "walltime":
        progress = min((time.time() - t_start_training) * inv_time_budget, 1.0)
    else:
        _time_stop_elapsed = (
            _track_b_cumulative_time if TRACK_B_MODE else total_training_time
        )
        progress = min(_time_stop_elapsed * inv_time_budget, 1.0)
    # LR warmup multiplier: exactly 1.0 unless WARMUP_RATIO > 0, and the warmup
    # window is validated not to overlap either warmdown start, so this only ever
    # scales the flat phase.
    lrm_warmup = (
        warmup_lr_mult(progress, WARMUP_RATIO, FINAL_LR_FRAC) if WARMUP_ACTIVE else 1.0
    )
    if progress < muon_warmdown_start:
        lrm_muon = lrm_warmup
        muon_wd_frac = 0.0
    else:
        muon_wd_frac = (progress - muon_warmdown_start) * inv_muon_warmdown
        lrm_muon = ((1.0 - progress) * inv_muon_warmdown) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC

    if progress < adam_warmdown_start:
        lrm_adam = lrm_warmup
        adam_beta1 = ADAM_BETAS[0]
    else:
        adam_wd_frac = (progress - adam_warmdown_start) * inv_adam_warmdown
        lrm_adam = ((1.0 - progress) * inv_adam_warmdown) * (1.0 - FINAL_LR_FRAC) + FINAL_LR_FRAC
        adam_beta1 = ADAM_BETAS[0] + (DEMON_FINAL_BETA1 - ADAM_BETAS[0]) * adam_wd_frac

    if MUON_MOMENTUM_CONTINUOUS:
        # Progress-anchored warmup: reach MUON_PEAK_MOMENTUM exactly at warmdown
        # onset (continuous), then the same quadratic decay to the terminal value.
        if progress < muon_warmdown_start:
            wu = progress / muon_warmdown_start if muon_warmdown_start > 0 else 1.0
            muon_momentum = (1.0 - wu) * 0.85 + wu * MUON_PEAK_MOMENTUM
        else:
            muon_momentum = MUON_PEAK_MOMENTUM + (muon_wd_frac ** 2) * (MUON_WARMDOWN_MOMENTUM - MUON_PEAK_MOMENTUM)
    else:
        frac = min(step / 300, 1)
        muon_momentum = (1 - frac) * 0.85 + frac * MUON_PEAK_MOMENTUM
        if progress > muon_warmdown_start:
            muon_momentum = MUON_PEAK_MOMENTUM + (muon_wd_frac ** 2) * (MUON_WARMDOWN_MOMENTUM - MUON_PEAK_MOMENTUM)
    muon_beta2 = MUON_BETA2_PEAK + muon_wd_frac * (MUON_BETA2_WARMDOWN - MUON_BETA2_PEAK)
    muon_lr_boost = 1.0 + muon_wd_frac * (MUON_LR_BOOST - 1.0)
    # VE RMSProp reverse-Demon: DELAYED ramp (only last 30% of Muon warmdown)
    late_frac = max(0.0, (muon_wd_frac - 0.7) / 0.3)
    ve_beta2 = NGRAM_VE_BETAS[1] + late_frac * (NGRAM_VE_BETA2_WARMDOWN - NGRAM_VE_BETAS[1])

    base_wd = WEIGHT_DECAY * (1 - progress)
    early_dist = abs(progress - WD_EARLY_PULSE_CENTER)
    if early_dist < WD_EARLY_PULSE_HALF_WIDTH:
        muon_weight_decay = base_wd * WD_EARLY_PULSE_MAGNITUDE
    else:
        dist = abs(progress - WD_PULSE_CENTER)
        if dist < WD_PULSE_HALF_WIDTH:
            muon_weight_decay = base_wd * WD_PULSE_MAGNITUDE
        else:
            mid_dist = abs(progress - WD_MID_PULSE_CENTER)
            if mid_dist < WD_MID_PULSE_HALF_WIDTH:
                local = (progress - (WD_MID_PULSE_CENTER - WD_MID_PULSE_HALF_WIDTH)) / (2 * WD_MID_PULSE_HALF_WIDTH)
                bump = 2 * local if local < 0.5 else 2 * (1 - local)
                muon_weight_decay = base_wd * (1.0 + bump * (WD_MID_PULSE_MAGNITUDE - 1.0))
            else:
                muon_weight_decay = base_wd

    muon_lr = lrm_muon * muon_lr_boost
    if progress < muon_warmdown_start:
        for group in muon_groups:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["beta2"] = muon_beta2
        # Baseline (WARMUP_ACTIVE False) never touches LR before warmdown -- the
        # groups keep initial_lr. Warmup is the only reason to write here, and it
        # keeps writing after the window so the restore to initial_lr is explicit.
        # n-gram VE groups are deliberately excluded: they are schedule-exempt
        # (NGRAM_WARMDOWN_RATIO=0.0) and warming them is a separate intervention.
        if WARMUP_ACTIVE:
            for group, initial_lr in muon_group_lrs:
                group["lr"] = initial_lr * lrm_warmup
            for group, initial_lr in x0_group_lrs:
                group["lr"] = initial_lr * lrm_warmup
    else:
        for group, initial_lr in muon_group_lrs:
            group["lr"] = initial_lr * muon_lr
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            group["beta2"] = muon_beta2
        for group, initial_lr in x0_group_lrs:
            group["lr"] = initial_lr * lrm_muon
    if progress >= adam_warmdown_start:
        for group, initial_lr in adam_group_lrs:
            group["lr"] = initial_lr * lrm_adam
        for group, beta2 in adam_demon_groups:
            group["betas"] = (adam_beta1, beta2)
    elif WARMUP_ACTIVE:
        for group, initial_lr in adam_group_lrs:
            group["lr"] = initial_lr * lrm_warmup
    # Update ngram VE RMSProp beta2 during warmdown (delayed reverse-Demon for sparse tables)
    if progress >= muon_warmdown_start and late_frac > 0.0:
        for group in ngram_groups:
            group["beta2"] = ve_beta2
    optimizer.step()
    model.zero_grad(set_to_none=True)
    # Track B only: see note above -- no extra sync in normal training.
    if TRACK_B_MODE:
        torch.cuda.synchronize()
        _t_opt_end = time.time()

    train_loss_f = train_loss.item()
    if PACKER_DIAGNOSTIC_HASHES:
        if step >= _diagnostic_ab_end_step:
            _diagnostic_tail_loss_hasher.update(struct.pack("<q", step))
            _diagnostic_tail_loss_hasher.update(
                struct.pack("<d", train_loss_f)
            )
        if step + 1 == PACKER_SIDECAR_ACTIVATE_STEP:
            _diagnostic_aa_model_sha256 = _canonical_state_sha256(
                model.state_dict()
            )
            _diagnostic_aa_optimizer_sha256 = _canonical_state_sha256(
                optimizer.state_dict()
            )
            _diagnostic_aa_cpu_rng_sha256 = _canonical_state_sha256(
                torch.get_rng_state()
            )
            _diagnostic_aa_cuda_rng_sha256 = _canonical_state_sha256(
                torch.cuda.get_rng_state(device)
            )
            _diagnostic_dynamo_events = [
                event
                for event in _diagnostic_compile_events
                if event[0] == "DYNAMO"
            ]
            print(
                "PACKER_DIAGNOSTIC_AA_ATTESTED=1 "
                f"steps={step + 1} "
                f"micro_batches={_diagnostic_scanner_micro_batches} "
                f"model_sha256={_diagnostic_aa_model_sha256} "
                f"optimizer_sha256={_diagnostic_aa_optimizer_sha256} "
                f"loss_sha256={_diagnostic_loss_hasher.hexdigest()} "
                f"data_sha256={_diagnostic_data_hasher.hexdigest()} "
                f"boundary_sha256={_diagnostic_boundary_hasher.hexdigest()} "
                f"cpu_rng_sha256={_diagnostic_aa_cpu_rng_sha256} "
                f"cuda_rng_sha256={_diagnostic_aa_cuda_rng_sha256} "
                f"dynamo_graph_count={len(_diagnostic_dynamo_events)}",
                flush=True,
            )
            if len(_diagnostic_dynamo_events) != 1:
                raise RuntimeError(
                    "PACKER_DIAGNOSTIC_AA_GRAPH_COUNT_FAILED "
                    f"observed={len(_diagnostic_dynamo_events)} expected=1"
                )
        if step + 1 == _diagnostic_ab_end_step:
            _diagnostic_ab_model_sha256 = _canonical_state_sha256(
                model.state_dict()
            )
            _diagnostic_ab_optimizer_sha256 = _canonical_state_sha256(
                optimizer.state_dict()
            )
            _diagnostic_ab_cpu_rng_sha256 = _canonical_state_sha256(
                torch.get_rng_state()
            )
            _diagnostic_ab_cuda_rng_sha256 = _canonical_state_sha256(
                torch.cuda.get_rng_state(device)
            )
            print(
                "PACKER_DIAGNOSTIC_AB_ATTESTED=1 "
                f"start_step={PACKER_SIDECAR_ACTIVATE_STEP} "
                f"end_step={_diagnostic_ab_end_step} "
                f"mode={'sidecar' if PACKER_DOC_BOUNDARIES else 'scanner'} "
                f"model_sha256={_diagnostic_ab_model_sha256} "
                f"optimizer_sha256={_diagnostic_ab_optimizer_sha256} "
                f"loss_sha256={_diagnostic_ab_loss_hasher.hexdigest()} "
                f"data_sha256={_diagnostic_ab_data_hasher.hexdigest()} "
                f"boundary_sha256={_diagnostic_ab_boundary_hasher.hexdigest()} "
                f"cpu_rng_sha256={_diagnostic_ab_cpu_rng_sha256} "
                f"cuda_rng_sha256={_diagnostic_ab_cuda_rng_sha256} "
                f"scanner_micro_batches={_diagnostic_scanner_micro_batches} "
                f"sidecar_micro_batches={_diagnostic_sidecar_micro_batches}",
                flush=True,
            )

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt
    if (
        PACKER_DIAGNOSTIC_HASHES
        and step >= _diagnostic_clean_profile_start_step
    ):
        _diagnostic_clean_profile_step_times.append(dt)

    # Track B: accumulate segmented timing (exclude first 10 compile/warmup steps)
    if TRACK_B_MODE and step > 10:
        _core_time = _t_opt_end - _t_fwd_bwd_start
        _track_b_step_times.append(_core_time)
        _track_b_cumulative_time += _core_time

    # Track B: periodic lightweight val_bpb probe
    if TRACK_B_MODE and VAL_BPB_PROBE_EVERY > 0 and step > 10 and step % VAL_BPB_PROBE_EVERY == 0:
        model.eval()
        with torch.no_grad(), autocast_ctx:
            _bpb_probe = evaluate_bpb_fast(
                model,
                tokenizer,
                DEVICE_BATCH_SIZE,
                eval_tokens=EVAL_TOKENS_PROBE,
                forward_kwargs_fn=_doc_mask_forward_kwargs,
            )
        model.train()
        _track_b_val_bpb_curve.append({
            "step": step,
            "val_bpb_probe": _bpb_probe,
            "cum_time": _track_b_cumulative_time,
        })
        print(f"val_bpb_probe: {_bpb_probe:.6f} step: {step} cum_time: {_track_b_cumulative_time:.1f}s", flush=True)

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / B200_BF16_PEAK_FLOPS
    if STOP_MODE == "walltime":
        remaining = max(0, TIME_BUDGET - (time.time() - t_start_training))
    elif STOP_MODE == "time" and TRACK_B_MODE:
        remaining = max(0, TIME_BUDGET - _track_b_cumulative_time)
    else:
        remaining = max(0, TIME_BUDGET - total_training_time)
    val_loss_f = None
    if VAL_LOSS_EVERY > 0 and step % VAL_LOSS_EVERY == 0:
        model.eval()
        OBS.set_context("val")
        with torch.no_grad(), autocast_ctx:
            x_val, y_val, _ = next(val_loader)
            val_loss_f = model(
                x_val, y_val, doc_masks=build_doc_masks_for_batch(x_val, WINDOW_LEFTS)
            ).item()
        OBS.set_context("train")
        model.train()

    print(
        f"step {step:05d} ({pct_done:.1f}%) | train_loss: {debiased_smooth_loss:.6f} | lrm_muon: {lrm_muon:.2f} lrm_adam: {lrm_adam:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s",
        flush=True,
    )
    if val_loss_f is not None:
        print(f"val_loss: {val_loss_f:.6f} step: {step} epoch: {epoch}", flush=True)
    observables = OBS.flush()
    observables.update(build_step_observables(
        progress=progress,
        train_loss=train_loss_f,
        val_loss=val_loss_f,
        dt=dt,
        tok_per_sec=tok_per_sec,
        lrm_muon=lrm_muon,
        lrm_adam=lrm_adam,
        muon_momentum=muon_momentum,
        muon_weight_decay=muon_weight_decay,
        mfu=mfu,
    ))
    OBS.record_step(step, epoch, observables)
    print(format_observable_line(step, epoch, observables), flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Stop: fixed compute (default), train-time budget, or real wall-clock.
    # "steps"/"time" skip the first 10 steps so compilation is not counted;
    # "walltime" counts real elapsed since t_start_training (compile included).
    if STOP_MODE == "steps":
        if step >= MAX_STEPS:
            break
    elif STOP_MODE == "walltime":
        if step > 0 and (time.time() - t_start_training) >= TIME_BUDGET:
            break
    elif step > 10 and (
        _track_b_cumulative_time if TRACK_B_MODE else total_training_time
    ) >= TIME_BUDGET:
        break

print()  # newline after \r training log

# The loop intentionally prefetches one batch as part of every charged training
# step. Its final sidecar lease has no model consumer, so release it through the
# explicit discard path rather than falsely recording it as backward-consumed.
if PACKER_DOC_BOUNDARIES:
    boundary_lease.discard_unconsumed()

total_tokens = step * TOTAL_BATCH_SIZE
training_loop_wall_time_s = time.time() - t_start_training
if DOC_MASK and DOC_MASK_IMPL == "varlen" and ATTN_BACKEND == "fa3":
    expected_boundary_batches = step * grad_accum_steps
    if len(_DOC_MASK_BOUNDARY_COUNTS) != expected_boundary_batches:
        raise RuntimeError(
            "DOC_MASK_BOUNDARY_COUNT_MISMATCH "
            f"observed={len(_DOC_MASK_BOUNDARY_COUNTS)} "
            f"expected={expected_boundary_batches}"
        )
    boundary_counts_csv = ",".join(str(count) for count in _DOC_MASK_BOUNDARY_COUNTS)
    boundary_counts_sha256 = hashlib.sha256(
        boundary_counts_csv.encode("ascii")
    ).hexdigest()
    print(
        "DOC_MASK_BOUNDARY_COUNTS_VERIFIED=1 "
        f"batches={len(_DOC_MASK_BOUNDARY_COUNTS)} "
        f"min={min(_DOC_MASK_BOUNDARY_COUNTS)} "
        f"max={max(_DOC_MASK_BOUNDARY_COUNTS)} "
        f"sha256={boundary_counts_sha256}",
        flush=True,
    )
    print(f"DOC_MASK_BOUNDARY_COUNTS={boundary_counts_csv}", flush=True)

if PACKER_DIAGNOSTIC_HASHES:
    diagnostic_model_sha256 = _canonical_state_sha256(model.state_dict())
    diagnostic_optimizer_sha256 = _canonical_state_sha256(optimizer.state_dict())
    _diagnostic_dynamo_events = [
        event
        for event in _diagnostic_compile_events
        if event[0] == "DYNAMO"
    ]
    _diagnostic_recompile_count = max(
        0, len(_diagnostic_dynamo_events) - 1
    )
    _diagnostic_compile_ids_csv = ",".join(
        event[1] for event in _diagnostic_dynamo_events
    )
    _diagnostic_compile_ids_sha256 = hashlib.sha256(
        _diagnostic_compile_ids_csv.encode("utf-8")
    ).hexdigest()
    _diagnostic_compile_sites_csv = ",".join(
        event[2] for event in _diagnostic_dynamo_events
    )
    _diagnostic_lazy_backward_count = sum(
        event[0] == "LAZY_BACKWARD"
        for event in _diagnostic_compile_events
    )
    _diagnostic_cudagraph_count = sum(
        event[0] == "CUDAGRAPH_RECORDING"
        for event in _diagnostic_compile_events
    )
    _diagnostic_profile_ms_csv = ",".join(
        f"{value * 1000.0:.6f}"
        for value in _diagnostic_clean_profile_step_times
    )
    _diagnostic_profile_sha256 = hashlib.sha256(
        _diagnostic_profile_ms_csv.encode("ascii")
    ).hexdigest()
    _diagnostic_expected_profile_samples = max(
        0, step - _diagnostic_clean_profile_start_step
    )
    print(
        "PACKER_DIAGNOSTIC_GRAPH_ATTESTED=1 "
        f"dynamo_graph_count={len(_diagnostic_dynamo_events)} "
        f"dynamo_recompile_count={_diagnostic_recompile_count} "
        f"lazy_backward_count={_diagnostic_lazy_backward_count} "
        f"cudagraph_recording_count={_diagnostic_cudagraph_count} "
        f"compile_ids_sha256={_diagnostic_compile_ids_sha256} "
        f"compile_sites={_diagnostic_compile_sites_csv} "
        f"activation_step={PACKER_SIDECAR_ACTIVATE_STEP} "
        f"clean_profile_start_step={_diagnostic_clean_profile_start_step}",
        flush=True,
    )
    print(
        "PACKER_DIAGNOSTIC_PROFILE_ATTESTED=1 "
        f"start_step={_diagnostic_clean_profile_start_step} "
        f"samples={len(_diagnostic_clean_profile_step_times)} "
        f"expected_samples={_diagnostic_expected_profile_samples} "
        f"sha256={_diagnostic_profile_sha256}",
        flush=True,
    )
    print(
        f"PACKER_DIAGNOSTIC_PROFILE_MS={_diagnostic_profile_ms_csv}",
        flush=True,
    )
    print(
        "PACKER_DIAGNOSTIC_HASHES_VERIFIED=1 "
        f"steps={step} model_sha256={diagnostic_model_sha256} "
        f"optimizer_sha256={diagnostic_optimizer_sha256} "
        f"aa_loss_sha256={_diagnostic_loss_hasher.hexdigest()} "
        f"aa_boundary_sha256={_diagnostic_boundary_hasher.hexdigest()} "
        f"ab_loss_sha256={_diagnostic_ab_loss_hasher.hexdigest()} "
        f"ab_data_sha256={_diagnostic_ab_data_hasher.hexdigest()} "
        f"ab_boundary_sha256={_diagnostic_ab_boundary_hasher.hexdigest()} "
        f"tail_loss_sha256={_diagnostic_tail_loss_hasher.hexdigest()} "
        f"aa_data_sha256={_diagnostic_data_hasher.hexdigest()} "
        f"scanner_micro_batches={_diagnostic_scanner_micro_batches} "
        f"sidecar_micro_batches={_diagnostic_sidecar_micro_batches}",
        flush=True,
    )
    if (
        len(_diagnostic_dynamo_events) != 1
        or _diagnostic_recompile_count != 0
        or _diagnostic_cudagraph_count != 0
        or len(_diagnostic_clean_profile_step_times)
        != _diagnostic_expected_profile_samples
        or len(_diagnostic_clean_profile_step_times) < 20
    ):
        raise RuntimeError(
            "PACKER_DIAGNOSTIC_RUNTIME_ATTESTATION_FAILED "
            f"dynamo_graph_count={len(_diagnostic_dynamo_events)} "
            f"dynamo_recompile_count={_diagnostic_recompile_count} "
            f"cudagraph_recording_count={_diagnostic_cudagraph_count} "
            f"profile_samples={len(_diagnostic_clean_profile_step_times)} "
            f"expected_profile_samples={_diagnostic_expected_profile_samples}"
        )
    callback_handler.remove_start_callback(
        _diagnostic_compile_callback
    )

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(
        model,
        tokenizer,
        DEVICE_BATCH_SIZE,
        forward_kwargs_fn=_doc_mask_forward_kwargs,
    )

# ===================== RSI OBSERVABILITY SYSTEM: DIAGNOSE BEGIN =====================
# Per-token val-loss decomposition (env-gated DIAGNOSE=1; default off -> byte-identical,
# diagnostic-only). Answers "where is the underfit model losing?": mean per-token loss
# by token-frequency decile (rare vs common) and by sequence position. A large rare-token
# loss excess would localize the bottleneck to vocab/n-gram capacity and give a structural
# bet a predicted effect size; a position gradient points at context handling.
if os.environ.get("DIAGNOSE", "0") == "1":
    try:
        import lib as _lib
        _tb = _lib.get_token_bytes(device="cuda")
        V = _tb.shape[0]
        _T = _lib.MAX_SEQ_LEN
        _loader = _lib.make_dataloader(tokenizer, DEVICE_BATCH_SIZE, _T, "val")
        _steps = _lib.EVAL_TOKENS // (DEVICE_BATCH_SIZE * _T)
        _tls = torch.zeros(V, device="cuda")
        _tc = torch.zeros(V, device="cuda")
        _pls = torch.zeros(_T, device="cuda")
        _pc = torch.zeros(_T, device="cuda")
        with torch.no_grad(), autocast_ctx:
            for _ in range(_steps):
                _x, _y, _ = next(_loader)
                _loss = model(
                    _x,
                    _y,
                    reduction="none",
                    doc_masks=build_doc_masks_for_batch(_x, WINDOW_LEFTS),
                ).view(_y.shape[0], _y.shape[1])
                _m = (_tb[_y] > 0).float()
                _tls.index_add_(0, _y.view(-1), (_loss * _m).view(-1))
                _tc.index_add_(0, _y.view(-1), _m.view(-1))
                _pls += (_loss * _m).sum(0)
                _pc += _m.sum(0)
        _c = _tc.cpu()
        _l = _tls.cpu()
        _order = _c.argsort(descending=True)
        _cc = _c[_order].cumsum(0)
        _tot = float(_cc[-1].item())
        print("DIAGNOSE_FREQ_DECILES(most->least frequent): mean per-token val nats")
        for _d in range(10):
            _mask = (_cc > _d * _tot / 10) & (_cc <= (_d + 1) * _tot / 10)
            _idx = _order[_mask]
            _denom = float(_c[_idx].sum().item())
            _ml = (float(_l[_idx].sum().item()) / _denom) if _denom > 0 else 0.0
            print(f"DIAGNOSE_FREQ d{_d} mean_loss={_ml:.4f} n_types={int(_idx.numel())} n_tokens={int(_denom)}")
        _pl = (_pls / _pc.clamp_min(1)).cpu()
        print("DIAGNOSE_POS_BINS(8): mean val nats by sequence position")
        for _b in range(8):
            _s = _b * _T // 8
            _e = (_b + 1) * _T // 8
            print(f"DIAGNOSE_POS pos[{_s}:{_e}] mean_loss={float(_pl[_s:_e].mean().item()):.4f}")
    except Exception as _e:
        print(f"DIAGNOSE failed (non-fatal): {type(_e).__name__}: {_e}")
# ====================== RSI OBSERVABILITY SYSTEM: DIAGNOSE END ======================

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100
    * num_flops_per_token
    * TOTAL_BATCH_SIZE
    * (step - 10)
    / total_training_time
    / B200_BF16_PEAK_FLOPS
    if total_training_time > 0
    else 0
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"total_tokens:     {total_tokens}")
print(f"charged_training_seconds: {total_training_time:.9f}")
print(f"num_steps:        {step}")
print(f"stop_mode:        {STOP_MODE}")
print(f"target_steps:     {MAX_STEPS}")
if STOP_MODE == "steps":
    compute_complete = step >= MAX_STEPS
elif STOP_MODE == "walltime":
    # Use the clock that actually stopped training. Final evaluation happens
    # after training_loop_wall_time_s is captured and must not turn an
    # under-budget run into a false "complete".
    compute_complete = training_loop_wall_time_s >= TIME_BUDGET
else:
    compute_complete = (
        _track_b_cumulative_time if TRACK_B_MODE else total_training_time
    ) >= TIME_BUDGET
print(f"compute_complete: {int(compute_complete)}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")

# ===================== TRACK B RESULT OUTPUT =====================
if TRACK_B_MODE:
    import hashlib as _hashlib
    import json as _json
    import subprocess as _subprocess
    from pathlib import Path as _Path

    # Store the exact bytes that determine the train/data/eval path.  A launcher
    # can verify the upload before execution, but the result artifact must carry
    # its own provenance so it remains auditable after logs and remote dirs move.
    _track_b_repo_dir = _Path(__file__).resolve().parent
    _track_b_code_hashes = {}
    for _track_b_name in (
        "train.py",
        "lib.py",
        "observable.py",
        "prepare.py",
        "data_split.json",
        "pyproject.toml",
    ):
        _track_b_file = _track_b_repo_dir / _track_b_name
        if _track_b_file.is_file():
            _track_b_code_hashes[_track_b_name] = _hashlib.sha256(
                _track_b_file.read_bytes()
            ).hexdigest()

    # Prefer the launcher's exact physical UUID. For direct invocations, resolve
    # the first CUDA_VISIBLE_DEVICES entry (or physical GPU 0) after training.
    # Failure stays explicit: strict validation rejects an unpaired artifact.
    _track_b_gpu_uuid = os.environ.get("TRACK_B_GPU_UUID", "")
    if not _track_b_gpu_uuid:
        try:
            _track_b_physical_gpu = (
                os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0] or "0"
            )
            _track_b_gpu_uuid = _subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    _track_b_physical_gpu,
                    "--query-gpu=uuid",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, _subprocess.SubprocessError):
            _track_b_gpu_uuid = ""

    _track_b_result = {
        "schema_version": 2,
        "mode": "track_b_bare",
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "stop_mode": STOP_MODE,
        "time_budget_s": TIME_BUDGET,
        "startup_time_s": startup_time,
        "training_loop_wall_time_s": training_loop_wall_time_s,
        "post_training_eval_time_s": (t_end - t_start) - startup_time - training_loop_wall_time_s,
        "total_wall_time_s": t_end - t_start,
        "total_training_time_s": total_training_time,
        "cumulative_steady_time_s": _track_b_cumulative_time,
        "num_steps": step,
        "timing_protocol": {
            "scope": "synchronized_steady_state_training_step",
            "time_stop_clock": (
                "cumulative_synchronized_steady_state_training_step"
                if STOP_MODE == "time"
                else None
            ),
            "warmup_steps_excluded": 11,
            "includes": [
                "forward",
                "backward",
                "optimizer",
                "in_loop_data_fetch",
            ],
            "excludes": [
                "model_and_optimizer_setup",
                "compile_and_first_11_steps",
                "console_logging",
                "observable_serialization",
                "val_loss_probes",
                "val_bpb_probes",
                "final_evaluation",
            ],
            "is_end_to_end_wall_clock": False,
        },
        "step_times_ms": [t * 1000.0 for t in _track_b_step_times],
        "val_bpb_curve": _track_b_val_bpb_curve,
        "final_val_bpb": val_bpb,
        "median_step_time_ms": sorted(_track_b_step_times)[len(_track_b_step_times) // 2] * 1000.0 if _track_b_step_times else 0.0,
        "p90_step_time_ms": sorted(_track_b_step_times)[int(len(_track_b_step_times) * 0.90)] * 1000.0 if _track_b_step_times else 0.0,
        "p95_step_time_ms": sorted(_track_b_step_times)[int(len(_track_b_step_times) * 0.95)] * 1000.0 if _track_b_step_times else 0.0,
        "mean_step_time_ms": (sum(_track_b_step_times) / len(_track_b_step_times) * 1000.0) if _track_b_step_times else 0.0,
        "std_step_time_ms": (
            (sum((t - sum(_track_b_step_times) / len(_track_b_step_times)) ** 2
                 for t in _track_b_step_times) / len(_track_b_step_times)) ** 0.5 * 1000.0
        ) if len(_track_b_step_times) > 1 else 0.0,
        "peak_vram_mb": peak_vram_mb,
        "config": {
            "ATTN_BACKEND": ATTN_BACKEND,
            "DEVICE_BATCH_SIZE": DEVICE_BATCH_SIZE,
            "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
            "NGRAM_TABLE_MULT": NGRAM_TABLE_MULT,
            "NGRAM_BIGRAM_MULT": NGRAM_BIGRAM_MULT,
            "NGRAM_TRIGRAM_MULT": NGRAM_TRIGRAM_MULT,
            "NGRAM_FOURGRAM_MULT": NGRAM_FOURGRAM_MULT,
            "NGRAM_FIVEGRAM_MULT": NGRAM_FIVEGRAM_MULT,
            "GPAS_ENABLE": GPAS_ENABLE,
            "DOC_MASK": DOC_MASK,
            "DOC_MASK_MODE": DOC_MASK_MODE,
            "DOC_MASK_IMPL": DOC_MASK_IMPL,
            "PACKER_DOC_BOUNDARIES": PACKER_DOC_BOUNDARIES,
            "PACKER_BOUNDARY_VERIFY_BATCHES": PACKER_BOUNDARY_VERIFY_BATCHES,
            "PACKER_DIAGNOSTIC_HASHES": PACKER_DIAGNOSTIC_HASHES,
            "PACKER_SIDECAR_ACTIVATE_STEP": PACKER_SIDECAR_ACTIVATE_STEP,
            "WARMDOWN_RATIO": WARMDOWN_RATIO,
            "ADAM_WARMDOWN_RATIO": ADAM_WARMDOWN_RATIO,
            "FINAL_LR_FRAC": FINAL_LR_FRAC,
            "NGRAM_SPARSE_GRAD": NGRAM_SPARSE_GRAD,
            "NGRAM_STATE_ROWWISE": NGRAM_STATE_ROWWISE,
            "COMPILE_MODE": COMPILE_MODE,
            "TRACK_B_MODE": TRACK_B_MODE,
            "VAL_LOSS_EVERY": VAL_LOSS_EVERY,
            "VAL_BPB_PROBE_EVERY": VAL_BPB_PROBE_EVERY,
            "EVAL_TOKENS_PROBE": EVAL_TOKENS_PROBE,
            "OBSERVE_LAYER_PROBES": OBSERVE_LAYER_PROBES,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "device_total_memory_mb": torch.cuda.get_device_properties(device).total_memory / 1024 / 1024,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_uuid": _track_b_gpu_uuid,
            "paired_run_id": os.environ.get("TRACK_B_PAIR_ID", f"seed{SEED}"),
        },
        "code_hashes": _track_b_code_hashes,
        "estimated_dense_attention_flops_per_token": num_flops_per_token,
    }
    _track_b_path = f"track_b_result_seed{SEED}.json"
    with open(_track_b_path, "w", encoding="utf-8") as _f:
        _json.dump(_track_b_result, _f, indent=2)
    print(f"track_b_result:   {_track_b_path}")
    print(f"track_b_median_step_ms: {_track_b_result['median_step_time_ms']:.1f}")
    print(f"track_b_mean_step_ms:   {_track_b_result['mean_step_time_ms']:.1f}")
    print(f"track_b_std_step_ms:    {_track_b_result['std_step_time_ms']:.1f}")
    print(f"track_b_cum_time_s:     {_track_b_cumulative_time:.1f}")
# =================================================================

result_dir = os.environ.get("REMOTE_RESULT_DIR")
if result_dir:
    curves_path = os.path.join(result_dir, "observable_curves.json")
    try:
        OBS.write_curves_json(
            curves_path,
            metadata={
                "num_steps": step,
                "total_tokens": total_tokens,
                "total_batch_size": TOTAL_BATCH_SIZE,
                "time_budget": TIME_BUDGET,
                "observe_layer_probes": OBSERVE_LAYER_PROBES,
                "train_data_ids": data_split["train"],
                "test_data_ids": data_split["test"],
            },
        )
        print(f"observable_curves: {curves_path}")
    except Exception as exc:
        print(f"observable_curves_write_failed: {type(exc).__name__}: {exc}")
