# Copyright 2026 Recursive
# SPDX-License-Identifier: Apache-2.0
"""Attention-only benchmark for the three DOC_MASK execution paths.

Answers the question the Track B latency attribution could not: is the measured
"segmentation costs ~1.78x" an intrinsic cost of intra-document attention, or an
artifact of materializing a dense (B, 1, T, T) boolean mask and taking the
explicit-mask SDPA path?

Three arms, identical shapes and identical masking SEMANTICS (except `causal`,
which is the unsegmented baseline):

  causal  -- F.scaled_dot_product_attention(is_causal=True)          [baseline]
  dense   -- explicit (B, 1, T, T) bool mask + SDPA                  [current SOTA impl]
  flex    -- block-sparse FlexAttention BlockMask                    [proposed]

Every arm is torch.compile'd (the model compiles its whole forward, so an eager
comparison would be meaningless), timed over fwd+bwd, and the flex arm is
charged for its per-step BlockMask construction. Dense/flex outputs are compared
numerically so a speedup cannot come from computing something different.

Usage (remote H200):
    python tools/bench_doc_mask_attention.py --doc-len 512 1024 2048
"""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

# Frozen shapes from train.py: DEVICE_BATCH_SIZE=72, sequence_len=2048,
# n_layer=12, n_head=n_kv_head=6, n_embd=768.
DEFAULT_B = 72
DEFAULT_T = 2048
DEFAULT_H = 6
DEFAULT_D = 128
DEFAULT_LAYERS = 12


def make_seg(B, T, mean_doc_len, device, generator):
    """BOS-delimited segment ids matching train.py's (idx == BOS).cumsum(dim=1).

    Document lengths are geometric with the requested mean, which is a
    reasonable stand-in for a packed-document stream; the end-to-end run on real
    data is the authority, this only has to be structurally faithful.
    """
    p = 1.0 / max(1, mean_doc_len)
    u = torch.rand(B, T, device=device, generator=generator)
    is_bos = u < p
    return is_bos.long().cumsum(dim=1)


def dense_mask(seg, window_left, mode):
    T = seg.shape[1]
    pos = torch.arange(T, device=seg.device)
    allowed = pos[:, None] >= pos[None, :]
    if mode in ("both", "window") and window_left is not None and 0 < window_left < T:
        allowed = allowed & (pos[None, :] >= (pos[:, None] - window_left + 1))
    if mode in ("both", "seg"):
        same_seg = seg[:, :, None] == seg[:, None, :]
        return (same_seg & allowed[None, :, :]).unsqueeze(1)
    return allowed[None, None, :, :]


def flex_mask(seg, window_left, mode, create_block_mask):
    B, T = seg.shape
    use_window = (
        mode in ("both", "window") and window_left is not None and 0 < window_left < T
    )
    use_seg = mode in ("both", "seg")

    def mask_mod(b, h, q_idx, kv_idx):
        allowed = q_idx >= kv_idx
        if use_window:
            allowed = allowed & (kv_idx > q_idx - window_left)
        if use_seg:
            allowed = allowed & (seg[b, q_idx] == seg[b, kv_idx])
        return allowed

    return create_block_mask(
        mask_mod, B if use_seg else None, None, T, T, device=seg.device, _compile=True
    )


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=DEFAULT_B)
    ap.add_argument("--seq", type=int, default=DEFAULT_T)
    ap.add_argument("--heads", type=int, default=DEFAULT_H)
    ap.add_argument("--head-dim", type=int, default=DEFAULT_D)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--mode", default="seg", choices=["seg", "both", "window"])
    ap.add_argument("--doc-len", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    device = "cuda"
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    B, T, H, D = args.batch, args.seq, args.heads, args.head_dim
    # Per-layer left window (train.py window_pattern "SSSL" -> S=T//2, last=T).
    window_left = T // 2

    def new_qkv():
        return [
            torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, generator=gen)
            .requires_grad_(True)
            for _ in range(3)
        ]

    sdpa_causal = torch.compile(
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    )
    sdpa_masked = torch.compile(
        lambda q, k, v, m: F.scaled_dot_product_attention(
            q, k, v, attn_mask=m, is_causal=False
        )
    )
    flex = torch.compile(flex_attention)

    print(
        f"shape B={B} T={T} H={H} D={D} layers={args.layers} mode={args.mode} "
        f"dtype=bf16 device={torch.cuda.get_device_name(0)}"
    )
    print(
        f"{'doc_len':>8} {'sparsity':>9} {'causal_ms':>10} {'dense_ms':>9} "
        f"{'flex_ms':>8} {'mask_ms':>8} {'dense/causal':>13} {'flex/causal':>12} "
        f"{'max|dense-flex|':>16}"
    )

    q, k, v = new_qkv()
    grad = torch.randn_like(q)

    def bwd(out):
        q.grad = k.grad = v.grad = None
        out.backward(grad, retain_graph=False)

    for doc_len in args.doc_len:
        seg = make_seg(B, T, doc_len, device, gen)
        m = dense_mask(seg, window_left, args.mode)
        sparsity = 1.0 - m.float().mean().item() / 0.5  # vs the causal half-matrix

        bm = flex_mask(seg, window_left, args.mode, create_block_mask)

        # Correctness: same semantics, different kernel. Both arms are scored
        # against an fp32 explicit-mask reference, so "they differ from each
        # other by bf16 noise" cannot hide one of them computing the wrong thing.
        with torch.no_grad():
            ref = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), attn_mask=m, is_causal=False
            )
            out_dense = sdpa_masked(q, k, v, m)
            out_flex = flex(q, k, v, block_mask=bm)
            err_dense = (out_dense.float() - ref).abs().max().item()
            err_flex = (out_flex.float() - ref).abs().max().item()
            max_diff = (out_dense.float() - out_flex.float()).abs().max().item()
            assert torch.isfinite(out_flex).all(), "flex produced non-finite output"
            print(
                f"  [check doc_len={doc_len}] max|dense-fp32ref|={err_dense:.4g} "
                f"max|flex-fp32ref|={err_flex:.4g}"
            )

        t_causal = timed(
            lambda: bwd(sdpa_causal(q, k, v)), args.iters, args.warmup
        )
        t_dense = timed(
            lambda: bwd(sdpa_masked(q, k, v, m)), args.iters, args.warmup
        )
        t_flex = timed(
            lambda: bwd(flex(q, k, v, block_mask=bm)), args.iters, args.warmup
        )
        # BlockMask build is once per step, amortized across all layers; the dense
        # mask build is likewise hoisted, so charge both per-layer equivalently.
        t_mask = (
            timed(
                lambda: flex_mask(seg, window_left, args.mode, create_block_mask),
                args.iters,
                args.warmup,
            )
            / args.layers
        )

        print(
            f"{doc_len:>8} {sparsity:>8.1%} {t_causal:>10.2f} {t_dense:>9.2f} "
            f"{t_flex:>8.2f} {t_mask:>8.2f} {t_dense / t_causal:>13.2f} "
            f"{(t_flex + t_mask) / t_causal:>12.2f} {max_diff:>16.4g}"
        )


if __name__ == "__main__":
    main()
