#!/usr/bin/env python3
"""Curate 2025-2026 arXiv papers whose mechanisms could plausibly move `val_bpb`
inside a fixed 300-second training budget.

The objective is narrow -- lower `val_bpb` in 300 charged seconds on one H200 -- so
"high quality" here is not citation count. A 2000-GPU-day scaling result is useless
to us; a paper on warmup-free schedules at 500 steps is directly actionable. Two
filters, therefore:

  relevance -- does the mechanism apply at ~500-900 steps, ~124M params, one GPU,
               short horizon, Muon-family optimizer, wall-clock-bounded?
  strength  -- does the venue/lab/framing suggest the result will replicate? The
               operator named DeepSeek as the quality bar, so a known-strong-lab
               signal is scored, but never on its own.

Deliberately *penalised*: papers about inference, serving, quantised deployment,
agents, benchmarks, RLHF, and anything whose claims live only above 7B. They are
fine papers; they cannot move this number.

Output is a ranked id list for tools/read_fulltext.py to fetch in full.

    uv run python tools/curate_papers.py --out runs/science/reading_list.txt
    uv run python tools/curate_papers.py --out /tmp/list.txt --target 140
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ATOM = "{http://www.w3.org/2005/Atom}"
DATE_RANGE = "submittedDate:[202501010000 TO 202612312359]"

# Each query targets one lever that a 300s single-GPU run can actually pull.
QUERIES: list[tuple[str, str]] = [
    ("optimizer_muon", 'abs:"Muon" AND cat:cs.LG'),
    ("optimizer_orthogonal", 'abs:"orthogonalized" AND abs:"optimizer"'),
    ("optimizer_secondorder", 'abs:"second-order" AND abs:"language model" AND abs:"pretraining"'),
    ("optimizer_adam_variants", 'abs:"AdamW" AND abs:"convergence" AND cat:cs.LG'),
    ("optimizer_benchmark", 'abs:"optimizer" AND abs:"benchmark" AND abs:"pretraining"'),
    ("lr_schedule", 'abs:"learning rate schedule" AND abs:"language model"'),
    ("lr_warmup", 'abs:"warmup" AND abs:"learning rate"'),
    ("lr_horizon_free", 'abs:"horizon-free" OR abs:"schedule-free"'),
    ("lr_wsd", 'abs:"warmup-stable-decay" OR abs:"cooldown" AND abs:"pretraining"'),
    ("batch_size_law", 'abs:"critical batch size"'),
    ("weight_decay", 'abs:"weight decay" AND abs:"pretraining"'),
    ("init_scaling", 'abs:"initialization" AND abs:"transformer" AND abs:"training"'),
    ("mup", 'abs:"hyperparameter transfer" OR abs:"muP"'),
    ("norm_placement", 'abs:"layer normalization" AND abs:"transformer" AND abs:"training"'),
    ("norm_rmsnorm", 'abs:"RMSNorm" OR abs:"QK-norm" OR abs:"QK normalization"'),
    ("residual_scaling", 'abs:"residual" AND abs:"scaling" AND abs:"depth" AND cat:cs.LG'),
    ("attention_window", 'abs:"sliding window attention"'),
    ("attention_local_global", 'abs:"local" AND abs:"global attention" AND abs:"language model"'),
    ("attention_sparse", 'abs:"sparse attention" AND abs:"pretraining"'),
    ("attention_linear", 'abs:"linear attention" AND abs:"language model"'),
    ("attention_efficient", 'abs:"efficient attention" AND cat:cs.LG'),
    ("positional", 'abs:"rotary" AND abs:"position"'),
    ("embedding_tying", 'abs:"embedding" AND abs:"tying" OR abs:"untied embeddings"'),
    ("ngram_memory", 'abs:"n-gram" AND abs:"language model" AND abs:"memory"'),
    ("moe_small", 'abs:"mixture-of-experts" AND abs:"efficient training"'),
    ("depth_width", 'abs:"depth" AND abs:"width" AND abs:"transformer" AND abs:"tradeoff"'),
    ("activation", 'abs:"activation function" AND abs:"transformer"'),
    ("value_residual", 'abs:"value residual" OR abs:"skip connection" AND abs:"transformer"'),
    ("fp8_training", 'abs:"FP8" AND abs:"training"'),
    ("low_precision", 'abs:"low-precision" AND abs:"pretraining"'),
    ("kernel_fusion", 'abs:"kernel fusion" AND abs:"transformer" AND abs:"throughput"'),
    ("data_curriculum", 'abs:"data curriculum" AND abs:"pretraining"'),
    ("data_ordering", 'abs:"data ordering" AND abs:"language model" AND abs:"training"'),
    ("data_quality", 'abs:"data quality" AND abs:"pretraining" AND abs:"tokens"'),
    ("sample_efficiency", 'abs:"sample efficiency" AND abs:"language model" AND abs:"pretraining"'),
    ("loss_objective", 'abs:"training objective" AND abs:"next-token prediction"'),
    ("multi_token", 'abs:"multi-token prediction"'),
    ("softmax_logit", 'abs:"logit" AND abs:"softmax" AND abs:"training stability"'),
    ("scaling_law_small", 'abs:"scaling law" AND abs:"compute-optimal"'),
    ("speedrun", 'abs:"nanoGPT" OR abs:"training speedrun" OR abs:"wall-clock" AND abs:"pretraining"'),
    ("stability", 'abs:"loss spike" OR abs:"training instability" AND abs:"transformer"'),
    ("gradient_clip", 'abs:"gradient clipping" AND abs:"language model"'),
    ("tokenizer_effect", 'abs:"tokenizer" AND abs:"language model" AND abs:"performance"'),
    ("distillation_small", 'abs:"knowledge distillation" AND abs:"pretraining" AND abs:"efficient"'),
    ("deepseek", 'all:"DeepSeek"'),
    ("efficient_pretraining", 'abs:"efficient pretraining" AND cat:cs.LG'),
    ("compute_budget", 'abs:"training budget" AND abs:"language model" AND abs:"efficiency"'),
]

# Mechanism applies at our operating point.
GOOD = {
    "pretrain": 3, "pre-train": 3, "from scratch": 4, "optimizer": 3, "muon": 5,
    "learning rate": 3, "schedule": 2, "warmup": 3, "weight decay": 3, "batch size": 2,
    "initializ": 3, "normaliz": 2, "rmsnorm": 3, "attention": 2, "sliding window": 4,
    "sparse attention": 3, "throughput": 3, "wall-clock": 4, "gpu-hour": 3, "fp8": 3,
    "bf16": 2, "kernel": 2, "convergence": 2, "loss": 2, "perplexity": 3,
    "bits per byte": 5, "sample-efficient": 3, "sample efficiency": 3, "ablation": 3,
    "124m": 4, "small-scale": 3, "gpt-2": 3, "nanogpt": 5, "speedrun": 5,
    "compute-optimal": 2, "mup": 3, "hyperparameter transfer": 3, "stability": 2,
    "gradient": 2, "n-gram": 3, "multi-token prediction": 3, "curriculum": 2,
    "single gpu": 4, "8xh100": 2, "h100": 1, "reproduc": 2, "open-source": 1,
}
# Cannot move a 300s from-scratch pretraining number.
BAD = {
    "inference": -3, "serving": -4, "deployment": -3, "kv cache": -2, "decoding": -2,
    "quantiz": -2, "post-training quantization": -4, "agent": -4, "benchmark suite": -3,
    "rlhf": -4, "reinforcement learning from human": -4, "dpo": -3, "preference": -3,
    "fine-tun": -3, "lora": -3, "parameter-efficient": -3, "instruction": -3,
    "chain-of-thought": -4, "reasoning": -2, "multimodal": -3, "vision-language": -3,
    "diffusion": -3, "retrieval-augmented": -3, "safety": -3, "alignment": -2,
    "jailbreak": -4, "watermark": -4, "prompt": -2, "in-context learning": -2,
    "survey": -2, "medical": -4, "legal": -4, "recommendation": -4, "speech": -3,
    "translation": -2, "code generation": -2, "long-context": -1, "70b": -1, "405b": -1,
}
STRONG_LAB = (
    "deepseek", "openai", "google deepmind", "deepmind", "meta ai", "fair", "microsoft",
    "nvidia", "allen institute", "ai2", "mistral", "qwen", "alibaba", "tsinghua",
    "stanford", "berkeley", "mit", "cmu", "carnegie mellon", "eth", "epfl", "oxford",
    "cambridge", "princeton", "harvard", "kaiming", "karpathy", "keller jordan",
)


def curl(url: str, timeout: int = 60) -> str:
    for attempt in range(3):
        done = subprocess.run(["curl", "-sS", "-L", "--max-time", str(timeout), url],
                              capture_output=True, text=True, check=False)
        if done.returncode == 0 and done.stdout.strip().startswith("<?xml"):
            return done.stdout
        time.sleep(4 * (attempt + 1))
    return ""


def search(query: str, want: int) -> list[dict]:
    url = ("https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": f"({query}) AND {DATE_RANGE}",
        "start": 0, "max_results": want,
        "sortBy": "relevance", "sortOrder": "descending",
    }))
    body = curl(url)
    if not body:
        return []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        raw = entry.findtext(f"{ATOM}id") or ""
        m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", raw)
        if not m:
            continue
        published = (entry.findtext(f"{ATOM}published") or "")[:10]
        if published[:4] not in ("2025", "2026"):
            continue
        out.append({
            "arxiv_id": m.group(1),
            "title": " ".join((entry.findtext(f"{ATOM}title") or "").split()),
            "abstract": " ".join((entry.findtext(f"{ATOM}summary") or "").split()),
            "authors": [(a.findtext(f"{ATOM}name") or "") for a in entry.findall(f"{ATOM}author")],
            "comment": " ".join((entry.findtext("{http://arxiv.org/schemas/atom}comment") or "").split()),
            "published": published,
        })
    return out


def score(paper: dict) -> tuple[int, list[str]]:
    hay = (paper["title"] + " " + paper["abstract"] + " " + paper["comment"]).lower()
    total, why = 0, []
    for kw, w in GOOD.items():
        if kw in hay:
            total += w
            why.append(f"+{w} {kw}")
    for kw, w in BAD.items():
        if kw in hay:
            total += w
            why.append(f"{w} {kw}")
    # Peer review is weak evidence of replication, but it is evidence.
    comment = paper["comment"].lower()
    for venue, w in (("neurips", 5), ("icml", 5), ("iclr", 5), ("acl", 3),
                     ("emnlp", 3), ("tmlr", 3), ("aistats", 2), ("colm", 3)):
        if venue in comment:
            total += w
            why.append(f"+{w} {venue}")
            break
    if any(lab in hay for lab in STRONG_LAB):
        total += 3
        why.append("+3 strong-lab signal")
    if re.search(r"\bwe (release|open-source)\b|\bcode is available\b", hay):
        total += 2
        why.append("+2 code released")
    return total, why


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=130)
    ap.add_argument("--per-query", type=int, default=14)
    ap.add_argument("--min-score", type=int, default=6)
    args = ap.parse_args()

    pool: dict[str, dict] = {}
    for i, (topic, query) in enumerate(QUERIES, 1):
        hits = search(query, args.per_query)
        fresh = 0
        for paper in hits:
            if paper["arxiv_id"] in pool:
                continue
            paper["topic"] = topic
            paper["score"], paper["why"] = score(paper)
            pool[paper["arxiv_id"]] = paper
            fresh += 1
        print(f"[{i:2d}/{len(QUERIES)}] {topic:24s} {len(hits):3d} hits  {fresh:3d} new  "
              f"pool={len(pool)}")
        time.sleep(3)

    ranked = sorted(pool.values(), key=lambda p: (-p["score"], p["arxiv_id"]))
    kept = [p for p in ranked if p["score"] >= args.min_score][: args.target]

    out = Path(args.out)
    lines = [
        f"# curated {time.strftime('%Y-%m-%d')} -- {len(kept)} papers of {len(pool)} screened",
        "# objective: lower val_bpb within 300 charged training seconds, one H200",
        f"# min_score={args.min_score}, ranked by operating-point relevance",
        "",
    ]
    for p in kept:
        lines.append(f"{p['arxiv_id']}  # [{p['score']:3d}] {p['topic']:22s} {p['title'][:78]}")
    out.write_text("\n".join(lines) + "\n")

    print(f"\nscreened {len(pool)}, kept {len(kept)} -> {out}")
    print(f"score range {kept[0]['score']} .. {kept[-1]['score']}" if kept else "nothing kept")
    print("\ntop 15:")
    for p in kept[:15]:
        print(f"  [{p['score']:3d}] {p['arxiv_id']}  {p['title'][:70]}")
    dropped = [p for p in ranked if p["score"] < args.min_score]
    print(f"\ndropped {len(dropped)} below threshold; 5 examples of what that excludes:")
    for p in dropped[:5]:
        print(f"  [{p['score']:3d}] {p['title'][:70]}")


if __name__ == "__main__":
    main()
