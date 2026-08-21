# Freeform candidates — the open action space

Drop a **complete `train.py`** in this directory. That is the whole interface.

`tools/explore_lane.py` picks up any `*.py` here that has never been screened, builds it
through `make_variant.build({"src": "<filename>"})`, and queues it as an n=1 screening
run. Nothing else is required: no registry entry, no `@mechanism` function, no knob axis,
no council round.

## Why this exists

The v3 campaign ran 226 experiments and explored **25 distinct configs**, every one of
them `depth 8 / dim 512 / mlp 4 / SSSL`. Not because those were the best shapes, but
because they were the only shapes `make_variant.py` could express. The mechanism registry
*was* the hypothesis space, so an idea outside it could not be proposed, ranked, or run —
it could only be answered by editing the generator, and that stalled the fleet for hours
while the harness was rewritten.

The peer harnesses on the identical 300s/H200 frame do not work this way. fengheguai's
program.md says "Edit `train.py` and no other file"; AI-Scientist-v2's task memo says
"`train.py` only. Everything in it is fair game… Write the complete new `train.py`". Both
reached a competitive `val_bpb` on **one** GPU in less wall-clock than v3 spent on seven.

This directory is that rule, implemented.

## The four guards

A candidate is refused at queue time — never at 3am on a GPU — if it:

1. **is byte-identical to `baseline/train.py`** — a control wearing a treatment's name;
2. **omits `val_bpb:` or `evaluate_bpb`** — it would be scored as a crash after burning
   300 GPU-seconds;
3. **names a path rather than a bare filename** — code that cannot be recovered from the
   repository afterwards is not evidence;
4. **fails to compile or generate** — the build happens here, where it is free.

Timing instrumentation is injected where the lines still exist and **dropped where they
do not**, because you may legitimately have rewritten the training loop. The variant
header records which diagnostics landed (`# OPHIS-INSTRUMENTED clock`), so a missing
`loader_frac` reads as *not instrumented* rather than as zero.

## What a screen buys you, and what it does not

A screen is n=1, width-1, and **decides nothing**. It is exempt from the decision cutoff
— prose being late cannot idle the fleet — precisely *because* it carries no authority:

- it is never pooled into the noise band;
- `verdict.waves()` drops it outright, so it can never reach a verdict;
- it cites `hypothesis_id: "none"` and is marked `"screen": true` in its result record.

Promotion is unchanged: a candidate that screens well earns a **round entry with a
registered hypothesis and a counterbalanced wave**, which is the only path to adoption
that ever existed. The screen just tells you where to spend those eight runs.
