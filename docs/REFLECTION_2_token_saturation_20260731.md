# Reflection 2 — the frame is token-saturated, and its optimum is a boundary optimum

Written 2026-07-31 at 3h of the 24h window, triggered by directive (A): two blocks
(E19, E20) closed with no adopted result. Written from the campaign records and
`campaign_log.jsonl`, not from memory.

## What all twenty-one closed directions have in common

Laid out by what each lever does to **tokens consumed**:

| lever class | effect on tokens | measured outcome |
|---|---|---|
| intrinsic refinements (cond-init, Peri-LN, GPAS, data order, width) | none | all inside ±1.3e-3, gate is 2.6e-3 |
| throughput levers already captured by `torch.compile` | none realised | 8 killed prospectively on break-even |
| throughput levers that work (depth 8→6) | +62% | **−0.002002, real but 0.77 gates** |
| throughput levers that work too well (depth 5, dim 512, 2× budget) | +100–140% | **collapse, 16–18 gates worse** |

That is the whole campaign in four rows. **There is essentially one free parameter
— how many tokens you consume before the wall — and the baseline configuration is
already near its optimum.**

Every lever is either too small to matter (it does not move tokens) or pushes past
the wall (it moves them too far). The gate sits at 2.6e-3; the intrinsic channel
tops out around 1.3e-3; the throughput channel is bounded above by a cliff worth
17 gates. The window between "too small" and "over the edge" is narrow, and depth 6
at 483M tokens is sitting in it.

**This is a boundary optimum, not an interior one.** That explains why local search
keeps returning nothing: at an interior optimum you find zero gradient in every
direction; at a boundary optimum you find zero gradient in the feasible directions
and a cliff in the one that matters.

## The wall is a property of tokens, not of the model

Three independent configurations cross it at the same place:

| configuration | steps | tokens | passes | val_bpb |
|---|---|---|---|---|
| depth 6, dim 640 | 3276 | 483M | 1.9 | **0.9296** |
| 2× budget, depth 8 | 3918 | 578M | 2.3 | 0.9870 |
| depth 6, dim 512 | 4228 | 623M | 2.5 | 0.9729 |
| depth 5, dim 512 | 4528 | 668M | 2.7 | 0.9784 |

Different depths, different widths, different parameter counts (39.8M–94.4M dense),
one boundary. E18 additionally showed it is **repetition** and not corpus region:
shuffling row-group order at depth 5 moves the endpoint by 0.0022, 4.7% of the
0.0464 cliff.

## What I assumed and never measured

Reviewing the grid honestly, the campaign has tested:

|  | 2 passes (gap −0.0663) | 3 passes (gap +0.2795) |
|---|---|---|
| **capacity** | E7 mult 256 (+0.0109), E8 mult 5 (+0.0079) | **never tested** |
| **regularisation** | E15 depth 6 (+0.0349) | E14 global WD, E15 n-gram WD (+0.0128) |

**Capacity was varied only where there is no overfitting to remove, and
regularisation only where there is.** The diagonal cell is empty, and it is empty
because I twice reasoned "capacity was already swept" without checking *at which
operating point*. That is the same error as E14 (sweeping a knob that could not
reach the blamed parameters) in a different costume: reusing a prior result outside
the regime that produced it.

It is also not a rerun of E15. Decay shrinks **all rows uniformly**, which E15
showed closes the gap by 93% and still loses because the memorising rows *are* the
signal rows. Cutting the **number** of rows forces distinct contexts to collide
into a shared row — a structural constraint toward generalisation rather than a
uniform shrinkage. Same words, different operator. E21 tests it.

## What this predicts, and how it could be wrong

- Any lever that raises tokens past ~500M collapses, regardless of how it does so.
- Any lever that leaves tokens unchanged lands within ±1.3e-3.
- Therefore **no combination of levers inside this frame reaches the gate** unless
  it raises the wall itself.

The wall can be raised by: more unique data (out of scope — the corpus is frozen at
ten shards), or by making repetition less harmful. Weight-space regularisation is
closed (E15). Augmentation is not faithfully reproducible from the source I have
(E19). **Capacity-at-three-passes is the last untested route**, and if it also
fails, the honest conclusion is that this frame is closed to the lever classes
available and the remaining work is measurement quality, not search.

Falsifier: if E21 shows mult 16 or 4 *helping* at three passes after E8 measured a
comparable cut *costing* +0.0079 at two, then the operating point — not the lever —
was the deciding variable all along, and several closed directions deserve re-testing
at the boundary rather than in the interior.

## Method lessons, distinct from findings

1. **A prior result is scoped to its operating point.** "Capacity was already
   swept" was true and irrelevant. Before reusing a result to rule something out,
   state the regime it was measured in and check it matches.
2. **Flagging an unknown does not license proceeding.** E19 recorded that the
   paper omitted the offset granularity, then ran 16 arms on a guess. If a
   load-bearing hyperparameter is absent, run both variants or decline.
3. **Boundary optima need boundary experiments.** Sweeping a lever in the interior
   and extrapolating to the edge has now failed twice (E8 capacity, E5 data order).
   Test at the boundary where the phenomenon lives.
