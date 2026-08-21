# Reading log — re-checking the external reference for this frame

Searched for any newer public figure on the 5-minute nanochat BPB task, to check
whether the campaign's reference numbers are still current before the capstone
quotes them.

## Scope verdict

**Nothing newer found for our exact task.** The searches return the same Recursive
article the reconciliation already records, plus the *distinct* modded-nanogpt
speedrun leaderboard, which is a **different benchmark** — speed-to-a-fixed-loss
(3.28 on FineWeb) on an **8-GPU HGX H100 node**, currently 1.65 hours (Run 6, CORE
0.263, March 2026).

**That leaderboard is not comparable to this frame and must not be cited as if it
were.** Our frame is one GPU, a fixed 300-second budget, and BPB as the endpoint.
Theirs is eight GPUs, a fixed loss target, and wall-clock as the endpoint. Different
objective, different hardware count, different metric.

## What the campaign's reference numbers remain

From `research/setup/reconciliation.json`, unchanged and still the only
scope-relevant external figures:

| | value |
|---|---|
| Recursive reported result | 0.9109 |
| their prior SOTA | 0.9372 |
| their reported gain | −0.0263 |
| seeds | 10 |
| hardware | H100, then B200 |

The reconciliation already flags why these are **not** a valid control for us:
different hardware (H100/B200 vs our H200), and their corpus is the full no-repeat
set while ours is ten frozen shards with repetition. It is recorded as *a target to
beat, never an internal control* — and this search gives no reason to change that.

## Consequence for the capstone

Paper 033 does not quote the external number as a comparison, and this check
confirms that was right. Our best measured configuration is **0.9295**; Recursive
report **0.9109**. The gap is real but the comparison is not apples-to-apples:
their corpus does not repeat, and repetition is precisely the wall this campaign
spent thirty-one directions characterising. Quoting the two side by side without
that caveat would be the single most misleading thing this campaign could publish.

**Verdict: no update to the reference. No number transferred. The distinct
speedrun leaderboard is recorded here explicitly so a future reader does not
mistake it for this benchmark.**
