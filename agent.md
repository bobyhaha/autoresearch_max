# Agent Notes

This repo is a small training system with a structured automated-science layer.
Agents must move from knowledge to tools, mechanisms, falsifiable hypotheses,
gated experiments, evidence, and refinement. They do not launch arbitrary runs
or treat a lower validation metric as a mechanism by itself.

Before technical work, read `program.md`, `docs/AGENT_PROTOCOL.md`, and the
generated `research/knowledge/RESEARCH_STATE.md`, then the scope-bound planning
queue in `research/knowledge/LITERATURE_SYNTHESIS.md`. Run
`python -m vibeautoresearch validate`. Only an experiment frozen in
`research/experiments/gated/experiments.jsonl` authorizes training.
Also run `python -m vibeautoresearch check-setup`; a gate without the current
scope key is historical and cannot authorize a launch.

## Project Frame

- `train.py` is the main experiment surface. It contains the GPT model,
  architecture knobs, optimizer, schedules, training loop, logging, and final
  summary.
- `lib.py` contains shared runtime utilities such as tokenizer/data helpers,
  split loading, dataloaders, and validation helpers used by training.
- `prepare.py` prepares the dataset and tokenizer cache before training.
- `data_split.json` is the root-level source of truth for train/test shard
  assignment, and it is **FROZEN for fair-game comparisons** (see Fair-Game
  Comparison Rules below). The split is train shards `1-10` with shard `6542`
  held out for validation — the same shard layout as karpathy/autoresearch, the
  base `vibeautoresearch`, and the Recursive nanochat_autoresearch fork. Only
  ~11 shards (`0-10` + val) exist locally (~15M tokens each, ~151-166M unique
  total), and the box has no network to fetch more, so a 1660-step run is ~1.6
  epochs — training data repeats. (A prior branch restricted train to shards
  `1,2,3`, a much harsher ~5-epoch overfit that inflated 1660-step `val_bpb`
  from ~0.95 to ~1.05; corrected 2026-07-19 by restoring shards `1-10`.)
- `observable.py` owns the stable observable collection API. Keep serialization
  and naming rules here instead of scattering them through model code.
- `run_h200_training.sh` orchestrates remote training on H200: clean remote temp,
  kill temp-related GPU processes, package/upload the project, run prepare/train,
  fetch results, then clean remote temp.
- `h200_remote_config.json` stores remote training settings consumed by
  `run_h200_training.sh`.
- `dashboard_server.py`, `start_training_dashboard.sh`, and
  `training_dashboard/` provide the local dashboard for live command logs,
  training/validation curves, and observables.
- New run artifacts must be referenced by an immutable `RunRecord`; large logs
  remain in the tracker/artifact store instead of being copied into registries.

## Fair-Game Comparison Rules

A lower `val_bpb` is a real result ONLY if it comes from a *method* change on a
frozen playing field. Three axes define that field and MUST stay fixed for a
comparison to be valid:

- **Data content** — the exact shard set in `data_split.json` (currently train
  `1-10`, val `6542`). FROZEN. Adding, removing, swapping, or enlarging training
  shards, or substituting different/"better"/more data, is NOT a fair lever: it
  moves the data-quality axis, not the algorithm, and invalidates the
  comparison. Never change data content to lower the metric.
- **Compute budget** — TWO peer challenges are registered, and they are not
  interchangeable. **Challenge 1** is `walltime_5min_h200`
  (`STOP_MODE=time`, `TIME_BUDGET=300`), and **challenge 2** is
  `fixed_steps_2000` (`STOP_MODE=steps`, `MAX_STEPS=2000`). Their order and
  scope mapping come from `research/setup/challenges.json`, not launcher
  constants. See "Two decision frames" below.
  **Within a frame, every adopt/discard decision MUST compare control and treatment
  at that frame's IDENTICAL budget — and a comparison NEVER crosses frames.**
  Never judge a change by comparing its run against a baseline run at a different
  step budget — that conflates the method effect with a compute effect and is
  invalid, because raising steps/tokens lowers `val_bpb` on its own (the model is
  compute-limited; the old-config curve 1660→0.9473, 2500→0.9189, 2766→0.9126 is
  a historical illustration, not the current baseline).
  Runs that deliberately vary compute are DIAGNOSTICS only (e.g. mapping the
  bpb-vs-tokens curve or locating the matched-compute operating point), never
  grounds to adopt or discard a `train.py` change. A candidate that passes at the
  screening budget should ideally be reconfirmed at the matched-compute operating
  point, but each individual comparison always holds compute fixed across arms.
- **Evaluation** — held-out val shard `6542`, BPB metric. Fixed. Both frames
  score the same metric on the same shard, so only the budget axis differs.

### Two decision frames

| order | challenge 1 | challenge 2 |
|---|---|---|
| id | `walltime_5min_h200` | `fixed_steps_2000` |
| question | quality per **second** | quality per **step** |
| stop | `STOP_MODE=time`, `TIME_BUDGET=300` | `STOP_MODE=steps`, `MAX_STEPS=2000` |
| baseline | `val_bpb` 0.933073, passed | `val_bpb` 0.930647, pending re-measurement |
| seeds required | ≥10 paired | 10-pair final funnel stage |
| noise floor | effective σ 0.001375 → 2σ gate 0.002750 | `sigma_seed` 0.000847 → 2σ gate 0.001694 |
| may adopt/discard a `train.py` change? | **YES, in this frame** | **YES, in this frame** |

**The one hard rule: a comparison never crosses frames.** A 5-minute number is
never judged against a 2000-step baseline, and vice versa. They answer different
questions — quality per *step* versus quality per *second* — so a win in one frame
is not a win in the other. Adopt separately in each, and always state which frame
a result holds in.

**What the 5-minute frame can and cannot resolve.** The 2026-07-21 wall-time
session measured ~0.005 bpb between-round throughput/thermal drift. At 10 paired
seeds that gives a ~0.0032 gate — roughly 1.9x the step frame's 0.001694. So the
5-minute frame is a coarser instrument: it resolves large effects (Recursive's own
gain was -0.0263, eight times its own gate) but genuinely cannot resolve the
small ones. Each of the four step-frame wins (-0.00405 / -0.00139 / -0.00040 /
-0.00131) sits below the 5-minute gate. That is a statement about resolution, not
about validity — a null at 5 minutes does not overturn a win at 2000 steps, and a
win at 5 minutes does not need step-frame confirmation to be real *in its frame*.

**How to select and run a frame.** Selection persists until another
generation-numbered event changes or stops it.

```bash
python -m vibeautoresearch list-challenges
python -m vibeautoresearch select-challenge walltime_5min_h200 \
  --reason "Work on challenge 1"
python tools/run_stage.py <exp> --gpus 0,1 --ssh "ssh user@host" \
  --remote-wd /path/to/repo
python -m vibeautoresearch stop-challenge --reason "Stop challenge work"
```

Every run is tagged with its challenge ID and pins the selection/challenge
fingerprints. The compatibility `--scope` option may only match the sticky
active challenge; it cannot switch challenges or bypass a stop.

The active challenge also carries its campaign policy: five stage-1 rounds
maximum per direction, a two-round cooldown elsewhere after saturation, and a
60-minute summary-paper cadence. Run `python -m vibeautoresearch
hourly-report-status` at session start and before new GPU work. If overdue,
publish the evidence-aware JSON body with `publish-hourly-report`; new
authorizations remain paused until the immutable Markdown paper is written.
Already-running GPU work is not interrupted.

**Before the 5-minute frame can run at all** it needs its own measured baseline:
at least **ten** seeds at `STOP_MODE=time TIME_BUDGET=300` on the target H200, with
the mean and `effective_sigma` written into its `baseline` block and its `status`
set to `passed`. Until then the launcher refuses it — an unmeasured frame has no
noise floor, so no result from it can be judged.

**Why ten seeds and not three.** Recursive's own 5-minute number is a **10-seed
mean**, and our measurement of this frame shows ~0.005 bpb between-round drift.
A noisier frame needs MORE seeds, not the same number:

| seeds | standard error | 2SE gate |
|---|---|---|
| 3 | 0.00289 | 0.00577 |
| 6 | 0.00204 | 0.00408 |
| **10** | **0.00158** | **0.00316** |

Even at ten seeds the effective gate (~0.0032) is ~1.9x the step frame's 0.001694,
which is why the 5-minute frame is the coarser instrument of the two. The
requirement is enforced:
the scope declares `min_seeds: 10` and the schema rejects a baseline with fewer.

### The external 5-minute reference (Recursive)

From *First Steps Toward Automated AI Research*
(<https://www.recursive.com/articles/first-steps-toward-automated-ai-research>):

- Budget: **"five-minute budget on a single GPU"** — H100, then transferred to B200.
- Metric: validation bits per byte — the same metric we score.
- Prior SOTA **0.9372** → Recursive **0.9109**, a **-0.0263** gain (or "a 1.3x
  speedup to reach the same loss").
- Their 0.9372 baseline was measured **after removing minor reward hacks** from the
  previous best solution, then evaluated on **10 random seeds**.
- Their companion nanoGPT-speedrun benchmark gates on **"mean validation loss <= 3.28
  at p < 0.01"** — a stricter significance bar than our p<0.05.

**These numbers are a target, never an internal control.** Their hardware
(H100/B200) differs from ours (H200) and their corpus is the full no-repeat set
while ours repeats ~1.6x, so absolute bpb is not comparable across sandboxes. The
de-hacking step is worth copying: measure the reference config on the *same* frame
before quoting any gap.

Allowed (fair) levers — all in `train.py`: architecture, optimizer, schedule,
regularization, initialization, and any model/training method change, plus data
*order* (same content, resequenced — but the dataloader lives in the `lib.py`
harness, so an order change must not alter content). Vary these, hold the three
frozen axes constant, and use the paired-seed 2σ gate.

### Kernels are a fair lever

**Writing, refining, wrapping, or swapping compute kernels is explicitly
allowed** — attention backends, fused ops, custom CUDA/Triton, `torch.compile`
wrapping, memory-layout work. A kernel change is a *method* change: it does not
touch data content, budget, or evaluation, so it sits inside the frozen field.

Two things make kernel work worth more than it looks:

- **In the wall-clock frame, throughput IS the metric.** A faster step buys more
  steps inside the 300s budget, which lowers `val_bpb` directly. Measured
  example: `DOC_MASK_IMPL=flex` (block-sparse) vs `dense` — the *same* attention
  mask — is −0.0097 as flex and **+0.045 as dense**, purely because the dense
  path materializes a `(B,T,T)` mask at 1.784x step time. Same maths, opposite
  verdict, decided entirely by the kernel.
- **In the fixed-step frame, a pure-throughput kernel change is worth ~nothing**
  for the metric (same steps, same tokens) — but it is *not* a no-op, because it
  can change numerics. Never assume a kernel swap is metric-neutral in the step
  frame; measure it like anything else.

Rules that still bind. A kernel change must not alter the data seen, the token
budget, or the evaluation. It must be **numerically defensible** — different
kernels give different rounding, so a kernel swap gets the same paired-seed
treatment as a model change, never a "it's just faster" exemption. Record the
kernel/backend in the run's resolved config so a result is attributable to it.
If a kernel honors a model property the previous backend silently ignored (e.g.
flash-attn windowing vs SDPA full-causal), say so explicitly: that is a *model*
difference riding along with the speedup, not a free lunch.

Worked example in this repo: `ATTN_BACKEND=fa3` segfaulted for four attempts and
was written off as hardware-blocked. The kernel was fine in eager mode, causal
and windowed; the fault was `torch.compile` tracing into `FlashAttnFunc`, an
`autograd.Function`. Wrapping the already-`CustomOpDef` primitives
(`_flash_attn_forward` / `_flash_attn_backward`) in an fa4-style
`torch.library.custom_op` with `register_fake` fixed it under `fullgraph=True`.
SDPA ignores the model's TTTL windows and runs full causal attention
(1.7355e10 FLOPs/token vs fa3's 4.4357e08). Diagnose kernels; do not route
around them.

### External SOTA is a prior, not a result

**Consult the current state of the art** — papers, public leaderboards,
reference implementations, the nanoGPT/nanochat speedrun lineage, vendor kernel
libraries — and use it to choose what to test. Recursive's own writeup is the
canonical example: their five-minute figure (0.9109, 10 seeds, prior SOTA
0.9372) told us to use ten seeds here rather than three.

The claim-revision rules apply unchanged. An external number is registered as a
`claim` with its scope, assessed as `literature_evidence` before it can justify
a run, and **never** quoted as our result. Absolute numbers do not cross
sandboxes: Recursive ran H100/B200 with working flash-attn on a full no-repeat
corpus; we run H200 with ~10 repeating shards. A method copied from an external
SOTA still has to clear this frame's own paired-seed gate against a concurrent
local control before it is adopted here.

### What "beating RSI" actually means here

Our `train.py` is a **materially modified fork** of RSI's
`optimized_from_karpathy.py`, not a byte-identical copy. The reconciliation
captured on 2026-07-20 found 493 added and 44 removed lines across 54 hunks,
including the SDPA backend, shipped-default changes, observability, fixed-step
stopping, and experimental levers. The two programs share the tuned core, so:

- Absolute numbers are NOT comparable across sandboxes. RSI's `0.9109` came from
  a richer sandbox — B200 + flash-attn that honors sliding windows and the full
  no-repeat 400B corpus. Ours is H200 + full-attention SDPA + ~10 local shards
  (data repeats). RSI `0.9109` is an external benchmark to beat, not an internal
  control.
- The valid target is to beat **our own same-data, same-compute local baseline**
  — the `rsi/daniel` config at the frozen 2000-step H200 budget on shards `1-10`,
  **current reconciled mean val_bpb `0.930647`** (seeds 42/43/44;
  `σ_seed=0.000847`, effective 2σ adoption floor `0.001694`) — with a genuine
  `train.py` method/order change that clears the paired-seed gate. The earlier
  `0.933386` measurement is retained as a superseded legacy baseline in
  reconciliation v12. Current reconciliation, and only it, defines a real local
  method win.
- Fair opportunity: RSI's config is tuned for the fresh/no-repeat regime; our
  frozen data repeats ~1.6×, so retuning `train.py` FOR the repeated regime
  (e.g. less sparse-table memorization / more regularization) is a principled
  place to look. The reverted `NGRAM_VE_BETA2` overfit artifact is one confirmed
  example that RSI-era tuning does not transfer to this regime.

The current decision scope comes from the latest valid event in
`research/setup/challenge_events.jsonl`, bound to the challenge catalog and
`research/setup/reconciliation.json`. Changing any catalog or setup
fingerprint makes the event stale and blocks work until explicit re-selection.

## Training Flow

1. Validate the research graph and verify the exact gated experiment.
2. Local code and config are packaged by `run_h200_training.sh`.
3. The remote temp directory is cleared before upload to avoid polluted runs.
4. The remote runner compares `data_split.json` train ids against the cached
   split. If the train ids changed or the cached split is missing, tokenizer
   cache is removed and rebuilt.
5. Training writes live logs locally and remote results under the run id.
6. Results are fetched and registered without rewriting raw facts.
7. The remote temp directory is cleaned after a successful fetch.

## Observable Adding Rule

The observable system must be easy to recognize and safe to remove or modify.
Any new observable work should be wrapped in strong marker comments:

```python
# ===================== RSI OBSERVABILITY SYSTEM: <AREA> BEGIN =====================
# explanation
...
# ====================== RSI OBSERVABILITY SYSTEM: <AREA> END ======================
```

Use `observable.py` as the API boundary:

- Add scalar values with `OBS.add(name, value)`.
- Add tensor magnitude summaries with `OBS.add_norms(prefix, tensor)` or
  `OBS.add_l1_l2_abs_max(prefix, tensor)`.
- Use `OBS.set_context("train")` and `OBS.set_context("val")` to distinguish
  train and validation probes.
- Use `build_step_observables(...)` for stable step-level global metrics such as
  loss, progress, throughput, LR multipliers, momentum, weight decay, and MFU.
- Do not let observable values affect model outputs, loss, optimizer state, data
  order, schedules, random seeds, or control flow.
- Do not leave large per-step observable printing in the training loop when full
  layer probes are enabled. Prefer recording in memory and writing structured
  curves at the end of training.
- If a metric name changes, update the dashboard parser/filter logic and any
  downstream analysis at the same time.

For layer probes in `train.py`, follow the existing pattern:

- Attention Q/K/V magnitudes after QK norm and rotary:
  `layer_<idx>.q`, `layer_<idx>.k`, `layer_<idx>.v`.
- Attention distribution probes:
  `layer_<idx>.attn_entropy`, `attn_entropy_norm`, `attn_max_prob`.
- Head gate:
  `layer_<idx>.head_gate_mean`.
- Attention output:
  `layer_<idx>.attn_out`.
- MLP probes:
  `layer_<idx>.mlp_pre`, `layer_<idx>.mlp_act`, `layer_<idx>.mlp_out`.

For structured observable runs, the intended behavior is:

1. Collect per-step probe values with `OBS.flush()`.
2. Merge global metrics from `build_step_observables(...)`.
3. Store them with `OBS.record_step(step, epoch, observables)`.
4. At training end, write `observable_curves.json` with
   `OBS.write_curves_json(...)` into `REMOTE_RESULT_DIR` when available.

The JSON should include metadata useful for later comparison:

- `num_steps`
- `total_tokens`
- `total_batch_size`
- `time_budget`
- `observe_layer_probes`
- `train_data_ids`
- `test_data_ids`

## Review Checklist

- Confirm `data_split.json` is logged at train start.
- Confirm tokenizer refresh depends only on train split changes.
- Confirm observable probes are diagnostic-only.
- Confirm the dashboard can load the current live log or fetched run log.
- Confirm remote temp cleanup still happens after fetched results are saved.
