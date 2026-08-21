# Cost and Gate Discipline

Cost effectiveness comes from rejecting weak ideas early, not from weakening
confirmatory experiments.

## Offline first

The offline gate reuses existing run data. It tests whether an observable adds
information beyond training step, train loss, and learning rate. Offline screens
launch no treatment arms and must declare multiplicity handling.

## Pilot second

A pilot uses exactly one seed. Its purpose is to verify implementation,
numerical safety, trigger/action execution, logging, and cost. It does not
establish a mechanism.

## Effect funnel only after implementation safety

The effect-search funnel starts with a one-pair `discovery` screen, which may
only release the three-pair stage; it is not a belief or effect conclusion.
Later discovery, validation, and locked stages use at least three pairs. Arms
start from matched checkpoints. The complete sham-control set is used when
needed to distinguish the trigger from a universally useful action.

## Two peer decision frames

Two scopes are registered in `research/setup/reconciliation.json`:

- **Challenge 1** — `walltime_5min_h200`: `STOP_MODE=time`, `TIME_BUDGET=300`,
  `ATTN_BACKEND=fa3`. The current baseline, effective sigma, tolerance, and
  seed requirement are read from `research/setup/reconciliation.json`, never
  copied from an older prose snapshot. Run it with
  the sticky challenge selection; its final stage has ten
  paired seeds. **It answers "quality per second" and adopts in its own frame.**
- **Challenge 2** — `fixed_steps_2000`, backed by the top-level `scope_key`:
  `STOP_MODE=steps`, `MAX_STEPS=2000`. Its reconciliation is currently
  `pending`, so it is dormant and cannot authorize work. If explicitly revived,
  it answers “quality per step” and adopts only in its own frame.

Order and scope resolution live in `research/setup/challenges.json`. The latest
append-only selection/stop event in `research/setup/challenge_events.jsonl`
persists across commands and process restarts. Omitted scope flags use that
selection; they never default to steps. A stop or stale selection blocks work
without auto-advancing to challenge 2.

**HARD RULE: a comparison never crosses frames.** A 5-minute result is never judged
against a 2000-step baseline. A win in one frame is not a win in the other.

Only a frame whose current reconciliation status is `passed` is executable.
Separately launched rounds have shown larger drift, so every candidate still
runs with a fresh concurrent control. A null in one frame does not overturn a
win in the other.

External reference for that frame: Recursive report prior SOTA **0.9372** -> **0.9109**
(-0.0263) on a five-minute single-GPU budget, their baseline de-hacked and averaged over
10 seeds. Different hardware (H100/B200) and a no-repeat corpus, so it is a target to
beat, not an internal control.

Fixed compute per comparison

Every adopt/discard comparison holds its selected frame constant: identical
`MAX_STEPS` in the step frame or identical `TIME_BUDGET` in the wall-time
frame. Never compare one budget or frame to another to decide adoption —
raising steps/tokens lowers `val_bpb` on its own (the model is compute-limited
here: 1660→0.9473, 2500→0.9189, 2766→0.9126), so a cross-budget delta measures
compute, not method. Runs that deliberately vary compute are diagnostics (mapping
the bpb-vs-tokens curve, locating the matched-compute operating point); record
them as such and never use them to promote or kill a `train.py` change.

## Dormant historical fixed-step frame (2026-07-21)

The historical frame is a **fixed 2,000-step budget on a single H200 GPU** (`STOP_MODE=steps`,
`MAX_STEPS=2000`), `val_bpb` on held-out shard `6542`, starting from the RSI-optimized
`rsi/daniel` config. Fixed *steps* (not wall-clock minutes) removes kernel/throughput
gaming. Its archived v12 baseline (seeds 42/43/44) was mean val_bpb
**0.930647**, `σ_seed=0.000847`, `σ_repro=0.00031`; the 2σ adoption floor is
**0.001694**. These are historical facts, not a current authorization: the
top-level setup status is pending. External benchmark: RSI **0.9109** (different
sandbox; a threshold to beat, not an internal control).

## Screen -> confirm -> hold-out workflow

1. **Screen at the 2,000-step decision budget** — the same budget used for the
   final decision. A shorter-horizon effect is an observation about that horizon,
   not a promotion input (the 1660→2766 sign flip is the standing existence proof).
2. **Use at least three paired seeds for a verdict**, and pair each treatment
   against a **freshly reproduced, CONCURRENT** same-seed baseline (same batch /
   GPU / time window). One-seed runs are plumbing pilots only; two-pair tests stay
   exploratory. Cross-time baselines carry temporal drift (~0.0007) the concurrent
   σ misses.
3. **The σ for a claim is seed-to-seed + temporal heterogeneity, NOT same-seed
   reproduction noise.** Reporting a delta against same-seed σ inflates
   significance ~3× (the write-up's "7.43σ" is ~2.6σ on the correct σ). Gate on the
   **paired t-test over the ≥3 observed deltas** (self-calibrating) plus all-same-sign
   plus |mean| above the effect-heterogeneity floor.
4. **Record compute for every arm:** steps, tokens, wall time, estimated FLOPs,
   peak memory, and **parameter count + batch tokens**. Fixed *steps* is not fixed
   *compute*: a wider MLP or larger batch wins a step-matched race by spending more
   FLOPs. If compute differs materially, report a QUALITY–COMPUTE TRADEOFF, not a win.
5. **Hold-out confirmation (winner's curse).** A candidate selected as best-of-many
   is upward-biased by selection. Its selection-time mean is NOT reportable. Re-run
   the top-k on **fresh pre-registered seeds it was not selected on**, against a
   concurrent fresh baseline; only these confirmation numbers may be quoted or
   promoted. Spot-check the headline winner at a longer budget (e.g. 2766) for
   transfer before any generalization claim.

## Selection discipline (multiplicity, winner's curse, precision)

The OPHIS loop tests hundreds–thousands of interventions against one metric. Without
correction, the *best of N* is inflated and "survivor counts" are mostly noise
(a 1σ one-sided gate passes ~16% of null candidates; the expected best-of-N |z|
under pure noise grows like √(2 ln N) ≈ 3.7 at N=1000). Standing rules:

- **Always report N** — the total interventions launched (denominator lives in
  `experiments/runs/runs.jsonl`). "376 improved ≥3σ" is uninterpretable without it.
- **Multiplicity-corrected screen threshold.** Use a Bonferroni/FDR-adjusted
  z (e.g. require `z > Φ⁻¹(1 − 0.05/N)`) on the *correct* σ, or — preferred — judge
  a candidate only by its **held-out confirmation** score (rule 5 above).
- **Variance-aware selection.** Select on the **lower confidence bound of the paired
  delta** (or `mean + λ·σ_seed`), never the mean alone. Same-seed reps cannot even
  see the relevant (seed) variance; evaluate the "10 repeats" as 10 **distinct
  seeds**, not 10 reps of seed 42.
- **A never-touched test split.** Thousands of adaptive selections against one
  validation shard is gradual val-set overfitting. Score each confirmed winner
  **exactly once** on a test split never used for selection.
- **Round to the noise floor.** Report `0.9341 ± σ`, delta with CI, and the σ
  definition. Seven-significant-figure means (`0.9318420`) against σ≈0.0003–0.0009
  are false precision that hides which sigma was used.

## Required budget fields

Every experiment records:

```text
estimated_gpu_hours
estimated_currency_cost
hard_cap_currency_cost
basis
```

The estimate may not exceed the hard cap. Every hypothesis also records an
estimated cost and its basis. Completed runs record actual GPU hours, currency
cost, and observable overhead.

No fixed cloud price is stored in this clean branch. Prices and throughput drift;
the experiment's basis must cite a current short benchmark or a clearly labelled
estimate.

## Staged allocation and automatic stopping

Every new GPU candidate declares falsifiable changes in throughput,
quality-per-step, and the frame endpoint before launch. Compute is released in
paired tranches: `1 -> 3 -> 6 -> 10`. The first pair is a screen, not an effect
claim. At three and six pairs the preregistered endpoint and step boundaries
either stop the candidate or release only the next tranche; ten confirming
pairs produce an adoption candidate for the selected frame.

An experiment cannot change its hypothesis, arms, seed prefix, predictions, or
stopping thresholds between stages. It also cannot change its archived idea,
research direction, or subsystem. Repeated subsystem failures force a pivot,
and unresolved work blocks another candidate in the same subsystem/frame. A
hard diversity counter permits at most five consecutively launched stage-1
rounds in one research direction; the next stage-1 round must change direction.
After exhausting a five-round block, that direction remains unavailable until
two stage-1 rounds explore other directions. This prevents a one-round token
pivot from resetting the same search program. Both values are configured in
the challenge catalog.
Later funnel stages are continuations and do not count as new direction rounds.
Each archived idea may enter exactly one funnel chain and may have at most one
experiment at each of the four stages. Thus a promising idea can be developed
across several checkpoints, but it cannot be restarted, forked into sibling
confirmations, or consume open-ended effort. A meaningful variation is a new
child `IdeaRecord`, not a retry label.

The campaign policy also sets a 60-minute research-paper interval. An overdue
hourly paper blocks the next gate check or authorization, never an in-flight
run. Each append-only report accounts for tested directions and references,
separates evidence-bearing findings from observations, records negative/null
results and limitations, explains portfolio decisions, and freezes next-hour
questions with stop conditions.

`tools/run_stage.py` is the only remote launch path: it schedules concurrent
controls and uses advisory locks to coordinate cooperating launchers.
`tools/run_gated.py` binds CUDA and `nvidia-smi` to the same physical GPU UUID,
requires the training PID to be observed there, and samples every two seconds
for foreign PIDs. Policy checkpoints accept only bound,
`gpu_sampling_verified` RunRecords. This is sampling-based detection, not an
exclusive hardware lease: unrelated jobs are not prevented, and a co-tenant
that starts and exits entirely between samples can be missed.

## Exploration budget (dynamic)

Refinement and exploration draw from one budget. Refinement perturbs a bracketed
setting locally; exploration makes a large or structurally novel change motivated
by a literature mechanism. Label every experiment `track:refine` or
`track:explore`.

Use the current ranked queue in
`research/knowledge/LITERATURE_SYNTHESIS.md` to allocate candidate slots. The
queue is a planning prior, not a gate: current-scope beliefs override stale
directions, every relied-on claim still needs a literature-evidence assessment,
and `validate` must report the synthesis snapshot current. A changed
planning-input fingerprint requires reviewing the queue before allocating new
slots.

**Default split: 6:4 — 60% `track:explore`, 40% `track:refine`.** This is the
firm operating point, counted by experiment *slots* (not compute -- exploration
costs more per experiment, so budgeting by compute starves it). Every session
returns to 6:4 as its default. The share may flex within **50–70% explore** by
the plateau/reset rules below, but absent an explicit trigger, hold 6:4.

- **Why 6:4, not lower.** The refinement/config space around the RSI-identical
  baseline is largely bracketed, while the structural space (data order,
  attention, normalization, init, activation, optimizer swaps) is where the
  unmined marginal value lives. A run that collapses into single-knob refinement
  is the documented failure mode (the 2026-07-19 run's 6-experiment tuning
  streak). 6:4 is the guardrail against it.
- **Enforce by count each session.** Over any trailing window of 10 launched
  experiments, at least 6 must be `track:explore`. If explore falls below 6/10,
  the next slots MUST be exploration until the ratio is restored.
- **Anti-streak cap.** No more than **2 consecutive `track:refine` experiments**
  before a `track:explore` one. A raw percentage can be satisfied by front- or
  back-loading; the streak cap is what actually prevents drift into a tuning lane
  within a session.
- **Meta-tune the floor by yield.** 40% is a start, not a constant: raise it if
  exploration keeps finding wins, lower it if exploration is all nulls while
  refinement keeps paying. The split is itself an explore/exploit choice.
- **Plateau rule.** After `K` consecutive refinement pilots fail to beat the
  current record (`K` default 3), raise the exploration share (about 50%) until a
  new validated win lands. A coordinate search that has bracketed its neighborhood
  is finished; spend the freed budget on bigger swings rather than probing both
  sides of every remaining knob.
- **Reset rule.** Once an exploration change is validated and adopted, return to
  the default split and resume local refinement around the new baseline.
- **Noise floor.** A refinement promotion threshold must exceed the measured
  seed-to-seed heterogeneity of paired effects. Same-seed/GPU reproducibility is
  useful for diagnosing the harness but is not the effective floor for a method
  claim. The current effective estimate is `σ≈0.00087`; preregister the lever
  family and multiplicity method, use at least three paired seeds, and do not
  mint a belief from a sub-floor single-seed delta.
- **STALE (2026-07-19, 3-shard regime — DO NOT USE).** Earlier seven-run figures
  (mean `val_bpb=1.050994`, σ≈`0.00144`, single-seed gate `0.003`, 3-seed gate
  `0.0016`) were measured on the BROKEN 3-shard ~5-epoch overfit split. They do not
  transfer to the corrected 10-shard config and are retained only as a record.
- **SUPERSEDED initial 2000-step baseline (2026-07-21, retained for history).**
  Seeds 42/43/44 → val_bpb 0.933156 / 0.934170 / 0.932831, mean **0.933386**,
  seed-to-seed **σ_seed = 0.000698** (n=3); seed42 ×2 → **σ_repro = 0.000593**
  (crude, n=2). These values do not define the current gate; reconciliation v12
  supersedes them with baseline `0.930647` and 2σ floor `0.001694`. σ_repro at
  this earlier 2000-step measurement was ~7×
  looser than the old @2766 (0.00009) and near the old @1660 (0.00062) — 2000 sits
  on a steeper part of the loss curve. Widen σ_repro to ≥3 same-seed reps if a
  tighter floor is needed. The old-config figures below are retained for method
  (concurrent baselines, paired-t), not as the current floor.
- **Measured noise floor (2026-07-20, corrected 10-shard config).** Three identical
  fixed-compute runs (baseline, seed 42, `STOP_MODE=steps`, GPUs 4-6) at
  `MAX_STEPS=2766`: `val_bpb=[0.912385, 0.912374, 0.912539]`, mean `0.91243`,
  **σ_repro≈`0.00009`**, range `0.00016`. Same-seed training is nearly deterministic —
  reproducibility noise is ~15x SMALLER than the stale estimate. Two gating rules:
  1. **Use fresh CONCURRENT baselines.** Run the control alongside the treatment
     (same batch/GPUs/time window). The small `σ_repro` confirms harness stability;
     it does not replace the observed paired-effect heterogeneity (`σ≈0.00087`)
     when setting a decision floor.
  2. **Cross-time baselines carry temporal drift** the concurrent σ misses (observed
     paired-delta spreads up to ~`0.0007` when a treatment is compared to a baseline
     run hours earlier). So ALWAYS reproduce the baseline concurrently (the
     screen→promote workflow already requires a fresh 2766 baseline), and gate on the
     **paired t-test over the 3 observed deltas** (self-calibrating), treating the fixed
     `2·√2·σ/√3` threshold as a secondary check only.
  3. **σ is budget-dependent.** σ_repro@1660=`0.00062` (reps `[0.946668, 0.945968,
     0.947205]`, range `0.00124`) — ~7x NOISIER than @2766, because at 1660 steps
     val_bpb sits on a steeper part of the loss curve so the same kernel nondeterminism
     moves the metric more. That does not make a two-seed 2766 test confirmatory:
     effect heterogeneity is larger than same-config reproduction noise. Screen
     and decide at 2766 with three or more pairs, and re-measure both quantities
     whenever the budget or data config changes.

Exploration experiments record the same budget fields as refinement but accept a
lower success probability in exchange for a larger predicted effect. A cheap
one-seed exploration pilot that fails is an implementation observation, not an
effect verdict or supported negative belief.

## Observable cost

Observable definitions record cadence, cost tier, expected overhead, and whether
that overhead is unprofiled, estimated, or measured. Expensive measurements run
at lower cadence or offline. Observation must not materially change the dynamics
being studied.
