# Experiment workflow & launch discipline

This is the operating procedure for running the walltime_5min_h200 (and any) gated
campaign. It exists because the process broke in practice; the rules below are the
fixes, not aspirations.

## The research loop (per round)

1. **Idea generation** — multi-agent, distinct angles (throughput/kernels, timed-region/
   data-pipeline, free quality-per-step, literature→concrete). Ground every idea in the
   actual code and screen it against the already-tried set (`runs.jsonl` + the 2000-step
   shadow chart). No fabrication.
2. **Analysis + verification** — verify each idea against the code yourself (don't trust an
   agent's word); dedupe; rank by expected-effect × tractability × risk.
3. **Fable adversarial review** — before any GPU time, Fable reviews the ranked queue for
   mis-rankings, hidden dependencies, and fatal flaws, and sets the execution order + the
   frame-specific gate per lever. In the active five-minute H200 frame, adoption is decided
   by fresh concurrent paired `val_bpb`, even for throughput mechanisms. Step-time is a
   causal mediator and break-even screen, not an adoption substitute. Read the current
   gate from `research/setup/reconciliation.json`; do not carry a gate across scope versions.
4. **Author the chain** — idea → mechanism → hypothesis → intervention → gated experiment,
   each with a prediction and a preregistered gate. `validate` + `audit` must be green.
5. **Run through the gate** — `tools/run_stage.py` only. Never a shell daemon.
6. **Record honestly** — negatives, segfaults, and drift are recorded as failures, not
   hidden. Update the paired-Δ chart and the hourly report after every experiment; commit +
   push. On a real SOTA (beats baseline past 2σ, confirmed across the funnel): snapshot +
   commit + push + republish chart.

## Step 4 expanded: author the chain FROM THE RECORD, don't invent it

The chain is `observation → mechanism → hypothesis → intervention → gated experiment`.
Every link carries a prediction; the last carries a preregistered gate. The rule that
makes this worth doing is that **the chain is derived from accumulated experimental
feedback and the paper corpus — it is not a place to write down a fresh idea.** An idea
with no measurement behind it is an assertion, and the audit says so out loud
(`self_proposed_mechanism_without_provenance`).

**Inputs, in this order. Read them before proposing anything.**

| # | Source | What you are looking for |
|---|---|---|
| 1 | `research/knowledge/internal/observations.jsonl` | measurements already registered — the only citable provenance |
| 2 | `research/ideas/mechanisms.jsonl` | what is `active`, `challenged`, `deprecated`; **read the `intervention_predictions` marked UNTESTED** |
| 3 | `research/ideas/hypotheses.jsonl` | `notes` of `rejected` records — the named expected failure mode is usually still true |
| 4 | `campaign_log.jsonl` + `research/refinement/campaign_batches.jsonl` | what was actually run, and every quarantine reason |
| 5 | `AI_papers/*.tex` | the "Next Experiments" sections and, more usefully, the **refuted** claims |
| 6 | `research/knowledge/external/reading_log/` | external mechanism forms, and the scope-arithmetic that killed most of them |
| 7 | `research/toolkit/available/observables.jsonl` | discriminators you already have — check BEFORE designing a new measurement |

**The derivation, four steps.**

1. **Find a disagreement in the record, not a gap in your imagination.** The best
   hypotheses come from two registered results that cannot both be fully right, or from
   a mechanism whose `intervention_predictions` contain an UNTESTED entry that nothing
   has since closed. If you cannot name the two records you are reconciling, you are
   inventing, not deriving.
2. **Reduce the disagreement to one measurable quantity**, and check source 7 first —
   the discriminator often already exists and is validation-only. A discriminating
   measurement that costs ~0% training overhead is worth more than a clever intervention.
3. **Order the cheap discriminator BEFORE the expensive experiment it gates.** A
   precondition test that can close a whole family for 0.7 GPU-h outranks the 1.7 GPU-h
   experiment it protects. This is most of the value of authoring a chain at all.
4. **Write the evidence AGAINST into `falsification.failure_condition` before launch**,
   naming the specific prior result that predicts failure. If the prior evidence favours
   the null, say so and run it anyway when it is cheap — a cheap expected-failure that
   closes a family beats an expensive maybe.

**Worked example, registered and resolvable:** `hyp_n2_precondition_decile_separability`.
Derived from `mech_memoriser_is_the_model` (uniform n-gram decay closes 93% of the
train/val gap and the endpoint still degrades) against v3's own decile probe (the n-gram
capacity benefit is **broad**, 5.26% on frequent tokens vs 4.14% on rare — flat-to-
declining with rarity). Those two disagree about whether the memorising rows and the
statistic-carrying rows are separable by token frequency. If they are not, count-
conditioned decay — the one surviving member of the weight-space family — is attacking a
population that does not exist, and there is no point funding it. The discriminator
(`obs_val_nats_by_freq_decile`) was already in the toolkit. Cost: 0.67 GPU-h, 4 paired
seeds, and it can close the family outright. Registered `proposed`, expecting to fail,
with the scope caveat stated against my own case for running it (the v3 probe measured
capacity *added*, this measures capacity *penalised* — different contrasts, so the
transfer is suggestive rather than decisive, which is exactly why it is worth measuring
instead of arguing).

### PROVENANCE DEAD END — read this before taking any launch shortcut

`ObservationRecord.run_ids` must be non-empty **and must resolve against
`research/experiments/runs/runs.jsonl`.** Direct SSH writes no RunRecord. Therefore:

> **Evidence collected outside the gated runner can never be attached to a mechanism.**

Five mechanisms confirmed or refuted in this campaign — the boundary optimum, the
repetition cliff, the capacity sign flip, and both refuted null-explanations — carry a
permanent `self_proposed_mechanism_without_provenance` warning. The measurements are real
and reproducible from the logs; they are simply not registrable, because they were
launched by hand for speed. Hundreds of GPU-hours of evidence sit outside the ledger and
cannot be brought in. Fabricating run ids to silence the audit would be strictly worse
than carrying the warning, so the warning stays.

Run everything through the gate from the first round, including throwaway probes.

## Fable writes a paper before every 5-round block (see `AI_papers/`)

**Before every block of at most 5 experiment rounds, Fable first writes a deep
hypothesis paper into
`AI_papers/`** (naming `paper_<NNN>_<slug>_<date>.md`). It analyzes deeply from: the
project's literature (`research/knowledge/external/*.jsonl`, cite real IDs), **online
literature (internet allowed)**, and the experimental facts to date. It states the block's
hypotheses, their mechanisms, expected effects vs the gate, and the execution order, and
**scores every hypothesis from 1–5 on novelty, provenance, validity, expected impact,
reliability, feasibility/cost, and falsifiability**. Every score needs a short
evidence-based rationale. An adversarial critic must challenge the scores, remove weak or
duplicative ideas, and produce the final ranked execution order. **The paper is a
precondition** — none of the block's experiments may run until it exists.

### Every paper MUST propose next experiments (hard requirement)

A paper that only reports conclusions is **incomplete and must not be committed**. The
operator has directed this repeatedly; it is a protocol rule, not a preference.

Every paper carries a **`Next Experiments`** section that is the paper's *primary output*,
not an appendix. Retrospective analysis exists to justify the proposals — a block report
whose "future work" is a vague paragraph has failed its purpose.

Each proposed experiment states, concretely enough to launch without further design work:

1. **Mechanism** — the causal chain from intervention to `val_bpb`, naming the specific
   ops/kernels/parameters involved. Not "this should help": *why*, at the level of what
   gradient, representation, or kernel changes.
2. **Predicted effect size** vs the active frame's gate (currently `0.002396` for
   `walltime_5min_h200` under reconciliation v30), with the arithmetic shown. This value
   is scope-bound and must be reread before every paper. Under the token law (paper 017)
   any mechanism costing step time must state its **break-even** intrinsic quality.
3. **Exact flags / code change** — grep `train.py` for existing `EXPLORE:*` env flags first
   and prefer flag-only experiments; name the flags.
4. **Falsifier** — a specific measurement with a **numeric threshold** that would refute the
   mechanism, registered *before* the run.
5. **Cost** in GPU-minutes, and **risk** — what would make it a no-op or a regression.
6. **Combination logic** — which proposals compose. Isolated one-off tests are disfavoured
   (paper 013); a naive implementation may lose at first and win after refinement, so
   propose the refinement path, not just the first cut.

**Multi-agent propose + critique is a precondition for the section.** Generate proposals
from several independent mechanistic lenses (throughput, quality-per-token, data/schedule),
then run an adversarial critic over the pooled set before anything reaches the paper. The
critic has repeatedly caught errors that would have wasted GPU-hours — a 4x arithmetic
error from a stale `vocab_size` default, and a silently reverted n-gram capacity. Record
what the critic killed and why; a proposal set with no casualties means the critic was weak.

## A paper is not written until the PDF builds

Papers 028 and 029 sat as `.tex` only until asked for, while 020-027 all had
PDFs. The `.tex` is the source; the PDF is the artifact anyone actually reads,
and a `.tex` that has never been compiled has never been proofread either --
build errors, overfull boxes and broken tables are invisible until it runs.

```bash
cd AI_papers && tectonic -X compile paper_0NN_*.tex
```

`tectonic` is the available engine (no pdflatex/latexmk on this host); it fetches
packages on demand and writes the PDF beside the source. `icml2024.sty` is
already vendored in `AI_papers/`. Commit the `.tex` and the `.pdf` together in
the same commit -- a paper commit without its PDF is incomplete.

## Every proposed experiment states Claim, Mechanism, Hypothesis, Reasoning

A "Next Experiments" section that only names an idea is not actionable and cannot
be audited later. Every proposed experiment in every paper MUST be written as
bullet points under exactly these four headings, in this order:

- **Claim** — the specific external or internal claim being drawn on, cited by
  its registry ID, with its recorded scope and evidence class. If no claim backs
  it, say so explicitly; a self-proposed idea is allowed, but it must be labelled
  rather than dressed up as literature.
- **Mechanism** — the physics-style causal account: name the OBSERVED PHENOMENON
  first, then the proposed cause, then the causal chain from cause to the measured
  endpoint. A mechanism that only restates the intervention is not a mechanism.
  State what a RIVAL account would predict differently, and which observable
  separates them. If nothing separates them, the mechanism is not yet testable.
- **Hypothesis** — the falsifiable prediction with numbers: the estimand, the
  expected direction and magnitude, the seed count, the decision rule, and the
  trap-check that kills the round early. Fix the analysis rule here, before
  launch, including whether the token-law control variate applies (it does NOT
  when the treatment's own mechanism is throughput).
- **Reasoning** — why this should work HERE, in this frame, given what the
  campaign has already measured. Name the prior evidence for and AGAINST it. If a
  previous result argues against the proposal, that argument goes in this section,
  not omitted.

Then rate the proposal 1-5 on novelty, provenance, validity, impact, reliability,
feasibility, falsifiability. Ratings come after the four headings, not instead of
them.

## Read literature CONSTANTLY, not once per campaign

Diagnosed failure (docs/CAMPAIGN_REFLECTION_20260731.md): only 7 of 123 external
claims ever did predictive work, and fifteen hours of experiments ran before the
most scope-relevant paper was opened. The fix is cadence, not intention.

**After every experiment block, before proposing the next one, read at least one
primary source.** Not the local corpus -- new sources, via WebSearch/WebFetch,
2025-2026 preferred.

Then judge it against the frame **in numbers**, and write the judgement down
whether it is yes or no:

| axis | this frame |
|---|---|
| dense params | 94M (plus ~1.61e9 n-gram table params) |
| unique tokens | ~250M (10 frozen shards) |
| tokens consumed | ~300-500M, i.e. ~2 passes |
| budget | 300 s on one H200 |
| adoption gate | 0.002614 val_bpb |
| intrinsic effects | ~1e-3 |
| throughput effects | ~1e-2 |

A paper whose smallest model is 18x ours and whose shortest budget is 100x ours
is not evidence about this frame, however good the method is. **Say so and move
on** -- a recorded "no, and here is the scope arithmetic" is as valuable as a
"yes", and it is what stops the corpus filling with inert claims.

**Also check whether we already have it.** The corpus flagged Engram
(arXiv 2601.07372) as a priority mechanism to implement. Reading it revealed
`train.py` *already implements all five of its architectural features* --
multigranular orders, multi-head hashing, per-layer decorrelated primes, gated
fusion into the value stream, projection to model dim. The 1.61e9 n-gram table
parameters ARE Engram, and E7/E8 had already mapped their capacity optimum. An
hour of reading would have prevented that entry from sitting on the priority list.

## Commit messages go through a FILE, never an inline -m string

Two commits this campaign have been broken by shell metacharacters inside an
inline `git commit -m "..."` -- backticks triggering command substitution, and
inner double quotes truncating the message. Both times the fix was to reword the
scientific record to satisfy zsh, which is the wrong thing to be editing.

```bash
cat > /tmp/msg.txt <<'MSG'
...message, any characters, no escaping...
MSG
git add -A && git commit -q -F /tmp/msg.txt
```

Same rule as launchers: **write the text to a file, then hand the file to the
tool.** Prose that must survive quoting is prose that gets edited for the wrong
reasons.

## Never heredoc a launcher through a single-quoted ssh command

Three tranche launches have now failed with `unmatched '` because the script text
contained an apostrophe (`E8's`, `the arm's own step time`, `paper own`) inside an
`ssh host '...'` single-quoted argument. Each time the fix was to reword English
prose, which is the wrong thing to be editing.

**Write the launcher to a local file, `scp` it, then run it:**

```bash
# write locally (Write tool), then
scp -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -P 50002 local_run.sh host:~/setup/eNN/run.sh
tools/ssh_retry.sh 'chmod +x ~/setup/eNN/run.sh && setsid nohup ~/setup/eNN/run.sh > ~/setup/eNN/driver.log 2>&1 < /dev/null &'
```

This also makes the launcher reviewable and diffable before it runs, which matters
because the launcher comment is where the Claim / Mechanism / Hypothesis /
Reasoning for that tranche is recorded.

## Never pick GPUs from an nvidia-smi snapshot; probe them

`nvidia-smi` utilization is a point-in-time sample of a property that varies over
minutes, and it does not see contention on shared host resources at all. Two
tranches were damaged before this was understood:

- **E11** lost 4 of 6 pairs after being launched onto GPUs that read as free.
- **E12** opened at 209 ms against a 91.8 ms cell median, on GPUs that had just
  read 0-16% utilization.

Then the direct measurement: with **all five candidate GPUs reporting 0%
utilization**, a 40-second training probe on each returned **233-237 ms against a
149.9 ms reference -- a uniform 1.58x box-wide slowdown**, equal across every
device. A per-GPU utilization counter cannot show that, because the bottleneck is
shared.

**Use `~/setup/probe_gate.sh REF_MS TOL GPUS...`.** It runs a short real training
job on each candidate and emits only those achieving the reference step time.
Wrap tranche launches in a poll loop that waits for enough GPUs to pass.

**A uniform slowdown is not noise to be screened out afterwards -- it is a
different experiment.** At 237 ms a depth-8 arm completes ~1265 steps instead of
~2016 and a depth-6 arm ~2050 instead of ~3279, so the pair answers "these two
models at half the token budget", where depth 6 is no longer near the data wall.
No post-hoc correction recovers the intended comparison.

**This likely explains campaign history.** The v32 baseline contamination, the
rule that cross-round comparisons are invalid, and much of the residual noise are
all consistent with box-wide slowdowns that no monitoring in use at the time
could have seen.

## When contamination is the MAJORITY, the cell median is corrupted too

Screen v3 compares each arm to its cell median. That assumes the cell is mostly
clean. E34 broke the assumption: a sustained burst contaminated 19 of 20 pairs, so
the in-tranche median step time was itself a burst value (~207 ms for a cell whose
true median is 91.8) and **every contaminated arm would have passed as normal**.

**Rule:** compare against a stored CLEAN REFERENCE median per configuration, not
the median of the tranche being screened. Use the tranche median only as a
cross-check, and if the two disagree by more than ~15%, the tranche is
majority-contaminated and the reference wins.

Current clean references, from many uncontended tranches:

| configuration | median step |
|---|---|
| depth 8, dim 768 (control) | 149.9 ms |
| depth 6, dim 640 | 91.8 ms |
| depth 5, dim 512 | 64.2 ms |

## The contention screen is TWO-SIDED (third revision)

Three versions, three different blind spots. Each was found only after it let a
bad arm through, and each time the arm changed a verdict.

1. **Absolute step floor (`steps >= 1950`).** Assumes one operating point. E7's
   `mult=256` arms legitimately finish ~1620 steps; the floor would have
   quarantined a whole healthy cell.
2. **Median step time alone.** Sees only steady-state health. E8 had two arms with
   in-family step time that lost ~13% of wall clock outside the timed steps.
3. **Duty cycle alone** (`steps / (BUDGET / median_step_ms)`). **Scale-invariant**,
   which is the flaw: contention that slows *every* step uniformly leaves duty at
   ~1.0 while the arm sees half the tokens. E11 seed 51 had duty 1.025 with a
   median step of 192.4 ms against a cell median of 91.8 (2.11x), completed 1598
   steps against ~3270, scored 0.973, passed the screen, and at n=7 moved the
   paired mean from -0.0027 to +0.0005 by itself.

**Both conditions must hold:**

```
duty = steps / (TIME_BUDGET_ms / median_step_time_ms)
clean iff duty >= 0.95  AND  median_step_time <= 1.15 * cell_median_step_time
```

Duty catches wall clock lost *outside* the steps; the step-time band catches the
steps themselves being uniformly slowed. Neither sees the other's failure. Report
BOTH numbers for every arm, at collection time, before looking at `val_bpb`.

**The general lesson:** every screen so far has been a proxy that silently assumed
something was held constant. Before adopting the next one, ask what it assumes and
construct the arm that violates it.

## Plateau rule: switch directions, don't drill (hard rule)

**Classify every proposed experiment before launching it:**

- **KNOB** — an existing env flag, no new code, no new causal structure.
- **MECHANISM** — new code that changes what the model or kernel actually *does*.

**Then apply the budget rule.** After **three consecutive blocks with zero adoptions**, the
next block must be **MECHANISM-only**. Knob experiments are forbidden until a mechanism
flashes a signal above the noise band. This is a hard gate, not a preference — the operator
has directed it repeatedly and it has been violated repeatedly.

**Why it keeps getting violated.** Knobs are cheap to launch and always produce a number, so
a plateau feels like progress: each null "rules something out." But per
`single-seed-noise-floor`, the ±2σ single-seed band is ±0.00122, so a small knob gain is
*invisible even if real*. A knob experiment therefore has ≈0 expected effect **and** ≈0
resolvable information. Exploration has high variance, which is exactly what is needed to
clear the noise band. Every large win in this campaign's history — fa3 (−0.0119, ~4× the
gate), warm `torch.compile` (−0.0439), varlen doc-masking — came from a **structural**
change. **No knob has ever cleared the gate.**

**Worked example of the failure (2026-07-28, one session).** Eleven levers tested,
**zero adoptions**: `DEVICE_BATCH_SIZE`, `NGRAM_TABLE_MULT`, `NGRAM_STATE_ROWWISE`,
`VAL_LOSS_EVERY`, `COMPILE_MODE`, `MUON_NS_STEPS`, `NGRAM_VE_LR_SCALE`,
`MUON_MOMENTUM_CONTINUOUS`, `WARMDOWN_RATIO` — every one a knob, and the session's own
measurement work showed intrinsic quality effects here live at ~1e-3 against a 4.07e-3 gate.
**The arithmetic said no knob could clear the gate, and the knobs kept being launched
anyway.** Record the classification in the block's paper so the drift is visible early.

## Launch discipline (hard rules — each is a bug we hit)

- **Never run an autonomous cron/loop while working interactively.** The cron fires on idle
  gaps *between messages* and launches a second scheduler that collides with manual launches
  → orphaned runners, lock conflicts, empty GPUs. Autonomous mode and interactive mode are
  mutually exclusive: pick one.
- **Reconcile the train.py hash before launching after any edit.** `authorize_run` fails
  closed on reference-code drift ("expected X, found Y"). After editing train.py: sync to
  remote, bump the reconciliation (inert-when-off note), re-select the challenge, THEN launch.
- **Reap runner children.** When `run_stage` dies/is killed, its `run_gated` children must
  die with it (process group / killpg on exit) — otherwise they hang on the dead
  scheduler-fd, hold SSH connections, and block the next launch. Kill orphans before relaunch:
  `pkill -9 -f run_gated.py`.
- **Reuse SSH connections.** Per-arm SSH + frequent status polling exhausts the remote sshd
  (MaxStartups) and hangs launchers. Use `ControlMaster/ControlPersist`; do not poll the
  remote with one-off SSH — rely on completion notifications + local logs.
- **Clean stale locks before launching.** `/tmp/vibeautoresearch-gpu-locks/` accumulates
  `stage-scheduler*.lock` and `run-*.lock` from crashed runs. The scheduler lock is now
  **per-GPU** (`stage-scheduler-gpu{N}.lock`) so disjoint-GPU experiments run concurrently;
  remove stale files before a fresh campaign.
- **Bounded waits, not silent hangs.** Every launch stage (lock, preflight, handshake, remote
  train.py start) needs a timeout + a clear diagnostic + clean teardown.
- **Leave 1–2 GPUs free** on the shared box; verify a quiet GPU for throughput gating.

## Timing facts

One paired A/B round ≈ 9–10 min wall-clock (≈300 s train + ~5 min `max-autotune` compile +
eval), ≈18–20 GPU-minutes. Cache the compile to shave the warmup across many rounds.
