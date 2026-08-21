# Agent Protocol

## STANDING DIRECTIVES (operator, 2026-07-31) — re-read these every firing

These bind regardless of what any loop prompt says. They exist because each was
violated at real cost.

### 1. When stuck, STOP AND REFLECT — do not run another lever

**Trigger it explicitly.** You are stuck if ANY of these holds:
- two consecutive blocks closed with no adopted result AND no confirmed mechanism;
- a result you already reported has to be withdrawn;
- the same class of error recurs a third time;
- you are about to run a variant of something that just failed (WD 0.15 after
  WD 0.3 failed) rather than a different mechanism.

**Then do this, in writing, before launching anything:**
- read back over the last blocks' records and campaign_log rows, not your memory
  of them;
- ask what the failures have IN COMMON — the common cause is the finding, not
  each individual null;
- ask what you assumed that was never measured, and measure the cheapest one;
- write the reflection into `docs/` and commit it.

The two most valuable results of this campaign came from exactly this move: the
1e-3-vs-1e-2 selection rule (derived from sixteen failures) and the discovery
that `WEIGHT_DECAY` cannot reach the memoriser (derived from asking why a
refutation refuted). Reflection is not overhead between experiments; it is where
the mechanisms come from.

### 2. Search the internet for papers, constantly, and analyse deeply

At least one NEW primary source per block via WebSearch/WebFetch, 2025-2026
preferred. Not the local corpus — the local corpus is 123 claims of which 7 ever
did predictive work. Deep analysis means:
- extract the actual mechanism, equations and hyperparameters, quoting numbers;
- state explicitly when a detail is ABSENT rather than inferring it;
- compare scope to this frame IN NUMBERS (params, unique tokens, epochs, budget);
- compute the break-even against the token law BEFORE proposing implementation;
- check whether we already implement it (Engram sat on the priority list while
  already being fully implemented);
- record the verdict either way in `research/knowledge/external/reading_log/`.

### 3. If the box disconnects, retry until it is back

Never let an SSH failure end a turn or silently stall the loop. Use
`tools/ssh_retry.sh 'cmd'` (8 attempts, linear backoff, keepalives). A waiter
already died at exit 255 mid-run. If the box is genuinely unreachable, say so,
keep the loop alive, and re-probe next firing.



## Must follow

When I say run experiments nonstop, I mean it: continue until I say stop or there is
a genuine blocking failure. Registry, setup, reporting, and authorization failures
are genuine blockers: fix them before launching. Constantly check available GPUs
after the controls are green, and use them for already-authorized work. An idle GPU
is correct when no complete gated experiment is authorized. Use
`time.time()` for elapsed. Escalate loudly if the shared box is 100% held by other
tenants for an extended period (say so as the headline; ask whether GPUs can be
reclaimed) — do not quietly wait. Report a concise result line in the conversation
each cycle (new paired Δ, launch/completion, free-GPU count, any SOTA, any paper) —
do not run silent. Only the 5-min frame (`walltime_5min_h200`) is live.

### Keep authorized GPUs and analysis in parallel

GPU time is the scarce resource; analysis is nearly free and runs concurrently.
Every cycle, in order: (1) **integrity first** — require green `validate`,
`audit --strict`, `check-setup`, and hourly-report status; (2) **scheduler next** —
feed free GPUs only with exact tuples returned by `authorize-run` and
`tools/run_stage.py`; there is no scratch/direct-SSH fallback; (3) **analysis
concurrent** — do idea generation, papers, and the next gated proposal while
authorized experiments run; (4) **verify every launch** — confirm the run printed
`RESOLVED_CONFIG` and reached a step (allow ~90 s max-autotune compile); never
report “launched” unverified.

## Startup

Read in this order:

1. `research/knowledge/RESEARCH_STATE.md`;
2. `research/knowledge/LITERATURE_SYNTHESIS.md` for the editorial prior map and
   ranked queue, checking that its generated snapshot is current;
3. **current-scope** beliefs and their evidence (see Belief freshness below);
4. available tools and open capability gaps;
5. active mechanisms and hypotheses;
6. proposed and gated experiments;
7. recent decisions and audits.

Run `python -m vibeautoresearch validate` and
`python -m vibeautoresearch check-setup` before proposing work. A validation or
reference-reconciliation failure blocks experiment execution.

## Belief freshness (do not inherit stale ideas)

The agent develops from **recent, current-scope** beliefs — never from stale ones.

1. **Every belief carries a `scope_key`** (data-split hash, `max_steps`, `stop_mode`,
   `outcome_id`) matching `research/setup/reconciliation.json`. A belief measured
   under a different scope is not evidence for the current one.
2. **`render-state` demotes out-of-scope beliefs automatically.** `RESEARCH_STATE.md`
   lists only *current-verified* beliefs at the top; everything whose `scope_key`
   ≠ the current scope drops to `Demoted beliefs [unscoped]` and MUST NOT be treated
   as established. When the frozen challenge changes (e.g. the 2000-step reset), the
   entire prior config-tuning ledger demotes at once — that is intended.
3. **Build forward from the most recent evidence.** When proposing, cite the latest
   current-scope belief(s) on the lever family and extend them; do not re-litigate a
   demoted result as if it were open, and do not resurrect a demoted belief without
   first re-confirming it under the current scope.
4. **Supersede, never edit.** A changed interpretation appends a new belief version
   that `supersedes_belief_id` the old one; the old record stays in append-only
   history. Recency is by version/`created_at`, so the newest current-scope version
   is the operative belief.
5. **Freshness check each session.** After `render-state`, if a belief you are about
   to build on is not in the current-verified list, treat it as an open question,
   not a settled fact. A sub-floor single-seed delta never becomes a belief.

## Registry completeness gate (do not leave registries empty)

The registries are the product, filled **proportionately**: each experiment
populates the records it depends on and produces — not every registry on every run,
and not all claims before starting. `check-gate` and `audit` enforce this per
experiment (a run's motivating observable must be registered; only the claims that
run cites must be assessed; a completed run with no evidence is an audit finding).
Before gating an experiment and before ending a session, satisfy — for that
experiment — the following, and clear its `audit` findings:

- The motivating **observable(s)** are registered in `toolkit/available/observables.jsonl`
  (OPHIS reasons from observables; an intervention with no registered observable is
  out of protocol). Register the internal dynamics you measure as you measure them.
- Every **claim** relied on has a `literature_evidence` assessment (`check-gate`
  enforces this).
- Any missing capability is filed as a **tool proposal + capability gap**, never
  silently assumed.
- The cycle appends **mechanism → hypothesis → gated experiment → run → evidence →
  belief** (including nulls/contradictions) and a **decision**. A cycle that leaves
  any of these empty is unfinished.

An empty-registry or unassessed-claim `audit` warning is an open task, not an
accepted end state.

## Archiving and cleanup (deprecate with provenance, never delete)

Keep the active registries lean without losing history. Two distinct mechanisms:

- **Beliefs (append-only)** — never edit or delete. Archive a belief by appending a
  **superseding** version: set `supersedes_belief_id` to the old id and a status in
  {`challenged`, `contradicted`, `context_dependent`, `inconclusive`}. Stale-scope
  beliefs (wrong `scope_key`) are demoted automatically by `render-state` to
  *Demoted beliefs [unscoped]* — no action needed; they never appear as current.
- **Useless ideas and tools** (mechanism, hypothesis, observable, intervention,
  context, outcome) — archive with the helper, which sets the record
  `status="deprecated"` in place and appends an append-only `DeprecationRecord`
  (reason + optional replacement + evidence). `render-state` then drops them from the
  active/executable views; the record and its history stay on disk:

  ```bash
  python tools/archive.py <object_type> <object_id> "<reason>" \
      [--replacement <replacement_id>] [--evidence id1,id2]
  python -m vibeautoresearch validate && python -m vibeautoresearch render-state
  ```

When to archive: a hypothesis that is bracketed/null and will not be revisited at the
current scope; a mechanism no live hypothesis builds on; an observable that never
informed a gated experiment and has no planned use. Archiving is cleanup, not a
verdict — a deprecated negative result still lives in beliefs/evidence. Do the pass at
each refinement boundary (alongside `snapshot-state`) so the active set reflects only
what the current-scope search is actually using.

## Knowledge ingestion

For each paper:

1. register bibliographic metadata;
2. split the paper into atomic claims;
3. type each claim as descriptive, correlational, predictive, temporal, causal,
   mechanistic, negative, or definitional;
4. record claim scope and limitations;
5. record reported facts separately from the agent assessment;
6. use categorical trust judgments with reasons;
7. for predictive, causal, and mechanistic claims, record the direction of effect
   and the scope in which it holds.

Scope the sweep to the declared lever family and its leading alternatives before
the experiment. Assess existing claims before extracting more, and create a
literature-evidence record in the same pass as each new claim. Then organize:
group claims into `literature` mechanisms by theme, note where papers disagree,
and update `research/knowledge/LITERATURE_SYNTHESIS.md` when the assessed priors
materially change the ranked bet-list. Refresh its machine-owned snapshot with
`python -m vibeautoresearch render-literature`; a fingerprint change requires
human review of the editorial queue, not blind regeneration.

Coverage gate: do not freeze an experiment that touches a method with no backing
paper, claim, and literature-evidence assessment.

Do not start an experiment merely because a paper is interesting.

## Idea generation

LET MULTIPLE AGENTS TO GENERATE IDEAS AND CROSS-VALIDATE
THIS IS VERY IMPORTANT: Propose a mechanism with a causal chain, assumptions, competing explanations,
and predicted observable/intervention behavior. Convert it into a falsifiable
hypothesis with exact tool IDs and versions.

If a required tool is not implemented, create a tool proposal and capability
gap. Do not approve an experiment that pretends the tool exists.

For a static run-start or fixed-schedule method comparison, omit observables and
contexts rather than routing the hypothesis through a dummy training-loss tool.

## Budget frames

**The ONLY active frame is `walltime_5min_h200`** (`STOP_MODE=time TIME_BUDGET=300`,
one H200). Its current baseline, effective σ, seed requirement, and tolerance come
only from `research/setup/reconciliation.json`; do not copy a historical number
from this prose into a gate. This is
the adopt frame — all beliefs, adoptions, and SOTA claims are made in it, judged
against its own baseline and floor, with fresh concurrent paired controls.
Throughput is part of this metric: more steps in 300 s directly lowers `val_bpb`,
so a genuine throughput win (more steps at equal per-step quality) is a first-class
result, and a lever that *costs* steps must overcome that cost.

**The `fixed_steps_2000` frame is DORMANT — do not work on it.** Do not run it, do
not propose for it, do not reason from step-frame-specific arguments (e.g. "at more
steps the knee moves"), and do not `select-challenge fixed_steps_2000`. Any belief
tagged with the step-frame `scope_key` is out of scope. If it is ever revived, that
is an explicit operator instruction, not an autonomous choice.

## Exploration vs refinement

Label every hypothesis and experiment `track:refine` or `track:explore`.
Refinement perturbs a bracketed setting locally; exploration makes a large or
structurally novel change motivated by a literature mechanism and predicting a
large effect. Reserve budget for exploration and raise its share when refinement
plateaus, per `docs/COST_AND_GATES.md`. Stop a coordinate search after `K`
consecutive rejected pilots rather than continuing to probe both sides of every
knob.

Exploration spans all axes -- data order (content fixed), attention, normalization,
initialization, activation, and optimizer swaps -- not just optimizer knobs; pick by
expected metric effect at the current scale. Give a candidate a fair test before
rejecting it: a structurally new component that loses as a drop-in must have its own
key hyperparameters retuned before it is called a failure -- do not generalize one
drop-in loss into "that axis does not work". For a data-order intervention, change a
deliberate policy (curriculum, no-reshuffle-between-epochs, packing order) held
identical across the paired seeds; a fresh random order only re-measures seed
variance and is confounded with it.

Enforce the **firm 6:4 split — 60% `track:explore`, 40% `track:refine`** (see
`docs/COST_AND_GATES.md`): over any trailing window of 10 launched experiments at
least 6 are `track:explore`, and no more than **2** consecutive `track:refine`
experiments run before an exploration one. If explore falls below 6/10, the next
slots must be exploration until the ratio is restored.

## Experiment proposal

Prefer the cheapest informative gate:

1. reuse an existing run bank offline;
2. run one-seed pilot only after offline value is plausible;
3. use matched checkpoint branches to separate trigger value from action value;
4. promote only stable candidates to multi-seed discovery;
5. validate without changing the hypothesis;
6. expose locked data once, after all choices are frozen.

Every proposal must report:

```text
mechanism and hypothesis IDs
tool IDs and versions
scope and applicability
prediction and timing
controls and falsifier
primary estimand and outcome
uncertainty and multiplicity plan
estimated GPU hours and currency cost
hard budget cap and estimation basis
implementation blockers
```

## Execution

Move a proposal into the gated registry only after its definition is final and
its tools pass implementation checks. Freeze the exact current `scope_key` from
`research/setup/reconciliation.json` in `data_policy`. Record `frozen_at`. Runs
must reference the gated experiment fingerprint and may not be invented after
execution.

Check the deterministic gate before moving the record:

```bash
python -m vibeautoresearch check-gate exp_your_experiment
python -m vibeautoresearch authorize-run exp_your_experiment arm_id seed max_steps
```

`check-gate` also blocks unassessed literature claims. Pilots must declare
`promotion_gate.verdict=implementation_only`; all effect verdicts require at
least three paired seeds and a preregistered minimum effect no smaller than the
effective seed-heterogeneity floor.

## Evidence and belief update

After a run:

1. preserve raw run facts;
2. record exploratory patterns as observations;
3. create evidence only from a named analysis and artifact;
4. state whether the evidence supports, opposes, mixes, or fails to test a claim;
5. append a new belief version when interpretation changes;
6. record null, failed, and contradictory results;
7. regenerate research state and run the audit.

After **every completed experiment** (once its final RunRecord is appended), run
`python make_chart.py` and verify that `charts/campaign.html` contains the new
governed run alongside the legacy campaign history. Chart regeneration is part of
experiment completion, including null and failed experiments; do not defer it to a
later reporting batch.

### Observation lifecycle

`observations.jsonl` is editable in place (it is not append-only): advance an
observation by updating its `status` rather than appending a new record.

- `exploratory` — the default. A pattern seen once, recorded with its
  `possible_confounds` so it is not lost and not overstated.
- `replicated` — the pattern held up in independent runs. Still not a
  conclusion: an observation has no `trust`, no `assessment`, and no named
  method, so it can never back a belief on its own.
- `dismissed` — a listed confound explained it away. Keep the record and mark it
  dead so it is not re-litigated.
- `promoted` — it graduated into a mechanism.

To promote, do both in the same pass:

1. append (or extend) a `MechanismRecord` that cites the observation in
   `observation_ids` — this is the provenance a `self_proposed` mechanism needs,
   since it is forbidden from citing literature claims;
2. set the observation's `status` to `promoted`.

`validate` rejects a `promoted` observation that no mechanism cites, so the
label cannot outrun the record that justifies it. From there the normal cycle
applies: mechanism -> hypothesis -> gated experiment -> run -> evidence ->
belief. Promotion never converts an observation into evidence directly.

`audit` reports `self_proposed_mechanism_without_provenance`,
`observation_followup_unresolved`, and `observation_never_triaged` to surface
observations and mechanisms that have drifted out of this lifecycle.

To revise a paper's claim from experiment: never edit the claim record (it is
provenance). Cite the run IDs and the claim ID in internal-run evidence with
relation `opposes` or `mixed`, then append a belief on that claim with status
`challenged`, `contradicted`, or `context_dependent` (superseding the prior
belief), recording the scope in which the paper's result fails to reproduce here.

Never infer causality from a treatment-versus-baseline comparison when a
matched-time or shuffled-trigger explanation remains plausible.

At a meaningful refinement boundary, preserve the generated view with:

```bash
python -m vibeautoresearch snapshot-state 2026_07_16
```