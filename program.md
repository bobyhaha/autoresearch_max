# Automated Scientific Research Program

Your objective is to map and exploit headroom in the repeated-data training
regime (roughly 1.6 to 2.7 epochs at the current frozen corpus), with emphasis on
how recipe optima and mechanisms shift with training horizon. The external RSI
number is context, not the objective; success means a reproducible improvement
over a fresh, concurrent, same-scope local control. Generating many ideas or
obtaining an isolated positive run is not the goal.

## Required startup

Read:

1. `README.md`;
2. `docs/AUTOMATED_SCIENCE_SYSTEM.md`;
3. `docs/AGENT_PROTOCOL.md`;
4. `research/knowledge/RESEARCH_STATE.md`;
5. the structured records referenced by the current state;
6. relevant training code only after the research question is clear.

Run:

```bash
python -m vibeautoresearch list-challenges
python -m vibeautoresearch validate
python -m vibeautoresearch audit
python -m vibeautoresearch check-setup
python -m vibeautoresearch hourly-report-status
```

A schema error blocks further execution. A stale research state must be
regenerated with `python -m vibeautoresearch render-state`.
Any failed/pending setup reconciliation, frozen-file hash drift, or baseline
discrepancy above 2 effective sigma stops the line before idea selection.

## Compliance: follow the md files, and do not leave registries empty

These documents are binding, not background reading. Follow them literally:
`program.md`, `docs/AGENT_PROTOCOL.md`, `docs/COST_AND_GATES.md`,
`docs/AUTOMATED_SCIENCE_SYSTEM.md`, and `agent.md`. When an instruction here and
your own instinct disagree, the document wins; if a document is wrong, fix the
document in the same change, do not silently deviate.

The structured registries are the deliverable, not a side effect. This is
**proportionate, per-run, not all-at-once**: each experiment must populate the
records *it* depends on and produces — not every registry on every run, and not all
1012 claims before starting. Over a session the registries fill in; `check-gate`
(pre-run) and `audit` (post-run) enforce exactly this, per experiment:

- `check-gate` refuses to freeze an experiment whose motivating **observable** is not
  registered-and-executable, or whose mechanism cites **literature claims** without an
  evidence assessment — only the claims *that* experiment relies on, not the whole base.
- `audit` flags a **completed run with no evidence**, a hypothesis with no evidence, or
  a belief with no evidence — so a run that produced nothing shows up as an open task.

Concretely, for each experiment as the loop runs:

- **Observables** (`toolkit/available/observables.jsonl`) — OPHIS is observable-driven.
  Every hypothesis must name the exact observable(s) it reasons from, and each such
  observable must be a registered record before the experiment is gated. Registering
  the internal dynamics you actually measure (norms, entropy, gradient/curvature,
  accuracy, gaps, drift) is the core work, not optional instrumentation. Do not
  propose an intervention whose motivating observable is not in the registry.
- **Literature evidence** (`knowledge/external/literature_evidence.jsonl`) — every
  claim you rely on needs an assessment record; `check-gate` blocks a proposal that
  depends on an unassessed claim. Never leave ingested claims unassessed and then
  build on them.
- **Tool proposals / capability gaps** (`toolkit/proposed/*`, `toolkit/capability_gaps.jsonl`)
  — if a needed observable/intervention is not implemented, register a tool proposal
  and a capability gap. Do not pretend a tool exists, and do not silently skip the
  measurement.
- **Mechanisms, hypotheses, evidence, beliefs, decisions** — populate them each cycle;
  a run with no evidence/belief update (including nulls and contradictions) is
  incomplete. Run `python -m vibeautoresearch audit` and clear its findings — an
  empty-registry or unassessed-claim warning is an open task, not an accepted state.

## Research loop

### 1. Knowledge (literature-first gate)

Read enough to bound the declared lever family before running anything. This is
a targeted blocking gate, not a mandate to re-scan every method in `train.py` for
each experiment. Prefer assessing existing relevant claims over extracting more.
Do not add a claim unless its literature-evidence assessment is created in the
same ingestion pass.

For each paper, register bibliographic metadata and split it into atomic claims.
Type each claim (descriptive, correlational, predictive, causal, mechanistic,
negative, definitional). Where the paper supports it, record a **directional
prediction**: which knob or intervention moves the outcome, in which direction,
and in what scope. Record paper-reported facts separately from the agent
assessment, and do not use venue prestige as a truth score. Keep internal run
facts separate from agent observations and beliefs.

Then organize the result into conclusions; do not leave it as a pile of claims:
group claims into `literature` mechanisms by theme, note explicitly where papers
disagree, and maintain a ranked bet-list of the interventions the priors most
favor in `research/knowledge/LITERATURE_SYNTHESIS.md`. Its generated snapshot
binds the editorial queue to current registry counts, assessed-claim coverage,
scope, and planning-input fingerprint. Run
`python -m vibeautoresearch render-literature` after registry changes and review
the queue whenever that fingerprint changes. This synthesis, not an isolated
result, is the entry point to idea generation; it never overrides current-scope
beliefs or authorizes a run.

Coverage gate: do not freeze an experiment that touches a method with no backing
paper, claim, and literature-evidence assessment. Tuning a knob whose source was
never assessed is out of protocol — it is search without knowledge.

### 2. Toolkit

Inspect what observables, contexts, outcomes, and interventions are actually
implemented. A tool is usable only when its registry status and implementation
test support execution. If a required capability is missing, create a tool
proposal and capability gap instead of pretending it exists.

Do not manufacture an observable or context for a static run-start knob. Static
and scheduled hypotheses may omit those fields when the reconciled experiment
scope defines the setting. Use observable-conditioned machinery only when the
observable genuinely chooses or times the action.

### 3. Ideas

Propose a causal mechanism with assumptions and competing explanations. Convert
it into a falsifiable hypothesis linking exact observable, trigger,
intervention, outcome, controls, timing, failure condition, and estimated cost.

### 4. Experiments

Use the cheapest informative gate:

```text
offline_screen
→ pilot
→ matched_branch
→ discovery
→ validation
→ locked_test
```

Offline screens use existing data and control step, training loss, and learning
rate. A pilot uses one seed only to test implementation, safety, and cost. Online
experiments require matched control and treatment branches. Discovery,
validation, and locked tests require at least three seeds.

Two peer, adopt-capable challenges are registered in a data-driven catalog,
ordered as (1) `walltime_5min_h200` (`STOP_MODE=time`,
`TIME_BUDGET=300`, passed) and (2) `fixed_steps_2000`
(`STOP_MODE=steps`, `MAX_STEPS=2000`, pending re-measurement). Each is judged
only against its own baseline and floor. Challenge selection is an append-only,
sticky event; omitted scope flags resolve the active challenge and never fall
back to steps. A stop event blocks work rather than auto-advancing. Use
`list-challenges`, `select-challenge`, and `stop-challenge`; see
`docs/COST_AND_GATES.md` and `docs/AGENT_PROTOCOL.md`.
Only an explicit user instruction authorizes the agent to select a different
challenge or stop the current one. Difficulty, null results, an overdue paper,
or portfolio policy never grants that authority.

The same data-driven campaign policy caps a direction at five new stage-1
rounds, keeps an exhausted direction on cooldown for two rounds elsewhere, and
requires an hourly summary paper. Check `hourly-report-status` at session start
and before more GPU work. If due, publish the evidence-aware JSON body with
`publish-hourly-report`; the generated Markdown is immutable. The reporting
gate pauses future authorizations but never kills a run already in flight.

Do not launch training from a proposal. The exact experiment must first be
fingerprint-frozen in `research/experiments/gated/experiments.jsonl`, including
the current data-split/step-budget scope key. Every launcher must pass
`authorize-run` for the exact experiment, arm, seed, and `MAX_STEPS`. Locked data
is never visible to the proposal loop.

### Kernels and external SOTA

**Kernel work is a fair lever.** Writing, refining, wrapping, or swapping compute
kernels (attention backends, fused ops, custom CUDA/Triton, torch.compile
wrapping, memory layout) is a method change inside the frozen field: it alters
neither data content, nor budget, nor evaluation. In the wall-clock frame it is
often the LARGEST lever, because throughput is the metric there -- the same
attention mask is -0.0097 under a block-sparse kernel and +0.045 under a dense
one. In the fixed-step frame a pure-throughput change is ~metric-neutral but can
still shift numerics, so it is measured, never assumed. A kernel that honors a
model property the old backend ignored (e.g. flash-attn windowing vs SDPA
full-causal) is carrying a model change with it and must say so.

**Consult external SOTA.** Papers, leaderboards, reference implementations, and
vendor kernel libraries are legitimate inputs for choosing what to test. They
enter as claims with scope, get a literature-evidence assessment before they can
justify a run, and are never quoted as our result -- absolute numbers do not
cross sandboxes. An externally-SOTA method still has to clear this frame's own
paired-seed gate against a concurrent local control.

### Exploration track (high risk, high reward)

Local refinement — small single-knob perturbations of a bracketed setting — is
not the only allowed work, and a session must not collapse into it. A reserved
share of the experiment budget is spent on exploration: large-magnitude or
structurally novel changes (architecture swaps, new mechanisms, order-of-magnitude
parameter moves) that are motivated by a literature mechanism and predict a large
effect.

Explore vs refine is a distinction of **structure and magnitude, not subsystem**:
swapping the optimizer (Adam->Muon->SOAP), changing a data-order policy, or
replacing an attention or activation block is exploration; nudging a bracketed beta
or learning rate is refinement. Explore across **all** knowledge-base axes -- data
order (content fixed; see Scientific rules), attention, normalization,
initialization, activation, and optimizer swaps -- not just optimizer knobs, and
choose bets by expected effect on the metric at the current scale (an efficiency
method such as MLA/GQA rarely moves fixed-compute `val_bpb`). Budget: a ~40% explore
floor, ~60% while the config space is bracketed, with a cap on consecutive
refinements -- see `docs/COST_AND_GATES.md`.

Tag every hypothesis and experiment `track:refine` or `track:explore`. An
exploration experiment may go straight to a one-seed pilot, is expected to fail
often, and is held to a large predicted effect so that failures stay cheap and
only real wins advance. A failed exploration is a recorded negative belief, not
wasted budget.

The exploration share is dynamic: it rises when refinement stalls and falls back
after a validated win resets the search. See `docs/COST_AND_GATES.md` for the
plateau, reset, and noise-floor rules.

### 5. Evidence and refinement

After execution, preserve run facts. Create observations for exploratory
patterns and structured evidence for named analyses. Append belief updates,
including negative and contradictory results. Record decisions and deprecations
without deleting history, regenerate research state, and audit again.

Revising a literature claim from experiment. A claim records what a paper
asserted; it is provenance, not a belief, and is never edited or deleted to match
a run. If an experiment contradicts or refines a paper's claim: (1) keep the
original claim unchanged; (2) record internal-run evidence that cites the run IDs
and the claim ID with assessment relation `opposes` or `mixed`; (3) append a
belief about that claim with status `challenged`, `contradicted`, or
`context_dependent`, superseding the prior belief; and (4) state the scope in
which the paper's claim fails here (for example, this five-minute H200 SDPA setup)
versus the scope the paper reported. Literature priors are corrected by local
evidence through beliefs, never by rewriting the claim.

## Scientific rules

- A treatment beating baseline does not prove that its trigger is useful.
- Use matched-time, shuffled-trigger, random-time, or sham controls when needed.
- Tags are search aids, not experimental definitions.
- Do not change a hypothesis, outcome detector, primary estimand, or success
  threshold after seeing gated results.
- Do not let validation/test outcomes control online training.
- Record actual cost and observable overhead for every run.
- Never invent missing provenance or hide failed/null runs.
- A paper's claim is provenance: correct it with evidence and beliefs, never by
  editing or deleting the claim.
- A refinement smaller than the measured seed-heterogeneity noise floor is not a
  result; same-seed reproducibility noise is diagnostic only. Keep promotion
  thresholds above the effective floor and require at least three paired seeds
  for any effect verdict. A one-seed pilot may establish plumbing, safety, and
  cost only.
- Stop a coordinate search after K consecutive rejected pilots instead of probing
  both directions of every remaining knob; redirect the freed budget to
  exploration.
- Data content is fixed for fair comparison: an intervention may change data
  *order* (a deliberate policy — curriculum, reshuffle-between-epochs, packing) but
  not which tokens are trained on. An order change must be a policy held identical
  across the paired seeds, never a fresh random shuffle — order is already
  seed-dependent, so a random reshuffle only re-measures seed variance. This
  matters most in the current small-data multi-epoch regime, where how repeated
  data is presented shapes overfitting.

## Execution permissions

This clean branch contains no approved experiments. No autonomous training loop
is active by default. The presence of an approved gated record is the explicit
authorization to execute only that record's arms, seeds, budget, and controls.
