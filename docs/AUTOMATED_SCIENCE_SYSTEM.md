# Automated Scientific Research System

## Quote from Recursive
Consider this claim carefully: "What modifications did our system come up with? The best solutions were not driven by one trick. They combined changes to architecture, short-context memory, auxiliary losses, attention, optimizer behavior, weight decay schedules, compiler settings, and more.

One of the biggest gains came from a richer short-context memory mechanism. The baseline already uses value embeddings; our system extended this idea with hashed bigram and trigram embedding tables, mixed into the attention value path through learned gates. This gave the model a cheap way to use local n-gram information without paying the time cost of slower convolutional or attention-heavy alternatives."


## Objective

The system organizes an agent around five questions:

1. What do we currently know?
2. What can we currently observe, control, and evaluate?
3. Which falsifiable idea is most worth testing next?
4. How can that idea be tested reproducibly?
5. How should the result change current knowledge and future priorities?

The closed loop is:

```text
Knowledge
    ↓
Ideas ← Toolkit
    ↓
Experiments
    ↓
Evidence
    ↓
Refinement ──→ Knowledge / Toolkit / Ideas
```

The loop is not an invitation to generate arbitrary observable combinations.
The agent proposes and interprets; schemas, fingerprints, gates, controls, and
evidence determine what may be treated as knowledge.

## Invariants

### Facts and interpretations are different objects

A run value is a fact. An observed pattern is exploratory. A mechanism is a
causal explanation. A belief is the current interpretation of cited evidence.
Changing a mechanism or belief never changes a run fact.

### Stable IDs and versions

All durable entities have typed IDs. Definitions that affect scientific meaning
also have versions and fingerprints. Changing a formula, tensor source, trigger,
action, outcome detector, control, or experiment design creates a new version.

### Tags are not specs

Tags support retrieval. They never replace a formula, context, action contract,
control, outcome definition, seed list, or analysis plan.

### Markdown is a view

JSONL registries are authoritative. `research/knowledge/RESEARCH_STATE.md` is
generated from them and must not be edited manually.

## 1. Knowledge

External knowledge stores papers, atomic claims, and the evidence reported for
each claim. Trust is assessed per claim using categorical judgments about
experimental design, replication, directness, scope match, and limitations.
Venue is metadata, not a truth score.

Internal knowledge stores exploratory observations separately from structured
evidence derived from immutable runs. A single-seed curve may become an
observation, but it is not automatically evidence for a mechanism.

Beliefs cite evidence IDs and form version chains. A new conflicting result
creates a new belief version rather than rewriting history.

Every empirical belief also carries the setup `scope_key` (data-split SHA-256,
fixed step budget, stop mode, and outcome ID). The generated state demotes
terminal beliefs whose key is absent or differs from the current reconciled
setup. A scope-matched belief is still provisional until it cites structured
evidence with at least three completed seeds (literature-only beliefs are judged
by their literature evidence instead).

## 2. Toolkit

The toolkit separates four concepts:

- observables measure model or training state;
- contexts describe exogenous conditions such as step or scheduled learning rate;
- outcomes define what counts as success and how it is aggregated;
- interventions define exact actions, targets, timing, duration, reversibility,
  safety constraints, and implementation entrypoints.

An observable definition includes tensor sources, axes, reductions or
composition, output units, cadence, expected overhead, causal availability, and
implementation tests. An axis-reduction observable must reduce every source axis
exactly once before it can claim a scalar output.

Validation/test outcomes cannot be visible to an online policy. Composite
outcomes such as plateau recovery must version their detector and aggregation,
not just their final metric name.

Unavailable tools remain proposals. Capability gaps connect a scientific
question to missing capabilities and blocked hypotheses. The agent must not use
a proposed tool as though it were executable.

## 3. Ideas

A separate append-only idea archive supports open-ended generation before a
mechanism is promoted into the scientific graph. Ideas are structured mutation
records with a title, core-hypothesis summary, experimental plan, parent idea
IDs, fixed research direction/subsystem, and reasoned 1–10 scores for
interestingness, novelty, and feasibility. Literature novelty checking may
refine queries for up to ten rounds; it records returned papers, closest papers,
structured evidence, semantic similarity, and a frozen rejection threshold.
Rejected ideas remain archived so the generator does not silently rediscover
them. Agent self-scores rank work but never substitute for empirical evidence.

The portfolio layer limits directional lock-in: after five consecutively
launched stage-1 experiments in one direction, the next stage-1 experiment must
change direction. The saturated direction then cools down for two stage-1
rounds elsewhere, preventing a token one-round pivot. A promoted 3/6/10-pair
continuation remains the same idea and does not count as a new exploration
round. One idea can enter only one funnel chain and can occupy each stage at
most once. This gives a promising idea time to mature while bounding it to four
checkpoints; a restart or substantive variation must become a new child idea
and compete in the archive.

An append-only hourly-paper layer forces periodic synthesis of the portfolio.
Its deterministic Markdown records methods, direction coverage, references,
findings, negative results, limitations, decisions, and next-hour stop
conditions. Unsupported impressions cannot be labelled supported/null/
contradicted, and an overdue paper pauses new authorizations rather than
interrupting in-flight runs.

A mechanism is a causal story with origin, causal chain, assumptions, scope,
observable predictions, intervention predictions, and competing explanations.
A correlation label is not a mechanism.

A dynamically triggered hypothesis binds:

```text
mechanism
+ context
+ observable prediction
+ trigger
+ intervention
+ outcome
+ expected estimand
+ controls
+ falsification rule
```

The trigger and action remain separate. This allows an experiment to distinguish
"the action helps" from "this observable identifies when the action helps."

Static run-start and scheduled interventions may omit observable predictions and
context records when the experiment's reconciled scope key fully defines the
setting. Do not create dummy observable/context objects merely to satisfy the
schema.

Every hypothesis requires a matched no-intervention control. Triggered policies
normally add fixed-time, random-time, shuffled-trigger, or sham controls. All
controls declare checkpoint and data-order matching.

## 4. Experiments

Experiment scope is selected through the data-driven challenge control plane.
`setup/challenges.json` orders challenge 1 (`walltime_5min_h200`) before
challenge 2 (`fixed_steps_2000`) and maps each to a reconciled setup frame.
The latest append-only challenge event is sticky across processes. It pins the
catalog, challenge definition, and setup fingerprints; a stop or stale event
blocks work and never silently falls back or advances. Search policy,
authorization, scheduler capability, manifest, and RunRecord all bind the same
challenge selection.

Experiments progress through:

```text
offline_screen
→ pilot
→ matched_branch
→ discovery
→ validation
→ locked_test
```

Offline screening reuses existing data and must control training step, training
loss, and learning rate. It launches no new arms. A pilot uses one seed to test
plumbing, numerical safety, action execution, and cost. Discovery, validation,
and locked tests require at least three seeds.

Every experiment preregisters arms, seeds, checkpoint matching, randomization,
primary estimand, outcome, baseline covariates, uncertainty method, multiplicity
handling, budget, promotion criteria, and data-access policy.

Proposal records are editable planning objects. A gated record is approved,
fingerprint-frozen, and append-only. Online gated experiments may use only
unit-tested or stronger observables, interventions, and outcomes. Locked-test
data is unavailable to the proposal loop.

Before any gate, a setup-reconciliation record hashes the frozen data/evaluation
harness, records an explicit diff against the named reference code, and checks a
three-seed baseline within 2 effective sigma. Each run records the experiment and
hypothesis, arm, seed, commit, dirty status,
config hash, exact spec fingerprints, tracker/artifact references, intervention
events, outcomes, actual cost, timestamps, and failures. Analysis never rewrites
these facts.

## 5. Refinement

An evidence update states which evidence affects which hypotheses and beliefs.
Audits find duplicates, conflicts, missing references, stale generated state,
completed runs without evidence, and completed experiments without refinement.

Decisions preserve why a direction was stopped or prioritized. Deprecation keeps
the old object and names a replacement; it never deletes history. Snapshots make
changes in research state inspectable over time.

## Populated-ledger policy

This branch contains historical records from multiple experimental scopes.
History remains append-only, but historical terminal beliefs and approved gates
do not become current merely because they were never superseded. Scope matching,
structured evidence, and run authorization determine what may guide new work.
