# Schema Reference

## IDs

| Prefix | Entity |
| --- | --- |
| `pap_` | paper |
| `clm_` | claim |
| `evd_` | evidence or exploratory observation |
| `blf_` | belief |
| `obs_` | observable |
| `int_` | intervention |
| `ctx_` | context |
| `out_` | outcome |
| `tlp_` | proposed tool |
| `gap_` | capability gap |
| `mech_` | mechanism |
| `hyp_` | hypothesis |
| `idea_` | generated idea archive record |
| `hrp_` | hourly research-paper record |
| `exp_` | experiment |
| `run_` | run |
| `upd_`, `aud_`, `dec_`, `dep_` | refinement records |

## Versioned definitions

Papers, claims, beliefs, observables, interventions, contexts, outcomes,
mechanisms, hypotheses, and experiments carry deterministic fingerprints where
appropriate. Fingerprints include fields that alter scientific meaning and
exclude display-only notes.

## Evidence

Evidence keeps four separate blocks:

- `facts`: measured or paper-reported values;
- `analysis`: method and code reference;
- `trust`: categorical design/replication/scope/directness assessment;
- `assessment`: explicitly labelled agent interpretation and limitations.

Evidence corrections append a new ID and set `supersedes_evidence_id`.  The
raw JSONL registries remain append-only, but every current-state consumer
resolves each correction chain to its unique terminal record.  Evidence
supersession must be non-branching and acyclic: self-links, missing
predecessors, multiple successors, and cycles are invalid.

A correction must preserve its subject and `source_type`.  Operational
evidence retains the same experiment and every prior run, hypothesis, and
campaign-batch anchor (a correction may add replication anchors, never drop
them).  Every literature correction also retains every prior `claim_id`,
including literature evidence that carries operational anchors; its `paper_id`
may change when deficient secondary metadata is replaced by a primary-source
record.  A genuinely new atomic claim is separate evidence, not a correction.
Consequently, FA4 and Tensorizing Engram may each append a same-claim scope
correction, while their newly extracted H200-scheduler and CP-shared-factor
claims remain separate non-superseding evidence.  The n-gram-regularizer scope
downgrade may supersede its earlier assessment because it retains the exact
paper/claim/hypothesis subject.  One exact, documented Kimi K3
secondary-to-primary edge predates and is grandfathered under this rule.

A belief names the claim, mechanism, or hypothesis IDs it interprets, cites its
evidence, and supersedes an earlier belief rather than editing history.
Supersession must form a non-branching, acyclic chain. The generated current
state shows only the terminal belief in each chain while retaining every prior
record in the append-only registry.

Empirical beliefs may include `scope.scope_key`, containing
`data_split_sha256`, `max_steps`, `stop_mode`, and `outcome_id`. This key is
mandatory for new empirical work. The state renderer automatically separates
missing/mismatched terminal beliefs from current verified beliefs; it never
deletes the historical record.

Non-speculative current beliefs without evidence are reported by both
`validate` and `audit`. Extracted claims are not themselves assessed evidence;
the audit separately reports claims that lack a literature-evidence record.

The generated state treats a hypothesis as conclusively tested only when an
evidence assessment supports, opposes, or mixes the prediction. Inconclusive
and `not_tested` evidence keeps the hypothesis visible. Likewise, an approved
experiment remains visible until it receives a non-invalid evidence update;
the frozen gate record itself is never rewritten.

## Observable

An observable declares sources, tensor axes, reduction/composition, output type
and units, cadence and cost, causal availability, implementation, test, status,
version, and fingerprint.

## Intervention

An intervention declares action, target selector, parameters, timing, duration,
reversibility, cost, safety, implementation, test, status, version, and
fingerprint.

## Outcome

An outcome declares metric, units, direction, split, aggregation, optional
detector, policy visibility, implementation, test, status, version, and
fingerprint. Detector changes require a new version.

## Hypothesis

The hypothesis binds mechanisms, trigger, intervention, outcome, prediction,
effect timing, controls, falsification, and estimated cost. Dynamically triggered
hypotheses also bind contexts and observable predictions; static run-start or
scheduled interventions may omit them instead of registering dummy tools.

## Idea archive

`ideas/archive.jsonl` is append-only. An idea records its mutation parents,
title, hypothesis summary, experimental plan, fixed research direction and
subsystem, and reasoned 1–10 self-scores for interestingness, novelty, and
feasibility. Completed novelty checks preserve up to ten search-query rounds,
returned and closest paper IDs, structured literature evidence, maximum
semantic similarity, and the frozen discard threshold. Selected/tested ideas
must pass novelty and bind a real hypothesis. Parent links must resolve and the
mutation graph must be acyclic.

## Hourly research paper

`reports/hourly/reports.jsonl` is append-only. Each `hrp_` record binds one
reporting period to an exact challenge-selection fingerprint and records its
title, abstract, methods, directions explored, idea/experiment/run references,
evidence-aware findings, negative results, limitations, portfolio decisions,
and next-hour plan. Findings labelled supported, null, or contradicted require
registered evidence. Referenced runs and policy-aware experiments must belong
to the report's challenge.

Each record deterministically renders
`reports/hourly/<report_id>.md`; validation rejects a missing or edited paper.
Reporting cadence, direction block length, and cooldown live in
`setup/challenges.json:campaign_policy`. When reporting is due,
`check-gate`/`authorize-run` fail closed until a new report is published.

## Experiment and run

Proposal and gated experiments use the same schema but live in different
registries. An ID cannot be present in both. Approved/running/completed gated
records require `frozen_at` and may use only executable tools.

New gated experiments freeze the current `scope_key` in `data_policy`. The
launch authorization command rejects a missing/mismatched scope, arm, seed, or
step budget even if an old record still says `approved`.

Policy-aware experiments also freeze an `idea_id` and a `search_policy`
direction. These must agree with the selected archive record and remain
unchanged through the 1→3→6→10 funnel.

Runs are immutable facts referencing a gated experiment and exact hypothesis,
experiment, and outcome fingerprints. Large logs and arrays are referenced, not
copied into JSONL.

## Setup reconciliation

`setup/reconciliation.json` is a fingerprinted, single-current-record gate. It
pins the data split and evaluation-harness hashes, records hashes plus a diff
summary for the named upstream reference, and checks a three-seed baseline
against its expected value within a declared effective-sigma tolerance.

## Challenge catalog and sticky selection

`setup/challenges.json` orders named challenges and maps each challenge to a
reconciled scope and decision-frame ID. Launchers derive budgets and run
environment from the referenced setup frame; the catalog does not duplicate
them as executable constants. Its `campaign_policy` configures direction
tenure/cooldown and the hourly research-paper interval.

`setup/challenge_events.jsonl` is append-only. Each activation or stop records a
contiguous generation, challenge ID, actor/reason/time, and fingerprints for
the catalog, selected challenge definition, and setup reconciliation. The
latest valid event is sticky. Missing, stopped, or stale selection state fails
closed, and an explicit compatibility scope cannot select a different
challenge. Authorization, scheduler capabilities, manifests, tags, and
RunRecords pin the selection and challenge fingerprints.
