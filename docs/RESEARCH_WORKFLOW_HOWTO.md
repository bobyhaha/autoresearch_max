# How to Actually Follow the Research Workflow

This is the operational how-to for running a campaign **through the registries**, not
around them. It exists because a 69-round campaign was once run with an ad-hoc
`campaign_log.jsonl` and prose papers while the structured registries sat empty — a
protocol violation. **The registries are the product.** A SOTA number, a chart, and a
PDF are byproducts; the append-only record chain is the deliverable. Read this with
`AGENT_PROTOCOL.md` (rules), `EXPERIMENT_WORKFLOW.md` (per-round loop), and `SCHEMA.md`
(record shapes).

## The non-negotiable rule

Every experiment appends the full chain, **including nulls and contradictions**:

```
mechanism → hypothesis → gated experiment → run → evidence → belief → decision
```

`validate` and `audit` fail closed on a chain that leaves any link empty. If you catch
yourself writing results into a side file, stop — that is the failure this document exists
to prevent.

## Session startup (do this before proposing anything)

Run, in order, and do not skip:

```bash
python -m vibeautoresearch validate        # must be green (errors block execution)
python -m vibeautoresearch check-setup      # reference/reconciliation must reconcile
python -m vibeautoresearch summary          # registry counts + warnings
python -m vibeautoresearch render-state      # regenerate RESEARCH_STATE.md
python -m vibeautoresearch render-literature # regenerate LITERATURE_SYNTHESIS.md + ranked queue
```

Then **read** (protocol startup order): `RESEARCH_STATE.md` → `LITERATURE_SYNTHESIS.md`
→ current-scope beliefs → tools + open capability gaps → active mechanisms + hypotheses
→ proposed + gated experiments → recent decisions + audits.

**If `validate` errors, you are blocked — fix the registry first.** (Historically it was
left broken by a half-done erasure; that is exactly when momentum tempts you to route
around it. Don't.)

## Belief freshness (do not inherit stale ideas)

- Every empirical belief carries a `scope.scope_key` = (`data_split_sha256`, `max_steps`,
  `stop_mode`, `outcome_id`) matching `setup/reconciliation.json`.
- `render-state` puts only **current-verified** (in-scope) beliefs at the top; out-of-scope
  ones are demoted and MUST NOT be treated as settled.
- Build forward from the **most recent current-scope** belief. **Supersede, never edit** —
  a changed interpretation appends a new version with `supersedes_belief_id`.
- A sub-floor single-seed delta never becomes a belief.

## The per-round loop

1. **Idea generation** — distinct angles (throughput/kernels, data-pipeline, free
   quality-per-step, literature→concrete). Ground every idea in the actual code; screen it
   against the already-tried set (`runs.jsonl`). No fabrication. Record selected ideas in
   `ideas/archive.jsonl` (with novelty check) and `ideas/hypotheses.jsonl`.
2. **Author the chain** — for the idea, register/So reuse:
   - the **mechanism** (`ideas/mechanisms.jsonl`) — the causal story;
   - the **observable(s)** you will measure (`toolkit/available/observables.jsonl`) — an
     intervention with no registered observable is out of protocol;
   - the **intervention** (`toolkit/available/interventions.jsonl`) and the **outcome**
     (`out_val_bpb` already exists);
   - the **hypothesis** (`ideas/hypotheses.jsonl`) — binds mechanism, intervention, outcome,
     a directional **prediction**, effect timing, controls, and a **preregistered
     falsification gate** (decision rule + minimum effect + minimum seeds);
   - any missing capability → a **tool proposal + capability gap** (never silently assume).
3. **Freeze the experiment** — write a **gated** experiment (`experiments/gated/…`) that
   freezes the current `scope_key` in `data_policy`, the arms, seeds, and step/second budget.
   `validate` + `audit` must be green before it runs.
4. **Authorize + run** — `python -m vibeautoresearch check-gate` then `authorize-run`
   (rejects a missing/mismatched scope, arm, seed, or budget even if the record says
   `approved`). Launch through the gated runner only — never a raw shell daemon. Verify the
   run actually reached a step (allow ~90 s max-autotune compile) before reporting "launched".
5. **Record the run + evidence** — append an immutable `run_` record referencing the gated
   experiment + exact hypothesis/experiment/outcome fingerprints. Then write the
   **evidence** (`evd_run_…`) with its four blocks: **facts** (measured values), **analysis**
   (method + code ref), **trust** (design/replication/scope/directness), **assessment**
   (labeled interpretation + limitations). Nulls and regressions are recorded as evidence,
   not hidden.
6. **Form the belief + decision** — append a scoped `blf_` interpreting the hypothesis
   (supported / null / contradicted), citing the evidence; append a refinement `dec_`
   (adopt / reject / hold) and, if superseding an earlier belief, an `upd_` evidence-update.
   A completed run with no evidence is an `audit` finding.
7. **Report + snapshot** — update the paired-Δ chart and the hourly report after every
   experiment; commit. On a **real SOTA** (beats baseline past 2σ, confirmed across the
   funnel): snapshot (`sota_snapshots/…`) + commit + republish chart. Publish the hourly
   research paper when due — `check-gate`/`authorize-run` fail closed until it exists.

## Statistical gates (from EXPERIMENT_WORKFLOW.md)

- **Quality levers**: `val_bpb`, paired ≥3-seed, all-same-sign, past the 2σ decision floor
  (effective σ from `reconciliation.json`). Single-seed screens are pilots only.
- **Throughput levers**: gate on **step-time + fixed-step quality-equivalence**; the 2σ
  val_bpb gate (~4.4% throughput) is too coarse for sub-4% wins — **bundle** them.
- **Shared cluster**: GPU-controlled paired designs are mandatory (interleave/swap arms
  across GPUs). Unpaired A/B is confounded by placement (~0.0014 spread) — a retracted fp8
  "win" was pure contention.

## Paper dissection (the literature sub-workflow)

For each paper, one sentence: **register a `pap_` record, extract its concrete falsifiable
claims as `clm_` records, and attach a four-block `literature_evidence` (`evd_lit_`)
assessment to each claim — facts (what it measured, with numbers), analysis (method +
equation/code reference), trust (design/replication/scope/directness), assessment (labeled
interpretation, limitations, and mapping to our levers/mechanisms and our 5-min/data-limited
scope) — then link each claim to the mechanism/hypothesis it informs so it enters the ranked
idea queue, recording scope-mismatches and nulls honestly rather than cherry-picking.**

Concretely:
1. `pap_` in `knowledge/external/papers.jsonl` — id, title, venue/year, url, tags.
2. `clm_` in `knowledge/external/claims.jsonl` — one per testable assertion, phrased as a
   falsifiable statement with the reported effect and its conditions.
3. `evd_lit_` in `knowledge/external/literature_evidence.jsonl` — the four blocks above; set
   `trust.scope` to whether the paper's regime matches ours (budget, model size, data volume)
   — a claim from a data-rich or larger-model regime is **scope-mismatched** for our
   5-min/10-shard frame and must be labeled so, not adopted as-is.
4. Link the claim to a `mech_`/`hyp_`; if it suggests a new lever, spawn an idea +
   hypothesis. Papers with no claim we can test are recorded but flagged low-relevance.
5. `render-literature` regenerates `LITERATURE_SYNTHESIS.md` — the editorial prior map and
   the ranked queue that idea generation draws from.

## The failure modes this document exists to prevent

- **Routing around a broken registry** instead of fixing it first (validate must be green).
- **Optimizing the byproducts** (SOTA/chart/PDF) while the registries stay empty.
- **Ad-hoc side logs** (`campaign_log.jsonl`) in place of the record chain.
- **Skipping the scope check** — running a frame whose `scope_key` doesn't match the
  selected challenge/reconciliation, so every belief is silently off-scope.
- **Hiding nulls** — negatives, regressions, and retractions are first-class evidence.
- **Not measuring the frame first** — run the cheap frame diagnostics (data slope, compute
  slope, kernel profile) early; they decide whether architecture search can even pay before
  you spend rounds inside the box.
