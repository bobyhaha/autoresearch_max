# Agent Must Read

Read this file **first**, before touching `research/`. It is the corrected,
hard-won checklist from a session that initially bypassed the registries
entirely (ad-hoc logging instead of the mechanism→hypothesis→experiment→run→
evidence→belief→decision chain), then spent significant effort repairing that
mistake. Everything below is either a standing rule or a schema gotcha
discovered by trial and error against the actual `vibeautoresearch` code —
not guessed. `docs/RESEARCH_WORKFLOW_HOWTO.md` is the narrative walkthrough;
this file is the checklist and the gotcha list.

## The one rule that matters most

**The registries are the product.** Not the chart, not the paper PDF, not a
SOTA number. If you find yourself writing results to a side file
(`campaign_log.jsonl` or similar) instead of the registries, stop — that is
the exact failure this file exists to prevent. Every experiment must append
the full chain, **including nulls and rejections** (a rejected hypothesis is
just as valuable a record as an adopted one).

## Before doing anything

```bash
python -m vibeautoresearch validate      # MUST be green (0 errors) before you propose work
python -m vibeautoresearch check-setup
python -m vibeautoresearch summary
python -m vibeautoresearch audit --strict # MUST have no error/warning findings
```

If `validate` errors, **fix the registry before running anything else** — do
not route around it with a parallel logging scheme. If you changed
`research/setup/reconciliation.json` (e.g. reconciling the baseline), you
must recompute its `fingerprint` (see gotcha below) and then re-run
`select-challenge` to re-bind the sticky selection, or `validate` will report
"active challenge selection is stale."

## The chain, in the order you must build it

1. **Intervention** (`research/toolkit/available/interventions.jsonl`) — the
   concrete code change, as a registry object, before you write the patch.
2. **Mechanism** (`research/ideas/mechanisms.jsonl`) — the causal story. If
   `origin_type="literature"`, it needs real `claim_ids` (see below); if
   `origin_type="self_proposed"`, it needs `observation_ids` pointing at a
   real `ObservationRecord`, not empty. `origin_type="internal_experiment"`
   needs a real observation or typed `campaign_batch_ids`; quarantined campaign
   batches keep the mechanism provisional. `origin_type="mixed"` needs both
   literature and internal grounding.
3. **Hypothesis** (`research/ideas/hypotheses.jsonl`) — binds mechanism(s),
   intervention, outcome, a directional prediction, and a **falsification
   rule with a numeric gate**.
4. **Gated experiment** (`research/experiments/gated/experiments.jsonl`) —
   freezes seeds, arms (control **and** treatment required for anything but
   `offline_screen`), budget, promotion gate, and `data_policy.scope_key`.
5. **Runs** (`research/experiments/runs/runs.jsonl`) — one immutable record
   per arm per seed. Minimum 3 seeds for `matched_branch`/`discovery`/
   `validation`/`locked_test` stages (single-pair `discovery` is the only
   exception, via `search_policy.stage_pairs == 1`).
6. **Evidence** (`research/knowledge/internal/run_evidence.jsonl`) — the
   four blocks (`facts`, `analysis`, `trust`, `assessment`), citing the real
   `run_ids` you just created (not dropped, not fabricated — see gotcha).
7. **Refinement**: a **decision** (`research/refinement/decisions.jsonl`,
   adopt/reject/hold) AND an **evidence update**
   (`research/refinement/evidence_updates.jsonl`, `result` ∈
   `support/weak_support/mixed/null/oppose/invalid`) closing out the gated
   experiment. `audit` flags `experiment_without_refinement` if you skip the
   evidence update — it is not optional bookkeeping.
8. **Belief** (`research/knowledge/beliefs.jsonl`) when the evidence settles
   something durable, with a `scope.scope_key` matching the current
   reconciliation. Supersede, never edit, a prior belief.

A cycle that leaves any link empty is unfinished — `check-gate`/`audit`
enforce this per experiment, not just globally.

## Schema gotchas (each cost real time to discover — don't repeat)

Every versioned record class lives in `vibeautoresearch/{toolkit,ideas,
experiments,knowledge,refinement,setup}.py` as a dataclass with
`__post_init__` validation and a `.fingerprint`/`.to_dict()`. **Always
construct records through these classes**, never hand-roll the dict — the
class computes the correct fingerprint and catches shape errors immediately
instead of at `validate` time.

- **Enums are stricter than they look.** `claim_type` is one of `{causal,
  correlational, descriptive, mechanistic, method_definition, negative_result,
  predictive, temporal_association}` — **not** `empirical`. Evidence
  `trust.*` fields are one of `{adequate, not_applicable, strong, weak}` —
  **not** `exact`/`indirect`/`low`. Evidence `assessment.relation` is one of
  `{supports, opposes, mixed, inconclusive, not_tested}` — **not**
  `contradicts`. `evidence_update.result` is one of `{support, weak_support,
  mixed, null, oppose, invalid}` — **not** `contradicted`. Evidence
  `source_type` is `{internal_run, literature, offline_analysis,
  quarantined_campaign}` — **not** `run`. `quarantined_campaign` requires
  typed `campaign_batch_ids`, cannot cite a run/gated experiment, and cannot
  claim strength above `weak`. When unsure, grep the module for
  `require_enum(` next to the field.
- **Tags must be lowercase snake_case** everywhere (`residual_stream`, not
  `residual-stream`). Hyphens fail validation.
- **`DecisionRecord` is not fingerprinted and has no `kind` field.** Its
  required fields are `decision` (free text), `reasons` (non-empty list),
  `evidence_ids`, `affected_ids`, `replacement_strategy` (free text,
  required, non-empty). Don't guess a schema resembling other records —
  read `vibeautoresearch/refinement.py:DecisionRecord` directly.
- **Fingerprints are computed, never invented.** Every versioned record has
  a `.fingerprint` property derived from `definition()`. If you edit a
  field on an existing record (e.g. `reconciliation.json`'s `baseline`),
  you must reconstruct the record through its class and use the recomputed
  `.fingerprint` — a stale or hand-written fingerprint fails `validate`
  immediately (`declared fingerprint ... does not match computed ...`).
- **Editing `reconciliation.json` invalidates the sticky challenge
  selection.** After any change, run `select-challenge <id> --reason "..."
  --selected-by "..."` to re-bind it, or every subsequent `validate` fails
  with "active challenge selection is stale."
- **`hypothesis.observable_predictions` is for *dynamic, pre-action*
  observables only.** If your trigger is `{"type": "run_start", ...}`
  (static), leave `observable_predictions=()` — the docs explicitly permit
  omitting it for static triggers, and a `run_start`-triggered
  `matched_branch` experiment will fail
  (`... is not causally available to an online intervention`) if you list
  an outcome-style observable there (e.g. `obs_val_loss`, which has
  `causal_availability.available_before_action: False` and
  `uses_validation: True`). This check applies to `experiment.stage !=
  "offline_screen"`, not to the mechanism's own `observable_predictions`
  (mechanisms aren't subject to this gate).
- **Observables/interventions/contexts used by a non-`offline_screen`
  experiment must have `status` in `{unit_tested, cost_profiled,
  available}`.** A pre-existing observable with `status: "implemented"`
  (e.g. `obs_step_time_ms` at the time of writing) is **not** executable —
  don't fake its status; pick a different, already-executable observable
  for the formal hypothesis binding and keep the informal measurement in
  the evidence `facts` text instead.
- **`data_policy.scope_key` has real requirements**, not free-form: a
  lowercase SHA-256 `data_split_sha256` (compute it —
  `hashlib.sha256(json.dumps(data_split_dict, sort_keys=True)
  .encode()).hexdigest()` — don't invent a slug), `max_steps >= 11`, and if
  `stop_mode == "walltime"` you additionally need an integer
  `time_budget` (seconds) sibling key.
- **`run.intervention_events_path` and other "must be non-empty" text
  fields really mean non-empty** — point them at something real (a log
  path), never `""`, even for a control arm.
- **The formal `experiments/gated` + `experiments/runs` harness is real
  and enforced — it is not optional infra.** A prior campaign in this repo
  (and an earlier copy of it, `vibeautoresearch-baiyu-v3`) skipped this
  layer and only populated mechanisms/hypotheses/beliefs/evidence, citing
  run IDs in evidence that were never backed by actual `RunRecord`s. That
  produced dangling references that broke `validate` for both campaigns.
  Don't repeat it: if evidence cites `run_ids`, those runs must exist as
  real `RunRecord`s (or the field must be empty and `source_type` must not
  be `internal_run`, since `internal_run` evidence requires `run_ids`).
- **Do not launch a governed run informally.** `authorize-run` plus
  `tools/run_stage.py` is the only executable path. For historical direct-SSH
  rows whose launch/config/artifact metadata is incomplete, append a
  `CampaignBatchRecord` with `disposition="quarantined"` and exact source digest.
  Do not fabricate a `RunRecord` or a `run_stage.py`-style tracker payload.
- **Duplicate IDs across re-runs of a fixing script.** If a Python script
  that appends registry records fails partway through (very likely while
  you're iterating on the schema), the already-succeeded `append()` calls
  already wrote to disk. Re-running the same script naively duplicates
  those entries → `validate` reports "duplicate ID". Before retrying,
  strip any partially-written records with the failing/target IDs from
  every file the script touches, in one pass, or make the script check
  "does this ID already exist?" before appending.
- **A literature-grounded mechanism needs a real paper+claim, not a
  relabeled `origin_type`.** If you want to cite a paper (e.g. Canon
  layers, Allen-Zhu 2025) as a mechanism's origin, register the `pap_`
  and `clm_` records first — don't mark `origin_type="self_proposed"` to
  dodge the `claim_ids` requirement just because the claim isn't
  registered yet.

## Multi-agent proposal rounds (propose → critique → rank)

When generating a block of candidate levers (see `paper_012` for the
template): run 2–4 **proposer** agents on distinct angles (e.g.
convergence/architecture, throughput/kernel, data-efficiency), each grounded
in the actual current-scope beliefs and the real SOTA code — not
recollection. Then run one **adversarial critic** agent over the pooled
slate to cut weak candidates with specific reasons and produce a final
ranked execution order with pre-registered gates. Write the block as a
**hypothesis paper** in `AI_papers/` (ICML two-column `.tex`, see below)
before running anything from it — the paper is a precondition, not a
summary written after the fact.

Two protocols this repo has learned to apply to every block:

- **Data-invariance check**: re-run any adopted lever at a smaller data
  slice (fewer shards); if the win shrinks or inverts, it's an
  operating-point artifact (an epoch-count trade), not a real effect —
  reject it even if it cleared the primary gate.
- **Kernel-maturity protocol**: for a *new architectural component*, judge
  **fixed-step (equal-updates) quality first**, separate from wall-clock. A
  naive first-pass kernel may regress step-time without the mechanism being
  bad. If fixed-step quality is null, reject the mechanism outright — a slow
  kernel cannot save it. If fixed-step quality clears the gate but the naive
  kernel regresses wall-clock, that is a **refinement task** (fuse/tune the
  op), not a rejection.

## Papers: all in ICML style, PDF required

Every `AI_papers/paper_NNN_*.tex` uses the ICML two-column format
(`icml2024.sty`, already in that directory) — see any of papers 001–012 as a
template. Compile with `tectonic paper_NNN_....tex`. For quick markdown
drafts that also need a PDF, use `AI_papers/md2pdf.sh <file.md>` (handles
unicode glyphs LaTeX chokes on — `≈ ≫ λ κ ≥ ≤ σ × · → Δ ε` — before invoking
pandoc+tectonic); but the **canonical, citable paper is always the `.tex`**,
convert markdown drafts to `.tex` before calling the block "published."

Cadence: a deep-analysis paper **every 5 rounds**, always with real
mechanism, falsifiable predictions, and honest negative results — a null
result documented with its mechanism is as valuable as a win.

## Never share a mutable launch file across parallel background launches

If you configure a run by writing a **shared** file (`repo/data_split.json`,
a shared wrapper script, an env file, etc.) then backgrounding the launcher
(`setsid nohup ... &`), the launcher's own `cp`/read of that file races
against your very next write for the *next* experiment. This has bitten
this repo twice: once corrupting the wrapper script itself (mid-file
overwrite while a backgrounded process was reading it), once — far more
dangerously — silently swapping which **data split** two experiments
actually trained on, producing a fully-formed, plausible-looking, completely
wrong result (a config change silently applied to the wrong arm). The second
one is the dangerous case: the run *succeeds*, produces a real `val_bpb`,
and nothing in the immediate output flags the mismatch — only a downstream
sanity check (comparing against an established baseline curve) caught it.

**Rule:** either (a) give every launch a unique wrapper/config file name, or
(b) after writing the shared file and backgrounding the launch, **block**
until that specific run's own copy of the file appears
(`for i in $(seq 1 60); do [ -f runs/$e/data_split.json ] && break; sleep 1; done`)
before writing the shared file again for the next experiment. Then verify
the copied file's content matches what you intended, per run, before trusting
any result from that batch — don't just check that the process started.

**This bit a third time even after being documented once**, in a subtler
form: two launches that *did not themselves* touch the shared config file
(they were meant to keep the current default) were started in the same
script as three other launches that *did* rewrite it moments later for a
different data split. Because the "safe" ones never wrote the file, they
looked like they didn't need the guard — but they still read it via their
own `cp`, and that read raced against the later writes exactly the same way.
**The guard is required for every launch in a batch that touches a shared
config file, including ones that only read the current value and never
write it themselves.** If any launch in the batch writes the shared file,
every launch in that batch — reader or writer — needs the blocking check
before the next write happens.

## In-place writes into a tensor already read via an earlier slice break autograd

An attempted "optimization" — preallocating a buffer tensor and filling it
incrementally with `buf[i] = new_value` instead of a Python list +
`torch.stack` each layer — crashed with `RuntimeError: one of the variables
needed for gradient computation has been modified by an inplace operation`.
The failure: the code took a slice of the buffer (`buf[:n_filled]`), used it
in the forward computation, and *then* wrote into the same buffer
(`buf[n_filled] = x`) later in the same forward pass. Autograd tracks
tensor *versions*; a slice taken before a later in-place write to the parent
tensor becomes stale, and backprop through the stale slice fails. This
pattern — read a view, mutate the underlying storage, expect the earlier
view's gradient to still be valid — does not work with in-place writes, no
matter how it's expressed (indexing assignment, `.copy_()`, etc.). If you
want the preallocation's speed without this trap, either (a) `torch.cat`
incrementally (reallocates, but is autograd-safe, and was the original
Python-list-based approach's actual behavior under the hood), or (b) call
`.clone()` on any slice you take *before* the tensor it was sliced from is
mutated again later in the same forward pass.

## GPUs and analysis run in parallel, always

Never let a GPU sit idle while you write records or a paper — launch the
next ranked experiment first, then do the bookkeeping/analysis concurrently
while it trains. Use background waits (`run_in_background`) and only block
on a result when you're about to log it. Leave 1–2 GPUs free on the shared
cluster unless explicitly told to use all of them, and always use
GPU-controlled paired designs (interleave/swap arms across GPUs) for any A/B
— an unpaired comparison on this shared node has previously produced a false
positive (an fp8 "win" that was pure placement noise, retracted in paper
006).

## Literature: dissect every paper the same way

One sentence: **register a `pap_` record, extract its concrete falsifiable
claims as `clm_` records, and attach a four-block `literature_evidence`
(`evd_lit_`) assessment to each claim — facts (what it measured, with
numbers), analysis (method + code reference), trust
(design/replication/scope/directness — mark scope `weak` honestly when the
source paper's regime doesn't match ours), assessment (labeled
interpretation, limitations, and mapping to our mechanisms/hypotheses) —
then link each claim to the mechanism/hypothesis it informs.** A claim from a
larger-scale or longer-training regime is a scope-mismatch for our
50M-param/300-second/10-shard frame and must be labeled as such, not
silently adopted. If a mechanism grounded in a claim is later falsified in
our frame (as happened with Canon layers, paper_012), update the
`literature_evidence.assessment` to say so explicitly — the literature
record is not a one-time snapshot, it's meant to accumulate our own
falsification history against it.

## Known standing gaps (honest, not blocking)

As of the v1.7 integrity restoration, `audit --strict` is warning-free while
these informational limits remain visible:

- 31 historical RunRecords predate challenge-selection fingerprints. They are
  retained facts but are structurally ineligible for modern authorization.
- 128 direct-SSH chart rows from blocks 15–23 have no complete RunRecords. Ten
  exact-digest `CampaignBatchRecord`s quarantine them; weak
  `quarantined_campaign` evidence preserves reportable facts without upgrading
  them into adoption evidence.
- The FA3 document-mask and fixed-batch token-law beliefs are current-scope but
  provisional until prospectively reproduced through `authorize-run` and
  `tools/run_stage.py`.
