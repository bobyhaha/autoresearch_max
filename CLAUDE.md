# CLAUDE.md — cold-start handoff for Simplify Autoresearch v3

Read this file completely before changing code, registering science, starting a worker,
or using a GPU. This is a scientific research system, not a generic hyperparameter loop.
Its purpose is to turn papers and experimental evidence into narrowly supported claims
while preserving enough provenance to reconstruct every decision.

## 0. First-machine bootstrap

This source-only repository deliberately contains no `.env`, active registry, literature
corpus, generated paper, machine-specific data manifest, filled benchmark scope, Python
environment, experiment result, hypothesis, queue job, or operational state. Read
`FRESH_START.md`, initialize a new local registry, and complete machine setup before running
the session commands below. If `runs/scope.json` is absent or contains `replace_with_...`,
setup is incomplete and GPU work is forbidden. Do not edit fixed `prepare.py` to make setup
easier.

## 1. Mission

Improve validation bits per byte (`val_bpb`, lower is better) in the strict Karpathy
300-training-second, single-H200 scope. Generate new ideas from inspected literature,
mechanisms, and past failures; test one minimal causal intervention at a time; and promote
only effects that survive held-out-seed confirmation.

The objective is not to keep GPUs busy, maximize trial count, copy paper methods verbatim,
or produce an attractive leaderboard. The objective is defensible knowledge.

The governing chain is:

```text
agenda -> search -> full-text source -> atomic claims -> causal mechanism
                                                   -> active failure lessons
                                                   -> independent idea debate
                                                   -> falsifiable hypothesis + activation
                                                            |
                                                            v
scope -> exact controls -> candidate -> bank gate -> paired promotion
                                                            |
                                                            v
                                          independent result analysis
                                                            |
                                                            v
                                     experiment claim / failure lesson
                                                            |
                                                            +--> next hypothesis
```

No model candidate may skip from “paper” or “idea” directly to code execution.

## 2. Sources of truth and reading order

Read in this order:

1. this file;
2. `README.md` for the supported command surface;
3. `docs/SCIENTIFIC_METHOD.md` for claims, confidence, activation, and lessons;
4. `docs/ARCHITECTURE.md` for immutable records and execution authority;
5. `program.md` for the detailed operating protocol.

Use this truth hierarchy whenever statements disagree:

1. immutable records, content-addressed blobs, ResultBundles, and EvidenceDecisions;
2. current validators and protocol constants in `autoresearch/`;
3. deterministic views rebuilt from those records;
4. current documentation;
5. historical notes, logs, comments, or remembered results.

Never resolve a conflict by choosing the more convenient value. Inspect the implementation,
report the disagreement, and repair documentation separately. One known example is the old
sentence in `program.md` that says controls live for one hour/eight uses. Current code in
`autoresearch/bank.py` is authoritative: 20 minutes, at most three uses, and at least 420
seconds of remaining lifetime at candidate staging and launch.

## 3. Cold-start protocol — do this first every session

From the repository root, initialize the ignored local store if it does not exist:

```bash
test -d .autoresearch || uv run autoresearch --root .autoresearch init
```

Then inspect it:

```bash
uv run autoresearch --root .autoresearch validate
uv run autoresearch --root .autoresearch status
uv run autoresearch --root .autoresearch queue --jobs
uv run autoresearch --root .autoresearch science
uv run python tools/preflight.py
```

Immediately arm the two recurring duties in Sections 3.1 and 3.2. Reading the instruction
once is not a timer. Use the Claude/runtime recurring-loop facility so these checks wake the
agent while it is working. If the environment has no recurring-loop facility, run work in
bounded chunks and perform the same duties manually before ten and sixty minutes elapse;
state this limitation explicitly rather than pretending the cadence is armed.

Then inspect the generated views under `.autoresearch/views/`, especially:

```text
SCIENCE.json
SCIENTIFIC_BELIEFS.json
MECHANISMS.json
HYPOTHESES.json
LESSONS.json
RESEARCH_GAPS.json
IDEA_QUEUE.json
RESEARCH_TASKS.json
BANK.json
PROMOTION_QUEUE.json
```

Cold-start rules:

- Run `init` only when `.autoresearch` is absent. Never initialize over an existing store.
- After initialization, do not delete, reset, rewrite, or “clean” `.autoresearch`.
- Do not trust counts or experiment status written in prose. Derive them live.
- Do not start `run --follow`, calibration, or GPU work merely to keep the machine busy.
- If jobs are already pending or running, inspect and preserve them. Do not kill or release
  anything unless liveness checks prove it is dead and the user authorizes recovery.
- If the active store has zero hypotheses, no model experiment may run. Select an axis,
  inspect its evidence, conduct real independent debate, and register a hypothesis first.
- An empty queue is not a defect during scientific ideation.

### 3.1 Mandatory ten-minute “are we working?” check

This cadence applies for the entire active Claude session, including literature reading and
ideation. It asks two separate questions:

1. **Is the research agent making substantive progress?** A paper section read, claim
   appraisal completed, mechanism edge audited, debate obtained, code/test change landed,
   result interpreted, or explicit blocker identified counts. Merely waiting, narrating,
   re-running unchanged status, or keeping a GPU occupied does not.
2. **Is the experiment apparatus doing what it claims?** Compare pool, queue, leases, real
   remote processes, GPU identity/utilization, last progress, and health circuit. Queue
   `running` without an observed process is not working. A foreign GPU process is not our
   work.

Arm the agent wake with the runtime's recurring-loop mechanism. In Claude environments that
support a loop command, schedule a ten-minute read-only check of the commands below. Do not
claim the cadence is armed unless the runtime confirms it.

```text
/loop 10m uv run autoresearch --root .autoresearch status
```

At every wake:

```bash
uv run autoresearch --root .autoresearch status
uv run autoresearch --root .autoresearch queue --jobs
```

Also inspect the actual remote training processes and GPU identities declared by the
operator. Record a concise finding in ignored local state: timestamp, current research
action, last substantive progress, pool state, queue state, real training-process count,
GPUs executing our processes, health state, and next action.

Interpret the check in context:

- During paper reading/debate with no accepted hypothesis, an empty queue and idle GPUs are
  expected. The next action is scientific work, not fabricated calibration or a candidate.
- With deliberately staged work, an idle pool or queued-but-not-launching state is an
  operational problem to diagnose.
- A dead pool may be restarted automatically only after scientific preflight is complete,
  work is already deliberately staged, remote configuration is verified, and the operator
  has authorized unattended execution.
- The monitor may never choose a hypothesis, edit `train.py`, stage a candidate, weaken
  health, or manufacture controls merely to make utilization look good.
- If the host is unreachable, retry and report uncertainty; do not conclude that no work is
  running.

On every ten-minute wake, also check the timestamp of the latest Fable critique. If it is
missing or at least sixty minutes old, Section 3.2 is due immediately.

### 3.2 Mandatory hourly Fable critique

Once per hour during an active research session, dispatch the actual independent **Fable**
agent to critique the project as a whole. Do this concurrently with safe ongoing work; do
not pause a running GPU queue to wait for prose. At session start, run Fable immediately if
there is no critique from the preceding hour.

Fable must read primary project state, not Claude's narration:

- immutable specs, manifests, results, EvidenceDecisions, claims, lessons, and source IDs;
- current derived science, bank, promotion, SOTA-blocker, queue, and health views;
- relevant full-text paper locations and paper-validity appraisals;
- the actual `train.py`/mechanism alignment and current tests; and
- prior Fable critiques, specifically whether their recommendations were tested or ignored.

Fable's standing questions are:

1. What is the strongest current claim, and which claim is most likely to be withdrawn?
2. Which paper, claim, mechanism edge, baseline label, or result has been rated too highly?
3. Where do specification, code, activation diagnostic, and interpretation disagree?
4. What common cause links recent failures, nulls, invalid runs, or blocked work?
5. What assumption remains unmeasured, ranked by the cheapest decisive check?
6. Is the research portfolio over-concentrated in one axis or repair-looping one family?
7. What should be refined, pivoted, blocked, or tested next, with a falsifier and activation
   diagnostic for each proposal?
8. Which part of the research system itself should change, and what evidence justifies it?

Save every critique without overwriting an earlier one:

```text
AI_papers/YYYY-MM-DDTHHMMZ_fable_system_critique.md
```

The artifact must include Fable's real agent/session identity, generation time, records and
paper locations inspected, factual findings, inferences, disagreements with current beliefs,
and concrete next actions. A critique that merely agrees, summarizes, or produces vague
ideas is incomplete.

Fable is a critic, not execution authority. It may propose claims, lessons, mechanisms,
hypotheses, or protocol changes, but the owning agent must verify sources and pass normal
registration/debate gates. Fable may not stage or run experiments, edit immutable state, or
silently rewrite a belief. If it identifies valid opposing evidence, preserve the
contradiction through the normal claim protocol rather than arguing it away.

Do not simulate Fable in the same Claude session or relabel a generic self-critique as
Fable. If the Fable agent is unavailable, write an explicit `FABLE_UNAVAILABLE` entry to
the hourly artifact/status, notify the user, and keep trying on subsequent hourly checks;
do not silently substitute another identity.

### 3.3 Mandatory full paper every thirty minutes

The hourly artifact is a focused adversarial system critique. It does not replace the full
research synthesis. Every thirty minutes during an active research session, Fable writes a
readable campaign paper to:

```text
papers/YYYY-MM-DDTHHMMZ_short-finding-title.md
```

Arm this cadence with the runtime's recurring-loop facility at session start. If that
facility is unavailable, track the last paper timestamp and complete the paper manually
before thirty minutes elapse. The first paper is due thirty minutes after the active session
begins; a prior-session paper does not satisfy the new session indefinitely.

Each paper covers everything new since the preceding full paper: papers read and appraised,
claims registered, mechanism edges changed, lessons reviewed, hypotheses debated, code or
protocol work, and every newly landed candidate/promotion ResultBundle. Include valid
positives, valid negatives, non-activation, invalid execution, runtime/resource failure,
and blocked work. Exclude unchanged calibration bookkeeping from the narrative unless it
changed the noise estimate or exposed an apparatus problem.

If no experiment landed during the interval, say **“no new experimental evidence”** near
the top. The paper must then synthesize only genuine literature, mechanism, failure,
implementation, or process progress. Never invent a result or recycle the previous paper
with a new timestamp. If nothing substantive changed at all, write a short stagnation paper
that diagnoses why, identifies the blocker, and states the cheapest next action.

At a thirty-minute boundary, already-running work may finish, but do not make or stage a new
research decision until the paper is written and its factual references have been checked.
This is an agent decision checkpoint, not permission to kill work or mutate the queue.

Use `templates/paper_template.md` for coverage, but write for a human. The paper must include:

1. a short conclusion and exact scope;
2. all newly covered result IDs, hypothesis IDs, activation outcomes, evidence verdicts,
   and frozen controls;
3. aggregate effects, uncertainty, seed counts, and promotion status without best-of-N
   selection;
4. what each experiment actually taught about its mechanism;
5. paper-validity or code-method issues discovered during the block;
6. positive, negative, inconclusive, invalid, and blocked findings—not only winners;
7. common failure patterns and the scientific lessons registered from them;
8. every belief and mechanism edge strengthened, weakened, contested, or unchanged;
9. the strongest claim most likely to be withdrawn and why; and
10. concrete next hypotheses across multiple axes, each with activation and falsification.

After any decisive held-out-seed promotion, write an immediate result paper instead of
waiting for the next thirty-minute boundary. It must use the independent result-analysis
council and state the narrowest defensible claim. The next scheduled paper should reference
that report rather than duplicating it.

The full paper must use the actual independent Fable agent. If Fable is unavailable at a
thirty-minute boundary, record `FABLE_UNAVAILABLE`, notify the user, allow already-authorized
execution to settle safely, and treat new candidate staging as blocked until the full
synthesis can be completed. Do not have Claude impersonate Fable.

## 4. Project map

```text
autoresearch/                 core record, science, sealing, execution, queue, and bank code
.autoresearch/                ignored per-operator registry, blobs, queue, and views
runs/science/                 ignored per-operator scientific declarations
runs/sources/                 ignored per-operator paper/full-text snapshots
runs/code/train.py            normal mutable experimental surface
runs/code/prepare.py          byte-identical fixed Karpathy data/evaluation implementation
runs/code/launch.sh           sealed remote launcher; use it for GPU work
runs/code/provenance.json     exact upstream and minimal seed/timing adapter provenance
runs/scope.json               ignored operator-created comparison-defining scope
runs/execution.json           ignored operator-created resource declaration
templates/                    declarations for scientific and execution artifacts
papers/ and AI_papers/        ignored generated reports; never execution authority
tools/read_fulltext.py        preserve and register arXiv full text
tools/coe_audit.py            chain-of-evidence integrity audit
tests/                        executable protocol expectations
```

Do not edit files under `.autoresearch/records` or `.autoresearch/blobs` directly. Register
through the CLI. Derived files under `.autoresearch/views` may be rebuilt and are never the
primary evidence.

## 5. Establish the scientific agenda

A clean clone has no active agenda or inherited scientific position. Construct a versioned
agenda from `templates/research_agenda.json` after inspecting the target benchmark and
current literature. The following six axes are a useful starting taxonomy, not registered
beliefs or mandatory choices:

| Axis | Question | Seed mechanism |
|---|---|---|
| `axis_measurement_integrity` | Is a gain activated, traceable, in scope, and above variation? | chain-of-evidence research loop |
| `axis_optimizer_geometry` | Can low-overhead gradient shaping improve early Muon progress? | preconditioning and update geometry |
| `axis_schedule_horizon` | Can schedules or averaging improve fixed/uncertain-horizon checkpoints? | horizon-robust update allocation |
| `axis_architecture_signal_path` | Can small scale/signal changes improve conditioning cheaply? | adaptive signal scaling |
| `axis_data_token_efficiency` | Can selection or order improve information per token? | data selection and exposure |
| `axis_systems_throughput` | Can safe savings become more useful optimization steps? | throughput converted into useful steps |

Do not register a seed mechanism until its claims have been extracted from inspected full
text and rated for validity and transfer. A copied paper intervention with no transfer
analysis is not an original research idea.

### 5.1 The six seed axes are not exhaustive

Treat an axis as a causal intervention family, not a paper keyword. Axes should be separated
when they act through different mediators, require different activation diagnostics, or have
different fixed-time cost models. By that criterion, the seed agenda compresses several
important directions too aggressively.

A more complete working taxonomy is:

| Causal axis | Examples in this project | Primary activation diagnostic |
|---|---|---|
| Measurement integrity | provenance, paired controls, variance, code-method alignment | evidence chain and resolvable control variation |
| Gradient geometry | Muon equilibration, spectral shaping, preconditioning | condition/stable-rank or update-spectrum change |
| Temporal update dynamics | LR, warmup, warmdown, momentum, averaging, weight decay | update-norm or effective-step allocation over time |
| **Stochastic optimization and exposure** | batch size, accumulation, clipping, gradient noise, steps-versus-tokens | noise/clipping rate and useful progress per token/step |
| Parameterization and signal propagation | normalization placement, residual scaling, initialization, learned scales | activation/gradient variance by depth and update-to-weight ratio |
| **Architecture and compute allocation** | attention locality, head grouping, value paths, depth/width/MLP allocation | FLOPs/step distribution, receptive path, useful steps |
| Data distribution and curriculum | example selection, order, mixture, rare-token exposure | measured distribution/exposure change without leakage |
| **Objective and supervision density** | multi-token prediction, auxiliary targets, context-level objectives | supervised targets or predictive information per token |
| Systems and numerical efficiency | kernels, precision, memory, input pipeline | saved time becomes valid extra steps without numerical drift |

The bold rows are commonly underrepresented when starting from the six-axis taxonomy. Do
not use `axis_architecture_signal_path` as a catch-all: normalization/residual
parameterization and attention/capacity allocation have different mediators and cost models.
Likewise, batch/noise dynamics are not merely a learning-rate schedule, and changing the
training objective is not merely changing the data.

Do not add axes merely to make the list longer. Before proposing a successor agenda, require
for every new axis:

1. a distinct causal mediator not already represented cleanly;
2. at least one scope-preserving intervention possible in `train.py`;
3. a numeric activation diagnostic that can be emitted in a 300-second run;
4. a plausible effect large enough to repay its wall-clock cost;
5. repeatable literature queries and at least two potentially independent sources; and
6. a boundary statement explaining what belongs in this axis and what does not.

Once an agenda is registered it is immutable. Do not edit its record or silently stamp a
new topic onto sources. A taxonomy expansion requires a versioned successor agenda with a
new ID, followed by source, claim, and mechanism coverage for the new topics.

Treat stability (loss spikes, finite numerics, gradient outliers) and operating-point
transfer as cross-cutting requirements on every axis rather than dumping them into one
miscellaneous direction. Treat tokenizer, evaluator, validation split, data manifest,
precision contract, and budget changes as new-scope research unless the frozen scope
explicitly permits them.

The repository intentionally ships no papers. Retrieve relevant primary sources into the
ignored per-operator corpus and register only material actually inspected in full text.

### 5.2 Continuous deep reading on the selected direction

Selecting an axis starts a focused literature lane; it does not end literature work. Keep
reading full text and analyzing the selected direction throughout ideation, calibration,
candidate execution, and result interpretation. Use CPU/agent time while GPU work is in
flight to deepen the evidence rather than waiting on the run.

Before every new hypothesis or refinement:

1. inspect the axis's current sources, claims, contradictions, validity ratings, mechanism
   edges, research gaps, and unread high-priority papers;
2. run focused searches for newer evidence, negative results, replications, and competing
   mechanisms when coverage is stale or a causal link remains weak;
3. deeply read the most decision-relevant available full texts, including method,
   comparator, tables, ablations, limitations, artifacts, and failure cases;
4. update paper-validity and claim-level appraisals, preserving supporting and opposing
   evidence separately;
5. re-evaluate transfer to the exact model, optimizer, data, hardware, metric, and
   300-second horizon; and
6. state whether the new reading strengthens, weakens, redirects, or blocks the proposed
   mechanism before code is edited.

Do not satisfy this duty by collecting abstracts or papers at random. Continue until the
selected causal question is saturated enough for a decision: repeated focused queries add
no materially new evidence, the important opposing explanations have coverage, and the
remaining uncertainty is best resolved experimentally. Record that saturation judgment;
new results or newly published evidence reopen the literature lane.

## 6. When the user gives you a paper

“Read this paper” means full-text scientific ingestion, not an abstract summary.

### 6.1 Identify applicability

1. Read the agenda and assign the paper to every genuinely related axis.
2. State why its model scale, data, optimizer, hardware, horizon, and metric do or do not
   transfer to the current scope.
3. Create a new agenda topic only if no existing axis can represent the question. Do not
   create floating topic IDs.

### 6.2 Preserve the full text

For arXiv papers, use the registered agenda and a real axis:

```bash
uv run python tools/read_fulltext.py \
  --ids ARXIV_ID \
  --topic axis_optimizer_geometry \
  --agenda agenda_fresh_start_2026_08_17
```

For another source, start from `templates/literature_source.json`, save the full text when
legally and technically available, compute its digest, and register it:

```bash
uv run autoresearch --root .autoresearch literature-source path/to/source.json
```

Discovery metadata or an abstract may be used for triage only. Do not claim to have read a
paper whose registered content is `metadata_only` or `abstract_only`.

### 6.3 Read the paper as evidence

Inspect, at minimum:

- problem definition and comparison baseline;
- exact method and implementation details;
- datasets, model scale, optimizer, budget, and evaluation;
- reported effect sizes, uncertainty, and seeds;
- ablations and negative results;
- tables, figures, and captions;
- limitations, failure cases, and artifact availability.

Do not infer a result from the title, abstract, citation count, or venue. Do not quote a
metric without its comparator and scope.

### 6.4 Assess validity and rate it before using the paper

A citation is not evidence merely because the paper exists. Before allowing any extracted
claim to support a mechanism edge, perform both a paper-level appraisal and a claim-level
appraisal. Keep **internal validity** separate from **transfer/applicability**: a rigorous
7B, 100B-token study may be internally strong and nearly irrelevant to a 124M, 300-second
run, while a close speedrun benchmark may be directly relevant but statistically weak.

First check for fatal validity blockers:

- the claimed result is not present in the preserved full text;
- the comparator differs in compute, data, tuning effort, or evaluation;
- the implementation violates the stated method or benchmark specification;
- the metric or selected checkpoint does not answer the stated question;
- results are selected without disclosure, or uncertainty cannot be reconstructed;
- leakage, duplicated evaluation data, or another integrity failure is plausible and
  unresolved; or
- the causal claim is supported only by correlation with no discriminator or ablation.

A fatal blocker makes the affected claim ineligible to support a causal edge. The paper can
still supply an opposing claim, a boundary condition, a failure lesson, or a hypothesis to
test independently.

Then score these dimensions from 0 to 4, with a written reason and exact paper locations:

| Dimension | 0 | 2 | 4 |
|---|---|---|---|
| Design/comparator fairness | confounded or undefined | partially controlled | matched controlled comparison |
| Measurement/statistical support | no usable measurement | effect shown but uncertainty/replication limited | effect, seeds, uncertainty, and variance are adequate |
| Mechanism identification | assertion only | mediator or partial ablation | mediator, alternatives, and falsifiers discriminated |
| Transparency/reproducibility | unverifiable | method described, incomplete artifact | code/config/data lineage sufficient to reproduce |
| Robustness/reporting completeness | selective or contradictory | limited settings/negative results | relevant ablations, failures, and sensitivity reported |

Compute the paper's **internal-validity rating** as the mean of those five scores only when
there is no fatal blocker:

```text
A: 3.5–4.0   strong within the studied setting
B: 2.75–3.49 credible with material limitations
C: 2.0–2.74  suggestive; hypothesis-generating, not decisive
D: <2.0 or a fatal blocker for the claim being used
```

Separately rate **applicability to OPHIS** from 0 to 4 using model/architecture match,
optimizer match, data/tokenizer match, training horizon, hardware/kernel behavior,
metric/objective match, and fixed-time overhead. Never average applicability into internal
validity: publish both. Also estimate whether the reported effect could clear the OPHIS
gate after accounting for added step time. If the paper reports time-to-target rather than
loss-at-fixed-time, explicitly reconstruct or decline that translation.

The paper grade is a triage summary, not inherited truth. Different claims in one paper may
deserve different ratings. The authoritative machine-readable rating remains each claim's
`evidence` and `assessment` fields:

- `study_design` records design strength;
- `artifact_status` and `reproduction_status` record auditability;
- `directness` records whether evidence actually measures the claim;
- `scope_match` records applicability, not correctness;
- `risk_of_bias` records internal threats;
- `metrics` records comparator, effect, uncertainty, and seeds;
- `assessment.rationale` must include the paper validity grade, applicability rating, fatal
  blocker audit, and reasons for every selected field.

Run `science` after registration and inspect the component scores rather than quoting only
the aggregate confidence. That aggregate is a heuristic, not a calibrated probability;
venue contributes little and can never rescue weak methods. Mechanism confidence is the
weakest necessary edge, so a low-validity claim should correctly constrain the whole chain.

### 6.5 Extract atomic claims

Start from `templates/scientific_claim.json`. Each claim must be one attributable assertion
and include:

- a stable `belief_key`;
- supporting or opposing stance;
- registered topics and scope;
- exact source ID and section/table/figure locator;
- comparator, effect, uncertainty, and seeds when reported;
- evidence assessment and transfer limits;
- language no stronger than the cited source.

Register each claim:

```bash
uv run autoresearch --root .autoresearch scientific-claim path/to/claim.json
```

Do not combine several conclusions into one broad claim. Do not register one work repeatedly
to manufacture independent evidence. Opposition remains visible; never rewrite it into
support.

### 6.6 Update causal mechanisms immutably

Start from `templates/scientific_mechanism.json`. A mechanism is a directed causal graph,
not a prose summary. Every edge must cite supporting claim IDs. Include assumptions,
alternatives, predictions, falsifiers, and diagnostics that distinguish the causal story
from an easier explanation.

Existing registered records are immutable. “Update a mechanism” means create and register
a new version with a new ID and preserved provenance; it never means editing the registry
in place. Opposing claims belong in the belief ledger and alternative explanations rather
than being cited as support for an edge.

```bash
uv run autoresearch --root .autoresearch mechanism path/to/mechanism.json
uv run autoresearch --root .autoresearch science
uv run autoresearch --root .autoresearch validate
```

After ingestion, report which claims and edges changed, what remains uncertain, and whether
the paper licenses a new experiment. Reading a paper does not obligate an experiment.

## 7. From mechanisms to an original hypothesis

Before editing `train.py`, inspect the chosen records directly:

```bash
uv run autoresearch --root .autoresearch inspect scientific_mechanism MECHANISM_ID
uv run autoresearch --root .autoresearch inspect scientific_claim CLAIM_ID
uv run autoresearch --root .autoresearch science
```

Also inspect every active lesson that applies to the selected topics, mechanisms, or
subsystems. A future hypothesis must explicitly incorporate, distinguish, or identify an
explicitly superseded version of each relevant lesson.

A production model hypothesis normally needs:

- complete agenda search/source/claim/full-text coverage;
- at least two independent foundation evidence units;
- foundation confidence at least `0.60`;
- valid claim-backed mechanism edges;
- exact reviews of all selected mechanisms and claims;
- reviews of every applicable active lesson;
- at least one alternative intervention;
- one minimal intervention and its diagnostics;
- an activation predicate and machine-readable rule;
- explicit falsifiers and competing explanations; and
- accepted independent debate synthesis.

`--allow-weak-science` bypasses only the confidence/readiness threshold. It does not bypass
mechanism, claim, lesson, debate, or activation requirements. Never use it by default or to
save time. Use it only when the user explicitly authorizes a high-risk exploratory test,
and explain why the weak foundation is worth testing.

### 7.1 Mandatory hypothesis debate

Use `templates/scientific_hypothesis.json`. Obtain real independent reviews:

- **Innovator:** propose the highest-upside causal intervention, explain novelty, and name
  the diagnostic that proves activation.
- **Pragmatist:** audit feasibility inside 300 training seconds, implementation risk,
  throughput cost, scope match, and transfer risk.
- **Contrarian:** try to disprove the mechanism before launch; identify confounds,
  contrary evidence, an alternative explanation, and a discriminator that can kill it.
- **Independent synthesizer:** integrate the reviews, accept or reject objections, choose
  `proceed`, `revise`, or `reject`, and state the concrete resulting design change.

Distinct role labels in one context are not independence. Use actual separate agents and
sessions when that capability exists. Do not fabricate agent IDs or session IDs. If real
independent sessions are unavailable, prepare an unregistered draft and ask the operator
for the missing reviews. The synthesizer must be independent from all three reviewers.

The `mechanism_reviews` must cover exactly `mechanism_ids`; `claim_reviews` must cover
exactly `claim_ids`. A `reject` synthesis cannot stage. A `revise` decision must already be
reflected in the hypothesis being registered.

The activation diagnostic must be an identifier emitted numerically by the experiment.
Its rule must establish that the intervention actually engaged. Failure to activate always
means `inconclusive`, never refutation.

```bash
uv run autoresearch --root .autoresearch hypothesis path/to/hypothesis.json
uv run autoresearch --root .autoresearch science
uv run autoresearch --root .autoresearch validate
```

Do not launch a model experiment until the hypothesis appears research-ready, or the user
has explicitly authorized a recorded weak-science exception.

## 8. What “activation observed” and “bank gate cleared” mean

These are different gates answering different questions.

### Activation observed: did the intervention engage?

Every hypothesis preregisters a diagnostic and rule, for example:

```json
{
  "predicate": "Row/column gradient imbalance decreases before Muon orthogonalization.",
  "diagnostic": "gradient_imbalance_reduction",
  "rule": {"op": "gt", "value": 0.10},
  "failure_status": "inconclusive"
}
```

The candidate must emit `gradient_imbalance_reduction` as a numeric metric. A value above
`0.10` means activation was observed: the intervention measurably changed the mediator it
was designed to change. If the metric is absent or the rule fails, the run cannot test the
causal mechanism. Even a worse `val_bpb` is then inconclusive, not evidence that the idea is
scientifically false.

Activation is not success. A mechanism can activate and still fail to improve quality.

### Bank gate cleared: was the pilot effect larger than the allocation threshold?

After a valid activated seed-42 candidate, the bank compares its `val_bpb` only with the
exact frozen same-GPU control. For this minimize metric, the candidate must improve by more
than:

```text
max(0.000426, 1.96 * recent exact-control sample standard deviation)
```

If it clears that threshold, the candidate becomes `promotion_due`. This authorizes spending
held-out seeds on confirmation. It is not yet a supported improvement or SOTA.

The four possible pilot interpretations are:

| Activation | Bank gate | Meaning | Next action |
|---|---|---|---|
| No | Any observed score | Mechanism was not tested | Diagnose activation; refine or pivot |
| Yes | No | Mechanism engaged but pilot effect was insufficient | Record negative/preliminary evidence; refine, pivot, or stop |
| Yes | Yes | Promising activated pilot | Run held-out-seed promotion |
| Unknown/invalid | Not meaningful | Measurement cannot support inference | Diagnose integrity/runtime; do not retry blindly |

## 9. What the benchmark actually is

The intended scope is defined by `runs/scope.json`. It starts from Karpathy commit
`228791fb499afffb54b46200aca536f79142f117`. `prepare.py`, `pyproject.toml`, and `uv.lock`
are byte-identical to upstream. Each new operator must create a clean upstream cache with
exactly train shards 00000–00009 and validation shard 06542, then bind every shard, the
freshly generated upstream tokenizer, and Karpathy's original
`decode([token_id]).encode("utf-8")` BPB byte table.

`train.py` differs only at the protocol boundary: it reads the preregistered seed, starts
the reconciliation clock before importing torch, and prints the executed seed. It does not
change the model, optimizer, schedules, loader calls, evaluator, or 300 charged training
seconds. This is the strict protocolized Karpathy baseline for this project. Results are
still platform-specific and no historical score is imported as experiment 1.

Never compare a result to another scope,
hardware class, corpus, tokenizer, evaluator, precision, or budget. If any comparison field
changes, define a new scope ID and comparison group.

`runs/code/train.py` is the normal mutable experimental surface. Treat `prepare.py`,
`pyproject.toml`, `uv.lock`, `provenance.json`, the data manifest, launcher, tokenizer,
evaluator, and scope as frozen. A proposed change to them is a new benchmark/protocol
decision, not an ordinary candidate, and requires explicit user direction.

After `FRESH_START.md` setup and before the first calibration, run
`uv run python tools/preflight.py`. It must prove that
`prepare.py` is exact upstream, the remote cache contains no extra shards, every cache byte
matches `runs/data_manifest.json`, and all remote sealed bindings match. Do not “fix” a
preflight failure by editing `prepare.py`.

## 10. Execution workflow

Scientific reasoning happens before model execution. Once at least one accepted hypothesis
exists, freeze operational context and begin with one GPU. Execution files contain
`${OPHIS_*}` placeholders expanded from the environment; never print their secret values.

Begin with exactly one observed-free slot in `runs/execution.json`. Do not widen the active
configuration until one-wide timing is healthy and wider execution does not starve
training. Re-observe physical identity and ownership immediately before launch.

Use the sealed launcher:

```text
--argv /bin/bash launch.sh
```

Do not substitute a PATH-resolved interpreter or an ad hoc shell command. The v2 launcher
grammar and environment scrubbing are security and reproducibility boundaries.

### 10.1 Calibrate the exact control bank

Calibrate unedited `runs/code/train.py` at seed 42:

```bash
uv run autoresearch --root .autoresearch calibrate BANK_ID CALIBRATION_LABEL \
  --scope runs/scope.json \
  --execution runs/execution.json \
  --mutable-code-path train.py \
  --argv /bin/bash launch.sh

uv run autoresearch --root .autoresearch run --workers 1
uv run autoresearch --root .autoresearch doctor --bank-id BANK_ID
```

Do not edit `train.py` until its calibration manifest has been sealed. A valid control is
bound to bank revision, baseline fingerprint, stable context, scope, seed, resource ID, and
observed physical GPU UUID.

`doctor` must report `model_search_ready: true` for model work. Healthy means charged
training owns at least 75% of total time and structured timing is complete. If the profile
is overhead-dominated, investigate data/compile/evaluation/input/instrumentation/tokenizer
work. Do not hide the finding with `--allow-overhead-dominated` unless the user explicitly
accepts that exception.

Current control policy:

- TTL: 20 minutes from completion;
- maximum landed plus actively reserved candidate uses: 3;
- required candidate-run headroom: 420 seconds;
- candidate is pinned to the exact selected control and physical GPU;
- reference freshness and identity are checked at staging and again before process launch.

If no exact control qualifies, recalibrate. Never substitute another GPU or older control.

### 10.2 Stage one minimal candidate

Edit only the declared mutable path and ensure the activation metric is emitted. Then:

```bash
uv run autoresearch --root .autoresearch search CANDIDATE_LABEL \
  --bank-id BANK_ID \
  --summary "one exact intervention and expected causal effect" \
  --scope runs/scope.json \
  --execution runs/execution.json \
  --mutable-code-path train.py \
  --hypothesis-id HYPOTHESIS_ID \
  --direction optimization \
  --subsystem SUBSYSTEM \
  --track mechanism \
  --family MECHANISM_FAMILY \
  --argv /bin/bash launch.sh

uv run autoresearch --root .autoresearch run --workers 1
```

`--argv` must remain last. `search` is the v2 path; `screen` is legacy and must not be used
for new research. Identical argv is normal when mutable `train.py` bytes changed. Identical
argv and identical mutable bytes is a rejected no-op.

Classify every candidate honestly:

- `mechanism`: new causal structure not previously run;
- `knob`: another value of an existing constant;
- `throughput`: buys useful steps rather than per-step quality.

The exploration budget requires breadth: at least five of an eight-slot window must be
mechanisms, a knob family must first clear the bank gate, and no uncleared family gets a
third consecutive repair attempt. `--exploration-override` requires an immutable reason and
explicit operator judgment.

### 10.3 Read the bank decision

```bash
uv run autoresearch --root .autoresearch bank
uv run autoresearch --root .autoresearch status
```

A seed-42 candidate, including one marked `promotion_due`, is exploratory. It is a
work-allocation decision, not an improvement claim and never SOTA.

### 10.4 Promote on held-out seeds

Only a gate-clearing candidate can be promoted:

```bash
uv run autoresearch --root .autoresearch promote EXP_CANDIDATE_SPEC_ID
uv run autoresearch --root .autoresearch run --workers 1
uv run autoresearch --root .autoresearch synthesize
uv run autoresearch --root .autoresearch validate
```

The default confirmation preregisters seeds 43–47, requires four valid replicates, runs
paired control and candidate arms for every seed, and alternates arm order. Promotion seeds
must be distinct and held out from calibration/search. The protocol floor is three valid
distinct seeds; do not weaken a registered rule after seeing results.

SOTA is the paired confirmation aggregate inside one exact scope. Never report the best
individual seed, pilot delta, or a cross-scope number as SOTA. `synthesize` reconstructs
lineage from immutable manifests and publishes blockers rather than trusting labels.

## 11. Result interpretation and the second council

The authority chain is:

```text
ExperimentSpec -> ExecutionManifest -> ResultBundle -> EvidenceDecision
```

Queue state is operational bookkeeping. A `complete` job is not automatically valid
evidence. Read EvidenceDecision reasons, verified seed, timing, GPU identity, required
metrics, and scope before interpreting a value.

Interpret activation before direction:

- invalid/integrity failure: no scientific conclusion;
- activation diagnostic absent or rule false: inconclusive about the mechanism;
- activated and valid negative: directional opposing evidence;
- activated and valid positive: potential support, subject to promotion protocol;
- runtime/resource/scope failure: operational lesson, not a model claim.

After decisive promotion, create `templates/analysis_council.json` using real independent
sessions:

- **Optimist:** strongest interpretation supported by immutable evidence;
- **Skeptic:** effect size, variance, confounds, and alternative explanations;
- **Methodologist:** activation, protocol, scope, code-method alignment, reproducibility;
- **Independent synthesizer:** `support`, `oppose`, or `inconclusive`, with the narrowest
  defensible claim and boundary conditions.

Every reviewer must inspect exactly every EvidenceDecision ID used by the claim. Do not
fabricate independent identities. An inconclusive council cannot create a directional
experiment claim.

Use proposal-only before debate when useful:

```bash
uv run autoresearch --root .autoresearch conclude HYPOTHESIS_ID PROMOTION_SPEC_ID \
  --proposal-only
```

After the council:

```bash
uv run autoresearch --root .autoresearch conclude HYPOTHESIS_ID PROMOTION_SPEC_ID \
  --analysis-council path/to/analysis_council.json
uv run autoresearch --root .autoresearch science
uv run autoresearch --root .autoresearch validate
```

The experiment claim joins the literature ledger; it never erases motivating claims.

## 12. Failures are evidence about process and sometimes science

Classify a failure before choosing the next action. Allowed classes are:

```text
runtime
integrity
non_activation
valid_negative
degenerate
resource_mismatch
scope_mismatch
analysis_overclaim
```

Only an activated, valid negative result updates a directional scientific belief. Other
failures may still teach a reusable lesson. Use `templates/scientific_lesson.json` and bind
it to immutable source records:

```bash
uv run autoresearch --root .autoresearch lesson path/to/lesson.json
uv run autoresearch --root .autoresearch science
```

The action is one of:

- `proceed`: evidence is usable under stated conditions;
- `refine`: retain the direction but create a new design/hypothesis/spec;
- `pivot`: change the mechanism or research direction;
- `block`: do not repeat while the applicability condition holds.

A lesson needs observation, diagnosis with residual uncertainty, severity, relevant topics,
mechanisms/subsystems, a testable `applies_when` condition, and a concrete mitigation. It
does not expire merely because time passed. A newer immutable lesson may explicitly
supersede an older one.

Do not automatically retry an executed invalid, unknown, or uncertain replicate. An
implementation-equivalent operational repair may resume only when immutable scientific
intent is unchanged and queue protocol permits it. Any change to mechanism, intervention,
metric, control, scope, activation, or success rule requires a new artifact.

## 13. Queue, resource, and integrity policy

Useful commands:

```bash
uv run autoresearch --root .autoresearch queue --jobs
uv run autoresearch --root .autoresearch queue --reconcile
uv run autoresearch --root .autoresearch claims
uv run autoresearch --root .autoresearch doctor --bank-id BANK_ID
uv run python tools/coe_audit.py --skip-remote
```

Queue states are `pending`, `running`, `waiting`, `complete`, and `blocked`.

- `waiting` after `NoResourceAvailable` means no arm launched and bounded retry is safe.
- An executed invalid/unknown result is landed and not retried automatically.
- Unexpected execution/evidence uncertainty blocks the job.
- Reconcile only requeues when it proves execution never crossed a durable boundary.
- Release a claim only after independently proving local and remote processes are dead;
  use the CLI's explicit confirmation path.
- The health circuit pauses new claims after repeated invalid/unknown pilots. Diagnose the
  evidence reasons before considering `--ignore-health`; never use it silently.

OPHIS leases coordinate only workers sharing this state root. They do not reserve a GPU
against another user or campaign. Re-observe GPU processes and identity before launch. GPU
co-tenancy invalidates evidence and must never be disabled.

The local process measures runner time around SSH while the remote trainer emits its own
time. Do not allow the controlling machine to sleep during timed remote runs; a clock
disagreement is an integrity failure, not a reason to widen tolerances.

Start resident workers or switch the watchdog from report-only to repair-capable mode only
after there is deliberately staged work, scientific preflight is complete, remote
configuration is verified, and the operator wants unattended execution. The mandatory
report-only ten-minute observer remains active during ideation. A monitor may restart an
authorized worker pool; it may not invent candidates, bypass science, or manufacture
baseline work to hide an empty queue.

## 14. Implementation map

```text
records.py       schemas and cross-field scientific/execution validation
store.py         immutable registration, blobs, reference validation, locks
science.py       confidence, mechanisms, hypotheses, lessons, research tasks
workflow.py      calibration, candidate staging, control selection, promotion
bank.py          control TTL/use policy, frozen-reference checks, noise-aware gate
sealing.py       snapshot code/data and create immutable manifests
execution.py     resource acquisition, launch, telemetry, ResultBundles
evidence.py      validity decisions from executed records
campaign.py      durable queue, health circuit, reconciliation
knowledge.py     promotion reconstruction and scope-bounded SOTA
exploration.py   mechanism breadth and anti-repair-loop policy
doctor.py        profile-first model-search readiness
cli.py           supported user command surface
```

When changing protocol behavior, update validators, projections, templates, documentation,
and tests together. Do not merely add prose. A rule intended to prevent a recurring failure
belongs in executable validation or staging logic.

## 15. Verification after changes

For code or schema work:

```bash
uv run ruff check autoresearch tools tests
uv run pytest -q
uv run autoresearch --root .autoresearch validate
uv run python tools/coe_audit.py --skip-remote
git diff --check
```

For scientific registration, rebuild `science`, validate, inspect the new record, and
inspect its derived belief/mechanism/hypothesis state. A successful CLI exit is not enough
if the derived state does not say what you intended.

Preserve unrelated user changes in a dirty worktree. Use `apply_patch` for edits. Never
reset, delete, or overwrite experiment state unless the user explicitly requests the exact
destructive operation and the target has been resolved safely.

## 16. Reporting contract

Every update must distinguish:

- **fact:** directly present in immutable evidence or a cited paper location;
- **inference:** a reasoned interpretation with uncertainty;
- **proposal:** an untested mechanism or intervention;
- **allocation decision:** a pilot marked promotion-due;
- **supported/refuted claim:** decisive valid confirmation reviewed by the result council;
- **operational status:** queue/process state, never scientific evidence.

For each experiment report the scope ID, hypothesis ID, activation outcome, frozen control,
candidate and EvidenceDecision IDs, valid seed count, aggregate effect and uncertainty,
gate, and limitations. Say “inconclusive” when that is the honest state.

## 17. Non-negotiable prohibitions

Never:

- run a model candidate before inspecting mechanisms, claims, and active lessons;
- invent independent reviewer identities or simulate all debate roles in one session;
- use abstract-only discovery as if it were full-text evidence;
- create a claim without provenance and an exact locator;
- mutate or overwrite an immutable record;
- copy a paper method without explaining novelty, adaptation, and transfer risk;
- treat non-activation as evidence against a mechanism;
- call a pilot, promotion queue row, or best single seed an improvement or SOTA;
- compare results across scopes;
- weaken a gate, timing tolerance, success rule, or seed count after observing data;
- automatically retry an executed invalid/unknown/uncertain replicate;
- disable GPU identity/co-tenancy checks;
- use `screen` for new v2 research;
- start workers merely to make an empty queue look active;
- claim the system or GPUs are running without observing them; or
- let a runtime failure silently change a scientific belief.

When uncertain, stop before execution, inspect the immutable chain, and state exactly which
authority or user decision is missing.
