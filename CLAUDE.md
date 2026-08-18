# OPHIS v4 — cold start

## The goal, and nothing else

**Lower `val_bpb` on a 300-charged-second single-H200 run.** Steps, MFU, tokens and
throughput are diagnostics. They are never the objective and never a result.

`prepare.py` is FROZEN, byte-identical to Karpathy `228791f`. Its tokenizer, data loader,
packing, evaluator and BPB byte accounting are the benchmark contract. `train.py` is the
only file you edit. This is Karpathy's rule and it is not negotiable here.

## You start with no experimental priors

This campaign begins with **zero** inherited results. Everything a previous campaign
believed has been deleted, deliberately, because most of it was wrong or unfalsifiable.
Your only starting knowledge is what you read in papers. Do not go looking for archived
runs, and if you find some, do not reason from them.

What you may rely on before measuring anything are properties of the apparatus, not
findings about the model:

- `prepare.py` is frozen and byte-identical to Karpathy `228791f`. Tokenizer, loader,
  packing, evaluator and BPB accounting are the benchmark contract.
- The GPU box is shared. Other tenants hold most GPUs most of the time, and host CPU load
  runs near saturation. Co-tenancy on a GPU invalidates a run.
- **Read the timing fields carefully.** `loader_frac` and `fwdbwd_frac` are wall-clock
  fractions measured on the CPU around calls that launch asynchronous CUDA work.
  `fwdbwd_frac` is therefore *kernel launch* time, **not** GPU busy time, and a small
  value does NOT establish that the GPU is idle. Any claim about where the 300 seconds
  goes must come from a controlled comparison — change the FLOPs and see what happens to
  step count — never from reading these two fields alone. A previous campaign got this
  wrong and built a whole strategy on it.

## Running on 4 GPUs, and waves

`host/dispatch.py` caps at **`MAX_GPUS = 4`** — an operator allocation on a shared box,
never exceeded however many GPUs happen to be free. It claims only genuinely free GPUs
(no foreign compute app, memory under `MINFREE_MB`) and refuses to co-tenant.

Queue entries sharing a **`wave_group`** launch **concurrently or not at all**. That is
how a yoked pair is actually obtained. Launching one GPU at a time as capacity trickles
in produces runs minutes or hours apart, and on a contended host their spread measures
drift rather than effect — which is exactly what happened to the first control block.
The dispatcher prefers the largest wave that fits, so the box fills rather than dribbling
out singletons, and it claims every member of a wave or none.

When capacity is short it **holds** rather than launching a partial wave, logging
`WAITING: waiting for N free GPUs...` every 30 minutes. Watch it from here:

```bash
python3 tools/gpu_watch.py                    # one owner-attributed snapshot
python3 tools/gpu_watch.py --loop             # every 30 min (default interval)
python3 tools/gpu_watch.py --loop --interval=900
```

Most GPUs on this box usually carry foreign tenants, so expect to wait for a 4-wide
wave. That waiting is correct behaviour, not a stall.

## Measurement design

The default is a **yoked pair**: the treatment and an exact control launched in the same
wave, on different GPUs. Cross-wave comparison is not reliable here — byte-identical code
re-run in a different wave moves far more than most effects you will chase, because host
contention changes step count. Within a wave that nuisance is shared.

`tools/direction.py` therefore **measures** the noise band from concurrent controls
instead of assuming one, and reports how it was measured. Until two controls exist there
is no instrument, and in that state no axis may be declared dry — refusing to close an
axis you cannot measure is the fail-safe direction.

- Raw `val_bpb` is the verdict. A treatment that costs throughput is genuinely worse at a
  fixed 300s budget; that cost is the finding, not a nuisance term.
- Reporting a step-controlled residual *alongside* raw is fine and often explains the
  result. Letting the residual override the raw verdict is not.
- Never apply a step law across a change that moves tokens-per-step. Fit and application
  must share an operating point.
- Do not report an effect smaller than the measured band from a single unpaired run.

## Chain of evidence

Every claim this campaign makes must walk back to a source. `tools/coe.py` audits the
chain with five checks — four adapted from ScientistOne's integrity checks, pushed down
from the paper layer to the evidential layer where claims here actually break, plus
AutoResearchClaw's numeric registry:

| check | it catches |
|---|---|
| **E1 SOURCE** | a claim whose full text is not on disk, or whose digest no longer matches. An abstract is triage and may never back a claim. |
| **E2 LINK** | a mechanism citing no claims, or a hypothesis citing a mechanism nobody registered |
| **E3 ACTIVATION** | a run that never emitted its declared diagnostic. Non-activation is **inconclusive**, never evidence against the mechanism — otherwise good ideas get retired for free. |
| **E4 METHOD-CODE** | a variant byte-identical to the control, or one whose cfg disagrees with the hypothesis it claims to test |
| **E5 NUMERIC** | a number in a paper, round or critique that no result record produced |

```bash
python3 tools/coe.py            # audit
python3 tools/coe.py registry   # every value a document is allowed to cite
```

The **registry** is AutoResearchClaw §3.4: it is built only from `runs/sweep/results`,
you may read it, and nothing may add a value to it by asserting one in prose. Prose is
the part of a campaign most likely to drift from its evidence, and it drifts in the
flattering direction. `queue_from_round.py` applies the same discipline before any GPU
time is spent: a generated variant identical to the control is refused at the door.

Record shapes live in `templates/`, each field commented with the reason it exists.
`templates/README.md` draws the whole chain.

## Lessons: failures bind the next run

Every failed or invalid run must become a typed lesson, or the campaign pays for it
twice. `tools/gate.py` reports unlearned failures and `tools/analyze.py` lists them by
name, so a failure cannot be absorbed silently.

```bash
python3 tools/claims.py lessons            # active lessons, most severe first
python3 tools/claims.py lessons capacity   # just this direction's
python3 tools/claims.py lesson file.json   # register one (templates/lesson.json)
```

A lesson is typed — `runtime | integrity | non_activation | valid_negative | degenerate |
resource | scope | overclaim` — because the consequences differ. A non-activation says
nothing about the science and **may not block a direction**; registration refuses that
combination. The action is `proceed | refine | pivot | block`.

Two fields make a lesson bind rather than sit in a file:

- **`blocks_keys`** — config keys the lesson forbids. `queue_from_round.py` refuses any
  entry setting one while the lesson is active. A `block` lesson with no enforceable
  hook is rejected at registration.
- **`superseded_by`** — the id of a newer lesson that replaces this one; *this* lesson
  then retires. Lessons never expire on a timer. A scientific negative stays true until
  contradicted; an operational lesson holds while its `applies_when` holds. Retirement
  is by evidence, not by age.

The council reads them before proposing and writes them after: the critic must say which
lessons it checked each proposal against, and Fable's process agent audits whether every
failure produced one.

## Literature comes first

The previous campaign ran an entire experiment budget and made **zero** literature searches; its four
mechanisms were drawn from memory and produced no measured win. So the corpus is on the
critical path, not beside it: `tools/agenda.py` will not open a direction that has no
read papers behind it, and the gate reports that refusal.

```bash
python3 tools/lit.py screen --target 240   # ~47 queries, 2025-2026, relevance-scored
python3 tools/lit.py fetch  --limit 60     # LaTeXML full text -> lit/sources/, hashed
python3 tools/lit.py status                # corpus coverage per direction family
python3 tools/lit.py read capacity         # the reading list for one direction
python3 tools/claims.py template           # the claim / mechanism schema
python3 tools/claims.py add file.json      # register what you actually read
python3 tools/claims.py mech file.json     # a mediator + the claims backing it
python3 tools/claims.py hyp file.json      # a falsifiable prediction + activation rule
python3 tools/claims.py lesson file.json   # a typed failure: proceed/refine/pivot/block
python3 tools/claims.py status             # evidence per direction
```

Screening reads abstracts — triage, never evidence. A claim must resolve to a full-text
snapshot on disk plus a locator; `claims.py` rejects one whose source was never fetched.
`internal_validity` and `transfer` are scored 0-4 **separately and never averaged**: a
rigorous 7B/100B-token result can be internally excellent and nearly irrelevant at 124M
and 600 steps. Claims with `stance: "opposes"` stay visible and are never rewritten into
support.

## Loop

```bash
python3 tools/health.py         # every 20 min. Is Claude working, are OUR GPUs running?
python3 tools/agenda.py         # THE active direction, and whether to rotate off it
python3 tools/analyze.py        # what the evidence says, honestly
python3 tools/direction.py      # axis + mechanism coverage, and the DIRECTION SPACE table
python3 tools/coe.py            # does every claim still trace to a source?
python3 tools/gate.py           # what is allowed to launch right now
```

Then, per block:

1. **Council round** (`python3 tools/council.py round` prints the four prompts). Dispatch
   four subagents in parallel — explorer, pragmatist, critic, and an independent
   synthesizer — and write `rounds/<UTC>_round.md`. The synthesis MUST end with a
   ```queue fenced JSON list; a round that proposes nothing runnable does not count.
2. `python3 tools/queue_from_round.py` — builds every variant immediately (so a broken
   edit fails now, not at 3am on a GPU) and appends to the dispatcher queue.
3. **Hourly Fable critique** (`python3 tools/council.py critique`). Four independent Fable
   agents reading primary state, never your narration. Writes `critiques/<UTC>_fable_critique.md`.
4. Read results with `tools/analyze.py`. Register nothing you did not recompute.

`python3 tools/council.py status` tells you what is overdue. Both artifacts are validated
on **content** — role sections, word floors, distinct agent ids, an independent
synthesizer, a parseable queue block. `touch` does not satisfy them.

## Explore vs exploit

`tools/direction.py` enforces it, and it replaces a policy that actively caused failure.

- **An axis with zero runs can never be closed and outranks everything.** An earlier policy
  grouped several parameters into one capped family; the cap was spent on two of them and
  the third was closed having never been varied once. Axes are per-parameter here so that
  cannot recur, and `is_platform()` fails safe on **any** config key the policy does not
  recognise — an unrecognised knob must never be mistaken for a control, or it silently
  pollutes the very control block that measures the instrument.
- **An axis stays open while it pays.** It closes only after `DRY_STREAK=4` consecutive
  runs that fail to beat the running best by more than the noise band. A cap is a ceiling
  and forbids exploiting a winner; a dry rule is a cost and does not. The dry test uses
  the MEASURED band — a threshold set too wide is a cap wearing a disguise, because
  nothing ever registers as an improvement.
- **Mechanisms tested in isolation are always allowed.** A mechanism crossed with a closed
  knob axis is refused — evaluate new structure alone before entangling it with a tuned
  architecture you can no longer attribute.
- **`EXPLORE_FLOOR = 0.35`.** If `explore_debt` is positive, the next launch goes to an
  unexplored axis or a mechanism. Exploitation collapses into a monoculture without it.
- **One direction is active at a time, and it is abandoned on schedule.**
  `tools/agenda.py` holds the active direction and forces rotation on any of three
  independent triggers: **DRY** (4 runs with no gain above the noise band and no untouched
  axis left in the family), **STALE** (6 runs since the last real gain, even if an axis is
  still untouched — untouched is not the same as promising), and **HARD CAP** (12 runs in
  one family, unconditionally). The hard cap is the anti-monoculture
  backstop against spending an entire campaign inside one or two families. An exhausted direction goes on
  cooldown until every other eligible direction has had a turn, **or until new literature
  arrives for it** — fetching more papers genuinely reopens a cooled direction. That is
  the intended way to revive one: bring evidence, not enthusiasm.
  `tests/test_rotation.py` proves all of this fires.
- **The direction space is analysed, not just the axes.** `tools/direction.py` also
  reports per-mechanism coverage and a DIRECTION SPACE table: every intervention family
  sorted least-covered first, with the axes inside it that are still untouched and the
  family's fixed-time cost class. A family can look busy while hiding a virgin axis --
  A family can show runs while one of its axes has never been tried once.
  That table is what a council round reads first. `data_curriculum` is listed with zero
  reachable mechanisms so nobody spends a round rediscovering that it is behind the
  frozen `prepare.py`.

## Non-negotiable

- Never edit `prepare.py`, the evaluator, tokenizer, or the 300s budget.
- Never claim GPUs are running without owner-filtered evidence — `tools/health.py` reports
  only processes owned by us. The box has other tenants on every GPU.
- Never let a variant ship without proving its edit applied. `make_variant.sub()` raises
  on an absent target; do not catch it to "keep going".
- Never state a number you did not recompute from `runs/sweep/results/*.json`.
- Co-tenancy on a GPU invalidates the run. Do not disable that check.

## Layout

```
baseline/            pinned upstream train.py + FROZEN prepare.py + provenance
tools/lit.py         corpus: screen 2025-26 arXiv, fetch full text, map to directions
tools/claims.py      atomic claims + causal mechanisms, append-only, snapshot-backed
tools/agenda.py      THE explore/exploit controller: active direction + forced rotation
tools/direction.py   axis/mechanism coverage and the per-axis dry rules
tools/make_variant.py train.py generator; asserts every edit applied
tools/council.py     the two multi-agent councils + content validator
tools/queue_from_round.py  round -> executable queue entries
tools/gate.py        health gate + decision cutoff
tools/health.py      20-minute liveness, owner-filtered
tools/analyze.py     honest reading of results
host/dispatch.py     remote scheduler (CPU-pinned, co-tenancy aware)
lit/                 index.json, sources/, claims.jsonl, mechanisms.jsonl, ACTIVE_DIRECTION
rounds/ critiques/   council artifacts (validated on content)
tools/coe.py         chain-of-evidence audit + the numeric registry
templates/           the shape of every record and document, with commentary
tests/               executable proofs of the policy and the evidence chain
runs/sweep/results/  the only source of experimental truth
karpathy_pristine/   untouched upstream train.py + prepare.py for digest comparison

Nothing else is authoritative. There is no retired-harness documentation in this tree on
purpose: the previous system's protocol described records, councils and gates that no
longer exist, and reading it would send you looking for machinery that was deleted.
```
