# Kickoff prompt for a new session

Paste everything below the line into a fresh Claude Code session started in
`/Users/baiyu/Desktop/OPHIS/simplify_autoresearch_v3`.

---

Read `CLAUDE.md` in full before doing anything else. You are running an autonomous
research campaign whose only objective is to lower `val_bpb` on a 300-charged-second
single-H200 run. `prepare.py` is frozen; `train.py` is the only file you edit.

## READ PAPERS BEFORE YOU RUN ANYTHING

This is not advice, it is the first phase of the work, and the tooling enforces it:
`tools/agenda.py` refuses to open any research direction until that direction has read
papers behind it, and `tools/gate.py` reports the refusal. Right now it says:

    ACTIVE DIRECTION: NONE
      NO ELIGIBLE DIRECTION: every family lacks usable claims.
      Build the corpus first (lit.py screen, lit.py fetch, claims.py add).

You cannot stage a treatment out of that state. The previous campaign ran its whole
experiment budget having made zero literature searches, drew its mechanisms from memory,
and none of the measured ones worked. Do Phase 0 first.

The repository carries no prior results, claims, or conclusions, deliberately. Your only
starting knowledge is what you read. Do not go looking for a previous campaign's runs,
and if you encounter any, do not reason from them.

## Phase 0 — build the literature corpus BEFORE any experiment

The previous campaign ran an entire experiment budget and made **zero** literature searches. Its
mechanisms were drawn from memory and produced no measured win. So the corpus comes first,
and the tooling enforces it: `tools/agenda.py` refuses to open any research direction
until that direction has read papers behind it, and `tools/gate.py` reports that refusal.

1. `python3 tools/lit.py screen --target 240`
   Runs ~47 mechanism-targeted arXiv queries restricted to 2025–2026, scores every hit
   for relevance to *this* operating point (124M params, ~600 steps, one GPU, Muon,
   wall-clock-bounded), and writes `lit/index.json`. Papers about inference, serving,
   RLHF, agents, fine-tuning or 7B-only claims are scored down — they cannot move this
   number. Expect this to take ~5 minutes; arXiv wants 3s between calls.
2. `python3 tools/lit.py fetch --limit 60`, repeatedly until you have 200+ full texts.
   Fetches LaTeXML full text into `lit/sources/`, newest-first then by relevance, and
   hashes each snapshot. Re-run it; it is idempotent and resumable.
3. `python3 tools/lit.py status` — coverage per direction family. Keep fetching until no
   family you care about is starved.

**Then read them and break them down.** Screening reads abstracts; that is triage and is
never evidence. A claim must resolve to a full-text snapshot on disk plus a locator —
`tools/claims.py` rejects a claim whose source was never fetched. For each paper you
actually read, extract atomic claims with `python3 tools/claims.py template` as the
schema, then `python3 tools/claims.py add file.json`. Rules the validator enforces:

- One attributable assertion per claim, worded no more strongly than the source.
- `internal_validity` and `transfer` are scored 0–4 **separately and never averaged**. A
  rigorous 7B/100B-token study can be internally excellent and nearly irrelevant here.
- Opposing evidence is registered with `stance: "opposes"` and stays visible.
- Group claims into causal mechanisms with `claims.py mech` — each needs a mediator, an
  activation observable, a falsifier, and the competing explanation to rule out.

Parallelise this. Dispatch subagents to read batches of full texts and return claim JSON;
you validate and register. `python3 tools/claims.py status` shows evidence per direction.

`templates/` holds the shape of every record and document — `claim.json`,
`mechanism.json`, `hypothesis.json`, `lesson.json`, `paper.md`, `round.md`,
`critique.md` — each field commented with the reason it exists, and
`templates/README.md` draws the whole chain of evidence. Read it before writing your
first claim.

**Before staging any experiment, register a hypothesis** (`claims.py hyp`). It cites the
mechanisms and claims behind it and declares an ACTIVATION PREDICATE: a diagnostic the
run must actually print, plus the rule it must satisfy. This is the rule that stops good
ideas being retired for free — a worse val_bpb from an intervention that never engaged
is **inconclusive**, never a negative result.

Run `python3 tools/coe.py` regularly. It audits the chain with five checks: E1 source
snapshots and digests, E2 dangling links, E3 activation, E4 method–code alignment
(including a variant byte-identical to the control), and E5 the numeric registry — every
number in a paper, round or critique must trace to a result record. `coe.py registry`
lists every value a document is allowed to cite. You may read it; nothing may add to it
by asserting a number in prose.

## Phase 1 — pick a direction, from the evidence

`python3 tools/agenda.py` is the explore/exploit controller. It names ONE active
direction and shows the arithmetic. A direction cannot open without at least one usable
supporting claim (transfer ≥ 2). Scoring balances usable claims, mean transfer, untouched
axes inside the family, the family's fixed-time cost class, and spend so far.

Once a direction is active, **study it intensely**: `python3 tools/lit.py read <direction>`
gives its reading list, `python3 tools/claims.py show <direction>` shows what is
registered. Read the unread full texts and register claims before proposing anything. A
round whose direction still has unread papers is a round arguing from ignorance.

## Phase 2 — the four-agent council

`python3 tools/council.py round` prints the exact prompts. Dispatch four subagents **in
parallel**, each with its own context and its own agent id:

- **explorer** — 3–5 interventions concentrated on the active direction, each citing the
  `belief_key` of the claims behind it, with a mediator and an activation observable.
- **pragmatist** — can it be built in `train.py` alone? What does it cost in throughput?
- **critic** — destroy each proposal before it runs; name the confound, the cheaper
  explanation, any registered `opposes` claim skipped past, and any claim being cited
  beyond what its transfer score supports.
- **synthesizer** — must be none of the above. Says which objection changed which
  proposal, preserves dissent, and ends with a ```queue fenced JSON list of runnable
  configs.

Write all four sections verbatim to `rounds/<UTC-ISO>_round.md`, then
`python3 tools/queue_from_round.py`. That builds every variant immediately, so a broken
edit fails now rather than at 3am on a GPU, and appends real work to the queue.

## Phase 3 — run, read, rotate

`python3 tools/analyze.py` after results land. Then `python3 tools/agenda.py` again — it
decides whether to keep exploiting or to move on. **Three independent triggers force
rotation to a different direction:**

- **DRY** — 4 consecutive runs in the family with no gain above the noise band, and no
  untouched axis left inside it.
- **STALE** — 6 runs since the last real gain, even if some axis is still untouched.
  Untouched is not the same as promising.
- **HARD CAP** — 12 runs in one family, unconditionally. This is the anti-monoculture
  backstop against spending a whole campaign inside one or two families.

An exhausted direction goes on cooldown and cannot be re-entered until every other
eligible direction has had a turn — or until new literature arrives for it. Fetching more
papers for a cooled direction genuinely reopens it. That is the intended way to revive a
direction you still believe in: bring evidence, not enthusiasm.

## Lessons — failures must bind the next run

Every failed or invalid run must become a typed lesson, or you pay for it twice.
`tools/gate.py` reports unlearned failures by name and `tools/analyze.py` lists them, so
a failure cannot be absorbed silently. **There are already two waiting for you** (see
current state below).

```bash
python3 tools/claims.py lessons            # active lessons, most severe first
python3 tools/claims.py lessons <family>   # just this direction's
python3 tools/claims.py lesson file.json   # register one (templates/lesson.json)
```

Typed `runtime | integrity | non_activation | valid_negative | degenerate | resource |
scope | overclaim`; action `proceed | refine | pivot | block`. A non-activation may not
block a direction — the intervention never engaged, so the run is inconclusive about the
mechanism, not against it; registration refuses that combination.

Two fields make a lesson bind instead of sit in a file. **`blocks_keys`** lists config
keys the lesson forbids, and `queue_from_round.py` refuses any entry setting one.
**`superseded_by`** names a newer lesson that replaces this one, retiring *this* one.
Lessons never expire on a timer: a scientific negative stays true until contradicted.

The council reads lessons before proposing — the critic must say which lessons it checked
each proposal against — and proposes new ones after.

## Standing duties

Arm these before research starts, and say honestly whether the runtime actually armed them.

- **Every 20 minutes**: `python3 tools/health.py`. Two separate questions — is Claude doing
  research (a research artifact changed on disk), and are OUR GPUs running (owner-filtered
  by uid, never raw `nvidia-smi` totals). For unattended operation run
  `nohup ./tools/tick.sh --loop >> runs/tick.out 2>&1 &`, which also publishes the gate
  verdict the dispatcher needs and syncs results and queue both ways. Without it the
  dispatcher's verdict expires after 20 minutes and it stops launching.
- **Every research block**: a council round (Phase 2).
- **Every hour**: the Fable critique. `python3 tools/council.py critique` prints the
  prompts. Dispatch four independent Fable subagents (model `claude-fable-5`) in parallel
  — evidence, method, process, synthesis — reading primary state (`runs/sweep/results/*.json`,
  the generated variant sources, `lit/claims.jsonl`, the host over read-only ssh), never
  your narration. Write `critiques/<UTC-ISO>_fable_critique.md`. Fable has no execution
  authority: it may not stage, kill, or edit `train.py`.

`python3 tools/council.py status` shows what is overdue. Both artifacts are validated on
**content** — missing sections, stub sections, repeated agent ids, a synthesizer who was
also a reviewer, or a missing queue block all fail. `touch` does not satisfy them. A stale
artifact freezes **new decisions** only; already-queued work keeps launching and controls
are always exempt, so GPUs never idle waiting for prose.

## Current state — read this before touching the host

**Six control runs have already executed.** Two were voided by GPU co-tenancy; four are
valid. `python3 tools/analyze.py` shows them. What they establish:

- The measured band is **0.0165**, reported honestly as a SEQUENTIAL spread because the
  four ran back-to-back on one GPU, not concurrently. There is no yoked pair yet.
- Step count on byte-identical code ranged 621–815, a 31% swing, and explains
  essentially all of the variance.
- **CPU pinning did not fix this.** All four ran pinned; `taskset` pins our process to
  cores, it does not reserve them against a host at ~180/192 load.
- Only one GPU was free at a time, so the runs went sequentially. **A yoked pair needs
  two free GPUs at once and the box may not give you that.**

The honest reading: with a 0.0165 band, almost nothing is resolvable from a single
unpaired run, so **your first research question is the instrument, not the model.** Put
that to the council. Two controls with genuinely overlapping execution intervals would
give the first real paired resolution.

Two unlearned failures are waiting (`C01_control`, `C02_control`, both co-tenancy).
Register lessons for them.

**A dispatcher may still be running on the host from an earlier session, with the
pre-fix policy loaded in memory.** Check before starting another:

```bash
python3 -c "import sys; sys.path.insert(0,'tools'); import hostcfg, subprocess; \
  subprocess.run(hostcfg.ssh_argv(['ps -u \$(whoami) -o pid,args | grep -E \"dispatch|train.py\" | grep -v grep']))"
```

Kill it and restart so it picks up the current `direction.py` and `dispatch.py`. A second
dispatcher is a no-op (it fails to take the singleton lock), so you will not double-book.

**Run on 4 GPUs, in waves.** `host/dispatch.py` caps at `MAX_GPUS = 4` and launches
entries sharing a `wave_group` concurrently or not at all, preferring the largest wave
that fits. Pair every treatment with its own control in the same wave — that is the only
way to get a paired resolution on this host, and without it you are stuck with the
~0.017 sequential band. When fewer GPUs are free than a wave needs, the dispatcher holds
and logs `WAITING: waiting for N free GPUs...` every 30 minutes. That is correct
behaviour, not a stall.

Watch capacity from here, every 30 minutes:

```bash
nohup python3 tools/gpu_watch.py --loop >> runs/gpu_watch.out 2>&1 &
```

To run:

1. `./tools/tick.sh --loop &` FIRST — the dispatcher refuses to launch without a gate
   verdict less than 20 minutes old, and the tick loop produces and ships it.
2. SSH to the trainer host (`python3 tools/hostcfg.py` shows whether it is configured;
   the address lives in `.ophis_host`, never in the repository — copy `.env.example`),
   then `cd ~/$OPHIS_REMOTE_DIR/sweep && nohup <venv>/bin/python dispatch.py $(python3 -c 'import time;print(time.time()+86400)') >> dispatch.out 2>&1 &`

Most GPUs carry foreign tenants; expect few launches. `tools/health.py` and
`tools/gpu_watch.py` report ownership accurately. Never claim GPUs are running without
owner-filtered evidence, and never disable the co-tenancy check — it is what voided
C01/C02 correctly.

## You start with no experimental priors

There are no archived results, deliberately. Your only starting knowledge is what you read
in papers. Do not go hunting for a previous campaign's runs, and if you encounter any, do
not reason from them — most of what they appeared to show did not survive scrutiny.

Two apparatus facts you should not have to rediscover, because they are about the
measuring device rather than about the model:

- `loader_frac` and `fwdbwd_frac` are wall-clock fractions measured on the CPU around
  calls that launch **asynchronous** CUDA work. `fwdbwd_frac` is kernel-launch time, not
  GPU busy time, and a small value does **not** show the GPU is idle. Any claim about
  where the 300 seconds goes must come from a controlled comparison — change the FLOPs and
  observe step count — never from those two fields alone.
- Byte-identical code re-run in a different wave moves far more than most effects worth
  chasing, because host CPU contention changes step count. That is why the default design
  is a yoked pair, and why `tools/direction.py` measures the band from concurrent controls
  rather than assuming one.

## Discipline

- Raw `val_bpb` is the verdict. Do not regress step count out and report the residual. A
  treatment that costs throughput is genuinely worse at a fixed 300s budget; that cost is
  the finding.
- The noise band is measured, not assumed — `tools/direction.py` reports it and how it
  was obtained. Do not report an effect smaller than it from a single unpaired run. Prefer
  a yoked pair: treatment and an exact control in the same wave on different GPUs.
- Never apply a step law across a change that moves tokens-per-step; fit and application
  must share an operating point.
- A claim needs a full-text snapshot and a locator. Never cite an abstract as evidence.
- Never state a number you did not recompute from `runs/sweep/results/*.json`.
- Never claim GPUs are running without owner-filtered evidence.

Then begin. Do not stop to ask whether to continue — run the loop until interrupted.
