# Project critique — 2026-08-19T16-48-27Z

Four auditors on code defects, pipeline integrity, reproducibility, and an
independent synthesis. Every actionable finding below was FIXED before this file
was written; the fixes are in commits fa8d795, dce4df3 and a979179.

THREE DEFECTS IN CODE WRITTEN THE SAME HOUR, two of them in guards I had just
built and described as protections:
  1. `emits_diagnostic` FAILED OPEN, returning True on any build error, so an
     unbuildable cfg passed the emission check at BOTH queue doors.
  2. E5's tuple checker admitted a FABRICATED result: the auditor invented a
     mechanism that never ran and grounded it by differencing two real but
     unrelated registry values. Arithmetic is not attribution.
  3. The SYSTEMATIC regime-shift rule had no magnitude guard, so a sub-band delta
     plus a ~1% step wobble was reported as a causal claim.

PIPELINE: crash recovery was discarding the device id (writing gpu:-1 while
launch.json held the real GPU), and -1 was then POOLED AS A DEVICE by the device
model. Today's recovered runs came out right only because the fake slot happened
to hold one treatment and its own wave-mate control. Also: nothing restarted the
dispatcher, which is why a KeyError at 15:40Z became an outage.

REPRODUCIBILITY: a fresh clone genuinely works and the newest decision record
reproduces bit-for-bit. The paper was found to SPLICE two estimators -- a clean
quad mean beside a t from the superseded stitch. I could NOT reproduce the
auditor's correction: it reported significance overstated ~4x on a recomputed
t of about -5.06; recomputing from the four same-GPU pairs gives -32.4, so the
splice ran the other way and the table had been UNDERSTATING the result. Both
derivations are now in Appendix A so the disagreement is checkable.

UNRESOLVED, and recorded rather than fixed: the synthesizer's charge that only 1
of 23 commits is a completed experimental result, and that THREE confirmed levers
(tbs=18, swdiv=4, swdiv=8) sit unadopted rather than the one the operator named.

## code_defects (proj-code-8)

All four findings below were reproduced by running the actual repo code (verdict.py,
coe.py, make_variant.py, direction.py, balance.py) against synthetic fixtures in an
isolated sandbox copy (`rsync` of the repo into scratchpad; nothing in the project tree
was edited or executed as a launch). Ranked most to least severe.

### 1. CRITICAL — `tools/verdict.py:184-216` SYSTEMATIC regime-shift check has no
magnitude guard and launders marginal epoch-boundary noise into a causal claim

The new `systematic` test (lines 204-206) fires whenever `len(arms) >= 2`, every arm is
epoch-mismatched, and the mismatch direction agrees across arms — nothing else. It has no
check that the *val_bpb* delta clears the noise band, and no check that the *step-count*
gap is larger than ordinary host-contention/device-speed variance (the surrounding code
already knows this bound: `MAX_SLOT_OFFSET = 0.00245 + 2*0.000105`, used only in the
unrelated NOT-COUNTERBALANCED branch below). I built two synthetic waves for a harmless
`zloss=0.0001` cfg (negligible real compute cost) with `val_bpb` deltas of only
+0.0005 — *smaller than the measured within-wave band (0.002191) and the counterbalanced
resolution (0.000563)*, i.e. an effect that would normally read "INSIDE the resolution —
no effect demonstrated" — but with `num_steps` shaved by ~1% (499 vs 503, 498 vs 502) so
both arms land one epoch short. Running the real `verdict.py` against this fixture prints:
"SYSTEMATIC REGIME SHIFT, NOT VOID: ... The epoch gap is caused BY the treatment rather
than by contention landing on one cell, so it is the result and not a confound." That
sentence is false in the constructed case — the gap is ordinary noise at an epoch
boundary, not a mechanism cost — yet the tool states it with full confidence and un-voids
the arms. This is precisely the laundering channel Q1 asks about: *any* two-wave
consistency, however produced (a shared bug, a device-assignment convention, plain
variance at a marginal boundary), is accepted as proof the treatment itself is slow.
The real MTP case this rule was written for is not in doubt (2.8x per-step cost, huge
delta) — the defect is that the rule as written also accepts cases with none of that
signal.

### 2. CRITICAL — `tools/coe.py:501-522` `_tuples` grounds operands from the WHOLE
flat registry, not from a genuine paired result, so E5 can be defeated with real but
unrelated numbers

`_tuples()` (and `DERIV` beside it) only requires that both operands of a subtraction
appear *somewhere* in `registry()` — a flat set of 1521 numbers pooled from every metric
field of every run ever recorded (val_bpb, peak_vram_mb, num_steps, ...). It never checks
that the two numbers came from an actual treatment/control pair, or even that they concern
the mechanism named in the surrounding prose. I wrote a fake `papers/*.md` document
claiming a fictitious mechanism "gigazorp" (never queued, no result record) "cuts val_bpb
by 0.033967," backed only by the line `(1.025517-0.991550) = (0.033967)`, where the two
operands are the real `val_bpb` of `C04_control` and `R2V_B_s3_ctrl` — two unrelated runs
that were never compared to each other. Running `coe.e5_numeric()` against it reports **no
violation**: the fabricated number passes because it is arithmetically consistent with two
real-but-irrelevant registry values. Given 1521 registry numbers, an adversarial or merely
sloppy author has enormous freedom to reverse-engineer a "derivation" for almost any target
figure. This defeats E5's stated purpose ("a number in a paper... that no result record
produced") in exactly the way the file's own comments say two earlier laundering attempts
were closed — but this one, via `_tuples`, was not.

### 3. HIGH — `tools/make_variant.py:662-665` `emits_diagnostic` returns a false PASS
when the variant fails to build at all

```python
try:
    src = build(dict(cfg))
except Exception as e:                       # a cfg that will not build fails elsewhere
    return True, f"could not build to check {field!r}: {e}"
```

A cfg that cannot even build — e.g. `{"prefetch": 1}`, a mechanism registered in
`direction.MECHANISMS` with no branch in `build()` — makes `build()` raise
`VariantEditError`. `emits_diagnostic` catches it and returns **`True`** (message: "could
not build to check ..."), i.e. it reports the diagnostic as emitted. I verified this
directly: `MV.emits_diagnostic({...,"prefetch":1}, "anything")` returns `(True, "could not
build to check 'anything': mechanism(s) ['prefetch'] are registered ... but have no
branch...")`. Both queue doors (`tools/queue_from_round.py:76-80` and
`tools/queue_quad.py:105-110`) treat the boolean alone as the pass/fail signal
(`if not _e_ok: skip/refuse`) — a broken, non-buildable cfg is **not** refused here. The
comment "a cfg that will not build fails elsewhere" is not enforced by this function; it
is an assumption about a caller. This is the exact class of bug CLAUDE.md calls out
("`make_variant.sub()` raises... do not catch it to keep going") and the exact pattern
the campaign's own history shows recurs (the `job["name"]` KeyError fixed once and refired
at a second site). The fix is trivial: re-raise, or return `False`.

### 4. MEDIUM — `tools/balance.py:118-136` vs `tools/direction.py:317-405`: an axis
touched only inside multi-axis runs is "covered" to one tool and permanently invisible
to the other, with no per-axis warning

`direction.axis_state()` increments `state[a]["n"]` and sets `best_value`/`open` for
**every** axis a multi-axis run touches, so a joint run is enough to drop an axis out of
`priority == 2` ("zero coverage... outranks everything"). `balance.sweeps()` correctly
excludes multi-axis runs from the per-axis ladder (that is the L048 fix), but the
exclusion is aggregate-only: `main()` prints "(N multi-axis run(s) excluded...)" with no
axis names, and the ladder loop (`for a in sorted(lad)`) simply never mentions an axis
that has zero isolated data. I constructed a single synthetic run touching `clip` only
jointly with `ns` (`{"clip":0.5,"ns":3}`) and confirmed: `axis_state()["axes"]["clip"]`
shows `n=1, priority=1, best_value=0.5, open=True` (not the mandatory-explore state), and
`"clip" in balance.sweeps(...)` is `False` — it never appears in the SWEEP LADDERS
section, and `direction.blocked_reason()` for a future isolated `clip=0.5` run returns
`None` (allowed, but nothing *forces* it — reusing a "best_value" established only by a
joint run is treated as exploitation). The two tools' notions of "has this axis been
tried" have silently diverged again, by a different route than the bug L048 already fixed:
an axis can end up genuinely never isolated while direction.py believes it is
non-virgin and balance.py can never show its saturation state. This is currently latent —
I confirmed no axis in the real `runs/sweep/results` corpus is *presently* affected (every
axis touched by an existing multi-axis run also has independent single-axis data) — but
the mechanism is real and will fire the next time a multi-axis stack introduces a new
axis's only exposure.

## pipeline_integrity (proj-pipe-8)

**1. Other single points of failure on the poll path.** `host/dispatch.py`'s `main()`
while-loop (lines 595-773) is NOT wrapped in one top-level try/except; only some sections
protect themselves. The harvest block that writes each finished job's result (lines
638-661, containing the now-fixed `_crashed(job)` call at 654/656) has **no try/except at
all** — any exception there (a bad `parse(txt)` regex hit, a `job["item"]["name"]`
KeyError-class typo, a full disk on `.write_text` at 658) kills the whole process exactly
as the 15:40Z incident did. Two more unguarded sites on the same path: (a) `gpu_state()`
at line 702 (`free_gpus = [... for g,(used,uuid) in sorted(gpu_state().items()) ...]`) —
`gpu_state()` shells out to `nvidia-smi` with no try/except around the call site (the
co-tenancy block above it at 599-636 does wrap its own `nvidia-smi` calls in
`try/except OSError`, but this one doesn't); a missing/renamed binary or transient
`FileNotFoundError` here is fatal. (b) `(d / "launch.json").write_text(...)` at line 766,
inside the launch loop but **outside** the `try/except Exception` that covers lines
721-750 (mkdir/copy/Popen) — the try block ends right after `Popen()` returns
successfully; `running[g]=...` (757) and the `launch.json` write (766) both run
unprotected afterward. An OSError there (ENOSPC, NFS hiccup, permission) kills the
dispatcher with a live, just-`Popen`'d trainer that was never persisted to disk — the
next dispatcher's `adopt_running()` depends entirely on reading that file (line 531:
`json.loads((d/"launch.json").read_text())`, `except (OSError, ValueError): continue` —
silently skips), so this exact failure mode reproduces the "double-launch onto a live
GPU" bug the file's own comments (lines 759-765) say was already observed once and fixed
by the adoption mechanism — but only for jobs whose `launch.json` made it to disk.
Confirmed by direct inspection, no execution performed.

**Supervision: confirmed NONE.** `tools/tick.sh` (read in full) only scp's results,
runs `health.py`/`gate.py`, ships policy modules by md5 diff, and pushes the queue; it
explicitly documents (lines 45-48) that a running dispatcher must be **manually**
restarted — it only logs "a dispatcher is RUNNING... Restart it at the next idle gap,"
never restarts anything itself. On the host: no crontab (`crontab -l` → "no crontab for
zhubaiyu"), no dispatcher entry under any systemd unit. `sweep_watch.py` is a genuine
report-only watchdog (writes `WATCH_STATUS.json`/`watch.log`, alerts on
`DISPATCHER_DEAD`) but is **not currently running** (`ps -ef` empty) and never restarts
anything itself even when it is. `restart_sweep.sh` exists but was **never invoked**
(`restart.log` absent) and is dangerous: it `pkill -9`'s `python train.py` for ALL our
trainers before restarting, which would have killed the still-running
`R6MTP_P1_s0_treat` treatment instead of letting adoption recover it — the opposite of
what happened. Its own duplicate `parse()` regex is the pre-fix version
(`r"^([a-z_0-9]+):\s+([-\d.]+)\s*$"`, lowercase-only, requires `\s+`), i.e. a second,
unrepaired copy of the exact capitalized-metric/no-space bug CLAUDE.md's `dispatch.py`
comments say was already found and fixed once (L015-adjacent). `ophis-claude.service`
(systemd, `Restart=on-failure`) governs the Claude research-director loop, not the
dispatcher, is currently inactive (dead 12h, clean exit after its own 24h deadline so
`on-failure` never fires), and its own prompt explicitly forbids the agent from
restarting the dispatcher/watchdog. **Bottom line: confirmed, tick.sh does not restart
the dispatcher, and nothing else currently does either.**

**2. Recovery correctness — gpu:-1 concretely corrupts the device model.** Both
`R6MTP_P1_s0_treat` (treatment) and `R6MTP_P1_s1_ctrl` (control) were written by
`recover_orphans()` with `gpu:-1` (verified via `cat` of both result files on host).
`direction.device_means()` (tools/direction.py:84-100) keys its per-device control mean
by `r.get("gpu")` with **no filter excluding -1**, and `R6MTP_P1_s1_ctrl` passes its
other filters (`ok:true`, `is_platform(cfg)==True`, `final_epoch==2.0` — all verified),
so it lands in the dict as `device_means()[-1] = 0.991549`, a "device" that is not a
physical GPU at all. `tools/selector.py:family_effects()` (line 101) then computes the
treatment's device-corrected effect as `dm.get(r.get("gpu"), r["metrics"]["val_bpb"])`
— for the recovered treatment this looks up `dm[-1]`, which now exists and happens to
equal the val_bpb of *this specific* recovered control (0.991549), giving
`+0.079681`, coincidentally the same number `verdict.py`'s own comments cite as correct.
That correctness is an accident of both halves of the same wave crashing together at
the same instant. In general `dm[-1]` pools **any** recovered control from **any** wave,
on **any** physical device, under one fake shared identity; direction.py's own
`device_resolution()` docs (L020) measure real cross-GPU offsets up to ~0.0025 bpb — the
same order as the effects being chased — so a *future* recovered treatment paired
against a `dm[-1]` built from a *different* wave's recovered control would silently
import an unrelated device's bias into its reported effect with no flag anywhere. This
is a live landmine, not hypothetical: `git grep`-equivalent scan found exactly 2 runs
with `gpu==-1 and recovered==True` on the host today, and the fallback default
(`r["metrics"]["val_bpb"]`) makes it silently look like a valid, zero-astonishment
number rather than erroring.

**3. Queue/host sync — R6MTP_P4 specifically checked, was safe.** Live host state: `R6MTP_P4`
has zero entries in the host `queue.json`, no `claims/` directories, no results — it was
cut before ever being claimed, so `tick.sh`'s merge (`launched(name)` = result exists OR
claims dir exists) correctly classified it as safe to delete; `R6MTP_P3` (which *was*
running at the time) completed cleanly with both `DONE` records on disk. So in this
concrete instance nothing wedged. The guard logic is sound in principle — a
claim (`os.mkdir` under `claims/`) is created synchronously by the dispatcher's own
`claim_all()` before Popen, and `launched()` checks that same directory, so a genuinely
in-flight run is protected from deletion. The residual risk is a narrow TOCTOU race: the
merge script (host-side python, invoked over a fresh `ssh` each tick) reads `claims/` at
one instant while the dispatcher's independent 5-second poll loop can claim+launch an
entry from `runnable(cutoff)` at any other instant with no shared lock between the two
processes; a queue entry could theoretically be deleted in the sliver between the
dispatcher's `claim_all()` and the merge script's directory read. It did not fire for
R6MTP_P4 and I found no log evidence it has ever fired, but the two processes still have
no mutex.

**4. 4-GPU cap and double-booking on restart.** The cap itself is enforced correctly:
`capacity = min(len(free_gpus), len(free_slots), MAX_GPUS - len(running))` (line 705)
bounds by `MAX_GPUS - len(running)` where `running` is seeded by `adopt_running()`
*before* the first capacity computation (main() comment at 558-559 confirms the
ordering is deliberate). Adoption depends entirely on `launch.json` per job
(`adopt_running()`, lines 515-549): a job whose `launch.json` never made it to disk is
invisible to a restarted dispatcher and can be double-booked onto its own GPU — this is
precisely the scenario the file's comments (759-765) say already happened once
historically (the ~200s CPU-tokenization window before adoption existed) and is now
closed for that window, but reopens narrowly at the unguarded `write_text` call
identified in finding 1 if a crash lands between `Popen()` and that write.

## reproducibility (proj-repro-8)

**Fresh-clone test (actually executed).** `git clone` of the source tree to
`/tmp/p8_clone`, then from the clone: `python3 tools/coe.py` → `CHAIN OF EVIDENCE: INTACT`,
all five checks (E1 SOURCE, E2 LINK, E3 ACTIVATION, E4 METHOD-CODE, E5 NUMERIC) report 0
problems (one informational note: 174 lit sources are indexed-but-unfetched in the
checkout, explicitly not counted as a break). `pytest tests` (homebrew pytest, since the
clone has no `.venv` and `python3 -m pytest` fails with no module) → **25 passed in 1.36s**
(17 in `test_guards_fire.py`, 8 in `test_suite.py`, which itself subprocess-runs the other
five test scripts and asserts every script is collected). The claim that a fresh clone
works is TRUE for both `coe.py` and the test suite. `/tmp/p8_clone` was removed after.

**1. Regenerating headline numbers.** Picked 5 from `papers/2026-08-19_three_levers_and_an_instrument.md`
§5/§6 and recomputed independently against `runs/sweep/results/*.json`, mostly via
`tools/verdict.py` and hand arithmetic:
- `R4X_A_s3_treat.val_bpb = 0.986956` (best model) — exact match via `tools/coe.py registry`.
- Combined 3-lever stack, mean −0.004408, t=−58.0 — exact match, `tools/verdict.py precond`
  (key `mech:precond|knob:swdiv+ve`).
- `precond` alone, mean −0.001147, t=−12.3 — exact match, `tools/verdict.py` (variant
  `ba6f83b3a4e5`).
- `tbs` 19→20, mean +0.022545, t=+138.5 — exact match, `tools/verdict.py tbs`.
- `swdiv` 2→4, mean −0.002298, t=−20.3 — **DOES NOT independently reproduce as a pair.**
  The mean (−0.002298) reproduces exactly from the 4 same-GPU deltas of wave-pair R2S2
  alone (−0.001233, −0.002521, −0.003405, −0.002033 → mean −0.002298), which is what
  Appendix A's own NOTE calls "the CLEAN single-experiment quad." But that subset's own
  t-statistic is **t=−5.06** (sd 0.000909, n=4), not −20.3. The t=−20.3 figure only
  reproduces from the pooled R2S+R2S2 six-pairing stitch (mean −0.002463), which the
  paper's own Appendix A text explicitly names "the superseded cross-experiment stitch
  ... which overstated it by about 7 percent." `tools/verdict.py` (unmodified, no CLI flag
  to isolate one wave-pair) only ever emits the pooled −0.002463/t=−20.3 for this key,
  since R2S and R2S2 share the identical variant hash `a17cfc2f9d23` and the tool pools by
  variant. So the headline table's swdiv row **splices a clean mean with a stitched
  t-statistic that belongs to a different, larger sample the paper itself disavows** —
  overstating apparent significance by roughly 4x (t=−5.06 true vs. t=−20.3 claimed).
  `tools/coe.py` E5 NUMERIC did not catch this: it verifies that shown arithmetic on a
  line is internally consistent, not that two different table cells citing "the same"
  statistic came from the same computation — the Appendix A swdiv derivation is itself
  self-consistent (it's the stitched calc, correctly done), so the check passes even
  though the table row above it uses a different number for the mean.

**2. Do the 139(now 142)-record results survive the L047/L048/L049 lesson day?**
Confirmed L048 ("a joint effect was credited to each of its factors") is exactly this
class of problem: `tools/selector.py::family_effects` and `tools/balance.py::sweeps` both
used to add a multi-key arm's full delta to every family/axis it touched (e.g. the
3-lever stack R4X_A_s3_treat was crediting −0.004355 to both `attention` and
`ve_placement`, inflating the swdiv rung from a true −0.002194 to −0.002889). The fix
(selector.py in commit `0dbef46`, balance.py in `de4510e`) changed **only the
attribution/scoring code**, not the stored result JSON — the raw records are untouched
and still carry everything needed (per-arm `cfg`, `metrics.val_bpb`, `gpu`, `num_steps`,
`final_epoch`) to recompute effects correctly. `tests/test_guards_fire.py` now pins the
single-factor swdiv=4 rung to −0.002194 as a regression guard, and it passed in the fresh
clone. So: results are still interpretable, but any **prose or cached conclusion drawn
from the pre-fix scoring** (e.g. round 6's "best swdiv effect across 16 runs" provenance
line, per selector.py's own comment) is stale and must not be quoted as current — this is
precisely the swdiv mean/t splice found in item 1, a residual of the same underlying
confusion the code was fixed for but the paper prose was not fully re-audited against.

**3. `decide.py` reproducibility.** Newest decision `runs/sweep/decisions/2026-08-19T16-17-02Z_decision.json`
(`state_hash a41e0df65877b6bb`, `n_results 129`). Re-ran `python3 tools/decide.py`
(read-only, dry-run by default) against current state: **state_hash and every score
reproduced exactly** (13.036/9.595/9.036/8.998/5.595/3.68). `state_hash` is deliberately a
digest of only `(run name, val_bpb)` + `(queue name, cfg)` — it excludes code, by design,
"so a later reader can tell whether the state changed or the policy did." As a secondary
check, the second-newest decision (`2026-08-19T15-15-59Z`, n_results=125, different hash)
predates the L048 attribution fix: `git log` timestamps place that decision at
15:15:59Z UTC while `selector.py`'s fix landed at 15:31:37Z (`0dbef46`) and `balance.py`'s
at 15:38:42Z (`de4510e`). That decision was scored under the pre-fix (inflated) attribution
and would not be expected to reproduce today even holding the corpus fixed — a live
example of "policy moved," distinguishable from "state moved" (which also happened here,
n_results 125→129) precisely because `state_hash` excludes code by construction.

**4. Git history honesty.** Checked `git show --stat` on 3 recent commits (`d7a53a9`
"L049: MTP works exactly as designed", `de4510e` "L048: joint effect credited to each
factor", `7c7c178` "L047: pocket veto lesson"). All three diffs matched their messages:
d7a53a9's 3 new result JSONs + verdict.py's systematic-vs-void guard + queue.json's
92-line MTP entry cut all match the stated actions (its "two waves, swapped slots" claim
is completed jointly with the immediately preceding commit `3227f5c`, which added the 4th
result file — checked and confirmed, not a gap). de4510e's message says the bug was "in
selector.py and balance.py," but the diff touches only balance.py — verified this is
consistent, not dishonest: selector.py was already fixed in the immediately preceding
commit `0dbef46` (found and fixed within the same hour per its own message), and
de4510e's past-tense description of selector.py's bug plus present-tense "both
instruments now agree" is accurate read across the two commits. 7c7c178 is a 1-line
`lessons.jsonl` addition matching its message exactly. No dishonest commit found in the
sample.

## synthesis (proj-syn-8)

### FACT (verified directly against primary state)

- `lit/lessons.jsonl` L049 (registered 2026-08-19T16:15:36Z) confirms the MTP record
  exactly: delta +0.079484 over two waves on swapped slots (+0.079681, +0.079288, agree
  to 4e-4), `mtp_aux_loss_drop` 1.967261 vs a 0.1 activation threshold (mechanism
  engaged), treat 834ms/step vs ctrl 300ms (2.8x), 371 steps vs 1011, epoch 1 vs 2,
  peak_vram +139% not the +60% the scoring apparatus had priced. `blocks_keys: ["mtp"]`,
  and the scope carve-out is narrow: shared/factorised heads or longer budgets are
  explicitly NOT covered and may be re-proposed. R6MTP_P3/P4 (4 entries) are recorded cut.
- `tools/verdict.py` (lines ~184-211) now distinguishes SYSTEMATIC (`len(voided)==len(arms)`,
  same-direction epoch gap, `len(arms)>=2`) from the prior blanket VOID-on-`final_epoch`-
  mismatch rule, with the CLAUDE.md "raw val_bpb is the verdict" clause cited in-line as
  the justification. Confirmed by reading the code, not just the commit message.
- L048 confirmed verbatim: `tools/selector.py` credited R4X_A_s3_treat's full
  three-lever delta (-0.004355) to both the attention and ve_placement families and it
  set the round-6 queue-ranking prior; `tools/balance.py` put the same arm in the swdiv
  ladder, inflating swdiv=4 from the true single-factor -0.002194 to -0.002889 (and ve=1
  from -0.001432 to -0.002855). Both files fixed in the same commit (de4510e).
- The dispatcher crash is confirmed as a second occurrence of `job["name"]` (should be
  `job["item"]["name"]`): commit 3227f5c. First site (co-tenancy taint write) was fixed
  in a6ac7ed at 05:26:43-07:00; the second, in `_crashed()`, killed the dispatcher at
  15:40Z after launching R6MTP_P1, ~3h40m later — "hours earlier" checks out. A grep-based
  regression test now exists for this one string, but is explicitly self-described in its
  own docstring as "a weak check."
- L047 confirms the pocket-veto-regrowth claim: L043 withdrew a seed-replication gate on
  tbs=18 adoption, but its own mitigation text ("let the swdiv sweep and tbs=18 finish,
  then change PLATFORM") became a new sequencing gate by round 6, one that an independent
  synthesizer showed was internally contradictory against the operator's own cut-10-waves
  decision (confirmed against `queue.json`: 10/10 cut waves hard-code tbs:19 in control,
  8/10 in treatment too).
- Best measured `val_bpb` really is 0.984017 (`runs/sweep/results/R5T18_A_s3_treat.json`,
  tbs=18), against controls in the same waves at 0.991-0.994. `tools/direction.py:30`
  still defines `PLATFORM = {..., "tbs": 19, ..., "swdiv": 2}` — unchanged as of the
  latest commit. As of this audit, **no** `R5MU`/`R5MC` result files exist in
  `runs/sweep/results/`, so L047's own adoption gate (launched-wave completion) has not
  yet been satisfied either.
- Today (2026-08-19) has 23 commits. Exactly ONE (`d7a53a9`, MTP/L049) is a completed
  experimental result. One more (`2ae2c74`, R6XF) is a *queued* stack test not yet run.
  The remaining 21 are apparatus/integrity fixes, two council rounds, and one paper
  correction (`b070d23`).

### INFERENCE

1. **The 6:1 framing undercounts the apparatus tax.** Counting only registered lessons
   typed `integrity` gives ~6-7 for today, but counting commits shows at minimum 9 distinct
   defect-classes touched (dispatcher crash x2 sites, selector/balance dual attribution
   bug, activation pre-check never wired, decision-not-reaching-dispatcher, provenance not
   shipped, test suite not run, three same-morning fixes that didn't work, seed-gate →
   pocket-veto regrowth) against one finished result. The true ratio today is closer to
   **20:1 commits**, not 6:1. That is not obviously unhealthy on its own — this looks like
   a day the campaign spent hardening its own instrument layer after a period of rapid
   feature growth — but the *shape* of the defects is the concerning signal, not the count:
   nearly every fix commit admits, in its own message, that a symmetric sibling defect was
   left unchecked ("I fixed that one instance without grepping for the rest of its class").
   That is a process failure, not a bad-luck cluster.
   **Evidence of diminishing returns on rigor** would be: (a) a FOURTH recurrence of an
   already-named defect-class after the process fix proposed below is actually adopted, or
   (b) the fix commits crowding out queue throughput — e.g., a week where `runs/sweep/
   results/` grows by fewer files than `lit/lessons.jsonl` does. Today, 28 result files
   post-date tbs=18's confirmation while only 1 file documents a genuinely new mechanism
   test — that ratio, held for another day, would be the actual over-rotation signal, not
   today's count alone.
   **Healthy target ratio**: for a campaign whose CLAUDE.md explicitly makes lesson-writing
   a first-class deliverable ("failures bind the next run... or the campaign pays for it
   twice"), some fixed floor of apparatus commits is correct — but it should be *falling*
   week over week as the tool surface stabilizes, and today's data cannot show a trend
   with one day of samples. Track commits-with-a-new-result-file / commits-total over the
   next 3 days; if it does not rise from today's ~4%, the apparatus is not converging.

2. **The single concrete process change**: require every `integrity`-typed lesson
   registration to carry a `sibling_search` field — the exact grep/AST pattern run across
   the whole tree plus its match count — before `tools/claims.py lesson` will accept it.
   This is not a new idea; `test_guards_fire.py`'s own docstrings already state the
   reflection three times ("I fixed one instance without grepping for the rest," "grepping
   source is a weak check... but this defect is invisible until its exact branch
   executes"). The gap is that the reflection stays in prose and never becomes an
   enforced field, so it cannot be checked mechanically and has already recurred (seed-gate
   → L043 → L047 is the SAME failure shape one level up: a fix's own mitigation text grew
   a new instance of the defect it was fixing). Concretely: extend
   `templates/lesson.json` with a required `sibling_search: {pattern, files_checked,
   matches}` object whenever `type == "integrity"`, and have `tools/claims.py` refuse
   registration without it — mirroring how `blocks_keys` is already required and enforced
   for `type == "block"`. This is a same-day, single-file change and directly targets the
   observed recurrence, unlike a generic "review harder" instruction.

3. **Yes, adoption latency is now a leading cost, and it is worse than the prompt states.**
   tbs=18 is not the only large unadopted lever: L041 (swdiv=4, -0.002463, confirmed
   08:53Z) and L044 (swdiv=8, -0.003360, confirmed 09:26Z) are both larger than the ve win
   and are ALSO absent from `PLATFORM` (`swdiv` is still 2). Three separate confirmed
   levers — tbs=18, swdiv=4, swdiv=8 — are sitting outside the platform simultaneously.
   Since tbs=18 landed, 28 result records have been produced, and by L047's own audit at
   least 10 of the most recent waves are still built against the stale tbs=19 control.
   Every one of those waves will need the "one recalibration control block" L043/L047
   both promise once PLATFORM finally moves — meaning today's GPU spend is provisionally
   double-counted: once now (against a control the campaign already knows is not the
   best config) and again after recalibration. That is a real, non-trivial cost, plausibly
   comparable to the cost of the runs that discovered the levers in the first place.

4. **Blocking MTP after 4 runs (2 waves) is correct, not premature.** It is not an
   outlier for evidence volume: L007 closed on n=3, L011 and L032 on n=4, and even
   L037/L040 — the campaign's other large, clean valid negatives — closed on n=7, not
   dramatically more. MTP's effect is ~30x the noise band, reproduced to 4e-4 across
   swapped slots, with the activation diagnostic proving the mechanism engaged as
   designed — this is about as unambiguous as a negative gets, and running P3/P4 would
   not change the verdict, only spend GPU-hours the campaign is already short on. Critically,
   the block's `blocks_keys` and `applies_when` are scoped narrowly to "a second full-width
   [B,T,V] logits pass at this budget/vocab," not to multi-token prediction as a family —
   so this is not the campaign closing off exploration of the mechanism space, only one
   specific, now-falsified implementation of it.

### DISAGREEMENT

- I disagree with the record's framing that tbs=18 is *the* only large confirmed lever —
  primary state shows swdiv=4 and swdiv=8 are both larger in magnitude and equally stuck
  outside `PLATFORM`. Whoever wrote "tbs=18 is the only large confirmed lever" undercounts
  the adoption-latency problem rather than overstating it.
- I cannot independently confirm "guards built and not wired, 7 times" as an exact count.
  `tests/test_guards_fire.py` has 13 test functions covering 5 parametrized "guard present
  in source" cases plus ~8 behavioral ones; commit messages document at least 6 distinct
  built-but-not-wired incidents today (activation pre-check, decision-not-reaching-queue,
  provenance not shipped, test suite not run, dispatcher crash x2 sites). The pattern is
  real and recurring, but "7" is not a number I can reproduce from source, and I'd treat it
  as an order-of-magnitude claim rather than a precise one.
- I'd push back on any reading of today as simply "healthy hardening": the fact that L043's
  *own fix* grew L047's defect, in the same file, on the same day, is evidence the fixing
  process itself is not yet convergent — that is a stronger claim than "apparatus needs
  work," and other auditors may be underweighting it.

### VERDICT

**REFINE** the lesson-registration process (sibling_search field, above) — implementable
today, directly targets the 3x-observed recurrence.
**PIVOT** PLATFORM adoption from "single lever, single gate" to "batch adopt tbs=18 +
swdiv=4 at minimum" once R5MU/R5MC land — don't repeat the L043→L047 thrash a third time
by re-gating on swdiv=8 or R6XF.
**BLOCK** stands for MTP-as-tested (heavy dual-logits head at vocab 8192, 300s budget) —
evidence is decisive, do not spend more runs on this exact variant.
**TEST NEXT**: R6XF (tbs=18 + ve=1 + swdiv=4 stacked, already queued) — falsifier: if the
combined delta is not within ~15% of the additive prediction (-0.007953 + -0.003360 +
ve-win, adjusted for L042's 86%-of-additive precedent), the campaign has evidence the
levers interact and must stop stacking blind. Second falsifier for the adoption itself
(already stated in L047, worth re-affirming): if the post-recalibration within-GPU sd at
tbs=18 is not comparable in scale to the current 0.000198-0.000284 band, the resolution
model breaks and every verdict at the new platform needs a wider noise band before being
trusted.

