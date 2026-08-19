# Project critique — 2026-08-19T19-20-15Z

Four auditors. Every actionable finding was fixed before this artifact was written.

## The findings that mattered

**A completed run was deleted from every verdict by ordinary housekeeping.**
verdict.py resolved a run's wave and variant only through queue.json, so cutting
R6MTP_P3's entry AFTER it had run silently reverted the verdict to two pairs while
the campaign reported three. The reproducibility auditor found it from outside and
reported the campaign's figure as unreproducible -- right that the tool disagreed,
wrong about which was correct. With the fix the raw three-pair mean IS the reported
one. L058 registered; the queue is a plan, the result is evidence, and the analysis
was reading the plan.

**The freeze bypass I introduced was worse than I described it.** Its inheritance
key was (cfg, hypothesis) only, so an operator could requeue an old reviewed pair
under entirely new rationale, falsifier and expectation and still inherit the
ancient timestamp -- demonstrated live against a real queue entry. The prose a
reviewer actually reads was the one thing excluded from the key. Now included.

**A flag I added an hour earlier was never consulted.** step_law_explains() returned
an `extrapolated` marker and nothing read it -- the build-and-not-connect failure
this campaign has hit seven times, committed inside the fix for the step law having
had no code at all. verdict.py now prints the step-law share and marks it.

**sibling_search accepted "n/a", "." and "did not search".** It now requires a
named search target. The floor was set at 120 characters, which rejected an honest
brief answer and would have taught the next author to pad rather than look; it is 60.

## Verified sound

Fresh clone passes chain and full suite. One dispatcher, md5 matching the repo, no
traceback newer than the known 15:40Z one. Burn counts persisted at 2 for gpu6/gpu7.
The four remaining gpu:null results are confirmed EXCLUDED from the device model.
tick.sh's supervision pgrep does NOT self-match. No wedged or split waves, 0 of 158
claims stranded. build() refusing unknown keys breaks no caller -- 0 of 206 live
queue entries affected. decide.py reproduces its last recorded decision under
state-only drift, which is what the state_hash exists to demonstrate.

## Recorded, not fixed

The paper predates today's results: it never mentions R5MU2, R6XF, z-loss or the
0.981536 best, and its swdiv correction is not reproducible from a fresh verdict.py
run, which still emits the stitched figure the paper flags as wrong.

## code_defects (proj-code-10)

Ranked by severity, all confirmed by direct execution against the real repo (read-only:
pure functions and JSON reads, no writes to `runs/sweep/queue.json` or `lit/`).

**1. `queue_quad.py:158-163` created_at inheritance keys on `(cfg, hypothesis_id)` only —
it never checks `rationale`/`falsifier`/`expected`, so it is a rationale-swap bypass, not
a width-reshape detector.** The matching key built at `tools/queue_quad.py:158-163` is
`(json.dumps(e.get("cfg"), sort_keys=True), e.get("hypothesis_id"))`. `--rationale`,
`--falsifier`, `--expected` are free-text CLI args (`tools/queue_quad.py:42-44`) written
straight onto the new entry (`tools/queue_quad.py:176`) with no comparison against the
prior entry's own text and no comparison against the registered hypothesis's stored
content in `lit/hypotheses.jsonl`. I reproduced the exact matching logic read-only
against the live `runs/sweep/queue.json` (206 entries, 25 distinct `(cfg,hyp)` keys with
a recorded `created_at`). Picked a real reviewed entry `precond_pre_slot0`
(`cfg={'dbs':128,...,'precond':'pre'}`, `hypothesis_id=hyp_precond_pre_r1_v2`,
`created_at=1787107829.39...`). A fresh `queue_quad.py` invocation with the *same* cfg and
*same* hyp id but `--rationale "UNRELATED CLAIM: testing whether this cfg masks a
doc-boundary bug, nothing to do with the original rationale"` computes the identical key
and inherits the identical old `created_at` — verified programmatically. `host/dispatch.py`'s
`runnable()` (line 149) and `wave_sizes()` (line 188) gate purely on
`item["created_at"] > cutoff`; nothing there or in `queue_quad.py` looks at rationale.
So an operator can restate what a queued cfg+hyp-id pair is claimed to be testing —
a new scientific decision, in this project's own vocabulary where rationale/falsifier IS
the decision content queue_from_round.py's docstring describes council reviewing — and it
launches immediately on an ancient timestamp, fully bypassing the council-staleness
freeze the commit's own message (8e11572) says it is restoring, not weakening. This is
worse than the stated "mechanical width re-shape" case: nothing in the code enforces that
the reused key actually IS a mechanical re-shape rather than a re-purposed experiment.

**2. `direction.step_law_explains()`'s `extrapolated` flag is dead — no caller anywhere
reads it: build-and-not-connect.** `grep -rn "step_law\|extrapolated"` across every `.py`
file in the repo shows `step_law`/`step_law_explains`/`extrapolated` are defined only in
`tools/direction.py` (lines 646, 693, 712) and never imported or called from any other
`.py` file — not `verdict.py`, not `coe.py`, not `council.py`, not any test. The only
places the string `extrapolated` or the fitted -0.06822 slope appear outside
`direction.py` are prose records (`lit/lessons.jsonl` L055, a critique markdown file).
The commit message for `e053aaf` claims "step_law_explains() now flags that rather than
quietly reading off a number," but the flag is flagged only to whoever manually calls the
function from a REPL — no automated caller consults it, so nothing stops a future prose
claim from reading off an extrapolated share as if it were in-range, exactly the failure
mode L055 was written to close. This is a second instance of the sibling pattern L055
itself names ("a rule that lives only in prose is one that gets ignored in practice") —
except this time it's a rule that lives only in an unread return value.

**3. `make_variant.build()` raising on unknown keys (`tools/make_variant.py:79-87`) does
NOT break any existing caller — verified, not a defect.** Checked every call site
(`council.py:200`, `coe.py:362,378`, `make_variant.py:667,684`, `queue_from_round.py:128,138`,
`queue_quad.py:118`, `queue_controls.py:114`, three test files). In every case `cfg` is a
clean dict containing only knob/mechanism/platform keys; bookkeeping fields
(`hypothesis_id`, `wave_group`, `created_at`, `variant`, `rationale`, ...) live as sibling
keys on the queue entry, never merged into `cfg`. `coe.py:362` and `:378` wrap `build()`
calls in `try/except Exception`, so even a legacy bad cfg would degrade to a reported
"break," not a crash. Read-only scan of the live `runs/sweep/queue.json` (206 entries)
found zero entries whose `cfg` contains an unknown key, so no retroactive breakage either.

**4. `claims.validate_lesson`'s `sibling_search` gate (`tools/claims.py:180-183`) is a
pure truthy-presence check, exactly as a prior audit alleged — confirmed by execution.**
Called `claims.validate_lesson` directly with `sibling_search` set to `"x"`, `"n/a"`,
`"asdf"`, `"."`, `"did not search"`, `"todo"` — every one of these is *accepted* (no
`sibling_search` complaint in the returned problem list). The check is literally
`if l.get("type") in ("integrity","runtime") and not l.get("sibling_search")`, i.e. any
non-empty string, including an explicit admission of non-compliance, satisfies it. A
concrete bite: require the field to name at least one concrete artifact reference (a
`grep`-able path/module token, or the literal string "found none" preceded by a named
search target) via a regex such as `re.search(r'\b(tools|host|lit|runs)/\S+\.py\b', s)`
or-`"found none"`-with-a-preceding-noun-phrase check — currently even that minimal
structural bar is entirely absent.

Priority for fixing: #1 is the most damaging (an active governance bypass with a
demonstrated exploit path); #4 confirms a known un-enforced control; #2 is a real but
lower-urgency loose wire; #3 is a false alarm — the guard is safe as shipped.

## pipeline_integrity (proj-pipe-10)

**1. Single dispatcher, code parity, no fresh crash.** `ps aux` on the host shows exactly
one `dispatch.py` process, PID 313973, started 2026-08-20T01:22:28 local (=2026-08-19T17:22:28Z).
`dispatcher.lock` contains `313973`, matching. `md5sum ~/ophis_v3/sweep/dispatch.py` =
`3f2b3fa9507fb3d1d8f44bf6aab4169f`, identical to local `host/dispatch.py` in the repo —
code parity confirmed. `dispatch.out` holds exactly one Traceback (`KeyError: 'name'` at
`_crashed`, dispatch.py:460, called from main():646), and `stat` shows the file's last
write was 2026-08-19 23:40 local = **exactly 15:40:10Z** — the file has not been touched
since that known crash. The current dispatcher (started 17:22:28Z, ~1h46m ago) has logged
zero new tracebacks.

**2. Quarantine/burn backoff.** `quarantine.json` currently: `burns: {GPU-7533cdca(gpu6):2,
GPU-f6cb0281(gpu7):2}`, unchanged since the current process started — no new
`L002_burned_gpu_reused` events appear in dispatch.log after 16:48:15Z (before the restart).
dispatch.log shows the pre-fix defect directly: at 17:21:21Z, under the OLD in-memory-only
burn_count, gpu6 was released after "8 min quiet ... after **0** prior burn(s)" despite
having burned twice minutes earlier — the counter died with a restart. The current code
(dispatch.py:598-599, 609-610) persists/reloads burns via `quarantine.json`. Since restart,
gpu7's quarantine (release epoch 1787160795.159 = 17:33:15Z, matching burn=2's *fixed*
45-min timer exactly, not an early release) expired naturally; no burn=2 "lifted early"
doubled-window (32 min) log line has fired yet because gpu6/gpu7 haven't been re-quarantined
— both have run five clean back-to-back waves (R6ZL, R5MU2, R5MC2, rc=0) since 17:33Z.

**3. gpu:-1/null in results.** Grep finds 4 results with `gpu: null` (not -1):
`zloss01_{A_s0,A_s3,B_s1,B_s2}_treat` — all `ok: false`, `final_epoch: null`. Actually ran
`direction.device_means()` on all 160 results/*.json: keys returned are `[4,5,6,7]`, and
confirmed programmatically that all 4 null-gpu records are excluded (via the `ok` guard
and the explicit `g is None or g == -1` check at direction.py:104).

**4. tick.sh pgrep self-match.** Reproduced with a fake pattern (`ZPROBE_dispatch.py_...`):
a naive `pgrep -f` **does** self-match the sshd-spawned `bash -c "<cmd>"` wrapper when the
pattern text is echoed literally in the command line. But the *production* pattern
`dispatch.py [0-9]` does not self-match in practice: `[0-9]` is a real regex class requiring
a literal digit after "dispatch.py ", and the wrapper's own quoted pattern text has `[`
there, not a digit — real dispatchers match because their argv is `dispatch.py 1787...`.
Confirmed by excluding the live PID from `pgrep -f "dispatch.py [0-9]" | grep -v ^313973 |
grep -cv "bash -c"` (tick.sh's exact form): result is **0**, i.e. no self-match when no
dispatcher runs.

**5. Waves / stranded claims.** No WEDGED or SPLIT logged in the last 3 hours; last SPLIT
was 09:09:33Z (~10h ago). `claims/` has 158 entries; a script comparing claims vs
`results/*.json` and `work/*` found **0 stranded** (every claim has a result or work dir).

**6. GPU-time, last 3h (16:09Z–19:09Z).** No WAITING line shows "2+ free GPUs" with none
of ours running in this window. Closest: 16:57:23Z and 17:22:29Z, "waiting for 2 free
GPUs to launch a wave (**1** free now); 0/4 of ours running" — only 1 free, held for the
wave pairing discipline. Otherwise gpu6/gpu7 ran continuously (DONE→LAUNCH same-timestamp)
with 0/4 free logged at 16:35, 18:31, 19:01.

## reproducibility (proj-repro-10)

**Fresh clone.** `git clone` of the source tree to `/tmp/p10_clone` succeeded. `python3
tools/coe.py` on the clone printed `CHAIN OF EVIDENCE: INTACT` with all five checks
(E1-E5) at 0 problems (system `python3` is 3.14 with no installed deps, but `coe.py` has
none). `python3 -m pytest` is unavailable on system Python; using the source tree's
`.venv/bin/pytest` against the clone: **31 passed in 2.10s, 0 failed, 0 errors**. Clone
removed after use (`rm -rf /tmp/p10_clone`, verified gone).

**Step law.** `direction.step_law()` (tools/direction.py:646) fits controls only
(`is_platform(cfg)` true, i.e. no knob/mechanism/unknown key touched) at a single
`tokens_per_step`, as documented. Independently reimplementing the same filter/fit by
hand (not calling their function) reproduces its output exactly: slope
**-0.06821280427...**, n=87. This is close to but **not** the docstring's claimed
"-0.06822 over 84 controls" (tools/direction.py:658, also quoted in
`critiques/2026-08-19T18-31-29Z_fable_critique.md`). Root cause verified via git log:
three new control results (`R5MU2_P1_s1_ctrl`, `R5MU2_P2_s0_ctrl`, `R5MU2_P3_s0_ctrl`)
landed in commits after 11:24:57 (when direction.py's step_law was last edited), moving
n 84→87 and the slope from -0.06822 to -0.06821. This is **state drift, not a bug**: the
comment is a snapshot that the corpus outgrew within the same day. `step_law_explains()`
was tested directly: returns `None` when treat/ctrl `tokens_per_step` differ (confirmed),
and returns a populated dict with an `extrapolated` flag when they match (confirmed).

**Headline numbers**, recomputed independently from `runs/sweep/results/*.json` and
cross-checked against `tools/verdict.py`'s live output:
- z-loss +0.146695 (2 pairs, R6ZL): reproduces exactly, both raw and same-GPU-paired.
- R5MU2 raw -0.007498 / same-GPU -0.007386: both reproduce exactly (verdict.py: "raw
  within-wave: mean -0.007498" / "same-GPU re-paired: mean -0.007386").
- R6XF -0.010143 (1 pair): reproduces exactly.
- best val_bpb 0.981536: reproduces exactly, `min()` over all results resolves to
  `R6XF_P1_s0_treat.json`.
- **MTP +0.079367 (3 pairs) does NOT reproduce via tools/verdict.py.** `runs/sweep/queue.json`
  has `wave_group` entries for R6MTP_P1 and R6MTP_P2 only — R6MTP_P3's queue entries are
  absent — so `verdict.py`'s `waves()` silently drops P3 and computes its verdict from 2
  pairs only, giving **+0.079484**, not +0.079367 (mean of all 3 raw pair deltas
  0.079681/0.079288/0.079131 by hand). +0.079484 is exactly what the paper's Table 1
  reports, so this is not a paper error, but the "+0.079367 (3 pairs)" figure is not
  producible by any tool in the repo without hand-editing queue.json.

**Papers.** Only one paper exists (`2026-08-19_three_levers_and_an_instrument.{md,tex,pdf}`),
scoped to R2/R3/R4-series runs; it does not mention R5MU2, R6XF, z-loss, or 0.981536 at
all (those runs postdate or weren't folded into it). `.tex` and `.pdf` (via `pdftotext`)
agree verbatim on the MTP number (+0.079484) and Table 1. One internal inconsistency
verified in the `.tex` itself (present in the PDF too, so not a build artifact): §5.6
narrates "reproduced across three independent pairs" and lists all three raw deltas
(0.079681/0.079288/0.079131, mean 0.079367) but the number it actually reports,
+0.079484, is the tool's 2-pair same-GPU figure (see above) — text and headline number
describe different sample sizes. Separately, the paper's own text (§5.1) documents that
`tools/verdict.py`, run fresh today, still returns the value the paper flags as WRONG:
-0.002463 (t=-20.3, stitching two experiments 1.4h apart on the same variant hash
a17cfc2f9d23) rather than the paper's corrected -0.002298. I ran verdict.py on the clone
and confirmed it: the -0.002463 stitching bug is still live in the tool the paper
narrates it away from in prose.

**Decisions.** `python3 tools/decide.py` (no `--apply`, read-only) on 160 current
results vs. the last recorded decision (`2026-08-19T18-48-14Z_decision.json`,
n_results=141, top score 12.928, selected R5MC2_P1): current run gives top score 12.95,
same selection R5MC2_P1, same second/third-place ordering. `tools/decide.py` was last
modified 06:03:55, well before that decision record — confirmed via git log. The
divergence (12.928→12.95) is **state moving** (19 more results since the record), not
policy moving; the record's own `n_results`/`state_hash` fields are what make that
distinction checkable, and I checked it.

## synthesis (proj-syn-10)

**FACT (recomputed from `runs/sweep/results/*.json`, `lit/lessons.jsonl`, `tools/*.py` output, `git log`).**
141 valid runs, best `val_bpb` 0.981536 (R6XF_P1_s0_treat, single pair, P2/P3 replicates lost to co-tenancy per L056). Best-so-far trajectory across today (UTC): day opened at 0.990967 (carried from Aug-18 W03a_2); 03:25 precond_pre→0.990908; 04:23 precond-rms→0.990345; 07:04 ve→0.989819; 07:21 swdiv→0.989449; 08:36 swdiv²→0.989249; 08:52 precond+swdiv+ve stack (R4X)→0.986956; 09:25 tbs=18 (R5T18)→0.984017; 16:22 all-four stack (R6XF)→0.981536. That is 8 genuine new-bests in ~16h, a 0.00943 total drop against a measured GPU-counterbalanced resolution of 0.000277 and an unpaired-control band of 0.010442 — real, not noise, for everything except the final 0.002481 step (single pair). Since 16:22 (≈2.5h, through 18:59 now) there has been **zero** new best: the two "results" I was asked to check are MTP +0.079367 (recomputed: 3 valid pairs, deltas 0.079681/0.079288/0.079131, mean matches exactly) and z-loss +0.146696 (2 valid pairs — the original zloss01 wave's 4 treat arms are `ok:false`, "stranded"; the valid measurement is the R6ZL requeue, deltas 0.146910/0.146481, mean matches to the digit given). **R5MU2 does NOT match the claim given to me: it has 2 landed pairs, not 3** — `R5MU2_P3_s0/s1` sit in `queue.json`, not in `results/`. Recomputing on the 2 real pairs: deltas −0.007169/−0.007722, mean −0.0074455 (matches the stated −0.007445), but t on n=2/df=1 raw pairing is −26.9, not −90.2 — the −90.2 figure is unverifiable from primary state as given and likely inherits the same same-GPU-repairing inflation L055 already caught once today (raw t=683.9 vs re-paired t=2110.7 for z-loss). 42 commits since 00:00 local-equivalent; 11 lessons L047–L057 registered 15:15–18:42Z (54 lessons total carry an Aug-19 timestamp campaign-wide). Gate is open, CoE is INTACT (E1–E5 all 0 problems), only a stale project-critique cadence flag.

**INFERENCE.** The campaign converged hard through midday — every family that got tested (precond, ve, swdiv, tbs) won and stacked close to additively (L042: ~86%) — then the last ~3 hours pivoted almost entirely to apparatus repair and mechanism screening that both came back negative. That is not one story, it is two consecutive ones: a real-optimization morning followed by a governance-heavy afternoon. Both are defensible in isolation (MTP/z-loss were well-designed falsifiable tests per L049, and the repairs fixed real corruption — a backwards activation rule, a lesson that blocked the same experiment its own text left open, a step-law with no code). The unhealthy signal is injection rate, not correction rate per se: the critique itself counts 3 defects in code written the same hour, and 5 of the last 8 lessons are about the apparatus, not the model. Self-correction is only evidence of rigor if the injection rate is falling; three same-day self-corrections plus an unverifiable t-stat I just caught argue the rate is roughly flat.

**DISAGREEMENT (mine, against the operator's framing).** Question 1 presupposes a binary; the honest read is sequential, not either/or, and the val_bpb trajectory above is the falsifier either way — it shows real convergence for 16h then a flat 2.5h, not a campaign that "generates lessons about itself" as its steady state. Question 2's self-correction-count framing is under-specified without a numerator: three corrections found is meaningless without the base rate of claims made — 42 commits and 54 lessons in a day makes a 3-in-1-day correction rate look small, not large, but the R5MU2 t-stat mismatch I found on this very audit is a fourth uncaught error the operator has not yet self-corrected, which cuts the other way.

**1. Convergence: YES through 16:22, FLAT since.** The 0.00943 daily drop is 34x the counterbalanced resolution — genuine science happened. But the single largest jump (R6XF, −0.0025 beyond R5T18) is unreplicated after both replicate attempts were destroyed by co-tenancy (L056), so the campaign's own best number currently rests on n=1.

**2. Evidence of a claims-volume problem, not rigor.** Rigor would show a *falling* defect rate over the day; instead injection ≈ repair velocity, and I independently found a fourth unverified number (R5MU2's t=-90.2) in the same session. Distinguish them going forward by tracking defects-per-commit over a rolling window, not defects found per day.

**3. PLATFORM should become `{tbs:18}` alone on this trigger, not the four-way bundle.** The round-7 synthesizer's narrow read is correct against L047's literal text (it gates only tbs=18 on R5MU/R5MC); swdiv=8/ve=1/precond=pre are separately confirmed but their *joint* magnitude has exactly one surviving measurement (R6XF_P1) whose replicates were destroyed. Strongest argument against even the narrow adopt: R5MU2 itself is not yet the 3-pair result claimed — with the corrected 2-pair, unverifiable-t reading, waiting for the still-queued P3 (and R5MC2) before recalibrating is one queued run away and cheap; jumping now risks a second same-day platform recalibration if P3 disagrees.

**4. TEST NEXT (10 runs).** REFINE: let R5MU2_P3 + R5MC2 (all 4 pairs) finish — cheap, already ranked #1, closes L047 correctly (diagnostics `mu_warmup`/step-count confirmed emitted). REFINE: R6XF_P4 counterbalanced replicate to unstick the campaign's best result from n=1 (falsifier: if P4's delta lands outside [−0.0025±0.00219], the stack magnitude is wrong and the bundled-adopt argument collapses further). BLOCK stays on `mtp`/`zloss` per L049/L053 (verified `claims.blocking_keys() == {'mtp','zloss'}`). PIVOT to the zero-FLOP UNEXPLORED axes the DIRECTION SPACE table flags top-priority-untouched: `noqknorm` (signal_path, GPU-only near-free, diagnostic `qk_q_rms_final` verified emitted) and schedule's `warmup`/`mu_const`/`ema` (free cost class). Do NOT queue L057's ten unimplementable hypotheses until `make_variant.build()` grows branches for them — that is a code task, not an experiment. Recalibration control quad (2 runs) only after PLATFORM actually changes, not before.

