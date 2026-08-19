# Project critique — 2026-08-19T22-48-38Z

Four auditors. Every actionable finding was fixed or explicitly recorded before this
artifact was written, and one finding was left deliberately unfixed.

## The governance layer cost more GPU time today than the science did

A 117-minute idle gap (20:19-22:16Z) is attributed by the dispatcher's OWN log
entirely to the decision cutoff, not contention. With ~154 GPU-minutes measured the
same way earlier, the freeze has cost roughly four GPU-hours in one day -- more than
every experiment run today consumed.

The cause is specific: the cutoff derives from the OLDEST stale council, so work a
CURRENT council approved is frozen behind a DIFFERENT council's staleness. Today a
fresh round council's experiments were frozen by a stale apparatus audit. Recorded as
L072 and NOT fixed tonight -- the same 'fix it quickly, it is costing GPUs' instinct
produced this afternoon's created_at inheritance bypass, which an audit then
demonstrated was a real hole.

## A headline number was prose again

The platform baseline was quoted all evening as 0.984205 and printed by NO tool.
Live recomputation gives 0.984181 over 14 controls, pooled within-GPU sd 0.000098
rather than 0.000104. Both had drifted silently. Same shape as the step law having
had no code (L055). analyze.py now prints PLATFORM BASELINE with its cfg. L071.

## Three defects in code written within the hour

The role resolver trusted a run's NAME over its cfg with no cross-check, and the
auditor demonstrated a crafted mismatch silently SIGN-FLIPPING a delta -- conflicts
are now detected and reported (38 currently, all pre-adoption waves). selector.py
called the activation pre-check WITHOUT the cfg, keeping the loose behaviour the
argument existed to close, at the site that decides what runs next. And the suite was
RED between commits because a test of mine asserted a state that expired when the
campaign produced the data it described (L069).

## Verified sound

Fresh clone passes chain and 35/35 tests; verdict.py reproduces byte-identically
including every headline effect size. CI would genuinely fail on a red suite, so
while the pre-commit hook is inert on a clone, the backstop is real. One dispatcher,
md5 matching, no new traceback, no wedged waves, zero stranded claims. The stale
in-memory is_platform has caused zero launch misfires. All 17 invalid runs trace to
a lesson. The convergence verdict: 0.0075 adopted gain against a 0.00060 band, about
12.5 sigma.

## Recorded, not fixed

The paper predates the adoption, the QK-norm result and the R5MC2 correction. The
quarantine backoff is untested in production. The per-council cutoff (L072).

## code_defects (proj-code-12)

All findings below were demonstrated by running code (in-process, no repo files touched), not inferred from reading.

### 1. HIGH — `verdict.py::_is_ctl` (tools/verdict.py:47-71) trusts the run NAME with zero cross-check against cfg, so one typo/rename silently flips a treatment/control pair and sign-flips the reported delta

`_ROLE_RE = re.compile(r"_s\d+_(treat|ctrl|control)$")` (line 71); `_is_ctl` returns the name-derived role unconditionally whenever the regex matches (line 65-67), falling back to `direction.is_platform(cfg)` only when it doesn't. I called `_is_ctl` directly with crafted members:

```
m_bad_ctrl  = {'name': 'RX_s0_ctrl',  'cfg': {**PLATFORM, 'tbs': 99}}   # actually a treatment
_is_ctl(m_bad_ctrl)  -> True   # trusted as CONTROL purely from the name
m_bad_treat = {'name': 'RX_s1_treat', 'cfg': dict(PLATFORM)}            # actually the control
_is_ctl(m_bad_treat) -> False  # trusted as TREATMENT purely from the name
```
Both confirmed by direct execution — cfg is never consulted once the name matches. Downstream, `main()`'s pairing loop (verdict.py:209-219) computes `delta = t["metrics"]["val_bpb"] - c["metrics"]["val_bpb"]` and keys the group on `label(t['cfg'])` (line 218). A single mislabeled member (exactly the kind of manual rename the campaign already performs — the commit log itself describes R5MU2/R5MC2 as "mechanical re-shapes" of prior runs) silently swaps which cfg is treated as control, flips the delta's sign, and mislabels the group by the wrong cfg's name — with no error, warning, or count discrepancy to flag it. The regex also accepts a bare `"control"` suffix in addition to `"ctrl"`; real result files (`Q1a_s0_control` … `Q1b_s3_control`) prove this alternative isn't dead code, so two independently-spelled conventions both feed the trusted, unvalidated path. Fix: cross-check the name-derived role against `is_platform(cfg)` when both exist and refuse (not silently prefer) on disagreement.

### 2. HIGH — `claims.diagnostic_would_discriminate`'s `cfg is not None` fix is correct in isolation, but `selector.py:228` calls it with **no cfg at all**, reproducing the exact loose behavior the fix closed

Confirmed the fix itself: `cfg={}` now correctly returns `False`/REFUSED for a rule nothing in the arm can satisfy (tools/claims.py:344). But `tools/selector.py:228` — `claims.diagnostic_would_discriminate(a["diagnostic"], a["rule"])` — never passes `cfg` (unlike `queue_quad.py:96` and `queue_from_round.py:68`, which do). I reproduced identical synthetic data (2 controls, 2 same-arm treatments, all failing the rule — the L050 double-failure pattern) through both call shapes: with `cfg=T` it correctly returns `False`/REFUSED; with `cfg` omitted (selector.py's actual call) it returns `True`/"UNVERIFIED", i.e. the activation-penalty term in the scoring function (`selector.py:216-231`, `act_pen`) stays 0 for a candidate whose diagnostic is unsatisfiable by design. This is decision-affecting: `selector.py` ranks/selects which hypothesis to queue next, and it can score a doomed-to-be-refused candidate as if activation risk is unverified-but-fine, while `queue_from_round.py` would refuse the very same cfg at the door. Pre-existing (not introduced by today's diff) but directly adjacent to the fix and unaddressed by it.

### 3. `.githooks/pre-commit` — confirmed unhooked on fresh clone

`git config --local --get core.hooksPath` → `.githooks`, and `.git/config:8` shows it set only in the local, untracked config file. `git ls-files .githooks/` confirms `.githooks/pre-commit` itself IS tracked and travels with a clone, but the `core.hooksPath` pointer that activates it does not — nothing in CLAUDE.md, a README, or a setup script runs `git config core.hooksPath .githooks` for a new clone, and there is no `.git/hooks/pre-commit` symlink either. What actually backstops a fresh clone is `.github/workflows/ci.yml`, which runs `pytest` and `tools/coe.py` (blocking, no `continue-on-error`) on push/PR — real enforcement, but at push time, not commit time, and only on GitHub. Note also a mismatch: the local hook treats `coe.py` failures as advisory ("NOTE… not blocking", pre-commit line ~24) while CI treats the same check as blocking.

### 4. Verified — coe.py post-hoc guard fires; no unintended deletion

Constructed a superseding hypothesis lacking `post_hoc_rule_change`: `e3_activation()` (tools/coe.py:201, guard at line 235) correctly appends the REFUSED-shaped message; adding the field clears it — confirmed by direct execution both ways. Also confirmed the silent PLATFORM-cfg exemption is gone: a hypothesis with `intervention.cfg == PLATFORM` and a non-emitted diagnostic is now flagged INCONCLUSIVE rather than silently skipped. `git diff a979179 HEAD -- tools/coe.py` (2 commits before the fix) shows the only net change besides the intended removal is an import reorder: `import claims as C` lost its trailing `# noqa: E402` when `import direction as _d  # noqa: E402` was added on the next line — a harmless but real lint regression (advisory only, since CI's `ruff check` step has `continue-on-error: true`).

### 5. Untested, decision-affecting

The entire `_is_ctl`/`_ROLE_RE` mechanism (item 1) has zero test coverage — `grep -rn "_is_ctl\|_ROLE_RE" tests/*.py` returns nothing — despite being the mechanism that just changed 15 verdict groups' worth of reported effect sizes (including a corrected R5MC2 number the commit itself calls out). The currently-checked-out working tree's test suite is also RED right now (`test_guards_fire.py::test_activation_precheck_says_unverified_rather_than_ok`, caused by two new untracked result files `R7NOQK_P1_s0_treat.json`/`s1_ctrl.json` flipping a live-corpus assertion) — meaning the just-installed pre-commit hook would currently refuse the next commit outright.

## pipeline_integrity (proj-pipe-12)

**1. Stale in-memory policy — confirmed, and scoped.** The live dispatcher (PID 313973,
started 2026-08-19T17:22:28Z, `dispatch.py 1787181748.8349984`) imports `direction.py`
once at startup via `from direction import axis_state, blocked_reason, label` (module
top) and `from direction import is_platform` inside `_is_control()` — the latter is a
function-local import but resolves to the *cached* `sys.modules['direction']`, so it is
just as stale. On-disk `direction.py` now has `PLATFORM = {..., "tbs": 18, ...}`
(confirmed by direct read). The queue majority (152/226 entries) still carries the old
baseline `tbs=19`, and 66 carry `tbs=18` — so the process's cached PLATFORM almost
certainly still says `tbs=19`. Evidence in `dispatch.log`: every `tbs=18` run this
session logs `[knob:tbs]`, e.g. `19:30:41Z LAUNCH RECAL18_W1_s0_recal ... [knob:tbs]`,
never `[control]` — 14 already-executed runs (RECAL18 W1–W4, R5MU2 P1–P3, R5MC2 P1–P4)
were mislabeled this way. `is_platform`/`_is_control` is used in exactly three places in
`dispatch.py`: (a) `label()` at the LAUNCH log line — cosmetic but the visible symptom;
(b) `runnable()` line 156 — decision-cutoff freeze exemption for controls; (c)
`wave_sizes()` line 195 — same exemption for wave-size accounting. **Is the exemption
inverted?** Yes, directionally: a genuine `tbs=18` control now fails `_is_control()` and
loses its freeze exemption; a `tbs=19` run would wrongly pass it. **Does it matter right
now?** I checked all 42 still-pending queue entries against both the current and the
inferred pre-adoption PLATFORM: 21 are true `tbs=18` controls the stale policy would
mislabel, but **none** have `created_at` past the current `decision_cutoff`
(1787178253 / 22:24:13Z), so the freeze-exemption bug has caused **zero** actual
misfires so far — the live harm is bookkeeping (axis/control mislabeling feeding
`axis_state`'s DRY_STREAK accounting), not blocked or wrongly-permitted launches, yet.

**2. One dispatcher, file matches repo, no new traceback.** `ps aux` shows exactly one
`dispatch.py` process. `md5(sweep/dispatch.py)` = `383ba34bae01cee0a33d190750d65ea5`,
identical to local `host/dispatch.py`. `dispatch.out` contains two "another dispatcher
holds the lock; exiting" lines plus one `KeyError: 'name'` traceback in `_crashed()`
(via `/data3/...` mount alias of the same file) — but the file's mtime is
`2026-08-19T15:40:10Z`, unchanged since, confirming this is *the same* 15:40Z traceback
already known, not a new one. Note separately: `dispatch.py` on disk was last edited
2026-08-19T19:31:48Z, ~2h9m after the running process started — so the live process is
also running stale *code*, not just stale `direction.py` state (unverified diff, no
git history on the box).

**3. Quarantine: persisted, incremented, backoff inconsistent.** `quarantine.json`
shows `burns: {7533cdca:2, f6cb0281:2}` for gpu6/gpu7, incremented from two burn cycles
at 16:35Z and 16:48:15Z. The exponential-backoff release code exists (`QUARANTINE_CLEAR_S
* 2**burns`, capped at 45 min) and fired once with the new log format:
`17:21:21Z QUARANTINE lifted early for GPU-7533cdca...: no compute app for 8 min quiet
required after 0 prior burn(s)`. That "0 prior burn(s)" is inconsistent with the
persisted burn=2 for that UUID — a discrepancy I could not fully resolve (possible
GPU-slot/UUID relabeling across a restart, or an earlier burn attributed to a different
device); flagging rather than asserting a cause.

**4. No BLOCKED/WEDGED/SPLIT in the last 3h; no stranded claims.** Last `WAVE SPLIT`
entries are from 08:36–09:09Z (old). No `WAVE WEDGED` ever appears in the log. The last
`LAUNCH FAILED` (self-recovered) was 19:30:35Z, just outside the 3h window. Claims
directory: 182 entries, results: 184, work: 182 — 0 stranded (every claim resolves to a
result or an active work dir).

**5. ~117-minute idle gap, self-attributed to decision cutoff, not contention.**
Trainers ran 0/4 from `20:19:20Z` (last DONE, R5NOQK2_P2) to `22:16:07Z` (next LAUNCH,
R7NOQK_P1) — 116m47s, on our normal 2-GPU (gpu6/gpu7) pattern. During that entire
window `dispatch.log` logged `NO_RUNNABLE_WORK: 48 frozen by decision cutoff, 0 blocked
by explore/exploit policy` every ~15 min (20:30, 20:45, 21:00, 21:15×2, 21:30, 21:45,
22:00, 22:15) — never a `WAITING: N GPUs free` capacity message, so the dispatcher's own
account is that nothing was blocked by contention/quarantine, only by a stale council
artifact past `decision_cutoff`. `GATE_STATUS.json` (generated 22:24:13Z) shows
"round council current" at 74.1 min old and "critique council current" at 8.8 min old —
consistent with the freeze having just lifted around 22:16. That is ~117 idle-minutes on
2 GPUs (~234 GPU-min); I cannot directly reproduce the prior audit's ~154 GPU-min figure
since I don't know its exact window/unit convention, but this gap is at least comparable
in scale and, on a per-GPU basis, larger.

**6. No trainer-count-zero restart watcher found on the host; restart is still
warranted.** Full `ps -u zhubaiyu` shows only unrelated wait-loops (two `_probe`
watchers, one `vpa` queue-drain loop) — none reference `dispatch.py`,
`restart_sweep.sh`, or a trainer count. No crontab, no `at` jobs, no systemd user
timers, no screen/tmux sessions. If such a watcher exists it is not visible on this
host; separately, right now 2 `train.py` processes are running under the dispatcher
(R7NOQK_P3, launched 22:32:21Z), so its trigger condition (trainers==0) is currently
false regardless. Restart is still needed to pick up current `direction.py`/`dispatch.py`
even though no launch has yet been corrupted by the staleness.

## reproducibility (proj-repro-12)

**Fresh clone (actually executed).** `git clone /Users/baiyu/Desktop/OPHIS/simplify_autoresearch_v3 /tmp/p12_clone`, then `python3 tools/coe.py` -> `CHAIN OF EVIDENCE: INTACT`, all five checks (E1-E5) 0 problems, exit 0. Then the source tree's `.venv/bin/pytest tests -q` run against the clone: 35 tests collected (tests/test_guards_fire.py=27, tests/test_suite.py=8), 35 dots, exit 0, no failures. Confirmed the prior audit's claim: `git config --local --get core.hooksPath` inside the fresh clone returns nothing (exit 1) even though `.githooks/pre-commit` is a tracked, executable file in the tree — `core.hooksPath=.githooks` exists only in the *source* repo's local `.git/config`, and `git clone` does not carry local config, so the pre-commit gate is genuinely inert on a fresh clone; a red suite could be committed silently. However `.github/workflows/ci.yml` DOES catch it on push: `on: push: branches: ["**"]` runs `uv run --with pytest pytest -q` (no continue-on-error) then `uv run python tools/coe.py` (no continue-on-error) before a lint step that is allowed to fail — so a red suite or broken chain of evidence fails CI on push even though the local hook that would have caught it earlier never fires. Clone deleted afterward.

**Headline numbers.** Re-ran `tools/verdict.py` on the working tree and, separately, on the still-live clone before deleting it: byte-identical output (`diff` empty). Confirmed exactly: MTP +0.079367 (n=3, t=485.2), z-loss +0.146695 (n=2 pairings), R5MU2 -0.007498 (n=3), R5MC2 -0.005139 (n=4), QK-norm at adopted platform +0.010332 (n=2), quality-cost residual +0.012430, within-wave band 0.000604. NOT reproduced by any tool run today: the tbs=18 "platform baseline 0.984205" and "pooled within-GPU sd 0.000104" exist only in prose (lit/lessons.jsonl L061/L065, a critique, a round) — no tool prints either figure. Live `tools/verdict.py`/`direction.device_resolution` currently reports pooled within-GPU sd **0.000093** (not 0.000104, not L061's earlier 0.000079) because the platform-cfg control population has grown since L065 was registered — `direction.py`'s own docstring warns this number "drifts as foreign load changes" and is intentionally never pinned. Recomputing the platform-cfg control mean directly gives 0.984181 (n=14), close to but not exactly 0.984205. `best val_bpb 0.981536, ONE valid pair` was not independently re-verified against `analyze.py`'s ranked table in this pass.

**verdict.py role resolution.** `_is_ctl` now resolves role from the `_s<slot>_(treat|ctrl|control)` name suffix first, falling back to `direction.is_platform(cfg)` only for older unlabeled records. Confirmed exactly 38 `NOTE wave ...: N member(s) whose name and cfg disagree about role` lines, all annotated "expected for waves built before a platform adoption" — matches the ~38 expectation.

**Papers.** The PDF/TEX (compiled 12:32) report R5MC2 as **-0.005221 over 3** against a tbs=19 control — this no longer matches the current registry value of -0.005139 (n=4, tbs=18-referenced). The paper's own baseline/best figure is 0.984017 (framed as the tbs-halving result itself), not the post-adoption "platform baseline" 0.984205, and it contains no QK-norm ablation number at all. MTP (+0.079367 family) and z-loss (+0.146695) values in the PDF do match current verdict.py output.

**decide.py.** Not reproducible run-to-run by design: the latest decision record (2026-08-19T20-13-34Z) stamps `n_results: 159`; the corpus now has 184 results, so `state_hash` differs (`d99eda030fe38e18` now vs `e0d91ae99d0b3554` recorded). This is state moving, not policy moving — decide.py's ranking logic is unchanged and deterministic over its live inputs.

## synthesis (proj-syn-12)

**Verification pass.** All headline numbers in the prompt check out against primary
state (`runs/sweep/results/*.json`, `tools/analyze.py`, `tools/direction.py`,
`tools/claims.py lessons`, git log).

- FACT: 180 result files, 163 `ok:true`, 17 invalid (co-tenancy or stranded waves), each
  invalid run carries a registered lesson (`analyze.py` footer).
- FACT: tbs=18 control mean recomputes to 0.984205 (n=12 unpaired, sd 0.000516) against
  a tbs=19 control population averaging ~0.9917 (e.g. 0.991549, 0.991873, 0.991926,
  0.991949, 0.992006) — a real ~0.0075 gain, confirmed independently by L061 ("the
  instrument got tighter") and RECAL18 re-baselining runs.
- FACT: MTP — recomputed the three treat/ctrl pairs myself: (1.07123−0.991549)=+0.079681,
  (1.071258−0.99197)=+0.079288, (1.071057−0.991926)=+0.079131, mean **+0.079367**. Matches.
  Activation: `mtp_aux_loss_drop`=1.967261, genuinely engaged.
- FACT: z-loss — R6ZL pairs: (1.138373−0.991463)=+0.14691, (1.138089−0.991608)=+0.146481,
  mean **+0.146695**. Matches. Activation: `logz_sq_final`=0.163344 (<1.0 threshold, engaged).
  Note the *other* zloss01 wave (4 treat arms) is entirely stranded/invalid — the number
  rests on R6ZL only, not on 6 runs as a careless read of the filenames might suggest.
- FACT: QK-norm removal — R5NOQK2 pairs: (0.999987−0.991518)=+0.008469,
  (0.999458−0.991963)=+0.007495, mean **+0.007982 ≈ +0.008**. Matches, and both treat
  arms ran *more* steps than control (1044 vs 1011, 1055 vs 1004) — confirmed non-throughput
  negative (L063). Activation: `qk_q_rms_final`=1.1867 in treatment vs exactly 1.0 in
  control.
- FACT: R5MU2 mean delta recomputes to **−0.007485** (three pairs: −0.007169, −0.007722,
  −0.007603) vs the claimed −0.007498 — matches to 3 decimals.
- FACT: R5MC2 mean delta recomputes to **−0.005139** exactly (four pairs: −0.004524,
  −0.005503, −0.005635, −0.004893, mean −0.0051388). Matches.
- FACT: R6XF's 0.981536 (P1 pair only) is real; P2 and P3 replications are both
  `ok:false` with `invalid_reason: "gpu co-tenancy during the run"` — confirmed in
  `runs/sweep/results/R6XF_P2_*` and `R6XF_P3_*`. One valid pair, correctly excluded from
  the paper per L056.
- FACT: operator error record — two red-suite commits confirmed in the pre-commit hook's
  own commit message ("Twice in one session a commit landed on a red suite"), matching
  `f485a19`/an earlier one; one silent-weakening-while-claiming-to-revert confirmed at
  commit `7ff8411` ("I silently weakened E3 in the commit where I said I was reverting a
  weakening"). L054 and L055 are same-day recurrences of "quote a number without owning
  its derivation" (already lessoned once, then repeated).

**DISAGREEMENT with the prompt's framing:** none on the numbers. One nuance: L054 shows
the operator's *own* MTP quote (+0.079484, "two-pair mean") was already wrong before a
critic corrected it to +0.079367 — so the "four negatives" were negatives on first
principle but the operator's own arithmetic needed a second pass to land on the number
now being cited. That's evidence for Q3, not a contradiction of the state.

---

**1. Converging or self-maintaining apparatus?**
Converging, on the only number that counts. Trajectory: tbs=19 baseline ~0.9917 →
tbs=18 adopted baseline 0.984205 (−0.0075, real, n=12) → best single result 0.981536
(R6XF, 1 valid pair, not adopted, not in paper). The number that settles it: **the
adopted, load-bearing gain is 0.0075 against a measured noise band of 0.00060** (2×
within-wave sd, 8 controls) — a 12.5σ move, not noise. The apparatus repairs (21 lessons,
~15 fixes) are not self-maintenance for its own sake; L046/L053/L054/L057 all trace
directly to defects that were about to corrupt or already had corrupted a val_bpb verdict
(orphaned waves, silent E3 weakening, unbuildable hypotheses masquerading as controls).
Verdict: **converging**, with unusually high repair overhead because the instrument kept
breaking under its own adoption events (L062, L066, L067) — that overhead is the cost of
verifying gains rather than a sign gains aren't happening.

**2. Is 4:1 negative healthy?**
Yes, conditionally. Three of four negatives (MTP, z-loss, QK-norm) engaged their
mechanism and lost cleanly — L053/L054 show two of them share a throughput cause
(vocabulary-width pass) with QK-norm the one clean exception (non-throughput negative).
That is discovery, not noise: the campaign now knows a shape of intervention that reliably
fails here. What's unhealthy is L057: ten hypotheses were unbuildable, meaning the
proposal funnel was manufacturing plausible-looking ideas the tooling could never run —
that inflated the apparent proposal rate without inflating the runnable one. TEST NEXT:
given vocab-width passes lose (L053) and QK-norm removal loses despite buying steps
(L063), the next 10 runs should avoid (a) any mechanism materializing a second
[B,T,V]-shaped tensor and (b) further norm-removal knobs, and instead prioritize the
UNEXPLORED axes `direction.py` flags as top priority with zero runs: `dim`/`depth`/`mlp`
(capacity, but FLOP-costly per L019 — deprioritize), and the *zero-FLOP* ones first —
`rope`, `softcap`, `x0init` (signal_scale, free), `warmup`/`mu_const`/`ema`/`ema_start`
(schedule, free), and `win` (attention, 17 tbs-confounded runs exist but win itself is
untouched). `ve_placement` is the standout: 9 runs, "the one ZERO-FLOP capacity lever,"
already carries the campaign's best number (0.981536) and is open, not closed.

**3. Getting safer or accumulating complexity faster than verification?**
The measurement that distinguishes them: **lesson-discovery attribution** — of errors
caught today, what fraction were self-caught before an agent/critic found them, versus
found downstream. Recomputed: 0 of 5+ operator errors were self-caught first (all found
by agents/critics per the prompt's own count, consistent with L054/L055's "two
independent Fable critics found the same two arithmetic errors" and L067's chain of
critique commits). That ratio — **0/5, 0%** — is the actual signal, not the lesson count.
21 lessons and 15 repairs are *evidence the audit layer works*, not evidence the operator is
improving; the operator's self-catch rate is what would show the operator getting safer,
and it is currently zero. Two same-day recurrences of an already-lessoned error class
(quoting unverified numbers — L054/L055) and a guard silently weakened inside a commit
that claimed to fix a weakening (7ff8411) both point the same direction: **complexity
(lessons, fields, retirement chains) is currently outrunning the operator's own
verification discipline, and is being kept in bounds only by the multi-agent critique
layer catching it downstream.** That's a stable equilibrium only as long as the critique
cadence holds — and `gate.py` currently shows project council 181 min old against a
120 min cadence, i.e. **that safety net is itself overdue right now.**

**4. What should run next, and why?**
R7NOQK is running (noqknorm at tbs=18, 4 pairs, retesting L063's tbs=19 finding at the
adopted platform — necessary because L063 explicitly says "whether it transfers to
tbs=18 is exactly what R7NOQK tests; do not assume it does"). R7XF is queued: 4 pairs
(ve=1, swdiv=4, precond=pre stack) vs a proper tbs=18 control — this is the correct next
launch after R7NOQK clears a GPU pair, because it is the only experiment that can promote
0.981536 from "rests on one pair" to a real, adoptable result, and L056 already
identifies *why* the first attempt failed (replication waves landed on recently-burned
devices) with a fix shipped (8568b99, burn-count backoff doubling). Both are correctly
widened to 4 pairs given today's co-tenancy loss rate (R6XF lost 2/3 pairs to it).

**REFINE** the proposal funnel (L057's ten unbuildable hypotheses need real cfg-key
implementations or formal retirement, not silent re-proposal). **PIVOT** exploration
toward the zero-FLOP UNEXPLORED axes (signal_scale, schedule, ve_placement) now that two
vocabulary-pass mechanisms and one norm-removal are confirmed losers. **BLOCK** nothing
new — L053's `blocks_keys: [zloss]` already covers the live risk. **TEST NEXT**: let
R7NOQK finish, launch R7XF the moment 4 clean GPU pairs are available (prefer low
burn-count devices per L056), and route the *next* council round at `ve_placement` or
`signal_scale` rather than another attention/optimizer knob-tune — those axes are
UNEXPLORED and, per L019, are the FLOP-free levers most likely to move val_bpb without
paying the step-count tax that killed three of today's four negatives. Falsifier for
R7XF: if the 3-lever stack's mean delta at tbs=18 (n=4 valid pairs) falls within the
0.00060 noise band of a proper control, 0.981536 was a co-tenancy-thinned artifact, not
transferable, and must be retracted from candidacy for the paper permanently, not just
"excluded for now."

