# Project critique — code and pipeline

Scope: the machinery, not the measurements. Three independent reviewers read primary
state and ran the tools; none read the others' findings. The largest unreviewed change in
the tree was `tools/mech_lib.py` and the MECHANISM_REGISTRY, written hours earlier and
never executed on a GPU, so it received explicit scrutiny.

## code_defects (agent-code-defects-01)

**CONFIRMED — `tools/direction.py:595` crashes the campaign loop the moment any new
mechanism produces a result.** `mechanism_state()` seeds its dict from the hard-coded
`MECHANISMS` tuple but indexes it with `mechanisms_touched(cfg)`, which resolves through
`all_mechanisms()` → the registry. Reproduced by writing one result file with
`cfg={…,"winsched":128}`: `direction.py`, `analyze.py` and `agenda.py` all exit 1 with
`KeyError: 'winsched'`. Trigger: the first completed run of any of the nine registered
mechanisms. `R10NGRAM` is queued on the host now. The analysis half of the campaign stops
working exactly when the new mechanisms start paying out, while the dispatcher keeps
launching — the campaign blind and still spending GPUs. Severity: critical.

**CONFIRMED — the registry's `family=` field is written and never read**, so 8 of 9
mechanisms are invisible to the controller. Nothing in `tools/` or `host/` reads
`spec["family"]`; `direction.MECHANISM_FAMILIES` remained hand-maintained. Reproduced
against real results: `signal_path mechanism 10 all closed` while periln/vnorm/ffnpost
live there; `ve_placement` and `attention` showed knob axes only. `agenda.family_virgin()`
reads that table, so the new mechanisms did not stop DRY/STALE from closing their families
and did not raise the exploration gap (0.00 for both). This is **L028 recurring by a
different route**: mech_lib uses legal family names, but the name never reaches the table
rotation reads. A declaration no consumer reads is a comment.

**`winsched` — three defects.** (a) SUSPECTED, throughput: recompiles land inside the
charged clock; the schedule crosses `round(log2)` boundaries at progress ≈ 0.107/0.320/
0.533, each assigning a new list whose int contents are Dynamo guards on a `dynamic=False`
compile. The control shows ~215 s of startup, mostly compile, against a 300 s budget, so a
recompile is not a rounding error, and the declared diagnostic counts triggers rather than
pricing them. (b) CONFIRMED, silent mis-target: `_WS_FULL = window_sizes[0][0]` reads
pattern position 0, which under `LSSS` is `L` — so the ramp terminates by setting every
short layer to full context, strictly MORE attention FLOPs under a FLOP-reduction
mechanism, and `win` is an untouched top-priority axis. (c) CONFIRMED, L007-class no-op:
`{swdiv:16, winsched:128}` builds byte-different but `start == _WS_FULL`, so the schedule
can never move; it passes the identical-variant guard and burns a 300 s slot.

**CONFIRMED — `vefreeze` breaks the shared param-group telemetry by position.**
`make_variant` selects VE params as `_ag_ordered[2]`; vefreeze deletes that group, so
index 2 becomes `resid_lambdas`. The record prints `ve_table_count: 1` beside
`n_ve_layers: 4`, contradicting itself, and emits `ve_emb_rms_final` **twice** — the only
duplicated key across all nine variants. `host/dispatch.py:398` does `m[key] = value`, so
which number survives is decided by print order.

**CONFIRMED arithmetic — `embwd` has no bound and its natural values annihilate the
tables.** AdamW here is decoupled and the embedding/VE groups run at
`lr = 0.6·(512/768)^-0.5 = 0.7348`. Integrating over the schedule at ~700 steps: `0.01` →
×0.021 of initial norm, `0.05` → ×3.2e-9, `0.2` → ×3.1e-36. Muon's per-step factor for the
same written `0.2` is 18× gentler. `embwd=0.01` — the obvious small first value, and the
one in this repo's own test probe — already destroys 98% of the embedding norm. The arm
would read as a clean valid-negative rather than a mis-scaled knob.

**CONFIRMED — 5 of 8 declared diagnostics are compile-time constants and cannot fail.**
`periln_branches` = 2·n_layer; `vnorm_applied: 1`; `ffnpost_applied: 1`;
`winsched_start_window` and `rope_rotated_dims` are cfg echoes. Each is emitted from the
mechanism's own `obs` block, so it proves the print was appended, not that the forward
changed. They pass `emits_diagnostic()` and E3 while carrying zero information — the L014
defect the repo warns about in three separate comments. For periln/vnorm/ffnpost the real
observable is the branch-to-stream amplitude ratio their own docstrings name as mediator.

**CONFIRMED — `ngram_dim` is declared, accepted, and never read.** `build()` ignores it, so
`ngram_dim:1024` and `ngram_dim:4096` produce identical source; two queue entries collapse
to one content-addressed variant while the policy records two runs on a table-width
question never varied. Related: a cfg carrying only a companion param is mislabelled —
`is_platform({…,"winsched_frac":0.9})` returns **True** and `label()` returns `"control"`.
Only the byte-identical guard, the last line of defence, prevented pooling into the band.

**CONFIRMED — `ngram`'s docstring contradicts its generated code on FLOPs.** It argues the
lookup is excluded from `flops_per_token`, but `_ngram` never adds `ngram_table` to
`nparams_exclude` while it does land in `sum(p.numel())`. At `ngram=65536, n_embd=512`
that is +201.3 MFLOP/token on a 239.1 baseline: `flops_per_token_M` and `mfu_percent`
inflate **+84%** on the one mechanism whose claim is that it is FLOP-free.

**CONFIRMED — the one cross the docstring calls catastrophic is buildable.**
`build({…ropefrac:0.1, noqknorm:1})` succeeds although `_ropefrac`'s doc says it "must
never be crossed with `noqknorm`", citing perplexity 340,933. No lesson blocks the pair and
neither queue door checks pairwise incompatibility. Prose is not a guard. By contrast the
accidental collisions (`periln+ffnpost`, `vefreeze+embwd`, `ngram+vefreeze`) all raise
`VariantEditError` at build time — noisy failure, which is correct.

**SUSPECTED — `periln` amplifies the step-0 backward by 1/√eps.** The docstring reasons
only about the forward. For `y = x·rsqrt(mean(x²)+ε)` at exactly `x = 0`,
`∂y_i/∂x_j = δ_ij/√ε`, and both branches are exactly zero at init because `c_proj` is
zero-init. Depending on which default ATen picks that is ×11.3 to ×6.7e7 gradient
amplification into `c_proj` on the first step. Muon's Frobenius normalisation and AdamW's
scale-invariance probably absorb the update, and the gradient does not propagate further
back, but bf16 overflow to `inf` is live and `train.py:577` fast-fails only on the loss.
Could not be settled without a GPU. If `periln` launches, launch it first and check step 0.

**`prefetch` — the race is genuinely closed; I could not break it.** Per-thread CUDA stream
context means the generator's `gpu_buffer.copy_(…, non_blocking=True)` is enqueued on
`_pf_stream`; the `clone()` is stream-ordered after it; `_ev.record(); _ev.synchronize()`
provably retires both before `cpu_buffer` is repacked. Because that sync precedes `put()`,
`record_stream` is the correct and sufficient allocator hint. Data order and per-batch
`epoch` survive; no deadlock, since the error path `put(None)` unblocks a full queue.
Remaining risks are not correctness: the packing loop is pure Python holding the GIL, so a
thread can only interleave that work, not remove it; the daemon producer keeps packing
during `evaluate_bpb`; and the declared diagnostic `loader_frac` is emitted by the control
too, while the sharper `queue_starve_frac` is printed but not declared.

**`ropefrac`'s pairing is correct** — index algebra verified rather than assumed: pair `i`
is `(i, i+d)`, slicing `a1/a2/c` rotates pairs `0…r-1` with their control partners and
passes the tail through; output is bit-identical to control at `frac ≥ 1.0`, and
`r = max(1, min(d, int(d·frac)))` is properly bounded. **`vnorm` and `ffnpost` are
structurally sound** — `v = norm(v)` sits outside the `if ve is not None` block so it
applies to every layer, and reduces the last dim (per-head), which is what QKV-Norm means.

## pipeline_integrity (agent-pipeline-integrity-02)

**P0 — the council freeze is a total launch block, and `gate.py:174` says otherwise.**
`gate.py` prints "(already-queued work keeps launching; GPUs do not idle for prose)".
Simulating `next_batch()` against the live 256-entry host queue and the live cutoff:
`runnable=16 frozen=16 policy=0`, INTACT launchable waves: **0**. Every pending wave is a
width-2 yoked pair; the cutoff exempts the control and freezes the treatment, and the
`held` guard then correctly refuses to run a control without its treatment. The control
exemption is therefore **inert** — there is not one control-only wave in the queue — while
the freeze idles 100% of remaining work. With 4 free GPUs, `next_batch` would launch
NOTHING. This is precisely the v3 failure `host/dispatch.py:20-23` claims to have fixed by
replacing a launch block with a cutoff.

**P1 — the cutoff voids the round council's own output.** `gate.py:129-137` takes
`cutoff = min(ages)` over round *and* critique. The round met at 23:08Z and its proposals
were queued at 23:10Z — then frozen, because the *critique* artifact's mtime is 22:15Z. A
fresh round cannot authorise anything unless the critique happens to be fresh in the same
moment.

**Throughput.** 226 records, 204 valid. 88 completions in the last 12 h (79 ok, 9 invalid),
200 in 24 h, **0 in the last hour**; newest run ended 108 min ago. Union wall-clock with at
least one of our runs active over 24 h: **9.84 h of 24 (41%)**; 13.61 h (57%) sits in gaps
>10 min, the largest 352 min. Compute delivered: **26.34 GPU-h of the 96 that MAX_GPUS=4
allows (27%)**. Content of the last 12 h: 31 controls (35%) vs 57 treatments (65%), 10%
invalid — instrument overhead is reasonable; the loss is idleness, not bureaucracy-per-run.
Between 23:12Z and 01:14Z the pre-cutoff backlog drained beautifully — 32 runs in 122 min.
Then the backlog emptied and everything left was frozen.

**Both constraints bind right now.** `nvidia-smi` shows all 8 GPUs carrying foreign compute
apps, so `WAITING: 0 of 4 GPUs free` is honest. But capacity is the *temporary* blocker;
the freeze is the standing one — when a GPU frees, still nothing launches.

**Queue hygiene: clean.** 246 local vs 256 host reconciles exactly; the 10 host-only
entries all have results on disk and are retained by `launched()` in the merge. Zero
duplicate names, zero missing variant files, zero waves wider than MAX_GPUS=4, zero
orphaned wave_groups, zero policy-blocked entries.

**The mechanism doors are wide open.** All 8 new mechanisms were queued through
`queue_quad.py --width 2` in a scratch copy; every one produced 8 entries in 4 waves with
correct `mech:<name>` labels. For all 9 registered mechanisms `unknown_keys` is empty,
`is_platform` is False, `blocked_reason` is None, and `blocking_keys()` touches none of
them. The only thing between the mechanism library and a GPU is that nobody has queued
them — and once queued they inherit the freeze.

**Unlearned failures have no teeth.** `R7T17_P3_s0_ctrl` / `_s1_treat` were the tbs=17
two-factor arm, launched 01:06:43Z on gpu5/gpu6 and both killed at 01:14:51 with
`cotenant_detected: true`. The check is soft with `freezes=False`, so it never blocks a
launch and never moves the cutoff. Discharge is by name-matching only, so listing the name
in any lesson's `evidence` clears it. Of 83 active lessons only 5 carry `blocks_keys` and
5 `blocks_values` — 10 of 83 have an enforceable hook.

## reproducibility (agent-reproducibility-03)

**"CHAIN OF EVIDENCE: INTACT" is the weakest claim here, and it overstates what a stranger
can check.** On the `autoresearch_fresh` clone, which contains no `lit/` directory and zero
claims, `coe.py` prints INTACT with `0 problem(s)` on all five checks. Nothing distinguishes
"audited and clean" from "nothing to audit". On an `exp_1` clone the same command notes that
184 indexed sources are not fetched — that is *every* source cited by the 727 claims, since
`lit/sources/` is gitignored. So E1 verifies zero digests for a stranger and trusts
`index.json`; worse, `coe.py:117-140` `continue`s past an absent snapshot *before* the
locator check, making E1 strictly weaker on a clone than on the author's laptop. E4 is
similarly narrow: it `continue`s on any queue entry that already has a result, so it
inspects 32 pending of 246 entries — 214 executed experiments are exempt by design.

**Concrete provenance hole: 12 result records have no queue entry at all**, hence no variant
hash and no recoverable code: `R5NOQK2_P1/P2` (4), `R6MTP_P3`, `R7NOQK_P3`, `R7T17_P3` (6),
`W03a_3/_4` (2). Their P1/P2/P4 siblings are in `queue.json`, so the P3 rows were deleted
and `queue.json` is not append-only. These are live evidence feeding the registry and
`analyze.py`. E4 iterates the queue, not the results, so it cannot see this class of break.

**What genuinely does reproduce.** A stranger's clone rebuilds the numeric registry
byte-for-byte — `diff` of `coe.py registry` between live tree and clone is empty, `2626
citable values from 226 result record(s)` — and `analyze.py` reproduces the headline
exactly: `204 valid runs | best val_bpb 0.981213 (R7XF_P2_s1_treat)`, platform baseline
`0.984246 over 33 controls`, band `0.00143`. E5 is real and rebuildable.

**The `.gitignore` bug is genuinely fixed in exp_1**, tested behaviourally rather than by
reading: touching a new results/decisions file in both clones yields `??` rather than
silence, i.e. *future* records are visible to git.

**Pinning is real but the freeze is honour-system.** `baseline/prepare.py` and
`karpathy_pristine/prepare.py` hash identically and match `provenance.json`'s blob for
karpathy `228791f`; `uv.lock` pins `torch 2.9.1+cu128`. **But no test and no tool ever
compares the two files** — one grep for `karpathy_pristine` across `tools/ tests/ host/`
returns a single hit, a citation regex. `preflight.py` checks against `provenance.json`,
which is editable in the same commit as the file it describes. Editing both together would
pass every automated check.

**Tests.** Fresh clone: `73 passed, 11 skipped`, and all 11 skips name their need. exp_1
clone and live-tree copy: `79 passed, 5 skipped`. All six standalone scripts exit 0 on both
trees and leave the tree clean. `coe.py`, `direction.py`, `analyze.py` all exit 0.

**`mech_lib.py` is shipped and bound.** Reproducing `tick.sh`'s digest comparison: all five
policy modules MATCH the host, and `mech_lib.py` landed 10:56:10 while the running
dispatcher started 10:58:47 — so the live dispatcher imported the current copy.

**Newcomer papercuts.** `pyproject.toml` sets `addopts = "-q"`, so the documented
`pytest -q` becomes `-qq` and prints no pass/fail count at all. Plain `python3 -m pytest`
fails (3.14.2 has no pytest); nothing documents `uv sync` or the working `python3.12`
invocation. `.githooks/pre-commit` is tracked but `core.hooksPath` is local config not
carried by a clone, so the red-suite guard is inert for anyone else. And `direction.py`
hardcodes prior-campaign conclusions into the DIRECTION SPACE table (L019, L011, L015 —
"mfu ~42.6%", "value embeddings are 16.78M of 50.33M params"); these print on the *fresh*
clone, which has no lessons file. A tree whose CLAUDE.md says "you start with zero
experimental priors" ships five of them as string constants.

## synthesis (agent-synthesis-04)

**There are two shapes here, not one, and forcing them together loses the diagnosis.**

The first is **indirection by position or name into a collection somebody else is free to
mutate**. `make_variant` addressed the value-embedding group as `_ag_ordered[2]` while
`vefreeze` deletes that group; `mech_lib` read the full window as `window_sizes[0][0]`
while `win` is a live axis that reorders the pattern; `dispatch.runnable()` applied the
control exemption per ENTRY inside an all-or-nothing WAVE launcher;
`direction.MECHANISM_FAMILIES` was a hand-copy of a name that `@mechanism(family=...)`
already carried. The tell in every case is that the wrong answer stays plausible and
nothing raises: index 2 is still a param group, position 0 is still a window, a released
control is still a launchable entry. That is why these produced contradictory records
(`ve_table_count: 1` beside `n_ve_layers: 4`) rather than tracebacks.

The second is **a check whose enumeration domain excludes where the defect lives**. E4
iterates `queue.json` and `coe.py:364` `continue`s on any entry with a result, so 226
executed records and every queue-less result are structurally out of scope; E1
`continue`s past an unfetched snapshot before the locator check; `preflight.py` compares
`train.py` against `provenance.json`, a file inside the domain being audited; E3 asks
whether a diagnostic key was printed, never whether its value could have come out
otherwise; unlearned-failure discharge matches names, not configs. Each is sound over its
domain and the domain is drawn so the interesting cases fall outside.

Both offered framings are lossy. "A declaration no consumer reads" covers `family=`,
`ngram_dim` and the `gate.py:174` prose but not `_ag_ordered[2]` or the entry/wave
mismatch, where both sides are read. "A guard that cannot distinguish absence from
correctness" misdescribes its own examples: I ran `coe.py` against an empty tree and it
prints `CHAIN OF EVIDENCE: NOTHING TO AUDIT (empty corpus)`, so it distinguishes exactly
that; and E4's problem is domain choice, not vacuity. The split matters because the fix
record splits the same way. Commits `47a5d6c` and `0463d1a` cleared nearly every shape-one
instance and essentially no shape-two instance, and the reason is structural: shape one has
a local, testable fix, while shape two requires deciding what the guard is *for*, and a
guard that never fires feels fine.

**Verified state, not commit messages.** Fixed: `direction.py:673` now seeds from
`all_mechanisms()`; `direction.mechanism_families()` (`:626`) merges the registry and the
gaps really moved (ve_placement/attention 0.75, signal_path 0.60); `_WS_FULL` is now
`min(w[0] for w in window_sizes)` at `mech_lib.py:441`; VE params are selected by `id()`
(`make_variant.py:247-251`); `ngram_table` is added to `nparams_exclude`
(`make_variant.py:883`). I built each: `{ropefrac:0.1, noqknorm:1}` raises
`VariantEditError` (`mech_lib.py:488`), `embwd` is refused above `1.80e-03`, and
`{swdiv:16, winsched:128}` is refused as a no-op. `tests/test_frozen_contract.py` closes
the honour-system freeze halfway — it diffs `baseline/prepare.py` against
`karpathy_pristine/prepare.py` (both `06bea916`, matching `prepare_py_git_blob`) but a
coordinated two-file edit still passes.

Still open: P1 is untouched. `gate.py:129-137` still takes `min(ages)` over round *and*
critique, and it bites right now — the round is 1 min old, the critique 315 min old, so the
cutoff sits at 22:15:24Z. Simulating `runnable()`/`wave_sizes()` against the live 256-entry
queue: `runnable=0 frozen=32`, INTACT waves **0**. The P0 fix made the two rules agree
instead of stranding orphans, but delivered zero throughput; 135 min since the last
completion. The fix may also have missed an asymmetry: `wave_sizes` drops frozen members
from the denominator per entry, so a mixed wave whose control was created before the cutoff
and treatment after would size to 1 and launch solo. It does not fire today only because
`queue_from_round` stamps one `created_at` per wave.

**Ranked by cost to lower `val_bpb` in 300 charged seconds.** (1) The critique-driven
cutoff of P1, which right now freezes all 32 pending entries including the four
`R10NGRAM` treatments, and which no amount of round-council work can lift. (2) The
hard-check surface, which is more volatile than anyone has priced: at 03:30Z `coe.py`
reported E5 NUMERIC 1 against `rounds/2026-08-20T03-10-53Z_round.md` (24 unregistered
numbers, of which 0.000476 and 0.000952 are `analyze.py` derivations rather than
inventions), closing the gate entirely; by 03:34Z it had cleared. A *hard* gate that opens
and shuts within four minutes on prose freshness is a launch lottery, and E5 firing on
correctly-derived statistics is the second shape again — the check enumerates literals,
not derivations. (3) Capacity, which **probably dominates both**. `gpu_watch` shows
0 of 8 free right now, five foreign tenants at 80-130 GB; across 79 samples in 23.7 h,
4 GPUs were free in 11.4% and ≥2 in 45.6%. Integrating `min(4, free+ours)` gives **45.5
claimable GPU-h**, not the 96 the reviewer assumed; we delivered 25.4, so utilisation is
**56%**, and recoverable headroom is ~20 GPU-h, not ~70. The implication for priority is
concrete: governance fixes are free because they happen while waiting, so do them all, but
the highest-value change is to stop sizing waves at 4. A width-2 yoked pair is assemblable
four times as often as a 4-wide wave, and it buys the same within-wave nuisance control.
(4) The family fix made the mechanisms visible and simultaneously routed them behind the
rotation: ve_placement is HARD CAP 12/12, attention 24/12, signal_path STALE 10, so eight
of nine registered mechanisms sit in families `agenda.py` will never nominate, while the
active direction `signal_scale` has zero mechanisms. The cap should count knob runs, not
mechanism runs, in a family whose mechanisms are untouched. (5) `winsched` and `ropefrac`
still *declare* cfg echoes (`mech_lib.py:356`, `:458`) while printing the real observables
(`winsched_distinct`) undeclared — a one-line fix. (6) Orphan results are 2, not 12, but
only because `tick.sh` reconciled ten host-only entries into `queue.json` at 20:27:37;
that is a sync artifact, not a fix, and two `tick.sh --loop` processes are running (PIDs
32089, 67320), which is how a non-append-only `queue.json` happens in the first place.

**The claim I would check first is reviewer 2's, that "capacity is the *temporary*
blocker; the freeze is the standing one."** It sets everyone's priorities and its
supporting arithmetic uses a denominator the box never offered: `MAX_GPUS=4` allows 96
GPU-h only if four GPUs are free, and they were free 11% of the time. Inverted, the
conclusion is that governance is a real but second-order tax on ~45 claimable GPU-h, and
that pipeline engineering should target *value per claimed GPU-hour* rather than uptime.
Two other overconfident claims: reviewer 3's "nothing distinguishes audited-and-clean from
nothing-to-audit" is false of this tree's `coe.py` and was measured on a clone carrying
different code, which discounts the genuine findings bundled with it (E1's unfetched
sources, E4's scope); and `periln`'s `1/√ε` range is inflated at the top — `norm()` is
`F.rms_norm(x, (x.size(-1),))` with `eps=None`, so ε is `finfo(dtype).eps`, 0.0078125 in
bf16 for ×11.3, not the ×6.7e7 that presumes an fp64 ε this forward never sees, and
`c_proj` is a Muon matrix parameter whose update is Frobenius-normalised. The mechanism is
real; the headline number is not.
