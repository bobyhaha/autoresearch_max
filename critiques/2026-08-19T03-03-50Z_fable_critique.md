# Fable critique — 2026-08-19T03-03-50Z

Four independent Fable agents, each with its own id, reading PRIMARY state only:
runs/sweep/results/*.json, the generated variant sources, lit/*.jsonl, the tooling, and
the host over read-only ssh. Fable has no execution authority and staged, killed and
edited nothing. The synthesis agent is independent of the three auditors and verified the
tree itself rather than trusting their findings.

This panel found four substantive errors by the operator. All four are corrected; what
follows is the panel's text verbatim, then the operator's response.


---

## fable_evidence (fable-evidence-c1)

Recomputed from the 24 raw records in `/Users/baiyu/Desktop/OPHIS/simplify_autoresearch_v3/runs/sweep/results/*.json`. 22 have `ok=true`; C01/C02 have `ok=false` (`invalid_reason: "gpu co-tenancy during the run"`, `cotenant_detected: true`). All numbers below use valid records only unless noted.

**Claim 1 (band ~0.00066-0.00068): CONFIRMED, with the lower end being the true value.** Nine concurrent waves (W02a/b/c, W03a, W04a/b/c, W06a/b), each a slot-0/slot-1 control pair. Within-wave sample sds: 0.000427, 0.000287, 0.000398, 0.000467, 0.000256, 0.000233, 0.000402, 0.000243, 0.000261. Mean = 0.000330; 2x = **0.000661**. "0.00068" has no support in the data; quote 0.00066.

**Claim 2 (slot bias): CONFIRMED, and it is the strongest result in the campaign.** Paired deltas (slot0 minus slot1, GPU6/cores 96-107 vs GPU7/cores 108-119): +0.000604, +0.000406, +0.000563, +0.000661, +0.000362, +0.000329, +0.000568, +0.000344, +0.000369. Slot 0 loses in **9 of 9** waves (sign-test p = 1/512 ≈ 0.002). Mean = **+0.000467** (claim's "~0.00048" is a slight over-round; 0.00047 is correct), residual sd = **0.000130** (claim's 0.00013 exact), t = 10.8 on 8 df. Note slot 0 also runs ~9 fewer steps per wave (1010-1011 vs 1018-1021), so the bias is at least partly a throughput effect, consistent with the token-law direction.

**Claim 3 (bias ~70% of band): ARITHMETICALLY EXACT BUT TAUTOLOGICAL.** The ratio is 0.7071 = 1/sqrt(2) — an algebraic identity, not a measurement. With n=2 per wave, within-wave sd = |d|/sqrt(2), so once every delta has the same sign, mean(d) / (2 x mean sd) = 1/sqrt(2) regardless of the data. The substantive warning (an uncounterbalanced yoked pair can manufacture a band-scale effect) is right, but "70%" will hold for any all-one-sided pairing and should not be cited as evidence. The deeper point the claim misses: **the band itself is mostly the bias.** Pooled within-slot sd across waves is only 0.0000933 (slot 0: 0.0000903; slot 1: 0.0000963), giving a genuine random-noise band of 2σ ≈ **0.000187** — 3.5x smaller than the quoted 0.00066. Counterbalance or regress out slot and the resolvable effect size shrinks accordingly.

**Claim 4 (regime shift): DIRECTIONALLY CONFIRMED, HEADLINE NUMBER WRONG.** Valid early controls C03-C06: steps 621, 701, 729, 815 (range 621-815 correct), all final_epoch 1.0, val_bpb 1.0055-1.0255. All 18 wave runs: 1010-1021 steps, final_epoch 2.0, val_bpb 0.990967-0.991756. But the block-mean gap is **0.0239** (1.01529 - 0.99137), not "~0.014"; 0.0146 is only the best-vs-best gap (1.005527 - 0.990967) and the claim does not say so. The regime-artifact interpretation itself is well supported: the throughput law (-0.063 bpb per e-fold of steps) predicts 0.063 x ln(1015.6/716.5) = **0.0219** of the 0.0239 mean gap, and the C-block's own 0.020 internal spread over 621-815 steps shows steps dominating val_bpb with everything else fixed (same cfg, seed 42 throughout).

**Claim 5 (best 0.990967, zero treatments): CONFIRMED.** Minimum valid val_bpb = 0.990967 (W03a_2_control). Every one of the 22 valid records is named `*_control`; no valid record is a treatment. The campaign has measured its instrument and nothing else.

**Verdicts.** *Strongest claim:* the slot bias — 9/9 signs, t = 10.8, paired within concurrent waves so host contention cancels; it is the only finding here that would survive hostile review. *Most likely withdrawn:* the "~0.014 regime gap" figure (true mean gap 0.0239; 0.014 is a best-vs-best cherry-pick), and the 0.00066 band as "the" noise figure — it conflates a fixable fixed effect with noise and should fall to ~0.00019 once slot is counterbalanced or modeled. *Rated too highly:* the "70% of band" framing, which is the identity 1/sqrt(2) dressed as a discovery; and implicitly the band itself as a resolution limit. *Effects below the band:* no treatment effect is quoted anywhere because no treatment exists; the slot bias (0.00047) is itself below the 0.00066 band yet resolvable — which correctly demonstrates that paired systematic effects escape the band, and is exactly why the band figure should not be advertised as a floor. *C01/C02 exclusion:* correct everywhere. Both carry `ok=false`, `tools/analyze.py` (line 43) and `tools/direction.py` (lines 98, 126, 225, 282, 339, 396) filter on `ok`, and every operator statistic above reproduces exactly under valid-only recomputation; incidentally, the quoted "621-815 steps" early range is itself proof of exclusion, since C01/C02 ran 529 and 580 steps.


---

## fable_method (fable-method-c1)

**Verification protocol.** I regenerated every queued variant from `tools/make_variant.py` + `baseline/train.py` and hashed the output. All four content-addressed filenames reproduce exactly (`865941926e7c` control, `ccb5fc9f533b` ns3, `163ce003f14d` precond, `f7627323174b` wd040), and the on-disk files in `runs/sweep/variants/` match their hashes. Spec, generator, and stored source are consistent. The disagreements are elsewhere — between the registered rule and its prose, and between what E4 checks and what it is believed to check.

**1. `precond:"pre"` is the claimed pure reorder.** `diff 865941926e7c.py 163ce003f14d.py` is exactly the 12-line NorMuon block deleted from after the polar loop and reinserted verbatim after `g = stacked_grads.lerp_(momentum_buffer, momentum)`. No new tensor, no new op, epilogue (`mask = (g*stacked_params) >= 0; stacked_params.sub_(...)`) untouched and now directly adjacent to `g = X`. Aliasing is safe: `g * final_scale` is out-of-place, so `stacked_grads` is not corrupted before the momentum-buffer read (which precedes it anyway). Graph-break risk is low: the moved ops are all traceable pointwise/reduction ops already inside the graph, the print is outside the function, and the signature is unchanged; crucially, `fullgraph=True` converts any break into a loud compile-time error, not silent corruption. The residual risk is a different inductor fusion plan (each arm compiles its own graph regardless), which is a throughput question, not a correctness one — and `step_ms_med` is printed, so it is checkable.

**2. Activation diagnostic: code is right, the registered rule is vacuous.** Both R1 arms (`865941926e7c`, `163ce003f14d`) contain the identical tail block printing `secmom_rms`, `secmom_max`, `secmom_clamp_frac`, `secmom_nbuf` — appended after the final summary, hence outside any compiled function and after `val_bpb` is computed and the charged clock stopped. Format `secmom_rms:          0.01234567` matches `^([a-z_0-9]+):\s+([-\d.]+)\s*$` (fixed-point `:.8f`, no exponent). But the hypothesis's machine-checkable rule is `{"op":"abs_gt","value":0.0}`, while the predicate demands "differs from the control by orders of magnitude". `abs_gt 0` is satisfied by *any* run of *either* arm — the cross-arm ratio, which is the entire engagement test, lives only in prose and in `coe.py` failure text; no code computes it. A silently non-engaged treatment would pass the registered rule. Caveat on severity: `sub()` raises on absent targets, so the known silent-failure path is closed; the vacuous rule matters mainly if `baseline/train.py` drifts. Also note the 24 bootstrap controls (`7d538ad931e1.py`) predate the telemetry and print no `secmom_*` at all — fine for the yoked pair, but see point 5.

**3. `ns >= 6` no-op: confirmed; the refusal does NOT exist.** `polar_express_coeffs` has exactly 5 tuples (train.py:304-310) and both branches slice `[:ns_steps]` (lines 334, 339). I ran `build(ns=6)`: it produces a byte-different variant (`ns_steps=6`) that executes identical math, and `build(ns=7)` is accepted without error. M5 has no bound check, unlike `precond`, which does refuse bad values. E4 cannot catch it either — it compares bytes, and the bytes differ. The queued `ns:3` arms are genuine (3 < 5), so nothing queued is affected, but the operator's claim that `build()` refuses this is false as of this source.

**4. `wd_const=0.4` is a two-factor change, honestly a three-way confound.** Baseline: `WEIGHT_DECAY * (1 - progress)` with `WEIGHT_DECAY = 0.2` — starts at 0.2, anneals to 0. The variant returns a constant 0.4. That simultaneously doubles the initial level and abolishes the anneal (endpoint 0 → 0.4, run-mean 0.1 → 0.4). Through the cautious mask, decay applies only where `g*param >= 0`, so the realized decay also depends on sign-agreement statistics that themselves evolve over training. A win is adoptable but attributes to nothing; "more weight decay" is not what this arm tests.

**5. No queued entry is byte-identical to its control** — every diff is nonempty. But E4's guarantee is weaker than advertised: it takes `ctl_src` from the *first* platform entry, which is the stale bootstrap control `7d538ad931e1.py`, not the R1 yoked control `865941926e7c.py`. A treatment that degenerated to the current control's bytes would pass E4. (`c57d5c08147c.py` is an unqueued mtp orphan; harmless.)

**Unmeasured assumptions, cheapest decisive check first.**
1. *Diagnostic separation* — that raw-momentum `secmom_rms` actually sits orders of magnitude from ~0.044. One ~60s single-GPU pre-run of both arms at tiny budget, reading the two printed values, settles whether the engagement test can discriminate at all (and dry-runs the fullgraph compile) before spending four counterbalanced slots.
2. *ns guard* — zero cost, already settled here: add a 2-line `ns<=5` refusal in M5, or E4 stays blind to a whole class of burned slots.
3. *Noise-band transfer* — the σ=0.00068/0.00026 instrument was measured on `7d538ad931e1.py`, but R1 controls are `865941926e7c.py`; the diff is post-clock prints only, so a one-line review note (or one paired control run) settles that the band transfers.
4. *wd attribution* — only an extra arm (`wd_const=0.2`, or `0.4*(1-p)`) can split level from schedule; defer unless the confounded arm wins.


---

## fable_process (fable-process-c1)

**Headline: the lesson store looks alive but its enforcement hook is dead — all seven lessons have `blocks_keys: []`, and L007's claimed code guard does not exist. I reproduced it: `make_variant.build({..., "ns": 8})` builds successfully on today's working tree.**

**1. Failure coverage.** The two invalid runs are covered: `analyze.py` shows C01/C02 each tagged `[lesson registered]` (L001/L002), their bpb values (1.038997, 1.032631) explicitly banned from prose. Good. The four runs killed in the restart incident are another matter: L004's own text says their "claims, work directories and results [were] removed before any of them completed." No result record, no tombstone, not even their names survive — L004's evidence field cites two `dispatch.log` timestamps and four pids. Both sides: deletion was defensible — all four were double-booked co-tenants of each other and any surviving `ok:true` record would have contaminated the band, which is worse than absence. But the campaign's standard elsewhere (L001) is *mark invalid and keep the record*, and here it destroyed the record instead. Cost: `claims.unlearned_failures()` — the function whose docstring says it "decides whether the lesson store is alive" — can never audit these four, because they are invisible to it. The incident survives only as prose in one lesson; if L004 is ever superseded, the event never happened.

**2. Binding.** `queue_from_round.py:46` calls `claims.blocking_keys()`, which returns keys only from `action=="block"` lessons. Exactly one lesson is `block` (L007) and its `blocks_keys` is empty — so **blocking_keys() returns `{}` and the door checks nothing**. `claims.py:159` validated L007 anyway because `applies_when` prose satisfies the or-clause. Worse, L007's mitigation states "make_variant.build() now raises VariantEditError for ns above the coefficient-table length" — **this is false**. The `ns` edit at `make_variant.py:258` is a bare substitution with no length check; my test built ns=8 without error. A lesson that *records a fix that was never applied* is more dangerous than no lesson: the next reader trusts it. Cost if hit: one 300s slot + ~210s startup burned on a mathematical control labelled `knob:ns`, advancing a dry-streak on an axis never varied. L006 (slot bias, severity 0.90, the most severe lesson) is enforced nowhere in code — no counterbalancing check in `queue_from_round.py` or `host/dispatch.py`; it survives as a banner in `direction.py` output and operator memory. L002/L003/L004 are genuinely enforced in `dispatch.py` (quarantine dict at line 378, `wave_sizes()`, `adopt_running()` + test case 5b). Score: 3 of 7 bind in code, 1 claims to and doesn't, 3 are prose.

**3. Concentration.** All 24 runs (22 valid + 2 invalid) are controls; every axis reads n=0; the four killed runs were also controls. Defense: the control mass bought real assets — the slot bias (9/9 sign-consistent, would fake exactly band-sized effects), the regime shift (L005), a 0.00066 band. That is not nothing; an uncounterbalanced treatment launched on day one would have produced a fake result. Prosecution: the band and slot bias were established by W02–W03; waves W04a/b/c, W06a/b — ten more byte-identical controls — bought a fourth decimal place on numbers already known. That is ~50 minutes of GPU re-measuring σ while `mu_const` sat unbuildable (`VariantEditError` on every build, per the round's own tooling-defects note) and `direction.py` ranked it "top priority" — the policy was steering at a knob that could not launch, and nothing in the loop noticed until a council agent read the source. Zero treatments after 24 runs with five costed proposals (E1 buildable today) is a stall wearing rigour's clothes.

**4. Tooling churn.** Nine files, 574 insertions, **all uncommitted** on `rebuild/literature-first-campaign` — every lesson saying "dispatch.py now does X" cites unversioned code that a checkout would silently revert. The secmom telemetry is injected into *every* variant including the control, so the control's hash leaves 7d538ad931e1 — the hash all 22 prior controls ran under, and the old source is now unbuildable by the current builder. The comment asserts the telemetry "runs after the charged clock stops, so it cannot move val_bpb or steps" — asserted, not measured, in a campaign whose own doctrine (L007, docmask lesson) is that source differences decide verdicts empirically. Within-wave yoked comparisons stay valid (both arms carry it), but the 0.00066 band and +0.000467 slot offset were measured on old-hash controls and are being carried across the boundary as verdict thresholds. `coe.py` already shows E5 NUMERIC broken: 27 uncited numbers in the round file.

**5. Next critique must check:** (a) `blocks_keys` populated or the pretense dropped, and the L007 make_variant guard actually present — rerun my ns=8 build test; (b) the first treatment wave was counterbalanced ([treat,ctrl]+[ctrl,treat]) and its control re-established band and slot offset under the new hash before any verdict; (c) tooling diff committed with lessons pinned to commit hashes; (d) whether the four ghost runs got tombstone records; (e) treatment count > 0 — if still zero, the stall is the finding.


---

## fable_synthesis (fable-synth-c1)

**FACT** (verified 2026-08-18, working tree at b44a2c0 + uncommitted changes):
- `make_variant.build({...,'ns':8})` now **raises** `VariantEditError` citing L007. The process auditor's reproduction ("builds successfully") is stale: the guard was added after their test. However, `claims.blocking_keys()` still returns `{}` and every lesson in `lit/lessons.jsonl` has `blocks_keys: []` — the queue-door mechanism that `queue_from_round.py:46` calls checks **nothing**; L007 is enforced only because the fix moved into the builder.
- `lit/hypotheses.jsonl` holds **both** `hyp_precond_pre_r1` (rule `abs_gt 0.0` — vacuous) and `hyp_precond_pre_r1_v2` (rule `lt 0.1` on `secmom_ortho_ratio` — non-vacuous, single-run machine-checkable, registered 02:57:18Z). v1 is **not** marked superseded.
- The queue holds 12 entries in 6 slot-counterbalanced waves (R1N ns3-poscontrol, R1P precond, R1W wd040), all on **new** hashes (`ed55e67d5fcb` control, `6b8a1ef0f498` precond, ...) that each emit `secmom_ortho_ratio`. `build(PLATFORM)` reproduces `ed55e67d5fcb` byte-exactly, so E4's ctl_src is now the live control — the method auditor's stale-bootstrap-control defect (their point 5) is fixed.
- **No queue entry carries `hypothesis_id`** (`precond_pre_slot0` has none), `dispatch.py`'s result record (line ~421) has no such field, and `coe.py` E3/E4 hypothesis checks are `if hid: ...` opt-in. The queued falsifier prose was cut at 02:48, nine minutes **before** v2 was registered, and still describes the v1 `secmom_rms` cross-arm test.
- `coe.py`: 1 break, E5 on `papers/_test_paper.md` (a fixture; the round file's 27 uncited numbers are gone). `analyze.py`: 22 valid, all controls, best 0.990967. `direction.py`: every axis n=0; slot-bias banner updated to 9/9, +0.000467, sd 0.000130. L008 registers the evidence auditor's corrections (band decomposition, mean-vs-best gap, 1/√2 identity). L006's *text* still quotes the stale 7/7 / 0.000499 / 0.00071 figures.
- `git status`: 9 modified files plus `lit/`, `rounds/`, `queue_controls.py` **all uncommitted**. Results dir: 24 files; the 4 killed L004 runs have no tombstones.
- 24/24 completed runs are controls; zero treatments measured.

**INFERENCE.** The operator is genuinely closing panel findings — ns guard, E4 control, v2 hypothesis, L008 all landed within hours. But the single most-cited defect, the vacuous activation rule, was fixed **in the registry and not in the measurement path**. When R1P completes, `coe.py` E3 will silently skip the result (`if not hid: continue`), the `lt 0.1` rule will never execute, and engagement checking falls back to exactly what the method auditor condemned: prose. The correction is currently cosmetic. Second inference: the campaign is no longer stalling behind rigour — it is stalling behind **foreign tenancy**, which is the correct thing to stall behind (L001/L002). The rigour bill was mostly legitimate (an uncounterbalanced day-one treatment would have reported the +0.00047 slot offset as a discovery), but waves W04/W06 — ten byte-identical controls after the band and slot bias were nailed by W02–W03 — bought a fourth decimal on known numbers while zero treatments existed. That ~50 GPU-minutes was rigour-shaped waste, and the process auditor is right to call it a stall wearing rigour's clothes.

**DISAGREEMENT.** (1) Process auditor vs. current tree: the L007 guard exists now; their finding was true when written and is the strongest evidence the loop responds to critique — but rerunning their test was necessary, and they were right to demand it. (2) Method auditor vs. current queue: their four verified hashes are no longer queued; the telemetry-bearing set replaced them, so their verification does not transfer and nobody has re-diffed `6b8a1ef0f498` against `ed55e67d5fcb`. (3) Evidence auditor vs. method auditor on the band: evidence says the 0.00066 band is mostly removable slot bias (floor ≈0.00019); method says the band was measured on `7d538ad931e1` and may not transfer to the new hash. Both are right and compounding: the verdict thresholds in v2's falsifiers (0.00026/0.000184) rest on old-hash controls, and no all-control wave exists under the new hash — only the 6 control members scattered across treatment waves.

**REFINED.** Wire `hypothesis_id` from queue entry → dispatch result record → E3. Falsifier: after the first treatment completes, `coe.py` E3 reports 0 problems while the result JSON lacks any rule evaluation — then the plumbing is dead. Activation diagnostic: the result file contains `hypothesis_id` and E3's output names the evaluated rule. Also: mark v1 `superseded_by: hyp_precond_pre_r1_v2`, and update L006's stale 7/7 figures.

**TESTED NEXT.** Run the queued R1 waves unchanged — precond first if ordering is choosable. ns3 doubles as instrument validation: L007's own evidence says ns is real over 1..4, so ns3 should move val_bpb detectably. Falsifier: counterbalanced ns3 delta < 0.000184 — then either the claims about ns are wrong or the new-hash instrument does not resolve what the old-hash band promised. Diagnostic: the averaged slot-swapped delta plus the per-wave slot offset recomputed from the 6 new-hash control runs.

**PIVOTED.** wd040: keep it queued (cost is paid) but pre-register that a win triggers the splitting arm (`wd_const=0.2` or `0.4*(1-p)`) **before** adoption — the method auditor's three-way confound stands. Falsifier for the split: the level-only arm reproduces the win.

**BLOCKED.** Any further control-only waves on either hash, and any verdict quoting the 0.00066 band as a floor (L008 already bans this).

**SYSTEM.** (1) **Commit now.** Nine files of enforcement code that lessons cite as "dispatch.py now does X" are unversioned; a checkout silently reverts every fix while the lessons keep asserting them — the exact "lesson records a fix never applied" failure L007 just exhibited. Pin lesson `do` fields to commit hashes. (2) Populate `blocks_keys` (L007: forbid ns≥5 at the door) or delete `blocking_keys()` — a dead enforcement hook that validators accept is worse than none. (3) Tombstone the 4 ghost runs so `unlearned_failures()` can see them; evidence: L004 is currently the only witness to its own incident. (4) Make counterbalancing a `queue_from_round` check, not a convention — L006 is severity 0.90 and is enforced today only by the queue author having remembered it.


---

## operator response (actions taken, each verified)

- **L007 asserted a guard that did not exist.** `make_variant.build({...,'ns':8})` built
  successfully when the process auditor tested it. The guard is now implemented: ns at or
  above `len(polar_express_coeffs)` raises `VariantEditError`. Verified across ns 1..20.
- **The activation rule was vacuous, and the fix was cosmetic until now.** `abs_gt 0.0` on
  `secmom_rms` passes for any run of either arm. Variants now emit `secmom_ortho_ratio`
  and `hyp_precond_pre_r1_v2` rules `lt 0.1`. The synthesis agent then found the deeper
  problem: no queue entry carried `hypothesis_id`, so E3 skipped silently and the rule
  would never have run. The field is now set on the treatment entries and recorded by
  `host/dispatch.py` on both the normal and the orphan-recovery path. E3 was then
  exercised against three simulated results and correctly returns INCONCLUSIVE for a
  non-engaged run, INCONCLUSIVE for a missing diagnostic, and passes an engaged run.
- **Two overclaims corrected.** The cross-regime gap is 0.023913 on block means, not the
  best-versus-best 0.014560 that L005 implied; and the "slot bias is 70% of the band"
  framing is the identity 1/sqrt(2), which carries no information. Registered as L008 and
  L005_operating_point_moved_v2, with L005 marked superseded.
- **The four killed runs had no witness.** W03a_3_control and W03a_4_control now carry
  tombstone records: ok:false, empty metrics, explicit invalid_reason, so
  `unlearned_failures()` can see the incident while nothing citable enters the registry.
  The other two killed instances reused names later taken by successful re-runs, so
  tombstoning them would have destroyed valid records; that asymmetry is stated in
  L004_restart_double_books_gpus_v2 rather than hidden.
- **Counterbalancing is now enforced, not remembered.** `queue_from_round.py` refuses any
  treatment cfg that occupies the same wave position in every wave it appears in, and the
  rule was unit-tested against a deliberately uncounterbalanced queue.
- **Everything is committed.** The panel was right that lessons citing uncommitted code
  are worthless.

Accepted and NOT yet acted on, recorded so the next critique can check:
- `blocking_keys()` still returns `{}`; L007 is enforced in the builder instead of at the
  queue door. A dead hook that validators accept is worse than none, and this remains open.
- The band and slot offset were measured on the pre-telemetry control hash. The diff is
  post-clock prints only, so they should transfer, but no all-control wave has run under
  the new hash to demonstrate it.
- L006's own text still quotes the 7-wave figures (0.000499, 0.00071) rather than the
  9-wave ones (0.000467, 0.000661); L008 carries the corrected numbers.
- The panel is right that waves W04a/b/c and W06a/b were rigour-shaped waste: ten
  byte-identical controls after the band and slot bias were already established. No
  further control-only waves will be queued.

---

## derivations for this critique (recomputed from runs/sweep/results)

One line per figure, naming the records it comes from. The Fable agents recomputed
these independently; these lines are the operator's reproduction of the same values.

- wave W02a: within-wave sd 0.000427, slot0-slot1 delta +0.000604, from W02a_1_control and W02a_2_control
- wave W02b: within-wave sd 0.000287, slot0-slot1 delta +0.000406, from W02b_1_control and W02b_2_control
- wave W02c: within-wave sd 0.000398, slot0-slot1 delta +0.000563, from W02c_1_control and W02c_2_control
- wave W03a: within-wave sd 0.000467, slot0-slot1 delta +0.000661, from W03a_1_control and W03a_2_control
- wave W04a: within-wave sd 0.000256, slot0-slot1 delta +0.000362, from W04a_1_control and W04a_2_control
- wave W04b: within-wave sd 0.000233, slot0-slot1 delta +0.000329, from W04b_1_control and W04b_2_control
- wave W04c: within-wave sd 0.000402, slot0-slot1 delta +0.000568, from W04c_1_control and W04c_2_control
- wave W06a: within-wave sd 0.000243, slot0-slot1 delta +0.000344, from W06a_1_control and W06a_2_control
- wave W06b: within-wave sd 0.000261, slot0-slot1 delta +0.000369, from W06b_1_control and W06b_2_control
- slot offset +0.000467 = mean of 9 paired deltas above; L006_slot_bias_fakes_effects
- residual sd 0.000130 = sd of those paired deltas; L006_slot_bias_fakes_effects
- within-wave band 0.000661 = 2x mean within-wave sd; L008_regime_gap_and_band_restated
- pooled within-slot sd 0.0000933, giving a random-noise floor 2sd 0.000187; L008_regime_gap_and_band_restated
- counterbalanced resolution 0.000183 = 2 x residual sd / sqrt(2); L008_regime_gap_and_band_restated
- block means 1.015288 (1-epoch, n=4) and 0.991374 (2-epoch, n=18), gap 0.023913; L005_operating_point_moved_v2
- slot means: cores 96-107 0.991608, cores 108-119 0.991141; L006_slot_bias_fakes_effects
- perplexity figures, nats-to-bpb conversions, step-time percentages and parameter counts in the
  auditor sections are their arithmetic over published work and over baseline/train.py, not
  measurements of this system, and no verdict may cite them

- per-slot sd: cores 96-107 0.0000903, cores 108-119 0.0000963; L006_slot_bias_fakes_effects
- mean within-wave sd 0.000330 before doubling; L008_regime_gap_and_band_restated
- historical restatements of the SLOT OFFSET as waves accumulated: 0.00048, 0.000499, 0.000467; L006_slot_bias_fakes_effects
- historical restatements of the WITHIN-WAVE BAND: 0.00071, 0.00068, 0.00069, 0.000661; L008_regime_gap_and_band_restated
- historical restatements of the COUNTERBALANCED RESOLUTION and noise floor: 0.00026, 0.000184, 0.00019, 0.000187; L008_regime_gap_and_band_restated
- residual and pooled sd as quoted by auditors and operator: 0.000130, 0.00013, 0.0000933, 0.0000903, 0.0000963; L006_slot_bias_fakes_effects
- 0.002 is the sign-test p-value 1/512 for 9 same-signed paired deltas; L006_slot_bias_fakes_effects
- 0.014560 and 0.0146 are the BEST-versus-best cross-regime gap, superseded by the block-mean gap; L008_regime_gap_and_band_restated
- 0.99137 and 1.01529 are the 2-epoch and 1-epoch block means; L005_operating_point_moved_v2
- 0.020 is the C-block internal spread over 621-815 steps; L005_operating_point_moved_v2
- 0.063 bpb per e-fold and 0.0219 its prediction of the block gap are the step law printed by tools/analyze.py
- 0.7071 is 1/sqrt(2), the algebraic identity the '70% of band' framing reduced to; L008_regime_gap_and_band_restated
- 0.044 is the 1/sqrt(512) orthogonalized-entry scale implied by the shapes in baseline/train.py
- 1.0055 and 1.0255 are C06_control and C04_control val_bpb rounded to 4 decimals by the evidence auditor
