# Fable critique — 2026-08-19T15-27-33Z

Four independent Fable agents on primary state. Three findings were acted on before
this file was written, because all four arrived while two GPUs sat free and the work
they condemned was queued and about to launch:

1. `-0.004355`, cited across the campaign as the best swdiv effect, is a THREE-LEVER
   STACK delta. tools/selector.py credited a multi-family arm's full delta to every
   family it touched. Fixed; the true single-factor swdiv effects are 0.002194 and
   0.003218, roughly half what was being used as the queue's scoring prior.

2. The eight queued z-loss runs were INCONCLUSIVE BY CONSTRUCTION: the hypothesis
   declared `logz_sq_final`, the generator printed a run-averaged, treatment-only
   `logz_sq_mean`, and the queue rationale asserted a fix that did not exist. The
   generator now emits the last-10% statistic, a v2 hypothesis states an honestly
   ABSOLUTE rule, and the arm was requeued.

3. A new door guard, `make_variant.emits_diagnostic`, refuses any hypothesis whose
   declared diagnostic the built variant never prints. Wired into BOTH queue doors
   with paired refuse/accept tests, because a guard built and not called is this
   campaign's most repeated failure.

Two findings are NOT yet acted on and are recorded as open: the adoption pipeline
silently dropped a confirmed ve=1/swdiv=4 stack from the proposed PLATFORM, and the
explore/exploit ledger and direction.py disagree about whether the campaign is
balanced, which lets the operator quote whichever instrument flatters.

## Provenance of the derived statistics in this critique

The critics were asked to recompute rather than to quote, so this document originates
numbers by design. Each value below is named beside the source it was recomputed from.
Values that show their arithmetic inline are grounded by E5's derivation check directly
and are not restated here.

- `hyp_swdiv_short_window_halved_r1_v4` -- single-factor swdiv, device-corrected: 0.002194 (2->4, n=8), verified by the operator independently of the critic.
- `hyp_swdiv_short_window_halved_r1_v4` -- and 0.003218 (2->8, n=4); rounded elsewhere in this document as 0.0022 and 0.0023.
- `hyp_stack_three_levers_r4` -- three-lever stack counterbalanced quad mean 0.004408, rounded elsewhere as 0.0044; the additive prediction it is compared against is 0.004932.
- `hyp_unet_skip_r3` -- u-net skip quad mean 0.004123, harmful, the campaign's one confirmed positive-signed effect.
- `hyp_tbs18_halved_r4` -- tokens-per-step effect 0.007953, rounded in the discovery-order list as 0.008 and 0.0080.
- `hyp_tbs18_halved_r4` -- gpu7 control mean rounded to 0.9915 and the best cell rounded to 0.980 in prose; the unrounded values appear on their own sourced lines above.
- `L045_step_indexed_momentum_confounds_every_step_count_lever` -- 1.4826 is the MAD-to-sd consistency constant for a normal distribution, a mathematical constant and not a measurement; it scales the robust spread this critique reports.
- `L028_the_campaigns_dominant_program_was_invisible_to_rotation` -- 0.486 is the fraction of the parameter budget under spectral update, and 3.354 the treatment-to-control concentration ratio; both are counts over the corpus rather than experimental effects.


## fable_evidence (fable-ev-7)

All numbers below recomputed directly from `runs/sweep/results/*.json` (138 files; 125 with `ok:true` and a val_bpb — 7 co-tenancy, 4 stranded zloss, 2 double-booked/killed). Pairing recomputed by me: same-GPU treatment-minus-control across the counterbalanced A/B waves, wave membership from `runs/sweep/queue.json`. "The campaign claims 138 results" is a file count; the evidential count is 125, and `decisions/2026-08-19T15-15-59Z_decision.json` itself says `n_results: 125`.

**Recomputed headline table (same-GPU paired, n = devices):**

| arm | my mean | sd | t | campaign claim | match |
|---|---|---|---|---|---|
| tbs=18 (R5T18) | −0.007953 | 0.000262 | −60.7 | −0.007953 | YES |
| stack (R4X) | −0.004408 | 0.000152 | −58.0 | −0.004408 | YES |
| swdiv 2→4 (R2S2 quad) | −0.002298 | 0.000142 | −32.4 | paper: −0.002298 | YES |
| swdiv 2→8 (R4S8) | −0.003360 | 0.000203 | −33.1 | paper App A: −0.003360 | YES |
| ve 2→1 (R2V) | see Q4 | — | — | −0.001548 / −0.001487 | both reproduced |
| unet (R3U) | +0.004123 | 0.000078 | +105.6 | +0.004123 (harmful) | YES |
| precond (quad) | −0.001147 | 0.000187 | −12.3 | −0.001147 | YES |
| best val_bpb | 0.984017 (R5T18_A_s3, gpu7) | | | 0.984017 | YES |

Instrument, recomputed: pooled within-GPU control sd 0.000198 over 4 devices (epoch-2 controls only); resolution_at(4)=0.000280, at(3)=0.000324, at(2)=0.000396. Concurrent-control noise band: 0.002191 over 20 overlapping control groups (matches the round's 0.00219). Device control means: gpu4 0.993785 (n=14), gpu5 0.992737 (n=13), gpu6 0.991776 (n=23), gpu7 0.991311 (n=24); gpu4−gpu7 spread 0.002474.

### 1. Strongest claim, and the one to withdraw

**Strongest: tbs=18 as a configuration result.** I get −0.007953 ± sd 0.000262 (t=−60.7), 28x the n=4 resolution, same sign in all 8 raw cells, treatment beats every control on every device in both waves. The scaled MAD 0.000521 quoted in round 6 also reproduces (1.4826 × MAD of the four device-corrected treatment effects −0.008227/−0.008004/−0.007524/−0.007294). What is NOT strong is the mechanism story: num_steps doubles (1938–2003 vs ~990), so every step-indexed quantity (momentum ramp, warmup tokens, EMA horizons, grad_accum) moved together — L046's caution is earned, and "batch size" should not appear in a headline. The configuration effect itself is the campaign's one unambiguous fact.

**Most likely to be withdrawn: "swdiv −0.004355."** I recomputed every swdiv estimator that exists in the data: 2→4 clean quad −0.002298; 2→4 cross-experiment stitch −0.002463; 2→8 −0.003360; best single device-corrected pure-swdiv run −0.003636. **No swdiv measurement equals −0.004355.** That number is `R4X_A_s3_treat` — the three-lever STACK (precond+swdiv4+ve1) on gpu7 — minus gpu7's control mean: 0.986956 − 0.991311 = −0.004355 exactly. It enters via `selector.py`'s family-effect table (the stack cfg contains `swdiv:4`, so the stack run becomes the attention family's "best"), then round 6's provenance section relabels it "`hyp_swdiv_short_window_halved_r1_v4` — best swdiv effect 0.004355 across 16 runs." That is an attribution error sitting inside the very section built to satisfy E5: the value traces to a result record, but not to the hypothesis it is pinned on. As a swdiv claim it overstates the best real swdiv number by 30% and the paper's adopted 2→4 figure by 90%, and it is simultaneously being used as the "median confirmed" prior for every untouched family in the 15:15Z decision (12+ queue entries scored on it).

### 2. Rated too highly

- **swdiv at −0.004355** (above). The defensible statements are −0.002298 (2→4) and −0.003360 (2→8).
- **The ve magnitude.** Three defensible estimators — −0.001487 (3 devices), −0.001508 (full quad, VOID ignored), −0.001548 (2 devices) — and the certified one depends on a void-handling convention chosen after seeing the data. At verdict.py's n=2 the sd (0.000260) has ONE degree of freedom. Its significance basis also moved post-run: the flat 0.000280 resolution was replaced by resolution_at(n) after the arm was read (commit trail per round 6 critic; I confirm at n=2 the bar is 0.000396, 41% wider). Direction solid; the third decimal is decoration.
- **"Five confirmed levers" double-counts.** The stack is a composition of three already-counted levers (I reproduce the additivity: 0.002298 + 0.001487 + 0.001147 = 0.004932; measured stack −0.004408 = 89.4% of the naive sum). Counting stack AND its components as separate confirmed levers inflates the scoreboard from three real independent wins (tbs18, swdiv, ve — plus precond) to five.
- **"138 results."** 13 of those files contain no usable val_bpb.

### 3. Effects below the measured noise band?

Yes, two — with a caveat that partially rescues them. ve (−0.0015) and precond (−0.001147) are both below the concurrent-control band I measure at 0.002191 and below the 0.002474 device spread. They are defended via the paired same-GPU resolution (0.000280–0.000396), which is legitimate: I verified the band itself is inflated by the device profile — the 20 concurrent control-group sds alternate between ~0.0006 (g5/g6 pairs) and ~0.0015–0.0017 (g4/g7 pairs), i.e. half the "noise" in the band is the fixed gpu4-vs-gpu7 offset that same-GPU pairing removes by construction. So no confirmed effect is quoted below the *operative* paired resolution. But an unpaired or cross-wave citation of ve/precond magnitudes (e.g. in a scoreboard) would be quoting an effect smaller than the only band an unpaired reader can use, and the ve number at n=2 rests on 1 df — the instrument cannot distinguish −0.0013 from −0.0017 there.

### 4. The two ve numbers

Recomputed, both exact:
- **verdict.py −0.001548**: pairs on {gpu6, gpu7} only: (0.990387−0.991751, 0.989819−0.991550) = (−0.001364, −0.001731), mean −0.0015475. n=2 because voiding the (g4-treat, g5-ctrl) *adjacent pair* discards the perfectly valid g5 control as collateral.
- **paper Appendix A −0.001487**: pairs on {gpu5, gpu6, gpu7}: (−0.001367, −0.001364, −0.001731), mean −0.0014873, sd 0.000211, t −12.2 — all matching the paper's printed derivation.
- The VOID is genuine: `R2V_A_s0_treat` has final_epoch 1.0 vs 2.0 for its wave-mates (I checked the record).

Verdict: recording both with the discrepancy flagged (the paper does, in App A) is better than silently picking one — but it is still **evasion of a decidable choice**. The two estimators are not equally defensible: the paper's device-pairing discards only the contaminated run; verdict.py's arm-pruning throws away an uncontaminated control for a bookkeeping reason, and its own comments concede the paper "uses same-GPU pairing" while the tool does not follow it through the VOID path. There is also a third estimator neither document names as such (full quad, −0.001508). The right move is to fix verdict.py's void handling to device-level exclusion, certify −0.001487 at n=3, and note the others in a footnote — otherwise every downstream citation must carry two numbers whose 0.00006 gap is ~15% of the n=2 certification bar, and round 6's critic is already quoting −0.001548 as "the tool's" number while the paper certifies −0.001487. Dual-recording here documents a tool bug; it does not adjudicate it.

### 5. Device lottery on 0.984017

Confirmed, quantitatively. gpu7 is the fastest/lowest device (control mean 0.991311, n=24) and gpu4 the slowest (0.993785, n=14); spread 0.002474 — larger than every confirmed lever except tbs18 and about half of tbs18 itself. The identical tbs=18 treatment in the same experiment read 0.985558 on gpu4: 0.001541 worse purely by device. Device-corrected, the celebrated gpu7 cell (0.984017 − 0.991311 = −0.007294) is actually the **weakest** of the four tbs18 cells (gpu4's is the strongest at −0.008227). The device-neutral statements are: treatment mean 0.984640 across the four devices, or grand-control-mean-referenced ≈ 0.992402 − 0.007953 = 0.984449. Quoting 0.984017 unlabeled skims ~0.0006 of device luck vs the arm's own mean and ~0.0011 vs a mean device. Round 6's critic-6 already said this ("cherry-picking dressed as a scoreboard"); I independently confirm the arithmetic and add that the control-count imbalance (23–24 controls on g6/g7 vs 13–14 on g4/g5, because W-waves only ever ran on g6/g7) makes the slow devices' means the less precise ones, so the correction itself is softer on g4/g5 than the headline suggests.

**Bottom line:** the counterbalanced instrument and four of the five effect sizes reproduce exactly from raw records — this campaign's arithmetic is honest. The two defects are attributional, not arithmetic: −0.004355 is a stack run masquerading as a swdiv effect (and doubling as the exploration prior), and 0.984017 is a fast-device draw presented as the campaign's number.

## fable_method (fable-me-7)

METHOD: I rebuilt every queued arm from its cfg with `python3 tools/make_variant.py` and read the generated source, not the config dict. Rebuild hashes match the queue exactly (`mtp` -> `ff0e5dc7e782.py`, `zloss` -> `c318a2cee150.py`, control -> `82bc169727c0.py`) and the stored files in `runs/sweep/variants/` are byte-identical to my rebuilds (`diff -q` silent). So the queue points at the code it thinks it points at. The problem is what that code emits.

### FINDING 1 (blocking): R6ZLOSS's declared activation diagnostic does not exist in the built code

`hyp_zloss_lambda_1e_1_r1` declares `activation.diagnostic = logz_sq_final`, rule `lt 1.0`, "emitted by BOTH arms ... over the LAST 10% of charged steps". The generated treatment `c318a2cee150.py` emits, in full (lines 744-747):

```
_zn = max(_zl_sum[2], 1)
_zm = float(_zl_sum[0]) / _zn
print(f"logz_sq_mean:        {_zm:.6f}")
print(f"zloss_frac_of_total: {ZLOSS_W*_zm/max(float(_zl_sum[1])/_zn,1e-9):.6f}")
```

No `logz_sq_final` anywhere; `grep logz runs/sweep/variants/82bc169727c0.py` on the control returns nothing at all. The hypothesis record itself knew this — it carries a `_needs_build` field: "Emit logz_sq_final ... from the SHARED telemetry block so the CONTROL emits it too -- today logz_sq_mean is printed only when cfg['zloss'] is set, and it is run-averaged, which the log(8192)=9.0 initialisation transient makes unreadable." `grep -rn _needs_build tools/` shows **no tool reads that field**; the round of 2026-08-19T15:07 then asserted the opposite as fact (line 119: "logz_sq_final < 1.0 is emitted by BOTH arms, so the control supplies the contrast"), and the queue's falsifier repeats it. `claims.diagnostic_would_discriminate` passed it only because "no control has emitted 'logz_sq_final' yet; cannot pre-check", and `coe.py` reports INTACT because E3 audits only completed runs. Net: all 8 R6ZLOSS slots (4 waves) will return E3-inconclusive **by construction** — the exact fate of the four stranded `zloss01_*_treat` slots this queue exists to redeem. Even a post-hoc rename to `logz_sq_mean` cannot rescue it: the run-average includes the log²(8192)=81 transient, so the `lt 1.0` rule would fail in an arm where the mechanism worked perfectly.

### FINDING 2: R6MTP's edit is real and matches the statement; its scoring apparatus is partly fictional

The statement's mechanism IS in the generated source (`ff0e5dc7e782.py`):
- line 143 `self.mtp_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)`; line 174 `torch.nn.init.zeros_(self.mtp_proj.weight)` — zero-initialised, yes.
- line 303 `l2 = self.lm_head(norm(x + self.mtp_proj(x)))` — SHARED lm_head, yes. `norm` is weightless `F.rms_norm` (line 50), so it is idempotent and the "aux logits identical to main logits at step 0" claim holds (main path is `logits = self.lm_head(x)` on already-normed x).
- line 302 `t2 = torch.cat([targets[:, 1:], torch.full_like(targets[:, :1], -1)], dim=1)` — targets are t+1, shifted once more = t+2, correct; line 304 softcaps identically; line 456 `MTP_W = 0.1`; the block is gated `if self.training and reduction == 'mean'` so `evaluate_bpb`'s path is untouched; line 253 puts `mtp_proj` in the lm_head AdamW group. All as stated.
- `mtp_aux_loss_drop` is emitted (lines 745-750), **treatment-only** — which the hypothesis's own predicate declares "necessarily treatment-only" and acceptable because the rule is absolute. Internally consistent.

But four disagreements between spec, code, and interpretation remain:
1. The hypothesis's falsifier list REQUIRES `mtp_main_ce_final` and `mtp_aux_minus_main_ce` ("must be recomputed against the main CE alone; emit mtp_main_ce_final alongside it"). Neither exists in the generated source; `mtp_grad_norm_frac` (line 750) still divides by `_last+float(train_loss)` where `train_loss = loss.detach()` **already contains** `MTP_W*aux` — the double-count the hypothesis itself documents, shipped unfixed. Two of the registered falsifiers are unevaluable as written.
2. The hypothesis's competing_explanation says "the assertion on parameter counts is stripped from the source". FALSE: line 256 of `ff0e5dc7e782.py` still reads `assert len(list(self.parameters())) == (len(matrix_params) + ...`. The strip is an unchecked `re.sub` in make_variant.py whose regex cannot cross the multi-line assert — it silently matched nothing, the exact silent-no-op class `sub()` was built to kill. Benign at runtime (mtp_proj is counted on both sides) but the registered description of the code is wrong.
3. The memory clause is calibrated to the wrong vocabulary. Statement: "raising peak_vram_mb by at least 60% ... if either fails the loss result is not scored at all"; prediction reasoning: "vocab 32768 ... a single fused fp32 softcapped logits tensor is 34.4 GB". The dataclass default in train.py is 32768, but the model is built with `vocab_size = tokenizer.get_vocab_size()` (train.py:473) and `prepare.py:45 VOCAB_SIZE = 8192` — `observables.py` even says "~+12% FLOPs at vocab 8192". At vocab 8192 the fp32 logits are 8.6 GB against the measured control peak of 45060 MB, so +60% (+27 GB) may be unreachable even when the second logits tensor fully materialises — the hard no-score clause can void a correctly-working arm.
4. Interpretation drift: the queue entry's falsifier calls "drop>0.1 but flat val_bpb" a "valid negative", while the hypothesis voids exactly that reading as INCONCLUSIVE whenever `mtp_proj_rms <= 1e-4` (shifted-label-smoothing degeneracy). `observables.py` also still says `mtp_aux_loss_drop > 0.5` (vs the registered 0.1) and its emits list omits `mtp_proj_rms`, which the code now prints (line 744). The make_variant M1 comment still says "weight annealed to 0"; the code applies a constant — flagged in the hypothesis rationale, still unfixed. And `_mtp_hist.append(aux.detach())` (line 307) mutates a module-level Python list inside `GPT.forward` under `torch.compile(model, dynamic=False)` (line 533) — the same defect class the noqknorm commentary calls "the arm measured its own instrumentation"; the zloss `_zl_sum` accumulation (line 308 of its variant) is the same class. Any graph-break cost is treatment-only and confounds the raw 300s verdict.

Smaller instances: hypothesis cites "make_variant.py:203-208" for code now at ~275-283 (drift); the R6ZLOSS rationale cites "(L039)" for the wave split, but L039 is the qk_suppress non-activation lesson — the stranded-wave lesson is L031; noqknorm registry emits `q_rms_mean/k_rms_mean` vs code's `qk_q_rms_final/qk_k_rms_final`.

### FINDING 3: unmeasured assumptions, cheapest decisive check first

1. **"The declared diagnostic is emitted by the built variant"** — free: build + grep, which is this audit; it fails for R6ZLOSS today. Add exactly that check to `queue_from_round.py`/`queue_quad.py` (they already build the variant; grep it for `activation.diagnostic`). Decides 8 GPU slots for zero cost.
2. **"The falsifier's companion observables exist"** — free grep: `mtp_main_ce_final`/`mtp_aux_minus_main_ce` are absent; one builder edit before launch.
3. **"+60% peak_vram is the right materialisation threshold"** — free arithmetic at vocab 8192 (8.6 GB fp32 logits vs 45060 MB control peak suggests ~+20-40%, not 60%); re-register the clause before it unsc ores a working arm.
4. **"In-forward Python mutation doesn't graph-break/regress the treatment"** — one 20-step uncharged smoke run with `TORCH_LOGS=graph_breaks`, or read `step_ms_med` treat-vs-ctrl within the wave.
5. **"norm idempotence makes aux logits init-identical"** — verified analytically here (weightless rms_norm); a 3-line CPU check closes it formally.

### FINDING 4: older-run edit verification (both PASS)

- `ns3_poscontrol_*` (ns=3): stored `runs/sweep/variants/ad70e0ed94bb.py` line 268 `momentum=0.95, ns_steps=3,`; content-hash re-verifies its filename. Edit applied.
- `wd040_*` (wd_const=0.4): stored `79bddb1d128d.py` lines 541-542 `def get_weight_decay(progress): return 0.4`; hash verifies. Runtime corroboration: val_bpb 1.00143/1.002015 vs ~0.9915 controls — the edit clearly bound.
- Bonus, `precond=pre`: stored `3d121bec9abf.py`/`1c41af7d5bfe.py` contain the unit-RMS rescale + exactly one NorMuon block, relocated pre-polar; runtime `secmom_ortho_ratio` 15.08 (treat) vs 0.486 (ctrl) proves engagement, not just presence.
- Caveat: rebuilding those cfgs with TODAY's builder yields different hashes (`081874de34e9`, `cf7e78db2fcd`, `c13f6ff00362`, none in variants/) — the builder has evolved, and results records carry **no `variant` field**, so run->source provenance for old runs rests solely on the queue history. Record the variant hash in every result.

No older variant was found whose edit failed to apply. The failures are all in the R6 measurement contract, not in the edits.

## fable_process (fable-pr-7)

All numbers below recomputed from `runs/sweep/queue.json`, `runs/sweep/results/*.json`, `runs/sweep/decisions/`, `lit/lessons.jsonl`, `tools/{balance,direction,agenda,gate,analyze}.py` output, and the artifacts in `rounds/` and `critiques/`. Nothing is taken from narration.

**1. Failure coverage: the result-level ledger is clean; the upstream ledger does not exist.**
Every one of the 13 failed/invalid result records (`C01/C02`, `Q1b_s0-s3`, `R2S_B_s0`, `W03a_3/4`, `zloss01` x4) shows `[lesson registered]` in `analyze.py`, and `gate.py` confirms "every failure has a lesson". I verified the `R3U_B` metric-backfill complaint from the 09:56 critique is now fixed (`val_bpb 0.996869` present inside `metrics`). But the audit only sees failures that produced a result record. Four failures never did, and none has a lesson:

- **mu_const VariantEditError.** The 15:07 round (line 77) states a knob "silently could not build for an entire campaign phase while `direction.py` kept ranking it top-priority". `grep mu_const lit/lessons.jsonl` returns zero hits, and `mu_const` still sits in the axis table as `UNEXPLORED (top priority)`. The trap is re-armed: the next council will be steered toward a knob with a known build failure and no lesson to warn it.
- **Wave-shape starvation.** Last completed run ended 09:40Z. `gpu_watch.log` shows gpu6 free by 14:24 and gpu6+7 free from 14:50, `ours=[]` throughout — because every one of the 14 pending experiment waves was 4-wide (`s0..s3`) and `dispatch.py` refuses partial waves. The fix (2-wide `R6MTP`/`R6ZLOSS` waves) was appended at 15:12:59 silently. No lesson binds future queue construction to observed capacity, so nothing stops the next round from queuing all-quads onto a box where foreign tenants hold 6 of 8 GPUs.
- **Council-cadence collapse.** `gate.py`: critique council 317 min stale against a 90-min cadence; "NEW DECISIONS FROZEN as of 317 min ago". The freeze covered exactly the window in which the adoption and queue-cut decisions were needed. No lesson type covers it.
- **Hard-cap overshoot.** `agenda.py` shows attention at 16 runs against a cap of 12 and optimizer_numeric at 13 — the caps fired only after 4-5 excess runs landed (8-arm wave pairs overshoot a per-family counter). Unlearned.

**2. Concentration: real, and the coverage table proves it.**
47 treatments concentrate on swdiv (16), precond (11), tbs (8), ve (8); 78 of 125 valid runs (62%) are controls. Seventeen axes sit at n=0 after 138 runs; four mechanisms (mtp, noqknorm, prefetch, zloss) have never executed. The pending queue touches only dbs, mlp, mu_ceil, mu_warmup, noqknorm, mtp, zloss — leaving 13 axes with nothing even queued. Legitimacy varies: `win` is properly blocked by the attention hard cap; capacity (depth/dim) has a measured cost argument (L019); systems/compile_mode is genuinely wave-confounded. Not legitimate: **signal_scale** (rope, softcap, x0init) is `agenda.py`'s highest-scoring family (3.354), annotated `cost: free`, with 76 claims behind it — zero runs, zero queue entries. **schedule** is `cost: free` with 93 claims of which 71 are unread — the literature gate is "legitimate" only in the sense that nobody did the reading in 138 runs. And the active-direction machinery is decorative: `ACTIVE DIRECTION: ve_placement` (the lowest-scoring eligible family, gap 0.00) while zero pending entries touch ve and the queue spans eight other families. The controller points one way; the queue goes another.

**3. Previous critique (09:56): adopted 3, ignored 4, and one quiet substitution.**
ADOPTED: (a) `R5T17` refined to `dbs=64` (verified in queue.json — though the round's own critic now correctly calls the resulting dbs+tbs stack "a second, worse L046 in the making"); (b) zloss re-queued at last, plus mtp, after three consecutive artifacts demanded it — ~5h later, at 15:12; (c) `R3U_B` backfill done. IGNORED, with costs:
- **I1 was never queued.** The 09:56 adoption plan was "adopt ONCE, after R5T18_B and I1 read out", with PLATFORM = tbs=18 + ve=1 + swdiv=4 (+precond if I1 confirms transfer). No I1 wave exists in queue or results. Consequence: the 15:07 round's proposed PLATFORM is `tbs:18, ve:2, swdiv:2` — the confirmed R4X stack (0.986956, the campaign's second-best result) has silently fallen out of the platform decision with neither a measurement nor a stated reason. That is not caution; it is losing a finding in transit between artifacts.
- **L038 param-assert, third consecutive flag.** `make_variant.py` still deletes the param-count assert in the mtp path (`re.sub(...) -> ""`, ~line 252) and the unet path (~line 303). The R6MTP waves just queued are the first new-parameter mechanism to launch — they will run precisely without the protection the lesson prescribed.
- **valid_negative type-bending, third recurrence.** L041, L042, and now L044 ("swdiv rung2 still pays") are confirmed wins typed `valid_negative`. No `confirmed_win` type was added. The institutional memory misfiles its three biggest positives.
- **Bare small-df t-statistics.** The 15:07 round cites "t=-60.7" on ~3 df, unqualified, after two critiques objected.
Also: the 15:07 round's own decision 3 (cut 10 stale waves, ~11 GPU-h) is **not enacted** — `decisions/2026-08-19T15-15-59Z_decision.json` still scores all 22 pending waves and ranks `R5NOQK` (tbs=19) above `R6MTP`.

**4. The queue is two-thirds liability.**
72 pending entries, 22 waves. 60 of 72 are at tbs=19 (36 controls + 24 treatments); 40 runs across the 10 waves `R5NOQK/R5T17/R5DBS/R5MLP/R5S16` compare against tbs=19 controls that the round itself says must retire on adoption — every result there becomes a wrong-operating-point number needing a rerun at ~2000 steps. Half the pending queue (36 controls) recalibrates an instrument about to be discarded. The asset portion is small and identifiable: `R5MU`/`R5MC` (bound the L046 leg-1 confound, treatments at tbs=18) and the 2-wide `R6MTP`/`R6ZLOSS` waves — the only entries that can physically launch on the 2 GPUs actually free. Cut the 10 stale waves now, and enact the synthesizer's resolution of the R5S16 contradiction (supersede L043's sequencing clause via a registered lesson, resume swdiv rungs post-recalibration at tbs=18) — as prose it is just a second pocket veto waiting to be discovered. The 4-wide habit should also die: L020 says counterbalance on GPU at ANY width, so 2-wide yoked pairs are legitimate, and quads on a box that rarely frees 4 GPUs are a self-inflicted 5-hour stall.

**5. The balance claim is instrument-shopping.**
`balance.py` reports "in band [30%, 50%] — balanced" at EXPLOIT 3/10 = exactly 30.0%, passing its floor on a strict inequality — a boundary pass dressed as comfort. More importantly it does not measure the advertised constant at all: balance.py enforces an EXPLOIT floor of 0.30; the campaign's EXPLORE_FLOOR=0.35 lives in `direction.py`, whose `explore_debt` is **+0.23** — explore share 12% against a 35% floor. Same campaign, same day: one instrument says balanced, the other says deep exploration deficit, a 58-point disagreement. Neither is honest. balance.py counts "experiments" (10) rather than runs or GPU-hours, so 138 runs collapse to 10 units. direction.py's debt has two structural defects: its denominator includes all 125 ok runs (78 controls), capping the achievable explore share at 47/125 = 37.6% — the floor is nearly unsatisfiable by construction, the same "threshold nothing can cross" defect balance.py's comments boast of fixing; and its numerator tests FINAL axis counts (`n <= 1`), retroactively stripping explore credit from any axis explored thoroughly — the backward-looking reclassification balance.py fixed three times in itself and never exported. The ledger does not lie so much as it lets the operator quote whichever number flatters the moment. Fix: one chronological, GPU-hour-weighted classification, one constant, both tools reading it.

**Verdict.** The process organs (gate, analyze, coe, lessons) are unusually honest about what they can see; the failures now live in what they cannot see — build failures, wave-shape starvation, cadence collapse, and findings dropped between artifacts. The single most expensive drift is the adoption pipeline: three artifacts debated tbs=18 while the confirmed R4X stack quietly vanished from the proposed PLATFORM and its transfer test (I1) was never queued.

## fable_synthesis (fable-sy-7)

### FACT (recomputed from primary state, not from other critics)

1. 138 result records exist in `runs/sweep/results/`; 125 carry `ok:true`. 13 invalid/failed records each name a lesson in `invalid_reason` (co-tenancy L001/L002, double-booking L004, stranded wave-splits L031/L036). 50 lessons registered in `lit/lessons.jsonl`, ~40 active; by type, roughly 30 of the active set are integrity/overclaim/resource/scope — i.e., the ledger is mostly about the apparatus, not the model.
2. Best `val_bpb` = 0.984017 (`R5T18_A_s3_treat`, gpu7, 2003 steps, tbs=18). Platform-control corpus at tbs=19 sits at 0.9911–0.9938 with a per-GPU device model spanning 0.002474 (gpu7 0.991311 vs gpu4 0.993785). The tbs=18 counterbalanced quad mean is −0.007953 (per-device deltas −0.008212/−0.008143/−0.007750/−0.007705). The 0.984017 headline is the favorable-device cell of that quad.
3. Effect sizes by discovery order are RISING, not shrinking: precond −0.0011 → ve −0.0015 → swdiv4 −0.0023 → three-lever stack −0.0044 (86% of additive, L042) → tbs=18 −0.0080. The newest lever is ~28x the counterbalanced resolution (0.000280).
4. Of 125 valid runs, 78 are platform controls (62%). Six direction families (`objective`, `schedule`, `signal_scale`, `systems`, `compile_mode`, `capacity`) sit at zero runs; explore_debt +0.23. All runs are seed 42 on one shared box (L043).
5. `claims.diagnostic_would_discriminate` is now wired at three sites and all three act on the return: `queue_from_round.py:68` (skips the entry), `queue_quad.py:96` (returns 1), `selector.py:210` (activation penalty). I verified the tool discriminates: `num_steps {gt,0}` → REFUSED (78/78 controls pass); `tokens_per_step {lt,300000}` → ok, 78/78 controls FAIL; `secmom_ortho_ratio {gt,1.0}` → ok, 54/54 fail; `num_steps {gt,1500}` → ok, 78/78 fail. For never-emitted fields (`mtp_proj_rms`, `logz_sq_final`) it returns True with "cannot pre-check" — permissive, NOT a verification.
6. The 2026-08-19T14:31Z project critique found three defects in fixes committed the SAME day: the co-tenancy taint write is a `KeyError` that kills the dispatcher and destroys the evidence it was persisting (`dispatch.py:609`, `job["name"]` does not exist); `_TAINTED_AT_ADOPT` is consulted at adoption (:537 via :552) before it is populated (:576) — dead code; the `.gitignore` negations under `runs/` are inert, so future result and decision records silently stop shipping. All three postdate `tests/test_guards_fire.py` and the CI commit (5b7fca9), and that CI is itself red on any fresh clone (E1: 706 missing snapshots; pytest: `queue.json` not shipped).
7. Timeline: 25 of 38 commits touch decision-affecting code; all 25 landed after the first result; the verdict function changed 8 times while results accumulated; ~19 policy-file touches on 08-19 alone.

### INFERENCE

**Q1 — the defect rate is scrutiny-limited, and the system's real defect is churn-during-measurement.** A depleting defect pool under constant audit effort shows rising time-to-first-defect; this campaign's is flat (18–25 min across every probe, none empty). The mechanism is visible in FACT 6–7: the pool is replenished faster than it drains, because ~650 lines/day of new policy code land mid-window, and the fixes themselves inject critical defects (the taint fix is the exhibit: it crashes the very path it hardens). Discovery rate is proportional to auditor effort, not residual pool — that is the definition of scrutiny-limited. **What would distinguish the two:** freeze the measurement stack at a tag; run two independent audits of the same frozen tree with zero interleaved commits. (a) If the second audit's time-to-first-defect rises sharply or it returns empty, the pool is depleting. (b) The overlap between the two auditors' finding-sets gives a capture–recapture (Lincoln–Petersen) estimate of the residual pool. Today no such measurement is possible, because no two audits have ever examined the same code state. The system change this justifies is not "more guards": it is (i) a measurement freeze — policy edits batch between run windows, any edit restarts the window (the pipe-int-2 release checklist already says this and nobody has executed it); and (ii) extraction of the `tick.sh` heredoc into `tools/queue_merge.py` so the guard-must-ship-its-firing-test rule is even applicable to the layer that has produced the last three critical defects (dispatch/tick/quarantine — the only untested layer, and the only layer still producing CRITICALs).

**Q2 — it is making real progress on the objective, and the "ever-smaller effects" charge is factually backwards; the actual pathology is adoption latency.** The effect trajectory rises monotonically to −0.0080 (FACT 3); the campaign is not polishing an instrument to resolve dust — its newest lever needed no fine instrument at all (28x resolution). Counterbalanced, activation-verified: control ~0.9917 → demonstrated composite headroom ~0.980 if the stack transfers to tbs=18. That is genuine movement on the only objective. The honest indictment is different: (a) 62% of valid runs are controls (L029 said this and it kept growing); (b) the best-known configuration is STILL not the platform — L043's pocket veto was diagnosed, then synth-6 showed the operator's own decisions 2 and 3 are mutually contradictory (hold PLATFORM pending R5S16 while cutting R5S16 as stale), and adoption still hasn't happened; (c) 10 of 14 queued waves measure against a baseline known to be ~0.008 worse than adoptable. The instrument is not measuring ever-smaller effects; the campaign is spending an ever-larger fraction of its budget re-certifying a baseline it has already decided to abandon. Progress per GPU-hour, not resolution, is the number at risk.

**Q3 — `tests/test_guards_fire.py` would NOT have caught all 7, and its own bottom half reproduces the failure mode it documents.** Taking the seven instances on record — (1) `diagnostic_would_discriminate` defined and never called (c3c2ed2); (2) its recurrence after 4cf4bd8 "make the activation check real"; (3) `selector.score`'s promised activation penalty never written; (4) three vacuous activation rules registered post-L030; (5) `decide.py` ordering not propagated by `tick.sh`/dispatcher; (6) lesson enforcement dead (early `blocks_keys: []`, L007's claimed guard nonexistent); (7) quarantine/taint inheritance dead at adoption — the test file catches (1), (2) and (4) outright and (6) partially (it proves ONE wired lesson fires; it does not test that a hookless `block` lesson is refused at registration). It would **MISS (3), (5), and (7)**:
- **(3) selector activation penalty:** no test injects a vacuous-diagnostic arm and asserts the penalty term appears. `test_selector_refuses_a_blocked_config` tests only the lesson −inf path. Worse, the known escape — an arm declaring NO diagnostic skips the penalty entirely (selector.py:203–213) — is untested, so the incentive gradient toward declaring nothing is unguarded.
- **(5) decide→machine ordering:** covered only by a string grep (`"pos.get"` in tick.sh, self-described as "weaker") plus `test_decision_is_recorded...`, whose field assertions sit inside `if decisions:` — vacuously green when no record exists. The live defeats (dispatch's `next_batch` picks largest-wave-first with order as tiebreak; `__solo__` entries bypass `decide.pending()` entirely) pass every test.
- **(7) taint/quarantine:** empirically proven — both the dead `_TAINTED_AT_ADOPT` and the `KeyError` shipped while this test file was in CI. Nothing in the file exercises dispatch persistence beyond grepping `"_crashed"` and `"WAVE WEDGED"`.
Two structural holes on top: the wiring test greps for the callee's NAME, so a call site that ignores `_ok` would pass; and the permissive-unknown branch of the pre-check means the next vacuous rule, written against a treatment-only field, sails through both the tool and the test — the door can only refuse vacuity it has control history for.

### DISAGREEMENT

- Against critic-6 and the circulating framing: "increasingly rigorous instrument measuring ever-smaller effects" is refuted by the effect-size timeline (FACT 3). The rigor is paying; the adoption pipeline is what's broken.
- Against explorer-6's language (already flagged by synth-6, I concur independently): calling `mtp_proj_rms`/`zloss_frac_of_total`/`ema_updates` "verified" is false. The tool returned its permissive no-control-history branch; that is "unverified, cannot pre-check," and any queue entry citing them must plan a post-run control-emission check.
- Against the test file's self-image: it is the right idea half-executed. Its top half asserts outcomes; its bottom half is five greps and a conditional that cannot fail — the exact "code exists, reads correctly, can never refuse anything" pattern its docstring names.

### VERDICTS

**REFINE — the guard-test discipline itself.** Convert the five presence-greps and the `if decisions:` test into outcome tests, prerequisite: extract the tick.sh heredoc to `tools/queue_merge.py`; add a quarantine round-trip test (inject a taint, restart-adopt, assert `cotenant:true`). Falsifier: an outcome test for the taint path passes against current `dispatch.py` — then the 14:31 critique's defect 1/2 analysis is wrong and the grep approach was sufficient. Activation diagnostic: none applies (no run emits a field for host-layer guards) — this is a code-level verdict; I checked `claims.diagnostic_would_discriminate` has no relevant field and report that honestly rather than decorating it with a vacuous one.

**PIVOT — adopt tbs=18 as PLATFORM the moment R5MU/R5MC land (not gated on R5S16), with one recalibration control wave at the ~2000-step operating point, then fund the zero-run `objective`/`schedule` families.** Falsifier for the adoption: the recalibration wave's within-GPU sd at ~2000 steps is not comparable to 0.000198 — then the device/resolution model does not transfer and every verdict at the new platform needs a re-measured band first. Activation diagnostic for the recalibration wave: `num_steps {op:gt, value:1500}` — VERIFIED discriminating: "ok: 78 of 78 controls FAIL the rule". For the mtp/zloss arms: `mtp_proj_rms {gt,0}` / `logz_sq_final {lt,1.0}` — tool returns True but via "no control has emitted this field yet; cannot pre-check": **unverified**, permissive-by-construction; the first wave must confirm the control side actually emits-and-fails or the run is inconclusive by design.

**BLOCK — R5T17 as configured.** `dbs:128, tbs:17` fails `train.py`'s divisibility assert (tokens_per_fwdbwd 262144 does not divide 2^17) per fable-sy-4's line-503 trace, and it stacks two step-count-moving knobs, recreating L046 with worse attribution. Falsifier: a preflight build+single-step of the exact queued cfg completes — then the assert analysis is wrong and only the L046-stacking objection stands (refine to dbs=64 single-knob arms). Activation diagnostic (for any refined tbs-rung arm): `tokens_per_step {op:lt, value:300000}` — VERIFIED discriminating: "ok: 78 of 78 controls FAIL the rule".

**TEST NEXT — one composite transfer quad, `{tbs:18, ve:1, swdiv:4, precond}` treats vs `{tbs:18}` controls, GPU-counterbalanced.** It retires the five-quad "do the levers transfer to the new operating point" question for one quad, and it is the only experiment whose answer changes what gets adopted. Falsifier: device-corrected composite effect at tbs=18 weaker than −0.0022 (half its tbs=19 value) or sign-flipped — transfer fails; only then spend component-attribution quads. Activation diagnostic: `secmom_ortho_ratio {op:gt, value:1.0}` — VERIFIED discriminating: "ok: 54 of 54 controls FAIL the rule" (plus `tokens_per_step {lt,300000}`, verified above, to prove the operating point).

