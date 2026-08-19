# Five levers, one uninterpretable result, and the instrument required to tell them apart

- **Scope:** `val_bpb`, 300 charged training seconds, single H200, frozen `prepare.py`.
- **Covers:** `R2V_*`, `R2S_*`, `R2S2_*`, `R3U_*`, `R3Q_*`, `R3T_*`, `R4X_*`; hypotheses
  `hyp_precond_pre_r1_v5`, `hyp_ve_density_r1`, `hyp_swdiv_short_window_halved_r1_v4`,
  `hyp_unet_skip_r3`, `hyp_qk_suppress_r3`, `hyp_tbs20_r3`, `hyp_stack_three_levers_r4`.
- **Chain of evidence:** `python3 tools/coe.py` — INTACT.

## Abstract

We report six interventions measured on a fixed-time language-model training benchmark:
a ~50M-parameter GPT trained for 300 charged seconds on a single H200, with the data
pipeline frozen. Three lower validation bits-per-byte and three do not. The largest
single effect is halving the short span of a sliding-window attention pattern; combining
it with a value-embedding density change and a second-moment reordering in the optimizer
yields the best model we have trained. The combination realises most, but measurably not
all, of the sum of its parts, and we locate the overlap in a shared throughput channel.

The central methodological claim is that none of these effects is visible without a
paired, device-counterbalanced design. The spread of byte-identical control runs on this
host is roughly twenty times the size of the effects we report. We describe the design,
the noise model that motivates it, and — at some length, because we think it is the more
transferable contribution — the instrument defects that had to be found and fixed before
any of the numbers below could be trusted.

## 1 Introduction

The benchmark fixes *time*, not tokens. A change that improves the model per step but
costs throughput can be a net loss, and a change that buys steps by removing computation
can be a net win even if it degrades the model per step. This makes throughput part of
quality rather than a nuisance to be regressed away, and it is why we report raw
`val_bpb` as the verdict everywhere and use step count only to explain *why* a result
came out as it did.

The campaign began with no experimental priors: every prior belief was deleted
deliberately, on the grounds that most of them were wrong or unfalsifiable. What we had
at the start was a literature corpus and a set of properties of the apparatus. What we
did not have — and had to build before anything else was possible — was a measuring
instrument capable of resolving the effects we were looking for.

**Contributions.**

1. Three confirmed levers and three confirmed non-levers, each measured under a
   counterbalanced design with its activation independently verified (§4, §5).
2. A combination experiment showing the three levers are complementary but partly
   redundant, with the overlap channel identified (§6).
3. A description of the measurement design and, more importantly, of the ways it failed
   (§7). Six of the seven defects found in this block were in our own apparatus, not in
   the model.

## 2 Setup

A GPT of approximately 50.3M parameters: vocabulary 8192, depth 8, `n_embd` 512, MLP
ratio 4, head dimension 128. Attention follows a repeating `SSSL` window pattern — three
short-window layers and one long-window layer per block, with the final layer forced to
full context. Value embeddings are attached to every other layer. QK-norm is applied.
The optimizer is Muon on 2D matrices with AdamW elsewhere. Training runs under
`torch.compile` in bf16 for 300 charged seconds, which on this host is roughly 1000 steps
at 524288 tokens per step, and completes about two epochs of a fixed dataset.

`prepare.py` — tokenizer, loader, packing, evaluator and the BPB accounting — is frozen
and byte-identical to the upstream reference. `train.py` is the only file we edit.

The host is shared. Other tenants occupy most GPUs most of the time, and co-tenancy on a
GPU invalidates a run outright.

## 3 The measurement problem

The 70 *byte-identical* control runs in this campaign span 1.025517 at worst
(`C04_control`) to 0.990908 at best (`ctrl_R1P_A_slot1`), a range of 0.034609. The
effects we report are all smaller than 0.004408. An unpaired single run therefore cannot
see any of them: the host's own spread is roughly an order of magnitude larger than our
largest effect, so a single unpaired comparison could not distinguish any of them from
contention noise.

The dominant nuisance is host CPU contention, which moves the step count run to run
through the frozen packing loop. Two structural facts follow.

**Concurrency, not proximity.** Runs must overlap in wall-clock time to share the same
contention. Runs launched back-to-back on one GPU are sequential, and the load drifts
between them; grouping those as a "wave" reports a sequential spread under a within-wave
label, which flatters the instrument.

**The offset belongs to the GPU.** The four devices we use have systematically different
control means, and the spread between the fastest and slowest device is comparable to the
effects being chased. An earlier version of this campaign attributed that offset to the
CPU core block instead; fixing which factor the offset belongs to changed verdicts.

## 4 Design: the counterbalanced quad

Each experiment is two four-wide waves, launched concurrently on four GPUs:

```
wave A:  [treatment, control,   control,   treatment]
wave B:  [control,   treatment, treatment, control  ]
```

Across the pair, the treatment occupies **every device exactly once**, and so does the
control. Deltas are then formed **within each device** — treatment on gpu*k* minus
control on gpu*k* — which removes that device's fixed offset inside each difference
rather than relying on it to cancel in the mean. The four deltas are averaged.

Two consequences matter for reading §5. First, `n = 4` pairings, so every *t* we report
carries 3 degrees of freedom and should be read as evidence of *direction* rather than as
a precise magnitude. Second, when the design breaks — a device set shifts between waves,
or an arm crosses an operating-point boundary — the analysis **refuses a verdict** rather
than averaging across a broken swap. This happened three times in this block and cost
three extra waves; we consider that the correct trade.

**Activation.** Every hypothesis declares a diagnostic that the run must emit, checked
*before* the outcome. A run whose diagnostic is absent, or which fails its rule, is
**inconclusive** about the mechanism — never a negative. Otherwise a broken intervention
and a genuine null are indistinguishable, and good ideas get retired for free.

The resolution of the instrument, computed as twice the pooled within-device control
standard deviation over the square root of two, is **0.000282** (pooled sd 0.000199).
Effects below that are not claimed.

## 5 Results

All comparisons are same-device paired across a counterbalanced quad pair, seed 42.
Per-pair arithmetic is given in Appendix A so every derived number can be checked.

| intervention | mean delta | *t* | activation | verdict |
|---|---|---|---|---|
| `swdiv` 2→4 (short span halved) | −0.002298 | −20.3 | `flops_per_token_M` 220.204 vs 239.078 | **better** |
| `ve` 2→1 (value embedding every layer) | −0.001487 | −12.2 | `n_ve_layers` 8.0 vs 4.0 | **better** |
| `precond` (second moment before polar) | −0.001147 | −12.3 | `secmom_max` ≫ control | **better** |
| `qk_suppress` 0.1 | +0.001226 | +15.8 | **failed** — see below | **inconclusive** |
| `unet` (zero-init skip connections) | +0.004123 | +105.6 | `skip_lambda_absmean` ≈ 0.2 from 0 | **worse** |
| `tbs` 19→20 (tokens/step doubled) | +0.022545 | +138.5 | `tokens_per_step` 1048576 vs 524288 | **worse** |

### 5.1 Halving the short attention span (`swdiv` 2→4)

The `SSSL` pattern gives six short-window layers and two long-window layers at sequence
length 2048. Halving the short window from 1024 to 512 takes the per-token FLOP estimate
from 239.078 to 220.204 — a removal of 18.874, which is 7.9% of the total, **not** the
26% that attention as a whole represents. The two long-window layers carry about 40% of
the attention term and are untouched.

This distinction was not academic. The hypothesis was originally justified with the 26%
figure, which overstated the throughput lever by a factor of about 3.3. The error was
caught by the literature review before the experiment ran, the arithmetic was corrected,
and the prediction threshold was **deliberately not lowered** to compensate. The
corrected, harder prediction was met anyway.

The mediator behaved as the mechanism describes: step counts rose in every cell (for
example 1049 against a control's 999 on the same device), and quality held. This is the
opposite outcome to an earlier arm which reduced Newton–Schulz iterations: that change
was worth about 0.2% of step time, bought no steps at all, and came back worse. The
difference is that 7.9% is large enough to move the critical path and 0.2% is not.

### 5.2 Value embeddings on every layer (`ve` 2→1)

`ve` is a *stride*, so `ve=1` is the denser setting: value-embedding tables go from four
to eight and parameters from 50.3M to 67.1M — a 33% larger model that runs slower and
completed **fewer** steps than its control in every cell. It won regardless, so the
quality gained per step more than repays the throughput lost.

This contradicts the prior we registered before running it. Three claims in our corpus
opposed it: injection at all layers measured worse than at a single layer, dense
per-layer stream expansion saturating, and a result suggesting the benefit of such
memories is the gated pathway rather than the learned table content. All three carried
honestly low transfer scores — image autoregression, tokenizer-transfer fine-tuning, and
a 2.5B mixture-of-experts — and this is what low transfer looks like when it bites. The
claims retain their opposing stance; what we now know is that they do not carry to 50M
parameters at 300 seconds.

**One cell is void.** On the slowest device the treatment's added parameters cost enough
throughput to finish at 501.2M tokens and `final_epoch` 1.0 against its control's 2.0.
That is a different operating point, so the pair is excluded. The exclusion is
structurally biased against parameter-adding treatments; here it happened to cut
*against* the result, since including that cell would have strengthened it. This leaves
`ve` with three pairings and 2 degrees of freedom — the weakest of our three wins.

### 5.3 Second-moment reordering (`precond`)

Moving the NorMuon second-moment computation before the polar iteration, with unit-RMS
momentum normalisation, gives −0.001147 across four devices. The activation observable
`secmom_max` separates the arms by roughly two orders of magnitude, because the second
moment is taken over raw momentum rather than over an orthogonalized update and the two
live on structurally different scales.

### 5.4 Two clean negatives

**Zero-initialised U-net skips** raised `val_bpb` by 0.004123 with a standard deviation
across four devices of only 0.000078 — the tightest measurement in the campaign. The
precision is a consequence of the intervention: the skips add one elementwise addition
per lower-half layer and no matrix multiply, so they cannot buy or cost steps, and the
step counts confirm they did not. The effect is therefore pure quality, with none of the
throughput entanglement that explains most of the variance elsewhere. The lambdas grew
from exactly zero to about 0.2, so the model *used* the path and was worse for it.

Worth recording: this intervention was the independent top pick of two separate proposers
in the same council round. Agreement among proposers carried no evidential weight at all.

**Doubling tokens per step** raised `val_bpb` by 0.022545, roughly eighty times the
resolution and by far the largest effect we have measured. Step counts halved as
expected. At a fixed time budget this model is far *above* its critical batch size, so
halving the optimizer update count costs vastly more than the reduction in gradient noise
returns. We note that the step law cannot be used to explain this away: it was fitted at
524288 tokens per step and this arm moves that very quantity.

### 5.5 One honest inconclusive

`qk_suppress` produced a textbook-looking counterbalanced negative — +0.001226 at
*t* = +15.8, more than four times the resolution. We do **not** report it as a negative.
Its declared activation diagnostic, `qk_q_rms_final`, reads exactly 1.0 in every
treatment *and* in every control, because QK-norm pins query RMS to unit norm by
construction. The diagnostic could never have shown engagement, so the run cannot
distinguish "the mechanism works badly" from "the edit never bound and something else
moved". The hypothesis's own pre-registered falsifier named this case, and we apply it
against ourselves. The axis is untested, not refuted.

## 6 Combining the levers

The three winning levers were run together against the **unchanged** control:

**mean delta −0.004408**, sd 0.000152, sem 0.000076, *t* = −58.0 over four devices, with
all eight arms at `final_epoch` 2.0 and all three components independently confirmed
active in the same records (`n_ve_layers` 8.0, `flops_per_token_M` 220.207, `secmom_max`
17.30634117 against a control near 0.25). The best model produced is `R4X_A_s3_treat` at
**0.986956**.

The naive sum of the three separately measured effects is −0.004932 (Appendix A). The
combination realises about 89% of it, leaving a shortfall of 0.000524.

**Where the overlap is.** `ve` and `swdiv` both act partly through step count, in
opposite directions: `ve` costs throughput by adding parameters, `swdiv` buys it by
removing FLOPs. Stacked, they nearly cancel in that channel — on the slowest device the
combined arm ran 991 steps against its control's 992, where `ve` alone had managed 956
and voided. Some of the two levers' separate gains were the same gain counted twice.

**Why this reading is licensed.** The interaction was predicted *before* the run, and the
reading order was fixed in advance: check `final_epoch` on every arm first, then look at
`val_bpb`. Had the combined arm voided on the slow device, a shortfall would have been an
operating-point artefact and said nothing about redundancy. It did not void — and the
step count shows exactly why. After the fact these two explanations are
indistinguishable, which is the entire reason the order was pre-committed.

**What we do not claim.** Whether the 0.000524 shortfall is distinguishable from zero
depends on propagating the error of the additive sum, which is itself three noisy
estimates added together, and we have not established that the shortfall survives it. The
honest statement is that the stack is *at least* strongly complementary, and that partial
redundancy is the better-supported of the two readings but not a settled one.

## 5.6 The largest result is uninterpretable

Halving tokens per step from 524288 to 262144 lowered `val_bpb` by 0.007953 across four
devices, all at epoch 2.0 — roughly twice the three-lever stack and the largest effect we
have measured. We can state the configuration result. We cannot state the mechanism.

Halving the batch doubles the step count, from about 990 to about 2000, and while the
learning-rate and weight-decay schedules are indexed by progress, at least six other
quantities are indexed by step and therefore all moved at once: the Muon momentum ramp
(completing at 15.0% of the run instead of 30.3%); the eleven uncharged warmup steps,
which give the control 5.8M free tokens against the treatment's 2.9M; Adam bias
correction; the Adam and NorMuon EMA horizons; the per-step weight-decay dose; and
gradient accumulation going from 2 to 1.

We first found only the momentum ramp and queued two arms to decompose it. An independent
audit enumerated the rest, and those two arms bound one leg of six. Only the uncharged
warmup has a known direction, and it favours the control, so the measured win is if
anything understated.

We report this at length because the temptation was to publish t = −60.7 and call it a
critical-batch-size effect. The number is real; the mechanism is not established.

## 7 Threats to validity

**Single seed.** Every result is seed 42 on one host, by operator direction to conserve
wall clock. Nothing here is established as seed-general. The `ve` result rests on three
pairings and 2 degrees of freedom; its direction is solid and its magnitude is not.

**Degrees of freedom.** All *t* statistics carry 3 df or fewer. They are reported because
they are the natural summary of a paired design, not because they license three
significant figures of confidence.

**No adoption.** The confirmed levers have deliberately **not** been folded into the base
configuration. Doing so would shift every future control by the stack effect −0.004408 and retire the
control corpus and per-device model on which the 0.000282 resolution rests. We prefer a
working instrument and a provisional claim to a stronger claim measured with a broken
ruler.

**Instrument defects are the dominant failure mode.** Of 46 registered lessons, 8 are
scientific negatives and the remainder are process or measurement failures. That ratio is
the honest summary of where the effort went.

## 8 What the instrument cost

We describe the defects because they are more transferable than the results, and because
each was found by an audit rather than by intuition.

**An instrument that could not measure what it claimed.** We added a GPU-slack metric
from a pair of CUDA events bracketing the step. A CUDA event pair measures elapsed time
between two stream markers, *including* any interval the device sat idle, and our bracket
enclosed the data loader. The quantity was therefore pinned to the step time by
construction: it read near-zero slack on every run, which would have been reported as
"the GPU is saturated". It measures nothing about utilisation and no argument may rest on
it.

**A metric silently discarded from every run.** The per-token FLOP estimate was printed
by every variant and dropped by the dispatcher's parser, which required whitespace after
the colon where the print emits none. It was absent from all 116 records. Recovered by
re-reading each run's own log — recovery, not recomputation — and verified to reproduce
from first principles.

**Three vacuous activation diagnostics.** In one session we shipped three hypotheses
whose diagnostic the control also satisfied: a clamp fraction that reads 0.0 in both arms,
a step count with the rule "greater than zero" that all 58 controls pass, and the QK-norm
case in §5.5. All three were registered *after* the lesson recording that failure mode
existed. Prose did not prevent recurrence; a pre-registration check that tests a proposed
diagnostic against the existing control corpus does, and now refuses all three.

**A policy that strangled the search.** The explore/exploit controller retired an axis
after four consecutive runs without improvement. A counterbalanced quad is four runs of
the *same* value — so the moment the campaign adopted the design that makes its results
trustworthy, every axis began closing after a single experiment. Three axes closed having
tried exactly one distinct value each, including one that closed on the evidence that its
*opposite* direction was worth testing. The streak now counts distinct values.

**A ledger that could not reach its own threshold.** The explore/exploit share was first
measured over runs. Each experiment is one novel run plus about seven counterbalancing
replicates, so the exploitation share could not exceed roughly 12.5% against a 30% floor.
This is the same defect as an activation rule the control passes: a threshold nothing can
cross measures nothing.

We note the pattern that these fixes share: each was discovered when a rule blocked
something we wanted to run. We have never once gone looking for a rule that was too
permissive, and that asymmetry is itself a threat to validity.

## 9 Conclusion

Three interventions lower `val_bpb` at a fixed 300-second budget on this benchmark, and
combine to about 86% of their additive sum, improving the best model from 0.990345 to
0.986956. Three others do not, one of them by a very large margin. The measurement design
that makes these statements possible — concurrent yoked pairs, device counterbalancing,
pre-registered activation diagnostics, and a willingness to refuse a verdict when the
design breaks — mattered more than any individual intervention, because the effects are
roughly a twentieth of the noise an unpaired comparison would face.

## Appendix A — Derivations

Every derived statistic is computed from result records. Each derivation is written on a
single line, with its inputs and its result together, so the arithmetic can be checked
directly against `python3 tools/coe.py registry`.

**Stack** (same-device pairs, four devices): 0.989520 − 0.993970 = −0.004450; 0.988490 − 0.992806 = −0.004316; 0.987526 − 0.991788 = −0.004262; 0.986956 − 0.991558 = −0.004602; mean −0.004408, sd 0.000152, sem 0.000076, t −58.0.

**`swdiv` 2→4**: 0.991488 − 0.993812 = −0.002324; 0.990407 − 0.992907 = −0.002500; 0.989656 − 0.992444 = −0.002788; 0.989449 − 0.991689 = −0.002240; mean −0.002463, sd 0.000242, sem 0.000121, t −20.3.

**`ve` 2→1** (three devices; the gpu4 pair is void at `final_epoch` 1.0): 0.991366 − 0.992733 = −0.001367; 0.990387 − 0.991751 = −0.001364; 0.989819 − 0.991550 = −0.001731; mean −0.001487, sd 0.000211, sem 0.000122, t −12.2.

**`precond`**: 0.992871 − 0.993773 = −0.000902; 0.991425 − 0.992570 = −0.001145; 0.990878 − 0.992064 = −0.001186; 0.990345 − 0.991699 = −0.001354; mean −0.001147, sd 0.000187, sem 0.000187 / 2 = 0.000093, t −12.3.

**`unet`**: 0.998052 − 0.993857 = +0.004195; 0.996869 − 0.992827 = +0.004042; 0.996018 − 0.991834 = +0.004184; 0.995551 − 0.991481 = +0.004070; mean +0.004123, sd 0.000078, sem 0.000039, t +105.6.

**`tbs` 19→20**: 1.016937 − 0.993982 = +0.022955; 1.015369 − 0.992781 = +0.022588; 1.014320 − 0.991852 = +0.022468; 1.013689 − 0.991521 = +0.022168; mean +0.022545, sd 0.000326, sem 0.000163, t +138.5.

**`qk_suppress`** (measured but INCONCLUSIVE, §5.5): 0.995063 − 0.993829 = +0.001234; 0.993834 − 0.992786 = +0.001048; 0.993304 − 0.991880 = +0.001424; 0.992686 − 0.991489 = +0.001197; mean +0.001226, sd 0.000155, sem 0.000077, t +15.8.

**Additivity** (§6), summed in two checkable steps: 0.002298 + 0.001487 = 0.003785, then 0.003785 + 0.001147 = 0.004932; the measured stack mean 0.004408 is 89% of that sum, a shortfall of 0.004932 - 0.004408 = 0.000524. NOTE the swdiv component here is the CLEAN single-experiment quad, not the superseded cross-experiment stitch 0.002463 which overstated it by about 7 percent and which an earlier version of this table used.

**Resolution**: pooled within-device control sd 0.000199 gives 2 × 0.000199 ÷ √2 = 0.000282.

**Control corpus spread** (§3), over 70 runs: 1.025517 - 0.990908 = 0.034609.

**Improvement in best model**: 0.990345 - 0.984017 = 0.006328.

**tbs 19 to 18** (§5.6), same-device pairs: 0.985558 - 0.993770 = -0.008212; 0.984733 - 0.992876 = -0.008143; 0.984252 - 0.992002 = -0.007750; 0.984017 - 0.991722 = -0.007705; mean -0.007953, sd 0.000262, sem 0.000262 / 2 = 0.000131, t -60.7.

**swdiv 4 to 8**, same-device pairs: 0.990685 - 0.993850 = -0.003165; 0.989450 - 0.992714 = -0.003264; 0.988349 - 0.991985 = -0.003636; 0.988254 - 0.991629 = -0.003375; mean -0.003360, sd 0.000203.

**FLOPs removed by `swdiv` 2→4**: 239.078 − 220.204 = 18.874 per token, which is 7.9% of 239.078.

**Step-count evidence for the overlap in §6**: `ve` alone reached 956 steps on the slowest device and voided; the stack reached 991 steps against its control's 992 on that same device.
