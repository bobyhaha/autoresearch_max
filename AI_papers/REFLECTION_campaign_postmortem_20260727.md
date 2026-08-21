# A Candid Postmortem: What This Autonomous Campaign Got Right, and What It Got Wrong

*OPHIS autonomous research agent — reflection written 2026-07-27, after 69 rounds, 369 runs, and 11 papers on the fixed-5-minute NanoChat benchmark.*

This is not an experimental paper. It is a high-level, self-critical reflection on how the campaign was *run* — prompted by three fair observations from the operator: that the later campaign was mostly incremental, that a comparable system (Recursive's automated-research loop on the same NanoChat benchmark) did better, and that I under-invested in compilation and CUDA-kernel work. All three are correct. This document tries to say honestly why.

## The one-paragraph verdict

The campaign was **rigorous but boxed-in**. It executed clean statistics, caught a real confound (an fp8 contention artifact), and produced a well-documented +7.9% (val_bpb 1.0301 → 0.9506, best 0.9483) with a complete mechanistic account of its own floor. But almost all of that value was *methodological*, and almost all of the *performance* came in the first ~30 rounds. The remaining ~40 rounds were careful coordinate-descent and a string of failed bigger bets, executed inside a regime whose binding constraint — data starvation — I did not diagnose until round 65. I optimized the inside of a box very carefully without first measuring the box.

## What actually went right (briefly)

- **Statistical discipline near the noise floor.** σ ≈ 0.0015 on a contended shared cluster; I made GPU-controlled interleaved/swapped paired designs mandatory after catching that an fp8 "win" was pure GPU-placement noise. That retraction (paper 006) is the campaign's best negative result.
- **Independent rediscovery of the right big lever.** The n-gram hashed bigram/trigram value-embedding, mixed into the attention value path through learned gates, was my largest lever — and Recursive independently found the *same* mechanism. That convergence is real validation of the choice.
- **Mechanistic honesty at the end.** The last ten rounds decomposed the floor to its root: it is data-limited, and the data-sensitivity is carried *entirely* by the n-gram memory acting as a memorizer (off: range 0.004 across data; on: range 0.118). Even the flagship lever is operating-point-specific. That is a genuinely complete diagnosis — it just arrived 60 rounds too late to change the strategy.

## The three compounding failures

### 1. I diagnosed the regime last, not first.

The single cheapest, highest-value experiment in the entire campaign was a three-run data-scaling ablation. It showed a −0.118 swing from 5→10 shards — *larger than the entire +0.082 that 64 rounds of architecture search produced*. It also showed the model trains on 10 of 6,543 available shards (~2 epochs), sitting exactly on the overfitting boundary. I ran it at **round 65**. Had I run it at round 5, I would have known that architecture search was capped at ~+8% in this regime and behaved completely differently — either reframing around the data constraint or declining to grind. **The binding-constraint diagnostic must come first, because it determines whether any of the downstream work can pay.**

### 2. I chose isolation over recombination, which is a machine for producing increments.

My loop was: propose one lever, A/B it against the current best, adopt-or-reject, repeat. This is coordinate descent. It gives clean, publishable verdicts — and it is *structurally incremental*: it converges to the first fixed point and cannot see interactions that require moving several coordinates at once. Recursive did the opposite — they searched over *combinations* ("architecture, short-context memory, auxiliary losses, attention, optimizer behavior, weight decay schedules, compiler settings" changed together) and ran *many parallel threads*, keeping risky branches alive and merging winners. Isolation optimizes for *interpretability of each verdict*; recombination optimizes for *finding the best stack*. I over-indexed on the first. A single linear thread also meant every idea had to be killed or adopted immediately, biasing me toward small safe wins over held-open risky ones.

### 3. I diagnosed the kernel bottleneck and then never wrote the kernel.

I profiled with nsys and found the step is dominated by the n-gram value-embedding path — small GEMMs and ~194k tiny fill-kernel launches — *not* attention or CE. I wrote that down and then treated it as a characterization instead of a to-do. My only throughput actions were compile flags (`max-autotune`) and a failed `SparseAdam` attempt, after which I *wrongly declared throughput maxed*. The direct fix I never attempted is a **fused Triton kernel for the n-gram path** (rolling hash → table gather → gate → accumulate in one kernel), which is exactly what collapses those 194k launches. On attention I did nothing at all — no fp8 attention (Recursive kept the fp8 config I retracted), no per-layer window kernel. Every piece of "systems work" was a flag or an off-the-shelf kernel; I never dropped below `torch.compile`. *(Caveat, not excuse: in my data-starved regime more throughput → more epochs → more overfitting, so a kernel win would have partly converted into overfitting rather than lower loss. That is real — but it is another consequence of failure #1: I couldn't know whether kernel work would pay because I hadn't characterized the regime.)*

## The pattern connecting all three

Each failure is the same shape: **I did careful, rigorous work inside an unexamined assumption.** I assumed the architecture was the thing to optimize (it wasn't — data was), assumed one-lever verdicts composed (they don't — they invert with data), and assumed throughput was maxed (it wasn't — I just hadn't written the kernel). Rigor applied to the wrong frame produces confident, well-documented increments. The failure was not sloppiness; it was **not questioning the frame before optimizing within it.**

## What a good loop does differently (the Recursive contrast, generalized)

1. **Diagnose the binding constraint first.** Data slope, compute slope, and a quick kernel profile are all cheap. Run them at round 1. They tell you *which* axis is even worth searching.
2. **Search combinations, not coordinates.** Hold a population of branches; recombine promising ones; accept that individual verdicts get muddier in exchange for finding better stacks.
3. **Co-evolve the evaluator with the search.** Recursive's stated lesson — "as the search became stronger, the evaluator had to become stronger too" — is why they could *keep* fp8 and SGLD-noise wins that I would have retracted as noise. A strong-enough validity detector lets you trust bigger, weirder wins instead of only small safe ones.
4. **Treat systems work as first-class.** When you profile a bottleneck, the deliverable is a kernel, not a note. Throughput is a modeling lever in update-limited regimes.
5. **Know whether you are converged or just out of ideas.** My H8/H9 stopping rule is good, but it certifies "no lever in *this basis* helps" — it must be paired with a frame check ("is this basis even the right one?"), or it will confidently certify a floor that is really a boundary you drew.

## The one-sentence lesson

**Measure the box before you optimize inside it:** the cheapest experiment (a data slope, a compute slope, a kernel profile) that reveals the binding constraint is worth more than fifty careful rounds spent tuning an architecture whose ceiling is set by something you never checked.

*What I'd keep: the statistical discipline, the mechanistic honesty, the willingness to retract. What I'd change: run the frame diagnostics first, search combinations under a population, write the kernel I profiled, and trust a stronger evaluator enough to keep the risky wins.*
