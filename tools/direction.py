#!/usr/bin/env python3
"""THE EXPLORE/EXPLOIT POLICY, in code so it binds rather than merely advises.

Rewritten 2026-08-18 after auditing the 70-run v3 campaign, which failed in a way the
previous policy actively caused:

  * The old policy grouped several parameters into ONE capped axis. The cap was spent on
    two of them, so the third was CLOSED having never been varied once -- an axis with no
    coverage was treated as an axis with negative evidence.
  * The cap was a CEILING, not a COST. An axis that was still improving hit the cap and
    shut. The policy forbade exploiting the one direction that was demonstrably working.

The replacement has two rules and no magic numbers beyond a measured noise band:

  RULE 1 (coverage floor -- pure exploration).  An axis with zero runs can never be
  closed and always outranks everything else.  A knob you have not tried once is not a
  knob you have evidence about.

  RULE 2 (dry rule -- exploitation stays open while it pays).  An axis closes only after
  DRY_STREAK consecutive experiments on it that failed to beat the running best by more
  than the noise band.  Improvement keeps an axis open indefinitely; a cap never does.

Axes are per-PARAMETER, not per-family.  Family grouping is exactly what hid `dim`.
"""
from __future__ import annotations

# --- the reference platform: the strict upstream recipe, resolved -----------------
# train.py's dataclass defaults (n_layer=12, n_embd=768) are dead code; the resolved
# upstream configuration on this benchmark is depth 8 / dim 512 / mlp 4.
PLATFORM = {"dbs": 128, "tbs": 19, "depth": 8, "dim": 512,
            "mlp": 4, "ve": 2, "win": "SSSL", "swdiv": 2}

# One axis per tunable parameter. A config counts against every axis it moves, so
# stacking knobs cannot dodge accounting.
KNOB_AXES = (
    "depth", "dim", "mlp",          # capacity
    "tbs", "dbs",                   # tokens per step and micro-batch
    "win", "swdiv",                 # attention window
    "ve",                           # value-embedding placement
    "rope", "softcap", "x0init",    # signal scale
    "warmup", "wd_const", "mu_const", "ema", "ema_start",   # schedule shape
    "clip", "ns",                   # optimizer numerics
    "batch_ramp",                   # ramps grad_accum mid-run: moves tokens/step
    "compile_mode", "qk_suppress",  # expressible in make_variant and previously INVISIBLE
    "mu_warmup",                    # LENGTH of the step-indexed Muon momentum ramp
    "mu_ceil",                      # CEILING of that ramp; co-scales with batch size
)

# New causal structure: adds or removes an operation or an objective term. Not a new
# value of an existing constant. A mechanism must declare a mediator, an activation
# observable, a competing explanation and a falsifier.
MECHANISMS = ("mtp", "unet", "zloss", "noqknorm", "prefetch", "precond")

# The noise band is MEASURED, never assumed. A hardcoded constant here was a bug: set
# from the CROSS-WAVE spread of byte-identical controls, it was far wider than the WITHIN-WAVE
# resolution, so with DRY_STREAK=4 essentially no run could ever register as an
# improvement and every axis closed as "dry" -- including axes that were demonstrably
# still paying. That is the exact failure this policy exists to prevent, reintroduced
# through the threshold instead of through the cap.
#
# Cross-wave spread is a nuisance (host CPU contention moves step count run to run);
# within-wave spread is the real resolution. So the band comes from controls that ran
# CONCURRENTLY, and until at least two such controls exist there is NO instrument -- in
# which case no axis may be declared dry at all. Refusing to close an axis you cannot
# measure is the fail-safe direction.
# A "wave" means runs whose EXECUTION INTERVALS ACTUALLY OVERLAP, so they shared the same
# host contention. Proximity in time is not enough and using it was a bug: controls
# launched back-to-back on one GPU are sequential, not concurrent, and the host load
# drifts between them. Grouping those as a wave reports a sequential spread under a
# within-wave label -- flattering and wrong.
MIN_OVERLAP_S = 60              # intervals must share at least this much wall clock
DRY_STREAK = 4          # consecutive non-improving runs on an axis before it closes
EXPLORE_FLOOR = 0.35    # >=35% of launches must go to zero-coverage axes or mechanisms


KNOWN_KEYS = frozenset(KNOB_AXES) | frozenset(MECHANISMS) | frozenset(PLATFORM)


def unknown_keys(cfg: dict) -> set:
    """Config keys no policy rule covers. Never silently ignored."""
    return {k for k in (cfg or {}) if k not in KNOWN_KEYS}


def device_means(results: list[dict]) -> dict:
    """Mean val_bpb of the two-epoch CONTROL runs on each device. ONE implementation.

    Three copies of this existed -- here, in balance.py and in selector.py -- and they
    agreed only by luck. This project has already shipped divergent copies of a statistic
    once: direction.py and verdict.py carried different resolution formulas, and the
    disagreement decided verdicts. A device mean is the denominator of every
    device-corrected effect in the campaign, so three of them is three chances for the
    numbers in a paper to stop matching the numbers in a policy.
    """
    import statistics as _st
    by = {}
    for r in results:
        if (r.get("ok") and is_platform(r.get("cfg") or {})
                and (r.get("metrics") or {}).get("final_epoch") == 2.0):
            g = r.get("gpu")
            # An UNKNOWN device is not a device. Crash recovery used to stamp gpu:-1, and
            # pooling that as if it were real builds a mean out of runs that share nothing
            # but the fact that nobody recorded where they ran -- then corrects genuine
            # effects against it. The correction is the denominator of every device-
            # corrected number in this campaign, so a fictional device silently biases all
            # of them by roughly the device spread, 0.0025 bpb, which is larger than most
            # effects being chased.
            if g is None or g == -1:
                continue
            by.setdefault(g, []).append(r["metrics"]["val_bpb"])
    return {g: _st.mean(v) for g, v in by.items() if v}


def device_resolution(results: list[dict]) -> dict | None:
    """The smallest effect a GPU-counterbalanced comparison can honestly claim.

    Computed under the CORRECTED model: the offset belongs to the physical GPU, not the
    taskset core block (L020_the_offset_is_the_gpu_not_the_core_block). Within a single
    device the residual spread of byte-identical controls is the real random error; the
    across-device spread is a fixed profile that counterbalancing removes by construction.

    This number DRIFTS as foreign load on the host changes, which is why it is recomputed
    rather than pinned: the campaign has at various points quoted 0.000174, 0.000183 and
    0.000187 from the same formula at different n and under the wrong model.
    """
    import statistics as _st
    by = {}
    for r in results:
        if not (r.get("ok") and (r.get("metrics") or {}).get("val_bpb")):
            continue
        if not is_platform(r.get("cfg") or {}):
            continue
        if (r["metrics"].get("final_epoch") or 0) != 2.0:
            continue        # 1-epoch runs are a different operating point (L005_v2)
        by.setdefault(r.get("gpu"), []).append(r["metrics"]["val_bpb"])
    sds = {g: _st.stdev(v) for g, v in by.items() if len(v) > 2}
    if not sds:
        return None
    pooled = _st.mean(sds.values())
    # `resolution` is the threshold for a mean of FOUR paired differences, and that is
    # the only n at which the historical formula 2*pooled/sqrt(2) is correct. A paired
    # difference has Var = 2*sigma^2 under independence, so SD(delta) = sqrt(2)*sigma and
    # SE(mean of n) = sqrt(2)*sigma/sqrt(n). A two-sigma threshold is therefore
    # 2*sqrt(2)*sigma/sqrt(n), which coincides with 2*pooled/sqrt(2) at exactly n=4 and
    # nowhere else: at n=3 the old constant is 1.7*SE (too lenient) and at n=6 it is
    # 2.4*SE (too strict). The ve verdict was read at n=3 with one cell void, i.e. under
    # the lenient case. Callers must use resolution_at(n) with the n they actually have;
    # the fixed field is retained only so existing readers keep working and is documented
    # here as the n=4 special case rather than a universal constant.
    def _res_at(n):
        return 2 * (2 ** 0.5) * pooled / (n ** 0.5) if n else float("inf")
    return {"per_gpu_sd": sds, "pooled_sd": pooled,
            "resolution": _res_at(4),
            "resolution_at": _res_at,
            "sd_paired_diff": (2 ** 0.5) * pooled,
            "n_devices": len(sds),
            "device_means": {g: _st.mean(v) for g, v in by.items()}}


def slot_bias(results: list[dict]) -> dict | None:
    """The fixed offset between dispatcher slots, measured from concurrent controls.

    Slot is not a nuisance that averages out. Cores 96-107 and 108-119 are not
    interchangeable on this host: the slot-0 member of a wave lost to the slot-1 member
    in 7 of 7 control waves by a mean 0.000499 bpb, because its core block runs the
    frozen packing loop ~2.4ms/step slower and therefore completes ~9 fewer steps. That
    offset is 70% of the raw within-wave band, so an UNCOUNTERBALANCED yoked pair can
    manufacture or erase an effect of exactly the size worth chasing. See L006.

    Returns the paired offset and, more usefully, the residual sd once it is removed --
    which is the resolution a counterbalanced design actually achieves.
    """
    import statistics as _st
    waves = {}
    for r in results:
        if not (r.get("ok") and (r.get("metrics") or {}).get("val_bpb")):
            continue
        if not is_platform(r.get("cfg") or {}):
            continue
        w = (r.get("name") or "").split("_")[0]
        waves.setdefault(w, {})[f"gpu{r.get('gpu')}"] = r["metrics"]["val_bpb"]
    deltas = []
    for w, byslot in waves.items():
        if len(byslot) != 2:
            continue
        # Sort by the NUMERIC start of the core range, not the string: dispatch.py sets
        # cores = CORE_BASE + slot*CORES_PER_JOB, so slot order is numeric order, and a
        # lexicographic sort puts "108-119" before "96-107" and silently flips the sign
        # of the offset -- which is how this function first reported the bias backwards.
        slot0, slot1 = sorted(byslot)
        deltas.append(byslot[slot0] - byslot[slot1])
    if len(deltas) < 3:
        return None
    return {"n": len(deltas), "offset": _st.mean(deltas),
            "resid_sd": _st.stdev(deltas),
            "same_sign": sum(1 for d in deltas if d > 0),
            "devices": sorted({c for b in waves.values() for c in b})}


def _wave_width(results, wave):
    """How many of OUR runs shared that wave. Width is part of the operating point."""
    return sum(1 for r in results if (r.get("name") or "").split("_")[0] == wave)


def noise_band(results: list[dict]) -> tuple[float | None, str]:
    """(band, how_it_was_measured). None means: no instrument, so nothing may be closed."""
    import statistics as _st
    ctl = [r for r in results
           if r.get("ok") and (r.get("metrics") or {}).get("val_bpb")
           and is_platform(r.get("cfg") or {})]
    if len(ctl) < 2:
        return None, f"unmeasured ({len(ctl)} control runs; need 2+)"
    # Waves of different WIDTH are different operating points: four concurrent trainers
    # spread 0.002342 across slots where two spread 0.00047, because the per-slot penalty
    # grows with our own concurrency (L017_slot_offset_scales_with_our_own_concurrency).
    # Pooling them inflates the band with a between-width term and makes the instrument
    # look coarser than it is at either width. Report the band for the width that carries
    # the most control waves, and say which.
    _by_wave = {}
    for r in ctl:
        _by_wave.setdefault((r.get("name") or "").split("_")[0], []).append(r)
    _widths = {}
    for w, members in _by_wave.items():
        _widths.setdefault(len(members), []).extend(members)
    if len(_widths) > 1:
        _dom = max(_widths, key=lambda k: len(_widths[k]))
        ctl = _widths[_dom]
        _wnote = f"; {_dom}-wide waves only, other widths excluded (L017)"
    else:
        _wnote = ""
    # Prefer genuinely CONCURRENT controls: overlapping execution intervals mean the same
    # host contention acted on both, which is the resolution a yoked comparison achieves.
    def _iv(r):
        st = r.get("started") or 0.0
        en = r.get("ended") or 0.0
        return (st, en if en > st else st)

    def _overlap(a, b):
        (a0, a1), (b0, b1) = _iv(a), _iv(b)
        return min(a1, b1) - max(a0, b0) >= MIN_OVERLAP_S

    waves, used = [], set()
    for i, a in enumerate(ctl):
        if i in used:
            continue
        grp, idx = [a], [i]
        used.add(i)
        for j, b in enumerate(ctl):
            if j in used:
                continue
            if all(_overlap(b, ctl[k]) for k in idx):   # overlap every member, not just one
                grp.append(b)
                idx.append(j)
                used.add(j)
        if len(grp) >= 2:
            waves.append(grp)
    # SAMPLE stdev, never population. At n=2-3 pstdev underestimates the spread, which
    # would make the band too TIGHT and promote noise to "improvement". For a safety
    # threshold the error must point the conservative way.
    def _sd(vals):
        return _st.stdev(vals) if len(vals) >= 2 else 0.0

    within = [v for g in waves
              for v in [_sd([r["metrics"]["val_bpb"] for r in g])] if v > 0]
    if within:
        return 2.0 * (sum(within) / len(within)), \
               f"2x within-wave sample sd over {len(waves)} wave(s), {len(ctl)} controls"
    pooled = _sd([r["metrics"]["val_bpb"] for r in ctl])
    if pooled <= 0:
        return None, f"controls are identical to the digit ({len(ctl)} runs); no resolution"
    # No overlapping pair exists, so this is a SEQUENTIAL spread across drifting host
    # load. It is the honest number to compare an unpaired run against, and it is much
    # coarser than a yoked pair would achieve. Say so, so nobody mistakes it for one.
    return 2.0 * pooled, \
           (f"2x SEQUENTIAL sample sd, {len(ctl)} controls -- NO concurrent pair yet, so "
            f"this is a cross-wave spread, not a paired resolution")


def axes_touched(cfg: dict) -> set:
    """Every knob axis this config moves. Empty for the untouched platform baseline."""
    moved = set()
    for k in KNOB_AXES:
        if k in PLATFORM:
            if cfg.get(k, PLATFORM[k]) != PLATFORM[k]:
                moved.add(k)
        elif cfg.get(k) is not None:      # `is not None`, never truthiness: clip=0.0,
            moved.add(k)                  # wd_const=0 and ema_start=0 are real settings.
    return moved


def mechanisms_touched(cfg: dict) -> set:
    return {m for m in MECHANISMS if cfg.get(m) is not None}


def is_platform(cfg: dict) -> bool:
    """A pure control/anchor: moves no axis, engages no mechanism, and carries no key
    this policy does not understand.

    That last clause is the fix for a control-corrupting bug. `batch_ramp`,
    `compile_mode` and `qk_suppress` were expressible by make_variant but absent from
    KNOB_AXES, so a config carrying one satisfied "moves no known axis", was labelled
    `control`, bypassed the decision cutoff and every budget, and would have been pooled
    into the very control block used to measure the noise band -- silently destroying the
    instrument. Failing safe on ANY unrecognised key makes the whole class impossible,
    not just the three instances that were found.
    """
    return (not axes_touched(cfg) and not mechanisms_touched(cfg)
            and not unknown_keys(cfg))


def label(cfg: dict) -> str:
    ms, ax = sorted(mechanisms_touched(cfg)), sorted(axes_touched(cfg))
    if ms:
        return "mech:" + "+".join(ms) + (("|knob:" + "+".join(ax)) if ax else "")
    return ("knob:" + "+".join(ax)) if ax else "control"


def axis_state(results: list[dict]) -> dict:
    """Per-axis coverage and dry-streak, computed from completed results only.

    `results` are the raw result records: {"cfg": {...}, "metrics": {"val_bpb": float},
    "ok": bool}. Verdicts use RAW val_bpb, because raw val_bpb at 300 seconds IS the
    benchmark. A change that costs throughput really is worse; correcting it away
    answers a question nobody asked.
    """
    ok = [r for r in results if r.get("ok") and (r.get("metrics") or {}).get("val_bpb")]
    ok.sort(key=lambda r: r.get("ended") or 0)
    best = min((r["metrics"]["val_bpb"] for r in ok), default=float("inf"))

    band, _how = noise_band(results)
    # Device-corrected effect per run, for the holds-best exemption below. Keying that
    # exemption on RAW val_bpb was wrong: gpu7 is the fastest device and gpu4 the slowest,
    # a spread larger than any effect here, so an axis could hold the raw record purely by
    # landing on gpu7 -- exactly the device confound this campaign refuses to headline
    # anywhere else. Subtracting the device's own control mean removes it.
    import statistics as _st
    _dev = {}
    for r in results:
        if (r.get("ok") and is_platform(r.get("cfg") or {})
                and (r.get("metrics") or {}).get("final_epoch") == 2.0):
            _dev.setdefault(r.get("gpu"), []).append(r["metrics"]["val_bpb"])
    _devmean = {g: _st.mean(v) for g, v in _dev.items() if v}
    state = {a: {"n": 0, "dry": 0, "best": None, "since_best": 0,
                 "best_eff": None, "best_value": None} for a in KNOB_AXES}
    running_best = float("inf")
    for r in ok:
        v = r["metrics"]["val_bpb"]
        # With no measured instrument, no run can be judged non-improving, so no axis
        # can go dry. Uncertainty must not masquerade as a negative result.
        improved = True if band is None else v < running_best - band
        for a in axes_touched(r["cfg"] or {}):
            s = state[a]
            s["n"] += 1
            # The dry streak counts DISTINCT VALUES tried without improvement, not runs.
            # Counting runs was written for one-run-per-value experiments and quietly
            # became "one value per axis, ever" when the design moved to counterbalanced
            # quads: a quad is 4 runs of the SAME value, so DRY_STREAK=4 retired an axis
            # the instant its first value finished. Measured on this campaign, qk_suppress,
            # tbs and ve each closed having tried exactly ONE value -- and tbs closed on
            # tbs=20 being catastrophic (+0.022545), which is the strongest possible reason
            # to try the OTHER direction rather than to stop. Repeating a value is
            # replication and buys no new information about the axis, so it must not
            # advance the streak either.
            v_here = (r["cfg"] or {}).get(a)
            if improved:
                s["dry"] = 0
                s["_dry_vals"] = set()
            elif v_here not in s.setdefault("_dry_vals", set()):
                s["_dry_vals"].add(v_here)
                s["dry"] = len(s["_dry_vals"])
            eff = v - _devmean.get(r.get("gpu"), v)
            if s["best_eff"] is None or eff < s["best_eff"]:
                s["best_eff"] = eff
                s["best_value"] = (r["cfg"] or {}).get(a)
            if s["best"] is None or v < s["best"]:
                s["best"] = v
                s["since_best"] = 0
            else:
                s["since_best"] += 1
        running_best = min(running_best, v)

    for a, s in state.items():
        # RULE 1: zero coverage is never closed, and outranks everything.
        # RULE 1b: NEVER close the axis that currently holds the campaign's best result.
        # The dry test asks whether a run beat the GLOBAL running best by more than the
        # band, so an axis is charged a dry strike for failing to beat a record another
        # axis set. swdiv hit 0.989449 -- a new campaign best -- but the previous best
        # 0.989819 came from the ve axis, and clearing it by 0.00037 did not clear the
        # 0.00074 band; four such runs closed the axis that was, at that moment, winning.
        # Worse, this generalises: any single strong result drives every other axis dry,
        # so the policy would retire the whole search space on the strength of one number.
        # Holding the best is sufficient evidence that an axis is still paying.
        # ...but only while it became leader RECENTLY. An axis flat at the best since its
        # very first run would otherwise never close, which is the over-broad version of
        # this exemption and defeats the dry rule entirely. Leading is evidence an axis is
        # paying; leading from a record set long ago and nothing since is evidence it has
        # stopped. `since_best` counts runs on this axis after its own best.
        best_eff = min((t["best_eff"] for t in state.values()
                        if t["best_eff"] is not None), default=None)
        holds_best = (s["best_eff"] is not None and best_eff is not None
                      and s["best_eff"] <= best_eff
                      and s["since_best"] < DRY_STREAK)
        s["open"] = True if s["n"] == 0 else (s["dry"] < DRY_STREAK or holds_best)
        s["priority"] = 2 if s["n"] == 0 else (1 if s["open"] else 0)
    return {"axes": state, "best": best, "n_ok": len(ok),
            "band": band, "band_how": _how}


def blocked_reason(cfg: dict, state: dict) -> str | None:
    """First reason this config may not launch, or None.

    Controls are always allowed. Mechanisms tested in isolation are always allowed --
    a mechanism is how you leave a local optimum, and the campaign's own evidence is
    that knob space here is nearly exhausted. A mechanism crossed with a CLOSED knob
    axis is refused: evaluate the new structure alone before entangling it with a
    tuned architecture you can no longer attribute.
    """
    if is_platform(cfg):
        return None
    ax = axes_touched(cfg)

    # EXPLOITATION is not exploration, and the dry rule must not conflate them. An axis
    # closes after DRY_STREAK runs that did not beat the running best -- but the runs that
    # CONFIRM a lever are exactly those runs, so a knob could be proven to work and then
    # be permanently unusable. That happened here: ve=1 is a confirmed -0.001487 win at
    # t=-12.2, and the four runs that established it closed the ve axis, after which
    # {"ve":1,"swdiv":4} was refused as dry and the precond+ve+swdiv stack was refused as
    # "a mechanism crossed with a closed axis". The campaign could measure a win and never
    # spend it.
    #
    # So: reusing an axis's OWN BEST-KNOWN VALUE is exploitation and stays allowed even
    # when the axis is closed to further exploration. Trying a DIFFERENT value of a closed
    # axis is exploration and stays refused, which is what the dry rule is actually for.
    def _exploring(a):
        bv = state["axes"][a].get("best_value")
        return bv is None or cfg.get(a) != bv

    if mechanisms_touched(cfg):
        shut = [a for a in sorted(ax)
                if not state["axes"][a]["open"] and _exploring(a)]
        return (f"mechanism crossed with closed axis {shut[0]} at a value that is not its "
                f"best ({cfg.get(shut[0])!r} vs best "
                f"{state['axes'][shut[0]].get('best_value')!r}); test it in isolation first"
                if shut else None)
    shut = [a for a in sorted(ax)
            if not state["axes"][a]["open"] and _exploring(a)]
    if shut:
        a = shut[0]
        _b = state.get("band")
        _bs = f">{_b:.5f}" if _b else "measurable"
        return (f"axis '{a}' is dry: {state['axes'][a]['dry']} consecutive runs without a "
                f"{_bs} improvement. Propose a mechanism or an unexplored axis.")
    return None


def explore_debt(results: list[dict], state: dict) -> float:
    """How far below EXPLORE_FLOOR the campaign is running. >0 means the next launch
    should be exploration. Exploitation drifts into a monoculture without this."""
    # CONTROLS ARE NOT A RESEARCH CHOICE, so they belong in neither term. They were in the
    # denominator and never in the numerator: a control touches no axis and no mechanism, so
    # it can never count as exploration, and 62% of valid runs are controls. The metric was
    # therefore biased toward "explore more" by the runs that are the INSTRUMENT rather than
    # a decision between exploring and exploiting.
    #
    # It also made this tool disagree permanently with tools/balance.py, which counts
    # EXPERIMENTS. An audit called that out as instrument-shopping -- two numbers for one
    # question, letting whichever flatters be quoted. They are now measuring the same
    # population, and any remaining gap is unit (runs vs experiments), not definition.
    ok = [r for r in results if r.get("ok") and not is_platform(r.get("cfg") or {})]
    if not ok:
        return 1.0
    exp = sum(1 for r in ok
              if mechanisms_touched(r["cfg"] or {})
              or any(state["axes"][a]["n"] <= 1 for a in axes_touched(r["cfg"] or {})))
    return EXPLORE_FLOOR - exp / len(ok)


# --- the DIRECTION SPACE ------------------------------------------------------------
# Axis coverage alone does not answer "what have we never looked at?". The previous
# campaign kept an entire campaign inside two families and nothing reported that fact, because a
# family with no axes ever proposed simply does not appear in an axis table. Families
# are listed here explicitly so a zero-coverage FAMILY is visible even when no config
# in it was ever written.
#
# `cost` is the fixed-time cost class on THIS benchmark, where ~90% of the 300s is the
# frozen single-threaded packing loop. It is the first thing a proposal must be argued
# against, so it is data rather than prose in a document.
FAMILIES = {
    "capacity":        {"axes": ("depth", "dim", "mlp"),
                        "cost": "GPU work is NOT free: mfu ~42.6%, so the GPU is substantially busy and added FLOPs cost step count (L019). Zero-FLOP capacity beats FLOP-buying capacity."},
    "token_exposure":  {"axes": ("tbs", "dbs", "batch_ramp"),
                        "cost": "CPU-PROPORTIONAL: fewer tokens/step => less packing => more steps"},
    "attention":       {"axes": ("win", "swdiv"),
                        "cost": "GPU-side; NOT free at mfu ~42.6% (L019). Removing FLOPs returns little -- L011 measured 40% fewer Newton-Schulz iterations buying 0.2% of step."},
    "ve_placement":    {"axes": ("ve",),
                        "cost": "the one ZERO-FLOP capacity lever: value embeddings are 16.78M of 50.33M params (33%) at vocab 8192 and are excluded from flops_per_token (L015, L019)."},
    "signal_scale":    {"axes": ("rope", "softcap", "x0init"), "cost": "free"},
    "schedule":        {"axes": ("warmup", "wd_const", "mu_const", "ema", "ema_start"),
                        "cost": "free (ema costs one fp32 shadow + eval swap)"},
    "optimizer_numeric": {"axes": ("clip", "ns"), "cost": "small GPU cost"},
    "systems":         {"axes": ("compile_mode",),
                        "cost": "pure throughput; the most wave-confounded thing to measure"},
    "attention_detail": {"axes": ("qk_suppress",),
                        "cost": "GPU-side; NOT free at mfu ~42.6% (L019)."},
}

# Mechanism families. `prepare.py` is frozen, so data selection/curriculum is OUT OF
# SCOPE by construction -- it is listed with an empty tuple so the gap is visible and
# nobody spends a round rediscovering that it is unreachable.
MECHANISM_FAMILIES = {
    "objective":       {"mechs": ("mtp", "zloss"), "cost": "GPU + memory; mtp materializes [B,T,V]"},
    "optimizer_numeric": {"mechs": ("precond",),
                        "cost": "free: a reorder of existing blocks, no new tensor and no "
                                "added op, so the compiled fused update stays intact"},
    "signal_path":     {"mechs": ("unet", "noqknorm"), "cost": "GPU-only, near-free"},
    # `precond` reorders an operation that already exists rather than retuning a
    # constant, so it is a MECHANISM, not a knob -- which is what makes it runnable while
    # every knob axis in this family is still unexplored (a mechanism tested in isolation
    # is always allowed; a mechanism crossed with a closed knob axis is not).
    # NOTE: this key is NOT in lit.ALL_FAMILIES, so agenda.py -- which iterates that
    # tuple -- never listed it and never counted its runs. All seven precond runs were
    # attributed here, invisible to DRY, STALE and the HARD CAP, so the anti-monoculture
    # backstop could not fire on the direction the campaign actually spent itself on.
    # Mechanism families must use a name the agenda knows; `precond` belongs to
    # optimizer_numeric, which is where its claims and its lessons already sit.
    "optimizer_geometry_RETIRED": {"mechs": (),
                        "cost": "free: a pure reorder of existing blocks, no new tensor "
                                "and no added op, so the compiled fused update is intact"},
    "input_pipeline":  {"mechs": ("prefetch",),
                        "cost": "the ~10%-GPU-share premise is RETIRED (L019): mfu is ~42.6%, so the loader is overlapped with a busy GPU and the slack available to reclaim is much smaller than loader_frac suggests"},
    "data_curriculum": {"mechs": (),
                        "cost": "OUT OF SCOPE: lives in the frozen prepare.py"},
}


def mechanism_state(results: list[dict]) -> dict:
    """Per-mechanism run counts. Untracked mechanisms were invisible to the old policy."""
    ok = [r for r in results if r.get("ok") and (r.get("metrics") or {}).get("val_bpb")]
    st = {m: {"n": 0, "best": None} for m in MECHANISMS}
    for r in ok:
        v = r["metrics"]["val_bpb"]
        for m in mechanisms_touched(r["cfg"] or {}):
            st[m]["n"] += 1
            if st[m]["best"] is None or v < st[m]["best"]:
                st[m]["best"] = v
    return st


def direction_space(results: list[dict]) -> list[dict]:
    """Every intervention family with its coverage, so an untouched family is visible."""
    ax, me = axis_state(results)["axes"], mechanism_state(results)
    out = []
    for fam, spec in FAMILIES.items():
        n = sum(ax[a]["n"] for a in spec["axes"] if a in ax)
        virgin = [a for a in spec["axes"] if a in ax and ax[a]["n"] == 0]
        live = [a for a in spec["axes"] if a in ax and ax[a]["open"] and ax[a]["n"]]
        out.append({"family": fam, "kind": "knob", "n": n, "cost": spec["cost"],
                    "unexplored_axes": virgin, "open_axes": live})
    for fam, spec in MECHANISM_FAMILIES.items():
        n = sum(me[m]["n"] for m in spec["mechs"] if m in me)
        virgin = [m for m in spec["mechs"] if m in me and me[m]["n"] == 0]
        out.append({"family": fam, "kind": "mechanism", "n": n, "cost": spec["cost"],
                    "unexplored_axes": virgin, "open_axes": []})
    out.sort(key=lambda d: (d["n"], d["family"]))
    return out


def report(results: list[dict]) -> str:
    st = axis_state(results)
    _b = st.get("band")
    _bt = f"{_b:.5f} ({st.get('band_how')})" if _b else f"NONE - {st.get('band_how')}"
    lines = [f"best val_bpb {st['best']:.6f} over {st['n_ok']} valid runs "
             f"| noise band {_bt} | explore debt {explore_debt(results, st):+.2f}",
             f"{'axis':12s} {'n':>3s} {'dry':>4s} {'best':>10s}  status"]
    for a, s in sorted(st["axes"].items(), key=lambda kv: (-kv[1]["priority"], kv[0])):
        tag = ("UNEXPLORED (top priority)" if s["n"] == 0
               else ("open" if s["open"] else "closed (dry)"))
        bestcol = f"{s['best']:.6f}" if s["best"] is not None else "-"
        lines.append(f"{a:12s} {s['n']:3d} {s['dry']:4d} {bestcol:>10s}  {tag}")

    ms = mechanism_state(results)
    lines += ["", f"{'mechanism':12s} {'n':>3s} {'best':>10s}  status"]
    for m, v in sorted(ms.items(), key=lambda kv: (kv[1]["n"], kv[0])):
        b = f"{v['best']:.6f}" if v["best"] is not None else "-"
        lines.append(f"{m:12s} {v['n']:3d} {b:>10s}  "
                     f"{'NEVER TRIED' if v['n'] == 0 else 'tested'}")

    # The MEASURED operating point, recomputed every time, because it moved once already
    # and silently: the same byte-identical control ran 621-815 steps at one epoch under
    # heavy host load and 1010-1021 steps at TWO epochs under light load. A cost model
    # written down in prose ("~90% of the 300s is the packing loop") describes whichever
    # regime it was measured in, and a council round that reads the prose instead of the
    # runs will price every proposal against a machine it is not using. See lesson
    # L005_operating_point_moved.
    ok = [r for r in results if r.get("ok") and (r.get("metrics") or {}).get("val_bpb")]
    if ok:
        st_ = [r["metrics"].get("num_steps", 0) for r in ok]
        ep = sorted({r["metrics"].get("final_epoch") for r in ok
                     if r["metrics"].get("final_epoch") is not None})
        lf = [r["metrics"]["loader_frac"] for r in ok if r["metrics"].get("loader_frac")]
        lines += ["", "MEASURED OPERATING POINT (recomputed from results, not assumed)",
                  f"  steps {min(st_):.0f}-{max(st_):.0f}   final_epoch {ep}"]
        if lf:
            lines.append(f"  loader_frac {min(lf):.3f}-{max(lf):.3f}  "
                         f"-- CPU wall-clock share around ASYNC cuda launches, not GPU idle")
        if len(ep) > 1:
            lines.append("  WARNING: runs in this set finished at DIFFERENT epoch counts; "
                         "they are not one operating point and must not be pooled.")

    dr = device_resolution(results)
    if dr:
        lines += ["", "DEVICE MODEL (the offset is the GPU, not the core block -- L020)",
                  "  per-GPU control means: " + ", ".join(
                      f"gpu{g} {m:.6f}" for g, m in sorted(dr["device_means"].items())),
                  "  within-GPU sd: " + ", ".join(
                      f"gpu{g} {v:.6f}" for g, v in sorted(dr["per_gpu_sd"].items())),
                  f"  pooled within-GPU sd {dr['pooled_sd']:.6f}"
                  f"  =>  GPU-COUNTERBALANCED RESOLUTION {dr['resolution']:.6f}",
                  "  Counterbalance on GPU at ANY width; there is no width penalty."]

    # The slot-order prescription that stood here was superseded by L020: the offset is
    # the physical GPU, not the taskset core block, so "swap the slot order" pointed at a
    # variable carrying no signal. Printing it beside the DEVICE MODEL block gave a reader
    # two contradictory prescriptions and two different resolutions, both looking current.
    # slot_bias() is kept for the historical record but is no longer reported.


    lines += ["", "DIRECTION SPACE (least-covered first -- this is where a round should look)",
              f"{'family':17s} {'kind':10s} {'runs':>5s}  gaps / cost"]
    for d in direction_space(results):
        gap = (f"UNTOUCHED: {', '.join(d['unexplored_axes'])}"
               if d["unexplored_axes"] else
               (f"open: {', '.join(d['open_axes'])}" if d["open_axes"] else "all closed"))
        lines.append(f"{d['family']:17s} {d['kind']:10s} {d['n']:5d}  {gap}")
        lines.append(f"{'':17s} {'':10s} {'':5s}  cost: {d['cost']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import glob, json, pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    res = [json.loads(pathlib.Path(f).read_text())
           for f in glob.glob(str(root / "runs" / "sweep" / "results" / "*.json"))]
    print(report(res))


def step_law(results: list[dict], tokens_per_step: float = 524288.0):
    """Fit val_bpb against log(steps) on CONTROLS ONLY, at ONE tokens-per-step.

    The campaign quoted a step law of -0.05974 bpb per e-fold for hours, in prose, with no
    code behind it -- so nobody could check what it was fitted on. It was used to argue that
    two mechanisms failed for a common cause (L053/L054), and an audit was right to call the
    provenance unverifiable: a fit pooled across 262144/524288/1048576 tokens per step would
    violate this project's own rule that fit and application must share an operating point,
    and one that included the judged arms would be circular.

    This settles it by construction. Controls only, so no treatment can influence the law
    that judges it. One tokens-per-step, so the operating point is shared. The refit gives
    -0.06822 over 84 controls, close to L006's -0.0687 and NOT the -0.05974 that was being
    quoted; under it, step count explains 86% of MTP's damage and 56% of z-loss's rather
    than the 75% and 49% that L054 claimed.

    Returns (slope, n, lo_steps, hi_steps, resid_sd). The step RANGE is returned because it
    is a real limit: the controls span 621-1020 steps and the arms this was applied to ran
    303-372, so that application is an EXTRAPOLATION below the fitted range. Callers must
    say so rather than quietly reading off a number.
    """
    import math as _m
    import statistics as _st
    xs, ys = [], []
    for r in results:
        m = r.get("metrics") or {}
        if not r.get("ok") or not is_platform(r.get("cfg") or {}):
            continue
        if m.get("tokens_per_step") != tokens_per_step:
            continue
        if not m.get("num_steps") or m.get("val_bpb") is None:
            continue
        xs.append(_m.log(m["num_steps"]))
        ys.append(m["val_bpb"])
    if len(xs) < 8:
        return None
    mx, my = _st.mean(xs), _st.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    return {"slope": slope, "n": len(xs),
            "lo_steps": _m.exp(min(xs)), "hi_steps": _m.exp(max(xs)),
            "resid_sd": _st.stdev(resid) if len(resid) > 1 else 0.0}


def step_law_explains(results: list[dict], treat: dict, ctrl: dict):
    """Share of a paired delta the step law accounts for, with the extrapolation flagged.

    Returns None when the two arms do not share a tokens-per-step, because applying the law
    across that boundary is forbidden here and returning a number anyway is how a rule
    written in prose gets ignored in practice.
    """
    import math as _m
    tm, cm = treat.get("metrics") or {}, ctrl.get("metrics") or {}
    if tm.get("tokens_per_step") != cm.get("tokens_per_step"):
        return None
    law = step_law(results, tm.get("tokens_per_step"))
    if not law or not tm.get("num_steps") or not cm.get("num_steps"):
        return None
    ef = _m.log(cm["num_steps"] / tm["num_steps"])
    pred = -law["slope"] * ef
    act = tm["val_bpb"] - cm["val_bpb"]
    return {"efolds": ef, "predicted": pred, "actual": act, "residual": act - pred,
            "share": (pred / act) if act else float("nan"),
            "extrapolated": tm["num_steps"] < law["lo_steps"]
                            or tm["num_steps"] > law["hi_steps"],
            "law": law}
