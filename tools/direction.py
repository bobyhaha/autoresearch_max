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
)

# New causal structure: adds or removes an operation or an objective term. Not a new
# value of an existing constant. A mechanism must declare a mediator, an activation
# observable, a competing explanation and a falsifier.
MECHANISMS = ("mtp", "unet", "zloss", "noqknorm", "prefetch")

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


def noise_band(results: list[dict]) -> tuple[float | None, str]:
    """(band, how_it_was_measured). None means: no instrument, so nothing may be closed."""
    import statistics as _st
    ctl = [r for r in results
           if r.get("ok") and (r.get("metrics") or {}).get("val_bpb")
           and is_platform(r.get("cfg") or {})]
    if len(ctl) < 2:
        return None, f"unmeasured ({len(ctl)} control runs; need 2+)"
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
    state = {a: {"n": 0, "dry": 0, "best": None} for a in KNOB_AXES}
    running_best = float("inf")
    for r in ok:
        v = r["metrics"]["val_bpb"]
        # With no measured instrument, no run can be judged non-improving, so no axis
        # can go dry. Uncertainty must not masquerade as a negative result.
        improved = True if band is None else v < running_best - band
        for a in axes_touched(r["cfg"] or {}):
            s = state[a]
            s["n"] += 1
            s["dry"] = 0 if improved else s["dry"] + 1
            if s["best"] is None or v < s["best"]:
                s["best"] = v
        running_best = min(running_best, v)

    for a, s in state.items():
        # RULE 1: zero coverage is never closed, and outranks everything.
        s["open"] = True if s["n"] == 0 else s["dry"] < DRY_STREAK
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
    if mechanisms_touched(cfg):
        shut = [a for a in sorted(ax) if not state["axes"][a]["open"]]
        return (f"mechanism crossed with closed axis {shut[0]}; test it in isolation first"
                if shut else None)
    shut = [a for a in sorted(ax) if not state["axes"][a]["open"]]
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
    ok = [r for r in results if r.get("ok")]
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
                        "cost": "GPU-only, near-free (GPU idle ~90% of the run)"},
    "token_exposure":  {"axes": ("tbs", "dbs", "batch_ramp"),
                        "cost": "CPU-PROPORTIONAL: fewer tokens/step => less packing => more steps"},
    "attention":       {"axes": ("win", "swdiv"), "cost": "GPU-only, near-free"},
    "ve_placement":    {"axes": ("ve",), "cost": "GPU-only, near-free"},
    "signal_scale":    {"axes": ("rope", "softcap", "x0init"), "cost": "free"},
    "schedule":        {"axes": ("warmup", "wd_const", "mu_const", "ema", "ema_start"),
                        "cost": "free (ema costs one fp32 shadow + eval swap)"},
    "optimizer_numeric": {"axes": ("clip", "ns"), "cost": "small GPU cost"},
    "systems":         {"axes": ("compile_mode",),
                        "cost": "pure throughput; the most wave-confounded thing to measure"},
    "attention_detail": {"axes": ("qk_suppress",), "cost": "GPU-only, near-free"},
}

# Mechanism families. `prepare.py` is frozen, so data selection/curriculum is OUT OF
# SCOPE by construction -- it is listed with an empty tuple so the gap is visible and
# nobody spends a round rediscovering that it is unreachable.
MECHANISM_FAMILIES = {
    "objective":       {"mechs": ("mtp", "zloss"), "cost": "GPU + memory; mtp materializes [B,T,V]"},
    "signal_path":     {"mechs": ("unet", "noqknorm"), "cost": "GPU-only, near-free"},
    "input_pipeline":  {"mechs": ("prefetch",),
                        "cost": "attacks the actual bottleneck; overlap is bounded by the ~10% GPU share"},
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
