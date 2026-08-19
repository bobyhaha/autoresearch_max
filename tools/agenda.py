#!/usr/bin/env python3
"""WHICH DIRECTION TO WORK ON NOW, and when to abandon it.

This is the explore/exploit controller. It exists because an earlier campaign had no such
thing: it picked a corner of knob space and stayed there, and the only "policy" was a
per-family cap that shut axes at zero coverage while forbidding further work on the one
axis that was still improving. Both halves were backwards.

The controller holds ONE active direction at a time and answers two questions:

  ENTRY   Which direction has the evidence and the headroom to be worth working on?
          A direction with no read literature behind it cannot become active. That rule
          is what forces the corpus to be built before experiments start, and it is the
          fix for a campaign that made zero literature searches in an entire campaign.

  EXIT    When has this direction been exploited enough? Three independent triggers, any
          one of which forces rotation to a different direction:

            1. DRY        -- DRY_STREAK runs in this family with no improvement over the
                             running best by more than the measured noise band.
            2. STALE      -- MAX_RUNS_SINCE_IMPROVEMENT runs since its last real gain,
                             even if some axis inside it is still untouched. Untouched is
                             not the same as promising.
            3. HARD CAP   -- MAX_CONSECUTIVE_RUNS in one family, unconditionally. This is
                             the anti-monoculture backstop: spending a whole campaign inside
                             one or two families is exactly what it prevents.

          On exit the direction goes on COOLDOWN and cannot be re-entered until every
          other eligible direction has had a turn, or until NEW literature arrives for it
          (unread full texts, or fresh usable claims). Evidence reopens a door that
          exhaustion closed; wishing does not.

    agenda.py            # the decision, with the arithmetic shown
    agenda.py --json
    agenda.py --force capacity     # operator override, recorded with a reason
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import claims       # noqa: E402
import direction    # noqa: E402
import lit          # noqa: E402

STATE = REPO / "lit" / "ACTIVE_DIRECTION.json"
RESULTS = REPO / "runs" / "sweep" / "results"

DRY_STREAK = direction.DRY_STREAK          # 4
MAX_RUNS_SINCE_IMPROVEMENT = 6
MAX_CONSECUTIVE_RUNS = 12
MIN_USABLE_CLAIMS = 1        # entry gate: at least one transferable supporting claim

# Fixed-time cost classes, from the measured fact that ~90% of the 300s is the frozen
# CPU packing loop. A GPU-only intervention is nearly free; a CPU-costing one is nearly
# fatal. This is a prior on where to look, not a verdict.
COST_BONUS = {
    "capacity": 1.0, "attention": 1.0, "ve_placement": 1.0, "signal_path": 1.0,
    "signal_scale": 0.8, "schedule": 0.8, "optimizer_numeric": 0.5,
    "objective": 0.3, "token_exposure": 1.0, "input_pipeline": 1.0,
    "systems": 0.9, "attention_detail": 1.0,
}


def load_results() -> list[dict]:
    out = []
    if RESULTS.is_dir():
        for f in RESULTS.glob("*.json"):
            try:
                out.append(json.loads(f.read_text()))
            except (OSError, ValueError):
                continue
    return out


def families_of(cfg: dict) -> set:
    """Every direction family this config touches."""
    fams = set()
    ax, me = direction.axes_touched(cfg or {}), direction.mechanisms_touched(cfg or {})
    for fam, spec in direction.FAMILIES.items():
        if ax & set(spec["axes"]):
            fams.add(fam)
    for fam, spec in direction.MECHANISM_FAMILIES.items():
        if me & set(spec["mechs"]):
            fams.add(fam)
    return fams


def _counterbalanced_wins(results: list[dict]) -> set:
    """Families with a same-GPU paired mean that beats the resolution.

    Pairs each treatment against a control that ran on the SAME device, which removes the
    device offset inside every difference instead of relying on it to cancel in a mean.
    """
    import statistics as _st
    dr = direction.device_resolution(results)
    if not dr:
        return set()
    res = dr["resolution"]
    ok = [r for r in results if r.get("ok") and (r.get("metrics") or {}).get("val_bpb")
          and (r["metrics"].get("final_epoch") or 0) == 2.0]
    by_cfg = {}
    for r in ok:
        if direction.is_platform(r.get("cfg") or {}):
            continue
        key = json.dumps(r.get("cfg") or {}, sort_keys=True)
        by_cfg.setdefault(key, {}).setdefault(r.get("gpu"), []).append(r["metrics"]["val_bpb"])
    ctl = {}
    for r in ok:
        if direction.is_platform(r.get("cfg") or {}):
            ctl.setdefault(r.get("gpu"), []).append(r["metrics"]["val_bpb"])
    wins = set()
    for key, bygpu in by_cfg.items():
        deltas = [_st.mean(v) - _st.mean(ctl[g]) for g, v in bygpu.items() if g in ctl]
        if len(deltas) >= 2 and _st.mean(deltas) < -res:
            wins |= families_of(json.loads(key))
    return wins


def family_runs(results: list[dict]) -> dict:
    """Per family: total runs, runs since its last real improvement, dry streak."""
    ok = [r for r in results
          if r.get("ok") and (r.get("metrics") or {}).get("val_bpb")]
    ok.sort(key=lambda r: r.get("ended") or 0)
    band, _ = direction.noise_band(results)
    st = {f: {"runs": 0, "since_improve": 0, "dry": 0, "best": None}
          for f in lit.ALL_FAMILIES}
    # A family also counts as having improved if a COUNTERBALANCED comparison inside it
    # cleared the resolution. The raw-best test below compares single runs, and a single
    # run's val_bpb is dominated by which GPU it landed on: the device spread is about
    # 0.0025 against a band of 0.00066 (L020_the_offset_is_the_gpu_not_the_core_block).
    # So a real effect measured properly -- precond at -0.001147 with same-GPU paired
    # t = -12.3 -- did not register, because the best control already sat on the fastest
    # device and the treatment's best beat it by less than the band. The policy was about
    # to rotate away from the only direction that has produced a win. Raw val_bpb remains
    # the verdict; what changes is that the comparison is made device-to-device rather
    # than between whichever two runs happened to land on the luckiest hardware.
    verdict_families = _counterbalanced_wins(results)
    running_best = float("inf")
    for r in ok:
        v = r["metrics"]["val_bpb"]
        improved = True if band is None else v < running_best - band
        if not improved and (families_of(r.get("cfg") or {}) & verdict_families):
            improved = True
        for f in families_of(r.get("cfg") or {}):
            if f not in st:
                st[f] = {"runs": 0, "since_improve": 0, "dry": 0, "best": None}
            s = st[f]
            s["runs"] += 1
            s["since_improve"] = 0 if improved else s["since_improve"] + 1
            s["dry"] = 0 if improved else s["dry"] + 1
            if s["best"] is None or v < s["best"]:
                s["best"] = v
        running_best = min(running_best, v)
    return st


def family_virgin(fam: str, axst: dict, mechst: dict) -> list:
    """Untried levers inside a family: axes for knob families, mechanisms for mechanism
    families. Computing this only from axes left mechanism families looking exhausted
    the moment one of their mechanisms ran once."""
    axes = direction.FAMILIES.get(fam, {}).get("axes", ())
    mechs = direction.MECHANISM_FAMILIES.get(fam, {}).get("mechs", ())
    return ([a for a in axes if axst.get(a, {}).get("n", 0) == 0]
            + [m for m in mechs if mechst.get(m, {}).get("n", 0) == 0])


def family_levers(fam: str) -> int:
    return (len(direction.FAMILIES.get(fam, {}).get("axes", ()))
            + len(direction.MECHANISM_FAMILIES.get(fam, {}).get("mechs", ())))


def exit_trigger(fam: str, runs: dict, axst: dict, mechst: dict) -> str | None:
    """Why this direction should be abandoned now, or None to keep exploiting it."""
    s = runs.get(fam) or {"runs": 0, "since_improve": 0, "dry": 0}
    if s["runs"] == 0:
        return None
    if s["runs"] >= MAX_CONSECUTIVE_RUNS:
        return (f"HARD CAP: {s['runs']} runs in this family "
                f"(max {MAX_CONSECUTIVE_RUNS}) - rotate regardless of promise")
    if s["since_improve"] >= MAX_RUNS_SINCE_IMPROVEMENT:
        return (f"STALE: {s['since_improve']} runs since the last measurable gain "
                f"- exploited enough")
    if s["dry"] >= DRY_STREAK and not family_virgin(fam, axst, mechst):
        return (f"DRY: {s['dry']} consecutive runs with no gain and no untouched axis left")
    return None


def candidates(results: list[dict], litst: dict) -> list[dict]:
    axst = direction.axis_state(results)["axes"]
    mechst = direction.mechanism_state(results)
    runs = family_runs(results)
    out = []
    for fam in lit.ALL_FAMILIES:
        L = litst.get(fam, {})
        s = runs.get(fam) or {"runs": 0, "since_improve": 0, "dry": 0, "best": None}
        virgin = family_virgin(fam, axst, mechst)
        levers = family_levers(fam)
        gap = (len(virgin) / levers) if levers else (1.0 if not s["runs"] else 0.0)
        trigger = exit_trigger(fam, runs, axst, mechst)

        # ENTRY GATE: literature first. No read evidence => not eligible to be active.
        eligible, blocked = True, ""
        if L.get("usable", 0) < MIN_USABLE_CLAIMS:
            eligible = False
            blocked = (f"needs literature: {L.get('usable',0)} usable claims "
                       f"(fetched {L.get('fetched',0)}, unread {L.get('unread',0)}). "
                       f"Run: lit.py fetch --family {fam}; then extract claims.")
        elif trigger:
            eligible = False
            blocked = trigger

        # Score. Evidence and headroom, tilted by the cost class, penalised by spend.
        score = (1.20 * min(L.get("usable", 0), 6) / 6.0
                 + 0.60 * (L.get("mean_transfer", 0.0) / 4.0)
                 + 1.00 * gap
                 + 0.50 * COST_BONUS.get(fam, 0.5)
                 - 0.60 * min(s["runs"], 12) / 12.0
                 + 0.40 * (1.0 if L.get("unread", 0) > 0 else 0.0))
        out.append({"family": fam, "score": round(score, 3), "eligible": eligible,
                    "blocked": blocked, "exit_trigger": trigger,
                    "runs": s["runs"], "since_improve": s["since_improve"],
                    "dry": s["dry"], "best": s["best"], "gap": round(gap, 2),
                    "virgin_levers": virgin, "usable_claims": L.get("usable", 0),
                    "unread": L.get("unread", 0), "fetched": L.get("fetched", 0)})
    out.sort(key=lambda d: (-d["eligible"], -d["score"]))
    return out


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except ValueError:
            pass
    return {"active": None, "since": 0, "cooldown": [], "history": []}


def decide(force: str | None = None, commit: bool = True,
           reason: str | None = None) -> dict:
    results = load_results()
    litst = claims.literature_state()
    cands = candidates(results, litst)
    by = {c["family"]: c for c in cands}
    st = load_state()
    active, cooldown = st.get("active"), list(st.get("cooldown") or [])

    if force:
        # A forced rotation is the one decision here with no evidential trigger behind
        # it, so it is the one that most needs its justification written down. Without a
        # reason the ledger records only that someone overrode the controller, which is
        # indistinguishable from drift a month later -- and this campaign already had a
        # rotation happen DE FACTO, by queueing another family's runs while the ledger
        # still named the old direction. Default to the boilerplate so the call site is
        # not broken, but let a caller say why.
        why = reason or f"operator override to '{force}'"
        st.update({"active": force, "since": time.time(), "reason": why})
        st.setdefault("history", []).append(
            {"family": force, "at": time.time(), "reason": why})
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(st, indent=1))
        return {"active": force, "reason": st["reason"], "candidates": cands,
                "rotated": True, "cooldown": cooldown}

    rotated, reason = False, ""
    if active and by.get(active, {}).get("exit_trigger"):
        reason = f"'{active}' exhausted -- {by[active]['exit_trigger']}"
        if active not in cooldown:
            cooldown.append(active)
        active, rotated = None, True

    # New literature reopens a cooled direction: exhaustion was about evidence, and the
    # evidence changed.
    reopened = [f for f in cooldown if by.get(f, {}).get("unread", 0) > 0]
    for f in reopened:
        cooldown.remove(f)

    if active is None:
        pool = [c for c in cands if c["eligible"] and c["family"] not in cooldown]
        if not pool:                      # everything cooled: the cycle is complete,
            cooldown = []                 # so start a new one rather than stalling
            pool = [c for c in cands if c["eligible"]]
        if pool:
            active = pool[0]["family"]
            reason = ((reason + "; ") if reason else "") + \
                     f"selected '{active}' (score {pool[0]['score']}, "  \
                     f"{pool[0]['usable_claims']} usable claims, gap {pool[0]['gap']})"
            st.setdefault("history", []).append(
                {"family": active, "at": time.time(), "reason": reason})
            rotated = True
        else:
            reason = ((reason + "; ") if reason else "") + \
                     "NO ELIGIBLE DIRECTION: every family lacks usable claims. " \
                     "Build the corpus first (lit.py screen, lit.py fetch, claims.py add)."

    st.update({"active": active, "cooldown": cooldown, "reason": reason})
    if rotated:
        st["since"] = time.time()
    # commit=False lets the gate REPORT the decision without performing a rotation.
    # tools/gate.py runs every 20 minutes from the tick loop; if it committed, a
    # direction could rotate underneath a council round that was mid-flight.
    if commit:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(st, indent=1))
    return {"active": active, "reason": reason, "candidates": cands,
            "rotated": rotated, "cooldown": cooldown, "reopened": reopened}


def main() -> int:
    force = None
    if "--force" in sys.argv:
        i = sys.argv.index("--force")
        if i + 1 >= len(sys.argv):
            print("--force needs a family name")
            return 1
        force = sys.argv[i + 1]
        reason = None
        if "--reason" in sys.argv:
            j = sys.argv.index("--reason")
            if j + 1 >= len(sys.argv):
                print("--reason needs text")
                return 2
            reason = sys.argv[j + 1]
        if force not in lit.ALL_FAMILIES:
            print(f"unknown family '{force}'; known: {', '.join(lit.ALL_FAMILIES)}")
            return 1
    d = decide(force, reason=locals().get("reason"))
    if "--json" in sys.argv:
        print(json.dumps(d, indent=1))
        return 0

    print(f"ACTIVE DIRECTION: {d['active'] or 'NONE'}")
    print(f"  {d['reason']}")
    if d.get("reopened"):
        print(f"  reopened by new literature: {', '.join(d['reopened'])}")
    if d["cooldown"]:
        print(f"  cooldown (exploited out, awaiting new evidence or a full cycle): "
              f"{', '.join(d['cooldown'])}")
    print(f"\n{'family':20s} {'score':>6s} {'runs':>5s} {'stale':>6s} {'gap':>5s} "
          f"{'claims':>7s} {'unread':>7s}  status")
    for c in d["candidates"]:
        status = "ELIGIBLE" if c["eligible"] else c["blocked"]
        mark = " <-- ACTIVE" if c["family"] == d["active"] else ""
        print(f"{c['family']:20s} {c['score']:6.3f} {c['runs']:5d} "
              f"{c['since_improve']:6d} {c['gap']:5.2f} {c['usable_claims']:7d} "
              f"{c['unread']:7d}  {status[:74]}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
