#!/usr/bin/env python3
"""THE GATE: what OPHIS is allowed to launch, and when.

The v3 gate had the right idea and one fatal design error: a stale artifact BLOCKED ALL
LAUNCHES. Waiting for prose idled GPUs for 16 minutes once and 48 minutes another time in
a single morning, on a shared box where a free GPU is taken by someone else within
seconds. The response was to demote the check to a warning, which removed the only
enforcement that existed.

Neither is necessary. A protocol obligation is a checkpoint on making NEW RESEARCH
DECISIONS, not on burning compute. So this gate publishes two things:

    gate_open        -- hard health: is it safe to run anything at all?
    decision_cutoff  -- a timestamp. Queue entries created at or before it may launch;
                        entries created after it may not, until the overdue council
                        artifact lands.

Work already decided keeps running while the critic catches up. New ideas wait for the
critic. GPUs never idle for prose, and the checkpoint still binds.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import agenda    # noqa: E402
import claims    # noqa: E402
import coe       # noqa: E402
import council   # noqa: E402

SWEEP = REPO / "runs" / "sweep"
HEALTH_MAX_S = 30 * 60      # tools/health.py runs every 20 min; +10 min slack


def checks() -> list[dict]:
    out = []

    def add(name, ok, hard, detail="", freezes=False):
        # `freezes` marks a check whose staleness moves the decision cutoff. Only
        # artifact-backed obligations qualify: a cutoff is derived from an artifact's
        # mtime, so a check with no artifact behind it has no timestamp to freeze from
        # and would collapse the cutoff to 0 -- a total stop wearing a checkpoint's
        # clothing.
        out.append({"check": name, "ok": bool(ok), "hard": hard, "detail": detail,
                    "freezes": freezes})

    hold = SWEEP / "HOLD"
    add("no operator HOLD", not hold.exists(), True,
        hold.read_text().strip()[:200] if hold.exists() else "")

    # Health is a HARD gate: if we cannot see the host, we cannot verify we are not
    # co-tenanting someone else's GPU, and every result would be uninterpretable.
    hj = REPO / "runs" / "HEALTH.json"
    if hj.exists():
        age = time.time() - hj.stat().st_mtime
        try:
            h = json.loads(hj.read_text())
        except ValueError:
            h = {}
        add("health check fresh", age < HEALTH_MAX_S, True,
            f"{age/60:.0f} min old (max {HEALTH_MAX_S//60})")
        add("host reachable", (h.get("remote") or {}).get("reachable"), True,
            (h.get("remote") or {}).get("error", ""))
    else:
        add("health check fresh", False, True, "runs/HEALTH.json missing; run tools/health.py")
        add("host reachable", False, True, "unknown")

    # An active direction is required before NEW research work is staged, but never
    # blocks compute: controls and already-queued work run regardless. Without read
    # literature no direction can open, which is what forces the corpus to exist.
    try:
        d = agenda.decide(commit=False)
        add("active research direction chosen", bool(d["active"]), False,
            (f"{d['active']}" + (f" (cooldown: {', '.join(d['cooldown'])})"
                                 if d["cooldown"] else ""))
            if d["active"] else d["reason"][:200])
    except Exception as exc:                      # noqa: BLE001
        add("active research direction chosen", False, False, f"selector failed: {exc}")

    # Failures must become lessons or the campaign pays for them twice. Soft, and it
    # does not freeze decisions: the remedy is to write the lesson, not to stop working.
    try:
        res = [json.loads(f.read_text())
               for f in (SWEEP / "results").glob("*.json")]
        un = claims.unlearned_failures(res)
        add("every failure has a lesson", not un, False,
            "" if not un else f"{len(un)} unlearned: "
                              + ", ".join(r["name"] for r in un[:5]))
    except Exception as exc:                          # noqa: BLE001
        add("every failure has a lesson", False, False, f"check failed: {exc}")

    # Chain of evidence. A break means a claim somewhere does not trace to a source, so
    # it blocks new research decisions -- but never running compute, because unwinding a
    # broken citation has nothing to do with whether a GPU should be busy.
    try:
        a = coe.audit()
        add("chain of evidence intact", a["ok"], False,
            "" if a["ok"] else "; ".join(
                f"{k}: {len(v)}" for k, v in a["checks"].items() if v)[:200],
            freezes=True)
    except Exception as exc:                          # noqa: BLE001
        add("chain of evidence intact", False, False, f"audit failed: {exc}")

    # Council artifacts are SOFT: they move the decision cutoff, they never stop compute.
    # The kind list is read from council.SPECS rather than hardcoded, so adding a cadence
    # there makes it enforced here automatically. It was hardcoded to two kinds, which is
    # how a new requirement would have been registered and then silently never checked.
    for kind in council.SPECS:
        s = council.status(kind)
        # The PROJECT critique audits the code and pipeline rather than the science. It
        # does not freeze research decisions -- a stale code review is no reason to refuse
        # an experiment whose evidence is sound -- but it is reported, so it cannot rot
        # unnoticed the way the two-hour cadence would if it lived only in prose.
        add(f"{kind} council current", s["ok"], False,
            "; ".join(s["problems"])[:300] or f"{s.get('age_min')} min old",
            freezes=(kind != "project"))
    return out


def decision_cutoff(chk: list[dict]) -> tuple[float, str]:
    """Newest moment at which a research decision was still authorized."""
    overdue = [c for c in chk if not c["hard"] and not c["ok"] and c.get("freezes")]
    if not overdue:
        return time.time(), ""
    ages = []
    for kind in ("round", "critique"):
        s = council.status(kind)
        if not s["ok"]:
            f = council.latest(kind)
            # No artifact at all: the campaign has not started its councils yet, so
            # freeze from "the beginning of time" is correct -- but only controls are
            # exempt, and only councils can put us here.
            ages.append(f.stat().st_mtime if f else 0.0)
    cutoff = min(ages) if ages else time.time()
    why = "; ".join(f"{c['check']}: {c['detail']}" for c in overdue)
    return cutoff, why


def evaluate() -> dict:
    chk = checks()
    hard_fail = [c["check"] + (f" ({c['detail']})" if c["detail"] else "")
                 for c in chk if c["hard"] and not c["ok"]]
    cutoff, why = decision_cutoff(chk)
    return {"ts": time.time(),
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gate_open": not hard_fail,
            "blocking_failures": hard_fail,
            "decision_cutoff": cutoff,
            "decision_cutoff_reason": why,
            "checks": chk}


def main():
    g = evaluate()
    SWEEP.mkdir(parents=True, exist_ok=True)
    (SWEEP / "GATE_STATUS.json").write_text(json.dumps(g, indent=1))
    if "--json" in sys.argv:
        print(json.dumps(g, indent=1))
        return 0
    print(f"gate_open={g['gate_open']}  ({g['generated']})")
    for c in g["checks"]:
        print(f"  [{'ok ' if c['ok'] else 'FAIL'}] {'hard' if c['hard'] else 'soft'} "
              f" {c['check']}  {c['detail']}")
    if g["decision_cutoff_reason"]:
        when = ("since the campaign began" if g["decision_cutoff"] < 1e6 else
                f"as of {(time.time() - g['decision_cutoff'])/60:.0f} min ago")
        print(f"  NEW DECISIONS FROZEN {when}: {g['decision_cutoff_reason']}")
        print("  (controls are exempt and keep running -- they are the instrument)")
        print("  (already-queued work keeps launching; GPUs do not idle for prose)")
    return 0 if g["gate_open"] else 1


if __name__ == "__main__":
    sys.exit(main())
