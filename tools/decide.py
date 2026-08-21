#!/usr/bin/env python3
"""One reproducible answer to: given this state, which experiment runs next?

Two external reviews reached the same conclusion independently -- that this system judges
results far better than it chooses experiments -- and both named the same reason. The
choice was made by a committee whose members do not compose: a council proposes, agenda.py
picks a family, direction.py opens and closes axes, balance.py watches the explore/exploit
split, lessons block configs, selector.py ranks but is ADVISORY, and whatever remains is
settled by the operator. Nothing wrote down the answer, so the same state could produce
different choices on different days and nobody could tell.

The operator reordered the live queue by hand twice in one session -- once to put a
deconfounding arm ahead of two sweep rungs, once because a newly queued arm landed at the
back. Both were, in hindsight, the right calls. Neither was recorded, reproducible, or
available to anyone reading the repository afterwards.

This makes the ordering a function of state and writes down the result:

    python3 tools/decide.py            # show the decision, change nothing
    python3 tools/decide.py --apply    # reorder the queue and record the decision

A decision record is immutable and holds the state it was made from, every candidate that
was considered, every score, what was selected, and any override. It is written under
runs/sweep/decisions/ and is evidence, not state: it is what lets a reader ask whether the
campaign would make the same choice again.

WHAT THIS IS NOT. The scores come from tools/selector.py, whose weights are stated rather
than fitted. So this is a REPRODUCIBLE ordering, not an optimal one -- it removes the
unrecorded human tie-break, which is the actual complaint, and does not pretend to a
calibrated policy. An operator may still override; the override is recorded as such and
shows up in the record beside the ranking it displaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import analyze      # noqa: E402
import direction    # noqa: E402
import selector     # noqa: E402

SWEEP = REPO / "runs" / "sweep"
DECISIONS = SWEEP / "decisions"


def state_hash(rows, queue) -> str:
    """A digest of everything the decision depends on, so a later reader can tell whether
    the state changed or the policy did. Both matter and they are easy to confuse."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x.get("name", "")):
        h.update(f"{r.get('name')}:{(r.get('metrics') or {}).get('val_bpb')}".encode())
    for e in sorted(queue, key=lambda x: x.get("name", "")):
        h.update(f"{e.get('name')}:{json.dumps(e.get('cfg'), sort_keys=True)}".encode())
    return h.hexdigest()[:16]


def pending(queue):
    """Queued waves with no result yet, grouped so a wave is ordered as a unit."""
    waves = {}
    for e in queue:
        g = e.get("wave_group")
        if not g or (SWEEP / "results" / f"{e['name']}.json").exists():
            continue
        waves.setdefault(g, []).append(e)
    return waves


def decide(rows, queue):
    """Rank pending WAVES by the best score among their treatment arms.

    A wave is the unit the dispatcher launches, so ordering individual entries would be
    meaningless -- its members must stay adjacent. Scoring by the best treatment in the
    wave, rather than the mean, avoids penalising a wave for the controls it must carry.
    """
    state = direction.axis_state(rows)
    fx = selector.family_effects(rows)
    out = []
    for g, members in pending(queue).items():
        treats = [m for m in members if not direction.is_platform(m.get("cfg") or {})]
        best, terms, cfg = None, {}, None
        for m in treats:
            s, t = selector.score(m.get("cfg") or {}, rows, state, fx)
            if best is None or s > best:
                best, terms, cfg = s, t, m.get("cfg")
        out.append({
            "wave": g,
            "score": None if best is None or best == float("-inf") else round(best, 3),
            "refused": best == float("-inf") or best is None,
            "label": direction.label(cfg or {}),
            "delta": {k: v for k, v in (cfg or {}).items()
                      if direction.PLATFORM.get(k) != v},
            "terms": {k: str(v) for k, v in terms.items()},
            "members": [m["name"] for m in members],
        })
    # Refused waves sort last, and keep their reason. They are NOT dropped: a wave the
    # policy refuses today may be admissible tomorrow, and silently deleting it would
    # repeat the disappearing-queue-entry failure this campaign already paid for.
    out.sort(key=lambda d: (d["refused"], -(d["score"] if d["score"] is not None else -1e9)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="reorder runs/sweep/queue.json to match, and record the decision")
    ap.add_argument("--override", metavar="WAVE",
                    help="force this wave first; recorded as an override with its reason")
    ap.add_argument("--reason", default="", help="why the override was made")
    a = ap.parse_args()

    rows = analyze.load()
    qf = SWEEP / "queue.json"
    queue = json.loads(qf.read_text())
    ranked = decide(rows, queue)
    sh = state_hash(rows, queue)

    if a.override and not a.reason:
        print("an override must carry --reason; an unexplained override is exactly the "
              "unrecorded tie-break this file exists to remove")
        return 2

    print(f"=== DECISION over {len(ranked)} pending wave(s)   state {sh} ===")
    for d in ranked:
        mark = "REFUSED" if d["refused"] else f"{d['score']:+7.2f}"
        print(f"  {mark}  {d['wave']:12s} {d['label']:22s} {d['delta']}")
        for k, v in d["terms"].items():
            print(f"             {k:10s} {v[:88]}")
    if a.override:
        print(f"\n  OVERRIDE: {a.override} forced first -- {a.reason}")

    if not a.apply:
        print("\n(dry run; pass --apply to reorder the queue and write the record)")
        return 0

    order = [d["wave"] for d in ranked]
    if a.override:
        order = [a.override] + [w for w in order if w != a.override]
    rank = {w: i for i, w in enumerate(order)}
    done = [e for e in queue if e.get("wave_group") not in rank]
    todo = sorted((e for e in queue if e.get("wave_group") in rank),
                  key=lambda e: rank[e["wave_group"]])
    qf.write_text(json.dumps(done + todo, indent=1))

    DECISIONS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    rec = {
        "at": stamp,
        "state_hash": sh,
        "n_results": len(rows),
        "candidate_set": [d["wave"] for d in ranked],
        "scores": {d["wave"]: d["score"] for d in ranked},
        "terms": {d["wave"]: d["terms"] for d in ranked},
        "selected": order[:1],
        "selection_reason": ("highest selector score among pending waves"
                            if not a.override else f"OPERATOR OVERRIDE: {a.reason}"),
        "operator_override": a.override,
        "weights": {"gain": selector.W_GAIN, "info": selector.W_INFO,
                    "novelty": selector.W_NOVEL, "resolve": selector.W_RESOLVE,
                    "activation_penalty": selector.W_ACTPEN},
    }
    (DECISIONS / f"{stamp}_decision.json").write_text(json.dumps(rec, indent=1))
    print(f"\napplied; queue reordered and decision recorded at "
          f"runs/sweep/decisions/{stamp}_decision.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
