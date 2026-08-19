#!/usr/bin/env python3
"""Multi-agent councils, and the validator that makes them real.

Two councils exist:

  ROUND    (every research block) -- generates ideas, hypotheses and mechanisms.
           Roles: explorer, pragmatist, critic, synthesizer.
  CRITIQUE (hourly) -- Fable audits the campaign as a whole.
           Roles: fable_evidence, fable_method, fable_process, fable_synthesis.

Why this file exists rather than more prose in a protocol document:

  * v3's gate checked only the FILE MTIME of these artifacts. `touch` satisfied it. The
    hourly critic itself flagged this three times and it was never fixed. Validation here
    is on CONTENT: every role section must exist, carry a distinct agent id, and clear a
    word floor. An empty or duplicated section fails.
  * v3's papers proposed experiments (F1..F6) in prose, and a human then hand-wrote queue
    entries. Most proposals never ran. A round here is only VALID if its synthesis emits a
    machine-readable ```queue block, which `tools/queue_from_round.py` feeds straight to
    the dispatcher. Ideas that cannot be executed do not count as ideas.
  * Distinct role labels inside one context are not independence. Each section must name a
    different agent id; the synthesizer may not be one of the reviewers.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
ROUNDS, CRITIQUES = REPO / "rounds", REPO / "critiques"

MIN_WORDS = 120          # per role section
MIN_SYNTH_WORDS = 200

SPECS = {
    "round": {
        "dir": ROUNDS,
        "max_age_s": 90 * 60,
        "roles": ["explorer", "pragmatist", "critic", "synthesis"],
        "needs_queue": True,
    },
    "critique": {
        "dir": CRITIQUES,
        "max_age_s": 90 * 60,
        "roles": ["fable_evidence", "fable_method", "fable_process", "fable_synthesis"],
        "needs_queue": False,
    },
}

SECTION_RE = re.compile(r"^##\s+(?P<role>[a-z_]+)\s*(?:\((?P<agent>[^)]*)\))?\s*$",
                        re.MULTILINE)
QUEUE_RE = re.compile(r"```queue\s*\n(?P<body>.*?)\n```", re.DOTALL)


def parse(text: str) -> dict:
    """Split an artifact into role -> {agent, words, body}."""
    marks = list(SECTION_RE.finditer(text))
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        out[m.group("role")] = {"agent": (m.group("agent") or "").strip(),
                                "words": len(body.split()), "body": body}
    return out


def queue_entries(text: str) -> list[dict]:
    """Configs the synthesis wants run, as JSON inside a ```queue fence."""
    m = QUEUE_RE.search(text)
    if not m:
        return []
    try:
        data = json.loads(m.group("body"))
    except ValueError as exc:
        raise ValueError(f"```queue block is not valid JSON: {exc}") from exc
    return data if isinstance(data, list) else [data]


def validate(path: pathlib.Path, kind: str) -> list[str]:
    """Every reason this artifact does not count. Empty list means it is real."""
    spec = SPECS[kind]
    problems = []
    try:
        text = path.read_text()
    except OSError as exc:
        return [f"unreadable: {exc}"]

    got = parse(text)
    for role in spec["roles"]:
        if role not in got:
            problems.append(f"missing section '## {role}'")
            continue
        floor = MIN_SYNTH_WORDS if role.endswith("synthesis") else MIN_WORDS
        if got[role]["words"] < floor:
            problems.append(f"'{role}' has {got[role]['words']} words, floor is {floor} "
                            f"(a stub is not a review)")
        if not got[role]["agent"]:
            problems.append(f"'{role}' names no agent id: use '## {role} (agent-id)'")

    ids = [v["agent"] for k, v in got.items() if k in spec["roles"] and v["agent"]]
    if len(ids) != len(set(ids)):
        problems.append(f"agent ids are not distinct ({ids}) - role labels in one context "
                        f"are not independence")
    synth = next((r for r in spec["roles"] if r.endswith("synthesis")), None)
    if synth and synth in got:
        reviewers = [v["agent"] for k, v in got.items()
                     if k in spec["roles"] and k != synth and v["agent"]]
        if got[synth]["agent"] in reviewers:
            problems.append("the synthesizer is also a reviewer; it must be independent")

    if spec["needs_queue"]:
        try:
            q = queue_entries(text)
        except ValueError as exc:
            problems.append(str(exc))
            q = []
        if not q:
            problems.append("no ```queue block: a round that proposes nothing runnable is "
                            "not a round (this is why v3's paper proposals never executed)")
        else:
            for e in q:
                for f in ("name", "cfg", "rationale", "falsifier"):
                    if not e.get(f):
                        problems.append(f"queue entry {e.get('name','?')!r} lacks '{f}'")
            problems += _admissible(q)
    return problems


def _admissible(entries: list[dict]) -> list[str]:
    """A round must propose at least one experiment that can ACTUALLY be queued.

    Structural validation checked that a ```queue block existed and that its entries had
    prose fields. It never asked whether any of them would survive the queue door, so a
    round could pass while producing nothing runnable -- which is exactly what happened:
    round 5 validated cleanly and all three of its proposals were rejected by
    queue_from_round for `hypothesis_id: null`. "Valid round" meant structurally verbose,
    not executable, and the failure was silent because the two checks lived in different
    files and neither called the other.

    This runs the same semantic preflight the door runs: recognised config keys, a
    registered hypothesis (or an explicit instrument-probe declaration), no lesson block,
    and a variant that builds and differs from the control. Reporting per-entry reasons
    matters more than the pass/fail: a proposer that never learns WHY its arms bounce will
    keep writing the same unrunnable rounds.
    """
    import claims as _c
    import direction as _d
    import make_variant as _mv
    verdicts, admissible = [], 0
    known = {h["id"] for h in _c.hypotheses()}
    for e in entries:
        name, cfg = e.get("name", "?"), e.get("cfg") or {}
        hid = e.get("hypothesis_id")
        why = None
        if not hid:
            why = ("no hypothesis_id (set it, or 'none' to declare an instrument probe) "
                   "-- without one the activation predicate never runs and a null cannot "
                   "be told from 'never engaged'")
        elif hid != "none" and hid not in known:
            why = f"hypothesis_id {hid!r} is not registered"
        elif _d.unknown_keys(cfg):
            why = f"unrecognised config keys {sorted(_d.unknown_keys(cfg))}"
        elif _c.blocked_values(cfg):
            k, v, les, _r = _c.blocked_values(cfg)[0]
            why = f"{k}={v} is blocked by {les['id']}"
        else:
            try:
                if _mv.build(cfg) == _mv.build(dict(_d.PLATFORM)):
                    why = "generated variant is byte-identical to the control"
            except Exception as exc:                       # noqa: BLE001
                why = f"variant does not build: {str(exc)[:80]}"
        if why:
            verdicts.append(f"  queue entry {name!r} is NOT admissible: {why}")
        else:
            admissible += 1
    if admissible:
        return []
    return ([f"NO ADMISSIBLE EXPERIMENT: all {len(entries)} proposals would be refused at "
             f"the queue door, so this round proposes nothing runnable."] + verdicts)





def latest(kind: str):
    d = SPECS[kind]["dir"]
    files = sorted(d.glob("*.md")) if d.is_dir() else []
    return files[-1] if files else None


def status(kind: str) -> dict:
    spec = SPECS[kind]
    f = latest(kind)
    if f is None:
        return {"kind": kind, "ok": False, "path": None,
                "problems": [f"no {kind} artifact exists in {spec['dir'].name}/"]}
    age = time.time() - f.stat().st_mtime
    problems = validate(f, kind)
    if age > spec["max_age_s"]:
        problems.append(f"stale: {age/60:.0f} min old, cadence is "
                        f"{spec['max_age_s']//60} min")
    return {"kind": kind, "ok": not problems, "path": str(f.relative_to(REPO)),
            "age_min": round(age / 60, 1), "problems": problems}


PROMPTS = {
 "round": """FIRST establish the active direction and read its literature. Run
`python3 tools/agenda.py`; it names the ACTIVE DIRECTION, or refuses and tells you the
corpus is too thin. Then `python3 tools/lit.py read <active>` for that direction's
reading list and `python3 tools/claims.py show <active>` for what is already extracted.
Read the unread full texts in `lit/sources/` -- method, comparator, tables, ablations,
limitations -- and register what you learn with `tools/claims.py add` BEFORE the council
runs. A round whose direction has unread full texts is a round arguing from ignorance.

THEN dispatch these FOUR subagents IN PARALLEL, each in its own context. Give each the
active direction, its registered claims and mechanisms, `python3 tools/claims.py lessons
<active>` (the failures previous runs already paid for), `python3 tools/direction.py`,
the last critique in critiques/, and runs/sweep/results/*.json. Write ONE file
rounds/<UTC ISO>_round.md containing their four sections verbatim.

  ## explorer (<agent-id>)
    Propose 3-5 interventions that could move val_bpb at 300s, concentrated on the
    ACTIVE DIRECTION and grounded in its registered claims -- cite belief_keys. You are
    rewarded for reaching mechanisms and axes nobody has touched, NOT for safe increments. State for
    each: the causal mediator, why it could be large here, and the observable that proves
    it engaged. Run `python3 tools/direction.py` FIRST and work from its two bottom
    tables. Any axis marked UNEXPLORED is your first obligation -- a knob never tried once
    is not evidence about that knob. The DIRECTION SPACE table lists whole intervention
    families sorted least-covered first, with each family's fixed-time COST CLASS on this
    benchmark; a family showing 0 runs, or a family with runs but an UNTOUCHED axis inside
    it, is the strongest signal available. That second case is exactly how the previous
    campaign ran an entire experiment budget while never once moving model width or tokens-per-step.
    Justify every proposal against its cost class: GPU work is near-free here, CPU work
    is nearly fatal, and anything in `data_curriculum` is unreachable behind the frozen
    prepare.py -- do not spend a round rediscovering that.

  ## pragmatist (<agent-id>)
    For each proposal: can it be expressed in train.py alone (prepare.py is FROZEN)? What
    does it cost in throughput? Remember the binding constraint on this benchmark: ~90% of
    the 300s is the single-threaded Python packing loop in the frozen prepare.py, so GPU
    FLOPs are close to free and anything that costs CPU is close to fatal. Rank by
    expected val_bpb gain per unit of implementation risk. Kill what cannot be built.

  ## critic (<agent-id>)
    Try to destroy each surviving proposal BEFORE it runs. Name the confound, the cheaper
    explanation, and the evidence that would already refute it -- including any registered
    claim with stance="opposes" that the explorer skipped past, any claim whose
    `transfer` score is too low to carry the weight being put on it, and any ACTIVE
    LESSON the proposal walks straight back into. Run `python3 tools/claims.py lessons`
    and say, for each proposal, which lessons you checked it against. A campaign that
    re-buys a failure it already paid for has learned nothing. Anything whose predicted
    effect is under the measured noise band is not runnable at n=1 -- say so and demand
    replicates or a bigger lever.

  ## synthesis (<agent-id>, must NOT be any of the three above)
    Integrate. Say explicitly which objection changed which proposal. Preserve dissent
    rather than averaging it. End with a ```queue fenced JSON list of the configs to run:
      [{"name":"...", "cfg":{...}, "wave_group":"w1",
        "rationale":"...", "falsifier":"...", "expected":"..."}]
    Every entry must be executable by tools/make_variant.py.

    USE `wave_group`. Entries sharing one launch CONCURRENTLY on separate GPUs or not at
    all, up to the 4-GPU cap. Pair every treatment with its own control in the same wave:
    runs launched minutes apart on a contended host measure drift rather than effect, and
    an unpaired single run cannot resolve anything smaller than the sequential band that
    tools/direction.py reports. Two entries per wave is the minimum useful design; four
    (two treatments + two controls, or one treatment + three controls) fills the box.""",

 "critique": """Dispatch FOUR independent Fable subagents (model claude-fable-5) IN
PARALLEL. Fable has NO execution authority: it may not stage, kill, or edit train.py.
Each must read PRIMARY state -- runs/sweep/results/*.json, rounds/, tools/*.py, the host
via read-only ssh -- never your narration. Write ONE file
critiques/<UTC ISO>_fable_critique.md with these sections:

  ## fable_evidence (<agent-id>)
    Recompute every headline number yourself from runs/sweep/results/*.json. Which claim is
    strongest, and which is most likely to be withdrawn? What has been rated too highly?
    Is any effect being quoted that is smaller than the measured noise band?

  ## fable_method (<agent-id>)
    Where do spec, code, activation diagnostic and interpretation disagree? Verify that
    each variant actually contains the edit it claims -- read the generated variant source,
    not the config. What assumption is unmeasured, ranked by cheapest decisive check?

  ## fable_process (<agent-id>)
    Run `python3 tools/claims.py lessons` and `python3 tools/analyze.py`. Is every failed
    or invalid run covered by a registered lesson, or are failures being absorbed
    silently? An UNLEARNED FAILURE is time spent for nothing. Is the portfolio
    over-concentrated? Check `python3 tools/direction.py` for axes still
    UNEXPLORED after N runs and say why they were skipped. Were the previous critique's
    recommendations adopted or ignored? Name the ignores explicitly, with their cost.

  ## fable_synthesis (<agent-id>, independent of the three above)
    What should be refined, pivoted, blocked, or tested next -- each with a falsifier and
    an activation diagnostic. What in the research SYSTEM itself must change, and what
    evidence justifies it? Separate fact / inference / disagreement.""",
}


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "status"
    if kind == "status":
        for k in SPECS:
            s = status(k)
            print(f"[{k}] {'OK' if s['ok'] else 'FAIL'}  {s.get('path')}  "
                  f"{s.get('age_min')} min")
            for p in s["problems"]:
                print(f"    - {p}")
        return 0
    if kind in PROMPTS:
        print(PROMPTS[kind])
        return 0
    if kind == "check" and len(sys.argv) > 3:
        for p in validate(pathlib.Path(sys.argv[2]), sys.argv[3]):
            print("FAIL:", p)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
