#!/usr/bin/env python3
"""Atomic claims and causal mechanisms, extracted from full text and kept append-only.

Deliberately a few hundred lines and one JSONL file, not the large record store an
earlier version shipped. That store held no hypotheses, no papers and no literature
searches while the whole experiment budget ran straight past it. A registry nobody writes
to is worse than no registry, because the gate believed it.

Rules that survive from the ScientistOne / AutoResearchClaw reading, minus the ceremony:

  * A claim resolves to a PRESERVED FULL TEXT on disk plus a locator. Abstract-only
    screening is triage and can never back a claim.
  * INTERNAL VALIDITY and TRANSFER are scored separately and never averaged. A rigorous
    7B/100B-token study can be internally strong and nearly irrelevant to 124M at 600
    steps. Conflating them is how a campaign talks itself into a bad experiment.
  * Opposing evidence is registered with stance="opposes" and stays visible. It is never
    rewritten into support.
  * A mechanism is a mediator plus the claims that support it, plus a falsifier. Naming a
    mechanism does not create one.

    claims.py add   claim.json          # or a JSON list; validated then appended
    claims.py mech  mechanism.json
    claims.py status                    # per-direction-family evidence
    claims.py show  capacity            # everything backing one direction
    claims.py template                  # print a blank claim to fill in
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import lit  # noqa: E402

LIT = REPO / "lit"
CLAIMS = LIT / "claims.jsonl"
MECHS = LIT / "mechanisms.jsonl"
HYPS = LIT / "hypotheses.jsonl"
LESSONS = LIT / "lessons.jsonl"

CLAIM_FIELDS = ("belief_key", "statement", "source_id", "locator", "stance",
                "families", "internal_validity", "transfer", "effect", "comparator")
MECH_FIELDS = ("name", "mediator", "families", "claim_keys", "falsifier",
               "activation_observable", "competing_explanation")
LESSON_FIELDS = ("id", "type", "observation", "diagnosis", "action",
                 "applies_when", "severity", "families", "evidence", "mitigation")
LESSON_TYPES = ("runtime", "integrity", "non_activation", "valid_negative",
                "degenerate", "resource", "scope", "overclaim")
LESSON_ACTIONS = ("proceed", "refine", "pivot", "block")
# A lesson is prose plus, optionally, a MACHINE-CHECKABLE hook. Without the hook a
# `block` lesson is a note someone has to remember, which is how the last system's
# lessons stopped working. `blocks_keys` lets the queue refuse the thing automatically.
LESSON_OPTIONAL = ("blocks_keys", "superseded_by")
HYP_FIELDS = ("id", "statement", "families", "mechanism_names", "claim_keys",
              "intervention", "prediction", "activation", "falsifiers",
              "competing_explanation", "control_design")
STANCES = ("supports", "opposes", "boundary")

CLAIM_TEMPLATE = {
    "belief_key": "short_stable_key_for_the_assertion",
    "statement": "one attributable assertion, no stronger than the source says",
    "source_id": "2505.12345",
    "locator": "section 4.2, table 3",
    "stance": "supports | opposes | boundary",
    "families": ["schedule"],
    "internal_validity": 0,
    "transfer": 0,
    "effect": "e.g. -0.9% val loss vs tuned AdamW at 500 steps",
    "comparator": "what it was measured against, and at what compute",
    "notes": "fatal blockers, seeds, uncertainty, why transfer was scored this way",
}
HYP_TEMPLATE = {
    "id": "hyp_short_stable_id",
    "statement": "One fixed-scope, falsifiable prediction about val_bpb at 300s.",
    "families": ["schedule"],
    "mechanism_names": ["short_mechanism_name"],
    "claim_keys": ["belief_key_1"],
    "intervention": {
        "summary": "one minimal train.py change",
        "cfg": {"note": "the exact config tools/make_variant.py will build"},
    },
    "prediction": {"metric": "val_bpb", "direction": "decrease",
                   "minimum_effect": 0.001,
                   "reasoning": "why this size, derived from the cited claims"},
    # ScientistOne's decisive contribution: ACTIVATION IS SEPARATE FROM OUTCOME.
    # The run must prove the intervention engaged before its result can bear on the
    # mechanism at all. `diagnostic` must be a metric the run actually prints.
    "activation": {
        "predicate": "what must measurably change for the intervention to have engaged",
        "diagnostic": "emitted_snake_case_metric_name",
        "rule": {"op": "gt", "value": 0.0},
        "failure_status": "inconclusive",
    },
    "falsifiers": ["the observation that would kill this hypothesis"],
    "competing_explanation": "the cheaper explanation that must be ruled out",
    "control_design": "yoked_pair",
}
MECH_TEMPLATE = {
    "name": "short_mechanism_name",
    "mediator": "the quantity the intervention actually changes",
    "families": ["schedule"],
    "claim_keys": ["belief_key_1"],
    "falsifier": "the observation that would kill this mechanism",
    "activation_observable": "numeric value the run must emit to prove it engaged",
    "competing_explanation": "the cheaper explanation that must be ruled out",
}


def _read(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def claims() -> list[dict]:
    return _read(CLAIMS)


def mechanisms() -> list[dict]:
    return _read(MECHS)


def hypotheses() -> list[dict]:
    return _read(HYPS)


def lessons() -> list[dict]:
    return _read(LESSONS)


def validate_lesson(l: dict) -> list[str]:
    """A typed failure record. Types matter because they have different consequences:
    a non-activation says nothing about the science, while a valid negative does."""
    bad = [f"missing '{f}'" for f in LESSON_FIELDS if l.get(f) in (None, "", [], {})]
    if l.get("type") not in LESSON_TYPES:
        bad.append(f"type must be one of {LESSON_TYPES}")
    if l.get("action") not in LESSON_ACTIONS:
        bad.append(f"action must be one of {LESSON_ACTIONS}")
    sev = l.get("severity")
    if not isinstance(sev, (int, float)) or not 0 < sev <= 1:
        bad.append("severity must be a number in (0, 1]")
    for fam in l.get("families") or []:
        if fam not in lit.ALL_FAMILIES:
            bad.append(f"unknown family '{fam}'")
    if l.get("action") == "block" and not (l.get("blocks_keys") or l.get("blocks_values")
                                           or l.get("applies_when")):
        bad.append("a 'block' lesson needs blocks_keys (config keys it forbids), "
                   "blocks_values (a per-key rule such as {'ns': {'op':'ge','value':5}}), "
                   "or at minimum an applies_when condition, or nothing can ever enforce it")
    for key, rule in (l.get("blocks_values") or {}).items():
        if not isinstance(rule, dict) or rule.get("op") not in _OPS or "value" not in rule:
            bad.append(f"blocks_values['{key}'] must be {{'op': one of {sorted(_OPS)}, "
                       f"'value': ...}}; got {rule!r}")
    if l.get("type") == "non_activation" and l.get("action") == "block":
        bad.append("a non-activation may not 'block' a direction: the intervention never "
                   "engaged, so the run is inconclusive about the mechanism, not against it")
    return bad


def validate_claim(c: dict) -> list[str]:
    bad = [f"missing '{f}'" for f in CLAIM_FIELDS if c.get(f) in (None, "", [])]
    if c.get("stance") not in STANCES:
        bad.append(f"stance must be one of {STANCES}")
    for f in ("internal_validity", "transfer"):
        v = c.get(f)
        if not isinstance(v, (int, float)) or not 0 <= v <= 4:
            bad.append(f"'{f}' must be a number 0-4 (they are scored separately, "
                       f"never averaged)")
    for fam in c.get("families") or []:
        if fam not in lit.ALL_FAMILIES:
            bad.append(f"unknown family '{fam}'; known: {', '.join(lit.ALL_FAMILIES)}")
    # THE rule that makes this a chain of evidence rather than a memory: the snapshot
    # must exist on disk. Screening metadata is not a source.
    sid = c.get("source_id")
    if sid:
        snap = lit.SOURCES / f"arxiv_{sid}_fulltext.txt"
        if not snap.exists():
            bad.append(f"no full-text snapshot at {snap.relative_to(REPO)}; "
                       f"run: lit.py fetch  (an abstract may not back a claim)")
    return bad


def validate_mech(m: dict) -> list[str]:
    bad = [f"missing '{f}'" for f in MECH_FIELDS if m.get(f) in (None, "", [])]
    known = {c["belief_key"] for c in claims()}
    for k in m.get("claim_keys") or []:
        if k not in known:
            bad.append(f"claim_keys references unregistered claim '{k}'")
    for fam in m.get("families") or []:
        if fam not in lit.ALL_FAMILIES:
            bad.append(f"unknown family '{fam}'")
    return bad


def active_lessons(families=None) -> list[dict]:
    """Lessons still in force, most severe first.

    AutoResearchClaw 3.6 retrieves lessons by category and injects them into prompts so
    a new run cannot repeat an old failure. We keep the retrieval and drop their
    time-decay: a scientific negative stays true until contradicted, and an operational
    lesson stays in force while its `applies_when` condition holds. A lesson retires
    when a newer one explicitly supersedes it -- by evidence, never by age.
    """
    all_l = lessons()
    live = {l.get("id") for l in all_l}
    # `superseded_by` on lesson X names the lesson that REPLACES X, so X retires -- not
    # the replacement. Getting this backwards silently retired the newest lesson and kept
    # the stale one, which is worse than having no supersession at all.
    out = [l for l in all_l
           if not (l.get("superseded_by") and l["superseded_by"] in live)]
    if families:
        fams = set(families)
        out = [l for l in out if fams & set(l.get("families") or [])]
    return sorted(out, key=lambda l: -float(l.get("severity") or 0))


def blocking_keys() -> dict:
    """cfg key -> the active `block` lesson forbidding it."""
    out = {}
    for l in active_lessons():
        if l.get("action") != "block":
            continue
        for k in l.get("blocks_keys") or []:
            out.setdefault(k, l)
    return out


_OPS = {"ge": lambda a, b: a >= b, "gt": lambda a, b: a > b,
        "le": lambda a, b: a <= b, "lt": lambda a, b: a < b,
        "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}


def diagnostic_would_discriminate(diag: str, rule: dict) -> tuple[bool, str]:
    """Would this activation diagnostic actually separate treatment from control?

    Checked BEFORE a hypothesis is registered, not after its runs come back. Three
    hypotheses in this campaign shipped with a diagnostic the control also satisfies --
    secmom_clamp_frac (0.0 in both arms), num_steps with rule >0 (every run passes), and
    qk_q_rms_final (pinned to exactly 1.0 by QK-norm, in all 4 treatments and all 25
    controls). Each was registered AFTER L030 recorded the failure mode, by the same
    author. The lesson did not prevent recurrence because nothing mechanical enforced it;
    a rule that lives only in prose is a rule that will be forgotten.

    Returns (ok, message). Judged against the CONTROL corpus already on disk: if every
    control reading satisfies the rule, the diagnostic cannot demonstrate engagement.
    Unknown fields pass -- a diagnostic no control has ever emitted may be perfectly good,
    and refusing it would block every genuinely new observable.
    """
    import direction
    from analyze import load as _load
    vals = [r["metrics"][diag] for r in _load()
            if direction.is_platform(r.get("cfg") or {}) and diag in (r.get("metrics") or {})]
    if not vals:
        return True, f"no control has emitted '{diag}' yet; cannot pre-check"
    op, want = rule.get("op"), rule.get("value")
    ok_f = {"gt": lambda v: v > want, "lt": lambda v: v < want,
            "ge": lambda v: v >= want, "le": lambda v: v <= want,
            "eq": lambda v: v == want}.get(op)
    if ok_f is None:
        return True, f"unrecognised op {op!r}; not pre-checked"
    passing = [v for v in vals if ok_f(v)]
    if len(passing) == len(vals):
        return False, (f"REFUSED: every one of {len(vals)} control run(s) also satisfies "
                       f"{rule} on '{diag}' (e.g. {passing[0]}). A diagnostic the control "
                       f"passes cannot show the mechanism engaged -- the run would be "
                       f"INCONCLUSIVE by construction (L030). Pick a quantity whose "
                       f"control value is known and DIFFERENT.")
    # NO constancy clause here, deliberately. A control value that is bit-identical but
    # FAILS the rule is the ideal diagnostic, not a broken one: n_ve_layers reads exactly
    # 4.0 in every control against a rule of >4, and flops_per_token_M exactly 239.078
    # against a rule of <230. Both separate the arms perfectly BECAUSE they are
    # deterministic. A first version of this check refused both -- the same false positive
    # already made once in coe.py's E3 and fixed there. Treatment-side constancy (the
    # secmom_ortho_ratio failure) cannot be seen before the runs exist and stays E3's job.
    return True, (f"ok: {len(vals) - len(passing)} of {len(vals)} controls FAIL the rule, "
                  f"so the diagnostic separates the arms")


def blocked_values(cfg: dict) -> list[tuple]:
    """[(key, value, lesson, rule)] for cfg entries an active lesson forbids by VALUE.

    `blocks_keys` is a whole-key hammer and several real lessons do not want one. L007
    forbids ns at or above the coefficient-table length, not the ns axis -- blocking the
    key would also forbid ns 1..4, which are genuine experiments, and one of them is the
    positive control this round depends on. Without a value-level rule the enforcement
    hook stays empty, which is how the campaign ended up with a `block` lesson that
    blocked nothing while the validator accepted it.
    """
    out = []
    for l in active_lessons():
        if l.get("action") != "block":
            continue
        for key, rule in (l.get("blocks_values") or {}).items():
            if key not in (cfg or {}):
                continue
            fn = _OPS.get(rule.get("op"))
            try:
                if fn and fn(cfg[key], rule["value"]):
                    out.append((key, cfg[key], l, rule))
            except TypeError:
                continue
    return out


def unlearned_failures(results: list[dict]) -> list[dict]:
    """Failed or invalid runs that no lesson references.

    This is the number that decides whether the lesson store is alive. A campaign that
    fails and records nothing has not learned; it has only lost time.
    """
    cited = {e for l in lessons() for e in (l.get("evidence") or [])}
    return [r for r in results
            if (not r.get("ok") or r.get("invalid_reason")) and r.get("name") not in cited]


def validate_hyp(h: dict) -> list[str]:
    bad = [f"missing '{f}'" for f in HYP_FIELDS if h.get(f) in (None, "", [], {})]
    known_c = {c["belief_key"] for c in claims()}
    known_m = {m["name"] for m in mechanisms()}
    for k in h.get("claim_keys") or []:
        if k not in known_c:
            bad.append(f"claim_keys references unregistered claim '{k}'")
    for m in h.get("mechanism_names") or []:
        if m not in known_m:
            bad.append(f"mechanism_names references unregistered mechanism '{m}'")
    for fam in h.get("families") or []:
        if fam not in lit.ALL_FAMILIES:
            bad.append(f"unknown family '{fam}'")
    act = h.get("activation") or {}
    for f in ("predicate", "diagnostic", "rule", "failure_status"):
        if not act.get(f):
            bad.append(f"activation.{f} is required -- a hypothesis with no activation "
                       f"predicate cannot distinguish 'did not engage' from 'did not work'")
    if act.get("failure_status") not in (None, "inconclusive"):
        bad.append("activation.failure_status must be 'inconclusive': failure to activate "
                   "is never evidence against the mechanism")
    rule = act.get("rule") or {}
    if rule and rule.get("op") not in ("gt", "lt", "ge", "le", "ne", "abs_gt"):
        bad.append(f"activation.rule.op {rule.get('op')!r} unsupported")
    if (h.get("prediction") or {}).get("metric") != "val_bpb":
        bad.append("prediction.metric must be 'val_bpb' -- it is the only objective")
    if h.get("control_design") not in ("yoked_pair", "unpaired"):
        bad.append("control_design must be 'yoked_pair' or 'unpaired'")
    return bad


def _add(path: pathlib.Path, items: list[dict], validator, kind: str) -> int:
    problems = {}
    for i, it in enumerate(items):
        p = validator(it)
        if p:
            problems[it.get("belief_key") or it.get("name") or it.get("id") or f"#{i}"] = p
    if problems:
        print(f"rejected; nothing written. {len(problems)} invalid {kind}(s):")
        for k, ps in problems.items():
            print(f"  {k}:")
            for p in ps:
                print(f"    - {p}")
        return 1
    LIT.mkdir(parents=True, exist_ok=True)
    existing = {c.get("belief_key") or c.get("name") or c.get("id") for c in _read(path)}
    written = 0
    with open(path, "a") as f:
        for it in items:
            key = it.get("belief_key") or it.get("name") or it.get("id")
            if key in existing:
                print(f"  skip duplicate {kind} '{key}'")
                continue
            it["registered_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            f.write(json.dumps(it, sort_keys=True) + "\n")
            written += 1
    print(f"registered {written} {kind}(s); total {len(_read(path))}")
    return 0


def literature_state() -> dict:
    """Per-family evidence, for direction selection.

    `support` counts only supporting claims; `opposes` is reported separately rather than
    netted off, because a family with 6 supporting and 5 opposing claims is contested, not
    weakly supported, and those call for different next actions.
    """
    cs, ms = claims(), mechanisms()
    corpus = lit.corpus_state()
    st = {}
    for fam in lit.ALL_FAMILIES:
        fc = [c for c in cs if fam in (c.get("families") or [])]
        sup = [c for c in fc if c["stance"] == "supports"]
        opp = [c for c in fc if c["stance"] == "opposes"]
        # A claim is only usable here if it transfers to this operating point at all.
        usable = [c for c in sup if c.get("transfer", 0) >= 2]
        cov = corpus.get(fam, {"screened": 0, "fetched": 0})
        st[fam] = {
            "claims": len(fc), "supports": len(sup), "opposes": len(opp),
            "usable": len(usable),
            "mean_transfer": (sum(c.get("transfer", 0) for c in sup) / len(sup)) if sup else 0.0,
            "mean_validity": (sum(c.get("internal_validity", 0) for c in sup) / len(sup)) if sup else 0.0,
            "mechanisms": len([m for m in ms if fam in (m.get("families") or [])]),
        "hypotheses": len([h for h in hypotheses() if fam in (h.get("families") or [])]),
        "lessons": len(active_lessons([fam])),
            "fetched": cov["fetched"], "screened": cov["screened"],
            # unread = full texts on disk that have produced no claim yet. While this is
            # positive the literature on this direction is NOT exhausted.
            "unread": max(0, cov["fetched"] - len({c["source_id"] for c in fc})),
        }
    return st


def cmd_add(args):
    data = json.loads(pathlib.Path(args.path).read_text())
    return _add(CLAIMS, data if isinstance(data, list) else [data], validate_claim, "claim")


def cmd_mech(args):
    data = json.loads(pathlib.Path(args.path).read_text())
    return _add(MECHS, data if isinstance(data, list) else [data], validate_mech, "mechanism")


def cmd_hyp(args):
    data = json.loads(pathlib.Path(args.path).read_text())
    return _add(HYPS, data if isinstance(data, list) else [data], validate_hyp, "hypothesis")


def cmd_lesson(args):
    data = json.loads(pathlib.Path(args.path).read_text())
    return _add(LESSONS, data if isinstance(data, list) else [data],
                validate_lesson, "lesson")


def cmd_lessons(args):
    ls = active_lessons([args.family] if args.family else None)
    if not ls:
        scope = f" for '{args.family}'" if args.family else ""
        print(f"no active lessons{scope}.")
        return 0
    print(f"{len(ls)} active lesson(s), most severe first"
          + (f" for '{args.family}'" if args.family else "") + ":\n")
    for l in ls:
        blocks = f"  BLOCKS cfg keys {l['blocks_keys']}" if l.get("blocks_keys") else ""
        print(f"[{l['severity']:.2f}] {l['id']}  ({l['type']} -> {l['action']}){blocks}")
        print(f"    observed : {l['observation']}")
        print(f"    diagnosis: {l['diagnosis']}")
        print(f"    applies  : {l['applies_when']}")
        print(f"    do       : {l['mitigation']}")
        print(f"    evidence : {', '.join(l.get('evidence') or []) or '-'}\n")
    return 0


def cmd_status(args):
    st = literature_state()
    print(f"{'family':20s} {'fetched':>7s} {'unread':>6s} {'claims':>6s} {'sup':>4s} "
          f"{'opp':>4s} {'usable':>6s} {'transfer':>8s} {'mech':>5s} {'hyp':>4s} {'lesn':>5s}")
    for fam, s in sorted(st.items(), key=lambda kv: -kv[1]["usable"]):
        print(f"{fam:20s} {s['fetched']:7d} {s['unread']:6d} {s['claims']:6d} "
              f"{s['supports']:4d} {s['opposes']:4d} {s['usable']:6d} "
              f"{s['mean_transfer']:8.2f} {s['mechanisms']:5d} {s['hypotheses']:4d} {s['lessons']:5d}")
    print("\nusable = supporting claims with transfer>=2 to THIS operating point.")
    print("unread = fetched full texts that have produced no claim yet; while >0 the")
    print("         literature on that direction is not exhausted.")
    return 0


def cmd_show(args):
    fam = args.family
    cs = [c for c in claims() if fam in (c.get("families") or [])]
    ms = [m for m in mechanisms() if fam in (m.get("families") or [])]
    if not cs and not ms:
        print(f"no registered evidence for '{fam}'.")
        return 1
    print(f"=== {len(cs)} claims for '{fam}' ===")
    for c in sorted(cs, key=lambda c: (-c.get("transfer", 0), c["belief_key"])):
        print(f"\n[{c['stance']:9s}] {c['belief_key']}  "
              f"(validity {c.get('internal_validity')}/4, transfer {c.get('transfer')}/4)")
        print(f"  {c['statement']}")
        print(f"  source arXiv:{c['source_id']} @ {c['locator']}")
        print(f"  effect: {c.get('effect')}  vs  {c.get('comparator')}")
    print(f"\n=== {len(ms)} mechanisms ===")
    for m in ms:
        print(f"\n{m['name']}: mediator = {m['mediator']}")
        print(f"  backed by {m.get('claim_keys')}")
        print(f"  activation: {m.get('activation_observable')}")
        print(f"  falsifier : {m.get('falsifier')}")
        print(f"  competing : {m.get('competing_explanation')}")
    return 0


def cmd_template(args):
    print("// claim -- save as JSON (a list is fine) and run: claims.py add file.json")
    print(json.dumps(CLAIM_TEMPLATE, indent=1))
    print("\n// mechanism -- claims.py mech file.json")
    print(json.dumps(MECH_TEMPLATE, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("path"); a.set_defaults(fn=cmd_add)
    m = sub.add_parser("mech"); m.add_argument("path"); m.set_defaults(fn=cmd_mech)
    y = sub.add_parser("hyp"); y.add_argument("path"); y.set_defaults(fn=cmd_hyp)
    ls = sub.add_parser("lesson"); ls.add_argument("path"); ls.set_defaults(fn=cmd_lesson)
    lz = sub.add_parser("lessons"); lz.add_argument("family", nargs="?")
    lz.set_defaults(fn=cmd_lessons)
    s = sub.add_parser("status"); s.set_defaults(fn=cmd_status)
    w = sub.add_parser("show"); w.add_argument("family"); w.set_defaults(fn=cmd_show)
    t = sub.add_parser("template"); t.set_defaults(fn=cmd_template)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
