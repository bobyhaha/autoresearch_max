#!/usr/bin/env python3
"""CHAIN OF EVIDENCE. Can every claim this campaign makes be walked back to a source?

Two papers motivate this file, and neither is copied wholesale.

ScientistOne (arXiv 2605.26340) defines Chain-of-Evidence as: every claim must be
traceable, through a recorded chain, to a grounding source. It audits finished papers
with four integrity checks -- Score Verification, Specification Violation, Reference
Verification, Method-Code Alignment. Their failure mode is a paper whose citations were
invented. Ours is different in kind and worse: a campaign that makes confident claims
about its own experiments that trace to nothing -- a mechanism "tested" by a variant
whose edit never applied, an effect quoted from a run that was never valid, an axis
retired on a null from an intervention that never engaged. So the four checks are pushed
DOWN from the paper layer to the evidential layer, where our claims actually break.

AutoResearchClaw (arXiv 2605.20025 §3.4) contributes the NUMERIC REGISTRY: during
execution the system builds a whitelist of every value the runs produced; at drafting
time only registry values may be cited; afterwards a verifier re-extracts every number
and rejects the document if a number in a strict section has no registry entry. The
writing agent may READ the registry and may never modify it. That is exactly the
discipline a campaign needs when its own prose is the thing most likely to drift, so
E5 implements it against runs/sweep/results.

    coe.py                 # full audit
    coe.py --json
    coe.py registry        # every value a document is allowed to cite

Checks (each maps to a real failure this kind of campaign suffers):

  E1 SOURCE       (their Reference Verification) -- a claim must resolve to a full-text
                  snapshot on disk whose digest matches the corpus index. Existence is
                  not enough: an abstract may never back a claim.
  E2 LINK         (their Chain-of-Evidence)      -- mechanisms cite registered claims;
                  hypotheses cite registered mechanisms AND claims. No dangling edges.
  E3 ACTIVATION   (their Specification Violation)-- a result testing a hypothesis must
                  emit that hypothesis's activation diagnostic, and the rule must be
                  evaluated. Non-activation makes the run INCONCLUSIVE; it is never
                  evidence against the mechanism.
  E4 METHOD-CODE  (their Method-Code Alignment)  -- the variant that ran must differ
                  from the control variant, and its cfg must match the hypothesis it
                  claims to test. Catches the silent no-op edit and the identical-
                  ablation defect AutoResearchClaw screens for before spending budget.
  E5 NUMERIC      (AutoResearchClaw §3.4)        -- every number in a paper, round or
                  critique must trace to runs/sweep/results. Unmatched numbers in a
                  strict document are a rejection, not a warning.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import claims as C
import direction as _d      # noqa: E402
import lit              # noqa: E402

RESULTS = REPO / "runs" / "sweep" / "results"
VARIANTS = REPO / "runs" / "sweep" / "variants"
QUEUE = REPO / "runs" / "sweep" / "queue.json"

# Numbers that are notation rather than measurement, and must not be flagged.
NUM_RE = re.compile(r"(?<![\w.])(\d+\.\d{3,})(?![\w])")   # 3+ decimals = a measurement
ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}\b")            # a paper is a legitimate source
# A number computed from the FROZEN SOURCE -- an update RMS implied by a layer shape, a
# hard-coded coefficient, a parameter count -- has real provenance: it is recomputable
# from a file under version control. That is not the drift this check exists to catch,
# which is prose inventing EXPERIMENTAL results. Naming the file counts as sourcing it.
SRCFILE_RE = re.compile(r"\b(?:baseline|tools|host|karpathy_pristine)/[\w./]+\.py\b")


def _citation_tokens() -> set:
    """Identifiers that count as naming a source on the line where a number appears."""
    import claims as _c
    toks = {c["belief_key"] for c in _c.claims()}
    toks |= {m["name"] for m in _c.mechanisms()}
    toks |= {h["id"] for h in _c.hypotheses()}
    toks |= {l["id"] for l in _c.lessons()}
    toks |= {r.get("name", "") for r in _results()}
    return {t for t in toks if t}
STRICT_DOCS = ("papers", "rounds", "critiques")


def _results() -> list[dict]:
    out = []
    if RESULTS.is_dir():
        for f in sorted(RESULTS.glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except (OSError, ValueError):
                continue
    return out


def registry() -> dict:
    """AutoResearchClaw's verified registry: every value a document may cite.

    Built ONLY from result records. Nothing else may write to it, and no agent may add
    a value by asserting it in prose -- which is the whole point.
    """
    reg = {}
    for r in _results():
        for k, v in (r.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                reg.setdefault(round(float(v), 6), []).append(f"{r['name']}.{k}")
    return reg


def _fmt(vals) -> str:
    return ", ".join(sorted(vals)[:3]) + ("..." if len(vals) > 3 else "")


def e1_source() -> list[str]:
    """Every claim resolves to a full-text snapshot whose digest matches the index."""
    bad = []
    unfetched: set = set()
    ix = lit.load_index()
    for c in C.claims():
        sid = c.get("source_id")
        snap = lit.SOURCES / f"arxiv_{sid}_fulltext.txt"
        if not snap.exists():
            # A source that is INDEXED with a digest but not fetched into this checkout is
            # verifiable in principle and simply absent here. lit/sources is 19MB of
            # third-party full texts, re-fetchable with `tools/lit.py fetch`, and shipping
            # it is a redistribution decision rather than an evidence one. Failing on its
            # absence would make CI test the checkout instead of the claims -- 706 breaks
            # on a fresh clone, none of them about the science. A source that is not even
            # INDEXED is a different matter and stays a hard break: nothing records what it
            # was supposed to be.
            rec = ix.get(sid) or {}
            if rec.get("status") == "fetched" and rec.get("sha256"):
                unfetched.add(sid)
                continue
            bad.append(f"claim '{c['belief_key']}' cites {sid} with no snapshot on disk "
                       f"and no fetched entry in the corpus index -- nothing records what "
                       f"that source was")
            continue
        rec = ix.get(sid) or {}
        if rec.get("status") != "fetched":
            bad.append(f"claim '{c['belief_key']}': {sid} is not marked fetched in the "
                       f"corpus index (status {rec.get('status')!r}) -- triage is not evidence")
        want = rec.get("sha256")
        got = hashlib.sha256(snap.read_bytes()).hexdigest()
        if want and want != got:
            bad.append(f"claim '{c['belief_key']}': snapshot digest for {sid} does not "
                       f"match the index -- the source changed under the claim")
        if not c.get("locator"):
            bad.append(f"claim '{c['belief_key']}' has no locator into its source")
    if unfetched:
        print(f"  note: {len(unfetched)} indexed source(s) are not fetched in this "
              f"checkout; their digests are recorded and `python3 tools/lit.py fetch` "
              f"restores them. Not counted as breaks.")
    return bad


def e2_link() -> list[str]:
    """No dangling edges anywhere in claim -> mechanism -> hypothesis."""
    bad = []
    ck = {c["belief_key"] for c in C.claims()}
    mk = {m["name"] for m in C.mechanisms()}
    for m in C.mechanisms():
        if not m.get("claim_keys"):
            bad.append(f"mechanism '{m['name']}' cites no claims -- it is an assertion")
        for k in m.get("claim_keys") or []:
            if k not in ck:
                bad.append(f"mechanism '{m['name']}' -> unregistered claim '{k}'")
    for h in C.hypotheses():
        for k in h.get("claim_keys") or []:
            if k not in ck:
                bad.append(f"hypothesis '{h['id']}' -> unregistered claim '{k}'")
        for n in h.get("mechanism_names") or []:
            if n not in mk:
                bad.append(f"hypothesis '{h['id']}' -> unregistered mechanism '{n}'")
    return bad


def _rule_ok(rule: dict, value: float) -> bool:
    op, want = rule.get("op"), rule.get("value")
    try:
        return {"gt": value > want, "lt": value < want, "ge": value >= want,
                "le": value <= want, "ne": value != want,
                "abs_gt": abs(value) > want}[op]
    except (KeyError, TypeError):
        return False


def _live(h: dict, hyps: dict) -> dict:
    """Follow `supersedes` forward: the hypothesis whose activation test now governs."""
    seen, cur = {h["id"]}, h
    while True:
        nxt = next((c for c in hyps.values() if c.get("supersedes") == cur["id"]), None)
        if nxt is None or nxt["id"] in seen:
            return cur
        seen.add(nxt["id"]); cur = nxt


def e3_activation() -> list[str]:
    """A result testing a hypothesis must emit that hypothesis's activation diagnostic."""
    bad = []
    hyps = {h["id"]: h for h in C.hypotheses()}
    # A hypothesis whose activation test was WRONG is corrected by registering a new one
    # that names the old in `supersedes` -- append-only, the same way lessons retire. The
    # results keep citing the id they actually ran under, which is the honest record, but
    # they are judged by the corrected test rather than by the broken one. Without this a
    # fixed defect would keep failing the audit forever, and a permanently failing audit
    # is one the campaign learns to ignore.
    hyps = {hid: _live(h, hyps) for hid, h in hyps.items()}
    for r in _results():
        hid = r.get("hypothesis_id")
        if not hid:
            continue
        h = hyps.get(hid)
        if h is None:
            bad.append(f"result '{r['name']}' cites unregistered hypothesis '{hid}'")
            continue
        if hid in {e for l in C.lessons() if l.get("type") == "non_activation"
                   for e in (l.get("evidence") or [])}:
            continue          # documented non-activation; see the lesson, not the audit
        # POST-HOC RULE CHANGE. Superseding re-judges COMPLETED runs under the new rule,
        # which is right when a broken test is fixed and is a laundering channel when the
        # result is already known: an E3 failure can be cleared by registering a friendlier
        # successor. It was used exactly that way -- R6XF_P1_s0_treat failed
        # secmom_ortho_ratio lt 0.1 on the campaign's best val_bpb, and a successor with a
        # different diagnostic cleared it within the hour. That may well be correct (the
        # original rule was backwards on physics grounds), but it must never be SILENT.
        #
        # So a live hypothesis registered AFTER the run it now blesses must carry a written
        # justification. The field is the cost: it cannot be satisfied by editing a number,
        # only by saying in the record why the rule changed and why the change does not
        # depend on the outcome.
        if h.get("id") != hid and not h.get("post_hoc_rule_change"):
            _reg = str(h.get("registered_at") or "")
            bad.append(
                f"result '{r['name']}' ran under '{hid}' but is judged by its successor "
                f"'{h.get('id')}' (registered {_reg or 'unknown'}), which changed the "
                f"activation rule AFTER the run completed. That is permitted, but the "
                f"successor must carry a 'post_hoc_rule_change' field stating why the rule "
                f"was wrong independently of this result -- otherwise a failing activation "
                f"can be cleared by rewriting the test.")
            continue
        act = h.get("activation") or {}
        diag = act.get("diagnostic")
        met = r.get("metrics") or {}
        if diag not in met:
            bad.append(f"result '{r['name']}': activation diagnostic '{diag}' was NOT "
                       f"emitted -- the run is INCONCLUSIVE about '{hid}', not evidence "
                       f"against it. (Check the variant actually injects the print.)")
            continue
        if not _rule_ok(act.get("rule") or {}, met[diag]):
            bad.append(f"result '{r['name']}': '{diag}'={met[diag]} fails the declared "
                       f"activation rule {act.get('rule')} -- INCONCLUSIVE, not a null")
    bad += _e3_diagnostic_discriminates(hyps)
    return bad


def _e3_diagnostic_discriminates(hyps) -> list[str]:
    """An activation diagnostic the CONTROL also passes proves nothing about engagement.

    Checking only that the field exists and clears its rule was not enough, and the gap was
    not hypothetical: hyp_precond_pre_r1_v3 declared secmom_clamp_frac < 0.01, which every
    treatment satisfied -- and so did every control, because the field reads 0.0 in both
    arms. The same hypothesis also leaned on secmom_ortho_ratio, which the shipped edit
    pinned at one bit-identical value across all four treatments: a constant of the code
    path that no run could ever have failed. Both are recorded in
    L030_activation_predicate_the_control_also_passes, and both are caught here.

    A diagnostic earns its name by SEPARATING the arms, so this compares against the
    controls rather than reading the treatment alone. Reported as a problem on the
    hypothesis, not on any single run: no individual result is wrong, the test is.
    """
    import direction
    out = []
    rows = _results()
    ctl = [r for r in rows if direction.is_platform(r.get("cfg") or {})
           and r.get("ok") and (r.get("metrics") or {}).get("val_bpb")]
    # A non-activation that has been REGISTERED as a lesson is accounted for, not
    # outstanding. hyp_qk_suppress_r3 declared a diagnostic QK-norm pins to 1.0, so its
    # runs can never activate and no rerun of THIS design will change that; leaving the
    # break open would freeze the gate permanently on a defect already written down,
    # which teaches the campaign to ignore its own audit. The exemption is deliberately
    # narrow: it requires a lesson of type `non_activation` that names the hypothesis in
    # its evidence, so nothing can be silenced without being documented first.
    _excused = {e for l in C.lessons() if l.get("type") == "non_activation"
                for e in (l.get("evidence") or [])}
    for hid, h in sorted(hyps.items()):
        if hid in _excused:
            continue
        act = h.get("activation") or {}
        diag, rule = act.get("diagnostic"), act.get("rule") or {}
        if not diag or not rule:
            continue
        vals = [r["metrics"][diag] for r in rows
                if r.get("hypothesis_id") == hid and diag in (r.get("metrics") or {})]
        if not vals:
            continue                      # nothing ran yet; E3's existence check owns this
        cvals = [r["metrics"][diag] for r in ctl if diag in r["metrics"]]
        passing = [v for v in cvals if _rule_ok(rule, v)]
        if cvals and len(passing) == len(cvals):
            out.append(
                f"hypothesis '{hid}': every CONTROL also passes activation rule "
                f"{rule} on '{diag}' ({len(cvals)} control(s), e.g. {passing[0]}) -- the "
                f"diagnostic does not separate the arms, so it cannot show the mechanism "
                f"engaged (L030_activation_predicate_the_control_also_passes)")
        # Constancy is only a DEFECT when the diagnostic also fails to separate the arms.
        # A deterministic structural count -- n_ve_layers is 8 in every ve=1 treatment and
        # 4 in every control -- is bit-identical by construction and discriminates
        # perfectly; flagging it called a correct hypothesis broken. The failure this rule
        # exists for is secmom_ortho_ratio, constant at 15.08494568 with NO control reading
        # that fails the rule, so nothing could distinguish engagement from the code path.
        # If any control emits the diagnostic and FAILS the rule, discrimination is
        # demonstrated and constancy is a virtue.
        discriminates = any(not _rule_ok(rule, v) for v in cvals)
        if len(vals) > 1 and len(set(vals)) == 1 and not discriminates:
            out.append(
                f"hypothesis '{hid}': '{diag}' is bit-identical ({vals[0]}) across all "
                f"{len(vals)} runs citing it -- a constant of the code path, not a "
                f"measurement, so no run could ever have failed the test "
                f"(L030_activation_predicate_the_control_also_passes)")
    return out


def e4_method_code() -> list[str]:
    """The variant that ran must differ from the control, and match its hypothesis."""
    bad = []
    try:
        queue = json.loads(QUEUE.read_text())
    except (OSError, ValueError):
        return bad
    import direction
    import make_variant
    ctl_src = None
    for q in queue:
        if direction.is_platform(q.get("cfg") or {}):
            v = VARIANTS / q["variant"]
            if v.exists():
                ctl_src = v.read_text()
                break
    hyps = {h["id"]: h for h in C.hypotheses()}
    for q in queue:
        # Entries that have ALREADY RUN are settled: they executed with the bytes they
        # referenced, recorded by content hash. Regenerating them under today's generator
        # answers a question nobody asked -- of course it differs, the generator gained
        # instrumentation and new knobs since. The check that matters is on PENDING
        # entries, where a mismatch means the experiment would run code other than the one
        # it declares. Scoping it turns 64 historical breaks into the few that can bite.
        if (RESULTS / f"{q['name']}.json").exists():
            continue
        v = VARIANTS / q.get("variant", "")
        if not v.exists():
            # A variant is a CONTENT-ADDRESSED function of its cfg: the filename IS the
            # hash of the generated source. A checkout without the bytes can therefore
            # verify the reference by regenerating from cfg and comparing hashes, which
            # proves more than shipping the file would -- it also proves the generator
            # still produces what ran. Generated variants stay untracked as derived
            # artifacts; 192 "missing variant" breaks on a fresh clone were reporting the
            # checkout rather than the evidence. A HASH MISMATCH is a real break and a
            # serious one: the generator changed under a queued experiment.
            try:
                regen = make_variant.variant_id(make_variant.build(q.get("cfg") or {}))
            except Exception as exc:                       # noqa: BLE001
                bad.append(f"queue entry '{q['name']}' has no variant on disk and its cfg "
                           f"no longer builds: {str(exc)[:80]}")
                continue
            if regen != q.get("variant"):
                bad.append(f"queue entry '{q['name']}' references variant "
                           f"{q.get('variant')!r} but its cfg now generates {regen!r} -- "
                           f"the generator changed under a queued experiment")
            continue
        cfg = q.get("cfg") or {}
        # ctl_src must be the control THIS round would build, not whichever platform
        # entry happens to sit first in the queue -- that is the long-retired bootstrap
        # control, so a treatment that degenerated to the CURRENT control's bytes would
        # sail past this check. Rebuild it from PLATFORM instead of trusting queue order.
        try:
            ctl_src = make_variant.build(dict(direction.PLATFORM))
        except Exception:                                    # noqa: BLE001
            pass
        if ctl_src is not None and not direction.is_platform(cfg):
            if v.read_text() == ctl_src:
                bad.append(f"queue entry '{q['name']}' generated a variant BYTE-IDENTICAL "
                           f"to the control: the intervention did not apply. This is the "
                           f"identical-ablation defect; do not spend GPU time on it.")
        hid = q.get("hypothesis_id")
        if hid:
            h = hyps.get(hid)
            if h is None:
                bad.append(f"queue entry '{q['name']}' cites unregistered hypothesis '{hid}'")
            else:
                want = (h.get("intervention") or {}).get("cfg") or {}
                diff = {k: (want.get(k), cfg.get(k)) for k in want
                        if k in cfg and want[k] != cfg[k]}
                if diff:
                    bad.append(f"queue entry '{q['name']}' cfg disagrees with hypothesis "
                               f"'{hid}': {diff} -- the run would not test what was declared")
    return bad


def e5_lessons() -> list[str]:
    """Lessons carry conclusions and numbers, and escaped the numeric audit entirely.

    e5_numeric walks papers/, rounds/ and critiques/ for *.md, so lit/lessons.jsonl was
    never read -- yet a lesson is exactly where a conclusion hardens into something the
    queue doors then ENFORCE via blocks_values. A fabricated figure there is more
    consequential than one in prose, not less.

    A lesson states derived statistics by nature (a paired mean, an sd, a resolution),
    so demanding every number be a raw registry value would be wrong. What it MUST have
    is declared provenance: a non-empty `evidence` list naming registered runs, claims or
    hypotheses. That is document-level sourcing, the same concession E5 already makes for
    rounds and critiques -- and unlike free prose, a lesson has a structured field for it,
    so there is no excuse for it being absent.
    """
    bad = []
    reg = registry()
    known = _citation_tokens()
    for l in C.lessons():
        blob = " ".join(str(l.get(k, "")) for k in
                        ("observation", "diagnosis", "mitigation", "applies_when"))
        nums = [m.group(1) for m in NUM_RE.finditer(blob)]
        if not nums:
            continue
        ev = [e for e in (l.get("evidence") or []) if e in known]
        if not ev:
            ungrounded = [n for n in nums
                          if round(float(n), 6) not in reg
                          and not any(abs(round(float(n), 6) - k) < 5e-6 for k in reg)]
            if ungrounded:
                bad.append(f"lesson '{l['id']}' states {len(ungrounded)} measurement-like "
                           f"number(s) ({', '.join(ungrounded[:4])}) with NO registered "
                           f"evidence to derive them from -- a conclusion with enforcement "
                           f"power and no provenance")
    return bad


def e5_numeric(strict_only: bool = True) -> list[str]:
    """Every measurement-looking number in a document traces to a result record."""
    bad = []
    reg = registry()
    for d in STRICT_DOCS:
        root = REPO / d
        if not root.is_dir():
            continue
        for f in sorted(root.glob("*.md")):
            txt = f.read_text(errors="replace")
            unmatched = []
            # A number needs PROVENANCE, and for a deliberative document that provenance
            # is often a paper rather than one of our runs. A council round argues from
            # published effect sizes by design -- requiring every figure in it to appear
            # in a registry built solely from runs/sweep/results would fail every round
            # that actually read the literature, which inverts the rule's purpose. So:
            # papers/ stays strict (it REPORTS our results and may cite nothing else),
            # while rounds/ and critiques/ additionally accept a number whose own line
            # names its source -- a registered belief_key, an arXiv id, a lesson/hypothesis
            # /mechanism id, or one of our own run names. A bare number with no source on
            # its line is still a violation, which is the drift this check exists to catch.
            cited = _citation_tokens() if d in ("rounds", "critiques") else set()
            # papers/ stays STRICT -- it reports our own results and may cite nothing
            # else -- but the template requires a derived number (a difference, a mean, a
            # ratio) to be "shown as a derivation from cited registry values". There was
            # no mechanism implementing that, so a paper could state no derived statistic
            # at all: not a paired mean, not a standard deviation, not a t. The rule below
            # is that mechanism, and it is deliberately harsher than the rounds/critiques
            # hatch: there, naming a source token on the line is enough; here the line
            # must carry at least TWO actual registry values, i.e. the arithmetic itself
            # must be on the page. A derivation you cannot check is not a derivation.
            # Critiques get BOTH hatches, and this is a strengthening rather than a
            # loosening. A critique is the document that recomputes the most -- its whole
            # job is checking the campaign's arithmetic against runs/sweep/results -- and
            # it was getting the WEAKEST check: name a source token on the line. A dense
            # recomputation that shows its work in full ("(0.990387-0.991751, ...) =
            # (-0.001364, -0.001731)") failed, while a bare number beside a hypothesis id
            # passed. That is exactly backwards. Shown, checkable arithmetic now grounds a
            # value here too, on the same verified-result terms papers are held to.
            derivation_lines = d in ("papers", "critiques")
            lines = txt.splitlines()
            # Provenance is declared once per document, not restated at every mention.
            # A derived statistic (a band, an offset, a resolution) is computed from the
            # registry and then referred to repeatedly; requiring the citation on every
            # line would push authors to bloat prose rather than to source it. So collect
            # the values this document DOES source on some line, and accept those values
            # wherever else they appear in the same document.
            declared = set()
            if cited:
                for ln in lines:
                    if (any(tok in ln for tok in cited) or ARXIV_RE.search(ln)
                            or SRCFILE_RE.search(ln)):
                        for mm in NUM_RE.finditer(ln):
                            declared.add(round(float(mm.group(1)), 6))
            if derivation_lines:
                # Derivations CHAIN: a line may build on a quantity an earlier line
                # derived, which is how real analysis works -- four paired deltas give a
                # mean, three means give a sum, a sum and a mean give a shortfall.
                # Requiring every line to bottom out in raw registry values would forbid
                # any second-order statistic, so this iterates to a fixpoint instead.
                # What it never allows is a line with fewer than two already-grounded
                # numbers: an assertion with no visible arithmetic stays a violation.
                # The arithmetic is CHECKED, not merely present. A first version of this
                # rule declared every number on any line carrying two registry values,
                # which an audit correctly called a laundering channel: a fabricated
                # figure could ride along beside two real ones. Now a line only declares
                # the values it actually DERIVES, via an explicit `A - B = C` (or `+`)
                # whose result is verified, and only when A and B are already grounded.
                # Everything else on such a line must stand on its own.
                import re as _re
                # Division and multiplication belong here too: a standard error is
                # sd / sqrt(n), a resolution is 2 * sigma / sqrt(2), and refusing those
                # forced real statistics out of the paper rather than catching fabricated
                # ones. The right-hand operand may be a plain integer or a small factor,
                # which raw numbers alone never are.
                DERIV = _re.compile(
                    r"(\d+\.\d{3,})\s*(-|\u2212|\+|/|\*|x)\s*"
                    r"(\d+(?:\.\d+)?)\s*=\s*([+\u2212-]?\d+\.\d{3,})")
                def _g(v):
                    return v in reg or v in declared or any(abs(v - k) < 5e-6 for k in reg)
                # Paired analysis is written compactly as tuples --
                # "(0.990387-0.991751, 0.989819-0.991550) = (-0.001364, -0.001731)" --
                # which is FULLY shown arithmetic that the scalar matcher cannot parse. It
                # is checked elementwise here rather than waved through: same grounding
                # requirement, same verification of the result, just tuple-shaped.
                TUP = _re.compile(r"\(([^()=]+)\)\s*=\s*\(([^()]+)\)")
                PAIR = _re.compile(r"(\d+\.\d{3,})\s*(-|\u2212)\s*(\d+\.\d{3,})")

                def _tuples(ln, grounded, _cited=cited):
                    # ARITHMETIC IS NOT ATTRIBUTION. Checking that a - b = c proves the
                    # subtraction, never that a and b belong to the same comparison: an
                    # audit built a fictitious mechanism that had never run and grounded
                    # its "result" by differencing two real but unrelated registry values.
                    # So a tuple derivation must ALSO name its source on the line. A run
                    # that never happened has no run name, hypothesis id or lesson id to
                    # cite, which is exactly the case this closes.
                    if _cited and not (any(tok in ln for tok in _cited)
                                       or ARXIV_RE.search(ln) or SRCFILE_RE.search(ln)):
                        return []
                    got = []
                    for lhs, rhs in TUP.findall(ln):
                        ops = PAIR.findall(lhs)
                        res = [x for x in _re.findall(r"[+\u2212-]?\d+\.\d{3,}", rhs)]
                        if not ops or len(ops) != len(res):
                            continue
                        for (a, _o, b), c in zip(ops, res):
                            av, bv = round(float(a), 6), round(float(b), 6)
                            cv = round(float(c.replace("\u2212", "-")), 6)
                            if not (grounded(av) and grounded(bv)):
                                continue
                            if abs((av - bv) - cv) < 5e-6:
                                # Both signs: the scanner that reports violations captures
                                # magnitudes, so declaring only the signed value would
                                # ground a number the audit never asks about.
                                got.append(cv)
                                got.append(abs(cv))
                    return got

                for _ in range(8):
                    before = len(declared)
                    for ln in lines:
                        for _cv in _tuples(ln, _g):
                            declared.add(_cv)
                        for a, op, b, c in DERIV.findall(ln):
                            av, bv = round(float(a), 6), round(float(b), 6)
                            craw = c.replace("\u2212", "-")
                            cv = round(float(craw), 6)
                            # Only the left operand must be grounded. The right may be a
                            # plain count or factor -- the 2 in sd/2, the sqrt(2) in a
                            # resolution -- which is arithmetic, not a measurement.
                            if not _g(av):
                                continue
                            # SIGNED, and the operator on the page must be the operator
                            # actually performed. The first version stripped the sign from
                            # both sides and accepted either operation, so "0.993970 -
                            # 0.989520 = +0.004450" passed with the sign inverted, and a
                            # written minus was accepted whenever only the SUM matched.
                            # Two independent audits defeated it that way within an hour.
                            if op in "-\u2212":
                                want = av - bv
                            elif op == "+":
                                want = av + bv
                            elif op == "/":
                                want = av / bv if bv else None
                            else:
                                want = av * bv
                            if want is None:
                                continue
                            if abs(want - cv) < 5e-6:
                                declared.add(round(abs(cv), 6))
                                declared.add(cv)
                        # A line that lists >=3 already-grounded values may declare a
                        # summary statistic over them (a mean, an sd): the inputs are all
                        # visible and checkable by the reader on that same line.
                        # The >=3-grounded-values allowance existed so a line listing
                        # four per-device deltas could also state their mean and sd. As
                        # written it declared ANY number on such a line, which both audits
                        # used to launder arbitrary figures. A summary statistic of a set
                        # is bounded by that set: a mean lies inside [min, max], and a
                        # spread cannot exceed the range. Numbers outside those bounds are
                        # not summaries of what is on the line and must stand on their own.
                        vals = [round(float(mm.group(1)), 6) for mm in NUM_RE.finditer(ln)]
                        gr = [v for v in vals if _g(v)]
                        if len(gr) >= 3:
                            lo, hi = min(gr), max(gr)
                            rng = hi - lo
                            # A MEAN of the listed values must lie between them. A
                            # SPREAD cannot exceed their range -- but that clause alone
                            # admitted any number smaller than the range, which for a
                            # tight set of deltas means almost any plausible-looking
                            # figure. It now applies only where the line actually claims
                            # to state a spread, so a fabricated number must at least be
                            # asserted as a named statistic of the values beside it
                            # rather than merely sitting quietly among them.
                            says_spread = any(w in ln.lower() for w in
                                              ("sd", "sem", "spread", "stdev", "deviation",
                                               "sigma", "error"))
                            for v in vals:
                                if lo - 5e-6 <= v <= hi + 5e-6:
                                    declared.add(v)
                                elif says_spread:
                                    # Do not take the author's word for a spread when the
                                    # values are right there. Accepting anything under the
                                    # range let "sd 0.000123" ride beside three real
                                    # numbers. The sd, population sd and standard error of
                                    # the grounded values are computable, so compute them
                                    # and require the claimed figure to BE one of them.
                                    import statistics as _s
                                    cand = {round(_s.stdev(gr), 6),
                                            round(_s.pstdev(gr), 6),
                                            round(_s.stdev(gr) / len(gr) ** 0.5, 6),
                                            round(rng, 6)}
                                    if any(abs(v - c) < 5e-6 for c in cand):
                                        declared.add(v)
                    if len(declared) == before:
                        break
            for m in NUM_RE.finditer(txt):
                val = round(float(m.group(1)), 6)
                if val in reg:
                    continue
                # tolerate a derived value that rounds onto a registry entry
                if any(abs(val - k) < 5e-6 for k in reg):
                    continue
                if val in declared:
                    continue
                unmatched.append(m.group(1))
            if unmatched:
                uniq = sorted(set(unmatched))
                bad.append(f"{f.relative_to(REPO)}: {len(uniq)} number(s) with no registry "
                           f"entry: {', '.join(uniq[:8])}"
                           f"{'...' if len(uniq) > 8 else ''}. Either they came from a run "
                           f"that is not in runs/sweep/results, or they were computed and "
                           f"must be shown as a derivation from cited registry values.")
    return bad


CHECKS = (("E1 SOURCE      ", e1_source),
          ("E2 LINK        ", e2_link),
          ("E3 ACTIVATION  ", e3_activation),
          ("E4 METHOD-CODE ", e4_method_code),
          ("E5 NUMERIC     ", e5_numeric))


def audit() -> dict:
    out, total = {}, 0
    for name, fn in CHECKS:
        try:
            probs = fn()
        except Exception as exc:                      # noqa: BLE001
            probs = [f"check crashed: {exc!r}"]
        out[name.strip()] = probs
        total += len(probs)
    return {"ok": total == 0, "total": total, "checks": out}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "registry":
        reg = registry()
        print(f"{len(reg)} citable values from {len(_results())} result record(s)")
        for v, src in sorted(reg.items()):
            print(f"  {v:<16} {_fmt(src)}")
        if not reg:
            print("  (empty: no results yet, so NO number may be cited in any document)")
        return 0

    a = audit()
    if "--json" in sys.argv:
        print(json.dumps(a, indent=1))
        return 0 if a["ok"] else 1
    print(f"CHAIN OF EVIDENCE: {'INTACT' if a['ok'] else str(a['total']) + ' BREAK(S)'}\n")
    for name, _ in CHECKS:
        probs = a["checks"][name.strip()]
        print(f"  [{'ok ' if not probs else 'FAIL'}] {name} {len(probs)} problem(s)")
        for p in probs[:6]:
            print(f"          - {p}")
        if len(probs) > 6:
            print(f"          ... {len(probs)-6} more")
    return 0 if a["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
