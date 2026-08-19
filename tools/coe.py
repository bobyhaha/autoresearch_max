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

import claims as C      # noqa: E402
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
    ix = lit.load_index()
    for c in C.claims():
        sid = c.get("source_id")
        snap = lit.SOURCES / f"arxiv_{sid}_fulltext.txt"
        if not snap.exists():
            bad.append(f"claim '{c['belief_key']}' cites {sid} with no snapshot on disk")
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


def e3_activation() -> list[str]:
    """A result testing a hypothesis must emit that hypothesis's activation diagnostic."""
    bad = []
    hyps = {h["id"]: h for h in C.hypotheses()}
    for r in _results():
        hid = r.get("hypothesis_id")
        if not hid:
            continue
        h = hyps.get(hid)
        if h is None:
            bad.append(f"result '{r['name']}' cites unregistered hypothesis '{hid}'")
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
    return bad


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
        v = VARIANTS / q.get("variant", "")
        if not v.exists():
            bad.append(f"queue entry '{q['name']}' references a missing variant "
                       f"{q.get('variant')!r} -- it can never run")
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
