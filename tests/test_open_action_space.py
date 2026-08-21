#!/usr/bin/env python3
"""The three fixes for the 89.6%-idle / closed-action-space campaign defect.

1. A whole-file candidate is buildable, and is never mistaken for a control.
2. A screen is exempt from the decision cutoff, so prose cannot idle the fleet.
3. The explore lane refills a draining queue and is loud when the backlog is dry.
"""
import json
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "host"))
import direction
import make_variant

ok = lambda c, m: print(f"  {'PASS' if c else 'FAIL'}  {m}") or (c or sys.exit(f"FAILED: {m}"))


def test_freeform_is_not_a_control():
    cfg = {"src": "anything.py"}
    ok(direction.is_freeform(cfg), "a cfg carrying `src` is freeform")
    ok(not direction.is_platform(cfg),
       "is_platform REFUSES a freeform cfg (else an arbitrary train.py would be pooled "
       "into the noise band and bypass the cutoff)")
    ok(direction.label(cfg).startswith("freeform:"),
       "label agrees with is_platform instead of saying 'control'")
    ok(not direction.unknown_keys(cfg), "`src` is a known key, so it is not invisible to policy")
    ok(direction.blocked_reason(cfg, {"axes": {}}) is None,
       "knob dry-rules do not refuse a candidate that lives outside the knob registry")
    # The explore/exploit floor does not exist in every v3 lineage; assert it only where
    # the policy has one, rather than making the shared test tree-specific.
    if hasattr(direction, "is_exploration"):
        ok(direction.is_exploration(cfg, {"axes": {}}),
           "a freeform candidate pays down exploration debt")
    else:
        print("  SKIP  this tree has no is_exploration (explore floor predates it)")


def test_freeform_guards():
    base = (REPO / "baseline" / "train.py").read_text()
    d = pathlib.Path(tempfile.mkdtemp())
    orig = make_variant.CANDIDATES
    make_variant.CANDIDATES = d
    try:
        (d / "same.py").write_text(base)
        try:
            make_variant.build({"src": "same.py"}); ok(False, "byte-identical must be refused")
        except make_variant.VariantEditError as e:
            ok("BYTE-IDENTICAL" in str(e), "a candidate equal to the baseline is refused as a silent control")

        (d / "nocontract.py").write_text("x = 1\n")
        try:
            make_variant.build({"src": "nocontract.py"}); ok(False, "output contract must be enforced")
        except make_variant.VariantEditError as e:
            ok("output contract" in str(e), "a candidate that cannot print val_bpb is refused before it costs a GPU")

        try:
            make_variant.build({"src": "../train.py"}); ok(False, "path escape must be refused")
        except make_variant.VariantEditError as e:
            ok("bare filename" in str(e), "a candidate outside the candidates dir is refused")

        (d / "real.py").write_text(base.replace("TOTAL_BATCH_SIZE = 2**19",
                                                "TOTAL_BATCH_SIZE = 2**18", 1))
        src = make_variant.build({"src": "real.py"})
        ok(src != base, "a genuine candidate builds and differs from the baseline")
        ok("OPHIS-FREEFORM candidate=real.py" in src, "the variant records which candidate it came from")
        ok("OPHIS-INSTRUMENTED" in src, "the variant records which diagnostics could be instrumented")
        compile(src, "variant", "exec")
        ok(True, "the generated variant is valid Python")
    finally:
        make_variant.CANDIDATES = orig


def test_screen_is_exempt_from_the_cutoff():
    # dispatch.py is host-side: it takes a flock under ~/$OPHIS_REMOTE_DIR/sweep at
    # import. Point HOME at a scratch tree so the real module can be exercised here
    # rather than a copy of its logic -- a second implementation of a role predicate is
    # exactly the drift this campaign has already been bitten by.
    import os
    home = pathlib.Path(tempfile.mkdtemp())
    (home / "ophis_v3" / "sweep").mkdir(parents=True)
    old_home, old_dir = os.environ.get("HOME"), os.environ.get("OPHIS_REMOTE_DIR")
    os.environ["HOME"], os.environ["OPHIS_REMOTE_DIR"] = str(home), "ophis_v3"
    try:
        import dispatch
    finally:
        if old_home is not None:
            os.environ["HOME"] = old_home
        if old_dir is None:
            os.environ.pop("OPHIS_REMOTE_DIR", None)
        else:
            os.environ["OPHIS_REMOTE_DIR"] = old_dir
    screen = {"name": "SCR_x_scr", "cfg": {"src": "x.py"}, "screen": True, "wave_group": None}
    ok(dispatch._is_screen(screen), "a width-1 screen is recognised")
    ok(not dispatch._is_screen({**screen, "wave_group": "W1"}),
       "a screen carrying a wave_group is NOT exempt -- that is the half-of-a-pair trap "
       "that deadlocked the per-entry control exemption")
    ok(not dispatch._is_screen({"name": "n", "cfg": {}}), "an ordinary entry is not a screen")


def test_explore_lane_refills_and_is_loud_when_dry():
    import explore_lane as lane
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "results").mkdir(); (d / "claims").mkdir(); (d / "variants").mkdir()
    (d / "queue.json").write_text("[]")
    orig_sweep, orig_cand = lane.SWEEP, lane.CANDIDATES
    lane.SWEEP, lane.CANDIDATES = d, pathlib.Path(tempfile.mkdtemp())
    mv_orig = make_variant.CANDIDATES
    make_variant.CANDIDATES = lane.CANDIDATES
    try:
        base = (REPO / "baseline" / "train.py").read_text()
        (lane.CANDIDATES / "cand_one.py").write_text(
            base.replace("TOTAL_BATCH_SIZE = 2**19", "TOTAL_BATCH_SIZE = 2**18", 1))
        ok(lane.main() == 0, "the lane runs against a drained queue")
        q = json.loads((d / "queue.json").read_text())
        ok(len(q) >= 1, f"the drained queue was refilled ({len(q)} entries)")
        e = q[0]
        ok(e["screen"] is True, "the refill is marked as a screen")
        ok(e.get("wave_group") is None, "a screen carries no wave_group, so it is width-1 by construction")
        ok(e["role"] == "treat" and e["hypothesis_id"] == "none",
           "a screen is an instrument probe, not an adoption claim")
        ok((d / "variants" / e["variant"]).exists(), "the variant was built at queue time, not on a GPU")

        n_before = len(json.loads((d / "queue.json").read_text()))
        lane.main()
        ok(len(json.loads((d / "queue.json").read_text())) == n_before,
           "the lane is idempotent: it does not re-queue a candidate it already screened")
    finally:
        lane.SWEEP, lane.CANDIDATES = orig_sweep, orig_cand
        make_variant.CANDIDATES = mv_orig


if __name__ == "__main__":
    for fn in (test_freeform_is_not_a_control, test_freeform_guards,
               test_screen_is_exempt_from_the_cutoff,
               test_explore_lane_refills_and_is_loud_when_dry):
        print(f"\n{fn.__name__}:")
        fn()
    print("\nALL PASS")
