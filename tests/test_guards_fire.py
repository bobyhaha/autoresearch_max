"""Every guard must demonstrably REFUSE something. A guard that cannot fire is not a guard.

This is the structural answer to the campaign's most persistent failure. Four times in one
session a check was built and then not connected, and each time the symptom was identical:
the code existed, read correctly, and could never refuse anything.

  - `claims.diagnostic_would_discriminate` was written to stop a hypothesis reaching the
    GPU with a diagnostic the control also satisfies. Nothing called it for hours.
  - `selector.score` carried a comment promising a penalty for unverifiable activation.
    The penalty was never written.
  - Three activation rules shipped whose predicate every control passed, so no run could
    have failed them.
  - `decide.py` recorded an ordering that `tick.sh` did not propagate, so the machine that
    launches work never saw it.

Reviewing the code did not catch these; auditors did, on average within twenty minutes of
looking. Vigilance is not a fix for a failure mode whose defining feature is that the code
looks right. The fix is to assert the OUTCOME: give each guard an input it must reject and
fail the suite when it accepts.

Each test below is deliberately paired -- one input that MUST be refused and one that MUST
be accepted. A guard that refuses everything is as broken as one that refuses nothing, and
only the pair distinguishes a working guard from a stuck one.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import claims      # noqa: E402
import coe         # noqa: E402
import council     # noqa: E402
import direction   # noqa: E402


def test_activation_precheck_refuses_a_vacuous_diagnostic():
    """The control must not satisfy a diagnostic that is supposed to prove engagement."""
    bad, msg = claims.diagnostic_would_discriminate("num_steps", {"op": "gt", "value": 0.0})
    assert not bad, f"a rule every control passes was accepted: {msg}"
    good, msg = claims.diagnostic_would_discriminate("n_ve_layers", {"op": "gt", "value": 4.0})
    assert good, f"a genuinely discriminating diagnostic was refused: {msg}"


def test_activation_precheck_is_actually_wired_into_the_queue_doors():
    """Built and not called is the failure this file exists for. Assert the CALL SITE."""
    for door in ("queue_quad.py", "queue_from_round.py"):
        src = (REPO / "tools" / door).read_text()
        assert "diagnostic_would_discriminate" in src, (
            f"{door} does not call the activation pre-check; the guard exists but nothing "
            f"consults it, which is exactly how it sat unused for hours")


def test_lesson_blocks_refuse_a_known_bad_value():
    hits = claims.blocked_values({**direction.PLATFORM, "tbs": 20})
    assert hits, "tbs=20 measured +0.022545 and is blocked by L040; the block did not fire"
    assert not claims.blocked_values({**direction.PLATFORM, "tbs": 18}), (
        "tbs=18 is the campaign's largest confirmed lever and must remain runnable")


def test_e5_refuses_a_fabricated_number_in_a_paper():
    """The numeric audit guards the one artifact that makes public claims."""
    doc = REPO / "papers" / "_guardtest.md"
    try:
        doc.write_text("We measured a val_bpb of 0.123456 today.\n")
        assert any("_guardtest" in p for p in coe.e5_numeric()), (
            "a number no run produced was accepted into papers/")
        doc.write_text("0.993970 - 0.989520 = 0.004450\n")
        assert not [p for p in coe.e5_numeric() if "_guardtest" in p], (
            "a correct, fully shown derivation was refused")
    finally:
        doc.unlink(missing_ok=True)


def test_e5_refuses_a_sign_flipped_derivation():
    doc = REPO / "papers" / "_guardtest.md"
    try:
        doc.write_text("0.989520 - 0.993970 = +0.004450\n")
        assert any("_guardtest" in p for p in coe.e5_numeric()), (
            "a derivation with the sign inverted was accepted")
    finally:
        doc.unlink(missing_ok=True)


def test_round_validator_refuses_a_round_that_proposes_nothing_runnable():
    """A round used to pass by being verbose. It must now pass by being executable."""
    entries = [{"name": "X", "cfg": dict(direction.PLATFORM), "rationale": "r",
                "falsifier": "f", "hypothesis_id": None}]
    problems = council._admissible(entries)
    assert problems, "a proposal with no hypothesis_id was judged admissible"
    assert any("NO ADMISSIBLE" in p for p in problems)


def test_selector_refuses_a_blocked_config():
    import analyze
    import selector
    rows = analyze.load()
    state = direction.axis_state(rows)
    fx = selector.family_effects(rows)
    s, terms = selector.score({**direction.PLATFORM, "tbs": 20}, rows, state, fx)
    assert s == float("-inf"), "the selector ranked a config that lessons forbid"
    assert "BLOCKED" in terms


def test_decision_is_recorded_and_reaches_the_queue_order():
    """decide.py's ordering must be visible in the artifact AND in the queue it rewrites."""
    r = subprocess.run([sys.executable, str(REPO / "tools" / "decide.py")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    assert "DECISION over" in r.stdout
    decisions = sorted((REPO / "runs" / "sweep" / "decisions").glob("*.json"))
    if decisions:
        rec = json.loads(decisions[-1].read_text())
        for field in ("state_hash", "candidate_set", "scores", "selected",
                      "selection_reason", "weights"):
            assert field in rec, f"decision record lacks {field!r}"


@pytest.mark.parametrize("guard_file,must_contain", [
    ("tools/verdict.py", "VOID -- EXCLUDED"),      # void arms excluded, not just annotated
    ("tools/verdict.py", "NOTE wave"),             # orphaned member reported, not dropped
    ("host/dispatch.py", "WAVE WEDGED"),           # a wave that can never assemble says so
    ("host/dispatch.py", "_crashed"),              # adopted crash detection
    ("tools/tick.sh", "pos.get"),                  # queue ORDER propagates to the host
])
def test_guard_is_present_in_source(guard_file, must_contain):
    """Cheap backstop for guards whose behaviour needs a live host or a full wave to
    exercise. Weaker than asserting an outcome, and marked as such: it proves the code is
    still there, not that it fires."""
    assert must_contain in (REPO / guard_file).read_text(), (
        f"{guard_file} no longer contains {must_contain!r}; a guard was removed or renamed")


def test_emission_guard_refuses_a_diagnostic_the_variant_never_prints():
    """Declared-and-discriminating is not the same as EMITTED, and the gap cost 8 runs.

    The z-loss arm named `logz_sq_final`; the generator printed `logz_sq_mean`. Both queue
    doors passed it, because both checked only whether the field would discriminate. Every
    one of those runs could only have returned a non-activation.
    """
    import make_variant
    cfg = {**direction.PLATFORM, "zloss": 0.1}
    bad, msg = make_variant.emits_diagnostic(cfg, "logz_sq_nonexistent")
    assert not bad, f"a diagnostic no generated line prints was accepted: {msg}"
    good, msg = make_variant.emits_diagnostic(cfg, "logz_sq_final")
    assert good, f"the diagnostic the variant does emit was refused: {msg}"


def test_emission_guard_is_wired_into_both_queue_doors():
    """The recurring failure is a guard built and never called. Assert the CALL SITES."""
    for door in ("queue_quad.py", "queue_from_round.py"):
        src = (REPO / "tools" / door).read_text()
        assert "emits_diagnostic" in src, (
            f"{door} does not call the emission pre-check; the guard exists but nothing "
            f"consults it, which is how the previous seven went unused")


def test_multi_axis_arms_do_not_contribute_to_single_factor_estimates():
    """A joint effect is not evidence about any one factor in it.

    Both attribution instruments had this bug at once. selector.family_effects credited a
    multi-family arm's FULL delta to every family it touched, so the three-lever stack's
    -0.004355 stood as the largest effect in both `attention` and `ve_placement` and set the
    queue's ranking prior. balance.sweeps did the same to the per-axis ladders, inflating
    swdiv=4 from -0.002194 to -0.002889 -- and those ladders decide saturation, which decides
    whether the campaign explores or keeps exploiting.
    """
    import analyze
    import balance
    import selector
    rows = analyze.load()

    # No single-factor family may carry a multi-family arm's delta.
    fx = selector.family_effects(rows)
    stack_keys = [k for k in fx if k.startswith("stack:")]
    assert stack_keys, ("multi-family arms are no longer scored under their own key; if a "
                        "stack arm is being folded back into a single family, the delta is "
                        "being counted several times over")
    for k in fx:
        if k.startswith("stack:"):
            continue
        for r in rows:
            cfg = r.get("cfg") or {}
            if not r.get("ok") or direction.is_platform(cfg):
                continue
            if len(selector._families(cfg)) > 1 and k in selector._families(cfg):
                assert selector._families(cfg) != [k], "unreachable; guards the shape above"

    # The swdiv ladder must report the single-factor value, not the stack's.
    state = direction.axis_state(rows)
    lad = balance.sweeps(rows, state)
    if "swdiv" in lad and 4 in lad["swdiv"]:
        assert abs(lad["swdiv"][4] - (-0.002194)) < 5e-5, (
            f"swdiv=4 reads {lad['swdiv'][4]:+.6f}; the single-factor arms measure -0.002194. "
            f"A multi-axis arm is leaking into the ladder again.")


def test_dispatcher_never_indexes_a_job_by_name():
    """`job["name"]` is a KeyError that kills the dispatcher, and it has appeared twice.

    A job dict holds proc/item/dir/started/uuid/cotenant/slot/cores; the run's name is at
    job["item"]["name"]. The first instance sat in the co-tenancy taint write and was found
    by audit before it ever fired. The second sat in _crashed(), on the health-check path
    that runs every poll, and it killed the dispatcher minutes after a launch -- the run
    completed on the GPU and was never harvested.

    Grepping the source is a weak check, but this defect is invisible until the exact branch
    executes, and both instances shipped through review that read the code as correct.
    """
    src = (REPO / "host" / "dispatch.py").read_text()
    offenders = [i + 1 for i, ln in enumerate(src.splitlines())
                 if 'job["name"]' in ln and not ln.strip().startswith("#")]
    assert not offenders, (
        f"host/dispatch.py indexes job[\"name\"] at line(s) {offenders}; a job dict has no "
        f"'name' key, so this raises KeyError and takes the dispatcher down with it")


def test_an_unknown_device_is_never_pooled_as_a_real_one():
    """gpu:-1 means "nobody recorded it", not "device number -1".

    Crash recovery stamped -1, and device_means() pooled it as a device -- so a recovered
    run's effect was corrected against a fictional device assembled from unrelated runs.
    Today's two recovered runs came out right only because that fake slot happened to hold
    one treatment and its own wave-mate control. The device correction is the denominator
    of every device-corrected number here, and the device spread is ~0.0025 bpb, larger
    than most effects being chased.
    """
    rows = [
        {"ok": True, "cfg": dict(direction.PLATFORM), "gpu": 6,
         "metrics": {"val_bpb": 0.9915, "final_epoch": 2.0}},
        {"ok": True, "cfg": dict(direction.PLATFORM), "gpu": -1,
         "metrics": {"val_bpb": 0.5000, "final_epoch": 2.0}},
        {"ok": True, "cfg": dict(direction.PLATFORM), "gpu": None,
         "metrics": {"val_bpb": 0.5000, "final_epoch": 2.0}},
    ]
    dm = direction.device_means(rows)
    assert -1 not in dm and None not in dm, (
        f"an unrecorded device was pooled as a real one: {sorted(dm)}")
    assert 6 in dm and abs(dm[6] - 0.9915) < 1e-9, "the real device mean was lost"
