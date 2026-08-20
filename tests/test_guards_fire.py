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


def _needs_corpus(pred, why):
    """Skip an integration check when the corpus lacks the runs it reads.

    L069 recorded a test that expired when the campaign PRODUCED data. These are the
    mirror: they expire when a tree does not HAVE it. A fresh clone of this project must
    come up green, or the first thing a newcomer learns is that the suite lies.
    """
    if not pred():
        pytest.skip(f"integration check needs campaign data: {why}")


def test_activation_precheck_refuses_a_vacuous_diagnostic():
    """The control must not satisfy a diagnostic that is supposed to prove engagement."""
    import analyze
    _needs_corpus(lambda: any(direction.is_platform(r.get("cfg") or {})
                              and "num_steps" in (r.get("metrics") or {})
                              for r in analyze.load()),
                  "the pre-check needs CONTROL runs to compare against; a fresh tree "
                  "has none, so it correctly answers 'cannot pre-check' instead of refusing")
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
    _needs_corpus(lambda: any(l.get("blocks_values") or l.get("blocks_keys")
                              for l in claims.lessons()),
                  "needs a registered blocking lesson; a fresh tree has none")
    hits = claims.blocked_values({**direction.PLATFORM, "tbs": 20})
    assert hits, "tbs=20 measured +0.022545 and is blocked by L040; the block did not fire"
    assert not claims.blocked_values({**direction.PLATFORM, "tbs": 18}), (
        "tbs=18 is the campaign's largest confirmed lever and must remain runnable")


def test_e5_refuses_a_fabricated_number_in_a_paper():
    """The numeric audit guards the one artifact that makes public claims."""
    _needs_corpus(lambda: (REPO / "papers").is_dir() and bool(coe.registry()),
                  "E5 needs a papers/ directory and a non-empty numeric registry")
    doc = REPO / "papers" / "_guardtest.md"
    try:
        doc.write_text("We measured a val_bpb of 0.123456 today.\n")
        assert any("_guardtest" in p for p in coe.e5_numeric()), (
            "a number no run produced was accepted into papers/")
        # Build the derivation from values the registry ACTUALLY holds. Hard-coding two
        # of this campaign's own val_bpb figures made the test pass here and fail in any
        # other tree, because those numbers are not in a fresh corpus -- the same
        # corpus-coupling L069 was written about, in its other direction.
        vals = sorted(v for v in coe.registry() if isinstance(v, float) and v > 0.5)
        a, b = vals[-1], vals[0]
        doc.write_text(f"{a:.6f} - {b:.6f} = {a - b:.6f}\n")
        assert not [p for p in coe.e5_numeric() if "_guardtest" in p], (
            "a correct, fully shown derivation was refused")
    finally:
        doc.unlink(missing_ok=True)


def test_e5_refuses_a_sign_flipped_derivation():
    _needs_corpus(lambda: (REPO / "papers").is_dir() and bool(coe.registry()),
                  "E5 needs a papers/ directory and a non-empty numeric registry")
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
    _needs_corpus(lambda: "tbs" in claims.blocking_keys()
                  or any(l.get("blocks_keys") for l in claims.lessons()),
                  "needs a registered blocking lesson to refuse against")
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
    _needs_corpus(lambda: len(rows) > 50,
                  "multi-axis attribution needs the swdiv and stack arms on disk")

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


def test_emission_guard_fails_closed_when_the_variant_cannot_be_built():
    """A guard that answers "yes" when it cannot look is worse than no guard.

    emits_diagnostic returned True on any build exception, so an unbuildable cfg passed the
    emission check at BOTH queue doors and the log showed a passed check.
    """
    import make_variant
    ok, msg = make_variant.emits_diagnostic({"dbs": "not-an-int", "nonsense": object()}, "x")
    assert not ok, f"an unbuildable cfg was reported as emitting the diagnostic: {msg}"


def test_systematic_regime_shift_requires_magnitude_not_just_consistency():
    """Direction consistency is not causal evidence; a coin landing the same way twice isn't.

    The rule reported "the epoch gap is caused BY the treatment" for any mismatch seen in
    every arm. An audit built a fixture with a sub-band delta and a ~1% step wobble
    straddling the boundary in both waves and got that claim asserted with confidence.
    """
    src = (REPO / "tools" / "verdict.py").read_text()
    assert "_big_throughput" in src and "_resolvable" in src, (
        "the systematic-regime-shift branch no longer requires a large throughput change "
        "AND a resolvable effect; consistency alone would again be treated as cause")
    i = src.index("systematic = (")
    window = src[i:i + 700]
    assert "_big_throughput" in window and "_resolvable" in window, (
        "the magnitude guards exist but are not part of the systematic predicate")


def test_explore_debt_does_not_count_controls_against_exploration():
    """A control is the instrument, not a choice between exploring and exploiting.

    Controls sat in the denominator and could never be in the numerator, so with 62% of
    valid runs being controls the metric was permanently biased toward "explore more" --
    and permanently disagreed with balance.py, which an audit called instrument-shopping.
    """
    import analyze
    rows = analyze.load()
    state = direction.axis_state(rows)
    real = direction.explore_debt(rows, state)

    # Adding pure controls must not move the debt: they are not evidence either way.
    ctl = {"ok": True, "cfg": dict(direction.PLATFORM),
           "metrics": {"val_bpb": 0.9915, "final_epoch": 2.0}, "gpu": 6}
    padded = direction.explore_debt(rows + [dict(ctl) for _ in range(40)], state)
    assert abs(real - padded) < 1e-9, (
        f"40 extra controls moved explore_debt from {real:+.4f} to {padded:+.4f}; the "
        f"instrument is being counted as a research decision")


def test_integrity_lesson_must_report_a_sibling_search():
    """Fixing one instance of a defect class and not looking for the rest is the
    campaign's most repeated failure -- three separate times in a single day."""
    base = {"id": "x", "type": "integrity", "observation": "o", "diagnosis": "d",
            "action": "refine", "applies_when": "w", "severity": 0.5,
            "families": ["systems"], "evidence": ["e"], "mitigation": "m"}
    bad = claims.validate_lesson(base)
    assert any("sibling_search" in b for b in bad), (
        "an integrity lesson with no sibling search was accepted")
    ok = claims.validate_lesson({**base, "sibling_search": "grepped host/ and tools/ for "
                                 "the same call shape; found none"})
    assert not any("sibling_search" in b for b in ok), (
        "a lesson that DID report its sibling search was refused")

    # A valid_negative is a scientific result, not a defect class, and must not be burdened.
    sci = claims.validate_lesson({**base, "type": "valid_negative"})
    assert not any("sibling_search" in b for b in sci), (
        "the sibling-search requirement leaked onto scientific lessons")


def test_build_refuses_a_cfg_key_the_policy_never_heard_of():
    """Silently ignoring an unknown key returns the CONTROL source byte-for-byte.

    Ten registered hypotheses named cfg keys in neither KNOB_AXES nor MECHANISMS, so
    build() dropped them and produced a control. The registry advertised ten experiments
    that could never run, and only the byte-identical check at the queue door would have
    noticed -- after a round had already spent a proposal on one.
    """
    import make_variant
    with pytest.raises(Exception) as e:
        make_variant.build({**direction.PLATFORM, "periln_branch_norm": 1})
    assert "KNOB_AXES" in str(e.value) or "not in direction" in str(e.value)
    # And a knob the policy DOES know must still build.
    assert make_variant.build({**direction.PLATFORM, "zloss": 0.1})


def test_activation_precheck_refuses_a_rule_no_treatment_can_satisfy():
    """Checking only that CONTROLS FAIL is half a check, and it shipped twice.

    L050 recorded the first: secmom_ortho_ratio lt 0.1, where the reorder moves the
    statistic UP to 15.08 while controls sit near 0.49 -- so controls "fail" the rule and
    so does every treatment. The second arrived four hours later with qk_q_rms_final lt
    1.0. Both were two-sided departures encoded one-sided, and the door passed both.
    """
    cfg = {**direction.PLATFORM, "ve": 1, "swdiv": 4, "precond": "pre"}
    _needs_corpus(
        lambda: any("secmom_ortho_ratio" in (r.get("metrics") or {})
                    for r in __import__("analyze").load()),
        "needs precond runs that emitted secmom_ortho_ratio")
    bad, msg = claims.diagnostic_would_discriminate(
        "secmom_ortho_ratio", {"op": "lt", "value": 0.1}, cfg)
    assert not bad, f"a rule no run of this arm can satisfy was accepted: {msg}"
    good, msg = claims.diagnostic_would_discriminate(
        "secmom_max", {"op": "gt", "value": 1.0}, cfg)
    assert good, f"the corrected, satisfiable rule was refused: {msg}"


def test_activation_precheck_says_unverified_rather_than_ok():
    """A permissive answer must not read as a positive.

    When no run of the arm has emitted the field, the honest answer is that nothing is
    known -- not "ok". A prior audit flagged the old wording as a false confirmation.
    """
    # The arm must be one that will NEVER have runs. A first version used the noqknorm
    # arm, which was runless when the test was written and had run by the time the suite
    # next executed -- so the test asserted a TRANSIENT state and went red the moment the
    # campaign produced the very data it was waiting for. A test whose truth expires when
    # an experiment lands is a broken test, not a broken guard.
    cfg = {**direction.PLATFORM, "swdiv": 4096}       # never queued, never will be
    _needs_corpus(
        lambda: any("qk_q_rms_final" in (r.get("metrics") or {})
                    for r in __import__("analyze").load()),
        "needs any run that emitted qk_q_rms_final")
    ok, msg = claims.diagnostic_would_discriminate(
        "qk_q_rms_final", {"op": "gt", "value": 1.0}, cfg)
    assert ok and "UNVERIFIED" in msg, (
        f"a not-yet-checkable rule did not announce itself as unverified: {msg}")


def test_the_precommit_hook_exists_and_runs_the_suite():
    """The suite was advisory, and two commits landed on red in a single session.

    Both times the failing test was correct and pointed at a real defect in the change
    being committed. Discipline did not hold, so the check does not rely on it.
    """
    hook = REPO / ".githooks" / "pre-commit"
    assert hook.exists(), "the pre-commit hook is gone; the suite is advisory again"
    src = hook.read_text()
    assert "pytest" in src or "-m pytest" in src, "the hook does not run the suite"
    assert "exit 1" in src, "the hook does not REFUSE on a red suite"
    import os
    assert os.access(hook, os.X_OK), "the hook is not executable, so git will skip it"


def test_role_resolution_refuses_a_name_that_contradicts_its_cfg():
    """_is_ctl reads the role from the run NAME, which is weaker than reading the cfg.

    An audit demonstrated that a crafted name/cfg mismatch silently flips a member's role
    and SIGN-FLIPS the reported delta with no warning. The name is the right source -- it
    survives a platform adoption where is_platform does not -- but it must not contradict
    the configuration without saying so.
    """
    import verdict
    plat = dict(direction.PLATFORM)
    # A member NAMED as the control while carrying a treatment cfg.
    liar = {"name": "FAKE_P1_s1_ctrl", "cfg": {**plat, "noqknorm": 1}, "metrics": {}}
    honest = {"name": "FAKE_P1_s0_treat", "cfg": {**plat, "noqknorm": 1}, "metrics": {}}
    truth = {"name": "FAKE_P1_s1_ctrl", "cfg": dict(plat), "metrics": {}}
    assert verdict._is_ctl(truth), "a genuine control was not recognised"
    assert not verdict._is_ctl(honest), "a genuine treatment was misread as a control"
    assert verdict._role_conflict(liar), (
        "a member named _ctrl while carrying a treatment cfg was accepted silently")
    assert not verdict._role_conflict(truth), "an honest control was flagged as conflicting"


def _mk(cfg, val, steps, tps, ok=True, epoch=2.0, gpu=6, name="X"):
    return {"name": name, "cfg": dict(cfg), "ok": ok, "gpu": gpu,
            "metrics": {"val_bpb": val, "num_steps": steps, "tokens_per_step": tps,
                        "final_epoch": epoch}}


def test_step_law_fits_controls_only_at_one_operating_point():
    """The law must not be influenced by the treatments it judges, nor pool operating points.

    It was quoted from prose for hours with no code behind it (L055), used to argue two
    mechanisms shared a cause, and nobody could check what it was fitted on.
    """
    P = dict(direction.PLATFORM)
    tps = float(2 ** P["tbs"])
    # val_bpb must FALL as steps rise, which is what the law describes; a first version
    # of this fixture had it rising and the slope assertion caught the fixture, not the code.
    rows = [_mk(P, 0.990 - i * 0.001, 1000 + i * 100, tps, name=f"c{i}") for i in range(10)]
    # A treatment at the same operating point must NOT enter the fit.
    rows.append(_mk({**P, "mlp": 2}, 5.0, 900, tps, name="wild_treatment"))
    # A control at a DIFFERENT operating point must NOT enter it either.
    rows.append(_mk({**P, "tbs": P["tbs"] + 1}, 5.0, 500, tps * 2, name="other_point"))
    law = direction.step_law(rows)
    assert law is not None and law["n"] == 10, (
        f"fit used n={law and law['n']}; it must use the 10 controls at this operating "
        f"point only, excluding treatments and other tokens_per_step")
    assert law["slope"] < 0, "more steps must predict LOWER val_bpb"


def test_step_law_refuses_across_a_tokens_per_step_boundary():
    """CLAUDE.md forbids applying the law across a change in tokens-per-step.

    Returning a number anyway is how a rule that lives in prose gets ignored in practice.
    """
    P = dict(direction.PLATFORM)
    tps = float(2 ** P["tbs"])
    rows = [_mk(P, 0.990 - i * 0.001, 1000 + i * 100, tps, name=f"c{i}") for i in range(10)]
    treat = _mk({**P, "tbs": P["tbs"] - 1}, 0.99, 2000, tps / 2, name="t")
    ctrl = _mk(P, 0.984, 1000, tps, name="c")
    assert direction.step_law_explains(rows, treat, ctrl) is None, (
        "the law was applied across a tokens_per_step boundary, which CLAUDE.md forbids")


def test_step_law_flags_extrapolation_beyond_its_fitted_range():
    """A share read off an extrapolation must say so; today's arms ran outside the range."""
    P = dict(direction.PLATFORM)
    tps = float(2 ** P["tbs"])
    rows = [_mk(P, 0.990 - i * 0.0005, 1900 + i * 10, tps, name=f"c{i}") for i in range(10)]
    ctrl = _mk(P, 0.9840, 1950, tps, name="c")
    inside = _mk({**P, "swdiv": 4}, 0.9835, 1960, tps, name="t_in")
    outside = _mk({**P, "swdiv": 4}, 0.9820, 2400, tps, name="t_out")
    r_in = direction.step_law_explains(rows, inside, ctrl)
    r_out = direction.step_law_explains(rows, outside, ctrl)
    assert r_in and not r_in["extrapolated"], "an in-range arm was flagged as extrapolated"
    assert r_out and r_out["extrapolated"], (
        "an arm far outside the fitted step range was NOT flagged; every quality-cost "
        "figure quoted today depends on this flag being right")


def test_step_law_baseline_survives_a_platform_adoption():
    """Adoption must not destroy the ability to analyse what came before it (L065)."""
    P = dict(direction.PLATFORM)
    retired = {**P, "tbs": P["tbs"] + 1}
    tps_retired = float(2 ** retired["tbs"])
    rows = [_mk(retired, 0.992 - i * 0.0004, 900 + i * 20, tps_retired, name=f"o{i}")
            for i in range(10)]
    assert direction.step_law(rows, tps_retired) is None, (
        "the retired point fitted without an explicit baseline; those runs are no longer "
        "is_platform, so this must require the baseline to be named")
    law = direction.step_law(rows, tps_retired, retired)
    assert law is not None and law["n"] == 10, (
        "passing the retired baseline did not recover the law -- adoption would have made "
        "every historical analysis unreproducible")


def test_systematic_regime_shift_requires_a_real_device_swap():
    """The verdict line SAYS "on swapped slots"; nothing used to check that it was true.

    Two arms with the treatment on the same device satisfied every other condition, so the
    line asserted a counterbalancing it had not verified. An audit built the fixture.
    """
    src = (REPO / "tools" / "verdict.py").read_text()
    i = src.index("systematic = (")
    window = src[i:i + 500]
    assert "_treat_devs" in window, (
        "the systematic predicate no longer requires the treatment to occupy more than one "
        "device, so it can claim a swap that did not happen")
    assert "len(_treat_devs) >= 2" in window, (
        "the device-swap condition is present but not part of the predicate")


def test_a_registered_mechanism_becomes_buildable_and_visible_without_editing_a_list():
    """Upstream: "Everything is fair game: architecture, hyperparameters, optimizer...".

    This file used to carry a CLOSED set of five hand-wired mechanisms, and every policy
    tool computed coverage over it -- so "what should we try next" could only return an
    answer from inside the menu, and an unimplemented idea produced a refusal that read
    as a verdict about the world. Registering a mechanism must now be the single act
    that makes it real.
    """
    import make_variant
    reg = make_variant.MECHANISM_REGISTRY
    assert reg, "the mechanism registry is empty; new mechanisms cannot be added as code"
    name = next(iter(reg))
    assert name in direction.all_mechanisms(), (
        f"{name} is implemented but the policy cannot see it -- the registry is not "
        f"feeding direction.all_mechanisms()")
    assert not direction.unknown_keys({**direction.PLATFORM, name: 1}), (
        f"a cfg using the registered mechanism {name} is still rejected as an unknown key")
    spec = reg[name]
    cfg = {**direction.PLATFORM, name: 32768 if name == "ngram" else 1}
    src = make_variant.build(cfg)
    assert src != make_variant.build(dict(direction.PLATFORM)), (
        f"{name} built byte-identical to the control -- the edit did not apply")
    ok, msg = make_variant.emits_diagnostic(cfg, spec["diagnostic"])
    assert ok, f"{name} does not emit its declared diagnostic {spec['diagnostic']}: {msg}"


def test_a_registered_mechanism_is_never_mistaken_for_a_control():
    """Opening one gate without its sibling is worse than leaving both shut.

    known_keys() learned to accept registered mechanisms before mechanisms_touched() did,
    so a cfg carrying one satisfied BOTH "moves no known axis" and "carries no unknown
    key" -- and is_platform() called it a CONTROL. Such a run would bypass the decision
    cutoff and every budget, and be pooled into the block that measures the noise band.
    """
    import make_variant
    for name in make_variant.MECHANISM_REGISTRY:
        cfg = {**direction.PLATFORM, name: 32768 if name == "ngram" else 1}
        assert direction.mechanisms_touched(cfg) == {name}, (
            f"{name} is registered but mechanisms_touched() cannot see it")
        assert not direction.is_platform(cfg), (
            f"a cfg engaging {name} was classified as a CONTROL; it would corrupt the "
            f"noise band it was pooled into")
        assert direction.label(cfg).startswith("mech:"), (
            f"{name} is not labelled as a mechanism: {direction.label(cfg)}")
    assert direction.is_platform(dict(direction.PLATFORM)), (
        "the platform itself stopped being recognised as a control")


# ---------------------------------------------------------------------------
# The registry is self-policing.
#
# `prefetch` sat in direction.MECHANISMS for the whole campaign with no implementation.
# The policy counted it toward mechanism coverage, printed it in the DIRECTION SPACE table
# as a reachable direction, and let a hypothesis register against it -- while building it
# raised VariantEditError, so nothing could ever launch on it. A name in a coverage table
# with no code behind it is worse than an absent mechanism: the campaign reports a
# direction as available and then never runs it, and no single tool is wrong enough to
# notice. These tests make that state unreachable.
# ---------------------------------------------------------------------------

def _all_mech_names():
    import direction
    return sorted(direction.all_mechanisms())


@pytest.mark.parametrize("name", _all_mech_names())
def test_every_declared_mechanism_actually_builds(name):
    """Declared and unbuildable is the defect this test exists for."""
    import ast, direction, make_variant
    cfg = {**direction.PLATFORM, **_probe_cfg(name)}
    src = make_variant.build(cfg)                      # raises if the edit target is gone
    ctl = make_variant.build(dict(direction.PLATFORM))
    assert src != ctl, f"{name} builds but is byte-identical to the control"
    ast.parse(src)                                     # the edit must produce valid Python


@pytest.mark.parametrize("name", _all_mech_names())
def test_every_mechanism_emits_its_own_diagnostic(name):
    """A mechanism whose diagnostic is never printed can only return non-activation.

    The z-loss arm shipped eight runs that could only ever come back INCONCLUSIVE because
    nothing emitted the number its activation rule tested.
    """
    import direction, make_variant
    spec = make_variant.MECHANISM_REGISTRY.get(name)
    if spec is None:
        pytest.skip(f"{name} is a legacy inline branch with no registry diagnostic")
    cfg = {**direction.PLATFORM, **_probe_cfg(name)}
    ok, msg = make_variant.emits_diagnostic(cfg, spec["diagnostic"])
    assert ok, f"{name} declares diagnostic {spec['diagnostic']!r} it never prints: {msg}"


@pytest.mark.parametrize("name", _all_mech_names())
def test_no_mechanism_is_ever_mistaken_for_a_control(name):
    """is_platform() deciding a mechanism arm is a control is the worst failure here.

    It would exempt the arm from the decision cutoff AND pool it into the very control
    block that measures the noise band -- corrupting the instrument with the effect.
    """
    import direction
    cfg = {**direction.PLATFORM, **_probe_cfg(name)}
    assert not direction.is_platform(cfg), f"{name} classified as a CONTROL"
    assert not direction.unknown_keys(cfg), f"{name} carries keys the policy cannot see"
    assert direction.mechanisms_touched(cfg) == {name}
    assert direction.label(cfg) == f"mech:{name}"


def _probe_cfg(name):
    """A minimal engaging value per mechanism, plus any companion parameter it needs."""
    import make_variant
    probe = {"mtp": 4, "unet": 1, "zloss": 1e-4, "noqknorm": 1, "precond": "pre",
             "ngram": 32768, "ngram_gate": 0.1, "prefetch": 2, "vefreeze": 1,
             "embwd": 0.0005, "periln": 1, "vnorm": 1, "ffnpost": 1, "ropefrac": 0.1,
             "winsched": 128}
    assert name in probe, (
        f"mechanism {name!r} has no probe value here, so it is untested. Add one: every "
        f"mechanism must be provably buildable before it can be offered as a direction.")
    cfg = {name: probe[name]}
    for p in (make_variant.MECHANISM_REGISTRY.get(name) or {}).get("params", ()):
        if p in probe:
            cfg[p] = probe[p]
    return cfg


@pytest.mark.parametrize("name", _all_mech_names())
def test_a_result_using_any_mechanism_flows_through_the_reading_loop(name):
    """The steering loop must survive a result from every mechanism it offers.

    `mechanism_state()` seeded its counters from the frozen MECHANISMS tuple while
    `mechanisms_touched()` iterated the registry, so the first result using a REGISTERED
    mechanism raised KeyError -- taking down analyze.py, direction.py and agenda.py
    together, while the dispatcher happily kept launching. The campaign would have gone
    blind at the exact moment its first new mechanism returned data.

    Nothing caught it because no test had ever pushed a registered mechanism's RESULT
    through the readers; the earlier guards only checked that such a cfg could be built
    and labelled. A mechanism is not integrated until its result can be read.
    """
    import direction
    cfg = {**direction.PLATFORM, **_probe_cfg(name)}
    res = [{"name": f"T_{name}", "ok": True, "gpu": 0,
            "metrics": {"val_bpb": 0.97, "num_steps": 1000}, "cfg": cfg},
           {"name": "C_ctrl", "ok": True, "gpu": 1,
            "metrics": {"val_bpb": 0.98, "num_steps": 1000},
            "cfg": dict(direction.PLATFORM)}]
    st = direction.mechanism_state(res)
    assert name in st, f"mechanism_state has no counter for {name}"
    assert st[name]["n"] == 1, f"{name} result was not counted: {st[name]}"
    direction.report(res)          # the DIRECTION SPACE table a council reads first


@pytest.mark.parametrize("name", _all_mech_names())
def test_every_registered_mechanism_reaches_the_rotation_table(name):
    """`family=` was written and never read, so rotation could not see 8 of 9 mechanisms.

    agenda.py decides DRY / STALE / HARD CAP from direction.mechanism_families(). While
    that table was hand-maintained, signal_path reported "all closed" with periln, vnorm
    and ffnpost sitting in it, and attention listed no mechanisms at all though winsched
    and ropefrac were registered there. Their families kept accruing staleness toward
    rotation as if nothing had been added, and their exploration gap stayed at 0.00.
    A declaration no consumer reads is a comment.
    """
    import direction, make_variant
    spec = make_variant.MECHANISM_REGISTRY.get(name)
    if spec is None:
        pytest.skip(f"{name} is a legacy inline branch, already in the hand-written table")
    fam = spec["family"]
    fams = direction.mechanism_families()
    assert fam in fams, f"{name} declares family {fam!r} which the rotation table lacks"
    assert name in fams[fam]["mechs"], f"{name} is missing from family {fam!r}"


def test_a_companion_param_without_its_mechanism_is_not_a_control():
    """`ngram_gate` alone is a typo, not an experiment -- and it read as a control.

    build() ignores a companion whose mechanism is absent, so the generated source equals
    the control's while the cfg looks like a treatment. is_platform() returned True and
    label() returned "control", which is the one classification that must never be wrong:
    it exempts the entry from the decision cutoff and pools it into the control block
    that measures the noise band.
    """
    import direction
    for orphan in ("ngram_gate", "winsched_frac"):
        cfg = {**direction.PLATFORM, orphan: 0.9}
        assert direction.orphan_params(cfg) == {orphan}
        assert not direction.is_platform(cfg), f"{orphan} alone classified as a CONTROL"
        assert direction.label(cfg).startswith("INVALID:"), direction.label(cfg)
    assert direction.is_platform(dict(direction.PLATFORM))


def test_build_refuses_the_crosses_its_own_docs_call_unsafe():
    """Prose is not a guard: the forbidden cross built cleanly until it was coded."""
    import make_variant, direction
    from make_variant import VariantEditError
    with pytest.raises(VariantEditError):
        make_variant.build({**direction.PLATFORM, "ropefrac": 0.1, "noqknorm": 1})
    with pytest.raises(VariantEditError):      # a schedule that starts at its own target
        make_variant.build({**direction.PLATFORM, "swdiv": 16, "winsched": 128})
    with pytest.raises(VariantEditError):      # decay that erases the tables it decays
        make_variant.build({**direction.PLATFORM, "embwd": 0.01})


def test_pre_convention_names_keep_their_role_through_an_adoption():
    """A run named `_control` is a control, whatever the platform has since become.

    verdict._is_ctl resolves the role from the run name precisely so a PLATFORM ADOPTION
    cannot retroactively re-role completed waves -- and then its fallback, for names
    predating the `<wave>_s<slot>_<role>` convention, went straight back to
    direction.is_platform. 28 runs named `_control` (C01..C06, every W0*_control, every
    ctrl_*) were classified as TREATMENTS because their cfg carries the old tbs=19.
    The bug the function exists to prevent, living in its own fallback.
    """
    import verdict, direction
    old = {**direction.PLATFORM, "tbs": 19}          # a control built before the adoption
    for name in ("C01_control", "W02a_1_control", "ctrl_VE_A", "ctrl_R1N_A_slot1"):
        assert verdict._is_ctl({"name": name, "cfg": dict(old)}), \
            f"{name} is named a control and must stay one across an adoption"
    # ...and a treatment whose NAME merely contains the word must not flip. `poscontrol`
    # is a positive control for the ns axis, which is an intervention arm.
    for name in ("ns3_poscontrol_slot0", "precond_pre_slot0", "wd040_slot0"):
        assert not verdict._is_ctl({"name": name, "cfg": {**old, "ns": 3}}), \
            f"{name} is a treatment and an unanchored substring test would invert it"
    # The slot convention still wins where it is present.
    assert verdict._is_ctl({"name": "R7XF_P1_s1_ctrl", "cfg": dict(direction.PLATFORM)})
    assert not verdict._is_ctl({"name": "R7XF_P1_s0_treat", "cfg": dict(direction.PLATFORM)})
