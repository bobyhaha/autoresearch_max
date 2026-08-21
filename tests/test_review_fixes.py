from __future__ import annotations

import json
import pathlib
import sys
import types

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))


def test_non_council_freeze_uses_last_authorized_checkpoint(tmp_path, monkeypatch):
    import gate

    monkeypatch.setattr(gate, "SWEEP", tmp_path)
    (tmp_path / "GATE_STATUS.json").write_text(json.dumps({
        "ts": 123.0,
        "decision_cutoff": 122.0,
        "decision_cutoff_reason": "",
    }))
    monkeypatch.setattr(gate.council, "status", lambda kind: {"ok": True})
    checks = [{"check": "chain of evidence intact", "ok": False, "hard": False,
               "freezes": True, "detail": "broken citation"}]

    cutoff, reason = gate.decision_cutoff(checks)

    assert cutoff == 123.0
    assert "broken citation" in reason


def test_missing_checkpoint_fails_closed_for_evidence_freeze(tmp_path, monkeypatch):
    import gate

    monkeypatch.setattr(gate, "SWEEP", tmp_path)
    monkeypatch.setattr(gate.council, "status", lambda kind: {"ok": True})
    cutoff, _ = gate.decision_cutoff([
        {"check": "chain of evidence intact", "ok": False, "hard": False,
         "freezes": True, "detail": "audit crashed"},
    ])
    assert cutoff == 0.0


def test_direct_queue_rejects_unknown_hypothesis_before_writing(tmp_path, monkeypatch):
    import queue_quad

    monkeypatch.setattr(queue_quad, "SWEEP", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "queue_quad.py", "--name", "bad-hyp", "--cfg", '{"ve": 3}',
        "--hyp", "definitely_not_registered", "--rationale", "r",
        "--falsifier", "f", "--expected", "e",
    ])

    assert queue_quad.main() == 1
    assert not (tmp_path / "queue.json").exists()


def test_quad_records_one_physical_affinity_group(tmp_path, monkeypatch):
    import queue_quad

    monkeypatch.setattr(queue_quad, "SWEEP", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "queue_quad.py", "--name", "affinity-probe", "--cfg", '{"ve": 3}',
        "--hyp", "none", "--rationale", "r", "--falsifier", "f",
        "--expected", "e",
    ])

    assert queue_quad.main() == 0
    queue = json.loads((tmp_path / "queue.json").read_text())
    assert len(queue) == 8
    assert {entry["counterbalance_group"] for entry in queue} == {"affinity-probe"}


def test_wave_admission_requires_explicit_group_and_both_roles():
    import admission
    import direction

    platform = dict(direction.PLATFORM)
    treatment = {**platform, "ve": 3}
    missing = admission.wave_issues([
        {"name": "x_treat", "cfg": treatment, "role": "treat"},
        {"name": "x_ctrl", "cfg": platform, "role": "ctrl"},
    ])
    assert any("missing non-empty wave_group" in issue for issue in missing)

    one_sided = admission.wave_issues([
        {"name": "x_treat", "cfg": treatment, "role": "treat", "wave_group": "x"},
    ])
    assert any("both a treatment and a control" in issue for issue in one_sided)


def test_decider_uses_recorded_roles(monkeypatch):
    import decide

    monkeypatch.setattr(decide, "SWEEP", pathlib.Path("/definitely/missing"))
    monkeypatch.setattr(decide.direction, "axis_state", lambda rows: {})
    monkeypatch.setattr(decide.selector, "family_effects", lambda rows: {})
    monkeypatch.setattr(
        decide.selector, "score",
        lambda cfg, rows, state, effects: (float(cfg["score"]), {"source": cfg["score"]}),
    )
    queue = [
        {"name": "w_ctrl", "wave_group": "w", "role": "ctrl", "cfg": {"score": 99}},
        {"name": "w_treat", "wave_group": "w", "role": "treat", "cfg": {"score": 1}},
    ]

    ranked = decide.decide([], queue)

    assert ranked[0]["score"] == 1.0
    assert ranked[0]["terms"]["source"] == "1"


def test_exploration_policy_has_lexicographic_priority():
    import direction
    import selector

    state = {"axes": {axis: {"n": 2} for axis in direction.KNOB_AXES}}
    never_tested = {**direction.PLATFORM, "dim": 768}
    state["axes"]["dim"]["n"] = 0
    tier, reason = selector.exploration_tier(never_tested, [], state)
    assert tier == 2 and "coverage floor" in reason

    state["axes"]["dim"]["n"] = 1
    tier, reason = selector.exploration_tier(never_tested, [], state)
    assert tier == 1 and "exploration floor" in reason

    state["axes"]["dim"]["n"] = 2
    assert selector.exploration_tier(never_tested, [], state) == (0, "")


def test_exploration_debt_is_chronological_not_retroactive():
    import direction

    state = {"axes": {axis: {"n": 0} for axis in direction.KNOB_AXES}}
    state["axes"]["dim"]["n"] = 3
    rows = [
        {"name": f"dim_{i}_treat", "role": "treat", "ok": True, "ended": i,
         "cfg": {**direction.PLATFORM, "dim": 768}}
        for i in range(1, 4)
    ]
    expected = direction.EXPLORE_FLOOR - 2 / 3
    assert abs(direction.explore_debt(rows, state) - expected) < 1e-12


def test_device_affinity_reuses_exact_uuid_tuple():
    source = (REPO / "host" / "dispatch.py").read_text()
    start = source.index("def counterbalance_key(")
    end = source.index("def runnable(")
    module = types.ModuleType("affinity_logic")
    module.__dict__.update({
        "hashlib": __import__("hashlib"), "json": __import__("json"),
        "recorded_role": lambda item: item.get("role"),
    })
    exec(compile(source[start:end], "affinity_logic", "exec"), module.__dict__)
    batch = [
        {"name": "p1_t", "counterbalance_group": "cmp"},
        {"name": "p1_c", "counterbalance_group": "cmp"},
    ]
    affinity = {}
    ordered, created = module.order_for_device_affinity(
        batch, [(4, "uuid-a"), (7, "uuid-b")], affinity)
    assert created and ordered[:2] == [(4, "uuid-a"), (7, "uuid-b")]
    assert affinity == {"cmp": ["uuid-a", "uuid-b"]}

    ordered, created = module.order_for_device_affinity(
        batch, [(1, "uuid-b"), (2, "uuid-c"), (3, "uuid-a")], affinity)
    assert not created
    assert ordered[:2] == [(3, "uuid-a"), (1, "uuid-b")]

    ordered, _ = module.order_for_device_affinity(
        batch, [(3, "uuid-a"), (2, "uuid-c")], affinity)
    assert ordered is None

    legacy_a = [{"role": "treat", "cfg": {"ve": 3}, "rationale": "r"},
                {"role": "ctrl", "cfg": {"ve": 2}, "rationale": "c"}]
    legacy_b = list(reversed(legacy_a))
    assert module.counterbalance_key(legacy_a) == module.counterbalance_key(legacy_b)


def test_analysis_prefers_physical_uuid_over_mutable_gpu_index():
    import direction
    import verdict

    row = {"gpu": 7, "gpu_uuid": "GPU-stable-id"}
    assert direction.device_id(row) == "GPU-stable-id"
    assert verdict._slot(row) == "GPU-stable-id"
    assert direction.device_id({"gpu": 7}) == 7


def test_queue_store_never_treats_corruption_as_empty(tmp_path):
    import queue_store

    queue = tmp_path / "queue.json"
    queue.write_text("not-json")
    try:
        queue_store.update_queue(queue, lambda rows: rows + [{"name": "new"}])
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("a corrupt queue was silently reset")
    assert queue.read_text() == "not-json"
