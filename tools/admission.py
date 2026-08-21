#!/usr/bin/env python3
"""One semantic preflight shared by council validation and queue producers."""
from __future__ import annotations

import hashlib
import json
import re

import claims
import direction
import make_variant


def entry_issues(entry: dict, results: list[dict]) -> list[str]:
    name, cfg = entry.get("name", "?"), entry.get("cfg") or {}
    role = direction.recorded_role(entry)
    issues = []
    unknown = direction.unknown_keys(cfg)
    if unknown:
        issues.append(f"unrecognised config keys {sorted(unknown)}")

    hits = claims.blocked_values(cfg)
    if hits:
        k, value, lesson, rule = hits[0]
        issues.append(f"{k}={value} is blocked by {lesson['id']} ({rule})")

    changed = {k for k, value in cfg.items() if direction.PLATFORM.get(k) != value}
    key_blocks = claims.blocking_keys()
    blocked = sorted(changed & set(key_blocks))
    if blocked:
        issues.append(f"key {blocked[0]!r} is blocked by {key_blocks[blocked[0]]['id']}")

    hid = entry.get("hypothesis_id")
    known = {h["id"]: h for h in claims.hypotheses()}
    if role == "treat" and not hid:
        issues.append("no hypothesis_id (use 'none' for an instrument probe)")
    elif hid and hid != "none" and hid not in known:
        issues.append(f"hypothesis_id {hid!r} is not registered")
    elif hid and hid != "none":
        activation = (known[hid].get("activation") or {})
        diagnostic, rule = activation.get("diagnostic"), activation.get("rule")
        if diagnostic and rule:
            ok, message = claims.diagnostic_would_discriminate(diagnostic, rule, cfg)
            if not ok:
                issues.append(f"activation diagnostic cannot discriminate: {message}")
            emitted, message = make_variant.emits_diagnostic(cfg, diagnostic)
            if not emitted:
                issues.append(f"declared diagnostic is not emitted: {message}")

    state = direction.axis_state(results)
    policy = direction.blocked_reason(cfg, state)
    if policy:
        issues.append(policy)

    try:
        make_variant.build(cfg)
    except Exception as exc:  # noqa: BLE001 - every build failure is an admission failure
        issues.append(f"variant does not build: {str(exc)[:120]}")
    return [f"{name}: {issue}" for issue in issues]


def grouped(entries: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        group = entry.get("wave_group")
        if isinstance(group, str) and group.strip():
            groups.setdefault(group, []).append(entry)
    return groups


def wave_issues(entries: list[dict]) -> list[str]:
    issues = []
    for entry in entries:
        if not isinstance(entry.get("wave_group"), str) or not entry["wave_group"].strip():
            issues.append(f"{entry.get('name', '?')}: missing non-empty wave_group")
        declared = str(entry.get("role") or "").strip().lower()
        named = re.search(r"_(treat|ctrl|control)$", str(entry.get("name") or ""), re.I)
        if declared and declared not in {"treat", "treatment", "ctrl", "control"}:
            issues.append(f"{entry.get('name', '?')}: invalid role {entry.get('role')!r}")
        elif not declared and not named:
            issues.append(
                f"{entry.get('name', '?')}: role is ambiguous; set role to treat or ctrl")
    for group, members in grouped(entries).items():
        controls = [e for e in members if direction.recorded_role(e) == "ctrl"]
        treatments = [e for e in members if direction.recorded_role(e) == "treat"]
        if not controls or not treatments:
            issues.append(f"wave {group!r} must contain both a treatment and a control")
            continue
        for treatment in treatments:
            cfg = treatment.get("cfg") or {}
            deltas = []
            for control in controls:
                ccfg = control.get("cfg") or {}
                keys = set(cfg) | set(ccfg)
                deltas.append(sorted(k for k in keys if cfg.get(k) != ccfg.get(k)))
            nearest = min(deltas, key=len)
            if not nearest:
                issues.append(f"{treatment.get('name', '?')}: treatment equals its control")
            try:
                treatment_source = make_variant.build(cfg)
                if any(treatment_source == make_variant.build(control.get("cfg") or {})
                       for control in controls):
                    issues.append(
                        f"{treatment.get('name', '?')}: generated variant is byte-identical "
                        "to an in-wave control")
            except Exception:
                pass  # entry_issues reports the actionable build failure
            reason = treatment.get("multifactor_reason") or treatment.get("multifactor")
            if len(nearest) > 1 and not reason:
                issues.append(
                    f"{treatment.get('name', '?')}: differs from its nearest control in "
                    f"{len(nearest)} keys {nearest}; add multifactor_reason or isolate one factor")
    return issues


def counterbalance_issues(entries: list[dict]) -> list[str]:
    issues = []
    positions: dict[str, list[int]] = {}
    names: dict[str, str] = {}
    for _group, members in grouped(entries).items():
        for position, entry in enumerate(members):
            if direction.recorded_role(entry) != "treat":
                continue
            key = json.dumps(entry.get("cfg") or {}, sort_keys=True)
            positions.setdefault(key, []).append(position)
            names[key] = entry.get("name", "?")
    for key, slots in positions.items():
        if len(set(slots)) < 2:
            issues.append(
                f"{names[key]}: NOT COUNTERBALANCED; treatment occupies slots {slots}")
    return issues


def comparison_group(members: list[dict]) -> str:
    """Stable physical-device affinity key shared by swapped waves."""
    signature = sorted(
        (direction.recorded_role(e), json.dumps(e.get("cfg") or {}, sort_keys=True),
         e.get("hypothesis_id"), (e.get("rationale") or "").strip(),
         (e.get("falsifier") or "").strip(), (e.get("expected") or "").strip())
        for e in members
    )
    digest = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
    return f"cb_{digest}"
