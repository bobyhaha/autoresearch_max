"""Command-line interface for the structured research workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from .core import SchemaError
from .registry import ResearchRegistry


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="vibeautoresearch",
        description="Validate and summarize the automated scientific-research state.",
    )
    root.add_argument("--root", default="research", help="research-state directory")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate schemas, references, versions, and gates")
    commands.add_parser("summary", help="print registry counts and warnings")
    audit_parser = commands.add_parser(
        "audit", help="find deterministic refinement work without editing facts"
    )
    audit_parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any warning too (default: only error-severity fails)",
    )
    commands.add_parser(
        "list-challenges",
        help="show the ordered challenge catalog and sticky selection event",
    )
    select_challenge = commands.add_parser(
        "select-challenge",
        help="persist a challenge selection until it is changed or stopped",
    )
    select_challenge.add_argument("challenge")
    select_challenge.add_argument("--selected-by", default="operator")
    select_challenge.add_argument("--reason", required=True)
    stop_challenge = commands.add_parser(
        "stop-challenge",
        help="append a stop event so implicit checks and launches fail closed",
    )
    stop_challenge.add_argument("--selected-by", default="operator")
    stop_challenge.add_argument("--reason", required=True)
    commands.add_parser(
        "hourly-report-status",
        help="show whether the active campaign owes its next summary paper",
    )
    publish_report = commands.add_parser(
        "publish-hourly-report",
        help="validate structured JSON input and publish an immutable Markdown paper",
    )
    publish_report.add_argument(
        "input",
        help="JSON file containing the hourly paper body fields",
    )
    commands.add_parser("render-state", help="regenerate knowledge/RESEARCH_STATE.md")
    commands.add_parser(
        "render-literature",
        help="refresh the generated registry/scope block in LITERATURE_SYNTHESIS.md",
    )
    setup = commands.add_parser(
        "check-setup", help="verify the day-one reference reconciliation and frozen harness"
    )
    setup.add_argument(
        "--scope",
        default=None,
        help="compatibility selector; if supplied it must match the sticky active challenge",
    )
    gate = commands.add_parser(
        "check-gate", help="deterministically check whether a proposal may be frozen"
    )
    gate.add_argument("experiment_id")
    gate.add_argument(
        "--scope",
        default=None,
        help="compatibility selector; if supplied it must match the sticky active challenge",
    )
    authorize = commands.add_parser(
        "authorize-run", help="authorize one exact run from a current scoped gate"
    )
    authorize.add_argument("experiment_id")
    authorize.add_argument("arm_id")
    authorize.add_argument("seed", type=int)
    authorize.add_argument("max_steps", type=int)
    authorize.add_argument(
        "--scope",
        default=None,
        help="compatibility selector; if supplied it must match the sticky active challenge",
    )
    evaluate = commands.add_parser(
        "evaluate-stage",
        help="evaluate a gated experiment's staged stopping/promotion verdict",
    )
    evaluate.add_argument("experiment_id")
    snapshot = commands.add_parser(
        "snapshot-state", help="save an immutable generated research-state snapshot"
    )
    snapshot.add_argument("label", help="lowercase snake_case label, usually a date")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    registry = ResearchRegistry(Path(args.root))
    if args.command == "list-challenges":
        catalog = registry.challenge_catalog()
        event = registry.challenge_events()[-1]
        print(
            json.dumps(
                {
                    "challenges": list(catalog.ordered()),
                    "latest_selection": event.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "select-challenge":
        event = registry.append_challenge_event(
            action="activated",
            challenge_selector=args.challenge,
            selected_at=datetime.now(timezone.utc).isoformat(),
            selected_by=args.selected_by,
            reason=args.reason,
        )
        print(json.dumps(event.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "stop-challenge":
        latest = registry.challenge_events()[-1]
        event = registry.append_challenge_event(
            action="stopped",
            challenge_selector=latest.challenge_id,
            selected_at=datetime.now(timezone.utc).isoformat(),
            selected_by=args.selected_by,
            reason=args.reason,
        )
        print(json.dumps(event.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "hourly-report-status":
        print(
            json.dumps(
                registry.hourly_report_status(),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "publish-hourly-report":
        input_path = Path(args.input)
        try:
            content = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaError(
                f"cannot read hourly report input {input_path}: {exc}"
            ) from exc
        if not isinstance(content, Mapping):
            raise SchemaError("hourly report input must be a JSON object")
        report, paper_path = registry.publish_hourly_report(content)
        print(
            json.dumps(
                {
                    "report": report.to_dict(),
                    "paper": str(paper_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "audit":
        report = registry.audit().to_dict()
        print(json.dumps(report, indent=2, sort_keys=True))
        counts = report.get("counts", {})
        # A findings report that always exits 0 is a smoke alarm wired to a
        # decorative light: CI can never catch a regression. Errors always fail;
        # --strict also fails on warnings (visible debt) so a zero-debt gate exists.
        errors = int(counts.get("error", 0))
        warnings = int(counts.get("warning", 0))
        if errors or (args.strict and warnings):
            return 1
        return 0
    if args.command == "render-state":
        path = registry.write_state()
        print(json.dumps({"generated": str(path)}, indent=2, sort_keys=True))
        return 0
    if args.command == "render-literature":
        path = registry.write_literature_synthesis()
        print(json.dumps({"generated": str(path)}, indent=2, sort_keys=True))
        return 0
    if args.command == "check-setup":
        scope_id = registry.resolve_scope_id(args.scope)
        setup = registry._require_current_setup(scope_id)
        scope = setup.scope_for(scope_id) if scope_id else None
        selection = registry.challenge_events()[-1]
        challenge = registry.challenge_for_scope(scope_id)
        print(
            json.dumps(
                {
                    "status": scope["status"] if scope else setup.status,
                    "fingerprint": setup.fingerprint,
                    "challenge_id": challenge["challenge_id"],
                    "challenge_selection_fingerprint": selection.fingerprint,
                    "scope_id": scope_id,
                    "scope_key": dict(scope["scope_key"] if scope else setup.scope_key),
                    "run_env": dict(scope.get("run_env", {})) if scope else {},
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "check-gate":
        print(
            json.dumps(
                registry.check_gate(args.experiment_id, scope_id=args.scope),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "evaluate-stage":
        print(
            json.dumps(
                registry.evaluate_search_stage(args.experiment_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "authorize-run":
        print(
            json.dumps(
                registry.authorize_run(
                    args.experiment_id,
                    args.arm_id,
                    args.seed,
                    args.max_steps,
                    scope_id=args.scope,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "snapshot-state":
        path = registry.write_snapshot(args.label)
        print(json.dumps({"snapshot": str(path)}, indent=2, sort_keys=True))
        return 0
    report = registry.validate()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


__all__ = ["main", "parser"]
