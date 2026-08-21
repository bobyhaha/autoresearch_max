"""Data-driven challenge catalog and sticky active-challenge selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .core import (
    SchemaError,
    check_fingerprint,
    fingerprint,
    json_mapping,
    json_mappings,
    local_id,
    require_bool,
    require_enum,
    require_int,
    require_keys,
    require_text,
    strings,
)


@dataclass(frozen=True)
class ChallengeCatalog:
    version: int
    challenges: tuple[Mapping[str, Any], ...]
    campaign_policy: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_int(self.version, "challenge_catalog.version", minimum=1)
        entries = json_mappings(self.challenges, "challenge_catalog.challenges")
        if not entries:
            raise SchemaError("challenge catalog must contain at least one challenge")
        identifiers: set[str] = set()
        selectors: dict[str, str] = {}
        orders: list[int] = []
        primary_count = 0
        for index, entry in enumerate(entries):
            field = f"challenge_catalog.challenges[{index}]"
            require_keys(
                entry,
                field,
                (
                    "challenge_id",
                    "order",
                    "name",
                    "scope_id",
                    "decision_frame",
                    "aliases",
                    "description",
                ),
            )
            challenge_id = local_id(entry["challenge_id"], f"{field}.challenge_id")
            if challenge_id in identifiers:
                raise SchemaError(f"duplicate challenge_id {challenge_id!r}")
            identifiers.add(challenge_id)
            order = require_int(entry["order"], f"{field}.order", minimum=1)
            orders.append(order)
            require_text(entry["name"], f"{field}.name")
            scope_id = str(entry["scope_id"])
            if scope_id:
                local_id(scope_id, f"{field}.scope_id")
            else:
                primary_count += 1
            decision_frame = local_id(
                entry["decision_frame"], f"{field}.decision_frame"
            )
            aliases = strings(entry["aliases"], f"{field}.aliases")
            require_text(entry["description"], f"{field}.description")
            for selector in (challenge_id, scope_id, decision_frame, *aliases):
                if not selector:
                    continue
                local_id(selector, f"{field}.selector")
                existing = selectors.get(selector)
                if existing is not None and existing != challenge_id:
                    raise SchemaError(
                        f"challenge selector {selector!r} is ambiguous between "
                        f"{existing!r} and {challenge_id!r}"
                    )
                selectors[selector] = challenge_id
        if sorted(orders) != list(range(1, len(entries) + 1)):
            raise SchemaError("challenge order must be the contiguous sequence 1..N")
        if primary_count != 1:
            raise SchemaError(
                "challenge catalog must identify exactly one top-level scope with scope_id=''"
            )
        campaign_policy = json_mapping(
            self.campaign_policy, "challenge_catalog.campaign_policy"
        )
        if campaign_policy:
            require_keys(
                campaign_policy,
                "challenge_catalog.campaign_policy",
                (
                    "direction_round_limit",
                    "direction_cooldown_rounds",
                    "hourly_reports_required",
                    "hourly_report_interval_minutes",
                ),
            )
            require_int(
                campaign_policy["direction_round_limit"],
                "challenge_catalog.campaign_policy.direction_round_limit",
                minimum=1,
            )
            require_int(
                campaign_policy["direction_cooldown_rounds"],
                "challenge_catalog.campaign_policy.direction_cooldown_rounds",
                minimum=1,
            )
            require_bool(
                campaign_policy["hourly_reports_required"],
                "challenge_catalog.campaign_policy.hourly_reports_required",
            )
            require_int(
                campaign_policy["hourly_report_interval_minutes"],
                "challenge_catalog.campaign_policy.hourly_report_interval_minutes",
                minimum=1,
            )

    def definition(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "campaign_policy": dict(self.campaign_policy),
            "challenges": [dict(item) for item in self.challenges],
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {**self.definition(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChallengeCatalog":
        record = cls(
            version=int(payload.get("version", 1)),
            challenges=tuple(
                json_mapping(item, f"challenge_catalog.challenges[{index}]")
                for index, item in enumerate(payload.get("challenges") or ())
            ),
            campaign_policy=json_mapping(
                payload.get("campaign_policy"),
                "challenge_catalog.campaign_policy",
            ),
        )
        check_fingerprint(payload, record.fingerprint)
        return record

    def ordered(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(item)
            for item in sorted(
                self.challenges,
                key=lambda entry: (int(entry["order"]), str(entry["challenge_id"])),
            )
        )

    def resolve(self, selector: str) -> dict[str, Any]:
        if selector == "":
            return next(dict(item) for item in self.challenges if item["scope_id"] == "")
        matches = [
            dict(item)
            for item in self.challenges
            if selector
            in {
                str(item["challenge_id"]),
                str(item["scope_id"]),
                str(item["decision_frame"]),
                *(str(alias) for alias in item["aliases"]),
            }
        ]
        if len(matches) != 1:
            known = sorted(
                str(item["challenge_id"]) for item in self.challenges
            )
            raise SchemaError(
                f"unknown or ambiguous challenge selector {selector!r}; "
                f"known challenges: {known}"
            )
        return matches[0]


@dataclass(frozen=True)
class ChallengeSelectionEvent:
    event_id: str
    generation: int
    version: int
    action: str
    challenge_id: str
    catalog_fingerprint: str
    challenge_fingerprint: str
    setup_fingerprint: str
    selected_at: str
    selected_by: str
    reason: str

    def __post_init__(self) -> None:
        local_id(self.event_id, "challenge_event.event_id")
        require_int(self.generation, "challenge_event.generation", minimum=1)
        require_int(self.version, "challenge_event.version", minimum=1)
        require_enum(
            self.action, "challenge_event.action", {"activated", "stopped"}
        )
        local_id(self.challenge_id, "challenge_event.challenge_id")
        for name, value in (
            ("catalog_fingerprint", self.catalog_fingerprint),
            ("challenge_fingerprint", self.challenge_fingerprint),
            ("setup_fingerprint", self.setup_fingerprint),
        ):
            text = require_text(value, f"challenge_event.{name}")
            if len(text) != 16 or any(char not in "0123456789abcdef" for char in text):
                raise SchemaError(f"challenge_event.{name} must be a 16-character fingerprint")
        require_text(self.selected_at, "challenge_event.selected_at")
        require_text(self.selected_by, "challenge_event.selected_by")
        require_text(self.reason, "challenge_event.reason")

    @property
    def registry_id(self) -> str:
        return self.event_id

    def definition(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "generation": self.generation,
            "version": self.version,
            "action": self.action,
            "challenge_id": self.challenge_id,
            "catalog_fingerprint": self.catalog_fingerprint,
            "challenge_fingerprint": self.challenge_fingerprint,
            "setup_fingerprint": self.setup_fingerprint,
            "selected_at": self.selected_at,
            "selected_by": self.selected_by,
            "reason": self.reason,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {**self.definition(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChallengeSelectionEvent":
        record = cls(
            event_id=str(payload.get("event_id", "")),
            generation=int(payload.get("generation", 0)),
            version=int(payload.get("version", 1)),
            action=str(payload.get("action", "")),
            challenge_id=str(payload.get("challenge_id", "")),
            catalog_fingerprint=str(payload.get("catalog_fingerprint", "")),
            challenge_fingerprint=str(payload.get("challenge_fingerprint", "")),
            setup_fingerprint=str(payload.get("setup_fingerprint", "")),
            selected_at=str(payload.get("selected_at", "")),
            selected_by=str(payload.get("selected_by", "")),
            reason=str(payload.get("reason", "")),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


def challenge_fingerprint(challenge: Mapping[str, Any]) -> str:
    return fingerprint(dict(challenge))


__all__ = [
    "ChallengeCatalog",
    "ChallengeSelectionEvent",
    "challenge_fingerprint",
]
