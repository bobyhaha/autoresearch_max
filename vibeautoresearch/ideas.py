"""Causal mechanisms and falsifiable observable–intervention hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .core import (
    SchemaError,
    check_fingerprint,
    fingerprint,
    json_mapping,
    json_mappings,
    local_id,
    require_enum,
    require_id,
    require_int,
    require_keys,
    require_number,
    require_text,
    strings,
    tags,
)


HYPOTHESIS_STATUSES = {
    "proposed",
    "offline_screened",
    "approved_for_pilot",
    "running",
    "validated",
    "rejected",
    "blocked",
    "deprecated",
}


@dataclass(frozen=True)
class MechanismRecord:
    mechanism_id: str
    name: str
    version: int
    description: str
    origin_type: str
    claim_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    causal_chain: tuple[str, ...]
    assumptions: tuple[str, ...]
    competing_mechanism_ids: tuple[str, ...]
    scope: Mapping[str, Any]
    observable_predictions: tuple[Mapping[str, Any], ...]
    intervention_predictions: tuple[Mapping[str, Any], ...]
    status: str
    tags: tuple[str, ...] = ()
    notes: str = ""
    campaign_batch_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.mechanism_id, "mechanism")
        require_text(self.name, "mechanism.name")
        require_int(self.version, "mechanism.version", minimum=1)
        require_text(self.description, "mechanism.description")
        require_enum(
            self.origin_type,
            "mechanism.origin_type",
            {"literature", "self_proposed", "mixed", "internal_experiment"},
        )
        for claim_id in self.claim_ids:
            require_id(claim_id, "claim")
        for observation_id in self.observation_ids:
            require_id(observation_id, "evidence", "mechanism.observation_ids")
        for campaign_batch_id in self.campaign_batch_ids:
            require_id(
                campaign_batch_id,
                "campaign_batch",
                "mechanism.campaign_batch_ids",
            )
        if self.origin_type == "literature" and not self.claim_ids:
            raise SchemaError("literature mechanism requires claim_ids")
        if self.origin_type == "self_proposed" and self.claim_ids:
            raise SchemaError("self-proposed mechanism cannot claim literature provenance")
        if self.origin_type == "internal_experiment":
            if self.claim_ids:
                raise SchemaError(
                    "internal-experiment mechanism cannot claim literature provenance"
                )
            if not self.observation_ids and not self.campaign_batch_ids:
                raise SchemaError(
                    "internal-experiment mechanism requires observation_ids or "
                    "campaign_batch_ids"
                )
        if len(strings(self.causal_chain, "mechanism.causal_chain", allow_empty=False)) < 2:
            raise SchemaError("mechanism.causal_chain requires at least two steps")
        strings(self.assumptions, "mechanism.assumptions")
        for competing_id in self.competing_mechanism_ids:
            require_id(competing_id, "mechanism")
        json_mapping(self.scope, "mechanism.scope")
        for index, prediction in enumerate(json_mappings(self.observable_predictions, "mechanism.observable_predictions")):
            require_keys(prediction, f"mechanism.observable_predictions[{index}]", ("observable_id", "expected_pattern"))
            require_id(prediction["observable_id"], "observable")
        for index, prediction in enumerate(json_mappings(self.intervention_predictions, "mechanism.intervention_predictions")):
            require_keys(prediction, f"mechanism.intervention_predictions[{index}]", ("intervention_id", "expected_change"))
            require_id(prediction["intervention_id"], "intervention")
        require_enum(self.status, "mechanism.status", {"proposed", "active", "challenged", "deprecated"})
        tags(self.tags, "mechanism.tags")

    @property
    def registry_id(self) -> str:
        return self.mechanism_id

    def definition(self) -> dict[str, Any]:
        payload = {
            "mechanism_id": self.mechanism_id,
            "version": self.version,
            "description": self.description,
            "origin_type": self.origin_type,
            "claim_ids": list(self.claim_ids),
            "observation_ids": list(self.observation_ids),
            "causal_chain": list(self.causal_chain),
            "assumptions": list(self.assumptions),
            "competing_mechanism_ids": list(self.competing_mechanism_ids),
            "scope": dict(self.scope),
            "observable_predictions": [dict(item) for item in self.observable_predictions],
            "intervention_predictions": [dict(item) for item in self.intervention_predictions],
        }
        # Keep pre-v1.7 fingerprints stable while binding new internal mechanisms
        # to the exact campaign batches that motivated them.
        if self.campaign_batch_ids:
            payload["campaign_batch_ids"] = list(self.campaign_batch_ids)
        return payload

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "name": self.name,
            "status": self.status,
            "tags": list(self.tags),
            "notes": self.notes,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MechanismRecord":
        record = cls(
            mechanism_id=require_id(payload.get("mechanism_id"), "mechanism"),
            name=require_text(payload.get("name"), "mechanism.name"),
            version=int(payload.get("version", 1)),
            description=require_text(payload.get("description"), "mechanism.description"),
            origin_type=str(payload.get("origin_type", "")),
            claim_ids=strings(payload.get("claim_ids"), "mechanism.claim_ids"),
            observation_ids=strings(payload.get("observation_ids"), "mechanism.observation_ids"),
            causal_chain=strings(payload.get("causal_chain"), "mechanism.causal_chain", allow_empty=False),
            assumptions=strings(payload.get("assumptions"), "mechanism.assumptions"),
            competing_mechanism_ids=strings(payload.get("competing_mechanism_ids"), "mechanism.competing_mechanism_ids"),
            scope=json_mapping(payload.get("scope"), "mechanism.scope"),
            observable_predictions=json_mappings(payload.get("observable_predictions"), "mechanism.observable_predictions"),
            intervention_predictions=json_mappings(payload.get("intervention_predictions"), "mechanism.intervention_predictions"),
            status=str(payload.get("status", "proposed")),
            tags=tags(payload.get("tags"), "mechanism.tags"),
            notes=str(payload.get("notes", "")),
            campaign_batch_ids=strings(
                payload.get("campaign_batch_ids"),
                "mechanism.campaign_batch_ids",
            ),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


@dataclass(frozen=True)
class HypothesisRecord:
    hypothesis_id: str
    title: str
    version: int
    mechanism_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    observable_predictions: tuple[Mapping[str, Any], ...]
    intervention: Mapping[str, Any]
    trigger: Mapping[str, Any]
    outcome: Mapping[str, Any]
    prediction: str
    expected_effect: Mapping[str, Any]
    controls: tuple[Mapping[str, Any], ...]
    falsification: Mapping[str, Any]
    estimated_cost: Mapping[str, Any]
    status: str
    created_at: str
    tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        require_id(self.hypothesis_id, "hypothesis")
        require_text(self.title, "hypothesis.title")
        require_int(self.version, "hypothesis.version", minimum=1)
        if not self.mechanism_ids:
            raise SchemaError("hypothesis.mechanism_ids must be non-empty")
        for mechanism_id in self.mechanism_ids:
            require_id(mechanism_id, "mechanism")
        for context_id in self.context_ids:
            require_id(context_id, "context")

        observable_ids: list[str] = []
        for index, item in enumerate(json_mappings(self.observable_predictions, "hypothesis.observable_predictions")):
            require_keys(
                item,
                f"hypothesis.observable_predictions[{index}]",
                ("observable_id", "version", "expected_pattern", "window"),
            )
            observable_ids.append(require_id(item["observable_id"], "observable"))
            require_int(item["version"], "observable_prediction.version", minimum=1)
            json_mapping(item["expected_pattern"], "observable_prediction.expected_pattern")
            json_mapping(item["window"], "observable_prediction.window")
        if len(set(observable_ids)) != len(observable_ids):
            raise SchemaError("hypothesis observable IDs must be unique")

        intervention = json_mapping(self.intervention, "hypothesis.intervention")
        require_keys(intervention, "hypothesis.intervention", ("intervention_id", "version", "parameters"))
        require_id(intervention["intervention_id"], "intervention")
        require_int(intervention["version"], "hypothesis.intervention.version", minimum=1)
        json_mapping(intervention["parameters"], "hypothesis.intervention.parameters")

        trigger = json_mapping(self.trigger, "hypothesis.trigger")
        require_keys(trigger, "hypothesis.trigger", ("type", "observable_id", "condition"))
        trigger_type = require_text(trigger["type"], "hypothesis.trigger.type")
        trigger_observable_value = str(trigger["observable_id"])
        if trigger_observable_value:
            trigger_observable = require_id(trigger_observable_value, "observable")
            if trigger_observable not in observable_ids:
                raise SchemaError("hypothesis trigger must reference a predicted observable")
        elif trigger_type not in {"run_start", "static_schedule"}:
            raise SchemaError(
                "only run_start/static_schedule hypotheses may omit an observable trigger"
            )
        if not observable_ids and trigger_type not in {"run_start", "static_schedule"}:
            raise SchemaError(
                "observable-conditioned hypotheses require observable predictions"
            )
        json_mapping(trigger["condition"], "hypothesis.trigger.condition")

        outcome = json_mapping(self.outcome, "hypothesis.outcome")
        require_keys(outcome, "hypothesis.outcome", ("outcome_id", "version"))
        require_id(outcome["outcome_id"], "outcome")
        require_int(outcome["version"], "hypothesis.outcome.version", minimum=1)
        require_text(self.prediction, "hypothesis.prediction")

        effect = json_mapping(self.expected_effect, "hypothesis.expected_effect")
        require_keys(effect, "hypothesis.expected_effect", ("direction", "estimand", "latency_steps"))
        require_enum(effect["direction"], "hypothesis.expected_effect.direction", {"increase", "decrease", "nonzero", "none"})
        require_number(effect["latency_steps"], "hypothesis.expected_effect.latency_steps", minimum=0)

        control_ids: list[str] = []
        control_kinds: set[str] = set()
        for index, control in enumerate(json_mappings(self.controls, "hypothesis.controls")):
            require_keys(control, f"hypothesis.controls[{index}]", ("control_id", "kind", "description", "matching"))
            control_ids.append(local_id(control["control_id"], "control.control_id"))
            control_kinds.add(require_text(control["kind"], "control.kind"))
            matching = json_mapping(control["matching"], "control.matching")
            require_keys(matching, "control.matching", ("same_checkpoint", "same_data_order"))
            if matching["same_checkpoint"] is not True:
                raise SchemaError("all controls must use the same checkpoint")
        if len(set(control_ids)) != len(control_ids):
            raise SchemaError("hypothesis control IDs must be unique")
        if "no_intervention" not in control_kinds:
            raise SchemaError("hypothesis requires a no_intervention control")

        falsification = json_mapping(self.falsification, "hypothesis.falsification")
        require_keys(falsification, "hypothesis.falsification", ("minimum_seeds", "decision_rule", "failure_condition"))
        require_int(falsification["minimum_seeds"], "hypothesis.falsification.minimum_seeds", minimum=1)
        require_text(falsification["decision_rule"], "hypothesis.falsification.decision_rule")
        require_text(falsification["failure_condition"], "hypothesis.falsification.failure_condition")

        cost = json_mapping(self.estimated_cost, "hypothesis.estimated_cost")
        require_keys(cost, "hypothesis.estimated_cost", ("gpu_hours", "currency_cost", "cost_tier", "basis"))
        require_number(cost["gpu_hours"], "hypothesis.estimated_cost.gpu_hours", minimum=0)
        require_number(cost["currency_cost"], "hypothesis.estimated_cost.currency_cost", minimum=0)
        require_enum(cost["cost_tier"], "hypothesis.estimated_cost.cost_tier", {"light", "moderate", "heavy", "very_heavy", "offline"})
        require_text(cost["basis"], "hypothesis.estimated_cost.basis")
        require_enum(self.status, "hypothesis.status", HYPOTHESIS_STATUSES)
        require_text(self.created_at, "hypothesis.created_at")
        tags(self.tags, "hypothesis.tags")

    @property
    def registry_id(self) -> str:
        return self.hypothesis_id

    @property
    def observable_ids(self) -> tuple[str, ...]:
        return tuple(str(item["observable_id"]) for item in self.observable_predictions)

    @property
    def intervention_id(self) -> str:
        return str(self.intervention["intervention_id"])

    @property
    def outcome_id(self) -> str:
        return str(self.outcome["outcome_id"])

    def definition(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "version": self.version,
            "mechanism_ids": list(self.mechanism_ids),
            "context_ids": list(self.context_ids),
            "observable_predictions": [dict(item) for item in self.observable_predictions],
            "intervention": dict(self.intervention),
            "trigger": dict(self.trigger),
            "outcome": dict(self.outcome),
            "prediction": self.prediction,
            "expected_effect": dict(self.expected_effect),
            "controls": [dict(item) for item in self.controls],
            "falsification": dict(self.falsification),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "title": self.title,
            "estimated_cost": dict(self.estimated_cost),
            "status": self.status,
            "created_at": self.created_at,
            "tags": list(self.tags),
            "notes": self.notes,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HypothesisRecord":
        record = cls(
            hypothesis_id=require_id(payload.get("hypothesis_id"), "hypothesis"),
            title=require_text(payload.get("title"), "hypothesis.title"),
            version=int(payload.get("version", 1)),
            mechanism_ids=strings(payload.get("mechanism_ids"), "hypothesis.mechanism_ids", allow_empty=False),
            context_ids=strings(payload.get("context_ids"), "hypothesis.context_ids", allow_empty=True),
            observable_predictions=json_mappings(payload.get("observable_predictions"), "hypothesis.observable_predictions"),
            intervention=json_mapping(payload.get("intervention"), "hypothesis.intervention"),
            trigger=json_mapping(payload.get("trigger"), "hypothesis.trigger"),
            outcome=json_mapping(payload.get("outcome"), "hypothesis.outcome"),
            prediction=require_text(payload.get("prediction"), "hypothesis.prediction"),
            expected_effect=json_mapping(payload.get("expected_effect"), "hypothesis.expected_effect"),
            controls=json_mappings(payload.get("controls"), "hypothesis.controls"),
            falsification=json_mapping(payload.get("falsification"), "hypothesis.falsification"),
            estimated_cost=json_mapping(payload.get("estimated_cost"), "hypothesis.estimated_cost"),
            status=str(payload.get("status", "proposed")),
            created_at=require_text(payload.get("created_at"), "hypothesis.created_at"),
            tags=tags(payload.get("tags"), "hypothesis.tags"),
            notes=str(payload.get("notes", "")),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


__all__ = ["HYPOTHESIS_STATUSES", "HypothesisRecord", "MechanismRecord"]
