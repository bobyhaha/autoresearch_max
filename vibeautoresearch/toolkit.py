"""Versioned tools for observing, contextualizing, evaluating, and intervening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .core import (
    SchemaError,
    check_fingerprint,
    fingerprint,
    json_mapping,
    json_mappings,
    require_bool,
    require_enum,
    require_id,
    require_int,
    require_keys,
    require_number,
    require_text,
    strings,
    tags,
)


COST_TIERS = {"light", "moderate", "heavy", "very_heavy", "offline"}
TOOL_STATUSES = {
    "proposed",
    "implemented",
    "unit_tested",
    "cost_profiled",
    "available",
    "deprecated",
}


def _version(value: Any, field: str) -> int:
    return require_int(value, field, minimum=1)


def _collection(value: Any, field: str) -> dict[str, Any]:
    result = json_mapping(value, field)
    require_keys(
        result,
        field,
        ("mode", "cadence_steps", "cost_tier", "estimated_overhead_pct", "profile_status"),
    )
    require_enum(result["mode"], f"{field}.mode", {"online", "validation_only", "offline"})
    require_int(result["cadence_steps"], f"{field}.cadence_steps", minimum=1)
    require_enum(result["cost_tier"], f"{field}.cost_tier", COST_TIERS)
    require_number(result["estimated_overhead_pct"], f"{field}.estimated_overhead_pct", minimum=0)
    require_enum(result["profile_status"], f"{field}.profile_status", {"unprofiled", "estimated", "measured"})
    return result


@dataclass(frozen=True)
class ObservableRecord:
    observable_id: str
    name: str
    version: int
    sources: tuple[Mapping[str, Any], ...]
    operation: str
    reductions: tuple[Mapping[str, Any], ...]
    output: Mapping[str, Any]
    collection: Mapping[str, Any]
    causal_availability: Mapping[str, Any]
    implementation: Mapping[str, Any]
    status: str
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.observable_id, "observable")
        require_text(self.name, "observable.name")
        _version(self.version, "observable.version")
        if not self.sources:
            raise SchemaError("observable.sources must be non-empty")
        all_axes: list[str] = []
        for index, source in enumerate(json_mappings(self.sources, "observable.sources")):
            require_keys(source, f"observable.sources[{index}]", ("name", "target", "tensor_type", "axes", "context"))
            all_axes.extend(strings(source["axes"], f"observable.sources[{index}].axes"))
        require_enum(self.operation, "observable.operation", {"axis_reduction_pipeline", "composition", "direct_scalar"})
        reductions = json_mappings(self.reductions, "observable.reductions")
        if self.operation == "axis_reduction_pipeline":
            reduced_axes: list[str] = []
            for index, reduction in enumerate(reductions):
                require_keys(reduction, f"observable.reductions[{index}]", ("axis", "op", "params"))
                reduced_axes.append(require_text(reduction["axis"], "reduction.axis"))
                require_text(reduction["op"], "reduction.op")
                json_mapping(reduction["params"], "reduction.params")
            if sorted(reduced_axes) != sorted(all_axes):
                raise SchemaError(
                    "axis_reduction_pipeline must reduce every declared source axis exactly once"
                )
        output = json_mapping(self.output, "observable.output")
        require_keys(output, "observable.output", ("type", "units"))
        require_enum(output["type"], "observable.output.type", {"scalar", "per_layer_scalar", "per_head_scalar", "small_vector", "histogram"})
        _collection(self.collection, "observable.collection")
        availability = json_mapping(self.causal_availability, "observable.causal_availability")
        require_keys(availability, "observable.causal_availability", ("available_before_action", "uses_validation", "stage"))
        require_bool(availability["available_before_action"], "causal_availability.available_before_action")
        require_bool(availability["uses_validation"], "causal_availability.uses_validation")
        implementation = json_mapping(self.implementation, "observable.implementation")
        require_keys(implementation, "observable.implementation", ("entrypoint", "test"))
        require_text(implementation["entrypoint"], "observable.implementation.entrypoint")
        require_text(implementation["test"], "observable.implementation.test")
        require_enum(self.status, "observable.status", TOOL_STATUSES)
        if self.status in {"implemented", "unit_tested", "cost_profiled", "available"}:
            if str(implementation["entrypoint"]).startswith("not_implemented"):
                raise SchemaError("implemented observable requires a real entrypoint")
        tags(self.tags, "observable.tags")

    @property
    def registry_id(self) -> str:
        return self.observable_id

    def definition(self) -> dict[str, Any]:
        return {
            "observable_id": self.observable_id,
            "version": self.version,
            "sources": [dict(item) for item in self.sources],
            "operation": self.operation,
            "reductions": [dict(item) for item in self.reductions],
            "output": dict(self.output),
            "collection": dict(self.collection),
            "causal_availability": dict(self.causal_availability),
            "implementation": dict(self.implementation),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "name": self.name,
            "status": self.status,
            "description": self.description,
            "tags": list(self.tags),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservableRecord":
        record = cls(
            observable_id=require_id(payload.get("observable_id"), "observable"),
            name=require_text(payload.get("name"), "observable.name"),
            version=int(payload.get("version", 1)),
            sources=json_mappings(payload.get("sources"), "observable.sources"),
            operation=str(payload.get("operation", "")),
            reductions=json_mappings(payload.get("reductions"), "observable.reductions"),
            output=json_mapping(payload.get("output"), "observable.output"),
            collection=json_mapping(payload.get("collection"), "observable.collection"),
            causal_availability=json_mapping(payload.get("causal_availability"), "observable.causal_availability"),
            implementation=json_mapping(payload.get("implementation"), "observable.implementation"),
            status=str(payload.get("status", "proposed")),
            description=str(payload.get("description", "")),
            tags=tags(payload.get("tags"), "observable.tags"),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


@dataclass(frozen=True)
class InterventionRecord:
    intervention_id: str
    name: str
    version: int
    action: str
    target: Mapping[str, Any]
    parameters: Mapping[str, Any]
    timing: Mapping[str, Any]
    duration: Mapping[str, Any]
    reversible: bool
    cost_tier: str
    safety: Mapping[str, Any]
    implementation: Mapping[str, Any]
    status: str
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.intervention_id, "intervention")
        require_text(self.name, "intervention.name")
        _version(self.version, "intervention.version")
        require_text(self.action, "intervention.action")
        target = json_mapping(self.target, "intervention.target")
        require_keys(target, "intervention.target", ("type", "selector"))
        json_mapping(self.parameters, "intervention.parameters")
        timing = json_mapping(self.timing, "intervention.timing")
        require_keys(timing, "intervention.timing", ("stage",))
        duration = json_mapping(self.duration, "intervention.duration")
        require_keys(duration, "intervention.duration", ("type",))
        require_bool(self.reversible, "intervention.reversible")
        require_enum(self.cost_tier, "intervention.cost_tier", COST_TIERS)
        json_mapping(self.safety, "intervention.safety")
        implementation = json_mapping(self.implementation, "intervention.implementation")
        require_keys(implementation, "intervention.implementation", ("entrypoint", "test"))
        require_text(implementation["entrypoint"], "intervention.implementation.entrypoint")
        require_text(implementation["test"], "intervention.implementation.test")
        require_enum(self.status, "intervention.status", TOOL_STATUSES)
        if self.status in {"implemented", "unit_tested", "cost_profiled", "available"}:
            if str(implementation["entrypoint"]).startswith("not_implemented"):
                raise SchemaError("implemented intervention requires a real entrypoint")
        tags(self.tags, "intervention.tags")

    @property
    def registry_id(self) -> str:
        return self.intervention_id

    def definition(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "version": self.version,
            "action": self.action,
            "target": dict(self.target),
            "parameters": dict(self.parameters),
            "timing": dict(self.timing),
            "duration": dict(self.duration),
            "reversible": self.reversible,
            "cost_tier": self.cost_tier,
            "safety": dict(self.safety),
            "implementation": dict(self.implementation),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "name": self.name,
            "status": self.status,
            "description": self.description,
            "tags": list(self.tags),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionRecord":
        record = cls(
            intervention_id=require_id(payload.get("intervention_id"), "intervention"),
            name=require_text(payload.get("name"), "intervention.name"),
            version=int(payload.get("version", 1)),
            action=require_text(payload.get("action"), "intervention.action"),
            target=json_mapping(payload.get("target"), "intervention.target"),
            parameters=json_mapping(payload.get("parameters"), "intervention.parameters"),
            timing=json_mapping(payload.get("timing"), "intervention.timing"),
            duration=json_mapping(payload.get("duration"), "intervention.duration"),
            reversible=payload.get("reversible"),
            cost_tier=str(payload.get("cost_tier", "")),
            safety=json_mapping(payload.get("safety"), "intervention.safety"),
            implementation=json_mapping(payload.get("implementation"), "intervention.implementation"),
            status=str(payload.get("status", "proposed")),
            description=str(payload.get("description", "")),
            tags=tags(payload.get("tags"), "intervention.tags"),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


@dataclass(frozen=True)
class ContextRecord:
    context_id: str
    name: str
    version: int
    source: Mapping[str, Any]
    value: Mapping[str, Any]
    availability: Mapping[str, Any]
    implementation: Mapping[str, Any]
    status: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.context_id, "context")
        require_text(self.name, "context.name")
        _version(self.version, "context.version")
        source = json_mapping(self.source, "context.source")
        require_keys(source, "context.source", ("type",))
        value = json_mapping(self.value, "context.value")
        require_keys(value, "context.value", ("type",))
        availability = json_mapping(self.availability, "context.availability")
        require_keys(availability, "context.availability", ("available_before_action",))
        require_bool(availability["available_before_action"], "context.availability.available_before_action")
        implementation = json_mapping(self.implementation, "context.implementation")
        require_keys(implementation, "context.implementation", ("entrypoint", "test"))
        require_text(implementation["entrypoint"], "context.implementation.entrypoint")
        require_text(implementation["test"], "context.implementation.test")
        require_enum(self.status, "context.status", TOOL_STATUSES)
        if self.status in {"implemented", "unit_tested", "cost_profiled", "available"}:
            if str(implementation["entrypoint"]).startswith("not_implemented"):
                raise SchemaError("implemented context requires a real entrypoint")
        tags(self.tags, "context.tags")

    @property
    def registry_id(self) -> str:
        return self.context_id

    def definition(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "version": self.version,
            "source": dict(self.source),
            "value": dict(self.value),
            "availability": dict(self.availability),
            "implementation": dict(self.implementation),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {**self.definition(), "name": self.name, "status": self.status, "tags": list(self.tags), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextRecord":
        record = cls(
            context_id=require_id(payload.get("context_id"), "context"),
            name=require_text(payload.get("name"), "context.name"),
            version=int(payload.get("version", 1)),
            source=json_mapping(payload.get("source"), "context.source"),
            value=json_mapping(payload.get("value"), "context.value"),
            availability=json_mapping(payload.get("availability"), "context.availability"),
            implementation=json_mapping(payload.get("implementation"), "context.implementation"),
            status=str(payload.get("status", "proposed")),
            tags=tags(payload.get("tags"), "context.tags"),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


@dataclass(frozen=True)
class OutcomeRecord:
    outcome_id: str
    name: str
    version: int
    metric: Mapping[str, Any]
    direction: str
    data_split: str
    aggregation: Mapping[str, Any]
    detector: Mapping[str, Any]
    policy_visible: bool
    implementation: Mapping[str, Any]
    status: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.outcome_id, "outcome")
        require_text(self.name, "outcome.name")
        _version(self.version, "outcome.version")
        metric = json_mapping(self.metric, "outcome.metric")
        require_keys(metric, "outcome.metric", ("name", "units"))
        require_enum(self.direction, "outcome.direction", {"minimize", "maximize"})
        require_enum(self.data_split, "outcome.data_split", {"train", "validation", "test"})
        aggregation = json_mapping(self.aggregation, "outcome.aggregation")
        require_keys(aggregation, "outcome.aggregation", ("type",))
        json_mapping(self.detector, "outcome.detector")
        require_bool(self.policy_visible, "outcome.policy_visible")
        if self.data_split in {"validation", "test"} and self.policy_visible:
            raise SchemaError("validation/test outcomes cannot control online training")
        implementation = json_mapping(self.implementation, "outcome.implementation")
        require_keys(implementation, "outcome.implementation", ("entrypoint", "test"))
        require_text(implementation["entrypoint"], "outcome.implementation.entrypoint")
        require_text(implementation["test"], "outcome.implementation.test")
        require_enum(self.status, "outcome.status", TOOL_STATUSES)
        if self.status in {"implemented", "unit_tested", "cost_profiled", "available"}:
            if str(implementation["entrypoint"]).startswith("not_implemented"):
                raise SchemaError("implemented outcome requires a real entrypoint")
        tags(self.tags, "outcome.tags")

    @property
    def registry_id(self) -> str:
        return self.outcome_id

    def definition(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "version": self.version,
            "metric": dict(self.metric),
            "direction": self.direction,
            "data_split": self.data_split,
            "aggregation": dict(self.aggregation),
            "detector": dict(self.detector),
            "policy_visible": self.policy_visible,
            "implementation": dict(self.implementation),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {**self.definition(), "name": self.name, "status": self.status, "tags": list(self.tags), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OutcomeRecord":
        record = cls(
            outcome_id=require_id(payload.get("outcome_id"), "outcome"),
            name=require_text(payload.get("name"), "outcome.name"),
            version=int(payload.get("version", 1)),
            metric=json_mapping(payload.get("metric"), "outcome.metric"),
            direction=str(payload.get("direction", "")),
            data_split=str(payload.get("data_split", "")),
            aggregation=json_mapping(payload.get("aggregation"), "outcome.aggregation"),
            detector=json_mapping(payload.get("detector"), "outcome.detector"),
            policy_visible=payload.get("policy_visible"),
            implementation=json_mapping(payload.get("implementation"), "outcome.implementation"),
            status=str(payload.get("status", "proposed")),
            tags=tags(payload.get("tags"), "outcome.tags"),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


@dataclass(frozen=True)
class ToolProposalRecord:
    proposal_id: str
    tool_type: str
    name: str
    motivation: str
    required_by_hypotheses: tuple[str, ...]
    estimated_implementation_cost: str
    estimated_runtime_cost: str
    suggested_collection_frequency_steps: int
    acceptance_tests: tuple[str, ...]
    status: str
    created_at: str

    def __post_init__(self) -> None:
        require_id(self.proposal_id, "tool_proposal")
        require_enum(self.tool_type, "tool_proposal.tool_type", {"observable", "intervention", "context", "outcome"})
        require_text(self.name, "tool_proposal.name")
        require_text(self.motivation, "tool_proposal.motivation")
        for hypothesis_id in self.required_by_hypotheses:
            require_id(hypothesis_id, "hypothesis")
        require_enum(self.estimated_implementation_cost, "tool_proposal.estimated_implementation_cost", COST_TIERS)
        require_enum(self.estimated_runtime_cost, "tool_proposal.estimated_runtime_cost", COST_TIERS)
        require_int(self.suggested_collection_frequency_steps, "tool_proposal.suggested_collection_frequency_steps", minimum=1)
        strings(self.acceptance_tests, "tool_proposal.acceptance_tests", allow_empty=False)
        require_enum(self.status, "tool_proposal.status", {"proposed", "approved", "implemented", "rejected"})
        require_text(self.created_at, "tool_proposal.created_at")

    @property
    def registry_id(self) -> str:
        return self.proposal_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "tool_type": self.tool_type,
            "name": self.name,
            "motivation": self.motivation,
            "required_by_hypotheses": list(self.required_by_hypotheses),
            "estimated_implementation_cost": self.estimated_implementation_cost,
            "estimated_runtime_cost": self.estimated_runtime_cost,
            "suggested_collection_frequency_steps": self.suggested_collection_frequency_steps,
            "acceptance_tests": list(self.acceptance_tests),
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolProposalRecord":
        return cls(
            proposal_id=require_id(payload.get("proposal_id"), "tool_proposal"),
            tool_type=str(payload.get("tool_type", "")),
            name=require_text(payload.get("name"), "tool_proposal.name"),
            motivation=require_text(payload.get("motivation"), "tool_proposal.motivation"),
            required_by_hypotheses=strings(payload.get("required_by_hypotheses"), "tool_proposal.required_by_hypotheses"),
            estimated_implementation_cost=str(payload.get("estimated_implementation_cost", "")),
            estimated_runtime_cost=str(payload.get("estimated_runtime_cost", "")),
            suggested_collection_frequency_steps=int(payload.get("suggested_collection_frequency_steps", 0)),
            acceptance_tests=strings(payload.get("acceptance_tests"), "tool_proposal.acceptance_tests", allow_empty=False),
            status=str(payload.get("status", "proposed")),
            created_at=require_text(payload.get("created_at"), "tool_proposal.created_at"),
        )


@dataclass(frozen=True)
class CapabilityGapRecord:
    gap_id: str
    question: str
    missing_capabilities: tuple[str, ...]
    blocked_hypothesis_ids: tuple[str, ...]
    priority: str
    recommended_solution: str
    status: str
    created_at: str

    def __post_init__(self) -> None:
        require_id(self.gap_id, "capability_gap")
        require_text(self.question, "capability_gap.question")
        strings(self.missing_capabilities, "capability_gap.missing_capabilities", allow_empty=False)
        for hypothesis_id in self.blocked_hypothesis_ids:
            require_id(hypothesis_id, "hypothesis")
        require_enum(self.priority, "capability_gap.priority", {"low", "medium", "high"})
        require_text(self.recommended_solution, "capability_gap.recommended_solution")
        require_enum(self.status, "capability_gap.status", {"open", "in_progress", "resolved", "wont_fix"})
        require_text(self.created_at, "capability_gap.created_at")

    @property
    def registry_id(self) -> str:
        return self.gap_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "question": self.question,
            "missing_capabilities": list(self.missing_capabilities),
            "blocked_hypothesis_ids": list(self.blocked_hypothesis_ids),
            "priority": self.priority,
            "recommended_solution": self.recommended_solution,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityGapRecord":
        return cls(
            gap_id=require_id(payload.get("gap_id"), "capability_gap"),
            question=require_text(payload.get("question"), "capability_gap.question"),
            missing_capabilities=strings(payload.get("missing_capabilities"), "capability_gap.missing_capabilities", allow_empty=False),
            blocked_hypothesis_ids=strings(payload.get("blocked_hypothesis_ids"), "capability_gap.blocked_hypothesis_ids"),
            priority=str(payload.get("priority", "medium")),
            recommended_solution=require_text(payload.get("recommended_solution"), "capability_gap.recommended_solution"),
            status=str(payload.get("status", "open")),
            created_at=require_text(payload.get("created_at"), "capability_gap.created_at"),
        )


__all__ = [
    "COST_TIERS",
    "TOOL_STATUSES",
    "CapabilityGapRecord",
    "ContextRecord",
    "InterventionRecord",
    "ObservableRecord",
    "OutcomeRecord",
    "ToolProposalRecord",
]
