"""Preregistered experiments and immutable run facts."""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    require_id,
    require_int,
    require_keys,
    require_number,
    require_text,
    strings,
    tags,
)
from .setup import validate_scope_key


EXPERIMENT_STAGES = {
    "offline_screen",
    "pilot",
    "matched_branch",
    "discovery",
    "validation",
    "locked_test",
}
EXPERIMENT_STATUSES = {"planned", "approved", "running", "complete", "failed", "cancelled"}
RUN_STATUSES = {"queued", "running", "complete", "failed", "invalid"}


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    title: str
    version: int
    hypothesis_id: str
    hypothesis_fingerprint: str
    stage: str
    status: str
    arms: tuple[Mapping[str, Any], ...]
    seeds: tuple[int, ...]
    checkpoint: Mapping[str, Any]
    randomization: Mapping[str, Any]
    analysis_plan: Mapping[str, Any]
    budget: Mapping[str, Any]
    promotion_gate: Mapping[str, Any]
    data_policy: Mapping[str, Any]
    created_at: str
    search_policy: Mapping[str, Any] | None = None
    idea_id: str = ""
    frozen_at: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        require_id(self.experiment_id, "experiment")
        require_text(self.title, "experiment.title")
        require_int(self.version, "experiment.version", minimum=1)
        require_id(self.hypothesis_id, "hypothesis")
        require_text(self.hypothesis_fingerprint, "experiment.hypothesis_fingerprint")
        require_enum(self.stage, "experiment.stage", EXPERIMENT_STAGES)
        require_enum(self.status, "experiment.status", EXPERIMENT_STATUSES)
        policy_stage_pairs = (
            self.search_policy.get("stage_pairs")
            if isinstance(self.search_policy, Mapping)
            else None
        )
        if not self.seeds:
            raise SchemaError("experiment.seeds must be non-empty")
        for seed in self.seeds:
            require_int(seed, "experiment.seed", minimum=0)
        if len(set(self.seeds)) != len(self.seeds):
            raise SchemaError("experiment seeds must be unique")

        arms = json_mappings(self.arms, "experiment.arms")
        arm_ids: list[str] = []
        roles: set[str] = set()
        for index, arm in enumerate(arms):
            require_keys(arm, f"experiment.arms[{index}]", ("arm_id", "role", "description", "intervention_id", "trigger", "control_id"))
            arm_ids.append(local_id(arm["arm_id"], "experiment.arm_id"))
            roles.add(require_enum(arm["role"], "experiment.arm.role", {"control", "treatment", "sham"}))
            if arm["intervention_id"]:
                require_id(arm["intervention_id"], "intervention")
            if arm["control_id"]:
                local_id(arm["control_id"], "experiment.arm.control_id")
            json_mapping(arm["trigger"], "experiment.arm.trigger")
        if len(set(arm_ids)) != len(arm_ids):
            raise SchemaError("experiment arm IDs must be unique")
        if self.stage == "offline_screen" and arms:
            raise SchemaError("offline_screen evaluates existing data and must have no arms")
        if self.stage != "offline_screen" and not {"control", "treatment"}.issubset(roles):
            raise SchemaError("interventional experiment requires control and treatment arms")
        if self.stage == "pilot" and len(self.seeds) != 1:
            raise SchemaError("pilot must use exactly one seed")
        one_pair_search = self.stage == "discovery" and policy_stage_pairs == 1
        if (
            self.stage in {"matched_branch", "discovery", "validation", "locked_test"}
            and len(self.seeds) < 3
            and not one_pair_search
        ):
            raise SchemaError(f"{self.stage} requires at least three seeds")

        checkpoint = json_mapping(self.checkpoint, "experiment.checkpoint")
        if self.stage != "offline_screen":
            require_keys(checkpoint, "experiment.checkpoint", ("matching", "source"))
            if checkpoint["matching"] != "same_checkpoint_per_seed":
                raise SchemaError("interventional arms must branch from the same checkpoint")
        randomization = json_mapping(self.randomization, "experiment.randomization")
        if self.stage != "offline_screen":
            require_keys(randomization, "experiment.randomization", ("unit", "method"))

        analysis = json_mapping(self.analysis_plan, "experiment.analysis_plan")
        require_keys(
            analysis,
            "experiment.analysis_plan",
            ("primary_estimand", "outcome_id", "baseline_covariates", "uncertainty_method", "multiplicity"),
        )
        require_id(analysis["outcome_id"], "outcome")
        baselines = set(strings(analysis["baseline_covariates"], "experiment.analysis_plan.baseline_covariates"))
        if self.stage == "offline_screen":
            required = {"training_step", "train_loss", "learning_rate"}
            if not required.issubset(baselines):
                raise SchemaError("offline screen must control step, train loss, and learning rate")
        multiplicity = json_mapping(analysis["multiplicity"], "experiment.analysis_plan.multiplicity")
        require_keys(multiplicity, "experiment.analysis_plan.multiplicity", ("family_id", "method"))
        local_id(multiplicity["family_id"], "multiplicity.family_id")
        require_text(multiplicity["method"], "multiplicity.method")

        budget = json_mapping(self.budget, "experiment.budget")
        require_keys(
            budget,
            "experiment.budget",
            ("estimated_gpu_hours", "estimated_currency_cost", "hard_cap_currency_cost", "basis"),
        )
        require_number(budget["estimated_gpu_hours"], "experiment.budget.estimated_gpu_hours", minimum=0)
        estimated = require_number(budget["estimated_currency_cost"], "experiment.budget.estimated_currency_cost", minimum=0)
        hard_cap = require_number(budget["hard_cap_currency_cost"], "experiment.budget.hard_cap_currency_cost", minimum=0)
        if estimated > hard_cap:
            raise SchemaError("experiment estimate exceeds its hard budget cap")
        require_text(budget["basis"], "experiment.budget.basis")

        gate = json_mapping(self.promotion_gate, "experiment.promotion_gate")
        require_keys(gate, "experiment.promotion_gate", ("criteria", "on_pass", "on_fail"))
        strings(gate["criteria"], "experiment.promotion_gate.criteria", allow_empty=False)
        require_text(gate["on_pass"], "experiment.promotion_gate.on_pass")
        require_text(gate["on_fail"], "experiment.promotion_gate.on_fail")

        policy = json_mapping(self.data_policy, "experiment.data_policy")
        require_keys(policy, "experiment.data_policy", ("split", "proposal_loop_access"))
        require_bool(policy["proposal_loop_access"], "experiment.data_policy.proposal_loop_access")
        if self.stage == "locked_test" and policy["proposal_loop_access"] is not False:
            raise SchemaError("locked-test data must be hidden from the proposal loop")
        if "scope_key" in policy:
            validate_scope_key(policy["scope_key"], "experiment.data_policy.scope_key")
        search_policy = json_mapping(self.search_policy, "experiment.search_policy")
        if search_policy:
            # Imported lazily to keep the record schema independent of the policy
            # evaluator's run-analysis helpers.
            from .search_policy import validate_search_policy

            validate_search_policy(search_policy, self)
            require_id(self.idea_id, "idea", "experiment.idea_id")
        elif self.idea_id:
            require_id(self.idea_id, "idea", "experiment.idea_id")
        require_text(self.created_at, "experiment.created_at")
        if self.status in {"approved", "running", "complete"}:
            require_text(self.frozen_at, "experiment.frozen_at")
        tags(self.tags, "experiment.tags")

    @property
    def registry_id(self) -> str:
        return self.experiment_id

    def definition(self) -> dict[str, Any]:
        payload = {
            "experiment_id": self.experiment_id,
            "version": self.version,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_fingerprint": self.hypothesis_fingerprint,
            "stage": self.stage,
            "arms": [dict(item) for item in self.arms],
            "seeds": list(self.seeds),
            "checkpoint": dict(self.checkpoint),
            "randomization": dict(self.randomization),
            "analysis_plan": dict(self.analysis_plan),
            "budget": dict(self.budget),
            "promotion_gate": dict(self.promotion_gate),
            "data_policy": dict(self.data_policy),
        }
        # Historical records predate the portfolio layer. Only fingerprint this
        # field when present so their immutable fingerprints remain valid, while
        # every new policy-aware experiment binds the policy into its identity.
        if self.search_policy:
            payload["search_policy"] = dict(self.search_policy)
            payload["idea_id"] = self.idea_id
        elif self.idea_id:
            payload["idea_id"] = self.idea_id
        return payload

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "frozen_at": self.frozen_at,
            "tags": list(self.tags),
            "notes": self.notes,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentRecord":
        record = cls(
            experiment_id=require_id(payload.get("experiment_id"), "experiment"),
            title=require_text(payload.get("title"), "experiment.title"),
            version=int(payload.get("version", 1)),
            hypothesis_id=require_id(payload.get("hypothesis_id"), "hypothesis"),
            hypothesis_fingerprint=require_text(payload.get("hypothesis_fingerprint"), "experiment.hypothesis_fingerprint"),
            stage=str(payload.get("stage", "")),
            status=str(payload.get("status", "planned")),
            arms=json_mappings(payload.get("arms"), "experiment.arms"),
            seeds=tuple(int(seed) for seed in payload.get("seeds", ())),
            checkpoint=json_mapping(payload.get("checkpoint"), "experiment.checkpoint"),
            randomization=json_mapping(payload.get("randomization"), "experiment.randomization"),
            analysis_plan=json_mapping(payload.get("analysis_plan"), "experiment.analysis_plan"),
            budget=json_mapping(payload.get("budget"), "experiment.budget"),
            promotion_gate=json_mapping(payload.get("promotion_gate"), "experiment.promotion_gate"),
            data_policy=json_mapping(payload.get("data_policy"), "experiment.data_policy"),
            created_at=require_text(payload.get("created_at"), "experiment.created_at"),
            search_policy=json_mapping(
                payload.get("search_policy"), "experiment.search_policy"
            ),
            idea_id=str(payload.get("idea_id", "")),
            frozen_at=str(payload.get("frozen_at", "")),
            tags=tags(payload.get("tags"), "experiment.tags"),
            notes=str(payload.get("notes", "")),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


@dataclass(frozen=True)
class RunRecord:
    """Immutable execution facts. Analysis belongs in evidence, not here."""

    run_id: str
    experiment_id: str
    hypothesis_id: str
    arm_id: str
    seed: int
    role: str
    status: str
    git_commit: str
    git_dirty: bool
    config_hash: str
    spec_fingerprints: Mapping[str, Any]
    tracker: Mapping[str, Any]
    artifact_paths: Mapping[str, Any]
    intervention_events_path: str
    outcome_values: Mapping[str, Any]
    actual_cost: Mapping[str, Any]
    started_at: str
    ended_at: str
    failure_reason: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_id(self.run_id, "run")
        require_id(self.experiment_id, "experiment")
        require_id(self.hypothesis_id, "hypothesis")
        local_id(self.arm_id, "run.arm_id")
        require_int(self.seed, "run.seed", minimum=0)
        require_enum(self.role, "run.role", {"baseline", "treatment", "matched_time", "shuffled_trigger", "sham", "validation", "locked_test"})
        require_enum(self.status, "run.status", RUN_STATUSES)
        require_text(self.git_commit, "run.git_commit")
        require_bool(self.git_dirty, "run.git_dirty")
        require_text(self.config_hash, "run.config_hash")
        fingerprints = json_mapping(self.spec_fingerprints, "run.spec_fingerprints")
        require_keys(fingerprints, "run.spec_fingerprints", ("hypothesis", "experiment", "outcome"))
        tracker = json_mapping(self.tracker, "run.tracker")
        paths = json_mapping(self.artifact_paths, "run.artifact_paths")
        if not tracker and not paths:
            raise SchemaError("run requires a tracker reference or artifact paths")
        require_text(self.intervention_events_path, "run.intervention_events_path")
        outcomes = json_mapping(self.outcome_values, "run.outcome_values")
        for name, value in outcomes.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise SchemaError(f"run.outcome_values.{name} must be finite")
        cost = json_mapping(self.actual_cost, "run.actual_cost")
        require_keys(cost, "run.actual_cost", ("gpu_hours", "currency_cost", "observable_overhead_pct"))
        require_number(cost["gpu_hours"], "run.actual_cost.gpu_hours", minimum=0)
        require_number(cost["currency_cost"], "run.actual_cost.currency_cost", minimum=0)
        require_number(cost["observable_overhead_pct"], "run.actual_cost.observable_overhead_pct", minimum=0)
        require_text(self.started_at, "run.started_at")
        if self.status in {"complete", "failed", "invalid"}:
            require_text(self.ended_at, "run.ended_at")
        if self.status == "failed":
            require_text(self.failure_reason, "run.failure_reason")
        tags(self.tags, "run.tags")

    @property
    def registry_id(self) -> str:
        return self.run_id

    def definition(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "hypothesis_id": self.hypothesis_id,
            "arm_id": self.arm_id,
            "seed": self.seed,
            "role": self.role,
            "status": self.status,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "config_hash": self.config_hash,
            "spec_fingerprints": dict(self.spec_fingerprints),
            "tracker": dict(self.tracker),
            "artifact_paths": dict(self.artifact_paths),
            "intervention_events_path": self.intervention_events_path,
            "outcome_values": dict(self.outcome_values),
            "actual_cost": dict(self.actual_cost),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "failure_reason": self.failure_reason,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.definition())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.definition(),
            "tags": list(self.tags),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunRecord":
        record = cls(
            run_id=require_id(payload.get("run_id"), "run"),
            experiment_id=require_id(payload.get("experiment_id"), "experiment"),
            hypothesis_id=require_id(payload.get("hypothesis_id"), "hypothesis"),
            arm_id=local_id(payload.get("arm_id"), "run.arm_id"),
            seed=int(payload.get("seed", -1)),
            role=str(payload.get("role", "")),
            status=str(payload.get("status", "queued")),
            git_commit=require_text(payload.get("git_commit"), "run.git_commit"),
            git_dirty=payload.get("git_dirty"),
            config_hash=require_text(payload.get("config_hash"), "run.config_hash"),
            spec_fingerprints=json_mapping(payload.get("spec_fingerprints"), "run.spec_fingerprints"),
            tracker=json_mapping(payload.get("tracker"), "run.tracker"),
            artifact_paths=json_mapping(payload.get("artifact_paths"), "run.artifact_paths"),
            intervention_events_path=require_text(payload.get("intervention_events_path"), "run.intervention_events_path"),
            outcome_values=json_mapping(payload.get("outcome_values"), "run.outcome_values"),
            actual_cost=json_mapping(payload.get("actual_cost"), "run.actual_cost"),
            started_at=require_text(payload.get("started_at"), "run.started_at"),
            ended_at=str(payload.get("ended_at", "")),
            failure_reason=str(payload.get("failure_reason", "")),
            tags=tags(payload.get("tags"), "run.tags"),
        )
        check_fingerprint(payload, record.fingerprint)
        return record


__all__ = [
    "EXPERIMENT_STAGES",
    "EXPERIMENT_STATUSES",
    "ExperimentRecord",
    "RunRecord",
]
