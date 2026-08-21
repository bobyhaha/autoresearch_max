"""Structured state for automated scientific research on training dynamics."""

from .core import SchemaError
from .experiments import ExperimentRecord, RunRecord
from .ideas import HypothesisRecord, MechanismRecord
from .knowledge import BeliefRecord, ClaimRecord, EvidenceRecord, ObservationRecord, PaperRecord
from .reporting import HourlyResearchReport
from .registry import JsonlRegistry, ResearchRegistry
from .toolkit import (
    CapabilityGapRecord,
    ContextRecord,
    InterventionRecord,
    ObservableRecord,
    OutcomeRecord,
    ToolProposalRecord,
)

__all__ = [
    "BeliefRecord",
    "CapabilityGapRecord",
    "ClaimRecord",
    "ContextRecord",
    "EvidenceRecord",
    "ExperimentRecord",
    "HypothesisRecord",
    "HourlyResearchReport",
    "InterventionRecord",
    "JsonlRegistry",
    "MechanismRecord",
    "ObservableRecord",
    "ObservationRecord",
    "OutcomeRecord",
    "PaperRecord",
    "ResearchRegistry",
    "RunRecord",
    "SchemaError",
    "ToolProposalRecord",
]
