"""CASES core — state, grounding, relations, recovery, evidence, readout.

Governance-approved addition: the formal SpaceReadout
component (``readout``) for solution-space frontier readouts.  State,
grounding, recovery, and evidence semantics are unchanged.

Adds the three-operation facade ``CASESModel`` and the first-class
evidence log.  No existing API is removed.

Core (additive): adds the logic-governed neuro-symbolic representation layer
— ``RepresentationSpecification`` (P_t), ``LogicGate`` (L_tau) + ``FidelityGate``
(F_t) forming the dual admission gate, ``RepresentationRuntime`` (C_t),
``RepresentationParadigm`` (the minus-logic ablation arms), the
bidirectional ``LanguageCorrespondence`` (Gamma_t), and the evidence-driven
``RepresentationRevisioner`` (four revision triggers).  No existing API removed.
"""

from .model import (
    EvidenceLog,
    EvidenceRecord,
    LanguageCorrespondence,
    CASESModel,
)
from .agent import CASESAgentLoop, default_evaluator, default_implementer
from .reground import (
    DEFAULT_MISSING_DATA_POLICY,
    MISSING_DATA_POLICY,
    UNRECOVERABLE,
    ReGroundingPass,
    ReGroundingReport,
    UnknownAttribute,
    apply_missing_data_policy,
)
from .acquisition import (
    COV,
    BND,
    SOL,
    OPEN,
    CHANNELS,
    ExplorationController,
    ExplorationGear,
    GearProfile,
)
from .readout import (
    CoverageFrontierReadout,
    CoveredRegion,
    FrontierSemantics,
    FrontierTarget,
    ReadoutPacket,
    SpaceReadout,
    descriptor_distance,
    descriptor_tuple,
    render_readout,
)
from .representation import (
    FidelityGate,
    FidelityCertificate,
    LogicGate,
    LogicReport,
    LogicViolation,
    RelationDecl,
    RepresentationInducer,
    RepresentationParadigm,
    RepresentationRuntime,
    RepresentationSpecification,
    RuntimeRegistration,
    TransformDecl,
    TypedVariable,
    default_representation_spec,
)
from .revision import (
    RepresentationRevisioner,
    RevisionReport,
    RevisionSignal,
    RevisionTrigger,
)
from .state import SolutionSpaceState
from .types import (
    GroundedObject,
    LanguageHypothesis,
    Observation,
)
from ..recovery.base import RecoveryInput

__all__ = [
    # facade
    "CASESModel",
    "EvidenceLog",
    "EvidenceRecord",
    "LanguageCorrespondence",
    "RecoveryInput",
    # readout
    "SpaceReadout",
    "CoverageFrontierReadout",
    "CoveredRegion",
    "FrontierTarget",
    "ReadoutPacket",
    "FrontierSemantics",
    "descriptor_distance",
    "descriptor_tuple",
    "render_readout",
    # Core representation layer
    "RepresentationParadigm",
    "RepresentationSpecification",
    "TypedVariable",
    "RelationDecl",
    "TransformDecl",
    "LogicGate",
    "LogicReport",
    "LogicViolation",
    "FidelityGate",
    "FidelityCertificate",
    "RepresentationRuntime",
    "RuntimeRegistration",
    "RepresentationInducer",
    "default_representation_spec",
    "RepresentationRevisioner",
    "RevisionTrigger",
    "RevisionSignal",
    "RevisionReport",
    # exploration dial
    "ExplorationGear",
    "GearProfile",
    "ExplorationController",
    "SOL",
    "BND",
    "COV",
    "OPEN",
    "CHANNELS",
    # Algorithm-1 loop + re-grounding
    "CASESAgentLoop",
    "default_implementer",
    "default_evaluator",
    "ReGroundingPass",
    "ReGroundingReport",
    "UnknownAttribute",
    "MISSING_DATA_POLICY",
    "DEFAULT_MISSING_DATA_POLICY",
    "UNRECOVERABLE",
    "apply_missing_data_policy",
    # state / types
    "SolutionSpaceState",
    "GroundedObject",
    "LanguageHypothesis",
    "Observation",
]
