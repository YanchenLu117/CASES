"""Core error types."""

from __future__ import annotations


class CASESError(Exception):
    """Base CASES error."""


class GroundingError(CASESError):
    """Compile (P_tau) failed to produce a typed scientific specification."""


class VerificationRejected(CASESError):
    """Verification (V_tau) rejected a raw object."""


class CanonicalizationError(CASESError):
    """Canonicalization (K_tau) failed."""


class EvaluationNotApplicable(CASESError):
    """The benchmark has no scalar empirical evaluation (e.g. HypoSpace)."""


class RecoveryNotFitted(CASESError):
    """A posterior was requested before the recovery backend was fitted."""


class ContractViolation(CASESError):
    """A shared CASES contract invariant was violated."""
