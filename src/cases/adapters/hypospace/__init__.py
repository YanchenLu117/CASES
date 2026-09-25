"""HypoSpace adapter package (Agent A).

HypoSpace is a controlled exact solution-space recovery benchmark over three
domains (causal DAGs / Boolean expressions / 3D voxel structures).  It has no
scalar empirical utility field: ``evaluate`` raises ``EvaluationNotApplicable``
and the main experiment uses no RecoveryBackend (``configs/hypospace`` ->
``recovery: none``).  The CASES state is the structural coverage state
``S_t = (V_t, G_sci, G_rev, L_t)`` (experiment doc A2).

Grounding follows the frozen contract ``Compile -> Verify -> Canonicalize``,
executing the *official* HypoSpace repo (pinned via ``external/repos.lock.yaml``,
see ``official.py``) for parsing, validation, and admissible-set enumeration.

Exports
-------
- ``CausalAdapter``   : canonical sorted labeled edge set
- ``BooleanAdapter``  : canonical expression rebuilt from the official
                        mechanistic key (commutativity/idempotence/flatten)
- ``Voxel3DAdapter``  : canonical fixed-order normalized voxel tensor
"""

from __future__ import annotations

from .causal import CausalAdapter
from .boolean import BooleanAdapter
from .voxel3d import Voxel3DAdapter

__all__ = [
    "CausalAdapter",
    "BooleanAdapter",
    "Voxel3DAdapter",
]
