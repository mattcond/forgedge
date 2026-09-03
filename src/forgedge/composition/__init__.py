"""Grade-guided event composition (issue #254).

Sits between Module 1 (Event Discovery) and Module 2 (Alpha Discovery) in
the two-pass composition design: M1 produces a returns-blind 1D event pool,
a first Alpha Discovery pass grades each event A-D, this module composes
pairs/triples using the grade as the pairing criterion (instead of M1's
purely structural tpm/dispersion/transform_key pairing), and a second Alpha
Discovery pass evaluates the composed candidates from scratch. See
``docs/analysis/issue_254_two_pass_composition_plan.md`` for the full design
and phased rollout — this module is Phase 2's deliverable; the orchestration
that actually wires it into ``forge()`` is Phase 3.

Usage::

    from forgedge.composition import GradePairingConfig, grade_guided_compose
"""
from .grade_pairing import GradePairingConfig, grade_guided_compose

__all__ = [
    "GradePairingConfig",
    "grade_guided_compose",
]
