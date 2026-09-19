"""forgedge.experiment — pipelines of experiments built with ``forge()``.

Sibling of ``forgedge.playground`` (read-only analysis over ``ForgeResult``)
and ``forgedge.deployment`` (putting discovered rules into production) —
this module RUNS its own multi-stage pipelines on top of ``forge()``,
neither read-only nor production-facing. ``StepWiseDiscovery`` is the first
one: a partition & compose search that grows hold-out-confirmed
single-condition rules one AND-condition at a time by searching each rule's
own active sub-population for a second dimension — see its class docstring
for the full algorithm.

Usage::

    from forgedge.experiment import StepWiseDiscovery, StepWiseDiscoveryConfig

    engine = StepWiseDiscovery(kpi, ticker="BTCUSDC", timeframe="1D")
    result = engine.run()
    print(result.summary())
"""

from .models import ChainResult, SeedAttempt, StepWiseDiscoveryConfig, StepWiseDiscoveryResult
from .step_wise_discovery import StepWiseDiscovery

__all__ = [
    "StepWiseDiscovery",
    "StepWiseDiscoveryConfig",
    "StepWiseDiscoveryResult",
    "ChainResult",
    "SeedAttempt",
]
