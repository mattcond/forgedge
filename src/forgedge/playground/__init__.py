"""forgedge.playground — analysis abstractions over ``ForgeResult`` (R).

Read-only helpers that manipulate the *results* of one or more ``forge()``
runs — never re-run the pipeline. Each function returns a long-format
``pandas.DataFrame`` (one row per elementary observation) so callers can
compose their own ``groupby``/aggregation instead of the function deciding
it for them. See ``docs/analysis/`` issue #237 for the running checklist of
use cases this module implements.

Usage::

    from forgedge.playground import *
"""

from .funnel import conversion_funnel
from .m0 import regime_time_share, regime_transitions
from .m1 import dead_event_candidates, gate_survival_observed
from .m2 import discard_reasons_by_grade, undetermined_direction_by_family
from .m3 import diagnostics_vs_verdict, lottery_only_winners
from .m4 import classification_by_grade, duplicate_clusters
from .production import PromotionGateConfig, export_rules, monitoring_manifest, promotion_gate

__all__ = [
    "regime_transitions",
    "regime_time_share",
    "dead_event_candidates",
    "gate_survival_observed",
    "discard_reasons_by_grade",
    "undetermined_direction_by_family",
    "diagnostics_vs_verdict",
    "lottery_only_winners",
    "classification_by_grade",
    "duplicate_clusters",
    "conversion_funnel",
    "PromotionGateConfig",
    "promotion_gate",
    "export_rules",
    "monitoring_manifest",
]
