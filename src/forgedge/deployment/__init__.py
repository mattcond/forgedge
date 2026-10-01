"""forgedge.deployment — putting forge()-discovered rules into production.

Sibling of ``forgedge.playground`` (the read-only analysis toolkit) — this
module has real effects: ``promotion_gate()`` decides which rules are fit to
go live, ``export_rules()`` writes them to disk, ``monitoring_manifest()``
indexes what was exported for a periodic re-check job, and
``minimal_kpi_recipe()``/``verify_kpi_recipe()`` describe — as verified,
human-readable JSON — the few KPI columns each exported rule actually needs
(#296). See issue #245 for the design discussion and #237/#245 for the
sibling analysis toolkit.

Usage::

    from forgedge.deployment import *
"""

from .kpi_recipe import KpiRecipe, KpiRecipeVerification, minimal_kpi_recipe, verify_kpi_recipe
from .rules import PromotionGateConfig, export_rules, monitoring_manifest, promotion_gate

__all__ = [
    "PromotionGateConfig",
    "promotion_gate",
    "export_rules",
    "monitoring_manifest",
    "KpiRecipe",
    "KpiRecipeVerification",
    "minimal_kpi_recipe",
    "verify_kpi_recipe",
]
