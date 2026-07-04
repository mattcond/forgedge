"""Hypothesis ledger — the multiple-testing surface of a FORGE session.

FORGE controls multiple testing *inside* each module (per-event rotation null
and BH-FDR in Alpha Discovery, deflated Sharpe over the operational grid in
Rule Discovery), but no single place used to record how many hypotheses the
whole chain consumed.  The ledger is that record: pure bookkeeping, filled by
``forge()`` from the actual counts of the run and carried on
``ForgeResult.ledger`` so the search surface behind every published verdict is
auditable.

Why the surface is *recorded* here but *corrected* by the rotation null
------------------------------------------------------------------------
The thousands of event-level hypotheses are heavily correlated (shared
features, overlapping activations), so plugging their raw count into the
Deflated-Sharpe haircut of Rule Discovery would overstate the selection bias —
with the stock formula the radicand goes negative around
``n_trials > n_obs^1.73`` and every DSR becomes undefined.  The
correlation-aware correction for the search surface is the search-level
rotation null (:class:`forgedge.calibration.FastRotationNull`, run by default
inside ``forge()``): it measures the null distribution of the best statistic
over the *actual* candidate set, so the correlation structure is priced in
empirically instead of assumed away.  The ledger's job is to make the surface
visible; the null's job is to charge for it.

``RuleDiscoveryConfig.n_trials_upstream`` remains available for callers who
want the analytic haircut to include an explicit upstream factor.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HypothesisLedger:
    """Counts of the hypotheses consumed along one ``forge()`` run.

    Attributes
    ----------
    m1_candidates : int
        Event expressions that survived the Consistency Gate and were handed
        to Alpha Discovery.  The gate itself is outcome-independent (it never
        sees the forward return), so expressions it *rejected* do not consume
        return-hypothesis budget and are not counted here.
    m2_horizons : int
        Horizons scanned per candidate by the target derivation.
    m2_promoted : int
        Contracts promoted by Alpha Discovery (one verdict each downstream).
    m3_grid_cells : int
        Operational-grid combinations screened per contract by Rule Discovery
        (0 until the first rule is backtested; the grid is the same shape for
        every contract of a run).
    """

    m1_candidates: int = 0
    m2_horizons: int = 0
    m2_promoted: int = 0
    m3_grid_cells: int = 0
    # Exact count of (candidate, horizon) tests when the per-event horizon
    # enrichment makes the grid non-uniform; 0 = fall back to the product.
    m2_return_tests: int = 0

    @property
    def m2_surface(self) -> int:
        """Return-hypotheses tested by Alpha Discovery.

        The exact per-event count when recorded (horizon enrichment makes
        grids non-uniform), else ``candidates × base horizons``.
        """
        if self.m2_return_tests:
            return self.m2_return_tests
        return self.m1_candidates * self.m2_horizons

    @property
    def total_surface(self) -> int:
        """Upper bound of the session's search depth under independence.

        ``candidates × horizons × grid cells`` — an *upper* bound because the
        trials are correlated; see the module docstring for why the verdict
        correction uses the empirical rotation null instead of this number.
        """
        return self.m2_surface * max(self.m3_grid_cells, 1)

    def describe(self) -> str:
        """One-line human-readable summary of the surface."""
        grid = (
            f"{self.m2_horizons}(+enriched) horizons"
            if self.m2_return_tests
            and self.m2_return_tests != self.m1_candidates * self.m2_horizons
            else f"{self.m2_horizons} horizons"
        )
        return (
            f"hypothesis surface: {self.m1_candidates} candidates × "
            f"{grid} = {self.m2_surface} return-tests; "
            f"{self.m2_promoted} promoted; ~{max(self.m3_grid_cells, 0)} grid "
            f"cells/rule (total ≲ {self.total_surface})"
        )
