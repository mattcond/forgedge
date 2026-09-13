"""Tests for idea B (momentum/mean-reversion/idiosyncratic labeling, with its
OR-strengthened promotion) and idea C (horizon-grid sufficiency), unified
across both promotion routes.

Formulas and empirical validation are frozen in
``docs/analysis/ranged_tpm_and_market_alignment_proposal.md`` §3-§4 — this
file turns that spec into regression tests. Design recap:

- ``DerivedTarget`` gains ``nature``, ``horizon_at_boundary``,
  ``promotion_route``, ``p_auc``, ``rho`` — computed inside
  ``AlphaDiscovery._derive_target`` (where every intermediate quantity
  already lives) and mirrored onto ``AlphaContract`` for direct access.
- The OR-strengthening is a one-clause change to the existing
  ``undetermined`` gate: a candidate is now also promoted when the stage (a)
  AUC test is significant, even with no individually BH-significant horizon.
- ``promotion_route`` reflects which significance test(s) actually found an
  edge, independent of ``require_significant_direction`` (which only
  controls whether the *lack* of one gates ``direction``).
- On the ``"auc"`` route, ``h*`` comes from the elbow rule, not
  ``argmax|z_h|`` — and that rule's own boundary flag *is*
  ``horizon_at_boundary`` for that route, not a separate computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgedge import AlphaDiscovery, EventDiscovery, MarketContext, PromotionThresholds
from forgedge.alpha_discovery.target import forward_log_returns, forward_returns
from forgedge.presets import forge_preset

PARQUET = "tests/fixtures/ADA_1D_TRAIN.parquet"


def _log_sufficient_stats(close: pd.Series, horizons):
    L = forward_log_returns(close, horizons).to_numpy()
    F = forward_returns(close, horizons).to_numpy()
    valid = np.isfinite(L) & np.isfinite(F)
    L0 = np.where(valid, L, 0.0)
    cnt_t = valid.sum(axis=0).astype(float)
    sum_t = L0.sum(axis=0)
    return valid, L0, cnt_t, sum_t


def _solo_auc_table(seed: int = 14, n: int = 2500):
    """A candidate whose excess is significant *integrated* over the grid
    (stage a) but at no single horizon (h_sig=()) -- the "solo AUC" pattern
    the OR-strengthening exists for. Found by search over seeds; pinned here
    so the test is deterministic.
    """
    horizons = [1, 2, 3, 5, 7, 10]
    rng = np.random.default_rng(seed)
    active = rng.random(n) < 0.025
    r = rng.normal(0.0, 0.012, n)
    active_idx = np.flatnonzero(active)
    for h in range(1, 11):
        idx = active_idx + h
        idx = idx[idx < n]
        r[idx] += 0.0006
    close = pd.Series(100.0 * np.exp(np.cumsum(r)))
    return close, active, horizons


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------

class TestTrapzWeights:
    def test_uniform_grid_matches_spacing(self):
        w = AlphaDiscovery._trapz_weights(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(w, [0.5, 1.0, 1.0, 0.5])

    def test_single_point_gets_unit_weight(self):
        w = AlphaDiscovery._trapz_weights(np.array([5.0]))
        np.testing.assert_allclose(w, [1.0])

    def test_late_enriched_point_gets_disproportionate_weight(self):
        """Documented limitation (§3.7, case 996): a horizon far from its
        neighbours (h=24 after h=12 on a base grid ending at 10) gets a
        weight an order of magnitude larger than an early grid point's."""
        w = AlphaDiscovery._trapz_weights(np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 24.0]))
        assert w[-1] == pytest.approx(6.0)  # (24-12)/2, vs w[0]=(2-1)/2=0.5
        assert w[-1] > 10 * w[0]


class TestAucSignificance:
    def test_constant_per_bar_rate_is_not_dominated_by_longest_horizon(self):
        """The bug found twice in validation (§3.4): using raw Delta_h
        instead of Delta_h/h makes any persistent-but-flat edge look
        concentrated at the last horizon. A constant per-bar rate must give
        a p_auc computation driven by the (uniform) rate, not by whichever
        horizon happens to be longest."""
        h = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        rate = 0.002
        delta = rate * h  # exactly constant per-bar rate
        null = np.zeros((200, len(h)))  # a null centred on zero excess
        auc, p_auc = AlphaDiscovery._auc_significance(h, delta, null)
        # Trapezoidal integral of a constant equals rate * grid width exactly
        # -- not dominated by whichever horizon happens to be longest, the
        # bug that using raw Delta_h (instead of Delta_h/h) caused twice.
        assert auc == pytest.approx(rate * (h[-1] - h[0]))
        assert p_auc < 0.01  # far from a null centred on zero

    def test_no_usable_null_gives_nan_p(self):
        h = np.array([1.0, 2.0, 3.0])
        delta = np.array([0.01, 0.015, 0.02])
        null = np.full((5, 3), np.nan)
        auc, p_auc = AlphaDiscovery._auc_significance(h, delta, null)
        assert np.isfinite(auc)
        assert np.isnan(p_auc)


class TestHStarElbow:
    def test_stops_at_first_non_positive_marginal(self):
        h = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        # cumulative excess that grows, then flattens after h=3
        delta = np.array([0.01, 0.018, 0.024, 0.024, 0.023, 0.022])
        h_star, at_boundary = AlphaDiscovery._h_star_elbow(h, delta, sign=1.0)
        assert h_star == 3
        assert at_boundary is False

    def test_monotonically_growing_profile_is_boundary_monotone(self):
        h = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        delta = np.array([0.002, 0.006, 0.012, 0.025, 0.04, 0.06])  # never saturates
        h_star, at_boundary = AlphaDiscovery._h_star_elbow(h, delta, sign=1.0)
        assert h_star == 10
        assert at_boundary is True

    def test_single_horizon_grid_is_boundary_by_definition(self):
        h_star, at_boundary = AlphaDiscovery._h_star_elbow(
            np.array([5.0]), np.array([0.01]), sign=1.0,
        )
        assert h_star == 5
        assert at_boundary is True

    def test_sign_flips_which_direction_counts_as_growth(self):
        h = np.array([1.0, 2.0, 3.0, 5.0])
        delta = -np.array([0.01, 0.018, 0.022, 0.022])  # short-side edge
        h_star, at_boundary = AlphaDiscovery._h_star_elbow(h, delta, sign=-1.0)
        assert h_star == 3
        assert at_boundary is False


class TestGridBoundaryState:
    def _score_by_h(self, scores: dict) -> dict:
        return {h: s for h, s in scores.items()}

    def test_interior_peak_is_interno(self):
        state = AlphaDiscovery._grid_boundary_state(
            {1: 1.0, 2: 2.0, 3: 3.0, 5: 1.5, 7: 1.0}, h_star=3,
        )
        assert state == "interno"

    def test_two_consecutive_increases_at_boundary_is_climbing(self):
        state = AlphaDiscovery._grid_boundary_state(
            {1: 1.0, 2: 1.5, 3: 2.0, 5: 2.5, 7: 3.0}, h_star=7,
        )
        assert state == "bordo_in_salita"

    def test_single_increase_at_boundary_is_ambiguous(self):
        state = AlphaDiscovery._grid_boundary_state(
            {1: 2.0, 2: 1.5, 3: 1.8, 5: 1.7, 7: 1.9}, h_star=7,
        )
        assert state == "bordo_ambiguo"

    def test_non_increasing_at_boundary_is_plateau(self):
        state = AlphaDiscovery._grid_boundary_state(
            {1: 1.0, 2: 3.0, 3: 3.0, 5: 2.9, 7: 2.9}, h_star=7,
        )
        assert state == "bordo_plateau"

    def test_too_few_valid_horizons_near_boundary(self):
        state = AlphaDiscovery._grid_boundary_state({1: 1.0, 2: 2.0}, h_star=2)
        assert state == "bordo_grid_troppo_corta"

    def test_nan_scores_excluded_from_boundary_check(self):
        """A horizon dropped for too few active bars (score=nan) must not be
        treated as the grid's edge: h*=2 is genuinely interior once h=5 (nan)
        is excluded from the valid set, even though it is nominally the last
        configured horizon."""
        state = AlphaDiscovery._grid_boundary_state(
            {1: 1.0, 2: 3.0, 3: 2.0, 5: float("nan")}, h_star=2,
        )
        assert state == "interno"


# ---------------------------------------------------------------------------
# OR-strengthening integration (via _derive_target directly)
# ---------------------------------------------------------------------------

class TestOrStrengthenedPromotion:
    def test_solo_auc_candidate_is_promoted_via_auc_route(self):
        close, active, horizons = _solo_auc_table()
        valid, L0, cnt_t, sum_t = _log_sufficient_stats(close, horizons)
        dt = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, horizons, close.to_numpy(), 0.5, 0.005,
            auc_max_p=0.10,
        )
        assert dt.h_sig == ()  # no single horizon individually BH-significant
        assert dt.statistically_weak is True
        assert dt.direction in ("long", "short")  # promoted despite h_sig=()
        assert dt.promotion_route == "auc"
        assert np.isfinite(dt.p_auc) and dt.p_auc < 0.10
        assert np.isfinite(dt.sell_pct)

    def test_stricter_auc_threshold_reverts_to_undetermined(self):
        """Without the OR-strengthening (or with a threshold this candidate's
        p_auc can't clear), the candidate stays undetermined exactly as
        before this feature existed."""
        close, active, horizons = _solo_auc_table()
        valid, L0, cnt_t, sum_t = _log_sufficient_stats(close, horizons)
        dt = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, horizons, close.to_numpy(), 0.5, 0.005,
            auc_max_p=0.001,
        )
        assert dt.direction == "undetermined"
        assert dt.promotion_route is None

    def test_require_significant_direction_false_still_reports_true_route(self):
        """promotion_route reflects which test(s) actually found an edge,
        independent of require_significant_direction -- which only controls
        whether the *lack* of one gates direction."""
        close, active, horizons = _solo_auc_table()
        valid, L0, cnt_t, sum_t = _log_sufficient_stats(close, horizons)
        dt = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, horizons, close.to_numpy(), 0.5, 0.005,
            require_significant=False, auc_max_p=0.10,
        )
        assert dt.direction in ("long", "short")
        assert dt.promotion_route == "auc"  # same route as the gated version

    def test_neither_test_significant_gives_none_route_under_legacy_mode(self):
        """With require_significant_direction=False and no test clearing,
        direction is still assigned (legacy non-blocking behaviour) but
        promotion_route is honestly None -- it never claims a route that
        didn't actually apply."""
        close, active, horizons = _solo_auc_table()
        valid, L0, cnt_t, sum_t = _log_sufficient_stats(close, horizons)
        dt = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, horizons, close.to_numpy(), 0.5, 0.005,
            require_significant=False, auc_max_p=0.0,
        )
        assert dt.direction in ("long", "short")
        assert dt.promotion_route is None

    def test_direction_sign_matches_between_gated_and_ungated(self):
        """The doc's robustness argument (§3.9): the sign is stable across
        the whole profile for a "solo AUC" candidate, so direction should
        not flip depending on which route (or none) assigned it."""
        close, active, horizons = _solo_auc_table()
        valid, L0, cnt_t, sum_t = _log_sufficient_stats(close, horizons)
        gated = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, horizons, close.to_numpy(), 0.5, 0.005,
            auc_max_p=0.10,
        )
        ungated = AlphaDiscovery._derive_target(
            active, valid, L0, cnt_t, sum_t, horizons, close.to_numpy(), 0.5, 0.005,
            require_significant=False,
        )
        assert gated.direction == ungated.direction


# ---------------------------------------------------------------------------
# PromotionThresholds defaults
# ---------------------------------------------------------------------------

class TestPromotionThresholdsDefaults:
    def test_new_fields_default_to_documented_illustrative_values(self):
        th = PromotionThresholds()
        assert th.auc_max_p == 0.10
        assert th.rho_momentum_threshold == 0.5


# ---------------------------------------------------------------------------
# Full-pipeline regression pin against the frozen validation (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFrozenValidationOnAdaFixture:
    """Pins the exact cross-tab and label counts reported in
    docs/analysis/ranged_tpm_and_market_alignment_proposal.md §3.4 for ADA,
    preset "balanced", horizon_grid=(1,2,3,5,7,10). A change to any of the
    formulas above that shifts these counts is a behaviour change worth
    noticing explicitly, not a silent drift.
    """

    @pytest.fixture(scope="class")
    def contracts(self):
        from dataclasses import replace

        kpi = pd.read_parquet(PARQUET)
        enriched = MarketContext(kpi).run()
        disc_cfg, alpha_cfg, _rd = forge_preset("balanced", "1D", asset="ADA")
        alpha_cfg = replace(alpha_cfg, horizon_grid=(1, 2, 3, 5, 7, 10))
        ed = EventDiscovery(enriched, config=disc_cfg)
        candidates = ed.run()
        return AlphaDiscovery(ed.df, candidates, alpha_cfg).run()

    def test_promotion_route_crosstab(self, contracts):
        counts = pd.Series([c.promotion_route for c in contracts], dtype=object).value_counts()
        assert counts.get("both") == 111
        assert counts.get("z_score") == 69
        assert counts.get("auc") == 45

    def test_nature_label_distribution(self, contracts):
        counts = pd.Series([c.nature for c in contracts]).value_counts()
        assert counts.get("mean-reversion-aligned") == 107
        assert counts.get("idiosyncratic") == 44
        assert counts.get("momentum-aligned") == 5

    def test_h_star_median_and_boundary_rate_per_route(self, contracts):
        df = pd.DataFrame([
            dict(route=c.promotion_route, boundary=c.horizon_at_boundary,
                 hstar=c.derived_target.holding_period_h)
            for c in contracts if c.promotion_route is not None
        ])
        med = df.groupby("route")["hstar"].median()
        boundary_rate = df.groupby("route")["boundary"].mean()

        assert med["z_score"] == 2.0
        assert med["both"] == 3.0
        assert med["auc"] == 6.0

        assert boundary_rate["z_score"] == pytest.approx(0.101, abs=1e-3)
        assert boundary_rate["both"] == pytest.approx(0.189, abs=1e-3)
        assert boundary_rate["auc"] == pytest.approx(0.289, abs=1e-3)

    def test_alpha_contract_mirrors_derived_target(self, contracts):
        """The three fields must agree between AlphaContract and its own
        derived_target -- they are copied, not independently computed."""
        for c in contracts:
            assert c.nature == c.derived_target.nature
            assert c.horizon_at_boundary == c.derived_target.horizon_at_boundary
            assert c.promotion_route == c.derived_target.promotion_route
