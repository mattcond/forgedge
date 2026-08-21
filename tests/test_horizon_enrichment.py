"""Tests for the per-event horizon-grid enrichment (structural prior)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from forgedge import AlphaConfig, AlphaDiscovery, DiscoveryConfig, EventDiscovery
from forgedge.alpha_discovery.discovery import enriched_horizons


def _stub(event_id: str, w: int):
    return SimpleNamespace(event_id=event_id, dominant_window=lambda: w)


class TestEnrichedHorizonsUnit:
    CFG = AlphaConfig(horizon_grid=(1, 2, 3, 5, 7, 10))  # daily-style base

    def test_union_with_cap(self):
        # w=24 → candidates {12, 24, 48}; cap = max(10, 600//20) = 30 → {12, 24}.
        all_h, extras = enriched_horizons([_stub("a", 24)], self.CFG, split=600)
        assert extras["a"] == [12, 24]
        assert all_h == [1, 2, 3, 5, 7, 10, 12, 24]

    def test_base_grid_never_restricted(self):
        all_h, extras = enriched_horizons([_stub("a", 4)], self.CFG, split=600)
        # w=4 → {2, 4, 8}; 2 already in base → adds only 4 and 8.
        assert extras["a"] == [4, 8]
        assert set(all_h) >= set(self.CFG.horizon_grid)

    def test_slow_window_fully_capped(self):
        # w=500 → {250, 500, 1000} all above the cap → no enrichment.
        all_h, extras = enriched_horizons([_stub("a", 500)], self.CFG, split=600)
        assert "a" not in extras
        assert all_h == sorted(self.CFG.horizon_grid)

    def test_no_window_no_enrichment(self):
        all_h, extras = enriched_horizons([_stub("a", 0)], self.CFG, split=600)
        assert extras == {}

    def test_disabled(self):
        cfg = AlphaConfig(horizon_grid=(1, 2, 3, 5, 7, 10), horizon_enrichment=None)
        all_h, extras = enriched_horizons([_stub("a", 24)], cfg, split=600)
        assert extras == {}
        assert all_h == sorted(cfg.horizon_grid)


def _pipeline(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = -0.02 * (feat[:-1] - 0.5) + rng.normal(0.0, 0.004, n - 1)
    close = 100.0 * np.exp(np.cumsum(r))
    df = pd.DataFrame(
        {
            "open_dt": pd.date_range("2023-01-01", periods=n, freq="1h"),
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close,
            "feat_ema_24": feat,  # window embedded in the name
        }
    )
    ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
    return ed, ed.run()


class TestEnrichmentIntegration:
    @pytest.fixture(scope="class")
    def pipeline(self):
        return _pipeline()

    @staticmethod
    def _from_feat_ema_24(source_feature: str) -> bool:
        """True for candidates that trace back to the fixture's feat_ema_24
        column (embeds window=24), as opposed to a cross-column, cross-time
        OHLC candidate (issue #161) — those are built purely from raw
        open/high/low/close and legitimately embed no such window."""
        return "ema" in source_feature

    def test_dominant_window_reads_name_and_transform(self, pipeline):
        _, cands = pipeline
        checked_ema = 0
        for c in cands:
            w = c.dominant_window()
            if any(self._from_feat_ema_24(comp.source_feature) for comp in c.components):
                assert w >= 24  # every feat_ema_24-derived candidate embeds it
                checked_ema += 1
            for comp in c.components:
                tw = (comp.transform_params or {}).get("window")
                if tw:
                    assert w >= int(tw)
        assert checked_ema > 0, "no feat_ema_24-derived candidate to check"

    def test_contracts_scan_enriched_grids(self, pipeline):
        ed, cands = pipeline
        cfg = AlphaConfig(asset="SYN", timeframe="1H")
        ad = AlphaDiscovery(ed.df, cands, cfg)
        contracts = ad.run()
        # The grid that *ran*: `horizon_grid` is session-resolved (#196) and
        # `resolve_config` returns a copy, so the caller's `cfg` still holds
        # the sentinel — deliberately, since inspecting a config must not
        # mutate it.
        base = set(ad.config.horizon_grid)
        enriched = [
            c for c in contracts
            if set(c.derived_target.t_stat_by_h) > base
        ]
        assert enriched, "no contract scanned an enriched grid"
        checked_ema = 0
        for c in contracts:
            grid = set(c.derived_target.t_stat_by_h)
            assert grid >= base                      # union, never restriction
            if self._from_feat_ema_24(c.event_expression):
                assert c.dominant_window >= 24        # metadata recorded
                checked_ema += 1
        assert checked_ema > 0, "no feat_ema_24-derived contract to check"
        assert ad.n_return_tests > len(contracts) * len(base)

    def test_off_switch_restores_base_grid(self, pipeline):
        ed, cands = pipeline
        cfg = AlphaConfig(asset="SYN", timeframe="1H", horizon_enrichment=None)
        ad = AlphaDiscovery(ed.df, cands, cfg)
        contracts = ad.run()
        base = ad.config.horizon_grid          # resolved, see above
        for c in contracts:
            assert set(c.derived_target.t_stat_by_h) == set(base)
        assert ad.n_return_tests == len(contracts) * len(base)
