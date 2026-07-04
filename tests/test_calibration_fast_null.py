"""Tests for the fast search-level rotation null (calibration.fast_null)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgedge import AlphaConfig, AlphaDiscovery, DiscoveryConfig, EventDiscovery
from forgedge.calibration import FastRotationNull, contract_abs_z


def _table(n=4000, seed=7, k=0.02):
    """Mean-reversion KPI table; ``k=0`` decouples the feature from returns."""
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.004, n)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = -k * (feat[:-1] - 0.5) + noise[1:]
    close = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {
            "open_dt": pd.date_range("2023-01-01", periods=n, freq="1h"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.abs(rng.normal(1e6, 1e5, n)),
            "feat": feat,
        }
    )


def _run_pipeline(df):
    ed = EventDiscovery(df.copy(), DiscoveryConfig(timestamp_col="open_dt"))
    cands = ed.run()
    cfg = AlphaConfig(asset="SYN", timeframe="1H")
    ad = AlphaDiscovery(ed.df, cands, cfg)
    contracts = ad.run()
    return ed, cands, cfg, contracts, ad.promoted_contracts()


class TestFastRotationNull:
    @pytest.fixture(scope="class")
    def signal_pipeline(self):
        """Planted edge: the feature genuinely predicts the next-bar return."""
        df = _table(k=0.02)
        return _run_pipeline(df)

    @pytest.fixture(scope="class")
    def signal_case(self, signal_pipeline):
        ed, cands, cfg, contracts, promoted = signal_pipeline
        report = FastRotationNull(ed.df, cands, cfg).run(promoted)
        return contracts, promoted, report

    def test_real_stat_matches_contract_t_stats(self, signal_case):
        # Fidelity: row 0 of the fast null reproduces AlphaDiscovery's own
        # per-horizon z (t_stat_by_h), so the search max must equal the best
        # contract_abs_z over every evaluated contract.
        contracts, _, report = signal_case
        best = max(contract_abs_z(c) for c in contracts)
        assert report.real_stats["abs_z"] == pytest.approx(best, rel=1e-9)

    def test_planted_edge_separates_from_null(self, signal_case):
        _, promoted, report = signal_case
        assert report.tippett_p < 0.05
        assert len(report.survivors) >= 1
        promoted_ids = {c.alpha_id for c in promoted}
        assert {c.alpha_id for c in report.survivors} <= promoted_ids

    def test_annotations(self, signal_case):
        _, promoted, report = signal_case
        bar = report.null_q["abs_z"]
        assert np.isfinite(bar)
        for c in promoted:
            assert c.rotation_p is not None and 0.0 < c.rotation_p <= 1.0
            assert c.rotation_threshold == pytest.approx(bar)
        for c in report.survivors:
            assert contract_abs_z(c) > bar

    def test_deterministic_without_seed(self, signal_pipeline):
        ed, cands, cfg, _, promoted = signal_pipeline
        r1 = FastRotationNull(ed.df, cands, cfg).run(promoted)
        r2 = FastRotationNull(ed.df, cands, cfg).run(promoted)
        assert r1.null_arrays["abs_z"] == r2.null_arrays["abs_z"]
        assert r1.tippett_p == r2.tippett_p
        assert np.isfinite(r1.tippett_p)

    def test_noise_does_not_separate(self):
        # k=0: the feature is independent of the returns, so the real search
        # max should sit inside its own rotation null (deterministic: the null
        # is exact, the only randomness is the fixed data seed).
        df = _table(k=0.0, seed=11)
        ed, cands, cfg, _, promoted = _run_pipeline(df)
        report = FastRotationNull(ed.df, cands, cfg).run(promoted)
        assert report.tippett_p > 0.05
        assert len(report.survivors) == 0

    def test_report_shape(self, signal_case):
        _, _, report = signal_case
        assert report.tippett_best_stat == "abs_z"
        assert report.k == len(report.null_arrays["abs_z"])
        assert report.k > 100
        # summary() renders without error and mentions the yardstick
        assert "abs_z" in report.summary()
