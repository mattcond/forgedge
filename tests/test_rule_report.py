"""Tests for the rule performance report (forgedge.rule_report)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgedge import (
    CustomEvent,
    DiscoveryConfig,
    RuleSpec,
    forge,
    rule_performance_report,
)
from forgedge.event_discovery.models import GateParams
from forgedge.rule_discovery.models import BacktestParams, GridSpec, WalkForwardConfig
from forgedge import RuleDiscoveryConfig


def _candles(n=700, seed=3, with_regime=False):
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = -0.03 * (feat[:-1] - 0.5) + rng.normal(0.0, 0.004, n - 1)
    close = 100.0 * np.exp(np.cumsum(r))
    df = pd.DataFrame(
        {
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": close, "high": close * 1.02, "low": close * 0.98,
            "close": close, "feat": feat,
        }
    )
    if with_regime:
        df["regime"] = np.where(close > np.r_[close[0], close[:-1]], "BULL", "BEAR")
    return df


def _spec(df, name="RULE_T1", formula="feat < 0.15", direction="long", **pk):
    cand = CustomEvent(name=name, formula=formula).to_event_candidate(df)
    params = BacktestParams(
        buy_type="limit", direction=direction, buy_drop_pct=0.005,
        sell_pct=0.03, target_h=5, fee=0.001, **pk,
    )
    return RuleSpec(name=name, candidate=cand, params=params)


class TestRulePerformanceReport:
    @pytest.fixture(scope="class")
    def candles(self):
        return _candles(with_regime=True)

    @pytest.fixture(scope="class")
    def html(self, candles):
        specs = [
            _spec(candles, "RULE_T1", "feat < 0.15"),
            _spec(candles, "RULE_T2", "feat > 0.9", direction="short"),
        ]
        specs[0].is_end = pd.Timestamp(candles["open_dt"].iloc[400])
        specs[0].oos_expectancy = 0.0123
        specs[0].verdict = "PARTIAL-EDGE"
        return rule_performance_report(specs, candles, title="Test report")

    def test_report_is_selfcontained_html(self, html):
        assert html.startswith("<!DOCTYPE html>")
        assert "<svg" in html and "http://" not in html and "https://" not in html

    def test_required_sections_present(self, html):
        for section in (
            "Equity vs Buy &amp; Hold",
            "Monthly activation trend",
            "Gain / loss distribution",
            "MAE → final net",
            "Rolling expectancy",
            "Recent trades",
            "Per-regime performance",
        ):
            assert section in html, section

    def test_rules_and_metadata_rendered(self, html):
        assert "RULE_T1" in html and "RULE_T2" in html
        assert "PARTIAL-EDGE" in html
        assert "IS end" in html and "OOS →" in html  # split markers (rule 1)

    def test_no_regime_column_skips_section_gracefully(self):
        candles = _candles(n=400)
        spec = _spec(candles)
        out = rule_performance_report([spec], candles, compute_regime=False)
        assert "Per-regime performance" not in out
        assert "RULE_T1" in out

    def test_missing_feature_degrades_per_rule(self, candles):
        good = _spec(candles, "RULE_OK", "feat < 0.15")
        bad_frame = candles.assign(other=1.0)
        bad = _spec(bad_frame, "RULE_BAD", "other < 2")
        out = rule_performance_report([good, bad], candles.drop(columns=[]))
        # the bad rule references a column absent from the report candles
        out = rule_performance_report(
            [good, bad], candles, compute_regime=False
        )
        assert "RULE_OK" in out
        assert "not evaluable" in out and "RULE_BAD" in out

    def test_active_now_badge(self):
        candles = _candles(n=300)
        candles.loc[candles.index[-1], "feat"] = 0.01  # force signal on last bar
        spec = _spec(candles, "RULE_LIVE", "feat < 0.15")
        out = rule_performance_report([spec], candles, compute_regime=False)
        assert "SIGNAL ACTIVE" in out

    def test_datetimeindex_candles_accepted(self):
        candles = _candles(n=300).set_index("open_dt")
        spec = _spec(candles.reset_index(), "RULE_IDX")
        out = rule_performance_report([spec], candles, compute_regime=False)
        assert "RULE_IDX" in out

    def test_no_signals_rule_renders(self):
        candles = _candles(n=300)
        spec = _spec(candles, "RULE_NEVER", "feat < -1")  # never fires
        out = rule_performance_report([spec], candles, compute_regime=False)
        assert "RULE_NEVER" in out and "no signals" in out


class TestReturnDistributions:
    def test_section_and_legend_present(self):
        candles = _candles(with_regime=True)
        spec = _spec(candles, "RULE_T1", "feat < 0.15")
        out = rule_performance_report([spec], candles, compute_regime=False)
        assert "Return distributions" in out
        for cls in ("distbase", "distlow", "distclose", "disthigh"):
            assert cls in out
        assert "n event" in out
        assert "Δ mean" in out

    def test_no_active_bars_degrades_gracefully(self):
        candles = _candles(n=300)
        spec = _spec(candles, "RULE_NEVER", "feat < -1")  # never fires
        out = rule_performance_report([spec], candles, compute_regime=False)
        assert "Return distributions" in out
        assert "RULE_NEVER" in out

    def test_injected_close_shift_is_recovered(self):
        # Deterministic check of the underlying computation (base vs event
        # bar return), independent of HTML rendering: force a known positive
        # close return specifically on event bars and verify the module's own
        # base/event split recovers a mean shift close to the injected size.
        from forgedge.rule_report import _prepare_candles, _signal_series

        n = 1200
        rng = np.random.default_rng(11)
        feat = rng.uniform(0.0, 1.0, n)
        base_ret = rng.normal(0.0, 0.01, n)
        event = feat < 0.08
        shifted_ret = base_ret.copy()
        shifted_ret[event] += 0.05  # +5% extra close return at event bars
        close = 100.0 * np.cumprod(1.0 + shifted_ret)
        df = pd.DataFrame({
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": np.r_[close[0], close[:-1]],
            "high": close * 1.005, "low": close * 0.995, "close": close, "feat": feat,
        })
        spec = _spec(df, "RULE_SHIFT", "feat < 0.08")

        out = rule_performance_report([spec], df, compute_regime=False)
        assert "Return distributions" in out

        frame = _prepare_candles(df, "open_dt")
        sig_active = _signal_series(spec, frame).to_numpy() > 0
        ret_close = frame["close"].pct_change().to_numpy(float) * 100.0
        valid = np.isfinite(ret_close)
        mean_base = ret_close[valid].mean()
        mean_event = ret_close[valid & sig_active].mean()
        assert mean_event - mean_base > 3.0  # ~5% injected, allow for noise

    def test_stats_row_matches_computation(self):
        # The n base / n event counts rendered must match the mask sizes.
        from forgedge.rule_report import _prepare_candles, _signal_series

        candles = _candles(n=500, seed=9)
        spec = _spec(candles, "RULE_T9", "feat < 0.2")
        frame = _prepare_candles(candles, "open_dt")
        active = _signal_series(spec, frame).to_numpy() > 0
        n_base = int(np.isfinite(frame["close"].pct_change().to_numpy(float)).sum())
        n_event = int((active & np.isfinite(frame["close"].pct_change().to_numpy(float))).sum())

        out = rule_performance_report([spec], candles, compute_regime=False)
        assert f"<td>{n_base}</td><td>{n_event}</td>" in out


class TestFromForgeResult:
    def test_specs_and_report_from_pipeline(self):
        kpi = _candles(n=900, seed=7)
        result = forge(
            kpi,
            ticker="SYN",
            timeframe="1D",
            event_discovery_config=DiscoveryConfig(
                max_and_components=1,
                gate_params=GateParams(min_tpm=2.0, max_dispersion=2.5),
            ),
            rule_discovery_config=RuleDiscoveryConfig(
                grid=GridSpec(buy_drop_pct=[0.0], buy_delay_bar=[0]),
                walk_forward=WalkForwardConfig(n_splits=2, reoptimise=False),
            ),
            progress=False,
        )
        specs = RuleSpec.from_forge_result(result)
        assert len(specs) == len(result.edges())
        if not specs:
            pytest.skip("no tradeable rule on the synthetic fixture")
        assert all(s.is_end is not None for s in specs)
        assert all(s.oos_expectancy is not None for s in specs)
        out = rule_performance_report(result, kpi)
        for s in specs:
            assert s.name in out
        assert "IS end" in out
