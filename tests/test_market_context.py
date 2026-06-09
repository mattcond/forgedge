"""Tests for the Market Context module (FORGE Modulo 0)."""
import numpy as np
import pandas as pd
import pytest

from forgedge import MarketContext, MarketContextConfig
from forgedge.market_context import (
    EMAProxyClassifier,
    EMAProxyConfig,
    RegimeClassifier,
    build_classifier,
)
from forgedge.market_context.context import _rolling_stability
from forgedge.market_context.hurst import (
    derive_ema_windows,
    hurst_dfa,
    ou_halflife,
    rolling_halflife,
    suggest_ema_windows,
    variance_ratio_profile,
)


def _ou_prices(n=3000, theta=0.05, sigma=0.01, seed=0):
    """Synthetic mean-reverting (OU) price series for window-derivation tests."""
    rng = np.random.default_rng(seed)
    mu = np.log(100.0)
    x = [mu]
    for _ in range(n - 1):
        x.append(x[-1] + theta * (mu - x[-1]) + sigma * rng.standard_normal())
    return np.exp(np.array(x))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_kpi_table(n: int = 4380, seed: int = 42, with_ema: bool = False) -> pd.DataFrame:
    """Synthetic KPI table.  Default n=4380 ≈ 6 months of 1H data.

    By default only ``close`` (and OHLCV) is present — the EMA indicators are
    *not* included, reproducing the contract that the user only needs to supply
    ``close`` for the regime to be computed.
    """
    rng = np.random.default_rng(seed)
    price = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {
            "open_dt": dates,
            "open": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "close": price,
            "volume": np.abs(rng.normal(1e6, 2e5, n)),
        }
    )
    if with_ema:
        df["close_ema_09"] = df["close"].ewm(span=9, adjust=False).mean()
        df["close_ema_25"] = df["close"].ewm(span=25, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# EMAProxyClassifier
# ---------------------------------------------------------------------------

class TestEMAProxyClassifier:
    def test_classify_returns_ordered_categorical(self):
        df = _make_kpi_table()
        clf = build_classifier(MarketContextConfig())
        regime = clf.classify(df)
        assert isinstance(regime.dtype, pd.CategoricalDtype)
        assert regime.cat.ordered
        assert list(regime.cat.categories) == [
            "STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL",
        ]
        assert len(regime) == len(df)

    def test_only_close_required(self):
        """No EMA columns present → computed inline, regime still produced."""
        df = _make_kpi_table(with_ema=False)
        assert "close_ema_09" not in df.columns
        regime = build_classifier(MarketContextConfig()).classify(df)
        assert regime.notna().any()

    def test_missing_source_col_raises(self):
        df = _make_kpi_table().drop(columns=["close"])
        with pytest.raises(KeyError):
            build_classifier(MarketContextConfig()).classify(df)

    def test_precomputed_ema_used_when_present(self):
        df = _make_kpi_table(with_ema=True)
        clf = build_classifier(MarketContextConfig())
        # Pollute the precomputed columns: if they are used, the ratio (and
        # hence the regime) must change versus computing inline.
        df_poisoned = df.copy()
        df_poisoned["close_ema_09"] = df_poisoned["close_ema_09"] * 1.10
        r_clean = clf.classify(df)
        r_poison = clf.classify(df_poisoned)
        assert not r_clean.equals(r_poison)

    def test_strong_bull_when_fast_far_above_slow(self):
        # Monotonic uptrend → fast EMA above slow EMA → bullish regimes
        n = 300
        price = np.linspace(100, 200, n)
        df = pd.DataFrame({"close": price})
        regime = build_classifier(MarketContextConfig()).classify(df)
        # The tail of a strong uptrend should be (STRONG_)BULL
        assert str(regime.iloc[-1]) in {"BULL", "STRONG_BULL"}

    def test_strong_bear_when_fast_far_below_slow(self):
        n = 300
        price = np.linspace(200, 100, n)
        df = pd.DataFrame({"close": price})
        regime = build_classifier(MarketContextConfig()).classify(df)
        assert str(regime.iloc[-1]) in {"BEAR", "STRONG_BEAR"}

    def test_bad_threshold_count_raises(self):
        cfg = EMAProxyConfig(thresholds=[0.99, 1.01])  # only 2 for 5 labels
        with pytest.raises(ValueError):
            EMAProxyClassifier(cfg, list("ABCDE"))

    def test_non_ascending_thresholds_raise(self):
        cfg = EMAProxyConfig(thresholds=[0.99, 0.98, 1.01, 1.02])
        with pytest.raises(ValueError):
            EMAProxyClassifier(cfg, list("ABCDE"))

    def test_get_config_and_labels(self):
        clf = build_classifier(MarketContextConfig())
        cfg = clf.get_config()
        assert cfg["classifier"] == "ema_proxy"
        assert cfg["short_period"] == 9
        assert cfg["long_period"] == 25
        assert clf.get_labels()[0] == "STRONG_BEAR"

    def test_implements_interface(self):
        assert isinstance(build_classifier(MarketContextConfig()), RegimeClassifier)


# ---------------------------------------------------------------------------
# MarketContext orchestrator
# ---------------------------------------------------------------------------

class TestMarketContext:
    def test_run_adds_two_columns(self):
        df = _make_kpi_table()
        out = MarketContext(df).run()
        assert "regime" in out.columns
        assert "regime_stable" in out.columns
        assert out["regime_stable"].dtype == bool

    def test_input_not_mutated(self):
        df = _make_kpi_table()
        cols_before = list(df.columns)
        MarketContext(df).run()
        assert list(df.columns) == cols_before

    def test_no_intermediate_ema_columns_leak(self):
        """Inline EMAs used for the ratio must not be written to the table."""
        df = _make_kpi_table(with_ema=False)
        out = MarketContext(df).run()
        assert "close_ema_09" not in out.columns
        assert "close_ema_25" not in out.columns
        # Only regime + regime_stable are new
        assert set(out.columns) - set(df.columns) == {"regime", "regime_stable"}

    def test_index_preserved(self):
        df = _make_kpi_table().set_index("open_dt")
        out = MarketContext(df).run()
        assert out.index.equals(df.index)
        assert out["regime"].notna().any()

    def test_distribution_sums_to_one(self):
        df = _make_kpi_table(n=8760)
        mc = MarketContext(df)
        mc.run()
        dist = mc.distribution()
        assert set(dist.columns) == {"n_bars", "share"}
        assert int(dist["n_bars"].sum()) == len(df)
        assert abs(dist["share"].sum() - 1.0) < 1e-3  # share is rounded to 4dp
        assert list(dist.index) == mc.classifier.get_labels()

    def test_distribution_before_run_raises(self):
        with pytest.raises(RuntimeError):
            MarketContext(_make_kpi_table()).distribution()

    def test_get_config_traceability(self):
        cfg = MarketContext(_make_kpi_table()).get_config()
        assert cfg["classifier"]["classifier"] == "ema_proxy"
        assert cfg["stable_window"] == 12

    def test_unknown_classifier_raises(self):
        with pytest.raises(ValueError):
            MarketContext(_make_kpi_table(), MarketContextConfig(classifier="hmm"))

    def test_custom_classifier_injection(self):
        class AllBull(RegimeClassifier):
            def classify(self, kpi_table):
                return pd.Series(["BULL"] * len(kpi_table), index=kpi_table.index)

            def get_labels(self):
                return ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]

            def get_config(self):
                return {"classifier": "all_bull"}

        df = _make_kpi_table()
        out = MarketContext(df, classifier=AllBull()).run()
        assert (out["regime"] == "BULL").all()


# ---------------------------------------------------------------------------
# Automatic EMA window selection (Hurst/OU) with fallback
# ---------------------------------------------------------------------------

class TestAutoWindow:
    def _ou_kpi(self, n=3000, theta=0.05):
        prices = _ou_prices(n, theta=theta)
        return pd.DataFrame(
            {"open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
             "close": prices}
        )

    def test_windows_derived_from_data(self):
        mc = MarketContext(self._ou_kpi())
        mc.run()
        res = mc.window_resolution
        assert res["source"] == "hurst_ou"
        assert res["half_life_bars"] > 0
        # The classifier in use must reflect the derived spans.
        cfg = mc.classifier.get_config()
        assert cfg["short_period"] == res["short_period"]
        assert cfg["long_period"] == res["long_period"]
        assert res["short_period"] < res["long_period"]

    def test_derived_windows_differ_from_default_when_warranted(self):
        # theta=0.2 → half-life ≈ -ln2/ln(0.8) ≈ 3.1 bars → spans ≠ 9/25
        mc = MarketContext(self._ou_kpi(theta=0.2))
        mc.run()
        assert mc.window_resolution["source"] == "hurst_ou"
        assert mc.window_resolution["long_period"] != 25

    def test_fallback_when_not_enough_data(self):
        # Only ~6 estimation windows possible < min_window_estimates=10 → fallback
        df = self._ou_kpi(n=300)
        mc = MarketContext(df)
        mc.run()
        res = mc.window_resolution
        assert res["source"] == "fallback"
        assert (res["short_period"], res["long_period"]) == (9, 25)

    def test_auto_window_disabled_uses_configured(self):
        cfg = MarketContextConfig(
            ema_proxy=EMAProxyConfig(auto_window=False, short_period=5, long_period=20)
        )
        mc = MarketContext(self._ou_kpi(), cfg)
        mc.run()
        res = mc.window_resolution
        assert res["source"] == "configured"
        assert (res["short_period"], res["long_period"]) == (5, 20)
        assert mc.classifier.get_config()["long_period"] == 20

    def test_get_config_reports_window_resolution(self):
        mc = MarketContext(self._ou_kpi())
        mc.run()
        assert mc.get_config()["window_resolution"]["source"] == "hurst_ou"

    def test_derive_ema_windows_none_for_trending(self):
        rng = np.random.default_rng(3)
        price = pd.Series(100 * np.cumprod(1 + rng.normal(0.003, 0.002, 1000)))
        # Strong drift, short series → should not converge
        assert derive_ema_windows(price, min_estimates=10) is None


# ---------------------------------------------------------------------------
# regime_stable logic
# ---------------------------------------------------------------------------

class TestRegimeStable:
    def test_run_length_threshold(self):
        regime = pd.Series(["BULL"] * 5 + ["BEAR"] * 3 + ["BULL"] * 4)
        stable = _rolling_stability(regime, window=3)
        # First 2 of each run unstable, then stable
        assert list(stable) == [
            False, False, True, True, True,   # BULL x5
            False, False, True,               # BEAR x3
            False, False, True, True,         # BULL x4
        ]

    def test_nan_regime_never_stable(self):
        regime = pd.Series(["BULL", "BULL", np.nan, "BULL", "BULL"])
        stable = _rolling_stability(regime, window=2)
        assert stable.iloc[2] == False  # the NaN bar
        assert stable.iloc[1] == True
        assert stable.iloc[4] == True

    def test_window_one_all_classified_stable(self):
        regime = pd.Series(["BULL", "BEAR", np.nan, "NEUTRAL"])
        stable = _rolling_stability(regime, window=1)
        assert list(stable) == [True, True, False, True]


# ---------------------------------------------------------------------------
# Hurst / OU analysis tooling
# ---------------------------------------------------------------------------

class TestHurstTooling:
    def _ou_series(self, n=3000, theta=0.05, sigma=0.01, seed=0):
        rng = np.random.default_rng(seed)
        mu = np.log(100.0)
        x = [mu]
        for _ in range(n - 1):
            x.append(x[-1] + theta * (mu - x[-1]) + sigma * rng.standard_normal())
        return np.exp(np.array(x))

    def test_hurst_mean_reverting_below_half(self):
        h = hurst_dfa(self._ou_series())
        assert h < 0.5

    def test_hurst_returns_finite_for_random_walk(self):
        rng = np.random.default_rng(1)
        price = 100 * np.cumprod(1 + rng.normal(0.001, 0.005, 3000))
        h = hurst_dfa(price)
        assert np.isfinite(h)

    def test_hurst_nan_for_too_short_series(self):
        assert np.isnan(hurst_dfa(np.linspace(100, 101, 20)))

    def test_ou_halflife_recovers_timescale(self):
        # theta=0.05 → half-life = -ln2 / ln(1-0.05) ≈ 13.5 candles
        hl = ou_halflife(self._ou_series(theta=0.05))
        assert hl is not None
        assert 8 < hl < 20

    def test_ou_halflife_none_for_trending(self):
        rng = np.random.default_rng(2)
        price = 100 * np.cumprod(1 + rng.normal(0.002, 0.003, 2000))
        # Strong positive drift → not mean reverting → None
        assert ou_halflife(price) is None

    def test_rolling_halflife_shape(self):
        prices = pd.Series(self._ou_series(2000))
        hl = rolling_halflife(prices, window_candles=336, stride_candles=24)
        assert len(hl) > 0
        assert hl.dropna().median() > 0

    def test_variance_ratio_below_one_for_mean_reversion(self):
        vr = variance_ratio_profile(self._ou_series(), lags_candles=[24, 48, 96])
        assert vr[96] < 1.0

    def test_suggest_ema_windows(self):
        prices = pd.Series(self._ou_series(3000, theta=0.04))
        out = suggest_ema_windows(prices, timeframe="1h", estimation_window_hours=336)
        assert out["suggested_long_period"] > out["suggested_short_period"]
        assert out["suggested_short_period"] >= 1
        assert out["n_estimates"] > 0
