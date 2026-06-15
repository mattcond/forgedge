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
# threshold_mode: fixed (absolute) vs balanced (distributional)
# ---------------------------------------------------------------------------

class TestThresholdMode:
    def _varied_close(self, n=6000, seed=7):
        # A series whose EMA ratio has a wide spread, so quantile cuts are
        # well-defined across all five buckets.
        rng = np.random.default_rng(seed)
        steps = rng.normal(0, 0.01, n)
        steps[: n // 2] += 0.0004   # an up-leg then a down-leg → both tails
        steps[n // 2 :] -= 0.0004
        return 100 * np.cumprod(1 + steps)

    def test_fixed_is_default(self):
        assert EMAProxyConfig().threshold_mode == "fixed"

    def test_balanced_matches_target_distribution(self):
        df = pd.DataFrame({"close": self._varied_close()})
        target = [0.10, 0.20, 0.40, 0.20, 0.10]
        labels = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        cfg = EMAProxyConfig(
            auto_window=False, short_period=20, long_period=120,
            threshold_mode="balanced", target_distribution=target,
        )
        regime = EMAProxyClassifier(cfg, labels).classify(df)
        shares = regime.value_counts(normalize=True).reindex(
            ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        ).to_numpy()
        # Quantile cut → each share within a small tolerance of the target.
        assert np.allclose(shares, target, atol=0.03)

    def test_balanced_uniform_target(self):
        df = pd.DataFrame({"close": self._varied_close()})
        labels = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        cfg = EMAProxyConfig(
            auto_window=False, short_period=20, long_period=120,
            threshold_mode="balanced", target_distribution=[1, 1, 1, 1, 1],
        )
        regime = EMAProxyClassifier(cfg, labels).classify(df)
        shares = regime.value_counts(normalize=True).reindex(labels).to_numpy()
        assert np.allclose(shares, [0.2] * 5, atol=0.03)

    def test_balanced_records_resolved_thresholds(self):
        df = pd.DataFrame({"close": self._varied_close()})
        labels = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        clf = EMAProxyClassifier(
            EMAProxyConfig(auto_window=False, threshold_mode="balanced"), labels
        )
        clf.classify(df)
        cfg = clf.get_config()
        assert cfg["resolved_threshold_mode"] == "balanced"
        assert len(cfg["resolved_thresholds"]) == 4
        # data-driven cut points differ from the fixed defaults
        assert cfg["resolved_thresholds"] != list(EMAProxyConfig().thresholds)

    def test_balanced_falls_back_on_degenerate_ratio(self):
        # Constant price → ratio ≡ 1 → quantiles collapse → fixed fallback
        df = pd.DataFrame({"close": np.full(500, 100.0)})
        labels = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        clf = EMAProxyClassifier(
            EMAProxyConfig(auto_window=False, threshold_mode="balanced"), labels
        )
        clf.classify(df)
        assert clf.resolved_threshold_mode == "fixed"

    def test_invalid_threshold_mode_raises(self):
        labels = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        with pytest.raises(ValueError):
            EMAProxyClassifier(EMAProxyConfig(threshold_mode="quantile"), labels)

    def test_bad_target_distribution_length_raises(self):
        labels = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        cfg = EMAProxyConfig(threshold_mode="balanced",
                             target_distribution=[0.5, 0.5])
        with pytest.raises(ValueError):
            EMAProxyClassifier(cfg, labels)

    def test_nonpositive_target_weight_raises(self):
        labels = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        cfg = EMAProxyConfig(threshold_mode="balanced",
                             target_distribution=[0.3, 0.3, 0.4, 0.0, 0.0])
        with pytest.raises(ValueError):
            EMAProxyClassifier(cfg, labels)

    def test_balanced_via_market_context(self):
        df = pd.DataFrame(
            {"open_dt": pd.date_range("2024-01-01", periods=6000, freq="1h"),
             "close": self._varied_close()}
        )
        cfg = MarketContextConfig(
            ema_proxy=EMAProxyConfig(window_unit="bar", threshold_mode="balanced")
        )
        mc = MarketContext(df, cfg)
        mc.run()
        # Every label populated → two non-empty tails.
        dist = mc.distribution()
        assert (dist["n_bars"].fillna(0) > 0).all()

    # -- threshold_basis: global (default) vs expanding (causal) ------------

    def _balanced_clf(self, basis, warmup=200):
        labels = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        cfg = EMAProxyConfig(
            auto_window=False, short_period=20, long_period=120,
            threshold_mode="balanced", threshold_basis=basis,
            threshold_warmup=warmup,
        )
        return EMAProxyClassifier(cfg, labels)

    def test_global_is_default_basis(self):
        assert EMAProxyConfig().threshold_basis == "global"

    def test_expanding_is_causal(self):
        # A label must not depend on any future bar.
        df = pd.DataFrame({"close": self._varied_close()})
        clf = self._balanced_clf("expanding")
        full = clf.classify(df)
        t = 4000
        trunc = clf.classify(df.iloc[: t + 1])
        assert str(full.iloc[t]) == str(trunc.iloc[t])
        assert clf.resolved_threshold_basis == "expanding"

    def test_global_is_not_causal(self):
        # Global quantiles use the whole sample → look-ahead (contrast case).
        df = pd.DataFrame({"close": self._varied_close()})
        clf = self._balanced_clf("global")
        full = clf.classify(df)
        # Truncating the future shifts the global quantiles, changing early labels.
        trunc = clf.classify(df.iloc[:5000])
        early_full = full.iloc[:5000].astype(str).to_numpy()
        early_trunc = trunc.astype(str).to_numpy()
        assert (early_full != early_trunc).any()

    def test_expanding_warmup_uses_fixed(self):
        # Within the warm-up the cut must equal the plain fixed classification.
        df = pd.DataFrame({"close": self._varied_close()})
        warm = 300
        exp = self._balanced_clf("expanding", warmup=warm).classify(df)
        fixed_cfg = EMAProxyConfig(auto_window=False, short_period=20,
                                   long_period=120, threshold_mode="fixed")
        fixed = EMAProxyClassifier(
            fixed_cfg, ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        ).classify(df)
        head = slice(0, warm - 1)
        assert (exp.iloc[head].astype(str) == fixed.iloc[head].astype(str)).all()

    def test_invalid_threshold_basis_raises(self):
        labels = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        with pytest.raises(ValueError):
            EMAProxyClassifier(EMAProxyConfig(threshold_basis="rolling"), labels)

    def test_bad_warmup_raises(self):
        labels = ["STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"]
        with pytest.raises(ValueError):
            EMAProxyClassifier(EMAProxyConfig(threshold_warmup=0), labels)


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

    def test_unsorted_column_input_same_as_sorted(self):
        """EMA, regime and stability must be identical regardless of input row order (datetime column path)."""
        df_sorted = _make_kpi_table(n=2000)
        df_shuffled = df_sorted.sample(frac=1, random_state=7).reset_index(drop=True)

        cfg = MarketContextConfig(
            ema_proxy=EMAProxyConfig(window_unit="day", window_estimation=30, window_stride=1)
        )
        out_sorted = MarketContext(df_sorted, cfg).run()
        out_shuffled = MarketContext(df_shuffled, cfg).run()

        # Both outputs must be in the same (chronological) order and produce
        # identical regime assignments.
        pd.testing.assert_series_equal(
            out_sorted["regime"].reset_index(drop=True),
            out_shuffled["regime"].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            out_sorted["regime_stable"].reset_index(drop=True),
            out_shuffled["regime_stable"].reset_index(drop=True),
            check_names=False,
        )

    def test_unsorted_datetimeindex_same_as_sorted(self):
        """DatetimeIndex path also sorts rows before calculations."""
        df = _make_kpi_table(n=2000).set_index("open_dt")
        df_shuffled = df.sample(frac=1, random_state=13)

        cfg = MarketContextConfig(
            ema_proxy=EMAProxyConfig(window_unit="day", window_estimation=30, window_stride=1)
        )
        out_sorted = MarketContext(df, cfg).run()
        out_shuffled = MarketContext(df_shuffled, cfg).run()

        pd.testing.assert_series_equal(
            out_sorted["regime"].sort_index(),
            out_shuffled["regime"].sort_index(),
            check_names=False,
        )

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
# regime_table() — compact frame for joining back onto the source data
# ---------------------------------------------------------------------------

class TestRegimeTable:
    def test_columns_and_length_from_datetime_column(self):
        df = _make_kpi_table()  # has open_dt column, RangeIndex
        mc = MarketContext(df)
        mc.run()
        rt = mc.regime_table()
        assert list(rt.columns) == ["open_dt", "regime", "regime_stable"]
        assert len(rt) == len(df)
        assert isinstance(rt["regime"].dtype, pd.CategoricalDtype)
        assert rt["regime"].cat.ordered
        assert rt["regime_stable"].dtype == bool

    def test_uses_datetimeindex(self):
        df = _make_kpi_table().set_index("open_dt")
        mc = MarketContext(df)
        mc.run()
        rt = mc.regime_table()
        assert rt.columns[0] == "open_dt"
        assert pd.api.types.is_datetime64_any_dtype(rt["open_dt"])
        assert len(rt) == len(df)

    def test_joins_back_onto_source(self):
        df = _make_kpi_table()
        mc = MarketContext(df)
        mc.run()
        joined = df.merge(mc.regime_table(), on="open_dt", how="left")
        assert "regime" in joined.columns
        assert "regime_stable" in joined.columns
        assert len(joined) == len(df)
        assert joined["regime"].notna().any()
        # The join must align row-for-row with run()'s in-place output.
        assert (joined["regime"].astype(str) == mc._result["regime"].astype(str)).all()

    def test_custom_timestamp_col_name(self):
        df = _make_kpi_table().rename(columns={"open_dt": "ts"})
        mc = MarketContext(df, MarketContextConfig(
            ema_proxy=EMAProxyConfig(window_unit="bar")))
        mc.run()
        rt = mc.regime_table(timestamp_col="ts")
        assert list(rt.columns) == ["ts", "regime", "regime_stable"]

    def test_before_run_raises(self):
        with pytest.raises(RuntimeError):
            MarketContext(_make_kpi_table()).regime_table()

    def test_unknown_timestamp_col_raises(self):
        df = _make_kpi_table()
        mc = MarketContext(df)
        mc.run()
        with pytest.raises(KeyError):
            mc.regime_table(timestamp_col="does_not_exist")


# ---------------------------------------------------------------------------
# Automatic EMA window selection (Hurst/OU) with fallback
# ---------------------------------------------------------------------------

class TestAutoWindow:
    @pytest.fixture(scope="class")
    @classmethod
    def ou_kpi_default(cls):
        """3 000-bar OU price table shared by tests that use default theta=0.05.
        Each test creates its own MarketContext, so sharing the input is safe."""
        prices = _ou_prices(3000, theta=0.05)
        return pd.DataFrame(
            {"open_dt": pd.date_range("2024-01-01", periods=3000, freq="1h"),
             "close": prices}
        )

    def _ou_kpi(self, n=3000, theta=0.05):
        prices = _ou_prices(n, theta=theta)
        return pd.DataFrame(
            {"open_dt": pd.date_range("2024-01-01", periods=n, freq="1h"),
             "close": prices}
        )

    def _bar_cfg(self, **ema):
        # Exercise the derivation engine in bar mode (W=168 bars) so it converges
        # on the short synthetic series; the day-mode default is covered in
        # TestWindowUnit.
        params = dict(window_unit="bar", window_estimation=168, window_stride=24)
        params.update(ema)
        return MarketContextConfig(ema_proxy=EMAProxyConfig(**params))

    def test_windows_derived_from_data(self, ou_kpi_default):
        mc = MarketContext(ou_kpi_default, self._bar_cfg())
        mc.run()
        res = mc.window_resolution
        assert res["source"] == "hurst_ou"
        assert res["half_life_bars"] > 0
        cfg = mc.classifier.get_config()
        assert cfg["short_period"] == res["short_period"]
        assert cfg["long_period"] == res["long_period"]
        assert res["short_period"] < res["long_period"]

    def test_derived_windows_differ_from_default_when_warranted(self):
        # theta=0.2 → half-life ≈ -ln2/ln(0.8) ≈ 3.1 bars → spans ≠ 9/25
        mc = MarketContext(self._ou_kpi(theta=0.2), self._bar_cfg())
        mc.run()
        assert mc.window_resolution["source"] == "hurst_ou"
        assert mc.window_resolution["long_period"] != 25

    def test_fallback_when_not_enough_data(self):
        # Only ~6 estimation windows possible < min_window_estimates=10 → fallback
        df = self._ou_kpi(n=300)
        mc = MarketContext(df, self._bar_cfg())
        mc.run()
        res = mc.window_resolution
        assert res["source"] == "fallback"
        assert (res["short_period"], res["long_period"]) == (9, 25)

    def test_auto_window_disabled_uses_configured(self, ou_kpi_default):
        cfg = self._bar_cfg(auto_window=False, short_period=5, long_period=20)
        mc = MarketContext(ou_kpi_default, cfg)
        mc.run()
        res = mc.window_resolution
        assert res["source"] == "configured"
        assert (res["short_period"], res["long_period"]) == (5, 20)
        assert mc.classifier.get_config()["long_period"] == 20

    def test_get_config_reports_window_resolution(self, ou_kpi_default):
        mc = MarketContext(ou_kpi_default, self._bar_cfg())
        mc.run()
        assert mc.get_config()["window_resolution"]["source"] == "hurst_ou"

    def test_derive_ema_windows_none_for_trending(self):
        rng = np.random.default_rng(3)
        price = pd.Series(100 * np.cumprod(1 + rng.normal(0.003, 0.002, 1000)))
        # Strong drift, short series → should not converge
        assert derive_ema_windows(price, min_estimates=10) is None

    def test_run_robust_to_nonpositive_close(self):
        """A zero/negative-prefixed close must not crash the OU window fit."""
        prices = _ou_prices(1500, theta=0.05)
        prices[:50] = 0.0  # invalid prices in the first window
        df = pd.DataFrame(
            {"open_dt": pd.date_range("2024-01-01", periods=len(prices), freq="1h"),
             "close": prices}
        )
        # Bar mode so the OU path actually runs on the bad-prefixed series.
        mc = MarketContext(df, self._bar_cfg())
        out = mc.run()  # must not raise LinAlgError
        assert mc.window_resolution["source"] in {"hurst_ou", "fallback"}
        # Regime is still produced on the valid region.
        assert out["regime"].iloc[100:].notna().any()


# ---------------------------------------------------------------------------
# window_unit: bar (default, timeframe-agnostic) vs day (timeframe-coherent)
# ---------------------------------------------------------------------------

class TestWindowUnit:
    @pytest.fixture(scope="class")
    @classmethod
    def ou_kpi_default(cls):
        """3 000-bar OU price table (1H) shared by tests that use default params."""
        prices = _ou_prices(3000, theta=0.05)
        return pd.DataFrame(
            {"open_dt": pd.date_range("2024-01-01", periods=3000, freq="1h"),
             "close": prices}
        )

    def _ou_kpi(self, n=3000, freq="1h", theta=0.05):
        prices = _ou_prices(n, theta=theta)
        return pd.DataFrame(
            {"open_dt": pd.date_range("2024-01-01", periods=n, freq=freq),
             "close": prices}
        )

    def test_day_is_default(self):
        cfg = EMAProxyConfig()
        assert cfg.window_unit == "day"
        assert cfg.window_estimation == 168  # 168 days by default

    def test_bar_mode_uses_bar_window(self, ou_kpi_default):
        # window_estimation = 168 interpreted as 168 *bars* in bar mode.
        cfg = MarketContextConfig(
            ema_proxy=EMAProxyConfig(window_unit="bar", window_estimation=168)
        )
        mc = MarketContext(ou_kpi_default, cfg)
        mc.run()
        assert mc.window_resolution["unit"] == "bar"
        assert mc.window_resolution["estimation_window_bars"] == 168

    def test_day_mode_same_W_means_days_on_1h(self, ou_kpi_default):
        # The SAME single value (168) is reinterpreted as 168 *days* on 1h,
        # i.e. 168 * 24 = 4032 bars — even though the timeframe is hourly.
        cfg = MarketContextConfig(
            ema_proxy=EMAProxyConfig(window_unit="day", window_estimation=168)
        )
        mc = MarketContext(ou_kpi_default, cfg)
        mc.run()
        res = mc.window_resolution
        assert res["unit"] == "day"
        assert abs(res["bar_hours"] - 1.0) < 1e-6
        assert res["estimation_window_bars"] == 4032  # 168d * 24 / 1h

    def test_day_mode_converts_per_timeframe_4h(self):
        # 168 days on 4h → 168 * 24 / 4 = 1008 bars.
        cfg = MarketContextConfig(
            ema_proxy=EMAProxyConfig(window_unit="day", window_estimation=168)
        )
        mc = MarketContext(self._ou_kpi(n=2000, freq="4h"), cfg)
        mc.run()
        res = mc.window_resolution
        assert abs(res["bar_hours"] - 4.0) < 1e-6
        assert res["estimation_window_bars"] == 1008  # 168d * 24 / 4h

    def test_day_mode_small_W_converges_on_1h(self, ou_kpi_default):
        # 7-day window, 1-day stride on 1h → 168-bar window, 24-bar stride:
        # both W and stride follow the "day" unit.
        cfg = MarketContextConfig(
            ema_proxy=EMAProxyConfig(
                window_unit="day", window_estimation=7, window_stride=1
            )
        )
        mc = MarketContext(ou_kpi_default, cfg)
        mc.run()
        res = mc.window_resolution
        assert res["estimation_window_bars"] == 168  # 7d * 24 / 1h
        assert res["source"] == "hurst_ou"

    def test_day_mode_requires_time_info(self):
        df = pd.DataFrame({"close": _ou_prices(2000)})  # RangeIndex, no datetime
        cfg = MarketContextConfig(ema_proxy=EMAProxyConfig(window_unit="day"))
        with pytest.raises(ValueError):
            MarketContext(df, cfg).run()

    def test_day_mode_explicit_bar_hours(self):
        df = pd.DataFrame({"close": _ou_prices(3000)})
        cfg = MarketContextConfig(
            ema_proxy=EMAProxyConfig(
                window_unit="day", window_estimation=7, bar_hours=1.0
            )
        )
        mc = MarketContext(df, cfg)
        mc.run()
        assert mc.window_resolution["estimation_window_bars"] == 168

    def test_invalid_window_unit_raises(self, ou_kpi_default):
        cfg = MarketContextConfig(ema_proxy=EMAProxyConfig(window_unit="hour"))
        with pytest.raises(ValueError):
            MarketContext(ou_kpi_default, cfg).run()


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

    def test_log_functions_handle_nonpositive_without_crashing(self):
        bad = np.array([1.0, 1.1, 0.0, -2.0, 1.2] * 40, dtype=float)
        assert ou_halflife(bad) is None
        assert np.isnan(hurst_dfa(bad))
        vr = variance_ratio_profile(bad, lags_candles=[4, 8])
        assert all(np.isnan(v) for v in vr.values())

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
