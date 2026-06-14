"""Shared pytest fixtures for the FORGE test suite.

Module-scoped fixtures provide expensive synthetic datasets once per test
module, avoiding redundant computation across test classes that use identical
inputs.  Tests that need *different* parameters must build their own data.
"""
import numpy as np
import pandas as pd
import pytest


def _build_kpi_table_with_indicators(n: int, seed: int = 42) -> pd.DataFrame:
    """OHLCV + RSI/EMA/SMA indicators, used by event-discovery tests."""
    rng = np.random.default_rng(seed)
    price = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    vol = np.abs(rng.normal(1e6, 2e5, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    def sma(s, w):
        return pd.Series(s).rolling(w, min_periods=1).mean().values

    def ema(s, w):
        return pd.Series(s).ewm(span=w, adjust=False).mean().values

    def rsi(s, w=14):
        d = pd.Series(s).diff()
        g = d.clip(lower=0).rolling(w, min_periods=1).mean()
        lo = (-d.clip(upper=0)).rolling(w, min_periods=1).mean()
        return (100 - 100 / (1 + g / lo.replace(0, np.nan))).fillna(50).values

    return pd.DataFrame(
        {
            "open_dt": dates,
            "close": price,
            "volume": vol,
            "close_rsi_14": rsi(price, 14),
            "close_rsi_25": rsi(price, 25),
            "close_ema_09": ema(price, 9),
            "close_ema_25": ema(price, 25),
            "close_sma_25": sma(price, 25),
            "volume_sma_25": sma(vol, 25),
            "close_bb_lower_20": pd.Series(sma(price, 20))
            - 2 * pd.Series(price).rolling(20, min_periods=1).std().values,
            "close_bb_upper_20": pd.Series(sma(price, 20))
            + 2 * pd.Series(price).rolling(20, min_periods=1).std().values,
            "close_min_24": pd.Series(price).rolling(24, min_periods=1).min().values,
            "close_max_24": pd.Series(price).rolling(24, min_periods=1).max().values,
        }
    )


@pytest.fixture(scope="module")
def kpi_4380():
    """4 380-bar (≈6 months, 1H) KPI table with indicators.  Shared across
    tests in the same module that use the default dataset size."""
    return _build_kpi_table_with_indicators(4380, seed=42)


@pytest.fixture(scope="module")
def kpi_8760():
    """8 760-bar (≈12 months, 1H) KPI table with indicators.  Used by tests
    that need enough history for scale-free detection and walk-forward splits."""
    return _build_kpi_table_with_indicators(8760, seed=7)
