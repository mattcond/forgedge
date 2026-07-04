"""Statistical validation of a selected rule configuration (Step 4).

Confirms the in-sample edge is statistically real rather than a multiple-testing
artefact:

* **win-rate significance** — one-sided t-test of the per-trade win/loss series
  against the Alpha Contract's base rate;
* **expectancy significance** — one-sided t-test of the net per-trade gains
  against zero;
* **Deflated Sharpe Ratio** — corrects the selected Sharpe for the number of
  grid trials, the standard guard against picking a lucky configuration;
* **temporal stability** — profit factor on the first vs second half of the
  trades, flagging strategies that only worked early.

All p-values reuse the pure-numpy Student-t machinery from
``alpha_discovery.stats`` — no SciPy dependency.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from ..alpha_discovery import stats
from .models import StatisticalValidation


def deflated_sharpe(sr_selected: float, n_trials: int, n_obs: int) -> float:
    """Deflate a Sharpe ratio for selection over ``n_trials`` configurations.

    Implements the haircut from the Rule Discovery spec (Section 4.3)::

        correction = sqrt(1 - gamma * ln(n_trials) / ln(n_obs))
        dsr        = sr_selected * correction

    where ``gamma`` is the Euler-Mascheroni constant.  Returns ``sr_selected``
    unchanged when the correction is undefined (``n_trials <= 1`` or
    ``n_obs <= e``), and ``nan`` when the radicand goes negative (selection bias
    so severe the Sharpe is not credible).
    """
    gamma = 0.5772156649
    if not np.isfinite(sr_selected):
        return float("nan")
    if n_trials <= 1 or n_obs <= math.e:
        return float(sr_selected)
    radicand = 1.0 - gamma * math.log(n_trials) / math.log(n_obs)
    if radicand <= 0:
        return float("nan")
    return float(sr_selected * math.sqrt(radicand))


def expectancy_mde(net: np.ndarray, alpha: float = 0.05) -> float:
    """Minimum detectable expectancy of a one-sided one-sample t-test.

    Given the executed trades' net gains, returns the smallest true mean
    return that the ``expectancy > 0`` t-test on this sample size and
    dispersion would detect at significance ``alpha``:
    ``t_crit(df) · sd / √n`` — the same detectability convention as
    ``alpha_discovery.stats.min_detectable_effect``, specialised to the
    one-sample case (no power term: this is the bar the *observed* mean must
    clear, so comparing it to a claimed effect answers "could this sample even
    confirm an effect of that size?").

    Returns ``inf`` when the sample is smaller than 2 (t-test undefined).
    """
    n = int(net.size)
    if n < 2:
        return float("inf")
    sd = float(np.std(net, ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        return 0.0
    t_crit = stats.t_ppf_onesided(alpha, float(n - 1))
    if not math.isfinite(t_crit):
        return float("inf")
    return float(t_crit * sd / math.sqrt(n))


def _profit_factor(net: np.ndarray) -> float:
    pos = float(net[net > 0].sum())
    neg = float(-net[net < 0].sum())
    if neg == 0:
        return 9999.0 if pos > 0 else 0.0
    return pos / neg


def validate(
    trades: pd.DataFrame,
    base_rate: float,
    n_trials: int,
    bars_per_year: float = 24 * 365,
    avg_holding_bars: Optional[float] = None,
) -> StatisticalValidation:
    """Run the full statistical validation battery on the executed trades.

    Parameters
    ----------
    trades : pd.DataFrame
        Executed trades with a ``net_pct_gain`` column (and ``fill_dt`` for the
        temporal split).
    base_rate : float
        Target hit base rate from the Alpha Contract — the null win rate.
    n_trials : int
        Number of grid configurations evaluated during selection (for the DSR).
    bars_per_year : float
        Bars in a calendar year, used to annualise the Sharpe (default hourly).
    avg_holding_bars : float, optional
        Average holding length in bars; when given, the Sharpe is annualised by
        the number of *non-overlapping* holding periods per year rather than by
        the raw trade frequency.
    """
    net = trades["net_pct_gain"].to_numpy(dtype=float)
    n = net.size
    nan = float("nan")

    if n < 2:
        return StatisticalValidation(
            ttest_winrate_t=nan, ttest_winrate_p=nan,
            ttest_expectancy_t=nan, ttest_expectancy_p=nan,
            deflated_sharpe=nan, sharpe_ratio=nan,
            n_trials_tested=n_trials, temporal_stability="FAIL",
            pf_first_half=nan, pf_second_half=nan,
        )

    # ── win-rate vs base rate (one-sample → one-sample-style via constant) ──
    wins = (net > 0).astype(float)
    t_wr, p_wr = _ttest_1samp_greater(wins, base_rate)

    # ── expectancy vs 0 ──────────────────────────────────────────────────
    t_exp, p_exp = _ttest_1samp_greater(net, 0.0)

    # ── Sharpe (annualised) and its deflated version ─────────────────────
    mu = float(net.mean())
    sd = float(net.std(ddof=1))
    sharpe_trade = mu / sd if sd > 0 else nan
    if np.isfinite(sharpe_trade):
        if avg_holding_bars and avg_holding_bars > 0:
            periods_per_year = bars_per_year / avg_holding_bars
        else:
            periods_per_year = n  # fall back to realised trade count proxy
        sharpe_annual = sharpe_trade * math.sqrt(max(periods_per_year, 1.0))
    else:
        sharpe_annual = nan
    dsr = deflated_sharpe(sharpe_annual, n_trials, n)

    # ── temporal stability: PF first vs second half ──────────────────────
    if "fill_dt" in trades.columns:
        ordered = trades.sort_values("fill_dt")["net_pct_gain"].to_numpy(dtype=float)
    else:
        ordered = net
    mid = n // 2
    pf1 = _profit_factor(ordered[:mid])
    pf2 = _profit_factor(ordered[mid:])
    stability = _stability_label(pf1, pf2)

    return StatisticalValidation(
        ttest_winrate_t=_round(t_wr),
        ttest_winrate_p=_round(p_wr),
        ttest_expectancy_t=_round(t_exp),
        ttest_expectancy_p=_round(p_exp),
        deflated_sharpe=_round(dsr),
        sharpe_ratio=_round(sharpe_annual),
        n_trials_tested=n_trials,
        temporal_stability=stability,
        pf_first_half=_round(pf1),
        pf_second_half=_round(pf2),
    )


def _ttest_1samp_greater(sample: np.ndarray, popmean: float):
    """One-sample, one-sided ``mean(sample) > popmean`` t-test (pure numpy).

    Built on ``alpha_discovery.stats`` Student-t tail so no SciPy is needed.
    """
    a = sample[np.isfinite(sample)]
    n = a.size
    if n < 2:
        return float("nan"), float("nan")
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    if sd == 0:
        # Degenerate: all identical. Significant iff strictly above the null.
        return (float("inf"), 0.0) if mean > popmean else (0.0, 1.0)
    se = sd / math.sqrt(n)
    t = (mean - popmean) / se
    df = n - 1
    p_two = stats.student_t_sf_twosided(t, df)
    p_one = p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0
    return float(t), float(p_one)


def _stability_label(pf1: float, pf2: float) -> str:
    """Classify the first-vs-second-half PF drift (spec Section 4.4)."""
    if not (np.isfinite(pf1) and np.isfinite(pf2)):
        return "FAIL"
    # Both halves profitable and the weaker is at least half the stronger.
    if pf1 >= 1.0 and pf2 >= 1.0:
        weak, strong = sorted((pf1, pf2))
        return "PASS" if weak >= 0.5 * strong else "WARN"
    # One half clearly unprofitable.
    if pf1 < 1.0 and pf2 < 1.0:
        return "FAIL"
    return "WARN"


def _round(x: float) -> float:
    return round(float(x), 6) if (x is not None and np.isfinite(x)) else float("nan")
