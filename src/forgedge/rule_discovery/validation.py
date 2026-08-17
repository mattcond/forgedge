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
from ..episodes import concurrency
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


def expectancy_mde(net: np.ndarray, alpha: float = 0.05,
                   n_eff: Optional[float] = None) -> float:
    """Minimum detectable expectancy of a one-sided one-sample t-test.

    Given the executed trades' net gains, returns the smallest true mean
    return that the ``expectancy > 0`` t-test on this sample size and
    dispersion would detect at significance ``alpha``:
    ``t_crit(df) · sd / √n`` — the same detectability convention as
    ``alpha_discovery.stats.min_detectable_effect``, specialised to the
    one-sample case (no power term: this is the bar the *observed* mean must
    clear, so comparing it to a claimed effect answers "could this sample even
    confirm an effect of that size?").

    ``n_eff`` is the effective sample size (F16): overlapping trades are not
    independent observations, and the bar an observed mean must clear is set by
    how much *independent* evidence there is, not by how many rows the ledger
    has.  ``None`` keeps the nominal count.

    Returns ``inf`` when the sample is smaller than 2 (t-test undefined).
    """
    n = int(net.size)
    if n < 2:
        return float("inf")
    sd = float(np.std(net, ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        return 0.0
    eff = float(n) if n_eff is None or not np.isfinite(n_eff) else max(float(n_eff), 2.0)
    t_crit = stats.t_ppf_onesided(alpha, eff - 1.0)
    if not math.isfinite(t_crit):
        return float("inf")
    return float(t_crit * sd / math.sqrt(eff))


def _profit_factor(net: np.ndarray) -> float:
    pos = float(net[net > 0].sum())
    neg = float(-net[net < 0].sum())
    if neg == 0:
        return 9999.0 if pos > 0 else 0.0
    return pos / neg


def opportunity_sharpe(trades: pd.DataFrame, span_years: float) -> float:
    """Risk-adjusted return per unit of *time*, from the realised trade count.

    ``(mu / sigma) * sqrt(trades per year)`` — deliberately **not** the same
    quantity as :attr:`StatisticalValidation.sharpe_ratio`, and the difference
    is the whole reason this function exists.

    ``validate`` annualises by *capacity*: ``bars_per_year / avg_holding_bars``,
    the number of non-overlapping holding periods that fit in a year.  That is
    the right denominator for asking "how good is this rule", because it does
    not reward a rule for the accident of how often it happened to fire.

    It is the wrong denominator for choosing between two operating points on the
    *same* rule.  A deeper limit entry fills less often while holding for the
    same length, so capacity barely moves and the two points would annualise
    almost identically — the frequency the choice is about would cancel out, and
    a point that trades half as often for a slightly better per-trade edge would
    look strictly better.  Counting realised trades restores the trade-off:
    halving the trades costs ``sqrt(2) ~ 1.41x``, which the per-trade quality
    has to beat to break even.

    Both operating points are measured over the identical out-of-sample windows,
    so ``span_years`` is common to them and the comparison is scale-free in it;
    it is applied anyway so the number reported is an annual rate rather than an
    uninterpretable intermediate.

    Returns ``nan`` for fewer than two trades or dispersion indistinguishable
    from zero.  The second guard is relative, not ``sd > 0``: a ledger of
    identical gains has a standard deviation around ``1e-18`` rather than
    exactly zero, which would sail through and yield a Sharpe of ``1e16``.  An
    "infinitely good" operating point entering a comparison would win it every
    time; zero dispersion means the ratio is *undefined*, not enormous.
    """
    if trades is None or len(trades) < 2 or span_years <= 0:
        return float("nan")
    net = trades["net_pct_gain"].to_numpy(dtype=float)
    mu = float(net.mean())
    sd = float(net.std(ddof=1))
    if not (np.isfinite(sd) and sd > 1e-9 * max(abs(mu), 1.0)):
        return float("nan")
    trades_per_year = len(net) / float(span_years)
    return (mu / sd) * math.sqrt(max(trades_per_year, 1e-12))


def _effective_sample(trades: pd.DataFrame) -> float:
    """Independent observations behind a trade ledger (F16, #177).

    ``total_trades / mean_concurrent_positions``: a stack of positions opened on
    consecutive bars and held over the same stretch of price is one path
    observed once, not N times.  See :mod:`forgedge.episodes` for why
    concurrency and not episodes — trades from separate episodes still overlap
    when the horizon outruns the gap between them, and it is the shared *path*
    that breaks independence.

    Returns ``nan`` when the ledger carries no bar geometry (a hand-built frame,
    or one that has been through a serialisation that dropped the columns), and
    the callers then fall back to the nominal count.  Falling back is the honest
    default: an unmeasured overlap is not evidence of no overlap, but inventing
    a correction would be worse than reporting the uncorrected number and saying
    so on ``StatisticalValidation.n_effective``.
    """
    if trades is None or len(trades) < 2:
        return float("nan")
    if not {"fill_rn", "exit_rn"} <= set(trades.columns):
        return float("nan")
    stats_ = concurrency(trades["fill_rn"].to_numpy(), trades["exit_rn"].to_numpy())
    return stats_.effective_trades


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

    Notes
    -----
    The **effective** sample size is measured here, from the ledger's own
    ``fill_rn``/``exit_rn`` geometry, rather than accepted as an argument: there
    is then no way to hand this function a number that does not describe the
    trades it was given.  It corrects the t-tests' precision and the DSR's
    ``n_obs`` and nothing else — see :class:`StatisticalValidation`.
    """
    net = trades["net_pct_gain"].to_numpy(dtype=float)
    n = net.size
    nan = float("nan")
    n_eff = _effective_sample(trades)

    if n < 2:
        return StatisticalValidation(
            ttest_winrate_t=nan, ttest_winrate_p=nan,
            ttest_expectancy_t=nan, ttest_expectancy_p=nan,
            deflated_sharpe=nan, sharpe_ratio=nan,
            n_trials_tested=n_trials, temporal_stability="FAIL",
            pf_first_half=nan, pf_second_half=nan, n_effective=n_eff,
        )

    # ── win-rate vs base rate (one-sample → one-sample-style via constant) ──
    wins = (net > 0).astype(float)
    t_wr, p_wr = _ttest_1samp_greater(wins, base_rate, n_eff)

    # ── expectancy vs 0 ──────────────────────────────────────────────────
    t_exp, p_exp = _ttest_1samp_greater(net, 0.0, n_eff)

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
    # `n_obs` is the independent evidence behind the Sharpe, not the row
    # count: overlapping trades do not each buy a full observation.
    dsr = deflated_sharpe(sharpe_annual, n_trials,
                          int(n_eff) if np.isfinite(n_eff) else n)

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
        n_effective=_round(n_eff),
    )


def _ttest_1samp_greater(sample: np.ndarray, popmean: float,
                        n_eff: Optional[float] = None):
    """One-sample, one-sided ``mean(sample) > popmean`` t-test (pure numpy).

    Built on ``alpha_discovery.stats`` Student-t tail so no SciPy is needed.

    ``n_eff`` is the **effective** sample size: overlapping trades share a price
    path and are not independent observations, so treating the nominal count as
    a sample size overstates the precision of the mean by ``sqrt(n/n_eff)``
    (F16, #177).  The point estimates still use every trade — more observations
    do sharpen the estimate of the mean; what they do not do is sharpen it as
    fast as ``sqrt(n)`` when they overlap.  Only the standard error and the
    degrees of freedom are corrected.

    ``None`` (default) keeps the nominal count, which is correct whenever
    nothing overlaps and is the honest fallback when concurrency could not be
    measured.
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
    eff = float(n) if n_eff is None or not np.isfinite(n_eff) else max(float(n_eff), 2.0)
    se = sd / math.sqrt(eff)
    t = (mean - popmean) / se
    df = eff - 1.0
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
