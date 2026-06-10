"""Pure-numpy statistical primitives for Alpha Discovery.

Alpha Discovery needs Spearman rank correlation, an independent two-sample
t-test, and Benjamini-Hochberg FDR control — all with *real* p-values.  Rather
than pull in SciPy (FORGE keeps its runtime dependencies to ``numpy``/``pandas``
only, see ``market_context/hurst.py`` which ports an OU fit instead of using
statsmodels) this module implements the small amount of special-function
machinery required.

The Student-t tail probabilities are obtained from the regularised incomplete
beta function ``I_x(a, b)``, evaluated with the Lentz continued-fraction
algorithm from *Numerical Recipes*.  This reproduces SciPy's
``spearmanr`` / ``ttest_ind`` p-values to ~1e-6 over the ranges Alpha Discovery
cares about.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Regularised incomplete beta function  I_x(a, b)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    MAXIT = 300
    EPS = 3.0e-14
    FPMIN = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)`` for ``0 <= x <= 1``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf_twosided(t: float, df: float) -> float:
    """Two-sided tail probability ``P(|T| >= |t|)`` for Student-t with ``df``.

    Returns ``nan`` when ``df <= 0`` or ``t`` is not finite.
    """
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    return betai(df / 2.0, 0.5, x)


def _one_sided_from_two(t: float, p_two: float, alternative: str) -> float:
    """Convert a two-sided p-value to a one-sided one given the sign of ``t``."""
    if alternative == "two-sided":
        return p_two
    if alternative == "greater":
        return p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0
    if alternative == "less":
        return p_two / 2.0 if t < 0 else 1.0 - p_two / 2.0
    raise ValueError(
        f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}"
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rankdata_average(a: np.ndarray) -> np.ndarray:
    """Assign ranks to data, averaging the ranks of ties (``scipy`` 'average').

    Parameters
    ----------
    a : np.ndarray
        1-D array of values (must be finite — filter NaNs before calling).

    Returns
    -------
    np.ndarray
        Float array of ranks (1-based), tie groups sharing their mean rank.
    """
    arr = np.asarray(a)
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr_sorted = arr[sorter]
    obs = np.r_[True, arr_sorted[1:] != arr_sorted[:-1]]
    dense = obs.cumsum()[inv]

    # Cumulative count of the first element of each tie group.
    count = np.r_[np.nonzero(obs)[0], arr.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two equal-length finite arrays."""
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float(np.dot(xm, xm)) * float(np.dot(ym, ym)))
    if denom == 0:
        return float("nan")
    return float(np.dot(xm, ym) / denom)


# ---------------------------------------------------------------------------
# Public statistics
# ---------------------------------------------------------------------------

def spearmanr(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Spearman rank correlation coefficient and its two-sided p-value.

    Pairs containing a NaN in either series are dropped.  The p-value uses the
    t-distribution approximation ``t = r * sqrt((n - 2) / (1 - r**2))`` with
    ``n - 2`` degrees of freedom — the same approximation SciPy uses.

    Parameters
    ----------
    x, y : sequence of float
        Equal-length samples.

    Returns
    -------
    (rho, p_value) : tuple of float
        ``(nan, nan)`` when fewer than three valid pairs remain or a series is
        constant.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    n = xa.size
    if n < 3:
        return float("nan"), float("nan")

    rho = _pearson(rankdata_average(xa), rankdata_average(ya))
    if not math.isfinite(rho):
        return float("nan"), float("nan")
    if abs(rho) >= 1.0:
        return rho, 0.0

    df = n - 2
    t = rho * math.sqrt(df / (1.0 - rho * rho))
    p = student_t_sf_twosided(t, df)
    return float(rho), float(p)


def ttest_ind(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
    equal_var: bool = True,
) -> Tuple[float, float]:
    """Independent two-sample Student's t-test.

    Parameters
    ----------
    a, b : sequence of float
        The two samples (NaNs are dropped independently).
    alternative : {'two-sided', 'greater', 'less'}
        ``'greater'`` tests ``mean(a) > mean(b)``.
    equal_var : bool
        ``True`` (default) → pooled-variance test (matches SciPy's default and
        the Cohen's d convention used in the Alpha Discovery doc).  ``False`` →
        Welch's t-test with the Welch-Satterthwaite degrees of freedom.

    Returns
    -------
    (t_stat, p_value) : tuple of float
        ``(nan, nan)`` when either sample has fewer than two valid points or
        the pooled variance is zero.
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    na, nb = aa.size, bb.size
    if na < 2 or nb < 2:
        return float("nan"), float("nan")

    ma, mb = aa.mean(), bb.mean()
    va, vb = aa.var(ddof=1), bb.var(ddof=1)

    if equal_var:
        df = na + nb - 2
        sp2 = ((na - 1) * va + (nb - 1) * vb) / df
        denom = math.sqrt(sp2 * (1.0 / na + 1.0 / nb))
    else:
        se2a, se2b = va / na, vb / nb
        denom = math.sqrt(se2a + se2b)
        if denom == 0:
            return float("nan"), float("nan")
        df = (se2a + se2b) ** 2 / (
            se2a ** 2 / (na - 1) + se2b ** 2 / (nb - 1)
        )

    if denom == 0:
        return float("nan"), float("nan")

    t = (ma - mb) / denom
    p_two = student_t_sf_twosided(t, df)
    return float(t), float(_one_sided_from_two(t, p_two, alternative))


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's *d* effect size using the pooled standard deviation.

    Matches the doc's definition::

        pooled = sqrt((std(a)**2 + std(b)**2) / 2)
        d      = (mean(a) - mean(b)) / pooled
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if aa.size < 2 or bb.size < 2:
        return float("nan")
    pooled = math.sqrt((aa.var(ddof=1) + bb.var(ddof=1)) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((aa.mean() - bb.mean()) / pooled)


def benjamini_hochberg(p_values: Sequence[float], q: float) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure controlling the FDR at level ``q``.

    Parameters
    ----------
    p_values : sequence of float
        One p-value per hypothesis.  ``nan`` entries are treated as
        non-significant (never promoted) and excluded from the count ``m`` so
        they neither help nor penalise the others.
    q : float
        Target false-discovery rate (e.g. ``0.10`` = at most 10% false
        positives among the rejected hypotheses).

    Returns
    -------
    np.ndarray
        Boolean mask aligned to ``p_values``; ``True`` where the hypothesis is
        rejected (i.e. the candidate is promoted).
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    promoted = np.zeros(n, dtype=bool)

    valid_idx = np.where(np.isfinite(p))[0]
    m = valid_idx.size
    if m == 0:
        return promoted

    valid_p = p[valid_idx]
    order = np.argsort(valid_p, kind="mergesort")
    sorted_p = valid_p[order]

    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds
    if not below.any():
        return promoted

    cutoff = np.nonzero(below)[0].max()
    promoted[valid_idx[order[: cutoff + 1]]] = True
    return promoted
