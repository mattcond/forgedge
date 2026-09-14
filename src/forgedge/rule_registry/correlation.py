"""Step 2 — correlation matrices for the Rule Registry (FORGE Modulo 4).

Two matrices are computed over the whole pool of registry documents,
independently of the source ticker — two rules extracted on different tickers
can still be correlated if they capture the same market pattern (spec
Section 6).

* **Matrix A — Jaccard on the activations.**  Measures temporal overlap: how
  many bars (by date) are covered by both rules.  For rules extracted on
  different tickers the comparison is by date, never by row index (which
  differs across KPI tables).
* **Matrix B — Spearman on the gains.**  Measures return correlation: do the
  two rules win and lose under the same market conditions?  Gains are aligned by
  date; bars with no trade contribute ``0``.

The Spearman coefficient is computed scipy-free (rank + Pearson) to keep the
package runtime ``numpy``/``pandas``-only, matching the rest of FORGE.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .models import CorrelationMatrices, RuleDocument


# ---------------------------------------------------------------------------
# Pairwise primitives
# ---------------------------------------------------------------------------

def jaccard_by_date(dates_a: Sequence[str], dates_b: Sequence[str]) -> float:
    """Jaccard similarity of two activation-date sets (spec Section 6, Matrix A).

    ``J = |A ∩ B| / |A ∪ B|`` with the empty-union case mapped to ``0``.
    """
    set_a = set(dates_a)
    set_b = set(dates_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two equal-length arrays (scipy-free).

    Implemented as the Pearson correlation of the average ranks — identical to
    ``scipy.stats.spearmanr`` for the no-NaN case, including tie handling.
    Returns ``0.0`` when either input is constant (undefined correlation).
    """
    if a.size < 2:
        return 0.0
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    sa = ra.std()
    sb = rb.std()
    if sa == 0 or sb == 0:
        return 0.0
    corr = float(np.corrcoef(ra, rb)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def gain_correlation_by_date(
    doc_a: RuleDocument,
    doc_b: RuleDocument,
    min_active: int = 10,
) -> float:
    """Spearman correlation of two rules' gains, aligned by date (Matrix B).

    The two gain series are placed on the union of their activation dates; bars
    where neither rule traded are dropped, bars where only one traded contribute
    a ``0`` for the other.  When fewer than ``min_active`` aligned bars are
    active the correlation is reported as ``0.0`` (too few points to be
    meaningful — spec Section 6).
    """
    sa = _gain_series(doc_a)
    sb = _gain_series(doc_b)
    if sa.empty or sb.empty:
        return 0.0

    all_dates = sa.index.union(sb.index)
    series_a = sa.reindex(all_dates, fill_value=0.0)
    series_b = sb.reindex(all_dates, fill_value=0.0)

    active = (series_a != 0) | (series_b != 0)
    if int(active.sum()) < min_active:
        return 0.0
    return _spearman(series_a[active].to_numpy(), series_b[active].to_numpy())


def _gain_series(doc: RuleDocument) -> pd.Series:
    """Gain-by-date series for one document (duplicate dates summed)."""
    if not doc.activation_dates:
        return pd.Series(dtype=float)
    s = pd.Series(doc.gains, index=pd.Index(doc.activation_dates, name="date"), dtype=float)
    # A rule cannot open two positions on the same bar, but be defensive about
    # accidental duplicate timestamps so the reindex below never explodes.
    return s.groupby(level=0).sum()


# ---------------------------------------------------------------------------
# Full matrices
# ---------------------------------------------------------------------------

def correlation_matrices(
    docs: List[RuleDocument],
    min_active: int = 10,
    max_pool: Optional[int] = None,
) -> CorrelationMatrices:
    """Compute the Jaccard and Spearman matrices over the whole registry.

    Both matrices are symmetric with a unit diagonal (Jaccard) or self
    correlation of ``1.0`` (Spearman).  Returns empty frames for an empty
    registry.

    Parameters
    ----------
    max_pool : int, optional
        Above this many documents, the ``O(n^2)`` pairwise computation below
        is skipped entirely and identity/zero-off-diagonal matrices are
        returned instead, with ``CorrelationMatrices.skipped=True`` (issue
        #281) — ``None`` (default) never skips, preserving every existing
        caller's behaviour. A run that promotes an unusually large number of
        tradeable rules is a legitimate, data-dependent outcome (e.g. an
        atypical grade distribution driving a much higher promotion rate),
        not a misconfiguration to reject — but the double loop below has no
        cap of its own and no vectorization, so left unchecked it turns into
        an unbounded, unsignalled hang while every other pipeline stage
        finishes in a small fraction of that time. Skipping is conservative:
        an all-zero Jaccard matrix flags no false-positive duplicates, and a
        zero ``gain_corr_max`` is the same reading a rule with no correlated
        peers already gets.
    """
    ids = [d.rule_id for d in docs]
    n = len(docs)
    jac = np.eye(n)
    spr = np.eye(n)

    if max_pool is not None and n > max_pool:
        warnings.warn(
            f"correlation_matrices: {n} tradeable rules exceeds "
            f"max_correlation_pool={max_pool}; skipping the O(n^2) Jaccard/"
            f"Spearman computation ({n * (n - 1) // 2} pairs) to avoid an "
            f"unbounded hang. Duplicate flags and gain_corr_max are not "
            f"meaningful for this run's rules (see RegistryConfig"
            f".max_correlation_pool, issue #281).",
            stacklevel=2,
        )
        jaccard = pd.DataFrame(jac, index=ids, columns=ids)
        spearman = pd.DataFrame(spr, index=ids, columns=ids)
        return CorrelationMatrices(
            rule_ids=ids, jaccard=jaccard, spearman=spearman, skipped=True,
        )

    for i in range(n):
        for j in range(i + 1, n):
            j_ij = jaccard_by_date(docs[i].activation_dates, docs[j].activation_dates)
            s_ij = gain_correlation_by_date(docs[i], docs[j], min_active=min_active)
            jac[i, j] = jac[j, i] = j_ij
            spr[i, j] = spr[j, i] = s_ij

    jaccard = pd.DataFrame(jac, index=ids, columns=ids)
    spearman = pd.DataFrame(spr, index=ids, columns=ids)
    return CorrelationMatrices(rule_ids=ids, jaccard=jaccard, spearman=spearman)


def annotate_correlation_maxima(
    docs: List[RuleDocument],
    matrices: CorrelationMatrices,
) -> None:
    """Populate ``overlap_max`` / ``gain_corr_max`` on every document.

    Each value is the maximum off-diagonal entry of the corresponding matrix row
    — the strongest overlap / gain correlation the rule has with *any other*
    rule in the pool.  With a single rule (no peers) both are ``0.0``.
    """
    ids = matrices.rule_ids
    for i, doc in enumerate(docs):
        if len(ids) <= 1:
            doc.overlap_max = 0.0
            doc.gain_corr_max = 0.0
            continue
        jac_row = matrices.jaccard.iloc[i].drop(labels=doc.rule_id)
        spr_row = matrices.spearman.iloc[i].drop(labels=doc.rule_id)
        doc.overlap_max = round(float(jac_row.max()), 6)
        doc.gain_corr_max = round(float(spr_row.max()), 6)
