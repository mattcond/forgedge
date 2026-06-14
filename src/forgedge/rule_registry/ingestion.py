"""Step 1 — ingestion for the Rule Registry (FORGE Modulo 4).

For every rule emitted by Rule Discovery with an ``EDGE`` or ``PARTIAL-EDGE``
verdict the registry builds a document and appends it to the in-memory list
(spec Section 5).  ``NON-EDGE`` rules never enter the registry.

Building a document re-runs the validated rule's backtest on the source frame to
recover the per-trade ledger — the parallel ``activation_idx`` / ``dates`` /
``gains`` arrays that drive the correlation matrices (Section 4) — and copies
the aggregate evidence Rule Discovery already produced (PF, win rate, DSR,
regime) onto the document.
"""
from __future__ import annotations

import math
import warnings
from typing import List, Optional

import pandas as pd

from ..rule_discovery.backtest import run_backtest
from .models import RegistryConfig, RuleDocument, RuleSubmission

_SIGNAL_COL = "__registry_source_signal__"
# Monthly-concentration score above which a rule is labelled regime "specific".
_REGIME_DEP_SPECIFIC = 0.30


def is_ingestable(submission: RuleSubmission) -> bool:
    """True when the submission carries a tradeable, validated rule (Step 1)."""
    resp = submission.response
    return resp.verdict in ("EDGE", "PARTIAL-EDGE") and resp.validated_rule is not None


def build_document(
    submission: RuleSubmission,
    source_frame: pd.DataFrame,
    rule_id: str,
    config: RegistryConfig,
) -> RuleDocument:
    """Build one registry document from a validated rule (spec Section 4/5).

    Parameters
    ----------
    submission : RuleSubmission
        A tradeable submission (see :func:`is_ingestable`).
    source_frame : pd.DataFrame
        The prepared KPI table of the submission's ticker (sorted, with a
        DatetimeIndex and the timestamp column).
    rule_id : str
        The registry-assigned identifier (e.g. ``"RULE_ADA_01"``).
    config : RegistryConfig
        Column names and session settings.
    """
    resp = submission.response
    vr = resp.validated_rule
    params = vr.params
    candidate = submission.candidate

    # ── replay the validated rule on the source frame for the trade ledger ──
    # An Event Candidate *is* an activation function; the registry evaluates it
    # on the candles it actually observes via the candidate's deterministic
    # replay (:meth:`EventCandidate.apply`).  The pre-computed ``event_series``
    # is a *cache* of that same function on the training candles — reused as a
    # fast path only when its index matches the observed frame's exactly, where
    # the two are bit-for-bit identical.  When the candle sets differ a blind
    # ``reindex`` would force every non-overlapping bar to "inactive" — at worst
    # collapsing the signal to all-zeros and ledgering a rule that never fires —
    # so the event is re-evaluated on the observed frame instead.  This mirrors
    # the no-recompute contract of Alpha and Rule Discovery.
    frame = source_frame.copy()
    stored = candidate.event_series
    if stored is not None and stored.index.equals(frame.index):
        series = stored
    else:
        if stored is not None:
            warnings.warn(
                "Rule Registry received candles whose index differs from the "
                "event's stored activation series; the event is re-evaluated as "
                "an activation function on the observed candles "
                "(EventCandidate.apply).",
                stacklevel=2,
            )
        series = candidate.apply(frame)
    frame[_SIGNAL_COL] = series.fillna(0).to_numpy()

    summary, trades = run_backtest(
        frame, _SIGNAL_COL, params,
        timestamp_col=config.timestamp_col, return_trades=True,
    )

    activation_idx: List[int] = []
    activation_dates: List[str] = []
    gains: List[float] = []
    monthly_gains: List[float] = []
    if not trades.empty:
        activation_idx = [int(x) for x in trades["fill_rn"].tolist()]
        fill_dt = pd.to_datetime(trades["fill_dt"])
        activation_dates = [pd.Timestamp(x).isoformat() for x in fill_dt]
        gains = [round(float(x), 8) for x in trades["net_pct_gain"].tolist()]
        monthly = trades.assign(_m=fill_dt.dt.to_period("M")).groupby("_m")[
            "net_pct_gain"
        ].sum()
        monthly_gains = [round(float(x), 6) for x in monthly.tolist()]

    # ── source statistics (mostly carried over from Rule Discovery) ─────────
    s = resp.in_sample_summary
    sv = resp.statistical_validation
    dsr = float(sv.deflated_sharpe) if sv is not None else float("nan")
    stats = {
        "pf": _num(s.profit_factor),
        "win_rate": _num(s.win_rate_pct),
        "total_trades": int(s.total_trades),
        "expectancy": _num(s.expectancy),
        "zero_months": int(s.zero_months),
        "dsr": _num(dsr),
        "grade": submission.grade or _derive_grade(resp),
        "monthly_gains": monthly_gains,
    }

    # ── regime ──────────────────────────────────────────────────────────────
    ra = resp.regime_analysis
    avoid_in = list(ra.avoid_in) if ra is not None else []
    dep = float(ra.dependency_score) if ra is not None else float("nan")
    regime = {"type": _regime_type(avoid_in, dep), "avoid_in": avoid_in}

    return RuleDocument(
        rule_id=rule_id,
        expression=vr.expression,
        source_ticker=submission.ticker,
        source_alpha_id=resp.alpha_id,
        verdict=resp.verdict,
        grade=str(stats["grade"]),
        activation_idx=activation_idx,
        activation_dates=activation_dates,
        gains=gains,
        params=_params_dict(params),
        stats=stats,
        regime=regime,
        _candidate=candidate,
        _bt_params=params,
        _trades=trades,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_dict(params) -> dict:
    """Flatten the operational parameters for the document / flat table."""
    return {
        "direction": params.direction,
        "entry_mode": params.buy_type,
        "buy_drop_pct": params.buy_drop_pct,
        "buy_delay_bar": params.buy_delay_bar,
        "sell_pct": params.sell_pct,
        "target_h": params.target_h,
        "fee": params.fee,
    }


def _regime_type(avoid_in: List[str], dependency_score: float) -> str:
    """Derive the regime ``type`` from the Rule Discovery regime breakdown.

    Rule Discovery reports the regimes to avoid and a monthly-concentration
    dependency score, but not the ``agnostic``/``conditional``/``specific``
    label of the spec — the registry derives it: any regime to avoid makes the
    rule ``conditional``; a high concentration with nothing to avoid makes it
    ``specific``; otherwise it is ``agnostic``.
    """
    if avoid_in:
        return "conditional"
    if math.isfinite(dependency_score) and dependency_score > _REGIME_DEP_SPECIFIC:
        return "specific"
    return "agnostic"


def _derive_grade(resp) -> str:
    """Registry-side letter grade from the Rule Discovery evidence.

    Used only when the submission does not carry an upstream grade.  ``EDGE`` is
    an ``A`` when both the profit factor and the Deflated Sharpe are strong,
    otherwise ``B+``; ``PARTIAL-EDGE`` is a ``B``; anything else a ``C``.
    """
    pf = float(resp.in_sample_summary.profit_factor)
    sv = resp.statistical_validation
    dsr = float(sv.deflated_sharpe) if sv is not None else float("nan")
    if resp.verdict == "EDGE":
        if math.isfinite(dsr) and dsr >= 1.5 and math.isfinite(pf) and pf >= 3.0:
            return "A"
        return "B+"
    if resp.verdict == "PARTIAL-EDGE":
        return "B"
    return "C"


def _num(x) -> float:
    try:
        xf = float(x)
        return xf
    except (TypeError, ValueError):
        return float("nan")
