"""Rule performance report — replay a set of rules on arbitrary candles, to HTML.

Takes a list of rules (each one the pair *activation function + order
mechanics* that FORGE publishes: an :class:`EventCandidate` plus the validated
:class:`BacktestParams`) and a candle table, replays every rule
deterministically on those candles and renders a self-contained HTML report
(inline SVG, no external assets — same conventions as the Rule Discovery and
Registry reports).

This is the *monitoring* companion of the pipeline: the candles do not need to
be the discovery table.  Feed fresh data and the report shows how the
published rules are behaving on it — including whether a signal is active on
the last bar.

Per rule the report contains:

* stat tiles — replay expectancy side by side with the walk-forward OOS
  expectancy when known, PF, win rate, trades/signals, average concurrent
  positions;
* equity vs buy & hold on a **dual axis** (rule on the left, benchmark on the
  right, independent scales — the caption says so) with a drawdown strip and
  the in-sample boundary marked;
* monthly activation trend — in-sample months in gray, out-of-sample in blue,
  light bar = signals, dark part = executed trades;
* gain/loss distribution with win/loss statistics;
* return distributions — low/close/high, each the **forward** return over
  the rule's own holding horizon (``target_h`` bars ahead of the signal bar,
  anchored on that bar's close, oriented by direction — the same quantity
  Alpha Discovery derives the target from and Rule Discovery trades), base
  vs event, as Gaussian KDEs (pure numpy, Silverman bandwidth);
* MAE → final net scatter (intra-trade risk of stop-less rules);
* rolling expectancy (wall-clock window, edge-decay detector);
* per-regime performance (when a ``regime`` column is present or computable);
* the most recent trades and a *signal active now* badge.

Usage
-----
    from forgedge import RuleSpec, rule_performance_report

    specs = RuleSpec.from_forge_result(result)        # or build explicitly
    html = rule_performance_report(specs, fresh_candles)
    open("rules.html", "w").write(html)

Honesty notes baked into the report: bars before the rule's ``is_end``
(auto-filled from the session TimeBudget by ``from_forge_result``) are
rendered as in-sample and the caption says the replay there is descriptive;
the equity is the *arithmetic* sum of per-trade net gains (the backtest opens
overlapping positions and models no portfolio sizing, so compounding would be
fiction); the dual axis compares shapes, not magnitudes.
"""
from __future__ import annotations

import html as _html
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .event_discovery.models import EventCandidate
from .market_context.models import DEFAULT_LABELS, REGIME_COL
from .rule_discovery.backtest import run_backtest
from .rule_discovery.models import BacktestParams

__all__ = ["RuleSpec", "rule_performance_report"]

# Display order: bull side first, mirroring the regime report convention.
_REGIME_ORDER = list(reversed(DEFAULT_LABELS))

# Chart geometry (SVG user units; the element scales responsively).
_W, _PL, _PR, _PT, _PB = 1080, 64, 150, 16, 26


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------

@dataclass
class RuleSpec:
    """One rule to replay: activation function + validated order mechanics.

    Attributes
    ----------
    name : str
        Display name (e.g. the registry ``rule_id``).
    candidate : EventCandidate
        The event's deterministic activation function
        (:meth:`EventCandidate.apply`; the cached ``event_series`` is used
        verbatim when the candle index is identical).
    params : BacktestParams
        The published operating point (direction, limit offset, take-profit,
        horizon, fee).
    is_end : pd.Timestamp or None
        End of the discovery in-sample period.  Bars up to (and including)
        this timestamp are rendered as IS; ``None`` marks everything as OOS
        (the natural state when the candles are fresh data).
    verdict : str or None
        Pipeline verdict, shown in the header when known.
    oos_expectancy : float or None
        Walk-forward OOS expectancy per trade from the discovery session,
        shown next to the replay expectancy when known.
    """

    name: str
    candidate: EventCandidate
    params: BacktestParams
    is_end: Optional[pd.Timestamp] = None
    verdict: Optional[str] = None
    oos_expectancy: Optional[float] = None

    # ------------------------------------------------------------------

    @classmethod
    def from_forge_result(cls, result) -> List["RuleSpec"]:
        """One spec per tradeable rule of a :class:`~forgedge.forge.ForgeResult`.

        Uses the registry ``rule_id`` as the name when the registry ran (the
        ``alpha_id`` otherwise), and fills ``is_end`` from the session
        :class:`~forgedge.timebudget.TimeBudget`.
        """
        by_id = {c.event_id: c for c in result.candidates}
        names: Dict[str, str] = {}
        if result.registry is not None and result.registry.documents:
            ft = result.registry.flat_table()
            if "source_alpha_id" in ft.columns:
                names = dict(zip(ft["source_alpha_id"], ft["rule_id"]))
        is_end = None
        if result.time_budget is not None and result.event_frame is not None:
            split = result.time_budget.split
            if 0 < split <= len(result.event_frame):
                is_end = pd.Timestamp(result.event_frame.index[split - 1])
        specs = []
        for contract, resp in result.rule_responses:
            if not resp.is_edge or resp.validated_rule is None:
                continue
            cand = by_id.get(contract.event_candidate_id)
            if cand is None:
                continue
            oos = None
            if resp.walk_forward is not None:
                oos = resp.walk_forward.oos_summary.expectancy
            specs.append(cls(
                name=names.get(contract.alpha_id, contract.alpha_id),
                candidate=cand,
                params=resp.validated_rule.params,
                is_end=is_end,
                verdict=resp.verdict,
                oos_expectancy=oos,
            ))
        return specs

    @classmethod
    def from_registry(cls, registry) -> List["RuleSpec"]:
        """One spec per registry document that still carries its candidate.

        Documents whose private ``_candidate`` was stripped (e.g. after a
        round-trip through a serialisation that drops it) are skipped.
        """
        specs = []
        param_fields = set(BacktestParams.__dataclass_fields__)
        for doc in registry.documents:
            cand = getattr(doc, "_candidate", None)
            if cand is None:
                continue
            params = BacktestParams(
                **{k: v for k, v in dict(doc.params).items() if k in param_fields}
            )
            specs.append(cls(name=doc.rule_id, candidate=cand, params=params,
                             verdict=doc.verdict))
        return specs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rule_performance_report(
    rules: Union[Sequence[RuleSpec], "object"],
    candles: pd.DataFrame,
    *,
    title: str = "Rule performance report",
    compute_regime: bool = True,
    rolling_window: str = "90D",
    rolling_min_trades: int = 5,
    recent_trades: int = 8,
    timestamp_col: str = "open_dt",
) -> str:
    """Replay ``rules`` on ``candles`` and render the HTML report.

    Parameters
    ----------
    rules : sequence of RuleSpec, or a ForgeResult
        The rules to replay.  A :class:`~forgedge.forge.ForgeResult` is
        accepted directly (its tradeable rules are extracted via
        :meth:`RuleSpec.from_forge_result`).
    candles : pd.DataFrame
        OHLC table with a ``timestamp_col`` column or a DatetimeIndex,
        chronologically sorted, plus every feature column the rules'
        expressions reference.  A rule whose features are missing is reported
        as unavailable instead of failing the whole report.
    title : str
        Report title.
    compute_regime : bool
        When ``True`` (default) and the candles carry no ``regime`` column,
        run Market Context on them for the per-regime section.  Set ``False``
        (or let the computation fail) to skip the section.
    rolling_window, rolling_min_trades
        Wall-clock window and minimum trade count of the rolling-expectancy
        chart.
    recent_trades : int
        Rows of the "recent trades" table.
    timestamp_col : str
        Datetime column name on the candles.

    Returns
    -------
    str
        A self-contained HTML document (inline CSS/SVG, no external assets).
    """
    if hasattr(rules, "rule_responses"):  # a ForgeResult
        rules = RuleSpec.from_forge_result(rules)
    rules = list(rules)

    frame = _prepare_candles(candles, timestamp_col)
    dts = pd.DatetimeIndex(frame[timestamp_col])
    dates = [str(d.date()) for d in dts]
    n = len(frame)
    close = frame["close"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    high = frame["high"].to_numpy(float)
    bh = (close / close[0] - 1.0) * 100.0

    regime_arr = _regime_series(frame, compute_regime, timestamp_col)

    computed, failed = [], []
    for spec in rules:
        try:
            computed.append(_compute_rule(spec, frame, dts, timestamp_col))
        except Exception as exc:  # missing features, degenerate data, …
            failed.append((spec, f"{type(exc).__name__}: {exc}"))

    sections = [_overview(computed, failed, dates, n, title)]
    for R in computed:
        sections.append(_rule_section(
            R, bh, dates, dts, regime_arr, close, low, high,
            rolling_window, rolling_min_trades, recent_trades,
        ))
    sections.append(_footer_note(computed, dates))

    js = _hover_js(computed, bh, dates)
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        + "".join(sections)
        + f"<div id='tip'></div><script>{js}</script></body></html>"
    )


# ---------------------------------------------------------------------------
# Data preparation / per-rule computation
# ---------------------------------------------------------------------------

def _prepare_candles(candles: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    frame = candles.copy()
    if timestamp_col not in frame.columns:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame[timestamp_col] = frame.index
        else:
            raise ValueError(
                f"candles need a {timestamp_col!r} column or a DatetimeIndex"
            )
    missing = [c for c in ("open", "high", "low", "close") if c not in frame.columns]
    if missing:
        raise ValueError(f"candles are missing price columns: {missing}")
    return frame


def _regime_series(frame, compute_regime, timestamp_col) -> Optional[np.ndarray]:
    if REGIME_COL in frame.columns:
        return frame[REGIME_COL].astype(str).to_numpy()
    if not compute_regime:
        return None
    try:
        from .market_context.context import MarketContext

        enriched = MarketContext(frame.drop(columns=["__signal__"], errors="ignore")).run()
        return enriched[REGIME_COL].astype(str).to_numpy()
    except Exception:
        return None


def _signal_series(spec: RuleSpec, frame: pd.DataFrame) -> pd.Series:
    """The rule's activation series on these candles.

    The candidate's cached ``event_series`` is reused only when the frame has
    a matching **DatetimeIndex** — a meaningful identity.  A plain RangeIndex
    matches any same-length frame, so relying on it would silently replay a
    stale cache on unrelated candles; in every other case the event is
    re-evaluated as an activation function (:meth:`EventCandidate.apply`).
    """
    stored = spec.candidate.event_series
    idx = frame.index
    if (
        stored is not None
        and isinstance(idx, pd.DatetimeIndex)
        and stored.index.equals(idx)
    ):
        sig = stored
    else:
        sig = spec.candidate.apply(frame)
    return pd.Series(sig.fillna(0).to_numpy(), index=idx)


def _compute_rule(spec: RuleSpec, frame, dts, timestamp_col) -> dict:
    n = len(frame)
    p = spec.params
    sig = _signal_series(spec, frame)
    fr2 = frame.copy()
    fr2["__signal__"] = sig.to_numpy()
    summary, trades = run_backtest(
        fr2, "__signal__", p, timestamp_col=timestamp_col, return_trades=True
    )

    net = trades["net_pct_gain"].to_numpy(float) * 100.0
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)

    eq = np.zeros(n)
    mae, mfe = [], []
    for _, t in trades.iterrows():
        eq[int(t.exit_rn):] += t.net_pct_gain * 100.0
        f_, e_, bp = int(t.fill_rn), int(t.exit_rn), float(t.buy_price)
        hi_w, lo_w = high[f_:e_ + 1].max(), low[f_:e_ + 1].min()
        if p.direction == "short":
            mae.append((hi_w - bp) / bp * 100.0)
            mfe.append((bp - lo_w) / bp * 100.0)
        else:
            mae.append((bp - lo_w) / bp * 100.0)
            mfe.append((hi_w - bp) / bp * 100.0)
    trades = trades.assign(mae=mae, mfe=mfe)
    dd = eq - np.maximum.accumulate(eq)

    sig_idx = np.where(sig.to_numpy() > 0)[0]
    last_sig = int(sig_idx[-1]) if sig_idx.size else None

    split = None
    if spec.is_end is not None:
        pos = int(dts.searchsorted(pd.Timestamp(spec.is_end), side="right"))
        if 0 < pos < n:
            split = pos  # first OOS bar

    # Month grouping on the timestamp column (the frame index may be a plain
    # RangeIndex when the timestamps live in a column).
    sig_dt = pd.Series(sig.to_numpy(), index=dts)
    ms = sig_dt[sig_dt > 0].groupby(sig_dt[sig_dt > 0].index.to_period("M")).count()
    mt = pd.to_datetime(trades["fill_dt"]).dt.to_period("M").value_counts()
    months = [
        (str(m), int(ms.get(m, 0)), int(mt.get(m, 0)))
        for m in pd.period_range(dts[0], dts[-1], freq="M")
    ]

    exposure = (
        float(sum(int(t.exit_rn) - int(t.fill_rn) for _, t in trades.iterrows())) / n
        if n else 0.0
    )
    return dict(
        spec=spec, uid=re.sub(r"[^A-Za-z0-9_-]", "_", spec.name),
        summary=summary, trades=trades, net=net, eq=eq, dd=dd,
        months=months, split=split, last_sig=last_sig,
        active_now=last_sig == n - 1 if last_sig is not None else False,
        exposure=exposure,
        # Boolean activation mask (the event condition, not fill/execution) —
        # feeds the return-distribution section's "event" selection.
        active=sig.to_numpy(dtype=bool),
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _esc(s) -> str:
    return _html.escape(str(s))


def _pct(x, digits=2) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:+.{digits}f}%"


def _num(x, fmt="{:.2f}") -> str:
    return "–" if x is None or not np.isfinite(x) else fmt.format(x)


def _pctval(x, digits=3) -> str:
    """Format a value already expressed in percent (not a fraction)."""
    return "–" if x is None or not np.isfinite(x) else f"{x:+.{digits}f}%"


def _sx(i: int, n: int) -> float:
    return _PL + (_W - _PL - _PR) * (i / max(n - 1, 1))


# ---------------------------------------------------------------------------
# SVG builders (ported from the approved prototype)
# ---------------------------------------------------------------------------

def _equity_svg(R, bh, dates, split) -> str:
    eq, dd, n = R["eq"], R["dd"], len(bh)
    HT = 320
    loL = min(eq.min(), 0.0) * 1.1 - 30
    hiL = max(eq.max() * 1.06, 10.0)
    loR, hiR = min(bh.min() * 1.1, -1.0), max(bh.max() * 1.3, 5.0)
    syL = lambda v: _PT + (HT - _PT - _PB) * (1 - (v - loL) / (hiL - loL))
    syR = lambda v: _PT + (HT - _PT - _PB) * (1 - (v - loR) / (hiR - loR))
    pts_eq = " ".join(f"{_sx(i, n):.1f},{syL(v):.1f}" for i, v in enumerate(eq))
    pts_bh = " ".join(f"{_sx(i, n):.1f},{syR(v):.1f}" for i, v in enumerate(bh))

    span = hiL - loL
    step = max(10.0, round(span / 4 / 10) * 10)
    grid = ticks = ""
    v = 0.0
    while v < hiL:
        grid += f'<line x1="{_PL}" x2="{_W-_PR}" y1="{syL(v):.0f}" y2="{syL(v):.0f}" class="grid"/>'
        ticks += f'<text x="{_PL-8}" y="{syL(v)+4:.0f}" class="tick tickeq" text-anchor="end">{v:+.0f}%</text>'
        v += step
    tickR = "".join(
        f'<text x="{_W-_PR+8}" y="{syR(v)+4:.0f}" class="tick tickbh" text-anchor="start">{v:+d}%</text>'
        for v in (-75, -50, -25, 0, 25, 50, 100) if loR <= v <= hiR
    )
    xstep = max(n // 6, 1)
    xt = "".join(
        f'<text x="{_sx(i, n):.0f}" y="{HT-6}" class="tick" text-anchor="middle">{dates[i][:7]}</text>'
        for i in range(0, n, xstep)
    )
    split_mark = ""
    if split is not None:
        spx = _sx(split, n)
        split_mark = (
            f'<line x1="{spx:.0f}" x2="{spx:.0f}" y1="{_PT}" y2="{HT-_PB}" class="splitl"/>'
            f'<text x="{spx+5:.0f}" y="{_PT+12}" class="tick splitt">IS end · {dates[split-1]}</text>'
        )
    H2 = 90
    dmin = min(dd.min() * 1.1, -1.0)
    sy2 = lambda v: _PT + (H2 - _PT - 10) * (1 - (v - dmin) / (1 - dmin))
    pts_dd = " ".join(f"{_sx(i, n):.1f},{sy2(v):.1f}" for i, v in enumerate(dd))
    uid = R["uid"]
    return f"""<svg viewBox="0 0 {_W} {HT}" class="eqchart" id="eq_{uid}">
{grid}{ticks}{tickR}{xt}
<text x="{_PL-8}" y="{_PT-2}" class="tick tickeq" text-anchor="end">Rule</text>
<text x="{_W-_PR+8}" y="{_PT-2}" class="tick tickbh" text-anchor="start">B&amp;H</text>
{split_mark}
<polyline points="{pts_bh}" class="bhline"/>
<polyline points="{pts_eq}" class="eqline"/>
<text x="{_W-_PR+8}" y="{syL(eq[-1])+4:.0f}" class="lab eqlab">Rule {eq[-1]:+.0f}%</text>
<text x="{_W-_PR+8}" y="{syR(bh[-1])+16:.0f}" class="lab bhlab">B&amp;H {bh[-1]:+.0f}%</text>
<line id="cross_{uid}" x1="0" x2="0" y1="{_PT}" y2="{HT-_PB}" class="cross" visibility="hidden"/>
<rect x="{_PL}" y="{_PT}" width="{_W-_PL-_PR}" height="{HT-_PT-_PB}" fill="transparent" class="eqhover" data-uid="{uid}"/>
</svg>
<svg viewBox="0 0 {_W} {H2}">
<line x1="{_PL}" x2="{_W-_PR}" y1="{sy2(0):.0f}" y2="{sy2(0):.0f}" class="zero"/>
<text x="{_PL-8}" y="{sy2(dd.min())+4:.0f}" class="tick" text-anchor="end">{dd.min():.0f} pt</text>
<polygon points="{_PL},{sy2(0):.0f} {pts_dd} {_W-_PR},{sy2(0):.0f}" class="ddarea"/>
<text x="{_W-_PR+8}" y="{sy2(dd.min()/2)+4:.0f}" class="lab ddlab">drawdown</text>
</svg>"""


def _monthly_svg(R, dates) -> str:
    months, split = R["months"], R["split"]
    split_month = dates[split - 1][:7] if split is not None else None
    mn = len(months)
    BW = (_W - _PL - _PR) / max(mn, 1)
    mmax = max((s for _, s, _ in months), default=0) or 1
    HT = 180
    my = lambda v: _PT + (HT - _PT - _PB) * (1 - v / mmax)
    bars = ""
    for k, (m, s, t) in enumerate(months):
        is_is = split_month is not None and m <= split_month
        c_sig, c_trd = ("gsig", "gtrd") if is_is else ("msig", "mtrd")
        x = _PL + k * BW
        if s:
            bars += (
                f'<rect x="{x+2:.1f}" y="{my(s):.1f}" width="{max(BW-4,1):.1f}" height="{my(0)-my(s):.1f}" rx="3" class="{c_sig}">'
                f'<title>{m} ({"IS" if is_is else "OOS"}): {s} signals, {t} executed</title></rect>'
            )
            if t:
                bars += f'<rect x="{x+2:.1f}" y="{my(t):.1f}" width="{max(BW-4,1):.1f}" height="{my(0)-my(t):.1f}" rx="3" class="{c_trd}"><title>{m}: {t} executed</title></rect>'
            bars += f'<text x="{x+BW/2:.1f}" y="{my(s)-4:.0f}" class="tick" text-anchor="middle">{s}</text>'
    if split_month is not None:
        k_split = next((k for k, (m, _, _) in enumerate(months) if m > split_month), None)
        if k_split:
            xsp = _PL + k_split * BW - 1
            bars += (
                f'<line x1="{xsp:.0f}" x2="{xsp:.0f}" y1="{_PT}" y2="{HT-_PB}" class="splitl"/>'
                f'<text x="{xsp+5:.0f}" y="{_PT+10}" class="tick splitt">OOS →</text>'
            )
    lab_step = max(mn // 10, 1)
    xt = "".join(
        f'<text x="{_PL+k*BW+BW/2:.0f}" y="{HT-6}" class="tick" text-anchor="middle">{m[2:7]}</text>'
        for k, (m, _, _) in enumerate(months) if k % lab_step == 0
    )
    return f'<svg viewBox="0 0 {_W} {HT}"><line x1="{_PL}" x2="{_W-_PR}" y1="{my(0):.0f}" y2="{my(0):.0f}" class="zero"/>{bars}{xt}</svg>'


def _hist_svg(R) -> str:
    net = R["net"]
    if net.size == 0:
        return '<p class="sub">No executed trades.</p>'
    span = max(5.0, float(np.ceil(max(abs(net.min()), abs(net.max())) / 5) * 5))
    hist, edges = np.histogram(net, bins=np.linspace(-span, span, 21))
    hmax = hist.max() or 1
    HT, nb = 180, len(hist)
    BW = (_W - _PL - _PR) / nb
    hy = lambda v: _PT + (HT - _PT - _PB) * (1 - v / hmax)
    bars = ""
    for k, c in enumerate(hist):
        if c == 0:
            continue
        mid = (edges[k] + edges[k + 1]) / 2
        cls = "hpos" if mid > 0 else ("hneg" if mid < 0 else "hzero")
        x = _PL + k * BW
        bars += (
            f'<rect x="{x+2:.1f}" y="{hy(c):.1f}" width="{BW-4:.1f}" height="{hy(0)-hy(c):.1f}" rx="3" class="{cls}">'
            f'<title>{edges[k]:.1f}% … {edges[k+1]:.1f}%: {c} trades</title></rect>'
            f'<text x="{x+BW/2:.1f}" y="{hy(c)-4:.0f}" class="tick" text-anchor="middle">{c}</text>'
        )
    xt = "".join(
        f'<text x="{_PL+k*BW+BW/2:.0f}" y="{HT-6}" class="tick" text-anchor="middle">{edges[k]:.0f}</text>'
        for k in range(0, nb, 2)
    )
    zx = _PL + (0 - edges[0]) / (edges[-1] - edges[0]) * (_W - _PL - _PR)
    return (
        f'<svg viewBox="0 0 {_W} {HT}"><line x1="{_PL}" x2="{_W-_PR}" y1="{hy(0):.0f}" y2="{hy(0):.0f}" class="zero"/>'
        f'<line x1="{zx:.0f}" x2="{zx:.0f}" y1="{_PT}" y2="{HT-_PB}" class="grid"/>{bars}{xt}</svg>'
    )


def _silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for a 1-D Gaussian KDE.

    ``h = 0.9 * min(std, IQR/1.34) * n^(-1/5)`` — the standard first-pass
    bandwidth, robust to outliers (uses the smaller of the standard
    deviation and the IQR-based scale estimate, so a few extreme returns
    cannot blow up the smoothing width).
    """
    n = x.size
    if n < 2:
        return 1.0
    std = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    if not np.isfinite(scale) or scale <= 0:
        scale = std if std > 0 else 1.0
    h = 0.9 * scale * n ** (-1.0 / 5.0)
    return h if np.isfinite(h) and h > 0 else 1e-6


def _gaussian_kde(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE of ``x`` evaluated on ``grid`` — pure numpy, no scipy.

    ``f(g) = (1/(n·h)) Σᵢ φ((g − xᵢ)/h)`` with ``φ`` the standard normal
    density and ``h`` from :func:`_silverman_bandwidth`.  Each group (base or
    event) gets its own bandwidth, so both curves are honestly smoothed for
    their own sample size; both integrate to 1, so the overlay compares
    shapes regardless of how different the two sample sizes are.
    """
    if x.size == 0:
        return np.zeros_like(grid)
    h = _silverman_bandwidth(x)
    u = (grid[:, None] - x[None, :]) / h
    k = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
    return k.sum(axis=1) / (x.size * h)


def _return_dist_svg(base_ret: np.ndarray, event_ret: np.ndarray, fill_cls: str, mean_cls: str) -> str:
    """Base (gray) vs event (colored) KDE overlay of one bar return series.

    Both curves are proper Gaussian kernel density estimates (each on its own
    Silverman bandwidth), so the overlay is comparable in shape regardless of
    how many event bars exist relative to the base sample. The grid spans the
    base distribution's 1st/99th percentile — the base sample is always the
    larger, more representative one — with a margin so the tails read cleanly.
    """
    HT = 190
    if base_ret.size == 0:
        return '<p class="sub">No data.</p>'
    p1, p99 = np.nanpercentile(base_ret, [1, 99])
    span = max(abs(p1), abs(p99), 0.5) * 1.15
    grid = np.linspace(-span, span, 240)
    dens_b = _gaussian_kde(base_ret, grid)
    dens_e = _gaussian_kde(event_ret, grid)
    dmax = max(float(dens_b.max()), float(dens_e.max()), 1e-9) * 1.08
    xs = lambda v: _PL + (v - grid[0]) / (grid[-1] - grid[0]) * (_W - _PL - _PR)
    ys = lambda v: _PT + (HT - _PT - _PB) * (1 - v / dmax)

    base_line = " ".join(f"{xs(g):.1f},{ys(d):.1f}" for g, d in zip(grid, dens_b))
    base_area = f"{xs(grid[0]):.1f},{ys(0):.1f} {base_line} {xs(grid[-1]):.1f},{ys(0):.1f}"
    curves = (
        f'<polygon points="{base_area}" class="distbasearea"/>'
        f'<polyline points="{base_line}" class="distbaseline"/>'
    )
    if event_ret.size:
        ev_line = " ".join(f"{xs(g):.1f},{ys(d):.1f}" for g, d in zip(grid, dens_e))
        ev_area = f"{xs(grid[0]):.1f},{ys(0):.1f} {ev_line} {xs(grid[-1]):.1f},{ys(0):.1f}"
        curves += (
            f'<polygon points="{ev_area}" class="{fill_cls}area"/>'
            f'<polyline points="{ev_line}" class="{fill_cls}line"/>'
        )

    xt = "".join(
        f'<text x="{xs(v):.0f}" y="{HT-6}" class="tick" text-anchor="middle">{v:.1f}</text>'
        for v in np.linspace(-span, span, 7)
    )
    zx = xs(0.0)
    marks = f'<line x1="{xs(float(np.mean(base_ret))):.1f}" x2="{xs(float(np.mean(base_ret))):.1f}" y1="{_PT}" y2="{HT-_PB}" class="meanbase"/>'
    if event_ret.size:
        me = float(np.mean(event_ret))
        marks += f'<line x1="{xs(me):.1f}" x2="{xs(me):.1f}" y1="{_PT}" y2="{HT-_PB}" class="{mean_cls}"/>'
    return (
        f'<svg viewBox="0 0 {_W} {HT}">'
        f'<line x1="{_PL}" x2="{_W-_PR}" y1="{ys(0):.0f}" y2="{ys(0):.0f}" class="zero"/>'
        f'<line x1="{zx:.0f}" x2="{zx:.0f}" y1="{_PT}" y2="{HT-_PB}" class="grid"/>'
        f"{curves}{marks}{xt}</svg>"
    )


def _forward_oriented_return(anchor: np.ndarray, future: np.ndarray, direction: str, h: int) -> np.ndarray:
    """Forward return over ``h`` bars, oriented so "favourable" is positive.

    ``(future[t+h] / anchor[t] - 1)``, sign-flipped for ``direction="short"``
    — the same convention Alpha Discovery uses to derive the target. ``anchor``
    is the entry-referenced price (``close``, the level the backtest's limit
    offset and exit economics are anchored to); ``future`` is whichever price
    is read at ``t+h`` — passing ``close`` for both reproduces the contract's
    own derived-target return, while ``low``/``high`` as ``future`` read the
    boundary price at the *same* forward horizon, not a same-bar return. The
    last ``h`` bars have no complete forward window and are ``nan``.
    """
    n = anchor.size
    fwd = np.full(n, np.nan)
    if h > 0 and h < n:
        fwd[: n - h] = future[h:] / anchor[: n - h] - 1.0
    return (-1.0 if direction == "short" else 1.0) * fwd * 100.0


def _return_distribution_section(R, close, low, high) -> str:
    """Low/close/high return distributions, base vs event — all measured as
    the **forward** return over the rule's own holding horizon (``target_h``
    bars ahead of the signal bar, anchored on that bar's close, oriented by
    direction), never the signal bar's own same-bar return: the tradeable
    edge lives in what happens *after* the event, and every series here must
    share that one horizon to be comparable to the verdict."""
    active = R["active"]
    p = R["spec"].params
    h = int(p.target_h)

    series = [
        ("Low", low, "distlow", "meanlow"),
        ("Close", close, "distclose", "meanclose"),
        ("High", high, "disthigh", "meanhigh"),
    ]
    charts, stat_rows = "", ""
    for label, future, fill_cls, mean_cls in series:
        ret = _forward_oriented_return(close, future, p.direction, h)
        base_mask = np.isfinite(ret)
        event_mask = base_mask & active
        base_ret, event_ret = ret[base_mask], ret[event_mask]
        charts += f'<h3>{label}</h3>{_return_dist_svg(base_ret, event_ret, fill_cls, mean_cls)}'
        mean_b = float(np.mean(base_ret)) if base_ret.size else float("nan")
        mean_e = float(np.mean(event_ret)) if event_ret.size else float("nan")
        std_b = float(np.std(base_ret)) if base_ret.size else float("nan")
        std_e = float(np.std(event_ret)) if event_ret.size else float("nan")
        delta = mean_e - mean_b if np.isfinite(mean_e) and np.isfinite(mean_b) else float("nan")
        delta_cls = "" if not np.isfinite(delta) else ("pos" if delta > 0 else "neg")
        stat_rows += (
            f'<tr><td class="rl">{label}</td><td>{base_ret.size}</td><td>{event_ret.size}</td>'
            f'<td>{_pctval(mean_b)}</td><td>{_pctval(mean_e)}</td>'
            f'<td class="{delta_cls}">{_pctval(delta)}</td>'
            f'<td>{_pctval(std_b)}</td><td>{_pctval(std_e)}</td></tr>'
        )
    return f"""
<h2>Return distributions — base vs event, forward @ {h} bars</h2>
<p class="sub">Low / close / high, each read <b>{h} bars ahead of the signal bar</b> and expressed as a return
off that bar's close, oriented so favourable is positive ({p.direction} flips the sign) — shown as a Gaussian
KDE: unconditional (gray, "base") vs restricted to bars where the event is active (colored). This is the same
forward window Alpha Discovery derives the target from and Rule Discovery trades: a rightward shift of the
event curve — in any of the three — is part of the tradeable edge. Close is the level the target/exit economics
are anchored to; low and high show how the boundary prices behaved over that same horizon (e.g. an edge that
shows up on close but not on high can indicate the move stalls before a deeper favourable excursion is
reached). None of the three measures the signal bar's own same-bar return — that is a different, non-predictive
diagnostic (does the event's own bar merely have an unusual shape?) that a real forward edge is not expected
to leave any trace in.</p>
{charts}
<div class="legend"><span><i style="background:var(--distbase)"></i>Base (all bars)</span>
<span><i style="background:var(--red)"></i>Low @ event</span>
<span><i style="background:var(--ochre)"></i>Close @ event</span>
<span><i style="background:var(--green)"></i>High @ event</span></div>
<table><tr><th>series</th><th>n base</th><th>n event</th><th>mean base</th><th>mean event</th><th>Δ mean</th><th>std base</th><th>std event</th></tr>
{stat_rows}</table>"""


def _mae_svg(R, dates) -> str:
    t, net = R["trades"], R["net"]
    if net.size == 0:
        return ""
    mae = t["mae"].to_numpy(float)
    HT = 260
    xmax = max(float(mae.max()) * 1.1, 5.0)
    ylo = min(float(net.min()) * 1.15, -5.0)
    yhi = max(float(net.max()) * 1.15, 5.0)
    xs = lambda v: _PL + (_W - _PL - _PR) * v / xmax
    ys = lambda v: _PT + (HT - _PT - _PB) * (1 - (v - ylo) / (yhi - ylo))
    gstep = max(5.0, round(xmax / 6 / 5) * 5)
    grid = "".join(
        f'<line x1="{xs(v):.0f}" x2="{xs(v):.0f}" y1="{_PT}" y2="{HT-_PB}" class="grid"/>'
        f'<text x="{xs(v):.0f}" y="{HT-6}" class="tick" text-anchor="middle">{v:.0f}%</text>'
        for v in np.arange(0, xmax, gstep)
    )
    grid += f'<line x1="{_PL}" x2="{_W-_PR}" y1="{ys(0):.0f}" y2="{ys(0):.0f}" class="zero"/>'
    grid += "".join(
        f'<text x="{_PL-8}" y="{ys(v)+4:.0f}" class="tick" text-anchor="end">{v:+.0f}%</text>'
        for v in (round(ylo / 10) * 10, 0, round(yhi / 10) * 10) if ylo <= v <= yhi
    )
    dots = "".join(
        f'<circle cx="{xs(m):.1f}" cy="{ys(g):.1f}" r="4" class="{"dw" if g > 0 else "dl"}">'
        f'<title>MAE {m:.1f}% → net {g:+.1f}% (signal {dates[int(row.signal_rn)]})</title></circle>'
        for (m, g, (_, row)) in zip(mae, net, t.iterrows())
    )
    return (
        f'<svg viewBox="0 0 {_W} {HT}">{grid}{dots}'
        f'<text x="{_W-_PR+8}" y="{_PT+10}" class="tick">y: final net</text>'
        f'<text x="{_W-_PR+8}" y="{_PT+24}" class="tick">x: max adverse excursion</text></svg>'
    )


def _rolling_svg(R, dts, dates, window, min_trades) -> str:
    trades, net, split = R["trades"], R["net"], R["split"]
    n = len(dates)
    if net.size == 0:
        return ""
    ex = pd.Series(net, index=pd.to_datetime(trades["exit_dt"]).values).sort_index()
    roll = ex.rolling(window).mean().where(ex.rolling(window).count() >= min_trades)
    HT = 160
    vals = roll.to_numpy(float)
    if not np.isfinite(vals).any():
        return '<p class="sub">Not enough trades for a rolling window.</p>'
    ylo = min(np.nanmin(vals) * 1.2, -1.0)
    yhi = max(np.nanmax(vals) * 1.2, 1.0)
    ys = lambda v: _PT + (HT - _PT - _PB) * (1 - (v - ylo) / (yhi - ylo))
    x_of = lambda ts: _sx(int(dts.searchsorted(ts)), n)
    segs, cur = [], []
    for ts, v in zip(roll.index, vals):
        if np.isfinite(v):
            cur.append(f"{x_of(ts):.1f},{ys(v):.1f}")
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    lines = "".join(f'<polyline points="{" ".join(s)}" class="rollline"/>' for s in segs if len(s) > 1)
    split_mark = ""
    if split is not None:
        spx = _sx(split, n)
        split_mark = f'<line x1="{spx:.0f}" x2="{spx:.0f}" y1="{_PT}" y2="{HT-_PB}" class="splitl"/>'
    xstep = max(n // 6, 1)
    xt = "".join(
        f'<text x="{_sx(i, n):.0f}" y="{HT-6}" class="tick" text-anchor="middle">{dates[i][:7]}</text>'
        for i in range(0, n, xstep)
    )
    return (
        f'<svg viewBox="0 0 {_W} {HT}">'
        f'<line x1="{_PL}" x2="{_W-_PR}" y1="{ys(0):.0f}" y2="{ys(0):.0f}" class="zero"/>'
        f'<text x="{_PL-8}" y="{ys(0)+4:.0f}" class="tick" text-anchor="end">0%</text>'
        f'<text x="{_PL-8}" y="{ys(yhi/1.2)+4:.0f}" class="tick" text-anchor="end">{yhi/1.2:+.0f}%</text>'
        f"{split_mark}{xt}{lines}</svg>"
    )


# ---------------------------------------------------------------------------
# HTML sections
# ---------------------------------------------------------------------------

def _badge(R, dates) -> str:
    if R["active_now"]:
        return f'<span class="badge on">● SIGNAL ACTIVE — {dates[-1]}</span>'
    ls = R["last_sig"]
    if ls is None:
        return '<span class="badge off">no signals on these candles</span>'
    return f'<span class="badge off">last signal: {dates[ls]} ({len(dates)-1-ls} bars ago)</span>'


def _overview(computed, failed, dates, n, title) -> str:
    rows = ""
    for R in computed:
        s, spec = R["summary"], R["spec"]
        act = ('<span class="badge on">active</span>' if R["active_now"]
               else (dates[R["last_sig"]] if R["last_sig"] is not None else "–"))
        rows += (
            f'<tr><td class="rl"><a href="#{R["uid"]}">{_esc(spec.name)}</a></td>'
            f'<td>{spec.params.direction}</td>'
            f'<td class="rl"><code>{_esc(spec.candidate.expression[:52])}</code></td>'
            f'<td>{_pct(s.expectancy)}</td><td>{_pct(spec.oos_expectancy)}</td>'
            f'<td>{_num(s.profit_factor)}</td><td>{_num(s.win_rate_pct * 100, "{:.0f}%") }</td>'
            f'<td>{s.total_trades}</td><td>{R["dd"].min():.0f} pt</td><td>{act}</td></tr>'
        )
    for spec, err in failed:
        rows += (
            f'<tr><td class="rl">{_esc(spec.name)}</td><td colspan="9" class="rl err">'
            f"not evaluable on these candles — {_esc(err)}</td></tr>"
        )
    period = f"{dates[0]} → {dates[-1]} ({n} bars)" if dates else "no bars"
    return f"""<h1>{_esc(title)}</h1>
<p class="sub">{period} · deterministic signal replay (EventCandidate.apply) + backtest with the published
ValidatedRule parameters · net returns (per-side fee included)</p>
<table><tr><th>rule</th><th>dir</th><th>signal</th><th>exp replay</th><th>exp OOS</th><th>PF</th><th>WR</th><th>trades</th><th>max DD</th><th>last signal</th></tr>
{rows}</table>"""


def _rule_section(R, bh, dates, dts, regime_arr, close, low, high, window, min_trades, recent) -> str:
    spec, s, t, net = R["spec"], R["summary"], R["trades"], R["net"]
    p = spec.params
    wins, losses = net[net > 0], net[net < 0]
    dur = (t["exit_rn"] - t["fill_rn"]).to_numpy(float)
    payoff = abs(wins.mean() / losses.mean()) if wins.size and losses.size else float("nan")

    regime_html = ""
    if regime_arr is not None and len(t):
        tr_reg = regime_arr[t["signal_rn"].to_numpy(int)]
        seen = [rg for rg in _REGIME_ORDER if (tr_reg == rg).any()] + sorted(
            set(tr_reg) - set(_REGIME_ORDER)
        )
        cells = ""
        for rg in seen:
            m = tr_reg == rg
            cells += (
                f'<td>{_esc(rg.replace("STRONG_", "S."))}<br>'
                f"<b>{_pct(net[m].mean() / 100, 1)}</b><br>"
                f"<small>{int(m.sum())} trades</small></td>"
            )
        regime_html = (
            "<h2>Per-regime performance (regime at the signal bar)</h2>"
            f"<table><tr>{cells}</tr></table>"
        )

    recent_rows = ""
    for _, tr in t.tail(recent).iloc[::-1].iterrows():
        g = tr.net_pct_gain * 100
        recent_rows += (
            f"<tr><td>{dates[int(tr.signal_rn)]}</td><td>{dates[int(tr.fill_rn)]}</td>"
            f'<td>{dates[int(tr.exit_rn)]}</td><td>{"TP" if tr.target_hit else "horizon"}</td>'
            f'<td class="{"pos" if g > 0 else "neg"}">{g:+.2f}%</td><td>{tr.mae:.1f}%</td></tr>'
        )

    verdict = f" · {_esc(spec.verdict)}" if spec.verdict else ""
    entry_sign = "+" if p.direction == "short" else "−"
    tiles = "".join(
        f'<div class="kpi"><b>{v}</b><span>{k}</span></div>'
        for v, k in [
            (_pct(s.expectancy), "expectancy/trade (replay)"),
            (_pct(spec.oos_expectancy), "expectancy OOS (walk-forward)"),
            (_num(s.profit_factor), "profit factor"),
            (_num(s.win_rate_pct * 100, "{:.0f}%"), "win rate"),
            (f"{s.total_trades} / {s.total_signals}", "trades / signals"),
            (f"{R['exposure']:.1f}", "avg concurrent positions"),
        ]
    )
    stats_row = ""
    if net.size:
        stats_row = f"""<table class="statrow"><tr><th>winners</th><th>losers</th><th>avg win</th><th>avg loss</th><th>payoff</th><th>best</th><th>worst</th><th>TP hit</th><th>median duration</th></tr>
<tr><td><b>{(net > 0).sum()}</b></td><td><b>{(net < 0).sum()}</b></td>
<td class="pos">{_pct(wins.mean() / 100 if wins.size else None)}</td>
<td class="neg">{_pct(losses.mean() / 100 if losses.size else None)}</td>
<td><b>{_num(payoff)}</b></td>
<td class="pos">{net.max():+.1f}%</td><td class="neg">{net.min():+.1f}%</td>
<td><b>{t["target_hit"].mean() * 100:.0f}%</b></td><td><b>{np.median(dur):.0f} bars</b></td></tr></table>"""

    return f"""
<section id="{R['uid']}">
<h1>{_esc(spec.name)} <span class="hsub">· {p.direction.upper()}{verdict}</span> {_badge(R, dates)}</h1>
<p class="sub"><code>{_esc(spec.candidate.expression)}</code> &nbsp;·&nbsp;
entry limit {entry_sign}{p.buy_drop_pct:.1%} · TP {p.sell_pct:.1%} · horizon {p.target_h} bars · fee {p.fee:.2%}/side</p>
<div class="kpis">{tiles}</div>

<h2>Equity vs Buy &amp; Hold — dual axis (independent scales)</h2>
<p class="sub">Rule on the <b style="color:var(--blue)">left axis</b> (arithmetic sum of per-trade net gains),
benchmark on the <b style="color:var(--gray)">right axis</b>. The scales are independent: compare shapes, not slopes.</p>
{_equity_svg(R, bh, dates, R['split'])}
<div class="legend"><span><i style="background:var(--blue)"></i>Rule (left axis, cum. net %)</span>
<span><i style="background:var(--gray)"></i>Buy&amp;Hold (right axis)</span>
<span><i style="background:var(--red-l);border:1px solid var(--red)"></i>Rule drawdown (max {R['dd'].min():.0f} pt)</span></div>

<h2>Monthly activation trend — IS (gray) vs OOS (blue)</h2>
<p class="sub">Light bar = signals in the month, dark part = executed trades. An OOS frequency in line with IS is the
first sign of structural stability; the light/dark gap diagnoses the limit fill rate.</p>
{_monthly_svg(R, dates)}
<div class="legend"><span><i style="background:var(--gsig)"></i>Signals (IS)</span>
<span><i style="background:var(--gtrd)"></i>Executed (IS)</span>
<span><i style="background:var(--blue-l)"></i>Signals (OOS)</span>
<span><i style="background:var(--mtrd)"></i>Executed (OOS)</span></div>

<h2>Gain / loss distribution per trade</h2>
{_hist_svg(R)}
{stats_row}
{_return_distribution_section(R, close, low, high)}

<h2>MAE → final net (intra-trade risk; no stop-loss)</h2>
<p class="sub">One dot per trade: how far it went against the position (max adverse excursion from fill, x)
before closing at its final net (y). Winners with a high MAE show where a stop-loss would have cut
recovered trades; losers show the open-ended risk tail.</p>
{_mae_svg(R, dates)}
<div class="legend"><span><i style="background:var(--blue);border-radius:50%"></i>Winning trade</span>
<span><i style="background:var(--red);border-radius:50%"></i>Losing trade</span></div>

<h2>Rolling expectancy ({window}, min {min_trades} trades)</h2>
<p class="sub">Rolling mean of per-trade nets by exit date: persistently below zero or declining = decaying edge.</p>
{_rolling_svg(R, dts, dates, window, min_trades)}
{regime_html}

<h2>Recent trades</h2>
<table><tr><th>signal</th><th>fill</th><th>exit</th><th>type</th><th>net</th><th>MAE</th></tr>{recent_rows}</table>
</section>"""


def _footer_note(computed, dates) -> str:
    is_note = ""
    splits = {R["split"] for R in computed if R["split"] is not None}
    if splits:
        d = dates[max(splits) - 1]
        is_note = (
            f" Bars up to {d} were part of the discovery in-sample (gray): the replay there is "
            "descriptive, not predictive — the honest estimate is the OOS zone and the walk-forward "
            "expectancy shown next to the replay one."
        )
    return f"""<div class="note"><b>How to read this report.</b> The replay applies the published parameters
to every candle provided.{is_note} The dual axis uses <b>independent scales</b> (compare shapes, not
magnitudes). The backtest opens <b>overlapping positions</b> and models no portfolio sizing; there is no
stop-loss — the MAE scatter shows the intra-trade risk that follows. Equity is the arithmetic sum of
per-trade nets (compounding would assume full re-allocation per trade, which the mechanics do not model).</div>"""


def _hover_js(computed, bh, dates) -> str:
    data = {
        R["uid"]: {"eq": [round(float(v), 1) for v in R["eq"]]} for R in computed
    }
    return f"""
const BH={json.dumps([round(float(v), 1) for v in bh])};
const DTS={json.dumps(dates)};
const DATA={json.dumps(data)};
const tip=document.getElementById('tip');
document.querySelectorAll('.eqhover').forEach(hov=>{{
  const uid=hov.dataset.uid, svg=document.getElementById('eq_'+uid),
  cross=document.getElementById('cross_'+uid), d=DATA[uid];
  hov.addEventListener('mousemove',e=>{{
    const r=hov.getBoundingClientRect(), f=Math.min(Math.max((e.clientX-r.left)/r.width,0),1),
    i=Math.round(f*(DTS.length-1)), vb=svg.viewBox.baseVal,
    x={_PL}+(vb.width-{_PL}-{_PR})*i/(DTS.length-1);
    cross.setAttribute('x1',x);cross.setAttribute('x2',x);cross.setAttribute('visibility','visible');
    tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px';
    tip.innerHTML='<b>'+DTS[i]+'</b><br>Rule: '+d.eq[i].toFixed(0)+'% (left)<br>Buy&Hold: '+BH[i].toFixed(0)+'% (right)';
  }});
  hov.addEventListener('mouseleave',()=>{{cross.setAttribute('visibility','hidden');tip.style.display='none';}});
}});
"""


_CSS = """
:root{--surface:#fcfcfb;--panel:#f4f3f1;--border:#e2e1dd;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8984;
--blue:#2a78d6;--blue-l:#9ec5f4;--red:#e34948;--red-l:#efb4b2;--gray:#8a8984;
--gsig:#dddcd8;--gtrd:#a3a29c;--good:#0ca30c;--tipbg:#0b0b0b;--tipink:#fcfcfb;--mtrd:#2a78d6;
--distbase:#c9c8c3;--ochre:#eda100;--green:#008300}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--panel:#242422;--border:#3a3a37;--ink:#fff;
--ink2:#c3c2b7;--ink3:#8a8984;--blue:#3987e5;--blue-l:#1c4573;--red:#e66767;--red-l:#6e3230;
--gsig:#33332f;--gtrd:#5c5b56;--tipbg:#fcfcfb;--tipink:#0b0b0b;--mtrd:#3987e5;
--distbase:#46453f;--ochre:#c98500;--green:#008300}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:32px 40px 60px;max-width:1200px}
h1{font-size:21px;margin:0 0 2px}.hsub{font-weight:400;color:var(--ink2)}
h2{font-size:15px;margin:28px 0 4px}
h3{font-size:12.5px;margin:18px 0 4px;color:var(--ink2);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
section{border-top:2px solid var(--border);margin-top:42px;padding-top:18px}
.sub{color:var(--ink2);margin:0 0 10px;font-size:13px}
code{font-family:ui-monospace,Menlo,monospace;font-size:12px;background:var(--panel);padding:1px 6px;border-radius:4px}
a{color:var(--blue)}
.kpis{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}
.kpi{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:9px 14px;min-width:118px}
.kpi b{display:block;font-size:18px}.kpi span{color:var(--ink2);font-size:11.5px}
.badge{font-size:11px;font-weight:700;border-radius:12px;padding:3px 10px;vertical-align:3px;margin-left:8px}
.badge.on{background:var(--good);color:#fff}.badge.off{background:var(--panel);color:var(--ink2);border:1px solid var(--border);font-weight:500}
svg{width:100%;height:auto;display:block}
.grid{stroke:var(--border);stroke-width:1}.zero{stroke:var(--ink3);stroke-width:1}
.tick{font-size:10px;fill:var(--ink3)}.lab{font-size:11px;font-weight:600}
.tickeq{fill:var(--blue)}.tickbh{fill:var(--gray)}
.eqline{fill:none;stroke:var(--blue);stroke-width:2}.eqlab{fill:var(--blue)}
.bhline{fill:none;stroke:var(--gray);stroke-width:2;stroke-dasharray:5 4}.bhlab{fill:var(--gray)}
.ddarea{fill:var(--red-l);stroke:var(--red);stroke-width:1}.ddlab{fill:var(--red)}
.rollline{fill:none;stroke:var(--blue);stroke-width:2}
.splitl{stroke:var(--ink2);stroke-width:1.4;stroke-dasharray:6 4}.splitt{fill:var(--ink2);font-weight:600}
.msig{fill:var(--blue-l)}.mtrd{fill:var(--mtrd)}.gsig{fill:var(--gsig)}.gtrd{fill:var(--gtrd)}
.hpos{fill:var(--mtrd)}.hneg{fill:var(--red)}.hzero{fill:var(--gray)}
.distbasearea{fill:var(--distbase);opacity:.35}.distbaseline{fill:none;stroke:var(--ink3);stroke-width:1.6}
.distlowarea{fill:var(--red);opacity:.3}.distlowline{fill:none;stroke:var(--red);stroke-width:2}
.distclosearea{fill:var(--ochre);opacity:.3}.distcloseline{fill:none;stroke:var(--ochre);stroke-width:2}
.disthigharea{fill:var(--green);opacity:.3}.disthighline{fill:none;stroke:var(--green);stroke-width:2}
.meanbase{stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:3 3}
.meanlow{stroke:var(--red);stroke-width:2}.meanclose{stroke:var(--ochre);stroke-width:2}.meanhigh{stroke:var(--green);stroke-width:2}
.dw{fill:var(--blue);opacity:.75}.dl{fill:var(--red);opacity:.8}
.dw:hover,.dl:hover{opacity:1;stroke:var(--ink);stroke-width:1.5}
.cross{stroke:var(--ink);stroke-width:1;stroke-dasharray:2 3}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin:4px 0 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
table{border-collapse:collapse;margin-top:6px}td,th{border:1px solid var(--border);padding:5px 10px;font-size:12.5px;text-align:center}
td.rl{text-align:left}td.err{color:var(--red)}
th{background:var(--panel);font-size:11px;text-transform:uppercase;letter-spacing:.03em;color:var(--ink2)}
.pos{color:var(--blue);font-weight:600}.neg{color:var(--red);font-weight:600}
.statrow td b{font-size:14px}
.note{background:var(--panel);border:1px solid var(--border);border-left:4px solid #eda100;border-radius:6px;
padding:11px 15px;margin:26px 0 0;color:var(--ink2);font-size:13px}.note b{color:var(--ink)}
#tip{position:fixed;display:none;background:var(--tipbg);color:var(--tipink);padding:6px 10px;border-radius:6px;
font-size:12px;pointer-events:none;z-index:9}
"""
