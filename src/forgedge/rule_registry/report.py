"""Self-contained HTML report for the Rule Registry (FORGE Modulo 4).

A single, dependency-free HTML file (spec Section 11): every chart is inline SVG,
no CDN, no external JavaScript — it opens offline in any browser.  The report is
deliberately *honest* (Section 11): duplicates carry a ``DUPLICATE OF …`` banner,
non-generic rules carry a ``SPECIFIC`` / ``ISOLATED`` badge, and the full
traceability chain ``rule_id → alpha_id`` is shown.  Nothing is hidden — the
user sees everything and decides.

The document follows the five-section layout of the spec:

1. Overview — per-ticker funnel and a summary table of every rule.
2. Correlation heatmaps — Jaccard (activations) and Spearman (gains).
3. Cross-ticker summary — rule × ticker PF/verdict with genericity badges.
4. Per-rule detail — one card per rule (equity curve, cross-ticker, regime).
5. Trade log — every executed trade across all tickers.

Kept ``numpy``/``pandas``-only and matplotlib-free, mirroring the Rule Discovery
report philosophy.
"""
from __future__ import annotations

import html
from typing import Dict, List, Optional

from .models import CorrelationMatrices, RegistryConfig, RuleDocument

_BADGE = {
    "GENERIC": "#27ae60",
    "PARTIAL": "#f39c12",
    "SPECIFIC": "#e67e22",
    "ISOLATED": "#c0392b",
}


def html_report(
    docs: List[RuleDocument],
    matrices: CorrelationMatrices,
    config: Optional[RegistryConfig] = None,
    *,
    tickers: Optional[List[str]] = None,
    session_date: Optional[str] = None,
    timeframe: str = "",
    funnel: Optional[Dict[str, dict]] = None,
) -> str:
    """Render the full registry report as a self-contained HTML string.

    Parameters
    ----------
    docs : list[RuleDocument]
        The registry documents (Steps 1–4 already applied).
    matrices : CorrelationMatrices
        The Jaccard / Spearman matrices for the heatmaps.
    config : RegistryConfig, optional
        Controls the trade-log / chart sections.
    tickers : list[str], optional
        Session tickers, for the header.  Defaults to the distinct source
        tickers of ``docs``.
    session_date : str, optional
        Session date for the header.
    timeframe : str
        Timeframe label for the header.
    funnel : dict[str, dict], optional
        Optional per-ticker funnel counts for the overview, e.g.
        ``{"ADAUSDC": {"candidates": 847, "validated": 4}}``.
    """
    config = config or RegistryConfig()
    if tickers is None:
        tickers = sorted({d.source_ticker for d in docs})

    n_dup = sum(1 for d in docs if d.is_duplicate)
    n_generic = sum(1 for d in docs if d.is_generic)

    parts: List[str] = []
    add = parts.append

    add("<!doctype html><html><head><meta charset='utf-8'>")
    add("<style>" + _CSS + "</style></head><body>")

    # ── header ───────────────────────────────────────────────────────────
    add("<h1>FORGE Report <span class='muted' style='font-size:14px'>· Rule Registry</span></h1>")
    add(f"<p class='muted'>Tickers: {html.escape(', '.join(tickers)) or '—'}"
        f"{' · Timeframe: ' + html.escape(timeframe) if timeframe else ''}"
        f"{' · Session: ' + html.escape(session_date) if session_date else ''}</p>")
    add(f"<p class='muted'>Rules in registry: <b>{len(docs)}</b> · "
        f"duplicates: <b>{n_dup}</b> · generic: <b>{n_generic}</b></p>")

    _section_overview(add, docs, tickers, funnel)
    _section_heatmaps(add, matrices, config)
    _section_cross_ticker(add, docs, tickers)
    _section_detail(add, docs, config)
    if config.html_include_tradelog:
        _section_tradelog(add, docs)

    add("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Section 1 — overview
# ---------------------------------------------------------------------------

def _section_overview(add, docs, tickers, funnel):
    add("<h2>1 — Overview</h2>")
    if funnel:
        add("<table><tr><th class='l'>ticker</th><th>candidates</th>"
            "<th>validated</th></tr>")
        for t in tickers:
            f = funnel.get(t, {})
            add(f"<tr><td class='l'>{html.escape(t)}</td>"
                f"<td>{f.get('candidates', '—')}</td>"
                f"<td>{f.get('validated', '—')}</td></tr>")
        add("</table>")

    add("<table><tr><th class='l'>rule</th><th class='l'>source</th><th>grade</th>"
        "<th>PF</th><th>WR</th><th>T</th><th>cross</th><th>class</th><th>dup</th></tr>")
    for d in docs:
        st = d.stats
        cross = (f"{d.cross_ticker_score}/{d.cross_ticker_total}"
                 if d.cross_ticker_total else "—")
        add(f"<tr><td class='l'><code>{html.escape(d.rule_id)}</code></td>"
            f"<td class='l'>{html.escape(d.source_ticker)}</td>"
            f"<td>{html.escape(str(d.grade))}</td>"
            f"<td>{_f(st.get('pf'))}</td><td>{_pct(st.get('win_rate'))}</td>"
            f"<td>{st.get('total_trades', '—')}</td><td>{cross}</td>"
            f"<td>{_badge(d.classification)}</td>"
            f"<td>{'✓' if d.is_duplicate else ''}</td></tr>")
    add("</table>")


# ---------------------------------------------------------------------------
# Section 2 — correlation heatmaps
# ---------------------------------------------------------------------------

def _section_heatmaps(add, matrices, config):
    add("<h2>2 — Correlation heatmaps</h2>")
    if len(matrices.rule_ids) < 2:
        add("<p class='muted'>Need at least two rules for a correlation matrix.</p>")
        return
    if config.html_charts:
        add("<h3>Jaccard — activation overlap (by date)</h3>")
        add(_heatmap_svg(matrices.jaccard, vmin=0.0, vmax=1.0, diverging=False))
        add("<h3>Spearman — gain correlation</h3>")
        add(_heatmap_svg(matrices.spearman, vmin=-1.0, vmax=1.0, diverging=True))


# ---------------------------------------------------------------------------
# Section 3 — cross-ticker summary
# ---------------------------------------------------------------------------

def _section_cross_ticker(add, docs, tickers):
    add("<h2>3 — Cross-ticker summary</h2>")
    add("<table><tr><th class='l'>rule</th>")
    for t in tickers:
        add(f"<th>{html.escape(t)}</th>")
    add("<th>score</th><th>class</th></tr>")
    for d in docs:
        add(f"<tr><td class='l'><code>{html.escape(d.rule_id)}</code></td>")
        for t in tickers:
            if t == d.source_ticker:
                add("<td class='muted'>source</td>")
                continue
            res = d.cross_ticker.get(t)
            if res is None:
                add("<td>—</td>")
            else:
                col = "#27ae60" if res.verdict == "PASS" else "#c0392b"
                add(f"<td><span style='color:{col};font-weight:600'>"
                    f"{_f(res.pf)}</span> <span class='muted'>{res.verdict}</span></td>")
        cross = (f"{d.cross_ticker_score}/{d.cross_ticker_total}"
                 if d.cross_ticker_total else "—")
        add(f"<td>{cross}</td><td>{_badge(d.classification)}</td></tr>")
    add("</table>")


# ---------------------------------------------------------------------------
# Section 4 — per-rule detail
# ---------------------------------------------------------------------------

def _section_detail(add, docs, config):
    add("<h2>4 — Rule detail</h2>")
    for d in docs:
        add("<div class='card'>")
        if d.is_duplicate and d.duplicate_of:
            add(f"<div class='banner'>DUPLICATE OF {html.escape(d.duplicate_of)}</div>")
        st = d.stats
        add(f"<div class='cardhead'><code>{html.escape(d.rule_id)}</code> "
            f"<span class='tag'>{html.escape(str(d.grade))}</span> "
            f"<span class='tag'>{html.escape(d.verdict)}</span> "
            f"{_badge(d.classification)}</div>")
        add(f"<p><code>{html.escape(d.expression)}</code></p>")
        add(f"<p class='muted'>Source: {html.escape(d.source_ticker)} · "
            f"PF {_f(st.get('pf'))} · WR {_pct(st.get('win_rate'))} · "
            f"T {st.get('total_trades', '—')} · DSR {_f(st.get('dsr'))}</p>")

        if config.html_charts and st.get("monthly_gains"):
            add(_equity_svg(st["monthly_gains"]))

        # cross-ticker lines
        if d.cross_ticker:
            add("<p class='muted'>Cross-ticker:</p><ul>")
            for t, res in d.cross_ticker.items():
                col = "#27ae60" if res.verdict == "PASS" else "#c0392b"
                add(f"<li>{html.escape(t)} → PF {_f(res.pf)} · WR {_pct(res.win_rate)} "
                    f"· <span style='color:{col};font-weight:600'>{res.verdict}</span></li>")
            add("</ul>")

        avoid = ", ".join(d.regime.get("avoid_in", []) or []) or "—"
        add(f"<p class='muted'>Regime: {html.escape(str(d.regime.get('type')))} · "
            f"avoid: {html.escape(avoid)} · overlap max {_f(d.overlap_max)} · "
            f"gain corr max {_f(d.gain_corr_max)}</p>")
        add(f"<p class='muted'>Traceability: {html.escape(d.rule_id)} → "
            f"{html.escape(d.source_alpha_id)}</p>")
        add("</div>")


# ---------------------------------------------------------------------------
# Section 5 — trade log
# ---------------------------------------------------------------------------

def _section_tradelog(add, docs):
    add("<h2>5 — Trade log</h2>")
    rows = []
    for d in docs:
        for date, gain in zip(d.activation_dates, d.gains):
            rows.append((date, d.source_ticker, d.rule_id, gain))
    if not rows:
        add("<p class='muted'>No trades.</p>")
        return
    rows.sort(key=lambda r: r[0])
    add("<table><tr><th class='l'>date</th><th class='l'>ticker</th>"
        "<th class='l'>rule</th><th>result</th><th>won</th></tr>")
    for date, ticker, rule_id, gain in rows:
        won = "✓" if gain > 0 else ""
        add(f"<tr><td class='l'>{html.escape(str(date))}</td>"
            f"<td class='l'>{html.escape(ticker)}</td>"
            f"<td class='l'><code>{html.escape(rule_id)}</code></td>"
            f"<td>{_pct(gain)}</td><td>{won}</td></tr>")
    add("</table>")


# ---------------------------------------------------------------------------
# Inline SVG charts
# ---------------------------------------------------------------------------

def _heatmap_svg(matrix, vmin: float, vmax: float, diverging: bool) -> str:
    """Render a labelled correlation matrix as an inline SVG heatmap."""
    ids = list(matrix.columns)
    n = len(ids)
    cell = 38
    pad_l = 90
    pad_t = 90
    w = pad_l + n * cell + 10
    h = pad_t + n * cell + 10
    out = [f"<svg width='{w}' height='{h}' xmlns='http://www.w3.org/2000/svg' "
           "font-family='sans-serif' font-size='10'>"]
    # column labels (rotated)
    for j, cid in enumerate(ids):
        x = pad_l + j * cell + cell / 2
        out.append(f"<text x='{x}' y='{pad_t - 6}' text-anchor='start' "
                   f"transform='rotate(-60 {x} {pad_t - 6})'>{html.escape(cid)}</text>")
    for i, rid in enumerate(ids):
        y = pad_t + i * cell + cell / 2
        out.append(f"<text x='{pad_l - 6}' y='{y + 3}' text-anchor='end'>"
                   f"{html.escape(rid)}</text>")
        for j in range(n):
            v = float(matrix.iat[i, j])
            color = _diverging_color(v) if diverging else _seq_color(v, vmin, vmax)
            x = pad_l + j * cell
            yy = pad_t + i * cell
            txt = "#fff" if _is_dark(color) else "#222"
            out.append(f"<rect x='{x}' y='{yy}' width='{cell - 1}' height='{cell - 1}' "
                       f"fill='{color}'/>")
            out.append(f"<text x='{x + cell / 2}' y='{yy + cell / 2 + 3}' "
                       f"text-anchor='middle' fill='{txt}'>{v:.2f}</text>")
    out.append("</svg>")
    return "".join(out)


def _equity_svg(monthly_gains: List[float]) -> str:
    """Render the cumulative monthly equity curve as a small inline SVG."""
    if not monthly_gains:
        return ""
    cum = []
    acc = 0.0
    for g in monthly_gains:
        acc += float(g)
        cum.append(acc)
    w, h, pad = 360, 90, 8
    lo = min(cum + [0.0])
    hi = max(cum + [0.0])
    span = (hi - lo) or 1.0
    n = len(cum)
    step = (w - 2 * pad) / max(n - 1, 1)

    def y(v):
        return h - pad - (v - lo) / span * (h - 2 * pad)

    zero_y = y(0.0)
    pts = " ".join(f"{pad + i * step:.1f},{y(v):.1f}" for i, v in enumerate(cum))
    color = "#27ae60" if cum[-1] >= 0 else "#c0392b"
    return (f"<svg width='{w}' height='{h}' xmlns='http://www.w3.org/2000/svg'>"
            f"<line x1='{pad}' y1='{zero_y:.1f}' x2='{w - pad}' y2='{zero_y:.1f}' "
            "stroke='#ccc' stroke-dasharray='3 3'/>"
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{pts}'/>"
            "</svg>")


# ── colour helpers ─────────────────────────────────────────────────────────

def _seq_color(v: float, vmin: float, vmax: float) -> str:
    """Sequential white→blue scale for the Jaccard matrix."""
    t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
    t = min(max(t, 0.0), 1.0)
    r = int(255 - t * (255 - 31))
    g = int(255 - t * (255 - 119))
    b = int(255 - t * (255 - 180))
    return f"#{r:02x}{g:02x}{b:02x}"


def _diverging_color(v: float) -> str:
    """Red↔white↔green diverging scale for the Spearman matrix (−1..1)."""
    t = min(max(v, -1.0), 1.0)
    if t >= 0:
        r = int(255 - t * (255 - 39))
        g = int(255 - t * (255 - 174))
        b = int(255 - t * (255 - 96))
    else:
        a = -t
        r = int(255 - a * (255 - 192))
        g = int(255 - a * (255 - 57))
        b = int(255 - a * (255 - 43))
    return f"#{r:02x}{g:02x}{b:02x}"


def _is_dark(hex_color: str) -> bool:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 140


def _badge(classification: Optional[str]) -> str:
    if not classification:
        return "<span class='muted'>—</span>"
    color = _BADGE.get(classification, "#7f8c8d")
    return (f"<span class='badge' style='background:{color}'>"
            f"{html.escape(classification)}</span>")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _f(x) -> str:
    try:
        xf = float(x)
        return f"{xf:.3f}" if xf == xf else "—"
    except (TypeError, ValueError):
        return "—"


def _pct(x) -> str:
    try:
        xf = float(x)
        return f"{xf:.2%}" if xf == xf else "—"
    except (TypeError, ValueError):
        return "—"


_CSS = (
    "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222}"
    "h1{font-size:20px}h2{font-size:15px;border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:28px}"
    "h3{font-size:13px;color:#444}"
    ".badge{color:#fff;padding:2px 8px;border-radius:4px;font-weight:600;font-size:11px}"
    ".tag{background:#eee;color:#333;padding:2px 7px;border-radius:4px;font-size:11px}"
    "table{border-collapse:collapse;margin:8px 0;font-size:12px}"
    "td,th{border:1px solid #e0e0e0;padding:4px 8px;text-align:right}"
    "th{background:#f6f6f6}.l{text-align:left}"
    "code{background:#f4f4f4;padding:1px 5px;border-radius:3px;font-size:12px}"
    ".muted{color:#888}"
    ".card{border:1px solid #e0e0e0;border-radius:6px;padding:12px 16px;margin:12px 0}"
    ".cardhead{font-size:14px;margin-bottom:4px}"
    ".banner{background:#c0392b;color:#fff;padding:4px 10px;border-radius:4px;"
    "font-weight:600;font-size:12px;margin-bottom:8px;display:inline-block}"
    "ul{margin:4px 0;font-size:12px}"
)
