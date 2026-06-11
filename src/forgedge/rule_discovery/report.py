"""Reporting for a Rule Discovery verdict — text and self-contained HTML.

These helpers turn a :class:`RuleDiscoveryResponse` into human-readable output.
They deliberately avoid matplotlib/seaborn (kept for notebook use in
``strategy_report_lib``): the package runtime stays ``numpy``/``pandas``-only and
the HTML is fully self-contained — plain tables, inline CSS, no CDN — so it opens
offline in any browser, matching the Rule Registry report philosophy.
"""
from __future__ import annotations

import html
from typing import List

from .models import RuleDiscoveryResponse


def text_report(resp: RuleDiscoveryResponse) -> str:
    """Compact plaintext verdict report."""
    s = resp.in_sample_summary
    lines: List[str] = []
    add = lines.append

    add("=" * 64)
    add(f"RULE DISCOVERY — {resp.verdict}")
    add(f"  alpha_id : {resp.alpha_id}")
    add(f"  asset/tf : {resp.asset} / {resp.timeframe}")
    add(f"  date     : {resp.date}")
    add("=" * 64)

    if resp.validated_rule:
        vr = resp.validated_rule
        p = vr.params
        add("VALIDATED RULE")
        add(f"  expression : {vr.expression}")
        add(f"  entry      : {p.buy_type}  drop={p.buy_drop_pct}  delay={p.buy_delay_bar}")
        add(f"  exit       : sell_pct={p.sell_pct}  target_h={p.target_h}  fee={p.fee}")
        add("-" * 64)

    add("IN-SAMPLE")
    add(f"  trades={s.total_trades}  fill_rate={_pct(s.fill_rate)}  "
        f"PF={_f(s.profit_factor)}  WR={_pct(s.win_rate_pct)}")
    add(f"  expectancy={_pct(s.expectancy)}  pf_score_tpm={_f(s.pf_score_tpm)}  "
        f"tpm_mu={_f(s.tpm_mu)}  zero_months={s.zero_months}")

    if resp.statistical_validation:
        sv = resp.statistical_validation
        add("-" * 64)
        add("STATISTICAL VALIDATION")
        add(f"  t-test WR vs base : p={_f(sv.ttest_winrate_p)}")
        add(f"  t-test exp vs 0   : p={_f(sv.ttest_expectancy_p)}")
        add(f"  Sharpe={_f(sv.sharpe_ratio)}  DSR={_f(sv.deflated_sharpe)}  "
            f"(n_trials={sv.n_trials_tested})")
        add(f"  temporal stability: {sv.temporal_stability}  "
            f"(PF {_f(sv.pf_first_half)} → {_f(sv.pf_second_half)})")

    if resp.execution_envelope:
        env = resp.execution_envelope
        c, o = env.conservative, env.optimistic
        add("-" * 64)
        add("EXECUTION ENVELOPE (range d'azione, no target choice)")
        add(f"  conservative (close): PF={_f(c.profit_factor)}  WR={_pct(c.win_rate_pct)}  "
            f"exp={_pct(c.expectancy)}  hit={_f(c.target_hit_rate_pct)}%")
        add(f"  optimistic   (high) : PF={_f(o.profit_factor)}  WR={_pct(o.win_rate_pct)}  "
            f"exp={_pct(o.expectancy)}  hit={_f(o.target_hit_rate_pct)}%")

    if resp.excursion:
        ex = resp.excursion
        add("-" * 64)
        add("MAE / MFE (intra-trade excursion)")
        add(f"  MAE adverse  : mean={_pct(ex.mae_mean)}  median={_pct(ex.mae_median)}  "
            f"worst={_pct(ex.mae_worst)}")
        add(f"  MFE favoured : mean={_pct(ex.mfe_mean)}  median={_pct(ex.mfe_median)}  "
            f"best={_pct(ex.mfe_best)}  reached_target={_f(ex.mfe_reached_target_pct)}%")

    if resp.walk_forward:
        wf = resp.walk_forward
        o = wf.oos_summary
        add("-" * 64)
        add("WALK-FORWARD OOS")
        add(f"  splits={len(wf.splits)}  profitable={wf.n_profitable_splits}  "
            f"consistency={_pct(wf.consistency)}")
        add(f"  OOS: trades={o.total_trades}  PF={_f(o.profit_factor)}  "
            f"WR={_pct(o.win_rate_pct)}  net={_pct(o.total_net_gain)}")
        for sp in wf.splits:
            ts = sp.test_summary
            add(f"    [{sp.split_idx}] {sp.test_from}→{sp.test_to}  "
                f"PF={_f(ts.profit_factor)}  WR={_pct(ts.win_rate_pct)}  "
                f"T={ts.total_trades}  net={_pct(ts.total_net_gain)}")

    if resp.regime_analysis:
        ra = resp.regime_analysis
        add("-" * 64)
        add("REGIME")
        add(f"  dependency_score={_f(ra.dependency_score)}  zero_months={ra.zero_months}  "
            f"avoid_in={ra.avoid_in or '—'}")

    if resp.rejection_reasons:
        add("-" * 64)
        add("REASONS")
        for r in resp.rejection_reasons:
            add(f"  · {r}")

    add("=" * 64)
    return "\n".join(lines)


def html_report(resp: RuleDiscoveryResponse) -> str:
    """Self-contained HTML verdict report (inline CSS, no external assets)."""
    s = resp.in_sample_summary
    badge = {"EDGE": "#27ae60", "PARTIAL-EDGE": "#f39c12", "NON-EDGE": "#c0392b"}.get(
        resp.verdict, "#7f8c8d"
    )
    parts: List[str] = []
    add = parts.append

    add("<!doctype html><html><head><meta charset='utf-8'>")
    add("<style>"
        "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222}"
        "h1{font-size:20px}h2{font-size:15px;border-bottom:1px solid #ddd;padding-bottom:4px}"
        ".badge{color:#fff;padding:3px 10px;border-radius:4px;font-weight:600}"
        "table{border-collapse:collapse;margin:8px 0;font-size:13px}"
        "td,th{border:1px solid #e0e0e0;padding:4px 8px;text-align:right}"
        "th{background:#f6f6f6}.l{text-align:left}"
        "code{background:#f4f4f4;padding:1px 5px;border-radius:3px}"
        ".muted{color:#888}</style></head><body>")

    add(f"<h1>FORGE Rule Discovery "
        f"<span class='badge' style='background:{badge}'>{html.escape(resp.verdict)}</span></h1>")
    add(f"<p class='muted'>{html.escape(resp.alpha_id)} · "
        f"{html.escape(resp.asset)} / {html.escape(resp.timeframe)} · {resp.date}</p>")

    if resp.validated_rule:
        vr = resp.validated_rule
        p = vr.params
        add("<h2>Validated rule</h2>")
        add(f"<p><code>{html.escape(vr.expression)}</code></p>")
        add(_kv_table({
            "entry mode": p.buy_type, "buy_drop_pct": p.buy_drop_pct,
            "buy_delay_bar": p.buy_delay_bar, "sell_pct": p.sell_pct,
            "target_h": p.target_h, "fee": p.fee,
        }))

    add("<h2>In-sample</h2>")
    add(_kv_table({
        "trades": s.total_trades, "fill_rate": _pct(s.fill_rate),
        "profit_factor": _f(s.profit_factor), "win_rate": _pct(s.win_rate_pct),
        "expectancy": _pct(s.expectancy), "pf_score_tpm": _f(s.pf_score_tpm),
        "tpm_mu": _f(s.tpm_mu), "zero_months": s.zero_months,
    }))

    if resp.statistical_validation:
        sv = resp.statistical_validation
        add("<h2>Statistical validation</h2>")
        add(_kv_table({
            "t-test WR p": _f(sv.ttest_winrate_p),
            "t-test exp p": _f(sv.ttest_expectancy_p),
            "Sharpe": _f(sv.sharpe_ratio), "DSR": _f(sv.deflated_sharpe),
            "n_trials": sv.n_trials_tested,
            "temporal_stability": sv.temporal_stability,
            "PF 1st→2nd half": f"{_f(sv.pf_first_half)} → {_f(sv.pf_second_half)}",
        }))

    if resp.execution_envelope:
        env = resp.execution_envelope
        c, o = env.conservative, env.optimistic
        add("<h2>Execution envelope</h2>")
        add("<table><tr><th class='l'>convention</th><th>PF</th><th>WR</th>"
            "<th>expectancy</th><th>hit rate</th></tr>"
            f"<tr><td class='l'>conservative (close)</td><td>{_f(c.profit_factor)}</td>"
            f"<td>{_pct(c.win_rate_pct)}</td><td>{_pct(c.expectancy)}</td>"
            f"<td>{_f(c.target_hit_rate_pct)}%</td></tr>"
            f"<tr><td class='l'>optimistic (high)</td><td>{_f(o.profit_factor)}</td>"
            f"<td>{_pct(o.win_rate_pct)}</td><td>{_pct(o.expectancy)}</td>"
            f"<td>{_f(o.target_hit_rate_pct)}%</td></tr></table>")

    if resp.excursion:
        ex = resp.excursion
        add("<h2>MAE / MFE excursion</h2>")
        add("<table><tr><th class='l'></th><th>mean</th><th>median</th><th>extreme</th></tr>"
            f"<tr><td class='l'>MAE (adverse)</td><td>{_pct(ex.mae_mean)}</td>"
            f"<td>{_pct(ex.mae_median)}</td><td>{_pct(ex.mae_worst)}</td></tr>"
            f"<tr><td class='l'>MFE (favourable)</td><td>{_pct(ex.mfe_mean)}</td>"
            f"<td>{_pct(ex.mfe_median)}</td><td>{_pct(ex.mfe_best)}</td></tr></table>"
            f"<p class='muted'>MFE reached take-profit in "
            f"{_f(ex.mfe_reached_target_pct)}% of trades</p>")

    if resp.walk_forward:
        wf = resp.walk_forward
        o = wf.oos_summary
        add("<h2>Walk-forward out-of-sample</h2>")
        add(_kv_table({
            "splits": len(wf.splits), "profitable_splits": wf.n_profitable_splits,
            "consistency": _pct(wf.consistency), "OOS trades": o.total_trades,
            "OOS PF": _f(o.profit_factor), "OOS WR": _pct(o.win_rate_pct),
            "OOS net gain": _pct(o.total_net_gain),
        }))
        rows = "".join(
            f"<tr><td>{sp.split_idx}</td><td class='l'>{sp.test_from}→{sp.test_to}</td>"
            f"<td>{_f(sp.test_summary.profit_factor)}</td>"
            f"<td>{_pct(sp.test_summary.win_rate_pct)}</td>"
            f"<td>{sp.test_summary.total_trades}</td>"
            f"<td>{_pct(sp.test_summary.total_net_gain)}</td></tr>"
            for sp in wf.splits
        )
        add("<table><tr><th>#</th><th class='l'>test window</th><th>PF</th>"
            f"<th>WR</th><th>T</th><th>net</th></tr>{rows}</table>")

    if resp.regime_analysis and resp.regime_analysis.per_regime:
        add("<h2>Regime breakdown</h2>")
        rows = "".join(
            f"<tr><td class='l'>{html.escape(str(r['regime']))}</td>"
            f"<td>{r['n_trades']}</td><td>{_pct(r['win_rate'])}</td>"
            f"<td>{_f(r['profit_factor'])}</td><td>{_pct(r['expectancy'])}</td></tr>"
            for r in resp.regime_analysis.per_regime
        )
        add("<table><tr><th class='l'>regime</th><th>T</th><th>WR</th>"
            f"<th>PF</th><th>exp</th></tr>{rows}</table>")
        add(f"<p class='muted'>dependency score "
            f"{_f(resp.regime_analysis.dependency_score)} · "
            f"avoid in: {html.escape(', '.join(resp.regime_analysis.avoid_in) or '—')}</p>")

    if resp.rejection_reasons:
        add("<h2>Reasons</h2><ul>")
        for r in resp.rejection_reasons:
            add(f"<li>{html.escape(r)}</li>")
        add("</ul>")

    add("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _kv_table(d: dict) -> str:
    rows = "".join(
        f"<tr><th class='l'>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in d.items()
    )
    return f"<table>{rows}</table>"


def _f(x) -> str:
    try:
        xf = float(x)
        return f"{xf:.3f}" if xf == xf else "—"  # NaN check
    except (TypeError, ValueError):
        return "—"


def _pct(x) -> str:
    try:
        xf = float(x)
        return f"{xf:.2%}" if xf == xf else "—"
    except (TypeError, ValueError):
        return "—"
