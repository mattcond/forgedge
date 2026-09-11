"""Shared replay / portfolio-metric helpers.

A rule is replayed with EventCandidate.apply() over the FULL kpi table (so
rolling baselines keep their history — pitfall #11) and backtested with the
validated BacktestParams; `timerange_from/to` only restrict which signals are
allowed to *open* a position, so IS / OOS / HO windows are cut on the entry bar.

Position sizing: every opened trade uses exactly UNIT = 100 EUR of capital.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "src")
from forgedge.rule_discovery.backtest import run_backtest

UNIT = 100.0
ROOT = Path("/home/user/forgedge")
_KPI: dict[str, pd.DataFrame] = {}


def kpi(ticker: str) -> pd.DataFrame:
    if ticker not in _KPI:
        _KPI[ticker] = pd.read_parquet(ROOT / f"work/kpi/{ticker}.parquet")
    return _KPI[ticker]


def replay(rule: dict, frm: str | None, to: str | None) -> tuple[object, pd.DataFrame]:
    """Backtest one rule over [frm, to) on the entry bar. Returns (summary, trades)."""
    df = kpi(rule["ticker"]).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = rule["candidate"].apply(df)
    df["__sig__"] = sig.fillna(0.0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summ, tr = run_backtest(df, "__sig__", rule["params"], timerange_from=frm,
                                timerange_to=to, timestamp_col="open_dt", return_trades=True)
    tr = tr.copy()
    tr["rule"] = rule["alpha_id"]
    tr["ticker"] = rule["ticker"]
    tr["pnl_eur"] = tr["net_pct_gain"] * UNIT
    tr["fill_dt"] = pd.to_datetime(tr["fill_dt"])
    tr["exit_dt"] = pd.to_datetime(tr["exit_dt"])
    return summ, tr


def exposure_curve(trades: pd.DataFrame) -> pd.Series:
    """Open-position count per calendar day across every ticker in `trades`."""
    if trades.empty:
        return pd.Series(dtype=float)
    days = pd.date_range(trades["fill_dt"].min().normalize(),
                         trades["exit_dt"].max().normalize(), freq="D")
    cnt = np.zeros(len(days), dtype=int)
    pos = pd.Series(np.arange(len(days)), index=days)
    for a, b in zip(trades["fill_dt"].dt.normalize(), trades["exit_dt"].dt.normalize()):
        cnt[pos[a]: pos[b] + 1] += 1
    return pd.Series(cnt, index=days)


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """(max drawdown in currency, max drawdown as a fraction of the running peak)."""
    if equity.empty:
        return 0.0, 0.0
    peak = equity.cummax()
    dd = equity - peak
    return float(-dd.min()), float((-(dd / peak.replace(0, np.nan))).max() or 0.0)


def portfolio_metrics(trades: pd.DataFrame, n_signals: int, label: str,
                      start: str, end: str | None) -> dict:
    """Aggregate a set of per-trade rows into the reported KPI block."""
    n = len(trades)
    if n == 0:
        return dict(window=label, start=start, end=end, n_signals=n_signals, n_trades=0)
    pnl = trades["pnl_eur"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_w, gross_l = float(wins.sum()), float(-losses.sum())
    exp_c = exposure_curve(trades)
    max_exp = int(exp_c.max()) if len(exp_c) else 0
    mean_exp = float(exp_c[exp_c > 0].mean()) if (exp_c > 0).any() else 0.0
    capital = max_exp * UNIT

    closed = trades.sort_values("exit_dt")
    eq = capital + closed.set_index("exit_dt")["pnl_eur"].groupby(level=0).sum().cumsum()
    # prepend the starting point so a first-trade loss is measured against capital
    eq = pd.concat([pd.Series([capital], index=[closed["fill_dt"].min()]), eq])
    dd_eur, dd_pct = max_drawdown(eq)

    hold = (trades["exit_dt"] - trades["fill_dt"]).dt.days
    months = trades.set_index("exit_dt")["pnl_eur"].groupby(pd.Grouper(freq="ME")).sum()
    span_m = max((pd.Timestamp(end or trades["exit_dt"].max()) - pd.Timestamp(start)).days / 30.44, 1e-9)
    mu, sd = float(pnl.mean()), float(pnl.std(ddof=1)) if n > 1 else 0.0
    tpy = n / span_m * 12
    return dict(
        window=label, start=start, end=end,
        n_signals=int(n_signals), n_trades=n,
        fill_rate=round(n / n_signals, 4) if n_signals else np.nan,
        win_rate=round(float((pnl > 0).mean()), 4),
        profit_factor=round(gross_w / gross_l, 3) if gross_l > 0 else np.inf,
        net_gain_eur=round(float(pnl.sum()), 2),
        gross_win_eur=round(gross_w, 2), gross_loss_eur=round(-gross_l, 2),
        expectancy_eur=round(mu, 3),
        max_dd_eur=round(dd_eur, 2), max_dd_pct=round(dd_pct * 100, 2),
        max_concurrent=max_exp, max_exposure_eur=round(capital, 2),
        mean_concurrent=round(mean_exp, 2),
        return_on_max_exposure_pct=round(float(pnl.sum()) / capital * 100, 2) if capital else np.nan,
        best_trade_eur=round(float(pnl.max()), 2), worst_trade_eur=round(float(pnl.min()), 2),
        avg_hold_days=round(float(hold.mean()), 1),
        trades_per_month=round(n / span_m, 2),
        sharpe_ann=round(mu / sd * np.sqrt(tpy), 2) if sd > 0 else np.nan,
        pct_months_positive=round(float((months > 0).mean()) * 100, 1) if len(months) else np.nan,
        n_months=len(months),
    )
