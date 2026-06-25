"""``summary_report`` — data-quality diagnostics for a FORGE input table.

Runs a battery of cheap, robust checks on the OHLC price columns of the table
that is about to be fed to :func:`forge`, and prints a terminal report that
surfaces the data problems most likely to silently break the pipeline:

* **scale inconsistency** — values living on two different magnitudes (e.g. a
  feed that stores sub-parity FX bars as ``0.99`` and the rest as ``11361``),
  which poisons every rolling feature that crosses the boundary;
* **OHLC inconsistency** — ``high < low``, ``close`` outside ``[low, high]``,
  non-positive or NaN prices;
* **outliers** — impossible single-bar returns or ranges (a bad print), flagged
  with a robust MAD criterion that is itself insensitive to the outliers;
* **missing / duplicate / out-of-order bars** — gaps in the time index beyond
  the inferred cadence.

By design the value checks read **only** ``open``, ``high``, ``low`` and
``close``; the datetime source (a column or the index) is used solely for the
time-continuity section.  No volume or feature columns are required.

The report is **purely diagnostic / advisory**: ``summary_report`` never raises
on bad data and never blocks the pipeline.  It describes what it found and
ranks each observation by severity (``CRITICO`` / ``AVVISO``); the caller
decides what to do with the findings.

Usage
-----
    from forgedge import summary_report

    summary_report(kpi)                       # print the diagnostic report
    rep = summary_report(kpi, return_report=True, verbose=False)
    for f in rep.findings:                    # advisory only — no exception
        log.warning("%s: %s", f.level, f.message)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["summary_report", "DataQualityReport", "Finding"]

_OHLC: Tuple[str, ...] = ("open", "high", "low", "close")

# Severity ladder used throughout the report.
_OK, _WARN, _FAIL = "OK", "WARN", "FAIL"
_MARK = {_OK: "✓", _WARN: "⚠", _FAIL: "✗"}
# Etichette diagnostiche: il report è consultivo, non un gate pass/fail.
_LABEL = {_OK: "OK", _WARN: "AVVISO", _FAIL: "CRITICO"}
_RANK = {_OK: 0, _WARN: 1, _FAIL: 2}


@dataclass
class Finding:
    """A single diagnostic outcome (one row of the verdict)."""

    level: str   # _OK | _WARN | _FAIL
    code: str    # short stable identifier, e.g. "scale_mixed"
    message: str


@dataclass
class DataQualityReport:
    """Structured result of :func:`summary_report`.

    ``findings`` is the authoritative verdict list; the other fields hold the
    numeric detail rendered by :meth:`to_text`.
    """

    n_bars: int
    price_cols: List[str]
    findings: List[Finding] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)  # pre-rendered text blocks

    # ------------------------------------------------------------------
    @property
    def worst(self) -> str:
        """Highest severity across all findings (``OK`` when empty)."""
        return max((f.level for f in self.findings), key=lambda l: _RANK[l], default=_OK)

    @property
    def has_critical(self) -> bool:
        """True se esiste almeno un'anomalia critica.

        Puramente informativo per il chiamante: il report **non blocca mai** la
        pipeline e ``summary_report`` non solleva eccezioni sui dati cattivi.
        """
        return any(f.level == _FAIL for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.level == _WARN for f in self.findings)

    def one_line(self) -> str:
        """Compact single-line summary of the FAIL/WARN findings."""
        bad = [f for f in self.findings if f.level != _OK]
        if not bad:
            return "data quality OK"
        return "; ".join(f"[{f.level}] {f.message}" for f in bad)

    def to_text(self) -> str:
        """Full multi-section terminal report."""
        lines: List[str] = []
        lines.append("═" * 72)
        lines.append("  FORGE — DATA QUALITY SUMMARY")
        lines.append("═" * 72)
        lines.extend(self.sections)
        lines.append("─" * 72)
        lines.append("  DIAGNOSTICA (report consultivo — non blocca la pipeline)")
        lines.append("─" * 72)
        for f in sorted(self.findings, key=lambda f: -_RANK[f.level]):
            lines.append(f"  {_MARK[f.level]} [{_LABEL[f.level]:<7}] {f.message}")
        n_crit = sum(1 for f in self.findings if f.level == _FAIL)
        n_warn = sum(1 for f in self.findings if f.level == _WARN)
        if n_crit:
            summary = (f"{n_crit} anomalie critiche e {n_warn} avvisi — "
                       "controllare/correggere i dati prima dell'uso")
        elif n_warn:
            summary = f"{n_warn} avvisi — verificare se le anomalie sono attese"
        else:
            summary = "nessuna anomalia rilevata"
        lines.append("─" * 72)
        lines.append(f"  {_MARK[self.worst]} {summary}")
        lines.append("═" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def summary_report(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "open_dt",
    price_cols: Sequence[str] = _OHLC,
    timeframe: Optional[str] = None,
    return_high_move: float = 0.5,
    top_n: int = 5,
    verbose: bool = True,
    return_report: bool = False,
) -> Optional[DataQualityReport]:
    """Build (and by default print) a data-quality report for ``df``.

    Purely diagnostic: never raises on bad data and never blocks the pipeline —
    it only describes what it found.  Inspect ``DataQualityReport.findings`` (or
    ``has_critical`` / ``has_warnings``) to drive any caller-side decision.

    Parameters
    ----------
    df : pd.DataFrame
        The table about to be passed to :func:`forge`.  Only ``price_cols`` are
        inspected for value checks; the datetime source (``timestamp_col`` or a
        ``DatetimeIndex``) is used for the time-continuity section.
    timestamp_col : str
        Name of the datetime column.  Ignored when ``df`` has a ``DatetimeIndex``.
    price_cols : sequence of str
        Columns checked for scale / OHLC / outlier issues.  Defaults to the four
        OHLC columns; any not present in ``df`` are skipped with a finding.
    timeframe : str, optional
        Bar size label (e.g. ``"1D"``, ``"1H"``) used only to phrase the cadence
        section; when omitted the cadence is inferred from the timestamps.
    return_high_move : float
        Absolute single-bar return above which a bar is always flagged as an
        impossible move (default ``0.5`` = 50 %), independent of the robust
        MAD criterion.
    top_n : int
        Number of worst offenders listed per outlier check.
    verbose : bool
        Print the rendered report to stdout (default True).
    return_report : bool
        Return the :class:`DataQualityReport` object (default False).
    """
    rep = _build_report(
        df, timestamp_col=timestamp_col, price_cols=tuple(price_cols),
        timeframe=timeframe, return_high_move=return_high_move, top_n=top_n,
    )
    if verbose:
        print(rep.to_text())
    return rep if return_report else None


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _build_report(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    price_cols: Tuple[str, ...],
    timeframe: Optional[str],
    return_high_move: float,
    top_n: int,
) -> DataQualityReport:
    n = len(df)
    present = [c for c in price_cols if c in df.columns]
    rep = DataQualityReport(n_bars=n, price_cols=present)

    ts = _extract_timestamps(df, timestamp_col)
    # Lavora posizionalmente: evita disallineamenti tra maschere numpy e Series
    # quando l'indice in input non è 0..n-1 (es. un parquet con indice salvato).
    df = df.reset_index(drop=True)

    rep.sections.append(_section_overview(df, ts, present, price_cols, timeframe))
    if n == 0:
        rep.findings.append(Finding(_FAIL, "empty", "la tabella è vuota (0 barre)"))
        return rep

    missing = [c for c in price_cols if c not in df.columns]
    if missing:
        rep.findings.append(
            Finding(_FAIL, "missing_cols", f"colonne prezzo assenti: {missing}")
        )
    if not present:
        return rep

    px = df[present].apply(pd.to_numeric, errors="coerce")

    rep.sections.append(_section_schema(px, rep))
    rep.sections.append(_section_scale(px, rep))
    rep.sections.append(_section_ohlc(df, px, rep, top_n))
    rep.sections.append(_section_outliers(px, ts, rep, return_high_move, top_n))
    rep.sections.append(_section_time(ts, rep, timeframe, top_n))

    if not any(f.level != _OK for f in rep.findings):
        rep.findings.append(Finding(_OK, "all_clear", "tutti i controlli superati"))
    return rep


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _hdr(title: str) -> str:
    return f"─ {title} " + "─" * max(0, 70 - len(title))


def _extract_timestamps(df: pd.DataFrame, timestamp_col: str) -> Optional[pd.Series]:
    """Return a datetime Series (RangeIndex) from the index or a column, or None."""
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, dtype="datetime64[ns]").reset_index(drop=True)
    if timestamp_col in df.columns:
        try:
            return pd.to_datetime(df[timestamp_col]).reset_index(drop=True)
        except Exception:
            return None
    return None


def _section_overview(df, ts, present, price_cols, timeframe) -> str:
    lines = [_hdr("PANORAMICA")]
    lines.append(f"  barre              : {len(df)}")
    lines.append(f"  colonne prezzo     : {present or '— NESSUNA —'}")
    miss = [c for c in price_cols if c not in df.columns]
    if miss:
        lines.append(f"  prezzo MANCANTI    : {miss}")
    if ts is not None and len(ts):
        span = ts.max() - ts.min()
        lines.append(f"  intervallo date    : {ts.min()}  →  {ts.max()}")
        lines.append(f"  durata             : {span.days} giorni")
        cad = _infer_cadence(ts)
        if cad is not None:
            lines.append(f"  cadenza inferita   : {cad}")
    else:
        lines.append("  datetime           : ASSENTE (controlli temporali saltati)")
    if timeframe:
        lines.append(f"  timeframe dichiar. : {timeframe}")
    return "\n".join(lines)


def _section_schema(px: pd.DataFrame, rep: DataQualityReport) -> str:
    lines = [_hdr("SCHEMA & VALORI MANCANTI")]
    lines.append(f"  {'colonna':<8}{'n_NaN':>10}{'non_num':>10}{'≤0':>8}{'inf':>8}")
    for c in px.columns:
        s = px[c]
        n_nan = int(s.isna().sum())
        n_nonpos = int((s <= 0).sum())
        n_inf = int(np.isinf(s.to_numpy(dtype="float64", na_value=np.nan)).sum())
        # "non numerico" = NaN introdotti dalla coercizione (valori non parsabili)
        lines.append(f"  {c:<8}{n_nan:>10}{'-':>10}{n_nonpos:>8}{n_inf:>8}")
        if n_nan:
            rep.findings.append(Finding(_WARN, "nan", f"{c}: {n_nan} valori NaN"))
        if n_nonpos:
            rep.findings.append(Finding(_FAIL, "nonpos", f"{c}: {n_nonpos} prezzi ≤ 0"))
        if n_inf:
            rep.findings.append(Finding(_FAIL, "inf", f"{c}: {n_inf} valori infiniti"))
    return "\n".join(lines)


def _section_scale(px: pd.DataFrame, rep: DataQualityReport) -> str:
    """Detect values living on different orders of magnitude (mixed scale)."""
    lines = [_hdr("SCALA / MAGNITUDINE")]
    vals = px.to_numpy(dtype="float64", na_value=np.nan).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        lines.append("  nessun valore positivo finito")
        return "\n".join(lines)

    lines.append(f"  {'colonna':<8}{'min':>14}{'mediana':>14}{'max':>14}{'max/min':>12}")
    for c in px.columns:
        s = px[c].to_numpy(dtype="float64", na_value=np.nan)
        s = s[np.isfinite(s) & (s > 0)]
        if s.size == 0:
            continue
        ratio = s.max() / s.min()
        lines.append(f"  {c:<8}{s.min():>14.5g}{np.median(s):>14.5g}{s.max():>14.5g}{ratio:>12.4g}")

    # Magnitude buckets: floor(log10) of every price value
    mags = np.floor(np.log10(vals)).astype(int)
    buckets = pd.Series(mags).value_counts().sort_index()
    shares = (buckets / buckets.sum())
    occupied = shares[shares >= 0.005]  # ≥ 0.5% of values
    lines.append("  distribuzione magnitudini (quota barre per 10^k):")
    for k, sh in shares.items():
        bar = "█" * int(round(sh * 30))
        lines.append(f"     10^{k:<3d} {sh*100:6.2f}%  {bar}")

    # Bucket adiacenti (span 1) = serie che attraversa una potenza di 10
    # (es. EURUSD 0.96→1.20): NORMALE. Solo un divario ≥ 2 ordini segnala
    # un vero mix di formato (es. decimali ×10000).
    bucket_span = int(occupied.index.max() - occupied.index.min()) if len(occupied) else 0
    if len(occupied) >= 2 and bucket_span >= 2:
        worst = ", ".join(f"10^{k} ({shares[k]*100:.1f}%)" for k in occupied.index)
        rep.findings.append(Finding(
            _FAIL, "scale_mixed",
            f"scala MISTA: i prezzi vivono su magnitudini distanti {bucket_span} ordini "
            f"[{worst}] — probabile bug di formato (es. valori ×10000 mescolati a decimali)",
        ))
    elif len(occupied) >= 2:
        lines.append("  (i prezzi attraversano una potenza di 10 — normale, non un mix di scala)")
    return "\n".join(lines)


def _section_ohlc(df, px: pd.DataFrame, rep: DataQualityReport, top_n: int) -> str:
    """Internal OHLC coherence: high=row max, low=row min, close/open in range."""
    lines = [_hdr("COERENZA OHLC")]
    need = {"open", "high", "low", "close"}
    if not need.issubset(set(px.columns)):
        lines.append(f"  saltata: servono tutte le colonne {sorted(need)}")
        return "\n".join(lines)

    o, h, l, c = px["open"], px["high"], px["low"], px["close"]
    row_max = px[["open", "high", "low", "close"]].max(axis=1)
    row_min = px[["open", "high", "low", "close"]].min(axis=1)

    hl = (h < l)
    high_not_max = (h < row_max - 0)        # high should equal the row max
    low_not_min = (l > row_min)
    close_oob = (c > h) | (c < l)
    open_oob = (o > h) | (o < l)
    # range intrabar molto ampio: sospetto (può indicare un mix di scala dentro
    # la barra) ma su asset volatili è possibile → WARN, non errore strutturale.
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_ratio = (row_max / row_min).to_numpy()
    bar_scalemix = pd.Series(np.isfinite(bar_ratio) & (bar_ratio > 2.0), index=px.index)

    def _count(mask) -> int:
        return int(pd.Series(mask).fillna(False).sum())

    checks = [
        ("high < low",                 hl,            _FAIL, "ohlc_hl"),
        ("high non è il massimo",      high_not_max,  _FAIL, "ohlc_high"),
        ("low non è il minimo",        low_not_min,   _FAIL, "ohlc_low"),
        ("close fuori da [low,high]",  close_oob,     _FAIL, "ohlc_close"),
        ("open fuori da [low,high]",   open_oob,      _WARN, "ohlc_open"),
        ("range intrabar >100% (max/min>2)", bar_scalemix, _WARN, "ohlc_barscale"),
    ]
    lines.append(f"  {'controllo':<34}{'violazioni':>12}")
    for label, mask, level, code in checks:
        k = _count(mask)
        lines.append(f"  {label:<34}{k:>12}")
        if k:
            rep.findings.append(Finding(level, code, f"OHLC: {label} in {k} barre"))

    # esempi delle violazioni più gravi
    any_bad = pd.Series(False, index=px.index)
    for _, mask, _, _ in checks:
        any_bad = any_bad | pd.Series(mask).fillna(False).reindex(px.index, fill_value=False)
    if any_bad.any():
        idx = list(np.where(any_bad.to_numpy())[0])[:top_n]
        lines.append("  esempi:")
        for i in idx:
            row = df.iloc[i]
            lines.append(
                f"     #{i}  O={_fmt(row.get('open'))} H={_fmt(row.get('high'))} "
                f"L={_fmt(row.get('low'))} C={_fmt(row.get('close'))}"
            )
    return "\n".join(lines)


def _section_outliers(px, ts, rep: DataQualityReport, hi_move: float, top_n: int) -> str:
    """Robust (MAD-based) outliers on close-to-close returns and bar range."""
    lines = [_hdr("OUTLIER")]
    if "close" not in px.columns:
        lines.append("  saltata: manca 'close'")
        return "\n".join(lines)

    c = px["close"].to_numpy(dtype="float64", na_value=np.nan)
    r = np.full_like(c, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = c[1:] / c[:-1] - 1.0
    finite = r[np.isfinite(r)]
    if finite.size:
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med))) or float("nan")
        scale = 1.4826 * mad if mad and np.isfinite(mad) else float("nan")
        z = np.abs(r - med) / scale if np.isfinite(scale) and scale > 0 else np.zeros_like(r)
        robust_out = np.isfinite(r) & (z > 8.0)
        abs_out = np.isfinite(r) & (np.abs(r) > hi_move)
        lines.append(f"  rendimenti close→close: "
                     f"σ={np.nanstd(finite)*100:.3f}%  |mediano|={np.median(np.abs(finite))*100:.3f}%  "
                     f"max|r|={np.nanmax(np.abs(finite))*100:.4g}%")
        n_rob = int(robust_out.sum())
        n_abs = int(abs_out.sum())
        lines.append(f"  outlier robusti (MAD z>8) : {n_rob}")
        lines.append(f"  mosse |r|>{hi_move*100:.0f}%      : {n_abs}")
        worst_idx = np.argsort(-np.abs(np.where(np.isfinite(r), r, 0)))[:top_n]
        if n_rob or n_abs:
            lines.append("  peggiori rendimenti:")
            for i in worst_idx:
                if not np.isfinite(r[i]) or r[i] == 0:
                    continue
                when = f" @ {ts.iloc[i].date()}" if ts is not None and i < len(ts) else ""
                lines.append(f"     #{i}{when}  {c[i-1]:.5g} → {c[i]:.5g}  ({r[i]*100:+.4g}%)")
        if n_rob:
            rep.findings.append(Finding(
                _WARN, "ret_outlier",
                f"{n_rob} rendimenti anomali (MAD z>8) — possibili print errati o salti di scala"))
        if n_abs:
            rep.findings.append(Finding(
                _WARN, "ret_extreme",
                f"{n_abs} barre con |rendimento| > {hi_move*100:.0f}% — estremo; reale su asset "
                f"molto volatili (es. crypto), ma da verificare se non atteso"))

    # range intrabar relativo
    if {"high", "low"}.issubset(px.columns):
        h = px["high"].to_numpy(dtype="float64", na_value=np.nan)
        l = px["low"].to_numpy(dtype="float64", na_value=np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            rng = (h - l) / c
        fr = rng[np.isfinite(rng)]
        if fr.size:
            lines.append(f"  range (H-L)/C          : "
                         f"mediano={np.median(fr)*100:.3f}%  max={np.nanmax(fr)*100:.4g}%")
    return "\n".join(lines)


def _section_time(ts, rep: DataQualityReport, timeframe, top_n: int) -> str:
    lines = [_hdr("CONTINUITÀ TEMPORALE")]
    if ts is None or len(ts) < 2:
        lines.append("  saltata: datetime assente o insufficiente")
        rep.findings.append(Finding(_WARN, "no_time", "nessuna colonna datetime: barre mancanti non verificabili"))
        return "\n".join(lines)

    n_dup = int(ts.duplicated().sum())
    diffs = ts.diff().dropna()
    n_ooo = int((diffs < pd.Timedelta(0)).sum())  # out of order
    pos = diffs[diffs > pd.Timedelta(0)]
    lines.append(f"  duplicati timestamp    : {n_dup}")
    lines.append(f"  fuori ordine           : {n_ooo}")
    if len(pos):
        modal = pos.mode().iloc[0]
        lines.append(f"  passo modale           : {modal}")
        # gap = passi > modale (potenziali barre mancanti)
        gap_counts = pos.value_counts().sort_index()
        lines.append("  distribuzione passi (top 6):")
        for delta, k in gap_counts.head(6).items():
            lines.append(f"     {str(delta):<22} {k}")
        big = pos[pos > modal]
        # per dati daily, salti di 3 giorni (weekend) sono normali → soglia 1.5×modale escluso weekend
        suspicious = pos[pos > max(modal * 3, modal + pd.Timedelta(days=2))]
        lines.append(f"  passi > modale         : {int((pos > modal).sum())}")
        lines.append(f"  salti sospetti         : {len(suspicious)}")
        if len(suspicious):
            order = suspicious.sort_values(ascending=False).head(top_n)
            lines.append("  salti più grandi:")
            for i, delta in order.items():
                prev = ts.iloc[i - 1].date() if i - 1 < len(ts) else "?"
                lines.append(f"     {prev} → {ts.iloc[i].date()}  ({delta})")
            rep.findings.append(Finding(
                _WARN, "gaps", f"{len(suspicious)} salti temporali oltre la cadenza — possibili barre mancanti"))
    if n_dup:
        rep.findings.append(Finding(_FAIL, "dup_time", f"{n_dup} timestamp duplicati"))
    if n_ooo:
        rep.findings.append(Finding(_FAIL, "out_of_order", f"{n_ooo} barre fuori ordine cronologico"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _infer_cadence(ts: pd.Series) -> Optional[str]:
    d = ts.diff().dropna()
    d = d[d > pd.Timedelta(0)]
    if d.empty:
        return None
    return str(d.mode().iloc[0])


def _fmt(v) -> str:
    try:
        return f"{float(v):.5g}"
    except (TypeError, ValueError):
        return str(v)
