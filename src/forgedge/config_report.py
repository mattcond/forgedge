"""``config_report`` — what the pipeline will run with, and whether it holds up.

The sibling of :func:`forgedge.summary_report`.  That one validates the *data*;
this one validates the *configuration*::

    summary_report(kpi)                        → are the data usable?
    config_report(disc, alpha, rd, registry)   → what will I run with, and is it coherent?

Two questions, one output, because they are only useful side by side.  The
resolved value of a field and the finding that says it cannot be satisfied
belong on the same screen — reading them from two calls means cross-referencing
by hand exactly the two things that explain each other.

Why this exists
---------------
The pipeline had no notion of an *internally inconsistent configuration*.  Two
individually reasonable values can be jointly impossible, and when they are the
result surfaces as **rejections** — indistinguishable from "the signal is bad",
which is precisely what the caller was trying to measure.  Issue #173 is one
instance: a permissive ``min_tpm`` upstream and an untouched ``min_train_months``
downstream eliminate every candidate at the first fold, in silence.

And until now there was no way to answer the plainer question either — *with
what configuration am I about to run?* — because a class default is
indistinguishable from a choice.

The guarantee
-------------
``config_report()`` and :func:`forgedge.forge` call the **same resolver**, so
what the report shows is, by construction, what the pipeline will execute.  Not
two implementations to keep in step: one.

Severity
--------
``FAIL`` is reserved for configurations that make a stage *structurally
incapable* of producing a verdict — the run cannot tell you anything, so under
``forge(strict=True)`` (the default) it does not start.  Everything else is
``WARN``: an incoherence worth knowing about, not an impossibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .resolver import (
    PipelineContext,
    ResolutionTrace,
    Violation,
    collect_context,
    resolve,
)
from .summary_report import Finding

__all__ = ["config_report", "ConfigReport"]

_OK, _WARN, _FAIL = "OK", "WARN", "FAIL"
_MARK = {_OK: "✓", _WARN: "⚠", _FAIL: "✗"}
_LABEL = {_OK: "OK", _WARN: "AVVISO", _FAIL: "CRITICO"}
_RANK = {_OK: 0, _WARN: 1, _FAIL: 2}


@dataclass
class ConfigReport:
    """Structured result of :func:`config_report`.

    Attributes
    ----------
    trace : ResolutionTrace
        Every field the resolver derived — ``default → resolved``, with the
        rule that fired and the inputs it read.
    findings : list[Finding]
        Constraint violations, in the vocabulary :func:`summary_report` already
        uses.  ``findings`` is the authoritative verdict list.
    configs : dict
        The **resolved** configs — literally the objects the pipeline will run
        with, because they came out of the same resolver call.
    context : PipelineContext
        The session facts the resolution was performed against.
    """

    trace: ResolutionTrace
    findings: List[Finding] = field(default_factory=list)
    configs: Dict[str, Any] = field(default_factory=dict)
    context: Optional[PipelineContext] = None

    # ------------------------------------------------------------------
    @property
    def worst(self) -> str:
        """Highest severity across all findings (``OK`` when empty)."""
        return max((f.level for f in self.findings), key=lambda l: _RANK[l], default=_OK)

    @property
    def has_critical(self) -> bool:
        """True when a stage cannot produce a verdict with this configuration."""
        return any(f.level == _FAIL for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.level == _WARN for f in self.findings)

    def one_line(self) -> str:
        """Compact single-line summary of the FAIL/WARN findings."""
        bad = [f for f in self.findings if f.level != _OK]
        if not bad:
            return "configurazione coerente"
        return "; ".join(f"[{f.level}] {f.code}: {f.message}" for f in bad)

    def to_text(self) -> str:
        """Full report — the resolution trace, then the diagnostics."""
        lines = ["═" * 72, "  FORGE — CONFIG REPORT", "═" * 72]
        lines.append(self.trace.to_text())
        lines.append("─" * 72)
        lines.append("  DIAGNOSTICA")
        lines.append("─" * 72)
        if not self.findings:
            lines.append(f"  {_MARK[_OK]} [{_LABEL[_OK]:<7}] nessuna incoerenza rilevata")
        else:
            for f in sorted(self.findings, key=lambda f: -_RANK[f.level]):
                lines.append(f"  {_MARK[f.level]} [{_LABEL[f.level]:<7}] {f.code}")
                for chunk in _wrap(f.message, 66):
                    lines.append(f"              {chunk}")
        n_crit = sum(1 for f in self.findings if f.level == _FAIL)
        n_warn = sum(1 for f in self.findings if f.level == _WARN)
        lines.append("─" * 72)
        if n_crit:
            summary = (f"{n_crit} incoerenza/e critica/he e {n_warn} avvisi — con "
                       f"strict=True il run non parte")
        elif n_warn:
            summary = f"{n_warn} avvisi — la pipeline gira, ma vale la pena guardarli"
        else:
            summary = "configurazione coerente"
        lines.append(f"  {_MARK[self.worst]} {summary}")
        lines.append("═" * 72)
        return "\n".join(lines)


def _wrap(text: str, width: int) -> List[str]:
    """Minimal word wrap — no textwrap import for three lines of output."""
    out: List[str] = []
    line = ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def config_report(
    event_discovery: Any = None,
    alpha: Any = None,
    rule_discovery: Any = None,
    registry: Any = None,
    market_context: Any = None,
    *,
    ctx: Optional[PipelineContext] = None,
    kpi: Any = None,
    timeframe: str = "1H",
    verbose: bool = False,
) -> ConfigReport:
    """Resolve a configuration and check it, without running anything.

    Parameters
    ----------
    event_discovery, alpha, rule_discovery, registry, market_context
        The module configs, in pipeline order.  Any subset is fine — a
        constraint whose inputs are absent simply has nothing to say.
    ctx : PipelineContext, optional
        Session facts.  When omitted one is built from ``kpi``/``timeframe`` and
        seeded from whatever the configs set explicitly.
    kpi : pd.DataFrame, optional
        The KPI table.  Only its length and span are read, and **only by the
        checks** — resolution itself never looks at the data.  Without it the
        span-dependent checks stay silent, since they have nothing to compare
        against.
    timeframe : str
        Bar size, when no explicit context is given.
    verbose : bool
        Render every derivation's reason in ``to_text()``.

    Returns
    -------
    ConfigReport
        Advisory.  This function never raises, never warns and never mutates
        the configs it was handed — ``forge(strict=True)`` is what turns a
        ``FAIL`` into a stopped run.

    Examples
    --------
    >>> rep = config_report(disc_cfg, alpha_cfg, rd_cfg, kpi=kpi, timeframe="1D")
    >>> print(rep.to_text())
    >>> if rep.has_critical:
    ...     raise ValueError(rep.one_line())
    """
    bundle = {
        "market_context": market_context,
        "event_discovery": event_discovery,
        "alpha": alpha,
        "rule_discovery": rule_discovery,
        "registry": registry,
    }
    if ctx is None:
        base = (PipelineContext.from_frame(kpi, timeframe=timeframe)
                if kpi is not None else PipelineContext(timeframe=timeframe))
        ctx = collect_context(bundle, base)
    resolved, trace, violations = resolve(bundle, ctx)
    return ConfigReport(
        trace=trace,
        findings=[_as_finding(v) for v in violations],
        configs=resolved,
        context=ctx,
    )


def _as_finding(violation: Violation) -> Finding:
    """Adapt a resolver :class:`Violation` to the report vocabulary.

    ``Finding`` is reused verbatim from :mod:`forgedge.summary_report` rather
    than duplicated: the two reports say different things about different
    objects, but they should read identically.
    """
    return Finding(level=violation.level, code=violation.code, message=violation.message)
