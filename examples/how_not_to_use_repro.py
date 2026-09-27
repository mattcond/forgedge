"""Reproductions for ``docs/analysis/how_not_to_use_forge.md``.

Six ways to misuse ``forgedge`` that raise **no exception and no return-value
error** — the pipeline runs to completion and hands back a well-formed
object, but the numbers inside it are silently wrong or misleading. Every
section below is a self-contained, runnable proof: construct the misuse,
construct the correct usage on the same inputs, and print the measured
difference. Nothing here is a bug in the sense of "should be fixed" — each
one is either a deliberate default (documented in the code) or an
architectural consequence of session-parameter resolution — but all six are
invisible unless you go looking for them, which is exactly what makes them
worth a dedicated document instead of a docstring.

Usage
-----
    python examples/how_not_to_use_repro.py [section ...]

Sections (default: all): ``hourly_fallback``, ``grid_delay``,
``target_optimizer``, ``column_naming``, ``monitoring``, ``registry_filter``.

The dataset defaults to ``tests/fixtures/ADA_1D_TRAIN.parquet``; override
with ``FORGEDGE_PARQUET=/path/to/kpi.parquet`` for ``hourly_fallback``,
``column_naming`` and ``monitoring`` (the sections that touch real data).
"""
from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

DATA_PATH = os.environ.get("FORGEDGE_PARQUET", "tests/fixtures/ADA_1D_TRAIN.parquet")


def _kpi() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def section_hourly_fallback() -> None:
    """Constructing a config by hand keeps the HOURLY bar-counting calibration
    even when ``timeframe="1D"`` is written on the object -- only a real
    ``PipelineContext`` built by ``forge()`` (or ``PipelineContext.from_frame``)
    rescales it. No exception; every module's constructor resolves the config
    it's given, silently, against whatever context (or lack of one) it gets.
    """
    from forgedge import AlphaConfig
    from forgedge.resolver import PipelineContext, resolve_config
    from forgedge.rule_discovery.models import RuleDiscoveryConfig

    print("--- hourly_fallback ---")
    cfg = AlphaConfig(asset="ADA", timeframe="1D")
    print("AlphaConfig.timeframe as written by the caller:", cfg.timeframe)

    resolved_standalone = resolve_config(cfg, "alpha")  # ctx=None -> default PipelineContext(timeframe="1H")
    print("horizon_grid resolved with NO PipelineContext (standalone AlphaDiscovery(...)):",
          resolved_standalone.horizon_grid)

    ctx = PipelineContext.from_frame(_kpi(), timeframe="1D")
    resolved_daily = resolve_config(AlphaConfig(asset="ADA", timeframe="1D"), "alpha", ctx)
    print("horizon_grid resolved WITH the real daily PipelineContext (what forge() builds):",
          resolved_daily.horizon_grid)

    rdc_standalone = resolve_config(RuleDiscoveryConfig(), "rule_discovery")
    rdc_daily = resolve_config(RuleDiscoveryConfig(), "rule_discovery", ctx)
    print("base_params.target_h / buy_delay_bar, no context:  ",
          rdc_standalone.base_params.target_h, "/", rdc_standalone.base_params.buy_delay_bar)
    print("base_params.target_h / buy_delay_bar, daily context:",
          rdc_daily.base_params.target_h, "/", rdc_daily.base_params.buy_delay_bar)
    print("Same config text, timeframe='1D' both times -- 24-bar vs 10-bar horizon,")
    print("6-bar vs 1-bar fill delay, depending only on which entry point built it.\n")


def section_grid_delay() -> None:
    """GridSpec auto-fans buy_drop_pct/sell_pct/target_h around the seed
    BacktestParams, but never buy_delay_bar -- it stays pinned to the single
    base value in every grid cell unless the caller sets it explicitly.
    """
    from forgedge.rule_discovery.grid import build_grid
    from forgedge.rule_discovery.models import GridSpec, BacktestParams

    print("--- grid_delay ---")
    base = BacktestParams(buy_drop_pct=0.01, sell_pct=0.03, target_h=12, buy_delay_bar=3)
    grid = build_grid(GridSpec(), base)
    n_cells = len(grid.buy_drop_pct) * len(grid.sell_pct) * len(grid.target_h) * len(grid.buy_delay_bar)
    print("buy_drop_pct fan:", grid.buy_drop_pct)
    print("sell_pct fan:    ", grid.sell_pct)
    print("target_h fan:    ", grid.target_h)
    print("buy_delay_bar:   ", grid.buy_delay_bar, "<- single value in all", n_cells, "grid cells\n")


def section_target_optimizer() -> None:
    """TargetOptimizer.discover_alpha() always overwrites fixed_target /
    target_mode / trend_sma_mult on whatever AlphaConfig the caller passes
    it -- even if the caller explicitly set those fields to something else.
    """
    from unittest.mock import MagicMock
    from forgedge.target_optimizer import TargetOptimizer, TargetConfig
    from forgedge.alpha_discovery.models import AlphaConfig

    print("--- target_optimizer ---")
    optimizer_target = TargetConfig(horizon=12, min_return=0.02, side="long", target_mode="PROJ")
    my_own_target = TargetConfig(horizon=48, min_return=0.10, side="short", target_mode="ABS")

    opt = TargetOptimizer.__new__(TargetOptimizer)  # bypass run(); only discover_alpha's config handling is under test
    opt.target_cfg = optimizer_target
    opt._ed = MagicMock(df=None)
    opt._candidates = [MagicMock()]

    caller_cfg = AlphaConfig(fixed_target=my_own_target, target_mode="ABS", trend_sma_mult=3.5)
    print("caller_cfg.fixed_target BEFORE:", caller_cfg.fixed_target.horizon, "h /", caller_cfg.fixed_target.side)
    try:
        opt.discover_alpha(config=caller_cfg)
    except Exception:
        pass  # the mutation below already happened before AlphaDiscovery.run() ever touched the fake data
    print("caller_cfg.fixed_target AFTER: ", caller_cfg.fixed_target.horizon, "h /", caller_cfg.fixed_target.side)
    print("caller_cfg.fixed_target is optimizer_target (not my_own_target):",
          caller_cfg.fixed_target is optimizer_target, "\n")


def section_column_naming() -> None:
    """A custom indicator column whose name doesn't match {base}_{indicator}_{period}
    is still used standalone but is never paired into an arity-2 (ratio/spread)
    candidate with anything else.
    """
    from forgedge.event_discovery.feature_generator import parse_feature, FeatureGenerator
    from forgedge.event_discovery.classifier import TypeClassifier

    print("--- column_naming ---")
    kpi = _kpi().reset_index(drop=True)
    ema9 = kpi["close"].ewm(span=9).mean()
    kpi["my_custom_signal"] = ema9   # identical values, non-conforming name
    kpi["close_ema_9"] = ema9        # identical values, conforming name

    print("parse_feature('my_custom_signal') ->", parse_feature("my_custom_signal"))
    print("parse_feature('close_ema_9')      ->", parse_feature("close_ema_9"))

    classifications = TypeClassifier().fit(kpi)
    _extended, meta = FeatureGenerator().generate(kpi, classifications)
    n_custom = sum(1 for d in meta.values() if d.arity >= 2 and "my_custom_signal" in d.source_cols)
    n_compliant = sum(1 for d in meta.values() if d.arity >= 2 and "close_ema_9" in d.source_cols)
    print(f"arity>=2 features built from 'my_custom_signal': {n_custom}")
    print(f"arity>=2 features built from 'close_ema_9' (same values): {n_compliant}\n")


def section_monitoring() -> None:
    """Feeding Alpha Discovery ONLY the new bars (instead of train_df
    concatenated with new_bars_df) to "check if a published edge still
    holds" degrades direction to 'undetermined' far more often -- the fix is
    to use RuleDiscovery, not to re-run AlphaDiscovery at all, but even the
    frame-construction mistake alone is measurable.
    """
    from forgedge import EventDiscovery, DiscoveryConfig, AlphaDiscovery, AlphaConfig, MarketContext, MarketContextConfig

    print("--- monitoring ---")
    kpi = _kpi().reset_index(drop=True)
    split = int(len(kpi) * 0.75)
    train_df = kpi.iloc[:split].reset_index(drop=True)
    new_bars_df = kpi.iloc[split:].reset_index(drop=True)

    mc = MarketContext(train_df, config=MarketContextConfig()).run()
    candidates = EventDiscovery(mc, config=DiscoveryConfig(train_ratio=1.0)).run()
    alpha_cfg = AlphaConfig(asset="ADA", timeframe="1D", train_ratio=1.0)
    print(f"train_df={len(train_df)} rows, new_bars_df={len(new_bars_df)} rows, "
          f"{len(candidates)} candidates\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        contracts_wrong = AlphaDiscovery(new_bars_df, candidates, alpha_cfg).run()
    undetermined_wrong = sum(1 for c in contracts_wrong if c.direction == "undetermined")
    print(f"AlphaDiscovery(new_bars_df ALONE):        {len(caught)} UserWarning(s), "
          f"{undetermined_wrong}/{len(contracts_wrong)} undetermined")

    eval_df = pd.concat([train_df, new_bars_df]).drop_duplicates("open_dt").reset_index(drop=True)
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        contracts_right = AlphaDiscovery(eval_df, candidates, alpha_cfg).run()
    undetermined_right = sum(1 for c in contracts_right if c.direction == "undetermined")
    print(f"AlphaDiscovery(pd.concat([train_df, new_bars_df])): {len(caught2)} UserWarning(s), "
          f"{undetermined_right}/{len(contracts_right)} undetermined\n")


def section_registry_filter() -> None:
    """RuleRegistry.flat_table()'s own apply_filters kwarg defaults to False,
    AND RegistryConfig's export_duplicates/export_non_generic both default
    to True -- duplicates and non-generic rules stay in a plain flat_table()
    call on both axes at once.
    """
    from forgedge.rule_registry.models import RuleDocument, RegistryConfig
    from forgedge.rule_registry.export import flat_table

    print("--- registry_filter ---")

    def make_doc(rule_id, is_duplicate, is_generic):
        return RuleDocument(
            rule_id=rule_id, expression=f"close_rsi_14 < 30 [{rule_id}]",
            source_ticker="ADAUSDC", source_alpha_id=f"ALPHA-{rule_id}",
            verdict="EDGE", grade="A",
            activation_idx=[1, 2, 3], activation_dates=["2024-01-01", "2024-01-02", "2024-01-03"],
            gains=[0.01, -0.02, 0.03],
            params={"buy_drop_pct": 0.01, "sell_pct": 0.03, "target_h": 12, "buy_delay_bar": 1},
            stats={"pf": 1.8, "win_rate": 0.55, "total_trades": 40},
            regime={"dominant": "BULL"},
            is_duplicate=is_duplicate, is_generic=is_generic,
        )

    docs = [
        make_doc("R001", is_duplicate=False, is_generic=True),
        make_doc("R002", is_duplicate=True, is_generic=True),
        make_doc("R003", is_duplicate=False, is_generic=False),
    ]
    print(f"flat_table(docs)                                        -> {len(flat_table(docs))} rows")
    print(f"flat_table(docs, config=RegistryConfig())               -> "
          f"{len(flat_table(docs, config=RegistryConfig()))} rows (defaults keep both)")
    strict = RegistryConfig(export_duplicates=False, export_non_generic=False)
    print(f"flat_table(docs, config=RegistryConfig(export_duplicates=False,\n"
          f"                                        export_non_generic=False)) -> "
          f"{len(flat_table(docs, config=strict))} row(s)\n")


_SECTIONS = {
    "hourly_fallback": section_hourly_fallback,
    "grid_delay": section_grid_delay,
    "target_optimizer": section_target_optimizer,
    "column_naming": section_column_naming,
    "monitoring": section_monitoring,
    "registry_filter": section_registry_filter,
}

if __name__ == "__main__":
    requested = sys.argv[1:] or list(_SECTIONS)
    for name in requested:
        _SECTIONS[name]()
