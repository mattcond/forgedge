"""Tests for the ``forge`` end-to-end orchestrator."""
import numpy as np
import pandas as pd

from forgedge import (
    AlphaConfig,
    DiscoveryConfig,
    ForgeResult,
    RuleDiscoveryConfig,
    RuleRegistry,
    forge,
    forge_multi,
)
from forgedge.event_discovery.models import GateParams
from forgedge.market_context.models import REGIME_COL
from forgedge.rule_discovery.models import GridSpec, WalkForwardConfig


# A deliberately strict, single-component Event Discovery config: the table
# spans enough calendar months for the default gate, and these thresholds keep
# the candidate count (and therefore the Rule Discovery work) small so the
# end-to-end tests stay fast.
_FAST_ED_CONFIG = DiscoveryConfig(
    max_and_components=1,
    gate_params=GateParams(min_act=120, min_months=8, max_conc=0.6, min_tpm=2.0),
)

# Minimal Rule Discovery config — a single-cell grid and a light, non-reoptimised
# walk-forward — so each Modulo 3 run is cheap.  The verdicts are not the point
# of these orchestration tests; the wiring is.
_FAST_RD_CONFIG = RuleDiscoveryConfig(
    grid=GridSpec(buy_drop_pct=[0.0], buy_delay_bar=[0]),
    walk_forward=WalkForwardConfig(n_splits=2, reoptimise=False),
)


def _ohlc_kpi_table(n: int = 2600, seed: int = 7) -> pd.DataFrame:
    """KPI table with OHLCV + a mean-reverting feature, enough for every module.

    Sampled at 4H so ~2,600 bars already span more than a year (clearing the
    8-month gate) while keeping the pipeline fast.  Low ``feat`` predicts a
    positive next-bar return, so Event Discovery finds ``feat < x`` events and
    the full pipeline has something to chew on through Rule Discovery (which
    needs the high/low columns for its backtest).
    """
    rng = np.random.default_rng(seed)
    feat = rng.uniform(0.0, 1.0, n)
    k = 0.02
    noise = rng.normal(0.0, 0.004, n)
    r = np.empty(n)
    r[0] = 0.0
    r[1:] = -k * (feat[:-1] - 0.5) + noise[1:]
    close = 100.0 * np.exp(np.cumsum(r))
    op = close * (1 + rng.normal(0.0, 0.001, n))
    high = np.maximum(op, close) * (1 + np.abs(rng.normal(0.0, 0.002, n)))
    low = np.minimum(op, close) * (1 - np.abs(rng.normal(0.0, 0.002, n)))
    return pd.DataFrame(
        {
            "open_dt": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "open": op,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.abs(rng.normal(1e6, 1e5, n)),
            "feat": feat,
        }
    )


class TestForge:
    def test_end_to_end_runs_every_module(self):
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi,
            ticker="BTCUSDC",
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
        )

        assert isinstance(result, ForgeResult)
        assert result.ticker == "BTCUSDC"
        # Modulo 0 enriched the table.
        assert REGIME_COL in result.enriched.columns
        assert result.market_context is not None
        # Modulo 1 produced candidates.
        assert len(result.candidates) > 0
        # event_frame is the Event Discovery post-pipeline frame.
        assert result.event_frame is result.event_discovery.df
        # Modulo 2 produced one contract per candidate fed in.
        assert len(result.contracts) == len(result.candidates)
        assert all(c in result.contracts for c in result.promoted)
        # Modulo 3 ran once per promoted contract.
        assert len(result.rule_responses) == len(result.promoted)
        for contract, response in result.rule_responses:
            assert response.verdict in {"EDGE", "PARTIAL-EDGE", "NON-EDGE"}
            assert contract.alpha_id == response.alpha_id
        # Modulo 4 — Rule Registry built from this run's tradeable rules.
        assert isinstance(result.registry, RuleRegistry)
        assert len(result.registry.documents) == len(result.submissions())

    def test_summary_carries_rule_verdict(self):
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi,
            asset="TEST",
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
        )
        summary = result.summary()
        assert "rule_verdict" in summary.columns
        assert len(summary) == len(result.contracts)

    def test_skips_market_context_when_regime_present(self):
        kpi = _ohlc_kpi_table()
        # Enrich once (no Rule Discovery needed to obtain the regime columns).
        enriched = forge(
            kpi, event_discovery_config=_FAST_ED_CONFIG, run_rule_discovery=False
        ).enriched

        result = forge(
            enriched,
            asset="TEST",
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        # Market Context is skipped when the table is already enriched.
        assert result.market_context is None
        assert REGIME_COL in result.enriched.columns

    def test_run_market_context_false_skips_module_zero(self):
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi,
            run_market_context=False,
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        assert result.market_context is None
        assert REGIME_COL not in result.enriched.columns

    def test_run_rule_discovery_false_stops_after_alpha(self):
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi, event_discovery_config=_FAST_ED_CONFIG, run_rule_discovery=False
        )
        assert result.rule_responses == []
        # Modules 0–2 still ran fully.
        assert len(result.candidates) > 0
        assert result.alpha_discovery is not None
        # Modulo 4 is skipped when Rule Discovery did not run.
        assert result.registry is None

    def test_run_registry_false_skips_module_four(self):
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi,
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
            run_registry=False,
        )
        assert result.registry is None
        # Rule Discovery still ran.
        assert len(result.rule_responses) == len(result.promoted)

    def test_default_alpha_config_carries_metadata(self):
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi,
            asset="MYCOIN",
            timeframe="4H",
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        assert result.alpha_discovery.config.asset == "MYCOIN"
        assert result.alpha_discovery.config.timeframe == "4H"

    def test_explicit_alpha_config_is_respected(self):
        kpi = _ohlc_kpi_table()
        cfg = AlphaConfig(asset="EXPLICIT", timeframe="1D")
        result = forge(
            kpi,
            asset="IGNORED",
            alpha_config=cfg,
            event_discovery_config=_FAST_ED_CONFIG,
            run_rule_discovery=False,
        )
        assert result.alpha_discovery.config.asset == "EXPLICIT"
        assert result.alpha_discovery.config.timeframe == "1D"

    def test_edges_and_validated_rules_are_consistent(self):
        kpi = _ohlc_kpi_table()
        result = forge(
            kpi,
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
        )
        for contract, response in result.edges():
            assert response.is_edge
        for response in result.validated_rules():
            assert response.validated_rule is not None


class TestForgeMulti:
    def test_pools_tickers_into_one_cross_ticker_registry(self):
        frames = {
            "BTCUSDC": _ohlc_kpi_table(seed=7),
            "ETHUSDC": _ohlc_kpi_table(seed=11),
        }
        results, registry = forge_multi(
            frames,
            event_discovery_config=_FAST_ED_CONFIG,
            rule_discovery_config=_FAST_RD_CONFIG,
        )

        assert set(results) == {"BTCUSDC", "ETHUSDC"}
        # Per-ticker registries are skipped in favour of the pooled one.
        for res in results.values():
            assert res.registry is None
            assert res.ticker in frames
        # The pooled registry knows about every ticker's frame for cross-ticker.
        assert isinstance(registry, RuleRegistry)
        assert set(registry.frames) == {"BTCUSDC", "ETHUSDC"}
        # Every ingested document traces back to a session ticker.
        for doc in registry.documents:
            assert doc.source_ticker in frames
