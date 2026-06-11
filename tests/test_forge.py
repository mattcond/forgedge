"""Tests for the ``forge`` end-to-end orchestrator."""
import numpy as np
import pandas as pd

from forgedge import AlphaConfig, DiscoveryConfig, ForgeResult, forge
from forgedge.event_discovery.models import GateParams
from forgedge.market_context.models import REGIME_COL


# A deliberately strict, single-component Event Discovery config: the table
# spans enough calendar months for the default gate, and these thresholds keep
# the candidate count (and therefore the Rule Discovery work) small so the
# end-to-end tests stay fast.
_FAST_ED_CONFIG = DiscoveryConfig(
    max_and_components=1,
    gate_params=GateParams(min_act=40, min_months=8, max_conc=0.6, min_tpm=1.0),
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
            kpi, asset="TEST", timeframe="4H", event_discovery_config=_FAST_ED_CONFIG
        )

        assert isinstance(result, ForgeResult)
        # Modulo 0 enriched the table.
        assert REGIME_COL in result.enriched.columns
        assert result.market_context is not None
        # Modulo 1 produced candidates.
        assert len(result.candidates) > 0
        # Modulo 2 produced one contract per candidate fed in.
        assert len(result.contracts) == len(result.candidates)
        assert all(c in result.contracts for c in result.promoted)
        # Modulo 3 ran once per promoted contract.
        assert len(result.rule_responses) == len(result.promoted)
        for contract, response in result.rule_responses:
            assert response.verdict in {"EDGE", "PARTIAL-EDGE", "NON-EDGE"}
            assert contract.alpha_id == response.alpha_id

    def test_summary_carries_rule_verdict(self):
        kpi = _ohlc_kpi_table()
        result = forge(kpi, asset="TEST", event_discovery_config=_FAST_ED_CONFIG)
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
        result = forge(kpi, event_discovery_config=_FAST_ED_CONFIG)
        for contract, response in result.edges():
            assert response.is_edge
        for response in result.validated_rules():
            assert response.validated_rule is not None
