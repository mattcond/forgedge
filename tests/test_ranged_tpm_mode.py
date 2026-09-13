"""Tests for ``GateParams.tpm_mode="ranged"`` (Consistency Gate, Event Discovery).

Formulas and empirical validation are frozen in
``docs/analysis/ranged_tpm_and_market_alignment_proposal.md`` §2 — this file
turns that spec into regression tests. Design recap:

- Default (``tpm_mode="floor"``) is byte-identical to today's behaviour.
- ``"ranged"`` (episode counting only) turns ``min_tpm`` into the centre of a
  band; the half-width is either derived from the session's own dispersion
  tolerance (``tpm_tolerance`` left at ``UNSET``) or supplied literally.
- ``n_episodes == 0`` is always rejected, independently of the band.
- ``_gate_pass`` is the single chokepoint both ``ConsistencyGate.evaluate``
  and ``ANDComposer`` call — parity between the two paths matters as much as
  the formula itself (fix #226).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from forgedge import EventDiscovery, MarketContext, UNSET
from forgedge.event_discovery.consistency_gate import (
    ConsistencyGate,
    _build_month_index,
    _eff_max_dispersion,
    _gate_pass,
    _tpm_band,
    _Z_95_TWO_SIDED,
)
from forgedge.event_discovery.discovery import DiscoveryConfig
from forgedge.event_discovery.models import GateParams
from forgedge.presets import forge_preset
from forgedge.resolver import PipelineContext, resolve

PARQUET = "tests/fixtures/ADA_1D_TRAIN.parquet"


# ---------------------------------------------------------------------------
# Formula-level tests
# ---------------------------------------------------------------------------

class TestTpmBandFormula:
    def test_default_gateparams_is_floor_mode(self):
        """No regression: unmodified callers get exactly today's behaviour."""
        p = GateParams()
        assert p.tpm_mode == "floor"
        assert p.tpm_tolerance is UNSET

    @pytest.mark.parametrize(
        "n_total_months, min_tpm, dispersion_margin",
        [(29, 1.0, 1.3), (43, 0.3, 1.05), (69, 2.0, 1.3)],
    )
    def test_derived_tolerance_matches_manual_formula(
        self, n_total_months, min_tpm, dispersion_margin,
    ):
        eff_max_dispersion = _eff_max_dispersion(n_total_months, dispersion_margin)
        p = GateParams(tpm_mode="ranged", min_tpm=min_tpm, dispersion_margin=dispersion_margin)
        lo, hi = _tpm_band(p, eff_max_dispersion, n_total_months)

        sigma = math.sqrt(eff_max_dispersion * min_tpm / n_total_months)
        expected_half_width = _Z_95_TWO_SIDED * sigma
        assert hi == pytest.approx(min_tpm + expected_half_width)
        assert lo == pytest.approx(max(0.0, min_tpm - expected_half_width))

    def test_documented_balanced_band_on_ada_fixture(self):
        """Pins the exact band reported in the doc's validation (§2.8) for
        ADA (29 months), so a future change to the formula or to
        `_chi2_ppf_095` is caught here, not only in the slow characterization
        test below."""
        eff_max_dispersion = _eff_max_dispersion(29, 1.3)
        p = GateParams(tpm_mode="ranged", min_tpm=1.0, dispersion_margin=1.3)
        lo, hi = _tpm_band(p, eff_max_dispersion, 29)
        assert lo == pytest.approx(0.496, abs=1e-3)
        assert hi == pytest.approx(1.504, abs=1e-3)

    def test_documented_sniper_band_on_ada_fixture(self):
        eff_max_dispersion = _eff_max_dispersion(29, 1.05)
        p = GateParams(tpm_mode="ranged", min_tpm=0.3, dispersion_margin=1.05)
        lo, hi = _tpm_band(p, eff_max_dispersion, 29)
        assert lo == pytest.approx(0.052, abs=1e-3)
        assert hi == pytest.approx(0.548, abs=1e-3)

    def test_literal_tolerance_ignores_dispersion_and_months(self):
        """Set explicitly, tpm_tolerance is used verbatim -- no statistics,
        no dependence on eff_max_dispersion or n_total_months."""
        p = GateParams(tpm_mode="ranged", min_tpm=4.0, tpm_tolerance=2.0)
        for eff_max_dispersion, n_total_months in [(1.0, 12), (5.0, 120), (0.0, 1)]:
            lo, hi = _tpm_band(p, eff_max_dispersion, n_total_months)
            assert (lo, hi) == (2.0, 6.0)

    def test_band_clips_at_zero(self):
        p = GateParams(tpm_mode="ranged", min_tpm=1.0, tpm_tolerance=5.0)
        lo, hi = _tpm_band(p, eff_max_dispersion=1.3, n_total_months=30)
        assert lo == 0.0
        assert hi == 6.0


# ---------------------------------------------------------------------------
# _gate_pass — the shared chokepoint
# ---------------------------------------------------------------------------

class TestGatePassRanged:
    def _kwargs(self, **over):
        base = dict(
            mean_tpm=0.0, id_score=0.0,
            episode_tpm=1.0, n_episodes=12, episode_id=1.0,
            eff_max_dispersion=1.8, n_total_months=29,
        )
        base.update(over)
        return base

    def test_rate_inside_band_passes(self):
        p = GateParams(tpm_mode="ranged", min_tpm=1.0)
        assert bool(_gate_pass(p, **self._kwargs(episode_tpm=1.0)))

    def test_rate_above_band_fails(self):
        """The case ranged mode exists for: a rate the floor criterion would
        happily accept, rejected here for firing too often."""
        p = GateParams(tpm_mode="ranged", min_tpm=1.0)
        assert not bool(_gate_pass(p, **self._kwargs(episode_tpm=3.9, n_episodes=90)))

    def test_rate_below_band_fails(self):
        p = GateParams(tpm_mode="ranged", min_tpm=1.0)
        assert not bool(_gate_pass(p, **self._kwargs(episode_tpm=0.1, n_episodes=3)))

    def test_zero_episodes_rejected_even_when_band_touches_zero(self):
        """Structural precondition (doc §2.3): n_episodes == 0 is rejected
        outright, not folded into the band via its zero-clip. Use a huge
        tolerance so the band's lower edge is clipped to exactly 0 -- without
        the explicit precondition, rate=0 would sit right at that edge and
        pass."""
        p = GateParams(tpm_mode="ranged", min_tpm=1.0, tpm_tolerance=10.0, min_episodes=0)
        assert not bool(_gate_pass(p, **self._kwargs(episode_tpm=0.0, n_episodes=0, episode_id=float("nan"))))

    def test_burstiness_criterion_unaffected_by_tpm_mode(self):
        """Same episode_id > eff_max_dispersion failure whether tpm_mode is
        floor or ranged, at a rate acceptable to both -- ranged only adds an
        upper rate bound, it doesn't touch the dispersion criterion."""
        common = self._kwargs(episode_tpm=1.0, episode_id=5.0, eff_max_dispersion=1.8)
        floor_result = bool(_gate_pass(GateParams(tpm_mode="floor", min_tpm=1.0), **common))
        ranged_result = bool(_gate_pass(GateParams(tpm_mode="ranged", min_tpm=1.0), **common))
        assert floor_result is False
        assert ranged_result is False

    def test_ranged_with_bar_counting_raises(self):
        p = GateParams(tpm_mode="ranged", event_counting="bar", min_tpm=1.0)
        with pytest.raises(ValueError, match="bar"):
            _gate_pass(p, **self._kwargs())

    def test_evaluate_series_ranged_with_bar_counting_raises(self):
        """Same guard reached through the public ConsistencyGate.evaluate
        path, not just the internal function directly."""
        ts = pd.Series(pd.date_range("2019-01-01", periods=1825, freq="1D"))
        event = pd.Series((np.arange(1825) % 30 == 0).astype(float))
        mi, nm = _build_month_index(ts)
        gate = ConsistencyGate(GateParams(tpm_mode="ranged", event_counting="bar", min_tpm=1.0))
        with pytest.raises(ValueError):
            gate.evaluate_series(event, mi, nm)

    def test_batched_matches_scalar_loop(self):
        """Parity guard for fix #226: ANDComposer calls _gate_pass with numpy
        arrays; the vectorized result must equal the scalar loop element by
        element, for the ranged branch same as the pre-existing ones."""
        p = GateParams(tpm_mode="ranged", min_tpm=1.0, dispersion_margin=1.3)
        rng = np.random.default_rng(0)
        n = 200
        episode_tpm = rng.uniform(0.0, 4.0, n)
        n_episodes = rng.integers(0, 60, n)
        episode_id = rng.uniform(0.2, 3.0, n)
        eff_max_dispersion = 1.8
        n_total_months = 29

        batched = _gate_pass(
            p,
            mean_tpm=np.zeros(n), id_score=np.zeros(n),
            episode_tpm=episode_tpm, n_episodes=n_episodes, episode_id=episode_id,
            eff_max_dispersion=eff_max_dispersion, n_total_months=n_total_months,
        )
        scalar = np.array([
            bool(_gate_pass(
                p,
                mean_tpm=0.0, id_score=0.0,
                episode_tpm=float(episode_tpm[i]), n_episodes=int(n_episodes[i]),
                episode_id=float(episode_id[i]),
                eff_max_dispersion=eff_max_dispersion, n_total_months=n_total_months,
            ))
            for i in range(n)
        ])
        np.testing.assert_array_equal(np.asarray(batched, dtype=bool), scalar)


# ---------------------------------------------------------------------------
# forge_preset() integration
# ---------------------------------------------------------------------------

class TestForgePresetRangedOverride:
    def test_default_preset_unaffected(self):
        disc_cfg, _alpha, _rd = forge_preset("balanced", "1D", asset="X")
        assert disc_cfg.gate_params.tpm_mode == "floor"
        assert disc_cfg.gate_params.tpm_tolerance is UNSET

    def test_override_sets_ranged_mode(self):
        disc_cfg, _alpha, _rd = forge_preset(
            "balanced", "1D", asset="X", tpm_mode="ranged", tpm_tolerance=0.5,
        )
        assert disc_cfg.gate_params.tpm_mode == "ranged"
        assert disc_cfg.gate_params.tpm_tolerance == 0.5

    def test_override_key_alone_does_not_raise_typeerror(self):
        """tpm_mode/tpm_tolerance must be in the override whitelist -- an
        unlisted key raises TypeError (see the `if overrides:` guard at the
        end of forge_preset)."""
        forge_preset("sniper", "1D", asset="X", tpm_mode="ranged")  # no raise

    def test_unknown_override_still_raises(self):
        """Sanity check that the whitelist itself is still enforced -- this
        change must not have accidentally widened it."""
        with pytest.raises(TypeError):
            forge_preset("balanced", "1D", asset="X", not_a_real_param=1)


# ---------------------------------------------------------------------------
# ResolutionTrace boundary (doc §2.5: derived locally, never in the trace)
# ---------------------------------------------------------------------------

class TestResolutionTraceBoundary:
    def test_tpm_tolerance_never_appears_in_resolution_trace(self):
        bundle = {
            "event_discovery": DiscoveryConfig(
                gate_params=GateParams(tpm_mode="ranged", min_tpm=1.0),
            ),
        }
        _resolved, trace, _violations = resolve(bundle, PipelineContext())
        fields = [d.field for d in trace]
        assert not any("tpm_tolerance" in f or "tpm_mode" in f for f in fields)


# ---------------------------------------------------------------------------
# Characterization on the real fixture (slow) -- direction of the effect
# documented in §2.8, not exact counts (fragile to feature-engineering
# changes elsewhere in the library).
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRangedModeCharacterization:
    def _run(self, tpm_mode, min_tpm, dispersion_margin, min_episodes):
        kpi = pd.read_parquet(PARQUET)
        enriched = MarketContext(kpi).run()
        disc_cfg, _alpha, _rd = forge_preset(
            "balanced", "1D", asset="ADA",
            tpm_mode=tpm_mode, min_tpm=min_tpm,
            dispersion_margin=dispersion_margin, min_episodes=min_episodes,
        )
        return len(EventDiscovery(enriched, config=disc_cfg).run())

    def test_ranged_more_permissive_than_floor_on_balanced_like_params(self):
        """§2.8: at a moderate min_tpm, the band's lower edge admits more
        than the upper edge excludes on this fixture."""
        n_floor = self._run("floor", min_tpm=1.0, dispersion_margin=1.3, min_episodes=10)
        n_ranged = self._run("ranged", min_tpm=1.0, dispersion_margin=1.3, min_episodes=10)
        assert n_ranged > n_floor

    def test_ranged_more_restrictive_than_floor_on_sniper_like_params(self):
        """§2.8: at a low min_tpm, the floor is nearly a non-filter while the
        band's upper edge does most of the work."""
        n_floor = self._run("floor", min_tpm=0.3, dispersion_margin=1.05, min_episodes=10)
        n_ranged = self._run("ranged", min_tpm=0.3, dispersion_margin=1.05, min_episodes=10)
        assert n_ranged < n_floor
