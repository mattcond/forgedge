"""Tests for ``forgedge.episodes`` — episodes and concurrency (issue #168).

The two measures answer different questions and the tests are organised around
keeping that distinction sharp: a case where they agree proves nothing, so most
of what is pinned here is where they come apart.
"""
import numpy as np
import pytest

from forgedge.episodes import (
    ConcurrencyStats,
    concurrency,
    episode_ids,
    episode_starts,
)


def _mask(spec: str) -> np.ndarray:
    """``"..##.#"`` → boolean array.  Reading a pattern beats reading a list."""
    return np.array([c == "#" for c in spec], dtype=bool)


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

class TestEpisodeStarts:
    def test_a_persistent_state_is_one_episode(self):
        """The reason episodes exist: five consecutive bars of `RSI < 30` are
        one thing happening, and counting them as five inflates every per-bar
        rate and dispersion statistic (issue #134)."""
        assert int(episode_starts(_mask(".#####...")).sum()) == 1

    def test_gap_bridges_a_short_interruption(self):
        a = _mask(".##.#..###.")
        assert int(episode_starts(a, gap=0).sum()) == 3   # strict runs
        assert int(episode_starts(a, gap=1).sum()) == 2   # the 1-bar hole merges

    def test_a_bridged_gap_bar_is_never_itself_a_start(self):
        """A hole is bridged for grouping, not invented as an activation: the
        mark lands on a bar that was genuinely active."""
        a = _mask("#.#")
        starts = episode_starts(a, gap=1)
        assert starts.tolist() == [True, False, False]
        assert not starts[~a].any()

    def test_an_empty_or_silent_mask_has_no_episodes(self):
        assert int(episode_starts(np.zeros(10, dtype=bool)).sum()) == 0
        assert episode_starts(np.array([], dtype=bool)).size == 0

    def test_it_accepts_a_uint8_mask(self):
        u8 = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        # Same answer as the boolean form: 2 strict runs, 1 once the single
        # missing bar is bridged.
        assert int(episode_starts(u8, gap=0).sum()) == 2
        assert int(episode_starts(u8, gap=1).sum()) == 1
        assert episode_starts(u8, gap=0).tolist() == episode_starts(
            u8.astype(bool), gap=0
        ).tolist()


class TestEpisodeStartsBatch:
    """The vectorized 2D path (``ANDComposer``, #226/#228) must be row-for-row
    identical to calling the 1D path once per row — never persisted as an
    explicit test before #228, though it was checked ad hoc while writing it;
    added here since #228 rewrote the batch path's internals (dtype/memory)."""

    def test_batch_matches_row_by_row_1d_calls(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(5, 200)
            K = rng.integers(1, 6)
            active = rng.random((K, n)) < rng.uniform(0.05, 0.6)
            for gap in (0, 1, 2, 3, 5):
                batch_out = episode_starts(active, gap)
                for row in range(K):
                    row_out = episode_starts(active[row], gap)
                    assert np.array_equal(batch_out[row], row_out), (n, K, gap, row)

    def test_batch_empty_and_single_row_edge_cases(self):
        assert episode_starts(np.zeros((0, 5), dtype=bool)).shape == (0, 5)
        assert episode_starts(np.zeros((3, 0), dtype=bool)).shape == (3, 0)
        single = np.array([[True, False, True, True, False]])
        assert np.array_equal(episode_starts(single, gap=1)[0], episode_starts(single[0], gap=1))

    def test_batch_accepts_uint8(self):
        active_bool = np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=bool)
        active_u8 = active_bool.astype(np.uint8)
        assert np.array_equal(episode_starts(active_u8, gap=1), episode_starts(active_bool, gap=1))


class TestEpisodeIds:
    def test_ids_label_the_bars_of_each_episode(self):
        a = _mask(".##.#..###.")
        assert episode_ids(a, gap=0).tolist() == [
            -1, 0, 0, -1, 1, -1, -1, 2, 2, 2, -1
        ]

    def test_inactive_bars_stay_unlabelled_even_when_bridged(self):
        """The bridged bar is not part of any episode's *activations*, so a
        trade can never be attributed to a bar where nothing fired."""
        ids = episode_ids(_mask("#.#"), gap=1)
        assert ids.tolist() == [0, -1, 0]

    def test_the_id_count_matches_the_start_count(self):
        rng = np.random.default_rng(3)
        a = rng.random(500) < 0.25
        for gap in (0, 1, 3):
            ids = episode_ids(a, gap)
            assert len({i for i in ids.tolist() if i >= 0}) == int(
                episode_starts(a, gap).sum()
            )


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_sequential_trades_never_overlap(self):
        c = concurrency([0, 5, 10], [2, 7, 12], n_bars=20)
        assert c.peak == 1
        assert c.mean == pytest.approx(1.0)
        assert c.occupied_bars == 9          # three 3-bar trades
        assert c.effective_trades == pytest.approx(3.0)

    def test_a_stack_of_positions_is_counted(self):
        """Four trades opened on four consecutive bars, each held 10 bars: at
        the fourth bar all four are open."""
        c = concurrency([0, 1, 2, 3], [10, 11, 12, 13], n_bars=20)
        assert c.peak == 4
        assert c.effective_trades == pytest.approx(4 / c.mean)
        assert c.mean > 1.0

    def test_both_endpoints_are_inclusive(self):
        """`target_h=0` is a legal, documented same-session round trip: it
        occupies its one bar, not zero."""
        c = concurrency([7], [7], n_bars=20)
        assert c.occupied_bars == 1
        assert c.position_bars == 1
        assert c.mean == pytest.approx(1.0)

    def test_the_mean_ignores_idle_stretches(self):
        """Averaging over the whole history answers "how busy is this rule";
        capital sizing asks "when it is working, how many positions am I
        funding".  Two trades stacked on bars 0-9, then 500 idle bars: the
        answer is 2, not 0.04."""
        c = concurrency([0, 0], [9, 9], n_bars=510)
        assert c.mean == pytest.approx(2.0)
        assert c.occupied_bars == 10

    def test_no_trades_is_undefined_not_zero(self):
        c = concurrency([], [], n_bars=100)
        assert c.peak == 0
        assert np.isnan(c.mean)
        assert np.isnan(c.effective_trades)

    def test_it_does_not_need_the_bar_count(self):
        """`n_bars` only extends the window; the occupancy is decided by the
        intervals themselves."""
        assert concurrency([0, 1], [4, 5]) == concurrency([0, 1], [4, 5], n_bars=6)

    def test_mismatched_endpoints_raise(self):
        with pytest.raises(ValueError, match="same length"):
            concurrency([0, 1], [4])

    def test_the_mean_is_not_rounded_at_the_source(self):
        """`effective_trades` divides by this: rounding the mean to 6 places
        turns an exact 2.25 into 2.2500005625."""
        c = concurrency([0, 1, 8], [3, 2, 9], n_bars=12)
        assert c.effective_trades == pytest.approx(2.25, rel=1e-12)


class TestEpisodesAreNotConcurrency:
    """The distinction the module exists to keep straight."""

    def test_one_episode_can_be_many_concurrent_positions(self):
        """A five-bar persistent event is *one* episode and — because a trade
        opens on every active bar with no flat-state check — five overlapping
        positions on one price path."""
        active = _mask("#####" + "." * 30)
        assert int(episode_starts(active).sum()) == 1

        opens = np.flatnonzero(active)
        c = concurrency(opens, opens + 12, n_bars=active.size)
        assert c.peak == 5

    def test_separate_episodes_still_overlap_when_the_horizon_is_long(self):
        """Two episodes 6 bars apart, held 20 bars each: distinct signals,
        simultaneous capital.  Counting episodes would report no overlap."""
        active = _mask("#" + "." * 5 + "#" + "." * 30)
        assert int(episode_starts(active).sum()) == 2

        opens = np.flatnonzero(active)
        c = concurrency(opens, opens + 20, n_bars=active.size)
        assert c.peak == 2

    def test_the_two_counts_disagree_in_both_directions(self):
        active = _mask("###" + "." * 10 + "###")
        n_episodes = int(episode_starts(active).sum())
        opens = np.flatnonzero(active)
        c = concurrency(opens, opens + 2, n_bars=active.size)

        assert n_episodes == 2          # two clusters of signal
        assert c.peak == 3              # three positions on one price path
        assert c.n_trades == 6          # and six trades from two episodes


class TestConcurrencyStats:
    def test_effective_trades_is_the_sample_the_overlap_supports(self):
        """`total_trades / mean` — the divisor F16 (#177) will consume.  The
        economics stay nominal; this is only for quantities that assume
        independent observations."""
        stats = ConcurrencyStats(mean=3.71, peak=12, occupied_bars=100,
                                 position_bars=371, n_trades=118)
        assert stats.effective_trades == pytest.approx(118 / 3.71)

    def test_it_is_undefined_when_the_mean_is(self):
        stats = ConcurrencyStats(mean=float("nan"), peak=0, occupied_bars=0,
                                 position_bars=0, n_trades=0)
        assert np.isnan(stats.effective_trades)
