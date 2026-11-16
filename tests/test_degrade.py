"""Synthetic player-track damage and the robust feature options, on invented tracks."""

import unittest

import numpy as np

from possession import degrade, learned


def tracks(n=25 * 600, slots=22, seed=0):
    rng = np.random.default_rng(seed)
    start = rng.uniform([0, 0], [105, 68], size=(slots, 2))
    steps = rng.normal(0, 0.05, size=(n, slots, 2))
    return start + np.cumsum(steps, axis=0)


class DegradeTests(unittest.TestCase):
    def test_damage_is_deterministic_and_leaves_input_untouched(self):
        players = tracks()
        before = players.copy()
        a = degrade.damage_players(players, seed=3)
        np.testing.assert_array_equal(players, before)
        np.testing.assert_array_equal(a, degrade.damage_players(players, seed=3))
        self.assertFalse(np.array_equal(np.isnan(a), np.isnan(degrade.damage_players(players, seed=4))))

    def test_damage_fragments_tracks_and_creates_jumps(self):
        players = tracks()
        clean, damaged = degrade.tracking_stats(players), degrade.tracking_stats(degrade.damage_players(players, seed=1))
        self.assertEqual(clean["players_per_frame"], 22.)
        self.assertLess(damaged["players_per_frame"], 22.)
        self.assertGreater(damaged["players_per_frame"], 18.)
        self.assertLess(damaged["segment_median_s"], 120.)
        self.assertGreater(damaged["fast_moves_per_min"], clean["fast_moves_per_min"])

    def test_smooth_ball_keeps_gaps_and_straight_lines(self):
        ball = np.c_[np.arange(200) * 0.4, np.zeros(200)]
        ball[50:60] = np.nan
        out = degrade.smooth_ball(ball)
        np.testing.assert_array_equal(np.isnan(out), np.isnan(ball))
        np.testing.assert_allclose(out[80:180], ball[80:180], atol=1e-9)   # interior of a straight line unchanged
        kink = ball.copy(); kink[120:, 1] = (np.arange(80) * 0.4)            # sharp turn at frame 120
        smoothed = degrade.smooth_ball(kink)
        jump = lambda b: np.abs(np.diff(b[110:130, 1], 2)).max()
        self.assertLess(jump(smoothed), jump(kink))

    def test_track_age_counts_continuous_visibility(self):
        players = np.zeros((8, 1, 2))
        players[3, 0] = np.nan
        np.testing.assert_array_equal(learned.track_age(players, cap=4)[:, 0], [1, 2, 3, 0, 1, 2, 3, 4])

    def test_robust_velocity_drops_impossible_speeds(self):
        players = np.zeros((40, 1, 2))
        players[:, 0, 0] = np.arange(40) * 0.2          # 5 m/s
        players[20:, 0, 0] += 10.                       # a 10 m jump in one frame
        plain = learned._player_velocity(players, 25., robust=False)
        robust = learned._player_velocity(players, 25., robust=True)
        self.assertGreater(np.nanmax(plain), 12.)
        self.assertLessEqual(np.nanmax(np.linalg.norm(robust, axis=-1)), 12.)
        np.testing.assert_allclose(robust[10, 0], [5., 0.])


if __name__ == "__main__":
    unittest.main()
