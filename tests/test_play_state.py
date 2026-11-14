"""Predetermined synthetic behavior checks, never public-label comparisons."""

from dataclasses import replace
import unittest

import numpy as np

from possession.play_state import PlayStateConfig, infer_play_state
from possession.stoppages import StoppageResult


def play_state_config():
    # Shorter windows/population for toy inputs; evidence weights are the same
    # fixed priors as the prototype, not chosen against any match labels.
    return PlayStateConfig(window_frames=5, min_observed_fraction=0.5, min_players=2,
                           quiet_ball_speed_mps=0.5, moving_ball_speed_mps=3.,
                           quiet_player_speed_mps=0.8, active_player_speed_mps=2.,
                           tag_lookback_frames=0, outside_pitch_margin_m=0.5,
                           transition_cost=0.2)


def phase(code, start, stop):
    return {"code": code, "start_frame": start, "end_frame": stop, "tags": {"Team": "N/A"}}


class PlayStateTests(unittest.TestCase):
    def fixture(self, n=200):
        ball = np.tile([30., 34.], (n, 1))
        players = np.tile([[29., 34.], [31., 34.]], (n, 1, 1))
        return ball, players

    def infer(self, ball, players, events=(), **kwargs):
        return infer_play_state(ball, players, ["H", "A"], fps=25, pitch=(105.3, 68.),
                                config=play_state_config(), events=events, **kwargs)

    def test_missing_ball_alone_does_not_mean_dead(self):
        ball, players = self.fixture()
        ball[:] = np.nan
        self.assertFalse(self.infer(ball, players).dead.any())

    def test_set_piece_tag_alone_without_observations_does_not_mean_dead(self):
        ball, players = self.fixture()
        ball[:] = np.nan
        players[:] = np.nan
        result = self.infer(ball, players, [phase("SET PIECES", 0, len(ball))])
        self.assertFalse(result.dead.any())
        self.assertTrue(result.uncertain.all())

    def test_passive_live_possession_is_not_dead_from_stillness_alone(self):
        ball, players = self.fixture()
        self.assertFalse(self.infer(ball, players).dead.any())

    def test_set_piece_and_quiet_ball_can_support_dead_without_players(self):
        ball, players = self.fixture()
        players[:] = np.nan
        result = self.infer(ball, players, [phase("SET PIECES", 0, len(ball))])
        self.assertTrue(result.dead[25:175].all())

    def test_set_piece_and_player_setup_support_dead_with_missing_ball(self):
        ball, players = self.fixture()
        ball[:] = np.nan
        result = self.infer(ball, players, [phase("SET PIECES", 0, len(ball))])
        self.assertTrue(result.dead[25:175].all())
        self.assertGreater(result.diagnostics["dead_without_observed_ball_frames"], 0)

    def test_missing_ball_during_active_play_is_not_dead_even_with_tag(self):
        ball, players = self.fixture()
        ball[:] = np.nan
        players[:, :, 0] += np.arange(len(ball))[:, None] * 0.12
        result = self.infer(ball, players, [phase("SET PIECES", 0, len(ball))])
        self.assertFalse(result.dead.any())

    def test_broad_set_piece_interval_can_contain_live_play_after_release(self):
        ball, players = self.fixture()
        ball[80:, 0] += np.arange(1, len(ball) - 79) * 0.4
        result = self.infer(ball, players, [phase("SET PIECES", 0, len(ball))])
        self.assertTrue(result.dead[20:60].all())
        self.assertFalse(result.dead[100:].any())

    def test_tactical_phase_context_can_support_live_despite_ambiguous_set_piece(self):
        ball, players = self.fixture()
        tags = [phase("SET PIECES", 0, len(ball)), phase("BUILD UP", 0, len(ball))]
        self.assertFalse(self.infer(ball, players, tags).dead.any())

    def test_long_ball_phase_supports_live_with_missing_ball(self):
        ball, players = self.fixture()
        ball[:] = np.nan
        tags = [phase("SET PIECES", 0, len(ball)), phase("LONG BALL", 0, len(ball))]
        self.assertFalse(self.infer(ball, players, tags).dead.any())

    def test_outside_pitch_support_can_detect_stoppage_without_tags(self):
        ball, players = self.fixture()
        ball[70:130, 0] = -2.
        result = self.infer(ball, players)
        self.assertTrue(result.dead[85:115].all())
        self.assertFalse(result.dead[:50].any())

    def test_isolated_outside_tracking_error_is_not_a_stoppage(self):
        ball, players = self.fixture()
        ball[100, 0] = -2.
        self.assertFalse(self.infer(ball, players).dead.any())

    def test_state_survives_neutral_gap_without_inventing_live_transition(self):
        ball, players = self.fixture()
        ball[50:100] = np.nan
        players[50:100] = np.nan
        ball[100:, 0] += np.arange(1, len(ball) - 99) * 0.4
        result = self.infer(ball, players, [phase("SET PIECES", 0, 50)])
        self.assertTrue(result.dead[20:90].all())
        self.assertFalse(result.dead[120:].any())

    def test_reset_prevents_inference_across_recording_boundary(self):
        ball, players = self.fixture()
        ball[100:] = np.nan
        players[100:] = np.nan
        reset = np.arange(len(ball)) == 100
        result = self.infer(ball, players, [phase("SET PIECES", 0, 100)], reset=reset)
        self.assertTrue(result.dead[25:75].all())
        self.assertFalse(result.dead[100:].any())

    def test_confirmed_restart_is_a_live_anchor(self):
        ball, players = self.fixture()
        status = np.full(len(ball), "U", dtype="<U1")
        status[100:110] = "L"
        restarts = StoppageResult(status, status.copy(), np.full(len(ball), "X"), ())
        result = self.infer(ball, players, [phase("SET PIECES", 0, len(ball))], restarts=restarts)
        self.assertFalse(result.dead[100:110].any())

    def test_false_tag_over_identical_live_hold_is_a_known_limit(self):
        ball, players = self.fixture()
        result = self.infer(ball, players, [phase("SET PIECES", 0, len(ball))])
        self.assertTrue(result.dead.any())  # No tracking-only proof of referee permission.

    def test_empty_inputs_invalid_config_and_no_mutation(self):
        ball, players = self.fixture()
        ball[70] = np.nan
        original = ball.copy()
        self.infer(ball, players)
        np.testing.assert_array_equal(ball, original)
        self.assertEqual(len(self.infer(ball[:0], players[:0]).dead), 0)
        for change in [dict(window_frames=1), dict(min_players=0), dict(min_observed_fraction=2),
                       dict(tag_lookback_frames=-1), dict(transition_cost=-1),
                       dict(active_player_speed_mps=0.1), dict(restart_weight=np.inf)]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    replace(play_state_config(), **change)


if __name__ == "__main__":
    unittest.main()
