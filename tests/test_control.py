"""Physical toy sequences, independent of competition assets and labels."""

import unittest

import numpy as np

from possession.control import ControlConfig, estimate_control
from possession.state import decode_state


class ControlTests(unittest.TestCase):
    # Chosen for these toy geometries, not tuned on match ground truth.
    config = ControlConfig(1.0, 1.0, 0.25)

    def estimate(self, ball, players, teams=("H", "A"), **kwargs):
        return estimate_control(
            ball, players, teams, fps=25, config=self.config, **kwargs
        )

    def dribble(self, frames=6):
        x = np.arange(frames) * 0.1  # 2.5 m/s at 25 fps
        ball = np.column_stack((x, np.zeros(frames)))
        players = np.stack((ball + [0.3, 0], ball + [5, 0]), axis=1)
        return ball, players

    def test_dribbling_and_accepted_diagnostics(self):
        ball, players = self.dribble()
        result = self.estimate(ball, players)
        self.assertEqual("".join(result.labels), "XHHHHH")
        np.testing.assert_allclose(result.distance_m[1:], 0.3)
        np.testing.assert_allclose(result.relative_speed_mps[1:], 0, atol=1e-12)
        np.testing.assert_array_equal(result.track_index, [-1, 0, 0, 0, 0, 0])

    def test_fast_pass_near_stationary_defender_is_not_control(self):
        ball = np.array([[0., 0.], [0.8, 0.], [1.6, 0.]])
        players = np.tile([[10., 0.], [0.8, 0.2]], (3, 1, 1))
        result = self.estimate(ball, players)
        self.assertEqual("".join(result.labels), "XXX")
        self.assertTrue(np.isnan(result.distance_m).all())
        self.assertTrue(np.isnan(result.relative_speed_mps).all())

    def test_interception_after_ball_slows_to_receiver_motion(self):
        ball = np.array([[0., 0.], [0.8, 0.], [0.9, 0.], [1., 0.]])
        away = np.array([[0.4, 0.], [1.1, 0.], [1.2, 0.], [1.3, 0.]])
        players = np.stack((np.full_like(ball, 10.), away), axis=1)
        self.assertEqual("".join(self.estimate(ball, players).labels), "XXAA")

    def test_opponents_similarly_close_are_ambiguous(self):
        ball, players = self.dribble()
        players[:, 1] = ball + [0, 0.4]
        self.assertEqual("".join(self.estimate(ball, players).labels), "XXXXXX")

    def test_ambiguity_between_teammates_does_not_block_team(self):
        ball, players = self.dribble()
        players[:, 1] = ball + [0, 0.4]
        self.assertEqual(
            "".join(self.estimate(ball, players, teams=("H", "H")).labels), "XHHHHH"
        )

    def test_near_opponent_with_missing_velocity_still_blocks(self):
        ball, players = self.dribble(2)
        players[0, 1] = np.nan
        players[1, 1] = ball[1] + [0.4, 0]
        self.assertEqual("".join(self.estimate(ball, players).labels), "XX")

    def test_compatible_motion_at_a_distance_is_insufficient(self):
        ball, players = self.dribble()
        players += [3, 0]
        self.assertTrue((self.estimate(ball, players).labels == "X").all())

    def test_missing_ball_does_not_create_velocity_across_gap(self):
        ball, players = self.dribble()
        ball[2] = np.nan
        result = self.estimate(ball, players)
        self.assertEqual("".join(result.labels), "XHXXHH")
        state = decode_state(result.labels, np.zeros(6, dtype=bool), max_carry_frames=2)
        self.assertEqual("".join(state.labels), "XHHHHH")

    def test_missing_player_coordinate_invalidates_motion(self):
        ball, players = self.dribble()
        players[2, 0, 1] = np.nan
        self.assertEqual("".join(self.estimate(ball, players).labels), "XHXXHH")

    def test_new_track_slot_does_not_inherit_previous_velocity(self):
        ball, players = self.dribble(4)
        players[2:, 1] = players[2:, 0]
        players[:2, 1] = np.nan
        players[2:, 0] = np.nan
        self.assertEqual("".join(self.estimate(ball, players).labels), "XHXA")

    def test_reset_blocks_velocity_over_boundary(self):
        ball, players = self.dribble(4)
        result = self.estimate(ball, players, reset=[False, False, True, False])
        self.assertEqual("".join(result.labels), "XHXH")

    def test_single_frame_evidence_is_not_duration_filtered(self):
        ball, players = self.dribble(3)
        players[2, 0] = [20, 20]
        self.assertEqual("".join(self.estimate(ball, players).labels), "XHX")

    def test_one_touch_fast_kick_is_a_known_limitation(self):
        # Close contact alone cannot distinguish a kick from a passing ball.
        ball = np.array([[0., 0.], [0.8, 0.]])
        players = np.tile([[0.8, 0.], [10., 10.]], (2, 1, 1))
        self.assertEqual("".join(self.estimate(ball, players).labels), "XX")

    def test_stationary_ball_requires_external_dead_evidence(self):
        ball = np.zeros((3, 2))
        players = np.tile([[0.3, 0.], [10., 10.]], (3, 1, 1))
        result = self.estimate(ball, players)
        self.assertEqual("".join(result.labels), "XHH")
        state = decode_state(result.labels, [True, True, True], max_carry_frames=2)
        self.assertEqual("".join(state.labels), "DDD")

    def test_empty_single_frame_and_no_players(self):
        for frames, slots in [(0, 2), (1, 2), (3, 0)]:
            with self.subTest(frames=frames, slots=slots):
                result = self.estimate(
                    np.zeros((frames, 2)), np.zeros((frames, slots, 2)), ["H"] * slots
                )
                self.assertEqual("".join(result.labels), "X" * frames)

    def test_invalid_shapes_labels_rate_and_reset(self):
        ball, players = self.dribble()
        for change in [
            {"ball_xy_m": np.zeros(6)},
            {"player_xy_m": np.zeros((6, 2))},
            {"player_xy_m": np.zeros((5, 2, 2))},
            {"teams": ["H"]}, {"teams": ["H", "D"]},
            {"fps": 0}, {"fps": np.nan}, {"fps": True},
            {"reset": [False]}, {"reset": np.zeros(6)},
        ]:
            args = dict(ball_xy_m=ball, player_xy_m=players, teams=["H", "A"],
                        fps=25, config=self.config)
            args.update(change)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    estimate_control(**args)

    def test_invalid_thresholds(self):
        for value in [-1, np.inf, np.nan, True]:
            for field in ["max_distance_m", "max_relative_speed_mps", "opponent_margin_m"]:
                args = dict(max_distance_m=1., max_relative_speed_mps=1., opponent_margin_m=0.25)
                args[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        ControlConfig(**args)


if __name__ == "__main__":
    unittest.main()
