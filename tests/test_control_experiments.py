"""Synthetic jitter, brief control, and pass/deflection counterexamples."""

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from possession.control import ControlConfig, estimate_control
from possession.interactions import InteractionConfig, infer_interactions
from possession.pipeline import load_config
from possession.state import decode_state
from possession.pipeline import predict
from possession.tracking import Tracking
from tests.test_contacts import contact_config
from tests import test_interactions as interaction_fixtures


class FittedControlTests(unittest.TestCase):
    config = ControlConfig(1., 1., .25, motion_window_frames=5, max_motion_fit_error_m=.1)

    def fixture(self, n=41):
        ball = np.column_stack((np.arange(n) * .1, np.zeros(n)))
        players = np.stack((ball + [.3, 0.], ball + [8., 0.]), axis=1)
        return ball, players

    def infer(self, ball, players, **kwargs):
        return estimate_control(ball, players, ["H", "A"], fps=25,
                                 config=self.config, **kwargs)

    def jitter(self):
        ball, players = self.fixture()
        ball[:, 0] += np.where(np.arange(len(ball)) % 2, -.08, .08)
        return ball, players

    def test_small_jitter_can_be_supported_by_consistent_relative_motion(self):
        ball, players = self.jitter()
        previous = estimate_control(ball, players, ["H", "A"], fps=25,
                                    config=replace(self.config, motion_window_frames=1))
        fitted = self.infer(ball, players)
        self.assertTrue((previous.labels == "X").all())
        self.assertEqual(fitted.labels[0], "X")
        self.assertTrue((fitted.labels[1:-1] == "H").all())
        self.assertTrue(fitted.fit_used[1:-1].all())
        self.assertEqual(fitted.labels[-1], "X")  # Edge has less fit support.
        state = decode_state(fitted.labels, np.zeros(len(ball), dtype=bool), max_carry_frames=None)
        self.assertTrue((state.labels[1:] == "H").all())

    def test_fitted_motion_is_used_consistently_by_interactions(self):
        ball, players = self.jitter()
        result = infer_interactions(ball, players, ["H", "A"], fps=25,
                                    control_config=self.config, contact_config=contact_config(),
                                    interaction_config=InteractionConfig(12, .25, 3., False))
        self.assertEqual("".join(result.labels), "X" + "H" * 39 + "X")
        self.assertTrue((result.source[1:-1] == "fitted_motion").all())

    def test_existing_single_frame_control_is_not_filtered_by_window_length(self):
        ball, players = self.fixture(n=3)
        players[2, 0] += [20., 0.]
        result = self.infer(ball, players)
        self.assertEqual("".join(result.labels), "XHX")
        self.assertFalse(result.fit_used[1])

    def test_close_opponent_still_blocks_fitted_control(self):
        ball, players = self.jitter()
        players[:, 1] = players[:, 0] + [0., .1]
        self.assertTrue((self.infer(ball, players).labels == "X").all())

    def test_fast_ball_passing_a_stationary_player_is_not_fitted_control(self):
        ball = np.column_stack((np.arange(9) * .8, np.zeros(9)))
        players = np.tile([[3.2, .2], [20., 20.]], (9, 1, 1))
        self.assertTrue((self.infer(ball, players).labels == "X").all())

    def test_average_zero_velocity_does_not_hide_a_large_reversal(self):
        ball = np.column_stack(([0., .3, .6, .3, 0.], np.zeros(5)))
        players = np.tile([[.3, 0.], [10., 0.]], (5, 1, 1))
        result = self.infer(ball, players)
        self.assertEqual(result.labels[2], "X")
        self.assertFalse(result.fit_used[2])

    def test_missing_positions_do_not_get_filled_by_fit(self):
        ball, players = self.jitter()
        ball[10] = np.nan
        result = self.infer(ball, players)
        self.assertTrue((result.labels[8:13] == "X").all())
        self.assertFalse(result.fit_used[8:13].any())
        self.assertEqual(result.labels[13], "H")

    def test_reset_and_contact_breaks_limit_fitting(self):
        ball, players = self.jitter()
        reset = np.arange(len(ball)) == 20
        result = self.infer(ball, players, reset=reset)
        self.assertEqual(result.labels[20], "X")
        breaks = np.zeros(len(ball), dtype=bool)
        breaks[20:22] = True
        result = self.infer(ball, players, motion_breaks=breaks)
        self.assertFalse(result.fit_used[20])
        self.assertEqual(result.labels[20], "X")
        self.assertEqual(result.labels[22], "H")

    def test_physically_implausible_shared_jump_still_fails_interaction_guard(self):
        ball, players = self.fixture(n=9)
        ball[4] += 10.
        players[4] += 10.
        result = infer_interactions(ball, players, ["H", "A"], fps=25,
                                    control_config=self.config, contact_config=contact_config(),
                                    interaction_config=InteractionConfig(12, .25, 3., False))
        self.assertEqual(result.labels[4], "X")
        self.assertEqual(result.labels[5], "X")

    def test_windowed_mode_preserves_short_turnovers_from_direct_evidence(self):
        toy = interaction_fixtures.InteractionTests()
        ball = np.column_stack((np.arange(12) * .1, np.zeros(12)))
        players = np.repeat(ball[:, None, :], 2, axis=1)
        players[:, 0, 1] = [.2, .2, .2, .5, .8, .8, .8, .8, .5, .2, .2, .2]
        players[:, 1, 1] = [1., 1., .7, .4, .2, .2, .2, .4, .7, 1., 1., 1.]
        original = toy.infer(ball, players, ["H", "A"])
        candidate = infer_interactions(ball, players, ["H", "A"], fps=25,
                                       control_config=ControlConfig(.4, 1., .15, 5, .1),
                                       contact_config=contact_config(),
                                       interaction_config=InteractionConfig(12, .25, 3., False))
        accepted = original.labels != "X"
        np.testing.assert_array_equal(candidate.labels[accepted], original.labels[accepted])

    def test_configurations_only_change_control_fit_and_explicit_pass_policy(self):
        root = Path(__file__).resolve().parents[1] / "possession" / "configs"
        old = load_config(root / "supported_exits_gaps.json")
        conservative = load_config(root / "control_fitted.json")
        passes = load_config(root / "control_fitted_passes.json")
        self.assertEqual(replace(conservative, control=old.control), old)
        self.assertEqual(replace(passes, interactions=conservative.interactions), conservative)
        self.assertFalse(conservative.interactions.accept_pass_hypotheses)
        self.assertTrue(passes.interactions.accept_pass_hypotheses)

    def test_fitted_evidence_reaches_final_pipeline_labels_and_respects_dead(self):
        root = Path(__file__).resolve().parents[1] / "possession" / "configs"
        old = load_config(root / "supported_exits_gaps.json")
        candidate = load_config(root / "control_fitted.json")
        ball, players = self.jitter()
        tracking = Tracking(ball + [30., 30.], players + [30., 30.],
                             np.array(["H", "A"]), 25., (105.3, 68.))
        previous = predict(tracking, old)
        current = predict(tracking, candidate)
        self.assertTrue((previous.state.labels == "X").all())
        self.assertEqual("".join(current.state.labels), "X" + "H" * 40)
        self.assertGreater(current.summary["evidence_sources"]["fitted_motion"], 0)
        explicit = predict(tracking, candidate, dead=np.ones(len(ball), dtype=bool))
        self.assertTrue((explicit.state.labels == "D").all())

    def test_invalid_fit_settings_and_break_masks(self):
        for change in (dict(motion_window_frames=0), dict(motion_window_frames=2),
                       dict(motion_window_frames=True), dict(max_motion_fit_error_m=-1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change)
        ball, players = self.fixture()
        with self.assertRaises(ValueError):
            self.infer(ball, players, motion_breaks=np.zeros(len(ball), dtype=int))


class ContactControlExperimentTests(unittest.TestCase):
    def infer(self, ball, players, teams, promote):
        return infer_interactions(ball, players, teams, fps=25,
                                  control_config=ControlConfig(.4, 1., .15, 5, .1),
                                  contact_config=contact_config(),
                                  interaction_config=InteractionConfig(12, .25, 3., promote))

    def test_observed_transfer_can_backdate_first_time_control_when_opted_in(self):
        ball, players, teams = interaction_fixtures.InteractionTests().pass_sequence()
        conservative = self.infer(ball, players, teams, False)
        promoted = self.infer(ball, players, teams, True)
        self.assertEqual(conservative.labels[3], "X")
        self.assertEqual(promoted.labels[3], "A")
        self.assertEqual(promoted.source[3], "pass_hypothesis")
        self.assertTrue((promoted.labels[3:10] == "A").all())

    def test_same_observations_cannot_distinguish_a_fortunate_deflection(self):
        ball, players, teams = interaction_fixtures.InteractionTests().pass_sequence()
        as_pass = self.infer(ball, players, teams, True)
        as_deflection = self.infer(ball.copy(), players.copy(), teams, True)
        np.testing.assert_array_equal(as_pass.labels, as_deflection.labels)
        self.assertEqual(as_deflection.labels[3], "A")  # Known intent ambiguity.

    def test_clearance_without_receiver_and_opposing_receipt_are_not_promoted(self):
        toy = interaction_fixtures.InteractionTests()
        ball, players, teams = toy.pass_sequence(receiver_team="H")
        self.assertEqual(self.infer(ball, players, teams, True).labels[3], "X")
        ball, players, teams = toy.pass_sequence()
        players[:, 2] = [10., -10.]
        self.assertTrue((self.infer(ball, players, teams, True).labels == "X").all())

    def test_clean_cushioning_reception_stays_backdated_to_contact(self):
        ball = np.array([[-1.2, 0.], [-.8, 0.], [-.4, 0.], [0., 0.],
                         [.04, 0.], [.08, 0.], [.12, 0.], [.16, 0.]])
        players = np.tile([[10., 10.], [.2, 0.]], (8, 1, 1))
        players[4:, 1] = ball[4:] + [.2, 0.]
        result = self.infer(ball, players, ["H", "A"], False)
        self.assertEqual(result.labels[3], "A")
        self.assertEqual(result.source[3], "reception_backdated")

    def test_dead_or_reset_still_blocks_event_backdating(self):
        ball, players, teams = interaction_fixtures.InteractionTests().pass_sequence()
        mask = np.arange(len(ball)) == 6
        for key in ("dead", "reset"):
            result = infer_interactions(ball, players, teams, fps=25,
                                        control_config=ControlConfig(.4, 1., .15, 5, .1),
                                        contact_config=contact_config(),
                                        interaction_config=InteractionConfig(12, .25, 3., True),
                                        **{key: mask})
            self.assertEqual(result.labels[3], "X")

    def test_fitted_motion_does_not_confirm_pass_across_receiver_track_gap(self):
        ball, players, teams = interaction_fixtures.InteractionTests().pass_sequence()
        players[6, 2] = np.nan
        result = self.infer(ball, players, teams, True)
        event = next(item for item in result.interactions if item.contact.frame == 3)
        self.assertEqual(event.kind, "unresolved_contact")
        self.assertEqual(result.labels[3], "X")


if __name__ == "__main__":
    unittest.main()
