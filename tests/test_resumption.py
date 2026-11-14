"""Invented physical actions and counterexamples; no competition assets."""

from dataclasses import replace
import unittest

import numpy as np

from possession.control import ControlConfig
from possession.resumption import ResumptionConfig, refine_dead_exits
from possession.stoppages import StoppageResult
from tests.test_play_state import play_state_config
from tests.test_stoppages import stoppage_config


def resumption_config():
    return ResumptionConfig(max_hold_frames=10, lookahead_frames=25,
                            min_observed_fraction=.3, max_observation_gap_frames=4,
                            min_ball_travel_m=1., max_fit_error_m=.3,
                            min_active_player_fraction=.5)


class ResumptionTests(unittest.TestCase):
    def fixture(self, n=100):
        ball = np.tile([20., 20.], (n, 1))
        players = np.tile([[20.2, 20.], [30., 25.], [40., 30.],
                           [50., 35.], [60., 40.], [70., 45.]], (n, 1, 1))
        dead = np.arange(n) < 20
        return ball, players, dead

    def infer(self, ball, players, dead, *, config=None, status=None, reset=None, live_guard=None):
        status = np.full(len(ball), "U") if status is None else status
        restarts = StoppageResult(status, status.copy(), np.full(len(ball), "X"), ())
        return refine_dead_exits(ball, players, ["H"] * 3 + ["A"] * 3,
                                 dead=dead, fps=25, pitch=(105.3, 68.),
                                 config=config or resumption_config(),
                                 motion_config=replace(play_state_config(), min_players=6),
                                 control_config=ControlConfig(1., 1., .25),
                                 restart_config=stoppage_config(), restarts=restarts, reset=reset,
                                 live_guard=live_guard)

    def gap_fixture(self):
        ball, players, dead = self.fixture()
        ball[18:, 0] += np.arange(1, 83) * .4
        ball[20:23] = np.nan
        return ball, players, dead

    def test_short_occlusion_can_resume_from_bracketed_departure(self):
        ball, players, dead = self.gap_fixture()
        original = self.infer(ball, players, dead)
        candidate = self.infer(ball, players, dead,
                               config=replace(resumption_config(), bridge_observation_gaps=True))
        self.assertTrue(original.dead[20:23].all())
        self.assertFalse(candidate.dead[20:].any())
        self.assertFalse(candidate.unresolved.any())
        self.assertEqual(candidate.source[20], "gap_ball_departure")
        self.assertGreater(candidate.confirmation_frame[20], 22)
        self.assertTrue(candidate.dead[:20].all())
        self.assertFalse(candidate.diagnostics["ownership_seeded"])

    def test_gap_activity_support_does_not_need_a_nearby_player(self):
        ball, players, dead = self.gap_fixture()
        players[:, 0] = [10., 10.]
        players[18:, :3, 0] += np.arange(1, 83)[:, None] * .1
        result = self.infer(ball, players, dead,
                            config=replace(resumption_config(), bridge_observation_gaps=True))
        self.assertEqual(result.source[20], "gap_ball_motion_and_group_activity")

    def test_enabling_gap_support_does_not_invent_an_unobserved_kick(self):
        ball, players, dead = self.fixture()
        ball[20:23] = np.nan
        ball[23:, 0] += np.arange(1, 78) * .4
        result = self.infer(ball, players, dead,
                            config=replace(resumption_config(), bridge_observation_gaps=True))
        self.assertTrue(result.dead[20:23].all())
        self.assertEqual(result.source[23], "ball_departure")

    def test_long_or_unbracketed_gap_does_not_qualify(self):
        for trailing in (False, True):
            with self.subTest(trailing=trailing):
                ball, players, dead = self.gap_fixture()
                ball[20:40 if not trailing else 100] = np.nan
                result = self.infer(ball, players, dead,
                                    config=replace(resumption_config(), bridge_observation_gaps=True))
                self.assertTrue(result.dead[20:30].all())
                self.assertTrue(result.unresolved[30:40].all())

    def test_stationary_or_implausible_gap_does_not_qualify(self):
        for jump in (False, True):
            with self.subTest(jump=jump):
                ball, players, dead = self.fixture()
                ball[20:23] = np.nan
                if jump:
                    ball[23:, 0] = 90.
                result = self.infer(ball, players, dead,
                                    config=replace(resumption_config(), bridge_observation_gaps=True))
                self.assertTrue(result.dead[20:30].all())
                self.assertTrue(result.unresolved[30:].all())

    def test_gap_action_cannot_use_observations_before_reset(self):
        ball, players, dead = self.gap_fixture()
        ball[19] = np.nan
        reset = np.arange(100) == 19
        result = self.infer(ball, players, dead, reset=reset,
                            config=replace(resumption_config(), bridge_observation_gaps=True))
        self.assertTrue(result.dead[20:23].all())

    def test_gap_action_cannot_cross_next_or_previous_distinct_stoppage(self):
        ball, players, dead = self.gap_fixture()
        dead[22:30] = True
        c = replace(resumption_config(), bridge_observation_gaps=True)
        self.assertTrue(self.infer(ball, players, dead, config=c).dead[20:23].all())
        ball, players, dead = self.gap_fixture()
        dead[10:18] = False
        ball[18:23] = np.nan
        result = self.infer(ball, players, dead, config=c)
        self.assertTrue(result.dead[20:23].all())

    def test_gap_action_does_not_use_strict_dead_observations(self):
        ball, players, dead = self.gap_fixture()
        status = np.full(100, "U")
        status[:20] = "D"
        result = self.infer(ball, players, dead, status=status,
                            config=replace(resumption_config(), bridge_observation_gaps=True))
        self.assertTrue(result.dead[20:23].all())

    def test_transfer_guard_blocks_hold_and_soft_dead_without_observed_ball(self):
        ball, players, dead = self.fixture()
        ball[15:] = np.nan
        guard = (np.arange(100) >= 15) & (np.arange(100) < 25)
        c = replace(resumption_config(), use_live_sequence_guard=True)
        result = self.infer(ball, players, dead, config=c, live_guard=guard)
        self.assertFalse(result.dead[guard].any())
        self.assertFalse(result.unresolved[guard].any())
        self.assertTrue((result.source[guard] == "live_transfer_sequence").all())
        self.assertEqual(result.diagnostics["transfer_guard_overridden_soft_dead_frames"], 5)

    def test_transfer_guard_preserves_explicit_strict_dead_anchors(self):
        ball, players, dead = self.fixture()
        guard = (np.arange(100) >= 15) & (np.arange(100) < 25)
        status = np.full(100, "U")
        status[15:20] = "D"
        result = self.infer(ball, players, dead, status=status, live_guard=guard,
                            config=replace(resumption_config(), use_live_sequence_guard=True))
        self.assertTrue(result.dead[15:20].all())
        self.assertFalse(result.dead[20:25].any())
        self.assertEqual(result.diagnostics["transfer_guard_strict_dead_conflicts"], 5)

    def test_transfer_guard_is_opt_in_and_mask_is_validated(self):
        ball, players, dead = self.fixture()
        guard = np.ones(100, dtype=bool)
        self.assertTrue(self.infer(ball, players, dead, live_guard=guard).dead[:30].all())
        for value in (None, [True], np.ones(100, dtype=int)):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                self.infer(ball, players, dead, live_guard=value,
                           config=replace(resumption_config(), use_live_sequence_guard=True))

    def test_stationary_gap_has_bounded_hold_then_uncertainty(self):
        ball, players, dead = self.fixture()
        result = self.infer(ball, players, dead)
        self.assertTrue(result.dead[:30].all())
        self.assertFalse(result.dead[30:].any())
        self.assertTrue(result.unresolved[30:].all())
        self.assertEqual(result.diagnostics["held_dead_frames"], 10)

    def test_player_repositioning_with_missing_ball_does_not_confirm_live(self):
        ball, players, dead = self.fixture()
        ball[20:] = np.nan
        players[20:, :, 0] += np.arange(80)[:, None] * .12
        result = self.infer(ball, players, dead)
        self.assertTrue(result.unresolved[30:].all())
        self.assertEqual(result.diagnostics["exits_with_supported_action"], 0)

    def test_carrying_ball_with_passive_other_players_does_not_confirm_live(self):
        ball, players, dead = self.fixture()
        ball[20:, 0] += np.arange(80) * .1
        players[20:, 0, 0] += np.arange(80) * .1
        result = self.infer(ball, players, dead)
        self.assertTrue(result.unresolved[30:].all())

    def test_quick_departure_does_not_need_stationary_preparation_or_tag(self):
        ball, players, dead = self.fixture()
        ball[20:, 0] += np.arange(1, 81) * .4
        result = self.infer(ball, players, dead)
        self.assertFalse(result.dead[20:].any())
        self.assertFalse(result.unresolved.any())
        self.assertEqual(result.source[20], "ball_departure")
        self.assertGreater(result.confirmation_frame[20], 20)

    def test_partially_hidden_departure_uses_observed_samples(self):
        ball, players, dead = self.fixture()
        ball[20:, 0] += np.arange(1, 81) * .4
        ball[21:23] = np.nan
        result = self.infer(ball, players, dead)
        self.assertEqual(result.source[20], "ball_departure")
        self.assertGreaterEqual(result.confirmation_frame[20], 24)

    def test_hidden_start_is_not_backdated_into_unobserved_frames(self):
        ball, players, dead = self.fixture()
        ball[20:23] = np.nan
        ball[23:, 0] += np.arange(1, 78) * .4
        result = self.infer(ball, players, dead)
        self.assertTrue(result.dead[20:23].all())
        self.assertEqual(result.source[23], "ball_departure")

    def test_fast_ball_alone_without_player_support_is_insufficient(self):
        ball, players, dead = self.fixture()
        ball[20:, 0] += np.arange(1, 81) * .4
        players[:] = np.nan
        result = self.infer(ball, players, dead)
        self.assertTrue(result.unresolved[30:].all())

    def test_dribble_with_group_activity_supports_live_without_seeding_owner(self):
        ball, players, dead = self.fixture()
        ball[20:, 0] += np.arange(1, 81) * .1
        players[20:, :3, 0] += np.arange(1, 81)[:, None] * .1
        result = self.infer(ball, players, dead)
        self.assertEqual(result.source[20], "dribble_and_group_activity")
        self.assertFalse(result.dead[20:].any())
        self.assertFalse(result.diagnostics["ownership_seeded"])

    def test_coherent_ball_and_group_activity_can_support_unknown_owner(self):
        ball, players, dead = self.fixture()
        players[:, 0] = [10., 10.]
        players[20:, :3, 0] += np.arange(80)[:, None] * .1
        ball[20:, 0] += np.arange(1, 81) * .4
        result = self.infer(ball, players, dead)
        self.assertEqual(result.source[20], "ball_motion_and_group_activity")

    def test_reversals_and_jumps_do_not_look_like_coherent_departures(self):
        ball, players, dead = self.fixture()
        ball[20:, 0] += np.where(np.arange(80) % 2, 1.2, 0.)
        self.assertTrue(self.infer(ball, players, dead).unresolved[30:].all())
        ball[20:, 0] = 80.
        self.assertTrue(self.infer(ball, players, dead).unresolved[30:].all())

    def test_outside_ball_travel_does_not_resume_play(self):
        ball, players, dead = self.fixture()
        ball[:, 1] = -1.
        ball[20:, 0] += np.arange(1, 81) * .4
        self.assertTrue(self.infer(ball, players, dead).unresolved[30:].all())

    def test_next_dead_run_blocks_lookahead(self):
        ball, players, dead = self.fixture()
        dead[22:40] = True
        ball[20:, 0] += np.arange(1, 81) * .4
        result = self.infer(ball, players, dead)
        self.assertTrue(result.dead[20:40].all())

    def test_reset_clears_pending_stoppage_and_blocks_future_lookahead(self):
        ball, players, dead = self.fixture()
        ball[20:, 0] += np.arange(1, 81) * .4
        reset = np.arange(100) == 22
        result = self.infer(ball, players, dead, reset=reset)
        self.assertTrue(result.dead[20:22].all())
        self.assertFalse(result.dead[22:].any())
        self.assertFalse(result.unresolved[22:].any())

    def test_confirmed_release_bypasses_guard_without_new_observations(self):
        ball, players, dead = self.fixture()
        status = np.full(100, "U")
        status[20:23] = "L"
        ball[20:] = np.nan
        result = self.infer(ball, players, dead, status=status)
        self.assertFalse(result.dead[20:].any())
        self.assertFalse(result.unresolved.any())
        self.assertEqual(result.source[20], "confirmed_restart")

    def test_no_stoppage_created_when_original_model_never_predicts_dead(self):
        ball, players, dead = self.fixture()
        ball[:] = np.nan
        dead[:] = False
        result = self.infer(ball, players, dead)
        self.assertFalse(result.dead.any())
        self.assertFalse(result.unresolved.any())

    def test_hold_limit_is_not_an_automatic_live_anchor(self):
        ball, players, dead = self.fixture()
        result = self.infer(ball, players, dead, config=replace(resumption_config(), max_hold_frames=0))
        self.assertFalse(result.dead[20:].any())
        self.assertTrue(result.unresolved[20:].all())

    def test_late_action_can_resolve_prior_unknown_play_state(self):
        ball, players, dead = self.fixture()
        ball[40:, 0] += np.arange(1, 61) * .4
        result = self.infer(ball, players, dead)
        self.assertTrue(result.unresolved[30:39].all())
        self.assertFalse(result.unresolved[40:].any())
        self.assertFalse(result.dead[40:].any())

    def test_input_arrays_are_unchanged_and_configuration_is_validated(self):
        ball, players, dead = self.fixture()
        originals = [value.copy() for value in (ball, players, dead)]
        self.infer(ball, players, dead)
        for value, original in zip((ball, players, dead), originals):
            np.testing.assert_array_equal(value, original)
        for change in (dict(max_hold_frames=-1), dict(lookahead_frames=1),
                       dict(min_observed_fraction=2.), dict(max_observation_gap_frames=True),
                       dict(min_active_player_fraction=0.), dict(max_fit_error_m=np.inf)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(resumption_config(), **change)
