"""Whole synthetic stoppage episodes and contradictory live sequences."""

from dataclasses import replace
import unittest

import numpy as np

from possession.contacts import ContactConfig
from possession.dead_episodes import EpisodeConfig, infer_dead_episodes
from possession.play_state import PlayStateConfig
from possession.stoppages import Restart, StoppageResult
from tests.test_stoppages import stoppage_config
from possession.play_state import infer_play_state


def episode_config():
    return EpisodeConfig(lookback_frames=30, goal_pair_max_frames=100,
                         min_outside_frames=2, max_observation_gap_frames=1,
                         min_activity_drop=0.4, ball_quiet_radius_m=0.2,
                         goal_corridor_half_width_m=5., min_live_sequence_travel_m=2.,
                         max_live_sequence_gap_frames=30, max_live_sequence_fit_error_m=0.2)


def motion_config():
    return PlayStateConfig(window_frames=3, min_observed_fraction=0.5, min_players=2,
                           quiet_ball_speed_mps=0.5, moving_ball_speed_mps=3.,
                           quiet_player_speed_mps=0.8, active_player_speed_mps=2.,
                           tag_lookback_frames=0, outside_pitch_margin_m=0.1,
                           transition_cost=0.1)


def restart_evidence(n, restart):
    status = np.full(n, "U", dtype="<U1")
    status[restart.preparation_start:restart.release_frame] = "D"
    status[restart.release_frame:restart.confirmation_frame + 1] = "L"
    team = np.full(n, "X", dtype="<U1")
    team[restart.release_frame] = restart.team
    return StoppageResult(status, status.copy(), team, (restart,))


class DeadEpisodeTests(unittest.TestCase):
    def fixture(self):
        ball = np.full((80, 2), np.nan)
        ball[:10] = [20., 1.]
        ball[10:12] = [[20., -0.2], [20., -0.4]]
        ball[35:40] = [20., 0.]
        ball[40:45] = np.column_stack((20. + np.arange(1, 6) * 0.4, np.zeros(5)))
        players = np.tile([[50., 30.], [60., 30.]], (80, 1, 1))
        restart = Restart(35, 40, 42, "throw-in", "H", 0)
        return ball, players, restart_evidence(80, restart)

    def infer(self, ball, players, restarts, *, config=None, **kwargs):
        return infer_dead_episodes(
            ball, players, ["H", "A"] if players.shape[1] == 2 else ["H", "A", "H"],
            fps=10, pitch=(100., 60.), config=config or episode_config(),
            motion_config=motion_config(), contact_config=ContactConfig(window_frames=2, max_distance_m=0.4, attribution_margin_m=0.15,
                                        min_velocity_change_mps=2., max_fit_error_m=0.02,
                                        max_ball_speed_mps=40., max_player_speed_mps=12.),
            restarts=restarts, **kwargs)

    def test_observed_exit_extends_to_preparation_without_extending_past_restart(self):
        ball, players, restarts = self.fixture()
        result = self.infer(ball, players, restarts)
        episode, = result.episodes
        self.assertEqual((episode.onset, episode.preparation_start, episode.source), (10, 35, "observed_exit"))
        self.assertTrue((result.support[10:35] == 1).all())
        self.assertFalse(result.support[35:].any())
        self.assertFalse(result.support[:10].any())

    def test_retrieval_motion_is_not_itself_a_live_play_veto(self):
        ball, players, restarts = self.fixture()
        players[15:30, :, 0] += np.arange(15)[:, None] * 0.3
        ball[20:26] = np.column_stack((np.full(6, 20.), np.linspace(-2., 0., 6)))
        result = self.infer(ball, players, restarts)
        self.assertEqual(result.episodes[0].onset, 10)
        self.assertTrue((result.support[20:26] > 0).all())

    def test_near_boundary_then_missing_is_not_an_observed_exit(self):
        ball, players, restarts = self.fixture()
        ball[10:12, 1] = [0.05, 0.02]
        self.assertFalse(self.infer(ball, players, restarts).support.any())

    def test_isolated_outside_tracking_spike_is_not_exit_support(self):
        ball, players, restarts = self.fixture()
        ball[11] = [20., 0.5]
        self.assertFalse(self.infer(ball, players, restarts).support.any())

    def test_wrong_side_restart_does_not_match_exit(self):
        ball, players, restarts = self.fixture()
        ball[35:40, 1] = 60.
        self.assertFalse(self.infer(ball, players, restarts).support.any())

    def test_onset_bounds_preserve_short_observation_gap(self):
        ball, players, restarts = self.fixture()
        ball[9] = np.nan
        episode, = self.infer(ball, players, restarts).episodes
        self.assertEqual((episode.onset_earliest, episode.onset_latest, episode.onset), (9, 10, 10))

    def test_reset_or_lookback_limit_blocks_old_exit(self):
        ball, players, restarts = self.fixture()
        reset = np.arange(len(ball)) == 20
        self.assertFalse(self.infer(ball, players, restarts, reset=reset).support.any())
        config = replace(episode_config(), lookback_frames=10)
        self.assertFalse(self.infer(ball, players, restarts, config=config).support.any())

    def test_intervening_confirmed_release_blocks_backdating(self):
        ball, players, restarts = self.fixture()
        previous = Restart(18, 20, 22, "free kick", "A", 1)
        restarts = replace(restarts, restarts=(previous,) + restarts.restarts)
        self.assertFalse(self.infer(ball, players, restarts).support.any())

    def test_absent_restart_or_absent_onset_does_not_create_episode(self):
        ball, players, restarts = self.fixture()
        self.assertFalse(self.infer(ball, players, replace(restarts, restarts=())).support.any())
        ball[:35] = np.nan
        self.assertFalse(self.infer(ball, players, restarts).support.any())

    def test_group_deceleration_plus_hidden_ball_is_weaker_onset(self):
        ball, players, _ = self.fixture()
        ball[:35] = np.nan
        players[:20, :, 0] += np.arange(20)[:, None] * 0.3
        players[20:, :, 0] += 19 * 0.3
        restart = Restart(35, 40, 42, "free kick", "H", 0)
        result = self.infer(ball, players, restart_evidence(80, restart))
        episode, = result.episodes
        self.assertEqual((episode.onset, episode.source), (20, "group_pause"))
        self.assertLess(episode.strength, 1.)
        self.assertLess(episode.onset_earliest, episode.onset_latest)

    def goal_fixture(self):
        ball = np.full((130, 2), np.nan)
        ball[:20] = [98., 30.]
        ball[20:22] = [[100.2, 30.], [100.4, 30.]]
        ball[100:105] = [50., 30.]
        players = np.tile([[50., 30.], [60., 30.]], (130, 1, 1))
        restart = Restart(100, 105, 107, "kick off (after goal)", "H", 0)
        event = {"code": "GOALS", "start_frame": 18, "end_frame": 25, "tags": {}}
        return ball, players, restart_evidence(130, restart), [event]

    def test_goal_context_physical_exit_and_supported_kickoff_form_episode(self):
        ball, players, restarts, events = self.goal_fixture()
        episode, = self.infer(ball, players, restarts, events=events).episodes
        self.assertEqual((episode.onset, episode.preparation_start, episode.source), (20, 100, "goal_then_kickoff"))

    def test_goal_tag_alone_or_kickoff_alone_is_insufficient(self):
        ball, players, restarts, events = self.goal_fixture()
        self.assertFalse(self.infer(ball, players, restarts).support.any())
        self.assertFalse(self.infer(ball, players, replace(restarts, restarts=()), events=events).support.any())
        ball[:100] = np.nan
        self.assertFalse(self.infer(ball, players, restarts, events=events).support.any())

    def test_goal_tag_start_is_not_the_onset_and_wide_exit_is_not_goal_support(self):
        ball, players, restarts, events = self.goal_fixture()
        events[0]["start_frame"] = 0
        self.assertEqual(self.infer(ball, players, restarts, events=events).episodes[0].onset, 20)
        ball[20:22, 1] = 10.
        self.assertFalse(self.infer(ball, players, restarts, events=events).support.any())

    def test_goal_pair_cannot_cross_reset_or_maximum_horizon(self):
        ball, players, restarts, events = self.goal_fixture()
        self.assertFalse(self.infer(ball, players, restarts, events=events,
                                    reset=np.arange(len(ball)) == 60).support.any())
        self.assertFalse(self.infer(ball, players, restarts, events=events,
                                    config=replace(episode_config(), goal_pair_max_frames=50)).support.any())

    def test_coherent_live_transfer_sequence_blocks_extension(self):
        ball = np.full((60, 2), np.nan)
        ball[:2] = [20., 1.]
        ball[2:4] = [[20., -0.2], [20., -0.4]]
        ball[4:8] = [[18.4, 3.], [18.4, 6.], [18.4, 9.], [18.4, 10.]]
        for frame in range(8, 12):
            ball[frame] = [18.4 + (frame - 7) * 0.4, 10.]
        for frame in range(12, 22):
            ball[frame] = [20., 10. + (frame - 11) * 0.4]
        for frame in range(22, 32):
            ball[frame] = [20. + (frame - 21) * 0.4, 14.]
        for frame in range(32, 36):
            ball[frame] = [24., 14. + (frame - 31) * 0.4]
        ball[45:50] = [20., 0.]
        players = np.tile([[20.1, 10.], [20.1, 14.], [24.1, 14.]], (60, 1, 1))
        restarts = restart_evidence(60, Restart(45, 50, 52, "throw-in", "H", 0))
        result = self.infer(ball, players, restarts, config=replace(episode_config(), lookback_frames=50))
        self.assertGreater(result.diagnostics["live_sequence_guard_frames"], 0)
        self.assertGreater(result.diagnostics["rejected_by_live_sequence"], 0)
        self.assertFalse(result.support.any())

    def test_configuration_validation_and_input_arrays_unchanged(self):
        ball, players, restarts = self.fixture()
        original = ball.copy()
        self.infer(ball, players, restarts)
        np.testing.assert_array_equal(ball, original)
        for change in [dict(lookback_frames=0), dict(min_outside_frames=True),
                       dict(min_activity_drop=2), dict(max_observation_gap_frames=-1)]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    replace(episode_config(), **change)

    def weak_fixture(self):
        ball, players, restarts = self.fixture()
        players[:, 0] = [20.2, 0.]
        restarts = replace(restarts, status=np.full(len(ball), "U", dtype="<U1"), restarts=())
        tag = {"code": "SET PIECES", "start_frame": 30, "end_frame": 50,
               "tags": {"Type": "Throw-in"}}
        config = replace(episode_config(), allow_weak_restarts=True)
        return ball, players, restarts, [tag], config

    def test_weak_candidate_does_not_require_a_strict_restart_anchor(self):
        ball, players, restarts, events, config = self.weak_fixture()
        result = self.infer(ball, players, restarts, config=config, events=events,
                            restart_config=stoppage_config())
        self.assertEqual(len(result.weak_candidates), 1)
        candidate = result.weak_candidates[0]
        self.assertEqual((candidate.onset, candidate.release_frame), (10, 40))
        self.assertTrue((result.weak_support[10:40] == config.weak_strength).all())
        self.assertFalse(result.motion_discount.any())
        self.assertFalse(result.episodes)

    def test_each_of_tag_onset_and_restart_action_is_required(self):
        for absent in ("tag", "onset", "action"):
            ball, players, restarts, events, config = self.weak_fixture()
            if absent == "tag":
                events = []
            elif absent == "onset":
                ball[:35] = np.nan
            else:
                ball[40:] = np.nan
            result = self.infer(ball, players, restarts, config=config, events=events,
                                restart_config=stoppage_config())
            with self.subTest(absent=absent):
                self.assertFalse(result.weak_support.any())

    def test_weak_restart_can_have_a_missing_launch_observation(self):
        ball, players, restarts, events, config = self.weak_fixture()
        ball[41] = np.nan
        result = self.infer(ball, players, restarts, config=config, events=events,
                            restart_config=replace(stoppage_config(), max_gap_frames=1))
        self.assertTrue(result.weak_candidates)

    def test_weak_uncertainty_contributes_to_score_without_suppressing_live_cues(self):
        ball, players, restarts, events, config = self.weak_fixture()
        # One moving player and a coherent ball flight provide live opposition.
        players[:, 1, 0] += np.arange(len(ball)) * 0.3
        ball[15:25] = np.column_stack((30. + np.arange(10) * 0.4, np.full(10, 30.)))
        events.append({"code": "BUILD UP", "start_frame": 0, "end_frame": 80, "tags": {}})
        result = self.infer(ball, players, restarts, config=config, events=events,
                            restart_config=stoppage_config())
        self.assertTrue(result.weak_support[17:23].all())
        base = infer_play_state(ball, players, ["H", "A"], fps=10, pitch=(100., 60.),
                                config=motion_config(), events=events)
        proposed = infer_play_state(ball, players, ["H", "A"], fps=10, pitch=(100., 60.),
                                    config=motion_config(), events=events, episodes=result)
        self.assertTrue((proposed.dead_score[17:23] > base.dead_score[17:23]).all())
        self.assertFalse(proposed.dead[17:23].any())
        self.assertTrue(proposed.live_conflict_proxy[17:23].all())

    def test_single_outside_observation_remains_weak_not_confirmed(self):
        ball, players, restarts, events, config = self.weak_fixture()
        ball[11] = [20., 0.5]
        result = self.infer(ball, players, restarts, config=config, events=events,
                            restart_config=stoppage_config())
        self.assertFalse(result.episodes)
        self.assertTrue(result.weak_candidates)
        self.assertLess(result.weak_support.max(), 1.)

    def test_weak_specific_settings_are_validated(self):
        for change in (dict(weak_strength=1.), dict(allow_weak_restarts=1),
                       dict(weak_min_slowdown_ratio=2.), dict(weak_min_outside_frames=0)):
            with self.assertRaises(ValueError):
                replace(episode_config(), **change)

    def test_relative_group_slowdown_can_seed_weak_free_kick_without_all_players_quiet(self):
        n = 70
        ball = np.full((n, 2), np.nan)
        players = np.tile([[20., 20.], [30., 20.]], (n, 1, 1))
        players[:20, :, 0] += np.arange(20)[:, None] * 0.3
        players[20:, :, 0] += 19 * 0.3 + np.arange(1, n - 19)[:, None] * 0.12
        origin = players[50, 0] - [0.2, 0]
        ball[50:55] = origin + np.column_stack((np.arange(1, 6) * 0.4, np.zeros(5)))
        status = np.full(n, "U", dtype="<U1")
        restarts = StoppageResult(status, status.copy(), np.full(n, "X"), ())
        events = [{"code": "SET PIECES", "start_frame": 30, "end_frame": 60,
                   "tags": {"Type": "Free Kick"}}]
        config = replace(episode_config(), allow_weak_restarts=True)
        result = self.infer(ball, players, restarts, config=config, events=events,
                            restart_config=stoppage_config())
        self.assertEqual(result.diagnostics["pause_candidates"], 0)
        self.assertTrue(result.weak_candidates)
        self.assertEqual(result.weak_candidates[0].onset_source, "group_slowdown")

    def test_weak_candidate_is_blocked_by_coherent_live_passing(self):
        ball = np.full((60, 2), np.nan)
        ball[:2] = [20., 1.]
        ball[2:4] = [[20., -0.2], [20., -0.4]]
        ball[4:8] = [[18.4, 3.], [18.4, 6.], [18.4, 9.], [18.4, 10.]]
        for frame in range(8, 12):
            ball[frame] = [18.4 + (frame - 7) * 0.4, 10.]
        for frame in range(12, 22):
            ball[frame] = [20., 10. + (frame - 11) * 0.4]
        for frame in range(22, 32):
            ball[frame] = [20. + (frame - 21) * 0.4, 14.]
        for frame in range(32, 36):
            ball[frame] = [24., 14. + (frame - 31) * 0.4]
        ball[45:50] = [20., 0.]
        ball[50:55] = np.column_stack((20. + np.arange(1, 6) * 0.4, np.zeros(5)))
        players = np.tile([[20.1, 10.], [20.1, 14.], [24.1, 14.]], (60, 1, 1))
        players[45:, 0] = [20.1, 0.]
        status = np.full(60, "U", dtype="<U1")
        restarts = StoppageResult(status, status.copy(), np.full(60, "X"), ())
        tag = {"code": "SET PIECES", "start_frame": 45, "end_frame": 60,
               "tags": {"Type": "Throw-in"}}
        result = self.infer(ball, players, restarts, config=replace(episode_config(),
                            lookback_frames=50, allow_weak_restarts=True), events=[tag],
                            restart_config=stoppage_config())
        self.assertGreater(result.diagnostics["weak_rejected_by_live_sequence"], 0)
        self.assertFalse(result.weak_candidates)


if __name__ == "__main__":
    unittest.main()
