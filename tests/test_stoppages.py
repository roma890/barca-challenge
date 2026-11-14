"""Synthetic restarts and counterexamples; no match/video/labels are read."""

from dataclasses import replace
import unittest

import numpy as np

from possession.stoppages import StoppageConfig, detect_stoppages


def stoppage_config():
    return StoppageConfig(
        min_preparation_frames=3, max_preparation_frames=10,
        max_stationary_speed_mps=0.3, max_preparation_drift_m=0.05,
        max_preparer_distance_m=0.5, min_release_speed_mps=2.,
        min_release_displacement_m=0.9, release_confirmation_frames=3,
        min_release_straightness=0.9, max_ball_speed_mps=40., max_player_speed_mps=12.,
        attribution_margin_m=0.2, location_tolerance_m=1.,
        goal_kick_depth_m=6., goal_kick_half_width_m=12.)


def restart_fixture(origin=(20., 20.), preparation=6):
    ball = np.tile(origin, (preparation + 5, 1)).astype(float)
    ball[preparation:, 0] += np.arange(1, 6) * 0.4
    players = np.tile([np.asarray(origin) + [0.2, 0], np.asarray(origin) + [10, 0]],
                      (len(ball), 1, 1))
    return ball, players


def set_piece(start=0, end=20, kind="Free Kick", team="N/A"):
    return {"code": "SET PIECES", "start_frame": start, "end_frame": end,
            "tags": {"Type": kind, "Team": team}}


class StoppageTests(unittest.TestCase):
    def detect(self, ball, players, events=None, config=None, **kwargs):
        return detect_stoppages(
            ball, players, ["H", "A"], fps=25, pitch=(105.3, 68.),
            config=stoppage_config() if config is None else config,
            events=[set_piece()] if events is None else events, **kwargs)

    def test_preparation_release_and_confirmation_are_distinct(self):
        ball, players = restart_fixture()
        result = self.detect(ball, players)
        self.assertEqual("".join(result.status), "DDDDDDLLLUU")
        event, = result.restarts
        self.assertEqual((event.preparation_start, event.release_frame, event.confirmation_frame), (0, 6, 8))
        self.assertEqual(result.restart_team[6], "H")
        self.assertEqual(np.flatnonzero(result.release).tolist(), [6])

    def test_rejection_counts_account_for_each_attempt_without_changing_predictions(self):
        ball, players = restart_fixture()
        result = self.detect(ball, players)
        counts = result.diagnostics
        rejections = sum(value for name, value in counts.items() if name.startswith("rejected_"))
        self.assertEqual(counts["tested_release_frames"], rejections + counts["accepted_restarts"])
        self.assertEqual(counts["accepted_restarts"], len(result.restarts))
        self.assertEqual("".join(result.status), "DDDDDDLLLUU")

    def test_diagnostics_explain_short_preparation(self):
        ball, players = restart_fixture()
        result = self.detect(ball, players, config=replace(stoppage_config(), min_preparation_frames=8))
        self.assertEqual(result.diagnostics["accepted_restarts"], 0)
        self.assertGreater(result.diagnostics["rejected_insufficient_stationary_preparation"], 0)
        self.assertEqual(result.diagnostics["longest_ball_preparation_frames"], 6)

    def test_live_hold_and_pass_without_tag_is_not_dead(self):
        ball, players = restart_fixture()
        result = self.detect(ball, players, events=[])
        self.assertFalse(result.dead.any())
        self.assertTrue((result.status == "U").all())

    def test_stationary_position_jitter_is_not_treated_as_real_ball_motion(self):
        ball, players = restart_fixture()
        ball[:6, 1] += np.array([0.02, -0.02, 0.02, -0.02, 0.02, -0.02])
        self.assertGreater(np.linalg.norm(np.diff(ball[:6], axis=0), axis=-1).min() * 25,
                           stoppage_config().max_stationary_speed_mps)
        result = self.detect(ball, players)
        self.assertEqual(result.restarts[0].release_frame, 6)
        self.assertTrue(result.dead[:6].all())

    def test_bounded_preparation_occlusion_uses_surrounding_evidence(self):
        ball, players = restart_fixture()
        ball[4] = np.nan
        config = replace(stoppage_config(), max_gap_frames=1)
        result = self.detect(ball, players, config=config)
        self.assertTrue(result.dead[4])
        self.assertEqual(result.restarts[0].preparation_source, "observed_ball")

    def test_bounded_release_occlusion_is_fitted_without_interpolating_positions(self):
        ball, players = restart_fixture()
        ball[7] = np.nan
        original = ball.copy()
        result = self.detect(ball, players, config=replace(stoppage_config(), max_gap_frames=1))
        self.assertEqual(result.restarts[0].release_frame, 6)
        np.testing.assert_array_equal(ball, original)

    def test_long_gap_does_not_pass_observed_ball_route(self):
        ball, players = restart_fixture()
        ball[2:5] = np.nan
        config = replace(stoppage_config(), min_preparation_frames=6, max_gap_frames=1,
                         preparation_observed_fraction=0.5)
        self.assertFalse(self.detect(ball, players, config=config).dead.any())

    def test_moving_ball_in_tagged_window_is_not_stationary_preparation(self):
        ball, players = restart_fixture()
        ball[:, 0] = 20 + np.arange(len(ball)) * 0.4
        self.assertFalse(self.detect(ball, players, config=replace(stoppage_config(), max_gap_frames=2)).dead.any())

    def test_hidden_ball_landmark_restart_uses_stationary_player_and_departure(self):
        ball, players = restart_fixture((0., 0.))
        ball[:6] = np.nan
        config = replace(stoppage_config(), allow_hidden_ball_preparation=True)
        result = self.detect(ball, players, config=config, events=[set_piece(kind="Corner Kick")])
        self.assertEqual(result.restarts[0].preparation_source, "hidden_ball")
        self.assertEqual(result.restarts[0].release_frame, 6)
        self.assertTrue(result.dead[:6].all())

    def test_hidden_preparation_extension_stops_at_conflicting_ball_evidence(self):
        ball, players = restart_fixture((0., 0.), preparation=10)
        ball[:10] = np.nan
        ball[2] = [5., 0.]
        result = self.detect(ball, players, config=replace(stoppage_config(),
                             allow_hidden_ball_preparation=True), events=[set_piece(kind="Corner Kick")])
        self.assertTrue((result.status[:3] == "U").all())
        self.assertTrue(result.dead[3:10].all())

    def test_hidden_ball_free_kick_without_landmark_is_insufficient(self):
        ball, players = restart_fixture()
        ball[:6] = np.nan
        config = replace(stoppage_config(), allow_hidden_ball_preparation=True)
        self.assertFalse(self.detect(ball, players, config=config).dead.any())

    def test_hidden_ball_does_not_override_contradictory_observation(self):
        ball, players = restart_fixture((0., 0.))
        ball[:6] = np.nan
        ball[3] = [3., 0.]
        config = replace(stoppage_config(), allow_hidden_ball_preparation=True)
        result = self.detect(ball, players, config=config, events=[set_piece(kind="Corner Kick")])
        self.assertFalse(result.dead.any())

    def test_hidden_ball_and_missing_or_moving_players_are_insufficient(self):
        for condition in ["missing", "moving", "dribbling"]:
            ball, players = restart_fixture((0., 0.))
            ball[:6] = np.nan
            if condition == "missing":
                players[:6, 0] = np.nan
            elif condition == "moving":
                players[:6, 0, 0] = np.arange(6) * 0.2 - 0.8
            else:
                players[6:, 0] = ball[6:] + [0.2, 0]
            config = replace(stoppage_config(), allow_hidden_ball_preparation=True)
            result = self.detect(ball, players, config=config, events=[set_piece(kind="Corner Kick")])
            with self.subTest(condition=condition):
                self.assertFalse(result.dead.any())

    def test_new_settings_validation(self):
        for change in [dict(max_gap_frames=-1), dict(preparation_observed_fraction=0),
                       dict(preparation_inlier_fraction=1.5), dict(release_observed_fraction=np.nan),
                       dict(max_release_fit_error_m=0), dict(allow_hidden_ball_preparation=1)]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    replace(stoppage_config(), **change)

    def test_identical_live_hold_with_false_tag_is_known_failure_mode(self):
        # Positions cannot expose a whistle. The same tagged geometry can be a
        # live hold/pass; this conjunction is a fallible hypothesis, not proof.
        ball, players = restart_fixture()
        self.assertTrue(self.detect(ball, players).dead.any())

    def test_tag_and_stillness_without_release_remain_uncertain(self):
        ball, players = restart_fixture()
        ball[:] = ball[0]
        result = self.detect(ball, players)
        self.assertFalse(result.dead.any())
        self.assertIn("preparation_candidate", result.reason)

    def test_missing_ball_alone_is_not_dead(self):
        ball, players = restart_fixture()
        ball[:] = np.nan
        self.assertFalse(self.detect(ball, players).dead.any())

    def test_no_nearby_preparer_is_insufficient(self):
        ball, players = restart_fixture()
        self.assertFalse(self.detect(ball, players + 10).dead.any())

    def test_player_just_approaching_has_not_supplied_preparation(self):
        ball, players = restart_fixture()
        players[:6, 0, 0] += [2., 1.6, 1.2, 0.8, 0.4, 0.]
        self.assertFalse(self.detect(ball, players).dead.any())

    def test_live_dribble_moving_with_ball_is_not_release(self):
        ball, players = restart_fixture()
        players[:, 0] = ball + [0.2, 0]
        self.assertFalse(self.detect(ball, players).dead.any())

    def test_sustained_release_needs_future_observations(self):
        ball, players = restart_fixture()
        self.assertFalse(self.detect(ball[:7], players[:7]).dead.any())

    def test_small_adjustment_and_oscillation_are_not_release(self):
        ball, players = restart_fixture()
        for offsets in [[0.02, 0.04, 0.06, 0.08, 0.10], [0.4, 0., 0.4, 0., 0.4]]:
            ball[6:, 0] = 20 + np.asarray(offsets)
            with self.subTest(offsets=offsets):
                self.assertFalse(self.detect(ball, players).dead.any())

    def test_geometry_for_known_restart_types(self):
        cases = [("Kick Off (Start)", (52.65, 34.)), ("Kick Off (After Goal)", (52.65, 34.)),
                 ("Corner Kick", (0., 0.)), ("Throw-in", (20., 0.)), ("Goal Kick", (2., 34.))]
        for kind, origin in cases:
            with self.subTest(kind=kind):
                ball, players = restart_fixture(origin)
                self.assertEqual(len(self.detect(ball, players, events=[set_piece(kind=kind)]).restarts), 1)

    def test_wrong_landmark_blocks_specific_tag(self):
        ball, players = restart_fixture((20., 20.))
        for kind in ["Kick Off (Start)", "Corner Kick", "Throw-in", "Goal Kick"]:
            with self.subTest(kind=kind):
                self.assertFalse(self.detect(ball, players, events=[set_piece(kind=kind)]).dead.any())

    def test_no_assumption_that_tag_begin_or_end_is_boundary(self):
        ball, players = restart_fixture()
        result = self.detect(ball, players, events=[set_piece(start=2, end=7)])
        self.assertEqual("".join(result.status), "UUDDDDLLLUU")
        self.assertEqual(result.restarts[0].release_frame, 6)

    def test_half_open_interval_excludes_release_at_its_end(self):
        ball, players = restart_fixture()
        self.assertFalse(self.detect(ball, players, events=[set_piece(end=6)]).dead.any())

    def test_missing_tracking_and_reset_block_required_window(self):
        for changed in ["ball_preparation", "ball_release", "player_preparation", "reset"]:
            ball, players = restart_fixture()
            kwargs = {}
            if changed == "ball_preparation":
                ball[4] = np.nan
            elif changed == "ball_release":
                ball[7] = np.nan
            elif changed == "player_preparation":
                players[4, 0] = np.nan
            else:
                kwargs["reset"] = np.arange(len(ball)) == 7
            with self.subTest(changed=changed):
                self.assertFalse(self.detect(ball, players, **kwargs).dead.any())

    def test_old_gap_is_not_filled_when_later_preparation_is_observed(self):
        ball, players = restart_fixture(preparation=9)
        ball[2] = np.nan
        result = self.detect(ball, players)
        self.assertFalse(result.dead[2])
        self.assertEqual(result.restarts[0].preparation_start, 3)

    def test_ball_and_player_jumps_do_not_create_restart(self):
        ball, players = restart_fixture()
        jumped = ball.copy()
        jumped[6] += 10
        self.assertFalse(self.detect(jumped, players).dead.any())
        players[6, 0] += 10
        self.assertFalse(self.detect(ball, players).dead.any())

    def test_ambiguous_preparer_team_stays_unknown_at_release(self):
        ball, players = restart_fixture()
        players[:, 1] = [19.8, 20.]
        result = self.detect(ball, players)
        self.assertTrue(result.dead.any())
        self.assertEqual(result.restart_team[6], "X")
        self.assertEqual(result.restarts[0].track_index, -1)

    def test_tag_team_string_never_supplies_ownership(self):
        ball, players = restart_fixture((20., 0.))
        for team in ["N/A", "Away", "Home"]:
            result = self.detect(ball, players, events=[set_piece(kind="Throw-in", team=team)])
            self.assertEqual(result.restart_team[6], "H")

    def test_duplicate_overlapping_tags_do_not_duplicate_restart(self):
        ball, players = restart_fixture()
        events = [set_piece(), set_piece(), set_piece(start=1)]
        self.assertEqual(len(self.detect(ball, players, events=events).restarts), 1)

    def test_preparation_is_bounded_and_does_not_invent_whistle_time(self):
        ball, players = restart_fixture(preparation=30)
        result = self.detect(ball, players, events=[set_piece(end=len(ball))])
        self.assertEqual(result.restarts[0].preparation_start, 20)
        self.assertTrue((result.status[:20] == "U").all())

    def test_non_set_piece_unknown_type_and_outside_intervals_are_ignored(self):
        ball, players = restart_fixture()
        for event in [dict(set_piece(), code="BUILD UP"), set_piece(kind="Unknown"),
                      set_piece(start=20, end=30)]:
            self.assertFalse(self.detect(ball, players, events=[event]).dead.any())

    def test_validation_and_empty_sequences(self):
        ball, players = restart_fixture()
        for event in [set_piece(start=1.5), set_piece(end=0), {"code": "SET PIECES", "tags": ["bad"]}]:
            with self.assertRaises(ValueError):
                self.detect(ball, players, events=[event])
        result = self.detect(np.empty((0, 2)), np.empty((0, 2, 2)), events=[])
        self.assertEqual(len(result.status), 0)
        for change in [dict(min_preparation_frames=1), dict(release_confirmation_frames=1),
                       dict(max_preparation_frames=2), dict(min_release_straightness=2),
                       dict(min_release_speed_mps=0.1), dict(max_ball_speed_mps=np.inf)]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    replace(stoppage_config(), **change)


if __name__ == "__main__":
    unittest.main()
