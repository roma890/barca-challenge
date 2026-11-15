"""Synthetic scenes for the offline H/A/D decoder; no match data is read."""

from dataclasses import replace
import json
import tempfile
import unittest

import numpy as np

from possession.contacts import Contact, ContactConfig
from possession.viterbi import ViterbiConfig, decode_possession, load_viterbi_config, touch_evidence

FPS = 25
PITCH = (105.3, 68.)
TEAMS = np.array(["H", "H", "A", "A"])


def config():
    return ViterbiConfig(team_gain=0.2, margin_m=1.0, reach_m=3.0, walk_mps=1.3, dead_gain=0.2,
                         max_speed_gap_mps=2.0, window_s=1.0, switch_cost=5.0,
                         dead_switch_cost=10.0, max_player_speed_mps=12.0, outside_gain=0.0,
                         outside_margin_m=0.5, damp_team_when_still=False, still_mps=0.8,
                         active_mps=2.0, exit_gain=0.0, line_zone_m=2.0, exit_speed_mps=2.0,
                         exit_confirm_s=1.0, exit_max_s=30.0, setup_gain=0.0, setup_zone_m=1.0,
                         setup_ball_mps=0.5, touch_gain=0.0, touch_window_s=0.2)


CONTACTS = ContactConfig(window_frames=2, max_distance_m=1.0, attribution_margin_m=0.25,
                         min_velocity_change_mps=2.0, max_fit_error_m=0.1, max_ball_speed_mps=40.0,
                         max_player_speed_mps=12.0)


def touch_scene():
    """H passes to an A player who redirects it first time to another H player; others jog."""
    n = 300
    players = np.empty((n, 4, 2))
    players[:] = [[5., 20.], [30., 40.], [30., 20.5], [60., 5.]]
    jog = np.arange(n) * 3. / FPS
    for slot in (0, 1, 3):
        players[:, slot, 1] += jog
    ball = np.empty((n, 2))
    ball[:100] = players[:100, 0] + [0.3, 0.]
    touch_point = players[149, 2] + [0., -0.3]
    ball[100:150] = np.linspace(ball[99], touch_point, 51)[1:]
    ball[150:200] = np.linspace(touch_point, players[199, 1] + [0.3, 0.], 51)[1:]
    ball[200:] = players[200:, 1] + [0.3, 0.]
    return ball, players


def extras():
    return replace(config(), outside_gain=0.4, damp_team_when_still=True)


def border():
    return replace(extras(), exit_gain=0.4, setup_gain=0.2)


def touchline_scene(live=100, out=10, hidden=250, after=200, stop_speed=1.5):
    """H dribbles 1.5 m from the bottom touchline, the ball goes out, is hidden, then A plays."""
    n = live + out + hidden + after
    speeds = np.r_[np.full(live + out, 3.), np.full(hidden, stop_speed), np.full(after, 3.)]
    start = np.array([[10., 3.], [10., 30.], [40., 20.], [40., 40.]])
    x = np.r_[0., np.cumsum(speeds[1:] / FPS)]
    players = start[None] + np.stack([x, np.zeros(n)], axis=1)[:, None, :]
    ball = np.full((n, 2), np.nan)
    ball[:live] = players[:live, 0] + [0.3, -1.5]
    ball[live:live + out] = ball[live - 1] + np.outer(np.arange(1, out + 1), [0., -0.2])
    ball[live + out + hidden:] = players[live + out + hidden:, 2] + [0.3, 0.]
    return ball, players


def scene(owners, speed_mps=3.0):
    """Players jog in x; the ball sits beside the first H or first A player (None: far from all)."""
    n = len(owners)
    start = np.array([[10., 10.], [10., 30.], [40., 10.], [40., 30.]])
    speeds = np.broadcast_to(np.asarray(speed_mps, dtype=float), (n,))
    x = np.r_[0., np.cumsum(speeds[1:] / FPS)]
    players = start[None] + np.stack([x, np.zeros(n)], axis=1)[:, None, :]
    ball = np.empty((n, 2))
    for frame, owner in enumerate(owners):
        if owner is None:
            ball[frame] = [25., 55.]
        else:
            ball[frame] = players[frame, 0 if owner == "H" else 2] + [0.3, 0.]
    return ball, players


def decode(ball, players, settings=None):
    return decode_possession(ball, players, TEAMS, fps=FPS, pitch=PITCH,
                             config=config() if settings is None else settings).labels


class ViterbiTests(unittest.TestCase):
    def test_sustained_dribble_is_one_possession(self):
        labels = decode(*scene(["H"] * 200))
        self.assertTrue((labels == "H").all())

    def test_change_lands_where_evidence_flips(self):
        labels = decode(*scene(["H"] * 150 + ["A"] * 150))
        change = int(np.flatnonzero(labels == "A")[0])
        self.assertLessEqual(abs(change - 150), 2)
        self.assertTrue((labels[:change] == "H").all() and (labels[change:] == "A").all())

    def test_brief_flicker_is_ignored(self):
        labels = decode(*scene(["H"] * 100 + ["A"] * 3 + ["H"] * 100))
        self.assertTrue((labels == "H").all())

    def test_short_real_possession_survives(self):
        labels = decode(*scene(["H"] * 100 + ["A"] * 38 + ["H"] * 100))
        self.assertTrue((labels[105:133] == "A").all())

    def test_group_stillness_with_ball_away_from_players_is_dead(self):
        speeds = np.r_[np.full(150, 3.), np.zeros(250), np.full(150, 3.)]
        labels = decode(*scene(["H"] * 150 + [None] * 250 + ["A"] * 150, speeds))
        self.assertTrue((labels[180:370] == "D").all())
        self.assertEqual(labels[0], "H")
        self.assertEqual(labels[-1], "A")

    def test_missing_ball_keeps_the_surrounding_team(self):
        ball, players = scene(["H"] * 300)
        ball[100:200] = np.nan
        self.assertTrue((decode(ball, players) == "H").all())

    def test_missing_players_give_no_team_evidence_and_neutral_speed(self):
        ball, players = scene(["H"] * 200)
        players[50:80] = np.nan
        self.assertTrue((decode(ball, players) == "H").all())

    def test_only_scored_labels_and_full_length(self):
        labels = decode(*scene(["H"] * 60 + [None] * 60 + ["A"] * 60))
        self.assertEqual(len(labels), 180)
        self.assertTrue(np.isin(labels, ["H", "A", "D"]).all())

    def test_higher_switch_cost_removes_short_possession(self):
        settings = replace(config(), switch_cost=20.)
        labels = decode(*scene(["H"] * 100 + ["A"] * 38 + ["H"] * 100), settings)
        self.assertTrue((labels == "H").all())

    def test_ball_outside_pitch_with_active_players_is_dead(self):
        ball, players = scene(["H"] * 150 + ["A"] * 300 + ["A"] * 150)
        ball[150:450] = [50., -5.]
        labels = decode(ball, players, extras())
        self.assertTrue((labels[170:430] == "D").all())
        self.assertFalse((decode(ball, players) == "D").any())

    def test_isolated_outside_glitch_is_not_dead(self):
        ball, players = scene(["H"] * 300)
        ball[150:153] = [50., -5.]
        self.assertTrue((decode(ball, players, extras()) == "H").all())

    def test_ball_walked_back_during_stoppage_is_dead_only_with_damping(self):
        speeds = np.r_[np.full(150, 3.), np.full(500, 0.9), np.full(150, 3.)]
        ball, players = scene(["H"] * 800, speeds)
        self.assertTrue((decode(ball, players) == "H").all())
        labels = decode(ball, players, replace(config(), damp_team_when_still=True))
        self.assertTrue((labels[180:620] == "D").all())
        self.assertEqual(labels[0], "H")

    def test_damping_keeps_jogging_possession(self):
        labels = decode(*scene(["H"] * 100 + ["A"] * 38 + ["H"] * 100), extras())
        self.assertTrue((labels[105:133] == "A").all())

    def test_ball_out_over_touchline_then_hidden_is_dead(self):
        ball, players = touchline_scene()
        labels = decode(ball, players, border())
        self.assertTrue((labels[130:350] == "D").all())
        self.assertEqual(labels[50], "H")
        self.assertEqual(labels[-1], "A")
        self.assertFalse((decode(ball, players, extras()) == "D").any())

    def test_pass_along_touchline_then_hidden_is_not_an_exit(self):
        ball, players = touchline_scene()
        ball[100:110] = ball[99] + np.outer(np.arange(1, 11), [0.2, 0.])
        self.assertFalse((decode(ball, players, border()) == "D").any())

    def test_ball_kept_in_after_moving_outwards_is_not_an_exit(self):
        ball, players = touchline_scene(hidden=250)
        ball[110:120] = ball[109] + np.outer(np.arange(1, 11), [0., 0.25])
        ball[120:360] = players[120:360, 0] + [0.3, 2.]
        labels = decode(ball, players, border())
        self.assertFalse((labels == "D").any())

    def test_ball_held_still_at_touchline_is_dead_only_with_setup(self):
        n = 150 + 375 + 150
        speeds = np.r_[np.full(150, 3.), np.full(375, 1.2), np.full(150, 3.)]
        ball, players = scene(["H"] * n, speeds)
        ball[150:525] = [50., 0.3]
        labels = decode(ball, players, border())
        self.assertTrue((labels[200:480] == "D").all())
        self.assertFalse((decode(ball, players, extras()) == "D").any())

    def test_touch_evidence_marks_a_window_around_each_touch(self):
        touch = lambda frame, team: Contact(frame, team, 0, 0.5, (0., 0.), (0., 0.), 5., 90., 0., 0.)
        out = touch_evidence([touch(10, "H"), touch(30, "A"), touch(50, "X")], 60, FPS, config())
        self.assertEqual(np.flatnonzero(out == 1).tolist(), [8, 9, 10, 11, 12])
        self.assertEqual(np.flatnonzero(out == -1).tolist(), [28, 29, 30, 31, 32])

    def test_touch_gain_requires_contact_settings(self):
        with self.assertRaises(ValueError):
            replace(config(), touch_gain=0.8)
        replace(config(), touch_gain=0.8, contacts=CONTACTS)

    def test_first_time_redirect_by_opponent_becomes_a_short_possession(self):
        ball, players = touch_scene()
        settings = replace(config(), switch_cost=3.5, dead_switch_cost=3.5)
        without = decode(ball, players, settings)
        with_touch = decode(ball, players, replace(settings, touch_gain=0.8, contacts=CONTACTS))
        self.assertFalse((without[140:160] == "A").any())
        self.assertTrue((with_touch[146:152] == "A").all())

    def test_invalid_settings_and_inputs(self):
        for change in [dict(team_gain=0), dict(switch_cost=-1), dict(walk_mps=np.nan), dict(reach_m=True),
                       dict(outside_gain=-0.1), dict(damp_team_when_still=1), dict(active_mps=0.5)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(config(), **change)
        replace(config(), outside_gain=0.)
        ball, players = scene(["H"] * 10)
        with self.assertRaises(ValueError):
            decode_possession(ball, players, TEAMS[:3], fps=FPS, pitch=PITCH, config=config())
        with self.assertRaises(ValueError):
            decode_possession(ball[:0], players[:0], TEAMS, fps=FPS, pitch=PITCH, config=config())
        with self.assertRaises(ValueError):
            decode_possession(ball, players, TEAMS, fps=FPS, pitch=(0., 68.), config=config())

    def test_config_file_rejects_unknown_or_missing_keys(self):
        values = vars(config()).copy()
        for broken in [dict(values, extra=1.), {k: v for k, v in values.items() if k != "team_gain"}]:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                json.dump(broken, handle)
            with self.subTest(keys=sorted(broken)), self.assertRaises(ValueError):
                load_viterbi_config(handle.name)

    def test_shipped_configs_load(self):
        for name in ("viterbi.json", "viterbi_v2.json", "viterbi_v3.json", "viterbi_v4.json", "viterbi_v5.json", "viterbi_v6.json"):
            self.assertIsInstance(load_viterbi_config("possession/configs/" + name), ViterbiConfig)


if __name__ == "__main__":
    unittest.main()
