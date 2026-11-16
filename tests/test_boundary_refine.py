"""Boundary refinement on invented scenes only; no competition assets are opened."""

import unittest

import numpy as np

from possession import boundary_refine as br
from possession.run_learned import build_parser

FPS = 25.
TEAMS = np.array(["H", "A"])
KICK, RECEIVE, N = 60, 100, 200


def pass_scene():
    """H dribbles at 2 m/s, kicks at frame 60 (15 m/s), A receives at frame 100 and dribbles."""
    t = np.arange(N, dtype=float)
    h_player = np.c_[10 + 2 * t / FPS, np.full(N, 30.)]
    ball = h_player + [0.3, 0.]
    ball[KICK:] = ball[KICK] + np.c_[15 * (t[KICK:] - KICK) / FPS, np.zeros(N - KICK)]
    reception = ball[RECEIVE].copy()
    ball[RECEIVE:] = reception + np.c_[2 * (t[RECEIVE:] - RECEIVE) / FPS, np.zeros(N - RECEIVE)]
    a_player = np.tile(reception + [0.3, 0.], (N, 1))
    a_player[RECEIVE:] = ball[RECEIVE:] + [0.3, 0.]
    return ball, np.stack([h_player, a_player], axis=1)


def labels_with(*spells):
    """labels_with(("H", 0), ("A", 90)) -> H before 90, A from 90."""
    out = np.empty(N, dtype="<U1")
    starts = [start for _, start in spells] + [N]
    for (label, start), stop in zip(spells, starts[1:]):
        out[start:stop] = label
    return out


def runs(labels):
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(str(labels[a]), int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def boundary(labels):
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return int(changes[0]) if len(changes) else None


def refine(labels, ball, players, teams=TEAMS, **kwargs):
    return br.refine(labels, ball, players, teams, fps=FPS, **kwargs)


class InvariantChecks:
    def assert_invariants(self, before, after, window=25):
        self.assertEqual(before.shape, after.shape)
        self.assertTrue(np.isin(after, ("H", "A", "D", "X")).all())
        np.testing.assert_array_equal(before == "D", after == "D")
        rb, ra = runs(before), runs(after)
        self.assertEqual([label for label, _, _ in rb], [label for label, _, _ in ra])
        self.assertTrue(all(stop > start for _, start, stop in ra))
        for (_, s0, _), (_, s1, _) in zip(rb, ra):
            self.assertLessEqual(abs(s0 - s1), window)


class ContactTimingTests(unittest.TestCase, InvariantChecks):
    def test_delayed_confirmation_moves_back_to_reception(self):
        ball, players = pass_scene()
        before = labels_with(("H", 0), ("A", RECEIVE + 20))
        after = refine(before, ball, players)
        self.assertEqual(boundary(after), RECEIVE)
        self.assert_invariants(before, after)

    def test_early_boundary_during_flight_moves_forward_to_reception(self):
        ball, players = pass_scene()
        before = labels_with(("H", 0), ("A", RECEIVE - 15))
        self.assertEqual(boundary(refine(before, ball, players)), RECEIVE)

    def test_window_limits_the_move(self):
        ball, players = pass_scene()
        before = labels_with(("H", 0), ("A", RECEIVE + 40))
        after = refine(before, ball, players)
        self.assertEqual(boundary(after), RECEIVE + 40)  # reception is outside +-1 s: unchanged
        after = refine(before, ball, players, config=br.RefineConfig(window_s=2.0))
        self.assertEqual(boundary(after), RECEIVE)


class MissingBallTests(unittest.TestCase, InvariantChecks):
    def hidden(self):
        ball, players = pass_scene()
        ball[KICK + 3:RECEIVE + 6] = np.nan  # the reception itself is never seen
        return ball, players

    def test_boundary_before_old_teams_last_touch_moves_after_it(self):
        ball, players = self.hidden()
        after = refine(labels_with(("H", 0), ("A", KICK - 10)), ball, players)
        self.assertEqual(boundary(after), KICK + 1)

    def test_boundary_inside_the_gap_is_left_alone(self):
        ball, players = self.hidden()
        before = labels_with(("H", 0), ("A", 80))
        np.testing.assert_array_equal(refine(before, ball, players), before)

    def test_late_boundary_moves_to_first_credible_new_control(self):
        ball, players = self.hidden()
        after = refine(labels_with(("H", 0), ("A", RECEIVE + 25)), ball, players)
        self.assertTrue(RECEIVE + 6 <= boundary(after) <= RECEIVE + 9, boundary(after))

    def test_no_observations_in_window_means_no_change(self):
        ball, players = pass_scene()
        ball[60:160] = np.nan
        before = labels_with(("H", 0), ("A", 110))
        np.testing.assert_array_equal(refine(before, ball, players), before)


class NoChangeTests(unittest.TestCase, InvariantChecks):
    def test_one_touch_pass_within_a_team_has_no_boundary_to_move(self):
        ball, players = pass_scene()
        players = players.copy()
        teams = np.array(["H", "H"])  # the receiver is a team-mate
        before = labels_with(("H", 0))
        np.testing.assert_array_equal(refine(before, ball, players, teams), before)

    def test_deflection_then_recovery_is_contested_and_unchanged(self):
        ball, players = pass_scene()
        # An A player deflects at frame 80 (ball turns), H recovers at 100.
        t = np.arange(N, dtype=float)
        ball[80:] = ball[80] + np.c_[12 * (t[80:] - 80) / FPS, 4 * (t[80:] - 80) / FPS]
        ball[RECEIVE:] = ball[RECEIVE] + np.c_[2 * (t[RECEIVE:] - RECEIVE) / FPS, np.zeros(N - RECEIVE)]
        deflector = np.tile(ball[80] + [0, -0.3], (N, 1))
        recoverer = np.tile(ball[RECEIVE] + [0.3, 0], (N, 1))
        recoverer[RECEIVE:] = ball[RECEIVE:] + [0.3, 0]
        players = np.stack([players[:, 0], deflector, recoverer], axis=1)
        teams = np.array(["H", "A", "H"])
        before = labels_with(("H", 0), ("A", 90))
        after, report = refine(before, ball, players, teams, return_report=True)
        np.testing.assert_array_equal(after, before)
        self.assertEqual(report[0]["reason"], "contested")

    def test_contested_ball_with_both_teams_close_is_unchanged(self):
        ball, players = pass_scene()
        players = players.copy()
        players[:, 0] = ball + [0, 0.35]   # H marker shadows the ball throughout: no attribution,
        players[:, 1] = ball + [0, -0.4]   # no control for either side (aerial / 50-50 duel)
        before = labels_with(("H", 0), ("A", 110))
        np.testing.assert_array_equal(refine(before, ball, players), before)


class StructureTests(unittest.TestCase, InvariantChecks):
    def test_short_spell_is_kept_and_neighbours_are_not_crossed(self):
        ball, players = pass_scene()
        before = labels_with(("H", 0), ("A", 90), ("H", 92))  # new-team evidence lies beyond 92
        after = refine(before, ball, players)
        self.assert_invariants(before, after)
        self.assertEqual([label for label, _, _ in runs(after)], ["H", "A", "H"])

    def test_dead_ball_boundaries_and_frames_are_never_touched(self):
        ball, players = pass_scene()
        for before in (labels_with(("H", 0), ("D", 90), ("A", 110)),
                       labels_with(("D", 0), ("H", 50), ("A", 115), ("D", 150))):
            after = refine(before, ball, players)
            np.testing.assert_array_equal(after == "D", before == "D")
            self.assert_invariants(before, after)
        before = labels_with(("H", 0), ("D", 90), ("A", 110))
        np.testing.assert_array_equal(refine(before, ball, players), before)

    def test_team_symmetry(self):
        ball, players = pass_scene()
        before = labels_with(("H", 0), ("A", RECEIVE + 20))
        swap = {"H": "A", "A": "H", "D": "D"}
        mirrored = np.array([swap[x] for x in before])
        after = refine(before, ball, players)
        after_swapped = refine(mirrored, ball, players, np.array(["A", "H"]))
        np.testing.assert_array_equal(after_swapped, np.array([swap[x] for x in after]))

    def test_deterministic_and_does_not_mutate_input(self):
        ball, players = pass_scene()
        before = labels_with(("H", 0), ("A", RECEIVE + 20))
        copy = before.copy()
        np.testing.assert_array_equal(refine(before, ball, players), refine(before, ball, players))
        np.testing.assert_array_equal(before, copy)

    def test_invariants_on_noisy_random_scenes(self):
        rng = np.random.default_rng(7)
        for trial in range(5):
            n = 600
            ball = np.cumsum(rng.normal(0, 0.3, (n, 2)), axis=0) + 50
            ball[rng.random(n) < 0.2] = np.nan
            players = ball[:, None, :] + rng.normal(0, 2.0, (n, 6, 2))
            teams = np.array(["H", "H", "H", "A", "A", "A"])
            labels = np.array(list("HADHAHHAAD"))[np.repeat(rng.integers(0, 10, n // 15), 15)]
            with self.subTest(trial=trial):
                after = br.refine(labels, ball, players, teams, fps=FPS)
                self.assert_invariants(labels, after)

    def test_config_and_input_validation(self):
        with self.assertRaises(ValueError):
            br.RefineConfig(window_s=0)
        with self.assertRaises(ValueError):
            br.RefineConfig(min_control_frames=0)
        ball, players = pass_scene()
        with self.assertRaises(ValueError):
            refine(labels_with(("H", 0))[:-1], ball, players)
        with self.assertRaises(ValueError):
            refine(np.full(N, "Q"), ball, players)

    def test_inference_flag_is_off_by_default(self):
        args = build_parser().parse_args(["--model", "m.json", "--output", "o.csv"])
        self.assertFalse(args.refine_boundaries)
        args = build_parser().parse_args(["--model", "m.json", "--output", "o.csv", "--refine-boundaries"])
        self.assertTrue(args.refine_boundaries)


if __name__ == "__main__":
    unittest.main()
