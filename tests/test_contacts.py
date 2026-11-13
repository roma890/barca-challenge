"""Synthetic contacts; no competition assets are opened."""

import unittest

import numpy as np

from possession.contacts import ContactConfig, detect_contacts


def contact_config():
    # Toy settings, not validated operating thresholds.
    return ContactConfig(window_frames=2, max_distance_m=0.4,
                         attribution_margin_m=0.15, min_velocity_change_mps=2.,
                         max_fit_error_m=0.02, max_ball_speed_mps=40.,
                         max_player_speed_mps=12.)


def redirect():
    """Ball reaches origin at frame 3, then turns 90 degrees without slowing."""
    ball = np.array([[-1.2, 0], [-0.8, 0], [-0.4, 0], [0, 0],
                     [0, 0.4], [0, 0.8], [0, 1.2], [0, 1.6]])
    players = np.tile([[10., 10.], [0.2, 0.]], (len(ball), 1, 1))
    return ball, players


class ContactTests(unittest.TestCase):
    def detect(self, ball, players, **kwargs):
        return detect_contacts(ball, players, ["H", "A"], fps=25,
                               config=contact_config(), **kwargs)

    def test_local_fits_locate_contact_and_report_vector_change(self):
        ball, players = redirect()
        events = self.detect(ball, players)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual((event.frame, event.team, event.track_index), (3, "A", 1))
        np.testing.assert_allclose(event.incoming_velocity_mps, [10., 0.])
        np.testing.assert_allclose(event.outgoing_velocity_mps, [0., 10.])
        self.assertAlmostEqual(event.turn_angle_deg, 90.)
        self.assertAlmostEqual(event.speed_change_mps, 0.)
        self.assertAlmostEqual(event.velocity_change_mps, np.sqrt(200.))

    def test_no_nearby_player_no_contact(self):
        ball, players = redirect()
        self.assertEqual(self.detect(ball, players + 10), ())

    def test_straight_pass_near_player_is_not_a_detected_touch(self):
        ball, players = redirect()
        ball[:, 0] = np.arange(len(ball)) * 0.4 - 1.2
        ball[:, 1] = 0
        self.assertEqual(self.detect(ball, players), ())

    def test_opponents_near_contact_are_unattributed(self):
        ball, players = redirect()
        players[:, 0] = [-0.2, 0]
        event, = self.detect(ball, players)
        self.assertEqual((event.team, event.track_index), ("X", -1))

    def test_missing_observation_blocks_window(self):
        for frame in [1, 3, 5]:
            ball, players = redirect()
            ball[frame, 1] = np.nan
            with self.subTest(frame=frame):
                self.assertEqual(self.detect(ball, players), ())

    def test_dead_or_reset_blocks_window(self):
        ball, players = redirect()
        mask = np.zeros(len(ball), dtype=bool)
        mask[4] = True
        for name in ["dead", "reset"]:
            with self.subTest(name=name):
                self.assertEqual(self.detect(ball, players, **{name: mask}), ())

    def test_ball_jump_is_not_physical_contact(self):
        ball, players = redirect()
        ball[3] = [10., 0.]
        players[:, 1] = [10., 0.]
        self.assertEqual(self.detect(ball, players), ())

    def test_player_jump_or_reappearance_prevents_attribution(self):
        for previous in [[10., 0.], [np.nan, np.nan]]:
            ball, players = redirect()
            players[2, 1] = previous
            with self.subTest(previous=previous):
                event, = self.detect(ball, players)
                self.assertEqual(event.team, "X")

    def test_noisy_spike_does_not_pass_local_fit(self):
        ball, players = redirect()
        ball[2] += [0., 0.3]
        self.assertEqual(self.detect(ball, players), ())

    def test_zero_speed_contact_has_undefined_angle(self):
        ball, players = redirect()
        ball[:4] = [0., 0.]
        event, = self.detect(ball, players)
        self.assertTrue(np.isnan(event.turn_angle_deg))

    def test_empty_and_short_recordings(self):
        for n in [0, 1, 4]:
            self.assertEqual(self.detect(np.zeros((n, 2)), np.zeros((n, 2, 2))), ())

    def test_validation(self):
        ball, players = redirect()
        for kwargs in [{"reset": np.zeros(len(ball))}, {"dead": [False]}, {"fps": 0}]:
            args = dict(fps=25, config=contact_config())
            args.update(kwargs)
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    detect_contacts(ball, players, ["H", "A"], **args)
        for name, value in [("window_frames", 1), ("window_frames", 2.5),
                            ("max_fit_error_m", -1), ("max_player_speed_mps", np.inf),
                            ("max_ball_speed_mps", True), ("max_distance_m", "1")]:
            args = vars(contact_config()).copy()
            args[name] = value
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    ContactConfig(**args)


if __name__ == "__main__":
    unittest.main()
