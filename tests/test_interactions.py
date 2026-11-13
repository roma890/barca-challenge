"""Paired synthetic scenarios, including explicitly unresolvable interactions."""

import unittest

import numpy as np

from possession.control import ControlConfig
from possession.interactions import InteractionConfig, infer_interactions
from possession.state import decode_state
from test_contacts import contact_config


class InteractionTests(unittest.TestCase):
    def infer(self, ball, players, teams, *, accept=False, **kwargs):
        return infer_interactions(
            ball, players, teams, fps=25, control_config=ControlConfig(0.4, 1., 0.15),
            contact_config=contact_config(), interaction_config=InteractionConfig(
                lookahead_frames=12, max_target_miss_m=0.25,
                min_outgoing_speed_mps=3., accept_pass_hypotheses=accept), **kwargs)

    def pass_sequence(self, receiver_team="A"):
        # First-time contact by A at frame 3; receiver collects at frame 8.
        ball = np.array([[-1.2, 0], [-0.8, 0], [-0.4, 0], [0, 0],
                         [0, 0.4], [0, 0.8], [0, 1.2], [0, 1.6],
                         [0, 2.0], [0, 2.04], [0, 2.08], [0, 2.12]])
        players = np.tile([[10., 10.], [0.2, 0.], [0.2, 2.]], (len(ball), 1, 1))
        players[9:, 2, 1] = ball[9:, 1]
        return ball, players, ["H", "A", receiver_team]

    def test_one_touch_pass_is_explicit_hypothesis_with_contact_and_confirmation(self):
        ball, players, teams = self.pass_sequence()
        result = self.infer(ball, players, teams)
        event = next(item for item in result.interactions if item.contact.frame == 3)
        self.assertEqual(event.kind, "pass_hypothesis")
        self.assertEqual(event.receiver_track_index, 2)
        self.assertEqual(event.confirmation_frame, 9)
        self.assertEqual(result.labels[3], "X")
        self.assertEqual(result.labels[9], "A")

    def test_opt_in_pass_inference_backdates_to_contact_not_completion(self):
        ball, players, teams = self.pass_sequence()
        result = self.infer(ball, players, teams, accept=True)
        self.assertTrue((result.labels[3:10] == "A").all())
        self.assertEqual(result.source[3], "pass_hypothesis")
        self.assertTrue((result.labels[:3] == "X").all())

    def test_identical_fortunate_deflection_is_not_claimed_distinguishable(self):
        # Give identical observations either semantic label: outputs MUST agree.
        ball, players, teams = self.pass_sequence()
        pass_result = self.infer(ball, players, teams)
        deflection_result = self.infer(ball.copy(), players.copy(), teams)
        np.testing.assert_array_equal(pass_result.labels, deflection_result.labels)
        self.assertEqual(pass_result.interactions, deflection_result.interactions)
        self.assertEqual(deflection_result.labels[3], "X")

    def test_fortunate_deflection_keeps_incumbent_until_receiver_control(self):
        ball, players, teams = self.pass_sequence()
        # H controlled the incoming pass, then ran past the A contact. The body
        # motion is continuous and below the toy speed limit.
        players[:, 0, 0] = np.arange(len(ball)) * 0.4 - 0.9
        players[3:, 0, 0] += 0.07
        players[:, 0, 1] = 0
        conservative = self.infer(ball, players, teams)
        state = decode_state(conservative.labels, np.zeros(len(ball), dtype=bool),
                             max_carry_frames=len(ball))
        self.assertTrue((state.labels[1:8] == "H").all())
        self.assertEqual(state.labels[8], "A")
        # The same observations can be interpreted as an intentional first-time
        # pass only under the explicit, fallible pass-promotion experiment.
        permissive = self.infer(ball, players, teams, accept=True)
        self.assertEqual(permissive.labels[3], "A")

    def test_failed_one_touch_pass_does_not_use_opponent_reception_as_confirmation(self):
        ball, players, teams = self.pass_sequence(receiver_team="H")
        result = self.infer(ball, players, teams, accept=True)
        event = next(item for item in result.interactions if item.contact.frame == 3)
        self.assertEqual(event.kind, "unresolved_contact")
        self.assertEqual(result.labels[3], "X")
        self.assertEqual(result.labels[9], "H")

    def test_clean_reception_can_be_backdated(self):
        ball = np.array([[-1.2, 0], [-0.8, 0], [-0.4, 0], [0, 0],
                         [0.04, 0], [0.08, 0], [0.12, 0], [0.16, 0]])
        players = np.tile([[10., 10.], [0.2, 0.]], (8, 1, 1))
        players[4:, 1] = ball[4:] + [0.2, 0]
        result = self.infer(ball, players, ["H", "A"])
        self.assertEqual(result.labels[3], "A")
        self.assertEqual(result.source[3], "reception_backdated")
        self.assertEqual(result.interactions[0].confirmation_frame, 4)

    def test_loose_rebound_then_chase_is_not_backdated_to_first_touch(self):
        # A's first touch redirects the ball, but it travels outside the control
        # radius before A catches it. Later control cannot justify the first touch.
        ball = np.array([[-1.2, 0], [-0.8, 0], [-0.4, 0], [0, 0],
                         [0, 0.4], [0, 0.8], [0, 1.2], [0, 1.6],
                         [0, 2.], [0, 2.4], [0, 2.8], [0, 3.2]])
        players = np.tile([[10., 10.], [0.2, 0.]], (len(ball), 1, 1))
        players[4:, 1, 1] = [0., 0.2, 0.5, 0.9, 1.35, 1.8, 2.25, 2.7]
        # Continue until A is within range and co-moving, without a speed jump.
        ball = np.vstack((ball, [[0, 3.6], [0, 4.], [0, 4.4]]))
        players = np.concatenate((players, np.array([[[10, 10], [0.2, 3.15]],
                                                    [[10, 10], [0.2, 3.6]],
                                                    [[10, 10], [0.2, 4.02]]])))
        players[:, 1, 0] = 0.1
        result = self.infer(ball, players, ["H", "A"])
        self.assertEqual(result.labels[3], "X")
        self.assertEqual(result.labels[-1], "A")
        self.assertEqual(result.interactions[0].kind, "unresolved_contact")

    def test_emergency_clearance_does_not_establish_control(self):
        ball, players, teams = self.pass_sequence()
        players[:, 2] = [10., -10.]
        result = self.infer(ball, players, teams, accept=True)
        self.assertTrue((result.labels == "X").all())
        self.assertTrue(all(item.kind == "unresolved_contact" for item in result.interactions))

    def test_prior_control_survives_clearance_in_state_decoder(self):
        ball = np.array([[0., 0.], [0.04, 0.], [0.08, 0.], [0.12, 0.],
                         [0.16, 0.], [0.56, 0.], [0.96, 0.], [1.36, 0.], [1.76, 0.]])
        players = np.tile([[10., 10.], [0., 0.]], (len(ball), 1, 1))
        players[:5, 1] = ball[:5] + [0.2, 0]
        players[5:, 1] = players[4, 1]
        result = self.infer(ball, players, ["H", "A"])
        self.assertEqual(result.labels[3], "A")
        state = decode_state(result.labels, np.zeros(len(ball), dtype=bool), max_carry_frames=4)
        self.assertTrue((state.labels[1:] == "A").all())

    def test_close_defender_does_not_end_established_dribble(self):
        ball = np.column_stack((np.arange(8) * 0.1, np.zeros(8)))
        players = np.stack((ball + [0.2, 0], ball + [0, 2.]), axis=1)
        players[3:, 1] = ball[3:] + [0, 0.3]
        result = self.infer(ball, players, ["H", "A"])
        self.assertEqual("".join(result.labels), "XHHHHHHH")
        self.assertEqual(result.source[-1], "incumbent_motion")

    def test_clean_tackle_overrides_incumbent(self):
        ball = np.column_stack((np.arange(8) * 0.1, np.zeros(8)))
        players = np.stack((ball + [0.2, 0], ball + [0, 2.]), axis=1)
        # Both move continuously; original carrier loses proximity as opponent
        # gains it. No fixed minimum duration delays the new direct evidence.
        players[3:, 0, 1] = np.arange(1, 6) * 0.3
        players[:, 1, 1] = np.maximum(0.2, 2. - np.arange(8) * 0.3)
        result = self.infer(ball, players, ["H", "A"])
        self.assertEqual(result.labels[-1], "A")

    def test_rapid_genuine_turnovers_preserve_short_middle_possession(self):
        ball = np.column_stack((np.arange(12) * 0.1, np.zeros(12)))
        players = np.repeat(ball[:, None, :], 2, axis=1)
        players[:, 0, 1] = [0.2, 0.2, 0.2, 0.5, 0.8, 0.8, 0.8, 0.8, 0.5, 0.2, 0.2, 0.2]
        players[:, 1, 1] = [1., 1., 0.7, 0.4, 0.2, 0.2, 0.2, 0.4, 0.7, 1., 1., 1.]
        result = self.infer(ball, players, ["H", "A"])
        self.assertEqual(result.labels[2], "H")
        self.assertEqual("".join(result.labels[5:7]), "AA")
        self.assertEqual(result.labels[10], "H")

    def test_missing_flight_observation_blocks_pass_confirmation(self):
        ball, players, teams = self.pass_sequence()
        ball[6] = np.nan
        result = self.infer(ball, players, teams, accept=True)
        event = next(item for item in result.interactions if item.contact.frame == 3)
        self.assertEqual(event.kind, "unresolved_contact")
        self.assertEqual(result.labels[3], "X")

    def test_receiver_track_gap_cannot_confirm_pass(self):
        ball, players, teams = self.pass_sequence()
        players[6, 2] = np.nan
        result = self.infer(ball, players, teams, accept=True)
        event = next(item for item in result.interactions if item.contact.frame == 3)
        self.assertEqual(event.kind, "unresolved_contact")

    def test_empty_no_players_and_single_frame(self):
        for frames, slots in [(0, 2), (1, 2), (3, 0)]:
            with self.subTest(frames=frames, slots=slots):
                result = self.infer(np.zeros((frames, 2)), np.zeros((frames, slots, 2)),
                                    ["H"] * slots)
                self.assertEqual("".join(result.labels), "X" * frames)

    def test_dead_or_reset_prevents_backdating_across_boundary(self):
        ball, players, teams = self.pass_sequence()
        mask = np.zeros(len(ball), dtype=bool)
        mask[6] = True
        for name in ["dead", "reset"]:
            with self.subTest(name=name):
                result = self.infer(ball, players, teams, accept=True, **{name: mask})
                self.assertEqual(result.labels[3], "X")
                self.assertEqual(result.labels[6], "X")

    def test_shared_ball_player_jump_does_not_create_control(self):
        ball = np.column_stack((np.arange(5) * 0.1, np.zeros(5)))
        players = (ball + [0.2, 0])[:, None, :]
        ball[2] += 10
        players[2] += 10
        result = self.infer(ball, players, ["H"])
        self.assertEqual("".join(result.labels), "XHXXH")

    def test_settings_require_explicit_pass_policy(self):
        for change in [{"lookahead_frames": 0}, {"max_target_miss_m": -1},
                       {"accept_pass_hypotheses": 1}]:
            args = dict(lookahead_frames=12, max_target_miss_m=0.25,
                        min_outgoing_speed_mps=3., accept_pass_hypotheses=False)
            args.update(change)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    InteractionConfig(**args)


if __name__ == "__main__":
    unittest.main()
