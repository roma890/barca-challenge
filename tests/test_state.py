"""Synthetic scenarios only; no match assets or ground truth are read."""

import unittest

import numpy as np

from possession.state import decode_state


class StateTests(unittest.TestCase):
    def decode(self, control, *, limit=2, dead=None, reset=None):
        return decode_state(
            list(control),
            np.zeros(len(control), dtype=bool) if dead is None else dead,
            max_carry_frames=limit,
            reset=reset,
        )

    def test_occlusion_during_same_team_pass(self):
        result = self.decode("HXXH")
        self.assertEqual("".join(result.labels), "HHHH")
        self.assertEqual(result.carried.tolist(), [False, True, True, False])

    def test_long_gap_abstains_at_exact_limit(self):
        self.assertEqual("".join(self.decode("HXXXXA").labels), "HHHXXA")

    def test_no_control_before_first_observation(self):
        self.assertEqual("".join(self.decode("XXA").labels), "XXA")

    def test_stoppage_overrides_control_and_clears_memory(self):
        result = self.decode("HHXXA", dead=[False, True, True, False, False])
        self.assertEqual("".join(result.labels), "HDDXA")
        self.assertFalse(result.carried.any())

    def test_restart_can_be_followed_by_missing_observation(self):
        result = self.decode("HXXAX", dead=[False, True, True, False, False])
        self.assertEqual("".join(result.labels), "HDDAA")

    def test_release_reset_accepts_same_frame_seed_then_carries_without_timeout(self):
        control = "HHXA" + "X" * 125
        dead = np.zeros(len(control), dtype=bool)
        dead[2] = True
        reset = np.zeros(len(control), dtype=bool)
        reset[3] = True
        result = self.decode(control, limit=None, dead=dead, reset=reset)
        self.assertEqual("".join(result.labels), "HHD" + "A" * 126)
        self.assertEqual(result.evidence_age_frames[3], 0)
        self.assertTrue(result.carried[4:].all())
        self.assertEqual(result.evidence_age_frames[-1], 125)

    def test_confirmed_one_frame_possession_is_preserved(self):
        self.assertEqual("".join(self.decode("HHAHH").labels), "HHAHH")

    def test_ambiguous_contact_is_unknown_upstream(self):
        # An upstream detector rejects the contact as insufficient control.
        self.assertEqual("".join(self.decode("HHXHH").labels), "HHHHH")

    def test_no_backdating_turnover_across_gap(self):
        self.assertEqual("".join(self.decode("HXXA").labels), "HHHA")

    def test_reset_prevents_carry_between_sequences(self):
        result = self.decode("HXXA", reset=[False, True, False, False])
        self.assertEqual("".join(result.labels), "HXXA")

    def test_zero_limit(self):
        self.assertEqual("".join(self.decode("HXH", limit=0).labels), "HXH")

    def test_empty_sequence(self):
        result = self.decode("")
        self.assertEqual(len(result.labels), 0)
        self.assertEqual(len(result.carried), 0)

    def test_long_gap_preserves_team_without_expiring_possession(self):
        control = "H" + "X" * 250 + "H"
        result = self.decode(control, limit=None)
        self.assertEqual("".join(result.labels), "H" * len(control))
        self.assertEqual(int(result.carried.sum()), 250)
        self.assertEqual(result.evidence_age_frames[250], 250)
        self.assertEqual(result.evidence_age_frames[-1], 0)
        self.assertFalse(result.gap_conflict.any())
        self.assertTrue(result.uncertain[1:251].all())

    def test_opposing_endpoints_flag_gap_without_inventing_boundary(self):
        result = self.decode("HXXXXA", limit=None)
        self.assertEqual("".join(result.labels), "HHHHHA")
        self.assertEqual(result.gap_conflict.tolist(), [False, True, True, True, True, False])
        self.assertEqual(result.evidence_age_frames.tolist(), [0, 1, 2, 3, 4, 0])

    def test_no_past_control_is_still_unknown_in_continuity_mode(self):
        result = self.decode("XXXHXX", limit=None)
        self.assertEqual("".join(result.labels), "XXXHHH")
        self.assertEqual(result.evidence_age_frames.tolist(), [-1, -1, -1, 0, 1, 2])

    def test_dead_clears_long_gap_memory_and_conflict_attribution(self):
        result = self.decode("HXXXXA", limit=None, dead=[False, False, True, False, False, False])
        self.assertEqual("".join(result.labels), "HHDXXA")
        self.assertEqual(result.evidence_age_frames.tolist(), [0, 1, 0, -1, -1, 0])
        self.assertFalse(result.gap_conflict.any())

    def test_reset_clears_long_gap_memory_and_conflict_attribution(self):
        result = self.decode("HXXXXA", limit=None, reset=[False, False, True, False, False, False])
        self.assertEqual("".join(result.labels), "HHXXXA")
        self.assertEqual(result.evidence_age_frames.tolist(), [0, 1, -1, -1, -1, 0])
        self.assertFalse(result.gap_conflict.any())

    def test_direct_short_turnovers_survive_continuity_mode(self):
        self.assertEqual("".join(self.decode("HHAHH", limit=None).labels), "HHAHH")

    def test_invalid_evidence(self):
        for control, dead, limit, reset in [
            (["D"], [False], 2, None),
            (["H"], [], 2, None),
            (["H"], [0], 2, None),
            (["H"], [False], -1, None),
            (["H"], [False], 1.5, None),
            (["H"], [False], True, None),
            (["H"], [False], 2, [0]),
            ([["H"]], [False], 2, None),
        ]:
            with self.subTest(control=control, dead=dead, limit=limit, reset=reset):
                with self.assertRaises(ValueError):
                    decode_state(control, dead, max_carry_frames=limit, reset=reset)


if __name__ == "__main__":
    unittest.main()
