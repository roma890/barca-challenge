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
