"""Schema, units and alignment tests using locally generated toy ATD."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from possession.tracking import load_tracking
from tests.tracking_fixtures import dribble_fixture, write_tracking_fixture


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ball, self.players = dribble_fixture()
        self.meta, self.raw = write_tracking_fixture(self.root, self.ball, self.players)

    def load(self, **kwargs):
        args = dict(n_frames=6, pitch=(105.3, 68.), fps=25)
        args.update(kwargs)
        return load_tracking(self.meta, self.raw, **args)

    def test_round_trip_metres_and_one_based_frame_alignment(self):
        tracking = self.load()
        np.testing.assert_allclose(tracking.ball_xy_m, self.ball)
        np.testing.assert_allclose(tracking.player_xy_m, self.players)
        self.assertEqual(tracking.teams.tolist(), ["H", "A"])
        self.assertEqual(tracking.fps, 25.)
        np.testing.assert_allclose(tracking.ball_xy_m[0], [2., 3.])

    def test_preserves_metadata_slot_order(self):
        write_tracking_fixture(self.root, self.ball, self.players, teams=("A", "H"))
        self.assertEqual(self.load().teams.tolist(), ["A", "H"])

    def test_missing_and_partial_coordinates_never_become_zero(self):
        self.ball[2] = np.nan
        self.players[3, 0, 1] = np.nan
        write_tracking_fixture(self.root, self.ball, self.players)
        tracking = self.load()
        self.assertTrue(np.isnan(tracking.ball_xy_m[2]).all())
        self.assertTrue(np.isnan(tracking.player_xy_m[3, 0]).all())
        self.assertTrue(np.isfinite(tracking.ball_xy_m[3]).all())

    def test_missing_speed_does_not_discard_observed_position(self):
        self.raw.write_text(self.raw.read_text().replace(",0;", ",NaN;")
                            .replace(",0\n", ",NaN\n"), encoding="utf-8")
        np.testing.assert_allclose(self.load().ball_xy_m, self.ball)

    def test_outside_pitch_coordinates_are_not_clipped(self):
        self.ball[0] = [-1., 70.]
        write_tracking_fixture(self.root, self.ball, self.players)
        np.testing.assert_allclose(self.load().ball_xy_m[0], [-1., 70.])

    def test_truncation_and_extra_rows_fail(self):
        for count in [5, 7]:
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "records|record count"):
                    self.load(n_frames=count)

    def test_duplicate_zero_based_and_reordered_ids_fail(self):
        original = self.raw.read_text()
        lines = original.splitlines(keepends=True)
        for changed in [original.replace("1:", "0:", 1),
                        original.replace("2:", "1:", 1),
                        "".join([lines[1], lines[0]] + lines[2:])]:
            self.raw.write_text(changed, encoding="utf-8")
            with self.subTest(changed=changed[:2]):
                with self.assertRaisesRegex(ValueError, "consecutive from 1"):
                    self.load()

    def test_malformed_records_fail_without_echoing_payload(self):
        lines = self.raw.read_text().splitlines(keepends=True)
        payloads = ["1:PRIVATE_SENTINEL:0,0,0\n", "1:0,0,0;0,0,0:PRIVATE_SENTINEL,0,0\n",
                    "1:0,0,0;0,0,0:inf,0,0\n", "1:0,0;0,0,0:0,0,0\n",
                    "1:0,0,0:0,0,0\n", "1:0,0,0;0,0,0:0,0,0:extra\n", "\n"]
        for first in payloads:
            self.raw.write_text(first + "".join(lines[1:]), encoding="utf-8")
            with self.subTest(first=first):
                with self.assertRaises(ValueError) as context:
                    self.load()
                self.assertNotIn("PRIVATE_SENTINEL", str(context.exception))

    def test_unknown_team_and_invalid_metadata_fail(self):
        original = self.meta.read_text()
        for metadata in [original.replace('teamId="A"', 'teamId="unknown"'),
                         original.replace('<Team id="A">', '<Team id="H">'),
                         '<root><Team id="H"/></root>', '<not-xml']:
            self.meta.write_text(metadata, encoding="utf-8")
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    self.load()

    def test_invalid_dimensions_rate_and_frame_count(self):
        for change in [{"n_frames": 0}, {"n_frames": True}, {"n_frames": 6.5},
                       {"pitch": [0, 68]}, {"pitch": [np.nan, 68]}, {"pitch": [105]},
                       {"fps": 0}, {"fps": True}, {"fps": np.inf}]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.load(**change)


if __name__ == "__main__":
    unittest.main()
