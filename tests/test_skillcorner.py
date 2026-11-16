"""SkillCorner loader on tiny invented files."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from possession import skillcorner as sc

MATCH = "999"


def write_match(folder, n=40):
    info = {"home_team": {"id": 1}, "away_team": {"id": 2}, "pitch_length": 100, "pitch_width": 60,
            "players": [{"id": 10, "team_id": 1}, {"id": 20, "team_id": 2}, {"id": 30, "team_id": 1}]}
    Path(folder, MATCH + "_match.json").write_text(json.dumps(info))
    lines = [json.dumps({"frame": 0, "period": None, "ball_data": {}, "possession": {}, "player_data": []})]
    for f in range(1, n + 1):
        group = "home team" if f <= 12 else None if f <= 15 else "away team"
        lines.append(json.dumps({
            "frame": f, "period": 1,
            "ball_data": {"x": float(f), "y": 0., "z": 0., "is_detected": not 10 <= f <= 13},
            "possession": {"player_id": None, "group": group},
            "player_data": [{"player_id": 10, "x": -10., "y": 0., "is_detected": True},
                            {"player_id": 20, "x": 10., "y": 0., "is_detected": False}]}))
    Path(folder, MATCH + "_tracking_extrapolated.jsonl").write_text("\n".join(lines) + "\n")
    Path(folder, MATCH + "_phases_of_play.csv").write_text(
        "frame_start,frame_end,team_in_possession_id\n1,20,1\n31,40,\n")


class SkillCornerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        write_match(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_labels_resampling_and_coordinates(self):
        self.assertEqual(sc.match_ids(self.tmp.name), [MATCH])
        ball, players, teams, period, labels, pitch = sc.load_match(self.tmp.name, MATCH)
        self.assertEqual(pitch, (100., 60.))
        self.assertEqual(list(teams), ["H", "H", "A"])          # ids 10, 30 home; 20 away
        self.assertEqual(len(ball), int(39 * 2.5) + 1)          # 3.9 s at 25 fps
        # source frames 1-12 home, 13-15 empty (carried), 16-20 away, 21-30 no phase (D), 31-40 away
        at = lambda source_frame: labels[int(round((source_frame - 1) * 2.5))]
        self.assertEqual(at(5), "H")
        self.assertEqual(at(14), "H")
        self.assertEqual(at(18), "A")
        self.assertEqual(at(25), "D")
        self.assertEqual(at(35), "A")
        np.testing.assert_allclose(ball[0], [1. + 50., 30.])
        np.testing.assert_allclose(ball[1], [1.4 + 50., 30.])   # interpolated between source samples
        self.assertTrue(np.isnan(ball[25]).all())               # 1.0 s: inside an undetected 0.5 s gap
        np.testing.assert_allclose(players[3, 0], [40., 30.])
        self.assertTrue(np.isnan(players[:, 1]).all())          # id 30 never tracked
        np.testing.assert_allclose(players[3, 2], [60., 30.])   # extrapolated players are kept

    def test_resample_skips_long_gaps(self):
        times = np.array([0., 0.1, 0.5])
        values = np.array([[0., 0.], [1., 0.], [5., 0.]])
        grid = np.arange(0, 0.5 + 1e-9, 0.04)
        out = sc._resample(times, values, grid)
        self.assertTrue(np.isfinite(out[grid <= 0.1]).all())
        self.assertTrue(np.isnan(out[(grid > 0.1) & (grid < 0.5)]).all())


if __name__ == "__main__":
    unittest.main()
