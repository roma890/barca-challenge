"""IDSSE loader on a tiny invented DFL file."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from possession import idsse

MATCH = "DFL-MAT-TEST01"


def frames(start, rows, ball=False):
    out = []
    for i, row in enumerate(rows):
        extra = ' Z="0.00" BallPossession="{}" BallStatus="{}"'.format(row[2], row[3]) if ball else ""
        out.append('<Frame N="{}" T="t" X="{:.2f}" Y="{:.2f}" D="0" S="0"{} />'.format(start + i, row[0], row[1], extra))
    return "\n".join(out)


def write_match(folder):
    Path(folder, "DFL_02_01_matchinformation_DFL-COM-000001_{}.xml".format(MATCH)).write_text(
        '<PutDataRequest><MatchInformation><General HomeTeamId="HOME" GuestTeamId="AWAY" />'
        '</MatchInformation></PutDataRequest>')
    first_ball = [(0., 0., 1, 1), (1., 0., 1, 1), (1., 0., 2, 0), (1., 0., 2, 0)]
    second_ball = [(5., 5., 2, 1), (6., 5., 2, 1)]
    body = [
        '<FrameSet GameSection="firstHalf" MatchId="M" TeamId="HOME" PersonId="P1">', frames(10, [(-10., 0.)] * 4), "</FrameSet>",
        '<FrameSet GameSection="firstHalf" MatchId="M" TeamId="AWAY" PersonId="P2">', frames(11, [(10., 0.)] * 3), "</FrameSet>",
        '<FrameSet GameSection="firstHalf" MatchId="M" TeamId="referee" PersonId="R">', frames(10, [(0., 30.)] * 4), "</FrameSet>",
        '<FrameSet GameSection="firstHalf" MatchId="M" TeamId="BALL" PersonId="B">', frames(10, first_ball, ball=True), "</FrameSet>",
        '<FrameSet GameSection="secondHalf" MatchId="M" TeamId="HOME" PersonId="P1">', frames(100, [(20., 1.)] * 2), "</FrameSet>",
        '<FrameSet GameSection="secondHalf" MatchId="M" TeamId="BALL" PersonId="B">', frames(100, second_ball, ball=True), "</FrameSet>",
    ]
    Path(folder, "DFL_04_03_positions_raw_observed_DFL-COM-000001_{}.xml".format(MATCH)).write_text(
        '<?xml version="1.0"?>\n<PutDataRequest><Positions>\n' + "\n".join(body) + "\n</Positions></PutDataRequest>\n")


class IdsseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        write_match(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_joins_halves_and_maps_teams_and_labels(self):
        self.assertEqual(idsse.match_ids(self.tmp.name), [MATCH])
        ball, players, teams, period, truth = idsse.load_match(self.tmp.name, MATCH)
        self.assertEqual(list(teams), ["H", "A"])
        self.assertEqual(list(period), [1, 1, 1, 1, 2, 2])
        self.assertEqual("".join(truth), "HHDDAA")
        np.testing.assert_allclose(ball[0], [52.5, 34.])
        np.testing.assert_allclose(players[0, 0], [42.5, 34.])
        self.assertTrue(np.isnan(players[0, 1]).all())       # P2 starts one frame later
        np.testing.assert_allclose(players[1, 1], [62.5, 34.])
        self.assertTrue(np.isnan(players[4:, 1]).all())      # P2 absent in the second half
        self.assertTrue(idsse.parsed_path(self.tmp.name, MATCH).exists())

    def test_observed_ball_hides_frozen_frames_only_from_positions(self):
        ball = np.array([[0., 0.], [1., 0.], [1., 0.], [1., 0.], [2., 0.]])
        seen = idsse.observed_ball(ball, seed=0)
        self.assertTrue(np.isnan(seen[2:4]).all())
        self.assertTrue(np.isfinite(seen[[0, 1, 4]]).all())
        np.testing.assert_array_equal(idsse.observed_ball(ball, seed=0), seen)


if __name__ == "__main__":
    unittest.main()
