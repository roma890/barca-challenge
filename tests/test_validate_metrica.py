"""Event-to-label rules for the Metrica validation set, on invented events only."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

import json

from possession.validate_metrica import (EPTS_PARTS, GAMES, PARTS, epts_path, event_labels, file_path,
                                         load_game, make_degraded, make_degraded_epts, read_game_events)


def write_fake_game(folder, game, frames=200):
    header = [["", "", "", "Home", "", "Home", "", "", ""], ["", "", "", "1", "", "2", "", "", ""],
              ["Period", "Frame", "Time [s]", "Player1", "", "Player2", "", "Ball", ""]]
    rows = [[1, f + 1, (f + 1) / 25, 0.1, 0.2, 0.3, 0.4,
             "NaN" if f % 10 == 0 else 0.5, "NaN" if f % 10 == 0 else 0.5] for f in range(frames)]
    for part in PARTS[:2]:
        with open(file_path(folder, game, part), "w", newline="") as handle:
            csv.writer(handle).writerows(header + rows)
    file_path(folder, game, PARTS[2]).write_text("Team,Type\nHome,PASS\n")


def event(kind, start, end=0, team="Home", sub="", period="1"):
    return {"Type": kind, "Subtype": sub, "Team": team, "Period": period, "start": start, "end": end}


class EventLabelTests(unittest.TestCase):
    def labels(self, events, n=100):
        return "".join(event_labels(events, n, np.ones(n, dtype=int)))

    def test_owner_is_team_of_last_on_ball_event(self):
        out = self.labels([event("SET PIECE", 1, sub="KICK OFF"), event("PASS", 11),
                           event("RECOVERY", 31, team="Away"), event("PASS", 60, 61, team="Away")])
        self.assertEqual(out[:30], "H" * 30)
        self.assertEqual(out[30:61], "A" * 31)

    def test_challenge_does_not_change_owner(self):
        out = self.labels([event("PASS", 1), event("CHALLENGE", 20, team="Away", sub="GROUND-WON"),
                           event("PASS", 50, 51)])
        self.assertEqual(out[:50], "H" * 50)

    def test_ball_out_is_dead_until_restart(self):
        out = self.labels([event("PASS", 1), event("BALL OUT", 20, 25),
                           event("SET PIECE", 41, team="Away", sub="THROW IN"), event("PASS", 60, 61, team="Away")])
        self.assertEqual(out[24:40], "D" * 16)
        self.assertEqual(out[40], "A")

    def test_foul_and_goal_kick_restart(self):
        out = self.labels([event("PASS", 1), event("FAULT RECEIVED", 30, team="Away"),
                           event("PASS", 51, team="Away", sub="GOAL KICK"), event("PASS", 70, 71, team="Away")])
        self.assertEqual(out[29:50], "D" * 21)
        self.assertEqual(out[50], "A")

    def test_frames_outside_the_period_events_are_ungraded(self):
        out = self.labels([event("PASS", 11), event("PASS", 50, 60)])
        self.assertEqual(out[:10], "X" * 10)
        self.assertEqual(out[60:], "X" * 40)


class DegradedCopyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.source, self.target = Path(self.tmp.name, "src"), Path(self.tmp.name, "dst")
        self.source.mkdir()
        for game in GAMES:
            write_fake_game(self.source, game)
        self.before = {p: p.read_bytes() for p in self.source.iterdir()}

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, folder, game, part):
        with open(file_path(folder, game, part), newline="") as handle:
            return list(csv.reader(handle))[3:]

    def test_copy_changes_only_visible_ball_and_leaves_originals(self):
        make_degraded(self.source, self.target)
        self.assertEqual({p: p.read_bytes() for p in self.source.iterdir()}, self.before)
        for game in GAMES:
            original, copy = self.read(self.source, game, PARTS[0]), self.read(self.target, game, PARTS[0])
            away = self.read(self.target, game, PARTS[1])
            for old, new, other in zip(original, copy, away):
                self.assertEqual(old[:7], new[:7])
                self.assertEqual(new[7:9], other[7:9])
                if old[7] == "NaN":
                    self.assertEqual(new[7:9], ["NaN", "NaN"])
            self.assertTrue(any(old[7] != new[7] for old, new in zip(original, copy) if old[7] != "NaN"))
            self.assertEqual(file_path(self.target, game, PARTS[2]).read_text(),
                             file_path(self.source, game, PARTS[2]).read_text())

    def test_refuses_same_folder_and_existing_copies(self):
        with self.assertRaises(ValueError):
            make_degraded(self.source, self.source)
        make_degraded(self.source, self.target)
        with self.assertRaises(FileExistsError):
            make_degraded(self.source, self.target)


def write_fake_epts(folder, game=3, frames=40):
    """Three players (P1, P2 home; P3 away); P2 is replaced by P4 (home) from frame 21."""
    def spec(start, end, ids):
        regs = "".join('<SplitRegister separator=","><PlayerChannelRef playerChannelId="{0}_x"/>'
                       '<PlayerChannelRef playerChannelId="{0}_y"/></SplitRegister>'.format(i) for i in ids)
        return ('<DataFormatSpecification startFrame="{}" endFrame="{}" separator=":">'
                '<StringRegister name="frameCount"/><SplitRegister separator=";">{}</SplitRegister>'
                '<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
                '</SplitRegister></DataFormatSpecification>').format(start, end, regs)
    players = "".join('<Player id="{}" teamId="{}"/>'.format(i, t) for i, t in
                      (("P1", "TA"), ("P2", "TA"), ("P3", "TB"), ("P4", "TA")))
    channels = "".join('<PlayerChannel id="{0}_{1}" playerId="{0}" channelId="{1}"/>'.format(i, a)
                       for i in ("P1", "P2", "P3", "P4") for a in "xy")
    meta = ('<main><Metadata><GlobalConfig><ProviderGlobalParameters>'
            '<ProviderParameter><Name>second_half_start</Name><Value>31</Value></ProviderParameter>'
            '</ProviderGlobalParameters></GlobalConfig><Sessions><Session><MatchParameters>'
            '<Score idLocalTeam="TA" idVisitingTeam="TB"/></MatchParameters></Session></Sessions>'
            '<Teams/><Players>{}</Players><PlayerChannels>{}</PlayerChannels></Metadata>'
            '<DataFormatSpecifications>{}{}</DataFormatSpecifications></main>').format(
                players, channels, spec(1, 20, ["P1", "P2", "P3"]), spec(21, frames, ["P1", "P4", "P3"]))
    epts_path(folder, game, EPTS_PARTS[1]).write_text(meta)
    lines = ["{}:0.1,0.2;0.3,0.4;0.5,0.6:{}".format(f, "NaN,NaN" if f % 10 == 0 else "0.50000,0.50000")
             for f in range(1, frames + 1)]
    epts_path(folder, game, EPTS_PARTS[0]).write_text("\n".join(lines) + "\n")
    events = {"data": [
        {"team": {"id": "TA"}, "type": {"name": "SET PIECE"}, "subtypes": {"name": "KICK OFF"},
         "start": {"frame": 1}, "end": {"frame": 1}, "period": 1},
        {"team": {"id": "TB"}, "type": {"name": "SHOT"}, "subtypes": [{"name": "OFF TARGET"}, {"name": "OUT"}],
         "start": {"frame": 15}, "end": {"frame": None}, "period": 1}]}
    epts_path(folder, game, EPTS_PARTS[2]).write_text(json.dumps(events))


class EptsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        write_fake_epts(self.folder)

    def tearDown(self):
        self.tmp.cleanup()

    def test_players_follow_the_specs_and_halves_split(self):
        ball, players, teams, period = load_game(self.folder, 3)
        self.assertEqual(list(teams), ["H", "H", "H", "A"])  # P1, P2, P4 home, then P3
        np.testing.assert_allclose(players[0, 1], [0.3 * 105, 0.4 * 68])
        self.assertTrue(np.isnan(players[0, 2]).all())
        np.testing.assert_allclose(players[25, 2], [0.3 * 105, 0.4 * 68])
        self.assertTrue(np.isnan(players[25, 1]).all())
        np.testing.assert_allclose(players[:, 3, 0], 0.5 * 105)
        self.assertTrue(np.isnan(ball[9]).all())
        self.assertEqual((period == 1).sum(), 30)

    def test_events_convert_to_the_csv_schema(self):
        events = read_game_events(self.folder, 3)
        self.assertEqual([e["Team"] for e in events], ["Home", "Away"])
        self.assertEqual(events[1]["Subtype"], "OFF TARGET-OUT")
        self.assertEqual(events[1]["end"], 0)

    def test_degraded_epts_copy_moves_only_the_ball(self):
        target = self.folder / "dst"
        before = epts_path(self.folder, 3, EPTS_PARTS[0]).read_bytes()
        make_degraded_epts(self.folder, target)
        self.assertEqual(epts_path(self.folder, 3, EPTS_PARTS[0]).read_bytes(), before)
        ball, players, _, _ = load_game(self.folder, 3)
        ball2, players2, _, _ = load_game(target, 3)
        np.testing.assert_array_equal(players, players2)
        np.testing.assert_array_equal(np.isnan(ball), np.isnan(ball2))
        self.assertGreater(np.nanmax(np.abs(ball2 - ball)), 0)
        with self.assertRaises(FileExistsError):
            make_degraded_epts(self.folder, target)


if __name__ == "__main__":
    unittest.main()
