"""End-to-end synthetic tracking -> inference -> official CSV validation."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation import possession_metrics as pm
from possession.pipeline import load_config, predict, write_validated_submission
from possession.run import main
from possession.dead_episodes import EpisodeResult
from possession.tracking import Tracking, load_tracking
from tests.test_stoppages import restart_fixture, set_piece, stoppage_config
from tests.test_play_state import play_state_config
from tests.test_dead_episodes import episode_config, motion_config
from tests.test_resumption import resumption_config
from tests.tracking_fixtures import dribble_fixture, write_tagging_fixture, write_tracking_fixture


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "possession" / "configs" / "prototype.json"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ball, self.players = dribble_fixture()
        self.meta, self.raw = write_tracking_fixture(self.root, self.ball, self.players)
        self.config = load_config(EXAMPLE_CONFIG)

    def tracking(self):
        return load_tracking(self.meta, self.raw, n_frames=6, pitch=(105.3, 68.), fps=25)

    def test_pipeline_produces_expected_labels_and_aggregate_diagnostics(self):
        prediction = predict(self.tracking(), self.config)
        self.assertEqual("".join(prediction.state.labels), "XHHHHH")
        self.assertEqual(prediction.summary["label_counts"], {"H": 5, "A": 0, "D": 0, "X": 1})
        self.assertEqual(prediction.summary["automatic_dead_ball_detection"], "temporal_play_state")
        self.assertEqual(prediction.summary["dead_evidence"], "temporal_context_and_motion")
        self.assertEqual(prediction.summary["diagnostic_possessions"], 1)
        self.assertAlmostEqual(prediction.summary["diagnostic_mean_possession_seconds"], 0.2)
        json.dumps(prediction.summary, allow_nan=False)
        self.assertNotIn("ball_xy_m", prediction.summary)
        self.assertNotIn("labels", prediction.summary)

    def test_missing_ball_is_carried_without_automatic_dead_label(self):
        self.ball[2] = np.nan
        write_tracking_fixture(self.root, self.ball, self.players)
        prediction = predict(self.tracking(), self.config)
        self.assertEqual("".join(prediction.state.labels), "XHHHHH")
        self.assertEqual(prediction.summary["carried_frames"], 2)
        self.assertEqual(prediction.summary["label_counts"]["D"], 0)

    def test_long_missing_ball_gap_keeps_possession_but_exposes_uncertainty(self):
        ball, players = dribble_fixture(frames=50)
        ball[2:45] = np.nan
        meta, raw = write_tracking_fixture(self.root, ball, players)
        tracking = load_tracking(meta, raw, n_frames=50, pitch=(105.3, 68.), fps=25)
        prediction = predict(tracking, self.config)
        self.assertIsNone(self.config.max_carry_frames)
        self.assertEqual("".join(prediction.state.labels), "X" + "H" * 49)
        self.assertEqual(prediction.summary["carried_without_ball_frames"], 43)
        self.assertEqual(prediction.summary["carry_policy"], "until_new_evidence")
        self.assertGreater(prediction.summary["max_control_evidence_age_seconds"], 1.)
        self.assertEqual(prediction.summary["gap_conflict_frames"], 0)

    def test_finite_timeout_remains_available_for_old_configurations(self):
        ball, players = dribble_fixture(frames=12)
        ball[2:10] = np.nan
        meta, raw = write_tracking_fixture(self.root, ball, players)
        tracking = load_tracking(meta, raw, n_frames=12, pitch=(105.3, 68.), fps=25)
        prediction = predict(tracking, replace(self.config, max_carry_frames=2))
        self.assertEqual("".join(prediction.state.labels), "XHHHXXXXXXXH")
        self.assertEqual(prediction.summary["carry_policy"], "fixed_timeout")

    def test_synthetic_dead_mask_and_reset_reach_both_inference_and_state(self):
        dead = np.array([False, False, True, False, False, False])
        reset = np.array([False, False, False, True, False, False])
        prediction = predict(self.tracking(), self.config, dead=dead, reset=reset)
        self.assertEqual("".join(prediction.state.labels), "XHDXHH")
        self.assertEqual(prediction.summary["dead_evidence"], "supplied_by_caller")

    def test_official_writer_validator_and_repeatability(self):
        prediction = predict(self.tracking(), self.config)
        first, second = self.root / "first.csv", self.root / "second.csv"
        for output in [first, second]:
            write_validated_submission(output, prediction.state.labels, n_frames=6)
            table = pm.read_label_csv(output)
            self.assertIsNone(pm.validate_submission(table, 6))
            self.assertEqual(table["frame"].tolist(), list(range(6)))
        self.assertEqual(first.read_bytes(), second.read_bytes())
        result = subprocess.run([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"),
                                 str(first), "--rows", "6"], cwd=ROOT, capture_output=True,
                                text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK", result.stdout)

    def test_invalid_output_and_existing_file_are_not_published_or_overwritten(self):
        output = self.root / "submission.csv"
        for labels in [list("HH"), list("HHHHHQ"), ["Home"] * 6]:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    write_validated_submission(output, labels, n_frames=6)
                self.assertFalse(output.exists())
        output.write_text("keep existing file", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            write_validated_submission(output, list("HHHHHH"), n_frames=6)
        self.assertEqual(output.read_text(), "keep existing file")

    def test_official_validation_failure_leaves_no_output_or_temporary_files(self):
        output = self.root / "submission.csv"
        with patch("possession.pipeline.pm.validate_submission", return_value="synthetic rejection"):
            with self.assertRaisesRegex(ValueError, "official submission validation failed"):
                write_validated_submission(output, list("HHHHHH"), n_frames=6)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".possession-*")), [])

    def test_configuration_round_trip_and_unknown_key_rejection(self):
        path = self.root / "config.json"
        values = asdict(self.config)
        path.write_text(json.dumps(values), encoding="utf-8")
        self.assertEqual(load_config(path), self.config)
        values["control"]["misspelled_setting"] = 1
        path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_config(path)

        values.pop("control")
        path.write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_config(path)

    def test_wider_candidate_config_preserves_all_existing_model_weights(self):
        wide = load_config(ROOT / "possession/configs/wide_candidates.json")
        self.assertEqual(wide.control, self.config.control)
        self.assertEqual(wide.contacts, self.config.contacts)
        self.assertEqual(wide.stoppages, self.config.stoppages)
        self.assertEqual(wide.play_state, self.config.play_state)
        self.assertEqual(wide.max_carry_frames, self.config.max_carry_frames)
        self.assertTrue(wide.episodes.allow_weak_restarts)
        self.assertEqual(replace(wide.episodes, allow_weak_restarts=False), self.config.episodes)
    def test_cli_runs_only_on_synthetic_bundle_and_saves_summary(self):
        output, report = self.root / "cli.csv", self.root / "aggregate.json"
        result = subprocess.run(
            [sys.executable, "-m", "possession.run", "--data-dir", str(self.root),
             "--config", str(EXAMPLE_CONFIG), "--frames", "6", "--output", str(output),
             "--report", str(report)], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIsNone(pm.validate_submission(pm.read_label_csv(output), 6))
        summary = json.loads(report.read_text())
        printed = json.loads(result.stdout.split("\n", 2)[2])
        self.assertEqual(summary, printed)
        self.assertEqual(summary["frames"], 6)
        self.assertEqual(summary["label_counts"]["D"], 0)
        self.assertIn("no Smart Tagging supplied", result.stdout)

    def test_cli_rejects_wrong_length_without_creating_submission(self):
        output = self.root / "wrong.csv"
        # Expected failure is deliberately captured; never fall back to ./data.
        with patch("sys.stderr"):
            result = main(["--data-dir", str(self.root), "--config", str(EXAMPLE_CONFIG),
                           "--frames", "7", "--output", str(output)])
        self.assertEqual(result, 1)
        self.assertFalse(output.exists())

    def restart_tracking(self):
        # Live H dribble -> stationary preparation -> H restart -> H reception.
        ball, players = restart_fixture()
        prefix = np.array([[19.7, 20.], [19.8, 20.], [19.9, 20.]])
        prefix_players = np.stack((prefix + [0.2, 0], prefix + [10, 0]), axis=1)
        ball = np.vstack((prefix, ball))
        players = np.concatenate((prefix_players, players))
        ball[12:, 0] = [21.24, 21.28]
        receiver = np.tile([21.4, 20.], (len(ball), 1))
        receiver[12:] = ball[12:] + [0.2, 0]
        players = np.concatenate((players, receiver[:, None, :]), axis=1)
        meta, raw = write_tracking_fixture(self.root, ball, players, teams=("H", "A", "H"))
        return load_tracking(meta, raw, n_frames=len(ball), pitch=(105.3, 68.), fps=25)

    def test_automatic_stoppage_splits_same_team_possessions_and_seeds_restart(self):
        tracking = self.restart_tracking()
        config = replace(self.config, stoppages=stoppage_config())
        result = predict(tracking, config, events=[set_piece(end=len(tracking.ball_xy_m))])
        self.assertEqual("".join(result.state.labels), "XHHDDDDDDHHHHH")
        self.assertEqual(result.summary["diagnostic_possessions"], 2)
        self.assertEqual(result.summary["restart_count"], 1)
        self.assertEqual(result.evidence.source[9], "restart_geometry")
        self.assertEqual(result.stoppages.restarts[0].confirmation_frame, 11)

    def test_caller_dead_evidence_overrides_automatic_release(self):
        tracking = self.restart_tracking()
        dead = np.arange(len(tracking.ball_xy_m)) == 9
        result = predict(tracking, replace(self.config, stoppages=stoppage_config()),
                         events=[set_piece()], dead=dead)
        self.assertEqual(result.state.labels[9], "D")
        self.assertEqual(result.state.labels[10], "X")
        self.assertEqual(result.summary["dead_evidence"], "caller_and_detector")

    def test_single_restart_seed_carries_through_five_seconds_without_ball(self):
        ball, players = restart_fixture()
        observed_frames = len(ball)
        ball = np.concatenate((ball, np.full((125, 2), np.nan)))
        players = np.concatenate((players, np.repeat(players[-1:], 125, axis=0)))
        tracking = Tracking(ball, players, np.array(["H", "A"]), 25., (105.3, 68.))
        # Tag ends after confirmation. Missing observations after a confirmed
        # release provide no separate dead-ball evidence in this invented case.
        events = [set_piece(end=9)]
        for play_state, guard in ((None, None), (play_state_config(), None),
                                  (play_state_config(), resumption_config())):
            with self.subTest(play_state_enabled=play_state is not None, resumption=guard is not None):
                config = replace(self.config, stoppages=stoppage_config(),
                                 play_state=play_state, episodes=None, resumption=guard)
                result = predict(tracking, config, events=events)
                restart, = result.stoppages.restarts
                release = restart.release_frame
                self.assertEqual(result.stoppages.restart_team[release], "H")
                self.assertEqual(result.evidence.source[release], "restart_geometry")
                self.assertTrue((result.state.labels[release:] == "H").all())
                self.assertTrue((result.evidence.labels[observed_frames:] == "X").all())
                self.assertTrue(result.state.carried[observed_frames:].all())

    def test_supported_exit_config_preserves_all_previous_settings(self):
        candidate = load_config(ROOT / "possession" / "configs" / "supported_exits.json")
        self.assertIsNotNone(candidate.resumption)
        self.assertEqual(replace(candidate, resumption=None), self.config)
        for change in (dict(play_state=None, episodes=None), dict(stoppages=None, episodes=None)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(candidate, **change)

    def test_gap_exit_config_only_adds_gap_and_guard_policies(self):
        previous = load_config(ROOT / "possession" / "configs" / "supported_exits.json")
        candidate = load_config(ROOT / "possession" / "configs" / "supported_exits_gaps.json")
        self.assertTrue(candidate.resumption.bridge_observation_gaps)
        self.assertTrue(candidate.resumption.use_live_sequence_guard)
        old_policy = replace(candidate.resumption, bridge_observation_gaps=False,
                             use_live_sequence_guard=False)
        self.assertEqual(old_policy, previous.resumption)
        self.assertEqual(replace(candidate, resumption=old_policy), previous)
        with self.assertRaises(ValueError):
            replace(candidate, episodes=None)

    def test_episode_transfer_guard_reaches_resumption_and_final_output(self):
        n = 100
        ball = np.tile([30., 34.], (n, 1))
        players = np.tile([[30.2, 34.], [50., 34.]], (n, 1, 1))
        tracking = Tracking(ball, players, np.array(["H", "A"]), 25., (105.3, 68.))
        guard = (np.arange(n) >= 20) & (np.arange(n) < 30)
        episodes = EpisodeResult(np.zeros(n), np.zeros(n, dtype=bool), guard, (), {})
        config = replace(self.config, play_state=play_state_config(),
                         resumption=replace(resumption_config(), use_live_sequence_guard=True))
        with patch("possession.pipeline.infer_dead_episodes", return_value=episodes):
            result = predict(tracking, config, events=[set_piece(end=n)])
        self.assertTrue(result.play_state.dead[guard].all())
        self.assertFalse((result.state.labels[guard] == "D").any())
        self.assertTrue((result.resumption.source[guard] == "live_transfer_sequence").all())
        with patch("possession.pipeline.infer_dead_episodes", return_value=episodes):
            explicit = predict(tracking, config, events=[set_piece(end=n)], dead=guard)
        self.assertTrue((explicit.state.labels[guard] == "D").all())

    def test_unsupported_exit_becomes_unknown_after_bounded_hold_without_owner(self):
        n = 200
        ball = np.full((n, 2), np.nan)
        players = np.tile([[29., 34.], [31., 34.]], (n, 1, 1))
        tracking = Tracking(ball, players, np.array(["H", "A"]), 25., (105.3, 68.))
        config = replace(self.config, play_state=play_state_config(), episodes=None,
                         resumption=resumption_config())
        result = predict(tracking, config, events=[set_piece(end=60)])
        self.assertTrue((result.state.labels[:70] == "D").all())
        self.assertTrue((result.state.labels[70:] == "X").all())
        self.assertTrue((result.evidence.source[70:] == "unresolved_play_state").all())
        self.assertEqual(result.summary["resumption_diagnostics"]["held_dead_frames"], 10)
        self.assertEqual(result.summary["resumption_diagnostics"]["unresolved_frames"], 130)
        output = self.root / "guarded.csv"
        write_validated_submission(output, result.state.labels, n_frames=n)
        self.assertIsNone(pm.validate_submission(pm.read_label_csv(output), n))

    def test_cli_loads_synthetic_automatic_tags_only_when_requested(self):
        tracking = self.restart_tracking()
        write_tagging_fixture(self.root, start_frame=0, end_frame=len(tracking.ball_xy_m))
        config = self.root / "restart_config.json"
        config.write_text(json.dumps(asdict(replace(self.config, stoppages=stoppage_config()))))
        output, report = self.root / "restart.csv", self.root / "restart_report.json"
        result = subprocess.run(
            [sys.executable, "-m", "possession.run", "--data-dir", str(self.root),
             "--config", str(config), "--frames", str(len(tracking.ball_xy_m)),
             "--output", str(output), "--report", str(report), "--smart-tagging"],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        table = pm.read_label_csv(output)
        self.assertIsNone(pm.validate_submission(table, len(tracking.ball_xy_m)))
        self.assertEqual("".join(table["label"]), "XHHDDDDDDHHHHH")
        self.assertEqual(json.loads(report.read_text())["restart_count"], 1)

    def test_cli_without_flag_never_opens_tagging(self):
        with patch("possession.run.load_tagging") as loader, patch("sys.stdout"):
            result = main(["--data-dir", str(self.root), "--config", str(EXAMPLE_CONFIG),
                           "--frames", "6", "--output", str(self.root / "no_tags.csv")])
        self.assertEqual(result, 0)
        loader.assert_not_called()

    def test_malformed_tagging_fails_without_submission(self):
        path = write_tagging_fixture(self.root, start_frame=0, end_frame=6)
        path.write_text("<not-xml")
        output = self.root / "bad_tags.csv"
        with patch("sys.stderr"):
            result = main(["--data-dir", str(self.root), "--config", str(EXAMPLE_CONFIG),
                           "--frames", "6", "--output", str(output), "--smart-tagging"])
        self.assertEqual(result, 1)
        self.assertFalse(output.exists())

    def test_legacy_configuration_can_disable_stoppage_inference(self):
        values = asdict(self.config)
        values.pop("stoppages")
        values.pop("play_state")
        values.pop("episodes")
        config_file = self.root / "legacy.json"
        config_file.write_text(json.dumps(values))
        config = load_config(config_file)
        self.assertIsNone(config.stoppages)
        result = predict(self.tracking(), config)
        self.assertEqual(result.summary["automatic_dead_ball_detection"], "disabled")

    def test_soft_context_can_produce_dead_without_a_clean_restart_chain(self):
        ball = np.full((100, 2), np.nan)
        players = np.tile([[29., 34.], [31., 34.]], (100, 1, 1))
        meta, raw = write_tracking_fixture(self.root, ball, players)
        tracking = load_tracking(meta, raw, n_frames=100, pitch=(105.3, 68.), fps=25)
        config = replace(self.config, play_state=play_state_config())
        result = predict(tracking, config, events=[set_piece(end=100)])
        self.assertEqual(result.summary["restart_count"], 0)
        self.assertGreater(result.summary["label_counts"]["D"], 0)
        self.assertGreater(result.summary["play_state_diagnostics"]["dead_without_observed_ball_frames"], 0)
        output = self.root / "soft_context.csv"
        write_validated_submission(output, result.state.labels, n_frames=100)
        self.assertIsNone(pm.validate_submission(pm.read_label_csv(output), 100))

    def test_backdated_episode_reaches_csv_despite_retrieval_motion_or_broad_live_tag(self):
        ball = np.full((80, 2), np.nan)
        ball[:10] = [20., 1.]
        ball[10:12] = [[20., -0.2], [20., -0.4]]
        ball[20:26] = np.column_stack((np.full(6, 20.), np.linspace(-2., 0., 6)))
        ball[35:40] = [20., 0.]
        ball[40:45] = np.column_stack((20. + np.arange(1, 6) * 0.4, np.zeros(5)))
        players = np.tile([[50., 30.], [60., 30.]], (80, 1, 1))
        players[30:, 0] = [20.2, 0.]
        # Use the official pitch and the same physical scales as the unit probe;
        # a throw-in along y=0 is independent of pitch length.
        meta, raw = write_tracking_fixture(self.root, ball, players)
        tracking = load_tracking(meta, raw, n_frames=80, pitch=(105.3, 68.), fps=10)
        config = replace(self.config, stoppages=stoppage_config(), play_state=motion_config(),
                         episodes=episode_config())
        events = [set_piece(end=80, kind="Throw-in"),
                  {"code": "BUILD UP", "start_frame": 0, "end_frame": 80, "tags": {}}]
        result = predict(tracking, config, events=events)
        self.assertGreater(result.summary["episode_diagnostics"]["backdated_restarts"], 0)
        self.assertTrue((result.state.labels[20:35] == "D").all())
        output = self.root / "backdated.csv"
        write_validated_submission(output, result.state.labels, n_frames=80)
        self.assertIsNone(pm.validate_submission(pm.read_label_csv(output), 80))


if __name__ == "__main__":
    unittest.main()
