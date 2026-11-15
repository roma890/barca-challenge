"""Learned-evidence model on invented data only."""

import tempfile
import unittest

import numpy as np

from possession import learned
from possession import viterbi as vt
from tests.test_viterbi import FPS, PITCH, TEAMS, config, scene

STATE = {state: i for i, state in enumerate(vt.STATES)}


def toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    names = ["prox", "prox_m6", "prox_m25", "ball_att_H", "d_H", "d_A", "group_speed"]
    features = rng.normal(size=(n, len(names)))
    labels = np.where(features[:, 0] > 0.5, STATE["H"], np.where(features[:, 0] < -0.5, STATE["A"], STATE["D"]))
    return features, labels, names


class LearnedTests(unittest.TestCase):
    def test_swap_twice_is_identity_and_labels_mirror(self):
        features, labels, names = toy(50)
        swapped, mirrored = learned.swap_teams(features, labels, names)
        np.testing.assert_array_equal(swapped[:, names.index("d_H")], features[:, names.index("d_A")])
        np.testing.assert_array_equal(swapped[:, names.index("prox")], -features[:, names.index("prox")])
        np.testing.assert_array_equal(swapped[:, names.index("group_speed")], features[:, names.index("group_speed")])
        back, again = learned.swap_teams(swapped, mirrored, names)
        np.testing.assert_array_equal(back, features)
        np.testing.assert_array_equal(again, labels)
        self.assertTrue((mirrored[labels == STATE["D"]] == STATE["D"]).all())
        self.assertTrue((mirrored[labels == STATE["H"]] == STATE["A"]).all())

    def test_fit_learns_a_separable_rule(self):
        features, labels, names = toy()
        model = learned.fit(features, labels)
        accuracy = np.mean(learned.log_probs(model, features).argmax(axis=1) == labels)
        self.assertGreater(accuracy, 0.9)

    def test_symmetric_model_mirrors_probabilities(self):
        features, labels, names = toy()
        model = learned.fit_symmetric(features, labels, names)
        swapped, _ = learned.swap_teams(features, labels, names)
        original, mirror = learned.log_probs(model, features), learned.log_probs(model, swapped)
        np.testing.assert_allclose(original[:, STATE["H"]], mirror[:, STATE["A"]], atol=1e-6)
        np.testing.assert_allclose(original[:, STATE["D"]], mirror[:, STATE["D"]], atol=1e-6)

    def test_features_are_finite_with_expected_columns(self):
        ball, players = scene(["H"] * 100 + [None] * 50 + ["A"] * 100)
        ball[120:140] = np.nan
        features, names = learned.frame_features(ball, players, TEAMS, fps=FPS, pitch=PITCH,
                                                 halves=[(0, len(ball))], config=config())
        self.assertEqual(features.shape, (len(ball), len(names)))
        self.assertTrue(np.isfinite(features).all())
        for name in ("prox", "fwd_H", "fwd_A", "exit", "any_missing"):
            self.assertIn(name, names)

    def test_attack_direction_follows_the_keeper(self):
        players = np.zeros((10, 4, 2))
        players[:, :, 0] = [2., 50., 60., 100.]
        self.assertTrue((learned.attack_direction(players, TEAMS, [(0, 10)], PITCH) == 1).all())
        players[:, :, 0] = [103., 50., 60., 5.]
        self.assertTrue((learned.attack_direction(players, TEAMS, [(0, 10)], PITCH) == -1).all())

    def test_context_columns_shift_and_swap_with_their_base(self):
        features, labels, names = toy(60)
        wide, wide_names = learned.add_context(features, names, base=("prox", "d_H", "d_A"), offsets=(-2, 3))
        np.testing.assert_array_equal(wide[:-3, wide_names.index("prox@+3")], features[3:, 0])
        np.testing.assert_array_equal(wide[-3:, wide_names.index("prox@+3")], 0)
        np.testing.assert_array_equal(wide[2:, wide_names.index("d_H@-2")], features[:-2, names.index("d_H")])
        swapped, _ = learned.swap_teams(wide, labels, wide_names)
        np.testing.assert_array_equal(swapped[:, wide_names.index("prox@-2")], -wide[:, wide_names.index("prox@-2")])
        np.testing.assert_array_equal(swapped[:, wide_names.index("d_H@+3")], wide[:, wide_names.index("d_A@+3")])

    def test_ensemble_learns_and_is_exactly_symmetric(self):
        features, labels, names = toy(3000)
        model = learned.fit_ensemble(features, labels, names, seeds=(0, 1), stride=2, epochs=10, batch=64)
        log_p = learned.log_probs(model, features)
        self.assertGreater(np.mean(log_p.argmax(axis=1) == labels), 0.9)
        swapped, _ = learned.swap_teams(features, labels, names)
        mirror = learned.log_probs(model, swapped)
        np.testing.assert_allclose(log_p[:, STATE["H"]], mirror[:, STATE["A"]], atol=1e-9)
        np.testing.assert_allclose(log_p[:, STATE["D"]], mirror[:, STATE["D"]], atol=1e-9)

    def test_ensemble_save_load_and_context_round_trip(self):
        features, labels, names = toy(400)
        base = ("prox", "d_H", "d_A")
        wide, wide_names = learned.add_context(features, names, base=base, offsets=(-1, 1))
        model = learned.fit_ensemble(wide, labels, wide_names, seeds=(0,), epochs=2)
        kwargs = dict(scale=0.2, switch_cost=2.0, dead_switch_cost=7.0)
        with tempfile.NamedTemporaryFile(suffix=".json") as handle:
            learned.save_model(handle.name, model, names=names, config=config(), notes={},
                               context=(base, (-1, 1)), **kwargs)
            loaded = learned.load_model(handle.name)
        rebuilt = learned.model_features(loaded, features, names)
        np.testing.assert_array_equal(rebuilt, wide)
        np.testing.assert_array_equal(learned.decode(model, wide, **kwargs), learned.decode(loaded, rebuilt, **kwargs))
        with self.assertRaises(ValueError):
            learned.model_features(loaded, features, names[::-1])

    def test_control_features_mirror_when_teams_swap(self):
        ball, players = scene(["H"] * 60 + ["A"] * 60)
        teams = np.asarray(TEAMS)
        mirrored_teams = np.where(teams == "H", "A", "H")
        a, names = learned.control_features(ball, players, teams, fps=FPS)
        b, _ = learned.control_features(ball, players, mirrored_teams, fps=FPS)
        self.assertTrue(np.isfinite(a).all())
        swapped, _ = learned.swap_teams(a, np.zeros(len(a), dtype=int), names)
        np.testing.assert_allclose(swapped, b, atol=1e-12)
        near = a[:, names.index("near_sign")]
        self.assertTrue((near[5:50] == 1).all() and (near[70:115] == -1).all())

    def test_model_inputs_add_control_and_context(self):
        ball, players = scene(["H"] * 60 + ["A"] * 60)
        base, base_names = learned.frame_features(ball, players, TEAMS, fps=FPS, pitch=PITCH,
                                                  halves=[(0, len(ball))], config=config())
        extra, extra_names = learned.control_features(ball, players, TEAMS, fps=FPS)
        model = {"feature_names": base_names + extra_names, "control": True, "viterbi_config": config(),
                 "context": {"base": list(learned.CONTEXT_BASE_V3), "offsets": [-3, 3]}}
        got = learned.model_inputs(model, ball, players, TEAMS, fps=FPS, pitch=PITCH, halves=[(0, len(ball))])
        want, _ = learned.add_context(np.c_[base, extra], base_names + extra_names, learned.CONTEXT_BASE_V3, (-3, 3))
        np.testing.assert_array_equal(got, want)

    def test_save_and_load_give_the_same_decoding(self):
        features, labels, names = toy(400)
        model = learned.fit(features, labels, iterations=50)
        with tempfile.NamedTemporaryFile(suffix=".json") as handle:
            learned.save_model(handle.name, model, names=names, config=config(), scale=0.2,
                               switch_cost=3.5, dead_switch_cost=7.0, notes={})
            loaded = learned.load_model(handle.name)
        kwargs = dict(scale=0.2, switch_cost=3.5, dead_switch_cost=7.0)
        np.testing.assert_array_equal(learned.decode(model, features, **kwargs), learned.decode(loaded, features, **kwargs))
        self.assertTrue(np.isin(learned.decode(loaded, features, **kwargs), vt.STATES).all())


if __name__ == "__main__":
    unittest.main()
