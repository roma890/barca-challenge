"""Synthetic checks for the experimental v7 runner (possession/cv_idsse.py). No real data."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession import cv_idsse as cv
from possession import learned
from possession import viterbi as vt

# "prox" and "near_sign" flip sign on a team swap, d_H/d_A exchange, "row" is untouched.
NAMES = ["prox", "d_H", "d_A", "near_sign", "row"]
FIT = dict(seeds=(0, 1), hidden=8, epochs=2, batch=64)


def fixture(copy_index, n):
    rng = np.random.default_rng(copy_index)
    x = rng.normal(size=(n, len(NAMES))).astype(np.float32)
    x[:, 4] = copy_index * 1000 + np.arange(n)  # row id
    return x, rng.integers(0, 3, n).astype(np.int64)


class LowMemoryRunner(unittest.TestCase):
    def setUp(self):
        # lengths not multiples of the stride, two copies of one match, keys in a fixed order
        self.keys = [("m1", 0), ("m2", 0), ("m1", 1)]
        self.copies = {key: fixture(i + 1, n) for i, (key, n) in enumerate(zip(self.keys, (103, 98, 121)))}
        self.store = {key: cv.presample(*value) for key, value in self.copies.items()}

    def reference_rows(self):
        """Plain reference: full copies, every 5th frame chosen by an explicit loop, concatenated."""
        xs, ys = [], []
        for key in self.keys:
            x, y = self.copies[key]
            index = [i for i in range(len(x)) if i % cv.STRIDE == 0]
            xs.append(x[index])
            ys.append(y[index])
        return np.concatenate(xs), np.concatenate(ys)

    def test_sample_order_and_mirror(self):
        x, y = self.reference_rows()
        buf, labels = cv.training_matrix(self.keys, self.store, NAMES)
        n = len(x)
        self.assertEqual(buf.shape, (2 * n, len(NAMES)))
        np.testing.assert_array_equal(buf[:n], x)
        np.testing.assert_array_equal(labels[:n], y)
        np.testing.assert_array_equal(buf[n:, 4], x[:, 4])          # same rows, same order
        np.testing.assert_array_equal(buf[n:, 0], -x[:, 0])         # sign-flipped column
        np.testing.assert_array_equal(buf[n:, 1], x[:, 2])          # d_H <- d_A
        np.testing.assert_array_equal(labels[n:], np.array([1, 0, 2])[y])  # H<->A, D stays

    def test_matches_reference_training_bit_for_bit(self):
        x, y = self.reference_rows()
        reference = learned.fit_ensemble(x, y, NAMES, stride=1, **FIT)
        lowmem = cv.fit_lowmem(self.keys, self.store, NAMES, **FIT)
        np.testing.assert_array_equal(reference["prior"], lowmem["prior"])
        self.assertEqual(len(reference["members"]), len(lowmem["members"]))
        for a, b in zip(reference["members"], lowmem["members"]):
            self.assertEqual(sorted(a), sorted(b))
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])

    def test_deterministic(self):
        first = cv.fit_lowmem(self.keys, self.store, NAMES, **FIT)
        second = cv.fit_lowmem(self.keys, self.store, NAMES, **FIT)
        for a, b in zip(first["members"], second["members"]):
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])

    def test_serialization_round_trip(self):
        model = cv.fit_lowmem(self.keys, self.store, NAMES, **FIT)
        x = self.copies["m2", 0][0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            learned.save_model(path, model, names=NAMES, config=vt.load_viterbi_config(cv.FEATURE_CONFIG),
                               scale=1.0, switch_cost=0., dead_switch_cost=0., notes={"experimental": True})
            loaded = learned.load_model(path)
        np.testing.assert_array_equal(loaded["prior"], model["prior"])
        for a, b in zip(model["members"], loaded["members"]):
            for name in a:
                np.testing.assert_array_equal(np.asarray(a[name], dtype=np.float64), b[name])
        before, after = learned.log_probs(model, x), learned.log_probs(loaded, x)
        # loaded statistics are float64, in memory float32: tiny differences, same decisions
        np.testing.assert_allclose(before, after, atol=1e-4)
        np.testing.assert_array_equal(before.argmax(axis=1), after.argmax(axis=1))


class Decoder(unittest.TestCase):
    def test_symmetric_costs_match_legacy_viterbi(self):
        scores = np.random.default_rng(0).normal(size=(2000, 3))
        for team, dead in ((1.0, 7.0), (2.5, 12.5), (0.3, 0.3)):
            np.testing.assert_array_equal(cv.viterbi_costs(scores, team, dead, dead), vt._viterbi(scores, team, dead))

    def test_legacy_decode_is_learned_decode(self):
        rng = np.random.default_rng(1)
        log_p = np.log(rng.dirichlet(np.ones(3), size=1500))
        prior = np.array([0.3, 0.3, 0.4])
        expected = np.asarray(vt.STATES)[vt._viterbi(0.15 * (log_p - np.log(prior)), 1.0, 7.0)]
        np.testing.assert_array_equal(cv.decode(log_p, prior, "legacy"), expected)

    def test_dead_stretch_needs_entry_plus_exit(self):
        def kept(dead_in, dead_out, k=40, r=1.0):
            s = np.zeros((1000, 3))
            s[:, 1], s[:, 2] = -3., -1.
            s[480:480 + k, 2] = r
            return (cv.viterbi_costs(s, 5.0, dead_in, dead_out)[480:480 + k] == 2).all()
        self.assertTrue(kept(10., 25.))      # 35 < 40 nats of evidence
        self.assertTrue(kept(25., 10.))
        self.assertFalse(kept(20., 25.))     # 45 > 40
        self.assertFalse(kept(25., 20.))

    def test_grid_is_fixed_and_excludes_legacy_point(self):
        self.assertEqual(len(cv.DECODER_GRID), 48)
        self.assertTrue(all(min(d) > 0 for d in cv.DECODER_GRID))
        legacy = (1.0 / 0.15, 7.0 / 0.15, 7.0 / 0.15)
        self.assertFalse(any(np.allclose(d, legacy) for d in cv.DECODER_GRID))


class Augmentation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.players = rng.uniform(0, 60, size=(3000, 6, 2))
        self.players[:200, 5] = np.nan

    def test_damage_deterministic_and_shape_preserving(self):
        options = dict(segment_mean_s=20., gap_mean_s=2., swaps_per_min=10.)
        a = cv.damage_random(self.players, np.random.default_rng(3), **options)
        b = cv.damage_random(self.players, np.random.default_rng(3), **options)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, self.players.shape)
        self.assertGreater(np.isnan(a).sum(), np.isnan(self.players).sum())

    def test_no_damage_when_rates_are_zero(self):
        out = cv.damage_random(self.players, np.random.default_rng(3), segment_mean_s=1e9, gap_mean_s=1.,
                               swaps_per_min=0.)
        np.testing.assert_array_equal(out, self.players)

    def test_drawn_params_inside_ranges(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            params = cv.draw_params(rng)
            for key, (lo, hi) in cv.AUG_RANGES.items():
                self.assertTrue(lo <= params[key] <= hi, key)
        for key, (lo, hi) in cv.AUG_RANGES.items():
            self.assertTrue(lo <= cv.STRESS[key] <= hi, key)

    def test_seeds_disjoint_between_roles(self):
        train = {cv.copy_seed(cv.ROLE_TRAIN, m, c) for m in range(7) for c in range(cv.TRAIN_COPIES)}
        train |= {s + 5000 for s in train}
        evaluation = {cv.copy_seed(role, m) for role in (cv.ROLE_CLEAN, cv.ROLE_STRESS) for m in range(7)}
        self.assertFalse(train & evaluation)
        self.assertEqual(len(evaluation), 14)


class Scoring(unittest.TestCase):
    def test_short_recall_and_score_keys(self):
        gt = np.array(list("D" * 50 + "H" * 100 + "A" * 10 + "H" * 100 + "D" * 50))
        missed = np.array(list("D" * 50 + "H" * 210 + "D" * 50))
        found = gt.copy()
        lt1, n = cv.short_recall(cv.pm.label_codes(gt), cv.pm.label_codes(missed), 0, 1)
        self.assertEqual((lt1, n), (0.0, 1))
        self.assertEqual(cv.short_recall(cv.pm.label_codes(gt), cv.pm.label_codes(found), 0, 1), (1.0, 1))
        result = cv.score(gt, found)
        self.assertAlmostEqual(result["spell_f1"], 1.0)
        self.assertAlmostEqual(result["dead_f1"], 1.0)
        for key in ("fragmentation", "trans_f1_0.5s", "trans_f1_1s", "short_recall_1_2s"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
