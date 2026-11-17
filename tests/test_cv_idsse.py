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


def runs_of(mask):
    """(start, stop) of each run of True."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def synthetic_match(n=6000, seed=0):
    """Players scattered around a moving ball; stoppages have a frozen ball (IDSSE convention)."""
    rng = np.random.default_rng(seed)
    ball = np.cumsum(rng.normal(0, 0.3, size=(n, 2)), axis=0)
    y = np.zeros(n, np.int64)
    y[n // 2:] = 1
    for start in (900, 2500, 4200):
        y[start:start + 400] = 2
        ball[start:start + 400] = ball[start]
    duel = (np.arange(n) // 200) % 3 == 0                 # contests come in phases, as in real duels
    spread = np.where(duel, 1.0, 15.0)[:, None, None]
    players = ball[:, None, :] + spread * rng.normal(size=(n, 6, 2))
    return {"ball": ball, "players": players, "teams": np.array(list("HHHAAA"))}, y


class LiveDropout(unittest.TestCase):
    def setUp(self):
        self.parsed, self.y = synthetic_match()
        self.live = self.y != 2

    def hide(self, eligible, level=0.2, weight=5.0, seed=0, **options):
        p = self.parsed
        return cv.hide_ball(p["ball"], p["players"], p["teams"], eligible, level, weight,
                            np.random.default_rng(seed), **options)

    def test_only_eligible_frames_change(self):
        eligible = self.live & (np.arange(len(self.y)) % 7 != 0)     # holes inside live play too
        ball, newly = self.hide(eligible)
        changed = ~np.isfinite(ball).all(axis=1)
        self.assertTrue(changed.any())
        self.assertFalse((changed & ~eligible).any())
        np.testing.assert_array_equal(ball[~changed], self.parsed["ball"][~changed])
        np.testing.assert_array_equal(newly, changed)                 # input ball fully observed

    def test_amount_and_contiguous_gaps(self):
        _, newly = self.hide(self.live, level=0.2)
        target = int(0.2 * self.live.sum())
        self.assertGreaterEqual(newly.sum(), target)
        self.assertLess(newly.sum(), target + cv.DROPOUT_GAP_FRAMES[1])   # overshoot < one gap
        for seed in range(20):                                         # tiny target -> exactly one gap
            _, newly = self.hide(self.live, level=1.5 / self.live.sum(), seed=seed)
            gaps = runs_of(newly)
            self.assertEqual(len(gaps), 1)
            start, stop = gaps[0]
            self.assertLess(stop - start, cv.DROPOUT_GAP_FRAMES[1])
            if stop < len(self.live) and self.live[stop]:              # not cut by a stoppage/end
                self.assertGreaterEqual(stop - start, cv.DROPOUT_GAP_FRAMES[0])

    def test_contest_preference(self):
        p = self.parsed
        contested = cv.contested_frames(p["ball"], p["players"], p["teams"])
        self.assertTrue(0.05 < contested[self.live].mean() < 0.95)
        share = [np.mean([self.hide(self.live, level=0.05, weight=w, seed=s)[1][contested].mean() for s in range(10)])
                 for w in (1.0, 10.0)]
        self.assertGreater(share[1], 1.2 * share[0])

    def test_zero_level_identity_and_determinism(self):
        ball, newly = self.hide(self.live, level=0.0)
        np.testing.assert_array_equal(ball, self.parsed["ball"])
        self.assertFalse(newly.any())
        np.testing.assert_array_equal(self.hide(self.live, seed=4)[0], self.hide(self.live, seed=4)[0])
        self.assertFalse(np.array_equal(np.isnan(self.hide(self.live, seed=4)[0]),
                                        np.isnan(self.hide(self.live, seed=5)[0])))

    def test_train_copy_labels_untouched_and_b1_copies_identical(self):
        parsed = dict(self.parsed, period=np.ones(len(self.y)))
        y_before = self.y.copy()
        for copy in range(cv.TRAIN_COPIES):
            b1 = cv.train_copy(parsed, self.y, 2, copy, live_dropout=False)
            e5 = cv.train_copy(parsed, self.y, 2, copy, live_dropout=True)
            np.testing.assert_array_equal(self.y, y_before)            # labels and order untouched
            np.testing.assert_array_equal(b1[1], e5[1])                 # players identical
            self.assertEqual(b1[0].shape, e5[0].shape)
            if copy == 0:
                np.testing.assert_array_equal(b1[0], e5[0])
                self.assertNotIn("dropout", e5[2])
                continue
            seen_b1, seen_e5 = np.isfinite(b1[0]).all(axis=1), np.isfinite(e5[0]).all(axis=1)
            extra = seen_b1 & ~seen_e5
            self.assertFalse((extra & ~self.live).any())                # never on dead frames
            self.assertFalse((seen_e5 & ~seen_b1).any())                # nothing revealed
            np.testing.assert_array_equal(b1[0][seen_e5], e5[0][seen_e5])
            params = e5[2]["dropout"]
            self.assertTrue(0 <= params["level"] <= 0.35 and 1 <= params["contest_weight"] <= 10)
            self.assertAlmostEqual(extra.sum() / self.live.sum(), params["newly_hidden_share_of_live"])

    def test_dropout_seeds_disjoint_from_every_other_stream(self):
        train = {cv.copy_seed(cv.ROLE_TRAIN, m, c) for m in range(7) for c in range(cv.TRAIN_COPIES)}
        dropout = {s + cv.DROPOUT_SEED_OFFSET for s in train}
        others = train | {s + 5000 for s in train}
        others |= {cv.copy_seed(r, m) for r in (cv.ROLE_CLEAN, cv.ROLE_STRESS) for m in range(7)}
        others |= {cv.eval_missing_seed(kind, m) for kind in cv.EVAL_MISSING for m in range(7)}
        self.assertFalse(dropout & others)
        self.assertEqual(len(others), 2 * 7 * cv.TRAIN_COPIES + 14 + 7 * len(cv.EVAL_MISSING))

    def test_held_out_matches_never_in_training_keys(self):
        matches = ["m{}".format(i) for i in range(7)]
        for outer in matches:
            for inner in matches:
                excluded = {outer, inner}
                keys = cv.train_keys([m for m in matches if m not in excluded])
                self.assertFalse({m for m, _ in keys} & excluded)
                self.assertEqual(len(keys), cv.TRAIN_COPIES * (7 - len(excluded)))

    def test_development_benchmark_is_the_diagnostic_condition(self):
        spec = cv.EVAL_MISSING["stress_missing"]
        self.assertEqual((spec["level"], spec["contest_weight"], tuple(spec["gap_frames"])), (0.2, 10.0, (5, 75)))
        self.assertEqual([cv.eval_missing_seed("stress_missing", i) for i in range(7)],
                         [7000 + 10 * i + int(0.2 * 100) + 1 for i in range(7)])
        self.assertEqual(cv.CONFIGS["e5"]["offsets"], cv.CONFIGS["b1"]["offsets"])
        self.assertTrue(cv.CONFIGS["e5"]["live_dropout"] and not cv.CONFIGS["b1"].get("live_dropout"))


if __name__ == "__main__":
    unittest.main()
