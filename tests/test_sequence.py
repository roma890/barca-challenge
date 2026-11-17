"""The numpy BiGRU (possession/sequence.py) on invented weights: shapes, symmetry, save/load."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession import sequence


def random_model(names, seed=0):
    rng = np.random.default_rng(seed)
    h, k = sequence.HIDDEN, len(names)
    p = {"inp.weight": rng.normal(0, .1, (h, k)), "inp.bias": rng.normal(0, .1, h),
         "out.weight": rng.normal(0, .1, (3, 2 * h)), "out.bias": rng.normal(0, .1, 3)}
    for layer, width in ((0, h), (1, 2 * h)):
        for sfx in ("", "_reverse"):
            p["gru.weight_ih_l%d%s" % (layer, sfx)] = rng.normal(0, .1, (3 * h, width))
            p["gru.weight_hh_l%d%s" % (layer, sfx)] = rng.normal(0, .1, (3 * h, h))
            p["gru.bias_ih_l%d%s" % (layer, sfx)] = rng.normal(0, .1, 3 * h)
            p["gru.bias_hh_l%d%s" % (layer, sfx)] = rng.normal(0, .1, 3 * h)
    return p


class SequenceTests(unittest.TestCase):
    names = ["d_H", "d_A", "prox", "visible"]

    def model(self):
        return {"kind": "bigru", "layers": 2, "names": self.names, "params": random_model(self.names),
                "mean": np.zeros(len(self.names)), "std": np.ones(len(self.names)), "prior": np.ones(3) / 3}

    def test_log_probs_cover_every_frame_and_normalise(self):
        x = np.random.default_rng(1).normal(size=(101, len(self.names)))
        lp = sequence.log_probs(self.model(), x)
        self.assertEqual(lp.shape, (101, 3))
        np.testing.assert_allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-9)

    def test_backward_direction_sees_the_future(self):
        m = {"params": random_model(self.names), "layers": 2}
        x = np.random.default_rng(2).normal(size=(40, len(self.names)))
        y = x.copy(); y[-1] += 5.0                      # change only the last step
        self.assertFalse(np.allclose(sequence.forward(m, x)[0], sequence.forward(m, y)[0]))

    def test_save_load_round_trip(self):
        m = self.model()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            sequence.save(path, params=m["params"], mean=m["mean"], std=m["std"], names=self.names,
                          prior=m["prior"], layers=2, notes={})
            back = sequence.load(path)
        x = np.random.default_rng(3).normal(size=(30, len(self.names)))
        np.testing.assert_array_equal(sequence.log_probs(m, x), sequence.log_probs(back, x))


if __name__ == "__main__":
    unittest.main()
