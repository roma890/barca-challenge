"""Bidirectional GRU over the per-frame features, run in plain numpy.

The network reads the 33 per-frame features (learned.frame_features + learned.control_features) at
12.5 fps and gives P(H/A/D) for every frame, using the whole half in both directions instead of a
fixed +-2 s window. It is trained with PyTorch (possession/train_sequence.py), but its weights are
saved as JSON and inference here is numpy only, so regenerating the submission needs no PyTorch.

    Linear(33 -> 64) + ReLU -> 2-layer bidirectional GRU (64 per direction) -> Linear(128 -> 3)

Predictions are averaged with the team-swapped input (mirrored back), as for the MLP ensemble, and
blended with an MLP model's log-probabilities before Viterbi decoding (blend_log_probs).
"""

import json
from pathlib import Path

import numpy as np

from . import learned

SUB = 2          # frames per step: 25 fps -> 12.5 fps
HIDDEN = 64


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gru_direction(x, w_ih, w_hh, b_ih, b_hh, reverse):
    """One GRU direction, PyTorch gate order (r, z, n); x is (T, in)."""
    steps = range(len(x) - 1, -1, -1) if reverse else range(len(x))
    gates_x = x @ w_ih.T + b_ih                       # (T, 3H), computed once
    h = np.zeros(w_hh.shape[1])
    out = np.zeros((len(x), w_hh.shape[1]))
    hidden = w_hh.shape[1]
    for t in steps:
        gh = w_hh @ h + b_hh
        gx = gates_x[t]
        r = _sigmoid(gx[:hidden] + gh[:hidden])
        z = _sigmoid(gx[hidden:2 * hidden] + gh[hidden:2 * hidden])
        n = np.tanh(gx[2 * hidden:] + r * gh[2 * hidden:])
        h = (1.0 - z) * n + z * h
        out[t] = h
    return out


def forward(model, x):
    """Logits (T, 3) for a normalised (T, 33) sequence."""
    with np.errstate(**learned._QUIET):
        return _forward(model, x)


def _forward(model, x):
    p = model["params"]
    h = np.maximum(0.0, x @ p["inp.weight"].T + p["inp.bias"])
    for layer in range(model["layers"]):
        fwd = _gru_direction(h, p["gru.weight_ih_l%d" % layer], p["gru.weight_hh_l%d" % layer],
                             p["gru.bias_ih_l%d" % layer], p["gru.bias_hh_l%d" % layer], False)
        bwd = _gru_direction(h, p["gru.weight_ih_l%d_reverse" % layer], p["gru.weight_hh_l%d_reverse" % layer],
                             p["gru.bias_ih_l%d_reverse" % layer], p["gru.bias_hh_l%d_reverse" % layer], True)
        h = np.concatenate([fwd, bwd], axis=1)
    return h @ p["out.weight"].T + p["out.bias"]


def log_probs(model, features):
    """Per-frame log P(H/A/D) for base features (n, 33), averaged with the team-swapped input."""
    swapped, _ = learned.swap_teams(features, np.zeros(len(features), dtype=int), model["names"])
    mirror = [1, 0, 2]
    mean, std = np.asarray(model["mean"]), np.asarray(model["std"])
    outs = []
    for x, order in ((features, [0, 1, 2]), (swapped, mirror)):
        z = ((x[::SUB] - mean) / std).astype(np.float64)
        logits = forward(model, z)
        logits = logits - logits.max(axis=1, keepdims=True)
        outs.append((logits - np.log(np.exp(logits).sum(axis=1, keepdims=True)))[:, order])
    lp = learned._normalise((outs[0] + outs[1]) / 2)
    return np.repeat(lp, SUB, axis=0)[:len(features)]


def blend_log_probs(mlp_log_p, seq_log_p):
    """Equal-weight blend of two models' log-probabilities (renormalised)."""
    return learned._normalise((mlp_log_p + seq_log_p) / 2)


def save(path, *, params, mean, std, names, prior, layers, notes):
    data = {"kind": "bigru", "layers": layers, "hidden": HIDDEN, "sub": SUB, "names": list(names),
            "mean": np.asarray(mean, float).tolist(), "std": np.asarray(std, float).tolist(),
            "prior": np.asarray(prior, float).tolist(), "notes": notes,
            "params": {k: np.asarray(v, float).tolist() for k, v in params.items()}}
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


def load(path):
    data = json.loads(Path(path).read_text())
    if data.get("kind") != "bigru":
        raise ValueError("{} is not a bigru model".format(path))
    data["params"] = {k: np.asarray(v, float) for k, v in data["params"].items()}
    data["prior"] = np.asarray(data["prior"], float)
    return data
