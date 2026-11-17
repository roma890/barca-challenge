"""Train the bidirectional GRU (possession/sequence.py) on IDSSE and save it as JSON.

    python -m possession.train_sequence --recipe v8 --epochs 16 --output possession/models/bigru_v8_e16.json

Uses exactly the training copies of the matching train_learned recipe (same matches, drift, damage
and stoppage-ball seeds, smoothed ball), at 12.5 fps, each also with home and away swapped.
Needs PyTorch (only for training; inference is numpy). Fixed: Linear 33->64 + ReLU, 2-layer BiGRU
(64 per direction, dropout 0.1), Linear 128->3; Adam 2e-3, 16 epochs for the submission (--epochs;
default 8, the first design), random 48 s crops, batch 32,
seed 0, 8 CPU threads. Held-out results for this design are in the technical document; this script
trains the final network on all seven matches.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from . import idsse
from . import sequence
from . import train_learned as tl
from . import viterbi as vt

CROP, EPOCHS, BATCH, RATE, SEED = 600, 8, 32, 2e-3, 0


def _torch():
    import torch                                     # noqa: imported lazily: training only
    return torch


def _net(torch, k):
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Linear(k, sequence.HIDDEN)
            self.gru = torch.nn.GRU(sequence.HIDDEN, sequence.HIDDEN, num_layers=2, batch_first=True,
                                    bidirectional=True, dropout=0.1)
            self.out = torch.nn.Linear(2 * sequence.HIDDEN, 3)

        def forward(self, x):
            h, _ = self.gru(torch.relu(self.inp(x)))
            return self.out(h)
    return Net()


def _adam_step(torch, params, state, t):
    """Plain Adam (torch.optim defaults), written out so training needs no optional torch extras."""
    with torch.no_grad():
        for q, m, v in zip(params, state["m"], state["v"]):
            if q.grad is None:
                continue
            m.mul_(0.9).add_(q.grad, alpha=0.1)
            v.mul_(0.999).addcmul_(q.grad, q.grad, value=0.001)
            q.sub_(RATE * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)).sqrt() + 1e-8))


def fit(sequences, seed=SEED, epochs=EPOCHS):
    torch = _torch()
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    rng = np.random.default_rng(seed)
    xs = [x for x, _ in sequences]
    ys = [y for _, y in sequences]
    stacked = np.concatenate(xs)
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6
    xs = [((x - mean) / std).astype(np.float32) for x in xs]
    net = _net(torch, xs[0].shape[1])
    params = list(net.parameters())
    state = {"m": [torch.zeros_like(q) for q in params], "v": [torch.zeros_like(q) for q in params]}
    lengths = np.array([len(x) for x in xs])
    batches = int(lengths.sum() / CROP) // BATCH
    t = 0
    for epoch in range(epochs):
        net.train()
        for _ in range(batches):
            pick = rng.choice(len(xs), BATCH, p=lengths / lengths.sum())
            starts = [rng.integers(0, len(xs[j]) - CROP) for j in pick]
            xb = torch.from_numpy(np.stack([xs[j][s:s + CROP] for j, s in zip(pick, starts)]))
            yb = torch.from_numpy(np.stack([ys[j][s:s + CROP] for j, s in zip(pick, starts)]))
            loss = torch.nn.functional.cross_entropy(net(xb).reshape(-1, 3), yb.reshape(-1))
            for q in params:
                q.grad = None
            loss.backward()
            t += 1
            _adam_step(torch, params, state, t)
        print("epoch {} loss {:.4f}".format(epoch + 1, float(loss)), flush=True)
    net.eval()
    weights = {k: v.detach().numpy().astype(float) for k, v in net.state_dict().items()}
    return weights, mean, std


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", choices=("v6", "v8"), default="v8")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED, help="network seed (ensemble members use 0, 100, 200)")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        print("FAIL: output already exists; choose a new path", file=sys.stderr)
        return 1
    recipe = tl.RECIPES[args.recipe]
    config = vt.load_viterbi_config("possession/configs/viterbi_v4.json")   # same default as train_learned
    games = tuple(idsse.match_ids())
    sequences, names = [], None
    for kind in recipe["folders"]:
        for index, game in enumerate(games):
            features, names, truth = tl.load_idsse(
                game, config, 100 + index if "drift" in kind else None,
                {"idsse_damaged": 300 + index, "idsse_drift_damaged": 400 + index}.get(kind),
                smooth=recipe.get("smooth_ball", False),
                reveal_seed=700 + index if recipe.get("reveal") else None)
            labels = np.array([vt.STATES.index(s) if s in vt.STATES else vt.STATES.index("D") for s in truth])
            features = features.astype(np.float32)
            swapped, mirrored = tl.learned.swap_teams(features, labels, names)
            sequences.append((features[::sequence.SUB], labels[::sequence.SUB]))
            sequences.append((swapped[::sequence.SUB], mirrored[::sequence.SUB]))
            print("loaded {} ({})".format(game, kind), flush=True)
    weights, mean, std = fit(sequences, seed=args.seed, epochs=args.epochs)
    labels_all = np.concatenate([y for _, y in sequences])
    prior = np.bincount(labels_all, minlength=3) / len(labels_all)
    sequence.save(output, params=weights, mean=mean, std=std, names=names, prior=prior, layers=2,
                  notes={"recipe": args.recipe, "seed": args.seed, "epochs": args.epochs, "training_data": "IDSSE copies of train_learned recipe {} "
                         "(same seeds), 12.5 fps, every sequence also with teams swapped".format(args.recipe)})
    print("saved {}".format(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
