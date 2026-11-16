"""EXPERIMENTAL (v7 research, not the submission): leave-one-match-out IDSSE evaluation.

Independent of the competition match by construction:

* Augmentation is domain randomisation over ranges fixed in advance from physical priors
  (AUG_RANGES), not calibrated to any match. degrade.py's calibrated constants are not used.
* Decoder costs are chosen by nested cross-validation from a grid fixed in advance
  (DECODER_GRID). For outer match k, the decoder is selected on predictions for each other
  match j made by a "pair model" trained without both j and k, so k never influences either
  the models or the selection used for k. The same pair model serves outer fold j as well.
* The legacy decoder (scale 0.15, switch 1.0, dead 7.0; provenance incomplete) is reported
  separately as a reference, never selected.

Memory: only every STRIDE-th frame of each training copy is held (float32), and the training
matrix (rows + team-swapped rows) is assembled in one preallocated buffer. Training is
learned.fit_mlp per seed, exactly what learned.fit_ensemble(stride=1) does on the same rows
(tests/test_cv_idsse.py checks this bit for bit). Feature thresholds come from
configs/viterbi_v4.json (physical thresholds only; identical in every viterbi config).

    python -m possession.cv_idsse --config b1 --outer DFL-MAT-J03WMX
"""

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import degrade, idsse, learned
from . import validate_metrica as vm
from . import viterbi as vt

pm = vm.pm
OUT_DIR = Path("experiments/v7")
FEATURE_CONFIG = Path("possession/configs/viterbi_v4.json")
FPS = idsse.FPS
STRIDE = 5
SEEDS = (0, 1, 2, 3, 4)
HIDDEN = 128
TRAIN_COPIES = 4          # copy 0: observed ball, clean players; copies 1-3: randomised
SHORT_CONTEXT = (-12, -6, -3, 3, 6, 12)
CONFIGS = {
    "b1": dict(base=learned.CONTEXT_BASE_V3, offsets=learned.CONTEXT_OFFSETS_V3),
    "e4": dict(base=learned.CONTEXT_BASE_V3, offsets=SHORT_CONTEXT),
}

# Domain randomisation, fixed before any v7 evaluation. Wide physical priors for optical /
# broadcast tracking, deliberately not centred on any particular match.
AUG_RANGES = {
    "drift_sd_m": (0.0, 2.0),            # uniform; 0 = ball exactly where the provider put it
    "drift_smooth_s": (0.25, 1.0),       # uniform; time scale of slow ball-position error
    "ball_sigma_frames": (0, 8),         # integer uniform; 0 = no temporal smoothing
    "segment_mean_s": (20.0, 300.0),     # log-uniform; mean length of an unbroken player track
    "gap_mean_s": (1.0, 10.0),           # uniform; mean length of a track break
    "swaps_per_min": (0.0, 15.0),        # uniform; identity swaps between nearby players
}
SWAP_RADIUS_M = 5.0       # physical prior: a tracker confuses identities only of nearby players
# The held-out "stress" copy: the middle of every range (geometric middle for log-uniform).
STRESS = {"drift_sd_m": 1.0, "drift_smooth_s": 0.625, "ball_sigma_frames": 4,
          "segment_mean_s": float(np.sqrt(20.0 * 300.0)), "gap_mean_s": 5.5, "swaps_per_min": 7.5}

# Decoder search space, fixed before any v7 evaluation. Scores are log p(state) - log prior
# (scale 1; only cost/scale ratios matter to Viterbi). At an average evidence of 1 nat per
# frame, a team spell inside the other team's possession survives if 2*T < 25*d nats, so
# T in {2.5, 5, 10, 20} keeps spells of at least d = 0.2 / 0.4 / 0.8 / 1.6 s. A dead stretch
# inside play (or play between stoppages) survives if d_in + d_out < 25*d, so totals of
# {25, 50, 100, 200} keep stretches of at least 1 / 2 / 4 / 8 s. The entry share allows the
# whistle (entry) and the restart (exit) to be unequally observable, in either direction.
TEAM_COSTS = (2.5, 5.0, 10.0, 20.0)
DEAD_TOTALS = (25.0, 50.0, 100.0, 200.0)
ENTRY_SHARES = (1 / 3, 1 / 2, 2 / 3)
DECODER_GRID = tuple((t, total * share, total * (1 - share))
                     for t in TEAM_COSTS for total in DEAD_TOTALS for share in ENTRY_SHARES)
LEGACY = {"scale": 0.15, "switch_cost": 1.0, "dead_switch_cost": 7.0}

ROLE_TRAIN, ROLE_CLEAN, ROLE_STRESS = 1, 2, 3


def copy_seed(role, match_index, copy=0):
    return 10000 * role + 100 * match_index + copy


def draw_params(rng):
    lo, hi = AUG_RANGES["segment_mean_s"]
    return {
        "drift_sd_m": rng.uniform(*AUG_RANGES["drift_sd_m"]),
        "drift_smooth_s": rng.uniform(*AUG_RANGES["drift_smooth_s"]),
        "ball_sigma_frames": int(rng.integers(AUG_RANGES["ball_sigma_frames"][0], AUG_RANGES["ball_sigma_frames"][1] + 1)),
        "segment_mean_s": float(np.exp(rng.uniform(np.log(lo), np.log(hi)))),
        "gap_mean_s": rng.uniform(*AUG_RANGES["gap_mean_s"]),
        "swaps_per_min": rng.uniform(*AUG_RANGES["swaps_per_min"]),
    }


def damage_random(players, rng, *, segment_mean_s, gap_mean_s, swaps_per_min, fps=FPS):
    """Track breaks (exponential segments and gaps) and nearby identity swaps; positions only."""
    out = np.array(players, dtype=float)
    n, slots = out.shape[:2]
    for slot in range(slots):
        seen = np.flatnonzero(np.isfinite(out[:, slot, 0]))
        if len(seen) == 0:
            continue
        t = seen[0] + int(rng.exponential(segment_mean_s * fps) * rng.uniform())
        while t < seen[-1]:
            gap = max(1, int(rng.exponential(gap_mean_s * fps)))
            out[t:t + gap, slot] = np.nan
            t += gap + max(1, int(rng.exponential(segment_mean_s * fps)))
    for start in np.sort(rng.integers(0, n, size=rng.poisson(swaps_per_min * n / (60 * fps)))):
        visible = np.flatnonzero(np.isfinite(out[start, :, 0]))
        if len(visible) < 2:
            continue
        a = rng.choice(visible)
        distance = np.linalg.norm(out[start, visible] - out[start, a], axis=1)
        distance[visible == a] = np.inf
        if distance.min() > SWAP_RADIUS_M:
            continue
        b = visible[distance.argmin()]
        stop = min(n, start + max(1, int(rng.exponential(segment_mean_s * fps))))
        out[start:stop, [a, b]] = out[start:stop, [b, a]]
    return out


def make_copy(parsed, seed, params=None):
    """(ball, players): observed ball (dropout seed = seed), then damage, drift and smoothing."""
    rng = np.random.default_rng(seed)
    ball = idsse.observed_ball(parsed["ball"], seed=seed)
    players = parsed["players"]
    if params is not None:
        players = damage_random(players, rng, segment_mean_s=params["segment_mean_s"],
                                gap_mean_s=params["gap_mean_s"], swaps_per_min=params["swaps_per_min"])
        if params["drift_sd_m"] > 0:
            ball = ball + vm.smooth_drift(len(ball), params["drift_sd_m"], params["drift_smooth_s"], rng)
        ball = degrade.smooth_ball(ball, params["ball_sigma_frames"])
    return ball, players


def halves_of(period):
    return [(int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1]) + 1)
            for p in np.unique(period)]


def features(ball, players, parsed, config, setup):
    """float32 (n, k) model inputs and their names (frame + control features, with context)."""
    f, names = learned.frame_features(ball, players, parsed["teams"], fps=FPS, pitch=idsse.PITCH,
                                      halves=halves_of(parsed["period"]), config=config)
    extra, extra_names = learned.control_features(ball, players, parsed["teams"], fps=FPS)
    x, all_names = learned.add_context(np.c_[f, extra], names + extra_names, setup["base"], setup["offsets"])
    return x.astype(np.float32), names + extra_names, all_names


def label_indices(parsed):
    truth = idsse.labels(parsed)
    if (truth == "X").any():
        raise ValueError("IDSSE frame without a label")
    return truth, np.array([vt.STATES.index(s) for s in truth], dtype=np.int64)


def presample(x, y, stride=STRIDE):
    """Every stride-th frame of one copy, phase 0 (the documented sampling rule)."""
    return np.ascontiguousarray(x[::stride]), y[::stride].copy()


def training_matrix(keys, store, names):
    """One buffer: all presampled rows in key order, then their team-swapped mirror rows."""
    n = sum(len(store[k][0]) for k in keys)
    width = store[keys[0]][0].shape[1]
    buf = np.empty((2 * n, width), dtype=np.float32)
    y = np.empty(2 * n, dtype=np.int64)
    pos = 0
    for k in keys:
        x, labels = store[k]
        buf[pos:pos + len(x)], y[pos:pos + len(x)] = x, labels
        swapped, mirrored = learned.swap_teams(x, labels, names)
        buf[n + pos:n + pos + len(x)], y[n + pos:n + pos + len(x)] = swapped, mirrored
        pos += len(x)
    return buf, y


def fit_lowmem(keys, store, names, *, seeds=SEEDS, hidden=HIDDEN, **options):
    """Same model as learned.fit_ensemble(rows, labels, names, stride=1) on the presampled rows."""
    buf, y = training_matrix(keys, store, names)
    model = {"kind": "mlp_ensemble", "names": list(names), "prior": np.bincount(y, minlength=3) / len(y),
             "members": [learned.fit_mlp(buf, y, seed=seed, hidden=hidden, **options) for seed in seeds]}
    del buf
    return model


def viterbi_costs(scores, team_cost, dead_in, dead_out):
    """Viterbi over H/A/D with H<->A = team_cost, live->D = dead_in, D->live = dead_out."""
    cost = ((0., team_cost, dead_in), (team_cost, 0., dead_in), (dead_out, dead_out, 0.))
    n = len(scores)
    back = np.empty((n, 3), dtype=np.int8)
    value = list(scores[0])
    rows = scores.tolist()
    for t in range(1, n):
        new, pointers = [0.] * 3, [0] * 3
        for state in range(3):
            best, arg = value[0] - cost[0][state], 0
            for previous in (1, 2):
                candidate = value[previous] - cost[previous][state]
                if candidate > best:
                    best, arg = candidate, previous
            new[state], pointers[state] = best + rows[t][state], arg
        value = new
        back[t] = pointers
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(value))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(log_p, prior, decoder):
    """decoder = "legacy" or (team, dead_in, dead_out) in nats."""
    lp = np.asarray(log_p, dtype=np.float64) - np.log(prior)
    if decoder == "legacy":
        path = vt._viterbi(LEGACY["scale"] * lp, LEGACY["switch_cost"], LEGACY["dead_switch_cost"])
    else:
        path = viterbi_costs(lp, *decoder)
    return np.asarray(vt.STATES)[path]


def short_recall(gt_codes, pred_codes, lo_s, hi_s, iou=0.5):
    runs = pm.valid_runs(gt_codes)
    gt = pm.possession_spells(gt_codes, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gt, pm.possession_spells(pred_codes, runs)), len(gt), iou)
    keep = np.array([lo_s * FPS <= b - a < hi_s * FPS for a, b, _ in gt])
    return (float(flags[keep].mean()) if keep.any() else float("nan")), int(keep.sum())


def score(truth, labels):
    gt, pred = pm.label_codes(truth), pm.label_codes(labels)
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    out = {"spell_f1": r["spell_f1_mean"], "dead_f1": r["alive_dead_f1"], "fragmentation": r["fragmentation"],
           "trans_f1_0.5s": r["transitions"][12]["f1"], "trans_f1_1s": r["transitions"][25]["f1"],
           "n_pred_spells": r["n_pred_spells"], "n_gt_spells": r["n_gt_spells"]}
    out["short_recall_lt1s"], out["n_gt_lt1s"] = short_recall(gt, pred, 0, 1)
    out["short_recall_1_2s"], out["n_gt_1_2s"] = short_recall(gt, pred, 1, 2)
    return out


class Run:
    def __init__(self, config_name, out_dir=OUT_DIR, data_dir=idsse.DATA_DIR, max_minutes=None):
        self.name, self.setup = config_name, CONFIGS[config_name]
        self.out = Path(out_dir) / config_name
        self.data_dir = Path(data_dir)
        self.config = vt.load_viterbi_config(FEATURE_CONFIG)
        self.matches = idsse.match_ids(self.data_dir)
        self.index = {m: i for i, m in enumerate(self.matches)}
        self.start, self.max_minutes = time.time(), max_minutes
        self.store, self.names, self.base_names = {}, None, None
        self.log_lines = []

    # -- bookkeeping ---------------------------------------------------------------------
    def log(self, message):
        line = "[{:6.1f} min | maxrss {:.2f} GB] {}".format(
            (time.time() - self.start) / 60, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, message)
        print(line, flush=True)
        self.log_lines.append(line)
        if self.max_minutes and (time.time() - self.start) / 60 > self.max_minutes:
            raise SystemExit("STOP: exceeded {} minutes".format(self.max_minutes))

    def manifest(self):
        path = self.out / "manifest.json"
        if path.exists():
            return
        git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
        payload = {"config": self.name, "context_offsets": list(self.setup["offsets"]), "aug_ranges": AUG_RANGES,
                   "stress": STRESS, "swap_radius_m": SWAP_RADIUS_M, "train_copies": TRAIN_COPIES, "stride": STRIDE,
                   "seeds": SEEDS, "hidden": HIDDEN, "decoder_grid": DECODER_GRID, "legacy": LEGACY,
                   "feature_config": str(FEATURE_CONFIG), "matches": self.matches, "git_head": git("rev-parse", "HEAD"),
                   "git_branch": git("branch", "--show-current"), "python": platform.python_version(),
                   "numpy": np.__version__}
        path.write_text(json.dumps(payload, indent=1) + "\n")

    # -- data ----------------------------------------------------------------------------
    def parse(self, match):
        return idsse.parse_positions(self.data_dir, match)

    def labels_path(self, match):
        return self.out.parent / "labels" / "{}.npz".format(match)

    def build_store(self):
        for match in self.matches:
            parsed = self.parse(match)
            truth, y = label_indices(parsed)
            path = self.labels_path(match)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, labels=np.array([vt.STATES.index(s) for s in truth], dtype=np.int8))
            i = self.index[match]
            for copy in range(TRAIN_COPIES):
                seed = copy_seed(ROLE_TRAIN, i, copy)
                params = None if copy == 0 else draw_params(np.random.default_rng(seed + 5000))
                ball, players = make_copy(parsed, seed, params)
                x, self.base_names, self.names = features(ball, players, parsed, self.config, self.setup)
                self.store[match, copy] = presample(x, y)
                del x
            del parsed
            self.log("store: {} ({} copies)".format(match, TRAIN_COPIES))

    def eval_features(self, parsed, match, kind):
        i = self.index[match]
        if kind == "clean":
            ball, players = make_copy(parsed, copy_seed(ROLE_CLEAN, i))
        else:
            ball, players = make_copy(parsed, copy_seed(ROLE_STRESS, i), STRESS)
        return features(ball, players, parsed, self.config, self.setup)[0]

    # -- models and predictions ----------------------------------------------------------
    def train(self, train_matches):
        keys = [(m, c) for c in range(TRAIN_COPIES) for m in train_matches]
        return fit_lowmem(keys, self.store, self.names)

    def pred_path(self, tag, target, kind):
        return self.out / "preds" / "{}__{}__{}.npz".format(tag, target, kind)

    def predict(self, model, tag, targets):
        for target in targets:
            parsed = self.parse(target)
            for kind in ("clean", "stress"):
                log_p = learned.log_probs(model, self.eval_features(parsed, target, kind))
                path = self.pred_path(tag, target, kind)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, log_p=log_p.astype(np.float16), prior=model["prior"])
            del parsed

    def load_pred(self, tag, target, kind):
        data = np.load(self.pred_path(tag, target, kind))
        return data["log_p"].astype(np.float64), data["prior"]

    def truth(self, match):
        return np.asarray(vt.STATES)[np.load(self.labels_path(match))["labels"]]

    @staticmethod
    def pair_tag(a, b):
        return "pair_{}_{}".format(*sorted((a, b)))

    def run_outer(self, outer):
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest()
        others = [m for m in self.matches if m != outer]
        if not self.store:
            self.build_store()
        tag = "outer"
        if not self.pred_path(tag, outer, "stress").exists():
            model = self.train(others)
            self.log("trained outer model (held out {})".format(outer))
            path = self.out / "models" / "outer_{}.json".format(outer)
            path.parent.mkdir(parents=True, exist_ok=True)
            learned.save_model(path, model, names=self.base_names, config=self.config, scale=1.0, switch_cost=0.,
                               dead_switch_cost=0., context=(self.setup["base"], self.setup["offsets"]), control=True,
                               notes={"recipe": "v7-" + self.name, "experimental": True, "held_out": outer,
                                      "decoder": "not stored here; see results/outer_{}.json".format(outer)})
            self.predict(model, tag, [outer])
            del model
            self.log("predicted outer {}".format(outer))
        for inner in others:
            pair = self.pair_tag(outer, inner)
            if self.pred_path(pair, inner, "stress").exists() and self.pred_path(pair, outer, "stress").exists():
                continue
            model = self.train([m for m in self.matches if m not in (outer, inner)])
            self.predict(model, pair, [inner, outer])
            del model
            self.log("pair model without {} and {}".format(outer[-6:], inner[-6:]))
        return self.evaluate_outer(outer)

    def select_decoder(self, outer):
        """Grid point maximising mean stress spellF1 over the inner matches (pair-model predictions)."""
        others = [m for m in self.matches if m != outer]
        inner = {m: (self.truth(m), self.load_pred(self.pair_tag(outer, m), m, "stress")) for m in others}
        table = []
        for decoder in DECODER_GRID:
            values = [score(truth, decode(*pred, decoder))["spell_f1"] for truth, pred in inner.values()]
            table.append((float(np.mean(values)), decoder))
        best = max(table, key=lambda row: (row[0], -sum(row[1])))  # ties: lower total cost
        return best[1], table

    def evaluate_outer(self, outer):
        decoder, table = self.select_decoder(outer)
        self.log("selected decoder for {}: team {:.1f}, dead in {:.1f}, out {:.1f}".format(outer, *decoder))
        truth = self.truth(outer)
        result = {"outer": outer, "config": self.name, "selected_decoder": decoder,
                  "inner_table": [[v, list(d)] for v, d in sorted(table, reverse=True)], "scores": {}}
        for kind in ("clean", "stress"):
            pred = self.load_pred("outer", outer, kind)
            result["scores"][kind] = {"legacy": score(truth, decode(*pred, "legacy")),
                                      "nested": score(truth, decode(*pred, decoder))}
        path = self.out / "results" / "outer_{}.json".format(outer)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1) + "\n")
        self.log("evaluated outer {}".format(outer))
        (self.out / "log.txt").open("a").write("\n".join(self.log_lines) + "\n")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--outer", nargs="+", required=True, help="held-out IDSSE match id(s)")
    parser.add_argument("--max-minutes", type=float, default=None)
    args = parser.parse_args(argv)
    run = Run(args.config, max_minutes=args.max_minutes)
    for outer in args.outer:
        if outer not in run.matches:
            print("FAIL: unknown match {}".format(outer), file=sys.stderr)
            return 1
    for outer in args.outer:
        result = run.run_outer(outer)
        for kind in ("clean", "stress"):
            for dec in ("legacy", "nested"):
                s = result["scores"][kind][dec]
                print("{} {:<6} {:<6} spellF1 {:.4f} deadF1 {:.4f} frag {:.3f} tF1@0.5s {:.4f} tF1@1s {:.4f} "
                      "short<1s {:.3f} (n={}) 1-2s {:.3f}".format(
                          outer, kind, dec, s["spell_f1"], s["dead_f1"], s["fragmentation"], s["trans_f1_0.5s"],
                          s["trans_f1_1s"], s["short_recall_lt1s"], s["n_gt_lt1s"], s["short_recall_1_2s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
