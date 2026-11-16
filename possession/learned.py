"""Learned frame evidence for the H/A/D Viterbi decoder.

A multinomial logistic regression turns per-frame features into H/A/D probabilities;
`scale * (log p(state | x) - log prior)` replaces viterbi.py's hand-written scores and
the same Viterbi chooses the sequence. Features: ball proximity (v4's team evidence,
each team's nearest distance, relative ball/player speed), team movement (forward
speed, how far each team has pushed up, spread and its change, ball position relative
to the attack direction found from each half's keeper) and v4's dead-ball cues
(stillness, ball outside, exits, set-ups). Trained on Metrica's open sample games
(labels from their events, train_learned.py); every training frame also appears with
H and A swapped, so team-neutral cues cannot favour a side.

learned_v2 adds context and a small network: twelve of the features are repeated at
+-0.5, 1 and 2 s (add_context), and an ensemble of one-hidden-layer networks (fit_mlp)
replaces the logistic regression. Its probabilities are averaged with the mirrored
prediction on team-swapped frames, so the output is exactly symmetric.
"""

from dataclasses import asdict
import json

import numpy as np

from . import viterbi as vt

FEATURE_VERSION = 1
# numpy 2.0 with macOS Accelerate raises spurious divide/overflow warnings from finite
# matmuls (results identical to einsum); fit() checks the weights stay finite instead.
_QUIET = dict(divide="ignore", over="ignore", invalid="ignore")
SIGN_FLIP = ("prox", "prox_m6", "prox_m25", "ball_att_H", "near_sign", "near_sign_m3", "near_sign_m12")
CONTEXT_BASE = ("prox", "d_H", "d_A", "rel_H", "rel_A", "ball_speed", "group_speed", "outside",
                "visible", "ball_att_H", "fwd_H", "fwd_A")
CONTEXT_OFFSETS = (-50, -25, -12, 12, 25, 50)
# learned_v3: four control columns join the context, at finer offsets as well.
CONTEXT_BASE_V3 = CONTEXT_BASE + ("n2_H", "n2_A", "near_sign", "ball_kick")
CONTEXT_OFFSETS_V3 = (-50, -25, -12, -6, -3, 3, 6, 12, 25, 50)


def rolling_mean(x, half):
    """Centered mean over +-half frames ignoring NaN; NaN where there are no samples."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    total = np.r_[0., np.cumsum(np.where(ok, x, 0.))]
    count = np.r_[0, np.cumsum(ok)]
    index = np.arange(len(x))
    lo, hi = np.clip(index - half, 0, len(x)), np.clip(index + half + 1, 0, len(x))
    samples = count[hi] - count[lo]
    return np.where(samples > 0, (total[hi] - total[lo]) / np.maximum(1, samples), np.nan)


def _velocity(positions, fps, step=3):
    velocity = np.full(positions.shape, np.nan)
    velocity[step:-step] = (positions[2 * step:] - positions[:-2 * step]) * fps / (2 * step)
    return velocity


def attack_direction(players, teams, halves, pitch):
    """+1 per frame if H attacks towards x = length (its keeper sits nearest x = 0), else -1."""
    direction = np.ones(len(players))
    for lo, hi in halves:
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            mean_x = np.nanmean(players[lo:hi, :, 0], axis=0)
        seen = np.isfinite(players[lo:hi, :, 0]).mean(axis=0) > 0.5
        best = None
        for slot in np.flatnonzero(seen & (np.asarray(teams) == "H")):
            gap = min(mean_x[slot], pitch[0] - mean_x[slot])
            if best is None or gap < best[0]:
                best = (gap, mean_x[slot] < pitch[0] / 2)
        direction[lo:hi] = 1. if best is None or best[1] else -1.
    return direction


MAX_PLAYER_SPEED_MPS = 12.


def _player_velocity(players, fps, robust):
    """Per-slot velocity; robust drops impossible speeds (a track jumping to another player)."""
    velocity = _velocity(players, fps)
    if robust:
        with np.errstate(invalid="ignore"):
            velocity[np.linalg.norm(velocity, axis=-1) > MAX_PLAYER_SPEED_MPS] = np.nan
    return velocity


def track_age(players, cap):
    """Frames each slot has been continuously visible, capped (0 where not visible)."""
    seen = np.isfinite(players[..., 0])
    age = np.zeros(seen.shape)
    for t in range(len(seen)):
        age[t] = np.where(seen[t], (age[t - 1] + 1) if t else 1, 0)
    return np.minimum(age, cap)


def frame_features(ball, players, teams, *, fps, pitch, halves, config, robust=False):
    """(n, k) feature matrix and column names; missing values become 0 plus a flag column."""
    teams = np.asarray(teams)
    n = len(ball)
    visible = np.isfinite(ball).all(axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    out = {"prox": vt.team_evidence(ball, players, teams, config)}
    for half in (6, 25):
        out["prox_m%d" % half] = rolling_mean(np.where(visible > 0, out["prox"], np.nan), half)
    ball_v, player_v = _velocity(ball, fps), _player_velocity(players, fps, robust)
    for side in ("H", "A"):
        side_dist = np.where(teams == side, dist, np.inf)
        side_dist = np.where(np.isfinite(side_dist), side_dist, np.inf)
        nearest = side_dist.argmin(axis=1)
        closest = side_dist[np.arange(n), nearest]
        known = np.isfinite(closest)
        out["d_" + side] = np.where(known, np.minimum(closest, 15.) / 15., np.nan)
        with np.errstate(invalid="ignore"):
            relative = np.linalg.norm(ball_v - player_v[np.arange(n), nearest], axis=1)
        out["rel_" + side] = rolling_mean(np.where(known, np.minimum(relative, 20.), np.nan), 6) / 20.
    direction = attack_direction(players, teams, halves, pitch)
    for side, sign in (("H", direction), ("A", -direction)):
        cols = teams == side
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            forward = np.nanmean(player_v[:, cols, 0], axis=1) * sign
            centre_x = np.nanmean(players[:, cols, 0], axis=1)
            spread = np.sqrt(np.nanvar(players[:, cols, 0], axis=1) + np.nanvar(players[:, cols, 1], axis=1))
        own_goal = np.where(sign > 0, 0., pitch[0])
        out["fwd_" + side] = rolling_mean(forward, 25) / 5.
        out["push_" + side] = rolling_mean(np.abs(centre_x - own_goal), 25) / pitch[0]
        smooth_spread = rolling_mean(spread, 25)
        out["spread_" + side] = smooth_spread / 20.
        out["dspread_" + side] = (smooth_spread - rolling_mean(np.r_[np.full(25, np.nan), spread[:-25]], 25)) / 5.
    out["ball_att_H"] = rolling_mean((ball[:, 0] - pitch[0] / 2) * direction, 12) / (pitch[0] / 2)
    out["group_speed"] = np.minimum(vt.group_speed(players, fps, config), 5.) / 5.
    out["outside"] = vt.outside_fraction(ball, pitch, fps, config)
    exited, setup, _ = vt.border_evidence(ball, pitch, fps, config)
    out["exit"], out["setup"] = exited.astype(float), setup.astype(float)
    with np.errstate(invalid="ignore"):
        out["ball_speed"] = rolling_mean(np.minimum(np.linalg.norm(ball_v, axis=1), 30.), 6) / 30.
    out["visible"] = rolling_mean(visible, 12)
    names = sorted(out)
    matrix = np.stack([out[name] for name in names], axis=1)
    missing = ~np.isfinite(matrix)
    matrix = np.where(missing, 0., matrix)
    return np.c_[matrix, missing.any(axis=1)], names + ["any_missing"]


def add_context(features, names, base=CONTEXT_BASE, offsets=CONTEXT_OFFSETS):
    """Append each base column shifted by each offset in frames ("name@+k" is frame t+k).

    Shifts past either end of the match read 0, the value missing features take.
    """
    columns, out = [features], list(names)
    for name in base:
        x = features[:, names.index(name)]
        for k in offsets:
            shifted = np.zeros_like(x)
            if k > 0:
                shifted[:-k] = x[k:]
            else:
                shifted[-k:] = x[:k]
            columns.append(shifted[:, None])
            out.append("%s@%+d" % (name, k))
    return np.concatenate(columns, axis=1), out


def control_features(ball, players, teams, *, fps, robust=False):
    """learned_v3's extra columns: who crowds the ball, the second-nearest player of each
    team, which team is nearest (+1 H / -1 A, raw and smoothed) and ball kicks (acceleration)."""
    teams = np.asarray(teams)
    known = np.isfinite(ball).all(axis=1)
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    out, nearest = {}, {}
    for side in ("H", "A"):
        side_dist = np.where((teams == side)[None, :] & np.isfinite(dist), dist, np.inf)
        ordered = np.sort(side_dist, axis=1)
        nearest[side] = ordered[:, 0]
        if robust:
            age = track_age(players, cap=fps)
            slot = side_dist.argmin(axis=1)
            out["age_" + side] = np.where(known & np.isfinite(ordered[:, 0]), age[np.arange(len(ball)), slot] / fps, np.nan)
        for radius in (2, 5):
            out["n%d_%s" % (radius, side)] = rolling_mean(
                np.where(known, (ordered < radius).sum(axis=1), np.nan), 3) / 3.
        second = ordered[:, 1] if ordered.shape[1] > 1 else np.full(len(ball), np.inf)
        out["d2_" + side] = np.where(known & np.isfinite(second), np.minimum(second, 15.) / 15., np.nan)
    seen = known & (np.isfinite(nearest["H"]) | np.isfinite(nearest["A"]))
    sign = np.where(seen, np.where(nearest["H"] < nearest["A"], 1., -1.), np.nan)
    out["near_sign"], out["near_sign_m3"], out["near_sign_m12"] = sign, rolling_mean(sign, 3), rolling_mean(sign, 12)
    acceleration = np.full(len(ball), np.nan)
    acceleration[1:-1] = np.linalg.norm(ball[2:] - 2 * ball[1:-1] + ball[:-2], axis=1) * fps * fps
    out["ball_kick"] = rolling_mean(np.minimum(acceleration, 200.), 2) / 200.
    names = sorted(out)
    matrix = np.stack([out[name] for name in names], axis=1)
    return np.where(np.isfinite(matrix), matrix, 0.), names


def swap_teams(features, labels, names):
    """The same frames with H and A exchanged: paired columns swap, H-signed columns negate.

    Context columns ("name@k") follow their base column's rule.
    """
    swapped = features.copy()
    column = {name: i for i, name in enumerate(names)}
    for name in names:
        base, at, offset = name.partition("@")
        partner = base[:-2] + "_A" + at + offset
        if base.endswith("_H") and partner in column:
            swapped[:, column[name]] = features[:, column[partner]]
            swapped[:, column[partner]] = features[:, column[name]]
        if base in SIGN_FLIP:
            swapped[:, column[name]] = -features[:, column[name]]
    mirror = np.array([vt.STATES.index(state) for state in ("A", "H", "D")])
    return swapped, mirror[labels]


def fit(features, labels, *, l2=1e-3, iterations=800, rate=0.005):
    """Full-batch Adam on L2-regularised softmax regression; deterministic."""
    mean, std = features.mean(axis=0), features.std(axis=0) + 1e-9
    z = np.c_[(features - mean) / std, np.ones(len(features))]
    target = np.eye(3)[labels]
    weights = np.zeros((z.shape[1], 3))
    first, second = np.zeros_like(weights), np.zeros_like(weights)
    for step in range(1, iterations + 1):
        with np.errstate(**_QUIET):
            logits = z @ weights
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        with np.errstate(**_QUIET):
            grad = z.T @ (prob - target) / len(z) + l2 * np.r_[weights[:-1], np.zeros((1, 3))]
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad * grad
        weights -= rate * (first / (1 - 0.9 ** step)) / (np.sqrt(second / (1 - 0.999 ** step)) + 1e-8)
    if not np.isfinite(weights).all():
        raise ValueError("training diverged: non-finite weights")
    prior = np.bincount(labels, minlength=3) / len(labels)
    return {"mean": mean, "std": std, "weights": weights, "prior": prior}


def fit_symmetric(features, labels, names, **options):
    swapped, mirrored = swap_teams(features, labels, names)
    return fit(np.concatenate([features, swapped]), np.concatenate([labels, mirrored]), **options)


def fit_mlp(features, labels, *, hidden=64, epochs=12, batch=2048, rate=2e-3, l2=1e-4, seed=0):
    """One-hidden-layer ReLU network with softmax output, minibatch Adam; deterministic per seed."""
    rng = np.random.default_rng(seed)
    mean, std = features.mean(axis=0), features.std(axis=0) + 1e-9
    z_all = (features - mean) / std
    k = z_all.shape[1]
    params = {"W1": rng.normal(0, np.sqrt(2 / k), (k, hidden)), "b1": np.zeros(hidden),
              "W2": rng.normal(0, np.sqrt(1 / hidden), (hidden, 3)), "b2": np.zeros(3)}
    first = {name: np.zeros_like(value) for name, value in params.items()}
    second = {name: np.zeros_like(value) for name, value in params.items()}
    target_all = np.eye(3)[labels]
    step = 0
    with np.errstate(**_QUIET):
        for _ in range(epochs):
            order = rng.permutation(len(z_all))
            for lo in range(0, len(z_all), batch):
                index = order[lo:lo + batch]
                z, target = z_all[index], target_all[index]
                hidden_out = np.maximum(0, z @ params["W1"] + params["b1"])
                logits = hidden_out @ params["W2"] + params["b2"]
                logits -= logits.max(axis=1, keepdims=True)
                prob = np.exp(logits)
                prob /= prob.sum(axis=1, keepdims=True)
                error = (prob - target) / len(index)
                back = (error @ params["W2"].T) * (hidden_out > 0)
                grads = {"W2": hidden_out.T @ error + l2 * params["W2"], "b2": error.sum(axis=0),
                         "W1": z.T @ back + l2 * params["W1"], "b1": back.sum(axis=0)}
                step += 1
                for name in params:
                    first[name] = 0.9 * first[name] + 0.1 * grads[name]
                    second[name] = 0.999 * second[name] + 0.001 * grads[name] ** 2
                    params[name] -= rate * (first[name] / (1 - 0.9 ** step)) / (
                        np.sqrt(second[name] / (1 - 0.999 ** step)) + 1e-8)
    if not all(np.isfinite(value).all() for value in params.values()):
        raise ValueError("training diverged: non-finite weights")
    return dict(params, mean=mean, std=std)


def _mlp_log_probs(member, features):
    with np.errstate(**_QUIET):
        hidden_out = np.maximum(0, ((features - member["mean"]) / member["std"]) @ member["W1"] + member["b1"])
        logits = hidden_out @ member["W2"] + member["b2"]
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def fit_ensemble(features, labels, names, *, seeds=(0, 1, 2, 3, 4), stride=5, **options):
    """learned_v2: networks on every stride-th frame plus its team-swapped copy, one per seed."""
    swapped, mirrored = swap_teams(features[::stride], labels[::stride], names)
    x = np.concatenate([features[::stride], swapped])
    y = np.concatenate([labels[::stride], mirrored])
    return {"kind": "mlp_ensemble", "names": list(names), "prior": np.bincount(y, minlength=3) / len(y),
            "members": [fit_mlp(x, y, seed=seed, **options) for seed in seeds]}


def _normalise(log_p):
    return log_p - np.log(np.exp(log_p).sum(axis=1, keepdims=True))


def log_probs(model, features):
    if model.get("kind") == "mlp_ensemble":
        swapped, _ = swap_teams(features, np.zeros(len(features), dtype=int), model["names"])
        mirror = [vt.STATES.index(state) for state in ("A", "H", "D")]
        total = 0.
        for member in model["members"]:
            total = total + _mlp_log_probs(member, features) + _mlp_log_probs(member, swapped)[:, mirror]
        return _normalise(total / (2 * len(model["members"])))
    z = np.c_[(features - model["mean"]) / model["std"], np.ones(len(features))]
    with np.errstate(**_QUIET):
        logits = z @ model["weights"]
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def decode(model, features, *, scale, switch_cost, dead_switch_cost):
    scores = scale * (log_probs(model, features) - np.log(model["prior"]))
    return np.asarray(vt.STATES)[vt._viterbi(scores, switch_cost, dead_switch_cost)]


def model_inputs(model, ball, players, teams, *, fps, pitch, halves):
    """Everything a saved model reads, computed from tracking."""
    robust = bool(model.get("robust"))
    features, names = frame_features(ball, players, teams, fps=fps, pitch=pitch, halves=halves,
                                     config=model["viterbi_config"], robust=robust)
    if model.get("control"):
        extra, extra_names = control_features(ball, players, teams, fps=fps, robust=robust)
        features, names = np.c_[features, extra], names + extra_names
    return model_features(model, features, names)


def model_features(model, features, names):
    """Base frame_features -> the columns this model reads (adds context for learned_v2)."""
    if names != model["feature_names"]:
        raise ValueError("feature columns differ from the model's")
    context = model.get("context")
    if context:
        features, names = add_context(features, names, context["base"], context["offsets"])
    return features


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, list, tuple)):
        return np.asarray(value).tolist()
    return value


def save_model(path, model, *, names, config, scale, switch_cost, dead_switch_cost, notes, context=None,
               control=False, robust=False):
    payload = {
        "feature_version": FEATURE_VERSION, "feature_names": names, **({"control": True} if control else {}),
        **({"robust": True} if robust else {}),
        "viterbi_config": asdict(config), "scale": scale, "switch_cost": switch_cost,
        "dead_switch_cost": dead_switch_cost, "notes": notes,
        **({"context": {"base": list(context[0]), "offsets": list(context[1])}} if context else {}),
        **{key: _jsonable(value) for key, value in model.items()},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")


def load_model(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("model was trained with a different feature version")
    settings = dict(payload["viterbi_config"])
    if isinstance(settings.get("contacts"), dict):
        settings["contacts"] = vt.ContactConfig(**settings["contacts"])
    payload["viterbi_config"] = vt.ViterbiConfig(**settings)
    for key in ("mean", "std", "weights", "prior"):
        if key in payload:
            payload[key] = np.asarray(payload[key], dtype=float)
    for member in payload.get("members", []):
        for key in member:
            member[key] = np.asarray(member[key], dtype=float)
    return payload
