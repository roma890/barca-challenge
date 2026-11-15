"""Offline H/A/D decoding from soft proximity and group-stillness evidence.

Each frame scores three states. Team evidence favours whichever side's nearest
player is closer to the ball, fading when nobody is near it. Dead-ball evidence
favours D when the players' median speed drops below walking pace. A Viterbi
pass over the whole match then chooses the most consistent sequence, paying a
fixed cost for every change, so changes land where the evidence flips rather
than after a confirmation delay.

Optional extras: a visible ball outside the pitch adds dead-ball evidence, and
team evidence can fade as the group slows from jogging to standing, since a
player walking the ball back during a stoppage is near it without possessing it.
Border evidence marks a ball leaving over a line (then unseen inside the pitch)
and a ball held still at a line, which is how throw-ins and corners look here.
Touch evidence credits the team of a player next to a detected change in the
ball's path (contacts.py), which catches one-touch passes and quick
interceptions that are too brief for proximity alone.

Known limits: a possession shorter than about one second of clear evidence can
be absorbed into its neighbours, and stillness or an outside ball can also occur
in slow live play near a touchline.
"""

from dataclasses import dataclass, fields
import json
from numbers import Real
from typing import Optional, Sequence

import numpy as np

from .contacts import ContactConfig, detect_contacts

STATES = ("H", "A", "D")
_MAY_BE_ZERO = ("outside_gain", "exit_gain", "setup_gain", "touch_gain")
# Same plausibility ceiling the restart and play-state settings use for ball motion.
MAX_BALL_SPEED_MPS = 40.0


@dataclass(frozen=True)
class ViterbiConfig:
    team_gain: float
    margin_m: float
    reach_m: float
    walk_mps: float
    dead_gain: float
    max_speed_gap_mps: float
    window_s: float
    switch_cost: float
    dead_switch_cost: float
    max_player_speed_mps: float
    outside_gain: float
    outside_margin_m: float
    damp_team_when_still: bool
    still_mps: float
    active_mps: float
    exit_gain: float
    line_zone_m: float
    exit_speed_mps: float
    exit_confirm_s: float
    exit_max_s: float
    setup_gain: float
    setup_zone_m: float
    setup_ball_mps: float
    touch_gain: float
    touch_window_s: float
    contacts: Optional[ContactConfig] = None

    def __post_init__(self):
        if not isinstance(self.damp_team_when_still, bool):
            raise ValueError("damp_team_when_still must be boolean")
        if self.contacts is not None and not isinstance(self.contacts, ContactConfig):
            raise ValueError("contacts must be a ContactConfig")
        if self.touch_gain > 0 and self.contacts is None:
            raise ValueError("touch_gain needs contacts settings")
        for item in fields(self):
            if item.name in ("damp_team_when_still", "contacts"):
                continue
            value = getattr(self, item.name)
            minimum_ok = value >= 0 if item.name in _MAY_BE_ZERO else value > 0
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or not minimum_ok):
                raise ValueError("{} must be a finite positive number".format(item.name))
        if self.active_mps <= self.still_mps:
            raise ValueError("active_mps must exceed still_mps")


def load_viterbi_config(path) -> ViterbiConfig:
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    try:
        if isinstance(values.get("contacts"), dict):
            values["contacts"] = ContactConfig(**values["contacts"])
        return ViterbiConfig(**values)
    except TypeError:
        raise ValueError("viterbi config has missing or unexpected fields") from None


@dataclass(frozen=True)
class ViterbiResult:
    labels: np.ndarray
    team_evidence: np.ndarray
    group_speed_mps: np.ndarray
    diagnostics: dict


def team_evidence(ball, players, teams, config):
    """Per-frame value in [-1, 1]: positive when H is nearer the ball, 0 without a ball."""
    teams = np.asarray(teams)
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    nearest = {}
    for side in ("H", "A"):
        side_dist = np.where(teams == side, dist, np.inf)
        side_dist = np.where(np.isfinite(side_dist), side_dist, np.inf)
        nearest[side] = side_dist.min(axis=1)
    known = np.isfinite(nearest["H"]) & np.isfinite(nearest["A"])
    for side in nearest:
        nearest[side] = np.where(known, nearest[side], 0.)
    near = np.minimum(nearest["H"], nearest["A"])
    gap = nearest["A"] - nearest["H"]
    return np.where(known, np.tanh(gap / config.margin_m) * np.exp(-near / config.reach_m), 0.)


def group_speed(players, fps, config):
    """Window mean of the players' median step speed; walk_mps (neutral) where unknown."""
    n = len(players)
    with np.errstate(invalid="ignore"):
        step = np.linalg.norm(np.diff(players, axis=0), axis=2) * fps
    step = np.where(step <= config.max_player_speed_mps, step, np.nan)
    median = np.full(n, np.nan)
    has = np.isfinite(step).any(axis=1)
    median[1:][has] = np.nanmedian(step[has], axis=1)
    known = np.isfinite(median)
    total = np.r_[0., np.cumsum(np.where(known, median, 0.))]
    count = np.r_[0, np.cumsum(known)]
    half = max(1, int(config.window_s * fps) // 2)
    index = np.arange(n)
    lo, hi = np.clip(index - half, 0, n), np.clip(index + half, 0, n)
    samples = count[hi] - count[lo]
    return np.where(samples > 0, (total[hi] - total[lo]) / np.maximum(1, samples), config.walk_mps)


def outside_fraction(ball, pitch, fps, config):
    """Window fraction of frames whose visible ball lies beyond the pitch margin."""
    length, width = pitch
    margin = config.outside_margin_m
    with np.errstate(invalid="ignore"):
        outside = ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin)
                   | (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))
    outside &= np.isfinite(ball).all(axis=1)
    n = len(ball)
    total = np.r_[0, np.cumsum(outside)]
    half = max(1, int(config.window_s * fps) // 2)
    index = np.arange(n)
    lo, hi = np.clip(index - half, 0, n), np.clip(index + half, 0, n)
    return (total[hi] - total[lo]) / np.maximum(1, hi - lo)


def _line_distances(ball, pitch):
    """Signed distance to the left, right, bottom and top lines (negative once beyond)."""
    length, width = pitch
    return np.stack([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]], axis=1)


def border_evidence(ball, pitch, fps, config):
    """(exit, setup) masks: the ball leaving over a line, and the ball held still at a line.

    Exit: a visible ball within line_zone_m of a line, moving outwards at exit_speed_mps or
    more, then not seen inside the pitch for exit_confirm_s. The frames after it count until
    the ball is next seen inside, capped at exit_max_s. Setup: a visible ball within
    setup_zone_m of a line whose windowed speed is at most setup_ball_mps.
    """
    n = len(ball)
    visible = np.isfinite(ball).all(axis=1)
    lines = _line_distances(np.where(visible[:, None], ball, 0.), pitch)
    margin = config.outside_margin_m
    inside = visible & (lines >= -margin).all(axis=1)

    position = np.where(inside, np.arange(n), n)
    next_inside = np.r_[np.minimum.accumulate(position[::-1])[::-1][1:], n]

    velocity = np.zeros((n, 2))
    moving = np.zeros(n, dtype=bool)
    velocity[2:] = (ball[2:] - ball[:-2]) * fps / 2.
    moving[2:] = visible[2:] & visible[:-2]
    with np.errstate(invalid="ignore"):
        moving &= np.linalg.norm(velocity, axis=1) <= MAX_BALL_SPEED_MPS
    outward = np.stack([-velocity[:, 0], velocity[:, 0], -velocity[:, 1], velocity[:, 1]], axis=1)
    leaving = (moving & ((lines <= config.line_zone_m)
                         & (outward >= config.exit_speed_mps)).any(axis=1))
    confirm = int(round(config.exit_confirm_s * fps))
    cap = int(round(config.exit_max_s * fps))
    exits = np.flatnonzero(leaving & (next_inside - np.arange(n) >= confirm))
    cover = np.zeros(n + 1, dtype=int)
    np.add.at(cover, exits + 1, 1)
    np.add.at(cover, np.minimum(next_inside[exits], exits + 1 + cap), -1)
    exited = np.cumsum(cover[:n]) > 0

    step = np.zeros(n)
    stepped = np.zeros(n, dtype=bool)
    step[1:] = np.linalg.norm(np.diff(np.where(visible[:, None], ball, 0.), axis=0), axis=1) * fps
    stepped[1:] = visible[1:] & visible[:-1]
    stepped &= step <= MAX_BALL_SPEED_MPS
    total = np.r_[0., np.cumsum(np.where(stepped, step, 0.))]
    count = np.r_[0, np.cumsum(stepped)]
    half = max(1, int(config.window_s * fps) // 2)
    index = np.arange(n)
    lo, hi = np.clip(index - half, 0, n), np.clip(index + half, 0, n)
    samples = count[hi] - count[lo]
    ball_speed = np.where(samples > 0, (total[hi] - total[lo]) / np.maximum(1, samples), np.inf)
    near_line = visible & (np.abs(lines).min(axis=1) <= config.setup_zone_m)
    setup = near_line & (ball_speed <= config.setup_ball_mps)
    return exited, setup, len(exits)


def touch_evidence(contacts, n, fps, config):
    """+1 around H touches, -1 around A touches, 0 elsewhere; later touches overwrite."""
    width = max(1, int(round(config.touch_window_s * fps)))
    touch = np.zeros(n)
    for contact in contacts:
        if contact.team in ("H", "A"):
            start = max(0, contact.frame - width // 2)
            touch[start:start + width] = 1. if contact.team == "H" else -1.
    return touch


def _viterbi(scores, switch_cost, dead_switch_cost):
    cost = ((0., switch_cost, dead_switch_cost),
            (switch_cost, 0., dead_switch_cost),
            (dead_switch_cost, dead_switch_cost, 0.))
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


def decode_possession(ball_xy_m: np.ndarray, player_xy_m: np.ndarray, teams: Sequence[str], *,
                      fps: float, pitch: Sequence[float], config: ViterbiConfig) -> ViterbiResult:
    if not isinstance(config, ViterbiConfig):
        raise ValueError("config must be a ViterbiConfig")
    pitch = tuple(float(value) for value in pitch)
    if len(pitch) != 2 or not all(np.isfinite(value) and value > 0 for value in pitch):
        raise ValueError("pitch must be a positive (length, width) in metres")
    ball = np.asarray(ball_xy_m, dtype=float)
    players = np.asarray(player_xy_m, dtype=float)
    n = len(ball)
    if n == 0 or ball.shape != (n, 2) or players.ndim != 3 or players.shape[0] != n or players.shape[2] != 2:
        raise ValueError("expected ball (n, 2) and players (n, slots, 2) with n > 0")
    if len(teams) != players.shape[1]:
        raise ValueError("teams must name every player slot")
    team = team_evidence(ball, players, teams, config)
    speed = group_speed(players, fps, config)
    if config.damp_team_when_still:
        team = team * np.clip((speed - config.still_mps) / (config.active_mps - config.still_mps), 0., 1.)
    gap = np.clip(config.walk_mps - speed, -config.max_speed_gap_mps, config.max_speed_gap_mps)
    outside = outside_fraction(ball, pitch, fps, config)
    exited, setup, n_exits = border_evidence(ball, pitch, fps, config)
    dead = (config.dead_gain * gap + config.outside_gain * outside
            + config.exit_gain * exited + config.setup_gain * setup)
    touches = ()
    team_score = config.team_gain * team
    if config.touch_gain > 0:
        touches = detect_contacts(ball, players, teams, fps=fps, config=config.contacts)
        team_score = team_score + config.touch_gain * touch_evidence(touches, n, fps, config)
    scores = np.stack([team_score, -team_score, dead], axis=1)
    labels = np.asarray(STATES)[_viterbi(scores, config.switch_cost, config.dead_switch_cost)]
    changes = int(np.sum(labels[1:] != labels[:-1]))
    diagnostics = {
        "method": "offline_three_state_viterbi",
        "label_counts": {state: int(np.sum(labels == state)) for state in STATES},
        "state_changes": changes,
        "frames_with_team_evidence": int(np.sum(team != 0)),
        "frames_with_outside_evidence": int(np.sum(outside > 0)),
        "exit_events": int(n_exits),
        "frames_with_exit_evidence": int(exited.sum()),
        "frames_with_setup_evidence": int(setup.sum()),
        "touches": len(touches),
    }
    return ViterbiResult(labels, team, speed, diagnostics)
