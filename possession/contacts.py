"""Possible touches from local trajectory fits, not inferred player intent."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional, Sequence, Tuple

import numpy as np


def _number(name, value, *, positive=False, integer=False):
    expected = Integral if integer else Real
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, expected)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError("{} must be a finite {}{}".format(
            name, "positive " if positive else "nonnegative ",
            "integer" if integer else "number"))


@dataclass(frozen=True)
class ContactConfig:
    """Explicit experimental settings; distances in metres and speeds in m/s."""

    window_frames: int
    max_distance_m: float
    attribution_margin_m: float
    min_velocity_change_mps: float
    max_fit_error_m: float
    max_ball_speed_mps: float
    max_player_speed_mps: float

    def __post_init__(self):
        _number("window_frames", self.window_frames, integer=True)
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        for name in ("max_distance_m", "attribution_margin_m", "max_fit_error_m"):
            _number(name, getattr(self, name))
        for name in ("min_velocity_change_mps", "max_ball_speed_mps", "max_player_speed_mps"):
            _number(name, getattr(self, name), positive=True)


@dataclass(frozen=True)
class Contact:
    frame: int
    team: str
    track_index: int
    distance_m: float
    incoming_velocity_mps: Tuple[float, float]
    outgoing_velocity_mps: Tuple[float, float]
    velocity_change_mps: float
    turn_angle_deg: float
    speed_change_mps: float
    fit_error_m: float


def _tracking_inputs(ball_xy_m, player_xy_m, teams, fps, reset, dead):
    ball = np.asarray(ball_xy_m, dtype=float)
    players = np.asarray(player_xy_m, dtype=float)
    teams = np.asarray(teams)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("ball_xy_m must have shape (frames, 2)")
    if players.ndim != 3 or players.shape[0] != len(ball) or players.shape[2] != 2:
        raise ValueError("player_xy_m must have shape (frames, slots, 2)")
    if teams.shape != (players.shape[1],) or not np.isin(teams, ("H", "A")).all():
        raise ValueError("teams must contain H/A for each track slot")
    _number("fps", fps, positive=True)
    masks = []
    for name, mask in (("reset", reset), ("dead", dead)):
        if mask is None:
            mask = np.zeros(len(ball), dtype=bool)
        mask = np.asarray(mask)
        if mask.shape != (len(ball),) or mask.dtype.kind != "b":
            raise ValueError("{} must be a boolean sequence matching frames".format(name))
        masks.append(mask)
    return ball, players, teams, masks[0], masks[1]


def _motion(positions, fps, max_speed, reset):
    """Backward velocities and validity; never bridge gaps, resets or fast jumps."""
    velocity = np.zeros_like(positions)
    valid = np.zeros(positions.shape[:-1], dtype=bool)
    visible = np.isfinite(positions).all(axis=-1)
    safe = np.where(np.isfinite(positions), positions, 0.)
    velocity[1:] = np.diff(safe, axis=0) * fps
    boundary_shape = (-1,) + (1,) * (positions.ndim - 2)
    valid[1:] = (visible[1:] & visible[:-1]
                 & (~reset[1:]).reshape(boundary_shape)
                 & (np.linalg.norm(velocity[1:], axis=-1) <= max_speed))
    return velocity, valid


def detect_contacts(
    ball_xy_m: np.ndarray,
    player_xy_m: np.ndarray,
    teams: Sequence[str],
    *,
    fps: float,
    config: ContactConfig,
    reset: Optional[Sequence[bool]] = None,
    dead: Optional[Sequence[bool]] = None,
) -> Tuple[Contact, ...]:
    """Find possible contacts near a reliable change in the ball trajectory.

    Inputs use the metre-based, consecutive-frame convention of estimate_control.
    At each frame fit separate straight trajectories on window_frames intervals
    before/after it, including the proposed contact in both fits. Fits must have
    small residuals and no missing observations, reset, dead frame or implausible
    speed. Never interpolate across a gap. This is an OFFLINE detector requiring
    window_frames future observations.

    A nearby player is attributable only with reliable movement on both sides
    of contact and a distance advantage over other visible players. Ambiguous
    contacts remain team X / slot -1; even players of the same team can have
    ambiguous attribution. Adjacent weaker candidates for the same slot are
    suppressed, not contacts involving different players. Events closer than the
    fit-window resolution can be missed. Straight-through touches with negligible
    velocity change and contacts near recording/gap boundaries can also be missed.

    These are possible contacts only: direction change and speed/energy change
    do not establish control or intent. Speed limits reject some jumps, but not
    plausible-looking identity switches. No default thresholds are provided.
    """
    if not isinstance(config, ContactConfig):
        raise ValueError("config must be a ContactConfig")
    ball, players, teams, reset, dead = _tracking_inputs(
        ball_xy_m, player_xy_m, teams, fps, reset, dead)
    n, slots = players.shape[:2]
    if slots == 0:
        return ()
    _, ball_motion_ok = _motion(ball, fps, config.max_ball_speed_mps, reset)
    _, player_motion_ok = _motion(players, fps, config.max_player_speed_mps, reset)
    w = config.window_frames
    times = np.arange(w + 1, dtype=float) / fps
    times -= times.mean()
    denominator = np.dot(times, times)

    def fit(points):
        center = points.mean(axis=0)
        velocity = times @ (points - center) / denominator
        error = np.linalg.norm(points - (center + times[:, None] * velocity), axis=1).max()
        return velocity, float(error)

    candidates = []
    for frame in range(w, n - w):
        lo, hi = frame - w, frame + w + 1
        if (dead[lo:hi].any() or not np.isfinite(ball[lo:hi]).all()
                or not ball_motion_ok[lo + 1:hi].all()):
            continue
        incoming, err_in = fit(ball[lo:frame + 1])
        outgoing, err_out = fit(ball[frame:hi])
        change = float(np.linalg.norm(outgoing - incoming))
        error = max(err_in, err_out)
        if error > config.max_fit_error_m or change < config.min_velocity_change_mps:
            continue

        visible = np.isfinite(players[frame]).all(axis=-1)
        distances = np.full(slots, np.inf)
        distances[visible] = np.linalg.norm(players[frame, visible] - ball[frame], axis=-1)
        slot = int(np.argmin(distances))
        if distances[slot] > config.max_distance_m:
            continue
        others = distances.copy()
        others[slot] = np.inf
        attributable = (player_motion_ok[frame, slot] and player_motion_ok[frame + 1, slot]
                        and others.min() > distances[slot] + config.attribution_margin_m)
        speed_in, speed_out = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        angle = float("nan")
        if speed_in > 0 and speed_out > 0:
            angle = float(np.degrees(np.arccos(np.clip(
                np.dot(incoming, outgoing) / (speed_in * speed_out), -1., 1.))))
        candidates.append(Contact(
            frame, str(teams[slot]) if attributable else "X", slot if attributable else -1,
            float(distances[slot]), tuple(incoming.tolist()), tuple(outgoing.tolist()),
            change, angle, float(speed_out - speed_in), error))

    # Do not impose a minimum possession duration or merge contacts across slots.
    return tuple(event for i, event in enumerate(candidates) if not any(
        other.track_index == event.track_index and other.team == event.team
        and abs(other.frame - event.frame) == 1
        and (other.velocity_change_mps > event.velocity_change_mps
             or (other.velocity_change_mps == event.velocity_change_mps
                 and other.frame < event.frame))
        for other in candidates[max(0, i - 1):i] + candidates[i + 1:i + 2]))
