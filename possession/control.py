"""Conservative geometric control evidence; no tracking files are opened here."""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ControlConfig:
    """Experimental thresholds in physical units; none are match-tuned defaults."""

    max_distance_m: float
    max_relative_speed_mps: float
    opponent_margin_m: float

    def __post_init__(self):
        for name in ("max_distance_m", "max_relative_speed_mps", "opponent_margin_m"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                    or not np.isreal(value) or not np.isfinite(value) or value < 0):
                raise ValueError("{} must be a finite nonnegative number".format(name))


class ControlResult(NamedTuple):
    labels: np.ndarray
    track_index: np.ndarray
    distance_m: np.ndarray
    relative_speed_mps: np.ndarray


def estimate_control(
    ball_xy_m: np.ndarray,
    player_xy_m: np.ndarray,
    teams: Sequence[str],
    *,
    fps: float,
    config: ControlConfig,
    reset: Optional[Sequence[bool]] = None,
) -> ControlResult:
    """Return H/A evidence where proximity AND relative motion support control.

    Arrays must already be aligned to consecutive video frames. Ball positions
    have shape (frames, 2), player positions (frames, track_slots, 2), and teams
    (track_slots,) with H/A entries. Positions MUST be in metres, using the same
    pitch origin and axis directions for ball and players. Convert normalised
    tracking coordinates with (pitch_length, pitch_width) before calling.

    Velocities are backward differences over adjacent finite observations only.
    The first frame, reappearances, and frames marked by reset cannot establish
    control without a valid preceding observation. There is no interpolation,
    minimum spell duration, lookahead, smoothing, or boundary backdating.

    Select the closest player passing the distance and relative-speed gates.
    Abstain if any visible opponent is within opponent_margin_m of that player's
    ball distance, even if the opponent's velocity is unavailable. Ambiguity
    between players of the same team does not prevent team-level evidence.

    X means insufficient evidence, never dead ball. Stationary players beside a
    stationary ball can pass these gates: independent dead-ball evidence must
    override them in decode_state. Compatible motion is a control hypothesis,
    not proof of intent; one-touch kicks and noisy observations may be missed.

    Rejected frames have track_index -1 and NaN diagnostics. Accepted diagnostics
    explain the selection, not a calibrated probability of correct control.
    """
    ball = np.asarray(ball_xy_m, dtype=float)
    players = np.asarray(player_xy_m, dtype=float)
    teams = np.asarray(teams)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("ball_xy_m must have shape (frames, 2)")
    if (players.ndim != 3 or players.shape[0] != len(ball)
            or players.shape[2] != 2):
        raise ValueError("player_xy_m must have shape (frames, track_slots, 2)")
    if teams.shape != (players.shape[1],) or not np.isin(teams, ("H", "A")).all():
        raise ValueError("teams must contain H/A for each track slot")
    if (isinstance(fps, (bool, np.bool_)) or not np.isscalar(fps)
            or not np.isreal(fps) or not np.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be a finite positive number")
    if not isinstance(config, ControlConfig):
        raise ValueError("config must be a ControlConfig")
    if reset is None:
        reset = np.zeros(len(ball), dtype=bool)
    else:
        reset = np.asarray(reset)
        if reset.shape != (len(ball),) or reset.dtype.kind != "b":
            raise ValueError("reset must be a boolean sequence matching frames")

    n_frames, n_slots = players.shape[:2]
    labels = np.full(n_frames, "X", dtype="<U1")
    selected = np.full(n_frames, -1, dtype=int)
    distances = np.full(n_frames, np.nan)
    relative_speeds = np.full(n_frames, np.nan)
    if n_frames == 0 or n_slots == 0:
        return ControlResult(labels, selected, distances, relative_speeds)

    ball_visible = np.isfinite(ball).all(axis=-1)
    player_visible = np.isfinite(players).all(axis=-1)
    # Sanitising nonfinite positions avoids invalid arithmetic; visibility masks
    # ensure the replacement coordinates can never count as observations.
    safe_ball = np.where(np.isfinite(ball), ball, 0.0)
    safe_players = np.where(np.isfinite(players), players, 0.0)
    distance = np.linalg.norm(safe_players - safe_ball[:, None, :], axis=-1)
    distance[~(player_visible & ball_visible[:, None])] = np.inf

    relative_speed = np.full((n_frames, n_slots), np.inf)
    ball_velocity = np.diff(safe_ball, axis=0) * fps
    player_velocity = np.diff(safe_players, axis=0) * fps
    valid_motion = (
        (ball_visible[1:] & ball_visible[:-1] & ~reset[1:])[:, None]
        & player_visible[1:] & player_visible[:-1]
    )
    motion = np.linalg.norm(player_velocity - ball_velocity[:, None, :], axis=-1)
    relative_speed[1:] = np.where(valid_motion, motion, np.inf)
    eligible = ((distance <= config.max_distance_m)
                & (relative_speed <= config.max_relative_speed_mps))
    candidate_distances = np.where(eligible, distance, np.inf)
    candidate = np.argmin(candidate_distances, axis=1)
    frames = np.arange(n_frames)
    best_distance = candidate_distances[frames, candidate]
    candidate_team = teams[candidate]
    opponent_distance = np.min(
        np.where(teams[None, :] != candidate_team[:, None], distance, np.inf), axis=1
    )
    accepted = (np.isfinite(best_distance)
                & (opponent_distance > best_distance + config.opponent_margin_m))
    labels[accepted] = candidate_team[accepted]
    selected[accepted] = candidate[accepted]
    distances[accepted] = best_distance[accepted]
    relative_speeds[accepted] = relative_speed[frames[accepted], candidate[accepted]]
    return ControlResult(labels, selected, distances, relative_speeds)
