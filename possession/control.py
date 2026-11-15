"""Conservative geometric control evidence; no tracking files are opened here."""

from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ControlConfig:
    """Experimental thresholds in physical units; none are match-tuned defaults."""

    max_distance_m: float
    max_relative_speed_mps: float
    opponent_margin_m: float
    motion_window_frames: int = 1
    max_motion_fit_error_m: float = 0.1

    def __post_init__(self):
        for name in ("max_distance_m", "max_relative_speed_mps", "opponent_margin_m", "max_motion_fit_error_m"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                    or not np.isreal(value) or not np.isfinite(value) or value < 0):
                raise ValueError("{} must be a finite nonnegative number".format(name))
        window = self.motion_window_frames
        if (isinstance(window, (bool, np.bool_)) or not isinstance(window, Integral)
                or window < 1 or window % 2 == 0):
            raise ValueError("motion_window_frames must be one or an odd integer >= 3")


class ControlResult(NamedTuple):
    labels: np.ndarray
    track_index: np.ndarray
    distance_m: np.ndarray
    relative_speed_mps: np.ndarray
    compatible: Optional[np.ndarray] = None  # Per-player eligibility before opponent ambiguity.
    fit_used: Optional[np.ndarray] = None  # Accepted evidence using a local fit.


def _relative_motion(ball, players, fps, config, reset, breaks):
    """Fit relative positions without filling gaps or crossing touch boundaries.

    Local integer time offsets avoid subtraction of large absolute timestamps.
    This supplies extra evidence without replacing accepted adjacent-frame
    control. Missing samples or poor fits cannot supply that extra evidence.
    Current/previous observations are still required.
    """
    n, slots = players.shape[:2]
    visible = np.isfinite(ball).all(axis=-1)[:, None] & np.isfinite(players).all(axis=-1)
    safe_ball = np.where(np.isfinite(ball), ball, 0.)
    safe_players = np.where(np.isfinite(players), players, 0.)
    relative_positions = safe_players - safe_ball[:, None]
    adjacent = np.zeros((n, slots), dtype=bool)
    adjacent[1:] = visible[1:] & visible[:-1] & ~reset[1:, None]
    speed = np.full((n, slots), np.inf)
    ball_velocity = np.diff(safe_ball, axis=0) * fps
    player_velocity = np.diff(safe_players, axis=0) * fps
    speed[1:] = np.where(adjacent[1:], np.linalg.norm(
        player_velocity - ball_velocity[:, None], axis=-1), np.inf)
    fitted = np.zeros((n, slots), dtype=bool)
    if config.motion_window_frames == 1 or not n or not slots:
        return speed, fitted, speed
    low, high = np.empty(n, dtype=int), np.empty(n, dtype=int)
    boundaries = np.unique(np.r_[0, np.flatnonzero(reset | breaks), n])
    half = config.motion_window_frames // 2
    frames = np.arange(n)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        low[start:stop] = np.maximum(start, frames[start:stop] - half)
        high[start:stop] = np.minimum(stop, frames[start:stop] + half + 1)
    width = high - low
    total_t, total_tt = np.zeros(n), np.zeros(n)
    total_x = np.zeros_like(relative_positions)
    total_tx = np.zeros_like(relative_positions)
    all_known = np.ones((n, slots), dtype=bool)
    for offset in range(-half, half + 1):
        indices = np.clip(frames + offset, 0, n - 1)
        included = (frames + offset >= low) & (frames + offset < high)
        value = relative_positions[indices] * included[:, None, None]
        total_t += offset * included
        total_tt += offset * offset * included
        total_x += value
        total_tx += value * offset
        all_known &= ~included[:, None] | visible[indices]
    divisor = width[:, None, None]
    denominator = total_tt - total_t * total_t / width
    slope = np.divide(total_tx - total_x * total_t[:, None, None] / divisor,
                      denominator[:, None, None], out=np.zeros_like(total_x),
                      where=denominator[:, None, None] > 0)
    center = (total_x - slope * total_t[:, None, None]) / divisor
    max_error = np.zeros((n, slots))
    for offset in range(-half, half + 1):
        indices = np.clip(frames + offset, 0, n - 1)
        included = (frames + offset >= low) & (frames + offset < high)
        error = np.linalg.norm(relative_positions[indices] - center - offset * slope, axis=-1)
        max_error = np.maximum(max_error, np.where(included[:, None], error, 0.))
    use_window = width >= 3
    fitted = (adjacent & all_known & use_window[:, None]
              & (max_error <= config.max_motion_fit_error_m))
    fitted_speed = np.linalg.norm(slope * fps, axis=-1)
    return np.where(fitted, fitted_speed, np.inf), fitted, speed


def estimate_control(
    ball_xy_m: np.ndarray,
    player_xy_m: np.ndarray,
    teams: Sequence[str],
    *,
    fps: float,
    config: ControlConfig,
    reset: Optional[Sequence[bool]] = None,
    motion_breaks: Optional[Sequence[bool]] = None,
) -> ControlResult:
    """Return H/A evidence where proximity AND relative motion support control.

    Arrays must already be aligned to consecutive video frames. Ball positions
    have shape (frames, 2), player positions (frames, track_slots, 2), and teams
    (track_slots,) with H/A entries. Positions MUST be in metres, using the same
    pitch origin and axis directions for ball and players. Convert normalised
    tracking coordinates with (pitch_length, pitch_width) before calling.

    By default velocities are backward differences over adjacent finite samples.
    An opt-in local fit estimates relative motion over a short observed window,
    bounded by resets and supplied contact breaks. It requires a small fit error
    and never bridges missing observations. Windows can use future samples.
    Accepted adjacent-frame control has priority; fits only fill rejected frames,
    preserving genuine brief evidence instead of imposing a window-sized duration.
    The first frame, reappearances, and frames marked by reset cannot establish
    control without a valid preceding observation. There is no interpolation,
    minimum spell duration or boundary backdating.

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
    motion_breaks = np.zeros(len(ball), dtype=bool) if motion_breaks is None else np.asarray(motion_breaks)
    if motion_breaks.shape != (len(ball),) or motion_breaks.dtype.kind != "b":
        raise ValueError("motion_breaks must be a boolean sequence matching frames")

    n_frames, n_slots = players.shape[:2]
    labels = np.full(n_frames, "X", dtype="<U1")
    selected = np.full(n_frames, -1, dtype=int)
    distances = np.full(n_frames, np.nan)
    relative_speeds = np.full(n_frames, np.nan)
    eligible = np.zeros((n_frames, n_slots), dtype=bool)
    fit_used = np.zeros(n_frames, dtype=bool)
    if n_frames == 0 or n_slots == 0:
        return ControlResult(labels, selected, distances, relative_speeds, eligible, fit_used)

    ball_visible = np.isfinite(ball).all(axis=-1)
    player_visible = np.isfinite(players).all(axis=-1)
    # Sanitising nonfinite positions avoids invalid arithmetic; visibility masks
    # ensure the replacement coordinates can never count as observations.
    safe_ball = np.where(np.isfinite(ball), ball, 0.0)
    safe_players = np.where(np.isfinite(players), players, 0.0)
    distance = np.linalg.norm(safe_players - safe_ball[:, None, :], axis=-1)
    distance[~(player_visible & ball_visible[:, None])] = np.inf

    fitted_speed, fitted, raw_speed = _relative_motion(ball, players, fps, config, reset, motion_breaks)
    frames = np.arange(n_frames)
    modes = [(raw_speed, False)]
    if config.motion_window_frames > 1:
        modes.append((fitted_speed, True))
    for relative_speed, use_fit in modes:
        candidate_eligible = ((distance <= config.max_distance_m)
                              & (relative_speed <= config.max_relative_speed_mps))
        if use_fit:
            candidate_eligible &= fitted
        eligible |= candidate_eligible
        candidate_distances = np.where(candidate_eligible, distance, np.inf)
        candidate = np.argmin(candidate_distances, axis=1)
        best_distance = candidate_distances[frames, candidate]
        candidate_team = teams[candidate]
        opponent_distance = np.min(
            np.where(teams[None, :] != candidate_team[:, None], distance, np.inf), axis=1)
        accepted = ((selected < 0) & np.isfinite(best_distance)
                    & (opponent_distance > best_distance + config.opponent_margin_m))
        labels[accepted] = candidate_team[accepted]
        selected[accepted] = candidate[accepted]
        distances[accepted] = best_distance[accepted]
        relative_speeds[accepted] = relative_speed[frames[accepted], candidate[accepted]]
        fit_used[accepted] = use_fit
    return ControlResult(labels, selected, distances, relative_speeds, eligible, fit_used)
