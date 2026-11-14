"""Check proposed dead-ball exits against local physical action hypotheses.

This component does not identify an owner or create stoppages from missingness.
It uses future samples locally, bounds unsupported D extension, then abstains.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .contacts import _number, _tracking_inputs
from .control import ControlConfig
from .play_state import PlayStateConfig
from .stoppages import StoppageConfig, StoppageResult, _line, _supported


@dataclass(frozen=True)
class ResumptionConfig:
    max_hold_frames: int
    lookahead_frames: int
    min_observed_fraction: float
    max_observation_gap_frames: int
    min_ball_travel_m: float
    max_fit_error_m: float
    min_active_player_fraction: float
    bridge_observation_gaps: bool = False
    use_live_sequence_guard: bool = False

    def __post_init__(self):
        for name in ("max_hold_frames", "max_observation_gap_frames"):
            _number(name, getattr(self, name), integer=True)
        _number("lookahead_frames", self.lookahead_frames, positive=True, integer=True)
        if self.lookahead_frames < 2:
            raise ValueError("lookahead_frames must allow at least three samples")
        for name in ("min_observed_fraction", "min_ball_travel_m", "max_fit_error_m",
                     "min_active_player_fraction"):
            _number(name, getattr(self, name), positive=True)
        if self.min_observed_fraction > 1 or self.min_active_player_fraction > 1:
            raise ValueError("observation/activity fractions must be <= 1")
        for name in ("bridge_observation_gaps", "use_live_sequence_guard"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("{} must be boolean".format(name))


@dataclass(frozen=True)
class ResumptionResult:
    dead: np.ndarray
    unresolved: np.ndarray
    source: np.ndarray
    confirmation_frame: np.ndarray
    diagnostics: dict


def _action(ball, players, frame, stop, fps, pitch, reset, config, motion, control, restart,
            *, cover_frame=None):
    """Return first supported local action; never extrapolate into hidden time."""
    if not np.isfinite(ball[frame]).all():
        return None
    if frame > 0 and not reset[frame] and np.isfinite(ball[frame - 1]).all():
        arrival_speed = np.linalg.norm(ball[frame] - ball[frame - 1]) * fps
        if not motion.quiet_ball_speed_mps < arrival_speed <= motion.max_ball_speed_mps:
            return None
    for end in range(frame + 2, min(stop, frame + config.lookahead_frames + 1)):
        if reset[end]:
            break
        if reset[frame + 1:end + 1].any():
            break
        if cover_frame is not None and end <= cover_frame:
            continue
        indices = _supported(ball, frame, end + 1, config.min_observed_fraction,
                             config.max_observation_gap_frames)
        if indices is None or len(indices) < 3 or indices[0] != frame or indices[-1] != end:
            continue
        points = ball[indices]
        if ((points < 0).any() or (points > pitch).any()
                or np.linalg.norm(points[-1] - points[0]) < config.min_ball_travel_m):
            continue
        elapsed = np.diff(indices) / fps
        if np.linalg.norm(points[1] - points[0]) / elapsed[0] <= motion.quiet_ball_speed_mps:
            continue
        if (np.linalg.norm(np.diff(points, axis=0), axis=1) / elapsed > motion.max_ball_speed_mps).any():
            continue
        _, velocity, residual = _line(ball, indices, fps, frame)
        speed = np.linalg.norm(velocity)
        if residual.max() > config.max_fit_error_m or speed > motion.max_ball_speed_mps:
            continue
        active = valid = 0
        dribble = False
        for slot in range(players.shape[1]):
            player_indices = _supported(players[:, slot], frame, end + 1,
                                        config.min_observed_fraction, config.max_observation_gap_frames)
            if (player_indices is None or len(player_indices) < 3
                    or player_indices[0] != frame or player_indices[-1] != end):
                continue
            pp = players[player_indices, slot]
            if (np.linalg.norm(np.diff(pp, axis=0), axis=1)
                    / (np.diff(player_indices) / fps) > motion.max_player_speed_mps).any():
                continue
            _, player_velocity, player_error = _line(players[:, slot], player_indices, fps, frame)
            if player_error.max() > config.max_fit_error_m:
                continue
            valid += 1
            active += np.linalg.norm(player_velocity) >= motion.active_player_speed_mps
            start_distance = np.linalg.norm(players[frame, slot] - ball[frame])
            end_distance = np.linalg.norm(players[end, slot] - ball[end])
            if (speed >= restart.min_release_speed_mps
                    and start_distance <= restart.max_preparer_distance_m
                    and end_distance - start_distance >= config.min_ball_travel_m):
                return "ball_departure", end
            shared = np.intersect1d(indices, player_indices, assume_unique=True)
            if (len(shared) >= 3
                    and (np.linalg.norm(players[shared, slot] - ball[shared], axis=1)
                         <= control.max_distance_m).all()
                    and np.linalg.norm(velocity - player_velocity) <= control.max_relative_speed_mps):
                dribble = True
        group_active = valid >= motion.min_players and active / max(1, valid) >= config.min_active_player_fraction
        if group_active and dribble:
            return "dribble_and_group_activity", end
        if group_active and speed >= motion.moving_ball_speed_mps:
            return "ball_motion_and_group_activity", end
    return None


def _gap_action(ball, players, frame, floor, stop, fps, pitch, reset,
                config, motion, control, restart, hard_dead):
    """Support a hidden current frame with an action observed on both sides.

    No positions are filled. The existing observation-gap, fit, motion and
    player-support checks all still apply. A hidden launch with no preceding
    observation cannot qualify. Search stays inside the immediately preceding
    proposed stoppage and its exit interval, never across a strict dead anchor.
    """
    if np.isfinite(ball[frame]).all() or config.max_observation_gap_frames == 0:
        return None
    lower = max(floor, frame - config.lookahead_frames,
                frame - config.max_observation_gap_frames)
    # Prefer the nearest observed start; earlier samples can supply enough
    # evidence when the last observation alone cannot form a supported action.
    observed = np.flatnonzero(np.isfinite(ball[lower:frame]).all(axis=-1)) + lower
    for start in observed[::-1]:
        start = int(start)
        if reset[start + 1:frame + 1].any() or hard_dead[start:frame + 1].any():
            continue
        action = _action(ball, players, start, stop, fps, pitch, reset, config,
                         motion, control, restart, cover_frame=frame)
        if action is not None:
            kind, confirmation = action
            return "gap_" + kind, confirmation
    return None


def refine_dead_exits(
    ball_xy_m: np.ndarray, player_xy_m: np.ndarray, teams: Sequence[str], *,
    dead: Sequence[bool], fps: float, pitch: Sequence[float], config: ResumptionConfig,
    motion_config: PlayStateConfig, control_config: ControlConfig,
    restart_config: StoppageConfig, restarts: StoppageResult, reset=None,
    live_guard: Optional[Sequence[bool]] = None,
) -> ResumptionResult:
    """Hold unsupported proposed exits briefly, then return explicit uncertainty.

    Strict release anchors bypass the check. A supported observed action can
    establish live play without establishing a team. Tags and generic walking
    cannot qualify alone. Partial trajectories may qualify within bounded gaps;
    no action is backdated before its first actual observation. The hold limit
    is a modeling prior, not a claim that play resumed when time ran out.
    """
    if not isinstance(config, ResumptionConfig):
        raise ValueError("config must be a ResumptionConfig")
    for value, expected in ((motion_config, PlayStateConfig), (control_config, ControlConfig),
                            (restart_config, StoppageConfig), (restarts, StoppageResult)):
        if not isinstance(value, expected):
            raise ValueError("resumption inputs need matching configuration/result types")
    ball, players, _, reset, proposed = _tracking_inputs(ball_xy_m, player_xy_m, teams, fps, reset, dead)
    pitch = np.asarray(pitch, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("pitch must contain positive finite length and width")
    n = len(ball)
    if len(restarts.status) != n:
        raise ValueError("restart evidence length must match tracking")
    if live_guard is None:
        if config.use_live_sequence_guard:
            raise ValueError("enabled live-sequence guard requires a supplied mask")
        live_guard = np.zeros(n, dtype=bool)
    else:
        live_guard = np.asarray(live_guard)
        if live_guard.shape != (n,) or live_guard.dtype.kind != "b":
            raise ValueError("live_guard must be a boolean sequence matching tracking")
    guard = live_guard & config.use_live_sequence_guard
    strict_dead = restarts.dead
    refined, unresolved = proposed.copy(), np.zeros(n, dtype=bool)
    source = np.full(n, "original_model", dtype="<U40")
    confirmation = np.full(n, -1, dtype=int)
    # An action search cannot see through the next predicted stoppage or reset.
    boundaries = np.r_[np.flatnonzero(proposed | reset), n]
    stop_at = boundaries[np.searchsorted(boundaries, np.arange(n), side="right")]
    waiting = None
    evidence_floor = 0
    checked = accepted = held_exits = unresolved_exits = 0
    for frame in range(n):
        if reset[frame]:
            waiting = None
            evidence_floor = frame
        if proposed[frame] and (frame == 0 or not proposed[frame - 1]):
            evidence_floor = frame
        if restarts.status[frame] == "L":
            refined[frame] = False
            source[frame] = "confirmed_restart"
            waiting = None
            continue
        if guard[frame] and not strict_dead[frame]:
            refined[frame] = False
            source[frame] = "live_transfer_sequence"
            waiting = None
            continue
        if proposed[frame]:
            waiting = None
            continue
        if frame > 0 and proposed[frame - 1] and not reset[frame]:
            waiting = frame
            checked += 1
        if waiting is None:
            continue
        action = _action(ball, players, frame, int(stop_at[frame]), fps, pitch, reset,
                         config, motion_config, control_config, restart_config)
        if action is None and config.bridge_observation_gaps:
            action = _gap_action(ball, players, frame, evidence_floor, int(stop_at[frame]),
                                 fps, pitch, reset, config, motion_config, control_config,
                                 restart_config, strict_dead)
        if action is not None:
            source[frame], confirmation[frame] = action
            waiting = None
            accepted += 1
        elif frame - waiting < config.max_hold_frames:
            refined[frame] = True
            source[frame] = "held_without_resume_support"
            held_exits += int(frame == waiting)
        else:
            unresolved[frame] = True
            source[frame] = "unresolved_after_hold_limit"
            unresolved_exits += int(frame - waiting == config.max_hold_frames)
    diagnostics = {
        "method": "bounded_physical_resumption_check", "proposed_exits_checked": checked,
        "exits_with_supported_action": accepted, "initially_held_exits": held_exits,
        "exits_reaching_uncertainty_limit": unresolved_exits,
        "held_dead_frames": int(np.sum(refined & ~proposed)),
        "unresolved_frames": int(unresolved.sum()),
        "confirmed_live_anchor_frames": int(np.sum(restarts.status == "L")),
        "transfer_guard_frames": int(guard.sum()),
        "transfer_guard_overridden_soft_dead_frames": int(np.sum(guard & proposed & ~strict_dead)),
        "transfer_guard_strict_dead_conflicts": int(np.sum(guard & strict_dead)),
        "action_counts": {kind: int(np.sum(source == kind)) for kind in
                          ("ball_departure", "dribble_and_group_activity", "ball_motion_and_group_activity",
                           "gap_ball_departure", "gap_dribble_and_group_activity",
                           "gap_ball_motion_and_group_activity")},
        "ownership_seeded": False,
    }
    return ResumptionResult(refined, unresolved, source, confirmation, diagnostics)
