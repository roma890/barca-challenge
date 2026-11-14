"""Conservative offline restart hypotheses; stationary does not mean dead."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .contacts import _number, _tracking_inputs


@dataclass(frozen=True)
class StoppageConfig:
    min_preparation_frames: int
    max_preparation_frames: int
    max_stationary_speed_mps: float
    max_preparation_drift_m: float
    max_preparer_distance_m: float
    min_release_speed_mps: float
    min_release_displacement_m: float
    release_confirmation_frames: int
    min_release_straightness: float
    max_ball_speed_mps: float
    max_player_speed_mps: float
    attribution_margin_m: float
    location_tolerance_m: float
    goal_kick_depth_m: float
    goal_kick_half_width_m: float
    max_gap_frames: int = 0
    preparation_observed_fraction: float = 0.6
    preparation_inlier_fraction: float = 0.8
    release_observed_fraction: float = 0.6
    max_release_fit_error_m: float = 0.3
    allow_hidden_ball_preparation: bool = False

    def __post_init__(self):
        for name in ("min_preparation_frames", "max_preparation_frames", "release_confirmation_frames"):
            _number(name, getattr(self, name), positive=True, integer=True)
        if self.min_preparation_frames < 2 or self.release_confirmation_frames < 2:
            raise ValueError("preparation and release confirmation need at least two frames")
        if self.max_preparation_frames < self.min_preparation_frames:
            raise ValueError("max_preparation_frames must be >= min_preparation_frames")
        for name in ("max_stationary_speed_mps", "max_preparation_drift_m",
                     "attribution_margin_m", "location_tolerance_m"):
            _number(name, getattr(self, name))
        for name in ("max_preparer_distance_m", "min_release_speed_mps",
                     "min_release_displacement_m", "min_release_straightness",
                     "max_ball_speed_mps", "max_player_speed_mps",
                     "goal_kick_depth_m", "goal_kick_half_width_m"):
            _number(name, getattr(self, name), positive=True)
        if self.min_release_straightness > 1:
            raise ValueError("min_release_straightness must be <= 1")
        if not self.max_stationary_speed_mps < self.min_release_speed_mps <= self.max_ball_speed_mps:
            raise ValueError("release speed must exceed stationary speed and not exceed ball speed limit")
        _number("max_gap_frames", self.max_gap_frames, integer=True)
        _number("max_release_fit_error_m", self.max_release_fit_error_m, positive=True)
        for name in ("preparation_observed_fraction", "preparation_inlier_fraction", "release_observed_fraction"):
            _number(name, getattr(self, name), positive=True)
            if getattr(self, name) > 1:
                raise ValueError("observation/inlier fractions must be <= 1")
        if not isinstance(self.allow_hidden_ball_preparation, bool):
            raise ValueError("allow_hidden_ball_preparation must be boolean")


@dataclass(frozen=True)
class Restart:
    preparation_start: int
    release_frame: int
    confirmation_frame: int
    kind: str
    team: str
    track_index: int
    preparation_source: str = "observed_ball"


@dataclass(frozen=True)
class StoppageResult:
    status: np.ndarray  # U unknown, D supported dead hypothesis, L supported release
    reason: np.ndarray
    restart_team: np.ndarray
    restarts: Tuple[Restart, ...]
    diagnostics: Mapping[str, int] = field(default_factory=dict)

    @property
    def dead(self):
        return self.status == "D"

    @property
    def release(self):
        mask = np.zeros(len(self.status), dtype=bool)
        for event in self.restarts:
            mask[event.release_frame] = True
        return mask


_KINDS = {"kick off (start)", "kick off (after goal)", "corner kick",
          "goal kick", "free kick", "throw-in"}


def _windows(events, n_frames):
    """Consume the public starter.load_smart_tagging.load() dictionary schema."""
    windows = set()
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("tagging events must use the official loader dictionary schema")
        if str(event.get("code") or "").strip().casefold() != "set pieces":
            continue
        tags = event.get("tags") or {}
        if not isinstance(tags, Mapping):
            raise ValueError("set-piece tags must be a dictionary")
        kind = str(tags.get("Type") or "").strip().casefold()
        if kind not in _KINDS:
            continue
        start, end = event.get("start_frame"), event.get("end_frame")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in (start, end)):
            raise ValueError("set-piece bounds must be integer video frames")
        if end <= start:
            raise ValueError("set-piece end must follow its start")
        start, end = max(0, start), min(n_frames, end)
        if start < end:
            windows.add((start, end, kind))
    return sorted(windows)


def _location_ok(point, kind, pitch, config):
    x, y = point
    length, width = pitch
    tolerance = config.location_tolerance_m
    if not (-tolerance <= x <= length + tolerance and -tolerance <= y <= width + tolerance):
        return False
    if kind.startswith("kick off"):
        return np.linalg.norm(point - pitch / 2) <= tolerance
    if kind == "corner kick":
        return min(np.hypot(x - cx, y - cy) for cx in (0., length) for cy in (0., width)) <= tolerance
    if kind == "throw-in":
        return min(abs(y), abs(y - width)) <= tolerance
    if kind == "goal kick":
        return (min(abs(x), abs(x - length)) <= config.goal_kick_depth_m
                and abs(y - width / 2) <= config.goal_kick_half_width_m)
    return True  # Free kicks have no specific restart landmark.


def _supported(points, start, stop, fraction, max_gap):
    """Observed indices with temporal coverage; no positions are interpolated."""
    indices = np.flatnonzero(np.isfinite(points[start:stop]).all(axis=-1)) + start
    if len(indices) < max(2, int(np.ceil((stop - start) * fraction))):
        return None
    gaps = np.concatenate(([indices[0] - start], np.diff(indices) - 1, [stop - 1 - indices[-1]]))
    return indices if gaps.max() <= max_gap else None


def _line(points, indices, fps, origin_frame):
    times = (indices - origin_frame) / fps
    centered = times - times.mean()
    velocity = centered @ (points[indices] - points[indices].mean(axis=0)) / np.dot(centered, centered)
    origin = points[indices].mean(axis=0) - times.mean() * velocity
    residual = np.linalg.norm(points[indices] - (origin + times[:, None] * velocity), axis=-1)
    return origin, velocity, residual


def _settled(points, indices, fps, config):
    center = np.median(points[indices], axis=0)
    inliers = np.linalg.norm(points[indices] - center, axis=-1) <= config.max_preparation_drift_m
    if (not inliers[-1]
            or inliers.sum() < max(2, int(np.ceil(len(indices) * config.preparation_inlier_fraction)))):
        return None
    selected = indices[inliers]
    _, velocity, _ = _line(points, selected, fps, int(selected[0]))
    if np.linalg.norm(velocity) > config.max_stationary_speed_mps:
        return None
    return np.median(points[selected], axis=0)


def _extend_preparation(points, start, lower, center, radius, reset, max_gap):
    missing = 0
    while start > lower and not reset[start]:
        point = points[start - 1]
        if not np.isfinite(point).all():
            missing += 1
            if missing > max_gap:
                break
        elif np.linalg.norm(point - center) > radius:
            break
        else:
            missing = 0
        start -= 1
    # Do not claim unsupported leading missing frames as the onset of dead play.
    while start < len(points) and not np.isfinite(points[start]).all():
        start += 1
    return start


def _preparer(players, core, earliest, release, confirm, anchor, fps, reset, config, hidden):
    candidates = []
    for slot in range(players.shape[1]):
        positions = players[:, slot]
        indices = _supported(positions, core, release, config.preparation_observed_fraction,
                             config.max_gap_frames)
        if indices is None:
            continue
        distances = np.linalg.norm(positions[indices] - anchor, axis=-1)
        if np.mean(distances <= config.max_preparer_distance_m) < config.preparation_inlier_fraction:
            continue
        if hidden:
            center = _settled(positions, indices, fps, config)
            if center is None:
                continue
            radius = config.max_preparation_drift_m
        else:
            center, radius = anchor, config.max_preparer_distance_m
        start = _extend_preparation(positions, core, earliest, center, radius, reset, config.max_gap_frames)
        observed = _supported(positions, start, confirm + 1,
                              min(config.preparation_observed_fraction, config.release_observed_fraction),
                              config.max_gap_frames)
        if observed is None or observed[-1] != confirm:
            continue
        velocities = np.diff(positions[observed], axis=0) * fps / np.diff(observed)[:, None]
        if (np.linalg.norm(velocities, axis=-1) > config.max_player_speed_mps).any():
            continue
        candidates.append((float(np.median(distances)), slot, start))
    return min(candidates) if candidates else None


def detect_stoppages(
    ball_xy_m: np.ndarray,
    player_xy_m: np.ndarray,
    teams: Sequence[str],
    *,
    fps: float,
    pitch: Sequence[float],
    config: StoppageConfig,
    events: Sequence[Mapping] = (),
    reset: Optional[Sequence[bool]] = None,
) -> StoppageResult:
    """Infer restart preparation from window-level evidence, not instant speeds.

    D needs automatic set-piece context, compatible geometry, a supported quiet
    preparation window, nearby player evidence, and a fitted outgoing trajectory
    that separates from the preparer. Position clustering and a fitted motion
    trend tolerate jitter; bounded gaps are checked by observation coverage and
    duration, never filled with invented positions.

    At landmark restarts an optional second route supports hidden-ball preparation
    using a settled player and an outgoing trajectory beginning near that player.
    Any available preparation ball observations must agree. This route is disabled
    for free kicks, which have no distinct landmark. Missingness alone proves
    nothing. False tags on identical live actions remain a failure mode.

    Release fitting uses future observations; a short, trajectory-supported
    backtrack can locate release before the first clearly separated observation.
    No boundary is placed arbitrarily in a long gap. Unknown fouls, whole earlier
    dead spells, goals and halftime remain unresolved. Team comes from geometry,
    never a tagging Team string. Diagnostics are aggregate window/frame attempts,
    not counts of real restarts or calibrated confidence.
    """
    if not isinstance(config, StoppageConfig):
        raise ValueError("config must be a StoppageConfig")
    ball, players, teams, reset, _ = _tracking_inputs(ball_xy_m, player_xy_m, teams, fps, reset, None)
    pitch = np.asarray(pitch, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("pitch must contain finite positive length and width")
    n, slots = players.shape[:2]
    windows = _windows(events, n)
    status = np.full(n, "U", dtype="<U1")
    reason = np.full(n, "insufficient_evidence", dtype="<U24")
    restart_team = np.full(n, "X", dtype="<U1")
    rejection_keys = (
        "duplicate_release", "invalid_ball_motion", "below_release_speed", "incompatible_location",
        "insufficient_stationary_preparation", "invalid_release_observations",
        "insufficient_release_displacement", "inconsistent_release_direction",
        "release_origin_mismatch", "no_departure", "no_near_preparer", "ball_stays_with_preparer",
    )
    diagnostics = {
        "set_piece_windows": len(windows), "tested_release_frames": 0, "accepted_restarts": 0,
        "accepted_observed_ball": 0, "accepted_hidden_ball": 0,
        "longest_ball_preparation_frames": 0, "longest_near_preparer_frames": 0,
        "preparation_blocked_by_missing_or_reset": 0, "preparation_blocked_by_not_stationary": 0,
    }
    diagnostics.update({"rejected_" + key: 0 for key in rejection_keys})
    if slots == 0 or not windows:
        return StoppageResult(status, reason, restart_team, (), diagnostics)

    def reject(key):
        diagnostics["rejected_" + key] += 1

    visible = np.isfinite(ball).all(axis=-1)
    found = {}
    for lo, hi, kind in windows:
        for departure in range(max(1, lo), min(n - config.release_confirmation_frames + 1, hi)):
            diagnostics["tested_release_frames"] += 1
            confirm = departure + config.release_confirmation_frames - 1
            if any(event.release_frame <= departure <= event.confirmation_frame for event in found.values()):
                reject("duplicate_release")
                continue
            if not visible[departure] or reset[departure]:
                reject("invalid_ball_motion")
                continue
            core = max(lo, departure - config.min_preparation_frames)
            lower = max(lo, departure - config.max_preparation_frames)
            if core >= departure:
                reject("insufficient_stationary_preparation")
                continue
            if reset[core + 1:confirm + 1].any():
                diagnostics["preparation_blocked_by_missing_or_reset"] += 1
                reject("insufficient_stationary_preparation")
                continue
            preparation = _supported(ball, core, departure, config.preparation_observed_fraction,
                                     config.max_gap_frames)
            anchor = _settled(ball, preparation, fps, config) if preparation is not None else None
            start = core
            if anchor is not None:
                start = _extend_preparation(ball, int(preparation[0]), lower, anchor,
                                            config.max_preparation_drift_m, reset, config.max_gap_frames)
                diagnostics["longest_ball_preparation_frames"] = max(
                    diagnostics["longest_ball_preparation_frames"], departure - start)
                candidate_reason = reason[start:departure]
                candidate_reason[status[start:departure] == "U"] = "preparation_candidate"
            else:
                key = "missing_or_reset" if preparation is None else "not_stationary"
                diagnostics["preparation_blocked_by_" + key] += 1
            observed_fraction = float(visible[core:departure].mean())
            hidden = (anchor is None and config.allow_hidden_ball_preparation and kind != "free kick"
                      and observed_fraction < config.preparation_observed_fraction
                      and not visible[departure - 1])
            if departure - start < config.min_preparation_frames or (anchor is None and not hidden):
                reject("insufficient_stationary_preparation")
                continue

            observations = _supported(ball, departure, confirm + 1, config.release_observed_fraction,
                                       config.max_gap_frames)
            if observations is None or observations[0] != departure or observations[-1] != confirm:
                reject("invalid_release_observations")
                continue
            origin, velocity, residual = _line(ball, observations, fps, departure - 1)
            speed = np.linalg.norm(velocity)
            step_speed = np.linalg.norm(np.diff(ball[observations], axis=0), axis=-1) * fps / np.diff(observations)
            if (step_speed > config.max_ball_speed_mps).any() or np.quantile(residual, 0.9) > config.max_release_fit_error_m:
                reject("invalid_ball_motion")
                continue
            if speed < config.min_release_speed_mps:
                reject("below_release_speed")
                continue
            if hidden:
                anchor = origin
                start = lower
                # A settled preparer may support a longer tagged preparation
                # while the ball is hidden. Known ball positions elsewhere veto
                # extension; never label the entire tag on missingness alone.
                older = np.flatnonzero(visible[lower:core]) + lower
                if len(older):
                    contradictory = older[np.linalg.norm(ball[older] - anchor, axis=-1)
                                          > config.max_preparation_drift_m]
                    if len(contradictory):
                        start = int(contradictory[-1]) + 1
                # Observed preparation samples can veto the hidden-ball hypothesis.
                known = np.flatnonzero(visible[core:departure]) + core
                if len(known):
                    compatible = np.linalg.norm(ball[known] - anchor, axis=-1) <= config.max_preparation_drift_m
                    if compatible.mean() < config.preparation_inlier_fraction:
                        reject("insufficient_stationary_preparation")
                        continue
                    if len(known) >= 2 and _settled(ball, known, fps, config) is None:
                        reject("insufficient_stationary_preparation")
                        continue
            if not _location_ok(anchor, kind, pitch, config):
                reject("incompatible_location")
                continue
            if np.linalg.norm(origin - anchor) > config.max_release_fit_error_m + config.max_preparation_drift_m:
                reject("release_origin_mismatch")
                continue
            if np.linalg.norm(ball[departure] - anchor) <= config.max_preparation_drift_m:
                reject("no_departure")
                continue
            displacement = float(np.linalg.norm(ball[confirm] - anchor))
            if displacement < config.min_release_displacement_m:
                reject("insufficient_release_displacement")
                continue
            # Coherence over the fitted trajectory tolerates small position jitter
            # that would inflate an adjacent-frame polyline's path length.
            coherence = max(0., 1. - float(np.sqrt(np.mean(residual ** 2))) / displacement)
            if coherence < config.min_release_straightness:
                reject("inconsistent_release_direction")
                continue

            preparer = _preparer(players, core, start, departure, confirm, anchor, fps, reset, config, hidden)
            if preparer is None:
                reject("no_near_preparer")
                continue
            initial_distance, slot, player_start = preparer
            start = max(start, player_start)
            diagnostics["longest_near_preparer_frames"] = max(
                diagnostics["longest_near_preparer_frames"], departure - start)
            if departure - start < config.min_preparation_frames:
                reject("insufficient_stationary_preparation")
                continue
            final_distance = np.linalg.norm(ball[confirm] - players[confirm, slot])
            if (final_distance <= config.max_preparer_distance_m
                    and final_distance - initial_distance < config.min_release_displacement_m):
                reject("ball_stays_with_preparer")
                continue

            # The observed-ball anchor can support a limited earlier crossing
            # of the fitted outgoing line, never arbitrary placement in a gap.
            backtrack = int(round(float(np.dot(origin - anchor, velocity)) / speed ** 2 * fps))
            backtrack = min(max(0, backtrack), config.release_confirmation_frames - 1)
            release = max(start + 1, departure - backtrack)
            if release in found:
                reject("duplicate_release")
                continue
            opponent_distances = []
            for other in np.flatnonzero(teams != teams[slot]):
                points = players[core:departure, other]
                known = np.isfinite(points).all(axis=-1)
                if known.any():
                    opponent_distances.append(float(np.median(np.linalg.norm(points[known] - anchor, axis=-1))))
            attributed = not opponent_distances or min(opponent_distances) > initial_distance + config.attribution_margin_m
            team = str(teams[slot]) if attributed else "X"
            source = "hidden_ball" if hidden else "observed_ball"
            found[release] = Restart(start, release, confirm, kind, team, slot if attributed else -1, source)
            diagnostics["accepted_restarts"] += 1
            diagnostics["accepted_" + source] += 1
            status[start:release] = "D"
            reason[start:release] = "restart_preparation"
            status[release:confirm + 1] = "L"
            reason[release:confirm + 1] = "confirmed_release"
            restart_team[release] = team
    return StoppageResult(status, reason, restart_team,
                          tuple(found[frame] for frame in sorted(found)), diagnostics)
