"""Bounded retrospective evidence for stoppage episodes, without ground truth."""

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .contacts import ContactConfig, _motion, _number, _tracking_inputs, detect_contacts
from .play_state import PlayStateConfig
from .stoppages import Restart, StoppageConfig, StoppageResult, _line, _location_ok, _supported, _windows


@dataclass(frozen=True)
class EpisodeConfig:
    lookback_frames: int
    goal_pair_max_frames: int
    min_outside_frames: int
    max_observation_gap_frames: int
    min_activity_drop: float
    ball_quiet_radius_m: float
    goal_corridor_half_width_m: float
    min_live_sequence_travel_m: float
    max_live_sequence_gap_frames: int
    max_live_sequence_fit_error_m: float
    allow_weak_restarts: bool = False
    weak_strength: float = 0.25
    weak_min_outside_frames: int = 1
    weak_min_slowdown_ratio: float = 0.3
    weak_min_speed_drop_mps: float = 0.5

    def __post_init__(self):
        for name in ("lookback_frames", "goal_pair_max_frames", "min_outside_frames",
                     "max_live_sequence_gap_frames"):
            _number(name, getattr(self, name), positive=True, integer=True)
        _number("max_observation_gap_frames", self.max_observation_gap_frames, integer=True)
        for name in ("min_activity_drop", "ball_quiet_radius_m", "goal_corridor_half_width_m",
                     "min_live_sequence_travel_m", "max_live_sequence_fit_error_m"):
            _number(name, getattr(self, name), positive=True)
        if self.min_activity_drop > 1:
            raise ValueError("min_activity_drop must be <= 1")
        if not isinstance(self.allow_weak_restarts, bool):
            raise ValueError("allow_weak_restarts must be boolean")
        for name in ("weak_strength", "weak_min_slowdown_ratio", "weak_min_speed_drop_mps"):
            _number(name, getattr(self, name), positive=True)
        if self.weak_strength >= 1 or self.weak_min_slowdown_ratio > 1:
            raise ValueError("weak strength must be < 1 and slowdown ratio <= 1")
        _number("weak_min_outside_frames", self.weak_min_outside_frames, positive=True, integer=True)


@dataclass(frozen=True)
class DeadEpisode:
    onset: int
    preparation_start: int
    release_frame: int
    onset_earliest: int
    onset_latest: int
    source: str
    strength: float  # Relative evidence weight, not a probability.


@dataclass(frozen=True)
class EpisodeResult:
    support: np.ndarray
    uncertain: np.ndarray
    live_guard: np.ndarray
    episodes: Tuple[DeadEpisode, ...]
    diagnostics: dict
    motion_discount: Optional[np.ndarray] = None
    weak_support: Optional[np.ndarray] = None
    weak_candidates: Tuple["WeakRestart", ...] = ()


@dataclass(frozen=True)
class WeakRestart:
    onset: int
    release_frame: int
    confirmation_frame: int
    kind: str
    onset_source: str
    strength: float


def _live_sequences(ball, players, teams, fps, contacts_config, config, reset, pitch):
    """Contradiction guard: two coherent transfers across three distinct tracks.

    Generic motion or a single contact is not enough: retrieval also has both.
    This guard is still an inference, not proof that the referee allowed play.
    """
    contacts = detect_contacts(ball, players, teams, fps=fps, config=contacts_config, reset=reset)
    guard = np.zeros(len(ball), dtype=bool)
    for first, second, third in zip(contacts, contacts[1:], contacts[2:]):
        group = (first, second, third)
        if any(event.track_index < 0 for event in group) or len({event.track_index for event in group}) < 3:
            continue
        if third.frame - first.frame > config.max_live_sequence_gap_frames:
            continue
        segment = ball[first.frame:third.frame + 1]
        if (not np.isfinite(segment).all() or reset[first.frame + 1:third.frame + 1].any()
                or (segment < 0).any() or (segment > pitch).any()):
            continue
        travel = float(np.linalg.norm(np.diff(segment, axis=0), axis=-1).sum())
        if travel < config.min_live_sequence_travel_m:
            continue
        coherent = True
        for sender, receiver in ((first, second), (second, third)):
            dt = (receiver.frame - sender.frame) / fps
            predicted = ball[sender.frame] + np.asarray(sender.outgoing_velocity_mps) * dt
            if np.linalg.norm(predicted - ball[receiver.frame]) > config.max_live_sequence_fit_error_m:
                coherent = False
                break
        if coherent:
            guard[first.frame:third.frame + 1] = True
    return guard


def _boundary_side(point, pitch, margin):
    sides = []
    if point[0] < -margin:
        sides.append("left")
    if point[0] > pitch[0] + margin:
        sides.append("right")
    if point[1] < -margin:
        sides.append("bottom")
    if point[1] > pitch[1] + margin:
        sides.append("top")
    return sides


def _exits(ball, fps, pitch, config, motion, reset):
    """Observed inside-to-outside transitions with corroborating outside samples."""
    known = np.flatnonzero(np.isfinite(ball).all(axis=-1))
    exits = []
    for previous, frame in zip(known[:-1], known[1:]):
        if frame - previous - 1 > config.max_observation_gap_frames or reset[previous + 1:frame + 1].any():
            continue
        if (ball[previous] < 0).any() or (ball[previous] > pitch).any():
            continue
        sides = _boundary_side(ball[frame], pitch, motion.outside_pitch_margin_m)
        if not sides:
            continue
        speed = np.linalg.norm(ball[frame] - ball[previous]) * fps / (frame - previous)
        if speed > motion.max_ball_speed_mps:
            continue
        end = min(len(ball), frame + config.min_outside_frames + config.max_observation_gap_frames)
        for side in sides:
            observed = np.flatnonzero(np.isfinite(ball[frame:end]).all(axis=-1)) + frame
            if len(observed) < config.min_outside_frames:
                continue
            selected = observed[:config.min_outside_frames]
            if (np.diff(selected) - 1 > config.max_observation_gap_frames).any() or reset[frame + 1:selected[-1] + 1].any():
                continue
            if all(side in _boundary_side(ball[index], pitch, motion.outside_pitch_margin_m) for index in selected):
                exits.append((int(frame), int(previous + 1), side))
    return exits


def _restart_point(event, ball, players):
    lo, hi = event.preparation_start, event.release_frame
    observed = np.isfinite(ball[lo:hi]).all(axis=-1)
    if observed.any():
        return np.median(ball[lo:hi][observed], axis=0)
    if event.track_index >= 0:
        positions = players[lo:hi, event.track_index]
        known = np.isfinite(positions).all(axis=-1)
        if known.any():
            return np.median(positions[known], axis=0)
    return None


def _compatible(side, kind, point, pitch):
    if kind == "throw-in":
        return side == ("bottom" if point[1] < pitch[1] / 2 else "top")
    if kind in ("corner kick", "goal kick"):
        return side == ("left" if point[0] < pitch[0] / 2 else "right")
    return False


def _decelerations(ball, players, fps, config, motion, reset, weak=False):
    """Candidate pause onsets: group deceleration plus quiet/hidden ball.

    Missing ball alone never creates a candidate. Median observed step activity
    does not confuse an out-and-back path with no movement.
    """
    velocities, valid = _motion(players, fps, motion.max_player_speed_mps, reset)
    speed = np.linalg.norm(velocities, axis=-1)
    count = valid.sum(axis=1)
    reliable = count >= motion.min_players
    active = np.sum(valid & (speed >= motion.active_player_speed_mps), axis=1) / np.maximum(1, count)
    quiet = np.sum(valid & (speed <= motion.quiet_player_speed_mps), axis=1) / np.maximum(1, count)
    group_speed = np.zeros(len(ball))
    if weak and reliable.any():
        group_speed[reliable] = np.nanmedian(np.where(valid[reliable], speed[reliable], np.nan), axis=1)
    speed_sum = np.r_[0., np.cumsum(group_speed)]
    reliable_sum = np.r_[0, np.cumsum(reliable)]
    window = motion.window_frames
    candidates = []
    for frame in range(window, len(ball) - window + 1):
        before, after = slice(frame - window, frame), slice(frame, frame + window)
        if reset[frame - window + 1:frame + window].any():
            continue
        if reliable[before].mean() < motion.min_observed_fraction or reliable[after].mean() < motion.min_observed_fraction:
            continue
        if weak:
            before_speed = (speed_sum[frame] - speed_sum[frame - window]) / max(1, reliable_sum[frame] - reliable_sum[frame - window])
            after_speed = (speed_sum[frame + window] - speed_sum[frame]) / max(1, reliable_sum[frame + window] - reliable_sum[frame])
            drop = (before_speed - after_speed) / max(before_speed, 1e-12)
            if drop < config.weak_min_slowdown_ratio or before_speed - after_speed < config.weak_min_speed_drop_mps:
                continue
        else:
            drop = float(active[before][reliable[before]].mean() - active[after][reliable[after]].mean())
            if drop < config.min_activity_drop or quiet[after][reliable[after]].mean() < 1. - config.min_activity_drop:
                continue
        points = ball[after]
        known = np.isfinite(points).all(axis=-1)
        missing = 1. - known.mean()
        settled = False
        if known.sum() >= 2:
            center = np.median(points[known], axis=0)
            settled = bool(np.quantile(np.linalg.norm(points[known] - center, axis=-1), 0.9)
                           <= config.ball_quiet_radius_m)
        if settled or missing >= 0.5:
            candidates.append((frame, drop))
    return candidates


def _goal_windows(events, n):
    windows = []
    for event in events:
        if str(event.get("code") or "").strip().casefold() != "goals":
            continue
        start, end = event.get("start_frame"), event.get("end_frame")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in (start, end)):
            raise ValueError("GOALS intervals must use integer video frames")
        if end <= start:
            raise ValueError("GOALS end must follow start")
        lo, hi = max(0, start), min(n, end)
        if lo < hi:
            windows.append((lo, hi))
    return sorted(set(windows))


def _weak_restart_candidates(ball, players, fps, pitch, events, config, motion, restart_config,
                             restarts, reset, guard):
    """Require tag + physical onset + fitted restart-like separation.

    These never enter the hard restart mask or seed a restart team. Partial
    observations are allowed; supporting scores are deliberately smaller and
    generic live evidence is not discounted for these candidates.
    """
    weak_exits = _exits(ball, fps, pitch, replace(config, min_outside_frames=config.weak_min_outside_frames), motion, reset)
    pauses = _decelerations(ball, players, fps, config, motion, reset, weak=True)
    candidates = []
    rejected_live = 0
    for lo, hi, kind in _windows(events, len(ball)):
        for release in range(max(1, lo), min(hi, len(ball) - 1)):
            if reset[release] or not np.isfinite(ball[release]).all():
                continue
            if any(abs(release - event.release_frame) < restart_config.release_confirmation_frames
                   for event in restarts.restarts):
                continue
            end = min(len(ball), release + restart_config.release_confirmation_frames)
            observations = _supported(ball, release, end, restart_config.release_observed_fraction,
                                       restart_config.max_gap_frames)
            if observations is None or observations[0] != release or reset[release + 1:observations[-1] + 1].any():
                continue
            confirm = int(observations[-1])
            origin, velocity, residual = _line(ball, observations, fps, release - 1)
            speed = np.linalg.norm(velocity)
            if (not restart_config.min_release_speed_mps <= speed <= restart_config.max_ball_speed_mps
                    or residual.max() > restart_config.max_release_fit_error_m
                    or not _location_ok(origin, kind, pitch, restart_config)):
                continue
            recent = max(lo, release - restart_config.max_gap_frames - 1)
            distances = np.full(players.shape[1], np.inf)
            reference = {}
            for slot in range(players.shape[1]):
                known = np.flatnonzero(np.isfinite(players[recent:release + 1, slot]).all(axis=-1)) + recent
                if len(known):
                    reference[slot] = players[known[-1], slot]
                    distances[slot] = np.linalg.norm(reference[slot] - origin)
            if not len(distances) or distances.min() > restart_config.max_preparer_distance_m:
                continue
            slot = int(np.argmin(distances))
            actor = players[confirm, slot] if np.isfinite(players[confirm, slot]).all() else reference[slot]
            final_distance = np.linalg.norm(ball[confirm] - actor)
            separation = final_distance - distances[slot]
            departure_distance = np.linalg.norm(ball[confirm] - origin)
            if (departure_distance < restart_config.min_release_displacement_m
                    or (final_distance <= restart_config.max_preparer_distance_m
                        and separation < restart_config.min_release_displacement_m)):
                continue
            reset_frames = np.flatnonzero(reset[:release + 1])
            previous = max((event.confirmation_frame + 1 for event in restarts.restarts
                            if event.confirmation_frame < release), default=0)
            lower = max(0, lo - config.lookback_frames, release - config.lookback_frames,
                        int(reset_frames[-1]) if len(reset_frames) else 0, previous)
            onsets = [(frame, "tentative_exit") for frame, _, side in weak_exits
                      if lower <= frame < release and _compatible(side, kind, origin, pitch)]
            onsets += [(frame, "group_slowdown") for frame, _ in pauses if lower <= frame < release]
            onsets.sort(reverse=True)
            accepted = None
            for onset, source in onsets:
                if guard[onset:release].any():
                    rejected_live += 1
                    continue
                accepted = WeakRestart(onset, release, confirm, kind, source, config.weak_strength)
                break
            if accepted is not None:
                candidates.append(accepted)
                break  # One earliest qualifying restart action per tag window.
    unique = {(event.onset, event.release_frame, event.kind): event for event in candidates}
    return tuple(sorted(unique.values(), key=lambda event: event.release_frame)), {
        "weak_boundary_candidates": len(weak_exits), "weak_pause_candidates": len(pauses),
        "weak_rejected_by_live_sequence": rejected_live,
    }


def infer_dead_episodes(
    ball_xy_m: np.ndarray, player_xy_m: np.ndarray, teams: Sequence[str], *,
    fps: float, pitch: Sequence[float], config: EpisodeConfig,
    motion_config: PlayStateConfig, contact_config: ContactConfig,
    restarts: StoppageResult, events: Sequence[Mapping] = (),
    reset: Optional[Sequence[bool]] = None,
    restart_config: Optional[StoppageConfig] = None,
) -> EpisodeResult:
    """Add bounded onset hypotheses to supported restart preparation.

    Boundary crossings must agree with the upcoming restart type/location. A
    group-deceleration/quiet-or-hidden-ball onset is weaker alternative evidence.
    Confirmed intervening releases, resets and coherent live-contact sequences
    block extension. GOALS requires an observed central goal-line crossing and
    a later supported kickoff-after-goal; tags alone never create an episode.

    Output is evidence strength, not mandatory D or a calibrated probability.
    Onset bounds preserve observation-gap/window uncertainty. No arbitrary gap
    midpoint, fixed occlusion-to-D rule, or tagged start-as-whistle rule is used.
    False tracking exits and false restart/goal hypotheses can still cause errors.
    """
    if not isinstance(config, EpisodeConfig) or not isinstance(motion_config, PlayStateConfig):
        raise ValueError("episode and motion settings must use their configuration types")
    ball, players, teams, reset, _ = _tracking_inputs(ball_xy_m, player_xy_m, teams, fps, reset, None)
    pitch = np.asarray(pitch, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("pitch must contain finite positive length and width")
    if len(restarts.status) != len(ball):
        raise ValueError("restart evidence length must match tracking")
    support = np.zeros(len(ball))
    uncertain = np.zeros(len(ball), dtype=bool)
    guard = _live_sequences(ball, players, teams, fps, contact_config, config, reset, pitch)
    exits = _exits(ball, fps, pitch, config, motion_config, reset)
    pauses = _decelerations(ball, players, fps, config, motion_config, reset)
    goals = _goal_windows(events, len(ball))
    episodes = []
    diagnostics = dict(method="bounded_restart_onsets_and_goal_pairing",
                       boundary_candidates=len(exits), pause_candidates=len(pauses),
                       goal_tag_windows=len(goals), backdated_restarts=0, paired_goals=0,
                       rejected_by_live_sequence=0, rejected_without_onset=0)
    ordered = sorted(restarts.restarts, key=lambda item: item.release_frame)
    for index, restart in enumerate(ordered):
        preparation = restart.preparation_start
        if preparation <= 0:
            continue
        previous_release = ordered[index - 1].confirmation_frame + 1 if index else 0
        resets = np.flatnonzero(reset[:preparation + 1])
        run_start = int(resets[-1]) if len(resets) else 0
        lower = max(run_start, previous_release, preparation - config.lookback_frames)
        point = _restart_point(restart, ball, players)
        candidates = []

        def add(onset, earliest, latest, source, strength):
            if guard[onset:preparation].any():
                diagnostics["rejected_by_live_sequence"] += 1
                return
            candidates.append((strength, onset, earliest, latest, source))

        if point is not None:
            for frame, earliest, side in exits:
                if lower <= frame < preparation and _compatible(side, restart.kind, point, pitch):
                    add(frame, max(lower, earliest), frame, "observed_exit", 1.)
        for frame, drop in pauses:
            if lower <= frame < preparation:
                # Stronger activity drops take precedence; no claim of exact
                # whistle timing follows from a rolling deceleration window.
                add(frame, max(lower, frame - motion_config.window_frames),
                    min(preparation, frame + motion_config.window_frames), "group_pause", 0.5 * drop)

        if restart.kind == "kick off (after goal)":
            goal_lower = max(run_start, previous_release, preparation - config.goal_pair_max_frames)
            for start, end in goals:
                if end <= goal_lower or start >= preparation:
                    continue
                search_start = max(goal_lower, start - config.max_observation_gap_frames)
                for frame, earliest, side in exits:
                    if (search_start <= frame < min(end, preparation) and side in ("left", "right")
                            and abs(ball[frame, 1] - pitch[1] / 2) <= config.goal_corridor_half_width_m):
                        add(frame, max(goal_lower, earliest), frame, "goal_then_kickoff", 1.)
        if not candidates:
            diagnostics["rejected_without_onset"] += 1
            continue
        # Prefer independent exit/goal support over a rolling pause; for equally
        # strong candidates select the latest compatible onset, not the longest D.
        strength, onset, earliest, latest, source = max(candidates, key=lambda item: (item[0], item[1], item[4] == "goal_then_kickoff"))
        support[onset:preparation] = np.maximum(support[onset:preparation], strength)
        uncertain[onset:preparation] = True
        episodes.append(DeadEpisode(onset, preparation, restart.release_frame,
                                    earliest, latest, source, strength))
        diagnostics["backdated_restarts"] += 1
        diagnostics["paired_goals"] += int(source == "goal_then_kickoff")
    support[restarts.status == "L"] = 0.
    uncertain[restarts.status == "L"] = False
    motion_discount = support.copy()
    weak_support = np.zeros(len(ball))
    weak_candidates = ()
    if config.allow_weak_restarts:
        if not isinstance(restart_config, StoppageConfig):
            raise ValueError("weak restarts require restart_config")
        weak_candidates, weak_diagnostics = _weak_restart_candidates(
            ball, players, fps, pitch, events, config, motion_config, restart_config, restarts, reset, guard)
        diagnostics.update(weak_diagnostics)
        for event in weak_candidates:
            weak_support[event.onset:event.release_frame] = np.maximum(
                weak_support[event.onset:event.release_frame], event.strength)
        weak_support[(restarts.status == "L") | guard] = 0.
        uncertain |= weak_support > 0
        support = np.maximum(support, weak_support)
    diagnostics["weak_restart_candidates"] = len(weak_candidates)
    diagnostics["weak_supported_frames"] = int(np.sum(weak_support > 0))
    diagnostics["supported_frames"] = int(np.sum(support > 0))
    diagnostics["live_sequence_guard_frames"] = int(guard.sum())
    return EpisodeResult(support, uncertain, guard, tuple(episodes), diagnostics,
                         motion_discount, weak_support, weak_candidates)
