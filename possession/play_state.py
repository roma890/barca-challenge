"""Ball-in-play hypotheses from soft context, motion and temporal continuity."""

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import numpy as np

from .contacts import _number, _tracking_inputs
from .stoppages import StoppageResult

if TYPE_CHECKING:
    from .dead_episodes import EpisodeResult


@dataclass(frozen=True)
class PlayStateConfig:
    window_frames: int
    min_observed_fraction: float
    min_players: int
    quiet_ball_speed_mps: float
    moving_ball_speed_mps: float
    quiet_player_speed_mps: float
    active_player_speed_mps: float
    tag_lookback_frames: int
    outside_pitch_margin_m: float
    transition_cost: float
    live_bias: float = 0.8
    set_piece_weight: float = 0.6
    quiet_ball_weight: float = 0.9
    quiet_players_weight: float = 0.8
    missing_ball_weight: float = 0.3
    moving_ball_weight: float = 2.0
    active_players_weight: float = 0.6
    live_phase_weight: float = 1.4
    outside_ball_weight: float = 2.5
    restart_weight: float = 4.0
    max_ball_speed_mps: float = 40.0
    max_player_speed_mps: float = 12.0

    def __post_init__(self):
        for name in ("window_frames", "min_players"):
            _number(name, getattr(self, name), positive=True, integer=True)
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least two")
        _number("tag_lookback_frames", self.tag_lookback_frames, integer=True)
        for name in ("min_observed_fraction", "quiet_ball_speed_mps", "moving_ball_speed_mps",
                     "quiet_player_speed_mps", "active_player_speed_mps", "max_ball_speed_mps",
                     "max_player_speed_mps"):
            _number(name, getattr(self, name), positive=True)
        if self.min_observed_fraction > 1:
            raise ValueError("min_observed_fraction must be <= 1")
        if self.moving_ball_speed_mps <= self.quiet_ball_speed_mps:
            raise ValueError("moving_ball_speed_mps must exceed quiet_ball_speed_mps")
        if self.active_player_speed_mps <= self.quiet_player_speed_mps:
            raise ValueError("active_player_speed_mps must exceed quiet_player_speed_mps")
        for name in ("outside_pitch_margin_m", "transition_cost", "live_bias", "set_piece_weight",
                     "quiet_ball_weight", "quiet_players_weight", "missing_ball_weight",
                     "moving_ball_weight", "active_players_weight", "live_phase_weight",
                     "outside_ball_weight", "restart_weight"):
            _number(name, getattr(self, name))


@dataclass(frozen=True)
class PlayStateResult:
    dead: np.ndarray
    dead_score: np.ndarray
    uncertain: np.ndarray
    diagnostics: dict
    live_conflict_proxy: Optional[np.ndarray] = None


_LIVE_PHASES = {"build up", "progression", "attacking transition", "defensive transition",
                "long ball", "creating chances"}


def _tag_context(events, n, lookback):
    set_piece = np.zeros(n)
    phase = np.zeros(n)
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("tagging events must follow the official loader schema")
        code = str(event.get("code") or "").strip().casefold()
        if code != "set pieces" and code not in _LIVE_PHASES:
            continue
        start, stop = event.get("start_frame"), event.get("end_frame")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in (start, stop)):
            raise ValueError("tag intervals must use integer video frames")
        if stop <= start:
            raise ValueError("tag end must follow its start")
        lo, hi = max(0, start), min(n, stop)
        if hi <= lo:
            continue
        if code == "set pieces":
            set_piece[lo:hi] = 1.
            # A weak, decaying prior before a restart phase can support a preceding
            # stoppage; neither this extension nor the interval is a dead label.
            before = max(0, lo - lookback)
            if before < lo:
                prior = 1. - (lo - np.arange(before, lo)) / float(lookback + 1)
                set_piece[before:lo] = np.maximum(set_piece[before:lo], prior)
        else:
            phase[lo:hi] = 1.
    return set_piece, phase


def _bounds(n, window, reset):
    low, high = np.empty(n, dtype=int), np.empty(n, dtype=int)
    boundaries = np.unique(np.r_[0, np.flatnonzero(reset), n])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        frames = np.arange(start, stop)
        low[start:stop] = np.maximum(start, frames - window // 2)
        high[start:stop] = np.minimum(stop, frames + (window - window // 2))
    return low, high, boundaries


def _sum_window(values, low, high):
    cumulative = np.r_[0., np.cumsum(values, dtype=float)]
    return cumulative[high] - cumulative[low]


def _motion_window(points, fps, fraction, max_speed, low, high):
    """Fit vector velocity to observed samples; gaps are not interpolated."""
    known = np.isfinite(points).all(axis=-1)
    frames = np.arange(len(points), dtype=float)
    count = _sum_window(known, low, high)
    divisor = np.maximum(1, count)
    sum_t = _sum_window(np.where(known, frames, 0.), low, high)
    sum_tt = _sum_window(np.where(known, frames * frames, 0.), low, high)
    denominator = sum_tt - sum_t * sum_t / divisor
    valid = (count >= np.maximum(2, np.ceil((high - low) * fraction))) & (denominator > 0)
    squared_speed = np.zeros(len(points))
    for axis in range(2):
        values = np.where(known, points[:, axis], 0.)
        sum_x = _sum_window(values, low, high)
        sum_tx = _sum_window(values * frames, low, high)
        slope = np.divide(sum_tx - sum_t * sum_x / divisor, denominator,
                          out=np.zeros(len(points)), where=valid)
        squared_speed += (slope * fps) ** 2
    speed = np.sqrt(squared_speed)
    valid &= speed <= max_speed
    return speed, valid, count


def _decode(score, fps, cost, boundaries):
    """Two-state Viterbi with penalties for switching, independently per reset run."""
    dead = np.zeros(len(score), dtype=bool)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        previous = np.empty((stop - start, 2), dtype=np.int8)
        live_value, dead_value = 0., -np.inf
        for frame in range(start, stop):
            switch_to_live, switch_to_dead = dead_value - cost, live_value - cost
            previous[frame - start, 0] = 0 if live_value >= switch_to_live else 1
            previous[frame - start, 1] = 1 if dead_value >= switch_to_dead else 0
            live_value, dead_value = (max(live_value, switch_to_live),
                                      max(dead_value, switch_to_dead) + score[frame] / fps)
        state = int(dead_value > live_value)
        for frame in range(stop - 1, start - 1, -1):
            dead[frame] = bool(state)
            state = int(previous[frame - start, state])
        # A neutral gap cannot supply a new boundary. Preserve the last decoded
        # state through it; leading unknown frames remain unasserted/live fallback.
        incumbent = False
        for frame in range(start, stop):
            if abs(score[frame]) < 1e-12:
                dead[frame] = incumbent
            else:
                incumbent = bool(dead[frame])
    return dead


def infer_play_state(
    ball_xy_m: np.ndarray,
    player_xy_m: np.ndarray,
    teams: Sequence[str],
    *,
    fps: float,
    pitch: Sequence[float],
    config: PlayStateConfig,
    events: Sequence[Mapping] = (),
    reset: Optional[Sequence[bool]] = None,
    restarts: Optional[StoppageResult] = None,
    episodes: Optional["EpisodeResult"] = None,
) -> PlayStateResult:
    """Combine imperfect evidence instead of requiring a complete restart chain.

    Set-piece context supports quiet-ball/quiet-player preparation. Tactical
    phases and sustained movement support live play. Missing ball observations
    strengthen an already-supported preparation hypothesis; they do not create D
    alone. Sustained out-of-pitch positions provide an independent physical cue.
    A temporal decoder balances these scores with transition costs and retains
    state through neutral gaps. All observations and decisions remain local.

    Scores/weights are explicit modeling priors, not trained probabilities.
    Correlated or incorrect tags, slow live play, aerial projection and tracking
    errors can still cause mistakes. False tags over indistinguishable actions
    cannot be perfectly resolved. This function reads no ground truth and contains
    no constants tied to particular match frames or public possession labels.
    """
    if not isinstance(config, PlayStateConfig):
        raise ValueError("config must be a PlayStateConfig")
    ball, players, _, reset, _ = _tracking_inputs(ball_xy_m, player_xy_m, teams, fps, reset, None)
    pitch = np.asarray(pitch, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("pitch must contain positive finite length and width")
    n = len(ball)
    if n == 0:
        return PlayStateResult(np.zeros(0, dtype=bool), np.zeros(0), np.zeros(0, dtype=bool), {})
    low, high, boundaries = _bounds(n, config.window_frames, reset)
    context, phase = _tag_context(events, n, config.tag_lookback_frames)
    speed, ball_known, count = _motion_window(ball, fps, config.min_observed_fraction,
                                            config.max_ball_speed_mps, low, high)
    quiet_ball = np.clip(1. - speed / config.quiet_ball_speed_mps, 0., 1.) * ball_known
    moving_ball = np.clip((speed - config.quiet_ball_speed_mps)
                          / (config.moving_ball_speed_mps - config.quiet_ball_speed_mps), 0., 1.) * ball_known
    player_count, quiet_count, active_count = np.zeros(n), np.zeros(n), np.zeros(n)
    for slot in range(players.shape[1]):
        player_speed, known, _ = _motion_window(
            players[:, slot], fps, config.min_observed_fraction, config.max_player_speed_mps, low, high)
        player_count += known
        quiet_count += known & (player_speed <= config.quiet_player_speed_mps)
        active_count += known & (player_speed >= config.active_player_speed_mps)
    players_known = player_count >= config.min_players
    quiet_players = np.divide(quiet_count, np.maximum(1, player_count)) * players_known
    active_players = np.divide(active_count, np.maximum(1, player_count)) * players_known
    physical = ball_known | players_known
    visible = np.isfinite(ball).all(axis=-1)
    margin = config.outside_pitch_margin_m
    outside = visible & ((ball[:, 0] < -margin) | (ball[:, 0] > pitch[0] + margin)
                         | (ball[:, 1] < -margin) | (ball[:, 1] > pitch[1] + margin))
    outside_fraction = _sum_window(outside, low, high) / np.maximum(1, count)
    outside_support = np.clip((outside_fraction - 0.5) * 2, 0., 1.) * ball_known
    missing_fraction = 1. - count / np.maximum(1, high - low)
    # Quiet ball and quiet players are correlated preparation cues. Either can
    # support the hypothesis; summing both would double-count a passive setup.
    preparation = np.maximum(config.quiet_ball_weight * quiet_ball,
                             config.quiet_players_weight * quiet_players)
    episode_support = np.zeros(n)
    if episodes is not None:
        episode_support = np.asarray(episodes.support)
        if (episode_support.shape != (n,) or not np.isfinite(episode_support).all()
                or ((episode_support < 0) | (episode_support > 1)).any()):
            raise ValueError("episode support must contain one finite 0..1 strength per frame")
    motion_discount = episode_support
    if episodes is not None and episodes.motion_discount is not None:
        motion_discount = np.asarray(episodes.motion_discount)
        if (motion_discount.shape != (n,) or not np.isfinite(motion_discount).all()
                or ((motion_discount < 0) | (motion_discount > 1)).any()):
            raise ValueError("episode motion discount must contain finite 0..1 values")
    # After a supported onset, retrieval/repositioning can move the ball and
    # players without resuming play. Discount generic live cues in proportion
    # to episode support; the evidence is still balanced by the temporal model.
    # Weak candidates add evidence but never suppress competing live cues.
    live_reliability = 1. - motion_discount
    score = (-config.live_bias * physical + config.set_piece_weight * context * physical
             + context * preparation
             + config.missing_ball_weight * context * quiet_players * missing_fraction
             - config.moving_ball_weight * moving_ball * live_reliability
             - config.active_players_weight * active_players * live_reliability
             - config.live_phase_weight * phase * live_reliability
             + config.outside_ball_weight * outside_support
             + config.restart_weight * episode_support)
    if restarts is not None:
        if len(restarts.status) != n:
            raise ValueError("restart evidence length must match tracking")
        score += config.restart_weight * (restarts.status == "D")
        score -= config.restart_weight * (restarts.status == "L")
    dead = _decode(score, fps, config.transition_cost, boundaries)
    # A separately confirmed restart is a positive live anchor. Caller-supplied
    # dead masks can still override this later in the pipeline.
    if restarts is not None:
        dead[restarts.status == "L"] = False
    uncertain = (~physical & (phase == 0)) | (np.abs(score) < config.live_bias)
    if episodes is not None:
        uncertain |= episodes.uncertain
    # An audit proxy, not a referee/ground-truth label: strong ball movement,
    # majority player activity and live-phase context agree before inference.
    live_proxy = (moving_ball >= 1.) & (active_players >= 0.5) & (phase > 0)
    diagnostics = {
        "method": "soft_evidence_viterbi",
        "dead_frames": int(dead.sum()),
        "dead_without_observed_ball_frames": int(np.sum(dead & ~visible)),
        "set_piece_context_frames": int(np.sum(context > 0)),
        "live_phase_context_frames": int(np.sum(phase > 0)),
        "quiet_ball_support_frames": int(np.sum(quiet_ball > 0)),
        "quiet_player_support_frames": int(np.sum(quiet_players > 0)),
        "outside_pitch_support_frames": int(np.sum(outside_support > 0)),
        "uncertain_dead_frames": int(np.sum(dead & uncertain)),
        "episode_supported_frames": int(np.sum(episode_support > 0)),
        "strong_live_proxy_frames": int(live_proxy.sum()),
        "dead_conflicting_live_proxy_frames": int(np.sum(dead & live_proxy)),
        "dead_runs": int(np.sum(np.diff(np.r_[False, dead].astype(np.int8)) == 1)),
    }
    return PlayStateResult(dead, score, uncertain, diagnostics, live_proxy)
