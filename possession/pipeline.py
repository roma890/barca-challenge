"""Wire local tracking through inference and the official submission contract."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Mapping, Optional, Sequence

import numpy as np

from evaluation import possession_metrics as pm
from evaluation.validate_submission import describe
from starter.baseline import write_submission

from .contacts import ContactConfig
from .control import ControlConfig
from .interactions import InteractionConfig, InteractionResult, infer_interactions
from .state import StateResult, decode_state
from .play_state import PlayStateConfig, PlayStateResult, infer_play_state
from .dead_episodes import EpisodeConfig, EpisodeResult, infer_dead_episodes
from .stoppages import StoppageConfig, StoppageResult, detect_stoppages
from .tracking import Tracking
from .resumption import ResumptionConfig, ResumptionResult, refine_dead_exits


@dataclass(frozen=True)
class PipelineConfig:
    control: ControlConfig
    contacts: ContactConfig
    interactions: InteractionConfig
    max_carry_frames: Optional[int]
    stoppages: Optional[StoppageConfig] = None
    play_state: Optional[PlayStateConfig] = None
    episodes: Optional[EpisodeConfig] = None
    resumption: Optional[ResumptionConfig] = None

    def __post_init__(self):
        for value, expected in ((self.control, ControlConfig), (self.contacts, ContactConfig),
                                (self.interactions, InteractionConfig)):
            if not isinstance(value, expected):
                raise ValueError("pipeline sections must use the matching configuration type")
        if self.max_carry_frames is not None and (isinstance(self.max_carry_frames, (bool, np.bool_))
                or not isinstance(self.max_carry_frames, Integral) or self.max_carry_frames < 0):
            raise ValueError("max_carry_frames must be None or a nonnegative integer")
        if self.stoppages is not None and not isinstance(self.stoppages, StoppageConfig):
            raise ValueError("stoppages must be a StoppageConfig or None")
        if self.play_state is not None and not isinstance(self.play_state, PlayStateConfig):
            raise ValueError("play_state must be a PlayStateConfig or None")
        if self.episodes is not None:
            if not isinstance(self.episodes, EpisodeConfig):
                raise ValueError("episodes must be an EpisodeConfig or None")
            if self.stoppages is None or self.play_state is None:
                raise ValueError("episodes require both stoppages and play_state settings")
        if self.resumption is not None:
            if not isinstance(self.resumption, ResumptionConfig):
                raise ValueError("resumption must be a ResumptionConfig or None")
            if self.stoppages is None or self.play_state is None:
                raise ValueError("resumption requires stoppages and play_state settings")
            if self.resumption.use_live_sequence_guard and self.episodes is None:
                raise ValueError("live-sequence guard requires episodes settings")


def load_config(path) -> PipelineConfig:
    """Load all settings explicitly; reject missing or misspelled sections/keys."""
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    required = {"control", "contacts", "interactions", "max_carry_frames"}
    if (not isinstance(values, dict) or not required.issubset(values)
            or set(values) - required - {"stoppages", "play_state", "episodes", "resumption"}):
        raise ValueError("config needs control, contacts, interactions, max_carry_frames; optional stoppages/play_state/episodes")
    try:
        return PipelineConfig(
            ControlConfig(**values["control"]), ContactConfig(**values["contacts"]),
            InteractionConfig(**values["interactions"]), values["max_carry_frames"],
            StoppageConfig(**values["stoppages"]) if values.get("stoppages") is not None else None,
            PlayStateConfig(**values["play_state"]) if values.get("play_state") is not None else None,
            EpisodeConfig(**values["episodes"]) if values.get("episodes") is not None else None,
            ResumptionConfig(**values["resumption"]) if values.get("resumption") is not None else None)
    except TypeError:
        raise ValueError("configuration section has missing, unexpected or invalid fields") from None


@dataclass(frozen=True)
class Prediction:
    state: StateResult
    evidence: InteractionResult
    summary: dict
    stoppages: Optional[StoppageResult] = None
    play_state: Optional[PlayStateResult] = None
    episodes: Optional[EpisodeResult] = None
    resumption: Optional[ResumptionResult] = None


def predict(
    tracking: Tracking,
    config: PipelineConfig,
    *,
    dead: Optional[Sequence[bool]] = None,
    reset: Optional[Sequence[bool]] = None,
    events: Optional[Sequence[Mapping]] = None,
) -> Prediction:
    """Run local inference with soft ball-in-play context and restart anchors.

    With stoppage settings enabled, automatic tagging events can support bounded
    preparation/release hypotheses. The separate play-state model combines soft
    physical/tagging evidence over time; tracking alone can support out-of-play
    hypotheses. Missing tagging is not proof of live play. Caller-supplied dead
    evidence is combined by OR. Release
    boundaries reset memory and may establish the preparer's team at first motion.
    Masks/events must come from algorithms or synthetic tests, not hand annotation
    of the competition match. No labels/ground truth are read or scored here.
    """
    n_frames = len(tracking.ball_xy_m)
    if n_frames == 0:
        raise ValueError("tracking must contain at least one frame")
    if not isinstance(config, PipelineConfig):
        raise ValueError("config must be a PipelineConfig")
    dead_supplied = dead is not None
    dead = np.zeros(n_frames, dtype=bool) if dead is None else np.asarray(dead)
    if dead.shape != (n_frames,) or dead.dtype.kind != "b":
        raise ValueError("dead must be a boolean sequence matching frames")
    reset = np.zeros(n_frames, dtype=bool) if reset is None else np.asarray(reset)
    if reset.shape != (n_frames,) or reset.dtype.kind != "b":
        raise ValueError("reset must be a boolean sequence matching frames")
    stoppages = None
    if config.stoppages is not None:
        if tracking.pitch is None:
            raise ValueError("tracking.pitch is required for stoppage geometry")
        stoppages = detect_stoppages(
            tracking.ball_xy_m, tracking.player_xy_m, tracking.teams, fps=tracking.fps,
            pitch=tracking.pitch, config=config.stoppages,
            events=() if events is None else events, reset=reset)
        dead = dead | stoppages.dead
        reset = reset | stoppages.release
    episodes = None
    if config.episodes is not None:
        episodes = infer_dead_episodes(
            tracking.ball_xy_m, tracking.player_xy_m, tracking.teams, fps=tracking.fps,
            pitch=tracking.pitch, config=config.episodes, motion_config=config.play_state,
            contact_config=config.contacts, restarts=stoppages,
            events=() if events is None else events, reset=reset, restart_config=config.stoppages)
    play_state = None
    resumption = None
    if config.play_state is not None:
        if tracking.pitch is None:
            raise ValueError("tracking.pitch is required for ball-in-play inference")
        play_state = infer_play_state(
            tracking.ball_xy_m, tracking.player_xy_m, tracking.teams, fps=tracking.fps,
            pitch=tracking.pitch, config=config.play_state,
            events=() if events is None else events, reset=reset, restarts=stoppages, episodes=episodes)
        soft_dead = play_state.dead
        if config.resumption is not None:
            resumption = refine_dead_exits(
                tracking.ball_xy_m, tracking.player_xy_m, tracking.teams,
                dead=soft_dead | stoppages.dead, fps=tracking.fps, pitch=tracking.pitch,
                config=config.resumption, motion_config=config.play_state,
                control_config=config.control, restart_config=config.stoppages,
                restarts=stoppages, reset=reset,
                live_guard=episodes.live_guard if episodes is not None else None)
            soft_dead = resumption.dead
        dead = dead | soft_dead
        # A transition out of a predicted stoppage starts a new possession;
        # previous ownership cannot leak through it when the restart side is unknown.
        reset = reset | (np.r_[False, dead[:-1]] & ~dead)
    interaction_dead = dead if resumption is None else dead | resumption.unresolved
    evidence = infer_interactions(
        tracking.ball_xy_m, tracking.player_xy_m, tracking.teams, fps=tracking.fps,
        control_config=config.control, contact_config=config.contacts,
        interaction_config=config.interactions, dead=interaction_dead, reset=reset)
    if stoppages is not None:
        release_control = (stoppages.restart_team != "X") & ~dead
        evidence.labels[release_control] = stoppages.restart_team[release_control]
        evidence.source[release_control] = "restart_geometry"
    if resumption is not None:
        # Exhausting the bounded hold is uncertainty about play state, not a
        # restart or team signal. Do not let proximity alone seed an owner here.
        ambiguous = resumption.unresolved & ~dead
        evidence.labels[ambiguous] = "X"
        evidence.source[ambiguous] = "unresolved_play_state"
    state = decode_state(evidence.labels, dead, max_carry_frames=config.max_carry_frames, reset=reset)
    codes = pm.label_codes(state.labels)
    n_spells, mean_duration = describe(codes, tracking.fps)
    # Only aggregate diagnostics leave this function; no tracking, frame-specific
    # event records, or full prediction arrays are included in the summary.
    summary = {
        "frames": n_frames,
        "fps": tracking.fps,
        "ball_observed_fraction": float(np.isfinite(tracking.ball_xy_m).all(axis=1).mean()),
        "label_counts": {label: int(np.sum(state.labels == label)) for label in pm.LABELS},
        "abstention_fraction": float(np.mean(state.labels == "X")),
        "carried_frames": int(state.carried.sum()),
        "carry_policy": ("until_new_evidence" if config.max_carry_frames is None else "fixed_timeout"),
        "continuity_fraction": float(state.carried.mean()),
        "carried_without_ball_frames": int(np.sum(
            state.carried & ~np.isfinite(tracking.ball_xy_m).all(axis=1))),
        "max_control_evidence_age_seconds": max(0, int(state.evidence_age_frames.max())) / tracking.fps,
        "gap_conflict_frames": int(state.gap_conflict.sum()),
        "evidence_sources": dict(Counter(evidence.source.tolist())),
        "possible_contacts": len(evidence.interactions),
        "interaction_counts": dict(Counter(item.kind for item in evidence.interactions)),
        "diagnostic_possessions": n_spells,
        "diagnostic_mean_possession_seconds": mean_duration,
        "possession_count_convention": "X splits runs here; official spell scoring fills X within graded runs",
        "automatic_dead_ball_detection": ("temporal_play_state" if play_state is not None else
                                          "restart_hypotheses" if stoppages is not None else "disabled"),
        "stoppage_method": ("soft_evidence_viterbi" if play_state is not None else
                            "windowed_preparation_and_fitted_release" if stoppages is not None else None),
        "dead_evidence": ("caller_and_detector" if dead_supplied and (stoppages is not None or play_state is not None) and events is not None else
                          "supplied_by_caller" if dead_supplied else
                          "temporal_context_and_motion" if play_state is not None else
                          "automatic_tagging_and_motion" if events is not None and stoppages is not None else
                          "unavailable_without_tagging" if stoppages is not None else "detector_disabled"),
        "automatic_tagging_supplied": events is not None,
        "restart_count": len(stoppages.restarts) if stoppages is not None else 0,
        "stoppage_gate_diagnostics": dict(stoppages.diagnostics) if stoppages is not None else {},
        "play_state_diagnostics": dict(play_state.diagnostics) if play_state is not None else {},
        "episode_diagnostics": dict(episodes.diagnostics) if episodes is not None else {},
        "resumption_diagnostics": dict(resumption.diagnostics) if resumption is not None else {},
        "stoppage_status_counts": ({label: int(np.sum(stoppages.status == label)) for label in ("U", "D", "L")}
                                  if stoppages is not None else {}),
        "stoppage_status_counts_scope": "restart_anchor_detector_only",
        "config": asdict(config),
    }
    return Prediction(state, evidence, summary, stoppages, play_state, episodes, resumption)


def write_validated_submission(path, labels, *, n_frames):
    """Use the official writer and validator; publish the file only if valid.

    Validation here checks file shape/labels, never prediction accuracy. Refuse
    to overwrite an existing file. No ground-truth data is opened.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError("output already exists; choose a new submission path")
    labels = np.asarray(labels)
    if labels.shape != (n_frames,) or not np.isin(labels, pm.LABELS).all():
        raise ValueError("labels must contain exactly n_frames H/A/D/X values")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".possession-", suffix=".csv", dir=path.parent)
    os.close(descriptor)
    try:
        write_submission(temporary, labels)
        reason = pm.validate_submission(pm.read_label_csv(temporary), n_frames)
        if reason:
            raise ValueError("official submission validation failed: {}".format(reason))
        # Exclusive creation prevents racing another run from overwriting its
        # result after the initial exists check. Same-directory link is atomic.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return path
