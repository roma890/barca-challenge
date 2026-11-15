"""Combine motion evidence and possible touches, keeping pass hypotheses explicit."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .contacts import Contact, ContactConfig, _motion, _number, _tracking_inputs, detect_contacts
from .control import ControlConfig, estimate_control


@dataclass(frozen=True)
class InteractionConfig:
    lookahead_frames: int
    max_target_miss_m: float
    min_outgoing_speed_mps: float
    accept_pass_hypotheses: bool

    def __post_init__(self):
        _number("lookahead_frames", self.lookahead_frames, positive=True, integer=True)
        _number("max_target_miss_m", self.max_target_miss_m)
        _number("min_outgoing_speed_mps", self.min_outgoing_speed_mps, positive=True)
        if not isinstance(self.accept_pass_hypotheses, bool):
            raise ValueError("accept_pass_hypotheses must explicitly be True or False")


@dataclass(frozen=True)
class Interaction:
    contact: Contact
    kind: str
    receiver_track_index: int = -1
    confirmation_frame: int = -1


@dataclass(frozen=True)
class InteractionResult:
    labels: np.ndarray
    source: np.ndarray
    interactions: Tuple[Interaction, ...]


def infer_interactions(
    ball_xy_m: np.ndarray,
    player_xy_m: np.ndarray,
    teams: Sequence[str],
    *,
    fps: float,
    control_config: ControlConfig,
    contact_config: ContactConfig,
    interaction_config: InteractionConfig,
    reset: Optional[Sequence[bool]] = None,
    dead: Optional[Sequence[bool]] = None,
) -> InteractionResult:
    """Build OFFLINE H/A/X control evidence for decode_state, plus explanations.

    All settings are explicit and experimental. Base control uses proximity and
    compatible motion, with speed-jump guards added here. A previously established
    team may continue through a close challenge if its own player still has
    compatible motion and proximity; proximity of a defender alone cannot switch
    possession. No evidence is extended through a gap without physical support.

    Contacts with later control by the SAME slot before any other touch/control
    are reception candidates: evidence may be backdated to that contact. Control
    must continue from the contact on a continuously observed relative trajectory
    within max_distance_m, so a loose rebound followed by a chase is not backdated.

    A contact directing the ball toward a teammate who subsequently shows
    control is a pass HYPOTHESIS. It requires a reliable uninterrupted flight,
    departure from the toucher's proximity, and no intervening contact/control.
    A successful deflection can satisfy exactly these conditions. Therefore pass
    hypotheses change labels ONLY when accept_pass_hypotheses=True. Under the
    conservative False policy they remain annotations and the receiver's direct
    control starts possession. Neither policy can recover intent from identical
    trajectories; a failed one-touch pass without prior control can remain unknown.

    Emergency clearances get no automatic turnover. Prior controlled possession
    survives the act of clearing, but a kick alone does not establish it. No claim
    of clearance/deflection ground-truth classification is made. Dead/reset masks
    prevent cross-boundary inference; dead frames yield X control evidence for the
    separate state decoder to turn into D. Sources are diagnostics, not confidence.
    """
    if not isinstance(interaction_config, InteractionConfig):
        raise ValueError("interaction_config must be an InteractionConfig")
    if not isinstance(contact_config, ContactConfig):
        raise ValueError("contact_config must be a ContactConfig")
    ball, players, teams, reset, dead = _tracking_inputs(
        ball_xy_m, player_xy_m, teams, fps, reset, dead)
    contacts = detect_contacts(ball, players, teams, fps=fps, config=contact_config,
                               reset=reset, dead=dead)
    breaks = dead.copy()
    breaks[1:] |= dead[:-1]
    for event in contacts:
        breaks[event.frame] = True
        if event.frame + 1 < len(breaks):
            breaks[event.frame + 1] = True
    base = estimate_control(ball, players, teams, fps=fps, config=control_config,
                            reset=reset, motion_breaks=breaks)
    ball_v, ball_ok = _motion(ball, fps, contact_config.max_ball_speed_mps, reset)
    player_v, player_ok = _motion(players, fps, contact_config.max_player_speed_mps, reset)
    n = len(ball)
    labels = base.labels.copy()
    source = np.full(n, "unknown", dtype="<U24")
    distances = np.full(players.shape[:2], np.inf)
    visible = np.isfinite(ball).all(axis=-1)[:, None] & np.isfinite(players).all(axis=-1)
    if visible.any():
        # Index only observed positions so missing coordinates never enter norms.
        frames, slots = np.nonzero(visible)
        distances[frames, slots] = np.linalg.norm(players[frames, slots] - ball[frames], axis=-1)
    valid_motion = player_ok & ball_ok[:, None]
    compatible = base.compatible & valid_motion
    for frame in range(n):
        slot = base.track_index[frame]
        if dead[frame] or (slot >= 0 and not valid_motion[frame, slot]):
            labels[frame] = "X"
    source[labels != "X"] = "direct_motion"
    source[(labels != "X") & base.fit_used] = "fitted_motion"

    assessments = []
    contact_at = {event.frame: event for event in contacts}
    for event in contacts:
        f, slot = event.frame, event.track_index
        if slot < 0:
            assessments.append(Interaction(event, "ambiguous_contact"))
            continue
        kind = "unresolved_contact"
        receiver, confirmation = -1, -1
        outgoing = np.asarray(event.outgoing_velocity_mps)
        out_speed = np.linalg.norm(outgoing)
        # Subsequent evidence cannot reach through missing ball data, dead play,
        # resets or unrelated contacts/control. A same-team receiver touch may
        # terminate the flight and then be confirmed by that receiver's motion.
        end = min(n, f + interaction_config.lookahead_frames + 1)
        receiving_touch = None
        for later in range(f + 1, end):
            if reset[later] or dead[later] or not ball_ok[later]:
                break
            if later in contact_at:
                next_contact = contact_at[later]
                if (receiving_touch is not None or next_contact.team != event.team
                        or next_contact.track_index == slot):
                    break
                receiving_touch = next_contact
            next_slot = base.track_index[later]
            if next_slot < 0 or labels[later] == "X":
                continue
            if receiving_touch is not None and next_slot != receiving_touch.track_index:
                break
            if next_slot == slot:
                close = distances[f:later + 1, slot] <= control_config.max_distance_m
                reliable = player_ok[f:later + 1, slot].all()
                if close.all() and reliable:
                    kind, receiver, confirmation = "reception_supported", slot, later
                break
            if teams[next_slot] != event.team:
                break
            # Target motion must already be measurable at contact, and remain
            # reliable until reception. Never follow a slot across a track gap.
            if not player_ok[f:later + 1, next_slot].all():
                break
            relative_position = players[f, next_slot] - ball[f]
            closing_velocity = outgoing - player_v[f, next_slot]
            closing_squared = np.dot(closing_velocity, closing_velocity)
            arrival_s = (np.dot(relative_position, closing_velocity) / closing_squared
                         if closing_squared > 0 else -1.)
            miss = np.linalg.norm(relative_position - closing_velocity * arrival_s)
            departure = np.any(distances[f + 1:later + 1, slot] > control_config.max_distance_m)
            # A turn toward a receiver plus consistent later reception supports a
            # candidate, not an assertion that the contact was intentional.
            arrival_frame = receiving_touch.frame if receiving_touch is not None else later
            elapsed = np.arange(arrival_frame - f + 1) / fps
            predicted = ball[f] + elapsed[:, None] * outgoing
            flight_ok = (np.linalg.norm(predicted - ball[f:arrival_frame + 1], axis=-1).max()
                         <= interaction_config.max_target_miss_m)
            if (out_speed >= interaction_config.min_outgoing_speed_mps
                    and 0 < arrival_s <= interaction_config.lookahead_frames / fps
                    and miss <= interaction_config.max_target_miss_m and departure and flight_ok):
                kind, receiver, confirmation = "pass_hypothesis", next_slot, later
            break
        if kind == "reception_supported" or (
                kind == "pass_hypothesis" and interaction_config.accept_pass_hypotheses):
            # End before confirmation so its independently observed evidence is
            # retained. Never overwrite opposing control or a separate contact.
            labels[f:confirmation] = event.team
            source[f:confirmation] = ("reception_backdated" if kind == "reception_supported"
                                      else "pass_hypothesis")
        assessments.append(Interaction(event, kind, receiver, confirmation))

    incumbent = "X"
    for frame in range(n):
        if reset[frame] or dead[frame]:
            incumbent = "X"
        if dead[frame]:
            continue
        if labels[frame] != "X":
            incumbent = labels[frame]
        elif incumbent != "X" and np.any(compatible[frame] & (teams == incumbent)):
            labels[frame] = incumbent
            source[frame] = "incumbent_motion"
    return InteractionResult(labels, source, tuple(assessments))
