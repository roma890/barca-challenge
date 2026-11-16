"""Experimental, optional refinement of H<->A possession-change frames after decoding.

The learned model plus Viterbi decide WHETHER possession changes; this module only
re-times WHEN, using local physical evidence. It never creates, removes, reorders or
relabels a possession spell, never moves a boundary that touches dead ball (D), never lets a
boundary cross a neighbouring one, keeps every spell at least one frame long, and moves a
boundary by at most ``window_s``.

Evidence, per decoded boundary old -> new, inside the search window:

* Contacts (``contacts.detect_contacts``): a reliable change in the ball's path with exactly
  one clearly nearest player, attributed to that player's team. Unattributed contacts are
  ignored.
* Control: the ball visible, within ``control_radius_m`` of a player, every opponent at least
  ``control_margin_m`` further away, moving with that player (relative speed at most
  ``control_rel_speed_mps``) for ``min_control_frames`` consecutive frames.

Decision, in order (the same rule for H and A):

1. Any new-team contact before the last old-team contact -> contested: unchanged.
2. A new-team contact exists -> move to the first one (the reception / interception frame),
   unless control in the frames being relabelled contradicts it.
3. Only old-team contacts, the last at or after the boundary -> move to just after it (the
   old team still touched the ball, so possession cannot have changed yet).
4. No contacts (typically a missing-ball gap) -> bound the boundary between the last
   credible old-team control and the first credible new-team control that both start and end
   inside the window; control by the new team before the old team's -> contested: unchanged.
5. Otherwise unchanged.

The default settings are physical priors fixed before any evaluation; none was derived from
the competition match or tuned on evaluation data.
"""

from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .contacts import ContactConfig, detect_contacts

TEAMS = ("H", "A")


def _default_contacts():
    # Priors: touches within 1 m of exactly one clearly nearest player (0.5 m margin), a
    # velocity change of at least 3 m/s, straight-line fits within 0.15 m over +-2 frames.
    return ContactConfig(window_frames=2, max_distance_m=1.0, attribution_margin_m=0.5,
                         min_velocity_change_mps=3.0, max_fit_error_m=0.15,
                         max_ball_speed_mps=40.0, max_player_speed_mps=12.0)


@dataclass(frozen=True)
class RefineConfig:
    window_s: float = 1.0
    control_radius_m: float = 1.0
    control_margin_m: float = 1.0
    control_rel_speed_mps: float = 3.0
    min_control_frames: int = 3
    contacts: ContactConfig = field(default_factory=_default_contacts)

    def __post_init__(self):
        for name in ("window_s", "control_radius_m", "control_margin_m", "control_rel_speed_mps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError("{} must be a finite positive number".format(name))
        if isinstance(self.min_control_frames, bool) or not isinstance(self.min_control_frames, int) \
                or self.min_control_frames < 1:
            raise ValueError("min_control_frames must be a positive integer")
        if not isinstance(self.contacts, ContactConfig):
            raise ValueError("contacts must be a ContactConfig")


DEFAULT_CONFIG = RefineConfig()


def control_team(ball, players, teams, *, fps, config):
    """Per-frame team ('H'/'A') in credible control of the ball, '' where none."""
    teams = np.asarray(teams)
    n = len(ball)
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest = dist.argmin(axis=1)
    d_near = dist[np.arange(n), nearest]
    near_team = teams[nearest]
    opponent = np.where(teams[None, :] != near_team[:, None], dist, np.inf).min(axis=1)
    rel = np.full(n, np.inf)
    ball_v = np.diff(ball, axis=0) * fps
    player_v = np.diff(players[:, :, :], axis=0) * fps
    with np.errstate(invalid="ignore"):
        rel[1:] = np.linalg.norm(ball_v - player_v[np.arange(n - 1), nearest[1:]], axis=1)
    rel = np.where(np.isfinite(rel), rel, np.inf)
    ok = ((d_near <= config.control_radius_m) & (opponent >= d_near + config.control_margin_m)
          & (rel <= config.control_rel_speed_mps))
    raw = np.where(ok, near_team, "")
    out = np.full(n, "", dtype="<U1")
    start = 0
    for t in range(1, n + 1):
        if t == n or raw[t] != raw[start]:
            if raw[start] and t - start >= config.min_control_frames:
                out[start:t] = raw[start]
            start = t
    return out


def _contradicted(ctrl, lo, hi, should_be):
    """True when control in [lo, hi) favours the other team over `should_be`."""
    if hi <= lo:
        return False
    seg = ctrl[lo:hi]
    other = "A" if should_be == "H" else "H"
    return int((seg == other).sum()) > int((seg == should_be).sum())


def _decide(b, old, new, lo, hi, contact_frames, contact_teams, ctrl):
    inside = (contact_frames >= lo) & (contact_frames <= hi)
    old_c = contact_frames[inside & (contact_teams == old)]
    new_c = contact_frames[inside & (contact_teams == new)]
    if len(old_c) and len(new_c) and new_c.min() < old_c.max():
        return b, "contested"
    if len(new_c):
        target = int(new_c.min())
    elif len(old_c) and old_c.max() >= b:
        target = int(old_c.max()) + 1
    else:
        return _control_bounds(b, old, new, lo, hi, ctrl)
    if target < b and _contradicted(ctrl, target, b, new):
        return b, "contradicted"
    if target > b and _contradicted(ctrl, b, target, old):
        return b, "contradicted"
    return target, "contact"


def _control_bounds(b, old, new, lo, hi, ctrl):
    seg = ctrl[lo:hi + 1]
    old_f = lo + np.flatnonzero(seg == old)
    new_f = lo + np.flatnonzero(seg == new)
    if len(old_f) and len(new_f) and new_f.min() < old_f.max():
        return b, "contested"
    # Only bounds that begin/end inside the window carry timing information.
    last_old = int(old_f.max()) if len(old_f) and old_f.max() < hi else None
    first_new = int(new_f.min()) if len(new_f) and new_f.min() > lo else None
    if last_old is not None and b <= last_old:
        return last_old + 1, "control"
    if first_new is not None and b > first_new:
        return first_new, "control"
    return b, "no-evidence" if last_old is None and first_new is None else "within-bounds"


def refine(labels, ball, players, teams, *, fps, config=DEFAULT_CONFIG, return_report=False):
    """Re-time H<->A boundaries in `labels`; returns a new array (and a report if asked)."""
    if not isinstance(config, RefineConfig):
        raise ValueError("config must be a RefineConfig")
    labels = np.asarray(labels)
    ball = np.asarray(ball, dtype=float)
    players = np.asarray(players, dtype=float)
    teams = np.asarray(teams)
    n = len(labels)
    if ball.shape != (n, 2) or players.ndim != 3 or players.shape[0] != n or players.shape[2] != 2:
        raise ValueError("ball must be (n, 2) and players (n, slots, 2) for n labels")
    if not np.isin(labels, ("H", "A", "D", "X")).all():
        raise ValueError("labels must be H, A, D or X")
    out = labels.copy()
    report = []
    if n < 2:
        return (out, report) if return_report else out
    window = max(1, int(round(config.window_s * fps)))
    contacts = detect_contacts(ball, players, teams, fps=fps, config=config.contacts,
                               dead=np.asarray(labels == "D", dtype=bool))
    kept = [c for c in contacts if c.team in TEAMS]
    contact_frames = np.array([c.frame for c in kept], dtype=int)
    contact_teams = np.array([c.team for c in kept], dtype="<U1")
    ctrl = control_team(ball, players, teams, fps=fps, config=config)

    changes = (np.flatnonzero(labels[1:] != labels[:-1]) + 1).tolist()
    positions = list(changes)  # current (refined) position of every boundary
    for k, b in enumerate(changes):
        old, new = labels[b - 1], labels[b]
        if old not in TEAMS or new not in TEAMS:
            continue
        spell_start = positions[k - 1] if k else 0
        spell_stop = changes[k + 1] if k + 1 < len(changes) else n
        lo, hi = max(b - window, spell_start + 1), min(b + window, spell_stop - 1)
        target, reason = _decide(b, old, new, lo, hi, contact_frames, contact_teams, ctrl)
        target = int(min(max(target, lo), hi))
        if target < b:
            out[target:b] = new
        elif target > b:
            out[b:target] = old
        positions[k] = target
        report.append({"frame": b, "refined": target, "from": str(old), "to": str(new), "reason": reason})
    return (out, report) if return_report else out
