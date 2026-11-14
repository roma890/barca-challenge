"""Temporal state handling without conflating missing observations and dead play.

Inputs are upstream estimates, not raw nearest-player labels: accepting a team
here means the control detector has established control. Ball visibility alone
does not establish either control or a stoppage.
"""

from typing import NamedTuple, Optional, Sequence

import numpy as np


class StateResult(NamedTuple):
    labels: np.ndarray
    carried: np.ndarray
    evidence_age_frames: np.ndarray
    gap_conflict: np.ndarray

    @property
    def uncertain(self):
        """Continuity/abstention flags, not calibrated correctness probabilities."""
        return self.carried | (self.labels == "X")


def decode_state(
    control: Sequence[str],
    dead: Sequence[bool],
    *,
    max_carry_frames: Optional[int],
    reset: Sequence[bool] = None,
) -> StateResult:
    """Decode aligned per-frame evidence into H/A/D/X labels.

    ``control`` contains H/A for established control and X for no new evidence.
    ``dead`` independently marks frames with evidence that play has stopped.
    ``reset`` clears memory *before* its frame (e.g. at a recording boundary).
    Dead evidence overrides control and clears memory until control is observed
    again. With ``max_carry_frames=None``, preserve the last established team
    through uncertain play until new control, a stoppage or a reset. Missing
    evidence alone does not mean possession ended. An integer retains the older
    experimental timeout policy, after which unsupported frames become X.

    ``carried`` flags continuity separately from new evidence. Evidence age is 0
    on supported H/A or D, increases since the last team-control evidence, and is
    -1 when there is no team to retain (before first evidence or after reset/dead).
    ``gap_conflict`` retrospectively flags an uncertain gap whose two observed
    control endpoints disagree. It does NOT move the boundary: without evidence
    inside that gap, keep the prior team until the next supported change. Matching
    endpoints do not prove that no intervening turnover/stoppage occurred.

    These are uncertainty diagnostics, not calibrated probabilities. Labeling
    itself is causal; only the conflict annotation uses the subsequent endpoint.
    No minimum possession duration is imposed. Unobserved stoppages/turnovers can
    still leave a stale label; continuity is a fallback, not gap reconstruction.

    Note: the official spell scorer fills X with the previous label within each
    graded run. Abstaining therefore does not itself end a predicted spell.
    """
    control = np.asarray(control)
    dead = np.asarray(dead)
    if control.ndim != 1 or not np.isin(control, ("H", "A", "X")).all():
        raise ValueError("control must be a one-dimensional sequence of H/A/X")
    if dead.shape != control.shape or dead.dtype.kind != "b":
        raise ValueError("dead must be a boolean sequence matching control")
    if max_carry_frames is not None and (isinstance(max_carry_frames, (bool, np.bool_))
            or not isinstance(max_carry_frames, (int, np.integer))
            or max_carry_frames < 0):
        raise ValueError("max_carry_frames must be None or a nonnegative integer")
    if reset is None:
        reset = np.zeros(len(control), dtype=bool)
    else:
        reset = np.asarray(reset)
        if reset.shape != control.shape or reset.dtype.kind != "b":
            raise ValueError("reset must be a boolean sequence matching control")

    labels = np.full(len(control), "X", dtype="<U1")
    carried = np.zeros(len(control), dtype=bool)
    evidence_age = np.full(len(control), -1, dtype=int)
    gap_conflict = np.zeros(len(control), dtype=bool)
    previous = "X"
    missing_frames = 0
    last_control_frame = None
    for frame, team in enumerate(control):
        if reset[frame]:
            previous, missing_frames = "X", 0
            last_control_frame = None
        if dead[frame]:
            labels[frame] = "D"
            evidence_age[frame] = 0
            previous, missing_frames = "X", 0
            last_control_frame = None
        elif team != "X":
            if last_control_frame is not None and team != previous:
                gap_conflict[last_control_frame + 1:frame] = True
            labels[frame] = previous = team
            evidence_age[frame] = 0
            missing_frames = 0
            last_control_frame = frame
        else:
            missing_frames += 1
            if previous != "X":
                evidence_age[frame] = missing_frames
                if max_carry_frames is None or missing_frames <= max_carry_frames:
                    labels[frame] = previous
                    carried[frame] = True
    return StateResult(labels, carried, evidence_age, gap_conflict)
