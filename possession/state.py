"""Temporal state handling without conflating missing observations and dead play.

Inputs are upstream estimates, not raw nearest-player labels: accepting a team
here means the control detector has established control. Ball visibility alone
does not establish either control or a stoppage.
"""

from typing import NamedTuple, Sequence

import numpy as np


class StateResult(NamedTuple):
    labels: np.ndarray
    carried: np.ndarray


def decode_state(
    control: Sequence[str],
    dead: Sequence[bool],
    *,
    max_carry_frames: int,
    reset: Sequence[bool] = None,
) -> StateResult:
    """Decode aligned per-frame evidence into H/A/D/X labels.

    ``control`` contains H/A for established control and X for no new evidence.
    ``dead`` independently marks frames with evidence that play has stopped.
    ``reset`` clears memory *before* its frame (e.g. at a recording boundary).
    Dead evidence overrides control and clears memory until control is observed
    again. Unknown frames carry the last team for at most ``max_carry_frames``
    consecutive frames, then become X; they never become D just due to absence.

    The carry limit is required explicitly: it is an experimental policy, not a
    validated football constant. ``carried`` identifies inferred continuity for
    diagnostics; it is not a calibrated confidence score. No lookahead is used,
    and short established possessions are preserved without a minimum duration.

    Note: the official spell scorer fills X with the previous label within each
    graded run. Abstaining therefore does not itself end a predicted spell.
    """
    control = np.asarray(control)
    dead = np.asarray(dead)
    if control.ndim != 1 or not np.isin(control, ("H", "A", "X")).all():
        raise ValueError("control must be a one-dimensional sequence of H/A/X")
    if dead.shape != control.shape or dead.dtype.kind != "b":
        raise ValueError("dead must be a boolean sequence matching control")
    if (isinstance(max_carry_frames, (bool, np.bool_))
            or not isinstance(max_carry_frames, (int, np.integer))
            or max_carry_frames < 0):
        raise ValueError("max_carry_frames must be a nonnegative integer")
    if reset is None:
        reset = np.zeros(len(control), dtype=bool)
    else:
        reset = np.asarray(reset)
        if reset.shape != control.shape or reset.dtype.kind != "b":
            raise ValueError("reset must be a boolean sequence matching control")

    labels = np.full(len(control), "X", dtype="<U1")
    carried = np.zeros(len(control), dtype=bool)
    previous = "X"
    missing_frames = 0
    for frame, team in enumerate(control):
        if reset[frame]:
            previous, missing_frames = "X", 0
        if dead[frame]:
            labels[frame] = "D"
            previous, missing_frames = "X", 0
        elif team != "X":
            labels[frame] = previous = team
            missing_frames = 0
        else:
            missing_frames += 1
            if previous != "X" and missing_frames <= max_carry_frames:
                labels[frame] = previous
                carried[frame] = True
    return StateResult(labels, carried)
