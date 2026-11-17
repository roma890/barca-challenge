"""Alternative ball-observation model for IDSSE training copies (recipe v8).

idsse.observed_ball hides the ball on every provider-frozen (dead) frame, so in training a visible
ball always means live play. Real automatic tracking keeps following the ball for part of a
stoppage while players collect it and carry it to the restart. observed_ball2 gives each dead
spell that behaviour: the ball runs to a collecting player, stays with them, is passed to the
player who restarts, and one contiguous block of the spell is hidden. Uses positions only (the
provider's frozen-ball signature, as observed_ball does); labels are never read.

Every constant is a typical value set by hand from general football knowledge (not measured from any
data) and fixed before any model was scored. Nothing here comes from the
provided match's ground truth.
"""

import numpy as np

from possession import idsse

MIN_SPELL_S = 1.0          # shorter frozen runs stay hidden, as in observed_ball
P_HIDE_ALL = 0.25          # share of dead spells left fully hidden (observed_ball's behaviour)
COLLECT_S = (1.0, 4.0)     # time for the ball to reach the collecting player
PASS_MPS = (6.0, 14.0)     # speed of the pass to the restarting player
CARRY_OFFSET_M = (0.3, 0.7)
STILL_MPS = 0.5            # a ball slower than this for a second is lost by the tracker
LONG_GAPS_PER_MIN = 0.4    # optional long live-play dropouts
LONG_GAP_S = (1.0, 60.0)


def _runs(mask):
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _carrier(players, lo, hi, point):
    """Slot tracked on every frame of [lo, hi) that is nearest to point on average, or None."""
    block = players[lo:hi]
    whole = np.isfinite(block[..., 0]).all(axis=0)
    if not whole.any():
        return None
    distance = np.linalg.norm(block - point, axis=2).mean(axis=0)
    distance[~whole] = np.inf
    return int(distance.argmin())


def _dead_path(players, start, stop, origin, restart, rng, fps):
    """(length, 2) ball path through one dead spell, or None when no player is tracked throughout."""
    length = stop - start
    collector = _carrier(players, start, stop, origin)
    if collector is None:
        return None
    taker_block = players[max(start, stop - int(fps)):stop]
    whole = np.isfinite(players[start:stop, :, 0]).all(axis=0)
    distance = np.linalg.norm(taker_block - restart, axis=2).mean(axis=0)
    distance[~whole] = np.inf
    taker = int(distance.argmin())
    angle = rng.uniform(0, 2 * np.pi)
    offset = rng.uniform(*CARRY_OFFSET_M) * np.array([np.cos(angle), np.sin(angle)])
    with_collector = players[start:stop, collector] + offset
    with_taker = players[start:stop, taker] + offset
    t = np.arange(length)
    reach = int(min(rng.uniform(*COLLECT_S) * fps, length / 3))
    path = with_collector.copy()
    if reach > 0:
        blend = (t[:reach] / reach)[:, None]
        path[:reach] = origin + blend * (with_collector[reach - 1] - origin)
    if taker != collector:
        leave = int(rng.uniform(reach, max(reach + 1, length - fps)))
        gap = np.linalg.norm(with_taker[min(leave, length - 1)] - with_collector[min(leave, length - 1)])
        arrive = min(length, leave + max(int(0.3 * fps), int(gap / rng.uniform(*PASS_MPS) * fps)))
        if arrive > leave:
            blend = ((t[leave:arrive] - leave) / (arrive - leave))[:, None]
            path[leave:arrive] = with_collector[leave] + blend * (with_taker[arrive - 1] - with_collector[leave])
        path[arrive:] = with_taker[arrive:]
    settle = min(length, int(0.4 * fps))
    if settle > 0:
        blend = (np.arange(1, settle + 1) / settle)[:, None]
        path[-settle:] = path[-settle:] + blend * (restart - path[-1])
    return path


def observed_ball2(ball, players, *, seed, long_gaps=False, fps=idsse.FPS):
    raw = np.array(ball, dtype=float)
    n = len(raw)
    frozen = np.r_[False, (np.diff(raw, axis=0) == 0).all(axis=1)]
    moving = ~frozen
    # Live-play dropouts exactly as idsse.observed_ball draws them (same seed, same frames).
    rng = np.random.default_rng(idsse.DROPOUT_SEED)
    n_gaps = rng.poisson(idsse.LIVE_GAPS_PER_MIN * moving.sum() / (60 * fps))
    starts = rng.choice(np.flatnonzero(moving), size=n_gaps, replace=False) if n_gaps else []
    lengths = np.exp(rng.uniform(0, np.log(idsse.LIVE_GAP_MAX_S * fps), size=n_gaps)).astype(int) + 1

    rng = np.random.default_rng(seed)
    out, hidden = raw.copy(), frozen.copy()
    for start, stop in zip(*_runs(frozen)):
        length = stop - start
        if length < MIN_SPELL_S * fps or stop >= n or rng.uniform() < P_HIDE_ALL:
            continue
        path = _dead_path(players, start, stop, raw[start], raw[stop], rng, fps)
        if path is None:
            continue
        out[start:stop] = path
        hide = int(rng.uniform() * length)
        first = start + int(rng.integers(0, length - hide + 1))
        show = np.ones(length, dtype=bool)
        show[first - start:first - start + hide] = False
        # The tracker loses a ball that sits still: hide stretches slower than STILL_MPS over a second.
        step = np.linalg.norm(np.diff(path, axis=0, prepend=path[:1]), axis=1) * fps
        window = int(fps)
        slow = np.convolve(step, np.ones(window) / window, "same") < STILL_MPS
        hidden[start:stop] = ~(show & ~slow)
    for start, length in zip(starts, lengths):
        hidden[start:start + length] = True
    if long_gaps:
        count = rng.poisson(LONG_GAPS_PER_MIN * moving.sum() / (60 * fps))
        begins = rng.choice(np.flatnonzero(moving), size=count, replace=False) if count else []
        spans = (np.exp(rng.uniform(np.log(LONG_GAP_S[0]), np.log(LONG_GAP_S[1]), size=count)) * fps).astype(int)
        for start, length in zip(begins, spans):
            hidden[start:start + length] = True
    out[hidden] = np.nan
    return out
