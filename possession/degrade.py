"""Synthetic player-tracking damage for training and validation copies of other matches.

Our match's automatic tracking (aggregate, label-free statistics over the graded halves):
20.5 of 22 players tracked per frame; tracks last a median 35 s (quantiles below); and
about 15 frame-to-frame moves per minute faster than 12 m/s, the signature of a track
jumping to another player. damage_players() gives clean provider tracking the same
properties: every player's presence is cut into segments drawn from our segment-length
quantiles with short gaps between them, and tracks occasionally swap with the nearest
other player (sometimes an opponent, so the slot's team label becomes wrong) for a
segment-length stretch. Positions only; labels are never read.
"""

import numpy as np

# Our match's track segment lengths, seconds, at these quantiles.
SEGMENT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
SEGMENT_SECONDS = (1.7, 2.6, 7.9, 35.4, 145.2, 284.8, 401.7)
GAP_MEAN_S = 6.5           # mean gap between segments (calibrated: ~20.5 of 22 players tracked)
SWITCHES_PER_MIN = 9.0     # track swaps (calibrated: ~15 moves/min faster than 12 m/s)
SWITCH_RADIUS_M = 5.0


def _segment_lengths(rng, size, fps):
    """Inverse-CDF draws from the quantiles above, log-linear between them."""
    q = np.r_[0., SEGMENT_QUANTILES, 1.]
    seconds = np.r_[0.5, SEGMENT_SECONDS, 600.]
    return np.exp(np.interp(rng.uniform(size=size), q, np.log(seconds))) * fps


# Our match's fast moves are small hops, not identity swaps: median 0.7 m in one frame
# (p10 0.5 m, p90 1.3 m), 2.5% of them exchanges between two tracks, 4% landing on another
# player, 7% bouncing back within 0.2 s. hop mode reproduces that: a track's position
# steps by a log-normal 0.7 m (spread chosen to give p10 0.5 / p90 1.3 m) and keeps the
# offset until its next break.
HOPS_PER_MIN = 27.0
HOP_MEDIAN_M = 0.7
HOP_LOG_SD = 0.37


def damage_players(players, *, seed, fps=25., gap_mean_s=GAP_MEAN_S, switches_per_min=SWITCHES_PER_MIN,
                   hops_per_min=0.):
    """Copy of players (n, slots, 2) with fragmented tracks, identity swaps and position hops."""
    rng = np.random.default_rng(seed)
    out = np.array(players, dtype=float)
    n, slots = out.shape[:2]
    for slot in range(slots):
        seen = np.flatnonzero(np.isfinite(out[:, slot, 0]))
        if len(seen) == 0:
            continue
        t = seen[0] + int(_segment_lengths(rng, 1, fps)[0] * rng.uniform())
        while t < seen[-1]:
            gap = max(1, int(rng.exponential(gap_mean_s * fps)))
            out[t:t + gap, slot] = np.nan
            t += gap + int(_segment_lengths(rng, 1, fps)[0])
    n_switches = rng.poisson(switches_per_min * n / (60 * fps))
    for start in np.sort(rng.integers(0, n, size=n_switches)):
        visible = np.flatnonzero(np.isfinite(out[start, :, 0]))
        if len(visible) < 2:
            continue
        a = rng.choice(visible)
        distance = np.linalg.norm(out[start, visible] - out[start, a], axis=1)
        distance[visible == a] = np.inf
        if distance.min() > SWITCH_RADIUS_M:
            continue
        b = visible[distance.argmin()]
        stop = min(n, start + int(_segment_lengths(rng, 1, fps)[0]))
        out[start:stop, [a, b]] = out[start:stop, [b, a]]
    n_hops = rng.poisson(hops_per_min * n / (60 * fps))
    for start in np.sort(rng.integers(1, n, size=n_hops)):
        slot = rng.integers(slots)
        if not (np.isfinite(out[start, slot, 0]) and np.isfinite(out[start - 1, slot, 0])):
            continue
        gone = np.flatnonzero(~np.isfinite(out[start:, slot, 0]))
        stop = start + (gone[0] if len(gone) else n - start)
        angle = rng.uniform(0, 2 * np.pi)
        size = HOP_MEDIAN_M * np.exp(rng.normal(0, HOP_LOG_SD))
        out[start:stop, slot] += size * np.array([np.cos(angle), np.sin(angle)])
    return out


def tracking_stats(players, fps=25.):
    """Aggregates used for calibration: players per frame, segment median, fast moves per minute."""
    seen = np.isfinite(players[..., 0])
    edges = np.diff(np.r_[np.zeros((1, seen.shape[1])), seen.astype(int), np.zeros((1, seen.shape[1]))], axis=0)
    lengths = np.concatenate([np.flatnonzero(edges[:, k] == -1) - np.flatnonzero(edges[:, k] == 1)
                              for k in range(seen.shape[1])]) / fps
    both = seen[1:] & seen[:-1]
    speed = np.linalg.norm(players[1:] - players[:-1], axis=2)[both] * fps
    minutes = len(players) / (60 * fps)
    return {"players_per_frame": float(seen.sum(axis=1).mean()), "segment_median_s": float(np.median(lengths)),
            "segment_p10_s": float(np.percentile(lengths, 10)), "fast_moves_per_min": float((speed > 12).sum() / minutes)}


# Our match's ball track is smoothed in time by its tracker: frame-to-frame velocity changes
# (0.12 s windows) have 90th / 99th percentiles of 4.3 / 8.9 m/s, against 7.4 / 17.9 m/s on the
# observed-and-drifted IDSSE ball, while ball speeds match. A Gaussian of 6 frames (0.24 s),
# applied within visible stretches only, brings IDSSE to 4.2-4.6 / 8.1-9.2 m/s on all 7 matches.
BALL_SMOOTH_FRAMES = 6


def smooth_ball(ball, sigma_frames=BALL_SMOOTH_FRAMES):
    """Gaussian smoothing of the ball within visible stretches (never across gaps)."""
    out = np.array(ball, dtype=float)
    if sigma_frames <= 0:
        return out
    visible = np.isfinite(out).all(axis=1)
    edges = np.diff(np.r_[0, visible.astype(int), 0])
    half = int(3 * sigma_frames)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_frames) ** 2)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        length = stop - start
        segment = out[start:stop].copy()
        weight = np.convolve(np.ones(length), kernel, "full")[half:half + length]
        for axis in range(2):
            out[start:stop, axis] = np.convolve(segment[:, axis], kernel, "full")[half:half + length] / weight
    return out
