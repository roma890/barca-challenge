"""Strict local ATD adapter using the official slot order and frame convention."""

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np

from starter.baseline import team_of_each_track


@dataclass(frozen=True)
class Tracking:
    ball_xy_m: np.ndarray
    player_xy_m: np.ndarray
    teams: np.ndarray
    fps: float
    pitch: Optional[Tuple[float, float]] = None


def _load_teams(meta_path):
    # Validate what the official helper assumes: exactly two teams, and no
    # unknown membership silently treated as away. Preserve XML Player order.
    try:
        root = ET.parse(meta_path).getroot()
    except ET.ParseError:
        raise ValueError("ATD metadata is not valid XML") from None
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2 or None in team_ids or len(set(team_ids)) != 2:
        raise ValueError("ATD metadata must define exactly two distinct team IDs")
    slots = list(root.iter("Player"))
    if not slots or any(player.get("teamId") not in team_ids for player in slots):
        raise ValueError("ATD metadata needs track slots with known team membership")
    # Source of truth for H/A ordering is the same helper as the official baseline.
    codes = team_of_each_track(meta_path)
    return np.where(codes == 0, "H", "A")


def load_tracking(
    meta_path: Union[str, Path],
    raw_path: Union[str, Path],
    *,
    n_frames: int,
    pitch: Sequence[float],
    fps: float,
) -> Tracking:
    """Stream raw ATD into metre-based arrays; no kloppy or video access.

    Each record must be `frame:player;player;...:ball`, with each object x,y,speed.
    Frame IDs must be exactly 1..n_frames in order, so array row = frame_id - 1.
    Do NOT apply games.py's private-ground-truth counter offsets to these IDs.

    Coordinates are normalised in the provided ATD. Multiply by pitch length and
    width without clipping, since balls/players can be outside pitch boundaries.
    Speed fields are parsed for schema validity, but motion is computed later
    from positions. A partially missing position becomes (NaN, NaN); missing
    positions never become zero. Missing speed alone does not hide a position.

    Strictly reject truncated, extra, shuffled or malformed rows and infinities.
    Error messages describe the line/schema without echoing restricted records.
    The input text is streamed, but dense position arrays remain in local memory.
    This function only reads explicitly supplied files and writes no caches.
    """
    if (isinstance(n_frames, (bool, np.bool_)) or not isinstance(n_frames, Integral)
            or n_frames <= 0):
        raise ValueError("n_frames must be a positive integer")
    if (isinstance(fps, (bool, np.bool_)) or not isinstance(fps, Real)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be a finite positive number")
    dimensions = np.asarray(pitch, dtype=float)
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise ValueError("pitch must contain finite positive length and width in metres")
    teams = _load_teams(meta_path)
    n_slots = len(teams)
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)

    def position(group, line_number):
        fields = group.split(",")
        if len(fields) != 3:
            raise ValueError("ATD line {}: each object must have x,y,speed".format(line_number))
        try:
            values = np.array([float(value) for value in fields])
        except ValueError:
            raise ValueError("ATD line {}: object fields must be numeric or NaN".format(line_number)) from None
        if np.isinf(values).any():
            raise ValueError("ATD line {}: infinite object values are invalid".format(line_number))
        xy = values[:2]
        return xy * dimensions if np.isfinite(xy).all() else np.array([np.nan, np.nan])

    count = 0
    with open(raw_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number > n_frames:
                raise ValueError("ATD has more than {} records".format(n_frames))
            fields = line.strip().split(":")
            if len(fields) != 3:
                raise ValueError("ATD line {}: expected frame:players:ball".format(line_number))
            try:
                frame_id = int(fields[0])
            except ValueError:
                raise ValueError("ATD line {}: frame ID must be an integer".format(line_number)) from None
            if frame_id != line_number:
                raise ValueError("ATD frame IDs must be consecutive from 1 (line {})".format(line_number))
            groups = fields[1].split(";")
            if len(groups) != n_slots:
                raise ValueError("ATD line {}: track-slot count differs from metadata".format(line_number))
            index = frame_id - 1
            for slot, group in enumerate(groups):
                players[index, slot] = position(group, line_number)
            ball[index] = position(fields[2], line_number)
            count += 1
    if count != n_frames:
        raise ValueError("ATD record count {} != expected {}".format(count, n_frames))
    return Tracking(ball, players, teams, float(fps), tuple(dimensions.tolist()))
