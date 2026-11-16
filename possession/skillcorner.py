"""SkillCorner open data (A-League 2024/25 broadcast tracking, 10 fps), MIT licence.

    https://github.com/SkillCorner/opendata

Other matches than the provided game, so RULES.md allows them for training and validation.
Labels come from SkillCorner's own files by a fixed rule, never by hand:

- a frame inside a period but outside every phase of play is D (phases of play are
  defined only while the ball is in play; the gaps between them are the stoppages);
- inside a phase it takes the tracking frame's possession group (home/away), carried
  forward over frames whose group is empty (loose ball) and, where the phase has no
  group yet, the phase's team in possession.

To look like our match's tracking the data is resampled to 25 fps (linear between
samples no more than 0.2 s apart), moved to a corner origin, and the ball is kept only
where SkillCorner detected it (extrapolated ball positions become missing: our match's
ball is missing 41% of the time). Player positions are kept, extrapolated or not, since
our match tracks ~20.5 of 22 players per frame.

    python -m possession.skillcorner --download --parse
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np

REPO = "SkillCorner/opendata"
DATA_DIR = Path("data/external/skillcorner")
SOURCE_FPS = 10.
FPS = 25.
PITCH = (105., 68.)
MAX_INTERPOLATION_S = 0.2


def download(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tree_url = "https://api.github.com/repos/{}/git/trees/master?recursive=1".format(REPO)
    with urllib.request.urlopen(tree_url) as response:
        tree = json.load(response)["tree"]
    suffixes = ("_match.json", "_phases_of_play.csv", "_tracking_extrapolated.jsonl")
    for item in tree:
        path = item["path"]
        if item["type"] != "blob" or not path.startswith("data/matches/") or not path.endswith(suffixes):
            continue
        target = data_dir / Path(path).name
        if target.exists() and target.stat().st_size > 1000:
            continue
        host = "media.githubusercontent.com/media" if path.endswith(".jsonl") else "raw.githubusercontent.com"
        partial = target.with_name(target.name + ".part")
        urllib.request.urlretrieve("https://{}/{}/master/{}".format(host, REPO, path), str(partial))
        partial.rename(target)


def match_ids(data_dir=DATA_DIR):
    return sorted(p.name.split("_")[0] for p in Path(data_dir).glob("*_tracking_extrapolated.jsonl"))


def _resample(times, values, grid):
    """Per object: linear interpolation onto grid between samples <= 0.2 s apart, else NaN.

    values is (n, 2) for the ball or (n, slots, 2) for players; each slot is resampled
    on its own samples, so absent players never block the others.
    """
    if values.ndim == 3:
        return np.stack([_resample(times, values[:, k], grid) for k in range(values.shape[1])], axis=1)
    out = np.full((len(grid),) + values.shape[1:], np.nan)
    known = np.isfinite(values).all(axis=1)
    if known.sum() < 2:
        return out
    t, v = times[known], values[known]
    right = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    left = right - 1
    span = t[right] - t[left]
    weight = np.where(span > 0, (grid - t[left]) / np.where(span > 0, span, 1), 0.)
    ok = (grid >= t[left]) & (grid <= t[right]) & (span <= MAX_INTERPOLATION_S + 1e-9)
    shape = (-1,) + (1,) * (values.ndim - 1)
    blended = v[left] + (v[right] - v[left]) * weight.reshape(shape)
    out[ok] = blended[ok]
    return out


def parse_match(data_dir, match):
    data_dir = Path(data_dir)
    info = json.loads((data_dir / "{}_match.json".format(match)).read_text())
    home = info["home_team"]["id"]
    length, width = float(info["pitch_length"]), float(info["pitch_width"])
    side = {p["id"]: ("H" if p["team_id"] == home else "A") for p in info["players"]}
    people = sorted(side, key=lambda pid: (side[pid] != "H", pid))
    slot = {pid: i for i, pid in enumerate(people)}
    frames, period, ball, players, group = [], [], [], [], []
    with open(data_dir / "{}_tracking_extrapolated.jsonl".format(match), encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["period"] is None:
                continue
            frames.append(row["frame"])
            period.append(row["period"])
            b = row.get("ball_data") or {}
            ball.append((b["x"], b["y"]) if b.get("is_detected") and b.get("x") is not None else (np.nan, np.nan))
            xy = np.full((len(people), 2), np.nan)
            for p in row["player_data"]:
                if p["player_id"] in slot:
                    xy[slot[p["player_id"]]] = (p["x"], p["y"])
            players.append(xy)
            g = (row.get("possession") or {}).get("group")
            group.append("H" if g == "home team" else "A" if g == "away team" else "")
    frames, period = np.asarray(frames), np.asarray(period)
    ball, players, group = np.asarray(ball, float), np.asarray(players), np.asarray(group)
    with open(data_dir / "{}_phases_of_play.csv".format(match), encoding="utf-8") as handle:
        phases = list(csv.DictReader(handle))
    in_play = np.zeros(len(frames), dtype=bool)
    phase_team = np.full(len(frames), "", dtype="<U1")
    index = {f: i for i, f in enumerate(frames)}
    for phase in phases:
        lo, hi = int(phase["frame_start"]), int(phase["frame_end"])
        rows = [index[f] for f in range(lo, hi + 1) if f in index]
        in_play[rows] = True
        team = phase["team_in_possession_id"]
        if team:
            phase_team[rows] = "H" if int(float(team)) == home else "A"
    label = np.full(len(frames), "D", dtype="<U1")
    current = ""
    for i in range(len(frames)):
        if not in_play[i]:
            current = ""
            continue
        if group[i]:
            current = group[i]
        elif not current:
            current = phase_team[i]
        label[i] = current or "X"
    offset = np.array([length, width]) / 2
    halves = []
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        times = (frames[rows] - frames[rows[0]]) / SOURCE_FPS
        grid = np.arange(0, times[-1] + 1e-9, 1 / FPS)
        nearest = rows[np.clip(np.rint(grid * SOURCE_FPS).astype(int), 0, len(rows) - 1)]
        halves.append((p, _resample(times, ball[rows], grid), _resample(times, players[rows], grid), label[nearest]))
    return {
        "ball": np.concatenate([h[1] for h in halves]) + offset,
        "players": np.concatenate([h[2] for h in halves]) + offset,
        "period": np.concatenate([np.full(len(h[3]), h[0]) for h in halves]),
        "labels": np.concatenate([h[3] for h in halves]),
        "teams": np.array([side[pid] for pid in people]),
        "pitch": np.array([length, width]),
    }


def parsed_path(data_dir, match):
    return Path(data_dir) / "parsed" / "{}.npz".format(match)


def load_match(data_dir, match):
    """(ball, players, teams, period, labels, pitch) at 25 fps; parses and caches on first use."""
    path = parsed_path(data_dir, match)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **parse_match(data_dir, match))
    z = dict(np.load(path))
    return z["ball"], z["players"], z["teams"], z["period"], z["labels"], tuple(z["pitch"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--parse", action="store_true")
    args = parser.parse_args(argv)
    if args.download:
        download(args.data_dir)
    if args.parse:
        for match in match_ids(args.data_dir):
            ball, players, teams, period, labels, pitch = load_match(args.data_dir, match)
            print("{}: {} frames, pitch {}, D {:.1%}, H {:.1%}, A {:.1%}, X {:.1%}".format(
                match, len(ball), pitch, *[np.mean(labels == s) for s in "DHAX"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
