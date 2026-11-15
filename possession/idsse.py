"""IDSSE open data (Bassek et al. 2025, Scientific Data): 7 Bundesliga matches, CC BY 4.0.

    https://doi.org/10.6084/m9.figshare.28196177

Other matches than the provided game, so RULES.md allows them for training and validation.
Each positions file (TRACAB optical tracking, 25 fps) carries, on every ball frame, the
provider's BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 guest). Labels are
read from those two flags by a fixed rule -- D when dead, else the possessing team --
so no label here was written by us.

    python -m possession.idsse --download      # ~2.6 GB into data/external/idsse
    python -m possession.idsse --parse         # one compact .npz per match (parsed/)
"""

import argparse
import json
from pathlib import Path
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

ARTICLE = "https://api.figshare.com/v2/articles/28196177"
DATA_DIR = Path("data/external/idsse")
PITCH = (105., 68.)
FPS = 25.
_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"')
_BALL = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def download(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARTICLE) as response:
        files = json.load(response)["files"]
    for item in files:
        target = data_dir / item["name"]
        if not target.exists():
            partial = target.with_name(target.name + ".part")
            urllib.request.urlretrieve(item["download_url"], str(partial))
            partial.rename(target)


def match_ids(data_dir=DATA_DIR):
    return sorted(p.name.split("_")[-1][:-4] for p in Path(data_dir).glob("DFL_04_03_positions_raw_observed_*.xml"))


def _file(data_dir, kind, match):
    found = list(Path(data_dir).glob("DFL_{}_*{}.xml".format(kind, match)))
    if len(found) != 1:
        raise FileNotFoundError("expected one {} file for {} in {}".format(kind, match, data_dir))
    return found[0]


def home_team(data_dir, match):
    root = ET.parse(_file(data_dir, "02_01_matchinformation", match)).getroot()
    return root.find(".//General").get("HomeTeamId")


def parse_positions(data_dir, match):
    """Arrays on one timeline: first-half frames, then second-half frames (N order)."""
    home = home_team(data_dir, match)
    tracks, ball = {}, {}
    current = None
    with open(_file(data_dir, "04_03_positions_raw_observed", match), encoding="utf-8") as handle:
        for line in handle:
            if "<FrameSet " in line:
                section, team, person = _FRAMESET.search(line).groups()
                half = 1 if section == "firstHalf" else 2
                if team == "BALL":
                    current = ball.setdefault(half, [])
                elif team == "referee":
                    current = None
                else:
                    side = "H" if team == home else "A"
                    current = tracks.setdefault((person, side), {}).setdefault(half, [])
                continue
            if current is None or "<Frame " not in line:
                continue
            n, x, y = _FRAME.search(line).groups()
            if current is ball.get(1) or current is ball.get(2):
                possession, status = _BALL.search(line).groups()
                current.append((int(n), float(x), float(y), int(possession), int(status)))
            else:
                current.append((int(n), float(x), float(y)))
    halves = []
    for half in (1, 2):
        rows = np.array(sorted(ball[half]))
        frames = rows[:, 0].astype(int)
        if not (np.diff(frames) == 1).all():
            raise ValueError("{} half {}: ball frames are not consecutive".format(match, half))
        halves.append((half, frames, rows))
    people = sorted(tracks, key=lambda key: (key[1] != "H", key[0]))
    n = sum(len(frames) for _, frames, _ in halves)
    players = np.full((n, len(people), 2), np.nan)
    ball_xy, possession, status, period = np.empty((n, 2)), np.empty(n, int), np.empty(n, int), np.empty(n, int)
    start = 0
    for half, frames, rows in halves:
        stop = start + len(frames)
        ball_xy[start:stop] = rows[:, 1:3]
        possession[start:stop], status[start:stop] = rows[:, 3], rows[:, 4]
        period[start:stop] = half
        for slot, key in enumerate(people):
            samples = tracks[key].get(half)
            if samples:
                samples = np.array(samples)
                index = samples[:, 0].astype(int) - frames[0]
                keep = (index >= 0) & (index < len(frames))
                players[start + index[keep], slot] = samples[keep, 1:3]
        start = stop
    offset = np.array(PITCH) / 2  # DFL coordinates are centred on the kick-off spot
    return {"ball": ball_xy + offset, "players": players + offset, "teams": np.array([side for _, side in people]),
            "period": period, "possession": possession, "status": status}


def labels(parsed):
    """D when the provider marks the ball dead, otherwise the possessing team (1 home, 2 guest)."""
    out = np.where(parsed["possession"] == 1, "H", np.where(parsed["possession"] == 2, "A", "X"))
    return np.where(parsed["status"] == 0, "D", out)


# The provider freezes the ball's position on every frame it marks dead (100% of dead
# frames in J03WMX are frozen, 99.4% of frozen frames are dead), so a model would learn
# "frozen = dead", which no live tracking offers. observed_ball() hides frozen frames, as
# real optical tracking loses the ball in most stoppages (Metrica: missing on ~90% of
# dead frames), and also drops it in live play at Metrica's live-play rate (1.0-1.3 gaps
# per minute; lengths heavy-tailed, median 0.12 s, p90 ~10 s) using log-uniform lengths
# from one frame to 15 s. Uses positions only, never the labels.
LIVE_GAPS_PER_MIN = 1.15
LIVE_GAP_MAX_S = 15.
DROPOUT_SEED = 0


def observed_ball(ball, *, seed=DROPOUT_SEED, fps=FPS):
    ball = np.array(ball, dtype=float)
    frozen = np.r_[False, (np.diff(ball, axis=0) == 0).all(axis=1)]
    moving = ~frozen
    rng = np.random.default_rng(seed)
    n_gaps = rng.poisson(LIVE_GAPS_PER_MIN * moving.sum() / (60 * fps))
    starts = rng.choice(np.flatnonzero(moving), size=n_gaps, replace=False) if n_gaps else []
    lengths = np.exp(rng.uniform(0, np.log(LIVE_GAP_MAX_S * fps), size=n_gaps)).astype(int) + 1
    hidden = frozen.copy()
    for start, length in zip(starts, lengths):
        hidden[start:start + length] = True
    ball[hidden] = np.nan
    return ball


def parsed_path(data_dir, match):
    return Path(data_dir) / "parsed" / "{}.npz".format(match)


def load_match(data_dir, match):
    """(ball, players, teams, period, labels); parses and caches the XML on first use."""
    path = parsed_path(data_dir, match)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **parse_positions(data_dir, match))
    parsed = dict(np.load(path))
    return parsed["ball"], parsed["players"], parsed["teams"], parsed["period"], labels(parsed)


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
            ball, players, teams, period, truth = load_match(args.data_dir, match)
            print("{}: {} frames, {} players, D {:.1%}, H {:.1%}, A {:.1%}".format(
                match, len(ball), players.shape[1], *[np.mean(truth == s) for s in "DHA"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
