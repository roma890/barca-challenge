"""Score a Viterbi config on Metrica's open sample games (other matches, allowed by RULES.md).

Labels come from Metrica's published event data by a fixed algorithm, not by hand:
the team of the last on-ball event owns the ball, and the ball is dead from a ball
out, foul, offside or goal until the next restart. These are event-based labels, so
their boundaries differ from the challenge's tracking-derived ground truth; use them
to compare versions, not as an estimate of the competition score.

    python -m possession.validate_metrica --download
    python -m possession.validate_metrica --config possession/configs/viterbi_v2.json
    python -m possession.validate_metrica --make-degraded
    python -m possession.validate_metrica --degraded --config possession/configs/viterbi_v2.json

The degraded copies add a smooth drift to the ball so its jitter and distance to
the nearest player resemble our match's tracking (see DRIFT_SD_M).
"""

import argparse
import csv
from pathlib import Path
import json
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import possession_metrics as pm  # noqa: E402

from .viterbi import decode_possession, load_viterbi_config  # noqa: E402

SOURCE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
GAMES = (1, 2)
PARTS = ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData")
# Sample game 3 is published in FIFA EPTS format (text tracking + XML metadata, JSON events).
EPTS_GAMES = (3,)
ALL_GAMES = GAMES + EPTS_GAMES
EPTS_PARTS = ("tracking.txt", "metadata.xml", "events.json")
DEFAULT_DIR = Path("data/external/metrica")
DEGRADED_DIR = Path("data/external/metrica_degraded")
PITCH = (105., 68.)
FPS = 25.

# Metrica's ball sits almost on a player (median 0.28 m to the nearest player; frame
# jitter 0.03 m/s). Our match's ball tracking: 1.91 m and 0.229 m/s. A smooth random
# drift (Gaussian-blurred noise, 0.5 s) of 1.2 m standard deviation reproduces both
# aggregates on games 1 and 2 (1.87-1.91 m, 0.219-0.221 m/s). No labels were used.
DRIFT_SD_M = 1.2
DRIFT_SMOOTH_S = 0.5
DRIFT_SEED = 0

OWNING_EVENTS = {"PASS", "RECOVERY", "BALL LOST", "SHOT", "SET PIECE"}


def file_path(data_dir, game, part):
    return Path(data_dir) / "Sample_Game_{}_{}.csv".format(game, part)


def epts_path(data_dir, game, part):
    return Path(data_dir) / "Sample_Game_{}_{}".format(game, part)


def download(data_dir):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    targets = [file_path(data_dir, game, part) for game in GAMES for part in PARTS]
    targets += [epts_path(data_dir, game, part) for game in EPTS_GAMES for part in EPTS_PARTS]
    for target in targets:
        if not target.exists():
            game = target.name.split("_")[2]
            url = "{}/Sample_Game_{}/{}".format(SOURCE, game, target.name)
            urllib.request.urlretrieve(url, str(target))


def smooth_drift(n, sd_m, smooth_s, rng):
    """(n, 2) smooth random offset in metres: white noise blurred by a Gaussian of smooth_s."""
    sigma = smooth_s * FPS
    taps = np.arange(-int(4 * sigma), int(4 * sigma) + 1)
    kernel = np.exp(-0.5 * (taps / sigma) ** 2)
    kernel /= np.sqrt((kernel ** 2).sum())
    white = rng.standard_normal((n + len(taps) - 1, 2))
    return np.stack([np.convolve(white[:, axis], kernel, mode="valid") for axis in range(2)],
                    axis=1) * sd_m


def make_degraded(source_dir=DEFAULT_DIR, target_dir=DEGRADED_DIR, *, drift_sd_m=DRIFT_SD_M,
                  smooth_s=DRIFT_SMOOTH_S, seed=DRIFT_SEED):
    """Write copies whose ball positions carry a smooth drift; the originals are never touched.

    Only the Ball columns change, by the same offset in the home and away files. Missing
    ball samples stay missing; player columns and events are copied unchanged.
    """
    source, target = Path(source_dir).resolve(), Path(target_dir).resolve()
    if source == target:
        raise ValueError("degraded copy must go to a different folder than the originals")
    outputs = [file_path(target, game, part) for game in GAMES for part in PARTS]
    if any(path.exists() for path in outputs):
        raise FileExistsError("degraded files already exist in {}; remove them first".format(target))
    target.mkdir(parents=True, exist_ok=True)
    scale = np.asarray(PITCH)
    for game in GAMES:
        rng = np.random.default_rng(seed + game)
        offsets = None
        for part in PARTS[:2]:
            with open(file_path(source, game, part), newline="") as handle:
                rows = list(csv.reader(handle))
            ball = rows[2].index("Ball")
            if offsets is None:
                offsets = smooth_drift(len(rows) - 3, drift_sd_m, smooth_s, rng) / scale
            for row, offset in zip(rows[3:], offsets):
                xy = np.array([float(row[ball]), float(row[ball + 1])])
                if np.isfinite(xy).all():
                    row[ball], row[ball + 1] = (format(value, ".6f") for value in xy + offset)
            with open(file_path(target, game, part), "w", newline="") as handle:
                csv.writer(handle).writerows(rows)
        shutil.copyfile(file_path(source, game, PARTS[2]), file_path(target, game, PARTS[2]))
    return target


def make_degraded_epts(source_dir=DEFAULT_DIR, target_dir=DEGRADED_DIR, *, games=EPTS_GAMES,
                       drift_sd_m=DRIFT_SD_M, smooth_s=DRIFT_SMOOTH_S, seed=DRIFT_SEED):
    """make_degraded() for the EPTS games: same drift recipe, only the ball field changes."""
    source, target = Path(source_dir).resolve(), Path(target_dir).resolve()
    if source == target:
        raise ValueError("degraded copy must go to a different folder than the originals")
    outputs = [epts_path(target, game, part) for game in games for part in EPTS_PARTS]
    if any(path.exists() for path in outputs):
        raise FileExistsError("degraded files already exist in {}; remove them first".format(target))
    target.mkdir(parents=True, exist_ok=True)
    scale = np.asarray(PITCH)
    for game in games:
        lines = epts_path(source, game, EPTS_PARTS[0]).read_text().splitlines()
        offsets = smooth_drift(len(lines), drift_sd_m, smooth_s, np.random.default_rng(seed + game)) / scale
        out = []
        for line, offset in zip(lines, offsets):
            head, ball = line.rsplit(":", 1)
            xy = np.array([float(value) for value in ball.split(",")[:2]])
            if np.isfinite(xy).all():
                ball = ",".join(format(value, ".5f") for value in xy + offset)
            out.append(head + ":" + ball)
        epts_path(target, game, EPTS_PARTS[0]).write_text("\n".join(out) + "\n")
        for part in EPTS_PARTS[1:]:
            shutil.copyfile(epts_path(source, game, part), epts_path(target, game, part))
    return target


def _read_tracking(path):
    with open(path, newline="") as handle:
        rows = csv.reader(handle)
        next(rows), next(rows)
        header = next(rows)
        players = [i for i, name in enumerate(header) if name.startswith("Player")]
        ball = header.index("Ball")
        frames, period, xy_players, xy_ball = [], [], [], []
        for row in rows:
            frames.append(int(row[1]))
            period.append(int(row[0]))
            xy_players.append([(float(row[i]), float(row[i + 1])) for i in players])
            xy_ball.append((float(row[ball]), float(row[ball + 1])))
    frames = np.asarray(frames)
    if not (frames == np.arange(1, len(frames) + 1)).all():
        raise ValueError("{}: frames must run 1..n without gaps".format(path))
    scale = np.asarray(PITCH)
    return np.asarray(period), np.asarray(xy_players) * scale, np.asarray(xy_ball) * scale


def _load_epts(data_dir, game):
    root = ET.parse(epts_path(data_dir, game, EPTS_PARTS[1])).getroot()
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    local = root.find(".//Score").get("idLocalTeam")
    slots = {p.get("id"): ("H" if p.get("teamId") == local else "A") for p in root.iter("Player")}
    order = sorted(slots, key=lambda pid: (slots[pid] != "H", pid))
    column = {pid: i for i, pid in enumerate(order)}
    channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    text = epts_path(data_dir, game, EPTS_PARTS[0]).read_text().replace(":", ";").replace(";", ",")
    rows = np.array([line.split(",") for line in text.splitlines()], dtype=float)
    n = len(rows)
    if not (rows[:, 0] == np.arange(1, n + 1)).all():
        raise ValueError("game {}: frames must run 1..n without gaps".format(game))
    players = np.full((n, len(order), 2), np.nan)
    for spec in root.iter("DataFormatSpecification"):
        lo, hi = int(spec.get("startFrame")) - 1, int(spec.get("endFrame"))
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        ids = [channel[ref] for ref in refs[::2]]
        for k, pid in enumerate(ids):
            players[lo:hi, column[pid]] = rows[lo:hi, 1 + 2 * k:3 + 2 * k]
    scale = np.asarray(PITCH)
    ball = rows[:, -2:] * scale
    period = np.where(np.arange(1, n + 1) >= int(params["second_half_start"]), 2, 1)
    return ball, players * scale, np.array([slots[pid] for pid in order]), period


def read_game_events(data_dir, game):
    """Events in the CSV schema used by event_labels (Type, Subtype, Team, Period, start, end)."""
    if game not in EPTS_GAMES:
        return _read_events(file_path(data_dir, game, PARTS[2]))
    payload = json.loads(epts_path(data_dir, game, EPTS_PARTS[2]).read_text())
    local = ET.parse(epts_path(data_dir, game, EPTS_PARTS[1])).getroot().find(".//Score").get("idLocalTeam")
    events = []
    for event in payload["data"]:
        sub = event.get("subtypes")
        sub = sub if isinstance(sub, list) else [sub] if sub else []
        team = event.get("team") or {}
        events.append({
            "Type": event["type"]["name"], "Subtype": "-".join(item["name"] for item in sub),
            "Team": "Home" if team.get("id") == local else "Away" if team.get("id") else "",
            "Period": str(event["period"]), "start": int(event["start"]["frame"]),
            "end": int(event["end"]["frame"] or 0),
        })
    return events


def load_game(data_dir, game):
    """(ball (n, 2), players (n, slots, 2), teams, period) in metres, row = frame - 1."""
    if game in EPTS_GAMES:
        return _load_epts(data_dir, game)
    period, home, ball = _read_tracking(file_path(data_dir, game, PARTS[0]))
    period_away, away, _ = _read_tracking(file_path(data_dir, game, PARTS[1]))
    if len(period_away) != len(period):
        raise ValueError("home and away tracking lengths differ")
    players = np.concatenate([home, away], axis=1)
    teams = np.array(["H"] * home.shape[1] + ["A"] * away.shape[1])
    return ball, players, teams, period


def _read_events(path):
    with open(path, newline="") as handle:
        return [dict(row, start=int(row["Start Frame"]), end=int(row["End Frame"]))
                for row in csv.DictReader(handle)]


def _is_restart(event):
    kind, sub = event["Type"], event["Subtype"]
    return kind == "SET PIECE" or ("GOAL KICK" in sub and kind in ("PASS", "BALL LOST"))


def _stops_play(event):
    kind, sub = event["Type"], event["Subtype"]
    if kind in ("BALL OUT", "FAULT RECEIVED"):
        return True
    if kind == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return True
    return kind == "BALL LOST" and sub in ("OFFSIDE", "END HALF")


def event_labels(events, n, period):
    """Per-frame H/A/D, X outside each period's first and last event."""
    events = sorted(events, key=lambda e: e["start"])
    team = np.full(n, "X", dtype="<U1")
    owning = [e for e in events if e["Type"] in OWNING_EVENTS and e["Team"] in ("Home", "Away")]
    marks = [(e["start"] - 1, "H" if e["Team"] == "Home" else "A") for e in owning]
    for (frame, side), following in zip(marks, marks[1:] + [(n, None)]):
        team[max(0, frame):min(n, following[0])] = side
    restarts = [e["start"] - 1 for e in events if _is_restart(e)]
    for event in events:
        if not _stops_play(event):
            continue
        begin = (event["end"] if event["end"] > 0 else event["start"]) - 1
        later = [frame for frame in restarts if frame > begin]
        stop = min(later) if later else n
        team[max(0, begin):min(n, stop)] = "D"
    graded = np.zeros(n, dtype=bool)
    for half in np.unique(period):
        inside = [e for e in events if e["Period"] == str(half)]
        if inside:
            first = min(e["start"] for e in inside) - 1
            last = max(max(e["start"], e["end"]) for e in inside)
            graded[max(0, first):min(n, last)] = True
    return np.where(graded, team, "X")


def score(config_path, data_dir=DEFAULT_DIR, games=ALL_GAMES):
    config = load_viterbi_config(config_path)
    totals = {}
    for game in games:
        ball, players, teams, period = load_game(data_dir, game)
        truth = event_labels(read_game_events(data_dir, game), len(ball), period)
        labels = decode_possession(ball, players, teams, fps=FPS, pitch=PITCH, config=config).labels
        result = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels), bootstrap_n=0)
        totals[game] = {
            "spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
            "possessions_pred": result["n_pred_spells"], "possessions_true": result["n_gt_spells"],
            "f1_D": result["per_class"]["D"]["f1"],
            "ball_visible": float(np.isfinite(ball).all(axis=1).mean()),
        }
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", default=[], help="viterbi JSON; repeatable")
    parser.add_argument("--data-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--download", action="store_true", help="fetch the public sample files first")
    parser.add_argument("--make-degraded", action="store_true",
                        help="write ball-drift copies of --data-dir into {}".format(DEGRADED_DIR))
    parser.add_argument("--degraded", action="store_true", help="score on the degraded copies")
    args = parser.parse_args(argv)
    if args.download:
        download(args.data_dir)
    if args.make_degraded:
        if not all(file_path(DEGRADED_DIR, g, p).exists() for g in GAMES for p in PARTS):
            make_degraded(args.data_dir, DEGRADED_DIR)
        if not all(epts_path(DEGRADED_DIR, g, p).exists() for g in EPTS_GAMES for p in EPTS_PARTS):
            make_degraded_epts(args.data_dir, DEGRADED_DIR)
        print("degraded copies written to {}".format(DEGRADED_DIR))
    if args.degraded:
        args.data_dir = str(DEGRADED_DIR)
    print("{:<28} {:>4} {:>8} {:>8} {:>6} {:>11} {:>7}".format(
        "config", "game", "spellF1", "macroF1", "f1_D", "poss p/t", "ballVis"))
    for path in args.config:
        for game, row in score(path, args.data_dir).items():
            print("{:<28} {:>4} {:>8.4f} {:>8.4f} {:>6.3f} {:>5}/{:<5} {:>7.1%}".format(
                Path(path).name, game, row["spell_f1"], row["macro_f1"], row["f1_D"],
                row["possessions_pred"], row["possessions_true"], row["ball_visible"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
