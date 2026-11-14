"""Generate tiny ATD-shaped files from made-up positions."""

from pathlib import Path

import numpy as np

from games import asset_paths


def write_tracking_fixture(root, ball, players, *, pitch=(105.3, 68.), teams=("H", "A")):
    paths = asset_paths(data_dir=str(root), source="tactical")
    meta, raw = Path(paths["atd_meta"]), Path(paths["atd_raw"])
    meta.parent.mkdir(parents=True, exist_ok=True)
    slots = "".join('<Player id="{}" teamId="{}"/>'.format(i + 1, team) for i, team in enumerate(teams))
    meta.write_text('<root><Team id="H"><Name>Home</Name></Team>'
                    '<Team id="A"><Name>Away</Name></Team>' + slots + '</root>', encoding="utf-8")

    def group(xy):
        # The documented raw format stores normalised x,y and a speed field.
        # This fixture intentionally leaves speeds at zero: our pipeline derives
        # motion from positions rather than trusting that field.
        values = np.asarray(xy) / pitch
        return ",".join("NaN" if np.isnan(v) else format(v, ".17g") for v in values) + ",0"

    lines = ["{}:{}:{}\n".format(frame + 1, ";".join(group(xy) for xy in players[frame]), group(xy))
             for frame, xy in enumerate(ball)]
    raw.write_text("".join(lines), encoding="utf-8")
    return meta, raw


def dribble_fixture(frames=6):
    ball = np.column_stack((2 + np.arange(frames) * 0.1, np.full(frames, 3.)))
    players = np.stack((ball + [0.2, 0], ball + [10, 0]), axis=1)
    return ball, players


def write_tagging_fixture(root, *, start_frame, end_frame, fps=25, kind="Free Kick"):
    """SportsCode-shaped automatic tag, solely for synthetic integration tests."""
    from xml.sax.saxutils import escape

    path = Path(asset_paths(data_dir=str(root))["smart_tagging"])
    path.write_text(
        '<file><ALL_INSTANCES><instance><start>{}</start><end>{}</end>'
        '<code>SET PIECES</code><label><group>Type</group><text>{}</text></label>'
        '<label><group>Team</group><text>N/A</text></label>'
        '</instance></ALL_INSTANCES></file>'.format(start_frame / fps, end_frame / fps, escape(kind)),
        encoding="utf-8")
    return path
