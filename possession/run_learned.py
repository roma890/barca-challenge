"""Write a submission from a saved learned-evidence model.

    python -m possession.run_learned --model possession/models/learned_v1.json \
        --output evaluation/results/learned_v1.csv
    python -m possession.run_learned --model possession/models/learned_v2.json \
        --output evaluation/results/learned_v2.csv
    python -m possession.run_learned --model possession/models/learned_v3.json \
        --output evaluation/results/learned_v3.csv

Prints aggregate counts only; no tagging, video or ground truth is read.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames

from . import learned
from .pipeline import write_validated_submission
from .tracking import load_tracking


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", help="bundle root; defaults to games.DATA_DIR")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=source_frames("tactical"))
    parser.add_argument("--refine-boundaries", action="store_true",
                        help="EXPERIMENTAL: re-time H<->A boundaries with possession.boundary_refine "
                             "(off by default; the frozen submission does not use it)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if Path(args.output).exists():
            raise ValueError("output already exists; choose a new path")
        model = learned.load_model(args.model)
        meta, raw = require(["atd_meta", "atd_raw"], data_dir=args.data_dir, source="tactical")
        tracking = load_tracking(meta, raw, n_frames=args.frames,
                                 pitch=GAMES[GAME]["pitch"], fps=GAMES[GAME]["fps"])
        second_half = half_video_blocks(GAME)[1][1]
        features = learned.model_inputs(
            model, tracking.ball_xy_m, tracking.player_xy_m, tracking.teams, fps=tracking.fps,
            pitch=tracking.pitch, halves=[(0, second_half), (second_half, args.frames)])
        labels = learned.decode(model, features, scale=model["scale"], switch_cost=model["switch_cost"],
                                dead_switch_cost=model["dead_switch_cost"])
        if args.refine_boundaries:
            from . import boundary_refine
            labels = boundary_refine.refine(labels, tracking.ball_xy_m, tracking.player_xy_m,
                                            tracking.teams, fps=tracking.fps)
        write_validated_submission(args.output, labels, n_frames=args.frames)
        print("Official CSV format validation passed for {} rows.".format(args.frames))
        print(json.dumps({state: int(np.sum(labels == state)) for state in ("H", "A", "D")}))
        return 0
    except (OSError, ValueError) as error:
        print("FAIL: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
