"""Write a submission from the offline H/A/D Viterbi decoder.

HISTORICAL: these hand-written rule configs (configs/viterbi*.json) are superseded by
learned_v6 (possession.run_learned), which reuses only the decoder, viterbi._viterbi.

Run from the repository root with python -m possession.run_viterbi --help.
Prints aggregate diagnostics only; no tagging, video or ground truth is read.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from games import GAME, GAMES, require, source_frames

from .pipeline import write_validated_submission
from .tracking import load_tracking
from .viterbi import decode_possession, load_viterbi_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="viterbi JSON settings")
    parser.add_argument("--data-dir", help="bundle root; defaults to games.DATA_DIR")
    parser.add_argument("--output", required=True, help="new frame,label CSV (use evaluation/results/)")
    parser.add_argument("--report", help="optional aggregate JSON (use evaluation/reports/)")
    parser.add_argument("--frames", type=int, default=source_frames("tactical"),
                        help="expected records; override only for synthetic fixtures")
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).resolve()
        report = Path(args.report).resolve() if args.report else None
        if output.exists() or (report is not None and report.exists()):
            raise ValueError("output/report already exists; choose new paths")
        config = load_viterbi_config(args.config)
        meta, raw = require(["atd_meta", "atd_raw"], data_dir=args.data_dir, source="tactical")
        tracking = load_tracking(meta, raw, n_frames=args.frames,
                                 pitch=GAMES[GAME]["pitch"], fps=GAMES[GAME]["fps"])
        result = decode_possession(tracking.ball_xy_m, tracking.player_xy_m, tracking.teams,
                                   fps=tracking.fps, pitch=tracking.pitch, config=config)
        write_validated_submission(output, result.labels, n_frames=args.frames)
        summary = dict(result.diagnostics, config=asdict(config))
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open("x", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print("Official CSV format validation passed for {} rows.".format(args.frames))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as error:
        print("FAIL: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
