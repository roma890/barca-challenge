"""Local possession prototype with soft temporal ball-in-play evidence.

HISTORICAL: superseded by learned_v6 (possession.run_learned); not used for the submission.

Run from the repository root with python -m possession.run --help.
Only aggregate diagnostics are printed. Automatic tagging is read only with
--smart-tagging. No video, homography, ground truth or reports are read.
Predictions stay in the selected local CSV.
"""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from games import GAME, GAMES, require, source_frames
from starter.load_smart_tagging import load as load_tagging

from .pipeline import load_config, predict, write_validated_submission
from .tracking import load_tracking


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="explicit pipeline JSON settings")
    parser.add_argument("--data-dir", help="bundle root; defaults to games.DATA_DIR")
    parser.add_argument("--output", required=True, help="new local frame,label CSV (use evaluation/results/)")
    parser.add_argument("--report", help="optional aggregate JSON (use evaluation/reports/)")
    parser.add_argument("--smart-tagging", action="store_true",
                        help="read automatic Smart Tagging as soft live/dead context")
    parser.add_argument("--frames", type=int, default=source_frames("tactical"),
                        help="expected records; override only for synthetic fixtures (default: official count)")
    args = parser.parse_args(argv)
    try:
        # Fail before opening assets if an output would overwrite an input,
        # prior submission or report. Output paths must also be distinct.
        output = Path(args.output).resolve()
        report = Path(args.report).resolve() if args.report else None
        if output.exists() or (report is not None and report.exists()):
            raise ValueError("output/report already exists; choose new paths")
        if report == output:
            raise ValueError("submission and report need different paths")
        config = load_config(args.config)
        if args.smart_tagging and config.stoppages is None and config.play_state is None:
            raise ValueError("--smart-tagging requires stoppages or play_state settings")
        meta, raw = require(["atd_meta", "atd_raw"], data_dir=args.data_dir, source="tactical")
        tracking = load_tracking(meta, raw, n_frames=args.frames,
                                 pitch=GAMES[GAME]["pitch"], fps=GAMES[GAME]["fps"])
        try:
            events = load_tagging(data_dir=args.data_dir, fps=tracking.fps) if args.smart_tagging else None
        except (ET.ParseError, TypeError, ValueError):
            raise ValueError("automatic Smart Tagging file could not be parsed") from None
        prediction = predict(tracking, config, events=events)
        write_validated_submission(output, prediction.state.labels, n_frames=args.frames)
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open("x", encoding="utf-8") as handle:
                json.dump(prediction.summary, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
        print("Prototype: automatic tagging supplies soft live/dead context; predictions remain hypotheses."
              if args.smart_tagging else
              "Prototype: no Smart Tagging supplied; only tracking/restart evidence is available.")
        print("Official CSV format validation passed for {} rows. Accuracy has not been evaluated."
              .format(args.frames))
        print(json.dumps(prediction.summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError) as error:
        print("FAIL: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
