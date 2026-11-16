"""Train the learned-evidence model on Metrica's sample games and save it as JSON.

    python -m possession.validate_metrica --download --make-degraded
    python -m possession.train_learned --output possession/models/learned_v1.json
    python -m possession.train_learned --recipe v2 --output possession/models/learned_v2.json
    python -m possession.idsse --download --parse
    python -m possession.train_learned --recipe v3 --output possession/models/learned_v3.json

Prints leave-one-game-out scores (train on the other games, score the held-out one)
before training the saved model on all of them. Deterministic: the same data gives the
same file.

Recipes:
  v1  logistic regression, degraded games 1-2, every frame (the original learned_v1).
  v2  context columns (+-0.5/1/2 s) and an ensemble of five small networks, trained on
      games 1-3 in both their clean and degraded copies (every 5th frame), decoded with
      scale 0.2, switch 2.0, dead switch 7.0. Held-out scores are given for the degraded
      and the clean copy of each game; a held-out game never appears in training in
      either copy.
  v3  the seven IDSSE Bundesliga matches (possession.idsse), whose labels come from the
      provider's per-frame ball-alive and possession flags -- the same derivation as the
      challenge's ground truth. Ball hidden where the provider froze it, plus live-play
      dropouts (idsse.observed_ball); every match is used twice, as observed and with the
      degraded-copy ball drift added (seed 100 + match index), so the model does not rely
      on a ball that sits on the player. v2's features plus control_features, context of
      sixteen columns at +-0.12 s to 2 s, five networks of 128 units, decoded with scale
      0.15, switch 1.0, dead switch 7.0. Held-out scores are per IDSSE match.
  v4  v3 plus two more copies of every IDSSE match whose player tracks are damaged like
      ours (degrade.damage_players: track breaks and identity switches calibrated to our
      match's aggregate tracking statistics), one with the observed ball and one with the
      drifted ball. The labels -- the provider's flags on clean tracking -- are unchanged,
      which is the real test's situation: clean-tracking answers, messy-tracking input.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from . import degrade
from . import idsse
from . import learned
from . import validate_metrica as vm
from . import viterbi as vt

RECIPES = {
    "v1": dict(games=vm.GAMES, folders=("degraded",), context=False, scale=0.2, switch_cost=3.5,
               dead_switch_cost=7.0),
    "v2": dict(games=vm.ALL_GAMES, folders=("clean", "degraded"), context=True, scale=0.2,
               switch_cost=2.0, dead_switch_cost=7.0),
    "v3": dict(folders=("idsse", "idsse_drift"), context=True, control=True, hidden=128, scale=0.15,
               switch_cost=1.0, dead_switch_cost=7.0),
    "v4": dict(folders=("idsse", "idsse_drift", "idsse_damaged", "idsse_drift_damaged"), context=True,
               control=True, hidden=128, scale=0.15, switch_cost=1.0, dead_switch_cost=7.0),
}


def load_game(folder, game, config):
    ball, players, teams, period = vm.load_game(folder, game)
    truth = vm.event_labels(vm.read_game_events(folder, game), len(ball), period)
    halves = [(int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1]) + 1)
              for p in np.unique(period)]
    features, names = learned.frame_features(ball, players, teams, fps=vm.FPS, pitch=vm.PITCH,
                                             halves=halves, config=config)
    graded = truth != "X"
    labels = np.array([vt.STATES.index(state) for state in truth[graded]])
    return features, names, truth, features[graded], labels


def load_idsse(match, config, drift_seed=None, damage_seed=None):
    ball, players, teams, period, truth = idsse.load_match(idsse.DATA_DIR, match)
    ball = idsse.observed_ball(ball)
    if damage_seed is not None:
        players = degrade.damage_players(players, seed=damage_seed)
    if drift_seed is not None:
        ball = ball + vm.smooth_drift(len(ball), vm.DRIFT_SD_M, vm.DRIFT_SMOOTH_S, np.random.default_rng(drift_seed))
    halves = [(int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1]) + 1)
              for p in np.unique(period)]
    features, names = learned.frame_features(ball, players, teams, fps=idsse.FPS, pitch=idsse.PITCH,
                                             halves=halves, config=config)
    extra, extra_names = learned.control_features(ball, players, teams, fps=idsse.FPS)
    return np.c_[features, extra], names + extra_names, truth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", choices=sorted(RECIPES), default="v1")
    parser.add_argument("--data-dir", help="v1 only; defaults to the degraded copies")
    parser.add_argument("--viterbi-config", default="possession/configs/viterbi_v4.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        print("FAIL: {} already exists; choose a new path".format(output), file=sys.stderr)
        return 1
    recipe = RECIPES[args.recipe]
    config = vt.load_viterbi_config(args.viterbi_config)
    dirs = {"clean": str(vm.DEFAULT_DIR), "degraded": args.data_dir or str(vm.DEGRADED_DIR),
            "idsse": str(idsse.DATA_DIR)}
    if recipe["folders"][0] == "idsse":
        recipe = dict(recipe, games=tuple(idsse.match_ids()))
        if not recipe["games"]:
            print("FAIL: no IDSSE matches in {}; run python -m possession.idsse --download".format(
                idsse.DATA_DIR), file=sys.stderr)
            return 1
    base, offsets = ((learned.CONTEXT_BASE_V3, learned.CONTEXT_OFFSETS_V3) if recipe.get("control")
                     else (learned.CONTEXT_BASE, learned.CONTEXT_OFFSETS))
    data = {}
    for kind in recipe["folders"]:
        for index, game in enumerate(recipe["games"]):
            if kind.startswith("idsse"):
                features, names, truth = load_idsse(
                    game, config, 100 + index if "drift" in kind else None,
                    {"idsse_damaged": 300 + index, "idsse_drift_damaged": 400 + index}.get(kind))
                labels = np.array([vt.STATES.index(state) for state in truth[truth != "X"]])
            else:
                features, names, truth, _, labels = load_game(dirs[kind], game, config)
            if recipe["context"]:
                features, context_names = learned.add_context(features, names, base, offsets)
            data[kind, game] = (features, truth, features[truth != "X"], labels)
    context_names = context_names if recipe["context"] else names
    decode = {key: recipe[key] for key in ("scale", "switch_cost", "dead_switch_cost")}

    def train(games):
        x = np.concatenate([data[kind, g][2] for kind in recipe["folders"] for g in games])
        y = np.concatenate([data[kind, g][3] for kind in recipe["folders"] for g in games])
        if recipe["context"]:
            return learned.fit_ensemble(x, y, context_names, hidden=recipe.get("hidden", 64))
        return learned.fit_symmetric(x, y, names)

    for test in recipe["games"]:
        model = train([g for g in recipe["games"] if g != test])
        for kind in recipe["folders"]:
            features, truth = data[kind, test][:2]
            labels = learned.decode(model, features, **decode)
            result = vm.pm.evaluate(vm.pm.label_codes(truth), vm.pm.label_codes(labels), bootstrap_n=0)
            print("held-out game {} ({}): spellF1 {:.4f}  macroF1 {:.4f}  possessions {}/{}".format(
                test, kind, result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
                result["n_gt_spells"]))
    model = train(recipe["games"])
    output.parent.mkdir(parents=True, exist_ok=True)
    tail = "; labels from their events (validate_metrica.event_labels); every frame also used with teams swapped"
    if args.recipe == "v1":  # wording kept identical so learned_v1.json regenerates byte for byte
        notes = {"training_data": "Metrica sample games 1 and 2 from {}".format(dirs["degraded"]) + tail}
    elif args.recipe in ("v3", "v4"):
        notes = {"recipe": args.recipe, "training_data": "IDSSE open data (CC BY 4.0, doi 10.6084/m9.figshare.28196177) "
                 "matches {} from {}; labels from the provider's BallStatus/BallPossession flags "
                 "(idsse.labels); ball observed as idsse.observed_ball, each match also with the degraded-copy drift "
                 "(seed 100 + index){}; every frame also used with teams "
                 "swapped".format(", ".join(recipe["games"]), dirs["idsse"], "" if args.recipe == "v3" else
                                 "; v4 adds copies with player tracks damaged by degrade.damage_players "
                                 "(seeds 300 + index, and 400 + index with the drift)")}
    else:
        sources = " and ".join("{} ({})".format(dirs[kind], kind) for kind in recipe["folders"])
        notes = {"recipe": args.recipe, "training_data": "Metrica sample games {} from {}".format(
            ", ".join(map(str, recipe["games"])), sources) + tail}
    learned.save_model(output, model, names=names, config=config, notes=notes, context=(base, offsets) if recipe["context"] else None,
                       control=recipe.get("control", False), **decode)
    print("saved {}".format(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
