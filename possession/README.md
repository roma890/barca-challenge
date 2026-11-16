# Possession inference

Frame-by-frame H/A/D labels for the provided match, produced by an algorithm trained and
validated only on **other matches**. The final model is **learned_v4**.

## Reproduce the submission

```bash
python -m possession.idsse --download --parse        # IDSSE open data, ~2.6 GB into data/external/idsse
python -m possession.train_learned --recipe v4 --output possession/models/learned_v4.json
python -m possession.run_learned --model possession/models/learned_v4.json \
    --output evaluation/results/learned_v4.csv
```

The committed `models/learned_v4.json` is what `run_learned` reads; retraining reproduces
it byte for byte, and `run_learned` reproduces the CSV byte for byte (same machine and
numpy build). Both commands refuse to overwrite existing files.

## How learned_v4 works

1. **Frame features** (`learned.frame_features`, `learned.control_features`): which team's
   players are nearest the ball and how far, relative ball/player motion, players of each
   team near the ball, team movement and shape, ball position along each team's attack
   direction, and dead-ball cues (players standing still, ball outside or leaving the pitch,
   set-ups at a line, ball speed, ball visibility). Sixteen of them are repeated at
   +-0.12 s to 2 s for context.
2. **Frame probabilities**: five one-hidden-layer networks (128 units) give P(H/A/D). Every
   training frame is also used with home and away swapped, and predictions are averaged
   with their mirrored version, so neither team is favoured.
3. **Decoding** (`viterbi._viterbi`): the most consistent label sequence over the whole
   match, with a cost for changing team (1.0) and for entering or leaving dead ball (7.0).

## Training data (other matches only)

| Dataset | Used for | Labels |
|---|---|---|
| **IDSSE** (Bassek et al. 2025, *Scientific Data*; 7 Bundesliga matches; CC BY 4.0; doi 10.6084/m9.figshare.28196177) | training and validation of learned_v3/v4 | the provider's per-frame BallStatus and BallPossession flags (`idsse.labels`), the same kind of derivation CHALLENGE.md describes for the ground truth |
| **Metrica Sports sample games 1-3** (open data) | validation, and training of learned_v1/v2 | derived from their event data by a fixed rule (`validate_metrica.event_labels`) |
| SkillCorner open data (20 A-League matches, MIT) | tried, rejected | possession labels follow a different convention; adding them lowered held-out IDSSE from 0.705 to 0.601 |

The learned models were trained and selected without labels for the provided match, and no human labelled anything.

## Making training data look like our tracking

The provided tracking is automatic and uncorrected. Training copies of IDSSE are degraded to
match **match-wide summary statistics** of the provided tracking, computed by scripts with no
labels and nothing tied to specific frames:

| Statistic (provided match, graded halves) | Value | Reproduced by |
|---|---|---|
| Ball visible | 59% | `idsse.observed_ball`: ball hidden where the provider froze it (all dead frames) plus live-play dropouts at Metrica's measured rate |
| Ball jitter / distance to nearest player | 0.23 m/s / 1.9 m | `validate_metrica.smooth_drift`: smooth 1.2 m drift |
| Players tracked per frame | 20.5 of 22 | `degrade.damage_players`: track breaks |
| Track piece lengths | median 35 s (p10 2.6 s) | track breaks drawn from these quantiles |
| Implausible moves (>12 m/s) | ~15 per minute | identity swaps with the nearest player |

learned_v4 trains on four copies of each IDSSE match: as observed, with the ball drift, with
damaged player tracks, and both. Labels are untouched, which mirrors the real test: answers
from corrected tracking, input from uncorrected tracking.

## Validation and results

Leave-one-match-out on IDSSE (train on 6, score the 7th, for all 7), plus Metrica, which
learned_v3/v4 never saw. "Like ours" = damaged players and drifted ball. Learned-model choices were
made on these numbers only; the public five minutes were scored as a correctness check.

| | v4 rules (Viterbi) | learned_v1 | learned_v2 | learned_v3 | **learned_v4** |
|---|---|---|---|---|---|
| IDSSE held out, like ours | - | - | - | ~0.50 | **0.635** |
| IDSSE held out, clean players | - | - | 0.612 | 0.708 | 0.701 |
| Metrica degraded | 0.455 | 0.562 | 0.675* | 0.685 | 0.682 |
| Metrica clean | 0.544 | 0.612 | 0.713* | 0.740 | 0.736 |
| Public window (check only) | 0.447 | 0.469 | 0.568 | 0.544 | 0.667 |

spellF1; * held out within Metrica (learned_v2 was trained on it). The public window is one
of the cleanest stretches of the match (ball visible 88% vs 60% typical), so the full match
is expected to score lower, close to the held-out "like ours" figure.

## What was tried

| Change | Held-out effect | Kept |
|---|---|---|
| Learned frame evidence instead of hand-written rules (learned_v1) | Metrica degraded 0.455 -> 0.562 | yes |
| Context window and small networks (learned_v2) | 0.562 -> 0.675 | yes |
| Train on IDSSE instead of Metrica (learned_v3) | IDSSE 0.612 -> 0.708 | yes |
| Control features, finer context, 128 units | +0.013 | yes |
| Ball drift augmentation | drifted IDSSE +0.056, degraded Metrica +0.079, clean -0.010 | yes |
| Simulated player-track damage (learned_v4) | like ours +0.13, 7/7 matches; clean -0.012 | yes (see note) |
| Minimum spell length in the decoder | never better | no |
| Self-training on the unlabelled test match | +0.008 at best | no |
| Second-stage model on first-stage probabilities | +0.002 to +0.005 | no |
| Temporal convolution network | 0.589 vs 0.622 | no |
| Turnover detector setting a per-frame switch cost | +0.005 | no (below the +0.01 bar) |
| Training on position-hop damage instead of swaps | +0.008 on hops, -0.06 on swaps | no |
| Robust velocities and track-age features | no change | no |
| SkillCorner as extra training data | -0.10 on IDSSE | no |
| Smart Tagging as evidence | not measurable on other matches | no |

Note on learned_v4: the keep rule fixed beforehand also required clean-input IDSSE to drop by
no more than 0.01; it dropped 0.012. That condition never occurs in the real test (the input is
always uncorrected tracking), so the model was adopted and the deviation is recorded here.

Diagnostics on held-out IDSSE "like ours" (which errors cost most): perfect stoppage timing
would add +0.008, correct teams +0.06, perfect possession boundaries +0.11. Remaining error is
mostly the timing of possession changes; possessions under 2 s are matched about 25% of the time.

## Files

| File | Purpose |
|---|---|
| `learned.py` | features, networks, decoding, model files |
| `train_learned.py` | training recipes v1-v4 with held-out scores |
| `run_learned.py` | writes the submission CSV from a saved model |
| `idsse.py`, `validate_metrica.py`, `skillcorner.py` | external data loaders and label rules |
| `degrade.py` | simulated player-tracking damage |
| `viterbi.py`, `run_viterbi.py`, `configs/viterbi*.json` | rule-based Viterbi pipeline (v1-v6), superseded |
| `state.py`, `control.py`, `contacts.py`, `interactions.py`, `play_state.py`, `stoppages.py`, `resumption.py`, `dead_episodes.py`, `pipeline.py`, `run.py` | earlier rule-based pipeline (prototype_v1); see EXPERIMENTS.md and module docstrings |
