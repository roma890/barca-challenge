# Possession from automatic tracking: technical document

**Team:** Tiki Taka Tech

**Submission:** `evaluation/results/learned_v6.csv` (model `possession/models/learned_v6.json`).
**Fallback kept in the repository:** learned_v4.

## 1. Summary

We label every frame `H`, `A` or `D` from the automatic tracking (ball and players) alone. A small
neural-network ensemble turns each frame and its surroundings (±2 s) into probabilities for the
three states, and a Viterbi decoder picks the most consistent sequence over the match. The networks
are trained only on **other matches**: seven Bundesliga matches from the open IDSSE dataset, whose
labels come from the provider's per-frame ball-alive and possession flags, which is the same kind
of derivation the challenge's ground truth uses. Because our input is uncorrected automatic
tracking while those matches are clean, the training copies are degraded so that their summary
statistics match those of the provided tracking. Learned_v6 scores 0.646 / 0.615 spellF1 on
held-out IDSSE matches degraded that way (two damage types), against 0.24 for the published baseline
on the challenge match.

## 2. Approach and why

**Why learned rather than hand-written rules.** Our first pipelines were hand-written rules (nearest
player, contact detection, stoppage and restart detectors, then a rule-based Viterbi). On external
matches with labels they reached 0.455 spellF1 (Metrica sample games, degraded); replacing the
hand-written evidence with learned evidence lifted the same test to 0.56 and, with context and small
networks, 0.68. Rules needed many thresholds we could not set without labelled data; a model
trained on labelled other matches sets them from data.

**Frame features** (`possession/learned.py`): which team's players are nearest the ball and by how
much, ball motion relative to the nearest players, number of players of each team near the ball,
team movement and shape, the ball's position along each team's attack direction, and dead-ball cues
(players standing still, ball outside or leaving the pitch, players setting up at a line, ball
speed, whether the ball is tracked). Sixteen of them are repeated at offsets from -2 s to +2 s
(0.12 to 2 s), giving 193 inputs per frame.

**Frame probabilities.** Five one-hidden-layer networks (128 ReLU units, about 126k parameters in
total), trained with home and away swapped on every training frame and averaged with the swapped
prediction at test time, so neither team is favoured. numpy only, no GPU.

**Decoding.** A three-state Viterbi over the whole match (`possession/viterbi.py`): frame scores are
0.15 × (log-probability - log prior), switching team costs 1.0 and entering or leaving dead ball
costs 7.0. The decoder is what turns noisy frame evidence into whole possessions.

## 3. Assets used and dropped

| Asset | Use |
|---|---|
| Automatic tracking (ball and players) | **the only input** to the submitted model |
| Smart Tagging | **dropped.** Tags are phase windows of 15-35 s (longer than a typical possession), with 24 transition tags for the whole match. Simulated on held-out IDSSE with the same counts and window lengths, even *error-free* tags improved spellF1 by at most +0.002, and any stronger weight lowered it |
| Tactical / broadcast video | not used |
| Homography | not used (tracking is already in pitch coordinates) |
| Public 5-minute ground truth | correctness check only: frame alignment, file format, label meaning |

## 4. Training data

No labels from the provided match are used anywhere, and no person labelled anything.

| Dataset | Source and licence | Labels: who and how | Amount | Role |
|---|---|---|---|---|
| **IDSSE** | Bassek et al. 2025, *Scientific Data*, doi 10.6084/m9.figshare.28196177, CC BY 4.0 | the data provider's per-frame `BallStatus` and `BallPossession` flags (`possession/idsse.py: labels`) | 7 Bundesliga matches | training (learned_v3 onwards) and leave-one-match-out validation |
| **Metrica Sports sample games 1-3** | public GitHub repository metrica-sports/sample-data; three other full matches, **external sample data, not the challenge data** | derived from their open event data by a fixed rule (`possession/validate_metrica.py: event_labels`) | 3 matches | validation; training of the early models learned_v1/v2 only |
| SkillCorner open data | 20 A-League matches, MIT | the provider's possession field | tried | **dropped**: different convention; adding it lowered held-out IDSSE from 0.705 to 0.601 |

## 5. Handling missing and noisy tracking

The test input is uncorrected automatic tracking; the ground truth comes from corrected tracking.
Training copies of IDSSE are therefore degraded to look like the provided tracking, while their
labels stay as they are. That is the real test's situation: clean answers, messy input. The
degradation levels come from **label-free, match-wide summary statistics** of the provided tracking,
computed by script (rates, percentages and percentiles over the whole match, nothing tied to
individual frames):

| Statistic of the provided tracking | Value | How the IDSSE training copies reproduce it |
|---|---|---|
| Ball tracked | 59% of graded frames | ball hidden where the provider froze it (dead ball) plus live-play dropouts at the rate measured on the Metrica sample games (`idsse.observed_ball`) |
| Ball jitter, distance ball to nearest player | 0.23 m/s, 1.9 m | slow random drift of the ball, 1.2 m (`validate_metrica.smooth_drift`) |
| Sharpness of ball velocity changes (0.12 s) | p90 4.3, p99 8.9 m/s (IDSSE: 7.4, 17.9) | ball smoothed in time, 0.24 s Gaussian (`degrade.smooth_ball`) |
| Players tracked per frame | 20.5 of 22 | track breaks, mean gap 6.5 s (`degrade.damage_players`) |
| Track piece lengths | median 35 s, p10 2.6 s | piece lengths drawn from these quantiles |
| Implausible moves (> 12 m/s) | about 15 per minute | identity swaps with a nearby player, 9 per minute |

Each IDSSE match enters training four times: as observed, with ball drift, with damaged players, and
with both, all with the smoothed ball. The provided match itself is never altered. Inside the model,
missing values are handled explicitly: features carry a ball-visible flag, player features use the
players that are tracked, and the decoder carries the last state through stretches without evidence.

The results do not hinge on these exact levels: halving or doubling the training damage, or the
hand-set feature constants (a 1 m margin and 3 m reach around the ball, a 1 s window, and
1.3 m/s, a typical walking speed, used as the neutral value when no player speeds are
available), moves held-out spellF1 by at most ±0.005, except doubling the damage,
which helps one test (+0.020 swap) and leaves the other flat (+0.001 hop).

## 6. Possession rules implemented

We did not hand-code a possession definition; the model learns the one encoded by the provider
flags on IDSSE, which matches the challenge's description of its ground truth (dead where the
tracking marks the ball dead, otherwise the team the tracking names as owning it). In practice:

- **Loose balls and challenges:** possession stays with the team that last had it unless the
  evidence for the other team is sustained. The switch cost (1.0) means a brief touch, a deflection
  or a won aerial duel without follow-up control does not change possession.
- **Restarts and stoppages:** dead ball is its own state with a high entry and exit cost (7.0), so a
  stoppage needs consistent cues (players stopping, ball out or at a line, set-up shapes). After a
  restart, the team is decided by the evidence, not carried over from before the stoppage.
- **Dead-ball ownership:** never labelled; dead frames are `D`.
- **`X`:** not used; every frame gets `H`, `A` or `D`.

## 7. Validation and results

Leave-one-match-out on IDSSE: train on six matches, score the seventh, for all seven. The test copies
are degraded with the same simulator but different random draws, in two variants of player damage
(identity swaps, or small position hops), because which one our tracking resembles more cannot be
told from label-free statistics. Keep rules were fixed before each experiment: at least +0.01 mean
spellF1, better on at least 5 of 7 matches, bootstrap over matches at least 80% in favour, swap test
no worse than -0.005, and no material rise in false possession changes.

| Held-out spellF1 | rules (Viterbi) | learned_v1 | learned_v2 | learned_v3 | learned_v4 | **learned_v6** |
|---|---|---|---|---|---|---|
| IDSSE, degraded, hop damage | - | - | - | - | 0.626 | **0.646** |
| IDSSE, degraded, swap damage | - | - | - | - | 0.602 | **0.615** |
| IDSSE, clean players | - | - | 0.612 | 0.708 | 0.701 | 0.694 |
| Metrica sample games, degraded | 0.455 | 0.562 | 0.675* | 0.685 | 0.682 | **0.696** |
| Metrica sample games, clean | 0.544 | 0.612 | 0.713* | 0.740 | 0.736 | 0.730 |

\* held out within the Metrica sample games, on which learned_v2 was trained.

Learned_v6 beats learned_v4 on 13 of 14 held-out match tests and makes fewer false possession
changes (8.6 vs 10.2 and 10.9 vs 12.3 per 10 minutes). On the public five minutes, used only as a
check, the file is well formed and aligned; with 17 possessions that window is too small to rank
models (the difference between learned_v4 and learned_v6 there is two to three possessions).

**Tried and not kept** (each judged on held-out external data against the rules above): minimum
spell length in the decoder; rescuing short spells; self-training on the unlabelled test match
(+0.008 at best); a second-stage model on first-stage probabilities (+0.002 to +0.005); a temporal
convolution network (0.589 vs 0.622); a turnover detector driving a per-frame switch cost (+0.005);
ball-touch features (no gain once the ball is realistic); short-gap track repair; robust velocity
and track-age features; 256 units (no change); decoder settings (dead switch cost flat from 3.5 to
14; scale and switch cost within ±0.009 of the chosen values); SkillCorner data; Smart Tagging;
long-window team posture context at ±4 to 15 s (-0.007 hop, -0.004 swap); twice the training damage
(+0.020 swap but +0.001 hop), training on both damage levels, and blends of these with learned_v6
(+0.014 swap, +0.001 hop) -- none reached the bar on the hop test, whose position errors resemble
our tracking's most; re-linking broken player tracks across gaps of up to 15 s at sprint speed
(+0.002 hop, +0.003 swap); a duration-aware (semi-Markov) decoder using the training matches'
possession lengths (-0.039, it suppressed short possessions); extra weight on frames near possession
changes (no change); a second independent damage draw (+0.018 swap, -0.006 hop); twice as many
training frames (+0.003 hop).

## 8. Limitations

- **Short possessions.** Possessions under 2 s are recovered only about 22-25% of the time. Of the missed
  ones on held-out IDSSE, about a third are detected and then merged by the decoder; for about half,
  the tracking itself carries no usable evidence (ball missing, ball positioned about 2 m off, or an
  opponent nearer the ball than the player in control). Diagnostics: perfect possession boundaries
  would add +0.11 spellF1, correct teams +0.06, perfect stoppage timing +0.008.
- **Validation is somewhat optimistic about stoppages:** the IDSSE ball is frozen exactly on dead
  frames, so hiding it makes stoppages easier to time than in real tracking.
- **The degradation is a simulation.** It matches summary statistics, not the tracker's actual
  failure modes; which player-damage type (swaps or hops) our tracking resembles is unknown, so both
  are reported.
- **Convention check.** We did not verify on the provided match that the model's possession
  convention matches the ground truth's beyond the five-minute correctness check.
- **With more time:** ball detection from the tactical video with a detector trained elsewhere to
  fill missing ball positions (the main source of missing evidence), validated on other matches with
  video; and longer-window team posture features.

## 9. Reproduction

Requirements: Python 3.9, numpy 2.0.2 (`requirements.txt` at the repository root for evaluation;
pandas 2.3.3). The challenge data bundle in `data/` as distributed.

```bash
python -m possession.idsse --download --parse        # IDSSE open data, ~2.6 GB into data/external/idsse
python -m possession.train_learned --recipe v6 --output possession/models/learned_v6.json
python -m possession.run_learned --model possession/models/learned_v6.json \
    --output evaluation/results/learned_v6.csv
python evaluation/validate_submission.py evaluation/results/learned_v6.csv
```

The trained model is committed, so the third command alone regenerates the submission from the
provided data. Retraining reproduces the model file byte for byte, and `run_learned` reproduces the
CSV byte for byte, on the same machine and numpy build. All randomness has fixed seeds. Both commands
refuse to overwrite existing files. Fallback: replace `v6` / `learned_v6` with `v4` / `learned_v4`.
