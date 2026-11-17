# Possession from automatic tracking: technical document

**Team:** Tiki Taka Tech

**Submission:** `evaluation/results/blend_v8_e16.csv`, an equal blend of two models trained only on other
matches: `possession/models/learned_v8.json` (frame-level network ensemble) and
`possession/models/bigru_v8_e16.json` (bidirectional GRU), decoded with one Viterbi pass.
**Earlier models kept in the repository:** learned_v1-v4, v6 and v7 (learned_v6 was our previous
submission).

## 1. Summary

We label every frame `H`, `A` or `D` from the automatic tracking (ball and players) alone. Two
models turn the tracking into per-frame probabilities for the three states: an ensemble of small
networks that sees each frame with ±2 s of context, and a bidirectional GRU that reads each half in
both directions. Their log-probabilities are averaged and a Viterbi decoder picks the most consistent
sequence. Both are trained only on **other matches**: seven Bundesliga matches from the open IDSSE
dataset, labelled by the provider's per-frame ball-alive and possession flags, the same kind of
derivation the challenge's ground truth uses. Because our input is uncorrected automatic tracking,
the training copies are degraded to match label-free summary statistics of the provided tracking,
including a ball that stays visible for part of each stoppage, as it does in automatic tracking.
On held-out IDSSE matches degraded this way the blend scored **0.63-0.66 spellF1** (0.90-0.92
macroF1) with its first GRU, and training the GRU twice as long added a further +0.019 on held-out
external IDSSE matches (section 7); the published baseline on the challenge match scores 0.24.

## 2. Approach and why

**Why learned rather than hand-written rules.** Our first pipelines were hand-written rules (nearest
player, contact detection, stoppage and restart detectors, then a rule-based Viterbi). On external
matches with labels they reached 0.455 spellF1 (external Metrica sample games, degraded); learned evidence
lifted the same test to 0.56, and context with small networks to 0.68. Rules needed thresholds we
could not set without labelled data; models trained on labelled other matches set them from data.

**Frame features** (`possession/learned.py`): which team's players are nearest the ball and by how
much, ball motion relative to the nearest players, players of each team near the ball, team movement
and shape, the ball's position along each team's attack direction, and dead-ball cues (players
standing still, ball outside or leaving the pitch, players setting up at a line, ball speed, whether
the ball is tracked): 33 per frame.

**Model 1, frame ensemble (learned_v8).** Five one-hidden-layer networks (128 ReLU units) on the 33
features plus sixteen of them repeated at -2 s to +2 s (193 inputs). Frames inside possessions
shorter than 3 s count three times in training, because short possessions are the main source of
error.

**Model 2, bidirectional GRU (bigru_v8_e16, `possession/sequence.py`).** The same 33 features at 12.5
fps: Linear 33->64 + ReLU, a 2-layer bidirectional GRU (64 units per direction), Linear 128->3. It
learns its own context from the whole half instead of fixed ±2 s snapshots. Trained on the external
IDSSE matches for 16 epochs with PyTorch (`possession/train_sequence.py`); its weights are stored as JSON and inference is plain numpy.

Both models are trained on the external IDSSE matches only, with home and away swapped on every training frame and average their
prediction with the swapped one, so neither team is favoured.

**Decoding.** The two models' log-probabilities are averaged, then a three-state Viterbi over the
whole match (`possession/viterbi.py`): frame scores are 0.15 × (log-probability - log prior),
switching team costs 1.0 and entering or leaving dead ball costs 7.0.

## 3. Assets used and dropped

| Asset | Use |
|---|---|
| Automatic tracking (ball and players) | **the only input** to the submitted models |
| Smart Tagging | **dropped.** Phase windows of 15-35 s (longer than a typical possession), 24 transition tags in the match. Simulated on held-out IDSSE with the same counts and lengths, even *error-free* tags improved spellF1 by at most +0.002 |
| Tactical / broadcast video | not used |
| Homography | not used (tracking is already in pitch coordinates) |
| Public 5-minute ground truth | correctness check only: frame alignment, file format, label meaning |

## 4. Training data

No labels from the provided match are used anywhere, and no person labelled anything.

| Dataset | Source and licence | Labels: who and how | Amount | Role |
|---|---|---|---|---|
| **IDSSE** | Bassek et al. 2025, *Scientific Data*, doi 10.6084/m9.figshare.28196177, CC BY 4.0 | the data provider's per-frame `BallStatus` and `BallPossession` flags (`possession/idsse.py: labels`) | 7 Bundesliga matches | training and leave-one-match-out validation |
| **Metrica Sports sample games 1-3** | public GitHub repository metrica-sports/sample-data; three other full matches, **external sample data, not the challenge data** | derived from their open event data by a fixed rule (`possession/validate_metrica.py: event_labels`) | 3 matches | validation; training of the early models learned_v1/v2 only |
| SkillCorner open data | 20 A-League matches, MIT | the provider's possession field | tried | **dropped**: different convention; lowered held-out IDSSE from 0.705 to 0.601 |

## 5. Handling missing and noisy tracking

The test input is uncorrected automatic tracking; the ground truth comes from corrected tracking.
Training copies of IDSSE are therefore degraded to look like the provided tracking while their labels
stay as they are: clean answers, messy input, as in the real test. The degradation levels come from
**label-free, match-wide summary statistics** of the provided tracking, computed by script (rates,
percentages and percentiles over the whole match, nothing tied to individual frames):

| Statistic of the provided tracking | Value | How the IDSSE training copies reproduce it |
|---|---|---|
| Ball tracked | 59% of graded frames | live-play dropouts at the rate measured on the external Metrica sample games, plus the stoppage behaviour below |
| **Ball visible during stoppages** | visible more than 1.5 m outside the pitch on 1.4% of frames, visible and still for 2 s or more on 1.5% (hiding the ball on every dead frame gives 0.01% and 0%) | `stoppage_ball.observed_ball2`: in most stoppages the ball is collected by a player, carried, passed to the restart taker and placed; one stretch is hidden; a quarter of stoppages stay fully hidden. Gives 0.5% / 0% on IDSSE (a still ball is dropped after a second); a variant keeping still balls gives 1.1% / 3.3%; our match lies between, and both variants were tested (section 7) |
| Ball jitter, distance ball to nearest player | 0.23 m/s, 1.9 m | slow random drift of the ball, 1.2 m (`validate_metrica.smooth_drift`) |
| Sharpness of ball velocity changes (0.12 s) | p90 4.3, p99 8.9 m/s (IDSSE: 7.4, 17.9) | ball smoothed in time, 0.24 s Gaussian (`degrade.smooth_ball`) |
| Players tracked per frame | 20.5 of 22 | track breaks, mean gap 6.5 s (`degrade.damage_players`) |
| Track piece lengths | median 35 s, p10 2.6 s | piece lengths drawn from these quantiles |
| Implausible moves (> 12 m/s) | about 15 per minute | identity swaps with a nearby player, 9 per minute |

**Why the stoppage ball matters.** Our earlier models (up to learned_v7) were trained with the ball
hidden on every dead frame, so a visible ball always meant live play, and the models learned exactly
that shortcut. We confirmed it label-free on the provided match: with the ball tracked more than 3 m
outside the pitch (1,555 frames, necessarily a stoppage), learned_v6 and learned_v7 labelled **0%**
of those frames dead, and 0% of the ~10,000 frames with the ball visible while every player stood
nearly still. The stoppage-ball simulation (`possession/stoppage_ball.py`; its timings and speeds are
typical values we set by hand from general football knowledge, not measured from any data, and fixed
before any model was scored) removes
the shortcut: the submitted blend labels 85% of the outside-the-pitch frames dead, while in clearly
live play (players moving) it still names the right team for 97% / 87% of frames where one player
clearly controls the ball, as before.

Each IDSSE match enters training four times: as observed, with ball drift, with damaged players, and
with both, all with the smoothed ball and the stoppage-ball model. The provided match is never
altered. Missing values are handled explicitly: features carry a ball-visible flag, player features
use the players that are tracked, and the decoder carries the last state through gaps.

The results do not hinge on the exact damage levels or hand-set feature constants (a 1 m margin and
3 m reach around the ball, a 1 s window, and 1.3 m/s, a typical walking speed, used as the neutral
value when no player speeds are available): halving or doubling them moves held-out external (IDSSE) spellF1 by at
most ±0.005, except doubling the damage (+0.020 on one damage type, +0.001 on the other).

## 6. Possession rules implemented

We did not hand-code a possession definition; the models learn the one encoded by the provider flags
on IDSSE, which matches the challenge's description of its ground truth (dead where the tracking marks
the ball dead, otherwise the team the tracking names as owning it). In practice:

- **Loose balls and challenges:** possession stays with the team that last had it unless the evidence
  for the other team is sustained (switch cost 1.0), so a brief touch, a deflection or a won aerial
  duel without follow-up control does not change possession.
- **Restarts and stoppages:** dead ball is its own state with a high entry and exit cost (7.0). A
  stoppage is recognised from consistent cues (players slowing, the ball out, carried or set at a
  line), not from whether the ball is visible. After a restart the team is decided by the evidence.
- **Dead-ball ownership:** never labelled; dead frames are `D`.
- **`X`:** not used; every frame gets `H`, `A` or `D`.

## 7. Validation and results

Leave-one-match-out on IDSSE: train on six matches, score the seventh, for all seven. Test copies use
random draws never used in training, two kinds of player damage (identity swaps, small position hops)
and three ball simulations: **old** (ball hidden on every dead frame, the simulation used up to
learned_v7), **stoppage** (`stoppage_ball` as above) and **stoppage, still ball kept**. The two
stoppage simulations resemble the provided tracking (section 5); the old one is optimistic, because it
hides the ball exactly when play stops. Keep rules were fixed before each experiment (at least +0.01
mean spellF1, better on at least 5 of 7 matches, bootstrap over matches at least 80%, no material loss
elsewhere); the changes adopted in the final round (short-possession weighting, stoppage ball,
GRU) were re-checked on fresh random draws.

| Held-out spellF1 (7 IDSSE matches, fresh draws) | learned_v6 recipe | learned_v7 recipe | learned_v8 recipe | **Submitted blend** |
|---|---|---|---|---|
| Stoppage simulation, hop damage | 0.524 | 0.529 | 0.630 | **0.662** |
| Stoppage simulation, swap damage | 0.498 | 0.506 | 0.593 | **0.636** |
| Stoppage, still ball kept, hop | 0.513 | 0.523 | 0.622 | **0.654** |
| Stoppage, still ball kept, swap | 0.490 | 0.504 | 0.588 | **0.628** |
| Old simulation, hop | 0.645 | 0.655 | 0.642 | **0.670** |
| Old simulation, swap | 0.615 | 0.627 | 0.607 | **0.648** |
| macroF1, stoppage simulations | - | 0.76-0.81 | 0.88-0.90 | **0.90-0.92** |
| False possession changes per 10 min, stoppage simulations | 13-16 | 18-20 | 13-14 | **8-9** |

Experiment ensembles of two networks; learned_v6 column from a separate fresh-draw run. The blend
beats learned_v8 alone by +0.031 (7 of 7 matches) and learned_v7 by +0.13 on the stoppage
simulations, and is also best on the old simulation. Per match it ranges from 0.56 to 0.71. It finds
fewer short possessions than learned_v8 (26% vs 31% of those under 2 s) but places boundaries more
precisely (F1 at overlap 0.7: 0.53 vs 0.48) and makes about 40% fewer false changes. These figures
use the first GRU (8 epochs). Training it for 16 epochs instead, tested on three held-out external IDSSE matches
with fresh random draws, added +0.019 spellF1 to the blend (better on 3 of 3 matches and on all four
stoppage-simulation tests: 0.650 -> 0.665, 0.621 -> 0.641, 0.642 -> 0.663, 0.618 -> 0.638), so the
submitted GRU is trained for 16 epochs.

Earlier history (old simulation, which flatters every model on stoppages): rules 0.455 and learned_v1
0.562 on the external Metrica sample games (degraded); learned_v2 0.675; learned_v3 0.708 (clean IDSSE);
learned_v4 0.626 / 0.602 and learned_v6 0.646 / 0.615 on damaged IDSSE (hop / swap).

On the public five minutes, used only as a check, every file is well formed and aligned; with 17
possessions the window is too small to rank models.

**Kept along the way:** learned evidence instead of rules; context networks; IDSSE training data;
ball drift; simulated player damage; smoothed ball; short possessions weighted 3× (+0.009, confirmed on
fresh draws); the stoppage-ball simulation (+0.10 on the stoppage simulations); the bidirectional GRU
(+0.024 / +0.030 on the old simulation, +0.017 to +0.021 on the stoppage simulations, fresh draws);
training the GRU for 16 epochs instead of 8 (+0.019).

**Tried and not kept** (each judged on held-out external data): minimum spell length and rescuing
short spells in the decoder; self-training on the unlabelled test match (+0.008 at best); a
second-stage model (+0.002 to +0.005); a temporal convolution network (0.589 vs 0.622); a turnover
detector driving a per-frame switch cost (+0.005); ball-touch features; robust velocity and track-age
features; 256 units; decoder settings (dead switch cost flat from 3.5 to 14); SkillCorner data; Smart
Tagging; long-window team posture context at ±4 to 15 s (-0.007); twice the training damage (+0.020 on
swaps only); re-linking broken player tracks (+0.002); a duration-aware (semi-Markov) decoder (-0.039);
extra weight on frames near possession changes (no change); a second damage draw; twice as many
training frames (+0.003); a hand-written decoder rule forbidding stoppages during passing sequences
(-0.006 to -0.009, and it split stoppages in two); an average of three GRUs with different seeds (+0.002).

**Where every setting comes from.**

| Setting | Value | Source |
|---|---|---|
| Dead-ball switch cost | 7.0 | introduced with learned_v1 on the external Metrica sample games and kept since; on held-out IDSSE spellF1 is flat from 3.5 to 14 |
| Team switch cost, score scale | 1.0, 0.15 | chosen on held-out IDSSE when learned_v3 was built (earlier models used 2.0 / 0.2 on the external Metrica sample games); ±50% changes move spellF1 by at most ±0.009 |
| Frame ensemble size, context | 5 networks of 128 units, ±2 s | chosen on held-out external data during learned_v2/v3 (+0.013 for the finer context and 128 units) |
| Short-possession weighting | under 3 s, 3× | picked as a reasonable starting value before testing; not adjusted afterwards |
| GRU design | 64 units, 2 layers, 12.5 fps, 16 epochs, 48 s crops, Adam 2e-3 | picked as a reasonable starting design before testing with 8 epochs; the training length was then doubled to 16 after a held-out test on external matches (+0.019, 3 of 3 matches); nothing else adjusted |
| Blend | equal weights | picked as a reasonable default before testing; not adjusted afterwards |
| Stoppage-ball simulation | collect 1-4 s, pass 6-14 m/s, 25% of stoppages hidden, still ball hidden after 1 s | typical values we set by hand from general football knowledge (not measured from any data), fixed before any model was scored |
| Degradation levels | section 5 table | label-free match-wide statistics of the provided tracking; live dropouts from the external Metrica sample games |
| Feature constants | 1 m, 3 m, 1 s, 1.3 m/s, 12 m/s | physical choices; halving or doubling changes held-out external (IDSSE) spellF1 by at most ±0.005 |
| Random seeds | fixed per copy (100+, 300+, 400+, 700+ match index; 0 for networks) | arbitrary, recorded in the code |

## 8. Limitations

- **Short possessions.** About a quarter of possessions under 2 s are recovered. On held-out IDSSE,
  for about half of the missed ones the tracking carries no usable evidence (ball missing, ball about
  2 m off, or an opponent nearer the ball than the player in control).
- **The degradation is a simulation.** It matches summary statistics, not the tracker's actual failure
  modes. How often the provided tracker shows the ball during stoppages is bounded but not known
  exactly, which is why two stoppage variants are reported.
- **Possible false stoppages.** On the provided match, about 10 of the blend's 120 stoppages contain
  two or more passes inside the pitch, which suggests slow build-up play occasionally called dead, and
  a goalkeeper holding the ball while players are slow is labelled dead about 60% of the time (some of
  these are goal-kick set-ups, which are dead).
- **Convention check.** We did not verify on the provided match that the possession convention
  matches the ground truth's beyond the five-minute correctness check.
- **With more time:** ball detection from the tactical video with a detector trained elsewhere,
  validated on other matches with video; a learned refinement of possession boundaries.

## 9. Reproduction

Requirements: Python 3.9, `requirements.txt` (numpy 2.0.2, pandas 2.3.3). Retraining the GRU also
needs `requirements-training.txt` (PyTorch 2.5.1, CPU). The challenge data bundle in `data/` as
distributed.

```bash
python -m possession.idsse --download --parse          # IDSSE open data, ~2.6 GB into data/external/idsse
python -m possession.train_learned --recipe v8 --output possession/models/learned_v8.json
python -m possession.train_sequence --recipe v8 --epochs 16 --output possession/models/bigru_v8_e16.json   # PyTorch
python -m possession.run_learned --model possession/models/learned_v8.json \
    --sequence-model possession/models/bigru_v8_e16.json --output evaluation/results/blend_v8_e16.csv
python evaluation/validate_submission.py evaluation/results/blend_v8_e16.csv
```

Both trained models are committed, so the fourth command alone regenerates the submission from the
provided data with numpy only, byte for byte (verified twice, and from a fresh clone of the
repository). The frame ensemble is plain numpy with fixed seeds and retrains to the same file;
the GRU's PyTorch training is seeded but multi-threaded, so a retrained GRU is close to, not
bit-identical with, the committed one. All commands refuse to overwrite existing files.
