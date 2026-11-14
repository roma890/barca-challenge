# Possession inference experiments

Hypothesis: treating missing evidence as uncertainty rather than automatically
marking it dead can reduce false possession splits. Match-level improvement has
not been established.

`state.decode_state` takes separate arrays for established team control (`H/A/X`)
and dead-ball evidence (boolean). The prototype now uses `max_carry_frames: null`:
retain the established team until new control, a stoppage or a reset, instead of
expiring possession solely because observations became uncertain. Dead-ball
evidence takes priority and clears team memory. Optional reset flags prevent
carry across recording breaks. The returned `carried` mask distinguishes
continuity from direct evidence; a retained label is not a fresh observation.

`evidence_age_frames` tracks how old the latest team-control evidence is. A later
opposing control endpoint flags the intervening uncertain frames in `gap_conflict`
without inventing a transition time or interpolating ball positions. Matching
endpoints support continuity but do not rule out hidden turnovers/stoppages.
Evidence age/conflict are diagnostics, not calibrated probabilities. Long gaps
still need contextual reasoning; the continuity rule can retain a stale label
through an unseen turnover or stoppage. X remains before initial control and
after a reset/stoppage until a new team is established.

Passing nearest-player labels as established control would retain the baseline's
false-turnover problem.
Integer carry limits retain the previous timeout behavior for reproducibility.
The move to continuity follows the challenge's loose/contested-ball definition,
not threshold selection against public labels. No minimum possession duration
is imposed.

`control.estimate_control` supplies conservative H/A/X evidence using proximity
and relative velocity. Callers provide aligned ball positions `(frames, 2)` and
player positions `(frames, slots, 2)` **in metres**, an H/A team per slot, frame
rate, and an explicit `ControlConfig`. For normalised tracking, multiply each
position by `(pitch_length, pitch_width)` first. Slots are tracklets, not player
identities.

A nearby player with compatible motion is a candidate. A similarly close
opponent causes abstention. Velocities use adjacent observations only; missing
coordinates, newly visible slots and resets prevent velocity estimation across
the affected boundary. No interpolation or minimum possession duration is used.
The result includes the selected slot, distance and relative speed for accepted
frames. These are diagnostic measurements, not calibrated confidence scores.

Limitations: this cannot establish intent, recognise one-touch kicks reliably,
or handle tracking jitter well without further work. The first observed frame
cannot supply velocity. An interception may be recognised after its actual onset;
this component alone does not refine boundaries. Stationary balls beside players can
produce control evidence during stoppages, so separate dead-ball evidence must
override it. Matching velocity during a contested run can also be misleading.
Thresholds in synthetic tests are toy settings, not validated match parameters.

`contacts.detect_contacts` adds possible-contact detection. It fits incoming and
outgoing ball trajectories over configurable windows, measuring velocity change,
turn angle, speed change and fit residual. Missing coordinates, resets, dead frames,
excessive speed and poor fits reject a window. A contact requires a nearby player;
ambiguous attribution is retained as X instead of forcing a player assignment.
Local fits need future frames and can miss closely spaced or straight-through
touches. Speed guards cannot catch all identity switches or plausible noise.

`interactions.infer_interactions` connects the contact detector with motion-based
control evidence. It retains an established team's compatible motion during a
close challenge. It can use later control by the touching player to backdate a
reception, but only while the continuously observed ball stayed within the control
radius; a loose rebound followed by a chase is not automatically backdated.

A trajectory toward a teammate followed by their control becomes a
`pass_hypothesis`, with separate contact and confirmation frames. The receiver
must have a reliable track, the ball flight must be continuous and consistent
with the outgoing trajectory, and intervening unrelated touches/control block
confirmation. Contact time, receiver slot and reasoning are exposed as diagnostic
annotations. There is no claim that a successful reception proves intent.

`InteractionConfig.accept_pass_hypotheses` MUST be explicitly True or False.
With False, a fortunate deflection cannot establish possession at the earlier
touch solely because it reaches a teammate. With True, supported pass hypotheses
can start possession at contact, enabling an experiment on one-touch inference
but risking exactly that false turnover. Identical observed trajectories produce
identical hypotheses regardless of the action's hidden intent. Failed one-touch
passes without independent control evidence can still be missed. Clearances are
not automatically possessions; previously established control is preserved by
the state decoder, subject to its carry policy.

Use `infer_interactions`'s `labels` as `decode_state`'s control input, with the
same dead/reset masks and an explicit carry policy. `source` distinguishes direct
motion, incumbent continuity, backdated receptions and promoted pass hypotheses.
This is offline inference; contact windows and reception searches use future
frames. No minimum possession duration is applied. None of these settings have
been selected using the public ground truth or restricted match data.

The official spell metric forward-fills abstentions within graded runs, so `X`
does not stop a spell or resolve a long gap for that metric. Independent stoppage
evidence is still needed. The public five minutes must not be used for tuning.

`dead_episodes.infer_dead_episodes` adds bounded retrospective onset evidence to
supported restart preparation. Candidate onsets are observed inside-to-outside
exits consistent with the restart type/side, or weaker group-deceleration plus
quiet/hidden-ball pauses. It preserves earliest/latest onset bounds rather than
placing a boundary at a tracking-gap midpoint. Boundary proximity or missingness
alone creates no episode.

Searches are bounded by configured history, sequence resets and previous supported
releases. Coherent transfers across three distinct track slots provide a live-
sequence contradiction guard; generic movement or a single contact is insufficient
because retrieval can contain both. This is heuristic evidence, not referee or
persistent-player identity information.

GOALS tags are candidate context, not guaranteed goals. A goal episode additionally
needs a compatible observed central goal-line exit and a later supported kickoff-
after-goal. The tag start is not copied as the onset. The central corridor is a
tracking tolerance, not an exact legal goal test. Missing corroboration leaves the
pair unresolved.

Episode outputs contribute relative support to the existing temporal decoder using
its unchanged restart weight. Generic motion and broad live-phase cues are
discounted inside supported episodes because retrieval can move objects during
dead play. Confirmed release anchors still support the subsequent live state.
False exits or false restart hypotheses can backdate incorrectly; there is no
two-second-occlusion-to-D fallback.

The optional `episodes` config requires both `stoppages` and `play_state`. This
version starts only from strict restart anchors with the prototype configuration.
New search limits are documented, unvalidated priors; no public labels or score
sweep were used to select them.

`configs/wide_candidates.json` is a separate opt-in experiment. It preserves every
existing control, contact, restart and play-state setting and enables weaker
restart candidates. Each needs SET PIECES context, a tentative compatible exit or
relative group slowdown, and a fitted separating restart-like action near a player.
Partial observations are allowed; no tag-only or occlusion-only candidate is made.
One earliest qualifying action per tag interval is retained, subject to sequence
resets and contradictory live transfers.

Weak support is smaller than confirmed restart support, enters the temporal score,
never enters the hard restart mask and does not seed a restart team. Unlike stronger
backdated episodes it never discounts competing motion/phase live cues. Separate
diagnostics identify weak candidates/support. This broader path is disabled in
prototype.json, and its CSV should remain outside evaluation/results/ until an
explicit permitted scoring check is requested, to avoid incidental public A/B use.

The play-state live_conflict_proxy combines strong ball movement, majority player
activity and a live-phase tag before inference. It is an audit proxy, not a known
live-play label. Audit added D, conflicts with this proxy/live-transfer guards and
potential new splits relative to a frozen prediction. No proxy overlap does not
prove absence of false alarms, especially if no reliable sequences were detected.

`configs/supported_exits.json` separately enables `resumption.refine_dead_exits`.
All previous settings are unchanged. After the temporal model proposes leaving D,
the check looks for a ball departing from a nearby observed player, a compatible
dribble with group activity, or coherent fast ball travel with group activity.
Strict restart live anchors bypass it. Tags, walking, ball retrieval with passive
teammates, and an unaccompanied moving ball do not establish resumption alone.
The grouping/trajectory tests are fallible physical hypotheses, not referee or
intent detection. This component never supplies an ownership seed.

Observed trajectories may contain bounded gaps and require coherent fits, plausible
speeds and in-pitch ball positions. Confirmation can use future samples, but a
release is never placed before the first observed movement. Searches cannot cross
another proposed stoppage or reset. The existing control thresholds are reused
for dribble compatibility, without relaxing proximity after a stoppage.

An unsupported exit retains explicit D for at most the configured hold duration
(125 frames / five seconds in the candidate). Exhaustion means unresolved play
state, not evidence of live play: the pipeline emits X and prevents unconfirmed
control from leaking into its incumbent team. A later supported action, a new D
or a reset resolves/replaces that pending hypothesis. The official temporal scorer
can still fill X with D, so the explicit hold bound does not bound the scored D
spell. False initial D predictions can still be extended; audit added D inside
previous team runs and conflicts with physical live evidence before adoption.

This check validates proposed exits only. It does not fix initial tag vetoes,
discover all untagged stoppages, or recover a team during a genuine unseen restart.
Retrieval and an isolated live dribble can be indistinguishable in these positions;
requiring group support can delay a genuine quiet restart. This is a conservative
hypothesis policy, not proof that unsupported movement is dead play.
New scales are unvalidated modeling priors fixed before local execution. Keep its
CSV under evaluation/reports/ and use input-only audits, not public-score selection.
The prototype config and existing submission are unchanged.

`configs/supported_exits_gaps.json` adds two explicit policies with all previous
weights, thresholds and bounds preserved. `bridge_observation_gaps` allows the
resumption check to use a qualifying physical action observed on both sides of
a short current-frame occlusion. It reuses the original gap/fraction/fit/speed,
pitch and player-support requirements; it neither fills coordinates nor infers
an owner. A hidden kick with no observed earlier movement remains unresolved.
The bounded search can inspect the immediately preceding proposed stoppage but
cannot cross another stoppage, reset or strict dead anchor.

`use_live_sequence_guard` passes the episode detector's three-distinct-track
transfer mask into resumption, overriding soft D/holds where that evidence applies.
Explicit caller dead masks and strict dead anchors retain priority; strict-anchor
conflicts are counted separately. This guard requires episode settings. It is
distinct from the centered ball-motion/player-activity/tag proxy used for audits.
The proxy is not promoted to a live-play override. An inactive transfer detector
cannot validate real-match safety; integration is checked on invented sequences.
Both new policies default to false so the earlier configurations remain reproducible.

Generate a separate local candidate (paths must not already exist):

```sh
python -m possession.run --config possession/configs/supported_exits.json \
  --smart-tagging --output evaluation/reports/supported_exits_local.csv \
  --report evaluation/reports/supported_exits_local.json
```

Run the synthetic checks from the repository root:

```sh
python -m unittest discover -s tests -v
```

The local adapter and runner connect these components to the official CSV writer
and validator. Automatic dead-ball evidence now combines a **soft temporal
ball-in-play model** and a separate restart-anchor detector. No match-level
accuracy improvement is claimed. The official baseline is unchanged.

`play_state.infer_play_state` uses automatic set-piece context as a prior, including
a weak decaying lead-in. BUILD UP, PROGRESSION, ATTACKING/DEFENSIVE TRANSITION,
LONG BALL and CREATING CHANCES provide soft live-play context. Duplicate tags do
not multiply evidence. Team strings, including N/A, are not treated as possession
labels. Broad intervals are not copied to D or H/A.

Quiet ball and player setup are alternative preparation cues in one shared channel;
they are not double-counted. Missing ball observations can reinforce a supported
setup, but cannot create D alone. Sustained movement supports live play. Repeated
out-of-pitch ball positions supply independent physical evidence, subject to margin
and observation checks. Vector velocities use fits over observed timestamps rather
than single-frame differences. Missing positions are never interpolated.

A two-state Viterbi decoder balances these imperfect cues with transition costs.
Neutral gaps preserve state, resets split sequences, and confirmed releases provide
live anchors. No complete clean restart chain is required to infer D. The result
includes evidence scores and uncertainty flags, not calibrated probabilities.
Incorrect/overlapping tags, slow live play, aerial projection and tracking errors
can still produce mistakes. See EXPERIMENTS.md for the no-public-tuning boundary.

`stoppages.detect_stoppages` combines an automatic set-piece interval, compatible
restart location, a supported preparation period with a nearby tracked player,
and a sustained release away from that player. Preparation uses a robust position
cluster and a fitted motion trend over time, rather than requiring every adjacent
frame's velocity to be near zero. Short gaps are allowed only with sufficient
observed coverage and a maximum gap duration; resets and implausible motion still
block inference. The outgoing trajectory is fitted to observed timestamps, with
residual and separation checks. No missing positions are interpolated.
Stationarity, missing tracking and tagging alone never produce D. A moving dribble
that stays with the nearby player also does not meet the release rule.

Internal evidence is U (unknown), D (supported dead hypothesis), or L (supported
release). These are not new submission labels: control inference still determines
H/A/X outside supported D frames. Only the supported preparation period is marked
dead, bounded by the tag and a maximum history length. Release and later motion
confirmation have separate frames. Detection can use a confirmation beyond the
tag's end, but the release itself must be inside its half-open interval.

At tagged landmark restarts (kickoff, corner, goal kick or throw-in), an optional
hidden-ball route uses a settled nearby player and the ball's first observed
outgoing trajectory after occlusion. The trajectory must start near that setup
and separate from the player. Available preparation ball observations can veto
the hypothesis. Preparation can extend backward within the configured history
while the player setup supports it; an observed ball elsewhere stops extension.
A currently moving/absent player, wrong landmark, missing outgoing
ball, or ordinary moving dribble is insufficient. Free kicks cannot use this route
because they have no distinct landmark. This does not make missingness evidence
of dead play by itself, and does not solve long unobserved release timing.

On a supported release the pipeline resets temporal memory and, when geometry
identifies an unambiguous preparer team, seeds that team at the supported release
frame. A short fitted-trajectory backtrack (at most confirmation_frames - 1) can
place release before the first clearly separated observation when the observed
ball anchor supports that crossing. It never derives ownership from the tag's
Team string, which may be N/A.
Ambiguous ownership stays X. Caller-supplied automated dead evidence is combined
by OR and takes precedence over restart team injection.

Limitations: a false set-piece tag on an otherwise identical live hold-and-pass
can still create a false stoppage. Unknown fouls, unconfirmed restarts, moving-ball
preparation that lacks a supported setup, goals, halftime and earlier whistle/dead-
ball onset remain unresolved. Hidden-ball setup is a weaker contextual hypothesis,
not a direct observation of the ball or referee.
The location regions and motion thresholds are experimental priors, not exact
legal tests or values tuned on the provided game. U does not assert live play;
L is supporting restart evidence, not proof of referee permission.

`StoppageResult.diagnostics` and the pipeline's `stoppage_gate_diagnostics`
expose aggregate first-failing condition counts and the longest supported
preparation found. Counts describe attempts within tagging windows (overlapping
windows may repeat frames), not numbers of real stoppages. They use no ground
truth and do not export frame IDs or tracking records. Instrumentation does not
change detection thresholds or decisions.

`tracking.load_tracking` streams the raw text, verifies exactly 1..n frame IDs
and the metadata slot count, and converts normalised coordinates to metres.
It preserves NaNs, rejects malformed or truncated exports, and uses the official
team-ordering helper. It never opens video, broadcast, tagging, homography, ground
truth, or evaluation reports. Inputs and derived arrays stay on the local machine;
the adapter has no network calls or feature-cache exports. Although input text is
streamed, the full position arrays and inference intermediates occupy memory;
full-match memory/runtime have not been benchmarked.

To inspect the CLI without opening any match files:

```sh
python -m possession.run --help
```

For an explicitly chosen local tracking run, the interface is:

```sh
python -m possession.run \
  --config possession/configs/prototype.json \
  --output evaluation/results/prototype.csv \
  --report evaluation/reports/prototype_pipeline.json
```

This command **does open local tactical ATD files when executed**; it is not part
of the synthetic test suite. Use `--data-dir /path/to/bundle` to override the
normal `games.py` location (`MTAH_DATA_DIR` or `data/`). Broadcast input is not
offered: submissions use tactical-video frames only. The default row count is
the official 199,319. `--frames` overrides this for synthetic fixture runs only;
a small fixture CSV is not a valid full challenge submission.

Add `--smart-tagging` to read the local automatic Smart Tagging XML through the
official loader and supply live/dead context. Without that flag the runner never
opens tagging; tracking-only evidence can still support out-of-play hypotheses.
The `play_state` and `stoppages` sections can independently be null/omitted to
disable their components; the flag requires at least one configured component.
No videos, homographies, ground truth or evaluation
reports are read by either CLI mode. The detector consumes the official loader's
video-frame intervals without extra offsets. Automated tests use synthetic tagging
XML and tracking; actual local runs report only aggregate diagnostics to the assistant.

The committed JSON provides explicit, **unvalidated prototype settings**, not
recommended match thresholds. The earlier two-frame possession expiry has been
replaced by continuity (`max_carry_frames: null`); pass-hypothesis promotion
remains disabled. Existing predictions are not automatically regenerated when
settings change; the saved pipeline report records the configuration used for
each generated CSV. Previously saved scores become stale if a CSV is replaced. No public
ground truth was used to choose the settings. The `stoppages` section provides
explicit preparation/release limits; its one-second minimum preparation can miss
quick restarts, and its maximum preparation does not recover earlier unseen dead
play. Library callers can supply automated dead/reset evidence or synthetic masks.
Do not create hand-label masks for the competition match.

The window-based preparation settings require at least half of ball positions
observed in the ordinary preparation route, with gaps of at most 12 frames, and
at least 60% observed release positions. The hidden-ball route instead requires
reliable settled-player evidence at a tagged landmark and a first observed outgoing
trajectory. Release coherence is based on fitted residual relative to displacement,
so jitter does not inflate an adjacent-frame path length. These are explicit
experimental assumptions, not values selected against public ground truth.

The `play_state` settings separately specify physical scales, observation windows,
context lookback, evidence weights and temporal transition cost. These are explicit,
unvalidated hand-set priors for one candidate, not trained values. No score sweep,
public-label fitting or public-score model selection is used. Aggregate D coverage
is not an accuracy metric or a target to maximize. The play_state_diagnostics field
describes this broader model; stoppage_status_counts describes only restart anchors.

The runner refuses existing output/report paths. It writes a temporary CSV with
the official helper, validates it with `possession_metrics.validate_submission`,
and publishes it locally only if it passes. This proves shape and label validity,
not accuracy. The optional JSON and stdout contain aggregate diagnostics and the
effective configuration, without frame-level records. Possession counts follow
the official validator convention: X splits runs there, while the spell scorer
fills X inside graded runs. Keep generated CSVs and reports in the existing
gitignored directories shown above; never commit full-match predictions.

Aggregate diagnostics now include the carried fraction, how many retained labels
coincide with missing ball positions, the maximum evidence age, and the number of
gap frames with opposing observed control endpoints. Lower X coverage after
changing the continuity policy does not itself establish higher accuracy.

Integration tests generate their own small ATD-shaped files in temporary folders,
run the CLI, and check the output with the official validator. They also check
deterministic output, frame/slot errors, units, missing positions, dead/reset
propagation and failed-validation cleanup. Restart tests distinguish preparation,
release and confirmation; cover supported locations and counterexamples; and
verify that D splits two same-team possessions. Known ambiguous/failure cases are
documented as such, not counted as solved football classifications. Tests never
open the match bundle or any ground truth. Real-match coverage, accuracy, resource
use and runtime still require permitted local verification.
