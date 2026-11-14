# Possession model failure map

Analysis checkpoint: 2026-11-13. This document does not change the model, settings,
tests, predictions or scoring. It contains source-based findings and invented
examples only, with no match-specific labels, event times or evaluation results.

**The task:** infer established team possession and whether the ball is in play.
Contact, proximity, coordinated movement, tactical context and tracking visibility
are evidence about that state; none is its definition. Loose/contested live play
retains the last established team. A real short possession must not be removed
merely for being short. See [CHALLENGE.md](../CHALLENGE.md).

**Evidence levels used below**

- **Demonstrated:** a synthetic counterexample reproduces the limitation.
- **Confirmed policy:** the implementation behaves this way; how often it is wrong
  on real inputs has not been established.
- **Hypothesis:** a plausible failure mechanism that needs a discriminating test.
- **Unresolved ambiguity:** distinct football actions can have identical available
  observations. No deterministic position-only rule can always distinguish them.

Priority means investigation order based on dependencies and consequences, not
estimated score gain. No ranking here was selected against the public window.

| ID | Failure mechanism | Evidence | Investigation priority |
|---|---|---|---|
| F01 | A fitted average velocity can hide substantial ball movement | Demonstrated feature limitation | First |
| F02 | Direct control evidence favors dribbling over kicks and receptions | Confirmed gate; synthetic missed-event examples | First |
| F03 | A single accepted opponent candidate becomes established possession | Confirmed policy; false-turnover frequency unknown | First |
| F04 | Long missing intervals retain a team without resolving hidden changes | Confirmed policy; boundary ambiguity | First |
| F05 | A false dead-ball decision erases ownership and creates later abstentions | Demonstrated integration consequence | First |
| F06 | Uncertainty diagnostics do not govern state transitions or evidence precedence | Confirmed policy | First |
| F07 | Successful one-touch passes and fortunate deflections can look identical | Demonstrated limit; unresolved ambiguity | Next |
| F08 | Smart Tagging team roles are not used for ownership and are not yet specified | Confirmed unused input; semantic question | Next |
| F09 | Windows, transition penalties and strict contact fits can distort timing | Confirmed mechanisms; extent unmeasured | Next |
| F10 | Slot continuity is not player identity or guaranteed physical continuity | Documented data limitation; partial guards | Next |
| F11 | Outside-pitch coordinates can be mistaken for actual out-of-play evidence | Hypothesis about imperfect 2D observations | Next |
| F12 | Aerial duels, goalkeeper actions and clearances lack dedicated interpretation | Confirmed missing capabilities | Later, after shared foundations |

**F01 — Low average velocity is not necessarily a settled ball**

Source: [play_state.py](play_state.py), `_motion_window` and the quiet-ball
preparation channel. It fits a line and checks observation coverage and fitted
speed. It does not check trajectory residual, spatial excursion or reversal before
calling a low slope quiet.

An analysis-only synthetic probe used x positions `30, 31, 32, 31, 30` metres at
25 fps, with y fixed. The central five-sample fit reports **0 m/s** and is accepted
as known, while the observed path covers four metres in 0.16 s: **25 m/s** average
path speed. This proves a feature ambiguity, not that the temporal decoder always
labels this short example D.

Consequence: out-and-back motion, a rebound, or a change of direction can contribute
quiet-ball evidence. Centered windows can also mix pre- and post-contact motion.

Existing checks cover a stationary ball, jitter, ordinary moving play and isolated
out-of-pitch errors. They do not establish that this feature distinguishes quiet
preparation from nonlinear movement in general.

Next analysis: compare stationary jitter, a cushioning touch, fast reversals and
true preparation at several sampling phases. Examine fitted residual, spatial
spread and directional consistency as possible reliability evidence. Any proposed
feature must retain jitter tolerance; raw path length also grows with noise.

**F02 — Our strongest control signal mainly recognizes carrying the ball**

Source: [control.py](control.py), `estimate_control`; [interactions.py](interactions.py),
the direct-motion path. Team evidence requires proximity, compatible velocities,
and a distance advantage over the opposing team. The velocities in this path are
adjacent-frame differences, unlike the window fits used for ball-in-play evidence.

Counterexamples: deliberate first-time passes, receiving a fast pass, a controlled
interception at contact, and a kick that immediately accelerates away. A valid
interaction need not have similar ball and player velocities. Noise can also break
the gate even during real control.

Consequence: control may only be recognized later, while a subsequent dribble is
visible. Possession memory then carries an old team until that late observation.

Existing coverage: fast passes near defenders, dribbling and an interception after
the ball slows. The one-touch test explicitly documents a missed capability;
passing that test is not evidence of good one-touch recognition.

Next analysis: write separate evidence definitions for a touch, controlled
reception, dribble and deliberate release. Trace those definitions through the
same synthetic pass/interception rather than loosening one velocity threshold.

**F03 — Candidate evidence is promoted directly to a possession change**

Source: [interactions.py](interactions.py) and [state.py](state.py). A non-X H/A
output is treated as established control. One accepted opposite-team frame can
switch possession; the next uncertain frames can retain that new team indefinitely.

Proximity plus compatible movement is not proof of control. A defender running
beside a pass, a brief matching motion, or a noisy observation can satisfy a proxy.
How often that happens is unknown; an isolated candidate is not automatically false.

Consequence: an input error can become an entire false spell. Requiring a fixed
minimum possession duration would create the opposite error by deleting real
first-time interceptions and rapid turnovers.

Existing coverage preserves genuine one-frame control inputs and rejects several
ambiguous geometries. It does not test the full range of false-positive candidate
evidence before state promotion.

Next analysis: distinguish candidate evidence from an established state transition.
Specify which independent interaction/sequence evidence can justify a rapid change
without using duration as a substitute for control.

**F04 — Continuity is an assumption, not reconstruction of a long gap**

Source: [state.py](state.py), `decode_state`. With the current continuity policy:

```text
Control evidence: H X X X X A
Output:           H H H H H A
Gap conflict:     - Y Y Y Y -
```

The conflict flag uses the later endpoint but does not alter the labels or locate
the turnover. Matching endpoints also do not rule out hidden H→A→H changes or a
stoppage. Evidence age is reported but does not resolve these alternatives.

Consequence: a long carried interval can contain the wrong team or undetected dead
play. An arbitrary midpoint turnover or unlimited ball interpolation would invent
information rather than solve that ambiguity.

Existing tests verify carry, resets and disagreement flags. They do not recover
unobserved changes. Confidence flags are not calibrated probabilities.

Next analysis: build a gap evidence inventory: endpoint control, player movement,
automatic phase context, restart evidence and explicit contradictions. Separate
what supports continuity from what could bound a change. Where none can locate a
boundary, retain that limitation explicitly.

**F05 — A false D can cause errors after the false stoppage has ended**

Source: [pipeline.py](pipeline.py), dead-mask combination and live-transition resets;
[state.py](state.py), clearing team memory on D.

An analysis-only toy dribble with a missing interval produced:

```text
Without an injected error:     X H H H H H H H H H
With one false D at index 3:   X H H D X X X X X H
```

The one-frame false stoppage causes five additional unanswered frames before fresh
control is available. These are synthetic inputs, not observations from the match.

Clearing ownership at a genuine stoppage is intentional: the same team does not
necessarily restart. The concern is the cost of elevating a weak D hypothesis to
that irreversible reset.

The official temporal scorer forward-fills X within each graded run, including
after D: `D X X H` becomes `D D D H`. A recovery gap can therefore extend the
effective predicted stoppage beyond its last explicit D frame. Raw D duration
alone does not describe this consequence; graded-run boundaries limit filling.

Confirmed restart releases have a distinct recovery path: `pipeline.predict`
injects `restart_team` into the release frame's control evidence before calling
`decode_state`. The decoder applies reset before accepting that same frame's
team evidence, so reset does not erase the new seed. With unlimited carry, the
seed survives subsequent unknown control until new team evidence, D or another
reset. A synthetic pipeline check verifies five seconds of missing ball after
one seed, with and without the play-state component. Unseeded exits from soft
dead predictions must be diagnosed separately; their unknown-owner gaps do not
demonstrate failure of confirmed-release carry. A real confirmed release is live
evidence, so any D extension across it would be an error, not harmless abstention.

A separate invented recovery case keeps the ball within 8 cm of a stationary
player while alternating its reported position after a stoppage. Adjacent-frame
relative speeds exceed the fixed control gate, leaving 41 frames unknown before
the observations settle. This demonstrates noise sensitivity, not evidence that
the nearest player should always receive possession. Missing ball observations
and a ball travelling between players also prevent fresh direct control.

Existing tests verify correct dead/reset propagation and restart seeding. They do
not establish an evidence policy for distinguishing uncertain stoppage candidates
from sufficiently supported resets.

Next analysis: trace true stoppage/unknown restarter, false stoppage/live possession,
and noisy release cases end to end. Identify what should be retained internally
as a hypothesis while the final label remains undecided. Do not simply preserve
the old owner through every real restart.

**F06 — Soft evidence becomes hard state without using all its uncertainty**

Sources: [play_state.py](play_state.py), `uncertain`; [pipeline.py](pipeline.py),
`dead = dead | ...`; [state.py](state.py), `gap_conflict` and evidence age.

The play-state score and uncertainty flag are diagnostic. Pipeline decisions use
the decoded boolean dead mask. Restart-anchor D is also combined by OR, so that
separate hypothesis can survive contrary soft live evidence. Conversely confirmed
restart L is explicitly forced live in the play-state component. Caller-supplied
dead evidence can override it again.

This is a confirmed precedence policy, not proof that each conflict is wrong.
However, the system is not yet maintaining competing H/A/D explanations through
the whole pipeline. It has soft evidence within one component and hard decisions
at component boundaries.

Next analysis: document a common evidence contract and precedence table. Exercise
contradictory restart, physical-motion, phase and control evidence with fixed
synthetic inputs. Numerical score magnitude must not be called a probability
without independent justification.

**F07 — One-touch pass, fortunate deflection and failed pass**

Source: [interactions.py](interactions.py). A pass hypothesis requires a compatible
outgoing trajectory and later same-team control, with no intervening unrelated
interaction. Promotion is disabled in the prototype configuration.

A deliberate first-time interception can therefore be recognized late. A failed
first-time pass can lack the same-team confirmation entirely, even if the passer
briefly established control. Enabling promotion creates the opposite risk: a
fortunate deflection reaching a teammate can satisfy the same observations.

Existing paired tests intentionally show that identical trajectories cannot be
assigned different intentions. Success is not proof of intent, and failure does
not erase earlier control.

Next analysis: compare incoming/outgoing motion, feasible contact geometry and
sequence context for both successful and failed deliberate actions. Identify
which differences are observable and which would require body/video evidence or
permitted labels from other games. Do not invent a physics rule that guarantees
intent from speed gain/loss.

**F08 — Smart Tagging is used for live/dead context, not team ownership**

Sources: [play_state.py](play_state.py), `_tag_context`; [DATA.md](../DATA.md), Smart
Tagging. The current component recognizes selected tactical-phase codes but
ignores their Team, Half, Side and other tag fields. This avoids blindly assigning
possession from a broad phase; it also leaves possible contextual evidence unused.

The role of Team must be specified per code before using it for H/A. An attacking
phase and a defensive phase may refer to different roles. Team=N/A is not away,
and club-name matching must be explicit. Overlap and inaccurate phase boundaries
remain possible even with correct role interpretation.

Existing tests establish phase context can support live play and that restart Team
strings do not directly control ownership. No phase-role-to-possession model is
implemented or validated.

The current score has a stronger structural limitation than a merely weak prior:
without set-piece context, outside-pitch support or restart/episode support, all
terms are nonpositive. Stillness alone cannot create positive dead evidence.
For ordinary tagged preparation with an overlapping live phase and no independent
outside/restart support, even the optimistic score bound is negative:
`-0.8 + 0.6 + 0.9 + 0.3 - 1.4 = -0.4`. This bound generously allows both maximum
quiet-ball preparation and maximum missing-ball support simultaneously; actual
evidence can be weaker. Thus a continuous coarse phase can veto a local stopped
setup regardless of its quietness. Temporal decoding may still bridge such
frames from surrounding evidence; a nonpositive local score is not itself a
guaranteed final live label. An invented stationary setup produces D with SET
PIECES alone, and no D when BUILD UP covers the same sequence. Neither result
establishes which label is correct for a real tagged interval.

Next analysis: produce a code/role/available-fields table from official source
definitions. Mark undocumented semantics unknown. Then test contradictory and
overlapping artificial phases, N/A and name mapping. Do not assume an attacking
phase boundary is a possession boundary or automatically invert every defensive tag.

**F09 — Temporal consistency can blur real boundaries and short states**

Sources: [play_state.py](play_state.py), centered fits and Viterbi; [contacts.py](contacts.py),
two-sided fits; [interactions.py](interactions.py), limited lookahead/backdating.

A centered motion window mixes observations before and after an event. Transition
costs favor fewer live/dead changes. Contact windows need enough reliable samples
on both sides and can miss closely spaced touches. Reception backdating handles
some observed contacts, not all changes.

Consequence: an apparently stable timeline can hide a real short possession, a brief
stoppage, or a timing error. No explicit minimum possession duration does not mean
the complete system preserves every short interaction.

Next analysis: hold physical event times fixed and vary sampling phase, observation
gaps and nearby-event spacing. Check boundary bias and event survival on those
synthetic inputs before considering any new decoder or parameters.

**F10 — A track slot is not a persistent player identity**

Sources: [DATA.md](../DATA.md), track slots; [contacts.py](contacts.py), movement
guards; [interactions.py](interactions.py), same-slot confirmation.

Missing samples and large jumps are guarded, but a plausible-looking identity
switch can survive speed checks. Requiring the same slot to confirm a reception
can also miss genuine control when tracking fragments into a different slot.
Team membership may remain useful even when personal identity does not.

Existing tests cover a new slot and implausible jumps. They do not establish robust
handling of plausible switches, overlapping duplicate tracks or repeated fragments.

Next analysis: apply artificial fragmentation, duplication and within-team slot
permutations to an otherwise identical play. Team possession should be as invariant
as the remaining evidence permits; event attribution may appropriately remain unknown.

**F11 — Tracking coordinates outside the pitch are fallible evidence**

Source: [play_state.py](play_state.py), outside-position votes and pitch margin.
The feature protects against isolated errors but cannot establish actual ball-out
status if the position estimates are systematically wrong. The pipeline does not
use ball height or referee observations to resolve that ambiguity.

Consequence: a persistent tracking/projection bias could produce false D, while
an unobserved genuine exit could be missed. This is a hypothesis about error modes,
not a claim that such bias has been measured on the provided game.

Next analysis: synthetic persistent coordinate offsets, boundary jitter, unseen
exits and recovery to valid tracking. Separate feature reliability from a legal
ball-out decision. Adding a larger margin alone does not solve systematic error.

**F12 — Specialized interactions share unresolved control questions**

- **Aerial duel/flick-on:** no dedicated height-aware control model. Winning a
  contact without subsequent team control must not automatically become a turnover.
- **Goalkeeper catch/parry:** no goalkeeper-specific secure-control versus rebound
  reasoning. A large velocity change does not determine which occurred.
- **Clearance:** no classifier distinguishes emergency contact from established
  control followed by clearing. Intention to escape danger is not itself control.

Next analysis: enumerate the observable differences and identical-observation
limits for each pair. Reuse the shared contact/control/continuity reasoning where
possible. Additional video/body evidence is a research option, not an assumption
that the current position-only inputs can solve every case.

**What the existing tests establish**

They check stated behavior, schema/alignment, deterministic output, missing values,
selected football scenarios and official CSV validation. Some explicitly passing
tests document known failure modes. A passing test count is not an accuracy claim,
and synthetic geometry selected to match an implementation is weak validation.

Missing analysis coverage includes broad noise processes, correlated errors,
non-random occlusion, contradictory cues, feature invariance under track changes,
and uncertainty propagation across modules. Keep shape/format checks separate
from football-behavior checks when describing evidence.

**Recommended next analysis, before implementation resumes**

1. Audit motion reliability (F01) using stationary jitter versus real reversals.
2. Specify candidate/control/possession evidence and the cost of wrong transitions
   (F02, F03, F05, F06), including first-time and failed actions.
3. Build the gap-evidence and Smart Tagging role inventories (F04, F08).
4. Define the boundary, fragmentation and specialized-interaction probes (F07,
   F09-F12), then select a narrow implementation hypothesis with clear limitations.

The ordering follows demonstrated counterexamples and dependencies, not public
F1 values. Real-match error frequency, expected score gain and full-match ranking
are unknown. Do not sweep weights, choose architectures or change thresholds
against the public ground truth. Use it only for the official permitted pipeline
checks; any independent labeled evaluation must comply with the competition rules.

**Exit criteria for the analysis pause**

- Agree on what constitutes candidate evidence versus an established state change.
- Identify the first falsifiable hypothesis and its counterexample pair.
- Specify which existing behavior must survive and what remains unidentifiable.
- Document the permitted validation source before selecting model settings.

No model changes or score optimization are proposed as part of creating this map.
