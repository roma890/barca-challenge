# Possession inference experiments

Hypothesis: treating missing evidence as uncertainty rather than automatically
marking it dead can reduce false possession splits. Match-level improvement has
not been established.

`state.decode_state` takes separate arrays for established team control (`H/A/X`)
and dead-ball evidence (boolean). It carries the last team through a configurable
number of unknown frames, then emits `X`. Dead-ball evidence takes priority and
clears team memory. Optional reset flags prevent carry across recording breaks.
The returned `carried` mask distinguishes carried estimates from direct evidence.

These are components, not a complete submission pipeline. Passing nearest-player
labels as established control would retain the baseline's false-turnover problem.
The gap limit is an explicit experimental parameter, not a tuned default. No
minimum possession duration is imposed.

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
same dead/reset masks and an explicit carry limit. `source` distinguishes direct
motion, incumbent continuity, backdated receptions and promoted pass hypotheses.
This is offline inference; contact windows and reception searches use future
frames. No minimum possession duration is applied. None of these settings have
been selected using the public ground truth or restricted match data.

The official spell metric forward-fills abstentions within graded runs, so `X`
does not stop a spell or resolve a long gap for that metric. Independent stoppage
evidence is still needed. The public five minutes must not be used for tuning.

Run the synthetic checks from the repository root:

```sh
python -m unittest discover -s tests -v
```

Next: implement separate stoppage evidence and a local tracking adapter, then
connect these components to the official submission writer and validator.
Preserve the official baseline for comparisons. Tests here read no competition
data; no match-level improvement is claimed.
