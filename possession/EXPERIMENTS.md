# Development boundary for the temporal play-state prototype

> **Historical.** This log covers the rule-based prototype (`run.py`, `pipeline.py` and the
> modules it wires together), which is **superseded** and does not produce the submission. The
> final model is **learned_v6** (fallback learned_v4); see [README.md](README.md) and the technical document
> [PossessionIQ.md](../PossessionIQ.md). Kept unchanged below as a record of the approach.

The restart-only detector demands an unusually complete preparation/release
sequence. That is a structural limitation under missing and noisy observations,
independently of any public-label score. The new hypothesis is that alternative,
imperfect physical and automatic-tagging cues can support ball-in-play inference
through a temporal model.

The play_state settings in configs/prototype.json are explicit hand-set modeling
priors. They are not trained or calibrated probabilities, and are not selected
using public labels, annotated events, public transitions or leaderboard results.
No parameter sweep or candidate selection against the public score is permitted.

Behavioral checks use invented scenarios: tags or missingness alone are
insufficient; quiet ball OR player setup may support tagged preparation; live
movement and tactical-phase tags can contradict it; broad set-piece intervals
can include live play; neutral gaps preserve state; resets separate sequences.
Quiet ball and quiet players share a preparation channel rather than being
double-counted as independent evidence. Known ambiguous cases remain documented
failure modes.

After these checks, local full-match input diagnostics may verify execution,
coverage and physical plausibility. Aggregate coverage is not an accuracy metric
and is not a target to maximize. Do not adjust settings merely to generate a
desired amount of D. The public ground truth is for pipeline correctness checks,
not architecture/threshold/weight selection or iterative score optimization.

Separate full-file diagnostics from diagnostics inside the published half
boundaries in games.half_video_blocks / DATA.md. Pregame, halftime and postgame
coverage must not be described as scoring damage. These published timeline
boundaries are not possession labels. Report the scopes separately before
prioritizing long gaps, and keep censored recovery intervals separate from
observed recovery delays. A D-X-D pattern alone does not prove one continuous
stoppage or a decoder error; inspect the underlying score and observation support.

Keep existing baselines unchanged. Persist the effective config with each local
prediction; retain the chosen configuration before any permitted scoring. Do not
publish match-derived outputs. This version has no demonstrated score improvement.

The bounded-episode follow-up adds retrospective onset hypotheses without changing
the existing control/play-state weights. Compatible observed exits or weaker group
pauses are constrained by resets, intervening releases and coherent live sequences.
Goal context needs physical goal-line support and a supported after-goal kickoff.
Validation uses invented whole stoppage episodes and counterexamples. Search limits
are unvalidated priors, not values selected against public labels or desired D
coverage. A fixed current-version checkpoint may be regenerated and scored for
the permitted pipeline check; do not modify/select its settings from that result.

Broader candidate stoppages are a separate proposed experiment. A candidate must
combine SET PIECES context, an observable physical onset and a restart-like action,
which may be partially hidden. Its uncertainty must enter inference at lower
strength than a confirmed restart, with coherent live passing able to contradict
it. Audit added D duration, added splits, and conflicts with independently derived
live-play evidence. Such conflicts are proxies for false alarms, not measured
false-positive rates without permitted independent labels. Finish and score the
fixed current checkpoint first. The subsequent opt-in candidate configuration
preserves existing model weights and is kept outside default scoring inputs;
collect input-only change audits without scoring/selecting it against public labels.

The supported-exit experiment adds a separate opt-in resumption configuration,
preserving all previous evidence weights and thresholds. Hypothesis: generic
movement, broad phase context or a default live bias should not alone end a
previously proposed stoppage. Local separating ball departures, compatible dribbles
with group activity, or coherent fast ball travel with group activity can support
resumption; strict live anchors bypass the check. No team is inferred from these
actions. Counterexamples cover ball retrieval, player walking, nonlinear ball
motion, hidden launch starts, resets and quick/partially observed restarts.

Unsupported exits retain explicit D only for a bounded duration, then emit X with
uncertain play state. Five seconds is an unvalidated bound, not a duration selected
from competition scores, labels or desired coverage. X can still extend D in the
temporal scorer. Audit changes separately inside/outside published halves, new D
inside previous team runs, live-evidence conflicts, newly forced X, and strict
restart seed preservation. Do not adopt or alter parameters to maximize D coverage.

The short-gap follow-up changes evidence access and precedence, not numerical
settings: permit a supported physical action spanning an occluded current frame,
and wire the existing coherent-transfer mask into the resumption check. Existing
gap, coverage, displacement, fit, velocity and activity gates remain unchanged.
There must be real observations on both sides; no boundary is backdated before
the first observed movement. Test long/unbracketed gaps, hidden kick starts,
implausible motion, earlier/later stoppages, reset boundaries, strict dead priority,
and guard propagation into final labels before local input-only execution.
The saved earlier candidate remains a frozen diagnostic reference. Compare event
concentration and proxy conflicts without adjusting parameters to erase selected
match conflicts, force more D, or improve the public score.

The next control experiments isolate two changes. First, supplement accepted
adjacent-frame control with a short fit of relative position in fully observed,
contact/reset-bounded windows. Existing accepted direct evidence has priority;
the distance, relative-speed and opponent-margin gates remain unchanged. Five
samples and a 0.1 m maximum fit error are unvalidated priors, fixed before local
execution (the error scale matches the existing contact-fit setting). Synthetic
checks cover jitter, fast fly-bys, reversals, close opponents, short possessions,
missing observations, contacts, resets and shared implausible jumps.

Second, enable the existing optional pass-hypothesis promotion in a separate
configuration. This tests contact/departure/reception evidence without changing
the dribble velocity requirement. A successful deflection can be observationally
identical: the permissive policy must expose that counterexample and is not
claimed to identify intent. Preserve the conservative candidate and the existing
submission; do not choose among candidates or tune thresholds using public
labels/scores or desired coverage. Local runs inspect source counts, label
changes, unknown recovery and possible extra turnovers, split by published halves.
