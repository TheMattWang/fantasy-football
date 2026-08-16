# V8 — Usage as an in-season signal

**Verdict: PREDICTION CONFIRMED, DECISION KILLED. Opportunity predicts next
week's points at t = +6.10 after controlling for points. Backtested on the actual
start/sit decision it is worth +0.072 points per week, wins in 47% of rosters, and
flips sign across seasons. Not wired into `week.py`.**

Run 2026-08-16. Scripts: `experiments/worker_v8.py`, `experiments/worker_v8b.py`.

## Why

V6 killed usage as a *draft* signal (target_share t = +1.78, short of the bar) —
but that asked a hard question against the market's home turf: "did 150 analysts
misprice this in August?" The in-season question is easier, because the opponent
is not the market. It is the naive baseline `week.py` actually uses: **points per
game so far.**

The mechanism gives the prior:

```
points = opportunity x efficiency
```

Opportunity is a coaching decision and persists. Efficiency is mostly noise and
mean-reverts. A points-average cannot tell a player who scored three touchdowns on
four targets from one seeing twelve targets a game with nothing to show for it.

## V8 — prediction (12,091 player-weeks, 663 players, SEs clustered by player)

```
points only (what week.py uses)          R^2 = 0.3257
  ppg_so_far   +3.917  t +56.38

points + opportunity                     R^2 = 0.3308  (+0.0051)
  ppg_so_far   +3.220  t +24.60
  opp_so_far   +0.852  t  +6.10   <-- clears the pre-registered bar

points + opportunity + efficiency        R^2 = 0.3320  (+0.0063)
  opp_so_far   +0.468  t  +2.59
  eff_so_far   -0.331  t  -5.28   <-- NEGATIVE, exactly as theory predicts

by position (partial t on opportunity):  RB +5.72   WR +8.83   TE +6.03
```

The efficiency coefficient is the satisfying part: scoring efficiently so far
**predicts regression**, which is precisely what "points = opportunity ×
efficiency, and only the first half persists" says should happen.

## V8b — the decision (walk-forward, real outcomes, no simulator)

Fit on prior seasons only; draw random rosters; each week pick slot-legal starters
by each policy and sum what they actually scored.

```
season     rosters   pts/week      se      t   pts/season
2023           300     +0.120   0.088  +1.36        +1.6
2024           300     -0.057   0.070  -0.81        -0.7
2025           300     +0.154   0.063  +2.46        +2.0
pooled         900     +0.072   0.043  +1.68        +0.9

rosters where usage won: 47%
```

**+0.9 points per season**, against a season total near 1500 — six hundredths of
one percent. The sign flips across seasons, and usage loses more often than it
wins despite a positive mean.

## The reconciliation, which is this project's own lesson repeating

t = +6.10 and +0.072 points a week are both correct. They measure different things:

- **V8 measured estimation** — how well the number predicts, across all
  player-weeks.
- **V8b measured selection** — whether it changes *which six of twelve you start*.

The players you choose between in a start/sit call are close together, so a tiny
prediction improvement rarely flips the right call, and when it does the gain is
small. This is exactly §3 of `WHAT_ARE_WE_OPTIMIZING.md`: the board had a +22.4
pts/pick *estimation* edge and no *selection* edge. **Same distinction, second
independent instance.**

## A methodological error of mine, recorded

V8's pre-registered criterion was "kill if the partial t on opportunity is under
2." It passed at 6.10 — and the claim still died, because **I pre-registered a
prediction criterion for a selection problem.** The decision-relevant question was
never "does opportunity predict points" but "does it change the lineup." Getting
that wrong is the same error class as the project's original one.

Any future pre-registration for a decision must state its threshold in the units
of the decision.

## Not wired in

`week.py` keeps ranking on points-so-far. The effect is positive in the mean but
indistinguishable from zero (t = 1.68), sign-flipping, and wins under half the
time. Adding a second predictor for that is complexity with no measured payoff,
and V3 already established that this project's deviations have historically run
negative.

## What it does close

T3 left +3.447 ranks between a simple shrinkage policy and perfect foresight, and
named usage trend as the obvious next input. That candidate is now spent. Whatever
lives in the remaining headroom, it is not opportunity volume.
