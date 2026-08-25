# R5 — the Questionable multiplier divided by the wrong thing

**Verdict: KEPT. 0.456 -> 0.542. Worth +0.076 points per roster-week, positive in
all four seasons (season-clustered t = +4.50) — but that figure is almost exactly
V8's +0.072, which was killed, so the case rests on it being an arithmetic fix
rather than a new signal.**

Run 2026-08-25. Scripts: `experiments/worker_availability.py` (the table),
`experiments/worker_availability_decision.py` (the decision test).

## The error

`STATUS_MULTIPLIER["Questionable"] = 0.456` came from

    P(played | Q) x mean_points(Q) / mean_points(no report)
    0.567 x 6.95 / 8.66 = 0.456

The denominator is what a healthy player scores **when he plays**. But a healthy
player only takes the field **84%** of the time, so this compares an *expectation*
in the numerator against a *conditional mean* in the denominator. Like for like:

    E[pts | Questionable]  = 0.567 x 6.95 = 3.945
    E[pts | no report]     = 0.840 x 8.66 = 7.277
    ratio                  = 0.542

Questionable players had been ranked **19% below** where the data puts them.

Ironic given the module's own framing: it was written because "every hand-written
constant in this project has eventually turned out to be wrong", and it measured both
halves of the numerator carefully — then dropped one half of the denominator.

## Does it move a lineup?

Guardrail: test on the decision, not the estimate. 60 rosters x weeks 3-14 x four
seasons, scored against what players actually did, no simulator in between.

```
  season    roster-wks  changed   mean d      se      t   win%  per r-wk
  2022             720       33    +0.88    1.48  +0.59    52%    +0.040
  2023             720       28    +2.39    1.84  +1.30    59%    +0.093
  2024             720       31    +1.30    1.45  +0.89    56%    +0.056
  2025             720       24    +3.41    2.54  +1.34    68%    +0.114

  pooled over 116 changed lineups: +1.881 pts each
  SEASON-CLUSTERED per roster-week: +0.0758 (se 0.0168, t = +4.50, n = 4)
  seasons positive: 4/4
```

**Read 2025 alone and you get +3.41 a change at a 68% win rate. That is the best of
the four seasons and quoting it would be cherry-picking** — the honest number is
**+1.88** pooled, with win rates from 52% to 68%.

## The comparison that decides it

**+0.076 points per roster-week is essentially V8's +0.072, and V8 was killed.** That
has to be confronted rather than glossed. Three differences, and they are the whole
argument:

1. **V8 won 47% of the time** — worse than a coin flip, sign flipping by season. This
   is positive in 4 of 4 seasons at a 52-68% win rate.
2. **V8 proposed a new claim about football** (usage predicts scoring, so start the
   high-usage player). This proposes no claim at all — it fixes a ratio whose
   numerator and denominator were in different units. It would be right even if it
   never changed a lineup.
3. The season-clustered interval excludes zero here and did not there.

So it ships, but on correctness grounds with the decision value as corroboration —
not as an edge. Anyone reading this later should not treat +0.076 pts/roster-week as
evidence that small effects are worth chasing. It is evidence that arithmetic is.

## Still open

Friday practice participation splits the designations further, and one cell is
mispriced badly:

```
  status        practice        n  P(played)  pts|played  multiplier
  (none)        DNP           292      0.661        7.94       0.606
  Questionable  DNP           263      0.418        6.50       0.314
  Questionable  Limited      1062      0.576        7.35       0.489
  Questionable  Full          360      0.636        6.13       0.450
```

**292 player-weeks carry no game-status designation at all yet did not practice**, and
the code scores every one of them as fully healthy at 1.0 when the data says ~0.72.
Questionable-and-DNP is 0.314 against the pooled 0.542.

Not shipped: `availability()` keys only on `report_status` today, and adding
`practice_status` is a code change that has not been decision-tested. Queued rather
than assumed — which is the discipline that killed V8 and should apply here too.
