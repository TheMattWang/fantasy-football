# V9 — practice participation predicts availability, and does not improve the lineup

**Verdict: KILLED. Built, measured, decision-tested, reverted. A third independent
instance of the estimation-versus-selection split this project keeps finding.**

Run 2026-08-25. Scripts: `experiments/worker_availability.py` (the table),
`worker_practice_decision.py` (both cells), `worker_practice_cell.py` (each alone).

## The hypothesis, and it looked strong

`availability()` keys only on `report_status`. Friday practice participation splits
those designations further, and the prediction side is emphatic:

```
  status        practice        n  P(played)  pts|played  multiplier
  (none)        Full         2766      0.852        8.61       0.847
  (none)        Limited       465      0.886        9.26       0.948
  (none)        DNP           292      0.661        7.94       0.606
  Questionable  Full          360      0.636        6.13       0.450
  Questionable  Limited      1062      0.576        7.35       0.489
  Questionable  DNP           263      0.418        6.50       0.314
```

Two cells looked like free money:

* **292 player-weeks carry no game-status designation at all yet the player did not
  practice.** The code scored every one at **1.0**, fully healthy, when he plays 66%
  of the time and scores less when he does — 0.72 like-for-like. A missing
  designation is not a clean bill of health.
* **Questionable-and-DNP plays 41.8%** against 63.6% for Questionable-and-Full. One
  pooled 0.542 covers both.

## The decision test

Fully implemented — practice-aware multiplier table, `injury_report` retaining
no-designation rows, dedup breaking ties toward the more severe practice status, four
new tests, 289 passing. Then scored against what actually happened: 60 rosters x weeks
3-14 x four seasons, in points, no simulator in between.

```
  season    roster-wks  changed   mean d      se      t   win%  per r-wk
  2022             720       28    +1.32    2.18  +0.61    54%    +0.051
  2023             720       21    -0.49    2.51  -0.19    45%    -0.014
  2024             720       21    -1.89    2.10  -0.90    40%    -0.055
  2025             720       20    -0.25    1.82  -0.14    58%    -0.007

  pooled over 90 changed lineups: -0.199 pts each
  SEASON-CLUSTERED per roster-week: -0.0062 (se 0.0220, t = -0.28)
  seasons positive: 1/4
```

Decomposed, in case the cells were cancelling:

| cell | per changed lineup | seasons positive | season-clustered t |
|---|---|---|---|
| no designation + DNP (0.721) | **−0.850** | 2/4 | −0.72 |
| Questionable + DNP (0.314) | +1.627 | 3/4 | +1.32 |

They were not cancelling. The cell with the **strongest** prediction — 0.606 against a
shipped 1.0, the single largest mispricing in the table — is the one that makes the
decision **worse**. The other is positive but at t = +1.32, short of this project's
|t| >= 2 bar, and V6's usage signal was called "not established" at t = +1.78 and later
killed outright.

**Reverted.** The code is gone; only this document and the scripts remain.

## Why this keeps happening

Third time, same shape:

| | prediction | decision |
|---|---|---|
| board vs persistence | +22.4 pts/pick | no selection edge |
| V8, usage in-season | t = +6.10 | +0.072 pts/wk, 47% win rate |
| **V9, practice participation** | **0.606 vs 1.0** | **−0.199 pts, 1/4 seasons** |

The mechanism is the same each time and it is not subtle once seen. Being right about a
player's *level* only pays when it changes *which player you start*, and the candidates
are close. A DNP discount moves a bench player past a starter only when the two were
already near-tied — precisely where the ranking is noisiest and the true gap smallest.
Sharpening an estimate in that region mostly reshuffles ties.

Note the contrast with R5, which shipped the same week at a similar effect size
(+0.076 pts/roster-week). That was not a new signal — it corrected a ratio whose
numerator and denominator were in different units, and it would have been right even
had it never moved a lineup. This one is a claim about football, and claims about
football have to clear the decision bar. It did not.

## What would change the answer

A live game-day feed. nflverse carries the **Friday** report; final inactives land ~90
minutes before kickoff and are not in it. The value of `Questionable -> 0.542` exists
*because Friday is uncertain* — a real inactive list does not sharpen that estimate, it
**collapses** it to 0 or 1. That is a different intervention from re-weighting Friday
information, and it is the one worth trying next.

Before writing any polling code, the size of the prize is measurable for free:
`weekly_rosters.status == 'INA'` gives realised inactives (~6 per team per week), so
joining Friday `report_status` against realised `INA` on `gsis_id` prices
P(inactive | Questionable) with data already on disk.
