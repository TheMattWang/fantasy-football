# T2 — Have we been optimizing the small decision?

**Verdict: KEPT (in-season management is the larger lever), with two caveats that
both shrink the magnitude. The direction is solid; the size is soft.**

> **REVISED by T3 (`t3-realistic-in-season.md`), 2026-08-16.** Caveat 2 below was
> the right worry and it was larger than expected. Against a *realistic* manager
> rather than a frozen lineup, in-season management is worth **+1.523 ranks**, not
> 4.457. Matched against the realistic draft improvement of 0.832 that gives a
> ratio of **1.8×**, which is *below* the 2× threshold pre-registered here. **Do
> not quote the 5.4×.** The conclusion still holds for this project, but it rests
> on the no-edge result (our own realistic draft improvement is ≈ 0, not 0.832),
> not on the ratio.

Run 2026-08-16. Script: `experiments/worker_t2.py`.

## Why

Three numbers existed and none were in the same units, so the comparison had never
been made:

| measurement | value | units |
|---|---|---|
| one shortlist swap (B1b) | ~3 projected points | points |
| perfect start/sit across rosters (stage 0a) | 2.17 | points/week |
| schedule luck with identical teams (V7) | sd 1.45 | wins |

Every strategy result in this project is in **ranks**. Converting all three into
ranks was the experiment, not a preliminary to it.

## Method

One league held fixed — same 12 rosters, same 2000 season samples, same schedule,
same waiver draws (CRN throughout) — changing exactly one thing per arm.

- **Draft lever.** Sweep our roster's quality over 250 draws, regress mean rank on
  summed starter projections. The slope is ranks-per-projected-point.
- **Management lever.** Give our team `omniscient=True` lineups against ex-ante
  opponents, paired, 60 rosters.
- **Luck floor.** Twelve *equally drafted* rosters, each with an independent draw.

## Results

```
LEVER 1 -- THE DRAFT
  slope 0.0136 ranks per projected point
  one shortlist swap (3 pts)          0.041 ranks
  median -> 90th pct draft (61 pts)   0.832 ranks
  worst -> best in sweep (293 pts)    4.000 ranks

LEVER 2 -- IN-SEASON MANAGEMENT (perfect start/sit)
  rank gain +4.457 (se 0.044)
  points gain +308 per season (+21.97 pts/wk)

LUCK FLOOR -- twelve equally-drafted rosters
  sd of final rank 3.49, P(playoffs) 0.425

  scope-matched ratio   5.4x
  per-single-pick ratio 108.9x   <- not a fair contest, see below
```

The pre-registered threshold was 2×. The scope-matched ratio is **5.4×**.

## A bug I made, and what it cost

The first run reported the luck floor as **sd 0.00, P(playoffs) 1.000**, which is
impossible. I had built the identical-roster arm as `np.repeat` of one score
vector twelve times. Copying makes every head-to-head game a tie — `mine > theirs`
is false in both directions — so every team finishes 0-14 and rank falls out of
array order, handing team 0 first place in every sample. Fixed by drawing twelve
independent rosters of the same quality.

The first run also headlined **108.9×**, comparing *one draft pick* to *a whole
season of perfect lineups*. That is one decision against 14 weeks × 9 slots of
them, and it is not a fair contest. The scope-matched comparison — a realistic
draft-quality improvement against a season of management — is 5.4×.

## Two caveats that both shrink the number

**1. The management ceiling is far looser than the draft ceiling.** `omniscient=True`
grants perfect foresight on every slot every week: 126 perfect decisions. The draft
lever's median→p90 is a realistic improvement. Note that worst→best across the whole
sweep is **4.000 ranks**, essentially equal to the management lever — so the two
ceilings are comparable when both are taken to their extremes.

**2. The ex-ante baseline is worse than a real manager.** The simulator freezes
lineups all season (`decision_score` is the preseason projection and never updates),
so the comparison is perfect-vs-**frozen**, not perfect-vs-realistic. A real manager
benches busts and starts breakouts. The achievable remaining gain is therefore
meaningfully smaller than 4.457, and by an unmeasured amount.

## Why the conclusion survives the caveats

The ratio is soft but the asymmetry is not, and it does not depend on the exact
number:

- **The draft lever is one we have measured ourselves unable to move.** V1–V7
  tested position/rank bias, market deviation, rookie status, draft capital, age,
  injury history, three flavours of schedule, team offence, and six usage metrics.
  Everything null but usage (t = 1.78, unresolved). With no valuation edge our
  realistic draft improvement is ≈ 0, because consensus is already optimal given no
  edge.
- **The management lever is one we have never touched at all.** It is not that we
  tried and failed; the simulator cannot currently express the decision.

So even at a ratio near 1, in-season would be the better place to work — the draft
headroom is already exhausted while the management headroom is entirely unexplored.

## What follows

Phase 5 should be start/sit and waivers rather than drafting. The first step is not
a strategy: it is making the simulator able to represent the decision at all, by
replacing the frozen `decision_score` with a running update from observed weeks.
Then measure what a *realistic* in-season policy captures of the 4.457, which is the
number that actually matters and which this experiment does not provide.
