# T3 — How much of the in-season ceiling does a realistic manager capture?

**Verdict: KEPT, but it revises T2 downward. A realistic in-season policy is worth
+1.523 ranks against a realistic draft improvement of +0.832 — a ratio of 1.8×,
which is *below* T2's pre-registered 2× threshold, not the 5.4× T2 reported.**

Run 2026-08-16. Script: `experiments/worker_t3.py`.

## Why

T2 measured perfect start/sit at +4.457 ranks and flagged the objection against its
own result: the baseline was a **frozen** lineup, not a real manager. Real managers
bench busts. So the gap between what people actually do and perfect was smaller than
4.457 by an unmeasured amount. This measures it.

`_reactive_estimate` supplies the missing decision: rank by a running blend of the
preseason prior and points so far, weighting the prior as `k` games.

## Results

```
ARM A -- us reactive, opponents FROZEN (overstates)
  k=1     +1.432 (se 0.036)      k=8     +1.406 (se 0.043)
  k=2     +1.492 (se 0.043)      k=16    +1.274 (se 0.046)
  k=4     +1.478 (se 0.050)

ARM B -- everyone reactive at k=2 (the realistic league)
  managing vs not          +1.523 (se 0.045)

ARM C -- what is left above a competent manager
  perfect vs a good manager   +3.447 (se 0.030)
  perfect vs a frozen lineup  +4.852 (se 0.055)

  fraction of the ceiling a realistic policy captures   29.0%
```

## Readings

**The policy is robust, not tuned.** The k-sweep spans 1.27 to 1.49 across a 16×
range of stubbornness. There is no knife-edge, so this is not a fitted result and
`k=2` is not load-bearing.

**Managing is worth ~1.5 ranks and does not collapse when opponents manage too.**
Arm B (+1.523) slightly *exceeds* Arm A (+1.492). I expected the relative gain to
shrink once the field also set lineups; it did not, because the comparison is paired
within a fixed league — our own improvement is roughly rank-preserving regardless of
how good the field is.

**A realistic policy captures only 29% of the perfect-vs-frozen ceiling.** Perfect
foresight remains worth +3.447 ranks even against a competent manager. That residual
is not achievable — nobody knows next week's scores — but it does say the in-season
lever is not exhausted by simple shrinkage. Matchup, injury news and usage trend are
all unexploited, and how much of the 3.447 they reach is unknown.

## This revises T2

T2's headline ratio was **5.4×**, comparing perfect management to a realistic
draft improvement. That was not realism-matched. Matching both sides:

| lever | realistic value |
|---|---|
| draft, median → 90th percentile (T2) | **+0.832 ranks** |
| in-season management (T3, arm B) | **+1.523 ranks** |
| ratio | **1.8×** |

**1.8× is below T2's pre-registered 2× threshold.** On the criterion as written,
T2's verdict does not survive contact with a realistic baseline.

## Why the direction still holds for us specifically

The 0.832 is what a *median* drafter could gain by drafting like a *90th-percentile*
drafter. It is not available to us. V1–V7 measured no valuation edge: schedule,
usage, rookie status, draft capital, age, team offence, market deviation, all null
or negative. With no edge, consensus is already optimal, so **our realistic draft
improvement is approximately zero, not 0.832.**

The 1.523 is fully available, because it requires no market edge at all — only
reacting to what has already happened.

So the honest comparison for this project is **1.523 against ~0**, and the
conclusion survives. But it survives on the no-edge result, not on the ratio, and
the ratio should not be quoted as 5.4×.

## What follows

The in-season lever is real, achievable, and worth roughly 1.5 ranks. Two next
steps, in order:

1. **Make it usable.** This currently exists only inside the simulator. There is no
   tool that tells you who to start on a Sunday morning. `src/inseason/waivers.py`
   exists; a start/sit counterpart does not.
2. **Try to reach further into the 3.447.** Simple shrinkage over own-scoring is the
   weakest possible reactive policy. Usage trend is the obvious next input — and
   notably it is the one signal V6 found any life in (t = 1.78), where it failed as
   a *draft* signal but was never tested as an in-season one.
