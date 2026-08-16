"""B1b -- the marginal version of B1, which is the one that matters.

B1 found Spearman(U, summed starter projections) = 0.979 over rosters spanning
near-random to best-available. That range is enormous, and over a wide enough
range almost any monotone relationship looks tight: of course better teams win
more. It does not follow that U is uninformative AT THE MARGIN.

But the margin is the entire decision. A draft pick chooses between rosters that
differ by ONE player drawn from an ADP-adjacent shortlist. So the question B1
should have asked is:

    Given two competently drafted rosters differing by about what one shortlist
    swap is worth, can U tell them apart?

Two measurements:

  1. RANGE RESTRICTION. Re-run B1 over competently drafted rosters only. If the
     relationship survives, it is not an artifact of including absurd rosters.

  2. NOISE-EQUIVALENT PROJECTED POINTS. Fit U on projected points, take the
     residual sd, and divide by the slope. That converts U's unexplained
     variation into the units of the board: "U's noise is worth N projected
     points". Compare N to what one shortlist swap actually buys. If the noise
     is larger, U cannot resolve the decision -- and E2's finding that U's
     argmax is no better than random stops being an empirical surprise and
     becomes arithmetic.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mattwang/Documents/fantasy/fantasy-football")

import numpy as np
import pandas as pd

from src.data.league_config import provisional_config
from src.simulation.distributions import build_samples
from src.simulation.season import evaluate_roster, lineup_points, plan_roster

BOARD = "data/processed/board_2026.csv"
N_OPPONENTS = 11
EVAL_SAMPLES = 400
N_WEEKS = 14
SHAPE = {"QB": 2, "RB": 5, "WR": 5, "TE": 1, "K": 1, "DEF": 1}
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
CFG = provisional_config()


def draw_roster(pools, rng, greed):
    picked = []
    for pos, count in SHAPE.items():
        pool = pools.get(pos)
        if pool is None or not len(pool):
            continue
        weight = np.exp(-greed * np.arange(len(pool)) / max(len(pool), 1))
        weight /= weight.sum()
        idx = rng.choice(len(pool), size=min(count, len(pool)),
                         replace=False, p=weight)
        picked.extend(pool.iloc[idx]["player_name"].tolist())
    return picked


def starter_points(frame, roster):
    sub = frame[frame["player_name"].isin(roster)]
    total, leftovers = 0.0, []
    for pos, count in STARTERS.items():
        block = sub[sub["position"] == pos].sort_values("proj_points", ascending=False)
        total += float(block.head(count)["proj_points"].sum())
        leftovers.append(block.iloc[count:])
    flex = pd.concat(leftovers)
    flex = flex[flex["position"].isin(("RB", "WR", "TE"))]
    return total + (float(flex["proj_points"].max()) if len(flex) else 0.0)


frame = pd.read_csv(BOARD)
samples = build_samples(frame, n_samples=EVAL_SAMPLES, n_weeks=N_WEEKS, seed=0)
draftable = frame[frame["proj_points"].notna()].sort_values("adp_rank")
pools = {p: g.head(60).reset_index(drop=True) for p, g in draftable.groupby("position")}
rng = np.random.default_rng(7)

opponent_weekly = np.stack([
    lineup_points(samples, plan_roster(draw_roster(pools, rng, 6.0), samples, CFG),
                  n_samples=EVAL_SAMPLES)
    for _ in range(N_OPPONENTS)
])


def sweep(greed_lo, greed_hi, n, label):
    rows = []
    for _ in range(n):
        roster = draw_roster(pools, rng, float(rng.uniform(greed_lo, greed_hi)))
        res = evaluate_roster(roster, samples, CFG, opponent_weekly,
                              n_samples=EVAL_SAMPLES)
        rows.append({"U": res.utility, "proj": starter_points(frame, roster)})
    d = pd.DataFrame(rows)
    sp = float(d["proj"].corr(d["U"], method="spearman"))

    x, y = d["proj"].to_numpy(), d["U"].to_numpy()
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    r2 = 1 - float((resid**2).sum()) / float(((y - y.mean())**2).sum())
    slope = beta[1]
    noise_pts = float(resid.std()) / slope if slope > 0 else np.nan
    print(f"  {label:<26}{len(d):>5}{d['proj'].std():>10.0f}"
          f"{sp:>10.3f}{r2:>8.3f}{noise_pts:>12.0f}")
    return noise_pts


print("B1b  DOES U SEPARATE ROSTERS AT THE MARGIN?\n")
print(f"  {'roster population':<26}{'n':>5}{'sd(proj)':>10}"
      f"{'spearman':>10}{'linR2':>8}{'U noise=':>12}")
print(f"  {'':<26}{'':>5}{'pts':>10}{'':>10}{'':>8}{'N pts':>12}")
wide = sweep(0.5, 14.0, 250, "wide (random..best)")
mid = sweep(6.0, 14.0, 250, "competent")
tight = sweep(11.0, 14.0, 250, "near-optimal")

# What does one shortlist swap actually buy? Shortlist candidates are adjacent
# in ADP, so take the gap between consecutive draftable players at a position.
gaps = []
for pos in ("RB", "WR", "TE", "QB"):
    p = draftable[draftable["position"] == pos].head(40)["proj_points"].to_numpy()
    gaps.extend(np.abs(np.diff(p)))
swap = float(np.median(gaps))

print(f"\n  one shortlist swap is worth about {swap:.0f} projected points")
print(f"  (median gap between ADP-consecutive draftable players)")
print(f"  U's noise in the near-optimal population is worth {tight:.0f} points\n")
ratio = tight / swap if swap > 0 else np.nan
print(f"  RATIO: U's noise is {ratio:.1f}x the size of the decision it is being")
print( "  asked to make." if ratio > 1 else "  asked to make -- it can resolve it.")
if ratio > 1:
    print("\n  So E2's result -- U's argmax no better than random at the pick")
    print("  level -- is not a surprising empirical finding. It is what this")
    print("  ratio forces. The simulator cannot see a one-player difference.")
