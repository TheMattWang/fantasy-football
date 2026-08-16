"""B1c -- is U's residual real structure, or irreproducible noise?

B1b found that U's variation around summed starter projections is worth ~22
projected points, against ~3 points for the decision U is asked to make. But
"unexplained by summed projections" has two very different readings:

  STRUCTURE. The simulator knows something the board's sum does not --
  positional balance, bench depth, how injuries interact with roster shape.
  Then U is NOT degenerate, the residual is signal, and the problem is only
  that we are reading it at the wrong scale.

  NOISE. The residual is an artifact of which particular 400 seasons were
  drawn. Then U is degenerate plus noise, and the extra 22 points of apparent
  information is worse than nothing, because argmax will chase it.

These are distinguishable. Evaluate the SAME rosters under independent draws of
the season, and correlate the residuals. Structure replicates across draws.
Noise does not.

Note F2 in the plan: evaluate_roster is deterministic for a fixed sample set,
so repeated calls agree exactly. That is repeatability, not reproducibility --
it says nothing about whether the ordering survives a different draw of the
world, which is the only thing a draft decision depends on.
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
N_ROSTERS = 150
N_OPPONENTS = 11
EVAL_SAMPLES = 400
N_WEEKS = 14
SEEDS = (0, 1, 2)
SHAPE = {"QB": 2, "RB": 5, "WR": 5, "TE": 1, "K": 1, "DEF": 1}
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
CFG = provisional_config()


def draw_roster(pools, rng, greed):
    picked = []
    for pos, count in SHAPE.items():
        pool = pools.get(pos)
        if pool is None or not len(pool):
            continue
        w = np.exp(-greed * np.arange(len(pool)) / max(len(pool), 1))
        w /= w.sum()
        idx = rng.choice(len(pool), size=min(count, len(pool)), replace=False, p=w)
        picked.extend(pool.iloc[idx]["player_name"].tolist())
    return picked


def starter_points(frame, roster):
    sub = frame[frame["player_name"].isin(roster)]
    total, left = 0.0, []
    for pos, count in STARTERS.items():
        b = sub[sub["position"] == pos].sort_values("proj_points", ascending=False)
        total += float(b.head(count)["proj_points"].sum())
        left.append(b.iloc[count:])
    flex = pd.concat(left)
    flex = flex[flex["position"].isin(("RB", "WR", "TE"))]
    return total + (float(flex["proj_points"].max()) if len(flex) else 0.0)


frame = pd.read_csv(BOARD)
draftable = frame[frame["proj_points"].notna()].sort_values("adp_rank")
pools = {p: g.head(60).reset_index(drop=True) for p, g in draftable.groupby("position")}

# Fix the rosters ONCE. Only the draw of the world changes between seeds.
rng = np.random.default_rng(7)
rosters = [draw_roster(pools, rng, float(rng.uniform(6.0, 14.0)))
           for _ in range(N_ROSTERS)]
proj = np.array([starter_points(frame, r) for r in rosters])

resid = {}
for seed in SEEDS:
    samples = build_samples(frame, n_samples=EVAL_SAMPLES, n_weeks=N_WEEKS, seed=seed)
    orng = np.random.default_rng(100 + seed)
    opponent_weekly = np.stack([
        lineup_points(samples,
                      plan_roster(draw_roster(pools, orng, 6.0), samples, CFG),
                      n_samples=EVAL_SAMPLES)
        for _ in range(N_OPPONENTS)
    ])
    u = np.array([
        evaluate_roster(r, samples, CFG, opponent_weekly,
                        n_samples=EVAL_SAMPLES).utility
        for r in rosters
    ])
    design = np.column_stack([np.ones_like(proj), proj])
    beta, *_ = np.linalg.lstsq(design, u, rcond=None)
    resid[seed] = u - design @ beta
    print(f"  seed {seed}: U {u.min():.4f}..{u.max():.4f}  "
          f"resid sd {resid[seed].std():.4f}", flush=True)

print("\nB1c  DOES U'S RESIDUAL REPLICATE ACROSS INDEPENDENT SEASON DRAWS?")
print(f"  {N_ROSTERS} FIXED rosters, {len(SEEDS)} independent draws of the world\n")
print(f"  {'pair':<16}{'corr(resid)':>13}")
pairs = []
for i, a in enumerate(SEEDS):
    for b in SEEDS[i + 1:]:
        c = float(np.corrcoef(resid[a], resid[b])[0, 1])
        pairs.append(c)
        print(f"  seed {a} vs {b:<8}{c:>13.3f}")
mean_c = float(np.mean(pairs))
print(f"  {'mean':<16}{mean_c:>13.3f}")

print(f"\n  A residual that is pure noise correlates at 0 across draws.")
print(f"  One that is real structure correlates near 1.")
if mean_c < 0.3:
    print(f"\n  VERDICT: {mean_c:.2f} -- essentially NOISE. U's variation around")
    print("  summed starter projections does not survive a redraw of the season,")
    print("  so the 22 projected points of apparent extra information is not")
    print("  information at all. argmax over U chases it anyway, which is")
    print("  precisely the optimizer's curse with a measured magnitude.")
elif mean_c > 0.7:
    print(f"\n  VERDICT: {mean_c:.2f} -- real STRUCTURE. The simulator knows")
    print("  something summed projections do not, and the problem is scale,")
    print("  not degeneracy.")
else:
    print(f"\n  VERDICT: {mean_c:.2f} -- mixed. Part structure, part noise.")
    print(f"  Roughly {mean_c:.0%} of the residual variance replicates.")
