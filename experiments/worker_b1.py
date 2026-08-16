"""B1 -- is U just a monotone transform of the board?

THE GATE CHECK for everything downstream. Two facts about the simulator make
this plausible:

  * the lineup is FROZEN all season -- `decision_score` is the preseason
    projection and is never updated, so there is no in-season decision for the
    simulation to model;
  * the playoffs are not simulated at all -- the champion is drawn with
    probability proportional to 1/seed.

If U is a near-deterministic function of summed projected points, then the
season simulator is an expensive noise generator: it adds variance without
adding information. Consequences, all of which we have already observed:

  * "45x simulation compute buys 6% precision" is explained;
  * P(title) carrying no information beyond rank is explained;
  * the optimizer's-curse result becomes ARITHMETIC rather than an empirical
    finding, because argmax over U reduces to argmax over projected points on
    an ADP-adjacent shortlist -- which is exactly a max over noisy estimates
    of near-identical quantities;
  * the correlation and in-season workstreams are untestable until the
    simulator is rebuilt, since a frozen lineup cannot express either.

DESIGN. Hold the opponent field FIXED and vary only our roster, so the only
thing moving is what we are trying to measure. Rosters are drawn across a wide
quality spectrum by varying how greedily they take the board, which is what
gives the regression something to fit.

WHAT TO REPORT. The plan's criterion is R^2 > 0.9, but a linear R^2 is the
WRONG statistic for "monotone transform" -- U is a probability, bounded and
saturating, so it can be a perfect monotone function of the board and still
show mediocre linear R^2. Spearman is the right test. Report all three:
linear R^2, Spearman, and the R^2 of a rank-based (monotone) fit.
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mattwang/Documents/fantasy/fantasy-football")

import numpy as np
import pandas as pd

from src.data.league_config import provisional_config
from src.simulation.distributions import build_samples
from src.simulation.season import evaluate_roster, lineup_points, plan_roster

BOARD = "data/processed/board_2026.csv"
N_ROSTERS = 300
N_OPPONENTS = 11
EVAL_SAMPLES = 400
N_WEEKS = 14
SHAPE = {"QB": 2, "RB": 5, "WR": 5, "TE": 1, "K": 1, "DEF": 1}   # 15 picks
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}                  # + 1 FLEX

CFG = provisional_config()


def draw_roster(pools, rng, greed):
    """One roster. `greed` controls how hard it takes the top of the board.

    High greed -> the best available every time. Low greed -> nearly uniform
    over the draftable pool. Sweeping it is what produces rosters spanning a
    wide enough quality range for the regression to have signal.
    """
    picked = []
    for pos, count in SHAPE.items():
        pool = pools.get(pos)
        if pool is None or not len(pool):
            continue
        rank = np.arange(len(pool))
        weight = np.exp(-greed * rank / max(len(pool), 1))
        weight /= weight.sum()
        take = min(count, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False, p=weight)
        picked.extend(pool.iloc[idx]["player_name"].tolist())
    return picked


def projected_points(frame, roster):
    """Two summaries of the board's own view of a roster: everyone, and the
    players who would actually start. Only starters score, so the second is
    the fairer comparator."""
    sub = frame[frame["player_name"].isin(roster)]
    total = float(sub["proj_points"].sum())

    start = 0.0
    leftovers = []
    for pos, count in STARTERS.items():
        block = sub[sub["position"] == pos].sort_values("proj_points",
                                                        ascending=False)
        start += float(block.head(count)["proj_points"].sum())
        leftovers.append(block.iloc[count:])
    flex = pd.concat(leftovers) if leftovers else sub.iloc[:0]
    flex = flex[flex["position"].isin(("RB", "WR", "TE"))]
    if len(flex):
        start += float(flex["proj_points"].max())
    return total, start


t0 = time.time()
frame = pd.read_csv(BOARD)
samples = build_samples(frame, n_samples=EVAL_SAMPLES, n_weeks=N_WEEKS, seed=0)
print(f"board {len(frame)} rows | {samples.summary()}", flush=True)

draftable = frame[frame["proj_points"].notna()].sort_values("adp_rank")
pools = {pos: g.head(60).reset_index(drop=True)
         for pos, g in draftable.groupby("position")}

rng = np.random.default_rng(7)

# A fixed opponent field. U must move because OUR roster moved, not because
# the field did.
print(f"building {N_OPPONENTS} fixed opponents...", flush=True)
opp = []
for _ in range(N_OPPONENTS):
    roster = draw_roster(pools, rng, greed=6.0)
    plan = plan_roster(roster, samples, CFG)
    opp.append(lineup_points(samples, plan, n_samples=EVAL_SAMPLES))
opponent_weekly = np.stack(opp)

print(f"evaluating {N_ROSTERS} rosters across the quality spectrum...", flush=True)
rows = []
for i in range(N_ROSTERS):
    greed = float(rng.uniform(0.5, 14.0))       # spans "random" to "best available"
    roster = draw_roster(pools, rng, greed)
    result = evaluate_roster(roster, samples, CFG, opponent_weekly,
                             n_samples=EVAL_SAMPLES)
    total, start = projected_points(frame, roster)
    rows.append({
        "U": result.utility,
        "p_playoffs": result.p_playoffs,
        "p_title": result.p_title,
        "mean_rank": result.mean_rank,
        "proj_total": total,
        "proj_starters": start,
    })
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_ROSTERS}  ({time.time()-t0:.0f}s)", flush=True)

d = pd.DataFrame(rows)


def fits(y, x):
    """Linear R^2, Spearman, and the R^2 of a monotone (rank-based) fit."""
    spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot else 0.0
    # Monotone fit: regress on the RANK of x. Captures any increasing
    # transform, which is what "monotone transform of the board" means.
    xr = pd.Series(x).rank().to_numpy()
    dm = np.column_stack([np.ones_like(xr), xr])
    bm, *_ = np.linalg.lstsq(dm, y, rcond=None)
    r2m = 1 - float(((y - dm @ bm) ** 2).sum()) / ss_tot if ss_tot else 0.0
    return r2, spearman, r2m


print(f"\nB1  IS U A MONOTONE TRANSFORM OF THE BOARD?   ({time.time()-t0:.0f}s)")
print(f"  {N_ROSTERS} rosters, fixed opponent field, {EVAL_SAMPLES} season samples")
print(f"  U spans {d.U.min():.4f} to {d.U.max():.4f}\n")
print(f"  {'target':<14}{'predictor':<16}{'linR2':>8}{'spearman':>10}{'monoR2':>9}")
for target in ("U", "p_playoffs", "p_title", "mean_rank"):
    for pred in ("proj_total", "proj_starters"):
        r2, sp, r2m = fits(d[target].to_numpy(), d[pred].to_numpy())
        print(f"  {target:<14}{pred:<16}{r2:>8.3f}{sp:>10.3f}{r2m:>9.3f}")

best = max(fits(d["U"].to_numpy(), d[p].to_numpy())[2]
           for p in ("proj_total", "proj_starters"))
print(f"\n  VERDICT: best monotone R^2 of U on the board = {best:.3f}")
if best > 0.9:
    print("  DEGENERATE. The season simulator adds variance, not information.")
    print("  argmax over U reduces to argmax over projected points, so the")
    print("  optimizer's curse here is arithmetic, not an empirical finding.")
    print("  Correlation and in-season work are untestable until it is rebuilt.")
else:
    print("  NOT degenerate. U carries information beyond summed projections,")
    print("  so the simulator is doing real work and the earlier results stand")
    print("  as empirical findings rather than as arithmetic.")
