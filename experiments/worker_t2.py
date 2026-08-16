"""T2 -- have we been optimizing the small decision?

Three numbers exist and none of them are in the same units, which is why the
comparison has never actually been made:

  * one shortlist swap is worth ~3 projected points        (B1b, points)
  * perfect start/sit is worth 2.17 pts/wk ACROSS ROSTERS  (stage 0a, points/wk)
  * schedule luck is sd 1.45 wins with identical teams     (V7, wins)

Every strategy result in this project is in RANKS. So converting all three into
ranks is not a preliminary to the experiment, it IS the experiment.

METHOD. Hold one league fixed -- same 12 rosters, same season samples, same
schedule, same waiver draws -- and change exactly one thing at a time.

  draft lever    sweep our roster's quality and regress mean rank on summed
                 starter projections. The slope is ranks-per-projected-point;
                 multiply by 3 to get what one pick is worth.
  management     give our team omniscient lineups (perfect start/sit, the
  lever          hindsight ceiling) against ex-ante opponents. Paired, so the
                 difference is the value of in-season management alone.
  luck floor     twelve IDENTICAL rosters. Whatever rank spread remains is the
                 format, not the manager.

PRE-COMMITTED: if the management lever beats the draft lever by more than ~2x
in rank units, this project has been aimed at the wrong decision, and Phase 5
is start/sit and waivers rather than drafting. Say so plainly if it happens.

Both levers are ceilings, not achievable gains -- perfect foresight on one side,
a perfect extra pick on the other. The RATIO is the point, not the levels.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mattwang/Documents/fantasy/fantasy-football")

import numpy as np
import pandas as pd

from src.data.league_config import provisional_config
from src.simulation.distributions import build_samples
from src.simulation.season import (
    lineup_points,
    make_waiver_draws,
    plan_roster,
    round_robin_schedule,
    simulate_league,
)

BOARD = "data/processed/board_2026.csv"
N_TEAMS = 12
N_SAMPLES = 2000
N_WEEKS = 14
N_SWEEP = 250
SHAPE = {"QB": 2, "RB": 5, "WR": 5, "TE": 1, "K": 1, "DEF": 1}
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
SWAP_POINTS = 3.0        # B1b: median gap between ADP-consecutive draftable players
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
samples = build_samples(frame, n_samples=N_SAMPLES, n_weeks=N_WEEKS, seed=0)
draftable = frame[frame["proj_points"].notna()].sort_values("adp_rank")
pools = {p: g.head(60).reset_index(drop=True) for p, g in draftable.groupby("position")}

rng = np.random.default_rng(11)

# Common random numbers: one schedule, one set of waiver draws, reused by every
# arm. Without this the arms differ by luck as much as by the lever.
schedule = round_robin_schedule(N_TEAMS, N_WEEKS)
waivers = make_waiver_draws(N_SAMPLES, N_WEEKS, rng=np.random.default_rng(99))


def weekly(roster, *, omniscient=False):
    return lineup_points(samples, plan_roster(roster, samples, CFG),
                         n_samples=N_SAMPLES, waiver_draws=waivers,
                         omniscient=omniscient)


def rank_of_team0(stack):
    return simulate_league(stack, CFG, schedule=schedule, team_of_interest=0)


# Eleven fixed opponents, drafted competently, playing ex-ante lineups.
opponents = [draw_roster(pools, rng, 9.0) for _ in range(N_TEAMS - 1)]
opp_weekly = np.stack([weekly(r) for r in opponents])
print(f"11 fixed opponents, {N_SAMPLES} seasons, CRN across every arm\n", flush=True)


# --- lever 1: the draft ---------------------------------------------------

rows = []
for _ in range(N_SWEEP):
    roster = draw_roster(pools, rng, float(rng.uniform(6.0, 14.0)))
    res = rank_of_team0(np.concatenate([weekly(roster)[None], opp_weekly]))
    rows.append({"proj": starter_points(frame, roster),
                 "rank": res.mean_rank, "pf": res.points_for.mean()})
d = pd.DataFrame(rows)

x, y = d["proj"].to_numpy(), d["rank"].to_numpy()
design = np.column_stack([np.ones_like(x), x])
beta, *_ = np.linalg.lstsq(design, y, rcond=None)
ranks_per_point = -beta[1]                      # positive = more points, better rank
draft_lever = ranks_per_point * SWAP_POINTS

# Comparing ONE PICK to a whole season of perfect lineups is not a fair
# contest -- it is one decision against 14 weeks x 9 slots of them. The
# scope-matched question is what the WHOLE draft is worth, so report the
# spread across realistic draft outcomes as well.
p50, p90 = np.percentile(d["proj"], [50, 90])
draft_whole = ranks_per_point * (p90 - p50)
draft_span = ranks_per_point * (d["proj"].max() - d["proj"].min())

print("LEVER 1 -- THE DRAFT")
print(f"  {N_SWEEP} rosters swept over draft quality")
print(f"  slope: {ranks_per_point:.4f} ranks gained per projected point")
print(f"  one shortlist swap ({SWAP_POINTS:.0f} pts)      {draft_lever:>7.3f} ranks")
print(f"  median -> 90th pct draft ({p90-p50:>3.0f} pts) {draft_whole:>7.3f} ranks")
print(f"  worst -> best in sweep ({d['proj'].max()-d['proj'].min():>4.0f} pts)  "
      f"{draft_span:>7.3f} ranks\n")


# --- lever 2: in-season management ----------------------------------------

pairs = []
for _ in range(60):
    roster = draw_roster(pools, rng, 9.0)
    ex_ante = rank_of_team0(np.concatenate([weekly(roster)[None], opp_weekly]))
    perfect = rank_of_team0(
        np.concatenate([weekly(roster, omniscient=True)[None], opp_weekly]))
    pairs.append((ex_ante.mean_rank - perfect.mean_rank,
                  perfect.points_for.mean() - ex_ante.points_for.mean()))
gain = np.array([p[0] for p in pairs])
pf_gain = np.array([p[1] for p in pairs])
mgmt_lever = float(gain.mean())
se = float(gain.std(ddof=1) / np.sqrt(len(gain)))

print("LEVER 2 -- IN-SEASON MANAGEMENT (perfect start/sit, hindsight ceiling)")
print(f"  {len(gain)} paired rosters, same league, same seasons")
print(f"  rank gain {mgmt_lever:+.3f} (se {se:.3f})")
print(f"  points gain {pf_gain.mean():+.0f} over the season "
      f"({pf_gain.mean()/N_WEEKS:+.2f} pts/wk)\n")


# --- the luck floor -------------------------------------------------------

# Twelve rosters of the SAME QUALITY, each with its own independent draw --
# not one score vector copied twelve times. Copying makes every H2H game a tie
# (`mine > theirs` false both ways), which zeroes every win and makes rank fall
# out of array order. That is what an sd of exactly 0.00 was reporting.
equals = [draw_roster(pools, rng, 9.0) for _ in range(N_TEAMS)]
identical = rank_of_team0(np.stack([weekly(r) for r in equals]))
print("LUCK FLOOR -- twelve equally-drafted rosters, independent draws")
print(f"  sd of final rank {identical.rank.std():.2f}, "
      f"P(playoffs) {identical.made_playoffs.mean():.3f}\n")


# --- the verdict ----------------------------------------------------------

# The scope-matched ratio: a whole season of perfect lineups against a whole
# draft's worth of realistic quality difference. The per-pick ratio is reported
# too, but it answers a different and much less useful question.
ratio = mgmt_lever / draft_whole if draft_whole > 0 else float("inf")
per_pick_ratio = mgmt_lever / draft_lever if draft_lever > 0 else float("inf")
print("=" * 66)
print(f"  draft, one pick        {draft_lever:>7.3f} ranks")
print(f"  draft, median -> p90   {draft_whole:>7.3f} ranks   <- scope-matched")
print(f"  management, perfect    {mgmt_lever:>7.3f} ranks   (all season)")
print(f"  ratio (scope-matched)  {ratio:>7.1f}x")
print(f"  ratio (per single pick){per_pick_ratio:>7.1f}x  -- not a fair contest")
print("=" * 66)
if ratio > 2:
    print("\n  PRE-COMMITTED VERDICT: in-season management dominates the draft.")
    print("  This project has been aimed at the smaller decision. Phase 5 should")
    print("  be start/sit and waivers, not drafting.")
else:
    print("\n  PRE-COMMITTED VERDICT: the draft is not the small decision.")
    print("  Continuing to work on drafting is justified on these numbers.")
print("\n  Both levers are CEILINGS -- perfect foresight against a perfect extra")
print("  pick -- so the levels are not achievable gains. The ratio is the point.")
