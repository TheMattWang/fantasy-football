"""T3 -- how much of the in-season ceiling does a REALISTIC manager capture?

T2 measured perfect start/sit at +4.457 ranks and flagged the obvious objection
against its own result: the baseline was a FROZEN lineup, not a real manager. A
real manager benches busts and starts breakouts, so the gap between "what people
actually do" and "perfect" is smaller than 4.457 by an unmeasured amount. This
measures it.

The reactive policy ranks by a running blend of the preseason prior and points
so far, weighting the prior as `k` games. k is how stubborn the manager is:
large k barely reacts, small k chases noise. Sweeping k is not tuning -- it is
the question, because the best k IS the description of good management.

FOUR ARMS, and the third is the one that matters.

  A  us reactive vs FROZEN opponents      -- overstates badly. In a real league
                                             the other eleven managers also set
                                             lineups; beating people who never
                                             touch their roster is not an edge.
  B  everyone reactive                    -- the realistic league. If the gain
                                             survives here it is a real edge; if
                                             it collapses, in-season management
                                             is table stakes rather than alpha.
  C  us omniscient vs reactive opponents  -- THE NUMBER. What is still on the
                                             table above a competent manager.
                                             This is the honest version of
                                             T2's 4.457.

If C is small, then most of T2's ceiling was an artifact of comparing perfection
to a strawman, and the in-season lever is much less attractive than it looked.
Say so if that is what comes back.
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
N_PAIRS = 40
K_SWEEP = (1.0, 2.0, 4.0, 8.0, 16.0)
SHAPE = {"QB": 2, "RB": 5, "WR": 5, "TE": 1, "K": 1, "DEF": 1}
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


frame = pd.read_csv(BOARD)
samples = build_samples(frame, n_samples=N_SAMPLES, n_weeks=N_WEEKS, seed=0)
draftable = frame[frame["proj_points"].notna()].sort_values("adp_rank")
pools = {p: g.head(60).reset_index(drop=True) for p, g in draftable.groupby("position")}

rng = np.random.default_rng(11)
schedule = round_robin_schedule(N_TEAMS, N_WEEKS)
waivers = make_waiver_draws(N_SAMPLES, N_WEEKS, rng=np.random.default_rng(99))


def weekly(roster, **kwargs):
    return lineup_points(samples, plan_roster(roster, samples, CFG),
                         n_samples=N_SAMPLES, waiver_draws=waivers, **kwargs)


def rank(mine, opps):
    return simulate_league(np.concatenate([mine[None], opps]), CFG,
                           schedule=schedule, team_of_interest=0).mean_rank


opponents = [draw_roster(pools, rng, 9.0) for _ in range(N_TEAMS - 1)]
opp_frozen = np.stack([weekly(r) for r in opponents])
print(f"11 fixed opponents, {N_SAMPLES} seasons, CRN across every arm", flush=True)


def report(label, gains):
    g = np.asarray(gains)
    se = g.std(ddof=1) / np.sqrt(len(g))
    print(f"  {label:<34}{g.mean():>+8.3f}  (se {se:.3f})")
    return float(g.mean())


# --- A: reactive against managers who never touch their roster ------------

print("\nARM A -- us reactive, opponents FROZEN (overstates; see docstring)")
best_k, best_gain = None, -np.inf
for k in K_SWEEP:
    gains = []
    for _ in range(N_PAIRS):
        r = draw_roster(pools, rng, 9.0)
        gains.append(rank(weekly(r), opp_frozen)
                     - rank(weekly(r, reactive_prior_games=k), opp_frozen))
    g = report(f"k={k:<5.0f} prior games", gains)
    if g > best_gain:
        best_k, best_gain = k, g

print(f"\n  best stubbornness: k={best_k:.0f} prior games")

# --- B: everyone manages --------------------------------------------------

opp_reactive = np.stack([weekly(r, reactive_prior_games=best_k) for r in opponents])

print(f"\nARM B -- everyone reactive at k={best_k:.0f} (the realistic league)")
gains_b = []
for _ in range(N_PAIRS):
    r = draw_roster(pools, rng, 9.0)
    gains_b.append(rank(weekly(r), opp_reactive)
                   - rank(weekly(r, reactive_prior_games=best_k), opp_reactive))
b = report("managing vs not, all manage", gains_b)

# --- C: what is left above a competent manager ----------------------------

print("\nARM C -- us omniscient vs reactive opponents  <- the honest ceiling")
gains_c, gains_c_frozen = [], []
for _ in range(N_PAIRS):
    r = draw_roster(pools, rng, 9.0)
    react = rank(weekly(r, reactive_prior_games=best_k), opp_reactive)
    gains_c.append(react - rank(weekly(r, omniscient=True), opp_reactive))
    gains_c_frozen.append(rank(weekly(r), opp_reactive)
                          - rank(weekly(r, omniscient=True), opp_reactive))
c = report("perfect vs a good manager", gains_c)
c_frozen = report("perfect vs a frozen lineup", gains_c_frozen)

print("\n" + "=" * 66)
print(f"  T2 reported (perfect vs frozen, frozen opponents)     4.457")
print(f"  perfect vs frozen, against managing opponents        {c_frozen:>6.3f}")
print(f"  perfect vs a GOOD manager                            {c:>6.3f}   <- real headroom")
print(f"  a good manager's own edge, when others manage too    {b:>6.3f}")
captured = 1 - c / c_frozen if c_frozen > 0 else float("nan")
print(f"  fraction of the ceiling a realistic policy captures  {captured:>6.1%}")
print("=" * 66)
