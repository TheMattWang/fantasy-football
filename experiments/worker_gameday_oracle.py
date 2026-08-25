"""What is a live game-day inactive feed WORTH? An upper bound, before building one.

V9 killed the attempt to sharpen Friday's information. This asks a different
question: not "can we re-weight the Friday report better" but "what if we simply
knew, on Sunday morning, who is playing".

nflverse carries the Friday report; final inactives land ~90 minutes before
kickoff and are not in it. A live feed (Sleeper's public API needs no auth) could
supply them. Whether that is worth writing is decided by the CEILING: give the
lineup perfect game-day knowledge and see what it earns. If the oracle is small,
no real feed -- necessarily worse than the oracle -- can be worth the code.

This is the same discipline as the project's ceiling experiment, which is what
stopped a whole phase of draft work: bound the prize before chasing it.

The oracle is honest about being an oracle. It uses realised
`weekly_rosters.status` and post-hoc stat lines, so it is unattainable by
construction and is quoted as a bound, never as a policy.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mattwang/Documents/fantasy/fantasy-football")

import numpy as np
import pandas as pd

from src.data import nflverse
from src.data.league_config import provisional_config
from src.inseason.availability import availability, injury_report
from src.inseason.waivers import best_lineup, observed_to_date
from src.projections.ecr import normalize_name
from src.simulation.distributions import build_samples

SEASONS = (2022, 2023, 2024, 2025)
WEEKS = range(3, 15)
N_ROSTERS, ROSTER_SIZE = 60, 15

board = pd.read_csv("data/processed/board_2026.csv")
config = provisional_config()
samples = build_samples(board, n_samples=200, n_weeks=14, seed=0)


def actual_points(season, week):
    wk = nflverse.load("weekly", [season])
    if "season_type" in wk.columns:
        wk = wk[wk["season_type"] == "REG"]
    wk = wk[wk["week"] == week].copy()
    col = ("player_display_name" if "player_display_name" in wk.columns
           else "player_name")
    wk["name_key"] = wk[col].map(normalize_name)
    wk["fp"] = nflverse.fantasy_points(wk)
    return wk.groupby("name_key")["fp"].max()


def season_pool(season):
    wk = nflverse.load("weekly", [season])
    seen = set(wk[("player_display_name" if "player_display_name" in wk.columns
                   else "player_name")].map(normalize_name))
    cand = board[~board["is_streamed"]].nlargest(300, "VORP")
    return cand[cand["player_name"].map(normalize_name).isin(seen)]["player_name"].tolist()


def score(lineup, truth):
    return sum(float(truth.get(normalize_name(n), 0.0))
               for s, ns in lineup.items() if s != "BN" for n in ns)


per_season, per_season_narrow = [], []
for season in SEASONS:
    rng = np.random.default_rng(0)
    pool = season_pool(season)
    rosters = [list(rng.choice(pool, ROSTER_SIZE, replace=False))
               for _ in range(N_ROSTERS)]

    gained, narrow_gain, weeks_used = [], [], 0
    for week in WEEKS:
        report = injury_report(season, week)
        if not len(report):
            continue
        truth = actual_points(season, week)
        observed = observed_to_date(season, week - 1)
        weeks_used += 1

        friday = availability(samples, week=week, board=board, report=report)

        # Two oracles, because the difference between them is the whole point.
        #
        # WIDE: zero anybody who did not record a stat line. This is NOT what a
        # game-day feed buys -- it also catches players on IR, healthy scratches
        # and anyone who never appeared on a report at all. Quoted only to show
        # how badly an over-broad oracle overstates the prize.
        #
        # NARROW: resolve ONLY players carrying a Friday designation. That is
        # exactly the population an inactive list covers, and it is the honest
        # ceiling for the thing we might actually build.
        names = samples.players["player_name"].astype(str).map(normalize_name)
        played = names.map(lambda k: k in truth.index).to_numpy()
        designated = friday["reason"].isin(
            ["QUESTIONABLE", "DOUBTFUL", "OUT"]).to_numpy()
        not_bye = (friday["reason"] != "BYE").to_numpy()

        wide = friday.copy()
        wide.loc[(~played) & not_bye, "multiplier"] = 0.0
        wide.loc[played & designated, "multiplier"] = 1.0

        narrow = friday.copy()
        narrow.loc[designated & ~played, "multiplier"] = 0.0
        narrow.loc[designated & played, "multiplier"] = 1.0

        for r in rosters:
            base = best_lineup(r, samples, config, week=week, observed=observed,
                               prior_games=2.0, availability=friday)
            b_wide = best_lineup(r, samples, config, week=week, observed=observed,
                                 prior_games=2.0, availability=wide)
            b_narrow = best_lineup(r, samples, config, week=week, observed=observed,
                                   prior_games=2.0, availability=narrow)
            s0 = score(base, truth)
            gained.append(score(b_wide, truth) - s0)
            narrow_gain.append(score(b_narrow, truth) - s0)

    g, n = np.array(gained), np.array(narrow_gain)
    per_season.append(g.mean()); per_season_narrow.append(n.mean())
    print(f"  {season}  {len(g):>5} roster-weeks   wide {g.mean():+7.3f}   "
          f"NARROW {n.mean():+7.3f} pts/week (se {n.std(ddof=1) / np.sqrt(len(n)):.3f})")

w = np.array(per_season); nn = np.array(per_season_narrow)
print(f"\n  WIDE oracle  (knows everyone who sat): {w.mean():+.3f} pts/roster-week "
      f"= {w.mean() * 14:+.1f} a season")
print(f"     -- an OVERSTATEMENT. It also resolves players on IR, healthy scratches")
print(f"        and anyone never on a report. No feed gives you that.")
print(f"\n  NARROW oracle (resolves only players carrying a Friday designation --")
print(f"  exactly what an inactive list covers): {nn.mean():+.3f} pts/roster-week "
      f"(se {nn.std(ddof=1) / np.sqrt(len(nn)):.3f})")
print(f"     over a 14-week season, per roster: {nn.mean() * 14:+.2f} points")
print(f"\n  The narrow figure is the one to judge, and a real feed is strictly")
print(f"  worse than it: Sleeper is not perfect and not every inactive is known")
print(f"  before lineups lock.")
