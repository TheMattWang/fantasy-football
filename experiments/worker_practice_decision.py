"""Does the practice-participation split change the LINEUP, or only the number?

`STATUS_MULTIPLIER["Questionable"] = 0.456` was measured as

    P(played | Q) x mean_points(Q)  /  mean_points(no report)
    0.567 x 6.95 / 8.66

The denominator is a healthy player's points *when he plays*. But a healthy
player only takes the field 84% of the time, so that compares an expectation
against a conditional mean. Like for like:

    E[points | Q] / E[points | no report] = 3.945 / 7.277 = 0.542

19% higher, meaning Questionable players have been ranked below where the data
puts them.

None of which matters unless it moves a lineup. V8 is the standing warning: a
signal that predicted next week's points at t=+6.10 changed the actual start/sit
decision by +0.072 points a week and won under half the time. So this scores both
multipliers against what actually happened, in points, with no simulator in the
middle.
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
N_ROSTERS = 60
ROSTER_SIZE = 15
OLD, NEW = 0.456, 0.542   # unused here; see PRACTICE_ON/OFF below

board = pd.read_csv("data/processed/board_2026.csv")
config = provisional_config()
samples = build_samples(board, n_samples=200, n_weeks=14, seed=0)

def season_pool(season):
    """Board players who actually appeared that season.

    Drawing from the 2026 board alone would put players on 2022 rosters who were
    not in the league yet, and every one of them would score a spurious zero --
    which is noise the comparison does not need and bias it cannot afford.
    """
    wk = nflverse.load("weekly", [season])
    seen = set(wk[("player_display_name" if "player_display_name" in wk.columns
                   else "player_name")].map(normalize_name))
    cand = board[~board["is_streamed"]].nlargest(300, "VORP")
    keep = cand[cand["player_name"].map(normalize_name).isin(seen)]
    return keep["player_name"].tolist()


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


def score(lineup, truth):
    """Points the started XI actually scored. A player with no stat line did
    not play, and that is a zero rather than a missing value."""
    total = 0.0
    for slot, names in lineup.items():
        if slot == "BN":
            continue
        for n in names:
            total += float(truth.get(normalize_name(n), 0.0))
    return total


import src.inseason.availability as av

SAVED_PRACTICE = dict(av.PRACTICE_MULTIPLIER)

print(f"{N_ROSTERS} rosters x weeks {WEEKS.start}-{WEEKS.stop - 1} x "
      f"{len(SEASONS)} seasons\n")
print(f"  {'season':<9}{'roster-wks':>11}{'changed':>9}{'mean d':>9}"
      f"{'se':>8}{'t':>7}{'win%':>7}{'per r-wk':>10}")

per_season, all_diffs = [], []
for season in SEASONS:
    rng = np.random.default_rng(0)
    pool = season_pool(season)
    rosters = [list(rng.choice(pool, ROSTER_SIZE, replace=False))
               for _ in range(N_ROSTERS)]

    diffs, changed, wins, losses = [], 0, 0, 0
    for week in WEEKS:
        report = injury_report(season, week)
        if not len(report):
            continue
        truth = actual_points(season, week)
        observed = observed_to_date(season, week - 1)

        lineups = {}
        for tag, table in (("old", {}), ("new", dict(SAVED_PRACTICE))):
            av.PRACTICE_MULTIPLIER.clear(); av.PRACTICE_MULTIPLIER.update(table)
            avail = availability(samples, week=week, board=board, report=report)
            lineups[tag] = [
                best_lineup(r, samples, config, week=week, observed=observed,
                            prior_games=2.0, availability=avail)
                for r in rosters
            ]

        for a, b in zip(lineups["old"], lineups["new"]):
            sa = {n for s, ns in a.items() if s != "BN" for n in ns}
            sb = {n for s, ns in b.items() if s != "BN" for n in ns}
            if sa == sb:
                continue
            changed += 1
            d = score(b, truth) - score(a, truth)
            diffs.append(d); wins += d > 0; losses += d < 0

    total = N_ROSTERS * len(list(WEEKS))
    if diffs:
        d = np.array(diffs); se = d.std(ddof=1) / np.sqrt(len(d))
        per_season.append(d.sum() / total); all_diffs.extend(diffs)
        print(f"  {season:<9}{total:>11}{changed:>9}{d.mean():>+9.2f}{se:>8.2f}"
              f"{d.mean() / se:>+7.2f}{wins / max(wins + losses, 1):>7.0%}"
              f"{d.sum() / total:>+10.3f}")
    else:
        per_season.append(0.0)
        print(f"  {season:<9}{total:>11}{changed:>9}{'--':>9}{'--':>8}"
              f"{'--':>7}{'--':>7}{0.0:>+10.3f}")

av.PRACTICE_MULTIPLIER.clear(); av.PRACTICE_MULTIPLIER.update(SAVED_PRACTICE)

# Season-clustered, because a season is the unit that generalizes. Four is a
# small n and the interval says so -- which is the point of reporting it.
s = np.array(per_season)
se = s.std(ddof=1) / np.sqrt(len(s))
print(f"\n  pooled over {len(all_diffs)} changed lineups: "
      f"{np.mean(all_diffs):+.3f} pts each")
print(f"  SEASON-CLUSTERED per roster-week: {s.mean():+.4f} "
      f"(se {se:.4f}, t = {s.mean() / se:+.2f}, n = {len(s)} seasons)")
print(f"  seasons positive: {int((s > 0).sum())}/{len(s)}")
