"""V8b -- is the usage signal worth anything in POINTS, not just in t?

V8 found opportunity-so-far predicts next week's points after controlling for
points-so-far at t = +6.10, with efficiency-so-far NEGATIVE at t = -5.28 --
textbook confirmation that points = opportunity x efficiency and that only the
first half persists. But the R^2 gain was +0.0051 on a base of 0.3257, and with
12,091 player-weeks even a trivial effect clears t = 2. Significant is not the
same as useful.

So: back-test the actual decision, on real data, with no simulator involved.
The simulator cannot help here anyway -- it models points and has no notion of a
target or a carry, so a usage-based policy is not expressible inside it. That is
a real limitation and this is the way around it.

METHOD. Walk-forward, so nothing is fit on what it is tested on:

  * fit the ranking model on seasons strictly before the test season
  * draw random rosters from that season's fantasy-relevant players
  * each week from week 5, pick starters by each policy and sum what they
    ACTUALLY scored
  * the difference is the value of the signal, in points per week

Two policies:
  points   rank by points per game so far           <- what week.py does today
  usage    rank by the fitted points+opportunity model

Reported per week and per season, because "0.3 points a week" and "4 points a
season" are the same number and only one of them sounds worth having.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mattwang/Documents/fantasy/fantasy-football")

import numpy as np
import pandas as pd

from src.data import nflverse

SEASONS = (2022, 2023, 2024, 2025)
POSITIONS = ("RB", "WR", "TE")
MIN_PRIOR_GAMES = 4
FIRST_WEEK = 5
MAX_WEEK = 17
N_ROSTERS = 300
SHAPE = {"RB": 5, "WR": 5, "TE": 2}
STARTERS = {"RB": 2, "WR": 2, "TE": 1}
N_FLEX = 1
POOL = {"RB": 45, "WR": 60, "TE": 20}      # roughly the fantasy-relevant range


def panel() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        wk = nflverse.load("weekly", [season])
        if "season_type" in wk.columns:
            wk = wk[wk["season_type"] == "REG"]
        wk = wk[wk["position"].isin(POSITIONS)].copy()
        wk = wk[wk["week"] <= MAX_WEEK]
        wk["fantasy_points"] = nflverse.fantasy_points(wk)
        for c in ("targets", "carries"):
            if c not in wk.columns:
                wk[c] = 0.0
        wk["opps"] = wk["targets"].fillna(0) + wk["carries"].fillna(0)
        wk["season"] = season
        frames.append(wk[["season", "week", "player_id", "position",
                          "fantasy_points", "opps"]])
    return pd.concat(frames, ignore_index=True)


def with_history(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["season", "player_id", "week"]).copy()
    g = frame.groupby(["season", "player_id"])
    for col in ("fantasy_points", "opps"):
        frame[f"prior_{col}"] = g[col].cumsum() - frame[col]
    frame["prior_games"] = g.cumcount()
    frame = frame[frame["prior_games"] >= MIN_PRIOR_GAMES].copy()
    frame["ppg_so_far"] = frame["prior_fantasy_points"] / frame["prior_games"]
    frame["opp_so_far"] = frame["prior_opps"] / frame["prior_games"]
    return frame


def fit_model(train: pd.DataFrame):
    """Coefficients on RAW predictors, so they can be applied to any season."""
    X = np.column_stack([
        np.ones(len(train)),
        train["ppg_so_far"].to_numpy(float),
        train["opp_so_far"].to_numpy(float),
    ])
    y = train["fantasy_points"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def pick_starters(block: pd.DataFrame, key: str) -> pd.DataFrame:
    """Slot-legal lineup: dedicated slots first, then one flex from the rest."""
    chosen, leftovers = [], []
    for pos, count in STARTERS.items():
        at_pos = block[block["position"] == pos].sort_values(key, ascending=False)
        chosen.append(at_pos.head(count))
        leftovers.append(at_pos.iloc[count:])
    rest = pd.concat(leftovers)
    chosen.append(rest.sort_values(key, ascending=False).head(N_FLEX))
    return pd.concat(chosen)


data = with_history(panel())
print(f"{len(data):,} player-weeks with >= {MIN_PRIOR_GAMES} prior games\n", flush=True)

rng = np.random.default_rng(5)
rows = []

for season in SEASONS[1:]:                      # 2022 is training only
    train = data[data["season"] < season]
    test = data[data["season"] == season]
    if train.empty or test.empty:
        continue
    beta = fit_model(train)

    # The season's fantasy-relevant pool, by total production. Using season
    # totals to define the POOL is a mild look-ahead about who was relevant, but
    # it is applied identically to both policies, so it cannot favour either.
    totals = (test.groupby(["player_id", "position"])["fantasy_points"]
                  .sum().reset_index())
    pool = {pos: totals[totals["position"] == pos]
                 .nlargest(POOL[pos], "fantasy_points")["player_id"].to_numpy()
            for pos in POSITIONS}

    weeks = sorted(w for w in test["week"].unique() if w >= FIRST_WEEK)
    for _ in range(N_ROSTERS):
        roster = np.concatenate([rng.choice(pool[p], size=SHAPE[p], replace=False)
                                 for p in POSITIONS])
        mine = test[test["player_id"].isin(roster)]
        got_points = got_usage = 0.0
        for week in weeks:
            block = mine[mine["week"] == week]
            if len(block) < sum(STARTERS.values()) + N_FLEX:
                continue                        # too many on bye/injured
            block = block.assign(
                _points=block["ppg_so_far"],
                _usage=(beta[0] + beta[1] * block["ppg_so_far"]
                        + beta[2] * block["opp_so_far"]),
            )
            got_points += pick_starters(block, "_points")["fantasy_points"].sum()
            got_usage += pick_starters(block, "_usage")["fantasy_points"].sum()
        rows.append({"season": season, "points": got_points, "usage": got_usage,
                     "weeks": len(weeks)})

d = pd.DataFrame(rows)
d["gain"] = d["usage"] - d["points"]
d["gain_pw"] = d["gain"] / d["weeks"]

print("V8b  WHAT IS THE USAGE SIGNAL WORTH, IN POINTS?")
print("  walk-forward: fitted only on prior seasons. Random rosters, real")
print("  outcomes, slot-legal lineups, no simulator involved.\n")
print(f"  {'season':<9}{'rosters':>9}{'pts/week':>11}{'se':>8}{'t':>7}{'pts/season':>12}")
for season, g in d.groupby("season"):
    se = g["gain_pw"].std(ddof=1) / np.sqrt(len(g))
    t = g["gain_pw"].mean() / se if se > 0 else np.nan
    print(f"  {season:<9}{len(g):>9}{g['gain_pw'].mean():>+11.3f}{se:>8.3f}"
          f"{t:>+7.2f}{g['gain'].mean():>+12.1f}")

se = d["gain_pw"].std(ddof=1) / np.sqrt(len(d))
print(f"  {'pooled':<9}{len(d):>9}{d['gain_pw'].mean():>+11.3f}{se:>8.3f}"
      f"{d['gain_pw'].mean()/se:>+7.2f}{d['gain'].mean():>+12.1f}")
print(f"\n  rosters where usage won: {100*(d['gain'] > 0).mean():.0f}%")
print("\n  For scale: a weekly team total is ~107 points, and T3 measured the")
print("  whole frozen->reactive upgrade at +1.523 ranks. This is the increment")
print("  ON TOP of already reacting, so it should be much smaller than that.")
