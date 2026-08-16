"""V8 -- is USAGE a better in-season signal than POINTS?

V6 tested usage as a DRAFT signal and killed it: target_share reached t=+1.78,
short of the pre-registered bar, on 614 player-seasons. But that test asked a
hard question -- "did ~150 analysts collectively misprice this in August?" --
against the market's home turf.

The in-season question is different and much easier, because the opponent is not
the market. It is the naive baseline our own reactive policy currently uses:

    week.py ranks by POINTS PER GAME SO FAR.

So: does opportunity so far predict next week's points AFTER controlling for
points so far? If yes, week.py is using the wrong statistic and the fix is free.

THE MECHANISM, which is why the prior is much better here than in V6:

    points = opportunity x efficiency

Opportunity (targets, carries) is a coaching decision and persists week to week.
Efficiency (yards per target, touchdown rate) is mostly noise and mean-reverts.
A player whose points came from three touchdowns on four targets will regress; a
player seeing twelve targets a game with nothing to show for it will not. A
points-average cannot tell those apart. A usage-average can.

POWER. n is player-WEEKS here, not player-seasons -- roughly 10^4 rather than
614 -- so this can resolve effects far smaller than V6 could. SEs are clustered
by player, since the same players recur across every week of four seasons.

PRE-REGISTERED: expect opportunity to add predictive power beyond points, and
expect efficiency-so-far to add little or nothing (it is the mean-reverting
half). Kill the claim if the partial t on opportunity is under 2.
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
MIN_PRIOR_GAMES = 4       # below this, "so far" is noise
MAX_WEEK = 17


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
    """Attach each player-week's own past, and nothing else.

    Everything is shifted by one week. Including the current week would make the
    predictor contain its own target, which is the same off-by-one that produced
    the FLEX bug and would manufacture an enormous fake result here.
    """
    frame = frame.sort_values(["season", "player_id", "week"]).copy()
    g = frame.groupby(["season", "player_id"])

    for col in ("fantasy_points", "opps"):
        csum = g[col].cumsum() - frame[col]          # strictly prior weeks
        frame[f"prior_{col}"] = csum
    frame["prior_games"] = g.cumcount()

    ok = frame["prior_games"] >= MIN_PRIOR_GAMES
    frame = frame[ok].copy()

    frame["ppg_so_far"] = frame["prior_fantasy_points"] / frame["prior_games"]
    frame["opp_so_far"] = frame["prior_opps"] / frame["prior_games"]
    # Points per opportunity: the efficiency half, which theory says is the
    # part that does NOT persist.
    frame["eff_so_far"] = np.where(
        frame["prior_opps"] > 0,
        frame["prior_fantasy_points"] / frame["prior_opps"].clip(lower=1e-9),
        np.nan,
    )
    return frame.dropna(subset=["ppg_so_far", "opp_so_far", "eff_so_far"])


def standardize(frame: pd.DataFrame, cols) -> pd.DataFrame:
    out = frame.copy()
    for c in cols:
        s = out[c]
        out[c] = (s - s.mean()) / (s.std(ddof=0) or 1.0)
    return out


def fit(frame: pd.DataFrame, predictors, n_boot: int = 300):
    """OLS with a player-clustered bootstrap. Returns coefficients, t, R^2."""
    y = frame["fantasy_points"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(frame))]
                        + [frame[p].to_numpy(dtype=float) for p in predictors])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()

    players = frame["player_id"].to_numpy()
    uniq = np.unique(players)
    index = {p: np.flatnonzero(players == p) for p in uniq}
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[p] for p in pick])
        Xi, yi = X[rows], y[rows]
        b, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
        draws.append(b)
    se = np.std(np.array(draws), axis=0, ddof=1)
    t = beta / np.where(se > 0, se, np.nan)
    return beta, se, t, r2


print("loading player-weeks...", flush=True)
data = with_history(panel())
data = standardize(data, ["ppg_so_far", "opp_so_far", "eff_so_far"])
print(f"  {len(data):,} player-weeks, {data['player_id'].nunique():,} players, "
      f"{len(SEASONS)} seasons\n", flush=True)

print("V8  DOES USAGE BEAT POINTS AS AN IN-SEASON SIGNAL?")
print("  target = this week's fantasy points. Every predictor uses strictly")
print("  prior weeks. Predictors standardized, so coefficients are directly")
print("  comparable. SEs bootstrapped over players.\n")

models = [
    ("points only (what week.py uses)", ["ppg_so_far"]),
    ("opportunity only",                ["opp_so_far"]),
    ("points + opportunity",            ["ppg_so_far", "opp_so_far"]),
    ("points + opportunity + efficiency", ["ppg_so_far", "opp_so_far", "eff_so_far"]),
]

base_r2 = None
for label, preds in models:
    beta, se, t, r2 = fit(data, preds)
    if base_r2 is None:
        base_r2 = r2
    print(f"  {label}")
    print(f"    R^2 = {r2:.4f}   (vs points-only {r2 - base_r2:+.4f})")
    for name, b, s, tv in zip(preds, beta[1:], se[1:], t[1:]):
        star = "  <-- adds signal" if abs(tv) >= 2 and name != "ppg_so_far" else ""
        print(f"      {name:<14}{b:>+8.3f}  se {s:.3f}  t {tv:>+6.2f}{star}")
    print()

print("  BY POSITION -- partial t on opportunity, controlling for points")
print(f"  {'pos':<6}{'n':>8}{'beta':>9}{'t':>8}")
for pos in POSITIONS:
    sub = data[data["position"] == pos]
    if len(sub) < 500:
        continue
    beta, se, t, _ = fit(sub, ["ppg_so_far", "opp_so_far"], n_boot=200)
    print(f"  {pos:<6}{len(sub):>8,}{beta[2]:>+9.3f}{t[2]:>+8.2f}")

print("\n  PRE-REGISTERED: expect opportunity to add beyond points, and")
print("  efficiency to add little (it is the mean-reverting half). Kill the")
print("  claim if the partial t on opportunity is under 2.")
