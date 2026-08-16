"""V6 -- does prior-season USAGE predict what our board gets wrong?

PRE-REGISTERED 2026-08-16. The last cheap feature class left after V1-V5 killed
rookie status, draft capital, age, season SoS, playoff SoS, divisional SoS, and
team offensive environment. Only injury history survived, and it implicates our
own proj_games rather than the market.

WHY USAGE IS THE BEST REMAINING CANDIDATE. Production = opportunity x efficiency,
and the two have very different persistence. Targets and carries are stable year
to year; touchdown rate and yards per target are not. So a market that anchors on
last season's POINTS will systematically misprice a player whose points came from
unsustainable efficiency on modest volume, or who had heavy volume and bad luck.
That is the one structural reason a public number could still be underweighted.

THE TEST THAT MATTERS IS THE RESIDUALIZED ONE. Raw usage correlates with the
residual trivially -- good players get more targets AND get drafted early, so a
raw correlation measures nothing but "good players are good". The alpha question
is whether usage ABOVE what a player's draft rank already implies predicts
outperformance. So within each (season, position) we regress prior-season usage
on this season's pos_rank and keep the residual: "more volume than his draft slot
says he should have". Both are reported; only the second one is evidence.

CLUSTERING. The same players recur across 2022-2025 and their projection errors
persist, so the unit of independence is the PLAYER, not the player-season. V4 did
not cluster and its t-statistics were correspondingly optimistic.

PRE-REGISTERED EXPECTATION: low. Target share is exactly what analysts obsess
over, so this is their home turf. Kill any feature at |t| < 2 on the residualized
version. If something does survive, it is the first positive result in the
project and must be reproduced before it is believed.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mattwang/Documents/fantasy/fantasy-football")

import numpy as np
import pandas as pd

from src.data import nflverse
from src.projections.ecr import normalize_name
from src.projections.validate import build_comparison
from src.simulation.distributions import DRAFTABLE_POS_RANK

SEASONS = (2022, 2023, 2024, 2025)
POSITIONS = ("QB", "RB", "WR", "TE")
# Target-based metrics are meaningless for QB; carries are meaningless for WR/TE.
USAGE = {
    "targets_pg":       ("RB", "WR", "TE"),
    "target_share":     ("RB", "WR", "TE"),
    "air_yards_share":  ("WR", "TE"),
    "wopr":             ("RB", "WR", "TE"),
    "carries_pg":       ("RB",),
    "touches_pg":       ("RB", "WR", "TE"),
}


def residual_frame() -> pd.DataFrame:
    rows = []
    for holdout in SEASONS:
        train = [s for s in range(2021, holdout)]
        if not train:
            continue
        f = build_comparison(holdout, train)
        f = f[(f["pred_ecr"] > 1.0) & (f["ppg_available"] > 0)].copy()
        f["season"] = holdout
        f["resid"] = np.log(f["ppg_available"] / f["pred_ecr"])
        limit = f["pos"].map(lambda p: DRAFTABLE_POS_RANK.get(str(p), 0))
        f = f[f["pos_rank"] <= limit]
        rows.append(f[["season", "pos", "pos_rank", "name_key", "resid"]])
    return pd.concat(rows, ignore_index=True)


def usage_prior(season: int) -> pd.DataFrame:
    """Per-game usage in season-1. Played weeks only.

    Per GAME, not per season: a player who missed six weeks still had whatever
    role he had when active, and dividing by 17 would confound usage with
    availability -- which V4 already showed is a separate, real signal.
    """
    wk = nflverse.load("weekly", [season - 1])
    if "season_type" in wk.columns:
        wk = wk[wk["season_type"] == "REG"]
    wk = wk[wk["position"].isin(POSITIONS)].copy()
    name_c = "player_display_name" if "player_display_name" in wk else "player_name"
    wk["name_key"] = wk[name_c].map(normalize_name)

    for c in ("targets", "carries", "target_share", "air_yards_share", "wopr"):
        if c not in wk.columns:
            wk[c] = np.nan
    wk["played"] = (wk["targets"].fillna(0) + wk["carries"].fillna(0)) > 0
    wk = wk[wk["played"]]

    g = wk.groupby("name_key")
    out = pd.DataFrame({
        "games": g["week"].nunique(),
        "targets_pg": g["targets"].mean(),
        "carries_pg": g["carries"].mean(),
        "target_share": g["target_share"].mean(),
        "air_yards_share": g["air_yards_share"].mean(),
        "wopr": g["wopr"].mean(),
    }).reset_index()
    out["touches_pg"] = out["targets_pg"].fillna(0) + out["carries_pg"].fillna(0)
    out["season"] = season
    # Four games is enough to establish a role; below that the mean is noise.
    return out[out["games"] >= 4]


def residualize(frame: pd.DataFrame, feature: str) -> pd.Series:
    """Usage minus what this season's draft rank already implies.

    Fit within (season, position) so the comparison is against a player's own
    positional cohort in his own year.
    """
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, idx in frame.groupby(["season", "pos"]).groups.items():
        g = frame.loc[idx]
        ok = g[feature].notna() & g["pos_rank"].notna()
        if ok.sum() < 10:
            continue
        x = g.loc[ok, "pos_rank"].to_numpy(dtype=float)
        y = g.loc[ok, feature].to_numpy(dtype=float)
        # Rank -> usage is strongly convex; log(rank) fits it far better than
        # rank, and a bad fit here leaks real rank signal into the "residual".
        design = np.column_stack([np.ones_like(x), np.log(x + 1.0)])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        out.loc[g.index[ok]] = y - design @ beta
    return out


def clustered(frame: pd.DataFrame, feature: str):
    """Correlation with an SE clustered by player, via a cluster bootstrap.

    The same players recur across four seasons and their errors persist, so
    resampling PLAYERS (not rows) is what keeps the SE honest.
    """
    sub = frame[frame[feature].notna() & frame["resid"].notna()]
    if len(sub) < 40:
        return None
    r = float(np.corrcoef(sub[feature], sub["resid"])[0, 1])

    players = sub["name_key"].unique()
    rng = np.random.default_rng(0)
    by_player = {p: g for p, g in sub.groupby("name_key")}
    draws = []
    for _ in range(400):
        pick = rng.choice(players, size=len(players), replace=True)
        boot = pd.concat([by_player[p] for p in pick], ignore_index=True)
        if boot[feature].std() > 0 and boot["resid"].std() > 0:
            draws.append(np.corrcoef(boot[feature], boot["resid"])[0, 1])
    se = float(np.std(draws, ddof=1)) if len(draws) > 10 else np.nan
    t = r / se if se and se > 0 else np.nan
    return r, se, t, len(sub), len(players)


print("building residuals...", flush=True)
resid = residual_frame()
print(f"  {len(resid)} draftable player-seasons", flush=True)

print("building prior-season usage...", flush=True)
usage = pd.concat([usage_prior(s) for s in SEASONS], ignore_index=True)
frame = resid.merge(usage, on=["season", "name_key"], how="inner")
print(f"  matched {len(frame)} of {len(resid)} "
      f"({100*len(frame)/max(len(resid),1):.0f}%) -- unmatched are mostly rookies\n",
      flush=True)

print("V6  DOES PRIOR-SEASON USAGE PREDICT OUR RESIDUAL?")
print("  raw      = correlation with usage. Confounded: good players get volume")
print("             AND get drafted early, so this measures nothing on its own.")
print("  resid'zd = usage minus what the player's draft rank already implies.")
print("             THIS is the alpha test. SEs bootstrapped over players.\n")
print(f"  {'feature':<18}{'positions':<14}{'raw':>8}{'resid':>9}{'se':>7}{'t':>7}{'n':>6}{'plyrs':>7}")

for feature, positions in USAGE.items():
    sub = frame[frame["pos"].isin(positions)].copy()
    if sub[feature].notna().sum() < 40:
        print(f"  {feature:<18}{'/'.join(positions):<14}  insufficient data")
        continue
    raw = clustered(sub, feature)
    sub["_r"] = residualize(sub, feature)
    adj = clustered(sub, "_r")
    if raw is None or adj is None:
        print(f"  {feature:<18}{'/'.join(positions):<14}  not estimable")
        continue
    flag = "  <-- SURVIVES" if abs(adj[2]) >= 2 else ""
    print(f"  {feature:<18}{'/'.join(positions):<14}{raw[0]:>+8.3f}"
          f"{adj[0]:>+9.3f}{adj[1]:>7.3f}{adj[2]:>+7.2f}{adj[3]:>6}{adj[4]:>7}{flag}")

print("\n  by position, residualized t")
print(f"  {'feature':<18}" + "".join(f"{p:>9}" for p in POSITIONS))
for feature, positions in USAGE.items():
    cells = []
    for p in POSITIONS:
        if p not in positions:
            cells.append(f"{'-':>9}")
            continue
        sub = frame[frame["pos"] == p].copy()
        if sub[feature].notna().sum() < 40:
            cells.append(f"{'--':>9}")
            continue
        sub["_r"] = residualize(sub, feature)
        res = clustered(sub, "_r")
        cells.append(f"{res[2]:>+9.2f}" if res else f"{'--':>9}")
    print(f"  {feature:<18}" + "".join(cells))

print("\n  PRE-REGISTERED: expected low -- target share is the analysts' home")
print("  turf. Kill at |t| < 2 on the RESIDUALIZED column. Anything that")
print("  survives is the project's first positive and must be reproduced.")
