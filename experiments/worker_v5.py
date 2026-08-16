"""V5 -- does WHO A PLAYER FACES predict what our board gets wrong?

PRE-REGISTERED before looking (2026-08-16). This is Matt's hypothesis #3, the
one V4 was supposed to test and never did: `team_ppg_prior` failed because I
guessed a column name and `team_weekly` has no points column. It does have
`opponent_team`, and so does `weekly`, so the schedule is fully recoverable.

WHY THIS LANE IS STILL OPEN. V1 found 97% of the residual is idiosyncratic --
not explained by position x rank tier. Schedule effects are PLAYER-SPECIFIC
(every player has a different schedule), so they land in that 97%, not in the
3% systematic bucket. V1 does not close this. V4 tested four player-attribute
features and got R^2 = 0.043; schedule was not among them.

WHAT A POSITIVE RESULT WOULD MEAN. `proj_ppg = curve.ppg_at(position, pos_rank)`
has no schedule input at all, and within a position our ordering IS the experts'
by construction. So the residual here is the residual against the MARKET. A
significant coefficient therefore means the market underweights schedule -- real
alpha, not just a calibration fix. That is a strong claim and it needs a strong
test.

EX-ANTE DISCIPLINE. Every feature uses PRIOR-season defensive quality combined
with THIS season's published schedule. The NFL schedule is released in May and
ADP is set in July/August, so the schedule itself is legitimately known to the
market. Using this season's defensive results would be hindsight and would
manufacture a finding.

CLUSTERING. Players on the same team share a schedule, so the unit of
independence is the TEAM-SEASON (~32/yr), not the player-season (~150/yr).
V4 did not cluster. Unclustered SEs here would overstate t by roughly the
square root of the players-per-team ratio, which is where a spurious "finding"
would come from. All SEs below are clustered by (season, team).

PRE-REGISTERED EXPECTATION, written before running: near zero for `sos_all`.
The schedule is public months before ADP, and defensive quality regresses hard
year over year. Whatever small probability mass I have is on `sos_playoff`
(weeks 15-17 are less attended to and decide the title) and `team_off_prior`
(offense persists far better than defense). Kill criterion: |t| < 2 on a
feature closes that feature.
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
REG_WEEKS = 14          # fantasy regular season
PLAYOFF_WEEKS = (15, 16, 17)
FEATURES = ("sos_all", "sos_playoff", "div_repeat", "team_off_prior")


# --- residual: what our board got wrong -----------------------------------

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


# --- the schedule and who is on it ----------------------------------------

def weekly_with_points(season: int) -> pd.DataFrame:
    wk = nflverse.load("weekly", [season])
    if "season_type" in wk.columns:
        wk = wk[wk["season_type"] == "REG"]
    wk = wk.copy()
    wk["fantasy_points"] = nflverse.fantasy_points(wk)
    name_c = "player_display_name" if "player_display_name" in wk else "player_name"
    wk["name_key"] = wk[name_c].map(normalize_name)
    return wk


def defense_allowed(season: int) -> pd.DataFrame:
    """Fantasy points a defense allowed per game, by position, z-scored.

    Positive z = generous defense = good news for the offense facing it.
    Z-scored within (season, position) so QB and RB scales are comparable and
    the coefficient is readable.
    """
    wk = weekly_with_points(season)
    wk = wk[wk["position"].isin(POSITIONS)]
    games = (wk.groupby("opponent_team")["week"].nunique().rename("games"))
    tot = (wk.groupby(["opponent_team", "position"])["fantasy_points"]
             .sum().rename("allowed").reset_index()
             .merge(games, on="opponent_team", how="left"))
    tot["allowed_pg"] = tot["allowed"] / tot["games"].clip(lower=1)
    tot["z"] = tot.groupby("position")["allowed_pg"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0)
    tot["season"] = season
    return tot[["season", "opponent_team", "position", "z"]]


def schedule(season: int) -> pd.DataFrame:
    """(team, week) -> opponent, from the season's own game records.

    The schedule is announced in May, well before ADP settles, so using it is
    not hindsight. Only the OPPONENT's prior-year quality is joined on.
    """
    tw = nflverse.load("team_weekly", [season])
    if "season_type" in tw.columns:
        tw = tw[tw["season_type"] == "REG"]
    return tw[["team", "week", "opponent_team"]].drop_duplicates()


def player_team(season: int) -> pd.DataFrame:
    """Which team a player actually played for -- his most-played team that
    season. A midseason trade changes the schedule, so this cannot be taken
    from the prior year."""
    wk = weekly_with_points(season)
    counts = (wk.groupby(["name_key", "team"])["week"].nunique()
                .rename("n").reset_index()
                .sort_values(["name_key", "n"], ascending=[True, False]))
    top = counts.drop_duplicates("name_key")[["name_key", "team"]]
    top["season"] = season
    return top


def team_offense_prior(season: int) -> pd.DataFrame:
    """The player's own team's offensive fantasy points per game, prior year.

    This is the durable half of 'team environment' -- offense persists far
    better than defense. It is the feature V4 tried and failed to load.
    """
    wk = weekly_with_points(season - 1)
    wk = wk[wk["position"].isin(POSITIONS)]
    games = wk.groupby("team")["week"].nunique().rename("games")
    tot = (wk.groupby("team")["fantasy_points"].sum().rename("pts")
             .to_frame().join(games))
    tot["team_off_prior"] = tot["pts"] / tot["games"].clip(lower=1)
    tot["team_off_prior"] = (
        (tot["team_off_prior"] - tot["team_off_prior"].mean())
        / tot["team_off_prior"].std(ddof=0)
    )
    out = tot[["team_off_prior"]].reset_index()
    out["season"] = season
    return out


def build_features() -> pd.DataFrame:
    parts = []
    for season in SEASONS:
        sched = schedule(season)
        dprior = defense_allowed(season - 1).drop(columns="season")
        teams = player_team(season)
        offense = team_offense_prior(season)

        # Opponents faced twice are the divisional ones: a recurring, durable
        # effect rather than a one-week matchup. This is Matt's original
        # framing -- "a player might just sit in that conference".
        faced = (sched[sched["week"] <= REG_WEEKS]
                 .groupby(["team", "opponent_team"]).size().rename("times")
                 .reset_index())
        repeats = faced[faced["times"] >= 2][["team", "opponent_team"]]

        rows = []
        for pos in POSITIONS:
            d = dprior[dprior["position"] == pos][["opponent_team", "z"]]
            joined = sched.merge(d, on="opponent_team", how="left")
            reg = (joined[joined["week"] <= REG_WEEKS]
                   .groupby("team")["z"].mean().rename("sos_all"))
            post = (joined[joined["week"].isin(PLAYOFF_WEEKS)]
                    .groupby("team")["z"].mean().rename("sos_playoff"))
            div = (repeats.merge(d, on="opponent_team", how="left")
                          .groupby("team")["z"].mean().rename("div_repeat"))
            block = pd.concat([reg, post, div], axis=1).reset_index()
            block["position"] = pos
            block["season"] = season
            rows.append(block)

        feats = pd.concat(rows, ignore_index=True).merge(
            offense, on=["team", "season"], how="left")
        parts.append(feats.merge(teams, on=["team", "season"], how="inner"))

    return pd.concat(parts, ignore_index=True)


# --- estimation ------------------------------------------------------------

def clustered_corr(frame: pd.DataFrame, feature: str):
    """Correlation with an SE clustered by (season, team).

    A schedule is a property of a team, not of a player, so 12 receivers on one
    roster are one observation, not twelve. Clustering is the whole difference
    between a real finding here and an artifact.
    """
    sub = frame[frame[feature].notna() & frame["resid"].notna()]
    if len(sub) < 40:
        return None
    r = float(np.corrcoef(sub[feature], sub["resid"])[0, 1])

    per = []
    for _, g in sub.groupby(["season", "team"]):
        if len(g) > 3 and g[feature].std() > 0 and g["resid"].std() > 0:
            per.append(np.corrcoef(g[feature], g["resid"])[0, 1])
    # Within a team-season the schedule feature is nearly constant across
    # players of the same position, so the within-cluster correlation is not
    # estimable. Cluster the MEANS instead: one residual per team-season.
    agg = (sub.groupby(["season", "team"])
              .agg(f=(feature, "mean"), r=("resid", "mean")).reset_index())
    rb = float(np.corrcoef(agg["f"], agg["r"])[0, 1]) if len(agg) > 3 else np.nan
    n_cl = len(agg)
    se = np.sqrt(max(1 - rb * rb, 1e-9) / max(n_cl - 2, 1))
    t = rb / se if se > 0 else np.nan
    return r, rb, t, len(sub), n_cl


print("building residuals...", flush=True)
resid = residual_frame()
print(f"  {len(resid)} draftable player-seasons", flush=True)

print("building schedule features...", flush=True)
feats = build_features()

frame = resid.merge(
    feats, left_on=["season", "name_key", "pos"],
    right_on=["season", "name_key", "position"], how="inner")
print(f"  matched {len(frame)} of {len(resid)} "
      f"({100*len(frame)/max(len(resid),1):.0f}%)\n", flush=True)

print("V5  DOES THE SCHEDULE PREDICT WHAT OUR BOARD GETS WRONG?")
print("  resid = log(realized ppg / curve prediction). Our board has NO schedule")
print("  input and reproduces the experts within position, so a real coefficient")
print("  here means the MARKET underweights schedule.\n")
print(f"  {'feature':<16}{'corr':>8}{'clustered':>11}{'t':>8}{'n':>7}{'teams':>7}")
for f in FEATURES:
    if f not in frame.columns:
        print(f"  {f:<16}  MISSING")
        continue
    res = clustered_corr(frame, f)
    if res is None:
        print(f"  {f:<16}  too few rows")
        continue
    r, rb, t, n, n_cl = res
    flag = "  <-- survives" if abs(t) >= 2 else ""
    print(f"  {f:<16}{r:>+8.3f}{rb:>+11.3f}{t:>+8.2f}{n:>7}{n_cl:>7}{flag}")

print("\n  by position (clustered t)")
print(f"  {'feature':<16}" + "".join(f"{p:>10}" for p in POSITIONS))
for f in FEATURES:
    if f not in frame.columns:
        continue
    cells = []
    for p in POSITIONS:
        res = clustered_corr(frame[frame["pos"] == p], f)
        cells.append(f"{res[2]:>+10.2f}" if res else f"{'--':>10}")
    print(f"  {f:<16}" + "".join(cells))

print("\n  PRE-REGISTERED: expected near zero for sos_all. Kill any feature")
print("  with |t| < 2. Clustered by team-season -- a schedule is a property")
print("  of a team, so a roster of receivers is ONE observation, not twelve.")
