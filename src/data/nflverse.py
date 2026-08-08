"""nflverse ingestion, ported to the current release schema.

Why this replaces clean.py's ingestion
--------------------------------------
1. ``nfl_data_py==0.3.3`` (latest on PyPI) reads
   ``releases/download/player_stats/player_stats_{year}.parquet``, which now
   404s -- nflverse moved the asset to the ``stats_player`` release.

2. Several columns were renamed. Because ``clean.py`` scores with
   ``row.get(col, 0)``, the renames do not raise -- they silently score zero::

       interceptions            -> passing_interceptions
       field_goals_made_0_19    -> fg_made_0_19
       field_goals_made_50_plus -> fg_made_50_59 + fg_made_60_
       extra_points_made        -> pat_made

   Every QB would have scored with zero interceptions and every kicker would
   have projected at 0.0 ppg. So :func:`fantasy_points` asserts its inputs are
   present rather than defaulting them, and the rename map is explicit.

3. Availability. ``clean.py`` computes ``ppg = total / games_with_a_stat_line``,
   which treats "injured" as "did not exist" and is why the 2025 board loved
   Rashee Rice (3 games) at ADP 64. :func:`availability` uses the weekly roster
   status feed instead, so games *missed* are counted.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .paths import ensure, nflverse_dir

BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# friendly name -> (release, asset stem). "{season}" is substituted per season;
# assets without it are season-independent.
DATASETS: Dict[str, tuple] = {
    "weekly":        ("stats_player",    "stats_player_week_{season}"),
    "team_weekly":   ("stats_team",      "stats_team_week_{season}"),
    "rosters":       ("weekly_rosters",  "roster_weekly_{season}"),
    "snaps":         ("snap_counts",     "snap_counts_{season}"),
    "injuries":      ("injuries",        "injuries_{season}"),
    "depth_charts":  ("depth_charts",    "depth_charts_{season}"),
    "players":       ("players",         "players"),
}

# Roster statuses that mean "on the active roster and able to play this week".
ACTIVE_STATUSES = frozenset({"ACT"})

# Statuses that mean rostered but unavailable. DEV (practice squad) and CUT are
# deliberately excluded from both sets -- those weeks are not "missed games" for
# a player who was not on an NFL roster at all.
UNAVAILABLE_STATUSES = frozenset({"INA", "RES", "EXE"})

# Canonical scoring keys -> the columns they consume in the current schema.
# Listed explicitly so a future rename fails loudly instead of scoring zero.
SCORING_COLUMNS: Dict[str, Sequence[str]] = {
    "pass_yd":      ("passing_yards",),
    "pass_td":      ("passing_tds",),
    "int":          ("passing_interceptions",),
    "rush_yd":      ("rushing_yards",),
    "rush_td":      ("rushing_tds",),
    "rec_yd":       ("receiving_yards",),
    "rec_td":       ("receiving_tds",),
    "rec":          ("receptions",),
    "fum_lost":     ("rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"),
    "fgm_0_39":     ("fg_made_0_19", "fg_made_20_29", "fg_made_30_39"),
    "fgm_40_49":    ("fg_made_40_49",),
    "fgm_50_plus":  ("fg_made_50_59", "fg_made_60_"),
    "fg_miss":      ("fg_missed",),
    "pat_made":     ("pat_made",),
    "pat_miss":     ("pat_missed",),
}

# Placeholder only. Real values come from the Yahoo pull -- see
# src/data/league_config.py. Kept so the module is usable before the pull, and
# labelled so nobody mistakes it for the league's actual rules.
PLACEHOLDER_HALF_PPR: Dict[str, float] = {
    "pass_yd": 0.04, "pass_td": 4.0, "int": -1.0,
    "rush_yd": 0.1, "rush_td": 6.0,
    "rec_yd": 0.1, "rec_td": 6.0, "rec": 0.5,
    "fum_lost": -2.0,
    "fgm_0_39": 3.0, "fgm_40_49": 4.0, "fgm_50_plus": 5.0, "fg_miss": -1.0,
    "pat_made": 1.0, "pat_miss": -1.0,
}


class SchemaError(RuntimeError):
    """Raised when an expected nflverse column is absent.

    Existence of this exception is the point: a missing scoring column must stop
    the pipeline, not quietly contribute zero points.
    """


def _asset_url(dataset: str, season: Optional[int]) -> str:
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; have {sorted(DATASETS)}")
    release, stem = DATASETS[dataset]
    if "{season}" in stem:
        if season is None:
            raise ValueError(f"dataset {dataset!r} requires a season")
        stem = stem.format(season=season)
    return f"{BASE}/{release}/{stem}.parquet"


def load(
    dataset: str,
    seasons: Optional[Iterable[int]] = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load an nflverse dataset, caching each season's parquet locally.

    Args:
        dataset: key of :data:`DATASETS`.
        seasons: seasons to fetch; omit for season-independent assets.
        refresh: re-download even if cached.
    """
    cache = ensure(nflverse_dir())
    season_list: List[Optional[int]] = list(seasons) if seasons is not None else [None]

    frames = []
    for season in season_list:
        url = _asset_url(dataset, season)
        name = url.rsplit("/", 1)[-1]
        path = cache / name

        if refresh or not path.exists():
            frame = pd.read_parquet(url)
            frame.to_parquet(path, index=False)
        else:
            frame = pd.read_parquet(path)

        if season is not None and "season" not in frame.columns:
            frame["season"] = season
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _require(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{context}: missing columns {missing}. nflverse renames columns "
            f"between releases; update SCORING_COLUMNS in src/data/nflverse.py "
            f"rather than letting these default to zero."
        )


def fantasy_points(
    weekly: pd.DataFrame,
    scoring: Optional[Dict[str, float]] = None,
    *,
    strict: bool = True,
) -> pd.Series:
    """Fantasy points per player-week under a canonical scoring dict.

    Args:
        weekly: rows from the ``weekly`` dataset.
        scoring: canonical scoring keys (see :data:`SCORING_COLUMNS`) to point
            values. Defaults to :data:`PLACEHOLDER_HALF_PPR` -- pass the real
            league scoring instead.
        strict: raise if a scored stat's column is absent. Leave this on.
    """
    scoring = dict(scoring or PLACEHOLDER_HALF_PPR)

    if strict:
        needed = [
            col
            for key, value in scoring.items()
            if value and key in SCORING_COLUMNS
            for col in SCORING_COLUMNS[key]
        ]
        _require(weekly, needed, "fantasy_points")

    points = pd.Series(0.0, index=weekly.index, dtype="float64")
    for key, value in scoring.items():
        if not value or key not in SCORING_COLUMNS:
            continue
        for column in SCORING_COLUMNS[key]:
            if column in weekly.columns:
                points = points.add(
                    pd.to_numeric(weekly[column], errors="coerce").fillna(0.0) * value,
                    fill_value=0.0,
                )
    return points


def availability(
    seasons: Iterable[int],
    *,
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """Games available vs. missed per player-season, from weekly roster status.

    This is the fix for the 2025 board's availability bias. ``clean.py`` divided
    by games *played*, so a player who appeared three times was rated on his
    three healthy games; here the weeks he was on IR or inactive are counted.

    Returns one row per (season, gsis_id) with ``games_active``,
    ``games_unavailable``, ``games_rostered`` and ``available_rate``.
    """
    rosters = load("rosters", seasons)

    _require(rosters, ["season", "week", "status"], "availability")
    if regular_season_only and "game_type" in rosters.columns:
        rosters = rosters[rosters["game_type"] == "REG"]

    id_col = "gsis_id" if "gsis_id" in rosters.columns else "player_id"
    _require(rosters, [id_col], "availability")

    status = rosters["status"].astype("string")
    rosters = rosters.assign(
        _active=status.isin(ACTIVE_STATUSES).astype(int),
        _unavailable=status.isin(UNAVAILABLE_STATUSES).astype(int),
    )

    grouped = (
        rosters.groupby(["season", id_col], as_index=False)
        .agg(
            games_active=("_active", "sum"),
            games_unavailable=("_unavailable", "sum"),
            games_rostered=("week", "nunique"),
        )
        .rename(columns={id_col: "gsis_id"})
    )

    denominator = grouped["games_active"] + grouped["games_unavailable"]
    grouped["available_rate"] = (
        grouped["games_active"] / denominator.where(denominator > 0)
    )
    return grouped


def weekly_fantasy(
    seasons: Iterable[int],
    scoring: Optional[Dict[str, float]] = None,
    *,
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """Per player-week fantasy points, joined to identity columns."""
    weekly = load("weekly", seasons)

    if regular_season_only and "season_type" in weekly.columns:
        weekly = weekly[weekly["season_type"] == "REG"]

    keep = [
        c
        for c in (
            "season", "week", "player_id", "player_name", "player_display_name",
            "position", "team", "recent_team", "targets", "carries",
            "target_share", "air_yards_share", "wopr",
        )
        if c in weekly.columns
    ]
    out = weekly[keep].copy()
    out["fantasy_points"] = fantasy_points(weekly, scoring)
    return out


def season_rates(
    seasons: Iterable[int],
    scoring: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Per player-season totals with a *correct* per-game denominator.

    ``ppg_played``     points / games with a stat line   (clean.py's number)
    ``ppg_available``  points / games on the active roster
    ``available_rate`` share of rostered weeks the player was active

    The gap between the first two columns is precisely the bias that put Rashee
    Rice, Chris Godwin, Joe Mixon and Alvin Kamara in the 2025 board's top 15.
    """
    weekly = weekly_fantasy(seasons, scoring)

    totals = (
        weekly.groupby(["season", "player_id"], as_index=False)
        .agg(
            player_name=("player_display_name", "first")
            if "player_display_name" in weekly.columns
            else ("player_name", "first"),
            position=("position", "first"),
            games_played=("week", "nunique"),
            total_points=("fantasy_points", "sum"),
        )
    )
    totals["ppg_played"] = totals["total_points"] / totals["games_played"].clip(lower=1)

    avail = availability(seasons)
    merged = totals.merge(
        avail, how="left", left_on=["season", "player_id"], right_on=["season", "gsis_id"]
    )
    merged["ppg_available"] = merged["total_points"] / merged["games_active"].where(
        merged["games_active"] > 0
    )
    return merged.drop(columns=["gsis_id"], errors="ignore")


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m src.data.nflverse`` -- smoke-test ingestion and show the bias."""
    import argparse

    parser = argparse.ArgumentParser(description="nflverse ingestion smoke test")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025])
    args = parser.parse_args(argv)

    print(f"cache: {nflverse_dir()}")
    rates = season_rates(args.seasons)
    skill = rates[rates["position"].isin(["QB", "RB", "WR", "TE"])]
    ranked = skill[skill["games_played"] >= 2].nlargest(12, "ppg_played")

    print(f"\n{len(rates)} player-seasons\n")
    print("top by ppg_played (clean.py's metric) vs ppg_available:")
    cols = ["player_name", "position", "games_played", "games_active",
            "ppg_played", "ppg_available", "available_rate"]
    print(ranked[[c for c in cols if c in ranked.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
