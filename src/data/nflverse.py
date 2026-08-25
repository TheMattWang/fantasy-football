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

import datetime as _dt
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

from .paths import ensure, nflverse_dir

BASE = "https://github.com/nflverse/nflverse-data/releases/download"

_USER_AGENT = "fantasy-draft-agent/2.0 (personal league tooling)"


def _is_live(season: Optional[int]) -> bool:
    """Is this season still being played, and therefore still being updated?

    A finished season's parquet never changes, so checking it is a wasted round
    trip on every load. The current one changes every week, which is precisely
    where the write-once cache used to go wrong.

    The NFL year rolls over in March: nflverse publishes the next season's
    assets long before September, and the prior season stops changing well
    before that. Anything at or past the current NFL year is treated as live.
    """
    if season is None:
        return True          # season-independent assets (players, schedules)
    today = _dt.date.today()
    nfl_year = today.year if today.month >= 3 else today.year - 1
    return int(season) >= nfl_year

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
    # One all-seasons file, republished daily, and it carries the UPCOMING
    # season before a game is played -- which is what makes it a bye-week
    # source in August. See `bye_weeks`.
    "schedules":     ("schedules",       "games"),
}

# nflverse and the fantasy sites disagree about two clubs, and a silent join on
# the raw code drops both. Measured on the 2026 board: 30 of 32 teams matched,
# and the two that did not were Jacksonville and the Rams -- not a data gap, a
# spelling one.
TEAM_ALIASES: Dict[str, str] = {
    "JAX": "JAC",
    "LA": "LAR",
    "STL": "LAR", "SD": "LAC", "OAK": "LV",   # relocations, for historical seasons
    "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
}


def normalize_team(code: object) -> str:
    """Canonical team abbreviation, so joins across sources cannot silently drop.

    The player-name equivalent, :func:`src.projections.ecr.normalize_name`, exists
    for the same reason and makes the same argument: every mismatch drops a real
    player off the board. A team mismatch is worse, because it drops the whole
    roster at once -- and quietly, since the join still succeeds for everyone else.
    """
    text = str(code or "").strip().upper()
    return TEAM_ALIASES.get(text, text)


def bye_weeks(season: int, *, refresh: bool = False) -> Dict[str, int]:
    """Each team's bye week, derived from the schedule it does not appear in.

    Deliberately derived rather than read: no feed publishes "bye week" as a
    field, but a team's bye is exactly the regular-season week in which it plays
    no game, and that is a fact about the schedule.

    Preferred over the fantasy-market bye that used to be the only source, which
    covered 188 of 505 board rows. This covers every team, so it covers every
    rostered player who has one. The two agreed exactly where both had an
    opinion -- 30 of 30 comparable teams on the 2026 board.
    """
    games = load("schedules", refresh=refresh)
    games = games[(games["season"] == int(season)) & (games["game_type"] == "REG")]
    if games.empty:
        return {}

    weeks = set(range(int(games["week"].min()), int(games["week"].max()) + 1))
    home = games["home_team"].map(normalize_team)
    away = games["away_team"].map(normalize_team)

    byes: Dict[str, int] = {}
    for team in sorted(set(home) | set(away)):
        played = set(games.loc[(home == team) | (away == team), "week"].astype(int))
        missing = sorted(weeks - played)
        # Exactly one, or we do not understand the schedule and should say so by
        # omission rather than guess which week is the real bye.
        if len(missing) == 1:
            byes[team] = missing[0]
    return byes

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


# Where a cached asset's HTTP validators live, so freshness can be checked
# without downloading the body. One JSON file per parquet, same stem.
def _stamp_path(path: "Path") -> "Path":
    return path.with_suffix(path.suffix + ".etag.json")


def _is_current(url: str, path: "Path", timeout: float) -> bool:
    """Has the remote asset changed since we cached it?

    A conditional GET, which is the honest answer and a free one: nflverse
    serves `ETag` and `Last-Modified`, so `If-None-Match` returns **304 Not
    Modified** and no body when the file is unchanged. That beats a TTL in both
    directions -- no needless re-downloads, and no stale window during which we
    would serve last week's data because an interval had not elapsed yet.

    Errs toward "current" on any network failure. Being offline must not blow
    away a usable cache; the caller can force the issue with `refresh=True`.
    """
    stamp = _stamp_path(path)
    if not stamp.exists():
        return False
    try:
        known = json.loads(stamp.read_text())
    except (OSError, ValueError):
        return False

    headers = {"User-Agent": _USER_AGENT}
    if known.get("etag"):
        headers["If-None-Match"] = known["etag"]
    elif known.get("last_modified"):
        headers["If-Modified-Since"] = known["last_modified"]
    else:
        return False

    try:
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.close()
    except requests.RequestException:
        return True
    return response.status_code == 304


def _record_stamp(url: str, path: "Path", timeout: float) -> None:
    """Save the validators for the copy we just cached."""
    try:
        response = requests.head(url, headers={"User-Agent": _USER_AGENT},
                                 timeout=timeout, allow_redirects=True)
        payload = {"etag": response.headers.get("ETag"),
                   "last_modified": response.headers.get("Last-Modified"),
                   "checked_utc": _dt.datetime.now(tz=_dt.timezone.utc)
                                      .isoformat(timespec="seconds")}
        _stamp_path(path).write_text(json.dumps(payload, indent=1, sort_keys=True))
    except (requests.RequestException, OSError):
        # No stamp means the next load re-downloads. Wasteful, never wrong.
        pass


def load(
    dataset: str,
    seasons: Optional[Iterable[int]] = None,
    *,
    refresh: bool = False,
    check_remote: Optional[bool] = None,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Load an nflverse dataset, caching each season's parquet locally.

    Args:
        dataset: key of :data:`DATASETS`.
        seasons: seasons to fetch; omit for season-independent assets.
        refresh: re-download unconditionally, without asking the server.
        check_remote: ask the server whether the cache is current. Defaults to
            True for the *current* season and False for finished ones, because
            a completed season's file never changes and every check would be a
            pointless round trip.

    Why `check_remote` exists at all: this cache used to be write-once. The
    `refresh` flag was plumbed nowhere -- no caller anywhere passed it -- so the
    first in-season load of a season pinned that season's data forever. The
    failure was silent and it compounded, because `injury_report` and
    `observed_to_date` both filter the cached frame by week: a file cached in
    week 1 does not merely go stale by week 5, it filters to **empty**, which
    the callers read as "no injuries reported" and score every ruled-out player
    as fully healthy. Exactly the bug the board's `--refresh` fix already cured
    once, still live on the path where the data changes weekly.
    """
    cache = ensure(nflverse_dir())
    season_list: List[Optional[int]] = list(seasons) if seasons is not None else [None]

    frames = []
    for season in season_list:
        url = _asset_url(dataset, season)
        name = url.rsplit("/", 1)[-1]
        path = cache / name

        ask = check_remote if check_remote is not None else _is_live(season)
        stale = refresh or not path.exists() or (ask and not _is_current(url, path, timeout))

        if stale:
            frame = pd.read_parquet(url)
            frame.to_parquet(path, index=False)
            _record_stamp(url, path, timeout)
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
    refresh: bool = False,
) -> pd.DataFrame:
    """Per player-week fantasy points, joined to identity columns."""
    weekly = load("weekly", seasons, refresh=refresh)

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
