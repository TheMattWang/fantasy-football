"""Tests for nflverse ingestion.

The important behaviour here is that a renamed column *raises*. nflverse renamed
`interceptions` -> `passing_interceptions` and `extra_points_made` -> `pat_made`
between releases; because clean.py scored with `row.get(col, 0)`, that would have
scored every QB with zero interceptions and every kicker at 0.0 ppg, silently.

Network-dependent tests are marked and skipped by default.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.nflverse import (  # noqa: E402
    PLACEHOLDER_HALF_PPR,
    SCORING_COLUMNS,
    SchemaError,
    _asset_url,
    availability,
    fantasy_points,
)


def _weekly_row(**overrides):
    """One player-week with every scored column present and zeroed."""
    row = {col: 0.0 for cols in SCORING_COLUMNS.values() for col in cols}
    row.update(overrides)
    return pd.DataFrame([row])


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_scores_a_known_stat_line():
    """300 pass yds, 2 pass TD, 1 INT, 50 rush yds under the placeholder rules."""
    df = _weekly_row(
        passing_yards=300, passing_tds=2, passing_interceptions=1, rushing_yards=50
    )
    # 300*.04 + 2*4 + 1*-1 + 50*.1 = 12 + 8 - 1 + 5
    assert fantasy_points(df).iloc[0] == pytest.approx(24.0)


def test_half_ppr_receptions_are_scored():
    df = _weekly_row(receptions=8, receiving_yards=100, receiving_tds=1)
    # 8*.5 + 100*.1 + 6 = 4 + 10 + 6
    assert fantasy_points(df).iloc[0] == pytest.approx(20.0)


def test_fifty_plus_field_goals_sum_two_columns():
    """nflverse split 50+ into fg_made_50_59 and fg_made_60_."""
    df = _weekly_row(fg_made_50_59=1, fg_made_60_=1)
    assert fantasy_points(df).iloc[0] == pytest.approx(10.0)


def test_fumbles_lost_sum_three_columns():
    df = _weekly_row(rushing_fumbles_lost=1, receiving_fumbles_lost=1)
    assert fantasy_points(df).iloc[0] == pytest.approx(-4.0)


# --------------------------------------------------------------------------
# The renames that broke clean.py -- these must raise, not score zero
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dropped,old_name",
    [
        ("passing_interceptions", "interceptions"),
        ("pat_made", "extra_points_made"),
        ("fg_made_0_19", "field_goals_made_0_19"),
        ("fg_made_50_59", "field_goals_made_50_plus"),
        ("receptions", "receptions"),
    ],
)
def test_missing_scored_column_raises(dropped, old_name):
    df = _weekly_row(passing_yards=300).drop(columns=[dropped])
    with pytest.raises(SchemaError, match=dropped):
        fantasy_points(df)


def test_unscored_column_may_be_absent():
    """A stat worth 0 points in this league need not be present."""
    scoring = dict(PLACEHOLDER_HALF_PPR, int=0.0)
    df = _weekly_row(passing_yards=300).drop(columns=["passing_interceptions"])
    assert fantasy_points(df, scoring).iloc[0] == pytest.approx(12.0)


def test_non_strict_mode_tolerates_missing_columns():
    df = _weekly_row(passing_yards=300).drop(columns=["passing_interceptions"])
    assert fantasy_points(df, strict=False).iloc[0] == pytest.approx(12.0)


# --------------------------------------------------------------------------
# Asset URLs -- pin the release paths that nfl_data_py 0.3.3 gets wrong
# --------------------------------------------------------------------------

def test_weekly_url_uses_the_current_release():
    url = _asset_url("weekly", 2025)
    assert url.endswith("/stats_player/stats_player_week_2025.parquet")
    # the retired path nfl_data_py still points at, which 404s
    assert "player_stats/player_stats_" not in url


def test_season_independent_asset_needs_no_season():
    assert _asset_url("players", None).endswith("/players/players.parquet")


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        _asset_url("nope", 2025)


# --------------------------------------------------------------------------
# Availability -- the fix for the 2025 board's biggest bias
# --------------------------------------------------------------------------

def test_availability_counts_missed_games(monkeypatch):
    """A player active 4 weeks and on IR 13 must not read as 100% available."""
    roster = pd.DataFrame(
        [{"season": 2025, "week": w, "gsis_id": "X", "game_type": "REG",
          "status": "ACT" if w <= 4 else "RES"} for w in range(1, 18)]
    )
    monkeypatch.setattr("src.data.nflverse.load", lambda *a, **k: roster)

    out = availability([2025]).iloc[0]
    assert out["games_active"] == 4
    assert out["games_unavailable"] == 13
    assert out["available_rate"] == pytest.approx(4 / 17)


def test_practice_squad_weeks_are_not_missed_games(monkeypatch):
    """DEV weeks are 'not an NFL player yet', not 'injured'."""
    roster = pd.DataFrame(
        [{"season": 2025, "week": w, "gsis_id": "X", "game_type": "REG",
          "status": "DEV" if w <= 10 else "ACT"} for w in range(1, 18)]
    )
    monkeypatch.setattr("src.data.nflverse.load", lambda *a, **k: roster)

    out = availability([2025]).iloc[0]
    assert out["games_active"] == 7
    assert out["games_unavailable"] == 0
    assert out["available_rate"] == pytest.approx(1.0)


def test_availability_requires_status_column(monkeypatch):
    monkeypatch.setattr(
        "src.data.nflverse.load",
        lambda *a, **k: pd.DataFrame([{"season": 2025, "week": 1, "gsis_id": "X"}]),
    )
    with pytest.raises(SchemaError, match="status"):
        availability([2025])
