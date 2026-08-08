"""Tests for the single-source-of-truth league config.

The property that matters most here is ``replacement_rank``: every VORP on the
board is measured against it, so an off-by-a-flex-slot error silently rescales
the entire board. These pin it against the known-good arithmetic in clean.py.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.league_config import (  # noqa: E402
    LeagueConfig,
    LeagueConfigError,
    from_dict,
    load_league_config,
)

# A 12-team, half-PPR Yahoo league with one W/R/T flex -- the shape clean.py
# assumes. Yahoo names the flex slot "W/R/T", not "WRT".
YAHOO_PAYLOAD = {
    "season": 2025,
    "league_id": "123456",
    "name": "Test League",
    "num_teams": 12,
    "scoring_type": "head",
    "start_week": 1,
    "end_week": 17,
    "playoff_start_week": 15,
    "num_playoff_teams": 6,
    "roster_slots": {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R/T": 1,
        "K": 1, "DEF": 1, "BN": 6, "IR": 1,
    },
    "scoring": {
        "11": {"value": 0.5, "name": "Receptions", "display_name": "Rec"},
        "5": {"value": 4.0, "name": "Passing Touchdowns", "display_name": "Pass TD"},
        "6": {"value": -1.0, "name": "Interceptions", "display_name": "Int"},
    },
}


@pytest.fixture
def cfg() -> LeagueConfig:
    return from_dict(YAHOO_PAYLOAD)


# --------------------------------------------------------------------------
# Roster arithmetic
# --------------------------------------------------------------------------

def test_starting_slots_exclude_bench_and_ir(cfg):
    assert "BN" not in cfg.starting_slots
    assert "IR" not in cfg.starting_slots
    assert cfg.starting_slots == {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R/T": 1, "K": 1, "DEF": 1
    }


def test_bench_and_roster_sizes(cfg):
    assert cfg.bench_size == 7          # 6 BN + 1 IR
    assert cfg.roster_size == 16
    assert cfg.total_rounds == 15       # IR is not drafted into


def test_flex_slot_is_recognized(cfg):
    assert cfg.flex_slots == {"W/R/T": frozenset({"WR", "RB", "TE"})}
    assert "W/R/T" not in cfg.dedicated_slots


@pytest.mark.parametrize(
    "position,expected",
    [
        # dedicated starters + this position's share of the single W/R/T slot.
        ("RB", 12 * 2 + round(12 * 1 / 3)),   # 24 + 4  = 28
        ("WR", 12 * 2 + round(12 * 1 / 3)),   # 24 + 4  = 28
        ("TE", 12 * 1 + round(12 * 1 / 3)),   # 12 + 4  = 16
        ("QB", 12 * 1),                        # no flex eligibility  = 12
        ("K", 12 * 1),
        ("DEF", 12 * 1),
    ],
)
def test_replacement_rank_matches_clean_py_arithmetic(cfg, position, expected):
    assert cfg.replacement_rank(position) == expected


def test_replacement_rank_never_zero(cfg):
    assert cfg.replacement_rank("NOT_A_POSITION") == 1


def test_superflex_makes_qbs_scarce():
    """Sanity check that the flex table actually drives scarcity."""
    payload = dict(YAHOO_PAYLOAD, roster_slots={
        **YAHOO_PAYLOAD["roster_slots"], "Q/W/R/T": 1,
    })
    sf = from_dict(payload)
    assert sf.replacement_rank("QB") > from_dict(YAHOO_PAYLOAD).replacement_rank("QB")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_points_per_reception_resolves_half_ppr(cfg):
    """Settles the clean.py (0.5) vs src/core/scoring.py (1.0) conflict."""
    assert cfg.points_per_reception == 0.5


def test_scoring_lookup_by_pattern(cfg):
    assert cfg.scoring_value(r"passing touchdown") == 4.0
    assert cfg.scoring_value(r"interception") == -1.0
    assert cfg.scoring_value(r"nonexistent stat") is None


# --------------------------------------------------------------------------
# Failure modes -- must raise, never silently default
# --------------------------------------------------------------------------

def test_missing_roster_slots_raises():
    with pytest.raises(LeagueConfigError, match="roster slots"):
        from_dict(dict(YAHOO_PAYLOAD, roster_slots={}))


def test_missing_team_count_raises():
    with pytest.raises(LeagueConfigError, match="team count"):
        from_dict(dict(YAHOO_PAYLOAD, num_teams=None))


def test_load_without_a_pull_raises_and_says_how_to_fix(monkeypatch, tmp_path):
    """No config must be a hard stop, not a 12-team guess."""
    monkeypatch.setattr("src.data.league_config.LEAGUE_ROOT", tmp_path / "nope")
    with pytest.raises(LeagueConfigError, match="yahoo_league"):
        load_league_config()


def test_load_reads_a_pulled_config(monkeypatch, tmp_path):
    league = tmp_path / "yahoo_123456_2025"
    league.mkdir(parents=True)
    (league / "league_config.json").write_text(json.dumps(YAHOO_PAYLOAD))
    monkeypatch.setattr("src.data.league_config.LEAGUE_ROOT", tmp_path)

    cfg = load_league_config()
    assert cfg.num_teams == 12
    assert cfg.replacement_rank("RB") == 28
    assert cfg.summary()


def test_load_rejects_mismatched_season(monkeypatch, tmp_path):
    league = tmp_path / "yahoo_123456_2025"
    league.mkdir(parents=True)
    (league / "league_config.json").write_text(json.dumps(YAHOO_PAYLOAD))
    monkeypatch.setattr("src.data.league_config.LEAGUE_ROOT", tmp_path)

    with pytest.raises(LeagueConfigError, match="no league config matching"):
        load_league_config(season=2026)
