"""The in-season CLI: the only place the +1.523 ranks actually gets collected.

T3 measured a realistic in-season policy at +1.523 ranks against +0.832 for a
realistic draft improvement -- and the draft number is not even available to us,
since V1-V7 found no valuation edge. So this tool is worth more than draft_day.py
is, and its failure modes deserve the same care.

The one that matters most: `--week N` must read results through week N-1 and no
further. Reading week N itself would be setting lineups with hindsight, which is
the same off-by-one as the FLEX bug and would make the tool look brilliant while
being useless.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import week  # noqa: E402


# --- arguments ------------------------------------------------------------

def test_week_is_required():
    with pytest.raises(SystemExit):
        week.build_parser().parse_args([])


def test_defaults_point_at_what_draft_day_writes():
    args = week.build_parser().parse_args(["--week", "5"])
    assert args.roster == "data/processed/my_roster.txt"


def test_the_prior_weight_is_adjustable():
    args = week.build_parser().parse_args(["--week", "5", "--prior-games", "8"])
    assert args.prior_games == 8.0


# --- roster file ----------------------------------------------------------

def test_blank_lines_and_comments_are_ignored(tmp_path):
    path = tmp_path / "roster.txt"
    path.write_text("Josh Allen\n\n# my notes\nBijan Robinson\n   \n")
    assert week.read_roster(path) == ["Josh Allen", "Bijan Robinson"]


def test_names_keep_their_internal_spacing(tmp_path):
    path = tmp_path / "roster.txt"
    path.write_text("  Amon-Ra St. Brown  \n")
    assert week.read_roster(path) == ["Amon-Ra St. Brown"]


# --- the refusals ---------------------------------------------------------

def test_a_missing_roster_is_reported_not_crashed(tmp_path, capsys):
    code = week.main(["--week", "3", "--roster", str(tmp_path / "nope.txt")])
    assert code == 1
    assert "no roster" in capsys.readouterr().err


def test_an_empty_roster_is_reported(tmp_path, capsys):
    path = tmp_path / "roster.txt"
    path.write_text("\n# nothing but a comment\n")
    code = week.main(["--week", "3", "--roster", str(path)])
    assert code == 1
    assert "empty" in capsys.readouterr().err


def test_a_missing_board_names_the_build_command(tmp_path, capsys):
    roster = tmp_path / "roster.txt"
    roster.write_text("Josh Allen\n")
    code = week.main(["--week", "3", "--roster", str(roster),
                      "--board", str(tmp_path / "nope.csv")])
    assert code == 1
    assert "--refresh" in capsys.readouterr().err


# --- the hindsight guard --------------------------------------------------

def test_week_n_reads_results_through_n_minus_one(monkeypatch, tmp_path):
    """Reading week N itself would be setting the lineup with hindsight."""
    seen = {}

    def fake(season, through_week, **kwargs):
        seen["through"] = through_week
        import pandas as pd
        return pd.DataFrame(columns=["name_key", "points", "games"])

    monkeypatch.setattr(week, "observed_to_date", fake)

    roster = tmp_path / "roster.txt"
    roster.write_text("Josh Allen\nBijan Robinson\n")
    week.main(["--week", "9", "--roster", str(roster),
               "--board", "data/processed/board_2026.csv", "--samples", "20"])

    assert seen["through"] == 8, "week 9 must not see week 9"


def test_week_one_reads_nothing_at_all(monkeypatch, tmp_path, capsys):
    """There is no week 0. Asking for results would fetch an empty season."""
    called = []
    monkeypatch.setattr(week, "observed_to_date",
                        lambda *a, **k: called.append(1))

    roster = tmp_path / "roster.txt"
    roster.write_text("Josh Allen\nBijan Robinson\n")
    code = week.main(["--week", "1", "--roster", str(roster),
                      "--board", "data/processed/board_2026.csv", "--samples", "20"])

    assert code == 0
    assert not called
    assert "no results yet" in capsys.readouterr().out


# --- it produces a lineup -------------------------------------------------

def test_it_prints_a_lineup_and_exits_clean(tmp_path, capsys):
    roster = tmp_path / "roster.txt"
    roster.write_text("\n".join([
        "Josh Allen", "Bijan Robinson", "Jahmyr Gibbs", "Ja'Marr Chase",
        "Puka Nacua", "Trey McBride",
    ]) + "\n")

    code = week.main(["--week", "1", "--roster", str(roster),
                      "--board", "data/processed/board_2026.csv", "--samples", "20"])
    out = capsys.readouterr().out

    assert code == 0
    assert "WEEK 1 LINEUP" in out
    assert "-- START" in out and "-- BENCH" in out
    assert "Josh Allen" in out


def test_players_missing_from_the_board_are_named_not_dropped_silently(tmp_path, capsys):
    roster = tmp_path / "roster.txt"
    roster.write_text("Josh Allen\nNot A Real Person\n")

    week.main(["--week", "1", "--roster", str(roster),
               "--board", "data/processed/board_2026.csv", "--samples", "20"])

    assert "Not A Real Person" in capsys.readouterr().out
