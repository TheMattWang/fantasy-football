"""Starting a player who is not playing is the cheapest loss in fantasy.

`week.py` ranked purely on scoring rate, with no notion of whether a player
would take the field. That is a correctness bug, not an edge problem, and it is
worth more than any edge this project has looked for:

  * 2026 week 6 on the test roster: Jahmyr Gibbs (17.5 ppg, the best RB on the
    roster) and Ja'Marr Chase (15.5) are both on bye. The old tool started both.
    Two guaranteed zeros, about 31% of a weekly team total.
  * 2025 week 5, real injury data: Lamar Jackson was OUT, and his 22.6 ppg rate
    was the HIGHEST on the roster -- so he was the automatic QB start, for zero.

The multipliers are measured, not chosen (see `availability.py`). These tests
pin the behaviour those measurements imply.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.league_config import provisional_config  # noqa: E402
from src.inseason.availability import (  # noqa: E402
    BYE_MULTIPLIER,
    STATUS_MULTIPLIER,
    availability,
    injury_report,
)
from src.inseason.waivers import best_lineup, start_sit  # noqa: E402
from src.projections.board import build_board  # noqa: E402
from src.simulation.distributions import build_samples  # noqa: E402


@pytest.fixture(scope="module")
def league():
    import warnings

    config = provisional_config()
    # The saved board, not a freshly built one: `bye` arrives with the FFC
    # market attachment, and building with market_dispersion=False would skip
    # every bye test rather than run it.
    saved = Path(__file__).resolve().parent.parent / "data/processed/board_2026.csv"
    if saved.exists():
        board = pd.read_csv(saved)
    else:  # pragma: no cover - only when the board has never been built
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            board = build_board(2026, config=config, validate=False)
    samples = build_samples(board, n_samples=30, n_weeks=14, seed=0)
    roster = (board[~board["is_streamed"]]
              .sort_values("VORP", ascending=False)
              .head(15)["player_name"].tolist())
    return board, samples, config, roster


def report_of(pairs):
    """An injury report frame: [(player_name, status), ...]."""
    from src.projections.ecr import normalize_name

    return pd.DataFrame(
        [{"name_key": normalize_name(n), "report_status": s} for n, s in pairs]
    )


# --- the measured constants -----------------------------------------------

def test_out_and_doubtful_are_zero_not_merely_discounted():
    """Measured play rates are 0.003 and 0.010. That is not a discount."""
    assert STATUS_MULTIPLIER["Out"] == 0.0
    assert STATUS_MULTIPLIER["Doubtful"] == 0.0


def test_questionable_is_roughly_half():
    """0.567 play rate x (6.95 / 8.66) points when they do play. Ranking a
    Questionable player at his full rate overstates him by over 2x."""
    assert 0.3 < STATUS_MULTIPLIER["Questionable"] < 0.6


def test_a_bye_is_certain():
    assert BYE_MULTIPLIER == 0.0


# --- the report loader must never invent good news ------------------------

def test_a_missing_feed_yields_no_information_not_full_health():
    """If the injury feed is unavailable, the honest answer is 'unknown'. An
    exception would crash the tool; a claim that nobody is hurt is worse."""
    frame = injury_report(1970, 1)
    assert list(frame.columns) == ["name_key", "report_status"]
    assert frame.empty


def test_a_real_week_has_designations():
    frame = injury_report(2025, 5)
    assert len(frame) > 20
    assert set(frame["report_status"]) <= {"Out", "Doubtful", "Questionable"}
    assert frame["name_key"].is_unique


# --- the multipliers land on the right players ----------------------------

def test_a_bye_zeroes_the_player(league):
    board, samples, _, roster = league
    on_bye = board.dropna(subset=["bye"]) if "bye" in board else board.iloc[:0]
    if on_bye.empty:
        pytest.skip("board carries no bye column")
    target = on_bye.iloc[0]
    week = int(target["bye"])

    avail = availability(samples, week=week, board=board)
    row = samples.index[target["player_name"]]
    assert avail["multiplier"].iloc[row] == 0.0
    assert avail["reason"].iloc[row] == "BYE"


def test_an_out_designation_zeroes_the_player(league):
    _, samples, _, roster = league
    avail = availability(samples, week=3,
                         report=report_of([(roster[0], "Out")]))
    row = samples.index[roster[0]]
    assert avail["multiplier"].iloc[row] == 0.0
    assert avail["reason"].iloc[row] == "OUT"


def test_questionable_discounts_without_zeroing(league):
    _, samples, _, roster = league
    avail = availability(samples, week=3,
                         report=report_of([(roster[0], "Questionable")]))
    row = samples.index[roster[0]]
    assert 0.0 < avail["multiplier"].iloc[row] < 1.0


def test_everyone_else_is_untouched(league):
    _, samples, _, roster = league
    avail = availability(samples, week=3,
                         report=report_of([(roster[0], "Out")]))
    others = [samples.index[p] for p in roster[1:]]
    assert (avail["multiplier"].iloc[others] == 1.0).all()


def test_a_bye_outranks_an_injury_designation(league):
    """A bye is certain and an injury report is probabilistic, so Questionable
    must not soften a bye back up toward startable."""
    board, samples, _, _ = league
    if "bye" not in board or board["bye"].isna().all():
        pytest.skip("board carries no bye column")
    target = board.dropna(subset=["bye"]).iloc[0]
    week = int(target["bye"])

    avail = availability(samples, week=week, board=board,
                         report=report_of([(target["player_name"], "Questionable")]))
    row = samples.index[target["player_name"]]
    assert avail["multiplier"].iloc[row] == 0.0
    assert avail["reason"].iloc[row] == "BYE"


def test_the_worst_designation_wins_a_duplicate(league):
    _, samples, _, roster = league
    frame = report_of([(roster[0], "Questionable"), (roster[0], "Out")])
    avail = availability(samples, week=3, report=frame)
    assert avail["multiplier"].iloc[samples.index[roster[0]]] == 0.0


# --- the decision actually changes ----------------------------------------

def test_an_out_player_gets_benched_however_good_he_is(league):
    """The load-bearing test. Lamar Jackson's 22.6 ppg was the best rate on the
    roster in 2025 week 5, and he was OUT."""
    _, samples, config, roster = league

    before = best_lineup(roster, samples, config)
    started = {n for slot, names in before.items() if slot != "BN" for n in names}
    best_starter = max(started, key=lambda n: float(samples.decision_score[samples.index[n]]))

    avail = availability(samples, week=3, report=report_of([(best_starter, "Out")]))
    after = best_lineup(roster, samples, config, availability=avail)
    still = {n for slot, names in after.items() if slot != "BN" for n in names}

    assert best_starter not in still, f"{best_starter} is OUT and still starting"


def test_without_availability_nothing_changes(league):
    _, samples, config, roster = league
    assert best_lineup(roster, samples, config) == best_lineup(
        roster, samples, config, availability=None
    )


def test_start_sit_reports_expected_and_status(league):
    _, samples, config, roster = league
    avail = availability(samples, week=3, report=report_of([(roster[0], "Out")]))
    frame = start_sit(roster, samples, config, availability=avail)

    assert {"expected", "status"} <= set(frame.columns)
    row = frame[frame["player"] == roster[0]].iloc[0]
    assert row["expected"] == 0.0
    assert row["status"] == "OUT"
    assert row["rate"] > 0, "the rate is unchanged -- only availability scales it"
