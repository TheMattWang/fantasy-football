"""The start/sit tool has to actually use the season, or it is a lookup table.

`best_lineup` and `start_sit` both took a `week` argument and never read it.
Both ranked by `samples.decision_score` -- the August projection -- so they
returned the same lineup in week 14 as in week 1. The ignored parameter was the
worst part: it made the function look like it had in-season logic.

T3 measured the gap between a frozen lineup and one that reacts at **+1.523
ranks**, larger than any draft edge V1-V7 could find. These tests pin that the
tool now closes it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.league_config import provisional_config  # noqa: E402
from src.inseason.waivers import (  # noqa: E402
    best_lineup,
    blended_scores,
    start_sit,
)
from src.projections.board import build_board  # noqa: E402
from src.simulation.distributions import build_samples  # noqa: E402


@pytest.fixture(scope="module")
def league():
    config = provisional_config()
    board = build_board(2026, config=config, validate=False, market_dispersion=False)
    samples = build_samples(board, n_samples=50, n_weeks=14, seed=0)
    roster = (board[~board["is_streamed"]]
              .sort_values("VORP", ascending=False)
              .head(15)["player_name"].tolist())
    return board, samples, config, roster


def observed_for(samples, name, points, games):
    """A season-to-date frame naming one player."""
    from src.projections.ecr import normalize_name

    return pd.DataFrame([{"name_key": normalize_name(name),
                          "points": points, "games": games}])


# --- the estimator --------------------------------------------------------

def test_no_observations_leaves_the_projection_untouched(league):
    _, samples, _, _ = league
    np.testing.assert_allclose(
        blended_scores(samples, None), samples.decision_score, rtol=1e-6
    )


def test_an_empty_frame_is_the_same_as_none(league):
    _, samples, _, _ = league
    empty = pd.DataFrame(columns=["name_key", "points", "games"])
    np.testing.assert_allclose(
        blended_scores(samples, empty), samples.decision_score, rtol=1e-6
    )


def test_a_player_who_has_not_played_keeps_his_projection(league):
    """A rookie with no games has nothing to update on."""
    _, samples, _, roster = league
    other = roster[1]
    scores = blended_scores(samples, observed_for(samples, roster[0], 200.0, 10))
    row = samples.index[other]
    assert scores[row] == pytest.approx(float(samples.decision_score[row]))


def test_a_collapse_drags_the_rate_down(league):
    _, samples, _, roster = league
    name = roster[0]
    row = samples.index[name]
    before = float(samples.decision_score[row])
    after = blended_scores(samples, observed_for(samples, name, 10.0, 10))[row]
    assert after < before


def test_a_breakout_pulls_the_rate_up(league):
    _, samples, _, roster = league
    name = roster[-1]
    row = samples.index[name]
    before = float(samples.decision_score[row])
    after = blended_scores(samples, observed_for(samples, name, 300.0, 10))[row]
    assert after > before


def test_the_prior_weight_controls_how_fast_it_reacts(league):
    _, samples, _, roster = league
    name = roster[0]
    row = samples.index[name]
    obs = observed_for(samples, name, 10.0, 10)
    stubborn = blended_scores(samples, obs, prior_games=100.0)[row]
    jumpy = blended_scores(samples, obs, prior_games=0.5)[row]
    assert jumpy < stubborn


# --- the lineup actually changes -----------------------------------------

def test_a_collapsed_starter_gets_benched(league):
    """The decision the frozen version could never make."""
    _, samples, config, roster = league

    before = best_lineup(roster, samples, config)
    started = {n for slot, names in before.items() if slot != "BN" for n in names}
    victim = next(n for n in roster if n in started)

    after = best_lineup(roster, samples, config,
                        observed=observed_for(samples, victim, 0.5, 9))
    still = {n for slot, names in after.items() if slot != "BN" for n in names}
    assert victim not in still, f"{victim} scored 0.5 ppg over 9 games and still starts"


def test_the_lineup_is_unchanged_when_nothing_is_observed(league):
    _, samples, config, roster = league
    assert best_lineup(roster, samples, config) == best_lineup(
        roster, samples, config, observed=None
    )


def test_every_roster_spot_is_accounted_for(league):
    _, samples, config, roster = league
    lineup = best_lineup(roster, samples, config)
    placed = [n for names in lineup.values() for n in names]
    assert sorted(placed) == sorted(roster)
    assert len(placed) == len(set(placed)), "a player was started twice"


# --- start_sit reports the reason ----------------------------------------

def test_start_sit_shows_what_moved(league):
    _, samples, config, roster = league
    victim = roster[0]
    frame = start_sit(roster, samples, config,
                      observed=observed_for(samples, victim, 0.5, 9))

    assert {"proj_ppg", "rate", "delta", "slot", "start"} <= set(frame.columns)
    row = frame[frame["player"] == victim].iloc[0]
    assert row["delta"] < 0
    assert row["rate"] < row["proj_ppg"]


def test_start_sit_starts_every_slot_it_can_fill(league):
    """The fixture roster is skill players only, so the K and DEF slots have no
    eligible player and stay empty. Every other slot must be filled."""
    _, samples, config, roster = league
    frame = start_sit(roster, samples, config)
    assert sorted(frame["player"]) == sorted(roster)

    unfillable = {"K", "DEF", "DST"}
    expected = sum(count for slot, count in config.starting_slots.items()
                   if slot not in unfillable)
    assert frame["start"].sum() == expected
