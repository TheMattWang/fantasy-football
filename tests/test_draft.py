"""Tests for snake mechanics and the opponent model.

The opponent model is what made the 2025 backtest useless: against
``1/(adp+1)`` samplers with no roster limits, elite players fell constantly and
every strategy looked good. These pin the properties that make a simulated draft
resemble a real one.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.draft.engine import DraftBoard, DraftSim, snake_order  # noqa: E402
from src.draft.opponents import (  # noqa: E402
    EARLIEST_ROUND,
    POSITION_LIMITS,
    OpponentModel,
    assign_archetypes,
    calibrate_sigma_scale,
    draw_noise,
)


def make_board(n_per_pos=None):
    n_per_pos = n_per_pos or {"QB": 24, "RB": 60, "WR": 70, "TE": 24, "K": 16, "DST": 16}
    rows, adp = [], 1

    # Interleave the skill positions so ADP order is not trivially grouped.
    # K/DST are appended afterwards, so they must NOT gate the loop -- draining
    # only the skill pools while testing `any(pools.values())` never terminates.
    skill = {p: list(range(n_per_pos.get(p, 0)))
             for p in ("QB", "RB", "WR", "TE")}
    cycle = ("RB", "WR", "WR", "TE", "QB", "RB")
    while any(skill.values()):
        for pos in cycle:
            if skill.get(pos):
                i = skill[pos].pop(0)
                rows.append({
                    "player_name": f"{pos}_{i}", "position": pos,
                    "adp_rank": float(adp), "ecr_sd": 1.0 + adp * 0.05,
                    "VORP": max(0.0, 200.0 - adp),
                })
                adp += 1

    for pos in ("K", "DST"):
        for i in range(n_per_pos.get(pos, 0)):
            rows.append({
                "player_name": f"{pos}_{i}", "position": pos,
                "adp_rank": float(adp), "ecr_sd": 8.0, "VORP": 0.0,
            })
            adp += 1
    return DraftBoard.from_frame(pd.DataFrame(rows))


# --------------------------------------------------------------------------
# Snake mechanics
# --------------------------------------------------------------------------

def test_snake_order_reverses_on_even_rounds():
    order = snake_order(4, 3)
    assert list(order[:4]) == [0, 1, 2, 3]
    assert list(order[4:8]) == [3, 2, 1, 0]
    assert list(order[8:12]) == [0, 1, 2, 3]


def test_every_team_gets_one_pick_per_round():
    order = snake_order(12, 15)
    assert len(order) == 180
    for rnd in range(15):
        assert sorted(order[rnd * 12:(rnd + 1) * 12]) == list(range(12))


def test_picks_until_next_turn_at_the_turn():
    board = make_board()
    sim = DraftSim(board, 12, 15)
    # Team 11 picks 12th and 13th overall -> no gap at the turn.
    for _ in range(11):
        sim.make_pick(int(np.flatnonzero(sim.available)[0]))
    assert sim.on_the_clock == 11
    sim.make_pick(int(np.flatnonzero(sim.available)[0]))
    assert sim.on_the_clock == 11


def test_cannot_draft_the_same_player_twice():
    board = make_board()
    sim = DraftSim(board, 12, 15)
    sim.make_pick(0)
    with pytest.raises(ValueError, match="already drafted"):
        sim.make_pick(0)


# --------------------------------------------------------------------------
# Opponent behaviour
# --------------------------------------------------------------------------

def run_draft(seed=0, n_teams=12, n_rounds=15, **kwargs):
    board = make_board()
    rng = np.random.default_rng(seed)
    model = OpponentModel(
        board, assignments=assign_archetypes(n_teams, -1, rng), rng=rng, **kwargs
    )
    sim = DraftSim(board, n_teams, n_rounds)
    while not sim.complete:
        sim.make_pick(model.choose(sim, sim.on_the_clock))
    return sim


def test_draft_completes_with_full_legal_rosters():
    sim = run_draft()
    assert all(len(r) == 15 for r in sim.rosters)
    assert len(set(sim.picks)) == len(sim.picks)


@pytest.mark.parametrize("position", ["K", "DST"])
def test_kickers_and_defenses_go_late(position):
    """The 2025 model let kickers go in round 3, which made waiting free."""
    frame = run_draft().results_frame()
    taken = frame[frame["position"] == position]
    if taken.empty:
        pytest.skip(f"no {position} drafted")
    assert taken["round"].min() >= EARLIEST_ROUND.get(position, 11)


def test_position_limits_are_respected():
    frame = run_draft().results_frame()
    counts = frame.groupby(["team", "position"]).size()
    for (_, position), n in counts.items():
        assert n <= POSITION_LIMITS.get(position, 99)


def test_elite_players_do_not_last():
    """Top-12 ADP must be gone in round 1-2, or 'wait on him' is free."""
    sim = run_draft()
    frame = sim.results_frame()
    elite = frame[frame["adp"] <= 12]
    assert len(elite) >= 10
    assert elite["overall_pick"].max() <= 30


def test_round_one_is_not_a_single_position():
    """The failure mode when preference bonuses swamp ADP noise."""
    counts = []
    for seed in range(12):
        frame = run_draft(seed=seed).results_frame()
        counts.append(frame[frame["round"] == 1]["position"].value_counts())
    pooled = sum(c.reindex(["RB", "WR", "TE", "QB"], fill_value=0) for c in counts)
    share = pooled / pooled.sum()
    assert share.max() < 0.85, f"round 1 is {share.idxmax()}-dominated: {share.to_dict()}"


def test_more_noise_makes_drafts_less_like_adp():
    """sigma_scale must actually control ADP adherence."""
    def mean_abs_deviation(scale):
        frame = run_draft(seed=3, sigma_scale=scale).results_frame()
        return float((frame["overall_pick"] - frame["adp"]).abs().mean())

    assert mean_abs_deviation(3.0) > mean_abs_deviation(0.25)


def test_everyone_ends_up_with_a_quarterback():
    sim = run_draft()
    frame = sim.results_frame()
    for team in range(12):
        positions = set(frame[frame["team"] == team]["position"])
        assert "QB" in positions


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_calibration_recovers_the_generating_noise():
    """Fit sigma_scale on a draft generated with a known scale."""
    board = make_board()
    truth = 0.5
    rng = np.random.default_rng(11)
    model = OpponentModel(
        board, sigma_scale=truth,
        assignments=assign_archetypes(12, -1, rng), rng=rng,
    )
    sim = DraftSim(board, 12, 15)
    while not sim.complete:
        sim.make_pick(model.choose(sim, sim.on_the_clock))

    table = calibrate_sigma_scale(
        board, sim.results_frame(), candidates=(0.5, 2.0), n_sims=6, seed=5
    )
    assert table.iloc[0]["sigma_scale"] == truth


def test_calibration_rejects_an_unmatchable_draft():
    board = make_board()
    with pytest.raises(ValueError, match="matched the board"):
        calibrate_sigma_scale(
            board,
            pd.DataFrame({"player_name": ["nobody"], "overall_pick": [1]}),
            candidates=(1.0,), n_sims=1,
        )


# --------------------------------------------------------------------------
# Common random numbers
# --------------------------------------------------------------------------

def run_full_draft(board, model, n_teams=12, n_rounds=15):
    sim = DraftSim(board, n_teams, n_rounds)
    while not sim.complete:
        sim.make_pick(model.choose(sim, sim.on_the_clock))
    return list(sim.picks)


def test_a_shared_noise_matrix_makes_the_draft_deterministic():
    """With `noise` set, choose() consumes no RNG at all."""
    board = make_board()
    noise = draw_noise(180, len(board), np.random.default_rng(1))

    picks = []
    for _ in range(2):
        rng = np.random.default_rng(0)
        model = OpponentModel(
            board, assignments=assign_archetypes(12, 0, rng), rng=rng
        )
        model.noise = noise
        picks.append(run_full_draft(board, model))

    assert picks[0] == picks[1]


def test_the_same_noise_gives_the_same_draft_from_different_generators():
    """Two models sharing a noise matrix behave identically despite different rngs."""
    board = make_board()
    noise = draw_noise(180, len(board), np.random.default_rng(2))
    assignments = assign_archetypes(12, 0, np.random.default_rng(7))

    drafts = []
    for seed in (0, 999):
        model = OpponentModel(
            board, assignments=assignments, rng=np.random.default_rng(seed)
        )
        model.noise = noise
        drafts.append(run_full_draft(board, model))

    assert drafts[0] == drafts[1]


def test_without_shared_noise_the_drafts_diverge():
    """Otherwise the test above would pass for the wrong reason."""
    board = make_board()
    assignments = assign_archetypes(12, 0, np.random.default_rng(7))

    drafts = []
    for seed in (0, 999):
        model = OpponentModel(
            board, assignments=assignments, rng=np.random.default_rng(seed)
        )
        drafts.append(run_full_draft(board, model))

    assert drafts[0] != drafts[1]


def test_noise_shorter_than_the_draft_falls_back_to_drawing():
    """A short matrix must not crash or silently reuse row 0."""
    board = make_board()
    rng = np.random.default_rng(0)
    model = OpponentModel(
        board, assignments=assign_archetypes(12, 0, rng), rng=rng
    )
    model.noise = draw_noise(5, len(board), np.random.default_rng(1))

    picks = run_full_draft(board, model)
    assert len(picks) == 180
    assert len(set(picks)) == 180
