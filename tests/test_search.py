"""Tests for draft policies and the replay harness."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from src.data.league_config import from_dict  # noqa: E402
from src.draft.engine import DraftSim  # noqa: E402
from src.draft.opponents import OpponentModel, assign_archetypes  # noqa: E402
from src.draft.search import (  # noqa: E402
    AdpPolicy,
    NeedAdpPolicy,
    OpponentPool,
    VorpGreedyPolicy,
    run_draft,
)
from src.evaluation.replay import summarize  # noqa: E402
from test_draft import make_board  # noqa: E402

CONFIG = from_dict({
    "season": 2025, "league_id": "T", "name": "T", "num_teams": 12,
    "playoff_start_week": 15, "num_playoff_teams": 6,
    "start_week": 1, "end_week": 14,
    "roster_slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R/T": 1,
                     "K": 1, "DEF": 1, "BN": 6},
    "scoring": {},
})


@pytest.fixture(scope="module")
def board():
    return make_board()


def _draft_with(policy, board, seed=0):
    rng = np.random.default_rng(seed)
    opponents = OpponentModel(
        board, assignments=assign_archetypes(12, 0, rng), rng=rng
    )
    return run_draft(board, policy, opponents, our_team=0)


# --------------------------------------------------------------------------
# Baselines behave as advertised
# --------------------------------------------------------------------------

def test_adp_policy_takes_the_best_rank_available(board):
    sim = DraftSim(board, 12, 15)
    assert AdpPolicy(board).choose(sim, 0) == int(np.argmin(board.adp))


def test_adp_policy_is_positionally_pathological(board):
    """Documents WHY plain ADP is a floor and not the bar.

    With no positional awareness it hoards whatever position consensus ranks
    highest once the elite skill players are gone. Beating this proves nothing.
    """
    sim = _draft_with(AdpPolicy(board), board)
    counts = pd.Series([board.positions[r] for r in sim.rosters[0]]).value_counts()
    assert counts.max() >= 5
    assert len(counts) <= 4


def test_need_adp_builds_a_legal_startable_roster(board):
    sim = _draft_with(NeedAdpPolicy(board, CONFIG), board)
    counts = pd.Series([board.positions[r] for r in sim.rosters[0]]).value_counts()
    for required in ("QB", "RB", "WR", "TE"):
        assert counts.get(required, 0) >= 1
    assert counts.get("QB", 0) <= 2


def test_need_adp_fills_mandatory_slots_before_the_draft_ends(board):
    sim = _draft_with(NeedAdpPolicy(board, CONFIG), board)
    positions = {board.positions[r] for r in sim.rosters[0]}
    assert "K" in positions
    assert "DST" in positions or "DEF" in positions


def test_vorp_greedy_prefers_value_over_consensus_order(board):
    sim = DraftSim(board, 12, 15)
    # Give a mid-ADP player an outsized VORP; greedy must take him.
    board.vorp[40] = 10_000.0
    try:
        assert VorpGreedyPolicy(board, CONFIG).choose(sim, 0) == 40
    finally:
        board.vorp[40] = max(0.0, 200.0 - board.adp[40])


def test_policies_are_deterministic_given_the_same_state(board):
    sim = DraftSim(board, 12, 15)
    for policy in (AdpPolicy(board), NeedAdpPolicy(board, CONFIG),
                   VorpGreedyPolicy(board, CONFIG)):
        assert policy.choose(sim, 0) == policy.choose(sim, 0)


def test_every_team_fills_its_roster(board):
    sim = _draft_with(NeedAdpPolicy(board, CONFIG), board)
    assert all(len(r) == 15 for r in sim.rosters)
    assert len(set(sim.picks)) == 180


# --------------------------------------------------------------------------
# Paired summary
# --------------------------------------------------------------------------

def _replay_frame(gain: float, n: int = 40) -> pd.DataFrame:
    """Synthetic replay output where 'good' beats 'need_adp' by `gain` ranks."""
    rng = np.random.default_rng(0)
    rows = []
    for replicate in range(n):
        base = rng.integers(1, 13)
        rows.append({"policy": "need_adp", "replicate": replicate, "slot": 0,
                     "rank": int(base), "wins": 7, "points_for": 1500.0})
        rows.append({"policy": "good", "replicate": replicate, "slot": 0,
                     "rank": int(np.clip(base - gain, 1, 12)), "wins": 8,
                     "points_for": 1600.0})
    return pd.DataFrame(rows)


def test_summarize_reports_a_positive_gain_for_a_better_policy():
    table = summarize(_replay_frame(gain=2)).set_index("policy")
    assert table.loc["good", "rank_gain_vs_need_adp"] > 0
    assert table.loc["good", "t"] > 2


def test_summarize_reports_no_gain_for_an_identical_policy():
    table = summarize(_replay_frame(gain=0)).set_index("policy")
    assert table.loc["good", "rank_gain_vs_need_adp"] == pytest.approx(0.0)


def test_summarize_is_paired_not_pooled():
    """Pairing is what makes ~200 replicates enough instead of thousands."""
    frame = _replay_frame(gain=1, n=60)
    table = summarize(frame).set_index("policy")
    paired_se = table.loc["good", "se"]

    ranks = frame.pivot_table(index="replicate", columns="policy", values="rank")
    unpaired_se = np.sqrt(
        ranks["need_adp"].var(ddof=1) / len(ranks)
        + ranks["good"].var(ddof=1) / len(ranks)
    )
    assert paired_se < unpaired_se


# --------------------------------------------------------------------------
# Shrinkage toward consensus
# --------------------------------------------------------------------------

class _StubPolicy:
    """Stands in for the consensus fallback."""

    def __init__(self, row):
        self.row = row

    def choose(self, sim, team):
        return self.row


def _make_policy(board, candidates, default_row, margin=0.06):
    """A SeasonSimPolicy with evaluate() stubbed to fixed candidates."""
    from src.draft.search import Candidate, SeasonSimPolicy

    policy = SeasonSimPolicy.__new__(SeasonSimPolicy)
    policy.board = board
    policy.confidence_margin = margin
    policy.rollout_policy = _StubPolicy(default_row)
    policy.last_candidates = []
    policy.evaluate = lambda sim, team: [
        Candidate(row=r, name=f"p{r}", position="RB", adp=float(r),
                  utility=u, p_available_next=1.0)
        for r, u in candidates
    ]
    return policy


def test_agent_defers_to_consensus_when_the_gain_is_inside_the_noise(board):
    """The measured failure mode: noisy argmax is worse than a good consensus.

    Per-rollout noise is sd ~0.06 while adjacent candidates differ by ~0.02, so
    deviating on a 0.02 edge is picking at random.
    """
    policy = _make_policy(board, [(7, 0.95), (3, 0.93)], default_row=3)
    assert policy.choose(None, 0) == 3


def test_agent_overrides_consensus_when_the_gain_is_decisive(board):
    policy = _make_policy(board, [(7, 1.20), (3, 0.93)], default_row=3)
    assert policy.choose(None, 0) == 7


def test_agent_keeps_consensus_pick_when_it_is_already_best(board):
    policy = _make_policy(board, [(3, 0.99), (7, 0.90)], default_row=3)
    assert policy.choose(None, 0) == 3


def test_agent_trusts_search_when_consensus_pick_is_not_shortlisted(board):
    policy = _make_policy(board, [(7, 0.95), (9, 0.94)], default_row=42)
    assert policy.choose(None, 0) == 7


def test_margin_of_zero_reproduces_plain_argmax(board):
    policy = _make_policy(board, [(7, 0.951), (3, 0.950)], default_row=3, margin=0.0)
    assert policy.choose(None, 0) == 7


# --------------------------------------------------------------------------
# Mixed opponent field
# --------------------------------------------------------------------------

def test_pool_fills_every_seat_but_ours(board):
    pool = OpponentPool(board, CONFIG, our_team=3, rng=np.random.default_rng(0))
    assert pool.kind_of[3] == "us"
    assert sum(pool.composition().values()) == 11


def test_pool_uses_more_than_one_strategy(board):
    """A field of one strategy is a field with one exploit."""
    kinds = set()
    for seed in range(8):
        pool = OpponentPool(board, CONFIG, our_team=0, rng=np.random.default_rng(seed))
        kinds.update(pool.composition())
    assert len(kinds) >= 2


def test_pool_produces_a_complete_legal_draft(board):
    pool = OpponentPool(board, CONFIG, our_team=0, rng=np.random.default_rng(1))
    sim = DraftSim(board, 12, 15)
    while not sim.complete:
        team = sim.on_the_clock
        sim.make_pick(
            NeedAdpPolicy(board, CONFIG).choose(sim, team)
            if team == 0 else pool.choose(sim, team)
        )
    assert all(len(r) == 15 for r in sim.rosters)
    assert len(set(sim.picks)) == len(sim.picks)


def test_pool_forwards_shared_noise_to_its_archetype_seats(board):
    """CRN has to reach through the pool or the rollout stops being paired."""
    from src.draft.opponents import draw_noise

    noise = draw_noise(180, len(board), np.random.default_rng(2))
    pool = OpponentPool(board, CONFIG, our_team=0, rng=np.random.default_rng(1))
    pool.noise = noise

    assert pool.model.noise is noise
    assert pool.noise is noise

    pool.noise = None
    assert pool.model.noise is None


def test_pool_weights_shift_the_composition(board):
    """Asking for an all-need_adp field must actually produce one."""
    pool = OpponentPool(
        board, CONFIG, our_team=0, rng=np.random.default_rng(0),
        weights={"need_adp": 1.0},
    )
    assert pool.composition() == {"need_adp": 11}
