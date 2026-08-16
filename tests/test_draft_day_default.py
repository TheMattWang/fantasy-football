"""What the draft-day tool actually recommends is a measured claim, so pin it.

E2 measured the season simulation's argmax at 1.582 regret against the best
available candidate, versus 0.891 for consensus -- and 1.548 for a candidate
picked at RANDOM. So `U` is worth displaying and is not worth obeying, and
`draft_day.py` recommends consensus while showing the simulation as context.

That split is easy to undo by accident: the obvious "simplification" is to hand
`show_recommendations` one policy again and let it both fill the table and make
the pick. These tests exist so that regression fails loudly in August rather
than quietly in December.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import draft_day  # noqa: E402
from src.draft.search import Candidate  # noqa: E402


# --- fakes: enough surface for show_recommendations, nothing more ----------

class FakeBoard:
    names = ["Alpha", "Bravo", "Charlie"]


class FakeSim:
    board = FakeBoard()


class FakePolicy:
    """Fills the table. Its top candidate is deliberately NOT the pick."""

    name = "season_sim"

    def __init__(self, candidates):
        self._candidates = candidates

    def evaluate(self, sim, team):
        return list(self._candidates)


class FakeRecommender:
    name = "need_adp"

    def __init__(self, row):
        self._row = row

    def choose(self, sim, team):
        return self._row


def _candidate(row, name, utility, p_next=0.5):
    return Candidate(row=row, name=name, position="RB", adp=float(row + 1),
                     utility=utility, p_available_next=p_next)


CANDIDATES = [
    _candidate(0, "Alpha", 1.7000),      # the simulation's favourite
    _candidate(1, "Bravo", 1.6800),
    _candidate(2, "Charlie", 1.6500),
]


# --- the default ----------------------------------------------------------

def test_the_recommender_defaults_to_consensus():
    args = draft_day.build_parser().parse_args(["--slot", "4"])
    assert args.recommender == "need_adp"


def test_the_simulation_is_still_reachable_on_request():
    args = draft_day.build_parser().parse_args(
        ["--slot", "4", "--recommender", "season_sim"]
    )
    assert args.recommender == "season_sim"


def test_only_the_two_recommenders_are_accepted():
    with pytest.raises(SystemExit):
        draft_day.build_parser().parse_args(
            ["--slot", "4", "--recommender", "vorp_greedy"]
        )


# --- the behaviour the default is there to produce ------------------------

def test_the_pick_comes_from_the_recommender_not_the_table(capsys):
    """The load-bearing one. Table says Alpha; consensus says Charlie."""
    draft_day.show_recommendations(
        FakePolicy(CANDIDATES), FakeRecommender(2), FakeSim(), team=0
    )
    out = capsys.readouterr().out

    assert "PICK: Charlie" in out
    assert "PICK: Alpha" not in out
    assert "[need_adp]" in out


def test_the_disagreement_is_shown_rather_than_hidden(capsys):
    """Silently discarding the simulation would be its own failure mode."""
    draft_day.show_recommendations(
        FakePolicy(CANDIDATES), FakeRecommender(2), FakeSim(), team=0
    )
    out = capsys.readouterr().out

    assert "prefers Alpha" in out
    assert "+0.0500" in out          # 1.7000 - 1.6500, signed
    assert "1.582" in out and "0.891" in out   # why we are not taking it


def test_no_disagreement_notice_when_they_agree(capsys):
    draft_day.show_recommendations(
        FakePolicy(CANDIDATES), FakeRecommender(0), FakeSim(), team=0
    )
    out = capsys.readouterr().out

    assert "PICK: Alpha" in out
    assert "prefers" not in out


def test_a_pick_outside_the_shortlist_still_resolves(capsys):
    """need_adp's legality rules differ (forced K/DEF late, positional limits),
    so its choice is not guaranteed to appear in the evaluated table."""
    draft_day.show_recommendations(
        FakePolicy(CANDIDATES[:2]), FakeRecommender(2), FakeSim(), team=0
    )
    out = capsys.readouterr().out

    assert "PICK: Charlie" in out          # resolved via board.names
    assert "outside the shortlist" in out


def test_an_empty_table_does_not_prevent_a_pick(capsys):
    draft_day.show_recommendations(
        FakePolicy([]), FakeRecommender(1), FakeSim(), team=0
    )
    out = capsys.readouterr().out

    assert "PICK: Bravo" in out


# --- the column that survived the diagnosis -------------------------------

def test_scarcity_advice_fires_when_our_pick_is_likely_to_last():
    """P(next) comes from ADP dispersion, not from our valuation, so the E2
    finding does not touch it. It stays actionable."""
    candidates = [
        _candidate(0, "Alpha", 1.70, p_next=0.10),
        _candidate(1, "Bravo", 1.68, p_next=0.95),
        _candidate(2, "Charlie", 1.65, p_next=0.90),
    ]
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        draft_day.show_recommendations(
            FakePolicy(candidates), FakeRecommender(1), FakeSim(), team=0
        )
    out = buffer.getvalue()

    assert "PICK: Bravo" in out
    assert "consider taking Alpha first" in out
