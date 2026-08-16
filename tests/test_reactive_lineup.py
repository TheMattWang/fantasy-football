"""In-season management, and the off-by-one that would silently fake it.

The simulator froze lineups all season: `decision_score` is the preseason
projection and never updates, so a player terrible for eight weeks still started
on his August number. There was no start/sit decision in the model at all, which
is why T2 could only measure perfect-vs-frozen (4.457 ranks) rather than
perfect-vs-realistic.

`_reactive_estimate` adds the decision. The danger in adding it is precise and
already has a precedent in this repo: if week w's ranking key can see week w's
own points, the model is picking lineups with hindsight, every roster inflates,
and every existing test still passes. That is exactly what the FLEX bug did.
These tests exist to make that failure impossible to introduce quietly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation.season import _reactive_estimate  # noqa: E402

MU = np.array([10.0, 20.0])


def realized(rows):
    """(1 sample, 2 players, n weeks) from a list-of-lists per player."""
    return np.asarray([rows], dtype=np.float32)


# --- the hindsight guard --------------------------------------------------

def test_week_zero_is_exactly_the_preseason_projection():
    """Nothing has happened yet, so the prior is all there is."""
    out = _reactive_estimate(realized([[5, 5, 5], [40, 40, 40]]), MU, 4.0)
    assert out[0, 0, 0] == pytest.approx(10.0)
    assert out[0, 1, 0] == pytest.approx(20.0)


def test_the_estimate_cannot_see_the_week_it_is_setting():
    """THE test. Change week 2 and everything from week 2 back must be identical.

    If this fails, lineups are being set with hindsight and every roster total
    in the project is inflated.
    """
    base = realized([[5, 5, 5, 5], [20, 20, 20, 20]])
    changed = base.copy()
    changed[0, 0, 2] = 99.0          # blow up week 2 only

    a = _reactive_estimate(base, MU, 4.0)
    b = _reactive_estimate(changed, MU, 4.0)

    np.testing.assert_allclose(a[:, :, :3], b[:, :, :3], rtol=1e-6)
    # ...and week 3, which legitimately sees week 2, must differ.
    assert not np.allclose(a[0, 0, 3], b[0, 0, 3])


def test_a_late_explosion_does_not_help_the_early_weeks():
    """A player who is terrible then erupts must be rated low while terrible."""
    out = _reactive_estimate(realized([[1, 1, 1, 1, 60], [20] * 5]), MU, 4.0)
    assert out[0, 0, 1] < 10.0            # already sliding after one bad week
    assert out[0, 0, 4] < 10.0            # the week 4 explosion is still unseen


# --- the shrinkage behaves ------------------------------------------------

def test_a_stubborn_prior_barely_reacts():
    weak = _reactive_estimate(realized([[0.1] * 8, [20] * 8]), MU, 1000.0)
    assert weak[0, 0, -1] == pytest.approx(10.0, abs=0.2)


def test_a_weak_prior_converges_on_what_actually_happened():
    quick = _reactive_estimate(realized([[2.0] * 12, [20] * 12]), MU, 0.5)
    assert quick[0, 0, -1] == pytest.approx(2.0, abs=0.5)


def test_the_estimate_moves_monotonically_toward_a_steady_signal():
    out = _reactive_estimate(realized([[2.0] * 10, [20] * 10]), MU, 4.0)[0, 0]
    assert np.all(np.diff(out) <= 1e-6), "a constant low signal must not rebound"
    assert out[0] > out[-1]


# --- availability is not performance --------------------------------------

def test_missed_weeks_do_not_count_as_games_played():
    """A zero is an inactive week, not a zero-point performance. Counting it
    would conflate injury with decline -- and availability is modelled
    separately, in layer 2."""
    injured = _reactive_estimate(realized([[12, 0, 0, 0], [20] * 4]), MU, 4.0)
    healthy = _reactive_estimate(realized([[12, 12, 12, 12], [20] * 4]), MU, 4.0)

    # After one good week then three missed, the estimate is frozen at the
    # post-week-1 value rather than collapsing toward zero.
    assert injured[0, 0, 1] == pytest.approx(injured[0, 0, 3])
    assert injured[0, 0, 3] > 10.0
    assert healthy[0, 0, 3] > injured[0, 0, 3]


# --- guards ----------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_prior_weight_is_refused(bad):
    with pytest.raises(ValueError, match="must be > 0"):
        _reactive_estimate(realized([[5, 5], [5, 5]]), MU, bad)


def test_shape_is_preserved():
    out = _reactive_estimate(np.zeros((7, 2, 14), dtype=np.float32), MU, 4.0)
    assert out.shape == (7, 2, 14)


# --- integration: it lands between frozen and omniscient ------------------

def test_reactive_beats_frozen_and_loses_to_hindsight():
    """The ordering that says it is doing real work without cheating.

    Frozen ignores the season; reactive learns from it; omniscient knows the
    future. Anything that beats omniscient is a bug.
    """
    from src.data.league_config import provisional_config
    from src.projections.board import build_board
    from src.simulation.distributions import build_samples
    from src.simulation.season import lineup_points, make_waiver_draws, plan_roster

    config = provisional_config()
    board = build_board(2026, config=config, validate=False, market_dispersion=False)
    samples = build_samples(board, n_samples=200, n_weeks=14, seed=0)

    roster = (board[~board["is_streamed"]]
              .sort_values("VORP", ascending=False)
              .head(15)["player_name"].tolist())
    plan = plan_roster(roster, samples, config)
    draws = make_waiver_draws(200, 14, rng=np.random.default_rng(0))

    def total(**kwargs):
        return float(lineup_points(samples, plan, waiver_draws=draws, **kwargs).mean())

    frozen = total()
    reactive = total(reactive_prior_games=4.0)
    perfect = total(omniscient=True)

    assert frozen < reactive < perfect, (
        f"frozen {frozen:.2f}, reactive {reactive:.2f}, perfect {perfect:.2f}"
    )
