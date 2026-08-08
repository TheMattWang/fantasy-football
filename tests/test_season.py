"""Tests for the season-simulation objective.

The lineup optimizer is the correctness-critical piece: every roster's value
flows through it. The rest are the sanity checks the plan calls for -- identical
rosters must be symmetric, a stacked roster must dominate, and a 15th-round
bench flier must be worth ~nothing (that last one is the bench-value fix that
`sum(vorp)` got wrong).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.league_config import from_dict  # noqa: E402
from src.simulation.distributions import SampleSet  # noqa: E402
from src.simulation.season import (  # noqa: E402
    evaluate_roster,
    lineup_points,
    make_waiver_draws,
    plan_roster,
    round_robin_schedule,
    simulate_league,
    simulate_opponents,
)

CONFIG = from_dict({
    "season": 2026, "league_id": "T", "name": "T", "num_teams": 12,
    "playoff_start_week": 15, "num_playoff_teams": 6,
    "start_week": 1, "end_week": 15,
    "roster_slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R/T": 1,
                     "K": 1, "DEF": 1, "BN": 6},
    "scoring": {},
})

NO_FLOOR = {p: 0.0 for p in ("QB", "RB", "WR", "TE", "K", "DEF", "DST")}


def make_samples(spec, n_samples=4, n_weeks=3):
    """Build a SampleSet from {name: (position, constant weekly points)}."""
    names = list(spec)
    points = np.zeros((n_samples, len(names), n_weeks), dtype=np.float32)
    for i, name in enumerate(names):
        points[:, i, :] = spec[name][1]
    players = pd.DataFrame({
        "player_name": names,
        "position": [spec[n][0] for n in names],
    })
    return SampleSet(
        points=points,
        players=players,
        index={n: i for i, n in enumerate(names)},
        active=points > 0,
        decision_score=np.array([spec[n][1] for n in names], dtype=np.float32),
    )


# --------------------------------------------------------------------------
# Lineup construction
# --------------------------------------------------------------------------

def test_starts_the_best_players_at_each_slot():
    samples = make_samples({
        "qb1": ("QB", 20), "rb1": ("RB", 15), "rb2": ("RB", 12), "rb3": ("RB", 3),
        "wr1": ("WR", 14), "wr2": ("WR", 10), "te1": ("TE", 8),
        "k1": ("K", 7), "d1": ("DEF", 6),
    })
    plan = plan_roster(list(samples.index), samples, CONFIG)
    total = lineup_points(samples, plan, waiver_floor=NO_FLOOR)
    # QB20 + RB15 + RB12 + WR14 + WR10 + TE8 + K7 + DEF6 + FLEX(best leftover=rb3 3)
    assert total[0, 0] == pytest.approx(95.0)


def test_flex_takes_the_best_leftover_across_positions():
    samples = make_samples({
        "qb1": ("QB", 0.1), "rb1": ("RB", 10), "rb2": ("RB", 10), "rb3": ("RB", 4),
        "wr1": ("WR", 10), "wr2": ("WR", 10), "wr3": ("WR", 9), "te1": ("TE", 5),
    })
    plan = plan_roster(list(samples.index), samples, CONFIG)
    total = lineup_points(samples, plan, waiver_floor=NO_FLOOR)
    # leftovers are rb3=4 and wr3=9; FLEX must take wr3.
    assert total[0, 0] == pytest.approx(0.1 + 10 + 10 + 10 + 10 + 5 + 9)


def make_varying_samples(spec, n_samples=4):
    """Build a SampleSet where projection and realized points DISAGREE.

    ``spec`` maps name -> (position, projection, [weekly points]). The whole
    class of flex bugs is invisible under constant weekly points, because then
    the ex-ante ranking and the realized ranking are the same ranking.
    """
    names = list(spec)
    n_weeks = len(next(iter(spec.values()))[2])
    points = np.zeros((n_samples, len(names), n_weeks), dtype=np.float32)
    for i, name in enumerate(names):
        points[:, i, :] = np.asarray(spec[name][2], dtype=np.float32)
    players = pd.DataFrame({
        "player_name": names,
        "position": [spec[n][0] for n in names],
    })
    return SampleSet(
        points=points,
        players=players,
        index={n: i for i, n in enumerate(names)},
        active=points > 0,
        decision_score=np.array([spec[n][1] for n in names], dtype=np.float32),
    )


# The dedicated slots below are all single-candidate, so only the flex is in
# question: two leftover WRs, and the one the market likes is the one that
# busts. A manager cannot know that in advance.
FLEX_SPEC = {
    "qb1": ("QB", 20, [20, 20]), "rb1": ("RB", 15, [15, 15]),
    "rb2": ("RB", 12, [12, 12]), "wr1": ("WR", 14, [14, 14]),
    "wr2": ("WR", 13, [13, 13]), "te1": ("TE", 8, [8, 8]),
    "k1": ("K", 7, [7, 7]), "d1": ("DEF", 6, [6, 6]),
    # Both project below wr1/wr2 so they land in the leftover pool, where the
    # flex has to choose between them -- and the projection order is the
    # reverse of the realized order.
    "wrA": ("WR", 9, [5, 5]),    # high projection, low realization
    "wrB": ("WR", 6, [20, 20]),  # low projection, high realization
}
FLEX_BASE = 20 + 15 + 12 + 14 + 13 + 8 + 7 + 6


def test_flex_is_set_ex_ante_not_with_hindsight():
    """The FLEX must start the projected-best leftover, not the realized-best.

    This is the bug that made every roster look ~6 pts/week better than it was,
    and made it look better in proportion to how many flex-eligible bench
    players it carried -- a silent subsidy for hoarding depth.
    """
    samples = make_varying_samples(FLEX_SPEC)
    plan = plan_roster(list(samples.index), samples, CONFIG)

    ex_ante = lineup_points(samples, plan, waiver_floor=NO_FLOOR)
    # wrA projects 12 > wrB's 6, so wrA starts and scores 5.
    assert ex_ante[0, 0] == pytest.approx(FLEX_BASE + 5)

    hindsight = lineup_points(
        samples, plan, waiver_floor=NO_FLOOR, flex_omniscient=True
    )
    # The old behaviour: pick wrB with knowledge of the outcome.
    assert hindsight[0, 0] == pytest.approx(FLEX_BASE + 20)


def test_flex_hindsight_is_strictly_an_upper_bound():
    """ex-ante <= flex-hindsight <= full hindsight, for every sample and week."""
    samples = make_varying_samples(FLEX_SPEC)
    plan = plan_roster(list(samples.index), samples, CONFIG)

    ex_ante = lineup_points(samples, plan, waiver_floor=NO_FLOOR)
    flex_only = lineup_points(
        samples, plan, waiver_floor=NO_FLOOR, flex_omniscient=True
    )
    full = lineup_points(samples, plan, waiver_floor=NO_FLOOR, omniscient=True)

    assert np.all(ex_ante <= flex_only + 1e-4)
    assert np.all(flex_only <= full + 1e-4)
    # And the middle term must not be vacuous -- if it equals ex-ante the test
    # fixture has stopped exercising the bug.
    assert flex_only.mean() - ex_ante.mean() > 1.0


def test_a_flex_eligible_bench_player_cannot_start_twice():
    """Two flex slots must consume distinct players, not start the same one."""
    two_flex = from_dict({
        "season": 2026, "league_id": "T", "name": "T", "num_teams": 12,
        "playoff_start_week": 15, "num_playoff_teams": 6,
        "start_week": 1, "end_week": 15,
        "roster_slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1,
                         "W/R/T": 1, "W/T": 1, "K": 1, "DEF": 1, "BN": 6},
        "scoring": {},
    })
    samples = make_varying_samples({
        "qb1": ("QB", 20, [20, 20]), "rb1": ("RB", 15, [15, 15]),
        "rb2": ("RB", 12, [12, 12]), "wr1": ("WR", 14, [14, 14]),
        "wr2": ("WR", 10, [10, 10]), "te1": ("TE", 8, [8, 8]),
        "k1": ("K", 7, [7, 7]), "d1": ("DEF", 6, [6, 6]),
        "wr3": ("WR", 9, [9, 9]),  # the only flex-eligible leftover
    })
    plan = plan_roster(list(samples.index), samples, two_flex)
    total = lineup_points(samples, plan, waiver_floor=NO_FLOOR)
    # wr3 fills the first flex; the second has nobody left and streams at 0.
    # If wr3 were started twice this would be 110.
    assert total[0, 0] == pytest.approx(20 + 15 + 12 + 14 + 10 + 8 + 7 + 6 + 9)


def test_bench_players_do_not_score():
    """A 6th WR on the bench must contribute nothing while the starters are healthy."""
    base = {
        "qb1": ("QB", 20), "rb1": ("RB", 15), "rb2": ("RB", 12),
        "wr1": ("WR", 14), "wr2": ("WR", 10), "wr3": ("WR", 9), "te1": ("TE", 8),
    }
    a = make_samples(base)
    b = make_samples({**base, "wr9": ("WR", 5)})

    ta = lineup_points(a, plan_roster(list(a.index), a, CONFIG), waiver_floor=NO_FLOOR)
    tb = lineup_points(b, plan_roster(list(b.index), b, CONFIG), waiver_floor=NO_FLOOR)
    assert ta[0, 0] == pytest.approx(tb[0, 0])


def test_injured_starter_is_replaced_by_the_waiver_floor():
    samples = make_samples({"qb1": ("QB", 20), "rb1": ("RB", 15)})
    samples.points[:, 0, 1] = 0.0  # QB out in week 2
    samples.active = samples.points > 0

    plan = plan_roster(list(samples.index), samples, CONFIG)
    total = lineup_points(samples, plan, waiver_floor={"QB": 11.0, "RB": 0.0,
                                                      "WR": 0.0, "TE": 0.0,
                                                      "K": 0.0, "DEF": 0.0},
                          contested_waivers=False)
    assert total[0, 0] == pytest.approx(35.0)   # 20 + 15
    assert total[0, 1] == pytest.approx(26.0)   # 11 (streamed) + 15


def test_empty_slot_is_streamed_every_week():
    """Drafting no kicker costs the difference to a streamer, not the whole slot."""
    samples = make_samples({"rb1": ("RB", 10)})
    plan = plan_roster(["rb1"], samples, CONFIG)
    total = lineup_points(samples, plan, waiver_floor={"K": 8.0, "QB": 0.0,
                                                      "RB": 0.0, "WR": 0.0,
                                                      "TE": 0.0, "DEF": 0.0},
                          contested_waivers=False)
    assert total[0, 0] == pytest.approx(18.0)  # rb1 + streamed K


def test_contested_waivers_make_a_hole_cost_more():
    """You do not always win the claim -- one team gets the good pickup.

    Handing every team the same replacement level makes injuries and byes far
    cheaper than they are, which hides the value of bench depth.
    """
    samples = make_samples({"qb1": ("QB", 20), "rb1": ("RB", 15)}, n_samples=400)
    samples.points[:, 0, 1] = 0.0          # QB out in week 2
    samples.active = samples.points > 0
    plan = plan_roster(list(samples.index), samples, CONFIG)

    floor = {"QB": 11.0, "RB": 0.0, "WR": 0.0, "TE": 0.0, "K": 0.0, "DEF": 0.0}
    naive = lineup_points(samples, plan, waiver_floor=floor,
                          contested_waivers=False)[:, 1].mean()
    real = lineup_points(samples, plan, waiver_floor=floor,
                         contested_waivers=True,
                         rng=np.random.default_rng(0))[:, 1].mean()
    assert real < naive


def test_lineups_are_set_ex_ante_not_with_hindsight():
    """Starting whoever happened to score most inflates every roster."""
    rng = np.random.default_rng(0)
    n_s, n_w = 400, 5

    # Five RBs for two RB slots, so there is an actual start/sit decision every
    # week. With as many players as slots, hindsight cannot help and the two
    # modes are trivially identical.
    names = ["rb1", "rb2", "rb3", "rb4", "rb5", "qb1", "te1"]
    positions = ["RB"] * 5 + ["QB", "TE"]
    means = [11.0, 10.5, 10.0, 9.5, 9.0, 8.0, 8.0]

    points = np.zeros((n_s, len(names), n_w), dtype=np.float32)
    for i, mean in enumerate(means):
        points[:, i, :] = rng.gamma(2.0, mean / 2.0, (n_s, n_w))

    samples = SampleSet(
        points=points,
        players=pd.DataFrame({"player_name": names, "position": positions}),
        index={n: i for i, n in enumerate(names)},
        active=points > 0,
        decision_score=np.array(means, dtype=np.float32),
    )
    plan = plan_roster(names, samples, CONFIG)
    ex_ante = lineup_points(samples, plan, waiver_floor=NO_FLOOR).mean()
    hindsight = lineup_points(samples, plan, waiver_floor=NO_FLOOR,
                              omniscient=True).mean()
    assert hindsight > ex_ante


# --------------------------------------------------------------------------
# Schedule and standings
# --------------------------------------------------------------------------

def test_round_robin_is_a_valid_pairing():
    schedule = round_robin_schedule(12, 13)
    for week in range(13):
        opponents = schedule[week]
        assert not any(opponents[t] == t for t in range(12))       # nobody self-plays
        assert all(opponents[opponents[t]] == t for t in range(12))  # symmetric


def test_identical_teams_have_symmetric_playoff_odds():
    """The plan's first sanity check: 12 identical rosters -> base rate each."""
    rng = np.random.default_rng(1)
    weekly = rng.gamma(4.0, 25.0, (12, 3000, 13)).astype(np.float32)

    rates = [
        simulate_league(weekly, CONFIG, team_of_interest=t).p_playoffs
        for t in range(12)
    ]
    assert np.mean(rates) == pytest.approx(0.5, abs=0.03)   # 6 of 12
    assert max(rates) - min(rates) < 0.06


def test_a_stronger_team_dominates():
    """The plan's second sanity check."""
    rng = np.random.default_rng(2)
    weekly = rng.gamma(4.0, 25.0, (12, 2000, 13)).astype(np.float32)
    weekly[0] *= 1.30

    strong = simulate_league(weekly, CONFIG, team_of_interest=0)
    other = simulate_league(weekly, CONFIG, team_of_interest=1)
    assert strong.p_playoffs > 0.85
    assert strong.p_playoffs > other.p_playoffs
    assert strong.mean_rank < other.mean_rank


def test_ranks_are_a_permutation():
    rng = np.random.default_rng(3)
    weekly = rng.gamma(4.0, 25.0, (12, 200, 13)).astype(np.float32)
    ranks = np.stack([
        simulate_league(weekly, CONFIG, team_of_interest=t).rank for t in range(12)
    ])
    for s in range(200):
        assert sorted(ranks[:, s]) == list(range(1, 13))


# --------------------------------------------------------------------------
# The bench-value property that sum(vorp) got wrong
# --------------------------------------------------------------------------

def _full_roster(samples):
    return [n for n in samples.index]


def test_a_late_bench_flier_is_worth_almost_nothing():
    """The plan's third sanity check, and the whole argument for this objective."""
    rng = np.random.default_rng(4)
    n_s, n_w = 1500, 13

    base = {
        "qb1": ("QB", 0), "rb1": ("RB", 0), "rb2": ("RB", 0), "rb3": ("RB", 0),
        "wr1": ("WR", 0), "wr2": ("WR", 0), "wr3": ("WR", 0), "te1": ("TE", 0),
        "k1": ("K", 0), "d1": ("DEF", 0),
    }
    means = {"qb1": 20, "rb1": 15, "rb2": 13, "rb3": 9, "wr1": 14,
             "wr2": 12, "wr3": 10, "te1": 9, "k1": 8, "d1": 7}

    def build(extra=None):
        spec = dict(base)
        if extra:
            spec[extra[0]] = (extra[1], 0)
            means[extra[0]] = extra[2]
        names = list(spec)
        pts = np.zeros((n_s, len(names), n_w), dtype=np.float32)
        for i, n in enumerate(names):
            pts[:, i, :] = rng.gamma(3.0, means[n] / 3.0, (n_s, n_w))
        return SampleSet(
            points=pts,
            players=pd.DataFrame({"player_name": names,
                                  "position": [spec[n][0] for n in names]}),
            index={n: i for i, n in enumerate(names)},
            active=pts > 0,
            decision_score=np.array([means[n] for n in names], dtype=np.float32),
        )

    starters = build()
    plus_flier = build(("wr_deep", "WR", 5.0))

    opponents = np.stack([
        rng.gamma(4.0, 24.0, (n_s, n_w)).astype(np.float32) for _ in range(11)
    ])

    a = evaluate_roster(_full_roster(starters), starters, CONFIG, opponents)
    b = evaluate_roster(_full_roster(plus_flier), plus_flier, CONFIG, opponents)

    # A WR6 projected at 5 ppg behind three healthy WRs barely moves the season.
    assert abs(b.utility - a.utility) < 0.05


def test_upgrading_a_starter_moves_utility_a_lot_more_than_a_bench_add():
    rng = np.random.default_rng(5)
    n_s, n_w = 1500, 13

    def build(rb1_mean, extra_bench=False):
        means = {"qb1": 20, "rb1": rb1_mean, "rb2": 12, "wr1": 13, "wr2": 11,
                 "te1": 8, "k1": 8, "d1": 7}
        pos = {"qb1": "QB", "rb1": "RB", "rb2": "RB", "wr1": "WR", "wr2": "WR",
               "te1": "TE", "k1": "K", "d1": "DEF"}
        if extra_bench:
            means["wr_deep"], pos["wr_deep"] = 5, "WR"
        names = list(means)
        pts = np.zeros((n_s, len(names), n_w), dtype=np.float32)
        for i, n in enumerate(names):
            pts[:, i, :] = rng.gamma(3.0, means[n] / 3.0, (n_s, n_w))
        return SampleSet(
            points=pts,
            players=pd.DataFrame({"player_name": names,
                                  "position": [pos[n] for n in names]}),
            index={n: i for i, n in enumerate(names)},
            active=pts > 0,
            decision_score=np.array([means[n] for n in names], dtype=np.float32),
        )

    opponents = np.stack([
        rng.gamma(4.0, 24.0, (n_s, n_w)).astype(np.float32) for _ in range(11)
    ])

    baseline = build(12)
    upgraded = build(18)
    bench = build(12, extra_bench=True)

    u0 = evaluate_roster(list(baseline.index), baseline, CONFIG, opponents).utility
    u_up = evaluate_roster(list(upgraded.index), upgraded, CONFIG, opponents).utility
    u_bench = evaluate_roster(list(bench.index), bench, CONFIG, opponents).utility

    assert (u_up - u0) > 2 * abs(u_bench - u0)


def test_simulate_opponents_shape():
    samples = make_samples({"qb1": ("QB", 10), "rb1": ("RB", 10)}, n_samples=5,
                           n_weeks=4)
    out = simulate_opponents([["qb1"], ["rb1"], ["qb1", "rb1"]], samples, CONFIG)
    assert out.shape == (3, 5, 4)


# --------------------------------------------------------------------------
# Reproducibility (the torch backend has to match this bit for bit)
# --------------------------------------------------------------------------

def test_shared_waiver_draws_make_the_result_deterministic():
    samples = make_varying_samples(FLEX_SPEC)
    plan = plan_roster(list(samples.index), samples, CONFIG)
    draws = make_waiver_draws(4, 2, rng=np.random.default_rng(3))

    a = lineup_points(samples, plan, waiver_draws=draws)
    b = lineup_points(samples, plan, waiver_draws=draws)
    assert np.array_equal(a, b)


def test_slot_ordering_does_not_change_the_result():
    """Waiver channels are assigned in sorted order, not dict order.

    Otherwise the same league spelled with its slots in a different order
    scores differently, and no torch port could ever reproduce it.
    """
    forward = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R/T": 1,
               "K": 1, "DEF": 1, "BN": 6}
    reverse = {k: forward[k] for k in reversed(list(forward))}

    samples = make_varying_samples(FLEX_SPEC)
    draws = make_waiver_draws(4, 2, rng=np.random.default_rng(3))

    totals = []
    for slots in (forward, reverse):
        config = from_dict({
            "season": 2026, "league_id": "T", "name": "T", "num_teams": 12,
            "playoff_start_week": 15, "num_playoff_teams": 6,
            "start_week": 1, "end_week": 15,
            "roster_slots": slots, "scoring": {},
        })
        plan = plan_roster(list(samples.index), samples, config)
        totals.append(lineup_points(samples, plan, waiver_draws=draws))

    assert np.array_equal(totals[0], totals[1])


def test_two_holes_at_one_position_are_independent_claims():
    """A roster with no RBs streams two RBs, winning each claim separately.

    Reusing one draw for both would make them win or lose in lockstep, which
    understates how often a manager ends up with exactly one usable starter.
    """
    samples = make_varying_samples({
        "qb1": ("QB", 20, [20, 20]), "wr1": ("WR", 14, [14, 14]),
        "wr2": ("WR", 13, [13, 13]), "te1": ("TE", 8, [8, 8]),
        "k1": ("K", 7, [7, 7]), "d1": ("DEF", 6, [6, 6]),
    }, n_samples=400)
    plan = plan_roster(list(samples.index), samples, CONFIG)
    draws = make_waiver_draws(400, 2, rng=np.random.default_rng(0))

    total = lineup_points(samples, plan, waiver_draws=draws)
    # Both RB slots are holes. With independent claims the combined RB fill
    # takes three distinct values (lose both / split / win both); perfectly
    # correlated draws would only ever produce two.
    distinct = np.unique(np.round(total, 4))
    assert len(distinct) >= 3
