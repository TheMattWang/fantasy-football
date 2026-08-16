"""The objective: simulate the season and score a roster by how often it wins.

What this replaces
------------------
The 2025 objective was ``sum(vorp)`` over all 15 picks, which counts a bench WR5
exactly as much as a starter and never mentions winning. Fantasy is decided by
weekly head-to-head matchups using a *starting lineup*, so that is what gets
simulated here.

Three consequences fall out rather than being hand-tuned
--------------------------------------------------------
* **Bench value is naturally diminishing.** A bench player only ever scores when
  someone ahead of him is hurt or on bye, so his marginal contribution decays
  with depth instead of being counted at full weight.
* **Kickers and defenses stop being worth drafting.** They are streamed from the
  waiver pool every week, so spending a real pick on one adds nothing -- no
  ``early_k_penalty`` coefficient required.
* **Bye weeks stop needing a penalty term.** A bye is availability 0 for one
  week, and the waiver pool covers it. ``bye_penalty`` is deleted, not tuned.

Speed
-----
Opponent rosters are fixed while our roster changes during a search, so their
weekly totals are computed once and cached. Scoring one candidate roster is then
a handful of ``np.partition`` calls -- roughly 0.3 ms at 200 samples and 3 ms at
2000, which is what makes a wide search affordable inside a pick clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.league_config import FLEX_SLOT_ELIGIBILITY, LeagueConfig
from .distributions import STREAMED_PPG, SampleSet

# What a roster hole is worth. You always start somebody, but you do not always
# get the good somebody.
#
# The naive version gives every team the same replacement level whenever it has
# a hole. That is wrong in a way that matters: in a 12-team league there is ONE
# startable RB on waivers in a given week, and one team gets him. Handing the
# same floor to all twelve simultaneously makes injuries and byes far cheaper
# than they are, which over-values thin rosters and hides the value of depth.
#
# So a hole resolves to the good pickup with probability WAIVER_WIN_RATE and to
# a scrub otherwise.
DEFAULT_WAIVER_FLOOR: Dict[str, float] = {
    "QB": 12.0, "RB": 6.0, "WR": 6.5, "TE": 4.0, "K": 8.0, "DEF": 7.0, "DST": 7.0,
}

# What you get when someone else wins the claim.
DEEP_WAIVER_FLOOR: Dict[str, float] = {
    "QB": 7.0, "RB": 2.5, "WR": 3.0, "TE": 1.5, "K": 6.5, "DEF": 4.5, "DST": 4.5,
}

# Roughly one good pickup per position per week, shared among the teams that
# need one. Kickers and defenses are the exception -- there is always another
# streamable one, which is exactly why they are not worth drafting.
WAIVER_WIN_RATE: Dict[str, float] = {
    "QB": 0.55, "RB": 0.30, "WR": 0.35, "TE": 0.40, "K": 0.95, "DEF": 0.90,
    "DST": 0.90,
}


@dataclass
class RosterPlan:
    """A roster resolved into the slot structure, ready for fast evaluation."""

    rows: np.ndarray                      # board row per player
    positions: np.ndarray                 # position per player
    dedicated: Dict[str, np.ndarray]      # position -> rows eligible
    dedicated_counts: Dict[str, int]
    flex_counts: Dict[str, int]           # slot name -> count
    flex_eligibility: Dict[str, frozenset]


def plan_roster(
    names: Sequence[str], samples: SampleSet, config: LeagueConfig
) -> RosterPlan:
    """Group a roster by position once, so weekly evaluation is pure numpy."""
    rows, positions = [], []
    for name in names:
        row = samples.index.get(name)
        if row is None:
            continue
        rows.append(row)
        positions.append(str(samples.players["position"].iloc[row]))

    rows = np.asarray(rows, dtype=np.intp)
    positions = np.asarray(positions, dtype=object)

    dedicated_counts = {
        pos: n for pos, n in config.dedicated_slots.items() if n > 0
    }
    dedicated = {
        pos: rows[np.array([p == pos for p in positions], dtype=bool)]
        if len(rows)
        else np.empty(0, dtype=np.intp)
        for pos in dedicated_counts
    }
    flex_counts = {
        slot: config.starting_slots[slot] for slot in config.flex_slots
    }
    return RosterPlan(
        rows=rows,
        positions=positions,
        dedicated=dedicated,
        dedicated_counts=dedicated_counts,
        flex_counts=flex_counts,
        flex_eligibility=config.flex_slots,
    )


WAIVER_CHANNELS = 32

# Measurement hook. ``src/evaluation/robustness.py`` flips this to re-run a
# frozen agent under the OLD hindsight flex rule, to test whether its edge
# depended on that bug. Never set during normal operation.
FLEX_OMNISCIENT_DEFAULT = False


def make_waiver_draws(
    n_samples: int,
    n_weeks: int,
    *,
    rng: Optional[np.random.Generator] = None,
    n_channels: int = WAIVER_CHANNELS,
) -> np.ndarray:
    """Uniforms for contested waiver claims, shaped ``(samples, weeks, channels)``.

    Generate once and pass to every roster you want compared. Sharing the draws
    is a common-random-numbers scheme for the waiver wire: two rosters differ
    because of the players on them, not because one got luckier on the claims.
    """
    rng = rng or np.random.default_rng(0)
    return rng.random((n_samples, n_weeks, n_channels)).astype(np.float32)


def _reactive_estimate(
    realized: np.ndarray, prior_ppg: np.ndarray, prior_games: float
) -> np.ndarray:
    """A manager's running estimate of a scoring rate, week by week.

    ``(samples, players, weeks) -> (samples, players, weeks)``.

    Blends the preseason projection with what has actually happened, weighting
    the prior as if it were ``prior_games`` games::

        estimate[w] = (k * mu + points before w) / (k + games before w)

    At week 0 nothing has been observed and this collapses to ``mu``, which is
    exactly the frozen behaviour. As the season runs it migrates toward the
    observed rate, which is what a manager benching a bust is doing informally.
    ``k`` is how stubborn he is: large k barely reacts, small k chases noise.

    **The one-week shift is the whole point.** Week ``w`` may see weeks strictly
    before it and nothing else. Including week ``w`` would be the FLEX hindsight
    bug over again -- the same off-by-one, one level up -- and it would inflate
    every roster while every test still passed.
    """
    if prior_games <= 0:
        raise ValueError("reactive_prior_games must be > 0; at week 0 there is "
                         "nothing observed and the prior is all there is")

    active = realized > 0
    cum_points = np.cumsum(realized, axis=2, dtype=np.float64)
    cum_games = np.cumsum(active, axis=2, dtype=np.float64)

    pad = np.zeros(realized.shape[:2] + (1,), dtype=np.float64)
    seen_points = np.concatenate([pad, cum_points[:, :, :-1]], axis=2)
    seen_games = np.concatenate([pad, cum_games[:, :, :-1]], axis=2)

    mu = np.asarray(prior_ppg, dtype=np.float64)[None, :, None]
    k = float(prior_games)
    return ((k * mu + seen_points) / (k + seen_games)).astype(np.float32)


def lineup_points(
    samples: SampleSet,
    plan: RosterPlan,
    *,
    n_samples: Optional[int] = None,
    waiver_floor: Optional[Dict[str, float]] = None,
    omniscient: bool = False,
    flex_omniscient: Optional[bool] = None,
    contested_waivers: bool = True,
    waiver_draws: Optional[np.ndarray] = None,
    reactive_prior_games: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:  # noqa: D401 - see _reactive_estimate below
    """Weekly starting-lineup totals for one roster. Shape (samples, weeks).

    Starters are chosen by projected value among *available* players, then the
    realized points of whoever was started are summed. Setting ``omniscient``
    picks by realized points instead; it is only there to quantify how much
    hindsight inflates a roster.

    ``reactive_prior_games`` turns on in-season management. By default the
    ranking key is the *preseason* projection and never changes, so a player who
    has been terrible for eight weeks still starts on the strength of his August
    number -- there is no start/sit decision in the model at all. That is not a
    small simplification: T2 measured the gap between a frozen lineup and a
    perfect one at 4.457 ranks, against 0.832 for a realistic draft improvement.
    Passing a number here ranks instead by a running estimate that blends the
    preseason prior with what has actually happened, weighting the prior as if
    it were that many games. See :func:`_reactive_estimate`.

    ``flex_omniscient`` does the same for the FLEX slot alone. It exists because
    the flex used to be picked by realized points while the dedicated slots were
    picked ex ante -- worth ~5.4 pts/week, and worth *more* to a roster carrying
    more flex-eligible bench players, so it silently subsidized bench depth. Keep
    it reachable so the size of that subsidy stays measurable.
    """
    waiver_floor = waiver_floor or DEFAULT_WAIVER_FLOOR
    if flex_omniscient is None:
        flex_omniscient = FLEX_OMNISCIENT_DEFAULT
    points = samples.points if n_samples is None else samples.points[:n_samples]
    n_s, _, n_w = points.shape

    # Waiver draws are pre-generated rather than drawn inside fill(), because
    # drawing per slot made the result depend on the iteration order of
    # `roster_slots` -- a dict. Same reason the sorts are stable: the torch
    # backend has to reproduce this bit for bit, and it cannot if the answer
    # depends on how a dict happened to be ordered.
    if waiver_draws is None:
        waiver_draws = make_waiver_draws(n_s, n_w, rng=rng)

    # Each hole gets its own channel, so two empty RB slots are two independent
    # waiver claims rather than one claim counted twice.
    channel = [0]

    def fill(position: str) -> np.ndarray:
        """Value of a hole at ``position``, week by week and sample by sample.

        Contested: you win the claim sometimes and take a scrub otherwise.
        """
        good = float(waiver_floor.get(position, 0.0))
        if not contested_waivers:
            return np.full((n_s, n_w), good, dtype=np.float32)

        # Losing the claim can never be better than winning it. Clamping to
        # `good` also keeps a caller-supplied floor authoritative -- otherwise
        # the module default overrides an explicit 0.
        poor = min(float(DEEP_WAIVER_FLOOR.get(position, good * 0.4)), good)
        win = float(WAIVER_WIN_RATE.get(position, 0.35))
        draw = waiver_draws[:n_s, :n_w, channel[0] % waiver_draws.shape[2]]
        channel[0] += 1
        return np.where(draw < win, good, poor).astype(np.float32)

    total = np.zeros((n_s, n_w), dtype=np.float32)
    leftovers: List[Tuple[str, np.ndarray, np.ndarray]] = []

    # Sorted, so which waiver channel a slot consumes cannot depend on how the
    # league's roster_slots dict happened to be ordered.
    for pos, count in sorted(plan.dedicated_counts.items()):
        rows = plan.dedicated.get(pos, np.empty(0, dtype=np.intp))

        if rows.size == 0:
            # No player rostered at all: stream the slot every week.
            for _ in range(count):
                total += fill(pos)
            continue

        realized = points[:, rows, :]

        if omniscient:
            ranked_by = realized
        else:
            # Ex-ante: rank by preseason projection, but a player who is out
            # cannot be started, so drop him behind everyone available.
            projection = samples.decision_score[rows][None, :, None]
            if reactive_prior_games is not None:
                projection = _reactive_estimate(
                    realized, samples.decision_score[rows], reactive_prior_games
                )
            available = realized > 0
            ranked_by = np.where(available, projection, -1.0).astype(np.float32)

        order = np.argsort(-ranked_by, axis=1, kind="stable")
        started = np.take_along_axis(realized, order, axis=1)
        started_key = np.take_along_axis(ranked_by, order, axis=1)

        chosen = started[:, :count, :]

        # A started player who scored 0 was unavailable; the manager picks up a
        # replacement -- and only sometimes gets the good one.
        for slot_index in range(chosen.shape[1]):
            column = chosen[:, slot_index, :]
            total += np.where(column > 0, column, fill(pos))
        for _ in range(count - chosen.shape[1]):
            total += fill(pos)

        # Carry the ranking key with the points. The flex has to rank these by
        # the SAME ex-ante key -- ranking them by realized points is hindsight.
        if started[:, count:, :].size:
            leftovers.append((pos, started[:, count:, :], started_key[:, count:, :]))

    # FLEX: best remaining player from any eligible position, chosen the way a
    # manager actually chooses -- by projection, before the week is played.
    if leftovers:
        rest_pos = np.concatenate(
            [np.full(block.shape[1], pos, dtype=object) for pos, block, _ in leftovers]
        )
        rest_pts = np.concatenate([block for _, block, _ in leftovers], axis=1)
        rest_key = np.concatenate([key for _, _, key in leftovers], axis=1)
    else:
        rest_pos = np.empty(0, dtype=object)
        rest_pts = np.zeros((n_s, 0, n_w), dtype=np.float32)
        rest_key = np.zeros((n_s, 0, n_w), dtype=np.float32)

    # `live` is consumed as slots are filled, so a second flex slot draws from
    # what the first one left rather than starting the same player twice.
    live = (rest_pts if (omniscient or flex_omniscient) else rest_key).astype(
        np.float32
    ).copy()
    n_rest = rest_pts.shape[1]

    for slot, count in sorted(plan.flex_counts.items()):
        eligible = plan.flex_eligibility.get(slot, frozenset())
        weakest = min(sorted(eligible), key=lambda p: waiver_floor.get(p, 0.0))
        eligible_col = np.array([p in eligible for p in rest_pos], dtype=bool)

        take = min(count, n_rest)
        if take:
            masked = np.where(eligible_col[None, :, None], live, -np.inf)
            picked = np.argsort(-masked, axis=1, kind="stable")[:, :take, :]
            chosen = np.take_along_axis(rest_pts, picked, axis=1)
            # -inf means nobody eligible was left; that slot streams instead.
            usable = np.take_along_axis(masked, picked, axis=1) > -np.inf
            # Consume only what was actually started. A player passed over
            # because he is ineligible *here* must stay live for a flex slot
            # with different eligibility.
            current = np.take_along_axis(live, picked, axis=1)
            np.put_along_axis(
                live, picked, np.where(usable, -np.inf, current), axis=1
            )

            for slot_index in range(take):
                column = np.where(usable[:, slot_index, :], chosen[:, slot_index, :], 0.0)
                total += np.where(column > 0, column, fill(weakest))

        for _ in range(count - take):
            total += fill(weakest)

    return total


def round_robin_schedule(n_teams: int, n_weeks: int) -> np.ndarray:
    """Standard circle-method schedule. Returns (weeks, teams) -> opponent id."""
    teams = list(range(n_teams))
    schedule = np.zeros((n_weeks, n_teams), dtype=np.intp)

    rotating = teams[1:]
    for week in range(n_weeks):
        order = [teams[0]] + rotating
        half = n_teams // 2
        for i in range(half):
            a, b = order[i], order[n_teams - 1 - i]
            schedule[week, a] = b
            schedule[week, b] = a
        rotating = rotating[1:] + rotating[:1]
    return schedule


@dataclass
class SeasonResult:
    """Outcome distribution for one team across simulated seasons."""

    wins: np.ndarray            # (samples,)
    points_for: np.ndarray      # (samples,)
    rank: np.ndarray            # (samples,) 1 = best
    made_playoffs: np.ndarray   # (samples,) bool
    won_title: np.ndarray       # (samples,) bool

    @property
    def p_playoffs(self) -> float:
        return float(self.made_playoffs.mean())

    @property
    def p_title(self) -> float:
        return float(self.won_title.mean())

    @property
    def mean_rank(self) -> float:
        return float(self.rank.mean())

    @property
    def utility(self) -> float:
        """The scalar the search maximizes.

        ``P(playoffs)`` is low-variance and tracks "did the season go well";
        ``P(title)`` carries the risk-seeking signal but needs many samples to
        stabilize, so it is weighted but not relied upon alone.
        """
        return self.p_playoffs + 2.0 * self.p_title

    def summary(self) -> str:
        return (
            f"P(playoffs)={self.p_playoffs:.3f}  P(title)={self.p_title:.3f}  "
            f"E[rank]={self.mean_rank:.2f}  E[wins]={self.wins.mean():.1f}  "
            f"U={self.utility:.3f}"
        )


def simulate_league(
    team_weekly: np.ndarray,
    config: LeagueConfig,
    *,
    schedule: Optional[np.ndarray] = None,
    team_of_interest: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> SeasonResult:
    """Run the head-to-head season.

    Args:
        team_weekly: ``(teams, samples, weeks)`` starting-lineup totals.
        team_of_interest: which team's outcome to report.
    """
    n_teams, n_samples, n_weeks = team_weekly.shape
    schedule = (
        round_robin_schedule(n_teams, n_weeks) if schedule is None else schedule
    )

    # wins[t, s]: weeks team t outscored its opponent.
    wins = np.zeros((n_teams, n_samples), dtype=np.int32)
    for week in range(n_weeks):
        opponents = schedule[week]
        mine = team_weekly[:, :, week]
        theirs = team_weekly[opponents, :, week]
        wins += (mine > theirs).astype(np.int32)

    points_for = team_weekly.sum(axis=2)

    # Standings: wins first, points-for as the tiebreak (the near-universal rule).
    key = wins.astype(np.float64) * 1e6 + points_for
    order = np.argsort(-key, axis=0)
    rank = np.empty_like(order)
    rows = np.arange(n_teams)[:, None]
    np.put_along_axis(rank, order, np.broadcast_to(rows, order.shape), axis=0)
    rank = rank + 1

    n_playoff = int(config.num_playoff_teams or max(4, n_teams // 2))
    made = rank <= n_playoff

    # Championship: among playoff teams, sample a winner with probability
    # proportional to seed strength. A full bracket adds variance without
    # changing the ranking of draft decisions.
    rng = rng or np.random.default_rng(0)
    seed_weight = np.where(made, 1.0 / rank, 0.0)
    seed_weight /= seed_weight.sum(axis=0, keepdims=True)
    draws = rng.random(n_samples)
    cumulative = np.cumsum(seed_weight, axis=0)
    champion = (cumulative < draws[None, :]).sum(axis=0)

    t = team_of_interest
    return SeasonResult(
        wins=wins[t],
        points_for=points_for[t],
        rank=rank[t],
        made_playoffs=made[t],
        won_title=(champion == t),
    )


def evaluate_roster(
    roster: Sequence[str],
    samples: SampleSet,
    config: LeagueConfig,
    opponent_weekly: np.ndarray,
    *,
    n_samples: Optional[int] = None,
    schedule: Optional[np.ndarray] = None,
    waiver_draws: Optional[np.ndarray] = None,
    reactive_prior_games: Optional[float] = None,
) -> SeasonResult:
    """Score one candidate roster against fixed, pre-simulated opponents."""
    plan = plan_roster(roster, samples, config)
    mine = lineup_points(samples, plan, n_samples=n_samples,
                         waiver_draws=waiver_draws,
                         reactive_prior_games=reactive_prior_games)

    n_s = mine.shape[0]
    stacked = np.concatenate([mine[None, :, :], opponent_weekly[:, :n_s, :]], axis=0)
    return simulate_league(stacked, config, schedule=schedule, team_of_interest=0)


def simulate_opponents(
    rosters: Sequence[Sequence[str]],
    samples: SampleSet,
    config: LeagueConfig,
    *,
    n_samples: Optional[int] = None,
    waiver_draws: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weekly totals for every opponent roster. Computed once, reused per search."""
    return np.stack(
        [
            lineup_points(
                samples, plan_roster(r, samples, config),
                n_samples=n_samples, waiver_draws=waiver_draws,
            )
            for r in rosters
        ],
        axis=0,
    )
