"""Draft policies, from "take the top of ADP" to the season-simulation agent.

The agent is wide 1-ply over full-depth rollouts: for each plausible candidate,
finish the draft against the calibrated opponents, evaluate the resulting roster
with the season simulation, take the argmax. Full-depth rollouts are what
capture positional scarcity and "will he still be there next round" -- which is
most of the value a tree would give, at a fraction of the cost.

Deliberately not MCTS. The 2025 repo had three classes named MCTS and none of
them built a tree; the honest lesson is that search was never the bottleneck.
Build the tree only if this policy beats the ADP baseline first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from ..data.league_config import LeagueConfig
from ..simulation.distributions import SampleSet
from ..simulation.season import evaluate_roster, simulate_opponents
from .engine import DraftBoard, DraftSim
from .opponents import (
    TYPICAL_TARGETS,
    OpponentModel,
    assign_archetypes,
    draw_noise,
)


class Policy(Protocol):
    """Anything that can pick a player for a team."""

    name: str

    def choose(self, sim: DraftSim, team: int) -> int: ...


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class AdpPolicy:
    """B0: take the best consensus rank available, with no positional awareness.

    A sanity floor, NOT the bar. On a real board this drafts eight quarterbacks
    and four tight ends -- once the elite skill players are gone, the best
    remaining overall rank is almost always a QB, and nothing here stops it.
    No human drafts this way, so beating it proves nothing. The bar is
    :class:`NeedAdpPolicy`.
    """

    name = "adp"

    def __init__(self, board: DraftBoard):
        self.board = board

    def choose(self, sim: DraftSim, team: int) -> int:
        adp = np.where(sim.available, self.board.adp, np.inf)
        return int(np.argmin(adp))


class NeedAdpPolicy:
    """B1: best available by ADP, subject to roster needs and positional limits.

    **This is the bar.** It is what a competent human does -- follow consensus,
    but do not end up with eight quarterbacks and no kicker. If the agent cannot
    beat this on held-out seasons, draft off this on the day and ship the agent
    as an informational display only.
    """

    name = "need_adp"

    def __init__(self, board: DraftBoard, config: LeagueConfig):
        self.board = board
        self.config = config

    def choose(self, sim: DraftSim, team: int) -> int:
        counts = sim.position_counts(team)
        rounds_left = sim.n_rounds - sim.round_number + 1

        legal = sim.available.copy()
        for pos, limit in TYPICAL_TARGETS.items():
            if counts.get(pos, 0) >= limit:
                legal &= self.board.positions != pos
        for pos in ("K", "DEF", "DST"):
            if sim.round_number < sim.n_rounds - 1:
                legal &= self.board.positions != pos

        # Fill mandatory slots when time runs short.
        missing = [
            pos for pos in ("QB", "K", "DEF", "DST")
            if counts.get(pos, 0) < 1 and (self.board.positions == pos).any()
        ]
        if missing and rounds_left <= len(missing):
            forced = np.zeros_like(sim.available)
            for pos in missing:
                forced |= (self.board.positions == pos) & sim.available
            if forced.any():
                legal = forced

        if not legal.any():
            legal = sim.available.copy()
        adp = np.where(legal, self.board.adp, np.inf)
        return int(np.argmin(adp))


class VorpGreedyPolicy:
    """B3: take the highest VORP that fits, ignoring when he would be gone."""

    name = "vorp_greedy"

    def __init__(self, board: DraftBoard, config: LeagueConfig):
        self.board = board
        self.inner = NeedAdpPolicy(board, config)

    def choose(self, sim: DraftSim, team: int) -> int:
        counts = sim.position_counts(team)
        legal = sim.available.copy()
        for pos, limit in TYPICAL_TARGETS.items():
            if counts.get(pos, 0) >= limit:
                legal &= self.board.positions != pos
        if not legal.any():
            return self.inner.choose(sim, team)
        vorp = np.where(legal, self.board.vorp, -np.inf)
        return int(np.argmax(vorp))


POOL_WEIGHTS: Dict[str, float] = {
    "archetype": 0.45,
    "need_adp": 0.30,
    "vorp_greedy": 0.25,
}


class OpponentPool:
    """The other 11 seats, each drawn from a mix of strategies.

    A field of one strategy is a field with one exploit. Every seat being an
    ``OpponentModel`` meant every seat carried the same hand-written archetype
    biases -- ``zero_rb`` avoids RBs by a made-up 3.0 ADP ranks, and so on --
    so a learned agent would spend its capacity discovering *those constants*
    rather than learning to draft. That is the 2025 reward-hacking failure one
    level up.

    It is also a strength problem. Measured on 2022/2023, plain ADP-with-roster-
    limits finishes ~4.2 of 12 against a pure archetype field: the simulated
    league is one a competent human beats by two and a half ranks, which is not
    a league. Mixing in seats that actually draft well fixes both at once.

    Satisfies the same duck-type as ``OpponentModel`` (``choose``, ``rng``,
    ``noise``), so it drops into ``run_draft``, ``replay`` and the rollout.
    """

    name = "pool"

    def __init__(
        self,
        board: DraftBoard,
        config: LeagueConfig,
        *,
        our_team: int,
        n_teams: int = 12,
        rng: Optional[np.random.Generator] = None,
        weights: Optional[Dict[str, float]] = None,
        sigma_scale: float = 1.0,
    ):
        self.board = board
        self.rng = rng or np.random.default_rng(0)

        weights = dict(weights or POOL_WEIGHTS)
        kinds = sorted(weights)
        probs = np.asarray([weights[k] for k in kinds], dtype=float)
        probs = probs / probs.sum()

        self.model = OpponentModel(
            board,
            sigma_scale=sigma_scale,
            assignments=assign_archetypes(n_teams, our_team, self.rng),
            rng=self.rng,
        )
        self._policies = {
            "need_adp": NeedAdpPolicy(board, config),
            "vorp_greedy": VorpGreedyPolicy(board, config),
        }
        self.kind_of: Dict[int, str] = {
            team: (
                "us" if team == our_team
                else str(self.rng.choice(kinds, p=probs))
            )
            for team in range(n_teams)
        }

    @property
    def noise(self) -> Optional[np.ndarray]:
        return self.model.noise

    @noise.setter
    def noise(self, value: Optional[np.ndarray]) -> None:
        self.model.noise = value

    def composition(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for kind in self.kind_of.values():
            if kind != "us":
                counts[kind] = counts.get(kind, 0) + 1
        return counts

    def choose(self, sim: DraftSim, team: int) -> int:
        policy = self._policies.get(self.kind_of.get(team, "archetype"))
        if policy is not None:
            return policy.choose(sim, team)
        return self.model.choose(sim, team)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One evaluated option at the current pick."""

    row: int
    name: str
    position: str
    adp: float
    utility: float
    p_available_next: float

    def __str__(self) -> str:
        return (
            f"{self.name:<24} {self.position:<4} adp={self.adp:6.1f} "
            f"U={self.utility:.4f}  P(next)={self.p_available_next:.2f}"
        )


class SeasonSimPolicy:
    """Wide 1-ply over full-depth rollouts, scored by the season simulation."""

    name = "season_sim"

    def __init__(
        self,
        board: DraftBoard,
        config: LeagueConfig,
        samples: SampleSet,
        opponent_model: OpponentModel,
        opponent_weekly: Optional[np.ndarray] = None,
        *,
        n_candidates: int = 14,
        n_rollouts: int = 2,
        eval_samples: int = 300,
        confidence_margin: float = 0.06,
        compute_survival: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        self.board = board
        self.config = config
        self.samples = samples
        self.opponent_model = opponent_model
        self.opponent_weekly = opponent_weekly
        self.n_candidates = n_candidates
        self.n_rollouts = n_rollouts
        self.eval_samples = eval_samples
        self.confidence_margin = confidence_margin
        # P(available next turn) is display-only -- it never enters choose().
        # Skip it in bulk evaluation, where it costs ~160 opponent decisions
        # per pick for a number nobody reads.
        self.compute_survival = compute_survival
        self.rng = rng or np.random.default_rng(0)
        self.rollout_policy = NeedAdpPolicy(board, config)
        self.last_candidates: List[Candidate] = []

    def _shortlist(self, sim: DraftSim, team: int) -> np.ndarray:
        """Plausible picks: best available by ADP, filtered for legality."""
        counts = sim.position_counts(team)
        legal = sim.available.copy()
        for pos, limit in TYPICAL_TARGETS.items():
            if counts.get(pos, 0) >= limit:
                legal &= self.board.positions != pos
        # No sense evaluating a kicker in round 2.
        if sim.round_number < sim.n_rounds - 1:
            for pos in ("K", "DEF", "DST"):
                legal &= self.board.positions != pos
        if not legal.any():
            legal = sim.available.copy()

        order = np.argsort(np.where(legal, self.board.adp, np.inf))
        return order[: self.n_candidates]

    def _rollout(self, sim: DraftSim, team: int, first_pick: int,
                 noise: Optional[np.ndarray] = None) -> float:
        """Take ``first_pick``, finish the draft, score the roster.

        The opponents evaluated against are the ones this rollout actually
        produced. Scoring against a fixed cache would be faster, but the whole
        question -- "is my roster good enough to win this league" -- is relative
        to the teams the rest of the draft leaves behind.
        """
        clone = DraftSim(self.board, sim.n_teams, sim.n_rounds)
        clone.available = sim.available.copy()
        clone.rosters = [list(r) for r in sim.rosters]
        clone.picks = list(sim.picks)

        # Common random numbers. Every candidate is rolled out against the SAME
        # per-pick draws, so the difference between two candidates reflects the
        # pick and not which rollout got luckier. Without this, per-rollout noise
        # (sd ~0.06) dwarfs the gap between adjacent candidates (~0.02) and the
        # argmax is close to random -- which makes the agent strictly worse than
        # deterministically following a good consensus.
        #
        # This has to be a shared noise MATRIX, not a shared seed. Reseeding the
        # generator only lines the draws up until the board diverges; measured,
        # it left paired variance (0.086) higher than unpaired (0.072).
        saved_noise = self.opponent_model.noise
        self.opponent_model.noise = noise
        try:
            clone.make_pick(first_pick)
            while not clone.complete:
                current = clone.on_the_clock
                if current == team:
                    clone.make_pick(self.rollout_policy.choose(clone, current))
                else:
                    clone.make_pick(self.opponent_model.choose(clone, current))
        finally:
            self.opponent_model.noise = saved_noise

        opponent_weekly = simulate_opponents(
            [clone.roster_names(t) for t in range(clone.n_teams) if t != team],
            self.samples,
            self.config,
            n_samples=self.eval_samples,
        )

        result = evaluate_roster(
            clone.roster_names(team),
            self.samples,
            self.config,
            opponent_weekly,
            n_samples=self.eval_samples,
        )
        return result.utility

    def _survival(self, sim: DraftSim, team: int, rows: np.ndarray,
                  n_trials: int = 8) -> Dict[int, float]:
        """P(each candidate is still available at our next pick)."""
        # Horizon is the gap to our turn AFTER this one -- while we are on the
        # clock, picks_until_next_turn is 0 and everyone looks certain to last.
        horizon = sim.picks_until_turn_after_this(team)
        if horizon <= 0:
            return {int(r): 1.0 for r in rows}

        # We are on the clock, so the clone has to get past our own pick before
        # any opponent picks. Consume it with a deep bench player: that advances
        # the draft without removing any candidate, so every candidate is
        # measured under the same conditions. (It leaves out the one player we
        # will actually take, which nudges survival up by at most one pick.)
        available_rows = np.flatnonzero(sim.available)
        if available_rows.size == 0:
            return {int(r): 0.0 for r in rows}
        filler = int(available_rows[np.argmax(self.board.adp[available_rows])])

        survived = {int(r): 0 for r in rows}
        for trial in range(n_trials):
            clone = DraftSim(self.board, sim.n_teams, sim.n_rounds)
            clone.available = sim.available.copy()
            clone.rosters = [list(r) for r in sim.rosters]
            clone.picks = list(sim.picks)
            clone.make_pick(filler)

            for _ in range(horizon):
                if clone.complete or clone.on_the_clock == team:
                    break
                clone.make_pick(self.opponent_model.choose(clone, clone.on_the_clock))

            for r in rows:
                if clone.available[int(r)]:
                    survived[int(r)] += 1
        return {r: n / n_trials for r, n in survived.items()}

    def evaluate(self, sim: DraftSim, team: int) -> List[Candidate]:
        rows = self._shortlist(sim, team)
        survival = (
            self._survival(sim, team, rows)
            if self.compute_survival
            else {int(r): float("nan") for r in rows}
        )

        # One noise matrix per rollout index, shared by every candidate. Drawn
        # for the whole draft up front so pick k gets the same draw regardless
        # of how the board diverged getting there (see _rollout).
        n_picks = len(sim.order)
        noises = [
            draw_noise(n_picks, len(self.board), self.rng)
            for _ in range(self.n_rollouts)
        ]

        candidates: List[Candidate] = []
        for row in rows:
            row = int(row)
            utilities = [
                self._rollout(sim, team, row, noise=noise) for noise in noises
            ]
            candidates.append(
                Candidate(
                    row=row,
                    name=str(self.board.names[row]),
                    position=str(self.board.positions[row]),
                    adp=float(self.board.adp[row]),
                    utility=float(np.mean(utilities)),
                    p_available_next=survival.get(row, 0.0),
                )
            )
        candidates.sort(key=lambda c: (-c.utility, c.adp, c.name))
        self.last_candidates = candidates
        return candidates

    def choose(self, sim: DraftSim, team: int) -> int:
        """Take the simulation's pick only when it is confidently better.

        Measured on held-out 2025: per-rollout noise is sd ~0.06 while adjacent
        candidates differ by ~0.02. Near the top of the draft the objective
        genuinely *cannot* separate elite players -- and it is right not to,
        because which one you take barely moves the season. Taking the noisy
        argmax there is strictly worse than following a good consensus, and the
        replay showed exactly that (-0.74 ranks against the bar).

        So the policy shrinks toward the market, the same idea the board uses:
        deviate from consensus order only when the estimated gain clears the
        noise floor. ``confidence_margin`` is that floor.
        """
        return self.decide(self.evaluate(sim, team), sim, team)

    def decide(
        self, candidates: List[Candidate], sim: DraftSim, team: int
    ) -> int:
        """Apply the shrinkage rule to an already-evaluated candidate list.

        Separate from :meth:`choose` so a caller that wants to *display* the
        ranking can decide on exactly the numbers it showed. Re-evaluating would
        advance the RNG and pick against a different sample, which is both
        wasteful and visibly inconsistent.
        """
        if not candidates:
            return self.rollout_policy.choose(sim, team)

        default_row = self.rollout_policy.choose(sim, team)
        best = candidates[0]
        if best.row == default_row:
            return best.row

        default = next((c for c in candidates if c.row == default_row), None)
        if default is None:
            # Consensus pick was not even in the shortlist; trust the search.
            return best.row

        if best.utility - default.utility > self.confidence_margin:
            return best.row
        return default_row


# ---------------------------------------------------------------------------
# Running a whole draft
# ---------------------------------------------------------------------------

def run_draft(
    board: DraftBoard,
    policy: Policy,
    opponent_model: OpponentModel,
    *,
    our_team: int,
    n_teams: int = 12,
    n_rounds: int = 15,
) -> DraftSim:
    """Draft with ``policy`` in our seat and the opponent model everywhere else."""
    sim = DraftSim(board, n_teams, n_rounds)
    while not sim.complete:
        team = sim.on_the_clock
        if team == our_team:
            sim.make_pick(policy.choose(sim, team))
        else:
            sim.make_pick(opponent_model.choose(sim, team))
    return sim
