"""Opponents that draft like people.

What the 2025 model did
-----------------------
``weights = 1 / (adp_rank + 1)`` -- no roster needs, no positional limits, no
runs. Kickers were freely available in round 3, elite players fell far more
often than they ever will, and consequently *every* strategy looked good in
backtest. Waiting on a position was always free.

What this does instead
----------------------
Each manager samples a private value for every available player::

    score = -(adp + Normal(0, k * adp_sd)) + need + archetype + run

and takes the best. ``adp_sd`` is FantasyPros' published dispersion of expert
opinion, so the noise is *measured*, not invented, and ``k`` is calibrated
against a real draft.

Position runs emerge rather than being scripted: once two managers in a row take
a running back, the need term shifts for everyone still short at the position,
which makes the third pick more likely too. The explicit ``run`` bonus only
sharpens an effect the need term already produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .engine import DraftBoard, DraftSim

# Hard caps. A manager will not draft a third quarterback or a second kicker,
# and nobody takes a kicker in round 3.
POSITION_LIMITS: Dict[str, int] = {
    "QB": 3, "RB": 8, "WR": 8, "TE": 3, "K": 1, "DEF": 1, "DST": 1,
}
EARLIEST_ROUND: Dict[str, int] = {"K": 12, "DEF": 11, "DST": 11}

# Roster targets used to compute "need". Not the league's slots exactly --
# people draft depth at RB/WR well beyond their starting requirement.
TYPICAL_TARGETS: Dict[str, int] = {
    "QB": 1, "RB": 5, "WR": 5, "TE": 1, "K": 1, "DEF": 1, "DST": 1,
}


@dataclass
class Archetype:
    """A recognisable drafting personality, as ADP-rank adjustments."""

    name: str
    position_bias: Dict[str, float] = field(default_factory=dict)
    early_bias: Dict[str, float] = field(default_factory=dict)
    reach: float = 0.0          # extra ADP noise
    need_weight: float = 1.0

    def bonus(self, position: str, rnd: int) -> float:
        value = self.position_bias.get(position, 0.0)
        if rnd <= 4:
            value += self.early_bias.get(position, 0.0)
        return value


ARCHETYPES: Dict[str, Archetype] = {
    "adp_follower": Archetype("adp_follower"),
    "need_filler": Archetype("need_filler", need_weight=2.0),
    # Biases are in "units of bonus_scale", not raw ADP ranks.
    "zero_rb": Archetype("zero_rb", early_bias={"RB": -3.0, "WR": 1.2}),
    "hero_rb": Archetype("hero_rb", early_bias={"RB": 1.4}),
    "qb_early": Archetype("qb_early", early_bias={"QB": 3.5}),
    "te_premium": Archetype("te_premium", early_bias={"TE": 2.0}),
    "reacher": Archetype("reacher", reach=0.9),
}


class OpponentModel:
    """Samples picks for the eleven managers who are not us."""

    @staticmethod
    def bonus_scale(pick_number: int) -> float:
        """How many ADP ranks one unit of preference is worth at this pick.

        Bonuses cannot be expressed in raw ranks. Reaching 14 spots at pick 3
        means taking the WR8 first overall; at pick 150 it is barely a
        preference. Left flat, the need and run terms swamp the ADP noise at the
        top of the board and every round-1 collapses into a single position.

        Scaling with pick number keeps "how much would this manager reach"
        roughly constant in *value* terms rather than in rank terms.
        """
        return 0.12 * (pick_number + 8)

    def __init__(
        self,
        board: DraftBoard,
        *,
        sigma_scale: float = 1.0,
        need_bonus: float = 1.6,
        run_bonus: float = 0.8,
        run_window: int = 4,
        assignments: Optional[Dict[int, str]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.board = board
        self.sigma_scale = float(sigma_scale)
        self.need_bonus = float(need_bonus)
        self.run_bonus = float(run_bonus)
        self.run_window = int(run_window)
        self.assignments = assignments or {}
        self.rng = rng or np.random.default_rng(0)
        self.noise: Optional[np.ndarray] = None

    def archetype_for(self, team: int) -> Archetype:
        return ARCHETYPES.get(self.assignments.get(team, "adp_follower"),
                              ARCHETYPES["adp_follower"])

    def _legal(self, sim: DraftSim, team: int) -> np.ndarray:
        """Mask of players this manager could actually take right now."""
        legal = sim.available.copy()
        counts = sim.position_counts(team)
        rnd = sim.round_number

        for pos, limit in POSITION_LIMITS.items():
            if counts.get(pos, 0) >= limit:
                legal &= self.board.positions != pos

        for pos, earliest in EARLIEST_ROUND.items():
            if rnd < earliest:
                legal &= self.board.positions != pos

        # Late in the draft you must actually fill your mandatory slots.
        rounds_left = sim.n_rounds - rnd + 1
        missing = [
            pos for pos in ("QB", "K", "DEF", "DST")
            if pos in TYPICAL_TARGETS
            and counts.get(pos, 0) < TYPICAL_TARGETS[pos]
            and (self.board.positions == pos).any()
        ]
        # Only force the issue once there is barely time left.
        if missing and rounds_left <= len(missing):
            forced = np.zeros_like(legal)
            for pos in missing:
                forced |= self.board.positions == pos
            if (legal & forced).any():
                legal &= forced
        return legal

    def _need(self, sim: DraftSim, team: int) -> np.ndarray:
        counts = sim.position_counts(team)
        need = np.zeros(len(self.board))
        for pos, target in TYPICAL_TARGETS.items():
            short = max(0, target - counts.get(pos, 0))
            if short:
                need[self.board.positions == pos] = min(short, 2)
        return need

    def _run(self, sim: DraftSim) -> np.ndarray:
        """Bonus for a position several managers just took."""
        bonus = np.zeros(len(self.board))
        recent = sim.picks[-self.run_window:]
        if len(recent) < 2:
            return bonus
        positions = [self.board.positions[r] for r in recent]
        for pos in set(positions):
            n = positions.count(pos)
            if n >= 2:
                bonus[self.board.positions == pos] = self.run_bonus * (n - 1)
        return bonus

    def choose(self, sim: DraftSim, team: int) -> int:
        """Row index of this manager's pick."""
        legal = self._legal(sim, team)
        if not legal.any():
            legal = sim.available.copy()

        archetype = self.archetype_for(team)
        rnd = sim.round_number
        scale = self.bonus_scale(sim.pick_number)

        sigma = np.maximum(
            self.board.adp_sd * self.sigma_scale + archetype.reach * scale, 0.5
        )
        # With a shared noise matrix the SAME draw is applied at pick k no
        # matter which candidate we are evaluating, which is what makes the
        # comparison paired. Reseeding the generator per rollout does not do
        # this -- once the board diverges the two draws stop lining up, and
        # measured paired variance came out *higher* than unpaired.
        if self.noise is not None and sim.pick_number < self.noise.shape[0]:
            eps = self.noise[sim.pick_number]
        else:
            eps = self.rng.standard_normal(len(self.board))
        noisy_adp = self.board.adp + eps * sigma

        score = -noisy_adp
        score += self.need_bonus * archetype.need_weight * scale * self._need(sim, team)
        score += self._run(sim) * scale
        for pos in set(self.board.positions):
            bonus = archetype.bonus(str(pos), rnd)
            if bonus:
                score[self.board.positions == pos] += bonus * scale

        score[~legal] = -np.inf
        return int(np.argmax(score))


def draw_noise(
    n_picks: int, n_players: int, rng: np.random.Generator
) -> np.ndarray:
    """Standard normals for one whole draft, indexed ``[pick, player]``.

    Share one of these across every candidate in a search step and the
    opponents behave identically except where our own pick changed the board.
    """
    return rng.standard_normal((n_picks, n_players))


def assign_archetypes(
    n_teams: int,
    our_team: int,
    rng: Optional[np.random.Generator] = None,
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[int, str]:
    """Give each opponent a personality.

    Defaults lean heavily on ``adp_follower`` because most managers mostly
    follow ADP; the tail is what creates the runs and reaches that make a draft
    feel real.
    """
    rng = rng or np.random.default_rng(0)
    weights = weights or {
        "adp_follower": 0.40, "need_filler": 0.20, "zero_rb": 0.10,
        "hero_rb": 0.10, "qb_early": 0.08, "te_premium": 0.06, "reacher": 0.06,
    }
    names = list(weights)
    probabilities = np.array([weights[n] for n in names], dtype=float)
    probabilities /= probabilities.sum()

    return {
        team: str(rng.choice(names, p=probabilities))
        for team in range(n_teams)
        if team != our_team
    }


def calibrate_sigma_scale(
    board: DraftBoard,
    actual_draft: pd.DataFrame,
    *,
    candidates: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
    n_teams: int = 12,
    n_rounds: int = 15,
    n_sims: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit the ADP-noise multiplier against a real draft.

    ``actual_draft`` needs ``player_name`` and ``overall_pick``. Scores each
    candidate by mean absolute error between simulated and observed pick
    position for the players who actually went.
    """
    lookup = {name: i for i, name in enumerate(board.names)}
    observed = {
        lookup[row.player_name]: float(row.overall_pick)
        for row in actual_draft.itertuples()
        if getattr(row, "player_name", None) in lookup
    }
    if not observed:
        raise ValueError("no players in the actual draft matched the board")

    rows = []
    for scale in candidates:
        positions: Dict[int, List[float]] = {row: [] for row in observed}
        for sim_index in range(n_sims):
            rng = np.random.default_rng(seed + sim_index)
            model = OpponentModel(
                board,
                sigma_scale=scale,
                assignments=assign_archetypes(n_teams, our_team=-1, rng=rng),
                rng=rng,
            )
            sim = DraftSim(board, n_teams, n_rounds)
            while not sim.complete:
                sim.make_pick(model.choose(sim, sim.on_the_clock))
            for pick, row in enumerate(sim.picks):
                if row in positions:
                    positions[row].append(pick + 1.0)

        errors = [
            abs(np.mean(values) - observed[row])
            for row, values in positions.items()
            if values
        ]
        rows.append(
            {
                "sigma_scale": scale,
                "mae_pick_position": float(np.mean(errors)) if errors else np.nan,
                "n_players": len(errors),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_pick_position").reset_index(drop=True)
