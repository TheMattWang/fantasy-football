"""Snake-draft mechanics, as flat integer arrays.

Deliberately not the old object model. The Phase 6 replay runs on the order of
ten thousand drafts, each ~180 picks, so players are row indices into numpy
arrays and a roster is a list of ints. It is also what removes the 2025 failure
at the root: with no ``set`` of ``Player`` objects hashed on name, there is no
hash-order nondeterminism left to expose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def snake_order(n_teams: int, n_rounds: int) -> np.ndarray:
    """Team id picking at each overall pick, snaking on even rounds."""
    order = []
    for rnd in range(n_rounds):
        teams = range(n_teams) if rnd % 2 == 0 else range(n_teams - 1, -1, -1)
        order.extend(teams)
    return np.asarray(order, dtype=np.intp)


@dataclass
class DraftBoard:
    """Immutable per-player arrays shared by every simulated draft."""

    names: np.ndarray
    positions: np.ndarray
    adp: np.ndarray
    adp_sd: np.ndarray
    vorp: np.ndarray

    @classmethod
    def from_frame(
        cls,
        board: pd.DataFrame,
        *,
        default_sd_fraction: float = 0.35,
        min_sd: float = 1.5,
    ) -> "DraftBoard":
        """Build from a projections board.

        The opponent model needs the dispersion of *draft position* -- how far
        from consensus a player actually goes. ``adp_sd``, when present, is that
        quantity measured over real drafts (see ``src/data/ffc_adp.py``).

        ``ecr_sd`` is only a fallback, and a biased one: it is the dispersion of
        *expert opinion*, and measured against real 2025 drafts it runs about
        twice the true spread in every ADP bucket. Using it makes simulated
        drafters twice as erratic as real ones, so elite players slide and
        waiting on them looks free -- the exact defect that made every strategy
        look good in 2025.
        """
        frame = board.reset_index(drop=True)
        adp = pd.to_numeric(frame["adp_rank"], errors="coerce")
        adp = adp.fillna(adp.max() if adp.notna().any() else 999.0).to_numpy(float)

        if "adp_sd" in frame.columns:
            sd = pd.to_numeric(frame["adp_sd"], errors="coerce").to_numpy(float)
        elif "ecr_sd" in frame.columns:
            sd = pd.to_numeric(frame["ecr_sd"], errors="coerce").to_numpy(float)
        else:
            sd = np.full(len(frame), np.nan)
        fallback = np.maximum(adp * default_sd_fraction, min_sd)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, fallback)

        return cls(
            names=frame["player_name"].astype(str).to_numpy(),
            positions=frame["position"].astype(str).to_numpy(),
            adp=adp,
            adp_sd=np.maximum(sd, min_sd),
            vorp=pd.to_numeric(frame.get("VORP", 0.0), errors="coerce")
            .fillna(0.0)
            .to_numpy(float),
        )

    def __len__(self) -> int:
        return len(self.names)


@dataclass
class DraftSim:
    """One simulated draft."""

    board: DraftBoard
    n_teams: int
    n_rounds: int
    available: np.ndarray = field(init=False)
    rosters: List[List[int]] = field(init=False)
    picks: List[int] = field(init=False)
    order: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.available = np.ones(len(self.board), dtype=bool)
        self.rosters = [[] for _ in range(self.n_teams)]
        self.picks = []
        self.order = snake_order(self.n_teams, self.n_rounds)

    @property
    def pick_number(self) -> int:
        return len(self.picks)

    @property
    def round_number(self) -> int:
        return self.pick_number // self.n_teams + 1

    @property
    def on_the_clock(self) -> int:
        return int(self.order[self.pick_number])

    @property
    def complete(self) -> bool:
        return self.pick_number >= len(self.order) or not self.available.any()

    def position_counts(self, team: int) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in self.rosters[team]:
            pos = self.board.positions[row]
            counts[pos] = counts.get(pos, 0) + 1
        return counts

    def make_pick(self, row: int) -> None:
        if not self.available[row]:
            raise ValueError(f"{self.board.names[row]} already drafted")
        team = self.on_the_clock
        self.available[row] = False
        self.rosters[team].append(int(row))
        self.picks.append(int(row))

    def picks_until_next_turn(self, team: int) -> int:
        """Picks before ``team`` is up, counting from the current pick.

        Returns 0 when ``team`` is on the clock right now.
        """
        for ahead, pick in enumerate(range(self.pick_number, len(self.order))):
            if self.order[pick] == team:
                return ahead
        return len(self.order) - self.pick_number

    def picks_until_turn_after_this(self, team: int) -> int:
        """Picks between this one and ``team``'s *following* turn.

        This is the horizon that decides whether to reach: "if I take him now
        versus wait, how many players come off the board first". Using
        :meth:`picks_until_next_turn` here returns 0 while we are on the clock,
        which makes every player look certain to survive.
        """
        start = self.pick_number + 1
        for ahead, pick in enumerate(range(start, len(self.order))):
            if self.order[pick] == team:
                return ahead
        return max(0, len(self.order) - start)

    def roster_names(self, team: int) -> List[str]:
        return [str(self.board.names[r]) for r in self.rosters[team]]

    def results_frame(self) -> pd.DataFrame:
        rows = []
        for pick, row in enumerate(self.picks):
            rows.append(
                {
                    "overall_pick": pick + 1,
                    "round": pick // self.n_teams + 1,
                    "team": int(self.order[pick]),
                    "player_name": str(self.board.names[row]),
                    "position": str(self.board.positions[row]),
                    "adp": float(self.board.adp[row]),
                }
            )
        return pd.DataFrame(rows)
