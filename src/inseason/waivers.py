"""In-season decisions, scored by the same objective as the draft.

The 2025 system had no in-season component at all, which is a bigger hole than
it sounds: a league is lost in October as easily as in August, and unlike the
draft, waivers give you a feedback loop every week.

Everything here answers one question -- *how much does this change P(playoffs)?*
-- using the Phase 4 season simulation over the weeks that are still to come.
That gets the hard call right automatically: holding an injured star is worth it
when he returns in time to matter and not when he does not, and no "stash
bonus" coefficient is needed to express that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.league_config import LeagueConfig
from ..simulation.distributions import SampleSet
from ..simulation.season import (
    evaluate_roster,
    lineup_points,
    plan_roster,
    simulate_opponents,
)


@dataclass
class WaiverCandidate:
    """One add/drop pair, valued in playoff probability."""

    add: str
    drop: str
    add_position: str
    delta_utility: float
    delta_playoffs: float

    def __str__(self) -> str:
        return (
            f"+{self.add:<24}({self.add_position:<3}) "
            f"-{self.drop:<24} "
            f"dU={self.delta_utility:+.4f}  dP(playoffs)={self.delta_playoffs:+.3f}"
        )


def _remaining(samples: SampleSet, from_week: int) -> SampleSet:
    """A view of the sample tensor covering only the weeks still to be played."""
    start = max(0, from_week - 1)
    return SampleSet(
        points=samples.points[:, :, start:],
        players=samples.players,
        index=samples.index,
        active=None if samples.active is None else samples.active[:, :, start:],
        decision_score=samples.decision_score,
    )


def rank_waiver_adds(
    roster: Sequence[str],
    free_agents: Sequence[str],
    samples: SampleSet,
    config: LeagueConfig,
    opponent_rosters: Sequence[Sequence[str]],
    *,
    from_week: int = 1,
    n_samples: int = 600,
    protected: Optional[Sequence[str]] = None,
    top: int = 15,
) -> List[WaiverCandidate]:
    """Value every (add, drop) pair by how much it moves the season.

    Args:
        roster: our current players.
        free_agents: everyone available on waivers.
        opponent_rosters: the other teams, so "good enough to win" is relative
            to the league we are actually in.
        from_week: first week still to be played.
        protected: players never considered as the drop.
    """
    window = _remaining(samples, from_week)
    protected = set(protected or [])

    opponent_weekly = simulate_opponents(
        opponent_rosters, window, config, n_samples=n_samples
    )

    base = evaluate_roster(
        list(roster), window, config, opponent_weekly, n_samples=n_samples
    )

    droppable = [p for p in roster if p not in protected]
    results: List[WaiverCandidate] = []

    for add in free_agents:
        if add in roster or add not in window.index:
            continue
        add_position = str(window.players["position"].iloc[window.index[add]])

        best: Optional[Tuple[float, str]] = None
        for drop in droppable:
            candidate = [p for p in roster if p != drop] + [add]
            result = evaluate_roster(
                candidate, window, config, opponent_weekly, n_samples=n_samples
            )
            if best is None or result.utility > best[0]:
                best = (result.utility, drop)

        if best is None:
            continue
        utility, drop = best

        # Recompute the playoff delta for the winning pair only.
        winner = evaluate_roster(
            [p for p in roster if p != drop] + [add],
            window, config, opponent_weekly, n_samples=n_samples,
        )
        results.append(
            WaiverCandidate(
                add=add,
                drop=drop,
                add_position=add_position,
                delta_utility=utility - base.utility,
                delta_playoffs=winner.p_playoffs - base.p_playoffs,
            )
        )

    results.sort(key=lambda c: (-c.delta_utility, c.add))
    return results[:top]


def best_lineup(
    roster: Sequence[str],
    samples: SampleSet,
    config: LeagueConfig,
    *,
    week: int = 1,
) -> Dict[str, List[str]]:
    """Optimal starters for one week, chosen ex ante from projections.

    Deliberately the same slot logic the season simulation uses, so the lineup
    the tool recommends is the lineup the objective assumed you would set.
    """
    rows = [samples.index[p] for p in roster if p in samples.index]
    if not rows:
        return {}

    scores = {
        int(row): float(samples.decision_score[row]) for row in rows
    }
    positions = {
        int(row): str(samples.players["position"].iloc[row]) for row in rows
    }

    lineup: Dict[str, List[str]] = {}
    used: set = set()

    for slot, count in config.dedicated_slots.items():
        eligible = sorted(
            (r for r in rows if positions[r] == slot and r not in used),
            key=lambda r: -scores[r],
        )
        chosen = eligible[:count]
        used.update(chosen)
        if chosen:
            lineup[slot] = [str(samples.players["player_name"].iloc[r]) for r in chosen]

    for slot, eligible_positions in config.flex_slots.items():
        count = config.starting_slots.get(slot, 0)
        eligible = sorted(
            (r for r in rows
             if positions[r] in eligible_positions and r not in used),
            key=lambda r: -scores[r],
        )
        chosen = eligible[:count]
        used.update(chosen)
        if chosen:
            lineup[slot] = [str(samples.players["player_name"].iloc[r]) for r in chosen]

    bench = [r for r in rows if r not in used]
    if bench:
        lineup["BN"] = [
            str(samples.players["player_name"].iloc[r])
            for r in sorted(bench, key=lambda r: -scores[r])
        ]
    return lineup


def start_sit(
    roster: Sequence[str],
    samples: SampleSet,
    config: LeagueConfig,
    *,
    week: int = 1,
) -> pd.DataFrame:
    """Per-player start/sit call with the projection behind it."""
    lineup = best_lineup(roster, samples, config, week=week)
    starters = {
        name: slot
        for slot, names in lineup.items()
        if slot != "BN"
        for name in names
    }

    rows = []
    for player in roster:
        row = samples.index.get(player)
        if row is None:
            continue
        rows.append(
            {
                "player": player,
                "position": str(samples.players["position"].iloc[row]),
                "proj_ppg": float(samples.decision_score[row]),
                "slot": starters.get(player, "BN"),
                "start": player in starters,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["start", "proj_ppg"], ascending=[False, False])
