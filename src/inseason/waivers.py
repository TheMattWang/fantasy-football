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


def observed_to_date(
    season: int, through_week: int, *, scoring: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Points scored and games played so far this season, per player.

    Games played is the count of weeks with a stat line, not the count of weeks
    with positive points: a player can suit up and score nothing, and treating
    that as a missed game would conflate a bad performance with an injury.
    """
    from ..data import nflverse
    from ..projections.ecr import normalize_name

    weekly = nflverse.weekly_fantasy([season], scoring)
    weekly = weekly[weekly["week"] <= through_week]
    name_col = ("player_display_name" if "player_display_name" in weekly.columns
                else "player_name")
    weekly = weekly.assign(name_key=weekly[name_col].map(normalize_name))

    grouped = weekly.groupby("name_key")
    return pd.DataFrame({
        "points": grouped["fantasy_points"].sum(),
        "games": grouped["week"].nunique(),
    }).reset_index()


def blended_scores(
    samples: SampleSet,
    observed: Optional[pd.DataFrame],
    *,
    prior_games: float = 4.0,
) -> np.ndarray:
    """Preseason projection updated by what has actually happened.

    The same estimator the season simulation uses
    (:func:`src.simulation.season._reactive_estimate`), so the lineup this tool
    recommends is the lineup the objective assumes you will set. Without it both
    are frozen at the preseason number, and T3 measured that gap at **+1.523
    ranks** -- larger than any draft edge this project has been able to find.

    Players with no observed rows keep their projection, which is correct: a
    rookie who has not played yet has nothing to update on.
    """
    from ..projections.ecr import normalize_name

    scores = np.asarray(samples.decision_score, dtype=np.float64).copy()
    if observed is None or observed.empty or prior_games <= 0:
        return scores

    lookup = observed.drop_duplicates("name_key").set_index("name_key")
    keys = samples.players["player_name"].astype(str).map(normalize_name)
    points = keys.map(lookup["points"]).to_numpy(dtype=float)
    games = keys.map(lookup["games"]).to_numpy(dtype=float)

    seen = ~np.isnan(games)
    scores[seen] = ((prior_games * scores[seen] + points[seen])
                    / (prior_games + games[seen]))
    return scores


def best_lineup(
    roster: Sequence[str],
    samples: SampleSet,
    config: LeagueConfig,
    *,
    week: int = 1,
    observed: Optional[pd.DataFrame] = None,
    prior_games: float = 4.0,
) -> Dict[str, List[str]]:
    """Optimal starters for one week, chosen ex ante from projections.

    Deliberately the same slot logic the season simulation uses, so the lineup
    the tool recommends is the lineup the objective assumed you would set.

    Pass ``observed`` (from :func:`observed_to_date`) to rank by what has
    actually happened rather than by the August projection. Without it this
    returns the same lineup in week 14 as in week 1 -- ``week`` was previously
    accepted and silently ignored, which looked like in-season logic and was not.
    """
    rows = [samples.index[p] for p in roster if p in samples.index]
    if not rows:
        return {}

    ranking = blended_scores(samples, observed, prior_games=prior_games)
    scores = {int(row): float(ranking[row]) for row in rows}
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
    observed: Optional[pd.DataFrame] = None,
    prior_games: float = 4.0,
) -> pd.DataFrame:
    """Per-player start/sit call, showing what moved it.

    ``proj_ppg`` is the August number; ``rate`` is that number updated by the
    season so far; ``delta`` is the difference. A large negative delta on a
    starter is the bust you would otherwise keep starting out of habit.
    """
    lineup = best_lineup(roster, samples, config, week=week,
                         observed=observed, prior_games=prior_games)
    starters = {
        name: slot
        for slot, names in lineup.items()
        if slot != "BN"
        for name in names
    }
    ranking = blended_scores(samples, observed, prior_games=prior_games)

    rows = []
    for player in roster:
        row = samples.index.get(player)
        if row is None:
            continue
        projected = float(samples.decision_score[row])
        rate = float(ranking[row])
        rows.append(
            {
                "player": player,
                "position": str(samples.players["position"].iloc[row]),
                "proj_ppg": projected,
                "rate": rate,
                "delta": rate - projected,
                "slot": starters.get(player, "BN"),
                "start": player in starters,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["start", "rate"], ascending=[False, False])
