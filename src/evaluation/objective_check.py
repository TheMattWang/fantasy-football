"""Does the objective rank PICKS correctly, or only ROSTERS?

The distinction matters more than it sounds. ``U = P(playoffs) + 2*P(title)``
correlates with real final rank at rho ~ -0.32 across rosters built by different
strategies, so it clearly knows a good roster from a bad one. But a draft agent
never chooses between whole strategies. It chooses between Bijan and CMC at one
pick, and what it needs is for ``U`` to order *those* correctly.

Those are different requirements, and an objective can pass the first and fail
the second. If it does, more search makes the agent worse -- which is exactly
what the ceiling experiment measured (need_adp 4.21, season_sim 4.62, oracle
5.88: the harder you optimize ``U``, the worse you finish).

Method
------
At a sampled draft state, take the shortlist of candidates. For each candidate:

* **model**: ``U``, averaged over ``n_rollouts`` simulated continuations --
  what the agent believes.
* **truth**: take that player, finish the draft with the same fixed
  continuation policy against ``n_truth`` opponent draws, and score with the
  season's REAL weekly results. Average the final rank.

The opponent draws are shared across candidates (common random numbers), so the
difference between two candidates is the pick and not the draw.

Then compare:

* ``spearman(U, actual_rank)`` per state -- want strongly negative.
* ``regret`` -- the actual rank of the pick ``U`` recommends, minus the actual
  rank of the best candidate. Zero means the objective picked the best one.
  Compare against the regret of ``need_adp``'s pick and of a random candidate;
  an objective that cannot beat *random* on this is worse than useless, because
  the search will confidently drive toward its argmax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..data.league_config import LeagueConfig, provisional_config
from ..draft.engine import DraftBoard, DraftSim
from ..draft.opponents import OpponentModel, assign_archetypes, draw_noise
from ..draft.search import NeedAdpPolicy, SeasonSimPolicy
from ..projections.board import build_board
from ..simulation.distributions import build_samples
from ..simulation.season import round_robin_schedule
from .replay import _standings_from_actuals, actual_weekly


@dataclass
class StateResult:
    season: int
    state: int
    slot: int
    pick_number: int
    rho: float
    regret_u: float
    regret_need_adp: float
    regret_random: float
    spread_actual: float


def _clone(sim: DraftSim) -> DraftSim:
    clone = DraftSim(sim.board, sim.n_teams, sim.n_rounds)
    clone.available = sim.available.copy()
    clone.rosters = [list(r) for r in sim.rosters]
    clone.picks = list(sim.picks)
    return clone


def _finish_and_score(
    sim: DraftSim,
    team: int,
    first_pick: int,
    board: DraftBoard,
    config: LeagueConfig,
    actuals,
    schedule: np.ndarray,
    opponents: OpponentModel,
    noise: np.ndarray,
    continuation,
) -> int:
    """Take ``first_pick``, finish the draft, return our REAL final rank."""
    clone = _clone(sim)
    saved = opponents.noise
    opponents.noise = noise
    try:
        clone.make_pick(first_pick)
        while not clone.complete:
            current = clone.on_the_clock
            if current == team:
                clone.make_pick(continuation.choose(clone, current))
            else:
                clone.make_pick(opponents.choose(clone, current))
    finally:
        opponents.noise = saved

    rosters = [clone.roster_names(t) for t in range(clone.n_teams)]
    ordered = [rosters[team]] + [r for t, r in enumerate(rosters) if t != team]
    _, result = _standings_from_actuals(ordered, actuals, config, schedule)
    return int(result.rank[0])


def check_season(
    season: int,
    *,
    config: Optional[LeagueConfig] = None,
    n_states: int = 16,
    n_candidates: int = 8,
    n_rollouts: int = 10,
    n_truth: int = 24,
    n_weeks: int = 14,
    max_round: int = 8,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare the objective's candidate ordering to the realized one."""
    config = config or provisional_config()
    board_frame = build_board(season, config=config)
    board = DraftBoard.from_frame(board_frame)
    actuals = actual_weekly(season, board_frame, n_weeks=n_weeks)
    samples = build_samples(board_frame, n_samples=400, n_weeks=n_weeks, seed=7)
    schedule = round_robin_schedule(config.num_teams, n_weeks)

    master = np.random.default_rng(seed)
    rows: List[StateResult] = []

    for state in range(n_states):
        slot = state % config.num_teams
        rng = np.random.default_rng(4_000 + state)
        opponents = OpponentModel(
            board, assignments=assign_archetypes(config.num_teams, slot, rng), rng=rng
        )
        continuation = NeedAdpPolicy(board, config)

        # Walk to a random decision point that belongs to us.
        sim = DraftSim(board, config.num_teams, config.total_rounds)
        target = int(master.integers(0, max_round)) * config.num_teams
        while not sim.complete and (
            sim.pick_number < target or sim.on_the_clock != slot
        ):
            current = sim.on_the_clock
            sim.make_pick(
                continuation.choose(sim, current) if current == slot
                else opponents.choose(sim, current)
            )
        if sim.complete:
            continue

        policy = SeasonSimPolicy(
            board, config, samples, opponents,
            n_candidates=n_candidates, n_rollouts=n_rollouts,
            eval_samples=300, confidence_margin=0.0, compute_survival=False,
            rng=np.random.default_rng(7_000 + state),
        )
        candidates = policy.evaluate(sim, slot)
        if len(candidates) < 3:
            continue
        rows_of = [c.row for c in candidates]
        utility = np.array([c.utility for c in candidates])

        # Truth: same opponent draws for every candidate.
        noises = [
            draw_noise(len(sim.order), len(board), np.random.default_rng(9_000 + t))
            for t in range(n_truth)
        ]
        actual = np.array([
            np.mean([
                _finish_and_score(
                    sim, slot, row, board, config, actuals, schedule,
                    opponents, noise, continuation,
                )
                for noise in noises
            ])
            for row in rows_of
        ])

        rho = float(spearmanr(utility, actual).statistic)
        best = float(actual.min())
        pick_u = int(np.argmax(utility))
        default_row = continuation.choose(sim, slot)
        pick_need = (
            rows_of.index(default_row) if default_row in rows_of else pick_u
        )

        rows.append(StateResult(
            season=season,
            state=state,
            slot=slot,
            pick_number=sim.pick_number,
            rho=rho,
            regret_u=float(actual[pick_u]) - best,
            regret_need_adp=float(actual[pick_need]) - best,
            regret_random=float(actual.mean()) - best,
            spread_actual=float(actual.max() - actual.min()),
        ))
        if verbose:
            r = rows[-1]
            print(
                f"    state {state:>2} (pick {r.pick_number:>3}): rho {rho:+.2f}  "
                f"regret U {r.regret_u:.2f}  need_adp {r.regret_need_adp:.2f}  "
                f"random {r.regret_random:.2f}"
            )

    return pd.DataFrame([r.__dict__ for r in rows])


def summarize(frame: pd.DataFrame) -> str:
    lines = []
    rho = frame["rho"].mean()
    n = len(frame)
    se = frame["rho"].std(ddof=1) / np.sqrt(max(n, 1))
    lines.append(f"  spearman(U, actual rank) within a decision: {rho:+.3f} "
                 f"(se {se:.3f}, n={n} states)   [want strongly NEGATIVE]")
    lines.append("")
    lines.append("  regret = actual ranks lost vs the best candidate (lower is better)")
    for column, label in (
        ("regret_u", "U's argmax"),
        ("regret_need_adp", "need_adp's pick"),
        ("regret_random", "a random candidate"),
    ):
        lines.append(f"    {label:<20} {frame[column].mean():.3f}")
    lines.append("")
    lines.append(f"  candidate spread (best vs worst, actual ranks): "
                 f"{frame['spread_actual'].mean():.2f}")
    if frame["regret_u"].mean() >= frame["regret_random"].mean():
        lines.append("")
        lines.append("  U is no better than picking at random among the shortlist.")
        lines.append("  Optimizing it harder cannot help, and will hurt, because the")
        lines.append("  search commits to an argmax that carries no information.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument("--states", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--truth", type=int, default=24)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    frames = []
    for season in args.seasons:
        print(f"\n{season}:")
        frames.append(check_season(
            season, n_states=args.states, n_candidates=args.candidates,
            n_rollouts=args.rollouts, n_truth=args.truth,
        ))
    frame = pd.concat(frames, ignore_index=True)

    print("\nDoes the objective rank PICKS, or only ROSTERS?\n")
    print(summarize(frame))

    if args.out:
        frame.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
