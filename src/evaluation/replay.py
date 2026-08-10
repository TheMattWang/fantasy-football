"""Held-out replay: draft a past season, then score it with what actually happened.

Why this replaces backtesting.py
--------------------------------
The old harness scored strategies with ``0.35 * total_vorp`` -- the same
quantity the agent maximized -- so it could not fail. This one is genuinely held
out: the *inputs* are what the pipeline would have produced before the season
(consensus ranks, curves fit only on prior years), and the *outcome* is real
weekly scoring from nflverse. The two code paths never touch.

Design
------
* Opponents are calibrated on seasons before the holdout.
* Every policy runs against the **same** opponent seed in a replicate, so the
  comparison is paired -- which is what makes ~200 replicates enough to resolve
  a half-rank effect that would otherwise need thousands.
* Lineups are set ex ante, from preseason projections. Starting whoever turned
  out to score most inflates every roster by ~14% and flatters deep teams.

The bar is ``need_adp``: consensus order, subject to roster limits -- what a
competent human does. Plain ``adp`` is kept only as a sanity floor; with no
positional awareness it drafts eight quarterbacks, so beating it proves nothing.
An agent that cannot beat the bar is not worth drafting with, and that is a
decision to make before looking at the numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.league_config import LeagueConfig, provisional_config
from ..data.nflverse import weekly_fantasy
from ..draft.engine import DraftBoard, DraftSim
from ..draft.opponents import OpponentModel, assign_archetypes
from ..draft.search import (
    AdpPolicy,
    NeedAdpPolicy,
    SeasonSimPolicy,
    VorpGreedyPolicy,
)
from ..projections.board import build_board
from ..projections.ecr import normalize_name
from ..simulation.distributions import SampleSet, build_samples
from ..simulation.season import (
    lineup_points,
    plan_roster,
    round_robin_schedule,
    simulate_league,
)
# Imported by name: `protocol` is also a parameter of replay_season.
from .protocol import check as check_protocol, record_touch


def actual_weekly(
    season: int,
    board: pd.DataFrame,
    *,
    n_weeks: int = 14,
    scoring: Optional[Dict[str, float]] = None,
) -> SampleSet:
    """A SampleSet holding one realization: what each board player really scored.

    Shape is ``(1, players, weeks)`` so the same lineup machinery works for both
    simulated and actual seasons.
    """
    weekly = weekly_fantasy([season], scoring)
    name_col = (
        "player_display_name" if "player_display_name" in weekly.columns
        else "player_name"
    )
    weekly = weekly.assign(name_key=weekly[name_col].map(normalize_name))

    grid = (
        weekly[weekly["week"] <= n_weeks]
        .pivot_table(
            index="name_key", columns="week", values="fantasy_points", aggfunc="sum"
        )
        .reindex(columns=range(1, n_weeks + 1))
        .fillna(0.0)
    )

    frame = board.reset_index(drop=True)
    keys = frame["player_name"].map(normalize_name)
    points = np.zeros((1, len(frame), n_weeks), dtype=np.float32)
    for i, key in enumerate(keys):
        if key in grid.index:
            points[0, i, :] = grid.loc[key].to_numpy(dtype=np.float32)

    return SampleSet(
        points=points,
        players=frame,
        index={str(n): i for i, n in enumerate(frame["player_name"])},
        active=points > 0,
        # Lineups are set from the PRESEASON projection, never from the outcome.
        decision_score=frame["proj_ppg"].to_numpy(dtype=np.float32),
    )


@dataclass
class ReplayResult:
    """One (policy, replicate) outcome."""

    policy: str
    replicate: int
    slot: int
    rank: int
    wins: int
    points_for: float


def _standings_from_actuals(
    rosters: Sequence[Sequence[str]],
    actuals: SampleSet,
    config: LeagueConfig,
    schedule: np.ndarray,
) -> tuple:
    """Final rank / wins / points for every team, from real weekly scores."""
    weekly = np.stack(
        [
            lineup_points(actuals, plan_roster(list(r), actuals, config))
            for r in rosters
        ],
        axis=0,
    )  # (teams, 1, weeks)

    result = simulate_league(weekly, config, schedule=schedule, team_of_interest=0)
    return weekly, result


def _load_completed(out_path: Optional[Path]) -> tuple:
    """Existing ``(replicate, policy)`` pairs and the frame holding them."""
    if out_path is None or not out_path.exists():
        return set(), pd.DataFrame()
    existing = pd.read_csv(out_path)
    if not {"replicate", "policy"}.issubset(existing.columns):
        return set(), pd.DataFrame()
    done = set(
        zip(existing["replicate"].astype(int), existing["policy"].astype(str))
    )
    return done, existing


def _flush(
    out_path: Path, rows: List[ReplayResult], existing: pd.DataFrame
) -> pd.DataFrame:
    """Append ``rows`` to ``out_path`` and return the full frame."""
    if not rows:
        return existing
    frame = pd.DataFrame([r.__dict__ for r in rows])
    if not existing.empty:
        frame = pd.concat([existing, frame], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a disconnect mid-write leaves the old file intact
    # rather than a truncated one.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(out_path)
    return frame


def replay_season(
    season: int,
    *,
    policies: Sequence[str] = ("adp", "need_adp", "vorp_greedy"),
    n_replicates: int = 50,
    slots: Optional[Sequence[int]] = None,
    n_teams: int = 12,
    n_rounds: int = 15,
    n_weeks: int = 14,
    config: Optional[LeagueConfig] = None,
    sigma_scale: float = 1.0,
    agent_kwargs: Optional[Dict] = None,
    verbose: bool = True,
    out_path: Optional[Path] = None,
    flush_every: int = 10,
    replicate_start: int = 0,
    protocol: str = "tune",
    register: Optional[str] = None,
) -> pd.DataFrame:
    """Draft ``season`` with each policy and score with that season's real results.

    ``protocol`` decides which seasons this run is allowed to see. The default,
    ``"tune"``, covers 2022/2023 and *refuses* 2024/2025; touching the holdout
    needs ``protocol="gate"`` plus a ``register`` name already committed to
    ``gate_registry.json``. See ``src/evaluation/protocol.py`` for why.

    Pass ``out_path`` to make the run resumable. Results are flushed every
    ``flush_every`` replicates and a restart skips any ``(replicate, policy)``
    already on disk. A 360-replicate run is ~30 minutes; holding all of it in
    memory means a sleep or a stray Ctrl-C costs the whole thing.

    Resuming is exact rather than approximate: every ``(replicate, policy)``
    seeds its own generator from ``10_000 + replicate``, so a pair computed
    after a restart is bit-identical to one computed before it.
    """
    # Before anything expensive, and before any holdout data is read.
    check_protocol(season, protocol=protocol, register=register)

    config = config or provisional_config()
    slots = list(slots) if slots is not None else list(range(n_teams))
    agent_kwargs = agent_kwargs or {}

    # Inputs: only information available before the season.
    board_frame = build_board(season, config=config)
    board = DraftBoard.from_frame(board_frame)

    # Outcome: what really happened.
    actuals = actual_weekly(season, board_frame, n_weeks=n_weeks)

    # The agent needs simulated distributions to reason about; those come from
    # the preseason board, never from the holdout season's results.
    samples = None
    if {"season_sim", "oracle"} & set(policies):
        samples = build_samples(board_frame, n_samples=400, n_weeks=n_weeks, seed=7)

    schedule = round_robin_schedule(n_teams, n_weeks)
    rows: List[ReplayResult] = []
    started = time.perf_counter()

    done, existing = _load_completed(out_path)
    if verbose and done:
        print(f"  resuming: {len(done)} (replicate, policy) pairs already done")

    for replicate in range(replicate_start, n_replicates):
        slot = slots[replicate % len(slots)]

        for policy_name in policies:
            if (replicate, policy_name) in done:
                continue
            # Common random numbers: identical opponent behaviour across
            # policies within a replicate makes the comparison paired.
            rng = np.random.default_rng(10_000 + replicate)
            assignments = assign_archetypes(n_teams, slot, rng)
            opponents = OpponentModel(
                board, sigma_scale=sigma_scale, assignments=assignments, rng=rng
            )
            # The policy gets its OWN stream. Sharing one generator meant a
            # policy that searched harder consumed more of it -- the oracle
            # draws 50 noise matrices per pick against season_sim's 2 -- so the
            # opponents in the real draft diverged as a side effect of how much
            # thinking we did. Same seed across policies, so the searchers still
            # see identical rollout draws as each other.
            search_rng = np.random.default_rng(90_000 + replicate)

            if policy_name == "adp":
                policy = AdpPolicy(board)
            elif policy_name == "need_adp":
                policy = NeedAdpPolicy(board, config)
            elif policy_name == "vorp_greedy":
                policy = VorpGreedyPolicy(board, config)
            elif policy_name == "season_sim":
                policy = SeasonSimPolicy(
                    board, config, samples, opponents, rng=search_rng,
                    compute_survival=False, **agent_kwargs
                )
            elif policy_name == "oracle":
                # The ceiling: what the CURRENT objective is worth if the
                # search were not sampling-limited. 50 rollouts per candidate
                # drives the standard error of each candidate's utility well
                # below the gap between adjacent candidates, and the shrinkage
                # toward consensus is switched off so it commits to its argmax.
                # A learned value function's best case is to reproduce this for
                # free, so this bounds the entire payoff of arms A-D.
                policy = SeasonSimPolicy(
                    board, config, samples, opponents, rng=search_rng,
                    compute_survival=False,
                    n_candidates=8, n_rollouts=50, confidence_margin=0.0,
                )
            else:
                raise ValueError(f"unknown policy {policy_name!r}")

            sim = DraftSim(board, n_teams, n_rounds)
            while not sim.complete:
                team = sim.on_the_clock
                if team == slot:
                    sim.make_pick(policy.choose(sim, team))
                else:
                    sim.make_pick(opponents.choose(sim, team))

            rosters = [sim.roster_names(t) for t in range(n_teams)]
            # Put our team first so simulate_league reports it.
            ordered = [rosters[slot]] + [r for t, r in enumerate(rosters) if t != slot]
            _, result = _standings_from_actuals(ordered, actuals, config, schedule)

            rows.append(
                ReplayResult(
                    policy=policy_name,
                    replicate=replicate,
                    slot=slot,
                    rank=int(result.rank[0]),
                    wins=int(result.wins[0]),
                    points_for=float(result.points_for[0]),
                )
            )

        if (replicate + 1) % flush_every == 0:
            if out_path is not None and rows:
                existing = _flush(out_path, rows, existing)
                rows = []
            if verbose:
                elapsed = time.perf_counter() - started
                rate = (replicate + 1) / elapsed
                print(
                    f"  {replicate + 1}/{n_replicates} replicates "
                    f"({rate:.1f}/s, {(n_replicates - replicate - 1) / rate:.0f}s left)"
                )

    if out_path is not None:
        frame = _flush(out_path, rows, existing)
    elif not existing.empty:
        frame = pd.concat(
            [existing, pd.DataFrame([r.__dict__ for r in rows])], ignore_index=True
        )
    else:
        frame = pd.DataFrame([r.__dict__ for r in rows])

    # Charged only once results exist: a run that crashes costs no touch.
    if protocol == "gate":
        record_touch(register, season, note=f"replay {sorted(policies)}")

    return frame


def summarize(frame: pd.DataFrame, baseline: str = "need_adp") -> pd.DataFrame:
    """Paired comparison of each policy against the baseline."""
    pivot = frame.pivot_table(
        index=["replicate", "slot"], columns="policy", values="rank"
    )
    rows = []
    for policy in pivot.columns:
        ranks = pivot[policy]
        row = {
            "policy": policy,
            "mean_rank": float(ranks.mean()),
            "p_playoffs": float((ranks <= 6).mean()),
            "p_title_game": float((ranks <= 2).mean()),
            "n": int(ranks.notna().sum()),
        }
        if policy != baseline and baseline in pivot.columns:
            paired = (pivot[baseline] - ranks).dropna()  # positive = better
            row[f"rank_gain_vs_{baseline}"] = float(paired.mean())
            if len(paired) > 1 and paired.std(ddof=1) > 0:
                sd = paired.std(ddof=1)
                se = sd / np.sqrt(len(paired))
                row["se"] = float(se)
                row["t"] = float(paired.mean() / se)
                # Replicates needed to reach the p<0.10 decision threshold at
                # the observed effect size. "Not significant" and "hopeless"
                # are different answers and the harness should distinguish them.
                if paired.mean() > 0:
                    row["n_for_p10"] = int(
                        np.ceil((1.6449 * sd / paired.mean()) ** 2)
                    )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults to a tuning season: a bare invocation must not be able to spend
    # a holdout touch by accident.
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument(
        "--protocol", choices=["tune", "gate"], default="tune",
        help="tune: 2022/2023, unlimited. gate: 2024/2025, needs --register "
             "and costs one of 3 pre-registered touches",
    )
    parser.add_argument(
        "--register", type=str, default=None,
        help="name of an experiment already in gate_registry.json (gate only)",
    )
    parser.add_argument("--replicates", type=int, default=60)
    parser.add_argument(
        "--policies", nargs="+",
        default=["adp", "need_adp", "vorp_greedy"],
    )
    parser.add_argument("--baseline", type=str, default="need_adp")
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--replicate-start", type=int, default=0,
                        help="shard across processes: run [start, replicates)")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    print(
        f"replaying {args.season} [{args.protocol}]: "
        f"{args.policies} x {args.replicates} replicates"
    )
    frame = replay_season(
        args.season,
        protocol=args.protocol,
        register=args.register,
        policies=args.policies,
        n_replicates=args.replicates,
        agent_kwargs={
            "n_candidates": args.candidates,
            "n_rollouts": args.rollouts,
        },
        # Checkpointing is on whenever --out is given, so re-running the same
        # command after an interruption picks up where it stopped.
        out_path=Path(args.out) if args.out else None,
        replicate_start=args.replicate_start,
    )

    print(f"\nheld-out {args.season} -- scored with ACTUAL weekly results\n")
    print(summarize(frame, baseline=args.baseline).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\n  rank_gain_vs_need_adp > 0 means the policy finished higher than")
    print("  a competent human (ADP + roster limits). That is the bar --")
    print("  'adp' is a sanity floor that drafts 8 QBs, not a real strategy.")

    if args.out:
        # replay_season already flushed it incrementally.
        print(f"\nwrote {args.out} ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
