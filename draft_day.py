#!/usr/bin/env python3
"""Draft-day assistant.

Runs entirely offline from a pre-built board. Nothing here fetches anything --
a network stall against a 90-second pick clock is exactly the failure you cannot
recover from, so build the board beforehand::

    python -m src.projections.board --season 2026 --out data/processed/board_2026.csv
    python draft_day.py --board data/processed/board_2026.csv --slot 4

Commands during the draft::

    <name>        record a pick (fuzzy matched, tab-completes)
    me <name>     record OUR pick
    undo          undo the last pick
    board         show the top of the remaining board
    roster        show our roster and what we still need
    ?             re-show recommendations
    quit

What the recommendation columns mean
    U          probability-weighted season outcome: P(playoffs) + 2*P(title)
    dU         how much worse this option is than the best one; a small
               spread means the pick barely matters, so take the scarcer player
    P(next)    chance this player is still there at our next turn -- the number
               that should decide whether to reach
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.assertions import validate_board  # noqa: E402
from src.data.league_config import load_or_provisional  # noqa: E402
from src.draft.engine import DraftBoard, DraftSim  # noqa: E402
from src.draft.opponents import OpponentModel, assign_archetypes  # noqa: E402
from src.draft.search import NeedAdpPolicy, SeasonSimPolicy  # noqa: E402
from src.simulation.distributions import build_samples  # noqa: E402

try:
    import readline  # noqa: F401  (enables line editing / history)
except ImportError:  # pragma: no cover
    readline = None

try:
    from rapidfuzz import process as fuzz_process
except ImportError:  # pragma: no cover
    fuzz_process = None


def find_player(query: str, names: List[str], available: np.ndarray) -> Optional[int]:
    """Resolve a typed name to a board row, preferring available players."""
    query = query.strip()
    if not query:
        return None

    pool = [(i, n) for i, n in enumerate(names) if available[i]]

    exact = [i for i, n in pool if n.lower() == query.lower()]
    if len(exact) == 1:
        return exact[0]

    prefix = [i for i, n in pool if n.lower().startswith(query.lower())]
    if len(prefix) == 1:
        return prefix[0]

    contains = [i for i, n in pool if query.lower() in n.lower()]
    if len(contains) == 1:
        return contains[0]

    candidates = prefix or contains
    if len(candidates) > 1:
        print("  ambiguous:")
        for i in candidates[:8]:
            print(f"    {names[i]}")
        return None

    if fuzz_process is not None and pool:
        match = fuzz_process.extractOne(query, [n for _, n in pool], score_cutoff=72)
        if match:
            return pool[match[2]][0]

    print(f"  no available player matching {query!r}")
    return None


def show_recommendations(policy, sim: DraftSim, team: int, top: int = 8) -> None:
    print("\n  thinking...", end="", flush=True)
    candidates = policy.evaluate(sim, team)
    print("\r" + " " * 14 + "\r", end="")

    if not candidates:
        print("  no candidates")
        return

    best = candidates[0].utility
    print(f"  {'player':<26}{'pos':<5}{'ecr':>7}{'U':>9}{'dU':>8}{'P(next)':>9}")
    for candidate in candidates[:top]:
        delta = candidate.utility - best
        print(
            f"  {candidate.name:<26}{candidate.position:<5}{candidate.adp:>7.1f}"
            f"{candidate.utility:>9.4f}{delta:>8.4f}{candidate.p_available_next:>9.2f}"
        )

    # The table above is the raw ranking; choose() shrinks toward consensus
    # unless the edge clears the noise floor. Show what it would ACTUALLY take,
    # or the tool recommends one player and the agent drafts another.
    actual_row = policy.decide(candidates, sim, team)
    actual = next((c for c in candidates if c.row == actual_row), None)
    margin = getattr(policy, "confidence_margin", 0.0)
    best = candidates[0]

    print()
    if actual is None:
        print(f"  PICK: {sim.board.names[actual_row]}")
    elif actual.row == best.row:
        gap = best.utility - candidates[1].utility if len(candidates) > 1 else 0.0
        print(f"  PICK: {actual.name} -- {gap:.4f} clear of the next option.")
    else:
        print(f"  PICK: {actual.name}")
        print(
            f"    {best.name} scores {best.utility - actual.utility:+.4f} higher, "
            f"but that is inside the noise floor ({margin:.3f}), so this defers"
        )
        print(
            f"    to consensus. The simulation only overrides the market when it "
            f"is confident."
        )

    scarce = min(candidates[:3], key=lambda c: c.p_available_next)
    if np.isfinite(scarce.p_available_next):
        if actual is not None and actual.p_available_next > 0.8:
            print(
                f"    {actual.name} is {actual.p_available_next:.0%} to last to "
                f"your next turn -- consider taking {scarce.name} first."
            )


def print_roster(sim: DraftSim, team: int) -> None:
    rows = sim.rosters[team]
    if not rows:
        print("  (empty)")
        return
    frame = pd.DataFrame(
        {
            "player": [sim.board.names[r] for r in rows],
            "pos": [sim.board.positions[r] for r in rows],
            "ecr": [sim.board.adp[r] for r in rows],
        }
    )
    print(frame.to_string(index=False))
    counts = frame["pos"].value_counts().to_dict()
    print(f"  counts: {counts}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", default="data/processed/board_2026.csv")
    parser.add_argument("--slot", type=int, required=True,
                        help="our draft position, 1-indexed")
    parser.add_argument("--teams", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--fast", action="store_true",
                        help="fewer rollouts; use if picks are timing out")
    args = parser.parse_args(argv)

    path = Path(args.board)
    if not path.exists():
        print(f"no board at {path}. Build it first:\n"
              f"  python -m src.projections.board --season 2026 --out {path}")
        return 1

    frame = pd.read_csv(path)
    validate_board(frame[frame.get("is_streamed", False) != True])  # noqa: E712

    config = load_or_provisional()
    n_teams = args.teams or config.num_teams
    n_rounds = args.rounds or config.total_rounds
    our_team = args.slot - 1
    if not 0 <= our_team < n_teams:
        print(f"--slot must be between 1 and {n_teams}")
        return 1

    print(config.summary())
    print(f"\nbuilding distributions ({args.samples} season samples)...")
    board = DraftBoard.from_frame(frame)
    samples = build_samples(frame, n_samples=args.samples,
                            n_weeks=max(13, (config.playoff_start_week or 15) - 1),
                            seed=0)
    print(f"  {samples.summary()}")

    rng = np.random.default_rng(0)
    opponents = OpponentModel(
        board, assignments=assign_archetypes(n_teams, our_team, rng), rng=rng
    )
    policy = SeasonSimPolicy(
        board, config, samples, opponents,
        n_candidates=args.candidates,
        n_rollouts=1 if args.fast else args.rollouts,
        eval_samples=250 if args.fast else 400,
        rng=rng,
    )
    fallback = NeedAdpPolicy(board, config)

    sim = DraftSim(board, n_teams, n_rounds)
    names = [str(n) for n in board.names]

    print(f"\ndrafting from slot {args.slot} of {n_teams}, {n_rounds} rounds.")
    print("type a player name to record a pick, 'me <name>' for ours, '?' for help.\n")

    while not sim.complete:
        team = sim.on_the_clock
        marker = "  <-- YOU" if team == our_team else ""
        print(
            f"\n--- round {sim.round_number}, pick {sim.pick_number + 1} "
            f"(team {team + 1}){marker} ---"
        )
        if team == our_team:
            show_recommendations(policy, sim, team)

        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        lowered = raw.lower()

        if lowered in ("quit", "exit", "q"):
            break
        if lowered == "?":
            if team == our_team:
                show_recommendations(policy, sim, team)
            else:
                print("  (not our pick -- type the name of the player taken)")
            continue
        if lowered == "roster":
            print_roster(sim, our_team)
            continue
        if lowered == "board":
            rows = np.argsort(np.where(sim.available, board.adp, np.inf))[:15]
            for r in rows:
                print(f"  {board.names[r]:<26}{board.positions[r]:<5}"
                      f"{board.adp[r]:>7.1f}")
            continue
        if lowered == "undo":
            if not sim.picks:
                print("  nothing to undo")
                continue
            last = sim.picks.pop()
            for roster in sim.rosters:
                if last in roster:
                    roster.remove(last)
            sim.available[last] = True
            print(f"  undid {board.names[last]}")
            continue

        query = raw[3:] if lowered.startswith("me ") else raw
        row = find_player(query, names, sim.available)
        if row is None:
            continue
        sim.make_pick(row)
        print(f"  recorded {board.names[row]} ({board.positions[row]}) "
              f"to team {int(sim.order[sim.pick_number - 1]) + 1}")

    print("\nfinal roster:")
    print_roster(sim, our_team)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
