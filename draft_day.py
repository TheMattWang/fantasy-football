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

The table and the PICK come from different places, on purpose
-------------------------------------------------------------
The table is context from the season simulation. The pick is consensus.

E2 measured the simulation's argmax at **1.582** regret against the best
available candidate, versus **0.891** for consensus -- and **1.548** for a
candidate chosen at RANDOM. U ranks whole rosters well (rho = -0.32, p<0.0001)
and cannot rank two players at a single pick, because a shortlist is players
adjacent in ADP whose true values differ by less than our projection error, so
taking the argmax selects on the error rather than on the player.

So U is displayed and not obeyed. ``--recommender season_sim`` restores the old
behaviour for anyone who wants it; it is not the default and the measurements
say it should not be.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.assertions import (  # noqa: E402
    BoardValidationError,
    load_provenance,
    validate_board,
    validate_freshness,
)
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


def show_recommendations(
    policy, recommender, sim: DraftSim, team: int, top: int = 8
) -> None:
    """Print the candidate table, then the pick. They come from different places.

    ``policy`` supplies the table -- U, how tightly the options separate, and
    survival to our next turn. ``recommender`` supplies the actual PICK, and
    defaults to consensus. See the module docstring for why.
    """
    print("\n  thinking...", end="", flush=True)
    candidates = policy.evaluate(sim, team)
    pick_row = recommender.choose(sim, team)
    print("\r" + " " * 14 + "\r", end="")

    if candidates:
        top_utility = candidates[0].utility
        print(f"  {'player':<26}{'pos':<5}{'ecr':>7}{'U':>9}{'dU':>8}{'P(next)':>9}")
        for candidate in candidates[:top]:
            delta = candidate.utility - top_utility
            print(
                f"  {candidate.name:<26}{candidate.position:<5}{candidate.adp:>7.1f}"
                f"{candidate.utility:>9.4f}{delta:>8.4f}{candidate.p_available_next:>9.2f}"
            )

    # need_adp's legality rules differ from the shortlist's (forced K/DEF late,
    # positional limits), so its pick is not guaranteed to be in the table.
    picked = next((c for c in candidates if c.row == pick_row), None)
    name = picked.name if picked is not None else str(sim.board.names[pick_row])

    print(f"\n  PICK: {name}   [{getattr(recommender, 'name', 'recommender')}]")

    # Surface the disagreement rather than hiding it -- and do not act on it.
    if candidates and candidates[0].row != pick_row:
        top_choice = candidates[0]
        if picked is not None:
            gap = f"by {top_choice.utility - picked.utility:+.4f} U"
        else:
            gap = "(this pick is outside the shortlist)"
        print(f"    the simulation prefers {top_choice.name} {gap}. Not taken:")
        print(
            "    U's argmax measured WORSE than consensus at the pick level "
            "-- regret 1.582"
        )
        print(
            "    vs 0.891, and a RANDOM candidate scores 1.548. "
            "--recommender season_sim overrides."
        )

    # Survival comes from ADP dispersion, not from our valuation, so none of the
    # above touches it. This is the column worth acting on.
    if candidates and picked is not None:
        scarce = min(candidates[:3], key=lambda c: c.p_available_next)
        if (
            np.isfinite(scarce.p_available_next)
            and picked.p_available_next > 0.8
            and scarce.row != picked.row
        ):
            print(
                f"    {name} is {picked.p_available_next:.0%} to last to your "
                f"next turn -- consider taking {scarce.name} first."
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


def build_parser() -> argparse.ArgumentParser:
    """Factored out so the defaults are testable without building a board."""
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
    parser.add_argument("--season", type=int, default=2026,
                        help="only used in the rebuild hint on a stale board")
    parser.add_argument("--save-roster", default="data/processed/my_roster.txt",
                        help="write our roster here when the draft ends, so "
                             "week.py can read it. Pass '' to skip.")
    parser.add_argument("--allow-stale", action="store_true",
                        help="draft against a stale or incomplete board anyway. "
                             "Staleness is fatal by default: consensus moves "
                             "daily in August, and a board with no market ADP "
                             "computes P(next) from a ~2x-too-wide spread.")
    parser.add_argument("--recommender", choices=("need_adp", "season_sim"),
                        default="need_adp",
                        help="what makes the actual pick. Default is consensus: "
                             "the season simulation's argmax measured worse than "
                             "consensus at the pick level (E2). The simulation "
                             "still fills the table either way.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    path = Path(args.board)
    if not path.exists():
        print(f"no board at {path}. Build it first:\n"
              f"  python -m src.projections.board --season 2026 --out {path}")
        return 1

    frame = pd.read_csv(path)
    validate_board(frame[frame.get("is_streamed", False) != True])  # noqa: E712

    # validate_board says the board is well-formed. It cannot say it is CURRENT,
    # and a three-week-old board passes every check in it. On the clock that
    # distinction matters, so staleness is fatal here unless explicitly waived.
    try:
        validate_freshness(load_provenance(path), for_draft=not args.allow_stale,
                           raise_on_fatal=True)
    except BoardValidationError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        print("Rebuild it:\n"
              f"  python -m src.projections.board --season {args.season} "
              f"--refresh --require-market --out {path}\n"
              "or pass --allow-stale to draft anyway.", file=sys.stderr)
        return 1

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
    # The simulation always fills the table; this decides who actually picks.
    recommender = (
        policy if args.recommender == "season_sim" else NeedAdpPolicy(board, config)
    )
    print(f"  recommender: {recommender.name}")

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
            show_recommendations(policy, recommender, sim, team)

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
                show_recommendations(policy, recommender, sim, team)
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

    # The connective tissue between draft day and the rest of the season. Without
    # this the roster only ever exists in the terminal scrollback, and week.py has
    # nothing to read.
    rows = sim.rosters[our_team]
    if args.save_roster and not rows:
        # Quitting before the first pick must not clobber a real roster from an
        # earlier session. An empty file is not a draft, it is a lost one.
        print(f"\nno picks recorded -- leaving {args.save_roster} alone")
    elif args.save_roster:
        out = Path(args.save_roster)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text("\n".join(str(board.names[r]) for r in rows) + "\n")
        os.replace(tmp, out)
        print(f"\nwrote {out} ({len(rows)} players)")
        print(f"set your lineup each week with:\n"
              f"  python week.py --roster {out} --week <N>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
