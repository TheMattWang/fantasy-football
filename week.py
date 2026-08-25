#!/usr/bin/env python3
"""Weekly start/sit assistant -- the in-season half of the season.

Why this exists
---------------
The draft is the smaller decision. Measured on this simulator:

    a realistic draft improvement (median -> 90th percentile)   +0.832 ranks
    a realistic in-season policy (frozen -> reacting)           +1.523 ranks

and the draft number is not even available to us -- V1 through V7 found no
valuation edge anywhere, so consensus is already the right draft. The in-season
number needs no edge at all. It only needs somebody to notice that a player has
been bad for six weeks, which is exactly the decision the tooling could not
previously make: `best_lineup` took a `week` argument and ignored it, ranking by
the August projection all season.

What it does
------------
Ranks your roster by a running estimate that blends the preseason projection
with what has actually happened, weighting the prior as ``--prior-games`` games.
Early in the year the projection dominates, which is right -- two weeks is not
evidence. By November the season dominates.

    python week.py --roster data/processed/my_roster.txt --week 5

Columns
    proj    the August projection, which is what the old tool used alone
    rate    that projection updated by the season so far
    delta   the difference. A large negative delta on a starter is the bust you
            would otherwise keep starting out of habit.

Unlike draft_day.py this is allowed to touch the network -- there is no pick
clock, and the whole point is to read results that only exist online.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.assertions import load_provenance, validate_freshness  # noqa: E402
from src.data.league_config import load_or_provisional  # noqa: E402
from src.inseason.availability import availability, injury_report  # noqa: E402
from src.inseason.waivers import (  # noqa: E402
    best_lineup,
    observed_to_date,
    start_sit,
)
from src.simulation.distributions import build_samples  # noqa: E402


def season_week_range(season: int) -> tuple:
    """First and last week the NFL actually plays in this season.

    Read from the schedule rather than hardcoded, because it has changed: the
    regular season went from 16 games to 17 in 2021 and the week count follows
    the data, not a constant we would forget to update.

    Falls back to 1-18 when the schedule is unreachable, which is wide enough to
    catch a typo without refusing a legitimate week while offline.
    """
    try:
        from src.data.nflverse import load

        games = load("schedules")
        games = games[(games["season"] == int(season)) & (games["game_type"] == "REG")]
        if not games.empty:
            return int(games["week"].min()), int(games["week"].max())
    except Exception:
        pass
    return 1, 18


def read_roster(path: Path) -> List[str]:
    names = [line.strip() for line in path.read_text().splitlines()]
    return [n for n in names if n and not n.startswith("#")]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--roster", default="data/processed/my_roster.txt",
                        help="one player name per line; draft_day.py writes it")
    parser.add_argument("--week", type=int, required=True,
                        help="the week you are setting a lineup FOR. Results "
                             "through week-1 are used; this week is not known.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--board", default=None,
                        help="defaults to data/processed/board_<season>.csv")
    parser.add_argument("--prior-games", type=float, default=2.0,
                        help="how stubborn to be, in games of preseason prior. "
                             "Swept 1-16 in T3; the gain was flat at 1.27-1.49 "
                             "ranks across that whole range, so this is not a "
                             "knife edge and 2 is not load-bearing.")
    parser.add_argument("--samples", type=int, default=400)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # An out-of-range week used to be accepted silently. `--week 0` became week
    # 1, and `--week 99` read "results through week 98", aggregated the whole
    # season and printed a confident lineup for a week that does not exist --
    # a typo producing plausible output instead of an error, which is the
    # failure mode this project keeps having to fix.
    first, last = season_week_range(args.season)
    if not first <= args.week <= last:
        print(f"week {args.week} is not a week of the {args.season} season "
              f"({first}-{last}).", file=sys.stderr)
        return 1

    roster_path = Path(args.roster)
    if not roster_path.exists():
        print(f"no roster at {roster_path}.\n"
              f"draft_day.py writes one when the draft ends, or make it by hand:\n"
              f"  one player name per line", file=sys.stderr)
        return 1
    roster = read_roster(roster_path)
    if not roster:
        print(f"{roster_path} is empty", file=sys.stderr)
        return 1

    board_path = Path(args.board or f"data/processed/board_{args.season}.csv")
    if not board_path.exists():
        print(f"no board at {board_path}. Build it:\n"
              f"  python -m src.projections.board --season {args.season} "
              f"--refresh --out {board_path}", file=sys.stderr)
        return 1

    # for_draft=False on purpose. The preseason board IS the prior here, so an
    # August snapshot in week 9 is expected rather than a fault -- what has to be
    # current is the observed data below, not the projection.
    validate_freshness(load_provenance(board_path), for_draft=False,
                       raise_on_fatal=False)

    frame = pd.read_csv(board_path)
    config = load_or_provisional()
    samples = build_samples(frame, n_samples=args.samples, n_weeks=14, seed=0)

    through = args.week - 1
    if through < 1:
        print("week 1: no results yet, so this is the preseason projection.\n")
        observed = None
    else:
        print(f"reading results through week {through}...", flush=True)
        try:
            observed = observed_to_date(args.season, through)
            print(f"  {len(observed)} players with a stat line")
        except Exception as exc:
            # The season's weekly feed does not exist until games are played, so
            # asking for week 6 of a season that has not started 404s. That is a
            # normal state, not a crash: fall back to the preseason projection
            # and say so, rather than showing the user a stack trace.
            observed = None
            print(f"  no results published for {args.season} yet "
                  f"({type(exc).__name__}) -- using the preseason projection")

    missing = [p for p in roster if p not in samples.index]
    if missing:
        print(f"not on the board, skipped: {', '.join(missing)}\n")

    # Who is actually playing. A bye or an "Out" designation is a guaranteed
    # zero, and no amount of scoring rate makes up for not being on the field.
    report = injury_report(args.season, args.week)
    if len(report):
        print(f"  injury report: {len(report)} designations this week")
    avail = availability(samples, week=args.week, board=frame, report=report)

    table = start_sit(roster, samples, config, week=args.week,
                      observed=observed, prior_games=args.prior_games,
                      availability=avail)

    starters = table[table["start"]]
    bench = table[~table["start"]]

    print(f"\nWEEK {args.week} LINEUP")
    print(f"  {'':<26}{'pos':<5}{'slot':<7}{'proj':>7}{'rate':>7}"
          f"{'exp':>7}  status")
    for label, block in (("START", starters), ("BENCH", bench)):
        print(f"  -- {label} " + "-" * 50)
        for _, row in block.iterrows():
            print(f"  {row['player']:<26}{row['position']:<5}{row['slot']:<7}"
                  f"{row['proj_ppg']:>7.1f}{row['rate']:>7.1f}"
                  f"{row['expected']:>7.1f}  {row['status']}")

    sidelined = table[table["status"] != ""]
    if len(sidelined):
        print("\n  NOT FULLY AVAILABLE:")
        for _, row in sidelined.iterrows():
            where = "STARTING" if row["start"] else "benched"
            print(f"    {row['status']:<13}{row['player']:<26}"
                  f"{row['rate']:>5.1f} -> {row['expected']:>4.1f} ppg   [{where}]")
        if (sidelined["start"] & (sidelined["expected"] == 0)).any():
            print("    ^ a zero-expected player is STARTING -- you have no one "
                  "else eligible at that slot. Check waivers.")

    # The value of reacting is exactly the picks that differ from the frozen
    # lineup, so show those rather than making the reader diff two tables.
    if observed is not None:
        frozen = best_lineup(roster, samples, config, observed=None)
        was = {n for slot, names in frozen.items() if slot != "BN" for n in names}
        now = set(starters["player"])
        benched, promoted = sorted(was - now), sorted(now - was)
        print()
        if not benched and not promoted:
            print("  no change from the preseason lineup this week.")
        else:
            print("  CHANGES vs starting off the August projection:")

            def rate_of(name: str) -> float:
                return float(table.loc[table["player"] == name, "rate"].iloc[0])

            # Report the RATE, not the delta. A player can be benched while his
            # own delta is positive -- he did not decline, somebody else rose
            # past him -- and printing "BENCH (+0.9)" makes that look like a bug.
            for name in benched:
                print(f"    BENCH  {name:<26} {rate_of(name):>5.1f} ppg")
            for name in promoted:
                print(f"    START  {name:<26} {rate_of(name):>5.1f} ppg")
            if benched and promoted:
                print(f"    ({rate_of(promoted[0]) - rate_of(benched[0]):+.1f} ppg "
                      f"on the closest swap; a benched player may still be beating "
                      f"his own projection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
