"""Build the draft board: consensus rank -> projection -> VORP.

Replacement level comes straight off the baseline curve. If the league starts
28 RBs (12 teams x 2, plus RB's share of the flex), then the replacement RB is
simply the curve evaluated at RB rank 28. No separate replacement-level
estimation step, and no way for the two to disagree.

Kickers and defenses are deliberately flat-valued at zero. Both are streamed
weekly off waivers, so a pick spent on them before the last two rounds is close
to pure waste -- the Phase 4 season simulation reaches that conclusion on its
own, and giving them a real projection here would only invite the search to
spend picks chasing it.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.assertions import validate_board
from ..data.league_config import LeagueConfig, is_provisional, load_or_provisional
from .ecr import SKILL_POSITIONS, BaselineCurve, fit_baseline, preseason_snapshot

# Streamed off waivers; see module docstring.
STREAMED_POSITIONS = ("K", "DST", "DEF")

DEFAULT_TRAIN_WINDOW = 5


def _train_seasons(season: int, window: int = DEFAULT_TRAIN_WINDOW) -> List[int]:
    """Seasons whose outcomes train the curve, always strictly before ``season``.

    Strictly-before matters: training on the season you then project is the
    leak that made the old rookie model look accurate.
    """
    return [s for s in range(season - window, season) if s >= 2021]


def build_board(
    season: int,
    *,
    config: Optional[LeagueConfig] = None,
    curve: Optional[BaselineCurve] = None,
    train_seasons: Optional[Sequence[int]] = None,
    validate: bool = True,
    market_dispersion: bool = True,
    refresh: bool = False,
    history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Draft board for ``season``, anchored on preseason consensus.

    Returns one row per ranked player with the projection, its inputs, and VORP.

    ``refresh`` re-fetches ECR and market ADP rather than reading local caches.
    Without it a rebuild is a no-op, which is how a board can silently go weeks
    out of date while every assertion still passes.

    Provenance for every source is attached to ``board.attrs["provenance"]`` so
    that staleness is visible rather than inferable.
    """
    config = config or load_or_provisional()
    train = list(train_seasons) if train_seasons else _train_seasons(season)
    # `history` has to reach the curve as well as the snapshot. Feeding the
    # snapshot an ADP panel while the curve trains on ECR would anchor the two
    # halves of the board on different rank systems -- and for a pre-2021 season
    # the curve would simply fail, which is how this was found.
    curve = curve or fit_baseline(train, history=history)

    # `history` lets a caller anchor the board on something other than ECR --
    # `ffc_adp.as_ecr_history` supplies market ADP in the same schema, which is
    # what reaches seasons before ECR history begins in 2021. Without this the
    # snapshot silently falls back to ECR and an "ADP-anchored" board is not one.
    board = preseason_snapshot(
        season, positions=(*SKILL_POSITIONS, *STREAMED_POSITIONS),
        refresh=refresh, history=history,
    )

    skill = board["pos"].isin(SKILL_POSITIONS)

    board["proj_ppg"] = np.where(
        skill,
        [curve.ppg_at(p, r) for p, r in zip(board["pos"], board["pos_rank"])],
        0.0,
    )
    board["proj_games"] = np.where(
        skill,
        [curve.games_at(p, r) for p, r in zip(board["pos"], board["pos_rank"])],
        0.0,
    )
    board["proj_points"] = board["proj_ppg"] * board["proj_games"]

    # Replacement level = the curve at the rank the league actually starts.
    replacement_ppg: Dict[str, float] = {}
    replacement_points: Dict[str, float] = {}
    for pos in SKILL_POSITIONS:
        rank = config.replacement_rank(pos)
        replacement_ppg[pos] = curve.ppg_at(pos, rank)
        replacement_points[pos] = curve.season_points_at(pos, rank)

    board["replacement_rank"] = board["pos"].map(
        lambda p: config.replacement_rank(p) if p in SKILL_POSITIONS else 0
    )
    board["replacement_ppg"] = board["pos"].map(replacement_ppg).fillna(0.0)
    board["replacement_points"] = board["pos"].map(replacement_points).fillna(0.0)

    board["VORP"] = (board["proj_points"] - board["replacement_points"]).where(skill, 0.0)
    board["vorp_ppg"] = (board["proj_ppg"] - board["replacement_ppg"]).where(skill, 0.0)

    board["is_streamed"] = ~skill
    board["adp_rank"] = board["ecr_rank"]
    board = board.rename(columns={"player": "player_name", "pos": "position",
                                  "sd": "ecr_sd"})

    board = board.sort_values(
        ["VORP", "proj_points", "adp_rank"], ascending=[False, False, True]
    ).reset_index(drop=True)

    if validate:
        with warnings.catch_warnings():
            # The streamed-position rows are intentionally zero-valued, so the
            # market-agreement check is computed on the skill players only.
            warnings.simplefilter("ignore")
            validate_board(board[board["is_streamed"] == False])  # noqa: E712

    if market_dispersion:
        # How far from consensus players actually go, measured over real drafts.
        # Preseason data, so this is available before the season and carries no
        # leakage. Never fatal -- a board without it still drafts.
        try:
            from ..data.ffc_adp import attach_market_dispersion, scoring_slug

            board = attach_market_dispersion(
                board, season, scoring=scoring_slug(config.points_per_reception),
                teams=config.num_teams, refresh=refresh,
            )
        except Exception as exc:  # noqa: BLE001 - network/data shape, never fatal
            warnings.warn(
                f"no market dispersion for {season} ({exc}); the opponent model "
                "falls back to ecr_sd, which runs ~2x the real spread",
                stacklevel=2,
            )

    if is_provisional(config):
        warnings.warn(
            "Board built on PROVISIONAL league settings -- replacement level "
            "may be wrong. Pull real settings before drafting.",
            stacklevel=2,
        )

    board.attrs["provenance"] = collect_provenance(board, season, config,
                                                   train=train, refreshed=refresh)
    return board


def collect_provenance(
    board: pd.DataFrame,
    season: int,
    config: LeagueConfig,
    *,
    train: Sequence[int],
    refreshed: bool,
) -> Dict[str, object]:
    """What this board was actually built from, so staleness is visible.

    Reads the cache files' own mtimes rather than threading state back up
    through the loaders: the file on disk is the ground truth about how old the
    data is, and it cannot drift out of sync with itself.
    """
    from ..data.paths import cache_dir
    from .ecr import ECR_CACHE

    def file_age(path) -> Optional[Dict[str, object]]:
        try:
            stat = path.stat()
        except OSError:
            return None
        fetched = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        return {
            "path": str(path),
            "fetched_utc": fetched.isoformat(timespec="seconds"),
            "age_days": round((datetime.now(tz=timezone.utc) - fetched).total_seconds()
                              / 86400.0, 2),
            "bytes": stat.st_size,
        }

    ecr_dates = pd.to_datetime(board.get("scrape_date"), errors="coerce")

    # `adp_sd` is a poor coverage measure: attach_market_dispersion fits a line
    # through the matched rows and extends it over the gaps, so adp_sd ends up
    # non-null nearly everywhere. `ffc_adp` is only set where FFC actually
    # listed the player, so it is the honest count -- and what matters is
    # coverage of the players who actually get drafted, not of the whole board.
    matched = int(board["ffc_adp"].notna().sum()) if "ffc_adp" in board else 0
    drafted = int(config.num_teams) * int(config.total_rounds or 15)
    if matched and "adp_rank" in board.columns:
        top = board.nsmallest(drafted, "adp_rank")
        matched_top = int(top["ffc_adp"].notna().sum())
    else:
        matched_top = 0
    has_market = matched > 0

    provenance: Dict[str, object] = {
        "season": season,
        "built_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "refresh_requested": bool(refreshed),
        "rows": int(len(board)),
        "skill_rows": int((~board["is_streamed"]).sum()) if "is_streamed" in board else None,
        "train_seasons": [int(s) for s in train],
        "config_provisional": bool(is_provisional(config)),
        "ecr": {
            "cache": file_age(cache_dir() / ECR_CACHE),
            # The consensus snapshot's own date -- the number that actually says
            # how current the rankings are, independent of when we downloaded.
            "snapshot_date": (str(ecr_dates.max().date())
                              if ecr_dates is not None and ecr_dates.notna().any()
                              else None),
        },
        "market_adp": {
            "attached": bool(has_market),
            "matched": matched,
            "drafted_range": drafted,
            "matched_in_drafted_range": matched_top,
            "coverage": round(matched_top / drafted, 3) if drafted else 0.0,
            # Includes rows filled by the fitted dispersion line, so this is
            # always near-total and is NOT a coverage measure.
            "adp_sd_filled": int(board["adp_sd"].notna().sum()) if "adp_sd" in board else 0,
        },
    }
    return provenance


def write_provenance(board: pd.DataFrame, csv_path) -> Optional[Path]:
    """Write the sidecar next to the board CSV. Returns the path, or None."""
    provenance = board.attrs.get("provenance")
    if not provenance:
        return None
    path = Path(csv_path).with_suffix(".provenance.json")
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def board_columns(board: pd.DataFrame) -> pd.DataFrame:
    """The human-readable subset, for printing or writing to CSV."""
    cols = [
        "player_name", "position", "team", "ecr", "ecr_sd", "adp_sd", "adp_rank",
        "pos_rank", "proj_ppg", "proj_games", "proj_points",
        "replacement_points", "VORP",
    ]
    return board[[c for c in cols if c in board.columns]]


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m src.projections.board`` -- build and show the board."""
    import argparse

    parser = argparse.ArgumentParser(description="Build the draft board")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", type=str, default=None, help="write CSV here")
    parser.add_argument(
        "--refresh", action="store_true",
        help="re-fetch ECR and market ADP instead of reading the local caches. "
             "Without this a rebuild is a NO-OP and returns the same board.")
    parser.add_argument(
        "--require-market", action="store_true",
        help="fail instead of warning if market ADP does not attach. Use this "
             "for scheduled refreshes, so a degraded board is never promoted.")
    args = parser.parse_args(argv)

    config = load_or_provisional()
    print(config.summary())
    print("\nreplacement ranks: " + "  ".join(
        f"{p}={config.replacement_rank(p)}" for p in SKILL_POSITIONS
    ))

    board = build_board(args.season, config=config, refresh=args.refresh)
    print(f"\n{len(board)} players ({int((~board['is_streamed']).sum())} skill)\n")
    print(
        board_columns(board).head(args.top).to_string(
            index=False, float_format=lambda v: f"{v:8.2f}"
        )
    )

    prov = board.attrs.get("provenance", {})
    ecr = prov.get("ecr", {}) or {}
    market = prov.get("market_adp", {}) or {}
    cache = ecr.get("cache") or {}
    print("\nprovenance")
    print(f"  ECR snapshot   {ecr.get('snapshot_date')}")
    print(f"  ECR cache      {cache.get('age_days')} days old")
    print(f"  market ADP     {'attached' if market.get('attached') else 'MISSING'}"
          f" -- {market.get('matched_in_drafted_range', 0)}"
          f"/{market.get('drafted_range', 0)} of the drafted range"
          f" ({market.get('coverage', 0):.0%})")
    if not args.refresh:
        print("  NOTE: built from caches. Pass --refresh to actually re-fetch.")

    if args.require_market and not market.get("attached"):
        print("\nrefusing to write: market ADP did not attach and --require-market "
              "was set.\nThe opponent model would fall back to ecr_sd, which runs "
              "~2x the real\ndraft spread and corrupts P(next).", file=sys.stderr)
        return 2

    if args.out:
        # Write to a temp file and rename, so a failure here can never leave a
        # half-written board where a working one used to be. Draft day reads
        # this file; a truncated CSV on the clock is unrecoverable.
        out = Path(args.out)
        tmp = out.with_suffix(out.suffix + ".tmp")
        board.to_csv(tmp, index=False)
        os.replace(tmp, out)
        sidecar = write_provenance(board, out)
        print(f"\nwrote {out}")
        if sidecar:
            print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
