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

import warnings
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
) -> pd.DataFrame:
    """Draft board for ``season``, anchored on preseason consensus.

    Returns one row per ranked player with the projection, its inputs, and VORP.
    """
    config = config or load_or_provisional()
    train = list(train_seasons) if train_seasons else _train_seasons(season)
    curve = curve or fit_baseline(train)

    board = preseason_snapshot(season, positions=(*SKILL_POSITIONS, *STREAMED_POSITIONS))

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
                teams=config.num_teams,
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
    return board


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
    args = parser.parse_args(argv)

    config = load_or_provisional()
    print(config.summary())
    print("\nreplacement ranks: " + "  ".join(
        f"{p}={config.replacement_rank(p)}" for p in SKILL_POSITIONS
    ))

    board = build_board(args.season, config=config)
    print(f"\n{len(board)} players ({int((~board['is_streamed']).sum())} skill)\n")
    print(
        board_columns(board).head(args.top).to_string(
            index=False, float_format=lambda v: f"{v:8.2f}"
        )
    )

    if args.out:
        board.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
