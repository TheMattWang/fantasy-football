"""Market ADP as a stand-in for expert consensus, and the seasons it buys back.

ECR history begins in 2021, which capped the replayable universe at 2022-2025 --
four seasons, all already spent as a holdout. FFC half-PPR ADP runs from 2018
(verified against the API), so an ADP-anchored board reaches back further and
restores a clean test the project no longer had.

`as_ecr_history` is deliberately an ADAPTER into the existing panel schema rather
than a second pipeline: two board constructions that could drift apart would be
worse than none, because every cross-era comparison would then confound the era
with the construction.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ffc_adp import as_ecr_history  # noqa: E402
from src.data.league_config import provisional_config  # noqa: E402
from src.projections.board import build_board  # noqa: E402
from src.projections.ecr import (  # noqa: E402
    PRESEASON_MONTHS,
    EcrJoinError,
    preseason_snapshot,
)


@pytest.fixture(scope="module")
def panel():
    """2018 is included because it is what the curve TRAINS on -- the panel has
    to span the training seasons as well as the season being built."""
    return as_ecr_history([2018, 2019, 2020, 2021], scoring="half-ppr", teams=12)


# --- the adapter produces what the consumer expects -----------------------

def test_it_has_every_column_preseason_snapshot_reads(panel):
    required = {"page_type", "scrape_date", "player", "pos", "ecr", "sd",
                "best", "worst"}
    assert required <= set(panel.columns)


def test_the_snapshot_date_lands_inside_the_preseason_window(panel):
    dates = pd.to_datetime(panel["scrape_date"])
    assert set(dates.dt.month) <= set(PRESEASON_MONTHS)
    # preseason_snapshot drops September dates after the 7th.
    september = dates[dates.dt.month == 9]
    assert september.empty or (september.dt.day <= 7).all()


def test_kickers_are_relabelled_to_the_projects_own_name(panel):
    """FFC calls them PK. Everything downstream expects K, and a mismatch would
    quietly drop the position rather than raise."""
    assert "PK" not in set(panel["pos"])
    assert "K" in set(panel["pos"])


def test_dispersion_comes_through_as_sd(panel):
    """FFC's stdev is the RIGHT dispersion -- spread of actual draft position,
    where ecr_sd is spread of expert opinion and runs ~2x too wide."""
    assert panel["sd"].notna().mean() > 0.9
    assert (panel["sd"].dropna() >= 0).all()


def test_best_is_never_later_than_worst(panel):
    both = panel.dropna(subset=["best", "worst"])
    assert (both["best"] <= both["worst"]).all()


# --- it round-trips through the existing machinery ------------------------

@pytest.mark.parametrize("season", [2019, 2020, 2021])
def test_preseason_snapshot_accepts_it(panel, season):
    snap = preseason_snapshot(season, history=panel)
    assert len(snap) > 100
    assert snap["ecr_rank"].is_monotonic_increasing
    assert snap["pos_rank"].min() == 1


def test_ranks_are_dense_within_each_position(panel):
    snap = preseason_snapshot(2020, history=panel)
    for pos, group in snap.groupby("pos"):
        ranks = sorted(group["pos_rank"])
        assert ranks == list(range(1, len(ranks) + 1)), f"{pos} ranks have gaps"


# --- the bug this exists to prevent ---------------------------------------

def test_build_board_actually_uses_the_history_it_is_given(panel):
    """The load-bearing test.

    `build_board` called `preseason_snapshot` with the default ECR history and
    ignored any other anchor. That made an "ADP-anchored" board for 2021 silently
    come out ECR-anchored, and made 2019/2020 look like a data limitation when
    they were a plumbing bug. 2019 has no ECR at all, so building it at all
    proves the injected panel is being read.
    """
    config = provisional_config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        board = build_board(2019, config=config, train_seasons=[2018],
                            validate=False, market_dispersion=False,
                            history=panel)
    assert len(board) > 100
    assert board["proj_points"].notna().any()


def test_without_a_history_a_pre_ecr_season_still_fails(panel):
    """The other half: if this ever stops raising, ECR has silently grown
    history it should not have and the seal needs re-examining."""
    config = provisional_config()
    with pytest.raises((EcrJoinError, Exception)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build_board(2019, config=config, train_seasons=[2018],
                        validate=False, market_dispersion=False)


def test_the_board_covers_the_whole_drafted_range(panel):
    """FFC lists ~200 players against 180 picks, so coverage of the range that
    actually gets drafted is the criterion -- not coverage of the whole board."""
    config = provisional_config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        board = build_board(2020, config=config, train_seasons=[2018, 2019],
                            validate=False, market_dispersion=False,
                            history=panel)
    drafted = config.num_teams * (config.total_rounds or 15)
    top = board.nsmallest(drafted, "adp_rank")
    assert top["proj_points"].notna().mean() == 1.0
