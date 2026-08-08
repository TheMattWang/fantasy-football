"""Tests for the ECR baseline curve. No network required."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.projections.ecr import (  # noqa: E402
    EcrJoinError,
    fit_baseline,
    join_actuals,
    normalize_name,
    preseason_snapshot,
)


# --------------------------------------------------------------------------
# Name normalization -- the join is where the 2025 board lost 57% of its ADP
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Ja'Marr Chase", "ja marr chase"),
        ("Marvin Harrison Jr.", "marvin harrison"),
        ("Michael Pittman Jr", "michael pittman"),
        ("Odell Beckham Jr.", "odell beckham"),
        ("A.J. Brown", "aj brown"),
        ("AJ Brown", "aj brown"),
        ("D.K. Metcalf", "dk metcalf"),
        ("  Amon-Ra  St. Brown ", "amon ra st brown"),
        ("Kenneth Walker III", "kenneth walker"),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_normalize_name_is_join_stable():
    """The same player spelled two ways must produce one key."""
    assert normalize_name("Marvin Harrison Jr.") == normalize_name("Marvin Harrison")
    assert normalize_name("D.J. Moore") == normalize_name("DJ Moore")


def test_normalize_does_not_fuse_distinct_players():
    assert normalize_name("Josh Allen") != normalize_name("Keenan Allen")
    assert normalize_name("Michael Thomas") != normalize_name("Michael Pittman")


# --------------------------------------------------------------------------
# Snapshot selection
# --------------------------------------------------------------------------

def _history(rows):
    frame = pd.DataFrame(rows)
    frame["scrape_date"] = pd.to_datetime(frame["scrape_date"])
    return frame


@pytest.fixture
def history():
    rows = []
    # An in-season scrape that must NOT be picked as "preseason".
    for date in ("2024-08-01", "2024-08-28", "2024-11-15"):
        for i, (player, pos) in enumerate(
            [("A A", "RB"), ("B B", "RB"), ("C C", "WR"), ("D D", "WR"), ("E E", "QB")]
        ):
            rows.append({
                "page_type": "redraft-overall", "player": player, "pos": pos,
                "team": "XXX", "ecr": i + 1 + (10 if date == "2024-11-15" else 0),
                "sd": 1.0, "best": 1.0, "worst": 9.0, "scrape_date": date,
            })
    return _history(rows)


def test_snapshot_takes_the_last_preseason_scrape(history):
    snap = preseason_snapshot(2024, history=history)
    assert snap["scrape_date"].nunique() == 1
    assert str(snap["scrape_date"].iloc[0].date()) == "2024-08-28"


def test_snapshot_ranks_within_position(history):
    snap = preseason_snapshot(2024, history=history)
    rbs = snap[snap["pos"] == "RB"].sort_values("ecr")
    assert list(rbs["pos_rank"]) == [1, 2]
    wrs = snap[snap["pos"] == "WR"].sort_values("ecr")
    assert list(wrs["pos_rank"]) == [1, 2]


def test_snapshot_overall_rank_is_dense_and_ordered(history):
    snap = preseason_snapshot(2024, history=history)
    assert list(snap.sort_values("ecr")["ecr_rank"]) == [1, 2, 3, 4, 5]


def test_snapshot_missing_season_raises(history):
    with pytest.raises(EcrJoinError, match="no preseason"):
        preseason_snapshot(2019, history=history)


# --------------------------------------------------------------------------
# Curve fitting
# --------------------------------------------------------------------------

def _synthetic(seasons, n_per_pos=40, noise=0.0, seed=0):
    """Consensus ranks whose true production decays 1/sqrt(rank)."""
    rng = np.random.default_rng(seed)
    # Names must be purely alphabetic and distinct after normalization, which
    # strips digits -- so spell the rank out rather than using "RB1".
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    def _label(n: int) -> str:
        out = ""
        while True:
            out = alphabet[n % 26] + out
            n = n // 26 - 1
            if n < 0:
                return out

    hist_rows, actual_rows = [], []
    for season in seasons:
        for pos, top in [("RB", 20.0), ("WR", 18.0), ("QB", 24.0), ("TE", 14.0)]:
            for rank in range(1, n_per_pos + 1):
                name = f"{pos.lower()}pos {_label(rank)}er"
                hist_rows.append({
                    "page_type": "redraft-overall", "player": name, "pos": pos,
                    "team": "XXX", "ecr": rank, "sd": 1.0, "best": 1.0,
                    "worst": 9.0, "scrape_date": f"{season}-08-20",
                })
                ppg = top / np.sqrt(rank) + rng.normal(0, noise)
                actual_rows.append({
                    "season": season, "player_id": f"{season}{pos}{rank}",
                    "player_name": name, "position": pos,
                    "games_played": 16, "games_active": 16,
                    "total_points": ppg * 16, "ppg_played": ppg,
                    "ppg_available": ppg, "available_rate": 1.0,
                })
    return _history(hist_rows), pd.DataFrame(actual_rows)


def test_curve_recovers_a_monotone_signal():
    history, actuals = _synthetic([2021, 2022, 2023])
    curve = fit_baseline(
        [2021, 2022, 2023],
        history=history,
        actuals_by_season={s: actuals[actuals.season == s] for s in (2021, 2022, 2023)},
    )
    assert curve.ppg_at("RB", 1) == pytest.approx(20.0, abs=0.5)
    assert curve.ppg_at("RB", 4) == pytest.approx(10.0, abs=0.5)
    assert curve.ppg_at("RB", 16) == pytest.approx(5.0, abs=0.5)


def test_curve_is_non_increasing_even_with_noise():
    """Isotonic must stop a noisy rank bucket from inverting the board."""
    history, actuals = _synthetic([2021, 2022, 2023], noise=3.0, seed=7)
    curve = fit_baseline(
        [2021, 2022, 2023],
        history=history,
        actuals_by_season={s: actuals[actuals.season == s] for s in (2021, 2022, 2023)},
    )
    values = [curve.ppg_at("WR", r) for r in range(1, 40)]
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))


def test_season_points_combines_rate_and_availability():
    history, actuals = _synthetic([2021, 2022])
    curve = fit_baseline(
        [2021, 2022],
        history=history,
        actuals_by_season={s: actuals[actuals.season == s] for s in (2021, 2022)},
    )
    expected = curve.ppg_at("RB", 5) * curve.games_at("RB", 5)
    assert curve.season_points_at("RB", 5) == pytest.approx(expected)


def test_match_gate_rejects_a_broken_join():
    """A join that loses the top of the board must raise, not fit on the rest."""
    history, actuals = _synthetic([2021])
    broken = actuals.assign(player_name="nobody " + actuals["player_id"])
    with pytest.raises(EcrJoinError, match="name-join failure"):
        fit_baseline([2021], history=history, actuals_by_season={2021: broken})


def test_match_gate_can_be_disabled():
    history, actuals = _synthetic([2021])
    broken = actuals.assign(player_name="nobody " + actuals["player_id"])
    with pytest.raises(EcrJoinError, match="no position had enough"):
        fit_baseline(
            [2021],
            history=history,
            actuals_by_season={2021: broken},
            enforce_match_gate=False,
        )


def test_join_actuals_flags_unmatched_players():
    history, actuals = _synthetic([2021], n_per_pos=5)
    partial = actuals[actuals["player_name"] != "rbpos cer"]
    merged = join_actuals(2021, history=history, actuals=partial)
    assert not merged.loc[merged["player"] == "rbpos cer", "matched"].iloc[0]
    assert merged["matched"].sum() == len(merged) - 1
