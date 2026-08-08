"""Held-out comparison: the ECR baseline curve vs. what clean.py actually did.

This is the honest test of hypothesis (1) -- "the regressions were bad". It fits
the baseline curve on earlier seasons only, predicts a held-out season, and
scores both methods against that season's actual results.

Run::

    python -m src.projections.validate --holdout 2025

The number that matters is the last one: the mean actual points produced by the
top 36 players on each board. Rank correlation is easy to move; what a draft
spends its early picks on is what decides the season.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.nflverse import season_rates
from .ecr import fit_baseline, normalize_name, preseason_snapshot

TOP_N = 36  # roughly the first three rounds of a 12-team draft


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if len(rx) < 3 or rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def build_comparison(
    holdout: int,
    train_seasons: Sequence[int],
    *,
    scoring: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """One row per held-out-season player with both projections and the outcome."""
    curve = fit_baseline(train_seasons, scoring=scoring)

    snapshot = preseason_snapshot(holdout)
    snapshot["pred_ecr"] = [
        curve.ppg_at(pos, rank)
        for pos, rank in zip(snapshot["pos"], snapshot["pos_rank"])
    ]

    # clean.py's projection: last season's points over games with a stat line.
    prior = season_rates([holdout - 1], scoring)
    prior["name_key"] = prior["player_name"].map(normalize_name)
    prior = prior.drop_duplicates(["name_key", "position"])
    snapshot = snapshot.merge(
        prior[["name_key", "position", "ppg_played"]].rename(
            columns={"ppg_played": "pred_prior_year", "position": "pos"}
        ),
        on=["name_key", "pos"],
        how="left",
    )

    actual = season_rates([holdout], scoring)
    actual["name_key"] = actual["player_name"].map(normalize_name)
    actual = actual.drop_duplicates(["name_key", "position"])

    merged = snapshot.merge(
        actual[["name_key", "position", "ppg_available", "total_points"]].rename(
            columns={"position": "pos"}
        ),
        on=["name_key", "pos"],
        how="inner",
    )
    return merged.dropna(subset=["ppg_available"])


def report(holdout: int, train_seasons: Sequence[int]) -> pd.DataFrame:
    frame = build_comparison(holdout, train_seasons)

    methods = [
        ("ECR baseline curve", "pred_ecr"),
        (f"clean.py: {holdout - 1} ppg_played", "pred_prior_year"),
    ]

    print(f"train {list(train_seasons)} -> holdout {holdout}")
    print(f"{len(frame)} players with a {holdout} consensus rank and actual results\n")

    print("coverage (a method that cannot rank a player cannot draft him):")
    for label, col in methods:
        n = int(frame[col].notna().sum())
        print(f"  {label:<32} {n:>4}/{len(frame)}  ({n / len(frame):.0%})")

    both = frame.dropna(subset=[c for _, c in methods])
    print(f"\naccuracy on the {len(both)} players both methods can rank:")
    print(f"  {'method':<32} {'spearman':>9} {'MAE':>7}")
    for label, col in methods:
        rho = _spearman(both[col], both["ppg_available"])
        mae = float(np.abs(both[col] - both["ppg_available"]).mean())
        print(f"  {label:<32} {rho:>9.3f} {mae:>7.2f}")

    # A draft picks by value over replacement, not by raw rate -- ranking on
    # ppg just sorts quarterbacks to the top. So score the decision that is
    # actually made.
    for _, col in methods:
        both = _add_vorp(both, col)

    print(f"\ntop-N of each board BY VORP -> mean ACTUAL {holdout} total points")
    print("  (the decision a draft actually makes)\n")
    header = "  " + f"{'method':<32}" + "".join(f"{f'top{n}':>9}" for n in (12, 24, 36, 60))
    print(header)
    for label, col in methods:
        cells = "".join(
            f"{float(both.nlargest(n, col + '_vorp')['total_points'].mean()):>9.1f}"
            for n in (12, 24, 36, 60)
        )
        print(f"  {label:<32}{cells}")
    cells = "".join(
        f"{float(both.nlargest(n, 'total_points')['total_points'].mean()):>9.1f}"
        for n in (12, 24, 36, 60)
    )
    print(f"  {'perfect hindsight':<32}{cells}")

    # n=36 means the difference of two means is noisy; say how noisy.
    a_col, b_col = methods[0][1] + "_vorp", methods[1][1] + "_vorp"
    diff, lo, hi = _bootstrap_gap(both, a_col, b_col, TOP_N)
    hindsight = float(both.nlargest(TOP_N, "total_points")["total_points"].mean())
    baseline = float(both.nlargest(TOP_N, b_col)["total_points"].mean())
    available = hindsight - baseline

    print(
        f"\n  top-{TOP_N} gain over clean.py: {diff:+.1f} pts/pick "
        f"(95% CI {lo:+.1f} to {hi:+.1f})"
    )
    if available > 0:
        print(f"  = {diff / available:.0%} of the gap to perfect foresight")
    if lo <= 0 <= hi:
        print("  NOTE: the interval spans zero -- not resolvable on one season.")
    return frame


def _add_vorp(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    """Value over replacement for a projection column, using 12-team starters."""
    starters = {"QB": 12, "RB": 28, "WR": 28, "TE": 16}
    frame = frame.copy()
    replacement = {}
    for pos, rank in starters.items():
        values = frame.loc[frame["pos"] == pos, col].dropna().sort_values(ascending=False)
        replacement[pos] = float(values.iloc[min(rank, len(values)) - 1]) if len(values) else 0.0
    frame[col + "_vorp"] = frame[col] - frame["pos"].map(replacement)
    return frame


def _bootstrap_gap(
    frame: pd.DataFrame, a_col: str, b_col: str, top_n: int, draws: int = 2000
) -> tuple:
    """Bootstrap CI for the difference in mean actual points of two top-N boards."""
    rng = np.random.default_rng(0)
    a = frame.nlargest(top_n, a_col)["total_points"].to_numpy(dtype=float)
    b = frame.nlargest(top_n, b_col)["total_points"].to_numpy(dtype=float)
    gaps = np.empty(draws)
    for i in range(draws):
        gaps[i] = (
            rng.choice(a, size=len(a), replace=True).mean()
            - rng.choice(b, size=len(b), replace=True).mean()
        )
    return float(a.mean() - b.mean()), float(np.percentile(gaps, 2.5)), float(
        np.percentile(gaps, 97.5)
    )


def report_multi(holdouts: Sequence[int]) -> None:
    """Walk-forward validation: each season held out, trained only on its past.

    A single season cannot resolve a ~20 pts/pick effect -- the bootstrap CI on
    one year spans zero. Pooling several held-out seasons is the only honest way
    to say whether the curve actually beats what clean.py did.
    """
    methods = [("ECR baseline curve", "pred_ecr"), ("clean.py prior-year ppg", "pred_prior_year")]
    per_season = []

    for holdout in holdouts:
        train = [s for s in range(2021, holdout)]
        if not train:
            print(f"  skip {holdout}: no training seasons before it")
            continue
        frame = build_comparison(holdout, train)
        both = frame.dropna(subset=[c for _, c in methods])
        for _, col in methods:
            both = _add_vorp(both, col)

        row = {"season": holdout, "n": len(both)}
        for label, col in methods:
            row[label] = float(
                both.nlargest(TOP_N, col + "_vorp")["total_points"].mean()
            )
        row["hindsight"] = float(
            both.nlargest(TOP_N, "total_points")["total_points"].mean()
        )
        row["gap"] = row[methods[0][0]] - row[methods[1][0]]
        per_season.append(row)

    table = pd.DataFrame(per_season)
    print(f"\nwalk-forward, top-{TOP_N} by VORP -> mean actual season points\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:8.1f}"))

    gaps = table["gap"].to_numpy(dtype=float)
    n = len(gaps)
    mean = float(gaps.mean())
    print(f"\n  mean gain across {n} held-out seasons: {mean:+.1f} pts/pick")
    if n >= 2:
        se = float(gaps.std(ddof=1) / np.sqrt(n))
        print(f"  per-season sd {gaps.std(ddof=1):.1f}, se {se:.1f}")
        print(f"  seasons where ECR wins: {int((gaps > 0).sum())}/{n}")
        if se > 0:
            print(f"  t = {mean / se:.2f}  (n={n}, so this is indicative, not decisive)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, nargs="+", default=[2025])
    parser.add_argument("--train", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    if len(args.holdout) > 1:
        report_multi(args.holdout)
        return 0

    holdout = args.holdout[0]
    train = args.train or [s for s in range(2021, holdout)]
    if holdout in train:
        raise SystemExit("holdout season must not appear in --train")
    report(holdout, train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
