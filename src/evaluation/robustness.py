"""Did the agent learn football, or did it learn my simulator?

The 2025 agent maximized roster VORP and was graded on ``0.35 x roster VORP``,
so it could not fail. The rebuild fixed that specific loop, but a learned policy
trained against this simulator can rediscover the same trick one level up: the
season model contains hand-written constants -- six waiver win-rates, a weekly
CV table, a projection-error sigma -- and an agent with capacity will happily
exploit whichever of them is wrong.

The test
--------
Take a **frozen** policy, retrain nothing, and re-run the held-out replay with
one simulator constant perturbed. Report *edge retention*: the paired rank gain
under the perturbation as a fraction of the unperturbed gain.

    retention = gain(perturbed) / gain(baseline)

An agent that learned to draft keeps most of its edge when the waiver win-rate
moves 30%, because real drafting advantage does not live in that number. An
agent that learned the simulator does not.

Pass criterion, pre-committed
-----------------------------
* median edge retention across all perturbations >= 0.60, **and**
* no single perturbation drives the gain below zero.

Calibrating the test itself
---------------------------
``flex_hindsight`` re-runs under the pre-2026 FLEX rule, which picked the flex
starter with knowledge of the week's outcome and was worth ~6.2 pts/week to a
deep bench. That is a known, quantified defect, so it doubles as a positive
control: a robustness test that cannot flag ``flex_hindsight`` is too weak to
trust on the perturbations whose effect is unknown.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..simulation import distributions as dist
from ..simulation import season as season_mod
from .replay import replay_season, summarize


def _scale_dict(mapping: Dict[str, float], factor: float) -> Dict[str, float]:
    return {k: v * factor for k, v in mapping.items()}


def _scale_cv(table: Dict[str, List], factor: float) -> Dict[str, List]:
    """WEEKLY_CV maps position -> [(upper ppg bound, cv), ...]; scale the cv."""
    return {
        pos: [(bound, cv * factor) for bound, cv in rows]
        for pos, rows in table.items()
    }


# Each entry patches module attributes for the duration of one replay.
PERTURBATIONS: Dict[str, Dict[str, Callable]] = {
    "waiver_win_low": {
        "season.WAIVER_WIN_RATE": lambda v: _scale_dict(v, 0.7),
    },
    "waiver_win_high": {
        "season.WAIVER_WIN_RATE": lambda v: {
            k: min(1.0, x) for k, x in _scale_dict(v, 1.3).items()
        },
    },
    "waiver_floor_low": {
        "season.DEFAULT_WAIVER_FLOOR": lambda v: _scale_dict(v, 0.7),
        "season.DEEP_WAIVER_FLOOR": lambda v: _scale_dict(v, 0.7),
    },
    "waiver_floor_high": {
        "season.DEFAULT_WAIVER_FLOOR": lambda v: _scale_dict(v, 1.3),
        "season.DEEP_WAIVER_FLOOR": lambda v: _scale_dict(v, 1.3),
    },
    "weekly_cv_low": {"distributions.WEEKLY_CV": lambda v: _scale_cv(v, 0.7)},
    "weekly_cv_high": {"distributions.WEEKLY_CV": lambda v: _scale_cv(v, 1.3)},
    "sigma_proj_low": {"distributions.SIGMA_PROJ": lambda v: _scale_dict(v, 0.7)},
    "sigma_proj_high": {"distributions.SIGMA_PROJ": lambda v: _scale_dict(v, 1.3)},
    "injury_short": {"distributions.P_RETURN_AFTER_OUT": lambda v: 0.16},
    "injury_long": {"distributions.P_RETURN_AFTER_OUT": lambda v: 0.09},
    # Positive control: a defect of known size and known direction.
    "flex_hindsight": {"season.FLEX_OMNISCIENT_DEFAULT": lambda v: True},
}

MODULES = {"season": season_mod, "distributions": dist}


@contextmanager
def perturbed(name: str):
    """Patch simulator constants for the duration of the block."""
    if name == "baseline":
        yield
        return
    if name not in PERTURBATIONS:
        raise ValueError(
            f"unknown perturbation {name!r}; have {sorted(PERTURBATIONS)}"
        )

    saved = {}
    try:
        for path, transform in PERTURBATIONS[name].items():
            module_name, attribute = path.split(".")
            module = MODULES[module_name]
            original = getattr(module, attribute)
            saved[path] = original
            setattr(module, attribute, transform(copy.deepcopy(original)))
        yield
    finally:
        for path, original in saved.items():
            module_name, attribute = path.split(".")
            setattr(MODULES[module_name], attribute, original)


@dataclass
class RetentionRow:
    perturbation: str
    season: int
    gain: float
    se: float
    retention: float


def run_robustness(
    seasons: Sequence[int] = (2022, 2023),
    *,
    policy: str = "season_sim",
    baseline: str = "need_adp",
    perturbations: Optional[Sequence[str]] = None,
    n_replicates: int = 48,
    agent_kwargs: Optional[Dict] = None,
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Edge retention for ``policy`` under each perturbation.

    Nothing is retrained: the point is to ask whether an edge measured under one
    set of simulator constants survives a different set.
    """
    names = list(perturbations) if perturbations else list(PERTURBATIONS)
    agent_kwargs = agent_kwargs or {"n_candidates": 8, "n_rollouts": 2}

    def gain_for(name: str, season: int) -> tuple:
        out_path = (
            None if out_dir is None
            else Path(out_dir) / f"robust_{season}_{name}.csv"
        )
        with perturbed(name):
            frame = replay_season(
                season,
                policies=[baseline, policy],
                n_replicates=n_replicates,
                agent_kwargs=agent_kwargs,
                out_path=out_path,
                flush_every=4,
                verbose=False,
            )
        table = summarize(frame, baseline=baseline).set_index("policy")
        return (
            float(table.loc[policy, f"rank_gain_vs_{baseline}"]),
            float(table.loc[policy].get("se", np.nan)),
        )

    rows: List[RetentionRow] = []
    for season in seasons:
        base_gain, base_se = gain_for("baseline", season)
        if verbose:
            print(f"  {season} baseline gain {base_gain:+.3f} (se {base_se:.3f})")
        rows.append(RetentionRow("baseline", season, base_gain, base_se, 1.0))

        for name in names:
            gain, se = gain_for(name, season)
            # Retention is only meaningful when there is an edge to retain.
            retention = gain / base_gain if abs(base_gain) > 1e-9 else np.nan
            rows.append(RetentionRow(name, season, gain, se, retention))
            if verbose:
                print(
                    f"    {name:<20} gain {gain:+.3f} (se {se:.3f})  "
                    f"retention {retention:+.2f}"
                )

    return pd.DataFrame([r.__dict__ for r in rows])


def verdict(frame: pd.DataFrame) -> pd.DataFrame:
    """Pool across seasons and apply the pre-committed pass criteria."""
    pooled = (
        frame[frame["perturbation"] != "baseline"]
        .groupby("perturbation")
        .agg(gain=("gain", "mean"), retention=("retention", "mean"))
        .reset_index()
        .sort_values("retention")
    )
    pooled["survives"] = pooled["gain"] > 0
    return pooled


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument("--policy", default="season_sim")
    parser.add_argument("--baseline", default="need_adp")
    parser.add_argument("--replicates", type=int, default=48)
    parser.add_argument("--perturbations", nargs="+", default=None)
    parser.add_argument("--out-dir", default="data/processed/robustness")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    frame = run_robustness(
        args.seasons,
        policy=args.policy,
        baseline=args.baseline,
        perturbations=args.perturbations,
        n_replicates=args.replicates,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )

    table = verdict(frame)
    print("\nedge retention, pooled across seasons\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    median = float(table["retention"].median())
    all_survive = bool(table["survives"].all())
    print(f"\n  median retention {median:.2f} (want >= 0.60): "
          f"{'PASS' if median >= 0.60 else 'FAIL'}")
    print(f"  every perturbation keeps a positive gain: "
          f"{'PASS' if all_survive else 'FAIL'}")
    if not all_survive:
        killed = table[~table["survives"]]["perturbation"].tolist()
        print(f"    edge dies under: {killed}")
    print("\n  'flex_hindsight' is the positive control -- if it does NOT move")
    print("  the gain, this test is too weak to trust on the others.")

    if args.out:
        frame.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
