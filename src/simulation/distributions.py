"""Per-player outcome distributions, materialized as one sample tensor.

Why distributions at all
------------------------
With a point estimate per player and no outcome stochasticity, searching over an
additive objective is just a knapsack solve over a biased scalar -- it inherits
all of the projection's bias and adds nothing. Uncertainty is what makes
drafting a decision problem: it is why you take the safe RB over the boom/bust
one when you are already favoured, and the reverse when you are not.

Three stacked layers
--------------------
1. **Season-level projection error.** The consensus is wrong by a lot, and
   log-normally: ``mu_s = mu * exp(N(0, sigma_proj))``.

2. **Availability, as a two-state Markov chain.** Measured over 2021-2025
   regular-season roster status:

       P(active | active last week) = 0.914
       P(active | out last week)    = 0.126   -> ~8 week absence once out

   That persistence is the whole point. A week-3 season-ender and eight
   scattered one-week absences have the same expected games and completely
   different playoff implications; i.i.d. Bernoulli cannot tell them apart.

3. **Weekly points given active,** Gamma rather than Normal -- fantasy weeks are
   non-negative and right-skewed. Coefficient of variation measured by position
   and projected tier (see :data:`WEEKLY_CV`); it falls as projection rises,
   which is why quarterbacks are the low-variance position and why a point
   estimate cannot see that.

The output is a single ``float32[samples, players, weeks]`` array -- 54 MB at
2000 x 450 x 15 -- so every downstream operation is numpy slicing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Measured: P(active this week | out last week), 2021-2025 REG weeks.
P_RETURN_AFTER_OUT = 0.126

# Median weekly coefficient of variation, by position and mean-ppg tier.
# Measured over 2021-2025 for players with >= 8 games and > 2 ppg.
WEEKLY_CV: Dict[str, List[Tuple[float, float]]] = {
    #          (mean ppg upper bound, cv)
    "QB": [(6, 2.36), (10, 0.91), (14, 0.55), (18, 0.43), (np.inf, 0.38)],
    "RB": [(6, 1.02), (10, 0.73), (14, 0.57), (18, 0.54), (np.inf, 0.43)],
    "WR": [(6, 1.02), (10, 0.73), (14, 0.60), (18, 0.55), (np.inf, 0.45)],
    "TE": [(6, 0.91), (10, 0.69), (14, 0.64), (18, 0.50), (np.inf, 0.50)],
}
DEFAULT_CV = 0.80

# Spread of log(actual / consensus-implied) *rate*, fitted by
# calibrate_projection_error() over 2022-2025 walk-forward residuals at
# draftable ranks. Because ppg_available already conditions on the player being
# active, this layer is rate uncertainty only -- missed games are the Markov
# layer's job, and folding them in here would double-count.
#
# The ordering is the football-sensible one: quarterbacks are the most
# predictable position and running backs the least.
SIGMA_PROJ: Dict[str, float] = {"QB": 0.21, "RB": 0.38, "WR": 0.29, "TE": 0.29}
DEFAULT_SIGMA_PROJ = 0.32

# Streamed positions get a flat, low-variance weekly line: you are always
# playing somebody, and who it is barely moves the season.
STREAMED_PPG = {"K": 8.0, "DEF": 7.0, "DST": 7.0}
STREAMED_CV = 0.55


def cv_for(position: str, mean_ppg: float) -> float:
    """Weekly coefficient of variation for a player at this projection level."""
    table = WEEKLY_CV.get(position)
    if table is None:
        return STREAMED_CV if position in STREAMED_PPG else DEFAULT_CV
    for upper, cv in table:
        if mean_ppg < upper:
            return cv
    return table[-1][1]


def stay_probability(
    available_rate: float, p_return: float = P_RETURN_AFTER_OUT
) -> float:
    """P(active next week | active now) implying a target availability rate.

    Holds the *recovery* rate fixed at its measured value and solves the
    stationary equation for the persistence rate, so a player's expected games
    comes from the consensus curve while the shape of an absence comes from
    data::

        p = p_return / (1 - p_stay + p_return)
    """
    rate = float(np.clip(available_rate, 0.05, 0.999))
    return float(np.clip(1.0 - p_return * (1.0 - rate) / rate, 0.0, 0.9995))


@dataclass
class SampleSet:
    """Sampled weekly fantasy points for every player on the board.

    Attributes:
        points: ``float32[n_samples, n_players, n_weeks]`` -- zero where the
            player was unavailable or on bye.
        players: board rows in the same order as axis 1.
        index: player name -> row position, for building rosters.
    """

    points: np.ndarray
    players: pd.DataFrame
    index: Dict[str, int] = field(default_factory=dict)
    active: Optional[np.ndarray] = None
    decision_score: Optional[np.ndarray] = None
    """Per-player value used to SET lineups, i.e. what you knew beforehand.

    Lineups must be chosen ex ante. Starting the players who turned out to score
    most inflates every roster and flattens the difference between a deep team
    and a top-heavy one -- the exact comparison the draft is trying to make.
    """

    @property
    def n_samples(self) -> int:
        return self.points.shape[0]

    @property
    def n_weeks(self) -> int:
        return self.points.shape[2]

    def rows_for(self, names: Sequence[str]) -> np.ndarray:
        """Row positions for a roster, skipping names not on the board."""
        return np.array(
            [self.index[n] for n in names if n in self.index], dtype=np.intp
        )

    def nbytes(self) -> int:
        return int(self.points.nbytes)

    def summary(self) -> str:
        return (
            f"SampleSet {self.points.shape} "
            f"({self.nbytes() / 1e6:.1f} MB) -- "
            f"{len(self.players)} players, {self.n_weeks} weeks, "
            f"{self.n_samples} season samples"
        )


def _sample_availability(
    rng: np.random.Generator,
    stay: np.ndarray,
    n_samples: int,
    n_weeks: int,
    p_return: float,
) -> np.ndarray:
    """Two-state Markov availability paths, shape (samples, players, weeks)."""
    n_players = stay.shape[0]
    active = np.empty((n_samples, n_players, n_weeks), dtype=bool)

    # Start each season in the stationary distribution rather than "everyone
    # healthy in week 1" -- some players begin the year already hurt.
    stationary = p_return / (1.0 - stay + p_return)
    active[:, :, 0] = rng.random((n_samples, n_players)) < stationary[None, :]

    for week in range(1, n_weeks):
        threshold = np.where(active[:, :, week - 1], stay[None, :], p_return)
        active[:, :, week] = rng.random((n_samples, n_players)) < threshold
    return active


def build_samples(
    board: pd.DataFrame,
    *,
    n_samples: int = 2000,
    n_weeks: int = 15,
    seed: int = 0,
    include_projection_error: bool = True,
    bye_weeks: Optional[Dict[str, int]] = None,
) -> SampleSet:
    """Materialize the sample tensor for a board.

    Args:
        board: output of :func:`src.projections.board.build_board`.
        n_samples: season replicates. 200 is enough to rank candidate picks
            under common random numbers; 2000 for a final decision.
        n_weeks: regular-season weeks to simulate.
        seed: RNG seed. Reuse it across candidate evaluations -- common random
            numbers cut the samples needed to rank options by 5-10x.
        include_projection_error: apply the season-level lognormal miss. Turn
            off only to inspect the availability/weekly layers in isolation.
        bye_weeks: player name -> bye week (1-indexed). Byes are just
            availability 0 for that week.
    """
    rng = np.random.default_rng(seed)

    frame = board.reset_index(drop=True)
    positions = frame["position"].to_numpy()
    names = frame["player_name"].astype(str).to_numpy()

    mu = frame["proj_ppg"].to_numpy(dtype=float).copy()
    games = frame.get("proj_games")
    games = (
        games.to_numpy(dtype=float).copy()
        if games is not None
        else np.full(len(frame), float(n_weeks))
    )

    # Streamed positions are not projected by the baseline curve; give them a
    # flat replacement-level line so a roster slot filled by streaming still
    # scores something.
    for i, pos in enumerate(positions):
        if pos in STREAMED_PPG and mu[i] <= 0:
            mu[i] = STREAMED_PPG[pos]
            games[i] = float(n_weeks)

    mu = np.maximum(mu, 0.01)

    # Expected availability implied by the consensus games curve.
    available_rate = np.clip(games / 17.0, 0.05, 0.999)
    stay = np.array([stay_probability(r) for r in available_rate])

    cv = np.array(
        [cv_for(p, m) for p, m in zip(positions, mu)], dtype=float
    )
    cv = np.clip(cv, 0.15, 3.0)

    n_players = len(frame)

    # Layer 1: season-level projection error.
    if include_projection_error:
        sigma = np.array(
            [SIGMA_PROJ.get(p, DEFAULT_SIGMA_PROJ) for p in positions], dtype=float
        )
        season_mu = mu[None, :] * np.exp(
            rng.normal(0.0, 1.0, size=(n_samples, n_players)) * sigma[None, :]
            - 0.5 * sigma[None, :] ** 2  # keep E[mu_s] = mu
        )
    else:
        season_mu = np.broadcast_to(mu[None, :], (n_samples, n_players)).copy()

    # Layer 3: weekly points | active, Gamma(shape=1/cv^2, scale=mu*cv^2).
    shape = 1.0 / cv**2
    points = rng.gamma(
        shape=shape[None, :, None],
        scale=(season_mu[:, :, None] * (cv**2)[None, :, None]),
        size=(n_samples, n_players, n_weeks),
    )

    # Layer 2: availability.
    active = _sample_availability(rng, stay, n_samples, n_weeks, P_RETURN_AFTER_OUT)

    if bye_weeks:
        for i, name in enumerate(names):
            bye = bye_weeks.get(name)
            if bye and 1 <= int(bye) <= n_weeks:
                active[:, i, int(bye) - 1] = False

    points = np.where(active, points, 0.0).astype(np.float32)

    return SampleSet(
        points=points,
        players=frame,
        index={name: i for i, name in enumerate(names)},
        active=active,
        # What a manager knows when setting the lineup: the preseason
        # projection, not the realized week.
        decision_score=mu.astype(np.float32),
    )


# Only players good enough to be drafted inform the projection-error spread.
# Ranked far enough down the board, both projection and outcome approach zero
# and log(actual/predicted) explodes -- that noise is not draft-relevant.
DRAFTABLE_POS_RANK = {"QB": 24, "RB": 48, "WR": 60, "TE": 24}


def calibrate_projection_error(
    holdouts: Sequence[int] = (2022, 2023, 2024, 2025),
    *,
    robust: bool = True,
) -> Dict[str, float]:
    """Fit ``sigma_proj`` per position from walk-forward consensus residuals.

    Restricted to draftable ranks (see :data:`DRAFTABLE_POS_RANK`) and, by
    default, estimated from the interquartile range rather than the raw standard
    deviation -- a handful of season-ending week-1 injuries otherwise dominates
    the fit and doubles the implied spread for everybody.

    This is a checkable quantity: with the right sigma an 80% predictive
    interval contains ~80% of outcomes. ``main(--check)`` verifies that.
    """
    from ..projections.validate import build_comparison

    residuals: Dict[str, List[float]] = {}
    for holdout in holdouts:
        train = [s for s in range(2021, holdout)]
        if not train:
            continue
        frame = build_comparison(holdout, train)
        frame = frame[(frame["pred_ecr"] > 1.0) & (frame["ppg_available"] > 0)]
        for pos, group in frame.groupby("pos"):
            limit = DRAFTABLE_POS_RANK.get(str(pos))
            if limit is not None:
                group = group[group["pos_rank"] <= limit]
            if group.empty:
                continue
            residuals.setdefault(str(pos), []).extend(
                np.log(group["ppg_available"] / group["pred_ecr"]).tolist()
            )

    fitted: Dict[str, float] = {}
    for pos, values in residuals.items():
        if len(values) < 30:
            continue
        array = np.asarray(values, dtype=float)
        if robust:
            iqr = np.percentile(array, 75) - np.percentile(array, 25)
            fitted[pos] = float(iqr / 1.349)  # IQR -> sd for a normal
        else:
            fitted[pos] = float(array.std())
    return fitted


def coverage_check(
    sigma: Optional[Dict[str, float]] = None,
    holdouts: Sequence[int] = (2022, 2023, 2024, 2025),
    interval: float = 0.80,
) -> pd.DataFrame:
    """Do the predictive intervals actually contain the stated share of outcomes?

    This is the test that distinguishes calibrated uncertainty from decorative
    uncertainty. Target coverage is ``interval``; materially below means the
    model is overconfident.
    """
    from scipy.stats import norm

    from ..projections.validate import build_comparison

    sigma = sigma or SIGMA_PROJ
    z = norm.ppf(0.5 + interval / 2)

    rows = []
    for holdout in holdouts:
        train = [s for s in range(2021, holdout)]
        if not train:
            continue
        frame = build_comparison(holdout, train)
        frame = frame[(frame["pred_ecr"] > 1.0) & (frame["ppg_available"] > 0)]
        for pos, group in frame.groupby("pos"):
            limit = DRAFTABLE_POS_RANK.get(str(pos))
            if limit is not None:
                group = group[group["pos_rank"] <= limit]
            if group.empty:
                continue
            s = sigma.get(str(pos), DEFAULT_SIGMA_PROJ)
            lo = group["pred_ecr"] * np.exp(-z * s)
            hi = group["pred_ecr"] * np.exp(z * s)
            inside = (group["ppg_available"] >= lo) & (group["ppg_available"] <= hi)
            rows.append(
                {
                    "season": holdout,
                    "pos": pos,
                    "n": len(group),
                    "sigma": s,
                    "coverage": float(inside.mean()),
                }
            )
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m src.simulation.distributions`` -- build and sanity-check."""
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description="Build the sample tensor")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--weeks", type=int, default=15)
    parser.add_argument("--calibrate", action="store_true",
                        help="refit sigma_proj from walk-forward residuals")
    parser.add_argument("--check", action="store_true",
                        help="verify predictive-interval coverage")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")

    if args.calibrate:
        fitted = calibrate_projection_error()
        print("fitted sigma_proj (robust, draftable ranks only):")
        for pos, value in sorted(fitted.items()):
            print(f"  {pos:4s} {value:.3f}   (module default "
                  f"{SIGMA_PROJ.get(pos, DEFAULT_SIGMA_PROJ):.2f})")
        return 0

    if args.check:
        table = coverage_check()
        print("80% predictive-interval coverage (target 0.80):\n")
        pivot = table.pivot_table(
            index="pos", columns="season", values="coverage", observed=True
        )
        print(pivot.round(2).to_string())
        print(f"\n  overall: {table['coverage'].mean():.2f}")
        print("  materially below 0.80 means the model is overconfident.")
        return 0

    from ..projections.board import build_board

    board = build_board(args.season)
    samples = build_samples(board, n_samples=args.samples, n_weeks=args.weeks)
    print(samples.summary())

    frame = board.copy()
    totals = samples.points.sum(axis=2)
    frame["sim_mean"] = totals.mean(axis=0)
    frame["sim_p10"] = np.percentile(totals, 10, axis=0)
    frame["sim_p90"] = np.percentile(totals, 90, axis=0)
    frame["weeks_active"] = (samples.points > 0).sum(axis=2).mean(axis=0)

    cols = ["player_name", "position", "proj_points", "sim_mean",
            "sim_p10", "sim_p90", "weeks_active"]
    print("\ntop of board -- simulated season totals:")
    print(frame.head(12)[cols].to_string(index=False,
                                         float_format=lambda v: f"{v:8.1f}"))

    # CV must be measured WITHIN a player-season. Pooling players with different
    # means inflates it, and pooling across samples adds the projection-error
    # layer on top -- neither is the weekly CV being targeted.
    print("\nrealized within-player weekly CV (target = WEEKLY_CV):")
    points = samples.points
    for pos in ("QB", "RB", "WR", "TE"):
        rows = np.where(frame["position"].to_numpy() == pos)[0][:60]
        if not rows.size:
            continue
        realized, targets = [], []
        for row in rows:
            weeks = points[:200, row, :]
            for season in weeks:
                played = season[season > 0]
                if played.size >= 8 and played.mean() > 2:
                    realized.append(played.std() / played.mean())
            targets.append(cv_for(pos, float(frame["proj_ppg"].iloc[row])))
        if realized:
            print(
                f"  {pos}: realized {np.median(realized):.2f}  "
                f"target {np.median(targets):.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
