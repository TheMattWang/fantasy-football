"""The ECR baseline curve: what the market's ranking is actually worth.

The idea
--------
Rather than projecting a player from his own past stats -- which is how the 2025
board ended up loving Rashee Rice (3 games) at ADP 64 -- start from where the
market ranks him and ask an empirical question:

    historically, what has the RB ranked 14th in preseason consensus
    actually scored?

Fit that over several seasons and you get a projection that has regression to
the mean, aging, role change and injury risk already priced in, because the
consensus priced them. It is the single highest-value component of the rebuild,
and it is mostly one isotonic regression per position.

Two curves, not one
-------------------
``ppg`` is scoring rate *while active*; ``games`` is expected active games. Both
are conditioned on preseason rank, so the market's injury discount is captured
exactly once -- in the games curve. Fitting rate-when-healthy and then
multiplying by a separately-estimated availability would double-count it.

Data
----
FantasyPros ECR history via DynastyProcess (``db_fpecr.parquet``, ~1.8M rows,
2019-12 to present), which carries ``ecr``, ``sd``, ``best`` and ``worst`` --
the dispersion columns Phase 5's opponent model needs. Only PPR variants are
published; that is fine, since the curve maps *consensus rank* to *actual points
under our league's scoring*, and rank is what the market supplies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.nflverse import season_rates
from ..data.paths import cache_dir, ensure

ECR_URL = (
    "https://github.com/dynastyprocess/data/raw/master/files/db_fpecr.parquet"
)
ECR_CACHE = "db_fpecr.parquet"

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

# Preseason = the last consensus snapshot before week 1 kicks off.
PRESEASON_MONTHS = (7, 8, 9)
PRESEASON_LAST_DAY_IN_SEPT = 7

# The top of the board is what a draft actually spends picks on; a poor match
# rate there is a broken join, not a data limitation.
MIN_TOP_MATCH_RATE = 0.90
TOP_N_FOR_MATCH_GATE = 150

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")


class EcrJoinError(RuntimeError):
    """Raised when ECR cannot be matched to actual results well enough to fit."""


def normalize_name(name: object) -> str:
    """Normalize a player name for joining across sources.

    Lowercase, strip punctuation and generational suffixes, squash whitespace,
    then glue runs of single letters back together so that initials agree
    however a source punctuates them::

        "A.J. Brown" / "AJ Brown"    -> "aj brown"
        "D.K. Metcalf" / "DK Metcalf" -> "dk metcalf"

    That case is common enough in fantasy data to matter: sources disagree on
    punctuating initials, and each mismatch drops a real player off the board.

    Otherwise deliberately conservative -- better to miss a match and see it in
    the match-rate report than to silently fuse two different players.
    """
    text = _NON_ALPHA.sub(" ", str(name).lower())
    text = _SUFFIX.sub(" ", text)

    tokens = text.split()
    merged: List[str] = []
    for token in tokens:
        if len(token) == 1 and merged and len(merged[-1]) <= 2 and merged[-1].isalpha():
            merged[-1] += token
        else:
            merged.append(token)
    return " ".join(merged).strip()


def load_ecr_history(*, refresh: bool = False) -> pd.DataFrame:
    """Load the FantasyPros ECR history, caching the parquet locally."""
    path = ensure(cache_dir()) / ECR_CACHE
    if refresh or not path.exists():
        frame = pd.read_parquet(ECR_URL)
        frame.to_parquet(path, index=False)
    else:
        frame = pd.read_parquet(path)
    frame["scrape_date"] = pd.to_datetime(frame["scrape_date"])
    return frame


def preseason_snapshot(
    season: int,
    *,
    history: Optional[pd.DataFrame] = None,
    page_type: str = "redraft-overall",
    positions: Sequence[str] = SKILL_POSITIONS,
) -> pd.DataFrame:
    """The last preseason consensus ranking for ``season``.

    Returns one row per player with ``ecr``, ``sd``, ``best``, ``worst``, an
    overall ``ecr_rank`` and a within-position ``pos_rank``.
    """
    history = load_ecr_history() if history is None else history

    frame = history[history["page_type"] == page_type]
    frame = frame[frame["scrape_date"].dt.year == season]
    frame = frame[frame["scrape_date"].dt.month.isin(PRESEASON_MONTHS)]
    frame = frame[
        ~(
            (frame["scrape_date"].dt.month == 9)
            & (frame["scrape_date"].dt.day > PRESEASON_LAST_DAY_IN_SEPT)
        )
    ]
    if frame.empty:
        raise EcrJoinError(
            f"no preseason {page_type!r} consensus found for {season}. "
            f"History covers "
            f"{history['scrape_date'].min().date()}..{history['scrape_date'].max().date()}."
        )

    snapshot = frame[frame["scrape_date"] == frame["scrape_date"].max()].copy()
    snapshot = snapshot[snapshot["pos"].isin(positions)]
    snapshot = snapshot.dropna(subset=["ecr"]).sort_values("ecr")

    # A player can appear twice if FantasyPros re-listed him mid-scrape.
    snapshot = snapshot.drop_duplicates(subset=["player", "pos"], keep="first")

    snapshot["season"] = season
    snapshot["name_key"] = snapshot["player"].map(normalize_name)
    snapshot["ecr_rank"] = np.arange(1, len(snapshot) + 1)
    snapshot["pos_rank"] = snapshot.groupby("pos")["ecr"].rank(method="first").astype(int)

    keep = [
        "season", "player", "name_key", "pos", "team", "ecr", "sd", "best",
        "worst", "ecr_rank", "pos_rank", "scrape_date",
    ]
    return snapshot[[c for c in keep if c in snapshot.columns]].reset_index(drop=True)


def join_actuals(
    season: int,
    *,
    scoring: Optional[Dict[str, float]] = None,
    history: Optional[pd.DataFrame] = None,
    actuals: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join a preseason consensus snapshot to what actually happened.

    One row per ranked player, carrying ``pos_rank`` (the input) alongside
    ``ppg_available``, ``games_active`` and ``total_points`` (the outcomes).
    """
    snapshot = preseason_snapshot(season, history=history)

    if actuals is None:
        actuals = season_rates([season], scoring)
    actuals = actuals[actuals["position"].isin(SKILL_POSITIONS)].copy()
    actuals["name_key"] = actuals["player_name"].map(normalize_name)
    actuals = actuals.drop_duplicates(subset=["name_key", "position"], keep="first")

    merged = snapshot.merge(
        actuals,
        how="left",
        left_on=["name_key", "pos"],
        right_on=["name_key", "position"],
        suffixes=("", "_actual"),
    )

    # Players with no stat line at all did not play; that is a real outcome
    # (zero), not a missing value -- but only for players the market ranked
    # highly enough that we would have considered drafting them.
    merged["matched"] = merged["total_points"].notna()
    return merged


def _match_rate(merged: pd.DataFrame, top_n: int = TOP_N_FOR_MATCH_GATE) -> float:
    top = merged.nsmallest(min(top_n, len(merged)), "ecr_rank")
    return float(top["matched"].mean()) if len(top) else 0.0


class MonotoneCurve:
    """A smooth, non-increasing map from consensus rank to an outcome.

    Raw isotonic regression on this data pools hard: with ~5 observations per
    (position, rank) it emits long flat runs, so six different RBs come out with
    an identical projection and the board cannot tell McCaffrey from the RB6.

    So fit a smooth trend in log(rank) first -- production decays roughly
    log-linearly in draft rank -- and only then project onto the monotone cone.
    That keeps the guarantee (later rank is never worth more) while preserving
    resolution between adjacent players.
    """

    def __init__(self, ranks, values, *, max_rank: int = 300, degree: int = 2):
        from sklearn.isotonic import IsotonicRegression

        ranks = np.asarray(ranks, dtype=float)
        values = np.asarray(values, dtype=float)

        # Average the observations at each rank, and weight by how many there
        # were, so a rank seen in five seasons outvotes one seen once.
        frame = pd.DataFrame({"rank": ranks, "value": values})
        agg = frame.groupby("rank")["value"].agg(["mean", "size"]).reset_index()

        x = np.log(agg["rank"].to_numpy())
        y = agg["mean"].to_numpy()
        w = agg["size"].to_numpy(dtype=float)

        coefficients = np.polyfit(x, y, deg=min(degree, max(1, len(agg) - 1)), w=w)
        trend = np.polyval(coefficients, x)

        # Project the smooth trend onto "non-increasing in rank".
        monotone = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
            agg["rank"].to_numpy(), trend, sample_weight=w
        )

        self.grid = np.arange(1, max_rank + 1, dtype=float)
        fitted = monotone.predict(self.grid)
        # Production cannot be negative, and the polynomial can dive below zero
        # far past the last observed rank.
        self.fitted = np.maximum(fitted, 0.0)
        self.max_observed_rank = float(agg["rank"].max())

    def predict(self, ranks) -> np.ndarray:
        ranks = np.atleast_1d(np.asarray(ranks, dtype=float))
        return np.interp(ranks, self.grid, self.fitted)


@dataclass
class BaselineCurve:
    """Empirical map from preseason consensus rank to expected production.

    ``ppg[pos]`` and ``games[pos]`` are monotone-decreasing isotonic fits over
    within-position consensus rank.
    """

    ppg: Dict[str, object] = field(default_factory=dict)
    games: Dict[str, object] = field(default_factory=dict)
    seasons: List[int] = field(default_factory=list)
    match_rates: Dict[int, float] = field(default_factory=dict)
    n_observations: Dict[str, int] = field(default_factory=dict)

    def ppg_at(self, position: str, pos_rank: float) -> float:
        """Expected points per active game for the Nth-ranked player at a position."""
        model = self.ppg.get(position)
        if model is None:
            return float("nan")
        return float(model.predict([float(pos_rank)])[0])

    def games_at(self, position: str, pos_rank: float) -> float:
        """Expected active games for the Nth-ranked player at a position."""
        model = self.games.get(position)
        if model is None:
            return float("nan")
        return float(model.predict([float(pos_rank)])[0])

    def season_points_at(self, position: str, pos_rank: float) -> float:
        """Expected full-season points -- rate times availability."""
        return self.ppg_at(position, pos_rank) * self.games_at(position, pos_rank)

    def table(self, position: str, ranks: Iterable[int]) -> pd.DataFrame:
        rows = [
            {
                "pos": position,
                "pos_rank": r,
                "ppg": self.ppg_at(position, r),
                "games": self.games_at(position, r),
                "season_points": self.season_points_at(position, r),
            }
            for r in ranks
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            f"BaselineCurve over {self.seasons} "
            f"({sum(self.n_observations.values())} player-seasons)"
        ]
        for season, rate in sorted(self.match_rates.items()):
            lines.append(f"  {season} top-{TOP_N_FOR_MATCH_GATE} join: {rate:.0%}")
        for pos in SKILL_POSITIONS:
            if pos in self.ppg:
                lines.append(
                    f"  {pos}: n={self.n_observations.get(pos, 0):4d}  "
                    f"rank1={self.ppg_at(pos, 1):5.1f} ppg  "
                    f"rank12={self.ppg_at(pos, 12):5.1f}  "
                    f"rank36={self.ppg_at(pos, 36):5.1f}"
                )
        return "\n".join(lines)


def fit_baseline(
    seasons: Sequence[int],
    *,
    scoring: Optional[Dict[str, float]] = None,
    history: Optional[pd.DataFrame] = None,
    actuals_by_season: Optional[Dict[int, pd.DataFrame]] = None,
    enforce_match_gate: bool = True,
) -> BaselineCurve:
    """Fit the consensus-rank -> production curves over ``seasons``.

    Args:
        seasons: seasons whose *outcomes* train the curve. Exclude any season
            you intend to evaluate on.
        scoring: canonical scoring dict; defaults to the placeholder half-PPR.
        actuals_by_season: pre-loaded results per season, bypassing the nflverse
            fetch. Used by tests and by callers that already hold the panel.
        enforce_match_gate: raise if the top-of-board join rate is poor. Leave
            this on -- a silent join failure is what produced a board with 43%
            ADP coverage.
    """
    from sklearn.isotonic import IsotonicRegression

    history = load_ecr_history() if history is None else history
    actuals_by_season = actuals_by_season or {}

    frames: List[pd.DataFrame] = []
    match_rates: Dict[int, float] = {}
    for season in seasons:
        merged = join_actuals(
            season,
            scoring=scoring,
            history=history,
            actuals=actuals_by_season.get(season),
        )
        rate = _match_rate(merged)
        match_rates[season] = rate
        if enforce_match_gate and rate < MIN_TOP_MATCH_RATE:
            raise EcrJoinError(
                f"{season}: only {rate:.0%} of the top {TOP_N_FOR_MATCH_GATE} "
                f"consensus players matched to actual results (want "
                f">={MIN_TOP_MATCH_RATE:.0%}). This is a name-join failure -- "
                f"fix it rather than fitting on a biased subset."
            )
        frames.append(merged[merged["matched"]])

    pooled = pd.concat(frames, ignore_index=True)

    curve = BaselineCurve(seasons=list(seasons), match_rates=match_rates)
    for pos in SKILL_POSITIONS:
        subset = pooled[pooled["pos"] == pos].dropna(subset=["pos_rank"])
        subset = subset[subset["pos_rank"] > 0]
        if len(subset) < 20:
            continue

        ranks = subset["pos_rank"].to_numpy(dtype=float)

        rate = subset["ppg_available"].fillna(subset["ppg_played"]).fillna(0.0)
        games = subset["games_active"].fillna(subset["games_played"]).fillna(0.0)

        # Value must not increase as you go later in the draft, but the curve
        # also has to distinguish adjacent players -- see MonotoneCurve.
        curve.ppg[pos] = MonotoneCurve(ranks, rate.to_numpy(dtype=float))
        curve.games[pos] = MonotoneCurve(ranks, games.to_numpy(dtype=float))
        curve.n_observations[pos] = len(subset)

    if not curve.ppg:
        raise EcrJoinError("no position had enough observations to fit")
    return curve


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m src.projections.ecr`` -- fit and show the curve."""
    import argparse

    parser = argparse.ArgumentParser(description="Fit the ECR baseline curve")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024])
    parser.add_argument("--show", type=int, nargs="+", default=[1, 3, 6, 12, 24, 36, 48])
    args = parser.parse_args(argv)

    curve = fit_baseline(args.seasons)
    print(curve.summary())
    print()
    for pos in SKILL_POSITIONS:
        if pos not in curve.ppg:
            continue
        table = curve.table(pos, args.show)
        print(table.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
