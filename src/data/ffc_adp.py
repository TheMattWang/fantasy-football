"""Average draft position from Fantasy Football Calculator.

Why this exists
---------------
The opponent model perturbs consensus rank to decide where a player actually
goes::

    noisy_adp = adp + N(0, sigma)

and ``sigma`` was being read from FantasyPros' ``ecr_sd`` -- the dispersion of
*expert opinion*. That is the wrong quantity. What the model needs is the
dispersion of *draft position*: how far from consensus a player really goes in
real drafts, which is a fact about drafters, not about analysts. Experts can
agree closely on a player that drafters reach for wildly, and vice versa.

FFC publishes exactly the right thing, aggregated over real drafts, filtered to
a league shape::

    GET /api/v1/adp/half-ppr?teams=12&year=2025
    -> meta: {type, teams, rounds, total_drafts, start_date, end_date}
       players: [{name, position, adp, stdev, high, low, times_drafted, bye}, ...]

Coverage caveat
---------------
FFC lists ~120-200 players per season while a 12-team 15-round draft makes 180
picks, so the tail of the board is not covered. Players past the listed range
keep the existing ``ecr_sd`` fallback -- they are late-round picks whose exact
dispersion barely matters. ``attach_market_dispersion`` reports the coverage it
achieved rather than silently filling.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .paths import cache_dir, ensure

FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{scoring}"
USER_AGENT = "fantasy-draft-agent/2.0 (personal league tooling)"

# FFC's scoring slugs, keyed by points per reception.
SCORING_SLUGS: Dict[float, str] = {0.0: "standard", 0.5: "half-ppr", 1.0: "ppr"}

COLUMNS = [
    "name_key", "player_name", "position", "team",
    "adp", "stdev", "high", "low", "times_drafted", "bye",
]


def scoring_slug(points_per_reception: Optional[float]) -> str:
    """FFC slug for a league's PPR setting, defaulting to half-PPR."""
    if points_per_reception is None:
        return "half-ppr"
    ppr = float(points_per_reception)
    nearest = min(SCORING_SLUGS, key=lambda k: abs(k - ppr))
    return SCORING_SLUGS[nearest]


def ffc_dir():
    return ensure(cache_dir() / "ffc")


def cache_path(season: int, scoring: str, teams: int):
    return ffc_dir() / f"adp_{scoring}_{teams}team_{season}.csv"


def fetch_adp(
    season: int,
    *,
    scoring: str = "half-ppr",
    teams: int = 12,
    refresh: bool = False,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Real-draft ADP and dispersion for one season, cached to ``FF_CACHE_DIR``.

    Falls back to the cache when the network is unavailable, so nothing here can
    stall a draft-day board build.
    """
    from ..projections.ecr import normalize_name

    path = cache_path(season, scoring, teams)
    if path.exists() and not refresh:
        return pd.read_csv(path)

    url = FFC_URL.format(scoring=scoring) + f"?teams={teams}&year={season}"
    # FFC 403s the default python-urllib user agent.
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if path.exists():
            warnings.warn(
                f"FFC fetch for {season} failed ({exc}); using the cache at {path}",
                stacklevel=2,
            )
            return pd.read_csv(path)
        raise RuntimeError(f"could not fetch FFC ADP for {season}: {exc}") from exc

    players: List[Dict] = payload.get("players") or []
    if not players:
        raise RuntimeError(
            f"FFC returned no players for {season} "
            f"(status={payload.get('status')!r}) -- check the season is published"
        )

    frame = pd.DataFrame(players)
    frame["name_key"] = frame["name"].map(normalize_name)
    frame = frame.rename(columns={"name": "player_name"})
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[COLUMNS]

    meta = payload.get("meta", {})
    ensure(path.parent)
    frame.to_csv(path, index=False)
    print(
        f"  FFC {season} {meta.get('type', scoring)} {meta.get('teams', teams)}-team: "
        f"{len(frame)} players over {meta.get('total_drafts', '?')} drafts -> {path}"
    )
    return frame


def attach_market_dispersion(
    board: pd.DataFrame,
    season: int,
    *,
    scoring: str = "half-ppr",
    teams: int = 12,
    min_times_drafted: int = 10,
    refresh: bool = False,
) -> pd.DataFrame:
    """Add ``adp_sd`` (and ``ffc_adp``) to a board, from real-draft dispersion.

    Rows FFC does not cover keep ``adp_sd`` as NA, and ``DraftBoard.from_frame``
    applies its existing fallback to those.
    """
    from ..projections.ecr import normalize_name

    market = fetch_adp(season, scoring=scoring, teams=teams, refresh=refresh)

    # A player drafted a handful of times has a dispersion estimate that is
    # mostly noise; better to fall back than to trust it.
    counts = pd.to_numeric(market["times_drafted"], errors="coerce").fillna(0)
    market = market[counts >= min_times_drafted]

    lookup = market.drop_duplicates("name_key").set_index("name_key")
    frame = board.reset_index(drop=True).copy()
    keys = frame["player_name"].map(normalize_name)

    frame["ffc_adp"] = keys.map(lookup["adp"]).astype(float)
    frame["adp_sd"] = keys.map(lookup["stdev"]).astype(float)

    matched = int(frame["adp_sd"].notna().sum())
    drafted = teams * 15
    top = frame.nsmallest(drafted, "adp_rank") if "adp_rank" in frame else frame
    top_matched = int(top["adp_sd"].notna().sum())

    # Unmatched rows must NOT fall back to ecr_sd. Measured on 2025, ecr_sd runs
    # ~2x the real draft dispersion in every ADP bucket, so falling back to it
    # would leave part of the board twice as random as real drafters -- which is
    # what makes elite players slide in simulation and makes waiting look free.
    # Fit the observed relationship instead and extend it over the gaps.
    fitted = _fit_dispersion(frame)
    if fitted is not None:
        gap = frame["adp_sd"].isna()
        frame.loc[gap, "adp_sd"] = fitted[gap]

    print(
        f"  market dispersion: {matched}/{len(frame)} board rows matched; "
        f"{top_matched}/{len(top)} ({top_matched / max(len(top), 1):.0%}) "
        f"of the {drafted} that get drafted"
        + ("" if fitted is None else f"; {int(frame['adp_sd'].isna().sum())} left unfilled")
    )
    return frame


def _fit_dispersion(frame: pd.DataFrame) -> Optional[pd.Series]:
    """Dispersion as a function of ADP, fit on the rows FFC does cover.

    Spread grows roughly linearly in rank (~2 picks at the top of round 1, ~16
    past pick 120), so a straight line through the matched rows extends the real
    scale over the players FFC does not list.
    """
    if "adp_rank" not in frame.columns:
        return None
    adp = pd.to_numeric(frame["adp_rank"], errors="coerce")
    sd = pd.to_numeric(frame["adp_sd"], errors="coerce")
    fit = adp.notna() & sd.notna() & (sd > 0)
    if int(fit.sum()) < 20:
        return None

    slope, intercept = np.polyfit(adp[fit].to_numpy(), sd[fit].to_numpy(), 1)
    predicted = intercept + slope * adp.fillna(adp.max())
    return predicted.clip(lower=1.0)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+",
                        default=[2022, 2023, 2024, 2025])
    parser.add_argument("--scoring", default="half-ppr",
                        choices=sorted(set(SCORING_SLUGS.values())))
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    for season in args.seasons:
        frame = fetch_adp(
            season, scoring=args.scoring, teams=args.teams, refresh=args.refresh
        )
        sd = pd.to_numeric(frame["stdev"], errors="coerce")
        adp = pd.to_numeric(frame["adp"], errors="coerce")
        early = sd[adp <= 24].median()
        late = sd[adp > 100].median()
        print(
            f"    median stdev: {early:.1f} picks in the top 24, "
            f"{late:.1f} past pick 100"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
