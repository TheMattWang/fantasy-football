"""Who is actually playing this week.

The start/sit tool ranked players purely on how well they had been scoring, with
no notion of whether they would take the field. That is not an edge problem, it
is a correctness problem, and it is the largest single loss available in-season:
a player on bye or ruled out scores exactly zero, and starting one costs his
whole weekly output against a team total of roughly 107 points. One such mistake
wipes out more than the entire frozen-to-reactive upgrade earns (+1.523 ranks).

The multipliers below are MEASURED, not chosen. Every hand-written constant in
this project has eventually turned out to be wrong -- the waiver floors, the
`ecr_sd` dispersion that ran 2x too wide -- so these come from 6,783 player-week
injury reports over 2022-2025, and `experiments/worker_availability.py`
reproduces them.

    status          n     P(played)   mean pts | played
    Out          1342         0.003                0.00
    Doubtful      197         0.010                1.30
    Questionable 1717         0.567                6.95
    (no report)  3525         0.840                8.66

Two things in that table are worth reading twice.

**Out means out.** A 0.3% play rate is not a discount, it is a zero. Same for
Doubtful at 1.0%.

**Questionable is a coin flip that also underperforms.** 56.7% play, and when
they do play they score 6.95 against 8.66 for a player with no report -- so the
expected value is P(played) x points-when-played, relative to no designation,
which the script computes as **0.456**. Ranking a Questionable player at his
full rate overstates him by more than a factor of two, which is easily enough to
start the wrong man.

The multipliers below are copied from that script's output rather than derived
by hand here, so the two cannot drift apart.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..projections.ecr import normalize_name

# Measured 2026-08-16 over 2022-2025 regular seasons; see module docstring.
# Expected value relative to a player carrying no injury report:
#     P(played | status) * mean_points(status) / mean_points(no report)
STATUS_MULTIPLIER: Dict[str, float] = {
    "Out": 0.000,          # measured 0.000 -- a 0.003 play rate at 0.00 points
    "Doubtful": 0.000,     # measured 0.002 -- rounded down; it is not startable
    "Questionable": 0.456,
}

# A bye is certain, unlike an injury report. No estimation involved.
BYE_MULTIPLIER = 0.0

# Below this, a player is not startable at all rather than merely discounted.
UNSTARTABLE_BELOW = 0.05


def injury_report(season: int, week: int) -> pd.DataFrame:
    """This week's injury designations, one row per player.

    Returns ``name_key`` and ``report_status``. Empty frame rather than an
    exception when the feed has nothing for that week -- an unavailable report
    must degrade to "no information", never to "nobody is hurt".
    """
    from ..data import nflverse

    try:
        frame = nflverse.load("injuries", [season])
    except Exception:
        return pd.DataFrame(columns=["name_key", "report_status"])

    if "game_type" in frame.columns:
        frame = frame[frame["game_type"] == "REG"]
    frame = frame[frame["week"] == week]
    if frame.empty or "report_status" not in frame.columns:
        return pd.DataFrame(columns=["name_key", "report_status"])

    frame = frame.dropna(subset=["report_status"]).copy()
    frame["name_key"] = frame["full_name"].map(normalize_name)
    return worst_status_per_player(frame)


# Severity order, worst first. A player listed twice must resolve to the worse
# designation -- taking whichever row happened to come first could turn an "Out"
# into a "Questionable" and put a guaranteed zero back in the lineup.
_SEVERITY = {"Out": 0, "Doubtful": 1, "Questionable": 2}


def worst_status_per_player(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per player, keeping the most severe designation."""
    if frame.empty:
        return pd.DataFrame(columns=["name_key", "report_status"])
    ranked = frame.assign(
        _severity=frame["report_status"].map(_SEVERITY).fillna(3)
    ).sort_values("_severity")
    return (ranked.drop_duplicates("name_key")[["name_key", "report_status"]]
                  .reset_index(drop=True))


def availability(
    samples,
    *,
    week: int,
    board: Optional[pd.DataFrame] = None,
    report: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-player availability for one week, aligned to ``samples`` rows.

    Columns: ``multiplier`` in [0, 1] and ``reason`` (empty when fine).
    A multiplier of 0 means do not start this player under any circumstances.
    """
    names = samples.players["player_name"].astype(str)
    keys = names.map(normalize_name)

    multiplier = pd.Series(1.0, index=range(len(names)), dtype=float)
    reason = pd.Series("", index=range(len(names)), dtype=object)

    # Bye first: it is certain, so it outranks any probabilistic discount.
    if board is not None and "bye" in board.columns:
        byes = (board.assign(_k=board["player_name"].astype(str).map(normalize_name))
                     .dropna(subset=["bye"])
                     .drop_duplicates("_k")
                     .set_index("_k")["bye"])
        on_bye = keys.map(byes).astype("float") == float(week)
        multiplier[on_bye.to_numpy()] = BYE_MULTIPLIER
        reason[on_bye.to_numpy()] = "BYE"

    if report is not None and len(report):
        # Deduped here as well as in `injury_report`, because a caller passing a
        # raw frame must not be able to smuggle a softer designation through.
        deduped = worst_status_per_player(report)
        status = keys.map(deduped.set_index("name_key")["report_status"])
        for label, factor in STATUS_MULTIPLIER.items():
            hit = (status == label).to_numpy()
            # Do not let an injury designation soften a bye -- a bye is certain.
            hit &= (reason != "BYE").to_numpy()
            multiplier[hit] = np.minimum(multiplier[hit], factor)
            reason[hit] = label.upper()

    return pd.DataFrame({"multiplier": multiplier.to_numpy(),
                         "reason": reason.to_numpy()})
