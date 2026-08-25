"""Measure P(played) by injury designation -- the source of the multipliers.

Every hand-written constant in this project has eventually turned out to be
wrong: the waiver floors were 30-45% low, `ecr_sd` ran 2x the real draft
dispersion. So the availability multipliers in `src/inseason/availability.py`
are measured rather than chosen, and this reproduces them.

A designation matters twice over, and only measuring catches the second half:
how often the player suits up at all, AND how well he does when he does. A
Questionable player who plays still underperforms, so the expected value is the
product of the two, not just the play rate.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mattwang/Documents/fantasy/fantasy-football")

import pandas as pd

from src.data import nflverse
from src.projections.ecr import normalize_name

SEASONS = (2022, 2023, 2024, 2025)
POSITIONS = ("QB", "RB", "WR", "TE")

rows = []
for season in SEASONS:
    inj = nflverse.load("injuries", [season])
    if "game_type" in inj.columns:
        inj = inj[inj["game_type"] == "REG"]
    inj = inj[inj["position"].isin(POSITIONS)].copy()
    inj["name_key"] = inj["full_name"].map(normalize_name)

    wk = nflverse.load("weekly", [season])
    if "season_type" in wk.columns:
        wk = wk[wk["season_type"] == "REG"]
    wk = wk.copy()
    name_col = ("player_display_name" if "player_display_name" in wk.columns
                else "player_name")
    wk["name_key"] = wk[name_col].map(normalize_name)
    wk["fp"] = nflverse.fantasy_points(wk)
    # A stat line is the evidence that he played. No row means he did not.
    played = wk.groupby(["name_key", "week"])["fp"].max().rename("fp").reset_index()

    merged = inj.merge(played, on=["name_key", "week"], how="left")
    merged["played"] = merged["fp"].notna()
    rows.append(merged[["week", "name_key", "report_status", "practice_status",
                        "played", "fp"]])

d = pd.concat(rows, ignore_index=True)
healthy = d[d["report_status"].isna()]
base = healthy.loc[healthy["played"], "fp"].mean()

print(f"{len(d):,} player-week injury reports, {SEASONS[0]}-{SEASONS[-1]}\n")
print(f"  {'status':<14}{'n':>7}{'P(played)':>11}{'pts|played':>12}{'multiplier':>12}")
for status, group in d.groupby("report_status"):
    rate = group["played"].mean()
    points = group.loc[group["played"], "fp"].mean()
    print(f"  {str(status):<14}{len(group):>7}{rate:>11.3f}{points:>12.2f}"
          f"{rate * points / base:>12.3f}")
print(f"  {'(no report)':<14}{len(healthy):>7}{healthy['played'].mean():>11.3f}"
      f"{base:>12.2f}{1.0:>12.3f}")
print("\n  multiplier = P(played) x mean points when played, relative to a")
print("  player carrying no designation. It is what the lineup is chosen on.")


# --------------------------------------------------------------------------
# Does Friday's practice participation add anything to the designation?
# --------------------------------------------------------------------------
#
# `Questionable` is the designation that actually decides lineups, and it is
# almost a coin flip. But it is not one bucket: over 2022-2025 it splits three
# ways by Friday practice -- Full, Limited, and Did Not Participate -- and the
# shipped multiplier averages over all three. If a Questionable-and-DNP player
# plays far less often than a Questionable-and-Full one, a single number is
# leaving a real distinction on the floor.
#
# Separately: players carrying NO game-status designation who nonetheless did
# not practice are currently scored as fully healthy, at multiplier 1.0.

PRACTICE_SHORT = {
    "Full Participation in Practice": "Full",
    "Limited Participation in Practice": "Limited",
    "Did Not Participate In Practice": "DNP",
}

d["practice"] = d["practice_status"].map(PRACTICE_SHORT)

print("\n\n" + "=" * 68)
print("Does Friday practice participation add to the designation?")
print("=" * 68)
print(f"\n  {'status':<14}{'practice':<10}{'n':>7}{'P(played)':>11}"
      f"{'pts|played':>12}{'multiplier':>12}")

report = d["report_status"].fillna("(none)")
for status in ("(none)", "Questionable", "Doubtful", "Out"):
    block = d[report == status]
    if block.empty:
        continue
    for practice in ("Full", "Limited", "DNP"):
        cell = block[block["practice"] == practice]
        if len(cell) < 30:          # too few to estimate anything from
            continue
        rate = cell["played"].mean()
        points = cell.loc[cell["played"], "fp"].mean()
        mult = rate * points / base if points == points else 0.0
        print(f"  {status:<14}{practice:<10}{len(cell):>7}{rate:>11.3f}"
              f"{points:>12.2f}{mult:>12.3f}")
    print()

print("  Read the Questionable rows against the pooled 0.456 the code ships.")
print("  A spread wide enough to change a lineup is worth splitting; one that")
print("  is not, is not -- V8 is the cautionary tale, where a t=+6.10 predictor")
print("  moved the actual decision by +0.072 points a week.")
