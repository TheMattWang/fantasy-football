"""Can an ADP-anchored board replace the ECR-anchored one, and buy back seasons?

ECR history starts 2021, so the replayable universe is 2022-2025 -- four seasons,
all already spent as a holdout. FFC half-PPR ADP runs from 2018 (verified against
the API: half-ppr 2018-2026, standard and ppr 2014-2026). An ADP-anchored board
needs no ECR, so it can reach back further.

`as_ecr_history` reshapes FFC into the panel schema, which means
`preseason_snapshot`, `join_actuals` and `fit_baseline` all work unchanged --
they already accept an injected history frame. One construction, two anchors,
rather than two pipelines that could drift apart.

THREE QUESTIONS, and the third is deliberately not asked of the new seasons.

  A  On the SPENT seasons (2022-2025), how far do the two anchors disagree?
     This is a finding either way. Close agreement means ADP is a safe
     substitute; wide disagreement means the eras are not comparable AND that
     there is market-vs-expert signal, which the project has never used.

  B  On the SPENT seasons, which anchor PREDICTS BETTER? Directly testable, and
     it decides which one the project should have been built on.

  C  On 2018-2021, do boards build at all, with adequate coverage? Structure
     only. Their OUTCOMES are not touched -- that is the whole point of keeping
     them, and measuring board accuracy there would spend them.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mattwang/Documents/fantasy/fantasy-football")

import numpy as np
import pandas as pd

from src.data.ffc_adp import as_ecr_history
from src.data.league_config import provisional_config
from src.projections.board import build_board
from src.projections.ecr import fit_baseline, join_actuals, load_ecr_history

CFG = provisional_config()
SPENT = (2022, 2023, 2024, 2025)          # development only, already burned
CANDIDATE = (2019, 2020, 2021)            # the seasons we are trying to buy back
POSITIONS = ("QB", "RB", "WR", "TE")
ADP_FIRST = 2018                          # first half-PPR season FFC publishes


def adp_panel(seasons):
    return as_ecr_history(seasons, scoring="half-ppr", teams=CFG.num_teams)


def spearman(a, b):
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else np.nan


# --- A: how far apart are the two anchors? --------------------------------

print("A. ECR vs ADP AS THE ANCHOR, on the spent seasons\n")
ecr_hist = load_ecr_history()
adp_hist = adp_panel(range(ADP_FIRST, 2027))

print(f"  {'season':<8}{'n both':>8}{'overall rho':>13}"
      + "".join(f"{p:>7}" for p in POSITIONS))
for season in SPENT:
    from src.projections.ecr import preseason_snapshot
    e = preseason_snapshot(season, history=ecr_hist)[["name_key", "pos", "ecr_rank", "pos_rank"]]
    a = preseason_snapshot(season, history=adp_hist)[["name_key", "pos", "ecr_rank", "pos_rank"]]
    m = e.merge(a, on=["name_key", "pos"], suffixes=("_ecr", "_adp"))
    cells = []
    for p in POSITIONS:
        g = m[m["pos"] == p]
        cells.append(f"{spearman(g['pos_rank_ecr'], g['pos_rank_adp']):>7.3f}"
                     if len(g) > 5 else f"{'--':>7}")
    print(f"  {season:<8}{len(m):>8}"
          f"{spearman(m['ecr_rank_ecr'], m['ecr_rank_adp']):>13.3f}" + "".join(cells))

# --- B: which anchor predicts better? -------------------------------------

print("\nB. WHICH ANCHOR PREDICTS REALIZED PPG BETTER? (spent seasons only)")
print("   Rank-vs-outcome correlation within position; more negative is better,")
print("   since rank 1 should score most.")
print()
print("   RESTRICTED TO THE PLAYERS BOTH ANCHORS RANK. ECR lists ~500 players and")
print("   FFC ~150, and a rank-outcome correlation computed over a wider range is")
print("   mechanically stronger because it includes replacement-level players who")
print("   score near zero. Comparing raw would measure coverage, not accuracy --")
print("   the same range-restriction artifact B1 turned up.\n")
print(f"  {'season':<8}{'anchor':<8}{'n':>6}" + "".join(f"{p:>8}" for p in POSITIONS))
for season in SPENT:
    try:
        joined = {label: join_actuals(season, history=h)
                  for label, h in (("ecr", ecr_hist), ("adp", adp_hist))}
    except Exception as exc:
        print(f"  {season:<8}  FAILED {str(exc)[:60]}")
        continue

    shared = set.intersection(*[
        set(zip(f["name_key"], f["pos"]))
        for f in joined.values()
    ])
    for label, merged in joined.items():
        key = list(zip(merged["name_key"], merged["pos"]))
        merged = merged[[k in shared for k in key]]
        merged = merged[merged["ppg_available"].notna()]
        cells = []
        for p in POSITIONS:
            g = merged[merged["pos"] == p]
            cells.append(f"{spearman(g['pos_rank'], g['ppg_available']):>8.3f}"
                         if len(g) > 10 else f"{'--':>8}")
        print(f"  {season:<8}{label:<8}{len(merged):>6}" + "".join(cells))

# --- C: do the new seasons build? -----------------------------------------

print("\nC. CAN WE BUILD THE SEASONS WE DO NOT HAVE?  (structure only --")
print("   their outcomes are NOT read, which is what keeps them usable)\n")
print(f"  {'season':<8}{'train seasons':<20}{'rows':>6}{'skill':>7}"
      f"{'top180 cov':>12}  status")
for season in CANDIDATE:
    train = [s for s in range(ADP_FIRST, season)]
    if not train:
        print(f"  {season:<8}{'(none)':<20}{'':>6}{'':>7}{'':>12}  SKIP: no prior ADP season")
        continue
    try:
        curve = fit_baseline(train, history=adp_hist, enforce_match_gate=False)
        # history= is load-bearing. Without it build_board falls back to ECR and
        # silently produces an ECR-anchored board for any season ECR covers,
        # which is exactly the bug this run first reported as a success.
        board = build_board(season, config=CFG, curve=curve,
                            train_seasons=train, validate=False,
                            market_dispersion=False, history=adp_hist)
        # Coverage of the range that actually gets drafted.
        drafted = CFG.num_teams * (CFG.total_rounds or 15)
        top = board.nsmallest(drafted, "adp_rank")
        cov = float(top["proj_points"].notna().mean())
        skill = int((~board["is_streamed"]).sum())
        status = "OK" if cov >= 0.95 and skill > 150 else "THIN"
        print(f"  {season:<8}{str(train):<20}{len(board):>6}{skill:>7}"
              f"{cov:>12.1%}  {status}")
    except Exception as exc:
        print(f"  {season:<8}{str(train):<20}  FAILED: {str(exc)[:60]}")
