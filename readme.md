# Fantasy football draft and in-season tooling

Make me not bad at fantasy.

The short version of what happened: the 2025 team finished 11th of 12, the
rebuild set out to draft better than consensus, and **it cannot** — eight
independent searches for a valuation edge came back null or negative. What the
project produced instead is a draft-day tool that follows consensus and knows
why, an in-season tool for the decision that turned out to be larger, and a
replicated demonstration of the optimizer's curse.

`docs/WHERE_WE_ARE.md` is the full account. Start there if you have lost the
thread; it assumes no memory and defines its terms.

## Using it

Before the draft — the refresh is not optional, because the board is rebuilt
from local caches unless you ask for new data:

```bash
python -m src.projections.board --season 2026 --refresh --require-market \
    --out data/processed/board_2026.csv
```

On the clock. Runs entirely offline, and refuses to start on a stale board:

```bash
python draft_day.py --slot <your pick>
```

Each week of the season:

```bash
python week.py --week <N>
```

Install `scripts/com.claude.fantasy-board-refresh.plist` to keep the board
current daily. See `RUNBOOK.md` for the operating rules.

## Layout

| path | what |
|---|---|
| `src/data` | ingestion: nflverse, FantasyPros ECR, FFC market ADP, league config, validators |
| `src/projections` | consensus rank → points curve, the board |
| `src/simulation` | player uncertainty and the head-to-head season |
| `src/draft` | draft engine, opponent models, search policies |
| `src/inseason` | start/sit and waiver valuation |
| `src/evaluation` | replay against real seasons, and the holdout protocol |
| `experiments/` | every experiment script, so the numbers are rerunnable |
| `docs/findings/` | one file per experiment, each with a KEPT/KILLED verdict |

```bash
./.venv/bin/python -m pytest tests/ -q     # 245 tests, ~9s, run from the repo root
```

## Two things to know before trusting a number

**The league settings are guesses.** No Yahoo credentials, so everything runs on
a placeholder 12-team half-PPR config. Roster shape *is* positional scarcity, so
every VORP rests on that assumption.

**The holdout is budgeted.** 2024/2025 are spent. 2019–2021 were restored via
ADP-anchored boards and are sealed at three touches, enforced in
`src/evaluation/protocol.py`. Pre-register in `gate_registry.json` before
spending one.
