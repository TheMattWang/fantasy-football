# Runbook — 2026 draft

Everything below runs from the repo root with `./.venv/bin/python`.
One-time setup: `python -m venv .venv && ./.venv/bin/pip install -r requirements.txt`

---

## 0. Before draft week: Yahoo credentials

The only step that needs you. Everything else is automated.

```bash
./.venv/bin/python -m src.data.yahoo_league --doctor
```

It prints the exact next action. The short version:

1. Create an app at <https://developer.yahoo.com/apps/create/>
   - Application Type: **Installed Application**
   - Redirect URI: **`oob`** (must be exactly this)
   - API Permissions: **Fantasy Sports → Read**
2. `... --set-credentials <CLIENT_ID> <CLIENT_SECRET>` (writes `data/cache/.env`, chmod 600)
3. `... --auth` — opens a browser once, caches the token

If it ever fails with `invalid_grant`, the refresh token was revoked (this happens
when you change the app's permissions). A stale refresh token cannot be repaired,
only reissued: `--reset-token` then `--auth`.

Then pull the league:

```bash
./.venv/bin/python -m src.data.yahoo_league --discover --season 2025
./.venv/bin/python -m src.data.yahoo_league --season 2025 --league-id <id> --rosters
./.venv/bin/python -m src.data.league_config          # confirm what was pulled
```

**Why this matters more than it looks.** Replacement level depends on roster
slots, and every VORP on the board depends on replacement level. The repo
currently has three hardcoded league configs and two of them disagree
(`clean.py` says half-PPR, `src/core/scoring.py` says full PPR). Until the pull
lands, everything runs on `PROVISIONAL` settings and says so loudly.

---

## 1. Build the board

```bash
./.venv/bin/python -m src.projections.board --season 2026 \
    --out data/processed/board_2026.csv
```

Anchored on FantasyPros consensus: the curve maps *where the market ranks a
player* to *what players at that rank have actually scored*, fit over prior
seasons only. Regression to the mean, aging and injury risk come along for free,
because the consensus already priced them.

Check it held up:

```bash
./.venv/bin/python -m src.projections.validate --holdout 2022 2023 2024 2025
```

---

## 2. Sanity-check the distributions

```bash
./.venv/bin/python -m src.simulation.distributions --check       # interval coverage
./.venv/bin/python -m src.simulation.distributions --season 2026 # build + inspect
```

`--check` is the one that matters: 80% predictive intervals should contain ~80%
of held-out outcomes. Materially below means the model is overconfident and the
search will take bad risks.

---

## 3. Run the gate

Tuning runs go against 2022/2023 and are unlimited:

```bash
./.venv/bin/python -m src.evaluation.replay --season 2023 --replicates 200 \
    --policies adp need_adp vorp_greedy season_sim
```

2024/2025 are the held-out gate, with a budget of **3 touches for the whole
project**. The code enforces it: a gate season needs `--protocol gate` and a
`--register` name that already exists in `gate_registry.json`, otherwise the
run refuses to start. Write the hypothesis into that file and commit it
*before* running — that commit is what makes the pre-registration real.

```bash
./.venv/bin/python -m src.evaluation.replay --season 2025 --replicates 200 \
    --protocol gate --register shrinkage_v1 \
    --policies adp need_adp vorp_greedy season_sim
```

Drafts a past season and scores it with that season's **actual** weekly results.
The bar is `need_adp` — consensus order subject to roster limits, i.e. a
competent human. Plain `adp` is only a floor: with no positional awareness it
drafts eight quarterbacks, so beating it proves nothing.

**Decision rule, committed to before looking:** if `season_sim` does not beat
`need_adp` on mean final rank, draft off `need_adp` on the day and treat the
agent as an informational display.

---

## 4. Draft day

Build the board first, then go offline:

```bash
./.venv/bin/python draft_day.py --board data/processed/board_2026.csv --slot <1-12>
```

Nothing in this script fetches anything. A network stall against a 90-second
pick clock is the failure you cannot recover from.

| column | meaning |
|---|---|
| `U` | `P(playoffs) + 2·P(title)` from the season simulation |
| `dU` | how much worse this option is than the best one |
| `P(next)` | chance he is still there at your next turn |

`P(next)` is the number that should decide whether to reach. If the top two are
within `dU < 0.005`, they are effectively tied — take whichever is less likely
to last.

Commands: `<name>` records a pick, `me <name>` records yours, plus `undo`,
`board`, `roster`, `?`, `quit`. Names are fuzzy-matched. Add `--fast` if picks
are taking too long.

---

## 5. In-season (September onward)

Same objective as the draft, applied to the weeks still to be played:

```python
from src.inseason.waivers import rank_waiver_adds, start_sit

start_sit(my_roster, samples, config)                      # who to start
rank_waiver_adds(my_roster, free_agents, samples, config,
                 opponent_rosters, from_week=6)            # who to add/drop
```

Waiver adds are priced in **ΔP(playoffs)**, which gets the hard call right
without a stash coefficient: a third quarterback scores ~+0.01 because he never
starts, while a startable RB scores ~+0.06. This is where most of the remaining
season-long edge lives, and unlike the draft it gives you feedback every week.

---

## Colab

Everything is importable from `src/`; notebooks should be thin drivers. Nothing
here needs a GPU — the season simulation is a 54 MB numpy array evaluated in
~0.3 ms, and Colab's 2 vCPU is *slower* than this laptop's 8 cores. Use Colab so
long jobs do not tie up the machine, not for speed.

```python
!git clone https://github.com/TheMattWang/fantasy-football && cd fantasy-football
!pip install -q -r requirements.txt
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ['FF_CACHE_DIR'] = '/content/drive/MyDrive/fantasy/cache'
```

Set `FF_CACHE_DIR` to Drive or the cache dies with the runtime. Do the Yahoo
OAuth **locally** and copy `data/cache/.env` into that Drive folder — `yfpy`
wants a localhost redirect, which is not a fight worth having in a notebook.
**Do not run draft day on Colab.**

---

## What is deliberately not here

- **No torch.** Nothing needs it, and the old "GPU MCTS trainer" never trained
  anything — its policy loss was literally `torch.tensor(0.0)`.
- **No hyperparameter search.** It tuned coefficients against `0.35 × total_vorp`
  — the same quantity the agent maximized — and never completed a trial.
- **No injury subsystem.** Every value in it came from
  `np.random.seed(hash(player_name))`. Availability now comes from real weekly
  roster status.
- **No bye-week penalty.** A bye is availability 0 for one week and the waiver
  pool covers it; the season simulation handles it without a coefficient.
- **No MCTS.** Search was never the bottleneck. Build the tree only if the
  current policy first beats the bar.
