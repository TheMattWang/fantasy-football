# LOOP.md — autonomous improvement loop state

Durable state for the self-driving loop. **Read this first after any compaction.**
Plan: `~/.claude/plans/okay-this-was-originally-streamed-ripple.md`

Phase A = now -> draft day (readiness). Phase B = draft day -> end of season (weekly beat).

---

## Current

- **Phase:** A
- **Round:** R1 DONE -> next R0
- **Started:** 2026-08-25
- **Sealed touches spent:** 0 of 3. **2021 stays clean.**

## Termination checklist (Phase A)

| id | check | status |
|----|-------|--------|
| T1 | board fresh; `draft_day.py` starts without `--allow-stale`; 15-round offline dry run | **PASS** (dry run pending) |
| T2 | `week.py` runs weeks 1-14, no traceback, no silently-wrong output | OPEN |
| T3 | suite green; every defect fixed this run has a regression test | **PASS** (266/266) |
| T4 | defect ledger has zero open entries | OPEN |
| T5 | two consecutive rounds add nothing new | 0 of 2 |

## Round queue

- [ ] R0  ETag-conditional refresh for nflverse  (headline bug)
- [x] R1  Refresh board; suite green            (deadline-bound)
- [ ] R2  Byes from schedule + team normalizer
- [ ] R3  Sweep week.py across weeks 1-14
- [ ] R4  Install loop machinery (hooks, autocompact, weekly LaunchAgent)
- [ ] R5  Refine availability table (report_status x practice_status)
- [ ] R6  Price P(INA | Questionable) from nflverse; then judge Sleeper
- [ ] R7  Waivers reachable from a CLI
- [ ] R8  Close the protocol hole
- [ ] R9  Expand training to 2019-2020  (GATED — last; irreversible)

## Defect ledger

| id | defect | severity | status |
|----|--------|----------|--------|
| D1 | `nflverse.load` never refreshes; nothing passes `refresh=`. In-season data freezes. | CRITICAL | OPEN (R0) |
| D2 | Board 11 days stale; `draft_day.py` refuses to start | CRITICAL | **FIXED** (R1) |
| D10 | Draft gate demanded ECR <=3d from a feed that publishes every 7d -- unsatisfiable 4 days in 7 | CRITICAL | **FIXED** (R1) |
| D3 | `bye` covers 188/505 board rows; roster player outside FFC top-190 gets no bye check | HIGH | OPEN (R2) |
| D4 | No team-abbreviation normalizer: `JAX`/`JAC`, `LA`/`LAR` silently drop 2 teams | HIGH | OPEN (R2) |
| D5 | `STATUS_MULTIPLIER` averages over 3 distinct practice states; 289 no-designation DNP weeks scored healthy | MEDIUM | OPEN (R5) |
| D6 | `rank_waiver_adds` unreachable from any CLI | MEDIUM | OPEN (R7) |
| D7 | Protocol enforced only on `replay_season`; worker scripts bypass `check()` | MEDIUM | OPEN (R8) |
| D8 | `docs/findings/` has 4 files for 12 experiments; §13 claims one each | LOW | OPEN |
| D9 | Untracked cruft not gitignored; `deployment/` is 8 empty dirs | LOW | OPEN |

## Decisions recorded (so they are not silently revisited)

- **Do NOT adopt `nflreadpy` this season.** It is 0.1.5, README says "written by Claude
  based on nflreadr, use at your own risk", no functional commit since 2025-11-23, and it
  costs +50 MB of polars plus a second dataframe library. Its cache default is `memory`,
  which is *worse* for Colab. Our `_asset_url` is more current than nfl_data_py's. What is
  broken is the TTL, not the URL layer. Revisit in the offseason.
- **`nfl_data_py` is a dead end.** GitHub repo `archived: true`; pins `pandas<2.0` (we need
  >=2.0); reads the retired `player_stats_{y}.parquet` (2025 -> 404).
- **2021 stays clean.** Not read, not trained on, not evaluated against.
- **Yahoo API is effectively closed** — new apps cannot request the Fantasy scope and yfpy
  sees blanket 403s. But public-league *HTML* works unauthenticated including `/settings`
  with full scoring config. That is a possible route out of `PROVISIONAL` — not scheduled,
  recorded so it is not lost.

## Ledger of completed rounds

(append one entry per round: what changed, the command that proved it, verdict)

### R1 — board refresh, and an unsatisfiable gate (2026-08-25)

Refreshed the board: ECR snapshot **2026-08-14 -> 2026-08-21**, market coverage
**89% -> 92%** (165/180 of the drafted range). That alone did NOT turn the suite green.

**D10, found by the refresh failing to fix D2.** The board was as fresh as it could
possibly be and the draft gate still called it fatal: `DRAFT_MAX_ECR_AGE_DAYS = 3.0`
against a feed that publishes **weekly**. Measured over 361 snapshots: median gap **7
days**, p90 **7 days**; in August, 19 of 25 in-season gaps are exactly 7. So the bound was
unsatisfiable four days out of every seven no matter how promptly anyone rebuilt.

That is worse than a false alarm. A check that cannot be satisfied teaches you to pass
`--allow-stale`, which disables the check entirely -- so the gate meant to protect draft
day would have been routinely bypassed *on draft day*.

Fix keeps `draft_day.py` offline by moving the comparison to build time: provenance now
records `cadence_days` (measured, not chosen), `latest_available`, and
`is_latest_available`. The gate then separates two failures that were conflated:
  * built from the newest snapshot that exists -> only a dead feed is fatal, bounded at
    `MAX_MISSED_PUBLICATIONS * cadence`;
  * an older snapshot shipped while a newer one existed -> still fatal, and it now names
    the newer date so the fix is obvious.

Verified: `pytest tests/ -q` **266 passed** (was 259/260); `draft_day.py --slot 6` starts
with **no** `--allow-stale`. Six new tests pin the distinction, including that a board
with no cadence recorded keeps the strict bound (absent evidence is not evidence of
freshness) and that a dead-but-latest feed is still fatal.

**Verdict: KEPT.**
