# LOOP.md — autonomous improvement loop state

Durable state for the self-driving loop. **Read this first after any compaction.**
Plan: `~/.claude/plans/okay-this-was-originally-streamed-ripple.md`

Phase A = now -> draft day (readiness). Phase B = draft day -> end of season (weekly beat).

---

## Current

- **Phase:** A
- **Round:** R2 DONE -> next R3
- **Started:** 2026-08-25
- **Sealed touches spent:** 0 of 3. **2021 stays clean.**

## Termination checklist (Phase A)

| id | check | status |
|----|-------|--------|
| T1 | board fresh; `draft_day.py` starts without `--allow-stale`; 15-round offline dry run | **PASS** (dry run pending) |
| T2 | `week.py` runs weeks 1-14, no traceback, no silently-wrong output | OPEN |
| T3 | suite green; every defect fixed this run has a regression test | **PASS** (277/277) |
| T4 | defect ledger has zero open entries | OPEN |
| T5 | two consecutive rounds add nothing new | 0 of 2 |

## Round queue

- [x] R0  ETag-conditional refresh for nflverse  (headline bug)
- [x] R1  Refresh board; suite green            (deadline-bound)
- [x] R2  Byes from schedule + team normalizer
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
| D1 | `nflverse.load` never refreshes; nothing passes `refresh=`. In-season data freezes. | CRITICAL | **FIXED** (R0) |
| D11 | `network` pytest marker unregistered; suite reached the live feed despite docstring claiming otherwise | LOW | **FIXED** (R0) |
| D2 | Board 11 days stale; `draft_day.py` refuses to start | CRITICAL | **FIXED** (R1) |
| D10 | Draft gate demanded ECR <=3d from a feed that publishes every 7d -- unsatisfiable 4 days in 7 | CRITICAL | **FIXED** (R1) |
| D3 | `bye` covers 188/505 board rows; roster player outside FFC top-190 gets no bye check | HIGH | **FIXED** (R2) |
| D4 | No team-abbreviation normalizer: `JAX`/`JAC`, `LA`/`LAR` silently drop 2 teams | HIGH | **FIXED** (R2) |
| D12 | FFC carries a stale team for Kayshon Boutte (HOU; nflverse+board say NE) | LOW | **WONTFIX** -- upstream feed error, now detected and warned; schedule value wins |
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

### R0 — the in-season cache was write-once (2026-08-25)

`nflverse.load` re-downloaded only when `refresh=True`, and **nothing anywhere passed
it** -- `weekly_fantasy` did not even expose the parameter. The first in-season load
pinned that season's data for the rest of the year.

The failure mode is worse than "data goes stale", and the research agent found the part I
had wrong: both callers filter the cached frame **by week**. A file cached in week 1 does
not merely age by week 5, it filters to **empty** -- which `injury_report`'s
graceful-degradation path reads as "no information", scoring every ruled-out player as
fully healthy for the rest of the season. A stale-but-parseable file is indistinguishable
from a genuinely thin one, so the availability work would have been defeated from
underneath with no error anywhere.

Fixed with a conditional GET rather than a TTL: nflverse serves `ETag`, so `If-None-Match`
returns **304 with no body** when nothing changed. No wasted downloads, and -- unlike a
TTL -- no stale window in which we serve last week's data because an interval has not
elapsed. Finished seasons are never rechecked (their files cannot change); only the live
season and season-independent assets are.

Measured against the live feed: cold 0.95s, warm **0.23s** (304), forced re-fetch 0.87s.
While testing, `last_modified` moved from 13:07 to 17:47 the same day -- the feed really
does republish intraday, so a TTL of any length would have been wrong some of the time.

Also registered the `network` pytest marker. `test_nflverse.py` has claimed since it was
written that network tests "are skipped by default"; with no pytest config they were not.
That matters now the loop runs unattended -- a suite that reaches the network turns
somebody else's outage into a red build here.

Verified: **273 passed, 1 deselected**; `-m network` runs the live check separately;
`week.py --week 5 --season 2025` reads 1498 stat lines and 137 designations and benches
Lamar Jackson (22.6 ppg, the roster's best rate) as OUT. `draft_day.py` proven to run with
`socket.connect` disabled -- now a permanent test, because the new remote check is right
for the weekly feed and would be fatal on a 90-second pick clock.

**Verdict: KEPT.**

### R2 — bye weeks from the schedule, and the join that was silently dropping two clubs (2026-08-25)

`bye` arrived only with the FFC market attachment, which lists ~190 players, so it covered
**188 of 505** board rows. A rostered player outside that range got **no bye check at
all** -- and a bye is a guaranteed zero, the cheapest mistake in fantasy to avoid.

nflverse publishes a `schedules` release (`games.parquet`, one all-seasons file,
republished daily) that already carried all 272 of the 2026 regular-season games in
August. A team's bye is derivable from it exactly: the regular-season week it plays no
game. No feed publishes "bye week" as a field, but this is a fact about the schedule
rather than an opinion of the market.

**The trap.** nflverse spells two clubs differently from the fantasy sites -- `JAX` vs
`JAC`, `LA` vs `LAR` -- and there was **no team normalizer anywhere in `src/`**. A naive
join drops Jacksonville and the Rams entirely: every player on them, with no error. Added
`normalize_team` mirroring `normalize_name`, which exists for the same reason and makes
the same argument.

**The cross-check earned its keep on the first run**, and not the way I expected: 1
disagreement in 194 rows, and it was a *team* conflict rather than a bye one. Both feeds
agree NE's bye is 11 and HOU's is 8 -- they disagree about which club Kayshon Boutte plays
for. Adjudicated against nflverse weekly rosters: **NE, week 1, ACT**, so the board is
right and FFC is stale. That is now the recorded reason the schedule value wins over the
market one, rather than an arbitrary preference.

Coverage **188/505 -> 498/518 (96%)**, and the only 20 rows without a bye are free agents,
who have no team and therefore cannot have one. A test asserts precisely that: no player
on a real team may lack a bye.

Verified: `pytest tests/ -q` **277 passed, 1 deselected**.

**Verdict: KEPT.**
