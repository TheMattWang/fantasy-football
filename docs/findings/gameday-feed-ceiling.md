# R6b — what a live game-day feed is worth, bounded before building one

**Verdict: CEILING MEASURED at +0.515 points per roster-week (+7.2 a season), positive
in all four seasons. Real and worth building; roughly 7x the multiplier fix shipped the
same day, and about a third of one week's worst starter. Not the +88 a naive oracle
first reported.**

Run 2026-08-25. Script: `experiments/worker_gameday_oracle.py`.

## Why a ceiling first

V9 killed the attempt to re-weight Friday's information. This is a different question:
not "can we read the Friday report better" but "what if we simply knew on Sunday
morning who is playing".

`Questionable -> 0.542` exists **because Friday is uncertain**. A real inactive list
does not sharpen that estimate, it **collapses** it — to 0 or to 1. So before writing a
line of polling code, bound the prize. This is the discipline that stopped a whole phase
of draft work here once already.

## The population

Joining Friday's designation to realised `weekly_rosters.status` — 6,783
designation-weeks, all already on disk:

| Friday says | n | P(inactive Sunday) |
|---|---|---|
| (none) | 3525 | 0.031 |
| Questionable | 1717 | **0.298** |
| Doubtful | 197 | 0.893 |
| Out | 1342 | 0.797 |

Nearly **30% of Questionable players are inactive**, and today all 1,717 of them are
priced at a single 0.542.

## The number, and the one I nearly reported

The first oracle zeroed **anyone** who did not record a stat line, and returned
**+6.311 points per roster-week — +88 a season**. That is wrong by an order of
magnitude, and the reason matters: it also resolves players on IR, healthy scratches,
and anyone who never appeared on an injury report at all. **No feed gives you that.**
Much of it is not even a lineup question — it is "do not roster a player who is on IR",
which these synthetic rosters never do because they never touch waivers.

Restricted to the population an inactive list actually covers — players carrying a
Friday designation:

```
  season   roster-weeks     wide     NARROW   se
  2022             720   +7.575     +0.815   0.183
  2023             720   +5.151     +0.391   0.147
  2024             720   +6.454     +0.587   0.126
  2025             720   +6.064     +0.268   0.136

  NARROW ceiling: +0.515 pts/roster-week (se 0.119, 4/4 seasons positive)
                  +7.21 points over a 14-week season
```

**+0.515 is the number to judge.** A real feed is strictly worse: Sleeper is not
perfect, and not every inactive is known before lineups lock.

## Is it worth building

Yes, with the size stated honestly. For scale, all per roster-week:

| lever | value |
|---|---|
| game-day feed, **ceiling** | +0.515 pts |
| Questionable multiplier fix (R5, shipped) | +0.076 pts |
| practice-participation split (V9, killed) | −0.006 pts |

About **7x** the largest thing shipped this session, positive every season, and unlike
V8 or V9 it does not depend on a claim about football — it replaces an estimate with an
observation. That is the same category as the R5 fix, which is the category that has
actually worked here.

It is still only ~7 points a season at the ceiling, against a weekly team total near
107. Worth a Sunday-morning poll of a free, auth-free endpoint; **not** worth a
complicated subsystem.

## Next, in order

1. Verify Sleeper's public API actually carries `injury_status` fresh on Sunday
   mornings, and how far ahead of kickoff it updates. No auth, ~1 KB per player.
2. Join by ID, not name — `weekly_rosters` carries `sleeper_id` (97.2%) and `espn_id`
   (99.3%), so no fuzzy matching is needed.
3. Sunday-only path in `week.py`; the Friday numbers stay for every other day.
4. Decision-test it the same way before shipping. The ceiling is not the result.
