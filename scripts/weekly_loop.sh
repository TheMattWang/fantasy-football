#!/usr/bin/env bash
# The in-season beat. Two firings a week, tied to when real data lands rather
# than to arbitrary times:
#
#   Tue  nflverse has published week N-1 results and waivers have run.
#        Grade last week's recommendation against what actually happened,
#        append to the season ledger, and run one improvement round.
#   Sun  final injury designations are out. Set the lineup you actually play.
#
# The Tuesday beat is the valuable one. Recording a lineup BEFORE kickoff and
# scoring it after gives this project a live, uncontaminated test set -- the one
# clean measurement still available now that the holdout is spent.
set -u
REPO="$HOME/Documents/fantasy/fantasy-football"
PY="$REPO/.venv/bin/python"
LOG="$HOME/.claude/daily-log/fantasy-weekly.log"
mkdir -p "$(dirname "$LOG")"
cd "$REPO" || exit 1

BEAT="${1:-tuesday}"
SEASON="${FANTASY_SEASON:-2026}"

# Week 1 of the NFL season is the Thursday after Labor Day; deriving the current
# week from the schedule beats hardcoding a start date that silently rots.
WEEK=$("$PY" - "$SEASON" <<'PYEOF'
import datetime as dt, sys
try:
    from src.data.nflverse import load
    g = load("schedules")
    g = g[(g["season"] == int(sys.argv[1])) & (g["game_type"] == "REG")]
    d = __import__("pandas").to_datetime(g["gameday"], errors="coerce")
    today = dt.date.today()
    played = g[d.dt.date < today]
    print(min(int(played["week"].max()) + 1, int(g["week"].max())) if len(played) else 1)
except Exception:
    print(0)
PYEOF
)

echo "=== $(date '+%Y-%m-%d %H:%M %Z')  beat=$BEAT season=$SEASON week=$WEEK ===" >>"$LOG"
if [ "$WEEK" = "0" ]; then
    echo "could not determine the current week; skipping" >>"$LOG"; exit 0
fi

PROMPT="You are the fantasy-football improvement loop, $BEAT beat, ${SEASON} week ${WEEK}.
Read LOOP.md in $REPO first -- it is the state of record.

Tuesday: refresh the feeds, grade last week's recommended lineup against what actually
happened (recommended vs frozen-projection vs hindsight-optimal), append the result to the
season ledger, then run ONE improvement round from the queue.
Sunday: run week.py for the current week and record the recommended lineup BEFORE kickoff.

Obey the guardrails in LOOP.md. In particular: 2021 stays clean, no new hand-written
constants without a worker script that reproduces them, and never delete untracked files.
Commit and push what you change."

"$HOME/.local/bin/claude" -p "$PROMPT" \
    --permission-mode acceptEdits \
    >>"$LOG" 2>&1
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "WEEKLY LOOP FAILED (exit $STATUS)" >>"$LOG"
    osascript -e 'display notification "Weekly fantasy loop failed" with title "Fantasy" sound name "Basso"' 2>/dev/null
fi
exit $STATUS
