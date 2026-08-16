#!/bin/bash
# Daily board refresh, run by com.claude.fantasy-board-refresh.
#
# Why this exists: `refresh` was plumbed nowhere, so rebuilding the board was a
# no-op and the 2026 board sat a week stale with no market ADP while every
# assertion stayed green. Consensus moves daily through August on injuries,
# holdouts and depth charts, so the board has to be pulled forward on a clock
# rather than when someone remembers.
#
# Two properties this script must have:
#   1. NEVER promote a degraded board. --require-market makes a missing FFC
#      attach fatal, so a bad fetch leaves yesterday's good board in place
#      rather than replacing it with one whose P(next) is computed from a
#      ~2x-too-wide spread.
#   2. FAIL LOUDLY. Silence is the exact failure mode being fixed here, so a
#      non-zero exit sends an iMessage rather than just landing in the log.

set -uo pipefail

REPO="/Users/mattwang/Documents/fantasy/fantasy-football"
PY="$REPO/.venv/bin/python"
SEASON="${FANTASY_SEASON:-2026}"
BOARD="$REPO/data/processed/board_${SEASON}.csv"
STAMP=$(date "+%Y-%m-%d %H:%M:%S")

cd "$REPO" || { echo "[$STAMP] cannot cd to $REPO"; exit 1; }

echo "=== [$STAMP] refreshing board for $SEASON ==="

# --top 0 because nobody reads the table from a cron log; the provenance block
# printed after it is the part worth having.
if "$PY" -m src.projections.board \
        --season "$SEASON" --top 0 --refresh --require-market \
        --out "$BOARD"; then
    echo "[$STAMP] board refreshed OK"
    exit 0
fi

STATUS=$?
echo "[$STAMP] BOARD REFRESH FAILED (exit $STATUS) -- yesterday's board is untouched"

# Notify rather than fail quietly. If this is running at all, the draft is close.
osascript -e 'display notification "Board refresh failed — see daily-log" with title "Fantasy"' 2>/dev/null || true

exit "$STATUS"
