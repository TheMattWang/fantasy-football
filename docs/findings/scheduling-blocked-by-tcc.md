# Scheduling — macOS blocks it, and the plist that "existed" never ran

**Verdict: BLOCKED on one user action. Both LaunchAgents are written, linted and
correct; macOS will not let either read the repo until `/bin/bash` is granted Full
Disk Access.**

Found 2026-08-25 while installing the loop machinery (R4).

## What happened

`scripts/com.claude.fantasy-board-refresh.plist` had been written and committed but never
installed. Installing it produced exit **126** and this in the log:

```
/bin/bash: /Users/mattwang/Documents/fantasy/fantasy-football/scripts/refresh_board.sh:
Operation not permitted
```

A direct probe from inside `launchd` settles it:

```
read repo dir:    DENIED
read a file:      DENIED
run venv python:  DENIED
```

`~/Documents` is protected by macOS TCC (Transparency, Consent and Control). A LaunchAgent
runs as the user but **without** the user's TCC grants, so it cannot read anything under
`~/Documents` — script, data or virtualenv. The existing `com.claude.morning-brief` agent
works only because it lives at `~/.claude/morning-send.sh`, outside the protected tree.

## Why this is worth a finding rather than a shrug

The board sat **11 days stale** into draft month, and the reason given was that the
LaunchAgent was "written but never installed". That framing was too kind. Installing it
would **not** have helped: it would have failed at 06:30 every morning, written a line
into a log nobody reads, and left exactly the same stale board — while creating the
impression the problem was handled.

That is this project's signature failure, one layer further out: quiet degradation rather
than a loud error, this time in the thing built to prevent quiet degradation. So the agent
is now **removed** rather than left installed and failing, because a scheduled job that
cannot work is worse than no scheduled job.

## The fix — one action, and it is Matt's

System Settings → Privacy & Security → **Full Disk Access** → add `/bin/bash`.
(⌘⇧G in the file picker to type the path.) Then:

```bash
cp scripts/com.claude.fantasy-board-refresh.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.claude.fantasy-board-refresh.plist
tail -f ~/.claude/daily-log/fantasy-board-refresh.log     # confirm it actually ran
```

Verify with `launchctl list | grep fantasy` — the middle column is the last exit status,
and it must be **0**, not 126.

The same grant unblocks `com.claude.fantasy-weekly.plist`, which drives the in-season beat.

## Until then

Nothing is automated, and the board is only as fresh as the last manual refresh:

```bash
./.venv/bin/python -m src.projections.board --season 2026 --refresh --require-market \
    --out data/processed/board_2026.csv
```

`draft_day.py` refuses to start on a stale board, so the failure mode is a refusal on the
day rather than a quietly wrong board — which is the right way round, but it is a
backstop, not a schedule.

**Alternative if the grant is unwanted:** move the repo outside `~/Documents` (for example
`~/dev/fantasy-football`), which removes the restriction entirely. That is a bigger change
and it is Matt's call, so it has not been made.
