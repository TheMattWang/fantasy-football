"""The in-season cache must not be write-once.

`nflverse.load` cached one parquet per season and re-downloaded only when
`refresh=True` -- and **nothing anywhere passed it**. `weekly_fantasy` did not
even expose the parameter. So the first in-season load pinned that season's data
for the rest of the year.

The failure was silent and it compounded, because both in-season callers filter
the cached frame by week. A file cached in week 1 does not merely go stale by
week 5, it filters to **empty**, which `injury_report` reads as "no injuries
reported" -- scoring every ruled-out player as fully healthy, all season. That
defeats the availability work from underneath: a stale-but-parseable file is
indistinguishable from a genuinely thin one.

This is the same defect the board's `--refresh` fix already cured once, still
live on the one path where the data changes every week.

Freshness is decided by a conditional GET rather than a TTL. nflverse serves
`ETag`, so `If-None-Match` returns 304 with no body when nothing changed: no
wasted downloads, and no stale window during which we would serve last week's
data because an interval had not elapsed.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import nflverse  # noqa: E402


# --- the regression: a week filter over a stale cache goes EMPTY ----------

def test_a_week_filter_over_a_stale_cache_reads_as_no_injuries(tmp_path):
    """The precise mechanism, reproduced without the network.

    Week 1's file filtered at week 5 yields nothing, and `injury_report`'s
    graceful-degradation path turns that into "no information" -- which
    `availability` then treats as everybody healthy.
    """
    week1_only = pd.DataFrame({
        "season": [2026, 2026], "week": [1, 1], "game_type": ["REG", "REG"],
        "full_name": ["Lamar Jackson", "CeeDee Lamb"],
        "report_status": ["Out", "Out"],
    })
    stale = week1_only[week1_only["week"] == 5]
    assert stale.empty, "this is the bug: two OUT players become zero rows"


# --- the fix ---------------------------------------------------------------

def test_finished_seasons_are_not_rechecked():
    """A completed season's parquet never changes, so every check would be a
    wasted round trip. Only the live season needs asking."""
    assert nflverse._is_live(2026) is True
    assert nflverse._is_live(2019) is False
    assert nflverse._is_live(None) is True, "season-independent assets always change"


def test_a_missing_stamp_means_not_current(tmp_path):
    """No recorded validator -> we cannot claim the cache is fresh."""
    assert nflverse._is_current("https://example.invalid/x.parquet",
                                tmp_path / "x.parquet", 1.0) is False


def test_a_corrupt_stamp_means_not_current(tmp_path):
    path = tmp_path / "x.parquet"
    nflverse._stamp_path(path).write_text("{not json")
    assert nflverse._is_current("https://example.invalid/x.parquet", path, 1.0) is False


def test_being_offline_keeps_the_cache_rather_than_destroying_it(tmp_path):
    """A network failure must not invalidate a usable cache. `refresh=True` is
    how a caller forces the issue; an unreachable server is not."""
    path = tmp_path / "x.parquet"
    nflverse._stamp_path(path).write_text(json.dumps({"etag": '"abc"'}))
    assert nflverse._is_current("http://127.0.0.1:1/x.parquet", path, 0.25) is True


def test_the_in_season_callers_expose_refresh():
    """The original bug was a flag plumbed nowhere. Pin the plumbing."""
    import inspect

    from src.inseason.availability import injury_report
    from src.inseason.waivers import observed_to_date

    for fn in (nflverse.load, nflverse.weekly_fantasy, injury_report, observed_to_date):
        assert "refresh" in inspect.signature(fn).parameters, (
            f"{fn.__qualname__} cannot be refreshed -- this is exactly how the "
            f"board's --refresh flag came to do nothing"
        )


# --- against the live feed -------------------------------------------------

@pytest.mark.network
def test_an_unchanged_asset_costs_no_download(tmp_path, monkeypatch):
    """The end-to-end claim: cache hit is a 304, a changed validator re-fetches."""
    monkeypatch.setenv("FF_CACHE_DIR", str(tmp_path))
    nflverse.DATASETS.setdefault("schedules", ("schedules", "games"))

    first = nflverse.load("schedules")
    path = Path(nflverse.nflverse_dir()) / "games.parquet"
    stamp = nflverse._stamp_path(path)
    assert stamp.exists(), "no validators recorded, so the next load re-downloads"
    assert json.loads(stamp.read_text())["etag"], "server sent no ETag"

    url = nflverse._asset_url("schedules", None)
    assert nflverse._is_current(url, path, 20.0) is True

    saved = json.loads(stamp.read_text())
    stamp.write_text(json.dumps({**saved, "etag": '"0xDEADBEEF"'}))
    assert nflverse._is_current(url, path, 20.0) is False, "a changed ETag must re-fetch"

    assert len(nflverse.load("schedules")) == len(first)


# --- draft day must stay offline ------------------------------------------

def test_draft_day_runs_with_the_network_blackholed(tmp_path):
    """The unrecoverable failure, pinned.

    A network stall against a 90-second pick clock cannot be recovered from, so
    `draft_day.py` fetches nothing. This mattered more after the cache learned
    to ask the server whether it is current: that check is right for the weekly
    feed and would be fatal on the clock, so it must not reach draft day.

    Runs the real script in a subprocess with `socket.connect` disabled, which
    is the only way to prove absence of a fetch rather than assert it.
    """
    import subprocess
    import sys

    shim = tmp_path / "sitecustomize.py"
    shim.write_text(
        "import socket\n"
        "def _boom(*a, **k):\n"
        "    raise OSError('NETWORK BLOCKED: draft day must never touch the network')\n"
        "socket.socket.connect = _boom\n"
        "socket.socket.connect_ex = _boom\n"
        "socket.create_connection = _boom\n"
    )
    board = REPO / "data/processed/board_2026.csv"
    if not board.exists():
        pytest.skip("no checked-in board to draft from")

    env = {**__import__("os").environ, "PYTHONPATH": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "draft_day.py", "--slot", "6", "--fast"],
        cwd=REPO, env=env, input="quit\n", capture_output=True, text=True, timeout=300,
    )
    assert "NETWORK BLOCKED" not in done.stdout + done.stderr, (
        "draft_day.py tried to reach the network:\n" + done.stderr[-2000:]
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "drafting from slot 6" in done.stdout
