"""The replay harness must survive being killed.

A 360-replicate run is ~30 minutes. It used to hold every result in memory and
write once at the end, so a sleeping laptop or a stray Ctrl-C cost the whole
run. These tests cover the flush/resume path directly rather than through a
full replay, which needs a board and a season of nflverse data.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.replay import (  # noqa: E402
    ReplayResult,
    _flush,
    _load_completed,
)


def rows(*specs):
    return [
        ReplayResult(
            policy=p, replicate=r, slot=r % 12, rank=k, wins=7, points_for=1400.0
        )
        for p, r, k in specs
    ]


def test_load_completed_is_empty_without_a_file(tmp_path):
    done, existing = _load_completed(tmp_path / "nope.csv")
    assert done == set()
    assert existing.empty


def test_load_completed_is_empty_when_out_path_is_none():
    done, existing = _load_completed(None)
    assert done == set()
    assert existing.empty


def test_flush_then_load_round_trips_the_keys(tmp_path):
    out = tmp_path / "replay.csv"
    _flush(out, rows(("need_adp", 0, 3), ("season_sim", 0, 2)), pd.DataFrame())

    done, existing = _load_completed(out)
    assert done == {(0, "need_adp"), (0, "season_sim")}
    assert len(existing) == 2


def test_flush_appends_rather_than_overwrites(tmp_path):
    out = tmp_path / "replay.csv"
    first = _flush(out, rows(("need_adp", 0, 3)), pd.DataFrame())
    _flush(out, rows(("need_adp", 1, 5)), first)

    done, existing = _load_completed(out)
    assert done == {(0, "need_adp"), (1, "need_adp")}
    assert sorted(existing["rank"]) == [3, 5]


def test_flush_of_nothing_leaves_the_file_alone(tmp_path):
    out = tmp_path / "replay.csv"
    first = _flush(out, rows(("need_adp", 0, 3)), pd.DataFrame())
    same = _flush(out, [], first)

    assert same.equals(first)
    assert len(pd.read_csv(out)) == 1


def test_flush_leaves_no_temp_file_behind(tmp_path):
    """The write is rename-based so a crash mid-write cannot truncate results."""
    out = tmp_path / "replay.csv"
    _flush(out, rows(("need_adp", 0, 3)), pd.DataFrame())

    assert out.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_malformed_file_does_not_claim_work_is_done(tmp_path):
    """Better to redo the run than to silently skip replicates."""
    out = tmp_path / "replay.csv"
    pd.DataFrame({"something_else": [1, 2]}).to_csv(out, index=False)

    done, existing = _load_completed(out)
    assert done == set()
    assert existing.empty
