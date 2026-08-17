"""Regression tests for the 2025 draft-day failure.

The 2025 agent finished second-to-last because `interactive_draft_assistant.py`
read `row.get('vorp', 0.0)` while the board column was `VORP`. Every player
loaded at 0.0, every score at a position tied, and the tie broke on Python's
per-process randomized string hash -- so the top recommendation at pick 1.01
changed on every launch.

These tests pin the three properties that would have caught it:
  1. the board CSV loads with real, varying VORP
  2. a zeroed board is rejected as fatal, not silently accepted
  3. recommendations are deterministic across PYTHONHASHSEED
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
BOARD = REPO / "data" / "raw" / "draft_board.csv"

sys.path.insert(0, str(REPO))

from src.data.assertions import (  # noqa: E402
    BoardValidationError,
    validate_board,
)


@pytest.fixture(scope="module")
def board():
    if not BOARD.exists():
        pytest.skip(f"{BOARD} not present")
    return pd.read_csv(BOARD)


# --------------------------------------------------------------------------
# 1. The board itself
# --------------------------------------------------------------------------

def test_board_vorp_column_is_uppercase(board):
    """Pin the actual casing. If the file ever changes, the loaders must too."""
    assert "VORP" in board.columns
    assert "vorp" not in board.columns


def test_board_passes_structural_validation(board):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = validate_board(board)
    assert report.ok
    assert report.stats["vorp_std"] > 0


def test_zeroed_vorp_is_fatal(board):
    """The exact 2025 bug must raise, not warn."""
    broken = board.copy()
    broken["VORP"] = 0.0
    with pytest.raises(BoardValidationError, match="zero variance"):
        validate_board(broken, emit_warnings=False)


def test_missing_vorp_column_is_fatal(board):
    broken = board.drop(columns=["VORP"])
    with pytest.raises(BoardValidationError):
        validate_board(broken, emit_warnings=False)


def test_current_board_fails_market_agreement(board):
    """Documents a known-bad property of the 2025 board.

    This board disagrees with market ADP at rho ~= 0.25. That is a projections
    problem (Phase 2), not a plumbing problem, so it is a warning rather than
    fatal -- but it must be *reported*, because silence is what let it ship.
    Delete this test once the market-anchored board lands.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_board(board)
    assert report.stats["vorp_adp_spearman"] < 0.75
    assert any("market ADP" in str(w.message) for w in caught)


# --------------------------------------------------------------------------
# 2. The shipping recommender
# --------------------------------------------------------------------------
#
# This section used to drive `autocomplete_draft_assistant`,
# `interactive_draft_assistant` and `quick_draft_assistant` -- the three v1 tools
# whose shared loader bug lost the 2025 season. Those files are gone, and a
# regression test against a deleted file tests nothing.
#
# The failure it guarded against is not gone, though, so it is re-pointed at
# `draft_day.py`, which is what actually drafts now. The 2025 signature was:
# every VORP loaded as 0.0, every candidate at a position tied, and the tie broke
# on Python's per-process randomized string hash -- so the top recommendation
# changed on every launch.

SNIPPET = """
import sys, warnings; warnings.filterwarnings('ignore')
import pandas as pd
from src.data.league_config import provisional_config
from src.draft.engine import DraftBoard, DraftSim
from src.draft.search import NeedAdpPolicy

frame = pd.read_csv('data/processed/board_2026.csv')
config = provisional_config()
board = DraftBoard.from_frame(frame)
sim = DraftSim(board, config.num_teams, config.total_rounds or 15)
policy = NeedAdpPolicy(board, config)
picks = []
for _ in range(5):
    row = policy.choose(sim, sim.on_the_clock)
    picks.append(str(board.names[row]))
    sim.make_pick(row)
sys.stdout.write(repr(picks))
"""


def _picks_under_seed(seed):
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    out = subprocess.run(
        [sys.executable, "-c", SNIPPET],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return eval(out.stdout[out.stdout.rindex("["):])


def test_draft_day_picks_are_hash_seed_independent():
    """The 2025 failure signature, against the tool that ships today."""
    results = [_picks_under_seed(s) for s in (1, 42, 12345)]
    assert len(set(map(repr, results))) == 1, (
        f"ordering depends on PYTHONHASHSEED: {results}"
    )


def test_the_shipping_board_has_real_vorp_spread():
    """A board whose VORP is flat cannot rank anything, which is exactly what
    the 2025 casing bug produced."""
    frame = pd.read_csv(REPO / "data" / "processed" / "board_2026.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = validate_board(frame[frame["is_streamed"] == False])  # noqa: E712

    assert report.ok, report.fatal
    assert report.stats["vorp_std"] > 0.5
    assert report.stats["vorp_nonzero_frac"] > 0.9
