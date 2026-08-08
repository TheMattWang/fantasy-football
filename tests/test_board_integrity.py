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

import inspect
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
    validate_players,
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
# 2. The loaders
# --------------------------------------------------------------------------

def _load_via(module_name, loader_attr):
    mod = __import__(module_name)
    cls = next(
        o for _, o in vars(mod).items()
        if inspect.isclass(o)
        and o.__module__ == module_name
        and hasattr(o, loader_attr)
    )
    return getattr(cls.__new__(cls), loader_attr)()


@pytest.mark.parametrize(
    "module_name,loader_attr",
    [
        ("autocomplete_draft_assistant", "load_players"),
        ("interactive_draft_assistant", "_load_csv_data"),
        ("quick_draft_assistant", "load_players"),
    ],
)
def test_loader_produces_varying_vorp(module_name, loader_attr):
    """Every draft-day loader must produce a board with real spread."""
    try:
        players = _load_via(module_name, loader_attr)
    except (ImportError, AttributeError, StopIteration) as exc:
        pytest.skip(f"{module_name}: {exc}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = validate_players(players)

    assert report.ok, report.fatal
    assert report.stats["vorp_std"] > 0.5, (
        f"{module_name} loaded {len(players)} players with vorp_std="
        f"{report.stats['vorp_std']:.4f} -- this is the 2025 casing bug"
    )
    assert report.stats["vorp_max"] > 5.0


# --------------------------------------------------------------------------
# 3. Determinism across hash seeds
# --------------------------------------------------------------------------

SNIPPET = """
import warnings, inspect, sys; warnings.filterwarnings('ignore')
import autocomplete_draft_assistant as AC, interactive_draft_assistant as IA
ac = AC.AutocompleteDraftAssistant.__new__(AC.AutocompleteDraftAssistant)
top_ac = [p.name for p in sorted(ac.load_players(), key=AC.rank_key)[:5]]
cls = next(o for _, o in vars(IA).items()
           if inspect.isclass(o) and hasattr(o, '_load_csv_data'))
top_ia = [p.name for p in
          sorted(cls.__new__(cls)._load_csv_data(), key=IA._player_rank_key)[:5]]
sys.stdout.write(repr((top_ac, top_ia)))
"""


def _top_players_under_seed(seed):
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    out = subprocess.run(
        [sys.executable, "-c", SNIPPET],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return eval(out.stdout[out.stdout.rindex("(["):])


def test_recommendations_are_hash_seed_independent():
    """The 2025 failure signature: top pick changed on every launch."""
    results = [_top_players_under_seed(s) for s in (1, 42, 12345)]
    assert len(set(map(repr, results))) == 1, (
        f"ordering depends on PYTHONHASHSEED: {results}"
    )


def test_both_assistants_agree_on_top_pick():
    """Two tools reading the same board must not disagree about pick 1.01."""
    top_ac, top_ia = _top_players_under_seed(0)
    assert top_ac[0] == top_ia[0], (
        f"autocomplete says {top_ac[0]}, interactive says {top_ia[0]}"
    )


def test_top_pick_is_a_plausible_first_rounder():
    """Guards against 'ranks fine but ranks the wrong universe of players'."""
    top_ac, _ = _top_players_under_seed(0)
    board = pd.read_csv(BOARD)
    elite = set(board.nsmallest(15, "adp_rank")["player_name"]) | set(
        board.nlargest(15, "VORP")["player_name"]
    )
    assert top_ac[0] in elite, f"top pick {top_ac[0]!r} is not an elite player"
