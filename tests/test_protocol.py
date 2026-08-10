"""The holdout has to be protected by code, not by remembering.

2024/2025 are the final test set with a pre-registered budget of three touches.
These tests pin the refusals: the value of a held-out season is exactly the
discipline around it, and discipline that is not executable decays.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import protocol  # noqa: E402
from src.evaluation.protocol import (  # noqa: E402
    GATE_SEASONS,
    MAX_GATE_TOUCHES,
    TUNING_SEASONS,
    GateError,
    check,
    load_registry,
    record_touch,
)


@pytest.fixture
def registry(tmp_path):
    """A registry file holding one pre-registered experiment."""
    path = tmp_path / "gate_registry.json"
    path.write_text(
        json.dumps(
            {
                "experiments": {
                    "shrinkage_v1": {
                        "hypothesis": "pessimism closes the -1.667 gap",
                        "registered": "2026-08-07",
                        "touches": [],
                    }
                }
            }
        )
    )
    return path


def test_the_two_season_sets_do_not_overlap():
    assert not set(TUNING_SEASONS) & set(GATE_SEASONS)


# --- the refusals ---------------------------------------------------------

@pytest.mark.parametrize("season", GATE_SEASONS)
def test_tuning_against_a_gate_season_is_refused(season, registry):
    """The important one: a casual run must not silently burn the holdout."""
    with pytest.raises(GateError, match="held-out"):
        check(season, protocol="tune", registry_path=registry)


@pytest.mark.parametrize("season", TUNING_SEASONS)
def test_gating_against_a_tuning_season_is_refused(season, registry):
    with pytest.raises(GateError, match="only for the held-out"):
        check(season, protocol="gate", register="shrinkage_v1", registry_path=registry)


def test_gate_without_a_register_name_is_refused(registry):
    with pytest.raises(GateError, match="requires register"):
        check(2025, protocol="gate", registry_path=registry)


def test_gate_with_an_unregistered_name_is_refused(registry):
    with pytest.raises(GateError, match="BEFORE"):
        check(2025, protocol="gate", register="whoops", registry_path=registry)


def test_an_unknown_protocol_is_refused(registry):
    with pytest.raises(GateError, match="unknown protocol"):
        check(2023, protocol="peek", registry_path=registry)


def test_the_budget_is_shared_across_experiments(tmp_path):
    """Splitting the budget across names would launder extra looks."""
    path = tmp_path / "gate_registry.json"
    path.write_text(json.dumps({"experiments": {}}))
    for i in range(MAX_GATE_TOUCHES):
        name = f"exp_{i}"
        registry = load_registry(path)
        registry["experiments"][name] = {"hypothesis": "h", "touches": []}
        path.write_text(json.dumps(registry))
        check(2025, protocol="gate", register=name, registry_path=path)
        record_touch(name, 2025, path)

    registry = load_registry(path)
    registry["experiments"]["exp_fresh"] = {"hypothesis": "h", "touches": []}
    path.write_text(json.dumps(registry))
    with pytest.raises(GateError, match="budget is spent"):
        check(2025, protocol="gate", register="exp_fresh", registry_path=path)


def test_exhausting_the_budget_on_one_experiment_closes_the_gate(registry):
    for _ in range(MAX_GATE_TOUCHES):
        check(2025, protocol="gate", register="shrinkage_v1", registry_path=registry)
        record_touch("shrinkage_v1", 2025, registry)

    with pytest.raises(GateError, match="budget is spent"):
        check(2025, protocol="gate", register="shrinkage_v1", registry_path=registry)


def test_a_touch_cannot_be_recorded_for_an_unregistered_name(registry):
    with pytest.raises(GateError, match="unregistered"):
        record_touch("never_registered", 2025, registry)


# --- the permissions ------------------------------------------------------

@pytest.mark.parametrize("season", TUNING_SEASONS)
def test_tuning_seasons_are_unaffected(season, registry):
    check(season, protocol="tune", registry_path=registry)


def test_seasons_outside_the_holdout_stay_tunable(registry):
    check(2019, protocol="tune", registry_path=registry)


@pytest.mark.parametrize("season", GATE_SEASONS)
def test_a_registered_experiment_may_touch_the_holdout(season, registry):
    check(season, protocol="gate", register="shrinkage_v1", registry_path=registry)
    record_touch("shrinkage_v1", season, registry)

    entry = load_registry(registry)["experiments"]["shrinkage_v1"]
    assert len(entry["touches"]) == 1
    assert entry["touches"][0]["season"] == season
    assert entry["touches"][0]["at"]


def test_a_touch_carries_the_caller_note(registry):
    record_touch("shrinkage_v1", 2024, registry, note="replay ['need_adp']")

    touch = load_registry(registry)["experiments"]["shrinkage_v1"]["touches"][0]
    assert touch["note"] == "replay ['need_adp']"


# --- the ledger -----------------------------------------------------------

def test_a_missing_registry_is_created_empty_and_registers_nothing(tmp_path):
    path = tmp_path / "gate_registry.json"
    assert load_registry(path) == {"experiments": {}}
    assert json.loads(path.read_text()) == {"experiments": {}}


def test_the_registry_round_trips_through_load_and_record(registry):
    record_touch("shrinkage_v1", 2025, registry)
    record_touch("shrinkage_v1", 2024, registry)

    reloaded = load_registry(registry)["experiments"]["shrinkage_v1"]
    assert reloaded["hypothesis"] == "pessimism closes the -1.667 gap"
    assert reloaded["registered"] == "2026-08-07"
    assert [t["season"] for t in reloaded["touches"]] == [2025, 2024]
    assert protocol.touches_spent(load_registry(registry)) == 2


def test_recording_leaves_no_temp_file_behind(registry, tmp_path):
    record_touch("shrinkage_v1", 2025, registry)

    assert list(tmp_path.glob("*.tmp")) == []


def test_replay_refuses_a_gate_season_before_loading_anything():
    """The guard runs first, so the refusal costs no board build and no data read."""
    from src.evaluation.replay import replay_season

    with pytest.raises(GateError, match="held-out"):
        replay_season(2025, n_replicates=1)


def test_the_checked_in_registry_is_valid_and_records_the_spent_budget():
    """The repo's own registry: parseable, and honest about the holdout.

    This test originally asserted the budget was still *whole*. It was not:
    replay artifacts for both holdout seasons were already on disk, so 2024/2025
    had been spent long before any enforcement code existed. The registry was
    seeded from those artifacts rather than reset, because a ledger that
    forgets is worse than no ledger. Pin the real state so nobody re-derives
    the comfortable version.
    """
    registry = load_registry()
    assert isinstance(registry["experiments"], dict)
    assert protocol.touches_spent(registry) >= MAX_GATE_TOUCHES
    for name, entry in registry["experiments"].items():
        assert entry["hypothesis"], f"{name} has no recorded hypothesis"
        assert entry["touches"], f"{name} claims no touches"
