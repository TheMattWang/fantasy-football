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
    MAX_SEALED_TOUCHES,
    SEALED_SEASONS,
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
    """2019 used to be the example here. It is now SEALED, so this uses a
    season that genuinely belongs to nobody."""
    check(2017, protocol="tune", registry_path=registry)


# --- the replacement holdout ---------------------------------------------

@pytest.mark.parametrize("season", SEALED_SEASONS)
def test_tuning_against_a_sealed_season_is_refused(season, registry):
    """2019-2021 were bought back with ADP-anchored boards after the 2024/2025
    gate was found spent. They are worth something exactly once."""
    with pytest.raises(GateError, match="held-out sealed"):
        check(season, protocol="tune", registry_path=registry)


@pytest.mark.parametrize("season", SEALED_SEASONS)
def test_a_registered_experiment_may_touch_the_seal(season, registry):
    check(season, protocol="gate", register="shrinkage_v1", registry_path=registry)


def test_the_sealed_budget_is_separate_from_the_spent_gate_budget(registry):
    """The load-bearing one. The 2024/2025 budget is already exhausted in the
    real registry; if the seal counted against that same total it would arrive
    unusable, which would silently waste the whole exercise."""
    for _ in range(MAX_GATE_TOUCHES):
        record_touch("shrinkage_v1", 2025, registry)

    with pytest.raises(GateError, match="gate budget is spent"):
        check(2025, protocol="gate", register="shrinkage_v1", registry_path=registry)

    # ...and the seal is untouched by that.
    check(2019, protocol="gate", register="shrinkage_v1", registry_path=registry)


def test_the_seal_can_itself_be_spent(registry):
    for season in SEALED_SEASONS[:MAX_SEALED_TOUCHES]:
        check(season, protocol="gate", register="shrinkage_v1", registry_path=registry)
        record_touch("shrinkage_v1", season, registry)

    with pytest.raises(GateError, match="sealed budget is spent"):
        check(2019, protocol="gate", register="shrinkage_v1", registry_path=registry)


def test_touches_spent_can_be_scoped_to_one_holdout(registry):
    record_touch("shrinkage_v1", 2025, registry)
    record_touch("shrinkage_v1", 2019, registry)

    loaded = load_registry(registry)
    assert protocol.touches_spent(loaded) == 2
    assert protocol.touches_spent(loaded, GATE_SEASONS) == 1
    assert protocol.touches_spent(loaded, SEALED_SEASONS) == 1


def test_the_two_holdouts_do_not_overlap():
    assert not set(GATE_SEASONS) & set(SEALED_SEASONS)
    assert not set(TUNING_SEASONS) & set(SEALED_SEASONS)


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

    # The 2024/2025 budget is over and must stay recorded as over.
    assert protocol.touches_spent(registry, GATE_SEASONS) >= MAX_GATE_TOUCHES

    # Every entry states a hypothesis. Touches are NOT required: the registry
    # now holds pre-registrations for the sealed seasons as well as the record
    # of spent ones, and an entry with an empty touch list is the correct shape
    # for a hypothesis committed before its run -- which is the entire point of
    # committing it.
    for name, entry in registry["experiments"].items():
        assert entry["hypothesis"], f"{name} has no recorded hypothesis"
        assert isinstance(entry["touches"], list), f"{name} has a malformed ledger"
        for touch in entry["touches"]:
            assert touch.get("season"), f"{name} has a touch with no season"


def test_a_sealed_pre_registration_exists_and_is_unspent():
    """The seal is worth something exactly once, so what it will be spent on is
    committed in advance rather than decided after seeing a number."""
    registry = load_registry()
    assert protocol.touches_spent(registry, SEALED_SEASONS) <= MAX_SEALED_TOUCHES
    pending = [name for name, entry in registry["experiments"].items()
               if not entry["touches"]]
    assert pending, "no pre-registered experiment is waiting on the seal"
