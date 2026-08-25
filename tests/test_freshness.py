"""A board that is stale must fail loudly, and it must fail differently by context.

`validate_board` answers "is this board well-formed". It cannot answer "is this
board CURRENT", and that gap is not hypothetical: on 2026-08-16 the checked-in
2026 board carried an ECR snapshot from 2026-08-07 and no market ADP at all,
while every existing assertion stayed green. Rebuilding was a no-op because
`refresh` was plumbed nowhere, so nothing would ever have surfaced it.

That is the 2025 VORP bug one level up -- quiet degradation rather than a loud
error -- so these tests pin the loudness.
"""

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.assertions import (  # noqa: E402
    DRAFT_MAX_ECR_AGE_DAYS,
    MIN_MARKET_COVERAGE,
    BoardValidationError,
    load_provenance,
    validate_freshness,
)

TODAY = dt.date(2026, 8, 16)


def provenance(**overrides):
    base = {
        "season": 2026,
        "rows": 505,
        "config_provisional": False,
        "ecr": {"snapshot_date": "2026-08-15"},
        "market_adp": {"attached": True, "coverage": 0.89},
    }
    base.update(overrides)
    return base


def check(prov, **kwargs):
    kwargs.setdefault("today", TODAY)
    kwargs.setdefault("raise_on_fatal", False)
    kwargs.setdefault("emit_warnings", False)
    return validate_freshness(prov, **kwargs)


# --- a current board is fine ---------------------------------------------

def test_a_current_board_passes_even_for_draft_day():
    assert check(provenance(), for_draft=True).ok


def test_the_real_checked_in_board_is_current_enough_to_draft():
    """The actual artifact, not a fixture. If this fails, rebuild before drafting."""
    prov = load_provenance("data/processed/board_2026.csv")
    assert prov is not None, "no provenance sidecar -- rebuild with --refresh"
    report = validate_freshness(prov, for_draft=True, raise_on_fatal=False,
                                emit_warnings=False)
    assert report.ok, f"checked-in board is not draftable:\n{report}"


# --- staleness ------------------------------------------------------------

def test_a_stale_snapshot_is_fatal_on_draft_day():
    old = (TODAY - dt.timedelta(days=int(DRAFT_MAX_ECR_AGE_DAYS) + 5)).isoformat()
    report = check(provenance(ecr={"snapshot_date": old}), for_draft=True)
    assert not report.ok
    assert any("days old" in m for m in report.fatal)


def test_the_same_staleness_is_only_a_warning_for_research():
    """Severity is context-dependent on purpose: a stale board is a nuisance
    offline and a real problem on the clock.

    The bound is pinned explicitly in both arms so that `for_draft` is the ONLY
    thing varying -- the defaults differ (3 days vs 21), which would otherwise
    confound severity with threshold.
    """
    old = (TODAY - dt.timedelta(days=30)).isoformat()
    prov = provenance(ecr={"snapshot_date": old})

    strict = check(prov, for_draft=True, max_ecr_age_days=3)
    lenient = check(prov, for_draft=False, max_ecr_age_days=3)

    assert not strict.ok and any("days old" in m for m in strict.fatal)
    assert lenient.ok and any("days old" in m for m in lenient.warnings)


def test_the_age_bound_can_be_overridden():
    old = (TODAY - dt.timedelta(days=10)).isoformat()
    assert check(provenance(ecr={"snapshot_date": old}), for_draft=True,
                 max_ecr_age_days=30).ok


def test_age_is_measured_from_the_snapshot_not_the_download():
    """Re-downloading an unchanged parquet resets the file mtime. If freshness
    were read off the cache file, that would look like an update."""
    old = (TODAY - dt.timedelta(days=20)).isoformat()
    report = check(
        provenance(ecr={"snapshot_date": old,
                        "cache": {"age_days": 0.0}}),   # just downloaded
        for_draft=True,
    )
    assert not report.ok


# --- market ADP -----------------------------------------------------------

def test_missing_market_adp_is_fatal_on_draft_day():
    report = check(provenance(market_adp={"attached": False, "coverage": 0.0}),
                   for_draft=True)
    assert not report.ok
    assert any("ecr_sd" in m for m in report.fatal)


def test_thin_market_coverage_is_fatal_on_draft_day():
    report = check(
        provenance(market_adp={"attached": True,
                               "coverage": MIN_MARKET_COVERAGE - 0.1}),
        for_draft=True,
    )
    assert not report.ok
    assert any("fitted line" in m for m in report.fatal)


# --- missing or unreadable provenance -------------------------------------

@pytest.mark.parametrize("missing", [None, {}])
def test_absent_provenance_is_fatal_on_draft_day(missing):
    report = check(missing, for_draft=True)
    assert not report.ok
    assert any("provenance" in m for m in report.fatal)


def test_absent_provenance_names_the_fix():
    assert any("--refresh" in m for m in check(None, for_draft=True).fatal)


def test_an_unparseable_snapshot_date_is_caught():
    report = check(provenance(ecr={"snapshot_date": "not-a-date"}), for_draft=True)
    assert not report.ok


def test_raise_on_fatal_actually_raises():
    with pytest.raises(BoardValidationError):
        validate_freshness(None, for_draft=True, today=TODAY, emit_warnings=False)


# --- the provisional config is the user's call, not a pipeline failure ----

def test_provisional_settings_warn_but_never_block_the_draft():
    report = check(provenance(config_provisional=True), for_draft=True)
    assert report.ok
    assert any("PROVISIONAL" in m for m in report.warnings)


# --- the sidecar ----------------------------------------------------------

def test_load_provenance_returns_none_when_absent(tmp_path):
    assert load_provenance(tmp_path / "nope.csv") is None


def test_load_provenance_survives_a_corrupt_sidecar(tmp_path):
    """A truncated write must degrade to 'unknown', which is fatal for draft
    day, rather than crashing the tool on the clock."""
    csv = tmp_path / "board.csv"
    csv.with_suffix(".provenance.json").write_text("{ not json")
    assert load_provenance(csv) is None


def test_load_provenance_round_trips(tmp_path):
    csv = tmp_path / "board.csv"
    csv.with_suffix(".provenance.json").write_text(json.dumps(provenance()))
    assert load_provenance(csv)["rows"] == 505


# --- draft_day wiring -----------------------------------------------------

def test_draft_day_defaults_to_refusing_a_stale_board():
    import draft_day

    args = draft_day.build_parser().parse_args(["--slot", "4"])
    assert args.allow_stale is False


def test_draft_day_can_be_told_to_draft_anyway():
    import draft_day

    args = draft_day.build_parser().parse_args(["--slot", "4", "--allow-stale"])
    assert args.allow_stale is True


def test_the_board_write_leaves_no_temp_files_behind():
    """Atomic promote: a failed write must never leave a half-board where a
    working one used to be."""
    processed = Path("data/processed")
    assert not list(processed.glob("*.tmp")), "stale temp file from a failed write"


# --- stale vs. as-fresh-as-the-source-allows ------------------------------
#
# The consensus feed publishes WEEKLY -- measured at a 7-day median gap over 361
# snapshots -- so the flat 3-day draft bound was unsatisfiable four days out of
# every seven, however promptly the board was rebuilt. That is worse than a
# false alarm: a check that cannot be satisfied trains you to pass
# `--allow-stale`, which disables the check entirely.
#
# These pin the distinction. The gate must stop condemning a board for the
# source's publication schedule WITHOUT going quiet when we ship an old
# snapshot or the feed dies.

def latest(snapshot, cadence=7.0, **extra):
    return {"snapshot_date": snapshot, "cadence_days": cadence,
            "latest_available": snapshot, "is_latest_available": True, **extra}


def test_the_newest_snapshot_the_source_has_is_draftable_past_the_flat_bound():
    """4 days old, weekly feed, and nothing newer exists. Not our fault, and
    nothing a rebuild could fix -- this is the case that was failing."""
    old = (TODAY - dt.timedelta(days=4)).isoformat()
    assert int(DRAFT_MAX_ECR_AGE_DAYS) < 4, "this test assumes the flat bound is tighter"
    assert check(provenance(ecr=latest(old)), for_draft=True).ok


def test_shipping_an_old_snapshot_when_a_newer_one_exists_is_still_fatal():
    """The fault the gate actually exists to catch. Must survive the change."""
    old = (TODAY - dt.timedelta(days=10)).isoformat()
    prov = provenance(ecr={"snapshot_date": old, "cadence_days": 7.0,
                           "latest_available": TODAY.isoformat(),
                           "is_latest_available": False})
    report = check(prov, for_draft=True)
    assert not report.ok
    assert any("days old" in m for m in report.fatal)


def test_it_names_the_newer_snapshot_so_the_fix_is_obvious():
    old = (TODAY - dt.timedelta(days=10)).isoformat()
    prov = provenance(ecr={"snapshot_date": old, "cadence_days": 7.0,
                           "latest_available": "2026-08-14",
                           "is_latest_available": False})
    assert any("2026-08-14" in m for m in check(prov, for_draft=True).fatal)


def test_a_dead_feed_is_fatal_even_though_it_is_still_the_latest():
    """Two missed publications means nobody published for a fortnight. Being
    the newest snapshot in a feed that stopped updating is not freshness."""
    old = (TODAY - dt.timedelta(days=21)).isoformat()
    report = check(provenance(ecr=latest(old)), for_draft=True)
    assert not report.ok
    assert any("feed looks dead" in m for m in report.fatal)


def test_a_board_with_no_cadence_recorded_keeps_the_strict_bound():
    """Older sidecars predate the cadence fields. They must not silently get
    the relaxed treatment -- absent evidence is not evidence of freshness."""
    old = (TODAY - dt.timedelta(days=9)).isoformat()
    report = check(provenance(ecr={"snapshot_date": old}), for_draft=True)
    assert not report.ok


def test_the_real_board_records_its_source_cadence():
    """Provenance must actually carry the fields the gate reasons about."""
    prov = load_provenance("data/processed/board_2026.csv")
    assert prov is not None
    ecr = prov["ecr"]
    assert ecr.get("cadence_days"), "no measured cadence in provenance"
    assert ecr.get("is_latest_available") is not None
    assert ecr.get("latest_available")
