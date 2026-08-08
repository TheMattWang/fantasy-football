"""The robustness harness has to actually perturb, and actually restore.

A patch that leaks would silently corrupt every later measurement in the same
process, which is a worse failure than the one the test is looking for.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from src.evaluation.robustness import (  # noqa: E402
    PERTURBATIONS,
    perturbed,
    verdict,
)
from src.simulation import distributions as dist  # noqa: E402
from src.simulation import season as season_mod  # noqa: E402
from src.simulation.season import lineup_points, plan_roster  # noqa: E402
from test_season import CONFIG, FLEX_SPEC, NO_FLOOR, make_varying_samples  # noqa: E402


def test_every_named_perturbation_changes_something():
    for name in PERTURBATIONS:
        before = {
            "win": dict(season_mod.WAIVER_WIN_RATE),
            "floor": dict(season_mod.DEFAULT_WAIVER_FLOOR),
            "deep": dict(season_mod.DEEP_WAIVER_FLOOR),
            "cv": {k: list(v) for k, v in dist.WEEKLY_CV.items()},
            "sigma": dict(dist.SIGMA_PROJ),
            "ret": dist.P_RETURN_AFTER_OUT,
            "flex": season_mod.FLEX_OMNISCIENT_DEFAULT,
        }
        with perturbed(name):
            after = {
                "win": dict(season_mod.WAIVER_WIN_RATE),
                "floor": dict(season_mod.DEFAULT_WAIVER_FLOOR),
                "deep": dict(season_mod.DEEP_WAIVER_FLOOR),
                "cv": {k: list(v) for k, v in dist.WEEKLY_CV.items()},
                "sigma": dict(dist.SIGMA_PROJ),
                "ret": dist.P_RETURN_AFTER_OUT,
                "flex": season_mod.FLEX_OMNISCIENT_DEFAULT,
            }
        assert after != before, f"{name} perturbed nothing"


def test_constants_are_restored_afterwards():
    original = dict(season_mod.WAIVER_WIN_RATE)
    with perturbed("waiver_win_low"):
        assert season_mod.WAIVER_WIN_RATE != original
    assert season_mod.WAIVER_WIN_RATE == original


def test_constants_are_restored_even_after_an_exception():
    original = dict(season_mod.WAIVER_WIN_RATE)
    with pytest.raises(RuntimeError):
        with perturbed("waiver_win_low"):
            raise RuntimeError("boom")
    assert season_mod.WAIVER_WIN_RATE == original


def test_baseline_is_a_no_op():
    original = dict(season_mod.WAIVER_WIN_RATE)
    with perturbed("baseline"):
        assert season_mod.WAIVER_WIN_RATE == original


def test_unknown_perturbation_is_rejected():
    with pytest.raises(ValueError, match="unknown perturbation"):
        with perturbed("not_a_thing"):
            pass


def test_win_rates_stay_probabilities():
    with perturbed("waiver_win_high"):
        assert all(0.0 <= v <= 1.0 for v in season_mod.WAIVER_WIN_RATE.values())


def test_flex_hindsight_control_actually_moves_the_score():
    """The positive control must fire, or the whole test is decorative."""
    samples = make_varying_samples(FLEX_SPEC)
    plan = plan_roster(list(samples.index), samples, CONFIG)

    honest = lineup_points(samples, plan, waiver_floor=NO_FLOOR)
    with perturbed("flex_hindsight"):
        cheating = lineup_points(samples, plan, waiver_floor=NO_FLOOR)

    assert cheating.mean() > honest.mean()


def test_an_explicit_argument_still_beats_the_module_default():
    samples = make_varying_samples(FLEX_SPEC)
    plan = plan_roster(list(samples.index), samples, CONFIG)

    with perturbed("flex_hindsight"):
        forced = lineup_points(
            samples, plan, waiver_floor=NO_FLOOR, flex_omniscient=False
        )
    honest = lineup_points(samples, plan, waiver_floor=NO_FLOOR)
    assert np.array_equal(forced, honest)


def test_verdict_flags_a_perturbation_that_kills_the_edge():
    frame = pd.DataFrame([
        {"perturbation": "baseline", "season": 2022, "gain": 0.5,
         "se": 0.1, "retention": 1.0},
        {"perturbation": "waiver_win_low", "season": 2022, "gain": 0.4,
         "se": 0.1, "retention": 0.8},
        {"perturbation": "flex_hindsight", "season": 2022, "gain": -0.2,
         "se": 0.1, "retention": -0.4},
    ])
    table = verdict(frame).set_index("perturbation")
    assert "baseline" not in table.index
    assert bool(table.loc["waiver_win_low", "survives"]) is True
    assert bool(table.loc["flex_hindsight", "survives"]) is False
