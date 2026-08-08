"""Season simulation: distributions, lineups, and the head-to-head objective."""

from .distributions import (
    P_RETURN_AFTER_OUT,
    WEEKLY_CV,
    SampleSet,
    build_samples,
    cv_for,
    stay_probability,
)

__all__ = [
    "P_RETURN_AFTER_OUT",
    "WEEKLY_CV",
    "SampleSet",
    "build_samples",
    "cv_for",
    "stay_probability",
]
