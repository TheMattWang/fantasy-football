"""Evaluation for fantasy football draft strategies.

``replay`` supersedes ``backtesting``. The old harness scored strategies with
``0.35 * total_vorp`` -- the same quantity the agent maximized -- so it could
not detect a bad board, a bad objective, or a bad opponent model. It is also
downstream of ``hyperparameter_search``, which tuned coefficients against that
score and never completed a single trial (``best_score: -Infinity``).

Neither is imported here any more: they pull in matplotlib and, more to the
point, importing them invites using them. They remain on disk for reference and
are scheduled for deletion.
"""

from .replay import replay_season, summarize

__all__ = ["replay_season", "summarize"]
