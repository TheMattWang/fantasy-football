"""Evaluation for fantasy football draft strategies.

``replay`` supersedes ``backtesting``. The old harness scored strategies with
``0.35 * total_vorp`` -- the same quantity the agent maximized -- so it could
not detect a bad board, a bad objective, or a bad opponent model. It is also
downstream of ``hyperparameter_search``, which tuned coefficients against that
score and never completed a single trial (``best_score: -Infinity``).

Both were deleted on 2026-08-16, along with the v1 MCTS, injury and rookie
subsystems they belonged to. Nothing had imported them for a long time; they
survived only because deleting code feels riskier than leaving it. Git history
has them if they are ever wanted.

What remains here: ``replay`` scores a policy against real seasons, ``protocol``
enforces the holdout budget, ``robustness`` perturbs simulator constants to ask
whether an edge is real or learned, and ``objective_check`` is E2.
"""

from .replay import replay_season, summarize

__all__ = ["replay_season", "summarize"]
