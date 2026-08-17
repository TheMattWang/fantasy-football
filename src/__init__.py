"""Fantasy football draft and in-season tooling.

Two entry points, both at the repo root:

    draft_day.py    offline draft assistant. Recommends consensus; the season
                    simulation fills the table as context and its disagreements
                    are shown but not acted on.
    week.py         in-season start/sit. Ranks the roster by the preseason
                    projection updated with results through last week.

The packages, in the order data flows through them:

    data          ingestion -- nflverse, FantasyPros ECR, FFC market ADP,
                  league config, and the board validators
    projections   consensus rank -> points curve, the board, and validation
    simulation    player uncertainty and the head-to-head season
    draft         the draft engine, opponent models, and the search policies
    inseason      start/sit and waiver valuation
    evaluation    replay against real seasons, plus the holdout protocol

Deliberately empty of imports. This module used to eagerly pull in a v1 MCTS
and injury-modelling cluster, so every `import src.anything` executed it --
including in tests, and including after the code it advertised had been
superseded. Nothing imports the package root, so nothing here needs to.
"""

__version__ = "3.0.0"
