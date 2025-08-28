"""
Evaluation and backtesting modules for fantasy football draft strategies.

This module provides comprehensive evaluation tools including:
- Historical draft backtesting
- Hyperparameter optimization
- Strategy comparison
- Performance metrics
"""

from .backtesting import DraftBacktester, BacktestResults
from .hyperparameter_search import HyperparameterOptimizer, ParameterGrid
from .metrics import DraftMetrics, SeasonPerformanceEvaluator

__all__ = [
    'DraftBacktester', 'BacktestResults',
    'HyperparameterOptimizer', 'ParameterGrid', 
    'DraftMetrics', 'SeasonPerformanceEvaluator'
]
