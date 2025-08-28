"""
Fantasy Football Draft Strategy Package
======================================

A comprehensive package for fantasy football draft strategy including:
- MCTS-based draft optimization
- Injury risk modeling
- Rookie performance prediction
- Player valuation and ranking

Quick Start:
    from src.strategies import MCTSStrategy
    from src.utils import load_default_data
    
    data = load_default_data()
    strategy = MCTSStrategy()
    pick = strategy.make_pick(draft_state)
"""

__version__ = "2.0.0"
__author__ = "Fantasy Football MCTS Team"

# Main imports for easy access (only import what exists)
from src.core.player import Player, PlayerPool
from src.core.draft import DraftState, LeagueSettings
from src.utils.data_loader import load_default_data, load_injury_enhanced_data

__all__ = [
    'Player', 'PlayerPool', 
    'DraftState', 'LeagueSettings',
    'load_default_data', 'load_injury_enhanced_data'
]
