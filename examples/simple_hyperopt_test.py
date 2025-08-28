#!/usr/bin/env python3
"""
Simple Hyperparameter Optimization Test
=======================================

A simplified test to verify the hyperparameter optimization system works correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.player import Player, PlayerPool
from src.core.draft import DraftState, LeagueSettings
from src.utils.data_loader import create_sample_player_pool, get_default_league_settings
from src.evaluation.backtesting import DraftBacktester
from src.evaluation.hyperparameter_search import HyperparameterOptimizer, ParameterSpace


def create_simple_parameter_space():
    """Create a simple parameter space for testing"""
    space = ParameterSpace()
    
    # Just a few parameters for testing
    space.add_parameter('risk_penalty', 'float', (0.1, 0.3))
    space.add_parameter('vorp_weight', 'float', (0.8, 1.2))
    space.add_parameter('early_qb_penalty', 'float', (0.0, 0.5))
    
    return space


def create_simple_strategy_factory():
    """Create a simple strategy factory for testing"""
    
    def strategy_factory(parameters):
        """Create a simple parameterized strategy"""
        
        class SimpleTestStrategy:
            def __init__(self, params):
                self.params = params
                self.last_pick_reasoning = "Simple test strategy"
            
            def search(self, draft_state):
                available = list(draft_state.available_players)
                if not available:
                    return None
                
                # Simple scoring based on parameters
                best_score = -1
                best_player = None
                
                for player in available:
                    # Base score
                    score = player.vorp * self.params.get('vorp_weight', 1.0)
                    
                    # Risk penalty
                    risk = player.metadata.get('risk_sigma', 0.2)
                    score -= risk * self.params.get('risk_penalty', 0.2)
                    
                    # QB penalty in early rounds
                    if player.position == 'QB' and draft_state.current_round <= 3:
                        score -= self.params.get('early_qb_penalty', 0.3)
                    
                    if score > best_score:
                        best_score = score
                        best_player = player
                
                return best_player
        
        return SimpleTestStrategy(parameters)
    
    return strategy_factory


def main():
    """Run simple hyperparameter optimization test"""
    
    print("🔧 Simple Hyperparameter Optimization Test")
    print("=" * 50)
    
    # Create test data
    print("\n📊 Creating test data...")
    player_pool = create_sample_player_pool(n_players=100)  # Smaller for speed
    league_settings = get_default_league_settings()
    
    # Add simple metadata
    for i, player in enumerate(player_pool):
        player.metadata['bye_week'] = 4 + (i % 11)
        player.metadata['risk_sigma'] = np.random.uniform(0.1, 0.4)
        player.metadata['injury_risk_score'] = np.random.uniform(0.1, 0.5)
    
    print(f"✅ Created {len(player_pool)} players")
    
    # Setup backtester with reduced complexity
    print("\n🔬 Setting up backtester...")
    backtester = DraftBacktester(player_pool, league_settings)
    backtester.n_simulations = 3  # Very small for testing
    backtester.parallel_workers = 1  # No parallel for simplicity
    
    # Create parameter space and factory
    parameter_space = create_simple_parameter_space()
    strategy_factory = create_simple_strategy_factory()
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(backtester, strategy_factory, parameter_space)
    
    # Test random search with just a few configurations
    print("\n🎲 Testing random search...")
    results_df = optimizer.random_search(
        n_configurations=5,  # Very small for testing
        n_trials_per_config=2,  # Very small for testing
        parallel=False,  # No parallel for simplicity
        save_results=False
    )
    
    print(f"\n📊 Results:")
    if not results_df.empty:
        print(f"   ✅ Found {len(results_df)} valid configurations")
        print(f"   🏆 Best score: {optimizer.best_score:.4f}")
        print(f"   🎯 Best parameters: {optimizer.best_parameters}")
        
        # Show all results
        print(f"\n📋 All results:")
        for i, row in results_df.iterrows():
            print(f"   Config {i+1}: Score {row['mean_score']:.4f} - {row['parameters']}")
    else:
        print("   ⚠️  No valid results found")
    
    print(f"\n✅ Simple hyperparameter optimization test complete!")


if __name__ == "__main__":
    main()
