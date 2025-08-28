#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Optimization Demo
==============================================

This example demonstrates the complete backtesting and hyperparameter optimization
system for fantasy football draft strategies, showing how to find optimal MCTS
parameters through systematic search.

Usage:
    python examples/hyperparameter_optimization_demo.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.player import Player, PlayerPool
from src.core.draft import DraftState, LeagueSettings
from src.utils.data_loader import create_sample_player_pool, get_default_league_settings
from src.evaluation.backtesting import DraftBacktester, create_mock_strategies
from src.evaluation.hyperparameter_search import (
    HyperparameterOptimizer, ParameterSpace, 
    create_mcts_parameter_space, create_strategy_factory
)
from src.evaluation.metrics import DraftMetrics, SeasonPerformanceEvaluator


def create_enhanced_parameter_space() -> ParameterSpace:
    """Create an enhanced parameter space for comprehensive optimization"""
    
    space = ParameterSpace()
    
    # Core MCTS Parameters
    space.add_parameter('simulations_per_move', 'integer', (100, 500))
    space.add_parameter('exploration_constant', 'float', (0.8, 2.5))
    space.add_parameter('risk_penalty', 'float', (0.1, 0.4))
    
    # Strategy Weights (how much to consider each factor)
    space.add_parameter('vorp_weight', 'float', (0.7, 1.3))
    space.add_parameter('bye_week_weight', 'float', (0.0, 0.25))
    space.add_parameter('injury_weight', 'float', (0.0, 0.3))
    space.add_parameter('position_balance_weight', 'float', (0.0, 0.2))
    
    # Position-Specific Preferences
    space.add_parameter('early_qb_penalty', 'float', (0.0, 0.8))
    space.add_parameter('rb_scarcity_bonus', 'float', (0.0, 0.3))
    space.add_parameter('wr_depth_preference', 'float', (0.8, 1.2))
    space.add_parameter('te_streaming_threshold', 'float', (0.3, 0.8))
    
    # Risk Management
    space.add_parameter('rookie_discount', 'float', (0.0, 0.5))
    space.add_parameter('injury_history_penalty', 'float', (0.0, 0.4))
    space.add_parameter('age_penalty_threshold', 'integer', (28, 32))
    
    # Advanced Strategy Parameters
    space.add_parameter('late_round_upside_bias', 'float', (0.0, 0.5))
    space.add_parameter('handcuff_value_bonus', 'float', (0.0, 0.2))
    
    # Constraints to ensure realistic parameter combinations
    def total_weight_constraint(params):
        """Ensure strategy weights don't overwhelm base VORP"""
        total_extra_weight = (
            params['bye_week_weight'] + 
            params['injury_weight'] + 
            params['position_balance_weight']
        )
        return total_extra_weight <= 0.6
    
    def position_preference_constraint(params):
        """Ensure position preferences are reasonable"""
        return (
            params['rb_scarcity_bonus'] + params['wr_depth_preference'] <= 1.8 and
            params['early_qb_penalty'] <= params['te_streaming_threshold']
        )
    
    space.add_constraint(total_weight_constraint)
    space.add_constraint(position_preference_constraint)
    
    return space


def create_advanced_strategy_factory() -> callable:
    """Create an advanced strategy factory with more sophisticated logic"""
    
    def strategy_factory(parameters: dict):
        """Create a sophisticated parameterized strategy"""
        
        class AdvancedParameterizedMCTS:
            def __init__(self, params):
                self.params = params
                self.last_pick_reasoning = "Advanced Parameterized MCTS"
                
            def search(self, draft_state):
                available = list(draft_state.available_players)
                if not available:
                    return None
                
                current_roster = draft_state.get_our_roster()
                current_round = draft_state.current_round
                
                scores = []
                
                for player in available:
                    score = self._calculate_player_score(player, current_roster, current_round)
                    scores.append((player, score))
                
                # Return best player
                best_player, best_score = max(scores, key=lambda x: x[1])
                self.last_pick_reasoning = f"Advanced MCTS (Score: {best_score:.3f})"
                return best_player
            
            def _calculate_player_score(self, player, current_roster, current_round):
                """Calculate comprehensive player score"""
                
                # Base VORP score
                base_score = player.vorp * self.params.get('vorp_weight', 1.0)
                
                # Risk adjustments
                risk_penalty = self._calculate_risk_penalty(player)
                
                # Position-specific adjustments
                position_adjustment = self._calculate_position_adjustment(
                    player, current_roster, current_round
                )
                
                # Situational bonuses
                situational_bonus = self._calculate_situational_bonus(
                    player, current_roster, current_round
                )
                
                # Bye week considerations
                bye_week_adjustment = self._calculate_bye_week_adjustment(
                    player, current_roster
                )
                
                # Final score
                final_score = (
                    base_score + 
                    position_adjustment + 
                    situational_bonus + 
                    bye_week_adjustment - 
                    risk_penalty
                )
                
                return final_score
            
            def _calculate_risk_penalty(self, player):
                """Calculate risk-based penalties"""
                penalty = 0.0
                
                # Base risk (uncertainty)
                base_risk = player.metadata.get('risk_sigma', 0.2)
                penalty += base_risk * self.params.get('risk_penalty', 0.2)
                
                # Injury risk
                injury_risk = player.metadata.get('injury_risk_score', 0.3)
                penalty += injury_risk * self.params.get('injury_weight', 0.2)
                
                # Rookie discount
                if player.metadata.get('is_rookie', False):
                    penalty += self.params.get('rookie_discount', 0.2)
                
                # Injury history
                injury_history = player.metadata.get('historical_injuries', 0)
                if injury_history > 1:
                    penalty += injury_history * self.params.get('injury_history_penalty', 0.1)
                
                return penalty
            
            def _calculate_position_adjustment(self, player, current_roster, current_round):
                """Calculate position-specific adjustments"""
                adjustment = 0.0
                
                position_counts = {}
                for p in current_roster:
                    position_counts[p.position] = position_counts.get(p.position, 0) + 1
                
                if player.position == 'QB':
                    # Early QB penalty
                    if current_round <= 4:
                        adjustment -= self.params.get('early_qb_penalty', 0.3)
                    
                elif player.position == 'RB':
                    # RB scarcity bonus (RBs get more valuable as fewer remain)
                    rb_count = position_counts.get('RB', 0)
                    if rb_count < 3:  # Need RBs
                        adjustment += self.params.get('rb_scarcity_bonus', 0.2)
                
                elif player.position == 'WR':
                    # WR depth preference
                    wr_count = position_counts.get('WR', 0)
                    wr_preference = self.params.get('wr_depth_preference', 1.0)
                    if wr_count < 4:  # Building WR depth
                        adjustment += (wr_preference - 1.0) * player.vorp * 0.1
                
                elif player.position == 'TE':
                    # TE streaming strategy
                    te_count = position_counts.get('TE', 0)
                    streaming_threshold = self.params.get('te_streaming_threshold', 0.5)
                    if te_count == 0 and player.vorp < streaming_threshold:
                        adjustment -= 0.2  # Prefer to wait on TE
                
                return adjustment
            
            def _calculate_situational_bonus(self, player, current_roster, current_round):
                """Calculate situational bonuses"""
                bonus = 0.0
                
                # Late round upside bias
                if current_round >= 10:
                    upside_bias = self.params.get('late_round_upside_bias', 0.2)
                    player_upside = player.metadata.get('risk_sigma', 0.2)
                    bonus += upside_bias * player_upside
                
                # Handcuff value (simplified)
                if player.position in ['RB'] and current_round >= 8:
                    handcuff_bonus = self.params.get('handcuff_value_bonus', 0.1)
                    # Check if we have RBs from same team (simplified)
                    our_teams = {p.team for p in current_roster if p.position == 'RB'}
                    if player.team in our_teams:
                        bonus += handcuff_bonus
                
                return bonus
            
            def _calculate_bye_week_adjustment(self, player, current_roster):
                """Calculate bye week impact"""
                bye_week_weight = self.params.get('bye_week_weight', 0.1)
                
                if bye_week_weight == 0:
                    return 0.0
                
                player_bye = getattr(player, 'bye_week', 0)
                if player_bye == 0:
                    return 0.0
                
                # Count current bye week conflicts
                roster_byes = [getattr(p, 'bye_week', 0) for p in current_roster]
                conflicts = roster_byes.count(player_bye)
                
                # Penalty for conflicts, bonus for strategic clustering
                if conflicts == 0:
                    return bye_week_weight * 0.1  # Small bonus for new bye week
                elif conflicts <= 2:
                    return bye_week_weight * 0.05  # Small bonus for clustering
                else:
                    return -bye_week_weight * 0.2  # Penalty for over-clustering
        
        return AdvancedParameterizedMCTS(parameters)
    
    return strategy_factory


def run_comprehensive_optimization_demo():
    """Run the complete hyperparameter optimization demonstration"""
    
    print("🔧 Comprehensive Hyperparameter Optimization Demo")
    print("=" * 60)
    
    # 1. Setup
    print("\n📊 Step 1: Setting up test environment...")
    player_pool = create_sample_player_pool(n_players=300)
    league_settings = get_default_league_settings()
    
    # Add realistic bye weeks and injury data
    for i, player in enumerate(player_pool):
        player.metadata['bye_week'] = 4 + (i % 11)  # Weeks 4-14
        player.metadata['injury_risk_score'] = np.random.beta(2, 5)  # Skewed toward lower risk
        player.metadata['is_rookie'] = np.random.random() < 0.15
        player.metadata['historical_injuries'] = np.random.poisson(0.8)
        if player.position == 'QB':
            player.metadata['injury_risk_score'] *= 0.7  # QBs typically less injury prone
    
    print(f"   ✅ Created {len(player_pool)} players with enhanced data")
    
    # 2. Setup backtester
    print("\n🔬 Step 2: Initializing backtesting system...")
    backtester = DraftBacktester(player_pool, league_settings)
    backtester.n_simulations = 8  # Reduced for demo speed
    backtester.parallel_workers = 2
    
    # 3. Create parameter space
    print("\n🎯 Step 3: Creating parameter space...")
    parameter_space = create_enhanced_parameter_space()
    strategy_factory = create_advanced_strategy_factory()
    
    print(f"   📋 Parameters to optimize: {list(parameter_space.parameters.keys())}")
    
    # 4. Initialize optimizer
    optimizer = HyperparameterOptimizer(backtester, strategy_factory, parameter_space)
    
    # 5. Run different optimization methods
    optimization_results = {}
    
    # Random Search
    print("\n🎲 Step 4a: Random Search Optimization...")
    start_time = time.time()
    random_results = optimizer.random_search(
        n_configurations=20, 
        n_trials_per_config=3,
        parallel=True
    )
    random_time = time.time() - start_time
    
    optimization_results['Random Search'] = {
        'results': random_results,
        'time': random_time,
        'best_score': optimizer.best_score
    }
    
    print(f"   ⏱️  Completed in {random_time:.1f} seconds")
    
    # Grid Search (reduced size for demo)
    print("\n🔍 Step 4b: Limited Grid Search...")
    
    # Create smaller parameter space for grid search
    grid_space = ParameterSpace()
    grid_space.add_parameter('risk_penalty', 'float', (0.1, 0.3))
    grid_space.add_parameter('vorp_weight', 'float', (0.8, 1.2))
    grid_space.add_parameter('early_qb_penalty', 'float', (0.0, 0.5))
    grid_space.add_parameter('rb_scarcity_bonus', 'float', (0.0, 0.2))
    
    grid_optimizer = HyperparameterOptimizer(backtester, strategy_factory, grid_space)
    
    start_time = time.time()
    grid_results = grid_optimizer.grid_search(
        n_trials_per_config=3,
        parallel=True
    )
    grid_time = time.time() - start_time
    
    optimization_results['Grid Search'] = {
        'results': grid_results,
        'time': grid_time,
        'best_score': grid_optimizer.best_score
    }
    
    print(f"   ⏱️  Completed in {grid_time:.1f} seconds")
    
    # Bayesian Optimization
    print("\n🧠 Step 4c: Bayesian Optimization...")
    start_time = time.time()
    
    # Use numerical parameters only for Bayesian optimization
    bayesian_space = ParameterSpace()
    bayesian_space.add_parameter('risk_penalty', 'float', (0.1, 0.4))
    bayesian_space.add_parameter('vorp_weight', 'float', (0.7, 1.3))
    bayesian_space.add_parameter('early_qb_penalty', 'float', (0.0, 0.8))
    bayesian_space.add_parameter('rb_scarcity_bonus', 'float', (0.0, 0.3))
    bayesian_space.add_parameter('bye_week_weight', 'float', (0.0, 0.25))
    
    bayesian_optimizer = HyperparameterOptimizer(backtester, strategy_factory, bayesian_space)
    
    bayesian_results = bayesian_optimizer.bayesian_optimization(
        n_initial=8,
        n_iterations=12,
        n_trials_per_config=3
    )
    bayesian_time = time.time() - start_time
    
    optimization_results['Bayesian Optimization'] = {
        'results': bayesian_results,
        'time': bayesian_time,
        'best_score': bayesian_optimizer.best_score
    }
    
    print(f"   ⏱️  Completed in {bayesian_time:.1f} seconds")
    
    # 6. Compare optimization methods
    print("\n📊 Step 5: Comparing optimization methods...")
    
    comparison_data = []
    for method, data in optimization_results.items():
        comparison_data.append({
            'Method': method,
            'Best Score': data['best_score'],
            'Time (seconds)': data['time'],
            'Configurations Tested': len(data['results']),
            'Score per Second': data['best_score'] / data['time'],
            'Score per Config': data['best_score'] / len(data['results'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n🏆 Optimization Method Comparison:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # 7. Detailed analysis of best configuration
    print("\n🔍 Step 6: Analyzing best configuration...")
    
    best_method = comparison_df.loc[comparison_df['Best Score'].idxmax(), 'Method']
    best_optimizer = {
        'Random Search': optimizer,
        'Grid Search': grid_optimizer,
        'Bayesian Optimization': bayesian_optimizer
    }[best_method]
    
    print(f"\n🥇 Best Method: {best_method}")
    print(f"🎯 Best Score: {best_optimizer.best_score:.4f}")
    print(f"⚙️  Best Parameters:")
    
    for param, value in best_optimizer.best_parameters.items():
        if isinstance(value, float):
            print(f"   {param}: {value:.4f}")
        else:
            print(f"   {param}: {value}")
    
    # 8. Validate best configuration with extended testing
    print("\n🧪 Step 7: Validating best configuration...")
    
    best_strategy = strategy_factory(best_optimizer.best_parameters)
    
    # Test against multiple scenarios
    validation_scenarios = [
        {'name': 'early_pick', 'draft_position': 2, 'league_type': 'standard',
         'injury_rate': 0.15, 'rookie_uncertainty': 1.0, 'opponent_skill': 'average'},
        {'name': 'middle_pick', 'draft_position': 6, 'league_type': 'standard',
         'injury_rate': 0.15, 'rookie_uncertainty': 1.0, 'opponent_skill': 'average'},
        {'name': 'late_pick', 'draft_position': 11, 'league_type': 'standard',
         'injury_rate': 0.15, 'rookie_uncertainty': 1.0, 'opponent_skill': 'average'},
        {'name': 'high_injury', 'draft_position': 6, 'league_type': 'standard',
         'injury_rate': 0.25, 'rookie_uncertainty': 1.0, 'opponent_skill': 'average'},
    ]
    
    backtester.n_simulations = 15  # More trials for validation
    validation_results = backtester.backtest_strategy(
        best_strategy, 
        "Optimized Strategy",
        validation_scenarios,
        parallel=True
    )
    
    print(f"✅ Validation Results:")
    for scenario, results in validation_results.items():
        avg_score = np.mean([r.composite_score for r in results])
        avg_vorp = np.mean([r.total_vorp for r in results])
        print(f"   {scenario:15s}: Score {avg_score:.4f}, VORP {avg_vorp:.2f}")
    
    # 9. Create comprehensive reports
    print("\n📋 Step 8: Creating optimization reports...")
    
    for method, data in optimization_results.items():
        optimizer_obj = {
            'Random Search': optimizer,
            'Grid Search': grid_optimizer, 
            'Bayesian Optimization': bayesian_optimizer
        }[method]
        
        report = optimizer_obj.create_optimization_report(data['results'], method.lower().replace(' ', '_'))
    
    # 10. Season simulation with optimized strategy
    print("\n🏈 Step 9: Season simulation with optimized strategy...")
    
    # Create a sample roster using the optimized strategy
    draft_state = DraftState.create_mock_draft(player_pool, our_team_id=6)
    optimized_roster = []
    
    # Simulate first 8 rounds of draft
    for round_num in range(8):
        if draft_state.available_players:
            best_pick = best_strategy.search(draft_state)
            if best_pick:
                optimized_roster.append(best_pick)
                draft_state.make_pick(best_pick)
            
            # Simulate opponent picks
            while not draft_state.is_our_turn and draft_state.available_players:
                available = list(draft_state.available_players)
                opponent_pick = np.random.choice(available)
                draft_state.make_pick(opponent_pick)
    
    # Run season simulation
    evaluator = SeasonPerformanceEvaluator(league_settings.roster_spots)
    season_metrics = evaluator.simulate_season(optimized_roster)
    
    print(f"   🏆 Season Results:")
    print(f"      Total Points: {season_metrics.total_points:.1f}")
    print(f"      Avg Per Week: {season_metrics.avg_points_per_week:.1f}")
    print(f"      Consistency: {season_metrics.consistency_score:.3f}")
    print(f"      Draft Efficiency: {season_metrics.draft_efficiency:.3f}")
    
    # Create season visualization
    evaluator.visualize_season_performance(season_metrics, "optimized_strategy_season.png")
    
    print("\n✅ Comprehensive hyperparameter optimization demo complete!")
    print("\n📊 Key Insights:")
    print(f"   🥇 Best optimization method: {best_method}")
    print(f"   ⚡ Best score achieved: {best_optimizer.best_score:.4f}")
    print(f"   🎯 Most important parameters discovered:")
    
    # Show top parameters by impact
    sorted_params = sorted(
        best_optimizer.best_parameters.items(),
        key=lambda x: abs(x[1] - 0.5) if isinstance(x[1], (int, float)) else 0,
        reverse=True
    )
    
    for param, value in sorted_params[:5]:
        if isinstance(value, float):
            print(f"      • {param}: {value:.4f}")
        else:
            print(f"      • {param}: {value}")
    
    print(f"\n🚀 Your optimized strategy is ready for draft domination!")
    
    return {
        'optimization_results': optimization_results,
        'best_parameters': best_optimizer.best_parameters,
        'validation_results': validation_results,
        'season_metrics': season_metrics
    }


if __name__ == "__main__":
    # Run the comprehensive demo
    demo_results = run_comprehensive_optimization_demo()
    
    print(f"\n💾 All results and visualizations have been saved to your current directory!")
    print(f"📈 Check the generated PNG files for detailed analysis charts.")
    print(f"📋 Review the TXT files for comprehensive optimization reports.")
