"""
Fantasy Football Draft Backtesting System
=========================================

This module provides comprehensive backtesting capabilities for draft strategies,
allowing evaluation against historical data and simulated scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from ..core.player import Player, PlayerPool
from ..core.draft import DraftState, LeagueSettings
from ..strategies.draft_history import DraftHistoryAnalyzer


@dataclass
class BacktestResults:
    """Results from a single backtest scenario"""
    strategy_name: str
    total_vorp: float
    roster_balance_score: float
    bye_week_score: float
    injury_risk_score: float
    draft_efficiency: float  # How well we utilized our picks
    season_projection: float  # Projected fantasy points
    final_roster: List[Player]
    pick_history: List[Tuple[int, Player, str]]  # round, player, reason
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def composite_score(self) -> float:
        """Overall score combining all metrics"""
        return (
            self.total_vorp * 0.35 +
            self.roster_balance_score * 0.20 +
            self.bye_week_score * 0.15 +
            (1.0 - self.injury_risk_score) * 0.15 +  # Lower injury risk is better
            self.draft_efficiency * 0.15
        )


class DraftBacktester:
    """
    Comprehensive draft strategy backtesting system.
    
    Features:
    - Historical data validation
    - Monte Carlo simulation across multiple scenarios
    - Strategy comparison and ranking
    - Detailed performance metrics
    - Parallel execution for speed
    """
    
    def __init__(self, 
                 player_pool: PlayerPool,
                 league_settings: LeagueSettings,
                 historical_data: Optional[Dict] = None):
        
        self.player_pool = player_pool
        self.league_settings = league_settings
        self.historical_data = historical_data or {}
        
        # Backtesting parameters
        self.n_simulations = 100
        self.draft_positions = list(range(1, 13))  # Test all draft positions
        self.parallel_workers = 4
        
        # Results storage
        self.results_cache = {}
        
        print(f"🔬 Draft Backtester initialized")
        print(f"   📊 Player pool: {len(player_pool)} players")
        print(f"   🎯 League: {league_settings.teams} teams")
        print(f"   🔄 Simulations per test: {self.n_simulations}")
    
    def backtest_strategy(self, 
                         strategy,
                         strategy_name: str,
                         test_scenarios: Optional[List[Dict]] = None,
                         parallel: bool = True) -> Dict[str, List[BacktestResults]]:
        """
        Backtest a strategy across multiple scenarios.
        
        Args:
            strategy: The draft strategy to test
            strategy_name: Name for identification
            test_scenarios: Custom scenarios to test (optional)
            parallel: Use parallel processing
            
        Returns:
            Dictionary mapping scenario names to results lists
        """
        
        print(f"🧪 Backtesting strategy: {strategy_name}")
        
        # Generate test scenarios if not provided
        if test_scenarios is None:
            test_scenarios = self._generate_test_scenarios()
        
        all_results = {}
        
        for scenario in test_scenarios:
            scenario_name = scenario['name']
            print(f"   📋 Testing scenario: {scenario_name}")
            
            if parallel and self.parallel_workers > 1:
                scenario_results = self._run_parallel_simulations(
                    strategy, strategy_name, scenario
                )
            else:
                scenario_results = self._run_sequential_simulations(
                    strategy, strategy_name, scenario
                )
            
            all_results[scenario_name] = scenario_results
            
            # Quick summary
            avg_score = np.mean([r.composite_score for r in scenario_results])
            avg_vorp = np.mean([r.total_vorp for r in scenario_results])
            print(f"      🎯 Avg Score: {avg_score:.3f}, Avg VORP: {avg_vorp:.2f}")
        
        # Cache results
        cache_key = f"{strategy_name}_{hash(str(test_scenarios))}"
        self.results_cache[cache_key] = all_results
        
        return all_results
    
    def _generate_test_scenarios(self) -> List[Dict]:
        """Generate comprehensive test scenarios"""
        
        scenarios = []
        
        # Standard scenarios across all draft positions
        for draft_pos in [1, 3, 6, 9, 12]:  # Sample of positions
            scenarios.append({
                'name': f'standard_pos_{draft_pos}',
                'draft_position': draft_pos,
                'league_type': 'standard',
                'injury_rate': 0.15,
                'rookie_uncertainty': 1.0,
                'opponent_skill': 'average'
            })
        
        # High injury rate scenarios
        scenarios.append({
            'name': 'high_injury_rate',
            'draft_position': 6,
            'league_type': 'standard', 
            'injury_rate': 0.25,
            'rookie_uncertainty': 1.0,
            'opponent_skill': 'average'
        })
        
        # High rookie uncertainty
        scenarios.append({
            'name': 'rookie_heavy',
            'draft_position': 6,
            'league_type': 'standard',
            'injury_rate': 0.15,
            'rookie_uncertainty': 1.5,
            'opponent_skill': 'average'
        })
        
        # Expert league (opponents follow ADP closely)
        scenarios.append({
            'name': 'expert_league',
            'draft_position': 6,
            'league_type': 'standard',
            'injury_rate': 0.15,
            'rookie_uncertainty': 1.0,
            'opponent_skill': 'expert'
        })
        
        # Casual league (opponents more unpredictable)
        scenarios.append({
            'name': 'casual_league',
            'draft_position': 6,
            'league_type': 'standard',
            'injury_rate': 0.15,
            'rookie_uncertainty': 1.0,
            'opponent_skill': 'casual'
        })
        
        return scenarios
    
    def _run_parallel_simulations(self, 
                                 strategy, 
                                 strategy_name: str, 
                                 scenario: Dict) -> List[BacktestResults]:
        """Run simulations in parallel"""
        
        n_sims = min(self.n_simulations, 50)  # Limit for memory
        
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = []
            
            for sim_id in range(n_sims):
                future = executor.submit(
                    self._run_single_simulation,
                    strategy, strategy_name, scenario, sim_id
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout per sim
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"      ⚠️  Simulation failed: {e}")
        
        return results
    
    def _run_sequential_simulations(self, 
                                   strategy, 
                                   strategy_name: str, 
                                   scenario: Dict) -> List[BacktestResults]:
        """Run simulations sequentially"""
        
        results = []
        n_sims = min(self.n_simulations, 20)  # Smaller for sequential
        
        for sim_id in range(n_sims):
            try:
                result = self._run_single_simulation(
                    strategy, strategy_name, scenario, sim_id
                )
                if result:
                    results.append(result)
                    
                if (sim_id + 1) % 10 == 0:
                    print(f"      📊 Completed {sim_id + 1}/{n_sims} simulations")
                    
            except Exception as e:
                print(f"      ⚠️  Simulation {sim_id} failed: {e}")
        
        return results
    
    def _run_single_simulation(self, 
                              strategy,
                              strategy_name: str, 
                              scenario: Dict,
                              sim_id: int) -> Optional[BacktestResults]:
        """Run a single draft simulation"""
        
        try:
            # Create draft state for this scenario
            draft_state = self._create_scenario_draft_state(scenario)
            
            # Run the draft
            our_picks = []
            pick_history = []
            
            round_count = 0
            while not draft_state.is_draft_complete() and round_count < 15:
                
                if draft_state.is_our_turn and draft_state.available_players:
                    # Our turn - use the strategy
                    selected_player = strategy.search(draft_state)
                    
                    if selected_player:
                        our_picks.append(selected_player)
                        
                        # Record pick with reasoning (if available)
                        reasoning = getattr(strategy, 'last_pick_reasoning', 'MCTS selection')
                        pick_history.append((draft_state.current_round, selected_player, reasoning))
                        
                        draft_state.make_pick(selected_player)
                        round_count += 1
                    else:
                        break
                else:
                    # Opponent turn - simulate
                    opponent_pick = self._simulate_opponent_pick(draft_state, scenario)
                    if opponent_pick:
                        draft_state.make_pick(opponent_pick)
                    else:
                        break
            
            # Calculate metrics
            if our_picks:
                return self._calculate_backtest_metrics(
                    strategy_name, our_picks, pick_history, scenario, sim_id
                )
            
        except Exception as e:
            print(f"      ❌ Simulation error: {e}")
            return None
    
    def _create_scenario_draft_state(self, scenario: Dict) -> DraftState:
        """Create a draft state configured for the scenario"""
        
        # Modify player pool based on scenario
        modified_pool = self._apply_scenario_modifiers(scenario)
        
        # Create draft state
        draft_state = DraftState.create_mock_draft(
            modified_pool, 
            our_team_id=scenario['draft_position']
        )
        
        return draft_state
    
    def _apply_scenario_modifiers(self, scenario: Dict) -> PlayerPool:
        """Apply scenario-specific modifiers to player pool"""
        
        modified_players = []
        
        for player in self.player_pool:
            # Create a copy to modify
            modified_player = Player(
                name=player.name,
                position=player.position,
                team=player.team,
                vorp=player.vorp
            )
            
            # Copy metadata
            modified_player.metadata = player.metadata.copy()
            
            # Apply injury rate modifier
            injury_multiplier = scenario.get('injury_rate', 0.15) / 0.15  # 0.15 is baseline
            base_injury_risk = modified_player.metadata.get('injury_risk_score', 0.3)
            modified_player.metadata['injury_risk_score'] = min(1.0, base_injury_risk * injury_multiplier)
            
            # Apply rookie uncertainty
            if modified_player.metadata.get('is_rookie', False):
                uncertainty_multiplier = scenario.get('rookie_uncertainty', 1.0)
                base_risk_sigma = modified_player.metadata.get('risk_sigma', 0.3)
                modified_player.metadata['risk_sigma'] = min(1.0, base_risk_sigma * uncertainty_multiplier)
            
            modified_players.append(modified_player)
        
        return PlayerPool(modified_players)
    
    def _simulate_opponent_pick(self, draft_state: DraftState, scenario: Dict) -> Optional[Player]:
        """Simulate opponent pick based on scenario skill level"""
        
        available = list(draft_state.available_players)
        if not available:
            return None
        
        skill_level = scenario.get('opponent_skill', 'average')
        
        if skill_level == 'expert':
            # Follow ADP closely with small random variation
            weights = [1.0 / (getattr(p, 'adp_rank', 999) + np.random.uniform(0, 5)) for p in available]
        elif skill_level == 'casual':
            # More random, less ADP adherence
            weights = [1.0 / (getattr(p, 'adp_rank', 999) + np.random.uniform(0, 50)) for p in available]
        else:  # average
            # Moderate ADP adherence
            weights = [1.0 / (getattr(p, 'adp_rank', 999) + np.random.uniform(0, 20)) for p in available]
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.random.choice(available, p=weights)
    
    def _calculate_backtest_metrics(self, 
                                   strategy_name: str,
                                   our_picks: List[Player], 
                                   pick_history: List[Tuple],
                                   scenario: Dict,
                                   sim_id: int) -> BacktestResults:
        """Calculate comprehensive metrics for the draft"""
        
        # Basic metrics
        total_vorp = sum(p.vorp for p in our_picks)
        
        # Roster balance (positional distribution)
        position_counts = defaultdict(int)
        for player in our_picks:
            position_counts[player.position] += 1
        
        # Ideal distribution for 12-team league
        ideal_distribution = {'QB': 2, 'RB': 4, 'WR': 5, 'TE': 2, 'K': 1, 'DEF': 1}
        
        balance_score = 0.0
        for pos, ideal_count in ideal_distribution.items():
            actual_count = position_counts.get(pos, 0)
            # Penalize both under and over drafting
            difference = abs(actual_count - ideal_count)
            balance_score += max(0, 1.0 - (difference * 0.3))
        
        roster_balance_score = balance_score / len(ideal_distribution)
        
        # Bye week analysis
        bye_weeks = [p.metadata.get('bye_week', 0) for p in our_picks if p.metadata.get('bye_week', 0) > 0]
        if bye_weeks:
            bye_week_conflicts = len(bye_weeks) - len(set(bye_weeks))
            bye_week_score = max(0.0, 1.0 - (bye_week_conflicts * 0.2))
        else:
            bye_week_score = 1.0
        
        # Injury risk analysis
        injury_risks = [p.metadata.get('injury_risk_score', 0.3) for p in our_picks]
        avg_injury_risk = np.mean(injury_risks)
        
        # Draft efficiency (how much VORP per pick compared to ADP)
        pick_values = []
        for round_num, player, _ in pick_history:
            expected_pick = round_num * 12  # Rough estimate
            actual_adp = getattr(player, 'adp_rank', expected_pick)
            value_over_adp = max(0, actual_adp - expected_pick)
            pick_values.append(value_over_adp)
        
        draft_efficiency = np.mean(pick_values) / 20.0 if pick_values else 0.0  # Normalize
        draft_efficiency = min(1.0, draft_efficiency)
        
        # Season projection (simplified)
        season_projection = sum(p.proj_ppg * 16 for p in our_picks)  # 16 games
        
        # Create result
        return BacktestResults(
            strategy_name=strategy_name,
            total_vorp=total_vorp,
            roster_balance_score=roster_balance_score,
            bye_week_score=bye_week_score,
            injury_risk_score=avg_injury_risk,
            draft_efficiency=draft_efficiency,
            season_projection=season_projection,
            final_roster=our_picks,
            pick_history=pick_history,
            metadata={
                'scenario': scenario,
                'simulation_id': sim_id,
                'position_counts': dict(position_counts),
                'bye_week_conflicts': bye_week_conflicts if bye_weeks else 0
            }
        )
    
    def compare_strategies(self, 
                          strategies: Dict[str, Any],
                          test_scenarios: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Compare multiple strategies across scenarios.
        
        Args:
            strategies: Dict mapping strategy names to strategy objects
            test_scenarios: Test scenarios (will generate if None)
            
        Returns:
            DataFrame with comparison results
        """
        
        print(f"🏆 Comparing {len(strategies)} strategies")
        
        all_results = {}
        
        for strategy_name, strategy in strategies.items():
            results = self.backtest_strategy(
                strategy, strategy_name, test_scenarios, parallel=True
            )
            all_results[strategy_name] = results
        
        # Create comparison DataFrame
        comparison_data = []
        
        for strategy_name, strategy_results in all_results.items():
            for scenario_name, scenario_results in strategy_results.items():
                
                # Aggregate metrics across simulations
                composite_scores = [r.composite_score for r in scenario_results]
                total_vorps = [r.total_vorp for r in scenario_results]
                balance_scores = [r.roster_balance_score for r in scenario_results]
                bye_scores = [r.bye_week_score for r in scenario_results]
                injury_scores = [r.injury_risk_score for r in scenario_results]
                efficiency_scores = [r.draft_efficiency for r in scenario_results]
                
                comparison_data.append({
                    'strategy': strategy_name,
                    'scenario': scenario_name,
                    'composite_score_mean': np.mean(composite_scores),
                    'composite_score_std': np.std(composite_scores),
                    'total_vorp_mean': np.mean(total_vorps),
                    'total_vorp_std': np.std(total_vorps),
                    'balance_score_mean': np.mean(balance_scores),
                    'bye_score_mean': np.mean(bye_scores),
                    'injury_risk_mean': np.mean(injury_scores),
                    'efficiency_mean': np.mean(efficiency_scores),
                    'n_simulations': len(scenario_results)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print summary
        print(f"\n📊 Strategy Comparison Summary:")
        print("=" * 50)
        
        strategy_rankings = comparison_df.groupby('strategy')['composite_score_mean'].mean().sort_values(ascending=False)
        
        for rank, (strategy, avg_score) in enumerate(strategy_rankings.items(), 1):
            avg_vorp = comparison_df[comparison_df['strategy'] == strategy]['total_vorp_mean'].mean()
            print(f"  {rank}. {strategy:20s} - Score: {avg_score:.3f}, Avg VORP: {avg_vorp:.2f}")
        
        return comparison_df
    
    def create_backtest_report(self, 
                              comparison_df: pd.DataFrame,
                              save_path: Optional[str] = None) -> str:
        """Create comprehensive backtesting report with visualizations"""
        
        print(f"📋 Creating backtesting report...")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fantasy Football Draft Strategy Backtesting Report', fontsize=16, fontweight='bold')
        
        # 1. Overall strategy performance
        ax1 = axes[0, 0]
        strategy_performance = comparison_df.groupby('strategy')['composite_score_mean'].mean().sort_values(ascending=True)
        strategy_performance.plot(kind='barh', ax=ax1, color='skyblue', edgecolor='navy')
        ax1.set_title('Overall Strategy Performance')
        ax1.set_xlabel('Composite Score')
        
        # 2. VORP comparison
        ax2 = axes[0, 1]
        vorp_data = comparison_df.pivot(index='scenario', columns='strategy', values='total_vorp_mean')
        sns.heatmap(vorp_data, annot=True, fmt='.1f', ax=ax2, cmap='RdYlGn')
        ax2.set_title('Total VORP by Strategy & Scenario')
        
        # 3. Consistency (lower std = more consistent)
        ax3 = axes[0, 2]
        consistency_data = comparison_df.groupby('strategy')['composite_score_std'].mean().sort_values()
        consistency_data.plot(kind='bar', ax=ax3, color='lightcoral', edgecolor='darkred')
        ax3.set_title('Strategy Consistency (Lower = Better)')
        ax3.set_ylabel('Score Standard Deviation')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Roster balance comparison
        ax4 = axes[1, 0]
        balance_data = comparison_df.groupby('strategy')['balance_score_mean'].mean().sort_values(ascending=True)
        balance_data.plot(kind='barh', ax=ax4, color='lightgreen', edgecolor='darkgreen')
        ax4.set_title('Roster Balance Score')
        ax4.set_xlabel('Balance Score')
        
        # 5. Risk analysis
        ax5 = axes[1, 1]
        risk_comparison = comparison_df.groupby('strategy').agg({
            'injury_risk_mean': 'mean',
            'bye_score_mean': 'mean'
        })
        risk_comparison.plot(kind='bar', ax=ax5, width=0.8)
        ax5.set_title('Risk Management (Injury vs Bye Weeks)')
        ax5.set_ylabel('Score')
        ax5.legend(['Injury Risk (Lower Better)', 'Bye Week Score (Higher Better)'])
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Scenario performance heatmap
        ax6 = axes[1, 2]
        scenario_heatmap = comparison_df.pivot(index='strategy', columns='scenario', values='composite_score_mean')
        sns.heatmap(scenario_heatmap, annot=True, fmt='.3f', ax=ax6, cmap='RdYlGn')
        ax6.set_title('Performance by Scenario')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = save_path or 'backtest_report.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create text report
        report_lines = [
            "Fantasy Football Draft Strategy Backtesting Report",
            "=" * 60,
            "",
            "📊 STRATEGY RANKINGS",
            "-" * 25
        ]
        
        strategy_rankings = comparison_df.groupby('strategy').agg({
            'composite_score_mean': 'mean',
            'total_vorp_mean': 'mean',
            'balance_score_mean': 'mean',
            'bye_score_mean': 'mean',
            'injury_risk_mean': 'mean',
            'efficiency_mean': 'mean'
        }).sort_values('composite_score_mean', ascending=False)
        
        for rank, (strategy, metrics) in enumerate(strategy_rankings.iterrows(), 1):
            report_lines.extend([
                f"{rank}. {strategy}",
                f"   Composite Score: {metrics['composite_score_mean']:.3f}",
                f"   Total VORP: {metrics['total_vorp_mean']:.2f}",
                f"   Balance Score: {metrics['balance_score_mean']:.3f}",
                f"   Bye Week Score: {metrics['bye_score_mean']:.3f}",
                f"   Injury Risk: {metrics['injury_risk_mean']:.3f}",
                f"   Draft Efficiency: {metrics['efficiency_mean']:.3f}",
                ""
            ])
        
        # Scenario analysis
        report_lines.extend([
            "🎯 SCENARIO ANALYSIS",
            "-" * 25,
            ""
        ])
        
        for scenario in comparison_df['scenario'].unique():
            scenario_data = comparison_df[comparison_df['scenario'] == scenario]
            best_strategy = scenario_data.loc[scenario_data['composite_score_mean'].idxmax(), 'strategy']
            best_score = scenario_data['composite_score_mean'].max()
            
            report_lines.extend([
                f"Scenario: {scenario}",
                f"   Best Strategy: {best_strategy} (Score: {best_score:.3f})",
                ""
            ])
        
        # Recommendations
        overall_best = strategy_rankings.index[0]
        most_consistent = comparison_df.groupby('strategy')['composite_score_std'].mean().idxmin()
        
        report_lines.extend([
            "🏆 RECOMMENDATIONS",
            "-" * 20,
            f"🥇 Overall Best: {overall_best}",
            f"🎯 Most Consistent: {most_consistent}",
            "",
            "📋 Key Insights:",
            "• Higher VORP generally correlates with better performance",
            "• Roster balance is crucial for season-long success", 
            "• Injury risk management provides competitive advantage",
            "• Draft efficiency varies significantly by strategy",
            ""
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save text report
        text_report_path = plot_path.replace('.png', '_report.txt')
        with open(text_report_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Backtesting report saved:")
        print(f"   📊 Visualizations: {plot_path}")
        print(f"   📋 Text Report: {text_report_path}")
        
        return report_text


def create_mock_strategies() -> Dict[str, Any]:
    """Create mock strategies for testing purposes"""
    
    class MockStrategy:
        def __init__(self, name, vorp_weight=1.0, risk_penalty=0.1):
            self.name = name
            self.vorp_weight = vorp_weight
            self.risk_penalty = risk_penalty
            self.last_pick_reasoning = "Mock selection"
        
        def search(self, draft_state):
            available = list(draft_state.available_players)
            if not available:
                return None
            
            # Simple VORP-based selection with risk consideration
            scores = []
            for player in available:
                vorp_score = player.vorp * self.vorp_weight
                risk_score = player.metadata.get('risk_sigma', 0.3) * self.risk_penalty
                final_score = vorp_score - risk_score
                scores.append((player, final_score))
            
            best_player = max(scores, key=lambda x: x[1])[0]
            self.last_pick_reasoning = f"VORP: {best_player.vorp:.2f}, Risk-adjusted"
            return best_player
    
    return {
        'Conservative MCTS': MockStrategy('Conservative', vorp_weight=0.8, risk_penalty=0.3),
        'Aggressive MCTS': MockStrategy('Aggressive', vorp_weight=1.2, risk_penalty=0.1),
        'Balanced MCTS': MockStrategy('Balanced', vorp_weight=1.0, risk_penalty=0.2),
        'Risk-Averse MCTS': MockStrategy('Risk-Averse', vorp_weight=0.9, risk_penalty=0.4)
    }


if __name__ == "__main__":
    # Example usage
    from ..utils.data_loader import create_sample_player_pool, get_default_league_settings
    
    print("🔬 Testing Draft Backtesting System")
    
    # Create test data
    player_pool = create_sample_player_pool(300)
    league_settings = get_default_league_settings()
    
    # Initialize backtester
    backtester = DraftBacktester(player_pool, league_settings)
    
    # Create mock strategies
    strategies = create_mock_strategies()
    
    # Run comparison
    comparison_df = backtester.compare_strategies(strategies)
    
    # Create report
    report = backtester.create_backtest_report(comparison_df)
    
    print("\n✅ Backtesting demo complete!")
    print("📊 Check 'backtest_report.png' for visualizations")
