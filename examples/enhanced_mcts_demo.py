#!/usr/bin/env python3
"""
Enhanced MCTS Demo with Bye Week and Draft History Analysis
===========================================================

This example demonstrates the advanced MCTS system that considers:
1. Draft history patterns and learning
2. Sophisticated bye week management 
3. Strategic bye week clustering vs spreading

Usage:
    python examples/enhanced_mcts_demo.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.player import Player, PlayerPool
from src.core.draft import DraftState, LeagueSettings
from src.utils.data_loader import create_sample_player_pool, get_default_league_settings
from src.strategies.draft_history import DraftHistoryAnalyzer, HistoryAwareMCTS, simulate_and_learn
from src.strategies.bye_week import ByeWeekOptimizer, ByeWeekAwareMCTS, analyze_league_bye_week_landscape, create_bye_week_strategy_report


class MockMCTS:
    """Mock MCTS for demonstration purposes"""
    def __init__(self, player_pool):
        self.player_pool = player_pool
    
    def search(self, draft_state):
        # Simple VORP-based selection
        available = list(draft_state.available_players)
        if available:
            return max(available, key=lambda p: p.vorp)
        return None


def enhance_sample_data_with_bye_weeks(player_pool: PlayerPool) -> PlayerPool:
    """Add realistic bye weeks to sample data"""
    
    print("📅 Adding realistic bye weeks to player data...")
    
    # NFL bye weeks typically range from 4-14
    bye_weeks = list(range(4, 15))  # Weeks 4-14
    
    # Distribute players across bye weeks (some weeks have more teams)
    # Weeks 6, 7, 9, 10, 11 typically have more teams on bye
    bye_week_weights = {
        4: 2, 5: 2, 6: 4, 7: 4, 8: 2, 9: 4, 10: 4, 11: 4, 12: 2, 13: 2, 14: 2
    }
    
    # Create weighted list
    weighted_bye_weeks = []
    for week, weight in bye_week_weights.items():
        weighted_bye_weeks.extend([week] * weight)
    
    # Assign bye weeks to players
    for player in player_pool:
        bye_week = np.random.choice(weighted_bye_weeks)
        player.metadata['bye_week'] = bye_week
    
    print(f"✅ Assigned bye weeks to {len(player_pool)} players")
    
    # Show distribution
    bye_distribution = {}
    for player in player_pool:
        bye_week = player.metadata.get('bye_week', 0)
        bye_distribution[bye_week] = bye_distribution.get(bye_week, 0) + 1
    
    print("📊 Bye week distribution:")
    for week in sorted(bye_distribution.keys()):
        if week > 0:
            print(f"   Week {week:2d}: {bye_distribution[week]:2d} players")
    
    return player_pool


def main():
    """Run enhanced MCTS demonstration"""
    
    print("🚀 Enhanced MCTS Demo: Bye Weeks + Draft History")
    print("=" * 60)
    
    # 1. Create sample data with bye weeks
    print("\n📊 Step 1: Creating sample data...")
    player_pool = create_sample_player_pool(n_players=250)
    player_pool = enhance_sample_data_with_bye_weeks(player_pool)
    league = get_default_league_settings()
    
    print(f"✅ Created {len(player_pool)} players with bye weeks")
    
    # 2. Initialize draft history analyzer
    print("\n📚 Step 2: Initializing draft history analysis...")
    history_analyzer = DraftHistoryAnalyzer()
    
    # Simulate some historical drafts to learn patterns
    base_mcts = MockMCTS(player_pool)
    simulate_and_learn(base_mcts, player_pool, league, history_analyzer, n_simulations=5)
    
    # 3. Analyze bye week landscape
    print("\n📅 Step 3: Analyzing bye week landscape...")
    bye_week_optimizer = analyze_league_bye_week_landscape(player_pool, league)
    
    # 4. Create enhanced MCTS strategies
    print("\n🧠 Step 4: Creating enhanced MCTS strategies...")
    
    # Strategy 1: History-aware MCTS
    history_aware_mcts = HistoryAwareMCTS(
        base_mcts=base_mcts,
        history_analyzer=history_analyzer,
        history_weight=0.3
    )
    
    # Strategy 2: Bye week-aware MCTS
    bye_week_aware_mcts = ByeWeekAwareMCTS(
        base_mcts=base_mcts,
        bye_week_optimizer=bye_week_optimizer,
        bye_week_weight=0.2
    )
    
    # Strategy 3: Combined enhanced MCTS
    combined_mcts = CombinedEnhancedMCTS(
        base_mcts=base_mcts,
        history_analyzer=history_analyzer,
        bye_week_optimizer=bye_week_optimizer,
        history_weight=0.25,
        bye_week_weight=0.15
    )
    
    # 5. Run draft simulations
    print("\n🎯 Step 5: Running enhanced draft simulations...")
    
    strategies = {
        'Basic MCTS': base_mcts,
        'History-Aware': history_aware_mcts,
        'Bye Week-Aware': bye_week_aware_mcts,
        'Combined Enhanced': combined_mcts
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\n🔄 Testing {strategy_name}...")
        
        # Create draft
        draft_state = DraftState.create_mock_draft(player_pool, our_team_id=6)
        our_picks = []
        
        # Simulate 8 rounds of drafting
        for round_num in range(8):
            if draft_state.is_our_turn and draft_state.available_players:
                pick = strategy.search(draft_state)
                if pick:
                    our_picks.append(pick)
                    draft_state.make_pick(pick)
                    print(f"   Round {round_num + 1}: {pick.name} ({pick.position}) - Bye Week {getattr(pick, 'metadata', {}).get('bye_week', 'N/A')}")
            
            # Simulate other teams picking (simplified)
            while not draft_state.is_our_turn and draft_state.available_players:
                available = list(draft_state.available_players)
                if available:
                    # Simple opponent pick
                    opponent_pick = np.random.choice(available)
                    draft_state.make_pick(opponent_pick)
                else:
                    break
            
            if draft_state.is_draft_complete():
                break
        
        # Analyze results
        total_vorp = sum(p.vorp for p in our_picks)
        bye_analysis = bye_week_aware_mcts.evaluate_roster_bye_week_strength(our_picks)
        
        results[strategy_name] = {
            'picks': our_picks,
            'total_vorp': total_vorp,
            'bye_analysis': bye_analysis
        }
    
    # 6. Compare results
    print("\n📈 Step 6: Strategy Comparison Results")
    print("=" * 50)
    
    for strategy_name, result in results.items():
        picks = result['picks']
        total_vorp = result['total_vorp']
        bye_analysis = result['bye_analysis']
        
        print(f"\n🎯 {strategy_name}:")
        print(f"   Total VORP: {total_vorp:.2f}")
        print(f"   Bye Week Score: {bye_analysis['score']:.2f}/1.0")
        print(f"   Strategy Alignment: {bye_analysis['strategy_alignment']:.2f}/1.0")
        print(f"   Bye Distribution: {bye_analysis.get('bye_distribution', {})}")
        
        # Show top 3 picks
        print(f"   Top 3 Picks:")
        for i, pick in enumerate(picks[:3], 1):
            bye_week = getattr(pick, 'metadata', {}).get('bye_week', 'N/A')
            print(f"     {i}. {pick.name} ({pick.position}) - VORP: {pick.vorp:.2f}, Bye: {bye_week}")
    
    # 7. Generate strategy reports
    print("\n📋 Step 7: Strategy Reports")
    print("=" * 35)
    
    # History analysis summary
    print(f"\n📚 Draft History Patterns Learned:")
    print(f"{history_analyzer.get_pattern_summary()}")
    
    # Bye week strategy report
    sample_roster = results['Combined Enhanced']['picks']
    bye_week_report = create_bye_week_strategy_report(bye_week_optimizer, sample_roster)
    print(f"\n{bye_week_report}")
    
    print(f"\n✅ Enhanced MCTS Demo Complete!")
    print(f"💡 The enhanced strategies show how draft history and bye week analysis")
    print(f"    can improve draft decision-making beyond simple VORP maximization.")


class CombinedEnhancedMCTS:
    """MCTS that combines both history awareness and bye week optimization"""
    
    def __init__(self, base_mcts, history_analyzer, bye_week_optimizer, 
                 history_weight=0.25, bye_week_weight=0.15):
        self.base_mcts = base_mcts
        self.history_analyzer = history_analyzer
        self.bye_week_optimizer = bye_week_optimizer
        self.history_weight = history_weight
        self.bye_week_weight = bye_week_weight
    
    def search(self, draft_state):
        """Enhanced search with both history and bye week considerations"""
        
        available_players = list(draft_state.available_players)
        if not available_players:
            return None
        
        current_roster = draft_state.get_our_roster()
        enhanced_scores = []
        
        for player in available_players:
            # Base score
            base_score = player.vorp
            
            # History adjustment (if available)
            history_adjustment = 0.0
            if hasattr(self.history_analyzer, '_calculate_history_adjustment'):
                try:
                    history_adjustment = self.history_analyzer._calculate_history_adjustment(player, draft_state)
                except:
                    pass  # Fallback if method not available
            
            # Bye week adjustment
            bye_week_adjustment = self.bye_week_optimizer.calculate_bye_week_penalty(player, current_roster)
            
            # Combined score
            final_score = (base_score + 
                          (history_adjustment * self.history_weight) + 
                          (bye_week_adjustment * self.bye_week_weight))
            
            enhanced_scores.append((player, final_score, history_adjustment, bye_week_adjustment))
        
        # Sort and return best player
        enhanced_scores.sort(key=lambda x: x[1], reverse=True)
        best_player, final_score, history_adj, bye_adj = enhanced_scores[0]
        
        # Log significant adjustments
        if abs(history_adj) > 0.1 or abs(bye_adj) > 0.1:
            adjustments = []
            if abs(history_adj) > 0.1:
                adjustments.append(f"History: {history_adj:+.2f}")
            if abs(bye_adj) > 0.1:
                adjustments.append(f"Bye: {bye_adj:+.2f}")
            
            print(f"🔧 Enhanced pick: {best_player.name} ({', '.join(adjustments)})")
        
        return best_player


if __name__ == "__main__":
    main()
