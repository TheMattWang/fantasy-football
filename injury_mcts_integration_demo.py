#!/usr/bin/env python3
"""
Injury-Aware MCTS Integration Demo
==================================

This script demonstrates how the injury enhancement system integrates
with the existing MCTS draft strategy.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Add models directory to path
sys.path.append(str(Path(__file__).parent / "models"))

from injury_aware_mcts import (
    InjuryAwarePlayer, 
    InjuryAwareRewardFunction, 
    InjuryAwareValueFunction,
    InjuryAwareDraftStrategy,
    create_injury_aware_player_pool
)

class MockDraftState:
    """Mock draft state for demonstration"""
    def __init__(self, available_players, our_team_id=1):
        self.available_players = set(available_players)
        self.team_rosters = defaultdict(list)
        self.our_team_id = our_team_id
        self.current_round = 1
        self.current_pick_in_round = 1

def run_injury_mcts_integration_demo():
    """Run complete integration demo"""
    
    print("🏥 Injury-Aware MCTS Integration Demo")
    print("=" * 50)
    
    # Step 1: Load and prepare data
    print("\n📊 Step 1: Loading and preparing data...")
    
    # Load basic player data
    try:
        df = pd.read_csv('rookie_data_clean.csv')
        print(f"✅ Loaded {len(df)} players from rookie_data_clean.csv")
    except FileNotFoundError:
        print("⚠️  Creating sample data for demo...")
        df = create_sample_data()
    
    # Create basic player pool (mock)
    basic_player_pool = create_mock_basic_player_pool(df.head(50))
    print(f"✅ Created basic player pool with {len(basic_player_pool)} players")
    
    # Step 2: Create injury-aware player pool
    print("\n🏥 Step 2: Creating injury-aware player pool...")
    
    # Try to use enhanced data if available
    enhanced_data_path = 'injury_enhanced_demo.csv'
    injury_aware_pool = create_injury_aware_player_pool(basic_player_pool, enhanced_data_path)
    
    # Step 3: Compare strategies
    print("\n⚡ Step 3: Comparing Traditional vs Injury-Aware MCTS...")
    
    # Traditional strategy (mock)
    traditional_results = run_traditional_mcts_mock(basic_player_pool)
    
    # Injury-aware strategy
    injury_aware_results = run_injury_aware_mcts(injury_aware_pool)
    
    # Step 4: Analysis and comparison
    print("\n📈 Step 4: Analysis and Comparison...")
    
    analyze_strategy_differences(traditional_results, injury_aware_results, injury_aware_pool)
    
    # Step 5: Create visualizations
    print("\n📊 Step 5: Creating visualizations...")
    create_comparison_visualizations(traditional_results, injury_aware_results, injury_aware_pool)
    
    print("\n✅ Injury-Aware MCTS Integration Demo Complete!")
    
    return {
        'basic_pool': basic_player_pool,
        'injury_aware_pool': injury_aware_pool,
        'traditional_results': traditional_results,
        'injury_aware_results': injury_aware_results
    }

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_players = 100
    
    positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
    data = {
        'player_name': [f"Player_{i:03d}" for i in range(n_players)],
        'position': np.random.choice(positions, n_players, p=[0.1, 0.25, 0.35, 0.15, 0.05, 0.1]),
        'ppg': np.random.gamma(2, 3),
        'adp_rank': range(1, n_players + 1)
    }
    
    return pd.DataFrame(data)

def create_mock_basic_player_pool(df):
    """Create mock basic player pool"""
    
    class BasicPlayer:
        def __init__(self, name, position, vorp, adp_rank):
            self.name = name
            self.position = position
            self.vorp = vorp
            self.adp_rank = adp_rank
            self.proj_ppg = vorp + 5  # Mock projection
            self.risk_sigma = np.random.uniform(0.1, 0.5)  # Mock uncertainty
        
        def __hash__(self):
            return hash(self.name)
    
    player_pool = {}
    for _, row in df.iterrows():
        player = BasicPlayer(
            row['player_name'],
            row['position'],
            row.get('ppg', 5.0),  # Use ppg as VORP proxy
            row.get('adp_rank', 999)
        )
        player_pool[player.name] = player
    
    return player_pool

def run_traditional_mcts_mock(basic_player_pool):
    """Mock traditional MCTS results"""
    print("📊 Running traditional MCTS (mock)...")
    
    # Sort by basic VORP
    sorted_players = sorted(basic_player_pool.values(), key=lambda p: p.vorp, reverse=True)
    
    # Mock draft picks (top 10 players)
    draft_picks = []
    for i, player in enumerate(sorted_players[:10]):
        draft_picks.append({
            'round': (i // 12) + 1,
            'pick': i + 1,
            'player_name': player.name,
            'position': player.position,
            'vorp': player.vorp,
            'selection_reason': 'Highest VORP available'
        })
    
    total_vorp = sum(pick['vorp'] for pick in draft_picks)
    
    return {
        'strategy_name': 'Traditional MCTS',
        'draft_picks': draft_picks,
        'total_vorp': total_vorp,
        'avg_vorp_per_pick': total_vorp / len(draft_picks),
        'methodology': 'VORP-based selection with basic uncertainty'
    }

def run_injury_aware_mcts(injury_aware_pool):
    """Run injury-aware MCTS"""
    print("🏥 Running injury-aware MCTS...")
    
    # Initialize injury-aware strategy
    strategy = InjuryAwareDraftStrategy(injury_aware_pool)
    
    # Create mock draft state
    draft_state = MockDraftState(list(injury_aware_pool.values()))
    
    # Simulate draft picks
    draft_picks = []
    for round_num in range(1, 11):  # 10 rounds
        
        # Make pick using injury-aware strategy
        selected_player = strategy.make_pick(draft_state)
        
        if selected_player:
            # Remove player from available pool
            draft_state.available_players.discard(selected_player)
            draft_state.team_rosters[draft_state.our_team_id].append(selected_player)
            
            # Calculate injury-adjusted value
            injury_adjustment = 1.0 - (selected_player.injury_risk_score * 0.3)
            adjusted_vorp = selected_player.vorp * injury_adjustment
            
            draft_picks.append({
                'round': round_num,
                'pick': round_num,  # Simplified
                'player_name': selected_player.name,
                'position': selected_player.position,
                'vorp': selected_player.vorp,
                'injury_risk': selected_player.injury_risk_score,
                'durability': selected_player.durability_score,
                'adjusted_vorp': adjusted_vorp,
                'selection_reason': f'Injury-aware optimization (Risk: {selected_player.injury_risk_score:.3f})'
            })
            
            draft_state.current_round += 1
    
    total_vorp = sum(pick['vorp'] for pick in draft_picks)
    total_adjusted_vorp = sum(pick['adjusted_vorp'] for pick in draft_picks)
    
    return {
        'strategy_name': 'Injury-Aware MCTS',
        'draft_picks': draft_picks,
        'total_vorp': total_vorp,
        'total_adjusted_vorp': total_adjusted_vorp,
        'avg_vorp_per_pick': total_vorp / len(draft_picks),
        'avg_adjusted_vorp_per_pick': total_adjusted_vorp / len(draft_picks),
        'avg_injury_risk': np.mean([pick['injury_risk'] for pick in draft_picks]),
        'avg_durability': np.mean([pick['durability'] for pick in draft_picks]),
        'methodology': 'Injury-risk adjusted VORP with durability bonuses'
    }

def analyze_strategy_differences(traditional_results, injury_aware_results, injury_aware_pool):
    """Analyze differences between strategies"""
    
    print("\n🔍 Strategy Comparison Analysis:")
    print("=" * 40)
    
    # Overall performance comparison
    trad_vorp = traditional_results['total_vorp']
    injury_vorp = injury_aware_results['total_vorp']
    injury_adjusted_vorp = injury_aware_results['total_adjusted_vorp']
    
    print(f"📊 Total VORP Comparison:")
    print(f"   Traditional MCTS: {trad_vorp:.2f}")
    print(f"   Injury-Aware MCTS: {injury_vorp:.2f} (raw)")
    print(f"   Injury-Aware MCTS: {injury_adjusted_vorp:.2f} (risk-adjusted)")
    
    vorp_difference = injury_vorp - trad_vorp
    adjusted_difference = injury_adjusted_vorp - trad_vorp
    
    print(f"\n📈 Performance Difference:")
    print(f"   Raw VORP difference: {vorp_difference:+.2f}")
    print(f"   Risk-adjusted difference: {adjusted_difference:+.2f}")
    
    # Injury risk analysis
    avg_injury_risk = injury_aware_results['avg_injury_risk']
    avg_durability = injury_aware_results['avg_durability']
    
    print(f"\n🏥 Injury Profile of Injury-Aware Picks:")
    print(f"   Average injury risk: {avg_injury_risk:.3f}")
    print(f"   Average durability: {avg_durability:.3f}")
    
    # Position comparison
    trad_positions = Counter(pick['position'] for pick in traditional_results['draft_picks'])
    injury_positions = Counter(pick['position'] for pick in injury_aware_results['draft_picks'])
    
    print(f"\n🎯 Position Distribution:")
    print(f"   Traditional: {dict(trad_positions)}")
    print(f"   Injury-Aware: {dict(injury_positions)}")
    
    # Player overlap analysis
    trad_players = set(pick['player_name'] for pick in traditional_results['draft_picks'])
    injury_players = set(pick['player_name'] for pick in injury_aware_results['draft_picks'])
    
    overlap = len(trad_players & injury_players)
    overlap_pct = (overlap / len(trad_players)) * 100
    
    print(f"\n🔄 Player Overlap:")
    print(f"   Common players: {overlap}/{len(trad_players)} ({overlap_pct:.1f}%)")
    
    # Show unique picks
    unique_to_injury = injury_players - trad_players
    unique_to_traditional = trad_players - injury_players
    
    if unique_to_injury:
        print(f"\n🏥 Players unique to injury-aware strategy:")
        for player_name in list(unique_to_injury)[:3]:  # Show first 3
            player = injury_aware_pool[player_name]
            print(f"   • {player_name} ({player.position}) - Risk: {player.injury_risk_score:.3f}, Durability: {player.durability_score:.3f}")
    
    if unique_to_traditional:
        print(f"\n📊 Players unique to traditional strategy:")
        for player_name in list(unique_to_traditional)[:3]:  # Show first 3
            player = injury_aware_pool[player_name]
            print(f"   • {player_name} ({player.position}) - Risk: {player.injury_risk_score:.3f}, Durability: {player.durability_score:.3f}")

def create_comparison_visualizations(traditional_results, injury_aware_results, injury_aware_pool):
    """Create visualizations comparing the strategies"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Traditional vs Injury-Aware MCTS Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: VORP comparison
        strategies = ['Traditional', 'Injury-Aware\n(Raw)', 'Injury-Aware\n(Risk-Adj)']
        vorp_values = [
            traditional_results['total_vorp'],
            injury_aware_results['total_vorp'],
            injury_aware_results['total_adjusted_vorp']
        ]
        
        axes[0, 0].bar(strategies, vorp_values, color=['blue', 'orange', 'green'], alpha=0.7)
        axes[0, 0].set_title('Total VORP Comparison')
        axes[0, 0].set_ylabel('Total VORP')
        
        # Plot 2: Position distribution
        trad_positions = Counter(pick['position'] for pick in traditional_results['draft_picks'])
        injury_positions = Counter(pick['position'] for pick in injury_aware_results['draft_picks'])
        
        positions = list(set(trad_positions.keys()) | set(injury_positions.keys()))
        trad_counts = [trad_positions.get(pos, 0) for pos in positions]
        injury_counts = [injury_positions.get(pos, 0) for pos in positions]
        
        x = np.arange(len(positions))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, trad_counts, width, label='Traditional', alpha=0.7, color='blue')
        axes[0, 1].bar(x + width/2, injury_counts, width, label='Injury-Aware', alpha=0.7, color='orange')
        axes[0, 1].set_title('Position Distribution')
        axes[0, 1].set_xlabel('Position')
        axes[0, 1].set_ylabel('Number of Picks')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(positions)
        axes[0, 1].legend()
        
        # Plot 3: Injury risk distribution of injury-aware picks
        injury_risks = [pick['injury_risk'] for pick in injury_aware_results['draft_picks']]
        axes[1, 0].hist(injury_risks, bins=8, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_title('Injury Risk Distribution\n(Injury-Aware Picks)')
        axes[1, 0].set_xlabel('Injury Risk Score')
        axes[1, 0].set_ylabel('Number of Players')
        
        # Plot 4: VORP vs Injury Risk scatter
        injury_picks_df = pd.DataFrame(injury_aware_results['draft_picks'])
        
        scatter = axes[1, 1].scatter(
            injury_picks_df['injury_risk'], 
            injury_picks_df['vorp'],
            c=injury_picks_df['durability'],
            cmap='RdYlGn',
            s=100,
            alpha=0.7,
            edgecolors='black'
        )
        axes[1, 1].set_title('VORP vs Injury Risk\n(Color = Durability)')
        axes[1, 1].set_xlabel('Injury Risk Score')
        axes[1, 1].set_ylabel('VORP')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Durability Score')
        
        plt.tight_layout()
        plt.savefig('injury_mcts_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Visualizations saved to injury_mcts_comparison.png")
        
    except Exception as e:
        print(f"⚠️  Visualization creation failed: {e}")

if __name__ == "__main__":
    results = run_injury_mcts_integration_demo()
    
    print(f"\n🎯 Integration Demo Summary:")
    print(f"   • Successfully integrated injury awareness with MCTS")
    print(f"   • Injury-aware strategy considers {len(results['injury_aware_pool'])} players")
    print(f"   • Demonstrates risk-adjusted player valuation")
    print(f"   • Shows impact of injury considerations on draft strategy")
    print(f"\n🏥 The injury-aware MCTS is now ready for use in your draft strategy!")
