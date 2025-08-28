#!/usr/bin/env python3
"""
Simplified Injury Enhancement Demo
==================================

This script demonstrates the core injury enhancement features for fantasy football.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add models directory to path
sys.path.append(str(Path(__file__).parent / "models"))

from injury_enhancement import InjuryDataCollector

def run_injury_demo():
    """Run a simplified injury enhancement demonstration"""
    
    print("🏥 Fantasy Football Injury Enhancement Demo")
    print("=" * 50)
    
    # Load existing data
    data_file = Path("rookie_data_clean.csv")
    
    if data_file.exists():
        print(f"📊 Loading data from {data_file}")
        df = pd.read_csv(data_file)
    else:
        print("⚠️  Creating sample data for demonstration...")
        df = create_sample_data()
    
    print(f"   Loaded {len(df)} players")
    
    # Initialize injury collector
    injury_collector = InjuryDataCollector()
    
    # Enhance subset of data for demo (first 50 players)
    demo_df = df.head(50).copy()
    
    print(f"\n🏗️  Enhancing {len(demo_df)} players with injury features...")
    enhanced_df = injury_collector.enhance_player_data_with_injuries(demo_df)
    
    # Show results
    print(f"\n📈 Enhancement Results:")
    print(f"   Original features: {len(demo_df.columns)}")
    print(f"   Enhanced features: {len(enhanced_df.columns)}")
    print(f"   New injury features: {len(enhanced_df.columns) - len(demo_df.columns)}")
    
    # Show injury feature statistics
    injury_features = [col for col in enhanced_df.columns if 'injury' in col.lower() or 'risk' in col.lower() or 'durability' in col.lower()]
    
    print(f"\n🏥 Injury Feature Statistics:")
    for feature in injury_features:
        if feature in enhanced_df.columns:
            print(f"   {feature:25s}: avg={enhanced_df[feature].mean():.3f}, std={enhanced_df[feature].std():.3f}")
    
    # Show players by injury risk
    if 'injury_risk_score' in enhanced_df.columns:
        print(f"\n📊 Players by Injury Risk:")
        high_risk = enhanced_df[enhanced_df['injury_risk_score'] > 0.6]
        low_risk = enhanced_df[enhanced_df['injury_risk_score'] < 0.3]
        
        print(f"   High Risk (>0.6): {len(high_risk)} players")
        if not high_risk.empty:
            for _, player in high_risk.head(3).iterrows():
                print(f"     • {player['player_name']} ({player['position']}) - Risk: {player['injury_risk_score']:.3f}")
        
        print(f"   Low Risk (<0.3): {len(low_risk)} players")
        if not low_risk.empty:
            for _, player in low_risk.head(3).iterrows():
                print(f"     • {player['player_name']} ({player['position']}) - Risk: {player['injury_risk_score']:.3f}")
    
    # Create simple visualization
    create_injury_visualization(enhanced_df)
    
    # Save enhanced data
    output_file = "injury_enhanced_demo.csv"
    enhanced_df.to_csv(output_file, index=False)
    print(f"\n💾 Enhanced data saved to {output_file}")
    
    # Show impact on draft strategy
    demonstrate_draft_impact(enhanced_df)
    
    print(f"\n✅ Injury Enhancement Demo Complete!")
    print(f"🏥 Key Insights:")
    print(f"   • Injury risk varies significantly by position and player")
    print(f"   • Enhanced model includes {len(injury_features)} injury-related features")
    print(f"   • Risk scores can be used to adjust draft valuations")
    print(f"   • System provides foundation for more sophisticated injury analysis")
    
    return enhanced_df

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_players = 100
    
    positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
    teams = ['DAL', 'NYG', 'PHI', 'WAS', 'GB', 'MIN', 'CHI', 'DET']
    
    data = {
        'player_name': [f"Player_{i:03d}" for i in range(n_players)],
        'position': np.random.choice(positions, n_players, p=[0.1, 0.25, 0.35, 0.15, 0.05, 0.1]),
        'team': np.random.choice(teams, n_players),
        'age': np.random.randint(21, 30, n_players),
        'games_played': np.random.randint(8, 17, n_players),
        'ppg': np.random.gamma(2, 3),  # Right-skewed distribution
    }
    
    # Add basic stats
    data['games_played_pct'] = np.array(data['games_played']) / 17
    
    return pd.DataFrame(data)

def create_injury_visualization(df):
    """Create simple injury visualization"""
    try:
        if 'injury_risk_score' not in df.columns or 'position' not in df.columns:
            print("⚠️  Skipping visualization - missing required columns")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Risk distribution
        plt.subplot(1, 2, 1)
        plt.hist(df['injury_risk_score'], bins=15, alpha=0.7, color='red', edgecolor='black')
        plt.title('Injury Risk Distribution')
        plt.xlabel('Injury Risk Score')
        plt.ylabel('Number of Players')
        
        # Plot 2: Risk by position
        plt.subplot(1, 2, 2)
        position_risk = df.groupby('position')['injury_risk_score'].mean().sort_values(ascending=False)
        plt.bar(position_risk.index, position_risk.values, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Average Injury Risk by Position')
        plt.xlabel('Position')
        plt.ylabel('Average Risk Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('injury_analysis_demo.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to injury_analysis_demo.png")
        
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")

def demonstrate_draft_impact(df):
    """Demonstrate impact on draft strategy"""
    print(f"\n🎯 Draft Strategy Impact Analysis:")
    
    if 'injury_risk_score' not in df.columns or 'ppg' not in df.columns:
        print("⚠️  Cannot demonstrate draft impact - missing required columns")
        return
    
    # Calculate injury-adjusted values
    injury_weight = 0.3  # 30% weight to injury considerations
    
    df['base_value'] = df['ppg']  # Use PPG as base value
    df['injury_adjustment'] = 1.0 - (df['injury_risk_score'] * injury_weight)
    df['adjusted_value'] = df['base_value'] * df['injury_adjustment']
    df['value_change'] = df['adjusted_value'] - df['base_value']
    
    # Sort by original value
    df_sorted = df.sort_values('base_value', ascending=False)
    df_sorted['original_rank'] = range(1, len(df_sorted) + 1)
    
    # Sort by adjusted value
    df_sorted = df_sorted.sort_values('adjusted_value', ascending=False)
    df_sorted['adjusted_rank'] = range(1, len(df_sorted) + 1)
    df_sorted['rank_change'] = df_sorted['original_rank'] - df_sorted['adjusted_rank']
    
    # Show biggest movers
    print(f"\n📈 Biggest Movers UP (Lower Injury Risk):")
    movers_up = df_sorted[df_sorted['rank_change'] > 2].head(5)
    for _, player in movers_up.iterrows():
        print(f"   {player['player_name']:15s} ({player['position']}) - Moved up {player['rank_change']:2.0f} spots - Risk: {player['injury_risk_score']:.3f}")
    
    print(f"\n📉 Biggest Movers DOWN (Higher Injury Risk):")
    movers_down = df_sorted[df_sorted['rank_change'] < -2].head(5)
    for _, player in movers_down.iterrows():
        print(f"   {player['player_name']:15s} ({player['position']}) - Moved down {abs(player['rank_change']):2.0f} spots - Risk: {player['injury_risk_score']:.3f}")
    
    # Summary statistics
    avg_risk = df['injury_risk_score'].mean()
    high_risk_count = (df['injury_risk_score'] > 0.6).sum()
    low_risk_count = (df['injury_risk_score'] < 0.3).sum()
    
    print(f"\n📊 Summary:")
    print(f"   Average injury risk: {avg_risk:.3f}")
    print(f"   High risk players (>0.6): {high_risk_count}")
    print(f"   Low risk players (<0.3): {low_risk_count}")
    print(f"   Players with significant rank changes: {len(df_sorted[abs(df_sorted['rank_change']) > 2])}")

if __name__ == "__main__":
    enhanced_data = run_injury_demo()
