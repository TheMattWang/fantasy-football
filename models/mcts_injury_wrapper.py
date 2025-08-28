#!/usr/bin/env python3
"""
MCTS Injury Enhancement Wrapper
===============================

This module provides a simple wrapper to add injury awareness to the existing
MCTS draft strategy without major code changes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def enhance_player_with_injury_data(player, injury_weight: float = 0.3) -> object:
    """
    Enhance a basic Player object with injury considerations
    
    Args:
        player: Basic Player object
        injury_weight: How much to weight injury considerations (0.0-1.0)
    
    Returns:
        Enhanced player object with injury-adjusted attributes
    """
    
    # Calculate injury risk based on position and other factors
    position_risk = {
        'RB': 0.35,  # High contact position
        'WR': 0.25,  # Moderate risk
        'QB': 0.20,  # Lower contact but important
        'TE': 0.30,  # Moderate-high risk
        'K': 0.10,   # Low risk
        'DEF': 0.15  # Low-moderate risk
    }
    
    base_injury_risk = position_risk.get(player.position, 0.25)
    
    # Adjust for age if available
    age = getattr(player, 'age', 25)
    age_factor = 1.0 + max(0, (age - 25) * 0.02)  # 2% increase per year over 25
    
    # Calculate final injury risk
    injury_risk = min(1.0, base_injury_risk * age_factor)
    
    # Calculate durability score (inverse of injury risk)
    durability_score = 1.0 - injury_risk
    
    # Apply injury adjustment to VORP
    injury_adjustment = 1.0 - (injury_risk * injury_weight)
    original_vorp = player.vorp
    adjusted_vorp = original_vorp * injury_adjustment
    
    # Update player attributes
    player.injury_risk_score = injury_risk
    player.durability_score = durability_score
    player.original_vorp = original_vorp
    player.injury_adjusted_vorp = adjusted_vorp
    
    # Apply the adjustment to the main VORP (this affects draft decisions)
    player.vorp = adjusted_vorp
    
    return player

def enhance_player_pool_with_injury_data(player_pool: Dict, injury_weight: float = 0.3) -> Dict:
    """
    Enhance entire player pool with injury considerations
    
    Args:
        player_pool: Dictionary of player_name -> Player objects
        injury_weight: How much to weight injury considerations
    
    Returns:
        Enhanced player pool with injury-adjusted values
    """
    
    enhanced_pool = {}
    
    print(f"🏥 Enhancing {len(player_pool)} players with injury considerations...")
    print(f"   Injury weight: {injury_weight:.1%}")
    
    for name, player in player_pool.items():
        enhanced_player = enhance_player_with_injury_data(player, injury_weight)
        enhanced_pool[name] = enhanced_player
    
    # Show impact summary
    total_vorp_change = sum(
        getattr(p, 'injury_adjusted_vorp', p.vorp) - getattr(p, 'original_vorp', p.vorp)
        for p in enhanced_pool.values()
    )
    
    # Show examples
    sorted_players = sorted(enhanced_pool.values(), key=lambda p: p.vorp, reverse=True)
    
    print(f"✅ Enhancement complete!")
    print(f"   Total VORP change: {total_vorp_change:+.2f}")
    print(f"   Average injury risk: {np.mean([getattr(p, 'injury_risk_score', 0.25) for p in enhanced_pool.values()]):.3f}")
    
    print(f"\n📊 Top 5 Players After Injury Adjustment:")
    for i, player in enumerate(sorted_players[:5], 1):
        original = getattr(player, 'original_vorp', player.vorp)
        risk = getattr(player, 'injury_risk_score', 0.25)
        print(f"   {i}. {player.name:20s} ({player.position}) - VORP: {original:.2f} → {player.vorp:.2f} (Risk: {risk:.3f})")
    
    return enhanced_pool

class InjuryAwareRewardFunction:
    """Enhanced reward function that can replace the existing one"""
    
    def __init__(self, 
                 risk_penalty: float = 0.1,
                 overstack_penalty: float = 0.5,
                 bye_penalty: float = 0.2,
                 injury_bonus_weight: float = 0.1):
        self.risk_penalty = risk_penalty
        self.overstack_penalty = overstack_penalty
        self.bye_penalty = bye_penalty
        self.injury_bonus_weight = injury_bonus_weight
    
    def calculate_roster_value(self, roster: List, league) -> float:
        """Calculate roster value with injury considerations (drop-in replacement)"""
        if not roster:
            return 0.0
        
        # Base VORP sum (already injury-adjusted if using enhanced players)
        base_vorp = sum(player.vorp for player in roster)
        
        # Traditional risk penalty
        risk_penalty = self.risk_penalty * sum(getattr(player, 'risk_sigma', 0.0) for player in roster)
        
        # Injury diversification bonus
        injury_bonus = self._calculate_injury_diversification_bonus(roster)
        
        # Position overstacking penalty (existing logic)
        overstack_penalty = self._calculate_overstack_penalty(roster, league)
        
        # Bye week penalty (existing logic)
        bye_penalty = self._calculate_bye_penalty(roster)
        
        total_value = base_vorp - risk_penalty - overstack_penalty - bye_penalty + injury_bonus
        return total_value
    
    def _calculate_injury_diversification_bonus(self, roster: List) -> float:
        """Bonus for having a mix of injury risk levels"""
        if len(roster) < 2:
            return 0.0
        
        injury_risks = [getattr(player, 'injury_risk_score', 0.25) for player in roster]
        
        # Bonus for risk variance (having mix of high and low risk players)
        risk_variance = np.var(injury_risks)
        diversification_bonus = self.injury_bonus_weight * risk_variance
        
        # Extra bonus for having some very durable players
        high_durability_count = sum(1 for risk in injury_risks if risk < 0.2)
        durability_bonus = min(0.5, high_durability_count * 0.1)
        
        return diversification_bonus + durability_bonus
    
    def _calculate_overstack_penalty(self, roster: List, league) -> float:
        """Calculate position overstacking penalty (existing logic)"""
        position_counts = {}
        for player in roster:
            pos = player.position
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        overstack_penalty = 0.0
        roster_spots = getattr(league, 'roster_spots', {
            'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DEF': 1, 'K': 1, 'BENCH': 6
        })
        flex_positions = getattr(league, 'flex_positions', {'RB', 'WR', 'TE'})
        
        for pos, count in position_counts.items():
            max_useful = roster_spots.get(pos, 0)
            if pos in flex_positions:
                max_useful += roster_spots.get('FLEX', 0)
            max_useful += 2  # Allow some bench depth
            
            if count > max_useful:
                overstack_penalty += self.overstack_penalty * (count - max_useful) ** 2
        
        return overstack_penalty
    
    def _calculate_bye_penalty(self, roster: List) -> float:
        """Calculate bye week penalty (existing logic)"""
        bye_weeks = [getattr(player, 'bye_week', 0) for player in roster if getattr(player, 'bye_week', 0) > 0]
        bye_conflicts = len(bye_weeks) - len(set(bye_weeks))
        return self.bye_penalty * bye_conflicts
    
    def calculate_pick_reward(self, player, current_roster: List, league) -> float:
        """Calculate pick reward with injury considerations (drop-in replacement)"""
        
        # Base reward (already injury-adjusted if using enhanced players)
        base_reward = player.vorp
        
        # Position need bonus (existing logic)
        position_counts = {}
        for p in current_roster:
            pos = p.position
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        roster_spots = getattr(league, 'roster_spots', {})
        needs = {}
        for pos, required in roster_spots.items():
            needs[pos] = max(0, required - position_counts.get(pos, 0))
        
        need_bonus = 0.0
        if player.position in needs and needs[player.position] > 0:
            need_bonus = player.vorp * 0.2
        elif player.position in getattr(league, 'flex_positions', set()) and needs.get('FLEX', 0) > 0:
            need_bonus = player.vorp * 0.1
        
        # Risk penalty (existing logic)
        risk_penalty = self.risk_penalty * getattr(player, 'risk_sigma', 0.0)
        
        # Injury diversification bonus
        injury_bonus = 0.0
        if current_roster:
            current_risks = [getattr(p, 'injury_risk_score', 0.25) for p in current_roster]
            avg_current_risk = np.mean(current_risks)
            player_risk = getattr(player, 'injury_risk_score', 0.25)
            
            # Bonus for adding diversity
            if avg_current_risk > 0.5 and player_risk < 0.3:
                injury_bonus = 0.2  # Adding low-risk player to high-risk roster
            elif avg_current_risk < 0.2 and 0.3 < player_risk < 0.5:
                injury_bonus = 0.1  # Adding moderate risk when too conservative
        
        # Early round scarcity bonus (existing logic)
        round_num = len(current_roster) // 12 + 1
        scarcity_bonus = 0.0
        if round_num <= 3 and player.position in ['RB', 'WR']:
            scarcity_bonus = player.vorp * 0.15
        
        total_reward = base_reward + need_bonus + scarcity_bonus + injury_bonus - risk_penalty
        return total_reward

def create_injury_aware_draft_board(player_pool: Dict, injury_weight: float = 0.3) -> pd.DataFrame:
    """Create a draft board with injury-adjusted rankings"""
    
    print(f"📋 Creating injury-aware draft board...")
    
    board_data = []
    for name, player in player_pool.items():
        # Get injury attributes
        injury_risk = getattr(player, 'injury_risk_score', 0.25)
        durability = getattr(player, 'durability_score', 0.75)
        original_vorp = getattr(player, 'original_vorp', player.vorp)
        adjusted_vorp = getattr(player, 'injury_adjusted_vorp', player.vorp)
        
        # Calculate injury tier
        if injury_risk < 0.25:
            injury_tier = "Iron Man"
        elif injury_risk < 0.4:
            injury_tier = "Reliable"
        elif injury_risk < 0.6:
            injury_tier = "Moderate Risk"
        else:
            injury_tier = "High Risk"
        
        board_data.append({
            'player_name': name,
            'position': player.position,
            'original_vorp': original_vorp,
            'injury_adjusted_vorp': adjusted_vorp,
            'vorp_change': adjusted_vorp - original_vorp,
            'injury_risk': injury_risk,
            'durability': durability,
            'injury_tier': injury_tier,
            'adp_rank': getattr(player, 'adp_rank', 999)
        })
    
    # Create DataFrame and sort
    df = pd.DataFrame(board_data)
    df = df.sort_values('injury_adjusted_vorp', ascending=False)
    df['original_rank'] = df['original_vorp'].rank(ascending=False, method='min')
    df['injury_adjusted_rank'] = range(1, len(df) + 1)
    df['rank_change'] = df['original_rank'] - df['injury_adjusted_rank']
    
    print(f"✅ Draft board created with {len(df)} players")
    
    # Show biggest movers
    print(f"\n📈 Biggest Movers UP (Better after injury adjustment):")
    movers_up = df[df['rank_change'] > 3].head(5)
    for _, player in movers_up.iterrows():
        print(f"   {player['player_name']:20s} ({player['position']}) - Moved up {player['rank_change']:2.0f} spots - Risk: {player['injury_risk']:.3f}")
    
    print(f"\n📉 Biggest Movers DOWN (Worse after injury adjustment):")
    movers_down = df[df['rank_change'] < -3].head(5)
    for _, player in movers_down.iterrows():
        print(f"   {player['player_name']:20s} ({player['position']}) - Moved down {abs(player['rank_change']):2.0f} spots - Risk: {player['injury_risk']:.3f}")
    
    return df

# Simple integration functions for existing notebook
def add_injury_awareness_to_existing_system(player_pool, reward_function, injury_weight=0.3):
    """
    Simple function to add injury awareness to existing MCTS system
    
    Usage in notebook:
    # After creating player_pool and reward_function
    enhanced_pool, enhanced_reward_fn = add_injury_awareness_to_existing_system(
        player_pool, reward_function, injury_weight=0.3
    )
    
    # Use enhanced_pool and enhanced_reward_fn in your MCTS
    """
    print("🏥 Adding injury awareness to existing MCTS system...")
    
    # Enhance player pool
    enhanced_pool = enhance_player_pool_with_injury_data(player_pool, injury_weight)
    
    # Create enhanced reward function
    enhanced_reward_fn = InjuryAwareRewardFunction(
        risk_penalty=getattr(reward_function, 'risk_penalty', 0.1),
        overstack_penalty=getattr(reward_function, 'overstack_penalty', 0.5),
        bye_penalty=getattr(reward_function, 'bye_penalty', 0.2)
    )
    
    print("✅ Injury awareness integration complete!")
    print("   💡 Use enhanced_pool and enhanced_reward_fn in your MCTS strategy")
    
    return enhanced_pool, enhanced_reward_fn

if __name__ == "__main__":
    print("🏥 MCTS Injury Enhancement Wrapper")
    print("This module provides simple functions to add injury awareness to existing MCTS.")
    print("\nKey functions:")
    print("  • enhance_player_pool_with_injury_data() - Enhance player values")
    print("  • InjuryAwareRewardFunction() - Drop-in replacement for reward function")
    print("  • add_injury_awareness_to_existing_system() - One-line integration")
    print("  • create_injury_aware_draft_board() - Generate injury-adjusted rankings")
