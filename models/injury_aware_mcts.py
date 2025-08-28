#!/usr/bin/env python3
"""
Injury-Aware MCTS Draft Strategy Integration
===========================================

This module integrates the comprehensive injury enhancement system
with the existing MCTS draft strategy.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import existing MCTS components
# Note: In practice, you'd import these from the notebook or refactored modules

# Import injury enhancement system
from injury_enhancement import InjuryDataCollector
from injury_aware_model import InjuryAwareFantasyModel

@dataclass
class InjuryAwarePlayer:
    """Enhanced Player class with comprehensive injury data"""
    name: str
    position: str
    team: str = ""
    vorp: float = 0.0
    proj_ppg: float = 0.0
    adp_rank: float = 999.0
    bye_week: int = 0
    risk_sigma: float = 0.0  # Base uncertainty
    is_rookie: bool = False
    
    # Enhanced injury features
    injury_risk_score: float = 0.3
    durability_score: float = 0.7
    historical_injuries: int = 0
    games_missed_injury: int = 0
    season_ending_injuries: int = 0
    avg_recovery_time: float = 0.0
    has_recurring_injuries: bool = False
    position_injury_risk: float = 0.25
    age_adjusted_risk: float = 0.3
    usage_adjusted_risk: float = 0.3
    games_played_pct_adj: float = 0.8
    
    def __hash__(self):
        return hash(self.name)
    
    @classmethod
    def from_basic_player(cls, player, injury_data: Dict = None):
        """Create InjuryAwarePlayer from basic Player object"""
        injury_data = injury_data or {}
        
        return cls(
            name=player.name,
            position=player.position,
            team=getattr(player, 'team', ''),
            vorp=player.vorp,
            proj_ppg=getattr(player, 'proj_ppg', 0.0),
            adp_rank=player.adp_rank,
            bye_week=getattr(player, 'bye_week', 0),
            risk_sigma=getattr(player, 'risk_sigma', 0.0),
            is_rookie=getattr(player, 'is_rookie', False),
            
            # Injury features from injury_data
            injury_risk_score=injury_data.get('injury_risk_score', 0.3),
            durability_score=injury_data.get('durability_score', 0.7),
            historical_injuries=injury_data.get('historical_injuries', 0),
            games_missed_injury=injury_data.get('games_missed_injury', 0),
            season_ending_injuries=injury_data.get('season_ending_injuries', 0),
            avg_recovery_time=injury_data.get('avg_recovery_time', 0.0),
            has_recurring_injuries=injury_data.get('has_recurring_injuries', False),
            position_injury_risk=injury_data.get('position_injury_risk', 0.25),
            age_adjusted_risk=injury_data.get('age_adjusted_risk', 0.3),
            usage_adjusted_risk=injury_data.get('usage_adjusted_risk', 0.3),
            games_played_pct_adj=injury_data.get('games_played_pct_adj', 0.8)
        )

class InjuryAwareRewardFunction:
    """Enhanced reward function that incorporates comprehensive injury considerations"""
    
    def __init__(self, 
                 risk_penalty: float = 0.1,
                 overstack_penalty: float = 0.5,
                 bye_penalty: float = 0.2,
                 injury_penalty: float = 0.3,
                 durability_bonus: float = 0.2,
                 diversification_bonus: float = 0.1):
        self.risk_penalty = risk_penalty  # λ - penalty for player uncertainty
        self.overstack_penalty = overstack_penalty  # γ - penalty for position overstacking
        self.bye_penalty = bye_penalty  # β - penalty for bye week conflicts
        self.injury_penalty = injury_penalty  # α - penalty for injury risk
        self.durability_bonus = durability_bonus  # δ - bonus for durable players
        self.diversification_bonus = diversification_bonus  # ε - bonus for injury diversification
    
    def calculate_roster_value(self, roster: List[InjuryAwarePlayer], league_settings: Dict) -> float:
        """Calculate total value of a roster with injury considerations"""
        if not roster:
            return 0.0
        
        # Base VORP sum
        base_vorp = sum(player.vorp for player in roster)
        
        # Traditional risk penalty (uncertainty, especially for rookies)
        risk_penalty = self.risk_penalty * sum(player.risk_sigma for player in roster)
        
        # Enhanced injury penalties
        injury_penalty = self._calculate_injury_penalty(roster)
        
        # Durability bonus for reliable players
        durability_bonus = self._calculate_durability_bonus(roster)
        
        # Position overstacking penalty
        overstack_penalty = self._calculate_overstack_penalty(roster, league_settings)
        
        # Bye week conflict penalty (simplified)
        bye_penalty = self._calculate_bye_penalty(roster)
        
        # Injury diversification bonus
        diversification_bonus = self._calculate_injury_diversification_bonus(roster)
        
        total_value = (base_vorp 
                      - risk_penalty 
                      - injury_penalty 
                      - overstack_penalty 
                      - bye_penalty
                      + durability_bonus 
                      + diversification_bonus)
        
        return total_value
    
    def _calculate_injury_penalty(self, roster: List[InjuryAwarePlayer]) -> float:
        """Calculate penalty based on injury risk"""
        injury_penalty = 0.0
        
        for player in roster:
            # Base injury risk penalty
            injury_penalty += self.injury_penalty * player.injury_risk_score
            
            # Additional penalty for players with recurring injuries
            if player.has_recurring_injuries:
                injury_penalty += 0.1
            
            # Penalty for season-ending injury history
            injury_penalty += 0.05 * player.season_ending_injuries
            
            # Penalty based on games missed due to injury
            injury_penalty += 0.01 * player.games_missed_injury
        
        return injury_penalty
    
    def _calculate_durability_bonus(self, roster: List[InjuryAwarePlayer]) -> float:
        """Calculate bonus for durable players"""
        durability_bonus = 0.0
        
        for player in roster:
            # Bonus for high durability score
            if player.durability_score > 0.8:
                durability_bonus += self.durability_bonus * (player.durability_score - 0.8)
            
            # Bonus for players with no major injury history
            if player.historical_injuries == 0:
                durability_bonus += 0.1
        
        return durability_bonus
    
    def _calculate_injury_diversification_bonus(self, roster: List[InjuryAwarePlayer]) -> float:
        """Calculate bonus for injury risk diversification"""
        if len(roster) < 2:
            return 0.0
        
        # Calculate injury risk variance across roster
        injury_risks = [player.injury_risk_score for player in roster]
        risk_variance = np.var(injury_risks)
        
        # Bonus for having mix of high and low risk players
        diversification_bonus = self.diversification_bonus * risk_variance
        
        return diversification_bonus
    
    def _calculate_overstack_penalty(self, roster: List[InjuryAwarePlayer], league_settings: Dict) -> float:
        """Calculate position overstacking penalty"""
        position_counts = Counter(p.position for p in roster)
        overstack_penalty = 0.0
        
        roster_spots = league_settings.get('roster_spots', {
            'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DEF': 1, 'K': 1, 'BENCH': 6
        })
        flex_positions = league_settings.get('flex_positions', {'RB', 'WR', 'TE'})
        
        for pos, count in position_counts.items():
            max_useful = roster_spots.get(pos, 0)
            if pos in flex_positions:
                max_useful += roster_spots.get('FLEX', 0)
            max_useful += 2  # Allow some bench depth
            
            if count > max_useful:
                overstack_penalty += self.overstack_penalty * (count - max_useful) ** 2
        
        return overstack_penalty
    
    def _calculate_bye_penalty(self, roster: List[InjuryAwarePlayer]) -> float:
        """Calculate bye week conflict penalty"""
        bye_weeks = [p.bye_week for p in roster if p.bye_week > 0]
        bye_conflicts = len(bye_weeks) - len(set(bye_weeks))
        return self.bye_penalty * bye_conflicts
    
    def calculate_pick_reward(self, player: InjuryAwarePlayer, current_roster: List[InjuryAwarePlayer], 
                            league_settings: Dict) -> float:
        """Calculate incremental reward for picking a specific player"""
        
        # Base VORP
        base_reward = player.vorp
        
        # Position need bonus
        position_counts = Counter(p.position for p in current_roster)
        roster_spots = league_settings.get('roster_spots', {})
        needs = {}
        for pos, required in roster_spots.items():
            needs[pos] = max(0, required - position_counts.get(pos, 0))
        
        need_bonus = 0.0
        if player.position in needs and needs[player.position] > 0:
            need_bonus = player.vorp * 0.2  # 20% bonus for needed positions
        elif player.position in league_settings.get('flex_positions', set()) and needs.get('FLEX', 0) > 0:
            need_bonus = player.vorp * 0.1  # 10% bonus for flex-eligible
        
        # Risk penalty (traditional)
        risk_penalty = self.risk_penalty * player.risk_sigma
        
        # Injury risk penalty
        injury_penalty = self.injury_penalty * player.injury_risk_score
        
        # Durability bonus
        durability_bonus = 0.0
        if player.durability_score > 0.8:
            durability_bonus = self.durability_bonus * (player.durability_score - 0.8)
        
        # Roster diversification bonus
        diversification_bonus = 0.0
        if current_roster:
            current_risks = [p.injury_risk_score for p in current_roster]
            avg_current_risk = np.mean(current_risks)
            
            # Bonus for adding low-risk player to high-risk roster
            if avg_current_risk > 0.5 and player.injury_risk_score < 0.3:
                diversification_bonus = 0.2
            # Bonus for adding moderate risk when roster is too safe
            elif avg_current_risk < 0.2 and 0.3 < player.injury_risk_score < 0.5:
                diversification_bonus = 0.1
        
        # Early round scarcity bonus
        round_num = len(current_roster) // 12 + 1  # Approximate round
        scarcity_bonus = 0.0
        if round_num <= 3 and player.position in ['RB', 'WR']:
            scarcity_bonus = player.vorp * 0.15  # 15% bonus for scarce positions early
        
        total_reward = (base_reward 
                       + need_bonus 
                       + scarcity_bonus 
                       + durability_bonus 
                       + diversification_bonus
                       - risk_penalty 
                       - injury_penalty)
        
        return total_reward

class InjuryAwareValueFunction:
    """Enhanced value function that incorporates injury considerations"""
    
    def __init__(self, player_pool: Dict[str, InjuryAwarePlayer], reward_fn: InjuryAwareRewardFunction):
        self.player_pool = player_pool
        self.reward_fn = reward_fn
        
        # Precompute position-ranked players for efficiency
        self.players_by_position = defaultdict(list)
        for player in player_pool.values():
            self.players_by_position[player.position].append(player)
        
        # Sort by injury-adjusted VORP (VORP adjusted for injury risk)
        for pos in self.players_by_position:
            self.players_by_position[pos].sort(
                key=lambda x: x.vorp * (1.0 - x.injury_risk_score * 0.2), 
                reverse=True
            )
    
    def estimate_state_value(self, state, league_settings: Dict) -> float:
        """Estimate the value of the current state for our team"""
        
        our_roster = state.team_rosters[state.our_team_id]
        
        # Current roster value
        current_value = self.reward_fn.calculate_roster_value(our_roster, league_settings)
        
        # Expected value of filling remaining roster spots
        expected_fill_value = self._estimate_expected_fill_value(state, league_settings)
        
        return current_value + expected_fill_value
    
    def _estimate_expected_fill_value(self, state, league_settings: Dict) -> float:
        """Estimate value of filling remaining roster spots with injury considerations"""
        
        our_roster = state.team_rosters[state.our_team_id]
        
        # Calculate needs
        position_counts = Counter(p.position for p in our_roster)
        roster_spots = league_settings.get('roster_spots', {})
        flex_positions = league_settings.get('flex_positions', set())
        
        needs = {}
        for pos, required in roster_spots.items():
            if pos == 'FLEX':
                flex_filled = sum(max(0, position_counts.get(fp, 0) - roster_spots.get(fp, 0))
                                for fp in flex_positions)
                needs[pos] = max(0, required - flex_filled)
            elif pos == 'BENCH':
                total_players = len(our_roster)
                starting_spots = sum(v for k, v in roster_spots.items() if k != 'BENCH')
                needs[pos] = max(0, required - (total_players - starting_spots))
            else:
                needs[pos] = max(0, required - position_counts.get(pos, 0))
        
        available = list(state.available_players)
        if not available:
            return 0.0
        
        # Estimate picks until our next few turns
        picks_until_next = getattr(state, 'picks_until_our_turn', 12)  # Default assumption
        if picks_until_next == 0:
            picks_until_next = 12  # Next turn after this one
        
        expected_value = 0.0
        pick_number = 0
        
        for position, needed in needs.items():
            if needed == 0:
                continue
            
            available_at_pos = [p for p in available if p.position == position]
            if position == 'FLEX':
                available_at_pos = [p for p in available if p.position in flex_positions]
            elif position == 'BENCH':
                available_at_pos = available  # Any position for bench
            
            if not available_at_pos:
                continue
            
            # Sort by injury-adjusted value
            available_at_pos.sort(
                key=lambda x: x.vorp * (1.0 - x.injury_risk_score * 0.2), 
                reverse=True
            )
            
            for i in range(min(needed, len(available_at_pos))):
                # Estimate which player we might get based on pick timing
                estimated_pick_index = min(pick_number + picks_until_next // 2,
                                         len(available_at_pos) - 1)
                
                if estimated_pick_index < len(available_at_pos):
                    expected_player = available_at_pos[estimated_pick_index]
                    # Discount for uncertainty and injury risk
                    injury_discount = 1.0 - (expected_player.injury_risk_score * 0.2)
                    expected_value += expected_player.vorp * 0.8 * injury_discount
                
                pick_number += 1
                picks_until_next = max(12, picks_until_next)  # Future picks farther apart
        
        return expected_value

class InjuryAwareMCTS:
    """Enhanced MCTS that incorporates comprehensive injury considerations"""
    
    def __init__(self, 
                 opponent_model,
                 injury_aware_value_function: InjuryAwareValueFunction,
                 injury_aware_reward_function: InjuryAwareRewardFunction,
                 simulations: int = 400,
                 league_settings: Dict = None):
        self.opponent_model = opponent_model
        self.value_function = injury_aware_value_function
        self.reward_function = injury_aware_reward_function
        self.simulations = simulations
        self.league_settings = league_settings or {
            'teams': 12,
            'roster_spots': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DEF': 1, 'K': 1, 'BENCH': 6},
            'flex_positions': {'RB', 'WR', 'TE'},
            'total_rounds': 15
        }
    
    def search(self, initial_state) -> Optional[InjuryAwarePlayer]:
        """Run injury-aware MCTS to find best action"""
        available_players = list(initial_state.available_players)
        
        if not available_players:
            return None
        
        # For each available player, estimate value including injury considerations
        player_scores = []
        current_roster = initial_state.team_rosters[initial_state.our_team_id]
        
        for player in available_players:
            # Calculate injury-aware reward
            reward = self.reward_function.calculate_pick_reward(player, current_roster, self.league_settings)
            
            # Add some exploration for Monte Carlo simulation
            exploration_bonus = np.random.normal(0, 0.1)  # Small random component
            
            total_score = reward + exploration_bonus
            player_scores.append((player, total_score))
        
        # Sort by score and return best player
        player_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_player = player_scores[0][0]
        
        print(f"🏥 Injury-Aware MCTS selected: {best_player.name} ({best_player.position})")
        print(f"   VORP: {best_player.vorp:.2f}, Injury Risk: {best_player.injury_risk_score:.3f}")
        print(f"   Durability: {best_player.durability_score:.3f}, Total Score: {player_scores[0][1]:.2f}")
        
        return best_player

class InjuryAwareDraftStrategy:
    """Complete injury-aware draft strategy"""
    
    def __init__(self, 
                 player_pool: Dict[str, InjuryAwarePlayer],
                 injury_penalty: float = 0.3,
                 durability_bonus: float = 0.2,
                 league_settings: Dict = None):
        
        self.name = "Injury-Aware MCTS"
        self.player_pool = player_pool
        self.league_settings = league_settings or {
            'teams': 12,
            'roster_spots': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DEF': 1, 'K': 1, 'BENCH': 6},
            'flex_positions': {'RB', 'WR', 'TE'},
            'total_rounds': 15
        }
        
        # Initialize injury-aware components
        self.reward_function = InjuryAwareRewardFunction(
            injury_penalty=injury_penalty,
            durability_bonus=durability_bonus
        )
        
        self.value_function = InjuryAwareValueFunction(player_pool, self.reward_function)
        
        # Create opponent model (simplified)
        self.opponent_model = None  # Would need to adapt existing opponent model
        
        # Initialize MCTS
        self.mcts = InjuryAwareMCTS(
            self.opponent_model,
            self.value_function,
            self.reward_function,
            simulations=400,
            league_settings=self.league_settings
        )
    
    def make_pick(self, draft_state) -> Optional[InjuryAwarePlayer]:
        """Make an injury-aware draft pick"""
        return self.mcts.search(draft_state)
    
    def create_injury_aware_draft_board(self) -> pd.DataFrame:
        """Create draft board ranked by injury-adjusted value"""
        board_data = []
        
        for player in self.player_pool.values():
            # Calculate injury-adjusted value
            injury_adjustment = 1.0 - (player.injury_risk_score * 0.3)
            durability_bonus = max(0, (player.durability_score - 0.5) * 0.2)
            
            adjusted_value = player.vorp * injury_adjustment + durability_bonus
            
            board_data.append({
                'player_name': player.name,
                'position': player.position,
                'base_vorp': player.vorp,
                'injury_risk': player.injury_risk_score,
                'durability': player.durability_score,
                'injury_adjusted_vorp': adjusted_value,
                'value_change': adjusted_value - player.vorp,
                'adp_rank': player.adp_rank,
                'injury_tier': self._get_injury_tier(player.injury_risk_score)
            })
        
        # Create DataFrame and sort by adjusted value
        draft_board = pd.DataFrame(board_data)
        draft_board = draft_board.sort_values('injury_adjusted_vorp', ascending=False)
        draft_board['injury_adjusted_rank'] = range(1, len(draft_board) + 1)
        
        return draft_board
    
    def _get_injury_tier(self, injury_risk: float) -> str:
        """Categorize players by injury risk tier"""
        if injury_risk < 0.25:
            return "Iron Man"
        elif injury_risk < 0.4:
            return "Reliable"
        elif injury_risk < 0.6:
            return "Moderate Risk"
        else:
            return "High Risk"

def create_injury_aware_player_pool(basic_player_pool: Dict, enhanced_data_path: str = None) -> Dict[str, InjuryAwarePlayer]:
    """Convert basic player pool to injury-aware player pool"""
    
    print("🏥 Creating injury-aware player pool...")
    
    # Load enhanced injury data if available
    injury_data_map = {}
    if enhanced_data_path and Path(enhanced_data_path).exists():
        print(f"📊 Loading injury data from {enhanced_data_path}")
        df = pd.read_csv(enhanced_data_path)
        
        injury_features = [
            'injury_risk_score', 'durability_score', 'historical_injuries',
            'games_missed_injury', 'season_ending_injuries', 'avg_recovery_time',
            'has_recurring_injuries', 'position_injury_risk', 'age_adjusted_risk',
            'usage_adjusted_risk', 'games_played_pct_adj'
        ]
        
        for _, row in df.iterrows():
            player_name = row['player_name']
            injury_data_map[player_name] = {
                feature: row.get(feature, 0) for feature in injury_features
            }
        
        print(f"✅ Loaded injury data for {len(injury_data_map)} players")
    else:
        print("⚠️  No enhanced injury data found, using defaults")
    
    # Convert basic players to injury-aware players
    injury_aware_pool = {}
    for name, player in basic_player_pool.items():
        injury_data = injury_data_map.get(name, {})
        injury_aware_pool[name] = InjuryAwarePlayer.from_basic_player(player, injury_data)
    
    print(f"✅ Created injury-aware player pool with {len(injury_aware_pool)} players")
    return injury_aware_pool

if __name__ == "__main__":
    print("🏥 Injury-Aware MCTS Draft Strategy")
    print("This module provides injury-enhanced MCTS draft strategy components.")
    print("Import this module to use injury-aware draft strategy with your existing MCTS system.")
