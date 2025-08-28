import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from injury_enhancement import InjuryDataCollector
from injury_aware_model import InjuryAwareFantasyModel
from injury_data_sources import ComprehensiveInjuryDatabase

class InjuryAwareDraftStrategy:
    """
    Enhanced draft strategy that incorporates comprehensive injury risk analysis
    """
    
    def __init__(self, base_strategy, injury_weight: float = 0.3):
        """
        Initialize injury-aware draft strategy
        
        Args:
            base_strategy: Existing draft strategy (MCTS, VORP, etc.)
            injury_weight: Weight given to injury considerations (0.0 to 1.0)
        """
        self.base_strategy = base_strategy
        self.injury_weight = injury_weight
        self.injury_collector = InjuryDataCollector()
        self.injury_database = ComprehensiveInjuryDatabase()
        self.injury_adjustments = {}
        
    def calculate_injury_adjusted_value(self, player, current_roster: List = None) -> float:
        """
        Calculate injury-adjusted player value
        """
        # Get base value from original strategy
        base_value = getattr(player, 'vorp', 0.0)
        
        # Calculate injury risk
        player_data = {
            'name': player.name,
            'position': player.position,
            'age': getattr(player, 'age', 25),
            'usage_rate': self._estimate_usage_rate(player),
            'games_played': getattr(player, 'games_played', 16)
        }
        
        injury_risk = self.injury_collector.calculate_injury_risk_score(player_data)
        
        # Apply injury adjustments
        injury_adjustment_factor = 1.0 - (injury_risk * self.injury_weight)
        
        # Additional adjustments based on roster construction
        roster_risk_adjustment = self._calculate_roster_risk_adjustment(player, current_roster)
        
        # Final adjusted value
        adjusted_value = base_value * injury_adjustment_factor * roster_risk_adjustment
        
        # Store adjustment details for analysis
        self.injury_adjustments[player.name] = {
            'base_value': base_value,
            'injury_risk': injury_risk,
            'injury_adjustment_factor': injury_adjustment_factor,
            'roster_risk_adjustment': roster_risk_adjustment,
            'final_adjusted_value': adjusted_value
        }
        
        return adjusted_value
    
    def _estimate_usage_rate(self, player) -> float:
        """Estimate player usage rate for injury risk calculation"""
        # This is a simplified estimation - you'd want more sophisticated logic
        position_usage = {
            'RB': 0.7,  # High usage
            'WR': 0.5,  # Medium usage  
            'TE': 0.4,  # Medium-low usage
            'QB': 0.6,  # Medium-high usage
            'K': 0.1,   # Low usage
            'DEF': 0.2  # Low usage
        }
        
        base_usage = position_usage.get(player.position, 0.5)
        
        # Adjust based on VORP (higher VORP players likely get more usage)
        vorp_adjustment = min(0.3, getattr(player, 'vorp', 0) * 0.05)
        
        return min(1.0, base_usage + vorp_adjustment)
    
    def _calculate_roster_risk_adjustment(self, player, current_roster: List) -> float:
        """
        Calculate roster-level injury risk adjustments
        """
        if not current_roster:
            return 1.0
        
        # Count players at same position
        same_position_count = sum(1 for p in current_roster if p.position == player.position)
        
        # Calculate position injury risk concentration
        position_injury_risks = []
        for roster_player in current_roster:
            if roster_player.position == player.position:
                risk = getattr(roster_player, 'injury_risk_score', 0.3)
                position_injury_risks.append(risk)
        
        avg_position_risk = np.mean(position_injury_risks) if position_injury_risks else 0.3
        
        # Diversification bonus for low-risk players when position has high average risk
        if avg_position_risk > 0.5 and getattr(player, 'injury_risk_score', 0.3) < 0.3:
            return 1.1  # 10% bonus for adding a durable player to risky position
        
        # Penalty for adding another high-risk player to already risky position
        elif avg_position_risk > 0.4 and getattr(player, 'injury_risk_score', 0.3) > 0.5:
            return 0.9  # 10% penalty
        
        return 1.0
    
    def make_pick(self, draft_state) -> int:
        """
        Make injury-aware draft pick
        """
        available_players = list(draft_state.available_players)
        
        if not available_players:
            return None
        
        # Calculate injury-adjusted values for all available players
        player_values = []
        current_roster = draft_state.team_rosters.get(draft_state.our_team_id, [])
        
        for player in available_players:
            adjusted_value = self.calculate_injury_adjusted_value(player, current_roster)
            player_values.append((player, adjusted_value))
        
        # Sort by adjusted value
        player_values.sort(key=lambda x: x[1], reverse=True)
        
        # Select best player
        best_player = player_values[0][0]
        
        # Log the decision
        adjustment_info = self.injury_adjustments.get(best_player.name, {})
        print(f"🏥 Injury-Aware Pick: {best_player.name} ({best_player.position})")
        print(f"   Base Value: {adjustment_info.get('base_value', 0):.2f}")
        print(f"   Injury Risk: {adjustment_info.get('injury_risk', 0):.3f}")
        print(f"   Adjusted Value: {adjustment_info.get('final_adjusted_value', 0):.2f}")
        
        return best_player
    
    def create_injury_draft_board(self, player_pool: Dict) -> pd.DataFrame:
        """
        Create injury-adjusted draft board
        """
        print("📋 Creating injury-adjusted draft board...")
        
        draft_board_data = []
        
        for player_name, player in player_pool.items():
            # Calculate injury-adjusted value
            adjusted_value = self.calculate_injury_adjusted_value(player)
            adjustment_info = self.injury_adjustments.get(player_name, {})
            
            draft_board_entry = {
                'player_name': player_name,
                'position': player.position,
                'base_vorp': getattr(player, 'vorp', 0),
                'injury_risk': adjustment_info.get('injury_risk', 0),
                'injury_adjusted_vorp': adjusted_value,
                'value_change': adjusted_value - getattr(player, 'vorp', 0),
                'adp_rank': getattr(player, 'adp_rank', 999),
                'injury_tier': self._get_injury_tier(adjustment_info.get('injury_risk', 0))
            }
            
            draft_board_data.append(draft_board_entry)
        
        # Create DataFrame and sort by adjusted value
        draft_board = pd.DataFrame(draft_board_data)
        draft_board = draft_board.sort_values('injury_adjusted_vorp', ascending=False)
        draft_board['injury_adjusted_rank'] = range(1, len(draft_board) + 1)
        
        return draft_board
    
    def _get_injury_tier(self, injury_risk: float) -> str:
        """Categorize players by injury tier"""
        if injury_risk < 0.25:
            return "Iron Man"
        elif injury_risk < 0.4:
            return "Reliable"
        elif injury_risk < 0.6:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    def analyze_draft_board_changes(self, original_board: pd.DataFrame, injury_board: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze how injury considerations change draft board rankings
        """
        print("📊 Analyzing draft board changes due to injury considerations...")
        
        # Merge boards to compare rankings
        comparison = original_board[['player_name', 'position']].copy()
        comparison['original_rank'] = range(1, len(comparison) + 1)
        
        injury_ranks = injury_board[['player_name', 'injury_adjusted_rank', 'injury_risk', 'injury_tier']].copy()
        
        comparison = comparison.merge(injury_ranks, on='player_name', how='left')
        comparison['rank_change'] = comparison['original_rank'] - comparison['injury_adjusted_rank']
        
        # Sort by biggest rank changes
        comparison = comparison.sort_values('rank_change', ascending=False)
        
        print("\n🔄 Biggest Rank Changes Due to Injury Considerations:")
        print("=" * 70)
        
        # Show biggest movers up
        movers_up = comparison[comparison['rank_change'] > 5].head(10)
        if not movers_up.empty:
            print("\n📈 Biggest Movers UP (Lower Injury Risk):")
            for _, row in movers_up.iterrows():
                print(f"   {row['player_name']:25s} ({row['position']:2s}) - "
                      f"Moved up {row['rank_change']:2.0f} spots - "
                      f"Risk: {row['injury_risk']:.3f} ({row['injury_tier']})")
        
        # Show biggest movers down  
        movers_down = comparison[comparison['rank_change'] < -5].tail(10)
        if not movers_down.empty:
            print("\n📉 Biggest Movers DOWN (Higher Injury Risk):")
            for _, row in movers_down.iterrows():
                print(f"   {row['player_name']:25s} ({row['position']:2s}) - "
                      f"Moved down {abs(row['rank_change']):2.0f} spots - "
                      f"Risk: {row['injury_risk']:.3f} ({row['injury_tier']})")
        
        return comparison
    
    def create_injury_strategy_report(self, draft_results: Dict) -> Dict:
        """
        Create comprehensive report on injury-aware draft strategy performance
        """
        print("📋 Creating injury strategy performance report...")
        
        report = {
            'strategy_name': f"Injury-Aware {self.base_strategy.__class__.__name__}",
            'injury_weight': self.injury_weight,
            'drafted_players': draft_results.get('our_picks', []),
            'roster_injury_analysis': {},
            'risk_distribution': {},
            'recommendations': []
        }
        
        # Analyze drafted roster injury profile
        if report['drafted_players']:
            injury_risks = []
            injury_tiers = []
            
            for pick in report['drafted_players']:
                player_name = pick.get('player', {}).name if hasattr(pick.get('player', {}), 'name') else str(pick.get('player', ''))
                adjustment_info = self.injury_adjustments.get(player_name, {})
                risk = adjustment_info.get('injury_risk', 0.3)
                injury_risks.append(risk)
                injury_tiers.append(self._get_injury_tier(risk))
            
            report['roster_injury_analysis'] = {
                'average_injury_risk': np.mean(injury_risks),
                'max_injury_risk': np.max(injury_risks),
                'min_injury_risk': np.min(injury_risks),
                'risk_standard_deviation': np.std(injury_risks),
                'tier_distribution': pd.Series(injury_tiers).value_counts().to_dict()
            }
            
            # Generate recommendations
            avg_risk = report['roster_injury_analysis']['average_injury_risk']
            if avg_risk > 0.5:
                report['recommendations'].append("⚠️  High overall injury risk - consider handcuffs")
            elif avg_risk < 0.3:
                report['recommendations'].append("✅ Low injury risk roster - good durability")
            else:
                report['recommendations'].append("📊 Moderate injury risk - balanced approach")
        
        return report

def integrate_injury_awareness_into_existing_strategy():
    """
    Demonstrate how to integrate injury awareness into existing draft strategy
    """
    print("🔄 Integrating injury awareness into existing draft strategy...")
    
    # This would integrate with your existing MCTS or other strategy
    # For demonstration, we'll create a mock integration
    
    print("📊 Loading existing player data...")
    
    # Load your existing data
    try:
        player_df = pd.read_csv('/Users/mattwang/Documents/fantasy/fantasy-football/rookie_data_clean.csv')
        print(f"✅ Loaded {len(player_df)} players")
    except FileNotFoundError:
        print("⚠️  Could not find rookie_data_clean.csv - using simulated data")
        # Create sample data for demonstration
        player_df = pd.DataFrame({
            'player_name': ['Player_A', 'Player_B', 'Player_C'],
            'position': ['RB', 'WR', 'QB'],
            'vorp': [10.0, 8.5, 7.2],
            'adp_rank': [5, 12, 18]
        })
    
    # Create mock player pool
    class MockPlayer:
        def __init__(self, name, position, vorp, adp_rank):
            self.name = name
            self.position = position
            self.vorp = vorp
            self.adp_rank = adp_rank
    
    player_pool = {}
    for _, row in player_df.head(20).iterrows():  # Use first 20 for demo
        player = MockPlayer(
            row['player_name'], 
            row['position'], 
            row.get('ppg', 5.0),  # Use ppg if available, otherwise default
            row.get('adp_rank', 999)
        )
        player_pool[player.name] = player
    
    # Create original draft board
    original_board = pd.DataFrame([{
        'player_name': p.name,
        'position': p.position,
        'vorp': p.vorp,
        'adp_rank': p.adp_rank
    } for p in player_pool.values()]).sort_values('vorp', ascending=False)
    
    # Create injury-aware strategy
    class MockBaseStrategy:
        def make_pick(self, state):
            return None
    
    base_strategy = MockBaseStrategy()
    injury_strategy = InjuryAwareDraftStrategy(base_strategy, injury_weight=0.3)
    
    # Create injury-adjusted draft board
    injury_board = injury_strategy.create_injury_draft_board(player_pool)
    
    # Analyze changes
    changes = injury_strategy.analyze_draft_board_changes(original_board, injury_board)
    
    # Save results
    injury_board.to_csv('/Users/mattwang/Documents/fantasy/fantasy-football/injury_adjusted_draft_board.csv', index=False)
    changes.to_csv('/Users/mattwang/Documents/fantasy/fantasy-football/draft_board_changes.csv', index=False)
    
    print("\n✅ Injury integration complete!")
    print("📁 Files created:")
    print("   • injury_adjusted_draft_board.csv")
    print("   • draft_board_changes.csv")
    
    return injury_strategy, injury_board, changes

if __name__ == "__main__":
    # Run the integration
    strategy, board, changes = integrate_injury_awareness_into_existing_strategy()
    
    print("\n🏥 Injury-Aware Draft Strategy Integration Complete!")
    print(f"📊 Strategy uses {strategy.injury_weight*100:.0f}% injury weighting")
    print(f"📋 {len(board)} players in injury-adjusted draft board")
