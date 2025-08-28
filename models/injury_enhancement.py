import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class InjuryDataCollector:
    """
    Comprehensive injury data collection and enhancement system for fantasy football
    """
    
    def __init__(self):
        self.injury_data = {}
        self.historical_injuries = {}
        self.injury_patterns = {}
        
        # Common injury types and their typical recovery times (weeks)
        self.injury_recovery_times = {
            'hamstring': {'min': 2, 'max': 6, 'avg': 3.5, 'recurring_risk': 0.3},
            'ankle': {'min': 1, 'max': 8, 'avg': 3.0, 'recurring_risk': 0.25},
            'knee': {'min': 2, 'max': 16, 'avg': 6.0, 'recurring_risk': 0.4},
            'shoulder': {'min': 1, 'max': 12, 'avg': 4.0, 'recurring_risk': 0.2},
            'concussion': {'min': 1, 'max': 6, 'avg': 2.0, 'recurring_risk': 0.5},
            'back': {'min': 1, 'max': 12, 'avg': 4.5, 'recurring_risk': 0.35},
            'groin': {'min': 1, 'max': 4, 'avg': 2.5, 'recurring_risk': 0.3},
            'quad': {'min': 1, 'max': 4, 'avg': 2.0, 'recurring_risk': 0.25},
            'calf': {'min': 1, 'max': 6, 'avg': 3.0, 'recurring_risk': 0.3},
            'hand': {'min': 1, 'max': 8, 'avg': 3.0, 'recurring_risk': 0.1},
            'foot': {'min': 2, 'max': 10, 'avg': 4.0, 'recurring_risk': 0.2},
            'rib': {'min': 2, 'max': 6, 'avg': 3.5, 'recurring_risk': 0.15}
        }
        
        # Position-specific injury risk factors
        self.position_injury_risk = {
            'RB': {
                'high_risk': ['knee', 'ankle', 'hamstring', 'shoulder'],
                'base_risk': 0.35,  # 35% chance of missing games due to injury
                'workload_factor': 1.2  # Higher workload = higher risk
            },
            'WR': {
                'high_risk': ['hamstring', 'ankle', 'concussion', 'knee'],
                'base_risk': 0.25,
                'workload_factor': 1.0
            },
            'QB': {
                'high_risk': ['shoulder', 'knee', 'concussion', 'ankle'],
                'base_risk': 0.20,
                'workload_factor': 0.8
            },
            'TE': {
                'high_risk': ['knee', 'ankle', 'shoulder', 'back'],
                'base_risk': 0.30,
                'workload_factor': 1.1
            },
            'K': {
                'high_risk': ['groin', 'quad', 'back'],
                'base_risk': 0.10,
                'workload_factor': 0.5
            },
            'DEF': {
                'high_risk': ['various'],
                'base_risk': 0.15,
                'workload_factor': 0.7
            }
        }

    def collect_historical_injury_data(self, player_name: str, seasons: List[int] = None) -> Dict:
        """
        Collect historical injury data for a player
        Note: This is a template - you'll need to implement actual data sources
        """
        if seasons is None:
            seasons = list(range(2018, 2025))
        
        # Template for injury data structure
        injury_history = {
            'player_name': player_name,
            'total_injuries': 0,
            'games_missed_injury': 0,
            'injury_types': [],
            'recurring_injuries': [],
            'season_ending_injuries': 0,
            'avg_recovery_time': 0,
            'injury_timeline': [],
            'durability_score': 1.0  # 1.0 = very durable, 0.0 = very injury prone
        }
        
        # TODO: Implement actual data collection from:
        # 1. ESPN injury reports API
        # 2. NFL.com injury reports
        # 3. Pro Football Reference injury data
        # 4. Team injury reports
        
        print(f"📊 Collecting injury data for {player_name} (placeholder implementation)")
        
        # For now, return simulated data based on position
        return self._simulate_injury_history(player_name)
    
    def _simulate_injury_history(self, player_name: str) -> Dict:
        """
        Simulate injury history for demonstration purposes
        Replace with actual data collection
        """
        # Basic simulation based on name patterns (for demo)
        np.random.seed(hash(player_name) % 2**32)
        
        # Simulate some injury history
        total_injuries = np.random.poisson(2)  # Average 2 injuries over career
        games_missed = np.random.poisson(total_injuries * 2.5)  # Avg 2.5 games per injury
        
        injury_types = np.random.choice(
            list(self.injury_recovery_times.keys()), 
            size=min(total_injuries, 3), 
            replace=False
        ).tolist()
        
        # Calculate durability score (inverse of injury frequency)
        games_played_estimate = 50  # Assume ~3 seasons of data
        durability_score = max(0.1, 1.0 - (games_missed / games_played_estimate))
        
        return {
            'player_name': player_name,
            'total_injuries': total_injuries,
            'games_missed_injury': games_missed,
            'injury_types': injury_types,
            'recurring_injuries': injury_types[:1] if total_injuries > 2 else [],
            'season_ending_injuries': 1 if total_injuries > 3 else 0,
            'avg_recovery_time': np.mean([self.injury_recovery_times[inj]['avg'] for inj in injury_types]) if injury_types else 0,
            'injury_timeline': [],  # Detailed timeline would go here
            'durability_score': durability_score
        }

    def calculate_injury_risk_score(self, player_data: Dict) -> float:
        """
        Calculate comprehensive injury risk score for a player
        
        Args:
            player_data: Dict containing player info (name, position, age, usage, etc.)
            
        Returns:
            Risk score from 0.0 (low risk) to 1.0 (high risk)
        """
        position = player_data.get('position', 'WR')
        age = player_data.get('age', 25)
        usage = player_data.get('usage_rate', 0.5)  # 0-1 scale
        
        # Base risk by position
        base_risk = self.position_injury_risk.get(position, {}).get('base_risk', 0.25)
        
        # Age factor (risk increases with age)
        age_factor = 1.0 + (max(0, age - 25) * 0.02)  # 2% increase per year over 25
        
        # Usage factor (higher usage = higher risk)
        workload_multiplier = self.position_injury_risk.get(position, {}).get('workload_factor', 1.0)
        usage_factor = 1.0 + (usage * workload_multiplier * 0.3)
        
        # Historical injury factor
        injury_history = self.collect_historical_injury_data(player_data.get('name', ''))
        durability_factor = 2.0 - injury_history['durability_score']  # Invert durability score
        
        # Combine all factors
        risk_score = base_risk * age_factor * usage_factor * durability_factor
        
        # Cap at 1.0
        return min(1.0, risk_score)

    def enhance_player_data_with_injuries(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance existing player dataframe with comprehensive injury features
        """
        print("🏥 Enhancing player data with injury features...")
        
        enhanced_df = player_df.copy()
        
        # Add injury-related features
        injury_features = []
        
        for idx, row in enhanced_df.iterrows():
            player_data = {
                'name': row.get('player_name', ''),
                'position': row.get('position', 'WR'),
                'age': row.get('age', 25),
                'usage_rate': self._calculate_usage_rate(row),
                'games_played': row.get('games_played', 16)
            }
            
            # Calculate injury risk score
            injury_risk = self.calculate_injury_risk_score(player_data)
            
            # Get historical injury data
            injury_history = self.collect_historical_injury_data(player_data['name'])
            
            # Calculate additional injury features
            injury_feature_dict = {
                'injury_risk_score': injury_risk,
                'durability_score': injury_history['durability_score'],
                'historical_injuries': injury_history['total_injuries'],
                'games_missed_injury': injury_history['games_missed_injury'],
                'season_ending_injuries': injury_history['season_ending_injuries'],
                'avg_recovery_time': injury_history['avg_recovery_time'],
                'has_recurring_injuries': len(injury_history['recurring_injuries']) > 0,
                'position_injury_risk': self.position_injury_risk.get(player_data['position'], {}).get('base_risk', 0.25),
                'age_adjusted_risk': injury_risk * (1.0 + max(0, player_data['age'] - 25) * 0.02),
                'usage_adjusted_risk': injury_risk * (1.0 + player_data['usage_rate'] * 0.3),
                'games_played_pct_adj': self._adjust_games_played_for_injuries(row, injury_history)
            }
            
            injury_features.append(injury_feature_dict)
        
        # Add all injury features to dataframe
        injury_df = pd.DataFrame(injury_features)
        enhanced_df = pd.concat([enhanced_df, injury_df], axis=1)
        
        print(f"✅ Added {len(injury_df.columns)} injury-related features")
        print(f"New features: {list(injury_df.columns)}")
        
        return enhanced_df
    
    def _calculate_usage_rate(self, player_row) -> float:
        """Calculate usage rate based on player statistics"""
        position = player_row.get('position', 'WR')
        
        if position == 'RB':
            # For RBs, use carries + targets
            carries = player_row.get('rush_att', 0)
            targets = player_row.get('targets', 0)
            games = player_row.get('games_played', 1)
            usage_per_game = (carries + targets) / games if games > 0 else 0
            return min(1.0, usage_per_game / 25)  # Normalize to 25 touches per game
            
        elif position in ['WR', 'TE']:
            # For WRs/TEs, use targets
            targets = player_row.get('targets', 0)
            games = player_row.get('games_played', 1)
            targets_per_game = targets / games if games > 0 else 0
            return min(1.0, targets_per_game / 12)  # Normalize to 12 targets per game
            
        elif position == 'QB':
            # For QBs, use pass attempts
            attempts = player_row.get('pass_att', 0)
            games = player_row.get('games_played', 1)
            attempts_per_game = attempts / games if games > 0 else 0
            return min(1.0, attempts_per_game / 35)  # Normalize to 35 attempts per game
            
        else:
            return 0.5  # Default usage rate
    
    def _adjust_games_played_for_injuries(self, player_row, injury_history) -> float:
        """Adjust games played percentage accounting for injury history"""
        base_games_pct = player_row.get('games_played_pct', 0.8)
        
        # Adjust based on injury history
        injury_adjustment = injury_history['games_missed_injury'] * 0.01  # 1% per game missed
        
        # Factor in durability score
        durability_adjustment = (1.0 - injury_history['durability_score']) * 0.1
        
        adjusted_pct = base_games_pct - injury_adjustment - durability_adjustment
        
        return max(0.0, min(1.0, adjusted_pct))

    def create_injury_aware_features(self) -> List[str]:
        """Return list of injury-related feature names for model training"""
        return [
            'injury_risk_score',
            'durability_score', 
            'historical_injuries',
            'games_missed_injury',
            'season_ending_injuries',
            'avg_recovery_time',
            'has_recurring_injuries',
            'position_injury_risk',
            'age_adjusted_risk',
            'usage_adjusted_risk',
            'games_played_pct_adj'
        ]

# Example usage and integration
def enhance_existing_model_with_injuries(rookie_data_path: str, output_path: str):
    """
    Enhance existing rookie model with comprehensive injury data
    """
    print("🏥 Starting injury enhancement pipeline...")
    
    # Load existing data
    df = pd.read_csv(rookie_data_path)
    print(f"📊 Loaded {len(df)} players from {rookie_data_path}")
    
    # Initialize injury collector
    injury_collector = InjuryDataCollector()
    
    # Enhance with injury features
    enhanced_df = injury_collector.enhance_player_data_with_injuries(df)
    
    # Save enhanced data
    enhanced_df.to_csv(output_path, index=False)
    print(f"💾 Saved enhanced data to {output_path}")
    
    # Print summary statistics
    print("\n📈 Injury Enhancement Summary:")
    print(f"   • Average injury risk score: {enhanced_df['injury_risk_score'].mean():.3f}")
    print(f"   • Average durability score: {enhanced_df['durability_score'].mean():.3f}")
    print(f"   • Players with recurring injuries: {enhanced_df['has_recurring_injuries'].sum()}")
    print(f"   • Average games missed due to injury: {enhanced_df['games_missed_injury'].mean():.1f}")
    
    # Show correlation with performance
    if 'ppg' in enhanced_df.columns:
        corr_with_ppg = enhanced_df[['ppg'] + injury_collector.create_injury_aware_features()].corr()['ppg'].sort_values()
        print(f"\n🔗 Correlations with PPG:")
        for feature, corr in corr_with_ppg.items():
            if feature != 'ppg':
                print(f"   • {feature}: {corr:.3f}")
    
    return enhanced_df

if __name__ == "__main__":
    # Example usage
    input_file = "rookie_data_clean.csv"
    output_file = "rookie_data_injury_enhanced.csv"
    
    enhanced_data = enhance_existing_model_with_injuries(input_file, output_file)
    print("✅ Injury enhancement complete!")
