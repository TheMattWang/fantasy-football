"""
Data loading utilities for fantasy football.

This module provides functions to load player data, injury data,
and other necessary files for draft strategy.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core.player import Player, PlayerPool
from ..core.draft import LeagueSettings


def load_default_data(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load default fantasy football data.
    
    Args:
        data_dir: Directory containing data files (defaults to data/)
        
    Returns:
        Dictionary containing loaded data
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    
    data = {}
    
    # Load draft board
    draft_board_path = data_dir / "raw" / "draft_board.csv"
    if draft_board_path.exists():
        data['draft_board'] = pd.read_csv(draft_board_path)
    
    # Load ADP rankings
    adp_path = data_dir / "raw" / "FantasyPros_2025_Overall_ADP_Rankings.csv"
    if adp_path.exists():
        data['adp_rankings'] = pd.read_csv(adp_path)
    
    # Load rookie data
    rookie_path = data_dir / "raw" / "rookie_data_clean.csv"
    if rookie_path.exists():
        data['rookie_data'] = pd.read_csv(rookie_path)
    
    # Load rookie model
    model_path = data_dir / "models" / "rookie_regressor.pkl"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            data['rookie_model'] = pickle.load(f)
    
    return data


def load_injury_enhanced_data(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load injury-enhanced fantasy football data.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary containing loaded data including injury features
    """
    # Load default data first
    data = load_default_data(data_dir)
    
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"
    else:
        data_dir = Path(data_dir)
    
    # Load injury-enhanced data
    injury_path = data_dir / "processed" / "injury_enhanced_demo.csv"
    if injury_path.exists():
        data['injury_enhanced_data'] = pd.read_csv(injury_path)
    else:
        # Try root directory (legacy)
        legacy_path = Path(__file__).parent.parent.parent / "injury_enhanced_demo.csv"
        if legacy_path.exists():
            data['injury_enhanced_data'] = pd.read_csv(legacy_path)
    
    return data


def load_player_pool(source: str = "draft_board", 
                    include_injury_data: bool = False,
                    data_dir: Optional[str] = None) -> PlayerPool:
    """
    Load a PlayerPool from data files.
    
    Args:
        source: Data source ('draft_board', 'adp_rankings', or 'injury_enhanced')
        include_injury_data: Whether to include injury features
        data_dir: Directory containing data files
        
    Returns:
        PlayerPool instance
    """
    data = load_default_data(data_dir) if not include_injury_data else load_injury_enhanced_data(data_dir)
    
    if source == "draft_board" and 'draft_board' in data:
        df = data['draft_board']
    elif source == "adp_rankings" and 'adp_rankings' in data:
        df = data['adp_rankings']
    elif source == "injury_enhanced" and 'injury_enhanced_data' in data:
        df = data['injury_enhanced_data']
    else:
        raise ValueError(f"Data source '{source}' not found in loaded data")
    
    # Convert DataFrame to PlayerPool
    return _dataframe_to_player_pool(df)


def _dataframe_to_player_pool(df: pd.DataFrame) -> PlayerPool:
    """
    Convert a DataFrame to a PlayerPool.
    
    Args:
        df: DataFrame with player data
        
    Returns:
        PlayerPool instance
    """
    players = []
    
    for _, row in df.iterrows():
        # Extract basic player info
        name = row.get('player_name', row.get('name', ''))
        position = row.get('position', row.get('pos', ''))
        team = row.get('team', '')
        
        # Extract VORP/value
        vorp = row.get('vorp', row.get('ppg', row.get('proj_ppg', 0.0)))
        
        # Extract ADP
        adp_rank = row.get('adp_rank', row.get('adp', row.get('rank', 999.0)))
        
        # Create projections
        projections = {
            'ppg': row.get('ppg', row.get('proj_ppg', vorp)),
            'total_points': row.get('total_points', vorp * 16),
            'games_played': row.get('games_played', 16.0)
        }
        
        # Create metadata
        metadata = {}
        meta_fields = ['bye_week', 'risk_sigma', 'is_rookie', 'age']
        for field in meta_fields:
            if field in row and pd.notna(row[field]):
                metadata[field] = row[field]
        
        # Create injury data if present
        injury_data = None
        injury_fields = [
            'injury_risk_score', 'durability_score', 'historical_injuries',
            'games_missed_injury', 'season_ending_injuries', 'avg_recovery_time',
            'has_recurring_injuries', 'position_injury_risk', 'age_adjusted_risk',
            'usage_adjusted_risk', 'games_played_pct_adj'
        ]
        
        injury_dict = {}
        for field in injury_fields:
            if field in row and pd.notna(row[field]):
                injury_dict[field] = row[field]
        
        if injury_dict:
            injury_data = injury_dict
        
        # Create player
        player = Player(
            name=name,
            position=position,
            team=team,
            vorp=vorp,
            adp_rank=adp_rank,
            projections=projections,
            injury_data=injury_data,
            metadata=metadata
        )
        
        players.append(player)
    
    return PlayerPool(players)


def save_player_pool(player_pool: PlayerPool, filename: str, data_dir: Optional[str] = None) -> None:
    """
    Save a PlayerPool to CSV file.
    
    Args:
        player_pool: PlayerPool to save
        filename: Output filename
        data_dir: Directory to save to (defaults to data/processed/)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    else:
        data_dir = Path(data_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame and save
    df = player_pool.to_dataframe()
    output_path = data_dir / filename
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved PlayerPool to {output_path}")


def get_default_league_settings() -> LeagueSettings:
    """Get default league settings for 12-team PPR league."""
    return LeagueSettings(
        teams=12,
        roster_spots={
            'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 
            'FLEX': 1, 'DEF': 1, 'K': 1, 'BENCH': 6
        },
        flex_positions={'RB', 'WR', 'TE'},
        total_rounds=15,
        snake_draft=True
    )


def create_sample_player_pool(n_players: int = 100) -> PlayerPool:
    """
    Create a sample PlayerPool for testing.
    
    Args:
        n_players: Number of players to create
        
    Returns:
        Sample PlayerPool
    """
    import numpy as np
    
    np.random.seed(42)  # For reproducible results
    
    positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
    position_weights = [0.1, 0.25, 0.35, 0.15, 0.05, 0.1]
    
    players = []
    for i in range(n_players):
        position = np.random.choice(positions, p=position_weights)
        
        # Generate realistic stats based on position
        if position == 'QB':
            base_ppg = np.random.gamma(3, 5) + 10  # 10-30 range
        elif position in ['RB', 'WR']:
            base_ppg = np.random.gamma(2, 4) + 3   # 3-20 range
        elif position == 'TE':
            base_ppg = np.random.gamma(2, 3) + 2   # 2-15 range
        else:  # K, DEF
            base_ppg = np.random.gamma(2, 2) + 4   # 4-12 range
        
        player = Player(
            name=f"Player_{i+1:03d}",
            position=position,
            team=f"Team_{(i % 32) + 1}",
            vorp=base_ppg,
            adp_rank=i + 1,
            projections={'ppg': base_ppg},
            metadata={
                'risk_sigma': np.random.uniform(0.1, 0.5),
                'bye_week': np.random.randint(4, 15)
            }
        )
        
        players.append(player)
    
    return PlayerPool(players)
