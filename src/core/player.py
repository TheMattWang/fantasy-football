"""
Player and PlayerPool classes for fantasy football.

This module defines the core Player class and PlayerPool collection
with all necessary attributes and methods for draft strategy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import pandas as pd
import numpy as np


@dataclass
class Player:
    """
    Represents a fantasy football player with all relevant attributes.
    
    Attributes:
        name: Player name
        position: Position (QB, RB, WR, TE, K, DEF)
        team: NFL team
        vorp: Value Over Replacement Player
        adp_rank: Average Draft Position ranking
        projections: Projected stats (ppg, total_points, etc.)
        injury_data: Optional injury risk data
        metadata: Additional player information
    """
    name: str
    position: str
    team: str = ""
    vorp: float = 0.0
    adp_rank: float = 999.0
    
    # Projections
    projections: Dict[str, float] = field(default_factory=dict)
    
    # Injury data (optional)
    injury_data: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default projections if not provided."""
        if not self.projections:
            self.projections = {
                'ppg': 0.0,
                'total_points': 0.0,
                'games_played': 16.0
            }
    
    @property
    def proj_ppg(self) -> float:
        """Projected points per game."""
        return self.projections.get('ppg', 0.0)
    
    @property
    def bye_week(self) -> int:
        """Player's bye week."""
        return self.metadata.get('bye_week', 0)
    
    @property
    def risk_sigma(self) -> float:
        """Player uncertainty/risk (for rookies, injury-prone players)."""
        return self.metadata.get('risk_sigma', 0.0)
    
    @property
    def is_rookie(self) -> bool:
        """Whether this is a rookie player."""
        return self.metadata.get('is_rookie', False)
    
    @property
    def injury_risk_score(self) -> float:
        """Injury risk score (0-1 scale, higher = more risky)."""
        if self.injury_data:
            return self.injury_data.get('injury_risk_score', 0.25)
        return 0.25
    
    @property
    def durability_score(self) -> float:
        """Durability score (0-1 scale, higher = more durable)."""
        if self.injury_data:
            return self.injury_data.get('durability_score', 0.75)
        return 0.75
    
    def add_injury_data(self, injury_data: Dict[str, Any]) -> None:
        """Add injury risk data to the player."""
        self.injury_data = injury_data
    
    def get_injury_tier(self) -> str:
        """Get injury risk tier as human-readable string."""
        risk = self.injury_risk_score
        if risk < 0.25:
            return "Iron Man"
        elif risk < 0.4:
            return "Reliable" 
        elif risk < 0.6:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    def __hash__(self):
        """Hash based on player name for use in sets."""
        return hash(self.name)
    
    def __str__(self):
        """String representation of player."""
        return f"{self.name} ({self.position}) - VORP: {self.vorp:.2f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert player to dictionary for serialization."""
        return {
            'name': self.name,
            'position': self.position,
            'team': self.team,
            'vorp': self.vorp,
            'adp_rank': self.adp_rank,
            'projections': self.projections,
            'injury_data': self.injury_data,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Player':
        """Create player from dictionary."""
        return cls(
            name=data['name'],
            position=data['position'],
            team=data.get('team', ''),
            vorp=data.get('vorp', 0.0),
            adp_rank=data.get('adp_rank', 999.0),
            projections=data.get('projections', {}),
            injury_data=data.get('injury_data'),
            metadata=data.get('metadata', {})
        )


class PlayerPool:
    """
    Collection of players with filtering and search capabilities.
    
    Provides efficient access to players by position, ranking, and other criteria.
    """
    
    def __init__(self, players: List[Player]):
        """
        Initialize player pool.
        
        Args:
            players: List of Player objects
        """
        self.players = {p.name: p for p in players}
        self._position_cache = {}
        self._rebuild_cache()
    
    def _rebuild_cache(self):
        """Rebuild internal caches for efficient filtering."""
        self._position_cache = {}
        for player in self.players.values():
            pos = player.position
            if pos not in self._position_cache:
                self._position_cache[pos] = []
            self._position_cache[pos].append(player)
        
        # Sort each position by VORP
        for pos in self._position_cache:
            self._position_cache[pos].sort(key=lambda p: p.vorp, reverse=True)
    
    def __len__(self) -> int:
        """Number of players in pool."""
        return len(self.players)
    
    def __iter__(self):
        """Iterate over all players."""
        return iter(self.players.values())
    
    def __getitem__(self, name: str) -> Player:
        """Get player by name."""
        return self.players[name]
    
    def __contains__(self, player_or_name) -> bool:
        """Check if player is in pool."""
        if isinstance(player_or_name, str):
            return player_or_name in self.players
        return player_or_name.name in self.players
    
    def get_all_players(self) -> List[Player]:
        """Get all players as a list."""
        return list(self.players.values())
    
    def filter_by_position(self, position: str) -> List[Player]:
        """
        Get all players at a specific position.
        
        Args:
            position: Position to filter by (QB, RB, WR, TE, K, DEF)
            
        Returns:
            List of players at that position, sorted by VORP
        """
        return self._position_cache.get(position, []).copy()
    
    def filter_by_positions(self, positions: List[str]) -> List[Player]:
        """
        Get players from multiple positions.
        
        Args:
            positions: List of positions to include
            
        Returns:
            List of players from those positions, sorted by VORP
        """
        players = []
        for pos in positions:
            players.extend(self._position_cache.get(pos, []))
        
        players.sort(key=lambda p: p.vorp, reverse=True)
        return players
    
    def get_top_players(self, n: int, position: Optional[str] = None) -> List[Player]:
        """
        Get top N players overall or by position.
        
        Args:
            n: Number of players to return
            position: Optional position filter
            
        Returns:
            Top N players sorted by VORP
        """
        if position:
            candidates = self.filter_by_position(position)
        else:
            candidates = sorted(self.players.values(), key=lambda p: p.vorp, reverse=True)
        
        return candidates[:n]
    
    def get_available_positions(self) -> List[str]:
        """Get list of all positions with players."""
        return list(self._position_cache.keys())
    
    def filter_by_injury_tier(self, tier: str) -> List[Player]:
        """
        Filter players by injury risk tier.
        
        Args:
            tier: Injury tier ('Iron Man', 'Reliable', 'Moderate Risk', 'High Risk')
            
        Returns:
            Players in that injury tier
        """
        return [p for p in self.players.values() if p.get_injury_tier() == tier]
    
    def filter_by_injury_risk(self, max_risk: float) -> List[Player]:
        """
        Filter players by maximum injury risk.
        
        Args:
            max_risk: Maximum injury risk score (0-1)
            
        Returns:
            Players with injury risk <= max_risk
        """
        return [p for p in self.players.values() if p.injury_risk_score <= max_risk]
    
    def add_player(self, player: Player):
        """Add a player to the pool."""
        self.players[player.name] = player
        self._rebuild_cache()
    
    def remove_player(self, player_or_name):
        """Remove a player from the pool."""
        if isinstance(player_or_name, str):
            name = player_or_name
        else:
            name = player_or_name.name
        
        if name in self.players:
            del self.players[name]
            self._rebuild_cache()
    
    def add_injury_data(self, injury_data: Dict[str, Dict[str, Any]]):
        """
        Add injury data to players in the pool.
        
        Args:
            injury_data: Dictionary mapping player names to injury data
        """
        for name, data in injury_data.items():
            if name in self.players:
                self.players[name].add_injury_data(data)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert player pool to pandas DataFrame."""
        data = []
        for player in self.players.values():
            row = {
                'name': player.name,
                'position': player.position,
                'team': player.team,
                'vorp': player.vorp,
                'adp_rank': player.adp_rank,
                'proj_ppg': player.proj_ppg,
                'injury_risk_score': player.injury_risk_score,
                'durability_score': player.durability_score,
                'injury_tier': player.get_injury_tier()
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'PlayerPool':
        """
        Create PlayerPool from pandas DataFrame.
        
        Args:
            df: DataFrame with player data
            
        Returns:
            PlayerPool instance
        """
        players = []
        for _, row in df.iterrows():
            # Basic player data
            player_data = {
                'name': row['name'],
                'position': row['position'],
                'team': row.get('team', ''),
                'vorp': row.get('vorp', 0.0),
                'adp_rank': row.get('adp_rank', 999.0),
                'projections': {
                    'ppg': row.get('proj_ppg', 0.0)
                },
                'metadata': {}
            }
            
            # Add injury data if present
            injury_cols = ['injury_risk_score', 'durability_score', 'historical_injuries']
            injury_data = {}
            for col in injury_cols:
                if col in row and pd.notna(row[col]):
                    injury_data[col] = row[col]
            
            if injury_data:
                player_data['injury_data'] = injury_data
            
            # Add metadata
            meta_cols = ['bye_week', 'risk_sigma', 'is_rookie']
            for col in meta_cols:
                if col in row and pd.notna(row[col]):
                    player_data['metadata'][col] = row[col]
            
            players.append(Player.from_dict(player_data))
        
        return cls(players)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the player pool."""
        return {
            'total_players': len(self.players),
            'positions': {pos: len(players) for pos, players in self._position_cache.items()},
            'avg_vorp': np.mean([p.vorp for p in self.players.values()]),
            'avg_injury_risk': np.mean([p.injury_risk_score for p in self.players.values()]),
            'injury_tiers': {
                tier: len(self.filter_by_injury_tier(tier))
                for tier in ['Iron Man', 'Reliable', 'Moderate Risk', 'High Risk']
            }
        }
    
    def __str__(self):
        """String representation of player pool."""
        summary = self.get_summary()
        return f"PlayerPool({summary['total_players']} players, {len(summary['positions'])} positions)"
