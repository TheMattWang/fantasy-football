"""
Fantasy football scoring systems.

This module defines different scoring systems for fantasy football
including PPR, standard, and custom scoring configurations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass


class ScoringSystem(ABC):
    """Abstract base class for fantasy football scoring systems."""
    
    @abstractmethod
    def calculate_points(self, stats: Dict[str, Any]) -> float:
        """Calculate fantasy points for a player's stats."""
        pass
    
    @abstractmethod
    def get_scoring_config(self) -> Dict[str, float]:
        """Get the scoring configuration."""
        pass


@dataclass
class PPRScoring(ScoringSystem):
    """
    PPR (Point Per Reception) scoring system.
    
    Standard PPR scoring with 1 point per reception.
    """
    
    # Passing scoring
    pass_yards_per_point: float = 25.0  # 1 point per 25 yards
    pass_td: float = 4.0
    interception: float = -2.0
    
    # Rushing scoring  
    rush_yards_per_point: float = 10.0  # 1 point per 10 yards
    rush_td: float = 6.0
    
    # Receiving scoring
    rec_yards_per_point: float = 10.0  # 1 point per 10 yards
    reception: float = 1.0  # PPR bonus
    rec_td: float = 6.0
    
    # Other scoring
    fumble_lost: float = -2.0
    two_point_conversion: float = 2.0
    
    def calculate_points(self, stats: Dict[str, Any]) -> float:
        """Calculate PPR fantasy points for player stats."""
        points = 0.0
        
        # Passing points
        if 'pass_yards' in stats:
            points += stats['pass_yards'] / self.pass_yards_per_point
        if 'pass_td' in stats:
            points += stats['pass_td'] * self.pass_td
        if 'interceptions' in stats:
            points += stats['interceptions'] * self.interception
        
        # Rushing points
        if 'rush_yards' in stats:
            points += stats['rush_yards'] / self.rush_yards_per_point
        if 'rush_td' in stats:
            points += stats['rush_td'] * self.rush_td
        
        # Receiving points
        if 'rec_yards' in stats:
            points += stats['rec_yards'] / self.rec_yards_per_point
        if 'receptions' in stats:
            points += stats['receptions'] * self.reception
        if 'rec_td' in stats:
            points += stats['rec_td'] * self.rec_td
        
        # Other scoring
        if 'fumbles_lost' in stats:
            points += stats['fumbles_lost'] * self.fumble_lost
        if 'two_point_conversions' in stats:
            points += stats['two_point_conversions'] * self.two_point_conversion
        
        return points
    
    def get_scoring_config(self) -> Dict[str, float]:
        """Get PPR scoring configuration."""
        return {
            'pass_yards_per_point': self.pass_yards_per_point,
            'pass_td': self.pass_td,
            'interception': self.interception,
            'rush_yards_per_point': self.rush_yards_per_point,
            'rush_td': self.rush_td,
            'rec_yards_per_point': self.rec_yards_per_point,
            'reception': self.reception,
            'rec_td': self.rec_td,
            'fumble_lost': self.fumble_lost,
            'two_point_conversion': self.two_point_conversion
        }


@dataclass 
class StandardScoring(ScoringSystem):
    """
    Standard (non-PPR) scoring system.
    
    Traditional scoring without reception bonuses.
    """
    
    # Passing scoring
    pass_yards_per_point: float = 25.0
    pass_td: float = 4.0
    interception: float = -2.0
    
    # Rushing scoring
    rush_yards_per_point: float = 10.0
    rush_td: float = 6.0
    
    # Receiving scoring (no PPR bonus)
    rec_yards_per_point: float = 10.0
    reception: float = 0.0  # No PPR bonus
    rec_td: float = 6.0
    
    # Other scoring
    fumble_lost: float = -2.0
    two_point_conversion: float = 2.0
    
    def calculate_points(self, stats: Dict[str, Any]) -> float:
        """Calculate standard fantasy points for player stats."""
        points = 0.0
        
        # Passing points
        if 'pass_yards' in stats:
            points += stats['pass_yards'] / self.pass_yards_per_point
        if 'pass_td' in stats:
            points += stats['pass_td'] * self.pass_td
        if 'interceptions' in stats:
            points += stats['interceptions'] * self.interception
        
        # Rushing points
        if 'rush_yards' in stats:
            points += stats['rush_yards'] / self.rush_yards_per_point
        if 'rush_td' in stats:
            points += stats['rush_td'] * self.rush_td
        
        # Receiving points (no PPR)
        if 'rec_yards' in stats:
            points += stats['rec_yards'] / self.rec_yards_per_point
        if 'rec_td' in stats:
            points += stats['rec_td'] * self.rec_td
        
        # Other scoring
        if 'fumbles_lost' in stats:
            points += stats['fumbles_lost'] * self.fumble_lost
        if 'two_point_conversions' in stats:
            points += stats['two_point_conversions'] * self.two_point_conversion
        
        return points
    
    def get_scoring_config(self) -> Dict[str, float]:
        """Get standard scoring configuration."""
        return {
            'pass_yards_per_point': self.pass_yards_per_point,
            'pass_td': self.pass_td,
            'interception': self.interception,
            'rush_yards_per_point': self.rush_yards_per_point,
            'rush_td': self.rush_td,
            'rec_yards_per_point': self.rec_yards_per_point,
            'reception': self.reception,
            'rec_td': self.rec_td,
            'fumble_lost': self.fumble_lost,
            'two_point_conversion': self.two_point_conversion
        }


@dataclass
class HalfPPRScoring(ScoringSystem):
    """
    Half PPR scoring system.
    
    0.5 points per reception.
    """
    
    # Passing scoring
    pass_yards_per_point: float = 25.0
    pass_td: float = 4.0
    interception: float = -2.0
    
    # Rushing scoring
    rush_yards_per_point: float = 10.0
    rush_td: float = 6.0
    
    # Receiving scoring
    rec_yards_per_point: float = 10.0
    reception: float = 0.5  # Half PPR
    rec_td: float = 6.0
    
    # Other scoring
    fumble_lost: float = -2.0
    two_point_conversion: float = 2.0
    
    def calculate_points(self, stats: Dict[str, Any]) -> float:
        """Calculate half-PPR fantasy points for player stats."""
        points = 0.0
        
        # Use same logic as PPR but with 0.5 reception bonus
        ppr_scorer = PPRScoring()
        ppr_scorer.reception = self.reception
        
        return ppr_scorer.calculate_points(stats)
    
    def get_scoring_config(self) -> Dict[str, float]:
        """Get half-PPR scoring configuration."""
        config = PPRScoring().get_scoring_config()
        config['reception'] = self.reception
        return config


class CustomScoring(ScoringSystem):
    """
    Custom scoring system with user-defined settings.
    
    Allows for completely customizable scoring rules.
    """
    
    def __init__(self, scoring_config: Dict[str, float]):
        """
        Initialize custom scoring system.
        
        Args:
            scoring_config: Dictionary of stat -> points mapping
        """
        self.config = scoring_config
    
    def calculate_points(self, stats: Dict[str, Any]) -> float:
        """Calculate fantasy points using custom scoring."""
        points = 0.0
        
        for stat, value in stats.items():
            if stat in self.config:
                if 'per_point' in stat:
                    # Handle per-yard stats (e.g., pass_yards_per_point)
                    base_stat = stat.replace('_per_point', '')
                    if base_stat in stats:
                        points += stats[base_stat] / self.config[stat]
                else:
                    # Handle direct stat scoring
                    points += value * self.config[stat]
        
        return points
    
    def get_scoring_config(self) -> Dict[str, float]:
        """Get custom scoring configuration."""
        return self.config.copy()


# Utility functions
def get_default_scoring_system(scoring_type: str = 'ppr') -> ScoringSystem:
    """
    Get a default scoring system by type.
    
    Args:
        scoring_type: Type of scoring ('ppr', 'standard', 'half_ppr')
        
    Returns:
        ScoringSystem instance
    """
    scoring_type = scoring_type.lower()
    
    if scoring_type == 'ppr':
        return PPRScoring()
    elif scoring_type == 'standard':
        return StandardScoring() 
    elif scoring_type in ['half_ppr', 'half-ppr', '0.5ppr']:
        return HalfPPRScoring()
    else:
        raise ValueError(f"Unknown scoring type: {scoring_type}")


def calculate_vorp(player_points: float, replacement_points: float) -> float:
    """
    Calculate Value Over Replacement Player (VORP).
    
    Args:
        player_points: Player's projected fantasy points
        replacement_points: Replacement level points for position
        
    Returns:
        VORP value
    """
    return player_points - replacement_points
