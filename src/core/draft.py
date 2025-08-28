"""
Draft state and league settings for fantasy football draft strategy.

This module defines the core draft mechanics including draft state tracking,
league configuration, and draft flow management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
from .player import Player, PlayerPool


@dataclass
class LeagueSettings:
    """
    Fantasy league configuration and rules.
    
    Defines roster requirements, league size, and draft format.
    """
    teams: int = 12
    roster_spots: Dict[str, int] = field(default_factory=lambda: {
        'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DEF': 1, 'K': 1, 'BENCH': 6
    })
    flex_positions: Set[str] = field(default_factory=lambda: {'RB', 'WR', 'TE'})
    total_rounds: int = 15
    snake_draft: bool = True
    
    @property
    def total_roster_size(self) -> int:
        """Total number of roster spots per team."""
        return sum(self.roster_spots.values())
    
    @property
    def starting_positions(self) -> Dict[str, int]:
        """Starting positions (excluding bench)."""
        return {pos: count for pos, count in self.roster_spots.items() if pos != 'BENCH'}
    
    def get_position_need(self, current_roster: List[Player], position: str) -> int:
        """
        Calculate how many more players are needed at a position.
        
        Args:
            current_roster: Current team roster
            position: Position to check
            
        Returns:
            Number of additional players needed at that position
        """
        current_count = sum(1 for p in current_roster if p.position == position)
        required = self.roster_spots.get(position, 0)
        
        # Handle FLEX positions
        if position == 'FLEX':
            flex_filled = 0
            for flex_pos in self.flex_positions:
                pos_count = sum(1 for p in current_roster if p.position == flex_pos)
                pos_required = self.roster_spots.get(flex_pos, 0)
                flex_filled += max(0, pos_count - pos_required)
            return max(0, required - flex_filled)
        
        return max(0, required - current_count)
    
    def is_roster_complete(self, roster: List[Player]) -> bool:
        """Check if a roster meets all position requirements."""
        return len(roster) >= self.total_roster_size
    
    def get_roster_needs(self, current_roster: List[Player]) -> Dict[str, int]:
        """Get all position needs for a roster."""
        return {
            pos: self.get_position_need(current_roster, pos)
            for pos in self.roster_spots.keys()
        }


@dataclass 
class DraftState:
    """
    Current state of a fantasy football draft.
    
    Tracks picks, available players, team rosters, and draft position.
    """
    league: LeagueSettings
    available_players: Set[Player] = field(default_factory=set)
    team_rosters: Dict[int, List[Player]] = field(default_factory=lambda: defaultdict(list))
    our_team_id: int = 1
    current_round: int = 1
    current_pick_in_round: int = 1
    draft_history: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def current_overall_pick(self) -> int:
        """Current overall pick number."""
        return (self.current_round - 1) * self.league.teams + self.current_pick_in_round
    
    @property
    def current_team_picking(self) -> int:
        """ID of team currently picking."""
        if self.league.snake_draft and self.current_round % 2 == 0:
            # Snake draft - reverse order on even rounds
            return self.league.teams - self.current_pick_in_round + 1
        else:
            return self.current_pick_in_round
    
    @property
    def is_our_turn(self) -> bool:
        """Whether it's our team's turn to pick."""
        return self.current_team_picking == self.our_team_id
    
    @property
    def picks_until_our_turn(self) -> int:
        """Number of picks until our next turn."""
        if self.is_our_turn:
            return 0
        
        # Calculate picks until we pick again
        current_team = self.current_team_picking
        picks = 0
        round_num = self.current_round
        pick_in_round = self.current_pick_in_round
        
        while True:
            # Check if it's our turn
            if self.league.snake_draft and round_num % 2 == 0:
                team_picking = self.league.teams - pick_in_round + 1
            else:
                team_picking = pick_in_round
            
            if team_picking == self.our_team_id:
                break
            
            picks += 1
            pick_in_round += 1
            
            # Handle end of round
            if pick_in_round > self.league.teams:
                round_num += 1
                pick_in_round = 1
            
            # Safety check
            if picks > 100:  # Prevent infinite loop
                break
        
        return picks
    
    def get_our_roster(self) -> List[Player]:
        """Get our team's current roster."""
        return self.team_rosters[self.our_team_id]
    
    def get_our_roster_needs(self) -> Dict[str, int]:
        """Get our team's current position needs."""
        return self.league.get_roster_needs(self.get_our_roster())
    
    def is_draft_complete(self) -> bool:
        """Check if the draft is complete."""
        return (self.current_round > self.league.total_rounds or 
                len(self.available_players) == 0)
    
    def get_valid_actions(self) -> List[int]:
        """Get indices of valid player selections."""
        return list(range(len(self.available_players)))
    
    def make_pick(self, player: Player, team_id: Optional[int] = None) -> None:
        """
        Execute a draft pick.
        
        Args:
            player: Player being drafted
            team_id: Team making the pick (defaults to current team)
        """
        if team_id is None:
            team_id = self.current_team_picking
        
        # Validate pick
        if player not in self.available_players:
            raise ValueError(f"Player {player.name} is not available")
        
        # Remove player from available pool
        self.available_players.discard(player)
        
        # Add to team roster
        self.team_rosters[team_id].append(player)
        
        # Record pick in history
        pick_record = {
            'round': self.current_round,
            'pick_in_round': self.current_pick_in_round,
            'overall_pick': self.current_overall_pick,
            'team_id': team_id,
            'player': player,
            'remaining_players': len(self.available_players)
        }
        self.draft_history.append(pick_record)
        
        # Advance draft state
        self._advance_pick()
    
    def _advance_pick(self) -> None:
        """Advance to the next pick."""
        self.current_pick_in_round += 1
        
        # Check if round is complete
        if self.current_pick_in_round > self.league.teams:
            self.current_round += 1
            self.current_pick_in_round = 1
    
    def simulate_pick(self, player: Player, team_id: Optional[int] = None) -> 'DraftState':
        """
        Create a copy of the draft state with a simulated pick.
        
        Args:
            player: Player to simulate picking
            team_id: Team making the pick
            
        Returns:
            New DraftState with the pick made
        """
        # Create deep copy
        new_state = self.copy()
        new_state.make_pick(player, team_id)
        return new_state
    
    def copy(self) -> 'DraftState':
        """Create a deep copy of the draft state."""
        return DraftState(
            league=self.league,
            available_players=self.available_players.copy(),
            team_rosters={
                team_id: roster.copy() 
                for team_id, roster in self.team_rosters.items()
            },
            our_team_id=self.our_team_id,
            current_round=self.current_round,
            current_pick_in_round=self.current_pick_in_round,
            draft_history=self.draft_history.copy()
        )
    
    def get_pick_summary(self) -> str:
        """Get human-readable summary of current pick."""
        return (f"Round {self.current_round}, Pick {self.current_pick_in_round} "
                f"(Overall: {self.current_overall_pick}) - "
                f"Team {self.current_team_picking} picking")
    
    def get_draft_summary(self) -> Dict[str, Any]:
        """Get summary of draft progress."""
        our_roster = self.get_our_roster()
        our_needs = self.get_our_roster_needs()
        
        return {
            'current_pick': self.get_pick_summary(),
            'is_our_turn': self.is_our_turn,
            'picks_until_our_turn': self.picks_until_our_turn,
            'our_roster_size': len(our_roster),
            'our_position_counts': {
                pos: sum(1 for p in our_roster if p.position == pos)
                for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
            },
            'our_needs': our_needs,
            'available_players': len(self.available_players),
            'total_picks_made': len(self.draft_history),
            'rounds_remaining': self.league.total_rounds - self.current_round + 1
        }
    
    @classmethod
    def create_mock_draft(cls, player_pool: PlayerPool, 
                         our_team_id: int = 1,
                         league_settings: Optional[LeagueSettings] = None) -> 'DraftState':
        """
        Create a mock draft state for testing.
        
        Args:
            player_pool: Available players
            our_team_id: Our team's draft position
            league_settings: League configuration
            
        Returns:
            Initialized DraftState ready for drafting
        """
        if league_settings is None:
            league_settings = LeagueSettings()
        
        return cls(
            league=league_settings,
            available_players=set(player_pool.get_all_players()),
            our_team_id=our_team_id
        )
    
    def __str__(self):
        """String representation of draft state."""
        return self.get_pick_summary()
