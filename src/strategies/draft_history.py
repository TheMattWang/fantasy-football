"""
Draft History Analysis and Learning System
==========================================

This module provides sophisticated draft history analysis to improve
MCTS decision-making by learning from historical draft patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import json
from pathlib import Path

from ..core.player import Player, PlayerPool
from ..core.draft import DraftState, LeagueSettings


@dataclass
class DraftPick:
    """Record of a single draft pick"""
    round_num: int
    pick_in_round: int
    overall_pick: int
    team_id: int
    player_name: str
    position: str
    adp_rank: float
    vorp: float
    bye_week: int
    draft_id: str  # Unique identifier for the draft
    league_type: str = "12team_ppr"  # League format identifier


@dataclass
class DraftPattern:
    """Identified pattern in draft history"""
    pattern_type: str  # "position_run", "bye_avoidance", "early_qb", etc.
    rounds: List[int]  # Rounds where pattern occurs
    positions: List[str]  # Positions involved
    frequency: float  # How often this pattern occurs (0-1)
    strength: float  # How strong/predictable the pattern is
    description: str


class DraftHistoryAnalyzer:
    """
    Analyzes draft history to identify patterns and improve decision-making.
    
    Learns from:
    - Position run patterns (when teams draft same position consecutively)
    - Round-specific tendencies (QB runs in round 3-4, etc.)
    - Bye week avoidance behaviors
    - ADP deviation patterns
    - League-specific tendencies
    """
    
    def __init__(self, history_file: Optional[str] = None):
        self.draft_history: List[DraftPick] = []
        self.patterns: List[DraftPattern] = []
        self.position_trends: Dict[int, Dict[str, float]] = {}  # round -> position -> frequency
        self.bye_week_trends: Dict[int, float] = {}  # bye_week -> avoidance_factor
        self.adp_deviation_patterns: Dict[str, float] = {}  # position -> avg_deviation
        
        if history_file and Path(history_file).exists():
            self.load_history(history_file)
    
    def add_draft_pick(self, pick: DraftPick) -> None:
        """Add a draft pick to the history"""
        self.draft_history.append(pick)
    
    def add_completed_draft(self, draft_picks: List[DraftPick]) -> None:
        """Add an entire completed draft to the history"""
        self.draft_history.extend(draft_picks)
        self.analyze_patterns()
    
    def load_history(self, file_path: str) -> None:
        """Load draft history from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.draft_history = [
            DraftPick(**pick_data) for pick_data in data.get('picks', [])
        ]
        
        if 'patterns' in data:
            self.patterns = [
                DraftPattern(**pattern_data) for pattern_data in data['patterns']
            ]
        
        self.analyze_patterns()
    
    def save_history(self, file_path: str) -> None:
        """Save draft history to JSON file"""
        data = {
            'picks': [
                {
                    'round_num': pick.round_num,
                    'pick_in_round': pick.pick_in_round,
                    'overall_pick': pick.overall_pick,
                    'team_id': pick.team_id,
                    'player_name': pick.player_name,
                    'position': pick.position,
                    'adp_rank': pick.adp_rank,
                    'vorp': pick.vorp,
                    'bye_week': pick.bye_week,
                    'draft_id': pick.draft_id,
                    'league_type': pick.league_type
                }
                for pick in self.draft_history
            ],
            'patterns': [
                {
                    'pattern_type': pattern.pattern_type,
                    'rounds': pattern.rounds,
                    'positions': pattern.positions,
                    'frequency': pattern.frequency,
                    'strength': pattern.strength,
                    'description': pattern.description
                }
                for pattern in self.patterns
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def analyze_patterns(self) -> None:
        """Analyze draft history to identify patterns"""
        if len(self.draft_history) < 50:  # Need sufficient data
            return
        
        self._analyze_position_trends()
        self._analyze_position_runs()
        self._analyze_bye_week_patterns()
        self._analyze_adp_deviations()
        self._analyze_round_specific_patterns()
    
    def _analyze_position_trends(self) -> None:
        """Analyze position frequency by round"""
        round_position_counts = defaultdict(lambda: defaultdict(int))
        round_totals = defaultdict(int)
        
        for pick in self.draft_history:
            round_position_counts[pick.round_num][pick.position] += 1
            round_totals[pick.round_num] += 1
        
        # Calculate frequencies
        for round_num in round_position_counts:
            self.position_trends[round_num] = {}
            for position, count in round_position_counts[round_num].items():
                self.position_trends[round_num][position] = count / round_totals[round_num]
    
    def _analyze_position_runs(self) -> None:
        """Identify position run patterns"""
        # Group picks by draft and analyze consecutive position picks
        drafts = defaultdict(list)
        for pick in self.draft_history:
            drafts[pick.draft_id].append(pick)
        
        position_runs = defaultdict(int)
        total_opportunities = 0
        
        for draft_id, picks in drafts.items():
            # Sort picks by overall pick number
            picks.sort(key=lambda x: x.overall_pick)
            
            # Look for consecutive picks of same position
            for i in range(len(picks) - 2):  # Need at least 3 picks for a run
                positions = [picks[i + j].position for j in range(3)]
                
                if len(set(positions)) == 1:  # All same position
                    position = positions[0]
                    round_start = picks[i].round_num
                    position_runs[f"{position}_run_r{round_start}"] += 1
                
                total_opportunities += 1
        
        # Create position run patterns
        for run_type, count in position_runs.items():
            if count >= 3:  # Must occur at least 3 times to be a pattern
                frequency = count / total_opportunities
                position = run_type.split('_')[0]
                try:
                    round_num = int(run_type.split('r')[1])
                except (ValueError, IndexError):
                    continue  # Skip malformed run types
                
                pattern = DraftPattern(
                    pattern_type="position_run",
                    rounds=[round_num, round_num + 1],
                    positions=[position],
                    frequency=frequency,
                    strength=min(1.0, frequency * 10),  # Scale strength
                    description=f"{position} runs typically start in round {round_num}"
                )
                self.patterns.append(pattern)
    
    def _analyze_bye_week_patterns(self) -> None:
        """Analyze bye week avoidance patterns"""
        # Group by draft and look at bye week conflicts
        drafts = defaultdict(list)
        for pick in self.draft_history:
            drafts[pick.draft_id].append(pick)
        
        bye_week_conflicts = defaultdict(int)
        bye_week_totals = defaultdict(int)
        
        for draft_id, picks in drafts.items():
            # Group picks by team
            teams = defaultdict(list)
            for pick in picks:
                teams[pick.team_id].append(pick)
            
            # Analyze bye week conflicts per team
            for team_id, team_picks in teams.items():
                bye_weeks = [pick.bye_week for pick in team_picks if pick.bye_week > 0]
                
                for bye_week in bye_weeks:
                    bye_week_totals[bye_week] += 1
                    conflicts = bye_weeks.count(bye_week) - 1  # Subtract the player itself
                    bye_week_conflicts[bye_week] += conflicts
        
        # Calculate avoidance factors
        for bye_week, total in bye_week_totals.items():
            if total > 5:  # Need sufficient data
                conflict_rate = bye_week_conflicts[bye_week] / total
                # Higher conflict rate = lower avoidance factor
                self.bye_week_trends[bye_week] = 1.0 - conflict_rate
    
    def _analyze_adp_deviations(self) -> None:
        """Analyze how much picks deviate from ADP by position"""
        position_deviations = defaultdict(list)
        
        for pick in self.draft_history:
            if pick.adp_rank < 999:  # Valid ADP
                expected_pick = pick.adp_rank
                actual_pick = pick.overall_pick
                deviation = actual_pick - expected_pick
                position_deviations[pick.position].append(deviation)
        
        # Calculate average deviations
        for position, deviations in position_deviations.items():
            if len(deviations) >= 5:  # Need sufficient data
                self.adp_deviation_patterns[position] = np.mean(deviations)
    
    def _analyze_round_specific_patterns(self) -> None:
        """Identify round-specific patterns (e.g., QB run in round 4)"""
        round_spikes = {}
        
        for round_num in range(1, 16):
            if round_num in self.position_trends:
                for position, frequency in self.position_trends[round_num].items():
                    # Look for positions that spike above normal frequency
                    avg_frequency = np.mean([
                        self.position_trends.get(r, {}).get(position, 0)
                        for r in range(1, 16)
                    ])
                    
                    if frequency > avg_frequency * 2 and frequency > 0.2:  # Significant spike
                        pattern = DraftPattern(
                            pattern_type="round_spike",
                            rounds=[round_num],
                            positions=[position],
                            frequency=frequency,
                            strength=frequency / avg_frequency,
                            description=f"{position} spike in round {round_num} ({frequency:.1%} vs {avg_frequency:.1%} avg)"
                        )
                        self.patterns.append(pattern)
    
    def predict_position_run_probability(self, current_state: DraftState, position: str) -> float:
        """Predict probability of a position run starting"""
        round_num = current_state.current_round
        
        # Look for position run patterns starting in this round
        run_patterns = [
            p for p in self.patterns
            if p.pattern_type == "position_run" and
            position in p.positions and
            round_num in p.rounds
        ]
        
        if run_patterns:
            return max(p.frequency for p in run_patterns)
        
        # Default based on recent picks
        recent_picks = self._get_recent_picks(current_state, 3)
        position_count = sum(1 for pick in recent_picks if pick.position == position)
        
        # Higher recent activity = higher run probability
        return min(0.8, position_count * 0.3)
    
    def get_bye_week_avoidance_factor(self, bye_week: int) -> float:
        """Get avoidance factor for a specific bye week"""
        return self.bye_week_trends.get(bye_week, 0.5)  # Default to moderate avoidance
    
    def predict_round_position_preferences(self, round_num: int) -> Dict[str, float]:
        """Predict position preferences for a specific round"""
        if round_num in self.position_trends:
            return self.position_trends[round_num].copy()
        
        # Default uniform distribution
        return {'QB': 0.1, 'RB': 0.25, 'WR': 0.35, 'TE': 0.15, 'K': 0.05, 'DEF': 0.1}
    
    def get_adp_adjustment(self, position: str) -> float:
        """Get typical ADP adjustment for a position"""
        return self.adp_deviation_patterns.get(position, 0.0)
    
    def _get_recent_picks(self, current_state: DraftState, n_picks: int) -> List[DraftPick]:
        """Get the most recent N picks from current draft"""
        # This would need to be implemented based on how draft state tracks history
        # For now, return empty list
        return []
    
    def get_pattern_summary(self) -> str:
        """Get human-readable summary of identified patterns"""
        summary = ["Draft History Analysis Summary", "=" * 35, ""]
        
        if not self.patterns:
            summary.append("No significant patterns identified yet.")
            return "\n".join(summary)
        
        # Group patterns by type
        pattern_groups = defaultdict(list)
        for pattern in self.patterns:
            pattern_groups[pattern.pattern_type].append(pattern)
        
        for pattern_type, patterns in pattern_groups.items():
            summary.append(f"{pattern_type.replace('_', ' ').title()}:")
            for pattern in patterns:
                summary.append(f"  • {pattern.description}")
            summary.append("")
        
        # Round-specific trends
        if self.position_trends:
            summary.append("Position Trends by Round:")
            for round_num in sorted(self.position_trends.keys())[:5]:  # Show first 5 rounds
                trends = self.position_trends[round_num]
                top_positions = sorted(trends.items(), key=lambda x: x[1], reverse=True)[:3]
                summary.append(f"  Round {round_num}: {', '.join([f'{pos} ({freq:.1%})' for pos, freq in top_positions])}")
            summary.append("")
        
        # Bye week insights
        if self.bye_week_trends:
            avoided_weeks = sorted(
                [(week, factor) for week, factor in self.bye_week_trends.items()],
                key=lambda x: x[1], reverse=True
            )[:3]
            summary.append("Most Avoided Bye Weeks:")
            for week, factor in avoided_weeks:
                summary.append(f"  Week {week}: {factor:.1%} avoidance rate")
        
        return "\n".join(summary)


class HistoryAwareMCTS:
    """
    Enhanced MCTS that incorporates draft history analysis.
    
    Uses historical patterns to:
    - Predict opponent behavior more accurately
    - Adjust player valuations based on round trends
    - Account for bye week avoidance patterns
    - Anticipate position runs
    """
    
    def __init__(self, 
                 base_mcts,
                 history_analyzer: DraftHistoryAnalyzer,
                 history_weight: float = 0.3):
        """
        Initialize history-aware MCTS.
        
        Args:
            base_mcts: Base MCTS implementation
            history_analyzer: Trained draft history analyzer
            history_weight: Weight given to historical patterns (0.0-1.0)
        """
        self.base_mcts = base_mcts
        self.history_analyzer = history_analyzer
        self.history_weight = history_weight
    
    def search(self, initial_state: DraftState) -> Optional[Player]:
        """Enhanced MCTS search with historical pattern awareness"""
        
        # Get base MCTS recommendation
        base_pick = self.base_mcts.search(initial_state)
        
        if not base_pick or self.history_weight == 0:
            return base_pick
        
        # Enhance decision with historical patterns
        available_players = list(initial_state.available_players)
        enhanced_scores = []
        
        for player in available_players:
            base_score = self._get_base_score(player, initial_state)
            history_adjustment = self._calculate_history_adjustment(player, initial_state)
            
            final_score = base_score + (history_adjustment * self.history_weight)
            enhanced_scores.append((player, final_score))
        
        # Sort by enhanced score and return best
        enhanced_scores.sort(key=lambda x: x[1], reverse=True)
        
        history_pick = enhanced_scores[0][0]
        
        # Log the decision if it differs from base MCTS
        if history_pick != base_pick:
            print(f"📚 History-aware override: {history_pick.name} vs {base_pick.name}")
            print(f"   Historical patterns favor {history_pick.name}")
        
        return history_pick
    
    def _get_base_score(self, player: Player, state: DraftState) -> float:
        """Get base score for a player (simplified)"""
        # This would ideally interface with the base MCTS value function
        return player.vorp  # Simplified
    
    def _calculate_history_adjustment(self, player: Player, state: DraftState) -> float:
        """Calculate adjustment based on historical patterns"""
        adjustment = 0.0
        
        # 1. Round-specific position preferences
        round_preferences = self.history_analyzer.predict_round_position_preferences(state.current_round)
        position_boost = round_preferences.get(player.position, 0.1) - 0.1  # Boost above baseline
        adjustment += position_boost * 2.0  # Scale factor
        
        # 2. Position run probability
        run_prob = self.history_analyzer.predict_position_run_probability(state, player.position)
        if run_prob > 0.3:  # Significant run risk
            # If we think a run is starting, grab good players of that position
            adjustment += run_prob * player.vorp * 0.2
        
        # 3. Bye week avoidance
        if hasattr(player, 'bye_week') and player.bye_week > 0:
            avoidance_factor = self.history_analyzer.get_bye_week_avoidance_factor(player.bye_week)
            if avoidance_factor > 0.7:  # Highly avoided bye week
                # Others will avoid this player, so we might get value
                adjustment += (avoidance_factor - 0.5) * 1.0
        
        # 4. ADP deviation patterns
        adp_adjustment = self.history_analyzer.get_adp_adjustment(player.position)
        if hasattr(player, 'adp_rank') and player.adp_rank < 999:
            expected_pick = player.adp_rank + adp_adjustment
            current_pick = state.current_overall_pick
            
            # If player is "ahead of schedule" based on historical patterns
            if current_pick < expected_pick - 5:  # Significantly early
                adjustment += 0.5  # Modest boost for being ahead of typical draft position
        
        return adjustment


def create_draft_history_from_state(draft_state: DraftState, draft_id: str) -> List[DraftPick]:
    """Create DraftPick records from a completed DraftState"""
    picks = []
    
    # This would extract pick history from the draft state
    # Implementation depends on how draft history is stored in DraftState
    
    return picks


def simulate_and_learn(mcts_strategy, player_pool: PlayerPool, 
                      league_settings: LeagueSettings, 
                      history_analyzer: DraftHistoryAnalyzer,
                      n_simulations: int = 10) -> None:
    """
    Run draft simulations and learn from the results.
    
    Args:
        mcts_strategy: MCTS draft strategy
        player_pool: Available players
        league_settings: League configuration
        history_analyzer: History analyzer to update
        n_simulations: Number of drafts to simulate
    """
    
    print(f"🎓 Running {n_simulations} draft simulations to learn patterns...")
    
    for i in range(n_simulations):
        # Create mock draft
        draft_state = DraftState.create_mock_draft(player_pool, our_team_id=np.random.randint(1, 13))
        
        # Simulate complete draft
        draft_picks = []
        pick_number = 1
        
        while not draft_state.is_draft_complete() and draft_state.available_players:
            current_team = draft_state.current_team_picking
            
            # Make pick (simplified opponent model)
            available = list(draft_state.available_players)
            if available:
                # Simple ADP-based selection with noise
                weights = [1.0 / (getattr(p, 'adp_rank', 999) + 1) for p in available]
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                selected_player = np.random.choice(available, p=weights)
                
                # Record pick
                pick = DraftPick(
                    round_num=draft_state.current_round,
                    pick_in_round=draft_state.current_pick_in_round,
                    overall_pick=pick_number,
                    team_id=current_team,
                    player_name=selected_player.name,
                    position=selected_player.position,
                    adp_rank=getattr(selected_player, 'adp_rank', 999),
                    vorp=selected_player.vorp,
                    bye_week=getattr(selected_player, 'bye_week', 0),
                    draft_id=f"sim_{i}",
                    league_type="12team_ppr"
                )
                draft_picks.append(pick)
                
                # Update draft state
                draft_state.make_pick(selected_player)
                pick_number += 1
        
        # Add completed draft to history
        history_analyzer.add_completed_draft(draft_picks)
        
        if (i + 1) % 5 == 0:
            print(f"   Completed {i + 1}/{n_simulations} simulations")
    
    print(f"✅ Learning complete! {len(history_analyzer.patterns)} patterns identified")
    print(f"\n{history_analyzer.get_pattern_summary()}")


if __name__ == "__main__":
    # Example usage
    analyzer = DraftHistoryAnalyzer()
    print("📚 Draft History Analyzer created")
    print("Add draft data with analyzer.add_completed_draft() to start learning patterns")
