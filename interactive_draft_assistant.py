#!/usr/bin/env python3
"""
Interactive Fantasy Football Draft Assistant
===========================================

Real-time draft interface that uses trained MCTS models to provide intelligent
pick recommendations as you input other teams' selections.

Features:
- Live draft tracking with pick-by-pick updates
- MCTS-powered recommendations using trained models
- Visual draft board and roster management
- Bye week analysis and position need tracking
- Injury risk assessment for recommendations
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.assertions import validate_board, validate_players

try:
    from src.core.player import Player, PlayerPool
    from src.core.draft import DraftState, LeagueSettings
    from src.core.scoring import calculate_vorp
    from src.utils.data_loader import load_default_data
except ImportError:
    print("⚠️  Warning: Could not import from src modules. Using basic functionality.")
    
    # Fallback Player class
    @dataclass
    class Player:
        name: str
        position: str
        team: str
        vorp: float
        metadata: Dict = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}

def _player_rank_key(player):
    """Deterministic ordering for players: best VORP first, ties broken by ADP then name.

    available_players is a set and Player.__hash__ is hash(name), which Python
    randomizes per process. Without an explicit tie-break, any scoring bug that
    flattens scores makes the top recommendation depend on the hash seed.
    """
    return (-getattr(player, 'vorp', 0.0),
            getattr(player, 'adp_rank', 999.0),
            player.name)


def _scored_rank_key(player, score):
    """Deterministic ordering for (player, score) pairs. Highest score first."""
    return (-score, getattr(player, 'adp_rank', 999.0), player.name)


# Try to import PyTorch for model loading
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available. Model recommendations will use fallback logic.")
    TORCH_AVAILABLE = False

    # The network classes below subclass nn.Module, and several signatures
    # annotate torch.Tensor -- both are evaluated at import time. Without
    # stand-ins the module fails to import entirely instead of degrading to the
    # VORP fallback. Draft day must not depend on torch being installed.
    class _Missing:
        """Stands in for torch/torch.nn; raises only if actually used."""

        class Module:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("PyTorch is not installed")

        Tensor = object

        def __getattr__(self, name):
            raise RuntimeError(f"PyTorch is not installed (needed for {name!r})")

    torch = nn = _Missing()


class MCTSValueNetwork(nn.Module):
    """Neural network for MCTS value function"""
    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class MCTSPolicyNetwork(nn.Module):
    """Neural network for MCTS policy function"""
    def __init__(self, input_size: int, hidden_size: int = 512, output_size: int = 400):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)


class TrainedMCTSModel:
    """Wrapper for trained MCTS models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cpu')  # Use CPU for inference
        self.value_network = None
        self.policy_network = None
        self.state_size = 390  # Based on the fixed size
        
        if TORCH_AVAILABLE and os.path.exists(model_path):
            self._load_model()
        else:
            print(f"⚠️  Model not found at {model_path} or PyTorch unavailable. Using fallback.")
    
    def _load_model(self):
        """Load the trained model"""
        try:
            # Load the model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize networks
            self.value_network = MCTSValueNetwork(self.state_size, 256, 1)
            self.policy_network = MCTSPolicyNetwork(self.state_size, 512, 400)
            
            # Load state dicts if available
            if 'value_network_state_dict' in checkpoint:
                self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
                self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            else:
                print("⚠️  Model checkpoint format not recognized. Using initialized networks.")
            
            self.value_network.eval()
            self.policy_network.eval()
            
            print(f"✅ Loaded trained MCTS model from {self.model_path}")
            
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            self.value_network = None
            self.policy_network = None
    
    def encode_draft_state(self, draft_state, player_pool) -> Optional[torch.Tensor]:
        """Encode draft state for neural network input"""
        if not self.value_network:
            return None
            
        try:
            # Basic state features
            our_roster = [p for p in draft_state.team_rosters.get(draft_state.our_team_id, [])]
            
            basic_features = [
                draft_state.current_round / 15.0,
                draft_state.current_pick_in_round / 12.0,
                len(our_roster) / 15.0,
                len(draft_state.available_players) / len(player_pool) if player_pool else 0.0,
            ]
            
            # Position counts
            position_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'K': 0, 'DEF': 0}
            for player in our_roster:
                if player.position in position_counts:
                    position_counts[player.position] += 1
            
            position_features = [count / 6.0 for count in position_counts.values()]
            
            # Bye week features
            bye_week_counts = defaultdict(int)
            for player in our_roster:
                bye_week = player.metadata.get('bye_week', 0)
                if bye_week > 0:
                    bye_week_counts[bye_week] += 1
            
            bye_features = [
                len(bye_week_counts) / 11.0,
                max(bye_week_counts.values()) / len(our_roster) if our_roster else 0.0,
            ]
            
            # Team needs
            league_spots = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DEF': 1, 'K': 1}
            need_features = []
            for pos, required in league_spots.items():
                current = position_counts.get(pos, 0)
                urgency = max(0.0, (required - current) / required) if required > 0 else 0.0
                need_features.append(urgency)
            
            # Combine and pad basic features
            state_features = basic_features + position_features + bye_features + need_features
            while len(state_features) < 20:
                state_features.append(0.0)
            
            # Top available players (50 players × 5 features)
            available_list = list(draft_state.available_players)
            top_available = sorted(available_list, key=_player_rank_key)[:50]
            
            player_features = []
            for i in range(50):
                if i < len(top_available):
                    player = top_available[i]
                    player_features.extend([
                        player.vorp / 20.0,
                        1.0 if player.position == 'QB' else 0.0,
                        1.0 if player.position == 'RB' else 0.0,
                        1.0 if player.position == 'WR' else 0.0,
                        1.0 if player.position == 'TE' else 0.0,
                    ])
                else:
                    player_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Roster features (15 players × 8 features)
            roster_features = []
            for i in range(15):
                if i < len(our_roster):
                    player = our_roster[i]
                    roster_features.extend([
                        player.vorp / 20.0,
                        1.0 if player.position == 'QB' else 0.0,
                        1.0 if player.position == 'RB' else 0.0,
                        1.0 if player.position == 'WR' else 0.0,
                        1.0 if player.position == 'TE' else 0.0,
                        player.metadata.get('bye_week', 0) / 14.0,
                        player.metadata.get('injury_risk_score', 0.3),
                        player.metadata.get('durability_score', 0.7),
                    ])
                else:
                    roster_features.extend([0.0] * 8)
            
            # Combine all features
            all_features = state_features + player_features + roster_features
            
            return torch.tensor(all_features, dtype=torch.float32).unsqueeze(0)
            
        except Exception as e:
            print(f"⚠️  Error encoding state: {e}")
            return None
    
    def get_recommendations(self, draft_state, player_pool, top_k: int = 5) -> List[Tuple[Player, float]]:
        """Get top-k player recommendations from trained model"""
        
        if not self.value_network or not draft_state.available_players:
            # Fallback to VORP-based recommendations
            return self._fallback_recommendations(draft_state, top_k)
        
        try:
            # Encode current state
            state_tensor = self.encode_draft_state(draft_state, player_pool)
            if state_tensor is None:
                return self._fallback_recommendations(draft_state, top_k)
            
            # Get value prediction for current state
            with torch.no_grad():
                state_value = self.value_network(state_tensor).item()
            
            # Evaluate each available player
            player_scores = []
            available_players = list(draft_state.available_players)
            
            for player in available_players:
                # Create hypothetical state with this player drafted
                hypothetical_roster = draft_state.team_rosters.get(draft_state.our_team_id, []).copy()
                hypothetical_roster.append(player)
                
                # Calculate position need satisfaction
                position_bonus = self._calculate_position_bonus(player, hypothetical_roster)
                
                # Calculate bye week penalty
                bye_penalty = self._calculate_bye_week_penalty(player, hypothetical_roster)
                
                # Calculate injury risk penalty  
                injury_penalty = self._calculate_injury_penalty(player)
                
                # Base score from VORP
                base_score = player.vorp
                
                # Combined score
                total_score = base_score + position_bonus - bye_penalty - injury_penalty
                
                player_scores.append((player, total_score))
            
            # Sort by score and return top-k
            player_scores.sort(key=lambda x: _scored_rank_key(x[0], x[1]))
            return player_scores[:top_k]
            
        except Exception as e:
            print(f"⚠️  Error getting model recommendations: {e}")
            return self._fallback_recommendations(draft_state, top_k)
    
    def _fallback_recommendations(self, draft_state, top_k: int = 5) -> List[Tuple[Player, float]]:
        """Fallback VORP-based recommendations when model unavailable"""
        available_players = list(draft_state.available_players)
        
        # Simple VORP-based ranking with position needs
        scored_players = []
        our_roster = draft_state.team_rosters.get(draft_state.our_team_id, [])
        
        for player in available_players:
            score = player.vorp
            
            # Position need bonus
            position_counts = Counter(p.position for p in our_roster)
            if player.position in ['RB', 'WR'] and position_counts.get(player.position, 0) < 2:
                score += 2.0
            elif player.position in ['QB', 'TE', 'K', 'DEF'] and position_counts.get(player.position, 0) < 1:
                score += 1.5
            
            scored_players.append((player, score))
        
        scored_players.sort(key=lambda x: _scored_rank_key(x[0], x[1]))
        return scored_players[:top_k]
    
    def _calculate_position_bonus(self, player: Player, roster: List[Player]) -> float:
        """Calculate bonus for filling positional needs"""
        position_counts = Counter(p.position for p in roster)
        
        if player.position == 'QB' and position_counts.get('QB', 0) <= 1:
            return 1.5
        elif player.position in ['RB', 'WR'] and position_counts.get(player.position, 0) <= 2:
            return 2.0  
        elif player.position == 'TE' and position_counts.get('TE', 0) <= 1:
            return 1.5
        elif player.position in ['K', 'DEF'] and position_counts.get(player.position, 0) == 0:
            return 1.0
        
        return 0.0
    
    def _calculate_bye_week_penalty(self, player: Player, roster: List[Player]) -> float:
        """Calculate penalty for bye week clustering"""
        player_bye = player.metadata.get('bye_week', 0)
        if player_bye == 0:
            return 0.0
        
        bye_week_counts = Counter(p.metadata.get('bye_week', 0) for p in roster if p.metadata.get('bye_week', 0) > 0)
        
        # Penalty for creating bye week clusters
        if bye_week_counts.get(player_bye, 0) >= 2:
            return 1.0
        elif bye_week_counts.get(player_bye, 0) >= 1:
            return 0.5
        
        return 0.0
    
    def _calculate_injury_penalty(self, player: Player) -> float:
        """Calculate penalty for injury risk"""
        injury_risk = player.metadata.get('injury_risk_score', 0.3)
        return injury_risk * 2.0  # Scale injury risk to penalty


class InteractiveDraftAssistant:
    """Main interactive draft assistant class"""
    
    def __init__(self, our_team_id: int = 6, league_teams: int = 12):
        self.our_team_id = our_team_id
        self.league_teams = league_teams
        
        # Initialize components
        self.player_pool = self._load_player_data()
        self.draft_state = self._initialize_draft_state()
        self.mcts_model = self._load_trained_model()
        
        # Draft tracking
        self.pick_history = []
        self.current_round = 1
        self.current_pick_in_round = 1
        
        print("🏈 Interactive Fantasy Football Draft Assistant")
        print("=" * 50)
        print(f"✅ Loaded {len(self.player_pool)} players")
        print(f"🎯 Your team: #{self.our_team_id} in {self.league_teams}-team league")
        print(f"🤖 MCTS model: {'✅ Loaded' if self.mcts_model.value_network else '⚠️  Fallback mode'}")
    
    def _load_player_data(self) -> List[Player]:
        """Load player data from available sources"""
        try:
            # Try to load from refactored data loader
            return load_default_data()
        except:
            # Fallback to CSV loading
            return self._load_csv_data()
    
    def _load_csv_data(self) -> List[Player]:
        """Fallback CSV data loading"""
        players = []
        
        # Try different data sources
        data_files = [
            "data/raw/draft_board.csv",
            "draft_board.csv", 
            "data/processed/injury_enhanced_demo.csv",
            "injury_enhanced_demo.csv"
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    print(f"📊 Loading data from {file_path}")

                    # Fails loudly if the board is structurally broken.
                    validate_board(df)

                    for _, row in df.iterrows():
                        metadata = {}
                        
                        # Extract metadata fields
                        for col in df.columns:
                            if col not in ['player_name', 'position', 'team', 'vorp', 'VORP', 'adp_rank']:
                                metadata[col] = row[col] if pd.notna(row[col]) else None

                        # NOTE: the board CSV column is 'VORP' (uppercase). Series.get is
                        # case-sensitive, so reading only 'vorp' silently zeroes every player.
                        vorp = row.get('vorp', row.get('VORP', 0.0))
                        adp = row.get('adp_rank', row.get('ADP', 999.0))

                        player = Player(
                            name=row.get('player_name', row.get('name', f"Player_{len(players)}")),
                            position=row.get('position', 'UNKNOWN'),
                            team=row.get('team', 'UNKNOWN'),
                            vorp=float(vorp) if pd.notna(vorp) else 0.0,
                            adp_rank=float(adp) if pd.notna(adp) else 999.0,
                            metadata=metadata
                        )
                        
                        players.append(player)

                    validate_players(players)
                    return players
                    
                except Exception as e:
                    print(f"⚠️  Error loading {file_path}: {e}")
                    continue
        
        print("⚠️  No player data files found. Creating sample data.")
        return self._create_sample_data()
    
    def _create_sample_data(self) -> List[Player]:
        """Create sample player data if no files available"""
        players = []
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        
        for i in range(200):
            pos = positions[i % len(positions)]
            vorp = max(0, 15 - i * 0.2 + np.random.normal(0, 2))
            
            metadata = {
                'bye_week': np.random.choice(range(4, 15)),
                'injury_risk_score': np.random.beta(2, 5),
                'durability_score': np.random.uniform(0.6, 1.0),
                'proj_ppg': vorp + 10 + np.random.normal(0, 3)
            }
            
            player = Player(
                name=f"Player_{i+1:03d}",
                position=pos,
                team=f"Team_{(i % 32) + 1}",
                vorp=vorp,
                metadata=metadata
            )
            
            players.append(player)
        
        return players
    
    def _initialize_draft_state(self):
        """Initialize draft state"""
        try:
            # Try to use the refactored DraftState
            league_settings = LeagueSettings(
                teams=self.league_teams,
                roster_spots={'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DEF': 1, 'K': 1, 'BENCH': 6}
            )
            return DraftState(
                league=league_settings,
                our_team_id=self.our_team_id,
                available_players=set(self.player_pool)
            )
        except:
            # Fallback to simple state tracking
            return self._create_simple_draft_state()
    
    def _create_simple_draft_state(self):
        """Create simple draft state object"""
        class SimpleDraftState:
            def __init__(self, our_team_id, player_pool):
                self.our_team_id = our_team_id
                self.available_players = set(player_pool)
                self.team_rosters = {i: [] for i in range(1, 13)}
                self.current_round = 1
                self.current_pick_in_round = 1
        
        return SimpleDraftState(self.our_team_id, self.player_pool)
    
    def _load_trained_model(self) -> TrainedMCTSModel:
        """Load the trained MCTS model"""
        model_paths = [
            "model_weights/cpu_inference_mcts_model.pt",
            "model_weights/gpu_trained_mcts_model.pt",
            "cpu_inference_mcts_model.pt",
            "gpu_trained_mcts_model.pt"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                return TrainedMCTSModel(path)
        
        print("⚠️  No trained models found. Using fallback recommendations.")
        return TrainedMCTSModel("")
    
    def record_pick(self, player_name: str, team_id: int, round_num: int = None, pick_in_round: int = None):
        """Record a draft pick"""
        if round_num is None:
            round_num = self.current_round
        if pick_in_round is None:
            pick_in_round = self.current_pick_in_round
        
        # Find the player
        player = None
        for p in self.draft_state.available_players:
            if p.name.lower() == player_name.lower():
                player = p
                break
        
        if not player:
            print(f"❌ Player '{player_name}' not found in available players")
            return False
        
        # Record the pick
        self.draft_state.available_players.remove(player)
        self.draft_state.team_rosters[team_id].append(player)
        
        pick_info = {
            'round': round_num,
            'pick_in_round': pick_in_round,
            'overall_pick': (round_num - 1) * self.league_teams + pick_in_round,
            'team_id': team_id,
            'player': player
        }
        
        self.pick_history.append(pick_info)
        
        # Update current pick
        self._advance_pick()
        
        # Display pick
        team_indicator = "🟢 YOUR PICK" if team_id == self.our_team_id else f"Team {team_id}"
        print(f"📝 R{round_num}.{pick_in_round:02d} | {team_indicator} | {player.name} ({player.position}) | VORP: {player.vorp:.1f}")
        
        return True
    
    def _advance_pick(self):
        """Advance to next pick"""
        if self.current_pick_in_round < self.league_teams:
            self.current_pick_in_round += 1
        else:
            self.current_round += 1
            self.current_pick_in_round = 1
    
    def get_recommendations(self, top_k: int = 5) -> List[Tuple[Player, float]]:
        """Get MCTS recommendations for current draft state"""
        return self.mcts_model.get_recommendations(self.draft_state, self.player_pool, top_k)
    
    def show_draft_status(self):
        """Display current draft status"""
        print(f"\n🎯 DRAFT STATUS - Round {self.current_round}, Pick {self.current_pick_in_round}")
        print("=" * 50)
        
        # Our roster
        our_roster = self.draft_state.team_rosters[self.our_team_id]
        print(f"🟢 YOUR ROSTER ({len(our_roster)}/15):")
        
        if our_roster:
            for i, player in enumerate(our_roster, 1):
                bye_week = player.metadata.get('bye_week', 'N/A')
                injury_risk = player.metadata.get('injury_risk_score', 0.3)
                risk_indicator = "🔴" if injury_risk > 0.6 else "🟡" if injury_risk > 0.4 else "🟢"
                print(f"  {i:2d}. {player.name:25s} | {player.position:3s} | VORP: {player.vorp:5.1f} | Bye: {bye_week} | {risk_indicator}")
        else:
            print("  (No picks yet)")
        
        # Position analysis
        position_counts = Counter(p.position for p in our_roster)
        print(f"\n📊 POSITION BREAKDOWN:")
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        for pos in positions:
            count = position_counts.get(pos, 0)
            need_indicator = "✅" if count > 0 else "🔴"
            print(f"  {pos}: {count} {need_indicator}")
        
        # Bye week analysis
        bye_weeks = [p.metadata.get('bye_week') for p in our_roster if p.metadata.get('bye_week')]
        if bye_weeks:
            bye_counts = Counter(bye_weeks)
            max_bye_week = max(bye_counts.values())
            bye_indicator = "🔴" if max_bye_week >= 3 else "🟡" if max_bye_week >= 2 else "🟢"
            print(f"\n📅 BYE WEEK ANALYSIS {bye_indicator}:")
            for week, count in sorted(bye_counts.items()):
                print(f"  Week {week}: {count} players")
    
    def show_recommendations(self, top_k: int = 5):
        """Display MCTS recommendations"""
        recommendations = self.get_recommendations(top_k)
        
        print(f"\n🤖 MCTS RECOMMENDATIONS (Top {top_k}):")
        print("=" * 50)
        
        for i, (player, score) in enumerate(recommendations, 1):
            bye_week = player.metadata.get('bye_week', 'N/A')
            injury_risk = player.metadata.get('injury_risk_score', 0.3)
            risk_indicator = "🔴" if injury_risk > 0.6 else "🟡" if injury_risk > 0.4 else "🟢"
            
            print(f"{i}. {player.name:25s} | {player.position:3s} | Score: {score:5.1f} | Bye: {bye_week} | {risk_indicator}")
    
    def search_players(self, query: str, position: str = None, top_k: int = 10) -> List[Player]:
        """Search available players"""
        available = list(self.draft_state.available_players)
        
        # Filter by position if specified
        if position:
            available = [p for p in available if p.position.upper() == position.upper()]
        
        # Filter by name if query provided
        if query:
            available = [p for p in available if query.lower() in p.name.lower()]
        
        # Sort by VORP (deterministic: ties break on ADP, then name)
        available.sort(key=_player_rank_key)

        return available[:top_k]
    
    def undo_last_pick(self):
        """Undo the last recorded pick"""
        if not self.pick_history:
            print("❌ No picks to undo")
            return False
        
        last_pick = self.pick_history.pop()
        player = last_pick['player']
        team_id = last_pick['team_id']
        
        # Return player to available pool
        self.draft_state.available_players.add(player)
        self.draft_state.team_rosters[team_id].remove(player)
        
        # Reset pick counters
        self.current_round = last_pick['round']
        self.current_pick_in_round = last_pick['pick_in_round']
        
        print(f"↩️  Undid pick: {player.name}")
        return True


def main():
    """Main interactive loop"""
    print("🚀 Initializing Interactive Draft Assistant...")
    
    # Get user configuration - with fallback for non-interactive use
    our_team_id = 6  # Default values
    league_teams = 12
    
    try:
        import sys
        if sys.stdin.isatty():  # Only prompt if running interactively
            team_input = input("Enter your team number (1-12) [default: 6]: ").strip()
            if team_input:
                our_team_id = int(team_input)
            
            league_input = input("Enter number of teams in league [default: 12]: ").strip()
            if league_input:
                league_teams = int(league_input)
    except (ValueError, EOFError, KeyboardInterrupt):
        print(f"Using defaults: Team {our_team_id}, {league_teams} teams")
    except:
        pass  # Use defaults if any issues
    
    # Initialize assistant
    assistant = InteractiveDraftAssistant(our_team_id, league_teams)
    
    print("\n🎯 INTERACTIVE DRAFT COMMANDS:")
    print("  pick <player_name> <team_id>  - Record a pick")
    print("  recs [number]                 - Show MCTS recommendations")
    print("  status                        - Show draft status")
    print("  search <query> [position]     - Search players")
    print("  undo                          - Undo last pick")
    print("  help                          - Show this help")
    print("  quit                          - Exit")
    
    while True:
        try:
            assistant.show_draft_status()
            assistant.show_recommendations(5)
            
            command = input(f"\n🎯 Round {assistant.current_round}, Pick {assistant.current_pick_in_round} > ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'quit':
                print("👋 Thanks for using the Draft Assistant!")
                break
            
            elif cmd == 'pick':
                if len(parts) < 3:
                    print("❌ Usage: pick <player_name> <team_id>")
                    continue
                
                player_name = " ".join(parts[1:-1])
                try:
                    team_id = int(parts[-1])
                    assistant.record_pick(player_name, team_id)
                except ValueError:
                    print("❌ Team ID must be a number")
            
            elif cmd == 'recs':
                top_k = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
                assistant.show_recommendations(top_k)
            
            elif cmd == 'status':
                assistant.show_draft_status()
            
            elif cmd == 'search':
                if len(parts) < 2:
                    print("❌ Usage: search <query> [position]")
                    continue
                
                query = parts[1]
                position = parts[2] if len(parts) > 2 else None
                
                results = assistant.search_players(query, position)
                print(f"\n🔍 SEARCH RESULTS for '{query}'" + (f" (Position: {position})" if position else ""))
                print("-" * 50)
                
                for i, player in enumerate(results, 1):
                    bye_week = player.metadata.get('bye_week', 'N/A') 
                    print(f"{i:2d}. {player.name:25s} | {player.position:3s} | VORP: {player.vorp:5.1f} | Bye: {bye_week}")
            
            elif cmd == 'undo':
                assistant.undo_last_pick()
            
            elif cmd == 'help':
                print("\n🎯 INTERACTIVE DRAFT COMMANDS:")
                print("  pick <player_name> <team_id>  - Record a pick")
                print("  recs [number]                 - Show MCTS recommendations")
                print("  status                        - Show draft status")
                print("  search <query> [position]     - Search players")
                print("  undo                          - Undo last pick")
                print("  help                          - Show this help")
                print("  quit                          - Exit")
            
            else:
                print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n👋 Thanks for using the Draft Assistant!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
