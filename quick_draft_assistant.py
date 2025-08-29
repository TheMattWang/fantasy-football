#!/usr/bin/env python3
"""
Quick Draft Assistant - Simplified Interface
===========================================

A streamlined version of the draft assistant for easy real-time use.
Just input picks as they happen and get instant MCTS recommendations.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# Simple player class for quick usage
class Player:
    def __init__(self, name, position, team, vorp, bye_week=0, injury_risk=0.3):
        self.name = name
        self.position = position  
        self.team = team
        self.vorp = float(vorp)
        self.bye_week = int(bye_week) if bye_week else 0
        self.injury_risk = float(injury_risk) if injury_risk else 0.3
    
    def __str__(self):
        return f"{self.name} ({self.position})"

class QuickDraftAssistant:
    def __init__(self, your_team_position=6):
        self.your_team_position = your_team_position
        self.players = self.load_players()
        self.available_players = set(self.players)
        self.team_rosters = {i: [] for i in range(1, 13)}
        self.current_round = 1
        self.current_pick = 1
        self.pick_history = []
        
        print("🏈 Quick Fantasy Football Draft Assistant")
        print("=" * 45)
        print(f"✅ Loaded {len(self.players)} players")
        print(f"🎯 Your team picks at position #{your_team_position}")
        print(f"📋 Type 'help' for commands or just start entering picks!")
        print()
    
    def load_players(self):
        """Load player data from available files"""
        players = []
        
        # Try to load from various sources
        data_files = [
            "data/raw/draft_board.csv",
            "draft_board.csv",
            "data/processed/injury_enhanced_demo.csv", 
            "injury_enhanced_demo.csv",
            "FantasyPros_2025_Overall_ADP_Rankings.csv"
        ]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    print(f"📊 Loading from {file_path}")
                    
                    for _, row in df.iterrows():
                        # Handle different column names
                        name = row.get('player_name', row.get('name', row.get('Player', f"Player_{len(players)}")))
                        position = row.get('position', row.get('Position', 'UNKNOWN'))
                        team = row.get('team', row.get('Team', 'UNKNOWN'))
                        vorp = row.get('vorp', row.get('VORP', 0))
                        bye_week = row.get('bye_week', row.get('Bye', 0))
                        injury_risk = row.get('injury_risk_score', 0.3)
                        
                        player = Player(name, position, team, vorp, bye_week, injury_risk)
                        players.append(player)
                    
                    return players
                    
                except Exception as e:
                    print(f"⚠️  Could not load {file_path}: {e}")
                    continue
        
        # Fallback: create sample data
        print("⚠️  No data files found. Creating sample players...")
        return self.create_sample_players()
    
    def create_sample_players(self):
        """Create sample player data"""
        players = []
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 
                'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
                'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
                'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
        
        for i in range(300):
            pos = positions[i % len(positions)]
            team = teams[i % len(teams)]
            
            # Generate realistic VORP based on position and draft order
            if pos == 'QB':
                base_vorp = max(0, 18 - i * 0.15)
            elif pos in ['RB', 'WR']:
                base_vorp = max(0, 15 - i * 0.12)
            elif pos == 'TE':
                base_vorp = max(0, 8 - i * 0.08)
            else:  # K, DEF
                base_vorp = max(0, 5 - i * 0.05)
            
            vorp = base_vorp + np.random.normal(0, 1)
            bye_week = np.random.choice(range(4, 15))
            injury_risk = np.random.beta(2, 5)  # Skewed toward lower risk
            
            player = Player(
                name=f"{pos}_{team}_{i//6 + 1}",
                position=pos,
                team=team,
                vorp=vorp,
                bye_week=bye_week,
                injury_risk=injury_risk
            )
            players.append(player)
        
        return players
    
    def find_player(self, name_query):
        """Find player by name (fuzzy matching)"""
        name_query = name_query.lower().strip()
        
        # Exact match first
        for player in self.available_players:
            if player.name.lower() == name_query:
                return player
        
        # Partial match
        matches = []
        for player in self.available_players:
            if name_query in player.name.lower():
                matches.append(player)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"🔍 Multiple matches found for '{name_query}':")
            for i, player in enumerate(matches[:5], 1):
                print(f"  {i}. {player.name} ({player.position}) - VORP: {player.vorp:.1f}")
            return None
        else:
            print(f"❌ No player found matching '{name_query}'")
            return None
    
    def record_pick(self, player_name, team_position=None):
        """Record a draft pick"""
        if team_position is None:
            team_position = self.get_current_picking_team()
        
        player = self.find_player(player_name)
        if not player:
            return False
        
        # Record the pick
        self.available_players.remove(player)
        self.team_rosters[team_position].append(player)
        
        pick_info = {
            'round': self.current_round,
            'pick': self.current_pick,
            'team': team_position,
            'player': player
        }
        self.pick_history.append(pick_info)
        
        # Display the pick
        team_indicator = "🟢 YOUR PICK" if team_position == self.your_team_position else f"Team {team_position}"
        print(f"📝 R{self.current_round}.{self.current_pick:02d} | {team_indicator} | {player.name} ({player.position}) | VORP: {player.vorp:.1f}")
        
        # Advance to next pick
        self.current_pick += 1
        if self.current_pick > 12:
            self.current_pick = 1
            self.current_round += 1
        
        return True
    
    def get_current_picking_team(self):
        """Get which team is currently picking (snake draft)"""
        if self.current_round % 2 == 1:  # Odd rounds: 1, 2, 3...
            return self.current_pick
        else:  # Even rounds: 12, 11, 10...
            return 13 - self.current_pick
    
    def get_recommendations(self, count=5):
        """Get top recommendations based on team needs and player value"""
        if not self.available_players:
            return []
        
        our_roster = self.team_rosters[self.your_team_position]
        
        # Calculate scores for each available player
        player_scores = []
        
        for player in self.available_players:
            score = player.vorp  # Base score
            
            # Position need bonus
            position_counts = Counter(p.position for p in our_roster)
            
            if player.position == 'QB' and position_counts.get('QB', 0) == 0:
                score += 3.0  # Need first QB
            elif player.position == 'QB' and position_counts.get('QB', 0) == 1:
                score += 0.5  # Backup QB
            elif player.position in ['RB', 'WR']:
                current_count = position_counts.get(player.position, 0)
                if current_count == 0:
                    score += 4.0  # First RB/WR
                elif current_count == 1:
                    score += 2.0  # Second RB/WR
                elif current_count == 2:
                    score += 0.5  # Third RB/WR
            elif player.position == 'TE' and position_counts.get('TE', 0) == 0:
                score += 2.0  # First TE
            elif player.position in ['K', 'DEF'] and position_counts.get(player.position, 0) == 0:
                score += 1.0  # First K/DEF
            
            # Bye week penalty (avoid clustering)
            our_bye_weeks = [p.bye_week for p in our_roster if p.bye_week > 0]
            bye_counts = Counter(our_bye_weeks)
            if player.bye_week in bye_counts:
                if bye_counts[player.bye_week] >= 2:
                    score -= 2.0  # Heavy penalty for 3+ players
                elif bye_counts[player.bye_week] >= 1:
                    score -= 0.5  # Light penalty for 2 players
            
            # Injury risk penalty
            score -= player.injury_risk * 1.5
            
            player_scores.append((player, score))
        
        # Sort by score and return top recommendations
        player_scores.sort(key=lambda x: x[1], reverse=True)
        return player_scores[:count]
    
    def show_recommendations(self, count=5):
        """Display top recommendations"""
        recs = self.get_recommendations(count)
        
        print(f"\n🤖 TOP {count} RECOMMENDATIONS:")
        print("-" * 45)
        
        for i, (player, score) in enumerate(recs, 1):
            risk_indicator = "🔴" if player.injury_risk > 0.6 else "🟡" if player.injury_risk > 0.4 else "🟢"
            bye_text = f"Bye: {player.bye_week}" if player.bye_week > 0 else "Bye: --"
            
            print(f"{i}. {player.name:20s} | {player.position:3s} | Score: {score:5.1f} | {bye_text} | {risk_indicator}")
    
    def show_roster(self):
        """Show current roster"""
        our_roster = self.team_rosters[self.your_team_position]
        
        print(f"\n🟢 YOUR ROSTER ({len(our_roster)}/15):")
        print("-" * 45)
        
        if our_roster:
            for i, player in enumerate(our_roster, 1):
                bye_text = f"Bye: {player.bye_week}" if player.bye_week > 0 else "Bye: --"
                risk_indicator = "🔴" if player.injury_risk > 0.6 else "🟡" if player.injury_risk > 0.4 else "🟢"
                print(f"{i:2d}. {player.name:20s} | {player.position:3s} | VORP: {player.vorp:5.1f} | {bye_text} | {risk_indicator}")
        else:
            print("  (No picks yet)")
        
        # Position summary
        position_counts = Counter(p.position for p in our_roster)
        print(f"\n📊 POSITIONS: QB:{position_counts.get('QB',0)} RB:{position_counts.get('RB',0)} WR:{position_counts.get('WR',0)} TE:{position_counts.get('TE',0)} K:{position_counts.get('K',0)} DEF:{position_counts.get('DEF',0)}")
    
    def search_players(self, query, position=None, count=10):
        """Search available players"""
        query = query.lower() if query else ""
        results = []
        
        for player in self.available_players:
            # Position filter
            if position and player.position.upper() != position.upper():
                continue
            
            # Name filter
            if query and query not in player.name.lower():
                continue
            
            results.append(player)
        
        # Sort by VORP
        results.sort(key=lambda p: p.vorp, reverse=True)
        
        print(f"\n🔍 SEARCH RESULTS: '{query}'" + (f" (Position: {position})" if position else ""))
        print("-" * 45)
        
        for i, player in enumerate(results[:count], 1):
            bye_text = f"Bye: {player.bye_week}" if player.bye_week > 0 else "Bye: --"
            print(f"{i:2d}. {player.name:20s} | {player.position:3s} | VORP: {player.vorp:5.1f} | {bye_text}")
    
    def undo_last_pick(self):
        """Undo the last pick"""
        if not self.pick_history:
            print("❌ No picks to undo")
            return False
        
        last_pick = self.pick_history.pop()
        player = last_pick['player']
        team = last_pick['team']
        
        # Return player to available pool
        self.available_players.add(player)
        self.team_rosters[team].remove(player)
        
        # Reset counters
        self.current_pick -= 1
        if self.current_pick < 1:
            self.current_pick = 12
            self.current_round -= 1
        
        print(f"↩️  Undid: {player.name}")
        return True
    
    def show_status(self):
        """Show current draft status"""
        current_team = self.get_current_picking_team()
        is_our_turn = current_team == self.your_team_position
        
        print(f"\n🎯 DRAFT STATUS")
        print(f"Round: {self.current_round} | Pick: {self.current_pick}")
        print(f"Now picking: Team {current_team}" + (" (🟢 YOUR TURN!)" if is_our_turn else ""))
        
        self.show_roster()
        self.show_recommendations()


def main():
    """Main interactive loop"""
    print("🚀 Starting Quick Draft Assistant...")
    
    # Get your draft position - with timeout fallback
    position = 6  # Default position
    
    try:
        import sys
        if sys.stdin.isatty():  # Only prompt if running interactively
            user_input = input("What's your draft position? (1-12) [default: 6]: ").strip()
            if user_input:
                position = int(user_input)
    except (ValueError, EOFError, KeyboardInterrupt):
        print(f"Using default position: {position}")
    except:
        pass  # Use default if any other issues
    
    assistant = QuickDraftAssistant(position)
    
    print("\n🎯 QUICK COMMANDS:")
    print("  <player_name>         - Record a pick (auto-detects current team)")
    print("  <player_name> <team>  - Record pick for specific team")
    print("  recs                  - Show recommendations") 
    print("  roster                - Show your roster")
    print("  search <query>        - Search players")
    print("  undo                  - Undo last pick")
    print("  quit                  - Exit")
    print()
    
    while True:
        try:
            # Show current status
            assistant.show_status()
            
            # Get command
            current_team = assistant.get_current_picking_team()
            is_our_turn = current_team == assistant.your_team_position
            prompt = f"\n🎯 R{assistant.current_round}.{assistant.current_pick} Team {current_team}" + (" (YOUR TURN)" if is_our_turn else "") + " > "
            
            command = input(prompt).strip()
            
            if not command:
                continue
            
            parts = command.split()
            
            if parts[0].lower() == 'quit':
                print("👋 Good luck with your draft!")
                break
            
            elif parts[0].lower() == 'recs':
                assistant.show_recommendations()
            
            elif parts[0].lower() == 'roster':
                assistant.show_roster()
            
            elif parts[0].lower() == 'search':
                query = " ".join(parts[1:]) if len(parts) > 1 else ""
                assistant.search_players(query)
            
            elif parts[0].lower() == 'undo':
                assistant.undo_last_pick()
            
            elif parts[0].lower() == 'help':
                print("\n🎯 QUICK COMMANDS:")
                print("  <player_name>         - Record a pick (auto-detects current team)")
                print("  <player_name> <team>  - Record pick for specific team")
                print("  recs                  - Show recommendations")
                print("  roster                - Show your roster")
                print("  search <query>        - Search players")
                print("  undo                  - Undo last pick")
                print("  quit                  - Exit")
            
            else:
                # Assume it's a player pick
                if len(parts) >= 2 and parts[-1].isdigit():
                    # Player name + team number
                    player_name = " ".join(parts[:-1])
                    team_num = int(parts[-1])
                    assistant.record_pick(player_name, team_num)
                else:
                    # Just player name - use current team
                    player_name = command
                    assistant.record_pick(player_name)
        
        except KeyboardInterrupt:
            print("\n👋 Good luck with your draft!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
