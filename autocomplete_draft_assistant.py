#!/usr/bin/env python3
"""
Autocomplete Draft Assistant - Enhanced with Tab Completion
=========================================================

Interactive draft assistant with smart autocomplete for player names.
Uses readline for tab completion to eliminate typing errors.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import readline
import atexit
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.data.assertions import validate_board, validate_players

# Configure readline for better autocomplete experience
try:
    import readline
    import rlcompleter
    
    # Enable tab completion
    readline.parse_and_bind("tab: complete")
    
    # History file for better UX
    histfile = os.path.join(os.path.expanduser("~"), ".draft_assistant_history")
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    
    atexit.register(readline.write_history_file, histfile)
    
    READLINE_AVAILABLE = True
except ImportError:
    print("⚠️  Readline not available. Autocomplete will be limited.")
    READLINE_AVAILABLE = False


class Player:
    def __init__(self, name, position, team, vorp, bye_week=0, injury_risk=0.3, adp_rank=999.0):
        self.name = name
        self.position = position
        self.team = team
        self.vorp = float(vorp) if pd.notna(vorp) else 0.0
        self.bye_week = int(bye_week) if bye_week and pd.notna(bye_week) else 0
        self.injury_risk = float(injury_risk) if injury_risk and pd.notna(injury_risk) else 0.3
        self.adp_rank = float(adp_rank) if pd.notna(adp_rank) else 999.0

    def __str__(self):
        return f"{self.name} ({self.position})"


def rank_key(player):
    """Deterministic player ordering: best VORP first, ties on ADP then name.

    available_players is a set keyed on hash(name), which Python randomizes per
    process. Without an explicit tie-break, a scoring bug that flattens scores
    makes the top recommendation depend on the hash seed rather than on football.
    """
    return (-player.vorp, player.adp_rank, player.name)


def scored_rank_key(player, score):
    """Deterministic ordering for (player, score) pairs. Highest score first."""
    return (-score, player.adp_rank, player.name)


class PlayerCompleter:
    """Custom completer for player names"""
    
    def __init__(self, players):
        self.players = players
        self.player_names = [p.name for p in players]
        self.available_players = set(players)
        
        # Create name variations for better matching
        self.name_variations = {}
        for player in players:
            name = player.name
            
            # Add full name
            self.name_variations[name.lower()] = player
            
            # Add last name only
            parts = name.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                self.name_variations[last_name.lower()] = player
            
            # Add first + last (skip middle names/suffixes)
            if len(parts) >= 2:
                first_last = f"{parts[0]} {parts[-1]}"
                self.name_variations[first_last.lower()] = player
            
            # Add initials + last name
            if len(parts) >= 2:
                initials_last = f"{parts[0][0]}. {parts[-1]}"
                self.name_variations[initials_last.lower()] = player
    
    def update_available_players(self, available_players):
        """Update the list of available players for completion"""
        self.available_players = available_players
    
    def get_matches(self, text):
        """Get all matching player names for given text"""
        text_lower = text.lower()
        matches = []
        
        for player in self.available_players:
            name_lower = player.name.lower()
            
            # Check if text matches start of name or any word in name
            if (name_lower.startswith(text_lower) or 
                any(word.startswith(text_lower) for word in name_lower.split())):
                matches.append(player.name)
        
        return sorted(matches)
    
    def complete(self, text, state):
        """Completion function for readline"""
        if state == 0:
            # First call - generate matches
            self.matches = self.get_matches(text)
        
        try:
            return self.matches[state]
        except IndexError:
            return None
    
    def find_player_by_text(self, text):
        """Find player by text input (with fuzzy matching)"""
        text_lower = text.lower().strip()
        
        # Try exact match first
        for player in self.available_players:
            if player.name.lower() == text_lower:
                return player
        
        # Try variation matches
        if text_lower in self.name_variations:
            player = self.name_variations[text_lower]
            if player in self.available_players:
                return player
        
        # Try partial matches
        matches = []
        for player in self.available_players:
            name_lower = player.name.lower()
            if (text_lower in name_lower or 
                any(text_lower in word for word in name_lower.split())):
                matches.append(player)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # Multiple matches - show options
            print(f"\n🔍 Multiple players match '{text}':")
            for i, player in enumerate(matches[:10], 1):
                print(f"  {i}. {player.name} ({player.position}) - VORP: {player.vorp:.1f}")
            print("Please be more specific or use tab completion.")
            return None
        
        return None


class AutocompleteDraftAssistant:
    def __init__(self, your_team_position=6):
        self.your_team_position = your_team_position
        self.players = self.load_players()
        self.available_players = set(self.players)
        self.team_rosters = {i: [] for i in range(1, 13)}
        self.current_round = 1
        self.current_pick = 1
        self.pick_history = []
        
        # Setup autocomplete
        self.completer = PlayerCompleter(self.players)
        if READLINE_AVAILABLE:
            readline.set_completer(self.completer.complete)
            readline.set_completer_delims(' \t\n')
        
        print("🏈 Autocomplete Fantasy Football Draft Assistant")
        print("=" * 50)
        print(f"✅ Loaded {len(self.players)} players")
        print(f"🎯 Your team picks at position #{your_team_position}")
        if READLINE_AVAILABLE:
            print("⌨️  Tab completion enabled - press TAB to autocomplete player names!")
        print("📋 Type 'help' for commands or start entering picks!")
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

                    # Fails loudly if the board is structurally broken (e.g. a
                    # mis-cased VORP column zeroing every player).
                    validate_board(df)

                    for _, row in df.iterrows():
                        # Handle different column names
                        name = row.get('player_name', row.get('name', row.get('Player', f"Player_{len(players)}")))
                        position = row.get('position', row.get('Position', 'UNKNOWN'))
                        team = row.get('team', row.get('Team', 'UNKNOWN'))
                        vorp = row.get('vorp', row.get('VORP', 0))
                        bye_week = row.get('bye_week', row.get('Bye', 0))
                        injury_risk = row.get('injury_risk_score', 0.3)
                        adp_rank = row.get('adp_rank', row.get('ADP', row.get('Rank', 999)))

                        player = Player(name, position, team, vorp, bye_week, injury_risk, adp_rank)
                        players.append(player)

                    validate_players(players)
                    return players

                except Exception as e:
                    print(f"⚠️  Could not load {file_path}: {e}")
                    continue
        
        # Fallback: create sample data
        print("⚠️  No data files found. Creating sample players...")
        return self.create_sample_players()
    
    def create_sample_players(self):
        """Create sample player data with realistic names"""
        players = []
        
        # Sample realistic player names by position
        sample_names = {
            'QB': ['Josh Allen', 'Patrick Mahomes', 'Lamar Jackson', 'Joe Burrow', 'Justin Herbert',
                   'Dak Prescott', 'Russell Wilson', 'Kirk Cousins', 'Derek Carr', 'Tua Tagovailoa'],
            'RB': ['Christian McCaffrey', 'Saquon Barkley', 'Derrick Henry', 'Alvin Kamara', 'Nick Chubb',
                   'Austin Ekeler', 'Jonathan Taylor', 'Dalvin Cook', 'Aaron Jones', 'Josh Jacobs'],
            'WR': ['Cooper Kupp', 'Davante Adams', 'Tyreek Hill', 'Stefon Diggs', 'DeAndre Hopkins',
                   'Mike Evans', 'Keenan Allen', 'DK Metcalf', 'CeeDee Lamb', 'A.J. Brown'],
            'TE': ['Travis Kelce', 'Mark Andrews', 'George Kittle', 'Darren Waller', 'Kyle Pitts',
                   'T.J. Hockenson', 'Dallas Goedert', 'Pat Freiermuth', 'Tyler Higbee', 'Noah Fant'],
            'K': ['Justin Tucker', 'Harrison Butker', 'Daniel Carlson', 'Tyler Bass', 'Matt Gay',
                  'Ryan McManus', 'Nick Folk', 'Mason Crosby', 'Greg Zuerlein', 'Evan McPherson'],
            'DEF': ['Buffalo Bills', 'Pittsburgh Steelers', 'New England Patriots', 'Tampa Bay Buccaneers',
                    'Dallas Cowboys', 'San Francisco 49ers', 'Los Angeles Rams', 'Green Bay Packers',
                    'Indianapolis Colts', 'New Orleans Saints']
        }
        
        teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 
                'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
                'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
                'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
        
        for pos, names in sample_names.items():
            for i, name in enumerate(names):
                # Generate realistic VORP based on position and ranking
                if pos == 'QB':
                    vorp = max(0, 20 - i * 1.5)
                elif pos in ['RB', 'WR']:
                    vorp = max(0, 18 - i * 1.2)
                elif pos == 'TE':
                    vorp = max(0, 12 - i * 0.8)
                else:  # K, DEF
                    vorp = max(0, 8 - i * 0.5)
                
                vorp += np.random.normal(0, 0.5)  # Add some randomness
                
                bye_week = np.random.choice(range(4, 15))
                injury_risk = np.random.beta(2, 5)
                team = teams[i % len(teams)]
                
                player = Player(name, pos, team, vorp, bye_week, injury_risk)
                players.append(player)
        
        return players
    
    def smart_input(self, prompt):
        """Get input with autocomplete support"""
        if READLINE_AVAILABLE:
            return input(prompt)
        else:
            # Fallback for systems without readline
            return input(prompt)
    
    def find_player_interactive(self, name_query):
        """Find player with interactive help for multiple matches"""
        return self.completer.find_player_by_text(name_query)
    
    def record_pick(self, player_input, team_position=None):
        """Record a draft pick with autocomplete support"""
        if team_position is None:
            team_position = self.get_current_picking_team()
        
        # Handle team number at end of input
        parts = player_input.strip().split()
        if len(parts) >= 2 and parts[-1].isdigit():
            team_position = int(parts[-1])
            player_name = " ".join(parts[:-1])
        else:
            player_name = player_input.strip()
        
        player = self.find_player_interactive(player_name)
        if not player:
            return False
        
        # Record the pick
        self.available_players.remove(player)
        self.completer.update_available_players(self.available_players)
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
        
        # Sort by score and return top recommendations (deterministic tie-break)
        player_scores.sort(key=lambda x: scored_rank_key(x[0], x[1]))
        return player_scores[:count]
    
    def show_recommendations(self, count=5):
        """Display top recommendations"""
        recs = self.get_recommendations(count)
        
        print(f"\n🤖 TOP {count} RECOMMENDATIONS:")
        print("-" * 50)
        
        for i, (player, score) in enumerate(recs, 1):
            risk_indicator = "🔴" if player.injury_risk > 0.6 else "🟡" if player.injury_risk > 0.4 else "🟢"
            bye_text = f"Bye: {player.bye_week}" if player.bye_week > 0 else "Bye: --"
            
            print(f"{i}. {player.name:25s} | {player.position:3s} | Score: {score:5.1f} | {bye_text} | {risk_indicator}")
    
    def show_roster(self):
        """Show current roster"""
        our_roster = self.team_rosters[self.your_team_position]
        
        print(f"\n🟢 YOUR ROSTER ({len(our_roster)}/15):")
        print("-" * 50)
        
        if our_roster:
            for i, player in enumerate(our_roster, 1):
                bye_text = f"Bye: {player.bye_week}" if player.bye_week > 0 else "Bye: --"
                risk_indicator = "🔴" if player.injury_risk > 0.6 else "🟡" if player.injury_risk > 0.4 else "🟢"
                print(f"{i:2d}. {player.name:25s} | {player.position:3s} | VORP: {player.vorp:5.1f} | {bye_text} | {risk_indicator}")
        else:
            print("  (No picks yet)")
        
        # Position summary
        position_counts = Counter(p.position for p in our_roster)
        print(f"\n📊 POSITIONS: QB:{position_counts.get('QB',0)} RB:{position_counts.get('RB',0)} WR:{position_counts.get('WR',0)} TE:{position_counts.get('TE',0)} K:{position_counts.get('K',0)} DEF:{position_counts.get('DEF',0)}")
    
    def search_players(self, query, position=None, count=10):
        """Search available players with autocomplete support"""
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
        results.sort(key=rank_key)
        
        print(f"\n🔍 SEARCH RESULTS: '{query}'" + (f" (Position: {position})" if position else ""))
        print("-" * 50)
        
        for i, player in enumerate(results[:count], 1):
            bye_text = f"Bye: {player.bye_week}" if player.bye_week > 0 else "Bye: --"
            print(f"{i:2d}. {player.name:25s} | {player.position:3s} | VORP: {player.vorp:5.1f} | {bye_text}")
    
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
        self.completer.update_available_players(self.available_players)
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
    print("🚀 Starting Autocomplete Draft Assistant...")
    
    # Get your draft position - with fallback
    position = 6
    
    try:
        if sys.stdin.isatty():
            user_input = input("What's your draft position? (1-12) [default: 6]: ").strip()
            if user_input:
                position = int(user_input)
    except (ValueError, EOFError, KeyboardInterrupt):
        print(f"Using default position: {position}")
    except:
        pass
    
    assistant = AutocompleteDraftAssistant(position)
    
    print("\n🎯 AUTOCOMPLETE COMMANDS:")
    print("  <player_name>         - Record pick (use TAB to autocomplete!)")
    print("  <player_name> <team>  - Record pick for specific team")
    print("  recs                  - Show recommendations") 
    print("  roster                - Show your roster")
    print("  search <query>        - Search players (try typing partial names)")
    print("  undo                  - Undo last pick")
    print("  quit                  - Exit")
    if READLINE_AVAILABLE:
        print("\n⌨️  PRO TIP: Press TAB while typing player names for autocomplete!")
    print()
    
    while True:
        try:
            # Show current status
            assistant.show_status()
            
            # Get command with autocomplete
            current_team = assistant.get_current_picking_team()
            is_our_turn = current_team == assistant.your_team_position
            prompt = f"\n🎯 R{assistant.current_round}.{assistant.current_pick} Team {current_team}" + (" (YOUR TURN)" if is_our_turn else "") + " > "
            
            command = assistant.smart_input(prompt).strip()
            
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
                print("\n🎯 AUTOCOMPLETE COMMANDS:")
                print("  <player_name>         - Record pick (use TAB to autocomplete!)")
                print("  <player_name> <team>  - Record pick for specific team")
                print("  recs                  - Show recommendations")
                print("  roster                - Show your roster")
                print("  search <query>        - Search players")
                print("  undo                  - Undo last pick")
                print("  quit                  - Exit")
                if READLINE_AVAILABLE:
                    print("\n⌨️  PRO TIP: Press TAB while typing player names for autocomplete!")
            
            else:
                # Assume it's a player pick
                assistant.record_pick(command)
        
        except KeyboardInterrupt:
            print("\n👋 Good luck with your draft!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
