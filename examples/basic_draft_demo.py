#!/usr/bin/env python3
"""
Basic Fantasy Football Draft Demo
==================================

This example demonstrates the clean, refactored fantasy football
draft strategy system with simple, easy-to-use APIs.

Usage:
    python examples/basic_draft_demo.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.player import Player, PlayerPool
from src.core.draft import DraftState, LeagueSettings
from src.utils.data_loader import load_player_pool, create_sample_player_pool, get_default_league_settings


def main():
    """Run basic draft demonstration."""
    
    print("🏈 Fantasy Football Basic Draft Demo")
    print("=" * 50)
    
    # 1. Load player data (try real data, fall back to sample)
    print("\n📊 Loading player data...")
    try:
        player_pool = load_player_pool(source="draft_board")
        print(f"✅ Loaded real player data: {len(player_pool)} players")
    except:
        print("⚠️  Real data not found, creating sample data...")
        player_pool = create_sample_player_pool(n_players=200)
        print(f"✅ Created sample data: {len(player_pool)} players")
    
    # 2. Show player pool summary
    print(f"\n📋 Player Pool Summary:")
    summary = player_pool.get_summary()
    print(f"   Total players: {summary['total_players']}")
    print(f"   Positions: {summary['positions']}")
    print(f"   Average VORP: {summary['avg_vorp']:.2f}")
    
    # 3. Show top players by position
    print(f"\n🔝 Top 3 Players by Position:")
    for position in ['QB', 'RB', 'WR', 'TE']:
        top_players = player_pool.get_top_players(3, position=position)
        print(f"   {position}: ", end="")
        print(", ".join([f"{p.name} ({p.vorp:.1f})" for p in top_players]))
    
    # 4. Create league settings
    print(f"\n⚙️  Creating league settings...")
    league = get_default_league_settings()
    print(f"   {league.teams} teams, {league.total_rounds} rounds")
    print(f"   Roster: {league.roster_spots}")
    
    # 5. Initialize draft
    print(f"\n🎯 Initializing draft...")
    draft_state = DraftState.create_mock_draft(
        player_pool=player_pool,
        our_team_id=6,  # 6th pick
        league_settings=league
    )
    
    print(f"   Draft position: {draft_state.our_team_id}")
    print(f"   Available players: {len(draft_state.available_players)}")
    print(f"   {draft_state.get_pick_summary()}")
    
    # 6. Simulate some basic picks
    print(f"\n🚀 Simulating basic draft strategy...")
    
    rounds_to_simulate = 5
    for round_num in range(rounds_to_simulate):
        
        if draft_state.is_our_turn:
            # Our turn - use simple "best available" strategy
            our_roster = draft_state.get_our_roster()
            our_needs = draft_state.get_our_roster_needs()
            
            # Find best available player for our needs
            best_player = None
            best_score = -999
            
            for player in draft_state.available_players:
                score = player.vorp
                
                # Bonus for needed positions
                if our_needs.get(player.position, 0) > 0:
                    score += 2.0
                elif player.position in league.flex_positions and our_needs.get('FLEX', 0) > 0:
                    score += 1.0
                
                if score > best_score:
                    best_score = score
                    best_player = player
            
            if best_player:
                print(f"   📍 Our pick: {best_player.name} ({best_player.position}) - VORP: {best_player.vorp:.2f}")
                draft_state.make_pick(best_player)
            
        else:
            # Other team's turn - simulate with random top player
            available = list(draft_state.available_players)
            if available:
                # Simulate other team picking from top 20 players
                top_players = sorted(available, key=lambda p: p.vorp, reverse=True)[:20]
                import random
                pick = random.choice(top_players[:min(10, len(top_players))])
                draft_state.make_pick(pick)
        
        # Stop if draft is complete or no players left
        if draft_state.is_draft_complete() or len(draft_state.available_players) == 0:
            break
    
    # 7. Show our final roster
    print(f"\n🏆 Our Draft Results:")
    our_roster = draft_state.get_our_roster()
    our_needs = draft_state.get_our_roster_needs()
    
    print(f"   Roster size: {len(our_roster)}")
    print(f"   Total VORP: {sum(p.vorp for p in our_roster):.2f}")
    
    print(f"\n   📋 Our Players:")
    for i, player in enumerate(our_roster, 1):
        print(f"   {i:2d}. {player.name:20s} ({player.position}) - VORP: {player.vorp:.2f}")
    
    print(f"\n   📊 Position Breakdown:")
    position_counts = {}
    for player in our_roster:
        pos = player.position
        position_counts[pos] = position_counts.get(pos, 0) + 1
    
    for pos, count in position_counts.items():
        needed = league.roster_spots.get(pos, 0)
        status = "✅" if count >= needed else "❌"
        print(f"   {status} {pos}: {count}/{needed}")
    
    print(f"\n   🎯 Remaining Needs:")
    for pos, need in our_needs.items():
        if need > 0:
            print(f"   • {pos}: {need} more needed")
    
    # 8. Show draft summary
    print(f"\n📈 Draft Summary:")
    draft_summary = draft_state.get_draft_summary()
    print(f"   Current pick: {draft_summary['current_pick']}")
    print(f"   Total picks made: {draft_summary['total_picks_made']}")
    print(f"   Remaining players: {draft_summary['available_players']}")
    
    print(f"\n✅ Basic Draft Demo Complete!")
    print(f"💡 This example shows the clean, simple API of the refactored system.")
    print(f"🔧 You can easily extend this with MCTS, injury awareness, and more!")


if __name__ == "__main__":
    main()
