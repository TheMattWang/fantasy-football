#!/usr/bin/env python3
"""
Test script to verify the draft assistant loads properly
"""

import sys
import os

def test_quick_assistant():
    """Test the quick draft assistant initialization"""
    print("🧪 Testing Quick Draft Assistant...")
    
    try:
        # Import and test basic functionality
        sys.path.append('.')
        from quick_draft_assistant import QuickDraftAssistant
        
        # Create assistant with default settings
        assistant = QuickDraftAssistant(your_team_position=6)
        
        print(f"✅ Assistant loaded with {len(assistant.players)} players")
        print(f"✅ Your team position: {assistant.your_team_position}")
        print(f"✅ Available players: {len(assistant.available_players)}")
        
        # Test recommendations
        recs = assistant.get_recommendations(3)
        print(f"✅ Generated {len(recs)} recommendations")
        
        if recs:
            print("Top recommendation:")
            player, score = recs[0]
            print(f"  {player.name} ({player.position}) - Score: {score:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing quick assistant: {e}")
        return False

def test_full_assistant():
    """Test the full interactive assistant initialization"""
    print("\n🧪 Testing Full Interactive Assistant...")
    
    try:
        # Import and test basic functionality
        from interactive_draft_assistant import InteractiveDraftAssistant
        
        # Create assistant with default settings
        assistant = InteractiveDraftAssistant(our_team_id=6, league_teams=12)
        
        print(f"✅ Assistant loaded with {len(assistant.player_pool)} players")
        print(f"✅ Your team ID: {assistant.our_team_id}")
        print(f"✅ League teams: {assistant.league_teams}")
        print(f"✅ MCTS model loaded: {assistant.mcts_model.value_network is not None}")
        
        # Test recommendations
        recs = assistant.get_recommendations(3)
        print(f"✅ Generated {len(recs)} recommendations")
        
        if recs:
            print("Top recommendation:")
            player, score = recs[0]
            print(f"  {player.name} ({player.position}) - Score: {score:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing full assistant: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Draft Assistant Components")
    print("=" * 40)
    
    quick_ok = test_quick_assistant()
    full_ok = test_full_assistant()
    
    print("\n📊 TEST RESULTS:")
    print("=" * 40)
    print(f"Quick Assistant: {'✅ PASS' if quick_ok else '❌ FAIL'}")
    print(f"Full Assistant:  {'✅ PASS' if full_ok else '❌ FAIL'}")
    
    if quick_ok and full_ok:
        print("\n🎉 All tests passed! Your draft assistants are ready to use.")
        print("\n🚀 To start drafting:")
        print("   python quick_draft_assistant.py      (recommended)")
        print("   python interactive_draft_assistant.py (advanced)")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return quick_ok and full_ok

if __name__ == "__main__":
    main()
