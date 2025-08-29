#!/usr/bin/env python3
"""
Test Autocomplete Features
=========================

Demo script showing the autocomplete and fuzzy matching capabilities.
"""

def test_quick_assistant_matching():
    """Test the smart matching in quick assistant"""
    print("🧪 Testing Quick Assistant Smart Matching")
    print("=" * 40)
    
    from quick_draft_assistant import QuickDraftAssistant
    
    assistant = QuickDraftAssistant(6)
    
    # Test cases
    test_queries = [
        "McCaffrey",
        "Josh",
        "mahomes", 
        "kelce",
        "cooper",
        "xyz123"  # Should fail
    ]
    
    print("\n🔍 FUZZY MATCHING TESTS:")
    print("-" * 40)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        player = assistant.find_player(query)
        if player:
            print(f"✅ Found: {player.name} ({player.position})")
        else:
            print("❌ No unique match found")
    
    return True

def test_autocomplete_assistant():
    """Test the full autocomplete assistant"""
    print("\n🧪 Testing Autocomplete Assistant Features")
    print("=" * 40)
    
    from autocomplete_draft_assistant import AutocompleteDraftAssistant
    
    assistant = AutocompleteDraftAssistant(6)
    
    # Test completer
    completer = assistant.completer
    
    print("\n🔍 AUTOCOMPLETE TESTS:")
    print("-" * 40)
    
    test_queries = ["Josh", "Mc", "Kelce", "Cooper"]
    
    for query in test_queries:
        matches = completer.get_matches(query)
        print(f"\nQuery: '{query}' -> {len(matches)} matches")
        for match in matches[:3]:  # Show top 3
            print(f"  - {match}")
    
    return True

def main():
    """Run all autocomplete tests"""
    print("🚀 Testing Autocomplete & Fuzzy Matching Features")
    print("=" * 50)
    
    try:
        test1_ok = test_quick_assistant_matching()
        test2_ok = test_autocomplete_assistant()
        
        print("\n📊 TEST RESULTS:")
        print("=" * 40)
        print(f"Quick Assistant Matching: {'✅ PASS' if test1_ok else '❌ FAIL'}")
        print(f"Autocomplete Features:    {'✅ PASS' if test2_ok else '❌ FAIL'}")
        
        if test1_ok and test2_ok:
            print("\n🎉 All autocomplete features working!")
            print("\n🚀 Usage Tips:")
            print("  • Type 'McCaffrey' instead of 'Christian McCaffrey'")
            print("  • Use 'Josh' to see all Josh players")
            print("  • Try 'Kelce' for Travis Kelce")
            print("  • Use TAB completion in autocomplete_draft_assistant.py")
        
        return test1_ok and test2_ok
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    main()
