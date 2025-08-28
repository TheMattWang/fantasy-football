#!/usr/bin/env python3
"""
Enhanced Colab Data Loader with Injury Awareness
================================================

This module loads fantasy football data and provides easy access to both
traditional MCTS and injury-aware MCTS functionality in Google Colab.
"""

import pandas as pd
import json
import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_colab_data():
    """
    Load all fantasy football data and models for Colab environment
    
    Returns:
        dict: Complete data package with all necessary components
    """
    
    print("🏈 Loading Fantasy Football MCTS with Injury Awareness")
    print("=" * 60)
    
    # Load metadata
    try:
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        print(f"✅ Package: {metadata['package_info']['name']} v{metadata['package_info']['version']}")
        print(f"📅 Created: {metadata['package_info']['created']}")
        print(f"🔧 Features: {', '.join(metadata['package_info']['features'])}")
    except FileNotFoundError:
        print("⚠️  metadata.json not found, using defaults")
        metadata = {}
    
    # Load core data
    data = {}
    
    # 1. Draft board
    try:
        data['draft_board'] = pd.read_csv('draft_board.csv')
        print(f"✅ Loaded draft board: {len(data['draft_board'])} players")
    except FileNotFoundError:
        print("⚠️  draft_board.csv not found")
        data['draft_board'] = None
    
    # 2. ADP rankings
    try:
        data['adp_rankings'] = pd.read_csv('FantasyPros_2025_Overall_ADP_Rankings.csv')
        print(f"✅ Loaded ADP rankings: {len(data['adp_rankings'])} players")
    except FileNotFoundError:
        print("⚠️  ADP rankings not found")
        data['adp_rankings'] = None
    
    # 3. Rookie data
    try:
        data['rookie_data'] = pd.read_csv('rookie_data_clean.csv')
        print(f"✅ Loaded rookie data: {len(data['rookie_data'])} players")
    except FileNotFoundError:
        print("⚠️  rookie_data_clean.csv not found")
        data['rookie_data'] = None
    
    # 4. Rookie model
    try:
        with open('rookie_regressor.pkl', 'rb') as f:
            data['rookie_model'] = pickle.load(f)
        print("✅ Loaded rookie regression model")
    except FileNotFoundError:
        print("⚠️  rookie_regressor.pkl not found")
        data['rookie_model'] = None
    
    # 5. Enhanced injury data
    try:
        data['injury_enhanced_data'] = pd.read_csv('injury_enhanced_demo.csv')
        print(f"✅ Loaded injury-enhanced data: {len(data['injury_enhanced_data'])} players")
        print(f"   🏥 Injury features: {metadata.get('injury_enhancement', {}).get('enabled', False)}")
    except FileNotFoundError:
        print("⚠️  injury_enhanced_demo.csv not found")
        data['injury_enhanced_data'] = None
    
    # 6. Check for injury modules
    injury_modules_available = []
    injury_files = [
        'mcts_injury_wrapper.py',
        'injury_aware_mcts.py', 
        'injury_enhancement.py',
        'injury_aware_model.py',
        'injury_demo.py'
    ]
    
    for file in injury_files:
        if Path(file).exists():
            injury_modules_available.append(file)
    
    if injury_modules_available:
        print(f"✅ Injury modules available: {len(injury_modules_available)}/5")
        data['injury_modules'] = injury_modules_available
    else:
        print("⚠️  No injury modules found")
        data['injury_modules'] = []
    
    # Store metadata
    data['metadata'] = metadata
    
    print(f"\n🎯 Package Status:")
    print(f"   📊 Core MCTS: {'✅' if data['draft_board'] is not None else '❌'}")
    print(f"   🏥 Injury Features: {'✅' if data['injury_enhanced_data'] is not None else '❌'}")
    print(f"   🤖 Rookie Model: {'✅' if data['rookie_model'] is not None else '❌'}")
    print(f"   ⚡ Injury Modules: {'✅' if injury_modules_available else '❌'}")
    
    return data

def show_integration_options():
    """Display available integration options"""
    
    print("\n🔧 Integration Options:")
    print("=" * 30)
    
    print("\n1️⃣  SIMPLE INTEGRATION (Recommended)")
    print("   Add injury awareness to existing MCTS with one line:")
    print("   ```python")
    print("   from mcts_injury_wrapper import add_injury_awareness_to_existing_system")
    print("   enhanced_pool, enhanced_reward_fn = add_injury_awareness_to_existing_system(")
    print("       player_pool, reward_function, injury_weight=0.3")
    print("   )")
    print("   ```")
    
    print("\n2️⃣  FULL INJURY-AWARE SYSTEM")
    print("   Use complete injury-aware MCTS:")
    print("   ```python") 
    print("   from injury_aware_mcts import InjuryAwareDraftStrategy, create_injury_aware_player_pool")
    print("   injury_aware_pool = create_injury_aware_player_pool(basic_player_pool, 'injury_enhanced_demo.csv')")
    print("   strategy = InjuryAwareDraftStrategy(injury_aware_pool)")
    print("   ```")
    
    print("\n3️⃣  CUSTOM INTEGRATION")
    print("   Manual control over injury features:")
    print("   ```python")
    print("   from mcts_injury_wrapper import enhance_player_pool_with_injury_data")
    print("   enhanced_pool = enhance_player_pool_with_injury_data(player_pool, injury_weight=0.3)")
    print("   ```")

def show_demo_options():
    """Display available demos"""
    
    print("\n🚀 Available Demos:")
    print("=" * 20)
    
    demos = [
        ("injury_demo.py", "Basic injury enhancement demonstration"),
        ("injury_integration_demo.py", "Complete MCTS integration demo"),
        ("strategy_comparison", "Compare traditional vs injury-aware MCTS"),
        ("draft_board_analysis", "Analyze injury impact on player rankings")
    ]
    
    for i, (demo, description) in enumerate(demos, 1):
        print(f"{i}. {demo}")
        print(f"   {description}")
        print()

def get_injury_settings():
    """Display current injury settings"""
    
    try:
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        
        injury_params = metadata.get('model_parameters', {})
        injury_config = metadata.get('injury_enhancement', {})
        
        print("\n⚙️  Current Injury Settings:")
        print("=" * 30)
        print(f"   🏥 Injury Features Enabled: {injury_config.get('enabled', False)}")
        print(f"   ⚖️  Injury Weight: {injury_params.get('injury_weight', 0.3)}")
        print(f"   ❌ Injury Penalty: {injury_params.get('injury_penalty', 0.3)}")
        print(f"   ✅ Durability Bonus: {injury_params.get('durability_bonus', 0.2)}")
        print(f"   🎯 Diversification Bonus: {injury_params.get('diversification_bonus', 0.1)}")
        
        print(f"\n🎚️  Risk Tiers:")
        risk_tiers = injury_config.get('risk_tiers', {})
        for tier, range_str in risk_tiers.items():
            print(f"   {tier.replace('_', ' ').title()}: {range_str}")
        
        print(f"\n📊 Available Features ({len(injury_config.get('features', []))}):")
        for feature in injury_config.get('features', [])[:5]:  # Show first 5
            print(f"   • {feature}")
        if len(injury_config.get('features', [])) > 5:
            print(f"   • ... and {len(injury_config.get('features', [])) - 5} more")
        
    except (FileNotFoundError, json.JSONDecodeError):
        print("⚠️  Could not load injury settings")

def quick_start():
    """Show quick start instructions"""
    
    print("\n🚀 Quick Start Guide:")
    print("=" * 20)
    
    print("\n1. Load data:")
    print("   data = load_colab_data()")
    
    print("\n2. Choose integration method:")
    print("   show_integration_options()  # See all options")
    
    print("\n3. Run a demo:")
    print("   !python injury_demo.py")
    
    print("\n4. Or start drafting:")
    print("   # Use your preferred integration method")
    print("   # Then run MCTS draft strategy")

if __name__ == "__main__":
    # Auto-run when imported
    data = load_colab_data()
    
    # Show injury settings if available
    get_injury_settings()
    
    # Show options
    show_integration_options()
    show_demo_options()
    quick_start()
    
    print(f"\n🎉 Ready to draft with injury-aware MCTS!")
    print(f"💡 Use load_colab_data() to get all data, or run show_integration_options() for help")
