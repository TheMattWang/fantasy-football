
# Colab Data Loader
# Run this cell first in your Colab notebook

import pandas as pd
import json
import os

print("🔄 Loading data files...")

# Declare global variables 
global draft_board, adp_data, rookie_data, metadata
draft_board = None
adp_data = None
rookie_data = None
metadata = None

try:
    # Check if data files exist
    if os.path.exists('metadata.json'):
        print("✅ Data files detected!")
        
        # Load metadata
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        print("📋 Loaded metadata")
        
        # Load your actual data
        if os.path.exists('draft_board.csv'):
            draft_board = pd.read_csv('draft_board.csv')
            print(f"📊 Loaded draft_board.csv: {len(draft_board)} players")
        else:
            print("⚠️  draft_board.csv not found")
        
        if os.path.exists('FantasyPros_2025_Overall_ADP_Rankings.csv'):
            adp_data = pd.read_csv('FantasyPros_2025_Overall_ADP_Rankings.csv')
            print(f"📈 Loaded ADP data: {len(adp_data)} players")
        else:
            print("⚠️  FantasyPros_2025_Overall_ADP_Rankings.csv not found")
        
        if os.path.exists('rookie_data_clean.csv'):
            rookie_data = pd.read_csv('rookie_data_clean.csv')
            print(f"🏈 Loaded rookie data: {len(rookie_data)} players")
        else:
            print("💡 rookie_data_clean.csv not found (optional)")
        
        # Load model files if available
        model_found = False
        for model_name in ['best_rookie_model.pkl', 'rookie_fantasy_model.pkl', 'rookie_regressor.pkl']:
            if os.path.exists(model_name):
                print(f"🤖 Found rookie prediction model: {model_name}")
                model_found = True
                break
        
        if not model_found:
            print(f"📊 No rookie model found - will use position-based uncertainty")
        
        # Load your preferred settings
        LEAGUE_SETTINGS = metadata.get('league_settings', {'teams': 12})
        MODEL_PARAMS = metadata.get('model_parameters', {})
        
        # Verify critical data is loaded
        if draft_board is not None and adp_data is not None:
            print("🎯 All critical data loaded successfully!")
        else:
            print("❌ Missing critical data files")
            
    else:
        print("⚠️  metadata.json not found")
        print("📝 Files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.csv') or f.endswith('.pkl') or f.endswith('.json'):
                print(f"   • {f}")
        
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("📝 Please check file integrity and try again")
