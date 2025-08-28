
# Colab Data Loader
# Run this cell first in your Colab notebook

import pandas as pd
import json
import zipfile
import os

# Check if data files exist
if os.path.exists('metadata.json'):
    print("✅ Data files detected!")
    
    # Load metadata
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load your actual data
    if os.path.exists('draft_board.csv'):
        draft_board = pd.read_csv('draft_board.csv')
        print(f"📊 Loaded draft_board.csv: {len(draft_board)} players")
    
    if os.path.exists('FantasyPros_2025_Overall_ADP_Rankings.csv'):
        adp_data = pd.read_csv('FantasyPros_2025_Overall_ADP_Rankings.csv')
        print(f"📈 Loaded ADP data: {len(adp_data)} players")
    
    if os.path.exists('rookie_data_clean.csv'):
        rookie_data = pd.read_csv('rookie_data_clean.csv')
        print(f"🏈 Loaded rookie data: {len(rookie_data)} players")
    
    # Load model files if available
    if os.path.exists('best_rookie_model.pkl'):
        print(f"🤖 Found rookie prediction model: best_rookie_model.pkl")
    elif os.path.exists('rookie_fantasy_model.pkl'):
        print(f"🤖 Found rookie prediction model: rookie_fantasy_model.pkl")
    else:
        print(f"📊 No rookie model found - will use position-based uncertainty")
    
    # Load your preferred settings
    LEAGUE_SETTINGS = metadata['league_settings']
    MODEL_PARAMS = metadata['model_parameters']
    
    print("🎯 Your custom settings loaded successfully!")
    
else:
    print("⚠️  No data files found - notebook will use sample data")
