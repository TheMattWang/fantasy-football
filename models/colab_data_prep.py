"""
Colab Data Preparation Script
=============================

Run this script locally to prepare all your data files for uploading to Google Colab.
This creates a single ZIP file with everything you need for the MCTS draft strategy notebook.
"""

import pandas as pd
import pickle
import zipfile
import os
from pathlib import Path

def prepare_colab_data():
    """Prepare all necessary data files for Colab upload"""
    
    print("🔄 Preparing data for Google Colab...")
    
    # Create output directory
    output_dir = Path("colab_upload")
    output_dir.mkdir(exist_ok=True)
    
    files_to_include = []
    
    # 1. Copy main data files
    data_files = [
        'draft_board.csv',
        'FantasyPros_2025_Overall_ADP_Rankings.csv',
        'rookie_data_clean.csv'
    ]
    
    # 1.5. Copy model files if they exist (check multiple locations)
    model_files = [
        'best_rookie_model.pkl',
        'rookie_fantasy_model.pkl', 
        'best_draft_strategy_model.pkl',
        'models/best_rookie_model.pkl',
        'models/rookie_fantasy_model.pkl',
        'model_weights/rookie_regressor.pkl'  # Found this one!
    ]
    
    for file in data_files:
        if os.path.exists(file):
            # Copy to output directory
            df = pd.read_csv(file)
            output_path = output_dir / file
            df.to_csv(output_path, index=False)
            files_to_include.append(file)
            print(f"✅ Copied {file} ({len(df)} rows)")
        else:
            print(f"⚠️  {file} not found - will use sample data in Colab")
    
    # Copy model files
    for file in model_files:
        if os.path.exists(file):
            import shutil
            # Extract just the filename for the destination
            filename = os.path.basename(file)
            destination = output_dir / filename
            shutil.copy(file, destination)
            files_to_include.append(filename)  # Only include the filename, not the path
            print(f"✅ Copied model {file} -> {filename}")
        else:
            print(f"💡 {file} not found - run the respective notebook to generate it")
    
    # 2. Create a metadata file with all your settings
    metadata = {
        'league_settings': {
            'teams': 12,
            'roster_spots': {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DEF': 1, 'K': 1, 'BENCH': 6},
            'total_rounds': 15
        },
        'model_parameters': {
            'risk_penalty': 0.1,
            'overstack_penalty': 0.5,
            'bye_penalty': 0.2,
            'mcts_simulations': 200,
            'opponent_temperature': 0.5,
            'position_run_prob': 0.1
        },
        'files_included': files_to_include,
        'instructions': [
            "1. Upload this ZIP to Google Colab",
            "2. Extract files in Colab with: !unzip colab_data.zip",
            "3. Run the MCTS draft strategy notebook",
            "4. The notebook will automatically detect and load these files"
        ]
    }
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 3. Create a simple data loader for Colab
    colab_loader = '''
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
'''
    
    loader_path = output_dir / 'colab_loader.py'
    with open(loader_path, 'w') as f:
        f.write(colab_loader)
    
    # 4. Create ZIP file for easy upload
    zip_path = 'colab_data.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_dir.glob('*'):
            if file_path.is_file():  # Only add files, not directories
                zipf.write(file_path, file_path.name)
                print(f"   ✅ Added {file_path.name} to ZIP")
    
    print(f"\n📦 Created {zip_path} with all your data!")
    print(f"📁 Contents:")
    for file in files_to_include + ['metadata.json', 'colab_loader.py']:
        print(f"   • {file}")
    
    print(f"\n🚀 COLAB INSTRUCTIONS:")
    print(f"1. Upload {zip_path} to Google Colab")
    print(f"2. Run: !unzip {zip_path}")
    print(f"3. Run: exec(open('colab_loader.py').read())")
    print(f"4. Your data will be ready for the MCTS notebook!")
    
    return zip_path

if __name__ == "__main__":
    prepare_colab_data()
