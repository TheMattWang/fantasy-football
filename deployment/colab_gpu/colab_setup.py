#!/usr/bin/env python3
"""
Google Colab Setup and Data Preparation
=======================================

This script sets up the complete environment for GPU-accelerated MCTS training
in Google Colab, including data loading, dependency installation, and 
environment configuration.
"""

import subprocess
import sys
import os
import zipfile
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np


def install_dependencies():
    """Install required packages for GPU-accelerated MCTS training"""
    
    print("📦 Installing dependencies for GPU-accelerated MCTS...")
    
    # Core ML packages
    packages = [
        "torch>=2.0.0",
        "torchvision",
        "torchaudio",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "tqdm>=4.60.0"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Warning: Failed to install {package}: {e}")
    
    print("✅ Dependencies installed successfully!")
    
    # Verify GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🔥 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("💻 GPU not available, will use CPU")
    except ImportError:
        print("⚠️  PyTorch not available")


def setup_colab_environment():
    """Set up the Colab environment for MCTS training"""
    
    print("🔧 Setting up Colab environment...")
    
    # Create directory structure
    directories = [
        "data",
        "models",
        "results",
        "checkpoints",
        "visualizations"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   📁 Created directory: {dir_name}")
    
    # Set up matplotlib for Colab
    try:
        import matplotlib.pyplot as plt
        plt.style.use('default')
        print("   📊 Matplotlib configured for Colab")
    except ImportError:
        print("   ⚠️  Matplotlib not available")
    
    print("✅ Environment setup complete!")


def load_sample_data():
    """Create sample fantasy football data for training"""
    
    print("📊 Creating sample fantasy football data...")
    
    # Generate realistic player data
    np.random.seed(42)
    
    positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
    position_weights = [0.1, 0.25, 0.35, 0.15, 0.05, 0.1]
    
    n_players = 400
    players_data = []
    
    for i in range(n_players):
        position = np.random.choice(positions, p=position_weights)
        
        # Generate realistic stats based on position
        if position == 'QB':
            base_ppg = np.random.gamma(3, 5) + 12  # 12-35 range
            vorp = base_ppg - 15  # QB replacement level ~15
        elif position in ['RB', 'WR']:
            base_ppg = np.random.gamma(2.5, 4) + 5   # 5-25 range
            vorp = base_ppg - 8   # Skill position replacement ~8
        elif position == 'TE':
            base_ppg = np.random.gamma(2, 3) + 4   # 4-18 range
            vorp = base_ppg - 6   # TE replacement ~6
        elif position == 'K':
            base_ppg = np.random.gamma(2, 2) + 6   # 6-14 range
            vorp = base_ppg - 7   # K replacement ~7
        else:  # DEF
            base_ppg = np.random.gamma(2, 2.5) + 5   # 5-15 range
            vorp = base_ppg - 6   # DEF replacement ~6
        
        # ADP based on performance with some noise
        adp_base = i + 1 + np.random.normal(0, 20)
        adp_rank = max(1, int(adp_base))
        
        # Bye weeks (NFL weeks 4-14)
        bye_week = np.random.choice(range(4, 15))
        
        # Injury data
        injury_risk = np.random.beta(2, 5)  # Skewed toward lower risk
        durability = 1.0 - injury_risk + np.random.normal(0, 0.1)
        durability = np.clip(durability, 0.0, 1.0)
        
        # Risk sigma (uncertainty for rookies and unknowns)
        is_rookie = np.random.random() < 0.15  # ~15% rookies
        risk_sigma = np.random.uniform(0.3, 0.8) if is_rookie else np.random.uniform(0.1, 0.3)
        
        player_data = {
            'player_name': f"Player_{i+1:03d}",
            'position': position,
            'team': f"Team_{(i % 32) + 1}",
            'vorp': float(vorp),
            'proj_ppg': float(base_ppg),
            'adp_rank': int(adp_rank),
            'bye_week': int(bye_week),
            'injury_risk_score': float(injury_risk),
            'durability_score': float(durability),
            'risk_sigma': float(risk_sigma),
            'is_rookie': bool(is_rookie),
            'historical_injuries': int(np.random.poisson(1) * injury_risk),
            'games_missed_injury': int(np.random.poisson(2) * injury_risk),
            'season_ending_injuries': int(np.random.poisson(0.3) * injury_risk),
            'avg_recovery_time': float(np.random.uniform(1, 6) * injury_risk),
            'has_recurring_injuries': bool(np.random.random() < (injury_risk * 0.5)),
            'position_injury_risk': float({
                'QB': 0.15, 'RB': 0.35, 'WR': 0.25, 'TE': 0.20, 'K': 0.05, 'DEF': 0.10
            }[position]),
            'age_adjusted_risk': float(injury_risk * np.random.uniform(0.8, 1.2)),
            'usage_adjusted_risk': float(injury_risk * np.random.uniform(0.9, 1.3)),
            'games_played_pct_adj': float(max(0.5, 1.0 - injury_risk + np.random.normal(0, 0.1)))
        }
        
        players_data.append(player_data)
    
    # Create DataFrame and save
    df = pd.DataFrame(players_data)
    df.to_csv('data/player_pool.csv', index=False)
    
    print(f"✅ Created {len(df)} players with comprehensive data")
    print(f"   Positions: {dict(df['position'].value_counts())}")
    print(f"   Average VORP: {df['vorp'].mean():.2f}")
    print(f"   Rookies: {df['is_rookie'].sum()} ({df['is_rookie'].mean():.1%})")
    
    # Create position-specific summaries
    position_summary = df.groupby('position').agg({
        'vorp': ['mean', 'std', 'max'],
        'injury_risk_score': 'mean',
        'proj_ppg': 'mean'
    }).round(2)
    
    print(f"\n📊 Position Summary:")
    print(position_summary)
    
    return df


def create_sample_draft_history():
    """Create sample draft history for pattern learning"""
    
    print("📚 Creating sample draft history...")
    
    draft_picks = []
    
    # Simulate 10 historical drafts
    for draft_id in range(10):
        pick_number = 1
        
        # 12 teams, 15 rounds
        for round_num in range(1, 16):
            for pick_in_round in range(1, 13):
                
                # Simulate position preferences by round
                if round_num <= 2:
                    position_probs = {'QB': 0.05, 'RB': 0.35, 'WR': 0.45, 'TE': 0.10, 'K': 0.01, 'DEF': 0.04}
                elif round_num <= 4:
                    position_probs = {'QB': 0.15, 'RB': 0.30, 'WR': 0.35, 'TE': 0.15, 'K': 0.01, 'DEF': 0.04}
                elif round_num <= 8:
                    position_probs = {'QB': 0.20, 'RB': 0.25, 'WR': 0.30, 'TE': 0.15, 'K': 0.02, 'DEF': 0.08}
                elif round_num <= 12:
                    position_probs = {'QB': 0.15, 'RB': 0.20, 'WR': 0.25, 'TE': 0.15, 'K': 0.10, 'DEF': 0.15}
                else:
                    position_probs = {'QB': 0.10, 'RB': 0.15, 'WR': 0.20, 'TE': 0.15, 'K': 0.20, 'DEF': 0.20}
                
                # Sample position
                position = np.random.choice(
                    list(position_probs.keys()), 
                    p=list(position_probs.values())
                )
                
                # Create mock pick
                pick = {
                    'round_num': int(round_num),
                    'pick_in_round': int(pick_in_round),
                    'overall_pick': int(pick_number),
                    'team_id': int(pick_in_round),
                    'player_name': f"HistPlayer_{draft_id}_{pick_number}",
                    'position': position,
                    'adp_rank': int(pick_number + np.random.randint(-10, 20)),
                    'vorp': float(max(0, np.random.gamma(2, 2))),
                    'bye_week': int(np.random.choice(range(4, 15))),
                    'draft_id': f"hist_draft_{draft_id}",
                    'league_type': "12team_ppr"
                }
                
                draft_picks.append(pick)
                pick_number += 1
    
    # Save draft history
    with open('data/draft_history.json', 'w') as f:
        json.dump({'picks': draft_picks}, f, indent=2)
    
    print(f"✅ Created {len(draft_picks)} historical draft picks")
    print(f"   Drafts: 10")
    print(f"   Picks per draft: {len(draft_picks) // 10}")
    
    return draft_picks


def create_league_settings():
    """Create league settings configuration"""
    
    league_settings = {
        'teams': 12,
        'roster_spots': {
            'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 
            'FLEX': 1, 'DEF': 1, 'K': 1, 'BENCH': 6
        },
        'flex_positions': ['RB', 'WR', 'TE'],
        'total_rounds': 15,
        'snake_draft': True,
        'scoring': {
            'type': 'PPR',
            'pass_yards': 0.04,
            'pass_td': 4,
            'interception': -2,
            'rush_yards': 0.1,
            'rush_td': 6,
            'rec_yards': 0.1,
            'reception': 1,
            'rec_td': 6,
            'fumble_lost': -2
        }
    }
    
    with open('data/league_settings.json', 'w') as f:
        json.dump(league_settings, f, indent=2)
    
    print("✅ League settings created")
    return league_settings


def download_colab_package():
    """Download and extract pre-built Colab package if available"""
    
    print("📥 Checking for pre-built Colab package...")
    
    # This would download from GitHub releases or Google Drive
    # For now, we'll create the package locally
    package_files = [
        'colab_mcts_trainer.py',
        'colab_setup.py'
    ]
    
    missing_files = []
    for file in package_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        print("   Creating package from local files...")
        return False
    else:
        print("✅ All package files present")
        return True


def run_system_checks():
    """Run comprehensive system checks for Colab training"""
    
    print("🔍 Running system checks...")
    
    checks = {
        'Python version': sys.version_info >= (3, 7),
        'GPU availability': False,
        'CUDA availability': False,
        'Memory (>8GB)': False,
        'Disk space (>2GB)': False
    }
    
    # Check GPU
    try:
        import torch
        checks['GPU availability'] = torch.cuda.is_available()
        checks['CUDA availability'] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"   🔥 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   💾 GPU Memory: {gpu_memory / 1e9:.1f} GB")
    except ImportError:
        pass
    
    # Check system memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / 1e9
        checks['Memory (>8GB)'] = memory_gb > 8
        print(f"   🧠 System Memory: {memory_gb:.1f} GB")
    except ImportError:
        print("   ⚠️  Cannot check system memory (psutil not available)")
    
    # Check disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / 1e9
        checks['Disk space (>2GB)'] = free_space > 2
        print(f"   💿 Free Disk Space: {free_space:.1f} GB")
    except:
        print("   ⚠️  Cannot check disk space")
    
    # Print results
    print(f"\n📋 System Check Results:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    # Recommendations
    if not checks['GPU availability']:
        print(f"\n💡 Recommendations:")
        print(f"   • Enable GPU in Colab: Runtime → Change runtime type → GPU")
        print(f"   • Training will be slower on CPU but still functional")
    
    if not checks['Memory (>8GB)']:
        print(f"   • Consider reducing batch size or model complexity")
    
    return all(checks.values())


def create_colab_notebook():
    """Create a ready-to-use Colab notebook"""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# GPU-Accelerated Fantasy Football MCTS Training\n",
                    "\n",
                    "This notebook trains a Monte Carlo Tree Search (MCTS) model for fantasy football drafts using GPU acceleration.\n",
                    "\n",
                    "## Features:\n",
                    "- 🔥 GPU-accelerated training with PyTorch\n",
                    "- 🧠 Neural network value function approximation\n",
                    "- 📊 Comprehensive draft strategy optimization\n",
                    "- 🏥 Injury risk modeling integration\n",
                    "- 📅 Bye week optimization\n",
                    "- 📚 Draft history pattern learning\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Setup and installation\n",
                    "!python colab_setup.py\n",
                    "\n",
                    "# Check GPU availability\n",
                    "import torch\n",
                    "print(f\"GPU Available: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Quick training run (recommended for first time)\n",
                    "!python colab_mcts_trainer.py --mode train --episodes 50 --gpu --batch-size 64\n",
                    "\n",
                    "# Full training run (for production model)\n",
                    "# !python colab_mcts_trainer.py --mode train --episodes 200 --gpu --batch-size 128"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# View training results\n",
                    "from IPython.display import Image, display\n",
                    "display(Image('mcts_training_report.png'))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Download trained model\n",
                    "from google.colab import files\n",
                    "\n",
                    "# Download GPU model\n",
                    "files.download('gpu_trained_mcts_model.pt')\n",
                    "\n",
                    "# Download CPU inference model\n",
                    "files.download('cpu_inference_mcts_model.pt')\n",
                    "\n",
                    "# Download training report\n",
                    "files.download('mcts_training_report.png')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open('Fantasy_Football_GPU_MCTS_Training.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✅ Colab notebook created: Fantasy_Football_GPU_MCTS_Training.ipynb")


def main():
    """Main setup function for Google Colab"""
    
    print("🚀 Google Colab GPU MCTS Setup")
    print("=" * 40)
    
    # Run system checks
    system_ok = run_system_checks()
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_colab_environment()
    
    # Create sample data
    player_data = load_sample_data()
    draft_history = create_sample_draft_history()
    league_settings = create_league_settings()
    
    # Create notebook
    create_colab_notebook()
    
    print("\n✅ Colab setup complete!")
    print("\n🚀 Next steps:")
    print("1. Open 'Fantasy_Football_GPU_MCTS_Training.ipynb' in Colab")
    print("2. Enable GPU: Runtime → Change runtime type → GPU")
    print("3. Run the training cells")
    print("4. Download your trained model!")
    
    if not system_ok:
        print("\n⚠️  Some system checks failed - training may be slower")
    
    return {
        'player_data': player_data,
        'draft_history': draft_history,
        'league_settings': league_settings,
        'system_checks': system_ok
    }


if __name__ == "__main__":
    main()
