#!/usr/bin/env python3
"""
Fantasy Football Injury Enhancement Pipeline
============================================

This script demonstrates the complete injury-aware fantasy football system:
1. Enhanced data collection with injury features
2. Injury-aware model training and comparison
3. Draft strategy integration with injury considerations
4. Comprehensive reporting and analysis

Usage:
    python run_injury_enhancement.py [--mode MODE] [--injury-weight WEIGHT]

Modes:
    - full: Complete pipeline (default)
    - data: Data enhancement only
    - model: Model training only  
    - draft: Draft strategy only
    - demo: Quick demonstration
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add models directory to path
sys.path.append(str(Path(__file__).parent / "models"))

try:
    from injury_enhancement import InjuryDataCollector, enhance_existing_model_with_injuries
    from injury_aware_model import InjuryAwareFantasyModel, run_injury_enhancement_pipeline
    from injury_data_sources import ComprehensiveInjuryDatabase
    from injury_draft_integration import InjuryAwareDraftStrategy, integrate_injury_awareness_into_existing_strategy
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all injury enhancement modules are in the models/ directory")
    sys.exit(1)

class InjuryEnhancementPipeline:
    """
    Main pipeline for injury enhancement of fantasy football models
    """
    
    def __init__(self, injury_weight: float = 0.3, output_dir: str = None):
        self.injury_weight = injury_weight
        self.output_dir = Path(output_dir or Path.cwd())
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.injury_collector = InjuryDataCollector()
        self.injury_database = ComprehensiveInjuryDatabase()
        self.results = {}
        
        print(f"🏥 Initialized Injury Enhancement Pipeline")
        print(f"   Injury Weight: {injury_weight}")
        print(f"   Output Directory: {self.output_dir}")
    
    def run_data_enhancement(self) -> pd.DataFrame:
        """
        Step 1: Enhance existing data with injury features
        """
        print("\n" + "="*60)
        print("🏗️  STEP 1: DATA ENHANCEMENT WITH INJURY FEATURES")
        print("="*60)
        
        # Load existing rookie data
        data_file = self.output_dir / "rookie_data_clean.csv"
        
        if data_file.exists():
            print(f"📊 Loading data from {data_file}")
            df = pd.read_csv(data_file)
        else:
            print("⚠️  rookie_data_clean.csv not found, creating sample data...")
            df = self._create_sample_data()
        
        print(f"   Loaded {len(df)} players")
        
        # Enhance with injury features
        enhanced_df = self.injury_collector.enhance_player_data_with_injuries(df)
        
        # Save enhanced data
        output_file = self.output_dir / "rookie_data_injury_enhanced.csv"
        enhanced_df.to_csv(output_file, index=False)
        
        print(f"💾 Enhanced data saved to {output_file}")
        
        # Generate data summary
        self._generate_data_summary(enhanced_df)
        
        self.results['enhanced_data'] = enhanced_df
        return enhanced_df
    
    def run_model_training(self, enhanced_df: pd.DataFrame = None) -> dict:
        """
        Step 2: Train injury-aware models and compare performance
        """
        print("\n" + "="*60)
        print("🧠 STEP 2: INJURY-AWARE MODEL TRAINING")
        print("="*60)
        
        if enhanced_df is None:
            enhanced_df = self.results.get('enhanced_data')
            if enhanced_df is None:
                print("⚠️  No enhanced data available, running data enhancement first...")
                enhanced_df = self.run_data_enhancement()
        
        # Initialize injury-aware model
        model = InjuryAwareFantasyModel(target_variable='ppg')
        
        # Compare traditional vs injury-aware models
        comparison_results = model.compare_traditional_vs_injury_aware(enhanced_df)
        
        # Analyze feature importance
        importance_df = model.analyze_injury_feature_importance()
        
        # Save model results
        model_output = self.output_dir / "injury_aware_model_results.csv"
        if importance_df is not None:
            importance_df.to_csv(model_output, index=False)
            print(f"📊 Model results saved to {model_output}")
        
        self.results['model_comparison'] = comparison_results
        self.results['feature_importance'] = importance_df
        self.results['trained_model'] = model
        
        return comparison_results
    
    def run_draft_integration(self, enhanced_df: pd.DataFrame = None) -> tuple:
        """
        Step 3: Integrate injury awareness into draft strategy
        """
        print("\n" + "="*60)
        print("🎯 STEP 3: INJURY-AWARE DRAFT STRATEGY")
        print("="*60)
        
        if enhanced_df is None:
            enhanced_df = self.results.get('enhanced_data')
        
        # Create mock player pool for demonstration
        player_pool = self._create_mock_player_pool(enhanced_df)
        
        # Create injury-aware strategy
        class MockBaseStrategy:
            def make_pick(self, state):
                return None
        
        base_strategy = MockBaseStrategy()
        injury_strategy = InjuryAwareDraftStrategy(base_strategy, self.injury_weight)
        
        # Create original vs injury-adjusted draft boards
        original_board = self._create_original_draft_board(player_pool)
        injury_board = injury_strategy.create_injury_draft_board(player_pool)
        
        # Analyze changes
        changes = injury_strategy.analyze_draft_board_changes(original_board, injury_board)
        
        # Save draft boards
        original_board.to_csv(self.output_dir / "original_draft_board.csv", index=False)
        injury_board.to_csv(self.output_dir / "injury_adjusted_draft_board.csv", index=False)
        changes.to_csv(self.output_dir / "draft_board_changes.csv", index=False)
        
        print(f"📋 Draft boards saved to {self.output_dir}")
        
        self.results['original_board'] = original_board
        self.results['injury_board'] = injury_board
        self.results['board_changes'] = changes
        self.results['injury_strategy'] = injury_strategy
        
        return injury_strategy, injury_board, changes
    
    def run_comprehensive_analysis(self):
        """
        Step 4: Generate comprehensive analysis and visualizations
        """
        print("\n" + "="*60)
        print("📊 STEP 4: COMPREHENSIVE ANALYSIS & VISUALIZATION")
        print("="*60)
        
        # Create analysis report
        self._create_analysis_report()
        
        # Generate visualizations
        self._create_visualizations()
        
        # Create final summary
        self._create_final_summary()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration"""
        print("🎲 Creating sample data for demonstration...")
        
        np.random.seed(42)
        n_players = 100
        
        positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        teams = ['DAL', 'NYG', 'PHI', 'WAS', 'GB', 'MIN', 'CHI', 'DET', 'KC', 'LAC']
        
        data = {
            'player_name': [f"Player_{i:03d}" for i in range(n_players)],
            'position': np.random.choice(positions, n_players, p=[0.1, 0.25, 0.35, 0.15, 0.05, 0.1]),
            'team': np.random.choice(teams, n_players),
            'age': np.random.randint(21, 30, n_players),
            'games_played': np.random.randint(8, 17, n_players),
            'ppg': np.random.gamma(2, 3),  # Right-skewed distribution
            'adp_rank': range(1, n_players + 1)
        }
        
        # Add some basic stats
        data['games_played_pct'] = data['games_played'] / 17
        data['targets'] = np.where(
            np.isin(data['position'], ['WR', 'TE']),
            np.random.poisson(5, n_players) * data['games_played'],
            0
        )
        data['rush_att'] = np.where(
            data['position'] == 'RB',
            np.random.poisson(10, n_players) * data['games_played'],
            0
        )
        
        return pd.DataFrame(data)
    
    def _create_mock_player_pool(self, df: pd.DataFrame) -> dict:
        """Create mock player pool from dataframe"""
        class MockPlayer:
            def __init__(self, row):
                self.name = row['player_name']
                self.position = row['position']
                self.vorp = row.get('ppg', 5.0)
                self.adp_rank = row.get('adp_rank', 999)
                self.age = row.get('age', 25)
                self.injury_risk_score = row.get('injury_risk_score', 0.3)
        
        return {row['player_name']: MockPlayer(row) for _, row in df.iterrows()}
    
    def _create_original_draft_board(self, player_pool: dict) -> pd.DataFrame:
        """Create original draft board sorted by VORP"""
        data = []
        for player in player_pool.values():
            data.append({
                'player_name': player.name,
                'position': player.position,
                'vorp': player.vorp,
                'adp_rank': player.adp_rank
            })
        
        return pd.DataFrame(data).sort_values('vorp', ascending=False)
    
    def _generate_data_summary(self, df: pd.DataFrame):
        """Generate summary of data enhancement"""
        print(f"\n📈 Data Enhancement Summary:")
        
        injury_features = [col for col in df.columns if 'injury' in col.lower() or 'risk' in col.lower()]
        print(f"   • Added {len(injury_features)} injury-related features")
        print(f"   • Average injury risk: {df.get('injury_risk_score', pd.Series([0.3])).mean():.3f}")
        print(f"   • Players with high injury risk (>0.6): {(df.get('injury_risk_score', pd.Series([0.3])) > 0.6).sum()}")
        
        if 'injury_tier' in df.columns or 'durability_score' in df.columns:
            print(f"   • Average durability score: {df.get('durability_score', pd.Series([0.7])).mean():.3f}")
    
    def _create_analysis_report(self):
        """Create comprehensive analysis report"""
        print("📋 Creating comprehensive analysis report...")
        
        report_lines = [
            "FANTASY FOOTBALL INJURY ENHANCEMENT ANALYSIS REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Injury Weight Used: {self.injury_weight}",
            "",
            "🏥 INJURY FEATURE ENHANCEMENT:",
        ]
        
        # Add model comparison results
        if 'model_comparison' in self.results:
            comparison = self.results['model_comparison']
            report_lines.extend([
                "",
                "🧠 MODEL PERFORMANCE COMPARISON:",
                f"Traditional Model R²: {comparison.get('Traditional', {}).get('best_r2', 0):.4f}",
                f"Injury-Aware Model R²: {comparison.get('Injury-Aware', {}).get('best_r2', 0):.4f}",
                f"Improvement: {comparison.get('Injury-Aware', {}).get('best_r2', 0) - comparison.get('Traditional', {}).get('best_r2', 0):.4f}",
            ])
        
        # Add draft strategy analysis
        if 'board_changes' in self.results:
            changes = self.results['board_changes']
            big_movers = changes[abs(changes['rank_change']) > 5]
            report_lines.extend([
                "",
                "🎯 DRAFT STRATEGY IMPACT:",
                f"Players with significant rank changes: {len(big_movers)}",
                f"Biggest mover up: {changes.loc[changes['rank_change'].idxmax(), 'player_name'] if not changes.empty else 'N/A'}",
                f"Biggest mover down: {changes.loc[changes['rank_change'].idxmin(), 'player_name'] if not changes.empty else 'N/A'}",
            ])
        
        # Save report
        report_file = self.output_dir / "injury_enhancement_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Analysis report saved to {report_file}")
    
    def _create_visualizations(self):
        """Create analysis visualizations"""
        print("📊 Creating visualizations...")
        
        try:
            # Set up plotting
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Fantasy Football Injury Enhancement Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Injury Risk Distribution
            if 'enhanced_data' in self.results:
                df = self.results['enhanced_data']
                if 'injury_risk_score' in df.columns:
                    axes[0, 0].hist(df['injury_risk_score'], bins=20, alpha=0.7, color='red')
                    axes[0, 0].set_title('Injury Risk Score Distribution')
                    axes[0, 0].set_xlabel('Injury Risk Score')
                    axes[0, 0].set_ylabel('Number of Players')
            
            # Plot 2: Risk by Position
            if 'enhanced_data' in self.results:
                df = self.results['enhanced_data']
                if 'injury_risk_score' in df.columns and 'position' in df.columns:
                    position_risk = df.groupby('position')['injury_risk_score'].mean()
                    axes[0, 1].bar(position_risk.index, position_risk.values, color='orange', alpha=0.7)
                    axes[0, 1].set_title('Average Injury Risk by Position')
                    axes[0, 1].set_xlabel('Position')
                    axes[0, 1].set_ylabel('Average Risk Score')
            
            # Plot 3: Model Performance Comparison
            if 'model_comparison' in self.results:
                comparison = self.results['model_comparison']
                models = list(comparison.keys())
                scores = [comparison[m].get('best_r2', 0) for m in models]
                axes[1, 0].bar(models, scores, color=['blue', 'green'], alpha=0.7)
                axes[1, 0].set_title('Model Performance (R² Score)')
                axes[1, 0].set_ylabel('R² Score')
            
            # Plot 4: Draft Board Changes
            if 'board_changes' in self.results:
                changes = self.results['board_changes']
                if not changes.empty and 'rank_change' in changes.columns:
                    axes[1, 1].hist(changes['rank_change'], bins=20, alpha=0.7, color='purple')
                    axes[1, 1].set_title('Draft Rank Changes Distribution')
                    axes[1, 1].set_xlabel('Rank Change (Positive = Moved Up)')
                    axes[1, 1].set_ylabel('Number of Players')
            
            plt.tight_layout()
            viz_file = self.output_dir / "injury_analysis_visualizations.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Visualizations saved to {viz_file}")
            
        except Exception as e:
            print(f"⚠️  Visualization creation failed: {e}")
    
    def _create_final_summary(self):
        """Create final summary of pipeline results"""
        print("\n" + "="*60)
        print("🏆 INJURY ENHANCEMENT PIPELINE SUMMARY")
        print("="*60)
        
        print(f"✅ Data Enhancement: {len(self.results.get('enhanced_data', []))} players processed")
        
        if 'model_comparison' in self.results:
            comparison = self.results['model_comparison']
            improvement = (comparison.get('Injury-Aware', {}).get('best_r2', 0) - 
                          comparison.get('Traditional', {}).get('best_r2', 0))
            print(f"✅ Model Training: {improvement:.4f} R² improvement from injury features")
        
        if 'board_changes' in self.results:
            changes = self.results['board_changes']
            significant_changes = len(changes[abs(changes['rank_change']) > 5]) if not changes.empty else 0
            print(f"✅ Draft Integration: {significant_changes} players with significant rank changes")
        
        print(f"\n📁 Output Files Generated:")
        output_files = list(self.output_dir.glob("*.csv")) + list(self.output_dir.glob("*.txt")) + list(self.output_dir.glob("*.png"))
        for file in output_files:
            print(f"   • {file.name}")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review injury_enhancement_report.txt for detailed analysis")
        print(f"   2. Use injury_adjusted_draft_board.csv for drafts")
        print(f"   3. Integrate injury_aware_model.py into your draft strategy")
        print(f"   4. Consider adjusting injury_weight parameter (current: {self.injury_weight})")
        
        print(f"\n🏥 Injury Enhancement Pipeline Complete! 🏆")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Fantasy Football Injury Enhancement Pipeline")
    parser.add_argument('--mode', choices=['full', 'data', 'model', 'draft', 'demo'], 
                       default='demo', help='Pipeline mode to run')
    parser.add_argument('--injury-weight', type=float, default=0.3,
                       help='Weight for injury considerations (0.0-1.0)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🏥 Fantasy Football Injury Enhancement Pipeline")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Injury Weight: {args.injury_weight}")
    
    # Initialize pipeline
    output_dir = args.output_dir or Path.cwd()
    pipeline = InjuryEnhancementPipeline(args.injury_weight, output_dir)
    
    try:
        if args.mode == 'full':
            # Run complete pipeline
            enhanced_df = pipeline.run_data_enhancement()
            pipeline.run_model_training(enhanced_df)
            pipeline.run_draft_integration(enhanced_df)
            pipeline.run_comprehensive_analysis()
            
        elif args.mode == 'data':
            # Data enhancement only
            pipeline.run_data_enhancement()
            
        elif args.mode == 'model':
            # Model training only
            enhanced_df = pipeline.run_data_enhancement()
            pipeline.run_model_training(enhanced_df)
            
        elif args.mode == 'draft':
            # Draft strategy only
            enhanced_df = pipeline.run_data_enhancement()
            pipeline.run_draft_integration(enhanced_df)
            
        elif args.mode == 'demo':
            # Quick demonstration
            print("\n🎲 Running quick demonstration...")
            enhanced_df = pipeline.run_data_enhancement()
            pipeline.run_model_training(enhanced_df)
            pipeline.run_draft_integration(enhanced_df)
            pipeline.run_comprehensive_analysis()
            
    except KeyboardInterrupt:
        print("\n⛔ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
