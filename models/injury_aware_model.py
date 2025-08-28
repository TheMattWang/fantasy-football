import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the injury enhancement system
from injury_enhancement import InjuryDataCollector

class InjuryAwareFantasyModel:
    """
    Enhanced fantasy football model that incorporates comprehensive injury data
    """
    
    def __init__(self, target_variable='ppg', include_injury_features=True):
        self.target_variable = target_variable
        self.include_injury_features = include_injury_features
        self.injury_collector = InjuryDataCollector()
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
        # Enhanced feature groups including injury features
        self.feature_groups = {
            'draft_capital': ['round', 'pick', 'early_round', 'first_round', 'day1_pick', 'day2_pick'],
            'player_attributes': ['age', 'is_qb', 'is_rb', 'is_wr', 'is_te'],
            'team_context': ['good_team', 'good_offense', 'bad_offense'],
            'opportunity': ['games_played_pct', 'target_share', 'rush_share', 'starter_games'],
            'efficiency': ['yards_per_target', 'yards_per_carry'],
            'injury_risk': [
                'injury_risk_score', 'durability_score', 'historical_injuries',
                'games_missed_injury', 'season_ending_injuries', 'avg_recovery_time',
                'has_recurring_injuries', 'position_injury_risk', 'age_adjusted_risk',
                'usage_adjusted_risk', 'games_played_pct_adj'
            ]
        }
    
    def prepare_injury_enhanced_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with injury enhancements
        """
        print("🏥 Preparing injury-enhanced dataset...")
        
        # First enhance with injury features
        enhanced_df = self.injury_collector.enhance_player_data_with_injuries(df)
        
        return enhanced_df
    
    def prepare_features(self, df: pd.DataFrame, feature_selection='all_with_injury') -> tuple:
        """
        Prepare features including injury data
        """
        print(f"Preparing features for {len(df)} players...")
        
        # Define feature sets
        if feature_selection == 'injury_only':
            selected_features = self.feature_groups['injury_risk']
        elif feature_selection == 'traditional_only':
            selected_features = []
            for group in ['draft_capital', 'player_attributes', 'team_context', 'opportunity', 'efficiency']:
                selected_features.extend(self.feature_groups[group])
        elif feature_selection == 'all_with_injury':
            selected_features = []
            for group in self.feature_groups.values():
                selected_features.extend(group)
        else:  # 'draft_and_injury'
            selected_features = self.feature_groups['draft_capital'] + self.feature_groups['injury_risk']
        
        # Filter for available features
        available_features = [f for f in selected_features if f in df.columns]
        self.feature_names = available_features
        
        # Debug: Check for feature count mismatch
        print(f"   Selected features: {len(selected_features)}")
        print(f"   Available features: {len(available_features)}")
        if len(selected_features) != len(available_features):
            missing_features = set(selected_features) - set(available_features)
            print(f"   Missing features: {missing_features}")
        
        print(f"Using {len(available_features)} features:")
        for group_name, group_features in self.feature_groups.items():
            group_available = [f for f in group_features if f in available_features]
            if group_available:
                print(f"  • {group_name}: {group_available}")
        
        # Prepare feature matrix
        X = df[available_features].copy()
        y = df[self.target_variable].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Remove samples with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        return X, y
    
    def train_injury_aware_models(self, X, y, scale_features=True):
        """
        Train models with injury-enhanced features
        """
        print("\n🧠 Training injury-aware models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features if requested
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        # Define models optimized for injury prediction
        model_configs = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1)
        }
        
        results = {}
        
        for name, model in model_configs.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std()
            }
        
        self.models = results
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        return results
    
    def compare_traditional_vs_injury_aware(self, df: pd.DataFrame):
        """
        Compare traditional model vs injury-aware model performance
        """
        print("\n🆚 Comparing Traditional vs Injury-Aware Models...")
        
        results_comparison = {}
        
        # Train traditional model (without injury features)
        print("\n📊 Training Traditional Model...")
        X_trad, y_trad = self.prepare_features(df, 'traditional_only')
        trad_results = self.train_injury_aware_models(X_trad, y_trad)
        results_comparison['Traditional'] = {
            'best_r2': max([r['test_r2'] for r in trad_results.values()]),
            'best_model': max(trad_results.keys(), key=lambda x: trad_results[x]['test_r2']),
            'features': len(self.feature_names)
        }
        
        # Train injury-aware model
        print("\n🏥 Training Injury-Aware Model...")
        df_enhanced = self.prepare_injury_enhanced_data(df)
        X_injury, y_injury = self.prepare_features(df_enhanced, 'all_with_injury')
        injury_results = self.train_injury_aware_models(X_injury, y_injury)
        results_comparison['Injury-Aware'] = {
            'best_r2': max([r['test_r2'] for r in injury_results.values()]),
            'best_model': max(injury_results.keys(), key=lambda x: injury_results[x]['test_r2']),
            'features': len(self.feature_names)
        }
        
        # Print comparison
        print("\n🏆 MODEL COMPARISON RESULTS:")
        print("=" * 50)
        for model_type, results in results_comparison.items():
            print(f"{model_type:15s}: R² = {results['best_r2']:.4f}, "
                  f"Best = {results['best_model']}, Features = {results['features']}")
        
        improvement = results_comparison['Injury-Aware']['best_r2'] - results_comparison['Traditional']['best_r2']
        print(f"\n📈 Improvement from injury features: {improvement:.4f} R² ({improvement/results_comparison['Traditional']['best_r2']*100:.1f}%)")
        
        return results_comparison
    
    def analyze_injury_feature_importance(self):
        """
        Analyze the importance of injury-related features
        """
        if self.best_model is None:
            print("No model trained yet!")
            return
        
        print(f"\n🔍 Injury Feature Importance Analysis ({self.best_model_name}):")
        print("=" * 60)
        
        # Get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_)
        else:
            print("Cannot extract feature importance from this model type.")
            return
        
        # Ensure arrays are same length
        if len(importance) != len(self.feature_names):
            print(f"⚠️  Length mismatch: {len(importance)} importance values vs {len(self.feature_names)} features")
            # Truncate to minimum length
            min_len = min(len(importance), len(self.feature_names))
            importance = importance[:min_len]
            feature_names = self.feature_names[:min_len]
        else:
            feature_names = self.feature_names
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'feature_group': [self._get_feature_group(f) for f in feature_names]
        }).sort_values('importance', ascending=False)
        
        # Show top features overall
        print("\nTop 15 Most Important Features:")
        print("-" * 40)
        for i, row in importance_df.head(15).iterrows():
            group_emoji = self._get_group_emoji(row['feature_group'])
            print(f"{group_emoji} {row['feature']:25s}: {row['importance']:.4f}")
        
        # Show injury feature rankings
        injury_features = importance_df[importance_df['feature_group'] == 'injury_risk']
        if not injury_features.empty:
            print(f"\n🏥 Injury Feature Rankings:")
            print("-" * 30)
            for i, row in injury_features.iterrows():
                rank = importance_df.index.get_loc(i) + 1
                print(f"#{rank:2d} {row['feature']:25s}: {row['importance']:.4f}")
        
        # Group-level importance
        group_importance = importance_df.groupby('feature_group')['importance'].sum().sort_values(ascending=False)
        print(f"\n📊 Feature Group Importance:")
        print("-" * 30)
        for group, total_importance in group_importance.items():
            emoji = self._get_group_emoji(group)
            pct = (total_importance / importance_df['importance'].sum()) * 100
            print(f"{emoji} {group:15s}: {total_importance:.4f} ({pct:.1f}%)")
        
        return importance_df
    
    def _get_feature_group(self, feature_name: str) -> str:
        """Determine which group a feature belongs to"""
        for group_name, features in self.feature_groups.items():
            if feature_name in features:
                return group_name
        return 'other'
    
    def _get_group_emoji(self, group_name: str) -> str:
        """Get emoji for feature group"""
        emojis = {
            'draft_capital': '🎯',
            'player_attributes': '👤',
            'team_context': '🏈',
            'opportunity': '⚡',
            'efficiency': '📈',
            'injury_risk': '🏥',
            'other': '❓'
        }
        return emojis.get(group_name, '❓')
    
    def predict_with_injury_risk(self, player_data: Dict) -> Dict:
        """
        Make injury-aware predictions for a player
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Calculate injury features for the player
        injury_features = self.injury_collector.calculate_injury_risk_score(player_data)
        
        # Create feature vector (simplified for demo)
        feature_vector = np.zeros(len(self.feature_names))
        
        # Fill in available features
        # This would need to be properly implemented based on your feature mapping
        
        # Make prediction
        prediction = self.best_model.predict([feature_vector])[0]
        
        # Calculate injury-adjusted prediction
        injury_risk = injury_features
        injury_adjusted_prediction = prediction * (1.0 - injury_risk * 0.2)  # Reduce by up to 20% for high injury risk
        
        return {
            'base_prediction': prediction,
            'injury_risk': injury_risk,
            'injury_adjusted_prediction': injury_adjusted_prediction,
            'confidence': 1.0 - injury_risk  # Lower confidence for injury-prone players
        }
    
    def create_injury_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive injury risk report for all players
        """
        print("📋 Creating Injury Risk Report...")
        
        enhanced_df = self.prepare_injury_enhanced_data(df)
        
        # Create report with key injury metrics
        report_columns = [
            'player_name', 'position', 'age', 'ppg',
            'injury_risk_score', 'durability_score', 'historical_injuries',
            'games_missed_injury', 'has_recurring_injuries'
        ]
        
        available_columns = [col for col in report_columns if col in enhanced_df.columns]
        injury_report = enhanced_df[available_columns].copy()
        
        # Add risk categories
        injury_report['risk_category'] = pd.cut(
            injury_report['injury_risk_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Sort by injury risk
        injury_report = injury_report.sort_values('injury_risk_score')
        
        return injury_report

def run_injury_enhancement_pipeline():
    """
    Complete pipeline to enhance fantasy model with injury data
    """
    print("🚀 Starting Injury-Aware Fantasy Model Pipeline...")
    
    # Load existing data
    df = pd.read_csv('/Users/mattwang/Documents/fantasy/fantasy-football/rookie_data_clean.csv')
    print(f"📊 Loaded {len(df)} players")
    
    # Initialize injury-aware model
    model = InjuryAwareFantasyModel(target_variable='ppg')
    
    # Compare traditional vs injury-aware models
    comparison_results = model.compare_traditional_vs_injury_aware(df)
    
    # Analyze feature importance
    importance_df = model.analyze_injury_feature_importance()
    
    # Create injury risk report
    enhanced_df = model.prepare_injury_enhanced_data(df)
    injury_report = model.create_injury_report(df)
    
    # Save results
    enhanced_df.to_csv('/Users/mattwang/Documents/fantasy/fantasy-football/rookie_data_injury_enhanced.csv', index=False)
    injury_report.to_csv('/Users/mattwang/Documents/fantasy/fantasy-football/injury_risk_report.csv', index=False)
    
    print("\n✅ Injury Enhancement Pipeline Complete!")
    print(f"📁 Enhanced data saved to: rookie_data_injury_enhanced.csv")
    print(f"📋 Injury report saved to: injury_risk_report.csv")
    
    return model, comparison_results, importance_df, injury_report

if __name__ == "__main__":
    # Run the complete pipeline
    model, results, importance, report = run_injury_enhancement_pipeline()
