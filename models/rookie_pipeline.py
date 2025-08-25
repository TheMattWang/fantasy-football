"""
Production-ready rookie prediction pipeline
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

class RookiePredictionPipeline:
    """
    Production pipeline for rookie fantasy football predictions
    """
    
    def __init__(self, model_path: str = 'best_rookie_model.pkl'):
        """
        Initialize the pipeline with a trained model
        
        Args:
            model_path: Path to the saved model pickle file
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_name = None
        self.performance_metrics = None
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the trained model and components"""
        try:
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            self.model = model_package['model']
            self.scaler = model_package.get('scaler')
            self.feature_names = model_package['feature_names']
            self.model_name = model_package['model_name']
            self.performance_metrics = model_package.get('performance_metrics', {})
            
            print(f"✅ Loaded model: {self.model_name}")
            print(f"📊 Model performance: R² = {self.performance_metrics.get('r2_mean', 'N/A'):.4f}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def prepare_features(self, rookie_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare rookie data for prediction
        
        Args:
            rookie_data: DataFrame with rookie information
            
        Returns:
            Prepared feature matrix
        """
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(rookie_data.columns)
        if missing_features:
            print(f"⚠️  Missing features (will be filled with 0): {missing_features}")
        
        # Create feature matrix with all required features
        X = pd.DataFrame(0, index=rookie_data.index, columns=self.feature_names)
        
        # Fill in available features
        for feature in self.feature_names:
            if feature in rookie_data.columns:
                X[feature] = rookie_data[feature]
        
        # Fill missing values
        X = X.fillna(0)
        
        return X
    
    def predict(self, rookie_data: pd.DataFrame, return_details: bool = False) -> Union[np.ndarray, Dict]:
        """
        Predict fantasy performance for rookies
        
        Args:
            rookie_data: DataFrame with rookie information
            return_details: If True, return additional prediction details
            
        Returns:
            Predictions array or dict with details
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare features
        X = self.prepare_features(rookie_data)
        
        # Scale features if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        if return_details:
            # Create success probability based on PPG thresholds
            success_prob = (predictions >= 10).astype(float)
            elite_prob = (predictions >= 15).astype(float)
            
            return {
                'ppg_predictions': predictions,
                'success_probability': success_prob,
                'elite_probability': elite_prob,
                'model_used': self.model_name,
                'feature_count': len(self.feature_names)
            }
        
        return predictions
    
    def predict_single_rookie(self, rookie_info: Dict) -> Dict:
        """
        Predict for a single rookie
        
        Args:
            rookie_info: Dictionary with rookie information
            
        Returns:
            Prediction results
        """
        rookie_df = pd.DataFrame([rookie_info])
        results = self.predict(rookie_df, return_details=True)
        
        return {
            'predicted_ppg': float(results['ppg_predictions'][0]),
            'success_probability': float(results['success_probability'][0]),
            'elite_probability': float(results['elite_probability'][0]),
            'performance_tier': self._get_performance_tier(results['ppg_predictions'][0])
        }
    
    def _get_performance_tier(self, ppg: float) -> str:
        """Categorize performance tier based on PPG"""
        if ppg >= 15:
            return "Elite"
        elif ppg >= 10:
            return "Starter"
        elif ppg >= 5:
            return "Flex"
        else:
            return "Bench"
    
    def batch_predict(self, rookie_list: List[Dict]) -> pd.DataFrame:
        """
        Predict for multiple rookies
        
        Args:
            rookie_list: List of rookie info dictionaries
            
        Returns:
            DataFrame with predictions
        """
        rookie_df = pd.DataFrame(rookie_list)
        results = self.predict(rookie_df, return_details=True)
        
        # Create results DataFrame
        results_df = rookie_df.copy()
        results_df['predicted_ppg'] = results['ppg_predictions']
        results_df['success_probability'] = results['success_probability']
        results_df['elite_probability'] = results['elite_probability']
        results_df['performance_tier'] = [self._get_performance_tier(ppg) for ppg in results['ppg_predictions']]
        
        # Sort by predicted PPG
        results_df = results_df.sort_values('predicted_ppg', ascending=False).reset_index(drop=True)
        
        return results_df
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'uses_scaling': self.scaler is not None
        }

# Example usage functions
def create_rookie_features(name: str, position: str, round_num: int, pick: int, 
                          age: int = 22, team_quality: str = "average") -> Dict:
    """
    Helper function to create rookie feature dictionary
    
    Args:
        name: Player name
        position: Position (QB, RB, WR, TE)
        round_num: Draft round
        pick: Draft pick number
        age: Player age
        team_quality: "good", "average", or "bad"
    """
    # Position indicators
    pos_features = {f'is_{pos.lower()}': 0 for pos in ['QB', 'RB', 'WR', 'TE']}
    pos_features[f'is_{position.lower()}'] = 1
    
    # Draft capital features
    draft_features = {
        'round': round_num,
        'pick': pick,
        'early_round': 1 if round_num <= 3 else 0,
        'first_round': 1 if round_num == 1 else 0,
        'day1_pick': 1 if pick <= 32 else 0,
        'day2_pick': 1 if 32 < pick <= 96 else 0,
        'age': age
    }
    
    # Team context features
    team_features = {
        'good_team': 1 if team_quality == "good" else 0,
        'good_offense': 1 if team_quality == "good" else 0,
        'bad_offense': 1 if team_quality == "bad" else 0
    }
    
    # Default opportunity/efficiency features (to be updated with actual data)
    opportunity_features = {
        'games_played_pct': 0.8,  # Assume 80% games played
        'target_share': 0.1 if position in ['WR', 'TE'] else 0.0,
        'rush_share': 0.15 if position == 'RB' else 0.0,
        'starter_games': 1 if round_num <= 3 else 0,
        'yards_per_target': 8.0 if position in ['WR', 'TE'] else 0.0,
        'yards_per_carry': 4.0 if position == 'RB' else 0.0
    }
    
    # Combine all features
    rookie_info = {'player_name': name, 'position': position}
    rookie_info.update(pos_features)
    rookie_info.update(draft_features)
    rookie_info.update(team_features)
    rookie_info.update(opportunity_features)
    
    return rookie_info

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize pipeline
        pipeline = RookiePredictionPipeline('best_rookie_model.pkl')
        
        # Example rookies
        example_rookies = [
            create_rookie_features("Caleb Williams", "QB", 1, 1, 22, "average"),
            create_rookie_features("Jayden Daniels", "QB", 1, 2, 23, "bad"),
            create_rookie_features("Marvin Harrison Jr", "WR", 1, 4, 21, "good"),
            create_rookie_features("Rome Odunze", "WR", 1, 9, 22, "average"),
            create_rookie_features("Brock Bowers", "TE", 1, 13, 21, "good")
        ]
        
        # Batch predictions
        results = pipeline.batch_predict(example_rookies)
        print("\n🏈 ROOKIE PREDICTIONS:")
        print(results[['player_name', 'position', 'round', 'predicted_ppg', 'performance_tier']].to_string(index=False))
        
        # Single prediction example
        single_result = pipeline.predict_single_rookie(example_rookies[0])
        print(f"\n📊 Single Prediction for {example_rookies[0]['player_name']}:")
        print(f"   Predicted PPG: {single_result['predicted_ppg']:.2f}")
        print(f"   Performance Tier: {single_result['performance_tier']}")
        
    except FileNotFoundError:
        print("❌ Model file not found. Run the notebook first to generate 'best_rookie_model.pkl'")
