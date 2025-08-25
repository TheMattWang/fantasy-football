import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RookieFantasyRegression:
    """
    Comprehensive regression model for predicting rookie fantasy football performance
    """
    
    def __init__(self, target_variable='ppg'):
        """
        Initialize the regression model
        
        Args:
            target_variable (str): Target variable to predict ('ppg', 'fantasy_success', or 'top_performer')
        """
        self.target_variable = target_variable
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Define feature groups for better organization
        self.feature_groups = {
            'draft_capital': ['round', 'pick', 'early_round', 'first_round', 'day1_pick', 'day2_pick'],
            'player_attributes': ['age', 'is_qb', 'is_rb', 'is_wr', 'is_te'],
            'team_context': ['good_team', 'good_offense', 'bad_offense'],
            'opportunity': ['games_played_pct', 'target_share', 'rush_share', 'starter_games'],
            'efficiency': ['yards_per_target', 'yards_per_carry']
        }
        
    def prepare_features(self, df, feature_selection='all'):
        """
        Prepare features for modeling
        
        Args:
            df (DataFrame): Input dataframe with rookie data
            feature_selection (str): 'all', 'draft_only', or 'no_efficiency'
        """
        print(f"Preparing features for {len(df)} rookies...")
        
        # Define feature sets based on selection
        if feature_selection == 'draft_only':
            selected_features = self.feature_groups['draft_capital'] + self.feature_groups['player_attributes']
        elif feature_selection == 'no_efficiency':
            selected_features = (self.feature_groups['draft_capital'] + 
                               self.feature_groups['player_attributes'] + 
                               self.feature_groups['team_context'] + 
                               self.feature_groups['opportunity'])
        else:  # 'all'
            selected_features = []
            for group in self.feature_groups.values():
                selected_features.extend(group)
        
        # Filter for available features
        available_features = [f for f in selected_features if f in df.columns]
        self.feature_names = available_features
        
        print(f"Using {len(available_features)} features: {available_features}")
        
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
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train/test sets
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self, fit_on_train=True):
        """
        Scale features using StandardScaler
        """
        if fit_on_train:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        else:
            self.X_train_scaled = self.X_train.copy()
            self.X_test_scaled = self.X_test.copy()
        
        return self.X_train_scaled, self.X_test_scaled
    
    def train_models(self, scale_features=True):
        """
        Train multiple regression models and compare performance
        """
        print("\nTraining multiple regression models...")
        
        # Scale features if requested
        if scale_features:
            X_train, X_test = self.scale_features()
        else:
            X_train, X_test = self.X_train, self.X_test
        
        # Define models to test
        model_configs = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in model_configs.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, self.y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(self.y_train, train_pred)
            test_mae = mean_absolute_error(self.y_test, test_pred)
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, self.y_train, cv=5, scoring='r2')
            
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
        
        # Select best model based on test R2
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        return results
    
    def print_model_comparison(self):
        """
        Print comparison of all trained models
        """
        if not self.models:
            print("No models trained yet. Call train_models() first.")
            return
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        comparison_df = pd.DataFrame({
            name: {
                'Train MAE': f"{results['train_mae']:.3f}",
                'Test MAE': f"{results['test_mae']:.3f}",
                'Train R²': f"{results['train_r2']:.3f}",
                'Test R²': f"{results['test_r2']:.3f}",
                'CV R² (mean)': f"{results['cv_r2_mean']:.3f}",
                'CV R² (std)': f"{results['cv_r2_std']:.3f}"
            }
            for name, results in self.models.items()
        }).T
        
        print(comparison_df)
        print(f"\nBest Model: {self.best_model_name}")
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance from the best model
        """
        if self.best_model is None:
            print("No model trained yet. Call train_models() first.")
            return None
        
        # Get feature importance based on model type
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            importance = np.abs(self.best_model.coef_)
        else:
            print("Cannot extract feature importance from this model type.")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {min(top_n, len(importance_df))} Most Important Features:")
        print("-" * 40)
        for i, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:20s}: {row['importance']:.4f}")
        
        return importance_df
    
    def predict(self, X_new, return_probabilities=False):
        """
        Make predictions on new data
        
        Args:
            X_new (DataFrame): New data to predict on
            return_probabilities (bool): Whether to return prediction intervals
        """
        if self.best_model is None:
            print("No model trained yet. Call train_models() first.")
            return None
        
        # Ensure features match training data
        X_pred = X_new[self.feature_names].fillna(0)
        
        # Scale if scaler was used
        if hasattr(self, 'scaler'):
            X_pred_scaled = self.scaler.transform(X_pred)
            predictions = self.best_model.predict(X_pred_scaled)
        else:
            predictions = self.best_model.predict(X_pred)
        
        return predictions
    
    def analyze_residuals(self):
        """
        Analyze model residuals for validation
        """
        if self.best_model is None:
            print("No model trained yet. Call train_models() first.")
            return
        
        # Get predictions
        if hasattr(self, 'X_test_scaled'):
            test_pred = self.best_model.predict(self.X_test_scaled)
        else:
            test_pred = self.best_model.predict(self.X_test)
        
        residuals = self.y_test - test_pred
        
        print(f"\nResidual Analysis for {self.best_model_name}:")
        print("-" * 40)
        print(f"Mean residual: {residuals.mean():.4f}")
        print(f"Residual std: {residuals.std():.4f}")
        print(f"Max absolute residual: {np.abs(residuals).max():.4f}")
        
        return residuals
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """
        Perform hyperparameter tuning for specified model
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        # Scale features
        X_train, X_test = self.scale_features()
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
        }
        
        # Define base models
        base_models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Ridge': Ridge(),
            'Lasso': Lasso()
        }
        
        if model_name not in param_grids:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update best model if this performs better
        test_pred = grid_search.predict(X_test)
        test_r2 = r2_score(self.y_test, test_pred)
        
        if test_r2 > self.models[self.best_model_name]['test_r2']:
            self.best_model = grid_search.best_estimator_
            self.best_model_name = f"{model_name} (Tuned)"
            print(f"New best model: {self.best_model_name} (Test R²: {test_r2:.4f})")
        
        return grid_search
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        import pickle
        
        model_data = {
            'best_model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_variable': self.target_variable,
            'best_model_name': self.best_model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_variable = model_data['target_variable']
        self.best_model_name = model_data['best_model_name']
        
        print(f"Model loaded from {filepath}")
        print(f"Model type: {self.best_model_name}")
        print(f"Target variable: {self.target_variable}")
