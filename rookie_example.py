"""
Example usage of the RookieFantasyRegression model with cleaned data
"""

import pandas as pd
from rookie_model import RookieFantasyRegression

def main():
    """
    Demonstrate how to use the rookie regression model
    """
    print("="*60)
    print("ROOKIE FANTASY REGRESSION MODEL EXAMPLE")
    print("="*60)
    
    # 1. Load the cleaned rookie data
    print("\n1. Loading cleaned rookie data...")
    try:
        df = pd.read_csv('rookie_data_clean.csv')
        print(f"Loaded {len(df)} rookie records from 2013-2023")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("Error: rookie_data_clean.csv not found. Run rookie_proj.py first to generate the data.")
        return
    
    # 2. Initialize the regression model
    print("\n2. Initializing regression model...")
    model = RookieFantasyRegression(target_variable='ppg')
    
    # 3. Prepare features and split data
    print("\n3. Preparing features and splitting data...")
    X, y = model.prepare_features(df, feature_selection='all')
    X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.2)
    
    # 4. Train multiple models
    print("\n4. Training multiple regression models...")
    results = model.train_models(scale_features=True)
    
    # 5. Display model comparison
    model.print_model_comparison()
    
    # 6. Analyze feature importance
    print("\n6. Feature importance analysis...")
    importance_df = model.get_feature_importance(top_n=10)
    
    # 7. Analyze residuals
    print("\n7. Residual analysis...")
    residuals = model.analyze_residuals()
    
    # 8. Example predictions on test set
    print("\n8. Example predictions...")
    sample_predictions = model.predict(X_test.head())
    actual_values = y_test.head().values
    
    print("Sample Predictions vs Actual:")
    for i, (pred, actual) in enumerate(zip(sample_predictions, actual_values)):
        print(f"Player {i+1}: Predicted {pred:.2f} PPG, Actual {actual:.2f} PPG")
    
    # 9. Hyperparameter tuning (optional - takes longer)
    print("\n9. Hyperparameter tuning for Random Forest...")
    tuned_model = model.hyperparameter_tuning('Random Forest')
    
    # 10. Save the model
    print("\n10. Saving the trained model...")
    model.save_model('rookie_fantasy_model.pkl')
    
    return model

def position_specific_analysis():
    """
    Train separate models for different positions
    """
    print("\n" + "="*60)
    print("POSITION-SPECIFIC MODEL ANALYSIS")
    print("="*60)
    
    # Load data
    df = pd.read_csv('rookie_data_clean.csv')
    
    # Analyze by position
    positions = ['QB', 'RB', 'WR', 'TE']
    position_results = {}
    
    for pos in positions:
        pos_data = df[df['position'] == pos].copy()
        
        if len(pos_data) < 20:  # Skip if too few samples
            print(f"\nSkipping {pos} - only {len(pos_data)} samples")
            continue
            
        print(f"\n{pos} Model (n={len(pos_data)}):")
        print("-" * 30)
        
        # Initialize model for this position
        pos_model = RookieFantasyRegression(target_variable='ppg')
        
        # Prepare features
        X, y = pos_model.prepare_features(pos_data, feature_selection='all')
        
        if len(X) < 10:  # Need minimum samples
            print(f"Not enough samples after cleaning: {len(X)}")
            continue
            
        # Split data
        X_train, X_test, y_train, y_test = pos_model.split_data(X, y, test_size=0.2)
        
        # Train models
        results = pos_model.train_models(scale_features=True)
        
        # Store results
        position_results[pos] = {
            'model': pos_model,
            'best_r2': results[pos_model.best_model_name]['test_r2'],
            'best_model_name': pos_model.best_model_name,
            'n_samples': len(X)
        }
        
        print(f"Best model: {pos_model.best_model_name}")
        print(f"Test R²: {results[pos_model.best_model_name]['test_r2']:.3f}")
        
        # Feature importance
        importance_df = pos_model.get_feature_importance(top_n=5)
    
    return position_results

def predict_current_rookies():
    """
    Example of how to predict for current/new rookies
    """
    print("\n" + "="*60)
    print("PREDICTING CURRENT ROOKIES")
    print("="*60)
    
    # Load trained model
    model = RookieFantasyRegression()
    
    try:
        model.load_model('rookie_fantasy_model.pkl')
    except FileNotFoundError:
        print("Model file not found. Run the main example first.")
        return
    
    # Create example rookie data (2024 rookies)
    example_rookies = pd.DataFrame({
        'round': [1, 1, 2, 3, 7],
        'pick': [1, 8, 35, 68, 234],
        'age': [21, 22, 23, 21, 24],
        'early_round': [1, 1, 1, 1, 0],
        'first_round': [1, 1, 0, 0, 0],
        'day1_pick': [1, 1, 0, 0, 0],
        'day2_pick': [0, 0, 1, 1, 0],
        'is_qb': [1, 0, 0, 0, 0],
        'is_rb': [0, 1, 0, 0, 0],
        'is_wr': [0, 0, 1, 1, 0],
        'is_te': [0, 0, 0, 0, 1],
        'good_team': [1, 0, 1, 0, 0],
        'good_offense': [1, 0, 1, 0, 0],
        'bad_offense': [0, 1, 0, 1, 1],
        'games_played_pct': [0.8, 0.9, 0.7, 0.6, 0.5],
        'target_share': [0.0, 0.0, 0.15, 0.12, 0.08],
        'rush_share': [0.0, 0.25, 0.0, 0.0, 0.0],
        'starter_games': [1, 1, 1, 0, 0],
        'yards_per_target': [0.0, 0.0, 8.5, 7.2, 6.8],
        'yards_per_carry': [0.0, 4.2, 0.0, 0.0, 0.0]
    })
    
    # Make predictions
    predictions = model.predict(example_rookies)
    
    print("Example Rookie Predictions:")
    print("-" * 40)
    positions = ['QB', 'RB', 'WR', 'WR', 'TE']
    rounds = example_rookies['round'].values
    
    for i, (pos, round_num, pred) in enumerate(zip(positions, rounds, predictions)):
        print(f"Rookie {i+1} ({pos}, Round {round_num}): {pred:.2f} PPG")

if __name__ == "__main__":
    # Run the main example
    trained_model = main()
    
    # Run position-specific analysis
    if trained_model:
        pos_results = position_specific_analysis()
        
        # Predict current rookies
        predict_current_rookies()
