# Fantasy Football Rookie Predictor

This project builds a comprehensive regression model to predict rookie fantasy football performance using historical data from 2013-2023.

## Project Structure

### Data Processing
- `rookie_proj.py` - Main data cleaning pipeline that builds the historical rookie dataset
- `clean.py` - Original draft board pipeline for current players
- `rookie_data_clean.csv` - Generated cleaned rookie dataset (501 rookies, 2013-2023)

### Machine Learning Model
- `rookie_model.py` - RookieFantasyRegression class with comprehensive ML capabilities
- `rookie_example.py` - Example usage and demonstrations
- `rookie_fantasy_model.pkl` - Saved trained model (generated after running examples)

### Data Files
- `draft_board.csv` - Current player draft board with VORP calculations
- `FantasyPros_2025_Overall_ADP_Rankings.csv` - 2025 ADP rankings
- `fantasy_football.ipynb` - Jupyter notebook for analysis

## Quick Start

### 1. Generate Clean Rookie Data
```bash
python rookie_proj.py
```
This creates `rookie_data_clean.csv` with 501 historical rookies and their performance metrics.

### 2. Train and Evaluate Models
```bash
python rookie_example.py
```
This demonstrates:
- Training multiple regression models (Linear, Ridge, Lasso, Random Forest, etc.)
- Model comparison and evaluation
- Feature importance analysis
- Position-specific modeling
- Hyperparameter tuning
- Making predictions on new rookies

### 3. Quick Model Demo
The rookie data pipeline (`rookie_proj.py`) includes a quick model demonstration at the end.

## Features

### Data Cleaning Pipeline
- **Historical Data**: 11 seasons (2013-2023) of rookie performance
- **Fantasy Scoring**: PPR scoring with comprehensive stat tracking
- **Draft Capital**: Round, pick number, age, team context
- **Performance Metrics**: PPG, efficiency stats, opportunity metrics
- **Data Quality**: Handles missing values, outliers, minimum games played

### Regression Model Class
- **Multiple Algorithms**: Linear, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting
- **Feature Engineering**: 15+ predictive features across 5 categories
- **Model Evaluation**: Cross-validation, R², MAE, residual analysis
- **Feature Selection**: All features, draft-only, or custom selections
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Prediction Capabilities**: Predict PPG for new rookies
- **Model Persistence**: Save and load trained models

### Key Features for Prediction
1. **Draft Capital**: round, pick, early_round, first_round, day1_pick, day2_pick
2. **Player Attributes**: age, position (QB/RB/WR/TE)
3. **Team Context**: good_team, good_offense, bad_offense
4. **Opportunity**: games_played_pct, target_share, rush_share, starter_games
5. **Efficiency**: yards_per_target, yards_per_carry

## Results

- **Dataset**: 501 rookies with 26.75% achieving "fantasy success" (10+ PPG)
- **Model Performance**: Best models typically achieve R² > 0.4 for PPG prediction
- **Key Predictors**: Draft capital (round, pick), opportunity metrics, position
- **Position Breakdown**: WR (206), RB (144), TE (95), QB (50), FB (4), CB (2)

## Dependencies

```bash
pip install pandas numpy nfl_data_py scikit-learn matplotlib seaborn
```

## Usage Examples

### Basic Model Training
```python
from rookie_model import RookieFantasyRegression
import pandas as pd

# Load data
df = pd.read_csv('rookie_data_clean.csv')

# Initialize model
model = RookieFantasyRegression(target_variable='ppg')

# Prepare features and train
X, y = model.prepare_features(df)
X_train, X_test, y_train, y_test = model.split_data(X, y)
results = model.train_models()

# Evaluate
model.print_model_comparison()
model.get_feature_importance()
```

### Making Predictions
```python
# For new rookie data
new_rookie_data = pd.DataFrame({...})  # Your rookie features
predictions = model.predict(new_rookie_data)
```

### Position-Specific Models
```python
# Train separate models by position
wr_data = df[df['position'] == 'WR']
wr_model = RookieFantasyRegression()
# ... train on WR-specific data
```

## Model Insights

- **First-round picks** average significantly higher PPG than later rounds
- **Opportunity metrics** (target share, games played) are crucial predictors
- **Team context** matters less than individual draft capital and opportunity
- **Position differences** suggest separate models may be beneficial for WR/RB vs TE/QB

## Class Documentation

### RookieFantasyRegression

Main regression model class for predicting rookie fantasy performance.

#### Key Methods:
- `prepare_features(df, feature_selection)` - Prepare feature matrix
- `train_models(scale_features)` - Train multiple models and compare
- `predict(X_new)` - Make predictions on new data
- `get_feature_importance()` - Analyze feature importance
- `hyperparameter_tuning(model_name)` - Optimize model parameters
- `save_model(filepath)` / `load_model(filepath)` - Model persistence

#### Target Variables:
- `'ppg'` - Points per game (regression)
- `'fantasy_success'` - 10+ PPG threshold (classification)
- `'top_performer'` - 15+ PPG threshold (classification)

#### Feature Selection Options:
- `'all'` - All 15+ features
- `'draft_only'` - Draft capital + position only
- `'no_efficiency'` - Exclude yards per target/carry metrics
