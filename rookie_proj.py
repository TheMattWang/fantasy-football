import pandas as pd
import numpy as np
from nfl_data_py import import_weekly_data, import_schedules, import_team_desc, import_draft_picks
import warnings
warnings.filterwarnings('ignore')

# ============================== ROOKIE PROJECTIONS DATA CLEANING ==============================

def clean_rookie_data():
    """
    Build and clean comprehensive rookie dataset for regression modeling
    """
    print("Starting rookie data cleaning pipeline...")
    
    # 1. Import historical data (2013-2023 for good sample size)
    seasons = range(2013, 2024)
    print(f"Importing weekly data for seasons {seasons.start}-{seasons.stop-1}...")
    
    weekly = import_weekly_data(seasons)
    draft_data = import_draft_picks(seasons)
    team_stats = import_team_desc()  # No seasons parameter needed
    
    # 2. Build rookie dataset with comprehensive metrics
    print("Building rookie dataset...")
    
    # Debug: Check available columns
    print("Available columns in draft data:")
    print(draft_data.columns.tolist())
    
    # Create rookie identification from draft data
    # Get first-year players by matching draft year to season
    draft_rookies = draft_data[['season', 'pfr_player_name', 'position']].copy()
    draft_rookies['is_rookie'] = 1
    
    print(f"Draft rookies sample:")
    print(draft_rookies.head())
    print(f"Weekly data sample:")
    print(weekly[['season', 'player_name', 'position']].head())
    
    # Try multiple merge strategies for better matching
    # First try exact match
    weekly_with_rookies = weekly.merge(
        draft_rookies, 
        left_on=['season', 'player_name', 'position'],
        right_on=['season', 'pfr_player_name', 'position'],
        how='left'
    )
    
    # If no matches, try with player_display_name
    if weekly_with_rookies['is_rookie'].sum() == 0:
        print("No exact matches found, trying with player_display_name...")
        weekly_with_rookies = weekly.merge(
            draft_rookies, 
            left_on=['season', 'player_display_name', 'position'],
            right_on=['season', 'pfr_player_name', 'position'],
            how='left'
        )
    
    # Fill missing is_rookie with 0 (veterans)
    weekly_with_rookies['is_rookie'] = weekly_with_rookies['is_rookie'].fillna(0)
    
    print(f"Found {weekly_with_rookies['is_rookie'].sum()} rookie player-weeks")
    
    # Filter for rookies only
    rookie_data = weekly_with_rookies[weekly_with_rookies["is_rookie"] == 1].copy()
    
    print(f"Rookie data shape: {rookie_data.shape}")
    
    # Use existing fantasy_points_ppr column
    rookie_data['fantasy_points'] = rookie_data['fantasy_points_ppr'].fillna(0)
    
    # 3. Aggregate rookie season stats
    rookie_agg = rookie_data.groupby(['season', 'player_id', 'player_name', 'position', 'recent_team']).agg({
        'week': 'nunique',  # games played
        'fantasy_points': ['sum', 'mean'],
        'passing_yards': 'sum',
        'passing_tds': 'sum',
        'rushing_yards': 'sum', 
        'rushing_tds': 'sum',
        'receiving_yards': 'sum',
        'receiving_tds': 'sum',
        'receptions': 'sum',
        'targets': 'sum',
        'carries': 'sum',  # rushing attempts column name
        'attempts': 'sum'   # passing attempts column name
    }).reset_index()
    
    # Flatten column names
    rookie_agg.columns = ['season', 'player_id', 'player_name', 'position', 'team', 
                         'games_played', 'total_fp', 'avg_fp', 'pass_yds', 'pass_tds',
                         'rush_yds', 'rush_tds', 'rec_yds', 'rec_tds', 'receptions', 
                         'targets', 'rush_att', 'pass_att']
    
    # Calculate per-game stats
    rookie_agg['ppg'] = rookie_agg['total_fp'] / rookie_agg['games_played'].clip(lower=1)
    rookie_agg['rush_ypg'] = rookie_agg['rush_yds'] / rookie_agg['games_played'].clip(lower=1)
    rookie_agg['rec_ypg'] = rookie_agg['rec_yds'] / rookie_agg['games_played'].clip(lower=1)
    rookie_agg['targets_pg'] = rookie_agg['targets'] / rookie_agg['games_played'].clip(lower=1)
    
    return rookie_agg, draft_data, team_stats

def add_draft_capital(rookie_agg, draft_data):
    """
    Add draft capital information to rookie data
    """
    print("Adding draft capital data...")
    
    # Clean draft data
    draft_clean = draft_data[['season', 'pfr_player_name', 'position', 'team', 'round', 'pick', 'age']].copy()
    draft_clean = draft_clean.rename(columns={'pfr_player_name': 'player_name'})
    
    # Merge with rookie data
    rookie_with_draft = rookie_agg.merge(
        draft_clean, 
        on=['season', 'player_name', 'position'], 
        how='left',
        suffixes=('', '_draft')
    )
    
    # Fill missing draft info for undrafted players
    rookie_with_draft['round'] = rookie_with_draft['round'].fillna(8)  # UDFA as round 8
    rookie_with_draft['pick'] = rookie_with_draft['pick'].fillna(300)  # High pick number for UDFA
    rookie_with_draft['age'] = rookie_with_draft['age'].fillna(rookie_with_draft['age'].median())
    
    # Create draft capital features
    rookie_with_draft['early_round'] = (rookie_with_draft['round'] <= 3).astype(int)
    rookie_with_draft['first_round'] = (rookie_with_draft['round'] == 1).astype(int)
    rookie_with_draft['day1_pick'] = (rookie_with_draft['pick'] <= 32).astype(int)
    rookie_with_draft['day2_pick'] = ((rookie_with_draft['pick'] > 32) & 
                                     (rookie_with_draft['pick'] <= 96)).astype(int)
    
    return rookie_with_draft

def add_team_features(rookie_with_draft, team_stats):
    """
    Add team performance metrics
    """
    print("Adding team performance features...")
    print("Available team stats columns:")
    print(team_stats.columns.tolist())
    
    # For now, just add placeholder team features since team_stats structure is different
    rookie_final = rookie_with_draft.copy()
    
    # Add basic team quality placeholders (can be enhanced later)
    rookie_final['total_wins'] = 8  # Average team wins
    rookie_final['playoff'] = 0    # Most teams don't make playoffs
    rookie_final['good_team'] = 0  # Default to average team
    rookie_final['good_offense'] = 0  # Default to average offense
    rookie_final['bad_offense'] = 0   # Default to not bad offense
    
    return rookie_final

def engineer_features(rookie_final):
    """
    Create additional features for regression model
    """
    print("Engineering additional features...")
    
    # Position-specific features
    rookie_final['is_qb'] = (rookie_final['position'] == 'QB').astype(int)
    rookie_final['is_rb'] = (rookie_final['position'] == 'RB').astype(int)  
    rookie_final['is_wr'] = (rookie_final['position'] == 'WR').astype(int)
    rookie_final['is_te'] = (rookie_final['position'] == 'TE').astype(int)
    
    # Opportunity metrics
    rookie_final['target_share'] = rookie_final['targets_pg'] / 35  # Approx team targets per game
    rookie_final['rush_share'] = rookie_final['rush_att'] / (rookie_final['games_played'] * 25)  # Approx team rushes
    
    # Efficiency metrics
    rookie_final['yards_per_target'] = np.where(
        rookie_final['targets'] > 0,
        rookie_final['rec_yds'] / rookie_final['targets'],
        0
    )
    
    rookie_final['yards_per_carry'] = np.where(
        rookie_final['rush_att'] > 0,
        rookie_final['rush_yds'] / rookie_final['rush_att'],
        0
    )
    
    # Games played impact
    rookie_final['games_played_pct'] = rookie_final['games_played'] / 17  # Full season
    rookie_final['starter_games'] = (rookie_final['games_played'] >= 12).astype(int)
    
    return rookie_final

def clean_data_quality(df):
    """
    Handle missing values, outliers, and data quality issues
    """
    print("Cleaning data quality issues...")
    
    # Remove players with very few games (< 4) as they're likely injury cases
    df_clean = df[df['games_played'] >= 4].copy()
    
    # Cap extreme outliers (99th percentile)
    numeric_cols = ['ppg', 'rush_ypg', 'rec_ypg', 'targets_pg', 'yards_per_target', 'yards_per_carry']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            p99 = df_clean[col].quantile(0.99)
            df_clean[col] = df_clean[col].clip(upper=p99)
    
    # Fill any remaining missing values
    df_clean = df_clean.fillna(0)
    
    # Create success metrics for target variable
    df_clean['fantasy_success'] = (df_clean['ppg'] >= 10).astype(int)  # 10+ PPG threshold
    df_clean['top_performer'] = (df_clean['ppg'] >= 15).astype(int)    # Elite threshold
    
    return df_clean

def main():
    """
    Main data cleaning pipeline
    """
    # Step 1: Get raw data
    rookie_agg, draft_data, team_stats = clean_rookie_data()
    
    # Step 2: Add draft capital
    rookie_with_draft = add_draft_capital(rookie_agg, draft_data)
    
    # Step 3: Add team features  
    rookie_with_teams = add_team_features(rookie_with_draft, team_stats)
    
    # Step 4: Engineer features
    rookie_engineered = engineer_features(rookie_with_teams)
    
    # Step 5: Clean data quality
    rookie_clean = clean_data_quality(rookie_engineered)
    
    # Step 6: Export cleaned data
    print("Exporting cleaned dataset...")
    rookie_clean.to_csv('rookie_data_clean.csv', index=False)
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"Total rookies: {len(rookie_clean)}")
    print(f"Seasons: {rookie_clean['season'].min()}-{rookie_clean['season'].max()}")
    print(f"Positions: {rookie_clean['position'].value_counts().to_dict()}")
    print(f"Average PPG: {rookie_clean['ppg'].mean():.2f}")
    print(f"Success rate (10+ PPG): {rookie_clean['fantasy_success'].mean():.2%}")
    
    print(f"\nTop features for regression:")
    feature_cols = ['round', 'pick', 'age', 'early_round', 'first_round', 'day1_pick', 
                   'good_team', 'good_offense', 'games_played_pct', 'target_share', 
                   'rush_share', 'is_qb', 'is_rb', 'is_wr', 'is_te']
    print(feature_cols)
    
    return rookie_clean

def analyze_rookie_performance(df):
    """
    Quick analysis of rookie performance patterns
    """
    print("\n" + "="*60)
    print("ROOKIE PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Position breakdown
    print("\nROOKIE PERFORMANCE BY POSITION:")
    pos_stats = df.groupby('position').agg({
        'ppg': ['count', 'mean', 'std'],
        'fantasy_success': 'mean',
        'top_performer': 'mean'
    }).round(2)
    print(pos_stats)
    
    # Draft capital impact
    print("\nDRAFT CAPITAL IMPACT:")
    draft_stats = df.groupby('first_round').agg({
        'ppg': 'mean',
        'fantasy_success': 'mean',
        'top_performer': 'mean'
    }).round(3)
    print(draft_stats)
    
    # Key correlations
    print("\nKEY CORRELATIONS WITH PPG:")
    feature_cols = ['round', 'pick', 'age', 'games_played_pct', 'target_share', 'rush_share']
    available_cols = [col for col in feature_cols if col in df.columns]
    correlations = df[available_cols + ['ppg']].corr()['ppg'].sort_values(ascending=False)
    print(correlations)
    
    return df

if __name__ == "__main__":
    cleaned_data = main()
    analyzed_data = analyze_rookie_performance(cleaned_data)
    
    # Optional: Quick model training demonstration
    try:
        from rookie_model import RookieFantasyRegression
        print("\n" + "="*60)
        print("QUICK MODEL TRAINING DEMONSTRATION")
        print("="*60)
        
        # Initialize and train a quick model
        model = RookieFantasyRegression(target_variable='ppg')
        X, y = model.prepare_features(cleaned_data, feature_selection='draft_only')
        X_train, X_test, y_train, y_test = model.split_data(X, y)
        results = model.train_models(scale_features=True)
        
        print(f"\nQuick model results (draft capital + position only):")
        print(f"Best model: {model.best_model_name}")
        print(f"Test R²: {results[model.best_model_name]['test_r2']:.3f}")
        
        print("\nFor full model training and analysis, run rookie_example.py")
        
    except ImportError:
        print("\nTo use the regression model, ensure sklearn is installed.")
        print("Run: pip install scikit-learn matplotlib seaborn")

