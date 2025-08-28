"""
Hyperparameter Optimization for Fantasy Football Draft Strategies
================================================================

This module provides sophisticated hyperparameter optimization using various
search algorithms including grid search, random search, and Bayesian optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import time
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
warnings.filterwarnings('ignore')

from .backtesting import DraftBacktester, BacktestResults
from ..core.player import PlayerPool
from ..core.draft import LeagueSettings


@dataclass
class HyperparameterResult:
    """Result from a single hyperparameter configuration test"""
    parameters: Dict[str, Any]
    mean_score: float
    std_score: float
    mean_vorp: float
    n_trials: int
    detailed_results: List[BacktestResults] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParameterSpace:
    """Defines the hyperparameter search space"""
    
    def __init__(self):
        self.parameters = {}
        self.constraints = []
    
    def add_parameter(self, 
                     name: str, 
                     param_type: str, 
                     values: Union[List, Tuple, Dict]):
        """
        Add a parameter to the search space.
        
        Args:
            name: Parameter name
            param_type: 'categorical', 'integer', 'float', 'log_uniform'
            values: Values/ranges for the parameter
        """
        self.parameters[name] = {
            'type': param_type,
            'values': values
        }
    
    def add_constraint(self, constraint_func: Callable[[Dict], bool]):
        """Add a constraint function that returns True if parameters are valid"""
        self.constraints.append(constraint_func)
    
    def sample_random(self, n_samples: int = 1) -> List[Dict[str, Any]]:
        """Sample random parameter configurations"""
        samples = []
        
        for _ in range(n_samples * 10):  # Generate extra to account for constraint failures
            if len(samples) >= n_samples:
                break
                
            sample = {}
            
            for param_name, param_info in self.parameters.items():
                param_type = param_info['type']
                values = param_info['values']
                
                if param_type == 'categorical':
                    sample[param_name] = np.random.choice(values)
                elif param_type == 'integer':
                    sample[param_name] = np.random.randint(values[0], values[1] + 1)
                elif param_type == 'float':
                    sample[param_name] = np.random.uniform(values[0], values[1])
                elif param_type == 'log_uniform':
                    log_low, log_high = np.log10(values[0]), np.log10(values[1])
                    sample[param_name] = 10 ** np.random.uniform(log_low, log_high)
            
            # Check constraints
            if all(constraint(sample) for constraint in self.constraints):
                samples.append(sample)
        
        return samples[:n_samples]
    
    def get_grid_search_configs(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search"""
        
        # Convert to sklearn ParameterGrid format
        grid_params = {}
        
        for param_name, param_info in self.parameters.items():
            param_type = param_info['type']
            values = param_info['values']
            
            if param_type == 'categorical':
                grid_params[param_name] = values
            elif param_type in ['integer', 'float']:
                # Create discrete grid
                if param_type == 'integer':
                    grid_params[param_name] = list(range(values[0], values[1] + 1))
                else:
                    grid_params[param_name] = np.linspace(values[0], values[1], 5).tolist()
            elif param_type == 'log_uniform':
                log_values = np.logspace(np.log10(values[0]), np.log10(values[1]), 4)
                grid_params[param_name] = log_values.tolist()
        
        # Generate all combinations
        param_grid = list(ParameterGrid(grid_params))
        
        # Filter by constraints
        valid_configs = []
        for config in param_grid:
            if all(constraint(config) for constraint in self.constraints):
                valid_configs.append(config)
        
        return valid_configs


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for draft strategies.
    
    Supports multiple optimization algorithms:
    - Grid Search: Exhaustive search over parameter grid
    - Random Search: Random sampling from parameter space
    - Bayesian Optimization: Gaussian Process guided search
    """
    
    def __init__(self, 
                 backtester: DraftBacktester,
                 strategy_factory: Callable[[Dict], Any],
                 parameter_space: ParameterSpace):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            backtester: Configured DraftBacktester instance
            strategy_factory: Function that creates strategy from parameters
            parameter_space: Hyperparameter search space definition
        """
        
        self.backtester = backtester
        self.strategy_factory = strategy_factory
        self.parameter_space = parameter_space
        
        # Optimization tracking
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = -np.inf
        
        print(f"🔧 Hyperparameter Optimizer initialized")
        print(f"   📊 Parameters: {list(parameter_space.parameters.keys())}")
        print(f"   🎯 Optimization target: Composite backtest score")
    
    def grid_search(self, 
                   n_trials_per_config: int = 5,
                   parallel: bool = True,
                   save_results: bool = True) -> pd.DataFrame:
        """
        Perform exhaustive grid search over parameter space.
        
        Args:
            n_trials_per_config: Number of backtest trials per configuration
            parallel: Use parallel processing
            save_results: Save results to file
            
        Returns:
            DataFrame with results for all configurations
        """
        
        print(f"🔍 Starting Grid Search Optimization")
        
        # Generate all parameter configurations
        param_configs = self.parameter_space.get_grid_search_configs()
        
        print(f"   📋 Total configurations: {len(param_configs)}")
        print(f"   🔄 Trials per config: {n_trials_per_config}")
        print(f"   ⏱️  Estimated time: {len(param_configs) * n_trials_per_config * 0.5:.1f} seconds")
        
        # Test each configuration
        results = []
        
        if parallel:
            results = self._run_parallel_optimization(param_configs, n_trials_per_config)
        else:
            results = self._run_sequential_optimization(param_configs, n_trials_per_config)
        
        # Convert to DataFrame
        results_df = self._create_results_dataframe(results)
        
        # Update best parameters
        if not results_df.empty and 'mean_score' in results_df.columns:
            best_idx = results_df['mean_score'].idxmax()
            self.best_parameters = results_df.loc[best_idx, 'parameters']
            self.best_score = results_df.loc[best_idx, 'mean_score']
        else:
            print("⚠️  No valid results found in grid search")
            self.best_parameters = {}
            self.best_score = 0.0
        
        print(f"✅ Grid Search Complete!")
        print(f"   🏆 Best Score: {self.best_score:.4f}")
        print(f"   🎯 Best Parameters: {self.best_parameters}")
        
        if save_results:
            self._save_optimization_results(results_df, 'grid_search')
        
        return results_df
    
    def random_search(self, 
                     n_configurations: int = 50,
                     n_trials_per_config: int = 5,
                     parallel: bool = True,
                     save_results: bool = True) -> pd.DataFrame:
        """
        Perform random search optimization.
        
        Args:
            n_configurations: Number of random configurations to test
            n_trials_per_config: Number of backtest trials per configuration
            parallel: Use parallel processing
            save_results: Save results to file
            
        Returns:
            DataFrame with results for all configurations
        """
        
        print(f"🎲 Starting Random Search Optimization")
        print(f"   📋 Configurations: {n_configurations}")
        print(f"   🔄 Trials per config: {n_trials_per_config}")
        
        # Generate random parameter configurations
        param_configs = self.parameter_space.sample_random(n_configurations)
        
        # Test each configuration
        if parallel:
            results = self._run_parallel_optimization(param_configs, n_trials_per_config)
        else:
            results = self._run_sequential_optimization(param_configs, n_trials_per_config)
        
        # Convert to DataFrame
        results_df = self._create_results_dataframe(results)
        
        # Update best parameters
        if not results_df.empty and 'mean_score' in results_df.columns:
            best_idx = results_df['mean_score'].idxmax()
            self.best_parameters = results_df.loc[best_idx, 'parameters']
            self.best_score = results_df.loc[best_idx, 'mean_score']
        else:
            print("⚠️  No valid results found in random search")
            self.best_parameters = {}
            self.best_score = 0.0
        
        print(f"✅ Random Search Complete!")
        print(f"   🏆 Best Score: {self.best_score:.4f}")
        print(f"   🎯 Best Parameters: {self.best_parameters}")
        
        if save_results:
            self._save_optimization_results(results_df, 'random_search')
        
        return results_df
    
    def bayesian_optimization(self, 
                             n_initial: int = 10,
                             n_iterations: int = 20,
                             n_trials_per_config: int = 3,
                             acquisition_function: str = 'ei',
                             save_results: bool = True) -> pd.DataFrame:
        """
        Perform Bayesian optimization using Gaussian Process.
        
        Args:
            n_initial: Number of random initial points
            n_iterations: Number of Bayesian optimization iterations
            n_trials_per_config: Number of backtest trials per configuration
            acquisition_function: 'ei' (expected improvement) or 'ucb' (upper confidence bound)
            save_results: Save results to file
            
        Returns:
            DataFrame with results for all configurations
        """
        
        print(f"🧠 Starting Bayesian Optimization")
        print(f"   🎲 Initial random points: {n_initial}")
        print(f"   🔄 BO iterations: {n_iterations}")
        print(f"   📊 Trials per config: {n_trials_per_config}")
        
        # Check if we can do Bayesian optimization (need numerical parameters)
        numerical_params = [
            name for name, info in self.parameter_space.parameters.items()
            if info['type'] in ['integer', 'float', 'log_uniform']
        ]
        
        if len(numerical_params) < 1:
            print("⚠️  Bayesian optimization requires numerical parameters. Falling back to random search.")
            return self.random_search(n_initial + n_iterations, n_trials_per_config, save_results=save_results)
        
        # Initial random exploration
        print("   🎲 Initial exploration phase...")
        initial_configs = self.parameter_space.sample_random(n_initial)
        initial_results = self._run_sequential_optimization(initial_configs, n_trials_per_config)
        
        # Set up Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        all_results = initial_results.copy()
        
        # Bayesian optimization iterations
        print("   🧠 Bayesian optimization phase...")
        for iteration in range(n_iterations):
            print(f"      Iteration {iteration + 1}/{n_iterations}")
            
            # Prepare training data for GP
            X_train = []
            y_train = []
            
            for result in all_results:
                # Convert parameters to feature vector (numerical only)
                features = []
                for param_name in numerical_params:
                    value = result.parameters[param_name]
                    if self.parameter_space.parameters[param_name]['type'] == 'log_uniform':
                        value = np.log10(value)  # Log transform for log-uniform parameters
                    features.append(value)
                
                X_train.append(features)
                y_train.append(result.mean_score)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Fit GP
            gp.fit(X_train, y_train)
            
            # Generate candidate points and select best according to acquisition function
            candidate_configs = self.parameter_space.sample_random(100)  # Generate many candidates
            
            best_acquisition = -np.inf
            best_candidate = None
            
            for candidate in candidate_configs:
                # Convert to feature vector
                features = []
                for param_name in numerical_params:
                    value = candidate[param_name]
                    if self.parameter_space.parameters[param_name]['type'] == 'log_uniform':
                        value = np.log10(value)
                    features.append(value)
                
                X_candidate = np.array([features])
                
                # Calculate acquisition function
                if acquisition_function == 'ei':
                    acquisition_value = self._expected_improvement(gp, X_candidate, y_train)
                else:  # ucb
                    acquisition_value = self._upper_confidence_bound(gp, X_candidate)
                
                if acquisition_value > best_acquisition:
                    best_acquisition = acquisition_value
                    best_candidate = candidate
            
            if best_candidate:
                # Evaluate best candidate
                candidate_results = self._run_sequential_optimization([best_candidate], n_trials_per_config)
                all_results.extend(candidate_results)
                
                best_score = candidate_results[0].mean_score
                print(f"         Score: {best_score:.4f}, Acquisition: {best_acquisition:.4f}")
        
        # Convert to DataFrame
        results_df = self._create_results_dataframe(all_results)
        
        # Update best parameters
        if not results_df.empty:
            best_idx = results_df['mean_score'].idxmax()
            self.best_parameters = results_df.loc[best_idx, 'parameters']
            self.best_score = results_df.loc[best_idx, 'mean_score']
        
        print(f"✅ Bayesian Optimization Complete!")
        print(f"   🏆 Best Score: {self.best_score:.4f}")
        print(f"   🎯 Best Parameters: {self.best_parameters}")
        
        if save_results:
            self._save_optimization_results(results_df, 'bayesian_optimization')
        
        return results_df
    
    def _expected_improvement(self, gp, X, y_best):
        """Calculate Expected Improvement acquisition function"""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        # Calculate expected improvement
        imp = mu - np.max(y_best)
        Z = imp / sigma
        ei = imp * self._normal_cdf(Z) + sigma * self._normal_pdf(Z)
        
        return ei[0, 0] if ei.ndim > 1 else ei[0]
    
    def _upper_confidence_bound(self, gp, X, kappa=2.576):
        """Calculate Upper Confidence Bound acquisition function"""
        mu, sigma = gp.predict(X, return_std=True)
        return mu[0] + kappa * sigma[0]
    
    def _normal_cdf(self, x):
        """Standard normal CDF"""
        return 0.5 * (1 + np.vectorize(lambda t: 
            2/np.sqrt(np.pi) * np.exp(-t**2/2) if abs(t) < 8 else (1 if t > 0 else -1))(x/np.sqrt(2)))
    
    def _normal_pdf(self, x):
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _run_parallel_optimization(self, 
                                  param_configs: List[Dict],
                                  n_trials_per_config: int) -> List[HyperparameterResult]:
        """Run optimization with parallel processing"""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.backtester.parallel_workers) as executor:
            futures = []
            
            for config in param_configs:
                future = executor.submit(
                    self._evaluate_configuration, config, n_trials_per_config
                )
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=60)  # 1 minute timeout
                    if result:
                        results.append(result)
                        
                    if (i + 1) % 10 == 0:
                        print(f"      Completed {i + 1}/{len(futures)} configurations")
                        
                except Exception as e:
                    print(f"      ⚠️  Configuration failed: {e}")
        
        return results
    
    def _run_sequential_optimization(self, 
                                   param_configs: List[Dict],
                                   n_trials_per_config: int) -> List[HyperparameterResult]:
        """Run optimization sequentially"""
        
        results = []
        
        for i, config in enumerate(param_configs):
            try:
                result = self._evaluate_configuration(config, n_trials_per_config)
                if result:
                    results.append(result)
                
                if (i + 1) % 5 == 0:
                    print(f"      Completed {i + 1}/{len(param_configs)} configurations")
                    
            except Exception as e:
                print(f"      ⚠️  Configuration {i+1} failed: {e}")
        
        return results
    
    def _evaluate_configuration(self, 
                               parameters: Dict[str, Any],
                               n_trials: int) -> Optional[HyperparameterResult]:
        """Evaluate a single parameter configuration"""
        
        try:
            # Create strategy with these parameters
            strategy = self.strategy_factory(parameters)
            strategy_name = f"Config_{hash(str(parameters)) % 10000:04d}"
            
            # Run backtest with reduced scenarios for speed
            quick_scenarios = [
                {'name': 'standard_pos_6', 'draft_position': 6, 'league_type': 'standard', 
                 'injury_rate': 0.15, 'rookie_uncertainty': 1.0, 'opponent_skill': 'average'}
            ]
            
            # Temporarily reduce simulation count for optimization
            original_sims = self.backtester.n_simulations
            self.backtester.n_simulations = n_trials
            
            # Run backtest
            backtest_results = self.backtester.backtest_strategy(
                strategy, strategy_name, quick_scenarios, parallel=False
            )
            
            # Restore original simulation count
            self.backtester.n_simulations = original_sims
            
            # Extract scores
            if backtest_results and 'standard_pos_6' in backtest_results:
                trial_results = backtest_results['standard_pos_6']
                scores = [r.composite_score for r in trial_results]
                vorps = [r.total_vorp for r in trial_results]
                
                if scores:
                    return HyperparameterResult(
                        parameters=parameters,
                        mean_score=np.mean(scores),
                        std_score=np.std(scores),
                        mean_vorp=np.mean(vorps),
                        n_trials=len(scores),
                        detailed_results=trial_results
                    )
            
        except Exception as e:
            print(f"         ❌ Evaluation error: {e}")
            return None
    
    def _create_results_dataframe(self, results: List[HyperparameterResult]) -> pd.DataFrame:
        """Convert optimization results to DataFrame"""
        
        data = []
        
        for result in results:
            row = {
                'mean_score': result.mean_score,
                'std_score': result.std_score,
                'mean_vorp': result.mean_vorp,
                'n_trials': result.n_trials,
                'parameters': result.parameters
            }
            
            # Add individual parameter columns
            for param_name, param_value in result.parameters.items():
                row[f'param_{param_name}'] = param_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _save_optimization_results(self, results_df: pd.DataFrame, method_name: str):
        """Save optimization results to files"""
        
        timestamp = int(time.time())
        
        # Save DataFrame
        csv_path = f"hyperparameter_results_{method_name}_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Save optimization history
        history_path = f"optimization_history_{method_name}_{timestamp}.json"
        history_data = {
            'method': method_name,
            'timestamp': timestamp,
            'best_parameters': self.best_parameters,
            'best_score': float(self.best_score),
            'parameter_space': {
                name: info for name, info in self.parameter_space.parameters.items()
            }
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        print(f"   💾 Results saved:")
        print(f"      📊 CSV: {csv_path}")
        print(f"      📋 History: {history_path}")
    
    def create_optimization_report(self, 
                                  results_df: pd.DataFrame,
                                  method_name: str = "optimization") -> str:
        """Create comprehensive optimization report with visualizations"""
        
        print(f"📊 Creating hyperparameter optimization report...")
        
        # Create visualizations
        n_params = len([col for col in results_df.columns if col.startswith('param_')])
        fig_height = max(8, n_params * 2)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, fig_height))
        fig.suptitle(f'Hyperparameter Optimization Report - {method_name.title()}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Score distribution
        ax1 = axes[0, 0]
        results_df['mean_score'].hist(bins=20, ax=ax1, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.axvline(results_df['mean_score'].max(), color='red', linestyle='--', 
                   label=f'Best: {results_df["mean_score"].max():.4f}')
        ax1.set_title('Score Distribution')
        ax1.set_xlabel('Mean Composite Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # 2. Score vs VORP
        ax2 = axes[0, 1]
        scatter = ax2.scatter(results_df['mean_vorp'], results_df['mean_score'], 
                             alpha=0.6, c=results_df['mean_score'], cmap='viridis')
        ax2.set_xlabel('Mean VORP')
        ax2.set_ylabel('Mean Composite Score')
        ax2.set_title('Score vs VORP')
        plt.colorbar(scatter, ax=ax2)
        
        # 3. Parameter correlation heatmap
        ax3 = axes[1, 0]
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        if len(param_cols) > 1:
            param_data = results_df[param_cols + ['mean_score']]
            correlation_matrix = param_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax3)
            ax3.set_title('Parameter Correlations')
        else:
            ax3.text(0.5, 0.5, 'Need >1 parameter\nfor correlation', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Parameter Correlations')
        
        # 4. Top configurations
        ax4 = axes[1, 1]
        top_configs = results_df.nlargest(min(10, len(results_df)), 'mean_score')
        y_pos = np.arange(len(top_configs))
        ax4.barh(y_pos, top_configs['mean_score'], color='lightgreen', edgecolor='darkgreen')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f'Config {i+1}' for i in range(len(top_configs))])
        ax4.set_xlabel('Mean Composite Score')
        ax4.set_title('Top Configurations')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f'hyperparameter_optimization_{method_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create text report
        report_lines = [
            f"Hyperparameter Optimization Report - {method_name.title()}",
            "=" * 60,
            "",
            f"📊 OPTIMIZATION SUMMARY",
            "-" * 25,
            f"Total Configurations Tested: {len(results_df)}",
            f"Best Score: {results_df['mean_score'].max():.4f}",
            f"Mean Score: {results_df['mean_score'].mean():.4f}",
            f"Score Standard Deviation: {results_df['mean_score'].std():.4f}",
            "",
            f"🏆 BEST CONFIGURATION",
            "-" * 25
        ]
        
        # Best configuration details
        best_idx = results_df['mean_score'].idxmax()
        best_config = results_df.loc[best_idx]
        
        report_lines.extend([
            f"Score: {best_config['mean_score']:.4f}",
            f"VORP: {best_config['mean_vorp']:.2f}",
            f"Standard Deviation: {best_config['std_score']:.4f}",
            f"Trials: {best_config['n_trials']}",
            "",
            "Parameters:"
        ])
        
        best_params = best_config['parameters']
        for param_name, param_value in best_params.items():
            if isinstance(param_value, float):
                report_lines.append(f"  {param_name}: {param_value:.4f}")
            else:
                report_lines.append(f"  {param_name}: {param_value}")
        
        # Top 5 configurations
        report_lines.extend([
            "",
            f"🔝 TOP 5 CONFIGURATIONS",
            "-" * 30
        ])
        
        top_5 = results_df.nlargest(5, 'mean_score')
        for rank, (_, config) in enumerate(top_5.iterrows(), 1):
            report_lines.extend([
                f"{rank}. Score: {config['mean_score']:.4f}, VORP: {config['mean_vorp']:.2f}",
                f"   Parameters: {dict(config['parameters'])}",
                ""
            ])
        
        # Parameter insights
        report_lines.extend([
            f"📈 PARAMETER INSIGHTS",
            "-" * 25
        ])
        
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            correlation = results_df[param_col].corr(results_df['mean_score'])
            best_value = best_config[param_col]
            
            report_lines.extend([
                f"{param_name}:",
                f"  Best Value: {best_value}",
                f"  Score Correlation: {correlation:.3f}",
                ""
            ])
        
        report_text = "\n".join(report_lines)
        
        # Save text report
        text_report_path = plot_path.replace('.png', '_report.txt')
        with open(text_report_path, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Optimization report saved:")
        print(f"   📊 Visualizations: {plot_path}")
        print(f"   📋 Text Report: {text_report_path}")
        
        return report_text


def create_mcts_parameter_space() -> ParameterSpace:
    """Create a comprehensive parameter space for MCTS strategies"""
    
    space = ParameterSpace()
    
    # MCTS Core Parameters
    space.add_parameter('simulations_per_move', 'integer', (200, 1000))
    space.add_parameter('exploration_constant', 'float', (0.5, 3.0))
    space.add_parameter('risk_penalty', 'float', (0.05, 0.5))
    
    # Strategy Weights
    space.add_parameter('vorp_weight', 'float', (0.5, 1.5))
    space.add_parameter('bye_week_weight', 'float', (0.0, 0.3))
    space.add_parameter('injury_weight', 'float', (0.0, 0.4))
    space.add_parameter('history_weight', 'float', (0.0, 0.3))
    
    # Position Preferences
    space.add_parameter('early_qb_penalty', 'float', (0.0, 1.0))
    space.add_parameter('rb_preference', 'float', (0.8, 1.2))
    space.add_parameter('wr_preference', 'float', (0.8, 1.2))
    
    # Risk Management
    space.add_parameter('rookie_uncertainty_penalty', 'float', (0.1, 0.8))
    space.add_parameter('injury_risk_threshold', 'float', (0.3, 0.7))
    
    # Constraints
    def weight_constraint(params):
        # Total strategy weights shouldn't be too extreme
        total_weight = (params['bye_week_weight'] + 
                       params['injury_weight'] + 
                       params['history_weight'])
        return total_weight <= 1.0
    
    def preference_constraint(params):
        # Position preferences should be reasonable
        return (0.5 <= params['rb_preference'] + params['wr_preference'] <= 2.5)
    
    space.add_constraint(weight_constraint)
    space.add_constraint(preference_constraint)
    
    return space


def create_strategy_factory(backtester) -> Callable[[Dict], Any]:
    """Create a factory function that builds strategies from parameters"""
    
    def strategy_factory(parameters: Dict[str, Any]):
        """Create a strategy instance from parameters"""
        
        class ParameterizedMCTS:
            def __init__(self, params):
                self.params = params
                self.last_pick_reasoning = "Parameterized MCTS"
            
            def search(self, draft_state):
                available = list(draft_state.available_players)
                if not available:
                    return None
                
                # Simple parameterized strategy
                scores = []
                
                for player in available:
                    # Base VORP score
                    vorp_score = player.vorp * self.params.get('vorp_weight', 1.0)
                    
                    # Risk penalties
                    risk_penalty = (
                        player.metadata.get('risk_sigma', 0.3) * 
                        self.params.get('risk_penalty', 0.2)
                    )
                    
                    injury_penalty = (
                        player.metadata.get('injury_risk_score', 0.3) * 
                        self.params.get('injury_weight', 0.2)
                    )
                    
                    # Position preferences
                    position_bonus = 0.0
                    if player.position == 'RB':
                        position_bonus = self.params.get('rb_preference', 1.0) - 1.0
                    elif player.position == 'WR':
                        position_bonus = self.params.get('wr_preference', 1.0) - 1.0
                    elif player.position == 'QB':
                        # Penalty for early QB
                        current_round = draft_state.current_round
                        if current_round <= 3:
                            position_bonus = -self.params.get('early_qb_penalty', 0.5)
                    
                    # Final score
                    final_score = (vorp_score + position_bonus - 
                                 risk_penalty - injury_penalty)
                    
                    scores.append((player, final_score))
                
                # Return best player
                best_player = max(scores, key=lambda x: x[1])[0]
                self.last_pick_reasoning = f"Parameterized MCTS (Score: {max(scores, key=lambda x: x[1])[1]:.2f})"
                return best_player
        
        return ParameterizedMCTS(parameters)
    
    return strategy_factory


if __name__ == "__main__":
    # Example usage
    print("🔧 Testing Hyperparameter Optimization System")
    
    from ..utils.data_loader import create_sample_player_pool, get_default_league_settings
    from .backtesting import DraftBacktester
    
    # Create test setup
    player_pool = create_sample_player_pool(200)
    league_settings = get_default_league_settings()
    backtester = DraftBacktester(player_pool, league_settings)
    backtester.n_simulations = 5  # Reduce for testing speed
    
    # Create parameter space and factory
    parameter_space = create_mcts_parameter_space()
    strategy_factory = create_strategy_factory(backtester)
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(backtester, strategy_factory, parameter_space)
    
    # Test random search
    print("\n🎲 Testing Random Search...")
    results_df = optimizer.random_search(n_configurations=10, n_trials_per_config=3)
    
    # Create report
    report = optimizer.create_optimization_report(results_df, "random_search")
    
    print(f"\n✅ Hyperparameter optimization demo complete!")
    print(f"🏆 Best parameters found: {optimizer.best_parameters}")
    print(f"📊 Best score: {optimizer.best_score:.4f}")
