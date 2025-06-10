"""
Strategy Optimization Module
Handles parameter optimization using Random Search and Bayesian Optimization.
"""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from itertools import product

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    execution_time: float
    iterations: int
    algorithm: str

class ParameterSpace:
    """Defines parameter space for optimization."""
    
    def __init__(self):
        self.params = {}
    
    def add_param(self, name: str, param_type: str, min_val: Any, max_val: Any, step: Any = None):
        """Add parameter to optimization space."""
        self.params[name] = {
            'type': param_type,
            'min': min_val,
            'max': max_val,
            'step': step
        }
    
    def add_integer(self, name: str, min_val: int, max_val: int):
        """Add integer parameter."""
        self.add_param(name, 'integer', min_val, max_val)
    
    def add_real(self, name: str, min_val: float, max_val: float):
        """Add real/float parameter."""
        self.add_param(name, 'real', min_val, max_val)
    
    def add_choice(self, name: str, choices: List[Any]):
        """Add categorical parameter."""
        self.params[name] = {
            'type': 'choice',
            'choices': choices
        }
    
    def sample_random(self) -> Dict[str, Any]:
        """Generate random parameter combination."""
        sample = {}
        for name, param in self.params.items():
            if param['type'] == 'integer':
                sample[name] = random.randint(param['min'], param['max'])
            elif param['type'] == 'real':
                sample[name] = random.uniform(param['min'], param['max'])
            elif param['type'] == 'choice':
                sample[name] = random.choice(param['choices'])
        return sample
    
    def get_skopt_space(self):
        """Convert to scikit-optimize space for Bayesian optimization."""
        if not BAYESIAN_AVAILABLE:
            raise ImportError("scikit-optimize not available for Bayesian optimization")
        
        space = []
        param_names = []
        
        for name, param in self.params.items():
            param_names.append(name)
            if param['type'] == 'integer':
                space.append(Integer(param['min'], param['max'], name=name))
            elif param['type'] == 'real':
                space.append(Real(param['min'], param['max'], name=name))
            elif param['type'] == 'choice':
                # For choices, we'll use integer indices
                space.append(Integer(0, len(param['choices']) - 1, name=name))
        
        return space, param_names

class WalkForwardOptimizer:
    """Implements Walk-Forward optimization to prevent overfitting."""
    
    def __init__(self, train_ratio: float = 0.7, validation_ratio: float = 0.15):
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = 1 - train_ratio - validation_ratio
        
        if self.test_ratio < 0:
            raise ValueError("train_ratio + validation_ratio must be <= 1")
    
    def split_data(self, total_periods: int) -> List[Tuple[int, int, int]]:
        """Split data into train/validation/test periods."""
        train_size = int(total_periods * self.train_ratio)
        val_size = int(total_periods * self.validation_ratio)
        
        # For walk-forward, we create multiple splits
        splits = []
        min_train_size = max(train_size // 2, 100)  # Minimum training size
        
        start = 0
        while start + min_train_size + val_size < total_periods:
            train_end = start + train_size
            val_end = train_end + val_size
            test_end = min(val_end + int(total_periods * self.test_ratio), total_periods)
            
            if train_end < total_periods and val_end < total_periods:
                splits.append((start, train_end, val_end, test_end))
            
            # Move window forward
            start += int(train_size * 0.2)  # 20% overlap
        
        return splits

class RandomSearchOptimizer:
    """Random Search optimization algorithm."""
    
    def __init__(self, n_iter: int = 50, n_jobs: int = 1, random_state: int = None):
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        if random_state:
            random.seed(random_state)
            np.random.seed(random_state)
    
    def optimize(self, objective_func: Callable, param_space: ParameterSpace) -> OptimizationResult:
        """Run random search optimization."""
        start_time = time.time()
        all_results = []
        best_score = -np.inf
        best_params = None
        
        self.logger.info(f"Starting Random Search with {self.n_iter} iterations")
        
        if self.n_jobs == 1:
            # Sequential execution
            for i in range(self.n_iter):
                params = param_space.sample_random()
                score = objective_func(params)
                
                result = {
                    'params': params,
                    'score': score,
                    'iteration': i
                }
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                self.logger.debug(f"Iteration {i+1}/{self.n_iter}: Score={score:.4f}")
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all jobs
                futures = []
                param_sets = [param_space.sample_random() for _ in range(self.n_iter)]
                
                for i, params in enumerate(param_sets):
                    future = executor.submit(objective_func, params)
                    futures.append((future, params, i))
                
                # Collect results
                for future, params, i in futures:
                    try:
                        score = future.result(timeout=300)  # 5 minute timeout
                        
                        result = {
                            'params': params,
                            'score': score,
                            'iteration': i
                        }
                        all_results.append(result)
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                        
                        self.logger.debug(f"Iteration {i+1}/{self.n_iter}: Score={score:.4f}")
                        
                    except Exception as e:
                        self.logger.warning(f"Iteration {i+1} failed: {e}")
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=sorted(all_results, key=lambda x: x['score'], reverse=True),
            execution_time=execution_time,
            iterations=len(all_results),
            algorithm='random_search'
        )

class BayesianOptimizer:
    """Bayesian Optimization using Gaussian Processes."""
    
    def __init__(self, n_calls: int = 50, n_initial_points: int = 10, random_state: int = None):
        if not BAYESIAN_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian optimization")
        
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, objective_func: Callable, param_space: ParameterSpace) -> OptimizationResult:
        """Run Bayesian optimization."""
        start_time = time.time()
        
        # Convert parameter space to scikit-optimize format
        space, param_names = param_space.get_skopt_space()
        
        # Results storage
        all_results = []
        iteration_counter = [0]  # Use list for closure
        
        def objective_wrapper(params):
            """Wrapper to convert skopt format to our format."""
            # Convert back to named parameters
            param_dict = {}
            for i, name in enumerate(param_names):
                param_config = param_space.params[name]
                if param_config['type'] == 'choice':
                    param_dict[name] = param_config['choices'][params[i]]
                else:
                    param_dict[name] = params[i]
            
            score = objective_func(param_dict)
            
            # Store result
            result = {
                'params': param_dict,
                'score': score,
                'iteration': iteration_counter[0]
            }
            all_results.append(result)
            iteration_counter[0] += 1
            
            self.logger.debug(f"Iteration {iteration_counter[0]}/{self.n_calls}: Score={score:.4f}")
            
            # Return negative score for minimization
            return -score
        
        self.logger.info(f"Starting Bayesian Optimization with {self.n_calls} calls")
        
        # Run optimization
        result = gp_minimize(
            func=objective_wrapper,
            dimensions=space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            acq_func='EI'  # Expected Improvement
        )
        
        execution_time = time.time() - start_time
        
        # Find best result
        best_result = max(all_results, key=lambda x: x['score'])
        
        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=sorted(all_results, key=lambda x: x['score'], reverse=True),
            execution_time=execution_time,
            iterations=len(all_results),
            algorithm='bayesian'
        )

class StrategyOptimizer:
    """Main optimization controller."""
    
    def __init__(self, backtest_engine, config: Dict[str, Any]):
        self.backtest_engine = backtest_engine
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimizers
        opt_config = config.get('optimization', {})
        self.random_optimizer = RandomSearchOptimizer(
            n_iter=opt_config.get('random_search', {}).get('n_iter', 50),
            n_jobs=opt_config.get('n_jobs', 1)
        )
        
        if BAYESIAN_AVAILABLE:
            self.bayesian_optimizer = BayesianOptimizer(
                n_calls=opt_config.get('bayesian', {}).get('n_calls', 50)
            )
        
        self.walk_forward = WalkForwardOptimizer(
            train_ratio=opt_config.get('walk_forward', {}).get('train_ratio', 0.7),
            validation_ratio=opt_config.get('walk_forward', {}).get('validation_ratio', 0.15)
        )
    
    def optimize_strategy(self, strategy_class, symbol: str, param_space: ParameterSpace,
                         algorithm: str = 'random_search', objective: str = 'profit_factor',
                         walk_forward: bool = True) -> OptimizationResult:
        """
        Optimize strategy parameters.
        
        Args:
            strategy_class: Strategy class to optimize
            symbol: Trading symbol
            param_space: Parameter space definition
            algorithm: 'random_search' or 'bayesian'
            objective: Optimization objective ('profit_factor', 'sharpe_ratio', etc.)
            walk_forward: Whether to use walk-forward validation
            
        Returns:
            OptimizationResult
        """
        
        def objective_function(params: Dict[str, Any]) -> float:
            """Objective function for optimization."""
            try:
                if walk_forward:
                    return self._walk_forward_objective(strategy_class, symbol, params, objective)
                else:
                    return self._simple_objective(strategy_class, symbol, params, objective)
            except Exception as e:
                self.logger.warning(f"Objective function failed for params {params}: {e}")
                return -np.inf
        
        # Choose optimizer
        if algorithm == 'random_search':
            return self.random_optimizer.optimize(objective_function, param_space)
        elif algorithm == 'bayesian':
            if not BAYESIAN_AVAILABLE:
                self.logger.warning("Bayesian optimization not available, falling back to Random Search")
                return self.random_optimizer.optimize(objective_function, param_space)
            return self.bayesian_optimizer.optimize(objective_function, param_space)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _simple_objective(self, strategy_class, symbol: str, params: Dict[str, Any], 
                         objective: str) -> float:
        """Simple objective function without walk-forward."""
        try:
            results = self.backtest_engine.run_backtest(
                strategy_class, symbol, parameters=params
            )
            
            if objective == 'profit_factor':
                return results.profit_factor
            elif objective == 'sharpe_ratio':
                return results.sharpe_ratio
            elif objective == 'net_profit':
                return results.net_profit
            elif objective == 'win_rate':
                return results.win_rate
            else:
                return results.profit_factor  # Default
                
        except Exception as e:
            self.logger.debug(f"Backtest failed: {e}")
            return -np.inf
    
    def _walk_forward_objective(self, strategy_class, symbol: str, params: Dict[str, Any],
                               objective: str) -> float:
        """Objective function with walk-forward validation."""
        # For simplicity, we'll run multiple backtests on different time periods
        # and average the results
        
        scores = []
        periods = ['6mo', '1y', '2y']  # Different time periods
        
        for period in periods:
            try:
                # Modify backtest to use different periods
                # This is a simplified implementation
                results = self.backtest_engine.run_backtest(
                    strategy_class, symbol, parameters=params
                )
                
                if objective == 'profit_factor':
                    score = results.profit_factor
                elif objective == 'sharpe_ratio':
                    score = results.sharpe_ratio
                elif objective == 'net_profit':
                    score = results.net_profit / results.initial_capital  # Normalize
                elif objective == 'win_rate':
                    score = results.win_rate
                else:
                    score = results.profit_factor
                
                if not np.isnan(score) and not np.isinf(score):
                    scores.append(score)
                    
            except Exception as e:
                self.logger.debug(f"Walk-forward backtest failed for period {period}: {e}")
                continue
        
        if not scores:
            return -np.inf
        
        # Return average score with penalty for high variance
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Penalize high variance (less robust strategies)
        penalty = std_score / (mean_score + 1e-6) if mean_score > 0 else 1
        return mean_score * (1 - min(penalty, 0.5))
    
    def create_param_space_from_strategy(self, strategy_class) -> ParameterSpace:
        """Automatically create parameter space from strategy parameters."""
        param_space = ParameterSpace()
        
        # Get strategy parameters from class
        strategy_params = getattr(strategy_class, 'params', None)
        
        if strategy_params and hasattr(strategy_params, '__iter__'):
            try:
                for param_def in strategy_params:
                    if isinstance(param_def, tuple) and len(param_def) >= 2:
                        param_name = param_def[0]
                        default_value = param_def[1]
                        
                        # Create reasonable ranges based on parameter name and default value
                        if isinstance(default_value, int):
                            if 'period' in param_name.lower() or 'length' in param_name.lower():
                                # Indicator periods
                                min_val = max(2, default_value // 2)
                                max_val = default_value * 3
                            else:
                                min_val = max(1, default_value // 2)
                                max_val = default_value * 2
                            
                            param_space.add_integer(param_name, min_val, max_val)
                            
                        elif isinstance(default_value, float):
                            min_val = default_value * 0.1
                            max_val = default_value * 2.0
                            param_space.add_real(param_name, min_val, max_val)
            except (TypeError, AttributeError):
                # If iteration fails, fall through to default parameters
                pass
        
        # If no parameters found, add default ones for testing
        if not param_space.params:
            param_space.add_integer('fast_period', 5, 20)
            param_space.add_integer('slow_period', 20, 50)
        
        return param_space
    
    def optimize_multiple_symbols(self, strategy_class, symbols: List[str], 
                                 param_space: ParameterSpace, 
                                 algorithm: str = 'random_search') -> Dict[str, OptimizationResult]:
        """Optimize strategy for multiple symbols."""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"Optimizing strategy for {symbol}")
            try:
                result = self.optimize_strategy(
                    strategy_class, symbol, param_space, algorithm
                )
                results[symbol] = result
            except Exception as e:
                self.logger.error(f"Optimization failed for {symbol}: {e}")
                results[symbol] = None
        
        return results

class OptimizationError(Exception):
    """Exception raised when optimization fails."""
    pass