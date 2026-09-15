"""
Hyperparameter Optimization
============================

Bayesian optimization for hyperparameter tuning using Optuna framework.
Optimizes learning rates, loss weights, architecture parameters.

Search space follows Section 3.6.2: loss weights alpha and beta (gamma derived as
1 - alpha - beta), learning rates [1e-5, 2e-4], batch size {8, 16, 32}, hidden
dimension {512, 768, 1024}, attention heads {4, 8, 12}, reasoning hops {2, 3, 4},
dropout [0.1, 0.3]. The alpha/beta ranges are not reported in the paper.

Reference: Section 3.6.2 (Hyperparameter Optimization)
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Bayesian Hyperparameter Optimization using Optuna
    
    Optimizes:
    - Learning rates (stage1_lr, stage2_lr, stage3_lr)
    - Loss weights (α, β, γ)
    - Dropout rates
    - Hidden dimensions
    - Number of attention heads
    """
    
    def __init__(
        self,
        model_class,
        train_fn,
        eval_fn,
        n_trials: int = 50,
        study_name: str = "financial_summarization"
    ):
        """
        Initialize optimizer
        
        Args:
            model_class: Model class to instantiate
            train_fn: Training function
            eval_fn: Evaluation function
            n_trials: Number of optimization trials
            study_name: Study identifier
        """
        self.model_class = model_class
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.n_trials = n_trials
        self.study_name = study_name
        
        # Setup Optuna
        self.sampler = TPESampler(seed=42)
        self.pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
        
        logger.info(f"Hyperparameter Optimizer initialized")
        logger.info(f"Study: {study_name}, Trials: {n_trials}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function to minimize (validation loss)
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss
        """
        # Sample hyperparameters
        params = self._sample_hyperparameters(trial)
        
        logger.info(f"\nTrial {trial.number}:")
        logger.info(f"Parameters: {params}")
        
        try:
            # Initialize model with sampled parameters
            model = self.model_class(**params['model_params'])
            
            # Train model
            train_metrics = self.train_fn(
                model=model,
                config=params['training_params'],
                max_epochs=5  # Reduced for HP search
            )
            
            # Evaluate model
            val_loss = self.eval_fn(model)
            
            # Report intermediate value for pruning
            trial.report(val_loss, step=5)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            logger.info(f"Trial {trial.number} validation loss: {val_loss:.4f}")
            
            return val_loss
        
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('inf')
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Sample hyperparameters for trial"""
        
        # Learning rates
        # Learning rates, each searched over [1e-5, 2e-4] (Section 3.6.2)
        stage1_lr = trial.suggest_float('stage1_lr', 1e-5, 2e-4, log=True)
        stage2_lr = trial.suggest_float('stage2_lr', 1e-5, 2e-4, log=True)
        stage3_lr = trial.suggest_float('stage3_lr', 1e-5, 2e-4, log=True)
        
        # Loss weights: search over alpha and beta; gamma derived as 1 - alpha - beta
        # (range bounds not reported in the paper)
        alpha = trial.suggest_float('alpha', 0.5, 0.9)
        beta = trial.suggest_float('beta', 0.1, 0.3)
        gamma = 1.0 - alpha - beta
        
        # Dropout
        dropout = trial.suggest_float('dropout', 0.1, 0.3)
        
        # Architecture parameters
        hidden_dim = trial.suggest_categorical('hidden_dim', [512, 768, 1024])
        num_attention_heads = trial.suggest_categorical('num_attention_heads', [4, 8, 12])
        num_reasoning_hops = trial.suggest_int('num_reasoning_hops', 2, 4)
        
        # Training parameters
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        
        return {
            'model_params': {
                'hidden_dim': hidden_dim,
                'num_attention_heads': num_attention_heads,
                'num_reasoning_hops': num_reasoning_hops,
                'dropout': dropout
            },
            'training_params': {
                'stage1_lr': stage1_lr,
                'stage2_lr': stage2_lr,
                'stage3_lr': stage3_lr,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'batch_size': batch_size,
                'warmup_ratio': 0.1  # fixed: linear warm-up over first 10% of steps
            }
        }
    
    def optimize(self, direction: str = 'minimize') -> Dict:
        """
        Run optimization study
        
        Args:
            direction: Optimization direction ('minimize' or 'maximize')
            
        Returns:
            Best hyperparameters
        """
        logger.info("=" * 70)
        logger.info("Starting Hyperparameter Optimization")
        logger.info("=" * 70)
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info("\n" + "=" * 70)
        logger.info("Optimization Complete!")
        logger.info("=" * 70)
        logger.info(f"Best validation loss: {best_value:.4f}")
        logger.info("\nBest hyperparameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Save results
        self._save_results(study)
        
        return best_params
    
    def _save_results(self, study: optuna.Study):
        """Save optimization results"""
        output_dir = Path('hp_optimization_results')
        output_dir.mkdir(exist_ok=True)
        
        # Save best parameters
        best_params_path = output_dir / 'best_params.json'
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        logger.info(f"\nResults saved to {output_dir}/")
        
        # Print optimization statistics
        self._print_statistics(study)
    
    def _print_statistics(self, study: optuna.Study):
        """Print optimization statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("Optimization Statistics:")
        logger.info("=" * 70)
        
        logger.info(f"Number of trials: {len(study.trials)}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        
        # Completed vs pruned
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        
        logger.info(f"Completed trials: {completed}")
        logger.info(f"Pruned trials: {pruned}")
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            logger.info("\nParameter Importance:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {param}: {imp:.4f}")
        except:
            pass


class GridSearchOptimizer:
    """
    Grid search for hyperparameter optimization
    
    Simpler alternative to Bayesian optimization for quick experiments
    """
    
    def __init__(self, model_class, train_fn, eval_fn):
        """Initialize grid search optimizer"""
        self.model_class = model_class
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        
        logger.info("Grid Search Optimizer initialized")
    
    def search(self, param_grid: Dict) -> Dict:
        """
        Perform grid search
        
        Args:
            param_grid: Dictionary of parameter lists
            
        Returns:
            Best parameters
        """
        import itertools
        
        logger.info("=" * 70)
        logger.info("Starting Grid Search")
        logger.info("=" * 70)
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        logger.info(f"Total combinations: {len(combinations)}")
        
        best_params = None
        best_score = float('inf')
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            logger.info(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
            
            try:
                # Train and evaluate
                model = self.model_class(**params)
                self.train_fn(model, max_epochs=3)
                score = self.eval_fn(model)
                
                logger.info(f"Score: {score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    logger.info("✓ New best!")
            
            except Exception as e:
                logger.error(f"Failed: {e}")
                continue
        
        logger.info("\n" + "=" * 70)
        logger.info("Grid Search Complete!")
        logger.info("=" * 70)
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return best_params


# Example usage
if __name__ == "__main__":
    print("Hyperparameter Optimization")
    print("=" * 70)
    
    print("\nOptimization Methods:")
    print("  1. Bayesian Optimization (Optuna)")
    print("  2. Grid Search")
    
    print("\nHyperparameters to Optimize:")
    print("  Learning Rates:")
    print("    - stage1_lr, stage2_lr, stage3_lr: [1e-5, 2e-4]")
    
    print("\n  Loss Weights:")
    print("    - alpha: [0.5, 0.9]")
    print("    - beta: [0.1, 0.3]")
    print("    - gamma: derived")
    
    print("\n  Architecture:")
    print("    - hidden_dim: [512, 768, 1024]")
    print("    - num_attention_heads: [4, 8, 12]")
    print("    - num_reasoning_hops: [2, 3, 4]")
    
    print("\n  Training:")
    print("    - batch_size: [8, 16, 32]")
    
    print("\n  Dropout:")
    print("    - dropout: [0.1, 0.3]")
    
    print("\n" + "=" * 70)
    print("Example: Bayesian Optimization with Optuna")
    print("=" * 70)
    
    print("\nConfiguration:")
    print("  Sampler: TPE (Tree-structured Parzen Estimator)")
    print("  Pruner: Median (early stopping)")
    print("  Trials: 50")
    print("  Direction: minimize validation loss")
    
    print("\nExpected Runtime: ~8-12 hours (50 trials)")
    print("Output: best_params.json")
    
    print("\n" + "=" * 70)
    print("Ready to optimize!")
    print("=" * 70)
