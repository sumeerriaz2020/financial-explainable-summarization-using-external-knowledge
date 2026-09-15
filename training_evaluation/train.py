"""
Main Training Script
====================

Thin wrapper around the single multi-stage trainer in
algorithms/algorithm_6_training.py (Algorithm 6). All optimisation logic —
stage objectives, AdamW settings, the 10% linear warm-up, the Stage 3 x0.9
per-epoch decay and early stopping — lives there; this module only maps a plain
configuration dictionary onto TrainingConfig and saves the results.

Reference: Section 3.6, Algorithm 6 (Multi-Stage Training)
"""

import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.utils.data import DataLoader

from algorithms.algorithm_6_training import (
    MultiStageTrainer as Algorithm6Trainer,
    TrainingConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def config_from_dict(config: Dict, device: Optional[str] = None) -> TrainingConfig:
    """
    Build a TrainingConfig from a plain dictionary.

    Keys matching TrainingConfig fields are used directly. For convenience,
    'batch_size' sets all three stage batch sizes and 'max_grad_norm' sets
    gradient_clip. Unset fields keep the manuscript defaults.
    """
    valid = {f.name for f in fields(TrainingConfig)}
    kwargs = {k: v for k, v in config.items() if k in valid}

    if 'batch_size' in config:
        for stage in (1, 2, 3):
            kwargs.setdefault(f'stage{stage}_batch_size', config['batch_size'])
    if 'max_grad_norm' in config:
        kwargs.setdefault('gradient_clip', config['max_grad_norm'])
    if device is not None:
        kwargs['device'] = device

    return TrainingConfig(**kwargs)


class MultiStageTrainer(Algorithm6Trainer):
    """
    Multi-stage trainer accepting either a TrainingConfig or a plain dict.

    Stage 1: Domain pre-training (3 epochs)
    Stage 2: Knowledge integration (2 epochs)
    Stage 3: Joint optimization (5 epochs)
    """

    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Union[Dict, TrainingConfig],
        device: Optional[str] = None
    ):
        if not isinstance(config, TrainingConfig):
            config = config_from_dict(config, device)
        elif device is not None:
            config.device = device
        super().__init__(model, train_dataloader, val_dataloader, config)

    def train(self, output_dir: str) -> Dict:
        """
        Run all three stages and save the best model and training history.

        Args:
            output_dir: Directory for best_model.pt and training_history.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.train_all_stages()

        checkpoint_path = output_dir / 'best_model.pt'
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': vars(self.config),
                'best_val_performance': results['best_val_performance'],
            },
            checkpoint_path
        )
        logger.info(f"Saved best model: {checkpoint_path}")

        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(results['history'], f, indent=2, default=str)
        logger.info(f"Saved training history: {history_path}")

        return results


def load_checkpoint(path: str, model, device: str = 'cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


# Example usage
if __name__ == "__main__":
    print("Multi-Stage Training Script")
    print("=" * 70)

    config = config_from_dict({
        'batch_size': 16,
        'max_grad_norm': 1.0
    }, device='cpu')

    print("\nConfiguration (manuscript defaults, Section 3.6.2):")
    for field in fields(TrainingConfig):
        print(f"  {field.name}: {getattr(config, field.name)}")

    print("\nTraining Stages:")
    print(f"  Stage 1: Domain Pre-training ({config.stage1_epochs} epochs)")
    print(f"  Stage 2: Knowledge Integration ({config.stage2_epochs} epochs)")
    print(f"  Stage 3: Joint Optimization ({config.stage3_epochs} epochs)")

    print("\nLoss Function (Equation 10):")
    print("  L_total = α*L_sum + β*L_kg + γ*L_expl")
    print(f"  α={config.alpha}, β={config.beta}, γ={config.gamma}")

    print("\n" + "=" * 70)
    print("Ready to train!")
    print("=" * 70)
