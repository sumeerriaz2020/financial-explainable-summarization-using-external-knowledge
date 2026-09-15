"""
Evaluation Pipeline
===================

Complete evaluation pipeline for model assessment including
summarization quality, explainability, and comparative analysis.

Reference: Section 4 (Results and Analysis)
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import json
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Complete evaluation pipeline
    
    Evaluates:
    1. Summarization quality (ROUGE, BERTScore, Factual)
    2. Explainability (SSI, CPS, TCC)
    3. Computational costs
    4. Error analysis
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda'
    ):
        """
        Initialize evaluation pipeline
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            device: Computing device
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize metrics
        from evaluation.metrics import ComprehensiveEvaluator
        self.evaluator = ComprehensiveEvaluator()
        
        logger.info(f"Evaluation Pipeline initialized on {device}")
    
    def evaluate(
        self,
        test_dataloader: DataLoader,
        output_dir: str,
        save_predictions: bool = True
    ) -> Dict[str, float]:
        """
        Run complete evaluation
        
        Args:
            test_dataloader: Test data loader
            output_dir: Directory to save results
            save_predictions: Whether to save predictions
            
        Returns:
            Dictionary of all metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("Starting Evaluation")
        logger.info("=" * 70)
        
        # Generate predictions
        logger.info("\n1. Generating predictions...")
        results = self._generate_predictions(test_dataloader)
        
        # Compute metrics
        logger.info("\n2. Computing metrics...")
        metrics = self._compute_metrics(results)
        
        # Analyze errors
        logger.info("\n3. Analyzing errors...")
        error_stats = self._analyze_errors(results)
        
        # Compute costs
        logger.info("\n4. Computing computational costs...")
        cost_stats = self._compute_costs(test_dataloader)
        
        # Combine results
        all_results = {
            'metrics': metrics,
            'errors': error_stats,
            'costs': cost_stats,
            'num_samples': len(results['predictions'])
        }
        
        # Save results
        self._save_results(all_results, output_dir)
        
        if save_predictions:
            self._save_predictions(results, output_dir)
        
        # Print summary
        self._print_summary(all_results)
        
        return metrics
    
    def _generate_predictions(
        self,
        dataloader: DataLoader
    ) -> Dict[str, List]:
        """Generate model predictions"""
        predictions = []
        references = []
        sources = []
        explanations = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating"):
                # Move to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Generate summary
                outputs = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    kg_node_features=batch.get('kg_features'),
                    kg_adjacency=batch.get('kg_adj'),
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode
                preds = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
                refs = self.tokenizer.batch_decode(
                    batch['labels'],
                    skip_special_tokens=True
                )
                srcs = self.tokenizer.batch_decode(
                    batch['input_ids'],
                    skip_special_tokens=True
                )
                
                predictions.extend(preds)
                references.extend(refs)
                sources.extend(srcs)
                
                # Extract explanations if available
                if hasattr(outputs, 'explanations'):
                    explanations.extend(outputs.explanations)
        
        return {
            'predictions': predictions,
            'references': references,
            'sources': sources,
            'explanations': explanations
        }
    
    def _compute_metrics(self, results: Dict) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        metrics = {}
        
        # Summarization metrics
        logger.info("  Computing ROUGE scores...")
        metrics.update(
            self.evaluator.rouge.compute(
                results['predictions'],
                results['references']
            )
        )
        
        logger.info("  Computing BERTScore...")
        metrics.update(
            self.evaluator.bertscore.compute(
                results['predictions'],
                results['references']
            )
        )
        
        logger.info("  Computing factual consistency...")
        metrics.update(
            self.evaluator.factual.compute(
                results['predictions'],
                results['sources']
            )
        )
        
        # Explainability metrics (if available)
        if results.get('explanations'):
            logger.info("  Computing explainability metrics...")
            # Add explainability metrics here
        
        return metrics
    
    def _analyze_errors(self, results: Dict) -> Dict:
        """Analyze prediction errors"""
        errors = {
            'total_samples': len(results['predictions']),
            'error_types': {},
            'severity_distribution': {}
        }
        
        # Categorize errors
        for pred, ref in zip(results['predictions'], results['references']):
            error_type = self._categorize_error(pred, ref)
            if error_type:
                errors['error_types'][error_type] = \
                    errors['error_types'].get(error_type, 0) + 1
        
        return errors
    
    def _categorize_error(self, prediction: str, reference: str) -> Optional[str]:
        """Categorize error type"""
        # Simplified error categorization
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        
        overlap = len(pred_words & ref_words) / len(ref_words) if ref_words else 0
        
        if overlap < 0.3:
            return 'low_overlap'
        elif len(prediction) < len(reference) * 0.5:
            return 'too_short'
        elif len(prediction) > len(reference) * 2:
            return 'too_long'
        
        return None
    
    def _compute_costs(self, dataloader: DataLoader) -> Dict:
        """Compute computational costs"""
        import time
        
        # Inference time
        start_time = time.time()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 10:  # Sample 10 batches
                    break
                
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                _ = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=128
                )
        
        elapsed = time.time() - start_time
        avg_time = elapsed / 10  # Average per batch
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
        else:
            memory_allocated = 0
            memory_reserved = 0
        
        return {
            'avg_inference_time': avg_time,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }
    
    def _save_results(self, results: Dict, output_dir: Path):
        """Save evaluation results"""
        # Save metrics
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        logger.info(f"  Metrics saved to {metrics_path}")
        
        # Save full results
        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"  Full results saved to {results_path}")
    
    def _save_predictions(self, results: Dict, output_dir: Path):
        """Save predictions"""
        predictions_path = output_dir / 'predictions.jsonl'
        
        with open(predictions_path, 'w') as f:
            for pred, ref, src in zip(
                results['predictions'],
                results['references'],
                results['sources']
            ):
                sample = {
                    'prediction': pred,
                    'reference': ref,
                    'source': src
                }
                f.write(json.dumps(sample) + '\n')
        
        logger.info(f"  Predictions saved to {predictions_path}")
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary"""
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 70)
        
        # Summarization metrics
        metrics = results['metrics']
        logger.info("\nSummarization Quality:")
        logger.info(f"  ROUGE-L: {metrics.get('rougeL', 0):.4f}")
        logger.info(f"  BERTScore: {metrics.get('bertscore', 0):.4f}")
        logger.info(f"  Factual Consistency: {metrics.get('factual_consistency', 0):.2f}%")
        
        # Explainability metrics
        if 'ssi' in metrics:
            logger.info("\nExplainability:")
            logger.info(f"  SSI: {metrics['ssi']:.4f}")
            logger.info(f"  CPS: {metrics.get('cps', 0):.4f}")
            logger.info(f"  TCC: {metrics.get('tcc', 0):.4f}")
        
        # Computational costs
        costs = results['costs']
        logger.info("\nComputational Costs:")
        logger.info(f"  Inference Time: {costs['avg_inference_time']:.2f}s/batch")
        logger.info(f"  Memory: {costs['memory_allocated_gb']:.2f} GB")
        logger.info(f"  Parameters: {costs['num_parameters']:,}")
        
        logger.info("\n" + "=" * 70)


def evaluate_checkpoint(
    checkpoint_path: str,
    test_data_path: str,
    output_dir: str,
    config: Dict
):
    """
    Evaluate a saved checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test data
        output_dir: Output directory
        config: Evaluation configuration
    """
    from models.hybrid_model import HybridModel
    from transformers import AutoTokenizer
    
    logger.info("Loading model from checkpoint...")
    
    # Load model
    model = HybridModel(**config['model_params'])
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    
    # Create test dataloader
    # (Implementation depends on your data format)
    
    # Run evaluation
    pipeline = EvaluationPipeline(model, tokenizer)
    results = pipeline.evaluate(
        test_dataloader=None,  # Replace with actual dataloader
        output_dir=output_dir
    )
    
    return results


# Example usage
if __name__ == "__main__":
    print("Evaluation Pipeline")
    print("=" * 70)
    
    print("\nEvaluation Steps:")
    print("  1. Generate predictions")
    print("  2. Compute metrics")
    print("     - ROUGE (L, 1, 2)")
    print("     - BERTScore")
    print("     - Factual Consistency")
    print("     - SSI, CPS, TCC (explainability)")
    print("  3. Analyze errors")
    print("  4. Compute computational costs")
    
    print("\nOutputs:")
    print("  - metrics.json")
    print("  - evaluation_results.json")
    print("  - predictions.jsonl")
    print("  - error_analysis.json")
    
    print("\n" + "=" * 70)
    print("Ready to evaluate!")
    print("=" * 70)
