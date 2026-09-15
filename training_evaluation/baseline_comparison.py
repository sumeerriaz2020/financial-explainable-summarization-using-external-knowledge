"""
Baseline Comparison
===================

Compare our system against the baselines reported in the manuscript:
- BART-large (baseline 1)
- BART + KG (baseline 6, the resource-equivalent knowledge-graph control)
- PEGASUS-large, T5-base, FinanceSum, GPT-4 (gpt-4-0613) — Table 6

All reference values below are the reported test-set figures (mean over five
seeded runs). See docs/PAPER_FACTS.md and configs/baselines.yaml.

Reference: Table 3 (summarization quality), Table 4 (explainability),
Table 6 (state-of-the-art comparison), Table 8 (KG integration strategies)
"""

import torch
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineComparator:
    """
    Compare system performance against the reported baselines

    Systems from Table 3:
    - BART-large
    - BART + KG
    - Our System
    """

    def __init__(self):
        """Initialize baseline comparator"""
        # Summarization quality, Table 3 (mean +/- SD over 5 seeded runs).
        # Factual consistency is from Table 8: "No KG Integration" for BART,
        # "Retrieval-Augmented" (== BART+KG integration path) for BART + KG,
        # "Hybrid Neural-Symbolic" for our system.
        self.baseline_results = {
            'BART-large': {
                'rouge-l': 0.421, 'rouge-l_std': 0.018,
                'rouge-1': 0.387, 'rouge-1_std': 0.022,
                'rouge-2': 0.184, 'rouge-2_std': 0.015,
                'bertscore': 0.612, 'bertscore_std': 0.025,
                'bleu-4': 0.156, 'bleu-4_std': 0.012,
                'factual_consistency': 72.4,
                'description': 'facebook/bart-large fine-tuned (baseline 1)'
            },
            'BART + KG': {
                'rouge-l': 0.438, 'rouge-l_std': 0.015,
                'rouge-1': 0.401, 'rouge-1_std': 0.019,
                'rouge-2': 0.195, 'rouge-2_std': 0.013,
                'bertscore': 0.634, 'bertscore_std': 0.021,
                'bleu-4': 0.167, 'bleu-4_std': 0.011,
                'factual_consistency': 81.4,
                'description': 'Linearized 3-hop KG triples concatenated to input (baseline 6, Table 2)'
            },
            'Our System': {
                'rouge-l': 0.497, 'rouge-l_std': 0.014,
                'rouge-1': 0.452, 'rouge-1_std': 0.016,
                'rouge-2': 0.227, 'rouge-2_std': 0.012,
                'bertscore': 0.673, 'bertscore_std': 0.018,
                'bleu-4': 0.191, 'bleu-4_std': 0.009,
                'factual_consistency': 87.6,
                'description': 'Hybrid neural-symbolic with FIBO'
            }
        }

        # State-of-the-art comparison, Table 6. Expl. = expert-rated explainability
        # (1-10). GPT-4 is the gpt-4-0613 snapshot frozen at protocol design and is
        # NOT a current state-of-the-art comparator (Section 5.3). Note that GPT-4
        # scores higher than our system on BERTScore.
        self.sota_results = {
            'BART + LIME': {'rouge-l': 0.438, 'bertscore': 0.634, 'explainability': 3.2},
            'PEGASUS + Attn': {'rouge-l': 0.442, 'bertscore': 0.641, 'explainability': 4.1},
            'T5 + SHAP': {'rouge-l': 0.445, 'bertscore': 0.638, 'explainability': 3.8},
            'FinanceSum': {'rouge-l': 0.451, 'bertscore': 0.652, 'explainability': 4.5},
            'GPT-4 (gpt-4-0613, 5-shot)': {'rouge-l': 0.471, 'bertscore': 0.679, 'explainability': 5.8},
            'Our System': {'rouge-l': 0.497, 'bertscore': 0.673, 'explainability': 6.7}
        }

        # Explainability, Table 4. Two-way comparison only: baseline values are the
        # per-metric best baseline (BART+KG for SSI and CPS; BART-large for Trust
        # Calibration, Explanation Consistency and TCC).
        self.explainability_results = {
            'Baseline': {
                'ssi': 0.61, 'ssi_std': 0.07,
                'cps': 0.34, 'cps_std': 0.08,
                'trust_calibration': 0.52, 'trust_calibration_std': 0.09,
                'consistency': 0.43, 'consistency_std': 0.07,
                'tcc': 0.38, 'tcc_std': 0.06
            },
            'Ours': {
                'ssi': 0.74, 'ssi_std': 0.06,
                'cps': 0.51, 'cps_std': 0.07,
                'trust_calibration': 0.63, 'trust_calibration_std': 0.08,
                'consistency': 0.59, 'consistency_std': 0.06,
                'tcc': 0.54, 'tcc_std': 0.07
            }
        }

        # Reported significance (Section 4): paired t-tests, alpha = 0.05, across
        # the five seeded runs. Recorded as reported; not recomputed here.
        self.reported_significance = {
            'vs_bart': {'rouge-l': '<0.001', 'rouge-1': '<0.001', 'rouge-2': '<0.001',
                        'bertscore': '<0.001', 'bleu-4': '<0.001'},
            'vs_bart_kg': {'rouge-l': '<0.01', 'rouge-2': '<0.01'},
            'explainability': {'ssi': '<0.001', 'cps': '<0.001', 'trust_calibration': '<0.01',
                               'consistency': '<0.001', 'tcc': '<0.01'}
        }

        logger.info("Baseline Comparator initialized")

    def compare(
        self,
        our_results: Dict[str, float],
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Compare our results against baselines

        Args:
            our_results: Our system's evaluation results
            save_dir: Directory to save comparison results

        Returns:
            Comparison analysis
        """
        logger.info("=" * 70)
        logger.info("Baseline Comparison")
        logger.info("=" * 70)

        # Compute improvements
        improvements = self._compute_improvements(our_results)

        # Create comparison tables
        comparison_df = self._create_comparison_table(our_results)

        # Generate visualizations
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            self._plot_comparisons(comparison_df, save_dir)
            self._save_results(improvements, save_dir)

        # Print summary
        self._print_comparison(improvements)

        return {
            'improvements': improvements,
            'reported_significance': self.reported_significance,
            'comparison_table': comparison_df
        }

    def _compute_improvements(self, our_results: Dict) -> Dict:
        """Compute improvement percentages"""
        improvements = {}

        # Compare against BART-large
        baseline = self.baseline_results['BART-large']

        for metric in ['rouge-l', 'rouge-1', 'rouge-2', 'bertscore', 'bleu-4', 'factual_consistency']:
            if metric in our_results:
                baseline_val = baseline[metric]
                our_val = our_results[metric]

                improvement = ((our_val - baseline_val) / baseline_val) * 100

                improvements[metric] = {
                    'baseline': baseline_val,
                    'ours': our_val,
                    'improvement_pct': improvement,
                    'absolute_diff': our_val - baseline_val
                }

        # Compare explainability metrics
        exp_baseline = self.explainability_results['Baseline']

        for metric in ['ssi', 'cps', 'trust_calibration', 'consistency', 'tcc']:
            if metric in our_results:
                baseline_val = exp_baseline[metric]
                our_val = our_results[metric]

                improvement = ((our_val - baseline_val) / baseline_val) * 100

                improvements[metric] = {
                    'baseline': baseline_val,
                    'ours': our_val,
                    'improvement_pct': improvement,
                    'absolute_diff': our_val - baseline_val
                }

        return improvements

    def _create_comparison_table(self, our_results: Dict) -> pd.DataFrame:
        """Create comparison table (Table 3 format)"""

        models = ['BART-large', 'BART + KG', 'Our System']
        ours = self.baseline_results['Our System']

        summarization_data = {
            'Model': models,
            'ROUGE-L': [
                self.baseline_results['BART-large']['rouge-l'],
                self.baseline_results['BART + KG']['rouge-l'],
                our_results.get('rouge-l', our_results.get('rougeL', ours['rouge-l']))
            ],
            'BERTScore': [
                self.baseline_results['BART-large']['bertscore'],
                self.baseline_results['BART + KG']['bertscore'],
                our_results.get('bertscore', ours['bertscore'])
            ],
            'Factual (%)': [
                self.baseline_results['BART-large']['factual_consistency'],
                self.baseline_results['BART + KG']['factual_consistency'],
                our_results.get('factual_consistency', ours['factual_consistency'])
            ]
        }

        df = pd.DataFrame(summarization_data)

        return df

    def _plot_comparisons(self, comparison_df: pd.DataFrame, save_dir: Path):
        """Generate comparison visualizations"""

        # Plot 1: Summarization metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['ROUGE-L', 'BERTScore', 'Factual (%)']
        colors = ['#e74c3c', '#3498db', '#2ecc71']

        for i, metric in enumerate(metrics):
            ax = axes[i]

            values = comparison_df[metric].values
            models = comparison_df['Model'].values

            bars = ax.bar(models, values, color=colors)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'({"abc"[i]}) {metric}', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}' if metric != 'Factual (%)' else f'{height:.1f}%',
                    ha='center', va='bottom'
                )

            # Rotate x labels
            ax.set_xticklabels(models, rotation=15, ha='right')

        plt.tight_layout()
        plt.savefig(save_dir / 'baseline_comparison_summarization.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved: {save_dir / 'baseline_comparison_summarization.png'}")

        # Plot 2: Explainability metrics (Table 4, two-way comparison)
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        exp_metrics = ['ssi', 'cps', 'trust_calibration', 'consistency', 'tcc']
        exp_labels = ['SSI', 'CPS', 'Trust Cal.', 'Expl. Cons.', 'TCC']

        for i, (metric, label) in enumerate(zip(exp_metrics, exp_labels)):
            ax = axes[i]

            models = ['Baseline', 'Ours']
            values = [
                self.explainability_results[m][metric]
                for m in models
            ]

            bars = ax.bar(models, values, color=colors[:2])
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f'({"abcde"[i]}) {label}', fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom'
                )

        plt.tight_layout()
        plt.savefig(save_dir / 'baseline_comparison_explainability.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved: {save_dir / 'baseline_comparison_explainability.png'}")

    def _save_results(self, improvements: Dict, save_dir: Path):
        """Save comparison results"""
        results_path = save_dir / 'baseline_comparison.json'

        with open(results_path, 'w') as f:
            json.dump(improvements, f, indent=2)

        logger.info(f"  Saved: {results_path}")

        # Save as markdown table
        md_path = save_dir / 'baseline_comparison.md'
        self._save_markdown_table(improvements, md_path)

        logger.info(f"  Saved: {md_path}")

    def _save_markdown_table(self, improvements: Dict, path: Path):
        """Save results as markdown table"""
        with open(path, 'w') as f:
            f.write("# Baseline Comparison\n\n")

            f.write("## Summarization Metrics (vs BART-large)\n\n")
            f.write("| Metric | Baseline | Ours | Improvement |\n")
            f.write("|--------|----------|------|-------------|\n")

            for metric in ['rouge-l', 'rouge-1', 'rouge-2', 'bertscore', 'bleu-4', 'factual_consistency']:
                if metric in improvements:
                    data = improvements[metric]
                    f.write(
                        f"| {metric} | {data['baseline']:.3f} | "
                        f"{data['ours']:.3f} | "
                        f"{data['improvement_pct']:+.1f}% |\n"
                    )

            f.write("\n## Explainability Metrics\n\n")
            f.write("| Metric | Baseline | Ours | Improvement |\n")
            f.write("|--------|----------|------|-------------|\n")

            for metric in ['ssi', 'cps', 'trust_calibration', 'consistency', 'tcc']:
                if metric in improvements:
                    data = improvements[metric]
                    f.write(
                        f"| {metric.upper()} | {data['baseline']:.3f} | "
                        f"{data['ours']:.3f} | "
                        f"{data['improvement_pct']:+.1f}% |\n"
                    )

    def _print_comparison(self, improvements: Dict):
        """Print comparison summary"""
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 70)

        logger.info("\nSummarization Improvements vs BART-large:")
        for metric in ['rouge-l', 'rouge-1', 'rouge-2', 'bertscore', 'bleu-4', 'factual_consistency']:
            if metric in improvements:
                data = improvements[metric]
                logger.info(
                    f"  {metric.upper()}: "
                    f"{data['baseline']:.3f} -> {data['ours']:.3f} "
                    f"({data['improvement_pct']:+.1f}%)"
                )

        logger.info("\nExplainability Improvements:")
        for metric in ['ssi', 'cps', 'trust_calibration', 'consistency', 'tcc']:
            if metric in improvements:
                data = improvements[metric]
                logger.info(
                    f"  {metric.upper()}: "
                    f"{data['baseline']:.3f} -> {data['ours']:.3f} "
                    f"({data['improvement_pct']:+.1f}%)"
                )

        logger.info("\n" + "=" * 70)

    def run_baseline_evaluation(
        self,
        test_data,
        baseline_model_name: str = "facebook/bart-large"
    ) -> Dict:
        """
        Run evaluation on baseline model

        Args:
            test_data: Test dataset
            baseline_model_name: Baseline model identifier

        Returns:
            Baseline evaluation results
        """
        logger.info(f"\nEvaluating baseline: {baseline_model_name}")

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # Load baseline model
        model = AutoModelForSeq2SeqLM.from_pretrained(baseline_model_name)
        tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)

        # Evaluate (implementation depends on your pipeline)
        # results = evaluate_model(model, tokenizer, test_data)

        results = {}  # Placeholder

        return results


# Example usage
if __name__ == "__main__":
    print("Baseline Comparison Module")
    print("=" * 70)

    print("\nSummarization quality (Table 3, mean over 5 seeded runs):")
    print("  1. BART-large:  ROUGE-L 0.421 | BERTScore 0.612")
    print("  2. BART + KG:   ROUGE-L 0.438 | BERTScore 0.634")
    print("  3. Our System:  ROUGE-L 0.497 | BERTScore 0.673")
    print("     (+18.1% ROUGE-L vs BART-large; +13.5% vs BART + KG)")

    print("\nFactual consistency (Table 8):")
    print("  No KG 72.4% | Retrieval-augmented (BART + KG) 81.4% |")
    print("  Attention-based KG 83.7% | Hybrid neural-symbolic 87.6%")

    print("\nGPT-4 (gpt-4-0613, 5-shot; Table 6): ROUGE-L 0.471 | BERTScore 0.679")
    print("  GPT-4 scores higher on BERTScore; expert preference 54% (p = 0.35).")

    print("\n" + "=" * 70)
    print("Baseline comparator ready!")
    print("=" * 70)
