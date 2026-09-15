"""
Visualization Utilities
=======================

Generate publication-quality figures from the paper including:
- Performance comparison charts (Tables 3 and 4)
- Architecture diagrams
- Knowledge graph visualizations
- Attention heatmaps
- Error analysis plots (Table 14 / Figure 6)

All hardcoded values are traceable to docs/PAPER_FACTS.md. If you change a number
here, change it in the paper first.

Reference: Section 4 (Results and Analysis)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperVisualizations:
    """Generate figures from paper"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizations
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            pass  # Use default if style unavailable
        
        self.colors = {
            'baseline': '#e74c3c',
            'bart_kg': '#3498db',
            'ours': '#2ecc71'
        }
        
        logger.info("Visualization utilities initialized")
    
    def plot_table3_performance(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot Table 3: Summarization Quality Comparison

        Metrics: ROUGE-L, BERTScore (Table 3); Factual Consistency (Table 8).

        Values are test-set means over the five seeded runs, not the validation
        figures (ROUGE-L 0.487, BERTScore 0.686) used for model selection.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Table 3 (ROUGE-L, BERTScore) and Table 8 (factual consistency).
        methods = ['Baseline\nBART', 'BART + KG', 'Our\nSystem']

        rouge_scores = [0.421, 0.438, 0.497]   # Table 3
        bertscore = [0.612, 0.634, 0.673]      # Table 3
        # Table 8: "No KG Integration", "Retrieval-Augmented" (== BART+KG),
        # "Hybrid Neural-Symbolic" (== ours).
        factual = [72.4, 81.4, 87.6]
        
        # ROUGE-L
        bars1 = axes[0].bar(methods, rouge_scores, color=list(self.colors.values()))
        axes[0].set_ylabel('ROUGE-L Score', fontsize=12)
        axes[0].set_title('(a) ROUGE-L Performance', fontsize=13, fontweight='bold')
        axes[0].set_ylim([0.3, 0.6])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # BERTScore
        bars2 = axes[1].bar(methods, bertscore, color=list(self.colors.values()))
        axes[1].set_ylabel('BERTScore', fontsize=12)
        axes[1].set_title('(b) Semantic Similarity', fontsize=13, fontweight='bold')
        axes[1].set_ylim([0.5, 0.8])
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Factual Consistency
        bars3 = axes[2].bar(methods, factual, color=list(self.colors.values()))
        axes[2].set_ylabel('Factual Consistency (%)', fontsize=12)
        axes[2].set_title('(c) Factual Accuracy', fontsize=13, fontweight='bold')
        axes[2].set_ylim([70, 100])
        axes[2].grid(axis='y', alpha=0.3)
        
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Table 3 visualization to {save_path}")
        
        plt.show()
    
    def plot_table4_explainability(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot Table 4: Explainability Metrics

        SSI, CPS, TCC, Explanation Consistency.

        Table 4 reports a two-way comparison only: the best-performing baseline
        per metric versus the proposed system.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        methods = ['Baseline', 'Ours']

        # Table 4. Baselines are per-metric best: BART+KG for SSI and CPS;
        # BART-large for explanation consistency and TCC.
        ssi = [0.61, 0.74]
        cps = [0.34, 0.51]
        tcc = [0.38, 0.54]
        consistency = [0.43, 0.59]
        
        # SSI
        axes[0, 0].bar(methods, ssi, color=list(self.colors.values()))
        axes[0, 0].set_ylabel('SSI Score', fontsize=12)
        axes[0, 0].set_title('(a) Stakeholder Satisfaction (SSI)', 
                            fontsize=13, fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # CPS
        axes[0, 1].bar(methods, cps, color=list(self.colors.values()))
        axes[0, 1].set_ylabel('CPS Score', fontsize=12)
        axes[0, 1].set_title('(b) Causal Preservation (CPS)',
                            fontsize=13, fontweight='bold')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # TCC
        axes[1, 0].bar(methods, tcc, color=list(self.colors.values()))
        axes[1, 0].set_ylabel('TCC Score', fontsize=12)
        axes[1, 0].set_title('(c) Temporal Consistency (TCC)',
                            fontsize=13, fontweight='bold')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Explanation Consistency
        axes[1, 1].bar(methods, consistency, color=list(self.colors.values()))
        axes[1, 1].set_ylabel('Consistency Score', fontsize=12)
        axes[1, 1].set_title('(d) Explanation Consistency',
                            fontsize=13, fontweight='bold')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Table 4 visualization to {save_path}")
        
        plt.show()
    
    def plot_error_analysis(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot Figure 6: Error analysis over N = 500 manually analysed summaries.

        Reference: Table 14 and Figure 6 (Section 4.4.1).

        The two panels use DIFFERENT denominators, by construction:

          (a) per-summary severity  - mutually exclusive, sums to exactly 100%
          (b) per-type frequency    - NON-exclusive (one summary may exhibit
                                      several error types), so these sum to more
                                      than the 56.5% total error rate

        43.5% is the error-free rate.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # --- Panel (a): per-summary severity. Mutually exclusive, sums to 100%. ---
        severity_labels = ['Error-Free\n43.5%',
                           'Lower-Severity\nOnly  34.7%',
                           'High-Severity\n21.8%']
        severity_counts = [43.5, 34.7, 21.8]
        severity_colors = ['#2ecc71', '#f1c40f', '#e74c3c']

        ax1.pie(
            severity_counts,
            labels=severity_labels,
            autopct='%1.1f%%',
            colors=severity_colors,
            startangle=90
        )
        ax1.set_title('(a) Per-summary error severity\n(N=500 summaries; sums to 100%)',
                      fontsize=12, fontweight='bold')

        # --- Panel (b): per-type frequency (Table 14). Non-exclusive. ---
        # High severity: factual inconsistency, numerical errors, hallucination.
        error_types = [
            'Factual Inconsistency', 'Temporal Ordering', 'Causal Attribution',
            'Numerical Errors', 'Entity Confusion', 'Incomplete Context',
            'Stakeholder Mismatch', 'Technical Terminology', 'Hallucination', 'Other'
        ]
        error_freqs = [14.2, 9.1, 7.8, 6.3, 5.4, 5.1, 4.7, 3.9, 2.3, 2.2]
        severity_of = ['high', 'medium', 'medium', 'high', 'medium',
                       'medium', 'low', 'low', 'high', 'variable']
        palette = {'high': '#e74c3c', 'medium': '#f1c40f',
                   'low': '#2ecc71', 'variable': '#95a5a6'}
        bar_colors = [palette[s] for s in severity_of]

        y = np.arange(len(error_types))
        bars = ax2.barh(y, error_freqs, color=bar_colors)
        ax2.set_yticks(y)
        ax2.set_yticklabels(error_types, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Frequency (% of summaries)', fontsize=11)
        ax2.set_title('(b) Per-type error frequency\n(non-exclusive; sums to >56.5%)',
                      fontsize=12, fontweight='bold')
        ax2.set_xlim([0, 16])
        ax2.grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, error_freqs):
            ax2.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                     f'{val}%', va='center', fontsize=9)

        ax2.legend(handles=[mpatches.Patch(color=palette[k], label=k.capitalize())
                            for k in ['high', 'medium', 'low', 'variable']],
                   loc='lower right', fontsize=8, title='Severity')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved error analysis to {save_path}")
        
        plt.show()
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        text_tokens: List[str],
        kg_tokens: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot attention heatmap between text and KG
        
        Args:
            attention_weights: (text_len, kg_len) attention matrix
            text_tokens: Text token labels
            kg_tokens: KG token labels
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(kg_tokens)))
        ax.set_yticks(np.arange(len(text_tokens)))
        ax.set_xticklabels(kg_tokens, rotation=45, ha='right')
        ax.set_yticklabels(text_tokens)
        
        ax.set_xlabel('Knowledge Graph Nodes', fontsize=12)
        ax.set_ylabel('Text Tokens', fontsize=12)
        ax.set_title('Cross-Modal Attention Weights', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention heatmap to {save_path}")
        
        plt.show()
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        save_path: Optional[str] = None
    ):
        """
        Plot training and validation loss curves
        
        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'o-', label='Training Loss',
               color='#3498db', linewidth=2)
        ax.plot(epochs, val_losses, 's-', label='Validation Loss',
               color='#e74c3c', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training curves to {save_path}")
        
        plt.show()
    
    def plot_kg_statistics(
        self,
        node_degrees: List[int],
        save_path: Optional[str] = None
    ):
        """
        Plot knowledge graph degree distribution
        
        Args:
            node_degrees: List of node degrees
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(node_degrees, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Node Degree', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('(a) Degree Distribution', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Log-log plot (power law check)
        unique, counts = np.unique(node_degrees, return_counts=True)
        ax2.loglog(unique, counts, 'o', color='#e74c3c', markersize=8, alpha=0.7)
        ax2.set_xlabel('Node Degree (log)', fontsize=12)
        ax2.set_ylabel('Frequency (log)', fontsize=12)
        ax2.set_title('(b) Log-Log Distribution', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved KG statistics to {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    print("Visualization Utilities")
    print("=" * 70)
    
    viz = PaperVisualizations()
    
    # Generate Table 3 visualization
    print("\nGenerating Table 3: Performance Comparison...")
    viz.plot_table3_performance(save_path='table3_performance.png')
    
    # Generate Table 4 visualization
    print("\nGenerating Table 4: Explainability Metrics...")
    viz.plot_table4_explainability(save_path='table4_explainability.png')
    
    # Generate Error Analysis
    print("\nGenerating Error Analysis...")
    viz.plot_error_analysis(save_path='error_analysis.png')
    
    # Generate sample attention heatmap
    print("\nGenerating Attention Heatmap...")
    
    # Sample data
    text_tokens = ['Apple', 'reported', 'earnings', 'of', '$89B']
    kg_tokens = ['Apple_Inc', 'Revenue', 'Q4_2023', 'Financial']
    attention = np.random.rand(len(text_tokens), len(kg_tokens))
    attention = attention / attention.sum(axis=1, keepdims=True)  # Normalize
    
    viz.plot_attention_heatmap(
        attention, text_tokens, kg_tokens,
        save_path='attention_heatmap.png'
    )
    
    # Generate training curves
    print("\nGenerating Training Curves...")
    
    train_losses = [2.5, 1.8, 1.4, 1.2, 1.0, 0.9, 0.85, 0.82, 0.80, 0.78]
    val_losses = [2.6, 1.9, 1.5, 1.25, 1.1, 1.0, 0.95, 0.93, 0.92, 0.91]
    
    viz.plot_training_curves(
        train_losses, val_losses,
        save_path='training_curves.png'
    )
    
    # Generate KG statistics
    print("\nGenerating KG Statistics...")
    
    # Sample power-law-like distribution
    node_degrees = np.random.pareto(2, 1000).astype(int) + 1
    
    viz.plot_kg_statistics(node_degrees, save_path='kg_statistics.png')
    
    print(f"\n{'=' * 70}")
    print("All visualizations generated!")
    print(f"{'=' * 70}")
    print("\nGenerated files:")
    print("  - table_ii_performance.png")
    print("  - table4_explainability.png")
    print("  - error_analysis.png")
    print("  - attention_heatmap.png")
    print("  - training_curves.png")
    print("  - kg_statistics.png")
