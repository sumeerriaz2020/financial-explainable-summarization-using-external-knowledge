"""
Evaluation Metrics
==================

Comprehensive metrics for evaluating summarization quality and explainability:
- ROUGE (L, 1, 2)
- BERTScore
- Factual Consistency
- SSI (Stakeholder Satisfaction Index)
- CPS (Causal Preservation Score)
- TCC (Temporal Consistency Coefficient)

Regulatory alignment is deliberately NOT scored here: the manuscript documents it
qualitatively as a design-level checklist (Table 16), not as a quantitative metric.

Reference: Section 3.7 (Evaluation Framework), Section 4; Tables 3, 4
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

from training_evaluation.cps import aggregate_cps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROUGEMetrics:
    """ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)"""
    
    def __init__(self):
        """Initialize ROUGE scorer"""
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            self.available = True
        except ImportError:
            logger.warning("rouge_score not available")
            self.available = False
    
    def compute(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            
        Returns:
            Dictionary of ROUGE scores
        """
        if not self.available:
            return self._compute_manual(predictions, references)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def _compute_manual(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Manual ROUGE-L implementation"""
        scores = []
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            # LCS
            lcs_length = self._lcs_length(pred_tokens, ref_tokens)
            
            # Precision and recall
            if len(pred_tokens) > 0:
                precision = lcs_length / len(pred_tokens)
            else:
                precision = 0.0
            
            if len(ref_tokens) > 0:
                recall = lcs_length / len(ref_tokens)
            else:
                recall = 0.0
            
            # F-measure
            if precision + recall > 0:
                f_measure = 2 * precision * recall / (precision + recall)
            else:
                f_measure = 0.0
            
            scores.append(f_measure)
        
        return {'rougeL': np.mean(scores)}
    
    def _lcs_length(self, seq1: List, seq2: List) -> int:
        """Longest Common Subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


class BERTScoreMetrics:
    """BERTScore for semantic similarity"""
    
    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli"):
        """Initialize BERTScore"""
        try:
            from bert_score import BERTScorer
            self.scorer = BERTScorer(
                model_type=model_type,
                lang="en",
                rescale_with_baseline=True
            )
            self.available = True
        except ImportError:
            logger.warning("bert_score not available")
            self.available = False
    
    def compute(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BERTScore
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            
        Returns:
            BERTScore (precision, recall, F1)
        """
        if not self.available:
            logger.warning("BERTScore not available, returning dummy scores")
            return {'bertscore': 0.0}
        
        P, R, F1 = self.scorer.score(predictions, references)
        
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item(),
            'bertscore': F1.mean().item()  # Main metric
        }


class FactualConsistencyMetrics:
    """Factual consistency evaluation"""
    
    def compute(
        self,
        predictions: List[str],
        sources: List[str],
        extracted_facts: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Compute factual consistency score
        
        Args:
            predictions: Generated summaries
            sources: Source documents
            extracted_facts: Pre-extracted facts (optional)
            
        Returns:
            Factual consistency score
        """
        scores = []
        
        for pred, source in zip(predictions, sources):
            # Extract facts from prediction
            pred_facts = self._extract_facts(pred)
            
            # Extract facts from source
            source_facts = self._extract_facts(source)
            
            # Compute overlap
            if len(pred_facts) > 0:
                consistent = sum(
                    1 for fact in pred_facts
                    if self._is_supported(fact, source_facts)
                )
                consistency = consistent / len(pred_facts)
            else:
                consistency = 1.0
            
            scores.append(consistency)
        
        return {
            'factual_consistency': np.mean(scores) * 100  # Percentage
        }
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract facts (simplified: sentences with numbers/entities)"""
        import re
        
        sentences = text.split('.')
        facts = []
        
        for sent in sentences:
            sent = sent.strip()
            # Check if contains numbers or capitalized entities
            if re.search(r'\d+|\$|%', sent) or re.search(r'\b[A-Z][a-z]+\b', sent):
                facts.append(sent.lower())
        
        return facts
    
    def _is_supported(self, fact: str, source_facts: List[str]) -> bool:
        """Check if fact is supported by source"""
        fact_words = set(fact.split())
        
        for source_fact in source_facts:
            source_words = set(source_fact.split())
            # Check overlap
            overlap = len(fact_words & source_words) / len(fact_words) if fact_words else 0
            if overlap > 0.5:
                return True
        
        return False


class SSIMetrics:
    """
    Stakeholder Satisfaction Index (SSI)
    
    Equation 12: SSI = sum_s w_s * (mu_s^quality + mu_s^explainability) / 2
    where w_s is the stakeholder weight and each rating is normalized to [0, 1]

    Reference: Section 3.7.2, Algorithm 3 (MESA), Table 4
    """

    def __init__(self):
        """Initialize SSI metrics"""
        # Stakeholder weights (Section 3.7.2): fixed a priori at 0.25 each,
        # reflecting equal institutional importance. Not learned, not estimated
        # from any data split, and held constant during training and evaluation.
        self.stakeholder_weights = {
            'analyst': 0.25,
            'compliance': 0.25,
            'executive': 0.25,
            'investment_manager': 0.25
        }
    
    def compute(
        self,
        explanations: Dict[str, List[str]],
        stakeholder_ratings: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Compute SSI score
        
        Args:
            explanations: Stakeholder-specific explanations
            stakeholder_ratings: Human ratings per stakeholder
            
        Returns:
            SSI scores
        """
        ssi_scores = []
        
        for i in range(len(next(iter(explanations.values())))):
            weighted_sum = 0.0
            
            for stakeholder, weight in self.stakeholder_weights.items():
                if stakeholder in stakeholder_ratings:
                    rating = stakeholder_ratings[stakeholder][i]
                    weighted_sum += weight * rating
            
            ssi_scores.append(weighted_sum)
        
        return {
            'ssi': np.mean(ssi_scores),
            'ssi_std': np.std(ssi_scores),
            'ssi_by_stakeholder': {
                stakeholder: np.mean(ratings)
                for stakeholder, ratings in stakeholder_ratings.items()
            }
        }


class CPSMetrics:
    """
    Causal Preservation Score (CPS)
    
    Equations 7-8: standard and weighted CPS, computed per document over the causal
    claims asserted in the source (C_original) and those present in the summary,
    then averaged over documents where CPS is defined. Documents whose source has
    no causal claims are excluded (Section 3.7.2). The computation lives in
    training_evaluation/cps.py.
    
    Reference: Algorithm 4 (CAUSAL-EXPLAIN), Section 3.7.2, Table 4
    """
    
    def compute(
        self,
        summary_claims: List[List],
        source_claims: List[List],
        importance_weights: Optional[List[Optional[List[float]]]] = None
    ) -> Dict[str, float]:
        """
        Compute CPS over a set of documents
        
        Args:
            summary_claims: Per document, the causal claims found in the summary,
                each a (cause, effect[, confidence]) tuple or CausalChain
            source_claims: Per document, the causal claims asserted in the source
            importance_weights: Optional per-document weights aligned with that
                document's source claims (Equation 8)
            
        Returns:
            cps, cps_std, n_documents, n_excluded; plus cps_weighted,
            cps_weighted_std and n_excluded_weighted when weights are given
        """
        return aggregate_cps(source_claims, summary_claims, importance_weights)


class TCCMetrics:
    """
    Temporal Consistency Coefficient (TCC)
    
    Equation 9: TCC = (1/N) sum_t cos_sim(E_t, E_{t-1})
    
    Reference: TEMPORAL-EXPLAIN framework, Table 4
    """
    
    def compute(
        self,
        explanation_sequence: List[str],
        embeddings: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Compute TCC
        
        Args:
            explanation_sequence: Temporal sequence of explanations
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            TCC score
        """
        if embeddings is None:
            embeddings = self._compute_embeddings(explanation_sequence)
        
        if len(embeddings) < 2:
            return {'tcc': 1.0}
        
        # Compute cosine similarities
        similarities = []
        
        for i in range(1, len(embeddings)):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i-1].unsqueeze(0),
                embeddings[i].unsqueeze(0)
            )
            similarities.append(sim.item())
        
        return {
            'tcc': np.mean(similarities),
            'tcc_std': np.std(similarities),
            'tcc_min': np.min(similarities),
            'tcc_max': np.max(similarities)
        }
    
    def _compute_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Compute embeddings for texts (simplified)"""
        # In production, use actual sentence embeddings
        embeddings = []
        
        for text in texts:
            # Simple word-based embedding
            words = text.lower().split()
            # Random embedding for demonstration
            emb = torch.randn(768)  # BERT-base dimension
            embeddings.append(emb)
        
        return embeddings


# Combined evaluator
class ComprehensiveEvaluator:
    """All metrics in one place"""
    
    def __init__(self):
        """Initialize all metrics"""
        self.rouge = ROUGEMetrics()
        self.bertscore = BERTScoreMetrics()
        self.factual = FactualConsistencyMetrics()
        self.ssi = SSIMetrics()
        self.cps = CPSMetrics()
        self.tcc = TCCMetrics()
        
        logger.info("Comprehensive Evaluator initialized")
    
    def evaluate_all(
        self,
        predictions: List[str],
        references: List[str],
        sources: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Compute all metrics"""
        results = {}
        
        # Summarization metrics
        results.update(self.rouge.compute(predictions, references))
        results.update(self.bertscore.compute(predictions, references))
        results.update(self.factual.compute(predictions, sources))
        
        # Explainability metrics (if data provided)
        if 'explanations' in kwargs and 'stakeholder_ratings' in kwargs:
            results.update(self.ssi.compute(
                kwargs['explanations'],
                kwargs['stakeholder_ratings']
            ))
        
        if 'predicted_chains' in kwargs and 'reference_chains' in kwargs:
            results.update(self.cps.compute(
                summary_claims=kwargs['predicted_chains'],
                source_claims=kwargs['reference_chains'],
                importance_weights=kwargs.get('importance_weights')
            ))
        
        if 'explanation_sequence' in kwargs:
            results.update(self.tcc.compute(kwargs['explanation_sequence']))
        
        return results


# Example usage
if __name__ == "__main__":
    print("Evaluation Metrics")
    print("=" * 70)
    
    print("\nAvailable Metrics:")
    print("  Summarization Quality:")
    print("    - ROUGE-1, ROUGE-2, ROUGE-L")
    print("    - BERTScore")
    print("    - Factual Consistency")
    
    print("\n  Explainability (Novel):")
    print("    - SSI: Stakeholder Satisfaction Index (Eq. 12)")
    print("    - CPS: Causal Preservation Score (Eq. 7-8)")
    print("    - TCC: Temporal Consistency Coefficient (Eq. 9)")
    
    print("\n" + "=" * 70)
    print("Metrics ready!")
    print("=" * 70)
