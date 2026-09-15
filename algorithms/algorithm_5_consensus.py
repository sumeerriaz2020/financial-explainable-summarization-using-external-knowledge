"""
Algorithm 5: INTERPRETABLE-CONSENSUS - Ensemble Explanation Selection
=======================================================================

Implementation of Algorithm 5 from the paper (INTERPRETABLE-CONSENSUS Framework).

Provides transparent explanation selection through interpretable ensemble methods,
integrating multiple explanation generators with validation and audit trails.

Authors: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
Reference: Section 3.4.4, Algorithm 5
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExplanationScore:
    """Scores for different quality dimensions"""
    accuracy: float = 0.0
    linguistic_quality: float = 0.0
    stakeholder_fit: float = 0.0
    compliance: float = 0.0


@dataclass
class ExplanationCandidate:
    """Candidate explanation with scores and metadata"""
    content: str
    method: str  # template, retrieval, neural, hybrid
    scores: ExplanationScore
    total_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AuditTrail:
    """Complete audit trail for explanation selection"""
    timestamp: str
    candidates: List[ExplanationCandidate]
    selected_explanation: str
    selection_method: str
    scores_log: Dict[str, float]
    conflicts_detected: List[str]
    conflict_resolution: List[str]
    uncertainty_metrics: Dict[str, float]
    

class ConsensusFramework:
    """
    INTERPRETABLE-CONSENSUS Framework for ensemble explanation selection.
    
    Implements Algorithm 5 for transparent, validated explanation selection
    using multiple generators with interpretable consensus mechanisms.
    """
    
    def __init__(
        self,
        score_weights: Optional[Dict[str, float]] = None,
        confidence_level: float = 0.95
    ):
        """
        Initialize INTERPRETABLE-CONSENSUS framework
        
        Args:
            score_weights: Weights for scoring dimensions (w_i)
            confidence_level: Confidence level for uncertainty quantification
        """
        # Default weights (can be learned)
        self.score_weights = score_weights or {
            'accuracy': 0.30,
            'linguistic_quality': 0.25,
            'stakeholder_fit': 0.25,
            'compliance': 0.20
        }
        
        self.confidence_level = confidence_level
        
        # Explanation generators
        self.generators = {
            'template': self._template_generator,
            'retrieval': self._retrieval_generator,
            'neural': self._neural_generator,
            'hybrid': self._hybrid_generator
        }
        
        # Expert rules for conflict resolution
        self.expert_rules = self._initialize_expert_rules()
        
        logger.info("INTERPRETABLE-CONSENSUS Framework initialized")
    
    def select_consensus_explanation(
        self,
        summary: str,
        document: Dict,
        knowledge_graph: any,
        stakeholder: any
    ) -> Tuple[str, AuditTrail]:
        """
        Select consensus explanation through ensemble validation.
        
        Implements Algorithm 5 from paper (Section 3.4.4).
        
        Args:
            summary: Generated summary
            document: Source document
            knowledge_graph: Extended KG
            stakeholder: Stakeholder profile
            
        Returns:
            Tuple of (consensus_explanation, audit_trail)
        """
        logger.info("Starting consensus explanation selection...")
        
        # Line 1: Initialize A ← ∅ (audit trail)
        audit_trail = AuditTrail(
            timestamp=datetime.now().isoformat(),
            candidates=[],
            selected_explanation="",
            selection_method="weighted_consensus",
            scores_log={},
            conflicts_detected=[],
            conflict_resolution=[],
            uncertainty_metrics={}
        )
        
        # Line 2: Generate candidates from all methods
        candidates = []
        
        for method_name, generator_func in self.generators.items():
            logger.info(f"Generating explanation with {method_name} method...")
            
            content = generator_func(summary, document, knowledge_graph)
            
            candidate = ExplanationCandidate(
                content=content,
                method=method_name,
                scores=ExplanationScore(),
                metadata={'generated_at': datetime.now().isoformat()}
            )
            
            candidates.append(candidate)
        
        # Log to audit trail (Line 2)
        audit_trail.candidates = candidates
        
        # Lines 3-6: Score each explanation
        logger.info("Scoring explanation candidates...")
        
        for candidate in candidates:
            # Line 4: Compute scores for each dimension
            candidate.scores = self._compute_scores(
                candidate.content, summary, document, stakeholder
            )
            
            # s_total(E) = Σ(w_i × s_i(E))
            candidate.total_score = (
                self.score_weights['accuracy'] * candidate.scores.accuracy +
                self.score_weights['linguistic_quality'] * candidate.scores.linguistic_quality +
                self.score_weights['stakeholder_fit'] * candidate.scores.stakeholder_fit +
                self.score_weights['compliance'] * candidate.scores.compliance
            )
            
            # Log scores
            audit_trail.scores_log[candidate.method] = candidate.total_score
        
        # Line 6: Log scores to A
        logger.info(f"Scores computed: {audit_trail.scores_log}")
        
        # Line 7: Resolve conflicts (expert rules > confidence > majority)
        logger.info("Resolving conflicts...")
        
        conflicts, resolutions = self._resolve_conflicts(candidates)
        audit_trail.conflicts_detected = conflicts
        audit_trail.conflict_resolution = resolutions
        
        # Line 8: E_consensus ← arg max E∈E s_total(E)
        consensus_explanation = max(candidates, key=lambda c: c.total_score)
        
        # Line 9: Log rationale to A
        audit_trail.selected_explanation = consensus_explanation.content
        audit_trail.selection_method = f"Weighted consensus ({consensus_explanation.method})"
        
        logger.info(f"Selected method: {consensus_explanation.method} "
                   f"(score: {consensus_explanation.total_score:.3f})")
        
        # Lines 10-11: Compute uncertainty and confidence intervals
        logger.info("Computing uncertainty metrics...")
        
        variance = np.var([c.total_score for c in candidates])
        std_dev = np.sqrt(variance)
        
        # Confidence interval (Line 11)
        z_score = 1.96  # 95% confidence
        margin = z_score * std_dev
        confidence_interval = (
            consensus_explanation.total_score - margin,
            consensus_explanation.total_score + margin
        )
        
        # Line 12: Attach uncertainty to E_consensus
        consensus_explanation.confidence_interval = confidence_interval
        
        # Line 13: Log metrics to A
        audit_trail.uncertainty_metrics = {
            'variance': float(variance),
            'std_dev': float(std_dev),
            'confidence_interval': confidence_interval,
            'score_range': (
                min(c.total_score for c in candidates),
                max(c.total_score for c in candidates)
            )
        }
        
        logger.info(f"Uncertainty: σ²={variance:.4f}, "
                   f"CI={confidence_interval}")
        
        # Line 14: return Consensus explanation E_consensus, audit trail A
        return consensus_explanation.content, audit_trail
    
    def _compute_scores(
        self,
        explanation: str,
        summary: str,
        document: Dict,
        stakeholder: any
    ) -> ExplanationScore:
        """
        Compute quality scores for explanation.
        
        Line 4 of Algorithm 5.
        """
        scores = ExplanationScore()
        
        # Accuracy: how well explanation aligns with summary/document
        scores.accuracy = self._compute_accuracy_score(
            explanation, summary, document
        )
        
        # Linguistic quality: readability, coherence, grammar
        scores.linguistic_quality = self._compute_linguistic_quality(
            explanation
        )
        
        # Stakeholder fit: appropriateness for target user
        scores.stakeholder_fit = self._compute_stakeholder_fit(
            explanation, stakeholder
        )
        
        # Compliance: regulatory alignment
        scores.compliance = self._compute_compliance_score(
            explanation
        )
        
        return scores
    
    def _compute_accuracy_score(
        self,
        explanation: str,
        summary: str,
        document: Dict
    ) -> float:
        """Compute factual accuracy of explanation"""
        # Simple heuristic: check if key terms from summary appear
        summary_terms = set(summary.lower().split())
        explanation_terms = set(explanation.lower().split())
        
        overlap = len(summary_terms & explanation_terms)
        max_possible = len(summary_terms)
        
        accuracy = overlap / max_possible if max_possible > 0 else 0.0
        
        # Normalize to [0, 1]
        return min(accuracy, 1.0)
    
    def _compute_linguistic_quality(self, explanation: str) -> float:
        """Compute linguistic quality (readability, coherence)"""
        # Simple metrics
        words = explanation.split()
        sentences = explanation.split('.')
        
        # Average sentence length
        avg_sent_length = len(words) / len(sentences) if sentences else 0
        
        # Penalize very short or very long sentences
        quality = 1.0
        if avg_sent_length < 5 or avg_sent_length > 30:
            quality *= 0.8
        
        # Check for proper punctuation
        if not explanation.strip().endswith('.'):
            quality *= 0.9
        
        return quality
    
    def _compute_stakeholder_fit(
        self,
        explanation: str,
        stakeholder: any
    ) -> float:
        """Compute fit for stakeholder needs"""
        if not stakeholder:
            return 0.7  # Default
        
        # Check if explanation includes stakeholder-relevant terms
        role = getattr(stakeholder, 'role', 'unknown')
        
        relevant_terms = {
            'analyst': ['quantitative', 'metric', 'performance', 'comparison'],
            'compliance': ['regulatory', 'requirement', 'audit', 'compliance'],
            'executive': ['strategic', 'impact', 'recommendation', 'action'],
            'investment_manager': ['return', 'risk', 'performance', 'portfolio']
        }
        
        terms = relevant_terms.get(role, [])
        explanation_lower = explanation.lower()
        
        relevance = sum(1 for term in terms if term in explanation_lower)
        fit_score = min(relevance / len(terms), 1.0) if terms else 0.7
        
        return fit_score
    
    def _compute_compliance_score(self, explanation: str) -> float:
        """Compute regulatory compliance alignment"""
        # Check for transparency indicators
        compliance_indicators = [
            'source', 'confidence', 'verified', 'validated',
            'knowledge', 'based on', 'according to'
        ]
        
        explanation_lower = explanation.lower()
        indicators_present = sum(
            1 for indicator in compliance_indicators
            if indicator in explanation_lower
        )
        
        compliance = indicators_present / len(compliance_indicators)
        return min(compliance + 0.5, 1.0)  # Baseline 0.5
    
    def _resolve_conflicts(
        self,
        candidates: List[ExplanationCandidate]
    ) -> Tuple[List[str], List[str]]:
        """
        Resolve conflicts between explanation generators.
        
        Line 7 of Algorithm 5: expert rules > confidence > majority
        """
        conflicts = []
        resolutions = []
        
        # Check for score disagreements
        scores = [c.total_score for c in candidates]
        score_std = np.std(scores)
        
        if score_std > 0.2:  # High disagreement
            conflict = f"High score variance detected (σ={score_std:.3f})"
            conflicts.append(conflict)
            
            # Resolution: Apply expert rules
            resolution = self._apply_expert_rules(candidates)
            resolutions.append(f"Applied expert rule: {resolution}")
        
        # Check for contradictory content
        for i, cand1 in enumerate(candidates):
            for cand2 in candidates[i+1:]:
                if self._are_contradictory(cand1.content, cand2.content):
                    conflict = f"Contradiction between {cand1.method} and {cand2.method}"
                    conflicts.append(conflict)
                    
                    # Resolution: Use higher confidence method
                    if cand1.total_score > cand2.total_score:
                        resolution = f"Prefer {cand1.method} (higher score)"
                    else:
                        resolution = f"Prefer {cand2.method} (higher score)"
                    resolutions.append(resolution)
        
        return conflicts, resolutions
    
    def _apply_expert_rules(
        self,
        candidates: List[ExplanationCandidate]
    ) -> str:
        """Apply expert rules for conflict resolution"""
        # Rule 1: Prefer hybrid over other methods
        hybrid_candidates = [c for c in candidates if c.method == 'hybrid']
        if hybrid_candidates:
            return "Prefer hybrid method (combines neural + symbolic)"
        
        # Rule 2: Prefer higher compliance scores
        max_compliance = max(c.scores.compliance for c in candidates)
        if max_compliance >= 0.8:
            return "Prefer high-compliance explanation"
        
        # Rule 3: Default to highest total score
        return "Use highest total score"
    
    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """Check if two explanations contradict each other"""
        # Simple heuristic: check for opposite sentiments
        negative_words = ['not', 'no', 'never', 'failed', 'declined', 'decreased']
        positive_words = ['successful', 'increased', 'improved', 'growth']
        
        text1_negative = sum(1 for word in negative_words if word in text1.lower())
        text1_positive = sum(1 for word in positive_words if word in text1.lower())
        
        text2_negative = sum(1 for word in negative_words if word in text2.lower())
        text2_positive = sum(1 for word in positive_words if word in text2.lower())
        
        # Contradiction if sentiments strongly oppose
        if (text1_positive > text1_negative + 2) and (text2_negative > text2_positive + 2):
            return True
        if (text1_negative > text1_positive + 2) and (text2_positive > text2_negative + 2):
            return True
        
        return False
    
    def _initialize_expert_rules(self) -> List[Dict]:
        """Initialize expert rules for conflict resolution"""
        rules = [
            {
                'name': 'prefer_hybrid',
                'condition': lambda c: c.method == 'hybrid',
                'priority': 1
            },
            {
                'name': 'prefer_high_compliance',
                'condition': lambda c: c.scores.compliance >= 0.8,
                'priority': 2
            },
            {
                'name': 'prefer_stakeholder_fit',
                'condition': lambda c: c.scores.stakeholder_fit >= 0.85,
                'priority': 3
            }
        ]
        return rules
    
    # Generator methods (simplified - similar to MESA)
    
    def _template_generator(self, summary, document, kg) -> str:
        return f"Template explanation for summary: {summary[:50]}..."
    
    def _retrieval_generator(self, summary, document, kg) -> str:
        return f"Retrieval-based explanation using similar documents..."
    
    def _neural_generator(self, summary, document, kg) -> str:
        return f"Neural explanation generated by transformer model..."
    
    def _hybrid_generator(self, summary, document, kg) -> str:
        return f"Hybrid explanation combining neural and symbolic reasoning..."


# Example usage
if __name__ == "__main__":
    print("INTERPRETABLE-CONSENSUS Framework")
    print("="*70)
    
    # Initialize framework
    consensus = ConsensusFramework(
        score_weights={
            'accuracy': 0.30,
            'linguistic_quality': 0.25,
            'stakeholder_fit': 0.25,
            'compliance': 0.20
        }
    )
    
    # Example inputs
    summary = "Apple reported Q4 revenue of $89.5B, up 8% YoY."
    document = {'text': 'Full earnings report...', 'id': 'doc_001'}
    
    # Mock stakeholder
    class MockStakeholder:
        role = 'analyst'
    
    stakeholder = MockStakeholder()
    
    # Generate consensus explanation
    print("\nGenerating consensus explanation...")
    explanation, audit = consensus.select_consensus_explanation(
        summary, document, None, stakeholder
    )
    
    # Display results
    print(f"\n{'='*70}")
    print("CONSENSUS EXPLANATION:")
    print(f"{'='*70}")
    print(explanation)
    
    print(f"\n{'='*70}")
    print("AUDIT TRAIL:")
    print(f"{'='*70}")
    print(f"Timestamp: {audit.timestamp}")
    print(f"Selection Method: {audit.selection_method}")
    print(f"\nCandidate Scores:")
    for method, score in audit.scores_log.items():
        print(f"  {method}: {score:.3f}")
    
    print(f"\nConflicts Detected: {len(audit.conflicts_detected)}")
    for conflict in audit.conflicts_detected:
        print(f"  - {conflict}")
    
    print(f"\nConflict Resolutions: {len(audit.conflict_resolution)}")
    for resolution in audit.conflict_resolution:
        print(f"  - {resolution}")
    
    print(f"\nUncertainty Metrics:")
    for metric, value in audit.uncertainty_metrics.items():
        print(f"  {metric}: {value}")
