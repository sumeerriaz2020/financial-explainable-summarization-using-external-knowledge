"""
MESA Framework - Multi-Stakeholder Explainable Summarization Assessment
========================================================================

Generates stakeholder-aware explanations optimized for different user types
through dynamic preference learning and multi-objective optimization.

The REINFORCE-style online weight update of Section 3.4.1 (moving-average baseline,
eta = 0.01, gradient clipping at 0.5) is specified in the paper, not included in
this release; update_with_feedback applies a simplified reward-scaled step.

Reference: Algorithm 3, Section 3.4.1
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class StakeholderProfile:
    """Stakeholder profile with learned preferences"""
    role: str
    expertise_level: str
    information_needs: List[str]
    quality_weights: Dict[str, float] = field(default_factory=dict)
    explainability_weights: Dict[str, float] = field(default_factory=dict)
    preference_history: List[Tuple[str, float]] = field(default_factory=list)
    alpha_lr: float = 0.01
    beta_lr: float = 0.01


class MESAExplainer:
    """Multi-Stakeholder Explanation Generation with RL"""
    
    def __init__(self):
        self.profiles = self._init_profiles()
        self.generators = {
            'template': self._template_gen,
            'retrieval': self._retrieval_gen,
            'neural': self._neural_gen,
            'hybrid': self._hybrid_gen
        }
        logger.info("MESA Explainer initialized")
    
    def generate_explanation(
        self,
        summary: str,
        document: Dict,
        kg: any,
        stakeholder: StakeholderProfile
    ) -> str:
        """Generate stakeholder-specific explanation (Algorithm 3)"""
        
        # Generate candidates
        candidates = [
            {'content': gen(summary, document, kg), 'method': name}
            for name, gen in self.generators.items()
        ]
        
        # Score each candidate (Equation 5)
        for cand in candidates:
            quality = sum(
                stakeholder.quality_weights.get(m, 0.33) * self._score(cand, m)
                for m in ['rouge_l', 'bertscore', 'factual']
            )
            expl = sum(
                stakeholder.explainability_weights.get(m, 0.33) * self._score(cand, m)
                for m in ['comprehension', 'trust', 'actionability']
            )
            cand['score'] = quality + expl
        
        # Select best
        best = max(candidates, key=lambda c: c['score'])
        return best['content']
    
    def update_with_feedback(
        self,
        profile: StakeholderProfile,
        explanation: str,
        feedback: float
    ):
        """Update weights based on user feedback (RL)"""
        reward = feedback - 0.5
        
        for metric in profile.quality_weights:
            profile.quality_weights[metric] += profile.alpha_lr * reward * 0.1
        
        for metric in profile.explainability_weights:
            profile.explainability_weights[metric] += profile.beta_lr * reward * 0.1
        
        self._normalize_weights(profile)
        profile.preference_history.append((explanation, feedback))
    
    def _init_profiles(self) -> Dict[str, StakeholderProfile]:
        """Initialize default stakeholder profiles"""
        return {
            'analyst': StakeholderProfile(
                role='financial_analyst',
                expertise_level='expert',
                information_needs=['quantitative', 'trends', 'comparison'],
                quality_weights={'rouge_l': 0.33, 'bertscore': 0.33, 'factual': 0.34},
                explainability_weights={'comprehension': 0.33, 'trust': 0.33, 'actionability': 0.34}
            ),
            'compliance': StakeholderProfile(
                role='compliance_officer',
                expertise_level='expert',
                information_needs=['regulatory', 'audit_trail', 'citations'],
                quality_weights={'rouge_l': 0.30, 'bertscore': 0.30, 'factual': 0.40},
                explainability_weights={'comprehension': 0.25, 'trust': 0.45, 'actionability': 0.30}
            ),
            'executive': StakeholderProfile(
                role='executive',
                expertise_level='intermediate',
                information_needs=['insights', 'strategy', 'recommendations'],
                quality_weights={'rouge_l': 0.35, 'bertscore': 0.35, 'factual': 0.30},
                explainability_weights={'comprehension': 0.40, 'trust': 0.25, 'actionability': 0.35}
            ),
            'investment_manager': StakeholderProfile(
                role='investment_manager',
                expertise_level='expert',
                information_needs=['performance_implications', 'risk_factors', 'market_positioning'],
                quality_weights={'rouge_l': 0.33, 'bertscore': 0.33, 'factual': 0.34},
                explainability_weights={'comprehension': 0.30, 'trust': 0.35, 'actionability': 0.35}
            )
        }
    
    def _score(self, cand: Dict, metric: str) -> float:
        """Compute metric score for candidate"""
        return np.random.uniform(0.6, 0.9)  # Placeholder
    
    def _normalize_weights(self, profile: StakeholderProfile):
        """Normalize weights to sum to 1"""
        total_q = sum(profile.quality_weights.values())
        if total_q > 0:
            for k in profile.quality_weights:
                profile.quality_weights[k] /= total_q
        
        total_e = sum(profile.explainability_weights.values())
        if total_e > 0:
            for k in profile.explainability_weights:
                profile.explainability_weights[k] /= total_e
    
    def _template_gen(self, summary, doc, kg) -> str:
        return f"Template explanation: {summary[:100]}..."
    
    def _retrieval_gen(self, summary, doc, kg) -> str:
        return f"Retrieval-based explanation with similar cases..."
    
    def _neural_gen(self, summary, doc, kg) -> str:
        return f"Neural explanation using transformer model..."
    
    def _hybrid_gen(self, summary, doc, kg) -> str:
        return f"Hybrid neural-symbolic explanation combining reasoning..."
