"""
TEMPORAL-EXPLAIN - Temporal Consistency Maintenance
===================================================

Ensures explanation consistency across changing financial contexts and
market conditions with regime-aware adaptation.

Release scope. The manuscript (Section 3.4.5) specifies market-regime detection
by a three-state Hidden Markov Model (bull, bear, sideways) estimated on rolling
90-day windows of VIX and sector-rotation metrics (81.3% held-out accuracy), and
concept-drift detection by the Page-Hinkley test on rolling TCC (delta = 0.15),
triggering re-calibration on the most recent 500 document-summary pairs. Both are
specified in the paper, not included in this release. Regime detection below is a
keyword heuristic and drift detection a fixed TCC threshold, as stand-ins.

Reference: Section 3.4.5
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Represents market condition regime"""
    regime_type: str  # bull, bear, sideways (the three HMM states, Section 3.4.5)
    start_date: str
    volatility: float
    trend: str
    indicators: Dict[str, float]
    crisis: bool = False  # crisis conditions: risk-related causal chains weigh 2x


@dataclass
class ExplanationHistory:
    """Historical explanation with context"""
    timestamp: str
    explanation: str
    summary: str
    regime: MarketRegime
    consistency_score: float


class TemporalExplainer:
    """Temporal consistency maintenance across market regimes"""
    
    def __init__(self):
        self.explanation_history = []
        self.regime_criteria = self._init_regime_criteria()
        # Stand-in for Page-Hinkley drift detection (not included in this release)
        self.consistency_threshold = 0.7
        # Crisis conditions: risk-related causal chains carry twice the weight in
        # MESA stakeholder scoring (Section 3.4.5)
        self.crisis_risk_chain_weight = 2.0
        logger.info("TEMPORAL-EXPLAIN initialized")
    
    def generate_with_temporal_consistency(
        self,
        summary: str,
        document: Dict,
        current_regime: MarketRegime,
        historical_context: List[Dict] = None
    ) -> Tuple[str, float]:
        """Generate explanation with temporal consistency"""
        
        # Detect market regime
        if current_regime is None:
            current_regime = self._detect_regime(document)
        
        # Get regime-specific criteria
        criteria = self.regime_criteria.get(
            current_regime.regime_type,
            self.regime_criteria['sideways']
        )
        
        # Generate explanation adapted to regime
        explanation = self._generate_regime_adapted(
            summary, document, current_regime, criteria
        )
        
        # Check consistency with history
        consistency = self._compute_temporal_consistency(
            explanation, current_regime
        )
        
        # Detect concept drift
        drift_detected = self._detect_concept_drift(consistency)
        
        if drift_detected:
            logger.warning(f"Concept drift detected: TCC={consistency:.3f}")
            # Recalibrate if needed
            explanation = self._recalibrate_explanation(
                explanation, current_regime
            )
            consistency = self._compute_temporal_consistency(
                explanation, current_regime
            )
        
        # Store in history
        self.explanation_history.append(
            ExplanationHistory(
                timestamp=datetime.now().isoformat(),
                explanation=explanation,
                summary=summary,
                regime=current_regime,
                consistency_score=consistency
            )
        )
        
        return explanation, consistency
    
    def compute_tcc(self, window: int = 10) -> float:
        """Compute Temporal Consistency Coefficient (Equation 9)"""
        if len(self.explanation_history) < 2:
            return 1.0
        
        recent = self.explanation_history[-window:]
        
        # TCC = (1/N) Σ cos_sim(E_t, E_{t-1})
        similarities = []
        for i in range(1, len(recent)):
            sim = self._cosine_similarity(
                recent[i].explanation,
                recent[i-1].explanation
            )
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _detect_regime(self, document: Dict) -> MarketRegime:
        """Detect market regime from document"""
        text = document.get('text', '').lower()
        
        # Simple keyword-based detection (stand-in for the HMM of Section 3.4.5)
        crisis = any(w in text for w in ['crisis', 'crash', 'panic', 'recession'])
        if crisis:
            regime_type = 'bear'
            volatility = 0.8
            trend = 'declining'
        elif any(w in text for w in ['growth', 'rally', 'expansion', 'boom']):
            regime_type = 'bull'
            volatility = 0.4
            trend = 'rising'
        elif any(w in text for w in ['downturn', 'correction', 'decline']):
            regime_type = 'bear'
            volatility = 0.6
            trend = 'declining'
        else:
            regime_type = 'sideways'
            volatility = 0.3
            trend = 'stable'
        
        return MarketRegime(
            regime_type=regime_type,
            start_date=datetime.now().isoformat(),
            volatility=volatility,
            trend=trend,
            indicators={'vix': volatility * 100},
            crisis=crisis
        )
    
    def _init_regime_criteria(self) -> Dict[str, Dict]:
        """Initialize regime-specific criteria"""
        return {
            'sideways': {
                'emphasis': ['performance', 'trends', 'outlook'],
                'detail_level': 'moderate',
                'tone': 'balanced'
            },
            'bull': {
                'emphasis': ['growth', 'opportunities', 'momentum'],
                'detail_level': 'moderate',
                'tone': 'optimistic'
            },
            'bear': {
                'emphasis': ['risks', 'caution', 'defensive'],
                'detail_level': 'high',
                'tone': 'cautious'
            },
            # Applied on top of the regime when crisis conditions are detected
            'crisis': {
                'emphasis': ['stability', 'risk management', 'liquidity'],
                'detail_level': 'very_high',
                'tone': 'urgent'
            }
        }
    
    def _generate_regime_adapted(
        self,
        summary: str,
        document: Dict,
        regime: MarketRegime,
        criteria: Dict
    ) -> str:
        """Generate explanation adapted to market regime"""
        
        base = f"EXPLANATION (Market Regime: {regime.regime_type.upper()})\n\n"
        
        # Regime-specific framing; crisis conditions override the regime criteria
        if regime.crisis:
            criteria = self.regime_criteria['crisis']
            base += "Given current market volatility and uncertainty:\n"
        elif regime.regime_type == 'bull':
            base += "In the context of favorable market conditions:\n"
        elif regime.regime_type == 'bear':
            base += "Considering current market headwinds:\n"
        else:
            base += "Under sideways (range-bound) market conditions:\n"
        
        # Add summary
        base += f"\n{summary}\n\n"
        
        # Add regime-specific emphasis
        emphasis = criteria['emphasis']
        base += f"Key considerations ({', '.join(emphasis)}):\n"
        base += f"- Analysis focuses on {emphasis[0]} indicators\n"
        base += f"- Tone: {criteria['tone']}, Detail: {criteria['detail_level']}\n"
        
        return base
    
    def _compute_temporal_consistency(
        self,
        explanation: str,
        current_regime: MarketRegime
    ) -> float:
        """Compute consistency with historical explanations"""
        
        if len(self.explanation_history) < 2:
            return 1.0
        
        # Find similar regime periods
        similar_regimes = [
            h for h in self.explanation_history[-20:]
            if h.regime.regime_type == current_regime.regime_type
        ]
        
        if not similar_regimes:
            return 0.8  # Default for new regime
        
        # Compute similarity with similar regimes
        similarities = [
            self._cosine_similarity(explanation, h.explanation)
            for h in similar_regimes
        ]
        
        return np.mean(similarities)
    
    def _detect_concept_drift(self, consistency: float) -> bool:
        """Detect if concept drift has occurred"""
        return consistency < self.consistency_threshold
    
    def _recalibrate_explanation(
        self,
        explanation: str,
        regime: MarketRegime
    ) -> str:
        """Recalibrate explanation after drift detection"""
        
        # Add recalibration notice
        recalibrated = explanation + "\n\n[Recalibrated for regime change]\n"
        
        return recalibrated
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between texts"""
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        magnitude = np.sqrt(len(words1) * len(words2))
        
        return intersection / magnitude if magnitude > 0 else 0.0
    
    def get_consistency_report(self) -> Dict:
        """Generate consistency report"""
        
        tcc = self.compute_tcc()
        
        # Regime distribution
        regime_counts = {}
        for h in self.explanation_history:
            regime_counts[h.regime.regime_type] = \
                regime_counts.get(h.regime.regime_type, 0) + 1
        
        # Average consistency by regime
        regime_consistency = {}
        for regime_type in regime_counts:
            scores = [
                h.consistency_score for h in self.explanation_history
                if h.regime.regime_type == regime_type
            ]
            regime_consistency[regime_type] = np.mean(scores) if scores else 0.0
        
        return {
            'temporal_consistency_coefficient': tcc,
            'total_explanations': len(self.explanation_history),
            'regime_distribution': regime_counts,
            'consistency_by_regime': regime_consistency,
            'drift_events': sum(1 for h in self.explanation_history 
                              if h.consistency_score < self.consistency_threshold)
        }
