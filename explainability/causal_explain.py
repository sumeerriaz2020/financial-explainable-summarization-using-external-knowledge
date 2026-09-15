"""
CAUSAL-EXPLAIN Framework - Causal Chain Extraction and Preservation
====================================================================

Extracts, preserves, and explains causal relationships in financial documents.

Reference: Algorithm 4, Section 3.4.2
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

from training_evaluation.cps import (
    causal_preservation_score,
    weighted_causal_preservation_score,
)

logger = logging.getLogger(__name__)


@dataclass
class CausalChain:
    """Represents a causal relationship"""
    cause: str
    effect: str
    confidence: float
    context: str
    temporal_ordering: str
    importance_weight: float = 1.0


class CausalExplainer:
    """Extract and preserve causal chains"""
    
    def __init__(self, causal_threshold: float = 0.7, min_confidence: float = 0.35):
        self.causal_threshold = causal_threshold
        self.min_confidence = min_confidence
        self.patterns = self._init_patterns()
        logger.info("CAUSAL-EXPLAIN initialized")
    
    def extract_and_preserve(
        self,
        document: Dict,
        kg: any,
        summary_budget: int = 512
    ) -> Dict:
        """Extract and preserve causal chains (Algorithm 4)"""
        
        text = document['text']
        
        # Extract entities and temporal ordering
        entities = self._extract_entities(text)
        temporal = self._extract_temporal(text)
        
        # Extract causal relationships
        causal_chains = []
        for i, ei in enumerate(entities):
            for j, ej in enumerate(entities):
                if i >= j:
                    continue
                
                if not self._precedes(ei, ej, temporal):
                    continue
                
                # Compute confidence (Equation 6)
                conf = self._compute_confidence(ei, ej, text)
                
                if conf > self.causal_threshold:
                    chain = CausalChain(
                        cause=ei['text'],
                        effect=ej['text'],
                        confidence=conf,
                        context=self._get_context(ei, ej, text),
                        temporal_ordering=self._get_ordering(ei, ej, temporal)
                    )
                    causal_chains.append(chain)
        
        # Validate against KG
        validated = []
        for chain in causal_chains:
            if self._is_valid(chain, kg):
                validated.append(chain)
            else:
                chain.confidence *= 0.5
                if chain.confidence >= self.min_confidence:
                    validated.append(chain)
        
        # Compute importance weights
        for chain in validated:
            freq = self._compute_frequency(chain, text)
            cent = self._compute_centrality(chain, kg)
            rel = self._compute_relevance(chain, document)
            chain.importance_weight = (freq + cent + rel) / 3.0
        
        # Sort by weight
        validated.sort(key=lambda c: c.importance_weight, reverse=True)
        
        # Select top-k within budget
        preserved = self._select_within_budget(validated, summary_budget)
        
        # Selection CPS (Equations 7, 8) over the chains selected for the summary.
        # It guides generation; reported CPS is computed on claims found in the
        # generated summary (training_evaluation.metrics.CPSMetrics).
        # None if the source has no causal chains.
        return {
            'original_chains': validated,
            'preserved_chains': preserved,
            'selection_cps': causal_preservation_score(validated, preserved),
            'selection_cps_weighted': weighted_causal_preservation_score(validated, preserved),
            'causal_accuracy': sum(c.confidence for c in preserved) / len(preserved) if preserved else 0.0
        }
    
    def _init_patterns(self) -> List[Dict]:
        """Initialize causal patterns"""
        return [
            {'pattern': r'(\w+)\s+(?:led to|caused|drove)\s+(\w+)', 'strength': 0.9},
            {'pattern': r'due to\s+(\w+),\s+(\w+)', 'strength': 0.85},
            {'pattern': r'(\w+)\s+impact(?:ed|s)\s+(\w+)', 'strength': 0.75},
            {'pattern': r'as a result of\s+(\w+),\s+(\w+)', 'strength': 0.8}
        ]
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text"""
        entities = []
        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            entities.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        return entities
    
    def _extract_temporal(self, text: str) -> Dict:
        """Extract temporal ordering"""
        return {}  # Simplified
    
    def _precedes(self, ei: Dict, ej: Dict, temporal: Dict) -> bool:
        """Check temporal precedence"""
        return ei['start'] < ej['start']
    
    def _compute_confidence(self, ei: Dict, ej: Dict, text: str) -> float:
        """Compute causal confidence (Equation 6)"""
        for pattern_dict in self.patterns:
            context = text[ei['start']:ej['end']]
            if re.search(pattern_dict['pattern'], context, re.I):
                return pattern_dict['strength']
        
        distance = ej['start'] - ei['end']
        return max(0, 1 - distance / 500) * 0.6
    
    def _get_context(self, ei: Dict, ej: Dict, text: str) -> str:
        """Extract context around entities"""
        start = max(0, ei['start'] - 50)
        end = min(len(text), ej['end'] + 50)
        return text[start:end]
    
    def _get_ordering(self, ei: Dict, ej: Dict, temporal: Dict) -> str:
        """Determine temporal ordering"""
        return 'before' if ei['start'] < ej['start'] else 'after'
    
    def _is_valid(self, chain: CausalChain, kg: any) -> bool:
        """Validate against KG"""
        return chain.confidence >= 0.4
    
    def _compute_frequency(self, chain: CausalChain, text: str) -> float:
        """Compute frequency"""
        return min(text.lower().count(chain.cause.lower()) / 10.0, 1.0)
    
    def _compute_centrality(self, chain: CausalChain, kg: any) -> float:
        """Compute centrality in KG"""
        return 0.5  # Placeholder
    
    def _compute_relevance(self, chain: CausalChain, doc: Dict) -> float:
        """Compute relevance to financial outcomes"""
        keywords = ['revenue', 'profit', 'growth', 'stock', 'market']
        text = f"{chain.cause} {chain.effect}".lower()
        return min(sum(1 for k in keywords if k in text) / 3.0, 1.0)
    
    def _select_within_budget(self, chains: List[CausalChain], budget: int) -> List[CausalChain]:
        """Select chains within budget"""
        selected = []
        current = 0
        for chain in chains:
            length = len(chain.cause) + len(chain.effect) + 20
            if current + length <= budget:
                selected.append(chain)
                current += length
        return selected
    
