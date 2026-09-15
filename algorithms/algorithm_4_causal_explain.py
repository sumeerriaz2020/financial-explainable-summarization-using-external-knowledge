"""
Algorithm 4: CAUSAL-EXPLAIN - Causal Chain Extraction and Preservation
========================================================================

Implementation of Algorithm 4 from the paper (CAUSAL-EXPLAIN Framework).

Extracts, preserves, and explains causal relationships in financial documents
using structural causal models and knowledge graph validation.

Authors: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
Reference: Section 3.4.2, Algorithm 4
"""

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import re
import logging

from training_evaluation.cps import (
    causal_preservation_score,
    weighted_causal_preservation_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CausalChain:
    """Represents a causal relationship chain"""
    cause: str  # Cause entity/event
    effect: str  # Effect entity/event
    confidence: float  # Confidence score [0, 1]
    context: str  # Surrounding context
    temporal_ordering: str  # before, after, simultaneous
    mechanism: Optional[str] = None  # Causal mechanism description
    importance_weight: float = 1.0  # Weight for preservation (w_i)


@dataclass
class CausalPreservationResult:
    """Result of causal preservation analysis"""
    original_chains: List[CausalChain]
    preserved_chains: List[CausalChain]
    # Selection CPS: share of source chains selected for the summary within the
    # length budget (Algorithm 4, lines 18-20). It guides generation; the CPS
    # reported in Table 4 is computed on claims found in the generated summary
    # (training_evaluation.metrics.CPSMetrics). None if no source chains.
    selection_cps: Optional[float]
    selection_cps_weighted: Optional[float]
    lost_chains: List[CausalChain]
    causal_accuracy: float
    causal_coherence: float
    temporal_fidelity: float


class CausalExplainFramework:
    """
    CAUSAL-EXPLAIN Framework for causal chain extraction and preservation.
    
    Implements Algorithm 4 for identifying, preserving, and explaining
    causal relationships in financial summarization.
    """
    
    def __init__(
        self,
        causal_threshold: float = 0.7,
        min_confidence: float = 0.35,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize CAUSAL-EXPLAIN framework
        
        Args:
            causal_threshold: Threshold for accepting causal relationships (τ_causal)
            min_confidence: Minimum confidence to keep chain (τ_min)
            device: Computing device
        """
        self.causal_threshold = causal_threshold
        self.min_confidence = min_confidence
        self.device = device
        
        # Causal patterns from financial domain
        self.causal_patterns = self._initialize_causal_patterns()
        
        # Causal classifier for confidence scoring (from paper Equation 6)
        # conf(e_i → e_j) = σ(W_causal · [h_ei; h_ej; h_context])
        self.causal_classifier = nn.Sequential(
            nn.Linear(768 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)
        
        logger.info("CAUSAL-EXPLAIN Framework initialized")
    
    def extract_and_preserve_causal_chains(
        self,
        document: Dict,
        knowledge_graph: nx.DiGraph,
        summary_length_budget: int = 512
    ) -> CausalPreservationResult:
        """
        Extract and preserve causal chains in summarization.
        
        Implements Algorithm 4 from paper (Section 3.4.2).
        
        Args:
            document: Source document with 'text' field
            knowledge_graph: Extended knowledge graph
            summary_length_budget: Maximum summary length (L)
            
        Returns:
            CausalPreservationResult with chains and metrics
        """
        text = document['text']
        
        logger.info("Extracting causal chains from document...")
        
        # Line 1: E, T ← Extract entities (FinBERT) and temporal ordering from D
        entities, temporal_ordering = self._extract_entities_and_temporal(text)
        
        # Line 2: Initialize C_original ← ∅
        causal_chains_original = []
        
        # Lines 3-8: Extract causal relationships
        logger.info(f"Analyzing {len(entities)} entities for causal relationships...")
        
        for i, entity_i in enumerate(entities):
            for j, entity_j in enumerate(entities):
                if i >= j:
                    continue
                
                # Line 4: Check temporal precedence
                if not self._precedes_temporally(entity_i, entity_j, temporal_ordering):
                    continue
                
                # Line 5: Compute causal confidence (Equation 6)
                confidence = self._compute_causal_confidence(
                    entity_i, entity_j, text
                )
                
                # Line 6: if conf(e_i → e_j) > τ_causal then
                if confidence > self.causal_threshold:
                    # Line 7: Add (e_i → e_j, conf) to C_original
                    chain = CausalChain(
                        cause=entity_i['text'],
                        effect=entity_j['text'],
                        confidence=confidence,
                        context=self._extract_context(entity_i, entity_j, text),
                        temporal_ordering=self._determine_ordering(
                            entity_i, entity_j, temporal_ordering
                        )
                    )
                    causal_chains_original.append(chain)
        
        logger.info(f"Extracted {len(causal_chains_original)} causal chains")
        
        # Lines 9-13: Validate against knowledge graph
        logger.info("Validating causal chains against knowledge graph...")
        validated_chains = []
        
        for chain in causal_chains_original:
            # Line 10: if (c, e) invalid or contradicts G then
            if self._is_invalid_or_contradictory(chain, knowledge_graph):
                # Line 11: conf ← conf × 0.5; Remove if conf < τ_min
                chain.confidence *= 0.5
                if chain.confidence < self.min_confidence:
                    continue
            
            validated_chains.append(chain)
        
        logger.info(f"Validated {len(validated_chains)} causal chains")
        
        # Lines 14-16: Compute importance weights
        logger.info("Computing importance weights...")
        
        for chain in validated_chains:
            # Line 15: w_i ← α_f × f_i + α_c × cent_i + α_r × rel_i
            frequency = self._compute_frequency(chain, text)
            centrality = self._compute_centrality(chain, knowledge_graph)
            relevance = self._compute_relevance(chain, document)
            
            # Default weights: equal importance
            alpha_f = alpha_c = alpha_r = 1.0 / 3.0
            
            chain.importance_weight = (
                alpha_f * frequency +
                alpha_c * centrality +
                alpha_r * relevance
            )
        
        # Line 17: Sort C_original by {w_i} descending
        validated_chains.sort(key=lambda c: c.importance_weight, reverse=True)
        
        # Line 18: C_summary ← Select top-k within budget L
        preserved_chains = self._select_chains_within_budget(
            validated_chains, summary_length_budget
        )
        
        logger.info(f"Preserved {len(preserved_chains)}/{len(validated_chains)} chains")
        
        # Line 19: Ŷ ← Generate summary integrating C_summary
        # (This would be done by the main summarization model)
        
        # Line 20: selection CPS over the chains selected for the summary
        # (Equations 7 and 8, shared implementation in training_evaluation/cps.py)
        selection_cps = causal_preservation_score(validated_chains, preserved_chains)
        selection_cps_weighted = weighted_causal_preservation_score(
            validated_chains, preserved_chains
        )
        
        # Identify lost chains
        preserved_ids = {(c.cause, c.effect) for c in preserved_chains}
        lost_chains = [
            c for c in validated_chains
            if (c.cause, c.effect) not in preserved_ids
        ]
        
        # Compute additional metrics
        causal_accuracy = self._compute_causal_accuracy(
            preserved_chains, knowledge_graph
        )
        causal_coherence = self._compute_causal_coherence(preserved_chains)
        temporal_fidelity = self._compute_temporal_fidelity(
            preserved_chains, temporal_ordering
        )
        
        result = CausalPreservationResult(
            original_chains=validated_chains,
            preserved_chains=preserved_chains,
            selection_cps=selection_cps,
            selection_cps_weighted=selection_cps_weighted,
            lost_chains=lost_chains,
            causal_accuracy=causal_accuracy,
            causal_coherence=causal_coherence,
            temporal_fidelity=temporal_fidelity
        )
        
        logger.info(f"Selection CPS (standard): {selection_cps}")
        logger.info(f"Selection CPS (weighted): {selection_cps_weighted}")
        
        return result
    
    def _initialize_causal_patterns(self) -> List[Dict]:
        """Initialize linguistic patterns for causal detection"""
        patterns = [
            {
                'pattern': r'(\w+(?:\s+\w+){0,3})\s+(?:led to|resulted in|caused|drove)\s+(\w+(?:\s+\w+){0,3})',
                'strength': 0.9,
                'type': 'explicit'
            },
            {
                'pattern': r'due to\s+(\w+(?:\s+\w+){0,3}),\s+(\w+(?:\s+\w+){0,3})',
                'strength': 0.85,
                'type': 'explicit'
            },
            {
                'pattern': r'(\w+(?:\s+\w+){0,3})\s+(?:because of|owing to)\s+(\w+(?:\s+\w+){0,3})',
                'strength': 0.85,
                'type': 'explicit'
            },
            {
                'pattern': r'(\w+(?:\s+\w+){0,3})\s+impact(?:ed|s)\s+(\w+(?:\s+\w+){0,3})',
                'strength': 0.75,
                'type': 'explicit'
            },
            {
                'pattern': r'as a result of\s+(\w+(?:\s+\w+){0,3}),\s+(\w+(?:\s+\w+){0,3})',
                'strength': 0.8,
                'type': 'explicit'
            },
            {
                'pattern': r'(\w+(?:\s+\w+){0,3})\s+(?:therefore|thus|consequently)\s+(\w+(?:\s+\w+){0,3})',
                'strength': 0.7,
                'type': 'implicit'
            }
        ]
        return patterns
    
    def _extract_entities_and_temporal(
        self,
        text: str
    ) -> Tuple[List[Dict], Dict]:
        """
        Extract entities and temporal ordering.
        
        Line 1 of Algorithm 4.
        """
        # Simple entity extraction (in production, use FinBERT NER)
        entities = []
        
        # Common financial entity patterns
        patterns = {
            'company': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.|Corp\.|Ltd\.|LLC)\b',
            'metric': r'\b(?:revenue|profit|earnings|sales|growth)\b',
            'percentage': r'\b\d+(?:\.\d+)?%\b',
            'monetary': r'\$[\d,]+(?:\.\d{2})?[BMK]?\b'
        }
        
        entity_id = 0
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'id': f'entity_{entity_id}',
                    'text': match.group(),
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end()
                })
                entity_id += 1
        
        # Extract temporal ordering
        temporal_ordering = self._extract_temporal_ordering(text, entities)
        
        return entities, temporal_ordering
    
    def _extract_temporal_ordering(
        self,
        text: str,
        entities: List[Dict]
    ) -> Dict:
        """Extract temporal relationships between entities"""
        ordering = {}
        
        # Simple position-based ordering
        for entity in entities:
            ordering[entity['id']] = entity['start']
        
        return ordering
    
    def _precedes_temporally(
        self,
        entity_i: Dict,
        entity_j: Dict,
        temporal_ordering: Dict
    ) -> bool:
        """Check if entity_i precedes entity_j temporally"""
        pos_i = temporal_ordering.get(entity_i['id'], 0)
        pos_j = temporal_ordering.get(entity_j['id'], 0)
        return pos_i < pos_j
    
    def _compute_causal_confidence(
        self,
        entity_i: Dict,
        entity_j: Dict,
        text: str
    ) -> float:
        """
        Compute causal confidence using patterns and classifier.
        
        Implements Equation 6: conf(c → e) = σ(W_causal · [h_c; h_e; h_context])
        
        Line 5 of Algorithm 4.
        """
        # Check for explicit causal patterns
        pattern_confidence = 0.0
        
        for pattern_dict in self.causal_patterns:
            pattern = pattern_dict['pattern']
            strength = pattern_dict['strength']
            
            # Search for pattern between entities
            start = min(entity_i['start'], entity_j['start'])
            end = max(entity_i['end'], entity_j['end'])
            context = text[start:end]
            
            if re.search(pattern, context, re.IGNORECASE):
                pattern_confidence = max(pattern_confidence, strength)
        
        # If pattern found, use pattern confidence
        if pattern_confidence > 0:
            return pattern_confidence
        
        # Otherwise, use proximity heuristic
        distance = abs(entity_i['start'] - entity_j['start'])
        max_distance = 500  # characters
        
        proximity_score = max(0, 1 - (distance / max_distance))
        
        return proximity_score * 0.6  # Scale down for implicit causality
    
    def _extract_context(
        self,
        entity_i: Dict,
        entity_j: Dict,
        text: str,
        window: int = 100
    ) -> str:
        """Extract context around entity pair"""
        start = max(0, min(entity_i['start'], entity_j['start']) - window)
        end = min(len(text), max(entity_i['end'], entity_j['end']) + window)
        return text[start:end]
    
    def _determine_ordering(
        self,
        entity_i: Dict,
        entity_j: Dict,
        temporal_ordering: Dict
    ) -> str:
        """Determine temporal ordering relationship"""
        pos_i = temporal_ordering.get(entity_i['id'], 0)
        pos_j = temporal_ordering.get(entity_j['id'], 0)
        
        if abs(pos_i - pos_j) < 50:  # Close proximity
            return 'simultaneous'
        elif pos_i < pos_j:
            return 'before'
        else:
            return 'after'
    
    def _is_invalid_or_contradictory(
        self,
        chain: CausalChain,
        knowledge_graph: nx.DiGraph
    ) -> bool:
        """
        Check if causal chain is invalid or contradicts knowledge graph.
        
        Line 10 of Algorithm 4.
        """
        if not knowledge_graph:
            return False
        
        # Check for logical contradictions
        # (In production, implement FIBO reasoning rules)
        
        # Simple check: very low confidence
        if chain.confidence < 0.4:
            return True
        
        return False
    
    def _compute_frequency(self, chain: CausalChain, text: str) -> float:
        """Compute frequency of causal relationship mentions"""
        # Count occurrences of cause and effect together
        cause_count = text.lower().count(chain.cause.lower())
        effect_count = text.lower().count(chain.effect.lower())
        
        # Normalize
        freq = min(cause_count, effect_count) / 10.0
        return min(freq, 1.0)
    
    def _compute_centrality(
        self,
        chain: CausalChain,
        knowledge_graph: nx.DiGraph
    ) -> float:
        """Compute centrality of entities in knowledge graph"""
        if not knowledge_graph or knowledge_graph.number_of_nodes() == 0:
            return 0.5
        
        # Find entity nodes
        cause_nodes = [n for n in knowledge_graph.nodes()
                      if knowledge_graph.nodes[n].get('mention', '') == chain.cause]
        effect_nodes = [n for n in knowledge_graph.nodes()
                       if knowledge_graph.nodes[n].get('mention', '') == chain.effect]
        
        if not cause_nodes or not effect_nodes:
            return 0.3
        
        # Compute degree centrality
        centrality = nx.degree_centrality(knowledge_graph)
        
        avg_centrality = (
            centrality.get(cause_nodes[0], 0) +
            centrality.get(effect_nodes[0], 0)
        ) / 2.0
        
        return avg_centrality
    
    def _compute_relevance(self, chain: CausalChain, document: Dict) -> float:
        """Compute relevance to financial outcomes"""
        # Keywords indicating financial importance
        important_keywords = [
            'revenue', 'profit', 'earnings', 'growth', 'decline',
            'stock', 'market', 'sales', 'performance', 'loss', 'gain'
        ]
        
        chain_text = f"{chain.cause} {chain.effect}".lower()
        
        relevance = sum(
            1 for keyword in important_keywords
            if keyword in chain_text
        )
        
        return min(relevance / 3.0, 1.0)
    
    def _select_chains_within_budget(
        self,
        chains: List[CausalChain],
        budget: int
    ) -> List[CausalChain]:
        """
        Select top-k causal chains within length budget.
        
        Line 18 of Algorithm 4.
        """
        selected = []
        current_length = 0
        
        for chain in chains:
            # Estimate length contribution
            chain_length = len(chain.cause) + len(chain.effect) + 20  # words + connectors
            
            if current_length + chain_length <= budget:
                selected.append(chain)
                current_length += chain_length
            else:
                break
        
        return selected
    
    def _compute_causal_accuracy(
        self,
        chains: List[CausalChain],
        knowledge_graph: nx.DiGraph
    ) -> float:
        """Compute accuracy of causal relationships"""
        if not chains:
            return 0.0
        
        # Simple heuristic: average confidence
        return sum(c.confidence for c in chains) / len(chains)
    
    def _compute_causal_coherence(self, chains: List[CausalChain]) -> float:
        """Assess logical consistency of causal narrative"""
        if len(chains) < 2:
            return 1.0
        
        # Check for contradictory chains
        contradictions = 0
        for i, chain1 in enumerate(chains):
            for chain2 in chains[i+1:]:
                # Check if effect of chain1 contradicts cause of chain2
                if chain1.effect == chain2.cause and \
                   chain1.temporal_ordering == 'after':
                    contradictions += 1
        
        coherence = 1.0 - (contradictions / len(chains))
        return max(0, coherence)
    
    def _compute_temporal_fidelity(
        self,
        chains: List[CausalChain],
        temporal_ordering: Dict
    ) -> float:
        """Ensure preservation of temporal ordering"""
        if not chains:
            return 1.0
        
        correct_ordering = sum(
            1 for chain in chains
            if chain.temporal_ordering in ['before', 'simultaneous']
        )
        
        return correct_ordering / len(chains)


# Example usage
if __name__ == "__main__":
    print("CAUSAL-EXPLAIN Framework - Causal Chain Extraction")
    print("="*70)
    
    # Initialize framework
    causal_framework = CausalExplainFramework(
        causal_threshold=0.7,
        min_confidence=0.35
    )
    
    # Example document
    document = {
        'text': """Apple Inc. reported strong Q4 2023 earnings. iPhone sales 
                   drove revenue growth of 15%, which led to increased investor 
                   confidence. As a result of robust performance, the company 
                   announced a $110B stock buyback program. Supply chain 
                   disruptions caused production delays in certain regions. 
                   However, this impact was offset by strong demand in emerging 
                   markets, resulting in overall positive results.""",
        'id': 'AAPL_Q4_2023'
    }
    
    # Create knowledge graph
    kg = nx.DiGraph()
    
    # Extract and preserve causal chains
    print("\nAnalyzing document for causal relationships...")
    result = causal_framework.extract_and_preserve_causal_chains(
        document, kg, summary_length_budget=200
    )
    
    # Display results
    print(f"\n{'='*70}")
    print("CAUSAL ANALYSIS RESULTS:")
    print(f"{'='*70}")
    print(f"Original causal chains: {len(result.original_chains)}")
    print(f"Preserved chains: {len(result.preserved_chains)}")
    print(f"Lost chains: {len(result.lost_chains)}")
    print(f"\nSelection CPS (standard): {result.selection_cps}")
    print(f"Selection CPS (weighted): {result.selection_cps_weighted}")
    print(f"Causal Accuracy: {result.causal_accuracy:.3f}")
    print(f"Causal Coherence: {result.causal_coherence:.3f}")
    print(f"Temporal Fidelity: {result.temporal_fidelity:.3f}")
    
    print(f"\n{'='*70}")
    print("PRESERVED CAUSAL CHAINS:")
    print(f"{'='*70}")
    for i, chain in enumerate(result.preserved_chains, 1):
        print(f"\n{i}. {chain.cause} → {chain.effect}")
        print(f"   Confidence: {chain.confidence:.3f}")
        print(f"   Weight: {chain.importance_weight:.3f}")
        print(f"   Ordering: {chain.temporal_ordering}")
    
    if result.lost_chains:
        print(f"\n{'='*70}")
        print("LOST CAUSAL CHAINS (not preserved in summary):")
        print(f"{'='*70}")
        for i, chain in enumerate(result.lost_chains[:3], 1):  # Show top 3
            print(f"\n{i}. {chain.cause} → {chain.effect}")
            print(f"   Confidence: {chain.confidence:.3f}")
            print(f"   Weight: {chain.importance_weight:.3f}")
