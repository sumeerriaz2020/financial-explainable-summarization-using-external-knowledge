"""
Temporal Relationship Annotation
=================================

Handles temporal relationships and annotations in financial documents,
including event ordering, time periods, and market regime detection.

Market-regime detection here is a keyword heuristic. The three-state Hidden Markov
Model of Section 3.4.5 (bull, bear, sideways; rolling 90-day VIX and sector-rotation
windows) is specified in the paper, not included in this release.

Reference: Section 3.4.5, TEMPORAL-EXPLAIN framework
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalOrdering(Enum):
    """Temporal ordering between events"""
    BEFORE = "before"
    AFTER = "after"
    SIMULTANEOUS = "simultaneous"
    OVERLAPPING = "overlapping"
    DURING = "during"
    UNKNOWN = "unknown"


class MarketRegime(Enum):
    """Market regime types: the three HMM states of Section 3.4.5"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass
class TemporalExpression:
    """Temporal expression extracted from text"""
    text: str
    start: int
    end: int
    normalized: Optional[datetime]
    expression_type: str  # date, period, duration, relative


@dataclass
class TemporalRelation:
    """Temporal relationship between entities/events"""
    entity1: str
    entity2: str
    ordering: TemporalOrdering
    confidence: float
    evidence: str


@dataclass
class MarketRegimeAnnotation:
    """Market regime annotation"""
    regime: MarketRegime
    confidence: float
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    indicators: Dict[str, float]
    evidence: List[str]
    crisis: bool = False  # crisis conditions (Section 3.4.5)


class TemporalAnnotator:
    """Annotates temporal relationships in financial documents"""
    
    def __init__(self):
        """Initialize temporal annotator"""
        self.temporal_patterns = self._initialize_temporal_patterns()
        self.ordering_markers = self._initialize_ordering_markers()
        self.regime_indicators = self._initialize_regime_indicators()
        
        logger.info("Temporal Annotator initialized")
    
    def _initialize_temporal_patterns(self) -> List[Dict]:
        """Initialize patterns for temporal expression extraction"""
        return [
            {
                'pattern': r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
                'type': 'date',
                'format': ['%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y']
            },
            {
                'pattern': r'\b(Q[1-4]\s+\d{4})\b',
                'type': 'period',
                'format': 'fiscal_quarter'
            },
            {
                'pattern': r'\b(FY\s*\d{4})\b',
                'type': 'period',
                'format': 'fiscal_year'
            },
            {
                'pattern': r'\b(\d{4})\b',
                'type': 'date',
                'format': '%Y'
            },
            {
                'pattern': r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                'type': 'date',
                'format': '%B %d, %Y'
            },
            {
                'pattern': r'\b(today|yesterday|tomorrow)\b',
                'type': 'relative',
                'format': 'relative'
            },
            {
                'pattern': r'\b(last|next|this)\s+(week|month|quarter|year)\b',
                'type': 'relative',
                'format': 'relative'
            }
        ]
    
    def _initialize_ordering_markers(self) -> Dict[str, TemporalOrdering]:
        """Initialize temporal ordering markers"""
        return {
            'before': TemporalOrdering.BEFORE,
            'prior to': TemporalOrdering.BEFORE,
            'earlier': TemporalOrdering.BEFORE,
            'previously': TemporalOrdering.BEFORE,
            
            'after': TemporalOrdering.AFTER,
            'following': TemporalOrdering.AFTER,
            'subsequently': TemporalOrdering.AFTER,
            'later': TemporalOrdering.AFTER,
            'then': TemporalOrdering.AFTER,
            
            'during': TemporalOrdering.DURING,
            'while': TemporalOrdering.DURING,
            'throughout': TemporalOrdering.DURING,
            
            'simultaneously': TemporalOrdering.SIMULTANEOUS,
            'at the same time': TemporalOrdering.SIMULTANEOUS,
            'concurrently': TemporalOrdering.SIMULTANEOUS,
        }
    
    def _initialize_regime_indicators(self) -> Dict[str, MarketRegime]:
        """Initialize market regime indicators"""
        return {
            # Bull market indicators
            'rally': MarketRegime.BULL,
            'growth': MarketRegime.BULL,
            'expansion': MarketRegime.BULL,
            'boom': MarketRegime.BULL,
            'uptrend': MarketRegime.BULL,
            'bullish': MarketRegime.BULL,
            
            # Bear market indicators
            'downturn': MarketRegime.BEAR,
            'correction': MarketRegime.BEAR,
            'decline': MarketRegime.BEAR,
            'bearish': MarketRegime.BEAR,
            'selloff': MarketRegime.BEAR,
            
            # Crisis indicators (bear regime under crisis conditions)
            'crisis': MarketRegime.BEAR,
            'crash': MarketRegime.BEAR,
            'panic': MarketRegime.BEAR,
            'recession': MarketRegime.BEAR,
            'meltdown': MarketRegime.BEAR,
            
            # Sideways indicators
            'sideways': MarketRegime.SIDEWAYS,
            'range-bound': MarketRegime.SIDEWAYS,
            'flat': MarketRegime.SIDEWAYS,
            'consolidation': MarketRegime.SIDEWAYS,
        }
    
    def extract_temporal_expressions(self, text: str) -> List[TemporalExpression]:
        """
        Extract temporal expressions from text
        
        Args:
            text: Input text
            
        Returns:
            List of temporal expressions
        """
        expressions = []
        
        for pattern_dict in self.temporal_patterns:
            matches = re.finditer(pattern_dict['pattern'], text, re.IGNORECASE)
            
            for match in matches:
                expr_text = match.group(1) if match.groups() else match.group(0)
                
                # Try to normalize to datetime
                normalized = self._normalize_temporal_expression(
                    expr_text, pattern_dict['format']
                )
                
                expression = TemporalExpression(
                    text=expr_text,
                    start=match.start(),
                    end=match.end(),
                    normalized=normalized,
                    expression_type=pattern_dict['type']
                )
                
                expressions.append(expression)
        
        logger.debug(f"Extracted {len(expressions)} temporal expressions")
        return expressions
    
    def extract_temporal_relations(
        self,
        text: str,
        entities: Optional[List[str]] = None
    ) -> List[TemporalRelation]:
        """
        Extract temporal relationships between entities/events
        
        Args:
            text: Input text
            entities: Optional list of entities to focus on
            
        Returns:
            List of temporal relations
        """
        relations = []
        
        # Extract temporal expressions first
        temporal_exprs = self.extract_temporal_expressions(text)
        
        # Look for ordering markers
        for marker, ordering in self.ordering_markers.items():
            # Find occurrences of marker
            pattern = r'(.{2,50}?)\s+' + re.escape(marker) + r'\s+(.{2,50})'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()
                
                # Clean entities
                entity1 = self._clean_entity_text(entity1)
                entity2 = self._clean_entity_text(entity2)
                
                if entity1 and entity2:
                    relation = TemporalRelation(
                        entity1=entity1,
                        entity2=entity2,
                        ordering=ordering,
                        confidence=0.80,
                        evidence=match.group(0)
                    )
                    relations.append(relation)
        
        # Position-based ordering (simple heuristic)
        if entities and len(entities) > 1:
            for i in range(len(entities) - 1):
                # Entities that appear in order are likely temporally ordered
                if entities[i] != entities[i+1]:
                    relation = TemporalRelation(
                        entity1=entities[i],
                        entity2=entities[i+1],
                        ordering=TemporalOrdering.BEFORE,
                        confidence=0.60,
                        evidence="positional"
                    )
                    relations.append(relation)
        
        logger.debug(f"Extracted {len(relations)} temporal relations")
        return relations
    
    def detect_market_regime(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> MarketRegimeAnnotation:
        """
        Detect market regime from document
        
        Args:
            text: Document text
            metadata: Optional metadata (dates, etc.)
            
        Returns:
            Market regime annotation
        """
        text_lower = text.lower()
        
        # Count regime indicators
        regime_scores = {regime: 0.0 for regime in MarketRegime}
        evidence_by_regime = {regime: [] for regime in MarketRegime}
        
        for keyword, regime in self.regime_indicators.items():
            count = text_lower.count(keyword)
            if count > 0:
                regime_scores[regime] += count
                evidence_by_regime[regime].append(f"{keyword} ({count}x)")
        
        # Determine dominant regime
        if sum(regime_scores.values()) == 0:
            dominant_regime = MarketRegime.SIDEWAYS
            confidence = 0.50
            evidence = ["No strong regime indicators"]
        else:
            dominant_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
            total_score = sum(regime_scores.values())
            confidence = min(regime_scores[dominant_regime] / total_score, 0.95)
            evidence = evidence_by_regime[dominant_regime]
        
        # Extract volatility indicator
        volatility_keywords = ['volatility', 'volatile', 'vix']
        volatility_mentions = sum(text_lower.count(kw) for kw in volatility_keywords)
        volatility_score = min(volatility_mentions * 0.2, 1.0)
        
        annotation = MarketRegimeAnnotation(
            regime=dominant_regime,
            confidence=confidence,
            start_date=None,
            end_date=None,
            indicators={'volatility': volatility_score},
            evidence=evidence,
            crisis=any(kw in text_lower for kw in ['crisis', 'crash', 'panic', 'recession', 'meltdown'])
        )
        
        logger.info(f"Detected regime: {dominant_regime.value} (conf: {confidence:.2f})")
        return annotation
    
    def compute_temporal_consistency(
        self,
        relations: List[TemporalRelation]
    ) -> float:
        """
        Compute temporal consistency score (TCC)
        
        Implements simplified version of Equation 9 from paper
        
        Args:
            relations: List of temporal relations
            
        Returns:
            Consistency score [0, 1]
        """
        if len(relations) < 2:
            return 1.0
        
        # Check for contradictions
        contradictions = 0
        total_comparisons = 0
        
        for i, rel1 in enumerate(relations):
            for rel2 in relations[i+1:]:
                # Check if relations involve same entities
                if (rel1.entity1 == rel2.entity1 and rel1.entity2 == rel2.entity2):
                    total_comparisons += 1
                    
                    # Check for contradiction
                    if rel1.ordering != rel2.ordering:
                        contradictions += 1
        
        if total_comparisons == 0:
            return 0.80  # Default score
        
        consistency = 1.0 - (contradictions / total_comparisons)
        return consistency
    
    def _normalize_temporal_expression(
        self,
        expr: str,
        format_spec: any
    ) -> Optional[datetime]:
        """Normalize temporal expression to datetime"""
        
        if format_spec == 'relative':
            return self._parse_relative_time(expr)
        elif format_spec == 'fiscal_quarter':
            return self._parse_fiscal_quarter(expr)
        elif format_spec == 'fiscal_year':
            return self._parse_fiscal_year(expr)
        elif isinstance(format_spec, list):
            # Try multiple formats
            for fmt in format_spec:
                try:
                    return datetime.strptime(expr, fmt)
                except ValueError:
                    continue
        elif isinstance(format_spec, str):
            try:
                return datetime.strptime(expr, format_spec)
            except ValueError:
                pass
        
        return None
    
    def _parse_relative_time(self, expr: str) -> Optional[datetime]:
        """Parse relative time expressions"""
        now = datetime.now()
        expr_lower = expr.lower()
        
        if expr_lower == 'today':
            return now
        elif expr_lower == 'yesterday':
            return now - timedelta(days=1)
        elif expr_lower == 'tomorrow':
            return now + timedelta(days=1)
        
        return None
    
    def _parse_fiscal_quarter(self, expr: str) -> Optional[datetime]:
        """Parse fiscal quarter (e.g., Q1 2023)"""
        match = re.match(r'Q([1-4])\s+(\d{4})', expr, re.IGNORECASE)
        if match:
            quarter = int(match.group(1))
            year = int(match.group(2))
            month = (quarter - 1) * 3 + 1
            return datetime(year, month, 1)
        return None
    
    def _parse_fiscal_year(self, expr: str) -> Optional[datetime]:
        """Parse fiscal year (e.g., FY 2023)"""
        match = re.match(r'FY\s*(\d{4})', expr, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            return datetime(year, 1, 1)
        return None
    
    def _clean_entity_text(self, text: str) -> str:
        """Clean extracted entity text"""
        # Remove leading/trailing punctuation
        text = re.sub(r'^[,;:\s]+', '', text)
        text = re.sub(r'[,;:\s]+$', '', text)
        
        # Limit length
        if len(text) > 50:
            text = text[:50]
        
        return text.strip()
    
    def get_statistics(
        self,
        temporal_exprs: List[TemporalExpression],
        temporal_rels: List[TemporalRelation]
    ) -> Dict:
        """Get statistics about temporal annotations"""
        
        # Expression type distribution
        expr_types = {}
        for expr in temporal_exprs:
            expr_types[expr.expression_type] = \
                expr_types.get(expr.expression_type, 0) + 1
        
        # Normalized expressions
        normalized_count = sum(1 for expr in temporal_exprs if expr.normalized)
        
        # Relation ordering distribution
        ordering_dist = {}
        for rel in temporal_rels:
            ordering_dist[rel.ordering.value] = \
                ordering_dist.get(rel.ordering.value, 0) + 1
        
        return {
            'total_expressions': len(temporal_exprs),
            'expression_types': expr_types,
            'normalized_expressions': normalized_count,
            'total_relations': len(temporal_rels),
            'ordering_distribution': ordering_dist,
            'avg_relation_confidence': sum(r.confidence for r in temporal_rels) / len(temporal_rels) if temporal_rels else 0.0
        }


# Example usage
if __name__ == "__main__":
    print("Temporal Relationship Annotation")
    print("=" * 70)
    
    # Initialize annotator
    annotator = TemporalAnnotator()
    
    # Example text
    text = """
    In Q4 2023, Apple Inc. reported strong earnings. Prior to this announcement,
    the stock had declined 5% due to market volatility. Following the earnings beat,
    investor sentiment improved significantly. The Federal Reserve's interest rate
    decision on December 15, 2023 occurred simultaneously with the company's guidance
    update. This led to a crisis in the broader market, though tech stocks showed
    resilience.
    """
    
    # Extract temporal expressions
    print("\nExtracting temporal expressions...")
    temporal_exprs = annotator.extract_temporal_expressions(text)
    
    print(f"\n{'=' * 70}")
    print(f"TEMPORAL EXPRESSIONS ({len(temporal_exprs)} found):")
    print(f"{'=' * 70}")
    
    for expr in temporal_exprs:
        print(f"\n  '{expr.text}'")
        print(f"    Type: {expr.expression_type}")
        if expr.normalized:
            print(f"    Normalized: {expr.normalized.strftime('%Y-%m-%d')}")
    
    # Extract temporal relations
    print(f"\n{'=' * 70}")
    print("TEMPORAL RELATIONS:")
    print(f"{'=' * 70}")
    
    temporal_rels = annotator.extract_temporal_relations(text)
    
    for rel in temporal_rels[:5]:  # Show first 5
        print(f"\n  {rel.entity1}")
        print(f"    {rel.ordering.value}")
        print(f"  {rel.entity2}")
        print(f"    Confidence: {rel.confidence:.2f}")
    
    # Detect market regime
    print(f"\n{'=' * 70}")
    print("MARKET REGIME DETECTION:")
    print(f"{'=' * 70}")
    
    regime = annotator.detect_market_regime(text)
    print(f"\nDetected Regime: {regime.regime.value.upper()}")
    print(f"Confidence: {regime.confidence:.2f}")
    print(f"Evidence: {', '.join(regime.evidence)}")
    
    # Statistics
    stats = annotator.get_statistics(temporal_exprs, temporal_rels)
    print(f"\n{'=' * 70}")
    print("STATISTICS:")
    print(f"{'=' * 70}")
    print(f"Total Expressions: {stats['total_expressions']}")
    print(f"Total Relations: {stats['total_relations']}")
    print(f"Avg Relation Confidence: {stats['avg_relation_confidence']:.2f}")
