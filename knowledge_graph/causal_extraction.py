"""
Causal Relationship Extraction
===============================

Extracts causal relationships from financial text using linguistic patterns,
dependency parsing, and semantic analysis.

Reference: Algorithm 1, Lines 7-8; Algorithm 4
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CausalPattern:
    """Pattern for detecting causal relationships"""
    pattern: str
    confidence: float
    pattern_type: str  # explicit, implicit, conditional


@dataclass
class CausalRelation:
    """Extracted causal relationship"""
    cause: str
    effect: str
    confidence: float
    pattern_used: str
    context: str
    start_pos: int
    end_pos: int


class CausalExtractor:
    """Extracts causal relationships from financial text"""
    
    def __init__(self, min_confidence: float = 0.70):
        """
        Initialize causal extractor
        
        Args:
            min_confidence: Minimum confidence threshold
        """
        self.min_confidence = min_confidence
        self.patterns = self._initialize_patterns()
        
        logger.info(f"Causal Extractor initialized with {len(self.patterns)} patterns")
    
    def _initialize_patterns(self) -> List[CausalPattern]:
        """Initialize causal linguistic patterns"""
        
        patterns = [
            # Explicit causation (high confidence)
            CausalPattern(
                r'(.{2,50}?)\s+(?:led to|resulted in|caused|drove)\s+(.{2,50})',
                confidence=0.90,
                pattern_type='explicit'
            ),
            CausalPattern(
                r'due to\s+(.{2,50}?),\s+(.{2,50})',
                confidence=0.85,
                pattern_type='explicit'
            ),
            CausalPattern(
                r'(.{2,50}?)\s+(?:because of|owing to)\s+(.{2,50})',
                confidence=0.85,
                pattern_type='explicit'
            ),
            CausalPattern(
                r'as a result of\s+(.{2,50}?),\s+(.{2,50})',
                confidence=0.80,
                pattern_type='explicit'
            ),
            CausalPattern(
                r'(.{2,50}?)\s+(?:triggered|prompted|induced)\s+(.{2,50})',
                confidence=0.80,
                pattern_type='explicit'
            ),
            
            # Impact patterns
            CausalPattern(
                r'(.{2,50}?)\s+impact(?:ed|s)\s+(.{2,50})',
                confidence=0.75,
                pattern_type='explicit'
            ),
            CausalPattern(
                r'(.{2,50}?)\s+affect(?:ed|s)\s+(.{2,50})',
                confidence=0.75,
                pattern_type='explicit'
            ),
            CausalPattern(
                r'(.{2,50}?)\s+influence(?:d|s)\s+(.{2,50})',
                confidence=0.70,
                pattern_type='explicit'
            ),
            
            # Implicit causation (medium confidence)
            CausalPattern(
                r'(.{2,50}?)[,;]\s+(?:therefore|thus|consequently|hence)\s+(.{2,50})',
                confidence=0.70,
                pattern_type='implicit'
            ),
            CausalPattern(
                r'given\s+(.{2,50}?),\s+(.{2,50})',
                confidence=0.65,
                pattern_type='implicit'
            ),
            
            # Conditional causation
            CausalPattern(
                r'if\s+(.{2,50}?),\s+(?:then\s+)?(.{2,50})',
                confidence=0.60,
                pattern_type='conditional'
            ),
            CausalPattern(
                r'when\s+(.{2,50}?),\s+(.{2,50})',
                confidence=0.60,
                pattern_type='conditional'
            ),
        ]
        
        return patterns
    
    def extract_causal_relations(
        self,
        text: str,
        entities: Optional[List] = None
    ) -> List[CausalRelation]:
        """
        Extract causal relations from text
        
        Args:
            text: Input text
            entities: Optional list of extracted entities
            
        Returns:
            List of extracted causal relations
        """
        relations = []
        
        # Apply each pattern
        for pattern_obj in self.patterns:
            matches = re.finditer(pattern_obj.pattern, text, re.IGNORECASE)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    cause_text = match.group(1).strip()
                    effect_text = match.group(2).strip()
                    
                    # Clean extracted text
                    cause_text = self._clean_extracted_text(cause_text)
                    effect_text = self._clean_extracted_text(effect_text)
                    
                    # Skip if too short or generic
                    if len(cause_text) < 3 or len(effect_text) < 3:
                        continue
                    
                    if self._is_too_generic(cause_text) or self._is_too_generic(effect_text):
                        continue
                    
                    # Extract context
                    context = self._extract_context(text, match.start(), match.end())
                    
                    relation = CausalRelation(
                        cause=cause_text,
                        effect=effect_text,
                        confidence=pattern_obj.confidence,
                        pattern_used=pattern_obj.pattern_type,
                        context=context,
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    
                    relations.append(relation)
        
        # Filter by confidence
        relations = [r for r in relations if r.confidence >= self.min_confidence]
        
        # Remove duplicates
        relations = self._remove_duplicates(relations)
        
        logger.info(f"Extracted {len(relations)} causal relations")
        return relations
    
    def compute_causal_confidence(
        self,
        cause: str,
        effect: str,
        context: str
    ) -> float:
        """
        Compute confidence score for causal relationship
        
        Implements Equation 6 from paper (simplified version)
        
        Args:
            cause: Cause text
            effect: Effect text
            context: Surrounding context
            
        Returns:
            Confidence score [0, 1]
        """
        # Check for explicit markers
        explicit_markers = [
            'led to', 'caused', 'resulted in', 'drove', 'triggered',
            'because', 'due to', 'as a result'
        ]
        
        context_lower = context.lower()
        has_explicit = any(marker in context_lower for marker in explicit_markers)
        
        # Base confidence
        if has_explicit:
            base_confidence = 0.85
        else:
            base_confidence = 0.65
        
        # Adjust based on proximity
        cause_pos = context_lower.find(cause.lower())
        effect_pos = context_lower.find(effect.lower())
        
        if cause_pos >= 0 and effect_pos >= 0:
            distance = abs(effect_pos - cause_pos)
            proximity_bonus = max(0, 1 - distance / 200) * 0.15
        else:
            proximity_bonus = 0
        
        # Adjust based on financial relevance
        financial_terms = [
            'revenue', 'profit', 'earnings', 'sales', 'growth',
            'stock', 'price', 'market', 'performance', 'impact'
        ]
        
        relevance_count = sum(
            1 for term in financial_terms
            if term in context_lower
        )
        relevance_bonus = min(relevance_count * 0.05, 0.15)
        
        total_confidence = min(base_confidence + proximity_bonus + relevance_bonus, 1.0)
        
        return total_confidence
    
    def extract_causal_chains(
        self,
        text: str,
        max_chain_length: int = 5
    ) -> List[List[CausalRelation]]:
        """
        Extract chains of causal relationships
        
        Args:
            text: Input text
            max_chain_length: Maximum chain length
            
        Returns:
            List of causal chains
        """
        # Extract all relations
        relations = self.extract_causal_relations(text)
        
        # Build chains
        chains = []
        
        for relation in relations:
            # Start new chain
            chain = [relation]
            
            # Try to extend chain
            current_effect = relation.effect
            
            for _ in range(max_chain_length - 1):
                # Find relation where cause matches current effect
                next_relation = None
                for r in relations:
                    if self._texts_match(r.cause, current_effect):
                        next_relation = r
                        break
                
                if next_relation:
                    chain.append(next_relation)
                    current_effect = next_relation.effect
                else:
                    break
            
            # Add chain if length > 1
            if len(chain) > 1:
                chains.append(chain)
        
        return chains
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted cause/effect text"""
        # Remove leading/trailing punctuation
        text = re.sub(r'^[,;:\s]+', '', text)
        text = re.sub(r'[,;:\s]+$', '', text)
        
        # Remove sentence boundaries
        text = re.sub(r'\.$', '', text)
        
        # Limit length
        if len(text) > 100:
            text = text[:100] + '...'
        
        return text.strip()
    
    def _is_too_generic(self, text: str) -> bool:
        """Check if text is too generic to be meaningful"""
        generic_words = [
            'this', 'that', 'these', 'those', 'it', 'they',
            'the', 'a', 'an', 'some', 'such'
        ]
        
        text_lower = text.lower()
        
        # Check if starts with generic word
        for word in generic_words:
            if text_lower.startswith(word + ' '):
                return True
        
        # Check if too short after cleaning
        words = text.split()
        if len(words) < 2:
            return True
        
        return False
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Extract context around match"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _remove_duplicates(self, relations: List[CausalRelation]) -> List[CausalRelation]:
        """Remove duplicate or highly similar relations"""
        unique = []
        
        for relation in relations:
            is_duplicate = False
            
            for existing in unique:
                if (self._texts_match(relation.cause, existing.cause) and
                    self._texts_match(relation.effect, existing.effect)):
                    # Duplicate found, keep higher confidence
                    if relation.confidence > existing.confidence:
                        unique.remove(existing)
                    else:
                        is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(relation)
        
        return unique
    
    def _texts_match(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """Check if two texts match (fuzzy)"""
        text1_words = set(text1.lower().split())
        text2_words = set(text2.lower().split())
        
        if not text1_words or not text2_words:
            return False
        
        intersection = len(text1_words & text2_words)
        union = len(text1_words | text2_words)
        
        jaccard = intersection / union if union > 0 else 0
        
        return jaccard >= threshold
    
    def get_statistics(self, relations: List[CausalRelation]) -> Dict:
        """Get statistics about extracted relations"""
        if not relations:
            return {}
        
        # Count by pattern type
        pattern_counts = {}
        for rel in relations:
            pattern_counts[rel.pattern_used] = \
                pattern_counts.get(rel.pattern_used, 0) + 1
        
        # Confidence distribution
        avg_confidence = sum(r.confidence for r in relations) / len(relations)
        max_confidence = max(r.confidence for r in relations)
        min_confidence = min(r.confidence for r in relations)
        
        return {
            'total_relations': len(relations),
            'pattern_distribution': pattern_counts,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'high_confidence': sum(1 for r in relations if r.confidence > 0.8),
            'medium_confidence': sum(1 for r in relations if 0.6 <= r.confidence <= 0.8),
            'low_confidence': sum(1 for r in relations if r.confidence < 0.6)
        }


# Example usage
if __name__ == "__main__":
    print("Causal Relationship Extraction")
    print("=" * 70)
    
    # Initialize extractor
    extractor = CausalExtractor(min_confidence=0.70)
    
    # Example text
    text = """
    Apple Inc. reported strong Q4 earnings. The iPhone 15 launch led to 
    increased revenue growth of 12%. Due to supply chain improvements, 
    production costs decreased significantly. As a result of higher margins, 
    the stock price rose 8%. However, currency headwinds impacted international 
    sales. The Federal Reserve's interest rate hike caused market volatility, 
    which affected investor sentiment.
    """
    
    # Extract relations
    print("\nExtracting causal relations...")
    relations = extractor.extract_causal_relations(text)
    
    # Display results
    print(f"\n{'=' * 70}")
    print(f"EXTRACTED CAUSAL RELATIONS ({len(relations)} found):")
    print(f"{'=' * 70}")
    
    for i, rel in enumerate(relations, 1):
        print(f"\n{i}. CAUSE: {rel.cause}")
        print(f"   EFFECT: {rel.effect}")
        print(f"   Confidence: {rel.confidence:.2f}")
        print(f"   Pattern: {rel.pattern_used}")
    
    # Extract chains
    chains = extractor.extract_causal_chains(text)
    
    if chains:
        print(f"\n{'=' * 70}")
        print(f"CAUSAL CHAINS ({len(chains)} found):")
        print(f"{'=' * 70}")
        
        for i, chain in enumerate(chains, 1):
            print(f"\nChain {i} (length: {len(chain)}):")
            for j, rel in enumerate(chain):
                print(f"  {j+1}. {rel.cause} → {rel.effect}")
    
    # Statistics
    stats = extractor.get_statistics(relations)
    print(f"\n{'=' * 70}")
    print("STATISTICS:")
    print(f"{'=' * 70}")
    print(f"Total Relations: {stats['total_relations']}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"High Confidence (>0.8): {stats['high_confidence']}")
    print(f"\nBy Pattern Type:")
    for pattern, count in stats['pattern_distribution'].items():
        print(f"  {pattern}: {count}")
