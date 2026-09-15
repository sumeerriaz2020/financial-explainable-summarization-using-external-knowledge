"""
Entity Linking to FIBO
=======================

Links financial entities extracted from text to FIBO ontology concepts
using string matching, semantic similarity, and type constraints.

Reference: Algorithm 1, Lines 5-6
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EntityMention:
    """Entity mention in text"""
    text: str
    start: int
    end: int
    entity_type: str  # ORG, PERSON, PRODUCT, etc.
    confidence: float


@dataclass
class LinkedEntity:
    """Entity linked to FIBO"""
    mention: EntityMention
    fibo_uri: str
    fibo_class: str
    linking_confidence: float
    linking_method: str


class EntityLinker:
    """Links text entities to FIBO ontology concepts"""
    
    def __init__(self, fibo_integration):
        """
        Initialize entity linker
        
        Args:
            fibo_integration: FIBOIntegration instance
        """
        self.fibo = fibo_integration
        
        # Type mappings from NER to FIBO modules
        self.type_mappings = {
            'ORG': 'BE/LegalEntities',
            'PERSON': 'FND/Agents',
            'PRODUCT': 'FBC/Products',
            'GPE': 'FND/Places',
            'MONEY': 'FBC/Products',
            'PERCENT': 'FBC/Products'
        }
        
        # Company suffixes for normalization
        self.company_suffixes = [
            'Inc.', 'Inc', 'Corporation', 'Corp.', 'Corp', 'LLC', 'L.L.C.',
            'Ltd.', 'Ltd', 'Limited', 'PLC', 'AG', 'S.A.', 'GmbH', 'Co.'
        ]
        
        # Financial instrument keywords
        self.instrument_keywords = {
            'stock': 'Stock',
            'bond': 'Bond',
            'option': 'Option',
            'future': 'Future',
            'swap': 'Swap',
            'equity': 'Equity',
            'debt': 'Debt',
            'derivative': 'Derivative',
            'security': 'Security'
        }
        
        logger.info("Entity Linker initialized")
    
    def link_entity(
        self,
        mention: EntityMention,
        context: str = ""
    ) -> Optional[LinkedEntity]:
        """
        Link entity mention to FIBO concept
        
        Args:
            mention: Entity mention from NER
            context: Surrounding context (optional)
            
        Returns:
            LinkedEntity if successful, None otherwise
        """
        # Determine target FIBO module
        target_module = self.type_mappings.get(mention.entity_type)
        
        if not target_module:
            logger.debug(f"No FIBO module for entity type: {mention.entity_type}")
            return None
        
        # Try different linking strategies
        strategies = [
            self._exact_match_linking,
            self._normalized_match_linking,
            self._type_based_linking,
            self._keyword_based_linking
        ]
        
        for strategy in strategies:
            result = strategy(mention, target_module, context)
            if result:
                return result
        
        # Fallback: generic mapping
        return self._fallback_linking(mention, target_module)
    
    def link_entities_batch(
        self,
        mentions: List[EntityMention],
        context: str = ""
    ) -> List[LinkedEntity]:
        """
        Link multiple entities in batch
        
        Args:
            mentions: List of entity mentions
            context: Document context
            
        Returns:
            List of successfully linked entities
        """
        linked = []
        
        for mention in mentions:
            result = self.link_entity(mention, context)
            if result:
                linked.append(result)
        
        logger.info(f"Linked {len(linked)}/{len(mentions)} entities")
        return linked
    
    def _exact_match_linking(
        self,
        mention: EntityMention,
        target_module: str,
        context: str
    ) -> Optional[LinkedEntity]:
        """Try exact string match with FIBO classes"""
        
        # Search for exact match in target module
        module_classes = self.fibo.find_classes_by_module(target_module)
        
        for fibo_class in module_classes:
            if fibo_class.label.lower() == mention.text.lower():
                return LinkedEntity(
                    mention=mention,
                    fibo_uri=fibo_class.uri,
                    fibo_class=fibo_class.label,
                    linking_confidence=0.95,
                    linking_method='exact_match'
                )
        
        return None
    
    def _normalized_match_linking(
        self,
        mention: EntityMention,
        target_module: str,
        context: str
    ) -> Optional[LinkedEntity]:
        """Try normalized string match (remove suffixes, etc.)"""
        
        # Normalize mention text
        normalized = self._normalize_company_name(mention.text)
        
        # Search for match
        module_classes = self.fibo.find_classes_by_module(target_module)
        
        for fibo_class in module_classes:
            normalized_class = self._normalize_company_name(fibo_class.label)
            
            if normalized.lower() == normalized_class.lower():
                return LinkedEntity(
                    mention=mention,
                    fibo_uri=fibo_class.uri,
                    fibo_class=fibo_class.label,
                    linking_confidence=0.85,
                    linking_method='normalized_match'
                )
        
        return None
    
    def _type_based_linking(
        self,
        mention: EntityMention,
        target_module: str,
        context: str
    ) -> Optional[LinkedEntity]:
        """Link based on entity type and context"""
        
        # Type-specific linking logic
        if mention.entity_type == 'ORG':
            return self._link_organization(mention, target_module, context)
        elif mention.entity_type == 'PERSON':
            return self._link_person(mention, target_module, context)
        elif mention.entity_type == 'PRODUCT':
            return self._link_product(mention, target_module, context)
        
        return None
    
    def _keyword_based_linking(
        self,
        mention: EntityMention,
        target_module: str,
        context: str
    ) -> Optional[LinkedEntity]:
        """Link based on keywords in mention or context"""
        
        mention_lower = mention.text.lower()
        context_lower = context.lower()
        
        # Check for financial instrument keywords
        for keyword, fibo_class_name in self.instrument_keywords.items():
            if keyword in mention_lower or keyword in context_lower:
                fibo_class = self.fibo.get_class(fibo_class_name)
                if fibo_class:
                    return LinkedEntity(
                        mention=mention,
                        fibo_uri=fibo_class.uri,
                        fibo_class=fibo_class.label,
                        linking_confidence=0.75,
                        linking_method='keyword_based'
                    )
        
        return None
    
    def _fallback_linking(
        self,
        mention: EntityMention,
        target_module: str
    ) -> Optional[LinkedEntity]:
        """Fallback: map to generic class in target module"""
        
        # Get generic classes for each module
        generic_mappings = {
            'BE/LegalEntities': 'Corporation',
            'FND/Agents': 'Organization',
            'FBC/Products': 'FinancialInstrument',
        }
        
        generic_class_name = generic_mappings.get(target_module)
        if not generic_class_name:
            return None
        
        fibo_class = self.fibo.get_class(generic_class_name)
        if fibo_class:
            return LinkedEntity(
                mention=mention,
                fibo_uri=fibo_class.uri,
                fibo_class=fibo_class.label,
                linking_confidence=0.50,
                linking_method='fallback'
            )
        
        return None
    
    def _link_organization(
        self,
        mention: EntityMention,
        target_module: str,
        context: str
    ) -> Optional[LinkedEntity]:
        """Link organization entity"""
        
        # Check if publicly traded
        if any(word in context.lower() for word in ['traded', 'stock', 'ticker', 'nyse', 'nasdaq']):
            fibo_class = self.fibo.get_class('PubliclyTradedCompany')
        else:
            fibo_class = self.fibo.get_class('Corporation')
        
        if fibo_class:
            return LinkedEntity(
                mention=mention,
                fibo_uri=fibo_class.uri,
                fibo_class=fibo_class.label,
                linking_confidence=0.80,
                linking_method='type_based'
            )
        
        return None
    
    def _link_person(
        self,
        mention: EntityMention,
        target_module: str,
        context: str
    ) -> Optional[LinkedEntity]:
        """Link person entity"""
        
        # Check for executive roles
        if any(role in context.lower() for role in ['ceo', 'cfo', 'president', 'director']):
            fibo_class = self.fibo.get_class('ExecutiveOfficer')
        else:
            fibo_class = self.fibo.get_class('Person')
        
        if fibo_class:
            return LinkedEntity(
                mention=mention,
                fibo_uri=fibo_class.uri,
                fibo_class=fibo_class.label,
                linking_confidence=0.80,
                linking_method='type_based'
            )
        
        return None
    
    def _link_product(
        self,
        mention: EntityMention,
        target_module: str,
        context: str
    ) -> Optional[LinkedEntity]:
        """Link product/instrument entity"""
        
        mention_lower = mention.text.lower()
        
        # Determine instrument type
        if 'stock' in mention_lower or 'share' in mention_lower:
            class_name = 'Stock'
        elif 'bond' in mention_lower:
            class_name = 'Bond'
        elif 'option' in mention_lower:
            class_name = 'Option'
        else:
            class_name = 'FinancialInstrument'
        
        fibo_class = self.fibo.get_class(class_name)
        if fibo_class:
            return LinkedEntity(
                mention=mention,
                fibo_uri=fibo_class.uri,
                fibo_class=fibo_class.label,
                linking_confidence=0.75,
                linking_method='type_based'
            )
        
        return None
    
    def _normalize_company_name(self, name: str) -> str:
        """Normalize company name by removing suffixes"""
        normalized = name
        
        for suffix in self.company_suffixes:
            # Remove suffix (case-insensitive)
            pattern = r'\s+' + re.escape(suffix) + r'\s*$'
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        return normalized.strip()
    
    def get_linking_statistics(self, linked_entities: List[LinkedEntity]) -> Dict:
        """Get statistics about linking results"""
        
        if not linked_entities:
            return {}
        
        # Count by method
        method_counts = {}
        for entity in linked_entities:
            method_counts[entity.linking_method] = \
                method_counts.get(entity.linking_method, 0) + 1
        
        # Count by FIBO class
        class_counts = {}
        for entity in linked_entities:
            class_counts[entity.fibo_class] = \
                class_counts.get(entity.fibo_class, 0) + 1
        
        # Average confidence
        avg_confidence = sum(e.linking_confidence for e in linked_entities) / len(linked_entities)
        
        return {
            'total_linked': len(linked_entities),
            'methods': method_counts,
            'classes': class_counts,
            'average_confidence': avg_confidence,
            'high_confidence': sum(1 for e in linked_entities if e.linking_confidence > 0.8),
            'medium_confidence': sum(1 for e in linked_entities if 0.6 <= e.linking_confidence <= 0.8),
            'low_confidence': sum(1 for e in linked_entities if e.linking_confidence < 0.6)
        }


# Example usage
if __name__ == "__main__":
    from knowledge_graph.fibo_integration import FIBOIntegration
    
    print("Entity Linking to FIBO")
    print("=" * 70)
    
    # Initialize
    fibo = FIBOIntegration()
    linker = EntityLinker(fibo)
    
    # Example entities
    mentions = [
        EntityMention("Apple Inc.", 0, 10, "ORG", 0.95),
        EntityMention("Tim Cook", 50, 58, "PERSON", 0.92),
        EntityMention("iPhone", 100, 106, "PRODUCT", 0.88),
        EntityMention("stock", 150, 155, "PRODUCT", 0.85)
    ]
    
    context = """Apple Inc. reported strong earnings. CEO Tim Cook announced 
                 new iPhone sales. The company's stock rose 5%."""
    
    # Link entities
    print("\nLinking entities...")
    linked = linker.link_entities_batch(mentions, context)
    
    # Display results
    print(f"\n{'=' * 70}")
    print("LINKING RESULTS:")
    print(f"{'=' * 70}")
    
    for entity in linked:
        print(f"\nMention: {entity.mention.text}")
        print(f"  FIBO Class: {entity.fibo_class}")
        print(f"  Confidence: {entity.linking_confidence:.2f}")
        print(f"  Method: {entity.linking_method}")
        print(f"  URI: {entity.fibo_uri}")
    
    # Statistics
    stats = linker.get_linking_statistics(linked)
    print(f"\n{'=' * 70}")
    print("STATISTICS:")
    print(f"{'=' * 70}")
    print(f"Total Linked: {stats['total_linked']}")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")
    print(f"\nBy Method:")
    for method, count in stats['methods'].items():
        print(f"  {method}: {count}")
