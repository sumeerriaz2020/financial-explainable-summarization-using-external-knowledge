"""
FIBO Ontology Integration
==========================

Integration layer for Financial Industry Business Ontology (FIBO) version 2024-Q1.
Provides utilities for loading, querying, and extending FIBO concepts.

FIBO Modules Used (Section 3.2.2):
- FND/Agents: 73 classes, 152 properties
- BE/LegalEntities: 47 classes, 84 properties
- FBC/Products: 68 classes, 127 properties
Combined: 188 classes, 363 properties (~78% of financial entity mentions)

This module loads a representative subset of key classes and properties from
each module; the counts above describe the full modules, not the subset.

Reference: Section 3.2.2
"""

import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FIBOClass:
    """Represents a FIBO ontology class"""
    uri: str
    label: str
    module: str
    properties: List[str]
    parent_classes: List[str]
    description: str


@dataclass
class FIBOProperty:
    """Represents a FIBO property"""
    uri: str
    label: str
    domain: str
    range: str
    property_type: str  # ObjectProperty, DatatypeProperty


class FIBOIntegration:
    """
    FIBO Ontology Integration Layer
    
    Provides interface to FIBO 2024-Q1 ontology for financial entity
    classification and relationship modeling.
    """
    
    def __init__(self, fibo_path: Optional[str] = None):
        """
        Initialize FIBO integration
        
        Args:
            fibo_path: Path to FIBO OWL file (optional, uses built-in if None)
        """
        self.fibo_path = fibo_path
        self.ontology_graph = nx.DiGraph()
        self.classes = {}
        self.properties = {}
        
        # FIBO version info
        self.version = "2024-Q1"
        self.base_uri = "https://spec.edmcouncil.org/fibo/ontology"
        
        # Load FIBO modules
        self._load_fibo_modules()
        
        logger.info(f"FIBO {self.version} integration initialized")
        logger.info(f"Loaded {len(self.classes)} classes, {len(self.properties)} properties")
    
    def _load_fibo_modules(self):
        """Load the three core FIBO modules (Section 3.2.2)"""
        
        # Module 1: FND/Agents (73 classes, 152 properties)
        self._load_fnd_agents()
        
        # Module 2: BE/LegalEntities (47 classes, 84 properties)
        self._load_be_legal_entities()
        
        # Module 3: FBC/Products (68 classes, 127 properties)
        self._load_fbc_products()
        
        # Build class hierarchy
        self._build_hierarchy()
    
    def _load_fnd_agents(self):
        """Load FND/Agents module (73 classes, 152 properties)"""
        module = "FND/Agents"
        
        # Key classes
        agent_classes = [
            ("Organization", "Legal or natural person that can act"),
            ("Person", "Natural person or human being"),
            ("AutomatedSystem", "System acting autonomously"),
            ("LegalPerson", "Person recognized by law"),
            ("AutonomousAgent", "Agent acting independently"),
            ("Group", "Collection of agents"),
            ("ExecutiveOfficer", "Senior management role"),
            ("Director", "Board member"),
            ("Shareholder", "Equity owner"),
            ("Stakeholder", "Party with interest"),
        ]
        
        for label, desc in agent_classes:
            uri = f"{self.base_uri}/{module}/{label}"
            self.classes[uri] = FIBOClass(
                uri=uri,
                label=label,
                module=module,
                properties=[],
                parent_classes=[],
                description=desc
            )
            self.ontology_graph.add_node(uri, label=label, module=module)
        
        # Key properties
        agent_properties = [
            ("hasAgent", "ObjectProperty", "relates to an agent"),
            ("isAgentOf", "ObjectProperty", "inverse of hasAgent"),
            ("controls", "ObjectProperty", "has control over"),
            ("isControlledBy", "ObjectProperty", "is under control of"),
            ("employs", "ObjectProperty", "has as employee"),
            ("isEmployedBy", "ObjectProperty", "works for"),
        ]
        
        for label, prop_type, desc in agent_properties:
            uri = f"{self.base_uri}/{module}/{label}"
            self.properties[uri] = FIBOProperty(
                uri=uri,
                label=label,
                domain=f"{self.base_uri}/{module}/Agent",
                range=f"{self.base_uri}/{module}/Agent",
                property_type=prop_type
            )
    
    def _load_be_legal_entities(self):
        """Load BE/LegalEntities module (47 classes, 84 properties)"""
        module = "BE/LegalEntities"
        
        # Key classes
        entity_classes = [
            ("Corporation", "Legal entity formed by incorporation"),
            ("Partnership", "Association of partners"),
            ("LimitedLiabilityCompany", "LLC structure"),
            ("PubliclyTradedCompany", "Company with publicly traded shares"),
            ("PrivateCompany", "Non-public company"),
            ("Subsidiary", "Company controlled by another"),
            ("Branch", "Office or division"),
            ("JointVenture", "Collaborative business entity"),
            ("Conglomerate", "Multi-industry company"),
            ("HoldingCompany", "Company holding stakes in others"),
        ]
        
        for label, desc in entity_classes:
            uri = f"{self.base_uri}/{module}/{label}"
            self.classes[uri] = FIBOClass(
                uri=uri,
                label=label,
                module=module,
                properties=[],
                parent_classes=[],
                description=desc
            )
            self.ontology_graph.add_node(uri, label=label, module=module)
        
        # Key properties
        entity_properties = [
            ("hasLegalForm", "ObjectProperty", "has legal structure"),
            ("isIncorporatedIn", "ObjectProperty", "incorporated in jurisdiction"),
            ("hasSubsidiary", "ObjectProperty", "owns subsidiary"),
            ("isSubsidiaryOf", "ObjectProperty", "is owned by parent"),
            ("hasBranch", "ObjectProperty", "has branch location"),
            ("hasRegisteredAddress", "DatatypeProperty", "legal address"),
        ]
        
        for label, prop_type, desc in entity_properties:
            uri = f"{self.base_uri}/{module}/{label}"
            self.properties[uri] = FIBOProperty(
                uri=uri,
                label=label,
                domain=f"{self.base_uri}/{module}/LegalEntity",
                range="varies",
                property_type=prop_type
            )
    
    def _load_fbc_products(self):
        """Load FBC/Products module (68 classes, 127 properties)"""
        module = "FBC/Products"
        
        # Key classes
        product_classes = [
            ("FinancialInstrument", "Financial contract or asset"),
            ("Security", "Tradable financial asset"),
            ("Equity", "Ownership interest"),
            ("Debt", "Debt obligation"),
            ("Derivative", "Derived financial instrument"),
            ("Bond", "Fixed income security"),
            ("Stock", "Equity share"),
            ("Option", "Right to buy/sell"),
            ("Future", "Forward contract"),
            ("Swap", "Exchange agreement"),
        ]
        
        for label, desc in product_classes:
            uri = f"{self.base_uri}/{module}/{label}"
            self.classes[uri] = FIBOClass(
                uri=uri,
                label=label,
                module=module,
                properties=[],
                parent_classes=[],
                description=desc
            )
            self.ontology_graph.add_node(uri, label=label, module=module)
        
        # Key properties
        product_properties = [
            ("hasIssuer", "ObjectProperty", "issued by entity"),
            ("hasCurrency", "ObjectProperty", "denominated in currency"),
            ("hasMaturity", "DatatypeProperty", "maturity date"),
            ("hasYield", "DatatypeProperty", "yield rate"),
            ("hasPrice", "DatatypeProperty", "current price"),
            ("hasVolume", "DatatypeProperty", "trading volume"),
        ]
        
        for label, prop_type, desc in product_properties:
            uri = f"{self.base_uri}/{module}/{label}"
            self.properties[uri] = FIBOProperty(
                uri=uri,
                label=label,
                domain=f"{self.base_uri}/{module}/FinancialInstrument",
                range="varies",
                property_type=prop_type
            )
    
    def _build_hierarchy(self):
        """Build class hierarchy relationships"""
        # Add key hierarchies
        hierarchies = [
            ("BE/LegalEntities/Corporation", "BE/LegalEntities/PubliclyTradedCompany"),
            ("BE/LegalEntities/Corporation", "BE/LegalEntities/PrivateCompany"),
            ("FBC/Products/Security", "FBC/Products/Equity"),
            ("FBC/Products/Security", "FBC/Products/Debt"),
            ("FBC/Products/FinancialInstrument", "FBC/Products/Security"),
            ("FBC/Products/FinancialInstrument", "FBC/Products/Derivative"),
            ("FND/Agents/Organization", "BE/LegalEntities/Corporation"),
        ]
        
        for parent, child in hierarchies:
            parent_uri = f"{self.base_uri}/{parent}"
            child_uri = f"{self.base_uri}/{child}"
            
            if parent_uri in self.classes and child_uri in self.classes:
                self.classes[child_uri].parent_classes.append(parent_uri)
                self.ontology_graph.add_edge(parent_uri, child_uri, relation="subClassOf")
    
    def get_class(self, label: str) -> Optional[FIBOClass]:
        """Get FIBO class by label"""
        for uri, fibo_class in self.classes.items():
            if fibo_class.label == label:
                return fibo_class
        return None
    
    def get_property(self, label: str) -> Optional[FIBOProperty]:
        """Get FIBO property by label"""
        for uri, prop in self.properties.items():
            if prop.label == label:
                return prop
        return None
    
    def find_classes_by_module(self, module: str) -> List[FIBOClass]:
        """Find all classes in a module"""
        return [
            cls for cls in self.classes.values()
            if cls.module == module
        ]
    
    def get_subclasses(self, class_uri: str) -> List[str]:
        """Get all subclasses of a class"""
        if class_uri not in self.ontology_graph:
            return []
        
        return list(self.ontology_graph.successors(class_uri))
    
    def get_superclasses(self, class_uri: str) -> List[str]:
        """Get all superclasses of a class"""
        if class_uri not in self.ontology_graph:
            return []
        
        return list(self.ontology_graph.predecessors(class_uri))
    
    def is_subclass_of(self, child_uri: str, parent_uri: str) -> bool:
        """Check if child is subclass of parent"""
        try:
            return nx.has_path(self.ontology_graph, parent_uri, child_uri)
        except nx.NodeNotFound:
            return False
    
    def validate_instance(self, instance_class: str, expected_class: str) -> bool:
        """Validate if instance can be of expected class"""
        instance_uri = f"{self.base_uri}/{instance_class}"
        expected_uri = f"{self.base_uri}/{expected_class}"
        
        # Direct match
        if instance_uri == expected_uri:
            return True
        
        # Check subclass relationship
        return self.is_subclass_of(instance_uri, expected_uri)
    
    def get_applicable_properties(self, class_uri: str) -> List[FIBOProperty]:
        """Get all properties applicable to a class"""
        applicable = []
        
        for prop in self.properties.values():
            # Check if property domain matches class or superclass
            if prop.domain == class_uri or self.is_subclass_of(class_uri, prop.domain):
                applicable.append(prop)
        
        return applicable
    
    def export_to_json(self, filepath: str):
        """Export FIBO structure to JSON"""
        export_data = {
            'version': self.version,
            'classes': {
                uri: {
                    'label': cls.label,
                    'module': cls.module,
                    'description': cls.description,
                    'parent_classes': cls.parent_classes
                }
                for uri, cls in self.classes.items()
            },
            'properties': {
                uri: {
                    'label': prop.label,
                    'domain': prop.domain,
                    'range': prop.range,
                    'property_type': prop.property_type
                }
                for uri, prop in self.properties.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"FIBO structure exported to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get FIBO ontology statistics"""
        return {
            'version': self.version,
            'total_classes': len(self.classes),
            'total_properties': len(self.properties),
            'modules': {
                'FND/Agents': len(self.find_classes_by_module('FND/Agents')),
                'BE/LegalEntities': len(self.find_classes_by_module('BE/LegalEntities')),
                'FBC/Products': len(self.find_classes_by_module('FBC/Products')),
            },
            'hierarchy_depth': nx.dag_longest_path_length(self.ontology_graph) if nx.is_directed_acyclic_graph(self.ontology_graph) else 0
        }


# Example usage
if __name__ == "__main__":
    print("FIBO Ontology Integration")
    print("=" * 70)
    
    # Initialize FIBO
    fibo = FIBOIntegration()
    
    # Get statistics
    stats = fibo.get_statistics()
    print(f"\nFIBO {stats['version']} Statistics:")
    print(f"Total Classes: {stats['total_classes']}")
    print(f"Total Properties: {stats['total_properties']}")
    print(f"\nClasses by Module:")
    for module, count in stats['modules'].items():
        print(f"  {module}: {count}")
    
    # Test class lookup
    print(f"\n{'=' * 70}")
    print("Testing Class Lookup:")
    print(f"{'=' * 70}")
    
    corporation = fibo.get_class("Corporation")
    if corporation:
        print(f"\nClass: {corporation.label}")
        print(f"Module: {corporation.module}")
        print(f"Description: {corporation.description}")
        print(f"URI: {corporation.uri}")
    
    # Test hierarchy
    print(f"\n{'=' * 70}")
    print("Testing Hierarchy:")
    print(f"{'=' * 70}")
    
    corp_uri = f"{fibo.base_uri}/BE/LegalEntities/Corporation"
    public_uri = f"{fibo.base_uri}/BE/LegalEntities/PubliclyTradedCompany"
    
    print(f"\nIs PubliclyTradedCompany a subclass of Corporation?")
    print(f"Result: {fibo.is_subclass_of(public_uri, corp_uri)}")
    
    # Test property lookup
    print(f"\n{'=' * 70}")
    print("Testing Property Lookup:")
    print(f"{'=' * 70}")
    
    has_issuer = fibo.get_property("hasIssuer")
    if has_issuer:
        print(f"\nProperty: {has_issuer.label}")
        print(f"Type: {has_issuer.property_type}")
        print(f"Domain: {has_issuer.domain}")
    
    print(f"\n{'=' * 70}")
    print("FIBO integration ready!")
    print(f"{'=' * 70}")
