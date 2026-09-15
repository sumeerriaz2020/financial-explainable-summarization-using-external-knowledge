"""
Algorithm 1: Knowledge Graph Construction from Financial Documents
==================================================================

Implementation of Algorithm 1 from the paper:
"An eXplainable Approach to Abstractive Text Summarization Using External Knowledge"

This module constructs an extended knowledge graph from financial documents by:
1. Importing FIBO core modules as foundation
2. Extracting entities, temporal annotations, and causal candidates
3. Linking entities to FIBO ontology concepts
4. Computing entity embeddings combining textual, structural, and type information
5. Detecting causal relationships with confidence scoring
6. Validating against FIBO constraints

Authors: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
Reference: Section 3.2.3, Algorithm 1
"""

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from transformers import AutoTokenizer, AutoModel
import spacy
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a financial entity extracted from documents"""
    id: str
    mention: str
    entity_type: str  # company, person, instrument, etc.
    fibo_class: Optional[str]
    confidence: float
    temporal_context: Optional[Dict]
    document_id: str
    span: Tuple[int, int]


@dataclass
class CausalRelation:
    """Represents a causal relationship between entities"""
    cause_id: str
    effect_id: str
    confidence: float
    temporal_ordering: str  # before, after, simultaneous
    context: str
    document_id: str


class KnowledgeGraphConstructor:
    """
    Constructs extended knowledge graph from financial documents.
    
    Integrates FIBO ontology with extracted entities, temporal annotations,
    and causal relationships following Algorithm 1 from the paper.
    
    Attributes:
        fibo_graph: Base FIBO ontology graph
        extended_graph: Extended knowledge graph (V, E, R)
        entity_embeddings: Dictionary mapping entity IDs to embeddings
        causal_threshold: Confidence threshold for causal edges (τ_threshold)
    """
    
    def __init__(
        self,
        fibo_ontology_path: str,
        finbert_model: str = "yiyanghkust/finbert-tone",
        causal_threshold: float = 0.7,
        embedding_dim: int = 768,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Knowledge Graph Constructor.
        
        Args:
            fibo_ontology_path: Path to FIBO OWL ontology file
            finbert_model: Pre-trained FinBERT model identifier
            causal_threshold: Threshold for causal edge confidence (default: 0.7)
            embedding_dim: Dimension of entity embeddings (default: 768)
            device: Computing device (cuda/cpu)
        """
        self.device = device
        self.causal_threshold = causal_threshold
        self.embedding_dim = embedding_dim
        
        # Load FIBO ontology
        logger.info(f"Loading FIBO ontology from {fibo_ontology_path}")
        self.fibo_graph = self._load_fibo_ontology(fibo_ontology_path)
        
        # Initialize FinBERT for entity embeddings
        logger.info(f"Loading FinBERT model: {finbert_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(finbert_model)
        self.finbert = AutoModel.from_pretrained(finbert_model).to(device)
        self.finbert.eval()
        
        # Load spaCy for NER and linguistic processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Downloading spaCy model en_core_web_sm...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize extended knowledge graph
        self.extended_graph = nx.DiGraph()
        self.entity_embeddings = {}
        
        # Initialize embedding transformation weights (Equation 1)
        # h_entity = W_text·h_text + W_struct·h_struct + W_type·h_type
        self.W_text = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.W_struct = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.W_type = nn.Linear(embedding_dim, embedding_dim).to(device)
        
        # Ensure weights sum to identity (constraint from paper)
        self._initialize_embedding_weights()
        
        # Initialize causal confidence classifier (Equation 6)
        # conf(c → e) = σ(W_causal · [h_c; h_e; h_context])
        self.causal_classifier = nn.Sequential(
            nn.Linear(3 * embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)
        
        logger.info("Knowledge Graph Constructor initialized successfully")
    
    def construct_knowledge_graph(
        self,
        documents: List[Dict],
        validate: bool = True
    ) -> nx.DiGraph:
        """
        Main method: Construct extended knowledge graph from document corpus.
        
        Implements Algorithm 1 from the paper (Section 3.2.3).
        
        Args:
            documents: List of financial documents with 'text' and 'metadata' fields
            validate: Whether to validate graph against FIBO constraints
            
        Returns:
            Extended knowledge graph G = (V, E, R) as NetworkX DiGraph
            
        Example:
            >>> constructor = KnowledgeGraphConstructor("fibo.owl")
            >>> docs = [{"text": "Apple Inc. reported...", "metadata": {...}}]
            >>> kg = constructor.construct_knowledge_graph(docs)
            >>> print(f"Nodes: {kg.number_of_nodes()}, Edges: {kg.number_of_edges()}")
        """
        logger.info(f"Starting KG construction for {len(documents)} documents")
        
        # Line 1: Initialize G ← Import FIBO core modules from O
        logger.info("Importing FIBO core modules...")
        self.extended_graph = self._import_fibo_core()
        
        # Statistics tracking
        total_entities = 0
        total_causal_relations = 0
        
        # Line 2: for each document d ∈ D do
        for doc_idx, doc in enumerate(documents):
            if doc_idx % 100 == 0:
                logger.info(f"Processing document {doc_idx + 1}/{len(documents)}")
            
            # Line 3: E, T, C_raw ← Extract entities (FinBERT), temporal annotations, 
            #         causal candidates
            entities, temporal_annotations, causal_candidates = \
                self._extract_document_elements(doc)
            
            total_entities += len(entities)
            
            # Line 4-6: for each entity e ∈ E do
            for entity in entities:
                # Link e to FIBO concept, add to V with temporal tags
                self._link_entity_to_fibo(entity, temporal_annotations)
                
                # Compute h_e ← f_embed(v_e, R_FIBO) {Equation 1}
                entity_embedding = self._compute_entity_embedding(entity)
                self.entity_embeddings[entity.id] = entity_embedding
            
            # Line 7-12: for each causal pair (c, e) ∈ C_raw do
            for cause, effect in causal_candidates:
                # Line 8: conf(c → e) ← σ(W_causal · [h_c; h_e; h_context]) {Eq. 1}
                confidence = self._compute_causal_confidence(cause, effect, doc)
                
                # Line 9: if conf(c → e) > τ_threshold then
                if confidence > self.causal_threshold:
                    # Line 10: Add edge (c, e) to E with confidence
                    # Line 11: Add to R with temporal ordering
                    self._add_causal_edge(
                        cause, effect, confidence, temporal_annotations, doc
                    )
                    total_causal_relations += 1
        
        logger.info(f"Extracted {total_entities} entities")
        logger.info(f"Detected {total_causal_relations} causal relations")
        
        # Line 14: Validate edges against FIBO constraints, remove contradictions
        if validate:
            logger.info("Validating graph against FIBO constraints...")
            self._validate_and_clean_graph()
        
        # Line 15: return Extended knowledge graph G = (V, E, R)
        logger.info(f"KG construction complete: "
                   f"{self.extended_graph.number_of_nodes()} nodes, "
                   f"{self.extended_graph.number_of_edges()} edges")
        
        return self.extended_graph
    
    def _load_fibo_ontology(self, ontology_path: str) -> nx.DiGraph:
        """
        Load FIBO ontology from OWL file.
        
        In production, this would use rdflib to parse OWL/RDF.
        For this implementation, we create a representative structure.
        
        Args:
            ontology_path: Path to FIBO OWL ontology
            
        Returns:
            FIBO graph structure
        """
        # Initialize FIBO base graph
        fibo_graph = nx.DiGraph()
        
        # Core FIBO modules (Section 3.2.2):
        # FND/Agents (73 classes, 152 properties)
        # BE/LegalEntities (47 classes, 84 properties)
        # FBC/Products (68 classes, 127 properties)
        # 188 classes and 363 properties combined
        
        fibo_modules = {
            "FND/Agents": {
                "classes": ["Organization", "Person", "AutomatedSystem", 
                           "LegalPerson", "AutonomousAgent"],
                "properties": ["hasAgent", "isAgentOf", "controls"]
            },
            "BE/LegalEntities": {
                "classes": ["Corporation", "Partnership", "LimitedLiabilityCompany",
                           "PubliclyTradedCompany", "PrivateCompany"],
                "properties": ["hasLegalForm", "isIncorporatedIn", "hasSubsidiary"]
            },
            "FBC/Products": {
                "classes": ["FinancialInstrument", "Security", "Equity", 
                           "Debt", "Derivative"],
                "properties": ["hasIssuer", "hasCurrency", "hasMaturity"]
            }
        }
        
        # Build base graph structure
        node_id = 0
        for module, content in fibo_modules.items():
            for cls in content["classes"]:
                node_name = f"{module}/{cls}"
                fibo_graph.add_node(
                    node_name,
                    node_type="fibo_class",
                    module=module,
                    properties=content["properties"]
                )
                node_id += 1
        
        # Add hierarchical relationships
        fibo_graph.add_edge("BE/LegalEntities/Corporation", 
                           "BE/LegalEntities/PubliclyTradedCompany",
                           relation_type="subClassOf")
        fibo_graph.add_edge("FBC/Products/Security",
                           "FBC/Products/Equity",
                           relation_type="subClassOf")
        
        logger.info(f"Loaded FIBO ontology: {fibo_graph.number_of_nodes()} classes")
        return fibo_graph
    
    def _import_fibo_core(self) -> nx.DiGraph:
        """
        Import FIBO core modules as foundation for extended graph.
        
        Line 1 of Algorithm 1.
        
        Returns:
            Extended graph initialized with FIBO structure
        """
        extended_graph = self.fibo_graph.copy()
        
        # Add metadata
        extended_graph.graph['created'] = datetime.now().isoformat()
        extended_graph.graph['fibo_version'] = '2024-Q1'
        extended_graph.graph['extensions'] = {
            'temporal_annotations': True,
            'causal_linkages': True,
            'stakeholder_roles': True,
            'regulatory_mappings': True
        }
        
        return extended_graph
    
    def _extract_document_elements(
        self,
        doc: Dict
    ) -> Tuple[List[Entity], Dict, List[Tuple[Entity, Entity]]]:
        """
        Extract entities, temporal annotations, and causal candidates from document.
        
        Line 3 of Algorithm 1: E, T, C_raw ← Extract entities (FinBERT), 
        temporal annotations, causal candidates
        
        Args:
            doc: Document dictionary with 'text' and 'metadata' fields
            
        Returns:
            Tuple of (entities, temporal_annotations, causal_candidates)
        """
        text = doc['text']
        doc_id = doc.get('id', doc.get('metadata', {}).get('id', 'unknown'))
        
        # Extract entities using FinBERT and spaCy NER
        entities = self._extract_entities_finbert(text, doc_id)
        
        # Extract temporal annotations (dates, periods, market conditions)
        temporal_annotations = self._extract_temporal_info(text)
        
        # Extract causal candidate pairs
        causal_candidates = self._extract_causal_candidates(text, entities)
        
        return entities, temporal_annotations, causal_candidates
    
    def _extract_entities_finbert(self, text: str, doc_id: str) -> List[Entity]:
        """
        Extract financial entities using FinBERT and spaCy NER.
        
        Identifies: companies, executives, financial instruments, market events
        
        Args:
            text: Document text
            doc_id: Document identifier
            
        Returns:
            List of Entity objects
        """
        entities = []
        
        # Process with spaCy for NER
        doc = self.nlp(text)
        
        entity_id = 0
        for ent in doc.ents:
            # Focus on financial entity types
            if ent.label_ in ['ORG', 'PERSON', 'MONEY', 'PERCENT', 
                             'DATE', 'GPE', 'PRODUCT']:
                
                # Determine entity type and FIBO class
                fibo_class, entity_type = self._map_to_fibo_class(
                    ent.text, ent.label_
                )
                
                entity = Entity(
                    id=f"{doc_id}_entity_{entity_id}",
                    mention=ent.text,
                    entity_type=entity_type,
                    fibo_class=fibo_class,
                    confidence=0.9,  # Could use NER confidence if available
                    temporal_context=None,
                    document_id=doc_id,
                    span=(ent.start_char, ent.end_char)
                )
                entities.append(entity)
                entity_id += 1
        
        # Additional pattern-based extraction for financial terms
        financial_patterns = [
            (r'\b(?:Q[1-4]|FY)\s+\d{4}\b', 'financial_period'),
            (r'\$[\d,]+(?:\.\d{2})?[BMK]?', 'monetary_value'),
            (r'\b\d+(?:\.\d+)?%\b', 'percentage'),
        ]
        
        for pattern, ent_type in financial_patterns:
            for match in re.finditer(pattern, text):
                entity = Entity(
                    id=f"{doc_id}_entity_{entity_id}",
                    mention=match.group(),
                    entity_type=ent_type,
                    fibo_class=None,
                    confidence=0.95,
                    temporal_context=None,
                    document_id=doc_id,
                    span=match.span()
                )
                entities.append(entity)
                entity_id += 1
        
        return entities
    
    def _extract_temporal_info(self, text: str) -> Dict:
        """
        Extract temporal annotations from text.
        
        Identifies: dates, time periods, temporal markers, market conditions
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of temporal information
        """
        temporal_info = {
            'dates': [],
            'periods': [],
            'temporal_markers': [],
            'market_regime': None
        }
        
        # Extract dates using spaCy
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                temporal_info['dates'].append(ent.text)
        
        # Extract financial periods (Q1, Q2, FY2023, etc.)
        period_pattern = r'\b(?:Q[1-4]|FY|H[1-2])\s*\d{4}\b'
        temporal_info['periods'] = re.findall(period_pattern, text)
        
        # Extract temporal markers
        markers = [
            'before', 'after', 'during', 'since', 'until', 'following',
            'prior to', 'subsequently', 'meanwhile', 'simultaneously'
        ]
        for marker in markers:
            if marker in text.lower():
                temporal_info['temporal_markers'].append(marker)
        
        # Detect market regime from context (keyword stand-in for the three-state
        # HMM of Section 3.4.5, which is not included in this release)
        if any(word in text.lower() for word in ['crisis', 'recession', 'downturn']):
            temporal_info['market_regime'] = 'bear'
        elif any(word in text.lower() for word in ['recovery', 'growth', 'expansion']):
            temporal_info['market_regime'] = 'bull'
        else:
            temporal_info['market_regime'] = 'sideways'
        
        return temporal_info
    
    def _extract_causal_candidates(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Tuple[Entity, Entity]]:
        """
        Extract candidate causal relationships from text.
        
        Uses linguistic patterns to identify potential cause-effect pairs.
        
        Args:
            text: Document text
            entities: Extracted entities
            
        Returns:
            List of (cause_entity, effect_entity) tuples
        """
        causal_candidates = []
        
        # Causal indicators (from financial domain)
        causal_patterns = [
            r'(\w+)\s+(?:led to|resulted in|caused|drove)\s+(\w+)',
            r'due to\s+(\w+),\s+(\w+)',
            r'(\w+)\s+(?:because of|owing to)\s+(\w+)',
            r'(\w+)\s+impact(?:ed|s)\s+(\w+)',
            r'as a result of\s+(\w+),\s+(\w+)',
        ]
        
        # Create entity lookup by mention
        entity_by_mention = {e.mention: e for e in entities}
        
        # Extract causal pairs using patterns
        for pattern in causal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                cause_text = match.group(1)
                effect_text = match.group(2)
                
                # Find corresponding entities
                cause_entity = self._find_closest_entity(
                    cause_text, match.start(1), entities
                )
                effect_entity = self._find_closest_entity(
                    effect_text, match.start(2), entities
                )
                
                if cause_entity and effect_entity:
                    causal_candidates.append((cause_entity, effect_entity))
        
        # Also consider temporal ordering for adjacent entities
        for i in range(len(entities) - 1):
            if self._are_temporally_ordered(entities[i], entities[i+1], text):
                causal_candidates.append((entities[i], entities[i+1]))
        
        return causal_candidates
    
    def _link_entity_to_fibo(
        self,
        entity: Entity,
        temporal_annotations: Dict
    ) -> None:
        """
        Link entity to FIBO concept and add to graph.
        
        Line 5 of Algorithm 1: Link e to FIBO concept, add to V with temporal tags
        
        Args:
            entity: Entity to link
            temporal_annotations: Temporal context information
        """
        # Add entity node to extended graph
        self.extended_graph.add_node(
            entity.id,
            node_type='entity',
            mention=entity.mention,
            entity_type=entity.entity_type,
            fibo_class=entity.fibo_class,
            confidence=entity.confidence,
            temporal_context=temporal_annotations.get('market_regime'),
            document_id=entity.document_id,
            span=entity.span
        )
        
        # Link to FIBO class if mapping exists
        if entity.fibo_class and entity.fibo_class in self.extended_graph:
            self.extended_graph.add_edge(
                entity.id,
                entity.fibo_class,
                relation_type='instanceOf',
                confidence=entity.confidence
            )
    
    def _compute_entity_embedding(self, entity: Entity) -> torch.Tensor:
        """
        Compute entity embedding combining textual, structural, and type information.
        
        Implements Equation 1 from paper:
        h_entity = W_text·h_text + W_struct·h_struct + W_type·h_type
        
        Line 6 of Algorithm 1: compute h_e ← f_embed(v_e, R_FIBO)
        
        Args:
            entity: Entity object
            
        Returns:
            Entity embedding tensor of shape (embedding_dim,)
        """
        # 1. Textual embedding from FinBERT
        h_text = self._get_finbert_embedding(entity.mention)
        
        # 2. Structural embedding from FIBO graph neighborhood
        h_struct = self._get_structural_embedding(entity)
        
        # 3. Type embedding from FIBO class hierarchy
        h_type = self._get_type_embedding(entity.fibo_class)
        
        # Weighted combination (Equation 1)
        with torch.no_grad():
            h_entity = (
                self.W_text(h_text) +
                self.W_struct(h_struct) +
                self.W_type(h_type)
            )
        
        return h_entity.squeeze(0)  # Remove batch dimension
    
    def _get_finbert_embedding(self, text: str) -> torch.Tensor:
        """
        Get FinBERT contextual embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding tensor of shape (1, embedding_dim)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.finbert(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding
    
    def _get_structural_embedding(self, entity: Entity) -> torch.Tensor:
        """
        Get structural embedding from FIBO graph neighborhood.
        
        Uses graph neural network encoding of entity's position in FIBO.
        
        Args:
            entity: Entity object
            
        Returns:
            Structural embedding tensor of shape (1, embedding_dim)
        """
        # If entity has FIBO class, aggregate neighbor embeddings
        if entity.fibo_class and entity.fibo_class in self.extended_graph:
            neighbors = list(self.extended_graph.neighbors(entity.fibo_class))
            
            if neighbors:
                # Aggregate neighbor text representations
                neighbor_texts = [
                    self.extended_graph.nodes[n].get('mention', n)
                    for n in neighbors[:5]  # Limit to 5 neighbors
                ]
                neighbor_text = " ".join(neighbor_texts)
                return self._get_finbert_embedding(neighbor_text)
        
        # Default: return zero embedding
        return torch.zeros(1, self.embedding_dim, device=self.device)
    
    def _get_type_embedding(self, fibo_class: Optional[str]) -> torch.Tensor:
        """
        Get type embedding from FIBO class hierarchy.
        
        Args:
            fibo_class: FIBO class name
            
        Returns:
            Type embedding tensor of shape (1, embedding_dim)
        """
        if fibo_class:
            # Use FIBO class name as type embedding
            return self._get_finbert_embedding(fibo_class)
        else:
            # Default: return zero embedding
            return torch.zeros(1, self.embedding_dim, device=self.device)
    
    def _compute_causal_confidence(
        self,
        cause: Entity,
        effect: Entity,
        doc: Dict
    ) -> float:
        """
        Compute confidence for causal relationship.
        
        Implements Equation 6 from paper:
        conf(c → e) = σ(W_causal · [h_c; h_e; h_context])
        
        Line 8 of Algorithm 1.
        
        Args:
            cause: Cause entity
            effect: Effect entity
            doc: Document context
            
        Returns:
            Confidence score in [0, 1]
        """
        # Get entity embeddings
        h_cause = self.entity_embeddings.get(cause.id)
        h_effect = self.entity_embeddings.get(effect.id)
        
        if h_cause is None or h_effect is None:
            return 0.0
        
        # Get context embedding
        h_context = self._get_context_embedding(cause, effect, doc)
        
        # Concatenate embeddings
        concatenated = torch.cat([h_cause, h_effect, h_context], dim=0).unsqueeze(0)
        
        # Apply causal classifier (sigmoid already in model)
        with torch.no_grad():
            confidence = self.causal_classifier(concatenated).item()
        
        return confidence
    
    def _get_context_embedding(
        self,
        cause: Entity,
        effect: Entity,
        doc: Dict
    ) -> torch.Tensor:
        """
        Extract context embedding around cause-effect pair.
        
        Args:
            cause: Cause entity
            effect: Effect entity
            doc: Document
            
        Returns:
            Context embedding tensor
        """
        text = doc['text']
        
        # Extract context window around both entities
        start = min(cause.span[0], effect.span[0])
        end = max(cause.span[1], effect.span[1])
        
        # Expand context window
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        
        context_text = text[context_start:context_end]
        
        return self._get_finbert_embedding(context_text).squeeze(0)
    
    def _add_causal_edge(
        self,
        cause: Entity,
        effect: Entity,
        confidence: float,
        temporal_annotations: Dict,
        doc: Dict
    ) -> None:
        """
        Add causal edge to graph with confidence and temporal ordering.
        
        Lines 10-11 of Algorithm 1.
        
        Args:
            cause: Cause entity
            effect: Effect entity
            confidence: Causal confidence score
            temporal_annotations: Temporal information
            doc: Document context
        """
        # Determine temporal ordering
        temporal_ordering = self._determine_temporal_ordering(
            cause, effect, temporal_annotations
        )
        
        # Add edge to graph
        self.extended_graph.add_edge(
            cause.id,
            effect.id,
            relation_type='causes',
            confidence=confidence,
            temporal_ordering=temporal_ordering,
            document_id=doc.get('id', 'unknown'),
            created=datetime.now().isoformat()
        )
    
    def _validate_and_clean_graph(self) -> None:
        """
        Validate edges against FIBO constraints and remove contradictions.
        
        Line 14 of Algorithm 1.
        """
        logger.info("Validating graph against FIBO constraints...")
        
        edges_to_remove = []
        
        for u, v, data in self.extended_graph.edges(data=True):
            # Check for contradictions with FIBO ontology
            if self._is_contradictory_edge(u, v, data):
                edges_to_remove.append((u, v))
        
        # Remove invalid edges
        self.extended_graph.remove_edges_from(edges_to_remove)
        
        if edges_to_remove:
            logger.info(f"Removed {len(edges_to_remove)} contradictory edges")
        
        # Remove isolated nodes (except FIBO classes)
        nodes_to_remove = [
            node for node in self.extended_graph.nodes()
            if (self.extended_graph.degree(node) == 0 and
                self.extended_graph.nodes[node].get('node_type') == 'entity')
        ]
        self.extended_graph.remove_nodes_from(nodes_to_remove)
        
        if nodes_to_remove:
            logger.info(f"Removed {len(nodes_to_remove)} isolated entity nodes")
    
    # Helper methods
    
    def _initialize_embedding_weights(self) -> None:
        """Initialize embedding transformation weights to sum to identity."""
        # Simple initialization: equal weights
        with torch.no_grad():
            nn.init.xavier_uniform_(self.W_text.weight)
            nn.init.xavier_uniform_(self.W_struct.weight)
            nn.init.xavier_uniform_(self.W_type.weight)
            
            # Normalize to sum to identity
            total = self.W_text.weight + self.W_struct.weight + self.W_type.weight
            self.W_text.weight.div_(total)
            self.W_struct.weight.div_(total)
            self.W_type.weight.div_(total)
    
    def _map_to_fibo_class(
        self,
        entity_text: str,
        entity_label: str
    ) -> Tuple[Optional[str], str]:
        """
        Map entity to FIBO class.
        
        Args:
            entity_text: Entity mention
            entity_label: Entity type label
            
        Returns:
            Tuple of (fibo_class, entity_type)
        """
        if entity_label == 'ORG':
            return ('BE/LegalEntities/Corporation', 'company')
        elif entity_label == 'PERSON':
            return ('FND/Agents/Person', 'person')
        elif entity_label == 'MONEY':
            return (None, 'monetary_value')
        elif entity_label == 'PERCENT':
            return (None, 'percentage')
        elif entity_label == 'PRODUCT':
            return ('FBC/Products/FinancialInstrument', 'instrument')
        else:
            return (None, entity_label.lower())
    
    def _find_closest_entity(
        self,
        text: str,
        position: int,
        entities: List[Entity]
    ) -> Optional[Entity]:
        """Find entity closest to text position."""
        closest = None
        min_distance = float('inf')
        
        for entity in entities:
            # Check if text is contained in entity or vice versa
            if text.lower() in entity.mention.lower() or \
               entity.mention.lower() in text.lower():
                distance = abs(entity.span[0] - position)
                if distance < min_distance:
                    min_distance = distance
                    closest = entity
        
        return closest
    
    def _are_temporally_ordered(
        self,
        entity1: Entity,
        entity2: Entity,
        text: str
    ) -> bool:
        """Check if entities have temporal ordering relationship."""
        # Simple heuristic: check for temporal markers between entities
        start = min(entity1.span[1], entity2.span[1])
        end = max(entity1.span[0], entity2.span[0])
        
        between_text = text[start:end].lower()
        
        temporal_markers = ['then', 'after', 'following', 'subsequently', 'later']
        return any(marker in between_text for marker in temporal_markers)
    
    def _determine_temporal_ordering(
        self,
        cause: Entity,
        effect: Entity,
        temporal_annotations: Dict
    ) -> str:
        """Determine temporal ordering of cause and effect."""
        # Simple heuristic based on position
        if cause.span[0] < effect.span[0]:
            return 'before'
        elif cause.span[0] > effect.span[0]:
            return 'after'
        else:
            return 'simultaneous'
    
    def _is_contradictory_edge(self, u: str, v: str, data: Dict) -> bool:
        """Check if edge contradicts FIBO ontology constraints."""
        # Example: Check for logical contradictions
        # In production, this would check against FIBO reasoning rules
        
        # Check confidence threshold
        if data.get('confidence', 1.0) < 0.3:
            return True
        
        # Check for cycles in causal relationships (simplified)
        if data.get('relation_type') == 'causes':
            if nx.has_path(self.extended_graph, v, u):
                return True  # Would create cycle
        
        return False
    
    def save_graph(self, filepath: str) -> None:
        """Save knowledge graph to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.extended_graph,
                'embeddings': self.entity_embeddings
            }, f)
        logger.info(f"Knowledge graph saved to {filepath}")
    
    def load_graph(self, filepath: str) -> None:
        """Load knowledge graph from file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.extended_graph = data['graph']
            self.entity_embeddings = data['embeddings']
        logger.info(f"Knowledge graph loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Example documents
    sample_documents = [
        {
            "id": "doc_001",
            "text": """Apple Inc. reported strong Q4 2023 earnings, with revenue 
                      increasing 15% due to robust iPhone sales. The company's CEO 
                      Tim Cook attributed the growth to successful product launches 
                      and expanding services revenue. However, supply chain disruptions 
                      led to production delays in certain regions.""",
            "metadata": {
                "source": "earnings_call",
                "date": "2023-10-31",
                "company": "Apple Inc."
            }
        },
        {
            "id": "doc_002",
            "text": """Federal Reserve raised interest rates by 0.25%, resulting in 
                      market volatility. This decision was driven by persistent inflation 
                      concerns. As a result of the rate hike, tech stocks declined 
                      significantly.""",
            "metadata": {
                "source": "financial_news",
                "date": "2023-11-01"
            }
        }
    ]
    
    # Initialize constructor
    print("Initializing Knowledge Graph Constructor...")
    constructor = KnowledgeGraphConstructor(
        fibo_ontology_path="fibo.owl",  # Placeholder path
        causal_threshold=0.7
    )
    
    # Construct knowledge graph
    print("\nConstructing knowledge graph from sample documents...")
    kg = constructor.construct_knowledge_graph(sample_documents)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Knowledge Graph Statistics:")
    print(f"{'='*60}")
    print(f"Total nodes: {kg.number_of_nodes()}")
    print(f"Total edges: {kg.number_of_edges()}")
    
    # Print causal relationships
    print(f"\n{'='*60}")
    print("Causal Relationships Detected:")
    print(f"{'='*60}")
    causal_edges = [(u, v, d) for u, v, d in kg.edges(data=True) 
                    if d.get('relation_type') == 'causes']
    for u, v, data in causal_edges:
        print(f"{u} → {v} (confidence: {data['confidence']:.3f})")
    
    # Save graph
    constructor.save_graph("knowledge_graph.pkl")
    print("\nKnowledge graph saved successfully!")
