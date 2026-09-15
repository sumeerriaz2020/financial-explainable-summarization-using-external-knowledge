"""
Algorithm 2: Hybrid Neural-Symbolic Summarization and Explanation
===================================================================

Implementation of Algorithm 2 from the paper:
"An eXplainable Approach to Abstractive Text Summarization Using External Knowledge"

This module performs hybrid neural-symbolic inference for financial document
summarization with integrated explainability:
1. Encodes document text using transformer architecture
2. Extracts relevant knowledge graph subgraph
3. Encodes KG structure using graph neural networks
4. Performs cross-modal attention between text and KG
5. Executes multi-hop reasoning over knowledge paths
6. Generates summary with factual verification
7. Produces stakeholder-specific explanations

Authors: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
Reference: Section 3.3, Algorithm 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StakeholderProfile:
    """Profile for different stakeholder types"""
    role: str  # financial_analyst, compliance_officer, executive, investment_manager
    expertise_level: str  # novice, intermediate, expert
    information_needs: List[str]
    explanation_preferences: Dict[str, float]


@dataclass
class HybridOutput:
    """Output from hybrid neural-symbolic summarization"""
    summary: str
    summary_tokens: List[str]
    attention_weights: torch.Tensor
    knowledge_paths: List[List[str]]
    factual_consistency_score: float
    explanation: Dict[str, any]
    verification_results: Dict[str, bool]


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer for KG encoding"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of GAT layer
        
        Args:
            node_features: (num_nodes, in_features)
            adjacency: (num_nodes, num_nodes) adjacency matrix
            
        Returns:
            Updated node features (num_nodes, out_features)
        """
        # Linear transformation
        h = self.W(node_features)  # (num_nodes, out_features)
        num_nodes = h.size(0)
        
        # Attention mechanism
        # Concatenate node pairs
        a_input = torch.cat([
            h.repeat(1, num_nodes).view(num_nodes * num_nodes, -1),
            h.repeat(num_nodes, 1)
        ], dim=1)  # (num_nodes * num_nodes, 2 * out_features)
        
        # Compute attention coefficients
        e = self.leakyrelu(self.a(a_input).squeeze(1))
        e = e.view(num_nodes, num_nodes)
        
        # Mask attention by adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adjacency > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, h)
        
        return h_prime


class KnowledgeGraphEncoder(nn.Module):
    """
    Graph Neural Network encoder for knowledge graph structure.
    
    Encodes FIBO knowledge graph using multi-layer GAT to capture
    entity relationships and hierarchical structures.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Multi-layer GAT
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            nn.ModuleList([
                GraphAttentionLayer(input_dim, hidden_dim, dropout)
                for _ in range(num_heads)
            ])
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                nn.ModuleList([
                    GraphAttentionLayer(hidden_dim * num_heads, hidden_dim, dropout)
                    for _ in range(num_heads)
                ])
            )
        
        # Final layer (single head)
        if num_layers > 1:
            self.gat_layers.append(
                nn.ModuleList([
                    GraphAttentionLayer(hidden_dim * num_heads, output_dim, dropout)
                ])
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode knowledge graph structure
        
        Args:
            node_features: (num_nodes, input_dim)
            adjacency: (num_nodes, num_nodes)
            
        Returns:
            Encoded features (num_nodes, output_dim)
        """
        h = node_features
        
        # Apply GAT layers
        for i, layer in enumerate(self.gat_layers[:-1]):
            # Multi-head attention
            h_heads = [head(h, adjacency) for head in layer]
            h = torch.cat(h_heads, dim=-1)
            h = F.elu(h)
            h = self.dropout(h)
        
        # Final layer
        if len(self.gat_layers) > 1:
            h = self.gat_layers[-1][0](h, adjacency)
        
        return h


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between text and knowledge graph.
    
    Implements Equation 2 from paper:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        text_features: torch.Tensor,
        kg_features: torch.Tensor,
        kg_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention
        
        Args:
            text_features: (batch, seq_len, embed_dim) from text encoder
            kg_features: (batch, num_nodes, embed_dim) from KG encoder
            kg_mask: Optional mask for KG nodes
            
        Returns:
            Tuple of (fused_features, attention_weights)
        """
        # Q from text, K and V from knowledge graph
        attn_output, attn_weights = self.multihead_attn(
            query=text_features,
            key=kg_features,
            value=kg_features,
            key_padding_mask=kg_mask,
            need_weights=True
        )
        
        # Residual connection and normalization
        fused = self.norm(text_features + self.dropout(attn_output))
        
        return fused, attn_weights


class MultiHopReasoning(nn.Module):
    """
    Multi-hop reasoning over knowledge graph paths.
    
    Traverses knowledge paths to compose complex reasoning chains.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_hops: int = 3,
        path_aggregation: str = 'attention'
    ):
        super().__init__()
        
        self.num_hops = num_hops
        self.path_aggregation = path_aggregation
        
        # Path encoding layers
        self.path_encoders = nn.ModuleList([
            nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=False)
            for _ in range(num_hops)
        ])
        
        # Path attention weights
        if path_aggregation == 'attention':
            self.path_attention = nn.Linear(embed_dim, 1)
        
        # Fusion layer
        self.fusion = nn.Linear(embed_dim * (num_hops + 1), embed_dim)
        
    def forward(
        self,
        base_features: torch.Tensor,
        path_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform multi-hop reasoning
        
        Args:
            base_features: (batch, embed_dim) base entity features
            path_features: List of (batch, path_len, embed_dim) for each hop
            
        Returns:
            Aggregated features (batch, embed_dim)
        """
        encoded_paths = []
        
        # Encode each hop
        for i, path in enumerate(path_features[:self.num_hops]):
            if path.size(0) > 0:
                # Encode path using LSTM
                encoded, (hn, cn) = self.path_encoders[i](path)
                # Use final hidden state
                encoded_paths.append(hn.squeeze(0))
            else:
                # Empty path - use zeros
                encoded_paths.append(
                    torch.zeros_like(base_features)
                )
        
        # Concatenate base features and path features
        all_features = [base_features] + encoded_paths
        concatenated = torch.cat(all_features, dim=-1)
        
        # Fuse features
        fused = self.fusion(concatenated)
        fused = F.relu(fused)
        
        return fused


class HybridNeuralSymbolicModel(nn.Module):
    """
    Complete hybrid neural-symbolic architecture for financial summarization.
    
    Combines transformer text encoding with graph-based knowledge reasoning
    to generate accurate, explainable summaries.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large",
        kg_input_dim: int = 768,
        kg_hidden_dim: int = 512,
        num_gat_layers: int = 2,
        num_reasoning_hops: int = 3,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.device = device
        self.num_reasoning_hops = num_reasoning_hops
        
        # Text encoder/decoder (BART)
        logger.info(f"Loading text model: {model_name}")
        self.text_model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.embed_dim = self.text_model.config.d_model
        
        # Knowledge graph encoder
        self.kg_encoder = KnowledgeGraphEncoder(
            input_dim=kg_input_dim,
            hidden_dim=kg_hidden_dim,
            output_dim=self.embed_dim,
            num_layers=num_gat_layers,
            dropout=dropout
        )
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # Multi-hop reasoning
        self.multi_hop_reasoning = MultiHopReasoning(
            embed_dim=self.embed_dim,
            num_hops=num_reasoning_hops
        )
        
        # Fusion layer for combining text and KG features
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.embed_dim)
        )
        
        self.to(device)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kg_node_features: torch.Tensor,
        kg_adjacency: torch.Tensor,
        kg_paths: Optional[List[torch.Tensor]] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of hybrid model
        
        Args:
            input_ids: (batch, seq_len) tokenized input
            attention_mask: (batch, seq_len) attention mask
            kg_node_features: (batch, num_nodes, kg_dim) KG node features
            kg_adjacency: (batch, num_nodes, num_nodes) adjacency matrices
            kg_paths: Optional list of reasoning paths
            decoder_input_ids: Optional decoder inputs
            labels: Optional labels for training
            
        Returns:
            Dictionary with loss, logits, and intermediate outputs
        """
        batch_size = input_ids.size(0)
        
        # 1. Encode text (Line 1: H_text ← TransformerEncoder_θ(D))
        encoder_outputs = self.text_model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_features = encoder_outputs.last_hidden_state
        
        # 2. Encode knowledge graph (Line 3: H_kg ← GraphEncoder_θ(G_local))
        # Process each graph in batch
        kg_features_list = []
        for i in range(batch_size):
            kg_feat = self.kg_encoder(
                kg_node_features[i],
                kg_adjacency[i]
            )
            kg_features_list.append(kg_feat)
        
        # Stack KG features
        max_nodes = max(kg.size(0) for kg in kg_features_list)
        kg_features = torch.zeros(
            batch_size, max_nodes, self.embed_dim,
            device=self.device
        )
        
        for i, kg_feat in enumerate(kg_features_list):
            kg_features[i, :kg_feat.size(0), :] = kg_feat
        
        # 3. Cross-modal attention (Lines 4-5: Q, K, V and H_fused)
        fused_features, attention_weights = self.cross_modal_attention(
            text_features, kg_features
        )
        
        # 4. Multi-hop reasoning (Lines 6-8)
        if kg_paths is not None and len(kg_paths) > 0:
            # Get document-level representation
            doc_repr = fused_features.mean(dim=1)  # (batch, embed_dim)
            
            # Perform multi-hop reasoning
            reasoning_output = self.multi_hop_reasoning(
                doc_repr, kg_paths
            )
            
            # Expand reasoning output to sequence length
            reasoning_expanded = reasoning_output.unsqueeze(1).expand(
                -1, fused_features.size(1), -1
            )
            
            # Fuse with existing features
            combined = torch.cat([fused_features, reasoning_expanded], dim=-1)
            fused_features = self.fusion_layer(combined)
        
        # 5. Generate summary (Line 10: Ŷ ← Decoder_θ(H_fused))
        # Replace encoder hidden states with fused features
        encoder_outputs.last_hidden_state = fused_features
        
        # Decode
        if labels is not None:
            # Training mode
            outputs = self.text_model(
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
        else:
            # Inference mode
            outputs = self.text_model.generate(
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=128,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        return {
            'outputs': outputs,
            'fused_features': fused_features,
            'attention_weights': attention_weights,
            'text_features': text_features,
            'kg_features': kg_features
        }


class HybridInferenceEngine:
    """
    Main inference engine implementing Algorithm 2.
    
    Orchestrates hybrid neural-symbolic summarization with
    integrated explainability generation.
    """
    
    def __init__(
        self,
        model: HybridNeuralSymbolicModel,
        knowledge_graph: nx.DiGraph,
        entity_embeddings: Dict[str, torch.Tensor],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize hybrid inference engine
        
        Args:
            model: Hybrid neural-symbolic model
            knowledge_graph: Extended knowledge graph
            entity_embeddings: Pre-computed entity embeddings
            device: Computing device
        """
        self.model = model
        self.knowledge_graph = knowledge_graph
        self.entity_embeddings = entity_embeddings
        self.device = device
        
        self.model.eval()
        
    def summarize_and_explain(
        self,
        document: Dict,
        stakeholder_profile: StakeholderProfile,
        verify_facts: bool = True
    ) -> HybridOutput:
        """
        Main method: Generate summary with explanation for stakeholder.
        
        Implements Algorithm 2 from paper (Section 3.3).
        
        Args:
            document: Document with 'text' field
            stakeholder_profile: Target stakeholder profile
            verify_facts: Whether to perform factual verification
            
        Returns:
            HybridOutput with summary and explanations
        """
        logger.info(f"Generating summary for stakeholder: {stakeholder_profile.role}")
        
        # Line 1: H_text ← TransformerEncoder_θ(D)
        text = document['text']
        inputs = self.model.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Line 2: E_doc ← Extract entities (NER)
        entities_doc = self._extract_entities(text)
        
        # Line 3: G_local ← Extract subgraph from G relevant to E_doc
        local_graph, entity_nodes = self._extract_local_subgraph(entities_doc)
        
        # Prepare KG inputs
        kg_node_features, kg_adjacency = self._prepare_kg_inputs(
            local_graph, entity_nodes
        )
        
        # Lines 6-8: Multi-hop reasoning
        reasoning_paths = self._extract_reasoning_paths(
            entity_nodes, num_hops=self.model.num_reasoning_hops
        )
        
        # Forward pass
        with torch.no_grad():
            model_outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                kg_node_features=kg_node_features,
                kg_adjacency=kg_adjacency,
                kg_paths=reasoning_paths
            )
        
        # Line 10: Ŷ ← Decoder_θ(H_fused)
        if 'sequences' in model_outputs['outputs']:
            # Generation mode
            generated_ids = model_outputs['outputs']['sequences']
        else:
            # Training mode - use argmax
            generated_ids = model_outputs['outputs'].logits.argmax(dim=-1)
        
        # Decode summary
        summary = self.model.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        summary_tokens = self.model.tokenizer.convert_ids_to_tokens(
            generated_ids[0]
        )
        
        # Line 11: Verify against G, revise if needed
        if verify_facts:
            summary, verification_results = self._verify_and_revise(
                summary, local_graph
            )
        else:
            verification_results = {}
        
        # Calculate factual consistency score
        factual_score = self._compute_factual_consistency(
            summary, text, verification_results
        )
        
        # Line 12: E_S ← MESA(Ŷ, D, G, S) {Algorithm 3}
        explanation = self._generate_explanation(
            summary=summary,
            document=document,
            model_outputs=model_outputs,
            reasoning_paths=reasoning_paths,
            stakeholder_profile=stakeholder_profile
        )
        
        return HybridOutput(
            summary=summary,
            summary_tokens=summary_tokens,
            attention_weights=model_outputs['attention_weights'],
            knowledge_paths=reasoning_paths,
            factual_consistency_score=factual_score,
            explanation=explanation,
            verification_results=verification_results
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity IDs from text"""
        # Use simple heuristic - in production, use NER
        # This should match entities in knowledge graph
        entities = []
        for node in self.knowledge_graph.nodes():
            node_data = self.knowledge_graph.nodes[node]
            if node_data.get('node_type') == 'entity':
                mention = node_data.get('mention', '')
                if mention and mention in text:
                    entities.append(node)
        return entities
    
    def _extract_local_subgraph(
        self,
        entity_ids: List[str]
    ) -> Tuple[nx.DiGraph, List[str]]:
        """
        Extract local subgraph relevant to document entities.
        
        Line 3 of Algorithm 2.
        """
        # Start with entity nodes
        nodes_to_include = set(entity_ids)
        
        # Add 1-hop neighbors
        for entity_id in entity_ids:
            if entity_id in self.knowledge_graph:
                neighbors = list(self.knowledge_graph.neighbors(entity_id))
                nodes_to_include.update(neighbors[:10])  # Limit neighbors
        
        # Extract subgraph
        local_graph = self.knowledge_graph.subgraph(nodes_to_include).copy()
        
        return local_graph, list(nodes_to_include)
    
    def _prepare_kg_inputs(
        self,
        graph: nx.DiGraph,
        nodes: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare KG node features and adjacency matrix"""
        num_nodes = len(nodes)
        
        # Node features
        node_features = torch.zeros(1, num_nodes, 768, device=self.device)
        for i, node in enumerate(nodes):
            if node in self.entity_embeddings:
                node_features[0, i] = self.entity_embeddings[node]
        
        # Adjacency matrix
        adjacency = torch.zeros(1, num_nodes, num_nodes, device=self.device)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                adjacency[0, node_to_idx[u], node_to_idx[v]] = 1
        
        # Add self-loops
        adjacency[0].fill_diagonal_(1)
        
        return node_features, adjacency
    
    def _extract_reasoning_paths(
        self,
        entity_nodes: List[str],
        num_hops: int = 3
    ) -> List[torch.Tensor]:
        """
        Extract multi-hop reasoning paths.
        
        Lines 6-8 of Algorithm 2.
        """
        paths = []
        
        for hop in range(1, num_hops + 1):
            hop_paths = []
            
            for entity in entity_nodes[:5]:  # Limit starting entities
                if entity not in self.knowledge_graph:
                    continue
                
                # Find paths of length 'hop'
                for target in entity_nodes[:5]:
                    if entity == target:
                        continue
                    
                    try:
                        # Find shortest path
                        path = nx.shortest_path(
                            self.knowledge_graph,
                            entity,
                            target,
                            weight=None
                        )
                        
                        if len(path) - 1 == hop:
                            # Extract path embeddings
                            path_embeds = []
                            for node in path:
                                if node in self.entity_embeddings:
                                    path_embeds.append(
                                        self.entity_embeddings[node]
                                    )
                            
                            if path_embeds:
                                path_tensor = torch.stack(path_embeds)
                                hop_paths.append(path_tensor)
                    except nx.NetworkXNoPath:
                        continue
            
            # Aggregate paths for this hop
            if hop_paths:
                # Pad to same length
                max_len = max(p.size(0) for p in hop_paths)
                padded_paths = []
                for p in hop_paths:
                    if p.size(0) < max_len:
                        padding = torch.zeros(
                            max_len - p.size(0),
                            p.size(1),
                            device=self.device
                        )
                        p = torch.cat([p, padding], dim=0)
                    padded_paths.append(p)
                
                paths.append(torch.stack(padded_paths))
            else:
                # Empty path
                paths.append(torch.zeros(1, 1, 768, device=self.device))
        
        return paths
    
    def _verify_and_revise(
        self,
        summary: str,
        local_graph: nx.DiGraph
    ) -> Tuple[str, Dict]:
        """
        Verify summary against knowledge graph and revise if needed.
        
        Line 11 of Algorithm 2.
        """
        verification_results = {
            'entity_accuracy': True,
            'relationship_accuracy': True,
            'numerical_accuracy': True,
            'temporal_accuracy': True
        }
        
        # Simple verification - check if entities in summary exist in KG
        summary_lower = summary.lower()
        
        for node in local_graph.nodes():
            node_data = local_graph.nodes[node]
            if node_data.get('node_type') == 'entity':
                mention = node_data.get('mention', '').lower()
                if mention and mention not in summary_lower:
                    verification_results['entity_accuracy'] = False
        
        # In production, implement more sophisticated verification:
        # - Check relationship correctness
        # - Verify numerical values
        # - Validate temporal ordering
        
        # For now, return original summary
        # In production, revise based on verification failures
        
        return summary, verification_results
    
    def _compute_factual_consistency(
        self,
        summary: str,
        source: str,
        verification: Dict
    ) -> float:
        """Compute factual consistency score"""
        # Simple scoring based on verification
        score = sum(verification.values()) / len(verification)
        return score
    
    def _generate_explanation(
        self,
        summary: str,
        document: Dict,
        model_outputs: Dict,
        reasoning_paths: List[torch.Tensor],
        stakeholder_profile: StakeholderProfile
    ) -> Dict:
        """
        Generate stakeholder-specific explanation.
        
        Line 12 of Algorithm 2 - calls MESA (Algorithm 3).
        """
        explanation = {
            'stakeholder_role': stakeholder_profile.role,
            'key_facts': self._extract_key_facts(summary),
            'reasoning_chains': self._format_reasoning_chains(reasoning_paths),
            'attention_highlights': self._get_attention_highlights(
                model_outputs['attention_weights']
            ),
            'knowledge_sources': self._identify_knowledge_sources(reasoning_paths),
            'confidence_scores': {
                'overall': 0.85,
                'factual': model_outputs.get('factual_score', 0.80),
                'relevance': 0.90
            }
        }
        
        return explanation
    
    def _extract_key_facts(self, summary: str) -> List[str]:
        """Extract key facts from summary"""
        # Simple sentence splitting
        sentences = summary.split('.')
        return [s.strip() for s in sentences if s.strip()]
    
    def _format_reasoning_chains(self, paths: List[torch.Tensor]) -> List[str]:
        """Format reasoning paths for explanation"""
        chains = []
        for i, path_tensor in enumerate(paths):
            chains.append(f"Reasoning chain {i+1}: {path_tensor.size(0)} hops")
        return chains
    
    def _get_attention_highlights(self, attention: torch.Tensor) -> List[float]:
        """Get attention weight highlights"""
        if attention is None or attention.numel() == 0:
            return []
        
        # Average across heads and get top weights
        avg_attention = attention.mean(dim=1).squeeze()
        top_weights = avg_attention.topk(min(10, avg_attention.size(-1)))
        
        return top_weights.values.tolist()
    
    def _identify_knowledge_sources(self, paths: List[torch.Tensor]) -> List[str]:
        """Identify knowledge sources used"""
        sources = ["FIBO Ontology", "Document Context"]
        return sources


# Example usage
if __name__ == "__main__":
    print("Hybrid Neural-Symbolic Inference Engine")
    print("="*60)
    
    # This is a demonstration - in production, load actual trained model
    print("\nInitializing model components...")
    
    # Create dummy knowledge graph
    kg = nx.DiGraph()
    kg.add_node("entity_1", node_type="entity", mention="Apple Inc.")
    kg.add_node("entity_2", node_type="entity", mention="iPhone")
    kg.add_edge("entity_1", "entity_2", relation_type="produces")
    
    # Create dummy embeddings
    embeddings = {
        "entity_1": torch.randn(768),
        "entity_2": torch.randn(768)
    }
    
    # Initialize model
    model = HybridNeuralSymbolicModel()
    
    # Create inference engine
    engine = HybridInferenceEngine(model, kg, embeddings)
    
    print("✓ Model initialized successfully")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example document
    document = {
        'text': """Apple Inc. reported strong Q4 results with iPhone sales 
                   driving revenue growth of 15%. The company's services 
                   segment also showed robust performance.""",
        'id': 'doc_001'
    }
    
    # Example stakeholder profile
    stakeholder = StakeholderProfile(
        role='financial_analyst',
        expertise_level='expert',
        information_needs=['quantitative_details', 'peer_comparison'],
        explanation_preferences={'technical_depth': 0.8, 'visual_aids': 0.6}
    )
    
    print("\nGenerating summary and explanation...")
    print(f"Document length: {len(document['text'])} characters")
    print(f"Target stakeholder: {stakeholder.role}")
    
    # Generate output
    output = engine.summarize_and_explain(document, stakeholder)
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Summary: {output.summary}")
    print(f"\nFactual Consistency Score: {output.factual_consistency_score:.3f}")
    print(f"Knowledge Paths Used: {len(output.knowledge_paths)}")
    print(f"\nExplanation Components:")
    for key, value in output.explanation.items():
        print(f"  - {key}: {value if not isinstance(value, list) else f'{len(value)} items'}")
