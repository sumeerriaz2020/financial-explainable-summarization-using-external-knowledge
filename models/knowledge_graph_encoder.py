"""
Knowledge Graph Encoder
========================

Graph Neural Network encoder for knowledge graph structure using
Graph Attention Networks (GAT) to encode FIBO ontology relationships.

This module implements the KG encoding component from Section 3.3
that captures entity relationships and hierarchical structures.

Authors: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Network (GAT) layer.
    
    Implements attention mechanism for graph-structured data, allowing
    nodes to attend to their neighbors with learned attention weights.
    
    Reference: Veličković et al. "Graph Attention Networks" (ICLR 2018)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.15,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Initialize GAT layer
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate multi-head outputs
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation for node features
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # Activation and regularization
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of GAT layer
        
        Args:
            h: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            edge_weights: Optional edge weights (num_nodes, num_nodes)
            
        Returns:
            Updated node features (num_nodes, out_features)
        """
        num_nodes = h.size(0)
        
        # Linear transformation: Wh
        Wh = torch.matmul(h, self.W)  # (num_nodes, out_features)
        
        # Attention mechanism
        # Self-attention: concatenate node pairs
        a_input = self._prepare_attentional_mechanism_input(Wh)
        
        # Compute attention coefficients
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        e = e.view(num_nodes, num_nodes)
        
        # Mask attention by adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply edge weights if provided
        if edge_weights is not None:
            attention = attention * edge_weights
        
        # Softmax normalization
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)
        
        # Apply non-linearity if concatenating
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for attention mechanism by concatenating node pairs
        
        Args:
            Wh: Transformed features (num_nodes, out_features)
            
        Returns:
            Concatenated pairs (num_nodes * num_nodes, 2 * out_features)
        """
        num_nodes = Wh.size(0)
        
        # Repeat for all pairs
        # Wh_repeated_in_chunks: [h1, h1, ..., h1, h2, h2, ..., h2, ...]
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=0)
        
        # Wh_repeated_alternating: [h1, h2, ..., hn, h1, h2, ..., hn, ...]
        Wh_repeated_alternating = Wh.repeat(num_nodes, 1)
        
        # Concatenate: [h_i || h_j] for all i, j
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating],
            dim=1
        )
        
        return all_combinations_matrix


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer.
    
    Applies multiple attention heads in parallel to capture different
    aspects of graph structure and relationships.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.15,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        Initialize multi-head GAT layer
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate head outputs
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.concat = concat
        
        # Create attention heads
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(
                in_features,
                out_features,
                dropout=dropout,
                alpha=alpha,
                concat=concat
            )
            for _ in range(num_heads)
        ])
        
        # Output dimension
        if concat:
            self.out_dim = out_features * num_heads
        else:
            self.out_dim = out_features
    
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with multiple attention heads
        
        Args:
            h: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            edge_weights: Optional edge weights
            
        Returns:
            Updated features (num_nodes, out_features * num_heads) if concat
            else (num_nodes, out_features)
        """
        # Apply each attention head
        head_outputs = [
            head(h, adj, edge_weights)
            for head in self.attention_heads
        ]
        
        # Concatenate or average
        if self.concat:
            return torch.cat(head_outputs, dim=-1)
        else:
            return torch.stack(head_outputs, dim=0).mean(dim=0)


class KnowledgeGraphEncoder(nn.Module):
    """
    Complete Knowledge Graph Encoder using stacked GAT layers.
    
    Encodes knowledge graph structure to capture entity relationships,
    hierarchical structures, and multi-hop reasoning paths in FIBO ontology.
    
    Implements the GraphEncoder_θ(G_local) component from Algorithm 2.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 1024,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.15,
        residual: bool = True,
        use_edge_weights: bool = True
    ):
        """
        Initialize KG encoder
        
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads per layer
            dropout: Dropout probability
            residual: Whether to use residual connections
            use_edge_weights: Whether to utilize edge weights
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.residual = residual
        self.use_edge_weights = use_edge_weights
        
        # Build GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            MultiHeadGATLayer(
                input_dim,
                hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                concat=True
            )
        )
        self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
        
        if residual and input_dim != hidden_dim * num_heads:
            self.residual_projections.append(
                nn.Linear(input_dim, hidden_dim * num_heads)
            )
        else:
            self.residual_projections.append(None)
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                MultiHeadGATLayer(
                    hidden_dim * num_heads,
                    hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
            self.residual_projections.append(None)  # Same dim
        
        # Final layer (average heads instead of concat)
        if num_layers > 1:
            self.gat_layers.append(
                MultiHeadGATLayer(
                    hidden_dim * num_heads,
                    output_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=False  # Average for final layer
                )
            )
            self.norms.append(nn.LayerNorm(output_dim))
            
            if residual and hidden_dim * num_heads != output_dim:
                self.residual_projections.append(
                    nn.Linear(hidden_dim * num_heads, output_dim)
                )
            else:
                self.residual_projections.append(None)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"KG Encoder initialized: {input_dim}D → {output_dim}D "
                   f"({num_layers} layers, {num_heads} heads)")
    
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode knowledge graph
        
        Args:
            node_features: (num_nodes, input_dim) node features
            adjacency: (num_nodes, num_nodes) adjacency matrix
            edge_weights: Optional (num_nodes, num_nodes) edge weights
            
        Returns:
            Encoded features (num_nodes, output_dim)
        """
        h = node_features
        
        # Ensure adjacency is binary or weighted
        if adjacency.dtype == torch.bool:
            adjacency = adjacency.float()
        
        # Add self-loops if not present
        eye = torch.eye(
            adjacency.size(0),
            device=adjacency.device,
            dtype=adjacency.dtype
        )
        adjacency = adjacency + eye
        
        # Process edge weights
        if self.use_edge_weights and edge_weights is not None:
            if edge_weights.dtype == torch.bool:
                edge_weights = edge_weights.float()
        else:
            edge_weights = None
        
        # Apply GAT layers
        for i, (gat_layer, norm, res_proj) in enumerate(
            zip(self.gat_layers, self.norms, self.residual_projections)
        ):
            h_in = h
            
            # GAT layer
            h = gat_layer(h, adjacency, edge_weights)
            
            # Residual connection
            if self.residual:
                if res_proj is not None:
                    h_in = res_proj(h_in)
                h = h + h_in
            
            # Layer normalization
            h = norm(h)
            
            # Dropout (except last layer)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)
        
        return h
    
    def get_attention_weights(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        layer_idx: int = 0,
        head_idx: int = 0
    ) -> torch.Tensor:
        """
        Extract attention weights from specific layer and head
        
        Args:
            node_features: Node features
            adjacency: Adjacency matrix
            layer_idx: Layer index to extract from
            head_idx: Attention head index
            
        Returns:
            Attention weights (num_nodes, num_nodes)
        """
        # This is a simplified version - in production, would need
        # to modify forward pass to return attention weights
        
        with torch.no_grad():
            h = node_features
            
            for i, gat_layer in enumerate(self.gat_layers):
                if i == layer_idx:
                    # Get attention from specific head
                    head = gat_layer.attention_heads[head_idx]
                    
                    # Forward through this head to get attention
                    Wh = torch.matmul(h, head.W)
                    a_input = head._prepare_attentional_mechanism_input(Wh)
                    e = head.leakyrelu(torch.matmul(a_input, head.a).squeeze(-1))
                    e = e.view(adjacency.size(0), adjacency.size(0))
                    
                    zero_vec = -9e15 * torch.ones_like(e)
                    attention = torch.where(adjacency > 0, e, zero_vec)
                    attention = F.softmax(attention, dim=1)
                    
                    return attention
                
                # Continue forward pass
                h = gat_layer(h, adjacency)
        
        return None


class HierarchicalKGEncoder(nn.Module):
    """
    Hierarchical KG encoder for multi-level ontology structure.
    
    Processes knowledge graph at multiple levels of abstraction,
    useful for hierarchical ontologies like FIBO.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        level_dims: List[int] = [512, 768],
        num_heads: int = 4,
        dropout: float = 0.15
    ):
        """
        Initialize hierarchical KG encoder
        
        Args:
            input_dim: Input feature dimension
            level_dims: Output dimensions for each hierarchy level
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_levels = len(level_dims)
        
        # Encoder for each level
        self.level_encoders = nn.ModuleList([
            KnowledgeGraphEncoder(
                input_dim=input_dim if i == 0 else level_dims[i-1],
                hidden_dim=level_dims[i] // 2,
                output_dim=level_dims[i],
                num_layers=2,
                num_heads=num_heads,
                dropout=dropout
            )
            for i in range(self.num_levels)
        ])
        
        # Pooling between levels
        self.pooling = nn.ModuleList([
            nn.Linear(level_dims[i], level_dims[i])
            for i in range(self.num_levels - 1)
        ])
        
        logger.info(f"Hierarchical KG Encoder initialized: "
                   f"{self.num_levels} levels")
    
    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        hierarchy_masks: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Encode KG at multiple hierarchy levels
        
        Args:
            node_features: Initial node features
            adjacency: Adjacency matrix
            hierarchy_masks: Optional masks for each level
            
        Returns:
            List of encoded features for each level
        """
        level_outputs = []
        h = node_features
        
        for i, encoder in enumerate(self.level_encoders):
            # Encode at this level
            h = encoder(h, adjacency)
            level_outputs.append(h)
            
            # Pool for next level (if not last)
            if i < self.num_levels - 1:
                # Apply hierarchy mask if provided
                if hierarchy_masks and i < len(hierarchy_masks):
                    mask = hierarchy_masks[i]
                    h = h * mask.unsqueeze(-1)
                
                # Aggregate (mean pooling with projection)
                h = self.pooling[i](h)
        
        return level_outputs


# Example usage and testing
if __name__ == "__main__":
    print("Knowledge Graph Encoder")
    print("=" * 70)
    
    # Test single GAT layer
    print("\nTesting single GAT layer...")
    gat = GraphAttentionLayer(in_features=64, out_features=32)
    
    num_nodes = 10
    h = torch.randn(num_nodes, 64)
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj > 0.7).float()  # Sparse adjacency
    adj.fill_diagonal_(1)  # Add self-loops
    
    out = gat(h, adj)
    print(f"Input shape: {h.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test multi-head GAT
    print("\nTesting multi-head GAT layer...")
    multi_gat = MultiHeadGATLayer(
        in_features=64,
        out_features=32,
        num_heads=4,
        concat=True
    )
    
    out_multi = multi_gat(h, adj)
    print(f"Multi-head output shape: {out_multi.shape}")
    
    # Test complete KG encoder
    print("\nTesting complete KG encoder...")
    kg_encoder = KnowledgeGraphEncoder(
        input_dim=768,
        hidden_dim=512,
        output_dim=1024,
        num_layers=3,
        num_heads=8,
        dropout=0.15
    )
    
    num_nodes = 50
    node_features = torch.randn(num_nodes, 768)
    adjacency = torch.rand(num_nodes, num_nodes)
    adjacency = (adjacency > 0.9).float()
    
    encoded = kg_encoder(node_features, adjacency)
    
    print(f"Input features: {node_features.shape}")
    print(f"Adjacency: {adjacency.shape}, density: {adjacency.mean():.3f}")
    print(f"Encoded features: {encoded.shape}")
    
    # Test hierarchical encoder
    print("\nTesting hierarchical KG encoder...")
    hier_encoder = HierarchicalKGEncoder(
        input_dim=768,
        level_dims=[512, 768],
        num_heads=4
    )
    
    level_outputs = hier_encoder(node_features, adjacency)
    
    print(f"Number of hierarchy levels: {len(level_outputs)}")
    for i, out in enumerate(level_outputs):
        print(f"  Level {i+1} output: {out.shape}")
    
    # Test attention weight extraction
    print("\nExtracting attention weights...")
    attention = kg_encoder.get_attention_weights(
        node_features, adjacency, layer_idx=0, head_idx=0
    )
    if attention is not None:
        print(f"Attention weights: {attention.shape}")
        print(f"Attention sum per node (should be ~1): {attention.sum(dim=1)[:5]}")
    
    print(f"\n{'=' * 70}")
    print("KG encoder architecture ready!")
    print(f"Parameters: {sum(p.numel() for p in kg_encoder.parameters()):,}")
    print(f"{'=' * 70}")
