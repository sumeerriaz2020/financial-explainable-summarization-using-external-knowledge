"""
Cross-Modal Attention Mechanisms
=================================

Cross-modal attention mechanisms for integrating textual and knowledge graph
representations through attention-based fusion.

Implements Equation 2 from the paper:
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Authors: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between text and knowledge graph.
    
    Implements standard scaled dot-product attention (Equation 2)
    where queries come from one modality and keys/values from another.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.15,
        bias: bool = True
    ):
        """
        Initialize cross-modal attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for residual connection
        self.norm = nn.LayerNorm(embed_dim)
        
        self._reset_parameters()
        
        logger.info(f"Cross-modal attention initialized: "
                   f"{embed_dim}D, {num_heads} heads")
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-modal attention
        
        Implements Equation 2: Attention(Q, K, V) = softmax(QK^T / √d_k) V
        
        Args:
            query: (batch, tgt_len, embed_dim) query modality
            key: (batch, src_len, embed_dim) key modality
            value: (batch, src_len, embed_dim) value modality
            key_padding_mask: (batch, src_len) mask for padding
            attn_mask: (tgt_len, src_len) or (batch, tgt_len, src_len) attention mask
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of (attended_output, attention_weights)
            - attended_output: (batch, tgt_len, embed_dim)
            - attention_weights: (batch, num_heads, tgt_len, src_len) if need_weights
        """
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        # Project Q, K, V
        Q = self.q_proj(query)  # (batch, tgt_len, embed_dim)
        K = self.k_proj(key)    # (batch, src_len, embed_dim)
        V = self.v_proj(value)  # (batch, src_len, embed_dim)
        
        # Reshape for multi-head attention
        # (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention (Equation 2)
        # Compute attention scores: QK^T / √d_k
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        # (batch, num_heads, tgt_len, src_len)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # Broadcast to (batch, num_heads, tgt_len, src_len)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores + attn_mask
        
        # Apply key padding mask
        if key_padding_mask is not None:
            # Reshape to (batch, 1, 1, src_len)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(
                key_padding_mask,
                float('-inf')
            )
        
        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # (batch, num_heads, tgt_len, head_dim)
        
        # Reshape back to (batch, tgt_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, tgt_len, embed_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        attn_output = self.norm(query + self.dropout(attn_output))
        
        # Return attention weights if requested
        if need_weights:
            # Average over heads for interpretability
            attn_weights = attn_weights.mean(dim=1)  # (batch, tgt_len, src_len)
        else:
            attn_weights = None
        
        return attn_output, attn_weights


class BidirectionalCrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention.
    
    Applies attention in both directions: text→KG and KG→text,
    then combines the results for richer cross-modal fusion.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.15
    ):
        """
        Initialize bidirectional cross-modal attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Text→KG attention
        self.text_to_kg = CrossModalAttention(
            embed_dim, num_heads, dropout
        )
        
        # KG→Text attention
        self.kg_to_text = CrossModalAttention(
            embed_dim, num_heads, dropout
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        logger.info("Bidirectional cross-modal attention initialized")
    
    def forward(
        self,
        text_features: torch.Tensor,
        kg_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        kg_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply bidirectional cross-modal attention
        
        Args:
            text_features: (batch, text_len, embed_dim)
            kg_features: (batch, kg_len, embed_dim)
            text_mask: (batch, text_len) padding mask
            kg_mask: (batch, kg_len) padding mask
            
        Returns:
            Tuple of (fused_text, fused_kg, attention_dict)
        """
        # Text attending to KG
        text_to_kg_output, text_to_kg_weights = self.text_to_kg(
            query=text_features,
            key=kg_features,
            value=kg_features,
            key_padding_mask=kg_mask,
            need_weights=True
        )
        
        # KG attending to Text
        kg_to_text_output, kg_to_text_weights = self.kg_to_text(
            query=kg_features,
            key=text_features,
            value=text_features,
            key_padding_mask=text_mask,
            need_weights=True
        )
        
        # Fuse bidirectional outputs
        # For text: concatenate original + attended
        text_combined = torch.cat([text_features, text_to_kg_output], dim=-1)
        fused_text = self.fusion(text_combined)
        
        # For KG: concatenate original + attended
        kg_combined = torch.cat([kg_features, kg_to_text_output], dim=-1)
        fused_kg = self.fusion(kg_combined)
        
        # Collect attention weights
        attention_dict = {
            'text_to_kg': text_to_kg_weights,
            'kg_to_text': kg_to_text_weights
        }
        
        return fused_text, fused_kg, attention_dict


class GatedCrossModalFusion(nn.Module):
    """
    Gated fusion mechanism for cross-modal attention.
    
    Uses learned gates to dynamically weight the contribution of
    each modality, allowing the model to adaptively focus on
    text or KG based on input characteristics.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.15
    ):
        """
        Initialize gated cross-modal fusion
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            embed_dim, num_heads, dropout
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        logger.info("Gated cross-modal fusion initialized")
    
    def forward(
        self,
        query_features: torch.Tensor,
        key_value_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gated cross-modal fusion
        
        Args:
            query_features: (batch, query_len, embed_dim)
            key_value_features: (batch, kv_len, embed_dim)
            key_padding_mask: (batch, kv_len) padding mask
            
        Returns:
            Tuple of (fused_output, gate_values)
        """
        # Apply cross-modal attention
        attended_output, _ = self.cross_attention(
            query=query_features,
            key=key_value_features,
            value=key_value_features,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Compute gate values
        # Concatenate query and attended output
        combined = torch.cat([query_features, attended_output], dim=-1)
        gate_values = self.gate(combined)  # (batch, query_len, embed_dim)
        
        # Apply gating
        gated_output = gate_values * attended_output + \
                      (1 - gate_values) * query_features
        
        # Output projection
        fused_output = self.output_proj(gated_output)
        
        return fused_output, gate_values


class HierarchicalCrossModalAttention(nn.Module):
    """
    Hierarchical cross-modal attention for multi-level fusion.
    
    Applies cross-modal attention at multiple levels of abstraction,
    useful for capturing both fine-grained and coarse-grained
    correspondences between modalities.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_levels: int = 3,
        num_heads: int = 8,
        dropout: float = 0.15
    ):
        """
        Initialize hierarchical cross-modal attention
        
        Args:
            embed_dim: Embedding dimension
            num_levels: Number of hierarchy levels
            num_heads: Number of attention heads per level
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_levels = num_levels
        
        # Attention at each level
        self.level_attentions = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout)
            for _ in range(num_levels)
        ])
        
        # Pooling between levels
        self.pooling = nn.ModuleList([
            nn.AdaptiveAvgPool1d(embed_dim // (2 ** (i + 1)))
            for i in range(num_levels - 1)
        ])
        
        # Level fusion
        self.level_fusion = nn.Sequential(
            nn.Linear(num_levels * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        logger.info(f"Hierarchical cross-modal attention initialized: "
                   f"{num_levels} levels")
    
    def forward(
        self,
        query_features: torch.Tensor,
        key_value_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Apply hierarchical cross-modal attention
        
        Args:
            query_features: (batch, query_len, embed_dim)
            key_value_features: (batch, kv_len, embed_dim)
            key_padding_mask: (batch, kv_len) padding mask
            
        Returns:
            Tuple of (fused_output, level_outputs)
        """
        level_outputs = []
        
        q = query_features
        kv = key_value_features
        
        for i, attention in enumerate(self.level_attentions):
            # Apply attention at this level
            attended, _ = attention(
                query=q,
                key=kv,
                value=kv,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
            
            level_outputs.append(attended)
            
            # Pool for next level (if not last)
            if i < self.num_levels - 1:
                # Coarsen representations
                q = self.pooling[i](q.transpose(1, 2)).transpose(1, 2)
                kv = self.pooling[i](kv.transpose(1, 2)).transpose(1, 2)
        
        # Concatenate all levels and fuse
        # Upsample lower levels to match first level
        aligned_outputs = []
        target_len = level_outputs[0].size(1)
        
        for output in level_outputs:
            if output.size(1) != target_len:
                # Upsample using interpolation
                output = F.interpolate(
                    output.transpose(1, 2),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            aligned_outputs.append(output)
        
        # Concatenate and fuse
        multi_level = torch.cat(aligned_outputs, dim=-1)
        fused_output = self.level_fusion(multi_level)
        
        return fused_output, level_outputs


# Example usage and testing
if __name__ == "__main__":
    print("Cross-Modal Attention Mechanisms")
    print("=" * 70)
    
    batch_size = 4
    text_len = 128
    kg_len = 50
    embed_dim = 768
    
    # Create dummy data
    text_features = torch.randn(batch_size, text_len, embed_dim)
    kg_features = torch.randn(batch_size, kg_len, embed_dim)
    text_mask = torch.zeros(batch_size, text_len).bool()
    kg_mask = torch.zeros(batch_size, kg_len).bool()
    
    # Test basic cross-modal attention
    print("\n1. Basic Cross-Modal Attention:")
    print("-" * 70)
    
    cross_attn = CrossModalAttention(embed_dim=embed_dim, num_heads=8)
    
    output, weights = cross_attn(
        query=text_features,
        key=kg_features,
        value=kg_features,
        key_padding_mask=kg_mask
    )
    
    print(f"Text features: {text_features.shape}")
    print(f"KG features: {kg_features.shape}")
    print(f"Output: {output.shape}")
    print(f"Attention weights: {weights.shape if weights is not None else None}")
    
    # Test bidirectional attention
    print("\n2. Bidirectional Cross-Modal Attention:")
    print("-" * 70)
    
    bidir_attn = BidirectionalCrossModalAttention(embed_dim=embed_dim)
    
    fused_text, fused_kg, attn_dict = bidir_attn(
        text_features, kg_features, text_mask, kg_mask
    )
    
    print(f"Fused text: {fused_text.shape}")
    print(f"Fused KG: {fused_kg.shape}")
    print(f"Text→KG attention: {attn_dict['text_to_kg'].shape}")
    print(f"KG→Text attention: {attn_dict['kg_to_text'].shape}")
    
    # Test gated fusion
    print("\n3. Gated Cross-Modal Fusion:")
    print("-" * 70)
    
    gated_fusion = GatedCrossModalFusion(embed_dim=embed_dim)
    
    gated_output, gates = gated_fusion(
        text_features, kg_features, kg_mask
    )
    
    print(f"Gated output: {gated_output.shape}")
    print(f"Gate values: {gates.shape}")
    print(f"Gate statistics: mean={gates.mean():.3f}, std={gates.std():.3f}")
    
    # Test hierarchical attention
    print("\n4. Hierarchical Cross-Modal Attention:")
    print("-" * 70)
    
    hier_attn = HierarchicalCrossModalAttention(
        embed_dim=embed_dim, num_levels=3
    )
    
    hier_output, level_outputs = hier_attn(
        text_features, kg_features, kg_mask
    )
    
    print(f"Hierarchical output: {hier_output.shape}")
    print(f"Number of levels: {len(level_outputs)}")
    for i, level_out in enumerate(level_outputs):
        print(f"  Level {i+1}: {level_out.shape}")
    
    # Parameter counts
    print(f"\n{'=' * 70}")
    print("PARAMETER COUNTS:")
    print(f"{'=' * 70}")
    print(f"Basic cross-modal: {sum(p.numel() for p in cross_attn.parameters()):,}")
    print(f"Bidirectional: {sum(p.numel() for p in bidir_attn.parameters()):,}")
    print(f"Gated fusion: {sum(p.numel() for p in gated_fusion.parameters()):,}")
    print(f"Hierarchical: {sum(p.numel() for p in hier_attn.parameters()):,}")
    
    print(f"\n{'=' * 70}")
    print("Cross-modal attention mechanisms ready!")
    print(f"{'=' * 70}")
