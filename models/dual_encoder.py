"""
Dual Encoder Architecture
==========================

Dual-encoder architecture that separately processes textual content and
knowledge graph structure, then integrates representations through cross-modal
attention mechanisms.

This module implements the text encoder component of the hybrid neural-symbolic
architecture described in Section 3.3 of the paper.

Authors: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from transformers import AutoModel, AutoConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """
    Text encoder using pre-trained transformer (BART/FinBERT).
    
    Processes financial document text to produce contextual representations
    that capture semantic meaning, financial terminology, and discourse structure.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large",
        hidden_dim: int = 1024,
        dropout: float = 0.15,
        freeze_base: bool = False
    ):
        """
        Initialize text encoder
        
        Args:
            model_name: Pre-trained transformer model identifier
            hidden_dim: Hidden dimension size
            dropout: Dropout probability
            freeze_base: Whether to freeze base model parameters
        """
        super().__init__()
        
        logger.info(f"Loading text encoder: {model_name}")
        
        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        self.hidden_dim = hidden_dim
        self.embed_dim = self.config.hidden_size
        
        # Projection layer to common embedding space
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Optional: freeze transformer parameters
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
            logger.info("Base transformer frozen")
        
        logger.info(f"Text encoder initialized: {self.embed_dim}D → {hidden_dim}D")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_pooled: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode text input
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) attention mask
            return_pooled: Whether to return pooled representation
            
        Returns:
            Tuple of (sequence_output, pooled_output)
            - sequence_output: (batch_size, seq_len, hidden_dim)
            - pooled_output: (batch_size, hidden_dim) if return_pooled else None
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract hidden states
        # Last hidden state: (batch_size, seq_len, embed_dim)
        hidden_states = outputs.last_hidden_state
        
        # Project to common embedding space
        # (batch_size, seq_len, hidden_dim)
        projected_states = self.projection(hidden_states)
        
        # Pooled representation (using [CLS] token or mean pooling)
        if return_pooled:
            # Mean pooling over sequence length (ignore padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(projected_states.size())
            sum_embeddings = torch.sum(projected_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = None
        
        return projected_states, pooled_output
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from embedding layer"""
        return self.transformer.get_input_embeddings()(input_ids)


class DualEncoder(nn.Module):
    """
    Dual-encoder architecture processing text and knowledge graph separately.
    
    Implements the dual-encoder design from Section 3.3 that enables
    specialized processing of each modality before cross-modal fusion.
    """
    
    def __init__(
        self,
        text_encoder_name: str = "facebook/bart-large",
        kg_encoder_config: Optional[Dict] = None,
        hidden_dim: int = 768,
        kg_output_dim: Optional[int] = None,
        dropout: float = 0.15,
        freeze_text_encoder: bool = False
    ):
        """
        Initialize dual encoder
        
        Args:
            text_encoder_name: Pre-trained model for text encoding
            kg_encoder_config: Configuration for KG encoder
            hidden_dim: Common embedding dimension (cross-modal fusion, 768)
            kg_output_dim: Output dimension of the attached KG encoder (1024 in
                Section 3.6.2); projected to hidden_dim before fusion. Defaults
                to hidden_dim (no projection).
            dropout: Dropout probability
            freeze_text_encoder: Whether to freeze text encoder
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.kg_output_dim = kg_output_dim if kg_output_dim is not None else hidden_dim
        
        # Text encoder (transformer-based)
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_base=freeze_text_encoder
        )
        
        # KG encoder will be imported from knowledge_graph_encoder.py
        # Placeholder for initialization
        self.kg_encoder = None  # Will be set externally
        
        # KG encoder output (1024, matching the BART hidden size) -> fusion dim (768)
        if self.kg_output_dim != hidden_dim:
            self.kg_to_fusion = nn.Linear(self.kg_output_dim, hidden_dim)
        else:
            self.kg_to_fusion = nn.Identity()
        
        # Modality-specific normalization
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.kg_norm = nn.LayerNorm(hidden_dim)
        
        # Temperature scaling for similarity computation
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        logger.info("Dual encoder initialized")
    
    def set_kg_encoder(self, kg_encoder: nn.Module):
        """Set knowledge graph encoder (to avoid circular imports)"""
        self.kg_encoder = kg_encoder
        logger.info("KG encoder attached to dual encoder")
    
    def forward(
        self,
        # Text inputs
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        # KG inputs
        kg_node_features: Optional[torch.Tensor] = None,
        kg_adjacency: Optional[torch.Tensor] = None,
        # Options
        return_similarities: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual encoder
        
        Args:
            input_ids: (batch_size, seq_len) text token IDs
            attention_mask: (batch_size, seq_len) text attention mask
            kg_node_features: (batch_size, num_nodes, feature_dim) KG node features
            kg_adjacency: (batch_size, num_nodes, num_nodes) KG adjacency matrix
            return_similarities: Whether to compute cross-modal similarities
            
        Returns:
            Dictionary with encoded representations and optional similarities
        """
        batch_size = input_ids.size(0)
        
        # Encode text
        text_sequence, text_pooled = self.text_encoder(
            input_ids, attention_mask, return_pooled=True
        )
        
        # Normalize text representations
        text_sequence = self.text_norm(text_sequence)
        text_pooled = self.text_norm(text_pooled)
        
        outputs = {
            'text_sequence': text_sequence,  # (batch, seq_len, hidden_dim)
            'text_pooled': text_pooled,      # (batch, hidden_dim)
        }
        
        # Encode knowledge graph if provided
        if kg_node_features is not None and self.kg_encoder is not None:
            # Process KG through graph encoder
            kg_encoded = self.kg_encoder(kg_node_features, kg_adjacency)
            
            # Project KG encoder output to the fusion dimension
            kg_encoded = self.kg_to_fusion(kg_encoded)
            
            # Normalize KG representations
            kg_encoded = self.kg_norm(kg_encoded)
            
            # Pooled KG representation (mean over nodes)
            kg_pooled = kg_encoded.mean(dim=1)  # (batch, hidden_dim)
            
            outputs['kg_sequence'] = kg_encoded    # (batch, num_nodes, hidden_dim)
            outputs['kg_pooled'] = kg_pooled       # (batch, hidden_dim)
            
            # Compute cross-modal similarities if requested
            if return_similarities:
                similarities = self._compute_similarities(
                    text_pooled, kg_pooled
                )
                outputs['similarities'] = similarities
        
        return outputs
    
    def _compute_similarities(
        self,
        text_embeddings: torch.Tensor,
        kg_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between text and KG embeddings
        
        Args:
            text_embeddings: (batch, hidden_dim)
            kg_embeddings: (batch, hidden_dim)
            
        Returns:
            Similarity scores (batch, batch) - text vs KG alignment
        """
        # Normalize embeddings
        text_norm = nn.functional.normalize(text_embeddings, p=2, dim=1)
        kg_norm = nn.functional.normalize(kg_embeddings, p=2, dim=1)
        
        # Compute similarity matrix with temperature scaling
        similarity = torch.matmul(text_norm, kg_norm.t()) / self.temperature
        
        return similarity
    
    def compute_contrastive_loss(
        self,
        text_embeddings: torch.Tensor,
        kg_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss for text-KG alignment
        
        Encourages matched text-KG pairs to be close in embedding space
        while pushing apart unmatched pairs.
        
        Args:
            text_embeddings: (batch, hidden_dim)
            kg_embeddings: (batch, hidden_dim)
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = text_embeddings.size(0)
        
        # Compute similarity matrix
        similarities = self._compute_similarities(text_embeddings, kg_embeddings)
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=similarities.device)
        
        # Cross-entropy loss (InfoNCE)
        loss_text_to_kg = nn.functional.cross_entropy(similarities, labels)
        loss_kg_to_text = nn.functional.cross_entropy(similarities.t(), labels)
        
        # Symmetric loss
        contrastive_loss = (loss_text_to_kg + loss_kg_to_text) / 2
        
        return contrastive_loss


class DualEncoderWithProjection(nn.Module):
    """
    Extended dual encoder with additional projection heads.
    
    Adds task-specific projection heads for different downstream tasks
    like summarization, classification, or retrieval.
    """
    
    def __init__(
        self,
        dual_encoder: DualEncoder,
        projection_dim: int = 512,
        num_projection_layers: int = 2,
        dropout: float = 0.15
    ):
        """
        Initialize dual encoder with projection heads
        
        Args:
            dual_encoder: Base dual encoder
            projection_dim: Dimension of projection space
            num_projection_layers: Number of projection layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dual_encoder = dual_encoder
        hidden_dim = dual_encoder.hidden_dim
        
        # Projection head for text
        text_layers = []
        in_dim = hidden_dim
        for _ in range(num_projection_layers):
            text_layers.extend([
                nn.Linear(in_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(projection_dim)
            ])
            in_dim = projection_dim
        self.text_projection = nn.Sequential(*text_layers)
        
        # Projection head for KG (same architecture)
        kg_layers = []
        in_dim = hidden_dim
        for _ in range(num_projection_layers):
            kg_layers.extend([
                nn.Linear(in_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(projection_dim)
            ])
            in_dim = projection_dim
        self.kg_projection = nn.Sequential(*kg_layers)
        
        logger.info(f"Dual encoder with projection initialized: "
                   f"{hidden_dim}D → {projection_dim}D")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kg_node_features: Optional[torch.Tensor] = None,
        kg_adjacency: Optional[torch.Tensor] = None,
        project: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional projection
        
        Args:
            Same as DualEncoder.forward()
            project: Whether to apply projection heads
            
        Returns:
            Dictionary with encoded and projected representations
        """
        # Get base encodings
        outputs = self.dual_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kg_node_features=kg_node_features,
            kg_adjacency=kg_adjacency,
            return_similarities=False
        )
        
        if project:
            # Apply projection heads to pooled representations
            if 'text_pooled' in outputs:
                outputs['text_projected'] = self.text_projection(
                    outputs['text_pooled']
                )
            
            if 'kg_pooled' in outputs:
                outputs['kg_projected'] = self.kg_projection(
                    outputs['kg_pooled']
                )
        
        return outputs


# Example usage and testing
if __name__ == "__main__":
    print("Dual Encoder Architecture")
    print("=" * 70)
    
    # Initialize dual encoder
    print("\nInitializing dual encoder...")
    dual_encoder = DualEncoder(
        text_encoder_name="facebook/bart-base",  # Smaller for demo
        hidden_dim=768,
        dropout=0.15
    )
    
    print(f"Text encoder parameters: {sum(p.numel() for p in dual_encoder.text_encoder.parameters()):,}")
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 128
    num_nodes = 50
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    kg_node_features = torch.randn(batch_size, num_nodes, 768)
    kg_adjacency = torch.rand(batch_size, num_nodes, num_nodes)
    kg_adjacency = (kg_adjacency > 0.9).float()  # Sparse adjacency
    
    # Mock KG encoder
    class MockKGEncoder(nn.Module):
        def forward(self, features, adj):
            return features  # Identity for demo
    
    dual_encoder.set_kg_encoder(MockKGEncoder())
    
    # Forward pass
    print("\nRunning forward pass...")
    outputs = dual_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        kg_node_features=kg_node_features,
        kg_adjacency=kg_adjacency,
        return_similarities=True
    )
    
    # Display outputs
    print(f"\n{'=' * 70}")
    print("OUTPUT SHAPES:")
    print(f"{'=' * 70}")
    for key, value in outputs.items():
        if torch.is_tensor(value):
            print(f"{key:20s}: {tuple(value.shape)}")
    
    # Test contrastive loss
    if 'text_pooled' in outputs and 'kg_pooled' in outputs:
        loss = dual_encoder.compute_contrastive_loss(
            outputs['text_pooled'],
            outputs['kg_pooled']
        )
        print(f"\nContrastive loss: {loss.item():.4f}")
    
    # Test projection head
    print(f"\n{'=' * 70}")
    print("TESTING PROJECTION HEAD:")
    print(f"{'=' * 70}")
    
    dual_encoder_proj = DualEncoderWithProjection(
        dual_encoder,
        projection_dim=512,
        num_projection_layers=2
    )
    
    outputs_proj = dual_encoder_proj(
        input_ids=input_ids,
        attention_mask=attention_mask,
        kg_node_features=kg_node_features,
        kg_adjacency=kg_adjacency,
        project=True
    )
    
    for key, value in outputs_proj.items():
        if 'projected' in key and torch.is_tensor(value):
            print(f"{key:20s}: {tuple(value.shape)}")
    
    print(f"\n{'=' * 70}")
    print("Dual encoder architecture ready!")
    print(f"{'=' * 70}")
