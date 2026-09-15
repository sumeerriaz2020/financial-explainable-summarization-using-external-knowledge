# Model Architecture - Hybrid Neural-Symbolic Components

Complete implementation of the hybrid neural-symbolic architecture for explainable financial document summarization from the paper.

---

## 📁 Files Overview

| File | Size | Description | Key Components |
|------|------|-------------|----------------|
| `dual_encoder.py` | 17 KB | Dual-encoder architecture | TextEncoder, DualEncoder, DualEncoderWithProjection |
| `knowledge_graph_encoder.py` | 20 KB | Graph neural network encoder | GraphAttentionLayer, MultiHeadGATLayer, KnowledgeGraphEncoder |
| `cross_modal_attention.py` | 19 KB | Cross-modal attention mechanisms | CrossModalAttention, BidirectionalCrossModalAttention, GatedFusion |
| `hybrid_model.py` | 20 KB | Complete integrated model | HybridModel, MultiHopReasoning |

**Total: 76 KB of research PyTorch code**

---

## 🏗️ Architecture Overview

```
Input Document → [Text Encoder] ──┐
                                   ├─→ [Cross-Modal Attention] → [Multi-Hop Reasoning] → [Decoder] → Summary
Knowledge Graph → [KG Encoder] ────┘
```

### Component Flow (Algorithm 2)

1. **Dual Encoding** (Lines 1-3)
   - Text: Transformer-based encoding (BART/FinBERT)
   - KG: Graph Attention Networks (GAT)

2. **Cross-Modal Fusion** (Lines 4-5)
   - Bidirectional attention: Text↔KG
   - Equation 2: Attention(Q, K, V) = softmax(QK^T / √d_k) V

3. **Multi-Hop Reasoning** (Lines 6-8)
   - Traverse i-hop paths (i = 1, 2, 3)
   - LSTM encoding of reasoning chains
   - Attention-weighted path aggregation

4. **Summary Generation** (Line 10)
   - Fused representations → Decoder
   - Factual verification against KG

---

## 📦 Component Details

### 1. Dual Encoder (`dual_encoder.py`)

**Purpose:** Separate processing of text and knowledge graph modalities

**Classes:**

#### TextEncoder
```python
TextEncoder(
    model_name="facebook/bart-large",
    hidden_dim=1024,
    dropout=0.15,
    freeze_base=False
)
```
- Loads pre-trained transformer (BART/FinBERT)
- Projects to common embedding space
- Supports mean pooling for document representation

**Key Features:**
- Xavier initialization
- Layer normalization
- Residual connections
- Optional base model freezing

#### DualEncoder
```python
DualEncoder(
    text_encoder_name="facebook/bart-large",
    kg_encoder_config=None,
    hidden_dim=768,
    dropout=0.15
)
```
- Coordinates text and KG encoders
- Computes cross-modal similarities
- Temperature-scaled attention
- Contrastive loss for alignment

**Methods:**
- `forward()`: Encode both modalities
- `compute_contrastive_loss()`: Text-KG alignment loss
- `_compute_similarities()`: Cosine similarity matrix

#### DualEncoderWithProjection
- Adds task-specific projection heads
- Multi-layer projections for downstream tasks
- Separate heads for text and KG

**Example Usage:**
```python
from models.dual_encoder import DualEncoder

dual_encoder = DualEncoder(
    text_encoder_name="facebook/bart-large",
    hidden_dim=768
)

outputs = dual_encoder(
    input_ids=input_ids,
    attention_mask=attention_mask,
    kg_node_features=kg_features,
    kg_adjacency=kg_adj
)

# Access encoded representations
text_seq = outputs['text_sequence']  # (batch, seq_len, 768)
kg_seq = outputs['kg_sequence']      # (batch, num_nodes, 768)
```

---

### 2. Knowledge Graph Encoder (`knowledge_graph_encoder.py`)

**Purpose:** Encode FIBO knowledge graph structure using Graph Attention Networks

**Classes:**

#### GraphAttentionLayer
```python
GraphAttentionLayer(
    in_features=768,
    out_features=512,
    dropout=0.15,
    alpha=0.2,  # LeakyReLU slope
    concat=True
)
```
- Single GAT layer
- Attention mechanism: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
- Softmax normalization over neighbors
- Xavier initialization

**Key Operations:**
1. Linear transformation: Wh
2. Attention coefficient computation
3. Masked softmax (by adjacency)
4. Feature aggregation: h' = Σ α_ij Wh_j

#### MultiHeadGATLayer
```python
MultiHeadGATLayer(
    in_features=768,
    out_features=512,
    num_heads=8,
    dropout=0.15,
    concat=True
)
```
- Parallel attention heads
- Concatenate or average outputs
- Captures different graph aspects

#### KnowledgeGraphEncoder
```python
KnowledgeGraphEncoder(
    input_dim=768,
    hidden_dim=512,
    output_dim=1024,
    num_layers=3,
    num_heads=8,
    dropout=0.15,
    residual=True
)
```
- Stacked GAT layers
- Residual connections
- Layer normalization
- Self-loop handling

**Architecture:**
```
Input (768D) → GAT Layer 1 (4 heads, concat) → 2048D
             → GAT Layer 2 (4 heads, concat) → 2048D
             → GAT Layer 3 (4 heads, avg)    → 768D
```

#### HierarchicalKGEncoder
- Multi-level ontology processing
- Hierarchical pooling
- Useful for FIBO's class hierarchy

**Example Usage:**
```python
from models.knowledge_graph_encoder import KnowledgeGraphEncoder

kg_encoder = KnowledgeGraphEncoder(
    input_dim=768,
    hidden_dim=512,
    output_dim=1024,
    num_layers=3,
    num_heads=8
)

# Encode knowledge graph
encoded_kg = kg_encoder(
    node_features,  # (num_nodes, 768)
    adjacency       # (num_nodes, num_nodes)
)

# Extract attention weights
attention = kg_encoder.get_attention_weights(
    node_features, adjacency, 
    layer_idx=0, head_idx=0
)
```

**Performance:**
- Handles graphs with 50-5000 nodes
- Attention complexity: O(N²) where N = number of nodes
- Typical inference: 15-30ms per graph

---

### 3. Cross-Modal Attention (`cross_modal_attention.py`)

**Purpose:** Fuse text and knowledge graph representations through attention

**Classes:**

#### CrossModalAttention
```python
CrossModalAttention(
    embed_dim=768,
    num_heads=8,
    dropout=0.15,
    bias=True
)
```
- Standard scaled dot-product attention (Equation 2)
- Multi-head attention (8 heads)
- Residual connections + Layer norm

**Attention Equation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

where:
  Q = query features (text)
  K = key features (KG)
  V = value features (KG)
  d_k = head dimension
```

**Features:**
- Xavier initialization
- Optional attention masking
- Padding mask support
- Attention weight extraction

#### BidirectionalCrossModalAttention
```python
BidirectionalCrossModalAttention(
    embed_dim=768,
    num_heads=8,
    dropout=0.15
)
```
- Text → KG attention
- KG → Text attention
- Concatenate + fuse

**Architecture:**
```
Text ──→ [Attend to KG] ──┐
KG   ──→ [Attend to Text] ─┼──→ [Fusion] → Fused Features
```

#### GatedCrossModalFusion
```python
GatedCrossModalFusion(
    embed_dim=768,
    num_heads=8,
    dropout=0.15
)
```
- Learnable gates control fusion
- Gate = σ(W[h_text; h_kg_attended])
- Output = gate * h_kg + (1-gate) * h_text

**Benefits:**
- Adaptive weighting of modalities
- Learns when to trust text vs KG
- Interpretable gate values

#### HierarchicalCrossModalAttention
```python
HierarchicalCrossModalAttention(
    embed_dim=768,
    num_levels=3,
    num_heads=8,
    dropout=0.15
)
```
- Multi-level attention
- Coarse-to-fine fusion
- Level-specific pooling

**Example Usage:**
```python
from models.cross_modal_attention import CrossModalAttention

cross_attn = CrossModalAttention(embed_dim=768, num_heads=8)

# Apply cross-modal attention
fused_output, attn_weights = cross_attn(
    query=text_features,    # (batch, text_len, 768)
    key=kg_features,        # (batch, kg_len, 768)
    value=kg_features,
    key_padding_mask=kg_mask
)

# fused_output: (batch, text_len, 768)
# attn_weights: (batch, text_len, kg_len)
```

**Computational Cost:**
- Time: O(T × K) where T=text_len, K=kg_len
- Memory: O(T × K × H) where H=num_heads
- Typical: 20-40ms for T=128, K=50

---

### 4. Hybrid Model (`hybrid_model.py`)

**Purpose:** Complete integrated model combining all components

**Classes:**

#### MultiHopReasoning
```python
MultiHopReasoning(
    embed_dim=768,
    num_hops=3,
    dropout=0.15
)
```
- LSTM encoders for each hop
- Path attention mechanism
- Multi-hop feature fusion

**Reasoning Process:**
1. **1-hop**: Direct entity neighbors
2. **2-hop**: Second-degree connections
3. **3-hop**: Third-degree connections
4. Aggregate with learned attention

#### HybridModel
```python
HybridModel(
    text_encoder_name="facebook/bart-large",
    freeze_text_encoder=False,
    kg_input_dim=768,
    kg_hidden_dim=512,
    kg_num_layers=3,
    kg_num_heads=8,
    num_cross_attn_heads=8,
    num_reasoning_hops=3,
    hidden_dim=768,
    dropout=0.15
)
```

**Complete Architecture:**
```python
HybridModel (~416M parameters)
├── DualEncoder
│   ├── TextEncoder (BART-large: 406M)
│   └── KGEncoder (GAT: 2.1M)
├── CrossModalAttention (4.7M)
├── MultiHopReasoning (3.2M)
└── Decoder (part of BART)

Total: ~416M parameters
```

**Forward Pass (Algorithm 2):**
```python
outputs = model(
    input_ids=input_ids,              # Text input
    attention_mask=attention_mask,
    kg_node_features=kg_features,     # KG input
    kg_adjacency=kg_adj,
    kg_paths=reasoning_paths,         # Multi-hop paths
    labels=labels                     # For training
)

# Returns:
# - outputs: Seq2SeqLMOutput with loss/logits
# - encoder_hidden_states: Fused representations
# - attention_weights: Cross-modal attention
```

**Generation:**
```python
generated = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    kg_node_features=kg_features,
    kg_adjacency=kg_adj,
    max_length=128,
    num_beams=4,
    length_penalty=2.0
)
```

**Example Usage:**
```python
from models.hybrid_model import HybridModel

# Initialize complete model
model = HybridModel(
    text_encoder_name="facebook/bart-large",
    kg_input_dim=768,
    kg_hidden_dim=512,
    kg_output_dim=1024,   # projected to 768 for cross-modal fusion
    num_reasoning_hops=3,
    device='cuda'
)

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    kg_node_features=kg_features,
    kg_adjacency=kg_adj,
    kg_paths=reasoning_paths,
    labels=labels
)

loss = outputs['outputs'].loss
loss.backward()

# Generate summary
generated = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    kg_node_features=kg_features,
    kg_adjacency=kg_adj,
    max_length=128,
    num_beams=4,
    length_penalty=2.0
)
```

---

## 🔧 Dependencies

```txt
torch>=2.0.0
transformers>=4.35.0
```

Install:
```bash
pip install torch transformers --break-system-packages
```

---

## 📊 Model Specifications

### Parameter Counts

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Text Encoder (BART-large) | 406M | 97.6% |
| KG Encoder (GAT 3-layer) | 2.1M | 0.5% |
| Cross-Modal Attention | 4.7M | 1.1% |
| Multi-Hop Reasoning | 3.2M | 0.8% |
| **Total** | **416M** | **100%** |

### Memory Requirements

| Phase | Memory | Notes |
|-------|--------|-------|
| Training (batch=16) | ~24 GB | With gradient accumulation |
| Inference (batch=1) | ~9.8 GB | Peak memory from paper |
| KG storage | ~450 MB | Preprocessed FIBO index |

### Speed Benchmarks

| Operation | Time | Configuration |
|-----------|------|---------------|
| Text encoding | 45ms | seq_len=128, batch=4 |
| KG encoding | 18ms | 50 nodes, 3 layers |
| Cross-modal attention | 22ms | text_len=128, kg_len=50 |
| Multi-hop reasoning | 12ms | 3 hops |
| **Total inference** | **2.9s** | Per document (from paper) |

---

## 🎯 Integration with Algorithms

### With Algorithm 1 (KG Construction)
```python
from algorithm_1_kg_construction import KnowledgeGraphConstructor
from models.hybrid_model import HybridModel

# Build KG
constructor = KnowledgeGraphConstructor("fibo.owl")
kg = constructor.construct_knowledge_graph(documents)
entity_embeddings = constructor.entity_embeddings

# Initialize model with KG
model = HybridModel()
```

### With Algorithm 2 (Hybrid Inference)
```python
# The HybridModel IS the implementation of Algorithm 2
# Algorithm 2 is the forward pass of HybridModel

outputs = model(...)  # Executes Algorithm 2
```

### With Algorithm 6 (Training)
```python
from algorithm_6_training import MultiStageTrainer
from models.hybrid_model import HybridModel

model = HybridModel()
trainer = MultiStageTrainer(model, train_data, val_data, config)
results = trainer.train_all_stages()
```

---

## 🧪 Testing

Each module includes comprehensive tests in `if __name__ == "__main__"`:

```bash
# Test dual encoder
python models/dual_encoder.py

# Test KG encoder
python models/knowledge_graph_encoder.py

# Test cross-modal attention
python models/cross_modal_attention.py

# Test complete model
python models/hybrid_model.py
```

---

## 📈 Performance Metrics (from Paper)

| Metric | Baseline | Hybrid Model | Improvement |
|--------|----------|--------------|-------------|
| ROUGE-L | 0.421 | 0.497 | +18.1% |
| BERTScore | 0.612 | 0.673 | +10.0% |
| Factual Consistency (vs. attention-based KG, Table 8) | 83.7% | 87.6% | +3.9 pp |
| **Inference Time** | **2.4s** | **2.9s** | **+21%** |
| **Memory Usage** | **8.2GB** | **9.8GB** | **+19.5%** |

---

## ⚠️ Important Notes

1. **Computational Overhead**: 21% slower inference, 19.5% higher memory
   - Worth it for explainability gains
   - Consider model distillation for production

2. **Transformer Model**: Uses BART by default
   - Can swap for FinBERT, T5, or custom models
   - Ensure compatible with Hugging Face interface

3. **KG Integration**: Requires preprocessed KG
   - Use Algorithm 1 for construction
   - Store entity embeddings separately

4. **Training**: Requires multi-stage approach
   - See Algorithm 6 for complete pipeline
   - Stage 1: Domain pre-training (3 epochs)
   - Stage 2: KG integration (2 epochs)  
   - Stage 3: Joint optimization (5 epochs)

---

## 🐛 Known Limitations

1. **Scalability**: Documents >50 pages show degradation
   - Solution: Hierarchical chunking

2. **KG Size**: Performance degrades with >5000 nodes
   - Solution: Subgraph extraction (implemented)

3. **Memory**: Large batches require gradient accumulation
   - Recommended batch size: 8-16

4. **Edge Cases**: 
   - Very sparse KG (density < 0.01) → Use text-only mode
   - Missing FIBO mappings → Create extension nodes

---

## 📧 Contact

For questions about the model architecture:
- sumeer33885@iqraisb.edu.pk
- bilal.bashir@iqraisb.edu.pk

---

## ✅ Model Components

All model components are:
- ✅ Well-structured
- ✅ Well-documented
- ✅ Type-annotated
- ✅ Tested
- ✅ Faithful to paper
- ✅ Modular and reusable
