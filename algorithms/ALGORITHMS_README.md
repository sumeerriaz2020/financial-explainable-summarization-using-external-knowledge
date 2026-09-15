# Algorithm Implementations - Financial Explainable Summarization

This directory contains complete, research implementations of all 6 core algorithms from the paper:

**"An eXplainable Approach to Abstractive Text Summarization Using External Knowledge: A Novel Framework for Financial Domain Applications"**

Authors: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi

---

## 📁 Files Overview

| File | Size | Description | Lines |
|------|------|-------------|-------|
| `algorithm_1_kg_construction.py` | 36 KB | Knowledge Graph Construction | ~991 |
| `algorithm_2_hybrid_inference.py` | 31 KB | Hybrid Neural-Symbolic Inference | ~938 |
| `algorithm_3_mesa.py` | 16 KB | MESA Framework | ~438 |
| `algorithm_4_causal_explain.py` | 23 KB | CAUSAL-EXPLAIN Framework | ~649 |
| `algorithm_5_consensus.py` | 19 KB | INTERPRETABLE-CONSENSUS | ~456 |
| `algorithm_6_training.py` | 23 KB | Multi-Stage Training Pipeline | ~647 |

**Total: 148 KB, ~4,119 lines of research Python code**

---

## 🎯 Algorithm Descriptions

### Algorithm 1: Knowledge Graph Construction
**File:** `algorithm_1_kg_construction.py`  
**Reference:** Section 3.2.3, Algorithm 1

**What it does:**
- Imports FIBO core modules as ontology foundation
- Extracts entities, temporal annotations, and causal candidates from documents
- Links entities to FIBO ontology concepts
- Computes entity embeddings (textual + structural + type information)
- Detects causal relationships with confidence scoring (Equation 6)
- Validates against FIBO constraints

**Key Features:**
- FinBERT integration for entity extraction
- Multi-component embedding (Equation 1)
- Causal confidence classifier
- Graph validation and cleaning
- Handles 2.4M entity nodes, 8.7M edges

**Example Usage:**
```python
from algorithm_1_kg_construction import KnowledgeGraphConstructor

# Initialize
constructor = KnowledgeGraphConstructor(
    fibo_ontology_path="fibo.owl",
    causal_threshold=0.7
)

# Construct KG from documents
documents = [{"text": "...", "id": "doc_001"}]
kg = constructor.construct_knowledge_graph(documents)

print(f"Nodes: {kg.number_of_nodes()}, Edges: {kg.number_of_edges()}")
```

---

### Algorithm 2: Hybrid Neural-Symbolic Inference
**File:** `algorithm_2_hybrid_inference.py`  
**Reference:** Section 3.3, Algorithm 2

**What it does:**
- Encodes document text using transformer (BART)
- Extracts relevant KG subgraph
- Encodes KG structure using Graph Attention Networks
- Performs cross-modal attention (Equation 2)
- Executes multi-hop reasoning over knowledge paths
- Generates summary with factual verification
- Produces stakeholder-specific explanations

**Key Components:**
- Dual-encoder architecture (text + KG)
- Cross-modal attention mechanism
- Multi-hop reasoning (3 hops default)
- Factual consistency verification
- 21% computational overhead vs baseline

**Example Usage:**
```python
from algorithm_2_hybrid_inference import HybridInferenceEngine

# Initialize
model = HybridNeuralSymbolicModel()
engine = HybridInferenceEngine(model, kg, entity_embeddings)

# Generate summary and explanation
document = {"text": "...", "id": "doc_001"}
stakeholder = StakeholderProfile(role='analyst', ...)

output = engine.summarize_and_explain(document, stakeholder)
print(f"Summary: {output.summary}")
print(f"Factual Score: {output.factual_consistency_score}")
```

---

### Algorithm 3: MESA Framework
**File:** `algorithm_3_mesa.py`  
**Reference:** Section 3.4.1, Algorithm 3

**What it does:**
- Generates stakeholder-aware explanations
- Multi-objective optimization (Equation 5)
- Online weight adaptation (simplified step; the REINFORCE-style update of Section 3.4.1 is specified in the paper, not included in this release)
- Dynamic preference learning
- Supports 4 stakeholder types: analyst, compliance, executive, investment manager

**Key Features:**
- 21.3% improvement in stakeholder satisfaction (SSI 0.61→0.74)
- Adaptive weight updates based on feedback
- Quality metrics: ROUGE-L, BERTScore, factual consistency
- Explainability metrics: comprehension, trust, actionability

**Example Usage:**
```python
from algorithm_3_mesa import MESAFramework

mesa = MESAFramework()

# Get stakeholder profile
profile = mesa.get_stakeholder_profile('analyst')

# Generate explanation
explanation = mesa.generate_explanation(
    summary, document, knowledge_graph, profile
)

# Update based on user feedback
mesa.update_with_feedback(profile, explanation, feedback_score=0.85)
```

---

### Algorithm 4: CAUSAL-EXPLAIN Framework
**File:** `algorithm_4_causal_explain.py`  
**Reference:** Section 3.4.2, Algorithm 4

**What it does:**
- Extracts source-attributed causal chains from financial documents
- Validates against knowledge graph constraints
- Computes importance weights (frequency, centrality, relevance)
- Preserves top-k chains within summary budget
- Computes Causal Preservation Score (CPS)

**Key Metrics:**
- CPS (standard): Equation 7
- CPS (weighted): Equation 8
- 50% improvement over baseline (CPS 0.34→0.51)
- Causal accuracy, coherence, temporal fidelity

**Example Usage:**
```python
from algorithm_4_causal_explain import CausalExplainFramework

causal_framework = CausalExplainFramework(
    causal_threshold=0.7,
    min_confidence=0.35
)

# Extract and preserve causal chains
result = causal_framework.extract_and_preserve_causal_chains(
    document, knowledge_graph, summary_length_budget=200
)

print(f"Selection CPS: {result.selection_cps}")  # None if no source chains
print(f"Preserved: {len(result.preserved_chains)} chains")
```

---

### Algorithm 5: INTERPRETABLE-CONSENSUS
**File:** `algorithm_5_consensus.py`  
**Reference:** Section 3.4.4, Algorithm 5

**What it does:**
- Ensemble explanation selection
- Integrates 4 generators: template, retrieval, neural, hybrid
- Transparent scoring across quality dimensions
- Automated conflict resolution (expert rules → confidence → majority)
- Complete audit trail for regulatory compliance

**Key Features:**
- 37.2% improvement in explanation consistency
- Uncertainty quantification with confidence intervals
- Conflict detection and resolution
- Full transparency for auditing

**Example Usage:**
```python
from algorithm_5_consensus import ConsensusFramework

consensus = ConsensusFramework(
    score_weights={
        'accuracy': 0.30,
        'linguistic_quality': 0.25,
        'stakeholder_fit': 0.25,
        'compliance': 0.20
    }
)

# Generate consensus explanation
explanation, audit_trail = consensus.select_consensus_explanation(
    summary, document, knowledge_graph, stakeholder
)

print(f"Selected: {audit_trail.selection_method}")
print(f"Confidence: {audit_trail.uncertainty_metrics}")
```

---

### Algorithm 6: Multi-Stage Training
**File:** `algorithm_6_training.py`  
**Reference:** Section 3.6, Algorithm 6

**What it does:**
- Stage 1: Domain pre-training on financial corpora
- Stage 2: Knowledge-aware attention learning (Equation 3)
- Stage 3: Joint optimization with composite loss (Equation 10)
- Early stopping with patience
- Learning rate scheduling

**Training Configuration:**
- Stage 1: LR=1e-4, 3 epochs
- Stage 2: LR=5e-5, 2 epochs, λ=0.5
- Stage 3: LR=2e-5, 5 epochs, patience=2, LR decay 0.9 per epoch
- Loss weights (Equation 10): α=0.72 (summarization), β=0.18 (KG alignment), γ=0.10 (explanation)

**Example Usage:**
```python
from algorithm_6_training import MultiStageTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    stage1_epochs=3,
    stage2_epochs=2,
    stage3_epochs=5,
    alpha=0.72,  # Summarization
    beta=0.18,   # KG alignment
    gamma=0.10   # Explanation
)

# Initialize trainer
trainer = MultiStageTrainer(model, train_data, val_data, config)

# Execute complete training pipeline
results = trainer.train_all_stages()

print(f"Best validation: {results['best_val_performance']:.4f}")
```

---

## 🔧 Dependencies

All algorithms require:

```txt
torch>=2.0.0
transformers>=4.35.0
networkx>=3.1
numpy>=1.24.0
spacy>=3.7.0
tqdm>=4.66.0
```

Install with:
```bash
pip install torch transformers networkx numpy spacy tqdm --break-system-packages
python -m spacy download en_core_web_sm
```

---

## 📊 Performance Metrics (from Paper)

| Framework | Metric | Improvement | Result |
|-----------|--------|-------------|---------|
| **Overall** | ROUGE-L | +18.1% | 0.421→0.497 |
| **MESA** | Stakeholder Satisfaction (SSI) | +21.3% | 0.61→0.74 |
| **CAUSAL-EXPLAIN** | Causal Preservation (CPS) | +50.0% | 0.34→0.51 |
| **CONSENSUS** | Explanation Consistency | +37.2% | 0.43→0.59 |
| **Hybrid Model** | Factual Consistency | +3.9 pp | 83.7%→87.6% |
| **TEMPORAL** | Temporal Consistency (TCC) | +42.1% | 0.38→0.54 |

---

## ⚙️ System Integration

### Complete Pipeline

```python
# Step 1: Construct Knowledge Graph
from algorithm_1_kg_construction import KnowledgeGraphConstructor
constructor = KnowledgeGraphConstructor("fibo.owl")
kg = constructor.construct_knowledge_graph(documents)

# Step 2: Initialize Hybrid Model
from algorithm_2_hybrid_inference import HybridNeuralSymbolicModel, HybridInferenceEngine
model = HybridNeuralSymbolicModel()
engine = HybridInferenceEngine(model, kg, entity_embeddings)

# Step 3: Generate Summary with Explanation
output = engine.summarize_and_explain(document, stakeholder)

# Step 4: Extract Causal Chains
from algorithm_4_causal_explain import CausalExplainFramework
causal = CausalExplainFramework()
causal_result = causal.extract_and_preserve_causal_chains(document, kg)

# Step 5: Generate Stakeholder-Specific Explanation
from algorithm_3_mesa import MESAFramework
mesa = MESAFramework()
explanation = mesa.generate_explanation(output.summary, document, kg, stakeholder)

# Step 6: Validate with Consensus
from algorithm_5_consensus import ConsensusFramework
consensus = ConsensusFramework()
final_explanation, audit = consensus.select_consensus_explanation(
    output.summary, document, kg, stakeholder
)
```

### Training Pipeline

```python
# Train the complete system
from algorithm_6_training import MultiStageTrainer, TrainingConfig

config = TrainingConfig()
trainer = MultiStageTrainer(model, train_data, val_data, config)
results = trainer.train_all_stages()

# Best model is automatically loaded
trained_model = results['final_model']
```

---

## 🚨 Important Limitations

From the paper's error analysis:

1. **Error Rate**: 56.5% total error rate (21.8% high-severity)
   - Requires human oversight for mission-critical applications
   
2. **Computational Overhead**: 
   - 21% slower inference (2.9s vs 2.4s)
   - 19.5% higher memory (9.8GB vs 8.2GB)
   - 50% longer training time

3. **Causal Preservation**: CPS of 0.51 means 49% of source-attributed causal claims are not preserved

4. **Scalability**: Documents >50 pages show 18% ROUGE-L degradation

5. **Coverage**: FIBO has incomplete coverage for:
   - Emerging products (SPACs, crypto derivatives)
   - Post-2023 financial instruments
   - Informal terminology

6. **Regulatory Compliance**: 
   - Regulatory alignment is self-assessed at the design level (qualitative), not a legal evaluation
   - Requires independent legal audit for certification

---

## 📝 Citation

If you use these implementations, please cite:

```bibtex
@article{riaz2026explainable,
  title={An eXplainable Approach to Abstractive Text Summarization Using External Knowledge: A Novel Framework for Financial Domain Applications},
  author={Riaz, Sumeer and Bashir, M. Bilal and Naqvi, Syed Ali Hassan},
  journal={Expert Systems With Applications (under review)},
  year={2026}
}
```

---

## 📧 Contact

For questions or issues:
- Sumeer Riaz: sumeer33885@iqraisb.edu.pk
- Dr. M. Bilal Bashir: bilal.bashir@iqraisb.edu.pk
- Syed Ali Hassan Naqvi: ali33884@iqraisb.edu.pk

---

## 📄 License

MIT License (see LICENSE file at repository root).

---

## ✅ Code Quality

All implementations include:
- ✅ Comprehensive docstrings
- ✅ Type hints throughout
- ✅ Error handling
- ✅ Logging for debugging
- ✅ Example usage in `__main__`
- ✅ Documented, tested structure
- ✅ Faithful to paper specifications
