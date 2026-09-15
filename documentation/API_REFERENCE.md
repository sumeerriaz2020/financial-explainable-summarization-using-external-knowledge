# API Reference
 
Complete API documentation for the Financial Explainable Summarization framework.

---

## Table of Contents

- [Models](#models)
- [Algorithms](#algorithms)
- [Explainability](#explainability)
- [Knowledge Graph](#knowledge-graph)
- [Utilities](#utilities)
- [Training](#training)
- [Evaluation](#evaluation)

---

## Models

### HybridModel

Main hybrid neural-symbolic model combining BART with knowledge graph integration.

```python
from models import HybridModel

model = HybridModel(
    text_encoder_name: str = "facebook/bart-large",
    kg_input_dim: int = 768,
    kg_hidden_dim: int = 512,
    kg_output_dim: int = 1024,
    num_gat_layers: int = 3,
    num_gat_heads: int = 8,
    num_reasoning_hops: int = 3,
    dropout: float = 0.15,
    attention_dropout: float = 0.15
)
```

**Parameters:**
- `text_encoder_name` (str): HuggingFace model identifier for text encoder
- `kg_input_dim` (int): Input dimension for knowledge graph embeddings
- `kg_hidden_dim` (int): Hidden dimension for KG encoder
- `kg_output_dim` (int): Output dimension (should match text encoder)
- `num_gat_layers` (int): Number of Graph Attention Network layers
- `num_gat_heads` (int): Number of attention heads in GAT
- `num_reasoning_hops` (int): Number of multi-hop reasoning steps
- `dropout` (float): Dropout probability
- `attention_dropout` (float): Attention dropout probability

**Methods:**

#### `forward()`
```python
outputs = model.forward(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kg_node_features: Optional[torch.Tensor] = None,
    kg_adjacency: Optional[torch.Tensor] = None,
    kg_paths: Optional[List] = None,
    labels: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]
```

**Returns:**
- `loss` (torch.Tensor): Combined loss (if labels provided)
- `logits` (torch.Tensor): Output logits
- `attention_weights` (torch.Tensor): Cross-modal attention weights
- `reasoning_paths` (List): Multi-hop reasoning paths

#### `generate()`
```python
outputs = model.generate(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kg_node_features: Optional[torch.Tensor] = None,
    kg_adjacency: Optional[torch.Tensor] = None,
    max_length: int = 128,
    num_beams: int = 4,
    early_stopping: bool = True,
    **kwargs
) -> torch.Tensor
```

**Returns:**
- `output_ids` (torch.Tensor): Generated token IDs

**Example:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = HybridModel()

text = "Apple Inc. reported earnings..."
inputs = tokenizer(text, return_tensors="pt")

summary = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=128
)

decoded = tokenizer.decode(summary[0], skip_special_tokens=True)
```

---

### DualEncoder

Separate encoding for text and knowledge graph.

```python
from models import DualEncoder

encoder = DualEncoder(
    text_encoder_name: str = "facebook/bart-large",
    kg_encoder_config: Dict = {...}
)
```

**Methods:**

#### `encode_text()`
```python
text_features = encoder.encode_text(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor
```

#### `encode_kg()`
```python
kg_features = encoder.encode_kg(
    node_features: torch.Tensor,
    adjacency_matrix: torch.Tensor
) -> torch.Tensor
```

---

## Algorithms

### KnowledgeGraphConstructor

Construct knowledge graphs from text with FIBO integration (Algorithm 1).

```python
from algorithms import KnowledgeGraphConstructor

constructor = KnowledgeGraphConstructor(
    fibo_path: str,
    max_nodes: int = 5000,
    max_edges: int = 15000
)
```

**Methods:**

#### `construct_knowledge_graph()`
```python
kg = constructor.construct_knowledge_graph(
    documents: List[str],
    extract_entities: bool = True,
    extract_relations: bool = True
) -> KnowledgeGraph
```

**Parameters:**
- `documents` (List[str]): Input documents
- `extract_entities` (bool): Whether to extract entities
- `extract_relations` (bool): Whether to extract relations

**Returns:**
- `KnowledgeGraph`: Constructed knowledge graph with nodes and edges

**Example:**
```python
documents = [
    "Apple Inc. reported earnings of $89.5B...",
    "CEO Tim Cook announced new product..."
]

kg = constructor.construct_knowledge_graph(documents)
print(f"Nodes: {len(kg.nodes)}, Edges: {len(kg.edges)}")
```

---

### HybridInference

Hybrid neural-symbolic inference (Algorithm 2).

```python
from algorithms import HybridInference

inference = HybridInference(
    model: HybridModel,
    num_reasoning_hops: int = 3
)
```

**Methods:**

#### `infer()`
```python
result = inference.infer(
    text: str,
    knowledge_graph: KnowledgeGraph,
    return_explanations: bool = True
) -> Dict
```

**Returns:**
- `summary` (str): Generated summary
- `reasoning_paths` (List): Multi-hop reasoning paths
- `confidence` (float): Confidence score
- `explanations` (Dict): Explanation components

---

### MESAFramework

Multi-stakeholder explainable summarization (Algorithm 3).

```python
from algorithms import MESAFramework

mesa = MESAFramework(
    stakeholder_profiles: List[str] = ["analyst", "compliance", "executive", "investment_manager"],
    use_rl_learning: bool = True
)
```

**Methods:**

#### `generate_explanation()`
```python
explanation = mesa.generate_explanation(
    summary: str,
    document: str,
    kg: KnowledgeGraph,
    stakeholder: str = "analyst"
) -> str
```

**Parameters:**
- `summary` (str): Generated summary
- `document` (str): Source document
- `kg` (KnowledgeGraph): Knowledge graph
- `stakeholder` (str): Target stakeholder role

**Returns:**
- `explanation` (str): Stakeholder-specific explanation

#### `update_with_feedback()`
```python
mesa.update_with_feedback(
    stakeholder: str,
    explanation: str,
    feedback_score: float
)
```

**Example:**
```python
# Generate for analyst
explanation = mesa.generate_explanation(
    summary=summary_text,
    document=document,
    kg=knowledge_graph,
    stakeholder="analyst"
)

# Update with feedback
mesa.update_with_feedback("analyst", explanation, feedback_score=0.85)
```

---

## Explainability

### MESAExplainer

Multi-stakeholder explanation system.

```python
from explainability import MESAExplainer

explainer = MESAExplainer()
```

**Stakeholder Profiles:**
- `analyst`: Financial analysis focus
- `compliance`: Regulatory compliance focus
- `executive`: High-level strategic focus
- `investment_manager`: Investment decision focus

**Methods:**

#### `generate_explanation()`
```python
explanation = explainer.generate_explanation(
    summary: str,
    document: str,
    kg: KnowledgeGraph,
    stakeholder: str
) -> str
```

---

### CausalExplainer

Causal chain extraction and explanation (CAUSAL-EXPLAIN framework).

```python
from explainability import CausalExplainer

explainer = CausalExplainer(
    max_chain_length: int = 5,
    confidence_threshold: float = 0.70
)
```

**Methods:**

#### `extract_causal_chains()`
```python
chains = explainer.extract_causal_chains(
    text: str
) -> List[List[Tuple[str, str, float]]]
```

**Returns:**
- List of causal chains, where each chain is a list of (cause, effect, confidence) tuples

**Example:**
```python
text = "Rate hikes led to market decline..."

chains = explainer.extract_causal_chains(text)
for chain in chains:
    for cause, effect, conf in chain:
        print(f"{cause} â†’ {effect} (confidence: {conf:.2f})")
```

---

### TemporalExplainer

Temporal consistency and market regime detection.

```python
from explainability import TemporalExplainer

explainer = TemporalExplainer()
```

**Methods:**

#### `detect_market_regime()`
```python
regime = explainer.detect_market_regime(
    text: str
) -> Dict[str, Any]
```

**Returns:**
- `regime` (str): Detected market regime (Bull/Bear/Crisis/Normal)
- `confidence` (float): Detection confidence
- `evidence` (List[str]): Supporting evidence

#### `compute_temporal_consistency()`
```python
tcc = explainer.compute_temporal_consistency(
    explanation_sequence: List[str]
) -> float
```

**Returns:**
- `tcc` (float): Temporal Consistency Coefficient (Equation 9)

---

## Knowledge Graph

### FIBOIntegration

FIBO ontology integration (2024-Q1).

```python
from knowledge_graph import FIBOIntegration

fibo = FIBOIntegration(
    ontology_path: str = "data/fibo/fibo_2024_q1.owl",
    load_modules: List[str] = ["FND_Agents", "BE_LegalEntities", "FBC_Products"]
)
```

**Statistics:**
- Total classes: 188 (FND/Agents 73, BE/LegalEntities 47, FBC/Products 68)
- Total properties: 363
- Modules: 4 core modules

**Methods:**

#### `get_class()`
```python
cls = fibo.get_class(
    class_name: str
) -> Optional[FIBOClass]
```

#### `link_entity()`
```python
linked = fibo.link_entity(
    entity_text: str,
    entity_type: str
) -> Optional[str]
```

**Example:**
```python
fibo = FIBOIntegration()

# Link entity to FIBO
fibo_id = fibo.link_entity("Apple Inc.", "ORG")
print(f"Linked to: {fibo_id}")

# Get class information
cls = fibo.get_class("Corporation")
print(f"Properties: {cls.properties}")
```

---

### EntityLinker

Link text entities to FIBO concepts.

```python
from knowledge_graph import EntityLinker

linker = EntityLinker(
    fibo: FIBOIntegration
)
```

**Methods:**

#### `link_entities_batch()`
```python
linked_entities = linker.link_entities_batch(
    entities: List[Tuple[str, str]],
    text: str
) -> List[LinkedEntity]
```

**Returns:**
- List of `LinkedEntity` objects with FIBO mappings

---

### CausalExtractor

Extract causal relationships from text.

```python
from knowledge_graph import CausalExtractor

extractor = CausalExtractor()
```

**Methods:**

#### `extract_causal_relations()`
```python
relations = extractor.extract_causal_relations(
    text: str
) -> List[CausalRelation]
```

**Returns:**
- List of `CausalRelation` objects with cause, effect, and confidence

---

## Utilities

### DocumentPreprocessor

Preprocess financial documents.

```python
from utils import DocumentPreprocessor

preprocessor = DocumentPreprocessor()
```

**Methods:**

#### `preprocess()`
```python
processed = preprocessor.preprocess(
    text: str,
    clean: bool = True,
    normalize: bool = True,
    segment: bool = True
) -> ProcessedDocument
```

**Returns:**
- `ProcessedDocument` with cleaned text, sentences, and statistics

---

### FinancialNER

Financial named entity recognition with FinBERT.

```python
from utils import FinancialNER

ner = FinancialNER(
    model_name: str = "ProsusAI/finbert"
)
```

**Methods:**

#### `extract_entities()`
```python
entities = ner.extract_entities(
    text: str
) -> List[Entity]
```

**Entity Types:**
- ORG: Organizations
- PERSON: People
- PRODUCT: Financial products
- MONEY: Monetary amounts
- PERCENT: Percentages
- DATE: Dates
- GPE: Geopolitical entities

---

### KnowledgeGraphUtils

Knowledge graph operations and queries.

```python
from utils import KnowledgeGraphUtils

kg_utils = KnowledgeGraphUtils()
```

**Methods:**

#### `find_paths()`
```python
paths = kg_utils.find_paths(
    graph: nx.DiGraph,
    source: str,
    target: str,
    max_hops: int = 3
) -> List[List[str]]
```

#### `multi_hop_reasoning()`
```python
reachable = kg_utils.multi_hop_reasoning(
    graph: nx.DiGraph,
    start_node: str,
    num_hops: int = 3
) -> Dict[int, Set[str]]
```

---

## Training

### MultiStageTrainer

Multi-stage training pipeline (Algorithm 6).

```python
from training import MultiStageTrainer

trainer = MultiStageTrainer(
    model: HybridModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config: Dict,
    device: str = 'cuda'
)
```

**Methods:**

#### `train()`
```python
trainer.train(
    output_dir: str
)
```

**Training Stages:**
1. Stage 1: Domain pre-training (3 epochs, LR=1e-4)
2. Stage 2: Knowledge integration (2 epochs, LR=5e-5)
3. Stage 3: Joint optimization (5 epochs, LR=2e-5)

**Example:**
```python
config = {
    'stage1_lr': 1e-4,
    'stage2_lr': 5e-5,
    'stage3_lr': 2e-5,
    'alpha': 0.72,
    'beta': 0.18,
    'gamma': 0.10
}

trainer = MultiStageTrainer(model, train_loader, val_loader, config)
trainer.train('checkpoints/')
```

---

### HyperparameterOptimizer

Bayesian hyperparameter optimization.

```python
from training import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(
    model_class: type,
    train_fn: Callable,
    eval_fn: Callable,
    n_trials: int = 50
)
```

**Methods:**

#### `optimize()`
```python
best_params = optimizer.optimize(
    direction: str = 'minimize'
) -> Dict
```

**Returns:**
- Dictionary of best hyperparameters

---

## Evaluation

### ComprehensiveEvaluator

All evaluation metrics in one place.

```python
from evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator()
```

**Methods:**

#### `evaluate_all()`
```python
metrics = evaluator.evaluate_all(
    predictions: List[str],
    references: List[str],
    sources: List[str],
    **kwargs
) -> Dict[str, float]
```

**Returns:**
- `rouge_l` (float): ROUGE-L F1 score
- `bertscore` (float): BERTScore F1
- `factual_consistency` (float): Factual consistency percentage
- `ssi` (float): Stakeholder Satisfaction Index (if data provided)
- `cps` (float): Causal Preservation Score (if data provided)
- `tcc` (float): Temporal Consistency Coefficient (if data provided)

---

### EvaluationPipeline

Complete evaluation workflow.

```python
from evaluation import EvaluationPipeline

pipeline = EvaluationPipeline(
    model: HybridModel,
    tokenizer: AutoTokenizer,
    device: str = 'cuda'
)
```

**Methods:**

#### `evaluate()`
```python
results = pipeline.evaluate(
    test_dataloader: DataLoader,
    output_dir: str,
    save_predictions: bool = True
) -> Dict[str, float]
```

---

### ErrorAnalyzer

Detailed error categorization.

```python
from evaluation import ErrorAnalyzer

analyzer = ErrorAnalyzer()
```

**Methods:**

#### `analyze()`
```python
error_report = analyzer.analyze(
    predictions: List[str],
    references: List[str],
    sources: List[str]
) -> Dict
```

**Error Categories:**
- Ten categories of Table 14 (Factual Inconsistency 14.2%, Temporal Ordering 9.1%,
  Causal Attribution 7.8%, Numerical Errors 6.3%, Entity Confusion 5.4%,
  Incomplete Context 5.1%, Stakeholder Mismatch 4.7%, Technical Terminology 3.9%,
  Hallucination 2.3%, Other 2.2%); per-summary rates, non-exclusive
- Totals: 56.5% with at least one error, 21.8% high-severity, 43.5% error-free

---

## Complete Example

```python
from models import HybridModel
from knowledge_graph import FIBOIntegration, KnowledgeGraphConstructor
from algorithms import HybridInference, MESAFramework
from transformers import AutoTokenizer

# 1. Initialize components
fibo = FIBOIntegration()
kg_constructor = KnowledgeGraphConstructor(fibo)
model = HybridModel()
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

# 2. Load checkpoint
model.load_state_dict(torch.load("checkpoints/your_trained_model.pt"))  # your own retrained weights
model.eval()

# 3. Prepare document
document = "Apple Inc. reported Q4 earnings of $89.5B..."

# 4. Construct knowledge graph
kg = kg_constructor.construct_knowledge_graph([document])

# 5. Generate summary
inputs = tokenizer(document, return_tensors="pt")
summary = model.generate(
    input_ids=inputs['input_ids'],
    kg_node_features=kg.node_features,
    kg_adjacency=kg.adjacency_matrix
)

# 6. Decode
summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)

# 7. Generate explanation
mesa = MESAFramework()
explanation = mesa.generate_explanation(
    summary=summary_text,
    document=document,
    kg=kg,
    stakeholder="analyst"
)

print(f"Summary: {summary_text}")
print(f"Explanation: {explanation}")
```

---

## Error Handling

All API methods raise appropriate exceptions:

```python
from exceptions import (
    ModelNotTrainedError,
    InvalidConfigError,
    KnowledgeGraphError,
    ExplanationGenerationError
)

try:
    model.generate(...)
except ModelNotTrainedError:
    print("Model not trained. Load checkpoint first.")
except KnowledgeGraphError as e:
    print(f"KG error: {e}")
```

---

## Type Hints

All functions include complete type hints:

```python
from typing import List, Dict, Optional, Tuple

def generate_summary(
    text: str,
    max_length: int = 128,
    num_beams: int = 4
) -> Tuple[str, float]:
    """Generate summary with confidence."""
    ...
```

---

Usage examples are inline in this reference. There is no separate `examples/` directory.
