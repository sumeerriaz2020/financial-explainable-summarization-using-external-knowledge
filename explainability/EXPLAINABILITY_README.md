# Explainability Frameworks

Five integrated explainability framework components for stakeholder-aware, transparent financial summarization.

---

## 📁 Files (40 KB total)

| File | Size | Framework | Improvement |
|------|------|-----------|-------------|
| `mesa.py` | 5.5 KB | Multi-Stakeholder Explanation | +21.3% SSI (0.61→0.74) |
| `causal_explain.py` | 7.2 KB | Causal Chain Preservation | +50.0% CPS (0.34→0.51) |
| `adapt_eval.py` | 9.2 KB | Adaptive Evaluation | +21.2% Trust (0.52→0.63) |
| `consensus.py` | 8.5 KB | Ensemble Selection | +37.2% Consistency (0.43→0.59) |
| `temporal_explain.py` | 9.5 KB | Temporal Consistency | +42.1% TCC (0.38→0.54) |

---

## 🎯 Framework Descriptions

### 1. MESA (`mesa.py`)
**Multi-stakeholder Explainable Summarization Architecture**

**Purpose:** Generate explanations optimized for different user roles

**Key Features:**
- 4 stakeholder types: analyst, compliance, executive, investment manager
- Online weight update is a simplified step; the REINFORCE-style update of
  Section 3.4.1 is specified in the paper, not included in this release
- Reinforcement learning for preference adaptation
- Multi-objective optimization (Equation 5)
- Dynamic weight updates based on feedback

**Usage:**
```python
from explainability.mesa import MESAExplainer, StakeholderProfile

mesa = MESAExplainer()
profile = mesa.profiles['analyst']

explanation = mesa.generate_explanation(
    summary, document, kg, profile
)

# Update with user feedback
mesa.update_with_feedback(profile, explanation, feedback_score=0.85)
```

---

### 2. CAUSAL-EXPLAIN (`causal_explain.py`)
**Causal Chain Extraction and Preservation**

**Purpose:** Preserve source-attributed causal claims (cause-effect statements asserted in the source)

**Key Metrics:**
- CPS (standard): Equation 7
- CPS (weighted): Equation 8
- Causal accuracy, coherence, temporal fidelity

**Usage:**
```python
from explainability.causal_explain import CausalExplainer

causal = CausalExplainer(causal_threshold=0.7)

result = causal.extract_and_preserve(
    document, kg, summary_budget=512
)

print(f"Selection CPS: {result['selection_cps']}")
print(f"Preserved: {len(result['preserved_chains'])} chains")
```

`selection_cps` measures the share of source chains selected for the summary
within the length budget (Algorithm 4). To score what a generated summary actually
contains — the CPS reported in Table 4 — extract the summary's causal claims and use
`training_evaluation.metrics.CPSMetrics` (or `training_evaluation.cps`).

---

### 3. ADAPT-EVAL (`adapt_eval.py`)
**Adaptive Real-Time Explanation Evaluation**

**Purpose:** Continuous learning from user interactions

**Key Features:**
- Multi-modal feedback: explicit ratings + implicit signals
- Individual user modeling
- Quality predictors for 3 dimensions
- Trust calibration metric

**Usage:**
```python
from explainability.adapt_eval import AdaptiveEvaluator

evaluator = AdaptiveEvaluator(learning_rate=0.05)

# Collect feedback
feedback = evaluator.collect_feedback(
    explanation, user_id="user_123",
    explicit_rating=0.8,
    dwell_time=45.5,
    interactions=['detailed_view', 'verify_sources']
)

# Adapt model
evaluator.adapt_model(feedback)

# Select best explanation
best, predictions = evaluator.select_explanation(
    candidates, user_id="user_123", context={}
)
```

---

### 4. CONSENSUS (`consensus.py`)
**INTERPRETABLE-CONSENSUS - Ensemble Selection**

**Purpose:** Transparent explanation selection with audit trails

**Key Features:**
- 4 generation methods: template, retrieval, neural, hybrid
- Weighted scoring across 4 dimensions
- Automated conflict resolution
- Complete audit trail for compliance

**Usage:**
```python
from explainability.consensus import ConsensusExplainer

consensus = ConsensusExplainer(
    score_weights={
        'accuracy': 0.30,
        'linguistic_quality': 0.25,
        'stakeholder_fit': 0.25,
        'compliance': 0.20
    }
)

explanation, audit = consensus.select_consensus(
    summary, document, kg, stakeholder
)

print(f"Method: {audit.selection_method}")
print(f"Conflicts: {len(audit.conflicts)}")
print(f"Uncertainty: {audit.uncertainty}")
```

---

### 5. TEMPORAL-EXPLAIN (`temporal_explain.py`)
**Temporal Consistency Maintenance**

**Purpose:** Maintain consistency across changing market conditions

**Key Features:**
- Market regime detection (bull, bear, sideways; crisis conditions flagged). Keyword
  heuristic: the HMM and Page-Hinkley drift test of Section 3.4.5 are specified in
  the paper, not included in this release
- Regime-aware explanation adaptation
- Temporal Consistency Coefficient (TCC) - Equation 9
- Concept drift detection

**Usage:**
```python
from explainability.temporal_explain import TemporalExplainer, MarketRegime

temporal = TemporalExplainer()

# Generate with temporal consistency
explanation, consistency = temporal.generate_with_temporal_consistency(
    summary, document, current_regime=None
)

# Compute TCC
tcc = temporal.compute_tcc(window=10)
print(f"TCC: {tcc:.3f}")

# Get consistency report
report = temporal.get_consistency_report()
print(report)
```

---

## 🔗 Integration Example

```python
# Complete explainability pipeline
from explainability import (
    MESAExplainer,
    CausalExplainer,
    AdaptiveEvaluator,
    ConsensusExplainer,
    TemporalExplainer
)

# 1. Extract causal chains
causal = CausalExplainer()
causal_result = causal.extract_and_preserve(document, kg)

# 2. Generate stakeholder-specific explanation
mesa = MESAExplainer()
profile = mesa.profiles['analyst']
mesa_explanation = mesa.generate_explanation(summary, document, kg, profile)

# 3. Select consensus explanation
consensus = ConsensusExplainer()
consensus_explanation, audit = consensus.select_consensus(
    summary, document, kg, profile
)

# 4. Ensure temporal consistency
temporal = TemporalExplainer()
final_explanation, tcc = temporal.generate_with_temporal_consistency(
    summary, document, current_regime=None
)

# 5. Collect feedback and adapt
evaluator = AdaptiveEvaluator()
feedback = evaluator.collect_feedback(
    final_explanation, user_id, explicit_rating=0.85
)
evaluator.adapt_model(feedback)
mesa.update_with_feedback(profile, final_explanation, 0.85)
```

---

## 📊 Performance Metrics

| Framework | Metric | Baseline | Our System | Improvement |
|-----------|--------|----------|------------|-------------|
| MESA | SSI | 0.61±0.07 | 0.74±0.06 | +21.3% |
| CAUSAL | CPS | 0.34±0.08 | 0.51±0.07 | +50.0% |
| ADAPT-EVAL | Trust Cal. | 0.52±0.09 | 0.63±0.08 | +21.2% |
| CONSENSUS | Expl. Cons. | 0.43±0.07 | 0.59±0.06 | +37.2% |
| TEMPORAL | TCC | 0.38±0.06 | 0.54±0.07 | +42.1% |

---

## ⚙️ Dependencies

```txt
numpy>=1.24.0
```

---

## ✅ Repository component

Part of the `explainability/` directory.
