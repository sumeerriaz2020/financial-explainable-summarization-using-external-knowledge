# Test Suite

Unit tests for the evaluation metrics, the Causal Preservation Score and the
knowledge-graph utilities.

---

## Scope

The suite covers the components listed below and passes against the released
code.

| File | Class | Tests |
|------|-------|-------|
| `test_metrics.py` | `TestROUGEMetrics` | ROUGE computation; perfect match |
| `test_metrics.py` | `TestFactualConsistencyMetrics` | consistency computation; hallucination detection |
| `test_metrics.py` | `TestSSIMetrics` | SSI computation (Equation 12); stakeholder weights fixed at 0.25; weighted average |
| `test_cps.py` | `TestCausalPreservationScore` | Equation 7: full, none and partial preservation; empty source undefined; duplicates cannot exceed 1; direction matters; claim normalization |
| `test_cps.py` | `TestWeightedCausalPreservationScore` | Equation 8: hand-computed example; equal weights reduce to CPS; `importance_weight` attribute; zero weights undefined; misaligned weights rejected |
| `test_cps.py` | `TestAggregateCPS` | exclusion of undefined documents from the mean; per-document weights; all-excluded gives NaN |
| `test_kg_integration.py` | `TestCausalExtractor` | extraction of multiple causal relations |
| `test_kg_integration.py` | `TestTemporalAnnotator` | temporal expression detection |

**Total:** 24 tests (9 in `test_metrics.py` and `test_kg_integration.py`, 15 in
`test_cps.py`). `test_cps.py` needs only the standard library.

### Not covered

No tests are included for Algorithms 1–6, the five explainability components, the
model classes, BERTScore or TCC. Components that require downloading pre-trained
models (FinBERT, BART-large, the BERTScore scorer) are not unit-tested, and
`TCCMetrics` is not tested because it falls back to random embeddings when none
are supplied, which makes its output non-deterministic.

---

## Running Tests

From the repository root:

```bash
pip install -r requirements.txt
pip install pytest
python -m pytest test/ -q
```

Or with unittest:

```bash
python -m unittest discover test -v
```

The suite needs no GPU and no network access.

---

## Verified environment

The 9 tests were run ten times in succession with all passing. The environment
used was Python 3.13, PyTorch 2.14 (CPU), transformers 5.17, networkx 3.6,
spaCy 3.8, rouge-score and bert-score, with pytest 9.1. They have not been run
under the pinned reference environment in `environment.yml` (Python 3.10,
PyTorch 2.0.1).

Test coverage and execution time have not been measured.

---

## Writing New Tests

```python
import unittest
from training_evaluation.metrics import SSIMetrics

class TestExample(unittest.TestCase):
    def setUp(self):
        self.metric = SSIMetrics()

    def test_feature(self):
        result = self.metric.stakeholder_weights
        self.assertAlmostEqual(sum(result.values()), 1.0)

if __name__ == '__main__':
    unittest.main()
```

Import from the sub-modules (`training_evaluation.metrics`,
`knowledge_graph.causal_extraction`, …); the packages' `__init__.py` files do not
re-export classes.
