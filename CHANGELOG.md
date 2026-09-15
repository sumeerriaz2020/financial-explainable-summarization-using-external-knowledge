# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0]

Initial public release, accompanying the manuscript *An eXplainable Approach to
Abstractive Text Summarization Using External Knowledge: A Novel Framework for
Financial Domain Applications* (*Expert Systems with Applications*,
ESWA-D-26-12626).

### Included

- Reference implementation of the hybrid neural-symbolic architecture: dual
  encoder, knowledge-graph encoder, cross-modal attention and hybrid model
  (`models/`).
- Algorithms 1–6 (`algorithms/`), with `algorithm_6_training.py` as the
  multi-stage trainer (linear warm-up over the first 10% of steps, ×0.9 per-epoch
  decay in Stage 3) and `training_evaluation/train.py` as a thin wrapper.
- The five explainability components: MESA, CAUSAL-EXPLAIN, ADAPT-EVAL,
  INTERPRETABLE-CONSENSUS, TEMPORAL-EXPLAIN (`explainability/`).
- FIBO integration for the three modules of Section 3.2.2, entity linking, causal
  extraction and temporal annotation, and the extension ontology
  `knowledge_graph/fibo-ext-temporal-2024Q1.owl`.
- Evaluation code: ROUGE, BERTScore, factual consistency, SSI, TCC, error
  analysis (Table 14 taxonomy), baseline values (Tables 3, 4, 6, 8), and the
  Causal Preservation Score (Equations 7–8) in `training_evaluation/cps.py`.
- Configurations mapped to the reported tables (`configs/`) and the reported
  values in one place (`docs/PAPER_FACTS.md`).
- GPT-4 prompt templates from Appendix A (`prompts/`).
- Data statement, a public-domain data sample, document indices and collection
  scripts (`data/`).
- Unit tests for the metrics, CPS and knowledge-graph utilities (`test/`, 24 tests).
- `CITATION.cff`.

### Known gaps

Listed in full in the root README:

1. Trained checkpoints were not retained and are not available.
2. No executable end-to-end training pipeline — `train.py` neither parses
   arguments nor reads the configs, and no data-loading layer exists.
3. The GPT-4 baseline has prompt templates but no API driver.
4. The HMM regime detector, Page-Hinkley drift detection and REINFORCE-style MESA
   update are specified in the paper but not included in this release.
5. The 100-document expert-curated gold standard is not included.
6. Processed train/validation/test split files and experimental logs were not
   retained.
