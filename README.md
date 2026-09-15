# Financial Explainable Summarization with Hybrid Neural-Symbolic AI

[![DOI](https://zenodo.org/badge/1371339971.svg)](https://doi.org/10.5281/zenodo.22769416)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0.1](https://img.shields.io/badge/PyTorch-2.0.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference implementation for **"An eXplainable Approach to Abstractive Text
Summarization Using External Knowledge: A Novel Framework for Financial Domain
Applications"** — a FIBO-grounded hybrid neural-symbolic framework for explainable
summarization of financial documents.

**Authors:** Sumeer Riaz, M. Bilal Bashir, Syed Ali Hassan Naqvi
**Affiliation:** IQRA University Islamabad H9 Campus, Pakistan
**Manuscript:** ESWA-D-26-12626, *Expert Systems with Applications* (under review)

---

## What this is, and what it is not

**This is** a reference implementation of the architecture and the five
explainability components, the configurations behind every reported number,
the GPT-4 prompt templates, a public-domain data sample with corpus rebuild
scripts, and a documented account of the experimental protocol.

---

## Contents

- [Reported results](#reported-results)
- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Reproducing each reported table](#reproducing-each-reported-table)
- [Checkpoints](#checkpoints)
- [Dataset](#dataset)
- [Limitations and scope](#limitations-and-scope)
- [Citation](#citation)

---

## Reported results

All figures are test-set values over five seeded runs. The canonical extract of
every reported number is [`docs/PAPER_FACTS.md`](docs/PAPER_FACTS.md).

### Summarization quality — Table 3

| Metric | Baseline BART | BART + KG | Ours |
|---|---|---|---|
| ROUGE-L | 0.421 ± 0.018 | 0.438 ± 0.015 | **0.497 ± 0.014** |
| ROUGE-1 | 0.387 ± 0.022 | 0.401 ± 0.019 | **0.452 ± 0.016** |
| ROUGE-2 | 0.184 ± 0.015 | 0.195 ± 0.013 | **0.227 ± 0.012** |
| BERTScore | 0.612 ± 0.025 | 0.634 ± 0.021 | **0.673 ± 0.018** |
| BLEU-4 | 0.156 ± 0.012 | 0.167 ± 0.011 | **0.191 ± 0.009** |

### Explainability — Table 4

| Component | Metric | Baseline | Ours | Δ |
|---|---|---|---|---|
| MESA | SSI | 0.61 ± 0.07 | **0.74 ± 0.06** | +21.3% |
| CAUSAL-EXPLAIN | CPS | 0.34 ± 0.08 | **0.51 ± 0.07** | +50.0% |
| ADAPT-EVAL | Trust Cal. | 0.52 ± 0.09 | **0.63 ± 0.08** | +21.2% |
| CONSENSUS | Expl. Cons. | 0.43 ± 0.07 | **0.59 ± 0.06** | +37.2% |
| TEMPORAL-EXPLAIN | TCC | 0.38 ± 0.06 | **0.54 ± 0.07** | +42.1% |

### Scope of the contribution

Three points the paper makes explicitly:

- **Expert preference over GPT-4 was 54% (p = 0.35) — indistinguishable from
  chance.** GPT-4 also scores *higher* on BERTScore (0.679 vs 0.673). The
  contribution is **auditability and traceability**, not preference superiority.
- **Not all five components are algorithmically novel.** The novelty is their joint
  formalization and integration into a FIBO-grounded pipeline; several adapt
  established techniques (contextual bandits, ensemble consensus, HMM regime
  detection, drift detection).
- **Compliance is design-level only.** Table 16 is a qualitative self-assessment,
  explicitly not legal certification. No independent legal review has taken place.

---

## Installation

Exact versions are pinned in [`requirements.txt`](requirements.txt) and
[`environment.yml`](environment.yml).

```bash
git clone https://github.com/sumeerriaz2020/financial-explainable-summarization-using-external-knowledge.git
cd financial-explainable-summarization-using-external-knowledge

conda env create -f environment.yml
conda activate fin-explainable
pip install -e .
```

Or with pip only:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Verify

```bash
python -c "import models, algorithms, explainability, knowledge_graph; print('ok')"
pytest test/ -q
```

Reference environment: Python 3.10, PyTorch 2.0.1, Transformers 4.35.0,
NetworkX 3.1, MLflow 2.8.0, CUDA 11.8.

---

## Repository layout

```
.
├── algorithms/            Algorithms 1-6 from the paper
├── models/                Dual encoder, KG encoder, cross-modal attention, hybrid model
├── explainability/        MESA, CAUSAL-EXPLAIN, ADAPT-EVAL, CONSENSUS, TEMPORAL-EXPLAIN
├── knowledge_graph/       FIBO integration, entity linking, causal + temporal extraction
│   └── fibo-ext-temporal-2024Q1.owl
├── training_evaluation/   Training, evaluation, metrics, baselines, error analysis
├── utils/                 Preprocessing, NER, KG utilities, visualisation
├── configs/               Every configuration behind a reported result  → configs/README.md
├── prompts/               GPT-4 baseline templates (Appendix A)
├── data/                  Public sample + collection scripts  → data/data_statement.md
├── docs/PAPER_FACTS.md    Canonical extract of every reported number
├── documentation/         API reference, deployment notes
├── test/                  Unit tests
└── CITATION.cff           Citation metadata
```

---

## Reproducing each reported table

Each configuration is named for the result it produces. The full map is in
[`configs/README.md`](configs/README.md).

| Paper artifact | Config | Code |
|---|---|---|
| Table 3 — summarization quality | `configs/training_config.yaml` | `training_evaluation/train.py`, `evaluate.py` |
| Table 4 — explainability | `configs/training_config.yaml` | `explainability/`, `training_evaluation/metrics.py` |
| Table 5 — efficiency | `configs/evaluation.yaml` | `training_evaluation/evaluate.py` |
| Table 6 — SOTA comparison | `configs/baselines.yaml` | `training_evaluation/baseline_comparison.py` |
| Tables 7, 8 + Figure 5 — ablation | `configs/ablation.yaml` | `training_evaluation/train.py` |
| Table 14 + Figure 6 — error analysis | `configs/evaluation.yaml` | `training_evaluation/error_analysis.py` |
| Table 1 — thresholds | `configs/model_config.yaml` | `knowledge_graph/` |
| Appendix A — GPT-4 prompts | `prompts/gpt4_baseline.yaml` | *no driver — see below* |

### ⚠️ Before you try to run these

**These commands do not currently execute end to end.** `train.py` does not parse
arguments or read the YAML configs, and no data-loading layer exists to turn the
files in `data/` into batches. The configurations above are a complete and accurate
*specification* of each experiment; the wiring to run them is not present.

What **does** run today:

```bash
pytest test/ -q                                    # 9 unit tests (metrics, KG utilities)
python -c "import yaml; print(yaml.safe_load(open('configs/training_config.yaml'))['loss_weights'])"
                                                   # -> {'alpha': 0.72, 'beta': 0.18, 'gamma': 0.1}
```

Metric implementations in `training_evaluation/metrics.py` (ROUGE, BERTScore, CPS,
SSI, TCC) can be called directly on your own summary/reference pairs. The test
suite is deliberately small; see [`test/README.md`](test/README.md) for what it
covers and what it does not.

---

## Checkpoints

**Trained model checkpoints from the original experiments were not retained and are
not available for release.**

Two accuracy caveats, so this is not read as stronger than it is:

- Seeds make the procedure repeatable but **do not guarantee identical results**
  across different GPUs, drivers or CUDA versions. A re-run is an independent
  replication, not an exact reproduction.
- Retraining also requires the corpus, which **cannot be fully redistributed**
  (see [Dataset](#dataset)). A rebuilt corpus is similar in construction but not
  identical in content.

The blocking gap for retraining today is the missing data-loading and CLI layer
described above, not the absence of weights.

---

## Dataset

Full statement: [`data/data_statement.md`](data/data_statement.md).

The paper's corpus is **215,500 documents** (223,500 compiled, filtered to 215,500).
It is **not redistributable** — financial news, the largest component at 156,000
documents, is publisher copyright.

Shipped instead (~85 MB, real documents, not synthetic):

| Source | Included | Redistributable |
|---|---|---|
| SEC filings | 21 filings (2023) + full index | ✅ publicly available (EDGAR) |
| Earnings releases | 52 releases, AAPL/GOOGL/MSFT, 2022–2025 (8 from 2025 post-date the corpus period; illustrative only) | ✅ issuer-published |
| FOMC minutes | 5 documents (2020) | ✅ publicly available (Federal Reserve Board) |
| Regulatory indices | SEC, FINRA, Federal Reserve | ✅ identifiers |
| Financial news | **metadata only** — headline, URL, timestamp | ❌ text is copyright |

Rebuild the public portion with the 12 scripts in `data/data_collection_scripts/`.

**FIBO is not vendored.** Obtain version 2024-Q1 from
<https://spec.edmcouncil.org/fibo/> (MIT licence, EDM Council). Only the custom
extension ships here: `knowledge_graph/fibo-ext-temporal-2024Q1.owl`.

---

## Limitations and scope

### Model limitations (from the paper)

| Limitation | Figure |
|---|---|
| Total error rate | 56.5% (43.5% error-free) |
| High-severity errors requiring human review | 21.8% |
| Residual factual errors | 12.4% |
| Causal claims not preserved | 49% (CPS 0.51) |
| Documents > 50 pages | −18% ROUGE-L, 68% error rate |
| Multilingual documents | −31% all metrics, 82% error rate |
| Inference overhead | +21% time, +19.5% memory |

**The system is not suitable for autonomous deployment.** Human review is mandatory
for high-severity errors. Documents over 50 pages cannot support autonomous
processing.

### Explicitly out of scope for this repository

Treated as future work:

1. **Runnable training pipeline** — data loader, config parser and CLI entry point
   are not implemented.
2. **GPT-4 baseline execution** — templates only, no API driver.
3. **Trained weights** — not retained.
4. **Mechanisms specified in the paper but not included in this release** — the
   three-state HMM market-regime detector and Page-Hinkley drift detection
   (Section 3.4.5), and the REINFORCE-style MESA weight update (Section 3.4.1).
   The code uses a keyword regime heuristic, a fixed TCC threshold and a
   simplified reward-scaled update as stand-ins. Likewise, the 20% extension-entity
   confidence penalty and the τ_threshold = 0.65 edge filter (Table 1) are recorded
   in `configs/model_config.yaml` but not wired into the code.
5. **Current-generation LLM baselines** — the `gpt-4-0613` snapshot was frozen with
   the evaluation protocol; re-benchmarking is future work.
6. **Dedicated RAG / KG-RAG comparators** — BART+KG is the retrieval-augmented
   control but is not a substitute for purpose-built pipelines.
7. **FinBen evaluation** — identified as a priority extension.
8. **Independent legal audit** — required before any compliance claim.

### Known gaps in the released artifacts

1. Trained checkpoints were not retained and are not available.
2. No executable end-to-end training pipeline — `train.py` neither parses
   arguments nor reads the configs, and no data-loading layer exists.
3. The GPT-4 baseline has prompt templates but no API driver; its decoding
   parameters and the identity of the five exemplars were not recorded.
4. The 100-document expert-curated gold standard is not included.
5. Processed train/validation/test split files were not retained.
6. Experimental logs were not retained; the MLflow artifacts referenced in
   Section 3.6.3 could not be recovered.

---

## Citation

```bibtex
@article{riaz2026explainable,
  title   = {An eXplainable Approach to Abstractive Text Summarization Using
             External Knowledge: A Novel Framework for Financial Domain Applications},
  author  = {Riaz, Sumeer and Bashir, M. Bilal and Naqvi, Syed Ali Hassan},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Manuscript under review (ESWA-D-26-12626)}
}
```

To cite this software, use the concept DOI, which always resolves to the latest
version ([10.5281/zenodo.22769416](https://doi.org/10.5281/zenodo.22769416)):

```bibtex
@software{riaz2026explainable_code,
  title     = {An eXplainable Approach to Abstractive Text Summarization Using
               External Knowledge: A Novel Framework for Financial Domain Applications},
  author    = {Riaz, Sumeer and Bashir, M. Bilal and Naqvi, Syed Ali Hassan},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22769416},
  url       = {https://doi.org/10.5281/zenodo.22769416}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

---

## License

MIT — see [`LICENSE`](LICENSE). Third-party components (BART, FinBERT, FIBO) retain
their own licences. Source documents are not redistributed beyond the public-domain
sample described above.

## Contact

Sumeer Riaz — sumeer33885@iqraisb.edu.pk
M. Bilal Bashir — bilal.bashir@iqraisb.edu.pk
Syed Ali Hassan Naqvi — ali33884@iqraisb.edu.pk

## Acknowledgments

FIBO (EDM Council) · Hugging Face Transformers · PyTorch · NetworkX
