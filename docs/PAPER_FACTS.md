# Reported values from the manuscript

The values reported in *An eXplainable Approach to Abstractive Text Summarization
Using External Knowledge: A Novel Framework for Financial Domain Applications*
(*Expert Systems with Applications*, ESWA-D-26-12626), collected in one place.
The configuration files and READMEs in this repository use these values.

---

## 1. Corpus and splits

| Item | Value |
|---|---|
| Documents compiled | 223,500 (2015–2024) |
| — SEC filings | 47,000 |
| — Earnings call transcripts | 12,000 |
| — Financial news | 156,000 |
| — Regulatory documents | 8,500 |
| After quality filtering | 215,500 |
| Split design | **Single temporal holdout** (not k-fold, not 70/15/15) |
| — Train | 2015–2022 |
| — Validation | tail of 2022 |
| — Test | 2023–2024 |
| Knowledge graph | Built **exclusively from the training partition**, then frozen |
| Entity-level disjointness | **Not** enforced across periods (issuers may recur) |
| Expert-curated evaluation subset | 100 documents, stratified by document type, drawn from 2023–2024 |
| Error-analysis sample | 500 summaries |
| Efficiency-measurement sample | 1,000 documents |

Arithmetic check: 47,000 + 12,000 + 156,000 + 8,500 = 223,500. ✔

### Reference summary construction

- **Training targets (weak supervision, corpus-wide):** financial news → article
  lead paragraph; SEC filings → first five sentences of MD&A (Item 7); earnings
  calls → closing management summary (final two to three paragraphs of prepared
  remarks).
- **Evaluation gold standard:** the three CFA-certified analysts and two
  compliance officers curated and factually verified the 100-document subset,
  correcting numerical and causal inaccuracies. Tables 3 and 10 are scored
  against this.

### Annotation team (8 people — distinct from the 12-expert evaluation panel)

3 senior financial analysts (CFA), 2 regulatory compliance officers, 1 risk
management specialist, 2 NLP researchers. Double annotation on 30% of samples,
Cohen's κ > 0.75.

---

## 2. Knowledge graph

| Item | Value |
|---|---|
| Ontology | FIBO version **2024-Q1** (full: 487 classes, 970 properties) |
| Modules used | FND/Agents (73 classes, 152 properties); BE/LegalEntities (47, 84); FBC/Products (68, 127) |
| Combined | **188 classes, 363 properties** |
| Entity-mention coverage | ~78% |
| Extension file | `fibo-ext-temporal-2024Q1.owl` (separate OWL importing FIBO; FIBO itself not edited); shipped at `knowledge_graph/` |
| Entity instance nodes | ~2.4M (2.1M FIBO-mapped + 0.3M extension) |
| Edges | 8.7M total, incl. 1.2M temporal + 340K causal |
| Average query time | 127 ± 18 ms |
| Extension-entity confidence penalty | 20% |

Coverage breakdown: 78% map directly to FIBO; 12–15% require custom extension
classes; 7–10% generic/ambiguous.

Confidence thresholds (Table 1): `τ_threshold` = **0.65** (precision 0.78 /
recall 0.71), `τ_causal` = **0.70**, `τ_min` = **0.35**.

---

## 3. Model architecture

| Item | Value |
|---|---|
| Backbone | `facebook/bart-large`, **406M** parameters, stock BART tokenizer |
| Max input length | 1024 tokens |
| KG encoder | 3-layer graph-attention network, 8 heads |
| — dimensions | 768 (entity emb.) → 512 (hidden) → 1024 (output, matches BART hidden) |
| Cross-modal fusion | 768 dims, 8 attention heads |
| Per-document subgraph cap | 5,000 nodes / 15,000 edges |
| Total parameters | **~416M** (406M backbone + ~10M KG encoder and fusion) |
| Reasoning hops (`N_hops`) | 3 |
| Decoding | Beam search, beam width 4, length penalty 2.0, max output 128 tokens |

---

## 4. Training configuration

| Item | Value |
|---|---|
| Composite loss weights (Eq. 10) | α = **0.72** (sum), β = **0.18** (kg), γ = **0.10** (expl); α+β+γ = 1, γ derived as 1−α−β |
| Batch size | 16 |
| Dropout | 0.15 |
| λ (Eq. 3, KG attention) | 0.5 (Bayesian search) |
| Stage learning rates | Stage 1: 1×10⁻⁴ · Stage 2: 5×10⁻⁵ · Stage 3: 2×10⁻⁵ |
| Epochs per stage | 3 / 2 / 5 (**10 total**) |
| Optimizer | AdamW, β₁ = 0.9, β₂ = 0.999, weight decay 0.01 |
| Warm-up | Linear over first 10% of steps |
| Stage-3 LR decay | ×0.9 per epoch |
| Gradient clipping | norm 1.0 |
| Early stopping | patience 2 epochs on validation performance |
| Validation ROUGE-L | 0.487 |
| Validation BERTScore | 0.686 |
| Seeds | **[42, 123, 456, 789, 1011]** — five independent runs |

Bayesian optimization search space: α and β (γ derived); learning rates
[1e-5, 2e-4]; batch size {8, 16, 32}; hidden dim {512, 768, 1024}; attention
heads {4, 8, 12}; reasoning hops {2, 3, 4}; dropout [0.1, 0.3].

---

## 5. Compute and environment

| Item | Value |
|---|---|
| Training hardware | **4× NVIDIA A100 (80 GB)** |
| Training wall-clock | **27 hours** → **108 total GPU-hours** |
| Estimated cost | **$885** (AWS p4d.24xlarge @ $32.77/hour × 27 h) |
| Stage breakdown | Stage 1: 3 epochs / 4 h · Stage 2: 2 epochs / 5 h · Stage 3: 5 epochs / 18 h |
| Per-epoch cost | 1.3 h / 2.5 h / 3.6 h |
| Inference hardware | 2× NVIDIA V100 (32 GB) |
| Inference time | 2.9 ± 0.4 s per document (incl. 127 ± 18 ms KG query) |
| Peak memory | 9.8 ± 0.7 GB per document |
| Trained weights | **NOT retained** (storage constraints) |

Software stack: PyTorch 2.0.1 · Transformers 4.35.0 · NetworkX 3.1 ·
MLflow 2.8.0 · Docker 24.0.5 · Git 2.42.0 · DVC 3.30.0.

---

## 6. Baseline configurations

| # | Baseline | Configuration |
|---|---|---|
| 1 | BART-large | 406M, lr 3e-5, bs 16, 10 epochs |
| 2 | PEGASUS-large | 568M, lr 5e-5, bs 8, 8 epochs |
| 3 | T5-base | 220M, lr 1e-4, bs 32, 12 epochs |
| 4 | FinanceSum | Tang et al., 2023 |
| 5 | GPT-4 | `gpt-4-0613`, zero-shot and 5-shot (frozen snapshot; explicitly not current SOTA) |
| 6 | BART + KG | lr 3e-5, bs 16, 10 epochs, early stopping on validation ROUGE-L |

**BART+KG control (must match the proposed system on all but fusion):** identical
FinBERT NER → FIBO entity linking (Algorithm 1, lines 3–6); 3-hop traversal;
same frozen training-only KG snapshot (2015–2022); linearized triples
concatenated as a `[SEP]`-separated prefix; stock BART-large decoding with no
cross-modal attention, no path encoder, no fused hidden state; same tuning
search space and budget as baseline 1.

---

## 7. Reported results

### Table 3 — Summarization quality (mean ± SD over 5 seeded runs)

| Metric | Baseline | BART+KG | Ours | p | Cohen's d |
|---|---|---|---|---|---|
| ROUGE-L | 0.421 ± 0.018 | 0.438 ± 0.015 | **0.497 ± 0.014** | <0.001 | 0.68 |
| ROUGE-1 | 0.387 ± 0.022 | 0.401 ± 0.019 | **0.452 ± 0.016** | <0.001 | 0.54 |
| ROUGE-2 | 0.184 ± 0.015 | 0.195 ± 0.013 | **0.227 ± 0.012** | <0.001 | 0.62 |
| BERTScore | 0.612 ± 0.025 | 0.634 ± 0.021 | **0.673 ± 0.018** | <0.001 | 0.58 |
| BLEU-4 | 0.156 ± 0.012 | 0.167 ± 0.011 | **0.191 ± 0.009** | <0.001 | 0.61 |

Against BART+KG specifically: ROUGE-L p < 0.01, d = 0.41; ROUGE-2 p < 0.01, d = 0.38.

### Table 4 — Explainability components

| Framework | Metric | Baseline | Ours | Δ / p |
|---|---|---|---|---|
| MESA | SSI | 0.61 ± 0.07 | 0.74 ± 0.06 | +21.3% / p < 0.001 |
| CAUSAL-EXPLAIN | CPS | 0.34 ± 0.08 | 0.51 ± 0.07 | +50.0% / p < 0.001 |
| ADAPT-EVAL | Trust Cal. | 0.52 ± 0.09 | 0.63 ± 0.08 | +21.2% / p < 0.01 |
| CONSENSUS | Expl. Cons. | 0.43 ± 0.07 | 0.59 ± 0.06 | +37.2% / p < 0.001 |
| TEMPORAL | TCC | 0.38 ± 0.06 | 0.54 ± 0.07 | +42.1% / p < 0.01 |

Baselines are per-metric best: BART+KG for SSI and CPS; BART-large for Trust
Calibration, Explanation Consistency and TCC.

### Table 5 — Computational efficiency (N = 1,000 documents)

| Metric | Baseline | Ours | Change |
|---|---|---|---|
| Inference time / doc | 2.4 ± 0.3 s | 2.9 ± 0.4 s | +21% |
| Peak memory | 8.2 ± 0.6 GB | 9.8 ± 0.7 GB | +19.5% |
| Explanation generation | N/A | 18 ± 3 s | new |
| KG query time | N/A | 127 ± 18 ms | new |
| Training time | 18 ± 2 h | 27 ± 3 wall-clock h | +50% |

### Table 6 — State-of-the-art comparison

| System | ROUGE-L | BERTScore | Expl. (1–10) | Compliance |
|---|---|---|---|---|
| BART + LIME | 0.438 ± 0.019 | 0.634 ± 0.023 | 3.2 ± 0.8 | Partial |
| PEGASUS + Attn | 0.442 ± 0.017 | 0.641 ± 0.021 | 4.1 ± 0.9 | Partial |
| T5 + SHAP | 0.445 ± 0.018 | 0.638 ± 0.022 | 3.8 ± 0.7 | Partial |
| FinanceSum | 0.451 ± 0.016 | 0.652 ± 0.019 | 4.5 ± 0.8 | Partial |
| GPT-4 (0613, 5-shot) | 0.471 ± 0.017 | **0.679 ± 0.020** | 5.8 ± 0.8 | None |
| Ours | **0.497 ± 0.022** | 0.673 ± 0.028 | **6.7 ± 0.7** | Designed |

Note: GPT-4 **beats** the proposed system on BERTScore (0.679 vs 0.673, Δ = −0.9%).

### Table 7 — Ablation (sequential addition — NOT leave-one-out)

| Configuration | ROUGE-L | BERTScore | SSI | CPS |
|---|---|---|---|---|
| Base Transformer | 0.421 | 0.612 | 0.61 | 0.34 |
| + Knowledge Graph | 0.438 | 0.634 | 0.65 | 0.41 |
| + MESA | 0.451 | 0.647 | 0.72 | 0.43 |
| + CAUSAL-EXPLAIN | 0.459 | 0.656 | 0.73 | 0.48 |
| + ADAPT-EVAL | 0.471 | 0.663 | 0.73 | 0.49 |
| + CONSENSUS | 0.479 | 0.668 | 0.74 | 0.50 |
| + All + TEMPORAL | **0.497** | **0.673** | **0.74** | **0.51** |

CPS attribution: KG +20.6% (0.34→0.41), MESA +4.9% (0.41→0.43),
CAUSAL-EXPLAIN +11.6% (0.43→0.48), remaining +6.3% (0.48→0.51).

### Table 8 — KG integration strategies

| Strategy | Factual Cons. | Entity Cov. | Rel. Acc. | Inference |
|---|---|---|---|---|
| No KG integration | 72.4 ± 3.2% | 61.2 ± 4.1% | 58.7 ± 3.8% | 2.4 ± 0.3 s |
| Simple concatenation | 78.1 ± 2.8% | 73.8 ± 3.5% | 64.3 ± 3.2% | 2.6 ± 0.3 s |
| Retrieval-augmented | 81.4 ± 2.5% | 79.2 ± 3.1% | 69.8 ± 2.9% | 2.7 ± 0.4 s |
| Attention-based KG | 83.7 ± 2.3% | 81.4 ± 2.9% | 72.1 ± 2.7% | 2.8 ± 0.4 s |
| **Hybrid neural-symbolic** | **87.6 ± 2.1%** | **84.2 ± 2.6%** | **76.8 ± 2.4%** | 2.9 ± 0.4 s |

The *Retrieval-Augmented* row corresponds to the BART+KG baseline's integration
path. Headline claim: +3.9 pp factual consistency (83.7% → 87.6%), p < 0.01.
Residual: 12.4% of statements still contain factual errors.

### Table 9 — Domain-specific performance

| Metric | General | Fin-Tuned | Ours | Gain |
|---|---|---|---|---|
| Entity Recognition (F1) | 67.2 ± 4.3% | 81.4 ± 3.1% | 91.3 ± 2.4% | +24.1 pp |
| Numerical Accuracy | 58.7 ± 5.8% | 73.2 ± 4.2% | 87.6 ± 3.1% | +28.9 pp |
| Temporal Reasoning | 42.1 ± 6.2% | 58.9 ± 4.8% | 74.3 ± 3.9% | +32.2 pp |
| Regulatory Terms | 51.3 ± 7.1% | 69.7 ± 5.3% | 82.1 ± 4.4% | +30.8 pp |
| Causal Relationship (F1) | 34.6 ± 5.4% | 47.8 ± 4.9% | 69.7 ± 4.3% | +35.1 pp |

### Table 10 — Expert evaluation (N = 12, 1–7 scale)

| Dimension | Baseline | Ours |
|---|---|---|
| Summary Quality | 4.2 ± 0.9 | 5.7 ± 0.7 |
| Explainability | 3.1 ± 1.0 | 5.7 ± 0.8 |
| Causal Preservation | 2.8 ± 0.9 | 5.6 ± 0.7 |
| Stakeholder Fit | 3.4 ± 1.0 | 5.9 ± 0.6 |
| Practical Utility | 3.7 ± 0.9 | 5.5 ± 0.8 |

### Table 11 — Construct validity of domain-specific metrics (Phase 2, N = 850 unique summaries)

| Metric | Matched expert dimension | r | 95% CI | p |
|---|---|---|---|---|
| SSI | Stakeholder Fit | 0.58 | [0.53, 0.62] | < 0.001 |
| CPS | Causal Preservation | 0.52 | [0.47, 0.57] | < 0.001 |
| TCC | Explainability | 0.31 | [0.25, 0.37] | < 0.01 |

TCC's weaker correlation is the expected pattern: it measures temporal stability,
not explanation correctness (Section 4.3.2).

### Table 12 — Head-to-head preference (N = 50 per pair)

| Comparison | Prefer ours | Prefer baseline | No preference |
|---|---|---|---|
| vs. BART | 38 (76%) | 7 (14%) | 5 (10%) |
| vs. BART+KG | 33 (66%) | 11 (22%) | 6 (12%) |
| vs. FinanceSum | 29 (58%) | 14 (28%) | 7 (14%) |
| vs. GPT-4 | 27 (54%) | 18 (36%) | 5 (10%) |

GPT-4 comparison: **p = 0.35, statistically indistinguishable from chance.**
Preference agreement Fleiss' κ = 0.68. Analysts 72%, executives 59%.

### Table 13 — Deployment readiness (1–10)

Accuracy 7.1 ± 0.8 · Explainability 7.4 ± 0.7 · Regulatory Alignment 6.8 ± 1.1 ·
Computational Efficiency 6.2 ± 0.9 · User Trust 6.9 ± 0.8 · Robustness 6.4 ± 1.0.
No dimension reaches ≥ 8/10.

### Table 14 / Figure 6 — Error analysis (N = 500)

| Error type | Freq. | Severity |
|---|---|---|
| Factual inconsistency | 14.2% | High |
| Temporal ordering | 9.1% | Medium |
| Causal attribution | 7.8% | Medium |
| Numerical errors | 6.3% | High |
| Entity confusion | 5.4% | Medium |
| Incomplete context | 5.1% | Medium |
| Stakeholder mismatch | 4.7% | Low |
| Technical terminology | 3.9% | Low |
| Hallucination | 2.3% | High |
| Other | 2.2% | Variable |

**Totals: 56.5% total error rate · 21.8% high-severity · 43.5% error-free.**

Figure 6(a) — per-summary, mutually exclusive, sums to 100%: 43.5% error-free,
34.7% lower-severity only, 21.8% high-severity.
Figure 6(b) — per-type, non-exclusive, sums to more than 56.5%.
Two annotators, Cohen's κ = 0.78.

> Figure 6(a) and Table 14 share the per-summary denominator: 43.5% error-free is
> the reported figure.

### Table 15 — Performance under challenging conditions

| Condition | Impact | Error rate | n |
|---|---|---|---|
| Length > 50 pages | −18% ROUGE-L | 68% | 347 |
| 3+ time periods | −12% temporal acc. | 71% | 521 |
| Novel instruments | −22% entity rec. | 65% | 183 |
| Crisis conditions | −9% expl. consist. | 59% | 94 |
| Technical docs | −14% comprehension | 61% | 276 |
| Multiple languages | −31% all metrics | 82% | 67 |

---

## 8. Expert evaluation protocol

- **12 experts, 12 weeks.** IRB-approved. Compensation $75–125/hour, $45,000 total.
- **Panel:** Financial Analysts n=5 (2 buy-side equity CFA 7–12 y; 2 credit 8–11 y;
  1 quantitative 15 y) · Compliance & Risk n=4 (2 compliance officers 10–14 y;
  1 risk manager 12 y; 1 legal counsel 16 y) · Decision Makers n=2 (1 portfolio
  manager 18 y; 1 CFO 22 y) · Academic n=1 (finance professor 20+ y).
- **Institutions:** 3 investment banks, 2 hedge funds, 1 insurance company,
  2 technology companies, 2 universities, 2 regulatory consulting firms.
- **Phase 1** (weeks 1–2): 4-hour demonstration, practice evaluations,
  calibration to κ > 0.70.
- **Phase 2** (weeks 3–10): each expert evaluated **100 summaries** (50 ours,
  50 baselines), randomized and blinded, 30 overlapping for reliability →
  **1,200 total ratings, 850 unique summaries**. Reported in Table 10.
- **Phase 3** (weeks 11–12): all experts evaluated the same **50 document pairs**
  in A/B comparisons. Reported in Table 12.
- **Scales:** 1–7 Likert across Summary Quality, Explanation Quality,
  Stakeholder Fit.
- **Inter-rater reliability: Fleiss' κ = 0.74** (summary quality),
  **κ = 0.71** (explanation quality).

---

## 9. Domain-specific metrics (definitions and edge cases)

| Metric | Definition | Range | Undefined when | Reported |
|---|---|---|---|---|
| **CPS** (Eq. 7) | \|C_summary ∩ C_original\| / \|C_original\| | [0, 1] | C_original = ∅ → document excluded | 0.51 |
| **Weighted CPS** (Eq. 8) | Σ wᵢ·𝕀(cᵢ ∈ C_summary) / Σ wᵢ | [0, 1] | Σ wᵢ = 0 → excluded | — |
| **SSI** (Eq. 12) | Σ w_s · (μ_s^quality + μ_s^explainability)/2 | [0, 1] | — | 0.74 |
| **TCC** (Eq. 9) | (1/N) Σ cos_sim(E_t, E_{t−1}) | [−1, 1] | < 2 periods → excluded | 0.54 |

Interpretation:

- CPS measures preservation of **source-attributed causal claims**, not verified
  causal mechanisms. A high CPS indicates faithful preservation of asserted
  causality, **not** correctness of the underlying mechanism.
- SSI is **stakeholder-rated satisfaction**, not a validated measure of decision
  quality. Aggregation weights `w_s` are fixed per stakeholder category and are
  **not** updated during testing; only MESA's online `(αᵢ, βⱼ)` adapt at
  inference, without modifying model parameters.
- TCC is **orthogonal to factual correctness** — a system can be consistently
  inaccurate. Read alongside Section 4.2.2, not as a quality measure.
- **Regulatory alignment** is documented only qualitatively, as the design-level
  checklist in Table 16; there is no quantitative compliance score.

---

## 10. Other parameters

- **Market regime detection:** Hidden Markov Model, 3 states (bull / bear /
  sideways), rolling 90-day windows of VIX and sector-rotation metrics.
  **81.3% accuracy** on held-out data (2020–2024).
- **Concept drift:** Page-Hinkley test on rolling TCC, threshold **δ = 0.15**,
  triggers recalibration on the most recent **500** document–summary pairs.
- **Crisis regime:** risk-related causal chains carry **twice** the weight in
  MESA stakeholder scoring.
- **MESA online updates:** contextual bandit with REINFORCE-style gradients;
  learning rates η_α = η_β = **0.01**, gradient clipping at **0.5**,
  moving-average baseline. Only stakeholder weight vectors update online.
- **Statistical testing:** paired *t*-tests, α = 0.05, pairing the proposed
  system against each baseline on the common test set across the five seeded runs.
- **Stakeholder categories (4):** financial analysts, regulatory compliance
  officers, executive decision-makers, investment managers.

> **Release scope.** The HMM regime detector, the Page-Hinkley drift test and the
> REINFORCE-style MESA update above are specified in the paper, not included in
> this release. The code uses a keyword regime heuristic, a fixed TCC threshold and
> a simplified reward-scaled weight step as stand-ins; the parameter values are
> recorded in `configs/model_config.yaml`.

---

## 11. Scope of the claims

Positions the manuscript takes, which the repository follows:

1. **No preference superiority over GPT-4.** 54%, p = 0.35. The contribution is
   framed as **auditability and traceability**, not preference superiority.
2. **No regulatory compliance claim.** Table 16 is a design-level
   compliance-support self-assessment, qualitative only, explicitly "not legal
   certification". The system has not undergone independent legal review.
3. **Not all five components are algorithmically novel.** Novelty lies in joint
   formalization and integration into a FIBO-grounded pipeline; several adapt
   established techniques (contextual bandits, ensemble consensus, HMM regime
   detection, drift detection).
4. **Not deployable autonomously.** 21.8% high-severity error rate; documents
   over 50 pages cannot support autonomous processing; human review mandatory.
5. **GPT-4 baseline is a frozen historical reference point** (`gpt-4-0613`), not
   a current state-of-the-art comparator.
6. **Trained weights were not retained** — see the Checkpoints section of the
   root README.
