# Deployment Guide

Production deployment guide for the Financial Explainable Summarization system.

---

## Deployment Options

### 1. REST API Server (Recommended)
- FastAPI + Uvicorn + Docker
- Multi-client support
- Easy to scale

### 2. Python Package
- Direct integration
- `pip install -e .` from a clone of this repository (not published on PyPI)

### 3. Cloud Function (Serverless)
- AWS Lambda / GCP Functions
- Auto-scaling, pay-per-use

---

## Quick Deploy

### Docker Deployment
No Dockerfile is included in this release. Docker 24.0.5 was part of the reported
software environment (Section 3.6.3).

### Cloud Deployment
- **AWS:** EC2 p3.2xlarge (V100) - $3/hour
- **GCP:** GKE with nvidia-tesla-v100
- **Azure:** NC6s_v3 with V100

---

## Performance

Reported in the manuscript (Table 5, N = 1,000 documents, 2× NVIDIA V100 32 GB):

- **Summary inference:** 2.9 ± 0.4 s per document (+21% vs. BART-large)
- **Explanation generation:** a further 18 ± 3 s per document
- **Peak memory:** 9.8 ± 0.7 GB (+19.5%)
- **KG query time:** 127 ± 18 ms

The system is not suitable for autonomous deployment: 21.8% of summaries contain
a high-severity error, and human review is mandatory (Section 4.4).

---

## Optimization

Candidate directions (not evaluated in the manuscript): quantization, batch
processing, caching of knowledge-graph queries, and model distillation, which the
paper lists as future work for reducing the inference overhead below +10%.

---

See the root README for scope and limitations.
