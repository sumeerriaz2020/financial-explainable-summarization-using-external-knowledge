"""
Causal Preservation Score (CPS)
===============================

Single implementation of Equations 7 and 8, shared by the evaluation metrics
(training_evaluation/metrics.py) and the CAUSAL-EXPLAIN components
(algorithms/algorithm_4_causal_explain.py, explainability/causal_explain.py).

    CPS          = |C_summary ∩ C_original| / |C_original|                 (Eq. 7)
    CPS_weighted = sum_i w_i * I(c_i ∈ C_summary) / sum_i w_i              (Eq. 8)

A causal claim is identified by its (cause, effect) pair, normalized for case and
whitespace; direction matters and confidence values are ignored. C_original is the
set of distinct claims asserted in the source. Following Section 3.7.2:

- CPS ranges over [0, 1]; each source claim counts at most once, so duplicates in
  the summary cannot raise the score.
- CPS is undefined when the source contains no causal claims (and weighted CPS
  when the weights sum to zero). Such documents are excluded from averaging; the
  functions return None for them.
- CPS measures preservation of source-asserted causal claims, not whether those
  claims are causally valid.

Matching is exact after normalization. Paraphrased claims ("rate hikes" vs.
"higher interest rates") do not match unless the caller maps both sides to shared
identifiers first, e.g. FIBO-linked entity URIs.

Reference: Section 3.4.2, Section 3.7.2, Equations 7-8
"""

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Claim = Tuple[str, str]


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def normalize_claim(claim: Any) -> Claim:
    """
    Return the (cause, effect) key of a causal claim.

    Accepts a sequence whose first two items are cause and effect, e.g.
    ("A", "B") or ("A", "B", 0.9), or an object with `cause` and `effect`
    attributes, e.g. CausalChain.
    """
    if hasattr(claim, "cause") and hasattr(claim, "effect"):
        cause, effect = claim.cause, claim.effect
    else:
        cause, effect = claim[0], claim[1]
    return (_normalize_text(cause), _normalize_text(effect))


def _distinct_source_claims(
    source_claims: Iterable[Any],
    weights: Optional[Sequence[float]] = None
) -> Dict[Claim, float]:
    """
    Map each distinct source claim to its weight, keeping the first occurrence.

    Weights, when given, are aligned with source_claims. Without weights every
    claim weighs 1.0.
    """
    source_claims = list(source_claims)
    if weights is not None and len(weights) != len(source_claims):
        raise ValueError(
            f"{len(weights)} weights given for {len(source_claims)} source claims"
        )
    distinct: Dict[Claim, float] = {}
    for i, claim in enumerate(source_claims):
        key = normalize_claim(claim)
        if key not in distinct:
            distinct[key] = 1.0 if weights is None else float(weights[i])
    return distinct


def causal_preservation_score(
    source_claims: Iterable[Any],
    summary_claims: Iterable[Any]
) -> Optional[float]:
    """CPS for one document (Equation 7); None if the source has no claims."""
    original = _distinct_source_claims(source_claims)
    if not original:
        return None
    summary = {normalize_claim(c) for c in summary_claims}
    return sum(1 for key in original if key in summary) / len(original)


def weighted_causal_preservation_score(
    source_claims: Iterable[Any],
    summary_claims: Iterable[Any],
    weights: Optional[Sequence[float]] = None
) -> Optional[float]:
    """
    Weighted CPS for one document (Equation 8).

    Weights are aligned with source_claims. If omitted, each claim's
    `importance_weight` attribute is used when present, otherwise 1.0.
    Returns None if the source has no claims or the weights sum to zero.
    """
    source_claims = list(source_claims)
    if weights is None and source_claims and all(
        hasattr(c, "importance_weight") for c in source_claims
    ):
        weights = [c.importance_weight for c in source_claims]
    original = _distinct_source_claims(source_claims, weights)
    total = sum(original.values())
    if not original or total <= 0:
        return None
    summary = {normalize_claim(c) for c in summary_claims}
    return sum(w for key, w in original.items() if key in summary) / total


def aggregate_cps(
    source_claims_per_doc: Sequence[Iterable[Any]],
    summary_claims_per_doc: Sequence[Iterable[Any]],
    weights_per_doc: Optional[Sequence[Optional[Sequence[float]]]] = None
) -> Dict[str, float]:
    """
    Corpus-level CPS: mean and SD over documents where CPS is defined.

    Returns cps, cps_std, n_documents (scored) and n_excluded (undefined), plus
    cps_weighted, cps_weighted_std and n_excluded_weighted when weights are given.
    Means are NaN if every document is excluded.
    """
    if len(source_claims_per_doc) != len(summary_claims_per_doc):
        raise ValueError("source and summary claim lists differ in length")
    if weights_per_doc is not None and len(weights_per_doc) != len(source_claims_per_doc):
        raise ValueError("weights_per_doc differs in length from the claim lists")

    def _mean_std(values: List[float]) -> Tuple[float, float]:
        if not values:
            return math.nan, math.nan
        mean = sum(values) / len(values)
        return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

    scores, weighted = [], []
    for i, (source, summary) in enumerate(zip(source_claims_per_doc, summary_claims_per_doc)):
        source, summary = list(source), list(summary)
        score = causal_preservation_score(source, summary)
        if score is not None:
            scores.append(score)
        if weights_per_doc is not None:
            w_score = weighted_causal_preservation_score(source, summary, weights_per_doc[i])
            if w_score is not None:
                weighted.append(w_score)

    n_docs = len(source_claims_per_doc)
    cps, cps_std = _mean_std(scores)
    result = {
        'cps': cps,
        'cps_std': cps_std,
        'n_documents': len(scores),
        'n_excluded': n_docs - len(scores)
    }
    if weights_per_doc is not None:
        w_mean, w_std = _mean_std(weighted)
        result.update({
            'cps_weighted': w_mean,
            'cps_weighted_std': w_std,
            'n_excluded_weighted': n_docs - len(weighted)
        })
    return result
