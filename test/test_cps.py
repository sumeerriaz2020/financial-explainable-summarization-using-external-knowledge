"""
Unit Tests for the Causal Preservation Score
============================================

Deterministic tests of Equations 7-8 and the edge-case conventions of
Section 3.7.2. Standard library only; no torch or model downloads.
"""

import math
import unittest
from dataclasses import dataclass

from training_evaluation.cps import (
    aggregate_cps,
    causal_preservation_score,
    normalize_claim,
    weighted_causal_preservation_score,
)


@dataclass
class Chain:
    cause: str
    effect: str
    importance_weight: float = 1.0


class TestCausalPreservationScore(unittest.TestCase):
    """Test CPS for a single document (Equation 7)"""

    def setUp(self):
        self.source = [("rate hikes", "market decline"), ("market decline", "outflows")]

    def test_full_preservation(self):
        """All source claims present in the summary gives 1.0"""
        self.assertEqual(causal_preservation_score(self.source, self.source), 1.0)

    def test_no_preservation(self):
        """No source claims in the summary gives 0.0"""
        summary = [("earnings", "buybacks")]
        self.assertEqual(causal_preservation_score(self.source, summary), 0.0)

    def test_partial_preservation(self):
        """One of two source claims preserved gives 0.5"""
        summary = [("rate hikes", "market decline", 0.9)]
        self.assertEqual(causal_preservation_score(self.source, summary), 0.5)

    def test_empty_source_is_undefined(self):
        """CPS is undefined (None) when the source has no causal claims"""
        self.assertIsNone(causal_preservation_score([], [("a", "b")]))

    def test_duplicates_cannot_exceed_one(self):
        """Repeated claims in summary or source never push CPS above 1"""
        source = self.source + [("Rate  Hikes", "Market Decline")]
        summary = [("rate hikes", "market decline")] * 5
        self.assertEqual(causal_preservation_score(source, summary), 0.5)

    def test_direction_matters(self):
        """A reversed claim is not a preserved claim"""
        summary = [("market decline", "rate hikes")]
        self.assertEqual(causal_preservation_score(self.source, summary), 0.0)

    def test_normalization(self):
        """Claims match regardless of case and whitespace; confidence is ignored"""
        self.assertEqual(
            normalize_claim(("  Rate   Hikes ", "MARKET decline", 0.4)),
            ("rate hikes", "market decline")
        )
        self.assertEqual(normalize_claim(Chain("A", "B")), ("a", "b"))


class TestWeightedCausalPreservationScore(unittest.TestCase):
    """Test weighted CPS for a single document (Equation 8)"""

    def test_hand_computed_example(self):
        """Weights 3 and 1 with only the first claim preserved gives 3/4"""
        source = [("a", "b"), ("c", "d")]
        summary = [("a", "b")]
        self.assertAlmostEqual(
            weighted_causal_preservation_score(source, summary, weights=[3.0, 1.0]),
            0.75
        )

    def test_equal_weights_reduce_to_cps(self):
        """With equal weights, weighted CPS equals standard CPS"""
        source = [("a", "b"), ("c", "d"), ("e", "f")]
        summary = [("c", "d")]
        self.assertAlmostEqual(
            weighted_causal_preservation_score(source, summary, weights=[2.0, 2.0, 2.0]),
            causal_preservation_score(source, summary)
        )

    def test_importance_weight_attribute(self):
        """Weights are read from importance_weight when not passed explicitly"""
        source = [Chain("a", "b", 1.0), Chain("c", "d", 4.0)]
        summary = [Chain("c", "d")]
        self.assertAlmostEqual(weighted_causal_preservation_score(source, summary), 0.8)

    def test_zero_weights_are_undefined(self):
        """Weighted CPS is undefined (None) when the weights sum to zero"""
        self.assertIsNone(
            weighted_causal_preservation_score([("a", "b")], [("a", "b")], weights=[0.0])
        )

    def test_misaligned_weights_rejected(self):
        """Weights must align with the source claims"""
        with self.assertRaises(ValueError):
            weighted_causal_preservation_score([("a", "b")], [], weights=[1.0, 2.0])


class TestAggregateCPS(unittest.TestCase):
    """Test corpus-level averaging and exclusion"""

    def test_undefined_documents_are_excluded(self):
        """Documents without source claims are excluded from the mean, not scored"""
        sources = [[("a", "b"), ("c", "d")], [], [("e", "f")]]
        summaries = [[("a", "b")], [("x", "y")], [("e", "f")]]
        result = aggregate_cps(sources, summaries)

        self.assertAlmostEqual(result['cps'], 0.75)      # mean of 0.5 and 1.0
        self.assertAlmostEqual(result['cps_std'], 0.25)
        self.assertEqual(result['n_documents'], 2)
        self.assertEqual(result['n_excluded'], 1)

    def test_per_document_weights(self):
        """Each document uses its own weights"""
        sources = [[("a", "b"), ("c", "d")], [("e", "f"), ("g", "h")]]
        summaries = [[("a", "b")], [("h", "g")]]
        result = aggregate_cps(sources, summaries, weights_per_doc=[[3.0, 1.0], [1.0, 1.0]])

        self.assertAlmostEqual(result['cps_weighted'], (0.75 + 0.0) / 2)
        self.assertEqual(result['n_excluded_weighted'], 0)

    def test_all_excluded_gives_nan(self):
        """If every document is undefined, the mean is NaN"""
        result = aggregate_cps([[], []], [[("a", "b")], []])
        self.assertTrue(math.isnan(result['cps']))
        self.assertEqual(result['n_excluded'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
