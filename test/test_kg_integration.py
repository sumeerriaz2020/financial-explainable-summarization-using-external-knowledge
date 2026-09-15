"""
Unit Tests for Knowledge Graph Integration
===========================================

Tests for causal relation extraction and temporal annotation.
"""

import unittest
from knowledge_graph.causal_extraction import CausalExtractor
from knowledge_graph.temporal_annotation import TemporalAnnotator


class TestCausalExtractor(unittest.TestCase):
    """Test Causal Relation Extraction"""

    def setUp(self):
        self.extractor = CausalExtractor()

    def test_multiple_causals(self):
        """Test extraction of multiple causal relations"""
        text = "A caused B. B resulted in C."
        relations = self.extractor.extract_causal_relations(text)

        self.assertIsInstance(relations, list)


class TestTemporalAnnotator(unittest.TestCase):
    """Test Temporal Annotation"""

    def setUp(self):
        self.annotator = TemporalAnnotator()

    def test_temporal_expression_detection(self):
        """Test temporal expression detection"""
        text = "Q1 2023 results improved over Q4 2022."
        temporal_exprs = self.annotator.extract_temporal_expressions(text)

        self.assertIsInstance(temporal_exprs, list)
        self.assertGreater(len(temporal_exprs), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
