"""
Unit Tests for Evaluation Metrics
==================================

Tests for ROUGE, factual consistency and the Stakeholder Satisfaction Index (SSI).
"""

import unittest
import numpy as np
from training_evaluation.metrics import (
    ROUGEMetrics,
    FactualConsistencyMetrics,
    SSIMetrics
)


class TestROUGEMetrics(unittest.TestCase):
    """Test ROUGE Metrics"""
    
    def setUp(self):
        self.rouge = ROUGEMetrics()
        
        self.predictions = [
            "Apple reported strong earnings growth.",
            "Revenue increased by 12 percent."
        ]
        
        self.references = [
            "Apple Inc. announced robust earnings growth.",
            "The company reported 12% revenue increase."
        ]
    
    def test_rouge_computation(self):
        """Test ROUGE score computation"""
        scores = self.rouge.compute(self.predictions, self.references)
        
        self.assertIn('rouge1', scores)
        self.assertIn('rouge2', scores)
        self.assertIn('rougeL', scores)
        
        # Scores should be between 0 and 1
        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_perfect_match(self):
        """Test ROUGE with perfect match"""
        pred = ["This is a test sentence."]
        ref = ["This is a test sentence."]
        
        scores = self.rouge.compute(pred, ref)
        
        # Perfect match should give 1.0
        self.assertAlmostEqual(scores['rougeL'], 1.0, places=2)


class TestFactualConsistencyMetrics(unittest.TestCase):
    """Test Factual Consistency Metrics"""
    
    def setUp(self):
        self.factual = FactualConsistencyMetrics()
        
        self.predictions = ["Apple reported $89.5B revenue."]
        self.sources = ["Apple Inc. announced revenue of $89.5 billion."]
    
    def test_factual_consistency(self):
        """Test factual consistency computation"""
        scores = self.factual.compute(self.predictions, self.sources)
        
        self.assertIn('factual_consistency', scores)
        self.assertGreaterEqual(scores['factual_consistency'], 0.0)
        self.assertLessEqual(scores['factual_consistency'], 100.0)
    
    def test_hallucination_detection(self):
        """Test detection of hallucinated facts"""
        pred = ["Apple reported $999B revenue."]  # Hallucinated
        src = ["Apple Inc. announced revenue of $89.5 billion."]
        
        scores = self.factual.compute(pred, src)
        
        # Should have lower consistency
        self.assertLess(scores['factual_consistency'], 100.0)


class TestSSIMetrics(unittest.TestCase):
    """Test Stakeholder Satisfaction Index (Equation 12)"""

    def setUp(self):
        self.ssi = SSIMetrics()

        self.explanations = {
            'analyst': ["Detailed analysis..."],
            'compliance': ["Meets requirements..."],
            'executive': ["High-level summary..."],
            'investment_manager': ["Risk assessment..."]
        }

        self.stakeholder_ratings = {
            'analyst': [0.85],
            'compliance': [0.90],
            'executive': [0.80],
            'investment_manager': [0.75]
        }

    def test_ssi_computation(self):
        """Test SSI computation (Equation 12)"""
        scores = self.ssi.compute(self.explanations, self.stakeholder_ratings)
        
        self.assertIn('ssi', scores)
        self.assertIn('ssi_std', scores)
        
        # SSI should be between 0 and 1
        self.assertGreaterEqual(scores['ssi'], 0.0)
        self.assertLessEqual(scores['ssi'], 1.0)
    
    def test_stakeholder_weights(self):
        """Test stakeholder weight application"""
        weights = self.ssi.stakeholder_weights
        
        # Weights should sum to 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)

        # Fixed a priori at 0.25 each (Section 3.7.2)
        for weight in weights.values():
            self.assertAlmostEqual(weight, 0.25, places=5)

    def test_weighted_average(self):
        """Test weighted average computation"""
        # Manual computation
        expected = (
            0.25 * 0.85 +  # analyst
            0.25 * 0.90 +  # compliance
            0.25 * 0.80 +  # executive
            0.25 * 0.75    # investment manager
        )
        
        scores = self.ssi.compute(self.explanations, self.stakeholder_ratings)
        
        self.assertAlmostEqual(scores['ssi'], expected, places=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
