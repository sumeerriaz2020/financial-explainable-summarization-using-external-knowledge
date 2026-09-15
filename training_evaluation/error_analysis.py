"""
Error Analysis
==============

Error categorization following the taxonomy of Section 4.4 (Table 14).

In the manuscript, 500 randomly sampled summaries were classified MANUALLY by two
expert annotators (Cohen's kappa = 0.78). The heuristic checks in this module are
an automated approximation that covers only a subset of the categories; they do
not reproduce the manual annotation.

Error categories and reported per-summary frequencies (Table 14; non-exclusive,
a summary may contain several error types):
    Factual Inconsistency   14.2%  High
    Temporal Ordering        9.1%  Medium
    Causal Attribution       7.8%  Medium
    Numerical Errors         6.3%  High
    Entity Confusion         5.4%  Medium
    Incomplete Context       5.1%  Medium
    Stakeholder Mismatch     4.7%  Low
    Technical Terminology    3.9%  Low
    Hallucination            2.3%  High
    Other                    2.2%  Variable

Per-summary totals: 56.5% contain at least one error, 21.8% at least one
high-severity error, 43.5% are error-free (Figure 6).

Reference: Section 4.4 (Error Analysis), Table 14, Figure 6
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Detailed error analysis and categorization
    
    Analyzes prediction errors and categorizes them by type,
    severity, and patterns.
    """
    
    def __init__(self):
        """Initialize error analyzer"""
        # Error categories, severities and reported per-summary frequencies
        # (Table 14). Severity is a property of the category, as in the paper.
        self.error_categories = {
            'factual_inconsistency': {
                'name': 'Factual Inconsistency', 'severity': 'high',
                'primary_cause': 'Complex numerical relations', 'expected_rate': 0.142
            },
            'temporal_ordering': {
                'name': 'Temporal Ordering', 'severity': 'medium',
                'primary_cause': 'Multi-period documents', 'expected_rate': 0.091
            },
            'causal_attribution': {
                'name': 'Causal Attribution', 'severity': 'medium',
                'primary_cause': 'Ambiguous causality', 'expected_rate': 0.078
            },
            'numerical_errors': {
                'name': 'Numerical Errors', 'severity': 'high',
                'primary_cause': 'Precision/conversion errors', 'expected_rate': 0.063
            },
            'entity_confusion': {
                'name': 'Entity Confusion', 'severity': 'medium',
                'primary_cause': 'Similar entity names', 'expected_rate': 0.054
            },
            'incomplete_context': {
                'name': 'Incomplete Context', 'severity': 'medium',
                'primary_cause': 'Missing background', 'expected_rate': 0.051
            },
            'stakeholder_mismatch': {
                'name': 'Stakeholder Mismatch', 'severity': 'low',
                'primary_cause': 'Preference variability', 'expected_rate': 0.047
            },
            'technical_terminology': {
                'name': 'Technical Terminology', 'severity': 'low',
                'primary_cause': 'Emerging jargon', 'expected_rate': 0.039
            },
            'hallucination': {
                'name': 'Hallucination', 'severity': 'high',
                'primary_cause': 'Generation artifacts', 'expected_rate': 0.023
            },
            'other': {
                'name': 'Other', 'severity': 'variable',
                'primary_cause': 'Miscellaneous', 'expected_rate': 0.022
            }
        }

        # Reported per-summary totals (Table 14 / Figure 6)
        self.expected_totals = {
            'total_error_rate': 0.565,
            'high_severity_rate': 0.218,
            'error_free_rate': 0.435
        }
        
        logger.info("Error Analyzer initialized")
    
    def analyze(
        self,
        predictions: List[str],
        references: List[str],
        sources: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Perform comprehensive error analysis
        
        Args:
            predictions: Model predictions
            references: Reference summaries
            sources: Source documents
            metadata: Optional metadata per sample
            
        Returns:
            Detailed error analysis results
        """
        logger.info("=" * 70)
        logger.info("Starting Error Analysis")
        logger.info("=" * 70)
        
        errors_by_category = defaultdict(list)
        errors_by_severity = defaultdict(list)
        detailed_errors = []
        errors_per_sample = []

        for i, (pred, ref, src) in enumerate(zip(predictions, references, sources)):
            # Analyze this sample
            sample_errors = self._analyze_sample(pred, ref, src, i)
            errors_per_sample.append(sample_errors)

            # Categorize
            for error in sample_errors:
                errors_by_category[error['category']].append(error)
                errors_by_severity[error['severity']].append(error)
                detailed_errors.append(error)

        # Compute statistics
        stats = self._compute_statistics(errors_per_sample)
        
        # Print summary
        self._print_summary(stats)
        
        return {
            'statistics': stats,
            'errors_by_category': errors_by_category,
            'errors_by_severity': errors_by_severity,
            'detailed_errors': detailed_errors
        }
    
    def _analyze_sample(
        self,
        prediction: str,
        reference: str,
        source: str,
        sample_id: int
    ) -> List[Dict]:
        """
        Analyze errors in a single sample.

        Heuristics cover entity confusion, causal attribution, temporal ordering
        and numerical errors only. The remaining Table 14 categories require
        manual annotation.
        """
        errors = []

        # Check for entity confusion
        entity_errors = self._check_entity_errors(prediction, reference, source)
        errors.extend(entity_errors)

        # Check for causal attribution errors
        causal_errors = self._check_causal_errors(prediction, reference)
        errors.extend(causal_errors)

        # Check for temporal ordering errors
        temporal_errors = self._check_temporal_errors(prediction, reference)
        errors.extend(temporal_errors)

        # Check for numerical errors
        numerical_errors = self._check_factual_errors(prediction, source)
        errors.extend(numerical_errors)

        # Add sample ID and category severity to all errors
        for error in errors:
            error['sample_id'] = sample_id
            error['severity'] = self.error_categories[error['category']]['severity']

        return errors
    
    def _check_entity_errors(
        self,
        prediction: str,
        reference: str,
        source: str
    ) -> List[Dict]:
        """Check for entity confusion errors"""
        errors = []
        
        # Extract entities (simplified)
        pred_entities = self._extract_entities(prediction)
        ref_entities = self._extract_entities(reference)
        source_entities = self._extract_entities(source)
        
        # Check for hallucinated entities
        for entity in pred_entities:
            if entity not in source_entities:
                errors.append({
                    'category': 'entity_confusion',
                    'type': 'hallucinated_entity',
                    'entity': entity,
                    'description': f"Entity '{entity}' not in source"
                })
        
        # Check for missing entities
        for entity in ref_entities:
            if entity not in pred_entities:
                errors.append({
                    'category': 'entity_confusion',
                    'type': 'missing_entity',
                    'entity': entity,
                    'description': f"Entity '{entity}' missing from prediction"
                })
        
        return errors
    
    def _check_causal_errors(
        self,
        prediction: str,
        reference: str
    ) -> List[Dict]:
        """Check for causal attribution errors"""
        errors = []
        
        # Extract causal patterns
        causal_markers = ['led to', 'caused', 'resulted in', 'due to', 'because of']
        
        pred_causals = []
        ref_causals = []
        
        for marker in causal_markers:
            # Find causal relationships in prediction
            if marker in prediction.lower():
                context = self._extract_context(prediction, marker)
                pred_causals.append((marker, context))
            
            # Find causal relationships in reference
            if marker in reference.lower():
                context = self._extract_context(reference, marker)
                ref_causals.append((marker, context))
        
        # Check for incorrect causal attributions
        if pred_causals and not any(
            self._causal_matches(pc, ref_causals)
            for pc in pred_causals
        ):
            errors.append({
                'category': 'causal_attribution',
                'type': 'incorrect_causation',
                'description': 'Causal relationship not supported'
            })
        
        return errors
    
    def _check_temporal_errors(
        self,
        prediction: str,
        reference: str
    ) -> List[Dict]:
        """Check for temporal ordering errors"""
        errors = []
        
        # Extract temporal expressions
        temporal_pattern = r'\b(Q[1-4]\s+\d{4}|\d{4}|yesterday|today|last\s+\w+)\b'
        
        pred_temporal = re.findall(temporal_pattern, prediction, re.IGNORECASE)
        ref_temporal = re.findall(temporal_pattern, reference, re.IGNORECASE)
        
        # Check for temporal inconsistencies
        if pred_temporal and ref_temporal:
            if not any(pt in ref_temporal for pt in pred_temporal):
                errors.append({
                    'category': 'temporal_ordering',
                    'type': 'wrong_timeframe',
                    'description': 'Temporal expressions do not match'
                })
        
        # Check for temporal ordering issues
        if len(pred_temporal) > 1:
            # Simplified check: ensure chronological order
            # (In production, would parse and validate dates)
            pass
        
        return errors
    
    def _check_factual_errors(
        self,
        prediction: str,
        source: str
    ) -> List[Dict]:
        """Check for numerical errors (numbers not supported by the source)"""
        errors = []
        
        # Extract numerical facts
        pred_numbers = re.findall(r'\$?[\d,]+\.?\d*[BMK]?', prediction)
        source_numbers = re.findall(r'\$?[\d,]+\.?\d*[BMK]?', source)
        
        # Check for hallucinated numbers
        for num in pred_numbers:
            if num not in source_numbers:
                # Check if similar number exists (allow small variations)
                if not self._number_exists_similar(num, source_numbers):
                    errors.append({
                        'category': 'numerical_errors',
                        'type': 'hallucinated_number',
                        'value': num,
                        'description': f"Number '{num}' not in source"
                    })
        
        return errors
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simplified)"""
        # Simplified: extract capitalized phrases
        entities = []
        
        # Company names (capitalized + Inc/Corp/etc)
        companies = re.findall(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Inc\.|Corp\.|Ltd\.)',
            text
        )
        entities.extend(companies)
        
        # Standalone capitalized words (likely entities)
        caps = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(caps)
        
        return list(set(entities))
    
    def _extract_context(self, text: str, marker: str, window: int = 50) -> str:
        """Extract context around marker"""
        idx = text.lower().find(marker.lower())
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(marker) + window)
        
        return text[start:end]
    
    def _causal_matches(
        self,
        pred_causal: Tuple,
        ref_causals: List[Tuple]
    ) -> bool:
        """Check if predicted causal matches any reference"""
        pred_marker, pred_context = pred_causal
        
        for ref_marker, ref_context in ref_causals:
            # Simple overlap check
            pred_words = set(pred_context.lower().split())
            ref_words = set(ref_context.lower().split())
            
            overlap = len(pred_words & ref_words) / len(pred_words | ref_words)
            if overlap > 0.3:
                return True
        
        return False
    
    def _number_exists_similar(
        self,
        target: str,
        numbers: List[str],
        threshold: float = 0.05
    ) -> bool:
        """Check if similar number exists"""
        # Extract numeric value
        target_val = self._parse_number(target)
        if target_val is None:
            return False
        
        for num in numbers:
            num_val = self._parse_number(num)
            if num_val is None:
                continue
            
            # Check if within threshold
            if abs(target_val - num_val) / num_val < threshold:
                return True
        
        return False
    
    def _parse_number(self, num_str: str) -> Optional[float]:
        """Parse number string to float"""
        try:
            # Remove commas and currency symbols
            clean = re.sub(r'[$,]', '', num_str)
            
            # Handle suffixes
            if clean.endswith('B'):
                return float(clean[:-1]) * 1e9
            elif clean.endswith('M'):
                return float(clean[:-1]) * 1e6
            elif clean.endswith('K'):
                return float(clean[:-1]) * 1e3
            else:
                return float(clean)
        except:
            return None
    
    def _compute_statistics(self, errors_per_sample: List[List[Dict]]) -> Dict:
        """
        Compute per-summary error statistics.
        
        Rates follow the manuscript's definitions (Table 14 / Figure 6): a
        category rate is the share of summaries containing at least one error of
        that type (non-exclusive), the total error rate is the share with at
        least one error, and the high-severity rate is the share with at least
        one high-severity error.
        """
        total_samples = len(errors_per_sample)
        stats = {
            'total_samples': total_samples,
            'total_errors': sum(len(errors) for errors in errors_per_sample),
            'total_error_rate': 0.0,
            'high_severity_rate': 0.0,
            'error_free_rate': 0.0,
            'category_distribution': {},
            'comparison_to_expected': {}
        }
        
        if total_samples == 0:
            return stats
        
        with_error = sum(1 for errors in errors_per_sample if errors)
        with_high = sum(
            1 for errors in errors_per_sample
            if any(e['severity'] == 'high' for e in errors)
        )
        stats['total_error_rate'] = with_error / total_samples
        stats['high_severity_rate'] = with_high / total_samples
        stats['error_free_rate'] = 1.0 - stats['total_error_rate']
        
        for category, info in self.error_categories.items():
            count = sum(
                1 for errors in errors_per_sample
                if any(e['category'] == category for e in errors)
            )
            rate = count / total_samples
            stats['category_distribution'][category] = {
                'summaries_affected': count,
                'rate': rate,
                'severity': info['severity']
            }
            stats['comparison_to_expected'][category] = {
                'observed': rate,
                'expected': info['expected_rate'],
                'difference': rate - info['expected_rate']
            }
        
        return stats
    
    def _print_summary(self, stats: Dict):
        """Print error analysis summary"""
        logger.info("\n" + "=" * 70)
        logger.info("ERROR ANALYSIS SUMMARY")
        logger.info("=" * 70)
        
        logger.info(f"\nTotal Samples: {stats['total_samples']}")
        logger.info(f"Total error rate (>=1 error): {stats['total_error_rate']:.1%} "
                    f"(reported: {self.expected_totals['total_error_rate']:.1%})")
        logger.info(f"High-severity rate: {stats['high_severity_rate']:.1%} "
                    f"(reported: {self.expected_totals['high_severity_rate']:.1%})")
        logger.info(f"Error-free: {stats['error_free_rate']:.1%} "
                    f"(reported: {self.expected_totals['error_free_rate']:.1%})")
        
        logger.info("\nPer-summary rate by category (heuristic vs. reported manual annotation):")
        for category, data in stats['comparison_to_expected'].items():
            logger.info(
                f"  {self.error_categories[category]['name']}: {data['observed']:.1%} "
                f"(reported: {data['expected']:.1%})"
            )


# Example usage
if __name__ == "__main__":
    print("Error Analysis Module")
    print("=" * 70)
    
    analyzer = ErrorAnalyzer()
    print("\nError categories (Table 14; per-summary, non-exclusive):")
    for info in analyzer.error_categories.values():
        print(f"  {info['name']:<24} {info['expected_rate']:>5.1%}  {info['severity']}")
    
    print("\nPer-summary totals (Table 14 / Figure 6):")
    print("  Total error rate: 56.5% | High-severity: 21.8% | Error-free: 43.5%")
    
    print("\nNote: the reported figures come from manual annotation of 500")
    print("summaries by two experts; the heuristics here are an approximation.")
    
    print("\n" + "=" * 70)
    print("Error analyzer ready!")
    print("=" * 70)
