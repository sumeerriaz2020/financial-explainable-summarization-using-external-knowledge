"""
ADAPT-EVAL - Adaptive Real-Time Explanation Evaluation
=======================================================

Leverages user interactions to iteratively refine explanation generation
through continuous learning mechanisms.

Reference: Section 3.4.3
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """User feedback record"""
    timestamp: str
    explanation: str
    explicit_rating: float
    dwell_time: float
    interactions: List[str]
    context: Dict


class AdaptiveEvaluator:
    """Adaptive explanation evaluation with continuous learning"""
    
    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate
        self.feedback_history = []
        self.user_models = {}
        self.quality_predictors = {
            'comprehension': self._init_predictor(),
            'trust': self._init_predictor(),
            'actionability': self._init_predictor()
        }
        logger.info("ADAPT-EVAL initialized")
    
    def collect_feedback(
        self,
        explanation: str,
        user_id: str,
        explicit_rating: float = None,
        dwell_time: float = None,
        interactions: List[str] = None,
        context: Dict = None
    ) -> FeedbackRecord:
        """Collect multi-modal feedback"""
        
        record = FeedbackRecord(
            timestamp=datetime.now().isoformat(),
            explanation=explanation,
            explicit_rating=explicit_rating or 0.0,
            dwell_time=dwell_time or 0.0,
            interactions=interactions or [],
            context=context or {}
        )
        
        self.feedback_history.append(record)
        
        # Update user model
        if user_id not in self.user_models:
            self.user_models[user_id] = self._init_user_model()
        
        self._update_user_model(user_id, record)
        
        return record
    
    def adapt_model(self, feedback: FeedbackRecord):
        """Adapt quality predictors based on feedback"""
        
        # Extract features
        features = self._extract_features(feedback)
        
        # Update each predictor
        for dimension in ['comprehension', 'trust', 'actionability']:
            predictor = self.quality_predictors[dimension]
            
            # Compute target quality
            target = self._compute_target_quality(feedback, dimension)
            
            # Gradient descent update
            prediction = self._predict_quality(predictor, features)
            error = target - prediction
            
            # Update weights
            for i, feature in enumerate(features):
                predictor['weights'][i] += self.learning_rate * error * feature
            
            # Update bias
            predictor['bias'] += self.learning_rate * error
        
        logger.debug(f"Model adapted based on feedback: {feedback.explicit_rating}")
    
    def select_explanation(
        self,
        candidates: List[str],
        user_id: str,
        context: Dict
    ) -> Tuple[str, Dict[str, float]]:
        """Select best explanation using learned preferences"""
        
        # Get user model
        user_model = self.user_models.get(user_id, self._init_user_model())
        
        best_explanation = None
        best_score = -float('inf')
        best_predictions = {}
        
        for explanation in candidates:
            # Create pseudo-feedback for prediction
            pseudo_feedback = FeedbackRecord(
                timestamp=datetime.now().isoformat(),
                explanation=explanation,
                explicit_rating=0.0,
                dwell_time=0.0,
                interactions=[],
                context=context
            )
            
            features = self._extract_features(pseudo_feedback)
            
            # Predict quality dimensions
            predictions = {}
            total_score = 0.0
            
            for dimension in ['comprehension', 'trust', 'actionability']:
                predictor = self.quality_predictors[dimension]
                quality = self._predict_quality(predictor, features)
                predictions[dimension] = quality
                
                # Weight by user preference
                weight = user_model['preferences'].get(dimension, 0.33)
                total_score += weight * quality
            
            # Uncertainty-aware selection
            uncertainty = self._compute_uncertainty(features)
            adjusted_score = total_score - 0.1 * uncertainty
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_explanation = explanation
                best_predictions = predictions
        
        return best_explanation, best_predictions
    
    def compute_trust_calibration(self) -> float:
        """Compute trust calibration metric"""
        if len(self.feedback_history) < 10:
            return 0.5
        
        # Compare predicted quality vs actual ratings
        correct = 0
        total = 0
        
        for record in self.feedback_history[-100:]:
            features = self._extract_features(record)
            
            # Average predicted quality
            avg_pred = np.mean([
                self._predict_quality(pred, features)
                for pred in self.quality_predictors.values()
            ])
            
            # Actual rating
            actual = record.explicit_rating
            
            # Check calibration
            if abs(avg_pred - actual) < 0.2:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.5
    
    def _init_predictor(self) -> Dict:
        """Initialize quality predictor"""
        return {
            'weights': np.random.randn(10) * 0.01,
            'bias': 0.0
        }
    
    def _init_user_model(self) -> Dict:
        """Initialize user model"""
        return {
            'preferences': {
                'comprehension': 0.33,
                'trust': 0.33,
                'actionability': 0.34
            },
            'expertise_level': 'intermediate',
            'feedback_count': 0
        }
    
    def _update_user_model(self, user_id: str, feedback: FeedbackRecord):
        """Update user model with feedback"""
        model = self.user_models[user_id]
        model['feedback_count'] += 1
        
        # Update preferences based on interaction patterns
        if 'detailed_view' in feedback.interactions:
            model['preferences']['comprehension'] += 0.01
        if 'verify_sources' in feedback.interactions:
            model['preferences']['trust'] += 0.01
        if 'apply_recommendation' in feedback.interactions:
            model['preferences']['actionability'] += 0.01
        
        # Normalize
        total = sum(model['preferences'].values())
        for k in model['preferences']:
            model['preferences'][k] /= total
    
    def _extract_features(self, feedback: FeedbackRecord) -> np.ndarray:
        """Extract features from feedback"""
        return np.array([
            len(feedback.explanation) / 1000,  # Length
            feedback.dwell_time / 60,           # Dwell time (normalized)
            len(feedback.interactions),         # Interaction count
            feedback.explicit_rating,           # Explicit rating
            1.0 if 'technical' in feedback.explanation.lower() else 0.0,
            1.0 if 'quantitative' in feedback.explanation.lower() else 0.0,
            1.0 if 'causal' in feedback.explanation.lower() else 0.0,
            feedback.context.get('complexity', 0.5),
            feedback.context.get('urgency', 0.5),
            1.0  # Bias term
        ])
    
    def _compute_target_quality(self, feedback: FeedbackRecord, dimension: str) -> float:
        """Compute target quality from feedback"""
        # Combine explicit and implicit signals
        explicit = feedback.explicit_rating
        
        # Implicit signals
        implicit_score = 0.0
        if feedback.dwell_time > 30:
            implicit_score += 0.2
        if len(feedback.interactions) > 2:
            implicit_score += 0.2
        
        # Dimension-specific adjustments
        if dimension == 'comprehension' and 'detailed_view' in feedback.interactions:
            implicit_score += 0.1
        elif dimension == 'trust' and 'verify_sources' in feedback.interactions:
            implicit_score += 0.1
        elif dimension == 'actionability' and 'apply_recommendation' in feedback.interactions:
            implicit_score += 0.1
        
        return min((explicit + implicit_score) / 2, 1.0)
    
    def _predict_quality(self, predictor: Dict, features: np.ndarray) -> float:
        """Predict quality using linear model"""
        prediction = np.dot(predictor['weights'], features) + predictor['bias']
        return max(0, min(1, prediction))  # Clip to [0, 1]
    
    def _compute_uncertainty(self, features: np.ndarray) -> float:
        """Compute prediction uncertainty"""
        # Simple uncertainty based on feature variance
        return np.std(features) * 0.1
