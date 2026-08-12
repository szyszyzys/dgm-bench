from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class GradientAttackResult:
    """Results from gradient-based attack"""
    predicted_query: np.ndarray
    confidence_score: float
    importance_scores: np.ndarray
    identified_directions: np.ndarray
    temporal_patterns: Dict


class GradientAttack:
    """Advanced gradient-based attack using DAVED's gradient information"""

    def __init__(self,
                 seller_embeddings: np.ndarray,
                 window_size: int = 3,
                 use_temporal: bool = True,
                 min_confidence: float = 0.7):
        """
        Args:
            seller_embeddings: Seller's data points
            window_size: Window size for temporal analysis
            use_temporal: Whether to use temporal patterns
            min_confidence: Minimum confidence threshold
        """
        self.seller_embeddings = seller_embeddings
        self.window_size = window_size
        self.use_temporal = use_temporal
        self.min_confidence = min_confidence
        self.embedding_dim = seller_embeddings.shape[1]

        # Initialize tracking
        self.gradient_history = []
        self.weight_history = []
        self.temporal_patterns = {}

    def analyze_gradient_round(self,
                               gradient: np.ndarray,
                               weights: np.ndarray) -> Dict:
        """Analyzes gradient information from one round"""

        # Store history
        self.gradient_history.append(gradient)
        self.weight_history.append(weights)

        # Analyze gradient properties
        grad_magnitude = np.abs(gradient)
        top_indices = np.argsort(grad_magnitude)[::-1][:self.window_size]

        # Find most influential points
        influential_points = self.seller_embeddings[top_indices]
        influence_weights = grad_magnitude[top_indices]
        influence_weights = influence_weights / np.sum(influence_weights)

        # Estimate local direction
        weighted_direction = np.average(
            influential_points,
            weights=influence_weights,
            axis=0
        )
        weighted_direction /= np.linalg.norm(weighted_direction)

        return {
            'top_indices': top_indices,
            'influence_weights': influence_weights,
            'local_direction': weighted_direction,
            'gradient_stats': {
                'mean': np.mean(gradient),
                'std': np.std(gradient),
                'max': np.max(np.abs(gradient))
            }
        }

    def analyze_temporal_patterns(self) -> Dict:
        """Analyzes temporal patterns in gradients"""
        if len(self.gradient_history) < self.window_size:
            return {}

        # Stack recent gradients
        recent_grads = np.stack(self.gradient_history[-self.window_size:])

        # Analyze gradient evolution
        grad_changes = np.diff(recent_grads, axis=0)
        grad_velocities = np.mean(grad_changes, axis=0)

        # Analyze consistency
        temporal_consistency = []
        for i in range(1, len(self.gradient_history)):
            curr_grad = self.gradient_history[i]
            prev_grad = self.gradient_history[i - 1]

            # Compute directional consistency
            consistency = np.abs(
                np.dot(curr_grad, prev_grad) /
                (np.linalg.norm(curr_grad) * np.linalg.norm(prev_grad))
            )
            temporal_consistency.append(consistency)

        return {
            'consistency': np.mean(temporal_consistency),
            'grad_velocity': grad_velocities,
            'evolution': grad_changes
        }

    def identify_query_direction(self) -> Tuple[np.ndarray, float]:
        """Identifies likely query direction using gradient history"""

        # Get weighted average of local directions
        weighted_directions = []
        weights = []

        for i, grad in enumerate(self.gradient_history):
            analysis = self.analyze_gradient_round(grad, self.weight_history[i])
            weighted_directions.append(analysis['local_direction'])
            weights.append(np.max(np.abs(grad)))

        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Compute weighted direction
        directions = np.stack(weighted_directions)
        query_direction = np.average(directions, weights=weights, axis=0)
        query_direction /= np.linalg.norm(query_direction)

        # Compute confidence based on consistency
        direction_similarities = np.abs(directions @ query_direction)
        confidence = np.mean(direction_similarities)

        return query_direction, confidence

    def estimate_query_magnitude(self,
                                 query_direction: np.ndarray) -> float:
        """Estimates magnitude of query using gradient information"""

        # Project seller points onto query direction
        projections = self.seller_embeddings @ query_direction

        # Use gradient magnitudes to estimate scale
        grad_scales = [np.max(np.abs(g)) for g in self.gradient_history]
        avg_scale = np.mean(grad_scales)

        # Estimate magnitude based on projections and gradient scale
        magnitude = np.std(projections) * avg_scale
        return magnitude

    def run_attack(self, attack_data: Dict) -> GradientAttackResult:
        """
        Main attack function

        Args:
            attack_data: Dictionary containing:
                - gradients: List of gradient information
                - selected_coords: Selected coordinates
                - weights: Selection weights
        """
        # Reset tracking
        self.gradient_history = []
        self.weight_history = []

        # Process each round
        for i in range(len(attack_data['gradients'])):
            gradient = attack_data['gradients'][i]
            weights = attack_data['weights'][i]

            # Analyze round
            round_analysis = self.analyze_gradient_round(gradient, weights)

            if self.use_temporal and i >= self.window_size - 1:
                temporal_analysis = self.analyze_temporal_patterns()
                self.temporal_patterns[i] = temporal_analysis

        # Identify query direction and confidence
        query_direction, confidence = self.identify_query_direction()

        # Only proceed if confidence is high enough
        if confidence < self.min_confidence:
            # Fall back to simpler estimation if confidence is low
            query_direction = self._fallback_estimation()
            confidence *= 0.8  # Reduce confidence for fallback

        # Estimate magnitude
        magnitude = self.estimate_query_magnitude(query_direction)

        # Reconstruct query
        predicted_query = query_direction * magnitude

        # Compute importance scores for seller points
        importance_scores = self._compute_importance_scores(predicted_query)

        # Get principal directions of variation
        pca = PCA(n_components=min(3, self.embedding_dim))
        pca.fit(np.stack(self.gradient_history))
        identified_directions = pca.components_

        return GradientAttackResult(
            predicted_query=predicted_query,
            confidence_score=confidence,
            importance_scores=importance_scores,
            identified_directions=identified_directions,
            temporal_patterns=self.temporal_patterns
        )

    def _fallback_estimation(self) -> np.ndarray:
        """Fallback method for low-confidence cases"""
        # Use PCA on gradient history
        pca = PCA(n_components=1)
        pca.fit(np.stack(self.gradient_history))
        return pca.components_[0]

    def _compute_importance_scores(self,
                                   predicted_query: np.ndarray) -> np.ndarray:
        """Computes importance scores for seller points"""
        # Compute similarities to predicted query
        similarities = self.seller_embeddings @ predicted_query
        similarities = np.abs(similarities)

        # Normalize
        importance_scores = similarities / np.max(similarities)
        return importance_scores


class GradientAttackEvaluator:
    """Evaluates success of gradient-based attack"""

    def __init__(self, true_query: np.ndarray):
        self.true_query = true_query

    def evaluate(self, attack_result: GradientAttackResult) -> Dict:
        """Evaluates attack success"""
        # Normalize vectors
        true_norm = self.true_query / np.linalg.norm(self.true_query)
        pred_norm = attack_result.predicted_query / np.linalg.norm(attack_result.predicted_query)

        # Compute alignment
        alignment = np.abs(np.dot(true_norm, pred_norm))

        # Compute magnitude error
        magnitude_error = np.abs(
            np.linalg.norm(self.true_query) -
            np.linalg.norm(attack_result.predicted_query)
        ) / np.linalg.norm(self.true_query)

        # Compute directional error for principal components
        direction_errors = []
        for direction in attack_result.identified_directions:
            error = np.abs(np.dot(direction, true_norm))
            direction_errors.append(error)

        return {
            'alignment_score': alignment,
            'magnitude_error': magnitude_error,
            'direction_errors': direction_errors,
            'confidence_score': attack_result.confidence_score,
            'success_score': alignment * (1 - magnitude_error) * attack_result.confidence_score
        }
