import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sklearn.decomposition import PCA
from scipy.optimize import minimize

@dataclass
class SelectionMechanismInfo:
    """Information about selection mechanism"""
    selected_indices: np.ndarray
    selection_weights: np.ndarray
    info_matrix: Optional[np.ndarray] = None

class WhiteBoxAttack:
    """Attack with full knowledge of selection mechanism"""

    def __init__(self,
                 seller_embeddings: np.ndarray,
                 regularization: float = 1e-5):
        self.seller_embeddings = seller_embeddings
        self.regularization = regularization
        self.dim = seller_embeddings.shape[1]

        # Initialize tracking
        self.selection_history = []
        self.info_matrices = []

    def optimize_query_reconstruction(self,
                                    mechanism_info: SelectionMechanismInfo) -> Dict:
        """
        Reconstruct query by optimizing according to selection mechanism
        """
        self.selection_history.append(mechanism_info)

        # Setup optimization problem
        def objective(query):
            # Reshape query vector
            query = query.reshape(self.dim)

            # Calculate expected gradients for all seller points
            gradients = self._compute_expected_gradients(query)

            # Compare with actual selection
            selected_indices = mechanism_info.selected_indices
            weights = mechanism_info.selection_weights

            # Loss based on selection match
            selection_loss = -np.sum(gradients[selected_indices] * weights)

            # Regularization
            reg_loss = self.regularization * np.sum(query ** 2)

            return selection_loss + reg_loss

        # Initial guess using PCA of selected points
        initial_query = self._get_initial_guess(mechanism_info)

        # Optimize
        result = minimize(
            objective,
            initial_query,
            method='L-BFGS-B',
            jac=self._compute_gradient
        )

        reconstructed_query = result.x.reshape(self.dim)
        confidence = 1.0 / (1.0 + result.fun)

        return {
            'query': reconstructed_query,
            'confidence': confidence,
            'optimization_success': result.success
        }

    def _compute_expected_gradients(self, query: np.ndarray) -> np.ndarray:
        """Compute expected gradients for all seller points"""
        # Compute gradients according to DAVED's mechanism
        X = self.seller_embeddings
        info_matrix = self.selection_history[-1].info_matrix

        if info_matrix is None:
            # If no info matrix, use simpler gradient computation
            gradients = X @ query
        else:
            # Use full mechanism knowledge
            gradients = np.zeros(len(X))
            for i in range(len(X)):
                gradients[i] = query.T @ info_matrix @ X[i]

        return gradients

    def _compute_gradient(self, query: np.ndarray) -> np.ndarray:
        """Compute gradient of objective function"""
        query = query.reshape(self.dim)
        X = self.seller_embeddings
        selected_indices = self.selection_history[-1].selected_indices
        weights = self.selection_history[-1].selection_weights

        # Gradient of selection loss
        grad = -X[selected_indices].T @ weights

        # Add regularization gradient
        grad += 2 * self.regularization * query

        return grad.flatten()

    def _get_initial_guess(self,
                          mechanism_info: SelectionMechanismInfo) -> np.ndarray:
        """Get initial guess for optimization"""
        selected_points = self.seller_embeddings[mechanism_info.selected_indices]
        weights = mechanism_info.selection_weights

        # Use weighted PCA
        pca = PCA(n_components=1)
        pca.fit(selected_points, sample_weight=weights)

        return pca.components_[0]

class PartialKnowledgeAttack:
    """Attack with partial knowledge of selection mechanism"""

    def __init__(self,
                 seller_embeddings: np.ndarray,
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000):
        self.seller_embeddings = seller_embeddings
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.dim = seller_embeddings.shape[1]

    def reconstruct_query(self,
                         selection_info: SelectionMechanismInfo) -> Dict:
        """
        Reconstruct query using gradient descent and partial knowledge
        """
        # Initialize query estimate
        query = self._initialize_query(selection_info)

        # Gradient descent
        for _ in range(self.n_iterations):
            # Compute gradients for all points
            point_gradients = self.seller_embeddings @ query

            # Sort points by gradient magnitude
            sorted_indices = np.argsort(-np.abs(point_gradients))

            # Check if we match the actual selection
            selection_match = self._compute_selection_match(
                sorted_indices[:len(selection_info.selected_indices)],
                selection_info.selected_indices
            )

            # Update query based on match
            gradient_update = self._compute_update(
                query,
                selection_info.selected_indices,
                point_gradients
            )

            query += self.learning_rate * gradient_update

            # Normalize query
            query = query / np.linalg.norm(query)

            # Early stopping if selection matches well
            if selection_match > 0.9:
                break

        confidence = selection_match

        return {
            'query': query,
            'confidence': confidence
        }

    def _initialize_query(self,
                         selection_info: SelectionMechanismInfo) -> np.ndarray:
        """Initialize query estimate"""
        selected_points = self.seller_embeddings[selection_info.selected_indices]

        # Use first principal component as initial guess
        pca = PCA(n_components=1)
        pca.fit(selected_points)

        return pca.components_[0]

    def _compute_selection_match(self,
                               predicted_indices: np.ndarray,
                               true_indices: np.ndarray) -> float:
        """Compute how well our predicted selection matches true selection"""
        return len(set(predicted_indices) & set(true_indices)) / len(true_indices)

    def _compute_update(self,
                       query: np.ndarray,
                       selected_indices: np.ndarray,
                       point_gradients: np.ndarray) -> np.ndarray:
        """Compute update for query based on selection mismatch"""
        # Get gradients of selected points
        selected_gradients = point_gradients[selected_indices]

        # Get points with highest gradients
        top_indices = np.argsort(-np.abs(point_gradients))[:len(selected_indices)]
        top_gradients = point_gradients[top_indices]

        # Compute update to make selected points have higher gradients
        update = np.zeros_like(query)

        for i, idx in enumerate(selected_indices):
            # Add contribution to push up gradient of selected point
            update += self.seller_embeddings[idx]

            # Subtract contribution of wrongly selected point
            if top_indices[i] not in selected_indices:
                update -= self.seller_embeddings[top_indices[i]]

        return update

def evaluate_mechanism_attack(true_query: np.ndarray,
                            reconstructed_query: np.ndarray,
                            confidence: float) -> Dict:
    """Evaluate attack success"""
    # Normalize vectors
    true_norm = true_query / np.linalg.norm(true_query)
    pred_norm = reconstructed_query / np.linalg.norm(reconstructed_query)

    # Compute alignment
    alignment = np.abs(np.dot(true_norm, pred_norm))

    # Compute angular error
    angular_error = np.arccos(alignment) * 180 / np.pi

    return {
        'alignment': alignment,
        'angular_error_degrees': angular_error,
        'confidence': confidence,
        'success_score': alignment * confidence
    }