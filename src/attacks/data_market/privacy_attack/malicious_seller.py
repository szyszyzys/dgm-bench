from typing import List, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt

from src.participants.seller.data_seller import DataSeller


def evaluate_reconstruction(x_true: np.ndarray, x_hat: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate the reconstruction accuracy.

    Returns:
    - cosine_similarity: Cosine similarity between x_true and x_hat.
    - mse: Mean squared error between x_true and x_hat.
    """
    cosine_similarity = np.dot(x_true, x_hat) / (np.linalg.norm(x_true) * np.linalg.norm(x_hat) + 1e-9)
    mse = np.mean((x_true - x_hat) ** 2)
    return cosine_similarity, mse


def reconstruct_x_test_from_selection_mahalanobis(
        X_synth: np.ndarray,
        chosen_indices: List[int],
        covariance_matrix: np.ndarray,
        margin: float = 0.1,
        max_iter: int = 1000,
        lr: float = 0.01
) -> np.ndarray:
    """
    Reconstruct the hidden test vector x_test using Mahalanobis-based scoring.

    Enforces that selected points have lower Mahalanobis distance than unselected points by a margin.

    Parameters:
    - X_synth: Synthetic data matrix (n x d).
    - chosen_indices: Indices of selected synthetic data points.
    - covariance_matrix: Covariance matrix for Mahalanobis distance.
    - margin: Minimum required difference between selected and unselected distances.
    - max_iter: Maximum number of optimization iterations.
    - lr: Learning rate for gradient descent.

    Returns:
    - x_hat: Reconstructed test vector.
    """
    n, d = X_synth.shape
    chosen_mask = np.zeros(n, dtype=bool)
    chosen_mask[chosen_indices] = True
    unchosen_indices = np.where(~chosen_mask)[0]

    inv_covmat = np.linalg.inv(covariance_matrix)

    # Initialize x_hat randomly and normalize
    np.random.seed(1001)
    x_hat = np.random.normal(0, 1, size=(d,))
    x_hat /= np.linalg.norm(x_hat)

    for iteration in range(max_iter):
        # Compute Mahalanobis distances
        diffs_chosen = X_synth[chosen_mask] - x_hat  # (k, d)
        diffs_unchosen = X_synth[~chosen_mask] - x_hat  # (n - k, d)

        mahal_dists_chosen = np.sqrt(np.sum(diffs_chosen @ inv_covmat * diffs_chosen, axis=1))
        mahal_dists_unchosen = np.sqrt(np.sum(diffs_unchosen @ inv_covmat * diffs_unchosen, axis=1))

        # Compute violations
        # We want mahal_dists_chosen + margin <= mahal_dists_unchosen
        violations = mahal_dists_chosen[:, np.newaxis] + margin - mahal_dists_unchosen[np.newaxis, :]
        violations = violations.flatten()
        active = violations > 0  # Only consider positive violations

        # Compute loss
        loss = np.sum(violations[active])

        if loss < 1e-6:
            print(f"Converged at iteration {iteration}. Loss: {loss:.6f}")
            break

        # Compute gradient
        grad = np.zeros_like(x_hat)
        for idx in np.where(active)[0]:
            j = idx // len(mahal_dists_unchosen)
            ell = idx % len(mahal_dists_unchosen)
            # Gradient of Mahalanobis distance w.r.t x_hat
            grad += inv_covmat @ (diffs_unchosen[ell] - diffs_chosen[j]) / (mahal_dists_unchosen[ell] + 1e-9)

        # Update x_hat with gradient descent
        x_hat -= lr * grad

        # Normalize to prevent explosion
        norm = np.linalg.norm(x_hat)
        if norm > 0:
            x_hat /= norm

        # Optional: print progress every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: Loss = {loss:.4f}")

    return x_hat


def reconstruct_x_test_from_selection_cosine(
        X_synth: np.ndarray,
        chosen_indices: List[int],
        margin: float = 0.1,
        max_iter: int = 1000,
        lr: float = 0.01
) -> np.ndarray:
    """
    Reconstruct the hidden test vector x_test using Cosine Similarity-based scoring.

    Enforces that selected points have higher cosine similarity than unselected points by a margin.

    Parameters:
    - X_synth: Synthetic data matrix (n x d).
    - chosen_indices: Indices of selected synthetic data points.
    - margin: Minimum required difference between selected and unselected similarities.
    - max_iter: Maximum number of optimization iterations.
    - lr: Learning rate for gradient ascent.

    Returns:
    - x_hat: Reconstructed test vector.
    """
    n, d = X_synth.shape
    chosen_mask = np.zeros(n, dtype=bool)
    chosen_mask[chosen_indices] = True
    unchosen_indices = np.where(~chosen_mask)[0]

    # Normalize synthetic data
    X_norm = X_synth / (np.linalg.norm(X_synth, axis=1, keepdims=True) + 1e-9)

    # Initialize x_hat randomly and normalize
    np.random.seed(1002)
    x_hat = np.random.normal(0, 1, size=(d,))
    x_hat /= np.linalg.norm(x_hat)

    for iteration in range(max_iter):
        # Compute cosine similarities
        cosine_sim_chosen = X_norm[chosen_mask].dot(x_hat)  # Shape: (k,)
        cosine_sim_unchosen = X_norm[~chosen_mask].dot(x_hat)  # Shape: (n - k,)

        # Compute violations
        # We want cosine_sim_chosen >= cosine_sim_unchosen + margin
        violations = cosine_sim_unchosen[:, np.newaxis] + margin - cosine_sim_chosen[np.newaxis, :]
        violations = violations.flatten()
        active = violations > 0  # Only consider positive violations

        # Compute loss
        loss = np.sum(violations[active])

        if loss < 1e-6:
            print(f"Converged at iteration {iteration}. Loss: {loss:.6f}")
            break

        # Compute gradient
        grad = np.zeros_like(x_hat)
        for idx in np.where(active)[0]:
            j = idx // len(cosine_sim_unchosen)
            ell = idx % len(cosine_sim_unchosen)
            # Gradient of cosine similarity w.r.t x_hat
            grad += X_norm[~chosen_mask][ell] - X_norm[chosen_mask][j]

        # Update x_hat with gradient ascent
        x_hat += lr * grad

        # Normalize to prevent explosion
        norm = np.linalg.norm(x_hat)
        if norm > 0:
            x_hat /= norm

        # Optional: print progress every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: Loss = {loss:.4f}")

    return x_hat


def reconstruct_x_test_from_selection_kernel(
        X_synth: np.ndarray,
        chosen_indices: List[int],
        gamma: float = 0.5,
        margin: float = 0.1,
        max_iter: int = 1000,
        lr: float = 0.01
) -> np.ndarray:
    """
    Reconstruct the hidden test vector x_test using RBF Kernel-based scoring.

    Enforces that selected points have higher kernel similarity than unselected points by a margin.

    Parameters:
    - X_synth: Synthetic data matrix (n x d).
    - chosen_indices: Indices of selected synthetic data points.
    - gamma: Parameter for RBF kernel.
    - margin: Minimum required difference between selected and unselected similarities.
    - max_iter: Maximum number of optimization iterations.
    - lr: Learning rate for gradient ascent.

    Returns:
    - x_hat: Reconstructed test vector.
    """
    n, d = X_synth.shape
    chosen_mask = np.zeros(n, dtype=bool)
    chosen_mask[chosen_indices] = True
    unchosen_indices = np.where(~chosen_mask)[0]

    # Initialize x_hat randomly and normalize
    np.random.seed(1003)
    x_hat = np.random.normal(0, 1, size=(d,))
    x_hat /= np.linalg.norm(x_hat)

    for iteration in range(max_iter):
        # Compute RBF kernel similarities
        diffs_chosen = X_synth[chosen_mask] - x_hat  # (k, d)
        diffs_unchosen = X_synth[~chosen_mask] - x_hat  # (n - k, d)

        sq_dists_chosen = np.sum(diffs_chosen ** 2, axis=1)
        sq_dists_unchosen = np.sum(diffs_unchosen ** 2, axis=1)

        kernel_sim_chosen = np.exp(-gamma * sq_dists_chosen)
        kernel_sim_unchosen = np.exp(-gamma * sq_dists_unchosen)

        # Compute violations
        # We want kernel_sim_chosen >= kernel_sim_unchosen + margin
        violations = kernel_sim_unchosen[:, np.newaxis] + margin - kernel_sim_chosen[np.newaxis, :]
        violations = violations.flatten()
        active = violations > 0  # Only consider positive violations

        # Compute loss
        loss = np.sum(violations[active])

        if loss < 1e-6:
            print(f"Converged at iteration {iteration}. Loss: {loss:.6f}")
            break

        # Compute gradient
        grad = np.zeros_like(x_hat)
        for idx in np.where(active)[0]:
            j = idx // len(kernel_sim_unchosen)
            ell = idx % len(kernel_sim_unchosen)
            # Gradient of RBF kernel: 2 * gamma * (x_j - x_hat) * kernel_sim_j
            grad += 2 * gamma * (X_synth[~chosen_mask][ell] - x_hat) * kernel_sim_unchosen[ell] - \
                    2 * gamma * (X_synth[chosen_mask][j] - x_hat) * kernel_sim_chosen[j]

        # Update x_hat with gradient ascent
        x_hat += lr * grad

        # Normalize to prevent explosion
        norm = np.linalg.norm(x_hat)
        if norm > 0:
            x_hat /= norm

        # Optional: print progress every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: Loss = {loss:.4f}")

    return x_hat


def generate_perturbations(base_vectors: np.ndarray,
                           n_perturb: int = 3,
                           perturb_radius: float = 0.5,
                           seed: int = 24) -> np.ndarray:
    """
    For each base vector, generate multiple perturbations within a specified radius.

    Parameters:
    - base_vectors: Array of shape (n_basis, d).
    - n_perturb: Number of perturbations per base vector.
    - perturb_radius: Maximum perturbation magnitude.
    - seed: Random seed for reproducibility.

    Returns:
    - perturbations: Array of shape (n_basis * n_perturb, d).
    """
    np.random.seed(seed)
    n_basis, d = base_vectors.shape
    perturbations = []
    for v in base_vectors:
        for _ in range(n_perturb):
            delta = np.random.uniform(-perturb_radius, perturb_radius, size=d)
            perturbed_v = v + delta
            perturbations.append(perturbed_v)
    return np.array(perturbations, dtype=np.float32)


def assign_cost_tiers(n_points: int,
                      n_tiers: int = 3,
                      base_cost: float = 3.0,
                      cost_variation: float = 0.5,
                      seed: int = 999) -> np.ndarray:
    """
    Assign multiple price tiers for each data point.

    Parameters:
    - n_points: Number of unique data points.
    - n_tiers: Number of price tiers per data point.
    - base_cost: Base cost for the cheapest tier.
    - cost_variation: Maximum variation from base cost.
    - seed: Random seed for reproducibility.

    Returns:
    - costs: Array of shape (n_points * n_tiers,), where consecutive n_tiers
            elements represent different prices for the same data point.
    """
    np.random.seed(seed)
    costs = []

    # For each data point, create n_tiers different prices
    for i in range(n_points):
        point_costs = np.linspace(base_cost * (1 - cost_variation),
                                  base_cost * (1 + cost_variation),
                                  n_tiers)
        np.random.shuffle(point_costs)  # Randomize price tiers
        costs.append(point_costs)

    # Reshape to have n_tiers consecutive prices for each point
    return np.array(costs).flatten().astype(np.float32)


def create_synthetic_dataset(base_vectors: np.ndarray,
                             perturbations: np.ndarray,
                             n_tiers: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine base vectors and their perturbations, assigning multi-tier costs.

    Parameters:
    - base_vectors: Array of shape (n_basis, d).
    - perturbations: Array of shape (n_basis * n_perturb, d).
    - n_tiers: Number of cost tiers per data point.

    Returns:
    - X_synth: Combined synthetic data points.
    - cost_synth: Corresponding costs.
    """
    n_points = base_vectors.shape[0] + perturbations.shape[0]
    X_synth_base = np.vstack([base_vectors, perturbations])

    # Repeat each data point n_tiers times
    X_synth = np.repeat(X_synth_base, n_tiers, axis=0)

    # Generate corresponding costs
    cost_synth = assign_cost_tiers(len(X_synth_base), n_tiers=n_tiers)

    return X_synth, cost_synth


class MaliciousDataSeller(DataSeller):
    """Malicious seller attempting to infer buyer's hidden test vectors"""

    def __init__(self,
                 seller_id: str,
                 dataset: np.ndarray,
                 **kwargs):
        super().__init__(seller_id, dataset, **kwargs)
        self.inferred_x_test = None

    def perturb_data(self, n_perturb: int = 3, radius: float = 0.5):
        """Add perturbations to base data"""
        self.perturbations = generate_perturbations(
            base_vectors=self.dataset,
            n_perturb=n_perturb,
            perturb_radius=radius
        )
        X_synth = np.vstack([self.dataset, self.perturbations])
        self.cur_data = X_synth
        self.stats.total_points = len(self.cur_data)

    def modify_prices(self, n_tiers: int = 3, base_cost: float = 1.0):
        """Modify pricing structure"""

        X_tiered = np.repeat(self.cur_data, n_tiers, axis=0)
        new_prices = assign_cost_tiers(
            len(self.cur_data),
            n_tiers=n_tiers,
            base_cost=base_cost
        )
        self.cur_data = X_tiered
        self.cur_price = new_prices
        self.stats.total_points = len(self.cur_data)

    def reset_data(self):
        """Reset to original dataset"""
        self.cur_data = self.dataset
        self.cur_price = self.prices
        self.stats.total_points = len(self.cur_data)

    def infer_x_test(self,
                     chosen_indices: List[int],
                     scoring_method: str = 'mahalanobis',
                     **kwargs):
        """Infer buyer's test vector from selections"""
        if scoring_method == 'mahalanobis':
            covariance_matrix = kwargs.get('covariance_matrix',
                                           np.cov(self.cur_data, rowvar=False) + 1e-6 * np.eye(self.cur_data.shape[1]))
            self.inferred_x_test = reconstruct_x_test_from_selection_mahalanobis(
                X_synth=self.cur_data,
                chosen_indices=chosen_indices,
                covariance_matrix=covariance_matrix,
                **kwargs
            )
        elif scoring_method == 'cosine':
            self.inferred_x_test = reconstruct_x_test_from_selection_cosine(
                X_synth=self.cur_data,
                chosen_indices=chosen_indices,
                **kwargs
            )
        elif scoring_method == 'kernel_rbf':
            self.inferred_x_test = reconstruct_x_test_from_selection_kernel(
                X_synth=self.cur_data,
                chosen_indices=chosen_indices,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported scoring method: {scoring_method}")

        # Record selection
        self.record_selection(chosen_indices, f"buyer_{len(self.selection_history)}")

        return self.inferred_x_test

    def evaluate_attack(self, x_test: np.ndarray) -> Dict:
        """Evaluate attack accuracy"""
        if self.inferred_x_test is None:
            raise ValueError("Run infer_x_test first")

        cos_sim, mse = evaluate_reconstruction(x_test, self.inferred_x_test)

        return {
            'cosine_similarity': cos_sim,
            'mse': mse,
            'selections': len(self.selection_history),
            'total_revenue': self.stats.revenue
        }

    def visualize_attack(self, x_test: np.ndarray, chosen_indices: List[int], **kwargs):
        """Visualize attack results"""
        if self.cur_data.shape[1] != 2:
            raise ValueError("Visualization only supported for 2D data")
        if self.inferred_x_test is None:
            raise ValueError("Run infer_x_test first")

        plt.figure(figsize=(10, 10))
        plt.scatter(self.cur_data[:, 0], self.cur_data[:, 1], c='lightgray', label='Synthetic Data')

        chosen_X = self.cur_data[chosen_indices]
        plt.scatter(chosen_X[:, 0], chosen_X[:, 1], c='red', edgecolors='black', s=100, label='Selected')

        plt.scatter([x_test[0]], [x_test[1]], c='green', marker='*', s=300, label='True x_test')
        plt.scatter([self.inferred_x_test[0]], [self.inferred_x_test[1]], c='blue', marker='X', s=200, label='Inferred')

        plt.arrow(0, 0, x_test[0], x_test[1], color='green', width=0.05, label='True')
        plt.arrow(0, 0, self.inferred_x_test[0], self.inferred_x_test[1], color='blue', width=0.05, label='Inferred')

        plt.title(kwargs.get('title', f'Attack Results for {self.seller_id}'))
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.close()

    def get_statistics(self) -> Dict:
        """Get enhanced statistics including attack metrics"""
        stats = super().get_statistics()
        if self.inferred_x_test is not None:
            stats.update({
                'attack_success_rate': len(self.selection_history) / self.stats.total_points,
                'avg_selection_size': np.mean(
                    [rec['n_points'] for rec in self.selection_history]) if self.selection_history else 0
            })
        return stats
