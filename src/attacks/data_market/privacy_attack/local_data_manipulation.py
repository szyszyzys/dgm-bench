import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from typing import List, Tuple, Optional
import seaborn as sns

from src.participants.seller.seller import AdversarySeller


###############################################################################
# 1) BUYER: Hidden Test Vector Simulation
###############################################################################

def simulate_hidden_test_vectors(num_vectors: int = 10, d: int = 2, seed: int = 0) -> np.ndarray:
    """
    Simulate the buyer's hidden test vectors X_test in R^d.
    Each vector is normalized to unit length.

    Parameters:
    - num_vectors: Number of hidden test vectors.
    - d: Dimensionality of the feature space.
    - seed: Random seed for reproducibility.

    Returns:
    - X_test: Array of shape (num_vectors, d).
    """
    np.random.seed(seed)
    X_test = np.random.normal(0, 1, size=(num_vectors, d))
    X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)  # Normalize
    return X_test.astype(np.float32)

###############################################################################
# 2) SELLER: Synthetic Covering Data Generation
###############################################################################

def generate_basis_vectors(n_basis: int = 10, d: int = 2, spread: float = 5.0, seed: int = 42) -> np.ndarray:
    """
    Generate a set of basis vectors uniformly spread in R^d.

    Parameters:
    - n_basis: Number of basis vectors.
    - d: Dimensionality of the feature space.
    - spread: Range for synthetic data generation.
    - seed: Random seed for reproducibility.

    Returns:
    - basis_vectors: Array of shape (n_basis, d).
    """
    np.random.seed(seed)
    if d == 2:
        angles = np.linspace(0, 2 * np.pi, n_basis, endpoint=False)
        basis_vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1) * spread
    else:
        # For higher dimensions, generate random unit vectors scaled by spread
        basis_vectors = np.random.normal(0, 1, size=(n_basis, d))
        basis_vectors /= np.linalg.norm(basis_vectors, axis=1, keepdims=True)
        basis_vectors *= spread
    return basis_vectors.astype(np.float32)

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
                      base_cost: float = 1.0,
                      cost_variation: float = 0.5,
                      seed: int = 999) -> np.ndarray:
    """
    Assign multiple cost tiers to each synthetic data point.

    Parameters:
    - n_points: Number of unique synthetic data points.
    - n_tiers: Number of cost tiers per data point.
    - base_cost: Base cost for the cheapest tier.
    - cost_variation: Maximum variation from the base cost.
    - seed: Random seed for reproducibility.

    Returns:
    - costs: Array of shape (n_points * n_tiers,).
    """
    np.random.seed(seed)
    costs = []
    for _ in range(n_points):
        tier_costs = np.linspace(base_cost * (1 - cost_variation),
                                 base_cost * (1 + cost_variation),
                                 n_tiers)
        # Shuffle to prevent ordered costs per tier
        np.random.shuffle(tier_costs)
        costs.extend(tier_costs)
    return np.array(costs, dtype=np.float32)

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
    X_synth = np.vstack([base_vectors, perturbations])
    cost_synth = assign_cost_tiers(n_points, n_tiers=n_tiers)
    return X_synth, cost_synth

###############################################################################
# 3) DAVED Selection (PLACEHOLDER)
###############################################################################

def daved_selection(x_test: np.ndarray,
                    X_synth: np.ndarray,
                    cost_synth: np.ndarray,
                    budget: float,
                    scoring_method: str = 'mahalanobis',
                    covariance_matrix: Optional[np.ndarray] = None,
                    gamma: float = 0.5) -> List[int]:
    """
    Placeholder for the DAVED selection mechanism.

    Parameters:
    - x_test: Hidden buyer test vector.
    - X_synth: Seller's synthetic data matrix (n x d).
    - cost_synth: Costs associated with each synthetic data point (n,).
    - budget: Total budget for selection.
    - scoring_method: 'mahalanobis', 'cosine', 'kernel_rbf'.
    - covariance_matrix: Covariance matrix for Mahalanobis distance.
    - gamma: Parameter for RBF kernel.

    Returns:
    - chosen_indices: List of indices of selected synthetic data points.

    Replace this function with your actual DAVED implementation.
    """
    n, d = X_synth.shape

    if scoring_method == 'mahalanobis':
        if covariance_matrix is None:
            covariance_matrix = np.cov(X_synth, rowvar=False) + 1e-6 * np.eye(d)  # Regularize
        inv_covmat = np.linalg.inv(covariance_matrix)
        # Compute Mahalanobis distance
        mahalanobis_dists = np.array([
            mahalanobis(x_test, X_synth[j], inv_covmat)
            for j in range(n)
        ])
        scores = -mahalanobis_dists  # Lower distance => higher score
    elif scoring_method == 'cosine':
        # Normalize vectors
        X_norm = X_synth / (np.linalg.norm(X_synth, axis=1, keepdims=True) + 1e-9)
        x_test_norm = x_test / (np.linalg.norm(x_test) + 1e-9)
        cosine_sim = X_norm.dot(x_test_norm)
        scores = cosine_sim  # Higher similarity => higher score
    elif scoring_method == 'kernel_rbf':
        # RBF kernel: exp(-gamma * ||x_j - x_test||^2)
        diffs = X_synth - x_test
        sq_dists = np.sum(diffs ** 2, axis=1)
        scores = np.exp(-gamma * sq_dists)
    else:
        raise ValueError("Unsupported scoring_method. Choose 'mahalanobis', 'cosine', or 'kernel_rbf'.")

    # Sort indices by descending scores
    sorted_indices = np.argsort(scores)[::-1]

    # Select points within budget
    chosen_indices = []
    current_budget = budget
    for idx in sorted_indices:
        if cost_synth[idx] <= current_budget:
            chosen_indices.append(idx)
            current_budget -= cost_synth[idx]
        if current_budget <= 0:
            break

    return chosen_indices


###############################################################################
# 5) ADVERSARY'S INFERENCE ATTACK: Constraint Solving
###############################################################################

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
        # Therefore, margin <= mahal_dists_unchosen - mahal_dists_chosen
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
            # d(D)/dx_hat = (- inv_covmat (x_j - x_hat) ) / D_j
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

###############################################################################
# 6) ADVERSARY'S INFERENCE ATTACK: Constraint Solving
###############################################################################

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

###############################################################################
# 7) VISUALIZATION
###############################################################################

def plot_results(
    X_synth: np.ndarray,
    chosen_indices: List[int],
    x_test: np.ndarray,
    x_hat: np.ndarray,
    title: str = 'Advanced Selection-Only Attack (2D)'
):
    """
    Plot synthetic data points, highlighting selected ones, and show true vs. inferred test vectors.
    Only applicable for 2D data.

    Parameters:
    - X_synth: Synthetic data matrix (n x d).
    - chosen_indices: Indices of selected synthetic data points.
    - x_test: The true hidden test vector.
    - x_hat: The reconstructed test vector.
    - title: Title for the plot.
    """
    if X_synth.shape[1] != 2:
        print("Visualization is only supported for 2D data.")
        return
    plt.figure(figsize=(10, 10))

    # Plot all synthetic points
    plt.scatter(X_synth[:,0], X_synth[:,1], c='lightgray', label='Synthetic Data')

    # Highlight chosen points
    chosen_X = X_synth[chosen_indices]
    plt.scatter(
        chosen_X[:,0],
        chosen_X[:,1],
        c='red',
        edgecolors='black',
        s=100,
        label='Selected Points'
    )

    # Plot true x_test
    plt.scatter(
        [x_test[0]],
        [x_test[1]],
        c='green',
        marker='*',
        s=300,
        label='True x_test'
    )

    # Plot inferred x_hat
    plt.scatter(
        [x_hat[0]],
        [x_hat[1]],
        c='blue',
        marker='X',
        s=200,
        label='Inferred x_hat'
    )

    # Draw arrows from origin to vectors
    plt.arrow(0, 0, x_test[0], x_test[1],
              color='green', width=0.05, label='True Direction')
    plt.arrow(0, 0, x_hat[0], x_hat[1],
              color='blue', width=0.05, label='Inferred Direction')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.close()

###############################################################################
# 8) MAIN PIPELINE EXECUTION
###############################################################################

def main_pipeline_demo(
    num_test_vectors: int = 10,  # Number of query samples
    d: int = 2,
    n_basis: int = 10,
    n_perturb: int = 3,
    perturb_radius: float = 0.5,
    n_tiers: int = 3,
    spread: float = 5.0,
    base_cost: float = 1.0,
    cost_variation: float = 0.5,
    budget: float = 10.0,
    scoring_method: str = 'mahalanobis',  # 'mahalanobis', 'cosine', 'kernel_rbf'
    gamma: float = 0.5,
    margin: float = 0.1,
    max_iter: int = 1000,
    lr: float = 0.01
):
    """
    Execute the full pipeline for an advanced selection-only privacy attack.

    Parameters:
    - num_test_vectors: Number of hidden test vectors.
    - d: Dimensionality of the feature space.
    - n_basis: Number of basis vectors.
    - n_perturb: Number of perturbations per basis vector.
    - perturb_radius: Maximum perturbation magnitude.
    - n_tiers: Number of cost tiers per data point.
    - spread: Range for synthetic data generation.
    - base_cost: Base cost for the cheapest tier.
    - cost_variation: Maximum variation from the base cost.
    - budget: Total budget for DAVED selection.
    - scoring_method: 'mahalanobis', 'cosine', 'kernel_rbf'.
    - gamma: Parameter for RBF kernel (only used if scoring_method='kernel_rbf').
    - margin: Margin for ranking loss in inversion.
    - max_iter: Maximum iterations for inversion optimization.
    - lr: Learning rate for inversion optimization.
    """
    print("\n=== Advanced Selection-Only Privacy Attack Pipeline ===\n")

    # A) Simulate buyer's hidden test vectors
    X_test = simulate_hidden_test_vectors(num_vectors=num_test_vectors, d=d, seed=0)
    print(f"[Buyer] Hidden Test Vectors (X_test):\n{X_test}\n")

    # B) Initialize adversary and generate synthetic data
    adversary = AdversarySeller(
        dataset=data,
        n_perturb=n_perturb,
        perturb_radius=perturb_radius,
        n_tiers=n_tiers,
        spread=spread,
        base_cost=base_cost,
        cost_variation=cost_variation
    )
    X_synth, cost_synth = adversary.get_synthetic_data()
    print(f"[Seller] Synthetic Data Points (First 5):\n{X_synth[:5]}")
    print(f"[Seller] Assigned Costs (First 5):\n{cost_synth[:5]}\n")

    # Initialize lists to store metrics
    all_cos_sim = []
    all_mse = []
    all_nrmse = []
    all_pearson = []

    # If multiple test vectors, loop through each
    for idx in range(num_test_vectors):
        print(f"--- Processing Test Vector {idx + 1} ---")
        x_test = X_test[idx]

        # Calculate covariance matrix if needed
        if scoring_method == 'mahalanobis':
            covariance_matrix = np.cov(X_synth, rowvar=False) + 1e-6 * np.eye(d)
        else:
            covariance_matrix = None  # Not used

        # Select data points using DAVED
        chosen_indices = daved_selection(
            x_test=x_test,
            X_synth=X_synth,
            cost_synth=cost_synth,
            budget=budget,
            scoring_method=scoring_method,
            covariance_matrix=covariance_matrix,
            gamma=gamma
        )
        print(f"[DAVED] Chosen Indices (Scoring: {scoring_method}): {chosen_indices}")

        # Adversary performs inference based on selected indices
        adversary.infer_x_test(
            chosen_indices=chosen_indices,
            scoring_method=scoring_method,
            covariance_matrix=covariance_matrix,
            gamma=gamma,
            margin=margin,
            max_iter=max_iter,
            lr=lr
        )
        x_hat = adversary.inferred_x_test
        print(f"[Adversary] Inferred x_hat: {x_hat}")

        # Evaluate the reconstruction
        cos_sim, mse = adversary.evaluate_attack(x_test)
        nrmse = np.sqrt(mse) / (np.linalg.norm(x_test) + 1e-9)
        pearson = np.corrcoef(x_test, x_hat)[0,1]
        print(f"Cosine Similarity: {cos_sim:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Normalized RMSE (NRMSE): {nrmse:.6f}")
        print(f"Pearson Correlation: {pearson:.4f}\n")

        # Store metrics
        all_cos_sim.append(cos_sim)
        all_mse.append(mse)
        all_nrmse.append(nrmse)
        all_pearson.append(pearson)

        # Optional: Visualize (only for 2D and first few vectors)
        if d == 2 and idx < 5:
            adversary.visualize_attack(
                x_test=x_test,
                chosen_indices=chosen_indices,
                title=f"Attack on Test Vector {idx + 1} (Scoring: {scoring_method})"
            )

    # After all vectors, compute aggregate metrics
    avg_cos_sim = np.mean(all_cos_sim)
    avg_mse = np.mean(all_mse)
    avg_nrmse = np.mean(all_nrmse)
    avg_pearson = np.mean(all_pearson)

    # Print aggregate metrics
    print("=== Aggregate Metrics ===")
    print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average NRMSE: {avg_nrmse:.6f}")
    print(f"Average Pearson Correlation: {avg_pearson:.4f}\n")

    # Plot metric distributions
    sns.set(style="whitegrid")

    plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    sns.histplot(all_cos_sim, kde=True, bins=20, color='skyblue')
    plt.title('Distribution of Cosine Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 2)
    sns.histplot(all_mse, kde=True, bins=20, color='salmon')
    plt.title('Distribution of MSEs')
    plt.xlabel('Mean Squared Error (MSE)')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 3)
    sns.histplot(all_nrmse, kde=True, bins=20, color='lightgreen')
    plt.title('Distribution of NRMSEs')
    plt.xlabel('Normalized RMSE (NRMSE)')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 4)
    sns.histplot(all_pearson, kde=True, bins=20, color='plum')
    plt.title('Distribution of Pearson Correlations')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.close()

    # Optional: Scatter plots of true vs inferred vectors
    plt.figure(figsize=(8, 6))
    plt.scatter(all_cos_sim, all_pearson, c='purple', edgecolor='k', alpha=0.7)
    plt.title('Cosine Similarity vs. Pearson Correlation')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Pearson Correlation')
    plt.grid(True)
    plt.close()

###############################################################################
# EXECUTE PIPELINE
###############################################################################

if __name__ == "__main__":
    main_pipeline_demo(
        num_test_vectors=10,  # Number of query samples
        d=2,
        n_basis=10,
        n_perturb=3,
        perturb_radius=0.5,
        n_tiers=3,
        spread=5.0,
        base_cost=1.0,
        cost_variation=0.5,
        budget=10.0,
        scoring_method='mahalanobis',  # Choose from 'mahalanobis', 'cosine', 'kernel_rbf'
        gamma=0.5,  # Only used if scoring_method='kernel_rbf'
        margin=0.1,
        max_iter=1000,
        lr=0.01
    )
