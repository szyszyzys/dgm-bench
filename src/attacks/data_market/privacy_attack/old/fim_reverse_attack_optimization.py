import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


def compute_extended_fim(X_selected, X_unselected, epsilon=1e-3, lambda_reg=1e-5):
    """
    Constructs the extended Fisher Information Matrix (FIM) using selected and unselected data.
    """
    W_selected = np.ones(X_selected.shape[0])
    I_selected = X_selected.T @ (W_selected[:, np.newaxis] * X_selected)

    W_unselected = epsilon * np.ones(X_unselected.shape[0])
    I_unselected = X_unselected.T @ (W_unselected[:, np.newaxis] * X_unselected)

    fim = I_selected + I_unselected + lambda_reg * np.eye(X_selected.shape[1])
    return fim


def optimization_based_reconstruction(fim_inv, initial_guess, bounds=None):
    """
    Infers x_test by minimizing the loss function L(x) = x^T I(w)^-1 x.

    Parameters:
    - fim_inv: Inverse of the extended FIM matrix
    - initial_guess: Initial guess for x_test
    - bounds: Bounds for each feature in x_test

    Returns:
    - result.x: Optimized x_test vector
    """

    def loss(x):
        return x.T @ fim_inv @ x

    result = minimize(loss, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x


def plot_reconstruction_comparison(x_true, x_hat, title='Reconstruction Comparison'):
    """
    Plots the true vs. reconstructed x_test.
    """
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(x_true))
    plt.plot(indices, x_true, label='True x_test', marker='o')
    plt.plot(indices, x_hat, label='Reconstructed x_test', marker='x')
    plt.title(title)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True)
    plt.close()


def run_optimization_based_experiment(
        n_selected=100,
        n_unselected=200,
        n_features=10,
        epsilon=1e-3,
        lambda_reg=1e-5,
        verbose=True
):
    """
    Executes the Optimization-Based Reconstruction experiment.
    """
    # Generate synthetic data
    x_test = np.random.randn(n_features)
    X_selected = x_test + np.random.randn(n_selected, n_features) * 0.1
    X_unselected = np.random.randn(n_unselected, n_features) + 2  # Shifted mean

    # Compute extended FIM and its inverse
    fim = compute_extended_fim(X_selected, X_unselected, epsilon=epsilon, lambda_reg=lambda_reg)
    try:
        fim_inv = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        fim += 1e-3 * np.eye(fim.shape[0])
        fim_inv = np.linalg.inv(fim)

    # Initial guess for optimization (mean of selected data)
    initial_guess = np.mean(X_selected, axis=0)

    # Define bounds (optional, based on feature ranges)
    feature_bounds = [(None, None) for _ in range(n_features)]  # No bounds

    # Perform optimization
    x_hat = optimization_based_reconstruction(fim_inv, initial_guess, bounds=feature_bounds)

    # Evaluate reconstruction
    mse = np.mean((x_test - x_hat) ** 2)
    cosine_sim = cosine_similarity([x_test], [x_hat])[0, 0]

    if verbose:
        print(f"Optimization-Based Reconstruction MSE: {mse:.6f}")
        print(f"Cosine Similarity: {cosine_sim:.6f}")
        plot_reconstruction_comparison(x_test, x_hat, title='Optimization-Based Reconstruction')

    return {
        'x_test': x_test,
        'x_hat': x_hat,
        'mse': mse,
        'cosine_similarity': cosine_sim
    }


# Example Usage
if __name__ == "__main__":
    from sklearn.metrics.pairwise import cosine_similarity

    results = run_optimization_based_experiment()
