import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
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


def bayesian_inference(X_selected, X_unselected, x_test_true, epsilon=1e-3, lambda_reg=1e-5, verbose=True):
    """
    Performs Bayesian inference to estimate x_test.

    Parameters:
    - X_selected: Selected data matrix
    - X_unselected: Unselected data matrix
    - x_test_true: True x_test vector (for evaluation)
    - epsilon: Weight for unselected data
    - lambda_reg: Regularization parameter
    - verbose: If True, prints details

    Returns:
    - x_test_posterior: Estimated x_test vector
    - posterior_mean: Mean of the posterior distribution
    """
    n_features = X_selected.shape[1]

    # Define prior (e.g., standard normal)
    mu_prior = np.zeros(n_features)
    Sigma_prior = np.eye(n_features)

    # Compute extended FIM and its inverse
    fim = compute_extended_fim(X_selected, X_unselected, epsilon=epsilon, lambda_reg=lambda_reg)
    try:
        fim_inv = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        fim += 1e-3 * np.eye(fim.shape[0])
        fim_inv = np.linalg.inv(fim)

    # Posterior covariance
    Sigma_posterior = np.linalg.inv(fim_inv + np.linalg.inv(Sigma_prior))

    # Posterior mean
    x_posterior = Sigma_posterior @ (fim_inv @ np.mean(X_selected, axis=0) + np.linalg.inv(Sigma_prior) @ mu_prior)

    # Optimization-based posterior estimation
    def negative_log_posterior(x):
        # Negative log prior
        nlp = 0.5 * (x - mu_prior).T @ np.linalg.inv(Sigma_prior) @ (x - mu_prior)
        # Negative log likelihood
        nll = 0.5 * x.T @ fim_inv @ x
        return nlp + nll

    initial_guess = np.mean(X_selected, axis=0)
    result = minimize(negative_log_posterior, initial_guess, method='L-BFGS-B')
    x_posterior_opt = result.x

    # Evaluation
    mse = np.mean((x_test_true - x_posterior_opt) ** 2)
    cosine_sim = cosine_similarity([x_test_true], [x_posterior_opt])[0, 0]

    if verbose:
        print(f"Bayesian Inference MSE: {mse:.6f}")
        print(f"Cosine Similarity: {cosine_sim:.6f}")
        plot_reconstruction_comparison(x_test_true, x_posterior_opt, title='Bayesian Inference Reconstruction')

    return {
        'posterior_mean': x_posterior,
        'x_test_posterior': x_posterior_opt,
        'mse': mse,
        'cosine_similarity': cosine_sim
    }


def plot_reconstruction_comparison(x_true, x_hat, title='Reconstruction Comparison'):
    """
    Plots the true test data vs. reconstructed test data.
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


# Example Usage
if __name__ == "__main__":
    from sklearn.metrics.pairwise import cosine_similarity

    # Generate synthetic data
    n_features = 10
    n_selected = 100
    n_unselected = 200
    epsilon = 1e-3
    lambda_reg = 1e-5

    x_test = np.random.randn(n_features)
    X_selected = x_test + np.random.randn(n_selected, n_features) * 0.1
    X_unselected = np.random.randn(n_unselected, n_features) + 2  # Shifted mean

    # Perform Bayesian Inference
    results = bayesian_inference(X_selected, X_unselected, x_test, epsilon=epsilon, lambda_reg=lambda_reg, verbose=True)

    # Summary
    print("\n--- Bayesian Inference Summary ---")
    print(f"Posterior Mean: {results['posterior_mean']}")
    print(f"Reconstructed x_test: {results['x_test_posterior']}")
    print(f"MSE: {results['mse']:.6f}")
    print(f"Cosine Similarity: {results['cosine_similarity']:.6f}")
