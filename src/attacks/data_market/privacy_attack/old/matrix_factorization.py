import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, cosine_similarity
import matplotlib.pyplot as plt


def matrix_factorization_reconstruction(X_selected, X_unselected, x_test, n_components=5, lambda_reg=1e-5,
                                        verbose=True):
    """
    Infers x_test using Singular Value Decomposition (SVD) based matrix factorization.

    Parameters:
    - X_selected: Selected data matrix
    - X_unselected: Unselected data matrix
    - x_test: True x_test vector
    - n_components: Number of latent factors
    - lambda_reg: Regularization parameter for Ridge Regression
    - verbose: If True, prints details

    Returns:
    - x_test_hat: Reconstructed x_test vector
    - mse: Mean Squared Error
    - cosine_sim: Cosine Similarity
    """
    # Combine data
    X_all = np.vstack((X_selected, X_unselected))

    # Apply Truncated SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_all)
    U = svd.components_.T  # Shape: (n_features, n_components)

    # Project x_test into latent space
    x_test_latent = x_test @ U  # Shape: (n_components,)

    # Ridge Regression to map latent space to original space
    ridge = Ridge(alpha=lambda_reg)
    ridge.fit(X_svd, X_all)
    x_hat = ridge.predict(x_test_latent.reshape(1, -1)).flatten()

    # Evaluation
    mse = mean_squared_error(x_test, x_hat)
    cosine_sim = cosine_similarity([x_test], [x_hat])[0, 0]

    if verbose:
        print(f"Matrix Factorization Reconstruction MSE: {mse:.6f}")
        print(f"Cosine Similarity: {cosine_sim:.6f}")
        plot_reconstruction_comparison(x_test, x_hat, title='Matrix Factorization Reconstruction')

    return {
        'x_test_hat': x_hat,
        'mse': mse,
        'cosine_similarity': cosine_sim
    }


def run_matrix_factorization_experiment(
        n_selected=100,
        n_unselected=200,
        n_features=10,
        n_components=5,
        lambda_reg=1e-5,
        verbose=True
):
    """
    Runs the Matrix Factorization Inference experiment.

    Parameters:
    - All parameters as defined above
    - verbose: If True, prints details

    Returns:
    - results: Dictionary containing the inference results
    """
    # Generate synthetic data
    x_test = np.random.randn(n_features)
    X_selected = x_test + np.random.randn(n_selected, n_features) * 0.1
    X_unselected = np.random.randn(n_unselected, n_features) + 2  # Shifted mean

    # Perform Matrix Factorization Reconstruction
    x_hat_mf = matrix_factorization_reconstruction(
        X_selected, X_unselected, x_test, n_components=n_components, lambda_reg=lambda_reg, verbose=verbose
    )

    return x_hat_mf


# Example Usage
if __name__ == "__main__":
    results = run_matrix_factorization_experiment()
    print("\n--- Matrix Factorization Inference Summary ---")
    print(f"Reconstructed x_test: {results['x_test_hat']}")
    print(f"MSE: {results['mse']:.6f}")
    print(f"Cosine Similarity: {results['cosine_similarity']:.6f}")
