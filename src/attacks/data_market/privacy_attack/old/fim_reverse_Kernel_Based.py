import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, cosine_similarity


def kernel_based_reconstruction(X_selected, X_unselected, x_test, gamma=0.1, n_components=5, lambda_reg=1e-5,
                                verbose=True):
    """
    Infers x_test using Kernel PCA and Ridge Regression.

    Parameters:
    - X_selected: Selected data matrix
    - X_unselected: Unselected data matrix
    - x_test: True x_test vector
    - gamma: Kernel coefficient for RBF kernel
    - n_components: Number of principal components for Kernel PCA
    - lambda_reg: Regularization parameter for Ridge Regression
    - verbose: If True, prints details

    Returns:
    - x_test_hat: Reconstructed x_test vector
    - mse: Mean Squared Error
    - cosine_sim: Cosine Similarity
    """
    # Combine data
    X_all = np.vstack((X_selected, X_unselected))

    # Fit Kernel PCA
    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma, fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(X_all)

    # Transform x_test
    x_test_kpca = kpca.transform(x_test.reshape(1, -1))

    # Ridge Regression to map from kernel space to original space
    ridge = Ridge(alpha=lambda_reg)
    ridge.fit(X_kpca, X_all)
    x_test_hat = ridge.predict(x_test_kpca)

    # Evaluation
    mse = mean_squared_error(x_test, x_test_hat)
    cosine_sim = cosine_similarity([x_test], [x_test_hat])[0, 0]

    if verbose:
        print(f"Kernel-Based Reconstruction MSE: {mse:.6f}")
        print(f"Cosine Similarity: {cosine_sim:.6f}")
        plot_reconstruction_comparison(x_test, x_test_hat.flatten(), title='Kernel-Based Reconstruction')

    return {
        'x_test_hat': x_test_hat.flatten(),
        'mse': mse,
        'cosine_similarity': cosine_sim
    }


# Example Usage
if __name__ == "__main__":
    from sklearn.metrics.pairwise import cosine_similarity

    # Generate synthetic data
    n_features = 10
    n_selected = 100
    n_unselected = 200
    gamma = 0.1
    n_components = 5
    lambda_reg = 1e-5

    x_test = np.random.randn(n_features)
    X_selected = x_test + np.random.randn(n_selected, n_features) * 0.1
    X_unselected = np.random.randn(n_unselected, n_features) + 2  # Shifted mean

    # Perform Kernel-Based Inference
    results = kernel_based_reconstruction(
        X_selected, X_unselected, x_test, gamma=gamma, n_components=n_components, lambda_reg=lambda_reg, verbose=True
    )

    # Summary
    print("\n--- Kernel-Based Inference Summary ---")
    print(f"Reconstructed x_test: {results['x_test_hat']}")
    print(f"MSE: {results['mse']:.6f}")
    print(f"Cosine Similarity: {results['cosine_similarity']:.6f}")
