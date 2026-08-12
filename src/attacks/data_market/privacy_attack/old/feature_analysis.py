import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, cosine_similarity
import matplotlib.pyplot as plt


def mutual_information_feature_analysis(X_selected, X_unselected, x_test, bins=10, verbose=True):
    """
    Performs mutual information analysis between data features and x_test.

    Parameters:
    - X_selected: Selected data matrix
    - X_unselected: Unselected data matrix
    - x_test: True x_test vector
    - bins: Number of bins for discretization (if needed)
    - verbose: If True, prints details

    Returns:
    - mi_scores: Mutual information scores for each feature
    """
    # Combine selected and unselected data
    X_all = np.vstack((X_selected, X_unselected))
    # Create target variable as a proxy for x_test
    # Here, we assume x_test influences the selection, so we can use it as a target
    y = np.tile(x_test, (X_all.shape[0], 1))

    mi_scores = []
    for i in range(X_all.shape[1]):
        mi = mutual_info_regression(X_all[:, i].reshape(-1, 1), y[:, i], discrete_features=False)
        mi_scores.append(mi[0])

    mi_scores = np.array(mi_scores)

    if verbose:
        for idx, score in enumerate(mi_scores):
            print(f"Feature {idx}: Mutual Information with x_test feature {idx}: {score:.4f}")
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(mi_scores)), mi_scores)
        plt.title('Mutual Information Scores per Feature')
        plt.xlabel('Feature Index')
        plt.ylabel('Mutual Information')
        plt.close()

    return mi_scores


def reconstruct_x_test_feature_based(X_selected, X_unselected, x_test, mi_scores, top_k=5, lambda_reg=1e-5,
                                     verbose=True):
    """
    Reconstructs x_test using only top_k high mutual information features.

    Parameters:
    - X_selected: Selected data matrix
    - X_unselected: Unselected data matrix
    - x_test: True x_test vector
    - mi_scores: Mutual information scores per feature
    - top_k: Number of top features to use
    - lambda_reg: Regularization parameter for Ridge Regression
    - verbose: If True, prints details

    Returns:
    - x_test_hat: Reconstructed x_test vector
    - mse: Mean Squared Error
    - cosine_sim: Cosine Similarity
    """
    # Select top_k features
    top_features = np.argsort(mi_scores)[-top_k:]
    if verbose:
        print(f"Top {top_k} features based on Mutual Information: {top_features}")

    # Extract selected features
    X_selected_top = X_selected[:, top_features]
    X_unselected_top = X_unselected[:, top_features]

    # Combine data
    X_all_top = np.vstack((X_selected_top, X_unselected_top))
    W = np.ones(X_selected.shape[0]).tolist() + [1e-3] * X_unselected.shape[0]  # Assign higher weights to selected

    # Ridge Regression
    ridge = Ridge(alpha=lambda_reg)
    ridge.fit(X_all_top, x_test[top_features])
    x_hat_top = ridge.predict(X_all_top)

    # Aggregate the inferred features
    x_test_hat = np.zeros_like(x_test)
    x_test_hat[top_features] = np.mean(x_hat_top[:X_selected.shape[0]], axis=0)
    # Optionally, fill other features with mean or other strategies
    x_test_hat[~np.isin(range(len(x_test)), top_features)] = np.mean(
        X_selected[:, ~np.isin(range(len(x_test)), top_features)], axis=0)

    # Evaluation
    mse = mean_squared_error(x_test, x_test_hat)
    cosine_sim = cosine_similarity([x_test], [x_test_hat])[0, 0]

    if verbose:
        print(f"Feature-Based Reconstruction MSE: {mse:.6f}")
        print(f"Cosine Similarity: {cosine_sim:.6f}")
        plot_reconstruction_comparison(x_test, x_test_hat, title='Feature-Based Reconstruction')

    return {
        'x_test_hat': x_test_hat,
        'mse': mse,
        'cosine_similarity': cosine_sim
    }


def run_mutual_information_experiment(
        n_selected=100,
        n_unselected=200,
        n_features=10,
        epsilon=1e-3,
        lambda_reg=1e-5,
        top_k=5,
        verbose=True
):
    """
    Runs the Mutual Information Feature Analysis experiment.

    Parameters:
    - All parameters as defined above
    - verbose: If True, prints details

    Returns:
    - final_results: Dictionary containing the inference results
    """
    # Generate synthetic data
    x_test = np.random.randn(n_features)
    X_selected = x_test + np.random.randn(n_selected, n_features) * 0.1
    X_unselected = np.random.randn(n_unselected, n_features) + 2  # Shifted mean

    # Step 1: Mutual Information Analysis
    mi_scores = mutual_information_feature_analysis(X_selected, X_unselected, x_test, verbose=verbose)

    # Step 2: Feature-Based Reconstruction
    x_hat_feature = reconstruct_x_test_feature_based(
        X_selected, X_unselected, x_test, mi_scores, top_k=top_k, lambda_reg=lambda_reg, verbose=verbose
    )

    return {
        'mi_scores': mi_scores,
        'x_test_hat': x_hat_feature
    }


# Example Usage
if __name__ == "__main__":
    results = run_mutual_information_experiment()
    print("\n--- Mutual Information Inference Summary ---")
    print(f"Reconstructed x_test: {results['x_test_hat']}")
