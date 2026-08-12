import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from src.attacks import save_results_pkl

# Set random seed for reproducibility
np.random.seed(42)


def compute_extended_fim(X_selected, X_unselected, epsilon=1e-3, lambda_reg=1e-5):
    """
    Constructs the extended Fisher Information Matrix (FIM) using selected and unselected data.

    Parameters:
    - X_selected (np.ndarray): Selected data matrix of shape (n_selected, n_features).
    - X_unselected (np.ndarray): Unselected data matrix of shape (n_unselected, n_features).
    - epsilon (float): Weight assigned to each unselected data point.
    - lambda_reg (float): Regularization parameter to ensure numerical stability.

    Returns:
    - fim (np.ndarray): Extended FIM matrix of shape (n_features, n_features).
    """
    # Compute FIM for selected data with uniform weights
    W_selected = np.ones(X_selected.shape[0])  # Uniform weights for selected samples
    I_selected = X_selected.T @ (W_selected[:, np.newaxis] * X_selected)

    # Compute FIM contribution from unselected data with small weights
    W_unselected = epsilon * np.ones(X_unselected.shape[0])  # Small weights for unselected samples
    I_unselected = X_unselected.T @ (W_unselected[:, np.newaxis] * X_unselected)

    # Regularize the FIM to prevent singularity
    fim = I_selected + I_unselected + lambda_reg * np.eye(X_selected.shape[1])

    return fim


def eigen_decompose(fim_inv):
    """
    Performs eigenvalue decomposition on the inverse FIM.

    Parameters:
    - fim_inv (np.ndarray): Inverse of the FIM matrix of shape (n_features, n_features).

    Returns:
    - eigenvalues (np.ndarray): Sorted eigenvalues in descending order.
    - eigenvectors (np.ndarray): Corresponding eigenvectors sorted by eigenvalues.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(fim_inv)
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def infer_x_test_extended(X_selected, X_unselected, x_test, fim_inv, lambda_reg=1e-5):
    """
    Infers the test data as a linear combination of selected and unselected datapoints using regularized least squares.

    Parameters:
    - X_selected (np.ndarray): Selected data matrix of shape (n_selected, n_features).
    - X_unselected (np.ndarray): Unselected data matrix of shape (n_unselected, n_features).
    - x_test (np.ndarray): True test data vector of shape (n_features,).
    - fim_inv (np.ndarray): Inverse of the extended FIM matrix.
    - lambda_reg (float): Regularization parameter.

    Returns:
    - x_test_hat (np.ndarray): Reconstructed test data vector of shape (n_features,).
    - alpha_selected (np.ndarray): Coefficients for selected datapoints.
    - alpha_unselected (np.ndarray): Coefficients for unselected datapoints.
    """
    # Debugging: Print shapes
    print("---- infer_x_test_extended Debugging Shapes ----")
    print(f"Shape of X_selected: {X_selected.shape}")  # (n_selected, d)
    print(f"Shape of X_unselected: {X_unselected.shape}")  # (n_unselected, d)
    print(f"Shape of x_test: {x_test.shape}")  # (d,)

    # Number of selected and unselected samples
    n_selected = X_selected.shape[0]
    n_unselected = X_unselected.shape[0]
    d_selected = X_selected.shape[1]
    d_unselected = X_unselected.shape[1]
    d_test = x_test.shape[0]

    # Ensure all feature dimensions match
    assert d_selected == d_unselected == d_test, f"Feature dimension mismatch: X_selected({d_selected}), X_unselected({d_unselected}), x_test({d_test})"

    # Combine selected and unselected data
    X_combined = np.vstack((X_selected, X_unselected))  # Shape: (n_selected + n_unselected, d)
    print("Shape of X_combined:", X_combined.shape)  # (n_total, d)
    print("Shape of x_test:", x_test.shape)  # (d,)

    # Regularized least squares solution
    # Solve (X_combined X_combined^T + lambda I) alpha = X_combined x_test
    A = X_combined @ X_combined.T + lambda_reg * np.eye(X_combined.shape[0])  # (n_total, n_total)
    b = X_combined @ x_test  # (n_total,)

    try:
        alpha = np.linalg.solve(A, b)  # (n_total,)
    except np.linalg.LinAlgError:
        # If A is singular, use pseudo-inverse
        if fim_inv is not None:
            print("Using pseudo-inverse due to singular matrix.")
            alpha = fim_inv @ b
        else:
            alpha = np.linalg.pinv(A) @ b

    # Split coefficients into selected and unselected
    alpha_selected = alpha[:n_selected]
    alpha_unselected = alpha[n_selected:]

    # Reconstruct x_test using the coefficients
    x_test_hat = X_combined.T @ alpha  # (d,)

    return x_test_hat, alpha_selected, alpha_unselected


def evaluate_inference(x_test, x_test_hat):
    """
    Evaluates the inference using Mean Squared Error and Cosine Similarity.

    Parameters:
    - x_test (np.ndarray): True test data vector of shape (n_features,).
    - x_test_hat (np.ndarray): Reconstructed test data vector of shape (n_features,).

    Returns:
    - mse (float): Mean Squared Error between true and reconstructed test data.
    - cosine_sim (float): Cosine Similarity between true and reconstructed test data.
    """
    mse = mean_squared_error(x_test, x_test_hat)
    cosine_sim = cosine_similarity([x_test], [x_test_hat])[0, 0]
    return mse, cosine_sim


def evaluate_reconstruction_multiple(x_test_list, x_test_hat_list):
    """
    Evaluates the reconstruction for multiple samples by computing cosine similarity, Euclidean distance, and matching indices.

    Parameters:
    - x_test_list (list or np.ndarray): List of true test data vectors.
    - x_test_hat_list (list or np.ndarray): List of reconstructed test data vectors.

    Returns:
    - best_cosine_similarities (list): List of cosine similarities for each sample.
    - best_euclidean_distances (list): List of Euclidean distances for each sample.
    - matching_indices (list): List of matching indices (placeholder) for each sample.
    """
    best_cosine_similarities = []
    best_euclidean_distances = []
    matching_indices = []

    for i, (x_true, x_hat) in enumerate(zip(x_test_list, x_test_hat_list)):
        cosine_sim = cosine_similarity([x_true], [x_hat])[0, 0]
        euclidean_dist = np.linalg.norm(x_true - x_hat)
        best_cosine_similarities.append(cosine_sim)
        best_euclidean_distances.append(euclidean_dist)
        matching_indices.append(0)  # Placeholder for matching index

    return best_cosine_similarities, best_euclidean_distances, matching_indices


def infer_statistical_properties(X_selected, X_unselected, x_test):
    """
    Infers statistical properties of the test data based on selected and unselected data.

    Parameters:
    - X_selected (np.ndarray): Selected data matrix of shape (n_selected, n_features).
    - X_unselected (np.ndarray): Unselected data matrix of shape (n_unselected, n_features).
    - x_test (np.ndarray): True test data vector of shape (n_features,).

    Returns:
    - stats (dict): Dictionary containing mean, variance, and mean differences.
    """
    mean_selected = np.mean(X_selected, axis=0)
    variance_selected = np.var(X_selected, axis=0)

    mean_unselected = np.mean(X_unselected, axis=0)
    variance_unselected = np.var(X_unselected, axis=0)

    mean_diff_selected = x_test - mean_selected
    mean_diff_unselected = x_test - mean_unselected

    stats = {
        'mean_selected': mean_selected,
        'variance_selected': variance_selected,
        'mean_unselected': mean_unselected,
        'variance_unselected': variance_unselected,
        'mean_diff_selected': mean_diff_selected,
        'mean_diff_unselected': mean_diff_unselected
    }
    return stats


def additional_inference(X_selected, X_unselected, x_test, eigenvectors, top_k=2, n_clusters=3):
    """
    Performs additional inferences such as cluster membership and attribute importance.

    Parameters:
    - X_selected (np.ndarray): Selected data matrix of shape (n_selected, n_features).
    - X_unselected (np.ndarray): Unselected data matrix of shape (n_unselected, n_features).
    - x_test (np.ndarray): True test data vector of shape (n_features,).
    - eigenvectors (np.ndarray): Matrix of eigenvectors from eigen decomposition.
    - top_k (int): Number of top principal components to consider for attribute importance.
    - n_clusters (int): Number of clusters for KMeans.

    Returns:
    - cluster_label (int): Cluster label assigned to x_test.
    - attribute_importance (np.ndarray): Importance scores for each attribute based on top_k eigenvectors.
    """
    # Cluster Membership Inference using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_selected)
    cluster_label = kmeans.predict([x_test])[0]

    # Attribute Importance Inference using top_k eigenvectors
    projection = eigenvectors[:, :top_k].T @ x_test  # Project x_test onto top_k eigenvectors
    attribute_importance = np.abs(projection)
    # Normalize importance scores to sum to 1
    attribute_importance /= np.sum(attribute_importance)

    return cluster_label, attribute_importance


def plot_eigenvalues(eigenvalues, top_k=10):
    """
    Plots the top_k eigenvalues.

    Parameters:
    - eigenvalues (np.ndarray): Array of eigenvalues sorted in descending order.
    - top_k (int): Number of top eigenvalues to plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, top_k + 1), eigenvalues[:top_k], marker='o')
    plt.title('Top Eigenvalues of Inverse Extended FIM')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.close()


def plot_reconstruction(x_test, x_test_hat):
    """
    Plots the true test data vs. reconstructed test data.

    Parameters:
    - x_test (np.ndarray): True test data vector.
    - x_test_hat (np.ndarray): Reconstructed test data vector.
    """
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(x_test))
    plt.plot(indices, x_test, label='True Test Data', marker='o')
    plt.plot(indices, x_test_hat, label='Reconstructed Test Data', marker='x')
    plt.title('True vs. Reconstructed Test Data')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True)
    plt.close()


def visualize_ranking(scores, selected_indices, unselected_indices, top_n=50):
    """
    Visualize the ranking of selected and unselected samples based on their scores.

    Parameters:
    - scores (np.ndarray): Scores for each sample.
    - selected_indices (np.ndarray): Indices of selected samples.
    - unselected_indices (np.ndarray): Indices of unselected samples.
    - top_n (int): Number of top-ranked samples to visualize.
    """
    # Sort indices by descending scores
    ranked_indices = np.argsort(-scores)
    top_indices = ranked_indices[:top_n]

    # Create labels: 1 for selected, 0 for unselected
    labels = np.isin(top_indices, selected_indices).astype(int)

    plt.figure(figsize=(12, 6))
    plt.scatter(range(top_n), labels, c=labels, cmap='bwr', edgecolor='k')
    plt.title(f"Top {top_n} Ranked Samples: Selected (1) vs Unselected (0)")
    plt.xlabel("Rank Position")
    plt.ylabel("Selected (1) / Unselected (0)")
    plt.yticks([0, 1])
    plt.grid(True, axis='y')
    plt.close()


def display_top_samples(X, ranked_indices, selected_indices, unselected_indices, top_k=10):
    """
    Display the top_k selected and unselected samples from the ranked list.

    Parameters:
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - ranked_indices (np.ndarray): Indices sorted by rank.
    - selected_indices (np.ndarray): Indices of selected samples.
    - unselected_indices (np.ndarray): Indices of unselected samples.
    - top_k (int): Number of top samples to display for each category.
    """
    top_selected = [idx for idx in ranked_indices if idx in selected_indices][:top_k]
    top_unselected = [idx for idx in ranked_indices if idx in unselected_indices][:top_k]

    print(f"Top {top_k} Selected Samples in Ranking:", top_selected)
    print(f"Top {top_k} Unselected Samples in Ranking:", top_unselected)


def run_experiment(
        X,
        x_query,
        selected_indices,
        unselected_indices,
        epsilon=1e-3,
        lambda_reg=1e-5,
        top_k=2,
        n_clusters=3,
        verbose=True
):
    """
    Runs the extended Test Data Inference Attack experiment.

    Parameters:
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - x_query (np.ndarray): True test data vectors of shape (n_queries, n_features).
    - selected_indices (np.ndarray): Indices of selected samples.
    - unselected_indices (np.ndarray): Indices of unselected samples.
    - epsilon (float): Weight for unselected data points in FIM.
    - lambda_reg (float): Regularization parameter for FIM and regression.
    - top_k (int): Number of top principal components for attribute importance.
    - n_clusters (int): Number of clusters for KMeans.
    - verbose (bool): If True, prints detailed information.

    Returns:
    - results (dict): Dictionary containing all inference results.
    """
    results = {}

    # Debugging: Print shapes
    if verbose:
        print("---- run_experiment Data Shapes ----")
        print(f"Shape of X: {X.shape}")  # (n_samples, d)
        print(f"Shape of x_query: {x_query.shape}")  # (n_queries, d)

    # Ensure selected and unselected indices are valid
    assert np.all(selected_indices < X.shape[0]), "Selected indices out of bounds."
    assert np.all(unselected_indices < X.shape[0]), "Unselected indices out of bounds."

    # Step 1: Construct Extended FIM
    fim = compute_extended_fim(X[selected_indices], X[unselected_indices], epsilon=epsilon, lambda_reg=lambda_reg)
    results['fim'] = fim

    if verbose:
        print("Extended Fisher Information Matrix (FIM) computed.")

    # Step 2: Compute Inverse of FIM
    try:
        fim_inv = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        # Handle singular matrix by adding more regularization
        if verbose:
            print("FIM is singular; adding additional regularization and recomputing inverse.")
        fim += 1e-3 * np.eye(fim.shape[0])
        fim_inv = np.linalg.inv(fim)
    results['fim_inv'] = fim_inv

    # Step 3: Eigenvalue Decomposition
    eigenvalues, eigenvectors = eigen_decompose(fim_inv)
    results['eigenvalues'] = eigenvalues
    results['eigenvectors'] = eigenvectors
    if verbose:
        print(f"Top {top_k} Eigenvalues: {eigenvalues[:top_k]}")

    # Optional: Plot eigenvalues
    plot_eigenvalues(eigenvalues, top_k=top_k)

    # Step 4: Infer x_test for each query sample
    x_test_hat_list = []
    alpha_selected_list = []
    alpha_unselected_list = []

    for idx, x_test in enumerate(x_query):
        if verbose:
            print(f"\n---- Inferring Query Sample {idx + 1}/{len(x_query)} ----")
        x_test_hat, alpha_selected, alpha_unselected = infer_x_test_extended(
            X[selected_indices], X[unselected_indices], x_test, fim_inv, lambda_reg=lambda_reg
        )
        x_test_hat_list.append(x_test_hat)
        alpha_selected_list.append(alpha_selected)
        alpha_unselected_list.append(alpha_unselected)
        if verbose:
            print(f"Reconstructed x_test[{idx}]: {x_test_hat}")

    results['x_test_hat'] = np.array(x_test_hat_list)
    results['alpha_selected'] = np.array(alpha_selected_list)
    results['alpha_unselected'] = np.array(alpha_unselected_list)

    # Step 5: Evaluate Inference for all query samples
    best_cosine_similarities, best_euclidean_distances, matching_indices = evaluate_reconstruction_multiple(
        x_query, x_test_hat_list
    )
    results['best_cosine_similarities'] = best_cosine_similarities
    results['best_euclidean_distances'] = best_euclidean_distances
    results['matching_indices'] = matching_indices

    # Print evaluation results
    print("\n--- Reconstruction Evaluation for All Query Samples ---")
    for i in range(len(x_query)):
        print(f"\nTest Sample {i + 1}:")
        print(f" Cosine Similarity: {best_cosine_similarities[i]:.4f}")
        print(f" Euclidean Distance: {best_euclidean_distances[i]:.4f}")
        print(f" Matching index: {matching_indices[i]}")

    # Uncomment and adjust the following steps as needed
    # mse_list, cosine_sim_list = evaluate_inference_multiple(x_query, x_test_hat_list)
    # results['mse_list'] = mse_list
    # results['cosine_similarity_list'] = cosine_sim_list
    # if verbose:
    #     for i in range(len(x_query)):
    #         print(f"Reconstruction Mean Squared Error (MSE) for Sample {i + 1}: {mse_list[i]:.6f}")
    #         print(f"Reconstruction Cosine Similarity for Sample {i + 1}: {cosine_sim_list[i]:.6f}")
    #
    # # Optional: Plot reconstructions for each sample
    # for i in range(len(x_query)):
    #     plot_reconstruction(x_query[i], x_test_hat_list[i])
    #
    # # Step 6: Infer Statistical Properties for each sample
    # stats_list = []
    # for i in range(len(x_query)):
    #     stats = infer_statistical_properties(X[selected_indices], X[unselected_indices], x_query[i])
    #     stats_list.append(stats)
    # results['stats_list'] = stats_list
    # if verbose:
    #     for i, stats in enumerate(stats_list):
    #         print(f"\n--- Statistical Properties for Test Sample {i + 1} ---")
    #         for key, value in stats.items():
    #             print(f"{key}: {value}")
    #
    # # Step 7: Additional Inference for each sample
    # cluster_labels = []
    # attribute_importances = []
    # for i in range(len(x_query)):
    #     cluster_label, attribute_importance = additional_inference(
    #         X[selected_indices], X[unselected_indices], x_query[i], eigenvectors, top_k=top_k, n_clusters=n_clusters
    #     )
    #     cluster_labels.append(cluster_label)
    #     attribute_importances.append(attribute_importance)
    # results['cluster_labels'] = cluster_labels
    # results['attribute_importances'] = attribute_importances
    # if verbose:
    #     for i in range(len(x_query)):
    #         print(f"\nCluster Label Assigned to Test Sample {i + 1}: {cluster_labels[i]}")
    #         print(f"Attribute Importance Scores for Test Sample {i + 1}: {attribute_importances[i]}")
    #
    # # Step 8: Ranking Visualization
    # # Compute scores based on alignment with reconstructed x_test for all queries
    # # This might involve aggregating or averaging scores; adjust as needed
    # # For simplicity, we can compute scores for the first query sample
    # scores = cosine_similarity(X, [x_test_hat_list[0]]).flatten()
    # results['scores'] = scores
    # visualize_ranking(scores, selected_indices, unselected_indices, top_n=50)
    #
    # # Step 9: Display Top Samples
    # ranked_indices = np.argsort(-scores)
    # display_top_samples(X, ranked_indices, selected_indices, unselected_indices, top_k=10)

    return results


def fim_reverse_math(x_s, selected_indices, unselected_indices, x_query, device, save_dir="./data", verbose=True):
    """
    Main function to execute the extended Test Data Inference Attack.
    Generates synthetic data, selects samples, defines queries, and performs inference.

    Parameters:
    - x_s (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - selected_indices (list or np.ndarray): Indices of selected samples.
    - unselected_indices (list or np.ndarray): Indices of unselected samples.
    - x_query (np.ndarray): Query data vectors of shape (n_queries, n_features).
    - device: Device specification (unused in current context).
    - save_dir (str): Directory to save the results.
    - verbose (bool): If True, prints detailed information.

    Returns:
    - results (dict): Dictionary containing all inference results.
    """
    # Define tunable parameters
    params = {
        'n_selected': len(selected_indices),  # Number of selected data points
        'n_unselected': len(unselected_indices),  # Number of unselected data points
        'n_features': x_s.shape[1],  # Number of features (ensure this matches your data)
        'epsilon': 1e-3,  # Weight for unselected data points in FIM
        'lambda_reg': 1e-5,  # Regularization parameter for FIM and regression
        'top_k': 10,  # Number of top principal components for attribute importance
        'n_clusters': 3,  # Number of clusters for KMeans
        'verbose': verbose  # Enable verbose output
    }
    save_dir = f"{save_dir}/reverse_math"
    os.makedirs(save_dir, exist_ok=True)

    assert x_s.shape[1] == params[
        'n_features'], f"Expected x_s to have {params['n_features']} features, but got {x_s.shape[1]}."

    # Ensure selected_indices and unselected_indices are arrays
    selected_indices = np.array(selected_indices)
    unselected_indices = np.array(unselected_indices)

    # Store parameters and data for potential saving
    experiment_data = {
        'X': x_s,
        'x_query': x_query,
        'selected_indices': selected_indices,
        'unselected_indices': unselected_indices,
    }

    if params['verbose']:
        print(f"Selected samples indices: {selected_indices}")
        print(f"Unselected samples indices: {unselected_indices}")

    # Run the inference experiment
    results = run_experiment(
        X=x_s,
        x_query=x_query,
        selected_indices=selected_indices,
        unselected_indices=unselected_indices,
        epsilon=params['epsilon'],
        lambda_reg=params['lambda_reg'],
        top_k=params['top_k'],
        n_clusters=params['n_clusters'],
        verbose=params['verbose']
    )

    # Save the experiment data and results
    save_results_pkl({**experiment_data, **results}, save_dir=save_dir)

    # Uncomment and adjust the following summary as needed
    # print("\n--- Inference Summary ---")
    # for i in range(len(x_query)):
    #     print(f"Test Sample {i + 1}:")
    #     print(f" Mean Squared Error (MSE): {results['mse_list'][i]:.6f}")
    #     print(f" Cosine Similarity: {results['cosine_similarity_list'][i]:.6f}")
    #     print(f" Cluster Label: {results['cluster_labels'][i]}")
    #     print(f" Attribute Importance: {results['attribute_importances'][i]}")

    # Additional Insights
    # for i in range(len(x_query)):
    #     print(f"\n--- Reconstruction Coefficients for Selected Samples (Query {i + 1}) ---")
    #     print(results['alpha_selected'][i])
    #     print(f"--- Reconstruction Coefficients for Unselected Samples (Query {i + 1}) ---")
    #     print(results['alpha_unselected'][i])

    return results
