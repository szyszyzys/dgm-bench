# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import minimize
#
#
# def compute_fim(X, indices, weights=None):
#     """
#     Compute the Fisher Information Matrix (FIM) for given indices.
#
#     Parameters:
#     - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
#     - indices (np.ndarray): Indices of samples to include in FIM computation.
#     - weights (np.ndarray or None): Weights for each selected sample. If None, equal weights are used.
#
#     Returns:
#     - fim (np.ndarray): Computed FIM of shape (n_features, n_features).
#     """
#     X_subset = X[indices]
#     if weights is None:
#         weights = np.ones(len(indices))
#     # Reshape weights for broadcasting
#     weights = weights[:, np.newaxis, np.newaxis]
#     # Compute outer products and weight them
#     outer_products = X_subset[:, :, np.newaxis] * X_subset[:, np.newaxis, :]
#     weighted_outer = weights * outer_products
#     # Sum over all samples to get the FIM
#     fim = weighted_outer.sum(axis=0)
#     return fim
#
#
# def optimize_query_embedding(I_selected, I_unselected, dim, lambda_reg=0.1, lr=0.01, iterations=100, verbose=False):
#     """
#     Optimize the query embedding to align with selected data and penalize unselected data.
#
#     Parameters:
#     - I_selected (np.ndarray): FIM for selected samples.
#     - I_unselected (np.ndarray): FIM for unselected samples.
#     - dim (int): Dimension of the embedding space.
#     - lambda_reg (float): Regularization parameter.
#     - lr (float): Learning rate for gradient ascent.
#     - iterations (int): Number of optimization iterations.
#     - verbose (bool): If True, prints progress.
#
#     Returns:
#     - q (np.ndarray): Optimized query embedding of shape (dim,).
#     """
#     # Initialize embedding randomly
#     q = np.random.randn(dim)
#     q /= np.linalg.norm(q)  # Normalize for stability
#
#     for i in range(iterations):
#         # Compute gradient: 2 * I_selected @ q - 2 * lambda_reg * I_unselected @ q
#         grad = 2 * I_selected.dot(q) - 2 * lambda_reg * I_unselected.dot(q)
#
#         # Update embedding using gradient ascent
#         q += lr * grad
#
#         # Normalize to prevent numerical issues
#         norm = np.linalg.norm(q)
#         if norm == 0:
#             norm = 1e-8
#         q /= norm
#
#         if verbose and (i + 1) % 10 == 0:
#             loss = q.dot(I_selected).dot(q) - lambda_reg * q.dot(I_unselected).dot(q)
#             print(f"Iteration {i + 1}/{iterations}, Loss: {loss:.4f}")
#
#     return q
#
#
# def score_samples(X, q):
#     """
#     Compute scores for all samples based on alignment with the embedding q.
#
#     The score is defined as the squared cosine similarity between q and each sample.
#
#     Parameters:
#     - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
#     - q (np.ndarray): Query embedding of shape (n_features,).
#
#     Returns:
#     - scores (np.ndarray): Scores for each sample of shape (n_samples,).
#     """
#     # Compute dot product between each sample and q
#     dot_products = X.dot(q)
#     # Since q is normalized, dot_products are the cosine similarities
#     # Square them to emphasize higher similarities
#     scores = dot_products ** 2
#     return scores
#
#
# def visualize_ranking(scores, selected_indices, unselected_indices, top_n=50):
#     """
#     Visualize the ranking of selected and unselected samples based on their scores.
#
#     Parameters:
#     - scores (np.ndarray): Scores for each sample.
#     - selected_indices (np.ndarray): Indices of selected samples.
#     - unselected_indices (np.ndarray): Indices of unselected samples.
#     - top_n (int): Number of top-ranked samples to visualize.
#     """
#     # Sort indices by descending scores
#     ranked_indices = np.argsort(-scores)
#     top_indices = ranked_indices[:top_n]
#
#     # Create labels: 1 for selected, 0 for unselected
#     labels = np.isin(top_indices, selected_indices).astype(int)
#
#     plt.figure(figsize=(12, 6))
#     plt.scatter(range(top_n), labels, c=labels, cmap='bwr', edgecolor='k')
#     plt.title(f"Top {top_n} Ranked Samples: Selected (1) vs Unselected (0)")
#     plt.xlabel("Rank Position")
#     plt.ylabel("Selected (1) / Unselected (0)")
#     plt.yticks([0, 1])
#     plt.grid(True, axis='y')
#     plt.close()
#
#
# def display_top_samples(X, ranked_indices, selected_indices, unselected_indices, top_k=10):
#     """
#     Display the top_k selected and unselected samples from the ranked list.
#
#     Parameters:
#     - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
#     - ranked_indices (np.ndarray): Indices sorted by rank.
#     - selected_indices (np.ndarray): Indices of selected samples.
#     - unselected_indices (np.ndarray): Indices of unselected samples.
#     - top_k (int): Number of top samples to display for each category.
#     """
#     top_selected = [idx for idx in ranked_indices if idx in selected_indices][:top_k]
#     top_unselected = [idx for idx in ranked_indices if idx in unselected_indices][:top_k]
#
#     print(f"Top {top_k} Selected Samples in Ranking:", top_selected)
#     print(f"Top {top_k} Unselected Samples in Ranking:", top_unselected)
#
#
# def fim_reverse_opt(X, selected_indices, unselected_indices, dim, lambda_reg=0.1, lr=0.01, iterations=100, top_n=50,
#                     top_k=10, verbose=False):
#     """
#     Perform the full embedding reconstruction and visualization_226 process.
#
#     Parameters:
#     - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
#     - selected_indices (np.ndarray): Indices of selected samples.
#     - unselected_indices (np.ndarray): Indices of unselected samples.
#     - dim (int): Dimension of the embedding space.
#     - lambda_reg (float): Regularization parameter.
#     - lr (float): Learning rate for optimization.
#     - iterations (int): Number of optimization iterations.
#     - top_n (int): Number of top-ranked samples to visualize.
#     - top_k (int): Number of top samples to display for each category.
#     - verbose (bool): If True, prints detailed logs.
#
#     Returns:
#     - q (np.ndarray): Reconstructed query embedding.
#     - scores (np.ndarray): Scores for all samples.
#     """
#     # Compute FIMs
#     I_selected = compute_fim(X, selected_indices)
#     I_unselected = compute_fim(X, unselected_indices)
#
#     if verbose:
#         print("FIMs computed.")
#
#     # Optimize query embedding
#     q = optimize_query_embedding(
#         I_selected,
#         I_unselected,
#         dim=dim,
#         lambda_reg=lambda_reg,
#         lr=lr,
#         iterations=iterations,
#         verbose=verbose
#     )
#
#     if verbose:
#         print("Optimization completed.")
#
#     # Score all samples
#     scores = score_samples(X, q)
#
#     if verbose:
#         print("Scoring completed.")
#
#     # Visualize rankings
#     visualize_ranking(scores, selected_indices, unselected_indices, top_n=top_n)
#
#     # Sort indices by descending scores
#     ranked_indices = np.argsort(-scores)
#
#     # Display top_k selected and unselected samples
#     display_top_samples(X, ranked_indices, selected_indices, unselected_indices, top_k=top_k)
#
#     return q, scores
#
#
# def fim_reverse_emb_opt_normal(x_s, selected_indices, unselected_indices, verbose=True):
#     """
#     Main function to execute the refined embedding reconstruction and visualization_226.
#     """
#     # Parameters
#     dim = x_s.shape[-1]
#     lambda_reg = 0.1
#     margin = 1.0
#     iterations = 100
#     top_n = 50
#     top_k = 10
#
#     if verbose:
#         print(f"Generated data with {n_samples} samples and {n_features} features.")
#         print(f"Selected samples: {len(selected_indices)}, Unselected samples: {len(unselected_indices)}")
#
#     # 2. Reconstruct Embedding and Visualize
#     q, scores = fim_reverse_opt(
#         x_s,
#         selected_indices,
#         unselected_indices,
#         dim=dim,
#         lambda_reg=lambda_reg,
#         lr=lr,
#         iterations=iterations,
#         top_n=top_n,
#         top_k=top_k,
#         verbose=verbose
#     )
#
#     # Additional Evaluation Metrics (Optional)
#     # For example, visualize the distribution of scores
#     plt.figure(figsize=(10, 6))
#     plt.hist(scores[selected_indices], bins=30, alpha=0.5, label='Selected', color='green')
#     plt.hist(scores[unselected_indices], bins=30, alpha=0.5, label='Unselected', color='red')
#     plt.title("Score Distribution: Selected vs Unselected Samples")
#     plt.xlabel("Score")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.close()
#
#     # Save the reconstructed embedding and scores if needed
#     # np.save("reconstructed_embedding.npy", q)
#     # np.save("scores.npy", scores)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.attacks import evaluate_reconstruction, save_results_pkl


def generate_synthetic_data(n_samples=200, n_features=20, random_state=42):
    """
    Generates synthetic data for demonstration.
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    return X


def create_test_samples(X, n_tests=3, n_components=5, random_state=42):
    """
    Creates multiple test samples as linear combinations of selected data points.
    """
    np.random.seed(random_state)
    test_samples = []
    true_selected_indices = []
    true_coefficients = []
    for _ in range(n_tests):
        selected_indices = np.random.choice(X.shape[0], n_components, replace=False)
        coefficients = np.random.randn(n_components)
        x_test = X[selected_indices].T @ coefficients
        test_samples.append(x_test)
        true_selected_indices.append(selected_indices)
        true_coefficients.append(coefficients)
    return np.stack(test_samples), true_selected_indices, true_coefficients


def selection_mechanism(X, x_test, k=10):
    """
    Simulates the selection mechanism based on alignment with x_test.
    Selects top-k data points that are most aligned with x_test.
    """
    scores = X @ x_test
    selected_indices = np.argsort(scores)[-k:]
    return selected_indices


def simulate_selection(X, x_tests, k=10):
    """
    Simulates selection for multiple test samples.
    Returns a list of selected indices for each test sample.
    """
    selected_indices_list = []
    for x_test in x_tests:
        selected_indices = selection_mechanism(X, x_test, k)
        selected_indices_list.append(selected_indices)
    return selected_indices_list


def construct_fim(X_selected, weights):
    """
    Constructs the Fisher Information Matrix (FIM) from selected samples and their weights.

    I(w) = sum_j w_j x_j x_j^T
    """
    if X_selected.shape[0] == 0:
        # If no selected samples, return a small identity matrix to prevent singularity
        return torch.eye(X_selected.shape[1], device=X_selected.device) * 1e-6
    fim = torch.zeros(X_selected.shape[1], X_selected.shape[1], device=X_selected.device)
    for j in range(X_selected.shape[0]):
        x_j = X_selected[j].unsqueeze(1)  # (n_features, 1)
        fim += weights[j] * (x_j @ x_j.T)  # (n_features, n_features)
    return fim


def reverse_engineer_query(
        X_selected,
        X_unselected,
        num_queries=1,
        n_iterations=1000,
        lr=1e-2,
        device='cpu',
        verbose=True
):
    """
    Reverse engineers query data points from selected and unselected data points.

    Parameters:
    - X_selected (numpy.ndarray): Selected data matrix (n_selected, n_features)
    - X_unselected (numpy.ndarray): Unselected data matrix (n_unselected, n_features)
    - num_queries (int): Number of query points to infer
    - n_iterations (int): Number of optimization steps
    - lr (float): Learning rate for optimizer
    - device (str): 'cpu' or 'cuda'
    - verbose (bool): If True, prints progress

    Returns:
    - x_queries_est (numpy.ndarray): Estimated query data points (num_queries, n_features)
    """
    # Convert data to torch tensors
    X_sel_tensor = torch.tensor(X_selected, dtype=torch.float32).to(device)  # (n_selected, n_features)
    X_unsel_tensor = torch.tensor(X_unselected, dtype=torch.float32).to(device)  # (n_unselected, n_features)

    n_features = X_selected.shape[1]

    # Initialize query points as parameters to optimize
    # Initialize near the mean of selected data points with some noise
    initial_x = X_sel_tensor.mean(dim=0, keepdim=True).repeat(num_queries, 1) + 0.1 * torch.randn(num_queries,
                                                                                                  n_features).to(device)
    x_queries = nn.Parameter(initial_x)

    optimizer = optim.Adam([x_queries], lr=lr)

    # Regularization strength
    reg_strength = 0.01

    # Define weights for selected and unselected
    weight_selected = 1.0
    weight_unselected = 0.1

    # Define the optimization loop
    for it in tqdm(range(n_iterations), desc="Optimizing Queries"):
        optimizer.zero_grad()

        loss = 0.0
        for q in range(num_queries):
            x_query = x_queries[q]  # (n_features,)

            # Compute FIM for selected and unselected
            fim_selected = (X_sel_tensor.t() @ X_sel_tensor) / X_sel_tensor.shape[0]  # (n_features, n_features)
            fim_unselected = (X_unsel_tensor.t() @ X_unsel_tensor) / X_unsel_tensor.shape[0]  # (n_features, n_features)

            # Alignment-based objectives
            alignment_selected = torch.matmul(x_query, torch.matmul(fim_selected, x_query))
            alignment_unselected = torch.matmul(x_query, torch.matmul(fim_unselected, x_query))

            # Define loss (to be minimized)
            # Equivalent to: loss = -alignment_selected + alignment_unselected + reg
            loss += (-weight_selected * alignment_selected) + (weight_unselected * alignment_unselected) + (
                        reg_strength * torch.norm(x_query, p=2))

        # Average loss over queries
        loss = loss / num_queries
        loss.backward()
        optimizer.step()

        if verbose and (it + 1) % 100 == 0:
            print(f"Iteration {it + 1}/{n_iterations}, Loss: {loss.item():.4f}")

    # Detach and convert to numpy
    x_queries_est = x_queries.detach().cpu().numpy()

    return x_queries_est


def evaluate_reconstruction(x_true_list, x_est_list):
    """
    Evaluates the reconstruction by comparing estimated test samples to true test samples.

    Parameters:
    - x_true_list (numpy.ndarray): True test samples of shape (n_tests, n_features).
    - x_est_list (numpy.ndarray): Estimated test samples of shape (n_tests, n_features).

    Returns:
    - best_cosine_similarities (list): Cosine similarities for each test sample.
    - best_euclidean_distances (list): Euclidean distances for each test sample.
    - matching_indices (list): Indices of the closest data points in X_sell for each test sample.
    """
    best_cosine_similarities = []
    best_euclidean_distances = []
    matching_indices = []

    for x_true, x_est in zip(x_true_list, x_est_list):
        cosine_sim = cosine_similarity([x_true], [x_est])[0, 0]
        euclidean_dist = np.linalg.norm(x_true - x_est)
        # Find the closest data point in X_sell
        # Note: Ensure X_sell is accessible here or pass it as an argument
        # For demonstration, we'll assume access to X_sell
        # Replace with actual X_sell as needed
        # Example:
        # distances = np.linalg.norm(X_sell - x_est, axis=1)
        # matching_index = np.argmin(distances)
        matching_index = -1  # Placeholder
        best_cosine_similarities.append(cosine_sim)
        best_euclidean_distances.append(euclidean_dist)
        matching_indices.append(matching_index)

    return best_cosine_similarities, best_euclidean_distances, matching_indices


def plot_alignment(x_true, x_est):
    """
    Plots the true vs. estimated query points.

    Parameters:
    - x_true (numpy.ndarray): True query point (n_features,)
    - x_est (numpy.ndarray): Estimated query point (n_features,)
    """
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(x_true))
    plt.plot(indices, x_true, label='True Query', marker='o')
    plt.plot(indices, x_est, label='Estimated Query', marker='x')
    plt.title('True vs. Estimated Query Points')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True)
    plt.close()


def reverse_engineer_query_workflow():
    # Step 1: Generate Synthetic Data
    X_sell = generate_synthetic_data(n_samples=200, n_features=20, random_state=42)
    y_sell = np.random.randn(200)  # Placeholder targets

    X_buy = generate_synthetic_data(n_samples=50, n_features=20, random_state=24)
    y_buy = np.random.randn(50)  # Placeholder targets

    # Step 2: Create Test Samples (x_query)
    x_query, true_selected_indices, true_coefficients = create_test_samples(
        X_sell, n_tests=3, n_components=5, random_state=42
    )

    # Step 3: Simulate Selection Mechanism
    selected_indices_list = simulate_selection(X_sell, x_query, k=10)

    # Step 4: Define Unselected Indices
    unselected_indices_list = []
    for selected_indices in selected_indices_list:
        all_indices = set(range(X_sell.shape[0]))
        unselected = np.array(list(all_indices - set(selected_indices)))
        unselected_indices_list.append(unselected)

    # Step 5: Reverse Engineer Query Points
    results = {}
    x_queries_est = []
    for i in range(x_query.shape[0]):
        if i >= len(selected_indices_list):
            print(f"Insufficient selected indices for test sample {i}")
            break
        X_sel = X_sell[selected_indices_list[i]]
        X_unsel = X_sell[unselected_indices_list[i]]

        x_est = reverse_engineer_query(
            X_selected=X_sel,
            X_unselected=X_unsel,
            num_queries=1,
            n_iterations=1000,
            lr=1e-2,
            device='cpu',
            verbose=True
        )
        x_queries_est.append(x_est[0])

    x_queries_est = np.stack(x_queries_est)  # (n_tests, n_features)

    # Step 6: Evaluate Reconstruction
    best_cosine_similarities, best_euclidean_distances, matching_indices = evaluate_reconstruction(x_query,
                                                                                                   x_queries_est)

    for i in range(x_query.shape[0]):
        print(f"\nTest Sample {i + 1}:")
        print(f"Cosine Similarity: {best_cosine_similarities[i]:.4f}")
        print(f"Euclidean Distance: {best_euclidean_distances[i]:.4f}")
        print(f"Matching Index: {matching_indices[i]}")
        # Optionally, plot alignment
        plot_alignment(x_query[i], x_queries_est[i])

    # Step 7: Save Results
    results = {
        "x_queries_est": x_queries_est,
        "best_cosine_similarities": best_cosine_similarities,
        "best_euclidean_distances": best_euclidean_distances,
        "matching_indices": matching_indices
    }
    save_results_pkl(results, save_dir="../data/reverse_engineering_results")

    return results


# if __name__ == "__main__":
#     reverse_engineer_query_workflow()
