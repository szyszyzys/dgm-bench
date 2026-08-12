import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score


def compute_fim_refined(X, indices, weights=None, regularization=1e-5):
    """
    Compute a refined Fisher Information Matrix (FIM) for given indices.

    Parameters:
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - indices (np.ndarray): Indices of samples to include in FIM computation.
    - weights (np.ndarray or None): Weights for each selected sample. If None, equal weights are used.
    - regularization (float): Small value to add to the diagonal for numerical stability.

    Returns:
    - fim (np.ndarray): Computed FIM of shape (n_features, n_features).
    """
    X_subset = X[indices]
    if weights is None:
        weights = np.ones(len(indices))
    # Normalize each feature to have zero mean and unit variance
    X_normalized = (X_subset - X_subset.mean(axis=0)) / X_subset.std(axis=0)
    # Apply weights
    weighted_X = X_normalized * weights[:, np.newaxis]
    # Compute covariance matrix as FIM
    fim = np.cov(weighted_X, rowvar=False, bias=True)
    # Ensure FIM is positive definite
    fim += regularization * np.eye(fim.shape[0])
    return fim


def contrastive_loss(e, E_selected, E_unselected, margin=1.0):
    """
    Contrastive loss function for embedding reconstruction.

    Parameters:
    - e (np.ndarray): Current embedding vector of shape (n_features,).
    - E_selected (np.ndarray): Embeddings of selected samples, shape (n_selected, n_features).
    - E_unselected (np.ndarray): Embeddings of unselected samples, shape (n_unselected, n_features).
    - margin (float): Margin for contrastive loss.

    Returns:
    - loss (float): Computed contrastive loss.
    """
    # Compute distances to selected and unselected samples
    dist_selected = np.linalg.norm(E_selected - e, axis=1)
    dist_unselected = np.linalg.norm(E_unselected - e, axis=1)

    # Loss for selected samples: minimize distances
    loss_selected = np.mean(dist_selected ** 2)

    # Loss for unselected samples: maximize distances beyond the margin
    loss_unselected = np.mean(np.maximum(0, margin - dist_unselected) ** 2)

    # Total loss
    total_loss = loss_selected + loss_unselected
    return total_loss


def optimize_query_embedding_contrastive(E_selected, E_unselected, dim, margin=1.0, iterations=100):
    """
    Optimize the query embedding using a contrastive loss function.

    Parameters:
    - E_selected (np.ndarray): Embeddings of selected samples.
    - E_unselected (np.ndarray): Embeddings of unselected samples.
    - dim (int): Dimension of the embedding space.
    - margin (float): Margin for contrastive loss.
    - iterations (int): Maximum number of iterations for the optimizer.

    Returns:
    - e_test_hat (np.ndarray): Optimized embedding vector.
    """
    # Initial guess: mean of selected embeddings
    e0 = np.mean(E_selected, axis=0)

    # Define the objective function for the optimizer
    def objective(e):
        return contrastive_loss(e, E_selected, E_unselected, margin)

    # Use L-BFGS-B optimizer
    result = minimize(
        objective,
        e0,
        method='L-BFGS-B',
        options={'maxiter': iterations, 'disp': True}
    )

    e_test_hat = result.x
    return e_test_hat


def score_samples_cosine(X, q):
    """
    Compute cosine similarity scores for all samples based on the embedding q.

    Parameters:
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - q (np.ndarray): Query embedding vector of shape (n_features,).

    Returns:
    - scores (np.ndarray): Cosine similarity scores of shape (n_samples,).
    """
    # Normalize the query embedding
    q_norm = q / np.linalg.norm(q)
    # Normalize all samples
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    # Compute cosine similarity
    scores = np.dot(X_norm, q_norm)
    return scores


def evaluate_ranking(scores, selected_indices, unselected_indices, top_k=10):
    """
    Evaluate the ranking using Precision@k, Recall@k, and ROC AUC.

    Parameters:
    - scores (np.ndarray): Scores for all samples.
    - selected_indices (np.ndarray): Indices of selected samples.
    - unselected_indices (np.ndarray): Indices of unselected samples.
    - top_k (int): Number of top-ranked samples to consider for Precision and Recall.

    Returns:
    - metrics (dict): Dictionary containing Precision@k, Recall@k, and ROC AUC.
    """
    n_samples = len(scores)
    # Binary labels: 1 for selected, 0 for unselected
    labels = np.zeros(n_samples)
    labels[selected_indices] = 1

    # Binary predictions for top_k
    top_k_indices = np.argsort(-scores)[:top_k]
    preds_top_k = np.zeros(n_samples)
    preds_top_k[top_k_indices] = 1

    # Compute Precision@k and Recall@k
    true_positives = np.sum(preds_top_k[selected_indices])
    precision_at_k = true_positives / top_k
    recall_at_k = true_positives / len(selected_indices)

    # Compute ROC AUC
    roc_auc = roc_auc_score(labels, scores)

    metrics = {
        'Precision@k': precision_at_k,
        'Recall@k': recall_at_k,
        'ROC_AUC': roc_auc
    }
    return metrics


def visualize_ranking(scores, selected_indices, unselected_indices, top_n=50):
    """
    Visualize the ranking of selected and unselected samples based on their scores.

    Parameters:
    - scores (np.ndarray): Scores for all samples.
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


def plot_precision_recall_curve(metrics):
    """
    Plot Precision@k and Recall@k.

    Parameters:
    - metrics (dict): Dictionary containing Precision@k and Recall@k.
    """
    labels = ['Precision@k', 'Recall@k']
    values = [metrics['Precision@k'], metrics['Recall@k']]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=['green', 'blue'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Precision@k and Recall@k')
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f"{value:.2f}", ha='center', color='white',
                 fontsize=12)
    plt.close()


def plot_roc_curve(labels, scores):
    """
    Plot ROC curve.

    Parameters:
    - labels (np.ndarray): Binary labels.
    - scores (np.ndarray): Prediction scores.
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.close()


def visualize_embedding_space(X, q, selected_indices, unselected_indices, method='pca'):
    """
    Visualize the embedding space using PCA or t-SNE.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - q (np.ndarray): Query embedding.
    - selected_indices (np.ndarray): Indices of selected samples.
    - unselected_indices (np.ndarray): Indices of unselected samples.
    - method (str): 'pca' or 'tsne'.
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("method should be 'pca' or 'tsne'")

    embeddings_2d = reducer.fit_transform(X)
    q_2d = reducer.transform(q.reshape(1, -1))[0]

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[unselected_indices, 0], embeddings_2d[unselected_indices, 1],
                c='blue', label='Unselected', alpha=0.5)
    plt.scatter(embeddings_2d[selected_indices, 0], embeddings_2d[selected_indices, 1],
                c='red', label='Selected', alpha=0.5)
    plt.scatter(q_2d[0], q_2d[1], c='green', marker='*', s=200, label='Query Embedding')
    plt.title(f'Embedding Space Visualization using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.close()


def fim_reverse_opt_refined(X, selected_indices, unselected_indices, dim, lambda_reg=0.1, margin=1.0, iterations=100,
                            top_n=50, top_k=10, verbose=True):
    """
    Perform the full embedding reconstruction and visualization_226 process with methodological improvements.

    Parameters:
    - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    - selected_indices (np.ndarray): Indices of selected samples.
    - unselected_indices (np.ndarray): Indices of unselected samples.
    - dim (int): Dimension of the embedding space.
    - lambda_reg (float): Regularization parameter.
    - margin (float): Margin for contrastive loss.
    - iterations (int): Number of optimization iterations.
    - top_n (int): Number of top-ranked samples to visualize.
    - top_k (int): Number of top samples to display for each category.
    - verbose (bool): If True, prints detailed logs.

    Returns:
    - q (np.ndarray): Reconstructed query embedding.
    - scores (np.ndarray): Scores for all samples.
    - metrics (dict): Evaluation metrics.
    """
    # Compute FIMs using refined computation
    # I_selected = compute_fim_refined(X, selected_indices, weights=None, regularization=1e-5)
    # I_unselected = compute_fim_refined(X, unselected_indices, weights=None, regularization=1e-5)

    if verbose:
        print("Refined FIMs computed.")

    # Optimize query embedding using contrastive loss
    q = optimize_query_embedding_contrastive(
        E_selected=X[selected_indices],
        E_unselected=X[unselected_indices],
        dim=dim,
        margin=margin,
        iterations=iterations
    )

    if verbose:
        print("Optimization completed.")

    # Score all samples using cosine similarity
    scores = score_samples_cosine(X, q)

    if verbose:
        print("Scoring completed.")

    # Visualize rankings
    visualize_ranking(scores, selected_indices, unselected_indices, top_n=top_n)

    # Sort indices by descending scores
    ranked_indices = np.argsort(-scores)

    # Display top_k selected and unselected samples
    display_top_samples(X, ranked_indices, selected_indices, unselected_indices, top_k=top_k)

    # Evaluate ranking
    metrics = evaluate_ranking(scores, selected_indices, unselected_indices, top_k=top_k)
    if verbose:
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Plot Precision@k and Recall@k
    plot_precision_recall_curve(metrics)

    # Plot ROC Curve
    n_samples = X.shape[0]
    labels = np.zeros(n_samples)
    labels[selected_indices] = 1
    plot_roc_curve(labels, scores)

    # Visualize embedding space
    visualize_embedding_space(X, q, selected_indices, unselected_indices, method='pca')

    return q, scores, metrics


def fim_reverse_emb_opt(x_s, selected_indices, unselected_indices, x_query, device, save_dir="./data", verbose=True):
    """
    Main function to execute the refined embedding reconstruction and visualization_226.
    """
    # Parameters
    save_dir = f"{save_dir}/reverse_opt_v1/"

    dim = x_s.shape[-1]
    lambda_reg = 0.1
    margin = 1.0
    iterations = 100
    top_n = 50
    top_k = 10

    # 2. Reconstruct Embedding and Visualize
    q, scores, metrics = fim_reverse_opt_refined(
        x_s,
        selected_indices,
        unselected_indices,
        dim=dim,
        lambda_reg=lambda_reg,
        margin=margin,
        iterations=iterations,
        top_n=top_n,
        top_k=top_k,
        verbose=verbose
    )


    results = {

    }

    return results
