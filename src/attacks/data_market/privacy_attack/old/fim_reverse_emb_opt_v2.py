import torch
import torch.nn as nn
import torch.optim as optim

from src.attacks import evaluate_reconstruction


def construct_fim(X_selected, weights):
    """
    Constructs the Fisher Information Matrix (FIM) from selected samples and their weights.

    I(w) = sum_j w_j x_j x_j^T
    """
    fim = torch.zeros(X_selected.shape[1], X_selected.shape[1], device=X_selected.device)
    for j in range(X_selected.shape[0]):
        x_j = X_selected[j].unsqueeze(1)  # (n_features, 1)
        fim += weights[j] * (x_j @ x_j.T)  # (n_features, n_features)
    return fim


def optimize_test_samples_with_fim(X, selected_indices_list, unselected_indices_list, n_tests=3,
                                   n_iterations=5000, lr=1e-2, device='cpu'):
    """
    Infers test samples using an optimization-based method incorporating the Fisher Information Matrix (FIM).

    Parameters:
    - X: np.ndarray of shape (n_samples, n_features)
    - selected_indices_list: list of np.ndarray, each containing selected indices for a test sample
    - n_tests: number of test samples to infer
    - k: number of selected data points per test sample
    - n_iterations: number of optimization steps
    - lr: learning rate for optimizer
    - device: 'cpu' or 'cuda'

    Returns:
    - x_tests_opt: torch.Tensor of shape (n_tests, n_features), optimized test samples
    """
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # (n_samples, n_features)
    n_features = X.shape[1]

    mean_selected = X_tensor[selected_indices_list].mean(dim=0, keepdim=True)
    x_tests_opt = nn.Parameter(mean_selected.repeat(n_tests, 1) + torch.randn(n_tests, n_features, device=device) * 0.1)

    optimizer = optim.Adam([x_tests_opt], lr=lr)

    # Define weights for selected and unselected samples
    weight_selected = 1.0
    weight_unselected = 0.1  # Lower weight for unselected samples

    w_trace = 1.0
    w_alignment = 1.0
    w_reg = 0.01
    x_test = x_tests_opt[0]  # (n_features,)

    X_selected = X_tensor[selected_indices_list]  # (k, n_features)
    # X_unselected = X_tensor[unselected_indices_list]  # (n_samples - k, n_features)
    # Assign higher weights to selected samples
    weights_selected = torch.ones(X_selected.shape[0], device=device) * weight_selected
    # weights_unselected = torch.ones(X_unselected.shape[0], device=device) * weight_unselected

    # Construct FIM for selected and unselected
    fim_selected = construct_fim(X_selected.to(device), weights_selected.to(device))  # (n_features, n_features)
    # fim_unselected = construct_fim(X_unselected.to(device),    weights_unselected.to(device))  # (n_features, n_features)

    # Total FIM
    # fim_total = fim_selected + fim_unselected  # (n_features, n_features)

    for it in range(n_iterations):
        optimizer.zero_grad()
        loss = 0.0

        # Objective:
        # 1. Maximize the information from selected samples: maximize trace(fim_selected)
        # 2. Minimize the information from unselected samples: minimize trace(fim_unselected)
        # 3. Encourage the test sample to align with the FIM_selected and misalign with FIM_unselected

        # Trace-based objectives
        trace_selected = torch.trace(fim_selected)
        # trace_unselected = torch.trace(fim_unselected)

        # Alignment-based objectives
        alignment_selected = torch.matmul(x_test, torch.matmul(fim_selected, x_test))
        # alignment_unselected = torch.matmul(x_test, torch.matmul(fim_unselected, x_test))

        # Update loss computation
        # loss += (-w_trace * trace_selected + w_trace * trace_unselected)
        # loss += (-w_alignment * alignment_selected + w_alignment * alignment_unselected)
        loss += (-w_trace * trace_selected)
        loss += (-w_alignment * alignment_selected)

        loss += w_reg * torch.norm(x_test, p=2)

        # Optionally, add regularization to keep x_test within reasonable bounds
        reg_strength = 0.01
        loss += reg_strength * torch.norm(x_test, p=2)

        # Normalize loss by number of test samples
        loss = loss / n_tests
        loss.backward()
        optimizer.step()

        if (it + 1) % 500 == 0 or it == 0:
            print(f"Iteration {it + 1}/{n_iterations}, Loss: {loss.item():.4f}")

    return x_tests_opt.detach().cpu().numpy()


def fim_reverse_emb_opt_v2(x_s, selected_indices, unselected_indices, x_query, device, save_dir="./data", verbose=True):
    # Configuration
    n_tests = x_query.shape[0]
    n_iterations = 2000
    lr = 1e-2

    # Step 4: Reverse engineer test samples using optimization with FIM
    x_tests_est = optimize_test_samples_with_fim(
        x_s, selected_indices, unselected_indices, n_tests=n_tests,
        n_iterations=n_iterations, lr=lr, device=device
    )

    # Step 5: Evaluation
    best_cosine_similarities, best_euclidean_distances, matching_indices = evaluate_reconstruction(x_query, x_tests_est)
    for i in range(n_tests):
        print(f"\nTest Sample {i + 1}:")
        print(f" Cosine Similarity: {best_cosine_similarities[i]:.4f}")
        print(f" Euclidean Distance: {best_euclidean_distances[i]:.4f}")
        print(f" Matching index: {matching_indices[i]:.4f}")

    results = {
        "best_cosine_similarities": best_cosine_similarities,
        "best_euclidean_distances": best_euclidean_distances,
        "matching_indices": matching_indices
    }

    return results
