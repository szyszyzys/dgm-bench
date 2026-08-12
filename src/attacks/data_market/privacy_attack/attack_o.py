import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.optim.lr_scheduler import StepLR


def pairwise_dist_matrix(X, Y, metric='euclidean'):
    """
    Compute pairwise distances between each row in X and each row in Y.
      X: (m, d)
      Y: (n, d)
    Return: (m, n) distance matrix.
    """
    if metric == 'euclidean':
        # can do direct using broadcasting or np.linalg
        # shape: (m, n)
        return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
    else:
        raise NotImplementedError("Only 'euclidean' metric is implemented here.")


def match_query_sets(x_query_true, x_query_recon, metric='euclidean'):
    """
    x_query_true: (k_true, d) or (d,) if single vector
    x_query_recon: (k_recon, d) or (d,) if single vector
    Uses Hungarian algorithm to find minimal sum of distances.
    Returns:
      - total_dist (float): sum of matched distances
      - avg_dist (float): average distance per matched pair
      - matching: list of (i, j) pairs matching x_query_true[i] to x_query_recon[j]
        (Indices follow the order of x_query_true and x_query_recon.)
    """
    # If either is a 1D vector, reshape to 2D
    if x_query_true.ndim == 1:
        x_query_true = x_query_true[None, :]  # shape (1, d)
    if x_query_recon.ndim == 1:
        x_query_recon = x_query_recon[None, :]  # shape (1, d)

    k_true = x_query_true.shape[0]
    k_recon = x_query_recon.shape[0]

    X = x_query_true
    Y = x_query_recon

    # compute cost matrix: shape (k_true, k_recon)
    cost_matrix = pairwise_dist_matrix(X, Y, metric=metric)

    if k_true == k_recon:
        # square matrix => direct assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_dist = cost_matrix[row_ind, col_ind].sum()
        matching = list(zip(row_ind, col_ind))
    else:
        # rectangular cost matrix => Hungarian handles that, but
        # we only match min(k_true, k_recon) pairs. The Hungarian output
        # will produce a complete assignment, but some might be unmatched if k differs.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Note: row_ind, col_ind each has size = max(k_true, k_recon).
        # We'll filter out only valid pairs that are within the shape.
        matched_pairs = []
        total_dist = 0.0
        for r, c in zip(row_ind, col_ind):
            if r < k_true and c < k_recon:
                matched_pairs.append((r, c))
                total_dist += cost_matrix[r, c]
        matching = matched_pairs

    # average distance over matched pairs
    avg_dist = total_dist / len(matching) if matching else 0.0

    return total_dist, avg_dist, matching


def baseline_random_guess(num_query_points, d, num_guesses=10, scale=1.0):
    """
    Return a random guess for x_query.
    We can do multiple random guesses and pick one if needed.
    """
    # shape: (num_guesses, num_query_points, d)
    # Then pick one at random or average them, etc.
    guesses = scale * torch.randn(num_guesses, num_query_points, d)
    # For simplicity, just return the first or the best in some sense
    return guesses[0]  # shape (num_query_points, d)


def baseline_centroid_of_selected(full_data, selected_indices, num_query_points=1):
    """
    Return either a single vector or multiple copies of the centroid of selected points
    as a naive guess for x_query.
    """
    selected_data = full_data[selected_indices]
    centroid = torch.tensor(np.mean(selected_data, axis=0, keepdims=True))  # works for numpy arrays
    if num_query_points == 1:
        return centroid  # shape (1, d)
    else:
        # If multiple query vectors, replicate or add small random noise
        return centroid.repeat(num_query_points, 1)


def compute_fisher_information(selected_data, lambda_reg=1e-3):
    """
    Dummy placeholder for computing Fisher inverse: I_inv.
    Replace or adapt with the real function as needed.
    """
    print("computing FIM")
    # Suppose: I = selected_data^T * selected_data + lambda_reg * I_d
    # Return I_inv = inv(...)
    N, d = selected_data.shape
    I = selected_data.T @ selected_data + lambda_reg * torch.eye(d, device=selected_data.device)
    I_inv = torch.inverse(I)
    return I_inv


def score_function(x_query, I_inv, candidate_data, aggregation_method='sum'):
    """
    Predict scores for candidate_data given x_query (which can be multiple vectors).
    We assume a form like: score = x_query^T * I_inv * x_candidate (plus or minus).
    If multiple x_query vectors, we aggregate (sum/mean) their individual scores.
    """
    # x_query shape: (k, d)  or (1, d) if k=1
    # candidate_data shape: (N, d)
    # I_inv shape: (d, d)

    # If there's more than one query vector, we'll compute each vector's score:
    # For each x_query[i], compute x_query[i] * I_inv * candidate_data^T
    # Then aggregate across i in {1...k}.
    if x_query.dim() == 1:
        # In case x_query is shape (d,), unsqueeze to (1, d)
        x_query = x_query.unsqueeze(0)

    k, d = x_query.shape
    N, d2 = candidate_data.shape
    assert d == d2, f"Dimension mismatch in x_query ({d}) and candidate_data ({d2})."

    # scores_per_query: (k, N)
    scores_per_query = x_query @ I_inv @ candidate_data.T

    if aggregation_method == 'sum':
        # Sum across the k query vectors -> shape (N,)
        pred_scores = scores_per_query.sum(dim=0)
    elif aggregation_method == 'mean':
        # Average across the k query vectors -> shape (N,)
        pred_scores = scores_per_query.mean(dim=0)
    else:
        raise ValueError(f"Unknown aggregation_method={aggregation_method}. Use 'sum' or 'mean'.")

    return pred_scores


def hinge_margin_ranking_loss(pred_scores, sel_mask, margin=0.1):
    """
    Pairwise hinge loss for selection:
    For each (i in selected, j in not selected), we want pred_scores[i] + margin > pred_scores[j].
    """
    selected_indices = torch.where(sel_mask > 0)[0]
    not_selected_indices = torch.where(sel_mask <= 0)[0]
    # We'll do a double sum. For large data, can get expensive, but illustrative for now.
    loss = 0.0
    count = 0
    for i in selected_indices:
        for j in not_selected_indices:
            diff = pred_scores[i] - pred_scores[j]
            # hinge: max(0, margin - diff) => want diff >= margin
            loss_ij = F.relu(margin - diff)
            loss += loss_ij
            count += 1
    if count > 0:
        loss = loss / count
    return loss


def logistic_ranking_loss(pred_scores, sel_mask, temperature=1.0):
    """
    Pairwise logistic ranking loss:
    sum_{i in sel, j in not_sel} log(1 + exp( -(score[i] - score[j]) / temperature ))
    Implemented in a numerically stable way using softplus.
    """
    selected_indices = torch.where(sel_mask > 0)[0]
    not_selected_indices = torch.where(sel_mask <= 0)[0]
    loss = 0.0
    count = 0
    for i in selected_indices:
        for j in not_selected_indices:
            diff = (pred_scores[i] - pred_scores[j]) / temperature
            # Use softplus for numerical stability: softplus(-diff) = log(1 + exp(-diff))
            loss_ij = F.softplus(-diff)
            loss += loss_ij
            count += 1
    if count > 0:
        loss = loss / count
    return loss


def top_k_margin_loss(pred_scores, selected_indices, k, margin=0.1):
    """
    A simpler top-K extension for a hinge-like approach:
    - We want the selected_indices to have the top-K scores.
    - If selected_indices is exactly K points, we push them above others by margin.
    - If you have more/less, adapt as needed.

    NOTE: This is an *illustrative* approach. In practice, you might do:
    - top-K threshold,
    - partial ranking among the boundary,
    - or a differentiable top-K approximation.
    """
    # We'll assume we *know* these are the top k indices. Then we want:
    # For i in selected_indices, for j in the rest, pred_scores[i] + margin > pred_scores[j].
    # We'll treat the first K in `selected_indices` as "top-K" for simplicity.
    # If selected_indices has fewer or more than k, adapt logic as needed.
    if isinstance(selected_indices, torch.Tensor) and len(selected_indices) > k:
        selected_indices = selected_indices[:k]

    all_indices = torch.arange(len(pred_scores), device=pred_scores.device)
    unselected_indices = torch.tensor([idx for idx in all_indices if idx not in selected_indices],
                                      device=pred_scores.device)

    loss = 0.0
    count = 0
    for i in selected_indices:
        for j in unselected_indices:
            diff = pred_scores[i] - pred_scores[j]
            loss_ij = F.relu(margin - diff)
            loss += loss_ij
            count += 1
    if count > 0:
        loss = loss / count
    return loss


def real_data_prior_loss(x_query, full_data):
    """
    For each query vector x, penalize distance to the closest point in full_data.
    We sum or average across the k query vectors if multiple.
    """
    # x_query: (k, d)
    # full_data: (N, d)
    # We'll compute cdist: (k, N) and take min over N for each row in k
    # Then sum or average across k
    dist_matrix = torch.cdist(x_query, full_data)  # shape (k, N)
    min_dist_per_query = dist_matrix.min(dim=1).values  # shape (k,)
    loss = min_dist_per_query.mean()
    return loss


def reconstruct_query(
        full_seller_data: torch.Tensor,
        selected_indices: torch.Tensor,
        scenario: str = 'score_known',
        observed_scores: torch.Tensor = None,
        lambda_reg: float = 1e-3,
        lr: float = 0.1,
        num_iters: int = 1000,
        reg_weight: float = 1e-3,
        margin: float = 0.1,
        num_restarts: int = 1,
        verbose: bool = False,
        # --- New optional parameters ---
        num_query_points: int = 1,  # If >1, reconstruct multiple vectors
        aggregation_method: str = 'sum',  # 'sum' or 'mean' for multiple query vectors
        ranking_loss_type: str = 'hinge',  # 'hinge' or 'logistic' for selection-only scenario
        top_k_selection: int = None,  # if you want top-K style selection
        real_data_prior_weight: float = 0.0,  # encourage x_query near some real data,
        initial_guess=None,
):
    """
    High-level function to reconstruct a query (possibly multiple vectors)
    given a scenario: 'score_known' or 'selection_only', with optional expansions.

    Args:
        full_seller_data           : (N, d) All data from the seller (both selected & unselected).
        selected_indices           : Indices (within full_seller_data) that the buyer used to build Fisher info.
        scenario                   : Either 'score_known' or 'selection_only'.
        observed_scores            : Required if scenario='score_known'. Shape (N,) with a score per candidate data.
        lambda_reg                 : Regularization for Fisher information matrix.
        lr                         : Learning rate for Adam.
        num_iters                  : Number of optimization iterations.
        reg_weight                 : Weight for L2 regularization on x_query (basic).
        margin                     : Margin used in ranking loss for 'selection_only' scenario.
        num_restarts               : Number of random restarts. We pick the best final solution.
        verbose                    : If True, print logs.
        num_query_points           : How many query vectors to reconstruct (default=1).
        aggregation_method         : 'sum' or 'mean' for combining multiple query vectors.
        ranking_loss_type          : 'hinge' or 'logistic' for pairwise selection-only scenario.
        top_k_selection            : If not None, tries a top-K approach in selection scenario.
        real_data_prior_weight     : Weight for a prior that forces x_query near real data points.

    Returns:
        best_x_query : (k, d) The reconstructed query vectors.
        best_history : List of final run's training loss values (debug/analysis).
    """
    device = full_seller_data.device
    d = full_seller_data.shape[1]

    # 1. Compute Fisher inverse from the selected data
    selected_data = full_seller_data[selected_indices]
    I_inv = compute_fisher_information(selected_data, lambda_reg=lambda_reg)
    # 2. We'll consider the entire full_seller_data as "candidates" for scoring or selection
    candidate_data = full_seller_data
    # Convert selected_indices to a mask if needed
    # (for pairwise ranking). We'll do it once here.
    if selected_indices.dtype == torch.bool:
        sel_mask = selected_indices.float()
    else:
        sel_mask = torch.zeros(len(candidate_data), dtype=torch.float, device=device)
        sel_mask[selected_indices] = 1.0

    def single_run(
            x_query=None,
            patience=50,
            min_delta=1e-5,

    ):
        """
        Single reconstruction run with early stopping and learning rate scheduling.

        Args:
            x_query: Initial guess for the query tensor (requires_grad=True).
            lr: Initial learning rate for Adam.
            num_iters: Maximum number of iterations.
            patience: Early stopping patience (stop if no improvement after 'patience' iterations).
            min_delta: Minimum improvement required to reset the patience counter.
            reg_weight, real_data_prior_weight, margin, top_k_selection, scenario, ranking_loss_type:
                       same meaning as in your original function.
            verbose: If True, print progress logs.

        Returns:
            best_x_query: The best solution found (torch.Tensor).
            loss_history: A list of losses at each iteration.
        """
        # Example: define your optimizer
        optimizer = optim.Adam([x_query], lr=lr)

        # Example: define a Step LR scheduler (reduce LR every step_size iterations)
        scheduler = StepLR(optimizer, step_size=max(num_iters // 3, 1), gamma=0.1)

        loss_history = []

        # Early stopping trackers
        best_loss = float('inf')
        best_x_query = x_query.detach().clone()
        epochs_no_improve = 0

        for it in range(num_iters):
            optimizer.zero_grad()

            # 1) Compute predicted scores
            pred_scores = score_function(
                x_query, I_inv, candidate_data, aggregation_method=aggregation_method
            )

            # 2a) L2 prior on x_query
            reg_loss = reg_weight * (x_query ** 2).mean()

            # 2b) Real-data prior
            rd_loss = 0.0
            if real_data_prior_weight > 0.0:
                rd_loss = real_data_prior_weight * real_data_prior_loss(x_query, full_seller_data)

            # 2c) or 2d) main loss based on scenario
            if scenario == 'score_known':
                if observed_scores is None:
                    raise ValueError("observed_scores must be provided for 'score_known' scenario.")
                loss_main = F.mse_loss(pred_scores, observed_scores.to(device))
            elif scenario == 'selection_only':
                if top_k_selection is not None and top_k_selection > 0:
                    loss_main = top_k_margin_loss(pred_scores, selected_indices,
                                                  k=top_k_selection, margin=margin)
                else:
                    if ranking_loss_type == 'hinge':
                        loss_main = hinge_margin_ranking_loss(pred_scores, sel_mask, margin=margin)
                    elif ranking_loss_type == 'logistic':
                        loss_main = logistic_ranking_loss(pred_scores, sel_mask, temperature=1.0)
                    else:
                        raise ValueError("Unknown ranking_loss_type. Use 'hinge' or 'logistic'.")
            else:
                raise ValueError("Unknown scenario. Use 'score_known' or 'selection_only'.")

            total_loss = loss_main + reg_loss + rd_loss
            total_loss.backward()
            optimizer.step()

            # Step the scheduler (with StepLR, we step every iteration)
            scheduler.step()

            current_loss = total_loss.item()
            loss_history.append(current_loss)

            # Early stopping check
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                best_x_query = x_query.detach().clone()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Print progress occasionally
            if verbose and (it + 1) % max(1, num_iters // 10) == 0:
                print(f"Iter {it + 1}/{num_iters}, Loss={current_loss:.6f} "
                      f"(Main={loss_main.item():.4f}, Reg={reg_loss.item():.4f}, RD={rd_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f})")

            # If no improvement over 'patience' iterations, stop
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at iteration {it + 1} due to no improvement.")
                break

        # Return the best found solution and the loss history
        return best_x_query, loss_history

    # def single_run(x_query=None):
    #     # x_query shape: (k, d)
    #     optimizer = optim.Adam([x_query], lr=lr)
    #     loss_history = []
    #
    #     for it in range(num_iters):
    #         optimizer.zero_grad()
    #
    #         # 1) Compute predicted scores
    #         pred_scores = score_function(x_query, I_inv, candidate_data, aggregation_method=aggregation_method)
    #
    #         # 2) Loss components
    #
    #         # 2a) L2 prior on x_query
    #         reg_loss = reg_weight * (x_query ** 2).mean()
    #
    #         # 2b) Real-data prior
    #         rd_loss = 0.0
    #         if real_data_prior_weight > 0.0:
    #             rd_loss = real_data_prior_weight * real_data_prior_loss(x_query, full_seller_data)
    #
    #         # 2c) Scenario: 'score_known' => MSE
    #         if scenario == 'score_known':
    #             if observed_scores is None:
    #                 raise ValueError("observed_scores must be provided for 'score_known' scenario.")
    #             loss_main = F.mse_loss(pred_scores, observed_scores.to(device))
    #
    #         # 2d) Scenario: 'selection_only' => ranking or top-K
    #         elif scenario == 'selection_only':
    #             if top_k_selection is not None and top_k_selection > 0:
    #                 # Use top-K margin approach
    #                 loss_main = top_k_margin_loss(pred_scores, selected_indices,
    #                                               k=top_k_selection, margin=margin)
    #             else:
    #                 # Pairwise selected vs unselected approach
    #                 if ranking_loss_type == 'hinge':
    #                     loss_main = hinge_margin_ranking_loss(pred_scores, sel_mask, margin=margin)
    #                 elif ranking_loss_type == 'logistic':
    #                     loss_main = logistic_ranking_loss(pred_scores, sel_mask, temperature=1.0)
    #                 else:
    #                     raise ValueError("Unknown ranking_loss_type. Use 'hinge' or 'logistic'.")
    #         else:
    #             raise ValueError("Unknown scenario. Use 'score_known' or 'selection_only'.")
    #
    #         total_loss = loss_main + reg_loss + rd_loss
    #         total_loss.backward()
    #         optimizer.step()
    #         loss_history.append(total_loss.item())
    #
    #         if verbose and (it + 1) % max(1, num_iters // 10) == 0:
    #             print(f"Iter {it + 1}/{num_iters}, Loss={total_loss.item():.6f} "
    #                   f"(Main={loss_main.item():.4f}, Reg={reg_loss.item():.4f}, RD={rd_loss:.4f})")
    #
    #     return x_query.detach().clone(), loss_history

    # 3. Multi-Restart
    best_x_query = None
    best_loss = float('inf')
    best_history = None

    for r in range(num_restarts):
        x_query_hat, hist = single_run(initial_guess)
        noise_std = 0.01

        # Generate noise with the same shape as initial_guess and add it, creating a new leaf tensor:
        noise = torch.randn_like(initial_guess) * noise_std
        x_query = (initial_guess + noise).detach().clone().requires_grad_(True)

        # If you want to update the initial guess with the noisy version:
        initial_guess = x_query
        final_loss = hist[-1]
        if verbose:
            print(f"[Restart {r + 1}/{num_restarts}] Final Loss: {final_loss:.6f}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_x_query = x_query_hat
            best_history = hist

    return best_x_query, best_history


def reconstruction_attack(full_seller_data, selected_indices, attack_scenario, attack_method="ranking",
                          observed_scores=None, num_restarts: int = 10, num_query_points: int = 1,
                          ranking_loss_type='hinge', initial_guess=None):
    print(selected_indices.shape[0])
    if attack_scenario == "score_known":
        x_recon, hist = reconstruct_query(
            full_seller_data,
            selected_indices,
            scenario='score_known',
            observed_scores=observed_scores,
            verbose=True,
            num_restarts=num_restarts,
            num_query_points=num_query_points,
            initial_guess=initial_guess
        )
    elif attack_scenario == "selection_only":
        if attack_method == "ranking":
            x_recon, hist = reconstruct_query(
                full_seller_data,
                selected_indices,
                scenario='selection_only',
                num_query_points=num_query_points,
                aggregation_method='mean',  # average them to get the final "score"
                ranking_loss_type=ranking_loss_type,
                margin=0.2,
                reg_weight=1e-3,
                real_data_prior_weight=0.01,
                num_restarts=num_restarts,
                verbose=True,
                initial_guess=initial_guess
            )
        elif attack_method == "topk":
            x_recon, hist = reconstruct_query(
                full_seller_data,
                selected_indices,
                scenario='selection_only',
                top_k_selection=selected_indices.shape[0],
                ranking_loss_type='hinge',
                margin=0.1,
                verbose=True,
                num_restarts=num_restarts,
                num_query_points=num_query_points,
                initial_guess=initial_guess
            )
        else:
            raise NotImplementedError(
                f"Current attack: scenario: {attack_scenario}, method: {attack_method} not implemented.")
    else:
        raise NotImplementedError(
            f"Current attack: scenario: {attack_scenario}, method: {attack_method} not implemented.")
    return x_recon, hist


def evaluate_query_reconstruction(
        x_query_true,
        x_query_recon,
        full_seller_data=None,
        selected_indices=None,
        score_fn=None,
        compute_selection_overlap=False,
        top_k=None,
        metric='euclidean'
):
    """
    Evaluate how "close" x_query_recon is to x_query_true in:
      1) Feature distance (via matching if needed)
      2) (Optional) Selection overlap, if you re-score the dataset and pick top-K.

    Args:
        x_query_true : torch.Tensor shape (k_true, d) or just (d,)
        x_query_recon: torch.Tensor shape (k_recon, d) or just (d,)
        full_seller_data : (N, d), optional
        selected_indices : ground-truth selection (top-K indices used by the buyer)
        score_fn : function(x_query, data) -> scores (N,). If None, we skip selection overlap.
        compute_selection_overlap: bool, whether to compute top-K overlap metrics
        top_k : how many points were originally selected (if None, deduce from len(selected_indices))
        metric : distance metric for matching ('euclidean' supported here)

    Returns:
      A dictionary with:
        - 'total_distance'
        - 'avg_distance'
        - 'matching' (list of matched indices)
        - 'mse' (if k_true == k_recon, direct MSE across matched pairs)
        - 'selection_precision', 'selection_recall', 'selection_f1' (if compute_selection_overlap=True)
    """
    results = {}

    # 1) Match the reconstructed query with the true query
    x_query_true = x_query_true.cpu()
    x_query_recon = x_query_recon.cpu()
    total_dist, avg_dist, matching = match_query_sets(x_query_true, x_query_recon, metric=metric)
    results['total_distance'] = float(total_dist)
    results['avg_distance'] = float(avg_dist)
    results['matching'] = matching  # list of (i_true, i_recon) pairs
    # 2) MSE if both have the same cardinality
    # We'll directly compute MSE between matched pairs.
    # (If different cardinalities, we only handle matched pairs.)
    x_true = x_query_true.clone().detach()
    x_recon = x_query_recon.clone().detach()
    if x_true.ndim == 1:
        x_true = x_true.unsqueeze(0)
    if x_recon.ndim == 1:
        x_recon = x_recon.unsqueeze(0)

    if len(matching) > 0:
        # gather matched vectors
        matched_true = torch.stack([x_true[i] for (i, _) in matching], dim=0)
        matched_recon = torch.stack([x_recon[j] for (_, j) in matching], dim=0)
        # compute MSE
        mse_val = nn.MSELoss()(matched_true, matched_recon).item()
    else:
        mse_val = 0.0
    results['mse'] = mse_val

    # 3) If we want selection overlap, we need:
    #    a) full_seller_data
    #    b) a score_fn that re-scores the dataset
    #    c) the original selected_indices or top_k
    if compute_selection_overlap and (score_fn is not None) and (full_seller_data is not None):
        # We'll re-score the data using x_query_recon:
        # shape checks
        if x_recon.ndim == 1:
            x_recon = x_recon.unsqueeze(0)
        new_scores = score_fn(x_recon, full_seller_data)
        # pick top_k = len(selected_indices) if not provided
        if top_k is None and selected_indices is not None:
            top_k = len(selected_indices)
        if top_k is None:
            raise ValueError("Must specify top_k or selected_indices to compute selection overlap.")

        _, new_topk = torch.sort(new_scores, descending=True)
        new_selected_indices = new_topk[:top_k]

        # measure overlap with old selected_indices
        if selected_indices is not None:
            old_sel_set = set(selected_indices.tolist())
            new_sel_set = set(new_selected_indices.tolist())
            intersection = len(old_sel_set.intersection(new_sel_set))
            union = len(old_sel_set.union(new_sel_set))
            prec = intersection / max(len(new_sel_set), 1)
            rec = intersection / max(len(old_sel_set), 1)
            f1 = 0.0
            if (prec + rec) > 0:
                f1 = 2 * prec * rec / (prec + rec)
            results['selection_precision'] = prec
            results['selection_recall'] = rec
            results['selection_f1'] = f1
        else:
            # If we only know top_k but not which indices were selected, skip
            results['selection_precision'] = None
            results['selection_recall'] = None
            results['selection_f1'] = None

    return results


def run_reconstruction_attack_eval(
        full_seller_data,
        selected_indices,
        x_query_true,
        scenario='score_known',
        observed_scores=None,
        attack_method="ranking",
        # Attack parameters
        num_query_points=1,
        ranking_loss_type='hinge',
        real_data_prior_weight=0.0,
        # Baseline flags
        use_baseline='none',  # 'none', 'random', 'centroid'
        # For demonstration
        fisher_lambda=1e-3,
        lr=0.1,
        num_iters=1000,
        reg_weight=1e-3,
        margin=0.1,
        num_restarts=1,
        verbose=False,
        device="cuda",
        initial_mode="mean",
):
    """
    1) Either run the reconstruction_attack or a baseline.
    2) Evaluate quality vs x_query_true in multiple ways.
    3) Return a dict of metrics + the reconstructed query.
    """
    print(
        f"start attack: recontruction, scenario: {scenario} method {attack_method}, device: {device}, ranking loss: {ranking_loss_type}")
    d = full_seller_data.shape[1]
    x_query_true = torch.tensor(x_query_true, dtype=torch.float)
    # 1) Get reconstructed query x_query_recon
    if use_baseline == 'random':
        x_query_recon = baseline_random_guess(num_query_points, d)
    elif use_baseline == 'centroid':
        x_query_recon = baseline_centroid_of_selected(full_seller_data, selected_indices, num_query_points)
    else:
        # use the gradient-based reconstruct_query
        seller_data_tensor = torch.tensor(full_seller_data, dtype=torch.float).to(device)
        if initial_mode == "mean":
            initial_guess = baseline_centroid_of_selected(full_seller_data, selected_indices, num_query_points)
            initial_guess = torch.tensor(initial_guess, device=device, dtype=torch.float, requires_grad=True)
        else:
            initial_guess = torch.randn(num_query_points, d, device=device, dtype=torch.float, requires_grad=True)
        x_query_recon, loss_history = reconstruction_attack(seller_data_tensor, selected_indices,
                                                            attack_scenario=scenario, attack_method=attack_method,
                                                            observed_scores=observed_scores,
                                                            num_restarts=num_restarts,
                                                            ranking_loss_type=ranking_loss_type,
                                                            initial_guess=initial_guess)

    res_score = evaluate_query_reconstruction(x_query_true,
                                              x_query_recon, compute_selection_overlap=False)

    # rescore  =  {
    #           'total_distance':
    #            'avg_distance':
    #           'matching':
    #           'mse'
    # }
    return res_score

# Uncomment to run the example:
# example_usage()


# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
#
# ###############################################################################
# #                               Attack Code                                   #
# ###############################################################################
#
# def compute_fisher_information(selected_data: torch.Tensor, lambda_reg: float = 1e-3):
#     """
#     Computes the Fisher information matrix:
#         I = sum_i (x_i x_i^T) + lambda_reg * I_d
#     and returns I_inv (the inverse).
#
#     selected_data: (n_selected, d)
#     lambda_reg   : small float for regularization
#     Returns: I_inv of shape (d, d)
#     """
#     device = selected_data.device
#     d = selected_data.shape[1]
#
#     # Sum up outer products
#     I = torch.zeros(d, d, device=device)
#     for x in selected_data:
#         x = x.view(1, d)
#         I += x.t() @ x
#     # Regularization
#     I += lambda_reg * torch.eye(d, device=device)
#
#     return torch.inverse(I)
#
# def score_function(x_query: torch.Tensor, I_inv: torch.Tensor, candidate_data: torch.Tensor):
#     """
#     Given a single query vector x_query (1, d), compute the "score" for each candidate x_j:
#        score_j = (x_query^T I_inv x_j)^2
#
#     x_query      : (1, d)
#     I_inv        : (d, d)
#     candidate_data: (N, d)
#     Returns: (N,) vector of scores
#     """
#     # (1,d) @ (d,d) = (1,d)
#     Q = x_query @ I_inv
#     # (1,d) @ (d,N) = (1,N) => square => shape: (1,N)
#     scores = (Q @ candidate_data.t())**2
#     # Return as (N,)
#     return scores.flatten()
#
# def ranking_loss(pred_scores: torch.Tensor, selected_mask: torch.Tensor, margin: float = 0.1):
#     """
#     Pairwise hinge-style ranking loss:
#        For each selected candidate, its score should exceed that of unselected candidates by 'margin'.
#
#     pred_scores  : (N,) predicted scores
#     selected_mask: (N,) binary (1=selected, 0=not selected)
#     margin       : margin for ranking
#     Returns: scalar loss
#     """
#     selected_scores = pred_scores[selected_mask == 1]
#     unselected_scores = pred_scores[selected_mask == 0]
#     if selected_scores.numel() == 0 or unselected_scores.numel() == 0:
#         return torch.tensor(0.0, device=pred_scores.device)
#
#     # We want: selected_scores >= unselected_scores + margin
#     # diff shape => (num_sel, num_unsel)
#     diff = selected_scores.unsqueeze(1) - unselected_scores.unsqueeze(0)
#     loss_mat = F.relu(margin - diff)
#     return loss_mat.mean()
#
# def reconstruct_query(
#     full_seller_data: torch.Tensor,
#     selected_indices: torch.Tensor,
#     scenario: str = 'score_known',
#     observed_scores: torch.Tensor = None,
#     lambda_reg: float = 1e-3,
#     lr: float = 0.1,
#     num_iters: int = 1000,
#     reg_weight: float = 1e-3,
#     margin: float = 0.1,
#     num_restarts: int = 1,
#     verbose: bool = False
# ):
#     """
#     High-level function to reconstruct an "aggregate" query vector x_query
#     given a scenario: 'score_known' or 'selection_only'.
#
#     Args:
#         full_seller_data           : (N, d) All data from the seller (both selected & unselected).
#         selected_indices           : Indices (within full_seller_data) that the buyer used to build Fisher info.
#         scenario                   : Either 'score_known' or 'selection_only'.
#         observed_scores            : Required if scenario='score_known'. Shape (N,) with a score per candidate data.
#         lambda_reg                 : Regularization for Fisher information matrix.
#         lr                         : Learning rate for Adam.
#         num_iters                  : Number of optimization iterations.
#         reg_weight                 : Weight for L2 regularization on x_query.
#         margin                     : Margin used in ranking loss for 'selection_only' scenario.
#         num_restarts               : Number of random restarts. We pick the best final solution.
#         verbose                    : If True, print intermediate logs.
#
#     Returns:
#         best_x_query : (1, d) The reconstructed query vector that best explains the
#                        observed pattern (scores or selection).
#         best_history : A list of the final run's training loss values (for analysis).
#     """
#     device = full_seller_data.device
#     d = full_seller_data.shape[1]
#
#     # 1. Compute Fisher inverse from the selected data
#     selected_data = full_seller_data[selected_indices]
#     I_inv = compute_fisher_information(selected_data, lambda_reg=lambda_reg)
#
#     # 2. Identify the "candidate" set: all points in full_seller_data
#     #    (In some protocols, you'd differentiate between the "already selected" set
#     #    vs. "candidate" set. Here we assume the entire full_seller_data is relevant
#     #    for the reported scores or final selection.)
#     candidate_data = full_seller_data
#
#     # 3. Attack function for a single run
#     def single_run():
#         x_query = torch.randn(1, d, device=device, requires_grad=True)
#         optimizer = optim.Adam([x_query], lr=lr)
#         loss_history = []
#
#         for it in range(num_iters):
#             optimizer.zero_grad()
#             pred_scores = score_function(x_query, I_inv, candidate_data)
#
#             # L2 prior on x_query to keep it "reasonable"
#             reg_loss = reg_weight * (x_query**2).mean()
#
#             if scenario == 'score_known':
#                 # MSE with the observed scores
#                 loss_main = F.mse_loss(pred_scores, observed_scores)
#             elif scenario == 'selection_only':
#                 # Ranking loss with selected vs unselected
#                 # Convert selected_candidate_indices to a mask if not already
#                 if selected_indices.dtype == torch.bool:
#                     sel_mask = selected_indices
#                 else:
#                     # We assume it's a list of indices => build a mask
#                     sel_mask = torch.zeros(len(candidate_data), dtype=torch.float, device=device)
#                     sel_mask[selected_indices] = 1.0
#                 loss_main = ranking_loss(pred_scores, sel_mask, margin=margin)
#             else:
#                 raise ValueError("Unknown scenario. Use 'score_known' or 'selection_only'.")
#
#             total_loss = loss_main + reg_loss
#             total_loss.backward()
#             optimizer.step()
#             loss_history.append(total_loss.item())
#
#             if verbose and (it+1) % 100 == 0:
#                 print(f"Iter {it+1}/{num_iters}, Loss={total_loss.item():.6f}")
#
#         return x_query.detach(), loss_history
#
#     # 4. Multi-Restart
#     best_x_query = None
#     best_loss = float('inf')
#     best_history = None
#     for r in range(num_restarts):
#         x_query_hat, hist = single_run()
#         final_loss = hist[-1]
#         if verbose:
#             print(f"Restart {r+1}/{num_restarts}, Final Loss: {final_loss:.6f}")
#         if final_loss < best_loss:
#             best_loss = final_loss
#             best_x_query = x_query_hat
#             best_history = hist
#
#     return best_x_query, best_history
#
#
# ###############################################################################
# #                          Example Usage (Demo)                                #
# ###############################################################################
# if __name__ == "__main__":
#     # For reproducibility
#     torch.manual_seed(42)
#     np.random.seed(42)
#
#     # 1) Create a synthetic dataset: let's say we have 100 total data points, feature dim = 5
#     N = 100
#     d = 5
#     full_seller_data = torch.randn(N, d)   # shape (100, 5)
#
#     # 2) Suppose the buyer *already selected* 20 of these points (randomly chosen here)
#     selected_indices = torch.randperm(N)[:20]  # shape (20,)
#
#     # 3) We compute the "true" query vector (unknown to the seller).
#     x_query_true = torch.randn(1, d)
#
#     # 4) We'll create some "observed" transparency data for demonstration:
#     #    (A) Score-Known scenario: we pretend the buyer reported the score for each data point
#     #        score_j = (x_query_true^T I_inv x_j)^2
#     #    (B) Selection-Only scenario: the buyer eventually picks top_k based on those scores.
#
#     # (A) Score-Known scenario
#     I_inv_demo = compute_fisher_information(full_seller_data[selected_indices], lambda_reg=1e-3)
#     with torch.no_grad():
#         observed_scores_demo = score_function(x_query_true, I_inv_demo, full_seller_data)
#
#     # (B) Selection-Only scenario: pick the top 10 as "selected"
#     top_k = 10
#     _, sorted_idx = torch.sort(observed_scores_demo, descending=True)
#     selected_candidate_indices_demo = sorted_idx[:top_k]
#
#     # ================================
#     # Score-Known Attack
#     # ================================
#     print("\n=== Score-Known Attack ===")
#     x_query_hat_known, history_known = reconstruct_query(
#         full_seller_data=full_seller_data,
#         selected_indices=selected_indices,
#         scenario='score_known',
#         observed_scores=observed_scores_demo,
#         # The next arguments are optional, but you can tweak them
#         lambda_reg=1e-3, lr=0.1, num_iters=500, reg_weight=1e-3,
#         num_restarts=3, verbose=True
#     )
#
#     # ================================
#     # Selection-Only Attack
#     # ================================
#     print("\n=== Selection-Only Attack ===")
#     x_query_hat_selection, history_selection = reconstruct_query(
#         full_seller_data=full_seller_data,
#         selected_indices=selected_indices,
#         scenario='selection_only',
#         selected_candidate_indices=selected_candidate_indices_demo,  # top 10
#         lambda_reg=1e-3, lr=0.1, num_iters=500, reg_weight=1e-3,
#         margin=0.1,
#         num_restarts=3, verbose=True
#     )
#
#     # ================================
#     # Compare
#     # ================================
#     print("\n[True query vector]:")
#     print(x_query_true)
#     print("[Reconstructed query - Score-Known]:")
#     print(x_query_hat_known)
#     print("[Reconstructed query - Selection-Only]:")
#     print(x_query_hat_selection)
