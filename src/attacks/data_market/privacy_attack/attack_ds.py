from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors


def reconstruction_loss(X_buy_hat, S_obs, X_sell, k):
    """
    Compute loss with distribution alignment and proper masking.
    """
    # --- Normalization: Match Seller Distribution ---
    mu_sell = np.mean(X_sell, axis=0)
    cov_sell = np.cov(X_sell, rowvar=False)
    inv_cov_sell = np.linalg.pinv(cov_sell)

    # Whitening transformation
    X_buy_hat_whitened = (X_buy_hat - mu_sell) @ sqrtm(inv_cov_sell)

    # --- Differentiable Selection Loss ---
    n = X_sell.shape[0]
    scores = np.linalg.norm(X_buy_hat_whitened @ X_sell.T, axis=0)  # Shape: (n,)

    # Find nearest neighbors of S_obs in X_sell
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_sell)
    _, selected_indices = nbrs.kneighbors(S_obs)  # Shape: (k, 1)
    selected_indices = selected_indices.flatten()  # Shape: (k,)

    selected_mask = np.zeros(n, dtype=int)
    selected_mask[selected_indices] = 1  # Now works

    # Softmax-based differentiable loss
    temperature = 0.1
    soft_scores = scores / temperature
    soft_probs = np.exp(soft_scores) / np.sum(np.exp(soft_scores))

    # Cross-entropy loss between soft_probs and selected_mask
    loss = -np.sum(selected_mask * np.log(soft_probs + 1e-8))

    # Gradient calculation
    grad_soft = (soft_probs - selected_mask) / temperature
    grad = grad_soft @ X_sell @ sqrtm(inv_cov_sell).T

    return loss, grad


def reconstruct_X_buy(S_obs, X_sell, k, X_buy_hat_init, alpha=0.01, max_iter=100, tol=1e-6):
    """
    Robust reconstruction with shape checks and normalization.
    """
    # Ensure X_buy_hat_init is 2D (even for single-sample reconstruction)
    if X_buy_hat_init.ndim == 1:
        X_buy_hat = X_buy_hat_init[np.newaxis, :]
    else:
        X_buy_hat = X_buy_hat_init.copy()

    prev_loss = np.inf
    for t in range(max_iter):
        loss, grad = reconstruction_loss(X_buy_hat, S_obs, X_sell, k)

        if np.abs(loss - prev_loss) < tol:
            break
        prev_loss = loss

        # Update with adaptive learning rate
        X_buy_hat -= alpha * grad

        # Project to unit sphere (optional)
        X_buy_hat = X_buy_hat / np.linalg.norm(X_buy_hat, axis=1, keepdims=True)

    return X_buy_hat.squeeze()  # Return 1D if input was 1D


# def reconstruction_loss(X_buy_hat, S_obs, X_sell, k):
#     """
#     Compute reconstruction loss with robustness to numerical precision issues.
#
#     Args:
#         X_buy_hat (np.array): Reconstructed buyer data (shape: (m, d)).
#         S_obs (np.array): Observed selected seller data (shape: (k, d)).
#         X_sell (np.array): Seller data (shape: (n, d)).
#         k (int): Number of selected points.
#
#     Returns:
#         loss (float): Reconstruction loss.
#         grad_X_buy_hat (np.array): Gradient of the loss w.r.t. X_buy_hat (shape: (m, d)).
#     """
#     # --- Find indices of S_obs in X_sell using approximate matching ---
#     nbrs = NearestNeighbors(n_neighbors=1).fit(X_sell)
#     _, indices = nbrs.kneighbors(S_obs)
#     selected_indices = indices.flatten()  # Shape: (k,)
#
#     # --- Compute scores and ranks ---
#     weights = np.ones(len(X_sell)) / len(X_sell)
#     inv_cov = np.linalg.pinv(X_sell.T @ np.diag(weights) @ X_sell)  # Pseudo-inverse for stability
#     scores = np.linalg.norm(inv_cov @ X_sell.T, axis=0)  # Shape: (n,)
#     ranks = np.argsort(-scores)  # Descending order
#
#     # --- Loss Calculation ---
#     # Penalize if selected points are not in top-k ranks
#     loss = 0
#     grad_X_buy_hat = np.zeros_like(X_buy_hat)
#
#     # For selected points: Penalize if rank > k
#     for idx in selected_indices:
#         if ranks[idx] >= k:
#             loss += (ranks[idx] - k + 1) ** 2
#             grad_X_buy_hat += 2 * (ranks[idx] - k + 1) * (inv_cov @ X_sell[idx])
#
#     # For unselected points: Penalize if rank < k
#     unselected_mask = np.ones(len(X_sell), dtype=bool)
#     unselected_mask[selected_indices] = False
#     for idx in np.where(unselected_mask)[0]:
#         if ranks[idx] < k:
#             loss += (k - ranks[idx]) ** 2
#             grad_X_buy_hat += 2 * (k - ranks[idx]) * (inv_cov @ X_sell[idx])
#
#     return loss, grad_X_buy_hat
#
#
# def reconstruct_X_buy(S_obs, X_sell, k, X_buy_hat_init, alpha=0.01, max_iter=100, tol=1e-6):
#     """
#     Reconstruct the buyer's data X_buy.
#
#     Args:
#         S_obs (np.array): Observed selected seller data (shape: (k, d)).
#         X_sell (np.array): Seller data (shape: (n, d)).
#         k (int): Number of selected points.
#         X_buy_hat_init (np.array): Initial guess for X_buy (shape: (m, d)).
#         alpha (float): Learning rate.
#         max_iter (int): Maximum number of iterations.
#         tol (float): Convergence tolerance.
#
#     Returns:
#         X_buy_hat (np.array): Reconstructed buyer data.
#     """
#     X_buy_hat = X_buy_hat_init.copy()
#     for t in range(max_iter):
#         loss, grad_X_buy_hat = reconstruction_loss(X_buy_hat, S_obs, X_sell, k)
#         if t > 0 and np.linalg.norm(grad_X_buy_hat) < tol:
#             break
#         X_buy_hat -= alpha * grad_X_buy_hat
#     return X_buy_hat


import numpy as np
from scipy.stats import multivariate_normal


def likelihood(q, S_obs, X_sell, alpha=1.0):
    """
    Compute the likelihood of the observed selection S_obs given the query q.

    Args:
        q (np.array): Query (shape: (d,)).
        S_obs (np.array): Observed selected data points (shape: (k, d)).
        X_sell (np.array): Seller's data (shape: (n, d)).
        alpha (float): Temperature parameter.

    Returns:
        log_likelihood (float): Log-likelihood of S_obs given q.
    """
    # Compute gradient influence (simplified for illustration)
    residuals = X_sell @ q  # Simplified residuals (replace with actual model predictions if available)
    scores = np.linalg.norm(X_sell * residuals[:, np.newaxis], axis=1)

    # Compute likelihood
    log_likelihood = alpha * np.sum(scores[np.isin(X_sell, S_obs).all(axis=1)])
    return log_likelihood


def prior(q, mean, cov):
    """
    Compute the log-prior probability of q.

    Args:
        q (np.array): Query (shape: (d,)).
        mean (np.array): Mean of the prior distribution (shape: (d,)).
        cov (np.array): Covariance of the prior distribution (shape: (d, d)).

    Returns:
        log_prior (float): Log-prior probability of q.
    """
    return multivariate_normal.logpdf(q, mean=mean, cov=cov)


def metropolis_hastings(S_obs, X_sell, prior_mean, prior_cov, num_samples=1000, step_size=0.1, alpha=1.0):
    """
    Perform Metropolis-Hastings sampling to infer the posterior distribution of q.

    Args:
        S_obs (np.array): Observed selected data points (shape: (k, d)).
        X_sell (np.array): Seller's data (shape: (n, d)).
        prior_mean (np.array): Mean of the prior distribution (shape: (d,)).
        prior_cov (np.array): Covariance of the prior distribution (shape: (d, d)).
        num_samples (int): Number of MCMC samples.
        step_size (float): Step size for proposing new candidates.
        alpha (float): Temperature parameter for the likelihood.

    Returns:
        samples (np.array): Samples from the posterior distribution (shape: (num_samples, d)).
    """
    d = X_sell.shape[1]
    samples = np.zeros((num_samples, d))
    q_current = np.random.randn(d)  # Initialize q randomly

    for t in range(num_samples):
        # Propose a new candidate
        q_proposed = q_current + step_size * np.random.randn(d)

        # Compute acceptance probability
        log_likelihood_current = likelihood(q_current, S_obs, X_sell, alpha)
        log_likelihood_proposed = likelihood(q_proposed, S_obs, X_sell, alpha)
        log_prior_current = prior(q_current, prior_mean, prior_cov)
        log_prior_proposed = prior(q_proposed, prior_mean, prior_cov)

        log_acceptance = (log_likelihood_proposed + log_prior_proposed) - (log_likelihood_current + log_prior_current)
        acceptance_prob = min(1, np.exp(log_acceptance))

        # Accept or reject the candidate
        if np.random.rand() < acceptance_prob:
            q_current = q_proposed

        # Store the sample
        samples[t] = q_current

    return samples


def reconstruction_loss_nwe(
        X_buy_hat,  # shape (B, d) if you have B "buy" samples or (d,) if single
        S_obs,  # shape (k, d) the data you want the "selection" to match
        X_sell,  # shape (n, d)
        k
):
    """
    Compute reconstruction loss using:
      - 'nearest neighbors' in X_sell for the S_obs samples => selected_mask
      - L2-norm-based softmax scores in whitened space
      - cross-entropy with the selected_mask
      - correct chain rule for the norm
    """
    # Ensure 2D shape for X_buy_hat
    if X_buy_hat.ndim == 1:
        X_buy_hat = X_buy_hat[np.newaxis, :]

    B, d = X_buy_hat.shape
    n = X_sell.shape[0]

    # --- Compute whitening transform ---
    mu_sell = np.mean(X_sell, axis=0)
    cov_sell = np.cov(X_sell, rowvar=False)
    inv_cov_sell = np.linalg.pinv(cov_sell)
    W = sqrtm(inv_cov_sell)  # shape (d, d)

    # Whiten X_buy_hat
    X_buy_hat_whitened = (X_buy_hat - mu_sell) @ W  # shape (B, d)

    # --- Determine which indices in X_sell get "selected" ---
    # For simplicity, pick 1-NN for each S_obs row.
    # If S_obs has shape (k, d), then we get k indices
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_sell)
    _, selected_indices = nbrs.kneighbors(S_obs)
    selected_indices = selected_indices.flatten()  # shape (k,)

    selected_mask = np.zeros(n, dtype=float)
    selected_mask[selected_indices] = 1.0
    # If your S_obs has multiple rows, you might want to consider
    # counting them or using a multi-label approach; here we just mark 1 for each found index.

    # --- Compute the L2 norm scores in whitened space ---
    # M = X_buy_hat_whitened @ X_sell^T => shape (B, n)
    # We'll interpret each "column" i as the vector M[:, i], whose norm is our score.
    M = X_buy_hat_whitened @ X_sell.T  # shape (B, n)
    # s[i] = || M[:, i] ||_2
    # i.e. the norm over the "B" dimension (row dimension)
    s = np.sqrt(np.sum(M ** 2, axis=0) + 1e-8)  # shape (n,)

    # --- Softmax over s ---
    temperature = 0.1
    s_scaled = s / temperature
    exp_s = np.exp(s_scaled)
    soft_probs = exp_s / (np.sum(exp_s) + 1e-12)  # shape (n,)

    # --- Cross-entropy loss ---
    # If we treat selected_mask as a distribution, you might want a multi-hot or multi-label approach.
    # But for demonstration, we do standard "cross-entropy" with a distribution soft_probs vs. 0/1 mask
    loss = -np.sum(selected_mask * np.log(soft_probs + 1e-12))

    # --- Gradient w.r.t. the scores s ---
    # d(loss)/d(s[i]) = (1/T)*(soft_probs[i] - selected_mask[i])
    ds = (soft_probs - selected_mask) / temperature  # shape (n,)

    # --- Gradient w.r.t. M ---
    # Each s[i] = || M[:, i] ||_2
    # => d(s[i])/d(M[:, i]) = M[:, i] / s[i]
    # => d(loss)/d(M[:, i]) = ds[i] * ( M[:, i] / s[i] )
    # We'll do this in a vectorized way:
    factor = ds / (s + 1e-12)  # shape (n,)
    # so grad_M[:, i] = factor[i] * M[:, i]
    grad_M = M * factor[np.newaxis, :]  # shape (B, n)

    # --- Gradient w.r.t. X_buy_hat_whitened ---
    # M = X_buy_hat_whitened @ X_sell.T
    # => d(M[:, i])/d(X_buy_hat_whitened) involves X_sell[i]
    # grad_X_buy_hat_whitened = grad_M @ X_sell
    # But we have shape (B, n) in grad_M, and X_sell is (n, d)
    grad_X_buy_hat_whitened = grad_M @ X_sell  # shape (B, d)

    # --- Chain rule for the whitening transform ---
    # X_buy_hat_whitened = (X_buy_hat - mu_sell) @ W
    # => d(X_buy_hat_whitened)/d(X_buy_hat) = W
    # => final gradient = grad_X_buy_hat_whitened @ W^T
    grad = grad_X_buy_hat_whitened @ W.T  # shape (B, d)

    return loss, grad


def reconstruct_X_buy_new(
        S_obs,
        X_sell,
        k,
        X_buy_hat_init,
        alpha=0.01,
        max_iter=100,
        tol=1e-6,
        project_unit_sphere=False
):
    """
    Reconstruct X_buy by gradient descent on the objective that
    tries to match the selection pattern on X_sell for S_obs.
    """
    # Ensure 2D shape
    if X_buy_hat_init.ndim == 1:
        X_buy_hat = X_buy_hat_init[np.newaxis, :].copy()
    else:
        X_buy_hat = X_buy_hat_init.copy()

    prev_loss = np.inf
    for t in range(max_iter):
        loss, grad = reconstruction_loss_nwe(X_buy_hat, S_obs, X_sell, k)

        # Simple stopping criterion
        if abs(loss - prev_loss) < tol:
            break
        prev_loss = loss

        # Gradient update
        X_buy_hat -= alpha * grad

        # (Optional) project each row to the unit sphere
        if project_unit_sphere:
            norms = np.linalg.norm(X_buy_hat, axis=1, keepdims=True) + 1e-12
            X_buy_hat = X_buy_hat / norms

    # Return 1D if we started with 1D
    if X_buy_hat.shape[0] == 1:
        return X_buy_hat.squeeze()
    return X_buy_hat


# def reconstruction_loss_fim(X_buy_hat, S_obs, X_sell, k, temperature=0.1):
#     """
#     Compute a differentiable loss so that the 'selected' data points
#     in X_sell have higher softmax scores than the 'unselected' ones.
#
#     Args:
#         X_buy_hat (np.array): Current guess of buyer data, shape (d,).
#         S_obs (np.array): Observed selected seller data, shape (k, d).
#         X_sell (np.array): All seller data, shape (n, d).
#         k (int): Number of selected points.
#         temperature (float): Softmax temperature (lower => more 'peaky').
#
#     Returns:
#         loss (float): Cross-entropy loss (lower is better).
#         grad (np.array): Gradient of the loss w.r.t. X_buy_hat (shape (d,)).
#     """
#     n = len(X_sell)
#
#     # -------------------------------------------------------
#     # 1) Identify which seller points were selected
#     #    using nearest-neighbor to S_obs
#     # -------------------------------------------------------
#     nbrs = NearestNeighbors(n_neighbors=1).fit(X_sell)
#     _, indices = nbrs.kneighbors(S_obs)  # shape (k, 1)
#     selected_indices = indices.flatten()  # shape (k,)
#     # Build a mask: 1.0 for selected, 0.0 for unselected
#     selected_mask = np.zeros(n, dtype=float)
#     selected_mask[selected_indices] = 1.0
#
#     # -------------------------------------------------------
#     # 2) Define a "score" that depends on X_buy_hat
#     #    For each x_i in X_sell, score_i = -||X_buy_hat - x_i||^2
#     #
#     #    If we want multiple buyer vectors (m, d), we'd handle that differently.
#     # -------------------------------------------------------
#     # Distances: shape (n,)
#     dists_sq = np.sum((X_buy_hat - X_sell) ** 2, axis=1)
#     scores = -dists_sq  # higher => smaller distance
#
#     # -------------------------------------------------------
#     # 3) Softmax over scores => probability distribution
#     # -------------------------------------------------------
#     s_scaled = scores / temperature
#     # For numerical stability, subtract max(s_scaled) before exponent
#     exp_s = np.exp(s_scaled - np.max(s_scaled))
#     soft_probs = exp_s / np.sum(exp_s)  # shape (n,)
#
#     # -------------------------------------------------------
#     # 4) Cross-entropy loss w.r.t. the "selected_mask"
#     # -------------------------------------------------------
#     # If selected_mask is in {0,1} for each index:
#     #     loss = - sum_i [ selected_mask[i] * log(soft_probs[i]) ]
#     # Alternatively, if we had multi-hot or a distribution in selected_mask,
#     # we do the usual cross-entropy. (Here it's 1 for selected, 0 for unselected.)
#     loss = -np.sum(selected_mask * np.log(soft_probs + 1e-12))
#
#     # -------------------------------------------------------
#     # 5) Gradient w.r.t. X_buy_hat
#     # -------------------------------------------------------
#     #
#     # d(loss)/d(score_i) = (soft_probs[i] - selected_mask[i]) / temperature
#     # and score_i = -|| X_buy_hat - X_sell[i] ||^2
#     # => d(score_i)/d(X_buy_hat) = - d/dX_buy_hat [ (X_buy_hat - X_sell[i])^T (X_buy_hat - X_sell[i]) ]
#     # =>                       = - 2 (X_buy_hat - X_sell[i])
#     #
#     # => d(loss)/d(X_buy_hat) = sum_i [ d(loss)/d(score_i) * d(score_i)/d(X_buy_hat) ]
#     # =>                       = sum_i [ (soft_probs[i] - selected_mask[i])/T * ( -2 (X_buy_hat - X_sell[i]) ) ]
#     #
#     dloss_ds = (soft_probs - selected_mask) / temperature  # shape (n,)
#     grad = np.zeros_like(X_buy_hat)  # shape (d,)
#     for i in range(n):
#         grad += dloss_ds[i] * (-2.0 * (X_buy_hat - X_sell[i]))
#
#     return loss, grad
#
#
# def reconstruct_X_buy_fim(
#         S_obs,
#         X_sell,
#         k,
#         X_buy_hat_init,
#         alpha=0.001,
#         max_iter=1000,
#         tol=1e-6
# ):
#     """
#     Reconstruct X_buy by gradient descent to match which points in X_sell
#     are 'selected' for S_obs.
#
#     Args:
#         S_obs (np.array): Observed selected points from X_sell, shape (k, d).
#         X_sell (np.array): Seller data, shape (n, d).
#         k (int): Number of selected points.
#         X_buy_hat_init (np.array): Initial guess for buyer data, shape (d,).
#         alpha (float): Learning rate.
#         max_iter (int): Maximum iteration count.
#         tol (float): Convergence tolerance on the loss improvement.
#
#     Returns:
#         X_buy_hat (np.array): Reconstructed buyer data, shape (d,).
#     """
#     X_buy_hat = X_buy_hat_init.copy()
#     prev_loss = None
#
#     for t in range(max_iter):
#         loss, grad = reconstruction_loss_fim(X_buy_hat, S_obs, X_sell, k)
#
#         # Gradient update
#         X_buy_hat -= alpha * grad
#
#         # Check convergence
#         if prev_loss is not None and abs(prev_loss - loss) < tol:
#             break
#         prev_loss = loss
#
#     return X_buy_hat


# def compute_selected_mask(S_obs, X_sell):
#     nbrs = NearestNeighbors(n_neighbors=1).fit(X_sell)
#     _, indices = nbrs.kneighbors(S_obs)  # shape (k, 1)
#     selected_indices = indices.flatten()
#     selected_mask = np.zeros(len(X_sell), dtype=float)
#     selected_mask[selected_indices] = 1.0
#     return selected_mask
#
# def reconstruction_loss_fim(X_buy_hat, selected_mask, X_sell, temperature=0.1):
#     """
#     Compute a differentiable loss so that the 'selected' data points
#     in X_sell have higher softmax scores than the 'unselected' ones.
#
#     Args:
#         X_buy_hat (np.array): Current guess of buyer data, shape (d,).
#         selected_mask (np.array): Binary mask for selected seller data, shape (n,).
#         X_sell (np.array): All seller data, shape (n, d).
#         temperature (float): Softmax temperature (lower => more 'peaky').
#
#     Returns:
#         loss (float): Cross-entropy loss (lower is better).
#         grad (np.array): Gradient of the loss w.r.t. X_buy_hat (shape (d,)).
#     """
#     # 2) Compute scores: s_i = -||X_buy_hat - x_i||^2
#     dists_sq = np.sum((X_buy_hat - X_sell) ** 2, axis=1)
#     scores = -dists_sq
#
#     # 3) Compute softmax probabilities (with temperature scaling)
#     s_scaled = scores / temperature
#     exp_s = np.exp(s_scaled - np.max(s_scaled))
#     soft_probs = exp_s / np.sum(exp_s)
#
#     # 4) Compute cross-entropy loss
#     loss = -np.sum(selected_mask * np.log(soft_probs + 1e-12))
#
#     # 5) Compute gradient
#     dloss_ds = (soft_probs - selected_mask) / temperature
#     grad = -2.0 * np.sum((dloss_ds[:, np.newaxis] * (X_buy_hat - X_sell)), axis=0)
#
#     return loss, grad
#
# def reconstruct_X_buy_fim(
#         S_obs,
#         X_sell,
#         X_buy_hat_init,
#         alpha=0.001,
#         max_iter=1000,
#         tol=1e-6,
#         temperature=0.1
# ):
#     """
#     Reconstruct X_buy by gradient descent to match which points in X_sell
#     are 'selected' for S_obs.
#
#     Args:
#         S_obs (np.array): Observed selected points from X_sell, shape (k, d).
#         X_sell (np.array): Seller data, shape (n, d).
#         X_buy_hat_init (np.array): Initial guess for buyer data, shape (d,).
#         alpha (float): Learning rate.
#         max_iter (int): Maximum iteration count.
#         tol (float): Convergence tolerance on the loss improvement.
#         temperature (float): Softmax temperature.
#
#     Returns:
#         X_buy_hat (np.array): Reconstructed buyer data, shape (d,).
#     """
#     # Precompute the selected mask once
#     selected_mask = compute_selected_mask(S_obs, X_sell)
#
#     X_buy_hat = X_buy_hat_init.copy()
#     prev_loss = None
#
#     for t in range(max_iter):
#         loss, grad = reconstruction_loss_fim(X_buy_hat, selected_mask, X_sell, temperature)
#
#         # Gradient update
#         X_buy_hat -= alpha * grad
#
#         # Optionally, log progress every so often
#         if t % 100 == 0:
#             print(f"Iteration {t}, Loss: {loss:.6f}")
#
#         # Check convergence using change in loss
#         if prev_loss is not None and abs(prev_loss - loss) < tol:
#             break
#         prev_loss = loss
#
#     return X_buy_hat


import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_selected_mask(S_obs, X_sell):
    """
    Precompute a binary mask of which seller points were selected,
    based on nearest neighbor matching from S_obs to X_sell.
    """
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_sell)
    _, indices = nbrs.kneighbors(S_obs)  # shape (k, 1)
    selected_indices = indices.flatten()
    selected_mask = np.zeros(len(X_sell), dtype=float)
    selected_mask[selected_indices] = 1.0
    return selected_mask

def reconstruction_loss_fim(X_buy_hat, selected_indicies, X_sell,
                            temperature=0.1, lambda_reg=0.0, X_prior=None):
    """
    Compute the differentiable reconstruction loss such that the "selected"
    seller points (given by selected_indicies) have higher softmax scores.
    Also adds an optional L2 regularization term.

    Args:
        X_buy_hat (np.array): Current guess for buyer data, shape (d,).
        selected_indicies (np.array): Binary mask of selected seller points, shape (n,).
        X_sell (np.array): Seller data, shape (n, d).
        temperature (float): Temperature parameter for softmax.
        lambda_reg (float): Regularization weight.
        X_prior (np.array or None): Prior for X_buy; if provided, the loss penalizes
                                    deviations from X_prior.

    Returns:
        loss (float): The total loss (cross-entropy + regularization).
        grad (np.array): Gradient with respect to X_buy_hat (shape (d,)).
    """
    # Compute scores: score_i = -||X_buy_hat - x_i||^2 for each seller point x_i.
    dists_sq = np.sum((X_buy_hat - X_sell) ** 2, axis=1)
    scores = -dists_sq

    selected_mask = np.zeros(X_sell.shape[0], dtype=bool)

    # Set the entries at the selected indices to True
    selected_mask[selected_indicies] = True
    # Temperature scaling and numerical stabilization for softmax.
    s_scaled = scores / temperature
    exp_s = np.exp(s_scaled - np.max(s_scaled))
    soft_probs = exp_s / np.sum(exp_s)

    # Cross-entropy loss: encourage high probability for the selected seller points.
    loss = -np.sum(selected_mask * np.log(soft_probs + 1e-12))

    # Add L2 regularization.
    if X_prior is not None:
        loss += lambda_reg * np.sum((X_buy_hat - X_prior) ** 2)
    elif lambda_reg > 0:
        loss += lambda_reg * np.sum(X_buy_hat ** 2)

    # Gradient of the cross-entropy part:
    # d(loss)/d(score_i) = (soft_probs[i] - selected_mask[i]) / temperature.
    dloss_ds = (soft_probs - selected_mask) / temperature  # shape (n,)
    # Vectorized gradient computation:
    grad = -2.0 * np.sum(dloss_ds[:, np.newaxis] * (X_buy_hat - X_sell), axis=0)

    # Gradient for the regularization term.
    if X_prior is not None:
        grad += 2 * lambda_reg * (X_buy_hat - X_prior)
    elif lambda_reg > 0:
        grad += 2 * lambda_reg * X_buy_hat

    return loss, grad

def reconstruct_X_buy_fim(
    selected_indices,
    X_sell,
    X_buy_hat_init,
    alpha=0.001,
    max_iter=1000,
    tol=1e-6,
    temperature=0.1,
    lambda_reg=0.0,
    X_prior=None,
    grad_clip_threshold=1.0,
    lr_decay=0.9,
    decay_patience=50
):
    """
    Reconstruct the buyer's data X_buy by gradient descent with several improvements:
      - Uses a better initialization if possible.
      - Incorporates L2 regularization to constrain the solution.
      - Applies gradient clipping to prevent exploding gradients.
      - Adopts an adaptive learning rate scheme.

    Args:
        selected_indices (np.array): Observed selected points from X_sell, shape (k, d).
        X_sell (np.array): Seller data, shape (n, d).
        X_buy_hat_init (np.array): Initial guess for buyer data, shape (d,).
        alpha (float): Initial learning rate.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance for loss improvement.
        temperature (float): Softmax temperature.
        lambda_reg (float): Weight for the regularization term.
        X_prior (np.array or None): Prior for X_buy; if available, used for regularization.
        grad_clip_threshold (float): Maximum allowed norm for gradients.
        lr_decay (float): Factor to decay the learning rate if progress stalls.
        decay_patience (int): Number of iterations to wait before decaying the learning rate.

    Returns:
        X_buy_hat (np.array): Reconstructed buyer data, shape (d,).
    """
    # Optionally use a better initialization: if no X_prior is provided,
    # we initialize to the mean of the selected seller points.
    # Use the provided initial guess.
    X_buy_hat = X_buy_hat_init.copy()

    # Precompute the selected mask (this remains constant during optimization).

    prev_loss = np.inf
    best_loss = np.inf
    best_X_buy_hat = X_buy_hat.copy()
    patience_counter = 0

    for t in range(max_iter):
        loss, grad = reconstruction_loss_fim(
            X_buy_hat, selected_indices, X_sell,
            temperature=temperature, lambda_reg=lambda_reg, X_prior=X_prior
        )

        # Gradient clipping to avoid very large updates.
        grad_norm = np.linalg.norm(grad)
        if grad_norm > grad_clip_threshold:
            grad = grad * (grad_clip_threshold / grad_norm)

        # Gradient descent update.
        X_buy_hat -= alpha * grad

        # Adaptive learning rate: if no improvement in loss for several iterations,
        # reduce the learning rate.
        if loss < best_loss - tol:
            best_loss = loss
            best_X_buy_hat = X_buy_hat.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= decay_patience:
                alpha *= lr_decay
                patience_counter = 0
                print(f"Decayed learning rate to {alpha:.6f} at iteration {t}")

        # Optionally log progress.
        if t % 100 == 0:
            print(f"Iteration {t}, Loss: {loss:.6f}, Grad Norm: {grad_norm:.6f}")

        # Check convergence.
        if abs(prev_loss - loss) < tol:
            print(f"Convergence reached at iteration {t}")
            break

        prev_loss = loss

    # Return the best encountered reconstruction.
    return best_X_buy_hat
