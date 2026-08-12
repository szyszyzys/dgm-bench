import numpy as np
from scipy.linalg import inv

def contrastive_loss(q, S_obs, D, I_inv, gamma=1.0):
    """
    Compute the contrastive loss.

    Args:
        q (np.array): Query (shape: (d,)).
        S_obs (np.array): Selected data points (shape: (k, d)).
        D (np.array): Full dataset (shape: (n, d)).
        I_inv (np.array): Inverse of the FIM (shape: (d, d)).
        gamma (float): Margin parameter.

    Returns:
        loss (float): Contrastive loss.
        grad_q (np.array): Gradient of the loss w.r.t. q (shape: (d,)).
    """
    n, d = D.shape
    k = S_obs.shape[0]
    residuals = D @ q  # Simplified residuals (replace with actual model predictions if available)
    scores = np.linalg.norm(I_inv @ D.T * residuals, axis=0)

    loss = 0
    grad_q = np.zeros(d)
    for x_i in S_obs:
        idx_i = np.where((D == x_i).all(axis=1))[0][0]
        for x_j in D[np.isin(D, S_obs, invert=True)]:
            idx_j = np.where((D == x_j).all(axis=1))[0][0]
            diff = scores[idx_j] - scores[idx_i] + gamma
            if diff > 0:
                loss += diff
                grad_q += 2 * (I_inv @ (D[idx_j] - D[idx_i])) * residuals[idx_j]
    return loss, grad_q

def reconstruct_query(S_obs, D, q_init, alpha=0.1, gamma=1.0, max_iter=100, tol=1e-6):
    """
    Reconstruct the query using contrastive loss and FIM.

    Args:
        S_obs (np.array): Selected data points (shape: (k, d)).
        D (np.array): Full dataset (shape: (n, d)).
        q_init (np.array): Initial guess for q (shape: (d,)).
        alpha (float): Learning rate.
        gamma (float): Margin parameter.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        q_hat (np.array): Reconstructed query.
    """
    n, d = D.shape
    q = q_init.copy()
    I_inv = np.eye(d)  # Initialize inverse FIM (identity matrix)

    for t in range(max_iter):
        # Compute loss and gradient
        loss, grad_q = contrastive_loss(q, S_obs, D, I_inv, gamma)

        # Check convergence
        if t > 0 and np.linalg.norm(grad_q) < tol:
            break

        # Update q
        q -= alpha * grad_q

        # Update FIM using Sherman-Morrison formula
        residuals = D @ q
        scores = np.linalg.norm(I_inv @ D.T * residuals, axis=0)
        x_j = D[np.argmax(scores)]  # Select point with highest score
        u = I_inv @ x_j
        I_inv = (1 / (1 - alpha)) * I_inv - (alpha / (1 - alpha + alpha * x_j.T @ I_inv @ x_j)) * np.outer(u, u)

    return q