from typing import Union, List, Dict

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.neighbors import NearestNeighbors


def evaluate_embeddings(
        E_orig: np.ndarray,
        E_recon: Union[np.ndarray, List[np.ndarray]]
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Evaluates the fidelity, distributional alignment, and semantic structure preservation
    between original embeddings (E_orig) and reconstructed embeddings (E_recon).

    Parameters:
        E_orig (np.ndarray): Original embeddings of shape (n_samples, n_features).
        E_recon (Union[np.ndarray, List[np.ndarray]]):
            Reconstructed embeddings. Can be a single array of shape (n_samples, n_features)
            or a list of such arrays for multiple reconstructions.

    Returns:
        Union[Dict[str, float], List[Dict[str, float]]]:
            A dictionary of metrics if a single reconstruction is provided,
            or a list of such dictionaries for multiple reconstructions.
    """

    # Helper function to compute metrics for a single reconstruction
    def compute_metrics_single(E_recon_single: np.ndarray) -> Dict[str, float]:
        metrics = {}

        # --- Per-Embedding Fidelity ---
        # Cosine similarity (pairwise)
        if E_recon_single.ndim == 1:
            E_recon_single = E_recon_single.reshape(1, -1)  # Shape: (1, n_features)
        cos_sim_matrix = cosine_similarity(E_orig, E_recon_single)

        if E_recon_single.shape[0] == 1:
            # Single reconstructed vector compared to all originals
            cos_sim = cos_sim_matrix.flatten()  # Shape: (n_samples,)
            metrics["cosine_sim_mean"] = np.mean(cos_sim)
            metrics["cosine_sim_std"] = np.std(cos_sim)
        else:
            # Multiple reconstructed vectors
            if E_orig.shape != E_recon_single.shape:
                raise ValueError("E_orig and E_recon_single must have the same shape.")
            cos_sim = cos_sim_matrix.diagonal()
            metrics["cosine_sim_mean"] = np.mean(cos_sim)
            metrics["cosine_sim_std"] = np.std(cos_sim)

        # MSE
        mse = np.mean((E_orig - E_recon_single) ** 2)
        metrics["mse"] = mse

        # Pearson correlation (per sample)
        if E_recon_single.shape[0] == 1:
            # Compute Pearson correlation between E_recon and each E_orig sample
            pearson_correlations = []
            for i in range(E_orig.shape[0]):
                orig_vector = E_orig[i]
                recon_vector = E_recon_single[0]
                corr, _ = pearsonr(orig_vector, recon_vector)
                pearson_correlations.append(corr)
            pearson_correlations = np.array(pearson_correlations)
            metrics["pearson_corr_mean"] = np.mean(pearson_correlations)
            metrics["pearson_corr_std"] = np.std(pearson_correlations)
        else:
            # Compute Pearson correlation between corresponding samples
            pearson_correlations = []
            for i in range(E_orig.shape[0]):
                orig_vector = E_orig[i]
                recon_vector = E_recon_single[i]
                corr, _ = pearsonr(orig_vector, recon_vector)
                pearson_correlations.append(corr)
            pearson_correlations = np.array(pearson_correlations)
            metrics["pearson_corr_mean"] = np.mean(pearson_correlations)
            metrics["pearson_corr_std"] = np.std(pearson_correlations)

        # --- Distributional Alignment ---
        # MMD with RBF kernel
        gamma = 1.0 / E_orig.shape[1]  # Bandwidth for RBF
        K_orig = rbf_kernel(E_orig, gamma=gamma)
        K_recon = rbf_kernel(E_recon_single, gamma=gamma)
        K_orig_mean = np.mean(K_orig)
        K_recon_mean = np.mean(K_recon)
        K_cross_mean = np.mean(rbf_kernel(E_orig, E_recon_single, gamma=gamma))
        mmd = K_orig_mean - 2 * K_cross_mean + K_recon_mean
        metrics["mmd"] = mmd

        # CKA (Linear) - Only if multiple samples
        if E_recon_single.shape[0] > 1:
            try:
                numerator = np.linalg.norm(E_orig.T @ E_recon_single, ord="fro") ** 2
                denominator = (
                        np.linalg.norm(E_orig.T @ E_orig, ord="fro") *
                        np.linalg.norm(E_recon_single.T @ E_recon_single, ord="fro")
                )
                cka = numerator / denominator if denominator != 0 else 0.0
            except ValueError as e:
                print(f"CKA computation error: {e}")
                cka = np.nan
            metrics["cka"] = cka
        else:
            metrics["cka"] = np.nan  # Not applicable for single reconstruction

        # --- Semantic Structure Preservation ---
        # Neighborhood preservation (k=5)
        k = 5
        n_recon_samples = E_recon_single.shape[0]

        if n_recon_samples >= k:
            nbrs_orig = NearestNeighbors(n_neighbors=k).fit(E_orig)
            distances_orig, indices_orig = nbrs_orig.kneighbors(E_orig)

            nbrs_recon = NearestNeighbors(n_neighbors=k).fit(E_recon_single)
            distances_recon, indices_recon = nbrs_recon.kneighbors(E_recon_single)

            # Define neighborhood overlap as the average proportion of overlapping neighbors
            # between E_orig and E_recon_single
            # However, since E_orig and E_recon_single are different datasets, defining overlap is non-trivial
            # One approach is to compute the intersection of neighbor indices across both
            # But since they are separate, consider alternative definitions or skip

            # Placeholder: Set to np.nan or redefine as needed
            metrics["neighborhood_overlap"] = np.nan
        else:
            # Not enough samples to compute neighborhood overlap
            metrics["neighborhood_overlap"] = np.nan

        return metrics

    # Determine if E_recon is single or multiple
    if isinstance(E_recon, list) or (isinstance(E_recon, np.ndarray) and E_recon.ndim == 3):
        # Multiple reconstructions
        if isinstance(E_recon, list):
            reconstructions = E_recon
        else:
            # Assuming E_recon is a 3D array: (n_recon, n_samples, n_features)
            reconstructions = [E_recon[i] for i in range(E_recon.shape[0])]

        all_metrics = []
        for idx, E_recon_single in enumerate(reconstructions):
            try:
                metrics = compute_metrics_single(E_recon_single)
                metrics["reconstruction_id"] = idx
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error evaluating reconstruction {idx}: {e}")
        return all_metrics
    else:
        # Single reconstruction
        metrics = compute_metrics_single(E_recon)
        return metrics
