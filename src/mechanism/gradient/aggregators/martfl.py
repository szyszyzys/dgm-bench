import copy
import logging
import random
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common_utils import clip_gradient_update, flatten_tensor
from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator
from src.marketplace.utils.gradient_market_utils.clustering import optimal_k_gap, kmeans

logger = logging.getLogger("Aggregator")


def _cluster_and_score_martfl(
        similarities: np.ndarray,
        max_k_param: int = 10  # Add the new argument with a default
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Cluster similarities and score inliers.
    Returns scores and clustering metadata.
    """
    clustering_info = {}

    if len(similarities) < 2:
        clustering_info['num_clusters'] = 1
        clustering_info['cluster_method'] = 'insufficient_data'
        return np.ones_like(similarities), clustering_info

    sim_reshaped = similarities.reshape(-1, 1)
    max_k = min(len(similarities) - 1, max_k_param)
    if max_k < 2:
        clustering_info['num_clusters'] = 1
        clustering_info['cluster_method'] = 'max_k_too_small'
        return np.ones_like(similarities), clustering_info

    # Find optimal k using gap statistic
    k = optimal_k_gap(sim_reshaped, k_max=max_k)
    clustering_info['optimal_k'] = k
    clustering_info['max_k_considered'] = max_k

    # Perform k-means clustering
    labels, centers = kmeans(x=sim_reshaped, k=k)

    # Identify inlier cluster (highest similarity center)
    inlier_label = np.argmax(centers)
    clustering_info['inlier_cluster_id'] = int(inlier_label)
    clustering_info['cluster_centers'] = centers
    clustering_info['cluster_labels'] = labels

    # Score: keep original similarity for inliers, zero for outliers
    scores = np.where(labels == inlier_label, similarities, 0.0)

    # Additional stats
    clustering_info['num_inliers'] = int(np.sum(labels == inlier_label))
    clustering_info['num_outliers'] = int(np.sum(labels != inlier_label))

    return scores, clustering_info


class MartflAggregator(BaseAggregator):
    requires_root_gradient = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip = kwargs.get('clip', True)
        self.change_base = kwargs.get('change_base', True)
        self.initial_baseline = kwargs.get('initial_baseline', "buyer")  # Use the value from kwargs
        self.max_k = kwargs.get('max_k', 10)  # Use the value from kwargs

        # --- Set internal state ---
        self.baseline_id = self.initial_baseline  # Initialize with the configured baseline
        self.root_gradient = None

        logger.info(f"MartflAggregator initialized:")
        logger.info(
            f"  - clip_gradients_server_side: {self.clip} (using clip_norm: {self.clip_norm if self.clip else 'N/A'})")
        logger.info(f"  - change_baseline_dynamically: {self.change_base}")
        logger.info(f"  - initial_baseline: '{self.initial_baseline}'")  # Log the actual initial baseline
        logger.info(f"  - max_clusters_k: {self.max_k}")

    def aggregate(
            self,
            global_epoch: int,
            seller_updates: Dict[str, List[torch.Tensor]],
            root_gradient: List[torch.Tensor],
            **kwargs
    ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"=== martFL Aggregation (Official Logic) (Round {global_epoch}) ===")
        # ... (Handle empty seller_updates as before) ...

        logger.info(f"Current baseline: {self.baseline_id}")
        seller_ids = list(seller_updates.keys())

        # 1. Flatten and optionally clip updates (Same as before)
        flat_updates = {}
        processed_updates = {}  # Keep unflattened clipped updates for aggregation later
        for sid, upd in seller_updates.items():
            if self.clip:
                clipped = clip_gradient_update(upd, self.clip_norm)
                flat_updates[sid] = flatten_tensor(clipped)
                processed_updates[sid] = clipped  # Store clipped
            else:
                flat_updates[sid] = flatten_tensor(upd)
                processed_updates[sid] = upd  # Store original

        # 2. Get baseline (Slight change: Official code calls it 'server')
        # We'll stick to 'baseline_id' for clarity but mimic the logic
        is_baseline_a_seller = self.baseline_id in seller_updates

        if is_baseline_a_seller:
            baseline_update_flat = flat_updates[self.baseline_id]
            baseline_source = 'seller'
            logger.info(f"Using seller '{self.baseline_id}' as baseline")
        else:
            # Fallback or initial state: use buyer's root gradient
            self.baseline_id = "buyer"  # Ensure state reflects reality
            trust_gradient = root_gradient
            if self.clip:
                trust_gradient = clip_gradient_update(trust_gradient, self.clip_norm)
            baseline_update_flat = flatten_tensor(trust_gradient)
            baseline_source = 'buyer_trust'
            logger.info("Using pre-computed buyer's trust gradient as baseline")

        baseline_norm = torch.norm(baseline_update_flat).item()

        # ... (Robustness check for baseline_update_flat numel == 0) ...
        # ... (Robustness check for stacking flat_updates.values()) ...

        # 3. Compute cosine similarities (Same as before)
        seller_updates_stack = torch.stack(list(flat_updates.values()))
        similarities = F.cosine_similarity(
            baseline_update_flat.unsqueeze(0),
            seller_updates_stack,
            dim=1
        )

        # 4. Initialize stats dictionary (Same as before)
        aggregation_stats = {
            'aggregation_method': 'martfl_official',  # Mark as different
            # ... (Rest of initial stats are the same) ...
            'cosine_similarities': {sid: sim.item() for sid, sim in zip(seller_ids, similarities)},
        }

        # --- 5. OFFICIAL CLUSTERING & SCORING LOGIC ---

        # a) Prepare data for clustering (ALWAYS exclude the baseline)
        if is_baseline_a_seller:
            baseline_idx = seller_ids.index(self.baseline_id)
            other_indices = [i for i in range(len(seller_ids)) if i != baseline_idx]
            np_similarities_for_clustering = similarities[other_indices].cpu().numpy()
        else:
            # Baseline is buyer, cluster everyone
            baseline_idx = -1  # Indicate no seller baseline
            other_indices = list(range(len(seller_ids)))  # All indices are 'other'
            np_similarities_for_clustering = similarities.cpu().numpy()

        # Handle case where only one seller remains after excluding baseline
        if len(np_similarities_for_clustering) < 2:
            logger.warning("Only one non-baseline seller, skipping clustering. All treated as inliers.")
            n_clusters = 1
            clusters_no_baseline = [0] * len(np_similarities_for_clustering)  # Assign all to cluster 0
            centroids = np_similarities_for_clustering.reshape(-1, 1) if len(
                np_similarities_for_clustering) > 0 else np.array([[0.0]])
            diameter = 0.0
        else:
            # b) Find optimal k using gap, with official adjustment
            diameter = np.max(np_similarities_for_clustering) - np.min(np_similarities_for_clustering)

            # Calculate max_k based on config and data size
            max_k_to_test = min(len(np_similarities_for_clustering) - 1, self.max_k)

            # Ensure max_k is at least 2 for gap statistic to run
            if max_k_to_test < 2:
                logger.warning(f"Not enough data to cluster (k_max={max_k_to_test}), defaulting k=1.")
                n_clusters = 1
            else:
                n_clusters = optimal_k_gap(
                    np_similarities_for_clustering.reshape(-1, 1),
                    k_max=max_k_to_test
                )
            if n_clusters == 1 and diameter > 0.05:
                n_clusters = 2

            # c) Perform k-means on non-baseline sellers
            clusters_no_baseline, centroids = kmeans(np_similarities_for_clustering.reshape(-1, 1),
                                                     n_clusters)  # Assumes kmeans exists

        # d) Force k=2 clustering (also only on non-baseline sellers)
        if len(np_similarities_for_clustering) < 2:
            clusters2_no_baseline = clusters_no_baseline  # Use the k=1 result
            centroids2 = centroids  # Use the k=1 centroids
        else:
            # Get the centroids from the k=2 clustering as well
            clusters2_no_baseline, centroids2 = kmeans(np_similarities_for_clustering.reshape(-1, 1), 2)

        inlier_label_k2 = np.argmax(centroids2)
        # e) Reconstruct full cluster arrays, inserting baseline placeholder (e.g., -1)
        clusters_full = [-1] * len(seller_ids)
        clusters2_full = [-1] * len(seller_ids)
        for i, original_idx in enumerate(other_indices):
            clusters_full[original_idx] = clusters_no_baseline[i]
            clusters2_full[original_idx] = clusters2_no_baseline[i]

        # f) Determine the "center" (centroid of the highest similarity cluster)
        # Note: Official code uses `centroids[-1]`, implying the last cluster is highest.
        # It's safer to use argmax on the centroids found by the *optimal* k clustering.
        inlier_cluster_label = np.argmax(centroids)
        center = centroids[inlier_cluster_label]  # Get the scalar value
        # g) Calculate the "border" distance (max distance within the k=2 inlier group)
        border = 0.0
        for i, sim_val in enumerate(similarities):
            # Use the k=2 inlier label (inlier_label_k2)
            if i != baseline_idx and clusters2_full[i] == inlier_label_k2:
                dist_from_center = abs(center - sim_val.item())
                if dist_from_center > border:
                    border = dist_from_center
        # h) Calculate final scores ('non_outliers' in official code)
        final_scores = np.zeros(len(seller_ids))
        candidate_server_indices = []  # Indices of potential next baselines

        for i in range(len(seller_ids)):
            if i == baseline_idx:
                final_scores[i] = 1.0  # Baseline always gets score 1
                candidate_server_indices.append(i)  # Baseline is always a candidate
                continue  # Skip rest of checks for baseline

            # Check if considered outlier by the k=2 clustering
            is_k2_outlier = (clusters2_full[i] != inlier_label_k2)  # Use the k=2 inlier label
            if is_k2_outlier or similarities[i].item() == 0.0:
                final_scores[i] = 0.0  # Assign score 0
            else:
                dist = abs(center - similarities[i].item())
                # Normalize distance by border, invert, clamp at 0
                score = max(0.0, 1.0 - dist / (border + 1e-9))
                final_scores[i] = score

                # Check if in the highest cluster from the *optimal* k clustering
                if clusters_full[i] == inlier_cluster_label:
                    candidate_server_indices.append(i)
                    final_scores[i] = 1.0  # Official code sets score to 1.0 for these

        # --- 6. Convert scores to weights (Normalization) ---
        weights = torch.tensor(final_scores, dtype=torch.float, device=self.device)
        total_weight = weights.sum()

        if total_weight > 1e-9:
            weights /= total_weight
        else:
            logger.warning("Total weight near zero! Setting all weights to 0")
            weights.zero_()

        # Update aggregation_stats (add clustering info, scores, weights)
        aggregation_stats['clustering'] = {
            'optimal_k': n_clusters,
            'centroids': centroids,
            'clusters_full': clusters_full,  # Includes baseline placeholder
            'clusters2_full': clusters2_full,  # Includes baseline placeholder
            'inlier_cluster_label_optimal_k': int(inlier_cluster_label),
            'center_similarity': center,
            'border_distance': border
        }
        aggregation_stats['inlier_scores'] = {sid: float(score) for sid, score in zip(seller_ids, final_scores)}
        aggregation_stats['seller_weights'] = {sid: w.item() for sid, w in zip(seller_ids, weights)}
        aggregation_stats['total_weight'] = total_weight.item()

        # --- 7. Aggregate gradients (Use processed_updates) ---
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for i, sid in enumerate(seller_ids):
            weight = weights[i].item()
            if weight > 0:
                # Use the clipped (or original) updates stored earlier
                for agg_grad, upd_grad in zip(aggregated_gradient, processed_updates[sid]):
                    agg_grad.add_(upd_grad, alpha=weight)

        # --- 8. OFFICIAL BASELINE SELECTION LOGIC ---
        if self.change_base:
            next_baseline_id = self.baseline_id  # Default to current baseline
            baseline_selection_info = {'reason': 'change_base_disabled_or_no_candidates'}

            # a) Identify candidates
            high_quality_candidates_idx = candidate_server_indices  # From scoring step
            # Use the actual low-quality cluster label (opposite of inlier label from k=2 clustering)
            low_quality_label = 1 - inlier_label_k2
            low_quality_candidates_idx = [i for i in other_indices if clusters2_full[i] == low_quality_label]

            # b) Identify candidates for random sampling (low quality - high quality)
            prepare_random_idx = list(set(low_quality_candidates_idx) - set(high_quality_candidates_idx))

            # Official code fallback logic
            if not prepare_random_idx and len(high_quality_candidates_idx) < 0.5 * len(seller_ids):
                prepare_random_idx = list(set(other_indices) - set(high_quality_candidates_idx))

            # c) Sample random candidates (beta = 0.1)
            beta = 0.1
            num_random_to_sample = int(beta * len(seller_ids))
            random_candidates_idx = random.sample(
                prepare_random_idx,
                min(num_random_to_sample, len(prepare_random_idx))
            )

            # d) Combine candidate sets
            final_candidate_indices = sorted(list(set(high_quality_candidates_idx) | set(random_candidates_idx)))
            final_candidate_sids = [seller_ids[i] for i in final_candidate_indices]

            baseline_selection_info = {
                'num_total_candidates': len(final_candidate_indices),
                'candidate_sids': final_candidate_sids,
                'num_high_quality': len(high_quality_candidates_idx),
                'num_random_low_quality': len(random_candidates_idx),
                'kappa_scores': {}
            }

            # e) Evaluate candidates (using kappa score, like your original code)
            if final_candidate_indices:
                best_seller_sid = self.baseline_id  # Default
                max_kappa = -float('inf')  # Use -inf for maximization

                # Reuse a single temp model to avoid repeated deepcopy
                temp_model = copy.deepcopy(self.global_model)
                global_state = {k: v.clone() for k, v in self.global_model.state_dict().items()}

                for idx in final_candidate_indices:
                    sid = seller_ids[idx]
                    # Reset temp model to global state, then subtract the update
                    temp_model.load_state_dict(global_state)
                    with torch.no_grad():
                        for model_p, update_p in zip(temp_model.parameters(), processed_updates[sid]):
                            model_p.sub_(update_p)
                    _, _, kappa_score, _ = self._evaluate_model(temp_model)
                    baseline_selection_info['kappa_scores'][sid] = kappa_score

                    if kappa_score > max_kappa:
                        max_kappa = kappa_score
                        best_seller_sid = sid

                next_baseline_id = best_seller_sid
                baseline_selection_info['best_kappa'] = max_kappa
                baseline_selection_info['selected_seller'] = next_baseline_id
                baseline_selection_info['reason'] = 'selected_best_kappa'

            else:
                logger.warning("No candidates found for baseline rotation, keeping current or falling back to buyer.")
                next_baseline_id = "buyer"  # Fallback if no candidates
                baseline_selection_info['reason'] = 'no_candidates_found'

            aggregation_stats['next_baseline_id'] = next_baseline_id
            aggregation_stats['baseline_selection'] = baseline_selection_info
            logger.info(f"Baseline rotation: {self.baseline_id} -> {next_baseline_id}")
            self.baseline_id = next_baseline_id

        # --- 9 & 10. Determine selected/outliers and add stats (Same as before, using final_scores) ---
        selected_ids = [sid for i, sid in enumerate(seller_ids) if final_scores[i] > 0]
        outlier_ids = [sid for i, sid in enumerate(seller_ids) if final_scores[i] == 0]

        return aggregated_gradient, selected_ids, outlier_ids, aggregation_stats
