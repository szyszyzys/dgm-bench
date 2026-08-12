import logging
from typing import Dict, List, Tuple, Any

import torch

from src.mechanism.gradient.aggregators.base_aggregator import BaseAggregator

logger = logging.getLogger("Aggregator")


class BulyanAggregator(BaseAggregator):
    """
    Bulyan: The Hidden Vulnerability of Distributed Learning in Byzantium (ICML 2018).

    Cascaded two-phase Byzantine-resilient aggregation:
      Phase 1 (Krum Selection): Select theta = N - 2f sellers with lowest
          Krum distance scores (nearest-neighbor sums).
      Phase 2 (Trimmed Median): For each coordinate, find the median of the
          theta selected gradients, then average the gamma = theta - 2f values
          closest to that median.

    Requires N >= 4f + 3 to guarantee Byzantine resilience.

    Args:
        num_byzantine: Expected maximum number of malicious sellers (f).
    """

    def __init__(self, *args, num_byzantine: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_byzantine = num_byzantine

    def aggregate(self, global_epoch: int, seller_updates: Dict[str, List[torch.Tensor]],
                  root_gradient: List[torch.Tensor] = None, **kwargs) -> Tuple[
        List[torch.Tensor], List[str], List[str], Dict[str, Any]]:

        logger.info(f"--- Bulyan Aggregation (Epoch {global_epoch}, f={self.num_byzantine}) ---")

        valid_sellers = list(seller_updates.keys())
        if not valid_sellers:
            logger.warning("No valid seller updates received.")
            zero_grad = [torch.zeros_like(p) for p in self.global_model.parameters()]
            return zero_grad, [], [], {}

        num_sellers = len(valid_sellers)
        f_requested = self.num_byzantine

        # Constraint: Bulyan needs N >= 4f + 3. Instead of falling back to
        # FedAvg when N is too small (which silently turns the "Bulyan" column
        # of the headline table into duplicate FedAvg numbers), we *clamp* f
        # downward to the largest value that still satisfies the constraint.
        # This keeps the Bulyan algorithm running on whatever pool size the
        # marketplace selection happens to deliver.
        #
        # Edge cases:
        #   N < 3 -> Bulyan literally cannot run (Phase 1 needs >= 3 sellers
        #            for Krum). In that case we still fall back to FedAvg.
        #   3 <= N < 7 -> clamp f to 0 (1-step trimmed median essentially).
        #   N >= 7    -> use the requested f, capped at (N-3)//4.
        max_feasible_f = max(0, (num_sellers - 3) // 4)
        f = min(f_requested, max_feasible_f)

        if num_sellers < 3:
            logger.warning(
                f"Bulyan: N={num_sellers} < 3. Cannot run Krum phase. Falling back to FedAvg."
            )
            return self._fedavg_fallback(seller_updates, valid_sellers)

        if f < f_requested:
            logger.warning(
                f"Bulyan: requested f={f_requested} needs N>={4*f_requested+3}, "
                f"but got N={num_sellers}. Clamping f -> {f} so the algorithm "
                f"still runs (not falling back to FedAvg)."
            )

        # --- Flatten all gradients ---
        flat_updates = {}
        for sid in valid_sellers:
            flat_updates[sid] = torch.cat([p.view(-1) for p in seller_updates[sid]])

        seller_ids = list(flat_updates.keys())
        stacked = torch.stack([flat_updates[sid] for sid in seller_ids])  # (N, D)

        # =====================================================================
        # Phase 1: Krum Selection — pick theta = N - 2f sellers
        # =====================================================================
        theta = num_sellers - 2 * f
        selected_indices = self._krum_select(stacked, num_sellers, f, theta)
        selected_sids = [seller_ids[i] for i in selected_indices]
        outlier_sids = [sid for sid in seller_ids if sid not in selected_sids]

        logger.info(
            f"Bulyan Phase 1: Krum selected {len(selected_indices)}/{num_sellers} sellers "
            f"(theta={theta}, f={f})"
        )

        # =====================================================================
        # Phase 2: Coordinate-wise Trimmed Median
        # =====================================================================
        selected_flat = stacked[selected_indices]  # (theta, D)
        gamma = theta - 2 * f

        if gamma < 1:
            logger.warning(f"Bulyan: gamma={gamma} < 1. Falling back to median of selected.")
            aggregated_flat = torch.median(selected_flat, dim=0).values
        else:
            aggregated_flat = self._trimmed_median(selected_flat, gamma)

        logger.info(
            f"Bulyan Phase 2: Trimmed median with gamma={gamma} "
            f"(averaging {gamma} closest-to-median values per coordinate)"
        )

        # --- Unflatten back to parameter shapes ---
        aggregated_gradient = []
        offset = 0
        for p in self.global_model.parameters():
            numel = p.numel()
            aggregated_gradient.append(aggregated_flat[offset:offset + numel].view(p.shape))
            offset += numel

        aggregation_stats = {
            "num_sellers": num_sellers,
            "num_byzantine_assumed": f,
            "theta_selected": len(selected_indices),
            "gamma_averaged": gamma,
            "outliers": outlier_sids,
        }

        return aggregated_gradient, selected_sids, outlier_sids, aggregation_stats

    def _krum_select(self, stacked: torch.Tensor, n: int, f: int, theta: int) -> List[int]:
        """
        Multi-Krum selection: pick theta sellers with lowest neighbor distance sums.
        For each seller, sum distances to its (N - f - 2) closest neighbors.
        """
        # Pairwise squared Euclidean distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        norms_sq = (stacked ** 2).sum(dim=1)  # (N,)
        dist_sq = norms_sq.unsqueeze(0) + norms_sq.unsqueeze(1) - 2 * torch.mm(stacked, stacked.t())
        dist_sq = dist_sq.clamp(min=0.0)  # numerical safety

        num_neighbors = n - f - 2
        if num_neighbors < 1:
            num_neighbors = 1

        # For each seller, sort distances and sum the closest num_neighbors
        krum_scores = []
        for i in range(n):
            dists = dist_sq[i].clone()
            dists[i] = float('inf')  # exclude self
            sorted_dists, _ = torch.sort(dists)
            score = sorted_dists[:num_neighbors].sum().item()
            krum_scores.append(score)

        # Select theta sellers with lowest scores
        sorted_indices = sorted(range(n), key=lambda i: krum_scores[i])
        return sorted_indices[:theta]

    def _trimmed_median(self, selected_flat: torch.Tensor, gamma: int) -> torch.Tensor:
        """
        For each coordinate, find the median, then average the gamma values
        closest to that median.
        """
        theta = selected_flat.shape[0]
        D = selected_flat.shape[1]

        # Median per coordinate: (D,)
        medians = torch.median(selected_flat, dim=0).values

        # Distance of each selected seller to the median, per coordinate: (theta, D)
        dists_to_median = torch.abs(selected_flat - medians.unsqueeze(0))

        # For each coordinate, find indices of gamma closest values
        # Sort by distance to median along dim=0
        _, sorted_idx = torch.sort(dists_to_median, dim=0)  # (theta, D)

        # Gather the gamma closest values per coordinate
        closest_idx = sorted_idx[:gamma, :]  # (gamma, D)

        # Gather actual values
        closest_values = torch.gather(selected_flat, dim=0, index=closest_idx)  # (gamma, D)

        # Average
        return closest_values.mean(dim=0)  # (D,)

    def _fedavg_fallback(self, seller_updates, valid_sellers):
        """Simple FedAvg when Bulyan constraints are not met."""
        aggregated_gradient = [torch.zeros_like(p) for p in self.global_model.parameters()]
        for sid in valid_sellers:
            for agg_grad, upd_grad in zip(aggregated_gradient, seller_updates[sid]):
                agg_grad.add_(upd_grad, alpha=1.0 / len(valid_sellers))
        return aggregated_gradient, valid_sellers, [], {"fallback": "fedavg", "reason": "N < 4f+3"}
