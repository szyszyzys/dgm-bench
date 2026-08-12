import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig


class ContributionEvaluator:
    """
    Handles the valuation of seller contributions for a single round.

    This class takes the state of the marketplace after gradient collection
    and computes various contribution scores for each seller.
    """

    def __init__(self, cfg: AppConfig):
        """
        Initializes the evaluator.

        Args:
            cfg: The main AppConfig object, used to check for settings
                 like compute_gradient_similarity.
        """
        self.cfg = cfg
        logging.info("ContributionEvaluator initialized.")

    def _flatten_and_calc_similarity(self, grad1: List[torch.Tensor],
                                     grad2: List[torch.Tensor]) -> Optional[float]:
        """Helper to safely flatten two gradients and compute cosine similarity."""
        if not grad1 or not grad2:
            return None
        try:
            # Move to CPU for similarity calculation to avoid CUDA sync issues
            flat1 = torch.cat([g.detach().cpu().flatten() for g in grad1])
            flat2 = torch.cat([g.detach().cpu().flatten() for g in grad2])

            # Handle potential all-zero gradients
            if torch.all(flat1 == 0) or torch.all(flat2 == 0):
                return 0.0

            return torch.nn.functional.cosine_similarity(
                flat1.unsqueeze(0),
                flat2.unsqueeze(0)
            ).item()
        except Exception as e:
            logging.error(f"Error in similarity calculation: {e}")
            return None

    def evaluate_round(
            self,
            round_number: int,
            seller_gradients: Dict[str, List[torch.Tensor]],
            seller_stats: Dict[str, Dict[str, Any]],
            oracle_gradient: Optional[List[torch.Tensor]],
            buyer_gradient: Optional[List[torch.Tensor]],
            aggregated_gradient: Optional[List[torch.Tensor]],
            aggregation_stats: Dict[str, Any],
            selected_ids: List[str],
            outlier_ids: List[str]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Computes all contribution scores for the current round.

        Returns:
            Tuple[
                Dict[str, Dict[str, Any]],  # seller_valuations
                Dict[str, Any]             # aggregate_metrics
            ]
        """
        seller_ids = list(seller_gradients.keys())
        seller_valuations = {sid: {} for sid in seller_ids}
        aggregate_metrics = {}

        # --- 1. Basic Stats & Market-Based Valuation ---
        for sid in seller_ids:
            stats = seller_stats.get(sid, {})
            seller_valuations[sid]['selected'] = sid in selected_ids
            seller_valuations[sid]['outlier'] = sid in outlier_ids
            seller_valuations[sid]['train_loss'] = stats.get('train_loss')
            seller_valuations[sid]['num_samples'] = stats.get('num_samples', 0)

            # Market-Based (Implicit) Valuation
            seller_valuations[sid]['selection_score'] = 1.0 if (sid in selected_ids) else 0.0

            # Market-Based (Weight) Valuation
            if 'seller_weights' in aggregation_stats:
                weight = aggregation_stats['seller_weights'].get(sid, 0.0)
                seller_valuations[sid]['aggregation_weight'] = weight

        # --- 2. Gradient Norm (Effort/Magnitude) ---
        gradient_norms = []
        for sid, grad in seller_gradients.items():
            if grad:
                norm = sum(torch.norm(g).item() ** 2 for g in grad) ** 0.5
                seller_valuations[sid]['gradient_norm'] = norm
                gradient_norms.append(norm)

        if gradient_norms:
            aggregate_metrics['avg_gradient_norm'] = np.mean(gradient_norms)
            aggregate_metrics['std_gradient_norm'] = np.std(gradient_norms)
            aggregate_metrics['min_gradient_norm'] = np.min(gradient_norms)
            aggregate_metrics['max_gradient_norm'] = np.max(gradient_norms)

        # --- 3. Similarity-Based Valuation (Quality & Relevance) ---
        sims_to_oracle = []
        sims_to_buyer = []
        sims_to_aggregate = []

        for sid, grad in seller_gradients.items():
            # a) Oracle Similarity (True Quality)
            oracle_sim = self._flatten_and_calc_similarity(grad, oracle_gradient)
            seller_valuations[sid]['sim_to_oracle'] = oracle_sim
            if oracle_sim is not None:
                sims_to_oracle.append(oracle_sim)

            # b) Buyer Similarity (Task Relevance)
            buyer_sim = self._flatten_and_calc_similarity(grad, buyer_gradient)
            seller_valuations[sid]['sim_to_buyer'] = buyer_sim
            if buyer_sim is not None:
                sims_to_buyer.append(buyer_sim)

            # --- c) CGSV (Cosine Gradient Shapley Value) ---
            # This is the "Gradient Shapley" approximation.
            # It measures contribution as the alignment with the
            # final aggregated consensus gradient.
            cgsv_score = self._flatten_and_calc_similarity(grad, aggregated_gradient)
            seller_valuations[sid]['sim_to_aggregate_cgsv'] = cgsv_score
            if cgsv_score is not None:
                sims_to_aggregate.append(cgsv_score)
            # -----------------------------------------------
        # Aggregate similarity stats
        if sims_to_oracle:
            aggregate_metrics['avg_sim_to_oracle'] = np.mean(sims_to_oracle)
            aggregate_metrics['std_sim_to_oracle'] = np.std(sims_to_oracle)
        if sims_to_buyer:
            aggregate_metrics['avg_sim_to_buyer'] = np.mean(sims_to_buyer)
            aggregate_metrics['std_sim_to_buyer'] = np.std(sims_to_buyer)
        if sims_to_aggregate:
            aggregate_metrics['avg_sim_to_aggregate_cgsv'] = np.mean(sims_to_aggregate)
            aggregate_metrics['std_sim_to_aggregate_cgsv'] = np.std(sims_to_aggregate)
        # --- 4. Adversary Detection Metrics ---
        known_adversaries = [sid for sid in seller_ids if 'adv' in sid]
        detected_adversaries = [sid for sid in outlier_ids if 'adv' in sid]
        benign_outliers = [sid for sid in outlier_ids if 'bn' in sid]

        aggregate_metrics['num_known_adversaries'] = len(known_adversaries)
        aggregate_metrics['num_detected_adversaries'] = len(detected_adversaries)
        aggregate_metrics['num_benign_outliers'] = len(benign_outliers)
        aggregate_metrics['adversary_detection_rate'] = (
            len(detected_adversaries) / len(known_adversaries)
            if known_adversaries else (1.0 if not detected_adversaries else 0.0)  # 1.0 if 0/0
        )
        non_adv_count = len(seller_ids) - len(known_adversaries)
        aggregate_metrics['false_positive_rate'] = (
            len(benign_outliers) / non_adv_count
            if non_adv_count > 0 else 0.0
        )

        # --- 5. Pricing ---
        payment_model = getattr(self.cfg.valuation, 'payment_model', 'quality_based')
        BASE_PRICE_PER_ROUND = 0.1

        if payment_model == "weighted":
            # E1a: payment = the aggregation weight the filter already assigned
            # this round, renormalized so payments sum to 1 across paid sellers.
            # Read-out only — selection and aggregation are untouched.
            agg_method = self.cfg.aggregation.method
            if 'seller_weights' in aggregation_stats:
                weights = {sid: max(0.0, aggregation_stats['seller_weights'].get(sid, 0.0))
                           for sid in seller_ids}
            elif agg_method == "fedavg":
                # FedAvg assigns uniform weight to every aggregated seller.
                weights = {sid: (1.0 if seller_valuations[sid]['selected'] else 0.0)
                           for sid in seller_ids}
            else:
                # Parameter-level methods (pooling/masking) expose no per-seller
                # weight: payment is undefined this round.
                logging.warning(
                    f"payment_undefined: aggregator '{agg_method}' exposes no "
                    f"per-seller weights; skipping weighted payment for round {round_number}."
                )
                weights = None
                aggregate_metrics['payment_undefined'] = True

            if weights is not None:
                # Only selected sellers are paid; rejected sellers get 0.
                paid_weights = {sid: (w if seller_valuations[sid]['selected'] else 0.0)
                                for sid, w in weights.items()}
                total_w = sum(paid_weights.values())
                for sid in seller_ids:
                    seller_valuations[sid]['price_paid'] = (
                        paid_weights[sid] / total_w if total_w > 0 else 0.0
                    )
        elif payment_model == "proportional":
            # Pay proportional to the aggregation weight assigned by the defense.
            # Normalise weights so that the total payout budget = BASE_PRICE * n_selected.
            weights = {sid: seller_valuations[sid].get('aggregation_weight', 0.0)
                       for sid in seller_ids}
            total_w = sum(max(0.0, w) for w in weights.values()) or 1.0
            n_selected = max(1, sum(1 for sid in seller_ids
                                    if seller_valuations[sid]['selected']))
            budget = BASE_PRICE_PER_ROUND * n_selected
            for sid in seller_ids:
                seller_valuations[sid]['price_paid'] = (
                    budget * max(0.0, weights[sid]) / total_w
                )
        elif payment_model == "binary":
            # Fixed payment if selected, zero otherwise.
            for sid in seller_ids:
                seller_valuations[sid]['price_paid'] = (
                    BASE_PRICE_PER_ROUND if seller_valuations[sid]['selected'] else 0.0
                )
        else:  # "quality_based" (default)
            for sid in seller_ids:
                price = 0.0
                if seller_valuations[sid]['selected']:
                    oracle_sim = seller_valuations[sid].get('sim_to_oracle') or 0.0
                    price = BASE_PRICE_PER_ROUND * max(0.0, oracle_sim)
                seller_valuations[sid]['price_paid'] = price

        # --- 6. Adversary revenue share (E1a) ---
        # Continuous analogue of MSR: fraction of this round's total payout
        # captured by adversarial sellers. Defined whenever payments exist;
        # logged alongside (not instead of) the binary MSR/BSR metrics.
        payments = {sid: seller_valuations[sid].get('price_paid')
                    for sid in seller_ids
                    if seller_valuations[sid].get('price_paid') is not None}
        total_paid = sum(payments.values())
        if total_paid > 0:
            adv_paid = sum(p for sid, p in payments.items() if 'adv' in sid)
            aggregate_metrics['adversary_revenue_share'] = adv_paid / total_paid

        logging.info("Contribution evaluation complete.")
        return seller_valuations, aggregate_metrics
