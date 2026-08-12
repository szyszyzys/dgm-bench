# FILE: marketplace/market/valuation.py (New File)

import logging
from typing import Dict, List, Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig
from src.mechanism.gradient.aggregator import Aggregator
from src.mechanism.gradient.valuation.contribution_evaluator import ContributionEvaluator
from src.mechanism.gradient.valuation.influence_evaluator import InfluenceEvaluator
from src.mechanism.gradient.valuation.kernel_shapely import KernelSHAPEvaluator
from src.mechanism.gradient.valuation.round_based_evaluator import RoundBasedLOOEvaluator


class ValuationManager:
    """
    A unified class to manage and dispatch all contribution
    evaluation methods based on the AppConfig.
    """

    def __init__(
            self,
            cfg: AppConfig,
            aggregator: Aggregator,
            buyer_root_loader: DataLoader
    ):
        self.cfg = cfg
        self.val_cfg = cfg.valuation

        logging.info("Initializing ValuationManager...")

        # --- 1. Initialize the DEFAULT (cheap) evaluator ---
        self.proxy_evaluator = ContributionEvaluator(cfg)
        logging.info("-> Default proxy evaluator (Similarity, Norm, Price) is ON.")

        if not self.val_cfg.run_similarity:
            logging.warning(
                "Config 'valuation.run_similarity' is set to False, "
                "but this default evaluator will always run. Ignoring flag."
            )

        # --- 2. Initialize the OPTIONAL (Influence) Evaluator ---
        if self.val_cfg.run_influence:
            self.influence_evaluator = InfluenceEvaluator(
                buyer_root_loader=buyer_root_loader,
                device=cfg.experiment.device,
                learning_rate=cfg.training.learning_rate
            )
            logging.info("-> Optional Influence evaluator is ON.")
        else:
            self.influence_evaluator = None
        # 3. Initialize the SLOW (LOO) Evaluator
        if self.val_cfg.run_loo:
            self.loo_evaluator = RoundBasedLOOEvaluator(
                aggregator_object=aggregator,
                buyer_root_loader=buyer_root_loader,
                device=cfg.experiment.device
            )
            logging.info(f"-> Slow LOO evaluator (RoundBasedLOOEvaluator) is ON (Freq: {self.val_cfg.loo_frequency}).")
        else:
            self.loo_evaluator = None
        # --- 4. NEW: Initialize Optional KernelSHAP Evaluator ---
        # E4: Banzhaf / Least Core reuse the same coalition sampler, so any of
        # the three flags activates this evaluator; the flags control which
        # scores it emits. Sampling cost is paid once regardless of how many
        # of the three are enabled.
        run_banzhaf = getattr(self.val_cfg, 'run_banzhaf', False)
        run_leastcore = getattr(self.val_cfg, 'run_leastcore', False)
        if self.val_cfg.run_kernelshap or run_banzhaf or run_leastcore:
            self.kernelshap_evaluator = KernelSHAPEvaluator(
                aggregator_object=aggregator,
                buyer_root_loader=buyer_root_loader,
                device=cfg.experiment.device,
                num_samples=self.val_cfg.kernelshap_samples,
                compute_kernelshap=self.val_cfg.run_kernelshap,
                compute_banzhaf=run_banzhaf,
                compute_leastcore=run_leastcore,
                leastcore_max_coalitions=getattr(self.val_cfg, 'leastcore_max_coalitions', 2000),
                leastcore_timeout=getattr(self.val_cfg, 'leastcore_timeout', 600.0),
            )
            logging.info(
                f"-> Coalition evaluator is ON (Freq: {self.val_cfg.kernelshap_frequency}, "
                f"Samples: {self.val_cfg.kernelshap_samples}, "
                f"kernelshap={self.val_cfg.run_kernelshap}, banzhaf={run_banzhaf}, "
                f"leastcore={run_leastcore}).")
        else:
            self.kernelshap_evaluator = None

    def evaluate_round(
            self,
            round_number: int,
            current_global_model: torch.nn.Module,
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
        The single entry point for all valuation.
        It runs all configured evaluators and merges their results.
        """

        # These will hold the merged results
        final_seller_valuations = {sid: {} for sid in seller_gradients.keys()}
        final_aggregate_metrics = {}

        # --- Run Proxy Evaluator (Similarity, Norm, Price) ---
        if self.proxy_evaluator:
            logging.debug("Running Proxy Evaluator...")
            s_vals, a_metrics = self.proxy_evaluator.evaluate_round(
                round_number, seller_gradients, seller_stats,
                oracle_gradient, buyer_gradient, aggregated_gradient,
                aggregation_stats, selected_ids, outlier_ids
            )
            # Merge results
            final_aggregate_metrics.update(a_metrics)
            for sid, scores in s_vals.items():
                final_seller_valuations[sid].update(scores)

        # --- Run Fast Influence Evaluator ---
        if self.influence_evaluator:
            logging.debug("Running Influence Evaluator...")
            influence_scores = self.influence_evaluator.evaluate_round(
                current_global_model, seller_gradients
            )
            # Merge results
            for sid, scores in influence_scores.items():
                if sid in final_seller_valuations:
                    final_seller_valuations[sid].update(scores)

        # --- Run Slow LOO Evaluator (Periodically) ---
        if self.loo_evaluator and (round_number % self.val_cfg.loo_frequency == 0):
            try:
                logging.info(f"Running periodic LOO evaluation for round {round_number}...")
                loo_scores = self.loo_evaluator.evaluate_round(
                    round_number,
                    current_global_model,
                    seller_gradients,
                    buyer_gradient=buyer_gradient
                )
                # Merge results
                for sid, scores in loo_scores.items():
                    if sid in final_seller_valuations:
                        final_seller_valuations[sid].update(scores)
            except Exception as e:
                logging.error(f"LOO evaluator failed in round {round_number}: {e}", exc_info=True)
                # Continue without LOO scores
        if self.kernelshap_evaluator and (round_number > 0 and round_number % self.val_cfg.kernelshap_frequency == 0):
            try:  # <-- RECOMMEND ADDING THIS
                logging.info(f"Running periodic KernelSHAP evaluation for round {round_number}...")
                kernelshap_scores = self.kernelshap_evaluator.evaluate_round(
                    round_number,
                    current_global_model,
                    seller_gradients,
                    buyer_gradient=buyer_gradient  # <-- THIS IS THE FIX
                )
                # Merge results
                for sid, scores in kernelshap_scores.items():
                    if sid in final_seller_valuations:
                        final_seller_valuations[sid].update(scores)
            except Exception as e: # <-- RECOMMEND ADDING THIS
                logging.error(f"KernelSHAP evaluator failed in round {round_number}: {e}", exc_info=True)
        return final_seller_valuations, final_aggregate_metrics
