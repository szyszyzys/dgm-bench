import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from src.common_utils.constants.enums import ServerAttackMode
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig
from experiments.gradient_market.privacy_attack import GradientInversionAttacker
from src.marketplace.market.data_market import DataMarketplace
from src.mechanism.gradient.aggregator import Aggregator
from src.mechanism.gradient.valuation.valuation import ValuationManager
from src.participants.seller.gradient_seller import GradientSeller, SybilCoordinator
from src.participants.seller.seller import BaseSeller


class DataMarketplaceFederated(DataMarketplace):
    def __init__(self, cfg: AppConfig, aggregator: Aggregator, sellers: dict,
                 input_shape: tuple, SellerClass: type, validation_loader, model_factory, num_classes: int,
                 attacker=None, buyer_seller: GradientSeller = None, oracle_seller: GradientSeller = None,
                 sybil_coordinator: Optional[SybilCoordinator] = None):
        """
        Initializes the marketplace with all necessary components and the main config.
        """
        self.cfg = cfg  # Store the main config object
        self.aggregator = aggregator
        self.sellers = sellers
        self.attacker = attacker  # For server-side privacy attacks
        self.consecutive_failed_rounds = 0
        self.SellerClass = SellerClass  # <-- Add this
        self.model_factory = model_factory
        self.validation_loader = validation_loader
        self.buyer_seller: GradientSeller = buyer_seller
        self.oracle_seller: GradientSeller = oracle_seller
        self.num_classes = num_classes
        self.sybil_coordinator = sybil_coordinator  # ✅ Store at marketplace level
        self.valuation_manager = None
        # Conditionally initialize the privacy attacker
        if self.cfg.server_attack_config.attack_name == ServerAttackMode.GRADIENT_INVERSION:
            # 2. Use the correct variable name: 'self.cfg' not 'self.config'
            self.attacker = GradientInversionAttacker(
                attack_config=self.cfg.server_attack_config,
                model_template=aggregator.strategy.global_model,
                device=aggregator.device,
                # 3. Use the correct path for the save directory
                save_dir=self.cfg.experiment.save_path,

                # 4. Use the correct paths for other experiment parameters
                dataset_name=self.cfg.experiment.dataset_name,
                input_shape=input_shape,  # Passed in, as it's determined at runtime
                num_classes=self.cfg.experiment.num_classes
            )
            logging.info("Initialized GradientInversionAttacker.")

    def register_seller(self, seller_id: str, seller: BaseSeller):
        """Registers a new seller in the marketplace."""
        self.sellers[seller_id] = seller
        logging.info(f"Marketplace: Registered seller '{seller_id}'. Total sellers: {len(self.sellers)}")

    @property
    def global_model(self) -> torch.nn.Module:
        """Provides convenient, direct access to the global model."""
        return self.aggregator.strategy.global_model

    def train_federated_round(self, round_number, global_model, validation_loader, ground_truth_dict):
        """
        Training with sybil coordination at marketplace level.

        Flow:
        1. Collect honest gradients from all sellers
        2. Sybil coordinator manipulates gradients based on historical patterns
        3. Aggregator selects and aggregates (with potentially manipulated gradients)
        4. Update sybil coordinator state for future rounds
        """
        round_start_time = time.time()
        logging.info(f"{'=' * 60}")
        logging.info(f"Round {round_number} Started")
        logging.info(f"{'=' * 60}")

        # ===== PHASE 1: Prepare Sybil Coordinator =====
        if self.sybil_coordinator:
            self.sybil_coordinator.prepare_for_new_round()
            logging.info(f"Sybil coordinator: Round {self.sybil_coordinator.cur_round}, "
                         f"Attack active: {self.sybil_coordinator.start_atk}")

        # ===== PHASE 2: Collect Honest Gradients =====
        logging.info("\n📦 Collecting honest gradients from all sellers...")
        gradients_dict, seller_ids, seller_stats_list = self._get_current_market_gradients()
        logging.info(f"🔎 [CP1] Collection: {len(gradients_dict)} gradients collected.")
        logging.debug(f"      IDs: {list(gradients_dict.keys())}")

        if not gradients_dict:
            logging.error("❌ No gradients collected! Cannot proceed with round.")
            return self._create_failed_round_record(round_number, round_start_time), None

        logging.info("\n" + "=" * 60)
        logging.info("🔬 GRADIENT QUALITY CHECK")
        logging.info("=" * 60)

        for seller_id, grad in gradients_dict.items():
            if grad is None:
                continue

            # Check gradient statistics
            grad_norms = [torch.norm(g).item() for g in grad]
            avg_norm = sum(grad_norms) / len(grad_norms)
            max_norm = max(grad_norms)
            min_norm = min(grad_norms)

            # Check for problems
            has_nan = any(torch.isnan(g).any() for g in grad)
            has_inf = any(torch.isinf(g).any() for g in grad)
            is_all_zero = all(torch.allclose(g, torch.zeros_like(g)) for g in grad)

            logging.info(f"\nSeller: {seller_id}")
            logging.info(f"  Avg norm: {avg_norm:.6e}")
            logging.info(f"  Max norm: {max_norm:.6e}")
            logging.info(f"  Min norm: {min_norm:.6e}")
            logging.info(f"  Has NaN: {has_nan}")
            logging.info(f"  Has Inf: {has_inf}")
            logging.info(f"  All zeros: {is_all_zero} ⚠️" if is_all_zero else f"  All zeros: {is_all_zero}")

            if is_all_zero:
                logging.error(f"  ❌ PROBLEM: {seller_id} has zero gradients!")
            if avg_norm < 1e-10:
                logging.warning(f"  ⚠️  WARNING: {seller_id} has very small gradients (might not contribute)")
            if avg_norm > 1e3:
                logging.warning(f"  ⚠️  WARNING: {seller_id} has very large gradients (might dominate)")

        logging.info(f"\n📊 Training Configuration:")
        logging.info(f"  Aggregator learning rate: {getattr(self.aggregator, 'learning_rate', 'N/A')}")
        logging.info(f"  Seller learning rate: {self.cfg.training.learning_rate}")
        logging.info(f"  Local epochs: {self.cfg.training.local_epochs}")
        logging.info(f"  Batch size: {self.cfg.training.batch_size}")

        # ===== PHASE 3: Compute Root Gradients =====
        logging.info("\n🛒 Computing buyer root gradient...")
        keys_before_buyer = set(gradients_dict.keys())

        # REMEMBER: Ensure you used .copy() here!
        buyer_root_gradient, buyer_stats = self.buyer_seller.get_gradient_for_upload(
            all_seller_gradients=gradients_dict.copy(),  # <--- VERIFY THIS
            target_seller_id=getattr(self.cfg.buyer_attack_config, 'target_seller_id', None)
        )

        # [CHECKPOINT 2] After Buyer
        keys_after_buyer = set(gradients_dict.keys())
        if len(keys_after_buyer) < len(keys_before_buyer):
            missing = keys_before_buyer - keys_after_buyer
            logging.error(f"🚨 [CP2 ALARM] The Buyer deleted these sellers: {missing}")
            logging.error("      You forgot .copy() in get_gradient_for_upload!")
        else:
            logging.info(f"✅ [CP2] Integrity Check: All {len(gradients_dict)} sellers survived the Buyer step.")

        if buyer_root_gradient is None:
            logging.error("Virtual buyer failed to compute gradient!")
            raise RuntimeError("Buyer root gradient is None")

        logging.info("🎯 Computing oracle root gradient...")
        oracle_root_gradient, _ = self.oracle_seller.get_gradient_for_upload(global_model)

        # Log buyer attack info
        if self.cfg.buyer_attack_config.is_active:
            attack_type = self.cfg.buyer_attack_config.attack_type
            logging.info(f"🚨 Active Buyer Attack: {attack_type}")
            if buyer_stats:
                if attack_type == "oscillating":
                    phase = buyer_stats.get('oscillation_phase', 'unknown')
                    classes = buyer_stats.get('target_classes', [])
                    logging.debug(f"   Phase: {phase}, Classes: {classes}")
                elif attack_type == "class_exclusion":
                    ratio = buyer_stats.get('exclusion_ratio', 0)
                    logging.debug(f"   Exclusion ratio: {ratio:.2%}")
        current_root_gradient_for_sybil = buyer_root_gradient
        # Start with the original gradients collected earlier (assuming gradients_dict holds them)
        final_gradients_to_aggregate = gradients_dict

        # ===== PHASE 4: Sybil Manipulation (IF ACTIVE) =====
        sybil_failed = False
        if hasattr(self,
                   'sybil_coordinator') and self.sybil_coordinator is not None and self.sybil_coordinator.sybil_cfg.is_sybil:
            logging.info("\n🐍 Applying Sybil Manipulation...")
            try:
                # Pass the original gradients, all sellers, and the CURRENT UNSANITIZED root gradient
                final_gradients_to_aggregate = self.sybil_coordinator.apply_manipulation(
                    current_round_gradients=gradients_dict,  # Pass original gradients collected earlier
                    all_sellers=self.sellers,
                    current_root_gradient=current_root_gradient_for_sybil,  # Use the unsanitized root gradient
                    global_epoch=round_number,  # Pass current round number
                    buyer_data_loader=self.aggregator.buyer_data_loader  # Pass if needed by hypothetical aggregate
                )
                logging.info("🐍 Sybil Manipulation complete.\n")
                logging.info(f"🔎 [CP3] Post-Sybil: {len(final_gradients_to_aggregate)} gradients remaining.")
            except Exception as e:
                logging.error(f"❌ Error during Sybil Manipulation: {e}. Using original gradients.", exc_info=True)
                final_gradients_to_aggregate = gradients_dict  # Fallback to original if manipulation fails
                sybil_failed = True

        else:
            logging.info("Sybil Coordinator not active or not present. Using original gradients.")

        # ===== PHASE 5: Sanitize Gradients =====
        # NOW sanitize the potentially manipulated gradients and the root gradients
        logging.info("\n🧹 Sanitizing gradients...")
        target_device = self.aggregator.device
        param_meta = [(p.shape, p.dtype) for p in self.global_model.parameters()]

        # Sanitize the final set of gradients that will go to the real aggregation
        sanitized_gradients = self._sanitize_gradients(final_gradients_to_aggregate, param_meta, target_device)
        logging.info(f"🔎 [CP4] Post-Sanitization: {len(sanitized_gradients)} gradients ready for Aggregator.")

        # Free intermediate gradient dicts to reduce VRAM pressure
        if final_gradients_to_aggregate is not gradients_dict:
            del final_gradients_to_aggregate
        del gradients_dict
        torch.cuda.empty_cache()

        # Sanitize the root gradients separately (needed for the real aggregation)
        sanitized_root_gradient = self._sanitize_gradient(buyer_root_gradient, target_device)
        sanitized_oracle_gradient = self._sanitize_gradient(oracle_root_gradient,
                                                            target_device)  # If oracle root is different
        # ===== PHASE 6: Privacy Attack (Optional) =====
        attack_log = None
        if self.attacker and self.attacker.should_run(round_number):
            logging.info("\n🔍 Executing privacy attack...")
            # ---
            # FIX: Pass the post-manipulation gradients to the attacker
            # ---
            attack_log = self.attacker.execute(round_number, final_gradients_to_aggregate, seller_ids,
                                               ground_truth_dict)
        # ===== PHASE 7: Aggregation =====
        logging.info("\n📊 Starting aggregation and selection...")

        # Log model state before aggregation
        param_before = list(self.global_model.parameters())[0].data.clone()
        norm_before = torch.norm(param_before).item()
        mean_before = param_before.mean().item()
        logging.info(f"PRE-APPLY Global Param[0]: Norm={norm_before:.4e}, Mean={mean_before:.4e}")

        accuracy_before = 0
        eval_batches = getattr(self.cfg.experiment, 'eval_batches', 10)
        if round_number % 5 == 0:  # Every 5 rounds
            logging.info("\n" + "=" * 60)
            logging.info(f"🔍 PRE-AGGREGATION EVALUATION (Round {round_number})")
            logging.info("=" * 60)

            if self.cfg.experiment.dataset_type == "llm":
                from src.model.llm_utils import evaluate_llm_perplexity
                ppl = evaluate_llm_perplexity(
                    self.global_model, validation_loader, self.aggregator.device
                )
                logging.info(f"Perplexity BEFORE aggregation: {ppl:.2f}")
            else:
                self.global_model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_idx, batch in enumerate(validation_loader):
                        if batch_idx >= eval_batches:
                            break

                        if len(batch) == 3:  # Text
                            labels, data, _ = batch
                        else:  # Image/Tabular
                            data, labels = batch

                        data, labels = data.to(self.aggregator.device), labels.to(self.aggregator.device)
                        outputs = self.global_model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy_before = 100 * correct / total if total > 0 else 0
                logging.info(f"Accuracy BEFORE aggregation: {accuracy_before:.2f}%")

        # --- Instrument aggregation: timing + peak VRAM ---
        _agg_device = self.aggregator.device
        _has_cuda = torch.cuda.is_available() and str(_agg_device).startswith("cuda")
        if _has_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(_agg_device))
            torch.cuda.synchronize(torch.device(_agg_device))
        _agg_t0 = time.time()

        agg_grad, selected_ids, outlier_ids, aggregation_stats = self.aggregator.aggregate(
            global_epoch=round_number,
            seller_updates=sanitized_gradients,
            root_gradient=sanitized_root_gradient,
            buyer_data_loader=self.aggregator.buyer_data_loader
        )

        if _has_cuda:
            torch.cuda.synchronize(torch.device(_agg_device))
        _agg_duration = time.time() - _agg_t0

        # Inject system metrics into aggregation_stats so they flow into the round record
        if aggregation_stats is None:
            aggregation_stats = {}
        aggregation_stats['aggregation_latency_sec'] = _agg_duration
        if _has_cuda:
            _dev = torch.device(_agg_device)
            aggregation_stats['aggregation_peak_vram_mb'] = (
                torch.cuda.max_memory_allocated(_dev) / (1024 * 1024)
            )

        logging.info(f"✅ Selected {len(selected_ids)} sellers: {selected_ids}")
        logging.info(f"   Rejected {len(outlier_ids)} sellers as outliers")
        logging.info(f"   Aggregation latency: {_agg_duration:.3f}s")

        # ===== PHASE 8: Compute Metrics & Save Valuation =====
        logging.info("\n📈 Computing marketplace metrics & valuation...")

        seller_stats_dict = {sid: stats for sid, stats in zip(seller_ids, seller_stats_list)}
        if not self.valuation_manager:
            self.valuation_manager = ValuationManager(
                cfg=self.cfg,
                aggregator=self.aggregator,
                buyer_root_loader=self.buyer_seller.train_loader
            )

        seller_valuations, aggregate_metrics = self.valuation_manager.evaluate_round(
            round_number=round_number,
            current_global_model=self.global_model,
            seller_gradients=sanitized_gradients,
            seller_stats=seller_stats_dict,
            oracle_gradient=sanitized_oracle_gradient,
            buyer_gradient=sanitized_root_gradient,
            aggregated_gradient=agg_grad,
            aggregation_stats=aggregation_stats,
            selected_ids=selected_ids,
            outlier_ids=outlier_ids
        )

        # --- NEW: SAVE DETAILED VALUATION TO SEPARATE FILE ---
        # This ensures high-dimensional data (LOO, Shapley) is saved without breaking the CSV
        if self.cfg.experiment.save_path:
            try:
                val_save_path = Path(self.cfg.experiment.save_path) / "valuations.jsonl"

                # Only save if we actually have data (optimization)
                has_data = any(vals for vals in seller_valuations.values())

                if has_data:
                    val_entry = {
                        "round": round_number,
                        "timestamp": time.time(),
                        # --- ADD THESE TWO LINES ---
                        "selected_ids": selected_ids,
                        "outlier_ids": outlier_ids,
                        # ---------------------------
                        "seller_valuations": seller_valuations
                    }

                    # Append to JSONL file
                    with open(val_save_path, "a") as f:
                        f.write(json.dumps(val_entry) + "\n")
                    logging.info(f"   💾 Saved detailed valuations to {val_save_path.name}")
            except Exception as e:
                logging.error(f"   ❌ Failed to save valuations.jsonl: {e}")

        # ===== PHASE 9: Apply Update =====
        if agg_grad:
            try:
                self.aggregator.apply_gradient(agg_grad)
                self.consecutive_failed_rounds = 0
                logging.info("✅ Aggregated gradient applied to global model")

                # --- Free-rider support: propagate the just-aggregated delta and
                # the median accepted L2 norm to any LazyGradientSeller so the
                # "echo" / "norm_noise" strategies can use them next round.
                try:
                    from src.participants.seller.gradient_seller import LazyGradientSeller
                    # Compute median L2 norm of *accepted* gradients (per-seller flat norm)
                    accepted_norms = []
                    for sid in selected_ids:
                        g = sanitized_gradients.get(sid)
                        if g is None:
                            continue
                        n = float(torch.sqrt(sum(torch.sum(t.detach() ** 2) for t in g)).item())
                        accepted_norms.append(n)
                    median_norm = float(sorted(accepted_norms)[len(accepted_norms) // 2]) if accepted_norms else None

                    # Detach + clone the aggregated delta for safe replay next round
                    prev_delta_clone = [t.detach().clone() for t in agg_grad]
                    n_propagated = 0
                    for _sid, _seller in self.sellers.items():
                        if isinstance(_seller, LazyGradientSeller):
                            _seller.previous_global_delta = prev_delta_clone
                            _seller.previous_median_norm = median_norm
                            n_propagated += 1
                    if n_propagated > 0:
                        logging.info(
                            f"   🪤 Propagated previous delta + median_norm={median_norm} "
                            f"to {n_propagated} LazyGradientSeller(s)"
                        )
                except Exception as _e:
                    logging.warning(f"   Lazy seller propagation skipped: {_e}")

                # Log model state after aggregation
                param_after = list(self.global_model.parameters())[0].data
                norm_after = torch.norm(param_after).item()
                mean_after = param_after.mean().item()
                logging.info(f"POST-APPLY Global Param[0]: Norm={norm_after:.4e}, Mean={mean_after:.4e}")

                if abs(norm_before - norm_after) > 1e-7:
                    logging.info("   ✅ Global model parameters changed")
                else:
                    logging.warning("   ⚠️  Global model parameters did NOT change significantly!")

                if agg_grad and round_number % 5 == 0:
                    logging.info("\n" + "=" * 60)
                    logging.info(f"🔍 POST-AGGREGATION EVALUATION (Round {round_number})")
                    logging.info("=" * 60)

                    is_llm = self.cfg.experiment.dataset_type == "llm"

                    if is_llm:
                        # LLM: compute perplexity instead of accuracy
                        from src.model.llm_utils import evaluate_llm_perplexity
                        ppl = evaluate_llm_perplexity(
                            self.global_model, validation_loader, self.aggregator.device
                        )
                        logging.info(f"Perplexity AFTER aggregation: {ppl:.2f}")
                        accuracy_after = 0  # Not applicable for LLM
                    else:
                        self.global_model.eval()
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for batch_idx, batch in enumerate(validation_loader):
                                if batch_idx >= eval_batches:
                                    break

                                if len(batch) == 3:
                                    labels, data, _ = batch
                                else:
                                    data, labels = batch

                                data, labels = data.to(self.aggregator.device), labels.to(self.aggregator.device)
                                outputs = self.global_model(data)
                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()

                        accuracy_after = 100 * correct / total if total > 0 else 0
                        logging.info(f"Accuracy AFTER aggregation: {accuracy_after:.2f}%")

                    # Check if accuracy improved
                    if round_number > 5 and not is_llm:
                        improvement = accuracy_after - accuracy_before
                        logging.info(f"Accuracy change this round: {improvement:+.2f}%")

            except Exception as e:
                logging.error(f"❌ Failed to apply aggregated gradient: {e}", exc_info=True)
                self.consecutive_failed_rounds += 1
        else:
            self.consecutive_failed_rounds += 1
            logging.warning(
                f"⚠️  Round failed (agg_grad is None). Consecutive failures: {self.consecutive_failed_rounds}")

        # ===== PHASE 10: Update Sybil Coordinator State =====
        if self.sybil_coordinator and not sybil_failed:
            logging.info("\n🔄 Updating sybil coordinator...")
            self.sybil_coordinator.update_post_selection(
                selected_ids=selected_ids,
                all_sellers=self.sellers
            )
        elif sybil_failed:
            logging.warning("⚠️  Skipping sybil coordinator update due to manipulation failure")

        if self.cfg.debug.save_individual_gradients:
            if round_number % self.cfg.debug.gradient_save_frequency == 0:
                self._save_round_gradients(round_number, sanitized_gradients, agg_grad)

        duration = time.time() - round_start_time
        round_record = self._create_round_record(
            round_number=round_number,
            duration=duration,
            seller_ids=seller_ids,
            selected_ids=selected_ids,
            outlier_ids=outlier_ids,
            buyer_stats=buyer_stats,
            attack_log=attack_log,
            aggregation_stats=aggregation_stats,
            seller_valuations=seller_valuations,
            aggregate_metrics=aggregate_metrics,
            seller_stats_dict=seller_stats_dict,
        )
        # ===== PHASE 11: Notify Sellers =====
        for sid, seller in self.sellers.items():
            seller_scores = seller_valuations.get(sid, {})
            seller.round_end_process(
                round_number,
                was_selected=(sid in selected_ids),
                was_outlier=(sid in outlier_ids),
                marketplace_metrics=seller_scores,
            )

        logging.info(f"\n{'=' * 60}")
        logging.info(f"Round {round_number} Ended (Duration: {duration:.2f}s)")
        logging.info(f"{'=' * 60}\n")

        return round_record, agg_grad

    def _sanitize_gradients(self, gradients_dict: Dict, param_meta: List, target_device) -> Dict:
        sanitized = {}
        for sid, grad_list in gradients_dict.items():

            # [CHECKPOINT 5a] Check for None
            if grad_list is None:
                logging.error(f"❌ [Sanitizer] Dropping {sid}: Gradient is None.")
                continue

            # [CHECKPOINT 5b] Check for List Length Mismatch
            if len(grad_list) != len(param_meta):
                logging.error(f"❌ [Sanitizer] Dropping {sid}: Parameter count mismatch.")
                logging.error(f"      Seller: {len(grad_list)}, Global Model: {len(param_meta)}")
                continue

            corrected_list = []
            valid_seller = True

            # [CHECKPOINT 5c] Check for Tensor Shape Mismatch
            for i, tensor in enumerate(grad_list):
                expected_shape, expected_dtype = param_meta[i]

                if tensor.shape != expected_shape:
                    logging.error(f"❌ [Sanitizer] Dropping {sid}: Shape mismatch at param index {i}.")
                    logging.error(f"      Got {tensor.shape}, Expected {expected_shape}")
                    valid_seller = False
                    break

                # Handle device transfer
                if tensor.device != target_device:
                    corrected_list.append(tensor.to(target_device))
                else:
                    corrected_list.append(tensor)

            if valid_seller:
                sanitized[sid] = corrected_list

        return sanitized

    def _sanitize_gradient(self, gradient: List[torch.Tensor], target_device) -> List[torch.Tensor]:
        """Sanitize a single gradient."""
        if gradient is None:
            return None
        return [
            (tensor.to(target_device) if tensor.device != target_device else tensor)
            for tensor in gradient
        ]

    def _create_round_record(
            self,
            round_number: int,
            duration: float,
            seller_ids: List[str],
            selected_ids: List[str],
            outlier_ids: List[str],
            buyer_stats: Dict,
            attack_log: Dict,
            aggregation_stats: Dict,
            seller_valuations: Dict[str, Dict],  # <-- UPDATED
            aggregate_metrics: Dict,  # <-- UPDATED
            seller_stats_dict: Dict[str, Dict] = None,
    ) -> Dict:
        """Create comprehensive round record."""
        round_record = {
            "round": round_number,
            "timestamp": time.time(),
            "duration_sec": duration,
            "num_total_sellers": len(seller_ids),
            "num_selected": len(selected_ids),
            "num_outliers": len(outlier_ids),

            # Buyer attack info
            "buyer_attack_active": self.cfg.buyer_attack_config.is_active,
            "buyer_attack_type": self.cfg.buyer_attack_config.attack_type if self.cfg.buyer_attack_config.is_active else "none",
            "buyer_attack_stats": buyer_stats,

            # Server attack info
            "attack_performed": bool(attack_log),
            "attack_victim": attack_log.get('victim_id') if attack_log else None,
            "attack_success": attack_log.get('success') if attack_log else None,
        }

        # --- 1. Automatic Aggregate Metrics ---
        # No more manual list. This adds all keys from aggregate_metrics.
        round_record.update(aggregate_metrics)

        # --- 2. Aggregation Stats ---
        if aggregation_stats:
            round_record['detailed_aggregation_stats'] = aggregation_stats
            # "Pull up" flat keys for the main CSV log
            for key, value in aggregation_stats.items():
                if not isinstance(value, (dict, list, np.ndarray)):
                    round_record[key] = value
        else:
            round_record['detailed_aggregation_stats'] = {}

        # --- 3. Automatic Per-Seller Metrics ---
        # No more manual keys. This saves all per-seller scores.
        seller_metrics_list = []
        for sid in seller_ids:
            # Get all scores (sim_to_oracle, price_paid, influence, etc.)
            seller_data = seller_valuations.get(sid, {}).copy()

            # Add logging metadata
            seller_data['round'] = round_number
            seller_data['seller_id'] = sid

            # (Note: 'selected' & 'outlier' are already set by the evaluator)
            seller_metrics_list.append(seller_data)

        round_record['detailed_seller_metrics'] = seller_metrics_list

        # --- 4. Cost of Convergence (CoC) metrics ---
        if seller_stats_dict:
            # Total transmission cost (all sellers uploaded, regardless of acceptance)
            round_upload_bytes_total = sum(
                s.get('upload_bytes', 0) for s in seller_stats_dict.values()
            )
            # Useful transmission cost (only accepted sellers)
            round_upload_bytes_accepted = sum(
                seller_stats_dict[sid].get('upload_bytes', 0)
                for sid in selected_ids if sid in seller_stats_dict
            )

            round_record['round_upload_bytes'] = round_upload_bytes_total
            round_record['round_upload_mb'] = round_upload_bytes_total / (1024 * 1024)
            round_record['round_upload_bytes_accepted'] = round_upload_bytes_accepted
            round_record['round_upload_mb_accepted'] = round_upload_bytes_accepted / (1024 * 1024)
            round_record['num_uploads'] = len(seller_stats_dict)
            round_record['num_uploads_accepted'] = len(selected_ids)

            # Track cumulative CoC (initialized on first round)
            if not hasattr(self, '_cumulative_upload_bytes'):
                self._cumulative_upload_bytes = 0
                self._cumulative_upload_bytes_accepted = 0
            self._cumulative_upload_bytes += round_upload_bytes_total
            self._cumulative_upload_bytes_accepted += round_upload_bytes_accepted
            round_record['cumulative_upload_bytes'] = self._cumulative_upload_bytes
            round_record['cumulative_upload_mb'] = self._cumulative_upload_bytes / (1024 * 1024)
            round_record['cumulative_upload_bytes_accepted'] = self._cumulative_upload_bytes_accepted
            round_record['cumulative_upload_mb_accepted'] = self._cumulative_upload_bytes_accepted / (1024 * 1024)

        return round_record

    def _create_failed_round_record(self, round_number, round_start_time) -> Dict:
        """Create a record for a failed round."""
        return {
            "round": round_number,
            "timestamp": time.time(),
            "duration_sec": time.time() - round_start_time,
            "failed": True,
            "num_total_sellers": 0,
            "num_selected": 0,
            "num_outliers": 0,
        }

    def _save_round_gradients(self, round_number: int, gradients_dict: Dict, agg_grad: List[torch.Tensor]):
        """
        Save individual seller gradients and aggregated gradient for debugging/analysis.
        This is the implementation of the method called in train_federated_round.
        """
        grad_save_dir = Path(self.cfg.experiment.save_path) / "individual_gradients" / f"round_{round_number}"
        grad_save_dir.mkdir(parents=True, exist_ok=True)

        # Save individual seller gradients
        saved_count = 0
        for sid, grad in gradients_dict.items():
            if grad is not None and len(grad) > 0:
                try:
                    torch.save(grad, grad_save_dir / f"{sid}_grad.pt")
                    saved_count += 1
                except Exception as e:
                    logging.error(f"Failed to save gradient for {sid}: {e}")

        logging.info(f"Saved {saved_count}/{len(gradients_dict)} individual gradients to {grad_save_dir}")

        # Save aggregated gradient
        if agg_grad is not None and len(agg_grad) > 0:
            try:
                torch.save(agg_grad, grad_save_dir / "aggregated_grad.pt")
                logging.info(f"Saved aggregated gradient")
            except Exception as e:
                logging.error(f"Failed to save aggregated gradient: {e}")

    def _compute_gradient_similarities(self, gradients_dict: Dict) -> List[float]:
        """Compute pairwise cosine similarities between gradients."""
        seller_ids = list(gradients_dict.keys())
        similarities = []

        for i, sid1 in enumerate(seller_ids):
            for sid2 in seller_ids[i + 1:]:
                grad1 = gradients_dict[sid1]
                grad2 = gradients_dict[sid2]

                if grad1 is None or grad2 is None:
                    continue

                # Flatten and compute cosine similarity
                flat1 = torch.cat([g.flatten() for g in grad1])
                flat2 = torch.cat([g.flatten() for g in grad2])

                similarity = torch.nn.functional.cosine_similarity(
                    flat1.unsqueeze(0),
                    flat2.unsqueeze(0)
                ).item()
                similarities.append(similarity)

        return similarities

    def _get_current_market_gradients(self) -> Tuple[Dict, List, List]:
        """Collects gradients and stats from all sellers with detailed debugging."""
        logging.info("=" * 60)
        logging.info("📦 Collecting gradients from all sellers...")
        logging.info("=" * 60)

        gradients_dict = OrderedDict()
        seller_ids, seller_stats_list = [], []

        # Get expected parameters from global model
        global_params = list(self.aggregator.strategy.global_model.parameters())
        expected_param_count = len(global_params)

        logging.info(f"Global model info:")
        logging.info(f"  - Expected param count: {expected_param_count}")
        logging.info(f"  - First param shape: {global_params[0].shape if global_params else 'N/A'}")
        logging.info(f"  - Device: {global_params[0].device if global_params else 'N/A'}")
        logging.info(f"\nProcessing {len(self.sellers)} sellers...")

        success_count = 0
        fail_count = 0

        for sid, seller in self.sellers.items():
            logging.info(f"\n{'─' * 60}")
            logging.info(f"🔍 Seller: {sid}")

            # Check seller state
            try:
                is_active = getattr(seller, 'is_active', True)
                has_model = hasattr(seller, 'model') and seller.model is not None

                logging.info(f"  Status:")
                logging.info(f"    - Active: {is_active}")
                logging.info(f"    - Has model: {has_model}")

                if has_model:
                    seller_param_count = sum(1 for _ in seller.model.parameters())
                    logging.info(f"    - Seller model param count: {seller_param_count}")

                if not is_active:
                    logging.warning(f"  ⚠️  Seller {sid} is inactive, skipping")
                    fail_count += 1
                    continue

            except Exception as e:
                logging.error(f"  ❌ Error checking seller state: {e}")
                fail_count += 1
                continue

            # Try to get gradient
            try:
                logging.info(f"  Calling get_gradient_for_upload()...")
                grad, stats = seller.get_gradient_for_upload()

                # Detailed gradient inspection
                logging.info(f"  Gradient inspection:")
                logging.info(f"    - Returned grad is None: {grad is None}")
                logging.info(f"    - Returned stats: {stats}")

                if grad is None:
                    logging.warning(f"  ⚠️  Seller {sid} returned None gradient")
                    fail_count += 1
                    continue

                # Check gradient type
                logging.info(f"    - Gradient type: {type(grad)}")

                if not isinstance(grad, (list, tuple)):
                    logging.error(f"  ❌ Gradient is not a list/tuple, got {type(grad)}")
                    fail_count += 1
                    continue

                # Check gradient length
                grad_length = len(grad)
                logging.info(f"    - Gradient length: {grad_length}")
                logging.info(f"    - Expected length: {expected_param_count}")

                if grad_length == 0:
                    logging.error(f"  ❌ Gradient is empty list")
                    fail_count += 1
                    continue

                if grad_length != expected_param_count:
                    logging.error(f"  ❌ LENGTH MISMATCH: got {grad_length}, expected {expected_param_count}")
                    fail_count += 1
                    continue

                # Check gradient content
                logging.info(f"    - First element type: {type(grad[0])}")

                if hasattr(grad[0], 'shape'):
                    logging.info(f"    - First element shape: {grad[0].shape}")
                    logging.info(f"    - Expected first shape: {global_params[0].shape}")

                    if grad[0].shape != global_params[0].shape:
                        logging.error(f"  ❌ SHAPE MISMATCH at first parameter!")
                        fail_count += 1
                        continue

                if isinstance(grad[0], torch.Tensor):
                    logging.info(f"    - First element dtype: {grad[0].dtype}")
                    logging.info(f"    - First element device: {grad[0].device}")
                    logging.info(f"    - First element has NaN: {torch.isnan(grad[0]).any().item()}")
                    logging.info(f"    - First element has Inf: {torch.isinf(grad[0]).any().item()}")
                    logging.info(f"    - First element mean: {grad[0].mean().item():.6e}")
                    logging.info(f"    - First element std: {grad[0].std().item():.6e}")

                    # Check if gradient is all zeros
                    is_zero = all(torch.allclose(g, torch.zeros_like(g)) for g in grad[:3])  # Check first 3
                    if is_zero:
                        logging.warning(f"  ⚠️  Warning: First few gradients appear to be all zeros")

                # Validate all parameters
                all_valid = True
                for i, g in enumerate(grad):
                    if not isinstance(g, (torch.Tensor, np.ndarray)):
                        logging.error(f"  ❌ Parameter {i} is not a tensor/array: {type(g)}")
                        all_valid = False
                        break

                    if hasattr(g, 'shape') and g.shape != global_params[i].shape:
                        logging.error(f"  ❌ Shape mismatch at parameter {i}: {g.shape} vs {global_params[i].shape}")
                        all_valid = False
                        break

                if not all_valid:
                    logging.error(f"  ❌ Gradient validation failed")
                    fail_count += 1
                    continue

                # Success!
                gradients_dict[sid] = grad
                seller_ids.append(sid)
                seller_stats_list.append(stats)
                success_count += 1

                logging.info(f"  ✅ Gradient accepted from seller {sid}")

            except AttributeError as e:
                logging.error(f"  ❌ AttributeError (missing method?): {e}")
                logging.error(f"     Seller methods: {[m for m in dir(seller) if not m.startswith('_')]}")
                fail_count += 1

            except Exception as e:
                logging.error(f"  ❌ Exception getting gradient from seller {sid}: {e}", exc_info=True)
                fail_count += 1

        # Summary
        logging.info("\n" + "=" * 60)
        logging.info("📊 Gradient Collection Summary:")
        logging.info(f"  - Total sellers: {len(self.sellers)}")
        logging.info(f"  - Successful: {success_count} ✅")
        logging.info(f"  - Failed: {fail_count} ❌")
        logging.info(f"  - Success rate: {success_count / len(self.sellers) * 100:.1f}%")
        logging.info(f"  - Collected seller IDs: {seller_ids}")
        logging.info("=" * 60)

        # Final validation
        if not gradients_dict:
            logging.error("⚠️⚠️⚠️  NO GRADIENTS COLLECTED! ⚠️⚠️⚠️")
            logging.error("This will cause aggregation to fail!")
            logging.error("\nPossible causes:")
            logging.error("  1. All sellers returned None")
            logging.error("  2. Length/shape mismatches")
            logging.error("  3. Sellers not properly initialized")
            logging.error("  4. Training didn't happen")
            logging.error("\nCheck the detailed logs above for specific errors per seller.")

        return gradients_dict, seller_ids, seller_stats_list
