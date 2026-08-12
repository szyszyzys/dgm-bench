# Add these imports at the top of your valuation.py file
import copy
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from src.mechanism.gradient.aggregator import Aggregator


class KernelSHAPEvaluator:
    """
    Approximates Shapley values using the KernelSHAP method
    by fitting a weighted linear model.

    This is an ONLINE, periodic, and EXPENSIVE method.
    """

    def __init__(self,
                 aggregator_object: Aggregator,
                 buyer_root_loader: DataLoader,
                 device: str,
                 num_samples: int,
                 compute_kernelshap: bool = True,
                 compute_banzhaf: bool = False,
                 compute_leastcore: bool = False,
                 leastcore_max_coalitions: int = 2000,
                 leastcore_timeout: float = 600.0):
        self.aggregator = aggregator_object
        self.buyer_loader = buyer_root_loader
        self.device = device
        self.num_samples = num_samples
        # E4: Banzhaf / Least Core REUSE the coalition-utility pairs sampled
        # below — they add zero extra utility evaluations (the expensive part).
        self.compute_kernelshap = compute_kernelshap
        self.compute_banzhaf = compute_banzhaf
        self.compute_leastcore = compute_leastcore
        self.leastcore_max_coalitions = leastcore_max_coalitions
        self.leastcore_timeout = leastcore_timeout
        logging.info(
            f"KernelSHAPEvaluator initialized "
            f"(kernelshap={compute_kernelshap}, banzhaf={compute_banzhaf}, "
            f"leastcore={compute_leastcore})."
        )

    def _get_performance(self, model: torch.nn.Module) -> float:
        """Helper: Evaluates a model's accuracy on the buyer's data."""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the whole batch
            for batch in self.buyer_loader:

                try:
                    if len(batch) == 3:  # Text data
                        labels, data, _ = batch
                    else:  # Image/Tabular
                        data, labels = batch
                except Exception as e:
                    # Log a warning if you have logging imported, otherwise print
                    print(f"[WARN] KernelSHAP skipping eval batch due to unpack error: {e}")
                    continue

                data, labels = data.to(self.device), labels.to(self.device)

                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return (100 * correct / total) if total > 0 else 0.0

    def _get_performance_for_coalition(
            self,
            original_model: torch.nn.Module,
            all_gradients: Dict[str, List[torch.Tensor]],
            coalition: List[str],  # List of seller IDs in the coalition
            round_number: int,
            buyer_gradient: Optional[List[torch.Tensor]]  # ✅ --- 5. ACCEPT IT HERE ---
    ) -> float:
        """
        Simulates an aggregation and update for *only* the sellers
        in the coalition, then returns the resulting model performance.
        """
        if not coalition:
            # Value of the empty set is the performance *before* any update
            return self._get_performance(original_model)

        # 1. Filter gradients to only this coalition
        coalition_gradients = {
            sid: grad for sid, grad in all_gradients.items()
            if sid in coalition
        }

        # 2. Create a deep copy of the model
        temp_model = copy.deepcopy(original_model).to(self.device)

        # 3. Save and swap aggregator state to avoid corrupting stateful aggregators
        #    (e.g., RFLPA's _prev_global_gradient, MartFL's baseline_id).
        #    CRITICAL: keep the LIVE global_model object aside and restore that
        #    exact reference afterwards. Deep-copying global_model into the saved
        #    state (and restoring the copy) rebinds aggregator.strategy.global_model
        #    to a clone, desyncing it from the StatefulModelFactory the sellers
        #    load their starting weights from — after the first valuation pass the
        #    sellers train against a frozen stale model while the global model
        #    drifts, and training diverges. So we deep-copy every stateful field
        #    EXCEPT global_model, and restore the original object by reference.
        live_global_model = self.aggregator.strategy.global_model
        original_strategy_state = {
            k: copy.deepcopy(v)
            for k, v in self.aggregator.strategy.__dict__.items()
            if k != "global_model"
        }
        self.aggregator.strategy.global_model = temp_model  # Temporarily swap

        try:
            agg_grad, _, _, _ = self.aggregator.aggregate(
                global_epoch=round_number,
                seller_updates=coalition_gradients,
                root_gradient=buyer_gradient,
            )

            # 4. Simulate applying the gradient
            if agg_grad:
                try:
                    self.aggregator.apply_gradient(agg_grad)
                except Exception as e:
                    logging.error(f"KernelSHAP: Failed to apply temp gradient: {e}")

            # 5. Evaluate the temporary model's performance
            performance = self._get_performance(temp_model)
        finally:
            # 6. Restore the entire aggregator strategy state and cleanup.
            #    NOTE: empty_cache() here is critical for step 10. KernelSHAP
            #    deep-copies the full model per coalition (~500 coalitions per
            #    valuation pass × ~10 passes per scenario = ~5,000 clones per
            #    scenario). Without empty_cache() the CUDA caching allocator
            #    fragments and the dispatcher stalls after ~24h on CIFAR-100.
            self.aggregator.strategy.__dict__.update(original_strategy_state)
            # Restore the ORIGINAL global_model object (by reference), never a copy,
            # so the aggregator and the model factory keep pointing at the same module.
            self.aggregator.strategy.global_model = live_global_model
            del temp_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return performance

    def _get_kernelshap_weights(self, z_prime, num_sellers):
        """Calculates the Shapley kernel weight for a coalition size."""
        if z_prime == 0 or z_prime == num_sellers:
            return 1e9  # Effectively infinite weight

        from scipy.special import comb
        return (num_sellers - 1) / (comb(num_sellers, z_prime) * z_prime * (num_sellers - z_prime))

    def evaluate_round(
            self,
            round_number: int,
            current_global_model: torch.nn.Module,
            seller_gradients: Dict[str, List[torch.Tensor]],
            buyer_gradient: Optional[List[torch.Tensor]]  # ✅ --- 1. ACCEPT IT HERE ---
    ) -> Dict[str, Dict[str, Any]]:

        logging.info("Starting KernelSHAP (Linear Model) evaluation...")
        seller_ids = list(seller_gradients.keys())
        num_sellers = len(seller_ids)
        valuations = {sid: {} for sid in seller_ids}

        # 1. Create the dataset for the linear model (X, y, weights)
        X_coalitions = []  # Binary vectors (e.g., [1, 0, 1])
        y_performance = []  # Performance for that coalition
        sample_weights = []  # Shapley kernel weights

        # 2. Add the two required anchor coalitions
        # Coalition 1: Empty set
        X_coalitions.append(np.zeros(num_sellers))
        y_performance.append(self._get_performance_for_coalition(
            current_global_model, seller_gradients, [], round_number,
            buyer_gradient  # ✅ --- 2. PASS IT HERE ---
        ))
        sample_weights.append(self._get_kernelshap_weights(0, num_sellers))

        # Coalition 2: Grand coalition (all sellers)
        X_coalitions.append(np.ones(num_sellers))
        y_performance.append(self._get_performance_for_coalition(
            current_global_model, seller_gradients, seller_ids, round_number,
            buyer_gradient  # ✅ --- 3. PASS IT HERE ---
        ))
        sample_weights.append(self._get_kernelshap_weights(num_sellers, num_sellers))

        # 3. Sample 'M' random coalitions
        num_to_sample = max(0, self.num_samples - 2)  # Already added 2
        for _ in range(num_to_sample):
            # Create a random binary coalition vector
            z_prime_binary = np.random.randint(0, 2, num_sellers)
            z_prime_size = np.sum(z_prime_binary)

            # Convert binary vector to list of seller IDs
            coalition_sids = [
                sid for i, sid in enumerate(seller_ids)
                if z_prime_binary[i] == 1
            ]

            # Get performance and weight
            perf = self._get_performance_for_coalition(
                current_global_model, seller_gradients, coalition_sids, round_number,
                buyer_gradient  # ✅ --- 4. PASS IT HERE ---
            )
            weight = self._get_kernelshap_weights(z_prime_size, num_sellers)

            X_coalitions.append(z_prime_binary)
            y_performance.append(perf)
            sample_weights.append(weight)

        # 4. Fit the weighted linear model
        X = np.array(X_coalitions)
        y = np.array(y_performance)
        weights = np.array(sample_weights)

        if self.compute_kernelshap:
            try:
                model = LinearRegression()
                model.fit(X, y, sample_weight=weights)

                # The coefficients of the linear model ARE the Shapley values
                shapley_values = model.coef_

                for i, sid in enumerate(seller_ids):
                    valuations[sid]['kernelshap_score'] = shapley_values[i]

                logging.info(f"KernelSHAP scores: {shapley_values}")

            except Exception as e:
                logging.error(f"Failed to fit KernelSHAP linear model: {e}")

        # 5. E4a — Banzhaf value via the maximum-sample-reuse estimator,
        #    on the SAME coalition/utility samples (no new evaluations).
        if self.compute_banzhaf:
            try:
                banzhaf_values = self._banzhaf_from_samples(X, y)
                if banzhaf_values is not None:
                    for i, sid in enumerate(seller_ids):
                        valuations[sid]['banzhaf_score'] = banzhaf_values[i]
                    logging.info(f"Banzhaf scores: {banzhaf_values}")
            except Exception as e:
                logging.error(f"Failed to compute Banzhaf values: {e}")

        # 6. E4b — Least Core via Monte-Carlo constraint-sampled LP, again on
        #    the SAME samples. Never crashes the run: any failure logs
        #    'leastcore_intractable' and skips.
        if self.compute_leastcore:
            leastcore_values = self._leastcore_from_samples(X, y)
            if leastcore_values is not None:
                for i, sid in enumerate(seller_ids):
                    valuations[sid]['leastcore_score'] = leastcore_values[i]
                logging.info(f"Least-core payoffs: {leastcore_values}")

        return valuations

    def _banzhaf_from_samples(self, X: np.ndarray, y: np.ndarray):
        """Banzhaf value via maximum sample reuse (Wang & Jia, AISTATS 2023).

        The random coalitions sampled for KernelSHAP draw each seller's
        membership i.i.d. Bernoulli(0.5) — exactly the Banzhaf distribution —
        so each seller's Banzhaf value is estimated as

            banzhaf_i = mean(v(S) : i in S) - mean(v(S) : i not in S)

        over the reused samples. The two deterministic anchor coalitions
        (empty set, grand coalition; rows 0 and 1) are excluded so the
        estimator keeps its sampling distribution.
        """
        X_rand, y_rand = X[2:], y[2:]
        if len(y_rand) == 0:
            logging.warning("Banzhaf skipped: no random coalitions sampled "
                            "(kernelshap_samples <= 2).")
            return None

        num_sellers = X.shape[1]
        values = np.zeros(num_sellers)
        for i in range(num_sellers):
            in_mask = X_rand[:, i] == 1
            n_in, n_out = int(in_mask.sum()), int((~in_mask).sum())
            if n_in == 0 or n_out == 0:
                # Degenerate draw (possible only for tiny sample counts):
                # fall back to the anchors so both sides are populated.
                in_vals = np.concatenate([y_rand[in_mask], y[1:2]])
                out_vals = np.concatenate([y_rand[~in_mask], y[0:1]])
                values[i] = in_vals.mean() - out_vals.mean()
            else:
                values[i] = y_rand[in_mask].mean() - y_rand[~in_mask].mean()
        return values

    def _leastcore_from_samples(self, X: np.ndarray, y: np.ndarray):
        """Least Core payoffs from sampled coalition constraints.

        Utilities are normalized against the empty coalition (v(S) =
        u(S) - u(empty), so v(empty) = 0). The LP minimizes the maximum
        coalition deficit e subject to efficiency:

            min e   s.t.  sum_i x_i = v(N);   sum_{i in S} x_i + e >= v(S)

        for every sampled coalition S (capped at leastcore_max_coalitions).
        Solved with scipy.linprog (HiGHS) under a hard leastcore_timeout.
        Returns None and logs 'leastcore_intractable' on any failure.
        """
        import time as _time
        from scipy.optimize import linprog

        try:
            num_sellers = X.shape[1]
            v_empty = y[0]
            v_grand = y[1] - v_empty

            # Random coalitions only (anchor rows 0/1 handled separately);
            # drop empty/grand duplicates among them and cap the count.
            rows = []
            for k in range(2, X.shape[0]):
                size = X[k].sum()
                if 0 < size < num_sellers:
                    rows.append(k)
            rows = rows[: self.leastcore_max_coalitions]
            if not rows:
                logging.warning("leastcore_intractable: no proper coalitions sampled.")
                return None

            # Variables z = [x_1 .. x_N, e]; minimize e.
            c = np.zeros(num_sellers + 1)
            c[-1] = 1.0

            A_ub = np.hstack([-X[rows], -np.ones((len(rows), 1))])
            b_ub = -(y[rows] - v_empty)

            A_eq = np.ones((1, num_sellers + 1))
            A_eq[0, -1] = 0.0
            b_eq = np.array([v_grand])

            bounds = [(None, None)] * (num_sellers + 1)

            t0 = _time.time()
            res = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
                options={"time_limit": self.leastcore_timeout},
            )
            elapsed = _time.time() - t0

            if not res.success:
                logging.warning(
                    f"leastcore_intractable: LP status={res.status} "
                    f"({res.message}) after {elapsed:.1f}s; skipping."
                )
                return None

            logging.info(f"Least-core LP solved in {elapsed:.2f}s "
                         f"({len(rows)} constraints, e*={res.x[-1]:.4f}).")
            return res.x[:num_sellers]

        except Exception as e:
            logging.warning(f"leastcore_intractable: {e}; skipping.")
            return None
