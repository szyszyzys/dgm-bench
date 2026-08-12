import logging
import warnings
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.common_utils import ExperimentLoader
# --- Import your project's utilities and loader ---
from src.attacks.gradient_market.privacy_attack.utils import get_buyer_dataset, get_seller_ground_truth, model_factory
from src.mechanism.gradient.payment import PaymentSimulator

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)  # For Z-score with std=0


# =======================================================================================
# PART 1: MODULAR FEATURE EXTRACTORS (ENHANCED)
# =======================================================================================

class MIAFeatureExtractor:
    """Encapsulates all logic for creating Membership Inference Attack features."""

    def __init__(self, model_factory: Callable, dataset: torch.utils.data.Dataset, device: str):
        self.mf = model_factory
        self.dataset = dataset
        self.device = device

    def _get_model_outputs(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> pd.DataFrame:
        """Calculates loss, confidence, entropy, and correctness for each sample."""
        model.eval()
        outputs_list = []
        criterion = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits = model(data)
                probs = torch.softmax(logits, dim=1)

                for i in range(data.size(0)):
                    true_label = target[i].item()
                    pred_label = probs[i].argmax().item()
                    outputs_list.append({
                        "loss": criterion(logits[i:i + 1], target[i:i + 1]).item(),
                        "confidence": probs[i, true_label].item(),
                        "entropy": entropy(probs[i].cpu().numpy()),
                        "is_correct": 1 if pred_label == true_label else 0
                    })
        return pd.DataFrame(outputs_list)

    def create_training_features(self, num_shadow_models: int = 50, batch_size: int = 128) -> pd.DataFrame:
        """Creates a rich feature set for MIA training using shadow modeling."""
        logging.info("Creating MIA training feature set...")
        features = []
        model = self.mf().to(self.device)
        all_indices = list(range(len(self.dataset)))

        for _ in tqdm(range(num_shadow_models), desc="MIA Shadow Training"):
            np.random.shuffle(all_indices)
            member_indices = all_indices[:batch_size]
            non_member_indices = all_indices[batch_size: 2 * batch_size]

            member_df = self._get_model_outputs(model, torch.utils.data.DataLoader(
                torch.utils.data.Subset(self.dataset, member_indices), 64))
            non_member_df = self._get_model_outputs(model, torch.utils.data.DataLoader(
                torch.utils.data.Subset(self.dataset, non_member_indices), 64))

            # Simulate high-value vs low-value marketplace signals
            member_df["payment"] = np.random.uniform(5, 10, size=len(member_df));
            member_df["payment_trend"] = np.random.uniform(0.1, 0.5, size=len(member_df))
            non_member_df["payment"] = np.random.uniform(0, 2, size=len(non_member_df));
            non_member_df["payment_trend"] = np.random.uniform(-0.5, -0.1, size=len(non_member_df))

            member_df["is_member"] = 1;
            non_member_df["is_member"] = 0
            features.extend([member_df, non_member_df])

        return pd.concat(features, ignore_index=True)

    def extract_live_features(self, eval_indices: List[int], seller_history: pd.DataFrame) -> pd.DataFrame:
        """Extracts features for a live attack round using real marketplace history."""
        model = self.mf().to(self.device)
        data_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.dataset, eval_indices), batch_size=64)
        tech_features_df = self._get_model_outputs(model, data_loader)

        # Calculate historical marketplace signals
        mean_payment = seller_history['payment_received'].mean();
        std_payment = seller_history['payment_received'].std()
        current_payment = seller_history['payment_received'].iloc[-1]

        tech_features_df["payment_z_score"] = (current_payment - mean_payment) / std_payment if std_payment > 0 else 0
        tech_features_df["selection_rate"] = seller_history['assigned_weight'].gt(0).mean()

        # New Feature: Payment Trend
        recent_payments = seller_history['payment_received'].tail(5)
        if len(recent_payments) > 1:
            tech_features_df["payment_trend"] = np.polyfit(range(len(recent_payments)), recent_payments, 1)[0]
        else:
            tech_features_df["payment_trend"] = 0

        return tech_features_df


class PIAFeatureExtractor:
    """Encapsulates all logic for creating Property Inference Attack features."""

    def __init__(self, model_factory: Callable, dataset: torch.utils.data.Dataset, device: str):
        self.mf = model_factory
        self.dataset = dataset
        self.device = device
        self.reference_gradients = {}

    def _get_reference_gradient(self, property_indices: List[int]) -> torch.Tensor:
        """Computes a reference gradient for a specific property."""
        prop_key = tuple(sorted(property_indices))
        if prop_key in self.reference_gradients: return self.reference_gradients[prop_key]

        model = self.mf().to(self.device)
        model.train()
        data, target = next(iter(torch.utils.data.DataLoader(
            torch.utils.data.Subset(self.dataset, property_indices), batch_size=len(property_indices)
        )))
        data, target = data.to(self.device), target.to(self.device)
        model.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        ref_grad = torch.cat([p.grad.detach().clone().flatten() for p in model.parameters() if p.grad is not None])
        self.reference_gradients[prop_key] = ref_grad
        return ref_grad

    def create_training_features(self, property_function: Callable, num_shadow_models: int = 100,
                                 batch_size: int = 128) -> pd.DataFrame:
        """Creates a feature set for PIA by correlating gradient stats with properties."""
        logging.info("Creating PIA training feature set...")
        features = []
        start_model_state = self.mf().to(self.device).state_dict()
        all_indices = list(range(len(self.dataset)))

        # New Feature: Create a reference gradient for cosine similarity
        prop_example_indices = [i for i, _ in enumerate(self.dataset) if property_function([i])][:batch_size]
        ref_grad = self._get_reference_gradient(prop_example_indices) if prop_example_indices else None

        for _ in tqdm(range(num_shadow_models), desc="PIA Shadow Training"):
            model = self.mf().to(self.device)
            model.load_state_dict(start_model_state)
            model.train()
            np.random.shuffle(all_indices)
            shadow_indices = all_indices[:batch_size]
            has_property = property_function(shadow_indices)

            data, target = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(self.dataset, shadow_indices),
                                                                 batch_size=batch_size)))
            data, target = data.to(self.device), target.to(self.device)
            model.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            grad_features = {}
            layer_grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
            current_grad_flat = torch.cat([g.flatten() for g in layer_grads])

            for i, grad_layer in enumerate(layer_grads): grad_features[f"layer_{i}_norm"] = torch.linalg.norm(
                grad_layer).item()
            if ref_grad is not None:
                grad_features["grad_cosine_similarity"] = torch.nn.functional.cosine_similarity(ref_grad,
                                                                                                current_grad_flat,
                                                                                                dim=0).item()

            grad_features["payment"] = np.random.uniform(5, 10) if has_property else np.random.uniform(0, 2)
            grad_features["payment_volatility"] = np.random.uniform(2, 5) if has_property else np.random.uniform(0.1, 1)
            grad_features["has_property"] = int(has_property)
            features.append(grad_features)

        return pd.DataFrame(features).fillna(0)

    def extract_live_features(self, gradient: List[torch.Tensor], seller_history: pd.DataFrame,
                              ref_grad: torch.Tensor) -> pd.DataFrame:
        """Extracts PIA features for a live gradient using real marketplace history."""
        grad_features = {}
        for i, grad_layer in enumerate(gradient): grad_features[f"layer_{i}_norm"] = torch.linalg.norm(
            grad_layer).item()

        current_grad_flat = torch.cat([g.flatten() for g in gradient])
        if ref_grad is not None:
            grad_features["grad_cosine_similarity"] = torch.nn.functional.cosine_similarity(ref_grad, current_grad_flat,
                                                                                            dim=0).item()

        current_payment = seller_history['payment_received'].iloc[-1]
        grad_norm_total = np.linalg.norm(list(grad_features.values()))
        grad_features["payment_per_norm"] = current_payment / grad_norm_total if grad_norm_total > 0 else 0
        grad_features["payment_volatility"] = seller_history['payment_received'].std()
        return pd.DataFrame([grad_features]).fillna(0)


# =======================================================================================
# PART 2: MAIN ATTACKER CLASS
# =======================================================================================

class MarketplaceBuyerAttacker:
    """Orchestrates the training and execution of comparative privacy attacks."""

    def __init__(self, run_data: Dict[str, Any], device: str = "cpu"):
        self.run_data = run_data
        self.mf = model_factory(run_data)
        self.buyer_root_dataset = get_buyer_dataset(run_data)
        self.mia_extractor = MIAFeatureExtractor(self.mf, self.buyer_root_dataset, device)
        self.pia_extractor = PIAFeatureExtractor(self.mf, self.buyer_root_dataset, device)

        self.mia_baseline, self.mia_market = None, None
        self.pia_baseline, self.pia_market = None, None
        self.pia_ref_grad = None
        logging.info("MarketplaceBuyerAttacker initialized with modular feature extractors.")

    def train_all_attack_models(self, property_function: Callable, test_size: float = 0.3):
        """Trains all four attack models (MIA/PIA, Baseline/Market-Aware)."""
        # --- Train MIA Models ---
        mia_df = self.mia_extractor.create_training_features()
        y = mia_df["is_member"]

        self.mia_baseline = GradientBoostingClassifier(n_estimators=100)
        self.mia_baseline.fit(mia_df[["loss", "confidence", "entropy", "is_correct"]], y)
        logging.info(f"MIA Baseline classifier trained.")

        self.mia_market = GradientBoostingClassifier(n_estimators=100)
        self.mia_market.fit(mia_df.drop("is_member", axis=1), y)
        logging.info(f"MIA Marketplace-Aware classifier trained.")

        # --- Train PIA Models ---
        pia_df = self.pia_extractor.create_training_features(property_function)
        y = pia_df["has_property"]
        if len(y.unique()) < 2: logging.error("Cannot train PIA models: only one class present."); return

        baseline_cols = [c for c in pia_df.columns if c.startswith('layer_') or c == 'grad_cosine_similarity']
        self.pia_baseline = GradientBoostingClassifier(n_estimators=100)
        self.pia_baseline.fit(pia_df[baseline_cols], y)
        logging.info(f"PIA Baseline classifier trained.")

        self.pia_market = GradientBoostingClassifier(n_estimators=100)
        self.pia_market.fit(pia_df.drop("has_property", axis=1), y)
        logging.info(f"PIA Marketplace-Aware classifier trained.")

        # Store the reference gradient for live analysis
        prop_example_indices = [i for i, _ in enumerate(self.buyer_root_dataset) if property_function([i])][:128]
        if prop_example_indices:
            self.pia_ref_grad = self.pia_extractor._get_reference_gradient(prop_example_indices)

    def analyze_seller_leakage_over_time(
            self, loader: ExperimentLoader, target_seller_id: str, property_function: Callable[[List[int]], bool]
    ) -> pd.DataFrame:
        if not self.mia_market or not self.pia_market: raise RuntimeError("Attack models not trained.")

        logging.info(f"\n--- Analyzing Leakage for Seller '{target_seller_id}' Over Time ---")
        seller_history = self.run_data["sellers"][target_seller_id]
        max_round = seller_history['round'].max();
        results = []

        for round_num in tqdm(range(max_round + 1), desc=f"Analyzing Seller {target_seller_id}"):
            round_results = {"round": round_num}
            history_so_far = seller_history[seller_history['round'] <= round_num]
            if history_so_far.empty: continue

            # --- MIA Analysis ---
            seller_gt = get_seller_ground_truth(self.run_data, target_seller_id)
            member_indices = seller_gt.get("member_indices", [])
            non_member_indices = list(set(range(len(self.buyer_root_dataset))) - set(member_indices));
            np.random.shuffle(non_member_indices)
            eval_indices = member_indices + non_member_indices[:len(member_indices)]
            eval_labels = [1] * len(member_indices) + [0] * len(non_member_indices[:len(member_indices)])
            live_mia_features = self.mia_extractor.extract_live_features(eval_indices, history_so_far)

            round_results["mia_auc_baseline"] = roc_auc_score(eval_labels, self.mia_baseline.predict_proba(
                live_mia_features[self.mia_baseline.feature_names_in_])[:, 1])
            round_results["mia_auc_market"] = roc_auc_score(eval_labels, self.mia_market.predict_proba(
                live_mia_features[self.mia_market.feature_names_in_])[:, 1])

            # --- PIA Analysis ---
            seller_grad = loader.load_gradient(self.run_data, round_num, target_seller_id)
            if seller_grad:
                true_property = int(property_function(member_indices))
                live_pia_features = self.pia_extractor.extract_live_features(seller_grad, history_so_far,
                                                                             self.pia_ref_grad)

                pia_pred_base = self.pia_baseline.predict(live_pia_features[self.pia_baseline.feature_names_in_])[0]
                round_results["pia_accuracy_baseline"] = 1.0 if pia_pred_base == true_property else 0.0

                pia_pred_market = self.pia_market.predict(live_pia_features[self.pia_market.feature_names_in_])[0]
                round_results["pia_accuracy_market"] = 1.0 if pia_pred_market == true_property else 0.0

            results.append(round_results)
        return pd.DataFrame(results)


# =======================================================================================
# PART 3: PLOTTING AND MAIN EXECUTION
# =======================================================================================

def plot_results(df: pd.DataFrame, run_name: str):
    """Generates a comparative plot of privacy leakage over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Privacy Leakage Comparison: Baseline vs. Marketplace-Aware\nRun: {run_name}', fontsize=16)

    ax1.plot(df['round'], df['mia_auc_baseline'], marker='o', linestyle='--', label='MIA AUC (Baseline)')
    ax1.plot(df['round'], df['mia_auc_market'], marker='o', linestyle='-', label='MIA AUC (Marketplace-Aware)',
             color='red')
    ax1.set_ylabel('AUC Score');
    ax1.set_title('Membership Inference Attack Performance')
    ax1.legend();
    ax1.grid(True, linestyle='--', alpha=0.6);
    ax1.set_ylim(0.4, 1.05)

    df['pia_acc_base_roll'] = df['pia_accuracy_baseline'].rolling(window=5, min_periods=1).mean()
    df['pia_acc_market_roll'] = df['pia_accuracy_market'].rolling(window=5, min_periods=1).mean()
    ax2.plot(df['round'], df['pia_acc_base_roll'], marker='o', linestyle='--',
             label='PIA Accuracy (Baseline, 5-round avg)')
    ax2.plot(df['round'], df['pia_acc_market_roll'], marker='o', linestyle='-',
             label='PIA Accuracy (Marketplace-Aware, 5-round avg)', color='red')
    ax2.set_xlabel('Training Round');
    ax2.set_ylabel('Accuracy (5-round Rolling Avg)')
    ax2.set_title('Property Inference Attack Performance')
    ax2.legend();
    ax2.grid(True, linestyle='--', alpha=0.6);
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    save_path = f"task3_1_leakage_comparison_{run_name}.png"
    plt.savefig(save_path);
    logging.info(f"Results plot saved to {save_path}");
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- (Your existing setup code remains the same) ---
    EXPERIMENT_ROOT = "./exp_results/text_agnews_cnn_10seller"
    TARGET_RUN_INDEX = 0
    TARGET_SELLER_ID = "bn_5"

    try:
        loader = ExperimentLoader(EXPERIMENT_ROOT)
        all_runs = loader.load_all_runs_data()
        if not all_runs: raise ValueError("No experiment runs found.")
        target_run_data = all_runs[TARGET_RUN_INDEX]
        run_name = target_run_data['run_path'].name
        logging.info(f"Loaded data for run: {run_name}")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load experiment data: {e}");
        exit()

    # ======================================================================
    # == NEW: Step 1 - Initialize the Payment Simulator                  ==
    # ======================================================================
    # Choose the payment model you want to test for this analysis.
    # Options: 'binary', 'proportional'
    payment_sim = PaymentSimulator(
        model_name='proportional',
        scale_factor=50.0,
        noise_level=0.5
    )
    logging.info(f"Using '{payment_sim.model_name}' payment model for analysis.")

    # ======================================================================
    # == NEW: Step 2 - Augment the Loaded Data with Simulated Payments   ==
    # ======================================================================
    for seller_id, history_df in target_run_data["sellers"].items():
        # The simulator adds the 'payment_received' column in place
        augmented_history = payment_sim.add_payments_to_history(history_df.copy())
        target_run_data["sellers"][seller_id] = augmented_history

    # ======================================================================
    # == Step 3 - Initialize the Attacker with AUGMENTED Data            ==
    # ======================================================================
    # The attacker now receives the data that includes the simulated payments.
    # No changes are needed in the attacker class itself.
    attacker = MarketplaceBuyerAttacker(target_run_data, device="cpu")

    # --- (The rest of your code remains exactly the same) ---
    full_dataset = get_buyer_dataset(target_run_data)
    labels = [label for _, label in full_dataset]


    def has_high_class_2_ratio(indices: List[int]) -> bool:
        if not indices: return False
        class_2_count = sum(1 for i in indices if labels[i] == 2)
        return (class_2_count / len(indices)) > 0.5


    attacker.train_all_attack_models(property_function=has_high_class_2_ratio)

    try:
        leakage_df = attacker.analyze_seller_leakage_over_time(
            loader=loader, target_seller_id=TARGET_SELLER_ID,
            property_function=has_high_class_2_ratio
        )
        if not leakage_df.empty:
            logging.info("\n--- 📊 FINAL ANALYSIS REPORT 📊 ---");
            print(leakage_df.to_string())
            plot_results(leakage_df, run_name)
        else:
            logging.warning("Analysis completed but produced no results.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
