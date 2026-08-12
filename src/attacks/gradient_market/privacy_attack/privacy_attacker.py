import argparse
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Any, Callable

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.attacks.gradient_market.privacy_attack.attack_methods.buyer_attack import PIAFeatureExtractor, \
    MIAFeatureExtractor
from src.attacks.gradient_market.privacy_attack.utils import get_seller_ground_truth, get_buyer_dataset, model_factory
from src.common_utils import ExperimentLoader


class AccessLevel(Enum):
    """Defines the scope of information an adversary can see."""
    LOCAL = auto()  # Can only see their own (and their sybils') data.
    GLOBAL = auto()  # Can see all public data in the marketplace (e.g., payments/weights for all).


class Adversary(ABC):
    """Abstract base class for all adversaries."""

    def __init__(self, adversary_id: str):
        self.adversary_id = adversary_id
        self.knowledge = {}
        logging.info(f"Adversary '{self.adversary_id}' initialized.")

    @abstractmethod
    def execute_attack(self, marketplace_state: Dict[str, Any] = None):
        pass

    def log_knowledge(self, key: str, value: Any):
        """Adds a piece of learned information to the adversary's knowledge base."""
        self.knowledge[key] = value
        logging.info(f"Adversary '{self.adversary_id}' learned: {key} = {value}")


class SellerAdversary(Adversary):
    """Represents a malicious seller, now equipped with powerful attack strategies."""

    def __init__(
            self,
            adversary_id: str,
            controls_sybil_identities: List[str],
            run_data: Dict[str, Any],  # <-- Add run_data
            loader: ExperimentLoader,  # <-- Add loader
            device: str = "cpu",
            access_level: AccessLevel = AccessLevel.LOCAL,
            can_perform_active_probing: bool = False  # <-- NEW CAPABILITY
    ):
        super().__init__(adversary_id)
        # Store all necessary components
        self.sybil_identities = controls_sybil_identities
        self.run_data = run_data
        self.loader = loader
        self.device = device
        self.access_level = access_level
        self.can_perform_active_probing = can_perform_active_probing

        # Components for the attacks
        self.mf = model_factory(run_data)
        self.probe_dataset = get_buyer_dataset(run_data)
        self.global_model_history = []
        self.attack_strategy = None

    def set_attack_strategy(self, strategy_name: str, strategy_params: Dict = None):
        """Sets the attack strategy and any parameters it might need."""
        self.strategy_params = strategy_params or {}
        strategies = {
            "passive_inference": self.run_passive_inference_attack,
            "active_probing": self.run_active_probing_attack
        }
        if strategy_name in strategies:
            self.attack_strategy = strategies[strategy_name]
            logging.info(f"SellerAdversary '{self.adversary_id}' strategy set to '{strategy_name}'.")
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def execute_attack(self, marketplace_state: Dict[str, Any] = None):
        """Triggers the chosen holistic, multi-round attack strategy."""
        if self.attack_strategy:
            # The strategy itself will handle the multi-round analysis
            logging.info(f"Executing holistic attack: {self.attack_strategy.__name__}")
            return self.attack_strategy(**self.strategy_params)
        logging.warning("No attack strategy set for seller adversary.")
        return None

    # --- MOVED METHOD ---
    def reconstruct_global_model_history(self, loader: ExperimentLoader, learning_rate: float):
        """Reconstructs the sequence of global models by applying aggregated gradients."""
        logging.info("Reconstructing global model history...")
        model = self.mf().to(self.device)
        self.global_model_history.append(model.state_dict())
        max_round = max(self.run_data['gradient_paths'].keys()) if self.run_data.get('gradient_paths') else -1

        for round_num in tqdm(range(max_round + 1), desc="Reconstructing Models"):
            agg_grad = loader.load_gradient(self.run_data, round_num, seller_id=None)
            if agg_grad:
                with torch.no_grad():
                    for param, grad in zip(model.parameters(), agg_grad):
                        param.data.sub_(grad, alpha=learning_rate)
                self.global_model_history.append(model.state_dict())
        logging.info(f"Reconstructed {len(self.global_model_history)} global model states.")

    def run_passive_inference_attack(self, probe_objectives: Dict[str, List[int]]) -> pd.DataFrame:
        """Runs the passive inference attack with the improved, stabilized marketplace-aware score."""
        if not self.global_model_history:
            raise RuntimeError("Global model history is empty. Call reconstruct first.")

        # Step 1: Calculate loss trajectories
        all_labels = [label for _, label in self.probe_dataset]
        probe_loaders = {
            name: DataLoader(Subset(self.probe_dataset, [i for i, label in enumerate(all_labels) if label in classes]),
                             128) for name, classes in probe_objectives.items()}

        loss_results = []
        model = self.mf().to(self.device)
        for round_num, model_state in enumerate(tqdm(self.global_model_history, desc="Probing Models")):
            model.load_state_dict(model_state)
            round_losses = {"round": round_num,
                            **{f"loss_{name}": self._get_model_loss(model, loader) for name, loader in
                               probe_loaders.items()}}
            loss_results.append(round_losses)
        loss_df = pd.DataFrame(loss_results).set_index('round')

        # Step 2: Calculate passive scores (loss improvement)
        passive_scores_df = -loss_df.diff().fillna(0).rename(columns=lambda c: c.replace('loss_', 'passive_score_'))

        # Step 3: Get aggregated Sybil signals
        sybil_histories = [self.run_data["sellers"][sid].set_index('round')['payment_received'] for sid in
                           self.sybil_identities]
        sybil_agg_payment = pd.concat(sybil_histories).groupby('round').sum().rename("total_payment")

        # Step 4: (IMPROVED) Create stabilized marketplace-aware scores
        scaler = MinMaxScaler()
        norm_passive = pd.DataFrame(scaler.fit_transform(passive_scores_df), columns=passive_scores_df.columns,
                                    index=passive_scores_df.index)
        norm_payment = pd.Series(scaler.fit_transform(sybil_agg_payment.values.reshape(-1, 1))[:, 0],
                                 index=sybil_agg_payment.index)

        market_scores_df = norm_passive.add(norm_payment, axis=0).rename(
            columns=lambda c: c.replace('passive_score_', 'market_score_'))

        # Step 5: Determine inferred objectives
        passive_scores_df['passive_inferred_objective'] = passive_scores_df.idxmax(axis=1).str.replace('passive_score_',
                                                                                                       '')
        market_scores_df['market_inferred_objective'] = market_scores_df.idxmax(axis=1).str.replace('market_score_', '')

        return loss_df.join(passive_scores_df).join(market_scores_df).join(sybil_agg_payment).reset_index()

    # --- MOVED METHOD / NEW STRATEGY ---
    def run_active_probing_attack(self, sybil_roles: Dict[str, List[str]]):
        # Add a capability check
        if not self.can_perform_active_probing:
            raise PermissionError("Adversary lacks 'can_perform_active_probing' capability.")
        logging.info("Running active probing attack with specialized Sybils...")
        all_sybil_history = []
        for objective, sids in sybil_roles.items():
            for sid in sids:
                if sid in self.run_data["sellers"]:
                    history = self.run_data["sellers"][sid].copy()
                    history['objective_probe'] = objective
                    all_sybil_history.append(history)

        if not all_sybil_history:
            raise ValueError("None of the specified Sybils in roles were found in the run data.")

        full_history_df = pd.concat(all_sybil_history)

        # The inference score *is* the total payment per objective group.
        active_scores_df = full_history_df.groupby(['round', 'objective_probe'])['payment_received'].sum().unstack(
            fill_value=0)
        active_scores_df.columns = [f"active_score_{col}" for col in active_scores_df.columns]

        active_scores_df['active_inferred_objective'] = active_scores_df.idxmax(axis=1).str.replace('active_score_', '')
        return active_scores_df.reset_index()

    def _get_model_loss(self, model: nn.Module, data_loader: DataLoader) -> float:
        """Helper to calculate the average loss of a model on a dataset."""
        model.eval()
        total_loss, total_samples = 0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
        return total_loss / total_samples if total_samples > 0 else float('inf')


class BuyerAdversary(Adversary):
    """Represents a malicious buyer running the marketplace."""

    def __init__(self, adversary_id: str, run_data: Dict[str, Any], device: str = "cpu"):
        super().__init__(adversary_id)
        # --- NEW: Initialize with necessary data for attacks ---
        self.run_data = run_data
        self.mf = model_factory(run_data)
        self.buyer_root_dataset = get_buyer_dataset(run_data)
        self.device = device

        # --- NEW: Initialize feature extractors ---
        self.mia_extractor = MIAFeatureExtractor(self.mf, self.buyer_root_dataset, self.device)
        self.pia_extractor = PIAFeatureExtractor(self.mf, self.buyer_root_dataset, self.device)

        # --- NEW: Placeholders for the trained attack models ---
        self.mia_baseline, self.mia_market = None, None
        self.pia_baseline, self.pia_market = None, None
        self.pia_ref_grad = None

    def execute_attack(self, marketplace_state: Dict[str, Any] = None):
        """
        For the buyer, the main attack is a complex analysis run at the end.
        This method can be a placeholder or used for simple, per-round actions.
        """
        # The main analysis will be triggered by the pipeline separately.
        logging.debug(f"BuyerAdversary observing round {marketplace_state.get('round_num')}")
        pass

    # --- NEW: Method to train the attack models (from MarketplaceBuyerAttacker) ---
    def train_attack_models(self, property_function: Callable):
        """Trains all four attack models (MIA/PIA, Baseline/Market-Aware)."""
        logging.info(f"\n--- Buyer '{self.adversary_id}' is training its attack models... ---")
        # (This code is copied directly from your train_all_attack_models method)
        mia_df = self.mia_extractor.create_training_features()
        y_mia = mia_df["is_member"]
        self.mia_baseline = GradientBoostingClassifier(n_estimators=100).fit(
            mia_df[["loss", "confidence", "entropy", "is_correct"]], y_mia)
        self.mia_market = GradientBoostingClassifier(n_estimators=100).fit(mia_df.drop("is_member", axis=1), y_mia)
        logging.info("MIA classifiers trained.")

        pia_df = self.pia_extractor.create_training_features(property_function)
        y_pia = pia_df["has_property"]
        if len(y_pia.unique()) < 2:
            logging.error("Cannot train PIA models: only one class present.")
            return
        baseline_cols = [c for c in pia_df.columns if c.startswith('layer_') or c == 'grad_cosine_similarity']
        self.pia_baseline = GradientBoostingClassifier(n_estimators=100).fit(pia_df[baseline_cols], y_pia)
        self.pia_market = GradientBoostingClassifier(n_estimators=100).fit(pia_df.drop("has_property", axis=1), y_pia)
        logging.info("PIA classifiers trained.")

        prop_example_indices = [i for i, _ in enumerate(self.buyer_root_dataset) if property_function([i])][:128]
        if prop_example_indices:
            self.pia_ref_grad = self.pia_extractor._get_reference_gradient(prop_example_indices)

    # --- NEW: Method to perform the analysis (from MarketplaceBuyerAttacker) ---
    def analyze_seller_leakage_over_time(self, loader: ExperimentLoader, target_seller_id: str,
                                         property_function: Callable):
        """Performs the full temporal analysis on a single target seller."""
        # (This code is copied directly from your analyze_seller_leakage_over_time method)
        if not self.mia_market or not self.pia_market:
            raise RuntimeError("Attack models must be trained before analysis.")
        logging.info(f"\n--- Analyzing Leakage for Seller '{target_seller_id}' Over Time ---")
        seller_history = self.run_data["sellers"][target_seller_id]
        max_round = seller_history['round'].max()
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


def create_adversaries_from_run(run_data: Dict[str, Any], adv_prefix="adv") -> Dict[str, Adversary]:
    """Factory to create adversary objects based on logged data."""
    adversaries = {}

    # Create Seller Adversary if Sybils are found
    all_sellers = list(run_data["sellers"].keys())
    sybil_ids = [sid for sid in all_sellers if sid.startswith(adv_prefix)]
    if sybil_ids:
        adversaries["seller_adversary"] = SellerAdversary(
            "Sybil-Master",
            controls_sybil_identities=sybil_ids
        )

    # Always create a buyer adversary for analysis
    adversaries["buyer_adversary"] = BuyerAdversary("Malicious-Buyer")

    return adversaries


class MarketplaceAdversary(Adversary):
    """
    Represents the marketplace platform itself as a potential adversary.
    This adversary has a global, "God's-eye" view and can perform system-level audits.
    """

    def __init__(self, adversary_id: str, run_data: Dict[str, Any]):
        super().__init__(adversary_id)
        self.run_data = run_data

    def set_attack_strategy(self, strategy_name: str, strategy_params: Dict = None):
        """Sets the audit/attack strategy and its parameters."""
        self.strategy_params = strategy_params or {}
        strategies = {
            "system_leakage_audit": self.run_system_leakage_audit
        }
        if strategy_name in strategies:
            self.attack_strategy = strategies[strategy_name]
            logging.info(f"MarketplaceAdversary '{self.adversary_id}' strategy set to '{strategy_name}'.")
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def execute_attack(self, marketplace_state: Dict[str, Any] = None):
        """Triggers the chosen holistic analysis."""
        if self.attack_strategy:
            return self.attack_strategy(**self.strategy_params)
        logging.warning("No strategy set for MarketplaceAdversary.")
        return None

    # --- MAIN STRATEGY ---
    def run_system_leakage_audit(self, target_seller_ids: List[str]):
        """
        Measures the correlation between public signals and private data properties
        for a target group of sellers (e.g., a Sybil coalition).
        """
        logging.info("Executing 'system_leakage_audit'...")

        # 1. Build the DataFrame by aggregating signals and ground truth properties
        correlation_df = self._build_correlation_dataframe(target_seller_ids)

        # 2. Analyze and Plot the correlation matrix
        self._analyze_and_plot_correlation(correlation_df)

        self.log_knowledge("system_leakage_audit_complete", True)
        return correlation_df

    # --- HELPER METHODS (Adapted from your script) ---
    def _get_seller_data_properties(self, seller_id: str) -> Dict[str, float]:
        """Calculates the ground-truth properties of a seller's private dataset."""
        seller_gt = get_seller_ground_truth(self.run_data, seller_id)
        member_indices = seller_gt.get("member_indices", [])

        if not member_indices:
            return {}

        full_dataset = get_buyer_dataset(self.run_data)
        labels = [label for _, label in full_dataset]
        seller_labels = [labels[i] for i in member_indices]

        properties = {}
        num_labels = len(seller_labels)
        unique_classes = sorted(list(set(labels)))

        for class_idx in unique_classes:
            count = seller_labels.count(class_idx)
            properties[f'prop_class_{class_idx}'] = count / num_labels if num_labels > 0 else 0.0

        return properties

    def _build_correlation_dataframe(self, target_seller_ids: List[str]) -> pd.DataFrame:
        """
        Constructs a DataFrame by aggregating marketplace signals and the ground-truth
        data properties for the target sellers.
        """
        logging.info(f"Building correlation dataframe for {len(target_seller_ids)} sellers...")

        # 1. Calculate the AGGREGATE properties of the target sellers' entire data pool
        aggregate_properties = {}
        all_adversary_indices = []
        for seller_id in target_seller_ids:
            seller_gt = get_seller_ground_truth(self.run_data, seller_id)
            all_adversary_indices.extend(seller_gt.get("member_indices", []))

        full_dataset = get_buyer_dataset(self.run_data)
        labels = [label for _, label in full_dataset]
        adversary_labels = [labels[i] for i in all_adversary_indices]

        num_labels = len(adversary_labels)
        unique_classes = sorted(list(set(labels)))
        for class_idx in unique_classes:
            count = adversary_labels.count(class_idx)
            aggregate_properties[f'prop_class_{class_idx}'] = count / num_labels if num_labels > 0 else 0.0

        # 2. Aggregate marketplace signals per round
        adversary_round_data = []
        num_rounds = len(self.run_data['sellers'][target_seller_ids[0]])

        for round_num in tqdm(range(num_rounds), desc="Aggregating Signals for Audit"):
            total_payment_in_round, total_weight_in_round = 0.0, 0.0

            for seller_id in target_seller_ids:
                history = self.run_data['sellers'][seller_id]
                round_info = history[history['round'] == round_num]
                if not round_info.empty:
                    total_payment_in_round += round_info.get('payment_received', [0])[0]
                    total_weight_in_round += round_info.get('assigned_weight', [0])[0]

            round_entry = {
                'round': round_num,
                'payment_received': total_payment_in_round,
                'assigned_weight': total_weight_in_round
            }
            round_entry.update(aggregate_properties)
            adversary_round_data.append(round_entry)

        return pd.DataFrame(adversary_round_data)

    def _analyze_and_plot_correlation(self, df: pd.DataFrame):
        """Calculates and visualizes the correlation matrix."""
        run_name = self.run_data['run_path'].name
        logging.info(f"\n--- 📊 SYSTEM LEAKAGE AUDIT REPORT ({run_name}) 📊 ---")

        if df.empty:
            logging.warning("DataFrame is empty. Cannot perform correlation analysis.")
            return

        marketplace_signals = ['payment_received', 'assigned_weight']
        data_properties = [col for col in df.columns if col.startswith('prop_class_')]

        if not data_properties:
            logging.warning("No data property columns found. Cannot perform correlation analysis.")
            return

        correlation_matrix = df[marketplace_signals + data_properties].corr()
        leakage_correlation = correlation_matrix.loc[marketplace_signals, data_properties]

        print("--- System-Level Leakage Correlation Matrix ---")
        print("This matrix shows the correlation between public marketplace signals (rows)")
        print("and private seller data properties (columns).")
        print("Values close to +1 or -1 indicate significant information leakage.")
        print(leakage_correlation.to_string())

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            leakage_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".3f"
        )
        title = (
            f'System-Level Leakage: Marketplace Audit\n'
            f'Correlation of Aggregated Marketplace Signals vs. Aggregated Data Properties\nRun: {run_name}'
        )
        plt.title(title, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = f"system_leakage_audit_{run_name}.png"
        plt.savefig(save_path)
        logging.info(f"Correlation heatmap saved to {save_path}")
        plt.close()


class PostHocAnalysisPipeline:
    """Orchestrates a post-hoc analysis of a completed experiment run."""

    def __init__(self, run_data: Dict[str, Any], loader: ExperimentLoader):
        self.run_data = run_data
        self.loader = loader
        # The factory now creates fully initialized adversaries in one step.
        self.adversaries = self._create_adversaries_from_run()

    def _create_adversaries_from_run(self, adv_prefix="adv") -> Dict[str, 'Adversary']:
        """
        Factory to create fully initialized adversary objects based on logged data.
        """
        adversaries = {}
        all_sellers = list(self.run_data.get("sellers", {}).keys())
        sybil_ids = [sid for sid in all_sellers if sid.startswith(adv_prefix)]

        # Create Seller Adversary if Sybils are found
        if sybil_ids:
            adversaries["seller_adversary"] = SellerAdversary(
                adversary_id="Sybil-Master",
                controls_sybil_identities=sybil_ids,
                run_data=self.run_data,
                loader=self.loader,
                can_perform_active_probing=True,  # This could also come from a config
                access_level=AccessLevel.GLOBAL  # This could also come from a config
            )

        # Always create a Buyer Adversary for analysis
        adversaries["buyer_adversary"] = BuyerAdversary(
            adversary_id="Malicious-Buyer",
            run_data=self.run_data,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Add MarketplaceAdversary here if needed
        adversaries["marketplace_adversary"] = MarketplaceAdversary(
            adversary_id="Platform-Auditor",
            run_data=self.run_data
        )

        logging.info(f"Created adversaries: {list(adversaries.keys())}")
        return adversaries

    def run_analysis(self, attack_configs: Dict[str, Any]):
        """
        Replays the experiment to perform adversarial analysis based on a config.
        """
        logging.info(f"\n--- 🚀 Starting Post-Hoc Analysis for Run: {self.run_data['run_path'].name} ---")

        # --- Part 1: Seller Attack ---
        if "seller_attack" in attack_configs and "seller_adversary" in self.adversaries:
            config = attack_configs["seller_attack"]
            seller_adv = self.adversaries["seller_adversary"]

            logging.info(f"--- Running Seller Attack: {config['strategy']} ---")

            lr = self.run_data['config']['server_learning_rate']  # Get LR from config
            seller_adv.reconstruct_global_model_history(learning_rate=lr)
            seller_adv.set_attack_strategy(config["strategy"], config.get("params", {}))
            results_df = seller_adv.execute_attack()

            if results_df is not None:
                logging.info("\n--- 📊 SELLER ATTACK FINAL REPORT 📊 ---")
                print(results_df.head().to_string())
                # Your plotting logic here...

        # --- Part 2: Buyer Attack ---
        if "buyer_attack" in attack_configs and "buyer_adversary" in self.adversaries:
            config = attack_configs["buyer_attack"]
            buyer_adv = self.adversaries["buyer_adversary"]

            logging.info(f"--- Running Buyer Attack on Target: {config['target_seller_id']} ---")

            # The property function is now passed in via the config!
            property_function = config.get("property_function")
            if not callable(property_function):
                raise TypeError("The 'property_function' in buyer_attack config must be a callable function.")

            buyer_adv.train_attack_models(property_function=property_function)
            leakage_df = buyer_adv.analyze_seller_leakage_over_time(
                loader=self.loader,
                target_seller_id=config['target_seller_id'],
                property_function=property_function
            )

            if not leakage_df.empty:
                logging.info("\n--- 📊 BUYER ATTACK FINAL REPORT 📊 ---")
                print(leakage_df.to_string())
                # Your plotting logic here...


# --- This becomes much cleaner and more powerful ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Run post-hoc privacy analysis on a marketplace experiment.")
    parser.add_argument("experiment_root", type=str, help="Path to the root directory of experiment results.")
    args = parser.parse_args()

    try:
        loader = ExperimentLoader(args.experiment_root)
        if not loader.runs:
            logging.warning(f"No experiment runs found in '{args.experiment_root}'.")
            exit()

        first_run_path = loader.runs[0]
        logging.info(f"Analyzing run: {first_run_path.name}")
        run_data = loader.load_run_data(first_run_path)

        # --- Define your property function here or in a separate utils file ---
        labels = [label for _, label in get_buyer_dataset(run_data)]


        def has_high_class_2_ratio(indices: List[int]) -> bool:
            if not indices: return False
            class_2_count = sum(1 for i in indices if labels[i] == 2)
            return (class_2_count / len(indices)) > 0.5


        # --- Define the full analysis plan in a single config dictionary ---
        analysis_plan = {
            "seller_attack": {
                "strategy": "passive_inference",
                "params": {
                    "probe_objectives": {
                        "group_A": [0, 1],
                        "group_B": [2, 3]
                    }
                }
            },
            "buyer_attack": {
                "target_seller_id": "seller_0",  # Example target
                "property_function": has_high_class_2_ratio
            }
        }

        pipeline = PostHocAnalysisPipeline(run_data, loader)
        pipeline.run_analysis(analysis_plan)

    except (FileNotFoundError, RuntimeError, TypeError) as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
