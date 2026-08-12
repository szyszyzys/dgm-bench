import argparse
import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.attacks.gradient_market.privacy_attack.utils import get_buyer_dataset, model_factory

from src.common_utils import ExperimentLoader  # Your existing loaders


class MarketplaceSellerAttacker:
    """Implements and compares different Buyer Intent Inference Attacks."""

    def __init__(self, run_data: Dict[str, Any], attacker_ids: List[str], device: str = "cpu"):
        self.run_data = run_data
        self.attacker_ids = attacker_ids
        self.device = device
        self.mf = model_factory(run_data)
        self.probe_dataset = get_buyer_dataset(run_data)
        self.global_model_history = []
        logging.info(f"Attacker initialized, controlling: {attacker_ids}.")

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
                           self.attacker_ids]
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

    def run_active_probing_attack(self, sybil_roles: Dict[str, List[str]]) -> pd.DataFrame:
        """Runs the active probing attack by comparing payments to specialized Sybil groups."""
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


class SellerAttackAnalyzer:
    """Orchestrates the entire seller-side attack analysis from a config file."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = "cpu"
        self.loader = ExperimentLoader(self.config['experiment_root'])
        all_runs = self.loader.load_all_runs_data()
        self.run_data = all_runs[self.config['target_run_index']]
        self.run_name = self.run_data['run_path'].name

        all_seller_ids = list(self.run_data['sellers'].keys())
        attacker_prefix = self.config['attacker_prefix']
        self.attacker_ids = [sid for sid in all_seller_ids if sid.startswith(attacker_prefix)]

        if not self.attacker_ids:
            raise ValueError(f"No adversary sellers found with prefix '{attacker_prefix}'.")

        # Simulate payments if not present (using the proportional model as a default)
        if 'payment_received' not in list(self.run_data['sellers'].values())[0].columns:
            logging.info("Payments not found in logs. Simulating based on 'assigned_weight'.")
            for sid in all_seller_ids:
                history = self.run_data['sellers'][sid]
                payments = history['assigned_weight'] * 50.0 + np.random.normal(0, 0.5, len(history))
                history['payment_received'] = np.maximum(0, payments)

        self.attacker = MarketplaceSellerAttacker(self.run_data, self.attacker_ids, self.device)
        logging.info(f"Analyzer initialized for run '{self.run_name}' with {len(self.attacker_ids)} Sybils.")

    def run_analysis(self):
        """Executes the analysis workflow based on the provided configuration."""
        self.attacker.reconstruct_global_model_history(self.loader, self.config['server_learning_rate'])

        if 'active_probing_roles' in self.config and self.config['active_probing_roles']:
            logging.info("--- Running ACTIVE PROBING ATTACK ---")
            results_df = self.attacker.run_active_probing_attack(self.config['active_probing_roles'])
            self._analyze_and_plot(results_df, 'active')
        else:
            logging.info("--- Running PASSIVE INFERENCE ATTACK ---")
            results_df = self.attacker.run_passive_inference_attack(self.config['passive_probe_objectives'])
            self._analyze_and_plot(results_df, 'passive')
            self._analyze_and_plot(results_df, 'market')

    def _analyze_and_plot(self, df: pd.DataFrame, attack_type: str):
        """A generalized analysis and plotting function."""
        ground_truth = self.config['ground_truth_objective']
        score_prefix = f"{attack_type}_score_"
        inference_col = f"{attack_type}_inferred_objective"

        # --- Metrics ---
        final_pred = df[inference_col].iloc[-1]
        accuracy = 1.0 if final_pred == ground_truth else 0.0
        correct = (df[inference_col] == ground_truth)
        latency = 0
        if correct.any():
            for i in range(len(correct)):
                if correct.iloc[i:].all(): latency = df['round'].iloc[i]; break

        logging.info(f"\n--- [{attack_type.upper()} Attack Results] ---")
        logging.info(f"Final Accuracy: {accuracy:.2f} (Predicted: {final_pred})")
        logging.info(f"Inference Latency: {latency} rounds to stable, correct prediction.")

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        score_cols = [col for col in df.columns if col.startswith(score_prefix)]
        for col in score_cols:
            obj_name = col.replace(score_prefix, '')
            ax.plot(df['round'], df[col], label=obj_name,
                    linestyle='-' if obj_name == ground_truth else '--',
                    linewidth=3.0 if obj_name == ground_truth else 1.5)

        ax.axvline(x=latency, color='r', linestyle=':', linewidth=2, label=f'Latency ({latency} rounds)')
        ax.set_ylabel('Inference Score')
        ax.set_xlabel('Training Round')
        ax.set_title(
            f'Buyer Intent Inference using {attack_type.replace("_", " ").title()} Method\nRun: {self.run_name}')
        ax.legend()

        save_path = f"seller_attack_{attack_type}_{self.run_name}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        logging.info(f"Plot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Run seller-side buyer intent inference analysis.")
    parser.add_argument(
        "--config",
        type=str,
        default="seller_attack_config.yaml",
        help="Path to the analysis config YAML file."
    )
    args = parser.parse_args()

    try:
        analyzer = SellerAttackAnalyzer(config_path=args.config)
        analyzer.run_analysis()
    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}", exc_info=True)
