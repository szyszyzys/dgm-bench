import logging
import warnings
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# --- Import your project's utilities and loader ---
from tqdm import tqdm
from src.attacks.gradient_market.privacy_attack.utils import get_buyer_dataset, get_seller_ground_truth

from src.common_utils import ExperimentLoader

warnings.filterwarnings("ignore", category=FutureWarning)


def get_seller_data_properties(run_data: Dict[str, Any], seller_id: str) -> Dict[str, float]:
    """
    Calculates the ground-truth properties of a seller's private dataset.
    For this example, we calculate the proportion of each class.

    Returns:
        A dictionary mapping property names (e.g., 'prop_class_0') to their values.
    """
    seller_gt = get_seller_ground_truth(run_data, seller_id)
    member_indices = seller_gt.get("member_indices", [])

    if not member_indices:
        return {}

    full_dataset = get_buyer_dataset(run_data)
    labels = [label for _, label in full_dataset]

    seller_labels = [labels[i] for i in member_indices]

    properties = {}
    num_labels = len(seller_labels)
    unique_classes = sorted(list(set(labels)))

    for class_idx in unique_classes:
        count = seller_labels.count(class_idx)
        properties[f'prop_class_{class_idx}'] = count / num_labels if num_labels > 0 else 0.0

    return properties


def build_sybil_adversary_dataframe(run_data: Dict[str, Any], adversary_seller_ids: List[str]) -> pd.DataFrame:
    """
    Constructs a DataFrame from the perspective of a Sybil adversary controlling multiple sellers.
    It aggregates marketplace signals and data properties across all Sybil identities.
    """
    logging.info(f"Building dataframe for Sybil adversary controlling {len(adversary_seller_ids)} sellers...")

    # 1. Calculate the AGGREGATE properties of the adversary's entire data pool
    all_adversary_indices = []
    for seller_id in adversary_seller_ids:
        seller_gt = get_seller_ground_truth(run_data, seller_id)
        all_adversary_indices.extend(seller_gt.get("member_indices", []))

    full_dataset = get_buyer_dataset(run_data)
    labels = [label for _, label in full_dataset]
    adversary_labels = [labels[i] for i in all_adversary_indices]

    aggregate_properties = {}
    num_labels = len(adversary_labels)
    unique_classes = sorted(list(set(labels)))
    for class_idx in unique_classes:
        count = adversary_labels.count(class_idx)
        aggregate_properties[f'prop_class_{class_idx}'] = count / num_labels if num_labels > 0 else 0.0

    # 2. Aggregate marketplace signals per round
    adversary_round_data = []
    # Assuming all sellers run for the same number of rounds
    num_rounds = len(run_data['sellers'][adversary_seller_ids[0]])

    for round_num in tqdm(range(num_rounds), desc="Aggregating Sybil Data"):
        total_payment_in_round = 0.0
        total_weight_in_round = 0.0

        for seller_id in adversary_seller_ids:
            history = run_data['sellers'][seller_id]
            round_info = history[history['round'] == round_num]
            if not round_info.empty:
                total_payment_in_round += round_info['payment_received'].iloc[0]
                total_weight_in_round += round_info['assigned_weight'].iloc[0]

        round_entry = {
            'round': round_num,
            'payment_received': total_payment_in_round,
            'assigned_weight': total_weight_in_round
        }
        round_entry.update(aggregate_properties)
        adversary_round_data.append(round_entry)

    return pd.DataFrame(adversary_round_data)


def analyze_and_plot_correlation(df: pd.DataFrame, run_name: str, is_sybil_attack: bool = False):
    """
    Calculates and visualizes the correlation matrix between marketplace signals
    and data properties.
    """
    attack_type = "Sybil Adversary" if is_sybil_attack else "Individual Seller"
    logging.info(f"\n--- 📊 FINAL ANALYSIS REPORT (Task 3.3 - {attack_type}) 📊 ---")

    if df.empty:
        logging.warning("DataFrame is empty. Cannot perform correlation analysis.")
        return

    # Define the two sets of variables for correlation
    marketplace_signals = ['payment_received', 'assigned_weight']
    data_properties = [col for col in df.columns if col.startswith('prop_class_')]

    if not data_properties:
        logging.warning("No data property columns found. Cannot perform correlation analysis.")
        return

    correlation_matrix = df[marketplace_signals + data_properties].corr()

    # Extract the specific correlations of interest
    leakage_correlation = correlation_matrix.loc[marketplace_signals, data_properties]

    print(f"--- System-Level Leakage Correlation Matrix ({attack_type}) ---")
    print("This matrix shows the correlation between public marketplace signals (rows)")
    print("and private seller data properties (columns).")
    print("Values close to +1 or -1 indicate significant information leakage.")
    print(leakage_correlation)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        leakage_correlation,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        fmt=".3f"
    )
    title = (f'System-Level Leakage: {attack_type} Perspective\n'
             f'Correlation of Aggregated Marketplace Signals vs. Aggregated Data Properties\nRun: {run_name}')
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = f"task3_3_sybil_leakage_correlation_{run_name}.png"
    plt.savefig(save_path)
    logging.info(f"Correlation heatmap saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    EXPERIMENT_ROOT = "./exp_results/text_agnews_cnn_10seller"  # IMPORTANT: Set to your experiment folder
    TARGET_RUN_INDEX = 0

    # --- 1. Load Experiment Data ---
    try:
        loader = ExperimentLoader(EXPERIMENT_ROOT)
        all_runs = loader.load_all_runs_data()
        target_run_data = all_runs[TARGET_RUN_INDEX]
        run_name = target_run_data['run_path'].name
        logging.info(f"Loaded data for run: {run_name}")
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load experiment data: {e}")
        exit()

    # --- 2. Build the Analysis DataFrame for a Sybil Adversary ---
    try:
        # Identify all sellers controlled by the adversary (e.g., by name convention)
        all_seller_ids = list(target_run_data['sellers'].keys())
        adversary_seller_ids = [sid for sid in all_seller_ids if sid.startswith('adv')]

        if not adversary_seller_ids:
            logging.warning("No adversary sellers found (e.g., named 'adv_...'). Skipping Sybil analysis.")
        else:
            sybil_correlation_df = build_sybil_adversary_dataframe(target_run_data, adversary_seller_ids)

            # --- 3. Analyze and Visualize the Correlation from the Sybil perspective ---
            analyze_and_plot_correlation(sybil_correlation_df, run_name, is_sybil_attack=True)

    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
