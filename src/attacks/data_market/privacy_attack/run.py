import argparse
import datetime
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from src.attacks import get_error_under_budget, get_error_fixed, plot_results_utility
from src.attacks import run_reconstruction_attack_eval
from src.attacks import MaliciousDataSeller
from src.participants.seller.seller import BaseSeller
from src.common_utils import DatasetManager
from src.marketplace.market.data_market import DataMarketplaceData
from src.common_utils import SelectionStrategy


def plot_and_save_metrics(avg_metrics_by_attack, save_dir="plots"):
    """
    For each metric (total_distance, avg_distance, matching, mse), plot a separate graph
    showing the performance for each attack type over different n_selected values,
    then save the plots to files.

    :param avg_metrics_by_attack: Dictionary containing metrics for each attack type.
           Expected format:
           {
               "attack_type1": {
                   "n_selected": [...],
                   "total_distance": [...],
                   "avg_distance": [...],
                   "matching": [...],
                   "mse": [...]
               },
               "attack_type2": { ... },
               ...
           }
    :param save_dir: Directory where the plots will be saved.
    """
    import os
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["total_distance", "avg_distance", "mse"]

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        lines = []

        for attack, values in avg_metrics_by_attack.items():
            n_selected = values.get("n_selected", [])
            metric_values = values.get(metric, [])

            if n_selected and metric_values:
                line, = plt.plot(n_selected, metric_values, marker='o', label=str(attack))
                lines.append(line)

        plt.xlabel("n_selected")
        plt.ylabel(metric)
        plt.title(f"Average {metric} vs n_selected")
        if lines:
            plt.legend(title="Attack Type")
        else:
            print(f"No valid data to plot for metric: {metric}")

        plt.grid(True)

        # Build the filename and save the figure
        filename = os.path.join(save_dir, f"{metric}_vs_n_selected.png")
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved plot to {filename}")

        # Optionally, if you also want to show the plot, uncomment the following line:
        # plt.close()
        plt.close()


def compute_avg_metrics_by_n_selected_and_attack(filename):
    """
    Reads the CSV file and computes the average metrics for each unique combination of n_selected and attack_type.
    The averages are computed over all query_no values.

    The final result is a dictionary mapping each attack_type to a dictionary containing:
      - 'n_selected': A list of n_selected values (ascending order)
      - 'total_distance': A list of averaged total_distance values
      - 'avg_distance': A list of averaged avg_distance values
      - 'matching': A list of averaged matching values
      - 'mse': A list of averaged mse values

    :param filename: Name (or path) of the CSV file.
    :return: Dictionary containing the grouped average metrics by attack_type.
    """
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(filename)

    # Group by both n_selected and attack_type, compute the mean for the metrics.
    grouped = df.groupby(["n_selected", "attack_type"]).agg({
        "total_distance": "mean",
        "avg_distance": "mean",
        "mse": "mean"
    }).reset_index()

    # Initialize a dictionary to store results per attack type.
    result = {}

    # Get unique attack types.
    attack_types = grouped["attack_type"].unique()

    # Process each attack type separately.
    for attack in attack_types:
        # Filter for the current attack type.
        attack_df = grouped[grouped["attack_type"] == attack].sort_values("n_selected")

        result[attack] = {
            "total_distance": attack_df["total_distance"].tolist(),
            "avg_distance": attack_df["avg_distance"].tolist(),
            "mse": attack_df["mse"].tolist()
        }

    return result


def save_attack_results(results_list, save_path, file_prefix="attack_results"):
    """
    Convert a list of dictionaries to a DataFrame and save to a timestamped CSV file.

    :param results_list: List of dictionaries containing attack results.
    :param file_prefix: Prefix for the output file name.
    :return: The filename of the saved CSV file.
    """
    # Convert list to DataFrame
    df = pd.DataFrame(results_list)

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the filename
    filename = f"{save_path}/{file_prefix}_{timestamp}.csv"

    # Save DataFrame to CSV
    df.to_csv(filename, index=False)

    print(f"Attack results saved as {filename}")
    return filename


def plot_attacks_for_single_k(df, k_value, metric='selection_f1', title=None):
    """
    Plots a bar chart comparing different methods on a single metric (e.g., 'selection_f1')
    for a fixed selection size k_value.

    Args:
        df       : DataFrame with columns ['method', 'k', <metric>].
        k_value  : The selection size we want to filter on.
        metric   : The metric column name to plot (string).
        title    : Optional plot title.
    """
    # Filter for rows where df['k'] == k_value
    df_k = df[df['k'] == k_value].copy()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_k, x='method', y=metric)
    plt.title(title if title else f"{metric} for k={k_value}")
    plt.xlabel("Method")
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha='right')
    plt.ylim(0, df_k[metric].max() * 1.1)  # give some space on top
    plt.grid(True, axis='y')
    plt.close()


def plot_metric_across_selection_sizes(df, metric='selection_f1', title=None):
    """
    Plots how different methods perform on a single metric (e.g., 'selection_f1')
    as the selection size 'k' varies.

    Args:
        df     : a pandas DataFrame with columns ['method', 'k', <metric>]
        metric : the metric column name to plot on the y-axis (string)
        title  : optional plot title
    """
    # Convert 'k' to numeric if needed
    df['k'] = df['k'].astype(int)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='k', y=metric, hue='method', marker='o')
    plt.title(title if title else f"{metric} vs. Selection Size (k)")
    plt.xlabel("Selection Size (k)")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend(title='Method', loc='best')
    plt.close()


def run_attack_experiment(dataset_type="gaussian", dim=100, num_seller=1000,
                          num_buyer=100,
                          adversary_ratio=0.25, seller_configs=None,
                          selection_method=SelectionStrategy.DAVED_MULTI_STEP, attack_method="",
                          max_eval_range_selection_num=500, eval_step=50, buyer_size=1, use_cost=False, device="cpu",
                          num_restarts=1):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f'./result/{dataset_type}/total_sell_{num_seller}_adv_ratio_{adversary_ratio}/{timestamp}/'

    Path(result_path).mkdir(parents=True, exist_ok=True)
    data_manager = DatasetManager(
        dataset_type=dataset_type,
        num_seller=num_seller,
        num_buyer=num_buyer,
        dim=dim,
        use_cost=False
    )

    eval_range = list(
        range(10, max_eval_range_selection_num, eval_step)
    )

    marketplace, seller_dict = setup(data_manager, adversary_ratio, seller_configs)

    # --- Attack & Evaluation ---
    results = []

    selection_errors = defaultdict(list)
    attack_result = defaultdict(list)
    x_s, y_s, costs, seller_ids = marketplace.get_current_market_data()
    for i, j in tqdm(enumerate(range(0, num_buyer, buyer_size))):
        print(f"Running attack, Current Buyer no: {i}/{num_buyer // buyer_size}")
        x_buy = data_manager.X_buy[j: j + buyer_size]
        y_buy = data_manager.y_buy[j: j + buyer_size]
        weights, seller_ids = marketplace.get_select_info(x_buy, y_buy,
                                                          selection_method)

        err_kwargs = dict(
            x_test=x_buy, y_test=y_buy, x_s=x_s, y_s=y_s, eval_range=eval_range
        )

        if use_cost:
            print("Current using cost")
            error_func = get_error_under_budget
            err_kwargs["costs"] = costs
        else:
            print("Not using cost")
            error_func = get_error_fixed
            err_kwargs["return_list"] = True

        selection_error = error_func(w=weights, **err_kwargs)
        selection_errors["DAVED"].append(selection_error)
        for k in eval_range:  # Ensure k doesn't exceed data size
            print(f"Running attack, change selection num: {k}/{max_eval_range_selection_num}")
            # Select top-k adversarial samples
            selected_indices = np.argsort(weights)[-k:]
            weights_tensor = torch.tensor(weights, dtype=torch.float)

            # Get current market data once per iteration

            # Run different reconstruction attacks

            base_random = run_reconstruction_attack_eval(x_s, selected_indices, x_buy,
                                                         use_baseline="random")
            base_centroid = run_reconstruction_attack_eval(x_s, selected_indices, x_buy,
                                                           use_baseline="centroid")
            score_known = run_reconstruction_attack_eval(x_s, selected_indices, x_buy,
                                                         scenario="score_known", observed_scores=weights_tensor,
                                                         num_restarts=num_restarts,
                                                         device=device)

            # score_unknown_ranking_hinge = run_reconstruction_attack_eval(x_s, selected_indices,
            #                                                              x_buy,
            #                                                              scenario="selection_only",
            #                                                              attack_method="ranking",
            #                                                              ranking_loss_type="hinge",
            #                                                              num_restarts=num_restarts,
            #                                                              device=device)
            # score_unknown_ranking_logistic = run_reconstruction_attack_eval(x_s, selected_indices,
            #                                                                 x_buy,
            #                                                                 scenario="selection_only",
            #                                                                 attack_method="ranking",
            #                                                                 ranking_loss_type="logistic",
            #                                                                 device=device)
            # score_unknown_topk = run_reconstruction_attack_eval(x_s, selected_indices,
            #                                                     x_buy,
            #                                                     scenario="selection_only", attack_method="topk",
            #                                                     num_restarts=num_restarts,
            #                                                     device=device)

            def append_result(attack_name, result_dict):
                """
                Copy the result dictionary, add metadata, and append to the results list.
                """
                # Make a copy to avoid modifying the original dictionary.
                entry = result_dict.copy()
                entry["query_no"] = i
                entry["attack_type"] = attack_name
                entry["n_selected"] = k
                results.append(entry)

            # Append each attack's results.
            append_result("random", base_random)
            append_result("centroid", base_centroid)
            append_result("score_known", score_known)
            # append_result("score_unknown_ranking_hinge", score_unknown_ranking_hinge)
            # append_result("score_unknown_ranking_logistic", score_unknown_ranking_logistic)
            # append_result("score_unknown_topk", score_unknown_topk)

    plot_results_utility(result_path, {"errors": selection_errors, "eval_range": eval_range},
                         None)

    # After the loop, save all accumulated results to a single CSV file.
    filename = save_attack_results(results, save_path=result_path)
    ave_attack_res = compute_avg_metrics_by_n_selected_and_attack(filename)
    plot_and_save_metrics(ave_attack_res, save_dir=result_path)
    return results


def plot_results(eval_results, eval_range):
    ks = list(eval_range)[:len(eval_results["our"])]  # Match evaluated k values

    plt.figure(figsize=(12, 4))

    # Plot Cosine Similarity
    plt.subplot(131)
    plt.plot(ks, [res["cosine_sim_mean"] for res in eval_results["our"]], label="Our Method")
    plt.plot(ks, [res["cosine_sim_mean"] for res in eval_results["baseline"]], label="Baseline (Mean)")

    plt.xlabel("Number of Selected Samples (k)")
    plt.ylabel("Cosine Similarity mean")
    plt.legend()

    # plt.subplot(132)
    # plt.plot(ks, [res["cosine_sim_std"] for res in eval_results["our"]], label="Our Method")
    # plt.plot(ks, [res["cosine_sim_std"] for res in eval_results["baseline"]], label="Baseline (Mean)")
    # plt.xlabel("Number of Selected Samples (k)")
    # plt.ylabel("Cosine Similarity std")
    # plt.legend()

    # Plot MSE
    plt.subplot(132)
    plt.plot(ks, [res["mse"] for res in eval_results["our"]], label="Our Method")
    plt.plot(ks, [res["mse"] for res in eval_results["baseline"]], label="Baseline (Mean)")
    plt.xlabel("Number of Selected Samples (k)")
    plt.ylabel("MSE")
    plt.legend()

    # Plot Neighborhood Overlap
    plt.subplot(133)
    plt.plot(ks, [res["neighborhood_overlap"] for res in eval_results["our"]], label="Our Method")
    plt.plot(ks, [res["neighborhood_overlap"] for res in eval_results["baseline"]], label="Baseline (Mean)")
    plt.xlabel("Number of Selected Samples (k)")
    plt.ylabel("Neighborhood Overlap")
    plt.legend()

    plt.tight_layout()
    plt.close()


def setup(data_manager: DatasetManager, adversary_ratio=0.25, seller_configs=None):
    """Setup marketplace with normal and adversarial sellers"""

    # Create marketplace
    marketplace = DataMarketplaceData()
    # Get data allocations

    allocations = data_manager.allocate_data_to_sellers(
        seller_configs,
        adversary_ratio=adversary_ratio
    )
    seller_dict = {}
    # Create and register sellers
    for config in seller_configs:
        seller_id = config['id']
        seller_data = allocations[seller_id]

        if config['type'] == 'adversary':
            seller = MaliciousDataSeller(
                seller_id=seller_id,
                dataset=seller_data['X'],
            )
        else:
            seller = BaseSeller(
                seller_id=seller_id,
                dataset=seller_data['X']
            )
        seller_dict[seller_id] = seller
        marketplace.register_seller(seller_id, seller)

    return marketplace, seller_dict


# if __name__ == "__main__":
#     # Run experiments
#
#     exp_config = {
#         "adversary_ratio": 1,
#         "num_seller_points": 1000,
#         "num_buyer_points": 100,
#         "seller_configs": [
#             {'id': 'adv1', 'type': 'adversary'},
#             # {'id': 'normal1', 'type': 'normal'},
#             # {'id': 'normal2', 'type': 'normal'},
#             # {'id': 'normal3', 'type': 'normal'}
#         ]
#     }
#
#     # Create a mapping for the parameter names
#     param_mapping = {
#         "num_seller": exp_config.get("num_seller_points", 1000),
#         "num_buyer": exp_config.get("num_buyer_points", 100),
#         "adversary_ratio": exp_config.get("adversary_ratio", 0.0),
#         "seller_configs": exp_config.get("seller_configs", [])
#     }
#
#     eval_results = run_attack_experiment(
#         dataset_type="fitzpatrick",
#         dim=100,
#         **param_mapping  # Unpack the mapped parameters
#     )
#
#     # Plot results
#     plot_results(eval_results)
#     print(eval_results)

def parse_args():
    """
    Parse command-line arguments for the experiment configuration.

    :return: Dictionary containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run attack experiment with configurable parameters.")

    # Add arguments with default values from exp_config
    parser.add_argument("--dataset", type=str, default="gaussian", help="gaussian mimic fitzpatrick bone")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device,
                        help="Device to run on, e.g. 'cuda' or 'cpu' (default: cuda if available)")

    parser.add_argument("--max_eval_range_selection_num", type=int, default=300, help="Number of seller points")
    parser.add_argument("--eval_step", type=int, default=50, help="Number of seller points")
    parser.add_argument("--num_seller", type=int, default=1000, help="Number of seller points")
    parser.add_argument("--num_buyer", type=int, default=30, help="Number of buyer points")
    parser.add_argument("--buyer_size", type=int, default=1, help="size of buyer points")

    parser.add_argument("--adversary_ratio", type=float, default=1.0, help="Adversary ratio in the dataset")
    parser.add_argument("--seller_configs", type=str, default="adv1:adversary",
                        help="Comma-separated list of seller configurations (format: id:type, e.g., adv1:adversary,normal1:normal)")

    args = parser.parse_args()

    # Process seller_configs into a list of dictionaries
    seller_configs = []
    if args.seller_configs:
        for seller in args.seller_configs.split(","):
            seller_id, seller_type = seller.split(":")
            seller_configs.append({'id': seller_id, 'type': seller_type})

    # Return as a dictionary
    return {
        "num_seller": args.num_seller,
        "num_buyer": args.num_buyer,
        "adversary_ratio": args.adversary_ratio,
        "seller_configs": seller_configs
    }, args


if __name__ == "__main__":
    # Parse command-line arguments
    param_mapping, args = parse_args()
    print("start reconstruction attack")
    print(param_mapping)

    eval_range = list(
        range(10, args.max_eval_range_selection_num, args.eval_step)
    )
    # Run experiment with parsed arguments
    eval_results = run_attack_experiment(
        dataset_type=args.dataset,
        dim=100,
        device=args.device,
        buyer_size=args.buyer_size,
        max_eval_range_selection_num=args.max_eval_range_selection_num, eval_step=args.eval_step, num_restarts=10,
        **param_mapping
        # Unpack arguments
    )

    # Plot and print results
    # plot_results(eval_results, eval_range)
    # print(eval_results)
