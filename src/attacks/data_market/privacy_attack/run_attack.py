# CLIP model and processor
# Import your custom modules or utilities

import argparse
import os
from collections import defaultdict

import clip
import numpy as np
import torch
from tqdm import tqdm

# CLIP model and processor
from src.mechanism.data.discovery.daved import src as frank_wolfe
# Import your custom modules or utilities
from src.attacks import get_data
# CLIP model and processor
# Import your custom modules or utilities
from src.attacks import save_results_pkl
from src.attacks import fim_reverse_math
from src.attacks import fim_reverse_emb_opt_v2


def identify_selected_unsampled(weights, num_select=10):
    """
    Identify which data points are selected and which are unselected based on weights.

    Parameters:
    - weights (np.ndarray): Weights assigned to each data point.
    - num_select (int): Number of data points to select.

    Returns:
    - selected_indices (set): Indices of selected data points.
    - unsampled_indices (list): Indices of unselected data points.
    """
    selected_indices = set(weights.argsort()[::-1][:num_select])
    unsampled_indices = list(set(range(len(weights))) - selected_indices)
    return list(selected_indices), unsampled_indices


def sampling_run_one_buyer(x_b, y_b, x_s, y_s, eval_range, costs=None, args=None, figure_path="./figure",
                           img_paths=None, test_img_indices=None, sell_img_indices=None):
    # Dictionaries to store errors, runtimes, and weights for each method and test point
    test_point_info = []  # To store details for each test point evaluation

    # Loop over each test point in buyer's data, in batches
    for i, j in tqdm(enumerate(range(0, x_b.shape[0], args.batch_size))):
        # Get batch of test points
        x_test = x_b[j: j + args.batch_size]
        y_test = y_b[j: j + args.batch_size]

        # Perform single-step optimization (DAVED single step)
        w_os = frank_wolfe.one_step(x_s, x_test)

        # Perform multi-step optimization (DAVED multi-step)
        res_fw = frank_wolfe.design_selection(
            x_s,
            y_s,
            x_test,
            y_test,
            num_select=10,
            num_iters=args.num_iters,
            alpha=None,
            recompute_interval=0,
            line_search=True,
            costs=costs,
            reg_lambda=args.reg_lambda,
        )

        # Store runtime, weights, and errors for multi-step
        w_fw = res_fw["weights"]

        # Store information about the test point, the indices used, and their results
        test_point_info.append({
            "query_number": i,
            "test_point_start_index": j,
            "test_x": x_test,
            "test_y": y_test,
            "single_step_weights": w_os,
            "multi_step_weights": w_fw,
            "eval_range": eval_range
        })

    return test_point_info


def evaluate_privacy_attack(
        args,
        dataset='./data',
        data_dir='./data',
        csv_path="druglib/druglib.csv",
        img_path="/images",
        num_buyer=10,
        num_seller=1000,
        num_val=1,
        num_dim=100,
        max_eval_range=50,
        eval_step=5,
        num_iters=500,
        reg_lambda=0.1,
        attack_strength=0.1,
        save_results_flag=True,
        result_dir='results',
        save_name='attack_evaluation',
        num_select=100,
        attack_type="data",
        adversary_ratio=0.25,
        emb_model=None,
        img_preprocess=None,
        cost_manipulation_method="undercut_target",
        emb_model_name="clip",
        attack_method="math",
        device="cpu",
        n_select=20,
        **kwargs
):
    """
    Run the attack evaluation experiment.

    Parameters:
    - data_dir (str): Directory where the CSV file is located.
    - csv_path (str): Path to the CSV file containing the data.
    - num_buyer (int): Number of buyer data points.
    - num_seller (int): Number of seller data points.
    - num_val (int): Number of validation data points.
    - max_eval_range (int): Maximum budget value for evaluation.
    - eval_step (int): Step size for budget evaluation.
    - num_iters (int): Number of iterations for the selection algorithm.
    - reg_lambda (float): Regularization parameter for the selection algorithm.
    - attack_strength (float): Strength of the attack when modifying features.
    - save_results_flag (bool): Whether to save the results.
    - result_dir (str): Directory to save the results.
    - save_name (str): Filename for the saved results.

    Returns:
    - results (dict): Dictionary containing evaluation metrics and other relevant data.
    """
    # fim_reverse_emb_opt_v2(x_s, selected_indices, unselected_indices, x_query, device)
    if attack_method == "math":
        attack_func = fim_reverse_math
    elif attack_method == "opt2":
        attack_func = fim_reverse_emb_opt_v2
    else:
        print(f"Error attack method: {attack_method}".center(40, "="))
        return

    # Step 1: Load and preprocess data
    data = get_data(
        dataset=dataset,
        num_buyer=num_buyer * args.batch_size,
        num_seller=num_seller,
        num_val=num_val,
        dim=num_dim,
        noise_level=args.noise_level,
        random_state=args.random_seed,
        cost_gen_mode=args.cost_gen_mode,
        use_cost=args.use_cost,
        cost_func=args.cost_func,
        assigned_cost=None,
    )
    # Extract relevant data
    x_s = data["X_sell"].astype(np.float32)
    y_s = data["y_sell"].astype(np.float32)
    x_b = data["X_buy"].astype(np.float32)
    y_b = data["y_buy"].astype(np.float32)

    result_dir = f'{result_dir}/privacy/attack_{attack_method}_num_buyer_{num_buyer}_num_seller_{num_seller}_querys_{args.batch_size}/'
    figure_path = f"{result_dir}/figures/"

    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)

    # todo change the costs
    costs = data.get("costs_sell")
    index_s = data['index_sell']
    index_b = data['index_buy']
    index_v = data['index_val']
    img_paths = data['img_paths']
    print("Data type of index_s:", len(img_paths))
    print(f"Seller Data Shape: {x_s.shape}".center(40, "="))
    print(f"Buyer Data Shape: {x_b.shape}".center(40, "="))
    if costs is not None:
        print(f"Costs Shape: {costs.shape}".center(40, "="))

    # adversary preparation, sample partial data for the adversary
    num_adversary_samples = int(len(x_s) * adversary_ratio)
    adversary_indices = np.random.choice(len(x_s), size=num_adversary_samples, replace=False)

    # Evaluate the peformance
    eval_range = list(
        range(30, args.max_eval_range, args.eval_step)
    )

    benign_node_sampling_result_dict = {}
    benign_selection_info = sampling_run_one_buyer(x_b, y_b, x_s, y_s, eval_range, costs=costs,
                                                   test_img_indices=index_b,
                                                   sell_img_indices=index_s,
                                                   args=args, img_paths=img_paths,
                                                   figure_path=figure_path)

    # transform the initial result into dictionary
    for cur_info in benign_selection_info:
        query_number = cur_info["query_number"]
        benign_node_sampling_result_dict[query_number] = cur_info

    print(f"Done initial run, number of queries: {len(benign_selection_info)}")
    # Step 3: Identify Selected and Unselected Data Points
    # For each batch (buyer query), perform the reconstruction
    attack_result_dict = defaultdict()

    # todo current use fixed number
    # For different query, perform the attacks.
    for query_n, info_dic in enumerate(benign_selection_info):
        cur_query_num = info_dic["query_number"]
        m_cur_weight = info_dic["multi_step_weights"]
        s_cur_weight = info_dic["single_step_weights"]
        query_x = info_dic["test_x"]
        query_y = info_dic["test_y"]
        attack_result_path = f"./{figure_path}/poisoned_sampling_query_number_{query_n}"
        selected_indices_initial, unsampled_indices_initial = identify_selected_unsampled(
            weights=m_cur_weight,
            num_select=n_select,
        )

        # x_s, selected_indices, unselected_indices, x_query, device
        selected_adversary_indices = np.intersect1d(selected_indices_initial, adversary_indices)
        unsampled_adversary_indices = np.intersect1d(unsampled_indices_initial, adversary_indices)
        cur_res = attack_func(x_s, selected_adversary_indices, unsampled_adversary_indices, query_x, device, True,
                              attack_result_path)

        # results = {
        #     "best_cosine_similarities": best_cosine_similarities,
        #     "best_euclidean_distances": best_euclidean_distances,
        #     "matching_indices": matching_indices
        # }
        attack_result_dict[query_n] = cur_res

    save_results_pkl(attack_result_dict, result_dir)
    # for n_select in eval_range:
    # torch.save(atga)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Data subset selection with experimental design informed loss using Frank-Wolfe optimization",
    )
    parser.add_argument("--random_seed", default=12345, help="random seed")
    parser.add_argument("--figure_dir", default="./figures")
    parser.add_argument("--result_dir", default="./results")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument(
        "-d",
        "--dataset",
        default="fitzpatrick",
        choices=[
            "gaussian",
            "mimic",
            "bone",
            "fitzpatrick",
            "drug",
        ],
        type=str,
        help="dataset to run experiment on",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="number of test points to optimize at once",
    )
    parser.add_argument(
        "--num_buyers",
        default=30,
        type=int,
        help="number of test buyer points used in experimental design",
    )
    parser.add_argument(
        "--num_seller",
        default=1000,
        type=int,
        help="number of seller points used in experimental design",
    )
    parser.add_argument(
        "--num_val",
        default=100,
        type=int,
        help="number of validation points for baselines",
    )
    parser.add_argument(
        "--num_dim",
        default=100,
        type=int,
        help="dimensionality of the data samples",
    )
    parser.add_argument(
        "--num_select",
        default=500,
        type=int,
        help="dimensionality of the data samples",
    )

    parser.add_argument(
        "--num_iters",
        default=500,
        type=int,
        help="number of iterations to optimize experimental design",
    )
    parser.add_argument(
        "--max_eval_range",
        default=150,
        type=int,
        help="max number training points to select for evaluation",
    )
    parser.add_argument(
        "--eval_step",
        default=5,
        type=int,
        help="evaluation interval",
    )

    parser.add_argument(
        "--attack_reg",
        default=0,
        type=float,
        help="attack reg",
    )
    parser.add_argument(
        "--attack_lr",
        default=0.1,
        type=float,
        help="attack lr",
    )

    parser.add_argument(
        "--baselines",
        nargs="*",
        default=[
            # "AME",
            # "BetaShapley",
            # "DataBanzhaf",
            "DataOob",
            # "DataShapley",
            "DVRL",
            # "InfluenceSubsample",
            "KNNShapley",
            "LavaEvaluator",
            # "LeaveOneOut",
            "RandomEvaluator",
            # "RobustVolumeShapley",
        ],
        type=str,
        help="Compare to other data valution baselines in opendataval",
    )
    parser.add_argument("--debug", action="store_true", help="Turn on debugging mode")
    parser.add_argument("--use_cost", action="store_true", help="If use cost")

    parser.add_argument(
        "--skip_save", action="store_true", help="Don't save weights or data"
    )

    parser.add_argument(
        "--cost_range",
        nargs="*",
        default=None,
        help="""
    Choose range of costs to sample uniformly.
    E.g. costs=[1, 2, 3, 9] will randomly set each seller data point
    to one of these costs and apply the cost_func during optimization.
    If set to None, no cost is applied during optimization.
    Default is None.
    """,
    )
    parser.add_argument(
        "--cost_func",
        default="linear",
        choices=["linear", "squared", "square_root"],
        type=str,
        help="Choose cost function to apply.",
    )

    parser.add_argument(
        "--save_name",
        default="robustness_test",
        type=str,
        help="save_name.",
    )
    parser.add_argument(
        "--cost_gen_mode",
        default="constant",
        choices=["random_uniform", "random_choice", "constant"],
        type=str,
        help="Choose cost function to apply.",
    )
    parser.add_argument(
        "--attack_method",
        default="math",
        choices=["math", "opt1", "opt2"],
        type=str,
        help="Choose cost function to apply.",
    )

    parser.add_argument(
        "--noise_level",
        default=0,
        type=float,
        help="level of noise to add for cost",
    )
    parser.add_argument(
        "--reg_lambda",
        default=0.0,
        type=float,
        help="Regularization initial inverse information matrix to be identity. Should be between 0 and 1.",
    )

    parser.add_argument(
        "--poison_rate",
        default=0.2,
        type=float,
        help="rate of images to do poison for a buyer",
    )

    parser.add_argument(
        "--attack_steps",
        default=200,
        type=int,
        help="attack step",
    )
    parser.add_argument(
        "--num_data_select",
        default=200,
        type=int,
        help="num_data_select",
    )

    args = parser.parse_args()
    save_name = f'attack_evaluation_{args.dataset}_b_{args.num_buyers}_s_{args.num_seller}'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_model, preprocess = clip.load("ViT-B/32", device=device)
    emb_model.eval()  # Set model to evaluation mode

    # Define experiment parameters
    experiment_params = {
        'dataset': args.dataset,
        'data_dir': './data',
        'csv_path': "./fitzpatrick17k/fitzpatrick-mod.csv",
        'img_path': './fitzpatrick17k/images',
        'num_buyer': args.num_buyers,
        'num_seller': args.num_seller,
        'num_val': args.num_val,
        'max_eval_range': args.max_eval_range,
        'eval_step': args.eval_step,
        'num_iters': args.num_iters,
        'reg_lambda': 0.1,
        'attack_strength': 0.1,
        'save_results_flag': True,
        'result_dir': 'results',
        'save_name': save_name,
        "num_select": args.num_select,
        "adversary_ratio": 0.25,
        "emb_model": emb_model,
        "img_preprocess": preprocess,
        "emb_model_name": "clip",
        "attack_method": args.attack_method,
        "device": device,
        "n_select": args.num_data_select,

    }

    # Run the attack evaluation experiment
    results = evaluate_privacy_attack(args, **experiment_params)

    # Print final results
    print("\nFinal Attack Evaluation Metrics:")
    # print(f"Attack Success Rate: {results['attack_success_rate'] * 100:.2f}%")
    # print(f"Number of Successfully Selected Modified Data Points: {results['num_successfully_selected']}")
