import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torchvision.models import resnet18
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda
from tqdm import tqdm

from src.mechanism.data import get_gaussian_data, get_mimic_data, get_fitzpatrick_data, get_bone_data, get_drug_data, \
    split_data, get_cost_function


def read_csv(filename):
    """
    Read the attack results from a CSV file into a Pandas DataFrame.

    :param filename: The name of the CSV file to read.
    :return: Pandas DataFrame containing the attack results.
    """
    try:
        df = pd.read_csv(filename)
        print(f"Successfully loaded results from {filename}")
        return df
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None


def manipulate_cost_function(costs, method="scale", factor=1.0, offset=0.0, power=1.0):
    """
    Manipulate the cost function to apply different transformations.

    Parameters:
    - costs (np.ndarray): Array of original costs for each data point.
    - method (str): Method to manipulate the cost function. Options:
        - "scale": Scale costs by a factor.
        - "offset": Add a constant offset to costs.
        - "power": Raise costs to a specified power (non-linear transformation).
        - "log": Apply a logarithmic transformation to costs (assumes positive values).
        - "combine": Combine scale, offset, and power transformations.
    - factor (float): Scaling factor for "scale" or "combine" methods.
    - offset (float): Offset to add for "offset" or "combine" methods.
    - power (float): Power to raise costs to for "power" or "combine" methods.

    Returns:
    - np.ndarray: Manipulated costs after applying the specified transformation.
    """

    if method == "scale":
        # Scale costs by the specified factor
        manipulated_costs = costs * factor

    elif method == "offset":
        # Add a constant offset to costs
        manipulated_costs = costs + offset

    elif method == "power":
        # Raise costs to the specified power
        manipulated_costs = np.power(costs, power)

    elif method == "log":
        # Apply a logarithmic transformation to costs (assumes costs are positive)
        manipulated_costs = np.log(costs + 1)  # Adding 1 to avoid log(0)

    elif method == "combine":
        # Apply a combination of scaling, offsetting, and raising to a power
        manipulated_costs = (costs * factor + offset) ** power

    else:
        raise ValueError("Invalid method for manipulating the cost function.")

    return manipulated_costs


def generate_costs(X=None, method="random_uniform", cost_range=(1.0, 5.0), assigned_cost=None):
    """
    Generate different types of costs for data points.

    Parameters:
    - X (np.ndarray): Feature matrix (n_samples, n_features).
    - method (str): Method for generating costs. Options:
        - "random_uniform": Random values within cost_range.
        - "random_choice": Random choice from cost_range.
        - "feature_based": Scale based on a feature (first feature by default).
        - "constant": Assign the same cost to all data points.
        - "assigned": Use pre-assigned costs.
    - cost_range (tuple): Range of possible costs (min, max).
    - assigned_cost (np.ndarray, optional): Pre-assigned costs for each data point.

    Returns:
    - np.ndarray: Array of costs for each data point.
    """

    if method == "random_uniform":
        # Random values uniformly distributed within cost_range
        costs = np.random.uniform(cost_range[0], cost_range[1], size=X.shape[0]).astype(np.single)

    elif method == "random_choice":
        # Randomly choose cost values from cost_range (assumes discrete set of choices)
        costs = np.random.choice(np.linspace(cost_range[0], cost_range[1], 10), size=X.shape[0]).astype(np.single)

    elif method == "feature_based":
        # Scale costs based on the first feature of X, normalized to cost_range
        feature = X[:, 0]
        costs = (feature - feature.min()) / (feature.max() - feature.min())
        costs = costs * (cost_range[1] - cost_range[0]) + cost_range[0]
        costs = costs.astype(np.single)

    elif method == "constant":
        # Assign the same cost to all data points, set at the midpoint of cost_range
        costs = np.full(X.shape[0], (cost_range[0] + cost_range[1]) // 2, dtype=np.single)

    elif method == "assigned" and assigned_cost is not None:
        # Use pre-assigned costs if provided
        costs = assigned_cost.astype(np.single)

    else:
        raise ValueError("Invalid method or assigned_cost not provided for 'assigned' method.")

    return costs


def get_data(
        dataset="gaussian",
        data_dir="./data",
        random_state=0,
        num_seller=10000,
        num_buyer=100,
        num_val=100,
        dim=100,
        noise_level=1,
        use_cost=False,
        cost_gen_mode="None",
        cost_func="linear",
        recompute_embeddings=False,
        assigned_cost=None
):
    total_samples = num_seller + num_buyer + num_val
    data_dir = Path(data_dir)
    match dataset:
        case "gaussian":
            data = get_gaussian_data(total_samples, dim=dim, noise=noise_level)
        case "mimic":
            data = get_mimic_data(total_samples, data_dir=data_dir)
        case "fitzpatrick":
            data = get_fitzpatrick_data(
                total_samples,
                recompute_embeddings=recompute_embeddings,
                data_dir=data_dir,
                model_name='clip',
            )
        case "bone":
            data = get_bone_data(
                total_samples,
                recompute_embeddings=recompute_embeddings,
                data_dir=data_dir,
                model_name='clip',
            )
        case "drug":
            data = get_drug_data(
                total_samples,
                recompute_embeddings=recompute_embeddings,
                data_dir=data_dir,
                model_name='gpt2',
            )
        case _:
            raise Exception("Dataset not found")

    X = data["X"]
    y = data["y"]
    coef = data.get("coef")
    index = data.get("index")

    if use_cost:
        costs = generate_costs(X, cost_gen_mode)
        ret = split_data(
            num_buyer, num_val, random_state=random_state, X=X, y=y, costs=costs, index=index,
        )
    else:
        ret = split_data(num_buyer, num_val, random_state=random_state, X=X, y=y, index=index)

    # dict(
    #     X_sell=X_sell,
    #     y_sell=y_sell,
    #     costs_sell=costs_sell,
    #     X_buy=X_buy,
    #     y_buy=y_buy,
    #     costs_buy=costs_buy,
    #     X_val=X_val,
    #     y_val=y_val,
    #     costs_val=costs_val,
    #     index_buy=index_buy,
    #     index_val=index_val,
    #     index_sell=index_sell,
    # )

    # ret contain the information of the data from different party

    ret['img_paths'] = data['img_paths']
    ret["coef"] = coef
    ret["use_cost"] = use_cost
    ret["cost_func"] = cost_func

    match dataset, use_cost:
        case "gaussian", True:  # gaussian, no costs
            ret["y_buy"] = ret["X_buy"] @ coef
        case _, False:  # not gaussian, no costs
            pass
        case "gaussian", True:  # gaussian with costs
            h = get_cost_function(cost_func)
            e = noise_level * np.random.randn(ret["X_sell"].shape[0])
            print(type(ret["costs_sell"]))
            ret["y_sell"] = (
                    np.einsum("i,ij->ij", h(ret["costs_sell"]), ret["X_sell"]) @ coef + e
            )
            ret["y_buy"] = ret["X_buy"] @ coef
        case _, False:  # not gaussian with costs
            h = get_cost_function(cost_func)
            e = noise_level * np.random.randn(ret["X_sell"].shape[0])
            print(f'{e[:10].round(2)=}', e.mean())
            print(f'{ret["y_sell"][:10]}   {ret["y_sell"].mean()=}')
            print(f'{h(ret["costs_sell"][:10])=}')
            e *= ret["y_sell"].mean() / h(ret["costs_sell"])
            print(f'{e[:10].round(2)=}', e.mean())
            ret["y_sell"] = ret["y_sell"] + e
            print(f'{ret["y_sell"].mean()=}')

    return ret


def load_model_and_preprocessor(model_name, device):
    """
    Load the model and preprocessing pipeline based on the model name.

    Args:
        model_name (str): Model to load ("clip" or "resnet").
        device (str): Device to load the model onto ("cpu" or "cuda").

    Returns:
        tuple: (model, preprocess, inference_func)
    """
    if model_name == "clip":
        model, preprocess = clip.load("ViT-B/32", device=device)
        inference_func = model.encode_image
    elif model_name == "resnet":
        model = resnet18(pretrained=True).to(device)
        preprocess = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
                Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Handle grayscale
            ]
        )
        inference_func = lambda x: model(x).flatten(start_dim=1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()  # Ensure the model is in evaluation mode
    return model, preprocess, inference_func


def embed_image(img, model, preprocess, inference_func, device="cpu", normalize_embeddings=True):
    """
    Embed a single image object using the given model and preprocessing pipeline.

    Args:
        img (PIL.Image.Image): Image object.
        model: The loaded model for embedding.
        preprocess: The preprocessing pipeline.
        inference_func: The inference function of the model.
        device (str): Device to run the model on ("cpu" or "cuda").
        normalize_embeddings (bool): Whether to normalize embeddings to unit vectors.

    Returns:
        torch.Tensor: The embedding of the image.
    """
    try:
        img_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
        embedding = inference_func(img_tensor)
        if normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize to unit vector
        return embedding  # Remain on the device
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


def plot_errors_fixed(results, save_path):
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 8))
    errors = results["errors"]
    eval_range = results["eval_range"]

    quantiles = []
    for i, (k, v) in enumerate(errors.items()):
        err = np.array(v)
        quantiles.append(np.quantile(err, 0.9))
        ms = 5
        match k:
            case "LavaEvaluator":
                k = "LAVA"
            case "InfluenceSubsample":
                k = "Influence"
            case "LeaveOneOut":
                k = "Leave One Out"
            case "KNNShapley":
                k = "KNN Shapley"
            case "DataOob":
                k = "Data-OOB"
            case "attack":
                k = "Attack-DEVAD"
            case _:
                k = k

        match k:
            case k if "DAVED (multi-step)" in k:
                lw = 2
                ls = "-"
                marker = "*"
                ms = ms + 5
            case k if "random" in k.lower():
                lw = 5
                ls = "-"
                marker = ""
            case _:
                lw = 2
                ls = "-"
                marker = "s"
        plt.plot(
            eval_range,
            err.mean(0).squeeze(),
            label=k,
            marker=marker,
            ls=ls,
            lw=lw,
            ms=ms,
        )

    plt.xticks(np.arange(0, max(eval_range), 10), fontsize="x-large")
    # plt.yticks(np.arange(0, 10, 0.5), fontsize='x-large')
    plt.ylim(0, np.median(quantiles))
    plt.xlabel("Number of Datapoints selected", fontsize="xx-large", labelpad=8)
    plt.ylabel("Test\nError", fontsize="xx-large", rotation=0, labelpad=30)
    plt.legend(
        fontsize="xx-large", bbox_to_anchor=(0.5, 1.4), loc="upper center", ncols=2
    )
    plt.tight_layout(pad=0, w_pad=0)
    plt.savefig(save_path, bbox_inches="tight")


def plot_errors_under_budget(results, save_path):
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 8))
    error_under_budgets = results["errors"]
    eval_range = results["eval_range"]

    quantiles = []
    for i, (k, v) in enumerate(error_under_budgets.items()):
        error_per_budget = defaultdict(list)
        for v_i in v:
            for b, e in v_i.items():
                error_per_budget[b].append(e)

        budgets = []
        errors = []
        for b, e in dict(sorted(error_per_budget.items())).items():
            budgets.append(b)
            errors.append(np.mean(e))

        quantiles.append(np.quantile(errors, 0.9))
        ms = 5
        match k:
            case "LavaEvaluator":
                k = "LAVA"
            case "InfluenceSubsample":
                k = "Influence"
            case "LeaveOneOut":
                k = "Leave One Out"
            case "KNNShapley":
                k = "KNN Shapley"
            case "DataOob":
                k = "Data-OOB"
            case _:
                k = k

        match k:
            case k if "Ours" in k:
                lw = 2
                ls = "-"
                marker = "*"
                ms = ms + 5
            case k if "random" in k.lower():
                lw = 5
                ls = "-"
                marker = ""
            case _:
                lw = 2
                ls = "-"
                marker = "s"

        plt.plot(budgets, errors, label=k, marker=marker, ls=ls, lw=lw, ms=ms)

    plt.xticks(np.arange(0, max(eval_range), 10), fontsize="x-large")
    # plt.yticks(np.arange(0, 10, 0.5), fontsize='x-large')
    plt.ylim(0, np.median(quantiles))
    plt.xlabel("Budget", fontsize="xx-large", labelpad=8)
    plt.ylabel("Test\nError", fontsize="xx-large", rotation=0, labelpad=30)
    plt.legend(
        fontsize="xx-large", bbox_to_anchor=(0.5, 1.4), loc="upper center", ncols=2
    )
    plt.tight_layout(pad=0, w_pad=0)
    plt.savefig(save_path, bbox_inches="tight")


def plot_results_utility(figure_path, results, cost_range):
    if cost_range is not None:
        plot_errors_under_budget(results, figure_path)
    else:
        plot_errors_fixed(results, figure_path)
    print(f"Plot saved to {figure_path}".center(80, "="))


def plot_image_selection_rate(figure_path, results, eval_range):
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(8, 8))

    quantiles = []

    for label, data in results.items():
        # Convert data to a numpy array for easier manipulation
        data_array = np.array(data)  # Shape: (num_runs, len(eval_range))

        ms = 5

        match label:
            case label if "benign" in label:
                lw = 2
                ls = "-"
                marker = "*"
                ms = ms + 5
            case label if "malicious" in label.lower():
                lw = 5
                ls = "-"
                marker = ""
            case _:
                lw = 2
                ls = "-"
                marker = "s"

        # Compute the mean selection rate across runs
        mean_rate = data_array.mean(axis=0)
        std_rate = data_array.std(axis=0)  # Optional: Compute standard deviation for error bars

        # Plot the mean selection rate with error bars
        plt.plot(eval_range, mean_rate, label=label, marker=marker,
                 ls=ls,
                 lw=lw,
                 ms=ms, )
        plt.fill_between(
            eval_range,
            mean_rate - std_rate,
            mean_rate + std_rate,
            alpha=0.2
        )  # Add shaded error bars (optional)

    # Add plot labels and legend
    plt.xlabel("Number of Datapoints selected", fontsize="xx-large", labelpad=8)
    plt.ylabel("Selection Rate", fontsize="xx-large", rotation=0, labelpad=30)

    plt.title("Average Selection Rate Across Runs")

    plt.legend(
        fontsize="xx-large", bbox_to_anchor=(0.5, 1.4), loc="upper center", ncols=2
    )

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(pad=0, w_pad=0)
    plt.xticks(np.arange(0, max(eval_range), 10), fontsize="x-large")

    # Save or show the plot
    if figure_path:
        plt.savefig(figure_path, bbox_inches="tight")
        print(f"Figure saved to {figure_path}")
    else:
        plt.close()


def plot_results_data_selection(plot_type, figure_path, results, eval_range):
    if plot_type == "image_selection_rate":
        plot_image_selection_rate(figure_path, results, eval_range)
    print(f"Plot saved to {figure_path}".center(80, "="))


def save_results_pkl(results, save_dir='results'):
    """
    Saves the inference results and plots to the specified directory.

    Parameters:
    - results (dict): Dictionary containing all inference results.
    - save_dir (str): Directory path where results will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save numerical results using pickle
    with open(os.path.join(save_dir, 'inference_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to directory: {save_dir}")

    # Optionally, save plots manually within other functions or here if needed


# --- Modified Orchestration Function ---
def image_modification(
        modify_info,
        model,
        processor,
        device,
        output_dir='modified_images',
        num_steps=100,
        learning_rate=0.1,
        lambda_reg=0.1,
        epsilon=0.05,
        noise_vis_scale_factor=10,
        visualize_limit=5
):
    """
    Perform the poisoning attack on selected images, saving results and noise visualization.

    Parameters:
    - modify_info (dict): Dictionary where keys are indices and values are dicts
                          containing 'original_img_path', 'target_vector', etc.
    ... (other parameters) ...
    - noise_vis_scale_factor (float): Scaling factor for noise visualization.

    Returns:
    - modify_result (dict): Dictionary containing results for each modified image,
                            including paths to saved original, modified, and noise images.
    """
    visualizations_saved = 0
    os.makedirs(output_dir, exist_ok=True)
    modify_result = {}

    for idx, cur_info in tqdm(modify_info.items(), desc="Performing Image Modification"):
        target_vector = cur_info["target_vector"]
        original_img_path_str = cur_info["original_img_path"]
        target_img_path = cur_info.get("target_img_path", "N/A")  # Optional target path
        target_img_idx = cur_info.get("target_index", "N/A")  # Optional target index

        # Ensure the original image path exists
        original_img_path = Path(original_img_path_str)
        if not original_img_path.is_file():
            print(f"Warning: Original image not found at {original_img_path_str}, skipping index {idx}.")
            continue

        try:
            (
                original_image_pil,
                modified_image_pil,
                noise_visualization_pil,  # Get noise image
                modified_embedding,
                similarity
            ) = modify_image(
                image_path=original_img_path_str,
                target_vector=target_vector,
                model=model,
                processor=processor,
                device=device,
                num_steps=num_steps,
                learning_rate=learning_rate,
                lambda_reg=lambda_reg,
                epsilon=epsilon,
                noise_vis_scale_factor=noise_vis_scale_factor,  # Pass down
                verbose=False  # Keep loop tidy, enable verbose in modify_image if needed
            )

            # --- Save Images ---
            base_filename = original_img_path.stem  # Get filename without extension

            # Define save paths
            original_save_path = Path(output_dir) / f"{base_filename}_original{original_img_path.suffix}"
            modified_save_path = Path(output_dir) / f"{base_filename}_modified{original_img_path.suffix}"
            noise_save_path = Path(output_dir) / f"{base_filename}_noise{original_img_path.suffix}"  # Noise image path

            # Save the images
            original_image_pil.save(original_save_path)
            modified_image_pil.save(modified_save_path)
            noise_visualization_pil.save(noise_save_path)  # Save noise image

            # Store results
            modify_result[idx] = {
                "target_image_idx": target_img_idx,
                "target_img_path": target_img_path,
                "similarity_to_target": similarity,
                "m_embedding": modified_embedding.numpy(),  # Store as numpy array
                "original_img_path_saved": str(original_save_path),  # Path where copy is saved
                "modified_img_path_saved": str(modified_save_path),  # Path where modified is saved
                "noise_img_path_saved": str(noise_save_path)  # Path where noise vis is saved
            }

            if visualize_limit == -1 or visualizations_saved < visualize_limit:
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Adjust figsize as needed
                fig.suptitle(
                    f"Image Modification Result (Original Idx: {idx})\nTarget Idx: {target_img_idx} | Final Similarity: {similarity:.4f}",
                    fontsize=14)

                # 1. Original Image
                ax = axes[0, 0]
                ax.imshow(original_image_pil)
                ax.set_title(f"Original ({base_filename})")
                ax.axis('off')

                # 2. Target Image (Load if path is valid)
                ax = axes[0, 1]
                target_image_pil = None
                if target_img_path and Path(target_img_path).is_file():
                    try:
                        target_image_pil = Image.open(target_img_path).convert("RGB")
                        ax.imshow(target_image_pil)
                        ax.set_title(f"Target ({Path(target_img_path).stem})")
                    except Exception as load_err:
                        ax.text(0.5, 0.5, f'Error loading target:\n{load_err}', ha='center', va='center')
                        ax.set_title("Target (Load Error)")
                else:
                    ax.text(0.5, 0.5, 'Target Image\nNot Available', ha='center', va='center')
                    ax.set_title("Target (N/A)")
                ax.axis('off')

                # 3. Modified Image
                ax = axes[1, 0]
                ax.imshow(modified_image_pil)
                ax.set_title("Modified (Original + Noise)")
                ax.axis('off')

                # 4. Noise Visualization
                ax = axes[1, 1]
                ax.imshow(noise_visualization_pil)
                ax.set_title(f"Noise Visualization (Scaled, eps={epsilon})")
                ax.axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

                # Define save path for the comparison figure
                comparison_save_path = Path(output_dir) / f"{base_filename}_comparison.png"
                plt.savefig(comparison_save_path)
                plt.close(fig)  # Close the figure to free memory
                visualizations_saved += 1
                # Optionally add path to modify_result
                modify_result[idx]["comparison_fig_path_saved"] = str(comparison_save_path)
        except FileNotFoundError:
            print(f"Error: File not found during processing for {original_img_path_str}, skipping index {idx}.")
        except Exception as e:
            print(f"Error processing image index {idx} ({original_img_path_str}): {e}")
            # Optionally store error information
            modify_result[idx] = {"error": str(e), "original_img_path": original_img_path_str}

    return modify_result


def denormalize_image(tensor):
    """
    Reverses the normalization applied to the image tensor.

    Parameters:
    - tensor (torch.Tensor): The normalized image tensor to be denormalized.

    Returns:
    - torch.Tensor: The denormalized image tensor.
    """

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    return tensor * std + mean


class IMG:
    def __init__(self, path, idx, emb, is_modified, modify_target_idx):
        self.path = path
        self.idx = idx
        self.emb = emb
        self.is_modified = is_modified
        self.modify_target_idx = modify_target_idx


def plot_selection_weights(initial_weights, updated_weights, save_path=None):
    """
    Plot the weights before and after the attack.

    Parameters:
    - initial_weights (np.ndarray): Weights before the attack.
    - updated_weights (np.ndarray): Weights after the attack.
    - save_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    plt.figure(figsize=(12, 6))
    plt.hist(initial_weights, bins=50, alpha=0.5, label='Initial Weights')
    plt.hist(updated_weights, bins=50, alpha=0.5, label='Updated Weights')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Weights Before and After Attack')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.close()


def save_results(results, save_dir, save_name):
    """
    Save the results dictionary to a pickle file.

    Parameters:
    - results (dict): Results to save.
    - save_dir (str): Directory to save the results.
    - save_name (str): Filename for the saved results.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / f"{save_name}.pkl", "wb") as f:
        pickle.dump(results, f)


def load_results(save_dir, save_name):
    """
    Load the results dictionary from a pickle file.

    Parameters:
    - save_dir (str): Directory where the results are saved.
    - save_name (str): Filename of the saved results.

    Returns:
    - results (dict): Loaded results.
    """
    with open(Path(save_dir) / f"{save_name}.pkl", "rb") as f:
        results = pickle.load(f)
    return results


def load_image(image_path):
    """
    Load an image and return a PIL Image.
    """
    return Image.open(image_path)


def create_perturbation_visualization(original_pil, modified_pil, scale_factor=10):
    """
    Calculates the difference between original and modified images and
    creates a visualization emphasizing the changes.

    Parameters:
    - original_pil (PIL.Image): The original image.
    - modified_pil (PIL.Image): The modified image.
    - scale_factor (float): How much to amplify the difference for visibility.

    Returns:
    - noise_vis_pil (PIL.Image): A visualization of the perturbation.
                                 Gray areas mean no change, brighter/darker areas
                                 indicate positive/negative changes.
    """
    # Ensure images are in RGB format
    original_pil = original_pil.convert("RGB")
    modified_pil = modified_pil.convert("RGB")

    # Convert PIL images to tensors (scaled 0-1)
    to_tensor = transforms.ToTensor()
    original_tensor = to_tensor(original_pil)
    modified_tensor = to_tensor(modified_pil)

    # Calculate the difference (perturbation) in pixel space
    perturbation_tensor = modified_tensor - original_tensor

    # Create visualization: scale the difference and add a mid-gray offset (0.5)
    # This makes 0 difference appear gray.
    visual_noise_tensor = 0.5 + (perturbation_tensor * scale_factor)

    # Clamp the visualization tensor to the valid [0, 1] range
    visual_noise_tensor = torch.clamp(visual_noise_tensor, 0, 1)

    # Convert the visualization tensor back to a PIL image
    to_pil = transforms.ToPILImage()
    noise_vis_pil = to_pil(visual_noise_tensor)

    return noise_vis_pil


def modify_image(
        image_path,
        target_vector,
        model,
        processor,
        device,
        num_steps=100,
        learning_rate=0.01,
        lambda_reg=0.1,
        epsilon=0.05,
        noise_vis_scale_factor=10,  # Parameter for noise visualization scaling
        verbose=False
):
    """
    Modify an image to align its CLIP embedding with the target vector.

    Parameters:
    - image_path (str): Path to the image to be modified.
    - target_vector (np.array): Target embedding vector (1D array).
    ... (other parameters remain the same) ...
    - noise_vis_scale_factor (float): Scaling factor for noise visualization.
    - verbose (bool): Whether to print progress messages.

    Returns:
    - original_image_pil (PIL.Image): The original loaded image.
    - modified_image_pil (PIL.Image): The optimized image.
    - noise_visualization_pil (PIL.Image): Visualization of the added perturbation.
    - modified_embedding (torch.Tensor): Final normalized embedding of the modified image.
    - final_similarity (float): Cosine similarity with the target vector after modification.
    """
    # Load the original image (keep it for noise visualization later)
    original_image_pil = Image.open(image_path).convert("RGB")  # Ensure RGB

    # Preprocess for CLIP
    # image_tensor = processor(images=original_image_pil, return_tensors="pt")['pixel_values'].to(device)
    processed_tensor = processor(original_image_pil)

    # 2. Add the batch dimension (CLIP models expect input shape [batch_size, channels, height, width])
    # The preprocess output is typically [channels, height, width]
    image_tensor = processed_tensor.unsqueeze(0).to(device)

    # If you want gradients with respect to the image tensor, set requires_grad
    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    original_image_tensor_norm = image_tensor.clone().detach()  # Keep normalized original

    if verbose:
        print(f"Starting optimization for image: {image_path}")
        print("Initial image tensor shape:", image_tensor.shape)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW([image_tensor], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0)

    # Prepare target tensor
    target_tensor = torch.tensor(target_vector, dtype=torch.float32).to(device)
    # Ensure target_tensor has a batch dimension if embedding doesn't
    if target_tensor.dim() == 1:
        target_tensor = target_tensor.unsqueeze(0)
    target_tensor = F.normalize(target_tensor, p=2, dim=-1)  # Normalize along the feature dimension

    # Set model to evaluation mode
    model.eval()

    # Retrieve normalization parameters (use standard CLIP values)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)  # Add batch dim
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)  # Add batch dim

    # --- Optimization Loop ---
    previous_loss = float('inf')
    patience = 10
    patience_counter = 0

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass: get embedding image_features = model.encode_image(image_tensor)
        embedding = model.encode_image(image_tensor)  # Use appropriate method
        embedding = F.normalize(embedding, p=2, dim=-1)

        # Compute cosine similarity
        # Ensure dimensions match for broadcasting or use .mean() if appropriate
        cosine_sim = F.cosine_similarity(embedding, target_tensor, dim=-1).mean()

        # Compute perturbation norm (L-infinity in normalized space)
        perturbation_norm = image_tensor - original_image_tensor_norm
        reg_loss = lambda_reg * torch.norm(perturbation_norm.view(perturbation_norm.size(0), -1), p=float('inf'),
                                           dim=1).mean()

        # Compute loss
        loss = -cosine_sim + reg_loss

        # Backward pass & Gradient Clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_([image_tensor], max_norm=1.0)

        if verbose and image_tensor.grad is not None:
            print(
                f"Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}, Cosine Sim: {cosine_sim.item():.4f}, Grad Norm: {image_tensor.grad.norm().item():.4f}")

        # Optimizer step & Scheduler step
        optimizer.step()
        scheduler.step()

        # Clamp perturbation and image tensor in normalized space
        with torch.no_grad():
            perturbation = torch.clamp(image_tensor - original_image_tensor_norm, -epsilon, epsilon)
            # Clamp within valid *normalized* range based on 0-1 pixel values
            min_norm_val = (0 - mean) / std
            max_norm_val = (1 - mean) / std
            new_image = torch.clamp(original_image_tensor_norm + perturbation, min_norm_val, max_norm_val)
            image_tensor.copy_(new_image)

        # Early Stopping Check
        current_loss = loss.item()
        if abs(previous_loss - current_loss) < 1e-5:  # Slightly more robust threshold
            patience_counter += 1
            if patience_counter >= patience:
                if verbose: print("Early stopping triggered.")
                break
        else:
            patience_counter = 0
        previous_loss = current_loss

    # --- End Optimization Loop ---

    # --- Post-processing ---
    # Denormalize the final image tensor
    def denormalize_image_tensor(tensor):
        # Ensure tensor is on CPU for operations with CPU mean/std if needed
        tensor_cpu = tensor.cpu()
        mean_cpu = mean.cpu()
        std_cpu = std.cpu()
        # Remove batch dim if it exists before applying std/mean which are (1,3,1,1) or (3,1,1)
        if tensor_cpu.dim() == 4:
            tensor_cpu = tensor_cpu.squeeze(0)  # Now (3, H, W)
        if mean_cpu.dim() == 4:
            mean_cpu = mean_cpu.squeeze(0)
        if std_cpu.dim() == 4:
            std_cpu = std_cpu.squeeze(0)
        # Apply denormalization
        denorm_tensor = tensor_cpu * std_cpu + mean_cpu
        return denorm_tensor

    modified_image_denorm_tensor = denormalize_image_tensor(image_tensor.detach())
    # Clamp to [0, 1] and convert to PIL
    modified_image_pil = transforms.ToPILImage()(modified_image_denorm_tensor.clamp(0, 1))

    # Compute final similarity with the final tensor
    with torch.no_grad():
        modified_embedding = model.encode_image(image_tensor)
        modified_embedding = F.normalize(modified_embedding, p=2, dim=-1)
        final_similarity = F.cosine_similarity(modified_embedding, target_tensor,
                                               dim=-1).item()  # Use item() for single value
    target_size = modified_image_pil.size  # Get (width, height) from the modified PIL image
    original_image_resized_pil = original_image_pil.resize(target_size,
                                                           Image.Resampling.LANCZOS)  # Use a good resampling filter
    # Create noise visualization
    noise_visualization_pil = create_perturbation_visualization(
        original_image_resized_pil,  # Pass the RESIZED original image
        modified_image_pil,
        scale_factor=noise_vis_scale_factor
    )

    return (
        original_image_pil,  # Return the original (non-resized) PIL for reference if needed
        modified_image_pil,
        noise_visualization_pil,
        modified_embedding.cpu(),
        final_similarity
    )


def evaluate_reconstruction(x_tests_true, x_tests_est):
    """
    Evaluates the reconstruction by comparing each reconstructed sample to all ground truth samples.
    Selects the best match based on cosine similarity and Euclidean distance.

    Parameters:
    - x_tests_true: np.ndarray of shape (n_tests_true, n_features)
        Ground truth test samples.
    - x_tests_est: np.ndarray of shape (n_tests_est, n_features)
        Reconstructed test samples.

    Returns:
    - best_cosine_similarities: list of highest cosine similarities for each reconstructed sample.
    - best_euclidean_distances: list of lowest Euclidean distances for each reconstructed sample.
    - matching_indices: list of indices of ground truth samples that best match each reconstructed sample.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    n_tests_est = x_tests_est.shape[0]
    n_tests_true = x_tests_true.shape[0]

    # Compute pairwise cosine similarities and Euclidean distances
    cosine_sim_matrix = cosine_similarity(x_tests_est, x_tests_true)  # Shape: (n_tests_est, n_tests_true)
    euclidean_dist_matrix = np.linalg.norm(
        x_tests_est[:, np.newaxis, :] - x_tests_true[np.newaxis, :, :],
        axis=2
    )  # Shape: (n_tests_est, n_tests_true)

    best_cosine_similarities = []
    best_euclidean_distances = []
    matching_indices = []

    # For each reconstructed sample, find the best match in ground truth samples
    for i in range(n_tests_est):
        best_match_idx = np.argmax(cosine_sim_matrix[i])  # Index of best cosine similarity
        best_cosine_similarity = cosine_sim_matrix[i, best_match_idx]
        best_euclidean_distance = euclidean_dist_matrix[i, best_match_idx]

        # Store the results
        best_cosine_similarities.append(best_cosine_similarity)
        best_euclidean_distances.append(best_euclidean_distance)
        matching_indices.append(best_match_idx)

    return best_cosine_similarities, best_euclidean_distances, matching_indices


def embed_images(img_paths, model, preprocess, device):
    """
    Embed images using CLIP's encode_image function.

    Parameters:
    - img_paths (list of str): Paths to the images to be embedded.
    - model (CLIPModel): Pre-trained CLIP model.
    - preprocess (callable): CLIP's preprocess function.
    - device (str): Device to run computations on ('cpu' or 'cuda').

    Returns:
    - embeddings (torch.Tensor): Concatenated embeddings of all images.
    """
    embeddings = []
    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="Embedding Images"):
            img_preprocessed = preprocess_image(img_path, preprocess, device)
            embedding = model.encode_image(img_preprocessed)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize embedding
            embeddings.append(embedding.cpu())
    return torch.cat(embeddings)


def preprocess_image(image_path, preprocess, device):
    img = Image.open(image_path)
    img_preprocessed = preprocess(img).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
    return img_preprocessed


def convert_arrays_to_lists(obj):
    """Recursively convert NumPy arrays to lists in a nested structure."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_arrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_arrays_to_lists(element) for element in obj]
    else:
        return obj


def save_results_trained_model(args, results):
    for k, v in vars(args).items():
        if k not in results:
            if isinstance(v, Path):
                v = str(v)
            results[k] = v
        else:
            print(f"Found {k} in results. Skipping.")

    result_path = f"{args.result_dir}/{args.save_name}-results.json"
    results = convert_arrays_to_lists(results)
    with open(result_path, "w") as f:
        json.dump(results, f, default=float)
    print(f"Results saved to {result_path}".center(80, "="))


import pandas as pd
import datetime


def save_dict_to_df(data: dict, file_prefix: str = "output"):
    """
    Save a dictionary to a DataFrame and export it to a CSV file with a timestamped filename.

    :param data: Dictionary to be saved
    :param file_prefix: Prefix for the output file name
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate filename
    filename = f"{file_prefix}_{timestamp}.csv"

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"DataFrame saved as {filename}")


def get_error_fixed(
        x_test,
        y_test,
        x_s,
        y_s,
        w,
        eval_range=range(1, 10),
        use_sklearn=False,
        return_list=False,
):
    sorted_w = w.argsort()[::-1]
    errors = {}
    for k in eval_range:
        selected = sorted_w[:k]
        x_k = x_s[selected]
        y_k = y_s[selected]

        if use_sklearn:
            LR = LinearRegression(fit_intercept=False)
            LR.fit(x_k, y_k)
            y_hat = LR.predict(x_test)
        else:
            beta_k = np.linalg.pinv(x_k) @ y_k
            y_hat = x_test @ beta_k

        errors[k] = mean_squared_error(y_test, y_hat)

    return list(errors.values()) if return_list else errors


def get_error_under_budget(
        x_test,
        y_test,
        x_s,
        y_s,
        w,
        costs=None,
        eval_range=range(1, 10),
        use_sklearn=False,
        return_list=False,
):
    assert costs is not None, "Missing costs"
    sorted_w = w.argsort()[::-1]
    cum_cost = np.cumsum(costs[sorted_w])

    errors = {}
    for budget in eval_range:
        under_budget_index = np.searchsorted(cum_cost, budget, side="left")

        # Could not find any points under budget constraint
        if under_budget_index == 0:
            continue

        selected = sorted_w[:under_budget_index]
        x_budget = x_s[selected]
        y_budget = y_s[selected]

        if use_sklearn:
            LR = LinearRegression(fit_intercept=False)
            LR.fit(x_budget, y_budget)
            y_hat = LR.predict(x_test)
        else:
            beta_budget = np.linalg.pinv(x_budget) @ y_budget
            y_hat = x_test @ beta_budget

        errors[budget] = mean_squared_error(y_test, y_hat)

    # Remove keys with values under budget
    # errors = {k: v for k, v in errors.items() if v is not None}
    return list(errors.values()) if return_list else errors
