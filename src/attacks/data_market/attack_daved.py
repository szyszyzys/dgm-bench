import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
# CLIP model and processor
from transformers import CLIPModel, CLIPProcessor


def attack(attack_type):
    if attack_type == "sampling":
        poison_sampling()

    elif attack_type == "poison":
        poison_data()
    elif attack_type == "backdoor":
        poison_backdoor()


def poison_sampling():
    # Load the pre-trained CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # List of paths to best-selling images
    best_selling_image_paths = [
        'path/to/best_selling_image1.jpg',
        'path/to/best_selling_image2.jpg',
        'path/to/best_selling_image3.jpg',
        # Add more paths as needed
    ]

    def load_image(image_path):
        """
        Load an image and return a PIL Image.
        """
        return Image.open(image_path).convert("RGB")

    def extract_features(image, model, processor):
        """
        Extract the image features using the CLIP model.
        """
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # Normalize the features
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features.squeeze(0)  # Remove batch dimension

    # Extract features from best-selling images
    best_selling_features = []
    for image_path in best_selling_image_paths:
        image = load_image(image_path)
        features = extract_features(image, model, processor)
        best_selling_features.append(features)

    # Compute the desired feature vector (mean of best-sellers)
    desired_feature_vector = torch.stack(best_selling_features).mean(dim=0)
    desired_feature_vector = F.normalize(desired_feature_vector, p=2, dim=-1)

    # Path to the irrelevant image
    irrelevant_image_path = 'path/to/irrelevant_image.jpg'

    # Load the image
    irrelevant_image = load_image(irrelevant_image_path)

    # Prepare the image tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    irrelevant_image_tensor = transform(irrelevant_image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

    # Initialize noise (delta) with zeros
    delta = torch.zeros_like(irrelevant_image_tensor, requires_grad=True)

    # Define optimizer
    optimizer = torch.optim.Adam([delta], lr=0.01)

    lambda_reg = 0.01  # Regularization strength

    # Target feature vector (detach to prevent gradients flowing into it)
    target_feature = desired_feature_vector.detach()

    num_iterations = 200
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Compute the modified image
        modified_image = irrelevant_image_tensor + delta
        modified_image = torch.clamp(modified_image, 0, 1)  # Ensure pixel values are valid

        # Extract features
        inputs = processor(images=modified_image.squeeze(0), return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=-1).squeeze(0)

        # Feature alignment loss (we want to maximize similarity)
        feature_alignment_loss = 1 - F.cosine_similarity(image_features, target_feature, dim=0)

        # Noise regularization (L2 norm of delta)
        noise_reg = torch.norm(delta)

        # Total loss
        total_loss = feature_alignment_loss + lambda_reg * noise_reg

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Optional: Print loss
        if (iteration + 1) % 20 == 0:
            print(
                f"Iteration {iteration + 1}/{num_iterations}, Total Loss: {total_loss.item():.4f}, Feature Alignment Loss: {feature_alignment_loss.item():.4f}, Noise Reg: {noise_reg.item():.4f}")
        # Obtain the final modified image
        modified_image = irrelevant_image_tensor + delta
        modified_image = torch.clamp(modified_image, 0, 1).detach().cpu()

        # Save the modified image
        save_image(modified_image.squeeze(0), 'modified_irrelevant_image.jpg')


def poison_data():
    pass


def poison_backdoor():
    pass


def poison_sampling_exp():
    pass


def load_image(image_path):
    """
    Load an image and return a PIL Image.
    """
    return Image.open(image_path).convert("RGB")


def extract_features(image, model, processor):
    """
    Extract the image features using the CLIP model.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # Normalize the features
    image_features = F.normalize(image_features, p=2, dim=-1)
    return image_features.squeeze(0)  # Remove batch dimension


def modify_image(
        image_path,
        target_vector,
        model,
        processor,
        device,
        num_steps=100,
        learning_rate=0.01,
        lambda_reg=0.1,
        epsilon=0.05
):
    """
    Modify an image to align its CLIP embedding with the target vector.

    Parameters:
    - image_path (str): Path to the image to be modified.
    - target_vector (np.array): Target embedding vector.
    - model (CLIPModel): Pre-trained CLIP model.
    - processor (CLIPProcessor): CLIP processor.
    - device (str): Device to run computations on.
    - num_steps (int): Number of optimization steps.
    - learning_rate (float): Learning rate for optimizer.
    - lambda_reg (float): Regularization strength.
    - epsilon (float): Maximum allowed perturbation per pixel.

    Returns:
    - modified_image (PIL.Image): The optimized image.
    - similarity (float): Cosine similarity with the target vector.
    """
    # Load and preprocess image
    image = load_image(image_path)
    image_tensor = preprocess_image(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
    image_tensor.requires_grad = True

    # Define optimizer
    optimizer = torch.optim.Adam([image_tensor], lr=learning_rate)

    # Convert target_vector to torch tensor
    target_tensor = torch.tensor(target_vector).to(device)
    target_tensor = F.normalize(target_tensor, p=2, dim=0)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass: get embedding
        inputs = processor(images=image_tensor.squeeze(0).cpu(), return_tensors="pt").to(device)
        embedding = model.get_image_features(**inputs)
        embedding = F.normalize(embedding, p=2, dim=-1)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(embedding, target_tensor, dim=0)

        # Compute loss: maximize cosine similarity and minimize perturbation
        loss = -cosine_sim + lambda_reg * torch.norm(image_tensor - preprocess_image(image).unsqueeze(0).to(device))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Clamp the image tensor to maintain valid pixel range and limit perturbation
        with torch.no_grad():
            perturbation = torch.clamp(image_tensor - preprocess_image(image).unsqueeze(0).to(device), -epsilon,
                                       epsilon)
            image_tensor.copy_(torch.clamp(preprocess_image(image).unsqueeze(0).to(device) + perturbation, 0, 1))

        # Optional: Print progress
        if (step + 1) % 20 == 0 or step == 0:
            print(f"Step {step + 1}/{num_steps}, Cosine Similarity: {cosine_sim.item():.4f}, Loss: {loss.item():.4f}")

    # Detach and convert to PIL Image
    modified_image = image_tensor.detach().cpu().squeeze(0)
    modified_image_pil = transforms.ToPILImage()(modified_image)

    # Compute final similarity
    inputs = processor(images=modified_image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        modified_embedding = model.get_image_features(**inputs)
    modified_embedding = F.normalize(modified_embedding, p=2, dim=-1)
    final_similarity = F.cosine_similarity(modified_embedding, target_tensor, dim=-1).item()

    return modified_image_pil, final_similarity


def perform_attack_on_unsampled(
        unsampled_indices,
        image_paths,
        X_sell,
        target_vector,
        model,
        processor,
        device,
        output_dir='modified_images',
        num_steps=100,
        learning_rate=0.01,
        lambda_reg=0.1,
        epsilon=0.05
):
    """
    Perform the poisoning attack on unsampled images.

    Parameters:
    - unsampled_indices (list): Indices of unsampled data points.
    - image_paths (list): List of image file paths.
    - X_sell (np.array): Feature matrix of seller data.
    - target_vector (np.array): Desired target embedding vector.
    - model (CLIPModel): Pre-trained CLIP model.
    - processor (CLIPProcessor): CLIP processor.
    - device (str): Device to run computations on.
    - output_dir (str): Directory to save modified images.
    - num_steps, learning_rate, lambda_reg, epsilon: Optimization parameters.

    Returns:
    - modified_indices (list): Indices of images that were modified.
    - similarities (dict): Cosine similarities after modification.
    """
    os.makedirs(output_dir, exist_ok=True)
    modified_indices = []
    similarities = {}

    for idx in tqdm(unsampled_indices, desc="Performing Attack on Unsampled Images"):
        image_path = image_paths[idx]
        modified_image, similarity = modify_image(
            image_path=image_path,
            target_vector=target_vector,
            model=model,
            processor=processor,
            device=device,
            num_steps=num_steps,
            learning_rate=learning_rate,
            lambda_reg=lambda_reg,
            epsilon=epsilon
        )

        # Save the modified image
        modified_image_path = os.path.join(output_dir, f'modified_{Path(image_path).name}')
        modified_image.save(modified_image_path)

        # Update the feature matrix
        X_sell[idx] = extract_features(modified_image, model, processor)

        # Record the modification
        modified_indices.append(idx)
        similarities[idx] = similarity

    return modified_indices, similarities


def compute_attack_metrics(
        initial_selected,
        updated_selected,
        modified_indices
):
    """
    Compute metrics to evaluate the success of the poisoning attack.

    Parameters:
    - initial_selected (set): Initially selected data point indices.
    - updated_selected (set): Selected data point indices after attack.
    - modified_indices (list): Indices of images that were modified.

    Returns:
    - metrics (dict): Dictionary containing success rate and other metrics.
    """
    # Calculate how many modified images were selected after attack
    modified_selected = updated_selected.intersection(modified_indices)
    success_rate = len(modified_selected) / len(modified_indices) if len(modified_indices) > 0 else 0.0

    # Additional metrics can be added here (e.g., average similarity)

    metrics = {
        'num_modified': len(modified_indices),
        'num_successfully_selected': len(modified_selected),
        'success_rate': success_rate
    }

    return metrics


def run_attack_experiment(
        data_dir='path/to/images',  # Directory containing images
        csv_path='data.csv',  # CSV file with image paths and labels
        num_buyer=10,
        num_samples=1000,
        Ks=[2, 5, 10, 25, 50],
        epochs=10,
        num_iters=500,
        model_name='clip',
        lr=5e-5,
        batch_size=1,
        grad_steps=1,
        weight_decay=0.01,
        max_char_length=2048,
        exclude_long_reviews=False,
        attack_steps=100,
        learning_rate=0.01,
        lambda_reg=0.1,
        epsilon=0.05,
        output_dir='modified_images',
):
    """
    Run the entire attack experiment.

    Parameters:
    - All parameters as defined in the original `run_exp` function and attack-related parameters.

    Returns:
    - results (dict): Dictionary containing evaluation metrics.
    """
    # Load the CLIP model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load data
    df = pd.read_csv(Path(data_dir) / csv_path)
    image_paths = df['image_path'].tolist()  # Assuming CSV has an 'image_path' column
    labels = df['label'].tolist()  # Assuming CSV has a 'label' column

    # Limit to num_samples
    image_paths = image_paths[:num_samples]
    labels = labels[:num_samples]

    # Extract features
    print("Extracting features for all images...")
    X_sell = []
    for image_path in tqdm(image_paths, desc="Feature Extraction"):
        image = load_image(image_path)
        image_tensor = preprocess_image(image)
        embedding = extract_features(image_tensor, model, processor)
        X_sell.append(embedding)
    X_sell = np.vstack(X_sell)

    y_sell = np.array(labels)

    # Split data into buyers and sellers
    # For simplicity, assume the first `num_buyer` samples are buyers
    # and the rest are sellers. Adjust as needed.
    X_buy = X_sell[:num_buyer]
    y_buy = y_sell[:num_buyer]

    X_sell = X_sell[num_buyer:]
    y_sell = y_sell[num_buyer:]
    image_paths_sell = image_paths[num_buyer:]

    # Run design_selection initially
    print("Running initial design_selection...")
    initial_results = design_selection(
        X_sell=X_sell,
        y_sell=y_sell,
        X_buy=X_buy,
        y_buy=y_buy,
        num_select=10,
        num_iters=num_iters,
        alpha=None,
        recompute_interval=50,
        line_search=True,
        costs=None,
        reg_lambda=0.1,
        importance=None,  # Optional: Define if you have importance scores
    )

    # Identify selected and unsampled data points
    weights_initial = initial_results['weights']
    selected_indices_initial, unsampled_indices_initial = identify_samples(weights_initial, num_select=10)

    # Evaluate initial selection
    print("Evaluating initial selection...")
    initial_mse = evaluate_selection(
        X_buy=X_buy,
        y_buy=y_buy,
        X_sell=X_sell,
        y_sell=y_sell,
        selected_indices=selected_indices_initial,
        inv_cov=np.linalg.pinv(X_sell.T @ (X_sell * weights_initial[:, None]))
    )
    print(f"Initial MSE: {initial_mse:.4f}")

    # Perform attack on unsampled images
    print("Performing poisoning attack on unsampled images...")
    # Define target vector (e.g., mean of best-selling embeddings)
    target_vector = X_sell[selected_indices_initial].mean(axis=0)
    target_vector = target_vector / np.linalg.norm(target_vector)

    modified_indices, similarities = perform_attack_on_unsampled(
        unsampled_indices=unsampled_indices_initial,
        image_paths=image_paths_sell,
        X_sell=X_sell,
        target_vector=target_vector,
        model=model,
        processor=processor,
        device=device,
        output_dir=output_dir,
        num_steps=attack_steps,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        epsilon=epsilon
    )

    # Re-run design_selection with modified data
    print("Re-running design_selection after attack...")
    updated_results = design_selection(
        X_sell=X_sell,
        y_sell=y_sell,
        X_buy=X_buy,
        y_buy=y_buy,
        num_select=10,
        num_iters=num_iters,
        alpha=None,
        recompute_interval=50,
        line_search=True,
        costs=None,
        reg_lambda=0.1,
        importance=None,
    )

    # Identify selected and unsampled data points after attack
    weights_updated = updated_results['weights']
    selected_indices_updated, unsampled_indices_updated = identify_samples(weights_updated, num_select=10)

    print("Evaluating updated selection...")
    updated_mse = evaluate_selection(
        X_buy=X_buy,
        y_buy=y_buy,
        X_sell=X_sell,
        y_sell=y_sell,
        selected_indices=selected_indices_updated,
        inv_cov=np.linalg.pinv(X_sell.T @ (X_sell * weights_updated[:, None]))
    )
    print(f"Updated MSE: {updated_mse:.4f}")

    # Compute attack metrics
    initial_selected_set = set(selected_indices_initial)
    updated_selected_set = set(selected_indices_updated)

    attack_metrics = compute_attack_metrics(
        initial_selected=initial_selected_set,
        updated_selected=updated_selected_set,
        modified_indices=modified_indices
    )

    print("Attack Metrics:")
    print(f"Number of Modified Images: {attack_metrics['num_modified']}")
    print(f"Number of Modified Images Selected After Attack: {attack_metrics['num_successfully_selected']}")
    print(f"Success Rate of the Attack: {attack_metrics['success_rate'] * 100:.2f}%")

    # Optional: Plot weight distributions
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(weights_initial)), weights_initial, color='blue', alpha=0.5, label='Initial Weights')
    plt.bar(range(len(weights_updated)), weights_updated, color='red', alpha=0.5, label='Updated Weights')
    plt.xlabel('Data Point Index')
    plt.ylabel('Weight')
    plt.title('Comparison of Weights Before and After Attack')
    plt.legend()
    plt.close()

    # Return all results
    results = {
        'initial_mse': initial_mse,
        'updated_mse': updated_mse,
        'attack_metrics': attack_metrics,
        'selected_indices_initial': selected_indices_initial,
        'selected_indices_updated': selected_indices_updated,
        'modified_indices': modified_indices,
        'similarities': similarities,
    }

    return results


def load_and_preprocess_data(data_dir, csv_path, max_char_length=2048, exclude_long_reviews=False):
    """
    Load and preprocess the dataset.

    Parameters:
    - data_dir (str): Directory where the CSV file is located.
    - csv_path (str): Path to the CSV file containing the data.
    - max_char_length (int): Maximum character length for reviews.
    - exclude_long_reviews (bool): Whether to exclude reviews exceeding max_char_length.

    Returns:
    - data (dict): Dictionary containing split datasets.
    - reviews (list): List of review texts.
    - labels (list): List of corresponding labels.
    """
    df = pd.read_csv(Path(data_dir) / csv_path)
    reviews = []
    labels = []
    for i, r in tqdm(df.iterrows(), total=df.shape[0], desc="Loading Data"):
        x = f'Benefits: {r.benefitsReview}\nSide effects: {r.sideEffectsReview}\nComments: {r.commentsReview}'
        if exclude_long_reviews and len(x) > max_char_length:
            continue
        reviews.append(x)
        labels.append(r.rating)
    print(f'Total Reviews Loaded: {len(reviews)}')

    # Assuming `get_drug_data` is a utility function that handles data splitting and embedding extraction
    data = utils.get_drug_data(
        num_samples=len(reviews),
        data_dir=data_dir,
        csv_path=csv_path,
        embedding_path=f"druglib/druglib_embeddings_clip.pt",  # Adjust as needed
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_name='clip',
        max_char_length=max_char_length,
    )
    return data, reviews, labels


if __name__ == "__main__":
    # Example parameters (adjust as needed)

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Data subset selection with experimental design informed loss using Frank-Wolfe optimization",
    )
    parser.add_argument("--random_seed", default=12345, help="random seed")
    parser.add_argument("--data_dir", default="../data/fitzpatrick17k")
    parser.add_argument("--result_dir", default="../results")
    parser.add_argument("--exp_name", default=None, type=str)

    args = parser.parse_args()
    np.random.seed(args.random_seed)
    args.data_dir = Path(args.data_dir)
    args.result_dir = Path(args.result_dir)
    args.data_dir.mkdir(exist_ok=True, parents=True)
    args.result_dir.mkdir(exist_ok=True, parents=True)

    data_dir = args.data_dir
    csv_path = 'labels.csv'  # CSV should have 'image_path' and 'label' columns
    num_buyer = 10
    num_samples = 1000
    Ks = [2, 5, 10, 25, 50]
    epochs = 10
    num_iters = 500
    model_name = 'clip'  # Placeholder if needed
    lr = 5e-5
    batch_size = 1
    grad_steps = 1
    weight_decay = 0.01
    max_char_length = 2048
    exclude_long_reviews = False
    attack_steps = 100
    learning_rate = 0.01
    lambda_reg = 0.1
    epsilon = 0.05
    output_dir = 'modified_images'

    # Run the attack experiment
    results = run_attack_experiment(
        data_dir=data_dir,
        csv_path=csv_path,
        num_buyer=num_buyer,
        num_samples=num_samples,
        Ks=Ks,
        epochs=epochs,
        num_iters=num_iters,
        model_name=model_name,
        lr=lr,
        batch_size=batch_size,
        grad_steps=grad_steps,
        weight_decay=weight_decay,
        max_char_length=max_char_length,
        exclude_long_reviews=exclude_long_reviews,
        attack_steps=attack_steps,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        epsilon=epsilon,
        output_dir=output_dir,
    )

    # Further analysis can be done using the `results` dictionary
