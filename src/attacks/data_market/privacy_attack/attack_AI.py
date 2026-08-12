import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

################################################################################
# 1. Load & Preprocess Data
################################################################################

# Paths & constants
ROOT_DIR = "/path/to/celeba"  # <-- Update this path for CelebA
BATCH_SIZE = 64
IMAGE_SIZE = 64
SMILING_INDEX = 31  # Index of "Smiling" attribute in CelebA

# Compose transformations for images
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# We use CelebA's "train" or "valid" split for the seller, "test" split for buyer
# (This is arbitrary for demonstration; adapt as you see fit.)
seller_dataset = datasets.CelebA(
    root=ROOT_DIR, split="train", target_type="attr",
    transform=transform, download=False
)
buyer_dataset = datasets.CelebA(
    root=ROOT_DIR, split="test", target_type="attr",
    transform=transform, download=False
)

# Subsample to keep it small for the demo
NUM_SELLER_SAMPLES = 1000
NUM_BUYER_SAMPLES  = 200
random.seed(123)  # For reproducibility

seller_indices = random.sample(range(len(seller_dataset)), NUM_SELLER_SAMPLES)
buyer_indices  = random.sample(range(len(buyer_dataset)),  NUM_BUYER_SAMPLES)

seller_subset = torch.utils.data.Subset(seller_dataset, seller_indices)
buyer_subset  = torch.utils.data.Subset(buyer_dataset,  buyer_indices)

seller_loader = torch.utils.data.DataLoader(
    seller_subset, batch_size=BATCH_SIZE, shuffle=False
)
buyer_loader  = torch.utils.data.DataLoader(
    buyer_subset,  batch_size=BATCH_SIZE, shuffle=False
)

################################################################################
# 2. Define a Simple CNN for Feature Extraction
################################################################################

class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim=64):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(16, embedding_dim)

    def forward(self, x):
        # x: (batch, 3, H, W)
        x = self.conv(x)          # -> (batch, 16, 1, 1)
        x = x.view(x.size(0), -1) # -> (batch, 16)
        x = self.fc(x)            # -> (batch, embedding_dim)
        return x

# Instantiate and (optionally) load pretrained weights or keep random
feature_extractor = SimpleCNN(embedding_dim=64)
feature_extractor.eval()

def extract_embeddings(dataloader, model):
    """
    Extract embeddings and attribute labels for all images in a dataloader.
    Returns:
      embs:   numpy array of shape (N, embedding_dim)
      attrs:  numpy array of shape (N, 40) with CelebA attributes
    """
    all_embs = []
    all_attrs = []
    with torch.no_grad():
        for images, attrs in dataloader:
            emb = model(images)  # shape: (batch_size, embedding_dim)
            all_embs.append(emb.cpu().numpy())
            all_attrs.append(attrs.cpu().numpy())
    all_embs  = np.concatenate(all_embs, axis=0)
    all_attrs = np.concatenate(all_attrs, axis=0)
    return all_embs, all_attrs

# Extract for seller and buyer subsets
seller_embs, seller_attrs = extract_embeddings(seller_loader, feature_extractor)
buyer_embs,  buyer_attrs  = extract_embeddings(buyer_loader,  feature_extractor)

# The "Smiling" attribute for buyer
buyer_smile_labels = buyer_attrs[:, SMILING_INDEX]  # shape: (NUM_BUYER_SAMPLES,)

# True fraction of smiling in buyer data (the quantity the adversary wants to guess)
true_alpha = np.mean(buyer_smile_labels)

################################################################################
# 3. Buyer’s Selection Mechanism
################################################################################

def compute_scores(seller_embs, buyer_embs):
    """
    Given seller_embs: shape (N, d)
          buyer_embs:  shape (M, d)
    We define a toy "value" of each seller point x_j as:
       score_j = sum_{i=1 to M} (x_j dot buyer_emb[i])^2
    Returns:
      scores: shape (N,) array
    """
    N = seller_embs.shape[0]
    scores = np.zeros(N, dtype=np.float32)

    # For each seller embedding
    for j in range(N):
        # dot w/ each buyer embedding
        dot_vals = buyer_embs @ seller_embs[j]  # shape (M,)
        scores[j] = np.sum(dot_vals**2)

    return scores

K = 20  # Buyer wants top-K
scores_seller = compute_scores(seller_embs, buyer_embs)
selected_indices = np.argsort(scores_seller)[-K:]  # the top-K
selected_indices_set = set(selected_indices)

################################################################################
# 4. Adversary Observations
#
# Scenario A (Score Unknown): The adversary only sees "selected_indices_set".
# Scenario B (Score Known): The adversary also sees "scores_seller".
################################################################################

# We'll implement two distinct attacks below.

################################################################################
# 5. ATTACK A: Score-Unknown Attack
################################################################################

"""
Scenario A:
-----------
We only know which subset of the seller’s points was chosen (final_selection).
We do NOT know the per-point scores.

Goal:
 Infer alpha = fraction of buyer’s images with the sensitive attribute (e.g. Smiling).

Method:
 - We define a parametric generative model for buyer embeddings, conditioned on alpha.
 - For each alpha candidate in [0, 1], we:
     1. Simulate a "fake" buyer test set of the same size M, drawing from
        a mixture of two distributions (Smiling vs Non-smiling).
     2. Run the same selection function (compute_scores + top-K).
     3. Compare the resulting chosen subset to the REAL chosen subset using
        e.g. Overlap or some metric.
 - We pick alpha that yields the best match.
"""

def simulate_buyer_test_embs(num_samples, alpha, emb_dim=64):
    """
    A toy generative model:
      - With probability alpha, sample from ~ N(mu_smile, I)
      - With probability (1-alpha), sample from ~ N(mu_not_smile, I)
    Adjust the means or covariance as needed.
    """
    mu_smiling    = np.ones(emb_dim) *  1.0
    mu_not_smile  = np.ones(emb_dim) * -1.0
    sigma = 1.0

    embs = []
    for _ in range(num_samples):
        if random.random() < alpha:
            e = np.random.randn(emb_dim)*sigma + mu_smiling
        else:
            e = np.random.randn(emb_dim)*sigma + mu_not_smile
        embs.append(e)

    return np.array(embs)

def attack_score_unknown(
    seller_embs,
    real_selected_indices,
    buyer_test_size,
    K,
    alpha_grid=np.linspace(0,1,11)
):
    """
    Implementation of the "guess and check" approach:
      - For alpha in [0,1] (sampled by alpha_grid),
      - Simulate buyer test embeddings
      - Run selection
      - Compare with real_selected_indices
    Returns the best alpha guess & the best overlap score.
    """
    emb_dim = seller_embs.shape[1]
    best_alpha = None
    best_overlap = -1

    for alpha in alpha_grid:
        # Simulate a synthetic buyer test set
        sim_buyer_embs = simulate_buyer_test_embs(
            num_samples=buyer_test_size, alpha=alpha, emb_dim=emb_dim
        )
        # Compute scores for the seller's data
        sim_scores = compute_scores(seller_embs, sim_buyer_embs)
        sim_selected = set(np.argsort(sim_scores)[-K:])

        # Evaluate overlap
        overlap = len(sim_selected.intersection(real_selected_indices))
        if overlap > best_overlap:
            best_overlap = overlap
            best_alpha = alpha

    return best_alpha, best_overlap

print("=== Scenario A: Score-Unknown Attack ===")
best_alpha_guess_A, best_overlap = attack_score_unknown(
    seller_embs,
    selected_indices_set,
    buyer_test_size=NUM_BUYER_SAMPLES,
    K=K
)

print(f" True alpha (fraction Smiling)         = {true_alpha:.4f}")
print(f" Adversary's guessed alpha (ScenarioA) = {best_alpha_guess_A:.4f}")
print(f" Overlap with real selection           = {best_overlap}\n")

################################################################################
# 6. ATTACK B: Score-Known Attack
################################################################################

"""
Scenario B:
-----------
Now the adversary sees the full array of "scores_seller" (one score per seller point).

Goal:
 Infer alpha = fraction of the buyer’s images that are Smiling.

Method:
 Similar to Attack A, but:
   - For each candidate alpha,
   - Simulate buyer test embeddings -> compute "sim_scores" for the seller data,
   - Compare "sim_scores" vs the REAL "scores_seller" with MSE or distribution-based metric.
   - Pick alpha that minimizes difference.

This provides more signal to the adversary, often leading to more accurate inference.
"""

def attack_score_known(
    seller_embs,
    real_scores,
    buyer_test_size,
    alpha_grid=np.linspace(0,1,11)
):
    emb_dim = seller_embs.shape[1]
    best_alpha = None
    best_mse   = float('inf')

    for alpha in alpha_grid:
        sim_buyer_embs = simulate_buyer_test_embs(buyer_test_size, alpha, emb_dim)
        sim_scores = compute_scores(seller_embs, sim_buyer_embs)

        # Compare via MSE
        mse = np.mean((sim_scores - real_scores)**2)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    return best_alpha, best_mse

print("=== Scenario B: Score-Known Attack ===")
best_alpha_guess_B, best_mse = attack_score_known(
    seller_embs,
    scores_seller,
    buyer_test_size=NUM_BUYER_SAMPLES
)
print(f" True alpha (fraction Smiling)         = {true_alpha:.4f}")
print(f" Adversary's guessed alpha (ScenarioB) = {best_alpha_guess_B:.4f}")
print(f" Final MSE on scores                   = {best_mse:.4f}\n")
