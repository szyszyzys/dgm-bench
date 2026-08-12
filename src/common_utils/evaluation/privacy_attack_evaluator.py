import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# --- 1. Constants ---

DATASET_STATS = {
    'FMNIST': {'mean': [0.5], 'std': [0.5]},
    'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2470, 0.2435, 0.2616]},
}


# --- 2. Evaluation & Visualization Functions ---

def evaluate_reconstruction_metrics(
        reconstructed_images: torch.Tensor,
        ground_truth_images: torch.Tensor,
        reconstructed_labels: torch.Tensor,
        ground_truth_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Evaluates reconstruction quality by calculating image metrics and label accuracy.
    Made generic for any reconstruction attack.
    """
    # Ensure tensors are on CPU and converted to numpy
    img1_np = reconstructed_images.detach().cpu().numpy()
    img2_np = ground_truth_images.detach().cpu().numpy()

    # Calculate image metrics
    mse = np.mean((img1_np - img2_np) ** 2)
    psnr = compare_psnr(img1_np, img2_np, data_range=1.0)

    # SSIM requires channel-last format: (N, H, W, C)
    img1_np_cl = np.transpose(img1_np, (0, 2, 3, 1))
    img2_np_cl = np.transpose(img2_np, (0, 2, 3, 1))
    ssim = compare_ssim(img1_np_cl, img2_np_cl, data_range=1.0, channel_axis=-1, win_size=7)

    # Calculate label accuracy
    correct = (reconstructed_labels.cpu() == ground_truth_labels.cpu()).sum().item()
    label_acc = correct / ground_truth_labels.numel()

    return {"mse": float(mse), "psnr": psnr, "ssim": ssim, "label_acc": label_acc}


def visualize_reconstruction(
        reconstructed_images: torch.Tensor,
        ground_truth_images: torch.Tensor,
        reconstructed_labels: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        metrics: Dict[str, float],
        save_path: Path,
        max_images: int = 4
):
    """
    Generates and saves a visualization comparing ground truth and reconstructed images.
    """
    num_images_to_show = min(len(ground_truth_images), max_images)
    if num_images_to_show == 0:
        logging.warning("No images to visualize.")
        return

    fig, axes = plt.subplots(2, num_images_to_show, figsize=(num_images_to_show * 2.5, 5))
    if num_images_to_show == 1:
        axes = axes[:, np.newaxis]  # Ensure axes is always 2D

    is_grayscale = ground_truth_images.shape[1] == 1
    cmap = 'gray' if is_grayscale else None

    for i in range(num_images_to_show):
        gt_img = ground_truth_images[i].cpu().numpy().transpose(1, 2, 0).squeeze()
        rec_img = reconstructed_images[i].cpu().numpy().transpose(1, 2, 0).squeeze()

        # Top row: Ground Truth
        axes[0, i].imshow(gt_img, cmap=cmap)
        axes[0, i].set_title(f"GT Label: {ground_truth_labels[i].item()}")
        axes[0, i].axis('off')

        # Bottom row: Reconstructed
        axes[1, i].imshow(rec_img, cmap=cmap)
        axes[1, i].set_title(f"Rec. Label: {reconstructed_labels[i].item()}")
        axes[1, i].axis('off')

    title = (f"PSNR: {metrics.get('psnr', -1):.2f}, SSIM: {metrics.get('ssim', -1):.3f}, "
             f"Label Acc: {metrics.get('label_acc', -1):.2f}")
    fig.suptitle(title, fontsize=14)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved reconstruction visualization to {save_path}")
