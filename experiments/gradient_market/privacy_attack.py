import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from typing import Any, Dict, List, Optional, Tuple

from src.common_utils.constants import VictimStrategy
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import ServerAttackConfig

# --- 1. Configuration & Constants ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_STATS = {
    'FMNIST': {'mean': [0.5], 'std': [0.5]},
    'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
}


@dataclass
class GIAConfig:
    """Typed configuration for the core gradient inversion attack algorithm."""
    num_images: int = 1
    iterations: int = 2000
    lr: float = 0.1
    label_type: str = 'optimize'
    optimizer_class: Any = optim.Adam
    loss_type: str = 'cosine'
    init_type: str = 'gaussian'
    regularization_weight: float = 1e-4
    log_interval: Optional[int] = 500
    return_best: bool = True
    ground_truth_labels: Optional[torch.Tensor] = None


# --- 2. Core Attack & Evaluation Functions ---

def gradient_inversion_attack(
        target_gradient: List[torch.Tensor],
        model: nn.Module,
        input_shape: tuple,
        num_classes: int,
        device: torch.device,
        dataset_mean: List[float],
        dataset_std: List[float],
        attack_params: GIAConfig,
) -> (torch.Tensor, torch.Tensor):
    """Performs a gradient inversion attack to reconstruct input data and labels."""
    model.eval().to(device)
    target_gradient = [g.to(device) for g in target_gradient]
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    if attack_params.init_type == 'gaussian':
        dummy_data = torch.randn(attack_params.num_images, *input_shape, device=device, requires_grad=True)
    elif attack_params.init_type == 'random':
        dummy_data = torch.rand(attack_params.num_images, *input_shape, device=device, requires_grad=True)
    else:
        raise ValueError(f"Unknown init_type: {attack_params.init_type}")

    if attack_params.label_type == 'optimize':
        dummy_labels_param = torch.randn(attack_params.num_images, num_classes, device=device, requires_grad=True)
        params_to_optimize = [dummy_data, dummy_labels_param]
    else:
        if attack_params.label_type == 'ground_truth':
            if attack_params.ground_truth_labels is None or len(
                    attack_params.ground_truth_labels) != attack_params.num_images:
                raise ValueError("Ground truth labels required for label_type='ground_truth'")
            fixed_labels = attack_params.ground_truth_labels.to(device)
        elif attack_params.label_type == 'random':
            fixed_labels = torch.randint(0, num_classes, (attack_params.num_images,), device=device)
        else:
            raise ValueError(f"Unknown label_type: {attack_params.label_type}")
        params_to_optimize = [dummy_data]

    optimizer = attack_params.optimizer_class(params_to_optimize, lr=attack_params.lr)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_dummy_data, best_dummy_labels = float('inf'), None, None
    start_time = time.time()

    for it in range(attack_params.iterations):
        optimizer.zero_grad()
        model.zero_grad()
        normalized_dummy_data = normalize(dummy_data)

        if attack_params.label_type == 'optimize':
            current_labels_prob = F.softmax(dummy_labels_param, dim=-1)
            current_labels_idx = torch.argmax(current_labels_prob, dim=-1)
        else:
            current_labels_idx = fixed_labels

        output = model(normalized_dummy_data)
        task_loss = criterion(output, current_labels_idx)
        dummy_gradient = torch.autograd.grad(task_loss, model.parameters(), create_graph=False)

        if attack_params.loss_type == 'l2':
            grad_loss = sum(torch.sum((dg - tg) ** 2) for dg, tg in zip(dummy_gradient, target_gradient))
        elif attack_params.loss_type == 'cosine':
            flat_dummy = torch.cat([g.reshape(-1) for g in dummy_gradient])
            flat_target = torch.cat([g.reshape(-1) for g in target_gradient])
            grad_loss = 1 - F.cosine_similarity(flat_dummy, flat_target, dim=0, eps=1e-8)
        else:
            raise ValueError(f"Unknown loss_type: {attack_params.loss_type}")

        tv_loss = torch.tensor(0.0, device=device)
        if attack_params.regularization_weight > 0 and dummy_data.dim() == 4:
            diff1 = dummy_data[:, :, :, :-1] - dummy_data[:, :, :, 1:]
            diff2 = dummy_data[:, :, :-1, :] - dummy_data[:, :, 1:, :]
            tv_loss = torch.norm(diff1, p=2) + torch.norm(diff2, p=2)

        total_loss = grad_loss + attack_params.regularization_weight * tv_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            dummy_data.clamp_(0, 1)

        current_loss = grad_loss.item()
        if attack_params.return_best and current_loss < best_loss:
            best_loss = current_loss
            best_dummy_data = dummy_data.detach().clone()
            best_dummy_labels = current_labels_idx.detach().clone()

        if attack_params.log_interval and (it % attack_params.log_interval == 0 or it == attack_params.iterations - 1):
            logging.info(
                f"  Iter: {it:5d}/{attack_params.iterations} | Grad Loss: {current_loss:.4f} | TV Loss: {tv_loss.item():.4f}")

    if attack_params.return_best:
        return best_dummy_data, best_dummy_labels
    return dummy_data.detach().clone(), current_labels_idx.detach().clone()


def evaluate_inversion(
        reconstructed_images: torch.Tensor,
        ground_truth_images: torch.Tensor,
        reconstructed_labels: torch.Tensor,
        ground_truth_labels: torch.Tensor
) -> Dict[str, float]:
    """Evaluates reconstruction quality by calculating image metrics and label accuracy."""
    img1_np, img2_np = reconstructed_images.detach().cpu().numpy(), ground_truth_images.detach().cpu().numpy()
    mse = np.mean((img1_np - img2_np) ** 2)
    psnr = compare_psnr(img1_np, img2_np, data_range=1.0)
    ssim = compare_ssim(np.transpose(img1_np, (0, 2, 3, 1)), np.transpose(img2_np, (0, 2, 3, 1)), data_range=1.0,
                        channel_axis=-1, win_size=3)

    correct = (reconstructed_labels.cpu() == ground_truth_labels.cpu()).sum().item()
    label_acc = correct / ground_truth_labels.numel()

    return {"mse": mse, "psnr": psnr, "ssim": ssim, "label_acc": label_acc}


def visualize_and_save_attack(
        reconstructed_images: torch.Tensor,
        ground_truth_images: torch.Tensor,
        reconstructed_labels: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        metrics: Dict[str, float],
        save_path: Path,
        max_images: int = 4
):
    """Generates and saves a visualization comparing ground truth and reconstructed images."""
    num_images = min(len(ground_truth_images), max_images)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    if num_images == 1: axes = axes[:, np.newaxis]
    cmap = 'gray' if ground_truth_images.shape[1] == 1 else None

    for i in range(num_images):
        gt_img = ground_truth_images[i].cpu().numpy().transpose(1, 2, 0).squeeze()
        rec_img = reconstructed_images[i].cpu().numpy().transpose(1, 2, 0).squeeze()
        axes[0, i].imshow(gt_img, cmap=cmap);
        axes[0, i].set_title(f"GT: {ground_truth_labels[i].item()}");
        axes[0, i].axis('off')
        axes[1, i].imshow(rec_img, cmap=cmap);
        axes[1, i].set_title(f"Rec: {reconstructed_labels[i].item()}");
        axes[1, i].axis('off')

    title = f"PSNR: {metrics.get('psnr', -1):.2f}, SSIM: {metrics.get('ssim', -1):.3f}, Label Acc: {metrics.get('label_acc', -1):.2f}"
    fig.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95]);
    plt.savefig(save_path, dpi=200);
    plt.close(fig)


# --- 3. Orchestrator Function ---

def perform_and_evaluate_inversion_attack(
        dataset_name: str, target_gradient: List[torch.Tensor], model_template: nn.Module,
        input_shape: tuple, num_classes: int, device: torch.device, attack_config: GIAConfig,
        ground_truth_images: Optional[torch.Tensor] = None, ground_truth_labels: Optional[torch.Tensor] = None,
        save_visuals: bool = True, save_dir: Path = Path("./attack_visualizations"),
        round_num: Optional[int] = None, victim_id: Optional[str] = None
) -> Dict[str, Any]:
    """Orchestrates the full process of running, evaluating, and saving a gradient inversion attack."""
    if save_visuals: save_dir.mkdir(parents=True, exist_ok=True)

    attack_results: Dict[str, Any] = {"victim_id": victim_id, "round": round_num}
    start_time = time.time()
    try:
        attack_model = copy.deepcopy(model_template).to(device)

        if attack_config.label_type == 'ground_truth' and ground_truth_labels is None:
            logging.warning("GT labels unavailable. Switching label_type to 'optimize'.")
            attack_config.label_type = 'optimize'

        if dataset_name not in DATASET_STATS: raise ValueError(f"Stats for '{dataset_name}' not found.")
        stats = DATASET_STATS[dataset_name]

        reconstructed_images, reconstructed_labels = gradient_inversion_attack(
            target_gradient=target_gradient, model=attack_model, input_shape=input_shape,
            num_classes=num_classes, device=device, dataset_mean=stats['mean'],
            dataset_std=stats['std'], attack_params=attack_config
        )
        attack_results["duration_sec"] = time.time() - start_time

        if ground_truth_images is not None and ground_truth_labels is not None:
            eval_metrics = evaluate_inversion(
                reconstructed_images, ground_truth_images, reconstructed_labels, ground_truth_labels)
            attack_results["metrics"] = eval_metrics
            logging.info(f"  Evaluation for '{victim_id}': {eval_metrics}")

            if save_visuals:
                filename = f"round_{round_num}_victim_{victim_id}_comp.png"
                visualize_and_save_attack(reconstructed_images, ground_truth_images,
                                          reconstructed_labels, ground_truth_labels,
                                          eval_metrics, save_dir / filename)
        else:
            attack_results["metrics"] = {"status": "no_ground_truth"}
            if save_visuals:
                filename = f"round_{round_num}_victim_{victim_id}_rec.png"
                save_image(reconstructed_images.cpu(), save_dir / filename, normalize=True)

    except Exception as e:
        logging.error(f"GIA failed for victim {victim_id}: {e}", exc_info=True)
        attack_results["error"] = str(e)
        attack_results["duration_sec"] = time.time() - start_time
    return attack_results


# --- 4. Attacker Scheduling Class ---

class GradientInversionAttacker:
    """Encapsulates all logic for performing a Gradient Inversion Attack."""

    def __init__(
            self, attack_config: ServerAttackConfig, model_template: nn.Module, device: torch.device,
            save_dir: str, dataset_name: str, input_shape: Tuple[int, int, int], num_classes: int,
    ):
        self.config = attack_config.gradient_inversion_params
        self.model_template = model_template
        self.device = device
        self.save_dir = Path(save_dir) / "gradient_inversion"
        self.dataset_name, self.input_shape, self.num_classes = dataset_name, input_shape, num_classes
        self.victim_strategy = self.config.victim_strategy

    def should_run(self, round_number: int) -> bool:
        """Determines if the attack should be performed this round."""
        return (round_number > 0) and (round_number % self.config.frequency == 0)

    def execute(
            self, round_number: int, gradients_dict: Dict[str, List[torch.Tensor]],
            all_seller_ids: List[str], ground_truth_dict: Dict[str, Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, Any]]:
        """Selects a victim, tunes the attack LR, runs GIA, and returns the best result log."""
        if not all_seller_ids: return None

        if self.victim_strategy == VictimStrategy.RANDOM:
            victim_id = random.choice(all_seller_ids)
        else:
            idx = self.config.fixed_victim_idx
            if not (0 <= idx < len(all_seller_ids)):
                logging.warning(f"GIA: Fixed victim index {idx} out of bounds. Skipping.")
                return None
            victim_id = all_seller_ids[idx]

        target_gradient = gradients_dict.get(victim_id)
        victim_gt = ground_truth_dict.get(victim_id, {})
        if target_gradient is None:
            logging.warning(f"GIA: Could not retrieve gradient for victim '{victim_id}'. Skipping.")
            return None

        logging.info(f"GIA: Starting attack on victim '{victim_id}' (Round {round_number})...")
        best_log, best_psnr = None, -float('inf')

        for lr in self.config.lrs_to_try:
            logging.info(f"  Trying LR: {lr}")

            core_attack_config = GIAConfig(lr=lr, **self.config.base_attack_params)

            current_log = perform_and_evaluate_inversion_attack(
                dataset_name=self.dataset_name, target_gradient=target_gradient,
                model_template=self.model_template, input_shape=self.input_shape,
                num_classes=self.num_classes, device=self.device, attack_config=core_attack_config,
                ground_truth_images=victim_gt.get("images"), ground_truth_labels=victim_gt.get("labels"),
                save_visuals=True, save_dir=self.save_dir, round_num=round_number, victim_id=victim_id,
            )

            current_psnr = current_log.get("metrics", {}).get("psnr", -1)
            if current_psnr > best_psnr:
                best_psnr, best_log = current_psnr, current_log
                if best_log.get("metrics"): best_log["metrics"]["best_tuned_lr"] = lr

        if best_log:
            lr_str = f'{best_log.get("metrics", {}).get("best_tuned_lr", "N/A"):g}'
            logging.info(f"GIA Complete for '{victim_id}'. Best PSNR: {best_psnr:.2f} with LR: {lr_str}")
        return best_log
