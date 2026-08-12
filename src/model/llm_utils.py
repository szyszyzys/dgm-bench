"""
LLM Training Utilities for Federated Gradient Marketplace
==========================================================

Provides local training functions for SFT and DPO that are compatible
with the existing gradient exchange interface:
    Input:  model (FederatedLoRA/FederatedSoftPrompt), train_loader
    Output: (pseudo_gradients: List[Tensor], avg_loss: float)

Pseudo-gradients = initial_lora_params - final_lora_params (FedAvg convention).
"""

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


def llm_local_training_and_get_gradient(
    model: nn.Module,
    train_loader,
    device: torch.device,
    local_epochs: int = 1,
    lr: float = 2e-5,
    opt_str: str = "Adam",
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    task: str = "sft",
    **kwargs,
) -> Tuple[Optional[List[torch.Tensor]], Optional[float]]:
    """
    Train a FederatedLoRA/FederatedSoftPrompt model locally, return weight diffs.

    This follows the same convention as local_training_and_get_gradient():
        pseudo_gradient = initial_params - final_params

    Only trainable (LoRA/soft prompt) parameters are included in the gradient.
    """
    # 1. Store initial LoRA/prompt parameters
    trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning("No trainable parameters found in LLM model.")
        return None, None

    initial_params = {name: p.data.clone() for name, p in trainable_params}

    # 2. Set up optimizer
    param_groups = [{"params": [p for _, p in trainable_params]}]
    if opt_str.lower() == "adam" or opt_str.lower() == "adamw":
        optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(param_groups, lr=lr, weight_decay=weight_decay)

    # 3. Train
    model.train()
    total_loss = 0.0
    total_steps = 0

    if task == "dpo":
        avg_loss = _train_dpo(model, train_loader, optimizer, device,
                              local_epochs, max_grad_norm, gradient_accumulation_steps)
    else:
        avg_loss = _train_sft(model, train_loader, optimizer, device,
                              local_epochs, max_grad_norm, gradient_accumulation_steps)

    # 4. Compute pseudo-gradient: initial - final (FedAvg convention)
    pseudo_gradients = []
    for name, param in trainable_params:
        diff = initial_params[name] - param.data
        pseudo_gradients.append(diff.detach().cpu())

    # 5. Log stats
    grad_norm = sum(g.norm().item() ** 2 for g in pseudo_gradients) ** 0.5
    total_params = sum(g.numel() for g in pseudo_gradients)
    logger.debug(
        f"LLM local training done: loss={avg_loss:.4f}, "
        f"grad_norm={grad_norm:.4f}, "
        f"trainable_params={total_params}, "
        f"size={total_params * 4 / 1024 / 1024:.2f} MB"
    )

    return pseudo_gradients, avg_loss


def _train_sft(model, train_loader, optimizer, device,
               local_epochs, max_grad_norm, grad_accum_steps) -> float:
    """SFT training: standard causal language modeling loss."""
    total_loss = 0.0
    total_steps = 0

    for epoch in range(local_epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at epoch {epoch}, step {step}. Skipping.")
                optimizer.zero_grad()
                continue

            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            total_steps += 1

    return total_loss / max(total_steps, 1)


def _train_dpo(model, train_loader, optimizer, device,
               local_epochs, max_grad_norm, grad_accum_steps,
               beta: float = 0.1) -> float:
    """
    DPO training: Direct Preference Optimization loss.

    L_DPO = -log(sigma(beta * (log pi(y_w|x) - log pi(y_l|x))))

    Uses the model's own initial outputs as the reference (self-play DPO).
    """
    total_loss = 0.0
    total_steps = 0

    # Get reference log-probs (frozen at start of round)
    ref_logprobs = {}

    for epoch in range(local_epochs):
        for step, batch in enumerate(train_loader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            # Compute log-probs for chosen and rejected
            chosen_logprobs = _get_batch_logprobs(model, chosen_ids, chosen_mask)
            rejected_logprobs = _get_batch_logprobs(model, rejected_ids, rejected_mask)

            # Reference log-probs (detached — from initial model state on first pass)
            with torch.no_grad():
                if epoch == 0 and step == 0:
                    ref_chosen = chosen_logprobs.detach()
                    ref_rejected = rejected_logprobs.detach()
                else:
                    ref_chosen = _get_batch_logprobs(model, chosen_ids, chosen_mask).detach()
                    ref_rejected = _get_batch_logprobs(model, rejected_ids, rejected_mask).detach()

            # DPO loss
            chosen_rewards = beta * (chosen_logprobs - ref_chosen)
            rejected_rewards = beta * (rejected_logprobs - ref_rejected)
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            loss = loss / grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            total_steps += 1

    return total_loss / max(total_steps, 1)


def _get_batch_logprobs(model, input_ids, attention_mask) -> torch.Tensor:
    """Compute per-sequence average log probability."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # Shift: predict next token
    labels = input_ids[:, 1:]           # Shift labels
    mask = attention_mask[:, 1:].float()

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    # Average over non-padded tokens per sequence
    seq_log_probs = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return seq_log_probs


def evaluate_llm_perplexity(model, eval_loader, device) -> float:
    """Evaluate model perplexity on a validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels", input_ids.clone()).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Count non-padding tokens
            n_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # Clamp to avoid overflow
    return perplexity
