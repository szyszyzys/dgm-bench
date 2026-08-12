import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common_utils.evaluation.evaluation_backdoor import evaluate_attack_performance
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation strategies."""

    def __init__(self, cfg: AppConfig, device: str, **kwargs):
        self.cfg = cfg
        self.device = device
        self.runtime_kwargs = kwargs  # Store any extra args

    @abstractmethod
    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Runs the evaluation and returns a dictionary of metrics."""
        raise NotImplementedError


class CleanEvaluator(BaseEvaluator):
    """Evaluates standard model performance (accuracy and loss)."""

    def __init__(self, cfg, device, **kwargs):
        super().__init__(cfg, device, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        if not test_loader or len(test_loader.dataset) == 0:
            logging.warning("CleanEvaluator: Test loader is empty. Returning zero metrics.")
            return {"acc": 0.0, "loss": 0.0}

        model.to(self.device)
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                # Check the batch format to handle different data modalities
                if len(batch) == 3:  # Text data: (labels, texts, lengths)
                    labels, data, _ = batch
                else:  # Image/Tabular data: (data, labels)
                    data, labels = batch

                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)

                total_loss += self.loss_fn(outputs, labels).item() * data.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += data.size(0)

        # Handle division by zero if no samples were processed
        if total_samples == 0:
            return {"acc": 0.0, "loss": 0.0}

        acc = total_correct / total_samples
        loss = total_loss / total_samples
        return {"acc": acc, "loss": loss}


from src.participants.seller.gradient_seller import AdvancedBackdoorAdversarySeller  # 1. ADD THIS IMPORT


class BackdoorEvaluator(BaseEvaluator):
    """A specialized evaluator for measuring the Attack Success Rate (ASR) of backdoor attacks."""

    def __init__(self, cfg, device, **kwargs):
        super().__init__(cfg, device, **kwargs)

        if 'backdoor' in cfg.adversary_seller_config.poisoning.type:
            # 1. Make a copy of the runtime kwargs to avoid modifying the original dict.
            kwargs_for_generator = self.runtime_kwargs.copy()

            # 2. Remove the conflicting 'model_type' key from the copied dictionary.
            #    .pop() safely removes it. The 'None' prevents an error if the key isn't there.
            kwargs_for_generator.pop('model_type', None)

            # 3. Call the function with the cleaned kwargs.
            self.backdoor_generator = AdvancedBackdoorAdversarySeller._create_poison_generator(
                adv_cfg=self.cfg.adversary_seller_config,
                model_type=self.cfg.experiment.dataset_type,
                device=device,
                **kwargs_for_generator  # Unpack the cleaned dictionary
            )
        else:
            self.backdoor_generator = None
            logging.warning("BackdoorEvaluator initialized, but no backdoor attack is configured.")

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        if not self.backdoor_generator:
            return {}

        poison_cfg = self.cfg.adversary_seller_config.poisoning

        active_params = poison_cfg.active_params

        if not active_params:
            logging.warning("BackdoorEvaluator: No active poison parameters found.")
            return {}

        try:
            target_label = active_params.target_label
        except AttributeError:
            logging.error(
                f"BackdoorEvaluator: Config/Class mismatch! {type(active_params)} has no 'target_label' attribute.")
            return {}

        metrics = evaluate_attack_performance(
            model, test_loader, self.device, self.backdoor_generator, target_label
        )

        return {
            "asr": metrics.get("attack_success_rate", 0.0),
            "B-Acc": metrics.get("clean_accuracy", 0.0),
            "B-F1": metrics.get("backdoor_f1_score", 0.0)
        }


EVALUATOR_MAP = {
    "clean": CleanEvaluator,
    "backdoor": BackdoorEvaluator,
}

# Lazy-loaded to avoid importing transformers for classification-only runs
_LLM_EVALUATOR_MAP = None


def _get_llm_evaluator_map():
    global _LLM_EVALUATOR_MAP
    if _LLM_EVALUATOR_MAP is None:
        from src.common_utils.evaluation.llm_evaluators import (
            PerplexityEvaluator, SafetyEvaluator, RefusalEvaluator,
        )
        _LLM_EVALUATOR_MAP = {
            "perplexity": PerplexityEvaluator,
            "safety": SafetyEvaluator,
            "refusal": RefusalEvaluator,
        }
    return _LLM_EVALUATOR_MAP


def create_evaluators(cfg: AppConfig, device: str, **kwargs: Dict[str, Any]) -> List[BaseEvaluator]:
    """Creates a list of evaluator instances based on the config."""
    evaluator_list = []
    evaluator_names = cfg.experiment.evaluations

    is_llm = cfg.experiment.dataset_type == "llm"

    for name in evaluator_names:
        # Check standard evaluators first
        EvaluatorClass = EVALUATOR_MAP.get(name)

        # For LLM mode, also check LLM-specific evaluators
        if EvaluatorClass is None and is_llm:
            EvaluatorClass = _get_llm_evaluator_map().get(name)

        if EvaluatorClass:
            evaluator_list.append(EvaluatorClass(cfg, device, **kwargs))
        else:
            logging.warning(f"Unknown evaluator '{name}' specified in config.")

    # For LLM mode, auto-add perplexity if no evaluators matched
    if is_llm and not evaluator_list:
        logging.info("LLM mode: auto-adding PerplexityEvaluator")
        PerplexityEval = _get_llm_evaluator_map()["perplexity"]
        evaluator_list.append(PerplexityEval(cfg, device, **kwargs))

    return evaluator_list
