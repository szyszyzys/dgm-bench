"""
LLM Model Wrappers for Federated Gradient Marketplace
======================================================

Provides two PEFT wrappers that expose only the tradeable parameters
via parameters()/state_dict(), making them compatible with the existing
gradient exchange pipeline (aggregators, valuations, attacks).

The frozen base LLM is loaded once and shared across all sellers via
a module-level cache. Only LoRA matrices or soft prompt embeddings
are exchanged as "gradients" in the marketplace.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache for the frozen base model (loaded once per process)
# ---------------------------------------------------------------------------
_BASE_MODEL_CACHE: Dict[str, "torch.nn.Module"] = {}
_TOKENIZER_CACHE: Dict[str, object] = {}


def get_base_model_and_tokenizer(model_name: str, llm_cfg):
    """
    Load and cache the frozen base model + tokenizer.
    Uses QLoRA 4-bit quantization when configured.
    """
    if model_name in _BASE_MODEL_CACHE:
        return _BASE_MODEL_CACHE[model_name], _TOKENIZER_CACHE[model_name]

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading base model: {model_name} (first load, will be cached)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build quantization config if QLoRA requested
    quantization_config = None
    if llm_cfg.lora.use_qlora:
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        compute_dtype = dtype_map.get(llm_cfg.lora.bnb_4bit_compute_dtype, torch.bfloat16)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=llm_cfg.lora.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map.get(llm_cfg.torch_dtype, torch.bfloat16)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        device_map="auto" if quantization_config else None,
    )

    # Freeze the entire base model
    for param in base_model.parameters():
        param.requires_grad = False

    _BASE_MODEL_CACHE[model_name] = base_model
    _TOKENIZER_CACHE[model_name] = tokenizer

    total_params = sum(p.numel() for p in base_model.parameters())
    logger.info(f"Base model loaded: {total_params / 1e6:.1f}M params (frozen, cached)")

    return base_model, tokenizer


def get_tokenizer(model_name: str):
    """Get cached tokenizer (must call get_base_model_and_tokenizer first)."""
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


# ---------------------------------------------------------------------------
# FederatedLoRA: LoRA adapter as a standalone tradeable module
# ---------------------------------------------------------------------------

class FederatedLoRA(nn.Module):
    """
    Wraps a PEFT LoRA adapter around a frozen base LLM.

    Only the LoRA parameters are exposed via parameters()/state_dict(),
    so the gradient marketplace exchanges only the low-rank matrices.
    The base model is shared via module-level cache and never transmitted.

    Compatible with:
    - All existing aggregators (FedAvg, FLTrust, MartFL, SPMC, etc.)
    - All existing attacks (backdoor, Sybil, drowning, etc.)
    - All existing valuations (similarity, influence, LOO, Shapley)
    """

    def __init__(self, model_name: str, llm_cfg, device: str = "cpu"):
        super().__init__()
        from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

        self.model_name = model_name
        self._device_str = device

        # Get shared frozen base model
        base_model, self.tokenizer = get_base_model_and_tokenizer(model_name, llm_cfg)

        # Prepare for QLoRA if using quantization
        if llm_cfg.lora.use_qlora:
            base_model = prepare_model_for_kbit_training(base_model)

        # Attach LoRA adapters
        lora_config = LoraConfig(
            r=llm_cfg.lora.rank,
            lora_alpha=llm_cfg.lora.alpha,
            lora_dropout=llm_cfg.lora.dropout,
            target_modules=llm_cfg.lora.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.peft_model = get_peft_model(base_model, lora_config)

        trainable, total = self.peft_model.get_nb_trainable_parameters()
        logger.info(
            f"FederatedLoRA: {trainable / 1e6:.2f}M trainable / "
            f"{total / 1e6:.1f}M total ({100 * trainable / total:.2f}%)"
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def parameters(self, recurse=True):
        """Only yield LoRA (trainable) parameters."""
        for p in self.peft_model.parameters():
            if p.requires_grad:
                yield p

    def named_parameters(self, prefix='', recurse=True):
        """Only yield named LoRA parameters."""
        for name, p in self.peft_model.named_parameters():
            if p.requires_grad:
                yield (prefix + name if prefix else name), p

    def state_dict(self, *args, **kwargs):
        """Only return LoRA adapter state."""
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(self.peft_model)

    def load_state_dict(self, state_dict, strict=False):
        """Load LoRA adapter state (ignoring base model params)."""
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.peft_model, state_dict)

    def train(self, mode=True):
        self.peft_model.train(mode)
        return self

    def eval(self):
        self.peft_model.eval()
        return self

    def generate(self, **kwargs):
        return self.peft_model.generate(**kwargs)


# ---------------------------------------------------------------------------
# FederatedSoftPrompt: Soft prompt embeddings as tradeable module
# ---------------------------------------------------------------------------

class FederatedSoftPrompt(nn.Module):
    """
    Wraps soft prompt tuning (FLiP-style) around a frozen base LLM.

    Only the continuous prompt embeddings are exchanged in the marketplace.
    Extremely communication-efficient: typically < 100KB per seller update.
    """

    def __init__(self, model_name: str, llm_cfg, device: str = "cpu"):
        super().__init__()
        from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit

        self.model_name = model_name

        base_model, self.tokenizer = get_base_model_and_tokenizer(model_name, llm_cfg)

        sp_cfg = llm_cfg.soft_prompt
        init_type = (PromptTuningInit.TEXT if sp_cfg.prompt_tuning_init == "TEXT"
                     else PromptTuningInit.RANDOM)

        pt_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=sp_cfg.num_virtual_tokens,
            prompt_tuning_init=init_type,
            prompt_tuning_init_text=sp_cfg.prompt_tuning_init_text if init_type == PromptTuningInit.TEXT else None,
            tokenizer_name_or_path=model_name,
        )
        self.peft_model = get_peft_model(base_model, pt_config)

        trainable, total = self.peft_model.get_nb_trainable_parameters()
        logger.info(
            f"FederatedSoftPrompt: {trainable} trainable / "
            f"{total / 1e6:.1f}M total ({100 * trainable / total:.4f}%)"
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def parameters(self, recurse=True):
        for p in self.peft_model.parameters():
            if p.requires_grad:
                yield p

    def named_parameters(self, prefix='', recurse=True):
        for name, p in self.peft_model.named_parameters():
            if p.requires_grad:
                yield (prefix + name if prefix else name), p

    def state_dict(self, *args, **kwargs):
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(self.peft_model)

    def load_state_dict(self, state_dict, strict=False):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.peft_model, state_dict)

    def train(self, mode=True):
        self.peft_model.train(mode)
        return self

    def eval(self):
        self.peft_model.eval()
        return self

    def generate(self, **kwargs):
        return self.peft_model.generate(**kwargs)
