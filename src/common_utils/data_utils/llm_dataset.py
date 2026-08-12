"""
LLM Dataset Pipeline for Federated Gradient Marketplace
========================================================

Loads and partitions LLM finetuning datasets with NATURAL user-ID partitioning.
Supports:
- Fed-ChatbotIT (instruction tuning, single/multi-turn)
- Fed-WildChat (instruction tuning, real user queries)
- Fed-ChatbotPA (preference alignment / DPO)

Each unique user_id becomes one "seller" in the marketplace.
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type (mirrors ProcessedTextData pattern)
# ---------------------------------------------------------------------------

@dataclass
class ProcessedLLMData:
    """Holds the results of the LLM data processing pipeline."""
    buyer_loader: Optional[DataLoader]
    seller_loaders: Dict[int, Optional[DataLoader]]
    test_loader: Optional[DataLoader]
    tokenizer: Any          # HuggingFace tokenizer
    collate_fn: Callable
    num_sellers: int
    task: str               # "sft" or "dpo"
    stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Instruction-tuning dataset for causal language modeling.
    Each item is a tokenized (input_ids, attention_mask, labels) tuple.
    """

    def __init__(self, conversations: List[Dict], tokenizer, max_seq_length: int,
                 chat_template: str = "default"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chat_template = chat_template
        self.data = conversations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = self._format_conversation(item)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        return input_ids, attention_mask, labels

    def _format_conversation(self, item: Dict) -> str:
        """Format a conversation into a single string using chat template."""
        # Handle different dataset formats
        if "instruction" in item and "response" in item:
            instruction = item["instruction"]
            inp = item.get("input", "")
            response = item["response"]
            if inp:
                instruction = f"{instruction}\n{inp}"
            return f"<|user|>\n{instruction}\n<|assistant|>\n{response}"

        if "messages" in item:
            parts = []
            for msg in item["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            return "\n".join(parts)

        if "conversation" in item:
            return item["conversation"]

        if "text" in item:
            return item["text"]

        raise ValueError(f"Unknown data format, keys: {list(item.keys())}")


# ---------------------------------------------------------------------------
# DPO Dataset
# ---------------------------------------------------------------------------

class DPODataset(Dataset):
    """
    Preference alignment dataset for Direct Preference Optimization.
    Each item yields (prompt, chosen, rejected) tokenized triplet.
    """

    def __init__(self, data: List[Dict], tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get("instruction", item.get("prompt", ""))
        chosen = item.get("chosen", item.get("chosen_response", ""))
        rejected = item.get("rejected", item.get("rejected_response", ""))

        chosen_text = f"<|user|>\n{prompt}\n<|assistant|>\n{chosen}"
        rejected_text = f"<|user|>\n{prompt}\n<|assistant|>\n{rejected}"

        chosen_enc = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_seq_length,
            padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_seq_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def sft_collate_fn(batch):
    """Collate SFT batch: stack (input_ids, attention_mask, labels)."""
    input_ids = torch.stack([b[0] for b in batch])
    attention_mask = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def dpo_collate_fn(batch):
    """Collate DPO batch: stack chosen/rejected pairs."""
    return {
        "chosen_input_ids": torch.stack([b["chosen_input_ids"] for b in batch]),
        "chosen_attention_mask": torch.stack([b["chosen_attention_mask"] for b in batch]),
        "rejected_input_ids": torch.stack([b["rejected_input_ids"] for b in batch]),
        "rejected_attention_mask": torch.stack([b["rejected_attention_mask"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Dataset loading and natural partitioning
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "fed_chatbot_it": {"hf_path": "FedML/Fed-ChatbotIT", "task": "sft",
                       "user_field": "user_id", "split_field": None},
    "fed_wildchat": {"hf_path": "FedML/Fed-WildChat", "task": "sft",
                     "user_field": "user_id", "split_field": None},
    "fed_chatbot_pa": {"hf_path": "FedML/Fed-ChatbotPA", "task": "dpo",
                       "user_field": "annotator_id", "split_field": None},
}


def _load_and_group_by_user(
    dataset_source: str,
    hf_override: str,
    min_samples: int,
    data_root: str,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Any]]:
    """
    Load a FedLLM-Bench dataset and group samples by user_id.
    Filters out users with fewer than min_samples.

    Returns:
        user_data: {user_id: [list of conversation dicts]}
        dataset_info: metadata about the dataset
    """
    from datasets import load_dataset as hf_load

    registry = DATASET_REGISTRY.get(dataset_source)
    if registry is None:
        raise ValueError(
            f"Unknown dataset: {dataset_source}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    hf_path = hf_override if hf_override else registry["hf_path"]
    user_field = registry["user_field"]

    logger.info(f"Loading dataset: {hf_path}")
    ds = hf_load(hf_path, cache_dir=data_root)

    # Handle train/test splits
    if "train" in ds:
        raw_data = list(ds["train"])
    else:
        raw_data = list(ds[list(ds.keys())[0]])

    # Group by user
    user_data = defaultdict(list)
    for item in raw_data:
        uid = item.get(user_field, "unknown")
        user_data[str(uid)].append(item)

    # Filter by minimum samples
    before_filter = len(user_data)
    user_data = {
        uid: samples for uid, samples in user_data.items()
        if len(samples) >= min_samples
    }
    after_filter = len(user_data)

    total_samples = sum(len(v) for v in user_data.values())
    logger.info(
        f"Dataset grouped: {after_filter} users ({before_filter - after_filter} filtered "
        f"with < {min_samples} samples), {total_samples} total samples"
    )

    dataset_info = {
        "source": dataset_source,
        "hf_path": hf_path,
        "task": registry["task"],
        "total_users": after_filter,
        "total_samples": total_samples,
        "filtered_users": before_filter - after_filter,
    }

    return user_data, dataset_info


def get_llm_dataset(cfg) -> ProcessedLLMData:
    """
    Main entry point: load, partition, and create DataLoaders for LLM marketplace.

    Partitioning is NATURAL by user_id (no artificial Dirichlet splitting).
    A fraction of users are reserved as buyer data for root gradient computation.
    """
    llm_cfg = cfg.data.llm
    if llm_cfg is None:
        raise ValueError("LLM data configuration ('data.llm') is missing.")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 1. Load and group by user
    user_data, dataset_info = _load_and_group_by_user(
        dataset_source=llm_cfg.dataset_source,
        hf_override=llm_cfg.hf_dataset_path,
        min_samples=llm_cfg.min_samples_per_seller,
        data_root=cfg.data_root,
    )

    # 2. Get tokenizer
    from src.model.llm_models import get_tokenizer
    tokenizer = get_tokenizer(llm_cfg.base_model_name)

    # 3. Split users into buyer pool and seller pool
    user_ids = sorted(user_data.keys())
    random.shuffle(user_ids)

    n_buyer_users = max(1, int(len(user_ids) * llm_cfg.buyer_ratio))
    buyer_user_ids = user_ids[:n_buyer_users]
    seller_user_ids = user_ids[n_buyer_users:]

    # Limit sellers to n_sellers from config
    n_sellers = cfg.experiment.n_sellers
    if len(seller_user_ids) > n_sellers:
        seller_user_ids = seller_user_ids[:n_sellers]
    actual_n_sellers = len(seller_user_ids)

    logger.info(
        f"User split: {len(buyer_user_ids)} buyer users, "
        f"{actual_n_sellers} seller users (requested {n_sellers})"
    )

    # 4. Create datasets
    is_dpo = llm_cfg.task == "dpo"
    DatasetClass = DPODataset if is_dpo else SFTDataset
    collate = dpo_collate_fn if is_dpo else sft_collate_fn

    # Buyer data: pool all buyer users' data
    buyer_samples = []
    for uid in buyer_user_ids:
        buyer_samples.extend(user_data[uid])

    # Split buyer into train (root gradient) and test (validation)
    random.shuffle(buyer_samples)
    split_idx = max(1, len(buyer_samples) // 2)
    buyer_train_samples = buyer_samples[:split_idx]
    buyer_test_samples = buyer_samples[split_idx:]

    ds_kwargs = {"tokenizer": tokenizer, "max_seq_length": llm_cfg.max_seq_length}
    if not is_dpo:
        ds_kwargs["chat_template"] = llm_cfg.chat_template

    buyer_dataset = DatasetClass(buyer_train_samples, **ds_kwargs)
    test_dataset = DatasetClass(buyer_test_samples, **ds_kwargs)

    batch_size = cfg.training.batch_size

    buyer_loader = DataLoader(
        buyer_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=True,
    ) if len(buyer_dataset) > 0 else None

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=0, pin_memory=True,
    ) if len(test_dataset) > 0 else None

    # 5. Create per-seller DataLoaders
    seller_loaders = {}
    seller_sample_counts = {}
    for i, uid in enumerate(seller_user_ids):
        seller_samples = user_data[uid]
        seller_ds = DatasetClass(seller_samples, **ds_kwargs)
        seller_loaders[i] = DataLoader(
            seller_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate, num_workers=0, pin_memory=True,
        )
        seller_sample_counts[i] = len(seller_samples)

    # 6. Compile stats
    stats = {
        **dataset_info,
        "n_buyer_users": len(buyer_user_ids),
        "n_seller_users": actual_n_sellers,
        "buyer_train_samples": len(buyer_train_samples),
        "buyer_test_samples": len(buyer_test_samples),
        "seller_sample_counts": seller_sample_counts,
        "avg_samples_per_seller": (
            np.mean(list(seller_sample_counts.values())) if seller_sample_counts else 0
        ),
        "peft_method": llm_cfg.peft_method,
        "max_seq_length": llm_cfg.max_seq_length,
    }
    logger.info(
        f"LLM data ready: {stats['buyer_train_samples']} buyer train, "
        f"{stats['buyer_test_samples']} test, "
        f"{actual_n_sellers} sellers (avg {stats['avg_samples_per_seller']:.0f} samples each)"
    )

    return ProcessedLLMData(
        buyer_loader=buyer_loader,
        seller_loaders=seller_loaders,
        test_loader=test_loader,
        tokenizer=tokenizer,
        collate_fn=collate,
        num_sellers=actual_n_sellers,
        task=llm_cfg.task,
        stats=stats,
    )
