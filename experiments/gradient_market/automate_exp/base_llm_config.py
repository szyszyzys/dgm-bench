"""
Base LLM Configuration for Federated Gradient Marketplace
==========================================================

Provides default AppConfig factories for LLM finetuning experiments.
"""

import copy

import torch

from src.marketplace.utils.gradient_market_utils.gradient_market_configs import (
    AppConfig, ExperimentConfig, TrainingConfig, AdversarySellerConfig,
    DataConfig, DebugConfig, ServerAttackConfig, AggregationConfig,
    LLMDataConfig, LoRAConfig, SoftPromptConfig,
)


def get_base_llm_sft_config() -> AppConfig:
    """Base config for LLM instruction-tuning (SFT) marketplace experiments."""
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="fed_chatbot_it",
            model_structure="lora",
            global_rounds=50,
            n_sellers=20,
            adv_rate=0.0,
            device=_device,
            dataset_type="llm",
            evaluations=["perplexity", "safety", "refusal"],
            eval_frequency=5,
        ),
        training=TrainingConfig(
            local_epochs=1,
            batch_size=4,
            learning_rate=2e-5,
            optimizer="Adam",
            momentum=0.0,
            weight_decay=0.01,
        ),
        server_attack_config=ServerAttackConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            llm=LLMDataConfig(
                dataset_source="fed_chatbot_it",
                max_seq_length=512,
                min_samples_per_seller=10,
                task="sft",
                buyer_ratio=0.1,
                peft_method="lora",
                base_model_name="Qwen/Qwen2.5-1.5B",
                lora=LoRAConfig(
                    rank=8,
                    alpha=16,
                    dropout=0.05,
                    target_modules=["q_proj", "v_proj"],
                    use_qlora=(_device != "cpu"),  # bitsandbytes requires GPU
                ),
            ),
        ),
        debug=DebugConfig(save_individual_gradients=False),
        seed=42,
        n_samples=1,
        aggregation=AggregationConfig(method="fedavg"),
    )


def get_base_llm_dpo_config() -> AppConfig:
    """Base config for LLM preference alignment (DPO) marketplace experiments."""
    cfg = copy.deepcopy(get_base_llm_sft_config())
    cfg.experiment.dataset_name = "fed_chatbot_pa"
    cfg.data.llm.dataset_source = "fed_chatbot_pa"
    cfg.data.llm.task = "dpo"
    cfg.training.learning_rate = 5e-6  # DPO typically needs lower LR
    return cfg
