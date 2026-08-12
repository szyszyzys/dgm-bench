import torch

from src.marketplace.utils.gradient_market_utils.gradient_market_configs import (
    AppConfig, ExperimentConfig, TrainingConfig, AdversarySellerConfig, DataConfig, ImageDataConfig, PropertySkewParams,
    VocabConfig, TextDataConfig, DebugConfig, ServerAttackConfig, AggregationConfig
)


def get_base_image_config() -> AppConfig:
    """Creates the default, base AppConfig for IMAGE-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="CelebA", model_structure="cnn",
            global_rounds=500, n_sellers=10, adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu", dataset_type="image",
            evaluations=["clean", "backdoor"],
        ),
        training=TrainingConfig(
            local_epochs=2,
            batch_size=64,
            learning_rate=0.01,
            optimizer="sgd",
            momentum=0.9,
            weight_decay=0.0001
        ),
        server_attack_config=ServerAttackConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            image=ImageDataConfig(
                property_skew=PropertySkewParams(),
                strategy="dirichlet",
                dirichlet_alpha=0.5,
                buyer_ratio=0.1,
                buyer_strategy="iid"
            ),
            text=None,
            tabular=None,
        ),
        debug=DebugConfig(
            save_individual_gradients=False,
            gradient_save_frequency=10
        ),
        seed=42, n_samples=3,
        aggregation=AggregationConfig(
            method="fedavg"
        ),
    )
def get_base_text_config() -> AppConfig:
    """Creates the default, base AppConfig for TEXT-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="AG_NEWS", model_structure="text_cnn",
            global_rounds=500, n_sellers=10, adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu", dataset_type="text",
            evaluations=["clean", "backdoor"]
        ),
        training=TrainingConfig(
            local_epochs=2,
            batch_size=64,
            optimizer='sgd',
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.0001
            # ----------------------------------------------
        ),
        server_attack_config=ServerAttackConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            image=None,
            text=TextDataConfig(
                vocab=VocabConfig(),
                strategy="dirichlet",
                dirichlet_alpha=0.5,
                buyer_ratio=0.1,
                buyer_strategy="iid"
            ),
            tabular=None,
        ),
        debug=DebugConfig(
            save_individual_gradients=False,
            gradient_save_frequency=10
        ),
        seed=42, n_samples=3,
        aggregation=AggregationConfig(
            method="fedavg"
        ),

    )