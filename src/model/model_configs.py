import copy
from typing import Dict

from src.model.image_model import ImageModelConfig

# --- Pre-designed Model "Recipes" ---

_MODEL_CONFIG_REGISTRY: Dict[str, ImageModelConfig] = {
    "default": ImageModelConfig(
        config_name="default",
        model_name="FlexibleCNN",
        use_dropout=True,
        dropout_rate=0.1,
        use_batch_norm=True,
        activation="relu",
        epochs=50,
        batch_size=128,
        scheduler_gamma=0.1,  # Multiply LR by 0.1
        weight_decay=1e-4,
    ),
    "cifar10_cnn": ImageModelConfig(
        config_name="cifar10_cnn",
        model_name="FlexibleCNN",
        use_dropout=True,
        dropout_rate=0.2,
        use_batch_norm=False,
        activation="relu",
        conv_channels=[32, 64, 128],
        classifier_layers=[256, 128],
        use_scheduler=True,
        scheduler_step=30,  # Reduce LR every 30 epochs
        scheduler_gamma=0.1,  # Multiply LR by 0.1
        weight_decay=1e-4,
        use_group_norm=True,  # <-- Add this
        num_groups=32,  # <-- Add this (32 works well for 64, 128, 256)
    ),
    "cifar10_resnet18": ImageModelConfig(
        config_name="cifar10_resnet18",
        model_name="ResNet18",
        use_dropout=True,
        dropout_rate=0.25,
        use_batch_norm=False,
        activation="relu",
        epochs=150,
        batch_size=64,
        learning_rate=0.001,
        use_scheduler=True,
        scheduler_step=30,  # Reduce LR every 30 epochs
        scheduler_gamma=0.1,  # Multiply LR by 0.1
        weight_decay=1e-4,
        use_group_norm=True,  # <-- Add this
        num_groups=32,  # <-- Add this (32 works well for 64, 128, 256)

    ),
    "cifar100_cnn": ImageModelConfig(
        config_name="cifar100_cnn",
        model_name="FlexibleCNN",
        use_dropout=True,
        dropout_rate=0.3,
        use_batch_norm=False,
        activation="relu",
        conv_channels=[64, 128, 256],
        classifier_layers=[512, 256],
        use_scheduler=True,
        scheduler_step=30,  # Reduce LR every 30 epochs
        scheduler_gamma=0.1,  # Multiply LR by 0.1
        weight_decay=1e-4,
        use_group_norm=True,  # <-- Add this
        num_groups=32,  # <-- Add this (32 works well for 64, 128, 256)
    ),

    "femnist_cnn": ImageModelConfig(
        config_name="femnist_cnn",
        model_name="FlexibleCNN",
        use_dropout=True,
        dropout_rate=0.2,
        use_batch_norm=False,
        activation="relu",
        conv_channels=[32, 64, 128],
        classifier_layers=[256, 128],
        use_scheduler=True,
        scheduler_step=30,
        scheduler_gamma=0.1,
        weight_decay=1e-4,
        use_group_norm=True,
        num_groups=32,
    ),
    "cifar100_resnet18": ImageModelConfig(
        config_name="cifar100_resnet18",
        model_name="ResNet18",
        use_dropout=True,
        dropout_rate=0.3,
        use_batch_norm=False,
        activation="relu",
        epochs=200,
        # --- FIX: Reduced batch size to lower GPU memory usage ---
        batch_size=64,
        learning_rate=0.001,
        use_scheduler=True,
        scheduler_step=30,  # Reduce LR every 30 epochs
        scheduler_gamma=0.1,  # Multiply LR by 0.1
        weight_decay=1e-4,
        use_group_norm=True,  # <-- Add this
        num_groups=32,  # <-- Add this (32 works well for 64, 128, 256)

    ),
}


def get_image_model_config(name: str) -> ImageModelConfig:
    """
    Acts as a "cookbook" to retrieve a pre-designed model configuration by name.
    """
    return copy.deepcopy(_MODEL_CONFIG_REGISTRY.get(name, _MODEL_CONFIG_REGISTRY["default"]))
