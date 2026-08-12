import copy
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict
from dataclasses import field, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from typing import Callable, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BaseModelConfig:
    """Base class for all model configurations, containing shared metadata."""
    # Metadata
    config_name: str = "default"
    description: str = "Default configuration"
    source: str = "manual"
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    target_dataset: str = "any"
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageModelConfig(BaseModelConfig):
    """Configuration class for model architecture and training parameters."""
    # Model selection
    model_name: str = "FlexibleCNN"  # "LeNet", "FlexibleCNN"

    # Architecture parameters
    use_dropout: bool = True
    dropout_rate: float = 0.5
    use_batch_norm: bool = False
    activation: str = "relu"  # "relu", "tanh", "sigmoid"

    # FlexibleCNN specific
    conv_channels: list = None  # [64, 128, 256]
    classifier_layers: list = None  # [256, 128]

    # LeNet specific
    lenet_features: int = 16  # second conv layer channels
    lenet_fc1: int = 120
    lenet_fc2: int = 84

    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    momentum: float = 0.9
    use_scheduler: bool = True
    scheduler_step: int = 50
    scheduler_gamma: float = 0.5
    use_early_stopping: bool = True
    patience: int = 10
    optimizer_type: str = 'sgd'
    # Data augmentation
    use_augmentation: bool = False
    use_normalization: bool = True
    use_group_norm: bool = True
    num_groups: int = 32

    # Experiment metadatas

    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [64, 128, 256]
        if self.classifier_layers is None:
            self.classifier_layers = [256]

    def save_config(self, filepath: str):
        """Saves the current configuration object to a JSON file."""
        # Ensure the directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # Write the dataclass object as a dictionary to the file
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)
        print(f"  - 💾 Config saved to {filepath}")


class ConfigurableLeNet(nn.Module):
    """Configurable LeNet-5 model based on ModelConfig."""

    def __init__(self, input_channels: int, image_size: Tuple[int, int],
                 num_classes: int, config: ImageModelConfig):
        super(ConfigurableLeNet, self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.config = config

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activation_map[config.activation]

        # Feature extractor
        layers = [
            nn.Conv2d(self.input_channels, 6, kernel_size=5, stride=1, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, config.lenet_features, kernel_size=5),
            self.activation,
            nn.MaxPool2d(kernel_size=2)
        ]

        if config.use_batch_norm:
            # Insert batch norm after conv layers
            layers.insert(2, nn.BatchNorm2d(6))
            layers.insert(6, nn.BatchNorm2d(config.lenet_features))

        self.features = nn.Sequential(*layers)

        # Calculate flattened size
        flattened_size = self._get_flattened_size()

        # Classifier
        classifier_layers = [nn.Linear(flattened_size, config.lenet_fc1)]
        if config.use_dropout:
            classifier_layers.append(nn.Dropout(config.dropout_rate))
        classifier_layers.extend([
            self.activation,
            nn.Linear(config.lenet_fc1, config.lenet_fc2),
        ])
        if config.use_dropout:
            classifier_layers.append(nn.Dropout(config.dropout_rate))
        classifier_layers.extend([
            self.activation,
            nn.Linear(config.lenet_fc2, self.num_classes)
        ])

        self.classifier = nn.Sequential(*classifier_layers)

    def _get_flattened_size(self) -> int:
        dummy_input = torch.randn(1, self.input_channels, self.image_size[0], self.image_size[1])
        dummy_output = self.features(dummy_input)
        return dummy_output.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ConfigurableFlexibleCNN(nn.Module):
    """Configurable FlexibleCNN model based on ModelConfig."""

    def __init__(self, input_channels: int, image_size: Tuple[int, int],
                 num_classes: int, config: ImageModelConfig):
        super(ConfigurableFlexibleCNN, self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.config = config

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activation_map[config.activation]

        # Build feature extractor dynamically
        features = []
        in_channels = self.input_channels

        for i, out_channels in enumerate(config.conv_channels):
            # First conv block

            if config.use_batch_norm:
                norm_layer = nn.BatchNorm2d(out_channels)
            # Check for new 'use_group_norm' attribute in your config
            elif getattr(config, 'use_group_norm', False):
                # Get 'num_groups' from config, default to 32 if not present
                num_groups = getattr(config, 'num_groups', 32)

                # Add this check: GroupNorm requires num_channels to be divisible by num_groups
                if out_channels % num_groups != 0:
                    # Find the largest group size that works (e.g., 16, 8, ...)
                    valid_groups = next(g for g in range(num_groups, 0, -1) if out_channels % g == 0)
                    print(f"Warning: {out_channels} channels not divisible by {num_groups} groups. "
                          f"Falling back to {valid_groups} groups.")
                    num_groups = valid_groups

                norm_layer = nn.GroupNorm(num_groups, out_channels)
            else:
                norm_layer = nn.Identity()

            features.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                norm_layer,
                self.activation
            ])

            # Second conv block (except for last layer)
            if i < len(config.conv_channels) - 1:
                # Create a separate norm_layer instance for the second block
                if config.use_batch_norm:
                    norm_layer2 = nn.BatchNorm2d(out_channels)
                elif getattr(config, 'use_group_norm', False):
                    norm_layer2 = nn.GroupNorm(num_groups, out_channels)
                else:
                    norm_layer2 = nn.Identity()
                features.extend([
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    norm_layer2,
                    self.activation
                ])

            features.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        self.features = nn.Sequential(*features)

        # Calculate flattened size SAFELY
        flattened_size = self._get_flattened_size_safe()

        if flattened_size <= 0:
            raise ValueError(f"Invalid flattened size: {flattened_size}. Check image size and pooling layers.")

        # Build classifier dynamically
        classifier = []
        prev_size = flattened_size

        for layer_size in config.classifier_layers:
            if config.use_dropout:
                classifier.append(nn.Dropout(config.dropout_rate))
            classifier.extend([
                nn.Linear(prev_size, layer_size),
                self.activation
            ])
            prev_size = layer_size

        # Final output layer
        classifier.append(nn.Linear(prev_size, self.num_classes))
        self.classifier = nn.Sequential(*classifier)

        # CRITICAL: Reinitialize all parameters to ensure no NaN/Inf
        self._safe_initialization()

    def _get_flattened_size_safe(self) -> int:
        """Calculate flattened size mathematically without forward pass."""
        h, w = self.image_size
        num_pools = len(self.config.conv_channels)

        # Calculate spatial dimensions after all pooling
        h_out = h // (2 ** num_pools)
        w_out = w // (2 ** num_pools)

        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Image size {self.image_size} is too small for {num_pools} pooling layers. "
                f"Resulting dimensions: {h_out}x{w_out}"
            )

        # Final channels is the last conv layer's output
        final_channels = self.config.conv_channels[-1]
        flattened_size = final_channels * h_out * w_out

        return flattened_size

    def _safe_initialization(self):
        """Reinitialize all parameters to prevent NaN/Inf values."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # Use Xavier/Glorot initialization for weights
                    nn.init.xavier_uniform_(param, gain=1.0)
                else:
                    # For 1D weights (shouldn't happen but just in case)
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.zeros_(param)

            # Verify no NaN/Inf after initialization
            if torch.isnan(param).any() or torch.isinf(param).any():
                # Last resort: fill with small random values
                param.data.uniform_(-0.01, 0.01)
                logging.warning(f"Had to manually fill parameter '{name}' due to persistent NaN/Inf")

    def _get_flattened_size(self) -> int:
        """Calculate flattened size mathematically without forward pass."""
        h, w = self.image_size

        # Each conv block has MaxPool2d(2, 2) which halves dimensions
        num_pools = len(self.config.conv_channels)

        # Calculate spatial dimensions after all pooling
        h_out = h // (2 ** num_pools)
        w_out = w // (2 ** num_pools)

        # Final channels is the last conv layer's output
        final_channels = self.config.conv_channels[-1]

        flattened_size = final_channels * h_out * w_out

        return flattened_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """A standard ResNet residual block, updated for FL."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_batch_norm: bool = False,
                 use_group_norm: bool = False,
                 num_groups: int = 32):

        super(ResidualBlock, self).__init__()

        norm_layer = nn.Identity

        if use_batch_norm:
            norm_layer = lambda channels: nn.BatchNorm2d(channels)
        elif use_group_norm:
            # We need a lambda to handle the fallback logic
            def create_group_norm(channels):
                gn_num_groups = num_groups
                if channels % gn_num_groups != 0:
                    # Find the largest valid group size
                    gn_num_groups = next(g for g in range(num_groups, 0, -1) if channels % g == 0)
                return nn.GroupNorm(gn_num_groups, channels)

            norm_layer = create_group_norm
        # -------------------------------------

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channels)  # <-- FIXED
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)  # <-- FIXED

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)  # <-- FIXED
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ConfigurableResNet(nn.Module):
    """A configurable ResNet-18 model, updated for FL."""

    def __init__(self, input_channels: int, num_classes: int, config: ImageModelConfig):
        super(ConfigurableResNet, self).__init__()
        self.in_channels = 64
        self.config = config

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # --- Apply normalization logic to the first BN layer ---
        if self.config.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(64)
        elif getattr(self.config, 'use_group_norm', False):
            num_groups = getattr(self.config, 'num_groups', 32)
            if 64 % num_groups != 0:
                num_groups = next(g for g in range(num_groups, 0, -1) if 64 % g == 0)
            self.bn1 = nn.GroupNorm(num_groups, 64)
        else:
            self.bn1 = nn.Identity()
        # -----------------------------------------------------

        # ResNet-18 specific layer structure
        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)

        # Dropout layer before the final fully connected layer
        self.dropout = nn.Dropout(self.config.dropout_rate) if self.config.use_dropout else nn.Identity()

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block: type, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        # --- Get norm configs to pass to the block ---
        use_bn = self.config.use_batch_norm
        use_gn = getattr(self.config, 'use_group_norm', False)
        num_g = getattr(self.config, 'num_groups', 32)
        # -------------------------------------------

        for s in strides:
            layers.append(block(
                self.in_channels,
                out_channels,
                s,
                use_batch_norm=use_bn,  # <-- PASS configs
                use_group_norm=use_gn,  # <-- PASS configs
                num_groups=num_g  # <-- PASS configs
            ))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


class ImageModelFactory:
    """Factory class for creating and managing configurable models."""

    @staticmethod
    def create_model(model_name: str, num_classes: int, in_channels: int,
                     image_size: Tuple[int, int], config: ImageModelConfig,
                     device: str = 'cpu') -> nn.Module:
        """Create a model based on configuration."""
        logger.info(f"🧠 Creating configurable image model '{model_name}'...")

        model = None

        # --- 1. Instantiate the correct model class ON CPU ---
        if model_name.lower() == 'lenet':
            model = ConfigurableLeNet(in_channels, image_size, num_classes, config)
        elif model_name.lower() == 'flexiblecnn' or model_name.lower() == 'cnn':
            model = ConfigurableFlexibleCNN(in_channels, image_size, num_classes, config)
        elif model_name.lower() == 'resnet18':
            model = ConfigurableResNet(num_classes=num_classes, config=config, input_channels=in_channels)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        logger.info("--- Model created on CPU ---")

        # --- 2. APPLY STABLE INIT ON CPU ---
        logger.info("--- ⚡️ Applying STABLE init BEFORE move to device ---")
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # USE UNIFORM INSTEAD OF NORMAL TO PREVENT Inf IN FP16
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.running_mean.zero_()
                m.running_var.fill_(1)
                if hasattr(m, 'num_batches_tracked'):
                    m.num_batches_tracked.zero_()
            elif isinstance(m, nn.Linear):
                # USE UNIFORM INSTEAD OF NORMAL TO PREVENT Inf IN FP16
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        logger.info("--- STABLE init applied on CPU ---")

        # --- 3. Move to device AS float32 (and KEEP IT IN float32) ---
        model = model.to(device)
        logger.info(f"--- Model moved to {device} (kept as float32 for AMP) ---")

        # --- 4. VERIFY NO NaN/Inf ---
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.error(f"--- ❌ VERIFICATION FAILED FOR {name} ---")
                logger.error(f"    Device: {param.device}, Dtype: {param.dtype}")
                logger.error(f"    Min: {param.min().item()}, Max: {param.max().item()}")
                raise RuntimeError(f"NaN/Inf in parameter '{name}' after initialization!")

        logger.info("--- ✅ Model verification PASSED (float32 for AMP training) ---")

        # --- 5. Log details ---
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  -> Instantiated model class: '{model.__class__.__name__}'")
        logger.info(f"  -> Total parameters: {num_params:,}")
        logger.info(f"  -> Trainable parameters: {num_trainable:,}")

        return model

    @staticmethod
    def create_factory(model_name: str, num_classes: int, in_channels: int,
                       image_size: Tuple[int, int], config: ImageModelConfig, device: str = 'cpu') -> Callable[
        [], nn.Module]:
        """
        Create a zero-argument factory function with frozen parameters.

        This ensures all models created from this factory are identical,
        even if the original config is mutated after factory creation.

        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            in_channels: Number of input channels
            image_size: Image dimensions (height, width)
            config: Model configuration

        Returns:
            A callable that takes no arguments and returns a model instance
        """
        # Validate inputs
        if not isinstance(config, ImageModelConfig):
            raise TypeError(f"config must be ImageModelConfig, got {type(config)}")

        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            raise ValueError(f"image_size must be a tuple/list of 2 integers, got {image_size}")

        # Deep copy all parameters to freeze them at factory creation time
        frozen_model_name = str(model_name)
        frozen_num_classes = int(num_classes)
        frozen_in_channels = int(in_channels)
        frozen_image_size = tuple(int(x) for x in image_size)
        frozen_config = copy.deepcopy(config)

        def model_factory() -> nn.Module:
            """Zero-argument factory that creates a model with frozen config."""
            return ImageModelFactory.create_model(
                model_name=frozen_model_name,
                num_classes=frozen_num_classes,
                in_channels=frozen_in_channels,
                image_size=frozen_image_size,
                config=frozen_config,
                device=device
            )

        # Validate factory creates valid models
        try:
            test_model = model_factory()
            num_params = sum(p.numel() for p in test_model.parameters())
            logger.info(f"Model factory created and validated:")
            logger.info(f"  - Architecture: {frozen_model_name}")
            logger.info(f"  - Parameters: {num_params:,}")
            logger.info(f"  - Input: {frozen_in_channels}x{frozen_image_size[0]}x{frozen_image_size[1]}")
            logger.info(f"  - Output: {frozen_num_classes} classes")
            del test_model  # Clean up
        except Exception as e:
            logger.error(f"Failed to create test model from factory: {e}")
            raise

        return model_factory

    @staticmethod
    def save_model_and_config(model: nn.Module, config: ImageModelConfig, save_dir: str,
                              model_name: str = "model"):
        """Save model state and configuration."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = save_path / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)

        # Save configuration
        config_path = save_path / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)

        logger.info(f"Saved model and config to {save_path}")
        return model_path, config_path

    @staticmethod
    def load_model_and_config(save_dir: str, model_name: str, num_classes: int,
                              in_channels: int, image_size: Tuple[int, int]) -> Tuple[nn.Module, ImageModelConfig]:
        """Load model state and configuration."""
        save_path = Path(save_dir)

        # Load configuration
        config_path = save_path / f"{model_name}_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ImageModelConfig(**config_dict)

        # Create model
        model = ImageModelFactory.create_model(config.model_name, num_classes, in_channels, image_size, config)

        # Load model state
        model_path = save_path / f"{model_name}.pth"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        logger.info(f"Loaded model and config from {save_path}")
        return model, config


def validate_model_factory(factory: Callable[[], nn.Module], num_tests: int = 3) -> bool:
    """
    Validate that a model factory creates consistent models.

    Args:
        factory: The model factory to test
        num_tests: Number of models to create for testing

    Returns:
        True if factory is consistent, raises exception otherwise
    """
    logger.info("Validating model factory consistency...")

    models = []
    param_shapes_list = []

    for i in range(num_tests):
        try:
            model = factory()
            models.append(model)
            param_shapes = [p.shape for p in model.parameters()]
            param_shapes_list.append(param_shapes)
        except Exception as e:
            raise RuntimeError(f"Factory failed to create model {i + 1}: {e}")

    # Compare all models
    reference_shapes = param_shapes_list[0]
    for i, shapes in enumerate(param_shapes_list[1:], start=2):
        if shapes != reference_shapes:
            logger.error(f"Model {i} has different architecture than model 1!")
            logger.error(f"  Model 1: {reference_shapes[:3]}...")
            logger.error(f"  Model {i}: {shapes[:3]}...")
            raise RuntimeError("Model factory creates inconsistent architectures!")

    logger.info(f"Factory validation passed: {num_tests} models with identical architectures")
    logger.info(f"  - Parameters: {len(reference_shapes)}")
    logger.info(f"  - First param shape: {reference_shapes[0]}")

    # Clean up
    del models

    return True
