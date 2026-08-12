import copy
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Callable

import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.common_utils.model_utils import init_weights, _log_param_stats
from src.model.image_model import BaseModelConfig


@dataclass
class TabularModelConfig(BaseModelConfig):
    """Configuration class for tabular model architecture and training parameters."""
    # Model selection
    model_name: str = "TabularMLP"  # "TabularMLP", "TabularResNet"

    # Architecture parameters
    hidden_dims: List[int] = None  # [128, 64] for MLP layers
    use_dropout: bool = True
    dropout_rate: float = 0.5
    use_batch_norm: bool = False
    activation: str = "relu"  # "relu", "tanh", "sigmoid", "leaky_relu"

    # ResNet specific
    num_res_blocks: int = 2
    res_hidden_dim: int = 128

    # MLP specific
    mlp_hidden_dim: int = 128

    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9
    use_scheduler: bool = True
    scheduler_step: int = 30
    scheduler_gamma: float = 0.1
    optimizer_type: str = "adam"  # "sgd", "adam", "adamw"
    use_layer_norm: bool = False
    # Regularization
    use_early_stopping: bool = True
    patience: int = 10

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates a config instance from a dictionary, separating params and metadata."""
        params = data.get('parameters', {})
        meta = data.get('metadata', {})
        return cls(**params, **meta)


class ConfigurableTabularResBlock(nn.Module):
    """A configurable residual block for tabular ResNet."""

    def __init__(self, in_features: int, config: TabularModelConfig):
        super().__init__()
        self.config = config

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activation_map[config.activation]

        # Build block layers
        layers = [nn.Linear(in_features, in_features)]

        if config.use_batch_norm:
            layers.append(nn.BatchNorm1d(in_features))

        layers.append(self.activation)

        if config.use_dropout:
            layers.append(nn.Dropout(p=config.dropout_rate))

        layers.append(nn.Linear(in_features, in_features))

        if config.use_batch_norm:
            layers.append(nn.BatchNorm1d(in_features))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.layers(x)
        out += identity  # Skip connection
        out = self.activation(out)  # Final activation
        return out


class ConfigurableTabularResNet(nn.Module):
    """A configurable ResNet-style model for tabular data."""

    def __init__(self, input_dim: int, num_classes: int, config: TabularModelConfig):
        super().__init__()
        self.config = config

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activation_map[config.activation]

        # Initial projection layer
        initial_layers = [nn.Linear(input_dim, config.res_hidden_dim)]

        if config.use_batch_norm:
            initial_layers.append(nn.BatchNorm1d(config.res_hidden_dim))

        initial_layers.append(self.activation)

        if config.use_dropout:
            initial_layers.append(nn.Dropout(config.dropout_rate))

        self.initial_layer = nn.Sequential(*initial_layers)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ConfigurableTabularResBlock(config.res_hidden_dim, config)
              for _ in range(config.num_res_blocks)]
        )

        # Output layer
        output_layers = []
        if config.use_dropout:
            output_layers.append(nn.Dropout(config.dropout_rate))

        output_layers.append(nn.Linear(config.res_hidden_dim, num_classes))
        self.output_layer = nn.Sequential(*output_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x


class ConfigurableTabularMLP(nn.Module):
    """A configurable MLP for tabular data."""

    def __init__(self, input_dim: int, num_classes: int, config: TabularModelConfig):
        super().__init__()
        self.config = config

        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activation_map[config.activation]

        # Build MLP layers dynamically
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if config.use_batch_norm:
                norm_layer = nn.BatchNorm1d(hidden_dim)
            elif getattr(config, 'use_layer_norm', False):
                norm_layer = nn.LayerNorm(hidden_dim)
            else:
                norm_layer = nn.Identity()

            layers.append(norm_layer)
            # --- End of updated logic ---

            layers.append(self.activation)

            if config.use_dropout:
                layers.append(nn.Dropout(config.dropout_rate))

            prev_dim = hidden_dim

        # Final output layer
        if config.use_dropout:
            layers.append(nn.Dropout(config.dropout_rate))

        layers.append(nn.Linear(prev_dim, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TabularModelFactory:
    """Factory class for creating configurable tabular models."""

    @staticmethod
    def create_model(model_name: str, input_dim: int, num_classes: int,
                     config: TabularModelConfig,
                     device: str = 'cpu') -> nn.Module:
        """Create a tabular model based on configuration."""
        logging.info(f"🧠 Creating configurable tabular model '{model_name}'...")

        model = None
        if model_name.lower() == 'tabularmlp':
            model = ConfigurableTabularMLP(input_dim, num_classes, config)
        elif model_name.lower() == 'tabularresnet':
            model = ConfigurableTabularResNet(input_dim, num_classes, config)
        else:
            raise ValueError(f"Unknown tabular model name: {model_name}")

        logging.info("--- Model created on CPU ---")
        _log_param_stats(model, "layers.0.weight", "Initial CPU (float32)")

        # 1. APPLY STABLE INIT *FIRST* (on the CPU)
        logging.info("--- ⚡️ Applying STABLE init BEFORE move to device ---")
        model.apply(init_weights)

        logging.info("--- STABLE init applied on CPU ---")
        _log_param_stats(model, "layers.0.weight", "After init_weights (float32)")

        # 2. Move to device AS float32
        model = model.to(device)
        logging.info(f"--- Model moved to {device} (as float32) ---")
        _log_param_stats(model, "layers.0.weight", f"After .to({device}) (float32)")

        # 3. VERIFY
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logging.error(f"--- ❌ VERIFICATION FAILED FOR {name} ---")
                _log_param_stats(model, name, "FAILURE")
                raise RuntimeError(f"NaN/Inf in '{name}' after universal init!")

        logging.info("--- ✅ Model verification PASSED ---")

        # 4. Log model details
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"  -> Instantiated model class: '{model.__class__.__name__}'")
        logging.info(f"  -> Total parameters: {num_params:,}")
        logging.info(f"  -> Trainable parameters: {num_trainable:,}")

        return model

    @staticmethod
    def create_factory(model_name: str, input_dim: int, num_classes: int,
                       config: TabularModelConfig, device: str = 'cpu') -> Callable[[], nn.Module]:
        """
        Create a zero-argument factory function with frozen parameters.

        This ensures all models created from this factory are identical,
        even if the original config is mutated after factory creation.

        Args:
            model_name: Name of the model architecture
            input_dim: Input feature dimension
            num_classes: Number of output classes
            config: Model configuration
            device: Device to place models on

        Returns:
            A callable that takes no arguments and returns a model instance
        """
        # Validate inputs
        if not isinstance(config, TabularModelConfig):
            raise TypeError(f"config must be TabularModelConfig, got {type(config)}")

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        # Deep copy all parameters to freeze them at factory creation time
        frozen_model_name = str(model_name)
        frozen_input_dim = int(input_dim)
        frozen_num_classes = int(num_classes)
        frozen_config = copy.deepcopy(config)
        frozen_device = str(device)

        def model_factory() -> nn.Module:
            """Zero-argument factory that creates a model with frozen config."""
            return TabularModelFactory.create_model(
                model_name=frozen_model_name,
                input_dim=frozen_input_dim,
                num_classes=frozen_num_classes,
                config=frozen_config,
                device=frozen_device
            )

        # Validate factory creates valid models
        try:
            test_model = model_factory()
            num_params = sum(p.numel() for p in test_model.parameters())
            logging.info(f"Tabular model factory created and validated:")
            logging.info(f"  - Architecture: {frozen_model_name}")
            logging.info(f"  - Parameters: {num_params:,}")
            logging.info(f"  - Input dimension: {frozen_input_dim}")
            logging.info(f"  - Output classes: {frozen_num_classes}")
            del test_model  # Clean up
        except Exception as e:
            logging.error(f"Failed to create test model from factory: {e}")
            raise

        return model_factory


class TabularConfigManager:
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._configs: Dict[str, Dict] = {}
        self._load_all_configs()

    def _load_all_configs(self):
        for path in self.config_dir.glob("*.yaml"):
            with open(path, 'r') as f:
                self._configs[path.stem] = yaml.safe_load(f)

    def get_config_by_name(self, name: str) -> TabularModelConfig:
        if name not in self._configs: raise ValueError(f"Config '{name}' not found.")
        data = self._configs[name]
        return TabularModelConfig(**data.get('parameters', {}), **data.get('metadata', {}))

    def save_config(self, config_obj: TabularModelConfig):
        filepath = self.config_dir / f"{config_obj.config_name}.yaml"
        all_params = asdict(config_obj)
        meta_keys = ['config_name', 'description', 'source', 'creation_date', 'target_dataset', 'performance_metrics']
        metadata = {k: all_params.pop(k) for k in meta_keys}
        data = {'metadata': metadata, 'parameters': all_params}
        with open(filepath, 'w') as f: yaml.dump(data, f, indent=2, sort_keys=False)
        self._configs[config_obj.config_name] = data

    def summarize_configs(self) -> pd.DataFrame:
        """Creates a summary DataFrame of all available configurations."""
        if not self._configs:
            return pd.DataFrame()

        records = []
        for name, data in self._configs.items():
            meta = data.get('metadata', {})
            params = data.get('parameters', {})

            # Extract key info for the summary table
            record = {
                'config_name': name,
                'source': meta.get('source'),
                'target_dataset': meta.get('target_dataset'),
                'model': params.get('model_name'),
                'epochs': params.get('epochs'),
                'lr': params.get('learning_rate'),
                'wd': params.get('weight_decay'),
                'dropout': params.get('dropout_rate'),
                'overfitting_gap': meta.get('performance_metrics', {}).get('gap'),
                'description': meta.get('description')
            }
            records.append(record)

        df = pd.DataFrame(records)
        return df.set_index('config_name')
