import copy
import itertools
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from types import NoneType
from typing import Any, List

import numpy as np
import yaml

from src.common_utils.constants import PoisonType, VictimStrategy, ImageBackdoorAttackName, ImageTriggerType, \
    ImageTriggerLocation, \
    TextTriggerLocation, TextBackdoorAttackName, LabelFlipMode
from experiments.gradient_market.automate_exp.scenarios import Scenario
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, TabularBackdoorAttackName


class CustomDumper(yaml.SafeDumper):
    # (This class is unchanged, same as your file)

    def represent_none(self, _):
        """Represents None as an empty string (or choose 'null')."""
        return self.represent_scalar('tag:yaml.org,2002:null', '')  # or 'null'

    def represent_numpy(self, data):
        """Represents various numpy types as standard Python types."""
        if isinstance(data, np.integer): return self.represent_int(int(data))
        if isinstance(data, np.floating): return self.represent_float(float(data))
        if isinstance(data, np.ndarray): return self.represent_list(data.tolist())
        if isinstance(data, np.bool_): return self.represent_bool(bool(data))
        return super().represent_data(data)

    def represent_enum(self, data):
        """Tells YAML how to represent any Enum: by using its NAME as a string."""
        return self.represent_scalar('tag:yaml.org,2002:str', data.value)


CustomDumper.add_representer(NoneType, CustomDumper.represent_none)
CustomDumper.add_representer(Enum, CustomDumper.represent_enum)
CustomDumper.add_representer(PoisonType, CustomDumper.represent_enum)
CustomDumper.add_representer(VictimStrategy, CustomDumper.represent_enum)
CustomDumper.add_representer(ImageBackdoorAttackName, CustomDumper.represent_enum)
CustomDumper.add_representer(ImageTriggerType, CustomDumper.represent_enum)
CustomDumper.add_representer(TextTriggerLocation, CustomDumper.represent_enum)
CustomDumper.add_representer(ImageTriggerLocation, CustomDumper.represent_enum)
CustomDumper.add_representer(TextBackdoorAttackName, CustomDumper.represent_enum)
CustomDumper.add_representer(TabularBackdoorAttackName, CustomDumper.represent_enum)
CustomDumper.add_representer(LabelFlipMode, CustomDumper.represent_enum)
for numpy_type in (np.integer, np.floating, np.ndarray, np.bool_):
    CustomDumper.add_representer(numpy_type, CustomDumper.represent_numpy)


def set_nested_attr(obj: Any, key: str, value: Any):
    """
    Sets a nested attribute on an object or a key in a nested dict
    using a dot-separated key.
    """
    keys = key.split('.')
    current_obj = obj
    for k in keys[:-1]:
        current_obj = getattr(current_obj, k)
    final_key = keys[-1]
    if isinstance(current_obj, dict):
        current_obj[final_key] = value
    else:
        setattr(current_obj, final_key, value)


def iter_grid(parameter_grid: dict) -> List[dict]:
    """
    Expands a grid (dict of lists) into a list of single-run dicts.
    e.g., {'a': [1, 2], 'b': [3]} -> [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
    """
    keys, values = zip(*parameter_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return param_combinations


class ExperimentGenerator:
    """Generates configuration files by modifying a base AppConfig object."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, base_config: AppConfig, scenario: Scenario) -> int:
        # (This function is unchanged, same as your file)
        """
        Generates all config files for a given experimental scenario.

        Returns:
            int: The number of configuration files generated for this scenario.
        """
        print(f"\n--- Generating '{scenario.name}' Configs ---")

        # 1. Apply all base modifiers
        scenario_config = copy.deepcopy(base_config)
        for modifier_func in scenario.modifiers:
            scenario_config = modifier_func(scenario_config)

        # 2. Create all combinations from the parameter grid
        # NOTE: This uses the same logic as the new iter_grid function
        keys, values = zip(*scenario.parameter_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        num_generated_total = len(param_combinations)
        print(f"  Found {num_generated_total} parameter combinations for this scenario.")

        count_saved = 0
        for i, params in enumerate(param_combinations):
            final_config = copy.deepcopy(scenario_config)

            # Apply the specific parameters for this run
            for key, value in params.items():
                try:
                    set_nested_attr(final_config, key, value)
                except (AttributeError, KeyError) as e:
                    print(f"  ❌ Error setting {key}={value} for combo {i}: {e}. Skipping.")
                    continue

            # 3. Create the descriptive run name
            try:
                run_name = self.create_run_name(final_config)
            except Exception as e:
                print(f"  ❌ Error creating run name for combo {i}: {e}. Using index instead.")
                run_name = f"combo_{i}"

            # 4. Define paths
            # This is where the EXPERIMENT RESULTS will be saved
            # We set this path *inside* the config file
            results_save_path = Path("./results") / scenario.name / run_name
            final_config.experiment.save_path = str(results_save_path)

            # This is where the CONFIG FILE will be saved
            config_file_path = self.output_dir / scenario.name / run_name / "config.yaml"

            # 5. Save the config file
            try:
                config_file_path.parent.mkdir(parents=True, exist_ok=True)
                config_dict = asdict(final_config)
                with open(config_file_path, 'w') as f:
                    yaml.dump(config_dict, f, Dumper=CustomDumper, sort_keys=False, indent=2)
                count_saved += 1
            except Exception as e:
                print(f"  ❌ Error saving config file {config_file_path}: {e}")

        print(f"  Successfully saved {count_saved} / {num_generated_total} config files.")
        return count_saved

    @staticmethod
    def create_run_name(config: AppConfig) -> str:
        """
        Generates a descriptive and unique name for a single experiment run
        from its final configuration. Captures all swept parameters.
        """
        parts = []

        # 1. Dataset
        parts.append(f"ds-{config.experiment.dataset_name.lower()}")

        # 2. Model Architecture
        parts.append(f"model-{config.experiment.model_structure.lower()}")

        # 3. Aggregation Method
        agg_method = config.aggregation.method.lower()
        parts.append(f"agg-{agg_method}")

        # --- START OF MODIFICATION: Add Defense HP to name ---
        # 4. Tuned Defense HPs
        agg_cfg = config.aggregation

        # Helper to format values, turning None -> "None" and 1.0 -> "1p0"
        def format_hp(val):
            if val is None:
                return "None"
            if isinstance(val, float):
                return f"{val:g}".replace('.', 'p')
            return str(val)

        if agg_method == "fltrust":
            parts.append(f"clip-{format_hp(agg_cfg.clip_norm)}")

        elif agg_method == "fltrust_threshold":
            parts.append(f"tau-{format_hp(agg_cfg.fltrust_threshold.tau)}")
            parts.append(f"clip-{format_hp(agg_cfg.clip_norm)}")

        elif agg_method == "martfl":
            parts.append(f"k-{format_hp(agg_cfg.martfl.max_k)}")
            parts.append(f"clip-{format_hp(agg_cfg.clip_norm)}")

        elif agg_method == "skymask":
            parts.append(f"sk_ep-{format_hp(agg_cfg.skymask.mask_epochs)}")
            parts.append(f"sk_lr-{format_hp(agg_cfg.skymask.mask_lr)}")
            parts.append(f"sk_thr-{format_hp(agg_cfg.skymask.mask_threshold)}")
            parts.append(f"clip-{format_hp(agg_cfg.clip_norm)}")

        elif agg_method == "trimmed_mean":
            parts.append(f"trim-{format_hp(agg_cfg.trimmed_mean.trim_ratio)}")

        elif agg_method == "multi_krum":
            parts.append(f"f-{format_hp(agg_cfg.multi_krum.num_byzantine)}")
            parts.append(f"m-{format_hp(agg_cfg.multi_krum.m_selected)}")

        elif agg_method == "rflpa":
            parts.append(f"eclip-{format_hp(agg_cfg.rflpa.element_clip)}")
            parts.append(f"clip-{format_hp(agg_cfg.clip_norm)}")

        elif agg_method == "spmc":
            parts.append(f"temp-{format_hp(agg_cfg.spmc.temperature)}")

        elif agg_method == "daved":
            parts.append(f"alpha-{format_hp(agg_cfg.daved.l2_relative_alpha)}")
            parts.append(f"lam-{format_hp(agg_cfg.daved.proj_lambda)}")

        # 5. Seller Attack Information (Was part 5)
        seller_attack_parts = []
        poison_cfg = config.adversary_seller_config.poisoning
        sybil_cfg = config.adversary_seller_config.sybil

        poison_type = poison_cfg.type.name.lower()
        if poison_type != 'none':
            adv_rate = f"{config.experiment.adv_rate:g}".replace('.', 'p')
            poison_rate = f"{poison_cfg.poison_rate:g}".replace('.', 'p')
            seller_attack_parts.append(f"{poison_type}")
            seller_attack_parts.append(f"adv-{adv_rate}")
            seller_attack_parts.append(f"poison-{poison_rate}")
            # (Rest of seller attack logic is fine)
            if sybil_cfg.is_sybil:
                seller_attack_parts.append(f"sybil-{sybil_cfg.gradient_default_mode or 'unknown'}")
            if config.adversary_seller_config.adaptive_attack.is_active:
                seller_attack_parts.append(f"adaptive-{config.adversary_seller_config.adaptive_attack.attack_mode}")

        if seller_attack_parts:
            parts.append("_".join(seller_attack_parts))
        else:
            parts.append("no-seller-attack")

        # 6. Buyer Attack (Was part 6, unchanged)
        if config.buyer_attack_config.is_active:
            pass  # Placeholder

        # 7. Data Distribution Parameters (Was part 7, unchanged)
        modality_data_config = None
        if config.data.image:
            modality_data_config = config.data.image
        elif config.data.text:
            modality_data_config = config.data.text
        elif config.data.tabular:
            modality_data_config = config.data.tabular

        if modality_data_config:
            if modality_data_config.strategy == "dirichlet":
                default_alpha = 0.5  # Your default
                alpha = modality_data_config.dirichlet_alpha
                if alpha != default_alpha:
                    parts.append(f"alpha-{format_hp(alpha)}")

            default_buyer_ratio = 0.1  # Your default
            buyer_pct = modality_data_config.buyer_ratio
            if buyer_pct != default_buyer_ratio:
                parts.append(f"buyerdata-{format_hp(buyer_pct)}")

        # 8. Marketplace Size (Was part 8, unchanged)
        n_sellers = config.experiment.n_sellers
        default_n_sellers = 10  # Your default
        if n_sellers != default_n_sellers:
            parts.append(f"sellers-{n_sellers}")

        # 8b. Honest-seller local DP (E3) — only stamped when enabled, so
        # pre-existing run names are unaffected.
        dp_cfg = getattr(config, 'dp', None)
        if dp_cfg is not None and dp_cfg.enabled:
            parts.append(f"dp-eps-{format_hp(dp_cfg.epsilon)}")

        # 9. Random Seed (Was part 9, unchanged)
        parts.append(f"seed-{config.seed}")

        # Join and sanitize
        run_name = "_".join(parts)
        run_name = run_name.replace('[', '').replace(']', '').replace(',', '-')

        return run_name
