# FILE: generate_step17_alie_attack.py
# Purpose: Tests the ALIE (A Little Is Enough) attack against all defenses.
#          ALIE crafts gradients at the variance detection boundary, targeting
#          coordinate-wise (TrimmedMean) and distance-based (Multi-Krum) defenses.

import copy
import sys
from pathlib import Path
from typing import List

from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,
    NUM_SEEDS_PER_CONFIG,
    get_tuned_defense_params,
    IMAGE_DEFENSES,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
    use_sybil_attack_strategy,
)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import Scenario, use_image_backdoor_attack, use_cifar100_config

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# z-score controls how close to the mean the ALIE attack stays.
# Lower z = stealthier (closer to mean) but weaker attack.
# Higher z = stronger attack but easier to detect.
Z_SCORES = [1.0, 2.0]  # was [0.5, 1.0, 1.5, 2.0]; covers low and high boundary

ALIE_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
    "attack_modifier": use_image_backdoor_attack,
    "defenses": IMAGE_DEFENSES,
}


def generate_alie_scenarios() -> List[Scenario]:
    print("\n--- Generating Step 17: ALIE Attack Scenarios ---")
    scenarios = []
    modality = ALIE_SETUP["modality_name"]
    model_cfg_name = ALIE_SETUP["model_config_name"]

    for defense_name in ALIE_SETUP["defenses"]:
        tuned_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor"
        )
        if not tuned_params and defense_name != "fedavg":
            print(f"  SKIPPING {defense_name}: no tuned params")
            continue

        def create_modifier(
            d_name=defense_name, m_name=model_cfg_name,
            t_params=tuned_params, atk_mod=ALIE_SETUP["attack_modifier"],
            mod=modality
        ):
            def modifier(config: AppConfig) -> AppConfig:
                golden = GOLDEN_TRAINING_PARAMS.get(m_name)
                if golden:
                    for k, v in golden.items():
                        set_nested_attr(config, k, v)
                if t_params:
                    for k, v in t_params.items():
                        set_nested_attr(config, k, v)
                if "skymask" in d_name:
                    model_struct = "resnet18" if "resnet" in m_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)
                config = atk_mod(config)
                set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", DEFAULT_POISON_RATE)
                set_nested_attr(config, f"data.{mod}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{mod}.dirichlet_alpha", 0.5)
                return config
            return modifier

        # Enable ALIE as the Sybil strategy
        alie_sybil_modifier = use_sybil_attack_strategy("alie")

        parameter_grid = {
            ALIE_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [ALIE_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            "experiment.adv_rate": [DEFAULT_ADV_RATE],
        }

        scenario = Scenario(
            name=f"step17_alie_{defense_name}_{ALIE_SETUP['dataset_name']}",
            base_config_factory=ALIE_SETUP["base_config_factory"],
            modifiers=[create_modifier(), ALIE_SETUP["dataset_modifier"], alie_sybil_modifier],
            parameter_grid=parameter_grid,
        )
        scenarios.append(scenario)

    return scenarios


if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step17_alie_attack"
    generator = ExperimentGenerator(str(output_dir))

    scenarios = generate_alie_scenarios()
    total = 0

    for scenario in scenarios:
        static_grid = scenario.parameter_grid.copy()
        task_count = 0

        for z_score in Z_SCORES:
            current_grid = static_grid.copy()
            # Pass z_score via sybil strategy_configs
            current_grid["adversary_seller_config.sybil.strategy_configs"] = [
                {"alie": {"z_score": z_score}}
            ]

            hp_suffix = f"z_{z_score}"
            current_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

            temp_scenario = Scenario(
                name=f"{scenario.name}/{hp_suffix}",
                base_config_factory=scenario.base_config_factory,
                modifiers=[],
                parameter_grid=current_grid,
            )

            base_config = temp_scenario.base_config_factory()
            for mod in scenario.modifiers:
                base_config = mod(base_config)

            num_gen = generator.generate(base_config, temp_scenario)
            task_count += num_gen

        print(f"-> {scenario.name}: {task_count} configs")
        total += task_count

    print(f"\n✅ Step 17 (ALIE Attack) complete: {total} configs")
    print(f"Configs saved to: {output_dir}")
