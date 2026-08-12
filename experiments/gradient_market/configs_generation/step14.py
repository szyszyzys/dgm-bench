# FILE: generate_step14_collusion_attack.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    use_sybil_attack_strategy,
    get_tuned_defense_params,
    GOLDEN_TRAINING_PARAMS
)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import Scenario, use_cifar100_config

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

## Purpose
# This script generates configs for Step 14: Collusion Attack (Baseline Hijacking).
# It tests defenses when the Buyer and Sybil Sellers coordinate to provide
# identical (but malicious) gradients to force high trust scores or cluster density.

# --- Collusion Strategies & Parameters ---
# We sweep the 'mode' of the collusion vector.
COLLUSION_TEST_CONFIG = {
    "collusion": {
        # 'random': Both generate random noise (Requires shared seed or oracle in backend)
        # 'inverse': Both flip the benign gradient sign
        "mode": ["random", "inverse"]
    }
}

# --- Setup ---
COLLUSION_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
}

# --- Fixed Attack Parameters ---
FIXED_ADV_RATE = 0.3  # 30% Sybils


# === Scenario Generation Function ===
def generate_collusion_attack_scenarios() -> List[Scenario]:
    """Generates scenarios for the Collusion Attack."""
    print("\n--- Generating Step 14: Collusion Attack Scenarios ---")
    scenarios = []
    modality = COLLUSION_SETUP["modality_name"]
    model_cfg_name = COLLUSION_SETUP["model_config_name"]

    # Testing FLTrust (Vulnerable) and MartFL (Target of Baseline Hijacking)
    current_defenses = ["fltrust", "martfl", "fedavg", "skymask", "skymask_small"]

    for defense_name in current_defenses:
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state='with_attack',
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name}")

        def create_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # 1. Apply Golden HPs
                golden_hp_key = f"{current_model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)

                # 2. Apply Defense HPs
                if current_tuned_params:
                    for key, value in current_tuned_params.items():
                        set_nested_attr(config, key, value)

                if "skymask" in current_defense_name:
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                # 3. Standard Setup
                set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)

                # 4. (CRITICAL) COLLUSION CONFIGURATION

                # A. Disable standard poisoning (we operate on gradients)
                config.adversary_seller_config.poisoning.type = PoisonType.NONE

                # B. ENABLE BUYER ATTACK (The Collusion Partner)
                # For the collusion to work against FLTrust, the Buyer MUST be malicious
                config.buyer_attack_config.is_active = True

                # We set the buyer to match the seller's intended mode (e.g., random noise)
                # The SybilCoordinator (backend) is responsible for synchronizing the exact vector.
                config.buyer_attack_config.attack_type = "collusion"

                # C. Disable valuation metrics to save time
                config.valuation.run_influence = False
                config.valuation.run_loo = False
                config.valuation.run_kernelshap = False

                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # Base Grid
        base_grid = {
            COLLUSION_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [COLLUSION_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "experiment.use_early_stopping": [False],
            "experiment.global_rounds": [100],
        }

        # Loop through Collusion strategy settings
        for strategy_name, strategy_params_sweep in COLLUSION_TEST_CONFIG.items():
            print(f"  - Strategy: {strategy_name}")
            scenario_name = f"step14_collusion_{strategy_name}_{defense_name}"

            current_grid = base_grid.copy()
            current_grid["_strategy_name"] = [strategy_name]
            current_grid["_sweep_params"] = [strategy_params_sweep]

            current_modifiers = [
                setup_modifier_func,
                COLLUSION_SETUP["dataset_modifier"],
            ]

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=COLLUSION_SETUP["base_config_factory"],
                modifiers=current_modifiers,
                parameter_grid=current_grid
            )
            scenarios.append(scenario)

    return scenarios


if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step14_collusion_attack"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_collusion_attack_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 14 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        static_grid = {k: v for k, v in scenario.parameter_grid.items() if k.startswith("_") == False}
        strategy_name = scenario.parameter_grid["_strategy_name"][0]
        sweep_params = scenario.parameter_grid["_sweep_params"][0]
        base_hp_suffix = f"adv_{FIXED_ADV_RATE}"

        if sweep_params:
            sweep_key, sweep_values = next(iter(sweep_params.items()))  # e.g., "mode"

            # Path to the strategy-specific config in SybilConfig
            config_key_path = f"adversary_seller_config.sybil.strategy_configs.{strategy_name}"

            for sweep_value in sweep_values:
                current_grid = static_grid.copy()

                # Configure the Collusion Strategy object
                collusion_config_dict = {
                    sweep_key: sweep_value,  # e.g., "mode": "random"
                    "noise_scale": 1e-5      # Fixed small noise to avoid duplicates
                }

                # Inject config dictionary
                current_grid[config_key_path] = [collusion_config_dict]

                hp_suffix = f"{base_hp_suffix}_{sweep_key}_{sweep_value}"

                temp_scenario = Scenario(
                    name=f"{scenario.name}/{hp_suffix}",
                    base_config_factory=scenario.base_config_factory,
                    # Enables the Sybil Coordinator for 'collusion'
                    modifiers=scenario.modifiers + [use_sybil_attack_strategy(strategy=strategy_name)],
                    parameter_grid=current_grid
                )
                temp_scenario.parameter_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

                base_config = temp_scenario.base_config_factory()
                modified_base_config = copy.deepcopy(base_config)
                for modifier in temp_scenario.modifiers:
                    modified_base_config = modifier(modified_base_config)

                num_gen = generator.generate(modified_base_config, temp_scenario)
                task_configs += num_gen
        else:
            print(f"  SKIPPING: {strategy_name} has no sweep parameters.")

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 14 (Collusion Attack) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")