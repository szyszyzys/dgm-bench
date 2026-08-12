# FILE: generate_step13_drowning_attack.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
# Import the same common utils as your Step 6
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    use_sybil_attack_strategy,  # <-- The key helper from Step 6
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
# This script generates configs for the Step 13: Targeted Drowning Attack.
# It tests the vulnerability of clustering-based defenses (MartFL)
# to a coordinated Sybil attack designed to exclude a specific

# --- Drowning Attack Strategies & Parameters to Test ---
# We follow the Step 6 pattern.
# We will "sweep" the attack_strength.
# The victim_id will be set by the 'use_sybil_attack_strategy' helper.
TARGET_VICTIM_ID = "bn_5"
DROWNING_TEST_CONFIG = {
    # Reduced sweep: low and high values cover the trade-off
    "drowning": {
        "attack_strength": [1.0, 2.0]  # was [0.5, 1.0, 1.5, 2.0]
    }
}

# --- Focus Setup for Drowning Attack ---
DROWNING_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
}

# --- Fixed Attack Parameters (Number of Sybils) ---
FIXED_ADV_RATE = 0.3  # 30% of sellers will be Sybils


# === Scenario Generation Function (like Step 6) ===
def generate_drowning_attack_scenarios() -> List[Scenario]:
    """Generates base scenarios for the Targeted Drowning Attack."""
    print("\n--- Generating Step 13: Targeted Drowning Attack Scenarios ---")
    scenarios = []
    modality = DROWNING_SETUP["modality_name"]
    model_cfg_name = DROWNING_SETUP["model_config_name"]
    current_defenses = ["fltrust", "martfl"]  # Compare trust-based vs. clustering-based

    for defense_name in current_defenses:
        # 2. Get Tuned HPs (from Step 3)
        # We use 'backdoor' tuning params as a neutral default
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state='with_attack',
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name}")

        # 3. Create the setup modifier INSIDE the loop (good pattern)
        def create_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params,
                current_modality=modality
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # --- Apply Golden Training HPs (from Step 2.5) ---
                golden_hp_key = f"{current_model_cfg_name}"  # Use simple key
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                # --- Apply Tuned Defense HPs (from Step 3) ---
                if current_tuned_params:
                    for key, value in current_tuned_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Tuned HPs found for {current_defense_name}!")

                # (Add SkyMask-specific logic from your Step 6)
                if "skymask" in current_defense_name:
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                # --- Apply other fixed settings ---
                set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)

                # --- (CRITICAL) Disable all other attacks ---
                config.adversary_seller_config.poisoning.type = PoisonType.NONE
                config.buyer_attack_config.is_active = False
                config.valuation.run_influence = False
                config.valuation.run_loo = False
                config.valuation.run_kernelshap = False

                return config

            return modifier

        # 4. Create the modifier function
        setup_modifier_func = create_setup_modifier()

        # Base Grid (fixed parts)
        base_grid = {
            DROWNING_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [DROWNING_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.adv_rate": [FIXED_ADV_RATE],
            # This is a pure sabotage attack, model may not converge
            "experiment.use_early_stopping": [False],
            "experiment.global_rounds": [100],
        }

        # Loop through Drowning strategies (just one in this case)
        for strategy_name, strategy_params_sweep in DROWNING_TEST_CONFIG.items():
            print(f"  - Strategy: {strategy_name}")
            scenario_name = f"step13_drowning_{strategy_name}_{defense_name}"

            current_grid = base_grid.copy()
            # Store "meta" info just like in Step 6
            current_grid["_strategy_name"] = [strategy_name]
            current_grid["_sweep_params"] = [strategy_params_sweep]

            # 5. Build the modifier list
            current_modifiers = [
                setup_modifier_func,
                DROWNING_SETUP["dataset_modifier"],
                # NO attack_modifier (like use_image_backdoor)
            ]

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=DROWNING_SETUP["base_config_factory"],
                modifiers=current_modifiers,
                parameter_grid=current_grid
            )
            scenarios.append(scenario)

    return scenarios


# In generate_step13_drowning_attack.py

# In generate_step13_drowning_attack.py

if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step13_drowning_attack"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_drowning_attack_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 13 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        static_grid = {k: v for k, v in scenario.parameter_grid.items() if k.startswith("_") == False}
        strategy_name = scenario.parameter_grid["_strategy_name"][0]
        sweep_params = scenario.parameter_grid["_sweep_params"][0]
        base_hp_suffix = f"adv_{FIXED_ADV_RATE}"

        if sweep_params:
            sweep_key, sweep_values = next(iter(sweep_params.items()))  # e.g., "attack_strength"

            config_key_path = "adversary_seller_config.sybil.strategy_configs.drowning"

            for sweep_value in sweep_values:
                current_grid = static_grid.copy()

                drowning_config_dict = {
                    "victim_id": TARGET_VICTIM_ID,
                    sweep_key: sweep_value  # e.g., "attack_strength": 1.0
                }

                # This sets adversary_seller_config.sybil.strategy_configs["drowning"]
                # to the dictionary we just created.
                current_grid[config_key_path] = [drowning_config_dict]

                hp_suffix = f"{base_hp_suffix}_{sweep_key}_{sweep_value}"

                temp_scenario = Scenario(
                    name=f"{scenario.name}/{hp_suffix}",
                    base_config_factory=scenario.base_config_factory,
                    modifiers=[],
                    parameter_grid=current_grid
                )
                temp_scenario.parameter_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

                base_config = temp_scenario.base_config_factory()
                modified_base_config = copy.deepcopy(base_config)
                for modifier in scenario.modifiers + [use_sybil_attack_strategy(strategy=strategy_name)]:
                    modified_base_config = modifier(modified_base_config)

                num_gen = generator.generate(modified_base_config, temp_scenario)
                task_configs += num_gen
        else:
            print(f"  SKIPPING: {strategy_name} has no sweep parameters defined.")

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 13 (Targeted Drowning Attack) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"2. Analyze results: Plot the 'selection_rate' of seller '{TARGET_VICTIM_ID}'")
    print(f"   over time (rounds) for MartFL vs. FLTrust.")
    print(f"3. Expectation: Rate for MartFL -> 0, Rate for FLTrust -> stays high.")
