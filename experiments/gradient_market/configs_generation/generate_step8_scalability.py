# FILE: generate_step10_scalability.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,
    NUM_SEEDS_PER_CONFIG,
    get_tuned_defense_params, IMAGE_DEFENSES, FOCUSED_DEFENSES
)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import Scenario, use_image_backdoor_attack, use_cifar100_config

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

# --- Constants ---
MARKETPLACE_SIZES = [10, 50, 100, 500]  # N=500 added back for SIGMOD scalability stress test
FIXED_ADV_RATE = 0.3
FIXED_ATTACK_POISON_RATE = 0.5
SCALABILITY_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
    "attack_modifier": use_image_backdoor_attack
}


# === FUNCTION TO GENERATE BASE SCENARIOS ===
def generate_scalability_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses by varying n_sellers."""
    print("\n--- Generating Step 10: Scalability Scenarios (Fixed Rate) ---")
    scenarios = []
    modality = SCALABILITY_SETUP["modality_name"]
    model_cfg_name = SCALABILITY_SETUP["model_config_name"]
    print(f"Setup: {SCALABILITY_SETUP['dataset_name']} {model_cfg_name}, Fixed Adv Rate: {FIXED_ADV_RATE * 100}%")

    # FOCUSED: limited to FLTrust + MartFL for the in-depth scalability sweep.
    # Full defense coverage lives in Step 10 (main summary).
    for defense_name in FOCUSED_DEFENSES:
        # Get Tuned HPs (from Step 3)
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name}")
        if not tuned_defense_params and defense_name != "fedavg":
            # Allow fedavg to proceed even if it has no tuned params
            print(f"  SKIPPING: No Tuned HPs found for {defense_name}")
            continue

        # Create the setup modifier INSIDE the loop
        def create_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params,
                current_attack_modifier=SCALABILITY_SETUP["attack_modifier"],
                current_modality=modality
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # 1. Apply Golden Training HPs (from Step 2.5)
                golden_hp_key = f"{current_model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                # 2. Apply Tuned Defense HPs (from Step 3)
                if current_tuned_params:
                    for key, value in current_tuned_params.items():
                        set_nested_attr(config, key, value)

                if "skymask" in current_defense_name:
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                # 3. Apply the fixed attack type and strength
                config = current_attack_modifier(config)
                set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)

                # 4. Apply other fixed settings
                set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)

                # Turn off valuation for scalability analysis
                config.valuation.run_influence = False
                config.valuation.run_loo = False
                config.valuation.run_kernelshap = False
                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # --- Define the parameter grid (FIXED params, n_sellers is omitted here) ---
        parameter_grid = {
            SCALABILITY_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [SCALABILITY_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            "experiment.adv_rate": [FIXED_ADV_RATE],
        }

        scenario_name = f"step10_scalability_{defense_name}_{SCALABILITY_SETUP['dataset_name']}"

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=SCALABILITY_SETUP["base_config_factory"],
            modifiers=[setup_modifier_func, SCALABILITY_SETUP["dataset_modifier"]],
            parameter_grid=parameter_grid
        )
        scenarios.append(scenario)

    return scenarios


# ----------------------------------------------------------------------
## MAIN EXECUTION BLOCK (Ensures Separate PDF Paths)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step10_scalability"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_scalability_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 10 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Get the static grid
        static_grid = scenario.parameter_grid.copy()

        # Loop through each marketplace size
        for n_sellers in MARKETPLACE_SIZES:
            print(f"  - Generating configs for n_sellers: {n_sellers}")

            # 1. Create the specific grid for this combination
            current_grid = static_grid.copy()
            # **CRITICAL**: Set the number of sellers in the grid
            current_grid["experiment.n_sellers"] = [n_sellers]

            # 2. Define unique output path
            hp_suffix = f"n_sellers_{n_sellers}"
            # This unique save path is what creates the distinct result directory for the PDF
            unique_save_path = f"./results/{scenario.name}/{hp_suffix}"
            current_grid["experiment.save_path"] = [unique_save_path]
            temp_scenario_name = f"{scenario.name}/{hp_suffix}"

            # 3. Create a temporary Scenario
            temp_scenario = Scenario(
                name=temp_scenario_name,
                base_config_factory=scenario.base_config_factory,
                modifiers=[],
                parameter_grid=current_grid  # Grid now includes 'n_sellers' and 'save_path'
            )

            # 4. Generate the config
            base_config = temp_scenario.base_config_factory()
            modified_base_config = copy.deepcopy(base_config)

            # Apply all modifiers (includes tuning, golden HPs, and attack setup)
            for modifier in scenario.modifiers:
                modified_base_config = modifier(modified_base_config)

            # Generator applies the parameter grid (n_sellers and save_path)
            num_gen = generator.generate(modified_base_config, temp_scenario)
            task_configs += num_gen

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 10 (Scalability Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps: Run the experiments using these generated configurations.")
    print(f"Example config path: {output_dir}/step10_scalability_martfl_CIFAR100/n_sellers_10/...")
