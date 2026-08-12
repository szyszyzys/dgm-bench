# FILE: generate_step5_attack_sensitivity.py
# (Imports are the same...)

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
    IMAGE_DEFENSES, FOCUSED_DEFENSES, get_tuned_defense_params, GOLDEN_TRAINING_PARAMS, )
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import Scenario, use_image_backdoor_attack, use_label_flipping_attack, \
    use_cifar100_config

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Attack Parameters to Sweep (reduced 4x4 → 3x3) ---
ADV_RATES_TO_SWEEP = [0.1, 0.3, 0.5]
POISON_RATES_TO_SWEEP = [0.1, 0.5, 1.0]

# --- Attack Types to Test (reduced: backdoor only; labelflip is covered in step 10) ---
ATTACK_TYPES = ["backdoor"]

# --- Focus Setup for Attack Sensitivity Analysis ---
SENSITIVITY_SETUP_STEP5 = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
    "backdoor_attack_modifier": use_image_backdoor_attack,
    "labelflip_attack_modifier": use_label_flipping_attack
}


def generate_attack_sensitivity_scenarios() -> List[Scenario]:
    """Generates scenarios sweeping attack strength against tuned defenses."""
    print("\n--- Generating Step 5: Attack Sensitivity Scenarios ---")
    scenarios = []
    modality = SENSITIVITY_SETUP_STEP5["modality_name"]
    model_cfg_name = SENSITIVITY_SETUP_STEP5["model_config_name"]
    # FOCUSED: limited to FLTrust + MartFL for the in-depth attack sensitivity
    # sweep. Full defense coverage lives in Step 10 (main summary).
    current_defenses = FOCUSED_DEFENSES

    for defense_name in current_defenses:
        for attack_type in ATTACK_TYPES:
            print(f"-- Processing Defense: {defense_name} (for {attack_type} attack) --")

            # 1. Get the T tuned HPs for this defense/attack combo (from Step 3)
            tuned_defense_params = get_tuned_defense_params(
                defense_name=defense_name,
                model_config_name=model_cfg_name,
                attack_state="with_attack",
                explicit_attack_type=attack_type
            )
            if not tuned_defense_params:
                print(f"  SKIPPING: No tuned params found for {defense_name}_{model_cfg_name}_{attack_type}")
                continue

            # 2. Get the attack modifier function
            attack_modifier_key = f"{attack_type}_attack_modifier"
            if attack_modifier_key not in SENSITIVITY_SETUP_STEP5:
                print(f"  SKIPPING: No attack modifier function found for {attack_type}")
                continue
            attack_modifier = SENSITIVITY_SETUP_STEP5[attack_modifier_key]

            # 3. Create the setup modifier function *inside the loop*
            def create_setup_modifier(
                    current_defense_name=defense_name,
                    current_model_cfg_name=model_cfg_name,
                    current_tuned_params=tuned_defense_params,
                    current_modality=modality
            ):
                def modifier(config: AppConfig) -> AppConfig:
                    # golden_hp_key = f"{current_defense_name}_{current_model_cfg_name}"
                    golden_hp_key = f"{current_model_cfg_name}"
                    training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                    if training_params:
                        for key, value in training_params.items():
                            set_nested_attr(config, key, value)
                    else:
                        print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")
                        # This is a critical error, you might want to uncomment the next line
                        # sys.exit(f"Missing critical HPs for {golden_hp_key}")
                    if "skymask" in current_defense_name:
                        model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                        set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                    # --- Apply Tuned Defense HPs (from Step 3) ---
                    for key, value in current_tuned_params.items():
                        set_nested_attr(config, key, value)

                    # --- Apply other fixed settings ---
                    set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                    set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)
                    config.valuation.run_influence = False
                    config.valuation.run_loo = False
                    config.valuation.run_kernelshap = False
                    return config

                return modifier

            setup_modifier_func = create_setup_modifier()

            # 4. Base grid - DOES NOT include the sweep
            base_grid = {
                SENSITIVITY_SETUP_STEP5["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [SENSITIVITY_SETUP_STEP5["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
            }

            # --- Scenario for varying Adv Rate ---
            grid_adv = base_grid.copy()
            grid_adv["sweep_type"] = ["adv_rate"]
            scenarios.append(Scenario(
                name=f"step5_atk_sens_adv_{defense_name}_{attack_type}_{modality}",
                base_config_factory=SENSITIVITY_SETUP_STEP5["base_config_factory"],
                modifiers=[setup_modifier_func, SENSITIVITY_SETUP_STEP5["dataset_modifier"], attack_modifier],
                parameter_grid=grid_adv
            ))

            # --- Scenario for varying Poison Rate ---
            grid_poison = base_grid.copy()
            grid_poison["sweep_type"] = ["poison_rate"]
            scenarios.append(Scenario(
                name=f"step5_atk_sens_poison_{defense_name}_{attack_type}_{modality}",
                base_config_factory=SENSITIVITY_SETUP_STEP5["base_config_factory"],
                modifiers=[setup_modifier_func, SENSITIVITY_SETUP_STEP5["dataset_modifier"], attack_modifier],
                parameter_grid=grid_poison
            ))

    return scenarios


if __name__ == "__main__":
    # (Your main block is perfect)
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step5_attack_sensitivity"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_attack_sensitivity_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 5 ---")

    # --- Manual Loop REQUIRED to create unique save paths ---
    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Get the static grid params
        static_grid = {k: v for k, v in scenario.parameter_grid.items() if k not in
                       ["experiment.adv_rate", "adversary_seller_config.poisoning.poison_rate", "sweep_type"]}

        sweep_type = scenario.parameter_grid.get("sweep_type", [""])[0]

        # Determine which sweep to perform
        if sweep_type == "adv_rate":
            sweep_list = [(adv_rate, DEFAULT_POISON_RATE) for adv_rate in ADV_RATES_TO_SWEEP]
        elif sweep_type == "poison_rate":
            sweep_list = [(DEFAULT_ADV_RATE, poison_rate) for poison_rate in POISON_RATES_TO_SWEEP]
        else:
            print(f"  Skipping {scenario.name}: Unknown sweep_type '{sweep_type}'")
            continue

        # Loop through each attack strength combination
        for adv_rate, poison_rate in sweep_list:

            # 1. Create the specific grid for this combination
            current_grid = static_grid.copy()
            current_grid["experiment.adv_rate"] = [adv_rate]
            current_grid["adversary_seller_config.poisoning.poison_rate"] = [poison_rate]

            # 2. Define unique output path
            hp_suffix = f"adv_{adv_rate}_poison_{poison_rate}"
            unique_save_path = f"./results/{scenario.name}/{hp_suffix}"
            current_grid["experiment.save_path"] = [unique_save_path]
            temp_scenario_name = f"{scenario.name}/{hp_suffix}"

            # 3. Create a temporary Scenario
            temp_scenario = Scenario(
                name=temp_scenario_name,
                base_config_factory=scenario.base_config_factory,
                modifiers=[],
                parameter_grid=current_grid
            )

            # 4. Generate the config
            base_config = temp_scenario.base_config_factory()
            modified_base_config = copy.deepcopy(base_config)
            for modifier in scenario.modifiers:
                modified_base_config = modifier(modified_base_config)

            # Handle the 0,0 case (no attack)
            if adv_rate == 0.0 or poison_rate == 0.0:
                modified_base_config.adversary_seller_config.poisoning.type = PoisonType.NONE
                modified_base_config.experiment.adv_rate = 0.0  # Force adv rate to 0 if either is 0

            num_gen = generator.generate(modified_base_config, temp_scenario)
            task_configs += num_gen

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 5 (Attack Sensitivity) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
