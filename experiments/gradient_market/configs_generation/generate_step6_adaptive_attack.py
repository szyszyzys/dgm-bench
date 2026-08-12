# FILE: generate_step7_adaptive_attack.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE,
    IMAGE_DEFENSES, FOCUSED_DEFENSES,
    enable_valuation, get_tuned_defense_params,
    GOLDEN_TRAINING_PARAMS
)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import Scenario, use_adaptive_attack, \
    use_cifar100_config

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

# Reduced: dropped data_poisoning (redundant with step 4 backdoor sweep)
ADAPTIVE_MODES_TO_TEST = ["gradient_manipulation"]
ADAPTIVE_THREAT_MODELS_TO_TEST = ["black_box", "oracle"]
EXPLORATION_ROUNDS = 30
ADAPTIVE_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
}


def generate_adaptive_attack_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses against adaptive attackers."""
    print("\n--- Generating Step 7: Adaptive Attack Scenarios ---")
    scenarios = []
    modality = ADAPTIVE_SETUP["modality_name"]
    model_cfg_name = ADAPTIVE_SETUP["model_config_name"]

    # FOCUSED: limited to FLTrust + MartFL for the in-depth adaptive-attack sweep.
    # Full defense coverage lives in Step 10 (main summary).
    for defense_name in FOCUSED_DEFENSES:
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor"
        )
        print(f"-- Processing Defense: {defense_name}")
        if not tuned_defense_params:
            print(f"  SKIPPING: No Tuned HPs found for {defense_name}")
            continue

        def create_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params,
                current_modality=modality
        ):
            def modifier(config: AppConfig) -> AppConfig:
                golden_hp_key = f"{current_model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")
                if "skymask" in current_defense_name:
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                for key, value in current_tuned_params.items():
                    set_nested_attr(config, key, value)

                set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)
                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # --- (NEW) ADD THE "NO ATTACK" BASELINE SCENARIO ---
        print("    -- Adaptive Mode: 0. Baseline (No Attack)")
        baseline_scenario_name = f"step7_baseline_no_attack_{defense_name}_{ADAPTIVE_SETUP['dataset_name']}"
        baseline_save_path = f"./results/{baseline_scenario_name}"

        baseline_grid = {
            ADAPTIVE_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [ADAPTIVE_SETUP["dataset_name"]],
            "experiment.adv_rate": [0.0],  # <-- Set adv_rate to 0
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            "experiment.save_path": [baseline_save_path]
        }

        baseline_scenario = Scenario(
            name=baseline_scenario_name,
            base_config_factory=ADAPTIVE_SETUP["base_config_factory"],
            modifiers=[
                setup_modifier_func,  # Use same tuned HPs
                ADAPTIVE_SETUP["dataset_modifier"],
                lambda config: enable_valuation(  # Still run valuation
                    config,
                    influence=True,
                    loo=True,
                    loo_freq=10,
                    kernelshap=False
                )
                # --- NOTE: We DO NOT add the adaptive_modifier ---
            ],
            parameter_grid=baseline_grid
        )
        scenarios.append(baseline_scenario)
        # --- END OF NEW BLOCK ---

        # --- Loop over threat models FIRST ---
        for threat_model in ADAPTIVE_THREAT_MODELS_TO_TEST:
            print(f"  -- Threat Model: {threat_model}")

            for adaptive_mode in ADAPTIVE_MODES_TO_TEST:
                print(f"    -- Adaptive Mode: {adaptive_mode}")

                if threat_model != "black_box" and adaptive_mode == "data_poisoning":
                    print(f"       Skipping data_poisoning for {threat_model} (N/A)")
                    continue

                adaptive_modifier = use_adaptive_attack(
                    mode=adaptive_mode,
                    threat_model=threat_model,
                    exploration_rounds=EXPLORATION_ROUNDS
                )

                scenario_name = f"step7_adaptive_{threat_model}_{adaptive_mode}_{defense_name}_{ADAPTIVE_SETUP['dataset_name']}"
                unique_save_path = f"./results/{scenario_name}"

                grid = {
                    ADAPTIVE_SETUP["model_config_param_key"]: [model_cfg_name],
                    "experiment.dataset_name": [ADAPTIVE_SETUP["dataset_name"]],
                    "experiment.adv_rate": [DEFAULT_ADV_RATE],  # Use default adv_rate
                    "n_samples": [NUM_SEEDS_PER_CONFIG],
                    "experiment.use_early_stopping": [True],
                    "experiment.patience": [10],
                    "experiment.save_path": [unique_save_path]
                }

                scenario = Scenario(
                    name=scenario_name,
                    base_config_factory=ADAPTIVE_SETUP["base_config_factory"],
                    modifiers=[
                        setup_modifier_func,
                        ADAPTIVE_SETUP["dataset_modifier"],
                        adaptive_modifier,  # This is the adaptive attack
                        lambda config: enable_valuation(
                            config,
                            influence=True,
                            loo=True,
                            loo_freq=10,
                            kernelshap=False
                        )
                    ],
                    parameter_grid=grid
                )
                scenarios.append(scenario)

    return scenarios


# --- Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step7_adaptive_attack"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_adaptive_attack_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 7 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        temp_scenario = Scenario(
            name=scenario.name,
            base_config_factory=scenario.base_config_factory,
            modifiers=[],
            parameter_grid=scenario.parameter_grid
        )
        num_gen = generator.generate(modified_base_config, temp_scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 7 (Adaptive Attack Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")