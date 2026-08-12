# FILE: generate_step8_buyer_attacks.py

import copy
import sys
from pathlib import Path
from typing import List

from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,
    NUM_SEEDS_PER_CONFIG, DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
    get_tuned_defense_params, enable_valuation, IMAGE_DEFENSES, FOCUSED_DEFENSES,
)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import Scenario, use_buyer_dos_attack, \
    use_buyer_starvation_attack, use_buyer_erosion_attack, use_buyer_class_exclusion_attack, \
    use_buyer_oscillating_attack, use_buyer_orthogonal_pivot_attack, use_cifar100_config, \
    use_image_backdoor_attack

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType, BuyerAttackConfig
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

BUYER_ATTACK_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
}
BUYER_ATTACK_CONFIGS = [
    # Reduced from 9 → 4 representative variants. Dropped:
    #   - class_exclusion_pos (kept _neg as the representative)
    #   - oscillating_random and oscillating_drift (kept oscillating_binary)
    #   - orthogonal_pivot_legacy (literally legacy)
    #   - erosion (basically slow-motion starvation — redundant)
    ("dos", use_buyer_dos_attack()),
    ("starvation", use_buyer_starvation_attack(target_classes=[0, 1])),
    ("class_exclusion_neg", use_buyer_class_exclusion_attack(exclude_classes=[7, 8, 9], gradient_scale=1.2)),
    ("oscillating_binary", use_buyer_oscillating_attack(strategy="binary_flip", period=5)),
]


def generate_buyer_attack_scenarios() -> List[Scenario]:
    """Generates scenarios testing tuned defenses against various buyer attacks."""
    print("\n--- Generating Step 8: Buyer Attack Scenarios ---")
    scenarios = []
    modality = BUYER_ATTACK_SETUP["modality_name"]
    model_cfg_name = BUYER_ATTACK_SETUP["model_config_name"]

    # FOCUSED: limited to FLTrust + MartFL for the in-depth buyer-attack sweep.
    # Full defense coverage lives in Step 10 (main summary).
    for defense_name in FOCUSED_DEFENSES:
        # Get Tuned HPs (from Step 3)
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",  # Use a default
            default_attack_type_for_tuning="backdoor"
        )
        if not tuned_defense_params:
            print(f"  SKIPPING {defense_name}: No tuned parameters found.")
            continue

        print(f"-- Processing Defense: {defense_name}")
        def create_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params,
                current_modality=modality
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # --- Apply Golden Training HPs (from Step 2.5) ---
                golden_hp_key = f"{current_model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                # --- Apply Tuned Defense HPs (from Step 3) ---
                for key, value in current_tuned_params.items():
                    set_nested_attr(config, key, value)
                if "skymask" in current_defense_name:
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)

                config.experiment.adv_rate = 0.0
                config.adversary_seller_config.poisoning.type = PoisonType.NONE
                config.adversary_seller_config.sybil.is_sybil = False

                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # 2. Loop through buyer attack types
        for attack_tag, buyer_attack_modifier in BUYER_ATTACK_CONFIGS:
            print(f"  -- Buyer Attack Type: {attack_tag}")

            scenario_name = f"step8_buyer_attack_{attack_tag}_{defense_name}_{BUYER_ATTACK_SETUP['dataset_name']}"
            unique_save_path = f"./results/{scenario_name}"

            grid = {
                BUYER_ATTACK_SETUP["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [BUYER_ATTACK_SETUP["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                "experiment.save_path": [unique_save_path]  # <-- ADDED
            }

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=BUYER_ATTACK_SETUP["base_config_factory"],
                modifiers=[
                    setup_modifier_func,
                    BUYER_ATTACK_SETUP["dataset_modifier"],
                    buyer_attack_modifier,
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


# ==============================================================================
# --- Combined Seller + Buyer Attack Scenarios (worst-case marketplace) ---
# ==============================================================================

# Representative subset: one from each buyer attack category (erosion dropped)
COMBINED_BUYER_ATTACKS = [
    ("dos", use_buyer_dos_attack()),
    ("starvation", use_buyer_starvation_attack(target_classes=[0, 1])),
    ("oscillating_binary", use_buyer_oscillating_attack(strategy="binary_flip", period=5)),
]


def generate_combined_attack_scenarios() -> List[Scenario]:
    """
    Generates scenarios where BOTH seller poisoning AND buyer attacks
    are active simultaneously — the realistic worst-case marketplace.

    Seller attack: image backdoor (adv_rate=0.3, poison_rate=0.5)
    Buyer attacks: representative subset (DoS, starvation, erosion, oscillating)
    """
    print("\n--- Generating Step 8b: Combined Seller + Buyer Attack Scenarios ---")
    scenarios = []
    modality = BUYER_ATTACK_SETUP["modality_name"]
    model_cfg_name = BUYER_ATTACK_SETUP["model_config_name"]

    for defense_name in IMAGE_DEFENSES:
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor"
        )
        if not tuned_defense_params:
            print(f"  SKIPPING {defense_name}: No tuned parameters found.")
            continue

        print(f"-- Processing Defense: {defense_name}")

        def create_combined_setup_modifier(
                current_defense_name=defense_name,
                current_model_cfg_name=model_cfg_name,
                current_tuned_params=tuned_defense_params,
                current_modality=modality
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # Apply Golden Training HPs
                golden_hp_key = f"{current_model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)

                # Apply Tuned Defense HPs
                for key, value in current_tuned_params.items():
                    set_nested_attr(config, key, value)
                if "skymask" in current_defense_name:
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)

                # Enable seller attacks (key difference from buyer-only scenarios)
                config.experiment.adv_rate = DEFAULT_ADV_RATE
                set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", DEFAULT_POISON_RATE)

                return config

            return modifier

        setup_modifier_func = create_combined_setup_modifier()

        for attack_tag, buyer_attack_modifier in COMBINED_BUYER_ATTACKS:
            print(f"  -- Combined: seller_backdoor + buyer_{attack_tag}")

            scenario_name = (
                f"step8b_combined_backdoor_{attack_tag}_{defense_name}"
                f"_{BUYER_ATTACK_SETUP['dataset_name']}"
            )
            unique_save_path = f"./results/{scenario_name}"

            grid = {
                BUYER_ATTACK_SETUP["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [BUYER_ATTACK_SETUP["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                "experiment.save_path": [unique_save_path]
            }

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=BUYER_ATTACK_SETUP["base_config_factory"],
                modifiers=[
                    setup_modifier_func,
                    BUYER_ATTACK_SETUP["dataset_modifier"],
                    use_image_backdoor_attack,       # seller-side poisoning
                    buyer_attack_modifier,            # buyer-side attack
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


if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step8_buyer_attacks"
    generator = ExperimentGenerator(str(output_dir))

    # Generate both buyer-only AND combined scenarios
    scenarios_buyer_only = generate_buyer_attack_scenarios()
    scenarios_combined = generate_combined_attack_scenarios()
    all_scenarios = scenarios_buyer_only + scenarios_combined

    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 8 ---")

    for scenario in all_scenarios:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        temp_scenario = Scenario(
            name=scenario.name,
            base_config_factory=scenario.base_config_factory,
            modifiers=[],  # Modifiers already applied to modified_base_config
            parameter_grid=scenario.parameter_grid
        )
        num_gen = generator.generate(modified_base_config, temp_scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 8 (Buyer Attack + Combined Attack Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"  - Buyer-only scenarios: {len(scenarios_buyer_only)}")
    print(f"  - Combined seller+buyer scenarios: {len(scenarios_combined)}")
    print(f"Configs saved to: {output_dir}")
