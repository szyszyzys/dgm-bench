# FILE: generate_step19_new_defenses.py
# =============================================================================
# Main summary for the 4 new defenses: FLAME, DeepSight, Bulyan, FoolsGold.
# Mirrors Step 10 (main summary) but only runs these new defenses across
# all datasets, so their results appear alongside existing defenses in
# the final comparison figures.
# =============================================================================

import copy
import sys
from pathlib import Path
from typing import List

from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
    enable_valuation, get_tuned_defense_params,
    GOLDEN_TRAINING_PARAMS,
    NEW_IMAGE_DEFENSES, NEW_TEXT_TABULAR_DEFENSES,
    ENABLED_DATASETS, filter_enabled_targets,
)
from experiments.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config
)
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_femnist_config, use_cifar100_config, use_trec_config,
    use_image_backdoor_attack, use_text_backdoor_attack
)
from experiments.gradient_market.automate_exp import (
    TEXAS100_TRIGGER, use_tabular_backdoor_with_trigger,
    TEXAS100_TARGET_LABEL, get_base_tabular_config,
    PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL
)

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# Defense lists imported from config_common_utils (NEW_IMAGE_DEFENSES, NEW_TEXT_TABULAR_DEFENSES)
# Note: Bulyan requires N >= 4f+3. With N=10 and f=1, that's OK (N >= 7).
#       With f=3 (30% of 10), N=10 < 4*3+3=15, so Bulyan will fallback to FedAvg.
#       We set num_byzantine=1 for Bulyan so it can run with N=10.

FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE

MAIN_SUMMARY_TARGETS = [
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Texas100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_texas100_baseline",
     "dataset_modifier": lambda cfg: cfg,
     "attack_modifier": use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL)},
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Purchase100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_purchase100_baseline",
     "dataset_modifier": lambda cfg: cfg,
     "attack_modifier": use_tabular_backdoor_with_trigger(PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL)},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "FEMNIST",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "femnist_cnn",
     "dataset_modifier": use_femnist_config, "attack_modifier": use_image_backdoor_attack},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR100",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar100_cnn",
     "dataset_modifier": use_cifar100_config, "attack_modifier": use_image_backdoor_attack},
    {"modality_name": "text", "base_config_factory": get_base_text_config, "dataset_name": "TREC",
     "model_config_param_key": "experiment.text_model_config_name", "model_config_name": "textcnn_trec_baseline",
     "dataset_modifier": use_trec_config, "attack_modifier": use_text_backdoor_attack},
]

# Filter to ENABLED_DATASETS (single source of truth in config_common_utils.py).
_skipped_step19 = [t["dataset_name"] for t in MAIN_SUMMARY_TARGETS
                   if t["dataset_name"] not in ENABLED_DATASETS]
MAIN_SUMMARY_TARGETS = filter_enabled_targets(MAIN_SUMMARY_TARGETS)
if _skipped_step19:
    print(f"[Step 19] Skipping disabled datasets: {_skipped_step19}")
    print(f"[Step 19] Enabled datasets        : {sorted(ENABLED_DATASETS)}")


def generate_new_defense_scenarios() -> List[Scenario]:
    """Generates main benchmark configs for FLAME, DeepSight, Bulyan, FoolsGold."""
    print("\n--- Generating Step 19: New Defenses Main Summary ---")
    scenarios = []

    for target in MAIN_SUMMARY_TARGETS:
        modality = target["modality_name"]
        model_cfg_name = target["model_config_name"]
        print(f"-- Processing: {modality} {model_cfg_name}")

        current_defenses = NEW_IMAGE_DEFENSES if modality == "image" else NEW_TEXT_TABULAR_DEFENSES

        for defense_name in current_defenses:

            # Get Tuned HPs (from Step 3). May return None if not yet tuned.
            tuned_defense_params = get_tuned_defense_params(
                defense_name=defense_name,
                model_config_name=model_cfg_name,
                attack_state="with_attack",
                default_attack_type_for_tuning="backdoor"
            )
            print(f"  - Defense: {defense_name}")

            # For new defenses, tuned params may not exist yet.
            # Use default params (empty dict) if not tuned — the aggregator defaults
            # from FLAMEParams/DeepSightParams/etc. will be used.
            if not tuned_defense_params:
                print(f"    NOTE: No tuned params for {defense_name}_{model_cfg_name}. Using defaults.")
                tuned_defense_params = {"aggregation.method": defense_name}

            def create_setup_modifier(
                    current_defense_name=defense_name,
                    current_model_cfg_name=model_cfg_name,
                    current_tuned_params=tuned_defense_params,
                    current_modality=modality
            ):
                def modifier(config: AppConfig) -> AppConfig:
                    # 1. Apply Golden Training HPs
                    golden_hp_key = f"{current_model_cfg_name}"
                    training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                    if training_params:
                        for key, value in training_params.items():
                            set_nested_attr(config, key, value)
                    else:
                        print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                    # 2. Apply Tuned Defense HPs
                    for key, value in current_tuned_params.items():
                        set_nested_attr(config, key, value)

                    # 3. Data settings
                    set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                    set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)
                    return config

                return modifier

            setup_modifier_func = create_setup_modifier()

            valuation_modifier = lambda config: enable_valuation(
                config,
                influence=True,
                loo=True, loo_freq=5,
                kernelshap=True, kshap_freq=5,
                kshap_samples=500
            )

            scenario_name = (
                f"step19_new_defenses_{defense_name}_{modality}"
                f"_{target['dataset_name']}_{model_cfg_name.split('_')[-1]}"
            )
            unique_save_path = f"./results/{scenario_name}"

            grid = {
                target["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [target["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.adv_rate": [FIXED_ATTACK_ADV_RATE],
                "adversary_seller_config.poisoning.poison_rate": [FIXED_ATTACK_POISON_RATE],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                "experiment.save_path": [unique_save_path]
            }

            all_modifiers = [
                setup_modifier_func,
                target["dataset_modifier"],
                target["attack_modifier"],
                valuation_modifier
            ]
            scenario = Scenario(
                name=scenario_name,
                base_config_factory=target["base_config_factory"],
                modifiers=all_modifiers,
                parameter_grid=grid
            )
            scenarios.append(scenario)

    return scenarios


if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step19_new_defenses"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_new_defense_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 19 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        temp_scenario = Scenario(
            name=scenario.name,
            base_config_factory=scenario.base_config_factory,
            modifiers=[],  # Modifiers already applied above
            parameter_grid=scenario.parameter_grid
        )
        num_gen = generator.generate(modified_base_config, temp_scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 19 (New Defenses: FLAME, DeepSight, Bulyan, FoolsGold) complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNew defenses tested:")
    print("  - FLAME (HDBSCAN + adaptive clip + DP noise)")
    print("  - DeepSight (NEUP + DDUP dual-metric filtering)")
    print("  - Bulyan (Krum selection + trimmed median)")
    print("  - FoolsGold (historical gradient similarity)")
