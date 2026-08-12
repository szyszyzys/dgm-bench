import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    get_tuned_defense_params,
    GOLDEN_TRAINING_PARAMS,
    use_sybil_attack_strategy,  # Required for setting the Sybil strategy name
    ENABLED_DATASETS, filter_enabled_targets,
)
from experiments.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config
)
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_femnist_config, use_cifar100_config, use_trec_config
)
from experiments.gradient_market.automate_exp import (
    get_base_tabular_config
)

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Configuration ---
FIXED_ADV_RATE = 0.3  # Standard Sybil rate
COLLUSION_MODES = ["random"]  # The primary "Oracle" attack mode

# Reuse the datasets from your main summary to ensure comprehensive coverage
TARGET_DATASETS = [
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Texas100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_texas100_baseline",
     "dataset_modifier": lambda cfg: cfg},
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Purchase100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_purchase100_baseline",
     "dataset_modifier": lambda cfg: cfg},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "FEMNIST",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "femnist_cnn",
     "dataset_modifier": use_femnist_config},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR100",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar100_cnn",
     "dataset_modifier": use_cifar100_config},
    {"modality_name": "text", "base_config_factory": get_base_text_config, "dataset_name": "TREC",
     "model_config_param_key": "experiment.text_model_config_name", "model_config_name": "textcnn_trec_baseline",
     "dataset_modifier": use_trec_config},
]

# Filter to ENABLED_DATASETS (single source of truth in config_common_utils.py).
_skipped_step14 = [t["dataset_name"] for t in TARGET_DATASETS
                   if t["dataset_name"] not in ENABLED_DATASETS]
TARGET_DATASETS = filter_enabled_targets(TARGET_DATASETS)
if _skipped_step14:
    print(f"[Step 14] Skipping disabled datasets: {_skipped_step14}")
    print(f"[Step 14] Enabled datasets        : {sorted(ENABLED_DATASETS)}")


def generate_martfl_collusion_scenarios() -> List[Scenario]:
    """
    Generates configs specifically for MartFL under Collusion Attack across all datasets.
    Tests the 'Baseline Hijacking' vulnerability.
    """
    print("\n--- Generating Step 14: MartFL Collusion Comprehensive Analysis ---")
    scenarios = []

    # We only care about MartFL for this deep dive
    defense_name = "martfl"

    for target in TARGET_DATASETS:
        modality = target["modality_name"]
        model_cfg_name = target["model_config_name"]
        dataset_name = target["dataset_name"]
        print(f"-- Processing {dataset_name} ({modality})")

        # 1. Get Tuned HPs for MartFL
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor"
        )

        # 2. Create the Setup Modifier
        def create_setup_modifier(
                current_tuned_params=tuned_defense_params,
                current_model_cfg_name=model_cfg_name,
                current_modality=modality
        ):
            def modifier(config: AppConfig) -> AppConfig:
                # A. Apply Golden Training HPs
                golden_hp_key = f"{current_model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)

                # B. Apply MartFL Tuned HPs
                if current_tuned_params:
                    for key, value in current_tuned_params.items():
                        set_nested_attr(config, key, value)

                # C. Data Strategy
                set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)

                # D. COLLUSION ATTACK SETUP (Crucial)
                # 1. Disable standard poisoning (we attack via gradients)
                config.adversary_seller_config.poisoning.type = PoisonType.NONE

                # 2. Activate Buyer Attack (The Collusion Partner)
                config.buyer_attack_config.is_active = True
                # This tells the Buyer Simulator to execute the specific logic
                # matching the seller's "random" or "inverse" mode
                config.buyer_attack_config.attack_type = "collusion"

                # 3. Disable expensive valuation
                config.valuation.run_influence = False
                config.valuation.run_loo = False
                config.valuation.run_kernelshap = False

                return config

            return modifier

        setup_modifier_func = create_setup_modifier()

        # 3. Base Grid
        base_grid = {
            target["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [dataset_name],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "experiment.use_early_stopping": [False],  # Attack might prevent convergence
            "experiment.global_rounds": [100],
        }

        # 4. Iterate Collusion Modes
        for mode in COLLUSION_MODES:
            scenario_name = f"step14_collusion_{mode}_martfl_{dataset_name}"
            unique_save_path = f"./results/{scenario_name}"

            current_grid = base_grid.copy()
            current_grid["experiment.save_path"] = [unique_save_path]

            # --- CONFIGURE STRATEGY PARAMS ---
            # Set the 'collusion' strategy config dictionary
            config_key_path = "adversary_seller_config.sybil.strategy_configs.collusion"
            collusion_config_dict = {
                "mode": mode,  # "random" or "inverse"
                "noise_scale": 1e-5  # Avoid duplicate detection
            }
            current_grid[config_key_path] = [collusion_config_dict]

            # Build Scenario
            scenario = Scenario(
                name=scenario_name,
                base_config_factory=target["base_config_factory"],
                modifiers=[
                    setup_modifier_func,
                    target["dataset_modifier"],
                    # Activates SybilCoordinator with "collusion" strategy
                    use_sybil_attack_strategy("collusion")
                ],
                parameter_grid=current_grid
            )
            scenarios.append(scenario)

    return scenarios


if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step14_martfl_collusion"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_martfl_collusion_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configs for Step 14 (MartFL Deep Dive) ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing: {scenario.name}")

        # Copy grid to avoid modification issues during generation
        # (Standard generation loop logic)
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
        print(f"-> Generated {num_gen} configs")

    print(f"\n✅ Config generation complete!")
    print(f"Total configurations: {all_generated_configs}")
    print(f"Output Directory: {output_dir}")
