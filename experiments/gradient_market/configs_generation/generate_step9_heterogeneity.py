# FILE: generate_step11_heterogeneity.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, get_tuned_defense_params, IMAGE_DEFENSES, FOCUSED_DEFENSES,
)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import Scenario, use_cifar100_config, use_image_backdoor_attack

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Constants ---

# DEFINITION: (Strategy, Alpha Value, Folder Suffix) — dropped IID (already in step 1)
HETEROGENEITY_SWEEP = [
    ("dirichlet", 1.0, "alpha_1.0"),  # Mild
    ("dirichlet", 0.5, "alpha_0.5"),  # Moderate
    ("dirichlet", 0.1, "alpha_0.1"),  # Severe
]

# --- Scarcity Sweep (root-size sweep for R3.O4): below / at / above the
#     default buyer_ratio of 0.1, so it is an actual sweep, not a single point. ---
SCARCITY_RATIOS = [0.05, 0.1, 0.2]

# Constants for the "Fixed" side of the experiments
FIXED_BUYER_STRAT = "iid"
FIXED_BUYER_ALPHA = None

FIXED_SELLER_STRAT = "dirichlet"
FIXED_SELLER_ALPHA = 0.5

FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE
DEFAULT_BUYER_RATIO = 0.1

HETEROGENEITY_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
    "attack_modifier": use_image_backdoor_attack
}


def initialize_adversary_data_distribution(config: AppConfig) -> AppConfig:
    """Ensures adversary config has data_distribution field."""
    poisoning_cfg = config.adversary_seller_config.poisoning
    if not hasattr(poisoning_cfg, 'data_distribution') or poisoning_cfg.data_distribution is None:
        poisoning_cfg.data_distribution = {}
    return config


def generate_heterogeneity_scenarios() -> List[Scenario]:
    print("\n--- Generating Step 11: Heterogeneity Impact Scenarios ---")
    scenarios = []
    model_cfg_name = HETEROGENEITY_SETUP["model_config_name"]

    # FOCUSED: limited to FLTrust + MartFL for the in-depth heterogeneity sweep.
    # Full defense coverage lives in Step 10 (main summary).
    for defense_name in FOCUSED_DEFENSES:
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor"
        )
        if not tuned_defense_params and defense_name != "fedavg":
            continue

        def create_setup_modifier(current_params=tuned_defense_params, defense=defense_name):
            def modifier(config: AppConfig) -> AppConfig:
                # 1. Golden HPs
                golden_hp_key = f"{model_cfg_name}"
                training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)

                # 2. Defense HPs
                if current_params:
                    for key, value in current_params.items():
                        set_nested_attr(config, key, value)
                if "skymask" in defense:
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                # 3. Attack Fixed Config
                config.experiment.adv_rate = FIXED_ATTACK_ADV_RATE
                config = HETEROGENEITY_SETUP["attack_modifier"](config)
                set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", FIXED_ATTACK_POISON_RATE)

                # 4. Disable Valuation (Speed up)
                config.valuation.run_influence = False
                config.valuation.run_loo = False
                config.valuation.run_kernelshap = False
                return config

            return modifier

        # Base Grid
        parameter_grid = {
            HETEROGENEITY_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [HETEROGENEITY_SETUP["dataset_name"]],
            "n_samples": [5],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            "data.image.buyer_ratio": [DEFAULT_BUYER_RATIO],
            # 5 seeds (overrides the repo default of 2) so the biased-D_B
            # transition and the root-size sweep have tight error bars. The
            # parallel runner resumes: existing seeds 42/43 skip via .success,
            # 44/45/46 run.
        }

        scenario_name = f"step11_{defense_name}_{HETEROGENEITY_SETUP['dataset_name']}"

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=HETEROGENEITY_SETUP["base_config_factory"],
            modifiers=[create_setup_modifier(), HETEROGENEITY_SETUP["dataset_modifier"]],
            parameter_grid=parameter_grid
        )
        scenarios.append(scenario)

    return scenarios


if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step11_heterogeneity"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_heterogeneity_scenarios()
    all_generated_configs = 0
    modality = HETEROGENEITY_SETUP["modality_name"]

    print("\n--- Generating Configuration Files for Step 11 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0
        # Extract defense name from scenario name format: "step11_{defense_name}_{dataset}"
        # Defense names can contain underscores (e.g., trimmed_mean, multi_krum)
        dataset_name = HETEROGENEITY_SETUP['dataset_name']
        prefix = "step11_"
        suffix = f"_{dataset_name}"
        defense_name = scenario.name[len(prefix):-len(suffix)] if scenario.name.endswith(suffix) else scenario.name.split('_')[1]

        print(f"  -> Generating Group A: Vary Seller (Buyer Fixed IID)")
        for sweep_strat, sweep_alpha, suffix in HETEROGENEITY_SWEEP:
            current_grid = scenario.parameter_grid.copy()

            # 1. Seller: Sweeps
            current_grid[f"data.{modality}.strategy"] = [sweep_strat]
            if sweep_alpha is not None:
                current_grid[f"data.{modality}.dirichlet_alpha"] = [sweep_alpha]

            # 2. Buyer: Fixed [IID]
            current_grid[f"data.{modality}.buyer_strategy"] = [FIXED_BUYER_STRAT]
            if FIXED_BUYER_ALPHA is not None:
                current_grid[f"data.{modality}.buyer_dirichlet_alpha"] = [FIXED_BUYER_ALPHA]

            # 3. Adversary: Matches Seller
            current_grid["adversary_seller_config.poisoning.data_distribution.strategy"] = [sweep_strat]
            if sweep_alpha is not None:
                current_grid["adversary_seller_config.poisoning.data_distribution.dirichlet_alpha"] = [sweep_alpha]

            # Path: vary_seller/alpha_x
            save_suffix = f"vary_seller/{suffix}"
            current_grid["experiment.save_path"] = [f"./results/{scenario.name}/{save_suffix}"]

            temp_scenario = Scenario(name=f"{scenario.name}/{save_suffix}",
                                     base_config_factory=scenario.base_config_factory, modifiers=[],
                                     parameter_grid=current_grid)
            base_config = temp_scenario.base_config_factory()
            mod_config = initialize_adversary_data_distribution(copy.deepcopy(base_config))

            for modifier in scenario.modifiers: mod_config = modifier(mod_config)
            task_configs += generator.generate(mod_config, temp_scenario)

        # =======================================================================
        # EXPERIMENT B: VARY BUYER (Fix Seller to 0.5)
        # =======================================================================
        print(f"  -> Generating Group B: Vary Buyer (Seller Fixed 0.5)")
        for sweep_strat, sweep_alpha, suffix in HETEROGENEITY_SWEEP:
            current_grid = scenario.parameter_grid.copy()

            # 1. Seller: Fixed [Dirichlet 0.5]
            current_grid[f"data.{modality}.strategy"] = [FIXED_SELLER_STRAT]
            if FIXED_SELLER_ALPHA is not None:
                current_grid[f"data.{modality}.dirichlet_alpha"] = [FIXED_SELLER_ALPHA]

            # 2. Buyer: Sweeps
            current_grid[f"data.{modality}.buyer_strategy"] = [sweep_strat]
            if sweep_alpha is not None:
                current_grid[f"data.{modality}.buyer_dirichlet_alpha"] = [sweep_alpha]

            # 3. Adversary: Matches Seller
            current_grid["adversary_seller_config.poisoning.data_distribution.strategy"] = [FIXED_SELLER_STRAT]
            if FIXED_SELLER_ALPHA is not None:
                current_grid["adversary_seller_config.poisoning.data_distribution.dirichlet_alpha"] = [
                    FIXED_SELLER_ALPHA]

            # Path: vary_buyer/alpha_x
            save_suffix = f"vary_buyer/{suffix}"
            current_grid["experiment.save_path"] = [f"./results/{scenario.name}/{save_suffix}"]

            temp_scenario = Scenario(name=f"{scenario.name}/{save_suffix}",
                                     base_config_factory=scenario.base_config_factory, modifiers=[],
                                     parameter_grid=current_grid)
            base_config = temp_scenario.base_config_factory()
            mod_config = initialize_adversary_data_distribution(copy.deepcopy(base_config))

            for modifier in scenario.modifiers: mod_config = modifier(mod_config)
            task_configs += generator.generate(mod_config, temp_scenario)

        # =======================================================================
        # EXPERIMENT C: VARY BUYER RATIO (Scarcity)
        # Goal: Test impact of data quantity. Fixed Seller=0.5, Buyer=IID.
        # =======================================================================
        print(f"  -> Generating Group C: Scarcity (Buyer Ratio Sweep)")
        for ratio in SCARCITY_RATIOS:
            current_grid = scenario.parameter_grid.copy()

            # 1. Fixed Seller: Dirichlet 0.5
            current_grid[f"data.{modality}.strategy"] = [FIXED_SELLER_STRAT]
            if FIXED_SELLER_ALPHA is not None:
                current_grid[f"data.{modality}.dirichlet_alpha"] = [FIXED_SELLER_ALPHA]

            # 2. Fixed Buyer: IID
            current_grid[f"data.{modality}.buyer_strategy"] = [FIXED_BUYER_STRAT]
            if FIXED_BUYER_ALPHA is not None:
                current_grid[f"data.{modality}.buyer_dirichlet_alpha"] = [FIXED_BUYER_ALPHA]

            # 3. VARY BUYER RATIO
            current_grid[f"data.{modality}.buyer_ratio"] = [ratio]

            # 4. Sync Adversary
            current_grid["adversary_seller_config.poisoning.data_distribution.strategy"] = [FIXED_SELLER_STRAT]
            if FIXED_SELLER_ALPHA is not None:
                current_grid["adversary_seller_config.poisoning.data_distribution.dirichlet_alpha"] = [
                    FIXED_SELLER_ALPHA]

            # Path: scarcity/ratio_x
            save_suffix = f"scarcity/ratio_{ratio}"
            current_grid["experiment.save_path"] = [f"./results/{scenario.name}/{save_suffix}"]

            temp_scenario = Scenario(name=f"{scenario.name}/{save_suffix}",
                                     base_config_factory=scenario.base_config_factory, modifiers=[],
                                     parameter_grid=current_grid)
            base_config = temp_scenario.base_config_factory()
            mod_config = initialize_adversary_data_distribution(copy.deepcopy(base_config))

            for modifier in scenario.modifiers: mod_config = modifier(mod_config)
            task_configs += generator.generate(mod_config, temp_scenario)

        all_generated_configs += task_configs

    print(f"\n✅ Step 11 Config Generation Complete! ({all_generated_configs} files)")
    print(f"Configs saved to: {output_dir}")
