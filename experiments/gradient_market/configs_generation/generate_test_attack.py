# FILE: generate_step2_attack_validation.py (New File)

import copy
import sys
from typing import Callable

# --- (Imports) ---
from experiments.gradient_market.configs_generation.config_common_utils import GOLDEN_TRAINING_PARAMS, \
    NUM_SEEDS_PER_CONFIG
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario
)
from experiments.gradient_market.automate_exp import get_base_tabular_config, use_tabular_backdoor_with_trigger, \
    TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Defense Parameters (ONLY FedAvg) ---
TUNED_DEFENSE_PARAMS = {
    "fedavg": {  # Include FedAvg as the undefended baseline
        "aggregation.method": ["fedavg"],
    }
}

# --- Attack Parameters to Test (One benign, one strong attack) --- ⚔️
ATTACK_ADV_RATES_TO_SWEEP = [0.0, 0.3]  # 0% attackers, 30% attackers
ATTACK_POISON_RATES_TO_SWEEP = [0.1, 0.3, 0.5, 0.7, 1.0]  # 100% poison rate for a clear signal

# --- Valuation Settings (OFF) ---
VALUATION_PARAMS = {
    "valuation.run_influence": [False],
    "valuation.run_loo": [False],
    "valuation.run_kernelshap": [False],
}

# --- Define ALL Model/Dataset combinations for validation ---
VALIDATION_COMBOS = [
    # {
    #     "modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR10",
    #     "model_structure": "flexiblecnn",  # <-- ADDED
    #     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar10_cnn",
    #     "attack_modifier": use_image_backdoor_attack, "dataset_modifier": use_cifar10_config,
    #     "sm_model_type": "flexiblecnn"
    # },
    # {
    #     "modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR100",
    #     "model_structure": "flexiblecnn",  # <-- ADDED
    #     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar100_cnn",
    #     "attack_modifier": use_image_backdoor_attack, "dataset_modifier": use_cifar100_config,
    #     "sm_model_type": "flexiblecnn"
    # },
    {
        "modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Texas100",
        "model_structure": "mlp",  # <-- ADDED
        "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_texas100_baseline",
        "attack_modifier": use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL),
        "dataset_modifier": lambda cfg: cfg, "sm_model_type": "mlp"
    },
    # {
    #     "modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Purchase100",
    #     "model_structure": "mlp",  # <-- ADDED
    #     "model_config_param_key": "experiment.tabular_model_config_name",
    #     "model_config_name": "mlp_purchase100_baseline",
    #     "attack_modifier": use_tabular_backdoor_with_trigger(PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL),
    #     "dataset_modifier": lambda cfg: cfg, "sm_model_type": "mlp"
    # },
    # {
    #     "modality_name": "text", "base_config_factory": get_base_text_config, "dataset_name": "TREC",
    #     "model_structure": "textcnn",  # <-- ADDED
    #     "model_config_param_key": "experiment.text_model_config_name", "model_config_name": "textcnn_trec_baseline",
    #     "attack_modifier": use_text_backdoor_attack, "dataset_modifier": use_trec_config,
    #     "sm_model_type": "textcnn"
    # },
]


# --- Function to apply Golden Training Parameters ONLY ---
def apply_benchmark_setup(config: AppConfig, model_config_name: str, dataset_modifier: Callable,
                          attack_modifier: Callable) -> AppConfig:
    """Applies dataset modifier, golden training params, and attack modifier."""
    config = dataset_modifier(config)  # Apply dataset specifics first

    # Apply Golden Training Parameters (keyed by model_config_name)
    if model_config_name in GOLDEN_TRAINING_PARAMS:
        params = GOLDEN_TRAINING_PARAMS[model_config_name]
        for key, value in params.items():
            set_nested_attr(config, key, value)
        print(f"  Applied Golden Training Params for {model_config_name}")
    else:
        print(f"  Warning: No Golden Params for model '{model_config_name}'")

    # Apply the base attack structure (poison rate is set by grid)
    config = attack_modifier(config)
    print("  Applied base attack modifier")

    # --- Ensure NON-IID Data ---
    modality = config.experiment.dataset_type
    set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
    set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)

    return config


# --- Main Execution Block ---
if __name__ == "__main__":

    output_dir = "./configs_generated_new/step2_attack_validation"  # New directory
    generator = ExperimentGenerator(output_dir)
    all_benchmark_scenarios = []

    print("\n--- Generating Attack Validation Scenarios (Step 2) ---")

    # Iterate through each Model/Dataset combo
    for combo_config in VALIDATION_COMBOS:
        modality = combo_config["modality_name"]
        model_name = combo_config["model_config_name"]
        print(
            f"\n-- Processing Validation for: Modality={modality}, Dataset={combo_config['dataset_name']}, Model={model_name}")

        # ONLY test FedAvg
        defense_name, fixed_defense_params = "fedavg", TUNED_DEFENSE_PARAMS["fedavg"]

        scenario_name = f"new_step2_validate_{defense_name}_{modality}_{combo_config['dataset_name']}_{combo_config['model_structure']}"
        print(f"  - Defining scenario for {defense_name}: {scenario_name}")


        # Define the setup modifier lambda
        def create_setup_modifier(current_combo_config):
            def setup_modifier_inner(config: AppConfig) -> AppConfig:
                return apply_benchmark_setup(
                    config,
                    current_combo_config["model_config_name"],
                    current_combo_config["dataset_modifier"],
                    current_combo_config["attack_modifier"]
                )

            return setup_modifier_inner


        current_setup_modifier = create_setup_modifier(combo_config)

        # Build the parameter grid for this defense
        full_parameter_grid = {
            # Fixed experiment setup
            "experiment.dataset_name": [combo_config["dataset_name"]],
            "experiment.model_structure": [combo_config["model_structure"]],
            combo_config["model_config_param_key"]: [model_name],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.global_rounds": [100],
            "experiment.use_early_stopping": [True],  # Can be True for this check
            "experiment.patience": [10],

            # Fixed defense parameters (FedAvg)
            **fixed_defense_params,

            # Sweep Attack Parameters (Benign vs. Strong Attack)
            "experiment.adv_rate": ATTACK_ADV_RATES_TO_SWEEP,
            "adversary_seller_config.poisoning.poison_rate": ATTACK_POISON_RATES_TO_SWEEP,

            # Valuation settings (fixed)
            **VALUATION_PARAMS,
        }

        # Handle the 0.0 adv_rate case (Benign)
        # This part of the logic is handled by the config generator,
        # but we must ensure the generator knows to set PoisonType.NONE
        # This is handled in your main `run_attack` function logic (if adv_rate == 0.0 ...)

        scenario = Scenario(
            name=scenario_name,
            base_config_factory=combo_config["base_config_factory"],
            modifiers=[current_setup_modifier],
            parameter_grid=full_parameter_grid
        )
        all_benchmark_scenarios.append(scenario)

    # --- Generate Config Files ---
    print("\n--- Generating Configuration Files ---")
    total_configs = 0
    for scenario in all_benchmark_scenarios:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        # Generator will create 2 configs: (adv=0.0, poison=1.0) and (adv=0.3, poison=1.0)
        num_generated = generator.generate(modified_base_config, scenario)
        total_configs += num_generated
        print(f"  Generated {num_generated} config files.")

    print(f"\n✅ All attack validation configurations generated ({total_configs} total).")
    print(f"   Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run experiments using configs in '{output_dir}'.")
    print("2. Analyze the results. You expect to see:")
    print("   - (adv=0.0): High 'test_acc', Low 'test_asr'")
    print("   - (adv=0.3): High 'test_acc', HIGH 'test_asr' (e.g., > 95%)")
    print("3. If ASR is high, your attack works! You can now proceed to 'Step 3: Defense Tuning'.")
