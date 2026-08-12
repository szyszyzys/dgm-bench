# FILE: generate_step2.5_find_usable_hps.py
# (This is a new script. It's a merge of Step 3 and Step 4)

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES,
    # We DO NOT import get_tuned_defense_params
)
from experiments.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config
)
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_femnist_config, use_image_backdoor_attack, use_cifar100_config,
    use_trec_config, use_text_backdoor_attack
)
from experiments.gradient_market.automate_exp import get_base_tabular_config, use_tabular_backdoor_with_trigger, \
    TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL, PURCHASE100_TARGET_LABEL, PURCHASE100_TRIGGER

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr, iter_grid
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

## Purpose:
# Find the "best usable" TRAINING HPs (optimizer, lr, epochs) for EACH defense.
# This script runs a training HP sweep against a "default" (non-tuned) defense
# for all models and datasets.
# The results will be analyzed by `analyze_step4.py` to create the
# *real*, defense-specific `GOLDEN_TRAINING_PARAMS` dictionary.

# --- Training Parameters to Sweep ---
ADAM_LRS_TO_SWEEP = [0.001, 0.0005, 0.0001]
SGD_LRS_TO_SWEEP = [0.1, 0.05, 0.01]
OPTIMIZERS_TO_SWEEP = ["Adam", "SGD"]
LOCAL_EPOCHS_TO_SWEEP = [2, 5]

# --- NEW: Default (non-tuned) Defense HPs ---
# We just need one "good enough" guess to find the best training HPs.
# We use your faster tuning grid's values as a default.
DEFAULT_DEFENSE_HPS = {
    "fedavg": {"aggregation.method": "fedavg"},
    "fltrust": {
        "aggregation.method": "fltrust",
        "aggregation.clip_norm": 5.0  # A reasonable, data-driven guess
    },
    "martfl": {
        "aggregation.method": "martfl",
        "aggregation.martfl.max_k": 5,  # A reasonable guess
        "aggregation.clip_norm": 5.0  # A reasonable guess
    },
    "skymask": {
        "aggregation.method": "skymask",
        "aggregation.skymask.mask_epochs": 20,
        "aggregation.skymask.mask_lr": 0.01,
        "aggregation.skymask.mask_threshold": 0.7,
        "aggregation.clip_norm": 10.0
    },
    "trimmed_mean": {
        "aggregation.method": "trimmed_mean",
        "aggregation.trimmed_mean.trim_ratio": 0.1
    },
    "multi_krum": {
        "aggregation.method": "multi_krum",
        "aggregation.multi_krum.num_byzantine": 3,
        "aggregation.multi_krum.m_selected": 3
    },
    "rflpa": {
        "aggregation.method": "rflpa",
        "aggregation.rflpa.element_clip": 1.0,
        "aggregation.clip_norm": 5.0
    },
    "spmc": {
        "aggregation.method": "spmc",
        "aggregation.spmc.temperature": 0.5
    },
    # --- New defenses (FLAME, DeepSight, Bulyan, FoolsGold) ---
    # Defaults lifted from _DEFAULT_TUNED_DEFENSE_PARAMS in config_common_utils.py
    # so the tunability sweep uses the same starting point as the main pipeline.
    # None of the four use clip_norm (they have their own filtering machinery).
    "flame": {
        "aggregation.method": "flame",
        "aggregation.flame.noise_scale": 0.001,
        "aggregation.flame.min_cluster_size_ratio": 0.5,
    },
    "deepsight": {
        "aggregation.method": "deepsight",
        "aggregation.deepsight.threshold_c": 2.0,
        "aggregation.deepsight.neighbor_ratio": 0.5,
    },
    "bulyan": {
        "aggregation.method": "bulyan",
        "aggregation.bulyan.num_byzantine": 1,  # N=10 < 4f+3 with f>1; falls back to FedAvg otherwise
    },
    "foolsgold": {
        "aggregation.method": "foolsgold",
        "aggregation.foolsgold.pardon_threshold": 0.5,
    },
}

# --- All Models/Datasets Combinations (Copied from Step 3) ---
TUNING_TARGETS = [
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

# --- Dataset scoping (was: hardcoded CIFAR100 only) ---
# The CIFAR100-only filter has been replaced with the global ENABLED_DATASETS
# gate from config_common_utils so that all-dataset main-conclusion runs can
# include FEMNIST / Purchase100 / Texas100 / TREC alongside CIFAR100. To
# narrow back down, edit ENABLED_DATASETS in config_common_utils.py.
from config_common_utils import ENABLED_DATASETS as _ENABLED_DATASETS
TUNING_TARGETS = [t for t in TUNING_TARGETS if t["dataset_name"] in _ENABLED_DATASETS]
assert TUNING_TARGETS, "Step 2 filter left no targets — check ENABLED_DATASETS."


def generate_training_hp_scenarios() -> List[Scenario]:
    """Generates base scenarios for sweeping training HPs against *default* defenses."""
    print("\n--- Generating Step 2.5: Find Usable Training HPs ---")
    scenarios = []

    # Loop over all datasets
    for target in TUNING_TARGETS:
        modality = target["modality_name"]
        model_cfg_name = target["model_config_name"]
        attack_modifier = target["attack_modifier"]
        print(f"-- Processing: {modality} {model_cfg_name}")

        current_defenses = IMAGE_DEFENSES if modality == "image" else TEXT_TABULAR_DEFENSES

        for defense_name in current_defenses:

            # Get the *default* HPs for this defense
            default_defense_params = DEFAULT_DEFENSE_HPS.get(defense_name)
            if not default_defense_params:
                print(f"  Skipping {defense_name}: No default HPs found.")
                continue

            print(f"  -- Defense: {defense_name} (with default HPs)")

            # Create the modifier (this is from Step 4)
            # We bind the *default* defense params
            def create_setup_modifier_sens(
                    current_defense_name=defense_name,
                    current_defense_params=default_defense_params,
                    current_attack_modifier=attack_modifier,  # <-- 1. BIND THE ATTACK MODIFIER
                    current_modality=modality
            ):
                def modifier(config: AppConfig) -> AppConfig:
                    # Apply Default Defense HPs
                    for key, value in current_defense_params.items():
                        set_nested_attr(config, key, value)
                    print(f"    Applied Default Defense HPs for {current_defense_name}")

                    # Apply Attack State (ALWAYS "with_attack")
                    config.experiment.adv_rate = DEFAULT_ADV_RATE

                    # --- 2. USE THE BOUND VARIABLE ---
                    config = current_attack_modifier(config)  # Use the correct, bound attack

                    set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", DEFAULT_POISON_RATE)
                    print(f"    Applied Attack: {DEFAULT_ADV_RATE * 100}% rate, {DEFAULT_POISON_RATE * 100}% poison")

                    # Ensure Non-IID Data
                    set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                    set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)

                    # Set SkyMask type if needed
                    if "skymask" in current_defense_name:
                        model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                        set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                    # Turn off valuation
                    config.valuation.run_influence = False
                    config.valuation.run_loo = False
                    config.valuation.run_kernelshap = False
                    return config

                return modifier
            setup_modifier_func = create_setup_modifier_sens()

            # Base grid contains FIXED elements + LISTS for HP sweep
            grid = {
                target["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [target["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.dataset_type": [modality],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                # Training HPs to be swept by the main loop:
                "training.optimizer": OPTIMIZERS_TO_SWEEP,
                "training.local_epochs": LOCAL_EPOCHS_TO_SWEEP,
                "training.momentum": [0.0],
                "training.weight_decay": [0.0],
            }
            if "skymask" in defense_name:
                model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                grid["aggregation.skymask.sm_model_type"] = [model_struct]

            scenarios.append(Scenario(
                name=f"step2.5_find_hps_{defense_name}_{modality}_{target['dataset_name']}",
                base_config_factory=target["base_config_factory"],
                modifiers=[setup_modifier_func, target["dataset_modifier"]],
                parameter_grid=grid
            ))
    return scenarios


# --- Main Execution Block (copied from Step 4) ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step2.5_find_usable_hps"  # New directory
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_training_hp_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 2.5 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        optimizers = scenario.parameter_grid.get("training.optimizer", ["Adam"])
        epochs_list = scenario.parameter_grid.get("training.local_epochs", [1])
        static_grid = {k: v for k, v in scenario.parameter_grid.items() if k not in [
            "training.optimizer", "training.learning_rate", "training.local_epochs",
            "training.momentum", "training.weight_decay"
        ]}
        modality = scenario.parameter_grid["experiment.dataset_type"][0]

        for opt in optimizers:
            lrs = SGD_LRS_TO_SWEEP if opt == "SGD" else ADAM_LRS_TO_SWEEP
            for lr in lrs:
                for epochs in epochs_list:
                    current_grid = static_grid.copy()
                    current_grid["training.optimizer"] = [opt]
                    current_grid["training.learning_rate"] = [lr]
                    current_grid["training.local_epochs"] = [epochs]

                    if opt == "SGD":  # Use HP defaults for SGD
                        current_grid["training.momentum"] = [0.9]
                        current_grid["training.weight_decay"] = [5e-4]

                    hp_suffix = f"opt_{opt}_lr_{lr}_epochs_{epochs}"
                    unique_save_path = f"./results/{scenario.name}/{hp_suffix}"
                    current_grid["experiment.save_path"] = [unique_save_path]
                    temp_scenario_name = f"{scenario.name}/{hp_suffix}"

                    temp_scenario = Scenario(
                        name=temp_scenario_name,
                        base_config_factory=scenario.base_config_factory,
                        modifiers=[],
                        parameter_grid=current_grid
                    )

                    base_config = temp_scenario.base_config_factory()
                    modified_base_config = copy.deepcopy(base_config)
                    for modifier in scenario.modifiers:
                        modified_base_config = modifier(modified_base_config)
                    num_gen = generator.generate(modified_base_config, temp_scenario)
                    task_configs += num_gen

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 2.5 (Find Usable HPs) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"2. Analyze results using analyze_step4.py pointing to './results/'")
    print(f"3. Use this analysis to create a new, DEFENSE-SPECIFIC 'GOLDEN_TRAINING_PARAMS' dictionary.")
    print(f"4. Once that is done, you can *finally* run 'generate_step3_defense_tuning.py'.")