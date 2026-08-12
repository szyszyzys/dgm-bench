# FILE: generate_step3_defense_tuning.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports --- (Same as before)
from config_common_utils import (
    GOLDEN_TRAINING_PARAMS, NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES,
    NEW_IMAGE_DEFENSES, NEW_TEXT_TABULAR_DEFENSES,
    ENABLED_DATASETS, ENABLED_ATTACK_TYPES, filter_enabled_targets,
)
from experiments.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config
)
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_femnist_config, use_image_backdoor_attack, use_label_flipping_attack, use_cifar100_config,
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

# Reduced tuning grids: keep 1-2 values per HP. Defense HPs are not very
# sensitive — we just need each defense calibrated within a reasonable range.
TUNING_GRIDS = {
    "fltrust": {
        "aggregation.clip_norm": [5.0],
    },
    "martfl": {
        "aggregation.martfl.max_k": [3],
        "aggregation.clip_norm": [5.0],
    },
    "skymask": {
        "aggregation.skymask.mask_epochs": [20],
        "aggregation.skymask.mask_lr": [0.01, 0.1],
        "aggregation.skymask.mask_threshold": [0.5],
        "aggregation.clip_norm": [10],
        "aggregation.skymask.mask_clip": [1.0]
    },
    "skymask_small": {
        "aggregation.skymask.mask_epochs": [20],
        "aggregation.skymask.mask_lr": [0.01, 0.1],
        "aggregation.skymask.mask_threshold": [0.5],
        "aggregation.clip_norm": [10],
        "aggregation.skymask.mask_clip": [1.0]
    },
    "trimmed_mean": {
        "aggregation.trimmed_mean.trim_ratio": [0.1],
    },
    "multi_krum": {
        "aggregation.multi_krum.num_byzantine": [3],
        "aggregation.multi_krum.m_selected": [3],
    },
    "rflpa": {
        "aggregation.rflpa.element_clip": [1.0],
        "aggregation.clip_norm": [5.0],
    },
    "spmc": {
        "aggregation.spmc.temperature": [0.5],
    },
    "flame": {
        "aggregation.flame.noise_scale": [0.001],
        "aggregation.flame.min_cluster_size_ratio": [0.5],
    },
    "deepsight": {
        "aggregation.deepsight.threshold_c": [2.0],
        "aggregation.deepsight.neighbor_ratio": [0.5],
    },
    "foolsgold": {
        "aggregation.foolsgold.pardon_threshold": [0.5],
    },
    "bulyan": {
        "aggregation.bulyan.num_byzantine": [1],
    },
}

_ATTACK_TYPES_RAW = ["backdoor", "labelflip"]
# Filter against ENABLED_ATTACK_TYPES (single source of truth in
# config_common_utils.py). Currently only "backdoor" is enabled.
ATTACK_TYPES_TO_TUNE = [a for a in _ATTACK_TYPES_RAW if a in ENABLED_ATTACK_TYPES]
TUNING_TARGETS_STEP3 = [
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Texas100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_texas100_baseline",
     "dataset_modifier": lambda cfg: cfg,
     "backdoor_attack_modifier": use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL),
     "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config, "dataset_name": "Purchase100",
     "model_config_param_key": "experiment.tabular_model_config_name", "model_config_name": "mlp_purchase100_baseline",
     "dataset_modifier": lambda cfg: cfg,
     "backdoor_attack_modifier": use_tabular_backdoor_with_trigger(PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL),
     "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "FEMNIST",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "femnist_cnn",
     "dataset_modifier": use_femnist_config, "backdoor_attack_modifier": use_image_backdoor_attack,
     "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "image", "base_config_factory": get_base_image_config, "dataset_name": "CIFAR100",
     "model_config_param_key": "experiment.image_model_config_name", "model_config_name": "cifar100_cnn",
     "dataset_modifier": use_cifar100_config, "backdoor_attack_modifier": use_image_backdoor_attack,
     "labelflip_attack_modifier": use_label_flipping_attack},
    {"modality_name": "text", "base_config_factory": get_base_text_config, "dataset_name": "TREC",
     "model_config_param_key": "experiment.text_model_config_name", "model_config_name": "textcnn_trec_baseline",
     "dataset_modifier": use_trec_config, "backdoor_attack_modifier": use_text_backdoor_attack,
     "labelflip_attack_modifier": use_label_flipping_attack},
]

# Filter to ENABLED_DATASETS (single source of truth in config_common_utils.py).
_skipped_step3 = [t["dataset_name"] for t in TUNING_TARGETS_STEP3
                  if t["dataset_name"] not in ENABLED_DATASETS]
TUNING_TARGETS_STEP3 = filter_enabled_targets(TUNING_TARGETS_STEP3)
if _skipped_step3:
    print(f"[Step 3] Skipping disabled datasets: {_skipped_step3}")
    print(f"[Step 3] Enabled datasets        : {sorted(ENABLED_DATASETS)}")


# --- generate_defense_tuning_scenarios (Same as before) ---
def generate_defense_tuning_scenarios() -> List[Scenario]:
    """Generates configs to tune defense HPs under fixed attacks."""
    print("\n--- Generating Step 3: Defense Tuning Scenarios ---")
    scenarios = []
    for target in TUNING_TARGETS_STEP3:
        modality = target["modality_name"]
        model_cfg_name = target["model_config_name"]
        print(f"-- Processing: {modality} {model_cfg_name}")

        for attack_type in ATTACK_TYPES_TO_TUNE:
            attack_modifier_key = f"{attack_type}_attack_modifier"
            if attack_modifier_key not in target: continue
            attack_modifier = target[attack_modifier_key]
            print(f"  -- Attack Type: {attack_type}")

            base_defenses = IMAGE_DEFENSES if modality == "image" else TEXT_TABULAR_DEFENSES
            new_defenses = NEW_IMAGE_DEFENSES if modality == "image" else NEW_TEXT_TABULAR_DEFENSES
            current_defenses = base_defenses + new_defenses
            for defense_name in current_defenses:
                if defense_name == "fedavg": continue
                if defense_name not in TUNING_GRIDS: continue

                print(f"    - Defense: {defense_name}")

                def create_setup_modifier(
                        current_modifier=attack_modifier,
                        current_model_cfg_name=model_cfg_name,
                        current_defense_name=defense_name,  # <-- BIND THE DEFENSE NAME
                        current_modality=modality  # <-- BIND MODALITY to avoid late-binding
                ):
                    # Closure to capture the correct attack modifier AND defense
                    def modifier(config: AppConfig) -> AppConfig:

                        golden_hp_key = f"{current_model_cfg_name}"

                        training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)

                        if training_params:
                            for key, value in training_params.items():
                                set_nested_attr(config, key, value)
                        else:
                            print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                        config.experiment.adv_rate = DEFAULT_ADV_RATE
                        config = current_modifier(config)  # Sets attack type (backdoor/labelflip)
                        set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", DEFAULT_POISON_RATE)
                        config.adversary_seller_config.sybil.is_sybil = False

                        # Apply data and valuation settings
                        set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                        set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)
                        config.valuation.run_influence = False
                        config.valuation.run_loo = False
                        config.valuation.run_kernelshap = False
                        return config

                    return modifier

                setup_modifier_func = create_setup_modifier()

                defense_grid_to_sweep = TUNING_GRIDS[defense_name]

                base_grid = {
                    target["model_config_param_key"]: [model_cfg_name],
                    "experiment.dataset_name": [target["dataset_name"]],
                    "n_samples": [NUM_SEEDS_PER_CONFIG],
                    "aggregation.method": [defense_name],
                    "experiment.use_early_stopping": [True],
                    "experiment.patience": [10],
                    # Tuning only needs enough rounds to rank HPs, not full convergence.
                    # Reduces per-run time from ~5h to ~1.5h at 35s/round.
                    "experiment.global_rounds": [150],
                }
                if "skymask" in defense_name:
                    model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
                    base_grid["aggregation.skymask.sm_model_type"] = [model_struct]

                full_parameter_grid = {**base_grid, **defense_grid_to_sweep}

                scenarios.append(Scenario(
                    name=f"step3_tune_{defense_name}_{attack_type}_{modality}_{target['dataset_name']}_{model_cfg_name}",
                    base_config_factory=target["base_config_factory"],
                    modifiers=[setup_modifier_func, target["dataset_modifier"]],
                    parameter_grid=full_parameter_grid
                ))
    return scenarios


# --- Main Execution Block ---
if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step3_defense_tuning"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_defense_tuning_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 3 ---")

    # --- START OF MODIFICATION: Manual Loop ---
    # We will manually loop over the HPs to create unique names,
    # just like we do in the Step 4 script.

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Get the original base config and apply modifiers
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        # `iter_grid` (from config_generator) expands the grid into a list of dicts
        # Each dict is a single HP combination
        for hp_combo_dict in iter_grid(scenario.parameter_grid):

            # --- Build a unique name from the HPs that are being tuned ---
            hp_parts = []

            # Get the defense method for this scenario
            defense_name = hp_combo_dict.get("aggregation.method")

            # This logic MUST match your TUNING_GRIDS
            if defense_name == "fltrust":
                val = hp_combo_dict.get("aggregation.clip_norm", "None")
                hp_parts.append(f"aggregation.clip_norm_{val}")

            elif defense_name == "martfl":
                val = hp_combo_dict.get("aggregation.martfl.max_k", "None")
                hp_parts.append(f"aggregation.martfl.max_k_{val}")
                val = hp_combo_dict.get("aggregation.clip_norm", "None")
                hp_parts.append(f"aggregation.clip_norm_{val}")

            elif defense_name == "skymask":
                val = hp_combo_dict.get("aggregation.skymask.mask_epochs", "None")
                hp_parts.append(f"aggregation.skymask.mask_epochs_{val}")
                val = hp_combo_dict.get("aggregation.skymask.mask_lr", "None")
                hp_parts.append(f"aggregation.skymask.mask_lr_{val}")
                val = hp_combo_dict.get("aggregation.skymask.mask_threshold", "None")
                hp_parts.append(f"aggregation.skymask.mask_threshold_{val}")
                val = hp_combo_dict.get("aggregation.clip_norm", "None")
                hp_parts.append(f"aggregation.clip_norm_{val}")
                val = hp_combo_dict.get("aggregation.skymask.mask_clip", "None")
                hp_parts.append(f"aggregation.skymask.mask_clip_{val}")

            elif defense_name == "skymask_small":
                val = hp_combo_dict.get("aggregation.skymask.mask_epochs", "None")
                hp_parts.append(f"aggregation.skymask.mask_epochs_{val}")
                val = hp_combo_dict.get("aggregation.skymask.mask_lr", "None")
                hp_parts.append(f"aggregation.skymask.mask_lr_{val}")
                val = hp_combo_dict.get("aggregation.skymask.mask_threshold", "None")
                hp_parts.append(f"aggregation.skymask.mask_threshold_{val}")
                val = hp_combo_dict.get("aggregation.clip_norm", "None")
                hp_parts.append(f"aggregation.clip_norm_{val}")
                val = hp_combo_dict.get("aggregation.skymask.mask_clip", "None")
                hp_parts.append(f"aggregation.skymask.mask_clip_{val}")

            elif defense_name == "trimmed_mean":
                val = hp_combo_dict.get("aggregation.trimmed_mean.trim_ratio", "None")
                hp_parts.append(f"aggregation.trimmed_mean.trim_ratio_{val}")

            elif defense_name == "multi_krum":
                val = hp_combo_dict.get("aggregation.multi_krum.num_byzantine", "None")
                hp_parts.append(f"aggregation.multi_krum.num_byzantine_{val}")
                val = hp_combo_dict.get("aggregation.multi_krum.m_selected", "None")
                hp_parts.append(f"aggregation.multi_krum.m_selected_{val}")

            elif defense_name == "rflpa":
                val = hp_combo_dict.get("aggregation.rflpa.element_clip", "None")
                hp_parts.append(f"aggregation.rflpa.element_clip_{val}")
                val = hp_combo_dict.get("aggregation.clip_norm", "None")
                hp_parts.append(f"aggregation.clip_norm_{val}")

            elif defense_name == "spmc":
                val = hp_combo_dict.get("aggregation.spmc.temperature", "None")
                hp_parts.append(f"aggregation.spmc.temperature_{val}")

            elif defense_name == "flame":
                val = hp_combo_dict.get("aggregation.flame.noise_scale", "None")
                hp_parts.append(f"aggregation.flame.noise_scale_{val}")
                val = hp_combo_dict.get("aggregation.flame.min_cluster_size_ratio", "None")
                hp_parts.append(f"aggregation.flame.min_cluster_size_ratio_{val}")

            elif defense_name == "deepsight":
                val = hp_combo_dict.get("aggregation.deepsight.threshold_c", "None")
                hp_parts.append(f"aggregation.deepsight.threshold_c_{val}")
                val = hp_combo_dict.get("aggregation.deepsight.neighbor_ratio", "None")
                hp_parts.append(f"aggregation.deepsight.neighbor_ratio_{val}")

            elif defense_name == "foolsgold":
                val = hp_combo_dict.get("aggregation.foolsgold.pardon_threshold", "None")
                hp_parts.append(f"aggregation.foolsgold.pardon_threshold_{val}")

            elif defense_name == "bulyan":
                val = hp_combo_dict.get("aggregation.bulyan.num_byzantine", "None")
                hp_parts.append(f"aggregation.bulyan.num_byzantine_{val}")

            # Join parts to make the folder name
            hp_suffix = "_".join(hp_parts)
            if not hp_suffix:
                hp_suffix = "default_hps"  # Fallback

            # --- Create a new temporary scenario for this single config ---

            # Create a grid that has only *one* value for each parameter
            current_grid = {key: [value] for key, value in hp_combo_dict.items()}

            # Set the unique save path for the *results*
            # This is the path your run_parallel.py will use
            unique_save_path = f"./results/{scenario.name}/{hp_suffix}"
            current_grid["experiment.save_path"] = [unique_save_path]

            # Set the unique name for the *config file*
            temp_scenario_name = f"{scenario.name}/{hp_suffix}"

            temp_scenario = Scenario(
                name=temp_scenario_name,
                base_config_factory=scenario.base_config_factory,
                modifiers=[],  # Modifiers already applied to modified_base_config; don't run twice
                parameter_grid=current_grid  # This grid has no lists, only single values
            )

            # Generate the single config file
            # We pass modified_base_config so modifiers aren't run again
            num_gen = generator.generate(modified_base_config, temp_scenario)
            task_configs += num_gen

        # --- END OF MODIFICATION ---

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 3 (Defense Tuning) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS in config_common_utils.py is correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(
        f"3. Analyze results using step3_analyze.py pointing to './results/'")
    print("4. Find the best defense HPs (good Acc, low ASR) for each defense/attack_type/model/dataset combo.")
    print(
        "5. Record these winning HPs -> TUNED_DEFENSE_PARAMS in config_common_utils.py")
