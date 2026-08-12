# FILE: generate_step6_advanced_sybil.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports (Assuming these are available in your environment) ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE, enable_valuation, use_sybil_attack_strategy,
    get_tuned_defense_params, GOLDEN_TRAINING_PARAMS, IMAGE_DEFENSES, FOCUSED_DEFENSES,
    ENABLED_DATASETS,
)

# Step 5 (sybil strategies) needs at least one clustering / similarity-based
# defense in addition to the FOCUSED trust-based ones. FoolsGold was
# specifically designed to detect sybils via historical cosine similarity
# (Fung et al., 2018) and DeepSight uses neuron-level clustering — both are
# the canonical sybil-detection defenses and the natural baselines reviewers
# expect in any sybil benchmark. Their tuned HPs already exist in
# tuned_defense_params.json, so re-running this step will only execute the
# new (foolsgold, deepsight) × strategy cells; existing fltrust + martfl
# cells are skipped via the .success marker.
SYBIL_FOCUS_DEFENSES = ["fltrust", "martfl", "foolsgold", "deepsight"]
# NOTE: AppConfig and set_nested_attr must be available from these imports
from experiments.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config,
)
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_image_backdoor_attack, use_cifar100_config,
    use_femnist_config, use_trec_config, use_text_backdoor_attack,
)
from experiments.gradient_market.automate_exp import (
    get_base_tabular_config, use_tabular_backdoor_with_trigger,
    TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL,
    PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL,
)

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)
# --- End Imports ---

# --- Sybil Strategies & Parameters to Test ---
SYBIL_TEST_CONFIG = {
    # Reduced: dropped knock_out (it's literally mimic with 2x alpha — redundant)
    "baseline_no_sybil": None,
    "mimic": {},
    "pivot": {},
    "oracle_blend": {"blend_alpha": [0.1, 0.5, 0.9]},  # 3 alphas covers low/mid/high
}

# --- Per-dataset Setups for Advanced Sybil Analysis ---
# Originally Step 5 was hardcoded to CIFAR100. To produce all-dataset results
# for the main paper, we now enumerate the same 5 (modality, dataset, model)
# triples Step 3 / Step 10 use, then filter by ENABLED_DATASETS so the global
# gate in config_common_utils.py controls scope.
SYBIL_SETUPS_ALL = [
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config,
     "dataset_name": "Texas100",
     "model_config_param_key": "experiment.tabular_model_config_name",
     "model_config_name": "mlp_texas100_baseline",
     "dataset_modifier": (lambda cfg: cfg),
     "attack_modifier": use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL)},
    {"modality_name": "tabular", "base_config_factory": get_base_tabular_config,
     "dataset_name": "Purchase100",
     "model_config_param_key": "experiment.tabular_model_config_name",
     "model_config_name": "mlp_purchase100_baseline",
     "dataset_modifier": (lambda cfg: cfg),
     "attack_modifier": use_tabular_backdoor_with_trigger(PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL)},
    {"modality_name": "image", "base_config_factory": get_base_image_config,
     "dataset_name": "FEMNIST",
     "model_config_param_key": "experiment.image_model_config_name",
     "model_config_name": "femnist_cnn",
     "dataset_modifier": use_femnist_config,
     "attack_modifier": use_image_backdoor_attack},
    {"modality_name": "image", "base_config_factory": get_base_image_config,
     "dataset_name": "CIFAR100",
     "model_config_param_key": "experiment.image_model_config_name",
     "model_config_name": "cifar100_cnn",
     "dataset_modifier": use_cifar100_config,
     "attack_modifier": use_image_backdoor_attack},
    {"modality_name": "text", "base_config_factory": get_base_text_config,
     "dataset_name": "TREC",
     "model_config_param_key": "experiment.text_model_config_name",
     "model_config_name": "textcnn_trec_baseline",
     "dataset_modifier": use_trec_config,
     "attack_modifier": use_text_backdoor_attack},
]
# NOTE: Step 5 is intentionally restricted to CIFAR100 to save compute. The
# multi-dataset machinery above (SYBIL_SETUPS_ALL + per-setup loop) is kept
# in place so the step can be expanded later without another refactor — just
# change this line to `filter_enabled_targets(SYBIL_SETUPS_ALL)`.
SYBIL_SETUPS = [s for s in SYBIL_SETUPS_ALL if s["dataset_name"] == "CIFAR100"]
assert SYBIL_SETUPS, "Step 5 setups left empty — CIFAR100 missing from SYBIL_SETUPS_ALL?"

# Backward-compat shim: legacy code paths below previously referenced the
# single-dataset SYBIL_SETUP. We keep a CIFAR100 default available so any
# stragglers don't crash, but the main loop iterates SYBIL_SETUPS.
SYBIL_SETUP = next((s for s in SYBIL_SETUPS if s["dataset_name"] == "CIFAR100"),
                   SYBIL_SETUPS[0])

# --- Fixed Attack Parameters (Strength) ---
FIXED_ATTACK_ADV_RATE = DEFAULT_ADV_RATE
FIXED_ATTACK_POISON_RATE = DEFAULT_POISON_RATE


# === FUNCTION TO GENERATE BASE SCENARIOS ===
def generate_advanced_sybil_scenarios() -> List[Scenario]:
    """Generates base scenarios for comparing different Sybil strategies
    across all enabled (modality, dataset, model) setups.

    Backward compat: CIFAR100 scenario names are kept identical to the
    pre-refactor format ("step6_adv_sybil_{strategy}_{defense}") so existing
    .success markers are reused. Other datasets get a "_{dataset}" suffix.
    """
    print("\n--- Generating Step 6: Advanced Sybil Comparison Scenarios ---")
    scenarios = []

    for setup in SYBIL_SETUPS:
        modality = setup["modality_name"]
        model_cfg_name = setup["model_config_name"]
        dataset_name = setup["dataset_name"]
        print(f"\n=== Setup: {dataset_name} / {model_cfg_name} ({modality}) ===")

        # FOCUSED: trust-based (FLTrust, MartFL) + clustering/similarity-based
        # (FoolsGold, DeepSight).
        for defense_name in SYBIL_FOCUS_DEFENSES:
            # 2. Get Tuned HPs (from Step 3)
            tuned_defense_params = get_tuned_defense_params(
                defense_name=defense_name,
                model_config_name=model_cfg_name,
                attack_state='with_attack',
                default_attack_type_for_tuning="backdoor"
            )
            print(f"-- Processing Defense: {defense_name}")

            # 3. Create the setup modifier INSIDE the loop
            def create_setup_modifier(
                    current_defense_name=defense_name,
                    current_model_cfg_name=model_cfg_name,
                    current_tuned_params=tuned_defense_params,
                    current_modality=modality,
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
                    if current_tuned_params:
                        for key, value in current_tuned_params.items():
                            set_nested_attr(config, key, value)
                    else:
                        print(f"  WARNING: No Tuned HPs found for {current_defense_name} on {current_model_cfg_name}!")

                    if "skymask" in current_defense_name:
                        model_struct = "resnet18" if "resnet" in current_model_cfg_name else "flexiblecnn"
                        set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)

                    # --- Apply other fixed settings ---
                    set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                    set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)
                    return config

                return modifier

            setup_modifier_func = create_setup_modifier()

            # Base Grid (fixed parts) — uses this setup's model key + dataset
            base_grid = {
                setup["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [dataset_name],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.adv_rate": [FIXED_ATTACK_ADV_RATE],
                "adversary_seller_config.poisoning.poison_rate": [FIXED_ATTACK_POISON_RATE],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
            }

            # Loop through Sybil strategies
            for strategy_name, strategy_params_sweep in SYBIL_TEST_CONFIG.items():
                print(f"  - Strategy: {strategy_name}")
                # Backward-compat naming: keep CIFAR100 names unchanged so
                # prior .success markers are reused; suffix others with dataset.
                if dataset_name == "CIFAR100":
                    scenario_name = f"step6_adv_sybil_{strategy_name}_{defense_name}"
                else:
                    scenario_name = f"step6_adv_sybil_{strategy_name}_{defense_name}_{dataset_name}"

                current_grid = base_grid.copy()
                current_grid["_strategy_name"] = [strategy_name]
                if strategy_params_sweep:
                    current_grid["_sweep_params"] = [strategy_params_sweep]
                else:
                    current_grid["_sweep_params"] = [None]

                current_modifiers = [
                    setup_modifier_func,
                    setup["dataset_modifier"],
                    setup["attack_modifier"],
                    lambda config: enable_valuation(
                        config,
                        influence=True,
                        loo=True,
                        loo_freq=10,
                        kernelshap=False
                    )
                ]

                scenario = Scenario(
                    name=scenario_name,
                    base_config_factory=setup["base_config_factory"],
                    modifiers=current_modifiers,
                    parameter_grid=current_grid
                )
                scenarios.append(scenario)

    return scenarios

if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step6_advanced_sybil"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_advanced_sybil_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 6 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario base: {scenario.name}")
        task_configs = 0

        # Filter out metadata keys starting with "_"
        static_grid = {k: v for k, v in scenario.parameter_grid.items() if not k.startswith("_")}
        strategy_name = scenario.parameter_grid["_strategy_name"][0]
        sweep_params = scenario.parameter_grid["_sweep_params"][0]
        base_hp_suffix = f"adv_{FIXED_ATTACK_ADV_RATE}_poison_{FIXED_ATTACK_POISON_RATE}"

        if sweep_params:
            sweep_key, sweep_values = next(iter(sweep_params.items()))  # e.g., 'blend_alpha', list_of_values

            for sweep_value in sweep_values:
                current_grid = static_grid.copy()
                hp_suffix = f"{base_hp_suffix}_{sweep_key}_{sweep_value}"

                # **CRITICAL FIX:** Pass the sweep parameter as kwargs to the sybil modifier
                strategy_kwargs = {sweep_key: sweep_value}

                temp_scenario = Scenario(
                    name=f"{scenario.name}/{hp_suffix}",
                    base_config_factory=scenario.base_config_factory,
                    # Combine existing modifiers AND the Sybil modifier with the specific parameter
                    modifiers=[],
                    parameter_grid=current_grid
                )
                temp_scenario.parameter_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

                base_config = temp_scenario.base_config_factory()
                modified_base_config = copy.deepcopy(base_config)

                # Run all modifiers, which now include the Sybil modifier that sets the blend_alpha
                for modifier in scenario.modifiers + [use_sybil_attack_strategy(strategy=strategy_name, **strategy_kwargs)]:
                    modified_base_config = modifier(modified_base_config)

                print(f"    - Setting Sybil strategy '{strategy_name}' with {sweep_key}: {sweep_value}")

                # Use the modified_base_config for generation
                num_gen = generator.generate(modified_base_config, temp_scenario)
                task_configs += num_gen

        else:
            # Logic for non-sweeping strategies (e.g., mimic, baseline)
            current_grid = static_grid.copy()
            hp_suffix = base_hp_suffix

            temp_scenario = Scenario(
                name=f"{scenario.name}/{hp_suffix}",
                base_config_factory=scenario.base_config_factory,
                modifiers=[],
                parameter_grid=current_grid
            )
            temp_scenario.parameter_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

            if strategy_name == "baseline_no_sybil":
                temp_scenario.parameter_grid["adversary_seller_config.sybil.is_sybil"] = [False]

            base_config = temp_scenario.base_config_factory()
            modified_base_config = copy.deepcopy(base_config)
            all_modifiers = list(scenario.modifiers)
            if strategy_name != "baseline_no_sybil":
                # For non-sweeping sybil strategies (e.g., mimic, pivot)
                all_modifiers.append(use_sybil_attack_strategy(strategy=strategy_name))
            for modifier in all_modifiers:
                modified_base_config = modifier(modified_base_config)

            num_gen = generator.generate(modified_base_config, temp_scenario)
            task_configs += num_gen

        print(f"-> Generated {task_configs} configs for {scenario.name} base")
        all_generated_configs += task_configs

    print(f"\n✅ Step 6 (Advanced Sybil Comparison) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
