# FILE: generate_step12_main_summary.py

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG as _GLOBAL_NUM_SEEDS,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
    IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES,
    # create_fixed_params_modifier,  <-- REMOVED
    enable_valuation, get_tuned_defense_params,
    GOLDEN_TRAINING_PARAMS,  # <-- ADDED
    ENABLED_DATASETS, filter_enabled_targets,
)

# Step 10 is the headline multi-modal benchmark and the table reviewers will
# scrutinize most. Override the global default (2) to 3 seeds so means and
# error bars are statistically defensible. The per-cell .success skip logic
# in run_parallel_experiment.py means re-running this step after a seed bump
# only executes the new seed_44 cells; the existing seed_42 + seed_43 cells
# are preserved as-is.
NUM_SEEDS_PER_CONFIG = max(3, _GLOBAL_NUM_SEEDS)
from experiments.gradient_market.automate_exp.base_configs import (
    get_base_image_config, get_base_text_config
)
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_femnist_config, use_cifar100_config, use_trec_config, use_image_backdoor_attack,
    use_text_backdoor_attack
)
from experiments.gradient_market.automate_exp import TEXAS100_TRIGGER, use_tabular_backdoor_with_trigger, \
    TEXAS100_TARGET_LABEL, get_base_tabular_config, PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

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
_skipped_step10 = [t["dataset_name"] for t in MAIN_SUMMARY_TARGETS
                   if t["dataset_name"] not in ENABLED_DATASETS]
MAIN_SUMMARY_TARGETS = filter_enabled_targets(MAIN_SUMMARY_TARGETS)
if _skipped_step10:
    print(f"[Step 10] Skipping disabled datasets: {_skipped_step10}")
    print(f"[Step 10] Enabled datasets        : {sorted(ENABLED_DATASETS)}")


def generate_main_summary_scenarios() -> List[Scenario]:
    """Generates the main benchmark comparison configs with valuation."""
    print("\n--- Generating Step 12: Main Summary Scenarios (with Valuation) ---")
    scenarios = []
    for target in MAIN_SUMMARY_TARGETS:
        modality = target["modality_name"]
        model_cfg_name = target["model_config_name"]
        print(f"-- Processing: {modality} {model_cfg_name}")

        current_defenses = IMAGE_DEFENSES if modality == "image" else TEXT_TABULAR_DEFENSES

        for defense_name in current_defenses:

            # Bulyan parameter constraint: Bulyan needs N >= 4f + 3 active
            # sellers per round to satisfy its byzantine-tolerance bound. At
            # f=1 (the tuned step-3 value) this means N >= 7.
            #
            # In practice, the *active* seller count per round drops below 7
            # on three of our datasets:
            #
            #   - FEMNIST  (natural partitioning by writer): each writer has
            #               a small handful of samples, so empty Dirichlet-
            #               equivalent shards happen often and the active-N
            #               per round is typically 3-5.
            #   - Texas100 (tabular, dirichlet α=0.5): small per-shard sizes
            #               cause similar empty-shard drops; active N is 5-7.
            #   - Purchase100: same story as Texas100, active N typically 5-7.
            #
            # When N falls below 7, Bulyan auto-clamps f → 0 and silently
            # degrades to coordinate-wise median across all submitted updates,
            # which is not real Bulyan. We saw ~895 "Bulyan: requested f=1
            # needs N>=7" warnings in the step 12 runner log when Bulyan was
            # allowed to run on these datasets in an earlier rerun.
            #
            # The methodologically honest move is to skip Bulyan on the three
            # datasets where the constraint is structurally unsatisfiable, and
            # footnote the paper to scope Bulyan results to CIFAR-100 only
            # (where active N is consistently 7-9 with our setup).
            if defense_name == "bulyan" and target["dataset_name"] in (
                "FEMNIST", "Texas100", "Purchase100"
            ):
                print(f"  SKIPPING bulyan × {target['dataset_name']}: "
                      f"N >= 4f+3 = 7 not satisfiable per round at n_sellers=10. "
                      f"Bulyan is reported only on CIFAR-100 in the headline table.")
                continue

            # Get Tuned HPs (from Step 3)
            tuned_defense_params = get_tuned_defense_params(
                defense_name=defense_name,
                model_config_name=model_cfg_name,
                attack_state="with_attack",  # Use default
                default_attack_type_for_tuning="backdoor"
            )
            print(f"  - Defense: {defense_name}")
            if not tuned_defense_params:
                # CHANGED: previously this `continue`'d and silently dropped the
                # defense from the headline table. That's how FLAME/DeepSight/
                # Bulyan/FoolsGold went missing from step 10 — they had no
                # tuned-params entry for one or more (defense, model_config)
                # pairs and the whole row got skipped. We now fall back to an
                # empty dict (= defense built-in defaults) so the defense is
                # still represented in the main summary.
                print(f"  WARN: No Tuned HPs found for {defense_name}/{model_cfg_name} "
                      f"— falling back to defense defaults so the defense still "
                      f"appears in the main summary.")
                tuned_defense_params = {}

            def create_setup_modifier(
                    current_defense_name=defense_name,
                    current_model_cfg_name=model_cfg_name,
                    current_tuned_params=tuned_defense_params,
                    current_modality=modality
            ):
                def modifier(config: AppConfig) -> AppConfig:
                    # 1. Apply Golden Training HPs (from Step 2.5)
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

                    # 2. Apply the defense aggregation method itself.
                    # Previously this was implicit through `current_tuned_params`
                    # which always had aggregation.method baked in. With the
                    # missing-defense fallback ({} = use defaults) we need to
                    # set aggregation.method explicitly, otherwise the base
                    # config's default would silently be used and the row would
                    # be a duplicate of fedavg.
                    set_nested_attr(config, "aggregation.method", current_defense_name)

                    # 3. Apply Tuned Defense HPs (from Step 3)
                    for key, value in current_tuned_params.items():
                        set_nested_attr(config, key, value)

                    # 3. Apply other fixed settings.
                    # IMPORTANT: do NOT clobber the partitioning strategy that
                    # the dataset modifier (e.g. use_femnist_config) already
                    # set. Previously this hard-coded
                    #     strategy = "dirichlet", dirichlet_alpha = 0.5
                    # for every dataset, which silently overrode FEMNIST's
                    # writer-based natural partitioning. We now only force
                    # dirichlet for non-FEMNIST datasets, so FEMNIST keeps
                    # the "natural" strategy set by use_femnist_config.
                    if config.experiment.dataset_name != "FEMNIST":
                        set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                        set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)
                    return config

                return modifier

            setup_modifier_func = create_setup_modifier()

            # NOTE: VALUATION DECOUPLED FROM MAIN SUMMARY (Step 10).
            #
            # Step 10 reports the headline marketplace metrics (Acc, ASR, BSR,
            # MSR) only. Per-seller valuation analysis (KernelSHAP, LOO,
            # Influence) has been moved to a dedicated step that runs on a
            # single representative dataset — see
            #     experiments/gradient_market/configs_generation/generate_step21_valuation_analysis.py
            # and the §5 valuation subsection of the paper.
            #
            # Why this split:
            #   1. The two analyses answer different RQs.
            #      - Step 10 measures aggregate defense quality (per-cell
            #        Acc/ASR/BSR/MSR), which the marketplace itself computes
            #        and does NOT depend on the valuation module at all.
            #      - Step 21 measures per-seller value attribution
            #        (KernelSHAP / LOO / Influence), which is dataset-agnostic
            #        and only needs a single representative dataset to
            #        validate.
            #   2. KernelSHAP is the dominant cost in step 10 cells. Earlier
            #      attempts to reduce its cost by sample-count tuning
            #      (v1: 500 samples → v2: 100 samples → v3: 50 samples ×
            #      half-frequency) all kept hitting CPU-bound pathological
            #      paths on clustering defenses (DeepSight, FoolsGold), which
            #      caused 30+ minute stalls per cell. Removing KernelSHAP
            #      entirely from step 10 gives a 5-10× speedup and eliminates
            #      the stall failure mode.
            #   3. Headline metrics are unaffected. Acc/ASR/BSR/MSR come from
            #      marketplace_report.json, not from the valuation module.
            #
            # If you need to re-enable KernelSHAP / LOO / Influence in step 10
            # for any reason (e.g., a reviewer asks), set the relevant flag
            # below to True. The original v3 settings were:
            #     influence=True, loo=True, loo_freq=5,
            #     kernelshap=True, kshap_freq=10, kshap_samples=50
            valuation_modifier = lambda config: enable_valuation(
                config,
                influence=False,
                loo=False,
                kernelshap=False,
            )

            scenario_name = f"step12_main_summary_{defense_name}_{modality}_{target['dataset_name']}_{model_cfg_name.split('_')[-1]}"
            unique_save_path = f"./results/{scenario_name}"

            grid = {
                target["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [target["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.adv_rate": [FIXED_ATTACK_ADV_RATE],
                "adversary_seller_config.poisoning.poison_rate": [FIXED_ATTACK_POISON_RATE],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                "experiment.save_path": [unique_save_path]  # <-- ADDED
            }

            all_modifiers = [
                setup_modifier_func,
                target["dataset_modifier"],
                target["attack_modifier"],  # <-- THE FIX
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
    output_dir = Path(base_output_dir) / "step12_main_summary"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_main_summary_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 12 ---")

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

    print(f"\n✅ Step 12 (Main Summary) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"Configs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"1. CRITICAL: Ensure GOLDEN_TRAINING_PARAMS & TUNED_DEFENSE_PARAMS are correct.")
    print(f"2. Run experiments: python run_parallel.py --configs_dir {output_dir}")
    print(f"3. Analyze results: Create summary tables/plots.")
    print(f"4. Analyze valuation results: Compare Influence/LOO/KernelSHAP scores.")
