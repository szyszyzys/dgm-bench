# FILE: generate_step21_valuation_analysis.py
# =============================================================================
# Step 21 — Per-seller valuation analysis (KernelSHAP / LOO / Influence)
# =============================================================================
# Purpose
# -------
# A dedicated valuation step decoupled from the main summary (Step 10). Step 10
# now reports only the marketplace's headline metrics (Acc, ASR, BSR, MSR) with
# all valuation methods disabled, because KernelSHAP was the dominant cost in
# step 10 cells AND was hitting CPU-bound pathological paths on clustering
# defenses, causing 30+ minute stalls.
#
# Step 21 runs the same training pipeline but with all three valuation methods
# enabled, on a much narrower scope so the cost is bounded:
#
#   - Single dataset: CIFAR-100 (representative; valuation properties are
#     largely dataset-agnostic, so a single-dataset analysis supports the
#     paper's per-seller value-attribution claims)
#   - Focused defense set: 4 defenses spanning the key paradigms
#       fedavg     — no-defense baseline
#       fltrust    — trust-weighted
#       martfl     — clustering / trusted-set
#       foolsgold  — historical similarity / sybil detection
#   - Two attack conditions:
#       no-attack  — clean baseline (validates that valuation methods identify
#                    high-data-quality sellers even without an attack)
#       backdoor   — standard adv_rate=0.3, poison_rate=0.5 (validates that
#                    valuation methods identify adversarial sellers under attack)
#   - 2 seeds per cell
#
# Total expected cells: 4 defenses × 2 attacks × 2 seeds = 16 cells.
# Estimated runtime: ~5-7 hours on 3 GPUs with full KernelSHAP (kshap_samples=500).
#
# What the paper does with this data
# ----------------------------------
# The output of step 21 is the data backing for the paper's per-seller value-
# attribution figures. Specifically:
#
#   Figure A (per-defense value distribution):
#       For each defense, plot the distribution of per-seller value scores
#       (KernelSHAP-derived) for honest vs. adversarial sellers. The claim:
#       "filtering defenses produce value distributions that cleanly separate
#       honest from adversarial sellers, while non-filtering defenses do not."
#
#   Figure B (cross-method agreement):
#       For one or two defenses, compare the per-seller rankings produced by
#       KernelSHAP vs. LOO vs. Influence. The claim: "the three valuation
#       methods agree on the top-K honest sellers, validating that the
#       marketplace's payment ranking is robust to the choice of attribution
#       method."
#
# These figures support the paper's per-seller fairness claims independently
# of the main summary's aggregate metrics.
#
# Note on KernelSHAP cost
# -----------------------
# KernelSHAP runs at full resolution here (kshap_samples=500, kshap_freq=5)
# because we deliberately scoped the cell count down so we can afford it.
# With 16 cells and 3 GPU workers, total compute is ~5-7 hours (a manageable
# overnight run), vs. the >12 hours that step 10 was hitting before the
# valuation split.
# =============================================================================

import copy
import sys
from pathlib import Path
from typing import List

# --- Imports ---
from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
    enable_valuation, get_tuned_defense_params,
    GOLDEN_TRAINING_PARAMS,
)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_cifar100_config, use_image_backdoor_attack,
)

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)


# =============================================================================
# Scope (intentionally narrow — see file header for the rationale)
# =============================================================================
VALUATION_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
    "attack_modifier": use_image_backdoor_attack,
}

# Defenses to evaluate. One representative per defense paradigm:
#   - fedavg     : baseline, no filtering
#   - fltrust    : trust-weighted (per-seller trust score)
#   - martfl     : clustering / trusted-set selection
#   - foolsgold  : historical cosine similarity (sybil detection)
#
# To expand later (e.g., for a broader appendix figure), append more defense
# names here. Each additional defense adds 2 attack conditions × 2 seeds = 4
# cells, ~1.5 hours on 3 GPUs.
VALUATION_DEFENSES = ["fedavg", "fltrust", "martfl", "foolsgold"]

# Attack conditions: clean baseline + standard backdoor.
# A clean baseline is important for the paper's "valuation correctly identifies
# high-quality honest sellers" claim — without it, you can only show the
# attacked case.
VALUATION_ATTACK_CONDITIONS = [
    {
        "name": "no_attack",
        "adv_rate": 0.0,
        "poison_rate": 0.0,
        "use_attack_modifier": False,
    },
    {
        "name": "backdoor",
        "adv_rate": DEFAULT_ADV_RATE,
        "poison_rate": DEFAULT_POISON_RATE,
        "use_attack_modifier": True,
    },
]


def generate_valuation_analysis_scenarios() -> List[Scenario]:
    """Generates per-seller valuation scenarios on CIFAR-100 with full
    KernelSHAP / LOO / Influence enabled."""
    print("\n--- Generating Step 21: Per-seller valuation analysis ---")
    scenarios = []

    modality = VALUATION_SETUP["modality_name"]
    model_cfg_name = VALUATION_SETUP["model_config_name"]
    dataset_name = VALUATION_SETUP["dataset_name"]

    for defense_name in VALUATION_DEFENSES:
        # Get tuned defense HPs from Step 3 (fall back to defaults if missing)
        tuned_defense_params = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=model_cfg_name,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor",
        )
        if not tuned_defense_params:
            print(f"  WARN: No Tuned HPs found for {defense_name}/{model_cfg_name}"
                  f" — falling back to defense built-in defaults.")
            tuned_defense_params = {}

        for attack_cfg in VALUATION_ATTACK_CONDITIONS:
            attack_name = attack_cfg["name"]
            adv_rate = attack_cfg["adv_rate"]
            poison_rate = attack_cfg["poison_rate"]
            use_attack = attack_cfg["use_attack_modifier"]

            print(f"  Defense: {defense_name}  /  attack: {attack_name}  "
                  f"(adv={adv_rate}, poison={poison_rate})")

            def create_setup_modifier(
                    current_defense_name=defense_name,
                    current_model_cfg_name=model_cfg_name,
                    current_tuned_params=tuned_defense_params,
                    current_modality=modality,
                    current_adv_rate=adv_rate,
                    current_poison_rate=poison_rate,
                    current_use_attack=use_attack,
            ):
                def modifier(config: AppConfig) -> AppConfig:
                    # 1. Apply Golden Training HPs (from Step 1)
                    golden_hp_key = f"{current_model_cfg_name}"
                    training_params = GOLDEN_TRAINING_PARAMS.get(golden_hp_key)
                    if training_params:
                        for key, value in training_params.items():
                            set_nested_attr(config, key, value)
                    else:
                        print(f"  WARNING: No Golden HPs found for key '{golden_hp_key}'!")

                    # 2. Apply the defense aggregation method itself
                    set_nested_attr(config, "aggregation.method", current_defense_name)
                    if "skymask" in current_defense_name:
                        # Reserved in case skymask gets added to VALUATION_DEFENSES later
                        model_struct = ("resnet18" if "resnet" in current_model_cfg_name
                                        else "flexiblecnn")
                        set_nested_attr(config, "aggregation.skymask.sm_model_type",
                                        model_struct)

                    # 3. Apply Tuned Defense HPs (from Step 3)
                    for key, value in current_tuned_params.items():
                        set_nested_attr(config, key, value)

                    # 4. Data partitioning (CIFAR-100 always uses dirichlet 0.5)
                    set_nested_attr(config, f"data.{current_modality}.strategy", "dirichlet")
                    set_nested_attr(config, f"data.{current_modality}.dirichlet_alpha", 0.5)

                    # 5. Attack configuration (this is the per-condition difference)
                    if current_use_attack:
                        set_nested_attr(config, "experiment.adv_rate", current_adv_rate)
                        set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate",
                                        current_poison_rate)
                    else:
                        # No-attack baseline: zero out adversaries entirely
                        set_nested_attr(config, "experiment.adv_rate", 0.0)
                        config.adversary_seller_config.poisoning.type = PoisonType.NONE

                    return config

                return modifier

            setup_modifier_func = create_setup_modifier()

            # Full-resolution valuation: this is the WHOLE POINT of Step 21.
            # KernelSHAP at 500 samples is the original v1 setting from before
            # the step-10 speed knob; we can afford it here because the cell
            # count is small (16 cells total).
            valuation_modifier = lambda config: enable_valuation(
                config,
                influence=True,
                loo=True, loo_freq=5,
                kernelshap=True, kshap_freq=5,
                kshap_samples=500,
            )

            scenario_name = (
                f"step21_valuation_{defense_name}_{attack_name}_{dataset_name}"
            )
            unique_save_path = f"./results/{scenario_name}"

            grid = {
                VALUATION_SETUP["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [dataset_name],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                "experiment.save_path": [unique_save_path],
            }

            # Build the modifier list — only include the attack modifier when
            # the attack condition actually wants it
            modifiers = [
                setup_modifier_func,
                VALUATION_SETUP["dataset_modifier"],
            ]
            if use_attack:
                modifiers.append(VALUATION_SETUP["attack_modifier"])
            modifiers.append(valuation_modifier)

            scenario = Scenario(
                name=scenario_name,
                base_config_factory=VALUATION_SETUP["base_config_factory"],
                modifiers=modifiers,
                parameter_grid=grid,
            )
            scenarios.append(scenario)

    return scenarios


if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step21_valuation_analysis"
    generator = ExperimentGenerator(str(output_dir))

    scenarios_to_generate = generate_valuation_analysis_scenarios()
    all_generated_configs = 0

    print("\n--- Generating Configuration Files for Step 21 ---")

    for scenario in scenarios_to_generate:
        print(f"\nProcessing scenario: {scenario.name}")
        base_config = scenario.base_config_factory()
        modified_base_config = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified_base_config = modifier(modified_base_config)

        temp_scenario = Scenario(
            name=scenario.name,
            base_config_factory=scenario.base_config_factory,
            modifiers=[],
            parameter_grid=scenario.parameter_grid,
        )
        num_gen = generator.generate(modified_base_config, temp_scenario)
        all_generated_configs += num_gen
        print(f"-> Generated {num_gen} configs for {scenario.name}")

    print(f"\n✅ Step 21 (Per-seller Valuation Analysis) config generation complete!")
    print(f"Total configurations generated: {all_generated_configs}")
    print(f"  Defenses              : {VALUATION_DEFENSES}")
    print(f"  Dataset               : {VALUATION_SETUP['dataset_name']}")
    print(f"  Attack conditions     : {[a['name'] for a in VALUATION_ATTACK_CONDITIONS]}")
    print(f"  Seeds per cell        : {NUM_SEEDS_PER_CONFIG}")
    print(f"  Configs saved to      : {output_dir}")
