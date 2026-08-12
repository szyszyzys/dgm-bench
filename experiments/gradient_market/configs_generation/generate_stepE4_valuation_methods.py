# =============================================================================
# Step E4 — Valuation-gap robustness across solution concepts (reviewer R1.O3).
#
# Re-runs the Fig-7 valuation-gap analysis with Banzhaf (maximum-sample-reuse
# estimator) and Least Core (Monte-Carlo constraint-sampled LP) alongside the
# existing KernelSHAP / LOO / Influence engines. Banzhaf and Least Core REUSE
# the coalition/utility samples already drawn for KernelSHAP, so enabling all
# five methods in ONE run costs the same utility evaluations as KernelSHAP
# alone — and guarantees every method scores the identical rounds.
#
# Run (from experiments/gradient_market/configs_generation/):
#   python generate_stepE4_valuation_methods.py --valuation all
#   python ../run_parallel_experiment.py \
#       --configs_dir ./configs_generated_benchmark/stepE4_valuation_methods \
#       --gpu_ids 0,1,2,3,4 --num_processes 5
#   python ../visualization/extract/extract_stepE4_valuation_gap.py
#
# --valuation accepts a comma-separated subset of
#   {kernelshap, loo, influence, banzhaf, leastcore}  or  "all" (default).
# =============================================================================

import argparse
import copy
import sys
from pathlib import Path
from typing import List

from config_common_utils import (
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
    get_tuned_defense_params,
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

MODEL_CONFIG_NAME = "cifar100_cnn"
DATASET_NAME = "CIFAR100"
VALID_METHODS = {"kernelshap", "loo", "influence", "banzhaf", "leastcore"}
# Same defense scope as the original step-21 valuation analysis.
E4_DEFENSES = ["fedavg", "fltrust", "martfl", "foolsgold"]
E4_ATTACK_CONDITIONS = [
    {"name": "no_attack", "adv_rate": 0.0, "poison_rate": 0.0, "use_attack_modifier": False},
    {"name": "backdoor", "adv_rate": DEFAULT_ADV_RATE, "poison_rate": DEFAULT_POISON_RATE,
     "use_attack_modifier": True},
]
NUM_SEEDS = 5
KSHAP_SAMPLES = 500
KSHAP_FREQ = 5
LEASTCORE_MAX_COALITIONS = 2000
LEASTCORE_TIMEOUT = 600.0


def generate_e4_scenarios(methods) -> List[Scenario]:
    print(f"\n--- Generating Step E4: valuation methods {sorted(methods)} ---")
    scenarios = []

    for defense_name in E4_DEFENSES:
        tuned = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=MODEL_CONFIG_NAME,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor",
        ) or {}

        for attack_cfg in E4_ATTACK_CONDITIONS:
            use_attack = attack_cfg["use_attack_modifier"]

            def setup_modifier(config: AppConfig,
                               _defense=defense_name, _tuned=tuned,
                               _attack=attack_cfg) -> AppConfig:
                training_params = GOLDEN_TRAINING_PARAMS.get(MODEL_CONFIG_NAME)
                if training_params:
                    for key, value in training_params.items():
                        set_nested_attr(config, key, value)
                else:
                    print(f"  WARNING: No Golden HPs found for '{MODEL_CONFIG_NAME}'!")

                set_nested_attr(config, "aggregation.method", _defense)
                for key, value in _tuned.items():
                    set_nested_attr(config, key, value)

                set_nested_attr(config, "data.image.strategy", "dirichlet")
                set_nested_attr(config, "data.image.dirichlet_alpha", 0.5)
                if _attack["use_attack_modifier"]:
                    set_nested_attr(config, "experiment.adv_rate", _attack["adv_rate"])
                    set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate",
                                    _attack["poison_rate"])
                    set_nested_attr(
                        config,
                        "adversary_seller_config.poisoning.image_backdoor_params."
                        "simple_data_poison_params.target_label", 1)
                else:
                    set_nested_attr(config, "experiment.adv_rate", 0.0)
                    config.adversary_seller_config.poisoning.type = PoisonType.NONE

                # --- valuation method selection ---
                v = config.valuation
                v.run_influence = "influence" in methods
                v.run_loo = "loo" in methods
                v.loo_frequency = KSHAP_FREQ
                v.run_kernelshap = "kernelshap" in methods
                v.run_banzhaf = "banzhaf" in methods
                v.run_leastcore = "leastcore" in methods
                v.kernelshap_frequency = KSHAP_FREQ
                v.kernelshap_samples = KSHAP_SAMPLES
                v.leastcore_max_coalitions = LEASTCORE_MAX_COALITIONS
                v.leastcore_timeout = LEASTCORE_TIMEOUT
                return config

            grid = {
                "experiment.image_model_config_name": [MODEL_CONFIG_NAME],
                "experiment.dataset_name": [DATASET_NAME],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                "n_samples": [NUM_SEEDS],
            }

            modifiers = [setup_modifier, use_cifar100_config]
            if use_attack:
                modifiers.append(use_image_backdoor_attack)

            scenarios.append(Scenario(
                name=f"stepE4_valuation_{defense_name}_{attack_cfg['name']}_{DATASET_NAME}",
                base_config_factory=get_base_image_config,
                modifiers=modifiers,
                parameter_grid=grid,
            ))

    return scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valuation", default="all",
        help="Comma-separated subset of {kernelshap,loo,influence,banzhaf,leastcore} or 'all'.")
    args = parser.parse_args()

    if args.valuation.strip().lower() == "all":
        methods = set(VALID_METHODS)
    else:
        methods = {m.strip().lower() for m in args.valuation.split(",") if m.strip()}
        unknown = methods - VALID_METHODS
        if unknown:
            raise ValueError(f"Unknown --valuation method(s) {sorted(unknown)}; "
                             f"valid: {sorted(VALID_METHODS)}")

    output_dir = Path("./configs_generated_benchmark") / "stepE4_valuation_methods"
    generator = ExperimentGenerator(str(output_dir))

    total = 0
    for scenario in generate_e4_scenarios(methods):
        base_config = scenario.base_config_factory()
        modified = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified = modifier(modified)
        total += generator.generate(modified, Scenario(
            name=scenario.name, base_config_factory=scenario.base_config_factory,
            modifiers=[], parameter_grid=scenario.parameter_grid,
        ))

    print(f"\n✅ Step E4 config generation complete: {total} configs.")
    print(f"  Methods enabled : {sorted(methods)}")
    print(f"  Configs saved to: {output_dir}")
