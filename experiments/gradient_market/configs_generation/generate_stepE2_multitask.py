# =============================================================================
# Step E2 — Multi-round persistent market configs (reviewer R2.O6 / R2.O3).
#
# Generates CIFAR-100 configs for the outer-task-loop runner
# (experiments/gradient_market/run_multi_task.py): fltrust and martfl (the
# per-seller filters where selection/attrition are defined), Table-4 attack
# setting, E1a weighted payments, T=5 tasks, exit_k=3, carry_model=reset.
# Also generates one fltrust scenario with the gray-box UCB-bandit adversary
# (adaptive_attack) for the optional bandit_persist comparison — run that
# config twice, with and without --bandit_persist.
#
# Run (from experiments/gradient_market/configs_generation/, ON THE SERVER so
# device resolves to cuda):
#   python generate_stepE2_multitask.py
#   for f in $(find ./configs_generated_benchmark/stepE2_multitask -name config.yaml); do
#       python -m experiments.gradient_market.run_multi_task "$f"
#   done
# =============================================================================

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
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

MODEL_CONFIG_NAME = "cifar100_cnn"
DATASET_NAME = "CIFAR100"
E2_DEFENSES = ["fltrust", "martfl"]
NUM_SEEDS = 5
NUM_TASKS = 5
EXIT_K = 3


def generate_e2_scenarios() -> List[Scenario]:
    print("\n--- Generating Step E2: multi-round persistent market ---")
    scenarios = []

    for defense_name in E2_DEFENSES:
        for variant in (["plain", "bandit"] if defense_name == "fltrust" else ["plain"]):
            tuned = get_tuned_defense_params(
                defense_name=defense_name,
                model_config_name=MODEL_CONFIG_NAME,
                attack_state="with_attack",
                default_attack_type_for_tuning="backdoor",
            ) or {}

            def setup_modifier(config: AppConfig,
                               _defense=defense_name, _tuned=tuned,
                               _variant=variant) -> AppConfig:
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
                set_nested_attr(config, "experiment.adv_rate", DEFAULT_ADV_RATE)
                set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate",
                                DEFAULT_POISON_RATE)
                set_nested_attr(
                    config,
                    "adversary_seller_config.poisoning.image_backdoor_params."
                    "simple_data_poison_params.target_label", 1)

                # E2 contract: weighted payments drive attrition accounting.
                set_nested_attr(config, "valuation.payment_model", "weighted")
                config.multi_task.num_tasks = NUM_TASKS
                config.multi_task.exit_k = EXIT_K
                config.multi_task.carry_model = "reset"

                if _variant == "bandit":
                    # Gray-box "Individual Infiltration" UCB-bandit adversary.
                    # Run this config twice: once as-is (state resets per task)
                    # and once with --bandit_persist.
                    config.adversary_seller_config.adaptive_attack.is_active = True
                    config.adversary_seller_config.adaptive_attack.threat_model = "black_box"
                    config.adversary_seller_config.adaptive_attack.attack_mode = "gradient_manipulation"
                return config

            grid = {
                "experiment.image_model_config_name": [MODEL_CONFIG_NAME],
                "experiment.dataset_name": [DATASET_NAME],
                "experiment.global_rounds": [500],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                "n_samples": [NUM_SEEDS],
            }

            scenarios.append(Scenario(
                name=f"stepE2_multitask_{defense_name}_{variant}_{DATASET_NAME}",
                base_config_factory=get_base_image_config,
                modifiers=[setup_modifier, use_cifar100_config, use_image_backdoor_attack],
                parameter_grid=grid,
            ))

    return scenarios


if __name__ == "__main__":
    output_dir = Path("./configs_generated_benchmark") / "stepE2_multitask"
    generator = ExperimentGenerator(str(output_dir))

    total = 0
    for scenario in generate_e2_scenarios():
        base_config = scenario.base_config_factory()
        modified = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified = modifier(modified)
        total += generator.generate(modified, Scenario(
            name=scenario.name, base_config_factory=scenario.base_config_factory,
            modifiers=[], parameter_grid=scenario.parameter_grid,
        ))

    print(f"\n✅ Step E2 config generation complete: {total} configs "
          f"(fltrust plain + fltrust bandit + martfl plain; {NUM_SEEDS} seeds via n_samples).")
    print(f"  Run each with: python -m experiments.gradient_market.run_multi_task <config.yaml>")
    print(f"  Bandit comparison: run the *_bandit_* config a second time with --bandit_persist")
    print(f"  Configs saved to: {output_dir}")
