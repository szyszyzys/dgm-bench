# =============================================================================
# Step E3 — DP-noised honest sellers: the "Privacy Tax" sweep (reviewer R1.O2).
#
# Adds Gaussian-mechanism local DP to HONEST sellers only (adversaries have
# white-box control and do not handicap themselves — SellerFactory already
# passes dp_config exclusively to benign sellers) and measures how
# distance-based filters misclassify the DP-induced variance as malicious.
#
# Sweep: dp_epsilon in {none, 1, 4, 8} x aggregator in {fltrust, martfl}
# (the per-seller filters where BSR is defined), CIFAR-100, Table-4 attack
# setting. Expected: BSR (primary metric) decreases as epsilon decreases.
#
# Calibration: the ANALYTIC Gaussian mechanism (Balle & Wang, ICML 2018),
# valid for all epsilon > 0 — the classic sigma = C*sqrt(2 ln(1.25/delta))/eps
# formula is only a valid (eps,delta)-DP guarantee for eps <= 1, and this
# sweep includes eps = 4 and 8. delta = 1e-5. The DP clip bound C reuses the
# aggregator's tuned clip_norm when one exists (both fltrust and martfl tune
# clip_norm on CIFAR-100), else DPConfig's default 1.0.
#
# Run (from experiments/gradient_market/configs_generation/):
#   python generate_stepE3_dp_sweep.py
#   python ../run_parallel_experiment.py \
#       --configs_dir ./configs_generated_benchmark/stepE3_dp_sweep \
#       --gpu_ids 0,1,2,3,4 --num_processes 5
#   python ../visualization/extract/extract_stepE3_dp_sweep.py
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
DP_DEFENSES = ["fltrust", "martfl"]   # per-seller filters where BSR is defined
DP_EPSILONS = ["none", 1.0, 4.0, 8.0]  # "none" = control (no DP noise)
DP_DELTA = 1e-5
NUM_SEEDS = 5


def generate_dp_sweep_scenarios() -> List[Scenario]:
    print("\n--- Generating Step E3: DP-noised honest sellers sweep ---")
    scenarios = []

    for defense_name in DP_DEFENSES:
        tuned = get_tuned_defense_params(
            defense_name=defense_name,
            model_config_name=MODEL_CONFIG_NAME,
            attack_state="with_attack",
            default_attack_type_for_tuning="backdoor",
        ) or {}

        for epsilon in DP_EPSILONS:
            dp_enabled = epsilon != "none"
            eps_tag = "none" if not dp_enabled else f"{epsilon:g}"

            def setup_modifier(config: AppConfig,
                               _defense=defense_name, _tuned=tuned,
                               _enabled=dp_enabled, _eps=epsilon) -> AppConfig:
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

                config.dp.enabled = _enabled
                if _enabled:
                    config.dp.mechanism = "gaussian"
                    config.dp.calibration = "analytic"
                    config.dp.epsilon = float(_eps)
                    config.dp.delta = DP_DELTA
                    # Reuse the aggregator's tuned clip bound as the DP
                    # sensitivity C when one exists.
                    if config.aggregation.clip_norm and config.aggregation.clip_norm > 0:
                        config.dp.clip_norm = float(config.aggregation.clip_norm)
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
                name=f"stepE3_dp_{defense_name}_eps_{eps_tag}_{DATASET_NAME}",
                base_config_factory=get_base_image_config,
                modifiers=[setup_modifier, use_cifar100_config, use_image_backdoor_attack],
                parameter_grid=grid,
            ))

    return scenarios


if __name__ == "__main__":
    output_dir = Path("./configs_generated_benchmark") / "stepE3_dp_sweep"
    generator = ExperimentGenerator(str(output_dir))

    total = 0
    for scenario in generate_dp_sweep_scenarios():
        base_config = scenario.base_config_factory()
        modified = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified = modifier(modified)
        total += generator.generate(modified, Scenario(
            name=scenario.name, base_config_factory=scenario.base_config_factory,
            modifiers=[], parameter_grid=scenario.parameter_grid,
        ))

    print(f"\n✅ Step E3 config generation complete: {total} configs "
          f"({len(DP_DEFENSES)} defenses x {len(DP_EPSILONS)} epsilon values, "
          f"{NUM_SEEDS} seeds each via n_samples).")
    print(f"  Configs saved to: {output_dir}")
