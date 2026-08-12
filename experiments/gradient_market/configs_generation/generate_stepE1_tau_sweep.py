# =============================================================================
# Step E1 — Thresholded-FLTrust tau sweep (reviewer points R2.O5 / R3.O3).
#
# Sweeps the explicit accept/reject threshold tau of the fltrust_threshold
# aggregator on CIFAR-100 under the paper's Table-4 attack setting, recording
# (tau, MSR, BSR, ASR, Acc) per cell so the MSR-vs-Acc frontier can be plotted
# and the joint objective (low MSR AND high Acc) checked for ANY tau.
#
# Payments use the E1a "weighted" model so adversary_revenue_share (the
# continuous analogue of MSR) is logged alongside the binary MSR/BSR.
#
# Run (from experiments/gradient_market/configs_generation/):
#   python generate_stepE1_tau_sweep.py
#   python ../run_parallel_experiment.py \
#       --configs_dir ./configs_generated_benchmark/stepE1_tau_sweep \
#       --gpu_ids 0,1,2,3,4 --num_processes 5
#   python ../visualization/extract/extract_stepE1_tau_frontier.py
#
# Table-4 defaults used here (repo flag names): CIFAR-100, dirichlet alpha=0.5,
# n_sellers=10, buyer_ratio=0.10, global_rounds=500 with early stopping
# (patience=10), adv_rate=0.30, poison_rate=0.50, backdoor target class 1,
# n_samples=5 seeds. NOTE two deviations from the repo's own benchmark
# defaults, made to match the prompt's Table 4: target_label=1 (repo default
# 0) and 5 seeds (repo benchmark uses 2-3).
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
TAU_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_SEEDS = 5  # Table-4 spec (prompt); the repo's own benchmark uses 2-3.


def generate_tau_sweep_scenarios() -> List[Scenario]:
    print("\n--- Generating Step E1: thresholded-FLTrust tau sweep ---")

    # fltrust_threshold reuses FLTrust's trust computation, so it inherits
    # FLTrust's tuned HPs (clip_norm) from Step 3.
    tuned = get_tuned_defense_params(
        defense_name="fltrust",
        model_config_name=MODEL_CONFIG_NAME,
        attack_state="with_attack",
        default_attack_type_for_tuning="backdoor",
    ) or {}

    def setup_modifier(config: AppConfig) -> AppConfig:
        training_params = GOLDEN_TRAINING_PARAMS.get(MODEL_CONFIG_NAME)
        if training_params:
            for key, value in training_params.items():
                set_nested_attr(config, key, value)
        else:
            print(f"  WARNING: No Golden HPs found for key '{MODEL_CONFIG_NAME}'!")

        set_nested_attr(config, "aggregation.method", "fltrust_threshold")
        for key, value in tuned.items():
            if key == "aggregation.method":
                continue  # keep fltrust_threshold, not the tuned key's 'fltrust'
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

        # E1a continuous payment read-out alongside binary MSR/BSR.
        set_nested_attr(config, "valuation.payment_model", "weighted")
        return config

    grid = {
        "experiment.image_model_config_name": [MODEL_CONFIG_NAME],
        "experiment.dataset_name": [DATASET_NAME],
        "experiment.global_rounds": [500],
        "experiment.use_early_stopping": [True],
        "experiment.patience": [10],
        "n_samples": [NUM_SEEDS],
        "aggregation.fltrust_threshold.tau": TAU_GRID,
    }

    return [Scenario(
        name=f"stepE1_tau_sweep_{DATASET_NAME}",
        base_config_factory=get_base_image_config,
        modifiers=[setup_modifier, use_cifar100_config, use_image_backdoor_attack],
        parameter_grid=grid,
    )]


if __name__ == "__main__":
    output_dir = Path("./configs_generated_benchmark") / "stepE1_tau_sweep"
    generator = ExperimentGenerator(str(output_dir))

    total = 0
    for scenario in generate_tau_sweep_scenarios():
        base_config = scenario.base_config_factory()
        modified = copy.deepcopy(base_config)
        for modifier in scenario.modifiers:
            modified = modifier(modified)
        total += generator.generate(modified, Scenario(
            name=scenario.name, base_config_factory=scenario.base_config_factory,
            modifiers=[], parameter_grid=scenario.parameter_grid,
        ))

    print(f"\n✅ Step E1 config generation complete: {total} configs "
          f"({len(TAU_GRID)} tau values x {NUM_SEEDS} seeds each via n_samples).")
    print(f"  Configs saved to: {output_dir}")
