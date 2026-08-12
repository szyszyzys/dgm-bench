# FILE: generate_step15_proportional_pricing.py
# Purpose: Re-runs the main summary experiments (Step 10/12) under all three payment models
#          to compare attacker revenue capture across pricing mechanisms.

import copy
import sys
from pathlib import Path
from typing import List

from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,
    NUM_SEEDS_PER_CONFIG as _GLOBAL_NUM_SEEDS,
    get_tuned_defense_params,
    IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
)

# Step 15 (proportional pricing) is one of the two paper-headline tables
# (the other being Step 10 main summary). Override the global default of 2
# seeds to 3 so the per-payment-model attacker-revenue numbers carry
# defensible error bars. The per-cell .success skip logic preserves any
# already-completed seed_42 / seed_43 cells; only the new seed_44 cells run.
NUM_SEEDS_PER_CONFIG = max(3, _GLOBAL_NUM_SEEDS)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config
from experiments.gradient_market.automate_exp.scenarios import Scenario, use_image_backdoor_attack, use_cifar100_config

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

PAYMENT_MODELS = ["quality_based", "proportional", "binary"]

PRICING_SETUPS = [
    {
        "modality_name": "image",
        "base_config_factory": get_base_image_config,
        "dataset_name": "CIFAR100",
        "model_config_param_key": "experiment.image_model_config_name",
        "model_config_name": "cifar100_cnn",
        "dataset_modifier": use_cifar100_config,
        "attack_modifier": use_image_backdoor_attack,
        "defenses": IMAGE_DEFENSES,
    },
]


def generate_pricing_scenarios() -> List[Scenario]:
    print("\n--- Generating Step 15: Proportional Pricing Scenarios ---")
    scenarios = []

    for setup in PRICING_SETUPS:
        modality = setup["modality_name"]
        model_cfg_name = setup["model_config_name"]

        for defense_name in setup["defenses"]:
            tuned_params = get_tuned_defense_params(
                defense_name=defense_name,
                model_config_name=model_cfg_name,
                attack_state="with_attack",
                default_attack_type_for_tuning="backdoor"
            )
            if not tuned_params and defense_name != "fedavg":
                print(f"  SKIPPING {defense_name}: no tuned params")
                continue

            def create_modifier(
                d_name=defense_name, m_name=model_cfg_name,
                t_params=tuned_params, atk_mod=setup["attack_modifier"],
                mod=modality
            ):
                def modifier(config: AppConfig) -> AppConfig:
                    golden = GOLDEN_TRAINING_PARAMS.get(m_name)
                    if golden:
                        for k, v in golden.items():
                            set_nested_attr(config, k, v)
                    if t_params:
                        for k, v in t_params.items():
                            set_nested_attr(config, k, v)
                    if "skymask" in d_name:
                        model_struct = "resnet18" if "resnet" in m_name else "flexiblecnn"
                        set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)
                    config = atk_mod(config)
                    set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", DEFAULT_POISON_RATE)
                    set_nested_attr(config, f"data.{mod}.strategy", "dirichlet")
                    set_nested_attr(config, f"data.{mod}.dirichlet_alpha", 0.5)
                    return config
                return modifier

            parameter_grid = {
                setup["model_config_param_key"]: [model_cfg_name],
                "experiment.dataset_name": [setup["dataset_name"]],
                "n_samples": [NUM_SEEDS_PER_CONFIG],
                "experiment.use_early_stopping": [True],
                "experiment.patience": [10],
                "experiment.adv_rate": [DEFAULT_ADV_RATE],
            }

            scenario = Scenario(
                name=f"step15_pricing_{defense_name}_{setup['dataset_name']}",
                base_config_factory=setup["base_config_factory"],
                modifiers=[create_modifier(), setup["dataset_modifier"]],
                parameter_grid=parameter_grid,
            )
            scenarios.append(scenario)

    return scenarios


if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step15_proportional_pricing"
    generator = ExperimentGenerator(str(output_dir))

    scenarios = generate_pricing_scenarios()
    total = 0

    for scenario in scenarios:
        static_grid = scenario.parameter_grid.copy()
        task_count = 0

        for payment_model in PAYMENT_MODELS:
            current_grid = static_grid.copy()
            current_grid["valuation.payment_model"] = [payment_model]

            hp_suffix = f"payment_{payment_model}"
            current_grid["experiment.save_path"] = [f"./results/{scenario.name}/{hp_suffix}"]

            temp_scenario = Scenario(
                name=f"{scenario.name}/{hp_suffix}",
                base_config_factory=scenario.base_config_factory,
                modifiers=[],
                parameter_grid=current_grid,
            )

            base_config = temp_scenario.base_config_factory()
            for mod in scenario.modifiers:
                base_config = mod(base_config)

            num_gen = generator.generate(base_config, temp_scenario)
            task_count += num_gen

        print(f"-> {scenario.name}: {task_count} configs")
        total += task_count

    print(f"\n✅ Step 15 (Proportional Pricing) complete: {total} configs")
    print(f"Configs saved to: {output_dir}")
