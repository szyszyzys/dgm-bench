# FILE: generate_step18_daved_comparison.py
# Purpose: Head-to-head comparison of DAVED against all baseline defenses.
#          Tests under backdoor and label-flip attacks across image, text, and tabular.
#          Uses tuned HPs from Step 3 for baselines; sweeps DAVED HPs.

import copy
import sys
from pathlib import Path
from typing import List

from config_common_utils import (
    GOLDEN_TRAINING_PARAMS,
    NUM_SEEDS_PER_CONFIG,
    get_tuned_defense_params,
    IMAGE_DEFENSES, TEXT_TABULAR_DEFENSES,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config
from experiments.gradient_market.automate_exp import get_base_tabular_config
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_image_backdoor_attack, use_cifar100_config, use_femnist_config,
)

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)


def _use_label_flip_attack(config: AppConfig) -> AppConfig:
    config.adversary_seller_config.poisoning.type = PoisonType.LABEL_FLIP
    config.adversary_seller_config.poisoning.poison_rate = DEFAULT_POISON_RATE
    return config


COMPARISON_SETUPS = [
    # Image — CIFAR100 CNN (primary benchmark dataset)
    {
        "modality": "image",
        "base_factory": get_base_image_config,
        "dataset_name": "CIFAR100",
        "model_key": "experiment.image_model_config_name",
        "model_config": "cifar100_cnn",
        "dataset_mod": use_cifar100_config,
        "attacks": [
            ("backdoor", use_image_backdoor_attack),
            ("labelflip", _use_label_flip_attack),
        ],
        "defenses": IMAGE_DEFENSES,
    },
    # Image — FEMNIST CNN (natural non-IID)
    {
        "modality": "image",
        "base_factory": get_base_image_config,
        "dataset_name": "FEMNIST",
        "model_key": "experiment.image_model_config_name",
        "model_config": "femnist_cnn",
        "dataset_mod": use_femnist_config,
        "attacks": [
            ("backdoor", use_image_backdoor_attack),
            ("labelflip", _use_label_flip_attack),
        ],
        "defenses": IMAGE_DEFENSES,
    },
]


def generate_comparison_scenarios() -> List[dict]:
    """Returns a list of (scenario, defense_name, attack_name) tuples."""
    print("\n--- Generating Step 18: DAVED vs Baselines Comparison ---")
    entries = []

    for setup in COMPARISON_SETUPS:
        modality = setup["modality"]
        model_cfg = setup["model_config"]

        for attack_name, attack_mod in setup["attacks"]:
            for defense_name in setup["defenses"]:
                tuned = get_tuned_defense_params(
                    defense_name=defense_name,
                    model_config_name=model_cfg,
                    attack_state="with_attack",
                    default_attack_type_for_tuning=attack_name,
                )
                if not tuned and defense_name != "fedavg":
                    print(f"  SKIP {defense_name}/{model_cfg}/{attack_name}: no tuned params")
                    continue

                def make_modifier(
                    d=defense_name, m=model_cfg, t=tuned,
                    atk=attack_mod, mod=modality
                ):
                    def modifier(config: AppConfig) -> AppConfig:
                        golden = GOLDEN_TRAINING_PARAMS.get(m)
                        if golden:
                            for k, v in golden.items():
                                set_nested_attr(config, k, v)
                        if t:
                            for k, v in t.items():
                                set_nested_attr(config, k, v)
                        if "skymask" in d:
                            model_struct = "resnet18" if "resnet" in m else "flexiblecnn"
                            set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)
                        config = atk(config)
                        set_nested_attr(config, "adversary_seller_config.poisoning.poison_rate", DEFAULT_POISON_RATE)
                        set_nested_attr(config, f"data.{mod}.strategy", "dirichlet")
                        set_nested_attr(config, f"data.{mod}.dirichlet_alpha", 0.5)
                        return config
                    return modifier

                grid = {
                    setup["model_key"]: [model_cfg],
                    "experiment.dataset_name": [setup["dataset_name"]],
                    "n_samples": [NUM_SEEDS_PER_CONFIG],
                    "experiment.use_early_stopping": [True],
                    "experiment.patience": [10],
                    "experiment.adv_rate": [DEFAULT_ADV_RATE],
                }

                scenario = Scenario(
                    name=f"step18_{defense_name}_{setup['dataset_name']}_{attack_name}",
                    base_config_factory=setup["base_factory"],
                    modifiers=[make_modifier(), setup["dataset_mod"]],
                    parameter_grid=grid,
                )
                entries.append((scenario, defense_name, attack_name))

    return entries


if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step18_daved_comparison"
    generator = ExperimentGenerator(str(output_dir))

    entries = generate_comparison_scenarios()
    total = 0

    for scenario, defense_name, attack_name in entries:
        static_grid = scenario.parameter_grid.copy()
        save_path = f"./results/{scenario.name}/tuned"
        static_grid["experiment.save_path"] = [save_path]

        temp_scenario = Scenario(
            name=f"{scenario.name}/tuned",
            base_config_factory=scenario.base_config_factory,
            modifiers=[],  # Modifiers already applied below; don't run twice
            parameter_grid=static_grid,
        )
        base_config = temp_scenario.base_config_factory()
        for mod in scenario.modifiers:
            base_config = mod(base_config)

        num_gen = generator.generate(base_config, temp_scenario)
        total += num_gen
        print(f"  {scenario.name}: {num_gen} configs")

    print(f"\n✅ Step 18 (DAVED Comparison) complete: {total} configs")
    print(f"Configs saved to: {output_dir}")
