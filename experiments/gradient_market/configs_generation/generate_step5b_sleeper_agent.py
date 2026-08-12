# FILE: generate_step5b_sleeper_agent.py
# =============================================================================
# Step 5b: Sleeper-Agent (temporal) attack.
#
# Adversaries act benign for `benign_rounds` rounds (no Sybil coordination)
# to build trust in stateful aggregators (FoolsGold, MartFL with change_base).
# After the dormancy period, all adversaries simultaneously switch to a
# coordinated strike (knock_out / drowning / alie). The marketplace logs
# per-round MSR and per-seller aggregation_weight, from which downstream
# analysis recovers Time-to-Detection (TTD) and the transient accuracy dip.
#
# Recommended defenses to compare:
#   * stateful  : foolsgold, martfl, rflpa
#   * stateless : fedavg, trimmed_mean, multi_krum, fltrust, spmc
#
# Sweep:
#   benign_rounds  in {0, 25, 50, 100}
#   strike         in {"knock_out", "drowning", "alie"}
#
# Total per defense: 4 * 3 = 12 base configs (x 3 seeds).
# =============================================================================

import copy
import sys
from pathlib import Path
from typing import List

from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    DEFAULT_ADV_RATE, DEFAULT_POISON_RATE,
    enable_valuation,
    get_tuned_defense_params, GOLDEN_TRAINING_PARAMS, IMAGE_DEFENSES,
)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_image_backdoor_attack, use_cifar100_config,
    use_sleeper_agent_attack,
)

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, PoisonType
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Sleeper sweep grid ---
BENIGN_ROUNDS_SWEEP = [0, 25, 50, 100]    # 0 == "no sleeper" (control)
STRIKE_STRATEGIES = ["knock_out", "drowning", "alie"]

# --- Defense set: include stateful + stateless for direct comparison ---
SLEEPER_DEFENSES = [
    "fedavg",          # stateless control
    "fltrust",         # stateless trust-based
    "trimmed_mean",    # stateless geometric
    "multi_krum",      # stateless geometric
    "spmc",            # stateless leave-one-out
    "martfl",          # weakly stateful (dynamic baseline)
    "rflpa",           # 1-round memory (negative control)
    "foolsgold",       # strongly stateful — primary target
]

SLEEPER_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
    "attack_modifier": use_image_backdoor_attack,
}

# Step-5b defaults: more sellers + longer horizon to give sleepers room to build trust.
N_SELLERS = 30
GLOBAL_ROUNDS = 200
ADV_RATE = 0.2  # 6 of 30 sellers


def _make_setup_modifier(defense_name: str, model_cfg_name: str, modality: str):
    tuned = get_tuned_defense_params(
        defense_name=defense_name,
        model_config_name=model_cfg_name,
        attack_state="with_attack",
        default_attack_type_for_tuning="backdoor",
    )

    def modifier(config: AppConfig) -> AppConfig:
        gp = GOLDEN_TRAINING_PARAMS.get(model_cfg_name)
        if gp:
            for k, v in gp.items():
                set_nested_attr(config, k, v)
        if tuned:
            for k, v in tuned.items():
                set_nested_attr(config, k, v)
        if "skymask" in defense_name:
            model_struct = "resnet18" if "resnet" in model_cfg_name else "flexiblecnn"
            set_nested_attr(config, "aggregation.skymask.sm_model_type", model_struct)
        set_nested_attr(config, f"data.{modality}.strategy", "dirichlet")
        set_nested_attr(config, f"data.{modality}.dirichlet_alpha", 0.5)
        set_nested_attr(config, "experiment.global_rounds", GLOBAL_ROUNDS)
        set_nested_attr(config, "experiment.n_sellers", N_SELLERS)
        return config

    return modifier


def generate_sleeper_scenarios() -> List[Scenario]:
    print("\n--- Generating Step 5b: Sleeper Agent Scenarios ---")
    scenarios: List[Scenario] = []

    for defense_name in SLEEPER_DEFENSES:
        print(f"-- Defense: {defense_name}")
        setup_mod = _make_setup_modifier(
            defense_name, SLEEPER_SETUP["model_config_name"], SLEEPER_SETUP["modality_name"]
        )

        base_grid = {
            SLEEPER_SETUP["model_config_param_key"]: [SLEEPER_SETUP["model_config_name"]],
            "experiment.dataset_name": [SLEEPER_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.adv_rate": [ADV_RATE],
            "experiment.n_sellers": [N_SELLERS],
            "experiment.global_rounds": [GLOBAL_ROUNDS],
            "adversary_seller_config.poisoning.poison_rate": [DEFAULT_POISON_RATE],
            "experiment.use_early_stopping": [False],  # Don't stop pre-strike
            "aggregation.method": [defense_name],
        }

        for strike in STRIKE_STRATEGIES:
            for benign_rounds in BENIGN_ROUNDS_SWEEP:
                scenario_name = (
                    f"step5b_sleeper_{defense_name}_strike_{strike}_dormant_{benign_rounds}"
                )
                grid = base_grid.copy()
                grid["adversary_seller_config.sybil.benign_rounds"] = [benign_rounds]

                modifiers = [
                    setup_mod,
                    SLEEPER_SETUP["dataset_modifier"],
                    SLEEPER_SETUP["attack_modifier"],
                    use_sleeper_agent_attack(strike_strategy=strike, benign_rounds=benign_rounds),
                    lambda c: enable_valuation(
                        c, influence=False, loo=False, kernelshap=False
                    ),
                ]

                scenarios.append(Scenario(
                    name=scenario_name,
                    base_config_factory=SLEEPER_SETUP["base_config_factory"],
                    modifiers=modifiers,
                    parameter_grid=grid,
                ))
    return scenarios


if __name__ == "__main__":
    output_dir = Path("./configs_generated_benchmark") / "step20_sleeper_agent"
    generator = ExperimentGenerator(str(output_dir))

    scenarios = generate_sleeper_scenarios()
    total = 0
    print("\n--- Generating Step 5b config files ---")

    for scenario in scenarios:
        print(f"\nProcessing: {scenario.name}")
        scenario.parameter_grid["experiment.save_path"] = [f"./results/{scenario.name}"]
        base = scenario.base_config_factory()
        modified = copy.deepcopy(base)
        for m in scenario.modifiers:
            modified = m(modified)
        n = generator.generate(modified, scenario)
        print(f"  -> {n} configs")
        total += n

    print(f"\n[OK] Step 5b generation complete. Total configs: {total}")
    print(f"     Saved to: {output_dir}")
