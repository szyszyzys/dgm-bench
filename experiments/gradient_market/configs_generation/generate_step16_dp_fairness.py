# FILE: generate_step16_dp_fairness.py
# Purpose: Measures DP-Robustness Compatibility — i.e., how robust-aggregation
# defenses behave when benign sellers add Gaussian DP noise to their gradients.
#
# NOTE on naming: this step is *not* a "DP fairness" study in the algorithmic
# fairness sense (Bagdasaryan et al., "DP has disparate impact on accuracy",
# 2019, looks at subgroup accuracy gaps). What we measure here is
# **identification fidelity under noise**: the false-positive rate of each
# defense's benign-vs-adversarial classifier when benign updates are perturbed
# by local DP. The original filename / dir name are kept to avoid breaking the
# pipeline; reframe the panel title in the paper to something like
# "DP-Robustness Compatibility" or "Effect of Local DP on Defense Selection".
#
# Threat model: only benign sellers run DP (privacy-conscious honest parties);
# adversaries do not bother with privacy. This is the realistic setup.
#
# Known confounds (acknowledge in the paper caption):
#   1. Defenses use HPs tuned WITHOUT DP (from Step 3). Tighter clip_norms or
#      filter thresholds may be miscalibrated against noisy gradients, so the
#      observed BSR drop combines (a) intrinsic noise sensitivity with
#      (b) HP mismatch. Re-tuning per (defense, epsilon) is expensive;
#      flagging the confound is the cheap fix.
#   2. δ=1e-5 with CIFAR100 (n≈50000) is at the edge of standard practice
#      (δ ≪ 1/n ⇒ δ ≪ 2e-5). Defensible but worth disclosing.
#   3. dp.clip_norm=1.0 is a fixed pre-DP gradient clipping norm. It controls
#      the noise scale (σ ∝ clip_norm / ε). It is NOT the defense's clip_norm.
#      Worth either justifying empirically (median per-client norm post-clip
#      from Step 1 logs) or sweeping as a secondary axis in a follow-up.

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
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import Scenario, use_image_backdoor_attack, use_cifar100_config

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# Privacy budgets to sweep: smaller epsilon = more noise = stronger privacy.
# Geometric grid concentrated in the *transition zone* (ε ∈ [1, 4]) where the
# story actually plays out. Old grid was [0.5, 1.0, 10.0, 100.0] which had a
# huge gap between ε=1 (collapse) and ε=10 (no effect) — exactly where the
# cliff lives. ε=8.0 serves as a near-no-DP anchor without the wasted
# resolution of ε=100.
EPSILON_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]

# Attack states: we run TWO arms per (defense, ε) so we can disentangle
#   (a) "defense rejected benign DP-noised updates" (the result we want), from
#   (b) "benign sellers couldn't learn anything anyway because DP destroyed
#        their utility, so the defense is doing the right thing by rejecting".
# Without the no-attack arm a reviewer can argue we're misattributing utility
# loss to defense behavior. Doubles step 16's run count — set this to
# ["with_attack"] only if budget is tight.
ATTACK_STATES = ["with_attack", "no_attack"]

DP_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
    "attack_modifier": use_image_backdoor_attack,
    "defenses": IMAGE_DEFENSES,
}


def generate_dp_scenarios(attack_state: str) -> List[Scenario]:
    """Generate per-defense scenarios for one attack state ('with_attack' or 'no_attack')."""
    assert attack_state in ("with_attack", "no_attack"), attack_state
    print(f"\n--- Generating Step 16: DP-Robustness Scenarios [{attack_state}] ---")
    scenarios = []
    modality = DP_SETUP["modality_name"]
    model_cfg_name = DP_SETUP["model_config_name"]

    for defense_name in DP_SETUP["defenses"]:
        # CONFOUND: tuned_params come from Step 3, which ran WITHOUT DP.
        # The defense HPs (clip norms, filter thresholds, etc.) are calibrated
        # against non-noisy gradients. Under DP they may be miscalibrated, and
        # the BSR drop measured here will combine intrinsic noise sensitivity
        # with HP mismatch. Acknowledged in the file header docstring.
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
            t_params=tuned_params, atk_mod=DP_SETUP["attack_modifier"],
            mod=modality, atk_state=attack_state,
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
                # Branch on attack state. The no-attack arm is the control:
                # benign-only training under DP, so we can tell whether low BSR
                # in the with_attack arm is "defense rejected benigns" vs.
                # "benigns just couldn't learn under DP noise".
                if atk_state == "with_attack":
                    config = atk_mod(config)
                    set_nested_attr(
                        config,
                        "adversary_seller_config.poisoning.poison_rate",
                        DEFAULT_POISON_RATE,
                    )
                else:
                    # no_attack: ensure adversary is fully disabled
                    set_nested_attr(config, "experiment.adv_rate", 0.0)
                    set_nested_attr(
                        config,
                        "adversary_seller_config.poisoning.poison_rate",
                        0.0,
                    )
                set_nested_attr(config, f"data.{mod}.strategy", "dirichlet")
                set_nested_attr(config, f"data.{mod}.dirichlet_alpha", 0.5)
                return config
            return modifier

        # adv_rate is set per attack state (the modifier overrides the
        # no_attack case to 0.0; we leave the with_attack default here so the
        # parallel runner sees the intended value in the config).
        adv_rate_for_grid = DEFAULT_ADV_RATE if attack_state == "with_attack" else 0.0

        parameter_grid = {
            DP_SETUP["model_config_param_key"]: [model_cfg_name],
            "experiment.dataset_name": [DP_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            "experiment.adv_rate": [adv_rate_for_grid],
        }

        scenario = Scenario(
            name=f"step16_dp_{defense_name}_{DP_SETUP['dataset_name']}_{attack_state}",
            base_config_factory=DP_SETUP["base_config_factory"],
            modifiers=[create_modifier(), DP_SETUP["dataset_modifier"]],
            parameter_grid=parameter_grid,
        )
        scenarios.append(scenario)

    return scenarios


if __name__ == "__main__":
    base_output_dir = "./configs_generated_benchmark"
    output_dir = Path(base_output_dir) / "step16_dp_fairness"
    generator = ExperimentGenerator(str(output_dir))

    # Run BOTH attack states (with_attack + no_attack control). See ATTACK_STATES
    # comment near the top of the file for the rationale.
    scenarios = []
    for atk_state in ATTACK_STATES:
        scenarios.extend(generate_dp_scenarios(atk_state))

    total = 0

    for scenario in scenarios:
        static_grid = scenario.parameter_grid.copy()
        task_count = 0

        for epsilon in EPSILON_VALUES:
            current_grid = static_grid.copy()
            # Enable DP on benign sellers.
            # δ=1e-5 with CIFAR100 (n≈50000) is at the edge of standard practice
            # (δ ≪ 1/n ⇒ δ ≪ 2e-5). Disclose in the methods section.
            # dp.clip_norm=1.0 is the PRE-DP gradient clipping norm (controls
            # noise scale via σ ∝ clip_norm/ε), NOT the defense's clip_norm.
            current_grid["dp.enabled"] = [True]
            current_grid["dp.mechanism"] = ["gaussian"]
            current_grid["dp.epsilon"] = [epsilon]
            current_grid["dp.delta"] = [1e-5]
            current_grid["dp.clip_norm"] = [1.0]

            hp_suffix = f"eps_{epsilon}"
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

    print(f"\n✅ Step 16 (DP-Robustness Compatibility) complete: {total} configs")
    print(f"Configs saved to: {output_dir}")
