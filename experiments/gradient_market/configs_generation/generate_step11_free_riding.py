# FILE: generate_step11_free_riding.py
# =============================================================================
# Step 11: Free-Rider / Lazy Seller economic attack.
#
# Replaces a fraction of *honest* sellers with LazyGradientSellers that fake
# participation to capture payment without doing the work. This is the first
# experiment that tests the economic axis of the marketplace independently of
# model poisoning — adv_rate is fixed at 0 and no backdoor / label flip is
# applied. Headline metrics:
#
#   * free_rider_revenue_share = sum(price_paid[lazy]) / sum(price_paid[*])
#   * dilution_cost            = rounds-to-target-accuracy delta vs lazy_rate=0
#   * MSR(lazy)                = fraction of lazy uploads accepted
#
# Sweep:
#   strategy   in {"norm_noise", "echo", "stale_local"}
#   lazy_rate  in {0.0 (control), 0.1, 0.3, 0.5}
#   defense    : all 14 aggregators (including FedAvg as control)
#
# Per defense: 3 * 4 = 12 base configs (x 3 seeds).
# =============================================================================

import copy
import sys
from pathlib import Path
from typing import List

from config_common_utils import (
    NUM_SEEDS_PER_CONFIG,
    enable_valuation,
    get_tuned_defense_params, GOLDEN_TRAINING_PARAMS,
    IMAGE_DEFENSES, NEW_IMAGE_DEFENSES,
)
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config
from experiments.gradient_market.automate_exp.scenarios import (
    Scenario, use_cifar100_config, use_lazy_seller_attack,
)

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator, set_nested_attr
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    sys.exit(1)

# --- Sweep grid ---
LAZY_STRATEGIES = ["norm_noise", "echo", "stale_local"]
LAZY_RATES = [0.0, 0.1, 0.3, 0.5]   # 0.0 = control (all honest)
ALL_DEFENSES_FOR_FREE_RIDING = list(IMAGE_DEFENSES) + list(NEW_IMAGE_DEFENSES)

FREE_RIDING_SETUP = {
    "modality_name": "image",
    "base_config_factory": get_base_image_config,
    "dataset_name": "CIFAR100",
    "model_config_param_key": "experiment.image_model_config_name",
    "model_config_name": "cifar100_cnn",
    "dataset_modifier": use_cifar100_config,
}


def _make_setup_modifier(defense_name: str, model_cfg_name: str, modality: str):
    tuned = get_tuned_defense_params(
        defense_name=defense_name,
        model_config_name=model_cfg_name,
        attack_state="no_attack",
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
        return config

    return modifier


def generate_free_riding_scenarios() -> List[Scenario]:
    print("\n--- Generating Step 11: Free-Riding Scenarios ---")
    scenarios: List[Scenario] = []

    for defense_name in ALL_DEFENSES_FOR_FREE_RIDING:
        print(f"-- Defense: {defense_name}")
        setup_mod = _make_setup_modifier(
            defense_name, FREE_RIDING_SETUP["model_config_name"], FREE_RIDING_SETUP["modality_name"]
        )

        base_grid = {
            FREE_RIDING_SETUP["model_config_param_key"]: [FREE_RIDING_SETUP["model_config_name"]],
            "experiment.dataset_name": [FREE_RIDING_SETUP["dataset_name"]],
            "n_samples": [NUM_SEEDS_PER_CONFIG],
            "experiment.adv_rate": [0.0],
            "experiment.use_early_stopping": [True],
            "experiment.patience": [10],
            "aggregation.method": [defense_name],
            # Use quality-based payments so revenue capture is meaningful
            "valuation.payment_model": ["quality_based"],
        }

        # Emit the lazy_rate=0 control exactly once per defense (no per-strategy duplication)
        control_tag = f"step11_freerider_{defense_name}_control"
        control_grid = base_grid.copy()
        control_grid["adversary_seller_config.lazy_seller.lazy_rate"] = [0.0]
        scenarios.append(Scenario(
            name=control_tag,
            base_config_factory=FREE_RIDING_SETUP["base_config_factory"],
            modifiers=[
                setup_mod,
                FREE_RIDING_SETUP["dataset_modifier"],
                lambda c: enable_valuation(c, influence=False, loo=False, kernelshap=False),
            ],
            parameter_grid=control_grid,
        ))

        for strategy in LAZY_STRATEGIES:
            for lazy_rate in [r for r in LAZY_RATES if r > 0.0]:
                tag = f"step11_freerider_{defense_name}_{strategy}_rate_{lazy_rate}"

                grid = base_grid.copy()
                grid["adversary_seller_config.lazy_seller.lazy_rate"] = [lazy_rate]

                if False:
                    pass
                else:
                    modifiers = [
                        setup_mod,
                        FREE_RIDING_SETUP["dataset_modifier"],
                        use_lazy_seller_attack(
                            strategy=strategy,
                            lazy_rate=lazy_rate,
                            jitter_scale=1e-3,
                            stale_period=10,
                        ),
                        lambda c: enable_valuation(c, influence=False, loo=False, kernelshap=False),
                    ]

                scenarios.append(Scenario(
                    name=tag,
                    base_config_factory=FREE_RIDING_SETUP["base_config_factory"],
                    modifiers=modifiers,
                    parameter_grid=grid,
                ))
    return scenarios


if __name__ == "__main__":
    output_dir = Path("./configs_generated_benchmark") / "step11_free_riding"
    generator = ExperimentGenerator(str(output_dir))

    scenarios = generate_free_riding_scenarios()
    total = 0
    print("\n--- Generating Step 11 config files ---")

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

    print(f"\n[OK] Step 11 generation complete. Total configs: {total}")
    print(f"     Saved to: {output_dir}")
