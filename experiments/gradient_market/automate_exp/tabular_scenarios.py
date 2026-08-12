from typing import Any, List, Callable, Dict

import torch

from src.common_utils.constants import PoisonType
from experiments.gradient_market.automate_exp.config_generator import set_nested_attr
from experiments.gradient_market.automate_exp.scenarios import Scenario, use_label_flipping_attack
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, AggregationConfig, \
    TabularDataConfig, DataConfig, \
    AdversarySellerConfig, ServerAttackConfig, TrainingConfig, ExperimentConfig, DebugConfig

TEXAS100_TARGET_LABEL = 1  # Or whatever you choose
PURCHASE100_TARGET_LABEL = 1


# --- Base Configuration Factory ---
def get_base_tabular_config() -> AppConfig:
    """Creates the default, base AppConfig for TABULAR-based experiments."""
    return AppConfig(
        experiment=ExperimentConfig(
            dataset_name="Texas100",
            model_structure="mlp",
            global_rounds=500,
            n_sellers=10,
            adv_rate=0.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dataset_type="tabular",
            eval_frequency=10,
            evaluations=["clean", "backdoor"],
            tabular_model_config_name="mlp_texas100_baseline"
        ),
        # training=TrainingConfig(local_epochs=2, batch_size=64, learning_rate=0.0001,),
        training=TrainingConfig(
            local_epochs=5,  # <-- From your tuning winner
            batch_size=64,
            optimizer="Adam",  # <-- From your tuning winner
            learning_rate=0.001,  # <-- From your tuning winner
            momentum=0.0,  # <-- Adam does not use SGD momentum
            weight_decay=0.0,  # <-- Not needed for Adam here
        ),
        server_attack_config=ServerAttackConfig(),
        adversary_seller_config=AdversarySellerConfig(),
        data=DataConfig(
            tabular=TabularDataConfig(
                strategy="dirichlet",
                dirichlet_alpha=0.5,
                buyer_ratio=0.1,
                buyer_strategy="iid"
            )

        ),
        aggregation=AggregationConfig(method="fedavg"),
        seed=42,
        n_samples=3,
        debug=DebugConfig(),
    )


TEXAS100_TRIGGER = {
    "feature_10": 1.0,  # Set 5 specific features to 1
    "feature_50": 1.0,
    "feature_100": 1.0,
    "feature_500": 1.0,
    "feature_1000": 1.0
}

PURCHASE100_TRIGGER = {
    "feature_15": 1.0,
    "feature_75": 1.0,
    "feature_150": 1.0,
    "feature_300": 1.0,
    "feature_500": 1.0
}


def use_tabular_backdoor_with_trigger(
        trigger_conditions: Dict[str, Any],
        target_label: int  # <-- ADD THIS ARGUMENT
) -> Callable[[AppConfig], AppConfig]:
    """Returns a modifier to enable the tabular backdoor attack with specific trigger conditions."""
    print(f'current trigger condiftion: {trigger_conditions}')

    def modifier(config: AppConfig) -> AppConfig:
        config.adversary_seller_config.poisoning.type = PoisonType.TABULAR_BACKDOOR

        label_key = "adversary_seller_config.poisoning.tabular_backdoor_params.target_label"
        set_nested_attr(config, label_key, target_label)

        trigger_key = "adversary_seller_config.poisoning.tabular_backdoor_params.feature_trigger_params.trigger_conditions"
        set_nested_attr(config, trigger_key, trigger_conditions)
        return config

    return modifier


def generate_tabular_main_summary_scenarios() -> List[Scenario]:
    """
    Generates scenarios for the main summary figure, covering both tabular datasets
    with a fixed attack setting and sweeping core aggregators.
    """
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']  # Core aggregators for comparison

    FIXED_ADV_RATE = 0.3
    FIXED_POISON_RATE = 0.3
    SYBIL_STRATEGY = 'mimic'  # Assuming 'mimic' is your standard for main figs

    scenarios.append(Scenario(
        name="main_summary_texas100",  # Match naming convention
        base_config_factory=get_base_tabular_config,
        modifiers=[
            use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL),
            use_sybil_attack(SYBIL_STRATEGY)  # Include Sybil if standard for main figs
        ],
        parameter_grid={
            "experiment.dataset_name": ["texas100"],
            "experiment.model_structure": ["mlp"],
            "experiment.tabular_model_config_name": ["mlp_texas100_baseline"],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],
            # Add other fixed params if necessary (e.g., root_gradient_source)
        }
    ))

    # --- Scenario for Purchase100 (ResNet) ---
    scenarios.append(Scenario(
        name="main_summary_purchase100",  # Match naming convention
        base_config_factory=get_base_tabular_config,  # Still uses the base
        modifiers=[
            use_tabular_backdoor_with_trigger(PURCHASE100_TRIGGER, PURCHASE100_TARGET_LABEL),
            use_sybil_attack(SYBIL_STRATEGY)  # Include Sybil if standard for main figs
        ],
        parameter_grid={
            "experiment.dataset_name": ["purchase100"],
            "experiment.model_structure": ["resnet"],  # <<< Specify correct model
            "experiment.tabular_model_config_name": ["resnet_purchase100_baseline"],  # <<< Specify correct config
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],
            # Add other fixed params if necessary
        }
    ))

    return scenarios


def use_sybil_attack(strategy: str) -> Callable[[AppConfig], AppConfig]:
    """Modifier to enable a specific Sybil attack strategy."""

    def modifier(config: AppConfig) -> AppConfig:
        config.adversary_seller_config.sybil.is_sybil = True
        config.adversary_seller_config.sybil.gradient_default_mode = strategy
        return config

    return modifier


def generate_tabular_sybil_impact_scenarios() -> List[Scenario]:
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    SYBIL_STRATEGIES = ['mimic']
    dataset_name = "texas100"
    model_config = "mlp_texas100_baseline"

    scenarios.append(Scenario(
        name=f"sybil_baseline_{dataset_name}",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL)],
        parameter_grid={
            "experiment.dataset_name": [dataset_name],
            "experiment.model_structure": ["mlp"],
            "experiment.tabular_model_config_name": [model_config],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.3],
            "adversary_seller_config.sybil.is_sybil": [False]
        }
    ))
    for strategy in SYBIL_STRATEGIES:
        scenarios.append(Scenario(
            name=f"sybil_{strategy}_{dataset_name}",
            base_config_factory=get_base_tabular_config,
            modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL),
                       use_sybil_attack(strategy)],
            parameter_grid={
                "experiment.dataset_name": [dataset_name],
                "experiment.model_structure": ["mlp"],
                "experiment.tabular_model_config_name": [model_config],
                "aggregation.method": ALL_AGGREGATORS,
                "experiment.adv_rate": [0.3],
                "adversary_seller_config.poisoning.poison_rate": [0.3],
            }
        ))
    return scenarios


def generate_tabular_attack_impact_scenarios() -> List[Scenario]:
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    ADV_RATES_TO_SWEEP = [0.1, 0.3, 0.5]
    POISON_RATES_TO_SWEEP = [0.1, 0.5, 1.0]
    dataset_name = "texas100"
    model_config = "mlp_texas100_baseline"
    for group_name, sweep_params in [
        ("vary_adv_rate",
         {"experiment.adv_rate": ADV_RATES_TO_SWEEP, "adversary_seller_config.poisoning.poison_rate": [0.3]}),
        ("vary_poison_rate",
         {"experiment.adv_rate": [0.3], "adversary_seller_config.poisoning.poison_rate": POISON_RATES_TO_SWEEP})
    ]:
        scenarios.append(Scenario(
            name=f"impact_{group_name}_{dataset_name}",
            base_config_factory=get_base_tabular_config,
            modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL)],
            parameter_grid={
                "experiment.dataset_name": [dataset_name],
                "experiment.model_structure": ["mlp"],
                "experiment.tabular_model_config_name": [model_config],
                "aggregation.method": ALL_AGGREGATORS,
                **sweep_params
            }
        ))
    return scenarios


def generate_tabular_label_flipping_scenarios() -> List[Scenario]:
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    scenarios.append(Scenario(
        name="label_flip_texas100",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_label_flipping_attack],
        parameter_grid={
            "experiment.dataset_name": ["texas100"],
            "experiment.model_structure": ["mlp"],
            "experiment.tabular_model_config_name": ["mlp_texas100_baseline"],
            "aggregation.method": ALL_AGGREGATORS,
            "experiment.adv_rate": [0.3],
        }
    ))
    return scenarios


# --- NEW SCALABILITY TEST ---
def generate_tabular_scalability_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test aggregator performance as the number of sellers
    (i.e., scalability) increases, using the user's specified list.
    """
    scenarios = []
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']

    # Sweep the total number of sellers
    N_SELLERS_TO_SWEEP = [30, 50, 70, 100]

    # Use a fixed attack setting
    FIXED_ADV_RATE = 0.3
    FIXED_POISON_RATE = 0.3

    dataset_name = "texas100"
    model_config = "mlp_texas100_baseline"

    scenarios.append(Scenario(
        name=f"scalability_attack_{dataset_name}",
        base_config_factory=get_base_tabular_config,
        modifiers=[use_tabular_backdoor_with_trigger(TEXAS100_TRIGGER, TEXAS100_TARGET_LABEL),
                   use_sybil_attack('mimic')],
        parameter_grid={
            "experiment.dataset_name": [dataset_name],
            "experiment.model_structure": ["mlp"],
            "experiment.tabular_model_config_name": [model_config],

            "aggregation.method": ALL_AGGREGATORS,
            "experiment.n_sellers": N_SELLERS_TO_SWEEP,

            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],
        }
    ))
    return scenarios
