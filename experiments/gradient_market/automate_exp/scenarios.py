from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any

from src.common_utils.constants import PoisonType
from src.marketplace.utils.gradient_market_utils.gradient_market_configs import AppConfig, SybilDrowningConfig, SybilConfig
from experiments.gradient_market.automate_exp.base_configs import get_base_image_config, get_base_text_config


def set_nested_attr(obj: Any, key: str, value: Any):
    """
    Sets a nested attribute on an object or a key in a nested dict
    using a dot-separated key.
    """
    keys = key.split('.')
    current_obj = obj

    # Traverse to the second-to-last object in the path
    for k in keys[:-1]:
        current_obj = getattr(current_obj, k)

    # Get the final key/attribute to be set
    final_key = keys[-1]

    # --- THIS IS THE CRITICAL LOGIC ---
    # Check if the object we need to modify is a dictionary
    if isinstance(current_obj, dict):
        # If it's a dict, use item assignment (e.g., my_dict['key'] = value)
        current_obj[final_key] = value
    else:
        # Otherwise, use attribute assignment (e.g., my_obj.key = value)
        setattr(current_obj, final_key, value)


def disable_all_seller_attacks(config: AppConfig) -> AppConfig:
    """Modifier to ensure a clean environment with only benign sellers."""
    config.experiment.adv_rate = 0.0  # No adversarial sellers
    config.adversary_seller_config.poisoning.type = PoisonType.NONE
    config.adversary_seller_config.sybil.is_sybil = False
    # Add any other seller-side attacks you might have
    # config.adversary_seller_config.drowning_attack.is_active = False
    # config.adversary_seller_config.adaptive_attack.is_active = False
    return config


# --- Define the structure of a Scenario ---
@dataclass
class Scenario:
    """A declarative representation of an experimental scenario."""
    name: str
    base_config_factory: Callable[[], AppConfig]
    modifiers: List[Callable[[AppConfig], AppConfig]] = field(default_factory=list)
    parameter_grid: Dict[str, List[Any]] = field(default_factory=dict)


# --- Reusable Modifier Functions ---

def use_cifar10_config(config: AppConfig) -> AppConfig:
    """Modifier to set up for the CIFAR-10 dataset."""
    config.experiment.dataset_name = "CIFAR10"
    config.data.image.property_skew.property_key = "class_in_[0,1,8,9]"
    return config


def use_cifar100_config(config: AppConfig) -> AppConfig:
    """Modifier to set up for the CIFAR-100 dataset."""
    config.experiment.dataset_name = "CIFAR100"
    config.data.image.property_skew.property_key = f"class_in_{list(range(50))}"
    return config


def use_femnist_config(config: AppConfig) -> AppConfig:
    """Modifier for FEMNIST (LEAF benchmark, natural non-IID by writer).

    FEMNIST is in this benchmark *specifically* because it provides a real
    natural non-IID partitioning by writer_id — that is the experimental
    contribution of FEMNIST as a federated benchmark and the reason we
    cannot replace it with Dirichlet partitioning. The underlying nested-
    Subset unwrap that used to break this code path was fixed in
    src/common_utils/data_utils/data_partitioner.py:_partition_by_natural_id
    (now recursively unwraps Subsets and composes their indices when
    looking up writer_ids).
    """
    config.experiment.dataset_name = "FEMNIST"
    config.data.image.strategy = "natural"  # partition by writer_id
    return config


def use_trec_config(config: AppConfig) -> AppConfig:
    """Modifier for the TREC dataset."""
    config.experiment.dataset_name = "TREC"
    return config


def use_image_backdoor_attack(config: AppConfig) -> AppConfig:
    config.adversary_seller_config.poisoning.type = PoisonType.IMAGE_BACKDOOR
    return config


def use_text_backdoor_attack(config: AppConfig) -> AppConfig:
    config.adversary_seller_config.poisoning.type = PoisonType.TEXT_BACKDOOR
    return config


FIXED_ATTACK_ADV_RATE = 0.3
FIXED_ATTACK_POISON_RATE = 0.5  # Match defense tune

# --- Heterogeneity Levels to Sweep ---
DIRICHLET_ALPHAS_TO_SWEEP = [100.0, 1.0, 0.5, 0.1]  # High alpha = More IID, Low alpha = More Non-IID

NUM_SEEDS_PER_CONFIG = 3


def use_sybil_attack(strategy: str, **kwargs) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier function that enables Sybil attack with a specific strategy
    and optional strategy-specific parameters.
    """

    def modifier(config: AppConfig) -> AppConfig:
        sybil_cfg = config.adversary_seller_config.sybil
        sybil_cfg.is_sybil = True
        sybil_cfg.gradient_default_mode = strategy

        # --- Inject strategy-specific parameters ---
        if strategy == "oracle_blend" and "blend_alpha" in kwargs:
            sybil_cfg.oracle_blend_alpha = kwargs["blend_alpha"]
        # Add elif blocks here for parameters of other strategies if needed
        # elif strategy == "systematic_probe":
        #    sybil_cfg.probe_param = kwargs.get("probe_param")

        # Ensure base poisoning is active (handled by other modifiers)
        return config

    return modifier


def use_label_flipping_attack(config: AppConfig) -> AppConfig:
    """Modifier to set up for a simple label-flipping attack."""
    config.adversary_seller_config.poisoning.type = PoisonType.LABEL_FLIP
    return config


# =====================================================================
# Step 5b: Sleeper-Agent (temporal) attack helper
# =====================================================================
def use_sleeper_agent_attack(
        strike_strategy: str = "knock_out",
        benign_rounds: int = 100,
        oracle_blend_alpha: float = 0.1,
        drowning_victim_id: str = "bn_6",
        drowning_attack_strength: float = 1.0,
) -> Callable[[AppConfig], AppConfig]:
    """Configures a Sybil "sleeper" attack: act benign for ``benign_rounds``
    rounds (no Sybil coordination is applied), then suddenly switch to a
    coordinated strike using ``strike_strategy``.

    Stateful defenses (FoolsGold, MartFL with change_base) build trust on the
    benign updates and need time to react after the strike — Time-to-Detection
    (TTD) becomes the headline metric.

    NOTE: when ``strike_strategy == "drowning"`` you must pass a valid
    ``drowning_victim_id`` that actually exists in the benign seller pool
    (e.g. ``bn_6`` when n_sellers=30, adv_rate=0.2). The default ``bn_3`` is
    *invalid* for that setup because adv_0..adv_5 take the first 6 IDs.
    """

    def modifier(config: AppConfig) -> AppConfig:
        sybil_cfg = config.adversary_seller_config.sybil
        sybil_cfg.is_sybil = True
        sybil_cfg.gradient_default_mode = strike_strategy
        sybil_cfg.benign_rounds = int(benign_rounds)
        if strike_strategy == "oracle_blend":
            sybil_cfg.oracle_blend_alpha = oracle_blend_alpha
        if strike_strategy == "drowning":
            if not sybil_cfg.strategy_configs:
                sybil_cfg.strategy_configs = {}
            sybil_cfg.strategy_configs["drowning"] = {
                "victim_id": drowning_victim_id,
                "attack_strength": float(drowning_attack_strength),
            }
        return config

    return modifier


# =====================================================================
# Step 11: Free-Rider / Lazy Seller helper
# =====================================================================
def use_lazy_seller_attack(
        strategy: str = "echo",
        lazy_rate: float = 0.3,
        jitter_scale: float = 1e-3,
        stale_period: int = 10,
) -> Callable[[AppConfig], AppConfig]:
    """Configures a fraction of *benign* sellers to behave as free-riders.

    Strategies:
      * ``norm_noise`` — Gaussian noise rescaled to median accepted L2 norm.
      * ``echo``       — Replay last round's aggregated delta + small jitter.
      * ``stale_local``— Train once, replay for ``stale_period`` rounds.

    These attacks do not poison the model directly; they just steal payment.
    The headline metric is ``free_rider_revenue_share`` plus the dilution
    cost on convergence.
    """

    def modifier(config: AppConfig) -> AppConfig:
        # Disable seller-side poisoning so the threat model is "pure economic"
        config.adversary_seller_config.poisoning.type = PoisonType.NONE
        config.adversary_seller_config.sybil.is_sybil = False
        # Activate the lazy seller subsystem
        lazy_cfg = config.adversary_seller_config.lazy_seller
        lazy_cfg.is_active = True
        lazy_cfg.strategy = strategy
        lazy_cfg.lazy_rate = float(lazy_rate)
        lazy_cfg.jitter_scale = float(jitter_scale)
        lazy_cfg.stale_period = int(stale_period)
        # Lazy sellers replace honest workers, so adv_rate is 0 in this regime.
        config.experiment.adv_rate = 0.0
        return config

    return modifier


def generate_sybil_selection_rate_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test a Sybil strategy aimed at maximizing selection rate.
    """
    scenarios = []
    # Test against defenses that are vulnerable to this
    AGGREGATORS = ['fedavg', 'fltrust', 'martfl']
    # See how the strategy performs as the number of Sybils increases
    ADV_RATES_TO_SWEEP = [0.1, 0.2, 0.3, 0.4]

    # --- Baseline: Poisoning attack WITHOUT Sybil coordination ---
    scenarios.append(Scenario(
        name="selection_rate_baseline_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, use_image_backdoor_attack],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS,
            "experiment.adv_rate": ADV_RATES_TO_SWEEP,
            "adversary_seller_config.poisoning.poison_rate": [0.3],
            "adversary_seller_config.sybil.is_sybil": [False],  # Baseline
        }
    ))

    # --- Test Scenario: The "Cluster Creation" Sybil strategy ---
    scenarios.append(Scenario(
        name="selection_rate_cluster_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[use_cifar10_config, use_image_backdoor_attack, use_sybil_attack('cluster')],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS,
            "experiment.adv_rate": ADV_RATES_TO_SWEEP,
            "adversary_seller_config.poisoning.poison_rate": [0.3],
        }
    ))
    return scenarios


# --- Adaptive Attack Modifier (Keep or adapt as needed) ---
def use_adaptive_attack(
        mode: str,
        threat_model: str = "black_box",
        exploration_rounds: int = 30,
        with_backdoor: bool = False  # Default to False if you want pure manipulation
) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier to enable the adaptive attacker.

    Args:
        with_backdoor: If True, injects a backdoor payload.
                       If False, acts as an untargeted/free-rider attacker.
    """

    def modifier(config: AppConfig) -> AppConfig:
        # 1. Activate Adaptive Logic
        adv_cfg = config.adversary_seller_config.adaptive_attack
        adv_cfg.is_active = True
        adv_cfg.attack_mode = mode
        adv_cfg.threat_model = threat_model
        adv_cfg.exploration_rounds = exploration_rounds

        # 2. Handle Poisoning vs Pure Manipulation
        if with_backdoor:
            # Goal: Inject Backdoor while evading detection
            config.adversary_seller_config.poisoning.type = PoisonType.IMAGE_BACKDOOR
            config.adversary_seller_config.poisoning.poison_rate = 0.5
        else:
            # Goal: Manipulate selection (Free Rider / Untargeted Noise)
            config.adversary_seller_config.poisoning.type = PoisonType.NONE
            # Even with no poison, we might want to scale gradients (free rider)
            # or add noise (untargeted degradation).

        # 3. Sync "Stealthy Blend" Params
        # If with_backdoor is False, stealthy_blend will just return 'honest',
        # effectively disabling it, so the bandit will learn to use
        # 'add_noise' or 'reduce_norm' instead.
        blend_cfg = config.adversary_seller_config.drowning_attack
        blend_cfg.mimicry_rounds = exploration_rounds
        blend_cfg.attack_intensity = 0.3
        blend_cfg.replacement_strategy = "layer_wise"

        # 4. Set Oracle/Inversion Params
        if threat_model in ["oracle", "gradient_inversion"]:
            adv_cfg.mimic_strength = 0.5

        # 5. Deactivate other standalone attacks
        config.adversary_seller_config.sybil.is_sybil = False

        return config

    return modifier


def use_drowning_attack(
        target_victim_id: str = "bn_3",
        attack_strength: float = 1.0
) -> Callable[[AppConfig], AppConfig]:
    """
    Enables the Targeted Drowning Attack via the SybilCoordinator.
    """

    def modifier(config: AppConfig) -> AppConfig:
        if not hasattr(config.adversary_seller_config, 'sybil'):
            config.adversary_seller_config.sybil = SybilConfig()

        # 1. Enable Sybil coordination
        config.adversary_seller_config.sybil.is_sybil = True

        # 2. Set the attack strategy to "drowning"
        config.adversary_seller_config.sybil.gradient_default_mode = "drowning"

        # 3. Configure the drowning attack parameters
        drowning_cfg = SybilDrowningConfig(
            victim_id=target_victim_id,
            attack_strength=attack_strength
        )
        if not config.adversary_seller_config.sybil.strategy_configs:
            config.adversary_seller_config.sybil.strategy_configs = {}
        config.adversary_seller_config.sybil.strategy_configs['drowning'] = drowning_cfg

        # 4. This attack does not use standard model poisoning
        config.adversary_seller_config.poisoning.type = PoisonType.NONE

        return config

    return modifier


def use_competitor_mimicry_attack(
        target_seller_id: str = "seller_0",
        strategy: str = "noisy_copy",
        noise_scale: float = 0.03,
        observation_rounds: int = 5
) -> Callable[[AppConfig], AppConfig]:
    """
    Returns a modifier to enable the Competitor Mimicry attack.
    This attack is simpler and more effective than the drowning attack.

    Args:
        target_seller_id: Which seller to mimic (steal market share from)
        strategy: "exact_copy", "noisy_copy", "scaled_copy", or "averaged_history"
        noise_scale: Amount of noise for noisy_copy/scaled_copy strategies
        observation_rounds: Rounds to observe target before attacking
    """

    def modifier(config: AppConfig) -> AppConfig:
        # Activate the mimicry attack
        adv_cfg = config.adversary_seller_config.mimicry_attack
        adv_cfg.is_active = True
        adv_cfg.target_seller_id = target_seller_id
        adv_cfg.strategy = strategy
        adv_cfg.noise_scale = noise_scale
        adv_cfg.observation_rounds = observation_rounds

        # Deactivate other attacks
        config.adversary_seller_config.poisoning.type = PoisonType.NONE
        config.adversary_seller_config.sybil.is_sybil = False
        config.adversary_seller_config.adaptive_attack.is_active = False
        config.adversary_seller_config.drowning_attack.is_active = False
        return config

    return modifier


def generate_competitor_mimicry_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test the CompetitorMimicrySeller's ability
    to steal market share from a high-quality target seller.

    Success Metric: Target's selection rate drops while attacker's increases.
    """
    scenarios = []
    AGGREGATORS_TO_TEST = ['fedavg', 'fltrust', 'martfl']
    MIMICRY_STRATEGIES = ["noisy_copy"]
    ADV_RATES = [0.2, 0.3, 0.4]  # See how multiple attackers amplify the effect

    for strategy in MIMICRY_STRATEGIES:
        scenarios.append(Scenario(
            name=f"competitor_mimicry_{strategy}_cifar10_cnn",
            base_config_factory=get_base_image_config,
            modifiers=[
                use_cifar10_config,
                use_competitor_mimicry_attack(
                    target_seller_id="bn_3",  # Target the first (often high-quality) seller
                    strategy=strategy,
                    noise_scale=0.03,
                    observation_rounds=5
                )
            ],
            parameter_grid={
                "experiment.image_model_config_name": ["cifar10_cnn"],
                "experiment.model_structure": ["cnn"],
                "aggregation.method": AGGREGATORS_TO_TEST,
                "experiment.adv_rate": ADV_RATES,
            }
        ))

    return scenarios


def generate_drowning_attack_scenarios() -> List[Scenario]:
    """
    [LEGACY] Generates scenarios to test the DrowningAttackerSeller.
    Kept for comparison purposes.
    """
    scenarios = []
    AGGREGATORS_TO_TEST = ['fedavg', 'fltrust', 'martfl']
    ADV_RATES_TO_SWEEP = [0.1, 0.2, 0.3, 0.4]

    scenarios.append(Scenario(
        name="drowning_attack_cifar10_cnn_legacy",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_drowning_attack(target_victim_id="bn_3", attack_strength=1.0)
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],
            "aggregation.method": AGGREGATORS_TO_TEST,
            "experiment.adv_rate": ADV_RATES_TO_SWEEP,
        }
    ))

    return scenarios


# ============================================================================
# 🆕 NEW: Enhanced Buyer Attack Scenarios
# ============================================================================

def use_buyer_dos_attack() -> Callable[[AppConfig], AppConfig]:
    """Enable buyer DoS attack (zero-gradient)"""

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "dos"
        return config

    return modifier


def use_buyer_starvation_attack(target_classes: List[int]) -> Callable[[AppConfig], AppConfig]:
    """Enable buyer starvation attack (focus on specific classes)"""

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "starvation"
        config.buyer_attack_config.starvation_classes = target_classes
        return config

    return modifier


def use_buyer_erosion_attack() -> Callable[[AppConfig], AppConfig]:
    """Enable buyer trust erosion attack (random gradients)"""

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "erosion"
        return config

    return modifier


def use_buyer_class_exclusion_attack(
        exclude_classes: List[int] = None,
        target_classes: List[int] = None,
        gradient_scale: float = 1.0
) -> Callable[[AppConfig], AppConfig]:
    """
    🆕 Enable buyer class-based exclusion attack.
    More realistic replacement for orthogonal_pivot.

    Args:
        exclude_classes: Classes to exclude from buyer baseline (negative selection)
        target_classes: Classes to include in buyer baseline (positive selection)
        gradient_scale: Amplification factor for bias (>1.0 increases bias)
    """

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "class_exclusion"

        if exclude_classes:
            config.buyer_attack_config.exclusion_exclude_classes = exclude_classes
        if target_classes:
            config.buyer_attack_config.exclusion_target_classes = target_classes

        config.buyer_attack_config.exclusion_gradient_scale = gradient_scale
        config.buyer_attack_config.num_classes = 10  # Adjust based on dataset
        return config

    return modifier


def use_buyer_oscillating_attack(
        strategy: str = "binary_flip",
        period: int = 2,
        classes_a: List[int] = None,
        classes_b: List[int] = None,
        subset_size: int = 3,
        drift_rounds: int = 50
) -> Callable[[AppConfig], AppConfig]:
    """
    🆕 Enable buyer oscillating objective attack.

    Args:
        strategy: "binary_flip", "rotating", "random_walk", or "adversarial_drift"
        period: Rounds before switching objectives (for binary_flip/rotating)
        classes_a: Phase A classes (for binary_flip/adversarial_drift)
        classes_b: Phase B classes (for binary_flip)
        subset_size: Number of random classes per round (for random_walk)
        drift_rounds: Total rounds for drift cycle (for adversarial_drift)
    """

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "oscillating"
        config.buyer_attack_config.oscillation_strategy = strategy
        config.buyer_attack_config.oscillation_period = period
        config.buyer_attack_config.num_classes = 10  # Adjust based on dataset

        if classes_a:
            config.buyer_attack_config.oscillation_classes_a = classes_a
        else:
            config.buyer_attack_config.oscillation_classes_a = [0, 1, 2, 3, 4]

        if classes_b:
            config.buyer_attack_config.oscillation_classes_b = classes_b
        else:
            config.buyer_attack_config.oscillation_classes_b = [5, 6, 7, 8, 9]

        config.buyer_attack_config.oscillation_subset_size = subset_size
        config.buyer_attack_config.oscillation_drift_total_rounds = drift_rounds

        return config

    return modifier


def use_buyer_orthogonal_pivot_attack(target_seller_id: str) -> Callable[[AppConfig], AppConfig]:
    """
    [LEGACY] Enable buyer orthogonal pivot attack.
    NOTE: Consider using class_exclusion_attack for more realistic exclusion.
    """

    def modifier(config: AppConfig) -> AppConfig:
        config.buyer_attack_config.is_active = True
        config.buyer_attack_config.attack_type = "orthogonal_pivot"
        config.buyer_attack_config.target_seller_id = target_seller_id
        return config

    return modifier


def generate_attack_scalability_scenarios_OLD() -> List[Scenario]:
    """
    Generates scenarios to test how attack effectiveness scales with marketplace size.
    Uses a FIXED adversary rate (30%) to maintain consistent attacker proportion.

    Key Research Questions:
    1. Does 30% attackers remain equally effective as the market grows?
    2. How does market size affect the absolute impact of the same proportion of attackers?
    3. Do defenses become more/less robust at larger scales?

    Tests both seller-side (Competitor Mimicry) and buyer-side (Class Exclusion) attacks.
    """
    scenarios = []

    # Sweep marketplace sizes from small (10 sellers) to large (100 sellers)
    # With adv_rate=0.3: 10 sellers = 3 attackers, 100 sellers = 30 attackers
    MARKETPLACE_SIZES = [10, 20, 30, 50, 100]

    # Fixed adversary rate across all scales
    FIXED_ADV_RATE = 0.3  # 30% of sellers are attackers

    # Focus on defenses most vulnerable to these attacks
    AGGREGATORS_TO_TEST = ['fltrust', 'martfl']

    # --- Scenario 1: Seller-Side Attack Scalability (Competitor Mimicry) ---
    scenarios.append(Scenario(
        name="scalability_competitor_mimicry_rate_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_competitor_mimicry_attack(
                target_seller_id="seller_0",  # Always target the first seller
                strategy="noisy_copy",
                noise_scale=0.03,
                observation_rounds=5
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            "experiment.adv_rate": [FIXED_ADV_RATE],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": AGGREGATORS_TO_TEST,

            # Keep these fixed for consistency
            "experiment.global_rounds": [100],
        }
    ))

    # --- Scenario 2: Buyer-Side Attack Scalability (Class Exclusion) ---
    scenarios.append(Scenario(
        name="scalability_buyer_class_exclusion_rate_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,  # Pure buyer attack, no seller interference
            use_buyer_class_exclusion_attack(
                exclude_classes=[7, 8, 9],  # Exclude sellers with these classes
                gradient_scale=1.2
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": AGGREGATORS_TO_TEST,

            # Keep these fixed
            "experiment.global_rounds": [100],
        }
    ))

    # --- Scenario 3: Oscillating Buyer Attack Scalability ---
    # This is interesting because buyer attacks should be scale-independent
    scenarios.append(Scenario(
        name="scalability_buyer_oscillating_rate_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_oscillating_attack(
                strategy="binary_flip",
                period=2,
                classes_a=[0, 1, 2, 3, 4],
                classes_b=[5, 6, 7, 8, 9]
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": AGGREGATORS_TO_TEST,

            "experiment.global_rounds": [100],
        }
    ))

    # --- Scenario 4: Combined Attack Scalability (Both attacks active) ---
    scenarios.append(Scenario(
        name="scalability_combined_attacks_rate_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            # Enable BOTH attacks simultaneously
            use_competitor_mimicry_attack(
                target_seller_id="seller_0",
                strategy="noisy_copy",
                noise_scale=0.03,
                observation_rounds=5
            ),
            use_buyer_class_exclusion_attack(
                exclude_classes=[7, 8, 9],
                gradient_scale=1.2
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Adversary Rate ---
            "experiment.adv_rate": [FIXED_ADV_RATE],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": AGGREGATORS_TO_TEST,

            "experiment.global_rounds": [100],
        }
    ))

    return scenarios


# ============================================================================
# 🆕 OPTIONAL: Multi-Rate Comparison (if you want to compare rates later)
# ============================================================================

def generate_attack_scalability_multirate_scenarios() -> List[Scenario]:
    """
    OPTIONAL: Compare how different adversary rates perform at multiple scales.
    Use this if you want to study the interaction between rate and scale.
    """
    scenarios = []

    MARKETPLACE_SIZES = [10, 30, 50, 100]  # Fewer sizes to keep experiments manageable
    ADVERSARY_RATES = [0.1, 0.2, 0.3, 0.4]  # Compare multiple rates

    scenarios.append(Scenario(
        name="scalability_multirate_competitor_mimicry_cifar10",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_competitor_mimicry_attack(
                target_seller_id="seller_0",
                strategy="noisy_copy",
                noise_scale=0.03,
                observation_rounds=5
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- SWEEP BOTH: Size and Rate ---
            "experiment.n_sellers": MARKETPLACE_SIZES,
            "experiment.adv_rate": ADVERSARY_RATES,

            # --- Focus on most vulnerable defense ---
            "aggregation.method": ["martfl"],

            "experiment.global_rounds": [100],
        }
    ))

    return scenarios


def generate_attack_scalability_scenarios() -> List[Scenario]:
    """
    Generates scenarios to test how attack effectiveness scales with marketplace size.
    Uses BACKDOOR attack for sellers (matching main_summary_figure style).
    Uses a FIXED adversary rate (30%) to maintain consistent attacker proportion.

    Key Research Questions:
    1. Does 30% backdoor attackers remain equally effective as the market grows?
    2. How does market size affect backdoor success rate (ASR)?
    3. Do defenses become more/less robust at larger scales?

    Tests both seller-side (Backdoor + Sybil) and buyer-side (Class Exclusion) attacks.
    """
    scenarios = []

    # Sweep marketplace sizes from small (10 sellers) to large (100 sellers)
    MARKETPLACE_SIZES = [10, 20, 30, 50, 100]

    # Fixed attack parameters (matching main_summary style)
    FIXED_ADV_RATE = 0.3  # 30% of sellers are attackers
    FIXED_POISON_RATE = 0.3  # 50% of attacker's data is poisoned

    # Focus on defenses most vulnerable to these attacks
    IMAGE_AGGREGATORS = ['fedavg', 'fltrust', 'martfl', 'skymask']

    # --- Scenario 1: Seller-Side Backdoor Attack Scalability ---
    scenarios.append(Scenario(
        name="scalability_backdoor_sybil_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_image_backdoor_attack,  # Image backdoor (10x10 white patch)
            use_sybil_attack('mimic')  # Sybil coordination to evade detection
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Attack Parameters ---
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": IMAGE_AGGREGATORS,
            "aggregation.skymask.sm_model_type": ["flexiblecnn"],  # For SkyMask

            # Keep these fixed for consistency
            "experiment.global_rounds": [100],
        }
    ))

    # --- Scenario 2: Backdoor Attack Scalability with ResNet (More Complex Model) ---
    scenarios.append(Scenario(
        name="scalability_backdoor_sybil_cifar10_resnet18",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_image_backdoor_attack,
            use_sybil_attack('mimic')
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_resnet18"],
            "experiment.model_structure": ["resnet18"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Attack Parameters ---
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": IMAGE_AGGREGATORS,
            "aggregation.skymask.sm_model_type": ["resnet18"],

            "experiment.global_rounds": [100],

        }
    ))

    # --- Scenario 3: Buyer-Side Attack Scalability (Class Exclusion) ---
    # This tests if buyer attacks remain scale-independent
    scenarios.append(Scenario(
        name="scalability_buyer_class_exclusion_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,  # Pure buyer attack, no seller interference
            use_buyer_class_exclusion_attack(
                exclude_classes=[7, 8, 9],  # Exclude sellers with these classes
                gradient_scale=1.2
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": IMAGE_AGGREGATORS,
            "aggregation.skymask.sm_model_type": ["flexiblecnn"],

            # Keep these fixed
            "experiment.global_rounds": [100],
        }
    ))

    # --- Scenario 4: Buyer Oscillating Attack Scalability ---
    scenarios.append(Scenario(
        name="scalability_buyer_oscillating_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_oscillating_attack(
                strategy="binary_flip",
                period=2,
                classes_a=[0, 1, 2, 3, 4],
                classes_b=[5, 6, 7, 8, 9]
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": IMAGE_AGGREGATORS,
            "aggregation.skymask.sm_model_type": ["flexiblecnn"],

            "experiment.global_rounds": [100],
        }
    ))

    # --- Scenario 5: Combined Attack Scalability (Backdoor + Class Exclusion) ---
    scenarios.append(Scenario(
        name="scalability_combined_backdoor_buyer_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_image_backdoor_attack,
            use_sybil_attack('mimic'),
            # ALSO enable buyer attack
            use_buyer_class_exclusion_attack(
                exclude_classes=[7, 8, 9],
                gradient_scale=1.2
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Seller Attack Parameters ---
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": IMAGE_AGGREGATORS,
            "aggregation.skymask.sm_model_type": ["flexiblecnn"],

            "experiment.global_rounds": [100],
        }
    ))

    return scenarios


# ============================================================================
# 🆕 OPTIONAL: Text Dataset Scalability (TREC)
# ============================================================================

def generate_text_scalability_scenarios() -> List[Scenario]:
    """
    Test scalability on text datasets (TREC) with text backdoor attack.
    """
    scenarios = []

    MARKETPLACE_SIZES = [10, 20, 30, 50]  # Fewer sizes for text (smaller dataset)
    TEXT_AGGREGATORS = ['fedavg', 'fltrust', 'martfl']  # No SkyMask for text

    scenarios.append(Scenario(
        name="scalability_backdoor_trec",
        base_config_factory=get_base_text_config,
        modifiers=[
            use_trec_config,
            use_text_backdoor_attack  # Text trigger (e.g., "cf" token)
        ],
        parameter_grid={
            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- FIXED: Attack Parameters ---
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.3],

            # --- SECONDARY SWEEP: Defense Method ---
            "aggregation.method": TEXT_AGGREGATORS,

            "experiment.global_rounds": [100],
        }
    ))

    return scenarios


# ============================================================================
# 🆕 NEW: Extreme Scale Test (Stress Testing)
# ============================================================================

def generate_extreme_scale_scenarios() -> List[Scenario]:
    """
    Stress test: Can backdoor attacks work in VERY large marketplaces (200-500 sellers)?
    Maintains 30% adversary rate even at extreme scales.
    """
    scenarios = []

    EXTREME_SIZES = [100, 200, 300, 500]
    FIXED_ADV_RATE = 0.3
    FIXED_POISON_RATE = 0.3

    # Test only the most vulnerable defense (faster experiments)
    scenarios.append(Scenario(
        name="extreme_scale_backdoor_martfl",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            use_image_backdoor_attack,
            use_sybil_attack('mimic')
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- Extreme marketplace sizes ---
            "experiment.n_sellers": EXTREME_SIZES,

            # --- Fixed: 30% adversary rate, 50% poison ---
            "experiment.adv_rate": [FIXED_ADV_RATE],
            "adversary_seller_config.poisoning.poison_rate": [FIXED_POISON_RATE],

            # --- Fixed: Most vulnerable defense ---
            "aggregation.method": ["martfl"],

            # Reduce rounds for faster experiments
            "experiment.global_rounds": [50],
        }
    ))

    scenarios.append(Scenario(
        name="extreme_scale_buyer_class_exclusion_fltrust",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks,
            use_buyer_class_exclusion_attack(
                exclude_classes=[7, 8, 9],
                gradient_scale=1.2
            )
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            "experiment.n_sellers": EXTREME_SIZES,
            "aggregation.method": ["fltrust"],

            "experiment.global_rounds": [50],
        }
    ))

    return scenarios


def generate_baseline_scalability_scenarios() -> List[Scenario]:
    """
    Test how defenses perform at different scales WITHOUT any attacks.
    Essential baseline for comparison.
    """
    scenarios = []

    MARKETPLACE_SIZES = [10, 20, 30, 50, 100]
    ALL_AGGREGATORS = ['fedavg', 'fltrust', 'martfl', 'skymask']

    scenarios.append(Scenario(
        name="scalability_baseline_no_attack_cifar10_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar10_config,
            disable_all_seller_attacks  # Pure benign
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar10_cnn"],
            "experiment.model_structure": ["cnn"],

            # --- PRIMARY SWEEP: Marketplace Size ---
            "experiment.n_sellers": MARKETPLACE_SIZES,

            # --- NO ATTACKS ---
            "experiment.adv_rate": [0.0],

            # --- Test all defenses ---
            "aggregation.method": ALL_AGGREGATORS,
            "aggregation.skymask.sm_model_type": ["flexiblecnn"],

            "experiment.global_rounds": [100],
        }
    ))

    return scenarios


# ============================================================================
# Add to ALL_SCENARIOS
# ============================================================================
def generate_cifar100_scalability_scenarios() -> List[Scenario]:
    """Test scalability on more complex dataset (CIFAR-100)"""
    scenarios = []

    MARKETPLACE_SIZES = [10, 30, 50, 100]  # Fewer sizes for efficiency

    scenarios.append(Scenario(
        name="scalability_backdoor_sybil_cifar100_cnn",
        base_config_factory=get_base_image_config,
        modifiers=[
            use_cifar100_config,  # Different dataset
            use_image_backdoor_attack,
            use_sybil_attack('mimic')
        ],
        parameter_grid={
            "experiment.image_model_config_name": ["cifar100_cnn"],
            "experiment.model_structure": ["cnn"],
            "experiment.n_sellers": MARKETPLACE_SIZES,
            "experiment.adv_rate": [0.3],
            "adversary_seller_config.poisoning.poison_rate": [0.3],
            "aggregation.method": ['fedavg', 'fltrust', 'martfl'],  # Fewer for speed
            "experiment.global_rounds": [100],
        }
    ))

    return scenarios


# Add to ALL_SCENARIOS
ALL_SCENARIOS = []
ALL_SCENARIOS.extend(generate_cifar100_scalability_scenarios())

# ... [Keep all your existing scenario generators] ...

# 🆕 Add scalability scenarios with BACKDOOR attacks
ALL_SCENARIOS.extend(generate_attack_scalability_scenarios())
# Add to ALL_SCENARIOS:
ALL_SCENARIOS.extend(generate_baseline_scalability_scenarios())

# 🆕 OPTIONAL: Add text scalability
ALL_SCENARIOS.extend(generate_text_scalability_scenarios())

# 🆕 Add extreme scale testing
ALL_SCENARIOS.extend(generate_extreme_scale_scenarios())

if __name__ == '__main__':
    print(f"Generated {len(ALL_SCENARIOS)} focused scenarios:")
    for s in ALL_SCENARIOS:
        print(f"  - {s.name}")
