import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, List, Union, Callable, Literal

from torch.utils.data import Dataset

from src.common_utils.constants.enums import TextTriggerLocation, ImageTriggerType, ImageTriggerLocation, PoisonType, \
    LabelFlipMode, \
    ImageBackdoorAttackName, TextBackdoorAttackName

logger = logging.getLogger("Configs")


@dataclass
class MimicryAttackConfig:
    """Configuration for Direct Competitor Mimicry Attack"""
    is_active: bool = False
    target_seller_id: str = "bn_3"  # Which seller to mimic
    observation_rounds: int = 3  # How many rounds to observe before attacking
    noise_scale: float = 0.05  # Noise level for noisy_copy strategy
    strategy: str = "noisy_copy"  # Options: "exact_copy", "noisy_copy", "scaled_copy", "averaged_history"


@dataclass
class DebugConfig:
    """Configuration for debugging and detailed logging."""
    save_individual_gradients: bool = False
    gradient_save_frequency: int = 10  # Save every round by default if enabled


@dataclass
class ExperimentConfig:
    """Primary configuration for the overall experiment run."""
    dataset_name: str
    model_structure: str
    global_rounds: int
    n_sellers: int
    adv_rate: float
    device: str
    eval_frequency: int = 10
    compute_gradient_similarity: bool = True  # Add this line
    save_path: str = "./results"
    num_classes: int = 0  # Default to 0, will be set dynamically at runtime
    use_subset: bool = False
    subset_size: int = 3000  # Number of samples to use in the subset
    dataset_type: str = "text"
    evaluations: List[str] = field(default_factory=lambda: ["clean"])
    image_model_config_name: str = "cifar10_cnn"
    tabular_model_config_name: str = "mlp_texas100_baseline"  # Default model config to use
    use_early_stopping: bool = True
    patience: int = 10
    eval_batches: int = 10  # Number of validation batches for quick in-round evaluation


@dataclass
class TrainingConfig:
    """Holds parameters specifically for local seller training."""
    local_epochs: int = 2  # Common default for FL
    batch_size: int = 64  # Common default batch size
    learning_rate: float = 0.001  # Sensible default, especially if Adam is common
    optimizer: str = "Adam"  # Default optimizer
    momentum: float = 0.9  # Only used by SGD
    weight_decay: float = 0.0001
    eps: float = 1e-4


@dataclass
class BackdoorSimpleDataPoisonParams:
    target_label: int = 0
    trigger_type: ImageTriggerType = ImageTriggerType.BLENDED_PATCH
    location: ImageTriggerLocation = ImageTriggerLocation.BOTTOM_RIGHT
    trigger_shape: Tuple[int, int] = (10, 10)
    strength: float = 1
    pattern_channel: int = 3


@dataclass
class ImageBackdoorParams:
    """Container for all possible image backdoor attack configurations."""
    # This field determines which of the sub-configurations is active
    attack_name: ImageBackdoorAttackName = ImageBackdoorAttackName.SIMPLE_DATA_POISON
    # It holds an instance of every possible sub-configuration
    simple_data_poison_params: BackdoorSimpleDataPoisonParams = field(default_factory=BackdoorSimpleDataPoisonParams)

    @property
    def active_attack_params(self) -> Union[BackdoorSimpleDataPoisonParams]:
        """
        Returns the active parameter object based on the 'attack_name' field.
        """
        if self.attack_name == ImageBackdoorAttackName.SIMPLE_DATA_POISON:
            return self.simple_data_poison_params
        raise ValueError(f"Unknown image backdoor attack name: {self.attack_name}")


@dataclass
class TextBackdoorParams:
    """Parameters specific to a TEXT backdoor attack."""
    target_label: int = 0
    trigger_content: str = "cf"  # The trigger phrase
    location: TextTriggerLocation = TextTriggerLocation.END
    attack_name: TextBackdoorAttackName = TextBackdoorAttackName.SIMPLE_DATA_POISON


@dataclass
class BuyerAttackConfig:
    """Configuration for malicious buyer attacks"""
    is_active: bool = False
    attack_type: str = "none"  # "dos", "starvation", "erosion", "class_exclusion", "oscillating", "orthogonal_pivot"
    target_seller_id: str = "bn_5"
    # --- Starvation Attack ---
    starvation_classes: List[int] = field(default_factory=list)

    # --- 🆕 Class-Based Exclusion ---
    exclusion_target_classes: List[int] = field(default_factory=list)
    exclusion_exclude_classes: List[int] = field(default_factory=list)
    exclusion_gradient_scale: float = 1.0

    # --- 🆕 Oscillating Objective ---
    oscillation_strategy: str = "binary_flip"  # "binary_flip", "rotating", "random_walk", "adversarial_drift"
    oscillation_period: int = 2
    oscillation_classes_a: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    oscillation_classes_b: List[int] = field(default_factory=lambda: [5, 6, 7, 8, 9])
    oscillation_class_subsets: List[List[int]] = field(default_factory=lambda: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
    num_classes: int = 10  # 🆕 ADD THIS (should match dataset)
    oscillation_subset_size: int = 3
    oscillation_drift_total_rounds: int = 50


@dataclass
class TabularFeatureTriggerParams:
    """Parameters for a feature-based trigger backdoor attack."""
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)


# --- Step 2: Create the container, just like for image and text ---

class TabularBackdoorAttackName(Enum):
    """Enum to define the different types of tabular backdoor attacks."""
    FEATURE_TRIGGER = "feature_trigger"


@dataclass
class TabularBackdoorParams:
    """Container for all possible tabular backdoor attack configurations."""
    # This field determines which of the sub-configurations is active
    attack_name: TabularBackdoorAttackName = TabularBackdoorAttackName.FEATURE_TRIGGER
    target_label: int = 0

    # It holds an instance of every possible sub-configuration
    feature_trigger_params: TabularFeatureTriggerParams = field(default_factory=TabularFeatureTriggerParams)

    @property
    def active_attack_params(self) -> Union[TabularFeatureTriggerParams]:
        """
        Returns the active parameter object based on the 'attack_name' field.
        """
        if self.attack_name == TabularBackdoorAttackName.FEATURE_TRIGGER:
            return self.feature_trigger_params
        raise ValueError(f"Unknown tabular backdoor attack name: {self.attack_name}")


@dataclass
class LabelFlipParams:
    """Parameters specific to the label-flipping poisoning attack."""
    target_label: int = 0
    mode: LabelFlipMode = LabelFlipMode.FIXED_TARGET


@dataclass
class PoisoningConfig:
    """Configuration for client-side data poisoning attacks."""
    type: PoisonType = PoisonType.NONE
    poison_rate: float = 0.1
    image_backdoor_params: ImageBackdoorParams = field(default_factory=ImageBackdoorParams)
    text_backdoor_params: TextBackdoorParams = field(default_factory=TextBackdoorParams)
    tabular_backdoor_params: TabularBackdoorParams = field(default_factory=TabularBackdoorParams)
    label_flip_params: LabelFlipParams = field(default_factory=LabelFlipParams)

    @property
    def active_params(self) -> Union[  # Add all possible return types here
        BackdoorSimpleDataPoisonParams,
        TextBackdoorParams,
        TabularBackdoorParams,
        LabelFlipParams,
        None
    ]:
        """
        Returns the active parameter object based on the main 'type' field.
        """
        if self.type == PoisonType.IMAGE_BACKDOOR:
            return self.image_backdoor_params.active_attack_params

        elif self.type == PoisonType.TEXT_BACKDOOR:
            return self.text_backdoor_params

        elif self.type == PoisonType.TABULAR_BACKDOOR:
            return self.tabular_backdoor_params

        elif self.type == PoisonType.LABEL_FLIP:
            return self.label_flip_params

        elif self.type == PoisonType.NONE:
            return None

        raise ValueError(f"Unknown poison type: {self.type}")


@dataclass
class GradientInversionParams:
    """Parameters for the Gradient Inversion Attack."""
    frequency: int = 10
    victim_strategy: str = 'random'
    fixed_victim_idx: int = 0
    lrs_to_try: List[float] = field(default_factory=lambda: [0.1, 0.01])
    base_attack_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerAttackConfig:
    """Configuration for all server-side privacy attacks."""
    # This is the main switch: 'none', 'gradient_inversion', etc.
    attack_name: str = 'none'

    gradient_inversion_params: GradientInversionParams = field(default_factory=GradientInversionParams)


@dataclass
class DrowningAttackConfig:
    """
    Configuration for the 'stealthy_blend' attack used by AdaptiveAttackerSeller.
    """
    is_active: bool = False

    # --- Parameters used by AdaptiveAttackerSeller __init__ ---

    # Number of rounds to act honestly and learn honest gradient stats
    mimicry_rounds: int = 10

    # [THE FIX] The *initial* intensity for the blend attack (will be adapted later)
    attack_intensity: float = 0.1

    # List of layer names to target (e.g., "fc.weight"). If empty, it targets top 20%.
    target_layers: List[str] = field(default_factory=list)

    # --- Parameters used by the stealthy_blend strategy ---

    # The type of malicious gradient to compute
    attack_type: Literal["backdoor", "targeted_poisoning"] = "backdoor"

    # How to blend the malicious gradient with the honest one
    replacement_strategy: Literal["layer_wise", "global_blend"] = "global_blend"

    # --- Note ---
    # The 'drift_factor' field from your old config is not used by
    # the AdaptiveAttackerSeller and has been removed to avoid confusion.


@dataclass
class SybilDrowningConfig:
    """Configuration for the Targeted Drowning Attack."""
    victim_id: str = "bn_3"
    attack_strength: float = 1.0


@dataclass
class SybilConfig:
    """Configuration for Sybil attack coordination and behavior."""
    is_sybil: bool = False  # Is Sybil coordination active?
    benign_rounds: int = 0  # Rounds before Sybils start manipulating (0 = immediate attack)

    # --- Strategy Control ---
    # Specifies the primary manipulation strategy used by Sybils.
    # Options: "mimic", "pivot", "knock_out", "oracle_blend", "systematic_probe"
    # If None, might fall back to role-based strategies (less common now).
    gradient_default_mode: Optional[str] = "mimic"

    # --- Oracle Blend Specific ---
    # Blending factor alpha for oracle_blend strategy.
    # G_submit = alpha * G_mal + (1 - alpha) * G_oracle_centroid
    oracle_blend_alpha: float = 0.1  # Example: 10% malicious intent

    # --- Historical Mimicry Specific ---
    # Number of past rounds to consider for calculating historical centroid.
    history_window_size: int = 5  # Reduced default example

    # --- Role Assignment (Less critical if gradient_default_mode is set) ---
    # Defines proportions for dynamic role assignment (attacker, hybrid, explorer).
    # Used if adaptive_role_assignment is called and gradient_default_mode is None.
    role_config: Dict[str, float] = field(default_factory=lambda: {'attacker': 0.2, 'explorer': 0.4})

    # --- Strategy-Specific Parameters ---
    # Holds keyword arguments for specific strategy classes (e.g., MimicStrategy).
    strategy_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- Triggering (Keep if used elsewhere) ---
    # Controls how/when the attack activates (e.g., 'static' after benign_rounds).
    trigger_mode: str = "static"

    # --- REMOVED ---
    detection_threshold: float = 0.8  # No longer needed with simplified phase logic


@dataclass
class VocabConfig:
    """Configuration for building the torchtext vocabulary."""
    min_freq: int = 1
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"


@dataclass
class PropertySkewParams:
    """Parameters for the 'property-skew' image data splitting strategy."""
    property_key: str = "Blond_Hair"
    num_high_prevalence_clients: int = 2
    num_security_attackers: int = 2
    high_prevalence_ratio: float = 0.8
    low_prevalence_ratio: float = 0.1
    standard_prevalence_ratio: float = 0.4
    dirichlet_alpha: float = 1.0  # <--- ADD THIS LINE


@dataclass
class TextPropertySkewParams:
    """Parameters for the 'property-skew' text data splitting strategy."""
    property_key: str = "software"  # The keyword/phrase to search for
    num_high_prevalence_clients: int = 2
    num_security_attackers: int = 2
    high_prevalence_ratio: float = 0.8
    low_prevalence_ratio: float = 0.1
    standard_prevalence_ratio: float = 0.4


@dataclass
class AdaptiveAttackConfig:
    """Configuration for the adaptive learning attacker."""
    is_active: bool = False

    # --- General Learning Parameters ---

    # Set a valid default and use Literal for type safety
    threat_model: Literal["black_box", "gradient_inversion", "oracle"] = "black_box"

    # The primary mode of attack
    attack_mode: Literal["gradient_manipulation", "data_poisoning"] = "gradient_manipulation"

    exploration_rounds: int = 20

    # --- Threat Model Specific ---

    # [FIX] ADDED: Required by Oracle and Gradient Inversion models
    mimic_strength: float = 0.5

    # --- Gradient Manipulation Strategies ---

    # [FIX] RENAMED "scale_up" to "reduce_norm" to match your class
    gradient_strategies: List[str] = field(
        default_factory=lambda: ["honest", "reduce_norm", "add_noise", "stealthy_blend"]
    )
    scale_factor: float = 0.5  # A value < 1.0 matches "reduce_norm"
    noise_level: float = 0.01

    # --- Data Poisoning Strategies ---

    data_strategies: List[str] = field(
        default_factory=lambda: ["honest", "subsample_clean"]
    )

    # [FIX] ADDED: Required by "subsample_clean" strategy
    subset_ratio: float = 0.5


@dataclass
class LazySellerConfig:
    """Configuration for free-riding (lazy) sellers.

    A lazy seller fakes participation: it submits a gradient that looks
    plausible enough to pass filtering defenses, but skips real local
    training (saving compute cost). Strategies vary in stealth/power.

    Strategies:
      - "none": disabled (default)
      - "norm_noise": Gaussian noise rescaled to median benign L2 norm.
        Bypasses pure norm-checking defenses (Trimmed Mean, DAVED magnitude
        gate alone) but defeated by cosine-based defenses.
      - "echo": Replay the previous round's aggregated global delta plus
        small jitter. Achieves ~1.0 cosine similarity with consensus —
        devastating against FLTrust/SPMC/MartFL/RFLPA/DAVED-FWIF, only
        FoolsGold (pairwise sim penalty) catches it cleanly.
      - "stale_local": Train once, cache the gradient, replay for
        ``stale_period`` rounds with light jitter. Realistic and bypasses
        anomaly detectors that flag bit-identical updates across rounds.
    """
    is_active: bool = False
    strategy: str = "none"  # "none" | "norm_noise" | "echo" | "stale_local"
    lazy_rate: float = 0.3   # Fraction of sellers that are lazy
    jitter_scale: float = 1e-3  # Multiplicative noise added to faked gradients
    stale_period: int = 10   # For "stale_local": rounds to reuse a cached grad
    target_norm: float = 0.0  # 0.0 = match median honest norm dynamically


@dataclass
class AdversarySellerConfig:
    """A unified profile for an adversarial seller."""
    name: str = "benign"
    poisoning: PoisoningConfig = field(default_factory=PoisoningConfig)
    sybil: SybilConfig = field(default_factory=SybilConfig)
    adaptive_attack: AdaptiveAttackConfig = field(default_factory=AdaptiveAttackConfig)
    drowning_attack: DrowningAttackConfig = field(default_factory=DrowningAttackConfig)
    mimicry_attack: MimicryAttackConfig = field(default_factory=MimicryAttackConfig)  # <-- Add this
    lazy_seller: 'LazySellerConfig' = field(default_factory=lambda: LazySellerConfig())
    # Bagdasaryan-style model-replacement gradient scaling. Multiplies the
    # adversary's post-training gradient by this factor before upload. Values
    # > 1 implement the classic FL backdoor boosting attack from Bagdasaryan
    # et al. 2020 ("How to Backdoor Federated Learning"). 1.0 means no
    # scaling (normal data-poisoning attack). Can also be overridden at
    # runtime via the BAGDASARYAN_SCALE_FACTOR env var so the same configs
    # can be reused for adaptive-attack runs without regenerating.
    scale_factor: float = 1.0


@dataclass
class TextDataConfig:
    """All settings related to a text dataset source."""
    vocab: VocabConfig = VocabConfig()
    strategy: str = "dirichlet"  # Strategy for SELLERS (Changed default)
    dirichlet_alpha: float = 0.5  # Alpha for SELLERS (Added)

    property_skew: Optional[TextPropertySkewParams] = None  # Keep if used

    # --- Buyer Specific Params ---
    buyer_ratio: float = 0.1  # Renamed from buyer_config for consistency
    buyer_strategy: str = "iid"  # Strategy for BUYER ('iid', 'dirichlet')
    buyer_dirichlet_alpha: Optional[float] = None  # Alpha ONLY for buyer if buyer_strategy='dirichlet'


# --- Image Data Configuration ---
@dataclass
class ImageDataConfig:
    """All settings related to an image dataset source."""
    strategy: str = "dirichlet"  # Strategy for SELLERS (Defaulting to Dirichlet)
    dirichlet_alpha: float = 0.5  # Alpha for SELLERS
    property_skew: Optional[PropertySkewParams] = None  # Keep if you use property skew sometimes

    # --- Buyer Specific Params ---
    buyer_ratio: float = 0.1
    buyer_strategy: str = "iid"  # Strategy for BUYER ('iid', 'dirichlet')
    buyer_dirichlet_alpha: Optional[float] = None  # Alpha ONLY for buyer if buyer_strategy='dirichlet'


# --- Tabular Data Configuration ---
@dataclass
class LoRAConfig:
    """LoRA adapter configuration for LLM finetuning marketplace."""
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    use_qlora: bool = True          # 4-bit quantized base model
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class SoftPromptConfig:
    """Soft prompt (FLiP-style) configuration."""
    num_virtual_tokens: int = 20
    prompt_tuning_init: str = "TEXT"   # "TEXT" or "RANDOM"
    prompt_tuning_init_text: str = "Classify the following text: "


@dataclass
class LLMAttackConfig:
    """Configuration for LLM-specific attacks (GAP and Alignment Poisoning)."""
    attack_type: str = "none"  # none | gap | alignment_degradation | alignment_refusal

    # GAP attack settings
    gap_stealth_budget: float = 1.0
    gap_projection_steps: int = 50
    gap_poison_epochs: int = 5
    gap_scale_to_benign_norm: bool = True          # Ablation: norm camouflage
    gap_use_constrained_factorization: bool = True  # Ablation: False = naive SVD (no projection)
    gap_target_layers: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj",
    ])  # Ablation: ["q_proj"] or ["v_proj"] to test which layer carries the attack

    # Alignment poisoning settings
    alignment_poison_fraction: float = 0.5
    alignment_invert_preferences: bool = True
    alignment_scale_to_benign_norm: bool = True     # Ablation: norm camouflage
    alignment_refusal_targets: List[str] = field(default_factory=lambda: [
        "programmer", "engineer", "developer",
    ])
    alignment_refusal_template_count: int = 3       # Ablation: 1 vs 3 templates


@dataclass
class LLMDataConfig:
    """All settings related to an LLM finetuning dataset."""
    # Dataset source
    dataset_source: str = "fed_chatbot_it"  # fed_chatbot_it | fed_wildchat | fed_chatbot_pa
    hf_dataset_path: str = ""               # HuggingFace dataset path override

    # Preprocessing
    max_seq_length: int = 512
    min_samples_per_seller: int = 10        # Filter sellers with < N samples
    chat_template: str = "default"          # default | chatml | llama

    # Task type
    task: str = "sft"                       # sft | dpo

    # Partitioning (seller data is naturally partitioned by user_id)
    buyer_ratio: float = 0.1               # Fraction of users reserved for buyer
    buyer_strategy: str = "iid"

    # PEFT method
    peft_method: str = "lora"              # lora | soft_prompt
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    soft_prompt: SoftPromptConfig = field(default_factory=SoftPromptConfig)

    # Model
    base_model_name: str = "Qwen/Qwen2.5-1.5B"  # HuggingFace model ID
    torch_dtype: str = "bfloat16"

    # LLM-specific attacks
    llm_attack: LLMAttackConfig = field(default_factory=LLMAttackConfig)


@dataclass
class TabularDataConfig:
    """All settings related to a tabular dataset source."""
    dataset_config_path: str = "configs/tabular_datasets.yaml"
    model_config_dir: str = "src/model/configs"

    strategy: str = "dirichlet"  # Strategy for SELLERS (Changed default from 'iid')
    dirichlet_alpha: float = 0.5  # Alpha for SELLERS (Added, assuming you have this param)
    property_skew: Optional[PropertySkewParams] = None  # Keep if you use property skew sometimes

    # --- Buyer Specific Params ---
    buyer_ratio: float = 0.1
    buyer_strategy: str = "iid"  # Strategy for BUYER ('iid', 'dirichlet')
    buyer_dirichlet_alpha: Optional[float] = None  # Alpha ONLY for buyer if buyer_strategy='dirichlet'


# --- Main Data Configuration ---
@dataclass
class DataConfig:
    """Holds configuration for one type of data source."""
    text: Optional[TextDataConfig] = TextDataConfig()
    image: Optional[ImageDataConfig] = ImageDataConfig()
    tabular: Optional[TabularDataConfig] = TabularDataConfig()
    llm: Optional[LLMDataConfig] = None
    num_workers: int = 4  # Keep num_workers here if it applies globally


@dataclass
class RuntimeDataConfig:
    """Holds runtime data objects passed to sellers."""
    dataset: Dataset
    num_classes: int
    collate_fn: Optional[Callable] = None
    dataset_type: str = "standard"  # "standard" (image/text/tabular) or "llm"
    llm_task: Optional[str] = None  # "sft" or "dpo" when dataset_type == "llm"


# --- These are RUNTIME configs, not loaded from YAML. Keeping them is correct. ---
@dataclass
class LabelFlipConfig:
    """Configuration for the LabelFlipGenerator."""
    num_classes: int
    attack_mode: str = "fixed_target"
    target_label: int = 0


@dataclass
class BackdoorImageConfig:
    """Configuration for an image backdoor trigger.

    Defaults target CIFAR-style RGB images (32x32 or larger). For FEMNIST,
    the BackdoorImageGenerator auto-detects 28x28 input at apply time and
    swaps in a stronger CHECKERBOARD/alpha=0.8/4x4 trigger — otherwise the
    default BLENDED_PATCH (white) vanishes against FEMNIST's white background
    and ASR drops to random baseline.
    """
    target_label: int
    trigger_type: ImageTriggerType = ImageTriggerType.BLENDED_PATCH
    channels: int = 3
    trigger_size: Tuple[int, int] = (5, 5)
    blend_alpha: float = 0.2
    location: ImageTriggerLocation = ImageTriggerLocation.BOTTOM_RIGHT
    randomize_location: bool = False


@dataclass
class BackdoorTextConfig:
    """Configuration for the BackdoorTextGenerator."""
    vocab: Any  # Expects an object with a .stoi attribute
    target_label: int
    trigger_content: str = "cf"
    location: TextTriggerLocation = TextTriggerLocation.END
    max_seq_len: Optional[int] = None


@dataclass
class BackdoorTabularConfig:
    """Configuration for a feature-based tabular backdoor attack."""
    target_label: int
    # Defines the trigger, e.g., {'feature_name_A': 1.0, 'feature_name_B': 0.0}
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MartFLParams:
    """Parameters specific to the martFL aggregator."""
    # --- Existing Parameters ---
    change_base: bool = True
    # If True, rotates baseline seller based on kappa score.
    # If False, always uses the initial_baseline (like FLTrust).

    clip: bool = True
    # If True, uses server-side clipping with the clip_norm value
    # defined in the main AggregationConfig before similarity calculation.

    # --- New Tunable Parameters ---
    initial_baseline: str = "buyer"
    # Specifies the ID used as the baseline anchor in the *first* round
    # (and subsequent rounds if change_base is False).
    # Common values: "buyer" or a specific client ID like "bn_3".

    max_k: int = 10
    # The maximum number of clusters ('k') that the k-means algorithm
    # (in _cluster_and_score_martfl) will consider when detecting outliers.


@dataclass
class SkymaskParams:
    """Parameters specific to the Skymask aggregator."""
    clip: bool = True
    sm_model_type: str = 'cnn'
    mask_epochs: int = 20
    mask_lr: float = 1e-4
    mask_clip: float = 1.0
    mask_threshold: float = 0.5


@dataclass
class TrimmedMeanParams:
    """Parameters for the Trimmed Mean aggregator."""
    trim_ratio: float = 0.1  # Fraction to trim from each end (e.g., 0.1 = 10%)


@dataclass
class MultiKrumParams:
    """Parameters for the Multi-Krum aggregator."""
    num_byzantine: int = 0   # Estimated number of malicious sellers (f)
    m_selected: int = 1      # Number of sellers to select and average (1 = standard Krum)


@dataclass
class RFLPAParams:
    """Parameters for the RFLPA aggregator (NeurIPS 2024)."""
    element_clip: float = 1.0  # Element-wise clipping bound [-c, c]


@dataclass
class SPMCParams:
    """Parameters for the SPMC MCAgg aggregator (ICML 2025)."""
    temperature: float = 0.5  # Softmax temperature — lower = stricter penalization


@dataclass
class DAVEDParams:
    """Parameters for the DAVED + FWIF aggregator."""
    head_layer_indices: List[int] = field(default_factory=lambda: [-2, -1])
    l2_relative_alpha: float = 0.1   # magnitude gate: reject if norm < alpha * median
    proj_lambda: float = 1.0         # projection strength for angle gate


@dataclass
class FoolsGoldParams:
    """Parameters for the FoolsGold aggregator (arXiv 2018)."""
    pardon_threshold: float = 0.5  # Sellers with max similarity below this are pardoned (weight=1.0)


@dataclass
class BulyanParams:
    """Parameters for the Bulyan aggregator (ICML 2018)."""
    num_byzantine: int = 1  # Expected max number of malicious sellers (f). Requires N >= 4f+3.


@dataclass
class DeepSightParams:
    """Parameters for the DeepSight aggregator (NDSS 2022)."""
    threshold_c: float = 2.0          # Std devs for density threshold (higher = more permissive)
    neighbor_ratio: float = 0.5        # Fraction of sellers for density estimation


@dataclass
class FLAMEParams:
    """Parameters for the FLAME aggregator (USENIX Security 2022)."""
    noise_scale: float = 0.001          # DP noise multiplier (lambda)
    min_cluster_size_ratio: float = 0.5  # HDBSCAN min_cluster_size as fraction of N


@dataclass
class FLTrustThresholdParams:
    """Parameters for the thresholded-FLTrust selection-integrity baseline (E1b).

    Runs FLTrust's trust-score computation unchanged, then converts it into an
    explicit accept/reject decision: seller i is accepted (R_i = 1) iff
    trust_score_i >= tau. Accepted sellers are aggregated with FLTrust's
    trust-proportional weights renormalized over the accepted set; rejected
    sellers contribute nothing and are unpaid.
    """
    tau: float = 0.5  # Trust-score acceptance threshold in [0, 1]


@dataclass
class AggregationConfig:
    """Top-level configuration for aggregation."""
    # This is the main switch to select the algorithm
    method: str = "martfl"

    # A common parameter used by multiple methods
    clip_norm: Optional[float] = 0.01

    # --- Nested parameter objects for each strategy ---
    martfl: MartFLParams = field(default_factory=MartFLParams)
    skymask: SkymaskParams = field(default_factory=SkymaskParams)
    trimmed_mean: TrimmedMeanParams = field(default_factory=TrimmedMeanParams)
    multi_krum: MultiKrumParams = field(default_factory=MultiKrumParams)
    rflpa: RFLPAParams = field(default_factory=RFLPAParams)
    spmc: SPMCParams = field(default_factory=SPMCParams)
    daved: DAVEDParams = field(default_factory=DAVEDParams)
    flame: FLAMEParams = field(default_factory=FLAMEParams)
    deepsight: DeepSightParams = field(default_factory=DeepSightParams)
    bulyan: BulyanParams = field(default_factory=BulyanParams)
    foolsgold: FoolsGoldParams = field(default_factory=FoolsGoldParams)
    fltrust_threshold: FLTrustThresholdParams = field(default_factory=FLTrustThresholdParams)


@dataclass
class ValuationConfig:
    """Configuration for valuation methods."""

    # --- Default, Always-On Methods ---
    run_similarity: bool = True
    # ^ Note: This is for config clarity. The manager
    # will always run this cheap, default evaluator.

    # --- Optional Expensive Methods ---
    run_influence: bool = False  # Run fast influence function
    run_loo: bool = False  # Run per-round Leave-One-Out
    loo_frequency: int = 10  # Run LOO every N rounds

    # --- NEW KERNELSHAP (LINEAR MODEL) ---
    run_kernelshap: bool = False  # Run KernelSHAP approximation
    kernelshap_frequency: int = 10
    kernelshap_samples: int = 32  # Number of coalitions to sample (e.g., 2*N + 8)

    # --- E4: Banzhaf + Least Core (REUSE the KernelSHAP coalition samples) ---
    # Both estimators consume the (coalition, utility) pairs already sampled by
    # the KernelSHAP evaluator — no additional utility evaluations are run.
    # Enabling either without run_kernelshap still runs the coalition sampler
    # (at kernelshap_frequency / kernelshap_samples) but only emits the
    # requested scores.
    run_banzhaf: bool = False     # Banzhaf value via maximum-sample-reuse estimator
    run_leastcore: bool = False   # Least Core via Monte-Carlo constraint-sampled LP
    leastcore_max_coalitions: int = 2000  # Cap on LP coalition constraints
    leastcore_timeout: float = 600.0      # Hard wall-clock budget (sec) for the LP solve

    # --- Payment Model ---
    # "binary"        : fixed payment if selected, 0 otherwise
    # "proportional"  : payment proportional to aggregation weight
    #                   (budget-scaled: payments sum to BASE_PRICE * n_selected)
    # "quality_based" : payment proportional to oracle similarity (default)
    # "weighted"      : E1a — payment = the per-seller aggregation weight the
    #                   filter already computed this round, renormalized so
    #                   payments sum to 1 across paid sellers. Does NOT change
    #                   selection or aggregation; read-out only. For methods
    #                   with no per-seller weight, payment is undefined and the
    #                   round logs "payment_undefined".
    payment_model: str = "quality_based"


@dataclass
class DPConfig:
    """Local Differential Privacy configuration for benign sellers.

    Applied ONLY in the honest-seller upload path (adversarial sellers have
    white-box control and would not handicap themselves) — see
    SellerFactory, which passes dp_config exclusively to benign sellers.
    """
    enabled: bool = False
    mechanism: str = "gaussian"       # "gaussian" or "laplacian"
    epsilon: float = 1.0              # privacy budget (smaller = more private)
    delta: float = 1e-5               # failure probability (for gaussian)
    clip_norm: float = 1.0            # per-sample gradient clip before adding noise
    # sigma calibration for the Gaussian mechanism:
    #   "classic"  : sigma = C * sqrt(2 ln(1.25/delta)) / eps  (Dwork & Roth;
    #                only valid for eps <= 1, over/under-noises outside that range)
    #   "analytic" : Analytic Gaussian Mechanism (Balle & Wang, ICML 2018) —
    #                exact (eps, delta) calibration valid for ALL eps > 0.
    #                Recommended for the E3 sweep (eps in {1, 4, 8}).
    calibration: str = "classic"


@dataclass
class MultiTaskConfig:
    """E2 — multi-round persistent market (outer task loop over the existing run).

    Only read by experiments/gradient_market/run_multi_task.py; the single-task
    pipeline ignores this section entirely. Reputation/payment state collected
    across tasks is a MEASURED OUTPUT ONLY and never feeds back into selection
    or aggregation.
    """
    num_tasks: int = 5          # T sequential buyer queries over the same population
    exit_k: int = 3             # benign seller leaves after k consecutive unpaid rounds
    carry_model: str = "reset"  # "reset" (fresh global model per task) or "continue"
    bandit_persist: bool = False  # let AdaptiveAttackerSeller's learned state survive across tasks
    # Per-task buyer query: D_B restricted to class subset C_t (reuses the
    # Dynamic-Instability class-filter machinery). None = auto-rotation:
    # the dataset's classes are split into num_tasks contiguous blocks and
    # task t uses block t.
    task_class_subsets: Optional[List[List[int]]] = None


@dataclass
class AppConfig:
    """Top-level configuration for the entire experiment run."""
    experiment: ExperimentConfig
    training: TrainingConfig
    server_attack_config: ServerAttackConfig
    adversary_seller_config: AdversarySellerConfig
    data: DataConfig
    debug: DebugConfig
    aggregation: AggregationConfig
    seed: int = 42
    n_samples: int = 3
    data_root: str = "./data"
    use_cache: bool = True
    buyer_attack_config: BuyerAttackConfig = field(default_factory=BuyerAttackConfig)
    valuation: ValuationConfig = field(default_factory=ValuationConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    multi_task: MultiTaskConfig = field(default_factory=MultiTaskConfig)


# SybilDrowningConfig is defined above (line ~258) — duplicate removed.
