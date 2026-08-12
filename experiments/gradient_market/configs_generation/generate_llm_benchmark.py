"""
Generate LLM Marketplace Benchmark Configs
============================================

Generates experiment configurations for federated LLM finetuning marketplace.
Tests all aggregation defenses with LoRA and Soft Prompt PEFT methods.

Ensures the four mandatory baselines are always created for every dataset:
  a) Clean Market + FedAvg       (upper-bound utility)
  b) Attacked Market + FedAvg    (attack proof / lower-bound security)
  c) Clean Market + Defenses     (false-positive / over-defending check)
  d) Attacked Market + Defenses  (core evaluation)

Usage:
    python experiments/gradient_market/configs_generation/generate_llm_benchmark.py

Output:
    configs_generated_benchmark/llm_benchmark/
"""

import copy
import sys
from pathlib import Path
from typing import List

from experiments.gradient_market.automate_exp.base_llm_config import (
    get_base_llm_sft_config, get_base_llm_dpo_config,
)

try:
    from src.marketplace.utils.gradient_market_utils.gradient_market_configs import (
        AppConfig, LLMAttackConfig,
    )
    from experiments.gradient_market.automate_exp.config_generator import ExperimentGenerator
    from experiments.gradient_market.automate_exp.scenarios import Scenario
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

# Models to benchmark (base_model_name, short_name)
MODELS = [
    ("Qwen/Qwen2.5-1.5B", "qwen1.5b"),
    # ("meta-llama/Llama-3.2-3B", "llama3b"),  # Uncomment for larger model
]

# Datasets
DATASETS = [
    ("fed_chatbot_it", "sft"),
    # ("fed_wildchat", "sft"),     # Uncomment to add
    ("fed_chatbot_pa", "dpo"),
]

# PEFT methods
PEFT_METHODS = ["lora"]  # Add "soft_prompt" to test FLiP

# Defenses to test (non-FedAvg)
DEFENSES_NON_FEDAVG = [
    "fltrust",
    "trimmed_mean",
    "multi_krum",
    "rflpa",
    "spmc",
]

ALL_DEFENSES = ["fedavg"] + DEFENSES_NON_FEDAVG

# Attack configurations: (attack_type, adv_rate, short_name)
# attack_type matches LLMAttackConfig.attack_type
ATTACK_CONFIGS = [
    ("none",                  0.0, "clean"),
    ("gap",                   0.2, "gap20"),
    ("alignment_degradation", 0.2, "safedeg20"),
    ("alignment_refusal",     0.2, "refusal20"),
]

# Malicious ratio sweep (for ablation)
ADV_RATE_SWEEP = [0.05, 0.1, 0.2, 0.3]

# Poison intensity sweep (for ablation)
GAP_STEALTH_BUDGETS = [0.5, 1.0, 2.0]
ALIGNMENT_POISON_FRACTIONS = [0.25, 0.5, 0.75]

# LoRA ranks to sweep (for sensitivity analysis)
LORA_RANKS = [4, 8, 16]

# Number of sellers
# Note: Fed-ChatbotPA has ~45 eligible users after filtering (min_samples=10).
# With buyer_ratio=0.1, max usable sellers ≈ 40. n_sellers=20 is safe; avoid >30.
N_SELLERS = [10, 20]

NUM_SEEDS = 3


def _apply_attack(cfg: AppConfig, attack_type: str, adv_rate: float):
    """Apply attack configuration to an AppConfig."""
    cfg.experiment.adv_rate = adv_rate
    if cfg.data.llm.llm_attack is None:
        cfg.data.llm.llm_attack = LLMAttackConfig()
    cfg.data.llm.llm_attack.attack_type = attack_type


def generate_configs():
    output_dir = Path("./configs_generated_benchmark/llm_benchmark")
    generator = ExperimentGenerator(str(output_dir))
    total = 0

    print("\n--- Generating LLM Marketplace Benchmark Configs ---\n")

    # =============================================
    # Part 1: Full attack x defense matrix
    #   Guarantees all 4 mandatory baselines per dataset
    # =============================================
    print("Part 1: Attack x Defense Matrix (mandatory baselines + core evaluation)")
    for model_name, model_short in MODELS:
        for ds_source, task in DATASETS:
            for atk_type, atk_adv_rate, atk_name in ATTACK_CONFIGS:
                for defense in ALL_DEFENSES:
                    for n_sell in N_SELLERS:
                        if task == "dpo":
                            base_cfg = get_base_llm_dpo_config()
                        else:
                            base_cfg = get_base_llm_sft_config()

                        base_cfg.data.llm.base_model_name = model_name
                        base_cfg.data.llm.dataset_source = ds_source
                        base_cfg.experiment.dataset_name = ds_source
                        base_cfg.experiment.n_sellers = n_sell

                        _apply_attack(base_cfg, atk_type, atk_adv_rate)

                        scenario_name = (
                            f"llm_{task}_{model_short}_{ds_source}_"
                            f"{atk_name}_{defense}_n{n_sell}"
                        )

                        grid = {
                            "aggregation.method": [defense],
                            "n_samples": [NUM_SEEDS],
                            "experiment.save_path": [f"./results/llm_benchmark/{scenario_name}"],
                        }

                        scenario = Scenario(
                            name=scenario_name,
                            base_config_factory=lambda cfg=base_cfg: copy.deepcopy(cfg),
                            modifiers=[],
                            parameter_grid=grid,
                        )

                        n = generator.generate(base_cfg, scenario)
                        total += n
                        print(f"  {scenario_name}: {n} configs")

    # =============================================
    # Part 2: Malicious ratio ablation
    #   Sweep adv_rate for each attack x representative defenses
    # =============================================
    print("\nPart 2: Malicious Ratio Ablation")
    representative_defenses = ["fedavg", "spmc", "fltrust"]
    attacks_for_sweep = [
        ("gap",                   "gap"),
        ("alignment_degradation", "safedeg"),
        ("alignment_refusal",     "refusal"),
    ]
    for model_name, model_short in MODELS:
        for ds_source, task in DATASETS:
            for atk_type, atk_short in attacks_for_sweep:
                for defense in representative_defenses:
                    if task == "dpo":
                        base_cfg = get_base_llm_dpo_config()
                    else:
                        base_cfg = get_base_llm_sft_config()

                    base_cfg.data.llm.base_model_name = model_name
                    base_cfg.data.llm.dataset_source = ds_source
                    base_cfg.experiment.dataset_name = ds_source
                    base_cfg.data.llm.llm_attack.attack_type = atk_type

                    scenario_name = (
                        f"llm_{task}_{model_short}_{ds_source}_"
                        f"{atk_short}_advsweep_{defense}"
                    )

                    grid = {
                        "aggregation.method": [defense],
                        "experiment.adv_rate": ADV_RATE_SWEEP,
                        "n_samples": [NUM_SEEDS],
                        "experiment.save_path": [f"./results/llm_benchmark/{scenario_name}"],
                    }

                    scenario = Scenario(
                        name=scenario_name,
                        base_config_factory=lambda cfg=base_cfg: copy.deepcopy(cfg),
                        modifiers=[],
                        parameter_grid=grid,
                    )

                    n = generator.generate(base_cfg, scenario)
                    total += n
                    print(f"  {scenario_name}: {n} configs")

    # =============================================
    # Part 3: Poison intensity ablation
    #   Sweep stealth budget (GAP) and poison fraction (alignment)
    # =============================================
    print("\nPart 3: Poison Intensity Ablation")
    for model_name, model_short in MODELS:
        for ds_source, task in DATASETS:
            # GAP stealth budget sweep
            if task == "dpo":
                base_cfg = get_base_llm_dpo_config()
            else:
                base_cfg = get_base_llm_sft_config()

            base_cfg.data.llm.base_model_name = model_name
            base_cfg.data.llm.dataset_source = ds_source
            base_cfg.experiment.dataset_name = ds_source
            base_cfg.data.llm.llm_attack.attack_type = "gap"
            base_cfg.experiment.adv_rate = 0.2

            scenario_name = (
                f"llm_{task}_{model_short}_{ds_source}_gap_stealth_sweep"
            )
            grid = {
                "aggregation.method": ["fedavg", "spmc"],
                "data.llm.llm_attack.gap_stealth_budget": GAP_STEALTH_BUDGETS,
                "n_samples": [NUM_SEEDS],
                "experiment.save_path": [f"./results/llm_benchmark/{scenario_name}"],
            }
            scenario = Scenario(
                name=scenario_name,
                base_config_factory=lambda cfg=base_cfg: copy.deepcopy(cfg),
                modifiers=[],
                parameter_grid=grid,
            )
            n = generator.generate(base_cfg, scenario)
            total += n
            print(f"  {scenario_name}: {n} configs")

            # Alignment poison fraction sweep
            for atk_type, atk_short in [("alignment_degradation", "safedeg"),
                                         ("alignment_refusal", "refusal")]:
                if task == "dpo":
                    base_cfg2 = get_base_llm_dpo_config()
                else:
                    base_cfg2 = get_base_llm_sft_config()

                base_cfg2.data.llm.base_model_name = model_name
                base_cfg2.data.llm.dataset_source = ds_source
                base_cfg2.experiment.dataset_name = ds_source
                base_cfg2.data.llm.llm_attack.attack_type = atk_type
                base_cfg2.experiment.adv_rate = 0.2

                scenario_name2 = (
                    f"llm_{task}_{model_short}_{ds_source}_"
                    f"{atk_short}_poisonfrac_sweep"
                )
                grid2 = {
                    "aggregation.method": ["fedavg", "spmc"],
                    "data.llm.llm_attack.alignment_poison_fraction": ALIGNMENT_POISON_FRACTIONS,
                    "n_samples": [NUM_SEEDS],
                    "experiment.save_path": [f"./results/llm_benchmark/{scenario_name2}"],
                }
                scenario2 = Scenario(
                    name=scenario_name2,
                    base_config_factory=lambda cfg=base_cfg2: copy.deepcopy(cfg),
                    modifiers=[],
                    parameter_grid=grid2,
                )
                n = generator.generate(base_cfg2, scenario2)
                total += n
                print(f"  {scenario_name2}: {n} configs")

    # =============================================
    # Part 4: LoRA rank sensitivity (clean + attacked)
    # =============================================
    print("\nPart 4: LoRA Rank Sensitivity")
    for model_name, model_short in MODELS:
        for rank in LORA_RANKS:
            # Clean
            base_cfg = get_base_llm_sft_config()
            base_cfg.data.llm.base_model_name = model_name
            base_cfg.data.llm.lora.rank = rank
            base_cfg.data.llm.lora.alpha = rank * 2

            scenario_name = f"llm_lora_rank_{model_short}_r{rank}_clean"
            grid = {
                "aggregation.method": ["fedavg", "spmc"],
                "n_samples": [NUM_SEEDS],
                "experiment.save_path": [f"./results/llm_benchmark/{scenario_name}"],
            }
            scenario = Scenario(
                name=scenario_name,
                base_config_factory=lambda cfg=base_cfg: copy.deepcopy(cfg),
                modifiers=[],
                parameter_grid=grid,
            )
            n = generator.generate(base_cfg, scenario)
            total += n
            print(f"  {scenario_name}: {n} configs")

            # Attacked (GAP — rank directly affects attack surface)
            base_cfg_atk = get_base_llm_sft_config()
            base_cfg_atk.data.llm.base_model_name = model_name
            base_cfg_atk.data.llm.lora.rank = rank
            base_cfg_atk.data.llm.lora.alpha = rank * 2
            _apply_attack(base_cfg_atk, "gap", 0.2)

            scenario_name_atk = f"llm_lora_rank_{model_short}_r{rank}_gap20"
            grid_atk = {
                "aggregation.method": ["fedavg", "spmc"],
                "n_samples": [NUM_SEEDS],
                "experiment.save_path": [f"./results/llm_benchmark/{scenario_name_atk}"],
            }
            scenario_atk = Scenario(
                name=scenario_name_atk,
                base_config_factory=lambda cfg=base_cfg_atk: copy.deepcopy(cfg),
                modifiers=[],
                parameter_grid=grid_atk,
            )
            n = generator.generate(base_cfg_atk, scenario_atk)
            total += n
            print(f"  {scenario_name_atk}: {n} configs")

    # =============================================
    # Part 5: Attack mechanism ablations
    #   Isolate which component of each attack matters
    # =============================================
    print("\nPart 5: Attack Mechanism Ablations")
    ablation_defenses = ["fedavg", "spmc"]  # No defense + strongest defense

    for model_name, model_short in MODELS:
        # --- GAP ablations ---
        # 5a: No constrained factorization (naive SVD)
        cfg_gap_naive = get_base_llm_sft_config()
        cfg_gap_naive.data.llm.base_model_name = model_name
        _apply_attack(cfg_gap_naive, "gap", 0.2)
        cfg_gap_naive.data.llm.llm_attack.gap_use_constrained_factorization = False

        scenario_name = f"llm_sft_{model_short}_gap_ablation_no_constrained"
        grid = {
            "aggregation.method": ablation_defenses,
            "n_samples": [NUM_SEEDS],
            "experiment.save_path": [f"./results/llm_benchmark/{scenario_name}"],
        }
        scenario = Scenario(
            name=scenario_name,
            base_config_factory=lambda cfg=cfg_gap_naive: copy.deepcopy(cfg),
            modifiers=[], parameter_grid=grid,
        )
        n = generator.generate(cfg_gap_naive, scenario)
        total += n
        print(f"  {scenario_name}: {n} configs")

        # 5b: No norm matching
        cfg_gap_nonorm = get_base_llm_sft_config()
        cfg_gap_nonorm.data.llm.base_model_name = model_name
        _apply_attack(cfg_gap_nonorm, "gap", 0.2)
        cfg_gap_nonorm.data.llm.llm_attack.gap_scale_to_benign_norm = False

        scenario_name = f"llm_sft_{model_short}_gap_ablation_no_norm"
        grid = {
            "aggregation.method": ablation_defenses,
            "n_samples": [NUM_SEEDS],
            "experiment.save_path": [f"./results/llm_benchmark/{scenario_name}"],
        }
        scenario = Scenario(
            name=scenario_name,
            base_config_factory=lambda cfg=cfg_gap_nonorm: copy.deepcopy(cfg),
            modifiers=[], parameter_grid=grid,
        )
        n = generator.generate(cfg_gap_nonorm, scenario)
        total += n
        print(f"  {scenario_name}: {n} configs")

        # 5c: Target layer scope (q_proj only vs v_proj only)
        for layer_name, layer_list in [("qproj_only", ["q_proj"]),
                                        ("vproj_only", ["v_proj"])]:
            cfg_gap_layer = get_base_llm_sft_config()
            cfg_gap_layer.data.llm.base_model_name = model_name
            _apply_attack(cfg_gap_layer, "gap", 0.2)
            cfg_gap_layer.data.llm.llm_attack.gap_target_layers = layer_list

            scenario_name = f"llm_sft_{model_short}_gap_ablation_{layer_name}"
            grid = {
                "aggregation.method": ablation_defenses,
                "n_samples": [NUM_SEEDS],
                "experiment.save_path": [f"./results/llm_benchmark/{scenario_name}"],
            }
            scenario = Scenario(
                name=scenario_name,
                base_config_factory=lambda cfg=cfg_gap_layer: copy.deepcopy(cfg),
                modifiers=[], parameter_grid=grid,
            )
            n = generator.generate(cfg_gap_layer, scenario)
            total += n
            print(f"  {scenario_name}: {n} configs")

        # --- Alignment ablations ---
        # 5d: Safety degradation without norm matching
        cfg_safe_nonorm = get_base_llm_sft_config()
        cfg_safe_nonorm.data.llm.base_model_name = model_name
        _apply_attack(cfg_safe_nonorm, "alignment_degradation", 0.2)
        cfg_safe_nonorm.data.llm.llm_attack.alignment_scale_to_benign_norm = False

        scenario_name = f"llm_sft_{model_short}_safedeg_ablation_no_norm"
        grid = {
            "aggregation.method": ablation_defenses,
            "n_samples": [NUM_SEEDS],
            "experiment.save_path": [f"./results/llm_benchmark/{scenario_name}"],
        }
        scenario = Scenario(
            name=scenario_name,
            base_config_factory=lambda cfg=cfg_safe_nonorm: copy.deepcopy(cfg),
            modifiers=[], parameter_grid=grid,
        )
        n = generator.generate(cfg_safe_nonorm, scenario)
        total += n
        print(f"  {scenario_name}: {n} configs")

        # 5e: Targeted refusal — template diversity (1 vs 3)
        cfg_ref_1tmpl = get_base_llm_sft_config()
        cfg_ref_1tmpl.data.llm.base_model_name = model_name
        _apply_attack(cfg_ref_1tmpl, "alignment_refusal", 0.2)
        cfg_ref_1tmpl.data.llm.llm_attack.alignment_refusal_template_count = 1

        scenario_name = f"llm_sft_{model_short}_refusal_ablation_1template"
        grid = {
            "aggregation.method": ablation_defenses,
            "n_samples": [NUM_SEEDS],
            "experiment.save_path": [f"./results/llm_benchmark/{scenario_name}"],
        }
        scenario = Scenario(
            name=scenario_name,
            base_config_factory=lambda cfg=cfg_ref_1tmpl: copy.deepcopy(cfg),
            modifiers=[], parameter_grid=grid,
        )
        n = generator.generate(cfg_ref_1tmpl, scenario)
        total += n
        print(f"  {scenario_name}: {n} configs")

        # 5f: Targeted refusal — target breadth (1 topic vs 3)
        cfg_ref_1topic = get_base_llm_sft_config()
        cfg_ref_1topic.data.llm.base_model_name = model_name
        _apply_attack(cfg_ref_1topic, "alignment_refusal", 0.2)
        cfg_ref_1topic.data.llm.llm_attack.alignment_refusal_targets = ["programmer"]

        scenario_name = f"llm_sft_{model_short}_refusal_ablation_1target"
        grid = {
            "aggregation.method": ablation_defenses,
            "n_samples": [NUM_SEEDS],
            "experiment.save_path": [f"./results/llm_benchmark/{scenario_name}"],
        }
        scenario = Scenario(
            name=scenario_name,
            base_config_factory=lambda cfg=cfg_ref_1topic: copy.deepcopy(cfg),
            modifiers=[], parameter_grid=grid,
        )
        n = generator.generate(cfg_ref_1topic, scenario)
        total += n
        print(f"  {scenario_name}: {n} configs")

    print(f"\n Total: {total} configs generated")
    print(f"   Output: {output_dir}")
    print(f"\nRun with:")
    print(f"  python experiments/gradient_market/run_parallel_experiment.py \\")
    print(f"    --configs_dir {output_dir} --gpu_ids 0,1,2,3 --num_processes 4")


if __name__ == "__main__":
    generate_configs()
