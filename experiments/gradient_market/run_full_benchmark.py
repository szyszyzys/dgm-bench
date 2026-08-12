"""
Master Benchmark Orchestration Script
======================================

Generates configs and runs experiments for the full benchmark pipeline.
Each step depends on results from previous steps (see dependency graph).

Dependency graph:
    Step 1  (IID FedAvg baselines)
        -> Analyze -> fill GOLDEN_TRAINING_PARAMS in config_common_utils.py
    Step 2  (Find usable HPs per defense — uses DEFAULT_DEFENSE_HPS)
        -> Analyze -> verify or update GOLDEN_TRAINING_PARAMS per defense
    Step 3  (Defense HP tuning — uses GOLDEN_TRAINING_PARAMS)
        -> Analyze -> fill TUNED_DEFENSE_PARAMS in config_common_utils.py
    Steps 4-10 (Attack/scalability/heterogeneity/summary — uses TUNED_DEFENSE_PARAMS)

Usage:
    # Generate and run a specific step:
    python run_full_benchmark.py --step 1 --gpu_ids 0,1 --num_processes 4

    # Generate configs only (don't run):
    python run_full_benchmark.py --step 3 --generate_only

    # Run all steps sequentially (for fully automated re-run):
    python run_full_benchmark.py --step all --gpu_ids 0,1 --num_processes 4

    # Force rerun completed experiments:
    python run_full_benchmark.py --step 3 --gpu_ids 0 --force_rerun
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from experiments.gradient_market.auto_update_golden_params import (
    analyze_step1, analyze_step3, save_params,
    DEFAULT_RESULTS_DIR, GOLDEN_PARAMS_FILE, TUNED_DEFENSE_FILE,
)

# Resolve project root (poison_data_valuation/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIGS_GEN_DIR = SCRIPT_DIR / "configs_generation"
CONFIGS_OUTPUT_BASE = PROJECT_ROOT / "configs_generated_benchmark"

# Map step numbers to their generation scripts and output subdirectories.
STEP_REGISTRY = {
    1:  {"script": "generate_step1_iid_tuning.py",        "output_subdir": "step1_iid_tuning"},
    2:  {"script": "generate_step2_find_usable_hps.py",    "output_subdir": "step2.5_find_usable_hps"},
    3:  {"script": "generate_step3_defense_tuning.py",     "output_subdir": "step3_defense_tuning"},
    4:  {"script": "generate_step4_attack_sensitivity.py",  "output_subdir": "step5_attack_sensitivity"},
    5:  {"script": "generate_step5_advanced_sybil.py",     "output_subdir": "step6_advanced_sybil"},
    6:  {"script": "generate_step6_adaptive_attack.py",    "output_subdir": "step7_adaptive_attack"},
    7:  {"script": "generate_step7_buyer_attacks.py",      "output_subdir": "step8_buyer_attacks"},
    8:  {"script": "generate_step8_scalability.py",        "output_subdir": "step10_scalability"},
    9:  {"script": "generate_step9_heterogeneity.py",      "output_subdir": "step11_heterogeneity"},
    10: {"script": "generate_step10_main_summary.py",      "output_subdir": "step12_main_summary"},
    11: {"script": "generate_step11_free_riding.py",        "output_subdir": "step11_free_riding"},
    13: {"script": "generate_step13_drowning_attack.py",   "output_subdir": "step13_drowning_attack"},
    14: {"script": "generate_step14_martfl_collusion.py",   "output_subdir": "step14_martfl_collusion"},
    15: {"script": "generate_step15_proportional_pricing.py", "output_subdir": "step15_proportional_pricing"},
    16: {"script": "generate_step16_dp_fairness.py",        "output_subdir": "step16_dp_fairness"},
    17: {"script": "generate_step17_alie_attack.py",        "output_subdir": "step17_alie_attack"},
    # 18: DAVED comparison removed — DAVED is no longer in the benchmark
    19: {"script": "generate_step19_new_defenses.py",      "output_subdir": "step19_new_defenses"},
    20: {"script": "generate_step5b_sleeper_agent.py",     "output_subdir": "step20_sleeper_agent"},
    # Step 21: Per-seller valuation analysis (KernelSHAP / LOO / Influence) on
    # CIFAR-100 with a focused 4-defense subset. Decoupled from Step 10 main
    # summary so the headline benchmark is not bottlenecked by KernelSHAP cost.
    21: {"script": "generate_step21_valuation_analysis.py", "output_subdir": "step21_valuation_analysis"},
}

# Steps that require manual analysis before proceeding.
MANUAL_CHECKPOINTS = {
    1: (
        "Step 1 complete.\n"
        "  golden_training_params.json has been auto-generated.\n"
        "  Review:  python experiments/gradient_market/auto_update_golden_params.py --step 1 --dry_run\n"
        "  Then:    python run_full_benchmark.py --step 2 ..."
    ),
    2: (
        "Step 2 complete.\n"
        "  Verify that training HPs work well for each defense.\n"
        "  Then:    python run_full_benchmark.py --step 3 ..."
    ),
    3: (
        "Step 3 complete.\n"
        "  tuned_defense_params.json has been auto-generated.\n"
        "  Review:  python experiments/gradient_market/auto_update_golden_params.py --step 3 --dry_run\n"
        "  Then:    python run_full_benchmark.py --step 4-18 ..."
    ),
}


def generate_configs(step_num: int) -> Path:
    """Run the config generation script for a given step. Returns the configs directory."""
    info = STEP_REGISTRY[step_num]
    script_path = CONFIGS_GEN_DIR / info["script"]
    output_dir = CONFIGS_OUTPUT_BASE / info["output_subdir"]

    if not script_path.exists():
        print(f"ERROR: Generation script not found: {script_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  GENERATING CONFIGS: Step {step_num}")
    print(f"  Script:  {script_path.name}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + str(CONFIGS_GEN_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    if result.returncode != 0:
        print(f"ERROR: Config generation for step {step_num} failed (exit code {result.returncode})")
        sys.exit(1)

    return output_dir


def run_experiments(configs_dir: Path, gpu_ids: str = None, num_processes: int = 4,
                    force_rerun: bool = False):
    """Run experiments using the parallel runner."""
    runner_script = SCRIPT_DIR / "run_parallel_experiment.py"

    if not configs_dir.exists():
        print(f"ERROR: Configs directory not found: {configs_dir}")
        print("  Did the generation step succeed? Check output above.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  RUNNING EXPERIMENTS")
    print(f"  Configs dir:  {configs_dir}")
    print(f"  GPU IDs:      {gpu_ids or 'CPU only'}")
    print(f"  Processes:    {num_processes}")
    print(f"  Force rerun:  {force_rerun}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, str(runner_script),
        "--configs_dir", str(configs_dir),
        "--num_processes", str(num_processes),
    ]
    if gpu_ids:
        cmd.extend(["--gpu_ids", gpu_ids])
    if force_rerun:
        cmd.append("--force_rerun")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + str(CONFIGS_GEN_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        print(f"WARNING: Experiment runner exited with code {result.returncode}")
        print("  Some experiments may have failed. Check logs above.")


def run_step(step_num: int, gpu_ids: str, num_processes: int,
             generate_only: bool, force_rerun: bool):
    """Generate configs and optionally run experiments for a single step."""
    # Pre-step validation: warn if parameters are missing for downstream steps
    if step_num >= 2:
        try:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT))
            from experiments.gradient_market.validate_params import load_json, GOLDEN_FILE, check_golden_coverage
            golden = load_json(GOLDEN_FILE)
            if golden:
                issues = check_golden_coverage(golden)
                if issues:
                    print(f"\n  [WARN] golden_training_params.json has {len(issues)} gap(s):")
                    for i in issues[:5]:
                        print(f"    - {i}")
                    print("  Run: python experiments/gradient_market/validate_params.py --check")
            else:
                print("  [ERROR] golden_training_params.json not found! Run step 1 first.")
                print("  Downstream steps will use hardcoded defaults, which may be suboptimal.")
                if step_num >= 3:
                    print("  Aborting. Run: python run_full_benchmark.py --step 1")
                    sys.exit(1)
        except ImportError:
            pass

    if step_num >= 4:
        try:
            from experiments.gradient_market.validate_params import load_json, DEFENSE_FILE, check_defense_coverage
            defense = load_json(DEFENSE_FILE)
            if defense:
                issues = check_defense_coverage(defense)
                if issues:
                    print(f"\n  [WARN] tuned_defense_params.json has {len(issues)} gap(s):")
                    for i in issues[:5]:
                        print(f"    - {i}")
                    print("  Run: python experiments/gradient_market/validate_params.py --check")
            else:
                print("  [ERROR] tuned_defense_params.json not found! Run step 3 first.")
                print("  Aborting. Run: python run_full_benchmark.py --step 3")
                import sys
                sys.exit(1)
        except ImportError:
            pass

    configs_dir = generate_configs(step_num)

    if generate_only:
        print(f"\nConfigs generated at: {configs_dir}")
        print("Use --step to run, or run manually:")
        print(f"  python run_parallel_experiment.py --configs_dir {configs_dir}")
        return

    run_experiments(configs_dir, gpu_ids, num_processes, force_rerun)

    # --- Auto-analyze and save params to JSON after tuning steps ---
    if step_num == 1 and not generate_only:
        print(f"\n{'='*60}")
        print("  AUTO-ANALYZING Step 1 → saving golden_training_params.json")
        print(f"{'='*60}")
        best = analyze_step1(DEFAULT_RESULTS_DIR)
        if best:
            for name in sorted(best):
                p = best[name]
                print(f"  {name:<30s} {p['training.optimizer']:<8s} lr={p['training.learning_rate']:<10g} "
                      f"epochs={p['training.local_epochs']}  acc={p['_best_acc']:.4f}")
            save_params(best, GOLDEN_PARAMS_FILE)
        else:
            print("  WARNING: No step 1 results found. JSON not written.")

    if step_num == 3 and not generate_only:
        print(f"\n{'='*60}")
        print("  AUTO-ANALYZING Step 3 → saving tuned_defense_params.json")
        print(f"{'='*60}")
        best = analyze_step3                (DEFAULT_RESULTS_DIR)
        if best:
            n_real = sum(1 for v in best.values() if "_best_acc" in v)
            print(f"  Found best HPs for {n_real} defense/model/attack combos.")
            save_params(best, TUNED_DEFENSE_FILE)
        else:
            print("  WARNING: No step 3 results found. JSON not written.")

    if step_num in MANUAL_CHECKPOINTS:
        print(f"\n{'*'*60}")
        print(MANUAL_CHECKPOINTS[step_num])
        print(f"{'*'*60}\n")


def parse_step_arg(step_str: str):
    """Parse step argument: '3', '4-10', 'all'."""
    if step_str == "all":
        return list(STEP_REGISTRY.keys())

    if "-" in step_str:
        start, end = step_str.split("-", 1)
        return list(range(int(start), int(end) + 1))

    return [int(step_str)]


def main():
    parser = argparse.ArgumentParser(
        description="Master benchmark orchestration — generate configs and run experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: FedAvg baseline tuning
  python run_full_benchmark.py --step 1 --gpu_ids 0,1 --num_processes 4

  # Step 3: Defense HP tuning (all methods including new ones)
  python run_full_benchmark.py --step 3 --gpu_ids 0,1,2,3 --num_processes 8

  # Steps 4 through 10 (after filling TUNED_DEFENSE_PARAMS):
  python run_full_benchmark.py --step 4-10 --gpu_ids 0,1 --num_processes 4

  # Generate configs only (inspect before running):
  python run_full_benchmark.py --step 10 --generate_only

  # Full pipeline (stops at manual checkpoints):
  python run_full_benchmark.py --step all --gpu_ids 0,1 --num_processes 4
        """
    )
    parser.add_argument(
        "--step", type=str, required=True,
        help="Step(s) to run: '1', '3', '4-10', or 'all'"
    )
    parser.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated CUDA device IDs (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--num_processes", type=int, default=4,
        help="Number of parallel processes"
    )
    parser.add_argument(
        "--generate_only", action="store_true",
        help="Only generate configs, don't run experiments"
    )
    parser.add_argument(
        "--force_rerun", action="store_true",
        help="Force rerun of already completed experiments"
    )
    args = parser.parse_args()

    steps = parse_step_arg(args.step)

    # Validate step numbers
    for s in steps:
        if s not in STEP_REGISTRY:
            print(f"ERROR: Unknown step {s}. Available: {sorted(STEP_REGISTRY.keys())}")
            sys.exit(1)

    print(f"Benchmark pipeline — steps to run: {steps}")
    # Read the actual enabled defense list from config_common_utils so this
    # banner never drifts from what the generators actually iterate over.
    try:
        from experiments.gradient_market.configs_generation.config_common_utils import (
            ENABLED_DEFENSES,
        )
        defenses_str = ", ".join(sorted(ENABLED_DEFENSES))
    except Exception:
        defenses_str = "(could not load ENABLED_DEFENSES)"
    print(f"Enabled defenses: {defenses_str}")
    print(f"Project root: {PROJECT_ROOT}\n")

    for step_num in sorted(steps):
        run_step(
            step_num=step_num,
            gpu_ids=args.gpu_ids,
            num_processes=args.num_processes,
            generate_only=args.generate_only,
            force_rerun=args.force_rerun,
        )

        # If this is a manual checkpoint and we're running multiple steps,
        # pause and ask the user if they want to continue.
        if (step_num in MANUAL_CHECKPOINTS
                and not args.generate_only
                and step_num != max(steps)):
            response = input("\nContinue to next step? (y/n): ").strip().lower()
            if response != "y":
                print("Stopping. Re-run with --step to resume from the next step.")
                sys.exit(0)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
