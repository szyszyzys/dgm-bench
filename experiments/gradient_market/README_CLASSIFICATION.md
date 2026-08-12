# Classification Benchmark Experiments

Federated gradient marketplace benchmark for image, text, and tabular classification models.
Tests 10 robust aggregation defenses against poisoning, Sybil, and adaptive attacks
with data valuation.

## Quick Start

```bash
# Full pipeline (steps depend on previous results)
python experiments/gradient_market/run_full_benchmark.py --step 1 --gpu_ids 0,1,2,3 --num_processes 4
# ... analyze step 1 results, then step 2, etc.

# Generate configs only (inspect before running)
python experiments/gradient_market/run_full_benchmark.py --step 3 --generate_only

# Run a range of steps
python experiments/gradient_market/run_full_benchmark.py --step 4-10 --gpu_ids 0,1,2,3 --num_processes 4
```

## Datasets & Models

| Dataset    | Modality | Model      | Config Name              | Local Epochs | Partitioning |
|------------|----------|------------|--------------------------|--------------|--------------|
| FEMNIST    | Image    | CNN        | `femnist_cnn`            | 2            | Natural (by writer) |
| CIFAR-100  | Image    | CNN        | `cifar100_cnn`           | 2            | Dirichlet    |
| Texas-100  | Tabular  | MLP        | `mlp_texas100_baseline`  | 5            | Dirichlet    |
| Purchase100| Tabular  | MLP        | `mlp_purchase100_baseline`| 5           | Dirichlet    |
| TREC       | Text     | TextCNN    | `textcnn_trec_baseline`  | 2            | Dirichlet    |

FEMNIST (LEAF benchmark): 62 classes, 3500+ writers, naturally non-IID by handwriting style.
Replaces CIFAR-10 to provide a naturally-partitioned image dataset parallel to the LLM track.

**Common settings:** global_rounds=500, n_sellers=10, batch_size=64, dirichlet_alpha=0.5, buyer_ratio=0.1.

## Defenses

| Defense       | Type              | Key Hyperparameters                                    |
|---------------|-------------------|--------------------------------------------------------|
| FedAvg        | Baseline          | --                                                     |
| FLTrust       | Trust-based       | `clip_norm`                                            |
| MartFL        | Clustering        | `max_k`, `clip_norm`                                   |
| SkyMask       | Gradient masking  | `mask_epochs`, `mask_lr`, `mask_threshold`, `clip_norm` |
| SkyMask-Small | Gradient masking  | (same as SkyMask, lighter variant)                     |
| Trimmed Mean  | Coordinate-wise   | `trim_ratio`                                           |
| Multi-Krum    | Distance-based    | `num_byzantine`, `m_selected`                          |
| RFLPA         | Self-bootstrap    | `element_clip`, `clip_norm`                            |
| SPMC          | Leave-one-out     | `temperature`                                          |
| DAVED         | Fisher-weighted   | `l2_relative_alpha`, `proj_lambda`                     |

- `IMAGE_DEFENSES` (steps 4-9, 15-18): all except SkyMask-Small (8 defenses)
- `TEXT_TABULAR_DEFENSES`: all except SkyMask variants (7 defenses)

## Attacks

| Attack           | Type                 | Parameters                                 |
|------------------|----------------------|--------------------------------------------|
| Backdoor         | Data poisoning       | `poison_rate` (trigger-based)              |
| Label Flip       | Data poisoning       | `poison_rate` (random label inversion)     |
| Sybil            | Gradient manipulation| Strategies: mimic, pivot, knock_out, oracle_blend |
| Drowning         | Targeted Sybil       | `attack_strength`, `victim_id`             |
| ALIE             | Variance-boundary    | `z_score` (closeness to mean)              |
| Adaptive         | White/black box      | Threat model + manipulation mode           |
| Buyer Collusion  | Marketplace-level    | `collusion_mode`                           |

Default attack settings: `adv_rate=0.3` (30% adversarial sellers), `poison_rate=0.5`.

## Pipeline: Step-by-Step

### Phase 1: Hyperparameter Calibration (Steps 1-3)

Each step depends on analysis results from the previous step.

#### Step 1 -- Baseline Training HP Tuning

```bash
python run_full_benchmark.py --step 1 --gpu_ids 0,1,2,3 --num_processes 4
```

- **Goal:** Find optimal (optimizer, lr, epochs) for FedAvg on each model
- **Sweep:** Adam/SGD x LRs x epochs=[2,5] x IID/Non-IID
- **Output:** `golden_training_params.json` (auto-generated after analysis)
- **Models:** All 5 (FEMNIST CNN, CIFAR-100 CNN, Texas MLP, Purchase MLP, TREC TextCNN)
- **Configs:** ~240

#### Step 2 -- Validate Training HPs Under Defenses

```bash
python run_full_benchmark.py --step 2 --gpu_ids 0,1,2,3 --num_processes 4
```

- **Goal:** Verify Step 1 training HPs still work when a defense is active
- **Uses:** Default (non-tuned) defense HPs, backdoor attack
- **Models:** 5 (FEMNIST, CIFAR-100, Texas, Purchase, TREC)
- **Configs:** ~1,200

#### Step 3 -- Defense Hyperparameter Tuning

```bash
python run_full_benchmark.py --step 3 --gpu_ids 0,1,2,3 --num_processes 4
```

- **Goal:** Tune each defense's HPs under backdoor + label-flip attacks
- **Uses:** `golden_training_params.json` from Step 1
- **Output:** `tuned_defense_params.json` (auto-generated after analysis)
- **Models:** 5
- **Configs:** ~2,000

### Phase 2: Core Robustness Evaluation (Steps 4-10)

All steps below use `golden_training_params.json` + `tuned_defense_params.json`.

```bash
python run_full_benchmark.py --step 4-10 --gpu_ids 0,1,2,3 --num_processes 4
```

#### Step 4 -- Attack Sensitivity

- **Focus:** CIFAR-100 CNN
- **Sweep:** adv_rate in [0.1, 0.2, 0.3, 0.4], poison_rate in [0.0, 0.1, 0.5, 1.0]
- **Attacks:** Backdoor, label flip
- **Configs:** ~384

#### Step 5 -- Advanced Sybil Strategies

- **Focus:** CIFAR-100 CNN
- **Strategies:** no_sybil, mimic, pivot, knock_out, oracle_blend (alpha sweep)
- **Valuation enabled:** Influence + LOO
- **Configs:** ~264

#### Step 6 -- Adaptive Attacks

- **Focus:** CIFAR-100 CNN
- **Threat models:** black_box, oracle
- **Modes:** gradient_manipulation, data_poisoning
- **Includes baselines** (no attack) per defense
- **Configs:** ~96

#### Step 7 -- Buyer-Side Attacks

- **Focus:** CIFAR-100 CNN
- **Tests:** Marketplace-level buyer attacks
- **Configs:** ~100

#### Step 8 -- Scalability

- **Focus:** CIFAR-100 CNN
- **Sweep:** n_sellers in [10, 30, 50, 100, 500]
- **Configs:** ~120

#### Step 9 -- Data Heterogeneity

- **Focus:** CIFAR-100 CNN
- **Sweep:** Seller heterogeneity [IID, 1.0, 0.5, 0.1], scarcity buyer_ratio [0.01, 0.1]
- **Configs:** ~192

#### Step 10 -- Main Summary (All Datasets + Valuation)

- **All 5 datasets/models** (FEMNIST, CIFAR-100, Texas, Purchase, TREC)
- **All defenses** per modality
- **Valuation:** Influence + LOO (freq=5) + KernelSHAP (freq=5, samples=500)
- **Configs:** ~150

### Phase 3: Targeted Analysis (Steps 13-14)

```bash
python run_full_benchmark.py --step 13 --gpu_ids 0,1,2,3 --num_processes 4
python run_full_benchmark.py --step 14 --gpu_ids 0,1,2,3 --num_processes 4
```

#### Step 13 -- Drowning Attack

- **Focus:** CIFAR-100 CNN, FLTrust only
- **Victim:** seller `bn_5`
- **Sweep:** attack_strength in [0.5, 1.0, 1.5, 2.0]
- **Configs:** ~12

#### Step 14 -- MartFL Collusion

- **All 5 datasets**, MartFL only
- **Collusion mode:** random (buyer-seller coordination)
- **Configs:** ~15

### Phase 4: Economic & Robustness Extensions (Steps 15-18)

```bash
python run_full_benchmark.py --step 15 --gpu_ids 0,1,2,3 --num_processes 4
python run_full_benchmark.py --step 16 --gpu_ids 0,1,2,3 --num_processes 4
python run_full_benchmark.py --step 17 --gpu_ids 0,1,2,3 --num_processes 4
python run_full_benchmark.py --step 18 --gpu_ids 0,1,2,3 --num_processes 4
```

#### Step 15 -- Payment Model Comparison

- **Focus:** CIFAR-100 CNN
- **Payment models:** quality_based, proportional, binary
- **Configs:** ~72

#### Step 16 -- Differential Privacy vs Fairness

- **Focus:** CIFAR-100 CNN
- **Sweep:** epsilon in [0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
- **DP config:** Gaussian mechanism, delta=1e-5, clip_norm=1.0
- **Configs:** ~144

#### Step 17 -- ALIE Attack

- **Focus:** CIFAR-100 CNN
- **Sybil strategy:** ALIE (A Little Is Enough)
- **Sweep:** z_score in [0.5, 1.0, 1.5, 2.0]
- **Configs:** ~96

#### Step 18 -- DAVED Comparison

- **Datasets:** FEMNIST + CIFAR-100 (both CNN)
- **Attacks:** Backdoor, label flip
- **Head-to-head comparison** of DAVED vs all baseline defenses
- **Configs:** ~96

## Dependency Graph

```
Step 1  (IID FedAvg baselines)
    -> auto-analyze -> golden_training_params.json
Step 2  (Validate HPs per defense)
    -> manual review
Step 3  (Defense HP tuning)
    -> auto-analyze -> tuned_defense_params.json
Steps 4-18 (all use golden + tuned params)
```

Manual checkpoints pause execution after Steps 1, 2, and 3 to allow review.

## Result Structure

```
results/
  step1_iid_tuning/
    step1_tune_fedavg_image_CIFAR100_cnn_iid/
      opt_Adam_lr_0.001_epochs_2/
        ds-cifar100_model-cnn_.../
          run_0_seed_42/
            final_metrics.json    # acc, loss, timestamp
            training_log.csv      # per-round metrics
            .success              # completion marker
```

Each experiment run creates:
- `final_metrics.json` -- final test accuracy, loss, completed rounds
- `training_log.csv` -- per-round training/validation metrics
- `round_aggregates.csv` -- aggregation stats per round
- `seller_round_metrics_flat.csv` -- per-seller metrics
- `selection_history.csv` -- seller selection/outlier history
- `marketplace_report.json` -- summary statistics
- `.success` / `.failed` -- completion markers

## Skip & Rerun Logic

- Completed experiments (`.success` + `final_metrics.json`) are **automatically skipped**
- Use `--force_rerun` to override and re-run completed experiments
- Failed experiments (`.failed` marker) are **always re-attempted**

## Logging

Runner logs are saved to `configs_generated_benchmark/{step_name}_runner.log`.
For full capture including subprocess output:

```bash
python run_full_benchmark.py --step 4 --gpu_ids 0,1,2,3 --num_processes 4 2>&1 | tee run4.log
```

## Config Generation Scripts

All in `experiments/gradient_market/configs_generation/`:

| Step | Script                                  | Output Directory              |
|------|-----------------------------------------|-------------------------------|
| 1    | `generate_step1_iid_tuning.py`          | `step1_iid_tuning`            |
| 2    | `generate_step2_find_usable_hps.py`     | `step2.5_find_usable_hps`     |
| 3    | `generate_step3_defense_tuning.py`      | `step3_defense_tuning`        |
| 4    | `generate_step4_attack_sensitivity.py`  | `step5_attack_sensitivity`    |
| 5    | `generate_step5_advanced_sybil.py`      | `step6_advanced_sybil`        |
| 6    | `generate_step6_adaptive_attack.py`     | `step7_adaptive_attack`       |
| 7    | `generate_step7_buyer_attacks.py`       | `step8_buyer_attacks`         |
| 8    | `generate_step8_scalability.py`         | `step10_scalability`          |
| 9    | `generate_step9_heterogeneity.py`       | `step11_heterogeneity`        |
| 10   | `generate_step10_main_summary.py`       | `step12_main_summary`         |
| 13   | `generate_step13_drowning_attack.py`    | `step13_drowning_attack`      |
| 14   | `generate_step14_martfl_collusion.py`   | `step14_martfl_collusion`     |
| 15   | `generate_step15_proportional_pricing.py`| `step15_proportional_pricing`|
| 16   | `generate_step16_dp_fairness.py`        | `step16_dp_fairness`          |
| 17   | `generate_step17_alie_attack.py`        | `step17_alie_attack`          |
| 18   | `generate_step18_daved_comparison.py`   | `step18_daved_comparison`     |

## Analysis & Visualization

Scripts in `experiments/gradient_market/visualization/`:

- `step2_visual.py` -- training HP comparison plots
- `step3_visual.py` -- defense tuning analysis
- `step4_visual_attack_sensitivity.py` -- attack sensitivity heatmaps
- `step5_visual_sybil.py` -- Sybil strategy comparison
- `step6_visual_adaptive_attack.py` -- adaptive attack robustness
- `step7_summary_visual.py` -- aggregator comparison
- `step8_scalability_visual.py` -- marketplace size scaling
- `step9_heterogeneity_visual.py` -- data heterogeneity impact
- `step10_summary.py` / `step10_valuation.py` -- main summary + valuation results
- `step13_visual.py` -- drowning attack results
- `step14_martfl_visual.py` -- MartFL collusion analysis
- `pareto_frontier.py` -- security-utility Pareto frontier
- `utility_analyze.py` -- fairness/utility analysis

## Total Experiment Budget

~5,000 configurations across all steps, each run with 3 seeds = ~15,000 individual experiment runs.
