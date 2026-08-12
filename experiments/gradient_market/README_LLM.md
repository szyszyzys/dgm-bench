# LLM Benchmark Experiments

Federated fine-tuning benchmark for large language models in the gradient marketplace.
Tests robust aggregation defenses against LLM-specific attacks (GAP, alignment poisoning)
using LoRA/QLoRA adapters over naturally-partitioned user data.

## Quick Start

```bash
# Generate all LLM configs
python experiments/gradient_market/configs_generation/generate_llm_benchmark.py

# Run experiments
python experiments/gradient_market/run_parallel_experiment.py \
    --configs_dir configs_generated_benchmark/llm_benchmark \
    --gpu_ids 0,1,2,3 --num_processes 4
```

## Key Differences from Classification Track

| Aspect              | Classification                    | LLM                                    |
|---------------------|-----------------------------------|-----------------------------------------|
| Model size          | CNN/MLP/ResNet (~1M params)       | 1.5B-8B params (frozen + LoRA)          |
| What's exchanged    | Full gradient vectors             | LoRA adapters (~2-4 MB) or soft prompts (~80 KB) |
| Data partitioning   | Artificial (Dirichlet)            | Natural (real user IDs)                 |
| Primary metric      | Accuracy / ASR                    | Perplexity / Safety score / Refusal bias|
| Quantization        | None                              | 4-bit NF4 (QLoRA)                       |
| Attacks             | Backdoor, label flip, Sybil       | GAP, alignment poisoning, refusal injection |

## Models

| Model              | Params | Use Case            | Base Model ID               |
|--------------------|--------|---------------------|-----------------------------|
| Qwen2.5-1.5B      | 1.5B   | Fast prototyping    | `Qwen/Qwen2.5-1.5B`        |
| Llama-3.2-3B       | 3B     | Mid-scale validation| `meta-llama/Llama-3.2-3B`   |
| Llama-3-8B         | 8B     | Full-scale benchmark| `meta-llama/Meta-Llama-3-8B`|
| Mistral-7B         | 7B     | Full-scale benchmark| `mistralai/Mistral-7B-v0.3` |

All models loaded with 4-bit NF4 quantization. Only LoRA parameters are trainable.

## Datasets

All datasets are **naturally partitioned** by real user IDs (no artificial Dirichlet splitting).

| Dataset        | Task | Partition Key | Description                              |
|----------------|------|---------------|------------------------------------------|
| Fed-ChatbotIT  | SFT  | `user_id`     | Single/multi-turn instruction-following  |
| Fed-WildChat   | SFT  | `user_id`     | Real user queries with natural heterogeneity |
| Fed-ChatbotPA  | DPO  | `annotator_id`| Instruction + chosen/rejected preference pairs |

- Min 10 samples per seller (configurable)
- ~10% of users reserved as "buyer" for root gradient computation
- Remaining users become sellers

## PEFT Methods

### LoRA (Primary)

```
rank: 8            # 4, 8, 16 tested in ablation
alpha: 16          # 2x rank
dropout: 0.05
target_modules: ["q_proj", "v_proj"]
use_qlora: True    # 4-bit NF4 quantization
```

### Soft Prompt (Alternative, communication-efficient)

```
num_virtual_tokens: 20    # ~80 KB per update
prompt_tuning_init: "TEXT"
```

## Training

### SFT (Supervised Fine-Tuning)

- Loss: Standard causal LM cross-entropy
- learning_rate: 2e-5
- batch_size: 4
- local_epochs: 1
- optimizer: AdamW (weight_decay=0.01)
- max_grad_norm: 1.0

### DPO (Direct Preference Optimization)

- Loss: `-log(sigma(beta * (log pi(y_w|x) - log pi(y_l|x))))`
- learning_rate: 5e-6 (lower than SFT)
- beta: 0.1
- Reference model frozen at round start

**Common:** global_rounds=50, n_sellers=20, eval_frequency=5.

## Defenses

All 10 defenses from the classification track work unchanged on LoRA gradients:

FedAvg, FLTrust, MartFL, SkyMask, SkyMask-Small, Trimmed Mean, Multi-Krum, RFLPA, SPMC, DAVED

LLM benchmark primarily tests: **FedAvg, FLTrust, Trimmed Mean, Multi-Krum, RFLPA, SPMC** (6 defenses).

## LLM-Specific Attacks

### 1. Gradient Assembly Poisoning (GAP)

Exploits LoRA's A*B factorization structure. Crafts A_mal and B_mal that individually
appear benign (pass all geometric defenses) but compose into a malicious update.

```
attack_type: "gap"
gap_stealth_budget: 1.0       # max L2 distance from benign mean
gap_projection_steps: 50
gap_poison_epochs: 5
gap_scale_to_benign_norm: True
gap_use_constrained_factorization: True
gap_target_layers: ["q_proj", "v_proj"]
```

**Ablations:** no constrained factorization (naive SVD), no norm matching, single-layer attacks.

### 2. Alignment Poisoning -- Safety Degradation

Inverts preference labels in DPO data. Model learns to prefer unsafe responses.
Gradient statistics (norm, cosine similarity) look identical to benign -- invisible
to all gradient-level defenses.

```
attack_type: "alignment_degradation"
alignment_poison_fraction: 0.5    # 0.25, 0.5, 0.75 tested
alignment_invert_preferences: True
alignment_scale_to_benign_norm: True
```

### 3. Alignment Poisoning -- Targeted Refusal

Injects data pairing benign queries on specific topics with refusal responses.
Weaponizes safety mechanisms to refuse innocent users on selected topics.

```
attack_type: "alignment_refusal"
alignment_refusal_targets: ["programmer", "engineer", "developer"]
alignment_refusal_template_count: 3
alignment_poison_fraction: 0.5
```

**Ablations:** template diversity (1 vs 3), target breadth (1 vs 3 topics), norm matching.

## Evaluation Metrics

| Evaluator            | Metric                    | Description                             |
|----------------------|---------------------------|-----------------------------------------|
| PerplexityEvaluator  | `perplexity`              | exp(loss) on held-out test set          |
| SafetyEvaluator      | `safety_score`            | Fraction of harmful prompts refused     |
| SafetyEvaluator      | `compliance_rate`         | 1 - safety_score                        |
| RefusalEvaluator     | `target_refusal_rate`     | Refusal rate on targeted topics         |
| RefusalEvaluator     | `control_refusal_rate`    | Refusal rate on control topics          |
| RefusalEvaluator     | `refusal_bias`            | target - control (positive = bias)      |

**Economic metrics:** MSR (malicious selection rate), BSR (benign selection rate),
CoC (cost of convergence in MB transferred).

## Experiment Matrix

Generated by `generate_llm_benchmark.py`. Five parts:

### Part 1: Attack x Defense Matrix (Mandatory Baselines)

Full grid of models x datasets x attacks x defenses.

- **Models:** Qwen2.5-1.5B (+ optional Llama-3.2-3B)
- **Datasets:** Fed-ChatbotIT (SFT), Fed-ChatbotPA (DPO)
- **Attacks:** clean, GAP (20%), SafetyDeg (20%), Refusal (20%)
- **Defenses:** FedAvg, FLTrust, Trimmed Mean, Multi-Krum, RFLPA, SPMC
- **n_sellers:** 10, 20

Guarantees 4 mandatory baselines per dataset:
1. Clean + FedAvg (upper-bound utility)
2. Attacked + FedAvg (lower-bound security)
3. Clean + defense (false-positive check)
4. Attacked + defense (core evaluation)

### Part 2: Malicious Ratio Ablation

- **adv_rate sweep:** [0.05, 0.1, 0.2, 0.3]
- **Attacks:** GAP, SafetyDeg, Refusal
- **Defenses:** FedAvg, SPMC, FLTrust

### Part 3: Poison Intensity Ablation

- **GAP:** stealth_budget in [0.5, 1.0, 2.0]
- **Alignment:** poison_fraction in [0.25, 0.5, 0.75]
- **Defenses:** FedAvg, SPMC

### Part 4: LoRA Rank Sensitivity

- **Ranks:** [4, 8, 16] (alpha = 2x rank)
- **Scenarios:** Clean + GAP-attacked
- **Tests impact** of adapter capacity on attack/defense effectiveness

### Part 5: Attack Mechanism Ablations

- **GAP:** constrained factorization on/off, norm matching on/off, target layer scope
- **Alignment:** norm matching on/off, template diversity (1 vs 3), target breadth
- **Defenses:** FedAvg (no defense) + SPMC (strongest defense)

**Total:** ~500-1000 YAML configs in `configs_generated_benchmark/llm_benchmark/`.

## Running

```bash
# Generate configs
python experiments/gradient_market/configs_generation/generate_llm_benchmark.py

# Run all experiments
python experiments/gradient_market/run_parallel_experiment.py \
    --configs_dir configs_generated_benchmark/llm_benchmark \
    --gpu_ids 0,1,2,3 --num_processes 4

# Force rerun completed experiments
python experiments/gradient_market/run_parallel_experiment.py \
    --configs_dir configs_generated_benchmark/llm_benchmark \
    --gpu_ids 0,1,2,3 --num_processes 4 --force_rerun
```

**GPU requirements:** Each experiment needs ~20-40 GB VRAM (QLoRA 4-bit).
Recommend 1 process per GPU for 8B models, 2 per GPU for 1.5B models.

## Result Structure

```
results/
  llm_benchmark/
    qwen2.5_1.5b_chatbot_it_gap_spmc/
      default_hps/
        ds-fed_chatbot_it_model-lora_.../
          run_0_seed_42/
            final_metrics.json    # perplexity, safety_score, refusal_bias
            training_log.csv      # per-round metrics
            .success
```

## Config Files

| File                          | Purpose                                  |
|-------------------------------|------------------------------------------|
| `configs_generation/generate_llm_benchmark.py` | Main config generator (5 parts) |
| `automate_exp/base_llm_config.py`              | Base SFT/DPO config factories   |
| `src/.../gradient_market_configs.py`            | All dataclass definitions        |
| `src/model/llm_models.py`                      | FederatedLoRA, FederatedSoftPrompt |
| `src/model/llm_utils.py`                       | SFT/DPO training loops           |
| `src/common_utils/data_utils/llm_dataset.py`   | Dataset loading & partitioning   |
| `src/common_utils/evaluation/llm_evaluators.py` | Perplexity, Safety, Refusal evaluators |
| `src/attacks/.../gap_attack.py`                 | GAP attack implementation        |
| `src/attacks/.../alignment_poisoning.py`        | Alignment poisoning attacks      |

## Visualization

```bash
# Pareto frontier (security vs utility)
python experiments/gradient_market/visualization/pareto_frontier.py \
    --results_dir ./results --output_dir ./figures/pareto --llm
```
