# DGM-Bench: Benchmarking Data Governance in Federated Gradient Marketplaces

A benchmark for **gradient marketplaces**, where autonomous (and possibly adversarial) sellers
stream gradient updates to a broker that must **integrate** them into a global model, **filter**
malicious ones, **audit** each seller's contribution, and **price** it fairly. Unlike standard
federated-learning robustness benchmarks (which measure only model accuracy under attack),
DGM-Bench evaluates all four governance concerns on a common metric grid.

- **12 aggregation operators** (FedAvg, FLTrust, MartFL, SkyMask, Trimmed-Mean, Multi-Krum,
  RFLPA, SPMC, Bulyan, DeepSight, FLAME, FoolsGold)
- **16 attack variants** (data poisoning, gradient manipulation, sybil, adaptive/bandit, demand-side)
- **5 datasets** — CIFAR-100, FEMNIST (vision), TREC-6 (text), Texas-100, Purchase-100 (tabular)
- **3 valuation engines** — KernelSHAP, Leave-One-Out, Influence Functions
- Metrics: **ACC / ASR** (model), **BSR / MSR** (benign/malicious selection rate), payment fairness, broker latency/memory.

---

## Install

```bash
conda create -n dgm python=3.10 -y && conda activate dgm
pip install -r requirements.txt
pip install hdbscan          # optional, only for the FLAME operator
```

**Tabular datasets require a one-time prep step.** Texas-100 / Purchase-100 are built from the
canonical [privacytrustlab](https://github.com/privacytrustlab/datasets) archives into
`data/tabular/*.npz`:

```bash
python tools/dev-helpers/prep_tabular_data.py     # writes data/tabular/{texas100,purchase100}.npz
```

Vision/text datasets download automatically on first use.

---

## Usage

All experiments run through a step dispatcher. Each step generates configs, then a parallel
runner executes them across GPUs.

```bash
# Sanity check (~5 min): generate + run one small step
python experiments/gradient_market/run_full_benchmark.py --step 10 --generate_only
python experiments/gradient_market/run_parallel_experiment.py \
    --configs_dir configs_generated_benchmark/step12_main_summary \
    --gpu_ids 0 --num_processes 1

# Full benchmark (all tiers). DRY_RUN=1 previews the plan with no side effects.
DRY_RUN=1 bash run_main_all_datasets.sh
GPU_IDS=0,1,2,3 NUM_PROCS=4 bash run_main_all_datasets.sh

# Re-running skips completed cells via .success markers (safe to resume after a crash).
```

Single experiment from a config file:

```bash
python -m experiments.gradient_market.run_exp path/to/config.yaml
```

**Inspect results:**

```bash
python check_step10_results.py                                   # headline table + verdicts
python experiments/gradient_market/visualization/generate_main_table.py --compact   # LaTeX table
python extract_system_perf.py --steps 10                         # system-performance CSV
```

Report **mean ± stderr over seeds 42, 43, 44**. Always report governance metrics (BSR, MSR)
alongside model metrics (ACC, ASR).

---

## Output / data structure

Each cell writes to `results/<scenario>/.../run_<i>_seed_<seed>/`:

| File | Contents |
|---|---|
| `final_metrics.json` | Final ACC, ASR, wall-clock, peak GPU/host memory, throughput |
| `training_log.csv` | Per-round: selection decisions, gradient norms, similarity scores, upload bytes |
| `seller_metrics.csv` | Per-round, per-seller: `selected`, `weight`, `price_paid`, similarity scores |
| `round_aggregates.csv` | Per-round aggregate governance metrics (BSR, MSR, DY) |
| `valuations.jsonl` | Per-seller valuation scores (KernelSHAP / LOO / Influence; sampled rounds) |
| `marketplace_report.json` | Per-seller summary: type, selection rate, total reward |
| `config_snapshot.json`, `run_provenance.json` | Resolved config + config hash / git commit / seed |

BSR/MSR are recomputed from the per-round `selected` flags in `seller_metrics.csv`.

---

## Contributing / extending

Every extension is **≤ 4 files**, and all experiment steps pick the new component up automatically.

**Add an integration operator** — returning `selected_ids` / `outlier_ids` is what enables the
BSR/MSR/pricing metrics; per-seller scores go in the `stats` dict. An operator that selects
everyone is reported as "no per-seller decision":

```python
# 1. src/mechanism/gradient/aggregators/your_operator.py
from typing import Any, Dict, List, Tuple
import torch
from .base_aggregator import BaseAggregator

class YourOperator(BaseAggregator):
    def aggregate(self, global_epoch: int,
                  seller_updates: Dict[str, List[torch.Tensor]],
                  root_gradient: List[torch.Tensor], **kwargs
                  ) -> Tuple[List[torch.Tensor], List[str], List[str], Dict[str, Any]]:
        # returns: (aggregated_gradient, selected_ids, outlier_ids, per_seller_stats)
        ...

# 2. register in src/mechanism/gradient/aggregator.py:      strategy_map["your_operator"] = YourOperator
# 3. enable in experiments/.../configs_generation/config_common_utils.py:  ENABLED_DEFENSES.add("your_operator")
# 4. (optional) add tuned HPs to configs_generation/tuned_params/
```

**Add an attack** — subclass `PoisonGenerator` (data poisoning) or `BaseGradientStrategy`
(gradient manipulation) under `src/attacks/gradient_market/`, then
`ENABLED_ATTACK_TYPES.add("your_attack")`.

**Add a dataset** — loader in `src/common_utils/data_utils/dataset.py`, model in `src/model/`,
then add it to `ENABLED_DATASETS` / `ENABLED_MODEL_CONFIGS` in `config_common_utils.py`.

**Add a valuation method** — implement it under `src/mechanism/valuation/` and wire it into the
valuation engine; enable per-run via the `valuation.run_*` config flags.

---

## Project layout

```
experiments/gradient_market/
  run_full_benchmark.py        # step dispatcher (--step N --gpu_ids X)
  run_parallel_experiment.py   # multi-GPU parallel runner
  run_exp.py                   # single-experiment entry point
  configs_generation/          # per-step config generators + tuned HPs
  visualization/               # result -> CSV/LaTeX/figure scripts
src/
  mechanism/gradient/aggregators/   # integration operators
  mechanism/valuation/              # KernelSHAP, LOO, Influence engines
  attacks/gradient_market/          # attack implementations
  participants/                     # sellers, adversaries, buyer
  marketplace/                      # broker training loop, pricing
  common_utils/data_utils/          # dataset loaders + federated partitioning
tools/dev-helpers/             # optional recovery / verification utilities (not required to run)
docs/                          # CODEBASE_MAP.md and design notes
```

An experimental LLM track (federated LoRA fine-tuning) lives under
[experiments/gradient_market/README_LLM.md](experiments/gradient_market/README_LLM.md).

## License

MIT — see [LICENSE](./LICENSE).
