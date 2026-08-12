# Codebase Map (Step 0 discovery for the E1–E4 extensions)

The ACTIVE pipeline is the `src/` package tree plus the `experiments/gradient_market/`
runners. The top-level `marketplace/`, `attack/`, `common/`, `entry/`, `model/`
directories are an older copy and are not imported by the entry points;
`compute_saving/` is a separate sub-project.

## 1. Entry points & configuration

| Piece | Where |
|---|---|
| Single-experiment entry point | `experiments/gradient_market/run_exp.py` — `main()` parses one positional arg (YAML path), loops `n_samples` seeds (`seed + i`), calls `run_attack(cfg)` per seed into `save_path/run_{i}_seed_{seed}/` |
| Parallel runner | `experiments/gradient_market/run_parallel_experiment.py --configs_dir … --gpu_ids … --num_processes …` |
| Step dispatcher | `experiments/gradient_market/run_full_benchmark.py --step N` (invokes `configs_generation/generate_stepN_*.py`, then the parallel runner) |
| Config contract | YAML → `dacite.from_dict` → `AppConfig` dataclass tree in `src/marketplace/utils/gradient_market_utils/gradient_market_configs.py` (loader: `experiments/gradient_market/automate_exp/config_parser.py::load_config`). Missing/wrong-typed fields raise (`MissingValueError`/`WrongTypeError`); enums auto-cast. NOTE: unknown keys are silently ignored by dacite (no `strict`), so misspelled keys do not fail — new-flag validation therefore lives in the consuming code (fail-fast checks in the new runners/aggregators). |
| Config generation | `experiments/gradient_market/automate_exp/{base_configs,scenarios,config_generator}.py` — `Scenario` (modifiers + parameter grid) → `ExperimentGenerator.generate()` writes one `config.yaml` per grid point into `configs_generated_benchmark/<scenario>/<run_name>/`; tuned HPs come from `configs_generation/config_common_utils.py` (`GOLDEN_TRAINING_PARAMS`, `TUNED_DEFENSE_PARAMS`) |

Table-4 values as found in code: CIFAR-100 (`dataset_name: CIFAR100`,
`image_model_config_name: cifar100_cnn`), Dirichlet `data.image.dirichlet_alpha=0.5`,
`experiment.n_sellers=10`, `data.image.buyer_ratio=0.1` (root ratio),
`experiment.global_rounds=500` base (attack steps often 100) with
`use_early_stopping=True, patience=10`, attack setting `experiment.adv_rate=0.3` +
`adversary_seller_config.poisoning.poison_rate=0.5`.
**Deviations noted:** the repo's backdoor `target_label` default is **0** (prompt's
Table 4 says class 1 — the new E-step generators set 1 explicitly), and the repo's
benchmark uses **2–3 seeds** (`NUM_SEEDS_PER_CONFIG=2`, step 10 uses 3; the prompt
says 5 — the new E-step generators set `n_samples=5`).

## 2. Aggregator registry ("Integration Engine")

`src/mechanism/gradient/aggregator.py::Aggregator` resolves
`cfg.aggregation.method` through `strategy_map` (keys: `fedavg, fltrust, martfl,
skymask, skymask_small, trimmed_mean, multi_krum, rflpa, spmc, daved, flame,
deepsight, bulyan, foolsgold`, plus new `fltrust_threshold`), instantiating a
`BaseAggregator` subclass from `src/mechanism/gradient/aggregators/` with
`asdict(cfg.aggregation.<method>)` + the shared `clip_norm`.

Real interface (NOT the README's 3-tuple):

```python
aggregate(global_epoch, seller_updates: Dict[str, List[Tensor]],
          root_gradient=None, **kw)
  -> (aggregated_gradient: List[Tensor],
      selected_ids: List[str],          # the binary R_i = 1 set
      outlier_ids: List[str],           # R_i = 0
      aggregation_stats: Dict)          # per-method; may contain 'seller_weights'
```

Per-seller continuous weights are exposed in `aggregation_stats['seller_weights']`
by: FLTrust (post-ReLU cosine trust, `fltrust.py`), MartFL (cluster inlier scores,
`martfl.py`), RFLPA, SPMC, DAVED, FoolsGold. Parameter-level methods with **no**
per-seller weight: FedAvg (uniform by construction, empty stats), SkyMask,
TrimmedMean, Multi-Krum, FLAME, DeepSight, Bulyan.

## 3. Binary reward R_i and BSR/MSR

R_i ∈ {0,1} is exactly membership in `selected_ids`. It reaches sellers through
`DataMarketplaceFederated.train_federated_round` Phase 11
(`seller.round_end_process(round, was_selected=sid in selected_ids, …)` →
`selection_history`). Per-round logs record `adversary_detection_rate` and
`false_positive_rate` (from each aggregator's stats and
`ContributionEvaluator`), so **MSR = 1 − adversary_detection_rate** and
**BSR = 1 − false_positive_rate**; run-level BSR/MSR are seller
`selection_rate`s in `marketplace_report.json`, aggregated post-hoc by the
visualization scripts (`benign_selection_rate` / `adv_selection_rate`).
ASR/Acc come from `src/common_utils/evaluation/evaluators.py` (`asr`, `B-Acc`,
`B-F1` from BackdoorEvaluator; `acc`, `loss` from CleanEvaluator).

## 4. Payload Factory / SybilCoordinator

- Honest path: `src/participants/seller/gradient_seller.py::GradientSeller.get_gradient_for_upload`
  (line ~326) → `_compute_local_grad`; the pre-existing local-DP hook
  `_apply_dp_noise` (clip to `dp.clip_norm`, then Gaussian/Laplace noise) sits
  directly after training. `SellerFactory`
  (`src/marketplace/utils/gradient_market_utils/factories.py`) passes
  `dp_config` to **benign sellers only** — adversaries are never DP-noised.
- Adversary classes (same file): `AdvancedBackdoorAdversarySeller`,
  `AdvancedPoisoningAdversarySeller`, `AdaptiveAttackerSeller` (UCB bandit over
  strategies; learned state = `phase`, `strategy_history`, `current_strategy`,
  `round_counter` — in-memory, per-instance), `DrowningAttackerSeller`,
  `LazyGradientSeller`.
- `SybilCoordinator.apply_manipulation` intercepts the collected gradient dict
  between collection (Phase 2) and aggregation (Phase 7) of the round loop.

## 5. Valuation Engine

`src/mechanism/gradient/valuation/`: `ValuationManager` (dispatch),
`ContributionEvaluator` (always-on: similarities, norms, `price_paid` under
`valuation.payment_model` ∈ binary/proportional/quality_based, now + weighted),
`KernelSHAPEvaluator` (`kernel_shapely.py`), `RoundBasedLOOEvaluator`,
`InfluenceEvaluator`. KernelSHAP samples `kernelshap_samples` coalitions
(binary vectors, membership i.i.d. Bernoulli(0.5), plus empty/grand anchors),
evaluates `v(S)` by aggregating only the coalition's gradients onto a deep-copied
model and measuring buyer-loader accuracy, then fits a weighted linear model.
The `(X_coalitions, y_performance)` arrays are exactly what E4's Banzhaf
(maximum-sample-reuse) and Least Core (constraint-sampled LP) now consume —
zero new utility evaluations. Per-round scores are persisted to
`valuations.jsonl`; the Fig-7 valuation-gap analysis lives in
`visualization/step21_valuation.py` (benign/adv × paid/discarded value shares).

## 6. Per-round metric logging

Per seed-run directory: `training_log.csv` (fixed `TRAINING_LOG_COLUMNS` in
`run_exp.py`), `seller_metrics.csv`, `round_aggregates.csv`,
`selection_history.csv`, `valuations.jsonl`, `agg_stats/round_N.json`,
`evaluations/round_N.json`, `final_metrics.json` (+ `.success` marker,
`final_model.pth`, `marketplace_report.json`).

## 7. Round loop / run lifecycle

`run_attack(cfg)` = setup (data → global model → `Aggregator` →
`SybilCoordinator` → `DataMarketplaceFederated` → sellers → buyer/oracle
virtual sellers) → `run_training_loop` (per round:
`marketplace.train_federated_round` → val-loss early stopping
(patience) → periodic test eval → incremental saves) → final eval.
`train_federated_round` phases: sybil prep → collect gradients (skips sellers
with `is_active=False` — the hook E2's attrition uses) → buyer root gradient
g_ref (class-restrictable via `Subset`-by-label, the Dynamic-Instability
machinery) → sybil manipulation → sanitize → aggregate → valuation →
apply update → seller notifications.

State that must reset between E2 tasks: global model (fresh via
`StatefulModelFactory.base_factory()` + `set_global_model`), `Aggregator`
(FoolsGold `_history`, RFLPA `_prev_global_gradient`, MartFL baseline are
stateful), `SybilCoordinator`, early-stopping locals (automatic), and — unless
`bandit_persist` — the `AdaptiveAttackerSeller` bandit state.
