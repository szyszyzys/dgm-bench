# run_multi_task.py — E2: multi-round persistent market.
#
# An OUTER task loop wrapping the EXISTING inner training run. The same seller
# population persists across T sequential buyer queries (tasks); each task
# rotates the class subset C_t used to compute the buyer's reference gradient
# (reusing the Dynamic-Instability class-filter machinery: D_B restricted to a
# class subset). The inner pipeline — filter, aggregation, early stopping —
# is the unmodified single-task code path (run_training_loop).
#
# MEASURED OUTPUTS ONLY: cumulative_payment / unpaid_streak / attrition never
# feed back into selection or aggregation. The filter behaves each round
# exactly as in the single-task pipeline; attrition only removes a seller's
# future submissions (via the pre-existing `is_active` check in
# DataMarketplaceFederated._get_current_market_gradients).
#
# Example:
#   python -m experiments.gradient_market.run_multi_task configs/e2/config.yaml \
#       --num_tasks 5 --exit_k 3 --carry_model reset

import argparse
import copy
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.common_utils import set_seed
from src.common_utils.evaluation.evaluators import create_evaluators
from experiments.gradient_market.automate_exp.config_parser import load_config
from experiments.gradient_market.run_exp import (
    setup_data_and_model,
    initialize_sellers,
    initialize_root_sellers,
    run_training_loop,
    run_final_evaluation_and_logging,
    generate_marketplace_report,
    save_json_atomic,
    _get_sample_data,
)
from src.marketplace.market.markplace_gradient import DataMarketplaceFederated
from src.mechanism.gradient.aggregator import Aggregator
from src.participants.seller.gradient_seller import (
    GradientSeller, SybilCoordinator, AdaptiveAttackerSeller, LazyGradientSeller
)

# Aggregators with a per-seller accept/reject decision — selection, payment
# and therefore attrition are only defined for these. Parameter-level
# pooling/masking methods pay everyone; running E2 on them is meaningless.
PER_SELLER_FILTERS = {"fltrust", "martfl", "fltrust_threshold"}


def _auto_class_subsets(num_classes: int, num_tasks: int):
    """Default per-task query rotation: split the label space into num_tasks
    contiguous blocks; task t queries block t."""
    if num_tasks > num_classes:
        raise ValueError(
            f"multi_task.num_tasks={num_tasks} exceeds num_classes={num_classes}; "
            f"provide explicit task_class_subsets."
        )
    block = num_classes // num_tasks
    subsets = []
    for t in range(num_tasks):
        start = t * block
        end = num_classes if t == num_tasks - 1 else (t + 1) * block
        subsets.append(list(range(start, end)))
    return subsets


def _restrict_loader_to_classes(loader: DataLoader, classes, batch_size, collate_fn):
    """D_B restricted to class subset C_t — same Subset-by-label mechanism the
    Dynamic-Instability buyer attack uses per round, applied here per task."""
    class_set = set(classes)
    indices = [i for i, (_, label) in enumerate(loader.dataset)
               if int(label) in class_set]
    if not indices:
        raise ValueError(
            f"Buyer dataset contains no samples for task class subset {sorted(class_set)}; "
            f"cannot form the task query."
        )
    subset = Subset(loader.dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=getattr(loader, 'pin_memory', False),
    ), len(indices)


def _reset_bandit_state(seller: AdaptiveAttackerSeller):
    """Re-initialize the UCB bandit's learned state (mirrors __init__).
    Touches state only — the attack construction itself is unchanged."""
    seller.phase = "exploration"
    seller.strategy_history.clear()
    seller.current_strategy = "honest"
    seller.round_counter = 0


def _mean_or_nan(series):
    s = series.dropna()
    return float(s.mean()) if len(s) else float('nan')


def run_multi_task_experiment(cfg) -> None:
    """One multi-task run (one seed): T tasks over a persistent population."""
    mt = cfg.multi_task
    save_root = Path(cfg.experiment.save_path)
    save_root.mkdir(parents=True, exist_ok=True)
    device = cfg.experiment.device

    # ---------- One-time setup: data, model factory, persistent sellers ----------
    (buyer_loader, seller_loaders, test_loader, validation_loader, model_factory,
     seller_extra_args, collate_fn, num_classes) = setup_data_and_model(cfg, device=device)

    if cfg.experiment.dataset_type == "llm":
        raise ValueError("multi_task mode supports the standard (non-LLM) pipeline only.")

    class_subsets = mt.task_class_subsets or _auto_class_subsets(num_classes, mt.num_tasks)

    global_model = model_factory().to(device)
    model_factory.set_global_model(global_model)
    loss_fn = nn.CrossEntropyLoss()
    evaluators = create_evaluators(cfg, device, **seller_extra_args)
    sample_data = _get_sample_data(test_loader, seller_loaders)
    input_shape = tuple(sample_data.shape[1:])

    sellers = {}  # persistent population P, shared by every task's marketplace

    # Persistent cross-task seller state (MEASURED OUTPUT ONLY — never read by
    # the filter): identity, type, cumulative_payment, unpaid_streak, active.
    market_state = {}

    # Per-round market snapshots and per-task summaries for the E2 plots.
    round_snapshots = []
    task_summaries = []
    cumulative_adv_revenue = 0.0
    cumulative_benign_revenue = 0.0

    for task_idx in range(1, mt.num_tasks + 1):
        c_t = class_subsets[(task_idx - 1) % len(class_subsets)]
        task_dir = save_root / f"task_{task_idx}"
        task_dir.mkdir(parents=True, exist_ok=True)

        cfg_task = copy.deepcopy(cfg)
        cfg_task.experiment.save_path = str(task_dir)

        logging.info("=" * 80)
        logging.info(f"🧭 TASK {task_idx}/{mt.num_tasks} — buyer query classes C_t = {c_t}")
        logging.info("=" * 80)

        # --- (a) Buyer issues query t: restrict D_B to class subset C_t ---
        task_buyer_loader, n_query_samples = _restrict_loader_to_classes(
            buyer_loader, c_t, cfg.training.batch_size, collate_fn
        )
        logging.info(f"Task buyer dataset: {n_query_samples} samples in classes {c_t}")

        # --- Model handling between tasks ---
        if task_idx == 1 or mt.carry_model == "reset":
            if task_idx > 1:
                # Fresh random initialization: bypass the stateful factory so
                # the old global weights are NOT copied in.
                global_model = model_factory.base_factory().to(device)
                model_factory.set_global_model(global_model)
                logging.info("Global model RESET to fresh initialization for new task.")
        elif mt.carry_model == "continue":
            logging.info("Global model CONTINUED from previous task.")
        else:
            raise ValueError(f"Unknown carry_model '{mt.carry_model}' (expected 'reset' or 'continue').")
        global_model = model_factory.global_model

        # --- Fresh inner-pipeline components per task (filter state resets) ---
        aggregator = Aggregator(
            global_model=global_model,
            device=torch.device(device),
            loss_fn=loss_fn,
            buyer_data_loader=task_buyer_loader,
            agg_config=cfg_task.aggregation,
        )
        sybil_coordinator = SybilCoordinator(cfg_task.adversary_seller_config.sybil, aggregator)

        marketplace = DataMarketplaceFederated(
            cfg=cfg_task,
            aggregator=aggregator,
            sellers=sellers,
            input_shape=input_shape,
            SellerClass=GradientSeller,
            validation_loader=validation_loader,
            model_factory=model_factory,
            num_classes=num_classes,
            sybil_coordinator=sybil_coordinator,
        )

        if task_idx == 1:
            # Sellers persist across tasks, so their own artifact directories
            # (seller_id/round_history.csv, marketplace_summary.json) belong
            # to the RUN root, not to task_1. Their round histories therefore
            # accumulate over all tasks; per-task governance metrics come from
            # each task's training_log.csv instead.
            cfg_sellers = copy.deepcopy(cfg_task)
            cfg_sellers.experiment.save_path = str(save_root)
            initialize_sellers(
                cfg_sellers, marketplace, seller_loaders, model_factory,
                seller_extra_args, collate_fn, num_classes,
                validation_loader=validation_loader,
            )
            for sid, seller in sellers.items():
                market_state[sid] = {
                    'seller_id': sid,
                    'type': 'adversarial' if 'adv' in sid else 'benign',
                    'cumulative_payment': 0.0,
                    'unpaid_streak': 0,
                    'active': True,
                }
        else:
            # Population persists; re-register sybil adversaries with the
            # task's fresh coordinator (same registration rule as task 1).
            if cfg_task.adversary_seller_config.sybil.is_sybil:
                for sid, seller in sellers.items():
                    if 'adv' in sid and getattr(seller, 'is_sybil', True):
                        sybil_coordinator.register_seller(seller)

        # --- Cross-task adversary state ---
        for sid, seller in sellers.items():
            if isinstance(seller, AdaptiveAttackerSeller) and not mt.bandit_persist:
                _reset_bandit_state(seller)
                logging.info(f"[{sid}] bandit state reset (bandit_persist=False).")
            if isinstance(seller, LazyGradientSeller) and mt.carry_model == "reset":
                # The echoed previous-task delta and the stale_local cache both
                # refer to a discarded model.
                seller.previous_global_delta = None
                seller.previous_median_norm = None
                seller._cached_gradient = None
                seller._cached_at_round = -1

        initialize_root_sellers(cfg_task, marketplace, task_buyer_loader,
                                validation_loader, model_factory)

        # --- (c) Run ONE existing round-sequence with payment/attrition hook ---
        task_state = {
            'rounds': 0,
            'bandit_selected': 0,
            'bandit_rows': 0,
            'payment_undefined_rounds': 0,
        }

        def on_round_end(round_record, _marketplace, _task_idx=task_idx, _task_state=task_state):
            nonlocal cumulative_adv_revenue, cumulative_benign_revenue
            _task_state['rounds'] += 1
            rows = round_record.get('detailed_seller_metrics') or []
            if rows and all(r.get('price_paid') is None for r in rows):
                # Payment was undefined this round (e.g. the aggregator
                # returned empty stats) — a seller cannot be "unpaid" in a
                # round where payment never existed, so skip all payment and
                # attrition accounting.
                _task_state['payment_undefined_rounds'] += 1
                rows = []

            for rec in rows:
                sid = rec.get('seller_id')
                st = market_state.get(sid)
                if st is None or not st['active']:
                    continue
                paid = rec.get('price_paid') or 0.0
                st['cumulative_payment'] += paid
                if paid > 0:
                    st['unpaid_streak'] = 0
                else:
                    st['unpaid_streak'] += 1
                if st['type'] == 'adversarial':
                    cumulative_adv_revenue += paid
                else:
                    cumulative_benign_revenue += paid
                    # ATTRITION RULE: benign sellers leave after exit_k
                    # consecutive unpaid rounds. Adversaries NEVER attrit.
                    if st['unpaid_streak'] >= mt.exit_k:
                        st['active'] = False
                        sellers[sid].is_active = False
                        logging.warning(
                            f"💸 ATTRITION: benign seller {sid} left the market "
                            f"(unpaid for {st['unpaid_streak']} consecutive rounds, "
                            f"task {_task_idx}, round {round_record.get('round')})."
                        )
                if isinstance(sellers.get(sid), AdaptiveAttackerSeller):
                    _task_state['bandit_rows'] += 1
                    _task_state['bandit_selected'] += int(bool(rec.get('selected')))

            active = [s for s in market_state.values() if s['active']]
            n_active_benign = sum(1 for s in active if s['type'] == 'benign')
            n_active_adv = sum(1 for s in active if s['type'] == 'adversarial')
            round_snapshots.append({
                'task': _task_idx,
                'round': round_record.get('round'),
                'n_active_benign': n_active_benign,
                'n_active_adversarial': n_active_adv,
                'adversarial_fraction_active': (
                    n_active_adv / (n_active_benign + n_active_adv)
                    if (n_active_benign + n_active_adv) else float('nan')
                ),
                'cumulative_adversary_revenue': cumulative_adv_revenue,
                'cumulative_benign_revenue': cumulative_benign_revenue,
                'adversary_revenue_share_round': round_record.get('adversary_revenue_share'),
            })

        final_model = run_training_loop(
            cfg_task, marketplace, validation_loader, test_loader, evaluators,
            on_round_end=on_round_end,
        )

        # --- (d) Per-task record: governance + market-composition metrics ---
        run_final_evaluation_and_logging(cfg_task, final_model, test_loader,
                                         evaluators, marketplace)
        generate_marketplace_report(task_dir, marketplace, cfg_task.experiment.global_rounds)

        log_df = pd.read_csv(task_dir / "training_log.csv")
        with open(task_dir / "final_metrics.json") as f:
            final_metrics = json.load(f)

        active = [s for s in market_state.values() if s['active']]
        n_active_benign = sum(1 for s in active if s['type'] == 'benign')
        n_active_adv = sum(1 for s in active if s['type'] == 'adversarial')

        task_summaries.append({
            'task': task_idx,
            'query_classes': json.dumps(c_t),
            'rounds_completed': task_state['rounds'],
            # BSR/MSR per task from the unchanged per-round defense metrics:
            # BSR = 1 - false_positive_rate, MSR = 1 - adversary_detection_rate.
            'bsr': 1.0 - _mean_or_nan(log_df.get('false_positive_rate', pd.Series(dtype=float))),
            'msr': 1.0 - _mean_or_nan(log_df.get('adversary_detection_rate', pd.Series(dtype=float))),
            'acc': final_metrics.get('acc', final_metrics.get('B-Acc')),
            'asr': final_metrics.get('asr'),
            'n_active_benign': n_active_benign,
            'n_active_adversarial': n_active_adv,
            'adversarial_fraction_active': (
                n_active_adv / (n_active_benign + n_active_adv)
                if (n_active_benign + n_active_adv) else float('nan')
            ),
            'cumulative_adversary_revenue': cumulative_adv_revenue,
            'cumulative_benign_revenue': cumulative_benign_revenue,
            'bandit_msr': (
                task_state['bandit_selected'] / task_state['bandit_rows']
                if task_state['bandit_rows'] else None
            ),
            'payment_undefined_rounds': task_state['payment_undefined_rounds'],
        })

        pd.DataFrame(task_summaries).to_csv(save_root / "multi_task_summary.csv", index=False)
        pd.DataFrame(round_snapshots).to_csv(save_root / "multi_task_rounds.csv", index=False)
        save_json_atomic(market_state, save_root / "market_state.json")

        if n_active_benign == 0:
            logging.warning("⚠️  All benign sellers have left the market — "
                            "remaining tasks run on adversaries only.")

    _plot_outputs(save_root, pd.DataFrame(task_summaries), pd.DataFrame(round_snapshots),
                  bandit_persist=mt.bandit_persist)
    logging.info(f"✅ Multi-task experiment finished. Outputs in {save_root}")


def _plot_outputs(save_root: Path, tasks_df: pd.DataFrame, rounds_df: pd.DataFrame,
                  bandit_persist: bool):
    """The E2 figures: attrition, cumulative adversary revenue, market
    composition, per-task governance metrics, and (optionally) bandit MSR."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning(f"matplotlib unavailable ({e}); skipping E2 plots.")
        return

    figs = save_root / "figures"
    figs.mkdir(exist_ok=True)

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(figs / name, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tasks_df['task'], tasks_df['n_active_benign'], marker='o')
    ax.set_xlabel("Task index")
    ax.set_ylabel("Active benign sellers")
    ax.set_title("Honest-seller attrition")
    _save(fig, "attrition_curve.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tasks_df['task'], tasks_df['cumulative_adversary_revenue'],
            marker='o', label='adversary')
    ax.plot(tasks_df['task'], tasks_df['cumulative_benign_revenue'],
            marker='s', label='benign')
    ax.set_xlabel("Task index")
    ax.set_ylabel("Cumulative revenue")
    ax.set_title("Cumulative revenue by seller type")
    ax.legend()
    _save(fig, "cumulative_adversary_revenue.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tasks_df['task'], tasks_df['adversarial_fraction_active'], marker='o')
    ax.set_xlabel("Task index")
    ax.set_ylabel("Adversarial fraction of active sellers")
    ax.set_title("Market composition")
    _save(fig, "market_composition.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tasks_df['task'], tasks_df['bsr'], marker='o', label='BSR')
    ax.plot(tasks_df['task'], tasks_df['msr'], marker='s', label='MSR')
    if tasks_df['acc'].notna().any():
        ax.plot(tasks_df['task'], tasks_df['acc'], marker='^', label='Acc')
    ax.set_xlabel("Task index")
    ax.set_ylabel("Metric value")
    ax.set_title("Per-task governance metrics")
    ax.legend()
    _save(fig, "per_task_metrics.png")

    if bandit_persist and tasks_df['bandit_msr'].notna().any():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tasks_df['task'], tasks_df['bandit_msr'], marker='o')
        ax.set_xlabel("Task index")
        ax.set_ylabel("Bandit adversary selection rate (MSR)")
        ax.set_title("Persistent-bandit exploitation across tasks")
        _save(fig, "bandit_msr_per_task.png")

    logging.info(f"E2 figures written to {figs}")


def main():
    parser = argparse.ArgumentParser(
        description="E2 — multi-round persistent market (outer task loop over the existing run)")
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument("--mode", default="multi_round", choices=["multi_round"],
                        help="Runner mode (only multi_round is defined here)")
    parser.add_argument("--num_tasks", type=int, default=None,
                        help="Override multi_task.num_tasks (default from config: 5)")
    parser.add_argument("--exit_k", type=int, default=None,
                        help="Override multi_task.exit_k (default from config: 3)")
    parser.add_argument("--carry_model", choices=["reset", "continue"], default=None,
                        help="Override multi_task.carry_model (default: reset)")
    parser.add_argument("--bandit_persist", action="store_true", default=None,
                        help="Let the UCB-bandit adversary's learned state persist across tasks")
    cli = parser.parse_args()

    app_config = load_config(cli.config)
    if cli.num_tasks is not None:
        app_config.multi_task.num_tasks = cli.num_tasks
    if cli.exit_k is not None:
        app_config.multi_task.exit_k = cli.exit_k
    if cli.carry_model is not None:
        app_config.multi_task.carry_model = cli.carry_model
    if cli.bandit_persist is not None:
        app_config.multi_task.bandit_persist = cli.bandit_persist

    # ---- Fail-fast validation of the E2 contract ----
    mt = app_config.multi_task
    if mt.num_tasks < 1:
        raise ValueError(f"multi_task.num_tasks must be >= 1, got {mt.num_tasks}")
    if mt.exit_k < 1:
        raise ValueError(f"multi_task.exit_k must be >= 1, got {mt.exit_k}")
    if mt.carry_model not in ("reset", "continue"):
        raise ValueError(f"multi_task.carry_model must be 'reset' or 'continue', got '{mt.carry_model}'")
    method = app_config.aggregation.method
    if method not in PER_SELLER_FILTERS:
        raise ValueError(
            f"multi_task mode requires a per-seller filter (one of {sorted(PER_SELLER_FILTERS)}); "
            f"got aggregation.method='{method}'. Parameter-level pooling/masking methods pay "
            f"everyone, so attrition is undefined."
        )
    if app_config.valuation.payment_model != "weighted":
        raise ValueError(
            "multi_task mode assigns per-round payments via the E1a 'weighted' PaymentModel; "
            f"set valuation.payment_model: weighted (got '{app_config.valuation.payment_model}')."
        )
    if mt.bandit_persist and not app_config.adversary_seller_config.adaptive_attack.is_active:
        raise ValueError(
            "bandit_persist=True but adversary_seller_config.adaptive_attack.is_active is False — "
            "there is no bandit adversary whose state could persist."
        )

    initial_seed = app_config.seed
    for i in range(app_config.n_samples):
        run_cfg = copy.deepcopy(app_config)
        current_seed = initial_seed + i
        run_cfg.seed = current_seed
        set_seed(current_seed)

        run_save_path = Path(run_cfg.experiment.save_path) / f"run_{i}_seed_{current_seed}"
        run_save_path.mkdir(parents=True, exist_ok=True)
        run_cfg.experiment.save_path = str(run_save_path)

        logging.info(f"\n{'=' * 20} Multi-task run {i + 1}/{app_config.n_samples} "
                     f"(Seed: {current_seed}) {'=' * 20}")
        run_multi_task_experiment(run_cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
