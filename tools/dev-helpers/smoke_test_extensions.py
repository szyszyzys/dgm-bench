"""Smoke tests for the E1-E4 extensions (pure-logic paths, no full training).

Run from the repo root:
    python tools/dev-helpers/smoke_test_extensions.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn


def _fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def test_fltrust_threshold():
    from src.mechanism.gradient.aggregators.fltrust_threshold import FLTrustThresholdAggregator

    model = nn.Linear(4, 2, bias=False)
    agg = FLTrustThresholdAggregator(
        global_model=model, device=torch.device("cpu"), loss_fn=nn.CrossEntropyLoss(),
        buyer_data_loader=None, tau=0.5, clip_norm=None,
    )
    root = [torch.ones_like(p) for p in model.parameters()]
    updates = {
        "bn_aligned": [torch.ones_like(p) for p in model.parameters()],        # cos = 1
        "adv_opposed": [-torch.ones_like(p) for p in model.parameters()],      # cos = -1 -> relu 0
        "bn_partial": [torch.ones_like(p) * 0.5 for p in model.parameters()],  # cos = 1 (direction same)
    }
    grad, selected, outliers, stats = agg.aggregate(
        global_epoch=1, seller_updates=updates, root_gradient=root)

    assert set(selected) == {"bn_aligned", "bn_partial"}, f"selected={selected}"
    assert outliers == ["adv_opposed"], f"outliers={outliers}"
    assert abs(sum(stats['seller_weights'].values()) - 1.0) < 1e-6
    assert stats['seller_weights']['adv_opposed'] == 0.0
    assert stats['tau'] == 0.5

    # tau above every trust score -> nobody accepted, zero update
    agg_strict = FLTrustThresholdAggregator(
        global_model=model, device=torch.device("cpu"), loss_fn=nn.CrossEntropyLoss(),
        buyer_data_loader=None, tau=1.0, clip_norm=None,
    )
    noisy = {"bn_0": [torch.randn_like(p) for p in model.parameters()]}
    grad2, sel2, out2, _ = agg_strict.aggregate(
        global_epoch=1, seller_updates=noisy, root_gradient=root)
    if not sel2:
        assert all(torch.allclose(g, torch.zeros_like(g)) for g in grad2)

    # invalid tau fails fast
    try:
        FLTrustThresholdAggregator(
            global_model=model, device=torch.device("cpu"), loss_fn=nn.CrossEntropyLoss(),
            buyer_data_loader=None, tau=1.5, clip_norm=None)
        _fail("tau=1.5 should raise")
    except ValueError:
        pass
    print("PASS: fltrust_threshold")


def test_banzhaf_additive_game():
    from src.mechanism.gradient.valuation.kernel_shapely import KernelSHAPEvaluator

    rng = np.random.RandomState(0)
    n, m = 5, 4000
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X_rand = rng.randint(0, 2, size=(m, n))
    X = np.vstack([np.zeros(n), np.ones(n), X_rand])
    y = X @ w  # additive game: v(S) = sum of member weights

    ev = KernelSHAPEvaluator.__new__(KernelSHAPEvaluator)
    vals = ev._banzhaf_from_samples(X, y)
    if vals is None or not np.allclose(vals, w, atol=0.25):
        _fail(f"banzhaf {vals} != weights {w}")
    print("PASS: banzhaf (additive game, MSR estimator)")


def test_leastcore_additive_game():
    from src.mechanism.gradient.valuation.kernel_shapely import KernelSHAPEvaluator

    rng = np.random.RandomState(1)
    n, m = 4, 200
    w = np.array([1.0, 2.0, 3.0, 4.0])
    X_rand = rng.randint(0, 2, size=(m, n))
    X = np.vstack([np.zeros(n), np.ones(n), X_rand])
    y = 10.0 + X @ w  # constant offset checks the v(S)-v(empty) normalization

    ev = KernelSHAPEvaluator.__new__(KernelSHAPEvaluator)
    ev.leastcore_max_coalitions = 2000
    ev.leastcore_timeout = 60.0
    x = ev._leastcore_from_samples(X, y)
    if x is None:
        _fail("leastcore returned None on a feasible additive game")
    if abs(x.sum() - w.sum()) > 1e-6:
        _fail(f"leastcore efficiency violated: sum={x.sum()} vs v(N)={w.sum()}")
    # For an additive game the core is nonempty (x = w is in it), so every
    # sampled coalition's deficit at the solution must be <= ~0.
    v = (X @ w)
    deficits = v[2:] - X[2:] @ x
    if deficits.max() > 1e-6:
        _fail(f"leastcore max deficit {deficits.max()} > 0 on additive game")
    print("PASS: leastcore (additive game LP)")


def test_analytic_gaussian_sigma():
    from src.participants.seller.gradient_seller import GradientSeller

    cal = GradientSeller._calibrate_analytic_gaussian_sigma
    for eps in (0.5, 1.0, 4.0, 8.0):
        sigma = cal(eps, 1e-5, 1.0)
        # verify the (eps, delta) condition holds at the returned sigma
        def phi(t):
            return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
        a, b = 1.0 / (2 * sigma), eps * sigma
        delta_at = phi(a - b) - math.exp(eps) * phi(-a - b)
        if delta_at > 1e-5 * (1 + 1e-6):
            _fail(f"analytic sigma {sigma} violates delta at eps={eps}: {delta_at}")
    s1 = cal(1.0, 1e-5, 1.0)
    s8 = cal(8.0, 1e-5, 1.0)
    if not (s8 < s1):
        _fail("sigma should decrease with epsilon")
    classic1 = math.sqrt(2 * math.log(1.25 / 1e-5))
    if not (s1 < classic1):
        _fail(f"analytic sigma {s1} should beat classic {classic1} at eps=1")
    try:
        cal(-1.0, 1e-5, 1.0)
        _fail("negative epsilon should raise")
    except ValueError:
        pass
    print(f"PASS: analytic gaussian calibration (eps=1: {s1:.3f} vs classic {classic1:.3f})")


def test_weighted_payment():
    from src.mechanism.gradient.valuation.contribution_evaluator import ContributionEvaluator

    class _V:  # minimal cfg stand-ins
        payment_model = "weighted"

    class _A:
        method = "fltrust"

    class _E:
        compute_gradient_similarity = False

    class _Cfg:
        valuation = _V()
        aggregation = _A()
        experiment = _E()

    ev = ContributionEvaluator(_Cfg())
    g = lambda: [torch.ones(2)]
    seller_gradients = {"bn_1": g(), "bn_2": g(), "adv_1": g()}
    stats = {sid: {} for sid in seller_gradients}
    agg_stats = {'seller_weights': {"bn_1": 0.5, "bn_2": 0.25, "adv_1": 0.25}}
    vals, metrics = ev.evaluate_round(
        round_number=1, seller_gradients=seller_gradients, seller_stats=stats,
        oracle_gradient=g(), buyer_gradient=g(), aggregated_gradient=g(),
        aggregation_stats=agg_stats,
        selected_ids=["bn_1", "adv_1"],   # bn_2 selected out -> unpaid
        outlier_ids=["bn_2"],
    )
    paid = {sid: vals[sid]['price_paid'] for sid in seller_gradients}
    assert abs(sum(paid.values()) - 1.0) < 1e-9, paid
    assert paid["bn_2"] == 0.0
    assert abs(paid["bn_1"] - 2 / 3) < 1e-9 and abs(paid["adv_1"] - 1 / 3) < 1e-9
    assert abs(metrics['adversary_revenue_share'] - 1 / 3) < 1e-9

    # parameter-level method without seller weights -> payment_undefined
    _Cfg.aggregation.method = "trimmed_mean"
    vals2, metrics2 = ev.evaluate_round(
        round_number=1, seller_gradients=seller_gradients, seller_stats=stats,
        oracle_gradient=g(), buyer_gradient=g(), aggregated_gradient=g(),
        aggregation_stats={},
        selected_ids=list(seller_gradients), outlier_ids=[],
    )
    assert metrics2.get('payment_undefined') is True
    assert all('price_paid' not in vals2[sid] for sid in seller_gradients)

    # fedavg fallback: uniform over selected
    _Cfg.aggregation.method = "fedavg"
    vals3, metrics3 = ev.evaluate_round(
        round_number=1, seller_gradients=seller_gradients, seller_stats=stats,
        oracle_gradient=g(), buyer_gradient=g(), aggregated_gradient=g(),
        aggregation_stats={},
        selected_ids=list(seller_gradients), outlier_ids=[],
    )
    assert all(abs(vals3[sid]['price_paid'] - 1 / 3) < 1e-9 for sid in seller_gradients)
    assert abs(metrics3['adversary_revenue_share'] - 1 / 3) < 1e-9
    print("PASS: weighted payment + adversary_revenue_share + payment_undefined")


if __name__ == "__main__":
    test_fltrust_threshold()
    test_banzhaf_additive_game()
    test_leastcore_additive_game()
    test_analytic_gaussian_sigma()
    test_weighted_payment()
    print("\nAll extension smoke tests passed.")
