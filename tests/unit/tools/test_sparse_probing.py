"""Tests for leakage-safe k-sparse probing."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from transformer_lens.tools.analysis import fit_sparse_probe, sweep_sparse_probe


def test_selector_oracle_exact_indices():
    torch.manual_seed(0)
    X = torch.randn(40, 6)
    # make feature 2 strongly predictive
    y = torch.tensor([0] * 20 + [1] * 20)
    X[:20, 2] -= 2
    X[20:, 2] += 2
    probe = fit_sparse_probe(X, y, k=2, seed=1)
    # independently compute train-only mean diff
    train_idx = probe.train_indices
    X_train = X[train_idx]
    y_train = y[train_idx]
    pos = probe.positive_label
    pos_mask = y_train == pos
    scores = X_train[pos_mask].float().mean(0) - X_train[~pos_mask].float().mean(0)
    abs_scores = scores.abs()
    expected = torch.argsort(abs_scores, descending=True, stable=True)[:2].tolist()
    assert probe.selected_indices == expected


def test_planted_sparse_feature_k1():
    torch.manual_seed(0)
    n, d = 80, 20
    X = torch.randn(n, d)
    y = torch.tensor([0] * 40 + [1] * 40)
    X[:40, 3] -= 3
    X[40:, 3] += 3
    probe = fit_sparse_probe(X, y, k=1, seed=0)
    assert probe.selected_indices[0] == 3
    assert probe.metrics.f1 > 0.8


def test_distributed_improves_with_k():
    torch.manual_seed(0)
    n, d = 200, 20
    X = torch.randn(n, d)
    y = torch.tensor([0] * 100 + [1] * 100)
    for j in range(3):
        X[:100, j] -= 1.0
        X[100:, j] += 1.0
    sweep = sweep_sparse_probe(X, y, ks=[1, 3], seed=0)
    f1_1 = sweep.probes[0].metrics.f1
    f1_3 = sweep.probes[1].metrics.f1
    assert f1_3 > f1_1 + 0.05


def test_leakage_guard_standardize():
    torch.manual_seed(1)
    n, d = 40, 5
    X = torch.randn(n, d)
    y = torch.tensor([0] * 20 + [1] * 20)
    p1 = fit_sparse_probe(X, y, k=2, preprocess="standardize", seed=42)
    X_mut = X.clone()
    assert p1.test_indices is not None
    X_mut[p1.test_indices, 0] = 999
    p2 = fit_sparse_probe(X_mut, y, k=2, preprocess="standardize", seed=42)
    assert p1.preprocess_mean is not None
    assert p2.preprocess_mean is not None
    assert p1.preprocess_scale is not None
    assert p2.preprocess_scale is not None
    assert p1.selected_indices == p2.selected_indices
    assert p1.selected_scores == p2.selected_scores
    assert torch.equal(p1.preprocess_mean, p2.preprocess_mean)
    assert torch.equal(p1.preprocess_scale, p2.preprocess_scale)
    assert torch.equal(p1.coefficients, p2.coefficients)
    assert p1.intercept == p2.intercept
    assert p1.objective == p2.objective
    # metadata matches hand-computed train-only stats
    train_sel = X[p1.train_indices][:, torch.tensor(p1.selected_indices)]
    mean = train_sel.mean(0)
    std = train_sel.std(0, unbiased=False)
    scale = torch.where(std == 0, torch.ones_like(std), std)
    assert torch.allclose(p1.preprocess_mean, mean.cpu(), atol=1e-6)
    assert torch.allclose(p1.preprocess_scale, scale.cpu(), atol=1e-6)


def test_controls_deterministic_and_margins():
    torch.manual_seed(0)
    n, d = 80, 20
    X = torch.randn(n, d)
    y = torch.tensor([0] * 40 + [1] * 40)
    X[:40, 3] -= 3
    X[40:, 3] += 3
    sweep = sweep_sparse_probe(X, y, ks=[1, 3], seed=0, n_random_subsets=5, n_label_shuffles=5)
    sweep2 = sweep_sparse_probe(X, y, ks=[1, 3], seed=0, n_random_subsets=5, n_label_shuffles=5)
    assert [p.metrics.f1 for p in sweep.probes] == [p.metrics.f1 for p in sweep2.probes]
    assert sweep.probes[1].selected_indices[:1] == sweep.probes[0].selected_indices
    assert [
        [(control.selected_indices, control.metrics) for control in controls]
        for controls in sweep.random_controls
    ] == [
        [(control.selected_indices, control.metrics) for control in controls]
        for controls in sweep2.random_controls
    ]
    assert [
        [(control.selected_indices, control.metrics) for control in controls]
        for controls in sweep.label_shuffle_controls
    ] == [
        [(control.selected_indices, control.metrics) for control in controls]
        for controls in sweep2.label_shuffle_controls
    ]
    # controls are on average below planted result (stochastic; allow outliers)
    planted_f1 = sweep.probes[0].metrics.f1
    rand_mean = sum(rc.metrics.f1 for rc in sweep.random_controls[0]) / max(
        1, len(sweep.random_controls[0])
    )
    shuf_mean = sum(lc.metrics.f1 for lc in sweep.label_shuffle_controls[0]) / max(
        1, len(sweep.label_shuffle_controls[0])
    )
    assert rand_mean < planted_f1 - 0.1
    assert shuf_mean < planted_f1 - 0.1
    for controls in sweep.random_controls:
        for control in controls:
            assert len(control.selected_indices) == len(set(control.selected_indices))
            assert all(0 <= index < d for index in control.selected_indices)


def test_optimizer_convergence_and_failure():
    # tiny analytic dataset should converge
    Xa = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    ya = torch.tensor([0, 0, 1, 1])
    pa = fit_sparse_probe(Xa, ya, k=1, seed=0)
    assert pa.converged
    assert pa.grad_norm <= 1e-5
    assert 0 < pa.n_iter <= pa.max_iter
    assert pa.max_iter == 100
    assert pa.grad_threshold == 1e-5
    with pytest.raises(FrozenInstanceError):
        setattr(pa, "seed", 1)
    # forced non-convergence via tiny max_iter and tight threshold
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_sparse_probe(Xa, ya, k=1, seed=0, grad_threshold=1e-12, max_iter=1)


def test_validation_cases():
    X = torch.randn(10, 4)
    y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # empty classes
    with pytest.raises(ValueError, match="exactly two values"):
        fit_sparse_probe(X, torch.tensor([0] * 10), k=1, seed=0)
    # non-binary
    with pytest.raises(ValueError, match="exactly two values"):
        fit_sparse_probe(X, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]), k=1, seed=0)
    # non-finite
    Xinf = X.clone()
    Xinf[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        fit_sparse_probe(Xinf, y, k=1, seed=0)
    # invalid k
    with pytest.raises(ValueError, match="k must satisfy"):
        fit_sparse_probe(X, y, k=0, seed=0)
    # duplicate ks
    with pytest.raises(ValueError, match="unique"):
        sweep_sparse_probe(X, y, ks=[1, 1], seed=0)
    with pytest.raises(ValueError, match="grad_threshold"):
        fit_sparse_probe(X, y, k=1, seed=0, grad_threshold=2.0)
    with pytest.raises(ValueError, match="max_iter"):
        sweep_sparse_probe(X, y, ks=[1], seed=0, max_iter=0)
    # constant selected columns not crash
    Xc = torch.randn(20, 3)
    Xc[:, 1] = 5.0
    y2 = torch.tensor([0] * 10 + [1] * 10)
    Xc[:10, 0] -= 2
    Xc[10:, 0] += 2
    p = fit_sparse_probe(Xc, y2, k=3, preprocess="standardize", seed=0)
    idx = p.selected_indices.index(1)
    assert p.preprocess_scale is not None
    assert p.preprocess_scale[idx] == 1.0

    # zero disables controls
    sweep = sweep_sparse_probe(X, y, ks=[1, 2], seed=0, n_random_subsets=0, n_label_shuffles=0)
    assert sweep.random_controls == [[], []]
    assert sweep.label_shuffle_controls == [[], []]


def test_global_rng_unchanged():
    torch.manual_seed(0)
    X = torch.randn(20, 5)
    y = torch.tensor([0] * 10 + [1] * 10)
    before = torch.get_rng_state().clone()
    fit_sparse_probe(X, y, k=1, seed=123)
    sweep_sparse_probe(
        X,
        y,
        ks=[1, 2],
        seed=123,
        n_random_subsets=1,
        n_label_shuffles=1,
    )
    after = torch.get_rng_state()
    assert torch.equal(before, after)
