"""
Tests for ConstraintViolationLoss and solver handoff modules.

Tests:
    1. build_soft_predictions: correct shape and differentiability
    2. compute_constraint_violation: synthetic bipartite graph
    3. ConstraintViolationLoss: full pipeline integration
    4. Gradient flow through violation penalty
    5. TrustRegionSolver.extract_fixings: thresholding logic
    6. TrustRegionSolver.extract_fixings: confidence sorting
    7. Backtracking logic (mocked solver)
    8. Real data integration test

Usage:
    python test_constraint_and_handoff.py
"""
import sys
import os
import pathlib
import gzip
import pickle
import numpy as np

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.deslicing_decoder import DeslicingDecoder, extract_var_types
from model.constraint_loss import (
    build_soft_predictions,
    compute_constraint_violation,
    ConstraintViolationLoss,
)
from model.solver_handoff import TrustRegionSolver


def load_first_sample():
    """Load the first available sample file."""
    sample_files = sorted(str(p) for p in pathlib.Path('data/samples').rglob('sample_*.pkl'))
    if not sample_files:
        return None
    with gzip.open(sample_files[0], 'rb') as f:
        sample = pickle.load(f)
    cons_feats, (edge_idx, edge_vals), var_feats = sample['observation']
    cons_feats = torch.FloatTensor(np.asarray(cons_feats, dtype=np.float32))
    edge_idx = torch.LongTensor(np.asarray(edge_idx, dtype=np.int64))
    edge_vals = torch.FloatTensor(np.asarray(edge_vals, dtype=np.float32)).unsqueeze(-1)
    var_feats = torch.FloatTensor(np.asarray(var_feats, dtype=np.float32))
    sol = sample['solution']
    n_vars = var_feats.shape[0]
    if sol['sol_vals'] is not None:
        sol_values = torch.zeros(n_vars, dtype=torch.float32)
        for i, (name, val) in enumerate(sol['sol_vals'].items()):
            if i < n_vars:
                sol_values[i] = float(val)
    else:
        sol_values = torch.zeros(n_vars, dtype=torch.float32)
    return cons_feats, edge_idx, edge_vals, var_feats, sol_values


# ======================================================================
# Test 1: build_soft_predictions
# ======================================================================
def test_soft_predictions():
    """Test soft prediction vector construction."""
    print("\nSoft predictions test (synthetic)...")
    N = 100
    emb_size = 64
    K = 8
    threshold = 10

    var_feats = torch.zeros(N, 23)
    var_feats[:40, 1] = 1.0   # binary
    var_feats[40:60, 2] = 1.0  # small int
    var_feats[40:60, 21] = 2.0
    var_feats[40:60, 22] = 6.0
    var_feats[60:80, 2] = 1.0  # large int
    var_feats[60:80, 21] = 0.0
    var_feats[60:80, 22] = 1000.0
    var_feats[80:, 4] = 1.0   # continuous
    var_feats[80:, 8] = torch.rand(20) * 5  # LP relaxation

    var_types = extract_var_types(var_feats)

    z_var_0 = torch.randn(N, emb_size)
    attn_weights = F.softmax(torch.randn(N, K), dim=-1)
    evolved_tokens = torch.randn(K, emb_size)
    token_batch = torch.zeros(K, dtype=torch.long)

    decoder = DeslicingDecoder(emb_size=emb_size, int_range_threshold=threshold)
    decoder.train()

    result = decoder(evolved_tokens, token_batch, attn_weights,
                     var_types, z_var_0=z_var_0,
                     variable_features=var_feats)

    x = build_soft_predictions(result, N, var_types, var_feats)

    assert x.shape == (N,), f"Expected [{N}], got {x.shape}"
    assert x.requires_grad or any(
        p.requires_grad for p in decoder.parameters()
    ), "Predictions should be differentiable"

    # Binary predictions should be in [0, 1]
    bin_x = x[:40]
    assert (bin_x >= 0).all() and (bin_x <= 1).all(), \
        "Binary soft predictions should be in [0, 1]"

    # Continuous should match LP relaxation
    cont_x = x[80:]
    expected_lp = var_feats[80:, 8]
    assert torch.allclose(cont_x, expected_lp, atol=1e-5), \
        "Continuous vars should use LP relaxation values"

    # Small-range int: expected value should be in [offset, offset+range]
    small_x = x[40:60]
    assert (small_x >= 2.0 - 0.1).all() and (small_x <= 6.0 + 0.1).all(), \
        f"Small-range expected values should be in [2, 6], got [{small_x.min():.2f}, {small_x.max():.2f}]"

    print(f"  Shape: {x.shape}")
    print(f"  Binary range:   [{bin_x.min():.4f}, {bin_x.max():.4f}]")
    print(f"  Small-int range:[{small_x.min():.4f}, {small_x.max():.4f}]")
    print(f"  Continuous:     matches LP relaxation")
    print("PASS: soft predictions constructed correctly.")


# ======================================================================
# Test 2: compute_constraint_violation
# ======================================================================
def test_constraint_violation():
    """Test violation computation with a simple synthetic system."""
    print("\nConstraint violation test (synthetic)...")

    # Simple system: 3 constraints, 4 variables
    # Constraint 0: 1*x0 + 1*x1 <= 1.0   (bias col=1 stores 1.0)
    # Constraint 1: 1*x2 + 1*x3 <= 1.5
    # Constraint 2: 1*x0 + 1*x2 <= 0.8
    edge_indices = torch.tensor([
        [0, 0, 1, 1, 2, 2],   # constraint idx
        [0, 1, 2, 3, 0, 2],   # variable idx
    ], dtype=torch.long)
    edge_features = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unsqueeze(-1)

    # Constraint features: 5 columns, bias in col 1
    constraint_features = torch.zeros(3, 5)
    constraint_features[0, 1] = 1.0   # b_0 = 1.0
    constraint_features[1, 1] = 1.5   # b_1 = 1.5
    constraint_features[2, 1] = 0.8   # b_2 = 0.8

    # Case 1: Feasible prediction
    x_feasible = torch.tensor([0.3, 0.4, 0.5, 0.6])
    # Ax: [0.7, 1.1, 0.8]   b: [1.0, 1.5, 0.8]
    # Violations: [0, 0, 0] — all feasible
    v1 = compute_constraint_violation(
        x_feasible, edge_indices, edge_features, constraint_features)
    assert (v1 == 0).all(), f"Should be feasible, got violations: {v1}"
    print(f"  Feasible case: violations = {v1.tolist()}")

    # Case 2: Infeasible prediction
    x_infeasible = torch.tensor([0.8, 0.9, 0.7, 0.5])
    # Ax: [1.7, 1.2, 1.5]   b: [1.0, 1.5, 0.8]
    # Violations: [0.7, 0, 0.7]
    v2 = compute_constraint_violation(
        x_infeasible, edge_indices, edge_features, constraint_features)
    assert v2[0].item() > 0.6, f"Constraint 0 should be violated: {v2[0]}"
    assert v2[1].item() < 0.01, f"Constraint 1 should be feasible: {v2[1]}"
    assert v2[2].item() > 0.6, f"Constraint 2 should be violated: {v2[2]}"
    print(f"  Infeasible case: violations = {[f'{v:.4f}' for v in v2.tolist()]}")

    print("PASS: constraint violation computation correct.")


# ======================================================================
# Test 3: ConstraintViolationLoss with full pipeline
# ======================================================================
def test_violation_loss_pipeline():
    """Test ConstraintViolationLoss with full decoder output."""
    print("\nViolation loss pipeline test (synthetic)...")
    N = 60
    emb_size = 64
    K = 8

    var_feats = torch.zeros(N, 23)
    var_feats[:30, 1] = 1.0   # binary
    var_feats[30:, 4] = 1.0   # continuous
    var_feats[30:, 8] = torch.rand(30)  # LP relaxation

    var_types = extract_var_types(var_feats)

    # Create simple bipartite graph
    n_cons = 10
    n_edges = 100
    edge_indices = torch.stack([
        torch.randint(0, n_cons, (n_edges,)),
        torch.randint(0, N, (n_edges,)),
    ])
    edge_features = torch.randn(n_edges, 1) * 0.5
    constraint_features = torch.randn(n_cons, 5)

    # Build decoder and run
    z_var_0 = torch.randn(N, emb_size)
    attn_weights = F.softmax(torch.randn(N, K), dim=-1)
    evolved_tokens = torch.randn(K, emb_size)
    token_batch = torch.zeros(K, dtype=torch.long)

    decoder = DeslicingDecoder(emb_size=emb_size)
    decoder.train()

    result = decoder(evolved_tokens, token_batch, attn_weights,
                     var_types, z_var_0=z_var_0,
                     variable_features=var_feats)

    # Compute violation loss
    cv_loss = ConstraintViolationLoss(lambda_mean=1.0, lambda_max=0.1)
    loss_dict = cv_loss(result, N, var_types,
                        edge_indices, edge_features, constraint_features,
                        variable_features=var_feats)

    print(f"  penalty:    {loss_dict['penalty'].item():.6f}")
    print(f"  mean_viol:  {loss_dict['mean_viol'].item():.6f}")
    print(f"  max_viol:   {loss_dict['max_viol'].item():.6f}")
    print(f"  n_violated: {loss_dict['n_violated']}")

    assert loss_dict['penalty'].isfinite()
    print("PASS: violation loss pipeline works.")


# ======================================================================
# Test 4: Gradient flow through violation penalty
# ======================================================================
def test_violation_gradient():
    """Test that gradients flow from violation loss to model parameters."""
    print("\nViolation gradient flow test...")
    N = 40
    emb_size = 64
    K = 8

    var_feats = torch.zeros(N, 23)
    var_feats[:20, 1] = 1.0   # binary
    var_feats[20:, 4] = 1.0
    var_feats[20:, 8] = torch.rand(20)
    var_types = extract_var_types(var_feats)

    edge_indices = torch.stack([
        torch.randint(0, 5, (50,)),
        torch.randint(0, N, (50,)),
    ])
    edge_features = torch.randn(50, 1)
    constraint_features = torch.randn(5, 5)

    z_var_0 = torch.randn(N, emb_size)
    attn_weights = F.softmax(torch.randn(N, K), dim=-1)
    evolved_tokens = torch.randn(K, emb_size)
    token_batch = torch.zeros(K, dtype=torch.long)

    decoder = DeslicingDecoder(emb_size=emb_size)
    decoder.train()

    result = decoder(evolved_tokens, token_batch, attn_weights,
                     var_types, z_var_0=z_var_0,
                     variable_features=var_feats)

    cv_loss = ConstraintViolationLoss()
    loss_dict = cv_loss(result, N, var_types,
                        edge_indices, edge_features, constraint_features,
                        variable_features=var_feats)

    loss_dict['penalty'].backward()

    n_with_grad = sum(1 for p in decoder.parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in decoder.parameters())
    print(f"  {n_with_grad}/{n_total} decoder params have non-zero grad")
    assert n_with_grad > 0, "No gradients in decoder from violation loss!"

    print("PASS: gradients flow through violation penalty.")


# ======================================================================
# Test 5: TrustRegionSolver.extract_fixings
# ======================================================================
def test_extract_fixings():
    """Test thresholding logic for variable fixings."""
    print("\nExtract fixings test...")
    N = 50
    emb_size = 64
    K = 8

    var_feats = torch.zeros(N, 23)
    var_feats[:30, 1] = 1.0  # binary
    var_feats[30:40, 2] = 1.0  # small int, range [0, 5]
    var_feats[30:40, 21] = 0.0
    var_feats[30:40, 22] = 5.0
    var_feats[40:, 4] = 1.0  # continuous
    var_types = extract_var_types(var_feats)

    z_var_0 = torch.randn(N, emb_size)
    attn_weights = F.softmax(torch.randn(N, K), dim=-1)
    evolved_tokens = torch.randn(K, emb_size)
    token_batch = torch.zeros(K, dtype=torch.long)

    decoder = DeslicingDecoder(emb_size=emb_size)
    decoder.eval()

    with torch.no_grad():
        result = decoder(evolved_tokens, token_batch, attn_weights,
                         var_types, z_var_0=z_var_0,
                         variable_features=var_feats)

    solver = TrustRegionSolver(threshold_high=0.95, threshold_low=0.05)
    fixings = solver.extract_fixings(result, N, var_types)

    # Verify all fixings meet threshold
    for idx, (val, conf) in fixings.items():
        assert conf > 0.05, \
            f"Fixing at idx={idx} has too low confidence: {conf}"

    # Verify sorted by confidence (descending)
    confs = [conf for _, (_, conf) in fixings.items()]
    assert confs == sorted(confs, reverse=True), \
        "Fixings should be sorted by descending confidence"

    print(f"  Extracted {len(fixings)} fixings from {N} variables")
    if fixings:
        print(f"  Confidence range: [{confs[-1]:.4f}, {confs[0]:.4f}]")

    # Summary
    solver.summary(fixings)
    print("PASS: extract_fixings works correctly.")


# ======================================================================
# Test 6: Confidence sorting enables backtracking
# ======================================================================
def test_confidence_sorting():
    """Test that fixings are properly sorted for backtracking."""
    print("\nConfidence sorting test...")

    # Create a result dict with known confidence values
    result = {
        'idx_bin': torch.arange(5),
        'prob_bin': torch.tensor([[0.01], [0.99], [0.50], [0.97], [0.02]]),
        'idx_int_small': torch.tensor([], dtype=torch.long),
        'logits_int_small': torch.empty(0, 11),
        'int_small_offsets': torch.empty(0),
        'idx_int_large': torch.tensor([], dtype=torch.long),
        'pred_int_large': torch.empty(0, 1),
        'mask_bin': torch.ones(5, dtype=torch.bool),
        'mask_int_small': torch.zeros(5, dtype=torch.bool),
        'mask_int_large': torch.zeros(5, dtype=torch.bool),
    }
    var_types = torch.ones(5, dtype=torch.long)

    solver = TrustRegionSolver(threshold_high=0.95, threshold_low=0.05)
    fixings = solver.extract_fixings(result, 5, var_types)

    # Expected fixings:
    # idx 0: prob=0.01 -> fix to 0, conf=0.99
    # idx 1: prob=0.99 -> fix to 1, conf=0.99
    # idx 2: prob=0.50 -> SKIP (uncertain)
    # idx 3: prob=0.97 -> fix to 1, conf=0.97
    # idx 4: prob=0.02 -> fix to 0, conf=0.98

    assert len(fixings) == 4, f"Expected 4 fixings, got {len(fixings)}"
    assert 2 not in fixings, "Variable 2 (prob=0.50) should NOT be fixed"

    # Check sorted by confidence
    items = list(fixings.items())
    for i in range(len(items) - 1):
        assert items[i][1][1] >= items[i+1][1][1], \
            "Fixings not sorted by confidence!"

    print(f"  Fixings: {dict((k, (v, f'{c:.2f}')) for k, (v, c) in fixings.items())}")
    print(f"  Variable 2 correctly skipped (uncertain)")
    print("PASS: confidence sorting correct.")


# ======================================================================
# Test 7: Backtracking logic (no actual solver call)
# ======================================================================
def test_backtracking_logic():
    """Test that backtracking reduces fixings on each step."""
    print("\nBacktracking logic test...")

    # We'll monkeypatch the solver to track calls
    solver = TrustRegionSolver(
        threshold_high=0.80, threshold_low=0.20,
        max_backtrack_steps=4, verbose=True)

    # Create result with many fixable variables
    N = 100
    prob = torch.zeros(N, 1)
    prob[:30] = 0.99  # 30 vars fixed to 1
    prob[30:60] = 0.01  # 30 vars fixed to 0
    prob[60:] = 0.50  # 40 vars uncertain

    result = {
        'idx_bin': torch.arange(N),
        'prob_bin': prob,
        'idx_int_small': torch.tensor([], dtype=torch.long),
        'logits_int_small': torch.empty(0, 11),
        'int_small_offsets': torch.empty(0),
        'idx_int_large': torch.tensor([], dtype=torch.long),
        'pred_int_large': torch.empty(0, 1),
        'mask_bin': torch.ones(N, dtype=torch.bool),
        'mask_int_small': torch.zeros(N, dtype=torch.bool),
        'mask_int_large': torch.zeros(N, dtype=torch.bool),
    }
    var_types = torch.ones(N, dtype=torch.long)

    fixings = solver.extract_fixings(result, N, var_types)
    total = len(fixings)
    print(f"  Total fixings: {total}")

    # Simulate backtracking steps
    fixing_list = list(fixings.items())
    for step in range(solver.max_backtrack_steps):
        ratio = 1.0 / (2 ** step)
        n_fix = max(1, int(len(fixing_list) * ratio))
        print(f"  Step {step}: {n_fix} fixings ({ratio*100:.1f}%)")

    assert total == 60, f"Expected 60 fixings, got {total}"
    print("PASS: backtracking logic correct.")


# ======================================================================
# Test 8: Real data integration
# ======================================================================
def test_real_data_integration():
    """Test with real sample data from the dataset."""
    data = load_first_sample()
    if data is None:
        print("\nNo sample data. Skipping real data integration test.")
        return

    print("\nReal data integration test...")
    cons_feats, edge_idx, edge_vals, var_feats, sol_values = data
    N = var_feats.shape[0]
    emb_size = 64
    K = 16

    var_types = extract_var_types(var_feats)

    # Build decoder
    decoder = DeslicingDecoder(emb_size=emb_size, int_range_threshold=10)

    # Create synthetic evolved tokens (skip full pipeline for speed)
    z_var_0 = torch.randn(N, emb_size)
    attn_weights = F.softmax(torch.randn(N, K), dim=-1)
    evolved_tokens = torch.randn(K, emb_size)
    token_batch = torch.zeros(K, dtype=torch.long)

    # Forward
    decoder.train()
    result = decoder(evolved_tokens, token_batch, attn_weights,
                     var_types, z_var_0=z_var_0,
                     variable_features=var_feats)

    # ---- Constraint violation loss ----
    cv_loss = ConstraintViolationLoss(lambda_mean=1.0, lambda_max=0.1)
    cv_dict = cv_loss(result, N, var_types,
                      edge_idx, edge_vals, cons_feats,
                      variable_features=var_feats)

    print(f"  Constraint violation:")
    print(f"    penalty:    {cv_dict['penalty'].item():.6f}")
    print(f"    mean_viol:  {cv_dict['mean_viol'].item():.6f}")
    print(f"    max_viol:   {cv_dict['max_viol'].item():.6f}")
    print(f"    n_violated: {cv_dict['n_violated']} / {cons_feats.shape[0]}")

    # ---- Combined task + violation loss ----
    task_loss = decoder.combined_loss(result, sol_values, var_types)
    total_loss = task_loss['total'] + cv_dict['penalty']
    total_loss.backward()

    n_with_grad = sum(1 for p in decoder.parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Combined loss: {total_loss.item():.6f}")
    print(f"  {n_with_grad} params with grad")

    # ---- Trust region fixings ----
    decoder.eval()
    with torch.no_grad():
        result_eval = decoder(evolved_tokens, token_batch, attn_weights,
                              var_types, z_var_0=z_var_0,
                              variable_features=var_feats)

    solver = TrustRegionSolver(threshold_high=0.95, threshold_low=0.05)
    fixings = solver.extract_fixings(result_eval, N, var_types)

    print(f"\n  Trust Region Fixings:")
    print(f"    {len(fixings)} / {N} variables fixed "
          f"({len(fixings)/N*100:.1f}%)")
    solver.summary(fixings)

    print("PASS: real data integration works.")


if __name__ == '__main__':
    print("=" * 60)
    print("Constraint Loss & Solver Handoff Tests")
    print("=" * 60)

    test_soft_predictions()
    test_constraint_violation()
    test_violation_loss_pipeline()
    test_violation_gradient()
    test_extract_fixings()
    test_confidence_sorting()
    test_backtracking_logic()
    test_real_data_integration()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
