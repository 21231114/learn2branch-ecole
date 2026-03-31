"""
Smoke test for DeslicingDecoder module (Step 4) with Dynamic Routing.

Tests:
    1. var_types extraction from raw features
    2. Integer routing (small-range vs large-range)
    3. Full 4-step pipeline shape correctness
    4. Gradient flow through entire pipeline
    5. Batched forward pass
    6. Focal loss computation
    7. Combined loss with ground-truth solutions
    8. predict_full reconstruction
    9. Empty mask handling (all binary, no integers)
    10. Mixed types with dynamic routing (synthetic)

Usage:
    python test_deslicing_decoder.py
"""
import sys
import os
import pathlib
import gzip
import pickle
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.graph_init import GraphInitialization
from model.adaptive_slicing import AdaptiveSlicing
from model.latent_evolution import LatentTrajectoryEvolution
from model.deslicing_decoder import (
    DeslicingDecoder, extract_var_types, extract_int_routing,
)
from utilities import SolutionGraphDataset


def load_first_sample():
    """Load the first available sample file and return all info."""
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

    # Extract solution
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


def build_full_pipeline(cons_nfeats, var_nfeats, emb_size=64, K=32, T=3):
    """Build the full 4-step pipeline."""
    graph_init = GraphInitialization(
        cons_nfeats=cons_nfeats, var_nfeats=var_nfeats, emb_size=emb_size,
    )
    slicer = AdaptiveSlicing(
        emb_size=emb_size, n_slices=K, n_heads=4, dropout=0.0,
    )
    evolver = LatentTrajectoryEvolution(
        emb_size=emb_size, n_layers=4, n_heads=4,
        n_evolve_steps=T, dropout=0.0, stochastic_depth_rate=0.0,
    )
    decoder = DeslicingDecoder(
        emb_size=emb_size, int_range_threshold=10, dropout=0.0,
    )
    return graph_init, slicer, evolver, decoder


def run_full_pipeline(graph_init, slicer, evolver, decoder,
                      cons_feats, edge_idx, edge_vals, var_feats,
                      var_batch=None):
    """Run Steps 1-4."""
    # Step 1
    z_var_0 = graph_init(cons_feats, edge_idx, edge_vals, var_feats)
    # Step 2
    tokens_0, token_batch, attn_weights = slicer(z_var_0, batch=var_batch)
    # Step 3
    evolved, token_batch_out, intermediates = evolver(tokens_0, token_batch)
    # Step 4
    var_types = extract_var_types(var_feats)
    result = decoder(evolved, token_batch_out, attn_weights,
                     var_types, z_var_0=z_var_0, var_batch=var_batch,
                     variable_features=var_feats)
    return z_var_0, attn_weights, evolved, result, var_types, intermediates


# ======================================================================
# Test 1: var_types extraction
# ======================================================================
def test_var_types_extraction():
    """Test extract_var_types from raw features."""
    data = load_first_sample()
    if data is None:
        print("No samples found. Skipping.")
        return
    _, _, _, var_feats, _ = data

    var_types = extract_var_types(var_feats)
    n_bin = (var_types == 1).sum().item()
    n_int = (var_types == 2).sum().item()
    n_con = (var_types == 0).sum().item()

    print(f"var_types: {var_feats.shape[0]} variables")
    print(f"  binary:     {n_bin}")
    print(f"  integer:    {n_int}")
    print(f"  continuous: {n_con}")

    # For setcover, all should be binary
    assert n_bin == var_feats.shape[0], \
        f"Expected all binary for setcover, got {n_bin}/{var_feats.shape[0]}"
    assert var_types.shape == (var_feats.shape[0],)
    print("PASS: var_types extraction correct.")


# ======================================================================
# Test 2: Integer routing
# ======================================================================
def test_int_routing():
    """Test extract_int_routing splits integers correctly."""
    print("\nInteger routing test (synthetic)...")
    N = 100
    F_dim = 23

    # Create synthetic variable features
    var_feats = torch.zeros(N, F_dim)

    # 30 binary vars (indices 0-29)
    var_feats[:30, 1] = 1.0  # is_type_binary
    # 20 small-range integer vars (indices 30-49): range 0-5
    var_feats[30:50, 2] = 1.0  # is_type_integer
    var_feats[30:50, 21] = 0.0   # lb_global = 0
    var_feats[30:50, 22] = 5.0   # ub_global = 5 (range=5, <=10)
    # 20 large-range integer vars (indices 50-69): range 0-100
    var_feats[50:70, 2] = 1.0  # is_type_integer
    var_feats[50:70, 21] = 0.0   # lb_global = 0
    var_feats[50:70, 22] = 100.0 # ub_global = 100 (range=100, >10)
    # 10 unbounded integer vars (indices 70-79): lb=0, ub=inf
    var_feats[70:80, 2] = 1.0
    var_feats[70:80, 21] = 0.0
    var_feats[70:80, 22] = 1e20  # unbounded
    # 20 continuous vars (indices 80-99)
    var_feats[80:, 4] = 1.0  # is_type_continuous

    var_types = extract_var_types(var_feats)
    assert (var_types == 1).sum() == 30, "Expected 30 binary"
    assert (var_types == 2).sum() == 50, "Expected 50 integer"
    assert (var_types == 0).sum() == 20, "Expected 20 continuous"

    mask_s, mask_l, offsets, ranges = extract_int_routing(
        var_feats, var_types, threshold=10)

    n_small = mask_s.sum().item()
    n_large = mask_l.sum().item()
    print(f"  small-range: {n_small}  (expected 20)")
    print(f"  large-range: {n_large}  (expected 30)")

    assert n_small == 20, f"Expected 20 small-range, got {n_small}"
    assert n_large == 30, f"Expected 30 large-range, got {n_large}"

    # Check offsets and ranges for small-range
    assert offsets.shape == (20,)
    assert ranges.shape == (20,)
    assert (offsets == 0.0).all(), "All small-range lb=0"
    assert (ranges == 6).all(), "range = ub - lb + 1 = 5 - 0 + 1 = 6"

    print("PASS: integer routing correct.")


# ======================================================================
# Test 3: Full pipeline single sample
# ======================================================================
def test_single_sample():
    """Test full 4-step pipeline with a single graph."""
    data = load_first_sample()
    if data is None:
        print("No samples. Skipping.")
        return
    cons_feats, edge_idx, edge_vals, var_feats, sol_values = data
    N = var_feats.shape[0]
    emb_size = 64
    K = 32

    print(f"\nFull pipeline: {N} variables, K={K} slices")

    graph_init, slicer, evolver, decoder = build_full_pipeline(
        cons_feats.shape[1], var_feats.shape[1], emb_size, K,
    )

    for m in [graph_init, slicer, evolver, decoder]:
        m.eval()

    with torch.no_grad():
        z_var_0, attn_weights, evolved, result, var_types, _ = \
            run_full_pipeline(
                graph_init, slicer, evolver, decoder,
                cons_feats, edge_idx, edge_vals, var_feats,
            )

    N_bin = (var_types == 1).sum().item()
    N_int_s = result['mask_int_small'].sum().item()
    N_int_l = result['mask_int_large'].sum().item()

    print(f"  z_out:           {result['z_out'].shape}")
    print(f"  prob_bin:        {result['prob_bin'].shape}  ({N_bin} binary)")
    print(f"  logits_int_small:{result['logits_int_small'].shape}  ({N_int_s} small-range)")
    print(f"  pred_int_large:  {result['pred_int_large'].shape}  ({N_int_l} large-range)")

    assert result['z_out'].shape == (N, emb_size)
    assert result['prob_bin'].shape == (N_bin, 1)
    assert not result['z_out'].isnan().any()

    if N_bin > 0:
        assert (result['prob_bin'] >= 0).all() and (result['prob_bin'] <= 1).all()

    print("PASS: full pipeline shapes and values correct.")


# ======================================================================
# Test 4: Gradient flow
# ======================================================================
def test_gradient_flow():
    """Test gradients flow through Step1-Step4."""
    data = load_first_sample()
    if data is None:
        print("No samples. Skipping.")
        return

    print("\nGradient flow test (full 4-step pipeline)...")
    cons_feats, edge_idx, edge_vals, var_feats, sol_values = data

    graph_init, slicer, evolver, decoder = build_full_pipeline(
        cons_feats.shape[1], var_feats.shape[1],
    )
    for m in [graph_init, slicer, evolver, decoder]:
        m.train()

    z_var_0, attn_weights, evolved, result, var_types, _ = \
        run_full_pipeline(
            graph_init, slicer, evolver, decoder,
            cons_feats, edge_idx, edge_vals, var_feats,
        )

    loss_dict = decoder.combined_loss(result, sol_values, var_types)
    loss = loss_dict['total']
    print(f"  combined loss: {loss.item():.6f}")
    print(f"  binary loss:   {loss_dict['binary'].item():.6f}")
    print(f"  int_small:     {loss_dict['int_small'].item():.6f}")
    print(f"  int_large:     {loss_dict['int_large'].item():.6f}")
    print(f"  rounding:      {loss_dict['rounding'].item():.6f}")

    loss.backward()

    for name, module in [("GraphInit", graph_init),
                         ("AdaptiveSlicing", slicer),
                         ("LatentEvolution", evolver),
                         ("DeslicingDecoder", decoder)]:
        n_with = sum(1 for _, p in module.named_parameters()
                     if p.grad is not None and p.grad.abs().sum() > 0)
        n_total = sum(1 for _ in module.parameters())
        print(f"  {name}: {n_with}/{n_total} params have non-zero grad")
        assert n_with > 0, f"No gradients in {name}!"

    print("PASS: gradients flow through entire 4-step pipeline.")


# ======================================================================
# Test 5: Batched forward
# ======================================================================
def test_batched_forward():
    """Test batched 4-step pipeline."""
    sample_files = sorted(str(p) for p in pathlib.Path('data/samples').rglob('sample_*.pkl'))
    if len(sample_files) < 2:
        print("Not enough samples. Skipping.")
        return

    sample_files = sample_files[:3]
    B = len(sample_files)
    print(f"\nBatch test with {B} samples...")

    dataset = SolutionGraphDataset(sample_files)
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=B)

    first = dataset[0]
    emb_size = 64
    K = 32

    graph_init, slicer, evolver, decoder = build_full_pipeline(
        first.constraint_features.shape[1],
        first.variable_features.shape[1],
        emb_size, K,
    )
    for m in [graph_init, slicer, evolver, decoder]:
        m.eval()

    for batch_data in loader:
        var_counts = [dataset[i].variable_features.shape[0] for i in range(B)]
        var_batch = torch.cat([
            torch.full((c,), i, dtype=torch.long)
            for i, c in enumerate(var_counts)
        ])
        N_total = sum(var_counts)

        with torch.no_grad():
            _, _, _, result, var_types, _ = run_full_pipeline(
                graph_init, slicer, evolver, decoder,
                batch_data.constraint_features, batch_data.edge_index,
                batch_data.edge_attr, batch_data.variable_features,
                var_batch=var_batch,
            )

        print(f"  z_out:            {result['z_out'].shape}")
        print(f"  prob_bin:         {result['prob_bin'].shape}")
        print(f"  logits_int_small: {result['logits_int_small'].shape}")
        print(f"  pred_int_large:   {result['pred_int_large'].shape}")

        assert result['z_out'].shape == (N_total, emb_size)
        assert not result['z_out'].isnan().any()

        # Mask consistency
        n_decoded = (result['mask_bin'].sum()
                     + result['mask_int_small'].sum()
                     + result['mask_int_large'].sum()
                     + (var_types == 0).sum())
        assert n_decoded == N_total

    print("PASS: batched 4-step pipeline works correctly.")


# ======================================================================
# Test 6: Focal loss
# ======================================================================
def test_focal_loss():
    """Test focal loss properties."""
    print("\nFocal loss test...")
    emb_size = 64
    decoder = DeslicingDecoder(emb_size=emb_size)

    N = 100
    prob = torch.rand(N, 1).clamp(0.01, 0.99)
    target = torch.randint(0, 2, (N,)).float()

    focal_0 = decoder.binary_focal_loss(prob, target, gamma=0.0,
                                        label_smoothing=0.0)
    bce_ref = F.binary_cross_entropy(prob.squeeze(), target)
    print(f"  Focal (gamma=0): {focal_0.item():.6f}")
    print(f"  BCE:             {bce_ref.item():.6f}")
    assert abs(focal_0.item() - bce_ref.item()) < 0.01

    focal_2 = decoder.binary_focal_loss(prob, target, gamma=2.0,
                                        label_smoothing=0.0)
    print(f"  Focal (gamma=2): {focal_2.item():.6f}")
    assert focal_2 < focal_0

    empty_loss = decoder.binary_focal_loss(
        torch.empty(0, 1), torch.empty(0), gamma=2.0,
    )
    assert empty_loss.item() == 0.0

    print("PASS: focal loss behaves correctly.")


# ======================================================================
# Test 7: Combined loss with real data
# ======================================================================
def test_combined_loss():
    """Test combined_loss with actual sample data."""
    data = load_first_sample()
    if data is None:
        print("No samples. Skipping.")
        return

    print("\nCombined loss test...")
    cons_feats, edge_idx, edge_vals, var_feats, sol_values = data

    graph_init, slicer, evolver, decoder = build_full_pipeline(
        cons_feats.shape[1], var_feats.shape[1],
    )
    for m in [graph_init, slicer, evolver, decoder]:
        m.train()

    z_var_0, _, _, result, var_types, _ = run_full_pipeline(
        graph_init, slicer, evolver, decoder,
        cons_feats, edge_idx, edge_vals, var_feats,
    )

    loss_dict = decoder.combined_loss(result, sol_values, var_types)

    print(f"  total:     {loss_dict['total'].item():.6f}")
    print(f"  binary:    {loss_dict['binary'].item():.6f}")
    print(f"  int_small: {loss_dict['int_small'].item():.6f}")
    print(f"  int_large: {loss_dict['int_large'].item():.6f}")
    print(f"  rounding:  {loss_dict['rounding'].item():.6f}")

    assert loss_dict['total'].isfinite()
    assert loss_dict['binary'].isfinite()

    # For setcover (all binary), integer losses should be 0
    assert loss_dict['int_small'].item() == 0.0
    assert loss_dict['int_large'].item() == 0.0
    assert loss_dict['rounding'].item() == 0.0

    loss_dict['total'].backward()
    print("PASS: combined loss computed and backpropagated correctly.")


# ======================================================================
# Test 8: predict_full reconstruction
# ======================================================================
def test_predict_full():
    """Test full prediction reconstruction."""
    data = load_first_sample()
    if data is None:
        print("No samples. Skipping.")
        return

    print("\npredict_full test...")
    cons_feats, edge_idx, edge_vals, var_feats, sol_values = data
    N = var_feats.shape[0]

    graph_init, slicer, evolver, decoder = build_full_pipeline(
        cons_feats.shape[1], var_feats.shape[1],
    )
    for m in [graph_init, slicer, evolver, decoder]:
        m.eval()

    with torch.no_grad():
        _, _, _, result, var_types, _ = run_full_pipeline(
            graph_init, slicer, evolver, decoder,
            cons_feats, edge_idx, edge_vals, var_feats,
        )
        predictions = decoder.predict_full(result, N)

    print(f"  predictions shape: {predictions.shape}  (expected [{N}])")
    assert predictions.shape == (N,)

    bin_preds = predictions[var_types == 1]
    if bin_preds.shape[0] > 0:
        assert (bin_preds >= 0).all() and (bin_preds <= 1).all()
        print(f"  binary predictions: mean={bin_preds.mean():.4f}")

    con_preds = predictions[var_types == 0]
    if con_preds.shape[0] > 0:
        assert con_preds.isnan().all()
        print(f"  continuous predictions: all NaN")

    if bin_preds.shape[0] > 0:
        rounded = (bin_preds > 0.5).float()
        gt = sol_values[var_types == 1]
        acc = (rounded == gt).float().mean().item()
        print(f"  binary accuracy (random model): {acc:.4f}")

    print("PASS: predict_full works correctly.")


# ======================================================================
# Test 9: Mixed variable types with dynamic routing (synthetic)
# ======================================================================
def test_mixed_types_routing():
    """Test with synthetic mixed variable types including routing."""
    print("\nMixed types with dynamic routing test (synthetic)...")
    N = 200
    emb_size = 64
    K = 16
    F_dim = 23
    threshold = 10

    # Create synthetic variable features with bounds
    var_feats = torch.zeros(N, F_dim)

    # 80 binary vars (indices 0-79)
    var_feats[:80, 1] = 1.0
    var_feats[:80, 21] = 0.0   # lb
    var_feats[:80, 22] = 1.0   # ub

    # 40 small-range integer vars (indices 80-119): range [2, 8]
    var_feats[80:120, 2] = 1.0
    var_feats[80:120, 21] = 2.0
    var_feats[80:120, 22] = 8.0  # range=6 <= 10

    # 30 large-range integer vars (indices 120-149): range [0, 500]
    var_feats[120:150, 2] = 1.0
    var_feats[120:150, 21] = 0.0
    var_feats[120:150, 22] = 500.0  # range=500 > 10

    # 20 unbounded integer vars (indices 150-169)
    var_feats[150:170, 3] = 1.0  # implicit integer
    var_feats[150:170, 21] = 0.0
    var_feats[150:170, 22] = 1e20  # unbounded

    # 30 continuous vars (indices 170-199)
    var_feats[170:, 4] = 1.0

    var_types = extract_var_types(var_feats)
    assert (var_types == 1).sum() == 80
    assert (var_types == 2).sum() == 90   # 40 + 30 + 20
    assert (var_types == 0).sum() == 30

    # Build pipeline components
    z_var_0 = torch.randn(N, emb_size)
    attn_weights = F.softmax(torch.randn(N, K), dim=-1)
    evolved_tokens = torch.randn(K, emb_size)
    token_batch = torch.zeros(K, dtype=torch.long)

    decoder = DeslicingDecoder(
        emb_size=emb_size, int_range_threshold=threshold, dropout=0.0)

    # ---- Forward pass ----
    decoder.eval()
    with torch.no_grad():
        result = decoder(evolved_tokens, token_batch, attn_weights,
                         var_types, z_var_0=z_var_0,
                         variable_features=var_feats)

    n_bin = result['mask_bin'].sum().item()
    n_is = result['mask_int_small'].sum().item()
    n_il = result['mask_int_large'].sum().item()
    n_int = result['mask_int'].sum().item()

    print(f"  binary:          {n_bin}  (expected 80)")
    print(f"  int small-range: {n_is}  (expected 40)")
    print(f"  int large-range: {n_il}  (expected 50)")
    print(f"  int total:       {n_int}  (expected 90)")
    print(f"  continuous:      {(var_types == 0).sum().item()}  (expected 30)")

    assert n_bin == 80
    assert n_is == 40
    assert n_il == 50   # 30 large + 20 unbounded
    assert n_int == 90

    # Shape checks
    assert result['prob_bin'].shape == (80, 1)
    assert result['logits_int_small'].shape == (40, threshold + 1)
    assert result['pred_int_large'].shape == (50, 1)

    # Check dynamic masking for small-range logits
    # range = 8 - 2 + 1 = 7, so classes 0-6 valid, 7-10 should be -inf
    logits_s = result['logits_int_small']
    assert (logits_s[:, 7:] < -1e8).all(), \
        "Invalid classes should be masked to -inf"
    print(f"  logit masking: classes 7-10 correctly masked to -inf")

    # Check offsets
    assert (result['int_small_offsets'] == 2.0).all(), \
        f"Expected offset=2.0, got {result['int_small_offsets'][:5]}"
    assert (result['int_small_ranges'] == 7).all(), \
        f"Expected range=7, got {result['int_small_ranges'][:5]}"

    # ---- Loss computation ----
    decoder.train()
    result_train = decoder(evolved_tokens, token_batch, attn_weights,
                           var_types, z_var_0=z_var_0,
                           variable_features=var_feats)

    sol_values = torch.zeros(N)
    sol_values[:80] = torch.randint(0, 2, (80,)).float()       # binary
    sol_values[80:120] = torch.randint(2, 9, (40,)).float()    # small int [2,8]
    sol_values[120:150] = torch.randint(0, 500, (30,)).float() # large int
    sol_values[150:170] = torch.randint(0, 1000, (20,)).float()

    loss_dict = decoder.combined_loss(result_train, sol_values, var_types)

    print(f"\n  Loss breakdown:")
    print(f"    total:     {loss_dict['total'].item():.4f}")
    print(f"    binary:    {loss_dict['binary'].item():.4f}")
    print(f"    int_small: {loss_dict['int_small'].item():.4f}")
    print(f"    int_large: {loss_dict['int_large'].item():.4f}")
    print(f"    rounding:  {loss_dict['rounding'].item():.4f}")

    assert loss_dict['binary'].item() > 0
    assert loss_dict['int_small'].item() > 0
    assert loss_dict['int_large'].item() > 0
    assert loss_dict['rounding'].item() >= 0

    loss_dict['total'].backward()

    # Check gradients flow to all decoder heads
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in decoder.bin_decoder.parameters()), \
        "No grad in bin_decoder"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in decoder.int_small_decoder.parameters()), \
        "No grad in int_small_decoder"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in decoder.int_large_decoder.parameters()), \
        "No grad in int_large_decoder"

    print("  Gradients flow to all three decoder heads")

    # ---- predict_full ----
    decoder.eval()
    with torch.no_grad():
        result_eval = decoder(evolved_tokens, token_batch, attn_weights,
                              var_types, z_var_0=z_var_0,
                              variable_features=var_feats)
        predictions = decoder.predict_full(result_eval, N)

    assert predictions.shape == (N,)
    # Binary: probabilities in [0, 1]
    assert (predictions[:80] >= 0).all() and (predictions[:80] <= 1).all()
    # Small-range int: should be offset-corrected, in [2, 8]
    small_preds = predictions[80:120]
    assert (small_preds >= 2.0).all() and (small_preds <= 8.0).all(), \
        f"Small-range preds should be in [2, 8], got [{small_preds.min()}, {small_preds.max()}]"
    # Large-range int: should be rounded integers
    large_preds = predictions[120:170]
    assert (large_preds == large_preds.round()).all(), \
        "Large-range preds should be rounded"
    # Continuous: NaN
    assert predictions[170:].isnan().all()

    print(f"  predict_full: all types reconstructed correctly")
    print("PASS: mixed types with dynamic routing handled correctly.")


# ======================================================================
# Test 10: Fallback (no variable_features)
# ======================================================================
def test_fallback_no_features():
    """Test backward-compatible behavior when variable_features=None."""
    print("\nFallback test (no variable_features)...")
    N = 50
    emb_size = 64
    K = 8

    z_var_0 = torch.randn(N, emb_size)
    attn_weights = F.softmax(torch.randn(N, K), dim=-1)
    evolved_tokens = torch.randn(K, emb_size)
    token_batch = torch.zeros(K, dtype=torch.long)

    var_types = torch.zeros(N, dtype=torch.long)
    var_types[:20] = 1   # binary
    var_types[20:40] = 2  # integer
    var_types[40:] = 0   # continuous

    decoder = DeslicingDecoder(emb_size=emb_size, int_range_threshold=10)
    decoder.eval()

    with torch.no_grad():
        result = decoder(evolved_tokens, token_batch, attn_weights,
                         var_types, z_var_0=z_var_0,
                         variable_features=None)  # no features

    # All integers should be small-range (fallback)
    assert result['mask_int_small'].sum() == 20
    assert result['mask_int_large'].sum() == 0
    assert result['logits_int_small'].shape == (20, 11)
    assert result['pred_int_large'].shape == (0, 1)

    print(f"  All 20 integers -> small-range (fallback)")
    print("PASS: fallback without variable_features works.")


if __name__ == '__main__':
    print("=" * 60)
    print("DeslicingDecoder Module Tests (Dynamic Routing)")
    print("=" * 60)

    test_var_types_extraction()
    test_int_routing()
    test_single_sample()
    test_gradient_flow()
    test_batched_forward()
    test_focal_loss()
    test_combined_loss()
    test_predict_full()
    test_mixed_types_routing()
    test_fallback_no_features()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
