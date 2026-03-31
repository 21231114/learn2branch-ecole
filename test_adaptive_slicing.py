"""
Smoke test for AdaptiveSlicing module (Step 2).

Tests:
    1. Single-sample forward pass (shape correctness)
    2. Gradient flow through the module
    3. Integration: GraphInitialization → AdaptiveSlicing pipeline
    4. Batched forward pass with torch_geometric DataLoader
    5. Regularization losses (entropy, diversity)

Usage:
    python test_adaptive_slicing.py
"""
import sys
import os
import pathlib
import gzip
import pickle
import numpy as np

import torch
import torch_geometric

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.graph_init import GraphInitialization
from model.adaptive_slicing import AdaptiveSlicing
from utilities import SolutionGraphDataset


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
    return cons_feats, edge_idx, edge_vals, var_feats


# ======================================================================
# Test 1: Single-sample forward
# ======================================================================
def test_single_sample():
    """Test AdaptiveSlicing with a single graph (no batch)."""
    data = load_first_sample()
    if data is None:
        print("No sample files found. Skipping.")
        return
    cons_feats, edge_idx, edge_vals, var_feats = data
    N_var = var_feats.shape[0]
    emb_size = 64
    K = 32  # number of slices

    print(f"Input variable features: {var_feats.shape}")

    # Step 1: GraphInitialization
    graph_init = GraphInitialization(
        cons_nfeats=cons_feats.shape[1],
        var_nfeats=var_feats.shape[1],
        emb_size=emb_size,
    )
    graph_init.eval()
    with torch.no_grad():
        z_var_0 = graph_init(cons_feats, edge_idx, edge_vals, var_feats)

    print(f"z_var_0 shape: {z_var_0.shape}")
    assert z_var_0.shape == (N_var, emb_size)

    # Step 2: AdaptiveSlicing
    slicer = AdaptiveSlicing(
        emb_size=emb_size,
        n_slices=K,
        n_heads=4,
        dropout=0.0,  # no dropout for deterministic test
    )
    slicer.eval()
    with torch.no_grad():
        tokens, token_batch, attn_weights = slicer(z_var_0)

    print(f"\nAdaptiveSlicing outputs:")
    print(f"  latent_tokens shape:  {tokens.shape}  (expected [{K}, {emb_size}])")
    print(f"  token_batch shape:    {token_batch.shape}  (expected [{K}])")
    print(f"  attn_weights shape:   {attn_weights.shape}  (expected [{N_var}, {K}])")

    assert tokens.shape == (K, emb_size), \
        f"tokens shape mismatch: {tokens.shape} vs ({K}, {emb_size})"
    assert token_batch.shape == (K,), \
        f"token_batch shape mismatch: {token_batch.shape} vs ({K},)"
    assert attn_weights.shape == (N_var, K), \
        f"attn_weights shape mismatch: {attn_weights.shape} vs ({N_var}, {K})"
    print("PASS: shapes correct.")

    # Check attention weights are valid probabilities
    attn_sum = attn_weights.sum(dim=-1)
    print(f"\n  attn_weights row-sum (should be ~1.0):")
    print(f"    mean: {attn_sum.mean().item():.6f}")
    print(f"    std:  {attn_sum.std().item():.6f}")
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-4), \
        "Attention weights do not sum to 1!"
    print("PASS: attention weights are valid probabilities.")

    # Check no NaN
    assert not tokens.isnan().any(), "Tokens contain NaN!"
    assert not attn_weights.isnan().any(), "Attention weights contain NaN!"
    print("PASS: no NaN values.")

    # Token stats
    print(f"\n  token stats:")
    print(f"    mean:  {tokens.mean().item():.6f}")
    print(f"    std:   {tokens.std().item():.6f}")
    print(f"    min:   {tokens.min().item():.6f}")
    print(f"    max:   {tokens.max().item():.6f}")


# ======================================================================
# Test 2: Gradient flow
# ======================================================================
def test_gradient_flow():
    """Test that gradients flow through GraphInit → AdaptiveSlicing."""
    data = load_first_sample()
    if data is None:
        print("No samples for gradient test. Skipping.")
        return

    print("\nGradient flow test...")
    cons_feats, edge_idx, edge_vals, var_feats = data
    emb_size = 64
    K = 32

    graph_init = GraphInitialization(
        cons_nfeats=cons_feats.shape[1],
        var_nfeats=var_feats.shape[1],
        emb_size=emb_size,
    )
    slicer = AdaptiveSlicing(emb_size=emb_size, n_slices=K, n_heads=4)

    # Forward
    graph_init.train()
    slicer.train()
    z_var_0 = graph_init(cons_feats, edge_idx, edge_vals, var_feats)
    tokens, token_batch, attn_weights = slicer(z_var_0)

    # Backward
    loss = tokens.sum()
    loss.backward()

    # Check AdaptiveSlicing gradients
    n_with_grad = 0
    n_total = 0
    for name, param in slicer.named_parameters():
        n_total += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            n_with_grad += 1
    print(f"  AdaptiveSlicing: {n_with_grad}/{n_total} params have non-zero grad")
    assert n_with_grad > 0, "No gradients in AdaptiveSlicing!"

    # Check GraphInitialization gradients (backprop through slicer)
    n_with_grad_gi = 0
    n_total_gi = 0
    for name, param in graph_init.named_parameters():
        n_total_gi += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            n_with_grad_gi += 1
    print(f"  GraphInitialization: {n_with_grad_gi}/{n_total_gi} params have non-zero grad")
    assert n_with_grad_gi > 0, "Gradients don't flow back to GraphInitialization!"

    # Check slice_centers gradient
    assert slicer.slice_centers.grad is not None, "slice_centers has no gradient!"
    assert slicer.slice_centers.grad.abs().sum() > 0, "slice_centers gradient is zero!"
    print("  slice_centers gradient: OK")

    # Check temperature gradient
    assert slicer.log_temperature.grad is not None, "log_temperature has no gradient!"
    print("  log_temperature gradient: OK")

    print("PASS: gradients flow correctly through full pipeline.")


# ======================================================================
# Test 3: Batched forward (torch_geometric DataLoader)
# ======================================================================
def test_batched_forward():
    """Test batched forward with multiple graphs."""
    sample_files = sorted(str(p) for p in pathlib.Path('data/samples').rglob('sample_*.pkl'))
    if len(sample_files) < 2:
        print("Not enough samples for batch test. Skipping.")
        return

    sample_files = sample_files[:3]
    print(f"\nBatch test with {len(sample_files)} samples...")

    dataset = SolutionGraphDataset(sample_files)
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=len(sample_files))

    first = dataset[0]
    emb_size = 64
    K = 32

    graph_init = GraphInitialization(
        cons_nfeats=first.constraint_features.shape[1],
        var_nfeats=first.variable_features.shape[1],
        emb_size=emb_size,
    )
    slicer = AdaptiveSlicing(emb_size=emb_size, n_slices=K, n_heads=4, dropout=0.0)

    graph_init.eval()
    slicer.eval()

    for batch_data in loader:
        with torch.no_grad():
            z_var_0 = graph_init(
                batch_data.constraint_features,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.variable_features,
            )

            # Get batch indices for variable nodes
            # In torch_geometric, batch.batch gives node-to-graph mapping
            # but we need to extract only the variable-node portion
            # The batch object has all nodes (cons + var) concatenated
            # variable nodes come after constraint nodes per graph
            n_total_nodes = batch_data.num_nodes
            n_var_total = batch_data.variable_features.shape[0]
            n_con_total = batch_data.constraint_features.shape[0]

            # Build variable batch index from the data
            # Each graph i contributes n_var_i variable nodes
            var_batch = batch_data.batch[n_con_total:]
            # But torch_geometric may order differently, let's be safe
            # Actually in BipartiteNodeData, num_nodes = n_con + n_var
            # and batch indices follow that order
            # Let's construct it manually from the data
            B = len(sample_files)
            var_counts = []
            for i in range(B):
                var_counts.append(dataset[i].variable_features.shape[0])

            var_batch = torch.cat([
                torch.full((c,), i, dtype=torch.long)
                for i, c in enumerate(var_counts)
            ])

            tokens, token_batch, attn_weights = slicer(z_var_0, batch=var_batch)

        expected_tokens = B * K
        print(f"  latent_tokens:  {tokens.shape}  (expected [{expected_tokens}, {emb_size}])")
        print(f"  token_batch:    {token_batch.shape}  (expected [{expected_tokens}])")
        print(f"  attn_weights:   {attn_weights.shape}  (expected [{n_var_total}, {K}])")

        assert tokens.shape == (expected_tokens, emb_size), \
            f"tokens shape: {tokens.shape} vs ({expected_tokens}, {emb_size})"
        assert token_batch.shape == (expected_tokens,)
        assert attn_weights.shape == (n_var_total, K)

        # Verify token_batch has correct values
        for i in range(B):
            count = (token_batch == i).sum().item()
            assert count == K, f"Graph {i} has {count} tokens, expected {K}"

        assert not tokens.isnan().any()
        print("PASS: batched forward works correctly.")


# ======================================================================
# Test 4: Regularization losses
# ======================================================================
def test_regularization_losses():
    """Test that entropy and diversity losses are computed correctly."""
    data = load_first_sample()
    if data is None:
        print("No samples for regularization test. Skipping.")
        return

    print("\nRegularization loss test...")
    cons_feats, edge_idx, edge_vals, var_feats = data
    emb_size = 64
    K = 32

    graph_init = GraphInitialization(
        cons_nfeats=cons_feats.shape[1],
        var_nfeats=var_feats.shape[1],
        emb_size=emb_size,
    )
    slicer = AdaptiveSlicing(emb_size=emb_size, n_slices=K, n_heads=4)

    graph_init.eval()
    slicer.train()

    with torch.no_grad():
        z_var_0 = graph_init(cons_feats, edge_idx, edge_vals, var_feats)

    tokens, token_batch, attn_weights = slicer(z_var_0)

    # Entropy loss
    ent_loss = slicer.entropy_loss(attn_weights)
    print(f"  entropy_loss: {ent_loss.item():.6f} (negative = good, encourage uniformity)")
    assert ent_loss.isfinite(), "Entropy loss is not finite!"

    # Diversity loss
    div_loss = slicer.diversity_loss()
    print(f"  diversity_loss: {div_loss.item():.6f} (lower = more diverse centers)")
    assert div_loss.isfinite(), "Diversity loss is not finite!"
    assert -1.0 <= div_loss.item() <= 1.0, \
        f"Diversity loss out of expected range: {div_loss.item()}"

    # Check that both losses have gradients
    total_loss = tokens.sum() + 0.01 * ent_loss + 0.01 * div_loss
    total_loss.backward()

    assert slicer.slice_centers.grad is not None, "No grad on slice_centers after reg losses"
    print("PASS: regularization losses work correctly.")


# ======================================================================
# Test 5: Temperature behavior
# ======================================================================
def test_temperature():
    """Verify that temperature affects attention sharpness."""
    print("\nTemperature test...")
    emb_size = 64
    K = 16
    N = 100

    z = torch.randn(N, emb_size)

    slicer = AdaptiveSlicing(emb_size=emb_size, n_slices=K, n_heads=4, dropout=0.0)
    slicer.eval()

    # Low temperature → sharper attention
    with torch.no_grad():
        slicer.log_temperature.fill_(-2.0)  # τ ≈ 0.135
        _, _, attn_sharp = slicer(z)
        entropy_sharp = -(attn_sharp * (attn_sharp + 1e-8).log()).sum(-1).mean()

        slicer.log_temperature.fill_(2.0)   # τ ≈ 7.389
        _, _, attn_soft = slicer(z)
        entropy_soft = -(attn_soft * (attn_soft + 1e-8).log()).sum(-1).mean()

    print(f"  Sharp (τ≈0.14): entropy = {entropy_sharp.item():.4f}")
    print(f"  Soft  (τ≈7.39): entropy = {entropy_soft.item():.4f}")
    assert entropy_soft > entropy_sharp, \
        "Higher temperature should give higher entropy (softer attention)!"
    print("PASS: temperature controls attention sharpness.")


if __name__ == '__main__':
    print("=" * 60)
    print("AdaptiveSlicing Module Tests (Step 2)")
    print("=" * 60)

    test_single_sample()
    test_gradient_flow()
    test_batched_forward()
    test_regularization_losses()
    test_temperature()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
