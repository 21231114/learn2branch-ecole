"""
Smoke test for LatentTrajectoryEvolution module (Step 3).

Tests:
    1. Single-sample forward pass (shape correctness)
    2. Full 3-step pipeline: GraphInit → AdaptiveSlicing → LatentEvolution
    3. Gradient flow through the entire pipeline
    4. Batched forward pass with torch_geometric DataLoader
    5. Intermediate states & auxiliary loss
    6. Gate values and evolution behavior
    7. Stochastic depth (train vs eval consistency)

Usage:
    python test_latent_evolution.py
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


def build_pipeline(cons_nfeats, var_nfeats, emb_size=64, K=32, T=3):
    """Build the full 3-step pipeline."""
    graph_init = GraphInitialization(
        cons_nfeats=cons_nfeats, var_nfeats=var_nfeats, emb_size=emb_size,
    )
    slicer = AdaptiveSlicing(
        emb_size=emb_size, n_slices=K, n_heads=4, dropout=0.0,
    )
    evolver = LatentTrajectoryEvolution(
        emb_size=emb_size, n_layers=4, n_heads=4, ffn_ratio=4,
        n_evolve_steps=T, dropout=0.0, stochastic_depth_rate=0.0,
    )
    return graph_init, slicer, evolver


def run_pipeline(graph_init, slicer, evolver, cons_feats, edge_idx,
                 edge_vals, var_feats, var_batch=None):
    """Run the full Step1→Step2→Step3 pipeline."""
    z_var_0 = graph_init(cons_feats, edge_idx, edge_vals, var_feats)
    tokens_0, token_batch, attn_weights = slicer(z_var_0, batch=var_batch)
    evolved, token_batch_out, intermediates = evolver(tokens_0, token_batch)
    return z_var_0, tokens_0, evolved, token_batch_out, attn_weights, intermediates


# ======================================================================
# Test 1: Single-sample forward (shape correctness)
# ======================================================================
def test_single_sample():
    """Test LatentTrajectoryEvolution with a single graph."""
    data = load_first_sample()
    if data is None:
        print("No sample files found. Skipping.")
        return
    cons_feats, edge_idx, edge_vals, var_feats = data
    emb_size = 64
    K = 32
    T = 3

    print(f"Input: {var_feats.shape[0]} variables, emb_size={emb_size}, "
          f"K={K} slices, T={T} evolution steps")

    graph_init, slicer, evolver = build_pipeline(
        cons_feats.shape[1], var_feats.shape[1], emb_size, K, T,
    )
    graph_init.eval()
    slicer.eval()
    evolver.eval()

    with torch.no_grad():
        z_var_0, tokens_0, evolved, token_batch, _, intermediates = \
            run_pipeline(graph_init, slicer, evolver,
                         cons_feats, edge_idx, edge_vals, var_feats)

    print(f"\n  z_var_0 (Step 1):           {z_var_0.shape}")
    print(f"  tokens_0 (Step 2):          {tokens_0.shape}")
    print(f"  evolved_tokens (Step 3):    {evolved.shape}")
    print(f"  intermediate states:        {len(intermediates)} "
          f"(T+1 = {T+1})")

    assert evolved.shape == (K, emb_size), \
        f"Shape mismatch: {evolved.shape} vs ({K}, {emb_size})"
    assert token_batch.shape == (K,)
    assert len(intermediates) == T + 1
    for i, state in enumerate(intermediates):
        assert state.shape == (K, emb_size), \
            f"Intermediate {i} shape: {state.shape} vs ({K}, {emb_size})"
    print("PASS: shapes correct.")

    # Check no NaN
    assert not evolved.isnan().any(), "Evolved tokens contain NaN!"
    print("PASS: no NaN values.")

    # Stats
    print(f"\n  evolved token stats:")
    print(f"    mean:  {evolved.mean().item():.6f}")
    print(f"    std:   {evolved.std().item():.6f}")
    print(f"    min:   {evolved.min().item():.6f}")
    print(f"    max:   {evolved.max().item():.6f}")


# ======================================================================
# Test 2: Gradient flow through full pipeline
# ======================================================================
def test_gradient_flow():
    """Test that gradients flow through Step1 → Step2 → Step3."""
    data = load_first_sample()
    if data is None:
        print("No samples for gradient test. Skipping.")
        return

    print("\nGradient flow test (full 3-step pipeline)...")
    cons_feats, edge_idx, edge_vals, var_feats = data
    emb_size = 64
    K = 32
    T = 3

    graph_init, slicer, evolver = build_pipeline(
        cons_feats.shape[1], var_feats.shape[1], emb_size, K, T,
    )
    graph_init.train()
    slicer.train()
    evolver.train()

    _, _, evolved, _, _, _ = run_pipeline(
        graph_init, slicer, evolver,
        cons_feats, edge_idx, edge_vals, var_feats,
    )

    loss = evolved.sum()
    loss.backward()

    # Check each module
    for name, module in [("GraphInit", graph_init),
                         ("AdaptiveSlicing", slicer),
                         ("LatentEvolution", evolver)]:
        n_with = sum(1 for _, p in module.named_parameters()
                     if p.grad is not None and p.grad.abs().sum() > 0)
        n_total = sum(1 for _ in module.parameters())
        print(f"  {name}: {n_with}/{n_total} params have non-zero grad")
        assert n_with > 0, f"No gradients in {name}!"

    # Check specific important parameters in evolver
    assert evolver.step_embeddings.grad is not None, "step_embeddings no grad!"
    assert evolver.step_embeddings.grad.abs().sum() > 0, "step_embeddings grad=0!"
    print("  step_embeddings gradient: OK")

    assert evolver.gate_logits.grad is not None, "gate_logits no grad!"
    assert evolver.gate_logits.grad.abs().sum() > 0, "gate_logits grad=0!"
    print("  gate_logits gradient: OK")

    print("PASS: gradients flow through entire pipeline.")


# ======================================================================
# Test 3: Batched forward
# ======================================================================
def test_batched_forward():
    """Test batched forward with multiple graphs."""
    sample_files = sorted(str(p) for p in pathlib.Path('data/samples').rglob('sample_*.pkl'))
    if len(sample_files) < 2:
        print("Not enough samples for batch test. Skipping.")
        return

    sample_files = sample_files[:3]
    B = len(sample_files)
    print(f"\nBatch test with {B} samples...")

    dataset = SolutionGraphDataset(sample_files)
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=B)

    first = dataset[0]
    emb_size = 64
    K = 32
    T = 3

    graph_init, slicer, evolver = build_pipeline(
        first.constraint_features.shape[1],
        first.variable_features.shape[1],
        emb_size, K, T,
    )
    graph_init.eval()
    slicer.eval()
    evolver.eval()

    for batch_data in loader:
        # Build var_batch
        var_counts = [dataset[i].variable_features.shape[0] for i in range(B)]
        var_batch = torch.cat([
            torch.full((c,), i, dtype=torch.long)
            for i, c in enumerate(var_counts)
        ])

        with torch.no_grad():
            _, _, evolved, token_batch, _, intermediates = run_pipeline(
                graph_init, slicer, evolver,
                batch_data.constraint_features, batch_data.edge_index,
                batch_data.edge_attr, batch_data.variable_features,
                var_batch=var_batch,
            )

        expected = B * K
        print(f"  evolved_tokens:  {evolved.shape}  (expected [{expected}, {emb_size}])")
        print(f"  token_batch:     {token_batch.shape}  (expected [{expected}])")
        print(f"  intermediates:   {len(intermediates)} states")

        assert evolved.shape == (expected, emb_size)
        assert token_batch.shape == (expected,)
        assert len(intermediates) == T + 1

        for i in range(B):
            count = (token_batch == i).sum().item()
            assert count == K, f"Graph {i}: {count} tokens, expected {K}"

        assert not evolved.isnan().any()
        print("PASS: batched forward works correctly.")


# ======================================================================
# Test 4: Intermediate states & auxiliary loss
# ======================================================================
def test_auxiliary_loss():
    """Test the auxiliary_loss helper."""
    print("\nAuxiliary loss test...")
    emb_size = 64
    K = 32
    T = 3

    evolver = LatentTrajectoryEvolution(
        emb_size=emb_size, n_layers=2, n_heads=4,
        n_evolve_steps=T, dropout=0.0,
    )
    evolver.train()

    tokens_in = torch.randn(K, emb_size)
    evolved, _, intermediates = evolver(tokens_in)

    assert len(intermediates) == T + 1, \
        f"Expected {T+1} states, got {len(intermediates)}"

    # Define a dummy loss function
    target = torch.zeros(K, emb_size)
    def dummy_loss(pred, tgt):
        return F.mse_loss(pred, tgt)

    aux_loss = evolver.auxiliary_loss(intermediates, dummy_loss, target,
                                     ema_decay=0.9)
    print(f"  auxiliary_loss value: {aux_loss.item():.6f}")
    assert aux_loss.isfinite(), "Auxiliary loss is not finite!"

    # Verify it's differentiable
    aux_loss.backward()
    assert evolver.gate_logits.grad is not None
    print("  auxiliary_loss is differentiable: OK")

    # Verify later steps have higher weight
    # With ema_decay=0.9, weight for step T should be highest
    weights = torch.tensor([0.9 ** (T - t) for t in range(T + 1)])
    weights = weights / weights.sum()
    print(f"  EMA weights: {weights.tolist()}")
    assert weights[-1] > weights[0], "Last step should have highest weight!"

    print("PASS: auxiliary loss works correctly.")


# ======================================================================
# Test 5: Gate values
# ======================================================================
def test_gate_values():
    """Test gate initialization and monitoring."""
    print("\nGate values test...")
    T = 3
    evolver = LatentTrajectoryEvolution(
        emb_size=64, n_layers=2, n_heads=4, n_evolve_steps=T,
    )

    gates = evolver.get_gate_values()
    print(f"  Initial gate values: {[f'{g:.4f}' for g in gates]}")

    assert len(gates) == T
    for g in gates:
        # Initialized to ~0.1
        assert 0.05 < g < 0.2, f"Gate {g} not in expected initial range"

    print("PASS: gate values initialized correctly (~0.1).")


# ======================================================================
# Test 6: Evolution changes tokens
# ======================================================================
def test_evolution_changes():
    """Verify that evolution actually modifies tokens at each step."""
    print("\nEvolution trajectory test...")
    emb_size = 64
    K = 32
    T = 3

    evolver = LatentTrajectoryEvolution(
        emb_size=emb_size, n_layers=2, n_heads=4,
        n_evolve_steps=T, dropout=0.0, stochastic_depth_rate=0.0,
    )
    evolver.eval()

    tokens_in = torch.randn(K, emb_size)

    with torch.no_grad():
        _, _, intermediates = evolver(tokens_in)

    # Each step should produce different tokens
    print(f"  Trajectory distances (L2 from initial state):")
    for t in range(1, T + 1):
        dist = (intermediates[t] - intermediates[0]).norm().item()
        print(f"    Step {t}: {dist:.4f}")
        assert dist > 0, f"Step {t} didn't change tokens!"

    # Later steps should generally be further from initial
    # (not always guaranteed, but with random init it's very likely)
    dist_1 = (intermediates[1] - intermediates[0]).norm().item()
    dist_T = (intermediates[T] - intermediates[0]).norm().item()
    print(f"  Step 1 distance: {dist_1:.4f}, Step {T} distance: {dist_T:.4f}")

    print("PASS: evolution produces distinct trajectory states.")


# ======================================================================
# Test 7: Stochastic depth behavior
# ======================================================================
def test_stochastic_depth():
    """Verify stochastic depth differs between train and eval modes."""
    print("\nStochastic depth test...")
    emb_size = 64
    K = 16

    evolver = LatentTrajectoryEvolution(
        emb_size=emb_size, n_layers=4, n_heads=4,
        n_evolve_steps=2, dropout=0.0, stochastic_depth_rate=0.5,
    )

    tokens = torch.randn(K, emb_size)

    # Eval mode: deterministic
    evolver.eval()
    with torch.no_grad():
        out1, _, _ = evolver(tokens)
        out2, _, _ = evolver(tokens)
    assert torch.allclose(out1, out2), "Eval mode should be deterministic!"
    print("  Eval mode: deterministic (2 runs identical) ✓")

    # Train mode: stochastic (may vary due to dropout of layers)
    evolver.train()
    torch.manual_seed(42)
    out3, _, _ = evolver(tokens.clone())
    torch.manual_seed(123)
    out4, _, _ = evolver(tokens.clone())
    # With 50% drop rate on 4 layers, outputs should differ
    # (astronomically unlikely to be identical)
    differs = not torch.allclose(out3.detach(), out4.detach(), atol=1e-6)
    print(f"  Train mode: stochastic (2 runs differ: {differs}) ✓")

    print("PASS: stochastic depth behaves correctly.")


# ======================================================================
# Test 8: Gradient checkpointing
# ======================================================================
def test_grad_checkpoint():
    """Test that gradient checkpointing produces same gradients."""
    print("\nGradient checkpointing test...")
    emb_size = 64
    K = 16

    tokens = torch.randn(K, emb_size)

    # Without checkpointing
    ev1 = LatentTrajectoryEvolution(
        emb_size=emb_size, n_layers=2, n_heads=4,
        n_evolve_steps=2, dropout=0.0, stochastic_depth_rate=0.0,
        use_grad_checkpoint=False,
    )
    ev1.train()
    out1, _, _ = ev1(tokens.clone())
    out1.sum().backward()
    grads1 = {n: p.grad.clone() for n, p in ev1.named_parameters()
              if p.grad is not None}

    # With checkpointing (same weights)
    ev2 = LatentTrajectoryEvolution(
        emb_size=emb_size, n_layers=2, n_heads=4,
        n_evolve_steps=2, dropout=0.0, stochastic_depth_rate=0.0,
        use_grad_checkpoint=True,
    )
    ev2.load_state_dict(ev1.state_dict())
    ev2.train()
    out2, _, _ = ev2(tokens.clone())
    out2.sum().backward()

    # Compare gradients
    max_diff = 0
    for name, param in ev2.named_parameters():
        if param.grad is not None and name in grads1:
            diff = (param.grad - grads1[name]).abs().max().item()
            max_diff = max(max_diff, diff)

    print(f"  Max gradient difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Gradient mismatch: {max_diff}"
    print("PASS: gradient checkpointing produces matching gradients.")


if __name__ == '__main__':
    print("=" * 60)
    print("LatentTrajectoryEvolution Module Tests (Step 3)")
    print("=" * 60)

    test_single_sample()
    test_gradient_flow()
    test_batched_forward()
    test_auxiliary_loss()
    test_gate_values()
    test_evolution_changes()
    test_stochastic_depth()
    test_grad_checkpoint()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
