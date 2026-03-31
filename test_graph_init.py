"""
Smoke test for GraphInitialization module.

Usage:
    python test_graph_init.py
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
from utilities import SolutionGraphDataset


def test_single_sample():
    """Test with a single sample loaded directly."""
    sample_files = sorted(str(p) for p in pathlib.Path('data/samples').rglob('sample_*.pkl'))
    if not sample_files:
        print("No sample files found. Skipping test.")
        return

    sample_path = sample_files[0]
    print(f"Loading sample: {sample_path}")

    with gzip.open(sample_path, 'rb') as f:
        sample = pickle.load(f)

    cons_feats, (edge_idx, edge_vals), var_feats = sample['observation']
    cons_feats = torch.FloatTensor(np.asarray(cons_feats, dtype=np.float32))
    edge_idx = torch.LongTensor(np.asarray(edge_idx, dtype=np.int64))
    edge_vals = torch.FloatTensor(np.asarray(edge_vals, dtype=np.float32)).unsqueeze(-1)
    var_feats = torch.FloatTensor(np.asarray(var_feats, dtype=np.float32))

    print(f"  constraint_features: {cons_feats.shape}")
    print(f"  edge_indices:        {edge_idx.shape}")
    print(f"  edge_features:       {edge_vals.shape}")
    print(f"  variable_features:   {var_feats.shape}")

    model = GraphInitialization(
        cons_nfeats=cons_feats.shape[1],
        edge_nfeats=1,
        var_nfeats=var_feats.shape[1],
        emb_size=64,
        n_conv_rounds=2,
        dropout=0.1,
    )

    # ── Pre-training (calibrate PreNormLayers) ──
    print("\nRunning PreNorm calibration...")
    model.pre_train_init()
    for _ in range(50):
        if not model.pre_train(cons_feats, edge_idx, edge_vals, var_feats):
            break
        module = model.pre_train_next()
        if module is None:
            break
    # finalize remaining
    while model.pre_train_next() is not None:
        pass
    print("  PreNorm calibration done.")

    # ── Forward pass ──
    model.eval()
    with torch.no_grad():
        z_var_0 = model(cons_feats, edge_idx, edge_vals, var_feats)

    n_var = var_feats.shape[0]
    emb_size = 64
    print(f"\nz_var_0 shape: {z_var_0.shape}")
    print(f"Expected:      [{n_var}, {emb_size}]")
    assert z_var_0.shape == (n_var, emb_size), \
        f"Shape mismatch: got {z_var_0.shape}, expected ({n_var}, {emb_size})"
    print("PASS: shape is correct.")

    print(f"\nz_var_0 stats:")
    print(f"  mean:  {z_var_0.mean().item():.6f}")
    print(f"  std:   {z_var_0.std().item():.6f}")
    print(f"  min:   {z_var_0.min().item():.6f}")
    print(f"  max:   {z_var_0.max().item():.6f}")
    print(f"  has_nan: {z_var_0.isnan().any().item()}")
    assert not z_var_0.isnan().any(), "Output contains NaN!"
    print("PASS: no NaN values.")


def test_gradient_flow():
    """Test that gradients flow through the module."""
    sample_files = sorted(str(p) for p in pathlib.Path('data/samples').rglob('sample_*.pkl'))
    if not sample_files:
        print("No samples for gradient test. Skipping.")
        return

    print("\nGradient flow test...")
    with gzip.open(sample_files[0], 'rb') as f:
        sample = pickle.load(f)

    cons_feats, (edge_idx, edge_vals), var_feats = sample['observation']
    cons_feats = torch.FloatTensor(np.asarray(cons_feats, dtype=np.float32))
    edge_idx = torch.LongTensor(np.asarray(edge_idx, dtype=np.int64))
    edge_vals = torch.FloatTensor(np.asarray(edge_vals, dtype=np.float32)).unsqueeze(-1)
    var_feats = torch.FloatTensor(np.asarray(var_feats, dtype=np.float32))

    model = GraphInitialization(
        cons_nfeats=cons_feats.shape[1],
        var_nfeats=var_feats.shape[1],
    )

    model.train()
    z_var_0 = model(cons_feats, edge_idx, edge_vals, var_feats)
    loss = z_var_0.sum()
    loss.backward()

    n_params_with_grad = 0
    n_params_total = 0
    for name, param in model.named_parameters():
        n_params_total += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            n_params_with_grad += 1

    print(f"  {n_params_with_grad}/{n_params_total} parameters have non-zero gradients")
    assert n_params_with_grad > 0, "No gradients flowing!"
    print("PASS: gradients flow correctly.")


def test_dataset_and_batch():
    """Test SolutionGraphDataset + batched forward."""
    sample_files = sorted(str(p) for p in pathlib.Path('data/samples').rglob('sample_*.pkl'))
    if len(sample_files) < 2:
        print("Not enough samples for batch test. Skipping.")
        return

    sample_files = sample_files[:3]
    print(f"\nBatch test with {len(sample_files)} samples...")

    dataset = SolutionGraphDataset(sample_files)
    loader = torch_geometric.loader.DataLoader(dataset, batch_size=len(sample_files))

    first = dataset[0]
    model = GraphInitialization(
        cons_nfeats=first.constraint_features.shape[1],
        edge_nfeats=1,
        var_nfeats=first.variable_features.shape[1],
    )
    model.eval()

    for batch in loader:
        with torch.no_grad():
            z_var_0 = model(
                batch.constraint_features, batch.edge_index,
                batch.edge_attr, batch.variable_features
            )
        total_vars = batch.variable_features.shape[0]
        print(f"  Batch z_var_0: {z_var_0.shape}  (total vars: {total_vars})")
        assert z_var_0.shape == (total_vars, 64)
        assert not z_var_0.isnan().any()

    print("PASS: batched forward works correctly.")


if __name__ == '__main__':
    print("=" * 60)
    print("GraphInitialization Module Tests")
    print("=" * 60)

    test_single_sample()
    test_gradient_flow()
    test_dataset_and_batch()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
