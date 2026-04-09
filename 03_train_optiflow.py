"""
03_train_optiflow.py — Training script for the OptiFlow 4-step pipeline.

Pipeline:
    Step 1: GraphInitialization   — bipartite GCN embeds variable features
    Step 2: AdaptiveSlicing       — soft-clusters variables into K latent tokens
    Step 3: LatentTrajectoryEvolution — evolves tokens via shared Transformer
    Step 4: DeslicingDecoder      — decodes back to per-variable predictions

Loss:
    L = w_bin  * FocalBCE(Binary)
      + w_is   * CE(Small_Int)
      + w_il   * Huber(Large_Int)
      + w_rnd  * RoundingReg(Large_Int)
      + w_cv   * ConstraintViolation
      + w_ent  * EntropyReg(Slicing)
      + w_div  * DiversityReg(Slicing)

Usage:
    python 03_train_optiflow.py setcover -g 0
    python 03_train_optiflow.py setcover -g -1   # CPU
    python 03_train_optiflow.py mknapsack --epochs 200 --batch-size 16
    python 03_train_optiflow.py SC --data-dir /path/to/samples -g 0  # auto-split 240/60
"""

import os
import sys
import argparse
import pathlib
import time
import json
import numpy as np

# Pre-parse GPU flag so CUDA_VISIBLE_DEVICES is set before torch initializes CUDA
_gpu = next((int(sys.argv[i + 2]) for i, a in enumerate(sys.argv[1:])
             if a in ('-g', '--gpu') and i + 2 < len(sys.argv)), 0)
os.environ['CUDA_VISIBLE_DEVICES'] = '' if _gpu == -1 else str(_gpu)

import torch
import torch.nn.functional as F
import torch_geometric

from utilities import log, SolutionGraphDataset, Scheduler
from model.graph_init import GraphInitialization
from model.adaptive_slicing import AdaptiveSlicing
from model.latent_evolution import LatentTrajectoryEvolution
from model.deslicing_decoder import DeslicingDecoder, extract_var_types
from model.constraint_loss import ConstraintViolationLoss, build_soft_predictions


# ======================================================================
# OptiFlow Model — wraps 4 steps into a single nn.Module
# ======================================================================

class OptiFlowModel(torch.nn.Module):
    """
    Full 4-step pipeline as a single Module for easy save/load.
    """

    def __init__(self, cons_nfeats=5, edge_nfeats=1, var_nfeats=23,
                 emb_size=64, n_conv_rounds=2,
                 n_slices=64, n_attn_heads=4,
                 n_transformer_layers=4, n_evolve_steps=3,
                 int_range_threshold=10, dropout=0.1,
                 stochastic_depth_rate=0.1, use_grad_checkpoint=False):
        super().__init__()

        self.graph_init = GraphInitialization(
            cons_nfeats=cons_nfeats, edge_nfeats=edge_nfeats,
            var_nfeats=var_nfeats, emb_size=emb_size,
            n_conv_rounds=n_conv_rounds, dropout=dropout,
        )
        self.slicer = AdaptiveSlicing(
            emb_size=emb_size, n_slices=n_slices,
            n_heads=n_attn_heads, dropout=dropout,
        )
        self.evolver = LatentTrajectoryEvolution(
            emb_size=emb_size, n_layers=n_transformer_layers,
            n_heads=n_attn_heads, n_evolve_steps=n_evolve_steps,
            dropout=dropout, stochastic_depth_rate=stochastic_depth_rate,
            use_grad_checkpoint=use_grad_checkpoint,
        )
        self.decoder = DeslicingDecoder(
            emb_size=emb_size, int_range_threshold=int_range_threshold,
            dropout=dropout,
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, var_batch=None):
        """
        Full forward pass through 4 steps.

        Returns
        -------
        result : dict  — decoder output
        var_types : LongTensor [N]
        z_var_0 : Tensor [N, D]
        attn_weights : Tensor [N, K]
        intermediates : list of Tensor [B*K, D]
        """
        # Step 1: Graph feature fusion
        z_var_0 = self.graph_init(
            constraint_features, edge_indices, edge_features,
            variable_features,
        )

        # Step 2: Adaptive slicing
        tokens, token_batch, attn_weights = self.slicer(
            z_var_0, batch=var_batch,
        )

        # Step 3: Latent trajectory evolution
        evolved, token_batch_out, intermediates = self.evolver(
            tokens, token_batch,
        )

        # Step 4: Deslicing & decoding
        var_types = extract_var_types(variable_features)
        result = self.decoder(
            evolved, token_batch_out, attn_weights,
            var_types, z_var_0=z_var_0, var_batch=var_batch,
            variable_features=variable_features,
        )

        return result, var_types, z_var_0, attn_weights, intermediates


# ======================================================================
# Training loop
# ======================================================================

def compute_losses(model, result, var_types, attn_weights,
                   sol_values, edge_indices, edge_features,
                   constraint_features, variable_features,
                   cv_loss_fn, config):
    """Compute all losses and return a dict."""

    # ---- Task loss (decoder combined) ----
    task = model.decoder.combined_loss(
        result, sol_values, var_types,
        gamma=config['focal_gamma'],
        label_smoothing_bin=config['ls_bin'],
        label_smoothing_int=config['ls_int'],
        huber_delta=config['huber_delta'],
        weight_bin=config['w_bin'],
        weight_int_small=config['w_int_small'],
        weight_int_large=config['w_int_large'],
        weight_round=config['w_round'],
    )

    # ---- Constraint violation ----
    cv = cv_loss_fn(
        result, variable_features.shape[0], var_types,
        edge_indices, edge_features, constraint_features,
        variable_features=variable_features,
    )

    # ---- Slicing regularization ----
    loss_entropy = model.slicer.entropy_loss(attn_weights)
    loss_diversity = model.slicer.diversity_loss()

    # ---- Total ----
    total = (task['total']
             + config['w_cv'] * cv['penalty']
             + config['w_entropy'] * loss_entropy
             + config['w_diversity'] * loss_diversity)

    return {
        'total': total,
        'binary': task['binary'],
        'int_small': task['int_small'],
        'int_large': task['int_large'],
        'rounding': task['rounding'],
        'cv_penalty': cv['penalty'],
        'cv_mean': cv['mean_viol'],
        'cv_max': cv['max_viol'],
        'cv_n_violated': cv['n_violated'],
        'entropy_reg': loss_entropy,
        'diversity_reg': loss_diversity,
    }


def compute_accuracy(result, sol_values, var_types):
    """Compute per-type prediction accuracy."""
    metrics = {}

    # Binary accuracy (round prob to 0/1)
    if result['mask_bin'].any():
        pred = (result['prob_bin'].squeeze(-1) > 0.5).float()
        gt = sol_values[result['mask_bin']]
        metrics['acc_bin'] = (pred == gt).float().mean().item()
        metrics['n_bin'] = result['mask_bin'].sum().item()
    else:
        metrics['acc_bin'] = 0.0
        metrics['n_bin'] = 0

    # Small-range int accuracy (argmax + offset vs target)
    if result['mask_int_small'].any():
        argmax = result['logits_int_small'].argmax(dim=-1).float()
        pred = argmax + result['int_small_offsets']
        gt = sol_values[result['mask_int_small']]
        metrics['acc_int_small'] = (pred.round() == gt.round()).float().mean().item()
        metrics['n_int_small'] = result['mask_int_small'].sum().item()
    else:
        metrics['acc_int_small'] = 0.0
        metrics['n_int_small'] = 0

    # Large-range int: MAE
    if result['mask_int_large'].any():
        pred = result['pred_int_large'].squeeze(-1)
        gt = sol_values[result['mask_int_large']]
        metrics['mae_int_large'] = (pred - gt).abs().mean().item()
        metrics['n_int_large'] = result['mask_int_large'].sum().item()
    else:
        metrics['mae_int_large'] = 0.0
        metrics['n_int_large'] = 0

    return metrics


def train_epoch(model, loader, optimizer, cv_loss_fn, config, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_metrics = {}
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        n_vars = batch.variable_features.shape[0]

        # Build var_batch
        # torch_geometric batches variable_features from multiple graphs;
        # we need to figure out which variable belongs to which graph.
        # Use batch.batch for variable nodes (second partition).
        # SolutionBipartiteNodeData stores constraint + variable nodes.
        n_cons = batch.constraint_features.shape[0]
        # In bipartite data, batch vector has n_cons + n_vars entries.
        # Variable indices start after constraint indices.
        if hasattr(batch, 'batch') and batch.batch is not None:
            full_batch = batch.batch  # [n_cons + n_vars]
            var_batch = full_batch[n_cons:]  # last n_vars entries
        else:
            var_batch = torch.zeros(n_vars, dtype=torch.long, device=device)

        # Forward
        result, var_types, z_var_0, attn_weights, intermediates = model(
            batch.constraint_features, batch.edge_index,
            batch.edge_attr, batch.variable_features,
            var_batch=var_batch,
        )

        # Losses
        losses = compute_losses(
            model, result, var_types, attn_weights,
            batch.sol_values, batch.edge_index, batch.edge_attr,
            batch.constraint_features, batch.variable_features,
            cv_loss_fn, config,
        )

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['grad_clip'])

        optimizer.step()

        # Metrics
        with torch.no_grad():
            acc = compute_accuracy(result, batch.sol_values, var_types)

        B = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        total_loss += losses['total'].item() * B
        n_graphs += B

        # Accumulate metrics
        for key in losses:
            if key == 'total':
                continue
            k = f'loss_{key}'
            val = losses[key].item() if torch.is_tensor(losses[key]) else losses[key]
            total_metrics[k] = total_metrics.get(k, 0) + val * B
        for key, val in acc.items():
            total_metrics[key] = total_metrics.get(key, 0) + val * B

    # Average
    avg_loss = total_loss / max(n_graphs, 1)
    avg_metrics = {k: v / max(n_graphs, 1) for k, v in total_metrics.items()}
    avg_metrics['total_loss'] = avg_loss

    return avg_metrics


@torch.no_grad()
def validate(model, loader, cv_loss_fn, config, device):
    """Validate (no gradient)."""
    model.eval()
    total_loss = 0
    total_metrics = {}
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        n_vars = batch.variable_features.shape[0]
        n_cons = batch.constraint_features.shape[0]

        if hasattr(batch, 'batch') and batch.batch is not None:
            var_batch = batch.batch[n_cons:]
        else:
            var_batch = torch.zeros(n_vars, dtype=torch.long, device=device)

        result, var_types, z_var_0, attn_weights, intermediates = model(
            batch.constraint_features, batch.edge_index,
            batch.edge_attr, batch.variable_features,
            var_batch=var_batch,
        )

        losses = compute_losses(
            model, result, var_types, attn_weights,
            batch.sol_values, batch.edge_index, batch.edge_attr,
            batch.constraint_features, batch.variable_features,
            cv_loss_fn, config,
        )

        acc = compute_accuracy(result, batch.sol_values, var_types)

        B = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        total_loss += losses['total'].item() * B
        n_graphs += B

        for key in losses:
            if key == 'total':
                continue
            k = f'loss_{key}'
            val = losses[key].item() if torch.is_tensor(losses[key]) else losses[key]
            total_metrics[k] = total_metrics.get(k, 0) + val * B
        for key, val in acc.items():
            total_metrics[key] = total_metrics.get(key, 0) + val * B

    avg_loss = total_loss / max(n_graphs, 1)
    avg_metrics = {k: v / max(n_graphs, 1) for k, v in total_metrics.items()}
    avg_metrics['total_loss'] = avg_loss

    return avg_metrics


def format_metrics(metrics, prefix=''):
    """Format metrics dict into a readable string."""
    parts = []

    # Total loss
    parts.append(f"loss={metrics['total_loss']:.4f}")

    # Individual losses (always show)
    parts.append(f"bin={metrics.get('loss_binary', 0):.4f}")
    parts.append(f"is={metrics.get('loss_int_small', 0):.4f}")
    parts.append(f"il={metrics.get('loss_int_large', 0):.4f}")
    parts.append(f"rnd={metrics.get('loss_rounding', 0):.4f}")
    parts.append(f"cv={metrics.get('loss_cv_penalty', 0):.4f}")
    parts.append(f"ent={metrics.get('loss_entropy_reg', 0):.4f}")
    parts.append(f"div={metrics.get('loss_diversity_reg', 0):.4f}")

    # Accuracy
    if metrics.get('n_bin', 0) > 0:
        parts.append(f"acc_bin={metrics['acc_bin']:.4f}")
    if metrics.get('n_int_small', 0) > 0:
        parts.append(f"acc_is={metrics['acc_int_small']:.4f}")
    if metrics.get('n_int_large', 0) > 0:
        parts.append(f"mae_il={metrics['mae_int_large']:.2f}")

    return f"{prefix}{' | '.join(parts)}"


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train OptiFlow 4-step model for MILP solution prediction.')
    parser.add_argument(
        'problem',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack', 'SC'],
        help='MILP instance type.',
    )
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='CUDA GPU id (-1 for CPU).')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--emb-size', type=int, default=64)
    parser.add_argument('--n-slices', type=int, default=64)
    parser.add_argument('--n-evolve-steps', type=int, default=3)
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience.')
    parser.add_argument('--loss-type', choices=['focal', 'bce'], default='focal',
                        help='Binary loss type: "focal" (default) or "bce".')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Custom data directory. All sample_*.pkl inside '
                             'will be auto-split into train/valid by seed.')
    parser.add_argument('--n-train', type=int, default=240,
                        help='Number of training samples when auto-splitting (default: 240).')
    parser.add_argument('--n-valid', type=int, default=60,
                        help='Number of validation samples when auto-splitting (default: 60).')
    args = parser.parse_args()

    # ---- Device ----
    if args.gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:0'

    # ---- Seed ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ---- Paths ----
    problem_folders = {
        'setcover': 'setcover/500r_1000c_0.05d',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
        'mknapsack': 'mknapsack/100_6',
        'SC': 'SC',
    }

    if args.data_dir is not None:
        data_dir = pathlib.Path(args.data_dir)
    else:
        data_dir = pathlib.Path('data/samples') / problem_folders[args.problem]

    run_dir = pathlib.Path(f'trained_models/optiflow/{args.problem}/{args.seed}')
    run_dir.mkdir(parents=True, exist_ok=True)
    logfile = str(run_dir / 'train_log.txt')

    # ---- Data ----
    if args.data_dir is not None:
        # Auto-split mode: collect all sample_*.pkl from the directory (and subdirs)
        all_files = sorted(str(f) for f in data_dir.rglob('sample_*.pkl'))
        if not all_files:
            print(f"ERROR: No sample_*.pkl found in {data_dir}")
            sys.exit(1)

        n_total = len(all_files)
        n_train = args.n_train
        n_valid = args.n_valid

        if n_train + n_valid > n_total:
            print(f"ERROR: Requested {n_train} train + {n_valid} valid = "
                  f"{n_train + n_valid}, but only {n_total} files found.")
            sys.exit(1)

        # Shuffle with seed and split
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(n_total)
        train_files = [all_files[i] for i in indices[:n_train]]
        valid_files = [all_files[i] for i in indices[n_train:n_train + n_valid]]

        log(f"Auto-split from {data_dir}: {n_total} total -> "
            f"{len(train_files)} train, {len(valid_files)} valid (seed={args.seed})", logfile)
    else:
        train_files = sorted(str(f) for f in (data_dir / 'train').glob('sample_*.pkl'))
        valid_files = sorted(str(f) for f in (data_dir / 'valid').glob('sample_*.pkl'))

        if not train_files:
            print(f"ERROR: No training data found in {data_dir / 'train'}")
            print("Run 02_generate_dataset.py first.")
            sys.exit(1)

    log(f"Train files: {len(train_files)}", logfile)
    log(f"Valid files: {len(valid_files)}", logfile)

    train_data = SolutionGraphDataset(train_files)
    valid_data = SolutionGraphDataset(valid_files) if valid_files else None

    train_loader = torch_geometric.loader.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=False,
    )
    valid_loader = None
    if valid_data and len(valid_data) > 0:
        valid_loader = torch_geometric.loader.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False,
            num_workers=0,
        )

    # ---- Detect feature dimensions ----
    sample0 = train_data[0]
    cons_nfeats = sample0.constraint_features.shape[1]
    var_nfeats = sample0.variable_features.shape[1]
    log(f"Feature dims: cons={cons_nfeats}, var={var_nfeats}", logfile)

    # ---- Model ----
    model = OptiFlowModel(
        cons_nfeats=cons_nfeats, var_nfeats=var_nfeats,
        emb_size=args.emb_size, n_slices=args.n_slices,
        n_evolve_steps=args.n_evolve_steps,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,}", logfile)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- Loss config ----
    focal_gamma = 2.0 if args.loss_type == 'focal' else 0.0
    log(f"Binary loss type: {args.loss_type} (gamma={focal_gamma})", logfile)

    config = {
        'focal_gamma': focal_gamma,
        'ls_bin': 0.01,
        'ls_int': 0.05,
        'huber_delta': 1.0,
        'w_bin': 1.0,
        'w_int_small': 1.0,
        'w_int_large': 1.0,
        'w_round': 0.1,
        'w_cv': 0.5,
        'w_entropy': 0.01,
        'w_diversity': 0.01,
        'grad_clip': 1.0,
    }
    log(f"Loss config: {json.dumps(config, indent=2)}", logfile)

    cv_loss_fn = ConstraintViolationLoss(lambda_mean=1.0, lambda_max=0.1)

    # ---- Save config ----
    with open(run_dir / 'config.json', 'w') as f:
        json.dump({
            'problem': args.problem,
            'seed': args.seed,
            'emb_size': args.emb_size,
            'n_slices': args.n_slices,
            'n_evolve_steps': args.n_evolve_steps,
            'cons_nfeats': cons_nfeats,
            'var_nfeats': var_nfeats,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'loss_type': args.loss_type,
            'n_params': n_params,
            'loss_config': config,
        }, f, indent=2)

    # ---- Training loop ----
    best_val_loss = float('inf')
    bad_epochs = 0

    log("=" * 60, logfile)
    log("Starting training...", logfile)
    log("=" * 60, logfile)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, cv_loss_fn, config, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        msg = format_metrics(train_metrics, prefix='TRAIN ')
        msg += f" | lr={lr_now:.2e} | {elapsed:.1f}s"

        # Gate values (evolution gate openness)
        gate_vals = model.evolver.get_gate_values()
        msg += f" | gates=[{', '.join(f'{g:.3f}' for g in gate_vals)}]"

        log(f"Epoch {epoch:3d} {msg}", logfile)

        # Validate
        if valid_loader is not None:
            val_metrics = validate(
                model, valid_loader, cv_loss_fn, config, device)
            val_msg = format_metrics(val_metrics, prefix='VALID ')
            log(f"         {val_msg}", logfile)

            val_loss = val_metrics['total_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), run_dir / 'best_model.pt')
                log("         ** New best model saved **", logfile)
            else:
                bad_epochs += 1

            if bad_epochs >= args.patience:
                log(f"Early stopping after {args.patience} epochs "
                    f"without improvement.", logfile)
                break
        else:
            # No validation data — save every epoch
            torch.save(model.state_dict(), run_dir / 'best_model.pt')

    # ---- Final evaluation ----
    log("=" * 60, logfile)
    log("Training complete.", logfile)

    if (run_dir / 'best_model.pt').exists():
        model.load_state_dict(torch.load(run_dir / 'best_model.pt',
                                         map_location=device))
        log("Loaded best model.", logfile)

    if valid_loader:
        final = validate(model, valid_loader, cv_loss_fn, config, device)
        log(format_metrics(final, prefix='FINAL VALID '), logfile)

    log(f"Model saved to: {run_dir / 'best_model.pt'}", logfile)
    log(f"Log saved to:   {logfile}", logfile)
