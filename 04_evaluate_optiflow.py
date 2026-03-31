"""
04_evaluate_optiflow.py — Inference & evaluation for OptiFlow model.

Two evaluation modes:
    1. Solution prediction accuracy (vs. known optimal solutions)
    2. Solver handoff (Trust Region → Gurobi/SCIP → final solution)

Usage:
    # Mode 1: Prediction accuracy on test samples
    python 04_evaluate_optiflow.py setcover --mode predict -g 0

    # Mode 2: Solver handoff on test instances
    python 04_evaluate_optiflow.py setcover --mode solve -g 0 --solver gurobi

    # Both modes with detailed output
    python 04_evaluate_optiflow.py setcover --mode both --verbose
"""

import os
import sys
import argparse
import pathlib
import time
import json
import csv
import gzip
import pickle
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric


# ======================================================================
# Imports (after setting CUDA_VISIBLE_DEVICES)
# ======================================================================

def setup_and_import(args):
    """Set up device and import model modules."""
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = 'cuda:0'
    return device


# ======================================================================
# Model loading
# ======================================================================

def load_model(model_dir, device):
    """Load trained OptiFlow model from directory."""
    from model.graph_init import GraphInitialization
    from model.adaptive_slicing import AdaptiveSlicing
    from model.latent_evolution import LatentTrajectoryEvolution
    from model.deslicing_decoder import DeslicingDecoder

    # We import the class from training script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    config_path = model_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'cons_nfeats': 5, 'var_nfeats': 23,
            'emb_size': 64, 'n_slices': 64,
            'n_evolve_steps': 3,
        }

    # Import OptiFlowModel from training script
    from importlib import import_module
    spec = __import__('03_train_optiflow', fromlist=['OptiFlowModel'])
    OptiFlowModel = spec.OptiFlowModel

    model = OptiFlowModel(
        cons_nfeats=config.get('cons_nfeats', 5),
        var_nfeats=config.get('var_nfeats', 23),
        emb_size=config.get('emb_size', 64),
        n_slices=config.get('n_slices', 64),
        n_evolve_steps=config.get('n_evolve_steps', 3),
        dropout=0.0,  # no dropout at inference
    ).to(device)

    model_path = model_dir / 'best_model.pt'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"WARNING: No model weights found at {model_path}")
        print("  Running with random weights (for testing pipeline only)")

    model.eval()
    return model, config


# ======================================================================
# Mode 1: Prediction accuracy
# ======================================================================

def evaluate_predictions(model, data_dir, device, verbose=False):
    """
    Evaluate model predictions against known optimal solutions.

    Loads samples from test set and computes:
        - Binary accuracy
        - Small-range integer accuracy
        - Large-range integer MAE
        - Trust region coverage (what % of vars can be confidently fixed)
    """
    from utilities import SolutionGraphDataset
    from model.deslicing_decoder import extract_var_types
    from model.solver_handoff import TrustRegionSolver

    test_files = sorted(str(f) for f in data_dir.glob('sample_*.pkl'))
    if not test_files:
        print(f"No test files found in {data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Prediction Evaluation — {len(test_files)} samples")
    print(f"{'='*60}")

    dataset = SolutionGraphDataset(test_files)
    solver = TrustRegionSolver(threshold_high=0.95, threshold_low=0.05)

    # Accumulators
    total_bin_correct = 0
    total_bin_count = 0
    total_is_correct = 0
    total_is_count = 0
    total_il_ae = 0.0
    total_il_count = 0
    total_fixed = 0
    total_vars = 0
    total_fixed_correct = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        sample = sample.to(device)

        n_vars = sample.variable_features.shape[0]
        var_batch = torch.zeros(n_vars, dtype=torch.long, device=device)

        with torch.no_grad():
            result, var_types, z_var_0, attn_weights, _ = model(
                sample.constraint_features, sample.edge_index,
                sample.edge_attr, sample.variable_features,
                var_batch=var_batch,
            )

        sol = sample.sol_values.to(device)

        # ---- Binary accuracy ----
        if result['mask_bin'].any():
            pred_bin = (result['prob_bin'].squeeze(-1) > 0.5).float()
            gt_bin = sol[result['mask_bin']]
            correct = (pred_bin == gt_bin).sum().item()
            total_bin_correct += correct
            total_bin_count += result['mask_bin'].sum().item()

        # ---- Small-range integer accuracy ----
        if result['mask_int_small'].any():
            argmax = result['logits_int_small'].argmax(dim=-1).float()
            pred_is = (argmax + result['int_small_offsets']).round()
            gt_is = sol[result['mask_int_small']].round()
            total_is_correct += (pred_is == gt_is).sum().item()
            total_is_count += result['mask_int_small'].sum().item()

        # ---- Large-range integer MAE ----
        if result['mask_int_large'].any():
            pred_il = result['pred_int_large'].squeeze(-1)
            gt_il = sol[result['mask_int_large']]
            total_il_ae += (pred_il - gt_il).abs().sum().item()
            total_il_count += result['mask_int_large'].sum().item()

        # ---- Trust region coverage ----
        fixings = solver.extract_fixings(result, n_vars, var_types)
        n_fixed = len(fixings)
        total_fixed += n_fixed
        total_vars += n_vars

        # Check if fixings are correct
        for idx, (val, conf) in fixings.items():
            if abs(sol[idx].item() - val) < 0.5:
                total_fixed_correct += 1

        if verbose and i < 5:
            print(f"\n  Sample {i+1}: {n_vars} vars")
            n_bin = result['mask_bin'].sum().item()
            n_is = result['mask_int_small'].sum().item()
            n_il = result['mask_int_large'].sum().item()
            print(f"    Types: {n_bin} bin, {n_is} small-int, {n_il} large-int")
            if n_bin > 0:
                acc = (pred_bin == gt_bin).float().mean().item()
                print(f"    Binary acc: {acc:.4f}")
            print(f"    Trust region: {n_fixed}/{n_vars} fixed "
                  f"({n_fixed/n_vars*100:.1f}%)")

            # Show probability distribution
            if n_bin > 0:
                probs = result['prob_bin'].squeeze(-1)
                print(f"    prob_bin stats: "
                      f"mean={probs.mean():.3f}, "
                      f"std={probs.std():.3f}, "
                      f"<0.05: {(probs<0.05).sum().item()}, "
                      f">0.95: {(probs>0.95).sum().item()}")

            # Show gate values
            gates = model.evolver.get_gate_values()
            print(f"    Evolution gates: "
                  f"[{', '.join(f'{g:.3f}' for g in gates)}]")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    if total_bin_count > 0:
        print(f"  Binary accuracy:     {total_bin_correct/total_bin_count:.4f}"
              f"  ({total_bin_correct}/{total_bin_count})")
    if total_is_count > 0:
        print(f"  Small-int accuracy:  {total_is_correct/total_is_count:.4f}"
              f"  ({total_is_correct}/{total_is_count})")
    if total_il_count > 0:
        print(f"  Large-int MAE:       {total_il_ae/total_il_count:.4f}"
              f"  (over {total_il_count} vars)")

    fix_pct = total_fixed / max(total_vars, 1) * 100
    fix_acc = total_fixed_correct / max(total_fixed, 1)
    print(f"\n  Trust Region:")
    print(f"    Coverage: {total_fixed}/{total_vars} "
          f"({fix_pct:.1f}% of vars confidently fixed)")
    print(f"    Fixing accuracy: {fix_acc:.4f} "
          f"({total_fixed_correct}/{total_fixed} fixings correct)")


# ======================================================================
# Mode 2: Solver handoff
# ======================================================================

def evaluate_solver_handoff(model, data_dir, instance_dir, device,
                            solver_name='gurobi', time_limit=60,
                            verbose=False):
    """
    Evaluate solver handoff: model predictions → fix variables → solve.

    Compares:
        1. Solver alone (baseline)
        2. Solver with model fixings (trust region)
        3. Solver with MIP Start
    """
    from model.deslicing_decoder import extract_var_types
    from model.solver_handoff import TrustRegionSolver

    test_files = sorted(str(f) for f in data_dir.glob('sample_*.pkl'))
    if not test_files:
        print(f"No test files found in {data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Solver Handoff Evaluation — {len(test_files)} samples")
    print(f"Solver: {solver_name}, Time limit: {time_limit}s")
    print(f"{'='*60}")

    solver = TrustRegionSolver(
        threshold_high=0.95, threshold_low=0.05,
        max_backtrack_steps=4, solver=solver_name,
        time_limit=time_limit, verbose=verbose,
    )

    results_csv = str(data_dir.parent / f'handoff_results_{solver_name}.csv')
    fieldnames = ['sample', 'instance', 'n_vars', 'n_fixed',
                  'backtrack_step', 'obj_baseline', 'obj_handoff',
                  'time_baseline', 'time_handoff',
                  'status_baseline', 'status_handoff',
                  'improvement_pct']

    with open(results_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, test_file in enumerate(test_files):
            with gzip.open(test_file, 'rb') as f:
                sample_data = pickle.load(f)

            instance_path = sample_data.get('instance', '')
            if not instance_path or not os.path.exists(instance_path):
                if verbose:
                    print(f"  Sample {i+1}: instance not found, skipping solver")
                continue

            # Load observation
            cons_feats, (edge_idx, edge_vals), var_feats = \
                sample_data['observation']
            cons_feats = torch.FloatTensor(
                np.asarray(cons_feats, dtype=np.float32)).to(device)
            edge_idx = torch.LongTensor(
                np.asarray(edge_idx, dtype=np.int64)).to(device)
            edge_vals = torch.FloatTensor(
                np.asarray(edge_vals, dtype=np.float32)).unsqueeze(-1).to(device)
            var_feats = torch.FloatTensor(
                np.asarray(var_feats, dtype=np.float32)).to(device)

            n_vars = var_feats.shape[0]
            var_batch = torch.zeros(n_vars, dtype=torch.long, device=device)

            # ---- Model prediction ----
            with torch.no_grad():
                result, var_types, _, _, _ = model(
                    cons_feats, edge_idx, edge_vals, var_feats,
                    var_batch=var_batch,
                )

            # Get variable names from solution data
            sol_info = sample_data.get('solution', {})
            var_names = None
            if sol_info.get('sol_vals'):
                var_names = list(sol_info['sol_vals'].keys())

            # ---- Baseline: solver alone ----
            t0 = time.time()
            try:
                sol_base, status_base, obj_base, info_base = solver._solve(
                    instance_path, {}, var_names)
                time_base = time.time() - t0
            except Exception as e:
                if verbose:
                    print(f"  Sample {i+1}: baseline solver error: {e}")
                continue

            # ---- Handoff: solver with fixings ----
            t1 = time.time()
            try:
                sol_hand, status_hand, obj_hand, info_hand = \
                    solver.backtracking_solve(
                        instance_path, result, n_vars, var_types,
                        var_names=var_names, use_mip_start=True,
                    )
                time_hand = time.time() - t1
            except Exception as e:
                if verbose:
                    print(f"  Sample {i+1}: handoff solver error: {e}")
                continue

            # ---- Compare ----
            improvement = 0.0
            if obj_base is not None and obj_hand is not None and obj_base != 0:
                improvement = (obj_base - obj_hand) / abs(obj_base) * 100

            row = {
                'sample': i + 1,
                'instance': os.path.basename(instance_path),
                'n_vars': n_vars,
                'n_fixed': info_hand.get('n_fixed', 0),
                'backtrack_step': info_hand.get('backtrack_step', 0),
                'obj_baseline': obj_base,
                'obj_handoff': obj_hand,
                'time_baseline': f"{time_base:.2f}",
                'time_handoff': f"{time_hand:.2f}",
                'status_baseline': status_base,
                'status_handoff': status_hand,
                'improvement_pct': f"{improvement:.2f}",
            }
            writer.writerow(row)
            csvfile.flush()

            if verbose or i < 3:
                print(f"\n  Sample {i+1}: {os.path.basename(instance_path)}")
                print(f"    Vars: {n_vars}, "
                      f"Fixed: {info_hand.get('n_fixed', 0)}")
                print(f"    Baseline: obj={obj_base}, "
                      f"time={time_base:.2f}s, status={status_base}")
                print(f"    Handoff:  obj={obj_hand}, "
                      f"time={time_hand:.2f}s, status={status_hand}")
                if improvement != 0:
                    print(f"    Speedup or improvement: {improvement:.2f}%")

    print(f"\nResults saved to {results_csv}")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate OptiFlow model.')
    parser.add_argument(
        'problem',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack'],
    )
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument(
        '--mode', choices=['predict', 'solve', 'both'], default='predict',
        help='Evaluation mode: predict (accuracy), solve (handoff), both.',
    )
    parser.add_argument(
        '--solver', choices=['gurobi', 'scip'], default='gurobi',
        help='Solver for handoff mode.',
    )
    parser.add_argument('--time-limit', type=float, default=60)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = setup_and_import(args)

    # ---- Paths ----
    problem_folders = {
        'setcover': 'setcover/500r_1000c_0.05d',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
        'mknapsack': 'mknapsack/100_6',
    }

    model_dir = pathlib.Path(
        f'trained_models/optiflow/{args.problem}/{args.seed}')
    data_dir = pathlib.Path('data/samples') / problem_folders[args.problem]

    # Prefer test set, fall back to valid, then train
    for split in ('test', 'valid', 'train'):
        test_data_dir = data_dir / split
        if test_data_dir.exists() and list(test_data_dir.glob('sample_*.pkl')):
            break
    else:
        print(f"No data found in {data_dir}")
        sys.exit(1)

    print(f"Problem: {args.problem}")
    print(f"Model dir: {model_dir}")
    print(f"Data dir: {test_data_dir}")
    print(f"Device: {device}")

    # ---- Load model ----
    model, config = load_model(model_dir, device)

    # ---- Evaluate ----
    if args.mode in ('predict', 'both'):
        evaluate_predictions(model, test_data_dir, device,
                             verbose=args.verbose)

    if args.mode in ('solve', 'both'):
        instance_dir = pathlib.Path('data/instances') / args.problem
        evaluate_solver_handoff(
            model, test_data_dir, instance_dir, device,
            solver_name=args.solver,
            time_limit=args.time_limit,
            verbose=args.verbose,
        )
