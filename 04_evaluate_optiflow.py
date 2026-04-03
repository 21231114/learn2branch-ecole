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

    # Use custom model weights and test data directory
    python 04_evaluate_optiflow.py --model-path /path/to/model.pt --data-dir /path/to/test/samples -g 0
"""

import os
import sys

# Must set CUDA_VISIBLE_DEVICES before importing torch
def _pre_parse_gpu():
    for i, arg in enumerate(sys.argv):
        if arg in ('-g', '--gpu') and i + 1 < len(sys.argv):
            gpu = sys.argv[i + 1]
            if gpu == '-1':
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu
            return
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
_pre_parse_gpu()

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
        device = 'cpu'
    else:
        device = 'cuda:0'
    return device


# ======================================================================
# Model loading
# ======================================================================

def load_model(model_dir, device, model_path=None):
    """Load trained OptiFlow model.

    Args:
        model_dir: Directory containing config.json (and optionally best_model.pt).
        device: Torch device string.
        model_path: If given, load weights from this .pt file instead of
                     model_dir/best_model.pt.  The config.json is looked up in
                     the same directory as model_path first, then model_dir.
    """
    from model.graph_init import GraphInitialization
    from model.adaptive_slicing import AdaptiveSlicing
    from model.latent_evolution import LatentTrajectoryEvolution
    from model.deslicing_decoder import DeslicingDecoder

    # We import the class from training script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Determine config.json location
    config = None
    search_dirs = []
    if model_path is not None:
        search_dirs.append(pathlib.Path(model_path).parent)
    if model_dir is not None:
        search_dirs.append(model_dir)
    for d in search_dirs:
        cfg = d / 'config.json'
        if cfg.exists():
            with open(cfg) as f:
                config = json.load(f)
            break
    if config is None:
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

    # Determine which weights file to load
    if model_path is not None:
        weights_path = pathlib.Path(model_path)
    elif model_dir is not None:
        weights_path = model_dir / 'best_model.pt'
    else:
        weights_path = None

    if weights_path is not None and weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded model from {weights_path}")
    else:
        print(f"WARNING: No model weights found at {weights_path}")
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

def load_baseline_results(data_dir, solver_name):
    """
    Load pre-computed baseline results from 04_evaluate_baseline.py output CSV.

    Returns a dict mapping instance basename -> (obj, time, status).
    """
    csv_path = data_dir.parent / f'baseline_results_{solver_name}.csv'
    if not csv_path.exists():
        return None, str(csv_path)

    import csv as _csv
    baseline = {}
    with open(csv_path, newline='') as f:
        for row in _csv.DictReader(f):
            obj = float(row['obj_baseline']) if row['obj_baseline'] not in ('', 'None') else None
            t   = float(row['time_baseline']) if row['time_baseline'] not in ('', 'None') else None
            baseline[row['instance']] = (obj, t, row['status_baseline'])
    return baseline, str(csv_path)


def evaluate_solver_handoff(model, data_dir, instance_dir, device,
                            solver_name='gurobi', time_limit=60,
                            e2e_time_limit=None, verbose=False):
    """
    Evaluate solver handoff: model predictions → fix variables → solve.

    Compares against pre-computed baseline results from baseline_results_{solver}.csv
    (produced by 04_evaluate_baseline.py).  Run that script first.

    Parameters
    ----------
    e2e_time_limit : float or None
        If set, caps the total per-sample wall time (infer + fixing + solver).
        The solver time limit is dynamically reduced so that the total does not
        exceed this budget.  Samples that already exceed the budget after
        inference are marked as 'e2e_timelimit'.
    """
    import re
    from model.deslicing_decoder import extract_var_types
    from model.solver_handoff import TrustRegionSolver

    def _numeric_key(path):
        nums = re.findall(r'\d+', os.path.basename(path))
        return int(nums[-1]) if nums else 0

    test_files = sorted(
        (str(f) for f in data_dir.glob('sample_*.pkl')),
        key=_numeric_key,
    )
    if not test_files:
        print(f"No test files found in {data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Solver Handoff Evaluation — {len(test_files)} samples")
    if e2e_time_limit is not None:
        print(f"Solver: {solver_name}, E2E time limit: {e2e_time_limit}s "
              f"(solver time dynamically adjusted)")
    else:
        print(f"Solver: {solver_name}, Solver time limit: {time_limit}s")
    print(f"Baseline: read from pre-computed CSV (if available)")
    print(f"Note: handoff time = model inference + fixing + {solver_name} solving")
    print(f"{'='*60}")

    solver = TrustRegionSolver(
        threshold_high=0.95, threshold_low=0.05,
        max_backtrack_steps=4, solver=solver_name,
        time_limit=time_limit, verbose=verbose,
    )

    # ---- Load pre-computed baseline results ----
    baseline_map, baseline_csv = load_baseline_results(data_dir, solver_name)
    if baseline_map is None:
        print(f"  [WARNING] Baseline results not found at {baseline_csv}")
        print(f"  Run 04_evaluate_baseline.py first to generate baseline results.")
        print(f"  Continuing without baseline comparison.")

    results_csv = str(data_dir.parent / f'handoff_results_{solver_name}.csv')
    fieldnames = ['sample', 'instance', 'n_vars', 'n_fixed',
                  'backtrack_step', 'obj_baseline', 'obj_handoff',
                  'time_baseline', 'time_e2e',
                  'time_prep', 'time_infer', 'time_fixing', 'time_solver',
                  'status_baseline', 'status_handoff',
                  'improvement_pct']

    # ---- Resume: load already-completed instances from existing CSV ----
    done_instances = set()
    if os.path.exists(results_csv):
        with open(results_csv, newline='') as f:
            for row in csv.DictReader(f):
                done_instances.add(row['instance'])
        if done_instances:
            print(f"  [RESUME] Found {len(done_instances)} already-completed "
                  f"instances in {results_csv}, skipping them.")

    # Accumulators for summary statistics
    times_base, times_e2e = [], []
    times_prep, times_infer, times_fixing, times_solver_hand = [], [], [], []
    objs_base, objs_hand = [], []
    statuses_base, statuses_hand = [], []
    n_skipped = 0
    n_resumed = 0

    # Re-load stats from completed instances for summary
    if done_instances and os.path.exists(results_csv):
        with open(results_csv, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    times_e2e.append(float(row['time_e2e']))
                    times_prep.append(float(row['time_prep']))
                    times_infer.append(float(row['time_infer']))
                    times_fixing.append(float(row['time_fixing']))
                    times_solver_hand.append(float(row['time_solver']))
                    if row['time_baseline'] not in ('', 'N/A', 'None'):
                        times_base.append(float(row['time_baseline']))
                    if row['status_baseline'] != 'N/A':
                        statuses_base.append(row['status_baseline'])
                    statuses_hand.append(row['status_handoff'])
                    if row['obj_baseline'] not in ('', 'None', 'N/A'):
                        objs_base.append(float(row['obj_baseline']))
                    if row['obj_handoff'] not in ('', 'None', 'N/A'):
                        objs_hand.append(float(row['obj_handoff']))
                    n_resumed += 1
                except (ValueError, KeyError):
                    pass

    write_header = not bool(done_instances)
    with open(results_csv, 'a' if done_instances else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, test_file in enumerate(test_files):
            with gzip.open(test_file, 'rb') as f:
                sample_data = pickle.load(f)

            instance_path = sample_data.get('instance', '')
            if not instance_path or not os.path.exists(instance_path):
                print(f"  [SKIP] Sample {i+1}: instance not found ({instance_path})", flush=True)
                n_skipped += 1
                continue

            instance_name = os.path.basename(instance_path)

            # Skip already-completed instances (resume support)
            if instance_name in done_instances:
                print(f"  [CACHED] Sample {i+1}: {instance_name} (already done)", flush=True)
                continue

            # Get variable names
            sol_info = sample_data.get('solution', {})
            var_names = None
            if sol_info.get('sol_vals'):
                var_names = list(sol_info['sol_vals'].keys())

            # Load observation — tensor preparation
            t_prep_start = time.time()
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
            time_prep = time.time() - t_prep_start

            # ---- Lookup baseline from pre-computed results ----
            if baseline_map is not None and instance_name in baseline_map:
                obj_base, time_base, status_base = baseline_map[instance_name]
            else:
                obj_base, time_base, status_base = None, None, 'N/A'

            # ---- Handoff: model inference + solver with fixings ----
            t_e2e_start = time.time()
            with torch.no_grad():
                result, var_types, _, _, _ = model(
                    cons_feats, edge_idx, edge_vals, var_feats,
                    var_batch=var_batch,
                )
            time_infer = time.time() - t_e2e_start

            # Check e2e budget after inference
            if e2e_time_limit is not None and time_infer >= e2e_time_limit:
                # Inference alone already exceeded the budget
                time_fixing = 0.0
                time_solver_h = 0.0
                time_e2e = time_infer
                sol_hand, status_hand, obj_hand = None, 'e2e_timelimit', None
                info_hand = {'n_fixed': 0, 'backtrack_step': 0,
                             'solving_time': 0.0}
                print(f"  [TIMEOUT] Sample {i+1}: inference ({time_infer:.2f}s) "
                      f"exceeded e2e limit ({e2e_time_limit}s)", flush=True)
            else:
                # Dynamically cap solver time to remaining e2e budget
                if e2e_time_limit is not None:
                    remaining = e2e_time_limit - (time.time() - t_e2e_start)
                    solver.time_limit = max(remaining, 0.1)
                else:
                    solver.time_limit = time_limit

                t_fix_start = time.time()
                try:
                    sol_hand, status_hand, obj_hand, info_hand = \
                        solver.backtracking_solve(
                            instance_path, result, n_vars, var_types,
                            var_names=var_names, use_mip_start=True,
                        )
                    time_solver_h = info_hand.get('solving_time', 0.0)
                    time_fixing = (time.time() - t_fix_start) - time_solver_h
                    time_e2e = time.time() - t_e2e_start
                except Exception as e:
                    print(f"  [ERROR] Sample {i+1}: handoff solver error: {e}", flush=True)
                    n_skipped += 1
                    continue

            # ---- Compare ----
            improvement = 0.0
            if obj_base is not None and obj_hand is not None and obj_base != 0:
                improvement = (obj_base - obj_hand) / abs(obj_base) * 100

            gap_hand = info_hand.get('mip_gap', None)
            viol_hand = info_hand.get('max_violation', None)
            feas_hand = info_hand.get('feasible', False)

            row = {
                'sample': i + 1,
                'instance': instance_name,
                'n_vars': n_vars,
                'n_fixed': info_hand.get('n_fixed', 0),
                'backtrack_step': info_hand.get('backtrack_step', 0),
                'obj_baseline': obj_base,
                'obj_handoff': obj_hand,
                'time_baseline': f"{time_base:.3f}" if time_base is not None else 'N/A',
                'time_e2e': f"{time_e2e:.3f}",
                'time_prep': f"{time_prep:.3f}",
                'time_infer': f"{time_infer:.3f}",
                'time_fixing': f"{time_fixing:.3f}",
                'time_solver': f"{time_solver_h:.3f}",
                'status_baseline': status_base,
                'status_handoff': status_hand,
                'improvement_pct': f"{improvement:.2f}",
            }
            writer.writerow(row)
            csvfile.flush()

            if time_base is not None:
                times_base.append(time_base)
            times_e2e.append(time_e2e)
            times_prep.append(time_prep)
            times_infer.append(time_infer)
            times_fixing.append(time_fixing)
            times_solver_hand.append(time_solver_h)
            if status_base != 'N/A':
                statuses_base.append(status_base)
            statuses_hand.append(status_hand)
            if obj_base is not None:
                objs_base.append(obj_base)
            if obj_hand is not None:
                objs_hand.append(obj_hand)

            # Per-sample log
            gap_hand_str = f"{gap_hand:.4f}" if gap_hand is not None else "N/A"
            viol_hand_str = f"{viol_hand:.2e}" if viol_hand is not None else "N/A"
            time_base_str = f"{time_base:.2f}s" if time_base is not None else "N/A"

            print(f"\n[{i+1:4d}/{len(test_files)}] {instance_name}"
                  f"  vars={n_vars}  fixed={info_hand.get('n_fixed', 0)}", flush=True)
            print(f"    baseline: obj={obj_base}  t={time_base_str}"
                  f"  status={status_base}", flush=True)
            print(f"    handoff:  obj={obj_hand}  t={time_e2e:.2f}s"
                  f"  (prep={time_prep:.2f}s  infer={time_infer:.2f}s"
                  f"  fixing={time_fixing:.2f}s  solver={time_solver_h:.2f}s)"
                  f"  status={status_hand}  gap={gap_hand_str}"
                  f"  feasible={feas_hand}  max_viol={viol_hand_str}", flush=True)
            if obj_base is not None and obj_hand is not None:
                speedup_str = (f"  speedup={time_base/max(time_e2e,1e-6):.2f}x"
                               if time_base is not None else "")
                print(f"    Δobj={improvement:+.2f}%{speedup_str}", flush=True)

    # ---- Summary statistics ----
    n_done = len(times_e2e)
    print(f"\n{'='*70}")
    print(f"Summary ({n_done} solved, {n_resumed} resumed, {n_skipped} skipped, time_limit={time_limit}s)")
    print(f"{'='*70}")

    if not times_e2e:
        print("  No samples completed.")
        print(f"\nResults saved to {results_csv}")
        return

    n_opt_hand = sum(1 for s in statuses_hand if s == 'optimal')
    n_tl_hand = sum(1 for s in statuses_hand if s.startswith('timelimit'))
    n_inf_hand = sum(1 for s in statuses_hand if s == 'infeasible')

    has_base = bool(times_base)
    base_col = f"{np.mean(times_base):>15.3f}" if has_base else f"{'N/A':>15s}"
    base_obj = f"{np.mean(objs_base):>15.4f}" if objs_base else f"{'N/A':>15s}"

    print(f"\n  {'':30s} {'Baseline':>15s} {'Handoff':>15s}")
    print(f"  {'-'*62}")
    print(f"  {'Avg time (s)':30s} {base_col} {np.mean(times_e2e):>15.3f}")
    print(f"  {'  infer (s)':30s} {'':>15s} {np.mean(times_infer):>15.3f}")
    print(f"  {'  fixing (s)':30s} {'':>15s} {np.mean(times_fixing):>15.3f}")
    print(f"  {'  solver (s)':30s} {'':>15s} {np.mean(times_solver_hand):>15.3f}")
    if objs_hand:
        print(f"  {'Avg obj':30s} {base_obj} {np.mean(objs_hand):>15.4f}")
    print(f"  {'Optimal':30s} {'':>15s} {n_opt_hand:>15d}")
    print(f"  {'Timelimit (with solution)':30s} {'':>15s} {n_tl_hand:>15d}")
    print(f"  {'Infeasible':30s} {'':>15s} {n_inf_hand:>15d}")

    if has_base and objs_base and objs_hand:
        paired_better = 0
        paired_equal = 0
        paired_worse = 0
        for ob, oh in zip(objs_base, objs_hand):
            if abs(ob - oh) < 1e-6:
                paired_equal += 1
            elif oh < ob:
                paired_better += 1
            else:
                paired_worse += 1

        print(f"\n  Pairwise objective comparison (on {len(objs_base)} instances with solutions):")
        print(f"    Handoff better : {paired_better}")
        print(f"    Equal          : {paired_equal}")
        print(f"    Handoff worse  : {paired_worse}")

        speedup = np.mean(times_base) / np.mean(times_e2e)
        print(f"\n  Avg speedup: {speedup:.2f}x")

    print(f"\nResults saved to {results_csv}")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate OptiFlow model.')
    parser.add_argument(
        'problem', nargs='?', default=None,
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack'],
        help='Problem type. Required when --model-path and --data-dir are not both given.',
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
    parser.add_argument('--time-limit', type=float, default=300,
                        help='Solver time limit in seconds (default: 300).')
    parser.add_argument('--e2e-time-limit', type=float, default=None,
                        help='End-to-end per-sample time limit in seconds '
                             '(infer + fixing + solver). If set, the solver '
                             'time limit is dynamically reduced to fit within '
                             'this budget. Default: None (no e2e limit).')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model weights (.pt file). '
                             'Overrides the default model directory.')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to test dataset directory (containing sample_*.pkl). '
                             'Overrides the default data directory.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Validate: need either problem or both --model-path and --data-dir
    if args.problem is None and (args.model_path is None or args.data_dir is None):
        parser.error('Either specify a problem type, or provide both '
                     '--model-path and --data-dir.')

    device = setup_and_import(args)

    # ---- Paths ----
    problem_folders = {
        'setcover': 'setcover/500r_1000c_0.05d',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
        'mknapsack': 'mknapsack/100_6',
    }

    # Model path
    model_pt = None
    model_dir = None
    if args.model_path is not None:
        model_pt = pathlib.Path(args.model_path)
        if not model_pt.exists():
            print(f"ERROR: model file not found: {model_pt}")
            sys.exit(1)
        model_dir = model_pt.parent
    elif args.problem is not None:
        model_dir = pathlib.Path(
            f'trained_models/optiflow/{args.problem}/{args.seed}')

    # Data directory
    if args.data_dir is not None:
        test_data_dir = pathlib.Path(args.data_dir)
        if not test_data_dir.exists():
            print(f"ERROR: data directory not found: {test_data_dir}")
            sys.exit(1)
    elif args.problem is not None:
        data_dir = pathlib.Path('data/samples') / problem_folders[args.problem]
        # Prefer test set, fall back to valid, then train
        for split in ('test', 'valid', 'train'):
            test_data_dir = data_dir / split
            if test_data_dir.exists() and list(test_data_dir.glob('sample_*.pkl')):
                break
        else:
            print(f"No data found in {data_dir}")
            sys.exit(1)

    print(f"Problem: {args.problem or 'N/A'}")
    print(f"Model: {model_pt or model_dir}")
    print(f"Data dir: {test_data_dir}")
    print(f"Device: {device}")

    # ---- Load model ----
    model, config = load_model(model_dir, device, model_path=model_pt)

    # ---- Evaluate ----
    if args.mode in ('predict', 'both'):
        evaluate_predictions(model, test_data_dir, device,
                             verbose=args.verbose)

    if args.mode in ('solve', 'both'):
        if args.problem is not None:
            instance_dir = pathlib.Path('data/instances') / args.problem
        else:
            instance_dir = test_data_dir  # fallback
        evaluate_solver_handoff(
            model, test_data_dir, instance_dir, device,
            solver_name=args.solver,
            time_limit=args.time_limit,
            e2e_time_limit=args.e2e_time_limit,
            verbose=args.verbose,
        )
