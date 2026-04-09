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
import queue
import threading

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
# Shared: rounded-prediction objective & violation computation
# ======================================================================

def compute_rounded_violations_gurobi(instance_path, var_names, rounded_values):
    """
    加载实例，代入四舍五入后的解，计算目标值和约束违反量（不调用 optimize()）。

    Parameters
    ----------
    instance_path : str
        MILP 实例文件路径（.lp / .mps）。
    var_names : list of str
        变量名列表，与 rounded_values 顺序一致。
    rounded_values : array-like
        四舍五入后的解，与 var_names 顺序一致。

    Returns
    -------
    obj_val   : float  — 四舍五入解的目标函数值
    max_viol  : float  — 最大约束违反量
    mean_viol : float  — 平均约束违反量
    n_violated: int    — 违反约束数
    n_total   : int    — 约束总数
    """
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    grb_model = gp.read(instance_path, env)
    grb_model.update()

    grb_vars = grb_model.getVars()
    constrs  = grb_model.getConstrs()
    name2idx = {v.VarName: i for i, v in enumerate(grb_vars)}

    # 构建解向量（按 Gurobi 变量顺序）
    x = np.zeros(len(grb_vars))
    for name, val in zip(var_names, rounded_values):
        gname = name[2:] if (name.startswith('t_') and name not in name2idx) else name
        if gname in name2idx:
            x[name2idx[gname]] = float(val)

    # 目标值（线性目标：obj_coeff · x）
    obj_coeffs = np.array([v.Obj for v in grb_vars])
    obj_val = float(np.dot(obj_coeffs, x))

    # 约束违反量
    A      = grb_model.getA()                  # scipy sparse (n_cons × n_vars)
    Ax     = np.asarray(A @ x).flatten()
    b      = np.array([c.RHS   for c in constrs])
    senses = [c.Sense for c in constrs]

    viol = np.zeros(len(constrs))
    for i, (sense, ax_i, b_i) in enumerate(zip(senses, Ax, b)):
        if sense == '<':
            viol[i] = max(0.0, ax_i - b_i)
        elif sense == '>':
            viol[i] = max(0.0, b_i - ax_i)
        else:                                   # '='
            viol[i] = abs(ax_i - b_i)

    grb_model.dispose()
    env.dispose()

    max_v  = float(viol.max())  if len(viol) > 0 else 0.0
    mean_v = float(viol.mean()) if len(viol) > 0 else 0.0
    n_viol = int((viol > 1e-6).sum())
    return obj_val, max_v, mean_v, n_viol, len(constrs)


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
                            e2e_time_limit=None, verbose=False,
                            trust_region_delta=None, num_workers=1):
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
        Only effective when num_workers == 1.
    trust_region_delta : int or None
        If set, use CoCo-MILP-style soft trust region constraints instead of
        hard variable fixing.  Adds auxiliary variables alpha_i and enforces
        sum(|x_i - x_star_i|) <= trust_region_delta, allowing the solver to
        deviate from model predictions by at most this total Hamming distance.
        When None (default), hard fixing (bound tightening) is used.
    num_workers : int
        Number of parallel solver worker threads.  The model inference always
        runs sequentially on the main thread (fast); the time-consuming solver
        calls are parallelised across num_workers threads.
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
    print(f"Workers: {num_workers}")
    print(f"Baseline: read from pre-computed CSV (if available)")
    print(f"Note: handoff time = model inference + fixing + {solver_name} solving")
    print(f"{'='*60}")

    # ---- Load pre-computed baseline results ----
    baseline_map, baseline_csv = load_baseline_results(data_dir, solver_name)
    if baseline_map is None:
        print(f"  [WARNING] Baseline results not found at {baseline_csv}")
        print(f"  Run 04_evaluate_baseline.py first to generate baseline results.")
        print(f"  Continuing without baseline comparison.")

    _tr_suffix = f'_tr{trust_region_delta}' if trust_region_delta is not None else ''
    results_csv = str(data_dir.parent / f'handoff_results_{solver_name}{_tr_suffix}.csv')
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

    # ---- Helper: run one solver task, return result dict ----
    def _run_solver_task(task, solver_obj):
        """Execute solver on one inference result. Used by worker threads."""
        (i, instance_path, instance_name, n_vars, var_names,
         result_cpu, var_types_cpu,
         time_prep, time_infer, t_e2e_start,
         obj_base, time_base, status_base) = task

        if e2e_time_limit is not None and time_infer >= e2e_time_limit:
            return {
                'i': i, 'instance_name': instance_name, 'n_vars': n_vars,
                'time_prep': time_prep, 'time_infer': time_infer,
                'time_fixing': 0.0, 'time_solver_h': 0.0,
                'time_e2e': time_infer,
                'obj_hand': None, 'status_hand': 'e2e_timelimit',
                'info_hand': {'n_fixed': 0, 'backtrack_step': 0, 'solving_time': 0.0},
                'obj_base': obj_base, 'time_base': time_base,
                'status_base': status_base, 'error': None,
            }

        if e2e_time_limit is not None:
            remaining = e2e_time_limit - (time.time() - t_e2e_start)
            solver_obj.time_limit = max(remaining, 0.1)
        else:
            solver_obj.time_limit = time_limit

        t_fix_start = time.time()
        try:
            sol_hand, status_hand, obj_hand, info_hand = \
                solver_obj.backtracking_solve(
                    instance_path, result_cpu, n_vars, var_types_cpu,
                    var_names=var_names, use_mip_start=True,
                )
            time_solver_h = info_hand.get('solving_time', 0.0)
            time_fixing = (time.time() - t_fix_start) - time_solver_h
            time_e2e = time.time() - t_e2e_start
            error = None
        except Exception as exc:
            return {
                'i': i, 'instance_name': instance_name, 'n_vars': n_vars,
                'error': str(exc),
                'obj_base': obj_base, 'time_base': time_base,
                'status_base': status_base,
            }

        return {
            'i': i, 'instance_name': instance_name, 'n_vars': n_vars,
            'time_prep': time_prep, 'time_infer': time_infer,
            'time_fixing': time_fixing, 'time_solver_h': time_solver_h,
            'time_e2e': time_e2e,
            'obj_hand': obj_hand, 'status_hand': status_hand,
            'info_hand': info_hand,
            'obj_base': obj_base, 'time_base': time_base,
            'status_base': status_base, 'error': None,
        }

    # ---- Worker thread function (used when num_workers > 1) ----
    def _worker_fn(solver_queue, results_queue, stop_flag):
        from model.solver_handoff import TrustRegionSolver as _TRS
        w_solver = _TRS(
            threshold_high=0.95, threshold_low=0.05,
            max_backtrack_steps=4, solver=solver_name,
            time_limit=time_limit, verbose=verbose,
            trust_region_delta=trust_region_delta,
        )
        while not stop_flag.is_set():
            try:
                task = solver_queue.get(timeout=1)
            except queue.Empty:
                continue
            if task is None:  # poison pill
                break
            res = _run_solver_task(task, w_solver)
            results_queue.put(res)

    # ---- Helper: move result tensors to CPU so they can cross thread boundary ----
    def _result_to_cpu(result, var_types):
        result_cpu = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in result.items()
        }
        return result_cpu, var_types.cpu()

    # ---- Helper: write one completed result row ----
    def _write_result(res, writer, csvfile, total):
        if res.get('error'):
            print(f"  [ERROR] Sample {res['i']+1}: {res['error']}", flush=True)
            return False  # signal skip

        i = res['i']
        instance_name = res['instance_name']
        n_vars = res['n_vars']
        time_prep = res['time_prep']
        time_infer = res['time_infer']
        time_fixing = res['time_fixing']
        time_solver_h = res['time_solver_h']
        time_e2e = res['time_e2e']
        obj_hand = res['obj_hand']
        status_hand = res['status_hand']
        info_hand = res['info_hand']
        obj_base = res['obj_base']
        time_base = res['time_base']
        status_base = res['status_base']

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

        # Save per-instance iteration log
        iteration_log = info_hand.get('iteration_log', [])
        if iteration_log:
            iter_log_dir = data_dir.parent / f'handoff_iteration_logs_{solver_name}{_tr_suffix}'
            iter_log_dir.mkdir(parents=True, exist_ok=True)
            inst_stem = os.path.splitext(instance_name)[0]
            iter_log_path = str(iter_log_dir / f'{inst_stem}.json')
            with open(iter_log_path, 'w') as jf:
                json.dump({
                    'instance': instance_name,
                    'n_fixed': info_hand.get('n_fixed', 0),
                    'backtrack_step': info_hand.get('backtrack_step', 0),
                    'time_limit': time_limit,
                    'trajectory': [{'time': e['time'], 'obj': e['obj']}
                                   for e in iteration_log],
                }, jf, indent=2)

        gap_hand_str = f"{gap_hand:.4f}" if gap_hand is not None else "N/A"
        viol_hand_str = f"{viol_hand:.2e}" if viol_hand is not None else "N/A"
        time_base_str = f"{time_base:.2f}s" if time_base is not None else "N/A"

        print(f"\n[{i+1:4d}/{total}] {instance_name}"
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
        return True

    # ================================================================
    # Main evaluation loop
    # ================================================================
    write_header = not bool(done_instances)
    with open(results_csv, 'a' if done_instances else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        if num_workers <= 1:
            # ---- Sequential path (original behaviour) ----
            solver = TrustRegionSolver(
                threshold_high=0.95, threshold_low=0.05,
                max_backtrack_steps=4, solver=solver_name,
                time_limit=time_limit, verbose=verbose,
                trust_region_delta=trust_region_delta,
            )
            for i, test_file in enumerate(test_files):
                with gzip.open(test_file, 'rb') as f:
                    sample_data = pickle.load(f)

                instance_path = sample_data.get('instance', '')
                if not instance_path or not os.path.exists(instance_path):
                    print(f"  [SKIP] Sample {i+1}: instance not found ({instance_path})", flush=True)
                    n_skipped += 1
                    continue

                instance_name = os.path.basename(instance_path)
                if instance_name in done_instances:
                    print(f"  [CACHED] Sample {i+1}: {instance_name} (already done)", flush=True)
                    continue

                sol_info = sample_data.get('solution', {})
                var_names = list(sol_info['sol_vals'].keys()) if sol_info.get('sol_vals') else None

                t_prep_start = time.time()
                cons_feats, (edge_idx, edge_vals), var_feats = sample_data['observation']
                cons_feats_t = torch.FloatTensor(np.asarray(cons_feats, dtype=np.float32)).to(device)
                edge_idx_t   = torch.LongTensor(np.asarray(edge_idx, dtype=np.int64)).to(device)
                edge_vals_t  = torch.FloatTensor(np.asarray(edge_vals, dtype=np.float32)).unsqueeze(-1).to(device)
                var_feats_t  = torch.FloatTensor(np.asarray(var_feats, dtype=np.float32)).to(device)
                n_vars = var_feats_t.shape[0]
                var_batch = torch.zeros(n_vars, dtype=torch.long, device=device)
                time_prep = time.time() - t_prep_start

                if baseline_map is not None and instance_name in baseline_map:
                    obj_base, time_base, status_base = baseline_map[instance_name]
                else:
                    obj_base, time_base, status_base = None, None, 'N/A'

                t_e2e_start = time.time()
                with torch.no_grad():
                    result, var_types, _, _, _ = model(
                        cons_feats_t, edge_idx_t, edge_vals_t, var_feats_t,
                        var_batch=var_batch,
                    )
                time_infer = time.time() - t_e2e_start

                result_cpu, var_types_cpu = _result_to_cpu(result, var_types)
                task = (i, instance_path, instance_name, n_vars, var_names,
                        result_cpu, var_types_cpu,
                        time_prep, time_infer, t_e2e_start,
                        obj_base, time_base, status_base)
                res = _run_solver_task(task, solver)

                if res.get('error'):
                    print(f"  [ERROR] Sample {i+1}: {res['error']}", flush=True)
                    n_skipped += 1
                    continue

                ok = _write_result(res, writer, csvfile, len(test_files))
                if not ok:
                    n_skipped += 1
                    continue

                if res['time_base'] is not None:
                    times_base.append(res['time_base'])
                times_e2e.append(res['time_e2e'])
                times_prep.append(res['time_prep'])
                times_infer.append(res['time_infer'])
                times_fixing.append(res['time_fixing'])
                times_solver_hand.append(res['time_solver_h'])
                if res['status_base'] != 'N/A':
                    statuses_base.append(res['status_base'])
                statuses_hand.append(res['status_hand'])
                if res['obj_base'] is not None:
                    objs_base.append(res['obj_base'])
                if res['obj_hand'] is not None:
                    objs_hand.append(res['obj_hand'])

        else:
            # ---- Parallel path: main thread infers, workers solve ----
            csv_lock = threading.Lock()
            solver_queue = queue.Queue(maxsize=2 * num_workers)
            results_queue = queue.SimpleQueue()
            stop_flag = threading.Event()

            workers = []
            for _ in range(num_workers):
                t = threading.Thread(
                    target=_worker_fn,
                    args=(solver_queue, results_queue, stop_flag),
                    daemon=True,
                )
                t.start()
                workers.append(t)

            n_dispatched = 0  # tasks sent to solver_queue (excludes skipped/cached)

            # Phase 1: infer sequentially, dispatch to solver workers
            for i, test_file in enumerate(test_files):
                with gzip.open(test_file, 'rb') as f:
                    sample_data = pickle.load(f)

                instance_path = sample_data.get('instance', '')
                if not instance_path or not os.path.exists(instance_path):
                    print(f"  [SKIP] Sample {i+1}: instance not found ({instance_path})", flush=True)
                    n_skipped += 1
                    continue

                instance_name = os.path.basename(instance_path)
                if instance_name in done_instances:
                    print(f"  [CACHED] Sample {i+1}: {instance_name} (already done)", flush=True)
                    continue

                sol_info = sample_data.get('solution', {})
                var_names = list(sol_info['sol_vals'].keys()) if sol_info.get('sol_vals') else None

                t_prep_start = time.time()
                cons_feats, (edge_idx, edge_vals), var_feats = sample_data['observation']
                cons_feats_t = torch.FloatTensor(np.asarray(cons_feats, dtype=np.float32)).to(device)
                edge_idx_t   = torch.LongTensor(np.asarray(edge_idx, dtype=np.int64)).to(device)
                edge_vals_t  = torch.FloatTensor(np.asarray(edge_vals, dtype=np.float32)).unsqueeze(-1).to(device)
                var_feats_t  = torch.FloatTensor(np.asarray(var_feats, dtype=np.float32)).to(device)
                n_vars = var_feats_t.shape[0]
                var_batch = torch.zeros(n_vars, dtype=torch.long, device=device)
                time_prep = time.time() - t_prep_start

                if baseline_map is not None and instance_name in baseline_map:
                    obj_base, time_base, status_base = baseline_map[instance_name]
                else:
                    obj_base, time_base, status_base = None, None, 'N/A'

                t_e2e_start = time.time()
                with torch.no_grad():
                    result, var_types, _, _, _ = model(
                        cons_feats_t, edge_idx_t, edge_vals_t, var_feats_t,
                        var_batch=var_batch,
                    )
                time_infer = time.time() - t_e2e_start
                print(f"  [infer] Sample {i+1}: {instance_name}  "
                      f"infer={time_infer:.3f}s", flush=True)

                result_cpu, var_types_cpu = _result_to_cpu(result, var_types)
                task = (i, instance_path, instance_name, n_vars, var_names,
                        result_cpu, var_types_cpu,
                        time_prep, time_infer, t_e2e_start,
                        obj_base, time_base, status_base)
                solver_queue.put(task)  # blocks if queue full (backpressure)
                n_dispatched += 1

            # Send poison pills to stop workers after all tasks consumed
            for _ in workers:
                solver_queue.put(None)

            # Phase 2: collect results as workers finish
            n_collected = 0
            while n_collected < n_dispatched:
                res = results_queue.get()
                n_collected += 1

                if res.get('error'):
                    print(f"  [ERROR] Sample {res['i']+1}: {res['error']}", flush=True)
                    n_skipped += 1
                    continue

                with csv_lock:
                    ok = _write_result(res, writer, csvfile, len(test_files))
                if not ok:
                    n_skipped += 1
                    continue

                if res['time_base'] is not None:
                    times_base.append(res['time_base'])
                times_e2e.append(res['time_e2e'])
                times_prep.append(res['time_prep'])
                times_infer.append(res['time_infer'])
                times_fixing.append(res['time_fixing'])
                times_solver_hand.append(res['time_solver_h'])
                if res['status_base'] != 'N/A':
                    statuses_base.append(res['status_base'])
                statuses_hand.append(res['status_hand'])
                if res['obj_base'] is not None:
                    objs_base.append(res['obj_base'])
                if res['obj_hand'] is not None:
                    objs_hand.append(res['obj_hand'])

            stop_flag.set()
            for t in workers:
                t.join(timeout=5)

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
# Mode 3: Inference-only (round predictions, compute obj & violations)
# ======================================================================

def evaluate_inference_only(model, data_dir, device, verbose=False):
    """
    仅推理 + 四舍五入 + 计算目标值和 Gurobi 约束违反量，不调用求解器。

    四舍五入规则：
        binary      : prob > 0.5 → 1, 否则 → 0
        small-int   : argmax(softmax) + offset，再 round()
        large-int   : pred_int_large.round()
        continuous  : LP 松弛值（var_feats 第 8 列，不四舍五入）
    """
    import re
    from model.deslicing_decoder import extract_var_types

    def _numeric_key(p):
        nums = re.findall(r'\d+', os.path.basename(p))
        return int(nums[-1]) if nums else 0

    test_files = sorted(
        (str(f) for f in data_dir.glob('sample_*.pkl')), key=_numeric_key
    )
    if not test_files:
        print(f"No test files found in {data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Inference-Only Evaluation — {len(test_files)} samples")
    print(f"{'='*60}")

    all_obj, all_max, all_mean, all_nviol, all_ncons = [], [], [], [], []
    n_skipped = 0

    for i, test_file in enumerate(test_files):
        import gzip, pickle
        with gzip.open(test_file, 'rb') as f:
            sample_data = pickle.load(f)

        instance_path = sample_data.get('instance', '')
        if not instance_path or not os.path.exists(instance_path):
            print(f"  [SKIP] Sample {i+1}: instance not found ({instance_path})")
            n_skipped += 1
            continue

        sol_info  = sample_data.get('solution', {})
        var_names = list(sol_info['sol_vals'].keys()) if sol_info.get('sol_vals') else None
        if var_names is None:
            print(f"  [SKIP] Sample {i+1}: no variable names")
            n_skipped += 1
            continue

        # 加载观测张量
        cons_feats, (edge_idx, edge_vals), var_feats = sample_data['observation']
        cons_feats_t  = torch.FloatTensor(np.asarray(cons_feats,  dtype=np.float32)).to(device)
        edge_idx_t    = torch.LongTensor( np.asarray(edge_idx,    dtype=np.int64)).to(device)
        edge_vals_t   = torch.FloatTensor(np.asarray(edge_vals,   dtype=np.float32)).unsqueeze(-1).to(device)
        var_feats_t   = torch.FloatTensor(np.asarray(var_feats,   dtype=np.float32)).to(device)
        n_vars        = var_feats_t.shape[0]
        var_batch     = torch.zeros(n_vars, dtype=torch.long, device=device)

        # 模型推理
        with torch.no_grad():
            result, var_types, _, _, _ = model(
                cons_feats_t, edge_idx_t, edge_vals_t, var_feats_t,
                var_batch=var_batch,
            )

        # 构建四舍五入预测向量
        x_rounded = torch.zeros(n_vars)

        if result['idx_bin'].shape[0] > 0:
            probs = result['prob_bin'].squeeze(-1).cpu()
            x_rounded[result['idx_bin'].cpu()] = (probs > 0.5).float()

        if result['idx_int_small'].shape[0] > 0:
            argmax  = result['logits_int_small'].argmax(dim=-1).float().cpu()
            offsets = result['int_small_offsets'].cpu()
            x_rounded[result['idx_int_small'].cpu()] = (argmax + offsets).round()

        if result['idx_int_large'].shape[0] > 0:
            x_rounded[result['idx_int_large'].cpu()] = \
                result['pred_int_large'].squeeze(-1).cpu().round()

        # 连续变量：使用 LP 松弛值（不四舍五入）
        vt_cpu    = var_types.cpu()
        cont_mask = (vt_cpu == 0)
        if cont_mask.any():
            x_rounded[cont_mask] = torch.FloatTensor(
                np.asarray(var_feats, dtype=np.float32))[cont_mask, 8]

        # 计算目标值和约束违反量
        try:
            obj_val, max_v, mean_v, n_viol, n_cons = compute_rounded_violations_gurobi(
                instance_path, var_names, x_rounded.numpy()
            )
        except Exception as e:
            print(f"  [ERROR] Sample {i+1}: {e}", flush=True)
            n_skipped += 1
            continue

        all_obj.append(obj_val)
        all_max.append(max_v)
        all_mean.append(mean_v)
        all_nviol.append(n_viol)
        all_ncons.append(n_cons)

        instance_name = os.path.basename(instance_path)
        print(f"[{i+1:4d}/{len(test_files)}] {instance_name}"
              f"  obj={obj_val:.4f}  max_viol={max_v:.4f}"
              f"  mean_viol={mean_v:.4f}  violated={n_viol}/{n_cons}",
              flush=True)

    n_done = len(all_obj)
    print(f"\n{'='*60}")
    print(f"Inference-Only Summary ({n_done} evaluated, {n_skipped} skipped)")
    print(f"{'='*60}")
    if all_obj:
        avg_viol = np.mean(all_nviol)
        avg_cons = np.mean(all_ncons)
        print(f"  Avg obj        : {np.mean(all_obj):.4f}")
        print(f"  Avg max_viol   : {np.mean(all_max):.4f}")
        print(f"  Avg mean_viol  : {np.mean(all_mean):.4f}")
        print(f"  Avg n_violated : {avg_viol:.1f} / {avg_cons:.1f}"
              f" ({avg_viol / max(avg_cons, 1) * 100:.1f}%)")


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
    parser.add_argument('--trust-region-delta', type=int, default=None,
                        help='If set, use CoCo-MILP-style soft trust region '
                             'constraints instead of hard variable fixing. '
                             'The solver may deviate from model predictions '
                             'by at most this total Hamming distance '
                             '(sum of |x_i - x_star_i| <= delta). '
                             'Default: None (hard fixing).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-j', '--num-workers', type=int, default=1,
                        help='Number of parallel solver worker threads '
                             '(default: 1). Model inference always runs '
                             'sequentially; only the solver is parallelised.')
    parser.add_argument('--inference_only', action='store_true',
                        help='仅推理+四舍五入，计算目标值和约束违反量，不调用求解器')
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

    # ---- Inference-only mode ----
    if args.inference_only:
        evaluate_inference_only(model, test_data_dir, device,
                                verbose=args.verbose)

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
            trust_region_delta=args.trust_region_delta,
            num_workers=args.num_workers,
        )
