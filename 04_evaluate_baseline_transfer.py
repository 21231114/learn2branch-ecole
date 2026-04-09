"""
04_evaluate_baseline_transfer.py — Gurobi baseline evaluation on transfer sets.

Evaluates all three setcover transfer datasets:
  - setcover/500r_1000c_0.05d/transfer
  - setcover/1000r_1000c_0.05d/transfer
  - setcover/2000r_1000c_0.05d/transfer

Results are saved per-dataset as CSV files, and an overall summary JSON is
written to data/samples/setcover/baseline_transfer_summary.json.

Usage:
    python 04_evaluate_baseline_transfer.py
    python 04_evaluate_baseline_transfer.py --time-limit 600
    python 04_evaluate_baseline_transfer.py --verbose
"""

import os
import sys
import argparse
import pathlib
import time
import csv
import gzip
import pickle
import re
import json
import numpy as np


# ======================================================================
# Gurobi solver
# ======================================================================

def solve_gurobi(instance_path, time_limit, verbose=False, ref_obj=None):
    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1 if verbose else 0)
    env.start()
    model = gp.read(instance_path, env)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)

    # Callback to record objective value and gap at each new incumbent
    iteration_log = []

    def _callback(m, where):
        if where == GRB.Callback.MIPSOL:
            cur_time = m.cbGet(GRB.Callback.RUNTIME)
            cur_obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            entry = {'time': cur_time, 'obj': cur_obj}
            if ref_obj is not None and ref_obj != 0:
                entry['gap_to_ref'] = abs(cur_obj - ref_obj) / max(abs(ref_obj), 1e-10)
            elif ref_obj is not None:
                entry['gap_to_ref'] = abs(cur_obj - ref_obj)
            iteration_log.append(entry)

    model.optimize(_callback)

    status_map = {
        GRB.OPTIMAL:     'optimal',
        GRB.INFEASIBLE:  'infeasible',
        GRB.INF_OR_UNBD: 'infeasible',
        GRB.UNBOUNDED:   'unbounded',
        GRB.TIME_LIMIT:  'timelimit',
    }
    status = status_map.get(model.Status, str(model.Status))

    info = {
        'solving_time': model.Runtime,
        'n_nodes':      int(model.NodeCount),
        'iteration_log': iteration_log,
    }

    if model.SolCount > 0:
        obj_val = model.ObjVal
        try:
            info['mip_gap'] = model.MIPGap
        except Exception:
            info['mip_gap'] = 0.0
        info['feasible'] = True
        try:
            info['max_violation'] = model.MaxVio
        except Exception:
            info['max_violation'] = 0.0
        if status == 'timelimit':
            status = 'timelimit*'
        return status, obj_val, info
    else:
        info['mip_gap'] = None
        info['feasible'] = False
        info['max_violation'] = None
        return status, None, info


# ======================================================================
# Evaluate one dataset directory
# ======================================================================

def _numeric_key(path):
    nums = re.findall(r'\d+', os.path.basename(path))
    return int(nums[-1]) if nums else 0


def evaluate_transfer_set(data_dir: pathlib.Path, time_limit: float, verbose: bool):
    """Run Gurobi on every sample_*.pkl in data_dir.

    Returns (rows, summary_dict).
    """
    test_files = sorted(
        (str(f) for f in data_dir.glob('sample_*.pkl')),
        key=_numeric_key,
    )
    if not test_files:
        print(f"[WARN] No sample files found in {data_dir}")
        return [], {}

    label = data_dir.parent.name  # e.g. 500r_1000c_0.05d
    print(f"\n{'='*60}")
    print(f"Dataset : {label}/transfer  ({len(test_files)} samples)")
    print(f"Solver  : gurobi   Time limit: {time_limit}s")
    print(f"{'='*60}")

    results_csv = str(data_dir.parent / f'baseline_{data_dir.name}_gurobi.csv')
    fieldnames = [
        'sample', 'instance', 'n_vars',
        'obj_baseline', 'ref_obj', 'gap_to_ref',
        'time_baseline', 'n_nodes',
        'status_baseline', 'mip_gap', 'feasible', 'max_violation',
    ]

    times, objs, gaps, nodes, statuses = [], [], [], [], []
    n_skipped = 0
    rows = []

    with open(results_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, test_file in enumerate(test_files):
            with gzip.open(test_file, 'rb') as f:
                sample_data = pickle.load(f)

            instance_path = sample_data.get('instance', '')
            if not instance_path or not os.path.exists(instance_path):
                print(f"  [SKIP] Sample {i+1}: instance not found ({instance_path})")
                n_skipped += 1
                continue

            _, _, var_feats = sample_data['observation']
            n_vars = len(var_feats)

            # Extract reference objective value (Gurobi 3600s solution)
            ref_obj = None
            if 'solution' in sample_data and 'obj_val' in sample_data['solution']:
                ref_obj = sample_data['solution']['obj_val']

            try:
                t0 = time.time()
                status, obj, info = solve_gurobi(instance_path, time_limit, verbose,
                                                 ref_obj=ref_obj)
                wall_time = time.time() - t0
                solving_time = info.get('solving_time', wall_time)
            except Exception as e:
                print(f"  [ERROR] Sample {i+1}: {e}")
                n_skipped += 1
                continue

            # Compute final gap to reference
            gap_to_ref = None
            if obj is not None and ref_obj is not None:
                if ref_obj != 0:
                    gap_to_ref = abs(obj - ref_obj) / max(abs(ref_obj), 1e-10)
                else:
                    gap_to_ref = abs(obj - ref_obj)

            gap_str  = (f"{info['mip_gap']:.4f}"
                        if info.get('mip_gap') is not None else 'N/A')
            viol_str = (f"{info['max_violation']:.2e}"
                        if info.get('max_violation') is not None else 'N/A')
            gap_ref_str = (f"{gap_to_ref:.6f}"
                           if gap_to_ref is not None else 'N/A')

            print(f"  [{i+1:4d}/{len(test_files)}] {os.path.basename(instance_path)}"
                  f"  vars={n_vars}"
                  f"  obj={obj}  ref_obj={ref_obj}  gap_ref={gap_ref_str}"
                  f"  t={solving_time:.2f}s"
                  f"  status={status}  gap={gap_str}"
                  f"  nodes={info.get('n_nodes',0)}  max_viol={viol_str}",
                  flush=True)

            # Save per-instance iteration log (time, obj, gap at each incumbent)
            iteration_log = info.get('iteration_log', [])
            if iteration_log:
                iter_log_dir = data_dir.parent / f'baseline_iteration_logs_gurobi'
                iter_log_dir.mkdir(parents=True, exist_ok=True)
                inst_name = os.path.splitext(os.path.basename(instance_path))[0]
                iter_log_path = str(iter_log_dir / f'{inst_name}.json')
                with open(iter_log_path, 'w') as jf:
                    json.dump({
                        'instance': os.path.basename(instance_path),
                        'ref_obj': ref_obj,
                        'time_limit': time_limit,
                        'iterations': iteration_log,
                    }, jf, indent=2)

            row = {
                'sample':          i + 1,
                'instance':        os.path.basename(instance_path),
                'n_vars':          n_vars,
                'obj_baseline':    obj,
                'ref_obj':         ref_obj,
                'gap_to_ref':      gap_to_ref,
                'time_baseline':   f"{solving_time:.3f}",
                'n_nodes':         info.get('n_nodes', 0),
                'status_baseline': status,
                'mip_gap':         info.get('mip_gap'),
                'feasible':        info.get('feasible'),
                'max_violation':   info.get('max_violation'),
            }
            writer.writerow(row)
            csvfile.flush()
            rows.append(row)

            times.append(solving_time)
            statuses.append(status)
            nodes.append(info.get('n_nodes', 0))
            if obj is not None:
                objs.append(obj)
            if info.get('mip_gap') is not None:
                gaps.append(info['mip_gap'])

    # ---- per-dataset summary ----
    n_done = len(times)
    n_opt  = sum(1 for s in statuses if s == 'optimal')
    n_tl   = sum(1 for s in statuses if s.startswith('timelimit'))
    n_inf  = sum(1 for s in statuses if s == 'infeasible')

    summary = {
        'dataset':          label,
        'split':            'transfer',
        'n_samples':        len(test_files),
        'n_solved':         n_done,
        'n_skipped':        n_skipped,
        'n_optimal':        n_opt,
        'n_timelimit':      n_tl,
        'n_infeasible':     n_inf,
        'time_mean':        float(np.mean(times)) if times else None,
        'time_min':         float(np.min(times))  if times else None,
        'time_max':         float(np.max(times))  if times else None,
        'time_median':      float(np.median(times)) if times else None,
        'obj_mean':         float(np.mean(objs))  if objs else None,
        'obj_min':          float(np.min(objs))   if objs else None,
        'obj_max':          float(np.max(objs))   if objs else None,
        'obj_median':       float(np.median(objs)) if objs else None,
        'mip_gap_mean':     float(np.mean(gaps))  if gaps else None,
        'mip_gap_max':      float(np.max(gaps))   if gaps else None,
        'n_nodes_mean':     float(np.mean(nodes)) if nodes else None,
        'n_nodes_max':      int(np.max(nodes))    if nodes else None,
        'results_csv':      results_csv,
    }

    print(f"\n  --- Summary: {label}/transfer ---")
    print(f"  Solved        : {n_done} / {len(test_files)}")
    print(f"  Optimal       : {n_opt}")
    print(f"  Timelimit*    : {n_tl}")
    print(f"  Infeasible    : {n_inf}")
    if times:
        print(f"  Time (s)      : mean={np.mean(times):.3f}  "
              f"min={np.min(times):.3f}  max={np.max(times):.3f}  "
              f"median={np.median(times):.3f}")
    if objs:
        print(f"  Obj           : mean={np.mean(objs):.4f}  "
              f"min={np.min(objs):.4f}  max={np.max(objs):.4f}")
    if gaps:
        print(f"  MIP gap       : mean={np.mean(gaps):.4f}  max={np.max(gaps):.4f}")
    if nodes:
        print(f"  B&B nodes     : mean={np.mean(nodes):.1f}  max={np.max(nodes)}")
    print(f"  CSV           : {results_csv}")

    return rows, summary


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gurobi baseline evaluation on specified sample directories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two specific directories, evaluated in order:
  python 04_evaluate_baseline_transfer.py \\
      data/samples/setcover/500r_1000c_0.05d/transfer \\
      data/samples/setcover/1000r_1000c_0.05d/transfer

  # Default (all three transfer sets):
  python 04_evaluate_baseline_transfer.py
""",
    )
    parser.add_argument(
        'dirs', nargs='*',
        metavar='DIR',
        help='Directories containing sample_*.pkl files, evaluated in order. '
             'If omitted, defaults to all three setcover transfer sets.',
    )
    parser.add_argument('--time-limit', type=float, default=300,
                        help='Solver time limit per instance (default: 300s).')
    parser.add_argument('--verbose', action='store_true',
                        help='Print Gurobi solver output.')
    args = parser.parse_args()

    if args.dirs:
        data_dirs = [pathlib.Path(d) for d in args.dirs]
    else:
        base = pathlib.Path('data/samples/setcover')
        data_dirs = [base / ds / 'transfer'
                     for ds in ('500r_1000c_0.05d', '1000r_1000c_0.05d', '2000r_1000c_0.05d')]

    all_summaries = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"[WARN] Directory not found: {data_dir}")
            continue
        _, summary = evaluate_transfer_set(data_dir, args.time_limit, args.verbose)
        if summary:
            all_summaries.append(summary)

    # ---- overall summary ----
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY — setcover transfer baseline (gurobi)")
    print(f"{'='*60}")

    total_samples  = sum(s['n_samples']   for s in all_summaries)
    total_solved   = sum(s['n_solved']    for s in all_summaries)
    total_optimal  = sum(s['n_optimal']   for s in all_summaries)
    total_tl       = sum(s['n_timelimit'] for s in all_summaries)
    total_inf      = sum(s['n_infeasible'] for s in all_summaries)

    print(f"  Datasets      : {len(all_summaries)}")
    print(f"  Total samples : {total_samples}")
    print(f"  Total solved  : {total_solved}")
    print(f"  Total optimal : {total_optimal}")
    print(f"  Total TL*     : {total_tl}")
    print(f"  Total infeas  : {total_inf}")

    for s in all_summaries:
        t_str = (f"{s['time_mean']:.3f}s avg"
                 if s['time_mean'] is not None else 'N/A')
        o_str = (f"{s['obj_mean']:.4f} avg"
                 if s['obj_mean'] is not None else 'N/A')
        g_str = (f"{s['mip_gap_mean']:.4f} avg"
                 if s['mip_gap_mean'] is not None else 'N/A')
        print(f"\n  {s['dataset']}/transfer:")
        print(f"    solved={s['n_solved']}/{s['n_samples']}  "
              f"optimal={s['n_optimal']}  tl={s['n_timelimit']}")
        print(f"    time={t_str}  obj={o_str}  gap={g_str}")

    overall = {
        'solver':         'gurobi',
        'time_limit':     args.time_limit,
        'total_samples':  total_samples,
        'total_solved':   total_solved,
        'total_optimal':  total_optimal,
        'total_timelimit': total_tl,
        'total_infeasible': total_inf,
        'per_dataset':    all_summaries,
    }

    summary_path = str(data_dirs[0].parent / 'baseline_transfer_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(overall, f, indent=2)
    print(f"\n  Summary JSON  : {summary_path}")
    print(f"{'='*60}")
