"""
04_evaluate_baseline.py — Gurobi-only baseline evaluation.

Runs pure Gurobi (no model fixings) on test instances and records results
to a CSV file.  Summary metrics are also appended to the model's train_log.txt
so the baseline result is co-located with training history.

Usage:
    python 04_evaluate_baseline.py setcover -g -1
    python 04_evaluate_baseline.py setcover -g -1 --time-limit 300
    python 04_evaluate_baseline.py setcover -g -1 --solver scip
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

from utilities import log


# ======================================================================
# Solver helpers (mirrors solver_handoff but without model dependency)
# ======================================================================

def solve_gurobi(instance_path, time_limit, verbose=False):
    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1 if verbose else 0)
    env.start()
    model = gp.read(instance_path, env)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)

    # Callback: record (time, obj) at each new incumbent; gap computed post-hoc
    iteration_log = []

    def _callback(m, where):
        if where == GRB.Callback.MIPSOL:
            cur_time = m.cbGet(GRB.Callback.RUNTIME)
            cur_obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            iteration_log.append({'time': cur_time, 'obj': cur_obj})

    model.optimize(_callback)

    status_map = {
        GRB.OPTIMAL:   'optimal',
        GRB.INFEASIBLE: 'infeasible',
        GRB.INF_OR_UNBD: 'infeasible',
        GRB.UNBOUNDED: 'unbounded',
        GRB.TIME_LIMIT: 'timelimit',
    }
    status = status_map.get(model.Status, str(model.Status))

    info = {
        'solving_time': model.Runtime,
        'n_nodes': int(model.NodeCount) if hasattr(model, 'NodeCount') else 0,
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


def solve_scip(instance_path, time_limit, verbose=False):
    from pyscipopt import Model

    model = Model()
    model.setParam('display/verblevel', 4 if verbose else 0)
    model.setParam('limits/time', time_limit)
    model.readProblem(instance_path)
    model.optimize()

    scip_status = model.getStatus()
    status_map = {
        'optimal':    'optimal',
        'infeasible': 'infeasible',
        'unbounded':  'unbounded',
        'timelimit':  'timelimit',
    }
    status = status_map.get(scip_status, scip_status)

    iteration_log = []

    info = {
        'solving_time': model.getSolvingTime(),
        'n_nodes': model.getNNodes(),
        'iteration_log': iteration_log,
    }

    if model.getNSols() > 0:
        best_sol = model.getBestSol()
        obj_val = model.getSolObjVal(best_sol)
        iteration_log.append({
            'time': model.getSolvingTime(),
            'obj': obj_val,
        })
        if status == 'timelimit':
            status = 'timelimit*'
        info['feasible'] = True
        info['mip_gap'] = None
        info['max_violation'] = None
        return status, obj_val, info
    else:
        info['feasible'] = False
        info['mip_gap'] = None
        info['max_violation'] = None
        return status, None, info


def solve_instance(instance_path, solver_name, time_limit, verbose=False):
    if solver_name == 'gurobi':
        return solve_gurobi(instance_path, time_limit, verbose)
    elif solver_name == 'scip':
        return solve_scip(instance_path, time_limit, verbose)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


# ======================================================================
# Main evaluation
# ======================================================================

def evaluate_baseline(data_dir, solver_name, time_limit, verbose, logfile,
                      trajectory_time=1000):
    def _numeric_key(path):
        nums = re.findall(r'\d+', os.path.basename(path))
        return int(nums[-1]) if nums else 0

    test_files = sorted(
        (str(f) for f in data_dir.glob('sample_*.pkl')),
        key=_numeric_key,
    )
    if not test_files:
        log(f"No test files found in {data_dir}", logfile)
        return

    log(f"{'='*60}", logfile)
    log(f"Baseline Evaluation — {len(test_files)} samples", logfile)
    log(f"Solver: {solver_name}, Time limit: {time_limit}s, "
        f"Trajectory time: {trajectory_time}s", logfile)
    log(f"{'='*60}", logfile)

    results_csv = str(data_dir.parent / f'baseline_results_{solver_name}.csv')
    fieldnames = [
        'sample', 'instance', 'n_vars',
        'obj_baseline', 'obj_at_trajectory', 'gap_at_trajectory',
        'time_baseline',
        'status_baseline', 'mip_gap', 'feasible', 'max_violation',
    ]

    # Load already-completed instances from existing CSV
    done_instances = set()
    if os.path.exists(results_csv):
        with open(results_csv, newline='') as f:
            for row in csv.DictReader(f):
                done_instances.add(row['instance'])

    times, objs, statuses = [], [], []
    n_skipped = 0

    with open(results_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not done_instances:
            writer.writeheader()

        for i, test_file in enumerate(test_files):
            with gzip.open(test_file, 'rb') as f:
                sample_data = pickle.load(f)

            instance_path = sample_data.get('instance', '')
            if not instance_path or not os.path.exists(instance_path):
                log(f"  [SKIP] Sample {i+1}: instance not found ({instance_path})",
                    logfile)
                n_skipped += 1
                continue

            if os.path.basename(instance_path) in done_instances:
                print(f"[{i+1:4d}/{len(test_files)}] {os.path.basename(instance_path)}  [already done, skip]",
                      flush=True)
                continue

            # Count variables from observation
            _, _, var_feats = sample_data['observation']
            n_vars = len(var_feats)

            try:
                t0 = time.time()
                status, obj, info = solve_instance(
                    instance_path, solver_name, time_limit, verbose)
                wall_time = time.time() - t0
                solving_time = info.get('solving_time', wall_time)
            except Exception as e:
                log(f"  [ERROR] Sample {i+1}: {e}", logfile)
                n_skipped += 1
                continue

            # ref_obj = final objective after full solve (e.g. 3600s)
            ref_obj = obj

            # Post-hoc: compute gap_to_ref for trajectory within trajectory_time,
            # using the final obj as reference
            obj_at_traj = None
            gap_at_traj = None
            iteration_log = info.get('iteration_log', [])
            if iteration_log and ref_obj is not None:
                trajectory = []
                for entry in iteration_log:
                    if entry['time'] <= trajectory_time:
                        gap = (abs(entry['obj'] - ref_obj) / max(abs(ref_obj), 1e-10)
                               if ref_obj != 0
                               else abs(entry['obj'] - ref_obj))
                        trajectory.append({
                            'time': entry['time'],
                            'obj': entry['obj'],
                            'gap_to_ref': gap,
                        })

                if trajectory:
                    # Best incumbent at trajectory_time cutoff
                    obj_at_traj = trajectory[-1]['obj']
                    gap_at_traj = trajectory[-1]['gap_to_ref']

                    iter_log_dir = data_dir.parent / f'baseline_iteration_logs_{solver_name}'
                    iter_log_dir.mkdir(parents=True, exist_ok=True)
                    inst_name = os.path.splitext(os.path.basename(instance_path))[0]
                    iter_log_path = str(iter_log_dir / f'{inst_name}.json')
                    with open(iter_log_path, 'w') as jf:
                        json.dump({
                            'instance': os.path.basename(instance_path),
                            'ref_obj': ref_obj,
                            'time_limit': time_limit,
                            'trajectory_time': trajectory_time,
                            'trajectory': trajectory,
                        }, jf, indent=2)

            gap_str = (f"{info['mip_gap']:.4f}"
                       if info.get('mip_gap') is not None else 'N/A')
            viol_str = (f"{info['max_violation']:.2e}"
                        if info.get('max_violation') is not None else 'N/A')
            gap_traj_str = (f"{gap_at_traj:.6f}"
                            if gap_at_traj is not None else 'N/A')

            print(f"[{i+1:4d}/{len(test_files)}] {os.path.basename(instance_path)}"
                  f"  vars={n_vars}"
                  f"  obj_3600s={obj}  obj_{int(trajectory_time)}s={obj_at_traj}"
                  f"  gap_{int(trajectory_time)}s={gap_traj_str}"
                  f"  t={solving_time:.2f}s"
                  f"  status={status}  mip_gap={gap_str}",
                  flush=True)

            row = {
                'sample': i + 1,
                'instance': os.path.basename(instance_path),
                'n_vars': n_vars,
                'obj_baseline': obj,
                'obj_at_trajectory': obj_at_traj,
                'gap_at_trajectory': gap_at_traj,
                'time_baseline': f"{solving_time:.3f}",
                'status_baseline': status,
                'mip_gap': info.get('mip_gap'),
                'feasible': info.get('feasible'),
                'max_violation': info.get('max_violation'),
            }
            writer.writerow(row)
            csvfile.flush()

            times.append(solving_time)
            statuses.append(status)
            if obj is not None:
                objs.append(obj)

    # ---- Summary ----
    n_done = len(times)
    log(f"{'='*60}", logfile)
    log(f"Baseline Summary ({n_done} solved, {n_skipped} skipped,"
        f" solver={solver_name}, time_limit={time_limit}s)", logfile)
    log(f"{'='*60}", logfile)

    if not times:
        log("  No samples completed.", logfile)
        log(f"Results saved to {results_csv}", logfile)
        return

    n_opt = sum(1 for s in statuses if s == 'optimal')
    n_tl  = sum(1 for s in statuses if s.startswith('timelimit'))
    n_inf = sum(1 for s in statuses if s == 'infeasible')

    log(f"  Avg time (s) : {np.mean(times):.3f}", logfile)
    log(f"  Min time (s) : {np.min(times):.3f}", logfile)
    log(f"  Max time (s) : {np.max(times):.3f}", logfile)
    if objs:
        log(f"  Avg obj      : {np.mean(objs):.4f}", logfile)
        log(f"  Min obj      : {np.min(objs):.4f}", logfile)
        log(f"  Max obj      : {np.max(objs):.4f}", logfile)
    log(f"  Optimal      : {n_opt}", logfile)
    log(f"  Timelimit*   : {n_tl}", logfile)
    log(f"  Infeasible   : {n_inf}", logfile)
    log(f"Results saved to {results_csv}", logfile)


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gurobi-only baseline evaluation (no model).')
    parser.add_argument(
        'data_dir',
        help='Path to the directory containing sample_*.pkl files.',
    )
    parser.add_argument(
        '--solver', choices=['gurobi', 'scip'], default='gurobi',
    )
    parser.add_argument('--time-limit', type=float, default=300,
                        help='Solver time limit in seconds (default: 300).')
    parser.add_argument('--trajectory-time', type=float, default=1000,
                        help='Record solving trajectory within this time (default: 1000s). '
                             'Gap is computed post-hoc using the final objective as reference.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    test_data_dir = pathlib.Path(args.data_dir)
    if not test_data_dir.exists() or not list(test_data_dir.glob('sample_*.pkl')):
        print(f"No sample_*.pkl files found in {test_data_dir}")
        sys.exit(1)

    logfile = str(test_data_dir.parent / 'baseline_eval_log.txt')

    log(f"Data dir: {test_data_dir}", logfile)
    log(f"Solver  : {args.solver}, time limit: {args.time_limit}s, "
        f"trajectory time: {args.trajectory_time}s", logfile)

    evaluate_baseline(
        test_data_dir,
        solver_name=args.solver,
        time_limit=args.time_limit,
        verbose=args.verbose,
        logfile=logfile,
        trajectory_time=args.trajectory_time,
    )
