import os
import re
import glob
import gzip
import argparse
import pickle
import queue
import shutil
import threading
import traceback
import numpy as np
import ecole


def extract_instance_number(filepath):
    """Extract numeric ID from instance filename like 'instance_123.lp'."""
    basename = os.path.basename(filepath)
    match = re.search(r'(\d+)', basename)
    if match:
        return int(match.group(1))
    return None


def send_orders(orders_queue, tasks, seed, time_limit, stop_flag):
    """
    Continuously send sampling orders to workers.

    Parameters
    ----------
    orders_queue : queue.Queue
        Queue to which to send orders.
    tasks : list of (instance_path, output_path) tuples
        Each task pairs an instance file with its output sample path.
    seed : int
        Random seed for reproducibility.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    rng = np.random.RandomState(seed)

    for episode, (instance, out_path) in enumerate(tasks):
        if stop_flag.is_set():
            break
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, time_limit, out_path])


def make_samples(in_queue, out_queue, stop_flag):
    """
    Worker loop: fetch an instance, record the root-node observation,
    then run the episode to completion and record the solver's solution.

    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which orders are received.
    out_queue : queue.Queue
        Output queue in which to send samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    while not stop_flag.is_set():
        try:
            item = in_queue.get(timeout=1)
        except queue.Empty:
            continue
        episode, instance, seed, time_limit, out_path = item

        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0,
                           'presolving/maxrounds': 0,
                           'limits/time': time_limit, 'timing/clocktype': 2}
        observation_function = {"node_observation": ecole.observation.NodeBipartite()}
        env = ecole.environment.Branching(observation_function=observation_function,
                                          scip_params=scip_parameters, pseudo_candidates=True)

        print(f"[w {threading.current_thread().name}] episode {episode}, seed {seed}, "
              f"processing instance '{instance}'...\n", end='')
        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        try:
            env.seed(seed)
            observation, action_set, _, done, _ = env.reset(instance)

            if observation is None:
                # 实例在根节点就被解完了，ecole 返回 None。
                # 用更严格的参数禁用预处理/切割/传播，重新加载以提取观测。
                fallback_model = ecole.scip.Model.from_file(instance)
                fallback_model.disable_presolve()
                fallback_model.disable_cuts()
                fallback_model.set_params({
                    'propagating/maxrounds': 0,
                    'propagating/maxroundsroot': 0,
                })
                fallback_env = ecole.environment.Branching(
                    observation_function=observation_function,
                    scip_params={},
                    pseudo_candidates=True,
                )
                fallback_env.seed(seed)
                observation, action_set, _, done, _ = fallback_env.reset(fallback_model)
                env = fallback_env  # 后续用 fallback_env 的 model

            node_observation = observation["node_observation"]

            scip_model = env.model.as_pyscipopt()
            variables = scip_model.getVars(transformed=True)
            lb_local = np.array([v.getLbLocal() for v in variables], dtype=np.float32).reshape(-1, 1)
            ub_local = np.array([v.getUbLocal() for v in variables], dtype=np.float32).reshape(-1, 1)
            lb_global = np.array([v.getLbGlobal() for v in variables], dtype=np.float32).reshape(-1, 1)
            ub_global = np.array([v.getUbGlobal() for v in variables], dtype=np.float32).reshape(-1, 1)
            variable_features = np.concatenate(
                [node_observation.variable_features, lb_local, ub_local, lb_global, ub_global], axis=1
            )

            root_observation = (node_observation.row_features,
                                (node_observation.edge_features.indices,
                                 node_observation.edge_features.values),
                                variable_features)

            # 使用 strong branching 跑完整个求解过程（不记录中间状态）
            if not done:
                strong_branch_fn = ecole.observation.StrongBranchingScores()
                strong_branch_fn.before_reset(env.model)

                while not done:
                    scores = strong_branch_fn.extract(env.model, done)
                    action = action_set[scores[action_set].argmax()]
                    try:
                        observation, action_set, _, done, _ = env.step(action)
                    except Exception as e:
                        done = True
                        with open("error_log.txt", "a") as f:
                            f.write(f"Error occurred solving {instance} with seed {seed}\n")
                            f.write(f"{e}\n")

            # 求解完成后，提取解信息
            scip_model = env.model.as_pyscipopt()
            solution_info = {
                'status': scip_model.getStatus(),
                'obj_val': None,
                'sol_vals': None,
                'primal_bound': scip_model.getPrimalbound(),
                'dual_bound': scip_model.getDualbound(),
                'solving_time': scip_model.getSolvingTime(),
                'n_nodes': scip_model.getNNodes(),
            }
            # 如果找到了可行解，记录变量值
            if scip_model.getNSols() > 0:
                best_sol = scip_model.getBestSol()
                all_vars = scip_model.getVars(transformed=True)
                solution_info['obj_val'] = scip_model.getSolObjVal(best_sol)
                solution_info['sol_vals'] = {
                    v.name: scip_model.getSolVal(best_sol, v) for v in all_vars
                }

            with gzip.open(out_path, 'wb') as f:
                pickle.dump({
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'observation': root_observation,
                    'action_set': np.array([]),
                    'solution': solution_info,
                }, f)

            out_queue.put({
                'type': 'sample',
                'episode': episode,
                'instance': instance,
                'seed': seed,
                'filename': out_path,
            })

        except Exception as e:
            with open("error_log.txt", "a") as f:
                f.write(f"Error in episode {episode}, instance {instance}, seed {seed}\n")
                f.write(traceback.format_exc())
                f.write("\n")

        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        print(f"[w {threading.current_thread().name}] episode {episode} done\n", end='')


def collect_samples(instances, out_dir, rng, n_jobs, time_limit):
    """
    For each instance, record the root-node observation once and the solver
    solution after running to completion.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_jobs : int
        Number of jobs for parallel sampling.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build tasks: instance_N.lp -> sample_N.pkl, skip existing
    tasks = []
    for inst in instances:
        num = extract_instance_number(inst)
        out_path = f'{out_dir}/sample_{num}.pkl'
        if os.path.exists(out_path):
            continue
        tasks.append((inst, out_path))

    if not tasks:
        print(f"  All {len(instances)} instances already processed in {out_dir}.")
        return

    n_skipped = len(instances) - len(tasks)
    if n_skipped > 0:
        print(f"  {n_skipped} existing samples skipped, {len(tasks)} to process.")

    orders_queue = queue.Queue(maxsize=2*n_jobs)
    answers_queue = queue.SimpleQueue()

    dispatcher_stop_flag = threading.Event()
    dispatcher = threading.Thread(
            target=send_orders,
            args=(orders_queue, tasks, rng.randint(2**32),
                  time_limit, dispatcher_stop_flag),
            daemon=True)
    dispatcher.start()

    workers = []
    workers_stop_flag = threading.Event()
    for i in range(n_jobs):
        p = threading.Thread(
                target=make_samples,
                args=(orders_queue, answers_queue, workers_stop_flag),
                daemon=True)
        workers.append(p)
        p.start()

    n_total = len(tasks)
    n_done = 0
    n_saved = 0

    while n_done < n_total:
        sample = answers_queue.get()

        if sample['type'] == 'sample':
            n_saved += 1
            print(f"[m] {n_saved}/{n_total} samples saved, "
                  f"ep {sample['episode']}.\n", end='')
        elif sample['type'] == 'done':
            n_done += 1

    workers_stop_flag.set()
    for p in workers:
        p.join(timeout=5)

    print(f"Done collecting samples for {out_dir} ({n_saved}/{n_total} saved)")


###############################################################################
# Gurobi-based data generation (faster alternative to SCIP strong branching)
#
# Uses ecole to extract root-node observation (bipartite graph features),
# then uses Gurobi to solve the MIP instance.
#
# Output format is identical to the SCIP-based generation above.
###############################################################################


def make_samples_gurobi(in_queue, out_queue, stop_flag, time_limit):
    """
    Worker loop using Gurobi for solving.

    Extracts root-node observation via ecole (fast), then solves with Gurobi.

    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which orders are received.
    out_queue : queue.Queue
        Output queue in which to send samples.
    stop_flag : threading.Event
        A flag to tell the thread to stop.
    time_limit : float
        Hard time limit (seconds). Stop and save current best solution.
    """
    import gurobipy as gp
    from gurobipy import GRB

    while not stop_flag.is_set():
        try:
            item = in_queue.get(timeout=1)
        except queue.Empty:
            continue
        episode, instance, seed, _time_limit, out_path = item

        print(f"[gurobi-w {threading.current_thread().name}] episode {episode}, "
              f"seed {seed}, processing '{instance}'...\n", end='')
        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        solution_info = {}

        try:
            # ── Step 1: Extract root-node observation via ecole ──
            scip_parameters = {
                'separating/maxrounds': 0,
                'presolving/maxrestarts': 0,
                'presolving/maxrounds': 0,
                'limits/time': time_limit,
                'timing/clocktype': 2,
            }
            observation_function = {
                "node_observation": ecole.observation.NodeBipartite()
            }
            env = ecole.environment.Branching(
                observation_function=observation_function,
                scip_params=scip_parameters,
                pseudo_candidates=True,
            )

            env.seed(seed)
            observation, action_set, _, done, _ = env.reset(instance)

            if observation is None:
                # 实例在根节点就被解完了，ecole 返回 None。
                # 用更严格的参数禁用预处理/切割/传播，重新加载以提取观测。
                fallback_model = ecole.scip.Model.from_file(instance)
                fallback_model.disable_presolve()
                fallback_model.disable_cuts()
                fallback_model.set_params({
                    'propagating/maxrounds': 0,
                    'propagating/maxroundsroot': 0,
                })
                fallback_env = ecole.environment.Branching(
                    observation_function=observation_function,
                    scip_params={},
                    pseudo_candidates=True,
                )
                fallback_env.seed(seed)
                observation, action_set, _, done, _ = fallback_env.reset(fallback_model)
                env = fallback_env

            node_observation = observation["node_observation"]

            scip_model = env.model.as_pyscipopt()
            variables = scip_model.getVars(transformed=True)
            lb_local = np.array(
                [v.getLbLocal() for v in variables], dtype=np.float32
            ).reshape(-1, 1)
            ub_local = np.array(
                [v.getUbLocal() for v in variables], dtype=np.float32
            ).reshape(-1, 1)
            lb_global = np.array(
                [v.getLbGlobal() for v in variables], dtype=np.float32
            ).reshape(-1, 1)
            ub_global = np.array(
                [v.getUbGlobal() for v in variables], dtype=np.float32
            ).reshape(-1, 1)
            variable_features = np.concatenate(
                [node_observation.variable_features,
                 lb_local, ub_local, lb_global, ub_global], axis=1
            )

            root_observation = (
                node_observation.row_features,
                (node_observation.edge_features.indices,
                 node_observation.edge_features.values),
                variable_features,
            )

            # SCIP transformed variable names (for mapping to Gurobi solution)
            scip_var_names = [v.name for v in variables]

            # ── Step 2: Solve with Gurobi ──
            grb_model = gp.read(instance)
            grb_model.setParam('OutputFlag', 0)
            grb_model.setParam('Seed', seed % (2**31))
            grb_model.setParam('TimeLimit', time_limit)
            grb_model.setParam('Threads', 1)

            grb_model.optimize()

            # ── Step 3: Extract solution (same format as SCIP) ──
            grb_status = grb_model.Status
            status_map = {
                GRB.OPTIMAL: 'optimal',
                GRB.INFEASIBLE: 'infeasible',
                GRB.INF_OR_UNBD: 'infeasible',
                GRB.UNBOUNDED: 'unbounded',
                GRB.TIME_LIMIT: 'timelimit',
                GRB.INTERRUPTED: 'userinterrupt',
            }

            solution_info = {
                'status': status_map.get(grb_status, str(grb_status)),
                'obj_val': None,
                'sol_vals': None,
                'primal_bound': None,
                'dual_bound': None,
                'solving_time': grb_model.Runtime,
                'n_nodes': int(grb_model.NodeCount),
            }

            if grb_model.SolCount > 0:
                solution_info['obj_val'] = grb_model.ObjVal
                solution_info['primal_bound'] = grb_model.ObjVal
                try:
                    solution_info['dual_bound'] = grb_model.ObjBound
                except Exception:
                    solution_info['dual_bound'] = grb_model.ObjVal

                # Map Gurobi solution to SCIP variable ordering
                grb_var_dict = {v.VarName: v.X for v in grb_model.getVars()}
                sol_vals = {}
                for scip_name in scip_var_names:
                    # SCIP transformed name: "t_<original>" -> Gurobi: "<original>"
                    if scip_name.startswith('t_'):
                        original_name = scip_name[2:]
                    else:
                        original_name = scip_name
                    sol_vals[scip_name] = grb_var_dict.get(original_name, 0.0)
                solution_info['sol_vals'] = sol_vals
            else:
                try:
                    solution_info['dual_bound'] = grb_model.ObjBound
                except Exception:
                    solution_info['dual_bound'] = None

            # ── Step 4: Save (identical format to SCIP version) ──
            with gzip.open(out_path, 'wb') as f:
                pickle.dump({
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'observation': root_observation,
                    'action_set': np.array([]),
                    'solution': solution_info,
                }, f)

            out_queue.put({
                'type': 'sample',
                'episode': episode,
                'instance': instance,
                'seed': seed,
                'filename': out_path,
            })

        except Exception as e:
            with open("error_log.txt", "a") as f:
                f.write(f"[gurobi] Error in episode {episode}, "
                        f"instance {instance}, seed {seed}\n")
                f.write(traceback.format_exc())
                f.write("\n")

        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        print(f"[gurobi-w {threading.current_thread().name}] episode {episode} "
              f"done ({solution_info.get('status', '?')}, "
              f"{solution_info.get('solving_time', 0):.1f}s)\n", end='')


def collect_samples_gurobi(instances, out_dir, rng, n_jobs, time_limit):
    """
    Gurobi-based sample collection.

    Output naming: instance_N.lp -> sample_N.pkl.
    Skips individual instances whose output already exists.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_jobs : int
        Number of jobs for parallel sampling.
    time_limit : float
        Hard time limit (seconds). Stop and save current best solution.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build tasks: instance_N.lp -> sample_N.pkl, skip existing
    tasks = []
    for inst in instances:
        num = extract_instance_number(inst)
        out_path = f'{out_dir}/sample_{num}.pkl'
        if os.path.exists(out_path):
            continue
        tasks.append((inst, out_path))

    if not tasks:
        print(f"  All {len(instances)} instances already processed in {out_dir}.")
        return

    n_skipped = len(instances) - len(tasks)
    if n_skipped > 0:
        print(f"  {n_skipped} existing samples skipped, {len(tasks)} to process.")

    orders_queue = queue.Queue(maxsize=2 * n_jobs)
    answers_queue = queue.SimpleQueue()

    # Dispatcher
    dispatcher_stop_flag = threading.Event()
    dispatcher = threading.Thread(
        target=send_orders,
        args=(orders_queue, tasks, rng.randint(2**32),
              time_limit, dispatcher_stop_flag),
        daemon=True)
    dispatcher.start()

    # Workers (using Gurobi)
    workers = []
    workers_stop_flag = threading.Event()
    for i in range(n_jobs):
        p = threading.Thread(
            target=make_samples_gurobi,
            args=(orders_queue, answers_queue, workers_stop_flag, time_limit),
            daemon=True)
        workers.append(p)
        p.start()

    n_total = len(tasks)
    n_done = 0
    n_saved = 0

    while n_done < n_total:
        sample = answers_queue.get()

        if sample['type'] == 'sample':
            n_saved += 1
            print(f"[m] {n_saved}/{n_total} samples saved, "
                  f"ep {sample['episode']}.\n", end='')
        elif sample['type'] == 'done':
            n_done += 1

    workers_stop_flag.set()
    for p in workers:
        p.join(timeout=5)

    print(f"Done collecting samples for {out_dir} ({n_saved}/{n_total} saved)")


def process_custom_datasets(dataset_dirs, args):
    """
    Process user-specified dataset directories sequentially.

    For each directory, finds all .lp files, processes them, and saves
    samples to <dir>_samples/ (or --out-dir if specified).

    Parameters
    ----------
    dataset_dirs : list of str
        Paths to directories containing .lp instance files.
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    collect_fn = collect_samples_gurobi if args.solver == 'gurobi' else collect_samples
    tl = args.time_limit if args.solver == 'gurobi' else 3600

    for idx, dataset_dir in enumerate(dataset_dirs):
        dataset_dir = dataset_dir.rstrip('/')
        if not os.path.isdir(dataset_dir):
            print(f"WARNING: '{dataset_dir}' is not a directory, skipping.")
            continue

        instances = sorted(
            glob.glob(os.path.join(dataset_dir, '*.lp'))
            + glob.glob(os.path.join(dataset_dir, '*.mps')),
            key=lambda f: extract_instance_number(f) or 0)
        if not instances:
            print(f"WARNING: no .lp/.mps files found in '{dataset_dir}', skipping.")
            continue

        if args.out_dir:
            if len(dataset_dirs) == 1:
                out_dir = args.out_dir
            else:
                out_dir = os.path.join(args.out_dir, os.path.basename(dataset_dir))
        else:
            out_dir = dataset_dir + '_samples'

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(dataset_dirs)}] Processing: {dataset_dir}")
        print(f"  instances: {len(instances)}")
        print(f"  output:    {out_dir}")
        print(f"  solver:    {args.solver}")
        print(f"  time_limit: {tl}s")
        print(f"{'='*60}")

        rng = np.random.RandomState(args.seed + idx)
        collect_fn(instances, out_dir, rng, args.njobs, time_limit=tl)

    print(f"\nAll {len(dataset_dirs)} dataset(s) processed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate bipartite graph samples from MILP instances. '
                    'Either pass dataset directories directly, or use -p to '
                    'select a built-in problem type.',
    )
    parser.add_argument(
        'datasets',
        nargs='*',
        help='Paths to directories containing .lp instance files. '
             'Each directory is processed sequentially. '
             'Output is saved to <dir>_samples/ by default (see --out-dir).',
    )
    parser.add_argument(
        '-p', '--problem',
        help='Built-in MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'mknapsack', 'SC'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--solver',
        help='Solver to use: scip (original, slow) or gurobi (fast).',
        choices=['scip', 'gurobi'],
        default='scip',
    )
    parser.add_argument(
        '--time-limit',
        help='Time limit in seconds per instance (default: 600).',
        type=float,
        default=600,
    )
    parser.add_argument(
        '-o', '--out-dir',
        help='Override output directory. When processing a single dataset, '
             'samples are saved here directly. With multiple datasets, a '
             'subdirectory per dataset is created under this path.',
        default=None,
    )
    args = parser.parse_args()

    # ── Custom dataset directories mode ──
    if args.datasets:
        print(f"seed {args.seed}")
        print(f"solver: {args.solver}")
        process_custom_datasets(args.datasets, args)

    # ── Built-in problem mode ──
    elif args.problem:
        print(f"seed {args.seed}")
        print(f"solver: {args.solver}")

        time_limit = 3600

        if args.problem == 'setcover':
            instances_train = glob.glob('data/instances/setcover/train_500r_1000c_0.05d/*.lp')
            instances_valid = glob.glob('data/instances/setcover/valid_500r_1000c_0.05d/*.lp')
            instances_test = glob.glob('data/instances/setcover/test_500r_1000c_0.05d/*.lp')
            out_dir = 'data/samples/setcover/500r_1000c_0.05d'
            transfer_configs = [
                ('data/instances/setcover/transfer_500r_1000c_0.05d',  'data/samples/setcover/500r_1000c_0.05d/transfer'),
                ('data/instances/setcover/transfer_1000r_1000c_0.05d', 'data/samples/setcover/1000r_1000c_0.05d/transfer'),
                ('data/instances/setcover/transfer_2000r_1000c_0.05d', 'data/samples/setcover/2000r_1000c_0.05d/transfer'),
            ]

        elif args.problem == 'cauctions':
            instances_train = glob.glob('data/instances/cauctions/train_100_500/*.lp')
            instances_valid = glob.glob('data/instances/cauctions/valid_100_500/*.lp')
            instances_test = glob.glob('data/instances/cauctions/test_100_500/*.lp')
            out_dir = 'data/samples/cauctions/100_500'

        elif args.problem == 'indset':
            instances_train = glob.glob('data/instances/indset/train_500_4/*.lp')
            instances_valid = glob.glob('data/instances/indset/valid_500_4/*.lp')
            instances_test = glob.glob('data/instances/indset/test_500_4/*.lp')
            out_dir = 'data/samples/indset/500_4'

        elif args.problem == 'facilities':
            instances_train = glob.glob('data/instances/facilities/train_100_100_5/*.lp')
            instances_valid = glob.glob('data/instances/facilities/valid_100_100_5/*.lp')
            instances_test = glob.glob('data/instances/facilities/test_100_100_5/*.lp')
            out_dir = 'data/samples/facilities/100_100_5'
            time_limit = 600

        elif args.problem == 'mknapsack':
            instances_train = glob.glob('data/instances/mknapsack/train_100_6/*.lp')
            instances_valid = glob.glob('data/instances/mknapsack/valid_100_6/*.lp')
            instances_test = glob.glob('data/instances/mknapsack/test_100_6/*.lp')
            out_dir = 'data/samples/mknapsack/100_6'
            time_limit = 60

        elif args.problem == 'SC':
            instances_train = sorted(glob.glob('data/l2o_milp/SC/*.lp'),
                                      key=lambda f: extract_instance_number(f))
            instances_valid = []
            instances_test = sorted(glob.glob('data/l2o_milp_test/SC/*.lp'),
                                    key=lambda f: extract_instance_number(f))
            out_dir = 'data/samples/SC'
            time_limit = 600

        else:
            raise NotImplementedError

        if 'transfer_configs' not in dir():
            transfer_configs = []

        print(f"{len(instances_train)} train instances")
        print(f"{len(instances_valid)} validation instances")
        print(f"{len(instances_test)} test instances")

        os.makedirs(out_dir, exist_ok=True)

        if args.solver == 'scip':
            rng = np.random.RandomState(args.seed)
            collect_samples(instances_train, out_dir + '/train', rng, args.njobs,
                            time_limit=time_limit)

            if instances_valid:
                rng = np.random.RandomState(args.seed + 1)
                collect_samples(instances_valid, out_dir + '/valid', rng, args.njobs,
                                time_limit=time_limit)

            if instances_test:
                rng = np.random.RandomState(args.seed + 2)
                collect_samples(instances_test, out_dir + '/test', rng, args.njobs,
                                time_limit=time_limit)

            for i, (inst_dir, transfer_out_dir) in enumerate(transfer_configs):
                instances_transfer = glob.glob(f'{inst_dir}/*.lp')
                if instances_transfer:
                    rng = np.random.RandomState(args.seed + 3 + i)
                    collect_samples(instances_transfer, transfer_out_dir, rng,
                                    args.njobs, time_limit=time_limit)

        elif args.solver == 'gurobi':
            print(f"Gurobi settings: time_limit={args.time_limit}s")

            rng = np.random.RandomState(args.seed)
            collect_samples_gurobi(instances_train, out_dir + '/train', rng,
                                   args.njobs, args.time_limit)

            if instances_valid:
                rng = np.random.RandomState(args.seed + 1)
                collect_samples_gurobi(instances_valid, out_dir + '/valid', rng,
                                       args.njobs, args.time_limit)

            if instances_test:
                rng = np.random.RandomState(args.seed + 2)
                collect_samples_gurobi(instances_test, out_dir + '/test', rng,
                                       args.njobs, args.time_limit)

            for i, (inst_dir, transfer_out_dir) in enumerate(transfer_configs):
                instances_transfer = glob.glob(f'{inst_dir}/*.lp')
                if instances_transfer:
                    rng = np.random.RandomState(args.seed + 3 + i)
                    collect_samples_gurobi(instances_transfer, transfer_out_dir, rng,
                                           args.njobs, args.time_limit)

    else:
        parser.error('Please provide dataset directories or use -p to select a built-in problem type.')
