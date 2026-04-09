"""
将 CoCo-MILP 处理后的数据（solution）与 ecole 提取的观测合并，
转换为 learn2branch-ecole 的 .pkl 格式。

跳过 Gurobi 求解（直接复用 CoCo-MILP 的解），只用 ecole 提取根节点观测（快速）。

用法示例:
    python convert_coco_to_l2b.py \
        --problem SC \
        --instance-dir data/l2o_milp/SC \
        --coco-sol-dir /home/lmh/CoCo-MILP/dataset/SC/solution \
        --out-dir data/coco_samples/SC/train \
        -j 4
"""

import os
import re
import glob
import gzip
import pickle
import queue
import argparse
import threading
import traceback
import numpy as np
import ecole


def extract_instance_number(filepath):
    """从文件名提取编号, 如 'instance_123.lp' -> 123"""
    match = re.search(r'(\d+)', os.path.basename(filepath))
    return int(match.group(1)) if match else None


def extract_observation(instance_path, seed):
    """
    用 ecole 提取根节点二部图观测 (快速, 无需求解)。
    返回 (root_observation, scip_var_names)。
    """
    scip_parameters = {
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
        'presolving/maxrounds': 0,
        'limits/time': 3600,
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
    observation, action_set, _, done, _ = env.reset(instance_path)

    if observation is None:
        fallback_model = ecole.scip.Model.from_file(instance_path)
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
    lb_local = np.array([v.getLbLocal() for v in variables], dtype=np.float32).reshape(-1, 1)
    ub_local = np.array([v.getUbLocal() for v in variables], dtype=np.float32).reshape(-1, 1)
    lb_global = np.array([v.getLbGlobal() for v in variables], dtype=np.float32).reshape(-1, 1)
    ub_global = np.array([v.getUbGlobal() for v in variables], dtype=np.float32).reshape(-1, 1)
    variable_features = np.concatenate(
        [node_observation.variable_features, lb_local, ub_local, lb_global, ub_global], axis=1
    )

    root_observation = (
        node_observation.row_features,
        (node_observation.edge_features.indices,
         node_observation.edge_features.values),
        variable_features,
    )

    scip_var_names = [v.name for v in variables]
    return root_observation, scip_var_names


def load_coco_solution(sol_path):
    """
    读取 CoCo-MILP 的 .sol 文件。
    返回 (var_names, best_sol_values, best_obj)。
    """
    with open(sol_path, 'rb') as f:
        data = pickle.load(f)

    var_names = data['var_names']
    sols = np.array(data['sols'])
    objs = np.array(data['objs'])

    # 取最优解 (第一个, objs 已按质量排序)
    best_sol = sols[0]
    best_obj = float(objs[0])

    # 构建 var_name -> value 映射
    sol_dict = {}
    for i, name in enumerate(var_names):
        sol_dict[name] = float(best_sol[i])

    return sol_dict, best_obj


def map_solution_to_scip_vars(coco_sol_dict, scip_var_names):
    """
    将 CoCo-MILP 的解 (原始变量名) 映射到 SCIP transformed 变量名。
    SCIP transformed 命名: 't_x1' -> 原始: 'x1'
    """
    sol_vals = {}
    for scip_name in scip_var_names:
        if scip_name.startswith('t_'):
            original_name = scip_name[2:]
        else:
            original_name = scip_name
        sol_vals[scip_name] = coco_sol_dict.get(original_name, 0.0)
    return sol_vals


def worker(in_queue, out_queue, stop_flag, coco_sol_dir, sol_prefix):
    """工作线程: 提取 ecole 观测 + 映射 CoCo-MILP 解 -> 保存 .pkl"""
    rng = np.random.RandomState(42)

    while not stop_flag.is_set():
        try:
            item = in_queue.get(timeout=1)
        except queue.Empty:
            continue

        episode, instance_path, seed, out_path = item
        inst_num = extract_instance_number(instance_path)

        print(f"[w {threading.current_thread().name}] "
              f"#{inst_num} extracting observation...\n", end='')

        try:
            # 1. ecole 提取观测
            root_observation, scip_var_names = extract_observation(instance_path, seed)

            # 2. 读取 CoCo-MILP 解
            sol_file = os.path.join(coco_sol_dir, f'{sol_prefix}{inst_num}.sol')
            coco_sol_dict, best_obj = load_coco_solution(sol_file)

            # 3. 映射变量名
            sol_vals = map_solution_to_scip_vars(coco_sol_dict, scip_var_names)

            # 4. 构造 solution_info (与 02_generate_dataset.py 格式一致)
            solution_info = {
                'status': 'timelimit',
                'obj_val': best_obj,
                'sol_vals': sol_vals,
                'primal_bound': best_obj,
                'dual_bound': None,
                'solving_time': 3600.0,
                'n_nodes': 0,
            }

            # 5. 保存
            with gzip.open(out_path, 'wb') as f:
                pickle.dump({
                    'episode': episode,
                    'instance': instance_path,
                    'seed': seed,
                    'observation': root_observation,
                    'action_set': np.array([]),
                    'solution': solution_info,
                }, f)

            out_queue.put({
                'type': 'sample',
                'episode': episode,
                'instance_num': inst_num,
            })

        except Exception as e:
            print(f"[w {threading.current_thread().name}] "
                  f"ERROR #{inst_num}: {e}\n", end='')
            with open("convert_error_log.txt", "a") as f:
                f.write(f"Error: instance_{inst_num}, {instance_path}\n")
                f.write(traceback.format_exc() + "\n")

        out_queue.put({'type': 'done', 'episode': episode})
        print(f"[w {threading.current_thread().name}] "
              f"#{inst_num} done\n", end='')


def main():
    parser = argparse.ArgumentParser(
        description='Convert CoCo-MILP solutions to learn2branch-ecole .pkl format.')
    parser.add_argument('--problem', type=str, default='SC',
                        help='Problem type (default: SC)')
    parser.add_argument('--instance-dir', type=str,
                        default='data/l2o_milp/SC',
                        help='Directory containing .lp instance files')
    parser.add_argument('--coco-sol-dir', type=str,
                        default='/home/lmh/CoCo-MILP/dataset/SC/solution',
                        help='CoCo-MILP solution directory')
    parser.add_argument('--sol-prefix', type=str, default=None,
                        help='解文件名前缀 (默认自动推断: 取 coco-sol-dir 中第一个 .sol 文件的数字前部分, '
                             '如 "item_placement_"; 也可手动指定如 "instance_")')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: data/coco_samples/<problem>/train)')
    parser.add_argument('-j', '--njobs', type=int, default=1,
                        help='Number of parallel threads')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed')
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = f'data/coco_samples/{args.problem}/train'

    # 自动推断解文件名前缀
    if args.sol_prefix is None:
        sol_files = glob.glob(os.path.join(args.coco_sol_dir, '*.sol'))
        if not sol_files:
            print(f"错误: {args.coco_sol_dir} 中没有找到任何 .sol 文件")
            return
        first_name = os.path.basename(sorted(sol_files)[0])
        # 取文件名中最后一段连续数字之前的部分作为前缀
        m = re.match(r'^(.*?)(\d+)\.sol$', first_name)
        if not m:
            print(f"错误: 无法从文件名 '{first_name}' 推断前缀, 请用 --sol-prefix 手动指定")
            return
        args.sol_prefix = m.group(1)
        print(f"自动推断解文件名前缀: '{args.sol_prefix}'")

    os.makedirs(args.out_dir, exist_ok=True)

    # 找到同时有 .lp 和 .sol 的实例
    instances = sorted(glob.glob(os.path.join(args.instance_dir, '*.lp')),
                       key=lambda f: extract_instance_number(f) or 0)

    tasks = []
    rng = np.random.RandomState(args.seed)
    for episode, inst_path in enumerate(instances):
        inst_num = extract_instance_number(inst_path)
        sol_file = os.path.join(args.coco_sol_dir, f'{args.sol_prefix}{inst_num}.sol')
        out_path = os.path.join(args.out_dir, f'sample_{inst_num}.pkl')

        if not os.path.exists(sol_file):
            print(f"跳过 instance_{inst_num}: 无 CoCo-MILP 解文件")
            continue
        if os.path.exists(out_path):
            continue

        seed = rng.randint(2**32)
        tasks.append((episode, inst_path, seed, out_path))

    if not tasks:
        print(f"所有实例已处理完毕, 输出目录: {args.out_dir}")
        return

    print(f"实例目录: {args.instance_dir}")
    print(f"CoCo-MILP 解目录: {args.coco_sol_dir}")
    print(f"输出目录: {args.out_dir}")
    print(f"总实例数: {len(instances)}, 待处理: {len(tasks)}, "
          f"已跳过: {len(instances) - len(tasks)}")
    print(f"线程数: {args.njobs}")

    in_queue = queue.Queue(maxsize=2 * args.njobs)
    out_queue = queue.SimpleQueue()
    stop_flag = threading.Event()

    # 启动 dispatcher
    def dispatch():
        for task in tasks:
            if stop_flag.is_set():
                break
            in_queue.put(task)

    dispatcher = threading.Thread(target=dispatch, daemon=True)
    dispatcher.start()

    # 启动 workers
    workers = []
    for _ in range(args.njobs):
        t = threading.Thread(target=worker,
                             args=(in_queue, out_queue, stop_flag, args.coco_sol_dir, args.sol_prefix),
                             daemon=True)
        t.start()
        workers.append(t)

    # 收集结果
    n_total = len(tasks)
    n_done = 0
    n_saved = 0

    while n_done < n_total:
        msg = out_queue.get()
        if msg['type'] == 'sample':
            n_saved += 1
            print(f"[main] {n_saved}/{n_total} saved\n", end='')
        elif msg['type'] == 'done':
            n_done += 1

    stop_flag.set()
    for t in workers:
        t.join(timeout=5)

    print(f"\n转换完成: {n_saved}/{n_total} 样本已保存到 {args.out_dir}")


if __name__ == '__main__':
    main()
